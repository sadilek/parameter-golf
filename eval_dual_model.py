"""Evaluate a mixture of two models by averaging their logits."""
import io, os, sys, math, glob, torch, lzma, time
import numpy as np
import sentencepiece as spm
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
# CRITICAL: set quant mode before importing model classes
import train_combined
train_combined._QUANT_MODE = "int6"
from train_combined import GPT, deq_sd_int6, build_luts, ld_shard, restore_low_dim_params_to_fp32

def load_model(path, device, num_layers=7, xsa_layers=2, sse=True):
    with open(path, "rb") as f:
        raw = f.read()
    try:
        dec = lzma.decompress(raw)
    except Exception:
        import zstandard
        dec = zstandard.ZstdDecompressor().decompress(raw)
    loaded = torch.load(io.BytesIO(dec), map_location="cpu", weights_only=False)
    sd = deq_sd_int6(loaded, target_dtype=torch.bfloat16)

    model = GPT(
        vocab_size=1024, num_layers=num_layers, model_dim=512,
        num_heads=8, num_kv_heads=4, mlp_mult=2,
        tie_embeddings=True, tied_embed_init_std=0.005,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
        activation="swiglu", n_channels=1, xsa_layers=xsa_layers,
        sse_enabled=sse,
    )
    model = model.to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.float()
    restore_low_dim_params_to_fp32(model)
    model.load_state_dict(sd, strict=False)
    return model.eval()


def eval_mixture(models, alphas, val_tokens, seq_len, device, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    total_seqs = (val_tokens.numel() - 1) // seq_len
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    batch_seqs = 32
    with torch.inference_mode():
        for batch_start in range(0, total_seqs, batch_seqs):
            batch_end = min(batch_start + batch_seqs, total_seqs)
            raw_start = batch_start * seq_len
            raw_end = batch_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)

            mixed_logits = None
            for model, alpha in zip(models, alphas):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Use the model's internal methods exactly as forward() does
                    emb = model._embed(x)
                    h = model._run_blocks(emb, emb, causal=True)
                    h_flat = h.reshape(-1, h.size(-1))
                    logits = model._softcap(model._compute_logits(h_flat))
                    if model.sse is not None:
                        logits = model.sse(logits)
                if mixed_logits is None:
                    mixed_logits = alpha * logits.float()
                else:
                    mixed_logits = mixed_logits + alpha * logits.float()

            targets = y.reshape(-1)
            lse = torch.logsumexp(mixed_logits, dim=-1)
            target_logits = mixed_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
            batch_loss = (lse - target_logits).to(torch.float64).sum()
            n = float(targets.numel())
            loss_sum += batch_loss
            token_count += n

            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            tok_bytes = base_bytes_lut[tgt_ids].to(torch.float64)
            tok_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
            byte_count += tok_bytes.sum()

    val_loss = (loss_sum / token_count).item()
    bpb = (val_loss / math.log(2.0)) * (token_count.item() / byte_count.item())
    return val_loss, bpb


def main():
    device = torch.device("cuda")
    sp = spm.SentencePieceProcessor("./data/tokenizers/fineweb_1024_bpe.model")
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_luts(sp, 1024, device)

    val_files = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
    val_tokens = torch.cat([ld_shard(Path(p)) for p in val_files]).contiguous()
    max_tok = min(500001, val_tokens.numel())
    val_tokens = val_tokens[:max_tok]
    print(f"Val tokens: {val_tokens.numel():,}")

    seq_len = 1024

    print("Loading model A...")
    model_a = load_model("model_A.ptz", device)
    print("Loading model B...")
    model_b = load_model("model_B.ptz", device)

    configs = [
        ("A only", [model_a], [1.0]),
        ("B only", [model_b], [1.0]),
        ("50/50", [model_a, model_b], [0.5, 0.5]),
        ("60/40", [model_a, model_b], [0.6, 0.4]),
        ("70/30", [model_a, model_b], [0.7, 0.3]),
    ]

    for name, models, alphas in configs:
        t0 = time.time()
        val_loss, bpb = eval_mixture(models, alphas, val_tokens, seq_len, device,
                                      base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
        elapsed = time.time() - t0
        print(f"  {name:>10}: val_loss={val_loss:.4f} val_bpb={bpb:.4f} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()

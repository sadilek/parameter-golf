"""Stochastic quantization ensemble at eval time.

Instead of deterministic rounding, randomly round each weight up or down
(probability proportional to distance). Run N forward passes with different
random roundings, average the logits. Free ensemble diversity — zero artifact cost.

Usage:
  python stochastic_eval.py final_model_float.pt --n-passes 1 5 10 --methods uniform lloyd_max_int4 lloyd_max_int3
"""
import os, sys, math, io, argparse, time
import torch
import torch.nn.functional as F
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("QUANT_MODE", "int6")
os.environ.setdefault("N_CHANNELS", "1")

import train_combined as tc
tc._QUANT_MODE = "int6"
from train_combined import (
    GPT, Hyperparameters, build_luts, ld_val, QuantizedLinear, LowRankLinear,
    restore_low_dim_params_to_fp32, INT6_RANGE, INT6_KEEP_FLOAT_PATTERNS,
    quantize_int6,
)
from requantize_eval import (
    compute_lloyd_max_lut, get_lloyd_max_lut,
    quantize_uniform_int6, quantize_lloyd_max,
    dequantize_uniform, dequantize_lloyd_max,
)
import torch.nn as nn


def stochastic_quantize_uniform(t2d, clip_q=0.9999984):
    """Stochastic uniform int6: randomly round up or down per weight."""
    t32 = t2d.float()
    clip_abs = torch.quantile(t32.abs(), clip_q, dim=1).clamp_min(1e-8)
    scale = (clip_abs / float(INT6_RANGE)).clamp_min(1.0 / float(INT6_RANGE))
    clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
    continuous = clipped / scale[:, None]
    # Stochastic rounding: floor + Bernoulli(frac)
    floor_q = continuous.floor()
    frac = continuous - floor_q
    stoch = floor_q + torch.bernoulli(frac)
    q = stoch.clamp(-INT6_RANGE, INT6_RANGE).to(torch.int8)
    return q, scale.half()


def stochastic_quantize_lloyd_max(t2d, n_levels=15):
    """Stochastic Lloyd-Max: randomly assign to one of two nearest centroids."""
    lut = torch.tensor(compute_lloyd_max_lut(n_levels), dtype=torch.float32,
                        device=t2d.device)
    half = n_levels // 2
    t32 = t2d.float()
    sigma = t32.std(dim=1).clamp_min(1e-8)
    normalized = t32 / sigma[:, None]

    # Find two nearest LUT entries for each element
    diffs = (normalized.unsqueeze(-1) - lut.unsqueeze(0).unsqueeze(0))  # (rows, cols, levels)
    abs_diffs = diffs.abs()

    # Nearest
    idx1 = abs_diffs.argmin(dim=-1)  # (rows, cols)
    # Zero out nearest to find second nearest
    abs_diffs_masked = abs_diffs.clone()
    abs_diffs_masked.scatter_(-1, idx1.unsqueeze(-1), float('inf'))
    idx2 = abs_diffs_masked.argmin(dim=-1)

    # Distances to the two nearest
    d1 = abs_diffs.gather(-1, idx1.unsqueeze(-1)).squeeze(-1)
    d2 = abs_diffs.gather(-1, idx2.unsqueeze(-1)).squeeze(-1)

    # Probability of selecting nearest (inversely proportional to distance)
    total = (d1 + d2).clamp_min(1e-10)
    p_nearest = d2 / total  # higher probability for closer centroid

    # Stochastic selection
    use_nearest = torch.bernoulli(p_nearest).bool()
    codes = torch.where(use_nearest, idx1, idx2)
    codes = (codes - half).to(torch.int8)

    return codes, sigma.half()


def stochastic_quantize_sd(sd, method, stochastic=True):
    """Stochastically quantize a state dict."""
    result = {}
    for name, tensor in sd.items():
        if "mtp_heads" in name:
            continue
        t = tensor.detach().cpu().float().contiguous()
        is_keep_float = any(p in name for p in INT6_KEEP_FLOAT_PATTERNS)

        if t.ndim >= 2 and t.numel() > 4096 and not is_keep_float:
            t2d = t.reshape(t.shape[0], -1) if t.ndim > 2 else t
            if method == "uniform" or method == "uniform_int6":
                if stochastic:
                    q, s = stochastic_quantize_uniform(t2d)
                else:
                    q, s = quantize_uniform_int6(t2d)
                result[name] = (q.float() * s.float()[:, None]).reshape(t.shape)
            elif method.startswith("lloyd_max"):
                nl = {"lloyd_max": 63, "lloyd_max_int5": 31, "lloyd_max_int4": 15,
                      "lloyd_max_int3": 7}.get(method, 15)
                if stochastic:
                    q, s = stochastic_quantize_lloyd_max(t2d, nl)
                else:
                    q, s = quantize_lloyd_max(t2d, nl)
                lut = torch.tensor(compute_lloyd_max_lut(nl), dtype=torch.float32)
                half = nl // 2
                indices = (q.long() + half).clamp(0, nl - 1)
                recon = lut[indices] * s.float()[:, None]
                result[name] = recon.reshape(t.shape)
        elif t.ndim >= 2 and t.numel() > 4096 and is_keep_float:
            # Int8 for embeddings — always deterministic
            t2d = t.reshape(t.shape[0], -1) if t.ndim > 2 else t
            q, s = quantize_int6(t2d, 127)
            result[name] = (q.float() * s.float()[:, None]).reshape(t.shape)
        else:
            result[name] = t
    return result


def eval_bpb_stochastic(model, n_passes, sd, method, args, device, val_tokens, bl, hl, il):
    """Run N forward passes with stochastic quantization, average logits, compute BPB.

    Pre-computes N quantized state dicts, then cycles through them during eval.
    """
    seq_len = args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    batch_seqs = max(1, args.val_batch_size // (seq_len * 8))
    model_sd = model.state_dict()

    # Pre-compute N quantized state dicts (on CPU, then move to GPU once)
    quantized_sds = []
    for i in range(n_passes):
        recon = stochastic_quantize_sd(sd, method, stochastic=(n_passes > 1))
        loaded = {k: v.to(device=device, dtype=model_sd[k].dtype)
                  for k, v in recon.items() if k in model_sd}
        quantized_sds.append(loaded)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    with torch.inference_mode():
        for bs in range(0, total_seqs, batch_seqs):
            be = min(bs + batch_seqs, total_seqs)
            local = val_tokens[bs * seq_len:(be * seq_len) + 1].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)

            avg_logits = None
            for qsd in quantized_sds:
                model.load_state_dict(qsd, strict=False)
                model.eval()

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    emb = model._embed(x)
                    h = model._run_blocks(emb, emb, causal=True)
                    logits = model._softcap(model._compute_logits(
                        h.reshape(-1, h.size(-1))))
                    if model.sse:
                        logits = model.sse(logits)

                if avg_logits is None:
                    avg_logits = logits.float()
                else:
                    avg_logits = avg_logits + logits.float()

            avg_logits = avg_logits / n_passes
            targets = y.reshape(-1)
            lse = torch.logsumexp(avg_logits, dim=-1)
            tl = avg_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
            loss_sum += (lse - tl).to(torch.float64).sum()
            token_count += float(targets.numel())
            px, tx = x.reshape(-1), y.reshape(-1)
            tb = bl[tx].to(torch.float64) + (hl[tx] & ~il[px]).to(torch.float64)
            byte_count += tb.sum()

    vl = (loss_sum / token_count).item()
    return (vl / math.log(2.0)) * (token_count.item() / byte_count.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", help="Path to float weights (.pt)")
    parser.add_argument("--n-passes", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--methods", nargs="+", default=["uniform", "lloyd_max_int4", "lloyd_max_int3"])
    args_cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Hyperparameters()
    tc._LOW_RANK = args.low_rank
    tc._LORA_RANK = 0

    sd = torch.load(args_cli.weights, map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in sd.items()}
    print(f"Loaded {len(sd)} tensors")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    bl, hl, il = build_luts(sp, args.vocab_size, device)
    val_tokens = ld_val(args.val_files, args.train_seq_len)

    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        group_size=args.bitnet_group_size, activation=args.activation_type,
        n_channels=1, xsa_layers=args.xsa_layers,
        sse_enabled=args.sse_enabled, sse_clusters=args.sse_clusters,
        sse_entropy_bins=args.sse_entropy_bins,
    ).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, (nn.Linear, QuantizedLinear, LowRankLinear)):
            m.float()
    restore_low_dim_params_to_fp32(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    print(f"\n{'='*60}")
    print("Stochastic Quantization Ensemble")
    print(f"{'='*60}")

    results = {}
    for method in args_cli.methods:
        for n in args_cli.n_passes:
            label = f"{method}/N={n}"
            t0 = time.time()
            bpb = eval_bpb_stochastic(model, n, sd, method, args, device,
                                       val_tokens, bl, hl, il)
            elapsed = time.time() - t0
            results[label] = bpb
            print(f"  {label:30s}  BPB={bpb:.4f}  ({elapsed:.0f}s)")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for method in args_cli.methods:
        base = results.get(f"{method}/N=1", 0)
        print(f"\n  {method}:")
        for n in args_cli.n_passes:
            label = f"{method}/N={n}"
            bpb = results[label]
            delta = bpb - base if n > 1 else 0
            print(f"    N={n:2d}: {bpb:.4f} {f'({delta:+.4f} vs N=1)' if n > 1 else ''}")


if __name__ == "__main__":
    main()

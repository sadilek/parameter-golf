"""Standalone eval: load SOTA GPTQ artifact, run standard + sliding eval.

Usage:
    python eval_sota.py [artifact_path]                 # default: final_model.int6.ptz
    NO_COMPILE_EVAL=1 python eval_sota.py artifact      # disable torch.compile

Uses torch.compile by default (the OOM in the SOTA script was caused by having
both training and eval models in memory; here we only have one model).

Runs on a single GPU without torchrun/distributed.
"""
from __future__ import annotations
import glob, io, lzma, math, os, sys, time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor

# --- SOTA-specific defaults (must be set before Hyperparameters is loaded) ---
os.environ.setdefault("BIGRAM_VOCAB_SIZE", "3072")
os.environ.setdefault("BIGRAM_DIM", "112")
os.environ.setdefault("MAX_WALLCLOCK_SECONDS", "0")

# --- Import model definitions from SOTA script ---
# We exec everything before the if __name__ guard to get classes and helpers.
_sota_path = os.environ.get("SOTA_SCRIPT", "train_gpt_sota.py")
with open(_sota_path) as _f:
    _source = _f.read()
# Strip the if __name__ block so main() is defined but never called
_defs = _source.split('\nif __name__')[0]
_defs = _defs.replace("__file__", repr(_sota_path))
exec(_defs, globals())
# Now available: Hyperparameters, GPT, CastedLinear, build_sentencepiece_luts,
# load_validation_tokens, load_data_shard, eval_val, eval_val_sliding,
# _unbank_state_dict, _rebank_state_dict, dequantize_mixed_int6,
# restore_low_dim_params_to_fp32, CONTROL_TENSOR_NAME_PATTERNS, etc.


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Hyperparameters()

    # Load tokenizer and validation data
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    print(f"Val tokens: {val_tokens.numel():,}")

    # Build a fresh model to get the template state dict
    print("Building model...")
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
    ).to(device).bfloat16()

    # Load and dequantize GPTQ artifact
    artifact_path = sys.argv[1] if len(sys.argv) > 1 else "final_model.int6.ptz"
    print(f"Loading artifact: {artifact_path}")
    with open(artifact_path, "rb") as f:
        quant_blob = f.read()
    print(f"  Compressed size: {len(quant_blob):,} bytes")
    quant_state = torch.load(
        io.BytesIO(lzma.decompress(quant_blob)), map_location="cpu", weights_only=False
    )

    # Pipeline: unbank template → dequantize against unbanked template → rebank
    sd_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)
    deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)
    deq_state = _rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)

    # Convert banks to float32 for loading (matches EVAL_ONLY path in SOTA script)
    model.qo_bank.data = model.qo_bank.data.float()
    model.kv_bank.data = model.kv_bank.data.float()
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model)
    model.load_state_dict(deq_state, strict=True)
    model.eval()
    print(f"  Loaded {len(deq_state)} keys (strict=True)")

    # Standard eval
    print("\n=== Standard eval ===")
    t0 = time.perf_counter()
    val_loss, val_bpb = eval_val(
        args, model, rank=0, world_size=1, device=device, grad_accum_steps=1,
        val_tokens=val_tokens, base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
        eval_seq_len=args.train_seq_len,
    )
    print(f"  val_loss={val_loss:.6f}  val_bpb={val_bpb:.6f}  ({time.perf_counter()-t0:.1f}s)")

    # Sliding window eval
    stride = args.eval_stride
    if stride > 0 and stride < args.train_seq_len:
        torch.cuda.empty_cache()
        batch_seqs = int(os.environ.get("BATCH_SEQS", "8"))
        print(f"\n=== Sliding window eval (stride={stride}, batch_seqs={batch_seqs}) ===")
        t0 = time.perf_counter()
        sw_loss, sw_bpb = eval_val_sliding(
            args, model, rank=0, world_size=1, device=device,
            val_tokens=val_tokens, base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            stride=stride, batch_seqs=batch_seqs,
            eval_seq_len=args.train_seq_len,
        )
        print(f"  val_loss={sw_loss:.6f}  val_bpb={sw_bpb:.6f}  ({time.perf_counter()-t0:.1f}s)")
        print(f"\n  Delta from sliding: {sw_bpb - val_bpb:+.6f} BPB")
    else:
        print(f"\nSkipping sliding eval (EVAL_STRIDE={stride})")

    print("\nDone.")


if __name__ == "__main__":
    main()

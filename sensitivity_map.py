"""Per-matrix quantization sensitivity: quantize one matrix at a time, measure BPB delta.

Produces a sensitivity map showing which matrices need precision and which don't.
Then computes optimal bit allocation for a given artifact budget.

Usage:
  python sensitivity_map.py final_model_float.pt
"""
import os, sys, math, io, argparse, time, copy
import torch
import torch.nn.functional as F
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("QUANT_MODE", "int6")
os.environ.setdefault("N_CHANNELS", "1")

import train_combined as tc
tc._QUANT_MODE = "int6"
from train_combined import (
    GPT, Hyperparameters, build_luts, ld_val, eval_val,
    QuantizedLinear, LowRankLinear, restore_low_dim_params_to_fp32,
    INT6_KEEP_FLOAT_PATTERNS, quantize_int6, INT6_RANGE,
)
from requantize_eval import compute_lloyd_max_lut
import torch.nn as nn


def quantize_single_matrix(weight, method="int6", K=None):
    """Quantize a single weight matrix. Returns reconstructed tensor."""
    w = weight.float()
    if w.ndim < 2 or w.numel() <= 4096:
        return w

    t2d = w.reshape(w.shape[0], -1)

    if method == "int6":
        q, s = quantize_int6(t2d, INT6_RANGE)
        return (q.float() * s.float()[:, None]).reshape(w.shape)
    elif method == "int3":
        q, s = quantize_int6(t2d, 3)  # [-3, 3] = 7 levels
        return (q.float() * s.float()[:, None]).reshape(w.shape)
    elif method == "palette":
        # k-means palette
        flat = t2d.flatten().cuda()
        N = flat.numel()
        sorted_vals, _ = flat.sort()
        idx = torch.linspace(0, N - 1, K).long()
        centroids = sorted_vals[idx].clone()
        for _ in range(30):
            dists = (flat.unsqueeze(1) - centroids.unsqueeze(0)).abs()
            assignments = dists.argmin(dim=1)
            for j in range(K):
                mask = assignments == j
                if mask.any():
                    centroids[j] = flat[mask].mean()
        recon = centroids[assignments].cpu()
        return recon.reshape(w.shape)
    elif method == "zero":
        return torch.zeros_like(w)

    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", help="Path to float weights (.pt)")
    parser.add_argument("--method", default="int3",
                        help="Quantization to test sensitivity against (int6, int3, palette, zero)")
    parser.add_argument("--palette-k", type=int, default=8, help="K for palette method")
    args_cli = parser.parse_args()

    device = torch.device("cuda")
    args = Hyperparameters()
    tc._LOW_RANK = args.low_rank
    tc._LORA_RANK = 0

    sd = torch.load(args_cli.weights, map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in sd.items()}
    print(f"Loaded {len(sd)} tensors")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    bl, hl, il = build_luts(sp, args.vocab_size, device)
    val_tokens = ld_val(args.val_files, args.train_seq_len)

    # Build model
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init, group_size=args.bitnet_group_size,
        activation=args.activation_type, n_channels=1,
        xsa_layers=args.xsa_layers, sse_enabled=args.sse_enabled,
        sse_clusters=args.sse_clusters, sse_entropy_bins=args.sse_entropy_bins,
    ).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, (nn.Linear, QuantizedLinear, LowRankLinear)):
            m.float()
    restore_low_dim_params_to_fp32(model)

    model_sd = model.state_dict()

    def load_sd(state_dict):
        loaded = {k: v.to(device=device, dtype=model_sd[k].dtype)
                  for k, v in state_dict.items() if k in model_sd}
        model.load_state_dict(loaded, strict=False)

    # Baseline: all float weights
    load_sd(sd)
    compiled = torch.compile(model)
    torch._dynamo.reset()
    _, base_bpb = eval_val(args, compiled, 0, 1, device, 8, val_tokens, bl, hl, il)
    print(f"\nBaseline (all float): {base_bpb:.4f} BPB")

    # Quantize ALL matrices at once
    all_quant_sd = {}
    for k, v in sd.items():
        is_keep = any(p in k for p in INT6_KEEP_FLOAT_PATTERNS)
        if v.ndim >= 2 and v.numel() > 4096 and not is_keep:
            all_quant_sd[k] = quantize_single_matrix(
                v, args_cli.method, args_cli.palette_k)
        else:
            all_quant_sd[k] = v

    load_sd(all_quant_sd)
    torch._dynamo.reset()
    compiled = torch.compile(model)
    _, all_quant_bpb = eval_val(args, compiled, 0, 1, device, 8, val_tokens, bl, hl, il)
    total_gap = all_quant_bpb - base_bpb
    print(f"All quantized ({args_cli.method}): {all_quant_bpb:.4f} BPB (gap: {total_gap:+.4f})")

    # Per-matrix sensitivity: quantize one at a time, measure BPB delta
    print(f"\n{'='*70}")
    print(f"Per-matrix sensitivity ({args_cli.method})")
    print(f"{'='*70}")
    print(f"{'matrix':50s} {'params':>8s} {'BPB':>8s} {'delta':>8s} {'% of gap':>8s}")

    sensitivities = {}
    quantizable = [(k, v) for k, v in sd.items()
                   if v.ndim >= 2 and v.numel() > 4096
                   and not any(p in k for p in INT6_KEEP_FLOAT_PATTERNS)]

    for name, weight in quantizable:
        # Start from all-float, quantize just this one matrix
        test_sd = dict(sd)
        test_sd[name] = quantize_single_matrix(weight, args_cli.method, args_cli.palette_k)

        load_sd(test_sd)
        torch._dynamo.reset()
        compiled = torch.compile(model)
        _, bpb = eval_val(args, compiled, 0, 1, device, 8, val_tokens, bl, hl, il)
        delta = bpb - base_bpb
        pct = (delta / total_gap * 100) if total_gap > 0 else 0
        sensitivities[name] = delta
        print(f"{name:50s} {weight.numel():8d} {bpb:8.4f} {delta:+8.4f} {pct:7.1f}%")

    # Summary: sorted by sensitivity
    print(f"\n{'='*70}")
    print("Ranked by sensitivity (most sensitive first)")
    print(f"{'='*70}")
    ranked = sorted(sensitivities.items(), key=lambda x: -x[1])
    cumulative = 0
    for name, delta in ranked:
        cumulative += delta
        params = sd[name].numel()
        print(f"  {delta:+.4f} BPB  {params/1e6:.2f}M params  {name}")

    # Suggest bit allocation
    print(f"\n{'='*70}")
    print("Suggested bit allocation for 16MB budget")
    print(f"{'='*70}")
    total_params = sum(sd[k].numel() for k, _ in ranked)

    # Simple greedy: give more bits to sensitive matrices
    # High sensitivity → 5 bits (K=32), Medium → 4 bits (K=16), Low → 3 bits (K=8)
    sorted_names = [name for name, _ in ranked]
    third = len(sorted_names) // 3
    for i, name in enumerate(sorted_names):
        params = sd[name].numel()
        if i < third:
            bits, label = 5, "HIGH"
        elif i < 2 * third:
            bits, label = 4, "MED"
        else:
            bits, label = 3, "LOW"
        print(f"  {bits}b ({label:4s})  {sensitivities[name]:+.4f}  {params/1e6:.2f}M  {name}")


if __name__ == "__main__":
    main()

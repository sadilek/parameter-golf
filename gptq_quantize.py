"""GPTQ: post-training quantization with Hessian-based error compensation.

Implements the core GPTQ algorithm:
1. Collect Hessian (H = X^T X) per linear layer from calibration data
2. For each column: quantize, compute error, compensate remaining columns
3. Supports mixed precision: different bit-widths per layer/matrix

Usage:
  python gptq_quantize.py final_model_float.pt --scheme uniform_int6
  python gptq_quantize.py final_model_float.pt --scheme mixed_6_4_3
"""
import os, sys, math, io, argparse, time, copy
import torch
import torch.nn as nn
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


# ---------------------------------------------------------------------------
# Hessian collection
# ---------------------------------------------------------------------------
class HessianCollector:
    """Collects H = X^T X for a linear layer's inputs during forward passes."""

    def __init__(self):
        self.H = None
        self.n_samples = 0
        self.hook_handle = None

    def hook_fn(self, module, input, output):
        x = input[0]
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        x = x.float()
        if self.H is None:
            self.H = torch.zeros(x.shape[1], x.shape[1], device=x.device)
        self.H.addmm_(x.T, x)
        self.n_samples += x.shape[0]

    def register(self, module):
        self.hook_handle = module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.hook_handle:
            self.hook_handle.remove()

    def get_H(self):
        if self.n_samples == 0:
            return self.H
        return self.H / self.n_samples


def collect_hessians(model, calib_tokens, seq_len, device, n_calib_tokens=131072):
    """Run calibration data through model, collect Hessian per linear layer."""
    collectors = {}

    # Register hooks on all quantizable linear layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, QuantizedLinear)) and not any(
            p in name for p in ("tok_emb", "lm_head", "sse", "bigram")
        ):
            # Check if it has enough params to quantize
            if hasattr(module, 'weight') and module.weight.numel() > 4096:
                c = HessianCollector()
                c.register(module)
                collectors[name] = c

    print(f"  Registered {len(collectors)} Hessian collectors")

    # Run calibration
    model.eval()
    total_tokens = 0
    n_seqs = min(n_calib_tokens // seq_len, (calib_tokens.numel() - 1) // seq_len)
    batch_size = 32

    with torch.inference_mode():
        for i in range(0, n_seqs, batch_size):
            j = min(i + batch_size, n_seqs)
            chunk = calib_tokens[i * seq_len:(j * seq_len) + 1].to(device=device, dtype=torch.int64)
            x = chunk[:-1].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model._embed(x)
                emb = model._embed(x)
                h = model._run_blocks(emb, emb, causal=True)
            total_tokens += x.numel()
            if total_tokens >= n_calib_tokens:
                break

    # Extract Hessians and remove hooks
    hessians = {}
    for name, c in collectors.items():
        if c.H is not None:
            hessians[name] = c.get_H()
        c.remove()

    print(f"  Collected Hessians for {len(hessians)} layers ({total_tokens:,} calib tokens)")
    return hessians


# ---------------------------------------------------------------------------
# GPTQ core algorithm
# ---------------------------------------------------------------------------
def gptq_quantize_matrix(W, H, quant_range=31, blocksize=128, percdamp=0.01):
    """GPTQ quantization of a single weight matrix.

    W: (out_features, in_features) — weight matrix
    H: (in_features, in_features) — Hessian
    quant_range: half-range (31=int6, 7=int4, 3=int3)
    blocksize: columns to process together
    percdamp: damping factor for Hessian

    Returns (quantized_int8, scales_fp16).
    """
    dev = H.device
    W = W.float().clone().to(dev)
    rows, cols = W.shape

    # Damping
    damp = percdamp * H.diag().mean()
    diag_idx = torch.arange(cols, device=H.device)
    H = H.clone()
    H[diag_idx, diag_idx] += damp

    # Cholesky of H_inv
    try:
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    except Exception:
        H_inv = torch.linalg.pinv(H)
    try:
        Hinv_cho = torch.linalg.cholesky(H_inv, upper=True)
    except Exception:
        Hinv_cho = torch.linalg.cholesky(H_inv + 1e-6 * torch.eye(cols, device=H.device), upper=True)

    # Per-row scale for quantization
    scale = W.abs().amax(dim=1).clamp_min(1e-8) / quant_range

    Q = torch.zeros_like(W, dtype=torch.int8)
    Losses = torch.zeros(rows, device=dev)

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        bsize = i2 - i1

        W_block = W[:, i1:i2].clone()
        Hinv_block = Hinv_cho[i1:i2, i1:i2]

        for i in range(bsize):
            col_idx = i1 + i
            w = W_block[:, i]
            d = Hinv_block[i, i]

            # Quantize
            q = (w / scale).round().clamp(-quant_range, quant_range)
            Q[:, col_idx] = q.to(torch.int8)

            # Error
            err = (w - q * scale) / d

            # Compensate within block
            W_block[:, i:] -= err.unsqueeze(1) * Hinv_block[i, i:].unsqueeze(0)

        # Compensate remaining columns outside block
        if i2 < cols:
            W[:, i2:] -= (W[:, i1:i2] - Q[:, i1:i2].float() * scale.unsqueeze(1)) @ Hinv_cho[i1:i2, i2:]

    return Q.cpu(), scale.cpu().half()


# ---------------------------------------------------------------------------
# Full GPTQ quantization with mixed precision
# ---------------------------------------------------------------------------
def get_bit_scheme(scheme_name, n_layers):
    """Return dict mapping block_idx to quant_range per matrix type."""
    if scheme_name == "uniform_int6":
        return {i: {"attn": 31, "mlp": 31} for i in range(n_layers)}
    elif scheme_name == "uniform_int4":
        return {i: {"attn": 7, "mlp": 7} for i in range(n_layers)}
    elif scheme_name == "uniform_int3":
        return {i: {"attn": 3, "mlp": 3} for i in range(n_layers)}
    elif scheme_name == "mixed_6_6_4_4_5":
        # User's step 1: non-aggressive. First 2=int6, middle=int4, last=int5
        scheme = {}
        for i in range(n_layers):
            if i < 2:
                scheme[i] = {"attn": 31, "mlp": 31}  # int6
            elif i == n_layers - 1:
                scheme[i] = {"attn": 15, "mlp": 15}  # int5
            else:
                scheme[i] = {"attn": 7, "mlp": 7}    # int4
        return scheme
    elif scheme_name == "mixed_6_5_3_3_4":
        # User's step 2: aggressive. block0=int6, block1=int5, middle=int3, last=int4
        scheme = {}
        for i in range(n_layers):
            if i == 0:
                scheme[i] = {"attn": 31, "mlp": 31}  # int6
            elif i == 1:
                scheme[i] = {"attn": 15, "mlp": 15}  # int5
            elif i == n_layers - 1:
                scheme[i] = {"attn": 7, "mlp": 7}    # int4
            else:
                scheme[i] = {"attn": 3, "mlp": 3}    # int3
        return scheme
    elif scheme_name == "mixed_sensitivity":
        # Based on sensitivity map: block 0 int6, block 1 int5, blocks 2-3 int4, rest int3, last 2 int4
        scheme = {}
        for i in range(n_layers):
            if i == 0:
                scheme[i] = {"attn": 31, "mlp": 31}
            elif i == 1:
                scheme[i] = {"attn": 15, "mlp": 15}
            elif i < 4 or i >= n_layers - 2:
                scheme[i] = {"attn": 7, "mlp": 7}
            else:
                scheme[i] = {"attn": 3, "mlp": 3}
        return scheme
    else:
        raise ValueError(f"Unknown scheme: {scheme_name}")


def gptq_quantize_model(model, sd, hessians, scheme, device):
    """Apply GPTQ to all quantizable matrices with per-layer bit allocation."""
    model_sd = model.state_dict()
    result_sd = {}
    total_raw_bits = 0
    stats = {"gptq_params": 0, "int8_params": 0, "fp_params": 0}

    for name, tensor in sd.items():
        t = tensor.float()
        is_keep = any(p in name for p in INT6_KEEP_FLOAT_PATTERNS)

        if t.ndim >= 2 and t.numel() > 4096 and not is_keep:
            t2d = t.reshape(t.shape[0], -1) if t.ndim > 2 else t

            # Find block index and matrix type
            block_idx = -1
            mat_type = "mlp"
            for i in range(100):
                if f"blocks.{i}." in name:
                    block_idx = i
                    break
            if "attn" in name:
                mat_type = "attn"

            # Get quant range from scheme
            if block_idx >= 0 and block_idx in scheme:
                qrange = scheme[block_idx][mat_type]
            else:
                qrange = 3  # default int3

            # Find matching Hessian
            # Map state dict name to module name
            module_name = name.replace(".weight", "")
            H = hessians.get(module_name)

            if H is not None and H.shape[0] == t2d.shape[1]:
                # Full GPTQ
                q, s = gptq_quantize_matrix(t2d, H, quant_range=qrange)
                result_sd[name] = (q.float() * s.float()[:, None]).reshape(t.shape)
            else:
                # Fallback to uniform quantization (no Hessian available)
                q, s = quantize_int6(t2d, qrange)
                result_sd[name] = (q.float() * s.float()[:, None]).reshape(t.shape)

            bits = math.ceil(math.log2(2 * qrange + 1))
            total_raw_bits += t.numel() * bits
            stats["gptq_params"] += t.numel()

        elif t.ndim >= 2 and t.numel() > 4096 and is_keep:
            t2d = t.reshape(t.shape[0], -1) if t.ndim > 2 else t
            q, s = quantize_int6(t2d, 127)
            result_sd[name] = (q.float() * s.float()[:, None]).reshape(t.shape)
            total_raw_bits += t.numel() * 8
            stats["int8_params"] += t.numel()
        else:
            result_sd[name] = t
            total_raw_bits += t.numel() * 16
            stats["fp_params"] += t.numel()

    return result_sd, total_raw_bits, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", help="Path to float weights (.pt)")
    parser.add_argument("--schemes", nargs="+",
                        default=["uniform_int6", "mixed_6_6_4_4_3", "mixed_6_5_3_3_4", "mixed_sensitivity"])
    parser.add_argument("--calib-tokens", type=int, default=131072)
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
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        group_size=args.bitnet_group_size, activation=args.activation_type,
        n_channels=1, xsa_layers=args.xsa_layers, sse_enabled=args.sse_enabled,
        sse_clusters=args.sse_clusters, sse_entropy_bins=args.sse_entropy_bins,
    ).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, (nn.Linear, QuantizedLinear, LowRankLinear)):
            m.float()
    restore_low_dim_params_to_fp32(model)

    model_sd = model.state_dict()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, {args.num_layers}L/{args.model_dim}d")

    # Load float weights
    loaded = {k: v.to(device=device, dtype=model_sd[k].dtype) for k, v in sd.items() if k in model_sd}
    model.load_state_dict(loaded, strict=False)

    # Baseline eval
    torch._dynamo.reset()
    compiled = torch.compile(model)
    _, base_bpb = eval_val(args, compiled, 0, 1, device, 8, val_tokens, bl, hl, il)
    print(f"\nBaseline (float): {base_bpb:.4f} BPB")

    # Collect Hessians
    print(f"\nCollecting Hessians ({args_cli.calib_tokens:,} calib tokens)...")
    # Reload float weights (eval might have changed state)
    model.load_state_dict(loaded, strict=False)
    hessians = collect_hessians(model, val_tokens, args.train_seq_len, device, args_cli.calib_tokens)

    # Test each scheme
    print(f"\n{'='*70}")
    print("GPTQ Quantization Comparison")
    print(f"{'='*70}")

    for scheme_name in args_cli.schemes:
        t0 = time.time()
        scheme = get_bit_scheme(scheme_name, args.num_layers)

        # Reload float weights before each GPTQ run
        model.load_state_dict(loaded, strict=False)

        # GPTQ quantize
        recon_sd, raw_bits, stats = gptq_quantize_model(model, sd, hessians, scheme, device)

        # Load quantized weights and eval
        recon_loaded = {k: v.to(device=device, dtype=model_sd[k].dtype)
                       for k, v in recon_sd.items() if k in model_sd}
        model.load_state_dict(recon_loaded, strict=False)
        torch._dynamo.reset()
        compiled = torch.compile(model)
        _, bpb = eval_val(args, compiled, 0, 1, device, 8, val_tokens, bl, hl, il)

        gap = bpb - base_bpb
        raw_mb = raw_bits / 8 / 1e6
        elapsed = time.time() - t0

        # Bit allocation summary
        bit_summary = []
        for i in sorted(scheme.keys()):
            a = math.ceil(math.log2(2 * scheme[i]["attn"] + 1))
            m = math.ceil(math.log2(2 * scheme[i]["mlp"] + 1))
            bit_summary.append(f"{a}")
        bits_str = ",".join(bit_summary[:6]) + ("..." if len(bit_summary) > 6 else "")

        print(f"\n  {scheme_name:25s}  BPB={bpb:.4f}  gap={gap:+.4f}  raw~{raw_mb:.1f}MB  "
              f"bits=[{bits_str}]  ({elapsed:.0f}s)")

    print(f"\n{'='*70}")
    print("Done!")

    # Stop reminder
    print("\nREMEMBER: Stop the pod!")


if __name__ == "__main__":
    main()

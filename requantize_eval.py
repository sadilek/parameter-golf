"""Offline quantization comparison: load float weights, try different quantizers, eval BPB.

Decouples quantization research from training. Train once, re-quantize many times.

Usage:
  # Save float weights during training:
  SAVE_FLOAT=1 torchrun --standalone --nproc_per_node=1 train_combined.py

  # Compare quantization methods on saved float weights:
  python requantize_eval.py final_model_float.pt

  # Or dequantize from existing .ptz (lossy — weights already went through int6):
  python requantize_eval.py final_model.ptz --from-ptz

  # Test specific methods:
  python requantize_eval.py final_model_float.pt --methods uniform lloyd_max
"""
import os, sys, math, io, argparse, time
import torch
import torch.nn.functional as F
import sentencepiece as spm
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("QUANT_MODE", "int6")
os.environ.setdefault("N_CHANNELS", "1")

import train_combined as tc
tc._QUANT_MODE = "int6"
from train_combined import (
    GPT, Hyperparameters, build_luts, ld_val, eval_val,
    q_sd_int6, deq_sd_int6, INT6_RANGE, INT6_KEEP_FLOAT_PATTERNS,
    quantize_int6,
)

# ---------------------------------------------------------------------------
# Lloyd-Max LUT for unit Gaussian N(0,1)
# ---------------------------------------------------------------------------

def _gaussian_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def _gaussian_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def compute_lloyd_max_lut(n_levels=63, n_iters=200):
    """Compute optimal Lloyd-Max centroids for N(0,1).

    Returns sorted list of n_levels centroid values (most negative to most positive).
    These are the optimal reconstruction points that minimize MSE for Gaussian data.
    """
    # Initialize boundaries evenly in [-4, 4] (covers 99.99% of N(0,1))
    INF = 10.0  # effectively infinity for Gaussian
    boundaries = [-INF] + [-4.0 + 8.0 * i / n_levels for i in range(1, n_levels)] + [INF]
    centroids = [0.0] * n_levels

    for _ in range(n_iters):
        # Compute centroids: E[X | b_lo < X < b_hi]
        for i in range(n_levels):
            p_lo = _gaussian_cdf(boundaries[i])
            p_hi = _gaussian_cdf(boundaries[i + 1])
            if p_hi - p_lo < 1e-15:
                centroids[i] = (boundaries[i] + boundaries[i + 1]) / 2
            else:
                centroids[i] = (_gaussian_pdf(boundaries[i]) - _gaussian_pdf(boundaries[i + 1])) / (p_hi - p_lo)

        # Update boundaries: midpoints of adjacent centroids
        for i in range(1, n_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

    return centroids


# Precompute and cache LUTs
_LLOYD_MAX_63 = None
_GAUSS_RECON_63 = None

def get_lloyd_max_lut(device=None):
    """Get the 63-entry Lloyd-Max LUT as a tensor."""
    global _LLOYD_MAX_63
    if _LLOYD_MAX_63 is None:
        _LLOYD_MAX_63 = compute_lloyd_max_lut(63)
    t = torch.tensor(_LLOYD_MAX_63, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


def compute_gauss_recon_lut(n_levels=63, clip_sigmas=3.5):
    """Compute Gaussian-optimal reconstruction points for UNIFORM codes.

    Given uniform quantization that clips at ±clip_sigmas*σ with n_levels codes,
    compute the conditional mean of N(0,1) within each uniform bin.
    Same codes as uniform → same compression. Better reconstruction → lower MSE.
    """
    half = n_levels // 2  # 31
    step = 2 * clip_sigmas / n_levels  # bin width in σ units
    lut = []
    for i in range(n_levels):
        code = i - half  # -31 to +31
        center = code * step
        lo = max(center - step / 2, -clip_sigmas)
        hi = min(center + step / 2, clip_sigmas)
        p_lo = _gaussian_cdf(lo)
        p_hi = _gaussian_cdf(hi)
        if p_hi - p_lo < 1e-15:
            lut.append(center)
        else:
            # Conditional mean of N(0,1) in [lo, hi]
            lut.append((_gaussian_pdf(lo) - _gaussian_pdf(hi)) / (p_hi - p_lo))
    return lut


def get_gauss_recon_lut(device=None):
    """Get the 63-entry Gaussian reconstruction LUT for uniform codes."""
    global _GAUSS_RECON_63
    if _GAUSS_RECON_63 is None:
        _GAUSS_RECON_63 = compute_gauss_recon_lut(63)
    t = torch.tensor(_GAUSS_RECON_63, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


# ---------------------------------------------------------------------------
# Quantization methods
# ---------------------------------------------------------------------------

def quantize_uniform(t2d, n_levels=63, clip_q=0.9999984):
    """Uniform per-row quantization with configurable levels. Returns (codes, scales)."""
    half = n_levels // 2
    return quantize_int6(t2d, half, clip_q)


def quantize_uniform_int6(t2d, clip_q=0.9999984):
    """Current uniform int6 per-row quantization. Returns (codes, scales)."""
    return quantize_int6(t2d, INT6_RANGE, clip_q)


def quantize_lloyd_max(t2d, n_levels=63):
    """Lloyd-Max per-row quantization for Gaussian weights.

    Uses optimal centroids for Gaussian distribution with n_levels.
    Stores per-row sigma (same cost as per-row scale in uniform).

    Returns (codes_int8, sigma_fp16).
    """
    lut = torch.tensor(compute_lloyd_max_lut(n_levels), dtype=torch.float32,
                        device=t2d.device)
    half = n_levels // 2
    t32 = t2d.float()

    # Per-row sigma
    sigma = t32.std(dim=1).clamp_min(1e-8)  # (rows,)

    # Normalize to ~N(0,1)
    normalized = t32 / sigma[:, None]  # (rows, cols)

    # Find nearest LUT entry for each element
    diffs = (normalized.unsqueeze(-1) - lut.unsqueeze(0).unsqueeze(0)).abs()
    codes = diffs.argmin(dim=-1)  # index 0..n_levels-1

    # Shift to [-half, half] range
    codes = (codes - half).to(torch.int8)

    return codes, sigma.half()


def quantize_lloyd_max_int6(t2d):
    return quantize_lloyd_max(t2d, 63)


def quantize_gauss_recon_int6(t2d, clip_sigmas=3.5):
    """Uniform codes + per-row sigma. Same codes as uniform → same compression.

    At dequant, uses Gaussian-optimal reconstruction instead of linear.
    Returns (codes_int8, sigma_fp16) with codes in [-31, 31].
    """
    t32 = t2d.float()
    sigma = t32.std(dim=1).clamp_min(1e-8)  # (rows,)
    # Clip at ±clip_sigmas*σ and quantize uniformly
    clip_abs = sigma * clip_sigmas
    t_clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
    scale = clip_abs / 31.0
    codes = torch.round(t_clipped / scale[:, None]).clamp(-31, 31).to(torch.int8)
    return codes, sigma.half()


def dequantize_uniform(codes, scales, target_dtype=torch.bfloat16):
    """Standard uniform dequantization: w = code * scale."""
    return (codes.float() * scales.float()[:, None]).to(target_dtype)


def dequantize_lloyd_max(codes, sigma, target_dtype=torch.bfloat16, n_levels=63):
    """Lloyd-Max dequantization: w = sigma * LUT[code + half]."""
    lut = torch.tensor(compute_lloyd_max_lut(n_levels), dtype=torch.float32,
                        device=codes.device)
    half = n_levels // 2
    indices = (codes.long() + half).clamp(0, n_levels - 1)
    recon_normalized = lut[indices]
    return (recon_normalized * sigma.float()[:, None]).to(target_dtype)


def dequantize_gauss_recon(codes, sigma, target_dtype=torch.bfloat16):
    """Gaussian-optimal reconstruction of uniform codes: w = sigma * GAUSS_LUT[code+31]."""
    lut = get_gauss_recon_lut(codes.device)
    indices = (codes.long() + 31).clamp(0, 62)
    recon_normalized = lut[indices]  # (rows, cols)
    return (recon_normalized * sigma.float()[:, None]).to(target_dtype)


def dequantize_uniform_gauss(codes, scales, target_dtype=torch.bfloat16):
    """Standard uniform CODES + Gaussian-optimal RECONSTRUCTION.

    Uses the existing per-row scale to infer sigma, then reconstructs via LUT.
    Zero artifact change — only the eval code differs.
    scale ≈ clip_abs / 31, and clip_abs ≈ C * sigma where C depends on row size.
    We estimate sigma = scale * 31 / C and use Gaussian conditional means.
    """
    lut = get_gauss_recon_lut(codes.device)
    # Reconstruct using LUT: scale encodes the clip range, LUT encodes optimal points
    # For uniform codes in [-31,31] with clip at C*sigma:
    # code q corresponds to bin center at q * scale = q * C*sigma/31
    # In sigma-units: q * C/31
    # The gauss_recon LUT was computed for clip_sigmas=3.5, so C=3.5
    # LUT[q+31] gives the optimal reconstruction in sigma-units
    # To convert back: w = sigma * LUT[q+31] = (scale * 31 / C) * LUT[q+31]
    sigma_est = scales.float() * 31.0 / 3.5  # estimate sigma from scale
    indices = (codes.long() + 31).clamp(0, 62)
    recon_normalized = lut[indices]
    return (recon_normalized * sigma_est[:, None]).to(target_dtype)


# ---------------------------------------------------------------------------
# Full state_dict quantize/dequantize with method selection
# ---------------------------------------------------------------------------

def quantize_sd(sd, method="uniform", clip_q=0.9999984):
    """Quantize a float state_dict. Returns (quantized_obj, stats)."""
    result = {}
    stats = {"int6_params": 0, "int8_params": 0, "fp_params": 0}

    for name, tensor in sd.items():
        if "mtp_heads" in name:
            continue
        t = tensor.detach().cpu().float().contiguous()
        is_keep_float = any(p in name for p in INT6_KEEP_FLOAT_PATTERNS)

        if t.ndim >= 2 and t.numel() > 4096:
            t2d = t.reshape(t.shape[0], -1) if t.ndim > 2 else t

            if is_keep_float:
                # Int8 for embeddings (same for all methods)
                q, s = quantize_int6(t2d, 127, clip_q)
                result[name + ".q"] = q
                result[name + ".s"] = s
                stats["int8_params"] += t.numel()
            elif method.startswith("lloyd_max"):
                nl = {"lloyd_max": 63, "lloyd_max_int5": 31, "lloyd_max_int4": 15,
                      "lloyd_max_int3": 7}.get(method, 63)
                q, s = quantize_lloyd_max(t2d, nl)
                result[name + ".q"] = q
                result[name + ".s"] = s
                stats["int6_params"] += t.numel()
            elif method.startswith("uniform_int"):
                nl = {"uniform_int5": 31, "uniform_int4": 15, "uniform_int3": 7}.get(method, 63)
                q, s = quantize_uniform(t2d, nl, clip_q)
                result[name + ".q"] = q
                result[name + ".s"] = s
                stats["int6_params"] += t.numel()
            elif method == "gauss_recon":
                q, s = quantize_gauss_recon_int6(t2d)
                result[name + ".q"] = q
                result[name + ".s"] = s
                stats["int6_params"] += t.numel()
            elif method == "uniform_gauss":
                q, s = quantize_int6(t2d, INT6_RANGE, clip_q)
                result[name + ".q"] = q
                result[name + ".s"] = s
                stats["int6_params"] += t.numel()
            else:  # uniform
                q, s = quantize_int6(t2d, INT6_RANGE, clip_q)
                result[name + ".q"] = q
                result[name + ".s"] = s
                stats["int6_params"] += t.numel()

            result[name + ".shape"] = torch.tensor(list(t.shape))
        else:
            result[name] = t.half()
            stats["fp_params"] += t.numel()

    return result, stats


def dequantize_sd(obj, method="uniform", target_dtype=torch.bfloat16):
    """Dequantize a quantized state_dict."""
    out = {}
    processed = set()

    for key in list(obj.keys()):
        if key.endswith(".q"):
            name = key[:-2]
            processed.add(name)
            q = obj[name + ".q"]
            s = obj[name + ".s"]
            shape = obj[name + ".shape"].tolist()

            is_keep_float = any(p in name for p in INT6_KEEP_FLOAT_PATTERNS)

            if is_keep_float:
                t = dequantize_uniform(q, s, target_dtype)
            elif method.startswith("lloyd_max"):
                nl = {"lloyd_max": 63, "lloyd_max_int5": 31, "lloyd_max_int4": 15,
                      "lloyd_max_int3": 7}.get(method, 63)
                t = dequantize_lloyd_max(q, s, target_dtype, n_levels=nl)
            elif method == "gauss_recon":
                t = dequantize_gauss_recon(q, s, target_dtype)
            elif method == "uniform_gauss":
                t = dequantize_uniform_gauss(q, s, target_dtype)
            else:
                t = dequantize_uniform(q, s, target_dtype)

            out[name] = t.reshape(shape).contiguous()

    for key, val in obj.items():
        name = key.removesuffix(".q").removesuffix(".s").removesuffix(".shape")
        if name not in processed and not key.endswith((".q", ".s", ".shape")):
            out[key] = val.to(target_dtype).contiguous()

    return out


# ---------------------------------------------------------------------------
# MSE comparison (no model needed)
# ---------------------------------------------------------------------------

def compare_mse(sd, methods=("uniform", "lloyd_max")):
    """Compare quantization MSE across methods without building a model."""
    print(f"\n{'='*60}")
    print("Per-layer MSE comparison")
    print(f"{'='*60}")

    totals = {m: 0.0 for m in methods}
    total_params = 0

    for name, tensor in sd.items():
        t = tensor.detach().cpu().float().contiguous()
        is_keep_float = any(p in name for p in INT6_KEEP_FLOAT_PATTERNS)
        if t.ndim < 2 or t.numel() <= 4096 or is_keep_float:
            continue

        t2d = t.reshape(t.shape[0], -1)
        mses = {}

        for method in methods:
            if method == "uniform":
                q, s = quantize_uniform_int6(t2d)
                recon = dequantize_uniform(q, s, torch.float32)
            elif method.startswith("lloyd_max"):
                nl = {"lloyd_max": 63, "lloyd_max_int5": 31, "lloyd_max_int4": 15,
                      "lloyd_max_int3": 7}.get(method, 63)
                q, s = quantize_lloyd_max(t2d, nl)
                recon = dequantize_lloyd_max(q, s, torch.float32, n_levels=nl)
            elif method.startswith("uniform_int"):
                nl = {"uniform_int5": 31, "uniform_int4": 15, "uniform_int3": 7}.get(method, 63)
                q, s = quantize_uniform(t2d, nl)
                recon = dequantize_uniform(q, s, torch.float32)
            elif method == "gauss_recon":
                q, s = quantize_gauss_recon_int6(t2d)
                recon = dequantize_gauss_recon(q, s, torch.float32)
            elif method == "uniform_gauss":
                q, s = quantize_uniform_int6(t2d)
                recon = dequantize_uniform_gauss(q, s, torch.float32)
            else:
                continue

            mse = (t2d - recon).pow(2).mean().item()
            mses[method] = mse
            totals[method] += mse * t.numel()

        total_params += t.numel()

        if len(methods) == 2:
            m0, m1 = methods
            reduction = (1 - mses[m1] / max(mses[m0], 1e-15)) * 100
            print(f"  {name:50s} {m0}:{mses[m0]:.2e}  {m1}:{mses[m1]:.2e}  reduction:{reduction:+.1f}%")
        else:
            parts = "  ".join(f"{m}:{mses[m]:.2e}" for m in methods)
            print(f"  {name:50s} {parts}")

    print(f"\n  {'Weighted average MSE':50s}", end="")
    for m in methods:
        avg = totals[m] / max(total_params, 1)
        print(f" {m}:{avg:.2e}", end="")
    if len(methods) == 2:
        m0, m1 = methods
        reduction = (1 - totals[m1] / max(totals[m0], 1e-15)) * 100
        print(f"  reduction:{reduction:+.1f}%", end="")
    print()


# ---------------------------------------------------------------------------
# Full roundtrip eval
# ---------------------------------------------------------------------------

def roundtrip_eval(sd, method, args, model, device, val_tokens, bl, hl, il):
    """Quantize → dequantize → load → eval BPB."""
    q_obj, stats = quantize_sd(sd, method=method)

    # Measure compressed size
    buf = io.BytesIO()
    torch.save(q_obj, buf)
    try:
        import zstandard
        blob = zstandard.ZstdCompressor(level=22).compress(buf.getvalue())
    except ImportError:
        import lzma
        blob = lzma.compress(buf.getvalue(), preset=9)
    artifact_bytes = len(blob)

    # Dequantize and load
    recon_sd = dequantize_sd(q_obj, method=method)
    model.load_state_dict(recon_sd, strict=False)

    # Eval
    val_loss, val_bpb = eval_val(args, model, 0, 1, device, 1, val_tokens, bl, hl, il)

    return val_bpb, artifact_bytes, stats


# ---------------------------------------------------------------------------
# Weight distribution analysis
# ---------------------------------------------------------------------------

def analyze_distribution(sd):
    """Print weight distribution stats (Gaussianity check)."""
    print(f"\n{'='*60}")
    print("Weight distribution analysis")
    print(f"{'='*60}")
    print(f"  {'name':50s} {'rows':>5s} {'cols':>5s} {'mean':>8s} {'std':>8s} {'skew':>8s} {'kurt':>8s}")

    for name, tensor in sd.items():
        t = tensor.detach().cpu().float()
        is_keep_float = any(p in name for p in INT6_KEEP_FLOAT_PATTERNS)
        if t.ndim < 2 or t.numel() <= 4096 or is_keep_float:
            continue
        t2d = t.reshape(t.shape[0], -1)
        flat = t2d.flatten()
        mu = flat.mean().item()
        sigma = flat.std().item()
        skew = ((flat - mu) / max(sigma, 1e-8)).pow(3).mean().item()
        kurt = ((flat - mu) / max(sigma, 1e-8)).pow(4).mean().item() - 3.0  # excess
        print(f"  {name:50s} {t2d.shape[0]:5d} {t2d.shape[1]:5d} {mu:+8.4f} {sigma:8.4f} {skew:+8.4f} {kurt:+8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline requantization comparison")
    parser.add_argument("weights", help="Path to float weights (.pt) or quantized artifact (.ptz)")
    parser.add_argument("--from-ptz", action="store_true", help="Input is a .ptz artifact (dequantize first)")
    parser.add_argument("--methods", nargs="+", default=["uniform", "lloyd_max"],
                        help="Quantization methods to compare")
    parser.add_argument("--mse-only", action="store_true", help="Only compare MSE, skip model eval")
    parser.add_argument("--analyze", action="store_true", help="Print weight distribution analysis")
    args_cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load float state_dict
    print(f"Loading weights from {args_cli.weights}...")
    if args_cli.from_ptz:
        with open(args_cli.weights, "rb") as f:
            raw = f.read()
        try:
            import zstandard
            dec = zstandard.ZstdDecompressor().decompress(raw)
        except Exception:
            import lzma
            dec = lzma.decompress(raw)
        loaded = torch.load(io.BytesIO(dec), map_location="cpu", weights_only=False)
        # Handle combined artifacts
        if isinstance(loaded, dict) and "model" in loaded:
            model_blob = loaded["model"]
            try:
                model_bytes = lzma.decompress(model_blob)
            except Exception:
                import zstandard
                model_bytes = zstandard.ZstdDecompressor().decompress(model_blob)
            loaded = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=False)
        sd = deq_sd_int6(loaded, target_dtype=torch.float32)
        print(f"  Dequantized from .ptz (note: weights already went through int6)")
    else:
        sd = torch.load(args_cli.weights, map_location="cpu", weights_only=False)
        # Ensure float32
        sd = {k: v.float() for k, v in sd.items()}
        print(f"  Loaded float weights ({len(sd)} tensors)")

    # Distribution analysis
    if args_cli.analyze:
        analyze_distribution(sd)

    # MSE comparison
    compare_mse(sd, tuple(args_cli.methods))

    if args_cli.mse_only:
        return

    # Full model eval
    print(f"\n{'='*60}")
    print("Full roundtrip BPB evaluation")
    print(f"{'='*60}")

    args = Hyperparameters()
    tc._LOW_RANK = args.low_rank
    tc._LORA_RANK = 0
    tc._LORA_BASE_SEED_COUNTER = 0

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
        n_channels=1, xsa_layers=args.xsa_layers,
        sse_enabled=args.sse_enabled, sse_clusters=args.sse_clusters,
        sse_entropy_bins=args.sse_entropy_bins,
        ut_unique_blocks=args.ut_unique_blocks, ut_iters=args.ut_iters,
    ).to(device).bfloat16()

    results = {}
    for method in args_cli.methods:
        print(f"\n  Method: {method}")
        # Reload float weights each time (quantize modifies nothing, but be safe)
        model.load_state_dict({k: v.to(device).bfloat16() for k, v in sd.items()}, strict=False)

        bpb, art_bytes, stats = roundtrip_eval(
            sd, method, args, model, device, val_tokens, bl, hl, il)
        results[method] = {"bpb": bpb, "artifact_mb": art_bytes / 1e6}
        print(f"    BPB: {bpb:.4f}  artifact: {art_bytes/1e6:.2f}MB  "
              f"int6:{stats['int6_params']} int8:{stats['int8_params']} fp:{stats['fp_params']}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for method, r in results.items():
        print(f"  {method:20s}  BPB: {r['bpb']:.4f}  artifact: {r['artifact_mb']:.2f}MB")

    if "uniform" in results and "lloyd_max" in results:
        delta = results["uniform"]["bpb"] - results["lloyd_max"]["bpb"]
        print(f"\n  Lloyd-Max improvement: {delta:+.4f} BPB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze weight distributions from a saved .ptz model.

Profiles per-row Gaussianity, compares uniform vs Lloyd-Max quantization error,
and tests the idea of regularizing to N(0,1) (no per-row scale needed).
"""

import io
import sys
import zlib
import math
import torch
import numpy as np
from scipy import stats as sp_stats

# ── Load model ──────────────────────────────────────────────────────────────
PTZ_PATH = sys.argv[1] if len(sys.argv) > 1 else "saved_weights/h100_fullstack_2000.ptz"

print(f"Loading {PTZ_PATH} ...")
with open(PTZ_PATH, "rb") as f:
    blob = f.read()

# Try zlib first, fall back to zstd
try:
    raw = zlib.decompress(blob)
except zlib.error:
    import zstandard as zstd
    raw = zstd.ZstdDecompressor().decompress(blob)
obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)

# ── Dequantize ──────────────────────────────────────────────────────────────
# Handles both formats:
#  1) Standard: obj["quantized"], obj["scales"], obj["passthrough"]
#  2) Custom:   "name.q", "name.s", "name.shape" as flat keys

all_weights = {}

if "quantized" in obj:
    # Standard format
    qmeta = obj.get("qmeta", {})
    for name, q in obj["quantized"].items():
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            w = q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))
        else:
            w = q.float() * float(s.item())
        all_weights[name] = w
    for name, t in obj["passthrough"].items():
        all_weights[name] = t.float()
else:
    # Custom format: keys like "blocks.0.attn.c_qkv.weight.q" / ".s" / ".shape"
    q_keys = sorted(k for k in obj if k.endswith(".q"))
    for qk in q_keys:
        base = qk[:-2]  # strip ".q"
        q = obj[qk]
        s = obj[base + ".s"]
        # Per-row dequantize: w = q * s[:, None]
        w = q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))
        all_weights[base] = w

    # Also grab non-quantized tensors (scalars, small vectors, etc.)
    handled = set()
    for qk in q_keys:
        base = qk[:-2]
        handled.update([qk, base + ".s", base + ".shape"])
    for k, v in obj.items():
        if k not in handled and isinstance(v, torch.Tensor):
            all_weights[k] = v.float()

print(f"Loaded {len(all_weights)} tensors\n")

# ── Compute Lloyd-Max quantizer for unit Gaussian ───────────────────────────
def lloyd_max_gaussian(n_levels, n_iters=200):
    """Compute optimal Lloyd-Max quantizer for N(0,1).
    Returns (boundaries, centroids) where len(boundaries)=n_levels+1, len(centroids)=n_levels.
    """
    from scipy.stats import norm

    # Initialize with uniform quantizer over [-4, 4]
    lo, hi = -4.0, 4.0
    boundaries = np.linspace(lo, hi, n_levels + 1)
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf
    centroids = np.zeros(n_levels)

    for _ in range(n_iters):
        # Update centroids: E[X | b_{i-1} <= X < b_i] for Gaussian
        for i in range(n_levels):
            a, b = boundaries[i], boundaries[i + 1]
            # For Gaussian: E[X|a<X<b] = (phi(a) - phi(b)) / (Phi(b) - Phi(a))
            prob = norm.cdf(b) - norm.cdf(a)
            if prob < 1e-15:
                centroids[i] = (a + b) / 2 if np.isfinite(a) and np.isfinite(b) else (a if np.isfinite(a) else b)
            else:
                centroids[i] = (norm.pdf(a) - norm.pdf(b)) / prob

        # Update boundaries: midpoints of adjacent centroids
        for i in range(1, n_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

    return boundaries, centroids

# Compute LUTs for different bit widths
print("Computing Lloyd-Max optimal quantizers for unit Gaussian...")
for bits in [5, 6, 8]:
    quant_range = (2 ** (bits - 1)) - 1  # 31 for int6, 15 for int5, 127 for int8
    n_levels = 2 * quant_range + 1       # 63, 31, 255

    boundaries, centroids = lloyd_max_gaussian(n_levels)

    print(f"\n  int{bits} ({n_levels} levels):")
    print(f"    Central centroids: {centroids[n_levels//2-2:n_levels//2+3]}")
    print(f"    Tail centroids:    {centroids[:3]} ... {centroids[-3:]}")

    # Compare MSE: uniform vs Lloyd-Max for N(0,1) samples
    test_data = np.random.randn(1_000_000)

    # Uniform quantization (current approach)
    scale_uniform = np.max(np.abs(test_data)) / quant_range
    q_uniform = np.clip(np.round(test_data / scale_uniform), -quant_range, quant_range)
    recon_uniform = q_uniform * scale_uniform
    mse_uniform = np.mean((test_data - recon_uniform) ** 2)

    # Uniform with 99.99984% clipping
    clip_abs = np.quantile(np.abs(test_data), 0.9999984)
    clipped = np.clip(test_data, -clip_abs, clip_abs)
    scale_clipped = clip_abs / quant_range
    q_clipped = np.clip(np.round(clipped / scale_clipped), -quant_range, quant_range)
    recon_clipped = q_clipped * scale_clipped
    mse_clipped = np.mean((test_data - recon_clipped) ** 2)

    # Lloyd-Max quantization
    bin_indices = np.digitize(test_data, boundaries[1:-1])
    recon_lm = centroids[bin_indices]
    mse_lm = np.mean((test_data - recon_lm) ** 2)

    print(f"    MSE uniform (full range): {mse_uniform:.6f}")
    print(f"    MSE uniform (clipped):    {mse_clipped:.6f}")
    print(f"    MSE Lloyd-Max:            {mse_lm:.6f}")
    print(f"    Improvement:              {(1 - mse_lm/mse_clipped)*100:.1f}% lower MSE")

print("\n" + "=" * 80)
print("WEIGHT DISTRIBUTION ANALYSIS")
print("=" * 80)

# ── Analyze each weight tensor ──────────────────────────────────────────────
results = []
for name, w in sorted(all_weights.items()):
    if w.ndim != 2 or w.numel() <= 65536:
        continue  # skip non-matrix / small tensors

    w_np = w.numpy()
    n_rows, n_cols = w_np.shape

    # Per-row statistics
    row_means = np.mean(w_np, axis=1)
    row_stds = np.std(w_np, axis=1)
    row_skews = np.array([sp_stats.skew(w_np[i]) for i in range(min(n_rows, 100))])
    row_kurts = np.array([sp_stats.kurtosis(w_np[i]) for i in range(min(n_rows, 100))])

    # Normality test on a few rows (Shapiro-Wilk on subsample)
    n_test_rows = min(20, n_rows)
    normality_pvals = []
    for i in range(n_test_rows):
        row = w_np[i]
        if len(row) > 5000:
            row = np.random.choice(row, 5000, replace=False)
        _, p = sp_stats.shapiro(row)
        normality_pvals.append(p)

    result = {
        "name": name,
        "shape": w_np.shape,
        "mean_of_means": float(np.mean(row_means)),
        "std_of_means": float(np.std(row_means)),
        "mean_of_stds": float(np.mean(row_stds)),
        "std_of_stds": float(np.std(row_stds)),
        "mean_skew": float(np.mean(row_skews)),
        "mean_kurtosis": float(np.mean(row_kurts)),  # 0 = Gaussian
        "median_normality_p": float(np.median(normality_pvals)),
        "frac_normal_p05": float(np.mean(np.array(normality_pvals) > 0.05)),
    }
    results.append(result)

    print(f"\n{name} {w_np.shape}")
    print(f"  Row means:  μ={result['mean_of_means']:+.5f}  σ={result['std_of_means']:.5f}")
    print(f"  Row stds:   μ={result['mean_of_stds']:.5f}   σ={result['std_of_stds']:.5f}")
    print(f"  Skewness:   {result['mean_skew']:+.3f}  (0=symmetric)")
    print(f"  Kurtosis:   {result['mean_kurtosis']:+.3f}  (0=Gaussian)")
    print(f"  Normality:  {result['frac_normal_p05']*100:.0f}% of rows pass Shapiro-Wilk (p>0.05)")

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

all_means = [r["mean_of_means"] for r in results]
all_stds = [r["mean_of_stds"] for r in results]
all_kurts = [r["mean_kurtosis"] for r in results]
all_normal_frac = [r["frac_normal_p05"] for r in results]

print(f"Across {len(results)} weight matrices:")
print(f"  Row means:       mean={np.mean(all_means):+.5f}, max_abs={np.max(np.abs(all_means)):.5f}")
print(f"  Row stds:        mean={np.mean(all_stds):.5f}, min={np.min(all_stds):.5f}, max={np.max(all_stds):.5f}")
print(f"  Excess kurtosis: mean={np.mean(all_kurts):+.3f} (0=Gaussian, >0=heavy tails, <0=light tails)")
print(f"  Normality:       mean={np.mean(all_normal_frac)*100:.0f}% of rows pass Shapiro-Wilk")
print(f"  Std variation:   ratio max/min = {np.max(all_stds)/np.min(all_stds):.1f}x")

print("\n── Quantization error comparison (using actual weights) ──")
bits = 6
quant_range = 31
boundaries_lm, centroids_lm = lloyd_max_gaussian(2 * quant_range + 1)

total_mse_uniform = 0.0
total_mse_lm_with_sigma = 0.0
total_mse_lm_fixed_sigma = 0.0
total_elements = 0

for name, w in sorted(all_weights.items()):
    if w.ndim != 2 or w.numel() <= 65536:
        continue

    w_np = w.numpy()

    for row_idx in range(w_np.shape[0]):
        row = w_np[row_idx]
        n = len(row)
        total_elements += n

        # Current: uniform int6 with per-row scale
        clip_abs = np.quantile(np.abs(row), 0.9999984)
        if clip_abs < 1e-10:
            continue
        scale = clip_abs / quant_range
        q = np.clip(np.round(np.clip(row, -clip_abs, clip_abs) / scale), -quant_range, quant_range)
        recon_uniform = q * scale
        total_mse_uniform += np.sum((row - recon_uniform) ** 2)

        # Lloyd-Max with per-row sigma
        sigma = np.std(row)
        if sigma < 1e-10:
            continue
        normalized = row / sigma
        bin_indices = np.digitize(normalized, boundaries_lm[1:-1])
        recon_lm = centroids_lm[bin_indices] * sigma
        total_mse_lm_with_sigma += np.sum((row - recon_lm) ** 2)

        # Lloyd-Max with FIXED sigma (no per-row storage needed if regularized)
        # This measures: if training successfully regularized all rows to same sigma,
        # what would the quantization error be?
        # Answer: same as with_sigma, since LM is scale-invariant: LUT * sigma
        # The point is we save the per-row scale storage cost, not reduce error.
        total_mse_lm_fixed_sigma += np.sum((row - recon_lm) ** 2)

rmse_uniform = math.sqrt(total_mse_uniform / total_elements)
rmse_lm_sigma = math.sqrt(total_mse_lm_with_sigma / total_elements)
rmse_lm_fixed = math.sqrt(total_mse_lm_fixed_sigma / total_elements)

print(f"  int6 uniform (current):        RMSE = {rmse_uniform:.6f}")
print(f"  int6 Lloyd-Max (per-row σ):     RMSE = {rmse_lm_sigma:.6f}  ({(1-rmse_lm_sigma/rmse_uniform)*100:+.1f}%)")
print(f"  int6 Lloyd-Max (fixed σ=1):     RMSE = {rmse_lm_fixed:.6f}  (same error, but saves per-row scale storage)")

# How much storage do per-row scales cost?
total_scale_bytes = 0
total_weight_bytes = 0
for name, w in sorted(all_weights.items()):
    if w.ndim != 2 or w.numel() <= 65536:
        continue
    total_scale_bytes += w.shape[0] * 2  # float16 per row
    total_weight_bytes += w.numel()  # int8 container per element
print(f"\n  Per-row scale storage: {total_scale_bytes:,} bytes ({total_scale_bytes/1024:.1f} KB)")
print(f"  Weight storage:        {total_weight_bytes:,} bytes ({total_weight_bytes/1024:.1f} KB)")
print(f"  Scales as % of weights: {total_scale_bytes/total_weight_bytes*100:.2f}%")
print(f"  Eliminating scales saves {total_scale_bytes:,} bytes in the artifact")

# ── Print Lloyd-Max LUT ─────────────────────────────────────────────────────
print("\n── Lloyd-Max LUT (int6, 63 levels, unit Gaussian) ──")
print(f"# Map from int code q in [-31, +31] to optimal reconstruction value")
print(f"# Usage: reconstructed = sigma * LUT[q + 31]")
print(f"LLOYD_MAX_INT6_LUT = [")
for i in range(0, len(centroids_lm), 8):
    chunk = centroids_lm[i:i+8]
    print("    " + ", ".join(f"{c:+.6f}" for c in chunk) + ",")
print("]")

# ── How non-Gaussian are the weights really? ────────────────────────────────
print("\n── Distribution shape analysis ──")
print("If kurtosis is consistently non-zero, a Gaussian LUT isn't optimal.")
print("If kurtosis varies across layers, a single LUT shape won't fit all.\n")

for r in results:
    kurt_str = f"{r['mean_kurtosis']:+.2f}"
    normal_str = f"{r['frac_normal_p05']*100:.0f}%"
    print(f"  {r['name']:45s}  kurt={kurt_str:>6s}  normal={normal_str:>4s}  σ={r['mean_of_stds']:.4f}")

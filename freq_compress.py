"""Frequency-domain weight compression: DCT/wavelet analysis of weight matrices.

Tests whether weight matrices have exploitable structure in frequency domain,
like how JPEG exploits spatial correlation in images.

Usage:
  python freq_compress.py final_model_float.pt
"""
import sys, torch, io, math, argparse
import numpy as np

def dct1d(x):
    """Type-II DCT along last dimension, via FFT."""
    N = x.shape[-1]
    # Mirror: [x0, x1, ..., xN-1, xN-1, ..., x1, x0] but interleaved
    v = torch.cat([x[..., ::2], x[..., 1::2].flip(-1)], dim=-1)
    V = torch.fft.fft(v, dim=-1)
    k = torch.arange(N, device=x.device, dtype=torch.float32)
    shift = torch.exp(-1j * math.pi * k / (2 * N))
    return (V * shift).real * math.sqrt(2.0 / N)


def idct1d(X):
    """Type-II inverse DCT along last dimension."""
    N = X.shape[-1]
    k = torch.arange(N, device=X.device, dtype=torch.float32)
    shift = torch.exp(1j * math.pi * k / (2 * N))
    X_complex = X.to(torch.complex64) * shift * math.sqrt(N / 2.0)
    v = torch.fft.ifft(X_complex, dim=-1).real
    # Unshuffle
    out = torch.zeros_like(v)
    out[..., ::2] = v[..., :N // 2 + N % 2]
    out[..., 1::2] = v[..., N // 2 + N % 2:].flip(-1)
    return out


def analyze_energy(weight, name, fracs=[0.1, 0.25, 0.5, 0.75]):
    """Analyze DCT energy concentration along rows and columns."""
    w = weight.float()
    rows, cols = w.shape

    # Row-wise DCT (along input dimension)
    row_dct = dct1d(w)
    row_energy = row_dct.pow(2)
    row_total = row_energy.sum()

    # Column-wise DCT (along output dimension)
    col_dct = dct1d(w.T).T
    col_energy = col_dct.pow(2)

    # 2D DCT
    dct2d = dct1d(dct1d(w).T).T
    energy_2d = dct2d.pow(2)

    # Energy concentration: what fraction of energy is in first K coefficients?
    # Row-wise: cumulative energy along columns
    row_cum = row_energy.sum(dim=0).cumsum(dim=0) / row_total
    col_cum = col_energy.sum(dim=1).cumsum(dim=0) / row_total

    row_results = {f: row_cum[int(cols * f) - 1].item() for f in fracs}
    col_results = {f: col_cum[int(rows * f) - 1].item() for f in fracs}

    return row_results, col_results


def truncate_reconstruct(weight, keep_frac, dim="row"):
    """Truncate DCT coefficients and reconstruct. Returns (reconstructed, n_kept)."""
    w = weight.float()
    if dim == "row":
        coeffs = dct1d(w)
        K = max(1, int(w.shape[1] * keep_frac))
        truncated = coeffs.clone()
        truncated[:, K:] = 0
        recon = idct1d(truncated)
        return recon, K * w.shape[0]
    elif dim == "col":
        coeffs = dct1d(w.T).T
        K = max(1, int(w.shape[0] * keep_frac))
        truncated = coeffs.clone()
        truncated[K:, :] = 0
        recon = idct1d(truncated.T).T
        return recon, K * w.shape[1]
    else:  # 2D
        coeffs = dct1d(dct1d(w).T).T
        # Keep top-K by magnitude
        flat = coeffs.abs().flatten()
        K = max(1, int(len(flat) * keep_frac))
        threshold = flat.topk(K).values[-1]
        mask = coeffs.abs() >= threshold
        truncated = coeffs * mask
        recon = idct1d(idct1d(truncated).T).T
        return recon, K


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", help="Path to float weights (.pt)")
    args = parser.parse_args()

    sd = torch.load(args.weights, map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in sd.items()}

    # Filter to large 2D matrices only
    matrices = {k: v for k, v in sd.items()
                if v.ndim == 2 and v.numel() > 4096
                and not any(p in k for p in ("tok_emb", "lm_head", "embed_proj", "lm_head_correction"))}

    print(f"Analyzing {len(matrices)} weight matrices\n")

    # 1. Energy concentration analysis
    print("=" * 70)
    print("Energy concentration (fraction of total energy in first K% of DCT coefficients)")
    print("=" * 70)
    print(f"{'name':50s} {'row 10%':>8s} {'row 25%':>8s} {'row 50%':>8s} {'col 10%':>8s} {'col 25%':>8s} {'col 50%':>8s}")

    all_row = {f: [] for f in [0.1, 0.25, 0.5]}
    all_col = {f: [] for f in [0.1, 0.25, 0.5]}

    for name, w in matrices.items():
        row_r, col_r = analyze_energy(w, name, [0.1, 0.25, 0.5])
        print(f"{name:50s} {row_r[0.1]:8.1%} {row_r[0.25]:8.1%} {row_r[0.5]:8.1%} "
              f"{col_r[0.1]:8.1%} {col_r[0.25]:8.1%} {col_r[0.5]:8.1%}")
        for f in [0.1, 0.25, 0.5]:
            all_row[f].append(row_r[f])
            all_col[f].append(col_r[f])

    print(f"\n{'AVERAGE':50s}", end="")
    for f in [0.1, 0.25, 0.5]:
        print(f" {sum(all_row[f])/len(all_row[f]):8.1%}", end="")
    for f in [0.1, 0.25, 0.5]:
        print(f" {sum(all_col[f])/len(all_col[f]):8.1%}", end="")
    print()

    # 2. Reconstruction MSE at various keep fractions
    print(f"\n{'=' * 70}")
    print("Row-wise DCT truncation: MSE and equivalent bits")
    print(f"{'=' * 70}")

    # Also compute uniform int6 MSE for comparison
    import train_combined as tc
    tc._QUANT_MODE = "int6"
    from train_combined import quantize_int6, INT6_RANGE

    for keep_frac in [0.25, 0.5, 0.75, 1.0]:
        total_mse = 0
        total_params = 0
        total_kept = 0
        uni_mse = 0

        for name, w in matrices.items():
            recon, n_kept = truncate_reconstruct(w, keep_frac, dim="row")
            mse = (w - recon).pow(2).mean().item()
            total_mse += mse * w.numel()
            total_params += w.numel()
            total_kept += n_kept

            if keep_frac == 1.0:
                # Also compute uniform int6 MSE
                q, s = quantize_int6(w, INT6_RANGE)
                recon_uni = (q.float() * s.float()[:, None])
                uni_mse += (w - recon_uni).pow(2).mean().item() * w.numel()

        avg_mse = total_mse / total_params
        compression = total_params / total_kept
        eff_bits = 16.0 / compression  # if storing kept coeffs as fp16

        print(f"  keep={keep_frac:.0%}: MSE={avg_mse:.2e}, compression={compression:.1f}x, "
              f"eff_bits={eff_bits:.1f}b/weight, kept_params={total_kept/1e6:.1f}M")

        if keep_frac == 1.0:
            avg_uni = uni_mse / total_params
            print(f"  uniform int6: MSE={avg_uni:.2e} (for comparison)")

    # 3. Artifact size estimate
    print(f"\n{'=' * 70}")
    print("Estimated artifact sizes (DCT coefficients quantized to int8 + stored with zstd)")
    print(f"{'=' * 70}")

    import zstandard

    for keep_frac in [0.25, 0.5, 0.75]:
        all_coeffs = []
        all_scales = []
        for name, w in matrices.items():
            coeffs = dct1d(w)
            K = max(1, int(w.shape[1] * keep_frac))
            kept = coeffs[:, :K]  # first K coefficients per row
            # Quantize to int8 per-row
            scale = kept.abs().amax(dim=1).clamp_min(1e-8) / 127
            q = (kept / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
            all_coeffs.append(q)
            all_scales.append(scale.half())

        # Measure compressed size
        buf = io.BytesIO()
        torch.save({"coeffs": all_coeffs, "scales": all_scales}, buf)
        compressed = zstandard.ZstdCompressor(level=22).compress(buf.getvalue())

        # Also add embedding and small params (same as current)
        emb_size = 0
        for k, v in sd.items():
            if k not in matrices:
                emb_size += v.numel() * 2  # fp16

        total_est = len(compressed) + emb_size
        print(f"  keep={keep_frac:.0%}: DCT compressed={len(compressed)/1e6:.2f}MB + "
              f"emb/small={emb_size/1e6:.2f}MB = total ~{total_est/1e6:.2f}MB")


if __name__ == "__main__":
    main()

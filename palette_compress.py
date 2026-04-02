"""Weight Palette: learned codebook quantization for weight matrices.

Offline version: run k-means on saved float weights to find optimal codebook.
Each weight matrix gets its own small codebook (K entries). Each weight is an
index into the codebook. Unlike Lloyd-Max (assumes Gaussian, 1D per-row),
this adapts to the ACTUAL joint weight distribution per matrix.

Usage:
  python palette_compress.py final_model_float.pt --k 16 32 256
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
    GPT, Hyperparameters, build_luts, ld_val, eval_val,
    QuantizedLinear, LowRankLinear, restore_low_dim_params_to_fp32,
    INT6_KEEP_FLOAT_PATTERNS, quantize_int6,
)
import torch.nn as nn


def kmeans_1d(data, K, n_iters=50):
    """1D k-means on flattened weight values. Returns (centroids, assignments).

    Much faster than multi-dim k-means since we operate on scalar values.
    """
    flat = data.flatten().float()
    N = flat.numel()

    # Initialize centroids: evenly spaced quantiles
    indices = torch.linspace(0, N - 1, K).long()
    sorted_flat, _ = flat.sort()
    centroids = sorted_flat[indices].clone()

    for _ in range(n_iters):
        # Assign each value to nearest centroid
        diffs = (flat.unsqueeze(1) - centroids.unsqueeze(0)).abs()  # (N, K)
        assignments = diffs.argmin(dim=1)  # (N,)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(K, device=data.device)
        new_centroids.scatter_add_(0, assignments, flat)
        counts.scatter_add_(0, assignments, torch.ones_like(flat))
        mask = counts > 0
        new_centroids[mask] /= counts[mask]
        # Keep old centroids for empty clusters
        centroids = torch.where(mask, new_centroids, centroids)

    return centroids, assignments.reshape(data.shape)


def kmeans_1d_batched(data_2d, K, n_iters=50):
    """Per-row 1D k-means. Each row gets optimal codebook.
    Returns (codebooks [rows, K], assignments [rows, cols]).
    """
    rows, cols = data_2d.shape
    all_centroids = torch.zeros(rows, K, device=data_2d.device)
    all_assignments = torch.zeros(rows, cols, dtype=torch.long, device=data_2d.device)

    for r in range(rows):
        c, a = kmeans_1d(data_2d[r], K, n_iters)
        all_centroids[r] = c
        all_assignments[r] = a

    return all_centroids, all_assignments


def palette_quantize_matrix(weight_2d, K, per_row=False, n_iters=50):
    """Quantize a weight matrix using k-means palette.

    Returns (codebook, assignments, reconstructed).
    - per_row=False: one codebook for entire matrix (K × 2 bytes overhead)
    - per_row=True: one codebook per row (rows × K × 2 bytes overhead)
    """
    w = weight_2d.float()

    if per_row:
        codebook, assignments = kmeans_1d_batched(w, K, n_iters)
        # Reconstruct
        recon = codebook.gather(1, assignments)
        return codebook, assignments, recon
    else:
        codebook, assignments = kmeans_1d(w, K, n_iters)
        recon = codebook[assignments]
        return codebook, assignments, recon


def palette_quantize_sd(sd, K, per_row=False):
    """Quantize entire state dict with palette quantization."""
    result = {}
    stats = {"palette_params": 0, "int8_params": 0, "fp_params": 0,
             "codebook_bytes": 0}

    for name, tensor in sd.items():
        if "mtp_heads" in name:
            continue
        t = tensor.detach().cpu().float().contiguous()
        is_keep_float = any(p in name for p in INT6_KEEP_FLOAT_PATTERNS)

        if t.ndim >= 2 and t.numel() > 4096 and not is_keep_float:
            t2d = t.reshape(t.shape[0], -1) if t.ndim > 2 else t
            codebook, assignments, recon = palette_quantize_matrix(t2d, K, per_row)

            if K <= 256:
                result[name + ".idx"] = assignments.to(torch.uint8)
            else:
                result[name + ".idx"] = assignments.to(torch.int16)
            result[name + ".cb"] = codebook.half()
            result[name + ".shape"] = torch.tensor(list(t.shape))
            stats["palette_params"] += t.numel()
            if per_row:
                stats["codebook_bytes"] += codebook.numel() * 2
            else:
                stats["codebook_bytes"] += K * 2

        elif t.ndim >= 2 and t.numel() > 4096 and is_keep_float:
            t2d = t.reshape(t.shape[0], -1) if t.ndim > 2 else t
            q, s = quantize_int6(t2d, 127)
            result[name + ".q"] = q
            result[name + ".s"] = s
            result[name + ".shape"] = torch.tensor(list(t.shape))
            stats["int8_params"] += t.numel()
        else:
            result[name] = t.half()
            stats["fp_params"] += t.numel()

    return result, stats


def palette_dequantize_sd(obj, target_dtype=torch.bfloat16):
    """Reconstruct state dict from palette quantization."""
    out = {}
    processed = set()

    for key in list(obj.keys()):
        if key.endswith(".idx"):
            name = key[:-4]
            processed.add(name)
            idx = obj[name + ".idx"].long()
            cb = obj[name + ".cb"].float()
            shape = obj[name + ".shape"].tolist()

            if cb.ndim == 1:
                # Global codebook
                recon = cb[idx]
            else:
                # Per-row codebook
                recon = cb.gather(1, idx)

            out[name] = recon.to(target_dtype).reshape(shape).contiguous()

        elif key.endswith(".q"):
            name = key[:-2]
            if name not in processed:
                processed.add(name)
                q = obj[name + ".q"].float()
                s = obj[name + ".s"].float()
                shape = obj[name + ".shape"].tolist()
                t = (q * s[:, None]).to(target_dtype) if s.ndim > 0 else (q * s).to(target_dtype)
                out[name] = t.reshape(shape).contiguous()

    for key, val in obj.items():
        base = key.removesuffix(".idx").removesuffix(".cb").removesuffix(".q").removesuffix(".s").removesuffix(".shape")
        if base not in processed and not any(key.endswith(s) for s in (".idx", ".cb", ".q", ".s", ".shape")):
            out[key] = val.to(target_dtype).contiguous()

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", help="Path to float weights (.pt)")
    parser.add_argument("--k", nargs="+", type=int, default=[8, 16, 32, 256])
    parser.add_argument("--per-row", action="store_true", help="Per-row codebook (more overhead)")
    parser.add_argument("--eval", action="store_true", help="Run full BPB eval (needs GPU)")
    args_cli = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sd = torch.load(args_cli.weights, map_location="cpu", weights_only=False)
    sd = {k: v.float() for k, v in sd.items()}
    print(f"Loaded {len(sd)} tensors")

    # Filter to quantizable matrices for MSE comparison
    matrices = {k: v for k, v in sd.items()
                if v.ndim == 2 and v.numel() > 4096
                and not any(p in k for p in INT6_KEEP_FLOAT_PATTERNS)}
    total_params = sum(v.numel() for v in matrices.values())

    import zstandard

    print(f"\n{'='*70}")
    print(f"Weight Palette Quantization (per_row={args_cli.per_row})")
    print(f"{'='*70}")

    for K in args_cli.k:
        t0 = time.time()
        total_mse = 0

        # Quantize all matrices
        for name, w in matrices.items():
            t2d = w.reshape(w.shape[0], -1)
            cb, idx, recon = palette_quantize_matrix(t2d, K, args_cli.per_row)
            mse = (t2d - recon).pow(2).mean().item()
            total_mse += mse * w.numel()

        avg_mse = total_mse / total_params
        elapsed = time.time() - t0

        # Full state dict quantize + compress for artifact size
        q_obj, stats = palette_quantize_sd(sd, K, args_cli.per_row)
        buf = io.BytesIO()
        torch.save(q_obj, buf)
        compressed = zstandard.ZstdCompressor(level=22).compress(buf.getvalue())
        art_size = len(compressed)

        bits_per_idx = math.ceil(math.log2(K)) if K > 1 else 1
        fits = "FITS" if art_size + 20000 <= 16e6 else "OVER"

        print(f"\n  K={K:4d} ({bits_per_idx}b/weight): MSE={avg_mse:.2e}  "
              f"artifact={art_size/1e6:.2f}MB ({fits})  "
              f"codebook_overhead={stats['codebook_bytes']/1e3:.1f}KB  "
              f"time={elapsed:.0f}s")

        # Full BPB eval if requested
        if args_cli.eval and device.type == "cuda":
            args = Hyperparameters()
            tc._LOW_RANK = args.low_rank
            sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
            bl, hl, il = build_luts(sp, args.vocab_size, device)
            val_tokens = ld_val(args.val_files, args.train_seq_len)

            model = GPT(
                vocab_size=args.vocab_size, num_layers=args.num_layers,
                model_dim=args.model_dim, num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
                tie_embeddings=args.tie_embeddings,
                tied_embed_init_std=args.tied_embed_init_std,
                logit_softcap=args.logit_softcap, rope_base=args.rope_base,
                qk_gain_init=args.qk_gain_init,
                group_size=args.bitnet_group_size,
                activation=args.activation_type, n_channels=1,
                xsa_layers=args.xsa_layers, sse_enabled=args.sse_enabled,
                sse_clusters=args.sse_clusters,
                sse_entropy_bins=args.sse_entropy_bins,
            ).to(device).bfloat16()
            for m in model.modules():
                if isinstance(m, (nn.Linear, QuantizedLinear, LowRankLinear)):
                    m.float()
            restore_low_dim_params_to_fp32(model)

            # Load palette-quantized weights
            recon_sd = palette_dequantize_sd(q_obj)
            model_sd = model.state_dict()
            loaded = {k: v.to(device=device, dtype=model_sd[k].dtype)
                     for k, v in recon_sd.items() if k in model_sd}
            model.load_state_dict(loaded, strict=False)
            compiled = torch.compile(model)
            torch._dynamo.reset()

            vl, bpb = eval_val(args, compiled, 0, 1, device, 8, val_tokens, bl, hl, il)
            print(f"          BPB={bpb:.4f}")


if __name__ == "__main__":
    main()

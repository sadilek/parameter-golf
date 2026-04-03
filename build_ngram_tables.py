#!/usr/bin/env python3
"""Precompute n-gram top-K prediction tables from training shards.

For each n-gram order (2-8), maintains a hash table mapping
hash(context) -> token -> count. After processing all data,
extracts top-K most frequent next tokens per hash bucket.

Output: a compact .pt file containing:
  - top_tokens[order][bucket] = [top1_id, top2_id, top3_id]  (int16)
  - metadata (orders, num_buckets, top_k)

Usage:
  python3 build_ngram_tables.py [--shards N] [--buckets 131072] [--top-k 3]
"""
import argparse
import glob
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import time


def load_shard(path: Path) -> np.ndarray:
    """Load a binary shard file, return uint16 token array."""
    header = np.fromfile(path, dtype="<i4", count=256)
    n_tokens = int(header[2])
    return np.fromfile(path, dtype="<u2", count=n_tokens, offset=256 * 4)


def hash_context(tokens: np.ndarray, order: int, pos: int, num_buckets: int) -> int:
    """Hash the context tokens[pos-order+1:pos] to a bucket index."""
    h = 0
    for k in range(order - 1):
        h = (h * 1000003 + int(tokens[pos - order + 1 + k])) & 0xFFFFFFFF
    return h % num_buckets


def build_tables(
    shard_dir: str,
    num_shards: int,
    orders: list[int],
    num_buckets: int,
    top_k: int,
) -> dict:
    """Build n-gram top-K tables from training shards."""
    shard_files = sorted(glob.glob(f"{shard_dir}/fineweb_train_*.bin"))
    if num_shards > 0:
        shard_files = shard_files[:num_shards]
    print(f"Processing {len(shard_files)} shards, orders {orders}, {num_buckets} buckets, top-{top_k}")

    max_order = max(orders)

    # For memory efficiency, use numpy arrays instead of dicts.
    # For each order, track top-K tokens and their counts per bucket.
    # Strategy: process in chunks, maintain running top-K via count arrays.
    #
    # With 128K buckets and 1024 vocab, a full count matrix would be
    # 128K * 1024 * 4 bytes = 512MB per order. For 7 orders = 3.5GB.
    # That's tight but feasible.

    tables = {}
    for order in orders:
        print(f"  Initializing order-{order} count matrix ({num_buckets}x1024)...")
        # Use uint32 counts. With 10B tokens, max count per bucket is ~76K
        # (10B / 128K buckets), which fits in uint32.
        tables[order] = np.zeros((num_buckets, 1024), dtype=np.uint32)

    t0 = time.time()
    total_tokens = 0

    for shard_idx, shard_file in enumerate(shard_files):
        tokens = load_shard(Path(shard_file))
        n = len(tokens)
        total_tokens += n
        print(f"  Shard {shard_idx}/{len(shard_files)}: {n:,} tokens ({total_tokens:,} total, {time.time()-t0:.0f}s)")

        # Vectorized hashing for each order.
        for order in orders:
            if n < order:
                continue
            count_matrix = tables[order]

            # Process in chunks to limit memory for hash computation.
            chunk_size = 1_000_000
            for start in range(max_order, n, chunk_size):
                end = min(start + chunk_size, n)
                positions = np.arange(start, end)

                # Compute hashes for all positions in this chunk.
                h = np.zeros(len(positions), dtype=np.uint64)
                for k in range(order - 1):
                    h = (h * 1000003 + tokens[positions - order + 1 + k].astype(np.uint64)) & 0xFFFFFFFF
                bucket_indices = (h % num_buckets).astype(np.int64)
                next_tokens = tokens[positions].astype(np.int64)

                # Increment counts. Use np.add.at for unbuffered accumulation.
                np.add.at(count_matrix, (bucket_indices, next_tokens), 1)

    elapsed = time.time() - t0
    print(f"Processed {total_tokens:,} tokens in {elapsed:.0f}s ({total_tokens/elapsed/1e6:.1f}M tok/s)")

    # Extract top-K per bucket.
    result = {}
    for order in orders:
        count_matrix = tables[order]
        # argsort each row, take top-K (descending).
        # Use argpartition for efficiency (don't need full sort).
        if top_k < 1024:
            top_indices = np.argpartition(-count_matrix, top_k, axis=1)[:, :top_k]
            # Sort the top-K by count (descending).
            rows = np.arange(num_buckets)[:, None]
            top_counts = count_matrix[rows, top_indices]
            sort_order = np.argsort(-top_counts, axis=1)
            top_indices = top_indices[rows, sort_order]
        else:
            top_indices = np.argsort(-count_matrix, axis=1)[:, :top_k]

        result[order] = torch.from_numpy(top_indices.astype(np.int16))
        # Stats
        nonzero_buckets = (count_matrix.sum(axis=1) > 0).sum()
        print(f"  Order {order}: {nonzero_buckets:,}/{num_buckets:,} buckets active ({100*nonzero_buckets/num_buckets:.1f}%)")

    return result


def main():
    parser = argparse.ArgumentParser(description="Build n-gram top-K prediction tables")
    parser.add_argument("--data-dir", default="./data/datasets/fineweb10B_sp1024",
                        help="Directory containing training shard .bin files")
    parser.add_argument("--shards", type=int, default=0, help="Number of shards to process (0=all)")
    parser.add_argument("--buckets", type=int, default=131072, help="Hash table buckets per order")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K predictions to store")
    parser.add_argument("--min-order", type=int, default=2, help="Minimum n-gram order")
    parser.add_argument("--max-order", type=int, default=7, help="Maximum n-gram order")
    parser.add_argument("--output", default="./data/ngram_tables.pt", help="Output file path")
    args = parser.parse_args()

    orders = list(range(args.min_order, args.max_order + 1))
    tables = build_tables(args.data_dir, args.shards, orders, args.buckets, args.top_k)

    # Save as compact file.
    output = {
        "orders": orders,
        "num_buckets": args.buckets,
        "top_k": args.top_k,
    }
    for order in orders:
        output[f"order_{order}"] = tables[order]  # (num_buckets, top_k) int16

    torch.save(output, args.output)
    file_size = Path(args.output).stat().st_size
    print(f"\nSaved to {args.output}: {file_size:,} bytes ({file_size/1e6:.1f}MB)")
    print(f"Orders: {orders}, Buckets: {args.buckets}, Top-K: {args.top_k}")


if __name__ == "__main__":
    main()

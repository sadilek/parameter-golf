"""Pure n-gram language model baseline — no neural net.

Builds exact count tables from training data, applies Kneser-Ney smoothing
with backoff, and evaluates BPB on the validation set.

Usage: python3 eval_ngram.py [--shards 10] [--max-order 7]
"""
import argparse, glob, math, time
import numpy as np
import sentencepiece as spm
from pathlib import Path


def load_shard(path):
    header = np.fromfile(path, dtype="<i4", count=256)
    n = int(header[2])
    return np.fromfile(path, dtype="<u2", count=n, offset=256 * 4)


def build_counts(shard_dir, num_shards, max_order, vocab_size):
    """Build exact count tables for orders 1..max_order."""
    files = sorted(glob.glob(f"{shard_dir}/fineweb_train_*.bin"))
    if num_shards > 0:
        files = files[:num_shards]

    # Order 1 (unigram): just token counts
    # Order 2 (bigram): exact 1024x1024 matrix
    # Order 3+: use hash tables (exact would be too large)

    unigram = np.zeros(vocab_size, dtype=np.float64)
    bigram = np.zeros((vocab_size, vocab_size), dtype=np.float64)

    # For orders 3+, use hash tables: hash(context) -> array of 1024 counts
    # Use large-ish tables to minimize collisions
    HASH_BUCKETS = 2_000_000  # 2M buckets per order
    higher = {}
    for order in range(3, max_order + 1):
        # Each bucket: 1024 float32 counts = 4KB per bucket
        # 2M * 4KB = 8GB per order — too much!
        # Instead: store sparse counts as hash(context, token) -> count
        # Use two arrays: context_counts[hash(ctx)] and full_counts[hash(ctx, tok)]
        higher[order] = {
            "ctx": np.zeros(HASH_BUCKETS, dtype=np.float64),
            "full": np.zeros(HASH_BUCKETS, dtype=np.float64),
        }

    total_tokens = 0
    t0 = time.time()

    for si, sf in enumerate(files):
        tokens = load_shard(Path(sf))
        n = len(tokens)
        total_tokens += n
        elapsed = time.time() - t0
        print(f"  Shard {si+1}/{len(files)}: {n:,} tok ({total_tokens:,} total, {elapsed:.0f}s, {total_tokens/max(elapsed,1)/1e6:.1f}M tok/s)")

        # Unigram
        for t in range(n):
            unigram[tokens[t]] += 1

        # Bigram (exact)
        chunk = 2_000_000
        for start in range(0, n - 1, chunk):
            end = min(start + chunk, n - 1)
            prev = tokens[start:end].astype(np.int64)
            cur = tokens[start + 1:end + 1].astype(np.int64)
            np.add.at(bigram, (prev, cur), 1)

        # Higher orders (hash-based)
        for order in range(3, max_order + 1):
            if n < order:
                continue
            ctx_counts = higher[order]["ctx"]
            full_counts = higher[order]["full"]

            for start in range(order - 1, n, chunk):
                end = min(start + chunk, n)
                positions = np.arange(start, end)

                # Hash context (order-1 tokens before position)
                h_ctx = np.zeros(len(positions), dtype=np.uint64)
                for k in range(order - 1):
                    h_ctx = (h_ctx * 1000003 + tokens[positions - order + 1 + k].astype(np.uint64)) & 0xFFFFFFFF
                ctx_idx = (h_ctx % HASH_BUCKETS).astype(np.int64)
                np.add.at(ctx_counts, ctx_idx, 1)

                # Hash context + target token
                h_full = (h_ctx * 1000003 + tokens[positions].astype(np.uint64)) & 0xFFFFFFFF
                full_idx = (h_full % HASH_BUCKETS).astype(np.int64)
                np.add.at(full_counts, full_idx, 1)

    elapsed = time.time() - t0
    print(f"Built counts in {elapsed:.0f}s ({total_tokens:,} tokens)")
    return unigram, bigram, higher, total_tokens


def eval_bpb(shard_dir, unigram, bigram, higher, max_order, vocab_size, total_train_tokens):
    """Evaluate BPB on validation data using interpolated Kneser-Ney-style backoff."""
    # Load tokenizer for byte counting
    sp = spm.SentencePieceProcessor("./data/tokenizers/fineweb_1024_bpe.model")

    # Build byte count LUT
    sp_vocab = sp.vocab_size()
    bytes_per_token = np.zeros(max(sp_vocab, vocab_size), dtype=np.float64)
    for tid in range(sp_vocab):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            bytes_per_token[tid] = 0
            continue
        if sp.is_byte(tid):
            bytes_per_token[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            piece = piece[1:]
        bytes_per_token[tid] = len(piece.encode("utf-8"))

    # Precompute unigram distribution
    uni_total = unigram.sum()
    uni_prob = (unigram + 1e-8) / (uni_total + 1e-8 * vocab_size)  # add-epsilon smoothing

    # Precompute bigram distributions (per-row normalized with add-delta smoothing)
    delta = 0.5  # smoothing parameter
    bigram_prob = (bigram + delta) / (bigram.sum(axis=1, keepdims=True) + delta * vocab_size)

    HASH_BUCKETS = 2_000_000

    # Load val data
    val_files = sorted(glob.glob(f"{shard_dir}/fineweb_val_*.bin"))
    val_tokens = np.concatenate([load_shard(Path(f)) for f in val_files])
    max_val = min(500001, len(val_tokens))
    val_tokens = val_tokens[:max_val]
    print(f"Evaluating on {len(val_tokens):,} val tokens")

    # Interpolation weights (tunable)
    # Higher orders get more weight when they have data
    lambdas = {1: 0.05}  # unigram fallback
    remaining = 0.95
    for order in range(2, max_order + 1):
        w = remaining * 0.5  # each order gets half of remaining weight
        lambdas[order] = w
        remaining -= w
    lambdas[1] += remaining  # give remainder to unigram
    print(f"Interpolation weights: {lambdas}")

    total_log_prob = 0.0
    total_bytes = 0.0
    t0 = time.time()

    for pos in range(1, len(val_tokens)):
        target = int(val_tokens[pos])
        tb = bytes_per_token[target]
        # Handle leading space byte
        if pos > 0:
            prev = int(val_tokens[pos - 1])
            piece = sp.id_to_piece(target) if target < sp_vocab else ""
            prev_piece = sp.id_to_piece(prev) if prev < sp_vocab else ""
            if piece.startswith("\u2581") and not (sp.is_control(prev) or sp.is_unknown(prev)):
                tb += 1
        total_bytes += tb

        # Interpolated probability
        prob = lambdas[1] * uni_prob[target]

        # Bigram
        if pos >= 1:
            prev = int(val_tokens[pos - 1])
            prob += lambdas[2] * bigram_prob[prev, target]

        # Higher orders
        for order in range(3, min(max_order + 1, pos + 2)):
            ctx_counts = higher[order]["ctx"]
            full_counts = higher[order]["full"]

            # Hash context
            h_ctx = np.uint64(0)
            for k in range(order - 1):
                h_ctx = (h_ctx * np.uint64(1000003) + np.uint64(val_tokens[pos - order + 1 + k])) & np.uint64(0xFFFFFFFF)
            ctx_idx = int(h_ctx % HASH_BUCKETS)

            # Hash context + target
            h_full = (h_ctx * np.uint64(1000003) + np.uint64(target)) & np.uint64(0xFFFFFFFF)
            full_idx = int(h_full % HASH_BUCKETS)

            ctx_count = ctx_counts[ctx_idx]
            full_count = full_counts[full_idx]

            if ctx_count > 0:
                # Estimate P(target | context) with add-delta smoothing
                p_order = (full_count + delta) / (ctx_count + delta * vocab_size)
                prob += lambdas[order] * p_order
            else:
                # No data for this context — give weight to unigram
                prob += lambdas[order] * uni_prob[target]

        total_log_prob += math.log2(max(prob, 1e-20))

        if pos % 100000 == 0:
            elapsed = time.time() - t0
            running_bpb = -total_log_prob / max(total_bytes, 1)
            print(f"  pos {pos:,}: running BPB={running_bpb:.4f} ({elapsed:.0f}s)")

    bpb = -total_log_prob / total_bytes
    elapsed = time.time() - t0
    print(f"\nFinal BPB: {bpb:.4f} ({elapsed:.0f}s)")
    return bpb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--shards", type=int, default=10, help="Training shards to use")
    parser.add_argument("--max-order", type=int, default=7)
    parser.add_argument("--vocab-size", type=int, default=1024)
    args = parser.parse_args()

    print(f"Building n-gram counts (orders 1-{args.max_order}, {args.shards} shards)...")
    unigram, bigram, higher, total = build_counts(
        args.data_dir, args.shards, args.max_order, args.vocab_size
    )

    # Memory usage
    mem = unigram.nbytes + bigram.nbytes
    for order in range(3, args.max_order + 1):
        mem += higher[order]["ctx"].nbytes + higher[order]["full"].nbytes
    print(f"Total table memory: {mem/1e6:.0f}MB")

    print(f"\nEvaluating on validation set...")
    bpb = eval_bpb(args.data_dir, unigram, bigram, higher, args.max_order, args.vocab_size, total)
    print(f"\n{'='*50}")
    print(f"Pure n-gram model: {bpb:.4f} BPB")
    print(f"  Orders: 1-{args.max_order}")
    print(f"  Training tokens: {total:,}")
    print(f"  Hash buckets (order 3+): 2M")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

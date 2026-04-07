#!/bin/bash
# Step 1: Find max MLP_MULT at dim=384, 4 local layers that fits 16MB
# Step 2: Train the winner
set -e
cd /workspace/parameter-golf

echo "========== Sizing: dim=384, 4 local layers, vary MLP mult =========="
for M in 3 5 7 8 9 10; do
    python3 -c "
import torch, io, sys
sys.path.insert(0, '.')
from train_hierarchical import *

model = HierarchicalGPT(
    vocab_size=1024, embed_dim=32, model_dim=384,
    n_global_layers=6, n_local_layers=4,
    n_heads=6, mlp_mult=$M, window_size=256,
    total_seq=4096, prefix_len=64, pred_len=192)
n_params = sum(p.numel() for p in model.parameters())

sd = model.state_dict()
q_sd, stats = q_sd_int6(sd, 0.9999984)
buf = io.BytesIO()
torch.save(q_sd, buf)
raw = buf.tell()

try:
    import zstandard as zstd
    compressed = len(zstd.ZstdCompressor(level=22).compress(buf.getvalue()))
    method = 'zstd'
except ImportError:
    import lzma
    compressed = len(lzma.compress(buf.getvalue(), preset=9))
    method = 'lzma'

code_size = 45000
total = code_size + compressed
fit = 'OK' if total <= 16_000_000 else 'OVER'
print(f'MLP={$M}: params={n_params:,} artifact={compressed/1e6:.2f}MB total={total/1e6:.2f}MB [{method}] {fit}')
" 2>&1
done

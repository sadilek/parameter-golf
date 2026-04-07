#!/bin/bash
# Step 1: Find max N_LOCAL_LAYERS that fits in 16MB artifact
# Step 2: Train with BS=512, LR=0.03, 2000 steps
set -e
cd /workspace/parameter-golf

echo "========== Sizing: find max local layers =========="
for NL in 8 10 12 14; do
    SIZE=$(python3 -c "
import torch, io, sys
sys.path.insert(0, '.')
from train_hierarchical import *

model = HierarchicalGPT(
    vocab_size=1024, embed_dim=32, model_dim=384,
    n_global_layers=6, n_local_layers=$NL,
    n_heads=6, mlp_mult=3, window_size=256,
    total_seq=4096, prefix_len=64, pred_len=192)
n_params = sum(p.numel() for p in model.parameters())

sd = model.state_dict()
q_sd, stats = q_sd_int6(sd, 0.9999984)
buf = io.BytesIO()
torch.save(q_sd, buf)
raw = buf.tell()

import zlib
compressed = len(zlib.compress(buf.getvalue(), 9))
code_size = 45000  # approximate
total = code_size + compressed
print(f'NL={$NL}: params={n_params:,} raw={raw/1e6:.1f}MB compressed={compressed/1e6:.1f}MB total={total/1e6:.1f}MB', flush=True)
" 2>&1)
    echo "$SIZE"
done

echo ""
echo "========== Training: deep local stack =========="
# Based on sizing above, use the largest NL that fits under 16MB
# With ~2.5M params/layer, 14 local layers → ~48M total → should be ~15MB
PYTHONUNBUFFERED=1 \
RUN_ID=hier_deep_local \
ITERATIONS=2000 \
BATCH_SIZE=512 \
MATRIX_LR=0.03 \
PREFIX_LEN=64 \
PRED_LEN=192 \
N_LOCAL_LAYERS=14 \
MLP_MULT=3 \
VAL_LOSS_EVERY=200 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=50 \
WARMUP_STEPS=0 \
COMPILE_MODE=off \
torchrun --standalone --nproc_per_node=1 train_hierarchical.py 2>&1 | tee logs/hier_deep_local.txt

echo "=== Final ==="
grep "val_bpb\|roundtrip\|artifact" logs/hier_deep_local.txt | tail -5

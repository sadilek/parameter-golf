#!/bin/bash
# Hierarchical model: LR=0.02 with 128+128 (current best split)
# Compare against LR=0.03 which got BPB 2.716 @ 500 steps
set -e

cd /workspace/parameter-golf

echo "=== Experiment: LR=0.02, PREFIX=128, PRED=128, 500 steps ==="
PYTHONUNBUFFERED=1 \
RUN_ID=hier_lr002_p128 \
ITERATIONS=500 \
MATRIX_LR=0.02 \
EMBED_LR=0.6 \
HEAD_LR=0.008 \
PREFIX_LEN=128 \
PRED_LEN=128 \
VAL_LOSS_EVERY=100 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=10 \
torchrun --standalone --nproc_per_node=1 train_hierarchical.py 2>&1 | tee logs/hier_lr002_p128.txt

echo "=== Done ==="
grep "val_bpb" logs/hier_lr002_p128.txt | tail -5

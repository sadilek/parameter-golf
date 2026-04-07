#!/bin/bash
# Hierarchical model: PREFIX_LEN=64 PRED_LEN=192 with MATRIX_LR=0.02
# Compare against 128+128 which got BPB 2.708 @ 500 steps
set -e

cd /workspace/parameter-golf

echo "=== Experiment: LR=0.02, PREFIX=64, PRED=192, 500 steps ==="
PYTHONUNBUFFERED=1 \
RUN_ID=hier_lr002_p64 \
ITERATIONS=500 \
MATRIX_LR=0.02 \
EMBED_LR=0.6 \
HEAD_LR=0.008 \
PREFIX_LEN=64 \
PRED_LEN=192 \
VAL_LOSS_EVERY=100 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=10 \
torchrun --standalone --nproc_per_node=1 train_hierarchical.py 2>&1 | tee logs/hier_lr002_p64.txt

echo "=== Done ==="
grep "val_bpb" logs/hier_lr002_p64.txt | tail -5

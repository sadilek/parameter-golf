#!/bin/bash
# Hierarchical model LR sweep: 0.03 vs 0.04 baseline
# Then with the better LR: PREFIX_LEN=128 PRED_LEN=128 (vs default 192+64)
# Run on 1xA100 SXM, 500 steps each for quick comparison
set -e

cd /workspace/parameter-golf

# Experiment 1: LR=0.03 (new, lower)
echo "=== Experiment 1: MATRIX_LR=0.03, 500 steps ==="
PYTHONUNBUFFERED=1 \
RUN_ID=hier_lr003 \
ITERATIONS=500 \
MATRIX_LR=0.03 \
EMBED_LR=0.6 \
HEAD_LR=0.008 \
VAL_LOSS_EVERY=100 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=10 \
torchrun --standalone --nproc_per_node=1 train_hierarchical.py 2>&1 | tee logs/hier_lr003.txt

# Experiment 2: LR=0.04 (baseline, for fair comparison on same pod/data order)
echo "=== Experiment 2: MATRIX_LR=0.04 (baseline), 500 steps ==="
PYTHONUNBUFFERED=1 \
RUN_ID=hier_lr004 \
ITERATIONS=500 \
MATRIX_LR=0.04 \
EMBED_LR=0.6 \
HEAD_LR=0.008 \
VAL_LOSS_EVERY=100 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=10 \
torchrun --standalone --nproc_per_node=1 train_hierarchical.py 2>&1 | tee logs/hier_lr004.txt

echo "=== LR sweep done. Compare final val BPB from both runs ==="
echo "=== lr003:" && grep "val_bpb" logs/hier_lr003.txt | tail -3
echo "=== lr004:" && grep "val_bpb" logs/hier_lr004.txt | tail -3

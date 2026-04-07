#!/bin/bash
# Test batched streaming: compare batch_size=1 (old) vs batch_size=8 (new)
# Both with grad_accum=8 and 50 steps, measure GPU util + ms/step
set -e
cd /workspace/parameter-golf

echo "=== Test 1: BATCH_SIZE=1 (old behavior), 50 steps ==="
PYTHONUNBUFFERED=1 \
RUN_ID=batch_test_bs1 \
ITERATIONS=50 \
BATCH_SIZE=1 \
MATRIX_LR=0.02 \
PREFIX_LEN=64 \
PRED_LEN=192 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=10 \
torchrun --standalone --nproc_per_node=1 train_hierarchical.py 2>&1 | tee logs/batch_test_bs1.txt

# Quick GPU mem/util snapshot during bs=8
echo "=== Test 2: BATCH_SIZE=8 (batched), 50 steps ==="
PYTHONUNBUFFERED=1 \
RUN_ID=batch_test_bs8 \
ITERATIONS=50 \
BATCH_SIZE=8 \
MATRIX_LR=0.02 \
PREFIX_LEN=64 \
PRED_LEN=192 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=10 \
torchrun --standalone --nproc_per_node=1 train_hierarchical.py 2>&1 | tee logs/batch_test_bs8.txt

echo "=== Compare ms/step ==="
echo "BS=1:" && grep "ms/step" logs/batch_test_bs1.txt | tail -3
echo "BS=8:" && grep "ms/step" logs/batch_test_bs8.txt | tail -3

echo "=== GPU memory ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

#!/bin/bash
# Phase 1: Tighten LR around 0.08 at BS=512, 50 steps
# Phase 2: Run the winner for 500 steps
set -e
cd /workspace/parameter-golf

echo "========== PHASE 1: LR fine-tune =========="
for LR in 0.06 0.07 0.08 0.09 0.10; do
    echo "=== BS=512 LR=$LR ==="
    PYTHONUNBUFFERED=1 \
    RUN_ID=ft_bs512_lr${LR} \
    ITERATIONS=50 \
    BATCH_SIZE=512 \
    MATRIX_LR=$LR \
    PREFIX_LEN=64 \
    PRED_LEN=192 \
    VAL_LOSS_EVERY=0 \
    MAX_WALLCLOCK_SECONDS=0 \
    TRAIN_LOG_EVERY=10 \
    WARMUP_STEPS=0 \
    COMPILE_MODE=off \
    torchrun --standalone --nproc_per_node=1 train_hierarchical.py 2>&1 | tee logs/ft_bs512_lr${LR}.txt

    BPB=$(grep "roundtrip" logs/ft_bs512_lr${LR}.txt | grep -o 'val_bpb:[0-9.]*')
    echo "LR=$LR: $BPB"
    echo ""
done

echo "=== Phase 1 Summary ==="
BEST_LR=""
BEST_BPB="9.9999"
for LR in 0.06 0.07 0.08 0.09 0.10; do
    BPB=$(grep "roundtrip" logs/ft_bs512_lr${LR}.txt | grep -o '[0-9]\.[0-9]*$')
    MS=$(grep "ms/step" logs/ft_bs512_lr${LR}.txt | tail -1 | grep -o '[0-9]*ms/step')
    echo "LR=$LR: val_bpb:$BPB $MS"
    if [ "$(echo "$BPB < $BEST_BPB" | bc)" -eq 1 ]; then
        BEST_BPB=$BPB
        BEST_LR=$LR
    fi
done
echo "BEST: LR=$BEST_LR BPB=$BEST_BPB"

echo ""
echo "========== PHASE 2: Full 500-step run with LR=$BEST_LR =========="
PYTHONUNBUFFERED=1 \
RUN_ID=hier_bs512_best \
ITERATIONS=500 \
BATCH_SIZE=512 \
MATRIX_LR=$BEST_LR \
PREFIX_LEN=64 \
PRED_LEN=192 \
VAL_LOSS_EVERY=100 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=10 \
WARMUP_STEPS=0 \
COMPILE_MODE=off \
torchrun --standalone --nproc_per_node=1 train_hierarchical.py 2>&1 | tee logs/hier_bs512_best.txt

echo "=== Final ==="
grep "val_bpb\|roundtrip" logs/hier_bs512_best.txt | tail -5

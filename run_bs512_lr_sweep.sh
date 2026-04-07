#!/bin/bash
# LR sweep at BS=512, 50 steps, no compile
set -e
cd /workspace/parameter-golf

for LR in 0.08 0.12 0.16; do
    echo "=== BS=512 LR=$LR ==="
    PYTHONUNBUFFERED=1 \
    RUN_ID=sweep_bs512_lr${LR} \
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
    torchrun --standalone --nproc_per_node=1 train_hierarchical.py 2>&1 | tee logs/sweep_bs512_lr${LR}.txt

    BPB=$(grep "val_bpb" logs/sweep_bs512_lr${LR}.txt | tail -1 | grep -o 'val_bpb:[0-9.]*')
    echo "LR=$LR: $BPB"
    echo ""
done

echo "=== Summary ==="
for LR in 0.08 0.12 0.16; do
    BPB=$(grep "roundtrip" logs/sweep_bs512_lr${LR}.txt | grep -o 'val_bpb:[0-9.]*')
    MS=$(grep "ms/step" logs/sweep_bs512_lr${LR}.txt | tail -1 | grep -o '[0-9]*ms/step')
    echo "BS=512 LR=$LR: $BPB $MS"
done

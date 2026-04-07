#!/bin/bash
# Batch size sweep WITHOUT torch.compile to avoid recompilation overhead
# LR=0.04, 50 steps each for better signal
set -e
cd /workspace/parameter-golf

for BS in 64 128 256 512 1024; do
    echo "=== BATCH_SIZE=$BS ==="

    PYTHONUNBUFFERED=1 \
    RUN_ID=bs${BS}_nc \
    ITERATIONS=50 \
    BATCH_SIZE=$BS \
    MATRIX_LR=0.04 \
    PREFIX_LEN=64 \
    PRED_LEN=192 \
    VAL_LOSS_EVERY=0 \
    MAX_WALLCLOCK_SECONDS=0 \
    TRAIN_LOG_EVERY=10 \
    WARMUP_STEPS=0 \
    COMPILE_MODE=off \
    torchrun --standalone --nproc_per_node=1 train_hierarchical.py 2>&1 &
    PID=$!

    # Sample GPU during steady state
    sleep 30
    SUM_UTIL=0; SUM_MEM=0; N=0
    for i in 1 2 3 4 5 6 7 8 9 10; do
        LINE=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)
        UTIL=$(echo "$LINE" | cut -d',' -f1 | tr -d ' ')
        MEM=$(echo "$LINE" | cut -d',' -f2 | tr -d ' ')
        SUM_UTIL=$((SUM_UTIL + UTIL))
        SUM_MEM=$((SUM_MEM + MEM))
        N=$((N + 1))
        sleep 1
    done

    wait $PID 2>/dev/null
    STATUS=$?

    if [ $STATUS -ne 0 ]; then
        echo "BS=$BS: FAILED (exit $STATUS)"
        echo ""
        continue
    fi

    AVG_UTIL=$((SUM_UTIL / N))
    AVG_MEM=$((SUM_MEM / N))
    MS=$(grep "ms/step" logs/cuda/bs${BS}_nc.txt 2>/dev/null | tail -1 | grep -o '[0-9]*ms/step' || echo "?ms/step")
    BPB=$(grep "val_bpb" logs/cuda/bs${BS}_nc.txt 2>/dev/null | tail -1 | grep -o 'val_bpb:[0-9.]*' || echo "?")

    echo "BS=$BS: gpu=${AVG_UTIL}% mem=${AVG_MEM}MB $MS $BPB"
    echo ""
done

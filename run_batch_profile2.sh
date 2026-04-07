#!/bin/bash
set -e
cd /workspace/parameter-golf

for BS in 8 32 64 128; do
    echo "=== BATCH_SIZE=$BS ==="

    PYTHONUNBUFFERED=1 \
    RUN_ID=prof_bs${BS} \
    ITERATIONS=30 \
    BATCH_SIZE=$BS \
    MATRIX_LR=0.02 \
    PREFIX_LEN=64 \
    PRED_LEN=192 \
    VAL_LOSS_EVERY=0 \
    MAX_WALLCLOCK_SECONDS=0 \
    TRAIN_LOG_EVERY=10 \
    WARMUP_STEPS=3 \
    torchrun --standalone --nproc_per_node=1 train_hierarchical.py 2>&1 &
    PID=$!

    # Wait for warmup + a few training steps
    sleep 20

    # Sample GPU 5 times
    SUM_UTIL=0
    SUM_MEM=0
    N=0
    for i in 1 2 3 4 5; do
        LINE=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)
        UTIL=$(echo "$LINE" | cut -d',' -f1 | tr -d ' ')
        MEM=$(echo "$LINE" | cut -d',' -f2 | tr -d ' ')
        SUM_UTIL=$((SUM_UTIL + UTIL))
        SUM_MEM=$((SUM_MEM + MEM))
        N=$((N + 1))
        sleep 1
    done

    wait $PID 2>/dev/null || true

    AVG_UTIL=$((SUM_UTIL / N))
    AVG_MEM=$((SUM_MEM / N))

    # Get ms/step from training output
    MS=$(grep "ms/step" logs/cuda/prof_bs${BS}.txt 2>/dev/null | tail -1 | grep -o '[0-9]*ms/step' || echo "?ms/step")

    echo "BS=$BS: gpu_util=${AVG_UTIL}% mem=${AVG_MEM}MB $MS"
    echo ""
done

echo "=== Total GPU memory ==="
nvidia-smi --query-gpu=memory.total --format=csv,noheader

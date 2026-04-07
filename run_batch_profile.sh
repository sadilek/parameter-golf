#!/bin/bash
# Profile GPU utilization at different batch sizes
set -e
cd /workspace/parameter-golf

for BS in 8 16 32 64; do
    echo "=== BATCH_SIZE=$BS ==="

    # Start training in background
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
    WARMUP_STEPS=5 \
    torchrun --standalone --nproc_per_node=1 train_hierarchical.py &
    PID=$!

    # Wait for warmup to finish, then sample GPU stats
    sleep 12
    echo "gpu_util,mem_used_mb" > /tmp/prof_bs${BS}.csv
    for i in $(seq 1 20); do
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits >> /tmp/prof_bs${BS}.csv
        sleep 0.5
    done

    wait $PID 2>/dev/null

    echo "BS=$BS results:"
    awk -F', ' 'NR>1 {gu+=$1; mu+=$2; n++; if($2>mmax)mmax=$2} END {printf "  gpu_util=%.0f%% mem_avg=%.0fMB mem_peak=%.0fMB (n=%d)\n", gu/n, mu/n, mmax, n}' /tmp/prof_bs${BS}.csv
    grep "ms/step" logs/cuda/prof_bs${BS}.txt 2>/dev/null | tail -1 || true
    echo ""
done

#!/bin/bash
# Profile GPU utilization during hierarchical training
# Runs 50 steps and samples nvidia-smi every 0.5s
set -e
cd /workspace/parameter-golf

# Start training in background
PYTHONUNBUFFERED=1 \
RUN_ID=hier_profile \
ITERATIONS=50 \
MATRIX_LR=0.02 \
PREFIX_LEN=64 \
PRED_LEN=192 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=10 \
torchrun --standalone --nproc_per_node=1 train_hierarchical.py &
TRAIN_PID=$!

# Sample GPU utilization
echo "timestamp,gpu_util,mem_used_mb,mem_total_mb,power_w" > logs/gpu_profile.csv
while kill -0 $TRAIN_PID 2>/dev/null; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits >> logs/gpu_profile.csv
    sleep 0.5
done

wait $TRAIN_PID
echo "=== GPU Profile Summary ==="
echo "Samples:"
wc -l < logs/gpu_profile.csv
echo "GPU utilization stats (%):"
awk -F', ' 'NR>1 {sum+=$2; n++; if($2>max)max=$2} END {printf "avg=%.1f max=%.0f n=%d\n", sum/n, max, n}' logs/gpu_profile.csv
echo "Memory usage stats (MB):"
awk -F', ' 'NR>1 {sum+=$3; n++; if($3>max)max=$3} END {printf "avg=%.0f max=%.0f total=%s\n", sum/n, max, $4}' logs/gpu_profile.csv

#!/bin/bash
# Overnight experiment suite: SOTA reproduction + 10 ideas
# All on 1×A100 SXM, 1000 steps each, no compile for speed
# Results logged to individual files for comparison
set -e
cd /workspace/parameter-golf

mkdir -p logs/overnight

# Common SOTA-like config (matching competition leader as closely as possible)
# 11L, MLP×3, XSA-all, seq=2048, warmdown, Muon
COMMON="PYTHONUNBUFFERED=1 \
ITERATIONS=1000 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3 \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=524288 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=50 \
WARMUP_STEPS=20 \
QUANT_MODE=int6 \
COMPILE_MODE=default \
XSA_LAYERS=11 \
ACTIVATION=leaky_relu2 \
N_CHANNELS=1"

run_experiment() {
    local name=$1
    local extra_env=$2
    echo ""
    echo "================================================================"
    echo "  EXPERIMENT: $name"
    echo "================================================================"
    local cmd="cd /workspace/parameter-golf && $extra_env $COMMON RUN_ID=ovn_${name} torchrun --standalone --nproc_per_node=1 train_combined.py"
    eval "$cmd" 2>&1 | tee "logs/overnight/${name}.txt"
    # Extract key metrics
    local final_bpb=$(grep "final.*val_bpb" "logs/overnight/${name}.txt" | tail -1 | grep -o 'val_bpb:[0-9.]*' | cut -d: -f2)
    local rt_bpb=$(grep "roundtrip.*val_bpb" "logs/overnight/${name}.txt" | tail -1 | grep -o 'val_bpb:[0-9.]*' | cut -d: -f2)
    local artifact=$(grep "artifact:" "logs/overnight/${name}.txt" | tail -1 | grep -o '[0-9.]*MB' | head -1)
    local ms=$(grep "ms/step\|step_avg" "logs/overnight/${name}.txt" | tail -1 | grep -o '[0-9.]*ms' | head -1)
    echo ">>> $name: BPB=$final_bpb RT=$rt_bpb artifact=$artifact speed=$ms"
    echo "$name,$final_bpb,$rt_bpb,$artifact,$ms" >> logs/overnight/summary.csv
}

echo "name,final_bpb,roundtrip_bpb,artifact,ms_per_step" > logs/overnight/summary.csv

# ============================================================
# 0. SOTA BASELINE: warmdown schedule
# ============================================================
run_experiment "baseline" "LR_SCHEDULE=warmdown WARMDOWN_FRACTION=0.2"

# ============================================================
# 1. +SSE CALIBRATION
# ============================================================
run_experiment "sse" "LR_SCHEDULE=warmdown WARMDOWN_FRACTION=0.2 SSE_ENABLED=1 SSE_CLUSTERS=32 SSE_ENTROPY_BINS=16"

# ============================================================
# 2. +LLOYD-MAX QUANTIZATION (via STE type)
# ============================================================
run_experiment "lloyd_max" "LR_SCHEDULE=warmdown WARMDOWN_FRACTION=0.2 STE_TYPE=lloyd_max"

# ============================================================
# 3. +FEDERATED AVERAGING (simulated with 2 branches on 1 GPU)
# Cannot truly test on 1 GPU — federated needs multi-GPU.
# Instead test with smaller batch (simulates more optimizer steps = more diverse exploration)
# ============================================================
run_experiment "small_batch" "LR_SCHEDULE=warmdown WARMDOWN_FRACTION=0.2 TRAIN_BATCH_TOKENS=262144"

# ============================================================
# 4. +1CYCLE LR
# ============================================================
run_experiment "1cycle" "LR_SCHEDULE=1cycle ONECYCLE_PEAK_FRAC=0.3 ONECYCLE_MIN_DIV=4"

# ============================================================
# 5. +BIGRAM HASH (SOTA uses this)
# ============================================================
run_experiment "bigram_hash" "LR_SCHEDULE=warmdown WARMDOWN_FRACTION=0.2 BIGRAM_HASH=1"

# ============================================================
# 6. WIDER MLP (MLP×4 instead of ×3, fewer layers to compensate)
# ============================================================
run_experiment "mlp4_9L" "LR_SCHEDULE=warmdown WARMDOWN_FRACTION=0.2 MLP_MULT=4 NUM_LAYERS=9"

# ============================================================
# 7. DEEPER (13L instead of 11L, same MLP×3)
# ============================================================
run_experiment "deeper_13L" "LR_SCHEDULE=warmdown WARMDOWN_FRACTION=0.2 NUM_LAYERS=13"

# ============================================================
# 8. LONGER SEQ (4096 instead of 2048)
# ============================================================
run_experiment "seq4096" "LR_SCHEDULE=warmdown WARMDOWN_FRACTION=0.2 TRAIN_SEQ_LEN=4096 TRAIN_BATCH_TOKENS=524288"

# ============================================================
# 9. EMA (despite previous bad results, test with int6)
# ============================================================
run_experiment "ema" "LR_SCHEDULE=warmdown WARMDOWN_FRACTION=0.2 EMA_ENABLED=1 EMA_DECAY=0.999"

# ============================================================
# 10. SSE + BIGRAM_HASH + 1CYCLE (best combo attempt)
# ============================================================
run_experiment "combo" "LR_SCHEDULE=1cycle ONECYCLE_PEAK_FRAC=0.3 ONECYCLE_MIN_DIV=4 SSE_ENABLED=1 SSE_CLUSTERS=32 SSE_ENTROPY_BINS=16 BIGRAM_HASH=1"

echo ""
echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "================================================================"
echo ""
cat logs/overnight/summary.csv | column -t -s,

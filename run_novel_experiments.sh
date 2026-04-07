#!/bin/bash
# Novel ideas experiment suite
# Tests: MoE, Foveated Attention, Adaptive Depth, Distillation, Structured Embed, Progressive Merge
# All on 1×A100 SXM, 1000 steps each, compiled
set -e
cd /workspace/parameter-golf

mkdir -p logs/novel

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
XSA_LAYERS=11 \
ACTIVATION=leaky_relu2 \
N_CHANNELS=1 \
LR_SCHEDULE=warmdown \
WARMDOWN_FRACTION=0.2"

echo "name,final_bpb,roundtrip_bpb,artifact,ms_per_step" > logs/novel/summary.csv

run_experiment() {
    local name=$1
    local extra_env=$2
    echo ""
    echo "================================================================"
    echo "  EXPERIMENT: $name"
    echo "================================================================"
    eval "$extra_env $COMMON RUN_ID=nov_${name} COMPILE_MODE=default torchrun --standalone --nproc_per_node=1 train_combined.py" 2>&1 | tee "logs/novel/${name}.txt"
    local final_bpb=$(grep "final.*val_bpb" "logs/novel/${name}.txt" | tail -1 | grep -o 'val_bpb:[0-9.]*' | cut -d: -f2)
    local rt_bpb=$(grep "roundtrip.*val_bpb" "logs/novel/${name}.txt" | tail -1 | grep -o 'val_bpb:[0-9.]*' | cut -d: -f2)
    local artifact=$(grep "artifact:" "logs/novel/${name}.txt" | tail -1 | grep -o '[0-9.]*MB' | head -1)
    echo ">>> $name: BPB=$final_bpb RT=$rt_bpb artifact=$artifact"
    echo "$name,$final_bpb,$rt_bpb,$artifact" >> logs/novel/summary.csv
}

# Baseline (same as overnight, for comparison)
run_experiment "baseline" ""

# 5. Foveated attention: top 4 layers use windowed attention (window=256)
run_experiment "foveated_4L" "FOVEATED_LAYERS=4 FOVEATED_WINDOW=256 COMPILE_MODE=off"

# 6. Adaptive depth: per-token layer skipping
run_experiment "adaptive_depth" "ADAPTIVE_DEPTH=1 COMPILE_MODE=off"

# 7. MoE: 4 experts, top-1 routing (same param budget: fewer layers to compensate)
# 4 experts × MLP×3 per layer = 4× MLP cost. Use 5L instead of 11L to match params.
run_experiment "moe_4exp_5L" "MOE_EXPERTS=4 MOE_TOPK=1 NUM_LAYERS=5 COMPILE_MODE=off"

# 7b. MoE: 2 experts, 11L (total MLP params = 2× but only 1 active)
run_experiment "moe_2exp_11L" "MOE_EXPERTS=2 MOE_TOPK=1 COMPILE_MODE=off"

# 8. Train large → distill: train 16L for 500 steps, then use as teacher
# Phase 1: large model
echo "================================================================"
echo "  EXPERIMENT: distill (Phase 1: 16L teacher, 500 steps)"
echo "================================================================"
PYTHONUNBUFFERED=1 ITERATIONS=500 NUM_LAYERS=16 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
MLP_MULT=3 TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=50 WARMUP_STEPS=20 QUANT_MODE=int6 XSA_LAYERS=16 \
ACTIVATION=leaky_relu2 N_CHANNELS=1 LR_SCHEDULE=warmdown WARMDOWN_FRACTION=0.2 \
RUN_ID=nov_teacher COMPILE_MODE=default SAVE_FLOAT=1 \
torchrun --standalone --nproc_per_node=1 train_combined.py 2>&1 | tee logs/novel/teacher.txt
TEACHER_BPB=$(grep "final.*val_bpb" logs/novel/teacher.txt | tail -1 | grep -o 'val_bpb:[0-9.]*' | cut -d: -f2)
echo ">>> teacher: BPB=$TEACHER_BPB"

# Phase 2: student with distillation (needs DISTILL_FROM implementation)
# For now, just record teacher result — distillation needs training loop changes
echo "distill_teacher,$TEACHER_BPB,,$," >> logs/novel/summary.csv

# 9. Structured embeddings: use smaller embed_dim with projection
# This approximates structured embeddings by using a bottleneck embedding
run_experiment "embed_bottleneck" "EMBED_DIM=128"

# 10. Progressive merge: simulated with federated averaging every 200 steps
# On 1 GPU this means periodic weight perturbation + averaging
run_experiment "fedavg_200" "FEDAVG_EVERY=200"

echo ""
echo "================================================================"
echo "  ALL NOVEL EXPERIMENTS COMPLETE"
echo "================================================================"
echo ""
cat logs/novel/summary.csv | column -t -s,

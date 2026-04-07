#!/bin/bash
# Two full-length runs on 1×H100:
# 1. SOTA reproduction: warmdown only (matching competition leader config)
# 2. Our combo: warmdown + SSE (no 1cycle, no BigramHash)
# Both: 4000 steps, compiled, seq=2048, 11L MLP×3
set -e
cd /workspace/parameter-golf

mkdir -p logs/fulllength

COMMON="PYTHONUNBUFFERED=1 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3 \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=524288 \
VAL_LOSS_EVERY=500 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=100 \
WARMUP_STEPS=20 \
QUANT_MODE=int6 \
COMPILE_MODE=default \
XSA_LAYERS=11 \
ACTIVATION=leaky_relu2 \
N_CHANNELS=1 \
LR_SCHEDULE=warmdown \
WARMDOWN_FRACTION=0.2 \
ITERATIONS=4000"

echo "================================================================"
echo "  RUN 1: SOTA BASELINE (warmdown, 4000 steps)"
echo "================================================================"
eval "$COMMON RUN_ID=full_sota torchrun --standalone --nproc_per_node=1 train_combined.py" 2>&1 | tee logs/fulllength/sota.txt

SOTA_BPB=$(grep "final.*val_bpb" logs/fulllength/sota.txt | tail -1 | grep -o 'val_bpb:[0-9.]*' | cut -d: -f2)
SOTA_RT=$(grep "roundtrip.*val_bpb\|final_roundtrip.*val_bpb" logs/fulllength/sota.txt | tail -1 | grep -o 'val_bpb:[0-9.]*' | cut -d: -f2)
echo ">>> SOTA: BPB=$SOTA_BPB RT=$SOTA_RT"

echo ""
echo "================================================================"
echo "  RUN 2: WARMDOWN + SSE (4000 steps)"
echo "================================================================"
eval "$COMMON SSE_ENABLED=1 SSE_CLUSTERS=32 SSE_ENTROPY_BINS=16 RUN_ID=full_sse torchrun --standalone --nproc_per_node=1 train_combined.py" 2>&1 | tee logs/fulllength/sse.txt

SSE_BPB=$(grep "final.*val_bpb" logs/fulllength/sse.txt | tail -1 | grep -o 'val_bpb:[0-9.]*' | cut -d: -f2)
SSE_RT=$(grep "roundtrip.*val_bpb\|final_roundtrip.*val_bpb" logs/fulllength/sse.txt | tail -1 | grep -o 'val_bpb:[0-9.]*' | cut -d: -f2)
echo ">>> SSE: BPB=$SSE_BPB RT=$SSE_RT"

echo ""
echo "================================================================"
echo "  COMPARISON"
echo "================================================================"
echo "SOTA baseline: BPB=$SOTA_BPB  RT=$SOTA_RT"
echo "Warmdown+SSE:  BPB=$SSE_BPB  RT=$SSE_RT"

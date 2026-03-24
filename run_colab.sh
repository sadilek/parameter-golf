#!/bin/bash
# Run this in a Google Colab cell to set up and train.
# Usage in Colab:
#   !git clone https://github.com/YOUR_USERNAME/parameter-golf
#   %cd parameter-golf
#   !bash run_colab.sh

set -e

echo "=== Installing dependencies ==="
pip install -q torch sentencepiece

echo "=== Downloading data (1 shard for testing) ==="
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

echo "=== Checking GPU ==="
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"

echo "=== Training ternary GRU (single GPU) ==="
RUN_ID=colab_ternary \
ITERATIONS=500 \
TRAIN_BATCH_TOKENS=65536 \
TRAIN_SEQ_LEN=1024 \
VAL_LOSS_EVERY=100 \
VAL_BATCH_SIZE=65536 \
WARMUP_STEPS=10 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=10 \
NUM_LAYERS=8 \
MODEL_DIM=1024 \
LR=0.003 \
torchrun --standalone --nproc_per_node=1 train_ternary.py

echo "=== Done! Check logs/colab_ternary.txt for results ==="

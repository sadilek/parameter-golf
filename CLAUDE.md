# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Parameter Golf** is an OpenAI challenge to train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s. Models are evaluated by compression on the FineWeb validation set using tokenizer-agnostic bits-per-byte (BPB). The challenge runs March 18 – April 30, 2026.

## Key Commands

### Download dataset
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024              # full (80 shards)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1  # minimal smoke test
```

### Train locally (Apple Silicon / MLX)
```bash
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
```

### Train on GPU (PyTorch / CUDA)
```bash
# Single GPU
RUN_ID=baseline torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8xH100 (official evaluation config)
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Useful env overrides
- `MAX_WALLCLOCK_SECONDS=0` — remove the 10-minute wallclock cap
- `VAL_LOSS_EVERY=200` — periodic validation during training
- `ITERATIONS=200` — short run for iteration
- All hyperparameters are configurable via env vars (see `Hyperparameters` class at top of `train_gpt.py`)

## Architecture

### Training scripts
- **`train_gpt.py`** — Main PyTorch training script with distributed training (torchrun/NCCL). This is both the baseline and the template for submissions. Hard limit: 1500 lines.
- **`train_gpt_mlx.py`** — Apple MLX port for local M-series Mac development. Same model architecture, single-GPU only.

### Model
GPT-style transformer with:
- Encoder-decoder hybrid with U-Net-style skip connections between layers
- Grouped Query Attention (GQA) with RoPE and logit softcap
- ReLU² MLP (`relu(fc(x))²`)
- Learnable per-block residual scales and mixing parameters
- Tied embeddings (default) for parameter efficiency

### Optimization
- **Muon optimizer** for matrix parameters (Newton-Schulz orthogonalization)
- **Adam** for embeddings, head, scalars/vectors — each with separate LR
- Warmup + cosine warmdown schedule
- Mixed precision (bfloat16)

### Evaluation
- **val_bpb** (bits-per-byte) is the primary metric — tokenizer-agnostic compression
- Validation always runs on the full `fineweb_val_*` split (fixed first-50k documents)
- Int8 quantization + zlib compression for artifact size check (must be ≤ 16,000,000 bytes)

### Data pipeline
- Binary shard format (magic=20240520, uint16 tokens) in `data/datasets/fineweb10B_sp1024/`
- SentencePiece tokenizer models in `data/tokenizers/`
- `data/cached_challenge_fineweb.py` handles downloading from HuggingFace

### Submissions
- Live in `records/track_10min_16mb/` (official) or `records/track_non_record_16mb/` (experimental)
- Each submission folder contains: `README.md`, `submission.json`, `train_gpt.py`, and training logs
- New SOTA must beat existing by ≥0.005 nats at p < 0.01 (typically 3+ seeds)
- Artifact = code bytes + compressed model bytes, must be ≤ 16MB (decimal)

## Key Constraints

- No external downloads or network calls during evaluation
- Evaluation must also complete within 10 minutes on 8xH100s (separate from training time)
- Cannot access validation data during training
- Submission scripts must be self-contained and run from within their records folder

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
- **`train_gpt.py`** — Main PyTorch training script with distributed training (torchrun/NCCL). This is both the baseline and the template for submissions.
- **`train_combined.py`** — Combined ternary submission: forks the ternary record (1.157 BPB) with CDMA superposition training, masked diffusion, LeakyReLU(0.5)², EMA, and score-first TTT.
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

## Experiment Workflow Rules

- **Always copy results back to local machine.** After any GPU experiment, download float weights, logs, and artifacts to `saved_weights/` before stopping the pod. Pod volumes can be lost if pods are deleted.
- **Stop pods immediately** after experiments finish. Never leave pods idle.
- **Use shell script files** for nohup commands (not inline bash -c) to avoid env var escaping issues.
- **Use PYTHONUNBUFFERED=1** or `python -u` when redirecting output via nohup.
- **Use grad_accum>=8** for eval_val on 13L+ models — smaller batches cause silent OOM and wrong BPB.
- **Track all experiments** in `logbook.md` — keep it up-to-date with hypotheses, experiment ideas, implementation caveats (non-obvious stuff), results, and analyses.
- **Don't re-run baselines needlessly.** If we already have numbers from `logbook.md` or `hypotheses.md`, use them instead of burning GPU time.
- **Be cost-conscious with GPU time.** Prefer 1×A100 ($1.39/hr) for screening. Use H100 only when speed matters. Don't run long eval jobs without compile on expensive GPUs.
- **Artifact size depends on training duration.** Random weights compress ~2.5× better than trained weights. Always verify artifact size with TRAINED weights, not random init.
- **6-bit packing hurts with zstd** — zstd already handles the wasted bits in int8. Packing removes the patterns zstd exploits, making artifacts ~33% larger.
- **torch.compile segfaults on some pods** (PyTorch 2.4.1 + A100 PCIe pod `kj0v7b0djsxzsp`). Guard compile calls with COMPILE_MODE=off support.
- **1cycle LR destabilizes at long training.** Works great at 500-1000 steps, but diverges at 2000-5000 steps. Use warmdown schedule for long runs.
- **LOAD_WEIGHTS resumes model but NOT the LR schedule.** Don't resume training with a fresh 1cycle — it will ramp LR up on an already-converged model and diverge.

## SOTA Reference (PR #1019, 1.115 BPB)

The competition leader uses `train_gpt_sota.py` (downloaded from PR #1019). Key config:
- 11L, 512d, 8H/4KV GQA, MLP×3 LeakyReLU(0.5)²
- XSA all 11 layers, Partial RoPE 16/64, LN Scale 1/√(i+1)
- BigramHash 3072×112, SmearGate, VE128 (layers 9,10)
- U-Net skips, orthogonal init, logit softcap 30
- Muon WD=0.04, warmdown=4000, seq=2048, batch=786K
- EMA(0.997) + SWA(every 50), Late QAT at scale<0.15
- Full Hessian GPTQ (AR self-gen calibration), LZMA preset=9
- Sliding window eval stride=64 (gives -0.025 BPB free)
- ~7000 steps at 86ms/step on 8×H100, 27.1M params, 15.9MB artifact

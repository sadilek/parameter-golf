# Parameter Golf — Comprehensive Experiment Tracker

## Status: ✅ Works — ❌ Doesn't work — 🔬 Testing — 💡 Untested — 🤔 Speculative
## Novelty: **Novel** = untried in competition — **Known** = tried by others

---

## Current Best Results

| Config | BPB (pre-quant) | BPB (roundtrip) | Artifact | Steps | GPU | Quant |
|--------|----------------|-----------------|----------|-------|-----|-------|
| **13L DDP 10min** | **1.1708** | **1.2181** | 16.1MB | 4170 | 8×H100 | Lloyd-Max int4 |
| 13L DDP uncapped | 1.1870 | 1.2267 | 15.8MB | 3000 | 8×H100 | Lloyd-Max int4 |
| 13L resumed 4000 | 1.1999 | 1.2440 | 16.0MB | 4000 | 1×A100 | Lloyd-Max int4 |
| 13L 2000 steps | 1.2247 | 1.2593 | 15.9MB | 2000 | 1×A100 | Lloyd-Max int4 |
| 13L uniform int6 | 1.2247 | 1.2302 | 19.3MB OVER | 2000 | 1×A100 | Uniform int6 |
| Leaderboard SOTA | — | **1.1147** | 15.9MB | ~7000 | 8×H100 | GPTQ int6 |

---

## Architecture Experiments

### XSA (eXcluding Self-Attention) ✅ +0.008→+0.042 BPB | Known
- XSA-4 (top 4 layers): +0.008 BPB at 500 steps
- XSA-all (all 13 layers): contributes to the +0.042 total gain from seq2048+XSA-all combined
- **Surprise**: Competition SOTA uses XSA-all, and it stacks well with other techniques
- **Worth exploring**: Different XSA intensity per layer (learnable gate)

### Cascaded SSE Calibration ✅ +0.013 BPB | **Novel**
- 3 stages: per-token logit scale, prev-token-cluster bias, entropy-dependent temperature
- Corrects systematic biases from int6 quantization — reduces roundtrip gap
- Only ~33K extra params (negligible artifact cost)
- Cluster bias (stage 2) doesn't help when input_ids not forwarded — stage 1 and 3 do the work
- **Listed as untried in competition issue #140** — genuinely novel finding

### Universal Transformer ✅ 1.37 BPB @ 5MB | Known (PR #1110 got 1.22)
- 3 unique blocks × 4 iterations = 12 effective layers, only 3 blocks stored
- Best artifact efficiency: 5MB for competitive BPB
- At 2000 steps: 1.3664 BPB @ 5.15MB (artifact grows slightly with training)
- **Key insight**: UT artifact stays small regardless of training duration — solves the "artifact grows with training" problem
- **Surprise**: Artifact DID grow from 3.7MB (500 steps) to 5.15MB (2000 steps) — not as stable as expected

### Low-Rank Training (W = A·B) ✅ Massive compression | **Novel**
- r128: 1.4741 BPB @ 2.26MB (3.4× compression vs full-rank UT)
- r64: 1.5575 BPB @ 1.38MB (6.4× compression)
- r32: 1.6616 BPB @ ~1.0MB (but too weak)
- **SVD analysis of trained weights**: attn.proj has effective rank ~50-100, but QKV/MLP are nearly full-rank (~300+/512). Training from low-rank forces the model to adapt.
- **Worth exploring**: Mixed rank per layer (low rank for proj, full for QKV)

### Sequence Length ❌ (shorter) / ✅ (longer)
- seq=64/128/256: Same speed as seq=1024, worse BPB. Attention is <18% of compute for 512d models.
- seq=2048: **+0.042 BPP combined with XSA-all** (from 1.3996 → 1.3575). SOTA uses seq=2048.
- batch=786K (SOTA config): works, contributes to longer effective training
- **Surprise**: Shorter sequences don't save compute at all! MLPs dominate, not attention.
- **Insight**: For 512d models, attention is ~18% of per-layer FLOPs at seq=1024, <1% at seq=64

### Other Architecture Attempts
- **Causal conv MLP** (no attention) ❌: 1.71 BPB — too weak for ensemble
- **Token shift (RWKV)** ❌: 1.72 BPB — too weak
- **LeakyReLU(0.5)²** ❌ at 500 steps: -0.018 BPB vs SwiGLU. SOTA uses it but likely needs different LR tuning
- **Progressive layer growing** ❌: -0.034 BPB, 22% faster but layers can't catch up
- **Stochastic depth recurrence** ❌: -0.059 BPB, slower and worse
- **CDMA superposition** ❌: Was NEVER working in main training loop (bug: _force_causal only set in warmup). Diverges when fixed. 8-channel results were actually single-channel all along.

---

## Training & Optimization

### 1cycle LR ✅ +0.043 BPB | **Novel** (nobody in competition uses it)
- peak=0.3, min_div=4 (start at 1/4 peak, ramp to peak at 30%, cosine anneal)
- **Single biggest training improvement** we found
- Multicycle (3 cycles) ❌: Warm restarts lose progress at 500 steps. **Worth retesting at 5000+ steps** where each cycle gets more time.
- Peak=0.15 ❌: -0.007 vs default. Longer ramp helps.
- min_div=2 ❌: -0.026 vs default. Starting LR too high.

### Federated Averaging (Local SGD) ✅ Proven concept | **Novel**
- Each GPU trains independently on different data, periodic weight averaging
- **Tested on 1×A100 with multi-branch**: merged model (1.618 BPB) nearly matches ensemble (1.615)
- WITHOUT periodic averaging: merge catastrophic (7.22 BPB) — models diverge to different basins
- WITH averaging every 50 steps: merge works (1.618 vs 1.615 ensemble)
- **Implemented in train_combined.py**: `FEDAVG_EVERY=50` env var
- On 8×H100: replaces gradient all-reduce with periodic weight all-reduce
- **Worth exploring**: Optimal sync frequency (every 20? 100? 200 steps?)

### Multi-Branch Parallel Training ✅ 1.9× speedup (uncompiled) | **Novel**
- 2 independent model branches in one nn.Module, batch split
- 1.9× faster than sequential WITHOUT compile
- BUT: torch.compile doesn't work with custom forward → compiled single model is still faster
- GPU at 100% compute utilization with compile → no room for parallel branches
- **Key insight**: Without compile, GPU is ~50% idle (room for 2 branches). With compile, GPU is 100% saturated.
- Memory is the constraint for full-batch parallel (OOM with 2×7L + 1M token batch on 80GB)

### EMA ❌ Catastrophic for ALL quantization modes
- Ternary + EMA: 0.36+ BPB roundtrip gap
- Int6 + EMA: 0.71 BPB roundtrip gap (1.2752 → 1.9883)
- **Never use EMA with quantization**

### Batch Size 💡 Worth testing
- Competition SOTA: 786K tokens/step (vs baseline 524K)
- PR #73: -0.016 BPB from quarter batch (131K) on 1×GPU — more optimizer steps helps
- We haven't tested smaller batch on our setup

---

## Ensemble & Merging

### Seed Diversity Ensemble ✅ Best approach | **Novel**
- 2 models: +0.032 BPB (1.4414 → 1.4057)
- 4 models: +0.051 BPB (1.4399 → 1.3880)
- Gain increases with more models but individual quality matters more
- **Entropy-adaptive mixing**: +0.008 over uniform (confident model gets more weight)
- Same-architecture different-seed beats diverse-architecture ensemble

### Diverse Architecture Ensemble ❌
- SwiGLU + LeakyReLU²: worse than seed-only (weaker arch drags down)
- Transformer + CausalConv + TokenShift: much worse (MLP models too weak at 1.71 BPB)
- Asymmetric (10L+3L): +0.005 only — weak model hurts
- **Lesson**: Diversity from different seeds > diversity from different architectures

### Model Merging (Weight Soup) ❌ with different seeds, ✅ with periodic averaging
- Different seeds: catastrophic (4.0-9.0 BPB) — different loss basins
- Same seed, no averaging: catastrophic (7.22 BPB) — diverge in 500 steps
- **Same seed + avg every 50 steps: WORKS** (1.618 merged vs 1.615 ensemble)
- This is Federated Averaging — key novel finding
- **Nobody in competition tried cross-run model merging** (EMA/SWA are single-run only)

### Boosting Ensemble ❌ Too aggressive
- Train model B on model A's errors (upweight hard tokens [0.5, 3.0])
- Model B individually 0.11 BPB worse — over-specialized
- Ensemble: -0.029 vs non-boosted
- **Fix needed**: Softer weights [0.8, 1.5], or boost only last 20% of training

---

## Compression & Quantization

### Int6 + zstd-22 ✅ Standard
- Artifact grows with training steps: 500 steps ~10MB, 1000 ~14MB, 1500 ~17MB, 2000 ~20MB for 13L
- **Key constraint**: Longer training = less compressible weights (higher entropy)

### Post-training Magnitude Pruning ❌ Devastating
- 20% prune: +0.36 BPB loss. 30%: +1.27. 40%: +2.05. Near-random at 50%.
- Random zeros don't compress well with zstd (not contiguous)
- 40% pruning needed to fit budget (19.77→14.57MB) but destroys model quality
- **Root cause**: Model is already tiny (32M params) — no redundancy to prune
- Structured pruning (dropping dimensions) would be cleaner but requires re-training
- **Better approach**: Train smaller model from start, or add L1/sparsity regularization during training

### Low-Rank for Compression ✅
- r128 UT: 2.26MB (vs 5.15MB full-rank) — 2.3× compression
- Enables more models in 16MB budget for ensemble
- 5× r128 UT ensemble: 1.5540 BPB @ 9.5MB — biggest ensemble gain (+0.079)
- **Trade-off**: individual BPB suffers ~0.1-0.2 BPP from rank constraint

### LoRA (Frozen Random Base) ❌ SIGSEGV crash
- PR #1113 proved concept: 1.37 BPB @ 5.19MB with frozen random orthogonal + rank-32 LoRA
- Our implementation crashes — register_buffer + float() interaction bug
- **Worth fixing**: Would enable even more aggressive compression

### Codebook / Non-Uniform Quantization 🔬 Tested | **Novel**
- **Current int6**: uniform per-row scaling. `w_recon = q * scale` — evenly spaced values.

**Approach 1: Per-matrix codebook** (original idea)
- Store per-matrix codebook of 63 float16 centroids (Lloyd-Max / k-means on trained weights).
- `w_recon = codebook[q]` — non-uniform spacing adapts to actual weight distribution.
- Artifact cost: 63 centroids × 2 bytes × ~200 matrices = **~25KB** (negligible).

**Approach 2: Gaussian Lloyd-Max LUT** (better — zero extra artifact cost)
- Key insight: if weights are Gaussian, the optimal Lloyd-Max quantizer is fully determined by μ and σ.
- Since row means ≈ 0 (verified: max |μ| = 0.003), only σ is needed per row — **same cost as current per-row scale**.
- A single hardcoded LUT of 63 reconstruction values for unit Gaussian is baked into eval code (constant, not stored in artifact).
- `w_recon = σ * LLOYD_MAX_LUT[q]` instead of `w_recon = q * scale`.
- **Theoretical MSE reduction on Gaussian data**: int5: **66%**, int6: **52%**, int8: **33%** lower than uniform.

**Approach 3: σ-regularization + fixed LUT** (most aggressive — eliminates per-row scales entirely)
- Add regularization loss during training: push each weight row toward mean=0, std=σ_target (e.g., 1.0).
- At quantization: no per-row metadata at all. Just store int codes + hardcoded LUT.
- `w_recon = σ_target * LLOYD_MAX_LUT[q]` — **zero per-row storage**.
- Saves **~106 KB** (per-row scales for 53 matrices).
- Open question: does constraining σ hurt model quality? Currently σ varies 0.17–0.90 across layers (5.4× ratio). σ-regularization would need to be strong enough to close this gap without degrading the loss.
- This replaces STE entirely: instead of teaching the model to tolerate quantization noise, constrain the distribution so a fixed optimal quantizer needs no metadata.

**Weight distribution analysis (h100_fullstack_2000.ptz):**
- 53 quantized matrices. Means ≈ 0, skewness ≈ 0, excess kurtosis ≈ +0.09 (very Gaussian).
- Row σ range: 0.17 (mlp.proj) to 0.90 (tok_emb). Embedding layer is the outlier (kurtosis +0.90).
- 50% of rows pass Shapiro-Wilk normality test (p>0.05) — not perfect Gaussian but close.
- All `mlp.proj` layers show 0% Shapiro-Wilk pass rate — worth investigating (may have heavier tails).

**Why denser codes near zero helps:**
- Frequency: 95% of weights are in the dense center → center errors dominate total MSE.
- Relative error: weight 0.05 quantized to 0.0 = 100% error (neuron deleted). Lloyd-Max maps it to ~0.05 instead.
- Tail robustness: weight ±2.0 is insensitive to ±0.05 perturbation (2.5% relative).
- Caveat: tail weights have larger absolute contribution to output — MSE may not perfectly predict BPB impact. Empirical test needed.

**Reconstruction formula (zero stored params beyond σ):**
- Gaussian assumption → optimal centroids are derived analytically from σ alone (Lloyd-Max for N(0,σ)).
- `centroid(q) = σ · (φ(b_lo) - φ(b_hi)) / (Φ(b_hi) - Φ(b_lo))` where φ=PDF, Φ=CDF, boundaries=midpoints of adjacent centroids.
- With σ-regularization (approach 3), even σ is a constant → **zero stored params, zero artifact cost**. Just math in the eval code.

**Tested results (13L, 1000 steps no STE, requantized offline):**

| Method | BPB | Artifact | RT gap |
|--------|-----|----------|--------|
| uniform int6 | 1.817 | 23.2MB OVER | +0.009 |
| uniform int5 | 1.844 | 18.8MB OVER | +0.019 |
| uniform int4 | 2.006 | 13.4MB FITS | +0.181 |
| uniform int3 | 3.047 | 7.9MB FITS | +1.722 |
| **lloyd_max int6** | **1.815** | 26.1MB OVER | +0.002 |
| **lloyd_max int5** | **1.822** | 24.0MB OVER | +0.005 |
| **lloyd_max int4** | **1.850** | 19.6MB OVER | +0.025 |
| **lloyd_max int3** | **2.002** | 14.1MB FITS | +0.177 |

- Lloyd-Max int3 (2.002) beats uniform int4 (2.006) at similar artifact size
- Lloyd-Max degrades gracefully: int6→int3 costs +0.19 BPB. Uniform: +1.23 BPB
- **Key trade-off**: Lloyd-Max codes are max entropy → incompressible → larger per-param artifact, but stable across training steps
- **STE makes uniform artifacts LARGER** (+0.34MB, tested A/B): STE pushes weights toward uniform distribution → higher code entropy
- For competition: Lloyd-Max int4 enables >4000 steps where uniform int6 overflows at ~2000 steps

**Key insight**: MSE is the wrong objective. GPTQ (used by SOTA) achieves +0.002 gap by considering weight importance via Hessian, not just per-weight MSE.

### Federated UT Ensemble 🔬 Tested (single GPU) | **Novel**
- K low-rank UTs sharing one embedding, trained with batch splitting
- Multi-GPU: federated averaging (periodic weight sync, no gradient all-reduce)
- **Script**: `train_fed_ut_ensemble.py` (supports compile, federated, 1cycle LR, SSE, XSA)
- **Combines**: UT compression + low-rank + seed diversity ensemble + federated averaging

**Results (1×A100, 500 steps, 3×4 UT, r128, no compile):**

| K | Indiv BPB | Ensemble BPB | Artifact | RT Ensemble | RT Gap |
|---|-----------|-------------|----------|-------------|--------|
| 3 | 1.684 | **1.629** | 4.25MB | 1.943 | +0.314 |
| 5 | 1.725 | 1.667 | 6.99MB | 1.925 | +0.258 |
| 7 | 1.761 | 1.706 | 9.69MB | 1.939 | +0.233 |
| 3+STE | 1.691 | 1.635 | 4.26MB | 1.940 | +0.305 |

- **Ensemble gain**: +0.055 (K=3) to +0.058 (K=5) — solid
- **K=3 beats K=5 on single GPU** — batch splitting means each branch sees 1/K data; fewer branches = more data per branch
- **On 8 GPUs (federated)**: each branch would see 8× more data → individual quality should approach full-rank levels
- **BLOCKER: Roundtrip gap ~0.25-0.31 BPB** — quantizing A and B separately compounds error (A_q @ B_q ≠ (A@B)_q)
- **Per-factor STE does NOT fix it** — the cross-term error (A@εB + εA@B) is the real problem
- **Fixes needed**: int8 for factors, product-level STE, codebook quantization, or SVD-based serialization

---

## N-gram & Statistical Methods

### Pure N-gram Model ✅ 1.33 BPB with zero training
- Order 7: 1.3771 BPB, Order 10: 1.3319 BPB
- Competitive with 500-step neural model
- **Surprise**: Hash-based n-gram caches that "worked" in competition were a normalization bug

### N-gram Injection into Neural Net ❌ All variants negative
- Input concat+fuse: -0.027 BPP (fuse layer adds quantization noise)
- Input additive: -0.02 BPB
- All-layer residual: -0.02 BPB, 18% slower
- Output logit mix (32K buckets): neutral (±0.001)
- Output logit mix (4M buckets): neutral (+0.002)
- Exact bigram logit injection: neutral
- Context-dependent logit bias (hash→low-rank bias): -0.003
- **Lesson**: The neural net already captures everything n-grams know. Injecting redundant signal adds noise.

### Eval-time N-gram Cache ❌ Ruled illegal
- The 0.44-0.97 BPB "gains" were from hash collision normalization bug
- With proper normalization: n-gram cache gives ~0 improvement
- **Lesson**: If something sounds too good to be true in this competition, check the normalization

---

## GPU & Infrastructure

### GPU Benchmarks (full stack: 13L XSA-all SSE seq2048 batch786K)
- A100 SXM4 80GB: 1930ms/step (compiled, 1 GPU)
- H100 NVL 96GB: 1592ms/step (compiled, 18% faster)
- **8×H100 SXM DDP**: 144ms/step → **4170 steps in 600s** (tested!)
- 8×H100 SXM Federated: 128ms/step → 4710 steps in 600s (but worse BPB)

### GPU Utilization
- **With compile**: 100% GPU compute utilization — no room for parallel work
- **Without compile**: ~50% utilization — room for 2 branches
- Memory: 35GB/80GB used on A100 → headroom for activations, not for extra models

### 8×H200 NCCL Issue
- PyTorch 2.4.0 NCCL NVLS transport crashes on H200
- `NCCL_NVLS_ENABLE=0` doesn't fully fix
- Need newer PyTorch container or H100 SXM (which is what competition uses)

### Competition Data
- Training: 80 shards × 100M tokens = 8B tokens total (we've been using only 10-30)
- Validation: 1 shard = **62M tokens** (not 500K! We truncated with VAL_MAX_TOKENS)
- Data download does NOT count toward 10-min training budget (pre-staged by runner)

---

## 8×H100 Results

### DDP vs Federated (10-min wallclock, 13L full stack, Lloyd-Max int4)

| | DDP | Federated (avg every 50) |
|---|---|---|
| Steps completed | 4170 | 4710 |
| Step time | 144ms | 128ms |
| **Pre-quant BPB** | **1.1708** | 1.2637 |
| **Roundtrip BPB** | **1.2181** | 1.3272 |

- **DDP wins decisively** — gradient averaging quality > data diversity from independent training
- DDP pre-quant (1.171) is only 0.052 from leaderboard SOTA (1.119)

### 7L Scaling (for ensemble analysis)
- 7L at 6000 steps: 1.238 BPB pre-quant, 1.343 roundtrip (gap +0.105)
- Smaller models are 3× more fragile to quantization than 13L
- 2×7L ensemble doesn't beat single 13L

---

## GPTQ Implementation ✅ | Known (our implementation novel)

Implemented full Hessian GPTQ with Cholesky-based error compensation. **Script**: `gptq_quantize.py`

**Results on 13L competition-config weights (1.2278 BPB pre-quant, 2000 steps, 1cycle+XSA+SSE):**

| Scheme | Bits per layer | BPB | Gap | Raw size |
|--------|---------------|-----|-----|----------|
| **GPTQ uniform int6** | 6,6,6,...,6 | **1.2295** | **+0.002** | 23.6MB |
| GPTQ 6,6,4,4,...,5 | mixed conservative | 1.2503 | +0.022 | 17.4MB |
| GPTQ 6,5,3,3,...,4 | mixed aggressive | 1.3993 | +0.172 | 13.9MB |
| GPTQ sensitivity-guided | 6,5,4,4,3,...,4 | 1.3370 | +0.109 | 14.8MB |

- **GPTQ uniform int6 is nearly lossless** (+0.002 gap on competition weights!)
- Mixed precision with int3 still hurts (+0.17) — 7 levels can't represent weights even with error compensation
- Conservative mixed (6,6,4,4,...,5) trades +0.022 BPB for 6.2MB artifact savings
- Calibration: 131K tokens from validation set (SOTA uses AR self-generated; both work)

### Per-Matrix Sensitivity Map ✅ | **Novel**

**Script**: `sensitivity_map.py` — quantize one matrix at a time to int3, measure BPB impact.

**Block 0 accounts for 56% of total quantization damage:**
- blocks.0.attn.c_qkv: +0.737 BPB (24%) — most sensitive single matrix
- blocks.0.mlp.proj: +0.634 BPB (20%)
- blocks.0.mlp.gate_up: +0.214 BPB (7%)
- blocks.0.attn.proj: +0.164 BPB (5%)
- Block 1: ~10% total
- Blocks 3-12: <0.02 BPB each — **nearly insensitive**
- Last 2 blocks: slightly elevated (~0.005-0.008 each) — U-shaped sensitivity

**Key insight**: middle layers barely affect output when quantized → potential for extreme compression there. But GPTQ can't fully compensate int3 errors even in insensitive layers.

### Mixed-Precision Ternary Training ❌ Diverges | **Novel**

Trained with ternary STE on blocks 2-10, int6 STE on blocks 0-1 and 11-12. Per-layer `ste_override` attribute on QuantizedLinear.

| Step | Mixed ternary | Standard (same config) |
|------|-------------|----------------------|
| 500 | 1.584 | 1.416 |
| 1000 | 1.596 | 1.306 |
| 2000 | **1.647 (diverging!)** | **1.228** |

- Model gets WORSE after step 500 — ternary STE destabilizes joint optimization
- The ternary and non-ternary blocks fight each other during training
- Float weights saved: `saved_weights/float/13L_mixed_ternary_2000step_float.pt`

---

## Wild Ideas Explored

### Frequency-Domain Weight Compression ❌ No structure | **Novel**
- DCT analysis of all weight matrices: energy is perfectly flat (10% coefficients = 10.1% energy)
- Weight matrices are white noise in frequency domain — zero spatial correlation
- **Script**: `freq_compress.py`

### Stochastic Quantization Ensemble ❌ Marginal/harmful | **Novel**
- N forward passes with random rounding, average logits

| Method | N=1 | N=5 | N=10 |
|--------|-----|-----|------|
| uniform int6 | 1.817 | 1.812 | 1.812 |
| lloyd_max int4 | 1.850 | 1.857 | 1.853 |
| lloyd_max int3 | 2.002 | 2.098 | 2.082 |

- Uniform: -0.005 (stochastic removes bias). Lloyd-Max: HURTS (centroids already optimal)
- **Script**: `stochastic_eval.py`

### STE Effect on Artifact Size ❌ Makes it WORSE | **Novel finding**
- A/B: STE on → artifact 23.54MB vs STE off → 23.20MB (+0.34MB)
- STE pushes weights toward uniform distribution → higher code entropy → worse compression

### Weight Palette 🔬 Tested (offline) | **Novel**
- Per-matrix k-means codebook. **Script**: `palette_compress.py`

| K (entries) | Bits | MSE | Artifact | Budget |
|-------------|------|-----|----------|--------|
| 8 | 3 | 5.11e-04 | 14.2MB | FITS |
| 16 | 4 | 1.54e-04 | 19.5MB | OVER |
| 256 | 8 | 6.40e-06 | 38.9MB | OVER |
| uniform int6 | 6 | 8.67e-05 | ~15-23MB | varies |

- K=8 (3 bits): MSE comparable to Lloyd-Max int4, 14.2MB artifact
- zstd compresses unused high bits of uint8 indices effectively
- **Differentiable training version** (jointly optimize codebook + weights) untested but promising

### Why Lloyd-Max / MSE-Optimal Quantization Loses to GPTQ 🤔

Theoretical analysis of why information-theoretically optimal quantizers don't win in practice:
1. **Not all weights matter equally** — GPTQ uses Hessian to know which weights are important
2. **Co-adaptation** — GPTQ compensates errors across columns; independent quantizers can't
3. **Compression paradox** — max entropy codes (Lloyd-Max) are incompressible; "wasted" uniform codes compress well with zstd
4. **The system is model + compressor** — STE optimizes for the full system, not just the quantizer

---

## Hierarchical Multi-Resolution Transformer 🔬 Prototype tested | **Novel**

### Architecture Overview

**Core insight**: distant context needs less resolution. Compress old context with a shared convnet, attend globally at low cost, predict at full resolution for the last window.

```
Input: 4096 tokens at embed_dim=32

Windows 0-14 (past context):
  → Shared window encoder (convnet: 256 tokens → 1 token at dim 512)
  → 15 compressed tokens
  → Global transformer (BIDIRECTIONAL, 15×15 attention — nearly free)
  → Rich context representations

Window 15 (prediction window, 256 tokens):
  → Project embed_dim → model_dim
  → Local decoder (2 layers):
      - First 192 tokens: BIDIRECTIONAL self-attention (known context)
      - Last 64 tokens: CAUSAL self-attention (predictions)
      - ALL positions: cross-attention to 15 global summaries
  → Predict last 64 tokens only
```

### Design decisions and rationale

1. **Non-overlapping windows** (stride=256): each summary covers unique content. Overlap is redundant since the global transformer bridges windows.

2. **Window 15 NOT in global path**: including it causes information leakage — the convnet compresses ALL 256 tokens (including future tokens that haven't been predicted yet). The convnet has no causal ordering.

3. **Global transformer is BIDIRECTIONAL**: windows 0-14 are all in the past — no causality constraint. Bidirectional lets each summary be enriched by all other summaries. This is an encoder-decoder architecture: encoder (bidirectional global) + decoder (causal local).

4. **Prefix-causal local attention**: first 192 tokens are bidirectional (known from previous slides), last 64 are causal (being predicted). This gives every prediction 192 fine-grained neighbors + 15 global summaries.

5. **Sliding eval with stride=64**: each slide predicts 64 new tokens. The 192-token prefix provides fine-grained context at window boundaries. Global encoder can be cached every 4 slides (when window alignment repeats). **5× cheaper per predicted token than standard sliding window eval.**

6. **Training stride=256**: more independent gradients (75% less overlap between examples). Every token is trained as context even if not predicted. Eval uses stride=64 — the prediction task is identical regardless of stride.

### Window encoder (shared, ~1.5M params)
```
Conv1d(32→128, k=8, s=8):  256 tokens → 32 tokens
Conv1d(128→256, k=4, s=4): 32 tokens → 8 tokens
Conv1d(256→512, k=8, s=8): 8 tokens → 1 token
Linear(512→512):           final projection
```

### Compute comparison

| | Standard 13L (seq=1024) | Hierarchical (4096 ctx) |
|---|---|---|
| Context | 1024 tokens | **4096 tokens** |
| Compute/predicted token | 88 MFLOP | **34 MFLOP** |
| Steps in 10 min (est.) | ~4000 | **~37,000** |
| Sliding eval cost | 1400 MFLOP/token | **280 MFLOP/token** |

### Prototype results
- **1.3171 BPB** at 30K steps (same wall clock as standard 1000 steps)
- Standard at 1000 steps: 1.306 BPB → only 0.011 gap
- Used simple Adam optimizer, no Muon, no 1cycle — significant room for improvement
- 75ms/step on H100 SXM, 38.4M params
- **⚠️ Result may be inflated**: prototype had window 15 in global path (information leakage). Need to rerun with clean architecture.
- **Script**: `train_hierarchical.py`

### Smaller hierarchical for artifact budget
- Current: 38.4M params (same as 16L standard — doesn't save artifact space)
- Reduce to 6 global layers: ~31M params (same as 13L standard)
- Or reduce model_dim to 384: ~20M params (could fit in 16MB with int6 + GPTQ)
- The architecture's value is in **compute efficiency** (more steps/minute) not param efficiency

---

## Leaderboard SOTA Analysis (PR #1019, 1.1147 BPB)

Key techniques from @abaybektursun:
- **GPTQ (Full Hessian)**: +0.002 gap. AR self-generated calibration data
- **11L, MLP 3×**: ~7000 steps at 86ms/step. "MLP 3× is the single largest contributor"
- **XSA all layers, BigramHash 3072×112, Late QAT, EMA+SWA**
- **Sliding window eval**: stride 64, free -0.025 to -0.035 BPB
- Blog: https://abay.tech/posts/pr-1019-model-autopsy
- MLP needs 7-8 bits, attention survives at 4-5 bits (stable rank predicts sensitivity)

### Width vs Depth (from competition analysis)
- SOTA uses 11L/MLP 3× (~22M params). We used 13L/MLP 2× (31M params).
- SOTA's wider MLP is more important than our extra depth
- Middle layers are quantization-insensitive → possibly superfluous for this task size
- **Untested**: fewer but much wider layers (e.g., 7L/MLP 5×, 5L/MLP 8×)
- Going wider saves latency (parallel GPU compute) while depth adds sequential steps

---

## Ideas Worth Pursuing

### Hierarchical architecture improvements 💡
- Implement clean version (no window 15 leakage, bidirectional global, prefix-causal local)
- Add Muon optimizer, 1cycle LR, proper training pipeline
- Test smaller versions (6 global layers, dim=384) that fit in 16MB
- Combine with GPTQ for serialization

### FP8 training 💡
- torch.float8_e4m3fn available on PyTorch 2.4.1 + A100/H100
- Could give ~1.5-2× faster matmuls → more steps in 10 min
- Implementation needed (convert matmuls to FP8)

### Width experiments 💡
- Test 7L/MLP 5× vs 11L/MLP 3× vs 13L/MLP 2× at same param count
- Measure BPB vs step time trade-off
- Extreme: 5L/MLP 8× — maximum width, minimum depth

### Differentiable codebook training 💡
- Weight palette with K=8-16 entries, jointly trained with the model
- Straight-through or Gumbel-softmax for discrete index selection
- Could discover non-obvious weight distributions that quantize optimally

### Reduced embedding dimension 💡
- Vocab=1024 tokens could theoretically be represented in 10 bits
- Current embed_dim=512 is 50× overkill for token identity
- Test embed_dim=32-64 with projection to model_dim — saves embedding params and could improve generalization

---

## Priority Stack

1. **Hierarchical clean implementation** — Fix leakage, add Muon/1cycle, test at competition scale
2. **Width experiments** — 7L/MLP 5× or similar, measure compute-BPB frontier
3. **GPTQ integration** — Deploy our GPTQ into the training pipeline for competition submission
4. **Sliding window eval** — Implement for both standard and hierarchical models
5. **FP8 training** — Faster steps → more training in 10 min
6. **Differentiable codebook** — Train with learned quantization end-to-end
7. **Trim code** — 150KB train_combined.py wastes artifact budget

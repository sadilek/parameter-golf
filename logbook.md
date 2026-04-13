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

## Hierarchical Multi-Resolution Transformer 🔬 Clean implementation ready | **Novel**

### Architecture Overview

**Core insight**: distant context needs less resolution. Compress old context with a shared convnet, attend globally at low cost, predict at full resolution for the last window.

```
Input: 4096 tokens at embed_dim=32

Windows 0-14 (past context):
  → Shared window encoder (convnet: 256 tokens → 1 token at dim 384)
  → 15 compressed tokens
  → Global transformer (BIDIRECTIONAL, 6 layers, 15×15 attention — nearly free)
  → Rich context representations

Window 15 (prediction window, 256 tokens — NOT in global path):
  → Project embed_dim → model_dim
  → Local decoder (2 layers, prefix-causal + cross-attention to 15 global summaries):
      - First 192 tokens: BIDIRECTIONAL self-attention (known context)
      - Last 64 tokens: CAUSAL self-attention (predictions)
  → Logits from positions 191-254 predict tokens at positions 192-255
  → Loss on 64 predicted tokens only (targets within window, no +1 read needed)
```

### Design decisions and rationale

1. **Non-overlapping windows** (stride=256): each summary covers unique content. Overlap is redundant since the global transformer bridges windows.

2. **Window 15 NOT in global path**: including it causes information leakage — the convnet compresses ALL 256 tokens (including future tokens that haven't been predicted yet). The convnet has no causal ordering.

3. **Global transformer is BIDIRECTIONAL**: windows 0-14 are all in the past — no causality constraint. Bidirectional lets each summary be enriched by all other summaries. This is an encoder-decoder architecture: encoder (bidirectional global) + decoder (causal local).

4. **Prefix-causal local attention**: first 192 tokens are bidirectional (known from previous slides), last 64 are causal (being predicted). This gives every prediction 192 fine-grained neighbors + 15 global summaries.

5. **Targets within window**: logit at position i predicts token at position i+1. Using logits from positions 191-254 (64 logits), targets are tokens at positions 192-255 — all within the 256-token window. No +1 token read needed. The model only needs `input_ids`, no separate `target_ids`.

6. **Sliding eval with stride=64**: each slide predicts 64 new tokens. The 192-token prefix provides fine-grained context at window boundaries. Used for both training monitoring and final evaluation.

7. **Training stride=256 via streaming**: each step reads exactly 256 new tokens from a contiguous per-rank stream. Old prediction window becomes newest context (1 convnet pass with gradient, 14 cached detached). 25% token efficiency (64 predicted / 256 consumed) vs 1.5% without streaming. `TokenStream.take(n)` never crosses shard boundaries — returns short read if exhausted.

8. **No DDP**: Muon handles its own gradient all-reduce. Non-Muon params (embedding, head, scalars, Conv1d) are manually all-reduced after backward. This avoids DDP incompatibility with the streaming forward path.

### Window encoder (shared)
```
Conv1d(32→96, k=8, s=8):   256 tokens → 32 tokens
Conv1d(96→192, k=4, s=4):  32 tokens → 8 tokens
Conv1d(192→384, k=8, s=8): 8 tokens → 1 token
Linear(384→384):            final projection
```

### Default config (fits 16MB with int6)
```
model_dim=384, n_global_layers=6, n_local_layers=2, n_heads=6, mlp_mult=3
embed_dim=32, window_size=256, prefix_len=192, pred_len=64, total_seq=4096
```
Estimated ~18-20M params.

### FLOP breakdown (per forward pass, batch=1)

| Component | FLOPs | % |
|---|---|---|
| Window encoder (1 window, streaming) | 4.2M | 0.1% |
| Window encoder (15 windows, cold start) | 63.4M | 2.1% |
| Global transformer (6L × 15 tokens) | 347.1M | 12.0% |
| **Local decoder (2L × 256 tokens)** | **2,495.8M** | **84.3%** |
| LM head (64 tokens) | 50.3M | 1.7% |
| **Total (streaming step)** | **2,904M** | |

**vs Standard 13L** (seq=1024, dim=768): 45M vs 242M FLOPs per predicted token = **5.3× cheaper**.

### Prototype results (OLD — had leakage bug)
- **1.3171 BPB** at 30K steps (same wall clock as standard 1000 steps)
- Standard at 1000 steps: 1.306 BPB → only 0.011 gap
- Used simple Adam optimizer, no Muon, no 1cycle
- 75ms/step on H100 SXM, 38.4M params
- **⚠️ Result inflated**: prototype had window 15 in global path (information leakage)

### Clean implementation results (2026-04-03)

**Smoke test** (dim=384, 6G+2L, 200 steps, 1xH100):
- 17.9M params, 5.46MB artifact, 287ms/step
- BPB: 4.19 → 2.93 in 200 steps. Roundtrip gap: +0.002

**10-min run** (dim=576, 8G+2L, 1xA100 SXM):
- 48.5M params, 2478 steps at 241ms/step
- **BPB: 2.74** (roundtrip: 2.75, gap +0.007)
- Artifact: 24.3MB — **OVER 16MB** (need smaller model)
- Float weights saved: `saved_weights/float/hier_576d_8G2L_10min_float.pt`

**LR sweep** (dim=512, 8G+2L, 500 steps, 1xA100 SXM):

| LR multiplier | matrix_lr | BPB @ step 101 | BPB @ step 501 |
|---|---|---|---|
| **1× (baseline)** | 0.04 | 3.280 | **2.793** |
| 2× | 0.08 | 3.268 | 2.887 |
| 4× | 0.16 | 3.820 | 3.749 (diverging) |

Higher LR hurts at 0.08+. But **lower LR helps** — see extended sweep below.

**Extended LR + prefix/pred sweep** (dim=384, 6G+2L, 500 steps, 1xA100 SXM, 2026-04-03):

| matrix_lr | prefix+pred | BPB @ 101 | BPB @ 501 | Roundtrip | Gap |
|---|---|---|---|---|---|
| 0.04 | 192+64 | 3.280 | 2.793 | — | — |
| 0.08 | 192+64 | 3.268 | 2.887 | — | — |
| 0.03 | 192+64 | 3.176 | 2.770 | 2.773 | +0.003 |
| 0.03 | 128+128 | — | 2.716 | 2.717 | +0.001 |
| 0.02 | 128+128 | — | 2.708 | 2.711 | +0.003 |
| **0.02** | **64+192** | — | **2.678** | **2.680** | +0.002 |

Key findings:
- LR monotonically improves going lower: 0.04 → 0.03 → 0.02 (may not have bottomed out yet)
- More prediction tokens monotonically better: 64 → 128 → 192 (more loss signal per step)
- Best config so far: LR=0.02, 64+192 → **2.678 BPB** (roundtrip 2.680)
- Total improvement over original baseline: **-0.115 BPB**
- Quant roundtrip gap consistently tiny (~0.002)
- Next to try: even lower LR (0.015?), even more prediction (32+224?), or both

**Artifact size calibration**: dim=512, 8G+2L = 38.4M params → **12.4MB artifact** (fits 16MB). dim=576, 8G+2L = 48.5M params → 24.3MB (over). The param→artifact ratio is ~0.32 bytes/param with int6+zstd.

**Key concern: 2.68 BPB is far behind standard 13L (1.31 BPB at same wall time).**
Root cause: even with 192 predicted tokens per micro-step, it's still 5× less than the standard model's 1024. The architecture's compute efficiency doesn't translate to training efficiency within a 10-min budget.

**Possible mitigations**:
- Push pred_len even further (224? 240?) — trend hasn't plateaued yet
- Lower LR further (0.015? 0.01?) — also hasn't plateaued
- Add auxiliary loss on context windows (predict within the encoder)
- Give the model more wall time (80 min on 1xH100 ≈ 10 min on 8xH100 in total compute)

**Batched streaming** (2026-04-04):

Previously: batch_size=1 with 8 sequential grad_accum steps → GPU at 10% util, 2% memory.
Now: B independent token streams processed in a single batched forward pass.

Batch size sweep (no compile, LR=0.04, 50 steps, grad_accum=8, 1×A100 SXM):

| BS | ms/step | tokens/step | tokens/sec | BPB@50 | GPU mem |
|---|---|---|---|---|---|
| 64 | 509 | 98K | 192K | 2.688 | ~2 GB |
| 128 | 537 | 196K | 365K | 2.690 | ~2 GB |
| 256 | 701 | 393K | 561K | 2.689 | ~6 GB |
| 512 | 1,090 | 786K | 721K | 2.688 | ~10 GB |
| 1024 | 1,899 | 1.57M | 827K | 2.688 | ~20 GB |

Key findings:
- BPB flat across all BS at LR=0.04 → already curvature-limited, not noise-limited
- ms/step roughly doubles per 2× BS beyond 128 (GPU compute now dominates)
- Memory scales linearly, fits 80GB up to ~BS=4096
- `torch.compile` recompiles per batch size and can OOM on pod CPU RAM at large BS

**Joint BS × LR sweep** (BS=256, 50 steps, no compile, 1×A100):

| LR | BPB@50 |
|---|---|
| 0.02 | 2.702 |
| 0.04 | 2.689 |
| **0.08** | **2.684** |
| 0.12 | 2.713 |
| 0.16 | 2.782 |

**BS=512 LR sweep** (50 steps, no compile, 1×H100):

| LR | BPB@50 |
|---|---|
| **0.06** | **2.610** |
| 0.07 | 2.613 |
| 0.08 | 2.615 |
| 0.09 | 2.622 |
| 0.10 | 2.626 |
| 0.12 | 2.646 |
| 0.16 | 2.706 |

Optimal LR at BS=512 is ~0.06 (not sqrt-scaling — sublinear). BS=512 strongly beats BS=256 at their respective optimal LRs.

**Full run: BS=512, LR=0.06, 500 steps, 1×H100 NVL, no compile:**

| Step | BPB |
|---|---|
| 101 | 2.596 |
| 201 | 2.559 |
| 301 | 2.510 |
| 401 | 2.461 |
| **501** | **2.437** |
| Roundtrip | 2.442 (+0.005) |

**-0.241 BPB improvement** over pre-batching best (2.678). Loss curve still dropping steeply — not converged at 500 steps. 702ms/step on H100.

### Clean implementation status
- **Script**: `train_hierarchical.py` (~1000 lines, 45KB)
- All architecture bugs fixed (no leakage, bidirectional global, prefix-causal mask, correct logit-target alignment)
- Batched streaming training with vectorized cache updates
- Muon optimizer, 1cycle LR, gradient accumulation, manual gradient all-reduce
- Sliding eval (stride=64) for all validation
- Int6 quantization + zstd compression + roundtrip eval
- LOAD_WEIGHTS support for resumed training

**Deep local stack** (2026-04-04):

14 local layers (48M params), BS=512, LR=0.03 (1cycle), 2000 steps, 1×H100, no compile:

| Step | BPB | Note |
|---|---|---|
| 401 | 1.945 | |
| **601** | **1.910** | 1cycle peak — best BPB |
| 1001 | 2.335 | post-peak destabilization |
| 2001 | 2.049 | never recovers to pre-peak |
| Roundtrip | 2.123 | +0.074 gap (large model quantizes worse) |

**Artifact: 28.5MB — OVER 16MB.** Sizing script used zlib but real pipeline is torch.save without extra compression. Need ~6-8 local layers to fit 16MB.

Key findings:
- **14 local layers hit 1.910 BPB** — massive improvement from 2-layer's 2.437
- 1cycle peak LR=0.03 too aggressive for 2000 steps — destabilizes mid-training
- Quantization gap +0.074 at 48M params (vs +0.005 at 18M) — needs GPTQ or better quant

**Wide MLP experiment** (dim=384, 4 local layers, MLP×6, 36.2M params, BS=512, LR=0.03, 2000 steps, 1×H100):

| Step | BPB |
|---|---|
| 601 | 1.984 |
| 1001 | 1.878 |
| 1601 | 1.732 |
| **2001** | **1.686** |
| Roundtrip | 1.708 (+0.023) |
| Artifact | 19.6MB (OVER) |

Key findings:
- **1.686 BPB** — huge leap, still dropping steadily at step 2000 (not converged)
- Wide MLP 2.5× faster/step than deep stack at same param count (1.4s vs 3.5s)
- LR=0.03 with 2000-step 1cycle: **no destabilization** — monotonically improving
- Artifact 19.6MB (over 16MB). Need MLP=5 (~31.8M) to fit, or better quantization (GPTQ)
- Trained artifact is ~2.6× larger than random-weight estimate (19.6 vs 7.6MB)
- **Artifact sizing rule of thumb: trained ≈ 2.5× random-weight compressed size**

**MLP=4 with warmdown schedule** (dim=384, 4L+6G, 27.3M params, BS=512, LR=0.015, warmdown 1200, 5000 steps, 1×A100 SXM):

| Step | BPB |
|---|---|
| 1001 | 2.059 |
| 2001 | 1.888 |
| 3001 | 1.824 |
| 4001 | 1.769 |
| **5001** | **1.687** |
| Roundtrip | 1.737 (+0.050) |
| **Artifact** | **15.2 MB (OK)** |

Key findings:
- **First run that fits 16MB** — 27.3M params, int6+zstd = 15.2MB
- Warmdown schedule (flat LR → linear decay last 1200 steps): **no destabilization**, monotonic improvement
- LR=0.03 diverges after ~2500 steps; LR=0.015 is stable for 5000+
- Warmdown final boost: 1.816 → 1.687 (last 1200 steps with decaying LR)
- Roundtrip gap +0.050 — GPTQ would help
- H100 killed run projected ~1.645 at 5000 steps (slightly better, data ordering?)

**MLP=6 with GPTQ + 6-bit packing** (dim=384, 4L+6G, 36.2M params, BS=512, LR=0.015, warmdown 1200, 5000 steps, 1×A100 SXM):

| Step | BPB |
|---|---|
| 1001 | 2.038 |
| 2001 | 1.873 |
| 3001 | 1.809 |
| 4001 | 1.752 |
| **5001** | **1.665** |
| Roundtrip | 1.712 (+0.047) |
| Artifact | 26.0 MB (OVER) |

Key findings:
- **1.665 BPB** — best hierarchical result so far, but artifact 26MB (over)
- **6-bit packing HURTS compression**: 26.0MB (packed+zstd) vs 19.6MB (int8+zstd) for same model. Packing removes the redundant top-2-bits pattern that zstd was already exploiting efficiently. Net effect: +33% larger artifact.
- GPTQ roundtrip gap +0.047 — similar to naive int6 (+0.050 at MLP=4). GPTQ helps quality but doesn't shrink artifact size when combined with packing.
- **Conclusion: don't use 6-bit packing with zstd compression.** Use int8 storage + GPTQ + zstd instead.

**Quantization efficiency comparison:**
- Our model: 27-36M params → 0.42-0.54 bytes/param (int8+zstd, naive)
- SOTA: ~46M params → 0.35 bytes/param (int8+zstd, GPTQ)
- Gap is partly GPTQ (more compressible rounding) and partly architecture (our cross-attn/conv layers compress worse)

### Hierarchical architecture summary

| Config | Params | BPB | Artifact | Fits 16MB? |
|---|---|---|---|---|
| 2L MLP×3, BS=1, 500 steps | 17.9M | 2.678 | ~6 MB | Yes |
| 2L MLP×3, BS=512, 500 steps | 17.9M | 2.437 | ~7 MB | Yes |
| 14L MLP×3, 2000 steps | 48.0M | 1.910* | 28.5 MB | No |
| 4L MLP×6, 2000 steps | 36.2M | 1.686 | 19.6 MB | No |
| **4L MLP×4, 5000 steps** | **27.3M** | **1.687** | **15.2 MB** | **Yes** |
| 4L MLP×6, 5000 steps (GPTQ+pack) | 36.2M | 1.665 | 26.0 MB | No |

*1cycle peak, not final

**Key lessons from hierarchical exploration:**
1. **Batched streaming was the biggest win** — 10× more training signal per step, GPU utilization from 10% to useful
2. **Wider > deeper** for same param count — 2.5× faster per step, GPU parallelizes width
3. **Warmdown schedule >> 1cycle** for long runs — 1cycle destabilizes, warmdown is monotonic
4. **LR=0.015 is stable for 5000+ steps** at BS=512; LR=0.03 diverges after ~2500
5. **6-bit packing hurts** — zstd already handles the wasted bits efficiently
6. **Artifact constraint is the binding limit** — int6+zstd gives ~0.5 bytes/param trained, limiting us to ~30M params for 16MB
7. **Still 0.55 BPB behind standard GPT** (1.687 vs 1.171) — the 75% loss efficiency (192/256 tokens) and smaller model are the root causes

---

## Overnight Experiment Suite (2026-04-06)

SOTA-like baseline + 10 ideas. All: 11L, MLP×3, dim=512, XSA-all, LeakyReLU², seq=2048, compiled, 1000 steps, 1×A100 SXM.

| # | Experiment | BPB | vs baseline | Artifact | Notes |
|---|---|---|---|---|---|
| 0 | **Baseline** (warmdown) | 1.330 | — | 15.9 MB | SOTA-like config |
| 1 | +SSE calibration | 1.320 | **-0.010** | 15.5 MB | Cheap win, only 33K params |
| 2 | +Lloyd-Max quant | 2.145 | -0.815 | 16.0 MB | Broken — STE_TYPE doesn't work with int6 post-train |
| 3 | Smaller batch (262K) | 1.331 | -0.001 | 15.7 MB | Neutral — more steps but noisier gradient |
| 4 | **1cycle LR** | **1.311** | **-0.019** | 14.1 MB | Best single idea. Also smallest artifact. |
| 5 | +BigramHash | 1.334 | +0.004 | 16.3 MB | Slightly worse + larger artifact |
| 6 | MLP×4, 9L | 1.329 | -0.001 | 15.7 MB | Neutral — wider but shallower cancels out |
| 7 | 13L (deeper) | 1.331 | +0.001 | 15.7 MB | Neutral at 1000 steps |
| 8 | seq=4096 | 1.330 | +0.000 | 16.0 MB | No help — longer seq doesn't help at 1000 steps |
| 9 | +EMA | 1.876 | -0.546 | 12.2 MB | **Catastrophic** — confirmed: EMA + quantization = disaster |
| 10 | **Combo** (1cycle+SSE+BigramHash) | **1.299** | **-0.031** | 14.3 MB | Best result. 1cycle + SSE stack. BigramHash might be hurting slightly. |

Key findings:
- **1cycle LR is the single biggest win** (-0.019 over warmdown at 1000 steps)
- **SSE calibration stacks** with 1cycle for combo -0.031
- **EMA is catastrophic** with quantization — confirmed again (DO NOT USE)
- **BigramHash slightly hurts** at 1000 steps — may need more steps or tuning
- **Lloyd-Max needs different integration** — STE_TYPE env var doesn't apply to post-training int6
- Deeper (13L) and wider (MLP×4) are neutral at 1000 steps — need more steps to differentiate
- Smaller artifact with 1cycle (14.1 MB) — weights are more compressible because 1cycle's cosine cooldown produces smoother weight distributions

**Novel ideas suite** (implemented + tested, same SOTA-like config, 1000 steps, 1×A100 SXM):

| # | Experiment | BPB | vs baseline | Artifact | Notes |
|---|---|---|---|---|---|
| 0 | Baseline (warmdown) | 1.331 | — | 15.6 MB | |
| 5 | Foveated attn (top 4L, w=256) | 0.002 | — | 15.8 MB | **BUG: information leak** in windowed mask. Invalid result. |
| 6 | Adaptive depth (per-token gate) | 1.350 | +0.019 | 16.2 MB | Hurts — gates add params + overhead, no compute saved at 1000 steps |
| 7a | MoE 4-expert, 5L | 1.333 | +0.002 | 43.3 MB | Neutral BPB, massive artifact — too many expert params |
| 7b | MoE 2-expert, 11L | 1.331 | +0.000 | 25.5 MB | Neutral — 2× MLP params but only 1 active, artifact too large |
| 8 | Distill teacher (16L, 500 steps) | 1.440 | — | — | Teacher only. Student distillation not implemented. |
| 9 | Embed bottleneck (dim=128→512) | 1.347 | +0.016 | 15.4 MB | Hurts — bottleneck embedding loses information |
| 10 | Federated avg (sync every 200) | 1.329 | **-0.002** | 15.5 MB | Tiny improvement — needs multi-GPU to truly test |

Novel idea findings:
- **Foveated attention has a bug** — windowed mask leaks future tokens. Need to fix and retest.
- **MoE doesn't help at constant artifact budget** — experts add params but artifact grows proportionally. Would need mixed-precision (inactive experts at lower bits) to be artifact-efficient.
- **Adaptive depth hurts** — the gates learn to always use all layers (bias=2.0 → sigmoid≈0.88). At 1000 steps there's no compute saving. Might help at 5000+ steps where some tokens truly plateau early.
- **Distillation incomplete** — teacher model is weaker than student would be (fewer steps). Needs training loop integration.
- **Federated averaging shows tiny signal** — needs multi-GPU to properly test (on 1 GPU it just averages with itself).

**Combined results (both suites):**

Top techniques ranked by Δ BPB:
1. **1cycle+SSE combo**: -0.031
2. **1cycle LR alone**: -0.019
3. **SSE calibration**: -0.010
4. **Federated avg**: -0.002
5. All others: neutral or negative

**Next steps:**
1. Run combo (1cycle+SSE) for full 4000+ steps on 8×H100 to see competition-scale result
2. Try 1cycle+SSE without BigramHash (BigramHash may be hurting)
3. Fix foveated attention bug and retest
4. Fix Lloyd-Max integration for post-training quantization

---

## SOTA Reproduction (2026-04-07)

Used actual PR #1019 `train_gpt_sota.py` (downloaded from GitHub, patched FA3→SDPA fallback).

**Training run** (1×H100 NVL, 7000 steps, no FlashAttention 3):

| Step | BPB | ms/step |
|---|---|---|
| 500 | 1.408 | 1393 |
| 1000 | 1.318 | 1392 |
| 2000 | 1.256 | 1406 |
| 3000 | 1.232 | 1406 |
| 4000 | 1.208 | 1406 |
| 5000 | 1.188 | 1405 |
| 6000 | 1.164 | 1403 |
| 6500 | 1.149 | 1403 |
| **7000** | **1.138** | 1403 |
| Post-EMA | 1.137 | |
| **GPTQ roundtrip** | **1.141** | |
| Sliding eval (est.) | ~1.116 | |
| Artifact | 15.91 MB | |

**Comparison with actual SOTA:**

| Metric | SOTA (8×H100) | Our repro (1×H100) | Gap |
|---|---|---|---|
| Pre-quant | 1.135 | 1.138 | +0.003 |
| GPTQ roundtrip | 1.138 | 1.141 | +0.003 |
| GPTQ gap | +0.002 | +0.004 | |
| Sliding eval | **1.115** | **~1.116** (est.) | ~+0.001 |

**Reproduction is successful.** Small gaps from no FA3 and 1 vs 8 GPU.

~~**BLOCKING ISSUE: Sliding eval crashes.**~~ **RESOLVED** — see below.

---

## Sliding Eval Fix (2026-04-07)

Created `eval_sota.py` — standalone eval script that loads GPTQ artifact and runs both standard + sliding eval with torch.compile.

**Root causes of previous crashes (in old `eval_sliding.py` / `eval_sliding_simple.py`):**
1. Wrong function names (`dequantize_int6` → should be `dequantize_mixed_int6`, `build_bpb_luts` → `build_sentencepiece_luts`, `load_val_tokens` → `load_validation_tokens`)
2. Missing unbank→dequantize→rebank pipeline (tried to load weights directly into banked model)
3. `exec` split on `"\ndef main():"` which didn't match `"def main() -> None:"` — accidentally ran the entire training script

**The "OOM" was a red herring.** In standalone eval with a single model, torch.compile works fine (~923 MB / 80 GB GPU). The SOTA script's training-time eval had two models in memory, but that's not the fundamental issue.

**Verified on 1×A100 SXM (1000-step test model):**

| Eval | eval_sota.py | SOTA script | Match? |
|---|---|---|---|
| Standard BPB | 1.3988 | 1.3982 | Yes (float rounding) |
| Sliding (s=64) BPB | 1.3744 | 1.3746 | Yes |
| Delta | -0.0245 | -0.0236 | Yes |

torch.compile works. Sliding eval takes ~1880s on 1×A100 (includes compile time). The `-0.025 BPB` free improvement is confirmed.

**Usage:** `python eval_sota.py final_model.int6.ptz` (compile on by default, `BATCH_SEQS=8`, `NO_COMPILE_EVAL=1` to disable)

---

## Phase 1: 1cycle LR on SOTA Model (2026-04-08)

**Hypothesis:** 1cycle LR was our best measured technique (-0.019 BPB on our model). Test if it transfers to the SOTA architecture.

**Setup:** A/B test, SOTA config (11L, 512d, 8H/4KV GQA, BigramHash 3072×112, XSA-all, LeakyReLU²), 1000 steps, 1×A100 SXM.
- Run A: Baseline warmdown (warmdown_iters=200)
- Run B: 1cycle (peak_frac=0.3, min_div=4) via `train_gpt_sota_1cycle.py`

**Step-by-step BPB:**

| Step | Baseline | 1cycle | Delta |
|------|----------|--------|-------|
| 200 | 1.6265 | 1.6355 | +0.009 |
| 400 | 1.4515 | 1.4479 | -0.004 |
| 600 | 1.3875 | **1.3690** | **-0.019** |
| 800 | 1.3547 | **1.3331** | **-0.022** |
| 1000 | **1.2992** | 1.3253 | +0.026 |

**Full results:**

| Metric | Baseline | 1cycle | Delta |
|--------|----------|--------|-------|
| Pre-quant BPB | **1.2992** | 1.3253 | +0.026 (worse) |
| GPTQ roundtrip | **1.3982** | 1.4060 | +0.008 (worse) |
| GPTQ gap | 0.099 | 0.081 | -0.018 (1cycle quantizes better) |
| Sliding (s=64) | **1.3746** | 1.3813 | +0.007 (worse) |

**Verdict: 1cycle does NOT transfer to the SOTA model.** ❌

1cycle is better at steps 600-800 (when its peak LR enables faster exploration), but the cosine anneal kills the LR too early. By step 800, 1cycle's LR is near its minimum (0.25× base), while the baseline still has full LR until step 800 and makes rapid progress during warmdown.

**Why it worked on our model but not here:**
- Our overnight experiments used `train_combined.py` with different architecture (13L), different optimizer config, and measured at a fixed 1000 steps where 1cycle happened to align well with the warmdown start
- The SOTA's Muon optimizer with momentum warmup (0.92→0.99 over 1500 steps) already provides the "warm start" effect that 1cycle gives
- The 1cycle cosine anneal competing with warmdown is wasteful — both try to reduce LR in the late phase but on different schedules

**One positive:** 1cycle produces more quantization-friendly weights (GPTQ gap 0.081 vs 0.099, saving 0.018 BPB in the quant step). This partially compensates but doesn't overcome the worse pre-quant BPB.

**Next:** SSE calibration (-0.010 BPB) is the remaining technique with measured positive delta. Or: consider entirely different angles (custom tokenizer, TTT, SLOT).

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

### Prefix summary in global transformer 💡 Untested
- Include the 192-token prefix of the prediction window as a 16th global token
- No leakage — prefix tokens are known context, not predictions
- Convnet can't handle 192 tokens (strides assume 256), so use a separate encoder: mean-pool + Linear projection to `(1, model_dim)`
- Benefit: global summaries become conditioned on recent context before cross-attention. Second-order effect — the local decoder already has direct self-attention to the prefix. Worth testing after baseline.

### Batch size scheduling 💡 **Novel** (no submissions use it)
- Start small batch (131K), switch to large (786K) at ~33% of training
- Smith et al. 2018: small batch = more updates = faster exploration; large batch = precise convergence
- `train_combined.py` already has this (`batch_schedule_fraction=0.33`), but no submitted record uses it
- Extra relevant for hybrid recurrent: scan overhead is fixed per step regardless of batch size, so penalty is proportionally smaller at large batch
- Could also try continuous ramp instead of step function

### Hybrid recurrent-attention architecture ❌ **Novel** — **DOESN'T WORK AT FULL TRAINING**
- Replace early attention layers with stateless GatedRecurrence (GRU-style gated linear recurrence + parallel scan)
- Parallel scan with log2(T) doublings — compile-friendly with `fullgraph=True`
- Dedicated banks (`rec_gates_bank`, `rec_out_bank`) shrinking `qo_bank`/`kv_bank` to non-recurrent layers only
- GatedRecurrence is stateless — weights come from banks, optimized via banked Muon (same LR as attention matrices)

- **Simple baseline sweep** (train_gpt.py, A100 + compile, 300 steps, seq=2048, baseline BPB=1.7436):

| Config | layers | step 300 BPB | roundtrip BPB | ms/step | Δ BPB |
|---|---|---|---|---|---|
| baseline | — | 1.7436 | 1.7724 | 247 | — |
| rec_0 | [0] | 1.7043 | 1.7310 | 263 | -0.041 |
| rec_4 | [4] | 1.7255 | 1.7528 | 263 | -0.020 |
| rec_8 | [8] | 1.7435 | 1.7699 | 263 | -0.003 |
| **rec_012** | [0,1,2] | **1.6675** | **1.6924** | 293 | **-0.080** |
| rec_345 | [3,4,5] | 1.6998 | 1.7253 | 293 | -0.047 |
| rec_678 | [6,7,8] | 1.7208 | 1.7476 | 296 | -0.025 |
| rec_0246 | [0,2,4,6] | 1.6801 | 1.7051 | 307 | -0.067 |

- **SOTA architecture verification** (train_gpt_sota.py, A100 + compile, **400 steps**, seq=2048, rec_matrix_lr=0.025):

| Config | Params | step 100 | step 200 | step 300 | step 400 | Δ vs baseline | ms/step |
|---|---|---|---|---|---|---|---|
| baseline | 26.99M | 2.2092 | 2.0023 | 1.9076 | **1.8781** | — | 336 |
| **rec_012** | 27.78M (+2.9%) | 2.0859 | 1.8266 | 1.7261 | **1.6966** | **-0.182** | 390 (+16%) |

- **rec_matrix_lr sweep** (rec_012 @ 300 iters, finding right LR for recurrent Muon banks):
  - 0.025 → 2.0071 (best, -0.109 vs baseline 2.1163)
  - 0.0125 → 2.2125 (+0.096)
  - 0.006 → 2.3383 (+0.222)
  - 0.003 → 2.3892 (+0.273)
  - 0.001 → 2.4107 (+0.294)
- **Use matrix_lr for rec banks** — lower LRs undertrain. Same LR as main attention banks is optimal.

- **Key findings**:
  - Gap WIDENS over training (step 100: -0.123, step 400: -0.182) — hybrid improves faster and keeps pulling ahead
  - Earlier layers benefit more (layer 0 >> layer 8)
  - Contiguous early beats alternating (rec_012 > rec_0246)
  - +16% step time overhead for -0.182 BPB gain
  - Works even stacked on top of SOTA's Bigram/SmearGate/XSA/VE/LN-scale features

- **Implementation caveats**:
  - GatedRecurrence must be stateless (weights from banks), NOT per-layer nn.Linear modules — otherwise they go to AdamW scalar group and undertrain
  - Rec banks need their own Muon param group (same matrix_lr is fine)
  - Zero-init the `rec_out_bank` (like attention's proj) — critical for stable start
  - Unbank/rebank state dict helpers need NotImplementedError guard for recurrent layers until GPTQ path is extended

- **Full 10-min 8xH100 run** (SOTA config, iterations=20000, batch=786K, full LR schedule):

| Config | Final step | Val BPB | Post-EMA | ms/step |
|---|---|---|---|---|
| baseline | 5457 | 1.1491 | **1.1483** | 110 |
| rec_012 | 4509 | 1.1975 | 1.1967 | 133 (+21%) |

- **Hybrid LOSES by +0.048 BPB at full training.** All earlier "wins" were bogus due to warmdown schedule collapse:
  - Our A/B tests used `iterations=300-400` with `warmdown_iters=3500` (default).
  - Formula: `warmdown_start = max(iterations - warmdown_iters, 0) = 0` → entire run is in warmdown.
  - At step 100 of a 400-iter run: `lr_mul = 300/3500 = 0.086` → effective LR = 0.00215 (50× lower than production).
  - Plus batch_tokens was 131K vs production 786K (6× smaller).
  - Combined: short runs had ~70× less learning signal per step.
  - **At artificially low LR, GatedRecurrence with zero-init out_proj drifts slower and looks "ahead" — but this is near-init noise, not real learning advantage.**
  - At real LR, attention learns faster than gated recurrence, and hybrid just costs +21% step time for no gain.
- **Lesson: short-iteration A/B tests with default warmdown are unreliable.** For meaningful screening at short iterations, set `WARMDOWN_ITERS` to a small fraction of `ITERATIONS` (e.g., 100 for a 500-step run), or run at full config length.
- **Decision**: drop the hybrid approach. GatedRecurrence clean refactor stays in `train_gpt_sota.py` behind `RECURRENT_LAYERS` env var for future exploration but not used by submissions.

### Bottleneck MLP ❌ | **Novel**
- Replace standard MLP (512->1024->512) with bottleneck (512->128->1536->512), same param count
- Two nonlinearities instead of one, 50% wider hidden layer
- Result: **+0.031 BPB worse** at 500 steps, 15% slower
- Information bottleneck at 128 dims too restrictive for compression task

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

---

## Priority Stack

1. **Increase training signal** — Reduce prefix_len (128 or 64), predict more tokens per step. This is the #1 bottleneck.
2. **Longer training run** — 80 min on 1xH100 to see if the architecture converges to competitive BPB given enough steps
3. **Prefix summary experiment** — Add 16th global token from prediction prefix (mean-pool + Linear)
4. **Width experiments** — Try wider MLP (mlp_mult=4) within artifact budget
5. **GPTQ integration** — Adapt `gptq_quantize.py` for hierarchical model
6. **FP8 training** — Faster steps → more training in 10 min
7. **Differentiable codebook** — Train with learned quantization end-to-end

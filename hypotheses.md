# Parameter Golf — Comprehensive Experiment Tracker

## Status: ✅ Works — ❌ Doesn't work — 🔬 Testing — 💡 Untested — 🤔 Speculative
## Novelty: **Novel** = untried in competition — **Known** = tried by others

---

## Current Best Results

| Config | BPB (roundtrip) | Artifact | Steps | GPU | Notes |
|--------|----------------|----------|-------|-----|-------|
| **Full stack single** (13L+XSA-all+SSE+1cycle+seq2048) | **1.3575** | 14.05MB FITS | 500 | 1×A100 | Best budget-fit single model |
| Full stack 1000 steps | **1.2901** (pre-quant) | — | 1000 | 1×H100 | 🔬 Running to 2000 |
| 2×7L ensemble + SSE | 1.4057 | 15.5MB FITS | 500 | 1×A100 | Best budget-fit ensemble |
| 4×7L ensemble | 1.3880 | ~30MB OVER | 500 | 1×A100 | Best ensemble absolute (500 steps) |
| 2×7L ensemble | **1.2778** | 21.6MB OVER | 2000 | 1×A100 | Best absolute ever |

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

### Codebook / Non-Uniform Quantization 💡 Untested | **Novel**
- **Current int6**: uniform per-row scaling. `w_recon = q * scale` — evenly spaced values.
- **Nobody in competition does this** — all use uniform per-row scaling.

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

**Next step**: Train with σ-regularization on MLX smoke test, capture pre-quantization float weights, compare uniform vs Lloyd-Max roundtrip BPB gap.

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

### GPU Benchmarks (full stack: 13L XSA-all SSE 1cycle seq2048 batch786K)
- A100 SXM4 80GB: 1950ms/step (compiled)
- H100 NVL 96GB: 1592ms/step (compiled, 18% faster)
- Estimated 8×H100 SXM: ~200ms/step → ~3000 steps in 600s

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

## Wild Ideas Explored

### Random Seed Weight Search ❌
- Tested 20 seeds for untrained models: all give 4.178 ± 0.003 BPB
- No magic seeds — model needs training regardless of init

### Fractal/Structured Weight Generation 🤔 Unexplored
- Tiny deterministic program generates weight structure, store only corrections
- Could compress beyond int6 if structure captures weight patterns

### Growing Model at Eval Time 💡
- Legal: artifact is 16MB, but eval code can create larger model
- Add LoRA adapters at eval → TTT trains them on 62M val tokens
- Spawn N copies with different LoRA → ensemble at eval → no extra artifact cost

---

## Priority Stack for Next Session

1. **Test federated UT ensemble** 💡 — `train_fed_ut_ensemble.py` on A100. K=5, r128, 3×4 UT, 500 steps.
2. **Codebook quantization** 💡 — Per-matrix Lloyd-Max centroids. Should reduce roundtrip gap for ~25KB.
3. **TTT implementation** 💡 — #1 eval technique. 62M val tokens = ample data. Per-document LoRA.
4. **8×H100 deployment** — Run federated UT ensemble at scale. Fix NCCL if needed.
5. **Sliding window eval** 💡 — stride=64, used by SOTA, free +0.005 BPB
6. **Trim code for submission** — Remove unused features to shrink code size in artifact
7. **Fix LoRA crash** — For frozen-base compression experiments

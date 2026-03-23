# Ternary Neural Network Approach for Parameter Golf

## Thesis

The current leaderboard trend is clear: reducing weight precision frees storage budget
for more parameters, and more parameters improve compression quality. We extrapolate
this to the limit: **ternary weights {-1, 0, 1}** trained from scratch, paired with an
architecture designed for ternary from the ground up.

At 1.58 bits per weight (log2(3)), we get roughly **5x more parameters** than int8 within
the same 16MB artifact budget. The key question is whether ternary parameters carry
enough information to outweigh the per-parameter quality loss — especially at the small
model scales this challenge operates at.

## 1. Architecture: Gated Recurrence + Gated MLP (No Attention)

### Why not transformers?

Softmax attention is the weakest link under extreme quantization. It computes relative
magnitudes between query-key dot products — binarizing/ternarizing these destroys the
ordering that makes attention useful. Research confirms the robustness ranking under
ternary weights:

1. MatMul-free (MLGRU + BitLinear) — designed for ternary, best results
2. SSMs (Mamba) — 1-bit Mamba matches 8-bit Transformers
3. RNNs (GRU/RWKV) — multiply-accumulate becomes pure accumulation
4. Transformers + BitNet mods — viable but softmax remains the weak link

### Proposed architecture

We adopt a structure inspired by the MatMul-free LLM (Zhu et al., NeurIPS 2024), adapted
to our tiny-model regime:

```
Input tokens
    |
[Embedding]          (full-precision, tied with output head)
    |
[RMSNorm]
    |
    v
+---------------------------+
| Block (repeated N times)  |
|                           |
|  [TernaryGatedRecurrence] |  <-- token mixing (replaces attention)
|  [RMSNorm]                |
|  [TernaryGatedMLP]        |  <-- channel mixing
|  [RMSNorm]                |
|                           |
+---------------------------+
    |
[Final RMSNorm]
    |
[Output projection]  (tied with embedding)
    |
Cross-entropy loss
```

#### Token Mixer: Gated Recurrence

Replace attention with a minimal gated recurrent unit. With ternary weights, the
matrix-vector products become sign-flip-and-accumulate operations.

```
# Simplified MLGRU-style token mixer
forget = sigmoid(TernaryLinear(x))     # forget gate
candidate = TernaryLinear(x)           # candidate state (no activation needed)
h = forget * h_prev + (1 - forget) * candidate
output = TernaryLinear(h)
```

Key properties:
- **No softmax** — the forget gate uses sigmoid, which is smooth and quantization-friendly
- **O(1) memory per step** at inference (vs O(n) for attention)
- **Naturally causal** — no attention mask needed
- With ternary weights, the matrix multiplications become additions/subtractions only

#### Channel Mixer: Gated MLP

Similar to the existing ReLU^2 MLP but with a gate for better information flow:

```
gate = sigmoid(TernaryLinear(x))
value = TernaryLinear(x)
output = TernaryLinear(gate * value)
```

The gating is important for ternary networks because it provides a continuous
modulation mechanism that compensates for the discrete weight values.

#### Why keep embeddings full-precision?

The embedding table is a small fraction of total parameters (vocab 1024 * dim = ~0.5M
params) but critically determines the input representation quality. BitNet and all
successful binary/ternary LLMs keep embeddings at higher precision. We tie input/output
embeddings and store them in fp16, same as the baseline.

### Parameter Budget

Current baseline (int8 after quantization):
- ~16M parameter budget at 1 byte each
- 9 layers, dim 512: roughly 14M effective parameters

Ternary approach (1.58 bits per weight):
- Optimal packing: ~81M parameters in 16MB
- Practical 2-bit packing: ~64M parameters in 16MB
- With zlib on sparse ternary (many zeros): somewhere in between, ~70M
- Even at 4x the baseline parameter count, we get ~56M usable ternary parameters

This lets us significantly increase model width, depth, or both:
- **Option A: Go wide** — dim 1024-1536, 9 layers (~36-80M ternary params)
- **Option B: Go deep** — dim 768, 18+ layers (~54M ternary params)
- **Option C: Balanced** — dim 1024, 12 layers (~50M ternary params)

Going wide is likely better for ternary: wider layers mean more ternary weights participate
in each operation, giving a better "law of large numbers" averaging effect that compensates
for per-weight imprecision.

## 2. Training: Ternary Weights from Scratch

### Approach A: Shadow Weights + STE (Pragmatic Baseline)

The proven approach from BitNet. During training, maintain full-precision "shadow weights"
that accumulate gradients. Ternarize them for the forward pass:

```python
def ternarize(w):
    """Absmean quantization from BitNet b1.58"""
    scale = w.abs().mean()
    return torch.clamp(torch.round(w / scale), -1, 1), scale

class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Shadow weights: full precision, used for gradient accumulation
        self.shadow = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def forward(self, x):
        w_ternary, scale = ternarize(self.shadow)
        # STE: forward uses ternary, backward flows through as if identity
        w = self.shadow + (w_ternary * scale - self.shadow).detach()
        return F.linear(x, w)
```

The Straight-Through Estimator (STE) passes gradients through the non-differentiable
`round()` as if it were the identity function. This is mathematically questionable but
empirically proven at scale.

**Pros**: Well-understood, proven to work, easy to implement.
**Cons**: Shadow weights double memory during training (not an issue — training memory
isn't constrained, only the final artifact is). The STE gradient is biased.

### Approach B: Metaplastic Integer Counters (Bio-Inspired)

Instead of full-precision shadows, maintain a small integer counter behind each weight.
Inspired by Colombo et al. (2024) and biological synaptic metaplasticity:

```python
class MetaplasticTernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, counter_bits=8):
        super().__init__()
        self.max_val = 2 ** (counter_bits - 1) - 1
        # Integer counters: sign determines weight, magnitude = confidence
        self.counters = nn.Parameter(
            torch.randint(-3, 4, (out_features, in_features)).float()
        )

    def forward(self, x):
        w = torch.sign(self.counters)  # {-1, 0, 1}
        return F.linear(x, w)

    def update(self, grad, lr):
        # Probabilistic flipping: gradient magnitude = flip probability
        flip_prob = (grad.abs() * lr).clamp(0, 1)
        flip_mask = torch.bernoulli(flip_prob).bool()
        with torch.no_grad():
            # Nudge counters toward the negative gradient direction
            nudge = -torch.sign(grad)
            self.counters[flip_mask] += nudge[flip_mask]
            self.counters.clamp_(-self.max_val, self.max_val)
```

The counter serves dual purposes:
- **Sign** → the visible ternary weight
- **Magnitude** → confidence / resistance to flipping (metaplasticity)

A weight with counter=+50 is very confident in being +1 and won't flip easily.
A weight with counter=+1 is uncertain and may flip to 0 or -1 on the next update.
A weight with counter=0 is silent (pruned).

This is biologically plausible:
- Biological synapses are excitatory (+1), inhibitory (-1), or silent (0) — ternary
- Synaptic metaplasticity: frequently-used synapses resist change
- Learning is noisy/probabilistic — matching the stochastic flip
- No backpropagation needed through the discrete function — the counter update is local

### Approach C: Hebbian + Modulated Learning (Most Bio-Inspired)

Three-factor Hebbian rule: `delta_w ~ pre * post * modulator`

```python
def hebbian_update(pre_act, post_act, global_error, lr):
    """
    pre_act:      activations entering this layer
    post_act:     activations leaving this layer
    global_error: scalar loss signal (broadcast to all weights)
    """
    # Outer product gives Hebbian term, modulated by error
    hebbian = torch.sign(post_act.T @ pre_act)  # already ternary!
    return hebbian * global_error * lr
```

This eliminates backpropagation entirely. The global error signal (the loss value)
modulates local Hebbian updates — analogous to dopamine modulating synaptic plasticity.

**Pros**: No backward pass (faster training), biologically principled, naturally produces
ternary updates.
**Cons**: Much weaker learning signal than backprop. May not converge to competitive
quality. Best used as a secondary exploration after A/B are established.

### Recommended Training Strategy

1. **Start with Approach A** (shadow weights + STE). This is the proven path and gives
   us a reliable baseline for the ternary architecture.
2. **Experiment with Approach B** (metaplastic counters) as an alternative optimizer.
   If it matches Approach A, it's preferable because the counter-based representation
   naturally maps to the final ternary weights with no quantization step.
3. **Explore Approach C** (Hebbian) only if we have time and curiosity. It's the most
   novel but highest risk.

### Optimizer Choice

For Approach A, the existing **Muon optimizer** may not be ideal — its Newton-Schulz
orthogonalization is designed for continuous weight matrices. Options:

- **Adam on shadow weights** → simple, well-understood, likely sufficient
- **Muon on shadow weights** → might still help with matrix-shaped updates, but the
  orthogonalization interacts oddly with ternarization
- **Sign-SGD** → only uses the sign of the gradient, naturally complementary to ternary.
  Has been shown to work well in low-precision regimes.

Recommendation: start with Adam, try SignSGD as a comparison.

### SubLN: The Critical Normalization Trick

BitNet's key architectural addition: extra RMSNorm layers before each ternary projection.
Without SubLN, ternary training diverges. The norm stabilizes the input distribution to
the ternary layers, ensuring the absmean quantization produces meaningful values.

```python
class TernaryBlock(nn.Module):
    def forward(self, x, h_prev):
        # Token mixing with SubLN
        x_normed = rms_norm(x)
        h = self.gated_recurrence(x_normed, h_prev)  # ternary weights inside
        x = x + self.token_scale * rms_norm(h)        # SubLN after ternary op

        # Channel mixing with SubLN
        x_normed = rms_norm(x)
        mlp_out = self.gated_mlp(x_normed)            # ternary weights inside
        x = x + self.channel_scale * rms_norm(mlp_out) # SubLN after ternary op
        return x, h
```

### Activation Precision

BitNet uses 8-bit activations even with ternary weights. For training we keep activations
in bfloat16 (same as baseline). The activations provide the continuous "resolution" that
ternary weights lack — the combination of discrete weights and continuous activations is
what makes the system expressive.

## 3. Artifact Packing

The final artifact must be <= 16,000,000 bytes (code + compressed model).

### Ternary Weight Encoding

Each ternary weight is one of {-1, 0, 1}. Encoding options:

**2-bit packing** (simplest):
- Map: -1 → 0b00, 0 → 0b01, 1 → 0b10
- 4 weights per byte → 50% of int8 size
- 64M weights in 16MB

**Balanced ternary + entropy coding** (optimal):
- Theoretical minimum: log2(3) = 1.58 bits per weight
- Pack 5 ternary values into 8 bits (3^5 = 243 < 256): 1.6 bits/weight
- Or use arithmetic coding to approach 1.58 bits/weight
- ~81M weights in 16MB

**Sparse encoding** (if many zeros):
- If the trained model is 50%+ zeros, run-length encode the zero positions
- Then pack only the +1/-1 values (1 bit each)
- Compress with zlib on top
- Could be very efficient depending on sparsity pattern

Recommendation: start with 2-bit packing + zlib, which is simple and already gives ~4x
over int8. Optimize encoding later if needed.

### Embedding Storage

Embeddings stay fp16. With vocab 1024 and dim D:
- 1024 * D * 2 bytes = 2048*D bytes
- At dim 1024: ~2MB for embeddings (still leaves ~14MB for ternary weights)

### Scale Factors

Each TernaryLinear needs a per-layer scale factor (the absmean value from ternarization).
These are stored as fp16 — negligible space (one value per layer).

## 4. Local Evaluation Plan

### Setup

All experiments run on Apple Silicon via the MLX training script. We compare against the
existing baseline to measure the ternary approach's effectiveness.

### Quick Comparison Protocol

**Baseline reference (existing architecture, ~5 min):**
```bash
# Download minimal data (1 shard)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# Run baseline for 200 steps
RUN_ID=baseline_ref \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=200 \
VAL_BATCH_SIZE=8192 \
VAL_MAX_TOKENS=65536 \
python3 train_gpt_mlx.py
```

**Ternary experiment (new architecture, same budget):**
```bash
# Same data, same iteration count, comparable total parameter bytes
RUN_ID=ternary_v1 \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=200 \
VAL_BATCH_SIZE=8192 \
VAL_MAX_TOKENS=65536 \
python3 train_ternary_mlx.py
```

### What to Measure

1. **Training loss curve** — does the ternary model learn at all? Does the loss decrease
   smoothly or is it noisy/unstable? The first 50 steps will tell us a lot.

2. **Val BPB at step 200** — direct comparison with baseline. We don't expect the ternary
   model to win at 200 steps, but we want to see the gap. A ternary model within 20% of
   the baseline at 200 steps would be very encouraging.

3. **Parameter count vs artifact size** — verify that we actually get more parameters
   per byte. Compute: `(ternary_params * 1.58 / 8) + embedding_bytes` and compare with
   `baseline_params * 1` (int8).

4. **Training throughput** — tokens/second. Ternary forward passes should be faster
   (additions instead of multiplications), but the STE backward pass may negate this.
   On MLX/Apple Silicon the speedup may not materialize since the hardware is optimized
   for dense fp16/bf16 matmuls.

### Incremental Experiments

Run these in order, each building on the previous:

**Experiment 1: Ternary weights in existing architecture**
Keep the current GPT architecture but replace nn.Linear with TernaryLinear (shadow
weights + STE). This isolates the effect of ternary weights from the architecture change.
If this already works reasonably, the architecture change is purely additive.

```bash
# Just swap weight type, keep transformer
RUN_ID=exp1_ternary_transformer ITERATIONS=200 ...
```

**Experiment 2: Gated recurrence with full-precision weights**
Keep full-precision weights but replace attention with gated recurrence. This isolates
the architecture change from the weight type change.

```bash
# New architecture, normal weights
RUN_ID=exp2_gru_fp ITERATIONS=200 ...
```

**Experiment 3: Ternary + gated recurrence (the full approach)**
Both changes combined. Compare with experiments 1 and 2 to see if they compose well.

```bash
# Both changes
RUN_ID=exp3_ternary_gru ITERATIONS=200 ...
```

**Experiment 4: Width scaling**
Once experiment 3 works, scale up model width to use the ternary storage advantage.
Double or triple the dimension while keeping artifact size comparable to the baseline.

```bash
# Wider ternary model
RUN_ID=exp4_ternary_wide MODEL_DIM=1024 ITERATIONS=200 ...
```

**Experiment 5: Metaplastic counters (Approach B)**
Replace STE with the metaplastic integer counter update. Compare training dynamics
with Approach A.

```bash
# Alternative optimizer
RUN_ID=exp5_metaplastic ITERATIONS=200 ...
```

### Smoke Test Checklist

Before a longer run, verify each experiment passes:

- [ ] Loss decreases in the first 10 steps (model is learning)
- [ ] No NaN/Inf in loss or gradients
- [ ] Ternary weights actually contain only {-1, 0, 1} after ternarization
- [ ] Scale factors are reasonable (not exploding or vanishing)
- [ ] Memory usage is acceptable for local machine (< 8GB for smoke test)

### Longer Validation Run

Once a configuration looks promising in the 200-step smoke test:

```bash
# 2000-step run with periodic validation (~30 min on M-series Mac)
RUN_ID=ternary_long \
ITERATIONS=2000 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=200 \
VAL_BATCH_SIZE=8192 \
VAL_MAX_TOKENS=65536 \
python3 train_ternary_mlx.py
```

At 2000 steps with 8192 tokens/step = 16M training tokens. The full baseline sees
~10B tokens, so 2000 steps is a very rough proxy — but sufficient to see whether the
loss curve is converging to a competitive range.

## 5. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Ternary models don't work at small scale | Medium | Fatal | Experiment 1 tests this early; abandon if loss doesn't decrease |
| Gated recurrence underperforms attention for compression | Medium | Major | Experiment 2 tests this in isolation |
| STE gradients cause training instability | Low | Major | SubLN normalization; gradient clipping; fallback to SignSGD |
| Ternary weights don't compress well with zlib | Low | Moderate | 2-bit packing is already ~4x; sparse encoding if needed |
| Training is too slow on MLX | Low | Minor | Reduce model size for smoke tests; full runs on GPU |
| The combination doesn't compose (ternary + new arch) | Medium | Major | Experiments 1 & 2 test each factor independently first |

## 6. Key References

- **BitNet b1.58** (Ma et al., 2024): Ternary {-1,0,1} transformers matching FP16 at 3B params
- **MatMul-free LLM** (Zhu et al., NeurIPS 2024): MLGRU + BitLinear, no matrix multiply needed
- **Bi-Mamba** (2024): 1-bit SSMs outperforming 1-bit transformers
- **BinaryConnect** (Courbariaux & Bengio, 2015): Shadow weights + stochastic binarization
- **Colombo et al.** (2024): Local binary error signals with metaplastic integer weights
- **BEP** (Colombo et al., 2025): Binary error propagation with Hebbian outer-product updates
- **Equilibrium Propagation** (Scellier & Bengio, 2017): Bio-plausible training of binary nets
- **Laydevant et al.** (2021): EP training binary weight networks, achieving 98.86% MNIST

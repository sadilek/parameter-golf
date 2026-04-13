# Novel Ideas from Nemotron 3

Sources:
- [NVIDIA Nemotron 3 Nano Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf) (Dec 2025)
- [NVIDIA Nemotron 3 White Paper](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-White-Paper.pdf) (Dec 2025) — covers Nano, Super, and Ultra

---

## 1. Hybrid Mamba-Transformer Architecture

**Description:** Instead of a pure transformer, Nemotron 3 Nano interleaves Mamba-2 SSM (state-space model) layers with GQA attention layers in a specific repeating pattern. The pattern is: `[Mamba-MoE, Mamba-MoE, Attn-MoE] x5, [Mamba-MoE, Mamba-MoE] x3, [Mamba-MoE, Attn-MoE] x1, [Mamba-MoE] x4` — totaling 52 layers with only 6 attention layers out of 52. Each Mamba-2 layer has state dimension 128, 8 groups, 64 heads, and head dimension 64. The attention layers use 32 Q-heads and 2 KV-heads (16:1 GQA ratio).

**Problem solved:** Pure attention is O(n^2) in sequence length and requires large KV caches. Mamba layers provide O(n) recurrent processing with fixed-size state, enabling 3.3x higher inference throughput while matching accuracy. The few attention layers (6 out of 52) provide the global reasoning capabilities that pure SSMs lack.

### Applicability to Parameter Golf

**Relevance: MEDIUM-HIGH.** Mamba-2 layers are significantly more parameter-efficient than attention for modeling sequential dependencies — they don't need separate Q/K/V projections and have no KV cache overhead. For our 16MB artifact constraint, replacing some attention layers with Mamba could free parameters for a wider model or more layers.

**Challenges:**
- Mamba-2 requires custom CUDA kernels (`mamba_ssm` package). We'd need to include this in the submission or reimplement it in pure PyTorch.
- The SOTA script already uses XSA (extended self-attention) on all 11 layers — switching some to Mamba would be a major architectural change.
- Mamba layers have different optimization dynamics than attention; Muon optimizer may need tuning.
- The artifact must be self-contained — any Mamba kernel code adds to the 16MB budget.
- Our sequences are only 2048 tokens — the O(n) vs O(n^2) advantage is modest at this length.

**Verdict:** Interesting for parameter efficiency, but the implementation complexity and kernel requirements make it hard to integrate within competition constraints. Would need a pure-PyTorch Mamba implementation that fits in the code budget.

---

## 2. Granular Mixture-of-Experts (MoE) with Shared Experts

**Description:** Each MoE FFN layer has 128 total routable experts with granular (small) expert dimension 1856, activating only 6 per token plus 2 always-on shared experts. The router is a learnt MLP with sigmoid gating (not the typical top-k softmax). This gives 31.6B total parameters but only 3.2B active per forward pass (10% activation ratio). The MoE uses squared ReLU activation internally.

**Problem solved:** MoE decouples total model capacity from per-token compute cost. The granular design (many small experts vs. few large ones) combined with shared experts improves routing quality and ensures every token gets baseline capability from the shared experts.

### Applicability to Parameter Golf

**Relevance: LOW.** Our constraint is artifact size (16MB compressed), not compute. MoE increases total parameters — the opposite of what we need. Even though only 6/128 experts are active during inference, ALL 128 experts must be stored in the artifact. A model with 10x the stored parameters would blow past 16MB instantly.

**However**, a micro-scale MoE (e.g., 4 experts, 2 active) could potentially improve quality within a parameter budget by specializing different experts for different token patterns. The overhead would be the router weights + expert duplication. At our ~27M param scale, this overhead is significant.

**Verdict:** Not applicable. MoE makes total param count larger, which is the wrong direction for a size-constrained competition. The shared expert concept is interesting but adds parameters.

---

## 3. No Positional Embeddings (Position-Free Design)

**Description:** Nemotron 3 Nano uses NO positional embeddings at all — no RoPE, no ALiBi, no learned position embeddings. The Mamba layers naturally encode position through their recurrent state, and the few attention layers apparently learn positional patterns from the Mamba context.

**Problem solved:** Eliminates the parameter cost of positional encoding and removes the hard context-length ceiling that comes with fixed-position schemes. Supports context lengths up to 1M tokens without position extrapolation issues.

### Applicability to Parameter Golf

**Relevance: LOW for our current architecture.** Our SOTA model uses Partial RoPE (16/64 dims), which is already very lightweight. Removing RoPE entirely would save negligible parameters but could hurt the model's ability to distinguish positions in a pure-attention architecture. The position-free design only works because Mamba layers inherently encode position — without Mamba, we'd lose positional information entirely.

**If we adopted Mamba layers** (Idea #1), dropping RoPE could be viable. But standalone, this isn't useful.

**Verdict:** Only applicable if combined with Mamba hybrid architecture. Not independently useful.

---

## 4. Warmup-Stable-Decay (WSD) Learning Rate Schedule

**Description:** A three-phase LR schedule: (1) Warmup over 8.4B tokens to max LR of 10^-3, (2) Stable phase at max LR for 80% of training (20T tokens), (3) Decay to 10^-5 over last 20% (5T tokens). Used with AdamW (beta1=0.9, beta2=0.95, WD=0.1).

**Problem solved:** Compared to cosine schedules, WSD keeps the model at peak learning rate for much longer, enabling more effective learning during the bulk of training. The long stable phase lets the model explore the loss landscape more aggressively before settling.

### Applicability to Parameter Golf

**Relevance: MEDIUM.** Our SOTA uses a warmup + cosine warmdown schedule, which is conceptually similar. The key difference is that WSD maintains peak LR for 80% of training whereas cosine decay starts declining immediately after warmup. 

For our 7000-step training, this translates to: warmup ~100 steps, stable at peak LR until ~5600 steps, then decay for the final ~1400 steps. This might be better than our current warmdown=4000 approach which starts decaying at step 3000.

**Caveats from our logbook:** "1cycle LR destabilizes at long training." WSD is essentially a longer version of 1cycle's high phase. Careful tuning needed.

**Verdict:** Worth testing. A simple schedule change with no parameter cost. Could be combined with our Muon optimizer (which may respond differently than AdamW to schedule shape).

---

## 5. Two-Phase Pretraining Data Curriculum

**Description:** Training is split into two phases: Phase 1 (first 94% of training) uses a diverse data mixture emphasizing coverage, Phase 2 (final 6%) shifts to high-quality data sources (Wikipedia, curated crawl, SFT-style data). The transition happens at 94% of the total token budget.

**Problem solved:** Early training benefits from diversity to learn broad patterns; late training benefits from high-quality data to refine the model's distribution toward the target quality level. This avoids the trade-off between data diversity and data quality.

### Applicability to Parameter Golf

**Relevance: LOW-MEDIUM.** We train on a fixed FineWeb dataset with no control over data quality tiers. The FineWeb shards are pre-shuffled and we iterate through them sequentially. We don't have quality labels on individual documents.

**However**, we could create a simple two-phase approach: Phase 1 trains on all shards, Phase 2 replays the first N shards (which have different content distribution than later shards). Or we could implement a data quality heuristic (e.g., document length, token entropy) and upweight "better" documents in the final phase.

**Verdict:** Marginal applicability. Would require adding data quality scoring to a competition where the data pipeline is standardized.

---

## 6. Aux-Loss-Free MoE Load Balancing

**Description:** Instead of using only an auxiliary loss to encourage balanced expert utilization, they combine DeepSeek's aux-loss-free approach (which uses an EMA-updated bias term with update rate 10^-3 to adjust routing scores) with a traditional load-balancing loss at a small coefficient (10^-4). This avoids the representation collapse that strong aux losses cause while still maintaining basic balance.

**Problem solved:** Standard MoE load balancing losses (which penalize uneven expert usage) interfere with the primary training objective and can cause experts to become homogeneous. The bias-based approach adjusts routing without contaminating gradients.

### Applicability to Parameter Golf

**Relevance: ONLY IF WE USE MoE.** Since MoE is unlikely to help us (see Idea #2), this is not directly applicable. If we did explore micro-MoE, this technique would be the right way to balance it.

**Verdict:** Not applicable to our current architecture.

---

## 7. Selective Post-Training Quantization

**Description:** Rather than uniformly quantizing the entire model to FP8, they perform sensitivity analysis and keep the most sensitive components in BF16: all 6 attention layers + the 6 Mamba layers immediately preceding them + Conv1D within all Mamba layers. Everything else (including KV cache) is quantized to FP8. This achieves 99% median accuracy recovery with 3.5x throughput improvement.

**Problem solved:** Uniform quantization often causes disproportionate accuracy loss in a few critical layers. By keeping ~23% of layers in BF16 and quantizing the rest, they get most of the throughput benefit with minimal accuracy cost.

### Applicability to Parameter Golf

**Relevance: HIGH.** We already use GPTQ for quantization. The key insight — that different layers have very different quantization sensitivity — is directly applicable. Our SOTA uses uniform int8 GPTQ across all layers. We could:

1. **Run per-layer sensitivity analysis** on our 11-layer model to identify which layers hurt most from quantization.
2. **Use mixed-precision GPTQ**: higher precision (more bits) for sensitive layers, lower precision for robust layers. E.g., int8 for the first/last layers and int6 for middle layers, or varying group sizes.
3. **Keep the U-Net skip connection layers at higher precision** since they fuse information across the depth of the network.

**Known from our experiments:** "GPTQ gap is +0.008 BPB for uniform int6" and "mixed int3 still hurts (+0.18)". A smarter allocation across layers could reduce the GPTQ gap below +0.008.

**Verdict:** Directly applicable. Per-layer quantization sensitivity analysis could shave a few thousandths off BPB.

---

## 8. Squared ReLU Activation in FFN

**Description:** All MoE layers use squared ReLU: `relu(x)^2`. Combined with the granular expert design.

**Problem solved:** Squared ReLU creates sparser activations than standard ReLU or GELU, which can improve model capacity per parameter and helps with quantization (fewer large outliers).

### Applicability to Parameter Golf

**Relevance: ALREADY IMPLEMENTED.** Our SOTA already uses `LeakyReLU(0.5)^2` which is a variant of squared activation. The baseline uses `ReLU^2`. This confirms we're on the right track.

**Verdict:** No action needed — already using this.

---

## 9. Un-Tied Embedding and Output Projection

**Description:** Nemotron 3 Nano explicitly un-ties the token embedding and the output projection (lm_head) weights. They are separate learned parameters.

**Problem solved:** Tied embeddings force the same representation to serve dual purposes (input token features and output token predictions). Un-tying allows each to specialize, often improving quality at the cost of extra parameters.

### Applicability to Parameter Golf

**Relevance: LOW-MEDIUM.** Our baseline uses tied embeddings by default (`tie_embeddings=1`) for parameter efficiency. With vocab_size=1024 and model_dim=512, the embedding matrix is only 1024x512 = 524K params (~1MB). Un-tying would add another 524K params (~2% overhead on our 27M model).

The SOTA submission does explore this trade-off. At our small vocab size (1024), the cost of un-tying is small. But every extra MB counts toward the 16MB artifact limit. The GPTQ quantization would compress these params too.

**Verdict:** Marginal. The parameter budget increase is small but the benefit at our scale is uncertain. Our SOTA already explored this.

---

## 10. Gaussian Curriculum Sampling for RL Training

**Description:** For multi-environment RL training, they model the target pass-rate distribution as a Gaussian function per domain. Early in training, the Gaussian is centered on high pass-rate (easy examples); it linearly shifts toward low pass-rate (hard examples) as training progresses. This ensures the model always trains on examples at the frontier of its capability.

**Problem solved:** Random sampling wastes compute on too-easy or too-hard examples. Curriculum learning that always matches difficulty to current capability is more sample-efficient.

### Applicability to Parameter Golf

**Relevance: LOW.** We don't do RL training. However, the curriculum concept could apply to language modeling: sequence difficulty varies, and training on progressively harder sequences could improve efficiency. But measuring "difficulty" for language modeling sequences is non-trivial (unlike RL where you have pass/fail signals).

**Possible adaptation:** Order FineWeb shards by perplexity under a small model, train on low-perplexity (easy) shards first, hard shards later. But this adds significant preprocessing overhead.

**Verdict:** Interesting concept but impractical for our competition setup.

---

## 11. GRPO with Masked Importance Sampling and Frozen Router

**Description:** They use synchronous GRPO (Group Relative Policy Optimization) with masked importance sampling to mitigate training-inference misalignment. During RL, they freeze the MoE router weights to stabilize training. They use 128 prompts/step, 16 generations/prompt, batch size 2048, and on-policy updates.

**Problem solved:** RL training with MoE is unstable because router updates during RL can cause expert collapse. Freezing routers during RL and using masked importance sampling prevents this.

### Applicability to Parameter Golf

**Relevance: NONE.** We don't use RL or MoE. This is specific to the post-training pipeline.

**Verdict:** Not applicable.

---

## 12. Group Relative Length Control (GRLC)

**Description:** During RLHF, they decompose each response into a "thinking" part and an "answer" part. For each part, they compute a zero-mean, group-relative length penalty: `w_i = 1 - (l_i - l_min) / (l_max - l_min)`, centered across the group. The final reward is: `R_i = R_base + lambda_think * w_think + lambda_answer * w_answer`. This encourages conciseness without an absolute length penalty.

**Problem solved:** RLHF models tend to produce verbose outputs. Absolute length penalties bias against inherently complex problems. Group-relative penalties only penalize being longer than peers answering the same question.

### Applicability to Parameter Golf

**Relevance: NONE.** Not applicable to language model pretraining for compression.

**Verdict:** Not applicable.

---

## 13. Quality-Gated Conciseness Bonus

**Description:** An optional bonus added to the shortest response in a group, but ONLY if its quality score exceeds the p-th percentile (tau_p, set to 80th percentile). This prevents the model from learning to produce short but low-quality answers.

**Problem solved:** Naive length penalties encourage short, bad answers. Quality gating ensures conciseness is only rewarded when quality is preserved.

### Applicability to Parameter Golf

**Relevance: NONE.** Specific to RLHF training, not applicable to our setting.

**Verdict:** Not applicable.

---

## 14. Circular Comparison Strategy for GenRM

**Description:** Instead of comparing all N^2 pairs of responses for reward modeling (120 comparisons for N=16), they compare each response only with its successor in a ring: (r1,r2), (r2,r3), ..., (rN,r1). This gives exactly N comparisons (O(N) not O(N^2)) while still connecting all responses in a comparison graph. Each response appears twice (in different positions) to reduce positional bias.

**Problem solved:** Quadratic scaling of pairwise comparisons is prohibitively expensive for large N. Ring comparison maintains connectivity with linear cost.

### Applicability to Parameter Golf

**Relevance: NONE.** Specific to reward model training.

**Verdict:** Not applicable.

---

## 15. Overlong Filtering During RL

**Description:** During RL training, they filter out responses that exceed the maximum generation length of 49K tokens. They find this boosts performance on reasoning-intensive benchmarks.

**Problem solved:** Extremely long responses during RL training waste compute and provide noisy reward signals.

### Applicability to Parameter Golf

**Relevance: NONE.** Not applicable to our training setup.

**Verdict:** Not applicable.

---

## 16. Sigmoid-Gated MLP Router for MoE

**Description:** Unlike the standard top-k softmax router, they use a learnt MLP router with sigmoid gating. Each token's routing score for each expert is computed independently via sigmoid, then the top-6 are selected. This allows more flexible routing patterns than softmax (which creates competition between experts).

**Problem solved:** Softmax routing creates artificial competition — increasing one expert's score requires decreasing others'. Sigmoid routing lets each expert's selection be independent.

### Applicability to Parameter Golf

**Relevance: ONLY IF WE USE MoE.** An interesting router design, but only relevant if we adopt MoE.

**Verdict:** Not applicable to our current architecture.

---

## 17. InfiniByte Cross-Domain Synthetic Data Generation

**Description:** A novel data synthesis technique that "cross-breeds" problems from different domains. Starting from competitive coding problems, they inject concepts from math, physics, chemistry, and other sciences. An LLM critic evaluates candidates for clarity, difficulty, and cross-domain adherence. Two strategies: (1) obfuscation (reframing without changing difficulty), (2) complication (making problems genuinely harder by requiring multi-domain reasoning).

**Problem solved:** Synthetic data often lacks the diversity and complexity of real data. Cross-breeding creates novel problems at domain intersections that are rare in natural data.

### Applicability to Parameter Golf

**Relevance: NONE.** We train on the fixed FineWeb dataset. We cannot modify or augment the training data for the competition.

**Verdict:** Not applicable.

---

## 18. Dynamic Sampling for SFT Data Mixture

**Description:** For supervised fine-tuning, they use dynamic sampling where smaller datasets are trained for many epochs while larger datasets are trained for only a few epochs. The number of epochs per dataset is calibrated to achieve optimal single-task performance, then combined.

**Problem solved:** Naive uniform sampling under-represents small high-quality datasets. Dynamic sampling ensures every dataset gets sufficient exposure.

### Applicability to Parameter Golf

**Relevance: LOW.** We have a single dataset (FineWeb) with uniform quality. No mixture to balance.

**Verdict:** Not applicable.

---

## 19. RMSNorm (without bias) Throughout

**Description:** The model uses RMSNorm consistently for all normalization, with no bias terms anywhere in the network (no bias on linear layers either, no dropout).

**Problem solved:** RMSNorm is simpler and faster than LayerNorm (no mean subtraction), and removing biases reduces parameter count slightly. No dropout simplifies training.

### Applicability to Parameter Golf

**Relevance: ALREADY IMPLEMENTED.** Our model already uses RMSNorm-style normalization (LN Scale = 1/sqrt(i+1) per layer) and no dropout. This validates our approach.

**Verdict:** No action needed.

---

## 20. Long-Context Extension via Mixed-Length Sequences

**Description:** For the long-context phase (LC-Phase), they found that training only on long sequences (512K) slightly degraded short-context benchmarks. Instead, they mix 512K and 4K sequences, which improved both short-context (MMLU-Pro, Code) and long-context scores simultaneously.

**Problem solved:** Training only on long sequences shifts the model's attention patterns and hurts performance on typical short inputs.

### Applicability to Parameter Golf

**Relevance: LOW.** Our sequences are fixed at 2048 tokens and we don't need long-context capabilities. The FineWeb validation uses the same sequence length.

**Verdict:** Not applicable.

---

## 21. SFT-Based PTQ Calibration

**Description:** For post-training quantization calibration, they found that using 1K samples from the SFT reasoning dataset yielded better accuracy recovery than using the standard `cnn_dailymail` dataset or on-policy generations from the BF16 model.

**Problem solved:** Calibration data quality significantly impacts quantization accuracy. Domain-matched calibration data preserves the most important weight properties.

### Applicability to Parameter Golf

**Relevance: MEDIUM.** We use GPTQ which also requires calibration data. Our current GPTQ uses "AR self-gen calibration" (the model generates its own calibration data). The insight that calibration data domain matters could help: using FineWeb validation-distribution text for GPTQ calibration might give tighter quantization than random text.

**Verdict:** Worth exploring if we're trying to shrink the GPTQ gap below +0.008 BPB.

---

## 22. Multi-Token Prediction (MTP)

*From the Nemotron 3 White Paper — used in Super and Ultra, not Nano.*

**Description:** A shared model trunk connects to multiple independent output heads, each predicting successive future tokens from the same input representation. The architecture uses ~4 prediction heads (for tokens t+1, t+2, t+3, t+4). During training, all heads contribute to the loss, providing denser supervision per forward pass. During inference, the auxiliary heads can be used for speculative decoding (draft tokens come from the model itself, not a separate draft model) or simply discarded.

**Key results:**
- +2.4% average accuracy improvement across benchmarks on an 8B MoE base model (Table 2 in white paper): MMLU +1.2, MMLU-Pro +2.8, MBPP +1.3, ARC-Challenge +1.6, RACE +1.3, GSM8K +2.0
- 97% acceptance rate on the first two predicted tokens for speculative decoding
- Minimal additional FLOPs during training (heads are lightweight)
- Each MTP head is a small projection from the trunk's hidden state to vocab logits

**Problem solved:** Standard next-token prediction provides a single bit of supervision per position. MTP forces the model to build representations that are useful for predicting further into the future, which encourages better planning and multi-step reasoning. At inference, the extra heads enable speculative decoding without a separate draft model.

### Applicability to Parameter Golf

**Relevance: HIGH — as a training-only auxiliary loss.** This is one of the most promising ideas in the paper for our competition. Here's why:

1. **Free regularization during training.** We add 2-3 small auxiliary prediction heads (just linear projections from hidden dim to vocab: 512x1024 = 524K params each) during training. These heads force the model to build richer internal representations. At inference/submission time, we **discard the auxiliary heads entirely** — they cost zero artifact space.

2. **Denser gradient signal.** With 7000 training steps and a 10-minute budget, every forward pass needs to extract maximum learning signal. MTP gives 2-4x more gradient information per sequence position.

3. **Compatible with our architecture.** Our GPT model already has a trunk (transformer layers) + head (lm_head). Adding auxiliary heads is trivial:
   ```python
   # During training only:
   h = trunk(x)  # [B, T, 512]
   loss_t1 = ce(lm_head(h[:, :-1]), targets[:, 1:])     # standard next-token
   loss_t2 = ce(aux_head2(h[:, :-2]), targets[:, 2:])    # predict t+2
   loss_t3 = ce(aux_head3(h[:, :-3]), targets[:, 3:])    # predict t+3
   loss = loss_t1 + 0.3*loss_t2 + 0.1*loss_t3            # weighted sum
   ```

4. **Low implementation risk.** No architectural changes needed. Just add heads, add loss terms, train, discard heads before quantization. If it doesn't help, we just remove it.

**Potential concern:** The auxiliary heads add training memory and compute. With 3 auxiliary heads at 512x1024 each, that's ~1.5M extra params in training (but zero in artifact). The extra forward/backward through the heads is negligible vs. the transformer trunk.

**Verdict:** **Strongly recommended.** This is a rare "free lunch" — improved training signal at zero artifact cost. The +2.4% NVIDIA measured was on downstream tasks, but the denser supervision should also help with BPB on language modeling. Should be the first thing we try.

---

## 23. LatentMoE (Latent Mixture-of-Experts)

*From the Nemotron 3 White Paper — used in Super and Ultra.*

**Description:** Instead of routing tokens at the full model hidden dimension d to experts, LatentMoE first projects tokens down to a smaller latent dimension l < d (e.g., l = d/4). Experts operate entirely in this latent space. After expert computation, results are projected back to dimension d. This reduces per-expert weight loads by a factor of d/l (~4x), which is reinvested into having more total experts (N' = N * d/l) and more active experts per token (K' = K * d/l).

**Concrete example:** Standard MoE with d=4096 has 128 experts, 6 active. LatentMoE with l=1024 has 512 experts, 22 active. Same compute budget, but consistently higher accuracy (+4.6 MMLU-Pro, +3.2 Code, +1.9 Math).

**Problem solved:** In standard MoE, each expert's weight matrix is d x m (hidden dim x FFN intermediate). Making experts smaller while keeping them at full hidden dim d means they can't learn complex functions. LatentMoE decouples expert complexity from the routing dimension, allowing many more specialized experts at the same cost.

### Applicability to Parameter Golf

**Relevance: LOW-MEDIUM.** Same fundamental issue as standard MoE — total parameter count increases. However, the latent projection idea is interesting independently:

- **Latent-space FFN** (without the MoE routing): project down to a bottleneck, apply a wider FFN there, project back up. This is essentially a bottleneck/inverted-bottleneck MLP. Our model already uses `mlp_mult=3` with LeakyReLU(0.5)^2. A bottleneck variant might be more parameter-efficient.
- **Low-rank projections** in the attention layers could save parameters — our SOTA already explores this implicitly via GQA (fewer KV heads).

**Verdict:** The MoE aspect doesn't apply, but the latent projection concept could inform FFN design experiments.

---

## 24. NVFP4 Training (4-bit Floating-Point Pretraining)

*From the Nemotron 3 White Paper — used in Super and Ultra on GB300 hardware.*

**Description:** Training with native 4-bit floating-point (NVFP4) precision for weights, activations, and gradients. Key technical details:
- **Format:** E2M1 element format with E4M3 block scaling factors and FP32 global scale
- **Micro-block scaling:** Fine-grained 16-element groups, plus 2D block scaling for weights
- **Random Hadamard Transforms (RHTs)** on wgrad inputs to spread outliers across channels
- **Stochastic rounding** on gradients to preserve expected value despite low precision
- **Selective precision:** Last 15% of network kept in high precision; QKV and attention projections in BF16; Mamba output projections in MXFP8 (up to 40% flush-to-zero rate at FP4)
- **Result:** <1% relative loss gap vs BF16 on Nano; <0.6% on 8B active model

**Problem solved:** FP4 enables 3x higher throughput than FP8 on GB300 hardware, making training faster and cheaper. The challenge is maintaining accuracy with only 4 bits.

### Applicability to Parameter Golf

**Relevance: MEDIUM for training speed, not artifact size.** We train on H100s which don't have native FP4 support, so the throughput gains don't apply directly. However, two ideas transfer:

1. **Random Hadamard Transforms for quantization:** RHTs spread outlier values across channels, making the weight distribution more uniform and easier to quantize. We could apply RHTs before GPTQ to improve our int8 quantization quality. This is a known technique (QuIP#) but the Nemotron paper confirms it works well in practice.

2. **The principle of selective precision:** Their finding that the last 15% of the network, QKV projections, and attention projections are most sensitive aligns with the Nano paper's selective quantization finding. This further validates that our GPTQ should not be uniform.

**Verdict:** The RHT idea for improving GPTQ quality is worth investigating. The FP4 training itself requires hardware we don't have.

---

# Synthesis: Most Promising Ideas for Parameter Golf

Given our competition constraints (16MB artifact, 10-minute training on 8xH100, BPB on FineWeb), most of Nemotron's innovations target the wrong problem — they're about scaling up (MoE, RL, long-context) while we need to scale down (compression, parameter efficiency). But a few ideas are highly relevant.

### Tier 1: Directly Actionable — Strong Expected Value

1. **Multi-Token Prediction as Auxiliary Training Loss (#22)** — **Most promising overall.** Add 2-3 lightweight prediction heads during training that predict tokens t+2, t+3, t+4. This provides denser gradient signal per forward pass — critical when we only have 7000 training steps. The heads are discarded before quantization, so they cost ZERO artifact space. NVIDIA measured +2.4% average accuracy improvement from MTP. Even if BPB improvement is smaller, this is essentially free to try. Implementation is ~20 lines of code.

2. **Selective/Mixed-Precision Quantization (#7, #24)** — Our GPTQ currently uses uniform int8 across all layers. Per-layer sensitivity analysis + non-uniform bit allocation could reduce the GPTQ gap below +0.008 BPB. Both the Nano paper (attention layers most sensitive) and the white paper (last 15% of network, QKV projections) confirm that uniform quantization leaves quality on the table. Consider also Random Hadamard Transforms (#24) before GPTQ to spread outliers.

3. **Better GPTQ Calibration Data (#21)** — Simple and cheap. Switch from self-generated calibration to FineWeb-distribution calibration data. Could improve quantization fidelity for no cost.

### Tier 2: Worth Testing — Low Risk, Moderate Expected Value

4. **WSD Learning Rate Schedule (#4)** — Easy schedule change. Keep peak LR for ~80% of training instead of cosine warmdown starting at step 3000. Zero parameter cost. Risk: our Muon optimizer may respond differently than AdamW to the extended stable phase.

### Tier 3: Interesting but High-Risk

5. **Hybrid Mamba-Transformer (#1)** — Mamba layers have fewer params than attention for the same capacity, which helps with our 16MB budget. But requires custom CUDA kernels or a pure-PyTorch implementation. At 2048-token sequences, the compute advantage is small. High implementation risk this late in competition.

6. **LatentMoE Bottleneck Concept (#23)** — Not MoE itself, but the idea of bottleneck FFN layers (project down, compute, project back) could be parameter-efficient. Would need careful ablation.

### Not Applicable

Everything else (MoE routing, RL techniques, curriculum sampling, data synthesis, long-context, no-positional-embedding) targets problems we don't have or resources we can't control.

### Recommended Experiment Plan

| Priority | Experiment | GPU Cost | Expected BPB Gain |
|----------|-----------|----------|-------------------|
| 1 | **MTP auxiliary heads** (2-3 heads, weighted loss) | 1x 10-min run on 8xH100 | 0.005-0.015 |
| 2 | **Layer-wise GPTQ sensitivity** + non-uniform bit allocation | 1-2 hrs on 1xA100 | 0.002-0.005 |
| 3 | **RHT before GPTQ** (Random Hadamard Transform) | 30 min on 1xA100 | 0.001-0.003 |
| 4 | **FineWeb-calibrated GPTQ** | 30 min on 1xA100 | 0.001-0.002 |
| 5 | **WSD schedule** (stable 80%, decay 20%) | 1x 10-min run on 8xH100 | 0.002-0.005 |

Total GPU cost estimate: ~$10-15 for experiments that could collectively shave 0.01-0.03 BPB.

**The MTP experiment should be run first** — it has the highest expected value, zero artifact cost, and if successful, all subsequent experiments (GPTQ, schedule) benefit from the better-trained base model.

# Parameter Golf: Approaches Audit

Systematic audit of all approaches tried by others in openai/parameter-golf, covering merged records and open PRs as of 2026-04-01.

## BPB Progression of Merged Records

| # | BPB | PR | Author | Key Innovation |
|---|-----|-----|--------|----------------|
| 0 | 1.2244 | baseline | - | 9L 512d GQA, Muon+Adam, int8+zlib |
| 1 | 1.2197 | #42 | @chonchiog | FP16 tied embedding (quant gap 0.007->0.0005) |
| 2 | 1.1925 | #50 | @mattqlf | Sliding window eval stride=64 (-0.032 free) |
| 3 | 1.1748 | #60 | @notapplica | 10L + Muon WD + overtone spectral init |
| 4 | 1.1598 | #63 | @yahya010 | Int6 STE QAT (zero quant gap) + zstd-22 |
| 5 | 1.1556 | #65 | @aquariouseworkman | SmearGate + BigramHash + OrthoInit + MLP 3x |
| 6 | 1.1483 | #162 | @raahilshah | BigramHash(4096) + SWA + per-row int6 |
| 7 | 1.1428 | #180 | @thwu1 | Mixed int5/int6 (int5 MLP compresses 1.88x) |
| 8 | 1.1307 | #265 | @unnir | Exclusive Self Attention (XSA), last 3 layers |
| 9 | 1.1271 | #287 | @jfprincz | XSA 4 layers + EMA(0.997) replacing SWA |
| 10 | 1.1248 | #315 | @jfprincz | Partial RoPE (16/64 dims) + LN Scale |
| 11 | 1.1233 | #414 | @signalrush | GPTQ-lite (per-row optimal clip) |
| 12 | 1.1194 | #549 | @abaybektursun | LeakyReLU(0.5)^2 + score-first TTT + Parallel Muon |
| 13 | **1.1147** | #1019 | @abaybektursun | **Full Hessian GPTQ (AR self-gen calib) + XSA-all + BigramHash 3072** |

Current merged SOTA: **1.1147 BPB** (3-seed mean, std 0.0004).


## Technique Catalog

### Evaluation-Time Techniques

| Technique | Impact | Source | Details |
|-----------|--------|--------|---------|
| Sliding window eval | -0.025 to -0.035 | #50 @mattqlf | Stride 16-64, each token gets 960+ ctx instead of 0-1023. Free improvement, 70s on 8xH100 |
| Score-first TTT | -0.003 | #77 @samacqua, #549 @abaybektursun | Per-document LoRA/SGD adaptation. Score chunk BEFORE update (no leakage). Reset between docs |
| SLOT | -0.010 to -0.021 | #1176 @bigbag | Per-batch 512-dim delta vector, 8 AdamW steps at last hidden layer. Model weights frozen |
| GPTQ (full Hessian) | -0.005 | #1019 @abaybektursun | Cholesky-based, AR self-generated calibration (131K tokens). Closes 84% of val-vs-random gap |
| GPTQ-lite | -0.0006 | #414 @signalrush | Per-row clip percentile search (5 candidates, min MSE) |
| EGGROLL | additive | #1156 @haikosys | Post-GPTQ zeroth-order bin-shift optimization. Strictly non-degrading. 60s budget |

### Architecture

| Technique | Impact | Source | Details |
|-----------|--------|--------|---------|
| More layers (9->11) | -0.005 to -0.01 | #60, #265 | Funded by int5/int6 compression savings |
| MLP 3x-3.5x | large | #162 @raahilshah | Largest single improvement source at scale |
| SmearGate | -0.007 | #65 @aquariouseworkman | Learned per-dim gate blending current+previous token embedding |
| BigramHash | -0.01+ | #162 @raahilshah | Hash table (4096-10240 buckets, dim=112-160) for token-pair context |
| XSA (Exclusive Self Attention) | -0.002 | #265 @unnir | Removes self-value bias via orthogonal projection (arXiv:2603.09078). Free reshape, ~2ms |
| Partial RoPE | -0.001+ | #315 @jfprincz | Rotary on only 16/64 head dims, remaining position-free |
| LN Scale | stabilizing | #315 @jfprincz | RMSNorm * 1/sqrt(layer_idx+1) to damp deeper layers |
| U-Net skip connections | included | baseline | Encoder-decoder skips already in baseline |
| Parallel residuals | -0.002+ | #1204 @msisovic | Attn+MLP read from different residual lanes (from layer 7+) |
| Window attention | throughput | #1212 @Gusanidas | 512-token window on alternating layers via FA3, enables 12L |
| QK gain | -0.006 | #1176 @bigbag | QK_GAIN_INIT=4.0 for sharper attention |
| Value Embedding | small | #414 @signalrush | Shared value embedding (dim=128) injected at deep layers |

### Activations

| Technique | Impact | Source | Details |
|-----------|--------|--------|---------|
| LeakyReLU(0.5)^2 | -0.003 | #549, concept #493 @parinzee | One-line change, preserves negative gradient flow |
| SwiGLU | -0.004 | #73 @NishantDahal | Swish-gated linear unit (non-record submission) |

### Quantization & Compression

| Technique | Impact | Source | Details |
|-----------|--------|--------|---------|
| Int6 STE QAT | eliminates gap | #63 @yahya010 | Fake quantization during training, zero roundtrip gap |
| FP16 tied embedding | -0.007 gap | #42 @chonchiog | Keep tied embed in fp16 during export (~500KB extra) |
| Mixed int5/int6 | extra params | #180 @thwu1 | Int5 for MLP (3 zero high bits, zstd 1.88x), int6 for attention |
| Ternary (BitNet b1.58) | 4x params/MB | #640 @CiprianFlorin-Ifrim | {-1,0,+1}, 73.7M params in 15.99MB, base-3+LZMA packing |
| Binary | 6x params/MB | #641 @CiprianFlorin-Ifrim | {-1,+1}, 106.2M params, but needs 2h+ training |
| Brotli-11 | -400KB vs LZMA | #1179 @dexhunter | Alternative compressor, saves artifact space |
| rANS | varies | #1215, #1123 | Per-tensor adaptive rANS entropy coding |
| Soft-round QAT | small | #1179 @dexhunter | Differentiable rounding during training |

### Optimization

| Technique | Impact | Source | Details |
|-----------|--------|--------|---------|
| Parallel Muon | throughput | #549 @abaybektursun | Batched Newton-Schulz via torch.bmm (parameter banking) |
| Turbo-Muon | throughput | #1089 @mikeapedia | AOL preconditioning + Polar Express coefficients |
| Muon WD | generalization | #60 @notapplica | Decoupled weight decay 0.02-0.04 on Muon params |
| EMA | -0.001 to -0.006 | #287 @jfprincz | Exponential moving average (decay=0.997), smoother than SWA |
| SWA | similar | #162 @raahilshah | Stochastic weight averaging, every 50-120 steps |
| Coprime-stride loader | small | #1060 @dexhunter | Diverse batches from coprime-stride block sampling across shards |
| Warmdown tuning | -0.005 gap | #61 @saml212 | Always-decaying schedule reduces quant penalty 0.014->0.005 |
| Orthogonal init | convergence | #65 @aquariouseworkman | Orthogonal weight init for all linear layers |
| Split-LR | small | #1179 @dexhunter | Different learning rates for early vs late layers |


## Ternary / Low-Bit Track

| PR | BPB | Author | Details |
|----|-----|--------|---------|
| #640 (merged) | 1.1570 | @CiprianFlorin-Ifrim | 73.7M ternary, 10L dim=768, 4x MLP, NeoMuon, YaRN, 8192 BPE, FP8 QAT, base-3+LZMA |
| #641 (merged, non-record) | 1.1239 | @CiprianFlorin-Ifrim | 106.2M binary, 15L, 2h training (exceeds limit). Binary beats ternary with enough compute |

Key finding: width > depth for ternary. 768d/10L outperforms 512d/25L. 250+ experiments documented.


## Top Open PRs (Not Yet Merged)

### Tier 1: Best Claimed Results

| PR | BPB | Author | Approach |
|----|-----|--------|----------|
| #1184 | **0.9485** | @icryo | Scylla tokenizer + Full GPTQ + XSA-all + FA3 |
| #1143 | **1.0806** | @simon-marcus | **Scylla** custom TokenMonster tokenizer + legal TTT |
| #1176 | **1.0914** | @bigbag | QK-Gain 4.0 + XSA-11 + Muon-TTT + **SLOT** |
| #1172 | **1.1015** | @dexhunter | SLOT + Split-LR + Full GPTQ + XSA-all |
| #1105 | **1.1052** | @abaybektursun | Fused Triton+CUTLASS MLP + MLP 3.5x + mixed int5/int6 |
| #1204 | **1.1063** | @msisovic | Parallel residuals + mini depth recurrence |
| #1209 | **1.1064** | @andrewbaggio1 | Full GPTQ + Score-first TTT + SLOT |
| #1006 | **1.1085** | @NewyorkDev | JEPA + AdamW TTT + Full GPTQ + FA3 |
| #1089 | **1.1091** | @mikeapedia | Turbo-Muon + EngramLite + ParamBanking + GPTQ mixed-precision |
| #1120 | **1.1099** | @newjordan | "Rascal" -- simple stack, no GPTQ, naive int6+zstd |
| #1212 | **1.1108** | @Gusanidas | Window attention (512) + mixed seq_len + 12L |
| #1179 | **1.1105** | @dexhunter | Split-LR + BigramHash(2816x160) + Brotli-11 |

### Tier 2: Competitive

| PR | BPB | Author | Approach |
|----|-----|--------|----------|
| #1145 | 1.1109 | @AnirudhRahul | Full GPTQ + XSA-11 + online legal n-gram augment |
| #1135 | 1.1116 | @barneywohl | Fused Triton MLP + Full GPTQ + coprime loader |
| #1099 | 1.1133 | @Bortlesboat | Coprime-stride loader + Full GPTQ + XSA-all |
| #1130 | 1.1140 | @Gusanidas | Kitchen Sink V2 (12-seed validated) |
| #1169 | 1.1126 | @Bortlesboat | Turbo-Muon + EngramLite + ParamBanking |
| #1128 | 1.1154 | @AnubhavBharadwaaj | SLOT + LeakyReLU2 + Legal TTT |
| #1156 | 1.1161 | @haikosys | EGGROLL v2 (post-GPTQ zeroth-order optimization) |
| #1170 | 1.1199 | @Christopher-Lee-McClendon | NativeFlowMatcher: OT-CFM velocity network |


## Novel / Non-Transformer Architectures

| PR | BPB | Author | Architecture |
|----|-----|--------|-------------|
| #1208 | 1.176 | @newjordan | **Nightcrawler**: 5 flat + 1 crawler + 5 flat layers, shared TAP encoder |
| #1140 | 1.187 | @newjordan | **Micro Crawler**: Causal coordination at 3 temporal resolutions |
| #1061 | 1.34 | @rolandnsharp | **Causal Oscillator LM**: Damped harmonic oscillator bank, no transformer |
| #1067 | 1.42 | @dheeren-tejani | **BSM**: Bounded State Manifold, O(N) geometric bounding box intersection |
| #1044 | 1.90 | @greqone | **H-Net**: Learned byte-level tokenization, differentiable chunking |
| #1146 | - | @nguthiru | **Elastic Associative Memory**: Discards transformer at inference, uses EAM |
| #1152 | 1.79 | @ericdatum | **Connectome-JEPA**: Sparse I/O bottleneck, biologically-inspired |
| #980 | - | @slowomir33-arch | **LOGOS-44**: Toroidal CDMA Field Decoder, 44 passes through shared block |

### SSM / State-Space Models

| PR | BPB | Author | Details |
|----|-----|--------|---------|
| #1013 | 1.1682 | @himanshudongre | S4D-Lin hybrid, zero throughput penalty |
| #1107 | 1.5633 | @mradassaad | Mamba-3 SSD + Attention hybrid |
| #970 | 1.2907 | @dnldsz | GatedDeltaNet SSM (fla library) |
| #1197 | 3.32 | @dentity007 | Mamba-inspired 3:1 SSM:Attention |

### Diffusion Models

| PR | BPB | Author | Details |
|----|-----|--------|---------|
| #1100 | 1.1465 | @agalimova | **MDLM** -- first diffusion to beat AR baseline. Masking eps=0.1 critical |
| #1053 | 1.3600 | @ikermoel | Masked diffusion, bidirectional training |
| #1194 | 3.38 | @dentity007 | Hybrid AR+diffusion (30/70 split) |

### Universal Transformers / Depth Recurrence

| PR | BPB | Author | Details |
|----|-----|--------|---------|
| #363 (merged) | 1.1787 | @evangelinehelsinki | **Conclusion: recurrence doesn't help** under competition constraints. Flat 11L beats looped 3x3 by 0.025 |
| #1110 | 1.2249 | @gowtham0992 | 3 blocks x 4 iterations = 12 effective layers |
| #1096 | 1.3342 | @vimeto | Depth-recurrent UT + rank-1 LoRA |
| #1206 | - | @oneKn8 | Single 1024d block x 24 iterations (unlimited compute) |

### JEPA (Joint Embedding Predictive Architecture)

| PR | BPB | Author | Details |
|----|-----|--------|---------|
| #1006 | 1.1085 | @NewyorkDev | JEPA + AdamW TTT + Full GPTQ |
| #1012 | negative | @himanshudongre | Works on synthetic Markov chains but collapses on real text |
| #1116 | 1.4447 | @gowtham0992 | First JEPA attempt |

### Custom Tokenizers

| PR | BPB | Author | Details |
|----|-----|--------|---------|
| #1184 | 0.9485 | @icryo | Scylla + full modern stack |
| #1143 | 1.0806 | @simon-marcus | **Scylla**: TokenMonster-derived, iterative autoresearch selection |
| #1210 | - | @mikeapedia | Custom tokenizer with web-content symbols |
| #973 | - | @mrbese | **BESE**: 38-token structured alphabet + BPE (288 vocab) |


## N-Gram / Statistical Approaches (Controversial Legality)

These achieve extremely low BPB by memorizing training data statistics, but most violate competition rules.

| PR | BPB | Author | Details |
|----|-----|--------|---------|
| #1056 | 0.0180 | @sofiabod | Packed causal n-gram + Dirichlet backoff |
| #986 | 0.0830 | @sofiabod | Two-pass Dirichlet CTW, order 2-13 hash tables |
| #1095 | 0.0905 | @vimeto | Seed-regenerated random model + incremental n-gram |
| #968 | 0.1155 | @dentity007 | Order-20 Dirichlet posterior + phrase cache |

PR #1147 (@Robby955) proves mathematically that hashed n-gram caches produce invalid (unnormalized) probability distributions.


## Documented Negative Results

| Approach | Result | Source |
|----------|--------|--------|
| Depth recurrence | +0.025 BPB worse than flat | #363 @evangelinehelsinki |
| Knowledge distillation | +0.013 BPB worse | #1029 @fielding |
| MC Dropout ensemble | No diversity at 17M params | #1021 @abaybektursun |
| kNN-LM | Negative | #1103 @abaybektursun |
| TrigramHash | +0.045 BPB | #1186 @andrewbaggio1 |
| SGD+momentum TTT | +0.065 BPB | #1186 @andrewbaggio1 |
| Loss truncation | Negative | #1103 @abaybektursun |
| SWA sabotages QAT | -3.64 mBPB | #989 @alexanderaperry-arch |
| JEPA on real text | Collapses (-0.24% CE) | #1012 @himanshudongre |
| Mixed-precision GPTQ | Negative | #1103 @abaybektursun |
| Multi-layer kNN | Negative | #1103 @abaybektursun |


## Meta-Analysis & Research

- **PR #1162** (@abaybektursun): Mined 975 training runs from 409 PRs. Early BPB at step 1000 correlates 0.86 with final BPB. Sweet spot: 25-30M params at int6. BigramHash, EMA, XSA are the strongest technique associations.
- **PR #1048** (@mrdavtan): Cross-seed rotational symmetry -- Procrustes alignment gives 90% MSE reduction across seeds.
- **PR #1214** (@gersh): Emergent weight symmetry in layers 6-8 O projections (sym_energy=0.999998).


## Current Dominant Stack (What Top Submissions Converge On)

1. **Architecture**: 11-12L, 512d, GQA 8H/4KV, MLP 3-3.5x, U-Net skips, tied embeddings
2. **Attention**: XSA all layers, Partial RoPE (16/64 dims), QK gain 2.5-4.0, softcap 30
3. **Activation**: LeakyReLU(0.5)^2
4. **Bigram context**: SmearGate + BigramHash (2048-3072 buckets, dim=112-160)
5. **Optimizer**: Parallel Muon (Newton-Schulz) + Adam, WD 0.02-0.04
6. **Weight averaging**: EMA (0.997) + optional tight SWA
7. **Quantization**: Full Hessian GPTQ with AR self-gen calibration, mixed int5/int6
8. **Compression**: LZMA preset=9 or Brotli-11
9. **Eval**: Sliding window stride 16-64
10. **Data**: Coprime-stride multi-shard loader

## Highest-Leverage Unexploited Directions

Based on open PR results vs merged SOTA:

1. **Custom tokenizer** (Scylla): -0.034 to -0.17 BPB. Biggest single gain available, but requires retokenization infrastructure.
2. **SLOT**: -0.010 to -0.021 BPB at eval time. Per-batch delta optimization, no weight modification.
3. **Window attention + 12L**: Throughput savings fund an extra layer.
4. **Parallel residuals**: -0.002+ from dual-lane residual streams.
5. **Fused Triton/CUTLASS kernels**: Recover throughput lost to GPTQ, enabling MLP 3.5x.
6. **EGGROLL**: Free post-GPTQ improvement via zeroth-order bin shifts.

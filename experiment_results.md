# Experiment Results

All runs: 200 iterations, 8192 tokens/step, seq_len=512, 9 layers.
Total tokens seen per run: 200 * 8192 = 1,638,400.

## Summary Table

| Experiment | Arch | Weights | Dim | Params | Val BPB@200 | Artifact | tok/s |
|---|---|---|---|---|---|---|---|
| Baseline | Transformer | fp32+Muon | 512 | 17.1M | **2.486** | (int8 ref) | 28,000 |
| Exp 1: Ternary Transformer | Transformer | Ternary+Adam | 512 | 17.0M | 2.831 | 3.1MB | 33,400 |
| Exp 2: FP GRU | GRU | fp32+Adam | 512 | 24.1M | 2.768 | 24.1MB | 15,900 |
| Exp 3: Ternary GRU | GRU | Ternary+Adam | 512 | 24.1M | 2.844 | 4.7MB | 15,200 |
| Exp 4: Wide Ternary GRU | GRU | Ternary+Adam | 1024 | 95.4M | **2.654** | 17.9MB* | 5,400 |
| Exp 5: Metaplastic GRU | GRU | Metaplastic+Adam | 512 | 24.1M | 2.999 | 4.3MB | 15,600 |

*Exp 4 artifact is OVER the 16MB limit. Needs dim reduction (e.g., ~960) to fit.

## Validation BPB Progression

### Baseline (Transformer, fp32, Muon)
| Step | Val BPB |
|------|---------|
| 0 | 4.103 |
| 50 | 2.874 |
| 100 | 2.673 |
| 150 | 2.549 |
| 200 | 2.486 |

### Exp 1: Ternary Transformer (Adam)
| Step | Val BPB |
|------|---------|
| 0 | 4.103 |
| 50 | 3.241 |
| 100 | 3.012 |
| 150 | 2.870 |
| 200 | 2.831 |

### Exp 2: Full-precision GRU (Adam)
| Step | Val BPB |
|------|---------|
| 0 | 4.103 |
| 50 | 3.135 |
| 100 | 2.928 |
| 150 | 2.808 |
| 200 | 2.768 |

### Exp 3: Ternary GRU (Adam)
| Step | Val BPB |
|------|---------|
| 0 | 4.103 |
| 50 | 3.222 |
| 100 | 3.016 |
| 150 | 2.883 |
| 200 | 2.844 |

### Exp 4: Wide Ternary GRU (dim=1024, Adam)
| Step | Val BPB |
|------|---------|
| 0 | 4.178 |
| 50 | 3.065 |
| 100 | 2.832 |
| 150 | 2.697 |
| 200 | 2.654 |

### Exp 5: Metaplastic Ternary GRU (Adam)
| Step | Val BPB |
|------|---------|
| 0 | 4.103 |
| 50 | 3.471 |
| 100 | 3.155 |
| 150 | 3.022 |
| 200 | 2.999 |

## Training Loss Curves (every 5 steps)

### Baseline
step:5 5.290 | step:10 5.753 | step:15 5.737 | step:20 5.625 | step:25 5.468
step:30 5.375 | step:35 5.205 | step:40 5.081 | step:45 4.933 | step:50 4.867
step:55 4.980 | step:60 4.775 | step:65 4.772 | step:70 4.626 | step:75 4.492
step:80 4.548 | step:85 4.415 | step:90 4.506 | step:95 4.461 | step:100 4.482
step:105 4.412 | step:110 4.425 | step:115 4.384 | step:120 4.294 | step:125 4.259
step:130 4.406 | step:135 4.304 | step:140 4.224 | step:145 4.327 | step:150 4.402
step:155 4.243 | step:160 4.169 | step:165 4.206 | step:170 4.218 | step:175 4.072
step:180 4.274 | step:185 4.158 | step:190 4.093 | step:195 4.365 | step:200 4.407

### Exp 1: Ternary Transformer
step:5 6.463 | step:10 6.064 | step:15 6.083 | step:20 5.986 | step:25 5.997
step:30 5.852 | step:35 5.783 | step:40 5.664 | step:45 5.589 | step:50 5.471
step:55 5.506 | step:60 5.361 | step:65 5.385 | step:70 5.239 | step:75 5.000
step:80 5.175 | step:85 5.089 | step:90 5.120 | step:95 5.130 | step:100 5.075
step:105 5.001 | step:110 5.024 | step:115 4.954 | step:120 4.974 | step:125 4.928
step:130 4.927 | step:135 4.868 | step:140 4.825 | step:145 4.848 | step:150 4.916
step:155 4.802 | step:160 4.778 | step:165 4.809 | step:170 4.783 | step:175 4.695
step:180 4.780 | step:185 4.759 | step:190 4.668 | step:195 4.850 | step:200 4.935

### Exp 2: Full-precision GRU
step:5 6.468 | step:10 6.059 | step:15 6.025 | step:20 5.857 | step:25 5.751
step:30 5.569 | step:35 5.497 | step:40 5.436 | step:45 5.387 | step:50 5.284
step:55 5.355 | step:60 5.205 | step:65 5.229 | step:70 5.083 | step:75 4.853
step:80 5.027 | step:85 4.928 | step:90 4.971 | step:95 4.986 | step:100 4.943
step:105 4.871 | step:110 4.895 | step:115 4.821 | step:120 4.836 | step:125 4.810
step:130 4.814 | step:135 4.744 | step:140 4.701 | step:145 4.734 | step:150 4.806
step:155 4.704 | step:160 4.662 | step:165 4.698 | step:170 4.696 | step:175 4.581
step:180 4.680 | step:185 4.655 | step:190 4.565 | step:195 4.758 | step:200 4.846

### Exp 3: Ternary GRU
step:5 6.468 | step:10 6.061 | step:15 6.025 | step:20 5.938 | step:25 5.905
step:30 5.806 | step:35 5.733 | step:40 5.631 | step:45 5.566 | step:50 5.441
step:55 5.486 | step:60 5.354 | step:65 5.374 | step:70 5.225 | step:75 4.962
step:80 5.172 | step:85 5.072 | step:90 5.119 | step:95 5.140 | step:100 5.083
step:105 5.010 | step:110 5.028 | step:115 4.960 | step:120 4.987 | step:125 4.951
step:130 4.943 | step:135 4.890 | step:140 4.847 | step:145 4.863 | step:150 4.931
step:155 4.843 | step:160 4.810 | step:165 4.838 | step:170 4.822 | step:175 4.717
step:180 4.804 | step:185 4.797 | step:190 4.691 | step:195 4.874 | step:200 4.970

### Exp 4: Wide Ternary GRU (dim=1024)
step:5 6.189 | step:10 6.033 | step:15 5.963 | step:20 5.790 | step:25 5.646
step:30 5.462 | step:35 5.376 | step:40 5.320 | step:45 5.291 | step:50 5.161
step:55 5.240 | step:60 5.062 | step:65 5.089 | step:70 4.932 | step:75 4.710
step:80 4.877 | step:85 4.768 | step:90 4.807 | step:95 4.803 | step:100 4.800
step:105 4.694 | step:110 4.709 | step:115 4.652 | step:120 4.621 | step:125 4.586
step:130 4.636 | step:135 4.559 | step:140 4.510 | step:145 4.543 | step:150 4.635
step:155 4.509 | step:160 4.449 | step:165 4.499 | step:170 4.483 | step:175 4.361
step:180 4.519 | step:185 4.469 | step:190 4.363 | step:195 4.590 | step:200 4.645

### Exp 5: Metaplastic Ternary GRU
step:5 6.477 | step:10 6.081 | step:15 6.080 | step:20 6.021 | step:25 6.049
step:30 5.966 | step:35 5.983 | step:40 5.968 | step:45 5.925 | step:50 5.877
step:55 5.847 | step:60 5.761 | step:65 5.757 | step:70 5.607 | step:75 5.337
step:80 5.514 | step:85 5.398 | step:90 5.411 | step:95 5.402 | step:100 5.312
step:105 5.251 | step:110 5.273 | step:115 5.193 | step:120 5.246 | step:125 5.204
step:130 5.154 | step:135 5.143 | step:140 5.093 | step:145 5.086 | step:150 5.148
step:155 5.088 | step:160 5.075 | step:165 5.085 | step:170 5.076 | step:175 5.006
step:180 5.028 | step:185 5.052 | step:190 4.944 | step:195 5.167 | step:200 5.203

## Roundtrip Degradation (ternary experiments only)

| Experiment | Pre-RT BPB | Post-RT BPB | Degradation |
|---|---|---|---|
| Exp 1: Ternary Transformer | 2.831 | 2.882 | +0.051 |
| Exp 3: Ternary GRU | 2.844 | 2.876 | +0.032 |
| Exp 4: Wide Ternary GRU | 2.654 | 2.709 | +0.055 |
| Exp 5: Metaplastic GRU | 2.999 | 3.056 | +0.057 |

## Key Observations

1. **Baseline dominates at 200 steps** (BPB 2.486), largely thanks to Muon optimizer.
   All Adam-based experiments trail by 0.3-0.5 BPB. Muon vs Adam is a major confound.

2. **Ternary overhead is small**: Exp 2 (FP GRU, 2.768) vs Exp 3 (Ternary GRU, 2.844)
   shows only 0.076 BPB penalty from ternarization at same dim. Encouraging.

3. **GRU matches transformer** (with same optimizer): Exp 1 (Ternary Transformer, 2.831)
   vs Exp 3 (Ternary GRU, 2.844) — nearly identical. The GRU is a viable replacement.

4. **Width scaling works**: Exp 4 (dim=1024, 2.654) beats all other experiments and
   approaches the baseline (2.486). This validates the core thesis: more ternary params
   compensate for per-param precision loss. But artifact is 17.9MB — needs dim ~960.

5. **Metaplastic counters lag** (2.999 vs 2.844 for standard ternary). The approach
   learns but converges slower. May need longer runs or tuning of counter_bits/scale.

6. **All curves still improving at step 200** — longer runs would separate approaches further.

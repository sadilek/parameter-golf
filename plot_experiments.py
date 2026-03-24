#!/usr/bin/env python3
"""Parse experiment logs and generate comparison plots."""
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

LOGS = Path("logs")

EXPERIMENTS_200 = {
    "Baseline (Transformer+Muon)": "exp_baseline",
    "Exp1: Ternary Transformer": "exp1_ternary_transformer",
    "Exp2: FP GRU": "exp2_fp_gru",
    "Exp3: Ternary GRU": "exp3_ternary_gru",
    "Exp4: Wide Ternary GRU (1024)": "exp4_ternary_wide",
    "Exp5: Metaplastic GRU": "exp5_metaplastic",
    "ExpA: Think/Know (1T+6K)": "expA_think_know",
}

EXPERIMENTS_1K = {
    "Baseline (Transformer+Muon)": "cmp_baseline",
    "Exp4: Wide Ternary GRU (1024)": "cmp_exp4",
    "ExpA: Think/Know (1T+6K)": "cmp_expA",
}

COLORS_200 = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#d946ef"]
COLORS_1K = ["#2563eb", "#ea580c", "#d946ef"]

def parse_log(run_id):
    path = LOGS / f"{run_id}.txt"
    if not path.exists():
        return [], []
    text = path.read_text()
    train = [(int(m.group(1)), float(m.group(2)))
             for m in re.finditer(r"step:(\d+)/\d+ train_loss:([\d.]+)", text)]
    val = [(int(m.group(1)), float(m.group(2)))
           for m in re.finditer(r"step:(\d+)/\d+ val_loss:[\d.]+ val_bpb:([\d.]+)", text)]
    return train, val

# === 200-step plots ===
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
for (label, run_id), color in zip(EXPERIMENTS_200.items(), COLORS_200):
    train, _ = parse_log(run_id)
    if train:
        steps, losses = zip(*train)
        ax.plot(steps, losses, label=label, color=color, linewidth=1.5, alpha=0.85)
ax.set_xlabel("Step")
ax.set_ylabel("Training Loss")
ax.set_title("Training Loss (200 steps, 8192 tok/step, seq_len=512)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 200)

ax = axes[1]
for (label, run_id), color in zip(EXPERIMENTS_200.items(), COLORS_200):
    _, val = parse_log(run_id)
    if val:
        steps, bpbs = zip(*val)
        ax.plot(steps, bpbs, "o-", label=label, color=color, linewidth=2, markersize=6)
ax.set_xlabel("Step")
ax.set_ylabel("Validation BPB (bits-per-byte)")
ax.set_title("Validation BPB (lower is better)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 210)

plt.tight_layout()
plt.savefig("experiment_plots_200.png", dpi=150, bbox_inches="tight")
print("Saved experiment_plots_200.png")
plt.close()

# === 1K-step plots ===
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
for (label, run_id), color in zip(EXPERIMENTS_1K.items(), COLORS_1K):
    train, _ = parse_log(run_id)
    if train:
        steps, losses = zip(*train)
        ax.plot(steps, losses, label=label, color=color, linewidth=1.5, alpha=0.85)
ax.set_xlabel("Step")
ax.set_ylabel("Training Loss")
ax.set_title("Training Loss (1500 steps, 8192 tok/step, seq_len=512)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1550)

ax = axes[1]
for (label, run_id), color in zip(EXPERIMENTS_1K.items(), COLORS_1K):
    _, val = parse_log(run_id)
    if val:
        steps, bpbs = zip(*val)
        ax.plot(steps, bpbs, "o-", label=label, color=color, linewidth=2, markersize=6)
ax.set_xlabel("Step")
ax.set_ylabel("Validation BPB (bits-per-byte)")
ax.set_title("Validation BPB at 1500 steps (lower is better)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1550)

plt.tight_layout()
plt.savefig("experiment_plots_1k.png", dpi=150, bbox_inches="tight")
print("Saved experiment_plots_1k.png")
plt.close()

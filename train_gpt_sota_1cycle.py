"""SOTA script with 1cycle LR schedule patch.

This is a thin wrapper that patches the lr_mul function in train_gpt_sota.py
to support 1cycle LR scheduling. All other code is identical.

Usage:
    LR_SCHEDULE=1cycle torchrun --standalone --nproc_per_node=1 train_gpt_sota_1cycle.py

Env vars:
    LR_SCHEDULE=warmdown (default, original SOTA behavior)
    LR_SCHEDULE=1cycle (1cycle: ramp up then cosine down)
    ONECYCLE_PEAK_FRAC=0.3 (fraction of training at which LR peaks)
    ONECYCLE_MIN_DIV=4 (LR starts and ends at base_lr / min_div)
"""
import math
import os
import sys
from pathlib import Path

# Read the SOTA script source
_sota_path = os.environ.get("SOTA_SCRIPT", "train_gpt_sota.py")
_source = Path(_sota_path).read_text(encoding="utf-8")

# Parse 1cycle config from env
_lr_schedule = os.environ.get("LR_SCHEDULE", "warmdown")
_peak_frac = float(os.environ.get("ONECYCLE_PEAK_FRAC", "0.3"))
_min_div = float(os.environ.get("ONECYCLE_MIN_DIV", "4"))

if _lr_schedule == "1cycle":
    # Replace the lr_mul function with a 1cycle-aware version
    _old_lr_mul = """\
    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0"""

    _new_lr_mul = f"""\
    def lr_mul(step: int, elapsed_ms: float) -> float:
        # 1cycle LR schedule: ramp up to peak, cosine anneal down
        peak_frac = {_peak_frac}
        min_div = {_min_div}
        min_mul = 1.0 / min_div
        if max_wallclock_ms is not None:
            frac = elapsed_ms / max_wallclock_ms
        else:
            frac = step / max(args.iterations, 1)
        frac = min(frac, 1.0)
        if frac < peak_frac:
            scale = min_mul + (1.0 - min_mul) * (frac / peak_frac)
        else:
            t = (frac - peak_frac) / (1.0 - peak_frac)
            scale = min_mul + 0.5 * (1.0 - min_mul) * (1.0 + math.cos(math.pi * t))
        return scale"""

    if _old_lr_mul not in _source:
        print("ERROR: Could not find lr_mul to patch. SOTA script may have changed.", file=sys.stderr)
        sys.exit(1)
    _source = _source.replace(_old_lr_mul, _new_lr_mul)
    print(f"1cycle LR: peak_frac={_peak_frac}, min_div={_min_div}", file=sys.stderr)

# Execute the (possibly patched) SOTA script
exec(compile(_source, _sota_path, "exec"), {"__name__": "__main__", "__file__": _sota_path})

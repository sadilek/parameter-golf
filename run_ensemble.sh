#!/bin/bash
# Train two models sequentially and save logit dumps for ensemble eval.
# Both models train with the full Muon+1cycle pipeline via train_combined.py.
set -e

COMMON="QUANT_MODE=int6 NUM_LAYERS=7 N_CHANNELS=1 XSA_LAYERS=2 LR_SCHEDULE=1cycle ONECYCLE_MIN_DIV=4 SSE_ENABLED=1 ITERATIONS=500 VAL_LOSS_EVERY=500"

echo "=== Training Model A (seed=1337) ==="
rm -rf /tmp/torchinductor_root/
eval "$COMMON SEED=1337 RUN_ID=ens_A" torchrun --standalone --nproc_per_node=1 train_combined.py
cp final_model.ptz ensemble_model_A.ptz
echo "=== Model A saved ==="

echo "=== Training Model B (seed=42) ==="
rm -rf /tmp/torchinductor_root/
eval "$COMMON SEED=42 ENSEMBLE_PATH=ensemble_model_A.ptz RUN_ID=ens_B" torchrun --standalone --nproc_per_node=1 train_combined.py
echo "=== Done ==="

echo "Combined artifact: $(python3 -c "import os; a=os.path.getsize('ensemble_model_A.ptz'); b=os.path.getsize('final_model.ptz'); print(f'{(a+b)/1e6:.2f}MB ({a/1e6:.2f} + {b/1e6:.2f})')")"

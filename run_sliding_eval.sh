#!/bin/bash
# Run sliding window eval on the saved GPTQ artifact
# Loads the quantized model and evaluates with stride=64
set -e
cd /workspace/parameter-golf

python3 << 'PYEOF'
import os, sys, time, math, glob, io, lzma
import torch
import torch.nn.functional as F
import numpy as np
import sentencepiece as spm

# Import the SOTA model definition
sys.path.insert(0, '.')
# We need to load the script to get model classes
exec(open('train_gpt_sota.py').read().split('def main():')[0])

device = torch.device('cuda')
args = Hyperparameters()
args.bigram_vocab_size = 3072
args.bigram_dim = 112
args.eval_stride = 64

# Load tokenizer and val data
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
val_files = sorted(glob.glob(args.val_files))
val_tokens = load_val_tokens(val_files, args.train_seq_len)
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_bpb_luts(sp, args.vocab_size, device)
print(f"Val tokens: {val_tokens.numel():,}")

# Build model
model = GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers,
    model_dim=args.model_dim, num_heads=args.num_heads,
    num_kv_heads=args.num_kv_heads, mlp_mult=int(args.mlp_mult),
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base,
    qk_gain_init=args.qk_gain_init, bigram_vocab_size=args.bigram_vocab_size,
    bigram_dim=args.bigram_dim, xsa_last_n=args.xsa_last_n,
    rope_dims=args.rope_dims, ln_scale=args.ln_scale,
    ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
).to(device).bfloat16()

# Find and load the quantized artifact
artifact_files = sorted(glob.glob('*_artifact.ptz') + glob.glob('*_submission.ptz'))
if not artifact_files:
    # Try loading the GPTQ state from the run
    artifact_files = sorted(glob.glob('ovn_*_int6_*.pt') + glob.glob('*_int6_gptq.pt'))
print(f"Looking for artifacts: {artifact_files}")

# Load the int6 quantized state dict
# The SOTA script saves a combined artifact with model + optional bigram
quant_file = [f for f in glob.glob('*.ptz') if 'artifact' not in f]
if not quant_file:
    quant_file = glob.glob('*_submission.ptz')

# Actually, let's just find whatever .ptz files exist
all_ptz = sorted(glob.glob('*.ptz'))
print(f"Found .ptz files: {all_ptz}")

# The SOTA script saves the quantized artifact. Let's load it.
if all_ptz:
    ptz_path = all_ptz[-1]  # most recent
    print(f"Loading {ptz_path}...")
    with open(ptz_path, 'rb') as f:
        decompressed = lzma.decompress(f.read())
    loaded = torch.load(io.BytesIO(decompressed), map_location='cpu', weights_only=False)
    if isinstance(loaded, dict) and 'model' in loaded:
        model_data = loaded['model']
    else:
        model_data = loaded

    # Dequantize int6 back to float
    dequant_sd = dequantize_int6(model_data)

    # Load into model
    model_sd = model.state_dict()
    loaded_keys = {k: v.to(dtype=model_sd[k].dtype, device=device)
                   for k, v in dequant_sd.items() if k in model_sd}
    model.load_state_dict(loaded_keys, strict=False)
    print(f"Loaded {len(loaded_keys)}/{len(model_sd)} keys from quantized artifact")
else:
    print("No .ptz artifacts found! Trying float weights...")
    float_files = sorted(glob.glob('*_float.pt'))
    if float_files:
        sd = torch.load(float_files[-1], map_location='cpu', weights_only=False)
        loaded_keys = {k: v.to(device=device) for k, v in sd.items() if k in model.state_dict()}
        model.load_state_dict(loaded_keys, strict=False)
        print(f"Loaded float weights: {len(loaded_keys)} keys")

model.eval()

# Standard eval (non-sliding, for comparison)
print("\n=== Standard eval ===")
t0 = time.perf_counter()
val_loss, val_bpb = eval_val(args, model, 0, 1, device, val_tokens,
                              base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                              eval_seq_len=args.train_seq_len)
print(f"Standard: val_loss={val_loss:.4f} val_bpb={val_bpb:.4f} ({time.perf_counter()-t0:.1f}s)")

# Sliding window eval
print("\n=== Sliding window eval (stride=64) ===")
t0 = time.perf_counter()
sw_loss, sw_bpb = eval_val_sliding(args, model, 0, 1, device,
                                    val_tokens, base_bytes_lut, has_leading_space_lut,
                                    is_boundary_token_lut,
                                    stride=64, eval_seq_len=args.train_seq_len)
print(f"Sliding s64: val_loss={sw_loss:.4f} val_bpb={sw_bpb:.4f} ({time.perf_counter()-t0:.1f}s)")
print(f"\nDelta: {sw_bpb - val_bpb:+.4f} BPB from sliding eval")
PYEOF

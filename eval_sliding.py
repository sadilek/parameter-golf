"""Eval-only script: load GPTQ artifact, run standard + sliding eval."""
import os, sys, time, math, glob, io, lzma
os.environ.setdefault("BIGRAM_VOCAB_SIZE", "3072")
os.environ.setdefault("BIGRAM_DIM", "112")
import torch, torch.nn.functional as F
import numpy as np, sentencepiece as spm

# Load model definitions from SOTA script (everything before main())
with open("train_gpt_sota.py") as f:
    code = f.read()
# Replace __file__ references
code = code.replace("__file__", "'train_gpt_sota.py'")
exec(code.split("def main():")[0])

device = torch.device("cuda")
args = Hyperparameters()

sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
val_files = sorted(glob.glob(args.val_files))
val_tokens = load_val_tokens(val_files, args.train_seq_len)
bl, hl, il = build_bpb_luts(sp, args.vocab_size, device)
print(f"Val tokens: {val_tokens.numel():,}")

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

artifact_path = sys.argv[1] if len(sys.argv) > 1 else "final_model.int6.ptz"
print(f"Loading {artifact_path}...")
with open(artifact_path, "rb") as f:
    data = lzma.decompress(f.read())
loaded = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
model_data = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
dq = dequantize_int6(model_data)
ms = model.state_dict()
lk = {k: v.to(dtype=ms[k].dtype, device=device) for k, v in dq.items() if k in ms}
model.load_state_dict(lk, strict=False)
print(f"Loaded {len(lk)}/{len(ms)} keys")
model.eval()

# Standard eval
print("\n=== Standard eval ===")
t0 = time.perf_counter()
vl, vb = eval_val(args, model, 0, 1, device, val_tokens, bl, hl, il, eval_seq_len=args.train_seq_len)
print(f"Standard: val_loss={vl:.6f} val_bpb={vb:.6f} ({time.perf_counter()-t0:.1f}s)")

# Sliding eval stride=64
print("\n=== Sliding window eval (stride=64) ===")
t0 = time.perf_counter()
sl, sb = eval_val_sliding(args, model, 0, 1, device, val_tokens, bl, hl, il,
                           stride=64, eval_seq_len=args.train_seq_len)
print(f"Sliding s64: val_loss={sl:.6f} val_bpb={sb:.6f} ({time.perf_counter()-t0:.1f}s)")
print(f"\nDelta from sliding: {sb - vb:+.6f} BPB")

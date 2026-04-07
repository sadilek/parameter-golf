"""Simple sliding window eval on saved GPTQ artifact. No torch.compile."""
import os, sys, time, math, glob, io, lzma
os.environ["BIGRAM_VOCAB_SIZE"] = "3072"
os.environ["BIGRAM_DIM"] = "112"
os.environ["MAX_WALLCLOCK_SECONDS"] = "0"

import torch, torch.nn.functional as F
import numpy as np, sentencepiece as spm
from torch import Tensor, nn

# We need model classes. Import them by running the script's class definitions.
# Patch __file__ and prevent main() from running
__file_backup = globals().get('__file__', __name__)
import importlib.util
spec = importlib.util.spec_from_file_location("sota", "train_gpt_sota.py")
sota_module = importlib.util.module_from_spec(spec)
# Don't execute - just load to get the source
with open("train_gpt_sota.py") as f:
    source = f.read()

# Execute only class/function definitions (before main)
defs = source.split("\ndef main():")[0]
defs = defs.replace("__file__", "'train_gpt_sota.py'")
exec(defs, globals())

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

# Load GPTQ artifact
artifact = sys.argv[1] if len(sys.argv) > 1 else "final_model.int6.ptz"
print(f"Loading {artifact}...")
with open(artifact, "rb") as f:
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

# Sliding window eval (NO torch.compile — use forward_logits directly)
print("\n=== Sliding window eval (stride=64, no compile) ===")
seq_len = args.train_seq_len
stride = 64
total_tokens = val_tokens.numel() - 1
window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= 1]
batch_seqs = 16  # smaller batch to avoid OOM

loss_sum = torch.zeros((), device=device, dtype=torch.float64)
token_count = torch.zeros((), device=device, dtype=torch.float64)
byte_count = torch.zeros((), device=device, dtype=torch.float64)

t0 = time.perf_counter()
with torch.inference_mode():
    for bi in range(0, len(window_starts), batch_seqs):
        batch_ws = window_starts[bi:bi + batch_seqs]
        bsz = len(batch_ws)
        x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        wlens = []
        for i, ws in enumerate(batch_ws):
            end = min(ws + seq_len, total_tokens)
            wlen = end - ws
            wlens.append(wlen)
            chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
            x_batch[i, :wlen] = chunk[:-1]
            y_batch[i, :wlen] = chunk[1:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward_logits(x_batch)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y_batch.reshape(-1), reduction="none"
        ).reshape(bsz, seq_len)
        for i, ws in enumerate(batch_ws):
            wlen = wlens[i]
            s = 0 if ws == 0 else max(wlen - stride, 0)
            loss_sum += nll[i, s:wlen].to(torch.float64).sum()
            token_count += float(wlen - s)
            tgt = y_batch[i, s:wlen]
            prev = x_batch[i, s:wlen]
            tb = bl[tgt].to(torch.float64)
            tb += (hl[tgt] & ~il[prev]).to(torch.float64)
            byte_count += tb.sum()
        if bi % (batch_seqs * 50) == 0:
            elapsed = time.perf_counter() - t0
            pct = 100.0 * bi / len(window_starts)
            print(f"  {pct:.0f}% ({bi}/{len(window_starts)}) {elapsed:.0f}s")

sw_loss = (loss_sum / token_count).item()
sw_bpb = (sw_loss / math.log(2.0)) * (token_count.item() / byte_count.item())
print(f"\nSliding s64: val_loss={sw_loss:.6f} val_bpb={sw_bpb:.6f} ({time.perf_counter()-t0:.1f}s)")
print(f"Standard:    val_bpb={vb:.6f}")
print(f"Delta:       {sw_bpb - vb:+.6f} BPB")

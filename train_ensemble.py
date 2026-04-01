"""Train two diverse models and evaluate their ensemble.

Calls main() from train_combined.py twice with different seeds.
Both models go through the identical full training pipeline.

Usage:
  ENSEMBLE_LAYERS=7 ENSEMBLE_ITERS=500 torchrun --standalone --nproc_per_node=1 train_ensemble.py
"""
import os, sys, math, glob, time, torch
import torch.distributed as dist
import sentencepiece as spm
from pathlib import Path

# Set shared config
layers = os.environ.get("ENSEMBLE_LAYERS", "7")
iters = os.environ.get("ENSEMBLE_ITERS", "500")
os.environ.setdefault("QUANT_MODE", "int6")
os.environ.setdefault("N_CHANNELS", "1")
os.environ.setdefault("XSA_LAYERS", "2")
os.environ.setdefault("LR_SCHEDULE", "1cycle")
os.environ.setdefault("ONECYCLE_MIN_DIV", "4")
os.environ.setdefault("SSE_ENABLED", "1")
os.environ["NUM_LAYERS"] = layers
os.environ["ITERATIONS"] = iters
os.environ["VAL_LOSS_EVERY"] = os.environ.get("ENSEMBLE_VAL_EVERY", "500")  # periodic eval

sys.path.insert(0, os.path.dirname(__file__))
import train_combined
train_combined._QUANT_MODE = "int6"
from train_combined import main as train_main, build_luts, ld_val, Hyperparameters

def ensemble_eval(model_a, model_b, val_tokens, seq_len, device, bl, hl, il):
    total_seqs = (val_tokens.numel() - 1) // seq_len
    for name, alpha in [("A_only", 1.0), ("B_only", 0.0), ("50_50", 0.5), ("60_40", 0.6), ("70_30", 0.7)]:
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        token_count = torch.zeros((), device=device, dtype=torch.float64)
        byte_count = torch.zeros((), device=device, dtype=torch.float64)
        model_a.eval(); model_b.eval()
        with torch.inference_mode():
            for bs in range(0, total_seqs, 32):
                be = min(bs + 32, total_seqs)
                local = val_tokens[bs*seq_len:(be*seq_len)+1].to(device=device, dtype=torch.int64)
                x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ea = model_a._embed(x); ha = model_a._run_blocks(ea, ea, causal=True)
                    la = model_a._softcap(model_a._compute_logits(ha.reshape(-1, ha.size(-1))))
                    if model_a.sse: la = model_a.sse(la)
                    eb = model_b._embed(x); hb = model_b._run_blocks(eb, eb, causal=True)
                    lb = model_b._softcap(model_b._compute_logits(hb.reshape(-1, hb.size(-1))))
                    if model_b.sse: lb = model_b.sse(lb)
                mixed = alpha * la.float() + (1 - alpha) * lb.float()
                targets = y.reshape(-1)
                lse = torch.logsumexp(mixed, dim=-1)
                tgt = mixed.gather(1, targets.unsqueeze(1)).squeeze(1)
                loss_sum += (lse - tgt).to(torch.float64).sum()
                token_count += float(targets.numel())
                p, t = x.reshape(-1), y.reshape(-1)
                tb = bl[t].to(torch.float64) + (hl[t] & ~il[p]).to(torch.float64)
                byte_count += tb.sum()
        vl = (loss_sum / token_count).item()
        bpb = (vl / math.log(2.0)) * (token_count.item() / byte_count.item())
        print(f"  ensemble {name}: val_bpb={bpb:.4f}", flush=True)

# Configuration
N_MODELS = int(os.environ.get("N_MODELS", "2"))
SEEDS = [1337, 42, 7, 2024, 314][:N_MODELS]
BOOST = os.environ.get("BOOST", "0") == "1"  # Enable boosting: model N trains on model N-1's errors

models = []
for i, seed in enumerate(SEEDS):
    print(f"\n{'='*60}\nTraining Model {chr(65+i)} (seed={seed}){' [BOOSTED]' if BOOST and i > 0 else ''}\n{'='*60}", flush=True)
    os.environ["SEED"] = str(seed)
    os.environ["RUN_ID"] = f"ens_model{chr(65+i)}"
    # Per-model overrides
    for key in ["ACTIVATION", "MIXER_TYPE", "NUM_LAYERS", "MODEL_DIM", "NUM_HEADS", "NUM_KV_HEADS", "MLP_MULT", "UT_UNIQUE_BLOCKS", "UT_ITERS", "XSA_LAYERS"]:
        val = os.environ.get(f"MODEL{chr(65+i)}_{key}", "")
        if val:
            os.environ[key] = val
        else:
            os.environ.pop(key, None)
    # Boosting: pass previous model so new model focuses on its errors
    prev_model = models[-1] if (BOOST and i > 0) else None
    if prev_model is not None:
        prev_model.eval()
    m = train_main(return_model=True, boost_model=prev_model)
    torch._dynamo.reset()
    models.append(m)
    torch.cuda.empty_cache()

# Ensemble eval
device = next(models[0].parameters()).device
args = Hyperparameters()
sp = spm.SentencePieceProcessor(args.tokenizer_path)
bl, hl, il = build_luts(sp, args.vocab_size, device)
val_tokens = ld_val(args.val_files, args.train_seq_len)

print(f"\n{'='*60}\nEnsemble Evaluation ({len(models)} models)\n{'='*60}", flush=True)

# Individual evals
for i, m in enumerate(models):
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    m.eval()
    with torch.inference_mode():
        for bs in range(0, total_seqs, 32):
            be = min(bs + 32, total_seqs)
            local = val_tokens[bs*args.train_seq_len:(be*args.train_seq_len)+1].to(device=device, dtype=torch.int64)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                e = m._embed(x); h = m._run_blocks(e, e, causal=True)
                l = m._softcap(m._compute_logits(h.reshape(-1, h.size(-1))))
                if m.sse: l = m.sse(l)
            targets = y.reshape(-1)
            lse = torch.logsumexp(l.float(), dim=-1)
            tgt = l.float().gather(1, targets.unsqueeze(1)).squeeze(1)
            loss_sum += (lse - tgt).to(torch.float64).sum()
            token_count += float(targets.numel())
            p, t = x.reshape(-1), y.reshape(-1)
            tb = bl[t].to(torch.float64) + (hl[t] & ~il[p]).to(torch.float64)
            byte_count += tb.sum()
    vl = (loss_sum / token_count).item()
    bpb = (vl / math.log(2.0)) * (token_count.item() / byte_count.item())
    print(f"  Model {chr(65+i)}: val_bpb={bpb:.4f}", flush=True)

# Uniform ensemble
total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
loss_sum = torch.zeros((), device=device, dtype=torch.float64)
token_count = torch.zeros((), device=device, dtype=torch.float64)
byte_count = torch.zeros((), device=device, dtype=torch.float64)
for m in models: m.eval()
with torch.inference_mode():
    for bs in range(0, total_seqs, 32):
        be = min(bs + 32, total_seqs)
        local = val_tokens[bs*args.train_seq_len:(be*args.train_seq_len)+1].to(device=device, dtype=torch.int64)
        x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
        mixed = None
        for m in models:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                e = m._embed(x); h = m._run_blocks(e, e, causal=True)
                l = m._softcap(m._compute_logits(h.reshape(-1, h.size(-1))))
                if m.sse: l = m.sse(l)
            if mixed is None:
                mixed = l.float()
            else:
                mixed = mixed + l.float()
        mixed = mixed / len(models)
        targets = y.reshape(-1)
        lse = torch.logsumexp(mixed, dim=-1)
        tgt = mixed.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss_sum += (lse - tgt).to(torch.float64).sum()
        token_count += float(targets.numel())
        p, t = x.reshape(-1), y.reshape(-1)
        tb = bl[t].to(torch.float64) + (hl[t] & ~il[p]).to(torch.float64)
        byte_count += tb.sum()
vl = (loss_sum / token_count).item()
bpb = (vl / math.log(2.0)) * (token_count.item() / byte_count.item())
print(f"  Ensemble (uniform {len(models)}): val_bpb={bpb:.4f}", flush=True)

# Entropy-adaptive ensemble: weight each model by inverse entropy (confident → more weight)
if len(models) >= 2:
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for bs in range(0, total_seqs, 32):
            be = min(bs + 32, total_seqs)
            local = val_tokens[bs*args.train_seq_len:(be*args.train_seq_len)+1].to(device=device, dtype=torch.int64)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
            all_logits = []
            for m in models:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    e = m._embed(x); h = m._run_blocks(e, e, causal=True)
                    l = m._softcap(m._compute_logits(h.reshape(-1, h.size(-1))))
                    if m.sse: l = m.sse(l)
                all_logits.append(l.float())
            # Compute per-model entropy → inverse entropy as weight
            entropies = []
            for l in all_logits:
                p = torch.softmax(l, dim=-1)
                ent = -(p * torch.log(p + 1e-10)).sum(dim=-1)  # (B*T,)
                entropies.append(ent)
            ent_stack = torch.stack(entropies, dim=0)  # (N_models, B*T)
            # Inverse entropy: lower entropy = more confident = higher weight
            inv_ent = 1.0 / (ent_stack + 0.1)
            weights = inv_ent / inv_ent.sum(dim=0, keepdim=True)  # (N_models, B*T)
            # Weighted logit average
            mixed = torch.zeros_like(all_logits[0])
            for j, l in enumerate(all_logits):
                mixed = mixed + weights[j].unsqueeze(-1) * l
            targets = y.reshape(-1)
            lse = torch.logsumexp(mixed, dim=-1)
            tgt = mixed.gather(1, targets.unsqueeze(1)).squeeze(1)
            loss_sum += (lse - tgt).to(torch.float64).sum()
            token_count += float(targets.numel())
            p_ids, t_ids = x.reshape(-1), y.reshape(-1)
            tb = bl[t_ids].to(torch.float64) + (hl[t_ids] & ~il[p_ids]).to(torch.float64)
            byte_count += tb.sum()
    vl = (loss_sum / token_count).item()
    bpb = (vl / math.log(2.0)) * (token_count.item() / byte_count.item())
    print(f"  Ensemble (entropy-adaptive {len(models)}): val_bpb={bpb:.4f}", flush=True)

# Weight merging (model soup): average all model weights into one
if len(models) >= 2:
    print(f"\n{'='*60}\nWeight Merging (Model Soup)\n{'='*60}", flush=True)
    merged_sd = {}
    ref_sd = models[0].state_dict()
    for key in ref_sd:
        tensors = [models[i].state_dict()[key].float() for i in range(len(models))]
        merged_sd[key] = torch.stack(tensors).mean(dim=0).to(ref_sd[key].dtype)

    merged_model = models[0]
    merged_model.load_state_dict(merged_sd)

    # Eval merged model
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    merged_model.eval()
    with torch.inference_mode():
        for bs in range(0, total_seqs, 32):
            be = min(bs + 32, total_seqs)
            local = val_tokens[bs*args.train_seq_len:(be*args.train_seq_len)+1].to(device=device, dtype=torch.int64)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                e = merged_model._embed(x); h = merged_model._run_blocks(e, e, causal=True)
                l = merged_model._softcap(merged_model._compute_logits(h.reshape(-1, h.size(-1))))
                if merged_model.sse: l = merged_model.sse(l)
            targets = y.reshape(-1)
            lse = torch.logsumexp(l.float(), dim=-1)
            tgt = l.float().gather(1, targets.unsqueeze(1)).squeeze(1)
            loss_sum += (lse - tgt).to(torch.float64).sum()
            token_count += float(targets.numel())
            p, t = x.reshape(-1), y.reshape(-1)
            tb = bl[t].to(torch.float64) + (hl[t] & ~il[p]).to(torch.float64)
            byte_count += tb.sum()
    vl = (loss_sum / token_count).item()
    bpb = (vl / math.log(2.0)) * (token_count.item() / byte_count.item())
    print(f"  Merged (soup, {len(models)} models): val_bpb={bpb:.4f}", flush=True)

if dist.is_available() and dist.is_initialized():
    dist.destroy_process_group()

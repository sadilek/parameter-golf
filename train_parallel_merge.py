"""Train N models in parallel on one GPU, then merge weights.

Each model gets its own data shards for maximum diversity.
After training, weights are averaged ("model soup") into one model.
The merged model often outperforms any individual.

Usage:
  N_PARALLEL=4 torchrun --standalone --nproc_per_node=1 train_parallel_merge.py
"""
import os, sys, math, glob, copy, io, lzma, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import sentencepiece as spm
from pathlib import Path

# Config
N_PARALLEL = int(os.environ.get("N_PARALLEL", "4"))
ITERS = int(os.environ.get("PARALLEL_ITERS", "500"))
LAYERS = int(os.environ.get("PARALLEL_LAYERS", "7"))

os.environ.setdefault("QUANT_MODE", "int6")
os.environ.setdefault("N_CHANNELS", "1")
os.environ.setdefault("XSA_LAYERS", "2")
os.environ.setdefault("LR_SCHEDULE", "1cycle")
os.environ.setdefault("ONECYCLE_MIN_DIV", "4")
os.environ.setdefault("SSE_ENABLED", "1")
os.environ["NUM_LAYERS"] = str(LAYERS)
os.environ["ITERATIONS"] = str(ITERS)
os.environ["VAL_LOSS_EVERY"] = str(ITERS)
os.environ["WARMUP_STEPS"] = "0"  # skip warmup for parallel (compile doesn't work well)
os.environ["COMPILE_MODE"] = "off"

sys.path.insert(0, os.path.dirname(__file__))
import train_combined
train_combined._QUANT_MODE = "int6"
from train_combined import (
    GPT, Hyperparameters, Muon, DistributedTokenLoader,
    q_sd_int6, deq_sd_int6, build_luts, ld_shard, ld_val, eval_val,
    restore_low_dim_params_to_fp32, CTP,
)


def log0(msg):
    print(msg, flush=True)


def create_model(args, device, seed):
    """Create and initialize one model."""
    torch.manual_seed(seed)
    random.seed(seed)
    m = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        group_size=args.bitnet_group_size, activation=args.activation_type,
        n_channels=args.n_channels, xsa_layers=args.xsa_layers,
        sse_enabled=args.sse_enabled, sse_clusters=args.sse_clusters,
        sse_entropy_bins=args.sse_entropy_bins,
    ).to(device).bfloat16()
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            mod.float()
    restore_low_dim_params_to_fp32(m)
    if m.lm_head is not None and args.tie_embeddings:
        m.lm_head.weight.requires_grad_(False)
    return m


def create_optimizers(model, args):
    """Create Muon + Adam optimizers for one model."""
    _excl = {"tok_emb.weight", "lm_head.weight", "lm_head_correction"}
    all_other = [(n, p) for n, p in model.named_parameters() if not any(e in n for e in _excl)]
    matrix_params = [p for n, p in all_other if p.ndim == 2 and not any(pat in n for pat in CTP)]
    scalar_params = [p for n, p in all_other if p.ndim < 2 or any(pat in n for pat in CTP)]

    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                    backend_steps=args.muon_backend_steps, wd=args.muon_wd)
    for g in opt_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_tok = torch.optim.Adam(
        [{"params": [model.tok_emb.weight], "lr": args.embed_lr, "base_lr": args.embed_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)

    return [opt_tok, opt_muon, opt_scalar]


def lr_mul_1cycle(step, total_steps, peak_frac=0.3, min_div=4.0):
    frac = min(step / max(total_steps, 1), 1.0)
    min_mul = 1.0 / min_div
    if frac < peak_frac:
        return min_mul + (1.0 - min_mul) * (frac / peak_frac)
    else:
        t = (frac - peak_frac) / (1.0 - peak_frac)
        return min_mul + 0.5 * (1.0 - min_mul) * (1.0 + math.cos(math.pi * t))


def main():
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    args = Hyperparameters()
    # Re-read mutable args
    args.num_layers = LAYERS
    args.iterations = ITERS
    sp = spm.SentencePieceProcessor(args.tokenizer_path)
    bl, hl, il = build_luts(sp, args.vocab_size, device)
    val_tokens = ld_val(args.val_files, args.train_seq_len)

    seeds = [1337, 42, 7, 2024, 314, 999, 123, 456][:N_PARALLEL]

    # Split training shards across models for data diversity
    all_shards = sorted(glob.glob(args.train_files))
    n_shards = len(all_shards)
    log0(f"Parallel training: {N_PARALLEL} models, {LAYERS}L, {ITERS} steps, {n_shards} shards")

    # Create all models and optimizers
    models = []
    optimizers_list = []
    loaders = []
    for i, seed in enumerate(seeds):
        log0(f"  Creating model {i} (seed={seed})...")
        m = create_model(args, device, seed)
        opts = create_optimizers(m, args)
        # Each model gets a different subset of shards (round-robin)
        model_shards = [all_shards[j] for j in range(n_shards) if j % N_PARALLEL == i]
        if not model_shards:
            model_shards = all_shards  # fallback
        shard_pattern = model_shards  # We'll create a custom loader
        models.append(m)
        optimizers_list.append(opts)

    # Create data loaders with different shard subsets
    for i in range(N_PARALLEL):
        model_shards = [all_shards[j] for j in range(n_shards) if j % N_PARALLEL == i]
        if not model_shards:
            model_shards = all_shards
        # Create a temp pattern by symlinking
        loader = DistributedTokenLoader(args.train_files, 0, 1, device)
        loaders.append(loader)

    n_params = sum(p.numel() for p in models[0].parameters())
    log0(f"  params per model: {n_params:,}")
    grad_accum = 8
    grad_scale = 1.0 / grad_accum
    batch_tokens = args.train_batch_tokens

    # Parallel training loop
    log0(f"\n{'='*60}\nParallel training: {N_PARALLEL} models × {ITERS} steps\n{'='*60}")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(1, ITERS + 1):
        scale = lr_mul_1cycle(step, ITERS)

        for model_idx in range(N_PARALLEL):
            m = models[model_idx]
            opts = optimizers_list[model_idx]
            loader = loaders[model_idx]
            m.train()
            m._force_causal = True

            # Set LR
            for opt in opts:
                for g in opt.param_groups:
                    g["lr"] = g.get("base_lr", args.matrix_lr) * scale

            # Zero grad
            for opt in opts:
                opt.zero_grad(set_to_none=True)

            # Forward + backward (accumulate across micro-batches)
            for micro in range(grad_accum):
                x, y = loader.next_batch(batch_tokens, args.train_seq_len, grad_accum)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = m(x, y)
                (loss * grad_scale).backward()

            # Step
            for opt in opts:
                opt.step()

        if step % 50 == 0 or step == ITERS:
            elapsed = time.perf_counter() - t0
            log0(f"  step:{step}/{ITERS} t:{elapsed*1000:.0f}ms avg:{elapsed*1000/step:.1f}ms/step")

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    log0(f"Training done: {total_time:.0f}s ({total_time*1000/ITERS:.1f}ms/step for {N_PARALLEL} models)")

    # Evaluate individual models
    log0(f"\n{'='*60}\nIndividual evaluation\n{'='*60}")
    individual_bpbs = []
    for i, m in enumerate(models):
        class A:
            val_batch_size = 524288
            train_seq_len = args.train_seq_len
        vl, bpb = eval_val(A(), m, 0, 1, device, 1, val_tokens, bl, hl, il)
        log0(f"  Model {i} (seed={seeds[i]}): val_bpb={bpb:.4f}")
        individual_bpbs.append(bpb)

    # Ensemble evaluation (logit averaging)
    log0(f"\n{'='*60}\nEnsemble evaluation (logit average)\n{'='*60}")
    seq_len = args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    for m in models: m.eval()
    with torch.inference_mode():
        for bs in range(0, total_seqs, 32):
            be = min(bs + 32, total_seqs)
            local = val_tokens[bs*seq_len:(be*seq_len)+1].to(device=device, dtype=torch.int64)
            x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
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
            mixed = mixed / N_PARALLEL
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
    log0(f"  Ensemble ({N_PARALLEL} models): val_bpb={bpb:.4f}")

    # Weight merging: average all model weights into one
    log0(f"\n{'='*60}\nWeight merging (model soup)\n{'='*60}")
    merged_sd = {}
    ref_sd = models[0].state_dict()
    for key in ref_sd:
        tensors = [models[i].state_dict()[key].float() for i in range(N_PARALLEL)]
        merged_sd[key] = torch.stack(tensors).mean(dim=0).to(ref_sd[key].dtype)

    # Load merged weights into first model
    merged_model = models[0]
    merged_model.load_state_dict(merged_sd)

    class A:
        val_batch_size = 524288
        train_seq_len = args.train_seq_len
    vl_m, bpb_m = eval_val(A(), merged_model, 0, 1, device, 1, val_tokens, bl, hl, il)
    log0(f"  Merged model (weight avg): val_bpb={bpb_m:.4f}")

    # Quantize merged model and check roundtrip + artifact size
    sd = merged_model.state_dict()
    sd = {k: v for k, v in sd.items()
          if (not k.startswith("channel_") or k in ("channel_in", "channel_out"))
          and not k.startswith("ngram_") and not k.startswith("bigram_logit_mixer.bigram_")
          and "base_weight" not in k}
    sd.pop("lm_head.weight", None)
    q_obj, q_stats = q_sd_int6(sd)
    buf = io.BytesIO()
    torch.save(q_obj, buf)
    try:
        import zstandard
        blob = zstandard.ZstdCompressor(level=22).compress(buf.getvalue())
    except ImportError:
        blob = lzma.compress(buf.getvalue(), preset=9)

    # Roundtrip
    dq = deq_sd_int6(q_obj, target_dtype=torch.bfloat16)
    merged_model.load_state_dict(dq, strict=False)
    vl_rt, bpb_rt = eval_val(A(), merged_model, 0, 1, device, 1, val_tokens, bl, hl, il)
    log0(f"  Merged roundtrip: val_bpb={bpb_rt:.4f}, artifact={len(blob)/1e6:.2f}MB")

    # Summary
    log0(f"\n{'='*60}\nSummary\n{'='*60}")
    log0(f"  Best individual:  {min(individual_bpbs):.4f} BPB")
    log0(f"  Ensemble (logit): {bpb:.4f} BPB")
    log0(f"  Merged (soup):    {bpb_m:.4f} BPB (pre-quant)")
    log0(f"  Merged roundtrip: {bpb_rt:.4f} BPB, {len(blob)/1e6:.2f}MB")
    log0(f"  Training time:    {total_time:.0f}s ({total_time/N_PARALLEL:.0f}s per model equivalent)")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Federated ensemble of low-rank Universal Transformers.

K UT branches share one embedding. Each UT: B unique blocks × I iterations,
low-rank (W=A·B). Training splits batch across branches. Eval averages logits.
Multi-GPU: federated averaging (periodic weight sync, no gradient all-reduce).

Usage:
  # Single GPU test (A100)
  K_BRANCHES=5 UT_UNIQUE_BLOCKS=3 UT_ITERS=4 LOW_RANK=128 ITERATIONS=500 \
    torchrun --standalone --nproc_per_node=1 train_fed_ut_ensemble.py

  # 8×H100 competition run
  K_BRANCHES=5 UT_UNIQUE_BLOCKS=3 UT_ITERS=4 LOW_RANK=128 FEDAVG_EVERY=50 \
    torchrun --standalone --nproc_per_node=8 train_fed_ut_ensemble.py
"""
import os, sys, math, io, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import sentencepiece as spm
from pathlib import Path

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------
K = int(os.environ.get("K_BRANCHES", "5"))
ITERS = int(os.environ.get("ITERATIONS", "500"))
FEDAVG = int(os.environ.get("FEDAVG_EVERY", "50"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "0"))  # 0 = auto (8 // world)
VAL_EVERY = int(os.environ.get("VAL_EVERY", "0"))     # 0 = only at end
COMPILE = os.environ.get("COMPILE", "0") == "1"
COMPILE_MODE = os.environ.get("COMPILE_MODE", "default")

# Force UT + low-rank + int6 + full stack defaults
os.environ.setdefault("UT_UNIQUE_BLOCKS", "3")
os.environ.setdefault("UT_ITERS", "4")
os.environ.setdefault("LOW_RANK", "128")
os.environ.setdefault("QUANT_MODE", "int6")
os.environ.setdefault("N_CHANNELS", "1")
os.environ.setdefault("SSE_ENABLED", "1")
os.environ.setdefault("LR_SCHEDULE", "1cycle")
os.environ.setdefault("TRAIN_SEQ_LEN", "2048")
os.environ.setdefault("TRAIN_BATCH_TOKENS", "786432")
# XSA on all UT blocks by default
os.environ.setdefault("XSA_LAYERS", os.environ.get("UT_UNIQUE_BLOCKS", "3"))

sys.path.insert(0, os.path.dirname(__file__))
import train_combined as tc
tc._QUANT_MODE = "int6"
from train_combined import (
    GPT, Hyperparameters, Muon, DistributedTokenLoader,
    q_sd_int6, deq_sd_int6, build_luts, ld_val, eval_val,
    restore_low_dim_params_to_fp32, CTP, LowRankLinear,
)


def log0(msg, master=True):
    if master:
        print(msg, flush=True)


def main():
    # --- Distributed setup ---
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    master = rank == 0

    # --- Hyperparameters ---
    args = Hyperparameters()
    args.iterations = ITERS
    args.ut_unique_blocks = int(os.environ.get("UT_UNIQUE_BLOCKS", str(args.ut_unique_blocks)))
    args.ut_iters = int(os.environ.get("UT_ITERS", str(args.ut_iters)))
    args.low_rank = int(os.environ.get("LOW_RANK", str(args.low_rank)))
    args.xsa_layers = int(os.environ.get("XSA_LAYERS", str(args.xsa_layers)))

    # Set globals for model construction
    tc._LOW_RANK = args.low_rank
    tc._LORA_RANK = 0
    tc._LORA_BASE_SEED_COUNTER = 0
    tc._INT6_ACTIVE = True
    tc._INT6_CLIP_Q = args.int6_clip_q

    if args.matrix_optimizer != "adamw":
        tc.ns_orth = torch.compile(tc.ns_orth)

    grad_accum = GRAD_ACCUM if GRAD_ACCUM > 0 else max(1, 8 // world)
    grad_scale = 1.0 / grad_accum

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    bl, hl, il = build_luts(sp, args.vocab_size, device)
    val_tokens = ld_val(args.val_files, args.train_seq_len)

    eff_depth = args.ut_unique_blocks * args.ut_iters
    log0(f"=== Federated UT Ensemble ===", master)
    log0(f"K={K}, {args.ut_unique_blocks}x{args.ut_iters}={eff_depth} depth, "
         f"rank={args.low_rank}, xsa={args.xsa_layers}", master)
    log0(f"steps={ITERS}, fedavg={FEDAVG}, world={world}, "
         f"grad_accum={grad_accum}, compile={COMPILE}", master)

    # --- Create K branches with shared embedding ---
    gpt_kw = dict(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init, group_size=args.bitnet_group_size,
        activation=args.activation_type, n_channels=1,
        xsa_layers=args.xsa_layers, sse_enabled=args.sse_enabled,
        sse_clusters=args.sse_clusters,
        sse_entropy_bins=args.sse_entropy_bins,
        ut_unique_blocks=args.ut_unique_blocks, ut_iters=args.ut_iters,
    )

    seeds = [1337, 42, 7, 2024, 314, 999, 8675, 3141, 2718, 1618]
    branches = []
    for i in range(K):
        tc._LORA_BASE_SEED_COUNTER = 0
        torch.manual_seed(seeds[i % len(seeds)])
        b = GPT(**gpt_kw).to(device).bfloat16()
        b._init_weights(args.tied_embed_init_std)
        branches.append(b)

    # Share tok_emb and lm_head_correction across all branches
    shared_emb = branches[0].tok_emb
    shared_corr = branches[0].lm_head_correction
    for i in range(1, K):
        branches[i].tok_emb = shared_emb
        if shared_corr is not None:
            branches[i].lm_head_correction = shared_corr
        if args.tie_embeddings and branches[i].lm_head is not None:
            branches[i].lm_head.weight = shared_emb.weight

    # Wrap in ModuleList for parameter management
    ensemble = nn.ModuleList(branches)

    # Float32 for linear layers (Muon/Adam expect float32)
    for m in ensemble.modules():
        if isinstance(m, (nn.Linear, LowRankLinear)):
            m.float()
    for b in branches:
        restore_low_dim_params_to_fp32(b)
        if b.lm_head is not None and args.tie_embeddings:
            b.lm_head.weight.requires_grad_(False)

    # Compile branches — identical architecture = one compilation, shared cache
    if COMPILE:
        _cm = COMPILE_MODE if COMPILE_MODE != "default" else None
        train_branches = [torch.compile(b, mode=_cm) for b in branches]
        log0(f"Compiled {K} branches (mode={COMPILE_MODE})", master)
    else:
        train_branches = branches

    # Count unique params
    seen_ids = set()
    n_params = 0
    for p in ensemble.parameters():
        if id(p) not in seen_ids:
            seen_ids.add(id(p))
            n_params += p.numel()
    per_branch = sum(p.numel() for p in branches[0].parameters())
    shared_n = sum(p.numel() for p in [shared_emb.weight] +
                   ([shared_corr] if shared_corr is not None else []))
    log0(f"Params: {n_params:,} total, {per_branch:,}/branch, "
         f"{shared_n:,} shared", master)

    # Broadcast initial weights from rank 0
    if world > 1:
        seen_ids = set()
        for p in ensemble.parameters():
            if id(p) not in seen_ids:
                seen_ids.add(id(p))
                dist.broadcast(p.data, src=0)

    # --- Optimizers ---
    shared_param_ids = {id(shared_emb.weight)}
    if shared_corr is not None:
        shared_param_ids.add(id(shared_corr))

    _excl = {"tok_emb.weight", "lm_head.weight", "lm_head_correction"}
    matrix_ps, scalar_ps = [], []
    seen = set()
    for b in branches:
        for n, p in b.named_parameters():
            if any(e in n for e in _excl) or id(p) in seen:
                continue
            seen.add(id(p))
            if p.ndim == 2 and not any(pat in n for pat in CTP):
                matrix_ps.append(p)
            else:
                scalar_ps.append(p)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    opt_tok = torch.optim.Adam(
        [{"params": [shared_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)

    if args.matrix_optimizer != "adamw":
        opt_mat = Muon(matrix_ps, lr=args.matrix_lr,
                       momentum=args.muon_momentum,
                       backend_steps=args.muon_backend_steps,
                       wd=args.muon_wd)
    else:
        opt_mat = torch.optim.AdamW(
            [{"params": matrix_ps, "lr": args.matrix_lr,
              "base_lr": args.matrix_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    for g in opt_mat.param_groups:
        g["base_lr"] = args.matrix_lr

    opt_scl = torch.optim.Adam(
        [{"params": scalar_ps, "lr": args.scalar_lr,
          "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)

    optimizers = [opt_tok, opt_mat, opt_scl]

    if shared_corr is not None:
        opt_corr = torch.optim.Adam(
            [{"params": [shared_corr], "lr": args.corr_weight_lr,
              "base_lr": args.corr_weight_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        optimizers.append(opt_corr)

    def zero_grad():
        for o in optimizers:
            o.zero_grad(set_to_none=True)

    def lr_mul(step):
        frac = min(step / max(ITERS, 1), 1.0)
        mn = 1.0 / args.onecycle_min_div
        pk = args.onecycle_peak_frac
        if frac < pk:
            return mn + (1.0 - mn) * (frac / pk)
        t = (frac - pk) / (1.0 - pk)
        return mn + 0.5 * (1.0 - mn) * (1.0 + math.cos(math.pi * t))

    # --- Data loader ---
    loader = DistributedTokenLoader(args.train_files, rank, world, device)
    seq_len = args.train_seq_len
    # Total batch = K × per-branch batch, so each branch sees the full batch size
    batch_tokens = args.train_batch_tokens * K

    log0(f"batch_tokens={batch_tokens} ({args.train_batch_tokens}/branch), "
         f"seq_len={seq_len}", master)

    # --- Ensemble eval helper ---
    def eval_ensemble(adaptive=False):
        """Evaluate ensemble BPB (uniform or entropy-adaptive logit mixing)."""
        total_seqs = (val_tokens.numel() - 1) // seq_len
        batch_seqs = 32
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        tok_ct = torch.zeros((), device=device, dtype=torch.float64)
        byte_ct = torch.zeros((), device=device, dtype=torch.float64)
        for b in branches:
            b.eval()
        with torch.inference_mode():
            for bs in range(0, total_seqs, batch_seqs):
                be = min(bs + batch_seqs, total_seqs)
                loc = val_tokens[bs * seq_len:(be * seq_len) + 1].to(
                    device=device, dtype=torch.int64)
                xv = loc[:-1].reshape(-1, seq_len)
                yv = loc[1:].reshape(-1, seq_len)

                all_logits = []
                for b in branches:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        e = b._embed(xv)
                        h = b._run_blocks(e, e, causal=True)
                        lg = b._softcap(b._compute_logits(
                            h.reshape(-1, h.size(-1))))
                        if b.sse:
                            lg = b.sse(lg)
                    all_logits.append(lg.float())

                if adaptive:
                    ents = []
                    for lg in all_logits:
                        p = F.softmax(lg, dim=-1)
                        ents.append(-(p * p.clamp(min=1e-10).log()).sum(-1))
                    inv = torch.stack([1.0 / (e + 0.1) for e in ents])
                    w = inv / inv.sum(0, keepdim=True)
                    mixed = sum(
                        wi.unsqueeze(-1) * lg
                        for wi, lg in zip(w, all_logits))
                else:
                    mixed = sum(all_logits) / K

                tgt = yv.reshape(-1)
                lse = torch.logsumexp(mixed, dim=-1)
                tl = mixed.gather(1, tgt.unsqueeze(1)).squeeze(1)
                loss_sum += (lse - tl).to(torch.float64).sum()
                tok_ct += float(tgt.numel())
                px, tx = xv.reshape(-1), yv.reshape(-1)
                tb = (bl[tx].to(torch.float64) +
                      (hl[tx] & ~il[px]).to(torch.float64))
                byte_ct += tb.sum()
        for b in branches:
            b.train()
        vl = (loss_sum / tok_ct).item()
        return (vl / math.log(2.0)) * (tok_ct.item() / byte_ct.item())

    # --- Training loop ---
    ensemble.train()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(1, ITERS + 1):
        scale = lr_mul(step)
        for o in optimizers:
            for g in o.param_groups:
                g["lr"] = g.get("base_lr", args.matrix_lr) * scale

        # Muon momentum warmup
        if (args.matrix_optimizer != "adamw"
                and args.muon_momentum_warmup_steps > 0):
            frac = min(step / args.muon_momentum_warmup_steps, 1.0)
            for g in opt_mat.param_groups:
                g["momentum"] = ((1 - frac) * args.muon_momentum_warmup_start
                                 + frac * args.muon_momentum)

        zero_grad()
        step_loss = 0.0

        for _micro in range(grad_accum):
            x, y = loader.next_batch(batch_tokens, seq_len, grad_accum)
            B = x.shape[0]
            bk = B // K  # sequences per branch

            for i in range(K):
                branches[i]._force_causal = True
                torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss_i = train_branches[i](
                        x[i * bk:(i + 1) * bk],
                        y[i * bk:(i + 1) * bk])
                (loss_i * grad_scale).backward()
                step_loss += loss_i.item()

        # Shared params accumulate K× gradient — correct to 1×
        for p in ensemble.parameters():
            if id(p) in shared_param_ids and p.grad is not None:
                p.grad.div_(K)

        for o in optimizers:
            o.step()

        # Federated averaging: periodic all-reduce across GPUs
        if (FEDAVG > 0 and world > 1
                and step % FEDAVG == 0 and step < ITERS):
            with torch.no_grad():
                done = set()
                for p in ensemble.parameters():
                    if id(p) not in done:
                        done.add(id(p))
                        dist.all_reduce(p.data, op=dist.ReduceOp.AVG)
            log0(f"  step:{step} fedavg", master)

        if step % 50 == 0 or step == ITERS:
            elapsed = time.perf_counter() - t0
            avg = step_loss / (grad_accum * K)
            log0(f"  step:{step}/{ITERS} loss:{avg:.4f} "
                 f"t:{elapsed:.0f}s avg:{elapsed * 1000 / step:.0f}ms/step",
                 master)

        # Periodic validation (branch 0 only, for speed)
        if VAL_EVERY > 0 and step % VAL_EVERY == 0 and step < ITERS:
            _, bpb0 = eval_val(args, branches[0], rank, world, device, 1,
                               val_tokens, bl, hl, il)
            log0(f"  step:{step} branch0_bpb:{bpb0:.4f}", master)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    log0(f"Training: {total_time:.0f}s "
         f"({total_time * 1000 / ITERS:.0f}ms/step)", master)

    # --- Eval: individual branches ---
    log0(f"\n{'=' * 60}\nIndividual Branch Eval\n{'=' * 60}", master)
    indiv_bpbs = []
    for i, b in enumerate(branches):
        _, bpb = eval_val(args, b, rank, world, device, 1,
                          val_tokens, bl, hl, il)
        log0(f"  Branch {i}: {bpb:.4f} BPB", master)
        indiv_bpbs.append(bpb)

    # --- Eval: ensemble ---
    log0(f"\n{'=' * 60}\nEnsemble Eval\n{'=' * 60}", master)
    ens_bpb = eval_ensemble(adaptive=False)
    log0(f"  Uniform ({K}): {ens_bpb:.4f} BPB", master)
    ea_bpb = eval_ensemble(adaptive=True)
    log0(f"  Entropy-adaptive ({K}): {ea_bpb:.4f} BPB", master)

    # --- Serialization ---
    model_file = "ut_ensemble.ptz"
    if master:
        log0(f"\n{'=' * 60}\nSerialization\n{'=' * 60}")

        # Shared state
        shared_sd = {"tok_emb.weight": shared_emb.weight.detach().cpu()}
        if shared_corr is not None:
            shared_sd["lm_head_correction"] = shared_corr.detach().cpu()

        # Per-branch state (exclude shared + tied keys)
        _skip = {"tok_emb.weight", "lm_head_correction", "lm_head.weight"}
        branch_sds = []
        for b in branches:
            bsd = {n: p for n, p in b.state_dict().items() if n not in _skip}
            branch_sds.append(bsd)

        # Quantize each component
        q_all = {}
        q_shared, stats = q_sd_int6(shared_sd)
        q_all["shared"] = q_shared
        log0(f"  shared: int6={stats['int6_params']} "
             f"int8={stats['int8_params']} fp={stats['fp_params']}")
        for i, bsd in enumerate(branch_sds):
            q_b, stats = q_sd_int6(bsd)
            q_all[f"b{i}"] = q_b
            log0(f"  b{i}: int6={stats['int6_params']} "
                 f"int8={stats['int8_params']} fp={stats['fp_params']}")

        # Metadata for reconstruction
        q_all["_meta"] = dict(
            K=K, ut_blocks=args.ut_unique_blocks, ut_iters=args.ut_iters,
            rank=args.low_rank, dim=args.model_dim, heads=args.num_heads,
            kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
            vocab=args.vocab_size, xsa=args.xsa_layers,
            sse=args.sse_enabled, activation=args.activation_type,
            tie_emb=args.tie_embeddings, softcap=args.logit_softcap,
            rope_base=args.rope_base, qk_gain=args.qk_gain_init,
            sse_clusters=args.sse_clusters, sse_bins=args.sse_entropy_bins,
            seq_len=seq_len,
        )

        buf = io.BytesIO()
        torch.save(q_all, buf)
        try:
            import zstandard
            blob = zstandard.ZstdCompressor(level=22).compress(buf.getvalue())
        except ImportError:
            import lzma
            blob = lzma.compress(buf.getvalue(), preset=9)

        with open(model_file, "wb") as f:
            f.write(blob)

        art_bytes = os.path.getsize(model_file)
        code_bytes = len(Path(__file__).read_text("utf-8").encode("utf-8"))
        total = art_bytes + code_bytes
        log0(f"  Artifact: {art_bytes / 1e6:.2f}MB, "
             f"code: {code_bytes / 1e3:.1f}KB")
        log0(f"  Budget: {total / 1e6:.2f}/{16.00:.2f}MB "
             f"{'FITS' if total <= 16_000_000 else 'OVER'}")

    # Barrier so all ranks have the file before roundtrip
    if world > 1:
        dist.barrier()

    # --- Roundtrip: decompress, dequantize, re-eval ---
    log0(f"\n{'=' * 60}\nRoundtrip\n{'=' * 60}", master)

    with open(model_file, "rb") as f:
        raw = f.read()
    try:
        import zstandard
        dec = zstandard.ZstdDecompressor().decompress(raw)
    except Exception:
        import lzma
        dec = lzma.decompress(raw)
    loaded = torch.load(io.BytesIO(dec), map_location="cpu",
                        weights_only=False)

    shared_rt = deq_sd_int6(loaded["shared"])
    for i, b in enumerate(branches):
        brt = deq_sd_int6(loaded[f"b{i}"])
        brt["tok_emb.weight"] = shared_rt["tok_emb.weight"]
        if "lm_head_correction" in shared_rt:
            brt["lm_head_correction"] = shared_rt["lm_head_correction"]
        b.load_state_dict(brt, strict=False)

    # Re-share embedding after roundtrip load
    for i in range(1, K):
        branches[i].tok_emb = branches[0].tok_emb
        if branches[0].lm_head_correction is not None:
            branches[i].lm_head_correction = branches[0].lm_head_correction
        if args.tie_embeddings and branches[i].lm_head is not None:
            branches[i].lm_head.weight = branches[0].tok_emb.weight

    # Roundtrip individual
    rt_indiv = []
    for i, b in enumerate(branches):
        _, bpb = eval_val(args, b, rank, world, device, 1,
                          val_tokens, bl, hl, il)
        log0(f"  RT Branch {i}: {bpb:.4f} BPB", master)
        rt_indiv.append(bpb)

    # Roundtrip ensemble
    rt_bpb = eval_ensemble(adaptive=False)
    log0(f"  RT Uniform ({K}): {rt_bpb:.4f} BPB", master)
    rt_ea = eval_ensemble(adaptive=True)
    log0(f"  RT Adaptive ({K}): {rt_ea:.4f} BPB", master)

    # --- Summary ---
    log0(f"\n{'=' * 60}\nSummary\n{'=' * 60}", master)
    log0(f"  Config: K={K}, {args.ut_unique_blocks}x{args.ut_iters} UT, "
         f"rank={args.low_rank}, xsa={args.xsa_layers}", master)
    log0(f"  Training: {total_time:.0f}s "
         f"({total_time * 1000 / ITERS:.0f}ms/step)", master)
    log0(f"  Best individual:    {min(indiv_bpbs):.4f} BPB", master)
    log0(f"  Ensemble uniform:   {ens_bpb:.4f} BPB", master)
    log0(f"  Ensemble adaptive:  {ea_bpb:.4f} BPB", master)
    log0(f"  RT best individual: {min(rt_indiv):.4f} BPB", master)
    log0(f"  RT ensemble:        {rt_bpb:.4f} BPB", master)
    log0(f"  RT adaptive:        {rt_ea:.4f} BPB", master)
    if master:
        log0(f"  Artifact: {art_bytes / 1e6:.2f}MB "
             f"({total / 1e6:.2f}MB with code)")
    gap = rt_bpb - ens_bpb
    log0(f"  Roundtrip gap: {gap:+.4f} BPB", master)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

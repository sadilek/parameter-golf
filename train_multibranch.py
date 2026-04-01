"""Multi-branch parallel training: N independent model branches in one forward pass.

Each branch is a complete GPT model with its own weights. The batch is split
across branches — each sees different data. One backward pass trains all branches
simultaneously, with better GPU utilization than sequential training.

Usage:
  N_BRANCHES=4 torchrun --standalone --nproc_per_node=1 train_multibranch.py
"""
import os, sys, math, glob, copy, io, lzma, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import sentencepiece as spm
from pathlib import Path

N_BRANCHES = int(os.environ.get("N_BRANCHES", "2"))
ITERS = int(os.environ.get("MB_ITERS", "500"))
LAYERS = int(os.environ.get("MB_LAYERS", "7"))
SAME_SEED = os.environ.get("SAME_SEED", "0") == "1"  # All branches same init
AVG_EVERY = int(os.environ.get("AVG_EVERY", "0"))  # Periodic weight averaging (0=disabled)

os.environ.setdefault("QUANT_MODE", "int6")
os.environ.setdefault("N_CHANNELS", "1")
os.environ.setdefault("XSA_LAYERS", "2")
os.environ.setdefault("SSE_ENABLED", "1")
os.environ["NUM_LAYERS"] = str(LAYERS)
os.environ["COMPILE_MODE"] = "off"  # compile doesn't work well with custom forward

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


class MultiBranchModel(nn.Module):
    """N independent GPT branches that process different parts of the batch."""

    def __init__(self, n_branches: int, **gpt_kwargs):
        super().__init__()
        self.n_branches = n_branches
        self.branches = nn.ModuleList([
            GPT(**gpt_kwargs) for _ in range(n_branches)
        ])

    def forward(self, input_ids, target_ids):
        """Split batch across branches, backward each separately (correct gradient scaling)."""
        B = input_ids.shape[0]
        branch_B = B // self.n_branches
        total_loss = torch.zeros((), device=input_ids.device)
        self._branch_losses = []

        for i, branch in enumerate(self.branches):
            start = i * branch_B
            end = start + branch_B
            x_i = input_ids[start:end]
            y_i = target_ids[start:end]
            branch._force_causal = True
            loss_i = branch(x_i, y_i)
            self._branch_losses.append(loss_i)
            total_loss = total_loss + loss_i.detach()

        return total_loss / self.n_branches  # for logging only

    def backward_branches(self, grad_scale: float):
        """Backward each branch loss separately — correct gradient scaling."""
        for loss_i in self._branch_losses:
            (loss_i * grad_scale).backward()


def main():
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    args = Hyperparameters()
    args.num_layers = LAYERS
    args.iterations = ITERS
    sp = spm.SentencePieceProcessor(args.tokenizer_path)
    bl, hl, il = build_luts(sp, args.vocab_size, device)
    val_tokens = ld_val(args.val_files, args.train_seq_len)

    log0(f"Multi-branch: {N_BRANCHES} branches × {LAYERS}L/512d, {ITERS} steps, same_seed={SAME_SEED}, avg_every={AVG_EVERY}")

    gpt_kwargs = dict(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        group_size=args.bitnet_group_size, activation=args.activation_type,
        n_channels=args.n_channels, xsa_layers=args.xsa_layers,
        sse_enabled=args.sse_enabled, sse_clusters=args.sse_clusters,
        sse_entropy_bins=args.sse_entropy_bins,
    )

    # Init branches
    if SAME_SEED:
        # Same init for all branches (for federated averaging / merging)
        torch.manual_seed(1337)
        multi_model = MultiBranchModel(N_BRANCHES, **gpt_kwargs).to(device).bfloat16()
        # Copy branch 0's weights to all other branches
        ref_sd = multi_model.branches[0].state_dict()
        for i in range(1, N_BRANCHES):
            multi_model.branches[i].load_state_dict(ref_sd)
        log0(f"  SAME_SEED mode: all branches initialized identically")
    else:
        torch.manual_seed(0)
        multi_model = MultiBranchModel(N_BRANCHES, **gpt_kwargs).to(device).bfloat16()
        for i, branch in enumerate(multi_model.branches):
            seed = [1337, 42, 7, 2024, 314, 999][i % 6]
            torch.manual_seed(seed)
            branch._init_weights(args.tied_embed_init_std)
        log0(f"  Different seeds per branch")

    for m in multi_model.modules():
        if isinstance(m, nn.Linear):
            m.float()
    # Restore scalar params to fp32 per branch
    for branch in multi_model.branches:
        restore_low_dim_params_to_fp32(branch)
        if branch.lm_head is not None and args.tie_embeddings:
            branch.lm_head.weight.requires_grad_(False)

    n_params = sum(p.numel() for p in multi_model.parameters())
    log0(f"  total params: {n_params:,} ({n_params // N_BRANCHES:,} per branch)")

    # Optimizers: one per branch (so LR scheduling works correctly)
    all_optimizers = []
    for branch in multi_model.branches:
        _excl = {"tok_emb.weight", "lm_head.weight", "lm_head_correction"}
        all_other = [(n, p) for n, p in branch.named_parameters() if not any(e in n for e in _excl)]
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
            [{"params": [branch.tok_emb.weight], "lr": args.embed_lr, "base_lr": args.embed_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        all_optimizers.extend([opt_tok, opt_muon, opt_scalar])

    def zero_grad_all():
        for opt in all_optimizers:
            opt.zero_grad(set_to_none=True)

    def lr_mul(step):
        frac = min(step / max(ITERS, 1), 1.0)
        min_mul = 1.0 / 4.0
        peak = 0.3
        if frac < peak:
            return min_mul + (1.0 - min_mul) * (frac / peak)
        else:
            t = (frac - peak) / (1.0 - peak)
            return min_mul + 0.5 * (1.0 - min_mul) * (1.0 + math.cos(math.pi * t))

    # Data loader: one shared loader, branches get different parts of each batch
    loader = DistributedTokenLoader(args.train_files, 0, 1, device)
    # Multiply batch size by N_BRANCHES so each branch gets full-size batches
    batch_tokens = args.train_batch_tokens * N_BRANCHES
    grad_accum = 8
    grad_scale = 1.0 / grad_accum

    log0(f"  batch_tokens: {batch_tokens:,} ({args.train_batch_tokens:,} per branch)")
    log0(f"  grad_accum: {grad_accum}")

    # Training loop
    multi_model.train()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(1, ITERS + 1):
        scale = lr_mul(step)
        for opt in all_optimizers:
            for g in opt.param_groups:
                g["lr"] = g.get("base_lr", args.matrix_lr) * scale

        zero_grad_all()
        for micro in range(grad_accum):
            x, y = loader.next_batch(batch_tokens, args.train_seq_len, grad_accum)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = multi_model(x, y)
            # Separate backward for each branch (correct gradient scaling)
            multi_model.backward_branches(grad_scale)

        for opt in all_optimizers:
            opt.step()

        # Periodic weight averaging (Federated Averaging)
        if AVG_EVERY > 0 and step % AVG_EVERY == 0 and step < ITERS:
            with torch.no_grad():
                ref_sd = multi_model.branches[0].state_dict()
                avg_sd = {}
                for key in ref_sd:
                    tensors = [multi_model.branches[i].state_dict()[key].float() for i in range(N_BRANCHES)]
                    avg_sd[key] = torch.stack(tensors).mean(dim=0)
                for branch in multi_model.branches:
                    branch.load_state_dict({k: v.to(branch.state_dict()[k].dtype) for k, v in avg_sd.items()})
            log0(f"  step:{step} weight_avg applied")

        if step % 50 == 0 or step == ITERS:
            elapsed = time.perf_counter() - t0
            log0(f"  step:{step}/{ITERS} loss:{loss.item():.4f} t:{elapsed*1000:.0f}ms avg:{elapsed*1000/step:.1f}ms")

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    log0(f"Training done: {total_time:.0f}s")

    # Evaluate each branch individually
    log0(f"\n{'='*60}\nIndividual Branch Evaluation\n{'='*60}")
    class A:
        val_batch_size = 524288
        train_seq_len = args.train_seq_len

    individual_bpbs = []
    for i, branch in enumerate(multi_model.branches):
        vl, bpb = eval_val(A(), branch, 0, 1, device, 1, val_tokens, bl, hl, il)
        log0(f"  Branch {i}: val_bpb={bpb:.4f}")
        individual_bpbs.append(bpb)

    # Ensemble evaluation (logit average)
    log0(f"\n{'='*60}\nEnsemble Evaluation\n{'='*60}")
    seq_len = args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    for branch in multi_model.branches: branch.eval()
    with torch.inference_mode():
        for bs in range(0, total_seqs, 32):
            be = min(bs + 32, total_seqs)
            local = val_tokens[bs*seq_len:(be*seq_len)+1].to(device=device, dtype=torch.int64)
            x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
            mixed = None
            for branch in multi_model.branches:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    e = branch._embed(x); h = branch._run_blocks(e, e, causal=True)
                    l = branch._softcap(branch._compute_logits(h.reshape(-1, h.size(-1))))
                    if branch.sse: l = branch.sse(l)
                if mixed is None:
                    mixed = l.float()
                else:
                    mixed = mixed + l.float()
            mixed = mixed / N_BRANCHES
            targets = y.reshape(-1)
            lse = torch.logsumexp(mixed, dim=-1)
            tgt = mixed.gather(1, targets.unsqueeze(1)).squeeze(1)
            loss_sum += (lse - tgt).to(torch.float64).sum()
            token_count += float(targets.numel())
            p, t = x.reshape(-1), y.reshape(-1)
            tb = bl[t].to(torch.float64) + (hl[t] & ~il[p]).to(torch.float64)
            byte_count += tb.sum()
    vl = (loss_sum / token_count).item()
    ens_bpb = (vl / math.log(2.0)) * (token_count.item() / byte_count.item())
    log0(f"  Ensemble ({N_BRANCHES}): val_bpb={ens_bpb:.4f}")

    # Weight merging (soup) - test even though it probably won't work with different seeds
    log0(f"\n{'='*60}\nWeight Merge (Soup)\n{'='*60}")
    merged_sd = {}
    ref_sd = multi_model.branches[0].state_dict()
    for key in ref_sd:
        tensors = [multi_model.branches[i].state_dict()[key].float() for i in range(N_BRANCHES)]
        merged_sd[key] = torch.stack(tensors).mean(dim=0).to(ref_sd[key].dtype)
    merge_model = multi_model.branches[0]
    merge_model.load_state_dict(merged_sd)
    vl_m, bpb_m = eval_val(A(), merge_model, 0, 1, device, 1, val_tokens, bl, hl, il)
    log0(f"  Merged: val_bpb={bpb_m:.4f}")

    # Summary
    log0(f"\n{'='*60}\nSummary\n{'='*60}")
    log0(f"  {N_BRANCHES} branches, {LAYERS}L each, {ITERS} steps")
    log0(f"  Training time: {total_time:.0f}s ({total_time*1000/ITERS:.1f}ms/step)")
    log0(f"  Sequential baseline: ~{N_BRANCHES * total_time / N_BRANCHES:.0f}s (est)")
    log0(f"  Best individual: {min(individual_bpbs):.4f} BPB")
    log0(f"  Ensemble:         {ens_bpb:.4f} BPB")
    log0(f"  Merged:           {bpb_m:.4f} BPB")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

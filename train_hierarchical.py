"""Hierarchical multi-resolution transformer for Parameter Golf.

Architecture:
  1. Long input (4096 tokens) at low embed_dim (32)
  2. Split into overlapping windows (256 tokens, stride 128) → 31 windows
  3. Shared window encoder compresses each window → 1 token at model_dim
  4. Global transformer over 31 compressed tokens (cheap: 31×31 attention)
  5. Cross-attend from full-res last window to global context for prediction

Key insight: distant context needs less resolution. We get 4× the context
at ~2× the compute vs standard T=1024.

Usage:
  torchrun --standalone --nproc_per_node=1 train_hierarchical.py
"""
import os, sys, math, time, io, random, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import sentencepiece as spm
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOTAL_SEQ = int(os.environ.get("TOTAL_SEQ", "4096"))       # full context length
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "256"))     # local window size
WINDOW_STRIDE = int(os.environ.get("WINDOW_STRIDE", "128")) # overlap
EMBED_DIM = int(os.environ.get("EMBED_DIM", "32"))         # low-dim token embedding
MODEL_DIM = int(os.environ.get("MODEL_DIM", "512"))        # global transformer dim
N_GLOBAL_LAYERS = int(os.environ.get("N_GLOBAL_LAYERS", "8"))
N_LOCAL_LAYERS = int(os.environ.get("N_LOCAL_LAYERS", "2"))  # layers in window encoder
N_HEADS = int(os.environ.get("N_HEADS", "8"))
MLP_MULT = int(os.environ.get("MLP_MULT", "3"))
VOCAB_SIZE = 1024
ITERS = int(os.environ.get("ITERATIONS", "500"))

N_WINDOWS = (TOTAL_SEQ - WINDOW_SIZE) // WINDOW_STRIDE + 1


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),)) * self.scale


class SwiGLUMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_up = nn.Linear(dim, hidden * 2, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        gu = self.gate_up(x)
        g, u = gu.chunk(2, dim=-1)
        return self.proj(F.silu(g) * u)


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, causal=True):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class CrossAttention(nn.Module):
    """Query attends to key-value context."""
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, query, context):
        B, Tq, C = query.shape
        Tc = context.shape[1]
        q = self.q_proj(query).reshape(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, Tc, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.out_proj(out.transpose(1, 2).reshape(B, Tq, C))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_mult, cross_attn=False):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLUMLP(dim, dim * mlp_mult)
        self.cross_attn = None
        if cross_attn:
            self.norm_cross = RMSNorm(dim)
            self.cross_attn = CrossAttention(dim, n_heads)

    def forward(self, x, context=None, causal=True):
        x = x + self.attn(self.norm1(x), causal=causal)
        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Window encoder: compress (WINDOW_SIZE, EMBED_DIM) → (1, MODEL_DIM)
# ---------------------------------------------------------------------------
class WindowEncoder(nn.Module):
    """Compress a window of tokens into a single summary vector.

    Uses strided convolutions to progressively reduce sequence length
    while increasing channel dimension.
    """
    def __init__(self, embed_dim, model_dim, window_size):
        super().__init__()
        # Progressive compression: 256→32→4→1
        self.layers = nn.Sequential(
            nn.Conv1d(embed_dim, model_dim // 4, kernel_size=8, stride=8),
            nn.SiLU(),
            nn.Conv1d(model_dim // 4, model_dim // 2, kernel_size=4, stride=4),
            nn.SiLU(),
            nn.Conv1d(model_dim // 2, model_dim, kernel_size=window_size // 32, stride=window_size // 32),
            nn.SiLU(),
        )
        # Final projection to exactly 1 token
        compressed_len = 1  # 256 / 8 / 4 / 8 = 1
        self.proj = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        """x: (B*N_WINDOWS, WINDOW_SIZE, EMBED_DIM) → (B*N_WINDOWS, 1, MODEL_DIM)"""
        # Conv1d expects (B, C, T)
        h = self.layers(x.transpose(1, 2))  # → (B*N, MODEL_DIM, 1)
        h = h.squeeze(-1)  # → (B*N, MODEL_DIM)
        return self.proj(h).unsqueeze(1)  # → (B*N, 1, MODEL_DIM)


# ---------------------------------------------------------------------------
# Full hierarchical model
# ---------------------------------------------------------------------------
class HierarchicalGPT(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, model_dim=MODEL_DIM,
                 n_global_layers=N_GLOBAL_LAYERS, n_local_layers=N_LOCAL_LAYERS,
                 n_heads=N_HEADS, mlp_mult=MLP_MULT,
                 window_size=WINDOW_SIZE, window_stride=WINDOW_STRIDE,
                 total_seq=TOTAL_SEQ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.window_size = window_size
        self.window_stride = window_stride
        self.total_seq = total_seq
        self.n_windows = (total_seq - window_size) // window_stride + 1

        # Token embedding (low-dim)
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)

        # Window encoder (shared across all windows)
        self.window_encoder = WindowEncoder(embed_dim, model_dim, window_size)

        # Positional embedding for global sequence
        self.global_pos = nn.Parameter(torch.randn(1, self.n_windows, model_dim) * 0.02)

        # Global transformer (processes compressed windows)
        self.global_blocks = nn.ModuleList([
            TransformerBlock(model_dim, n_heads, mlp_mult, cross_attn=False)
            for _ in range(n_global_layers)
        ])

        # Local decoder: processes last window at full resolution with global context
        self.local_embed_proj = nn.Linear(embed_dim, model_dim, bias=False)
        self.local_pos = nn.Parameter(torch.randn(1, window_size, model_dim) * 0.02)
        self.local_blocks = nn.ModuleList([
            TransformerBlock(model_dim, n_heads, mlp_mult, cross_attn=True)
            for _ in range(n_local_layers)
        ])

        # Output
        self.final_norm = RMSNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

        # Tie embeddings: project model_dim → embed_dim → vocab lookup
        # (or use a separate head since dims differ)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids, target_ids):
        """
        input_ids: (B, TOTAL_SEQ) — full long context
        target_ids: (B, TOTAL_SEQ) — shifted targets

        We predict only the LAST WINDOW's tokens (the most recent).
        """
        B, T = input_ids.shape
        assert T == self.total_seq, f"Expected seq_len={self.total_seq}, got {T}"

        # 1. Embed all tokens at low dim
        emb = self.tok_emb(input_ids)  # (B, T, embed_dim)

        # 2. Extract overlapping windows
        windows = []
        for i in range(self.n_windows):
            start = i * self.window_stride
            end = start + self.window_size
            windows.append(emb[:, start:end, :])  # (B, window_size, embed_dim)
        windows = torch.stack(windows, dim=1)  # (B, N_WINDOWS, window_size, embed_dim)

        # 3. Compress each window → 1 token
        BN = B * self.n_windows
        flat_windows = windows.reshape(BN, self.window_size, self.embed_dim)
        compressed = self.window_encoder(flat_windows)  # (BN, 1, model_dim)
        global_seq = compressed.reshape(B, self.n_windows, self.model_dim)
        global_seq = global_seq + self.global_pos

        # 4. Global transformer (causal over compressed tokens)
        for block in self.global_blocks:
            global_seq = block(global_seq, causal=True)

        # 5. Local decoder: last window at full resolution + global context
        last_window_start = (self.n_windows - 1) * self.window_stride
        last_window_emb = emb[:, last_window_start:last_window_start + self.window_size, :]
        local_seq = self.local_embed_proj(last_window_emb) + self.local_pos  # (B, window_size, model_dim)

        for block in self.local_blocks:
            local_seq = block(local_seq, context=global_seq, causal=True)

        # 6. Predict
        logits = self.lm_head(self.final_norm(local_seq))  # (B, window_size, vocab_size)

        # Loss: only on the last window's tokens
        last_targets = target_ids[:, last_window_start:last_window_start + self.window_size]
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), last_targets.reshape(-1))

        return loss

    def get_prediction_tokens(self):
        """Number of tokens we predict per sequence (for BPB calculation)."""
        return self.window_size


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", "0"))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    master = rank == 0

    model = HierarchicalGPT().to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            m.float()

    n_params = sum(p.numel() for p in model.parameters())
    if master:
        print(f"HierarchicalGPT: {n_params:,} params")
        print(f"  total_seq={TOTAL_SEQ}, windows={N_WINDOWS}×{WINDOW_SIZE} (stride {WINDOW_STRIDE})")
        print(f"  embed_dim={EMBED_DIM}, model_dim={MODEL_DIM}")
        print(f"  global_layers={N_GLOBAL_LAYERS}, local_layers={N_LOCAL_LAYERS}")
        print(f"  predict last {WINDOW_SIZE} tokens per sequence")

    # Simple Adam optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # Data
    sys.path.insert(0, os.path.dirname(__file__))
    from train_combined import DistributedTokenLoader, ld_val, build_luts, Hyperparameters
    args = Hyperparameters()
    loader = DistributedTokenLoader(args.train_files, rank, 1, device)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = ld_val(args.val_files, TOTAL_SEQ)
    bl, hl, il = build_luts(sp, VOCAB_SIZE, device)

    # Batch: we need TOTAL_SEQ+1 tokens per sequence
    batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", "262144"))
    seqs_per_batch = batch_tokens // TOTAL_SEQ

    if master:
        print(f"  batch_tokens={batch_tokens}, seqs_per_batch={seqs_per_batch}")

    # Training loop
    model.train()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(1, ITERS + 1):
        # LR schedule: warmup + cosine decay
        warmup = 100
        if step < warmup:
            lr = 3e-4 * step / warmup
        else:
            frac = (step - warmup) / max(ITERS - warmup, 1)
            lr = 3e-4 * 0.5 * (1 + math.cos(math.pi * frac))
        for g in optimizer.param_groups:
            g['lr'] = lr

        optimizer.zero_grad(set_to_none=True)

        # Get batch
        x, y = loader.next_batch(batch_tokens, TOTAL_SEQ, 1)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == ITERS:
            elapsed = time.perf_counter() - t0
            if master:
                print(f"  step:{step}/{ITERS} loss:{loss.item():.4f} "
                      f"t:{elapsed:.0f}s avg:{elapsed*1000/step:.0f}ms/step")

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    if master:
        print(f"Training: {total_time:.0f}s ({total_time*1000/ITERS:.0f}ms/step)")

    # Eval: compute BPB on val set
    if master:
        model.eval()
        total_seqs = (val_tokens.numel() - 1) // TOTAL_SEQ
        loss_sum = 0.0
        token_count = 0
        byte_count = 0.0

        with torch.inference_mode():
            for i in range(0, min(total_seqs, 200), 4):  # batch of 4
                j = min(i + 4, total_seqs)
                chunk = val_tokens[i * TOTAL_SEQ:(j * TOTAL_SEQ) + 1].to(device=device, dtype=torch.int64)
                x = chunk[:-1].reshape(-1, TOTAL_SEQ)
                y = chunk[1:].reshape(-1, TOTAL_SEQ)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    batch_loss = model(x, y)

                # Count only last-window tokens for BPB
                last_start = (model.n_windows - 1) * model.window_stride
                n_pred = model.window_size * x.shape[0]
                loss_sum += batch_loss.item() * n_pred
                token_count += n_pred

                # Byte count for predicted tokens
                p = x[:, last_start:last_start + model.window_size].reshape(-1)
                t = y[:, last_start:last_start + model.window_size].reshape(-1)
                tb = bl[t].float() + (hl[t] & ~il[p]).float()
                byte_count += tb.sum().item()

        val_loss = loss_sum / token_count
        val_bpb = (val_loss / math.log(2.0)) * (token_count / byte_count)
        print(f"\nVal BPB: {val_bpb:.4f} (on last-{WINDOW_SIZE}-token predictions)")
        print(f"  (Predicts {WINDOW_SIZE}/{TOTAL_SEQ} = {WINDOW_SIZE/TOTAL_SEQ:.1%} of each sequence)")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Hierarchical multi-resolution transformer for Parameter Golf.

Clean architecture:
  1. Input: 4096 tokens at embed_dim=32
  2. 15 non-overlapping context windows (stride=256) → shared convnet encoder → 15 compressed tokens
  3. Bidirectional global transformer over 15 compressed tokens
  4. Prediction window (tokens 3840-4096) NOT in global path
  5. Local decoder: prefix-causal (192 bidir + 64 causal) + cross-attention to global
  6. Loss on last 64 tokens only

Streaming training: slide by 256 per step, cache 14/15 convnet outputs.
Streaming eval: slide by 64, re-encode all 15 windows per slide.

Usage:
  torchrun --standalone --nproc_per_node=1 train_hierarchical.py
  torchrun --standalone --nproc_per_node=8 train_hierarchical.py
"""
import os, sys, math, time, io, glob, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
import numpy as np
import sentencepiece as spm
from pathlib import Path

# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------
def _e(k, d, t=str):
    v = os.environ.get(k, str(d))
    if t == bool: return bool(int(v))
    return t(v)

class Hyperparameters:
    data_path = _e("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = _e("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", f"hier_{int(time.time())}")
    seed = _e("SEED", 1337, int)
    compile_mode = _e("COMPILE_MODE", "default")
    val_batch_size = _e("VAL_BATCH_SIZE", 524288, int)
    val_loss_every = _e("VAL_LOSS_EVERY", 500, int)
    train_log_every = _e("TRAIN_LOG_EVERY", 10, int)
    iterations = _e("ITERATIONS", 2000, int)
    lr_schedule = _e("LR_SCHEDULE", "1cycle")
    onecycle_peak_frac = _e("ONECYCLE_PEAK_FRAC", 0.3, float)
    onecycle_min_div = _e("ONECYCLE_MIN_DIV", 4.0, float)
    train_batch_tokens = _e("TRAIN_BATCH_TOKENS", 524288, int)
    max_wallclock_seconds = _e("MAX_WALLCLOCK_SECONDS", 0.0, float)
    vocab_size = _e("VOCAB_SIZE", 1024, int)
    # Architecture
    total_seq = _e("TOTAL_SEQ", 4096, int)
    window_size = _e("WINDOW_SIZE", 256, int)
    embed_dim = _e("EMBED_DIM", 32, int)
    model_dim = _e("MODEL_DIM", 384, int)
    n_global_layers = _e("N_GLOBAL_LAYERS", 6, int)
    n_local_layers = _e("N_LOCAL_LAYERS", 2, int)
    n_heads = _e("N_HEADS", 6, int)
    mlp_mult = _e("MLP_MULT", 3, int)
    prefix_len = _e("PREFIX_LEN", 192, int)
    pred_len = _e("PRED_LEN", 64, int)
    eval_stride = _e("EVAL_STRIDE", 64, int)
    # Optimizer
    embed_lr = _e("EMBED_LR", 0.6, float)
    head_lr = _e("HEAD_LR", 0.008, float)
    adam_lr = _e("ADAM_LR", 1e-3, float)
    adam_wd = _e("ADAM_WD", 0.05, float)
    matrix_lr = _e("MATRIX_LR", 0.04, float)
    muon_momentum = _e("MUON_MOMENTUM", 0.95, float)
    muon_momentum_warmup_start = _e("MUON_MOMENTUM_WARMUP_START", 0.85, float)
    muon_momentum_warmup_steps = _e("MUON_MOMENTUM_WARMUP_STEPS", 500, int)
    muon_backend_steps = _e("MUON_BACKEND_STEPS", 5, int)
    muon_wd = _e("MUON_WD", 0.0, float)
    beta1 = _e("BETA1", 0.9, float)
    beta2 = _e("BETA2", 0.95, float)
    adam_eps = _e("ADAM_EPS", 1e-8, float)
    grad_clip_norm = _e("GRAD_CLIP_NORM", 1.0, float)
    warmup_steps = _e("WARMUP_STEPS", 10, int)
    # Quantization / serialization
    int6_clip_q = _e("INT6_CLIP_Q", 0.9999984, float)
    save_float = _e("SAVE_FLOAT", 1, bool)
    load_weights = os.environ.get("LOAD_WEIGHTS", "")


# ---------------------------------------------------------------------------
# Int6 quantization
# ---------------------------------------------------------------------------
INT6_RANGE = 31
INT6_KEEP_FLOAT_PATTERNS = ("tok_emb", "lm_head", "embed_proj")
INT8_RANGE = 127

def quantize_int6(t: Tensor, quant_range: int = INT6_RANGE, clip_q: float = 0.9999984) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), clip_q, dim=1).clamp_min(1e-8) if t32.numel() else torch.zeros(t32.shape[0])
        scale = (clip_abs / float(quant_range)).clamp_min(1.0 / float(quant_range))
        clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
        q = torch.clamp(torch.round(clipped / scale[:, None]), -quant_range, quant_range).to(torch.int8)
        return q.contiguous(), scale.half().contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), clip_q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale.float()), -127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()

def q_sd_int6(state_dict: dict, clip_q: float = 0.9999984) -> tuple[dict, dict]:
    result, stats = {}, {"int6_params": 0, "int8_params": 0, "fp_params": 0}
    for name, tensor in state_dict.items():
        # Skip non-weight buffers (e.g. attention masks)
        if "mask" in name:
            result[name] = tensor.detach().cpu()
            continue
        t = tensor.detach().cpu().float().contiguous()
        is_keep_float = any(p in name for p in INT6_KEEP_FLOAT_PATTERNS)
        if t.ndim >= 2 and t.numel() > 4096 and not is_keep_float:
            t2d = t.reshape(t.shape[0], -1) if t.ndim > 2 else t
            q, s = quantize_int6(t2d, INT6_RANGE, clip_q)
            result[name + ".q"], result[name + ".s"], result[name + ".shape"] = q, s, torch.tensor(list(t.shape))
            stats["int6_params"] += t.numel()
        elif t.ndim >= 2 and t.numel() > 4096 and is_keep_float:
            t2d = t.reshape(t.shape[0], -1) if t.ndim > 2 else t
            q, s = quantize_int6(t2d, INT8_RANGE, clip_q)
            result[name + ".q"], result[name + ".s"], result[name + ".shape"] = q, s, torch.tensor(list(t.shape))
            stats["int8_params"] += t.numel()
        else:
            result[name] = t.half()
            stats["fp_params"] += t.numel()
    return result, stats

def deq_sd_int6(obj: dict, target_dtype=torch.bfloat16) -> dict:
    out, processed = {}, set()
    for key in list(obj.keys()):
        if key.endswith(".q"):
            name = key[:-2]
            processed.add(name)
            q, s, shape = obj[name + ".q"].float(), obj[name + ".s"].float(), obj[name + ".shape"].tolist()
            t = (q * s[:, None]).to(target_dtype) if s.ndim > 0 else (q * s).to(target_dtype)
            out[name] = t.reshape(shape).contiguous()
    for key, val in obj.items():
        name = key.removesuffix(".q").removesuffix(".s").removesuffix(".shape")
        if name not in processed and not key.endswith((".q", ".s", ".shape")):
            out[key] = val.to(target_dtype).contiguous()
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def ld_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    """Reads tokens from shards sequentially. Never crosses shard boundaries in a single take().

    take(n) returns EXACTLY n tokens if available within the current shard.
    If fewer than n remain, returns only what's left (short read = shard exhausted).
    Call advance_shard() to move to the next shard, then take() again.
    """
    def __init__(self, files: list[Path]):
        if not files:
            raise FileNotFoundError("No shard files provided")
        self.files = files
        self.file_idx = 0
        self.tokens = ld_shard(self.files[0])
        self.pos = 0

    def advance_shard(self):
        """Move to next shard (wraps around)."""
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = ld_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        """Read up to n tokens from current shard. Returns fewer if shard exhausted."""
        avail = self.tokens.numel() - self.pos
        k = min(n, avail)
        if k <= 0:
            return torch.empty(0, dtype=self.tokens.dtype)
        result = self.tokens[self.pos:self.pos + k]
        self.pos += k
        return result

    @staticmethod
    def from_pattern(pattern: str, rank: int = 0, world_size: int = 1) -> "TokenStream":
        """Create a stream with shards distributed across ranks."""
        all_files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not all_files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        my_files = all_files[rank::world_size]
        if not my_files:
            my_files = all_files  # fallback: all ranks get all files
        return TokenStream(my_files)


# ---------------------------------------------------------------------------
# BPB lookup tables
# ---------------------------------------------------------------------------
def build_luts(sp, vocab_size: int, device: torch.device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def ld_val(pattern, seq_len, max_tok=int(os.environ.get("VAL_MAX_TOKENS", 500000))):
    files = sorted(glob.glob(pattern))
    assert files, f"No files: {pattern}"
    tok = torch.cat([ld_shard(Path(p)) for p in files]).contiguous()
    if max_tok > 0: tok = tok[:max_tok]
    u = (tok.numel() // seq_len) * seq_len
    return tok[:u]


# ---------------------------------------------------------------------------
# Muon optimizer (Newton-Schulz orthogonalized momentum)
# ---------------------------------------------------------------------------
def ns_orth(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, wd: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, wd=wd))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = F.rms_norm(g.float(), (g.size(-1),)).bfloat16()
                    g = ns_orth(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("wd", 0.0)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# ---------------------------------------------------------------------------
# Model building blocks
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

    def forward(self, x, attn_mask=None, is_causal=False):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class CrossAttention(nn.Module):
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

    def forward(self, x, context=None, attn_mask=None, is_causal=False):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)
        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Window encoder: compress (window_size, embed_dim) → (1, model_dim)
# ---------------------------------------------------------------------------
class WindowEncoder(nn.Module):
    def __init__(self, embed_dim, model_dim, window_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(embed_dim, model_dim // 4, kernel_size=8, stride=8),
            nn.SiLU(),
            nn.Conv1d(model_dim // 4, model_dim // 2, kernel_size=4, stride=4),
            nn.SiLU(),
            nn.Conv1d(model_dim // 2, model_dim, kernel_size=window_size // 32, stride=window_size // 32),
            nn.SiLU(),
        )
        self.proj = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        """x: (B, window_size, embed_dim) → (B, 1, model_dim)"""
        h = self.layers(x.transpose(1, 2))  # (B, model_dim, 1)
        h = h.squeeze(-1)  # (B, model_dim)
        return self.proj(h).unsqueeze(1)  # (B, 1, model_dim)


# ---------------------------------------------------------------------------
# Hierarchical GPT
# ---------------------------------------------------------------------------
def build_prefix_causal_mask(window_size: int, prefix_len: int, device="cpu") -> Tensor:
    """Bidirectional prefix + causal suffix attention mask.

    Returns float additive mask (1, 1, T, T) for SDPA compatibility.
    0.0 = attend, -inf = block. 4D shape avoids Flash Attention fallback issues.
    """
    bool_mask = torch.zeros(window_size, window_size, dtype=torch.bool, device=device)
    bool_mask[:prefix_len, :prefix_len] = True
    causal = torch.tril(torch.ones(window_size, window_size, dtype=torch.bool, device=device))
    bool_mask[prefix_len:, :] = causal[prefix_len:, :]
    float_mask = torch.where(bool_mask, 0.0, float('-inf'))
    return float_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)


class HierarchicalGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, model_dim, n_global_layers, n_local_layers,
                 n_heads, mlp_mult, window_size, total_seq, prefix_len, pred_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.window_size = window_size
        self.total_seq = total_seq
        self.prefix_len = prefix_len
        self.pred_len = pred_len
        assert prefix_len + pred_len == window_size, f"{prefix_len}+{pred_len} != {window_size}"
        self.n_context_windows = (total_seq - window_size) // window_size  # non-overlapping
        assert self.n_context_windows * window_size + window_size == total_seq

        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.window_encoder = WindowEncoder(embed_dim, model_dim, window_size)
        self.global_pos = nn.Parameter(torch.randn(1, self.n_context_windows, model_dim) * 0.02)
        self.global_blocks = nn.ModuleList([
            TransformerBlock(model_dim, n_heads, mlp_mult, cross_attn=False)
            for _ in range(n_global_layers)
        ])
        self.local_embed_proj = nn.Linear(embed_dim, model_dim, bias=False)
        self.local_pos = nn.Parameter(torch.randn(1, window_size, model_dim) * 0.02)
        self.local_blocks = nn.ModuleList([
            TransformerBlock(model_dim, n_heads, mlp_mult, cross_attn=True)
            for _ in range(n_local_layers)
        ])
        self.final_norm = RMSNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

        # Prefix-causal mask: registered as buffer for compile compatibility
        self.register_buffer('prefix_causal_mask', build_prefix_causal_mask(window_size, prefix_len))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode_context(self, emb: Tensor) -> Tensor:
        """Encode context windows (tokens 0..n_context_windows*window_size).

        emb: (B, total_seq, embed_dim)
        Returns: global_context (B, n_context_windows, model_dim)
        """
        B = emb.shape[0]
        ctx_len = self.n_context_windows * self.window_size
        context_emb = emb[:, :ctx_len, :]  # (B, 3840, embed_dim)
        windows = context_emb.reshape(B * self.n_context_windows, self.window_size, self.embed_dim)
        compressed = self.window_encoder(windows)  # (B*15, 1, model_dim)
        global_seq = compressed.reshape(B, self.n_context_windows, self.model_dim)
        global_seq = global_seq + self.global_pos
        for block in self.global_blocks:
            global_seq = block(global_seq, is_causal=False)  # BIDIRECTIONAL
        return global_seq

    def encode_single_window(self, window_emb: Tensor) -> Tensor:
        """Encode a single window for streaming cache update.

        window_emb: (B, window_size, embed_dim)
        Returns: (B, 1, model_dim)
        """
        return self.window_encoder(window_emb)

    def run_global_transformer(self, compressed_windows: Tensor) -> Tensor:
        """Run bidirectional global transformer on pre-encoded windows.

        compressed_windows: (B, n_context_windows, model_dim) — from cached + fresh
        Returns: global_context (B, n_context_windows, model_dim)
        """
        global_seq = compressed_windows + self.global_pos
        for block in self.global_blocks:
            global_seq = block(global_seq, is_causal=False)
        return global_seq

    def decode_local(self, pred_emb: Tensor, global_context: Tensor) -> Tensor:
        """Run local decoder on prediction window.

        pred_emb: (B, window_size, embed_dim)
        global_context: (B, n_context_windows, model_dim)
        Returns: logits (B, pred_len, vocab_size)

        Logits come from positions prefix_len-1 to window_size-2 (shifted by 1 from targets).
        Logit at position i predicts the token at position i+1.
        """
        local_seq = self.local_embed_proj(pred_emb) + self.local_pos
        mask = self.prefix_causal_mask
        for block in self.local_blocks:
            local_seq = block(local_seq, context=global_context, attn_mask=mask)
        # Logits at positions [prefix_len-1, window_size-2] predict tokens [prefix_len, window_size-1]
        logits = self.lm_head(self.final_norm(local_seq[:, self.prefix_len - 1:-1, :]))
        return logits

    def forward(self, input_ids: Tensor, reduction: str = "mean") -> Tensor:
        """Full forward pass. Targets are derived from input_ids (no +1 read needed).

        input_ids: (B, total_seq)
        Returns: loss scalar (reduction="mean") or per-token loss (B, pred_len) (reduction="none")
        """
        B = input_ids.shape[0]
        emb = self.tok_emb(input_ids)  # (B, total_seq, embed_dim)

        global_context = self.encode_context(emb)

        pred_start = self.n_context_windows * self.window_size
        pred_emb = emb[:, pred_start:pred_start + self.window_size, :]
        logits = self.decode_local(pred_emb, global_context)  # (B, pred_len, vocab_size)

        # Targets: tokens at positions [prefix_len, window_size) within the prediction window
        pred_targets = input_ids[:, pred_start + self.prefix_len:pred_start + self.window_size]
        if reduction == "none":
            return F.cross_entropy(
                logits.reshape(-1, self.vocab_size), pred_targets.reshape(-1), reduction="none"
            ).reshape(B, self.pred_len)
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), pred_targets.reshape(-1))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def eval_val(args, model, rank, world_size, device, val_tokens,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    """Sliding window eval at stride=eval_stride."""
    total_seq = args.total_seq
    pred_len = args.pred_len
    stride = args.eval_stride
    tgt_start = (total_seq - args.window_size) + args.prefix_len

    total_tokens = val_tokens.numel()
    all_starts = list(range(0, total_tokens - total_seq, stride))
    n = len(all_starts)
    my_starts = all_starts[(n * rank) // world_size:(n * (rank + 1)) // world_size]
    batch_size = max(1, args.val_batch_size // total_seq)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for i in range(0, len(my_starts), batch_size):
            batch_starts = my_starts[i:i + batch_size]
            x = torch.stack([val_tokens[s:s + total_seq] for s in batch_starts])
            x = x.to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                per_token = model(x, reduction="none")  # (batch, pred_len)
            loss_sum += per_token.sum().to(torch.float64)
            token_count += float(per_token.numel())
            tgt_ids = x[:, tgt_start:tgt_start + pred_len].reshape(-1)
            prev_ids = x[:, tgt_start - 1:tgt_start + pred_len - 1].reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(torch.int16)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
            byte_count += tb.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bpb = (val_loss / math.log(2.0)) * (token_count.item() / byte_count.item())
    model.train()
    return val_loss, bpb


# ---------------------------------------------------------------------------
# Streaming training
# ---------------------------------------------------------------------------
class StreamingTrainer:
    """Manages per-rank contiguous token stream + convnet cache for streaming training.

    Data flow per step (after cold start):
      1. Old prediction window becomes newest context window → encode with convnet (1 pass, with grad)
      2. Drop oldest cached context window
      3. Read window_size new tokens from stream → new prediction window
      4. Assemble 15 cached (detached) + 1 fresh encoded → run global transformer
      5. Run local decoder on prediction window → loss on last pred_len tokens
      6. Backward + step

    On shard boundary or first step: cold start (encode all 15 context windows fresh).
    """

    def __init__(self, model: HierarchicalGPT, stream: TokenStream, device: torch.device,
                 total_seq: int, window_size: int):
        self.model = model
        self.stream = stream
        self.device = device
        self.total_seq = total_seq
        self.window_size = window_size
        self.n_ctx = model.n_context_windows
        self.cache: list[Tensor] = []  # list of (1, 1, model_dim) encoded windows
        self.prev_pred_window: Tensor | None = None  # (1, window_size) token IDs

    def _read(self, n: int) -> Tensor | None:
        """Read exactly n contiguous tokens. Returns None if shard exhausted."""
        tokens = self.stream.take(n)
        if tokens.numel() < n:
            self.stream.advance_shard()
            return None
        return tokens.to(self.device, non_blocking=True).to(torch.int64)

    def cold_start(self) -> tuple[Tensor, Tensor, Tensor]:
        """Read exactly total_seq contiguous tokens, encode all context windows.

        Returns: (pred_emb, pred_targets, global_input) — same format as streaming_step,
        so the training loop uses the same code path for both.
        """
        self.cache.clear()
        while True:
            tokens = self._read(self.total_seq)
            if tokens is not None:
                break

        x = tokens.unsqueeze(0)  # (1, total_seq)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            emb = self.model.tok_emb(x)
            ctx_len = self.n_ctx * self.window_size
            for i in range(self.n_ctx):
                start = i * self.window_size
                w_emb = emb[:, start:start + self.window_size, :]
                self.cache.append(self.model.encode_single_window(w_emb))

        self.prev_pred_window = x[:, ctx_len:ctx_len + self.window_size]

        pred_emb = emb[:, ctx_len:ctx_len + self.window_size, :]
        pred_targets = x[:, ctx_len + self.model.prefix_len:ctx_len + self.window_size]
        global_input = torch.cat(self.cache, dim=1)
        return pred_emb, pred_targets, global_input

    def streaming_step(self) -> tuple[Tensor, Tensor, Tensor] | None:
        """Read window_size new tokens for next prediction window.

        Returns None if shard was exhausted (caller should cold-start).
        Otherwise returns: (pred_emb, pred_targets, global_input)
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 1. Encode old prediction window as new context window (WITH gradient)
            prev_emb = self.model.tok_emb(self.prev_pred_window)
            fresh_encoded = self.model.encode_single_window(prev_emb)

        # 2. Update cache
        if len(self.cache) >= self.n_ctx:
            self.cache.pop(0)
        self.cache = [c.detach() for c in self.cache]
        self.cache.append(fresh_encoded)

        # 3. Read exactly window_size tokens (no +1 needed — targets are within the window)
        tokens = self._read(self.window_size)
        if tokens is None:
            self.cache.clear()
            self.prev_pred_window = None
            return None

        pred_x = tokens.unsqueeze(0)  # (1, window_size)
        pred_targets = pred_x[:, self.model.prefix_len:]  # (1, pred_len)

        # 4. Embed and assemble
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_emb = self.model.tok_emb(pred_x)
        global_input = torch.cat(self.cache, dim=1)

        # 5. Save for next step
        self.prev_pred_window = pred_x

        return pred_emb, pred_targets, global_input

    def needs_cold_start(self) -> bool:
        return len(self.cache) == 0 or self.prev_pred_window is None


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------
def main():
    args = Hyperparameters()

    # Compile ns_orth
    global ns_orth
    ns_orth = torch.compile(ns_orth)

    # Distributed setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = max(1, 8 // world_size)
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs("logs/cuda/", exist_ok=True)
    logfile = f"logs/cuda/{args.run_id}.txt" if master_process else None

    def log0(msg: str, console: bool = True):
        if not master_process: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(Path(__file__).read_text(encoding="utf-8"), console=False)
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Data
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = ld_val(args.val_files, args.total_seq)
    bl, hl, il = build_luts(sp, args.vocab_size, device)

    # Model
    base_model = HierarchicalGPT(
        vocab_size=args.vocab_size, embed_dim=args.embed_dim, model_dim=args.model_dim,
        n_global_layers=args.n_global_layers, n_local_layers=args.n_local_layers,
        n_heads=args.n_heads, mlp_mult=args.mlp_mult, window_size=args.window_size,
        total_seq=args.total_seq, prefix_len=args.prefix_len, pred_len=args.pred_len,
    ).to(device).bfloat16()

    # Float32 for linear/conv weights (mixed precision)
    for module in base_model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.float()

    # Load pre-trained weights for resume/fine-tuning
    if args.load_weights:
        ckpt = torch.load(args.load_weights, map_location="cpu", weights_only=False)
        model_sd = base_model.state_dict()
        loaded = {k: v.to(dtype=model_sd[k].dtype) for k, v in ckpt.items() if k in model_sd}
        base_model.load_state_dict(loaded, strict=False)
        log0(f"loaded weights from {args.load_weights} ({len(loaded)}/{len(model_sd)} keys)")

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"params:{n_params:,} ctx_windows:{base_model.n_context_windows} "
         f"d:{args.model_dim} gL:{args.n_global_layers} lL:{args.n_local_layers} "
         f"h:{args.n_heads} ws:{world_size} ga:{grad_accum_steps}")

    # Compile (no DDP — Muon handles its own gradient sync, we manually all-reduce the rest)
    if args.compile_mode == "off":
        model = base_model
    else:
        model = torch.compile(base_model, mode=args.compile_mode if args.compile_mode != "default" else None)

    # Optimizers: Muon for 2D Linear, Adam for embed/head/scalar/conv
    _excl = {"tok_emb.weight", "lm_head.weight"}
    all_other = [(n, p) for n, p in base_model.named_parameters() if not any(e in n for e in _excl)]
    matrix_params = [p for n, p in all_other if p.ndim == 2]
    scalar_params = [p for n, p in all_other if p.ndim != 2]  # includes 1D (norms) and 3D (Conv1d)
    # Collect non-Muon params for manual gradient all-reduce
    adam_params = [base_model.tok_emb.weight, base_model.lm_head.weight] + scalar_params

    opt_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": args.embed_lr, "base_lr": args.embed_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                    backend_steps=args.muon_backend_steps, wd=args.muon_wd)
    for g in opt_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.adam_lr, "base_lr": args.adam_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_head = torch.optim.Adam(
        [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [opt_tok, opt_muon, opt_scalar, opt_head]

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    @torch.no_grad()
    def allreduce_adam_grads():
        """Manually all-reduce gradients for non-Muon params (Muon does its own sync)."""
        if not (dist.is_available() and dist.is_initialized()):
            return
        for p in adam_params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    # LR schedule
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float):
        if args.lr_schedule == "1cycle":
            frac = elapsed_ms / max_wallclock_ms if max_wallclock_ms is not None else step / max(args.iterations, 1)
            frac = min(frac, 1.0)
            min_mul = 1.0 / args.onecycle_min_div
            peak = args.onecycle_peak_frac
            if frac < peak:
                return min_mul + (1.0 - min_mul) * (frac / peak)
            else:
                t = (frac - peak) / (1.0 - peak)
                return min_mul + 0.5 * (1.0 - min_mul) * (1.0 + math.cos(math.pi * t))
        return 1.0

    # Per-rank contiguous token stream + streaming trainer
    stream = TokenStream.from_pattern(args.train_files, rank, world_size)
    streamer = StreamingTrainer(base_model, stream, device, args.total_seq, args.window_size)

    # --- Compiler warmup (uses cold-start full forward) ---
    if args.warmup_steps > 0:
        log0(f"compiler warmup: {args.warmup_steps} steps")
        for _ in range(args.warmup_steps):
            zero_grad_all()
            pred_emb, pred_targets, global_input = streamer.cold_start()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                gc = base_model.run_global_transformer(global_input)
                logits = base_model.decode_local(pred_emb, gc)
                loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), pred_targets.reshape(-1))
            loss.backward()
        zero_grad_all()
        for opt in optimizers:
            opt.state.clear()
        streamer.cache.clear()
        streamer.prev_pred_window = None
        log0("warmup done")

    # --- Training loop ---
    step = 0
    t0 = time.perf_counter()
    train_loss_accum = 0.0
    cold_starts = 0

    while True:
        step += 1
        elapsed_s = time.perf_counter() - t0
        elapsed_ms = elapsed_s * 1000.0

        # Check termination
        last_step = (step > args.iterations)
        if max_wallclock_ms is not None and elapsed_ms >= max_wallclock_ms:
            last_step = True

        # Validation
        if args.val_loss_every > 0 and (step % args.val_loss_every == 1 or last_step):
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, val_tokens, bl, hl, il)
            log0(f"step:{step} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} t:{elapsed_s:.0f}s")

        if last_step:
            break

        # LR schedule
        scale = lr_mul(step, elapsed_ms)

        # Muon momentum warmup
        frac = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
        target_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in opt_muon.param_groups:
            g["momentum"] = target_momentum

        zero_grad_all()

        # Gradient accumulation: multiple streaming steps per optimizer step
        for _micro in range(grad_accum_steps):
            result = None
            if not streamer.needs_cold_start():
                result = streamer.streaming_step()
                if result is None:
                    log0(f"step:{step} shard exhausted — cold start", console=False)
            if result is None:
                result = streamer.cold_start()
                cold_starts += 1

            pred_emb, pred_targets, global_input = result
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                global_context = base_model.run_global_transformer(global_input)
                logits = base_model.decode_local(pred_emb, global_context)
                loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), pred_targets.reshape(-1))

            (loss * grad_scale).backward()
            train_loss_accum += loss.item() * grad_scale

        allreduce_adam_grads()  # Muon handles its own sync in opt.step()

        # Apply LR and step
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale
            opt.step()

        # Logging
        if step % args.train_log_every == 0:
            ms_per_step = elapsed_ms / step
            log0(f"step:{step}/{args.iterations} loss:{train_loss_accum / args.train_log_every:.4f} "
                 f"lr:{scale * args.matrix_lr:.5f} mom:{target_momentum:.3f} "
                 f"t:{elapsed_s:.0f}s {ms_per_step:.0f}ms/step cold:{cold_starts}")
            train_loss_accum = 0.0

    # --- Final eval ---
    log0("final validation...")
    val_loss, val_bpb = eval_val(args, model, rank, world_size, device, val_tokens, bl, hl, il)
    log0(f"final val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    if master_process:
        # Save float weights
        sd = base_model.state_dict()
        if args.save_float:
            float_path = f"{args.run_id}_float.pt"
            torch.save(sd, float_path)
            log0(f"saved float weights: {float_path} ({os.path.getsize(float_path)/1e6:.1f}MB)")

        # Quantize and save artifact
        q_sd, q_stats = q_sd_int6(sd, clip_q=args.int6_clip_q)
        log0(f"quant stats: {q_stats}")

        artifact_buf = io.BytesIO()
        torch.save(q_sd, artifact_buf)
        raw_size = artifact_buf.tell()

        try:
            import zstandard as zstd
            cctx = zstd.ZstdCompressor(level=22)
            compressed = cctx.compress(artifact_buf.getvalue())
        except ImportError:
            import lzma
            compressed = lzma.compress(artifact_buf.getvalue(), preset=9)

        artifact_path = f"{args.run_id}_artifact.ptz"
        with open(artifact_path, "wb") as f:
            f.write(compressed)
        artifact_bytes = os.path.getsize(artifact_path)
        code_bytes = len(Path(__file__).read_text(encoding="utf-8").encode("utf-8"))
        total_bytes = artifact_bytes + code_bytes
        log0(f"artifact: {artifact_bytes/1e6:.2f}MB code:{code_bytes} total:{total_bytes/1e6:.2f}MB "
             f"({'OK' if total_bytes <= 16_000_000 else 'OVER 16MB'})")

        # Roundtrip eval
        log0("roundtrip eval...")
        rt_sd = deq_sd_int6(q_sd, target_dtype=torch.bfloat16)
        base_model.load_state_dict(rt_sd, strict=True)
        for module in base_model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                module.float()
        rt_loss, rt_bpb = eval_val(args, base_model, rank, world_size, device, val_tokens, bl, hl, il)
        log0(f"roundtrip val_loss:{rt_loss:.4f} val_bpb:{rt_bpb:.4f} gap:{rt_bpb - val_bpb:+.4f}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

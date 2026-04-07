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
    lr_schedule = _e("LR_SCHEDULE", "warmdown")
    warmdown_iters = _e("WARMDOWN_ITERS", 1200, int)
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
    batch_size = _e("BATCH_SIZE", 8, int)
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
# Quantization: GPTQ + 6-bit packing
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


def gptq_quantize_matrix(W: Tensor, H: Tensor, quant_range: int = 31,
                          blocksize: int = 128, percdamp: float = 0.01) -> tuple[Tensor, Tensor]:
    """GPTQ: quantize weight matrix W using Hessian H = X^T X from calibration data."""
    dev = H.device
    W = W.float().clone().to(dev)
    rows, cols = W.shape

    damp = percdamp * H.diag().mean()
    diag_idx = torch.arange(cols, device=dev)
    H = H.clone()
    H[diag_idx, diag_idx] += damp

    try:
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    except Exception:
        H_inv = torch.linalg.pinv(H)
    try:
        Hinv_cho = torch.linalg.cholesky(H_inv, upper=True)
    except Exception:
        Hinv_cho = torch.linalg.cholesky(H_inv + 1e-6 * torch.eye(cols, device=dev), upper=True)

    scale = W.abs().amax(dim=1).clamp_min(1e-8) / quant_range
    Q = torch.zeros_like(W, dtype=torch.int8)

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W_block = W[:, i1:i2].clone()
        Hinv_block = Hinv_cho[i1:i2, i1:i2]

        for i in range(i2 - i1):
            w = W_block[:, i]
            d = Hinv_block[i, i]
            q = (w / scale).round().clamp(-quant_range, quant_range)
            Q[:, i1 + i] = q.to(torch.int8)
            err = (w - q * scale) / d
            W_block[:, i:] -= err.unsqueeze(1) * Hinv_block[i, i:].unsqueeze(0)

        if i2 < cols:
            W[:, i2:] -= (W[:, i1:i2] - Q[:, i1:i2].float() * scale.unsqueeze(1)) @ Hinv_cho[i1:i2, i2:]

    return Q.cpu(), scale.cpu().half()


class HessianCollector:
    """Collects H = X^T X for a linear layer's inputs during forward passes."""
    def __init__(self):
        self.H = None
        self.n_samples = 0
        self.hook = None

    def hook_fn(self, module, input, output):
        x = input[0]
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        x = x.float()
        if self.H is None:
            self.H = torch.zeros(x.shape[1], x.shape[1], device=x.device)
        self.H.addmm_(x.T, x)
        self.n_samples += x.shape[0]

    def register(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.hook:
            self.hook.remove()

    def get_H(self):
        return self.H / max(self.n_samples, 1) if self.H is not None else None


def collect_hessians(model, val_tokens, device, total_seq, n_calib_tokens=131072):
    """Collect Hessians for all quantizable linear layers via calibration forward passes."""
    collectors = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.numel() > 4096:
            if not any(p in name for p in INT6_KEEP_FLOAT_PATTERNS):
                c = HessianCollector()
                c.register(module)
                collectors[name] = c

    model.eval()
    n_seqs = min(n_calib_tokens // total_seq, (val_tokens.numel() - 1) // total_seq)
    with torch.inference_mode():
        for i in range(n_seqs):
            x = val_tokens[i * total_seq:(i + 1) * total_seq].unsqueeze(0).to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model(x)
    model.train()

    hessians = {}
    for name, c in collectors.items():
        H = c.get_H()
        if H is not None:
            hessians[name] = H
        c.remove()
    return hessians


def pack_int6(q: Tensor) -> Tensor:
    """Pack int6 values (stored as int8, range [-31,31]) into 6 bits per value.
    Packs 4 values into 3 bytes. Input length must be divisible by 4.
    Values shifted to unsigned [0,62] before packing."""
    flat = (q.flatten().to(torch.int16) + 31).to(torch.uint8)  # [0, 62], fits in 6 bits
    n = flat.numel()
    assert n % 4 == 0, f"pack_int6 requires length divisible by 4, got {n}"
    flat = flat.reshape(-1, 4)
    a, b, c, d = flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]
    # Pack 4×6bit = 24bit = 3 bytes: [aaaaaa|bb bbbb|cccc cc|dddddd] → byte0=a<<2|b>>4, byte1=(b&0xF)<<4|c>>2, byte2=(c&3)<<6|d
    b0 = (a << 2) | (b >> 4)
    b1 = ((b & 0x0F) << 4) | (c >> 2)
    b2 = ((c & 0x03) << 6) | d
    packed = torch.stack([b0, b1, b2], dim=1).reshape(-1)
    return packed.contiguous()


def unpack_int6(packed: Tensor, n_values: int) -> Tensor:
    """Unpack 6-bit packed values back to int8 in [-31,31]."""
    packed = packed.reshape(-1, 3)
    b0, b1, b2 = packed[:, 0].to(torch.int16), packed[:, 1].to(torch.int16), packed[:, 2].to(torch.int16)
    a = (b0 >> 2) & 0x3F
    b = ((b0 & 0x03) << 4) | ((b1 >> 4) & 0x0F)
    c = ((b1 & 0x0F) << 2) | ((b2 >> 6) & 0x03)
    d = b2 & 0x3F
    flat = torch.stack([a, b, c, d], dim=1).reshape(-1)[:n_values]
    return (flat.to(torch.int8) - 31).contiguous()


def q_sd_gptq(state_dict: dict, hessians: dict, clip_q: float = 0.9999984) -> tuple[dict, dict]:
    """Quantize state dict with GPTQ (where Hessians available) + 6-bit packing."""
    result, stats = {}, {"gptq_params": 0, "naive_params": 0, "int8_params": 0, "fp_params": 0}
    for name, tensor in state_dict.items():
        if "mask" in name:
            result[name] = tensor.detach().cpu()
            continue
        t = tensor.detach().cpu().float().contiguous()
        is_keep_float = any(p in name for p in INT6_KEEP_FLOAT_PATTERNS)

        if t.ndim >= 2 and t.numel() > 4096 and not is_keep_float:
            t2d = t.reshape(t.shape[0], -1) if t.ndim > 2 else t
            module_name = name.replace(".weight", "")
            H = hessians.get(module_name)

            if H is not None and H.shape[0] == t2d.shape[1]:
                q, s = gptq_quantize_matrix(t2d, H, quant_range=INT6_RANGE)
                stats["gptq_params"] += t.numel()
            else:
                q, s = quantize_int6(t2d, INT6_RANGE, clip_q)
                stats["naive_params"] += t.numel()

            # 6-bit pack: pad total values to multiple of 4
            rows, cols = q.shape
            pad = (4 - cols % 4) % 4
            if pad > 0:
                q = torch.cat([q, torch.zeros(rows, pad, dtype=torch.int8)], dim=1)
            packed = pack_int6(q)
            result[name + ".p6"] = packed
            result[name + ".s"] = s
            result[name + ".shape"] = torch.tensor(list(t.shape))  # original shape (may be 3D+)

        elif t.ndim >= 2 and t.numel() > 4096 and is_keep_float:
            t2d = t.reshape(t.shape[0], -1) if t.ndim > 2 else t
            q, s = quantize_int6(t2d, INT8_RANGE, clip_q)
            result[name + ".q"], result[name + ".s"], result[name + ".shape"] = q, s, torch.tensor(list(t.shape))
            stats["int8_params"] += t.numel()
        else:
            result[name] = t.half()
            stats["fp_params"] += t.numel()
    return result, stats


def deq_sd(obj: dict, target_dtype=torch.bfloat16) -> dict:
    """Dequantize: handles both packed int6 (.p6) and unpacked int8 (.q) formats."""
    out, processed = {}, set()
    for key in list(obj.keys()):
        if key.endswith(".p6"):
            name = key[:-3]
            processed.add(name)
            orig_shape = obj[name + ".shape"].tolist()
            s = obj[name + ".s"].float()
            # Flatten to 2D for dequant (rows × cols), same as quantization
            rows = orig_shape[0]
            cols = 1
            for d in orig_shape[1:]:
                cols *= d
            pad = (4 - cols % 4) % 4
            total_values = rows * (cols + pad)
            q = unpack_int6(obj[name + ".p6"], total_values).reshape(rows, cols + pad)[:, :cols].float()
            out[name] = (q * s[:, None]).to(target_dtype).reshape(orig_shape).contiguous()
        elif key.endswith(".q"):
            name = key[:-2]
            if name not in processed:
                processed.add(name)
                q, s = obj[name + ".q"].float(), obj[name + ".s"].float()
                shape = obj[name + ".shape"].tolist()
                t = (q * s[:, None]).to(target_dtype) if s.ndim > 0 else (q * s).to(target_dtype)
                out[name] = t.reshape(shape).contiguous()
    for key, val in obj.items():
        base = key.removesuffix(".p6").removesuffix(".q").removesuffix(".s").removesuffix(".shape")
        if base not in processed and not key.endswith((".p6", ".q", ".s", ".shape")):
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
class BatchedStreamingTrainer:
    """Batched streaming trainer: B independent token streams with tensor-based caches.

    Optimizations vs naive per-stream loop:
    - Token reads: all on CPU, single bulk .to(device) transfer
    - Cache: pre-allocated (B, n_ctx, model_dim) tensor, shift via roll + overwrite
    - GPU ops: single batched call for tok_emb and encode_single_window
    """

    def __init__(self, model: HierarchicalGPT, streams: list[TokenStream],
                 device: torch.device, total_seq: int, window_size: int):
        self.model = model
        self.B = len(streams)
        self.streams = streams
        self.device = device
        self.total_seq = total_seq
        self.window_size = window_size
        self.n_ctx = model.n_context_windows
        self.model_dim = model.model_dim
        self.embed_dim = model.embed_dim
        self.prefix_len = model.prefix_len
        # Cache as contiguous tensor: (B, n_ctx, model_dim) on device, bfloat16 to match autocast
        self.cache = torch.zeros(self.B, self.n_ctx, self.model_dim, device=device, dtype=torch.bfloat16)
        self.cache_len = torch.zeros(self.B, dtype=torch.int32)  # how many valid entries per stream
        # Previous prediction window tokens: (B, window_size) on device
        self.prev_pred = torch.zeros(self.B, self.window_size, dtype=torch.int64, device=device)
        self.has_prev = torch.zeros(self.B, dtype=torch.bool)  # which streams have valid prev

    def _read_cpu(self, stream_idx: int, n: int) -> Tensor | None:
        """Read n tokens, stay on CPU. Returns None if shard exhausted."""
        tokens = self.streams[stream_idx].take(n)
        if tokens.numel() < n:
            self.streams[stream_idx].advance_shard()
            return None
        return tokens.to(torch.int64)

    def step(self) -> tuple[Tensor, Tensor, Tensor, int]:
        """Advance all B streams. CPU reads batched, single GPU transfer, batched GPU ops."""
        ws = self.window_size

        # 1. Classify + read tokens on CPU (no GPU transfers yet)
        cold_ids = []
        stream_ids = []
        stream_tokens_list = []  # list of (ws,) CPU tensors
        cold_tokens_list = []    # list of (total_seq,) CPU tensors

        for b in range(self.B):
            if self.has_prev[b] and self.cache_len[b] >= self.n_ctx:
                tokens = self._read_cpu(b, ws)
                if tokens is not None:
                    stream_ids.append(b)
                    stream_tokens_list.append(tokens)
                else:
                    self.has_prev[b] = False
                    self.cache_len[b] = 0
                    cold_ids.append(b)
            else:
                cold_ids.append(b)

        for b in cold_ids:
            self.cache_len[b] = 0
            while True:
                tokens = self._read_cpu(b, self.total_seq)
                if tokens is not None:
                    break
            cold_tokens_list.append(tokens)

        n_stream = len(stream_ids)
        n_cold = len(cold_ids)

        # 2. Single bulk CPU→GPU transfers (stack on CPU first, one transfer each)
        if n_stream > 0:
            s_idx = torch.tensor(stream_ids, dtype=torch.long)
            new_pred_gpu = torch.stack(stream_tokens_list).to(self.device, non_blocking=True)
        if n_cold > 0:
            c_idx = torch.tensor(cold_ids, dtype=torch.long)
            cold_gpu = torch.stack(cold_tokens_list).to(self.device, non_blocking=True)

        # 3. Batched GPU ops for streaming streams
        if n_stream > 0:
            prev_batch = self.prev_pred[s_idx]  # (S, ws) — already on GPU

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prev_emb = self.model.tok_emb(prev_batch)
                fresh_enc = self.model.encode_single_window(prev_emb)  # (S, 1, model_dim)
                new_emb = self.model.tok_emb(new_pred_gpu)              # (S, ws, embed_dim)

            # Update cache: vectorized shift + write (no Python loop over B)
            self.cache[s_idx, :-1] = self.cache[s_idx, 1:].clone()
            self.cache[s_idx, -1] = fresh_enc[:, 0].detach()
            self.prev_pred[s_idx] = new_pred_gpu

        # 4. Batched GPU ops for cold starts
        if n_cold > 0:
            ctx_len = self.n_ctx * ws

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                cold_emb = self.model.tok_emb(cold_gpu)  # (C, total_seq, embed_dim)
                ctx_emb = cold_emb[:, :ctx_len, :].reshape(-1, ws, self.embed_dim)
                all_enc = self.model.encode_single_window(ctx_emb)  # (C*n_ctx, 1, model_dim)
                all_enc = all_enc.reshape(n_cold, self.n_ctx, self.model_dim)

            self.cache[c_idx] = all_enc.detach()
            self.cache_len[c_idx] = self.n_ctx
            self.has_prev[c_idx] = True
            self.prev_pred[c_idx] = cold_gpu[:, ctx_len:ctx_len + ws]

        # 5. Assemble final batched tensors (all B streams, original order)
        # Global input from cache
        global_in = self.cache.clone()  # (B, n_ctx, model_dim) — detached cache

        # Prediction embeddings: need tok_emb on all B prev_pred windows
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # For streaming: we already computed new_emb; for cold: reuse cold_emb slice
            # Simpler: just embed all B prev_pred windows (cheap, avoids complex indexing)
            all_pred_emb = self.model.tok_emb(self.prev_pred)  # (B, ws, embed_dim)

        pred_targets = self.prev_pred[:, self.prefix_len:]  # (B, pred_len)

        return all_pred_emb, pred_targets, global_in, n_cold


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------
def main():
    args = Hyperparameters()

    # Compile ns_orth
    global ns_orth
    if args.compile_mode != "off":
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
         f"h:{args.n_heads} bs:{args.batch_size} ws:{world_size} ga:{grad_accum_steps}")

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
        if args.lr_schedule == "warmdown":
            if args.warmdown_iters <= 0:
                return 1.0
            if max_wallclock_ms is None:
                warmdown_start = max(args.iterations - args.warmdown_iters, 0)
                if warmdown_start <= step < args.iterations:
                    return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                return 1.0
            step_ms = elapsed_ms / max(step, 1)
            warmdown_ms = args.warmdown_iters * step_ms
            remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
            return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
        elif args.lr_schedule == "1cycle":
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

    # Per-rank batched streaming trainer: B independent token streams
    all_files = sorted(glob.glob(args.train_files))
    my_files = all_files[rank::world_size] or all_files
    streams = []
    for b in range(args.batch_size):
        # Rotate starting file so each stream in the batch reads different data
        rotated = my_files[(b * len(my_files) // args.batch_size) % len(my_files):] + \
                  my_files[:(b * len(my_files) // args.batch_size) % len(my_files)]
        streams.append(TokenStream(rotated))
    streamer = BatchedStreamingTrainer(base_model, streams, device, args.total_seq, args.window_size)

    # --- Compiler warmup (uses cold-start full forward) ---
    if args.warmup_steps > 0:
        log0(f"compiler warmup: {args.warmup_steps} steps")
        for _ in range(args.warmup_steps):
            zero_grad_all()
            pred_emb, pred_targets, global_input, _ = streamer.step()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                gc = base_model.run_global_transformer(global_input)
                logits = base_model.decode_local(pred_emb, gc)
                loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), pred_targets.reshape(-1))
            loss.backward()
        zero_grad_all()
        for opt in optimizers:
            opt.state.clear()
        streamer.cache.zero_()
        streamer.cache_len.zero_()
        streamer.has_prev.zero_()
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

        # Gradient accumulation: grad_accum_steps × batch_size streams per optimizer step
        for _micro in range(grad_accum_steps):
            pred_emb, pred_targets, global_input, cold = streamer.step()
            cold_starts += cold

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

        # GPTQ calibration + quantize with 6-bit packing
        log0("collecting Hessians for GPTQ...")
        hessians = collect_hessians(base_model, val_tokens, device, args.total_seq)
        log0(f"  collected {len(hessians)} Hessians")
        q_sd, q_stats = q_sd_gptq(sd, hessians, clip_q=args.int6_clip_q)
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
        rt_sd = deq_sd(q_sd, target_dtype=torch.bfloat16)
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

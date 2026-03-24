#!/usr/bin/env python3
"""
Ternary neural network training for Parameter Golf.

Architecture: Gated recurrence (MLGRU-style) + Gated MLP with ternary {-1,0,1} weights.
Training: Shadow weights + Straight-Through Estimator (STE), Adam optimizer.
Serialization: 2-bit packed ternary weights + zlib.

Key env vars:
  USE_TERNARY=1  Use ternary weights (default 1). Set to 0 for full-precision comparison.
  MODEL_DIM=768  Model width (default 768).
  NUM_LAYERS=9   Number of blocks (default 9).
  ITERATIONS=200 Training steps.
  LR=0.003       Adam learning rate.
"""
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import sys
import time
import uuid
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

COMPUTE_DTYPE = mx.bfloat16

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================

class Hyperparameters:
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_max_tokens: int = int(os.environ.get("VAL_MAX_TOKENS", 0))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 512))
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    mlx_eager_eval: bool = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 10))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model.
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 9))
    model_dim: int = int(os.environ.get("MODEL_DIM", 768))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    use_ternary: bool = bool(int(os.environ.get("USE_TERNARY", "1")))
    use_gru: bool = bool(int(os.environ.get("USE_GRU", "1")))
    use_metaplastic: bool = bool(int(os.environ.get("USE_METAPLASTIC", "0")))
    use_hebbian: bool = bool(int(os.environ.get("USE_HEBBIAN", "0")))
    hebbian_lr: float = float(os.environ.get("HEBBIAN_LR", 0.01))
    hebbian_ema_decay: float = float(os.environ.get("HEBBIAN_EMA_DECAY", 0.99))
    # Think/Know architecture: shared "thinking" layers + unique "knowledge" layers.
    # THINK_DEPTH=0 disables this (all layers unique, original behavior).
    think_depth: int = int(os.environ.get("THINK_DEPTH", 0))
    num_know_layers: int = int(os.environ.get("NUM_KNOW_LAYERS", 6))
    num_cycles: int = int(os.environ.get("NUM_CYCLES", 1))

    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Optimizer (Adam for all parameters).
    lr: float = float(os.environ.get("LR", 0.003))
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


class TokenStream:
    def __init__(self, pattern: str, log_fn: Callable[[str], None] | None = None, dataset_name: str = ""):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn is not None:
                self.log_fn(f"WARNING: starting epoch:{self.epoch} dataset:{self.dataset_name} train_shards:{len(self.files)}")
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class TokenLoader:
    def __init__(self, pattern: str, log_fn: Callable[[str], None] | None = None, dataset_name: str = ""):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


# ==============================================================================
# MATH HELPERS
# ==============================================================================

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


# ==============================================================================
# TERNARY CORE
# ==============================================================================

def ternarize(w: mx.array, eps: float = 1e-8) -> tuple[mx.array, mx.array]:
    """Absmean quantization from BitNet b1.58 with STE."""
    scale = mx.mean(mx.abs(w)) + eps
    w_q = mx.clip(mx.round(w / scale), -1, 1)
    # STE: forward uses quantized weights, backward passes gradient to shadow weights.
    w_ste = w + mx.stop_gradient(w_q * scale - w)
    return w_ste, scale


class TernaryLinear(nn.Module):
    """Linear layer with ternary {-1,0,1} weights via STE during training."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        std = math.sqrt(2.0 / (in_dim + out_dim))
        self.weight = mx.random.normal((out_dim, in_dim)) * std

    def __call__(self, x: mx.array) -> mx.array:
        w, _ = ternarize(self.weight)
        return x @ w.astype(x.dtype).T


class CastedLinear(nn.Module):
    """Standard full-precision linear (for USE_TERNARY=0 comparison)."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class MetaplasticTernaryLinear(nn.Module):
    """Ternary linear with metaplastic integer counters (Approach B).

    Each weight has an integer counter. Sign = weight value, magnitude = confidence.
    Gradients nudge counters probabilistically instead of updating shadow weights.
    We still use STE for the forward pass but the counters replace Adam's role.
    """
    def __init__(self, in_dim: int, out_dim: int, counter_bits: int = 8):
        super().__init__()
        self.max_val = 2 ** (counter_bits - 1) - 1
        # Initialize counters with small random integers.
        self.weight = (mx.random.normal((out_dim, in_dim)) * 3.0).astype(mx.float32)
        # A fixed scale learned per-layer; initialized to a reasonable value.
        self.meta_scale = mx.array(0.02, dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        # Ternary weight = sign of counter (0 when counter is 0).
        w_q = mx.sign(self.weight)
        scale = self.meta_scale
        w = w_q * scale
        # STE: forward uses ternary, backward flows to counters.
        w_ste = self.weight * scale / (mx.mean(mx.abs(self.weight)) + 1e-8)
        w_ste = w_ste + mx.stop_gradient(w - w_ste)
        return x @ w_ste.astype(x.dtype).T


def make_linear(in_dim: int, out_dim: int, use_ternary: bool, use_metaplastic: bool = False) -> nn.Module:
    if use_metaplastic:
        return MetaplasticTernaryLinear(in_dim, out_dim)
    return TernaryLinear(in_dim, out_dim) if use_ternary else CastedLinear(in_dim, out_dim)


# ==============================================================================
# CAUSAL SELF-ATTENTION (FOR USE_GRU=0)
# ==============================================================================

def apply_rotary_emb(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], axis=-1)


class CausalSelfAttention(nn.Module):
    """Multi-head attention with RoPE and GQA, using optional ternary projections."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float,
                 qk_gain_init: float, use_ternary: bool, use_metaplastic: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = make_linear(dim, dim, use_ternary, use_metaplastic)
        self.c_k = make_linear(dim, kv_dim, use_ternary, use_metaplastic)
        self.c_v = make_linear(dim, kv_dim, use_ternary, use_metaplastic)
        self.proj = make_linear(dim, dim, use_ternary, use_metaplastic)
        self.proj.weight = mx.zeros_like(self.proj.weight)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y)


class ReluSquaredMLP(nn.Module):
    """ReLU² MLP from baseline (for USE_GRU=0 mode)."""
    def __init__(self, dim: int, mlp_mult: int, use_ternary: bool, use_metaplastic: bool = False):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = make_linear(dim, hidden, use_ternary, use_metaplastic)
        self.proj = make_linear(hidden, dim, use_ternary, use_metaplastic)
        self.proj.weight = mx.zeros_like(self.proj.weight)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc(x))
        return self.proj(x * x)


# ==============================================================================
# PARALLEL LINEAR SCAN
# ==============================================================================

def parallel_scan(a: mx.array, b: mx.array) -> mx.array:
    """Compute h[t] = a[t]*h[t-1] + b[t] for all t, with h[-1] = 0.

    Uses iterative doubling in O(log T) parallel steps instead of O(T) sequential.
    The operator (a1, b1) . (a2, b2) = (a1*a2, a2*b1 + b2) is associative,
    so we can combine pairs at exponentially growing strides.

    a: (B, T, D)  multiplicative coefficients (forget gates)
    b: (B, T, D)  additive coefficients ((1-f)*c)
    Returns: (B, T, D) all hidden states
    """
    # Work in float32 to avoid underflow from cumulative products of forget gates.
    orig_dtype = a.dtype
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    T = a.shape[1]
    num_rounds = int(math.ceil(math.log2(max(T, 1))))

    for k in range(num_rounds):
        stride = 2 ** k
        # Shifted versions: positions [0..stride-1] have no predecessor at this stride,
        # so use identity element (a=1, b=0).
        a_prev = mx.concatenate([mx.ones_like(a[:, :stride, :]), a[:, :-stride, :]], axis=1)
        b_prev = mx.concatenate([mx.zeros_like(b[:, :stride, :]), b[:, :-stride, :]], axis=1)
        # Associative combine: (a, b) . (a_prev, b_prev) = (a*a_prev, a*b_prev + b)
        b = a * b_prev + b
        a = a * a_prev

    return b.astype(orig_dtype)


# ==============================================================================
# GATED RECURRENCE (TOKEN MIXER)
# ==============================================================================

class GatedRecurrence(nn.Module):
    """MLGRU-style gated recurrence replacing attention.

    Computes gates in parallel, then runs a parallel scan for the recurrence.
    No softmax, no attention scores — just element-wise gating.
    """
    def __init__(self, dim: int, use_ternary: bool = True, use_metaplastic: bool = False):
        super().__init__()
        self.dim = dim
        # Fused projection for forget gate, candidate, and output gate.
        self.gates_proj = make_linear(dim, 3 * dim, use_ternary, use_metaplastic)
        self.out_proj = make_linear(dim, dim, use_ternary, use_metaplastic)
        # Zero-init output projection so initial blocks are near-identity.
        self.out_proj.weight = mx.zeros_like(self.out_proj.weight)
        # Forget gate bias: sigmoid(2.0) ≈ 0.88, so initial state mostly remembers.
        self.forget_bias = mx.ones((dim,)) * 2.0

    def __call__(self, x: mx.array) -> mx.array:
        B, T, D = x.shape

        # Compute all gate pre-activations in parallel (one big matmul).
        gates = self.gates_proj(x)  # (B, T, 3D)
        f_pre = gates[:, :, :D] + self.forget_bias
        c_pre = gates[:, :, D : 2 * D]
        o_pre = gates[:, :, 2 * D :]

        f = mx.sigmoid(f_pre)                       # forget gate
        c = c_pre * mx.sigmoid(c_pre)               # candidate (SiLU)
        o = mx.sigmoid(o_pre)                        # output gate

        # Parallel scan: O(log T) steps instead of O(T).
        h_all = parallel_scan(f, (1.0 - f) * c)     # (B, T, D)

        return self.out_proj(h_all * o)


# ==============================================================================
# GATED MLP (CHANNEL MIXER)
# ==============================================================================

class GatedMLP(nn.Module):
    """Sigmoid-gated MLP. Fuses gate+value into one projection for fewer kernel launches."""
    def __init__(self, dim: int, mlp_mult: int, use_ternary: bool = True, use_metaplastic: bool = False):
        super().__init__()
        self.hidden = dim * mlp_mult
        # Fused gate+value: one matmul (dim → 2*hidden) instead of two (dim → hidden).
        self.gate_value_proj = make_linear(dim, 2 * self.hidden, use_ternary, use_metaplastic)
        self.out_proj = make_linear(self.hidden, dim, use_ternary, use_metaplastic)
        self.out_proj.weight = mx.zeros_like(self.out_proj.weight)

    def __call__(self, x: mx.array) -> mx.array:
        gv = self.gate_value_proj(x)
        gate = mx.sigmoid(gv[:, :, :self.hidden])
        value = gv[:, :, self.hidden:]
        return self.out_proj(gate * value)


# ==============================================================================
# BLOCK AND MODEL
# ==============================================================================

class RMSNormNoWeight(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class GRUBlock(nn.Module):
    """Pre-norm → GRU → SubLN → residual, pre-norm → GatedMLP → SubLN → residual."""
    def __init__(self, dim: int, mlp_mult: int, use_ternary: bool = True, use_metaplastic: bool = False):
        super().__init__()
        self.pre_token_norm = RMSNormNoWeight()
        self.token_mixer = GatedRecurrence(dim, use_ternary, use_metaplastic)
        self.post_token_norm = RMSNormNoWeight()

        self.pre_mlp_norm = RMSNormNoWeight()
        self.channel_mixer = GatedMLP(dim, mlp_mult, use_ternary, use_metaplastic)
        self.post_mlp_norm = RMSNormNoWeight()

        self.token_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.token_mixer(self.pre_token_norm(x))
        h = self.post_token_norm(h)
        x = x + self.token_scale.astype(x.dtype)[None, None, :] * h
        m = self.channel_mixer(self.pre_mlp_norm(x))
        m = self.post_mlp_norm(m)
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * m
        return x


class AttnBlock(nn.Module):
    """Pre-norm → Attention → SubLN → residual, pre-norm → ReLU²MLP → SubLN → residual."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, use_ternary: bool, use_metaplastic: bool = False):
        super().__init__()
        self.pre_token_norm = RMSNormNoWeight()
        self.token_mixer = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, use_ternary, use_metaplastic)
        self.post_token_norm = RMSNormNoWeight()

        self.pre_mlp_norm = RMSNormNoWeight()
        self.channel_mixer = ReluSquaredMLP(dim, mlp_mult, use_ternary, use_metaplastic)
        self.post_mlp_norm = RMSNormNoWeight()

        self.token_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.token_mixer(self.pre_token_norm(x))
        h = self.post_token_norm(h)
        x = x + self.token_scale.astype(x.dtype)[None, None, :] * h
        m = self.channel_mixer(self.pre_mlp_norm(x))
        m = self.post_mlp_norm(m)
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * m
        return x


class TernaryModel(nn.Module):
    """Language model with optional Think/Know weight sharing.

    When think_depth=0: all layers are unique (original behavior).
    When think_depth>0: shared "thinking" layers alternate with unique "knowledge" layers.
    The execution sequence per cycle is: [T1..Td, K1, T1..Td, K2, ..., T1..Td, Kn]
    With num_cycles>1, the knowledge layers are revisited.
    """
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        logit_softcap: float,
        logit_chunk_tokens: int,
        tied_embed_init_std: float,
        use_ternary: bool,
        use_gru: bool,
        use_metaplastic: bool = False,
        rope_base: float = 10000.0,
        qk_gain_init: float = 1.5,
        think_depth: int = 0,
        num_know_layers: int = 6,
        num_cycles: int = 1,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_softcap = logit_softcap
        self.logit_chunk_tokens = logit_chunk_tokens
        self.think_depth = think_depth
        self.num_know_layers = num_know_layers
        self.num_cycles = num_cycles

        def make_block():
            if use_gru:
                return GRUBlock(dim, mlp_mult, use_ternary, use_metaplastic)
            return AttnBlock(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, use_ternary, use_metaplastic)

        self.tok_emb = nn.Embedding(vocab_size, dim)

        if think_depth > 0:
            # Think/Know architecture: shared think blocks + unique know blocks.
            self.think_blocks = [make_block() for _ in range(think_depth)]
            self.know_blocks = [make_block() for _ in range(num_know_layers)]
            # Build execution sequence as (source_list, index) pairs.
            self._sequence: list[tuple[str, int]] = []
            for _ in range(num_cycles):
                for k in range(num_know_layers):
                    for t in range(think_depth):
                        self._sequence.append(("think", t))
                    self._sequence.append(("know", k))
            self.blocks = []  # unused, but keeps parameter discovery simple
        else:
            # Original: all unique layers.
            self.blocks = [make_block() for _ in range(num_layers)]
            self.think_blocks = []
            self.know_blocks = []
            self._sequence = [("block", i) for i in range(num_layers)]

        self.final_norm = RMSNormNoWeight()

        # Initialize embedding (small std, fp16 — same as baseline).
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def effective_depth(self) -> int:
        return len(self._sequence)

    def unique_block_count(self) -> int:
        return len(self.blocks) + len(self.think_blocks) + len(self.know_blocks)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        for source, idx in self._sequence:
            if source == "think":
                x = self.think_blocks[idx](x)
            elif source == "know":
                x = self.know_blocks[idx](x)
            else:
                x = self.blocks[idx](x)
        return self.final_norm(x)

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits = self.softcap(x @ self.tok_emb.weight.astype(x.dtype).T)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")
        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits = self.softcap(x[s:e] @ self.tok_emb.weight.astype(x.dtype).T)
            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
        return loss_sum / float(n)


# ==============================================================================
# TERNARY SERIALIZATION (2-BIT PACKING)
# ==============================================================================

def pack_ternary_2bit(values: np.ndarray) -> np.ndarray:
    """Pack ternary {-1,0,1} into 2 bits per value. 4 values per byte."""
    flat = (values.flatten().astype(np.int8) + 1).astype(np.uint8)  # map to {0,1,2}
    pad_len = (4 - len(flat) % 4) % 4
    if pad_len:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.uint8)])
    flat = flat.reshape(-1, 4)
    packed = flat[:, 0] | (flat[:, 1] << 2) | (flat[:, 2] << 4) | (flat[:, 3] << 6)
    return packed.astype(np.uint8)


def unpack_ternary_2bit(packed: np.ndarray, numel: int) -> np.ndarray:
    """Unpack 2-bit ternary to {-1,0,1}."""
    vals = np.zeros(len(packed) * 4, dtype=np.int8)
    vals[0::4] = (packed & 0x03).astype(np.int8)
    vals[1::4] = ((packed >> 2) & 0x03).astype(np.int8)
    vals[2::4] = ((packed >> 4) & 0x03).astype(np.int8)
    vals[3::4] = ((packed >> 6) & 0x03).astype(np.int8)
    return vals[:numel] - 1  # map back from {0,1,2} to {-1,0,1}


def serialize_ternary_model(flat_state: dict[str, mx.array]) -> tuple[dict, dict[str, int]]:
    """Serialize: large 2D tensors → 2-bit ternary, small tensors → fp16."""
    ternary_packed: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    shapes: dict[str, tuple] = {}
    passthrough: dict[str, np.ndarray] = {}
    stats = {"ternary_params": 0, "other_params": 0, "packed_bytes": 0, "passthrough_bytes": 0}

    for name, arr in flat_state.items():
        # MLX bfloat16 doesn't round-trip through np.array(copy=True) directly.
        f32 = np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)

        # Large 2D tensors → ternary 2-bit packing.
        if arr.ndim == 2 and arr.size > 4096:
            scale = float(np.mean(np.abs(f32))) + 1e-8
            w_q = np.clip(np.round(f32 / scale), -1, 1).astype(np.int8)
            packed = pack_ternary_2bit(w_q)
            ternary_packed[name] = packed
            scales[name] = np.float16(scale)
            shapes[name] = tuple(int(s) for s in arr.shape)
            stats["ternary_params"] += int(arr.size)
            stats["packed_bytes"] += len(packed)
        else:
            # Small tensors (biases, scales, embeddings) → fp16.
            stored = f32.astype(np.float16)
            passthrough[name] = stored
            stats["other_params"] += int(arr.size)
            stats["passthrough_bytes"] += stored.nbytes

    stats["packed_bytes"] += sum(s.nbytes for s in scales.values())
    obj = {"format": "ternary_2bit_v1", "ternary": ternary_packed, "scales": scales,
           "shapes": shapes, "passthrough": passthrough}
    return obj, stats


def deserialize_ternary_model(obj: dict) -> dict[str, mx.array]:
    """Deserialize 2-bit ternary model back to mx.arrays.

    The shadow weights must be reconstructed so that ternarize() recovers the
    exact same (w_q, scale) pair.  Naively setting shadow = w_q * scale gives
    the wrong scale on the next ternarize() call because
    mean(|w_q * scale|) = scale * mean(|w_q|) != scale  (zeros pull the mean down).

    Fix: set shadow = w_q * (scale / mean(|w_q|)).  Then:
      new_scale = mean(|shadow|) = (scale / p) * p = scale       (exact)
      new_w_q   = clip(round(shadow / scale), -1, 1) = w_q       (for p > 0.33)
    """
    out: dict[str, mx.array] = {}
    for name, packed in obj["ternary"].items():
        shape = obj["shapes"][name]
        scale = float(obj["scales"][name])
        numel = int(np.prod(shape))
        w_q = unpack_ternary_2bit(np.asarray(packed, dtype=np.uint8), numel)
        # p = fraction of non-zero weights.  Guard against all-zero layers.
        p = float(np.mean(np.abs(w_q)))
        effective_scale = scale / p if p > 1e-8 else scale
        w = (w_q.astype(np.float32) * effective_scale).reshape(shape)
        out[name] = mx.array(w)
    for name, arr_np in obj["passthrough"].items():
        out[name] = mx.array(np.array(arr_np, copy=True))
    return out


# ==============================================================================
# VALIDATION
# ==============================================================================

def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None
    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None) if tokenizer_name else None)
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, manifest says {expected_train_files}")
    return dataset_dir.name, actual_train_files, expected_train_files


def load_validation_tokens(pattern: str, seq_len: int) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(file) for file in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    compiled_loss,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    if val_batch_tokens < args.train_seq_len:
        raise ValueError(f"VAL_BATCH_SIZE too small: got {args.val_batch_size}")
    val_batch_seqs = val_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.size - 1) // args.train_seq_len
    if args.val_max_tokens > 0:
        max_seqs = args.val_max_tokens // args.train_seq_len
        total_seqs = min(total_seqs, max(max_seqs, 1))
    total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    for batch_idx, batch_seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * args.train_seq_len
        raw_end = batch_seq_end * args.train_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, args.train_seq_len)
        y_np = chunk[1:].reshape(-1, args.train_seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        chunk_token_count = float(y.size)
        batch_loss = compiled_loss(x, y).astype(mx.float32)
        mx.eval(batch_loss)
        total_loss_sum += float(batch_loss.item()) * chunk_token_count
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())
        if log_fn is not None and total_batches > 1 and (batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb


# ==============================================================================
# TRAINING HELPERS
# ==============================================================================

def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def accumulate_flat_grads(accum: dict[str, mx.array] | None, grads_tree: dict, scale: float) -> dict[str, mx.array]:
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


def loss_and_grad_chunked(args: Hyperparameters, train_loader: TokenLoader, compiled_loss_and_grad) -> tuple[mx.array, dict]:
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
        if args.mlx_eager_eval:
            mx.eval(loss_value, grad_accum)
    return loss_value, tree_unflatten(list(grad_accum.items()))


def clip_grad_tree(grads_tree: dict, max_norm: float) -> dict:
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = 0.0
    for grad in flat.values():
        total_sq += float(np.sum(np.square(np.array(grad.astype(mx.float32), dtype=np.float32)), dtype=np.float64))
    if total_sq <= 0.0:
        return grads_tree
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_tree
    scale = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])


# ==============================================================================
# TERNARY WEIGHT DIAGNOSTICS
# ==============================================================================

def ternary_diagnostics(model: TernaryModel) -> dict[str, float]:
    """Report ternary weight statistics for smoke-test validation."""
    all_ternary = []
    for name, p in tree_flatten(model.parameters()):
        if p.ndim == 2 and p.size > 4096:
            p_f32 = np.array(p.astype(mx.float32), dtype=np.float32)
            scale = float(np.mean(np.abs(p_f32))) + 1e-8
            w_q = np.clip(np.round(p_f32 / scale), -1, 1)
            all_ternary.append(w_q.flatten())
    if not all_ternary:
        return {}
    all_t = np.concatenate(all_ternary)
    total = len(all_t)
    frac_neg = float(np.sum(all_t == -1)) / total
    frac_zero = float(np.sum(all_t == 0)) / total
    frac_pos = float(np.sum(all_t == 1)) / total
    return {"frac_neg1": frac_neg, "frac_zero": frac_zero, "frac_pos1": frac_pos, "total_ternary_params": total}


# ==============================================================================
# HEBBIAN TRAINING (APPROACH C — NO BACKPROPAGATION)
# ==============================================================================

def hebbian_step(
    model: TernaryModel,
    input_ids: mx.array,
    target_ids: mx.array,
    lr: float,
    baseline_loss: float | None,
    ema_decay: float = 0.99,
) -> tuple[float, float]:
    """One training step using three-factor Hebbian learning. No backpropagation.

    For each linear layer, the update is:
        Δw = -lr * error_signal * sign(post.T @ pre)
    where:
        pre  = input to the linear layer
        post = output of the linear layer
        error_signal = current_loss - EMA(loss)   (REINFORCE-style baseline)

    When loss is better than average (error < 0), reinforce current correlations.
    When loss is worse than average (error > 0), weaken current correlations.
    """
    # Collect (linear_module, pre_activation, post_activation) during forward pass.
    pairs: list[tuple[nn.Module, mx.array, mx.array]] = []

    x = rms_norm(model.tok_emb(input_ids).astype(COMPUTE_DTYPE))

    for block in model.blocks:
        # --- Token mixer (GRU) ---
        gru = block.token_mixer
        normed = block.pre_token_norm(x)

        pre_gates = normed
        gates = gru.gates_proj(normed)
        pairs.append((gru.gates_proj, pre_gates, gates))

        D = gru.dim
        f = mx.sigmoid(gates[:, :, :D] + gru.forget_bias)
        c = gates[:, :, D:2*D] * mx.sigmoid(gates[:, :, D:2*D])
        o = mx.sigmoid(gates[:, :, 2*D:])
        h_all = parallel_scan(f, (1.0 - f) * c)

        pre_out = h_all * o
        gru_out = gru.out_proj(pre_out)
        pairs.append((gru.out_proj, pre_out, gru_out))

        token_out = block.post_token_norm(gru_out)
        x = x + block.token_scale.astype(x.dtype)[None, None, :] * token_out

        # --- Channel mixer (GatedMLP) ---
        mlp = block.channel_mixer
        normed = block.pre_mlp_norm(x)

        gate_raw = mlp.gate_proj(normed)
        pairs.append((mlp.gate_proj, normed, gate_raw))

        value = mlp.value_proj(normed)
        pairs.append((mlp.value_proj, normed, value))

        gated = mx.sigmoid(gate_raw) * value
        mlp_out = mlp.out_proj(gated)
        pairs.append((mlp.out_proj, gated, mlp_out))

        mlp_out = block.post_mlp_norm(mlp_out)
        x = x + block.mlp_scale.astype(x.dtype)[None, None, :] * mlp_out

    # Compute loss (for error signal + logging).
    x_flat = model.final_norm(x).reshape(-1, model.tok_emb.weight.shape[1])
    y_flat = target_ids.reshape(-1)
    logits = model.softcap(x_flat @ model.tok_emb.weight.astype(x_flat.dtype).T)
    loss = nn.losses.cross_entropy(logits.astype(mx.float32), y_flat, reduction="mean")
    mx.eval(loss)
    loss_val = float(loss.item())

    # Update baseline (EMA).
    if baseline_loss is None:
        baseline_loss = loss_val
    else:
        baseline_loss = ema_decay * baseline_loss + (1.0 - ema_decay) * loss_val

    # Error signal: positive when worse than average.
    error_signal = loss_val - baseline_loss

    # Apply three-factor Hebbian updates to all linear layers.
    effective_lr = lr * error_signal
    for linear, pre, post in pairs:
        pre_flat = pre.reshape(-1, pre.shape[-1]).astype(mx.float32)
        post_flat = post.reshape(-1, post.shape[-1]).astype(mx.float32)
        # Normalized correlation — divide by N so LR doesn't depend on batch size.
        N = float(pre_flat.shape[0])
        correlation = post_flat.T @ pre_flat / N
        hebbian = mx.sign(correlation)
        linear.weight = linear.weight - (effective_lr * hebbian).astype(linear.weight.dtype)
    # Materialize all weight updates.
    mx.eval(*[p[0].weight for p in pairs])

    return loss_val, baseline_loss


# ==============================================================================
# TRAINING
# ==============================================================================

def main() -> None:
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Running Python {sys.version}", console=False)
    log(f"Running MLX {mx.__version__}", console=False)
    log("=" * 100, console=False)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}")
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(args.data_path, args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size)

    # ---- Model + Optimizer ----
    mx.random.seed(args.seed)
    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    model = TernaryModel(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_softcap=args.logit_softcap,
        logit_chunk_tokens=args.logit_chunk_tokens,
        tied_embed_init_std=args.tied_embed_init_std,
        use_ternary=args.use_ternary,
        use_gru=args.use_gru,
        use_metaplastic=args.use_metaplastic,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        think_depth=args.think_depth,
        num_know_layers=args.num_know_layers,
        num_cycles=args.num_cycles,
    )
    opt = optim.Adam(learning_rate=args.lr, betas=[args.beta1, args.beta2], eps=args.adam_eps)

    # ---- Compiled functions ----
    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state, outputs=model.state,
    )

    # ---- Logging ----
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    n_ternary = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()) if p.ndim == 2 and p.size > 4096)
    est_artifact_mb = (n_ternary * 2 / 8 + (n_params - n_ternary) * 2) / 1e6 if args.use_ternary else n_params * 1 / 1e6
    log(f"run_id:{args.run_id}")
    arch = "gru+gated_mlp" if args.use_gru else "attention+relu2_mlp"
    log(f"architecture:{arch} use_ternary:{args.use_ternary} use_metaplastic:{args.use_metaplastic} use_hebbian:{args.use_hebbian}")
    if args.think_depth > 0:
        log(f"think_know: think_depth:{args.think_depth} know_layers:{args.num_know_layers} cycles:{args.num_cycles} "
            f"unique_blocks:{model.unique_block_count()} effective_depth:{model.effective_depth()}")
    log(f"model_params:{n_params} ternary_params:{n_ternary} est_artifact:{est_artifact_mb:.1f}MB")
    log(f"layers:{args.num_layers} dim:{args.model_dim} mlp_mult:{args.mlp_mult} seq_len:{args.train_seq_len}")
    log(f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps}")
    log(f"optimizer:adam lr:{args.lr} beta1:{args.beta1} beta2:{args.beta2}")
    log(f"compute_dtype:{COMPUTE_DTYPE}")

    diag = ternary_diagnostics(model)
    if diag:
        log(f"init_ternary_dist: -1:{diag['frac_neg1']:.3f} 0:{diag['frac_zero']:.3f} +1:{diag['frac_pos1']:.3f}")

    # ---- Warmup ----
    if args.warmup_steps > 0:
        for warmup_step in range(args.warmup_steps):
            accum: dict[str, mx.array] | None = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
                accum = accumulate_flat_grads(accum, grads, grad_scale)
            mx.eval(warmup_loss, accum)
            mx.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")

        # Prime the eval graph.
        val_batch_tokens = args.val_batch_size // args.grad_accum_steps
        warm_val_seqs = min(val_batch_tokens // args.train_seq_len, (val_tokens.size - 1) // args.train_seq_len)
        if warm_val_seqs > 0:
            warm_chunk = val_tokens[: warm_val_seqs * args.train_seq_len + 1]
            x_val = mx.array(warm_chunk[:-1].reshape(-1, args.train_seq_len), dtype=mx.int32)
            y_val = mx.array(warm_chunk[1:].reshape(-1, args.train_seq_len), dtype=mx.int32)
            warm_val_loss = compiled_loss(x_val, y_val)
            mx.eval(warm_val_loss)
            mx.synchronize()

        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    # ---- Training loop ----
    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step: int | None = None
    hebbian_baseline: float | None = None
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, compiled_loss, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=log)
            if step % 25 == 0 or last_step:
                log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{train_time_ms:.0f}ms")
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        step_t0 = time.perf_counter()

        if args.use_hebbian:
            # --- Hebbian path: no backprop, no Adam ---
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len)
            train_loss_value, hebbian_baseline = hebbian_step(
                model, x, y,
                lr=args.hebbian_lr * lr_mul,
                baseline_loss=hebbian_baseline,
                ema_decay=args.hebbian_ema_decay,
            )
        else:
            # --- Standard gradient path ---
            accum: dict[str, mx.array] | None = None
            train_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
                accum = accumulate_flat_grads(accum, grads, grad_scale)
                train_loss = train_loss + loss.astype(mx.float32) * grad_scale
                if args.mlx_eager_eval:
                    mx.eval(train_loss, accum)

            grads = tree_unflatten(list(accum.items()))
            grads = clip_grad_tree(grads, args.grad_clip_norm)
            train_loss_value = float(train_loss.item())

            # Adam update with LR scheduling.
            opt.learning_rate = args.lr * lr_mul
            params = dict(tree_flatten(model.parameters()))
            flat_grads = dict(tree_flatten(grads))
            updated = opt.apply_gradients(flat_grads, params)
            model.update(tree_unflatten(list(updated.items())))
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log(f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f}")
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    # ---- Final ternary diagnostics ----
    if args.use_ternary:
        diag = ternary_diagnostics(model)
        if diag:
            log(f"final_ternary_dist: -1:{diag['frac_neg1']:.3f} 0:{diag['frac_zero']:.3f} +1:{diag['frac_pos1']:.3f}")

    # ---- Save raw model ----
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    out_path = out_dir / f"{args.run_id}_ternary_model.npz"
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    # ---- Ternary serialization + roundtrip test ----
    if args.use_ternary:
        quant_obj, quant_stats = serialize_ternary_model(flat_state)
        quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
        quant_blob = zlib.compress(quant_raw, level=9)
        quant_path = out_dir / f"{args.run_id}_ternary_model.t2b.ptz"
        with quant_path.open("wb") as f:
            f.write(quant_blob)
        artifact_bytes = len(quant_blob) + len(code.encode("utf-8"))
        log(f"ternary_artifact: packed:{quant_stats['packed_bytes']} passthrough:{quant_stats['passthrough_bytes']} "
            f"zlib:{len(quant_blob)} code:{len(code.encode('utf-8'))} total_artifact:{artifact_bytes} "
            f"({artifact_bytes / 1e6:.2f}MB) limit:16.00MB {'OK' if artifact_bytes <= 16_000_000 else 'OVER'}")
        log(f"ternary_params:{quant_stats['ternary_params']} other_params:{quant_stats['other_params']}")

        # Roundtrip: decompress → deserialize → load → validate.
        with quant_path.open("rb") as f:
            rt_blob = f.read()
        rt_state = deserialize_ternary_model(pickle.loads(zlib.decompress(rt_blob)))
        model.update(tree_unflatten(list(rt_state.items())))
        rt_t0 = time.perf_counter()
        rt_val_loss, rt_val_bpb = eval_val(args, compiled_loss, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=log)
        rt_ms = 1000.0 * (time.perf_counter() - rt_t0)
        log(f"ternary_roundtrip val_loss:{rt_val_loss:.4f} val_bpb:{rt_val_bpb:.4f} eval_time:{rt_ms:.0f}ms")


if __name__ == "__main__":
    main()

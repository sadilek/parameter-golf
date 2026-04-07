"Ternary training script for OpenAI's Parameter Golf Challenge. Ciprian-Florin Ifrim - 24 March 2026"

import copy
import glob
import io
import math
import os
import random
import sys
import time
import lzma
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from flash_attn_interface import flash_attn_func  # FA3 (Hopper)
except ImportError:
    try:
        from flash_attn import flash_attn_func  # FA2
    except ImportError:
        # Fallback: PyTorch SDPA with GQA support.
        def flash_attn_func(q, k, v, causal=True):
            B, T, H, D = q.shape
            _, _, KVH, _ = k.shape
            if KVH != H:
                rep = H // KVH
                k = k.unsqueeze(3).expand(B, T, KVH, rep, D).reshape(B, T, H, D)
                v = v.unsqueeze(3).expand(B, T, KVH, rep, D).reshape(B, T, H, D)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)

# ---------------------------------------------------------------------------
# Hyperparameters (all configurable via environment variables)
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
    run_id = os.environ.get("RUN_ID", f"run_{int(time.time())}")
    seed = _e("SEED", 1337, int)
    compile_mode = _e("COMPILE_MODE", "default")
    val_batch_size = _e("VAL_BATCH_SIZE", 524288, int)
    val_loss_every = _e("VAL_LOSS_EVERY", 500, int)
    train_log_every = _e("TRAIN_LOG_EVERY", 10, int)
    iterations = _e("ITERATIONS", 2000, int)
    warmdown_fraction = _e("WARMDOWN_FRACTION", 0.2, float)
    warmup_steps = _e("WARMUP_STEPS", 20, int)
    lr_schedule = _e("LR_SCHEDULE", "warmdown")  # "warmdown", "1cycle", or "multicycle"
    onecycle_peak_frac = _e("ONECYCLE_PEAK_FRAC", 0.3, float)  # fraction of training at which LR peaks
    onecycle_min_div = _e("ONECYCLE_MIN_DIV", 10.0, float)  # LR starts and ends at peak/min_div
    num_cycles = _e("NUM_CYCLES", 3, int)  # number of cycles for multicycle schedule
    train_batch_tokens = _e("TRAIN_BATCH_TOKENS", 524288, int)
    train_seq_len = _e("TRAIN_SEQ_LEN", 1024, int)
    max_wallclock_seconds = _e("MAX_WALLCLOCK_SECONDS", 0.0, float)
    vocab_size = _e("VOCAB_SIZE", 1024, int)
    num_layers = _e("NUM_LAYERS", 16, int)
    num_kv_heads = _e("NUM_KV_HEADS", 4, int)
    model_dim = _e("MODEL_DIM", 512, int)
    num_heads = _e("NUM_HEADS", 8, int)
    mlp_mult = _e("MLP_MULT", 2, int)
    tie_embeddings = _e("TIE_EMBEDDINGS", 1, int)
    rope_base = _e("ROPE_BASE", 10000.0, float)
    rope_type = _e("ROPE_TYPE", "rope")
    yarn_max_len = _e("YARN_MAX_LEN", 4096, int)
    logit_softcap = _e("LOGIT_SOFTCAP", 30.0, float)
    softcap_type = _e("SOFTCAP_TYPE", "poly")
    tied_embed_init_std = _e("TIED_EMBED_INIT_STD", 0.005, float)
    qk_gain_init = _e("QK_GAIN_INIT", 1.5, float)
    activation_type = _e("ACTIVATION", "swiglu")
    embed_dim = _e("EMBED_DIM", 0, int)
    bigram_hash = _e("BIGRAM_HASH", 0, bool)
    mtp_heads_count = _e("MTP_HEADS", 0, int)
    training_depth_recurrence = _e("TRAINING_DEPTH_RECURRENCE", 1, int)
    eval_depth_recurrence = _e("EVAL_DEPTH_RECURRENCE", 1, int)
    stochastic_recurrence = _e("STOCHASTIC_RECURRENCE", 0, int)  # if >0, with p=0.2 run this many extra recurrences during training
    stochastic_recurrence_prob = _e("STOCHASTIC_RECURRENCE_PROB", 0.2, float)
    progressive_layers_start = _e("PROGRESSIVE_LAYERS_START", 0, int)  # start with N layers, grow to full
    progressive_grow_frac = _e("PROGRESSIVE_GROW_FRAC", 0.5, float)  # fraction of training at which to grow
    attn_proj_type = _e("ATTN_PROJ_TYPE", "standard")
    logit_head_type = _e("LOGIT_HEAD_TYPE", "standard")
    tversky_num_features = _e("TVERSKY_NUM_FEATURES", 16, int)
    tversky_feature_pools = _e("TVERSKY_FEATURE_POOLS", 0, int)
    tversky_membership = _e("TVERSKY_MEMBERSHIP", "sigmoid")
    diff_attn = _e("DIFF_ATTN", 0, bool)
    xsa_layers = _e("XSA_LAYERS", 0, int)  # number of top layers to apply XSA (0=disabled)
    mixer_type = _e("MIXER_TYPE", "attention")  # "attention", "causal_conv", "token_shift"
    ut_unique_blocks = _e("UT_UNIQUE_BLOCKS", 0, int)  # Universal Transformer: N unique blocks iterated (0=disabled, use num_layers)
    ut_iters = _e("UT_ITERS", 4, int)  # iterations per unique block
    low_rank = _e("LOW_RANK", 0, int)  # 0=full rank, >0=factored W=A@B with this rank
    lora_rank = _e("LORA_RANK", 0, int)  # 0=disabled, >0=frozen random base + LoRA adapters
    fedavg_every = _e("FEDAVG_EVERY", 0, int)  # 0=standard DDP, >0=Local SGD with weight sync every N steps
    refiner = _e("REFINER", 0, bool)
    refiner_kernel = _e("REFINER_KERNEL", 3, int)
    mlp_groups = _e("MLP_GROUPS", 0, int)
    embed_lr = _e("EMBED_LR", 0.6, float)
    head_lr = _e("HEAD_LR", 0.008, float)
    adam_lr = _e("ADAM_LR", 1e-3, float)
    adam_wd = _e("ADAM_WD", 0.05, float)
    untie_at_fraction = _e("UNTIE_AT_FRACTION", 0.0, float)
    tied_embed_lr = _e("TIED_EMBED_LR", 0.05, float)
    corr_weight_lr = _e("CORR_WEIGHT_LR", 0.05, float)
    smear = _e("SMEAR", 0, bool)
    seq_len_start = _e("SEQ_LEN_START", 0, int)
    seq_schedule_fraction = _e("SEQ_SCHEDULE_FRACTION", 0.33, float)
    batch_tokens_start = _e("BATCH_TOKENS_START", 0, int)
    batch_schedule_fraction = _e("BATCH_SCHEDULE_FRACTION", 0.33, float)
    churn_log_every = _e("CHURN_LOG_EVERY", 500, int)
    matrix_lr = _e("MATRIX_LR", 0.04, float)
    scalar_lr = _e("SCALAR_LR", 0.04, float)
    muon_momentum = _e("MUON_MOMENTUM", 0.95, float)
    muon_backend_steps = _e("MUON_BACKEND_STEPS", 5, int)
    muon_wd = _e("MUON_WD", 0.0, float)
    matrix_optimizer = _e("MATRIX_OPTIMIZER", "muon")
    muon_momentum_warmup_start = _e("MUON_MOMENTUM_WARMUP_START", 0.85, float)
    muon_momentum_warmup_steps = _e("MUON_MOMENTUM_WARMUP_STEPS", 500, int)
    beta1 = _e("BETA1", 0.9, float)
    beta2 = _e("BETA2", 0.95, float)
    adam_eps = _e("ADAM_EPS", 1e-8, float)
    grad_clip_norm = _e("GRAD_CLIP_NORM", 0.0, float)
    bitnet_group_size = _e("BITNET_GROUP_SIZE", 64, int)
    sliding_eval = _e("SLIDING_EVAL", 0, bool)
    sliding_eval_stride = _e("SLIDING_EVAL_STRIDE", 64, int)
    sliding_batch_size = _e("SLIDING_BATCH_SIZE", 64, int)
    temp_scaling = _e("TEMP_SCALING", 0, bool)
    _fp_raw = os.environ.get("FP_STORAGE", "0")
    fp_storage = True if _fp_raw == "FP8" else ("fp4" if _fp_raw == "FP4" else False)
    # CDMA superposition training.
    n_channels = _e("N_CHANNELS", 1, int)
    p_causal = _e("P_CAUSAL", 0.2, float)
    mask_rate_min = _e("MASK_RATE_MIN", 0.15, float)
    mask_rate_max = _e("MASK_RATE_MAX", 0.85, float)
    # EMA weight averaging.
    # Local conv for n-gram pattern matching.
    local_conv_layers = _e("LOCAL_CONV_LAYERS", 0, int)
    local_conv_kernel = _e("LOCAL_CONV_KERNEL", 7, int)
    local_conv_bottleneck = _e("LOCAL_CONV_BOTTLENECK", 256, int)
    # N-gram table builder.
    ngram_enabled = _e("NGRAM_ENABLED", 0, bool)
    ngram_min_order = _e("NGRAM_MIN_ORDER", 2, int)
    ngram_max_order = _e("NGRAM_MAX_ORDER", 7, int)
    ngram_buckets = _e("NGRAM_BUCKETS", 131072, int)
    ngram_top_k = _e("NGRAM_TOP_K", 3, int)
    ngram_check_every = _e("NGRAM_CHECK_EVERY", 500, int)  # convergence check interval (steps)
    ngram_converge_threshold = _e("NGRAM_CONVERGE_THRESHOLD", 0.99, float)  # stop when 99% stable
    ngram_table_path = _e("NGRAM_TABLE_PATH", "")  # path to precomputed tables (.pt file)
    ngram_inject_layers = _e("NGRAM_INJECT_LAYERS", 0, int)  # 0=input only, N=inject at all N layers
    ngram_input_inject = _e("NGRAM_INPUT_INJECT", 1, bool)  # enable input-level injection
    ngram_logit_mix = _e("NGRAM_LOGIT_MIX", 1, bool)  # enable output-level logit mixing
    bigram_logit_mix = _e("BIGRAM_LOGIT_MIX", 0, bool)  # enable exact bigram logit injection
    ensemble_path = _e("ENSEMBLE_PATH", "")  # path to a second model checkpoint for ensemble eval
    sse_enabled = _e("SSE_ENABLED", 0, bool)  # enable cascaded SSE calibration
    sse_clusters = _e("SSE_CLUSTERS", 32, int)
    sse_entropy_bins = _e("SSE_ENTROPY_BINS", 16, int)
    context_logit_bias = _e("CONTEXT_LOGIT_BIAS", 0, bool)
    clb_clusters = _e("CLB_CLUSTERS", 512, int)
    clb_context_len = _e("CLB_CONTEXT_LEN", 4, int)
    clb_rank = _e("CLB_RANK", 8, int)
    # Quantization mode: "ternary" (1.85 bits, ~66M params) or "int6" (6 bits, ~22M params)
    quant_mode = _e("QUANT_MODE", "ternary")
    quant_scheme = _e("QUANT_SCHEME", "")  # "lloyd_max_int4", "lloyd_max_int5", etc. Empty = use quant_mode default
    int6_clip_q = _e("INT6_CLIP_Q", 0.9999984, float)
    late_qat_frac = _e("LATE_QAT_FRAC", 0.0, float)  # fraction of training to start int6 STE (0=always on)
    # EMA disabled by default — causes roundtrip gap with ternary quantization.
    ema_enabled = _e("EMA_ENABLED", 0, bool)
    ema_decay = _e("EMA_DECAY", 0.999, float)
    ema_start_frac = _e("EMA_START_FRAC", 0.1, float)
    # Novel features (overnight experiments)
    moe_experts = _e("MOE_EXPERTS", 0, int)  # 0=disabled, N=N expert MLPs per layer with top-1 routing
    moe_topk = _e("MOE_TOPK", 1, int)
    foveated_layers = _e("FOVEATED_LAYERS", 0, int)  # N top layers use windowed attention
    foveated_window = _e("FOVEATED_WINDOW", 256, int)
    adaptive_depth = _e("ADAPTIVE_DEPTH", 0, bool)  # learned per-token layer-skip gates
    distill_from = _e("DISTILL_FROM", "")  # path to teacher model for distillation
    distill_alpha = _e("DISTILL_ALPHA", 0.5, float)  # weight of distillation loss vs CE loss
    structured_embed = _e("STRUCTURED_EMBED", 0, bool)  # byte-level embedding composition
    progressive_merge = _e("PROGRESSIVE_MERGE", 0, bool)  # train 2 branches, merge midway

CTP = ("attn_scale","attn_scales","mlp_scale","mlp_scales","resid_mix","resid_mixes","q_gain","diff_lambda","skip_weight","skip_weights","vocab_bias","refiner.gate","ngram_injector.stat_weights","ngram_injector.input_scale","ngram_injector.layer_scales","ngram_logit_mixer.logit_boost","ngram_logit_mixer.gate","xsa_gate","bigram_logit_mixer.gate","sse.token_scale","sse.cluster_bias","sse.entropy_temp","context_logit_bias.")

# ---------------------------------------------------------------------------
# Ternary packing — base-3 encoding (5 trits/byte)
# ---------------------------------------------------------------------------
def pack_ternary(q: Tensor):
    f = (q.reshape(-1).to(torch.int8) + 1).numpy()
    n = len(f)
    p = (5 - n % 5) % 5
    if p: f = np.concatenate([f, np.zeros(p, dtype=np.int8)])
    g = f.reshape(-1, 5).astype(np.uint8)
    return (g[:,0] + g[:,1]*3 + g[:,2]*9 + g[:,3]*27 + g[:,4]*81).tobytes(), n

def unpack_ternary(data: bytes, n: int) -> Tensor:
    v = np.frombuffer(data, dtype=np.uint8).astype(np.int16)
    t = np.zeros((len(v), 5), dtype=np.int8)
    for i in range(5): t[:,i] = v % 3; v //= 3
    return torch.from_numpy(t.reshape(-1)[:n].astype(np.int8) - 1)

def pack_ternary_bitmask(q: Tensor):
    f = q.reshape(-1).to(torch.int8).numpy(); n = len(f)
    nz = (f != 0)
    return np.packbits(nz).tobytes() + np.packbits(f[nz] > 0).tobytes(), n

def unpack_ternary_bitmask(data: bytes, n: int) -> Tensor:
    ms = (n + 7) // 8
    nz = np.unpackbits(np.frombuffer(data[:ms], dtype=np.uint8))[:n].astype(bool)
    s = np.unpackbits(np.frombuffer(data[ms:], dtype=np.uint8))[:int(nz.sum())].astype(bool)
    w = np.zeros(n, dtype=np.int8); w[nz] = np.where(s, 1, -1)
    return torch.from_numpy(w)

# ---------------------------------------------------------------------------
# FP4 quantization (per-row absmax, 2 values packed per byte)
# ---------------------------------------------------------------------------
def quantize_to_int4(t: Tensor) -> tuple[Tensor, Tensor, list]:
    t32 = t.float()
    orig_shape = t32.shape
    if t32.ndim < 2:
        t32 = t32.unsqueeze(0)
    absmax = t32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 7.0
    q = torch.clamp(torch.round(t32 / scale), -7, 7).to(torch.int8)
    flat = q.reshape(-1)
    if flat.numel() % 2 != 0:
        flat = F.pad(flat, (0, 1))
    low = (flat[0::2] + 8).to(torch.uint8)
    high = (flat[1::2] + 8).to(torch.uint8)
    return low | (high << 4), scale.half().squeeze(-1), list(orig_shape)

def dequantize_from_int4(packed: Tensor, scale: Tensor, shape: list) -> Tensor:
    low = (packed & 0x0F).to(torch.int8) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    flat = torch.zeros(packed.numel() * 2, dtype=torch.int8)
    flat[0::2] = low
    flat[1::2] = high
    numel = 1
    for s in shape:
        numel *= s
    flat = flat[:numel].float()
    if len(shape) <= 1:
        return (flat * scale.float().squeeze()).reshape(shape)
    return (flat.reshape(-1, shape[-1]) * scale.float().unsqueeze(-1)).reshape(shape)


# --- Int6 serialization ---
INT6_RANGE = 31
INT6_CLIP_Q_DEFAULT = 0.9999984
INT6_KEEP_FLOAT_PATTERNS = ("tok_emb", "lm_head", "embed_proj", "bigram_emb", "lm_head_correction", "channel_")
INT8_RANGE = 127

def quantize_int6(t: Tensor, quant_range: int = INT6_RANGE, clip_q: float = INT6_CLIP_Q_DEFAULT) -> tuple[Tensor, Tensor]:
    """Per-row int6 quantization. Returns (int8 container, fp16 scale)."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), clip_q, dim=1).clamp_min(1e-8) if t32.numel() else torch.zeros(t32.shape[0])
        scale = (clip_abs / float(quant_range)).clamp_min(1.0 / float(quant_range))
        clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
        q = torch.clamp(torch.round(clipped / scale[:, None]), -quant_range, quant_range).to(torch.int8)
        return q.contiguous(), scale.half().contiguous()
    # 1D: use int8 per-tensor
    clip_abs = float(torch.quantile(t32.abs().flatten(), clip_q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale.float()), -127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()

def q_sd_int6(state_dict: dict, clip_q: float = INT6_CLIP_Q_DEFAULT) -> tuple[dict, dict]:
    """Quantize state dict: int6 for large 2D matrices, int8 for embeddings, fp16 for small."""
    result = {}
    stats = {"int6_params": 0, "int8_params": 0, "fp_params": 0}
    for name, tensor in state_dict.items():
        if "mtp_heads" in name:
            continue
        t = tensor.detach().cpu().float().contiguous()
        is_keep_float = any(p in name for p in INT6_KEEP_FLOAT_PATTERNS)
        if t.ndim >= 2 and t.numel() > 4096 and not is_keep_float:
            # Int6 per-row for block weights
            t2d = t.reshape(t.shape[0], -1) if t.ndim > 2 else t
            q, s = quantize_int6(t2d, INT6_RANGE, clip_q)
            result[name + ".q"] = q
            result[name + ".s"] = s
            result[name + ".shape"] = torch.tensor(list(t.shape))
            stats["int6_params"] += t.numel()
        elif t.ndim >= 2 and t.numel() > 4096 and is_keep_float:
            # Int8 per-row for embeddings (no STE protection)
            t2d = t.reshape(t.shape[0], -1) if t.ndim > 2 else t
            q, s = quantize_int6(t2d, INT8_RANGE, clip_q)
            result[name + ".q"] = q
            result[name + ".s"] = s
            result[name + ".shape"] = torch.tensor(list(t.shape))
            stats["int8_params"] += t.numel()
        else:
            result[name] = t.half()
            stats["fp_params"] += t.numel()
    return result, stats

def deq_sd_int6(obj: dict, target_dtype=torch.bfloat16) -> dict:
    """Reconstruct state dict from int6/int8 quantized representation."""
    out = {}
    processed = set()
    for key in list(obj.keys()):
        if key.endswith(".q"):
            name = key[:-2]
            processed.add(name)
            q = obj[name + ".q"].float()
            s = obj[name + ".s"].float()
            shape = obj[name + ".shape"].tolist()
            if s.ndim == 0:
                t = (q * s).to(target_dtype)
            else:
                t = (q * s[:, None]).to(target_dtype)
            out[name] = t.reshape(shape).contiguous()
    for key, val in obj.items():
        name = key.removesuffix(".q").removesuffix(".s").removesuffix(".shape")
        if name not in processed and not key.endswith((".q", ".s", ".shape")):
            out[key] = val.to(target_dtype).contiguous()
    return out

# --- Lloyd-Max quantization (Gaussian-optimal) ---
def q_sd_lloyd_max(state_dict: dict, n_levels: int = 15) -> tuple[dict, dict]:
    """Quantize state dict using Lloyd-Max (Gaussian-optimal) quantization.
    Stores per-row sigma + int codes. Artifact includes n_levels for dequant LUT selection.
    """
    lut = torch.tensor(_compute_lloyd_max_lut(n_levels), dtype=torch.float32)
    half = n_levels // 2
    result = {"_lloyd_max_levels": torch.tensor(n_levels)}
    stats = {"lm_params": 0, "int8_params": 0, "fp_params": 0}
    for name, tensor in state_dict.items():
        if "mtp_heads" in name:
            continue
        t = tensor.detach().cpu().float().contiguous()
        is_keep_float = any(p in name for p in INT6_KEEP_FLOAT_PATTERNS)
        if t.ndim >= 2 and t.numel() > 4096 and not is_keep_float:
            t2d = t.reshape(t.shape[0], -1) if t.ndim > 2 else t
            sigma = t2d.std(dim=1).clamp_min(1e-8)
            normalized = t2d / sigma[:, None]
            diffs = (normalized.unsqueeze(-1) - lut.unsqueeze(0).unsqueeze(0)).abs()
            codes = (diffs.argmin(dim=-1) - half).to(torch.int8)
            result[name + ".q"] = codes.contiguous()
            result[name + ".s"] = sigma.half().contiguous()
            result[name + ".shape"] = torch.tensor(list(t.shape))
            stats["lm_params"] += t.numel()
        elif t.ndim >= 2 and t.numel() > 4096 and is_keep_float:
            # Int8 per-row for embeddings (same as int6 path)
            t2d = t.reshape(t.shape[0], -1) if t.ndim > 2 else t
            q, s = quantize_int6(t2d, INT8_RANGE)
            result[name + ".q"] = q
            result[name + ".s"] = s
            result[name + ".shape"] = torch.tensor(list(t.shape))
            stats["int8_params"] += t.numel()
        else:
            result[name] = t.half()
            stats["fp_params"] += t.numel()
    return result, stats

def deq_sd_lloyd_max(obj: dict, target_dtype=torch.bfloat16) -> dict:
    """Reconstruct state dict from Lloyd-Max quantized representation."""
    n_levels = int(obj.get("_lloyd_max_levels", torch.tensor(15)).item())
    lut = torch.tensor(_compute_lloyd_max_lut(n_levels), dtype=torch.float32)
    half = n_levels // 2
    out = {}
    processed = set()
    for key in list(obj.keys()):
        if key.endswith(".q"):
            name = key[:-2]
            processed.add(name)
            q = obj[name + ".q"]
            s = obj[name + ".s"].float()
            shape = obj[name + ".shape"].tolist()
            is_keep_float = any(p in name for p in INT6_KEEP_FLOAT_PATTERNS)
            if is_keep_float:
                # Int8 uniform dequant for embeddings
                t = (q.float() * s[:, None]).to(target_dtype) if s.ndim > 0 else (q.float() * s).to(target_dtype)
            else:
                # Lloyd-Max dequant: w = sigma * LUT[code + half]
                indices = (q.long() + half).clamp(0, n_levels - 1)
                recon = lut[indices]
                t = (recon * s[:, None]).to(target_dtype)
            out[name] = t.reshape(shape).contiguous()
    for key, val in obj.items():
        name = key.removesuffix(".q").removesuffix(".s").removesuffix(".shape")
        if name not in processed and not key.endswith((".q", ".s", ".shape")) and key != "_lloyd_max_levels":
            out[key] = val.to(target_dtype).contiguous()
    return out

# ---------------------------------------------------------------------------
# State dict serialization (ternary + fp16/fp8/fp4)
# ---------------------------------------------------------------------------
def q_sd(state_dict: dict, group_size: int = 64, fp_storage=False, ternary_method="standard", ternary_override_names: set | None = None) -> tuple[dict, dict]:
    "Ternary for large 2D weight matrices, fp16/fp8/fp4 for everything else."
    quantized = {}
    stats = {"ternary_params": 0, "ternary_bytes": 0, "fp_params": 0, "fp_bytes": 0}
    for name, tensor in state_dict.items():
        if "mtp_heads" in name:
            continue
        t = tensor.detach().cpu().float().contiguous()
        t_orig_shape = list(t.shape)
        if t.ndim == 3:
            t = t.reshape(t.shape[0], -1)
        is_ternary_candidate = (
            t.ndim == 2 and t.numel() > 65_536
            and "tok_emb" not in name and "lm_head" not in name and "embed_proj" not in name and "bigram_emb" not in name and "lm_head_correction" not in name and "lm_head_U" not in name and "lm_head_V" not in name
            and "prototypes" not in name and "tversky" not in name
        ) or (ternary_override_names is not None and name in ternary_override_names)
        if is_ternary_candidate:
            pad = (group_size - t.shape[1] % group_size) % group_size
            t_padded = F.pad(t, (0, pad)) if pad > 0 else t
            t_grouped = t_padded.reshape(-1, group_size)
            scale = t_grouped.abs().mean(-1, keepdim=True).clamp(min=1e-8).half().float()
            q = (t_grouped / scale).round().clamp(-1, 1).to(torch.int8)

            if ternary_method == "standard":
                packed_bytes, n_trits = pack_ternary(q)
                entry_type = "ternary"
            else:
                packed_bytes, n_trits = pack_ternary_bitmask(q)
                entry_type = "ternary_bitmask"

            quantized[name] = {
                "type": entry_type, "packed": packed_bytes,
                "scale": scale.half().squeeze(-1),
                "shape": list(t.shape), "padded_cols": t_padded.shape[1],
                "group_size": group_size, "n_trits": n_trits,
                "orig_shape": t_orig_shape,
            }
            stats["ternary_params"] += t.numel()
            stats["ternary_bytes"] += len(packed_bytes) + scale.numel() * 2
        elif fp_storage == "fp4" and t.ndim == 2:
            packed, scale, orig_shape = quantize_to_int4(t)
            quantized[name] = {"type": "fp4", "packed": packed, "scale": scale, "shape": orig_shape}
            stats["fp_params"] += t.numel()
            stats["fp_bytes"] += packed.numel() + scale.numel() * 2
        elif fp_storage and t.ndim == 2:
            quantized[name] = {"type": "fp8", "data": t.to(torch.float8_e4m3fn)}
            stats["fp_params"] += t.numel()
            stats["fp_bytes"] += t.numel()
        else:
            quantized[name] = {"type": "fp16", "data": t.half(), "orig_shape": t_orig_shape}
            stats["fp_params"] += t.numel()
            stats["fp_bytes"] += t.numel() * 2
    return quantized, stats

def deq_sd(quantized: dict, target_dtype=torch.bfloat16):
    "Reconstruct full-precision state dict from quantized representation."
    out = {}
    for name, entry in quantized.items():
        if entry["type"] in ("ternary", "ternary_bitmask"):
            if entry["type"] == "ternary":
                q = unpack_ternary(entry["packed"], entry["n_trits"])
            else:
                q = unpack_ternary_bitmask(entry["packed"], entry["n_trits"])

            q = q.float().reshape(-1, entry["group_size"])
            scale = entry["scale"].float().unsqueeze(-1)
            q_absmean = q.abs().mean(-1, keepdim=True).clamp(min=1e-8)
            t = (q * (scale / q_absmean)).reshape(-1, entry["padded_cols"])
            shape = entry["shape"]
            result = t[:shape[0], :shape[1]].to(target_dtype)
            orig = entry.get("orig_shape")
            out[name] = result.reshape(orig).contiguous() if orig and orig != shape else result.contiguous()
        elif entry["type"] == "fp8":
            out[name] = entry["data"].to(torch.float32).to(target_dtype).contiguous()
        elif entry["type"] == "fp4":
            out[name] = dequantize_from_int4(entry["packed"], entry["scale"], entry["shape"]).to(target_dtype).contiguous()
        else:
            t = entry["data"].to(target_dtype)
            orig = entry.get("orig_shape")
            out[name] = t.reshape(orig).contiguous() if orig and list(t.shape) != orig else t.contiguous()
    return out

# ---------------------------------------------------------------------------
# Ternary diagnostics (logged during training)
# ---------------------------------------------------------------------------
def tern_stats(model: nn.Module, group_size: int = 64):
    total = zeros = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2 and ("weight" in name or "prototypes" in name) and p.shape[0] > 1 and p.numel() % group_size == 0:
                w = p.detach().float().reshape(-1, group_size)
                scale = w.abs().mean(-1, keepdim=True).clamp(min=1e-8).half().float()
                q = (w / scale).round().clamp(-1, 1)
                zeros += int((q == 0).sum().item())
                total += int(q.numel())
    return {"zero_frac": zeros / max(total, 1), "total_weights": total}

_prev_committed: dict = {}

def churn_fn(model: nn.Module, group_size: int = 64):
    global _prev_committed
    total = flipped = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2 and ("weight" in name or "prototypes" in name) and p.shape[0] > 1 and p.numel() % group_size == 0:
                w = p.detach().float().reshape(-1, group_size)
                scale = w.abs().mean(-1, keepdim=True).clamp(min=1e-8).half().float()
                q = (w / scale).round().clamp(-1, 1).cpu().numpy()
                if name in _prev_committed:
                    flipped += int(np.sum(q != _prev_committed[name]))
                    total += q.size
                _prev_committed[name] = q
    return flipped / max(total, 1)

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
# N-gram table builder (incremental, with convergence telemetry)
# ---------------------------------------------------------------------------

class NgramTableBuilder:
    """Incrementally builds n-gram top-K prediction tables from training tokens.

    During training, call update() with each batch of tokens. The builder maintains
    hash tables mapping context hashes to next-token counts. Call top_k_tables() to
    extract the current top-K predictions per bucket.

    Convergence telemetry: tracks what fraction of buckets have stable top-3
    predictions. When stable_frac >= threshold, stop updating to save compute.
    """
    def __init__(self, orders: list[int], num_buckets: int = 131072, top_k: int = 3,
                 vocab_size: int = 1024, device: str = "cpu"):
        self.orders = orders
        self.num_buckets = num_buckets
        self.top_k = top_k
        self.vocab_size = vocab_size
        self.converged = False
        self.tokens_seen = 0
        # Count matrices: (num_buckets, vocab_size) per order. Keep on CPU to save GPU memory.
        self.counts = {o: np.zeros((num_buckets, vocab_size), dtype=np.uint32) for o in orders}
        # For convergence tracking: snapshot of top-K from last check.
        self._prev_top_k: dict[int, np.ndarray] | None = None

    def update(self, tokens: torch.Tensor) -> None:
        """Update counts from a batch of tokens. tokens: 1D int tensor."""
        if self.converged:
            return
        tok = tokens.cpu().numpy().astype(np.int64)
        n = len(tok)
        self.tokens_seen += n
        max_order = max(self.orders)
        if n <= max_order:
            return
        for order in self.orders:
            cm = self.counts[order]
            # Vectorized hash computation.
            positions = np.arange(order, n)
            h = np.zeros(len(positions), dtype=np.uint64)
            for k in range(order - 1):
                h = (h * 1000003 + tok[positions - order + 1 + k].astype(np.uint64)) & 0xFFFFFFFF
            buckets = (h % self.num_buckets).astype(np.int64)
            next_tok = tok[positions]
            np.add.at(cm, (buckets, next_tok), 1)

    def check_convergence(self) -> float:
        """Check what fraction of active buckets have stable top-K. Returns stability fraction."""
        current_top_k = {}
        for order in self.orders:
            cm = self.counts[order]
            top_indices = np.argpartition(-cm, self.top_k, axis=1)[:, :self.top_k]
            rows = np.arange(self.num_buckets)[:, None]
            top_counts = cm[rows, top_indices]
            sort_order = np.argsort(-top_counts, axis=1)
            current_top_k[order] = top_indices[rows, sort_order]

        if self._prev_top_k is None:
            self._prev_top_k = current_top_k
            return 0.0

        # Compare current vs previous top-K across all orders.
        total_active = 0
        total_stable = 0
        for order in self.orders:
            curr = current_top_k[order]
            prev = self._prev_top_k[order]
            active = self.counts[order].sum(axis=1) > 0
            n_active = active.sum()
            if n_active == 0:
                continue
            # A bucket is stable if its top-K tokens haven't changed.
            same = np.all(curr[active] == prev[active], axis=1)
            total_active += n_active
            total_stable += same.sum()

        self._prev_top_k = current_top_k
        return total_stable / max(total_active, 1)

    def top_k_tables(self) -> dict[int, torch.Tensor]:
        """Extract current top-K token IDs per bucket per order."""
        result = {}
        for order in self.orders:
            cm = self.counts[order]
            top_indices = np.argpartition(-cm, self.top_k, axis=1)[:, :self.top_k]
            rows = np.arange(self.num_buckets)[:, None]
            top_counts = cm[rows, top_indices]
            sort_order = np.argsort(-top_counts, axis=1)
            result[order] = torch.from_numpy(top_indices[rows, sort_order].astype(np.int16))
        return result

    def serialize(self) -> dict:
        """Pack tables for artifact storage."""
        tables = self.top_k_tables()
        return {
            "ngram_orders": self.orders,
            "ngram_buckets": self.num_buckets,
            "ngram_top_k": self.top_k,
            **{f"ngram_order_{o}": tables[o] for o in self.orders},
        }

    def artifact_size_estimate(self) -> int:
        """Estimate artifact size in bytes (before compression)."""
        return len(self.orders) * self.num_buckets * self.top_k * 2  # int16


class NgramInjector(nn.Module):
    """Additive n-gram statistical prediction injection.

    For each token position, looks up top-K predicted next tokens across multiple
    n-gram orders. Creates a weighted embedding that is ADDED to the residual stream.
    Zero extra parameters beyond small order/rank encodings and per-layer scales.

    Can inject at input only (ngram_num_inject_layers=0) or at every transformer
    layer (residual-style, ngram_num_inject_layers=N).
    """
    def __init__(self, orders: list[int], num_buckets: int, top_k: int,
                 embed_dim: int, model_dim: int, tok_emb: nn.Embedding,
                 num_inject_layers: int = 0):
        super().__init__()
        self.orders = orders
        self.num_buckets = num_buckets
        self.top_k = top_k
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.tok_emb = tok_emb  # shared with model's token embedding

        # Learnable weights per order×rank to control contribution.
        self.stat_weights = nn.Parameter(torch.ones(max(orders) + 1, top_k) * 0.01)

        # Global scale for input injection (starts small so n-gram signal doesn't
        # overwhelm at init — model learns to turn it up).
        self.input_scale = nn.Parameter(torch.tensor(0.1))

        # Per-layer scales for residual injection (0 = input only).
        self.num_inject_layers = num_inject_layers
        if num_inject_layers > 0:
            self.layer_scales = nn.Parameter(torch.zeros(num_inject_layers))

        # If embed_dim != model_dim, we need a small projection for the stat signal.
        if embed_dim != model_dim:
            self.stat_proj = nn.Linear(embed_dim, model_dim, bias=False)
        else:
            self.stat_proj = None

        # Hash tables: registered as buffers (not parameters, no gradients).
        for order in orders:
            self.register_buffer(f"table_{order}", torch.zeros(num_buckets, top_k, dtype=torch.long))

        # Cache for the computed stat embedding (reused across layers).
        self._cached_stat_emb: Tensor | None = None

    def load_tables(self, tables: dict[int, torch.Tensor]) -> None:
        """Load precomputed n-gram tables."""
        for order in self.orders:
            getattr(self, f"table_{order}").copy_(tables[order].long())

    def _hash_context(self, tokens: Tensor, order: int) -> Tensor:
        """Hash the last (order-1) tokens at each position. tokens: (B, T)."""
        B, T = tokens.shape
        h = torch.zeros(B, T, dtype=torch.long, device=tokens.device)
        for k in range(order - 1):
            offset = order - 2 - k
            if offset < T:
                shifted = F.pad(tokens[:, :T-offset], (offset, 0), value=0) if offset > 0 else tokens
                h = (h * 1000003 + shifted) & 0xFFFFFFFF
        return h % self.num_buckets

    def compute_stat_emb(self, tokens: Tensor) -> Tensor:
        """Compute n-gram prediction embedding. Call once, reuse across layers.

        Returns:
            (B, T, model_dim) n-gram prediction embedding
        """
        B, T = tokens.shape
        E = self.embed_dim

        stat_emb = torch.zeros(B, T, E, device=tokens.device, dtype=torch.bfloat16)
        for order in self.orders:
            table = getattr(self, f"table_{order}")  # (num_buckets, top_k)
            bucket_idx = self._hash_context(tokens, order)  # (B, T)
            top_k_tokens = table[bucket_idx]  # (B, T, top_k)

            for r in range(self.top_k):
                with torch.no_grad():
                    pred_emb = self.tok_emb(top_k_tokens[:, :, r])  # (B, T, embed_dim)
                weight = self.stat_weights[order, r]
                stat_emb = stat_emb + weight * pred_emb

        if self.stat_proj is not None:
            stat_emb = self.stat_proj(stat_emb)

        self._cached_stat_emb = stat_emb
        return stat_emb

    def inject_input(self, tok_emb_out: Tensor, stat_emb: Tensor) -> Tensor:
        """Add scaled n-gram signal to token embeddings. Zero extra params."""
        return tok_emb_out + self.input_scale * stat_emb

    def inject_layer(self, x: Tensor, layer_idx: int) -> Tensor:
        """Add per-layer scaled n-gram signal to residual stream."""
        if self.num_inject_layers == 0 or self._cached_stat_emb is None:
            return x
        if layer_idx >= self.num_inject_layers:
            return x
        return x + self.layer_scales[layer_idx] * self._cached_stat_emb


class NgramLogitMixer(nn.Module):
    """Mixes n-gram predictions into model logits before softmax.

    For each position, looks up n-gram top-K predicted tokens and adds learned
    logit biases for those tokens. The model learns through backprop how much
    to trust n-gram predictions at each order/rank.

    ~20 scalar params total. No embedding lookups, no projection layers.
    """
    def __init__(self, orders: list[int], num_buckets: int, top_k: int, vocab_size: int):
        super().__init__()
        self.orders = orders
        self.num_buckets = num_buckets
        self.top_k = top_k
        self.vocab_size = vocab_size

        # Per-order, per-rank logit boost (learned).
        # Initialized to small positive values so n-gram predictions start helpful.
        self.logit_boost = nn.Parameter(torch.full((max(orders) + 1, top_k), 0.5))

        # Global gate (starts small positive to let gradients flow from the start).
        self.gate = nn.Parameter(torch.tensor(0.1))

        # Hash tables (buffers, not parameters).
        for order in orders:
            self.register_buffer(f"table_{order}",
                                 torch.zeros(num_buckets, top_k, dtype=torch.long))

    def load_tables(self, tables: dict[int, torch.Tensor]) -> None:
        for order in self.orders:
            getattr(self, f"table_{order}").copy_(tables[order].long())

    def _hash_context(self, tokens: Tensor, order: int) -> Tensor:
        B, T = tokens.shape
        h = torch.zeros(B, T, dtype=torch.long, device=tokens.device)
        for k in range(order - 1):
            offset = order - 2 - k
            if offset < T:
                shifted = F.pad(tokens[:, :T-offset], (offset, 0), value=0) if offset > 0 else tokens
                h = (h * 1000003 + shifted) & 0xFFFFFFFF
        return h % self.num_buckets

    def forward(self, tokens: Tensor, model_logits: Tensor) -> Tensor:
        """Add n-gram logit biases to model logits.

        Args:
            tokens: (B, T) input token IDs
            model_logits: (B, T, vocab_size)

        Returns:
            (B, T, vocab_size) adjusted logits
        """
        B, T, V = model_logits.shape

        # Build sparse n-gram logit adjustment.
        ngram_logits = torch.zeros(B, T, V, device=model_logits.device, dtype=model_logits.dtype)

        for order in self.orders:
            table = getattr(self, f"table_{order}")
            bucket_idx = self._hash_context(tokens, order)  # (B, T)
            top_k_tokens = table[bucket_idx]  # (B, T, top_k)

            for r in range(self.top_k):
                boost = self.logit_boost[order, r]
                token_ids = top_k_tokens[:, :, r].unsqueeze(-1)  # (B, T, 1)
                # Use expand + scatter_add_ (no .item() to stay compile-safe).
                boost_vals = boost.to(ngram_logits.dtype).expand(B, T).unsqueeze(-1)
                ngram_logits.scatter_add_(2, token_ids, boost_vals)

        return model_logits + self.gate * ngram_logits


class ContextLogitBias(nn.Module):
    """Context-dependent logit bias table (PAQ indirect context modeling).

    Hashes the previous K tokens to a cluster ID, looks up a learned logit
    bias vector. Captures context-specific token distribution shifts.
    Uses low-rank bias vectors to save parameters.

    With rank=8, 512 clusters: 512*8 + 8*1024 = 12K params ≈ 24KB.
    """
    def __init__(self, vocab_size: int = 1024, n_clusters: int = 512,
                 context_len: int = 4, rank: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_clusters = n_clusters
        self.context_len = context_len
        # Low-rank factorization: bias[cluster] = U[cluster] @ V
        self.U = nn.Parameter(torch.randn(n_clusters, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, vocab_size) * 0.01)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def _hash_context(self, tokens: Tensor) -> Tensor:
        """Hash previous context_len tokens at each position. tokens: (B, T)."""
        B, T = tokens.shape
        h = torch.zeros(B, T, dtype=torch.long, device=tokens.device)
        for k in range(self.context_len):
            offset = self.context_len - 1 - k
            if offset > 0:
                shifted = F.pad(tokens[:, :T-offset], (offset, 0), value=0)
            else:
                shifted = tokens
            h = (h * 1000003 + shifted) & 0xFFFFFFFF
        return h % self.n_clusters

    def forward(self, input_ids: Tensor, model_logits: Tensor) -> Tensor:
        """Add context-dependent logit bias."""
        cluster_ids = self._hash_context(input_ids)  # (B, T)
        # Low-rank lookup: U[cluster] @ V → (B, T, vocab_size)
        u = self.U[cluster_ids]  # (B, T, rank)
        bias = u @ self.V  # (B, T, vocab_size)
        return model_logits + self.gate * bias.to(model_logits.dtype)


class CascadedSSE(nn.Module):
    """Cascaded Secondary Symbol Estimation (PAQ-style calibration).

    Three stages of post-model calibration that correct systematic biases:
    1. Per-token logit scale — fixes per-token quantization/training bias
    2. Prev-token-cluster adjustment — fixes bigram-level miscalibration
    3. Entropy-dependent temperature — fixes confidence miscalibration

    Total params: ~1024 + n_clusters*1024 + n_entropy_bins ≈ small
    """
    def __init__(self, vocab_size: int = 1024, n_clusters: int = 32, n_entropy_bins: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_clusters = n_clusters
        self.n_entropy_bins = n_entropy_bins

        # Stage 1: per-token logit scale (init 1.0 = identity).
        self.token_scale = nn.Parameter(torch.ones(vocab_size))

        # Stage 2: prev-token-cluster → logit bias vector.
        # Cluster assignment: simple modular hash of prev_token.
        # Each cluster gets a small logit bias vector.
        self.cluster_bias = nn.Parameter(torch.zeros(n_clusters, vocab_size) * 0.01)

        # Stage 3: entropy-dependent temperature.
        # Indexed by quantized entropy bin → temperature multiplier.
        self.entropy_temp = nn.Parameter(torch.ones(n_entropy_bins))

    def forward(self, logits: Tensor, input_ids: Tensor | None = None) -> Tensor:
        """Apply cascaded calibration stages.

        Args:
            logits: (B, T, V) or (B*T, V) — model logits AFTER softcap
            input_ids: (B, T) — needed for stage 2 (prev token context)
        """
        was_flat = logits.dim() == 2

        # Stage 1: per-token scale.
        logits = logits * self.token_scale.to(logits.dtype)

        # Stage 2: prev-token-cluster bias (only if we have input_ids).
        if input_ids is not None and not was_flat:
            B, T = input_ids.shape
            cluster_ids = input_ids % self.n_clusters  # (B, T)
            bias = self.cluster_bias[cluster_ids]  # (B, T, V)
            logits = logits + bias.to(logits.dtype)

        # Stage 3: entropy-dependent temperature.
        with torch.no_grad():
            entropy = -(F.softmax(logits.float(), dim=-1) * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1)
            max_entropy = math.log(self.vocab_size)
            bin_idx = (entropy / max_entropy * (self.n_entropy_bins - 1)).long().clamp(0, self.n_entropy_bins - 1)
        temp = self.entropy_temp[bin_idx].unsqueeze(-1).to(logits.dtype)  # (..., 1)
        logits = logits * temp

        return logits


class BigramLogitMixer(nn.Module):
    """Exact bigram count-based logit injection.

    Stores a full 1024×1024 bigram count table. For each position, computes
    log P(token | prev_token) from smoothed counts and adds it to model logits
    with a learned per-order gate. Produces properly normalized distributions.

    Artifact cost: 1024×1024×2 bytes = 2MB (int16 counts).
    """
    def __init__(self, vocab_size: int = 1024, smoothing: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        # Learned gate: starts at 0.5 so n-gram has immediate effect.
        self.gate = nn.Parameter(torch.tensor(0.5))
        # Count table: registered as buffer (not trained).
        self.register_buffer("bigram_counts", torch.zeros(vocab_size, vocab_size, dtype=torch.float32))
        # Precomputed log-probs (updated after count table is loaded).
        self.register_buffer("bigram_logprobs", torch.zeros(vocab_size, vocab_size, dtype=torch.float32))

    def build_from_shards(self, shard_pattern: str, max_shards: int = 0) -> None:
        """Build exact bigram counts from training shards."""
        import glob as _glob
        files = sorted(_glob.glob(shard_pattern))
        if max_shards > 0:
            files = files[:max_shards]
        counts = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float64)
        total_tokens = 0
        for sf in files:
            header = np.fromfile(sf, dtype="<i4", count=256)
            n = int(header[2])
            tokens = np.fromfile(sf, dtype="<u2", count=n, offset=256 * 4)
            total_tokens += n
            # Vectorized bigram counting
            prev = tokens[:-1].astype(np.int64)
            cur = tokens[1:].astype(np.int64)
            np.add.at(counts, (prev, cur), 1)
        self.bigram_counts.copy_(torch.from_numpy(counts).float())
        self._compute_logprobs()
        return total_tokens

    def _compute_logprobs(self) -> None:
        """Convert counts to smoothed log-probabilities."""
        # Add-alpha smoothing: P(token|prev) = (count + alpha) / (total + alpha * V)
        alpha = self.smoothing
        V = self.vocab_size
        smoothed = self.bigram_counts + alpha
        row_sums = smoothed.sum(dim=1, keepdim=True)
        self.bigram_logprobs.copy_(torch.log(smoothed / row_sums))

    def load_counts(self, counts: torch.Tensor) -> None:
        """Load precomputed count table."""
        self.bigram_counts.copy_(counts.float())
        self._compute_logprobs()

    def forward(self, input_ids: Tensor, model_logits: Tensor) -> Tensor:
        """Add bigram log-probs to model logits.

        Args:
            input_ids: (B, T) input token IDs
            model_logits: (B, T, vocab_size)

        Returns:
            (B, T, vocab_size) adjusted logits
        """
        # Look up bigram log-probs for each position based on previous token.
        # prev_tokens[t] = input_ids[t] (predicting input_ids[t+1]).
        # But model_logits[t] predicts the next token after position t.
        prev_tokens = input_ids  # (B, T)
        bigram_lp = self.bigram_logprobs[prev_tokens]  # (B, T, vocab_size)
        return model_logits + self.gate * bigram_lp.to(model_logits.dtype)

    def serialize_counts(self) -> torch.Tensor:
        """Compress counts to int16 for artifact storage."""
        # Cap counts at 32767 (int16 max) and store.
        return self.bigram_counts.clamp(max=32767).to(torch.int16)


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
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = ld_shard(self.files[0])
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = ld_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start:start + per_rank_span].pin_memory().to(self.device, non_blocking=True).to(torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x, y

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

def apply_qat_ste(w: Tensor, fp_storage: str | bool) -> Tensor:
    """Applies Straight-Through Estimator (STE) for FP4 or FP8 simulated quantization."""
    if not fp_storage:
        return w
    if fp_storage == "fp4":
        absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = absmax / 7.0
        q = torch.clamp(torch.round(w / scale), -7.0, 7.0)
        w_sim = q * scale
        return (w_sim - w).detach() + w
    elif fp_storage is True or fp_storage == "fp8":
        w_sim = w.to(torch.float8_e4m3fn).to(w.dtype)
        return (w_sim - w).detach() + w
    return w

class QATLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, fp_storage: str | bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.fp_storage = fp_storage

    def forward(self, x: Tensor) -> Tensor:
        w_qat = apply_qat_ste(self.weight, self.fp_storage)
        return F.linear(x, w_qat.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class QATEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, fp_storage: str | bool = False):
        super().__init__(num_embeddings, embedding_dim)
        self.fp_storage = fp_storage

    def forward(self, input: Tensor) -> Tensor:
        w_qat = apply_qat_ste(self.weight, self.fp_storage)
        return F.embedding(input, w_qat, self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq, self.sparse)

# Global quantization mode — set from Hyperparameters in main().
_QUANT_MODE = "ternary"
_INT6_CLIP_Q = 0.9999984
_INT6_ACTIVE = True  # toggled by late QAT schedule
_STE_ENABLED = True  # set STE_ENABLED=0 to disable
_STE_TYPE = "uniform"  # "uniform" (current int6) or "lloyd_max" (Gaussian-optimal)

# Lloyd-Max LUT for Gaussian-optimal quantization (precomputed for N(0,1))
_LLOYD_MAX_LUT_CACHE: dict[int, Tensor] = {}

def _compute_lloyd_max_lut(n_levels: int) -> list[float]:
    """Compute optimal Lloyd-Max centroids for N(0,1) with n_levels."""
    import math as _m
    pdf = lambda x: _m.exp(-0.5*x*x) / _m.sqrt(2*_m.pi)
    cdf = lambda x: 0.5 * (1 + _m.erf(x / _m.sqrt(2)))
    INF = 10.0
    b = [-INF] + [-4.0 + 8.0*i/n_levels for i in range(1, n_levels)] + [INF]
    c = [0.0] * n_levels
    for _ in range(200):
        for i in range(n_levels):
            pl, ph = cdf(b[i]), cdf(b[i+1])
            c[i] = (pdf(b[i]) - pdf(b[i+1])) / (ph - pl) if ph - pl > 1e-15 else (b[i]+b[i+1])/2
        for i in range(1, n_levels):
            b[i] = (c[i-1] + c[i]) / 2
    return c

def get_lloyd_max_lut(n_levels: int, device) -> Tensor:
    """Get cached Lloyd-Max LUT as tensor."""
    if n_levels not in _LLOYD_MAX_LUT_CACHE or _LLOYD_MAX_LUT_CACHE[n_levels].device != device:
        _LLOYD_MAX_LUT_CACHE[n_levels] = torch.tensor(
            _compute_lloyd_max_lut(n_levels), dtype=torch.float32, device=device)
    return _LLOYD_MAX_LUT_CACHE[n_levels]

class QuantizedLinear(nn.Linear):
    """Quantized linear layer. Supports ternary ({-1,0,1}) or int6 ([-31,31]) STE.
    Per-layer override: set layer.ste_override = "ternary"/"int6"/"none" to override global mode.
    """
    def __init__(self, in_features, out_features, bias=False, group_size=64):
        super().__init__(in_features, out_features, bias=bias)
        self.group_size = group_size
        self.ste_override = None  # None = use global, "ternary"/"int6"/"none" = per-layer

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        # Determine STE mode: per-layer override or global
        mode = self.ste_override if self.ste_override is not None else (
            _QUANT_MODE if _STE_ENABLED else "none")
        if not self.training:
            mode = "none"
        if mode == "int6" and _INT6_ACTIVE:
            if _STE_TYPE == "lloyd_max":
                # Lloyd-Max STE: snap to nearest Gaussian-optimal centroid.
                with torch.no_grad():
                    w32 = w.float()
                    sigma = w32.std(dim=1).clamp_min(1e-8)
                    normalized = w32 / sigma[:, None]
                    lut = get_lloyd_max_lut(15, w.device)  # int4 = 15 levels
                    diffs = (normalized.unsqueeze(-1) - lut.unsqueeze(0).unsqueeze(0)).abs()
                    nearest = lut[diffs.argmin(dim=-1)]
                    w_q = (nearest * sigma[:, None]).to(x.dtype)
                w = w.to(x.dtype) + (w_q - w.to(x.dtype)).detach()
            else:
                # Uniform int6 STE: fake quantize to [-31, 31] per row.
                with torch.no_grad():
                    w32 = w.float()
                    clip_abs = torch.quantile(w32.abs(), _INT6_CLIP_Q, dim=1).clamp_min(1e-8)
                    scale = clip_abs / 31.0
                    w_clipped = torch.clamp(w32, -clip_abs[:, None], clip_abs[:, None])
                    w_q = (torch.round(w_clipped / scale[:, None]) * scale[:, None]).to(x.dtype)
                w = w.to(x.dtype) + (w_q - w.to(x.dtype)).detach()
        elif mode == "ternary":
            # Ternary STE: fake quantize to {-1, 0, 1} per group.
            w = w.bfloat16()
            g = self.group_size
            w_g = w.reshape(-1, g)
            scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
            q = (w_g / scale).round().clamp(-1, 1)
            w = w + ((q * scale).reshape(w.shape) - w).detach()
        else:
            w = w.to(x.dtype)
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)


class NormedQuantizedLinear(QuantizedLinear):
    "Ternary linear with RMSNorm on input — for output projections receiving un-normalized activations."
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.rms_norm(x, (x.size(-1),)))

class LowRankLinear(nn.Module):
    """Low-rank factored linear: W = A @ B where A is (out, rank) and B is (rank, in).
    Stores and trains only the factors — never materializes the full matrix.
    For compression: rank=64 gives ~4× fewer params than full 512×512.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 64, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        # Initialize with scaled random (matches Kaiming for the product A@B)
        scale = (2.0 / (in_features + out_features)) ** 0.5
        self.A = nn.Parameter(torch.randn(out_features, rank) * scale)
        self.B = nn.Parameter(torch.randn(rank, in_features) * scale)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        A, B = self.A, self.B
        if _QUANT_MODE == "int6" and self.training and _INT6_ACTIVE and _STE_ENABLED:
            # Int6 STE for both factors: fake quantize A and B per-row to [-31,31].
            with torch.no_grad():
                # Factor A (out, rank)
                a32 = A.float()
                clip_a = torch.quantile(a32.abs(), _INT6_CLIP_Q, dim=1).clamp_min(1e-8)
                sa = clip_a / 31.0
                a_q = (torch.round(torch.clamp(a32, -clip_a[:, None], clip_a[:, None]) / sa[:, None]) * sa[:, None]).to(x.dtype)
                # Factor B (rank, in)
                b32 = B.float()
                clip_b = torch.quantile(b32.abs(), _INT6_CLIP_Q, dim=1).clamp_min(1e-8)
                sb = clip_b / 31.0
                b_q = (torch.round(torch.clamp(b32, -clip_b[:, None], clip_b[:, None]) / sb[:, None]) * sb[:, None]).to(x.dtype)
            A = A.to(x.dtype) + (a_q - A.to(x.dtype)).detach()
            B = B.to(x.dtype) + (b_q - B.to(x.dtype)).detach()
        h = F.linear(x, B)  # (..., rank)
        out = F.linear(h, A, self.bias)  # (..., out_features)
        return out

    @property
    def weight(self):
        """Materialize full weight for compatibility."""
        return self.A @ self.B


class NormedLowRankLinear(LowRankLinear):
    "Low-rank linear with RMSNorm on input."
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.rms_norm(x, (x.size(-1),)))


class LoRALinear(nn.Module):
    """Frozen random base + trainable LoRA adapters.
    Base weights are regenerated from a seed at eval (0 bytes stored).
    Only LoRA A, B matrices are stored in the artifact.
    Effective: W = frozen_random + A @ B
    """
    def __init__(self, in_features: int, out_features: int, lora_rank: int = 16,
                 bias: bool = False, base_seed: int = 0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.base_seed = base_seed
        # Frozen random base (regenerated from seed, never stored)
        rng = torch.Generator()
        rng.manual_seed(base_seed)
        base = torch.randn(out_features, in_features, generator=rng) * (2.0 / (in_features + out_features)) ** 0.5
        self.register_buffer("base_weight", base)
        # Trainable LoRA: W_effective = base + A @ B
        self.lora_A = nn.Parameter(torch.zeros(out_features, lora_rank))
        self.lora_B = nn.Parameter(torch.randn(lora_rank, in_features) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        w = self.base_weight.to(x.dtype) + (self.lora_A @ self.lora_B).to(x.dtype)
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

    @property
    def weight(self):
        return self.base_weight + self.lora_A @ self.lora_B


class NormedLoRALinear(LoRALinear):
    "LoRA linear with RMSNorm on input."
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.rms_norm(x, (x.size(-1),)))


# Global low-rank / LoRA settings.
_LOW_RANK = 0
_LORA_RANK = 0
_LORA_BASE_SEED_COUNTER = 0

def make_linear(in_f, out_f, bias=False, group_size=64, normed=False):
    """Factory: returns LoRALinear, LowRankLinear, or QuantizedLinear based on global settings."""
    global _LORA_BASE_SEED_COUNTER
    if _LORA_RANK > 0:
        seed = _LORA_BASE_SEED_COUNTER
        _LORA_BASE_SEED_COUNTER += 1
        cls = NormedLoRALinear if normed else LoRALinear
        return cls(in_f, out_f, lora_rank=_LORA_RANK, bias=bias, base_seed=seed)
    elif _LOW_RANK > 0:
        cls = NormedLowRankLinear if normed else LowRankLinear
        return cls(in_f, out_f, rank=_LOW_RANK, bias=bias)
    else:
        cls = NormedQuantizedLinear if normed else QuantizedLinear
        return cls(in_f, out_f, bias=bias, group_size=group_size)


class GroupedQuantizedLinear(nn.Module):
    "Grouped linear with ternary STE. Weight stored as 2D [groups*group_out, group_in] for ternary quantization compatibility."
    def __init__(self, in_features, out_features, groups=4, group_size=64, normed=False):
        super().__init__()
        assert in_features % groups == 0 and out_features % groups == 0
        self.groups = groups
        self.group_in = in_features // groups
        self.group_out = out_features // groups
        self.group_size = group_size
        self.normed = normed
        self.weight = nn.Parameter(torch.randn(groups * self.group_out, self.group_in) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        if self.normed:
            x = F.rms_norm(x, (x.size(-1),))
        w = self.weight.bfloat16()
        g = self.group_size
        w_g = w.reshape(-1, g)
        scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
        q = (w_g / scale).round().clamp(-1, 1)
        w_ternary = w + ((q * scale).reshape(w.shape) - w).detach()
        w_grouped = w_ternary.reshape(self.groups, self.group_out, self.group_in)
        bsz = x.shape[:-1]
        x_g = x.reshape(*bsz, self.groups, self.group_in)
        out = torch.einsum('...gi,goi->...go', x_g, w_grouped)
        return out.reshape(*bsz, self.groups * self.group_out)

class TverskyProjection(nn.Module):
    "Tversky similarity: S = θ·f(A∩B) - α·f(A\\B) - β·f(B\\A). Three modes."
    def __init__(self, in_features: int, out_features: int, num_features: int = 16,
                 group_size: int = 64, use_shared_features: bool = False,
                 membership: str = "sigmoid"):
        super().__init__()
        self.group_size = group_size
        self.num_features = num_features
        self.membership_type = membership
        self.no_features_mode = (num_features == 0)

        if not self.no_features_mode and not use_shared_features:
            self.features = nn.Parameter(torch.empty(num_features, in_features).uniform_(-0.02, 0.02))
        else:
            self.register_parameter('features', None)

        self.prototypes = nn.Parameter(torch.empty(out_features, in_features).uniform_(-0.02, 0.02))
        self.theta = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def _ternary_ste(self, w: Tensor) -> Tensor:
        w_bf16 = w.bfloat16()
        g = self.group_size
        w_grouped = w_bf16.reshape(-1, g)
        scale = w_grouped.abs().mean(-1, keepdim=True).clamp(min=1e-8)
        q = (w_grouped / scale).round().clamp(-1, 1)
        w_ternary = w_bf16 + ((q * scale).reshape(w_bf16.shape) - w_bf16).detach()
        return w_ternary.reshape(w.shape)

    def _membership(self, t: Tensor) -> Tensor:
        if self.membership_type == "poly":
            return torch.clamp(t * 5.0 / 4.0 + 0.5, 0.0, 1.0)
        elif self.membership_type == "tanh":
            return (torch.tanh(t * 5.0) + 1.0) * 0.5
        else:
            return torch.sigmoid(t * 5.0)

    def forward(self, x: Tensor, shared_features: Tensor | None = None) -> Tensor:
        proto = self._ternary_ste(self.prototypes)

        if self.no_features_mode:
            # NoFeatures: prototypes are their own feature universe
            x_f = x @ proto.t()                          # [B, S, out]
            p_norm = F.normalize(proto, dim=-1)
            p_f = p_norm @ p_norm.t()                    # [out, out]
        else:
            feat = (shared_features if shared_features is not None else self.features).float()
            x_f = x @ feat.t()                           # [B, S, nf]
            p_f = proto @ feat.t()                       # [out, nf]

        x_s = self._membership(x_f)
        p_s = self._membership(p_f)
        x_a = x_f * x_s
        p_a = p_f * p_s

        t, a, b = self.theta.abs(), self.alpha.abs(), self.beta.abs()
        return t * (x_a @ p_a.t()) - a * (x_a @ (1 - p_s).t()) - b * ((1 - x_s) @ p_a.t())

def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CTP)) and param.dtype != torch.float32:
                param.data = param.data.float()

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, no_cache: bool = False,
                 rope_type: str = "rope", yarn_max_len: int = 4096, train_seq_len: int = 1024):
        super().__init__()
        self.no_cache = no_cache
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if rope_type == "yarn":
            scale = train_seq_len / yarn_max_len
            freq_idx = torch.arange(0, dim, 2, dtype=torch.float32)
            ramp = torch.clamp((freq_idx / dim - 0.25) / 0.75, 0.0, 1.0)
            inv_freq = inv_freq / (ramp * (1.0 / scale - 1.0) + 1.0)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len, device, dtype):
        if self.no_cache:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            return freqs.cos()[None, :, None, :].to(dtype=dtype), freqs.sin()[None, :, None, :].to(dtype=dtype)
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                 group_size=64, attn_proj_type="standard", tversky_num_features=16,
                 tversky_feature_pools=0, no_cache=False, rope_type="rope",
                 yarn_max_len=4096, train_seq_len=1024, tversky_membership="sigmoid",
                 diff_attn=False, xsa=False):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        self.diff_attn = diff_attn
        self.xsa = xsa
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.c_qkv = make_linear(dim, self.q_size + 2 * self.kv_size, bias=False, group_size=group_size)
        self.proj = make_linear(dim, dim, bias=False, group_size=group_size, normed=True) if attn_proj_type != "tversky" else None
        if self.proj is not None:
            self.proj._zero_init = True
        self.tversky_proj = TverskyProjection(
            dim, dim, num_features=tversky_num_features, group_size=group_size,
            use_shared_features=(tversky_feature_pools > 0),
            membership=tversky_membership,
        ) if attn_proj_type == "tversky" else None
        self.shared_features = None
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        if xsa:
            self.xsa_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        if diff_attn:
            self.diff_lambda = nn.Parameter(torch.full((num_heads,), 0.5, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base, no_cache=no_cache,
                             rope_type=rope_type, yarn_max_len=yarn_max_len,
                             train_seq_len=train_seq_len)

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        bsz, seqlen, dim = x.shape
        qkv_out = self.c_qkv(x)
        q_out, k_out, v_out = qkv_out.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q_out.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = k_out.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v_out.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        # FA2 requires bf16/fp16.
        attn_dtype = torch.bfloat16 if q.dtype == torch.float32 else q.dtype
        q, k, v = q.to(attn_dtype), k.to(attn_dtype), v.to(attn_dtype)
        if self.diff_attn:
            half = self.head_dim // 2
            q1, q2 = q[..., :half], q[..., half:]
            k1, k2 = k[..., :half], k[..., half:]
            v1, v2 = v[..., :half], v[..., half:]
            y1 = flash_attn_func(q1.contiguous(), k1.contiguous(), v1.contiguous(), causal=causal)
            y2 = flash_attn_func(q2.contiguous(), k2.contiguous(), v2.contiguous(), causal=causal)
            lam = self.diff_lambda.to(dtype=y1.dtype)[None, None, :, None]
            y = torch.cat([y1 - lam * y2, y1 + lam * y2], dim=-1)
        else:
            y = flash_attn_func(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                causal=causal
            )
        y = y.reshape(bsz, seqlen, dim)
        # XSA: subtract self-value component to force reliance on other tokens.
        if self.xsa:
            # v_self is the value at each position's own token.
            # With GQA, expand v to match num_heads then reshape.
            rep = self.num_heads // self.num_kv_heads
            v_exp = v_out.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            if rep > 1:
                v_exp = v_exp.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(bsz, seqlen, self.num_heads, self.head_dim)
            v_self = v_exp.reshape(bsz, seqlen, dim)
            y = y - self.xsa_gate.to(y.dtype) * v_self
        return self.tversky_proj(y, self.shared_features) if self.tversky_proj is not None else self.proj(y)

class LocalConv(nn.Module):
    """Causal depthwise conv with bottleneck projection for local n-gram patterns.

    Uses FP16 depthwise conv (continuous kernels) + ternary bottleneck projection.
    Bottleneck reduces dim → hidden → dim to save artifact space.
    """
    def __init__(self, dim: int, kernel_size: int = 7, num_layers: int = 1, group_size: int = 64,
                 bottleneck: int = 256):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "dw": nn.Conv1d(dim, dim, kernel_size, padding=kernel_size - 1, groups=dim, bias=False),
                "down": QuantizedLinear(dim, bottleneck, bias=False, group_size=group_size),
                "up": QuantizedLinear(bottleneck, dim, bias=False, group_size=group_size),
                "norm": nn.RMSNorm(dim),
            }))
            # Zero-init the up projection so conv starts as identity.
            self.layers[-1]["up"]._zero_init = True
            nn.init.zeros_(self.layers[-1]["up"].weight)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for layer in self.layers:
            # Causal conv: pad left, truncate right.
            h_conv = layer["dw"](h.transpose(1, 2))[..., :h.shape[1]].transpose(1, 2)
            h = h + layer["up"](F.silu(layer["down"](layer["norm"](h_conv))))
        return h


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult, group_size=64, activation="swiglu", mlp_groups=0):
        super().__init__()
        hidden = mlp_mult * dim
        self.activation = activation
        if mlp_groups > 0:
            if activation == "swiglu":
                self.gate_up = GroupedQuantizedLinear(dim, hidden * 2, groups=mlp_groups, group_size=group_size)
            else:
                self.fc = GroupedQuantizedLinear(dim, hidden, groups=mlp_groups, group_size=group_size)
            self.proj = GroupedQuantizedLinear(hidden, dim, groups=mlp_groups, group_size=group_size, normed=True)
        else:
            if activation == "swiglu":
                self.gate_up = make_linear(dim, hidden * 2, bias=False, group_size=group_size)
            else:
                self.fc = make_linear(dim, hidden, bias=False, group_size=group_size)
            self.proj = make_linear(hidden, dim, bias=False, group_size=group_size, normed=True)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        if self.activation == "swiglu":
            gu = self.gate_up(x)
            gate, up = gu.chunk(2, dim=-1)
            return self.proj(F.silu(gate) * up)
        elif self.activation == "relu":
            return self.proj(torch.relu(self.fc(x)))
        elif self.activation == "leaky_relu":
            return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.01))
        elif self.activation == "leaky_relu2":
            return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())
        else:  # relu2
            return self.proj(torch.relu(self.fc(x)).square())

class SmearModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        cumsum = x.cumsum(dim=1)
        counts = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        smeared = cumsum / counts
        gate = torch.tanh(self.gate.to(dtype=x.dtype))
        return x + gate * (smeared - x)


class CausalConvRefiner(nn.Module):
    "Causal Conv1d that refines hidden states using local n-gram context."
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=0, bias=False)
        self.gate = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        h = x.permute(0, 2, 1)  # [B, D, S]
        h = F.pad(h, (self.kernel_size - 1, 0))  # causal pad
        h = self.conv(h)
        h = h.permute(0, 2, 1)  # [B, S, D]
        return x + torch.tanh(self.gate.to(dtype=x.dtype)) * F.rms_norm(h, (h.size(-1),))

class CausalConvMixer(nn.Module):
    """Replaces attention with gated causal depthwise convolution.
    No attention at all — purely local pattern matching via wide causal conv.
    """
    def __init__(self, dim: int, kernel_size: int = 32, group_size: int = 64):
        super().__init__()
        # Depthwise causal conv for token mixing
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=0, groups=dim, bias=False)
        # Gate: project to 2*dim, split into value and gate
        self.proj_in = QuantizedLinear(dim, dim * 2, bias=False, group_size=group_size)
        self.proj_out = NormedQuantizedLinear(dim, dim, bias=False, group_size=group_size)
        self.proj_out._zero_init = True
        self.kernel_size = kernel_size

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        B, T, D = x.shape
        # Causal depthwise conv
        h = x.permute(0, 2, 1)  # (B, D, T)
        h = F.pad(h, (self.kernel_size - 1, 0))  # causal pad
        h = self.conv(h).permute(0, 2, 1)  # (B, T, D)
        # Gated projection
        gu = self.proj_in(h)
        gate, val = gu.chunk(2, dim=-1)
        return self.proj_out(F.silu(gate) * val)


class TokenShiftMixer(nn.Module):
    """RWKV-style token shift: learned interpolation between current and previous token.
    Very fast, no attention, captures bigram-level patterns.
    """
    def __init__(self, dim: int, group_size: int = 64):
        super().__init__()
        self.mix = nn.Parameter(torch.ones(dim) * 0.5)  # interpolation weight
        self.proj_in = QuantizedLinear(dim, dim * 2, bias=False, group_size=group_size)
        self.proj_out = NormedQuantizedLinear(dim, dim, bias=False, group_size=group_size)
        self.proj_out._zero_init = True

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        # Shift: interpolate between current and previous token
        mix = torch.sigmoid(self.mix.to(x.dtype))
        prev = F.pad(x[:, :-1], (0, 0, 1, 0))  # shift right, pad with zeros
        h = mix * x + (1 - mix) * prev
        # Gated projection
        gu = self.proj_in(h)
        gate, val = gu.chunk(2, dim=-1)
        return self.proj_out(F.silu(gate) * val)


# ---------------------------------------------------------------------------
# Novel features: MoE, Foveated Attention, Adaptive Depth
# ---------------------------------------------------------------------------

class MoEMLP(nn.Module):
    """Mixture of Experts MLP: N small expert MLPs with top-k routing.
    Total params = N × expert_params, but only top-k experts active per token.
    More capacity per artifact byte since inactive expert weights compress well."""
    def __init__(self, dim, mlp_mult, num_experts, topk=1, group_size=64, activation="swiglu"):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.experts = nn.ModuleList([
            MLP(dim, mlp_mult, group_size, activation) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        # Router: (B, T, num_experts)
        logits = self.gate(x.float())
        weights, indices = torch.topk(torch.softmax(logits, dim=-1), self.topk, dim=-1)  # (B, T, topk)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # renormalize

        # Dispatch to experts
        out = torch.zeros_like(x)
        for k in range(self.topk):
            expert_idx = indices[:, :, k]  # (B, T)
            w = weights[:, :, k].unsqueeze(-1).to(x.dtype)  # (B, T, 1)
            for e in range(self.num_experts):
                mask = (expert_idx == e)  # (B, T)
                if mask.any():
                    # Gather tokens for this expert
                    expert_input = x[mask]  # (N, D)
                    expert_out = self.experts[e](expert_input.unsqueeze(0)).squeeze(0)  # (N, D)
                    out[mask] += (w[mask] * expert_out)
        return out


class AdaptiveDepthGate(nn.Module):
    """Per-token gate that decides whether to skip this layer.
    Output: gate * layer_output + (1-gate) * input (soft skip)."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, 1, bias=True)
        nn.init.constant_(self.proj.bias, 2.0)  # bias toward using the layer initially

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.proj(x.float())).to(x.dtype)  # (B, T, 1)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, group_size: int=64,
                 activation: str="swiglu", attn_proj_type: str="standard",
                 tversky_num_features: int=16, tversky_feature_pools: int=0, no_cache: bool=False,
                 smear: bool=False, rope_type: str="rope", yarn_max_len: int=4096,
                 train_seq_len: int=1024, tversky_membership: str="sigmoid",
                 diff_attn: bool=False, mlp_groups: int=0, xsa: bool=False,
                 mixer_type: str="attention", moe_experts: int=0, moe_topk: int=1,
                 adaptive_depth: bool=False, foveated: bool=False, foveated_window: int=256):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.foveated = foveated
        self.foveated_window = foveated_window
        if mixer_type == "causal_conv":
            self.attn = CausalConvMixer(dim, kernel_size=32, group_size=group_size)
        elif mixer_type == "token_shift":
            self.attn = TokenShiftMixer(dim, group_size=group_size)
        else:
            self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                            group_size, attn_proj_type, tversky_num_features,
                                            tversky_feature_pools, no_cache, rope_type, yarn_max_len,
                                            train_seq_len, tversky_membership, diff_attn, xsa=xsa)
        if moe_experts > 0:
            self.mlp = MoEMLP(dim, mlp_mult, moe_experts, moe_topk, group_size, activation)
        else:
            self.mlp = MLP(dim, mlp_mult, group_size, activation, mlp_groups)
        self.depth_gate = AdaptiveDepthGate(dim) if adaptive_depth else None
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.smear = SmearModule(dim) if smear else None

    def forward(self, x: Tensor, x0: Tensor, causal: bool = True) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0] * x + mix[1] * x0
        n = self.attn_norm(x_in)
        # Foveated: use windowed causal attention (only attend to last W tokens)
        if self.foveated and hasattr(self.attn, 'c_qkv'):
            B, T, D = n.shape
            W = self.foveated_window
            if T > W:
                # Create windowed causal mask: each token attends to at most W previous tokens
                mask = torch.ones(T, T, dtype=torch.bool, device=n.device).tril()
                mask = mask & (torch.arange(T, device=n.device).unsqueeze(0) - torch.arange(T, device=n.device).unsqueeze(1) < W)
                float_mask = torch.where(mask, 0.0, float('-inf')).unsqueeze(0).unsqueeze(0)
                # Use SDPA path with mask (bypass flash_attn)
                qkv = self.attn.c_qkv(n)
                H, KVH = self.attn.num_heads, self.attn.num_kv_heads
                head_dim = D // H
                q = qkv[:, :, :D].reshape(B, T, H, head_dim).transpose(1, 2)
                k = qkv[:, :, D:D + KVH * head_dim].reshape(B, T, KVH, head_dim).transpose(1, 2)
                v = qkv[:, :, D + KVH * head_dim:].reshape(B, T, KVH, head_dim).transpose(1, 2)
                if KVH != H:
                    rep = H // KVH
                    k = k.unsqueeze(3).expand(B, KVH, T, rep, head_dim).reshape(B, H, T, head_dim)
                    v = v.unsqueeze(3).expand(B, KVH, T, rep, head_dim).reshape(B, H, T, head_dim)
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask)
                attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
                attn_out = self.attn.proj(attn_out)
                if hasattr(self.attn, 'xsa') and self.attn.xsa:
                    gate = torch.sigmoid(self.attn.xsa_gate.to(dtype=attn_out.dtype))
                    attn_out = gate * attn_out
                x_out = x_in + self.attn_scale.to(dtype=x_in.dtype) * attn_out
            else:
                x_out = x_in + self.attn_scale.to(dtype=x_in.dtype) * self.attn(n, causal=causal)
        else:
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype) * self.attn(n, causal=causal)
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype) * self.mlp(self.mlp_norm(x_out))
        if self.smear is not None:
            x_out = self.smear(x_out)
        # Adaptive depth: soft skip
        if self.depth_gate is not None:
            gate = self.depth_gate(x_in)
            x_out = gate * x_out + (1 - gate) * x_in
        return x_out

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult,
                 tie_embeddings, tied_embed_init_std, logit_softcap, rope_base, qk_gain_init,
                 group_size: int = 64, activation: str = "swiglu", mtp_heads_count: int = 0,
                 embed_dim: int = 0, attn_proj_type: str = "standard", logit_head_type: str = "standard",
                 tversky_num_features: int = 16, tversky_feature_pools: int = 0,
                 training_depth_recurrence: int=1, fp_storage=False, bigram_hash: bool=False,
                 softcap_type: str="poly", no_cache: bool=False,
                 smear: bool=False, rope_type: str="rope", yarn_max_len: int=4096,
                 train_seq_len: int=1024, tversky_membership: str="sigmoid",
                 diff_attn=False, mlp_groups=0, refiner=False, refiner_kernel=3,
                 n_channels=1, p_causal=0.2, mask_rate_min=0.15, mask_rate_max=0.85,
                 local_conv_layers=0, local_conv_kernel=7, local_conv_bottleneck=256,
                 ngram_orders=None, ngram_buckets=0, ngram_top_k=3,
                 ngram_num_inject_layers=0,
                 ngram_input_inject=True, ngram_logit_mix=True, bigram_logit_mix=False,
                 sse_enabled=False, sse_clusters=32, sse_entropy_bins=16,
                 context_logit_bias=False, clb_clusters=512, clb_context_len=4, clb_rank=8,
                 xsa_layers=0, mixer_type="attention",
                 ut_unique_blocks=0, ut_iters=4):
        super().__init__()
        self.training_depth_recurrence = training_depth_recurrence
        self.fp_storage = fp_storage
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.softcap_type = softcap_type
        self.n_channels = n_channels
        self.p_causal = p_causal
        self.mask_rate_min = mask_rate_min
        self.mask_rate_max = mask_rate_max
        self.embed_dim = embed_dim if embed_dim > 0 else model_dim
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        # +1 for [MASK] token used during superposition training.
        effective_vocab = vocab_size + 1 if n_channels > 1 else vocab_size
        self.mask_token_id = vocab_size
        self.tok_emb = QATEmbedding(effective_vocab, self.embed_dim, fp_storage=fp_storage)
        self.bigram_emb = QATEmbedding(vocab_size, self.embed_dim, fp_storage=fp_storage) if bigram_hash else None
        if self.bigram_emb is not None:
            nn.init.zeros_(self.bigram_emb.weight)
        self.lm_head_correction = nn.Parameter(
            torch.zeros(vocab_size, self.embed_dim)) if tie_embeddings == 2 else None
        self.embed_proj = QATLinear(self.embed_dim, model_dim, bias=False, fp_storage=fp_storage) if self.embed_dim != model_dim else None
        self.embed_proj_rev = QATLinear(model_dim, self.embed_dim, bias=False, fp_storage=fp_storage) if (
            self.embed_dim != model_dim and logit_head_type != "tversky") else None
        # Local causal conv for n-gram pattern matching.
        self.local_conv = LocalConv(model_dim, kernel_size=local_conv_kernel, num_layers=local_conv_layers,
                                    group_size=group_size, bottleneck=local_conv_bottleneck) if local_conv_layers > 0 else None
        # N-gram additive injection (input + optional per-layer residual).
        self.ngram_injector = NgramInjector(
            ngram_orders, ngram_buckets, ngram_top_k, self.embed_dim, model_dim, self.tok_emb,
            num_inject_layers=ngram_num_inject_layers,
        ) if ngram_orders and ngram_input_inject else None
        # N-gram logit mixer (output-level, ~20 scalar params).
        self.ngram_logit_mixer = NgramLogitMixer(
            ngram_orders, ngram_buckets, ngram_top_k, vocab_size,
        ) if ngram_orders and ngram_logit_mix else None
        # Exact bigram logit injection (2MB count table, 1 learned gate).
        self.bigram_logit_mixer = BigramLogitMixer(vocab_size) if bigram_logit_mix else None
        # Cascaded SSE calibration (PAQ-style).
        self.sse = CascadedSSE(vocab_size, sse_clusters, sse_entropy_bins) if sse_enabled else None
        # Context-dependent logit bias (PAQ indirect context modeling).
        self.context_logit_bias = ContextLogitBias(vocab_size, clb_clusters, clb_context_len, clb_rank) if context_logit_bias else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # CDMA superposition: channel embeddings for demultiplexing.
        if n_channels > 1:
            self.channel_in = nn.Parameter(torch.randn(n_channels, model_dim) * 0.02)
            self.channel_out = nn.Parameter(torch.randn(n_channels, model_dim) * 0.02)

        # Shared Tversky feature pools (if enabled and num_features > 0)
        if attn_proj_type == "tversky" and tversky_feature_pools > 0 and tversky_num_features > 0:
            self.tversky_feature_pools_list = nn.ParameterList([
                nn.Parameter(torch.empty(tversky_num_features, model_dim).uniform_(-0.02, 0.02))
                for _ in range(tversky_feature_pools)
            ])
        else:
            self.tversky_feature_pools_list = None

        # Novel feature flags (set via _set_novel_features before construction)
        if not hasattr(self, '_moe_experts'):
            self._moe_experts = 0
        if not hasattr(self, '_moe_topk'):
            self._moe_topk = 1
        if not hasattr(self, '_adaptive_depth'):
            self._adaptive_depth = False
        if not hasattr(self, '_foveated_layers'):
            self._foveated_layers = 0
        if not hasattr(self, '_foveated_window'):
            self._foveated_window = 256
        # Universal Transformer: create fewer unique blocks, iterate them.
        self.ut_iters = ut_iters if ut_unique_blocks > 0 else 1
        actual_blocks = ut_unique_blocks if ut_unique_blocks > 0 else num_layers
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  group_size, activation, attn_proj_type, tversky_num_features, tversky_feature_pools,
                  no_cache, smear, rope_type, yarn_max_len, train_seq_len, tversky_membership,
                  diff_attn, mlp_groups, xsa=(i >= actual_blocks - xsa_layers),
                  mixer_type=mixer_type,
                  moe_experts=self._moe_experts, moe_topk=self._moe_topk,
                  adaptive_depth=self._adaptive_depth,
                  foveated=(i >= actual_blocks - self._foveated_layers),
                  foveated_window=self._foveated_window)
            for i in range(actual_blocks)
        ])

        # Inject shared feature pool references into attention layers
        if self.tversky_feature_pools_list is not None:
            for i, block in enumerate(self.blocks):
                pool_idx = (i * tversky_feature_pools) // num_layers
                block.attn.shared_features = self.tversky_feature_pools_list[pool_idx]

        self.final_norm = RMSNorm()
        self.refiner = CausalConvRefiner(model_dim, kernel_size=refiner_kernel) if refiner else None
        self.mtp_heads = nn.ModuleList([
            nn.Linear(model_dim, vocab_size, bias=False) for _ in range(mtp_heads_count)
        ])
        for h in self.mtp_heads:
            nn.init.zeros_(h.weight)
        self.logit_head_type = logit_head_type
        if logit_head_type == "tversky" and tversky_num_features == 0 and vocab_size > 1024:
            raise ValueError(
                f"Tversky logit head with no-features mode creates O(V^2) = {vocab_size}x{vocab_size} "
                f"matrix per forward pass. Use tversky_num_features > 0 or a smaller vocab."
            )
        self.tversky_head = TverskyProjection(
            model_dim, vocab_size, num_features=tversky_num_features,
            membership=tversky_membership,
        ) if logit_head_type == "tversky" else None
        self.lm_head = QATLinear(model_dim, vocab_size, bias=False, fp_storage=fp_storage)
        self.lm_head._zero_init = True
        if self.lm_head is not None and (tie_embeddings or logit_head_type == "tversky"):
            self.lm_head.weight.requires_grad_(False)

        self.vocab_bias = nn.Parameter(torch.zeros(vocab_size, dtype=torch.float32))
        self._init_weights(tied_embed_init_std)

    def _init_weights(self, tied_embed_init_std: float) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, QuantizedLinear) and not getattr(module, "_zero_init", False):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _compute_logits(self, x: Tensor, input_ids: Tensor | None = None) -> Tensor:
        if self.tversky_head is not None:
            logits_raw = self.tversky_head(x)
        elif self.tie_embeddings:
            if self.embed_proj_rev is not None:
                proj = self.embed_proj_rev(x)
            else:
                proj = x
            weight = self.tok_emb.weight[:self.vocab_size]
            if self.lm_head_correction is not None:
                weight = weight + self.lm_head_correction
            logits_raw = F.linear(proj, weight.to(x.dtype))
        else:
            logits_raw = self.lm_head(x)
        logits = logits_raw + self.vocab_bias.to(x.dtype)
        # N-gram logit mixing: add learned biases for n-gram-predicted tokens.
        if self.ngram_logit_mixer is not None and input_ids is not None:
            was_flat = logits.dim() == 2
            if was_flat:
                B, T = input_ids.shape
                logits = logits.view(B, T, -1)
            logits = self.ngram_logit_mixer(input_ids, logits)
            if was_flat:
                logits = logits.view(-1, logits.size(-1))
        # Exact bigram logit injection: add smoothed bigram log-probs.
        if self.bigram_logit_mixer is not None and input_ids is not None:
            was_flat = logits.dim() == 2
            if was_flat:
                B, T = input_ids.shape
                logits = logits.view(B, T, -1)
            logits = self.bigram_logit_mixer(input_ids, logits)
            if was_flat:
                logits = logits.view(-1, logits.size(-1))
        # Context-dependent logit bias (indirect context model).
        if self.context_logit_bias is not None and input_ids is not None:
            was_flat = logits.dim() == 2
            if was_flat:
                B, T = input_ids.shape
                logits = logits.view(B, T, -1)
            logits = self.context_logit_bias(input_ids, logits)
            if was_flat:
                logits = logits.view(-1, logits.size(-1))
        return logits

    def _softcap(self, logits: Tensor) -> Tensor:
        s = self.logit_softcap
        if self.softcap_type == "tanh":
            return s * torch.tanh(logits / s)
        x_sc = torch.clamp(logits / s, -2.0, 2.0)
        x2 = x_sc * x_sc
        return s * torch.clamp(x_sc * (1.0 - x2 / 3.0 + x2 * x2 / 15.0), -1.0, 1.0)

    def _run_blocks(self, x: Tensor, x0: Tensor, causal: bool = True) -> Tensor:
        """Run blocks. Supports U-Net (standard) and Universal Transformer (weight-shared iteration)."""
        ngram = self.ngram_injector
        total_blocks = len(self.blocks)

        if self.ut_iters > 1:
            # Universal Transformer: iterate each block ut_iters times.
            for block_idx in range(total_blocks):
                for _ in range(self.ut_iters):
                    x = self.blocks[block_idx](x, x0, causal=causal)
            x = self.final_norm(x)
            if self.refiner is not None:
                x = self.refiner(x)
            return x

        # Standard U-Net path with skip connections.
        active = getattr(self, "_active_layers", total_blocks)
        if active >= total_blocks:
            enc_layers = self.num_encoder_layers
            dec_layers = self.num_decoder_layers
        else:
            enc_layers = active // 2
            dec_layers = active - enc_layers
        skips = []
        for i in range(enc_layers):
            for _ in range(max(1, self.training_depth_recurrence)):
                x = self.blocks[i](x, x0, causal=causal)
            if ngram is not None:
                x = ngram.inject_layer(x, i)
            skips.append(x)
        for i in range(dec_layers):
            bi = enc_layers + i
            if i < len(skips) and skips:
                x = x + self.skip_weights[min(i, self.num_skip_weights - 1)].to(dtype=x.dtype) * skips.pop()
            for _ in range(max(1, self.training_depth_recurrence)):
                x = self.blocks[bi](x, x0, causal=causal)
            if ngram is not None:
                x = ngram.inject_layer(x, bi)
        x = self.final_norm(x)
        if self.refiner is not None:
            x = self.refiner(x)
        return x

    def _ce_loss(self, x: Tensor, targets: Tensor, reduction: str = "mean", temperature: float = 1.0,
                 input_ids: Tensor | None = None) -> Tensor:
        """Compute cross-entropy + Z-loss from hidden states."""
        logits = self._softcap(self._compute_logits(x, input_ids=input_ids))
        # Cascaded SSE calibration.
        if self.sse is not None:
            logits = self.sse(logits, input_ids=input_ids)
        if temperature != 1.0:
            logits = logits / temperature
        if reduction == "none":
            return F.cross_entropy(logits.float(), targets, reduction="none")
        logits_f = logits.float()
        lse = torch.logsumexp(logits_f, dim=-1)
        target_logits = logits_f.gather(1, targets.unsqueeze(1)).squeeze(1)
        return (lse - target_logits).mean() + 1e-4 * (lse ** 2).mean()

    def _embed(self, input_ids: Tensor) -> Tensor:
        """Token embedding + bigram + n-gram injection + projection + local conv + RMSNorm."""
        x = self.tok_emb(input_ids)
        if x.dtype not in (torch.bfloat16, torch.float16):
            x = x.float()
        if self.bigram_emb is not None:
            prev = F.pad(input_ids[:, :-1], (1, 0), value=0)
            x = x + self.bigram_emb(prev).to(x.dtype)
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        if self.ngram_injector is not None:
            # Compute n-gram stat embedding once, add to token embedding.
            stat_emb = self.ngram_injector.compute_stat_emb(input_ids)
            x = self.ngram_injector.inject_input(x, stat_emb)
        if self.local_conv is not None:
            x = self.local_conv(x)
        return F.rms_norm(x, (x.size(-1),))

    def forward(self, input_ids: Tensor, target_ids: Tensor, reduction: str = "mean", temperature: float = 1.0) -> Tensor:
        B, T = input_ids.shape
        C = self.n_channels

        # --- Eval: always causal, single channel ---
        if not self.training or C <= 1:
            x = self._embed(input_ids)
            if C > 1:
                x = x + self.channel_in[0]
            x0 = x
            x = self._run_blocks(x, x0, causal=True)
            x_flat = x.reshape(-1, x.size(-1))
            if C > 1:
                x_flat = x_flat + self.channel_out[0]
            targets = target_ids.reshape(-1)
            loss = self._ce_loss(x_flat, targets, reduction=reduction, temperature=temperature,
                                 input_ids=input_ids.reshape(-1, T) if (self.ngram_logit_mixer is not None or self.bigram_logit_mixer is not None or self.context_logit_bias is not None) else None)

            # MTP auxiliary loss (training only, causal batches)
            if self.training and len(self.mtp_heads) > 0:
                mtp_loss = torch.zeros((), device=loss.device)
                x_normed = x
                for k, head in enumerate(self.mtp_heads):
                    shift = k + 2
                    if target_ids.shape[1] > shift:
                        mtp_tgt = target_ids[:, shift:].reshape(-1)
                        mtp_in = x_normed[:, :target_ids.shape[1] - shift, :].reshape(-1, x_normed.shape[-1])
                        mtp_loss = mtp_loss + F.cross_entropy(head(mtp_in).float(), mtp_tgt, reduction="mean")
                loss = loss + 0.1 * mtp_loss / len(self.mtp_heads)
            return loss

        # --- Training with CDMA superposition ---
        # _force_causal is set by the training loop to avoid graph breaks in torch.compile.
        use_causal = getattr(self, "_force_causal", True)
        if use_causal:
            # Standard causal AR training (single channel).
            x = self._embed(input_ids)
            x = x + self.channel_in[0]
            x0 = x
            x = self._run_blocks(x, x0, causal=True)
            x_flat = x.reshape(-1, x.size(-1)) + self.channel_out[0]
            return self._ce_loss(x_flat, target_ids.reshape(-1),
                                 input_ids=input_ids if (self.ngram_logit_mixer is not None or self.bigram_logit_mixer is not None or self.context_logit_bias is not None) else None)
        else:
            # Masked bidirectional + superposition.
            mask_rate = getattr(self, "_mask_rate", 0.5)
            group_size = B // C
            if group_size == 0:
                group_size, C = B, 1

            superimposed = torch.zeros(group_size, T, self.model_dim, device=input_ids.device, dtype=torch.bfloat16)
            all_masks, all_targets = [], []
            for c in range(C):
                ids_c = input_ids[c * group_size : (c + 1) * group_size]
                tgt_c = target_ids[c * group_size : (c + 1) * group_size]
                mask_c = torch.rand(group_size, T, device=input_ids.device) < mask_rate
                masked_ids_c = ids_c.clone()
                masked_ids_c[mask_c] = self.mask_token_id
                emb_c = self._embed(masked_ids_c)
                superimposed = superimposed + emb_c + self.channel_in[c]
                all_masks.append(mask_c)
                all_targets.append(tgt_c)

            x0 = superimposed
            x = self._run_blocks(superimposed, x0, causal=False)

            # Demultiplex: predict each channel's masked tokens.
            total_loss = torch.zeros((), device=x.device)
            n_valid = 0
            for c in range(C):
                x_c = x + self.channel_out[c]
                masked_x_c = x_c[all_masks[c]]
                if masked_x_c.numel() == 0:
                    continue
                logits_c = self._softcap(self._compute_logits(masked_x_c))
                logits_f = logits_c.float()
                lse = torch.logsumexp(logits_f, dim=-1)
                target_logits = logits_f.gather(1, all_targets[c][all_masks[c]].unsqueeze(1)).squeeze(1)
                total_loss = total_loss + (lse - target_logits).mean() + 1e-4 * (lse ** 2).mean()
                n_valid += 1
            return total_loss / max(n_valid, 1)

# ---------------------------------------------------------------------------
# Validation
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
    if max_tok > 0: tok = tok[:max_tok + 1]
    u = ((tok.numel() - 1) // seq_len) * seq_len
    return tok[:u + 1]

def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, temperature: float = 1.0):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = max(1, local_batch_tokens // args.train_seq_len)
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_start in range(seq_start, seq_end, local_batch_seqs):
            batch_end = min(batch_start + local_batch_seqs, seq_end)
            raw_start = batch_start * args.train_seq_len
            raw_end = batch_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_loss = model(x, y, temperature=temperature).detach()
            n = float(y.numel())
            loss_sum += batch_loss.to(torch.float64) * n
            token_count += n
            prev_ids, tgt_ids = x.reshape(-1), y.reshape(-1)
            tok_bytes = base_bytes_lut[tgt_ids].to(torch.int16)
            tok_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
            byte_count += tok_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = loss_sum / token_count
    bpb = (val_loss.item() / math.log(2.0)) * (token_count.item() / byte_count.item())
    model.train()
    return float(val_loss.item()), float(bpb)

def eval_val_sliding(args, model, rank, world_size, device, grad_accum_steps, val_tokens,
                     base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     stride: int = 64, temperature: float = 1.0):
    seq_len = args.train_seq_len
    batch_size = args.sliding_batch_size
    total_tokens = val_tokens.numel() - 1
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    all_starts = list(range(0, total_tokens - seq_len, stride))
    my_starts = all_starts[rank::world_size]

    model.eval()
    with torch.inference_mode():
        for i in range(0, len(my_starts), batch_size):
            batch_starts = my_starts[i:i + batch_size]

            starts_t = torch.tensor(batch_starts, dtype=torch.int64)
            offsets = torch.arange(seq_len + 1, dtype=torch.int64)
            indices = starts_t.unsqueeze(1) + offsets.unsqueeze(0)

            local_batch = val_tokens[indices].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local_batch[:, :-1]
            y = local_batch[:, 1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                per_token_loss = model(x, y, reduction="none", temperature=temperature).detach()

            for b, start in enumerate(batch_starts):
                score_from = 0 if start == 0 else seq_len - stride
                scored = per_token_loss[b, score_from:]
                sx, sy = x[b, score_from:], y[b, score_from:]

                loss_sum += scored.to(torch.float64).sum()
                token_count += scored.numel()

                tok_bytes = base_bytes_lut[sy].to(torch.int16)
                tok_bytes += (has_leading_space_lut[sy] & ~is_boundary_token_lut[sx]).to(torch.int16)
                byte_count += tok_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss = loss_sum / token_count
    bpb = (val_loss.item() / math.log(2.0)) * (token_count.item() / byte_count.item())
    model.train()
    return float(val_loss.item()), float(bpb)

# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
def find_temp(args, base_model, rank, world_size, device, grad_accum_steps,
                              calibration_tokens, base_bytes_lut, has_leading_space_lut,
                              is_boundary_token_lut):
    best_t, best_loss = 1.0, float("inf")
    for t in [0.90, 0.95, 1.00, 1.05, 1.10]:
        loss, _ = eval_val(args, base_model, rank, world_size, device, grad_accum_steps,
                           calibration_tokens, base_bytes_lut, has_leading_space_lut,
                           is_boundary_token_lut, temperature=t)
        if loss < best_loss:
            best_loss = loss
            best_t = t
    return best_t

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def set_mixed_ste(model, ternary_blocks: list[int] | None = None, int6_blocks: list[int] | None = None):
    """Set per-block STE modes for mixed-precision training.
    Blocks in ternary_blocks get ternary STE, int6_blocks get int6 STE, others get no STE.
    """
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            block_idx = -1
            for i in range(100):
                if f"blocks.{i}." in name:
                    block_idx = i
                    break
            if ternary_blocks and block_idx in ternary_blocks:
                module.ste_override = "ternary"
            elif int6_blocks and block_idx in int6_blocks:
                module.ste_override = "int6"
            else:
                module.ste_override = "none"
    # Count
    counts = {"ternary": 0, "int6": 0, "none": 0}
    for m in model.modules():
        if isinstance(m, QuantizedLinear) and m.ste_override:
            counts[m.ste_override] = counts.get(m.ste_override, 0) + 1
    return counts


def main(return_model: bool = False, boost_model: "GPT | None" = None) -> "GPT | None":
    args = Hyperparameters()
    # Re-read mutable env vars (may change between calls in ensemble mode).
    args.seed = int(os.environ.get("SEED", str(args.seed)))
    args.activation_type = os.environ.get("ACTIVATION", args.activation_type)
    args.num_layers = int(os.environ.get("NUM_LAYERS", str(args.num_layers)))
    args.model_dim = int(os.environ.get("MODEL_DIM", str(args.model_dim)))
    args.num_heads = int(os.environ.get("NUM_HEADS", str(args.num_heads)))
    args.num_kv_heads = int(os.environ.get("NUM_KV_HEADS", str(args.num_kv_heads)))
    args.mlp_mult = int(os.environ.get("MLP_MULT", str(args.mlp_mult)))
    args.iterations = int(os.environ.get("ITERATIONS", str(args.iterations)))
    args.run_id = os.environ.get("RUN_ID", args.run_id)
    args.mixer_type = os.environ.get("MIXER_TYPE", args.mixer_type)
    args.fedavg_every = int(os.environ.get("FEDAVG_EVERY", str(args.fedavg_every)))
    args.ut_unique_blocks = int(os.environ.get("UT_UNIQUE_BLOCKS", str(args.ut_unique_blocks)))
    args.ut_iters = int(os.environ.get("UT_ITERS", str(args.ut_iters)))
    args.low_rank = int(os.environ.get("LOW_RANK", str(args.low_rank)))
    args.lora_rank = int(os.environ.get("LORA_RANK", str(args.lora_rank)))
    global _LOW_RANK, _LORA_RANK, _LORA_BASE_SEED_COUNTER
    _LOW_RANK = args.low_rank
    _LORA_RANK = args.lora_rank
    _LORA_BASE_SEED_COUNTER = 0
    code = Path(__file__).read_text(encoding="utf-8")

    if args.matrix_optimizer != "adamw":
        global ns_orth
        ns_orth = torch.compile(ns_orth)

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
    if master_process:
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)

    log0(f"Python {sys.version}", console=False)
    log0(f"PyTorch {torch.__version__}", console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = ld_val(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_luts(
        sp, args.vocab_size, device)

    # --- Model ---
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        group_size=args.bitnet_group_size, activation=args.activation_type, mtp_heads_count=args.mtp_heads_count,
        embed_dim=args.embed_dim, attn_proj_type=args.attn_proj_type, logit_head_type=args.logit_head_type,
        tversky_num_features=args.tversky_num_features, tversky_feature_pools=args.tversky_feature_pools,
        training_depth_recurrence=args.training_depth_recurrence, fp_storage=args.fp_storage,
        bigram_hash=args.bigram_hash, softcap_type=args.softcap_type, no_cache=(args.compile_mode == "reduce-overhead"),
        smear=args.smear, rope_type=args.rope_type, yarn_max_len=args.yarn_max_len, train_seq_len=args.train_seq_len,
        tversky_membership=args.tversky_membership, diff_attn=args.diff_attn,
        refiner=args.refiner, refiner_kernel=args.refiner_kernel, mlp_groups=args.mlp_groups,
        n_channels=args.n_channels, p_causal=args.p_causal,
        mask_rate_min=args.mask_rate_min, mask_rate_max=args.mask_rate_max,
        local_conv_layers=args.local_conv_layers, local_conv_kernel=args.local_conv_kernel,
        local_conv_bottleneck=args.local_conv_bottleneck,
        ngram_orders=list(range(args.ngram_min_order, args.ngram_max_order + 1)) if args.ngram_enabled else None,
        ngram_buckets=args.ngram_buckets, ngram_top_k=args.ngram_top_k,
        ngram_num_inject_layers=args.ngram_inject_layers if args.ngram_enabled else 0,
        ngram_input_inject=args.ngram_input_inject if args.ngram_enabled else False,
        ngram_logit_mix=args.ngram_logit_mix if args.ngram_enabled else False,
        bigram_logit_mix=args.bigram_logit_mix,
        sse_enabled=args.sse_enabled, sse_clusters=args.sse_clusters, sse_entropy_bins=args.sse_entropy_bins,
        context_logit_bias=args.context_logit_bias, clb_clusters=args.clb_clusters,
        clb_context_len=args.clb_context_len, clb_rank=args.clb_rank,
        xsa_layers=args.xsa_layers, mixer_type=args.mixer_type,
        ut_unique_blocks=args.ut_unique_blocks, ut_iters=args.ut_iters,
    )
    # Set novel feature flags BEFORE .to(device) triggers parameter init
    base_model._moe_experts = args.moe_experts
    base_model._moe_topk = args.moe_topk
    base_model._adaptive_depth = args.adaptive_depth
    base_model._foveated_layers = args.foveated_layers
    base_model._foveated_window = args.foveated_window
    # Now rebuild blocks with novel features if any are enabled
    if args.moe_experts > 0 or args.adaptive_depth or args.foveated_layers > 0:
        actual_blocks = args.ut_unique_blocks if args.ut_unique_blocks > 0 else args.num_layers
        base_model.blocks = nn.ModuleList([
            Block(args.model_dim, args.num_heads, args.num_kv_heads, args.mlp_mult,
                  args.rope_base, args.qk_gain_init, args.bitnet_group_size, args.activation_type,
                  args.attn_proj_type, args.tversky_num_features, args.tversky_feature_pools,
                  (args.compile_mode == "reduce-overhead"), args.smear, args.rope_type,
                  args.yarn_max_len, args.train_seq_len, args.tversky_membership,
                  args.diff_attn, args.mlp_groups, xsa=(i >= actual_blocks - args.xsa_layers),
                  mixer_type=args.mixer_type,
                  moe_experts=args.moe_experts, moe_topk=args.moe_topk,
                  adaptive_depth=args.adaptive_depth,
                  foveated=(i >= actual_blocks - args.foveated_layers),
                  foveated_window=args.foveated_window)
            for i in range(actual_blocks)
        ])
    base_model = base_model.to(device).bfloat16()

    # Load precomputed n-gram tables if available.
    if args.ngram_enabled and args.ngram_table_path and (base_model.ngram_injector is not None or base_model.ngram_logit_mixer is not None):
        ngram_data = torch.load(args.ngram_table_path, map_location="cpu", weights_only=False)
        order_key = "ngram_orders" if "ngram_orders" in ngram_data else "orders"
        tables = {o: ngram_data[f"order_{o}"] for o in ngram_data[order_key] if f"order_{o}" in ngram_data}
        if base_model.ngram_injector is not None:
            base_model.ngram_injector.load_tables(tables)
        if base_model.ngram_logit_mixer is not None:
            base_model.ngram_logit_mixer.load_tables(tables)
        log0(f"loaded n-gram tables from {args.ngram_table_path}: {list(tables.keys())}")

    # Build exact bigram table from training shards.
    if base_model.bigram_logit_mixer is not None:
        t_bigram = time.perf_counter()
        n_bigram = base_model.bigram_logit_mixer.build_from_shards(args.train_files)
        elapsed_bigram = time.perf_counter() - t_bigram
        log0(f"bigram_table: built from {n_bigram:,} tokens in {elapsed_bigram:.1f}s")

    for module in base_model.modules():
        if isinstance(module, (nn.Linear, LowRankLinear, LoRALinear)):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    if base_model.lm_head is not None and (args.tie_embeddings or args.logit_head_type == "tversky"):
        base_model.lm_head.weight.requires_grad_(False)

    # Load pre-trained weights for resume/fine-tuning.
    load_weights_path = os.environ.get("LOAD_WEIGHTS", "")
    if load_weights_path:
        ckpt = torch.load(load_weights_path, map_location="cpu", weights_only=False)
        model_sd = base_model.state_dict()
        loaded = {k: v.to(dtype=model_sd[k].dtype) for k, v in ckpt.items() if k in model_sd}
        base_model.load_state_dict(loaded, strict=False)
        log0(f"loaded weights from {load_weights_path} ({len(loaded)}/{len(model_sd)} keys)")

    # Mixed-precision STE: TERNARY_BLOCKS=2,3,4,...,10 INTX_BLOCKS=0,1,11,12
    ternary_blocks_str = os.environ.get("TERNARY_BLOCKS", "")
    intx_blocks_str = os.environ.get("INTX_BLOCKS", "")
    if ternary_blocks_str:
        ternary_blocks = [int(x) for x in ternary_blocks_str.split(",")]
        intx_blocks = [int(x) for x in intx_blocks_str.split(",")] if intx_blocks_str else []
        counts = set_mixed_ste(base_model, ternary_blocks=ternary_blocks, int6_blocks=intx_blocks)
        log0(f"mixed_ste: ternary={counts.get('ternary',0)} int6={counts.get('int6',0)} none={counts.get('none',0)}")

    torch._dynamo.config.optimize_ddp = False

    if args.compile_mode == "off":
        compiled_model = base_model
    else:
        compiled_model = torch.compile(base_model, mode=args.compile_mode if args.compile_mode != "default" else None)
    use_find_unused = True  # Always on: XSA, superposition, etc. create varying grad paths
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False,
                find_unused_parameters=use_find_unused,
                static_graph=not use_find_unused,
                gradient_as_bucket_view=True) if distributed else compiled_model

    # --- Optimizers ---
    _excl = {"tok_emb.weight", "lm_head.weight", "lm_head_correction"}
    all_other_params = [(n, p) for n, p in base_model.named_parameters()
                        if not any(eh in n for eh in _excl)]
    matrix_params = [p for n, p in all_other_params
                     if p.ndim == 2 and not any(pat in n for pat in CTP)]
    scalar_params = [p for n, p in all_other_params
                     if p.ndim < 2 or any(pat in n for pat in CTP)]

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    opt_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    if args.matrix_optimizer == "adamw":
        opt_muon = torch.optim.AdamW(
            [{"params": matrix_params, "lr": args.adam_lr, "base_lr": args.adam_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    else:
        opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                        backend_steps=args.muon_backend_steps, wd=args.muon_wd)
    for g in opt_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_head = torch.optim.Adam(
        [{"params": [base_model.lm_head.weight], "lr": 0.0, "base_lr": 0.0}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)

    optimizers = [opt for opt in [opt_tok, opt_muon, opt_scalar, opt_head] if opt is not None]

    if base_model.lm_head_correction is not None:
        opt_corr = torch.optim.Adam(
            [{"params": [base_model.lm_head_correction],
              "lr": args.corr_weight_lr, "base_lr": args.corr_weight_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        optimizers.append(opt_corr)

    # --- Log all hyperparameters ---
    log0("--- Hyperparameters ---", console=False)
    log0(" ".join(f"{a}={getattr(args,a)}" for a in sorted(dir(args)) if not a.startswith("_") and a not in ("train_files","val_files") and not callable(getattr(args,a))), console=False)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"params:{n_params} L:{args.num_layers} d:{args.model_dim} h:{args.num_heads} kv:{args.num_kv_heads} ws:{world_size} ga:{grad_accum_steps} s:{args.seed}")
    # Set global quantization mode.
    global _QUANT_MODE, _INT6_CLIP_Q, _INT6_ACTIVE, _STE_ENABLED, _STE_TYPE
    _QUANT_MODE = args.quant_mode
    _INT6_CLIP_Q = args.int6_clip_q
    _INT6_ACTIVE = (args.late_qat_frac <= 0.0)  # if no late QAT, always active
    _STE_ENABLED = os.environ.get("STE_ENABLED", "1") != "0"
    _STE_TYPE = os.environ.get("STE_TYPE", "uniform")  # "uniform" or "lloyd_max"
    log0(f"quant_mode:{args.quant_mode} ste:{_STE_ENABLED} ste_type:{_STE_TYPE}")
    if args.n_channels > 1:
        log0(f"superposition: channels:{args.n_channels} p_causal:{args.p_causal} mask:[{args.mask_rate_min},{args.mask_rate_max}]")
    if args.local_conv_layers > 0:
        log0(f"local_conv: layers:{args.local_conv_layers} kernel:{args.local_conv_kernel}")
    if args.ema_enabled:
        log0(f"ema: decay:{args.ema_decay} start_frac:{args.ema_start_frac}")

    # --- Data loader & helpers ---
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float):
        if args.lr_schedule in ("1cycle", "multicycle"):
            if max_wallclock_ms is not None:
                frac = elapsed_ms / max_wallclock_ms
            else:
                frac = step / max(args.iterations, 1)
            frac = min(frac, 1.0)
            min_mul = 1.0 / args.onecycle_min_div

            if args.lr_schedule == "multicycle":
                # Multi-cycle: N cosine annealing cycles with warm restarts.
                # Each cycle: quick ramp up (10% of cycle), cosine down (90%).
                # Cycles get progressively shorter (1/N, 1/N, ...) for simplicity.
                n = args.num_cycles
                cycle_frac = frac * n  # which cycle are we in?
                cycle_frac = cycle_frac - int(cycle_frac)  # position within current cycle [0, 1)
                ramp = 0.1  # 10% of each cycle is ramp-up
                if cycle_frac < ramp:
                    return min_mul + (1.0 - min_mul) * (cycle_frac / ramp)
                else:
                    t = (cycle_frac - ramp) / (1.0 - ramp)
                    return min_mul + 0.5 * (1.0 - min_mul) * (1.0 + math.cos(math.pi * t))
            else:
                # 1cycle: ramp up to peak, then cosine anneal down
                peak = args.onecycle_peak_frac
                if frac < peak:
                    return min_mul + (1.0 - min_mul) * (frac / peak)
                else:
                    t = (frac - peak) / (1.0 - peak)
                    return min_mul + 0.5 * (1.0 - min_mul) * (1.0 + math.cos(math.pi * t))
        # Default: warmdown schedule
        if args.warmdown_fraction <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = int(args.iterations * (1.0 - args.warmdown_fraction))
            return max((args.iterations - step) / max(args.iterations * args.warmdown_fraction, 1), 0.0) if step >= warmdown_start else 1.0
        warmdown_ms = max_wallclock_ms * args.warmdown_fraction
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    _seq_switched = False
    _batch_switched = False
    active_seq_len = args.seq_len_start if args.seq_len_start > 0 else args.train_seq_len
    active_batch_tokens = args.batch_tokens_start if args.batch_tokens_start > 0 else args.train_batch_tokens

    # --- Compiler warmup ---
    if args.warmup_steps > 0:
        _ms = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        _os = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for mi in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = mi == grad_accum_steps - 1
                x, y = train_loader.next_batch(active_batch_tokens, active_seq_len, grad_accum_steps)
                # Set superposition mode before forward (avoids graph break in torch.compile).
                base_model._force_causal = (random.random() < args.p_causal) or args.n_channels <= 1
                base_model._mask_rate = args.mask_rate_min + random.random() * (args.mask_rate_max - args.mask_rate_min)
                torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(x, y)
                (loss * grad_scale).backward()
            for o in optimizers: o.step()
            zero_grad_all()
            log0(f"warmup:{ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(_ms, strict=True)
        for o, s in zip(optimizers, _os): o.load_state_dict(s)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # --- EMA setup ---
    ema_state: dict[str, Tensor] | None = None
    if args.ema_enabled:
        ema_start_step = int(args.iterations * args.ema_start_frac)
    else:
        ema_start_step = args.iterations + 1  # never

    # --- N-gram table preprocessing (within the 10-min window, before training) ---
    # Skip if tables were already loaded from a precomputed file.
    ngram_builder: NgramTableBuilder | None = None
    if args.ngram_enabled and not args.ngram_table_path:
        ngram_orders = list(range(args.ngram_min_order, args.ngram_max_order + 1))
        ngram_builder = NgramTableBuilder(ngram_orders, args.ngram_buckets, args.ngram_top_k, args.vocab_size)
        est_bytes = ngram_builder.artifact_size_estimate()
        log0(f"ngram_preprocess: orders:{ngram_orders} buckets:{args.ngram_buckets} top_k:{args.ngram_top_k} est:{est_bytes/1e6:.1f}MB")

        ngram_stream = TokenStream(args.train_files)
        ngram_chunk_size = 2_000_000
        ngram_check_interval = 50_000_000
        ngram_t0 = time.perf_counter()
        tokens_since_check = 0
        while not ngram_builder.converged:
            try:
                chunk = ngram_stream.take(ngram_chunk_size)
            except Exception:
                break
            ngram_builder.update(chunk)
            tokens_since_check += len(chunk)
            if tokens_since_check >= ngram_check_interval:
                stable = ngram_builder.check_convergence()
                elapsed = time.perf_counter() - ngram_t0
                rate = ngram_builder.tokens_seen / elapsed / 1e6
                log0(f"ngram_preprocess: {ngram_builder.tokens_seen/1e6:.0f}M tokens, "
                     f"stable:{stable:.4f}, {elapsed:.0f}s, {rate:.1f}M tok/s")
                tokens_since_check = 0
                if stable >= args.ngram_converge_threshold:
                    ngram_builder.converged = True
                    log0(f"ngram_converged at {ngram_builder.tokens_seen/1e6:.0f}M tokens ({stable:.4f} stable)")

        ngram_elapsed = time.perf_counter() - ngram_t0
        log0(f"ngram_preprocess_done: {ngram_builder.tokens_seen/1e6:.0f}M tokens in {ngram_elapsed:.0f}s")
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    elif args.ngram_enabled and args.ngram_table_path:
        log0(f"ngram: using precomputed tables from {args.ngram_table_path}, skipping preprocessing")

    # --- Main training loop ---
    training_time_ms = 0.0
    stop_after_step: int | None = None
    _untied = False
    train_loss = torch.zeros((), device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                         val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            tstats = tern_stats(base_model, group_size=args.bitnet_group_size)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms zero_frac:{tstats['zero_frac']:.3f}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # Late QAT: activate int6 STE after a fraction of training.
        if args.quant_mode == "int6" and args.late_qat_frac > 0:
            _INT6_ACTIVE = (scale < args.late_qat_frac)

        # Sequence length schedule
        if args.seq_len_start > 0 and not _seq_switched:
            if max_wallclock_ms is not None:
                should_switch_seq = elapsed_ms >= args.seq_schedule_fraction * max_wallclock_ms
            else:
                should_switch_seq = step >= int(args.iterations * args.seq_schedule_fraction)
            if should_switch_seq:
                active_seq_len = args.train_seq_len
                _seq_switched = True
                torch._dynamo.reset()
                train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
                log0(f"step:{step} seq_len_switch:{args.seq_len_start}->{active_seq_len}")

        # Batch size schedule
        if args.batch_tokens_start > 0 and not _batch_switched:
            if max_wallclock_ms is not None:
                should_switch_batch = elapsed_ms >= args.batch_schedule_fraction * max_wallclock_ms
            else:
                should_switch_batch = step >= int(args.iterations * args.batch_schedule_fraction)
            if should_switch_batch:
                active_batch_tokens = args.train_batch_tokens
                _batch_switched = True
                log0(f"step:{step} batch_switch:{args.batch_tokens_start}->{active_batch_tokens}")

        zero_grad_all()
        train_loss.zero_()

        # Set superposition mode and stochastic recurrence before micro-batch loop.
        base_model._force_causal = (random.random() < args.p_causal) or args.n_channels <= 1
        base_model._mask_rate = args.mask_rate_min + random.random() * (args.mask_rate_max - args.mask_rate_min)
        if args.stochastic_recurrence > 0 and random.random() < args.stochastic_recurrence_prob:
            base_model.training_depth_recurrence = 1 + args.stochastic_recurrence
        else:
            base_model.training_depth_recurrence = 1

        # Progressive layer growing: start with fewer layers, grow at configured fraction.
        if args.progressive_layers_start > 0:
            elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
            if max_wallclock_ms is not None:
                grow = elapsed_ms >= args.progressive_grow_frac * max_wallclock_ms
            else:
                grow = step >= int(args.iterations * args.progressive_grow_frac)
            old_active = getattr(base_model, "_active_layers", len(base_model.blocks))
            new_active = len(base_model.blocks) if grow else args.progressive_layers_start
            base_model._active_layers = new_active
            if new_active != old_active:
                torch._dynamo.reset()
                log0(f"step:{step} progressive_grow: {old_active} -> {new_active} layers")

        for micro in range(grad_accum_steps):
            if distributed:
                # Federated averaging: disable gradient sync, sync weights periodically instead.
                if args.fedavg_every > 0:
                    model.require_backward_grad_sync = False
                else:
                    model.require_backward_grad_sync = micro == grad_accum_steps - 1
            x, y = train_loader.next_batch(active_batch_tokens, active_seq_len, grad_accum_steps)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if boost_model is not None:
                    # Boosted loss: upweight tokens the boost_model gets wrong.
                    with torch.no_grad():
                        boost_loss = boost_model(x, y, reduction="none")  # per-token loss
                        # Normalize weights: high loss → high weight. Clamp to [0.5, 3.0].
                        weights = (boost_loss / boost_loss.mean()).clamp(0.5, 3.0)
                    # Train current model with weighted per-token loss.
                    per_token_loss = model(x, y, reduction="none")
                    loss = (per_token_loss * weights).mean()
                else:
                    loss = model(x, y)
            train_loss.add_(loss.detach())
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Untie lm_head at configured fraction of training
        if args.untie_at_fraction > 0:
            if max_wallclock_ms is not None:
                should_untie = not _untied and elapsed_ms >= args.untie_at_fraction * max_wallclock_ms
            else:
                should_untie = not _untied and step >= int(args.iterations * args.untie_at_fraction)
            if should_untie and base_model.tie_embeddings:
                with torch.no_grad():
                    base_weight = base_model.tok_emb.weight.float()
                    if base_model.lm_head_correction is not None:
                        base_weight = base_weight + base_model.lm_head_correction.float()
                    if base_model.embed_proj_rev is not None:
                        full_weight = base_weight @ base_model.embed_proj_rev.weight.float()
                    else:
                        full_weight = base_weight
                    base_model.lm_head.weight.copy_(full_weight)
                base_model.tie_embeddings = False
                base_model.lm_head.weight.requires_grad_(True)
                for g in opt_head.param_groups:
                    g["lr"] = g["base_lr"] = args.head_lr
                _untied = True
                torch._dynamo.reset()
                log0(f"step:{step} untied lm_head (head_lr={args.head_lr})")

        # Muon momentum warmup
        if args.matrix_optimizer != "adam":
            frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
            for g in opt_muon.param_groups:
                g["momentum"] = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum

        # LR scheduling
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale
            opt.step()
        zero_grad_all()
        step += 1

        # Federated averaging: periodic weight sync across GPUs.
        if args.fedavg_every > 0 and distributed and step % args.fedavg_every == 0:
            with torch.no_grad():
                for param in base_model.parameters():
                    dist.all_reduce(param.data, op=dist.ReduceOp.AVG)

        # EMA update — only on causal steps to preserve ternary-friendly weight distributions.
        if step >= ema_start_step and base_model._force_causal:
            if ema_state is None:
                ema_state = {n: t.detach().clone() for n, t in base_model.state_dict().items()}
            else:
                d = args.ema_decay
                for n, t in base_model.state_dict().items():
                    ema_state[n].mul_(d).add_(t.detach(), alpha=1.0 - d)
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        
        if args.train_log_every > 0 and step % args.train_log_every == 0:
            log0(f"step:{step}/{args.iterations} loss:{train_loss.item():.4f} t:{approx_ms:.0f}ms avg:{approx_ms/step:.1f}ms")
        if args.churn_log_every > 0 and step % args.churn_log_every == 0:
            log0(f"step:{step} churn:{churn_fn(base_model, args.bitnet_group_size):.4f} zero:{tern_stats(base_model, args.bitnet_group_size)['zero_frac']:.3f}")

        # Wallclock cap sync
        if stop_after_step is None and max_wallclock_ms is not None and step % 10 == 0:
            reached_cap = approx_ms >= max_wallclock_ms
            if distributed:
                cap_t = torch.tensor(int(reached_cap), device=device)
                dist.all_reduce(cap_t, op=dist.ReduceOp.MAX)
                reached_cap = bool(cap_t.item())
            if reached_cap:
                stop_after_step = step

    # --- Serialization ---
    model_file = "final_model.ptz"
    if master_process:
        if ema_state is not None:
            log0("using EMA weights for serialization")
            sd = ema_state
        else:
            sd = base_model.state_dict()
        sd = {k: v for k, v in sd.items()
              if (not k.startswith("channel_") or k in ("channel_in", "channel_out"))
              and not k.startswith("ngram_injector.table_")
              and not k.startswith("ngram_logit_mixer.table_")
              and not k.startswith("bigram_logit_mixer.bigram_")
              and "base_weight" not in k}
        if base_model.tie_embeddings or args.logit_head_type == "tversky":
            sd.pop("lm_head.weight", None)

        # Save float weights for offline requantization experiments.
        if os.environ.get("SAVE_FLOAT", "0") == "1":
            float_file = model_file.replace(".ptz", "_float.pt")
            torch.save({k: v.cpu().float() for k, v in sd.items()}, float_file)
            log0(f"saved float weights: {float_file} ({os.path.getsize(float_file)/1e6:.1f}MB)")

        if args.quant_scheme == "mixed_ternary":
            # Mixed serialization: ternary for blocks in TERNARY_BLOCKS, int6+GPTQ for the rest
            # For now: ternary for marked blocks, int6 for others
            ternary_names = set()
            for name in sd:
                for bi in (int(x) for x in os.environ.get("TERNARY_BLOCKS", "").split(",") if x):
                    if f"blocks.{bi}." in name:
                        ternary_names.add(name)
            methods = {}
            for method in ("standard", "bitmask"):
                q_obj, stats = q_sd(sd, group_size=args.bitnet_group_size,
                                    ternary_method=method, ternary_override_names=ternary_names)
                buf = io.BytesIO()
                torch.save(q_obj, buf)
                try:
                    import zstandard
                    blob = zstandard.ZstdCompressor(level=22).compress(buf.getvalue())
                except ImportError:
                    blob = lzma.compress(buf.getvalue(), preset=9)
                methods[method] = {"blob": blob, "stats": stats}
            best = min(methods, key=lambda m: len(methods[m]["blob"]))
            final_blob = methods[best]["blob"]
            q_stats = methods[best]["stats"]
            log0(f"mixed_ternary serialization: ternary_blocks={len(ternary_names)} "
                 f"ternary:{q_stats['ternary_params']} fp:{q_stats['fp_params']}")
        elif args.quant_scheme.startswith("lloyd_max"):
            # Lloyd-Max serialization: Gaussian-optimal quantization
            n_levels = {"lloyd_max_int3": 7, "lloyd_max_int4": 15, "lloyd_max_int5": 31,
                        "lloyd_max": 63}.get(args.quant_scheme, 15)
            q_obj, q_stats = q_sd_lloyd_max(sd, n_levels=n_levels)
            buf = io.BytesIO()
            torch.save(q_obj, buf)
            try:
                import zstandard
                final_blob = zstandard.ZstdCompressor(level=22).compress(buf.getvalue())
                compress_name = "zstd-22"
            except ImportError:
                final_blob = lzma.compress(buf.getvalue(), preset=9)
                compress_name = "lzma-9"
            log0(f"lloyd_max serialization ({compress_name}, {n_levels} levels): "
                 f"lm:{q_stats['lm_params']} int8:{q_stats['int8_params']} fp:{q_stats['fp_params']}")
        elif args.quant_mode == "int6":
            # Int6 serialization: int6 for block weights, int8 for embeddings, fp16 for small.
            q_obj, q_stats = q_sd_int6(sd, clip_q=args.int6_clip_q)
            buf = io.BytesIO()
            torch.save(q_obj, buf)
            try:
                import zstandard
                final_blob = zstandard.ZstdCompressor(level=22).compress(buf.getvalue())
                compress_name = "zstd-22"
            except ImportError:
                final_blob = lzma.compress(buf.getvalue(), preset=9)
                compress_name = "lzma-9"
            log0(f"int6 serialization ({compress_name}): int6:{q_stats['int6_params']} int8:{q_stats['int8_params']} fp:{q_stats['fp_params']}")
        else:
            # Ternary serialization: base-3 packing + LZMA.
            ternary_overrides = set()
            for n, m in base_model.named_modules():
                if isinstance(m, TverskyProjection) and m.no_features_mode:
                    ternary_overrides.add(n + ".prototypes")
            ternary_overrides = ternary_overrides or None
            methods = {}
            for method in ("standard", "bitmask"):
                q_obj, stats = q_sd(sd, group_size=args.bitnet_group_size, fp_storage=args.fp_storage, ternary_method=method, ternary_override_names=ternary_overrides)
                buf = io.BytesIO()
                torch.save(q_obj, buf)
                methods[method] = {"blob": lzma.compress(buf.getvalue(), preset=9), "stats": stats}
            best = min(methods, key=lambda m: len(methods[m]["blob"]))
            final_blob, q_stats = methods[best]["blob"], methods[best]["stats"]
            log0(f"ternary serialization: ternary:{q_stats['ternary_params']}({q_stats['ternary_bytes']}B) fp:{q_stats['fp_params']}({q_stats['fp_bytes']}B)")

        # Include bigram count table in the artifact if enabled.
        bigram_counts_blob = None
        if base_model.bigram_logit_mixer is not None:
            bc = base_model.bigram_logit_mixer.serialize_counts()
            bc_buf = io.BytesIO()
            torch.save({"bigram_counts": bc, "gate": base_model.bigram_logit_mixer.gate.detach().cpu()}, bc_buf)
            bigram_counts_blob = bc_buf.getvalue()
            log0(f"bigram artifact: {len(bigram_counts_blob)/1e6:.2f}MB (before compression)")

        # Include n-gram tables in the artifact if enabled.
        if ngram_builder is not None:
            ngram_data = ngram_builder.serialize()
            ngram_buf = io.BytesIO()
            torch.save(ngram_data, ngram_buf)
            ngram_blob = ngram_buf.getvalue()
            log0(f"ngram artifact: {len(ngram_blob)/1e6:.2f}MB (before compression)")
            # Combine model + ngram as a tuple.
            combined_buf = io.BytesIO()
            torch.save({"model": final_blob, "ngram": ngram_blob}, combined_buf)
            try:
                import zstandard
                final_combined = zstandard.ZstdCompressor(level=22).compress(combined_buf.getvalue())
            except ImportError:
                final_combined = lzma.compress(combined_buf.getvalue(), preset=9)
            with open(model_file, "wb") as f:
                f.write(final_combined)
        elif bigram_counts_blob is not None:
            combined_buf = io.BytesIO()
            torch.save({"model": final_blob, "bigram": bigram_counts_blob}, combined_buf)
            try:
                import zstandard
                final_combined = zstandard.ZstdCompressor(level=22).compress(combined_buf.getvalue())
            except ImportError:
                final_combined = lzma.compress(combined_buf.getvalue(), preset=9)
            with open(model_file, "wb") as f:
                f.write(final_combined)
        else:
            with open(model_file, "wb") as f:
                f.write(final_blob)
        artifact_bytes = os.path.getsize(model_file)
        code_bytes = len(code.encode("utf-8"))
        total = artifact_bytes + code_bytes
        log0(f"artifact:{artifact_bytes/1e6:.2f}MB code:{code_bytes}")
        log0(f"budget:{total}/{16000000} ({total/1e6:.2f}/{16.00:.2f}MB) {'FITS' if total <= 16000000 else 'OVER'}")

        if args.eval_depth_recurrence > 0:
            base_model.training_depth_recurrence = args.eval_depth_recurrence
            log0(f"eval_depth_recurrence:{args.eval_depth_recurrence}")

    # --- All ranks load roundtrip weights and evaluate ---
    if distributed:
        dist.barrier()

    with open(model_file, "rb") as f:
        raw = f.read()
    try:
        decompressed = lzma.decompress(raw)
    except Exception:
        import zstandard
        decompressed = zstandard.ZstdDecompressor().decompress(raw)
    loaded = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=False)

    # Handle combined artifacts (model + bigram/ngram).
    if isinstance(loaded, dict) and "model" in loaded:
        model_blob = loaded["model"]
        # model_blob is the compressed model bytes — decompress then load.
        try:
            model_bytes = lzma.decompress(model_blob)
        except Exception:
            import zstandard
            model_bytes = zstandard.ZstdDecompressor().decompress(model_blob)
        model_data = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=False)
        # Load bigram table if present.
        if "bigram" in loaded and base_model.bigram_logit_mixer is not None:
            bigram_data = torch.load(io.BytesIO(loaded["bigram"]), map_location="cpu", weights_only=False)
            base_model.bigram_logit_mixer.load_counts(bigram_data["bigram_counts"].float())
            base_model.bigram_logit_mixer.gate.data.copy_(bigram_data["gate"])
            log0(f"loaded bigram table from artifact, gate={bigram_data['gate'].item():.4f}")
        loaded = model_data

    if args.quant_scheme.startswith("lloyd_max"):
        base_model.load_state_dict(deq_sd_lloyd_max(loaded), strict=False)
    elif args.quant_scheme == "mixed_ternary" or args.quant_mode == "ternary":
        base_model.load_state_dict(deq_sd(loaded), strict=False)
    elif args.quant_mode == "int6":
        base_model.load_state_dict(deq_sd_int6(loaded), strict=False)
    else:
        base_model.load_state_dict(deq_sd(loaded), strict=False)
    torch._dynamo.reset()

    q_val_loss, q_val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                     val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    log0(f"final_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")

    # --- Ensemble eval: mix with a second model if ENSEMBLE_PATH is set ---
    if args.ensemble_path and master_process:
        log0(f"ensemble: loading second model from {args.ensemble_path}")
        with open(args.ensemble_path, "rb") as f:
            raw2 = f.read()
        try:
            dec2 = lzma.decompress(raw2)
        except Exception:
            import zstandard
            dec2 = zstandard.ZstdDecompressor().decompress(raw2)
        loaded2 = torch.load(io.BytesIO(dec2), map_location="cpu", weights_only=False)
        if isinstance(loaded2, dict) and "model" in loaded2:
            try:
                mb2 = lzma.decompress(loaded2["model"])
            except Exception:
                import zstandard
                mb2 = zstandard.ZstdDecompressor().decompress(loaded2["model"])
            loaded2 = torch.load(io.BytesIO(mb2), map_location="cpu", weights_only=False)
        # Create second model with same arch, load weights.
        model2 = GPT(
            vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
            num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
            activation=args.activation_type, n_channels=args.n_channels,
            xsa_layers=args.xsa_layers, mixer_type=args.mixer_type,
        ut_unique_blocks=args.ut_unique_blocks, ut_iters=args.ut_iters, sse_enabled=args.sse_enabled,
            sse_clusters=args.sse_clusters, sse_entropy_bins=args.sse_entropy_bins,
        ).to(device).bfloat16()
        for m in model2.modules():
            if isinstance(m, nn.Linear):
                m.float()
        restore_low_dim_params_to_fp32(model2)
        if args.quant_scheme.startswith("lloyd_max"):
            model2.load_state_dict(deq_sd_lloyd_max(loaded2), strict=False)
        elif args.quant_mode == "int6":
            model2.load_state_dict(deq_sd_int6(loaded2), strict=False)
        else:
            model2.load_state_dict(deq_sd(loaded2), strict=False)
        model2.eval()

        # Evaluate ensemble with different alphas.
        seq_len = args.train_seq_len
        total_seqs = (val_tokens.numel() - 1) // seq_len
        for alpha_name, alpha in [("50/50", 0.5), ("60/40", 0.6), ("70/30", 0.7)]:
            loss_sum = torch.zeros((), device=device, dtype=torch.float64)
            token_count = torch.zeros((), device=device, dtype=torch.float64)
            byte_count = torch.zeros((), device=device, dtype=torch.float64)
            with torch.inference_mode():
                for bs in range(0, total_seqs, 32):
                    be = min(bs + 32, total_seqs)
                    local = val_tokens[bs*seq_len:(be*seq_len)+1].to(device=device, dtype=torch.int64)
                    x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        # Model 1 logits (already loaded into base_model)
                        e1 = base_model._embed(x)
                        h1 = base_model._run_blocks(e1, e1, causal=True)
                        h1f = h1.reshape(-1, h1.size(-1))
                        l1 = base_model._softcap(base_model._compute_logits(h1f))
                        if base_model.sse is not None:
                            l1 = base_model.sse(l1)
                        # Model 2 logits
                        e2 = model2._embed(x)
                        h2 = model2._run_blocks(e2, e2, causal=True)
                        h2f = h2.reshape(-1, h2.size(-1))
                        l2 = model2._softcap(model2._compute_logits(h2f))
                        if model2.sse is not None:
                            l2 = model2.sse(l2)
                    mixed = alpha * l1.float() + (1-alpha) * l2.float()
                    targets = y.reshape(-1)
                    lse = torch.logsumexp(mixed, dim=-1)
                    tgt_l = mixed.gather(1, targets.unsqueeze(1)).squeeze(1)
                    loss_sum += (lse - tgt_l).to(torch.float64).sum()
                    n = float(targets.numel())
                    token_count += n
                    prev_ids, tgt_ids = x.reshape(-1), y.reshape(-1)
                    tb = base_bytes_lut[tgt_ids].to(torch.float64)
                    tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
                    byte_count += tb.sum()
            vl = (loss_sum / token_count).item()
            bpb = (vl / math.log(2.0)) * (token_count.item() / byte_count.item())
            log0(f"ensemble_{alpha_name} val_loss:{vl:.4f} val_bpb:{bpb:.4f}")
        del model2
        torch.cuda.empty_cache()

    opt_temp = 1.0
    if args.temp_scaling:
        torch.cuda.synchronize()
        t_temp = time.perf_counter()
        calibration_tokens = train_loader.stream.take(65536).to(device)
        opt_temp = find_temp(args, base_model, rank, world_size, device, grad_accum_steps,
                                            calibration_tokens, base_bytes_lut, has_leading_space_lut,
                                            is_boundary_token_lut)
        torch.cuda.synchronize()
        temp_time_ms = 1000.0 * (time.perf_counter() - t_temp)
        log0(f"temp_scaling optimal_T:{opt_temp:.2f} eval_time:{temp_time_ms:.0f}ms")

    if args.sliding_eval:
        torch.cuda.synchronize()
        t_sliding = time.perf_counter()
        sw_loss, sw_bpb = eval_val_sliding(args, base_model, rank, world_size, device, grad_accum_steps,
                                           val_tokens, base_bytes_lut, has_leading_space_lut,
                                           is_boundary_token_lut, stride=args.sliding_eval_stride,
                                           temperature=opt_temp)
        torch.cuda.synchronize()
        sliding_time_ms = 1000.0 * (time.perf_counter() - t_sliding)
        log0(f"final_sliding val_loss:{sw_loss:.4f} val_bpb:{sw_bpb:.4f} "
             f"(stride={args.sliding_eval_stride}, T={opt_temp:.2f}) eval_time:{sliding_time_ms:.0f}ms")

    if return_model:
        return base_model
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
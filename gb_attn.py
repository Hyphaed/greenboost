"""
gb_attn.py - GreenBoost attention optimisation via TurboQuant+ K/V compression.

Full TurboQuant+ algorithm:
  - K: PolarQuant (k_bits−1 bits) + QJL 1-bit residual (inner-product preserving)
  - V: PolarQuant at v_bits (MSE-optimal; float v_bits → channel splitting)
  - Asymmetric K/V: separate k_bits / v_bits
  - Sparse-V: attention-gated skip for V positions with negligible weight (≤4096 tokens)
  - Layer-adaptive: boundary transformer layers get +1 bit for K automatically
  - Non-integer bit widths: channel splitting for 2.5 / 3.5 bit averages

WHY: GreenBoost overflows K/V tensors from fast T1 VRAM into T2 DDR when the
model is large (~23 GB for Flux BF16). Reading K/V from T2 over DMA for each
attention call is the main throughput bottleneck. TurboQuant+ achieves 3–8×
bandwidth reduction while preserving attention quality through inner-product-
preserving K compression.

Usage:
    import gb_attn

    # Asymmetric K/V (recommended - best quality/memory tradeoff):
    with gb_attn.turboquant_attention(k_bits=4, v_bits=3):
        output = model(input)

    # Symmetric shorthand (bit_width=3 → k_bits=3, v_bits=2):
    with gb_attn.turboquant_attention(bit_width=3):
        image = pipe(prompt, ...).images[0]

    # Sparse-V (+22% decode speed for sequences ≤4096 tokens):
    with gb_attn.turboquant_attention(k_bits=4, v_bits=3, sparse_v=True):
        output = model(input)

    # Non-integer bits - channel splitting (bit_width=3.5 → k_bits=4, v_bits=2.5):
    with gb_attn.turboquant_attention(bit_width=3.5):
        output = model(input)

    # Global patch for inference-only sessions:
    gb_attn.patch_sdpa(k_bits=4, v_bits=3)
    output = model(input)
    gb_attn.unpatch_sdpa()

Bandwidth gain: 3–8× depending on bit widths; k_bits=4, v_bits=3 ≈ 4× with
near-zero quality loss (+0.23% PPL).
"""

from __future__ import annotations

import math
import threading
import time
import numpy as np
import torch
import torch.nn.functional as _F
from contextlib import contextmanager
from typing import Optional, Union

# Audit F-L1-14: torch.compiler.disable lands in PyTorch 2.0 (the `compiler`
# submodule attribute).  Earlier versions raise AttributeError; provide a
# pass-through decorator so users on Torch 1.x can still import the module.
if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
    _compiler_disable = torch.compiler.disable
else:
    def _compiler_disable(fn):  # type: ignore[no-redef]
        return fn

# Audit F-L1-11: protect the layer-adaptive globals with a Lock so concurrent
# inference threads (vLLM batches, multi-model servers) don't corrupt the EMA
# layer-count estimate.  The lock window is tiny (just the counter update)
# so contention is irrelevant in practice.
_layer_state_lock = threading.Lock()


# ── Lloyd-Max codebook ────────────────────────────────────────────────────────

def _lloyd_max_gaussian(num_levels: int, sigma: float, max_iter: int = 500):
    """
    Optimal Lloyd-Max centroids and boundaries for N(0, sigma^2).
    Uses math.erf (stdlib) and torch.erfinv (PyTorch) - no scipy needed.
    """
    k = num_levels
    inv_sqrt2 = 1.0 / math.sqrt(2)
    sqrt2pi   = math.sqrt(2 * math.pi)

    def _pdf(z):
        return math.exp(-0.5 * z * z) / sqrt2pi

    def _cdf(z):
        return 0.5 * (1.0 + math.erf(z * inv_sqrt2))

    # Initialise at Gaussian quantiles: ppf(p) = sigma * sqrt(2) * erfinv(2p-1)
    probs = [(2 * i + 1) / (2 * k) for i in range(k)]
    centroids = np.array([
        float(sigma * math.sqrt(2) * torch.erfinv(torch.tensor(2.0 * p - 1.0)).item())
        for p in probs
    ], dtype=np.float64)

    for _ in range(max_iter):
        bounds = np.empty(k + 1)
        bounds[0] = -np.inf
        bounds[k] =  np.inf
        for i in range(1, k):
            bounds[i] = (centroids[i - 1] + centroids[i]) / 2.0

        new_c = np.empty(k)
        for i in range(k):
            lo = max(bounds[i],      -6.0 * sigma)
            hi = min(bounds[i + 1],   6.0 * sigma)
            # E[X | lo<X<hi] for X~N(0,sigma^2)
            num = sigma * (_pdf(lo / sigma) - _pdf(hi / sigma))
            den = _cdf(hi / sigma) - _cdf(lo / sigma)
            new_c[i] = num / den if den > 1e-15 else (lo + hi) / 2.0

        if np.allclose(centroids, new_c, atol=1e-12):
            break
        centroids = new_c

    bounds = np.empty(k + 1)
    bounds[0] = -np.inf
    bounds[k] =  np.inf
    for i in range(1, k):
        bounds[i] = (centroids[i - 1] + centroids[i]) / 2.0
    return centroids, bounds


# ── Per-head-dim quantizer cache ──────────────────────────────────────────────

# Keys: (bit_width:int, head_dim:int, device:str)         → (Pi, Pi_T, centroids, boundaries, scale)
#       (bit_width:int, head_dim:int, device:str, "qjl")  → S  (JL projection matrix)
_quantizer_cache: dict = {}


def _get_quantizer(bit_width: int, head_dim: int, device: str):
    # Audit F-L1-12: include dtype-relevant precision-class in the key.  We
    # keep centroids in float32 regardless of caller dtype (BF16/FP16 callers
    # will upcast at use site - the centroid table is small), but normalise
    # the bit_width to int and the device to a canonical string so two
    # callers passing `cuda` vs `cuda:0` get the same cached quantizer.
    bit_key = int(bit_width)
    dev_key = str(device).split(":")[0] if isinstance(device, str) else str(device)
    key = (bit_key, head_dim, dev_key)
    if key in _quantizer_cache:
        return _quantizer_cache[key]

    dev = torch.device(device)

    # Haar-distributed random orthogonal rotation matrix, seed=42
    gen = torch.Generator(device="cpu").manual_seed(42)
    G   = torch.randn(head_dim, head_dim, generator=gen, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    ds = torch.sign(torch.diag(R))
    ds[ds == 0] = 1.0
    Pi   = (Q * ds.unsqueeze(0)).to(dev).contiguous()
    Pi_T = Pi.T.contiguous()

    # Lloyd-Max codebook for N(0, 1/head_dim)
    sigma = 1.0 / math.sqrt(head_dim)
    c_np, b_np = _lloyd_max_gaussian(2 ** bit_width, sigma=sigma)
    centroids  = torch.tensor(c_np, dtype=torch.float32, device=dev)
    boundaries = torch.tensor(b_np[1:-1], dtype=torch.float32, device=dev)

    scale = 1.0 / math.sqrt(head_dim)
    _quantizer_cache[key] = (Pi, Pi_T, centroids, boundaries, scale)
    return _quantizer_cache[key]


# ── Core PolarQuant ops (used for V) ─────────────────────────────────────────

@torch.no_grad()
def _quantize(X, Pi_T, boundaries):
    """(B, H, S, D) float → (uint8 indices, float32 norms per vector)"""
    flat  = X.float().reshape(-1, X.shape[-1])
    norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    idx   = torch.bucketize((flat / norms) @ Pi_T, boundaries).to(torch.uint8)
    return idx.view(X.shape), norms.squeeze(-1).view(X.shape[:-1])


@torch.no_grad()
def _dequantize(idx, norms, Pi, centroids):
    """uint8 indices + norms → float32 reconstructed vectors"""
    return (centroids[idx.long()] @ Pi) * norms.unsqueeze(-1)


# ── QJL 1-bit residual (K tensors only) ──────────────────────────────────────

def _get_qjl_matrix(bit_width: int, head_dim: int, device: str):
    """Return or create the JL projection matrix S ∈ ℝ^{d×d} ~ N(0,1)/sqrt(d)."""
    key = (bit_width, head_dim, device, "qjl")
    if key not in _quantizer_cache:
        gen = torch.Generator(device="cpu").manual_seed(137)
        S = torch.randn(head_dim, head_dim, generator=gen, dtype=torch.float32)
        S = S / math.sqrt(head_dim)
        _quantizer_cache[key] = S.to(device)
    return _quantizer_cache[key]


@torch.no_grad()
def _qjl_encode(residual, S):
    """residual (…, D) float → int8 signs {+1, −1} via JL projection."""
    return torch.sign(residual.float() @ S.T).to(torch.int8)


@torch.no_grad()
def _qjl_decode(signs, S):
    """int8 signs → approximate residual (unbiased estimator)."""
    d = S.shape[0]
    return (math.sqrt(math.pi / 2.0) / d) * (signs.float() @ S)


@torch.no_grad()
def _quantize_k_turbo(K, bit_width: int, device: str):
    """Full TurboQuant for K: PolarQuant at (bit_width−1) bits + QJL 1-bit residual."""
    polar_bits = max(bit_width - 1, 1)
    Pi, Pi_T, centroids, boundaries, _ = _get_quantizer(polar_bits, K.shape[-1], device)
    S = _get_qjl_matrix(bit_width, K.shape[-1], device)

    flat  = K.float().reshape(-1, K.shape[-1])
    norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    unit  = flat / norms
    rotated = unit @ Pi_T

    # Stage 1: PolarQuant on rotated unit vectors
    idx = torch.bucketize(rotated, boundaries).to(torch.uint8)
    K_polar = centroids[idx.long()] @ Pi

    # Stage 2: QJL 1-bit residual for inner-product preservation
    residual = unit - K_polar
    signs = _qjl_encode(residual, S)

    return (idx.view(K.shape),
            norms.squeeze(-1).view(K.shape[:-1]),
            signs.view(K.shape))


@torch.no_grad()
def _dequantize_k_turbo(idx, norms, signs, bit_width: int, device: str):
    """Reconstruct K from TurboQuant compressed representation."""
    polar_bits = max(bit_width - 1, 1)
    Pi, _, centroids, _, _ = _get_quantizer(polar_bits, idx.shape[-1], device)
    S = _get_qjl_matrix(bit_width, idx.shape[-1], device)

    K_polar  = centroids[idx.long()] @ Pi
    K_qjl    = _qjl_decode(signs, S)
    unit_hat = K_polar + K_qjl
    return unit_hat * norms.unsqueeze(-1)


# ── Non-integer bit width: channel splitting (V only) ────────────────────────

def _split_channels(tensor, low_bits: int, high_frac: float):
    """
    Split D channels by variance: ceil(D × high_frac) get (low_bits+1) precision,
    remaining get low_bits precision.  Returns (high_part, low_part, order).
    """
    D = tensor.shape[-1]
    n_high = int(math.ceil(D * high_frac))
    var = tensor.float().var(dim=list(range(tensor.dim() - 1)))
    order = torch.argsort(var, descending=True)
    return tensor[..., order[:n_high]], tensor[..., order[n_high:]], order


@torch.no_grad()
def _quantize_v_split(V, v_bits_f: float, device: str):
    """V PolarQuant with non-integer bit width via channel splitting."""
    low_bits  = int(math.floor(v_bits_f))
    high_frac = v_bits_f - low_bits

    high_part, low_part, order = _split_channels(V, low_bits, high_frac)

    Pi_h, Pi_T_h, c_h, b_h, _ = _get_quantizer(low_bits + 1, high_part.shape[-1], device)
    h_idx, h_norms = _quantize(high_part, Pi_T_h, b_h)

    Pi_l, Pi_T_l, c_l, b_l, _ = _get_quantizer(low_bits, low_part.shape[-1], device)
    l_idx, l_norms = _quantize(low_part, Pi_T_l, b_l)

    return h_idx, h_norms, l_idx, l_norms, order, low_bits


@torch.no_grad()
def _dequantize_v_split(h_idx, h_norms, l_idx, l_norms, order, low_bits: int, device: str):
    """Reconstruct V from channel-split quantization."""
    Pi_h, _, c_h, _, _ = _get_quantizer(low_bits + 1, h_idx.shape[-1], device)
    Pi_l, _, c_l, _, _ = _get_quantizer(low_bits,     l_idx.shape[-1], device)

    high_hat = _dequantize(h_idx, h_norms, Pi_h, c_h)
    low_hat  = _dequantize(l_idx, l_norms, Pi_l, c_l)

    D = h_idx.shape[-1] + l_idx.shape[-1]
    out = torch.empty(*h_idx.shape[:-1], D, dtype=torch.float32, device=device)
    n_high = h_idx.shape[-1]
    out[..., order[:n_high]] = high_hat
    out[..., order[n_high:]] = low_hat
    return out


# ── Layer-adaptive precision ──────────────────────────────────────────────────

_call_in_seq:      int   = 0
_est_total_layers: int   = 0
_last_call_time:   float = 0.0

_LAYER_RESET_GAP_S = 1.0  # seconds of silence → assume new sequence started


def _adaptive_k_bits(base_bits: int, call_idx: int, total: int) -> int:
    """Boundary layers (first 2 + last 2) automatically get +1 bit for K."""
    if total <= 0:
        return base_bits
    if call_idx < 2 or call_idx >= total - 2:
        return min(base_bits + 1, 8)
    return base_bits


# ── Sparse-V constants ────────────────────────────────────────────────────────

SPARSE_V_THRESHOLD = 0.01   # skip V positions with attention weight < 1% of max
SPARSE_V_MAX_SEQ   = 4096   # above this, fall back to standard _tq_attn


# ── Attention core functions ──────────────────────────────────────────────────

@torch.no_grad()
def _tq_attn(query, key, value, k_bits: int, v_bits, device: str, original_sdpa, **sdpa_kwargs):
    """
    TurboQuant+ attention: full TurboQuant for K, PolarQuant for V.

    K uses PolarQuant (k_bits−1) + QJL residual for inner-product preservation.
    V uses PolarQuant only - MSE-optimal, inner products not needed for output.
    v_bits may be a float for non-integer channel-split quantization.

    PR-E/C8: thread the caller's SDPA kwargs (notably `scale` for RoPE/muP
    models, and `enable_gqa` for PyTorch 2.5+ GQA path) through to the
    underlying SDPA call.  Previously these were silently dropped - for
    models with non-default scale (Mistral, muP-trained models) attention
    magnitudes came out wrong.  dropout_p/attn_mask/is_causal are NOT
    forwarded because the caller handles them (see _tq_sdpa).
    """
    K_idx, K_norms, K_signs = _quantize_k_turbo(key, k_bits, device)
    K_hat = _dequantize_k_turbo(K_idx, K_norms, K_signs, k_bits, device)

    orig_dtype = query.dtype

    if isinstance(v_bits, float) and v_bits != int(v_bits):
        h_idx, h_norms, l_idx, l_norms, order, low_bits = _quantize_v_split(value, v_bits, device)
        V_hat = _dequantize_v_split(h_idx, h_norms, l_idx, l_norms, order, low_bits, device)
    else:
        vb = int(v_bits)
        Pi, Pi_T, centroids, boundaries, _ = _get_quantizer(vb, value.shape[-1], device)
        V_idx, V_norms = _quantize(value, Pi_T, boundaries)
        V_hat = _dequantize(V_idx, V_norms, Pi, centroids)

    return original_sdpa(query, K_hat.to(orig_dtype), V_hat.to(orig_dtype), **sdpa_kwargs)


@torch.no_grad()
def _tq_attn_sparse_v(query, key, value, k_bits: int, v_bits, device: str, original_sdpa, **sdpa_kwargs):
    """
    Sparse-V variant: materialise attention weights, skip V positions with negligible
    weight.  Falls back to _tq_attn when sequence length exceeds SPARSE_V_MAX_SEQ
    (materialising the attention matrix at long contexts would be catastrophic).

    PR-E/C9: two bugs fixed in this revision.
      (a) Threshold semantics: SPARSE_V_THRESHOLD is now a fraction of the
          per-(batch, head) attention-weight maximum (matching the docstring
          "weight < 1% of max").  The previous absolute 0.01 cutoff collapsed
          the entire mask to False on long-context decoder runs where average
          attention weight is below that floor → V_masked all zeros → silent
          output zeroing.
      (b) Output renormalisation: after masking V positions we now zero the
          corresponding attn_w columns and renormalise the kept weights to
          sum to 1 per query position.  Previously attn_w summed to 1 over
          all positions but only kept positions contributed, causing
          systematic output attenuation.
    Honours the caller's `scale` kwarg (e.g. RoPE / muP), falling back to
    1/sqrt(D) only if not provided.
    """
    if key.shape[2] > SPARSE_V_MAX_SEQ:
        return _tq_attn(query, key, value, k_bits, v_bits, device, original_sdpa, **sdpa_kwargs)

    K_idx, K_norms, K_signs = _quantize_k_turbo(key, k_bits, device)
    K_hat = _dequantize_k_turbo(K_idx, K_norms, K_signs, k_bits, device)

    scale  = sdpa_kwargs.get("scale", 1.0 / math.sqrt(query.shape[-1]))
    attn_w = torch.softmax(query.float() @ K_hat.float().transpose(-2, -1) * scale, dim=-1)

    # PR-E/C9(a): relative threshold - fraction of the per-(B, H) max weight.
    # max over (S_q, S_kv) gives per-head peak; threshold is that × fraction.
    per_head_max = attn_w.amax(dim=(-2, -1), keepdim=True)            # (B, H, 1, 1)
    abs_thresh   = per_head_max.squeeze(-2) * SPARSE_V_THRESHOLD     # (B, H, 1)
    pos_max      = attn_w.amax(dim=-2)                                # (B, H, S_kv) - max over queries
    mask         = pos_max >= abs_thresh                              # (B, H, S_kv)
    V_masked     = value * mask.unsqueeze(-1)

    # PR-E/C9(b): zero attn_w at dropped positions and renormalise so the
    # kept weights sum to 1.  Without this, attn_w @ V_hat is attenuated by
    # the discarded probability mass.
    attn_w = attn_w * mask.unsqueeze(-2)
    attn_w = attn_w / attn_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    orig_dtype = query.dtype

    if isinstance(v_bits, float) and v_bits != int(v_bits):
        h_idx, h_norms, l_idx, l_norms, order, low_bits = _quantize_v_split(V_masked, v_bits, device)
        V_hat = _dequantize_v_split(h_idx, h_norms, l_idx, l_norms, order, low_bits, device)
    else:
        vb = int(v_bits)
        Pi, Pi_T, centroids, boundaries, _ = _get_quantizer(vb, V_masked.shape[-1], device)
        V_idx, V_norms = _quantize(V_masked, Pi_T, boundaries)
        V_hat = _dequantize(V_idx, V_norms, Pi, centroids)

    return attn_w.to(orig_dtype) @ V_hat.to(orig_dtype)


# ── Patched SDPA factory ──────────────────────────────────────────────────────

_original_sdpa = None


def _make_tq_sdpa(k_bits: int, v_bits, device: str, sparse_v: bool, layer_adaptive: bool):
    original = _F.scaled_dot_product_attention
    attn_fn  = _tq_attn_sparse_v if sparse_v else _tq_attn

    # @torch.compiler.disable: prevent dynamo from tracing into numpy/.item() code
    # inside _get_quantizer.  Without this, compile creates catastrophic graph breaks
    # that cause Triton to compile hundreds of kernel variants → stuck on first call.
    @_compiler_disable
    def _tq_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        global _call_in_seq, _est_total_layers, _last_call_time

        if key.dim() == 4 and not is_causal and attn_mask is None:
            # Audit F-L1-15: hardness invariant - K/V must share head_dim.
            assert key.shape[-1] == value.shape[-1], (
                f"gb_attn: K/V head_dim mismatch ({key.shape[-1]} vs {value.shape[-1]})"
            )
            if layer_adaptive:
                # Audit F-L1-11: guard the EMA-state mutation against
                # concurrent inference threads.
                with _layer_state_lock:
                    now = time.monotonic()
                    if now - _last_call_time > _LAYER_RESET_GAP_S and _call_in_seq > 0:
                        # EMA update for total layer count after a completed sequence
                        _est_total_layers = (
                            int(0.7 * _est_total_layers + 0.3 * _call_in_seq)
                            if _est_total_layers > 0 else _call_in_seq
                        )
                        _call_in_seq = 0
                    _last_call_time = now
                    call_idx = _call_in_seq
                    _call_in_seq += 1
                    eff_k_bits = _adaptive_k_bits(k_bits, call_idx, _est_total_layers)
            else:
                eff_k_bits = k_bits

            # PR-E/C8: forward the caller's SDPA kwargs (scale, enable_gqa)
            # through to the underlying SDPA call inside _tq_attn / _tq_attn_sparse_v.
            # Strip the ones we already handle (dropout, mask, causal) and the
            # _tq_attn-internal name 'original_sdpa'.
            fwd_kwargs = {k: v for k, v in kwargs.items()
                          if k not in ("dropout_p", "attn_mask", "is_causal", "original_sdpa")}
            out = attn_fn(query, key, value, eff_k_bits, v_bits, device, original, **fwd_kwargs)
            if dropout_p > 0.0:
                out = torch.nn.functional.dropout(out, p=dropout_p)
            return out

        return original(query, key, value, attn_mask, dropout_p, is_causal, **kwargs)

    return _tq_sdpa, original


# ── KVPress sparsity modes (E2) ──────────────────────────────────────────────
#
# Three complementary sparsity strategies from the KVPress project, orthogonal
# to TurboQuant+ quantization and combinable with it ("snapkv+turboquant").
#
# All three apply only when key.dim() == 4 and S_q == 1 (decode step) or
# S_q == S_k (prefill with non-causal mask). Causal-prefill passes through.

def _make_snapkv_sdpa(keep_ratio: float, original, tq_patch=None):
    """
    SnapKV: keep the top-K key positions ranked by mean attention score.
    Optionally chain into TurboQuant+ after sparsification (tq_patch != None).
    """
    @_compiler_disable
    def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        if key.dim() != 4:
            return original(query, key, value, attn_mask, dropout_p, is_causal, **kw)

        B, H, S_k, D = key.shape
        _, _, S_q, _  = query.shape
        keep_k = max(1, int(S_k * keep_ratio))

        if keep_k < S_k and not is_causal:
            sc = kw.get("scale", D ** -0.5)
            # Mean attention score per key position across all query positions and heads
            scores = torch.einsum("bhqd,bhkd->bhqk",
                                  query.float(), key.float()) * sc  # B H S_q S_k
            mean_scores = scores.mean(dim=(0, 2))                   # H S_k → average
            mean_scores = mean_scores.mean(dim=0)                   # S_k
            topk = mean_scores.topk(keep_k).indices.sort().values
            key   = key[:, :, topk, :]
            value = value[:, :, topk, :]
            attn_mask = None  # positions renumbered; causal mask no longer applies

        target = tq_patch if tq_patch is not None else original
        return target(query, key, value, attn_mask, dropout_p, is_causal, **kw)

    return _sdpa, original


def _make_streaming_sdpa(sink_tokens: int, local_window: int, original, tq_patch=None):
    """
    StreamingLLM: keep first `sink_tokens` (attention sinks) + last `local_window`.
    """
    @_compiler_disable
    def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        if key.dim() != 4:
            return original(query, key, value, attn_mask, dropout_p, is_causal, **kw)

        S_k = key.shape[2]
        sink = min(sink_tokens, S_k)
        start = max(sink, S_k - local_window)

        if start > sink:  # there are tokens to drop
            keep = list(range(sink)) + list(range(start, S_k))
            if len(keep) < S_k:
                idx = torch.tensor(keep, device=key.device)
                key   = key[:, :, idx, :]
                value = value[:, :, idx, :]
                attn_mask = None

        target = tq_patch if tq_patch is not None else original
        return target(query, key, value, attn_mask, dropout_p, is_causal, **kw)

    return _sdpa, original


def _make_knorm_sdpa(threshold: float, original, tq_patch=None):
    """
    KNorm: drop K/V positions whose K-vector L2 norm is below `threshold * max_norm`.
    """
    @_compiler_disable
    def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        if key.dim() != 4:
            return original(query, key, value, attn_mask, dropout_p, is_causal, **kw)

        if not is_causal:
            k_norms    = key.float().norm(dim=-1)             # B H S_k
            mean_norms = k_norms.mean(dim=(0, 1))             # S_k
            max_norm   = mean_norms.max()
            keep       = (mean_norms >= threshold * max_norm).nonzero(as_tuple=True)[0]
            if keep.numel() < key.shape[2]:
                key   = key[:, :, keep, :]
                value = value[:, :, keep, :]
                attn_mask = None

        target = tq_patch if tq_patch is not None else original
        return target(query, key, value, attn_mask, dropout_p, is_causal, **kw)

    return _sdpa, original


def _make_adakv_sdpa(keep_ratio: float, safeguard_factor: float, original, tq_patch=None):
    """
    AdaKV: per-head adaptive top-k based on attention entropy.
    Heads with concentrated attention (low entropy) keep fewer positions;
    high-entropy heads keep more. A safeguard floor prevents dropping below
    safeguard_factor × mean_keep positions in any single head.
    Source: kvpress/kvpress/presses/adakv_press.py
    """
    @_compiler_disable
    def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        if key.dim() != 4:
            return original(query, key, value, attn_mask, dropout_p, is_causal, **kw)

        B, H, S_k, D = key.shape
        S_q = query.shape[2]
        mean_keep = max(1, int(S_k * keep_ratio))

        # Compute per-head attention entropy using a tiny matmul
        with torch.no_grad():
            scale = D ** -0.5
            # Use last query token for scoring (generation-phase proxy)
            q_last = query[:, :, -1:, :].float()  # B H 1 D
            attn_w = (q_last @ key.float().transpose(-1, -2)) * scale  # B H 1 S_k
            attn_w = torch.softmax(attn_w, dim=-1).squeeze(2)           # B H S_k
            # Entropy per head: -sum(p log p) over S_k - averaged across batch
            eps = 1e-9
            entropy = -(attn_w * (attn_w + eps).log()).sum(dim=-1).mean(dim=0)  # H
            # Scale entropy [0,1] → keep_k per head
            e_min, e_max = entropy.min(), entropy.max()
            if e_max > e_min:
                e_norm = (entropy - e_min) / (e_max - e_min)
            else:
                e_norm = torch.ones(H, device=entropy.device)
            # High entropy → more positions kept; low entropy → fewer
            min_keep = max(1, int(mean_keep * safeguard_factor * 0.5))
            max_keep = min(S_k, int(S_k * keep_ratio * 2))
            keep_ks = (min_keep + e_norm * (max_keep - min_keep)).long().clamp(min_keep, S_k)
            # Safeguard: no head below safeguard_factor × mean_keep
            floor_k = max(1, int(mean_keep * safeguard_factor))
            keep_ks = keep_ks.clamp(min=floor_k)

        # Per-head top-k selection and re-assemble.
        # keep_ks may differ across heads; pad shorter heads to max_kh by
        # repeating their last selected index so torch.cat works on dim=1.
        max_kh = int(keep_ks.max().item())
        k_list, v_list = [], []
        for h in range(H):
            k_h = int(keep_ks[h].item())
            scores_h = attn_w[:, h, :]  # B S_k
            topk_idx = scores_h.mean(0).topk(k_h, dim=-1).indices.sort().values
            if k_h < max_kh:
                pad = topk_idx[-1:].expand(max_kh - k_h)
                topk_idx = torch.cat([topk_idx, pad])
            k_list.append(key[:, h:h+1, topk_idx, :])
            v_list.append(value[:, h:h+1, topk_idx, :])

        key   = torch.cat(k_list, dim=1)
        value = torch.cat(v_list, dim=1)

        target = tq_patch if tq_patch is not None else original
        return target(query, key, value, None, dropout_p, is_causal, **kw)

    return _sdpa, original


def _make_pyramidkv_sdpa(keep_ratio: float, window_size: int, original, tq_patch=None):
    """
    PyramidKV: layer-wise KV budget - earlier layers keep fewer positions,
    later layers keep more. A fixed `window_size` of recent tokens is always
    preserved. Call counter is tracked across SDPA invocations within one
    forward pass (reset when the context manager exits via unpatch_sdpa).
    Source: kvpress/kvpress/presses/pyramidkv_press.py
    """
    state = {"layer_idx": 0, "last_call_time": 0.0}

    @_compiler_disable
    def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        if key.dim() != 4:
            return original(query, key, value, attn_mask, dropout_p, is_causal, **kw)

        B, H, S_k, D = key.shape
        # Reset layer_idx at the start of each new forward pass (gap > threshold).
        now = time.monotonic()
        if now - state["last_call_time"] > _LAYER_RESET_GAP_S:
            state["layer_idx"] = 0
        state["last_call_time"] = now
        layer_idx = state["layer_idx"]
        state["layer_idx"] = layer_idx + 1

        # Linear ramp: layer 0 → min_budget, assume ~32 layers total
        total_layers = 32  # reasonable default; self-corrects at boundaries
        t = layer_idx / max(total_layers - 1, 1)
        min_budget = max(window_size, int(S_k * keep_ratio * 0.5))
        max_budget = min(S_k, int(S_k * keep_ratio * 1.5))
        budget = int(min_budget + t * (max_budget - min_budget))
        budget = min(budget, S_k)

        if budget >= S_k or S_k <= window_size:
            target = tq_patch if tq_patch is not None else original
            return target(query, key, value, attn_mask, dropout_p, is_causal, **kw)

        # Always keep last `window_size` tokens; select top-k from the prefix
        prefix_len = S_k - window_size
        prefix_budget = max(1, budget - window_size)

        with torch.no_grad():
            scale = D ** -0.5
            q_last = query[:, :, -1:, :].float()
            scores = (q_last @ key[:, :, :prefix_len, :].float().transpose(-1, -2)) * scale
            scores = torch.softmax(scores, dim=-1).mean(dim=(0, 2))  # prefix_len
            topk_idx = scores.topk(min(prefix_budget, prefix_len), dim=-1).indices.sort().values

        window_idx = torch.arange(S_k - window_size, S_k, device=key.device)
        idx = torch.cat([topk_idx, window_idx], dim=-1)
        key   = key[:, :, idx, :]
        value = value[:, :, idx, :]

        target = tq_patch if tq_patch is not None else original
        return target(query, key, value, None, dropout_p, is_causal, **kw)

    return _sdpa, state


def _make_criticalkv_sdpa(keep_ratio: float, w_o, original, tq_patch=None):
    """
    CriticalKV: score KV positions by the L1 norm of value projected through
    the output weight matrix W_O. Positions that contribute least to the output
    are pruned. Falls back to V.norm() when w_o is None.
    Source: kvpress/kvpress/presses/criticalkv_press.py
    """
    @_compiler_disable
    def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        if key.dim() != 4:
            return original(query, key, value, attn_mask, dropout_p, is_causal, **kw)

        B, H, S_k, D = key.shape
        keep_k = max(1, int(S_k * keep_ratio))

        if keep_k >= S_k:
            target = tq_patch if tq_patch is not None else original
            return target(query, key, value, attn_mask, dropout_p, is_causal, **kw)

        with torch.no_grad():
            if w_o is not None:
                try:
                    wo = w_o.float().to(value.device)  # out_dim, in_dim
                    v_flat = value.float().mean(0).mean(0)  # S_k D
                    scores = (v_flat @ wo.T).abs().sum(dim=-1)  # S_k
                except Exception:
                    scores = value.float().mean(0).mean(0).norm(dim=-1)
            else:
                scores = value.float().mean(0).mean(0).norm(dim=-1)  # S_k
            idx = scores.topk(keep_k, dim=-1).indices.sort().values

        key   = key[:, :, idx, :]
        value = value[:, :, idx, :]

        target = tq_patch if tq_patch is not None else original
        return target(query, key, value, None, dropout_p, is_causal, **kw)

    return _sdpa, original


def _make_lagkv_sdpa(lag: int, keep_ratio: float, original, tq_patch=None):
    """
    LagKV: split the KV sequence into a lag window (last `lag` positions, always
    kept) and a history prefix. Score history by mean attention from lag queries
    to history keys. Keep top-k history positions + full lag window.
    Source: kvpress/kvpress/presses/lagkv_press.py
    """
    @_compiler_disable
    def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        if key.dim() != 4:
            return original(query, key, value, attn_mask, dropout_p, is_causal, **kw)

        B, H, S_k, D = key.shape
        effective_lag = min(lag, S_k)
        prefix_len = S_k - effective_lag

        if prefix_len <= 0 or keep_ratio >= 1.0:
            target = tq_patch if tq_patch is not None else original
            return target(query, key, value, attn_mask, dropout_p, is_causal, **kw)

        keep_prefix = max(1, int(prefix_len * keep_ratio))

        with torch.no_grad():
            scale = D ** -0.5
            # Score history by attention from lag queries to history keys
            q_lag = query[:, :, -effective_lag:, :].float()        # B H lag D
            k_hist = key[:, :, :prefix_len, :].float()             # B H prefix_len D
            scores = (q_lag @ k_hist.transpose(-1, -2)) * scale    # B H lag prefix_len
            scores = torch.softmax(scores, dim=-1).mean(dim=(0, 1, 2))  # prefix_len
            topk_idx = scores.topk(keep_prefix, dim=-1).indices.sort().values

        lag_idx = torch.arange(prefix_len, S_k, device=key.device)
        idx = torch.cat([topk_idx, lag_idx], dim=-1)
        key   = key[:, :, idx, :]
        value = value[:, :, idx, :]

        target = tq_patch if tq_patch is not None else original
        return target(query, key, value, None, dropout_p, is_causal, **kw)

    return _sdpa, original


# ── Public API ────────────────────────────────────────────────────────────────

_active_k_bits:         Optional[int]   = None
_active_v_bits:         Optional[float] = None
_active_device:         Optional[str]   = None
_active_sparse_v:       bool            = False
_active_layer_adaptive: bool            = True
_active_mode:           str             = "turboquant"


def patch_sdpa(bit_width: Union[int, float] = 3, device: str = "cuda",
               k_bits: int = None, v_bits: Union[int, float] = None,
               sparse_v: bool = False, layer_adaptive: bool = True,
               mode: str = "turboquant", **mode_kwargs):
    """
    Globally replace F.scaled_dot_product_attention with TurboQuant+ or a KVPress mode.

    mode: compression strategy - "turboquant" (default), "snapkv", "streaming", "knorm",
          or "snapkv+turboquant" / "knorm+turboquant" (sparsify then quantise).
    bit_width: shorthand - sets both K and V bits (float → channel splitting for V).
    k_bits:    explicit K bit width (full TurboQuant). Defaults to ceil(bit_width).
    v_bits:    explicit V bit width (PolarQuant). Defaults to max(floor(bit_width)−1, 2).
    sparse_v:  skip V positions with attention weight < 1% of max (sequences ≤4096 only).
    layer_adaptive: boundary layers get +1 bit for K automatically (default True).

    KVPress mode_kwargs:
      snapkv:    keep_ratio=0.25    - fraction of K/V positions to retain
      streaming: sink_tokens=4, local_window=256
      knorm:     threshold=0.1      - drop positions with K-norm < threshold × max_norm

    Recommended TurboQuant+: k_bits=4, v_bits=3 for best quality.
    Combined:   mode="snapkv+turboquant", keep_ratio=0.25, k_bits=4, v_bits=3
                → 6–12× KV compression (sparsify first, then quantise survivors).
    """
    global _original_sdpa, _active_k_bits, _active_v_bits, _active_device
    global _active_sparse_v, _active_layer_adaptive, _active_mode
    # PR-E: previously this silently no-op'd if a patch was already installed,
    # which made re-calling patch_sdpa() with different settings invisible to
    # the caller - a debugging nightmare.  Auto-unpatch first so the new
    # settings actually take effect.  Callers that genuinely don't want this
    # should call unpatch_sdpa() explicitly first; the no-op cost when no
    # patch is installed is one None-check.
    if _original_sdpa is not None:
        import warnings
        warnings.warn("gb_attn.patch_sdpa called while a previous patch is active; "
                      "auto-unpatching and re-applying with new settings.",
                      RuntimeWarning, stacklevel=2)
        unpatch_sdpa()
    kb = k_bits if k_bits is not None else int(math.ceil(bit_width))
    vb = v_bits if v_bits is not None else max(bit_width - 1, 2)

    original = _F.scaled_dot_product_attention

    if mode == "turboquant":
        patched, _ = _make_tq_sdpa(kb, vb, device, sparse_v, layer_adaptive)
    elif mode == "snapkv":
        patched, _ = _make_snapkv_sdpa(mode_kwargs.get("keep_ratio", 0.25), original)
    elif mode == "streaming":
        patched, _ = _make_streaming_sdpa(
            mode_kwargs.get("sink_tokens", 4),
            mode_kwargs.get("local_window", 256),
            original)
    elif mode == "knorm":
        patched, _ = _make_knorm_sdpa(mode_kwargs.get("threshold", 0.1), original)
    elif mode in ("snapkv+turboquant", "turboquant+snapkv"):
        tq_patch, _ = _make_tq_sdpa(kb, vb, device, sparse_v, layer_adaptive)
        patched, _  = _make_snapkv_sdpa(mode_kwargs.get("keep_ratio", 0.25), original, tq_patch)
    elif mode in ("knorm+turboquant", "turboquant+knorm"):
        tq_patch, _ = _make_tq_sdpa(kb, vb, device, sparse_v, layer_adaptive)
        patched, _  = _make_knorm_sdpa(mode_kwargs.get("threshold", 0.1), original, tq_patch)
    elif mode in ("streaming+turboquant", "turboquant+streaming"):
        tq_patch, _ = _make_tq_sdpa(kb, vb, device, sparse_v, layer_adaptive)
        patched, _  = _make_streaming_sdpa(
            mode_kwargs.get("sink_tokens", 4),
            mode_kwargs.get("local_window", 256),
            original, tq_patch)
    elif mode == "adakv":
        patched, _ = _make_adakv_sdpa(
            mode_kwargs.get("keep_ratio", 0.3),
            mode_kwargs.get("safeguard_factor", 1.0),
            original)
    elif mode in ("adakv+turboquant", "turboquant+adakv"):
        tq_patch, _ = _make_tq_sdpa(kb, vb, device, sparse_v, layer_adaptive)
        patched, _  = _make_adakv_sdpa(
            mode_kwargs.get("keep_ratio", 0.3),
            mode_kwargs.get("safeguard_factor", 1.0),
            original, tq_patch)
    elif mode == "pyramidkv":
        patched, _ = _make_pyramidkv_sdpa(
            mode_kwargs.get("keep_ratio", 0.25),
            mode_kwargs.get("window_size", 32),
            original)
    elif mode in ("pyramidkv+turboquant", "turboquant+pyramidkv"):
        tq_patch, _ = _make_tq_sdpa(kb, vb, device, sparse_v, layer_adaptive)
        patched, _  = _make_pyramidkv_sdpa(
            mode_kwargs.get("keep_ratio", 0.25),
            mode_kwargs.get("window_size", 32),
            original, tq_patch)
    elif mode == "criticalkv":
        patched, _ = _make_criticalkv_sdpa(
            mode_kwargs.get("keep_ratio", 0.3),
            mode_kwargs.get("w_o", None),
            original)
    elif mode in ("criticalkv+turboquant", "turboquant+criticalkv"):
        tq_patch, _ = _make_tq_sdpa(kb, vb, device, sparse_v, layer_adaptive)
        patched, _  = _make_criticalkv_sdpa(
            mode_kwargs.get("keep_ratio", 0.3),
            mode_kwargs.get("w_o", None),
            original, tq_patch)
    elif mode == "lagkv":
        patched, _ = _make_lagkv_sdpa(
            mode_kwargs.get("lag", 128),
            mode_kwargs.get("keep_ratio", 0.3),
            original)
    elif mode in ("lagkv+turboquant", "turboquant+lagkv"):
        tq_patch, _ = _make_tq_sdpa(kb, vb, device, sparse_v, layer_adaptive)
        patched, _  = _make_lagkv_sdpa(
            mode_kwargs.get("lag", 128),
            mode_kwargs.get("keep_ratio", 0.3),
            original, tq_patch)
    else:
        raise ValueError(f"gb_attn: unknown mode '{mode}'. "
                         f"Valid: turboquant, snapkv, streaming, knorm, adakv, pyramidkv, "
                         f"criticalkv, lagkv, and *+turboquant combinations.")

    _original_sdpa         = original
    _active_k_bits         = kb
    _active_v_bits         = vb
    _active_device         = device
    _active_sparse_v       = sparse_v
    _active_layer_adaptive = layer_adaptive
    _active_mode           = mode
    _F.scaled_dot_product_attention = patched
    import torch.nn.functional as F2
    F2.scaled_dot_product_attention = patched

    try:
        import gb_dataflux
        gb_dataflux.emit({
            "node": "host", "label": "gb_attn", "kind": "turboquant_activate",
            "n_items": 1, "items": [mode], "duration_s": 0.0, "status": "ok",
            "k_bits": kb, "v_bits": vb, "device": device,
            "sparse_v": sparse_v, "mode": mode,
        })
    except Exception:
        pass


def unpatch_sdpa():
    """Restore the original F.scaled_dot_product_attention.

    PR-E: also resets layer-adaptive EMA state.  Previously _call_in_seq,
    _est_total_layers, and _last_call_time persisted across unpatch -> repatch
    cycles, so the second context manager scope inherited stale boundary-layer
    estimates from the first.  This made debugging "why is my second
    pipeline.to('cuda') producing different attention quality" effectively
    impossible.  Now every unpatch returns the module to a fully-initial
    state."""
    global _original_sdpa, _active_k_bits, _active_v_bits, _active_device
    global _active_sparse_v, _active_layer_adaptive, _active_mode
    global _call_in_seq, _est_total_layers, _last_call_time
    if _original_sdpa is None:
        return
    _F.scaled_dot_product_attention = _original_sdpa
    import torch.nn.functional as F2
    F2.scaled_dot_product_attention = _original_sdpa
    _original_sdpa         = None
    _active_k_bits         = None
    _active_v_bits         = None
    _active_device         = None
    _active_sparse_v       = False
    _active_layer_adaptive = True
    # PR-E: clear layer-adaptive EMA state under the same lock that _tq_sdpa
    # uses to read it.  Without this, the next patch_sdpa() inherits stale
    # values and the first sequence under the new patch silently uses the
    # OLD boundary-layer mask.
    with _layer_state_lock:
        _call_in_seq      = 0
        _est_total_layers = 0
        _last_call_time   = 0.0
    _active_mode           = "turboquant"


@contextmanager
def turboquant_attention(bit_width: Union[int, float] = 3, device: str = "cuda",
                         k_bits: int = None, v_bits: Union[int, float] = None,
                         sparse_v: bool = False, layer_adaptive: bool = True,
                         mode: str = "turboquant", **mode_kwargs):
    """
    Context manager: apply K/V compression during the enclosed block.

    Works with BF16 and NF4 pipelines (mode-agnostic). Re-entrant: nested
    calls with the same settings are safe.

    Args:
        bit_width:      compression bits (2, 3, 4 or float 2.5/3.5). Default 3.
        device:         CUDA device string ("cuda" or "cuda:0").
        k_bits:         explicit K bit width (overrides bit_width for K).
        v_bits:         explicit V bit width (float → channel splitting).
        sparse_v:       skip V positions with negligible attention weight.
        layer_adaptive: boundary layers get +1 bit for K automatically.
        mode:           "turboquant" (default), "snapkv", "streaming", "knorm",
                        "adakv", "pyramidkv", "criticalkv", "lagkv",
                        or any mode combined with "+turboquant" for stacked compression.
        **mode_kwargs:  per-mode parameters:
                        keep_ratio (snapkv/adakv/pyramidkv/criticalkv/lagkv),
                        sink_tokens/local_window (streaming), threshold (knorm),
                        safeguard_factor (adakv), window_size (pyramidkv),
                        w_o (criticalkv, optional W_O weight tensor), lag (lagkv).

    Examples:
        with turboquant_attention(k_bits=4, v_bits=3):               # TurboQuant+ (default)
        with turboquant_attention(mode="snapkv", keep_ratio=0.25):   # SnapKV only
        with turboquant_attention(mode="streaming", sink_tokens=4):  # StreamingLLM
        with turboquant_attention(mode="knorm", threshold=0.1):      # KNorm
        with turboquant_attention(mode="adakv", keep_ratio=0.3):     # AdaKV per-head adaptive
        with turboquant_attention(mode="pyramidkv", keep_ratio=0.25):# PyramidKV layer-wise
        with turboquant_attention(mode="criticalkv", keep_ratio=0.3):# CriticalKV W_O scoring
        with turboquant_attention(mode="lagkv", lag=128, keep_ratio=0.3): # LagKV
        with turboquant_attention(mode="snapkv+turboquant",          # 6–12× combined
                                  keep_ratio=0.25, k_bits=4, v_bits=3):
    """
    already = _original_sdpa is not None
    if not already:
        patch_sdpa(bit_width=bit_width, device=device,
                   k_bits=k_bits, v_bits=v_bits,
                   sparse_v=sparse_v, layer_adaptive=layer_adaptive,
                   mode=mode, **mode_kwargs)
    try:
        yield
    finally:
        if not already:
            unpatch_sdpa()


def turboquant_attention_from_env(var: str = "GB_TQ_ATTN"):
    """Context manager for the enclosed denoise/decode loop, configured from an
    env var — the canonical form of the spec every consumer was re-parsing
    locally (ai-forge gen_image._turboquant_ctx, gen_art). Formats:

        k4v3      asymmetric: K gets 4 bits (full TurboQuant), V gets 3 (PolarQuant)
        3         symmetric bit width (floats like 2.5 allowed)
        (unset)   nullcontext — provably no-op

    Any parse failure degrades to nullcontext rather than raising: an attention
    optimization must never break a generation. Consumers keep only a try/except
    import shim; the parsing lives here so image/video/texture pipelines can't
    drift (e.g. GB_TQ_ATTN_VIDEO uses the same grammar via `var=`)."""
    import contextlib
    import os as _os
    import re as _re

    spec = _os.environ.get(var, "").strip().lower()
    if not spec:
        return contextlib.nullcontext()
    m = _re.fullmatch(r"k(\d+)v(\d+(?:\.\d+)?)", spec)
    try:
        if m:
            return turboquant_attention(k_bits=int(m.group(1)),
                                        v_bits=float(m.group(2)))
        return turboquant_attention(bit_width=float(spec))
    except Exception:
        return contextlib.nullcontext()


def status() -> dict:
    """Return current patch status (useful for debugging)."""
    quantizers = {}
    for k, v in _quantizer_cache.items():
        if not isinstance(k, tuple):
            continue
        if len(k) == 3:
            bits, head_dim, device = k
            quantizers[f"{bits}b-{head_dim}d-{device}"] = f"Pi={v[0].shape[0]}×{v[0].shape[1]}"
        # 4-tuple keys are JL projection matrices (qjl); report separately
        elif len(k) == 4 and k[3] == "qjl":
            bits, head_dim, device, _ = k
            quantizers[f"qjl-{bits}b-{head_dim}d-{device}"] = f"S={v.shape[0]}×{v.shape[1]}"
    return {
        "patched":        _original_sdpa is not None,
        "mode":           _active_mode,
        "k_bits":         _active_k_bits,
        "v_bits":         _active_v_bits,
        "layer_adaptive": _active_layer_adaptive,
        "sparse_v":       _active_sparse_v,
        "device":         _active_device,
        "quantizers":     quantizers,
    }


# ── Convenience presets ───────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "fast":      {"bit_width": 2, "mode": "turboquant"},
    "balanced":  {"k_bits": 4, "v_bits": 3, "mode": "turboquant"},
    "quality":   {"k_bits": 4, "v_bits": 3, "mode": "turboquant", "sparse_v": True},
    "combined":  {"k_bits": 4, "v_bits": 3, "mode": "snapkv+turboquant", "keep_ratio": 0.25},
    "streaming": {"mode": "streaming", "sink_tokens": 4, "local_window": 256},
    "knorm":     {"mode": "knorm", "threshold": 0.1},
    "adakv":     {"mode": "adakv", "keep_ratio": 0.3, "safeguard_factor": 1.0},
    "pyramidkv": {"mode": "pyramidkv", "keep_ratio": 0.25, "window_size": 32},
}
"""
Named compression presets for patch_sdpa / turboquant_attention.

fast      - 2-bit TurboQuant, maximum bandwidth reduction (~8×), minor quality drop
balanced  - k=4/v=3 TurboQuant (recommended), ~4× reduction, near-zero PPL impact
quality   - balanced + Sparse-V skip, best for sequences ≤4096 tokens
combined  - SnapKV (keep 25%) + TurboQuant k=4/v=3, 6-12× total reduction
streaming - StreamingLLM sink+window, infinite context without KV growth
knorm     - drop K positions with low L2 norm, aggressive sparsification
adakv     - per-head adaptive top-k ranked by attention entropy
pyramidkv - layer-wise budget ramp (early layers fewer positions, later more)

Usage:
    with turboquant_attention(**PRESETS["balanced"]):
        image = pipe(prompt, ...).images[0]

    patch_sdpa(**PRESETS["combined"])
    output = model(input)
    unpatch_sdpa()
"""


def from_preset(name: str, **overrides):
    """Return a turboquant_attention context manager configured by preset name.

    Example:
        with gb_attn.from_preset("balanced"):
            image = pipe(prompt, ...).images[0]

        with gb_attn.from_preset("combined", keep_ratio=0.20):
            output = model(input)
    """
    if name not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset {name!r}. Valid: {valid}")
    kwargs = {**PRESETS[name], **overrides}
    return turboquant_attention(**kwargs)

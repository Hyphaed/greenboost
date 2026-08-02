# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_tier_kv.py , KV-cache tier-serde compression (missing_features.md item (e)).

Compresses KV-cache-shaped tensors at the T2/T3 tier-move boundary
(`gb_model_tier._t3_save`/`_t3_load`), not inside the attention call itself
(the mechanism `gb_attn.py`'s TurboQuant K/V patch uses). This is the
actually-reachable mechanism for models that never call
`F.scaled_dot_product_attention` directly (flash-attn-direct models, e.g.
LongLive , see missing_features.md Appendix A for the full reachability
analysis gb_attn's own monkeypatch can't cover). Operating at the tier-move
boundary works for ANY model regardless of its attention implementation,
because it compresses whatever bytes get spilled to T3 NVMe, below the
model's own code entirely.

Reuses gb_attn.py's tensor-generic PolarQuant/QJL primitives (`_get_quantizer`,
`_quantize`/`_dequantize`, `_quantize_k_turbo`/`_dequantize_k_turbo`) , they
take plain tensors and an explicit device string, with no dependency on
`patch_sdpa`'s global state.

Public API:
    encode_state(state, bits, keys=None, codec=None, device="cpu") -> (state, stats)
    decode_state(state) -> state

Both operate on a plain `state_dict`-shaped `{str: Tensor}` mapping (plus,
after encode, one reserved "__gb_kv__" manifest key). `decode_state` returns
its argument BY IDENTITY when the manifest key is absent , every existing
`.pt`/`.pt.zst` T3 checkpoint on disk today is completely unaffected.

Classification is per-register opt-in ONLY (see gb_model_tier.register's
kv_bits/kv_keys kwargs) , this module never guesses which tensors are
KV-cache-shaped from their name or shape alone. Guessing here would mean
silently lossy-compressing model WEIGHTS, which is unrecoverable and
invisible until output quality degrades. `_kv_eligible` is a SAFETY net for
tensors that are provably wrong to touch (non-float, wrong rank, unknown
head_dim, too small to be worth it, non-finite) , it narrows an
already-opted-in module's state_dict, it does not decide opt-in on its own.

Env:
    GB_TIER_KV_COMPRESS=0   , kill switch (mirrors GB_T3_COMPRESS). Decode
                              stays capability-driven regardless (an
                              already-encoded file on disk always decodes),
                              same rule _t3_load already applies to zstd.
    GB_TIER_KV_PRESET       , codec family when a caller doesn't pass one
                              explicitly: "polar4"/"polar8" -> "polarquant",
                              "turbo_k8v4" -> "turboquant", "off" -> disabled.
                              Chooses the ALGORITHM only; bit width always
                              comes from the caller's explicit kv_bits (a
                              preset never supplies its own bit count, so it
                              can never silently override an explicit
                              opt-in). Default "polar4".
    GB_TIER_KV_MIN_ELEMS    , eligibility floor (default 65536 elements).
"""
from __future__ import annotations

import os
import warnings
from typing import Dict, Iterable, Optional, Tuple

import torch

# Format/codebook invariants , recorded in every manifest and checked on
# decode. gb_attn._get_quantizer's rotation seed is hardcoded (42); if that
# ever changes, _FORMAT_VERSION must bump too, so an old on-disk manifest is
# refused rather than silently decoded against a different rotation matrix
# (see test_gb_model_tier_kv.py's determinism/mismatch tests).
_FORMAT_VERSION = 1
_PI_SEED = 42

_ALLOWED_HEAD_DIMS = frozenset({64, 80, 96, 112, 128, 160, 192, 256})
_VALID_BITS = (2, 3, 4, 8)

_PRESETS: Dict[str, Optional[str]] = {
    "polar4": "polarquant",
    "polar8": "polarquant",
    "turbo_k8v4": "turboquant",
    "off": None,
}


def _min_elems() -> int:
    try:
        return int(os.environ.get("GB_TIER_KV_MIN_ELEMS", "65536"))
    except ValueError:
        return 65536


def resolve_codec(explicit_codec: "Optional[str]" = None) -> "Optional[str]":
    """codec family: explicit arg > GB_TIER_KV_PRESET env > default
    "polarquant". Returns None when compression is disabled
    (GB_TIER_KV_COMPRESS=0, or the resolved preset is "off")."""
    if os.environ.get("GB_TIER_KV_COMPRESS", "1") == "0":
        return None
    if explicit_codec is not None:
        return explicit_codec
    name = os.environ.get("GB_TIER_KV_PRESET", "polar4")
    if name not in _PRESETS:
        warnings.warn(
            f"gb_tier_kv: unknown GB_TIER_KV_PRESET={name!r}, falling back to 'polar4'")
        name = "polar4"
    return _PRESETS[name]


def _kv_eligible(t) -> "Optional[str]":
    """Return a skip-reason string, or None when `t` may be quantized. A
    per-tensor safety guard , see module docstring for why classification
    itself is never done here (opt-in only, at gb_model_tier.register)."""
    if not isinstance(t, torch.Tensor):
        return "not_tensor"
    if t.is_meta or t.is_sparse:
        return "meta_or_sparse"
    if torch.is_complex(t):
        return "complex"
    if not t.is_floating_point():
        return "non_float"          # int/bool/long buffers (e.g. position ids)
    if t.dim() < 2:
        return "rank_lt_2"          # no head_dim axis to quantize against
    if t.shape[-1] not in _ALLOWED_HEAD_DIMS:
        return "head_dim"
    if t.numel() < _min_elems():
        return "too_small"          # overhead (norms + manifest) > saving
    if not torch.isfinite(t).all():
        return "nonfinite"          # norms/centroids clamp -> garbage on decode
    return None


def _encode_tensor(t: torch.Tensor, bits: int, codec: str, device: str
                   ) -> "Tuple[Dict[str, torch.Tensor], list]":
    """Returns ({part_name: tensor}, [part_names in manifest order])."""
    import gb_attn
    src = t.detach().to(device)
    if codec == "polarquant":
        D = src.shape[-1]
        Pi, Pi_T, centroids, boundaries, _ = gb_attn._get_quantizer(bits, D, device)
        idx, norms = gb_attn._quantize(src, Pi_T, boundaries)
        return {"idx": idx.contiguous(), "norms": norms.contiguous()}, ["idx", "norms"]
    if codec == "turboquant":
        idx, norms, signs = gb_attn._quantize_k_turbo(src, bits, device)
        return ({"idx": idx.contiguous(), "norms": norms.contiguous(),
                 "signs": signs.contiguous()}, ["idx", "norms", "signs"])
    raise ValueError(f"gb_tier_kv: unknown codec {codec!r}")


def _decode_tensor(parts: "Dict[str, torch.Tensor]", meta: dict, codec: str,
                   bits: int, device: str) -> torch.Tensor:
    import gb_attn
    shape = tuple(meta["shape"])
    dtype = getattr(torch, meta["dtype"])
    if codec == "polarquant":
        Pi, _, centroids, _, _ = gb_attn._get_quantizer(bits, meta["head_dim"], device)
        rec = gb_attn._dequantize(parts["idx"], parts["norms"], Pi, centroids)
    elif codec == "turboquant":
        rec = gb_attn._dequantize_k_turbo(
            parts["idx"], parts["norms"], parts["signs"], bits, device)
    else:
        raise ValueError(f"gb_tier_kv: unknown codec {codec!r} in manifest")
    return rec.to(dtype).reshape(shape)


def encode_state(state: dict, bits: int, keys: "Optional[Iterable[str]]" = None,
                 codec: "Optional[str]" = None, device: str = "cpu"
                 ) -> "Tuple[dict, dict]":
    """Quantize the KV-cache-shaped tensors in `state` (a state_dict-shaped
    mapping). Non-eligible / non-matching tensors pass through UNCHANGED and
    round-trip bit-exactly , only tensors this function actually touches are
    replaced.

    Args:
        state: {str: Tensor} mapping (a real state_dict works as-is).
        bits: codebook bit width, one of (2, 3, 4, 8) , see gb_kernel_backends
            for why these are the only packable widths.
        keys: restrict to keys starting with one of these prefixes (mirrors
            gb_model_tier.register's kv_keys). None = every eligible tensor
            in `state`.
        codec: "polarquant" (PolarQuant only, MSE-optimal, V-style) or
            "turboquant" (PolarQuant + QJL 1-bit residual, inner-product
            preserving, K-style). None -> resolve_codec()'s env-driven
            default.
        device: quantizer compute device , "cpu" for CPU-only T3 saves
            (the default and normal case, no GPU required), "cuda" only if
            the caller already holds the tensor on GPU.

    Returns:
        (new_state, stats) , stats: {"codec", "bits", "tensors",
        "tensors_skipped", "skip_reasons", "pre_bytes", "post_bytes"}.
        new_state IS state (same object) when nothing was encoded (disabled,
        no eligible tensors, or every candidate failed its own economics).
    """
    resolved_codec = resolve_codec(codec)
    stats: dict = {
        "codec": resolved_codec, "bits": bits, "tensors": 0,
        "tensors_skipped": 0, "skip_reasons": {}, "pre_bytes": 0, "post_bytes": 0,
    }
    if resolved_codec is None or not state:
        return state, stats
    if bits not in _VALID_BITS:
        raise ValueError(f"gb_tier_kv: bits={bits!r} not in {_VALID_BITS}")

    def _skip(reason: str) -> None:
        stats["tensors_skipped"] += 1
        stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1

    new_state = dict(state)
    manifest_entries: Dict[str, dict] = {}
    for name, t in state.items():
        if keys is not None and not any(name.startswith(k) for k in keys):
            continue
        reason = _kv_eligible(t)
        if reason is not None:
            _skip(reason)
            continue
        raw_bytes = t.numel() * t.element_size()
        try:
            parts, part_order = _encode_tensor(t, bits, resolved_codec, device)
        except Exception:
            _skip("encode_error")
            continue
        enc_bytes = sum(p.numel() * p.element_size() for p in parts.values())
        if enc_bytes >= raw_bytes:
            _skip("not_smaller")
            continue
        del new_state[name]
        for part_name, tensor in parts.items():
            new_state[f"__gb_kv__/{name}/{part_name}"] = tensor
        manifest_entries[name] = {
            "shape": list(t.shape), "dtype": str(t.dtype).rsplit(".", 1)[-1],
            "head_dim": t.shape[-1], "numel": t.numel(), "parts": part_order,
        }
        stats["tensors"] += 1
        stats["pre_bytes"] += raw_bytes
        stats["post_bytes"] += enc_bytes

    if not manifest_entries:
        return state, stats
    new_state["__gb_kv__"] = {
        "v": _FORMAT_VERSION, "codec": resolved_codec, "pi_seed": _PI_SEED,
        "bits": bits, "entries": manifest_entries,
    }
    return new_state, stats


def decode_state(state: dict) -> dict:
    """Reverse of encode_state. Returns `state` BY IDENTITY when no
    "__gb_kv__" manifest is present (every legacy checkpoint , the common
    case). Raises on a version/rotation-seed mismatch rather than silently
    decoding against the wrong quantizer (see module docstring)."""
    manifest = state.get("__gb_kv__")
    if manifest is None:
        return state
    if manifest.get("v") != _FORMAT_VERSION:
        raise ValueError(
            f"gb_tier_kv: manifest format v{manifest.get('v')!r} != "
            f"v{_FORMAT_VERSION!r} , refusing to decode (format changed)")
    if manifest.get("pi_seed") != _PI_SEED:
        raise ValueError(
            f"gb_tier_kv: manifest pi_seed={manifest.get('pi_seed')!r} != "
            f"{_PI_SEED!r} , the quantizer rotation would not match, "
            "refusing to decode rather than return garbage tensors")

    codec = manifest["codec"]
    bits = manifest["bits"]
    new_state = dict(state)
    del new_state["__gb_kv__"]
    for name, meta in manifest["entries"].items():
        parts = {}
        for part_name in meta["parts"]:
            key = f"__gb_kv__/{name}/{part_name}"
            parts[part_name] = new_state.pop(key)
        new_state[name] = _decode_tensor(parts, meta, codec, bits, device="cpu")
    return new_state

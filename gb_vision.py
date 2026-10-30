#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""gb_vision.py - vision projector sizing and image-input normalisation for gb-synapse.

Why this exists (live incidents, 2026-10-28, RTX 5070 12 GB, Qwen3.8-35B-A3B + mmproj):

  * The serving recipe pins nCpuMoe=28. That number was measured WITHOUT the projector
    and left ~0.8 GB of the card free once the projector was loaded. The first image
    request then ran llama-server's vision encoder out of CUDA memory and killed the
    engine ("CUDA error: out of memory" in clip_image_batch_encode, HTTP 502 upstream).
    nCpuMoe=32 (9.5 GB) works. Callers had started passing n_cpu_moe themselves, which is
    hardware-shaped logic in the wrong layer.
  * Qwen-VL class projectors need >= 1024 image tokens to be accurate (the engine warns:
    "Qwen-VL models require at minimum 1024 image tokens to function correctly"). A 16:9
    frame shrunk by a client to 896 px is ~575 tokens, and the model answered "no eyes, no
    hands, no shoes" for frames that clearly have them. Small crops need to be UPSCALED.
  * The proxy rejected any request body over 1 MiB (aiohttp's default client_max_size,
    HTTP 413), so each client grew its own resize workaround.

This module owns the arithmetic for all three, derived from the projector's own GGUF
metadata (no reference-box literals; the one physical constant, the 1024-token floor, is
keyed by the DETECTED projector type like the PCIe tables). It never touches the GPU.

Public surface
--------------
projector_meta(path)            GGUF clip.* metadata (empty dict when unreadable)
profile_from_meta(...)          VisionProfile (weights, token budget, encoder geometry)
vision_profile(path)            projector_meta + profile_from_meta
encoder_compute_gb(profile)     encoder activation buffers for the image token budget
extra_vram_gb(profile)          what the projector costs on the card (weights + buffers)
extra_cpu_moe_layers(...)       how many MoE layers' experts that cost converts into
cmd_flags(profile)              --image-min-tokens / --image-max-tokens for the engine
ImageBudget, budget_from_profile, target_size
normalize_image_bytes / normalize_data_url / normalize_openai_messages
VISION_FIT                      thread-local carrying the plan from serve() to the backend
"""
from __future__ import annotations

import base64
import io
import math
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

GIB = 1024 ** 3

# Projector types whose image size is dynamic (resolution-adaptive merger): the engine needs a
# token floor for them. Keyed by the detected clip.projector_type, a physical property of the
# architecture, not of the box (same standing as the PCIe per-lane table keyed by detected gen).
_DYNAMIC_PROJECTORS = ("qwen",)
# The Qwen-VL floor the engine itself prints ("require at minimum 1024 image tokens").
_QWEN_MIN_TOKENS = 1024
# Aim a little above the floor: ~1.25x the floor is ~1 Mpx for a 14 px patch merged 2x2.
_TARGET_OVER_MIN = 1.25


def _env_int(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, "") or default)
        return v if v > 0 else default
    except ValueError:
        return default


def max_body_bytes() -> int:
    """Largest request body the proxy accepts (GB_SYNAPSE_MAX_BODY_MB, default 32 MiB): several normalised images fit,
    a runaway upload does not."""
    return _env_int("GB_SYNAPSE_MAX_BODY_MB", 32) * (1 << 20)


def _env_float(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, "") or default)
        return v if v > 0 else default
    except ValueError:
        return default


# --------------------------------------------------------------------------- profile


@dataclass
class VisionProfile:
    path: str = ""
    weights_gb: float = 0.0
    projector_type: str = ""
    patch_size: int = 14
    merge: int = 2                  # spatial merge: merge x merge patches become one token
    n_embd: int = 0
    n_head: int = 0
    n_ffn: int = 0
    n_layer: int = 0
    min_tokens: int = 0             # 0 = fixed-size projector (no floor to pass)
    max_tokens: int = 0
    target_tokens: int = 0
    source: str = "default"         # "gguf" when the numbers came from the projector's metadata

    @property
    def px_per_token(self) -> int:
        return max(1, (self.patch_size * self.merge) ** 2)

    @property
    def dynamic(self) -> bool:
        return self.min_tokens > 0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["px_per_token"] = self.px_per_token
        return d


_NUMERIC_KEYS = ("clip.vision.patch_size", "clip.vision.spatial_merge_size", "clip.vision.embedding_length",
                 "clip.vision.attention.head_count", "clip.vision.feed_forward_length", "clip.vision.block_count",
                 "clip.vision.image_size", "clip.vision.image_max_pixels", "clip.vision.image_min_pixels")


def projector_meta(path: "str | Path") -> dict:
    """clip.* scalar metadata of a projector GGUF. {} when the file or the reader is unavailable."""
    try:
        from gb_gguf_tensor_map import _load_gguf_reader  # vendored llama.cpp GGUFReader
        reader = _load_gguf_reader()(str(path), mode="r")
    except Exception:
        return {}
    out: dict = {}
    for key in _NUMERIC_KEYS:
        f = reader.get_field(key)
        if f is not None and f.parts:
            try:
                out[key] = f.parts[-1][0].item()          # same access as gb_synapse._field_scalar
            except Exception:
                continue
    f = reader.get_field("clip.projector_type")
    if f is not None and f.parts:
        try:
            out["clip.projector_type"] = bytes(f.parts[-1]).decode("utf-8", errors="replace")   # same as _field_str
        except Exception:
            pass
    return out


def _first(meta: dict, *keys, default=0):
    for k in keys:
        v = meta.get(k)
        if v not in (None, "", 0):
            return v
    return default


def profile_from_meta(meta: dict, weights_bytes: int = 0, path: str = "") -> VisionProfile:
    """Build the profile from clip.* metadata. Missing keys fall back to the Qwen-VL geometry the
    engine itself assumes; `source` says which one you got."""
    ptype = str(_first(meta, "clip.projector_type", "clip.vision.projector_type", default="")).lower()
    patch = int(_first(meta, "clip.vision.patch_size", default=14))
    merge = int(_first(meta, "clip.vision.spatial_merge_size", default=2 if any(k in ptype for k in _DYNAMIC_PROJECTORS) else 1))
    n_embd = int(_first(meta, "clip.vision.embedding_length", default=0))
    n_head = int(_first(meta, "clip.vision.attention.head_count", default=0)) or (n_embd // 64 if n_embd else 0)
    n_ffn = int(_first(meta, "clip.vision.feed_forward_length", default=0)) or (4 * n_embd)
    n_layer = int(_first(meta, "clip.vision.block_count", default=0))
    prof = VisionProfile(path=path, weights_gb=weights_bytes / GIB, projector_type=ptype, patch_size=patch, merge=merge,
                         n_embd=n_embd, n_head=n_head, n_ffn=n_ffn, n_layer=n_layer,
                         source="gguf" if meta else "default")
    px = prof.px_per_token
    if any(k in ptype for k in _DYNAMIC_PROJECTORS) or (not meta and not ptype):
        # An unreadable projector is treated as the dynamic class: the safe side is to size for the larger buffers.
        floor = _env_int("GB_SYNAPSE_IMAGE_MIN_TOKENS", _QWEN_MIN_TOKENS)
        max_px = int(_first(meta, "clip.vision.image_max_pixels", default=0))
        target = max(floor, int(math.ceil(floor * _TARGET_OVER_MIN)))
        cap = _env_int("GB_SYNAPSE_IMAGE_MAX_TOKENS", max(target, (max_px // px) if max_px else target))
        prof.min_tokens, prof.max_tokens = floor, max(cap, floor)
        prof.target_tokens = min(target, prof.max_tokens)
    else:
        size = int(_first(meta, "clip.vision.image_size", default=0))
        fixed = (size // max(1, patch)) ** 2 // max(1, merge ** 2) if size else 0
        prof.min_tokens = 0
        prof.max_tokens = prof.target_tokens = fixed
    return prof


def vision_profile(mmproj: "str | Path") -> VisionProfile:
    p = Path(mmproj)
    try:
        size = p.stat().st_size
    except OSError:
        size = 0
    return profile_from_meta(projector_meta(p), size, str(p))


# --------------------------------------------------------------------------- VRAM plan


def encoder_compute_gb(profile: VisionProfile, tokens: "int | None" = None) -> float:
    """Activation buffers the vision encoder needs for an image of `tokens` LLM tokens (default: the
    profile's largest). The encoder sees merge^2 patches per token; the dominant term is the attention
    score matrix (patches^2 x heads, f16), plus the FFN and residual activations. Calibrated on the
    2026-10-28 incident: ~1275 tokens needed ~1.5 GB free beyond the loaded weights, 0.8 GB was not enough.
    GB_SYNAPSE_VISION_MARGIN (default 1.25) scales the estimate."""
    tokens = tokens or profile.max_tokens or profile.target_tokens
    if tokens <= 0:
        return 0.0
    patches = tokens * profile.merge ** 2
    heads = profile.n_head or 16
    embd = profile.n_embd or 1152
    ffn = profile.n_ffn or 4 * embd
    attn = patches * patches * heads * 2.0
    acts = patches * (ffn + 3 * embd) * 4.0
    return (attn + acts) * _env_float("GB_SYNAPSE_VISION_MARGIN", 1.25) / GIB


def extra_vram_gb(profile: VisionProfile) -> float:
    """What the projector costs on the card: its weights plus the encoder buffers for the token budget."""
    return profile.weights_gb + encoder_compute_gb(profile)


def extra_cpu_moe_layers(extra_gb: float, per_layer_expert_gb: float) -> int:
    """MoE layers whose experts must move to host RAM to pay for `extra_gb` of VRAM. 0 when not a MoE."""
    if extra_gb <= 0 or per_layer_expert_gb <= 0:
        return 0
    return int(math.ceil(extra_gb / per_layer_expert_gb))


def cmd_flags(profile: "VisionProfile | None") -> list:
    """Engine flags for a dynamic-resolution projector (empty for fixed-size ones)."""
    if profile is None or not profile.dynamic:
        return []
    return ["--image-min-tokens", str(profile.min_tokens), "--image-max-tokens", str(profile.max_tokens)]


class _FitCtx(threading.local):
    """Plan handed from serve() to the backend that builds the engine command line (the backend
    signature is shared by every engine, so the vision plan travels on a thread-local, the same way
    the load trigger does)."""
    mmproj: str = ""
    profile: "VisionProfile | None" = None
    extra_gb: float = 0.0


VISION_FIT = _FitCtx()


def set_fit(mmproj: str = "", profile: "VisionProfile | None" = None, extra_gb: float = 0.0) -> None:
    VISION_FIT.mmproj, VISION_FIT.profile, VISION_FIT.extra_gb = mmproj, profile, extra_gb


def clear_fit() -> None:
    set_fit("", None, 0.0)


# --------------------------------------------------------------------------- image input


@dataclass
class ImageBudget:
    target_tokens: int = 1280
    min_tokens: int = 1024
    max_tokens: int = 1280
    px_per_token: int = 784          # (14 * 2)^2 for Qwen-VL
    max_side: int = 0                # 0 = derived
    max_bytes: int = 0               # 0 = derived

    def __post_init__(self):
        if not self.max_side:
            # about 1.5x the side of a square of the target area
            self.max_side = int(math.sqrt(self.target_tokens * self.px_per_token) * 1.5)
        if not self.max_bytes:
            self.max_bytes = _env_int("GB_SYNAPSE_IMAGE_MAX_BYTES", 4 * (1 << 20))

    @property
    def target_px(self) -> int:
        return self.target_tokens * self.px_per_token

    def as_dict(self) -> dict:
        return asdict(self)


def budget_from_profile(profile: "VisionProfile | None") -> "ImageBudget | None":
    """None for a fixed-size projector (it resizes internally; nothing to normalise to)."""
    if profile is None or not profile.dynamic:
        return None
    return ImageBudget(target_tokens=profile.target_tokens, min_tokens=profile.min_tokens, max_tokens=profile.max_tokens,
                       px_per_token=profile.px_per_token)


def budget_from_dict(d: "dict | None") -> "ImageBudget | None":
    if not d:
        return None
    try:
        return ImageBudget(**{k: d[k] for k in ImageBudget.__dataclass_fields__ if k in d})
    except Exception:
        return None


def target_size(w: int, h: int, b: ImageBudget) -> tuple:
    """Output size for a w x h image: about target_px pixels (UP or down), never below the token floor unless
    the longest-side cap forbids it, never above the cap, sides snapped to the projector's patch-merge unit."""
    if w <= 0 or h <= 0:
        return w, h
    unit = max(1, int(round(math.sqrt(b.px_per_token))))
    k = math.sqrt(b.target_px / (w * h))
    k = min(k, b.max_side / max(w, h))
    floor_px = b.min_tokens * b.px_per_token
    if (w * h * k * k) < floor_px:
        k = min(math.sqrt(floor_px / (w * h)), b.max_side / max(w, h))
    snap = lambda v: max(unit, int(round(v * k / unit)) * unit)  # noqa: E731
    return snap(w), snap(h)


def tokens_for(w: int, h: int, b: ImageBudget) -> int:
    return int(round(w * h / b.px_per_token))


def normalize_image_bytes(raw: bytes, b: ImageBudget) -> tuple:
    """(jpeg_bytes, info). Needs Pillow; without it the original bytes come back with info['skipped']."""
    try:
        from PIL import Image, ImageOps
    except Exception:
        return raw, {"skipped": "Pillow is not installed in the proxy's environment"}
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        w0, h0 = im.size
        w, h = target_size(w0, h0, b)
        if (w, h) != (w0, h0):
            im = im.resize((w, h), Image.LANCZOS)
        out = raw
        for q in (92, 85, 78, 70, 62):
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=q)
            out = buf.getvalue()
            if len(out) <= b.max_bytes:
                break
        return out, {"from": [w0, h0], "to": [w, h], "tokens": tokens_for(w, h, b), "bytes": len(out)}
    except Exception as e:                                   # a corrupt image must reach the engine's own error, not vanish here
        return raw, {"skipped": f"{type(e).__name__}: {e}"}


_DATA_URL = re.compile(r"^data:(?P<mime>[^;,]+)?(?P<b64>;base64)?,(?P<data>.*)$", re.S)


def normalize_data_url(url: str, b: ImageBudget) -> tuple:
    """(url, info). http(s) and file URLs pass through untouched (this stack is local-only; no fetching)."""
    m = _DATA_URL.match(url or "")
    if not m or not m.group("b64"):
        return url, {"skipped": "not a base64 data URL"}
    try:
        raw = base64.b64decode(m.group("data"), validate=False)
    except Exception:
        return url, {"skipped": "undecodable base64"}
    out, info = normalize_image_bytes(raw, b)
    if "skipped" in info:
        return url, info
    return "data:image/jpeg;base64," + base64.b64encode(out).decode(), info


def normalize_openai_messages(messages: list, b: "ImageBudget | None") -> tuple:
    """Copy of `messages` with every image_url data URL normalised, plus the per-image infos. No budget = unchanged."""
    if b is None or not isinstance(messages, list):
        return messages, []
    infos: list = []
    out = []
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list):
            out.append(m)
            continue
        parts = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "image_url" and isinstance(p.get("image_url"), dict):
                url, info = normalize_data_url(p["image_url"].get("url", ""), b)
                infos.append(info)
                parts.append({**p, "image_url": {**p["image_url"], "url": url}})
            else:
                parts.append(p)
        out.append({**m, "content": parts})
    return out, infos

"""gb_compile: regional torch.compile for diffusion / video transformers (the repeated transformer blocks), with a persistent inductor cache.

Measured 2026-10-29 on the RTX 5070 (FLUX.2-klein-4B, gb-quant fp8, shim on, 768 px, 4 steps, torch 2.13 cu130, 4 images per process):
  eager                      1.79 s / image steady, first image 3.6 s (warm Triton autotune cache)
  regional compile, warm     1.55 to 1.60 s / image (11 to 13% faster), first image 4.3 s (fullgraph=False) / 5.4 s (fullgraph=True)
  regional compile, cold     first image 13.2 s once; the cache (~/.cache/greenboost/inductor) makes every later process warm
  same seed: composition and quality unchanged, fine detail differs a little (31.4 dB between the two images: bf16 fusion changes the numbers, as any kernel change does)
So it pays back after about 5 to 8 images in ONE process, i.e. for batches (gen_image_batch.py), not for a single image per process.
Policy: GB_COMPILE=auto (default) compiles when the caller says it will make at least GB_COMPILE_MIN_IMAGES (8) images; =1 always; =0 never.
fullgraph=False on purpose (as fast, and a graph break in a quantized layer cannot fail the run). Any failure leaves the model eager and returns False.
"""
from __future__ import annotations

import os
import sys
from typing import Optional


def should_compile(n_images: Optional[int]) -> bool:
    v = os.environ.get("GB_COMPILE", "auto").strip().lower()
    if v in ("1", "on", "true", "yes"):
        return True
    if v in ("0", "off", "false", "no"):
        return False
    try:
        floor = int(os.environ.get("GB_COMPILE_MIN_IMAGES", "8"))
    except ValueError:
        floor = 8
    return bool(n_images) and n_images >= floor


def _emit(compiled: bool, n_images: Optional[int], reason: str, status: str = "ok") -> None:
    """dataflux `compile_decision` (best effort)."""
    try:
        import gb_dataflux
        gb_dataflux.emit({"node": "host", "label": "gb_compile", "kind": "compile_decision", "n_items": 1, "items": ["on" if compiled else "off"], "duration_s": 0.0, "status": status,
                          "compiled": compiled, "n_images": n_images or 0, "setting": os.environ.get("GB_COMPILE", "auto"), "reason": reason})
    except Exception:  # noqa: BLE001
        pass


def apply(transformer, n_images: Optional[int] = None) -> bool:
    """Compile the repeated blocks of `transformer` in place when the policy says it pays. Returns True when compiled."""
    if not should_compile(n_images):
        _emit(False, n_images, "policy: batch below GB_COMPILE_MIN_IMAGES or GB_COMPILE=off")
        return False
    try:
        import torch
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/greenboost/inductor"))
        if hasattr(transformer, "compile_repeated_blocks"):
            transformer.compile_repeated_blocks(fullgraph=False, dynamic=False)
        else:
            n = 0
            for name in ("transformer_blocks", "single_transformer_blocks", "blocks", "layers"):
                blocks = getattr(transformer, name, None)
                if blocks is not None:
                    for i, b in enumerate(blocks):
                        blocks[i] = torch.compile(b, fullgraph=False, dynamic=False)
                        n += 1
            if n == 0:
                _emit(False, n_images, "no repeated block list found on the model")
                return False
        print("[gb_compile] regional torch.compile on (first image compiles; later processes reuse the cache)", file=sys.stderr, flush=True)
        _emit(True, n_images, "regional torch.compile applied")
        return True
    except Exception as e:  # noqa: BLE001  - a speed knob must never stop a run
        print(f"[gb_compile] left eager: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        _emit(False, n_images, f"{type(e).__name__}: {e}", "warn")
        return False

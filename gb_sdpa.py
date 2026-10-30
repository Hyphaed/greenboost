"""gb_sdpa: pick the fastest scaled-dot-product-attention backend for the GPU, once per process.

Measured 2026-10-29 on the RTX 5070 (sm_120, torch 2.13 cu130, bf16 and fp16, 1 sequence, 1k to 8k tokens):
  head_dim 128 (FLUX / DiT / LTX class): cuDNN attention 5 to 10% faster than PyTorch's FlashAttention-2 (4.08 vs 4.51 ms at 4608 tokens, 64 vs 58 TFLOP/s),
  both about 2.2x faster than the memory-efficient kernel (25 TFLOP/s). head_dim 64: cuDNN and FlashAttention tie (+-2%), cuDNN slower at 1k tokens.
  Results differ from FlashAttention by under 1e-3 (bf16 rounding) and are correct under the GreenBoost shim.
So on consumer Blackwell (compute capability 12.x) the priority order becomes cuDNN, Flash, Efficient, Math; the dispatcher still falls back down the list for inputs a backend cannot take
(masks, dtypes, head sizes), so nothing breaks. Every attention kernel here runs at 96% of the 66.5 TFLOP/s fp32-accumulate ceiling of this card: the next step up needs a kernel that
accumulates in fp16/fp8 (SageAttention class, see workflow/missing-5070-features.md), not another backend.

GB_ATTN_BACKEND = auto (default) | cudnn | flash | efficient | math | off. `auto` changes nothing on other GPUs. Called from gb_init when GREENBOOST_ACTIVE=1; prints to stderr only.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

_applied: Optional[str] = None
_ctx = None


def _backends():
    from torch.nn.attention import SDPBackend
    return {"cudnn": SDPBackend.CUDNN_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH}


def order(choice: str = "auto", cc: Optional[Tuple[int, int]] = None) -> Optional[List[str]]:
    """The priority order (names) for `choice` on a GPU of compute capability `cc`, or None to leave PyTorch's default."""
    choice = (choice or "auto").strip().lower()
    if choice in ("off", "0", "none", "default"):
        return None
    if choice == "auto":
        return ["cudnn", "flash", "efficient", "math"] if cc is not None and cc[0] == 12 else None
    if choice in ("cudnn", "flash", "efficient", "math"):
        rest = [n for n in ("flash", "cudnn", "efficient", "math") if n != choice]
        return [choice] + rest
    return None


def _emit(order_: Optional[str], cc, setting: str, reason: str, status: str = "ok") -> None:
    """dataflux `sdpa_backend` (best effort)."""
    try:
        import gb_dataflux
        gb_dataflux.emit({"node": "host", "label": "gb_sdpa", "kind": "sdpa_backend", "n_items": 1, "items": [order_ or "default"], "duration_s": 0.0, "status": status,
                          "order": order_ or "", "cc": f"{cc[0]}.{cc[1]}" if cc else "", "setting": setting, "reason": reason})
    except Exception:  # noqa: BLE001
        pass


def apply() -> Optional[str]:
    """Set the process-wide SDPA priority order (idempotent). Returns the order applied, or None when nothing was changed."""
    global _applied, _ctx
    if _applied is not None:
        return _applied
    try:
        import torch
        cc = tuple(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        setting = os.environ.get("GB_ATTN_BACKEND", "auto")
        names = order(setting, cc)
        if not names:
            _emit(None, cc, setting, "PyTorch default order kept (setting off, or not a compute-capability-12 GPU)")
            return None
        from torch.nn.attention import sdpa_kernel
        be = _backends()
        _ctx = sdpa_kernel([be[n] for n in names], set_priority=True)
        _ctx.__enter__()                      # never exited on purpose: the order holds for the life of the process
        _applied = ",".join(names)
        _emit(_applied, cc, setting, "priority order set")
        print(f"[gb_sdpa] attention backend order: {_applied}", file=sys.stderr, flush=True)
        return _applied
    except Exception as e:  # noqa: BLE001  - a tuning knob must never stop a worker
        print(f"[gb_sdpa] not applied: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        _emit(None, None, os.environ.get("GB_ATTN_BACKEND", "auto"), f"{type(e).__name__}: {e}", "warn")
        return None

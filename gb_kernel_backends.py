"""gb_kernel_backends — pluggable low-bit GEMM backend selection for gb_quant.

GreenBoost's weight-quantization layer (`gb_quant.py`) has historically had a
single GEMM backend: the GemLite Triton kernels in `third_party/gemlite`. This
module introduces `GB_KERNEL_BACKEND`, a per-precision / per-shape backend
selector, WITHOUT changing default behaviour: unset (or `gemlite`) resolves to
GemLite for every layer, exactly as before.

Backends shipped today:
  - ``gemlite``   — the GemLite Triton kernels (default; owns 4/8/fp8/nvfp4/tq).
  - ``scaled_mm`` — fp8 e4m3 weight-only via ``torch._scaled_mm`` (cuBLASLt).
                    Opt-in; wins on large-M prefill shapes on Blackwell.
  - ``bf16``      — passthrough (no quantization). A/B baseline lever only.
  - ``cutlass``   — reserved for the Stage-4 sm_120 NVFP4 GEMM (not built yet;
                    ``supports`` returns False so it falls back to gemlite).

Design constraints (see plan `make-a-plan-to-reflective-firefly.md`):
  - Import must never touch the GPU: torch is imported lazily inside the
    functions that need it, so `resolve_backend` stays pure and CPU-testable.
  - Per-layer `supports()` rejection falls back to gemlite — a requested
    backend never silently skips quantization.

`resolve_backend()` is the pure-Python decision function; `gb_quant`'s
`_build_processor`/`_delegate_patch` call it per layer.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

# Precision tokens each backend can serve. gb_quant's shape gating (in_features
# % 32 / % group_size) is applied separately in `_delegate_patch`; the
# per-backend `supports()` below adds any backend-specific shape constraints.
_GEMLITE_PRECISIONS = frozenset({4, 8, 3, 2, "fp8", "e4m3", "nvfp4", "tq3", "tq2"})
_SCALED_MM_PRECISIONS = frozenset({"fp8", "e4m3"})
_CUTLASS_PRECISIONS = frozenset({"nvfp4"})

KNOWN_BACKENDS = ("gemlite", "scaled_mm", "bf16", "cutlass")


def env_backend() -> str:
    """Normalized value of GB_KERNEL_BACKEND. Unset/blank/unknown -> 'gemlite'."""
    raw = (os.environ.get("GB_KERNEL_BACKEND", "") or "").strip().lower()
    if raw in ("", "gemlite"):
        return "gemlite"
    if raw in KNOWN_BACKENDS or raw == "auto":
        return raw
    # Unknown value: warn once via return, but never break — default to gemlite.
    return "gemlite"


def _scaled_mm_available() -> bool:
    """True iff this torch build exposes torch._scaled_mm and fp8 e4m3."""
    try:
        import torch
        return hasattr(torch, "_scaled_mm") and hasattr(torch, "float8_e4m3fn")
    except Exception:
        return False


def _cutlass_available() -> bool:
    """Stage-4 hook: the CUTLASS sm_120 extension is not built yet."""
    try:
        import gb_cutlass  # noqa: F401  (third_party/gb_cutlass, added in Stage 4)
        return bool(getattr(gb_cutlass, "available", lambda: False)())
    except Exception:
        return False


def device_cc(device: str = "cuda") -> Optional[Tuple[int, int]]:
    """(major, minor) compute capability of `device`, or None off-GPU/CPU.

    None is the CPU-test sentinel: `supports()` for shape/arch-gated backends
    returns False on None, so resolution falls back to gemlite cleanly.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        idx = 0
        if isinstance(device, str) and ":" in device:
            idx = int(device.rsplit(":", 1)[1])
        return torch.cuda.get_device_capability(idx)
    except Exception:
        return None


def supports(name: str, bits, in_f: Optional[int], out_f: Optional[int],
             cc: Optional[Tuple[int, int]]) -> bool:
    """Whether backend `name` can serve this (precision, shape, arch)."""
    if name == "bf16":
        return True  # passthrough is always valid (keeps the layer bf16)
    if name == "gemlite":
        return bits in _GEMLITE_PRECISIONS
    if name == "scaled_mm":
        if bits not in _SCALED_MM_PRECISIONS or not _scaled_mm_available():
            return False
        if cc is None or cc < (8, 9):        # fp8 tensor cores: Ada+ (sm_89+)
            return False
        # cuBLASLt fp8 GEMM wants both dims 16-aligned.
        return bool(in_f and out_f and in_f % 16 == 0 and out_f % 16 == 0)
    if name == "cutlass":
        if bits not in _CUTLASS_PRECISIONS or not _cutlass_available():
            return False
        return cc is not None and cc >= (12, 0)   # sm_120 GeForce Blackwell
    return False


def _auto_policy(bits, in_f, out_f, cc) -> str:
    """AUTO backend choice. Ships == gemlite (zero behaviour change).

    The flip to route fp8 GEMM-shaped layers to `scaled_mm` lives here and is
    intentionally disabled until the Stage-1 microbench proves the win on this
    hardware (owner rule: measured wins only). To enable, return "scaled_mm"
    for fp8 when supports() passes and the shape is GEMM-class (large in/out).
    """
    return "gemlite"


def resolve_backend(bits, in_f: Optional[int], out_f: Optional[int],
                    cc: Optional[Tuple[int, int]],
                    env: Optional[str] = None) -> str:
    """Pick the backend name for one layer. Pure function (CPU-testable).

    Falls back to gemlite whenever the requested backend can't serve the
    (precision, shape, arch) — so a requested backend never drops a layer.
    """
    name = (env if env is not None else env_backend())
    if name == "auto":
        name = _auto_policy(bits, in_f, out_f, cc)
    if name == "gemlite":
        return "gemlite"
    if supports(name, bits, in_f, out_f, cc):
        return name
    return "gemlite"


# ---------------------------------------------------------------------------
# scaled_mm backend: fp8 e4m3 weight-only via torch._scaled_mm (cuBLASLt).
# Row-wise weight scales (per output channel) + dynamic per-tensor activation
# scale. Mirrors the GemLite processor interface: `from_linear(layer)` returns
# a module whose `.forward` gb_quant rebinds onto the original Linear.
# ---------------------------------------------------------------------------
_FP8_E4M3_MAX = 448.0


def build_scaled_mm_processor(bits, device: str, dtype):
    """Return a processor exposing `from_linear(layer) -> module` (like GemLite)."""
    return _ScaledMMProcessor(device=device, dtype=dtype)


def build_cutlass_nvfp4_processor(bits, device: str, dtype):
    """Stage-A2 processor for the sm_120a CUTLASS NVFP4 GEMM (gb_cutlass).

    Returns None until the weight-packing reconciliation + numeric bench are
    done: a real processor must repack the nvfp4 weight into CUTLASS's
    interleaved block-scale layout and route forward() through
    gb_cutlass.gemm_nvfp4. gb_quant._build_processor falls back to the nvfp4
    gemlite path on None, and resolve_backend only names "cutlass" when
    gb_cutlass.available() is True (built + GB_CUTLASS_ENABLE=1), so returning
    None here is a safe no-op in the default configuration.

    See tests/bench/bench_cutlass_nvfp4.py — flip this to construct the real
    processor only after that bench passes on the target GPU."""
    return None


class _ScaledMMProcessor:
    W_nbits = 8  # storage width, for parity with GemLite processors

    def __init__(self, device: str = "cuda", dtype=None):
        self.device = device
        self.dtype = dtype

    def from_linear(self, layer):
        import torch
        w = layer.weight.data
        out_f, in_f = w.shape
        dtype = self.dtype or (w.dtype if w.dtype.is_floating_point else torch.bfloat16)
        wf = w.to(device=self.device, dtype=torch.float32)
        # Row-wise (per output channel) absmax scale.
        w_scale = (wf.abs().amax(dim=1, keepdim=True) / _FP8_E4M3_MAX).clamp_(min=1e-12)
        w_q = (wf / w_scale).clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        bias = None
        if layer.bias is not None:
            bias = layer.bias.data.to(device=self.device, dtype=dtype)
        return _ScaledMMLinear(w_q, w_scale.to(torch.float32), bias, in_f, out_f, dtype)


def _scaled_mm_linear_cls():
    """Lazily build the nn.Module delegate class (torch only imported here)."""
    import torch
    import torch.nn as nn

    class _ScaledMMLinear(nn.Module):
        """fp8 e4m3 weight-only linear delegate. An nn.Module so gb_quant's
        `add_module`/`.to()` propagate the fp8 weight + scale buffers."""

        def __init__(self, w_q, w_scale, bias, in_f, out_f, out_dtype):
            super().__init__()
            self.register_buffer("weight_q", w_q)        # (out_f, in_f) fp8
            self.register_buffer("weight_scale", w_scale)  # (out_f, 1) f32
            self.register_buffer("bias_t", bias)         # (out_f,) or None
            self.in_features = in_f
            self.out_features = out_f
            self.out_dtype = out_dtype

        def forward(self, x):
            orig_shape = x.shape
            x2 = x.reshape(-1, orig_shape[-1])
            # Row-wise (per-token) dynamic activation scale -> fp8.
            a_scale = (x2.abs().amax(dim=1, keepdim=True) / _FP8_E4M3_MAX
                       ).clamp_(min=1e-12).to(torch.float32).contiguous()
            a_q = (x2.to(torch.float32) / a_scale).clamp_(
                -_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
            # torch._scaled_mm RowWise: a (M,K) fp8, b (K,N) fp8,
            # scale_a (M,1), scale_b (1,N), both contiguous. weight_q is
            # (N,K) row-major so weight_q.t() is the (K,N) column-major b.
            out = torch._scaled_mm(
                a_q, self.weight_q.t(),
                scale_a=a_scale,
                scale_b=self.weight_scale.reshape(1, -1).contiguous(),
                bias=None,
                out_dtype=self.out_dtype,
            )
            if self.bias_t is not None:
                out = out + self.bias_t
            return out.reshape(*orig_shape[:-1], self.out_features)

    return _ScaledMMLinear


def _ScaledMMLinear(w_q, w_scale, bias, in_f, out_f, out_dtype):  # noqa: N802
    return _scaled_mm_linear_cls()(w_q, w_scale, bias, in_f, out_f, out_dtype)

# SPDX-License-Identifier: Apache-2.0
"""gb_cutlass — sm_120a block-scaled NVFP4 GEMM backend for gb_kernel_backends.

gb_kernel_backends._cutlass_available() imports this module and calls
available(); resolve_backend() routes nvfp4 GEMM-shaped layers here only when it
returns True. Until then the backend router falls back to gemlite/scaled_mm
(the owner's fp8 floor holds), so this staying unbuilt or inert is a no-op.

available() is deliberately conservative — it requires ALL of:
  1. the compiled CUDA extension importing,
  2. a real sm_120 (compute capability 12.x) device, and
  3. GB_CUTLASS_ENABLE=1 in the environment.
(3) means a merely-compiled-but-unbenched extension never silently routes a
production GEMM: flip the env only after tests/bench/bench_cutlass_nvfp4.py
confirms numerics + speed on the box (Stage-A2 timebox checkpoint).
"""
from __future__ import annotations

import os

try:
    from ._gb_cutlass_C import gemm_nvfp4  # noqa: F401  (compiled extension)
    _COMPILED = True
except Exception:
    _COMPILED = False

    def gemm_nvfp4(*_a, **_k):  # type: ignore[misc]
        raise RuntimeError("gb_cutlass extension not built — run "
                           "third_party/gb_cutlass/setup.py build_ext --inplace")


def _is_sm120() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability(0)[0] == 12
    except Exception:
        return False


def available() -> bool:
    """True only when built, on an sm_120 device, and explicitly enabled."""
    return (_COMPILED
            and os.environ.get("GB_CUTLASS_ENABLE") == "1"
            and _is_sm120())

# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""Shared gate for tests that need a REAL, USABLE CUDA device.

`torch.cuda.is_available()` answers "is there a driver and a device", not
"can I allocate on it". On this machine the two routinely disagree: GB-Synapse
keeps a llama-server resident holding ~10.3 GB of the 12 GB card, which leaves
too little headroom for a second process to even create a CUDA context. Tests
gated only on `is_available()` then die on their first `.cuda()` with
`torch.AcceleratorError: CUDA error: out of memory`, on a tensor of 8x8
floats , a failure that reads like a code regression and is not one.

Real incident (2026-08-19): three tests (test_gb_kernel_backends's fp8 GEMM
and two gb_prefetch end-to-end tests) failed the pre-commit suite for exactly
this reason while a served model held the card. The suite must report "skipped
, the GPU is busy", never "failed".

The free-VRAM probe deliberately shells out to nvidia-smi instead of calling
`torch.cuda.mem_get_info()`: mem_get_info initialises a CUDA context, which is
the very allocation that fails when the card is full, so asking torch would
crash on the question instead of answering it. nvidia-smi also reports
PHYSICAL free VRAM, unaffected by any GreenBoost shim's inflated "virtual"
figure.
"""
from __future__ import annotations

import functools
import subprocess

# A CUDA context alone costs a few hundred MiB before a single tensor lands.
# 768 MiB is "context plus small test tensors", not "enough for a model".
DEFAULT_MIN_FREE_MIB = 768


@functools.lru_cache(maxsize=1)
def free_vram_mib() -> int | None:
    """Physical free VRAM on CUDA device 0, in MiB, WITHOUT creating a context.

    Returns None when the probe itself is unavailable (no nvidia-smi, timeout,
    unparseable output). None means "unknown", and unknown must never skip a
    test , an unreadable probe is not evidence that the GPU is busy.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            return int(line)
    return None


def gpu_skip_reason(min_free_mib: int = DEFAULT_MIN_FREE_MIB) -> str | None:
    """None when the GPU is usable; otherwise a human-readable skip reason."""
    try:
        import torch
    except ImportError:
        return "torch is not installed"
    if not torch.cuda.is_available():
        return "requires a real CUDA device"
    free = free_vram_mib()
    if free is not None and free < min_free_mib:
        return (
            f"GPU busy: only {free} MiB VRAM free, this test needs "
            f"{min_free_mib} MiB to create a CUDA context. Free the card "
            f"(e.g. `gb synapse stop`) and re-run."
        )
    return None

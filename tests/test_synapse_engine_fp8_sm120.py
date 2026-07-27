#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
P5.3 probe result: does GreenBoost's vendored torch-core engine's FP8 block
GEMM actually run on Blackwell/sm120 without a new kernel?

The ledger (workflow/task_to_develop.md) originally scoped P5.3 as "port a
triton blockscale GEMM for sm120" on the assumption there was no fallback.
There is: synapse_engine/gllm/layers/quantization/fp8.py already ships a
generic Triton `w8a8_block_fp8_matmul` alongside two Hopper-only
accelerations (DeepGEMM, gated `cc[0] >= 9`; FlashInfer swapAB, gated
`cc == (9, 0)` exactly) that both self-test and degrade gracefully.

Live-probed 2026-07-26 on an RTX 5070 (cc 12,0) via
/usr/local/lib/greenboost/synapse-torch-env/bin/python (needs `gllm`
importable — the system Python running the rest of this suite doesn't have
it, hence the skip below on a normal `pytest tests/` run):
  - deepgemm_available()        -> False (DeepGEMM itself raises "Unknown
    recipe" for this arch — caught by its own self-test, degrades cleanly)
  - flashinfer_swapab_available() -> False (Hopper-only gate, as designed)
  - w8a8_block_fp8_matmul(...)  -> WORKS: correct shape, finite output

Conclusion: no kernel port needed for P5.3. sm120 rides the Triton path,
same as any pre-Hopper card. This file exists so that conclusion is
regression-tested, not just a one-time manual finding — run it under the
torch venv (`/usr/local/lib/greenboost/synapse-torch-env/bin/python -m
pytest tests/test_synapse_engine_fp8_sm120.py`) to re-verify.
"""
import pytest

pytest.importorskip("gllm")
torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("no CUDA device available", allow_module_level=True)

import triton

from gllm.layers.quantization.fp8 import (
    deepgemm_available,
    flashinfer_swapab_available,
    per_token_group_quant_fp8,
    w8a8_block_fp8_matmul,
)


def test_triton_block_fp8_matmul_runs_on_this_device():
    """The generic (non-Hopper-specific) path — this is the one every
    non-Hopper card, including sm120, actually uses."""
    M, K, N = 16, 512, 512
    block_n, block_k = 128, 128

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    aq, as_ = per_token_group_quant_fp8(a, block_k, column_major_scales=False)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    ws = torch.rand(triton.cdiv(N, block_n), triton.cdiv(K, block_k),
                    device="cuda", dtype=torch.float32) * 0.02

    out = w8a8_block_fp8_matmul(aq, w, as_, ws, [block_n, block_k],
                                output_dtype=torch.bfloat16)

    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


def test_hopper_only_accelerations_correctly_absent_off_hopper():
    """Documents (rather than asserts a hard requirement on) this specific
    card's behavior — both accelerations are OPTIONAL speedups; their
    absence must never be treated as a failure, only a slower path. Skips
    outright on an actual Hopper card, where both are expected to be
    available and this probe's premise doesn't apply."""
    cc = torch.cuda.get_device_capability()
    if cc == (9, 0):
        pytest.skip("Hopper — DeepGEMM/FlashInfer accelerations expected here, "
                    "not what this probe checks")
    # Both are lru_cache'd self-tests that must never raise, only report
    # availability — confirms the "degrades gracefully" contract fp8.py's
    # own docstrings promise, not just that they happen to return False.
    assert deepgemm_available() in (True, False)
    assert flashinfer_swapab_available() in (True, False)

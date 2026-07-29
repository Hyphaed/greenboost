# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Vendored verbatim (import line only adapted: this repo has no
# `vllm.triton_utils` shim, so `triton`/`triton.language` are imported
# directly) from vllm/model_executor/layers/mamba/ops/triton_helpers.py,
# 2026-07-29. See synapse_engine/NOTICE for the Mamba-2 SSD vendoring note.

import triton
import triton.language as tl


@triton.jit
def fast_exp(x):
    """Faster alternative to tl.exp() using the hardware exp2 instruction.

    tl.math.exp2 maps directly to a single ex2.approx.f32 PTX instruction,
    while tl.exp goes through libdevice __nv_expf which adds function call
    overhead and extra range checking.
    """
    # exp(x) = exp2(x * log2(e)), where log2(e) = 1/ln(2) = 1.4426950408889634
    LOG2E = tl.constexpr(1.4426950408889634)
    return tl.math.exp2(LOG2E * x)

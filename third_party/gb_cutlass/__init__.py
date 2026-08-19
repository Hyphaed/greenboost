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


# ---------------------------------------------------------------------------
# pack_nvfp4 (missing_features.md item (b), 2026-08-02): the flat NVFP4
# quantization half of the "weight-packing reconciliation" this module's own
# docstring names as the remaining Stage-A2 step.
#
# SCOPE, stated plainly: this produces ROW-MAJOR packed e2m1 data + a FLAT
# (one byte per 16-element block) ue4m3 scale array. It does NOT produce
# CUTLASS's Sm1xxBlkScaledConfig-interleaved scale-factor layout that
# gemm_nvfp4() actually requires for its SFA/SFB arguments (see
# gb_cutlass_ext.cu's tile_atom_to_shape_SFA/SFB calls) — that interleaving
# is a CuTe Layout coordinate-convention detail (block-vs-element K
# granularity, nested-tuple atom tiling) that could not be verified against
# a real compile+run cycle safely in this session; a wrong guess there
# produces silently WRONG GEMM NUMERICS, not a compile error, so it was not
# attempted rather than risk shipping that failure mode. gemm_nvfp4() (the
# compiled extension) therefore still cannot be called end-to-end from this
# packer's output — build_cutlass_nvfp4_processor() correctly stays a stub
# returning None (gb_kernel_backends.py) until that interleaving is done and
# verified.
#
# What IS real and tested here: the quantization math itself (block absmax
# scale, E2M1 4-bit rounding, UE4M3 8-bit scale encoding), which is a well-
# specified, independently-verifiable format — round-trip cosine similarity
# against a CPU reference, no GPU or CUTLASS build required. See
# tests/test_gb_cutlass_pack.py.
#
# E2M1 magnitude table (bias=1, per CUTLASS's own float_e2m1_t comment in
# cutlass/float_subbyte.h: "Range: [0,0.5,1,1.5,2,3,4,6], exp_bias: 1") —
# codes 0-7 are the 3-bit magnitude; bit 3 is the sign.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_NVFP4_BLOCK = 16   # NVFP4's fixed block size (InputSFVectorSize in the .cu)
_UE4M3_MAX = 448.0  # same max magnitude as signed e4m3 (CUTLASS: "Range: [0:448]")


def _e2m1_encode_table():
    """(magnitudes tensor, codes tensor) for nearest-value quantization —
    built once, cached by the caller. 8 unsigned codes; sign handled
    separately (bit 3 of the packed nibble)."""
    import torch
    return torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32)


def quantize_e2m1(x: "torch.Tensor") -> "torch.Tensor":
    """Round each element of `x` (already normalized to the E2M1 magnitude
    range) to the nearest representable value, returning the 3-bit
    UNSIGNED magnitude code (0-7) as int64, sign separately at bit 3 by
    the caller. Round-to-nearest (ties resolved by argmin's first-match,
    i.e. toward the lower code — matches standard nearest-value rounding
    for a non-uniform codebook where exact ties are a measure-zero case)."""
    import torch
    table = _e2m1_encode_table().to(x.device)
    diffs = (x.abs().unsqueeze(-1) - table).abs()
    return diffs.argmin(dim=-1)


def dequantize_e2m1(codes: "torch.Tensor", signs: "torch.Tensor") -> "torch.Tensor":
    """Inverse of quantize_e2m1: unsigned magnitude codes (0-7) + a
    {-1,+1} sign tensor -> float32 magnitudes."""
    import torch
    table = _e2m1_encode_table().to(codes.device)
    return table[codes] * signs


def quantize_ue4m3(x: "torch.Tensor") -> "torch.Tensor":
    """Encode non-negative `x` as UE4M3 (unsigned e4m3, bias=7, range
    [0,448]) via PyTorch's native float8_e4m3fn , since UE4M3 is signed
    E4M3 with the sign bit simply unused/always-0 (same exponent/mantissa
    bit layout, same bias, same max magnitude , CUTLASS's own comment:
    "Range: [0:448]", identical to signed e4m3's positive range), encoding
    a non-negative value through the signed dtype produces the IDENTICAL
    byte pattern UE4M3 would, with the sign bit already 0. This piggybacks
    on PyTorch's own well-tested rounding/subnormal handling instead of
    hand-deriving float8 bit-encoding rules. Returns raw uint8 bytes."""
    import torch
    clamped = x.clamp(min=0.0, max=_UE4M3_MAX)
    return clamped.to(torch.float8_e4m3fn).view(torch.uint8)


def dequantize_ue4m3(raw: "torch.Tensor") -> "torch.Tensor":
    """Inverse of quantize_ue4m3: raw uint8 bytes -> float32."""
    import torch
    return raw.view(torch.float8_e4m3fn).to(torch.float32)


def pack_nvfp4(w: "torch.Tensor", block: int = _NVFP4_BLOCK):
    """Quantize a [rows, cols] weight tensor to flat (non-interleaved) NVFP4:
    row-major packed e2m1 data (2 values/byte) + one UE4M3 scale byte per
    16-element block. See this module's docstring above for what this does
    and does NOT complete (the CUTLASS interleaved SF layout is NOT applied
    here).

    Returns (packed: uint8[rows, cols//2], scales: uint8[rows, cols//block],
    meta: dict) , meta carries {"rows","cols","block"} for dequantize_nvfp4.
    """
    import torch
    if w.dim() != 2:
        raise ValueError(f"pack_nvfp4: expected a 2D tensor, got shape {tuple(w.shape)}")
    rows, cols = w.shape
    if cols % block != 0:
        raise ValueError(f"pack_nvfp4: cols={cols} must be a multiple of block={block}")

    wf = w.detach().to(torch.float32)
    blocks = wf.view(rows, cols // block, block)
    absmax = blocks.abs().amax(dim=-1)                       # [rows, n_blocks]
    ideal_scale = (absmax / _E2M1_MAGNITUDES[-1]).clamp_min(1e-12)
    scale_bytes = quantize_ue4m3(ideal_scale)                 # [rows, n_blocks] uint8
    # Divide by the ACTUAL (rounded) scale, not the ideal float scale — the
    # dequant path only has the rounded byte to work with, so encode/decode
    # must agree on which scale value was really used.
    actual_scale = dequantize_ue4m3(scale_bytes).clamp_min(1e-12)
    normalized = blocks / actual_scale.unsqueeze(-1)

    signs = torch.where(normalized < 0, -1.0, 1.0)
    codes = quantize_e2m1(normalized)                         # [rows, n_blocks, block] int64, 0-7
    nibbles = codes.to(torch.uint8) | torch.where(
        signs < 0, torch.tensor(0x8, dtype=torch.uint8), torch.tensor(0x0, dtype=torch.uint8))
    nibbles = nibbles.view(rows, cols)
    low = nibbles[:, 0::2]
    high = nibbles[:, 1::2]
    packed = (low | (high << 4)).contiguous()                 # [rows, cols//2] uint8

    return packed, scale_bytes.contiguous(), {"rows": rows, "cols": cols, "block": block}


def unpack_nvfp4(packed: "torch.Tensor", scale_bytes: "torch.Tensor", meta: dict) -> "torch.Tensor":
    """Inverse of pack_nvfp4 , reconstructs the float32 [rows, cols] tensor.
    Used for round-trip fidelity testing (no GPU/CUTLASS required)."""
    import torch
    rows, cols, block = meta["rows"], meta["cols"], meta["block"]
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = torch.empty(rows, cols, dtype=torch.uint8, device=packed.device)
    nibbles[:, 0::2] = low
    nibbles[:, 1::2] = high
    codes = (nibbles & 0x7).to(torch.int64)
    signs = torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    magnitudes = dequantize_e2m1(codes, signs)
    magnitudes = magnitudes.view(rows, cols // block, block)
    scale = dequantize_ue4m3(scale_bytes).unsqueeze(-1)
    return (magnitudes * scale).view(rows, cols)

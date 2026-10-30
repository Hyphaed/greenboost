# SPDX-License-Identifier: Apache-2.0
"""NVFP4 packing tests for gb_cutlass.

Two independent layers, deliberately separate:

* The quantization FORMAT (block absmax scale, E2M1 rounding, UE4M3 scale
  encoding) is a well-specified numeric contract. It needs no GPU and no
  CUTLASS build, so it is always collected.
* The CUTLASS interleaved scale-factor LAYOUT can only be checked against the
  real kernel, so those tests skip unless the extension is built on an sm_120
  device. They exist because a wrong layout produces silently wrong numerics
  rather than an error — see gb_cutlass/__init__.py.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party"))

torch = pytest.importorskip("torch")
import gb_cutlass  # noqa: E402


def _rt(w):
    packed, scales, meta = gb_cutlass.pack_nvfp4(w)
    return gb_cutlass.unpack_nvfp4(packed, scales, meta)


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.reshape(1, -1).float(), b.reshape(1, -1).float()).item()


# --------------------------------------------------------------- format layer
def test_pack_roundtrip_shapes_and_dtypes():
    w = torch.randn(8, 64)
    packed, scales, meta = gb_cutlass.pack_nvfp4(w)
    assert packed.shape == (8, 32) and packed.dtype == torch.uint8
    assert scales.shape == (8, 4) and scales.dtype == torch.uint8
    assert meta == {"rows": 8, "cols": 64, "block": 16}


def test_pack_roundtrip_is_close():
    torch.manual_seed(0)
    w = torch.randn(16, 256) * 0.1
    assert _cos(_rt(w), w) > 0.97


def test_pack_preserves_sign():
    w = torch.full((2, 16), -1.0)
    assert (_rt(w) <= 0).all()


def test_pack_exact_on_representable_magnitudes():
    # 0, 0.5, 1, 1.5, 2, 3, 4, 6 scaled by a power of two is exactly representable.
    vals = torch.tensor(gb_cutlass._E2M1_MAGNITUDES, dtype=torch.float32)
    w = vals.repeat(2, 2)  # [2, 16]
    assert torch.allclose(_rt(w), w, atol=1e-6)


def test_pack_rejects_non_multiple_of_block():
    with pytest.raises(ValueError, match="multiple of block"):
        gb_cutlass.pack_nvfp4(torch.randn(4, 20))


def test_pack_rejects_non_2d():
    with pytest.raises(ValueError, match="2D"):
        gb_cutlass.pack_nvfp4(torch.randn(4, 16, 2))


# ----------------------------------------------------- CUTLASS SF layout layer
_HAVE_EXT = (
    gb_cutlass._COMPILED
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability(0)[0] == 12
)
needs_ext = pytest.mark.skipif(
    not _HAVE_EXT, reason="needs the gb_cutlass CUDA extension on an sm_120 GPU")


@needs_ext
def test_sf_buffer_is_sized_by_the_kernels_own_layout():
    """pack_sfa/pack_sfb size their output from cute::cosize of the SAME layout
    the GEMM indexes, so the buffer can never be too small for the kernel."""
    from gb_cutlass._gb_cutlass_C import pack_sfa, pack_sfb, sf_vec_size
    M, N, K = 512, 4096, 4096
    kb = K // sf_vec_size()
    flat_a = torch.zeros(M * kb, dtype=torch.uint8, device="cuda")
    flat_b = torch.zeros(N * kb, dtype=torch.uint8, device="cuda")
    assert pack_sfa(flat_a, M, N, K).numel() >= M * kb
    assert pack_sfb(flat_b, M, N, K).numel() >= N * kb


@needs_ext
def test_sf_packer_rejects_wrong_element_count():
    from gb_cutlass._gb_cutlass_C import pack_sfa
    bad = torch.zeros(7, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="expected"):
        pack_sfa(bad, 512, 4096, 4096)


@needs_ext
@pytest.mark.parametrize("M", [16, 512])
def test_gemm_matches_dequantized_reference(M):
    """The layout check that matters: compare against a reference over the SAME
    dequantized operands, which isolates kernel + SF layout from quant error.
    A wrong interleaving shows up here and nowhere else."""
    from gb_cutlass._gb_cutlass_C import gemm_nvfp4, pack_sfa, pack_sfb
    N = K = 4096
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda") * 0.1
    B = torch.randn(N, K, device="cuda") * 0.1
    pA, sA, mA = gb_cutlass.pack_nvfp4(A)
    pB, sB, mB = gb_cutlass.pack_nvfp4(B)
    SFA = pack_sfa(sA.reshape(-1).contiguous(), M, N, K)
    SFB = pack_sfb(sB.reshape(-1).contiguous(), M, N, K)
    D = gemm_nvfp4(pA.contiguous(), SFA, pB.contiguous(), SFB, M, N, K)
    ref = (gb_cutlass.unpack_nvfp4(pA, sA, mA).float()
           @ gb_cutlass.unpack_nvfp4(pB, sB, mB).float().T)
    assert _cos(D, ref) >= 0.99


@needs_ext
def test_available_still_requires_the_env_flag():
    """Building the extension must NOT be enough to route a production GEMM:
    NVFP4 measures ~0.991 cosine against bf16, below the fp8 floor."""
    saved = os.environ.pop("GB_CUTLASS_ENABLE", None)
    try:
        assert gb_cutlass.available() is False
    finally:
        if saved is not None:
            os.environ["GB_CUTLASS_ENABLE"] = saved


# ------------------------------------------------- fused activation quantizer
@needs_ext
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_quantizer_agrees_with_the_pytorch_packer(dtype):
    """The fused CUDA path and the PyTorch reference path must produce the same
    GEMM result. They are independent implementations of the same format, so
    agreement here is the real check on both."""
    from gb_cutlass._gb_cutlass_C import (gemm_nvfp4, pack_sfa, pack_sfb,
                                          quantize_act_nvfp4)
    M = N = K = 512
    torch.manual_seed(0)
    A = (torch.randn(M, K, device="cuda") * 0.1).to(dtype).contiguous()
    B = torch.randn(N, K, device="cuda") * 0.1
    pB, sB, _ = gb_cutlass.pack_nvfp4(B)
    SFB = pack_sfb(sB.reshape(-1).contiguous(), M, N, K)
    pB = pB.contiguous()

    pA_ref, sA_ref, _ = gb_cutlass.pack_nvfp4(A.float())
    SFA_ref = pack_sfa(sA_ref.reshape(-1).contiguous(), M, N, K)
    ref = gemm_nvfp4(pA_ref.contiguous(), SFA_ref, pB, SFB, M, N, K)

    pA, SFA = quantize_act_nvfp4(A, M, N, K)
    got = gemm_nvfp4(pA, SFA, pB, SFB, M, N, K)

    assert pA.shape == (M, K // 2) and pA.dtype == torch.uint8
    assert SFA.numel() == SFA_ref.numel()
    assert _cos(got, ref) >= 0.999


@needs_ext
def test_fused_quantizer_rejects_bad_input():
    from gb_cutlass._gb_cutlass_C import quantize_act_nvfp4
    with pytest.raises(RuntimeError, match="bfloat16 or float32"):
        quantize_act_nvfp4(torch.zeros(16, 512, dtype=torch.float64, device="cuda"),
                           16, 512, 512)
    with pytest.raises(RuntimeError, match=r"\[M, K\]"):
        quantize_act_nvfp4(torch.zeros(8, 512, dtype=torch.bfloat16, device="cuda"),
                           16, 512, 512)


@needs_ext
def test_fused_quantizer_handles_an_all_zero_block():
    """An all-zero block has amax 0, so the scale is 0 and the reciprocal is
    guarded. Without that guard this produces NaNs that only show up as a
    degraded model, never as an error."""
    from gb_cutlass._gb_cutlass_C import quantize_act_nvfp4
    A = torch.zeros(16, 512, dtype=torch.bfloat16, device="cuda")
    packed, sf = quantize_act_nvfp4(A, 16, 512, 512)
    assert int(packed.sum().item()) == 0
    assert int(sf.sum().item()) == 0

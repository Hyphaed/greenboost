#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_kernel_backends: the GB_KERNEL_BACKEND registry + resolve_backend.

All CPU-only, no CUDA, no GemLite. resolve_backend is a pure function; cc is
passed explicitly so no test initializes a CUDA context. The scaled_mm
processor's from_linear packs fp8 weights on CPU (storage only); its forward is
CUDA-gated and lives in the skipif block.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn

import gb_kernel_backends as kb

_BLACKWELL = (12, 0)
_ADA = (8, 9)
_TURING = (7, 5)


# ── env_backend normalization ────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    (None, "gemlite"),
    ("", "gemlite"),
    ("gemlite", "gemlite"),
    ("GEMLITE", "gemlite"),
    ("scaled_mm", "scaled_mm"),
    ("bf16", "bf16"),
    ("cutlass", "cutlass"),
    ("auto", "auto"),
    ("nonsense", "gemlite"),   # unknown -> safe default
])
def test_env_backend_normalization(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("GB_KERNEL_BACKEND", raising=False)
    else:
        monkeypatch.setenv("GB_KERNEL_BACKEND", raw)
    assert kb.env_backend() == expected


# ── resolve_backend: default path is always gemlite ──────────────────────────

def test_resolve_default_is_gemlite():
    """Unset env and explicit gemlite both resolve to gemlite for every bits."""
    for bits in (4, 8, "fp8", "nvfp4", "tq3", "tq2"):
        assert kb.resolve_backend(bits, 4096, 4096, _BLACKWELL, None) == "gemlite"
        assert kb.resolve_backend(bits, 4096, 4096, _BLACKWELL, "gemlite") == "gemlite"


def test_resolve_auto_ships_as_gemlite():
    """AUTO ships identical to gemlite (no behaviour change by default)."""
    assert kb.resolve_backend("fp8", 8192, 8192, _BLACKWELL, "auto") == "gemlite"


# ── resolve_backend: scaled_mm gating ────────────────────────────────────────

def test_scaled_mm_selected_for_aligned_fp8():
    assert kb.resolve_backend("fp8", 4096, 4096, _BLACKWELL, "scaled_mm") == "scaled_mm"
    assert kb.resolve_backend("e4m3", 4096, 4096, _ADA, "scaled_mm") == "scaled_mm"


def test_scaled_mm_falls_back_on_unaligned_shape():
    # in_f not %16 -> gemlite
    assert kb.resolve_backend("fp8", 4095, 4096, _BLACKWELL, "scaled_mm") == "gemlite"
    # out_f not %16 -> gemlite
    assert kb.resolve_backend("fp8", 4096, 100, _BLACKWELL, "scaled_mm") == "gemlite"


def test_scaled_mm_falls_back_on_non_fp8_bits():
    for bits in (4, 8, "nvfp4", "tq3"):
        assert kb.resolve_backend(bits, 4096, 4096, _BLACKWELL, "scaled_mm") == "gemlite"


def test_scaled_mm_falls_back_on_old_arch():
    assert kb.resolve_backend("fp8", 4096, 4096, _TURING, "scaled_mm") == "gemlite"


def test_scaled_mm_falls_back_when_cc_unknown():
    # cc None (CPU-test sentinel) must never select an arch-gated backend.
    assert kb.resolve_backend("fp8", 4096, 4096, None, "scaled_mm") == "gemlite"


def test_scaled_mm_falls_back_when_unavailable(monkeypatch):
    monkeypatch.setattr(kb, "_scaled_mm_available", lambda: False)
    assert kb.resolve_backend("fp8", 4096, 4096, _BLACKWELL, "scaled_mm") == "gemlite"


# ── resolve_backend: bf16 passthrough + cutlass placeholder ──────────────────

def test_bf16_passthrough_always():
    for bits in (4, 8, "fp8", "nvfp4"):
        assert kb.resolve_backend(bits, 4096, 4096, _BLACKWELL, "bf16") == "bf16"


def test_cutlass_unbuilt_falls_back():
    # Stage-4 backend not built yet -> _cutlass_available() False -> gemlite.
    assert kb.resolve_backend("nvfp4", 4096, 4096, _BLACKWELL, "cutlass") == "gemlite"


# ── supports() completeness ──────────────────────────────────────────────────

def test_supports_matrix():
    assert kb.supports("gemlite", 4, 4096, 4096, _BLACKWELL) is True
    assert kb.supports("gemlite", "nvfp4", 4096, 4096, _BLACKWELL) is True
    assert kb.supports("gemlite", "bogus", 4096, 4096, _BLACKWELL) is False
    assert kb.supports("bf16", 4, None, None, None) is True
    assert kb.supports("unknown-backend", 4, 4096, 4096, _BLACKWELL) is False


# ── device_cc is CPU-safe ────────────────────────────────────────────────────

def test_device_cc_none_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert kb.device_cc("cuda") is None


# ── scaled_mm processor: CPU packing + CUDA-gated forward ────────────────────

def test_scaled_mm_from_linear_packs_fp8_on_cpu():
    proc = kb.build_scaled_mm_processor("fp8", device="cpu", dtype=torch.bfloat16)
    lin = nn.Linear(64, 32, bias=True).to(torch.bfloat16)
    impl = proc.from_linear(lin)
    assert isinstance(impl, nn.Module)
    assert impl.weight_q.dtype == torch.float8_e4m3fn
    assert tuple(impl.weight_q.shape) == (32, 64)
    assert impl.weight_scale.dtype == torch.float32
    assert impl.bias_t is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp8 GEMM needs CUDA")
def test_scaled_mm_forward_matches_bf16_reference():
    torch.manual_seed(0)
    lin = nn.Linear(256, 128, bias=True).cuda().to(torch.bfloat16)
    x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
    ref = lin(x)
    proc = kb.build_scaled_mm_processor("fp8", device="cuda", dtype=torch.bfloat16)
    impl = proc.from_linear(lin)
    out = impl.forward(x)
    assert out.shape == ref.shape
    # fp8 is lossy; assert coarse agreement, not exactness.
    rel = (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)
    assert rel < 0.15

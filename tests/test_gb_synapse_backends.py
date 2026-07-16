#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_synapse_backends.py: format detection, legacy-engine mapping,
and the %-derived VRAM+T2 budget helper (2026-07-16 vLLM/backend redesign).

CPU-only. No GGUF, no CUDA, no real HF network calls.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import gb_synapse_backends as gsb


# ── detect_format: local paths ────────────────────────────────────────────

def test_detect_format_local_gguf_file(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"")
    assert gsb.detect_format(str(f)) == "gguf"


def test_detect_format_local_gguf_dir(tmp_path):
    (tmp_path / "model-00001-of-00002.gguf").write_bytes(b"")
    assert gsb.detect_format(str(tmp_path)) == "gguf"


def test_detect_format_local_diffusers_dir(tmp_path):
    (tmp_path / "model_index.json").write_text("{}")
    (tmp_path / "unet").mkdir()
    assert gsb.detect_format(str(tmp_path)) == "diffusers"


def test_detect_format_local_safetensors_dir(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"")
    assert gsb.detect_format(str(tmp_path)) == "safetensors"


def test_detect_format_local_unknown_dir(tmp_path):
    (tmp_path / "readme.txt").write_text("hi")
    assert gsb.detect_format(str(tmp_path)) == "unknown"


def test_detect_format_nonexistent_path_falls_back_to_hf_api(monkeypatch):
    """A path that doesn't exist locally is treated as a bare HF repo id —
    detect_format() should attempt the HfApi lookup and degrade to "unknown"
    when that's unavailable/fails, never raise."""
    assert gsb.detect_format("some/nonexistent-repo-id-xyz") == "unknown"


# ── select_backend: legacy "gbquant" + explicit engine routing ───────────

class _FakeEntry:
    def __init__(self, engine):
        self.engine = engine
        self.name = "fake"
        self.path = ""
        self.repo = ""
        self.quant = "FP8"
        self.arch = ""
        self.source = "hf"


def test_select_backend_llama_cpp_default():
    backend = gsb.select_backend(_FakeEntry("llama.cpp"))
    assert isinstance(backend, gsb.LlamaCppBackend)


def test_select_backend_transformers_explicit():
    backend = gsb.select_backend(_FakeEntry("transformers"))
    assert isinstance(backend, gsb.TransformersBackend)


def test_select_backend_diffusers_explicit():
    backend = gsb.select_backend(_FakeEntry("diffusers"))
    assert isinstance(backend, gsb.DiffusersBackend)


def test_select_backend_vllm_falls_back_to_transformers_when_unavailable(monkeypatch):
    monkeypatch.setattr(gsb, "_find_vllm_bin", lambda: None)
    backend = gsb.select_backend(_FakeEntry("vllm"))
    assert isinstance(backend, gsb.TransformersBackend)


def test_select_backend_gbquant_legacy_same_as_vllm(monkeypatch):
    """Legacy manifests carry engine=="gbquant" (pre-taxonomy) — select_backend
    must route it identically to "vllm", not just gb_synapse._load_manifest's
    on-read normalization (belt-and-braces for any caller that constructs a
    ModelEntry directly without going through the manifest)."""
    monkeypatch.setattr(gsb, "_find_vllm_bin", lambda: "/fake/vllm")
    backend = gsb.select_backend(_FakeEntry("gbquant"))
    assert isinstance(backend, gsb.VllmBackend)


# ── effective_vram_budget_mb: %-derived T2 sizing ─────────────────────────

def test_effective_budget_zero_fraction_is_shimless(monkeypatch):
    """GB_SYNAPSE_T2_FRACTION=0 must reproduce pre-shim (shimless) sizing
    exactly — the negative control in the T2 validation matrix."""
    monkeypatch.setenv("GB_SYNAPSE_T2_FRACTION", "0")

    class _FakeNvml:
        def mem(self):
            return (0, 8000.0, 12000.0, 0)

    monkeypatch.setitem(sys.modules, "gb_nvml", type(sys)("gb_nvml"))
    sys.modules["gb_nvml"].get_nvml = lambda *_a, **_k: _FakeNvml()

    real_read_text = Path.read_text

    def _fake_read_text(self, *a, **k):
        if str(self) == "/sys/class/greenboost/greenboost/pool_brief":
            return "T1:11GB T2:10/40GB(25%) T3:0/73GB\n"
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", _fake_read_text)

    host_free_mb, effective_free_mb, facts = gsb.effective_vram_budget_mb()
    assert host_free_mb == 8000.0
    assert effective_free_mb == host_free_mb   # T2 contributes nothing at frac=0
    assert facts["t2_fraction"] == 0.0
    assert facts["t2_free_mb"] == pytest.approx(30 * 1024)   # (40-10) GB free


def test_effective_budget_default_fraction_adds_half_t2_free(monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_T2_FRACTION", "0.5")

    class _FakeNvml:
        def mem(self):
            return (0, 8000.0, 12000.0, 0)

    monkeypatch.setitem(sys.modules, "gb_nvml", type(sys)("gb_nvml"))
    sys.modules["gb_nvml"].get_nvml = lambda *_a, **_k: _FakeNvml()

    real_read_text = Path.read_text

    def _fake_read_text(self, *a, **k):
        if str(self) == "/sys/class/greenboost/greenboost/pool_brief":
            return "T1:11GB T2:10/40GB(25%) T3:0/73GB\n"
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", _fake_read_text)

    host_free_mb, effective_free_mb, facts = gsb.effective_vram_budget_mb()
    t2_free_mb = 30 * 1024
    assert effective_free_mb == pytest.approx(host_free_mb + t2_free_mb * 0.5)

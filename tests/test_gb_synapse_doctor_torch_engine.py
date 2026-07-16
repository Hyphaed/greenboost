#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for gb_synapse.doctor()/status()/_format_doctor()'s
torch_engine_ready/torch_engine_env fields (P4.7 of the gb-synapse
unification, 2026-07-16).

CPU-only. No CUDA, no real feeder/NVML calls — monkeypatched."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gb_synapse as gs
import gb_synapse_backends as gsb


def test_format_doctor_ready(tmp_path):
    d = {"host_gpu_name": "Fake GPU", "host_vram_total_mb": 10000,
         "host_ram_total_mb": 60000, "feeders": [],
         "aggregate_vram_mb": 10000, "aggregate_ram_mb": 60000,
         "engine_installed": True, "engine_version": "abc123",
         "hf_token_set": True, "torch_engine_ready": True,
         "torch_engine_env": str(tmp_path / "synapse-torch-env")}
    out = gs._format_doctor(d)
    assert "Torch core: ready" in out
    assert str(tmp_path / "synapse-torch-env") in out


def test_format_doctor_missing():
    d = {"host_gpu_name": "Fake GPU", "host_vram_total_mb": 10000,
         "host_ram_total_mb": 60000, "feeders": [],
         "aggregate_vram_mb": 10000, "aggregate_ram_mb": 60000,
         "engine_installed": True, "engine_version": "abc123",
         "hf_token_set": True, "torch_engine_ready": False,
         "torch_engine_env": ""}
    out = gs._format_doctor(d)
    assert "Torch core: missing" in out
    assert "install-synapse-engine" in out


def test_doctor_reports_torch_engine_absent(monkeypatch):
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: None)
    monkeypatch.setattr(gs.gb_cluster, "feeders", lambda probe=True: [])
    monkeypatch.setattr(gs, "engine_installed", lambda: False)
    monkeypatch.setattr(gs, "engine_version", lambda: "")
    monkeypatch.setattr(gs, "hf_token", lambda: None)

    class _FakeNvml:
        def mem(self):
            return (0, 8000.0, 12000.0, 0)

        def device_name(self):
            return "Fake GPU"

    import types
    fake_mod = types.ModuleType("gb_nvml")
    fake_mod.get_nvml = lambda *_a, **_k: _FakeNvml()
    monkeypatch.setitem(sys.modules, "gb_nvml", fake_mod)

    d = gs.doctor(probe_feeders=False)
    assert d["torch_engine_ready"] is False
    assert d["torch_engine_env"] == ""


def test_status_reports_torch_engine_present(monkeypatch, tmp_path):
    venv = tmp_path / "synapse-torch-env"
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: venv)
    monkeypatch.setattr(gsb, "_find_vllm_bin", lambda: None)
    s = gs.status()
    assert s["torch_engine_ready"] is True
    assert s["torch_engine_env"] == str(venv)

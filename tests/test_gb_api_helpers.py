#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for the small, standalone helper functions gb_api.py's implementation
lands in (P7): gb_cluster.cudart_for(), gb_tiering.t2_pool(),
gb_monitor.gpu_state()/wait_for_vram().
"""
import subprocess
import time

import pytest

import gb_cluster
import gb_monitor
import gb_tiering


# ---------------------------------------------------------------------------
# gb_cluster.cudart_for()
# ---------------------------------------------------------------------------

def test_cudart_for_finds_lib_via_sysconfig_purelib(tmp_path, monkeypatch):
    purelib = tmp_path / "site-packages"
    cuda_dir = purelib / "nvidia" / "cuda_runtime" / "lib"
    cuda_dir.mkdir(parents=True)
    (cuda_dir / "libcudart.so.12").write_bytes(b"")

    def _fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=str(purelib) + "\n", stderr="")
    monkeypatch.setattr(subprocess, "run", _fake_run)

    result = gb_cluster.cudart_for("/fake/bin/python")
    assert result == str(cuda_dir / "libcudart.so.12")


def test_cudart_for_prefers_libcudart_13_over_12(tmp_path, monkeypatch):
    purelib = tmp_path / "site-packages"
    cuda_dir = purelib / "nvidia" / "cuda_runtime" / "lib"
    cuda_dir.mkdir(parents=True)
    (cuda_dir / "libcudart.so.12").write_bytes(b"")
    (cuda_dir / "libcudart.so.13").write_bytes(b"")

    def _fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=str(purelib) + "\n", stderr="")
    monkeypatch.setattr(subprocess, "run", _fake_run)

    result = gb_cluster.cudart_for("/fake/bin/python")
    assert result.endswith("libcudart.so.13")


def test_cudart_for_returns_none_when_nothing_found(monkeypatch, tmp_path):
    def _fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="err")
    monkeypatch.setattr(subprocess, "run", _fake_run)
    # Point the venv-glob fallback and system-toolkit globs somewhere empty,
    # AND make every direct is_file() check False too — this real box may
    # genuinely have a CUDA toolkit installed at /usr/local/cuda/lib64,
    # which isn't reached via .glob() at all (a literal path check).
    monkeypatch.setattr(gb_cluster.Path, "glob", lambda self, pat: iter(()))
    monkeypatch.setattr(gb_cluster.Path, "is_file", lambda self: False)
    result = gb_cluster.cudart_for(str(tmp_path / "bin" / "python"))
    assert result is None


def test_cudart_for_never_raises_on_subprocess_failure(monkeypatch):
    def _boom(cmd, **kw):
        raise OSError("no such interpreter")
    monkeypatch.setattr(subprocess, "run", _boom)
    # Must not raise even though the subprocess call fails.
    gb_cluster.cudart_for("/nonexistent/python")


# ---------------------------------------------------------------------------
# gb_tiering.t2_pool()
# ---------------------------------------------------------------------------

def test_t2_pool_extracts_t2_fields(monkeypatch):
    monkeypatch.setattr(gb_tiering, "tiering_status", lambda probe_gpu=True: {
        "t2_pool_mb": 8192, "t2_allocated_mb": 2048, "t2_available_mb": 6144,
        "t2_pressure": 1, "t1_vram_mb": 12288,  # unrelated field, must not leak in
    })
    result = gb_tiering.t2_pool()
    assert result == {"total_mb": 8192, "allocated_mb": 2048,
                      "available_mb": 6144, "pressure": 1}


def test_t2_pool_zero_fields_on_error(monkeypatch):
    monkeypatch.setattr(gb_tiering, "tiering_status", lambda probe_gpu=True: {"error": "no kmod"})
    result = gb_tiering.t2_pool()
    assert result == {"total_mb": 0, "allocated_mb": 0, "available_mb": 0, "pressure": 0}


# ---------------------------------------------------------------------------
# gb_monitor.gpu_state() / wait_for_vram()
# ---------------------------------------------------------------------------

def test_gpu_state_computes_free_from_used_and_total(monkeypatch):
    fake = gb_monitor.GbSnapshot(gpu_mem_used_mb=2000, gpu_mem_total_mb=12000, gpu_util_pct=45.0)
    monkeypatch.setattr(gb_monitor, "snapshot", lambda probe_gpu=True: fake)
    result = gb_monitor.gpu_state()
    assert result == {"used_mb": 2000, "total_mb": 12000, "free_mb": 10000, "util_pct": 45.0}


def test_gpu_state_free_never_negative(monkeypatch):
    fake = gb_monitor.GbSnapshot(gpu_mem_used_mb=13000, gpu_mem_total_mb=12000)
    monkeypatch.setattr(gb_monitor, "snapshot", lambda probe_gpu=True: fake)
    assert gb_monitor.gpu_state()["free_mb"] == 0


def test_gpu_state_zero_total_means_zero_free(monkeypatch):
    fake = gb_monitor.GbSnapshot(gpu_mem_used_mb=0, gpu_mem_total_mb=0)
    monkeypatch.setattr(gb_monitor, "snapshot", lambda probe_gpu=True: fake)
    assert gb_monitor.gpu_state()["free_mb"] == 0


def test_wait_for_vram_returns_true_when_already_satisfied(monkeypatch):
    monkeypatch.setattr(gb_monitor, "gpu_state", lambda: {"free_mb": 5000})
    assert gb_monitor.wait_for_vram(free_mb=1000, timeout_s=1.0) is True


def test_wait_for_vram_times_out_when_never_satisfied(monkeypatch):
    monkeypatch.setattr(gb_monitor, "gpu_state", lambda: {"free_mb": 100})
    monkeypatch.setattr(gb_monitor.time, "sleep", lambda s: None)  # don't actually sleep
    start = gb_monitor.time.time()
    result = gb_monitor.wait_for_vram(free_mb=99999, timeout_s=0.01, poll_s=0.001)
    assert result is False


def test_wait_for_vram_polls_until_satisfied(monkeypatch):
    calls = {"n": 0}

    def _fake_state():
        calls["n"] += 1
        return {"free_mb": 5000 if calls["n"] >= 3 else 100}
    monkeypatch.setattr(gb_monitor, "gpu_state", _fake_state)
    monkeypatch.setattr(gb_monitor.time, "sleep", lambda s: None)

    result = gb_monitor.wait_for_vram(free_mb=1000, timeout_s=10.0, poll_s=0.001)
    assert result is True
    assert calls["n"] == 3

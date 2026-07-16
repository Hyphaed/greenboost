#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for gb_synapse_backends._validate_placement() (renamed from
_validate_genuine_t2, event kind synapse_vllm_placement ->
synapse_engine_placement, field vllm_compat -> defer_init, Phase 4 backend
wiring, 2026-07-16).

CPU-only. No CUDA, no real shim_stats file — all reads monkeypatched."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gb_synapse_backends as gsb


class _FakeEntry:
    def __init__(self, name="fake", quant="FP8"):
        self.name = name
        self.quant = quant


class _FakeNvml:
    def __init__(self, free_mb, total_mb):
        self._free_mb, self._total_mb = free_mb, total_mb

    def mem(self):
        return (0, self._free_mb, self._total_mb, 0)


def _install_fake_gb_nvml(monkeypatch, free_mb, total_mb):
    import types
    fake_mod = types.ModuleType("gb_nvml")
    fake_mod.get_nvml = lambda *_a, **_k: _FakeNvml(free_mb, total_mb)
    monkeypatch.setitem(sys.modules, "gb_nvml", fake_mod)


def _install_fake_gb_dataflux(monkeypatch):
    import types
    events = []
    fake_mod = types.ModuleType("gb_dataflux")
    fake_mod.emit = lambda event: events.append(event)
    monkeypatch.setitem(sys.modules, "gb_dataflux", fake_mod)
    return events


def test_validate_placement_emits_synapse_engine_placement_kind(monkeypatch):
    monkeypatch.setattr(gsb, "_read_shim_stats", lambda: {})
    _install_fake_gb_nvml(monkeypatch, free_mb=9000, total_mb=10000)
    events = _install_fake_gb_dataflux(monkeypatch)

    gsb._validate_placement(_FakeEntry(), 0.9, {"t2_fraction": 0.5}, "torch")

    assert len(events) == 1
    assert events[0]["kind"] == "synapse_engine_placement"
    assert events[0]["engine"] == "torch"


def test_validate_placement_reads_defer_init_field(monkeypatch):
    monkeypatch.setattr(gsb, "_read_shim_stats", lambda: {"defer_init": "1"})
    _install_fake_gb_nvml(monkeypatch, free_mb=9000, total_mb=10000)
    events = _install_fake_gb_dataflux(monkeypatch)

    gsb._validate_placement(_FakeEntry(), 0.9, {}, "torch")

    assert events[0]["defer_init"] is True
    assert "vllm_compat" not in events[0]


def test_validate_placement_falls_back_to_legacy_vllm_compat_key(monkeypatch):
    """A shim built before the C-side rename (P4.6) only writes
    vllm_compat= — _validate_placement must still read it as defer_init."""
    monkeypatch.setattr(gsb, "_read_shim_stats", lambda: {"vllm_compat": "1"})
    _install_fake_gb_nvml(monkeypatch, free_mb=9000, total_mb=10000)
    events = _install_fake_gb_dataflux(monkeypatch)

    gsb._validate_placement(_FakeEntry(), 0.9, {}, "vllm")

    assert events[0]["defer_init"] is True


def test_validate_placement_rule1_warning_when_t2_used_but_vram_underfilled(monkeypatch):
    monkeypatch.setattr(gsb, "_read_shim_stats",
                        lambda: {"tier_t2_local_cur_mb": "500"})
    _install_fake_gb_nvml(monkeypatch, free_mb=8000, total_mb=10000)  # 20% used
    events = _install_fake_gb_dataflux(monkeypatch)

    gsb._validate_placement(_FakeEntry(), 0.9, {}, "torch")

    assert events[0]["rule1_warning"] is True
    assert events[0]["status"] == "warn"


def test_validate_placement_no_warning_when_vram_well_filled(monkeypatch):
    monkeypatch.setattr(gsb, "_read_shim_stats",
                        lambda: {"tier_t2_local_cur_mb": "500"})
    _install_fake_gb_nvml(monkeypatch, free_mb=500, total_mb=10000)  # 95% used
    events = _install_fake_gb_dataflux(monkeypatch)

    gsb._validate_placement(_FakeEntry(), 0.9, {}, "torch")

    assert events[0]["rule1_warning"] is False
    assert events[0]["status"] == "ok"


def test_validate_placement_never_raises_on_error(monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(gsb, "_read_shim_stats", _raise)
    gsb._validate_placement(_FakeEntry(), 0.9, {}, "torch")  # must not raise

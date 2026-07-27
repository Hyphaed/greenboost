#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for SnapshotRecorder._detect_shim_decisions()'s fix: it now reads
shim_stats via gb_monitor.read_shim_stats() (the canonical parser, with the
/tmp/greenboost_shim_stats fallback every other reader has) instead of
hand-opening /run/greenboost/shim_stats directly. The hand-rolled version
had no fallback, so on a box where the shim fell back to /tmp (the
documented raw-Ollama case), this method silently found nothing and zero
shim_decision events were ever produced — dataflux_decisions/
dataflux_tier_moves would return [] even with a live, working shim.

No real files: gb_monitor.read_shim_stats is monkeypatched directly.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gb_dataflux as gdf
import gb_monitor


class _FakeTelemetryManager:
    def __init__(self):
        self._callbacks = []

    def add_callback(self, fn):
        self._callbacks.append(fn)

    def fire(self, m):
        for cb in self._callbacks:
            cb(m)


class _FakeMetrics:
    def __init__(self, device=0):
        self.device = device
        self.fb_used_mb = 100
        self.fb_free_mb = 900
        self.fb_total_mb = 1000
        self.fb_used_pct = 10.0
        self.gpu_util_pct = 42.0
        self.temp_c = 55.0
        self.power_w = 120.0
        self.shim_phase = "INFERENCE"
        self.gb = None


def _make_recorder(interval_s=9999.0):
    """interval_s huge so the throttled snapshot emit never fires — isolates
    the unthrottled shim_decision path these tests target."""
    tm = _FakeTelemetryManager()
    rec = gdf.SnapshotRecorder(tm, interval_s=interval_s)
    return tm, rec


def test_first_observation_seeds_without_emitting(monkeypatch):
    monkeypatch.setattr(gb_monitor, "read_shim_stats",
                        lambda: {"tier_t2_local_alloc_count": "3",
                                 "tier_t2_local_lifetime_mb": "512", "_stale": False})
    tm, _ = _make_recorder()
    tm.fire(_FakeMetrics())
    events = [e for e in gdf.read_events() if e.get("kind") == "shim_decision"]
    assert events == []  # first observation only seeds _prev_tier


def test_second_observation_with_increased_count_emits_shim_decision(monkeypatch):
    calls = [
        {"tier_t2_local_alloc_count": "3", "tier_t2_local_lifetime_mb": "512", "_stale": False},
        {"tier_t2_local_alloc_count": "5", "tier_t2_local_lifetime_mb": "900", "_stale": False},
    ]
    it = iter(calls)
    monkeypatch.setattr(gb_monitor, "read_shim_stats", lambda: next(it))
    tm, _ = _make_recorder()
    tm.fire(_FakeMetrics())
    tm.fire(_FakeMetrics())

    events = [e for e in gdf.read_events() if e.get("kind") == "shim_decision"]
    assert len(events) == 1
    ev = events[0]
    assert ev["stage"] == "t2_local"
    assert ev["n_items"] == 2          # 5 - 3
    assert ev["bytes_mb"] == 388        # 900 - 512


def test_empty_shim_stats_produces_no_event_and_does_not_raise(monkeypatch):
    monkeypatch.setattr(gb_monitor, "read_shim_stats", lambda: {})
    tm, _ = _make_recorder()
    tm.fire(_FakeMetrics())  # must not raise
    events = [e for e in gdf.read_events() if e.get("kind") == "shim_decision"]
    assert events == []


def test_stale_shim_stats_produces_no_event(monkeypatch):
    """A stale read (shim process gone, file not refreshed) must be skipped
    entirely — including not disturbing _prev_tier's seeded state, so a
    later fresh read still diffs correctly against the last real
    observation rather than treating the stale gap as a fresh baseline."""
    calls = [
        {"tier_t2_local_alloc_count": "3", "tier_t2_local_lifetime_mb": "512", "_stale": False},
        {"tier_t2_local_alloc_count": "99", "tier_t2_local_lifetime_mb": "9999", "_stale": True},
        {"tier_t2_local_alloc_count": "4", "tier_t2_local_lifetime_mb": "600", "_stale": False},
    ]
    it = iter(calls)
    monkeypatch.setattr(gb_monitor, "read_shim_stats", lambda: next(it))
    tm, _ = _make_recorder()
    tm.fire(_FakeMetrics())  # seed at 3/512
    tm.fire(_FakeMetrics())  # stale — skipped entirely
    tm.fire(_FakeMetrics())  # fresh again, diffs against the seeded 3/512, not the stale 99

    events = [e for e in gdf.read_events() if e.get("kind") == "shim_decision"]
    assert len(events) == 1
    assert events[0]["n_items"] == 1     # 4 - 3, NOT 4 - 99
    assert events[0]["bytes_mb"] == 88   # 600 - 512, NOT a negative/garbage diff


def test_feeder_node_recorder_never_reads_local_shim_stats(monkeypatch):
    """Only the host recorder (_node in (None, 'host')) reads local
    shim_stats — a feeder-scoped recorder has no local shim to read and
    must not call read_shim_stats() at all."""
    called = []
    monkeypatch.setattr(gb_monitor, "read_shim_stats", lambda: called.append(1) or {})
    tm = _FakeTelemetryManager()
    rec = gdf.SnapshotRecorder(tm, interval_s=9999.0, node="feeder0")
    tm.fire(_FakeMetrics())
    assert called == []

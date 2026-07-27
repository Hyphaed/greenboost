#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_mem_pool.MemPoolManager's dataflux backfill (P5) , the one
decision-making module in the codebase with zero gb_dataflux reference at
all. trim()/trim_all() now emit a mem_pool_trim event with real before/
after allocated_mb, not just "trim was called".

No real CUDA: _Pool.stats()/_Pool.trim() are monkeypatched so this runs
identically with or without a GPU present.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gb_mem_pool as gmp


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    p = tmp_path / "dataflux.jsonl"
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(p))
    import gb_dataflux
    gb_dataflux._READ_EVENTS_MEMO.clear()
    yield p
    gb_dataflux._READ_EVENTS_MEMO.clear()


def _bare_manager():
    mgr = object.__new__(gmp.MemPoolManager)
    mgr._pools = {n: object.__new__(gmp._Pool) for n in gmp._POOL_NAMES}
    for name, pool in mgr._pools.items():
        pool.name = name
    return mgr


def test_trim_emits_before_after_and_reclaimed(log_path, monkeypatch):
    mgr = _bare_manager()
    stats_seq = iter([
        gmp.PoolStats(pool_name="weights", allocated_mb=100.0),
        gmp.PoolStats(pool_name="weights", allocated_mb=40.0),
    ])
    monkeypatch.setattr(mgr._pools["weights"], "stats", lambda: next(stats_seq))
    monkeypatch.setattr(mgr._pools["weights"], "trim", lambda target_mb=None: None)

    mgr.trim("weights")

    import gb_dataflux
    events = [e for e in gb_dataflux.read_events() if e.get("kind") == "mem_pool_trim"]
    assert len(events) == 1
    ev = events[0]
    assert ev["pool"] == "weights"
    assert ev["allocated_before_mb"] == 100.0
    assert ev["allocated_after_mb"] == 40.0
    assert ev["reclaimed_mb"] == 60.0


def test_trim_all_emits_one_event_per_pool(log_path, monkeypatch):
    mgr = _bare_manager()
    for name in gmp._POOL_NAMES:
        pool = mgr._pools[name]
        monkeypatch.setattr(pool, "stats", lambda n=name: gmp.PoolStats(pool_name=n, allocated_mb=10.0))
        monkeypatch.setattr(pool, "trim", lambda: None)

    mgr.trim_all()

    import gb_dataflux
    events = [e for e in gb_dataflux.read_events() if e.get("kind") == "mem_pool_trim"]
    assert {e["pool"] for e in events} == set(gmp._POOL_NAMES)


def test_reclaimed_mb_never_negative_when_pool_grows(log_path, monkeypatch):
    """A pool that (unusually) grew between the before/after stats read must
    report reclaimed_mb=0, not a nonsensical negative number."""
    mgr = _bare_manager()
    stats_seq = iter([
        gmp.PoolStats(pool_name="latents", allocated_mb=10.0),
        gmp.PoolStats(pool_name="latents", allocated_mb=25.0),
    ])
    monkeypatch.setattr(mgr._pools["latents"], "stats", lambda: next(stats_seq))
    monkeypatch.setattr(mgr._pools["latents"], "trim", lambda target_mb=None: None)

    mgr.trim("latents")

    import gb_dataflux
    ev = [e for e in gb_dataflux.read_events() if e.get("kind") == "mem_pool_trim"][0]
    assert ev["reclaimed_mb"] == 0.0

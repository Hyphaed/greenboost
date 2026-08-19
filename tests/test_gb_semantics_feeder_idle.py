"""feeder_idle_while_host_saturated must not fire on non-feeders or on a
feeder that is simply not connected.

Observed live 2026-08-18: the segment reported matched=true with evidence
{"host": 173, "?": 0} while the only configured feeder (omen) was unreachable.
Both halves of that were wrong. "?" is dataflux's unknown-node bucket, not a
feeder, and an unreachable feeder is `feeder_unreachable`'s case, which this
segment's own doc defers to.
"""
from __future__ import annotations

import pytest

import gb_semantics as gs


def _stub(monkeypatch, *, host_fill, items, reachable):
    def fake_resolve(metric, entity_id=None, **kw):
        return {
            "vram_fill_pct": {"metric": "vram_fill_pct", "value": host_fill},
            "feeder_items": {"metric": "feeder_items", "value": items},
            "feeder_reachable": {"metric": "feeder_reachable", "value": reachable},
        }[metric]
    monkeypatch.setattr(gs, "resolve", fake_resolve)


def test_unknown_node_bucket_is_not_a_feeder(monkeypatch):
    """The exact live shape: only host did work, plus the "?" bucket."""
    _stub(monkeypatch, host_fill=88.8, items={"host": 173, "?": 0}, reachable={})
    matched, _ = gs._seg_feeder_idle_while_host_saturated()
    assert matched is False


def test_unreachable_feeder_does_not_count_as_idle(monkeypatch):
    """feeder_unreachable owns this case. Firing both reports two violations
    for one condition, with different remediations."""
    _stub(monkeypatch, host_fill=88.8, items={"host": 173, "omen": 0},
          reachable={"omen": False})
    matched, _ = gs._seg_feeder_idle_while_host_saturated()
    assert matched is False


def test_reachable_idle_feeder_still_violates(monkeypatch):
    """The condition the segment exists for must keep matching."""
    _stub(monkeypatch, host_fill=88.8, items={"host": 173, "omen": 0},
          reachable={"omen": True})
    matched, ev = gs._seg_feeder_idle_while_host_saturated()
    assert matched is True
    assert [e["metric"] for e in ev] == [
        "vram_fill_pct", "feeder_items", "feeder_reachable"]


def test_unknown_reachability_is_treated_as_reachable(monkeypatch):
    """A feeder absent from the reachability map must still be able to trip
    this, or an unavailable probe would silently suppress a real violation."""
    _stub(monkeypatch, host_fill=88.8, items={"host": 173, "omen": 0},
          reachable={})
    matched, _ = gs._seg_feeder_idle_while_host_saturated()
    assert matched is True


def test_busy_feeder_never_matches(monkeypatch):
    _stub(monkeypatch, host_fill=88.8, items={"host": 173, "omen": 42},
          reachable={"omen": True})
    assert gs._seg_feeder_idle_while_host_saturated()[0] is False


def test_host_below_target_never_matches(monkeypatch):
    """The rule is about a feeder idling WHILE the host is the bottleneck."""
    _stub(monkeypatch, host_fill=40.0, items={"host": 173, "omen": 0},
          reachable={"omen": True})
    assert gs._seg_feeder_idle_while_host_saturated()[0] is False

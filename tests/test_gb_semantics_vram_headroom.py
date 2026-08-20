#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""vram_headroom_pct / vram_headroom_exhausted , the high side of Rule #1.

`rule1_underfilled` has always watched VRAM from below. Nothing watched it from
above, so on 2026-08-20 a card at 97.9% fill (254 MB free of 12227) produced no
governed verdict at all, while the same card at 84.9% produced a `violation`.
These tests pin the three behaviours that matter: it fires when the reserve is
gone, it stays quiet inside the band, and it says "cannot tell" rather than
"healthy" when fill cannot be read.
"""
from __future__ import annotations

import pytest

import gb_semantics


@pytest.fixture(autouse=True)
def _fresh():
    gb_semantics.load(force=True)
    yield


def _patch_fill(monkeypatch, value):
    """Drive the segment through the real resolve() path by replacing only the
    underlying fill resolver, so provenance/threshold wiring is exercised."""
    def fake(entity_id, window_s):
        if value is None:
            return {"value": None, "unit": "percent",
                    "raw_source": "unavailable", "error": "test: no source"}
        return {"value": value, "unit": "percent",
                "raw_source": "test.fb_phys_used_pct", "freshness_s": 1.0}
    monkeypatch.setattr(gb_semantics, "_res_vram_fill_pct", fake)


def test_headroom_is_derived_from_physical_fill(monkeypatch):
    _patch_fill(monkeypatch, 97.9)
    r = gb_semantics.resolve("vram_headroom_pct")
    assert r["value"] == pytest.approx(2.1, abs=0.05)
    assert r["governed"] is True
    # provenance must name where it came from, not just "derived"
    assert "test.fb_phys_used_pct" in r["provenance"]["raw_source"]


def test_segment_matches_when_the_reserve_is_gone(monkeypatch):
    _patch_fill(monkeypatch, 97.9)          # the measured 2026-08-20 session
    out = gb_semantics.evaluate_segment("vram_headroom_exhausted")
    assert out["matched"] is True


def test_segment_clear_inside_the_target_band(monkeypatch):
    _patch_fill(monkeypatch, 88.0)          # Rule #1's intended operating point
    out = gb_semantics.evaluate_segment("vram_headroom_exhausted")
    assert out["matched"] is False


def test_segment_clear_when_underfilled(monkeypatch):
    """Under-fill is rule1_underfilled's job. This segment must not double-report
    it, or a single condition raises two different alarms."""
    _patch_fill(monkeypatch, 50.0)
    out = gb_semantics.evaluate_segment("vram_headroom_exhausted")
    assert out["matched"] is False


def test_unknown_fill_is_none_not_false(monkeypatch):
    """A clean bill of health inferred from absent data is the exact failure
    the governed layer exists to prevent."""
    _patch_fill(monkeypatch, None)
    out = gb_semantics.evaluate_segment("vram_headroom_exhausted")
    assert out["matched"] is None


def test_never_use_trap_is_declared():
    """fb_free_mb is shim-inflated and reads gigabytes free on a card with
    megabytes free , the trap must be named, not just avoided in code."""
    m = gb_semantics.load()["metrics"]["vram_headroom_pct"]
    assert any(t["field"] == "fb_free_mb" for t in (m.never_use or []))

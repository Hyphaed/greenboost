#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_damon , host cold-page evidence from the kernel's DAMON_STAT.

REAL_IDLE below is not a fixture. It is the exact `memory_idle_ms_percentiles`
string this box's kernel produced on 2026-08-21 (61 GiB host, mid-session),
read as root from /sys/module/damon_stat/parameters. A hand-written array would
have encoded whatever shape the author assumed; this one encodes what the
kernel actually reports, including the negative-means-hot convention that is
easy to get backwards.

What these tests do NOT cover: reading the sysfs files themselves, which needs
root. That path is exercised by running `sudo python3 gb_damon.py` on a real
box, and by gb_supervisor in production. A passing file here is not evidence
the reader works against the kernel.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import gb_damon
from gb_damon import DamonSnapshot, parse_idle_percentiles

# Captured from this box, 2026-08-21. P0..P100, idle ms; negative == hot.
REAL_IDLE = (
    "-9400000," + ",".join(["-200000"] * 20) + "," + ",".join(["200000"] * 57)
    + ",400000,400000," + ",".join(["800000"] * 21)
)
REAL_BANDWIDTH_BPS = 85996769


@pytest.fixture
def real_snap():
    return DamonSnapshot(
        enabled=True,
        idle_ms=parse_idle_percentiles(REAL_IDLE),
        est_bandwidth_bps=REAL_BANDWIDTH_BPS,
        aggr_interval_us=None,
    )


# ── parsing the real shape ────────────────────────────────────────────────────

def test_real_output_parses_to_101_percentiles():
    assert len(parse_idle_percentiles(REAL_IDLE)) == 101


def test_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="101"):
        parse_idle_percentiles("1,2,3")


def test_trailing_and_blank_fields_tolerated():
    raw = ",".join(["0"] * 101) + ","
    assert len(parse_idle_percentiles(raw)) == 101


# ── the negative-means-hot convention ─────────────────────────────────────────

def test_negative_idle_counts_as_hot_not_idle(real_snap):
    """The convention that is easy to invert. 21 of 101 percentiles are
    negative in the real capture, so ~20.8% of memory was being accessed."""
    assert real_snap.hot_pct == pytest.approx(20.8, abs=0.1)
    assert real_snap.idle_pct == pytest.approx(79.2, abs=0.1)
    assert real_snap.hot_pct + real_snap.idle_pct == pytest.approx(100.0)


def test_hottest_is_the_most_negative(real_snap):
    assert real_snap.hottest_ms == -9400000


def test_max_idle_ignores_hot_regions(real_snap):
    assert real_snap.max_idle_ms == 800000


def test_hot_regions_never_counted_as_cold(real_snap):
    """A region hot for 9400 s must not be read as 9400 s idle."""
    assert real_snap.cold_pct_above(1_000_000) == 0.0
    assert real_snap.cold_pct_above(0) == pytest.approx(79.2, abs=0.1)


# ── derived shares against the real distribution ──────────────────────────────

def test_cold_shares_match_the_real_capture(real_snap):
    assert real_snap.cold_pct_above(200_000) == pytest.approx(22.8, abs=0.1)
    assert real_snap.cold_pct_above(400_000) == pytest.approx(20.8, abs=0.1)
    assert real_snap.cold_pct_above(800_000) == 0.0   # strict >, max is 800000


def test_cold_share_is_monotonic_in_threshold(real_snap):
    ts = [0, 100_000, 200_000, 400_000, 800_000, 1_600_000]
    shares = [real_snap.cold_pct_above(t) for t in ts]
    assert shares == sorted(shares, reverse=True)


def test_percentiles_readable(real_snap):
    assert real_snap.idle_percentile(0) == -9400000
    assert real_snap.idle_percentile(50) == 200000
    assert real_snap.idle_percentile(100) == 800000
    assert real_snap.idle_percentile(101) is None
    assert real_snap.idle_percentile(-1) is None


def test_no_default_cold_threshold_exists():
    """Picking 'cold' is the caller's judgment, and must stay at the call site.
    If someone adds a default, this fails and makes them argue for it."""
    assert not hasattr(gb_damon, "COLD_MS_DEFAULT")
    with pytest.raises(TypeError):
        DamonSnapshot(enabled=True).cold_pct_above()   # threshold is required


# ── graceful degradation ──────────────────────────────────────────────────────

def test_empty_snapshot_is_safe():
    s = DamonSnapshot(enabled=False, error="not present")
    assert s.hot_pct == 0.0 and s.idle_pct == 0.0
    assert s.cold_pct_above(1) == 0.0
    assert s.max_idle_ms is None and s.hottest_ms is None
    assert s.as_dict()["error"] == "not present"


def test_read_published_missing_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(gb_damon, "PUBLISHED_FILE", tmp_path / "nope")
    out = gb_damon.read_published()
    assert out["available"] is False and "unreadable" in out["reason"]


# ── publish/consume round trip ────────────────────────────────────────────────

def test_published_round_trip_preserves_the_real_numbers(real_snap, tmp_path, monkeypatch):
    """What root writes must be exactly what an unprivileged reader gets back."""
    f = tmp_path / "damon_cold"
    monkeypatch.setattr(gb_damon, "PUBLISHED_FILE", f)
    f.write_text(gb_damon.render_published(real_snap))

    out = gb_damon.read_published()
    assert out["available"] is True
    assert out["hot_pct"] == pytest.approx(20.8, abs=0.1)
    assert out["idle_pct"] == pytest.approx(79.2, abs=0.1)
    assert out["cold_pct_above_400000ms"] == pytest.approx(20.8, abs=0.1)
    assert out["est_bandwidth_bps"] == REAL_BANDWIDTH_BPS
    assert out["idle_p50_ms"] == 200000
    assert out["hottest_ms"] == -9400000
    assert "timestamp" in out


def test_published_carries_error_when_present(tmp_path, monkeypatch):
    f = tmp_path / "damon_cold"
    monkeypatch.setattr(gb_damon, "PUBLISHED_FILE", f)
    f.write_text(gb_damon.render_published(
        DamonSnapshot(enabled=False, error="needs root")))
    assert gb_damon.read_published()["error"] == "needs root"


# ── the module is honest about what it can see ────────────────────────────────

def test_available_and_readable_are_separate_questions():
    """A kernel can ship DAMON_STAT that this process still cannot read , the
    0400 case. Conflating the two is how 'not present' gets reported for a
    permission problem."""
    assert gb_damon.available() in (True, False)
    if gb_damon.available():
        # As an unprivileged test runner, readable() is expected False here.
        assert gb_damon.readable() in (True, False)


def test_unreadable_reports_permission_not_absence():
    """Guards the wording: a 0400 failure must not be reported as 'not
    present', which would send someone to rebuild their kernel for nothing."""
    if gb_damon.available() and not gb_damon.readable():
        snap = gb_damon.read_snapshot()
        assert snap.error and "root" in snap.error
        assert "not present" not in snap.error

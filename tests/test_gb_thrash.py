#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_thrash , tier-thrashing detection ported from UVM.

No GPU, no daemon, no torch. Time is injected via the `now` parameter rather
than slept, so the whole file runs in milliseconds.

These tests cover the counting logic , the part ported from upstream. They do
NOT establish that the detector works on a real workload; that came from an
actual FLUX.2-klein-9B run through ai-forge (see gb_thrash's module header and
docs/performance-casebook.md C-10), which is what calibrated lapse_s and what
put real tier_move events through the hook. A passing file here is not
evidence the feature works.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from gb_thrash import (
    PIN_THRESHOLD_DEFAULT,
    THRESHOLD_DEFAULT,
    ThrashDetector,
)

T1, T2, T3 = "T1_HBM", "T2_DDR", "T3_NVME"


@pytest.fixture
def det():
    return ThrashDetector(lapse_s=10.0, threshold=3, pin_threshold=10, max_resets=4)


def pingpong(det, name, cycles, t0=1000.0, step=1.0):
    """`cycles` full T1->T2->T1 round trips, `step` seconds apart."""
    t = t0
    for _ in range(cycles):
        det.record_move(name, T1, T2, now=t); t += step
        det.record_move(name, T2, T1, now=t); t += step
    return t


# ── the core counting rule ────────────────────────────────────────────────────

def test_reversal_within_lapse_counts_as_event(det):
    det.record_move("unet", T1, T2, now=100.0)
    st = det.record_move("unet", T2, T1, now=101.0)      # reverses, 1s < 10s
    assert st.num_thrashing_events == 1


def test_one_way_migration_is_not_thrashing(det):
    """A fast migration is a migration. Only a reversal is thrashing , this is
    the GB-Tiering stand-in for UVM requiring two different processors."""
    det.record_move("unet", T1, T2, now=100.0)
    st = det.record_move("unet", T2, T3, now=101.0)      # onward, not back
    assert st.num_thrashing_events == 0


def test_reversal_outside_lapse_is_not_an_event(det):
    det.record_move("unet", T1, T2, now=100.0)
    st = det.record_move("unet", T2, T1, now=115.0)      # 15s > 10s lapse
    assert st.num_thrashing_events == 0


def test_threshold_declares_thrashing(det):
    pingpong(det, "unet", cycles=1)
    assert not det.is_thrashing("unet")
    pingpong(det, "unet", cycles=2, t0=1010.0)
    assert det.is_thrashing("unet")
    assert "unet" in det.thrashing_components()


def test_first_move_never_thrashes(det):
    st = det.record_move("unet", T1, T2, now=100.0)
    assert st.num_thrashing_events == 0
    assert st.total_moves == 1


# ── epoch reset ───────────────────────────────────────────────────────────────

def test_epoch_of_quiet_resets_counters(det):
    pingpong(det, "unet", cycles=3)
    assert det.is_thrashing("unet")
    # One epoch of silence, then a move.
    det.record_move("unet", T1, T2, now=1000.0 + det.epoch_s + 50.0)
    assert not det.is_thrashing("unet")


def test_reset_is_counted(det):
    pingpong(det, "unet", cycles=2)
    det.record_move("unet", T1, T2, now=1000.0 + det.epoch_s + 50.0)
    assert det._state["unet"].num_resets == 1


def test_max_resets_stops_counting(det):
    """After max_resets, upstream gives up on the page. Same here."""
    t = 1000.0
    for _ in range(det.max_resets + 2):
        t = pingpong(det, "unet", cycles=2, t0=t)
        t += det.epoch_s + 50.0
        det.record_move("unet", T1, T2, now=t)
        t += 1.0
    st = det._state["unet"]
    assert st.num_resets >= det.max_resets
    # Counting has stopped: further ping-pong adds no events.
    before = st.num_thrashing_events
    pingpong(det, "unet", cycles=5, t0=t)
    assert st.num_thrashing_events == before


# ── pin recommendation ────────────────────────────────────────────────────────

def test_pin_recommended_only_past_pin_threshold(det):
    pingpong(det, "unet", cycles=3)
    assert det.is_thrashing("unet")
    assert not det.should_pin("unet")
    pingpong(det, "unet", cycles=4, t0=1010.0)
    assert det.should_pin("unet")
    assert "unet" in det.stats()["pin_recommended"]


def test_upstream_defaults_are_preserved():
    """The counting constants are ported verbatim; only the clock is rescaled."""
    d = ThrashDetector()
    assert d.threshold == THRESHOLD_DEFAULT == 3
    assert d.pin_threshold == PIN_THRESHOLD_DEFAULT == 10


# ── isolation between components ──────────────────────────────────────────────

def test_components_are_independent(det):
    pingpong(det, "unet", cycles=3)
    det.record_move("vae", T1, T2, now=1000.0)
    assert det.is_thrashing("unet")
    assert not det.is_thrashing("vae")
    assert det.stats()["thrashing_count"] == 1


# ── calibration support ───────────────────────────────────────────────────────

def test_intervals_recorded_for_calibration(det):
    """The raw intervals are the point: lapse_s is provisional and these are
    what will eventually replace it with a measured value."""
    det.record_move("unet", T1, T2, now=100.0)
    det.record_move("unet", T2, T1, now=103.0)
    det.record_move("unet", T1, T2, now=108.0)
    st = det._state["unet"]
    assert st.intervals == [3.0, 5.0]
    assert st.median_interval() == 4.0
    assert det.stats()["min_interval_s"] == 3.0


def test_interval_history_is_bounded(det):
    t = 1000.0
    for _ in range(200):
        det.record_move("unet", T1, T2, now=t)
        t += 1.0
    assert len(det._state["unet"].intervals) <= 32


# ── replay of REAL captured tier_move data ────────────────────────────────────
#
# These are not fixtures. Every timestamp below was recorded by dataflux during
# a real FLUX.2-klein-9B generation through ai-forge (2026-08-21), on an
# 11.26 GB RTX 5070 that was simultaneously running a desktop and a browser
# playing video. They exist because a reasoned 30 s window looked fine in unit
# tests and was wrong against the fast path: fp8's legitimate encode->denoise
# bounce is 21-23 s, so every healthy fp8 run scored a thrash event.

REAL_RUNS = {
    # (from_tier, to_tier) transitions of 'transformer', absolute unix ts
    "fp8_714119": [(1787348148.4653084, "T1_HBM", "T2_DDR"),
                   (1787348171.3138611, "T2_DDR", "T1_HBM"),   # +21.48s reversal
                   (1787348440.2150023, "T1_HBM", "T2_DDR")],
    "fp8_729664": [(1787348532.2432246, "T1_HBM", "T2_DDR"),
                   (1787348553.7213712, "T2_DDR", "T1_HBM"),   # +22.85s reversal
                   (1787348818.1293929, "T1_HBM", "T2_DDR")],
    "near_lossless_706019": [(1787347522.6237030, "T1_HBM", "T2_DDR"),
                             (1787347801.3472078, "T2_DDR", "T1_HBM"),  # +278.72s
                             (1787348074.3145256, "T1_HBM", "T2_DDR")],
}


@pytest.mark.parametrize("run_name", sorted(REAL_RUNS))
def test_real_healthy_runs_are_not_flagged(run_name):
    """No healthy real run may score a single thrash event.

    This is the regression guard for the false positive that shipping a
    reasoned-not-measured threshold introduced. If someone raises lapse_s back
    above ~21 s, the two fp8 cases fail."""
    d = ThrashDetector()                      # production defaults, not tuned
    for ts, frm, to in REAL_RUNS[run_name]:
        d.record_move("transformer", frm, to, now=ts)
    assert d._state["transformer"].num_thrashing_events == 0
    assert not d.is_thrashing("transformer")


def test_lapse_stays_below_fastest_measured_legitimate_bounce():
    """The invariant the real data established, stated as a test."""
    d = ThrashDetector()
    cal = d.stats()["params"]["calibration"]
    assert d.lapse_s < cal["fastest_legitimate_s"]
    assert cal["fastest_legitimate_s"] / d.lapse_s >= 2.0   # keep real margin


def test_genuine_ping_pong_is_still_caught():
    """Tightening the window must not blind the detector to the real thing."""
    d = ThrashDetector()
    t = 1000.0
    for _ in range(3):                        # sub-lapse bounces
        d.record_move("unet", "T1_HBM", "T2_DDR", now=t); t += 2.0
        d.record_move("unet", "T2_DDR", "T1_HBM", now=t); t += 2.0
    assert d.is_thrashing("unet")


def test_calibration_block_is_honest(det):
    """Honesty is a tested property here, not a comment.

    lapse_s was calibrated against a real FLUX.2-klein-9B run (2026-08-21),
    not a fixture. What the calibration block must keep saying is how thin the
    evidence is: two samples, one pipeline, and an explicit statement of what
    it does not establish. If someone widens the claim, this fails."""
    cal = det.stats()["params"]["calibration"]
    assert det.stats()["params"]["lapse_calibrated"] is True
    assert cal["runs"] == 3
    assert cal["concurrent_load"]           # measurement conditions recorded
    flat = [v for vals in cal["legitimate_reversal_s"].values() for v in vals]
    assert cal["fastest_legitimate_s"] == min(flat)
    assert "does NOT establish" in cal["caveat"]


# ── env overrides ─────────────────────────────────────────────────────────────

def test_env_overrides(monkeypatch):
    monkeypatch.setenv("GB_THRASH_LAPSE_S", "2.5")
    monkeypatch.setenv("GB_THRASH_THRESHOLD", "7")
    d = ThrashDetector()
    assert d.lapse_s == 2.5
    assert d.threshold == 7


def test_bad_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("GB_THRASH_LAPSE_S", "not-a-number")
    d = ThrashDetector()
    assert d.lapse_s == 10.0   # LAPSE_S_DEFAULT, measured


# ── the observation-only guarantee ────────────────────────────────────────────

def test_detector_exposes_no_mutating_placement_api(det):
    """Step 3 of the plan ships this behind a metric before it gates anything.
    If someone later adds an actuator, this test should fail and make them
    say so deliberately."""
    forbidden = {"promote", "demote", "evict", "pin", "apply", "actuate"}
    assert forbidden.isdisjoint(dir(det))

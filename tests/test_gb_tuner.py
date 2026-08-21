"""gb_tuner decides; nothing here touches hardware.

That is the point of the module: the awkward cases , a measurement taken
before an actuation landed, a threshold crossed by one sample, a target this
box cannot reach at all , are the ones that were previously only discoverable
live, on a machine that was busy serving. They are written down here instead.

Mirrors GameNative's TunerHarvestGuardTest.kt in intent (audited 2026-08-20),
with our own thresholds.
"""
from __future__ import annotations

import gb_tuner as T


def _snap(**kw):
    return T.TunerSnapshot(**kw)


def _warm(state, p=None, tok_s=5.0):
    """Feed enough throughput samples that the baseline is trusted.

    Deliberately warms on a card drawing well under its limit, so warming
    itself never triggers a harvest , a helper that actuates would leave every
    test that uses it starting mid-settle.
    """
    p = p or T.TunerPolicy()
    for _ in range(p.min_baseline_samples):
        _, state = T.decide(_snap(tok_s=tok_s, gpu_util_pct=30.0,
                                  t2_overflow_mb=6000.0,
                                  power_draw_w=40.0, power_limit_w=160.0),
                            state, p)
    assert state.settle_remaining == 0, "warming must not actuate"
    return state


# ── purity ─────────────────────────────────────────────────────────────────

def test_decide_does_not_mutate_the_state_it_was_given():
    st = T.TunerState()
    _, nxt = T.decide(_snap(tok_s=5.0, gpu_util_pct=95.0), st)
    assert st.ticks == 0 and nxt.ticks == 1
    assert st.frozen is not nxt.frozen


def test_nothing_is_decided_while_nothing_is_served():
    out, st = T.decide(_snap(serving=False), T.TunerState())
    assert st.bottleneck == T.Bottleneck.IDLE
    assert [d.action for d in out] == ["observe"]


# ── settle cycles ──────────────────────────────────────────────────────────

def test_a_reading_during_settle_is_not_acted_on():
    """The bug this exists to prevent: measuring before the change landed and
    rolling back on data that describes the previous state."""
    p = T.TunerPolicy()
    st = _warm(T.TunerState(), p)
    out, st = T.decide(_snap(tok_s=5.0, gpu_util_pct=100.0, t2_overflow_mb=6000.0,
                             power_draw_w=49.8, power_limit_w=300.0,
                             sm_clock_mhz=2685.0, sm_clock_max_mhz=2700.0), st, p)
    assert any(d.action == "harvest" for d in out)
    assert st.settle_remaining == p.settle_cycles
    # A catastrophic-looking reading arrives immediately afterwards.
    out2, st2 = T.decide(_snap(tok_s=0.1, gpu_util_pct=100.0, t2_overflow_mb=6000.0,
                               power_draw_w=49.8, power_limit_w=300.0,
                               sm_clock_mhz=2560.0, sm_clock_max_mhz=2700.0), st, p)
    assert [d.action for d in out2] == ["observe"]
    assert "settling" in out2[0].reason
    assert st2.settle_remaining == p.settle_cycles - 1


# ── latching hysteresis ────────────────────────────────────────────────────

def test_one_sample_below_the_enter_threshold_does_not_release_gpu_bound():
    p = T.TunerPolicy()
    st = T.TunerState()
    _, st = T.decide(_snap(tok_s=5.0, gpu_util_pct=90.0), st, p)
    assert st.bottleneck == T.Bottleneck.GPU_BOUND
    # 80% is below the 85 enter threshold but above the 75 release threshold.
    _, st = T.decide(_snap(tok_s=5.0, gpu_util_pct=80.0), st, p)
    assert st.bottleneck == T.Bottleneck.GPU_BOUND, "released on a single dip"
    _, st = T.decide(_snap(tok_s=5.0, gpu_util_pct=70.0), st, p)
    assert st.bottleneck != T.Bottleneck.GPU_BOUND


def test_spilled_weights_plus_low_utilisation_is_bandwidth_bound():
    _, st = T.decide(_snap(tok_s=3.0, gpu_util_pct=30.0, t2_overflow_mb=6687.0),
                     T.TunerState())
    assert st.bottleneck == T.Bottleneck.BANDWIDTH_BOUND


# ── the n=3 population guard ───────────────────────────────────────────────

def test_no_harvest_before_the_baseline_is_trusted():
    """Advising off three samples is the 2026-08-18 defect, in one line."""
    out, _ = T.decide(_snap(tok_s=3.0, gpu_util_pct=100.0, t2_overflow_mb=6687.0,
                            power_draw_w=49.8, power_limit_w=300.0,
                            sm_clock_mhz=2685.0, sm_clock_max_mhz=2700.0),
                      T.TunerState())
    assert all(d.action != "harvest" for d in out)
    assert "sample" in out[0].reason


# ── harvest ────────────────────────────────────────────────────────────────
#
# The lever here is the SM CLOCK, not the power limit, and that is a
# correction rather than a preference (2026-08-20). Under power-first
# classification a card near its board limit is BY DEFINITION compute-bound,
# so a power-limit harvest could only ever fire on a machine the engine had
# just called GPU-bound , dead code with a plausible docstring. These fixtures
# use the real reading from this box mid-decode: utilization.gpu 100% while
# the board draws 49.8 W of a 300 W limit with the SMs at 2685 MHz. That is
# what "spending boost to wait on PCIe" looks like in numbers, and the
# utilisation figure is the one that lies.

def test_harvest_gives_back_the_clock_when_boost_is_spent_waiting():
    p = T.TunerPolicy()
    st = _warm(T.TunerState(), p)
    out, st = T.decide(_snap(tok_s=5.0, gpu_util_pct=100.0, t2_overflow_mb=6687.0,
                             power_draw_w=49.8, power_limit_w=300.0,
                             sm_clock_mhz=2685.0, sm_clock_max_mhz=2700.0), st, p)
    h = [d for d in out if d.action == "harvest"]
    assert h and h[0].lever == "gpu_clocks_locked"
    assert h[0].value < 2700 and h[0].verify is True


def test_util_at_100_pct_does_not_block_the_harvest():
    """The regression that made the harvest unreachable.

    utilization.gpu counts a kernel stalled on a PCIe read as fully busy. If
    the classifier trusts it, this exact snapshot , the measured one , reads
    GPU-bound and the one decision this engine exists to make never fires.
    """
    p = T.TunerPolicy()
    st = _warm(T.TunerState(), p)
    out, st = T.decide(_snap(tok_s=5.0, gpu_util_pct=100.0, t2_overflow_mb=6687.0,
                             power_draw_w=49.8, power_limit_w=300.0,
                             sm_clock_mhz=2685.0, sm_clock_max_mhz=2700.0), st, p)
    assert st.bottleneck == T.Bottleneck.BANDWIDTH_BOUND
    assert any(d.action == "harvest" for d in out)


def test_nothing_is_harvested_without_clock_telemetry():
    """No sensor, no verdict , never a harvest computed off a missing reading."""
    p = T.TunerPolicy()
    st = _warm(T.TunerState(), p)
    out, _ = T.decide(_snap(tok_s=5.0, gpu_util_pct=100.0, t2_overflow_mb=6687.0,
                            power_draw_w=49.8, power_limit_w=300.0), st, p)
    assert all(d.action != "harvest" for d in out)
    assert any("clock telemetry" in d.reason for d in out)


def test_nothing_is_harvested_when_the_clock_is_already_low():
    """Nothing is being held that is not being used , leave it alone."""
    p = T.TunerPolicy()
    st = _warm(T.TunerState(), p)
    out, _ = T.decide(_snap(tok_s=5.0, gpu_util_pct=100.0, t2_overflow_mb=6687.0,
                            power_draw_w=49.8, power_limit_w=300.0,
                            sm_clock_mhz=900.0, sm_clock_max_mhz=2700.0), st, p)
    assert all(d.action != "harvest" for d in out)


def test_harvesting_stops_at_the_floor():
    p = T.TunerPolicy()
    st = _warm(T.TunerState(), p)
    st.harvest_level["gpu_clocks_locked"] = p.harvest_floor_frac
    out, _ = T.decide(_snap(tok_s=5.0, gpu_util_pct=100.0, t2_overflow_mb=6687.0,
                            power_draw_w=49.8, power_limit_w=300.0,
                            sm_clock_mhz=2685.0, sm_clock_max_mhz=2700.0), st, p)
    assert all(d.action != "harvest" for d in out)
    assert any("floor" in d.reason for d in out)


def test_a_regression_restores_the_lever_and_freezes_it():
    p = T.TunerPolicy()
    st = _warm(T.TunerState(), p)
    out, st = T.decide(_snap(tok_s=5.0, gpu_util_pct=100.0, t2_overflow_mb=6687.0,
                             power_draw_w=49.8, power_limit_w=300.0,
                             sm_clock_mhz=2685.0, sm_clock_max_mhz=2700.0), st, p)
    assert any(d.action == "harvest" for d in out)
    for _ in range(p.settle_cycles):          # wait out the settle window
        _, st = T.decide(_snap(tok_s=5.0, gpu_util_pct=100.0, t2_overflow_mb=6687.0,
                               power_draw_w=49.8, power_limit_w=300.0,
                               sm_clock_mhz=2560.0, sm_clock_max_mhz=2700.0), st, p)
    out, st = T.decide(_snap(tok_s=3.0, gpu_util_pct=100.0, t2_overflow_mb=6687.0,
                             power_draw_w=49.8, power_limit_w=300.0,
                             sm_clock_mhz=2560.0, sm_clock_max_mhz=2700.0), st, p)
    assert any(d.action == "restore" for d in out)
    assert st.frozen.get("gpu_clocks_locked") == p.freeze_cycles


def test_a_frozen_lever_thaws_and_is_re_probed():
    p = T.TunerPolicy(freeze_cycles=2)
    st = _warm(T.TunerState(), p)
    st.frozen["gpu_clocks_locked"] = 2
    snap = _snap(tok_s=5.0, gpu_util_pct=100.0, t2_overflow_mb=6687.0,
                 power_draw_w=49.8, power_limit_w=300.0,
                 sm_clock_mhz=2685.0, sm_clock_max_mhz=2700.0)
    out, st = T.decide(snap, st, p)
    assert all(d.action != "harvest" for d in out)
    assert st.frozen["gpu_clocks_locked"] == 1
    out, st = T.decide(snap, st, p)
    assert "gpu_clocks_locked" not in st.frozen, "never thawed"
    assert any(d.action == "harvest" for d in out), "never re-probed"


def test_becoming_gpu_bound_gives_the_clocks_back():
    p = T.TunerPolicy()
    st = _warm(T.TunerState(), p)
    _, st = T.decide(_snap(tok_s=5.0, gpu_util_pct=100.0, t2_overflow_mb=6687.0,
                           power_draw_w=49.8, power_limit_w=300.0,
                           sm_clock_mhz=2685.0, sm_clock_max_mhz=2700.0), st, p)
    st.settle_remaining = 0
    out, _ = T.decide(_snap(tok_s=9.0, gpu_util_pct=97.0, t2_overflow_mb=0.0,
                            power_draw_w=260.0, power_limit_w=300.0,
                            sm_clock_mhz=2560.0, sm_clock_max_mhz=2700.0), st, p)
    assert any(d.action == "restore" for d in out)


# ── speculative depth ──────────────────────────────────────────────────────

def test_poor_acceptance_lowers_the_draft_depth():
    p = T.TunerPolicy()
    st = _warm(T.TunerState(), p)
    st.settle_remaining = 0
    out, _ = T.decide(_snap(tok_s=5.0, gpu_util_pct=95.0, draft_depth=4,
                            draft_accept_pct=20.0), st, p)
    d = [x for x in out if x.lever == "mtp_draft_n"]
    assert d and d[0].value == 3


def test_high_acceptance_on_a_bandwidth_bound_pass_goes_deeper():
    p = T.TunerPolicy()
    st = _warm(T.TunerState(), p)
    st.settle_remaining = 0
    out, _ = T.decide(_snap(tok_s=5.0, gpu_util_pct=30.0, t2_overflow_mb=6687.0,
                            power_draw_w=60.0, power_limit_w=160.0,
                            draft_depth=4, draft_accept_pct=90.0), st, p)
    d = [x for x in out if x.lever == "mtp_draft_n"]
    assert d and d[0].value == 5


def test_depth_zero_is_left_alone():
    """The engine reports acceptance 1.0 when nothing was drafted , reading
    that as 'drafting is working perfectly' is the confusion gb_bench_spec.py
    already had to guard against."""
    p = T.TunerPolicy()
    st = _warm(T.TunerState(), p)
    st.settle_remaining = 0
    out, _ = T.decide(_snap(tok_s=5.0, gpu_util_pct=95.0, draft_depth=0,
                            draft_accept_pct=100.0), st, p)
    assert all(d.lever != "mtp_draft_n" for d in out)


def test_missing_acceptance_decides_nothing_about_depth():
    p = T.TunerPolicy()
    st = _warm(T.TunerState(), p)
    st.settle_remaining = 0
    out, _ = T.decide(_snap(tok_s=5.0, gpu_util_pct=95.0, draft_depth=4), st, p)
    assert all(d.lever != "mtp_draft_n" for d in out)

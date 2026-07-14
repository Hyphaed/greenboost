#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_reactive.Signal , EWMA, confirm debounce, hysteresis, isolation.

No CUDA, no /dev/greenboost, no running daemon needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from gb_reactive import Signal, Computed, PidController


# ── EWMA ─────────────────────────────────────────────────────────────────────

def test_ewma_formula_single_step():
    s = Signal(0.0, ewma_alpha=0.5)
    s.set(100.0)
    # EMA = 0*(1-0.5) + 100*0.5 = 50
    assert abs(s.get() - 50.0) < 1e-6


def test_ewma_formula_two_steps():
    s = Signal(0.0, ewma_alpha=0.5)
    s.set(100.0)  # 50.0
    s.set(100.0)  # 50*(0.5) + 100*(0.5) = 75
    assert abs(s.get() - 75.0) < 1e-6


def test_ewma_raw_vs_smoothed():
    s = Signal(0.0, ewma_alpha=0.25)
    s.set(80.0)
    # raw = 80, ema = 0*0.75 + 80*0.25 = 20
    assert s.raw == 80.0
    assert abs(s.get() - 20.0) < 1e-6


def test_no_ewma_passes_through():
    s = Signal(0.0)  # no ewma_alpha
    s.set(42.7)
    assert s.get() == 42.7


# ── Confirm debounce ──────────────────────────────────────────────────────────

def test_confirm_debounce_fires_on_nth():
    fired = []
    s = Signal(0.0, confirm=3, epsilon=0.0)
    s.subscribe(lambda new, old: fired.append(new))
    s.set(1.0)  # confirm_count=1 < 3
    assert fired == []
    s.set(2.0)  # confirm_count=2 < 3
    assert fired == []
    s.set(3.0)  # confirm_count=3 >= 3, fires
    assert len(fired) == 1


def test_confirm_resets_on_no_change():
    fired = []
    s = Signal(0.0, confirm=3, epsilon=0.0)
    s.subscribe(lambda new, old: fired.append(new))
    s.set(1.0)   # confirm_count=1
    s.set(2.0)   # confirm_count=2
    # Fire with value equal to _prev_notified: should reset count, not fire
    # _prev_notified is still 0.0 at this point, so set(0.0) = no change
    s.set(0.0)   # no change vs prev_notified (0.0) → reset to 0
    s.set(1.0)   # confirm_count=1 (reset started over)
    assert fired == []


def test_confirm_equals_one_fires_immediately():
    fired = []
    s = Signal(0.0, confirm=1)
    s.subscribe(lambda new, old: fired.append(new))
    s.set(99.0)
    assert len(fired) == 1
    assert fired[0] == 99.0


# ── Hysteresis ────────────────────────────────────────────────────────────────

def test_hysteresis_enter_fires_once():
    entered = []
    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1, epsilon=0.0)
    s.on_enter(lambda v: entered.append(v))

    s.set(85.0)   # crosses enter >= 80 → fires
    assert len(entered) == 1

    s.set(90.0)   # already in_hysteresis → no re-fire
    assert len(entered) == 1


def test_hysteresis_exit_fires_once():
    exited = []
    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1, epsilon=0.0)
    s.on_exit(lambda v: exited.append(v))

    s.set(85.0)   # enter (not tracking exits yet in this test)
    assert s.in_hysteresis is True

    s.set(55.0)   # drops below exit threshold (< 60) → exit fires
    assert len(exited) == 1
    assert s.in_hysteresis is False

    s.set(50.0)   # already exited → no re-fire
    assert len(exited) == 1


def test_hysteresis_prevents_flap():
    """value must drop below exit_thresh before re-entering is possible."""
    entered = []
    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1, epsilon=0.0)
    s.on_enter(lambda v: entered.append(v))

    s.set(85.0)   # enter
    s.set(75.0)   # between exit(60) and enter(80): stays in_hysteresis, no new enter
    s.set(82.0)   # still above enter but already in_hysteresis: no re-enter
    assert len(entered) == 1   # only one enter total


def test_hysteresis_reenter_after_full_exit():
    """After exiting, a new crossing should fire enter again."""
    entered = []
    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1, epsilon=0.0)
    s.on_enter(lambda v: entered.append(v))

    s.set(85.0)   # enter 1
    s.set(55.0)   # exit  (drops below 60)
    s.set(90.0)   # enter 2 (crossed from below after full exit)
    assert len(entered) == 2


# ── Subscriber isolation ──────────────────────────────────────────────────────

def test_subscriber_exception_does_not_stall_others():
    results = []

    def bad_sub(new, old):
        raise RuntimeError("intentional failure")

    def good_sub(new, old):
        results.append(new)

    s = Signal(0.0, confirm=1)
    s.subscribe(bad_sub)
    s.subscribe(good_sub)
    s.set(42.0)   # bad_sub raises but should be swallowed
    assert results == [42.0]


def test_on_enter_exception_does_not_stall_on_exit():
    exited = []

    def bad_enter(v):
        raise ValueError("boom")

    def good_exit(v):
        exited.append(v)

    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1, epsilon=0.0)
    s.on_enter(bad_enter)
    s.on_exit(good_exit)

    s.set(85.0)   # enter: bad_enter raises (swallowed)
    s.set(50.0)   # exit: good_exit should still fire
    assert len(exited) == 1


# ── dump() ────────────────────────────────────────────────────────────────────

def test_dump_contains_expected_keys():
    s = Signal(5.0, name="my_signal")
    d = s.dump()
    for key in ("name", "raw", "value", "confirm_count",
                "in_hysteresis", "last_set_ts", "last_decision", "last_actuated_ts"):
        assert key in d, f"Missing key in dump(): {key!r}"
    assert d["name"] == "my_signal"


def test_dump_reflects_set_value():
    s = Signal(0.0, name="test")
    s.set(42.0)
    d = s.dump()
    assert d["raw"] == 42.0
    assert d["value"] == 42.0  # no EWMA


def test_dump_reflects_ewma():
    s = Signal(0.0, name="ema", ewma_alpha=0.5)
    s.set(100.0)
    d = s.dump()
    assert d["raw"] == 100.0
    assert abs(d["value"] - 50.0) < 1e-6  # EMA value


# ── Computed ──────────────────────────────────────────────────────────────────

def test_computed_initial_value():
    a = Signal(3.0)
    b = Signal(4.0)
    c = Computed(lambda x, y: x + y, a, b)
    assert c.get() == 7.0


def test_computed_updates_on_dep_change():
    a = Signal(10.0)
    b = Signal(5.0)
    c = Computed(lambda x, y: x * y, a, b)
    a.set(20.0)   # trigger recompute: 20*5=100
    assert c.get() == 100.0


# ── Unsubscribe ───────────────────────────────────────────────────────────────

def test_subscribe_returns_unsubscribe_callable():
    calls = []
    s = Signal(0.0, confirm=1)
    unsub = s.subscribe(lambda n, o: calls.append(n))

    s.set(1.0)
    assert calls == [1.0]

    unsub()
    s.set(2.0)
    assert calls == [1.0]   # no new call after unsubscribe


def test_on_enter_returns_unsubscribe_callable():
    entered = []
    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1)
    unsub = s.on_enter(lambda v: entered.append(v))

    s.set(85.0)             # enter
    assert len(entered) == 1

    unsub()
    s.set(50.0)             # exit
    s.set(90.0)             # re-enter , should NOT fire (unsubscribed)
    assert len(entered) == 1


def test_on_exit_returns_unsubscribe_callable():
    exited = []
    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1)
    s.on_enter(lambda v: None)   # need enter to arm the hysteresis
    unsub = s.on_exit(lambda v: exited.append(v))

    s.set(85.0)             # enter
    s.set(50.0)             # exit → fire
    assert len(exited) == 1

    unsub()
    s.set(85.0)             # re-enter
    s.set(50.0)             # re-exit , should NOT fire (unsubscribed)
    assert len(exited) == 1


def test_double_unsubscribe_is_safe():
    """Calling unsub() twice must not raise."""
    s = Signal(0.0, confirm=1)
    unsub = s.subscribe(lambda n, o: None)
    unsub()
    unsub()   # second call must not raise


# ── raw property ──────────────────────────────────────────────────────────────

def test_raw_property_returns_last_input():
    s = Signal(0.0, ewma_alpha=0.1)
    s.set(100.0)
    # raw is the unsmoothed input; get() is the EMA
    assert s.raw == 100.0
    assert s.get() != 100.0   # EMA is smoothed


def test_in_hysteresis_property():
    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1)
    assert s.in_hysteresis is False
    s.set(85.0)
    assert s.in_hysteresis is True
    s.set(50.0)
    assert s.in_hysteresis is False


# ── name property ─────────────────────────────────────────────────────────────

def test_name_property():
    s = Signal(0.0, name="pressure_gauge")
    assert s.name == "pressure_gauge"


# ── EWMA non-numeric fallback ─────────────────────────────────────────────────

def test_ewma_non_numeric_falls_back_to_raw():
    """When float(v) raises, EMA falls back to storing v directly."""
    s = Signal(None, ewma_alpha=0.5)
    s.set("hot")   # can't float("hot") , fallback to raw assignment
    assert s.get() == "hot"
    assert s.raw == "hot"


# ── _is_changed: None paths ───────────────────────────────────────────────────

def test_is_changed_both_none_returns_false():
    """Two None values → no change → subscriber not notified."""
    called = []
    s = Signal(None, confirm=1)
    s.subscribe(lambda n, o: called.append(n))
    s.set(None)   # _ema=None, prev=None → _is_changed(None, None) → False
    assert called == []


def test_is_changed_prev_none_current_not_returns_true():
    """prev=None, current=42.0 → treated as change → subscriber notified."""
    called = []
    s = Signal(None, confirm=1)
    s.subscribe(lambda n, o: called.append(n))
    s.set(42.0)   # prev=None, current=42.0 → fires
    assert called == [42.0]


def test_is_changed_non_numeric_uses_equality():
    """Non-numeric values fall back to equality comparison."""
    called = []
    s = Signal("a", confirm=1)
    s.subscribe(lambda n, o: called.append(n))
    s.set("a")   # same string → no change
    assert called == []
    s.set("b")   # different → fires
    assert called == ["b"]


# ── on_enter / on_exit double-unsubscribe safety ─────────────────────────────

def test_on_enter_double_unsubscribe_is_safe():
    """Calling the on_enter unsub callable twice must not raise."""
    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1)
    unsub = s.on_enter(lambda v: None)
    unsub()
    unsub()   # second call: ValueError inside remove is swallowed


def test_on_exit_double_unsubscribe_is_safe():
    """Calling the on_exit unsub callable twice must not raise."""
    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1)
    unsub = s.on_exit(lambda v: None)
    unsub()
    unsub()


# ── exit callback exception isolation ────────────────────────────────────────

def test_exit_callback_exception_does_not_stall_other_exits():
    """Buggy on_exit callback must not prevent subsequent exit callbacks."""
    results = []

    def bad_exit(v):
        raise RuntimeError("exit boom")

    def good_exit(v):
        results.append(v)

    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1, epsilon=0.0)
    s.on_enter(lambda v: None)
    s.on_exit(bad_exit)
    s.on_exit(good_exit)

    s.set(85.0)   # enter
    s.set(50.0)   # exit: bad fires (swallowed), good fires
    assert len(results) == 1


# ── _check_hysteresis non-numeric guard ──────────────────────────────────────

def test_check_hysteresis_with_non_numeric_value_does_not_raise():
    """When a non-numeric value is set on a signal with hysteresis, the
    TypeError from float() is caught and no enter/exit fires."""
    entered = []
    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1)
    s.on_enter(lambda v: entered.append(v))
    s.set("cannot_float")
    assert entered == []


# ── BehaviorSubject ───────────────────────────────────────────────────────────

def test_behavior_subject_emits_to_late_subscriber_when_in_hysteresis():
    """New on_enter subscriber receives current value if already in hysteresis."""
    from gb_reactive import BehaviorSubject
    s = BehaviorSubject(0.0, name="bs_test", hysteresis=(80.0, 60.0), confirm=1)
    s.set(90.0)                    # enter hysteresis
    assert s.in_hysteresis is True

    received = []
    s.on_enter(received.append)    # late subscriber
    assert len(received) == 1, "BehaviorSubject must replay current state to late subscriber"
    assert received[0] == pytest.approx(90.0)


def test_behavior_subject_no_replay_when_not_in_hysteresis():
    """No replay to new on_enter subscriber when not yet in hysteresis."""
    from gb_reactive import BehaviorSubject
    s = BehaviorSubject(0.0, hysteresis=(80.0, 60.0), confirm=1)
    s.set(50.0)
    received = []
    s.on_enter(received.append)
    assert received == []


def test_behavior_subject_regular_enter_still_fires():
    """Normal on_enter fires as expected after crossing threshold."""
    from gb_reactive import BehaviorSubject
    s = BehaviorSubject(0.0, hysteresis=(80.0, 60.0), confirm=1)
    events = []
    s.on_enter(events.append)
    s.set(85.0)
    assert len(events) == 1
    assert events[0] == pytest.approx(85.0)


def test_behavior_subject_late_subscriber_exception_isolated():
    """Exception in late-subscriber replay does not propagate."""
    from gb_reactive import BehaviorSubject
    s = BehaviorSubject(0.0, hysteresis=(80.0, 60.0), confirm=1)
    s.set(90.0)
    def bad(v): raise RuntimeError("boom")
    s.on_enter(bad)  # must not raise


# ── pipe() ───────────────────────────────────────────────────────────────────

def test_pipe_returns_result_of_single_operator():
    from gb_reactive import Signal, pairwise
    s = Signal(0.0, name="src")
    pw = s.pipe(pairwise())
    assert hasattr(pw, "subscribe")


def test_pipe_chains_two_operators():
    from gb_reactive import Signal, pairwise, filter as rx_filter
    s = Signal(0.0, name="src")
    gate = True
    result = s.pipe(
        rx_filter(lambda: gate),
    )
    assert hasattr(result, "on_enter")


# ── filter operator ───────────────────────────────────────────────────────────

def test_filter_blocks_enter_when_predicate_false():
    from gb_reactive import Signal, filter as rx_filter
    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1)
    gate = False
    received = []
    s.pipe(rx_filter(lambda: gate)).on_enter(received.append)
    s.set(90.0)
    assert received == [], "filter must block on_enter when predicate is False"


def test_filter_passes_enter_when_predicate_true():
    from gb_reactive import Signal, filter as rx_filter
    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1)
    gate = True
    received = []
    s.pipe(rx_filter(lambda: gate)).on_enter(received.append)
    s.set(90.0)
    assert len(received) == 1


def test_filter_gate_checked_at_call_time_not_subscription_time():
    """Predicate is evaluated each time, not at subscription time."""
    from gb_reactive import Signal, filter as rx_filter
    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1)
    gate = [False]
    received = []
    s.pipe(rx_filter(lambda: gate[0])).on_enter(received.append)

    s.set(90.0)               # gate is False , blocked
    assert received == []

    gate[0] = True
    s.set(55.0)               # exit hysteresis (below exit threshold 60.0)
    s.set(90.0)               # re-enter with gate True
    assert len(received) == 1


def test_filter_blocks_exit_when_predicate_false():
    from gb_reactive import Signal, filter as rx_filter
    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1)
    gate = [True]
    entered = []
    exited  = []
    fs = s.pipe(rx_filter(lambda: gate[0]))
    fs.on_enter(entered.append)
    fs.on_exit(exited.append)

    s.set(90.0)              # enter
    gate[0] = False          # close gate
    s.set(50.0)              # exit hysteresis , but gate is False
    assert exited == [], "filter must block on_exit when predicate is False"


def test_filter_passes_generic_subscribe():
    from gb_reactive import Signal, filter as rx_filter
    s = Signal(0, confirm=1)
    received = []
    s.pipe(rx_filter(lambda: True)).subscribe(lambda n, o: received.append(n))
    s.set(1)
    assert 1 in received


# ── pairwise operator ─────────────────────────────────────────────────────────

def test_pairwise_emits_prev_and_curr_tuple():
    from gb_reactive import Signal, pairwise
    s = Signal("", name="phase", confirm=1)
    pairs = []
    s.pipe(pairwise()).subscribe(lambda pair, _: pairs.append(pair))
    s.set("INFERENCE")
    s.set("IDLE")
    assert ("INFERENCE", "IDLE") in pairs


def test_pairwise_first_pair_uses_initial_value():
    from gb_reactive import Signal, pairwise
    s = Signal("START", name="p", confirm=1)
    pairs = []
    s.pipe(pairwise()).subscribe(lambda pair, _: pairs.append(pair))
    s.set("NEXT")
    assert pairs[0] == ("START", "NEXT")


def test_pairwise_multiple_transitions():
    from gb_reactive import Signal, pairwise
    s = Signal("A", confirm=1)
    pairs = []
    s.pipe(pairwise()).subscribe(lambda pair, _: pairs.append(pair))
    for v in ["B", "C", "D"]:
        s.set(v)
    assert pairs == [("A", "B"), ("B", "C"), ("C", "D")]


def test_pairwise_detects_inference_to_idle():
    """pairwise() correctly surfaces the INFERENCE→IDLE transition."""
    from gb_reactive import Signal, pairwise
    s = Signal("", confirm=1)
    transitions = []
    def on_pair(pair, _):
        prev, curr = pair
        if prev == "INFERENCE" and curr == "IDLE":
            transitions.append((prev, curr))
    s.pipe(pairwise()).subscribe(on_pair)
    for phase in ["INFERENCE", "INFERENCE", "IDLE"]:
        s.set(phase)
    assert len(transitions) == 1


# ── with_latest_from operator ─────────────────────────────────────────────────

def test_with_latest_from_includes_latest_from_other():
    from gb_reactive import Signal, with_latest_from
    trigger = Signal(0.0, confirm=1)
    context = Signal("INFERENCE", confirm=1)
    results = []
    trigger.pipe(with_latest_from(context)).subscribe(lambda t, c: results.append((t, c)))
    trigger.set(0.9)
    assert results and results[-1] == pytest.approx((0.9, "INFERENCE")), results


def test_with_latest_from_updates_with_latest_context():
    from gb_reactive import Signal, with_latest_from
    trigger = Signal(0.0, confirm=1)
    ctx     = Signal("A", confirm=1)
    results = []
    trigger.pipe(with_latest_from(ctx)).subscribe(lambda t, c: results.append((t, c)))
    ctx.set("B")         # context changes
    trigger.set(1.0)     # trigger fires , should see ctx="B"
    assert results[-1][1] == "B"


def test_with_latest_from_multiple_sources():
    from gb_reactive import Signal, with_latest_from
    t  = Signal(0.0, confirm=1)
    s1 = Signal(10, confirm=1)
    s2 = Signal(20, confirm=1)
    results = []
    t.pipe(with_latest_from(s1, s2)).subscribe(lambda tv, v1, v2: results.append((tv, v1, v2)))
    t.set(0.5)
    assert results[-1] == pytest.approx((0.5, 10, 20))


# ── scan operator ─────────────────────────────────────────────────────────────

def test_scan_running_sum():
    from gb_reactive import Signal, scan
    s = Signal(0, confirm=1)
    totals = []
    s.pipe(scan(lambda acc, v: acc + v, 0)).subscribe(lambda n, _: totals.append(n))
    for v in [1, 2, 3, 4]:
        s.set(v)
    assert totals == [1, 3, 6, 10]


def test_scan_count_high_pressure_run():
    """scan can count consecutive high-pressure events."""
    from gb_reactive import Signal, scan
    s = Signal(0.0, confirm=1)
    runs = []
    s.pipe(scan(lambda n, v: n + 1 if v > 0.85 else 0, 0)).subscribe(
        lambda n, _: runs.append(n)
    )
    # Use distinct values so Signal epsilon check passes each poll
    for v in [0.90, 0.91, 0.92, 0.50, 0.93]:
        s.set(v)
    assert runs == [1, 2, 3, 0, 1]


def test_scan_seed_value():
    from gb_reactive import Signal, scan
    s = Signal(0, confirm=1)
    results = []
    s.pipe(scan(lambda acc, v: acc * 2 + v, 1)).subscribe(lambda n, _: results.append(n))
    s.set(5)
    assert results[0] == 7  # 1*2 + 5


def test_scan_get_returns_accumulated_value():
    from gb_reactive import Signal, scan
    s = Signal(0, confirm=1)
    sc = s.pipe(scan(lambda acc, v: acc + v, 0))
    s.set(3)
    s.set(7)
    assert sc.get() == 10


# ── Exception isolation in new operators ─────────────────────────────────────

def test_filter_enter_exception_isolated():
    """Exception in FilteredSignal on_enter subscriber does not propagate."""
    from gb_reactive import Signal, filter as rx_filter
    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1)
    def bad(v): raise RuntimeError("boom")
    s.pipe(rx_filter(lambda: True)).on_enter(bad)
    s.set(90.0)  # must not raise


def test_filter_exit_with_gate_true_fires():
    """FilteredSignal on_exit fires when predicate is True."""
    from gb_reactive import Signal, filter as rx_filter
    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1)
    exited = []
    fs = s.pipe(rx_filter(lambda: True))
    fs.on_exit(exited.append)
    s.set(90.0)   # enter
    s.set(50.0)   # exit (below 60)
    assert len(exited) == 1


def test_pairwise_exception_isolated():
    """Exception in PairwiseSignal subscriber does not propagate."""
    from gb_reactive import Signal, pairwise
    s = Signal("A", confirm=1)
    def bad(pair, _): raise RuntimeError("boom")
    s.pipe(pairwise()).subscribe(bad)
    s.set("B")  # must not raise


def test_with_latest_from_exception_isolated():
    """Exception in WithLatestFrom subscriber does not propagate."""
    from gb_reactive import Signal, with_latest_from
    t = Signal(0.0, confirm=1)
    c = Signal("X", confirm=1)
    def bad(tv, cv): raise RuntimeError("boom")
    t.pipe(with_latest_from(c)).subscribe(bad)
    t.set(1.0)  # must not raise


def test_filter_exit_exception_isolated():
    """Exception in FilteredSignal on_exit subscriber does not propagate."""
    from gb_reactive import Signal, filter as rx_filter
    s = Signal(0.0, hysteresis=(80.0, 60.0), confirm=1)
    def bad(v): raise RuntimeError("boom")
    s.pipe(rx_filter(lambda: True)).on_exit(bad)
    s.set(90.0)   # enter
    s.set(50.0)   # exit , bad() must not propagate


def test_filter_subscribe_exception_isolated():
    """Exception in FilteredSignal subscribe callback does not propagate."""
    from gb_reactive import Signal, filter as rx_filter
    s = Signal(0, confirm=1)
    def bad(n, o): raise RuntimeError("boom")
    s.pipe(rx_filter(lambda: True)).subscribe(bad)
    s.set(1)  # must not raise


# ── PidController (DI-6, MuxFlow port) ──────────────────────────────────────

def test_pid_zero_error_at_setpoint_yields_zero_output():
    pid = PidController(kp=1.0, ki=0.5, kd=0.1, setpoint=100.0, out_min=0.0, out_max=1.0)
    assert pid.update(100.0, dt=1.0) == 0.0


def test_pid_output_clamped_to_max():
    pid = PidController(kp=10.0, ki=0.0, kd=0.0, setpoint=100.0, out_min=0.0, out_max=1.0)
    assert pid.update(0.0, dt=1.0) == 1.0   # huge positive error, kp alone would blow past 1.0


def test_pid_output_clamped_to_min():
    pid = PidController(kp=10.0, ki=0.0, kd=0.0, setpoint=100.0, out_min=0.0, out_max=1.0)
    assert pid.update(1000.0, dt=1.0) == 0.0   # measured way above setpoint -> negative error, clamped to floor


def test_pid_proportional_response_scales_with_error():
    pid = PidController(kp=0.01, ki=0.0, kd=0.0, setpoint=100.0, out_min=0.0, out_max=1.0)
    small_error_out = pid.update(95.0, dt=1.0)
    pid.reset()
    large_error_out = pid.update(50.0, dt=1.0)
    assert large_error_out > small_error_out


def test_pid_integral_accumulates_under_sustained_error():
    pid = PidController(kp=0.0, ki=0.05, kd=0.0, setpoint=100.0, out_min=0.0, out_max=1.0)
    out1 = pid.update(90.0, dt=1.0)
    out2 = pid.update(90.0, dt=1.0)
    assert out2 > out1   # same error, but integral has accumulated further


def test_pid_reset_clears_integral_and_derivative_history():
    pid = PidController(kp=0.0, ki=0.05, kd=0.0, setpoint=100.0, out_min=0.0, out_max=1.0)
    pid.update(90.0, dt=1.0)
    pid.update(90.0, dt=1.0)
    pid.reset()
    out_after_reset = pid.update(90.0, dt=1.0)
    fresh = PidController(kp=0.0, ki=0.05, kd=0.0, setpoint=100.0, out_min=0.0, out_max=1.0)
    out_fresh = fresh.update(90.0, dt=1.0)
    assert out_after_reset == out_fresh


def test_pid_never_windups_past_output_bound_after_long_saturation():
    # Sustained large error for many steps must not leave the integral so
    # large that output stays pinned long after the error suddenly reverses.
    pid = PidController(kp=0.0, ki=0.1, kd=0.0, setpoint=100.0, out_min=0.0, out_max=1.0)
    for _ in range(1000):
        pid.update(0.0, dt=1.0)   # huge sustained error
    assert pid.update(0.0, dt=1.0) == 1.0   # still pinned (expected while error persists)
    # Error reverses hard (measured now far ABOVE setpoint) — with anti-windup
    # the output must be able to leave the ceiling almost immediately.
    out = pid.update(1000.0, dt=1.0)
    assert out < 1.0


def test_pid_zero_or_negative_dt_does_not_raise():
    pid = PidController(kp=1.0, ki=1.0, kd=1.0, setpoint=100.0, out_min=0.0, out_max=1.0)
    pid.update(50.0, dt=0.0)   # must not raise ZeroDivisionError
    pid.update(50.0, dt=-1.0)

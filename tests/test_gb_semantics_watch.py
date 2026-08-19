"""Governed segments must FIRE on change, not wait to be asked.

`gb_semantics.evaluate_segment()` is a pull API. Nothing polls it, so a matched
segment can sit unreported indefinitely — which is what happened on 2026-08-18:
`rule1_underfilled` was matched while CLAUDE.md's Rule #1 requires a tripwire to
fire on exactly that condition, and `weights_dont_fit_vram` had been true for a
whole session before anything surfaced it.

These tests pin the reactive contract: transitions only, unknown != clear, and
a stable condition costs one event rather than one per tick.
"""
from __future__ import annotations

import pytest

import gb_semantics_watch as gsw


class _FakeRegistry(dict):
    pass


@pytest.fixture
def watcher(monkeypatch):
    """A watcher over two synthetic segments with a scriptable verdict."""
    verdicts: dict = {"a": {"matched": False, "severity": "violation"},
                      "b": {"matched": False, "severity": "info"}}

    import gb_semantics
    monkeypatch.setattr(gb_semantics, "load",
                        lambda: {"segments": {"a": object(), "b": object()}})
    monkeypatch.setattr(gb_semantics, "evaluate_segment",
                        lambda name: dict(verdicts[name], segment=name))
    w = gsw.SegmentWatcher(confirm=1, emit=False)
    return w, verdicts


def test_first_tick_is_a_baseline_not_a_transition(watcher):
    """Announcing every segment at startup would be pure noise."""
    w, _ = watcher
    seen = []
    w.on_transition(lambda *a: seen.append(a))
    w.tick()
    assert seen == []


def test_transition_fires_once_and_only_on_change(watcher):
    w, verdicts = watcher
    seen = []
    w.on_transition(lambda n, o, nw, r: seen.append((n, o, nw)))

    w.tick()                                   # baseline: clear
    verdicts["a"]["matched"] = True
    w.tick()                                   # clear -> matched
    assert seen == [("a", "clear", "matched")]

    w.tick(); w.tick()                         # still matched
    assert len(seen) == 1, "a stable condition must not re-emit every tick"

    verdicts["a"]["matched"] = False
    w.tick()                                   # matched -> clear
    assert seen[-1] == ("a", "matched", "clear")


def test_unknown_is_not_clear(watcher):
    """A telemetry failure must never read as a clean bill of health.

    This is the whole reason the verdict is three-valued: collapsing None into
    False is how an unresolvable metric becomes "nothing is wrong".
    """
    w, verdicts = watcher
    seen = []
    w.on_transition(lambda n, o, nw, r: seen.append((n, o, nw)))
    w.tick()
    verdicts["a"] = {"matched": None, "severity": "violation"}
    w.tick()
    assert seen == [("a", "clear", "unknown")]

    verdicts["a"] = {"error": "resolver blew up", "severity": "violation"}
    w.tick()
    assert seen[-1][2] == "unknown", "an errored segment is unknown, not clear"


def test_evaluation_failure_does_not_break_the_tick(watcher, monkeypatch):
    """The watcher rides the snapshot recorder's loop; it must never kill it."""
    w, _ = watcher
    import gb_semantics
    monkeypatch.setattr(gb_semantics, "evaluate_segment",
                        lambda name: (_ for _ in ()).throw(RuntimeError("boom")))
    states = w.tick()
    assert set(states.values()) == {"unknown"}


def test_subscriber_exception_does_not_stop_other_subscribers(watcher):
    w, verdicts = watcher
    ok = []
    w.on_transition(lambda *a: (_ for _ in ()).throw(ValueError("bad sub")))
    w.on_transition(lambda n, o, nw, r: ok.append(n))
    w.tick()
    verdicts["a"]["matched"] = True
    w.tick()
    assert ok == ["a"]


def test_confirm_debounces_a_flapping_segment(monkeypatch):
    """Live telemetry flaps across a band edge; two agreeing reads before
    notifying is the same CONFIRM_POLLS convention the orchestrator uses."""
    verdicts = {"a": {"matched": False, "severity": "violation"}}
    import gb_semantics
    monkeypatch.setattr(gb_semantics, "load", lambda: {"segments": {"a": object()}})
    monkeypatch.setattr(gb_semantics, "evaluate_segment",
                        lambda name: dict(verdicts[name], segment=name))
    w = gsw.SegmentWatcher(confirm=2, emit=False)
    seen = []
    w.on_transition(lambda n, o, nw, r: seen.append((o, nw)))

    w.tick()                        # baseline
    verdicts["a"]["matched"] = True
    w.tick()                        # 1st changed read — not yet confirmed
    assert seen == []
    w.tick()                        # 2nd agreeing read — now it fires
    assert seen == [("clear", "matched")]


def test_matched_lists_worst_severity_first(watcher):
    w, verdicts = watcher
    verdicts["a"] = {"matched": True, "severity": "violation"}
    verdicts["b"] = {"matched": True, "severity": "info"}
    w.tick()
    assert [m["segment"] for m in w.matched()] == ["a", "b"]


def test_snapshot_does_not_re_evaluate(watcher, monkeypatch):
    """snapshot() is for a status line; it must be free to call."""
    w, _ = watcher
    w.tick()
    import gb_semantics
    monkeypatch.setattr(gb_semantics, "evaluate_segment",
                        lambda name: (_ for _ in ()).throw(AssertionError("re-evaluated")))
    snap = w.snapshot()
    assert snap["ticks"] == 1
    assert set(snap["segments"]) == {"a", "b"}


def test_real_registry_is_fully_watched():
    """Every segment in semantics/segments.yaml gets watched — a segment that
    exists but is not watched is invisible in exactly the way this fixes."""
    import gb_semantics
    w = gsw.SegmentWatcher(emit=False)
    assert set(w._watched) == set(gb_semantics.load()["segments"])

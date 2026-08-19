#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_semantics_watch.py , make GB-Semantics segments REACTIVE instead of pull-only.

Why this exists
---------------
`gb_semantics.evaluate_segment()` is a pull API: it computes a verdict when
somebody asks. Nothing asks on its own, so a segment can be matched for hours
and no one finds out until a human runs `gb semantics segments`. That is not a
theoretical gap , it is the reason three findings sat unnoticed on this box on
2026-08-18:

  * `rule1_underfilled` was matched, while CLAUDE.md's Rule #1 explicitly
    requires that "a shim_transition must fire when VRAM is under-filled while
    overflow to T2/T3 is active". The condition held; the tripwire never fired,
    because firing required someone to ask first.
  * `weights_dont_fit_vram` came back true the instant it was written. It had
    been true for the whole session , 15.86 GB of weights on an 11.94 GiB card.
  * `t2_overflow_active_mb` resolved to None on a shim_stats file the shim had
    simply stopped rewriting while idle, so the tripwire above could not have
    fired even if something had asked.

This module closes that by giving every governed segment a `gb_reactive.Signal`
and emitting a dataflux event ON TRANSITION ONLY , the same idiom
gb_orchestrator.py already uses for its own loops (Signal + confirm= hysteresis
+ pipe(pairwise())), rather than inventing a second reactive style beside it.

Design notes
------------
* **No new thread.** `tick()` is called by whatever loop is already running ,
  in practice gb_dataflux's SnapshotRecorder, which polls every 5 s anyway.
  Adding a second poller to watch the first poller's data would be the wrong
  shape.
* **Transitions, not levels.** A subscriber hears `False -> True`, not "still
  true". A segment that stays matched emits once, not every tick. This is what
  keeps an always-true condition (like weights not fitting) from becoming noise
  the moment it is finally surfaced.
* **`confirm=2` by default.** A segment reads live telemetry, and a single
  sample can flap , VRAM crosses a band boundary, a resolver misses one file
  write. Two consecutive agreeing evaluations before notifying, matching the
  CONFIRM_POLLS convention the orchestrator and gb_supervisor already use.
* **`matched: None` is not `False`.** A segment that cannot tell (missing
  telemetry) must never be recorded as a clean bill of health , that is the
  exact failure mode GB-Semantics exists to prevent. Unknown is tracked as its
  own state and transitions in and out of it are reported.
"""
from __future__ import annotations

import threading
import time

DEFAULT_CONFIRM = 2

# Segments whose transitions are worth an event. Everything in segments.yaml is
# watched by default; this only pins the ones that must never be dropped if the
# registry is ever filtered.
ALWAYS_WATCH = ("rule1_underfilled", "weights_on_t3", "below_quality_floor",
                "kmod_missing_silent_degrade", "swap_thrash_not_gpu_throttle")


def _state_of(result: dict) -> str:
    """Three-valued verdict as a string Signal can compare cheaply.

    Deliberately three-valued: collapsing "unknown" into "not matched" is how a
    silent telemetry failure turns into a clean bill of health.
    """
    if result.get("error"):
        return "unknown"
    m = result.get("matched")
    if m is None:
        return "unknown"
    return "matched" if m else "clear"


class SegmentWatcher:
    """Reactive wrapper over the governed segment registry.

    Usage (from a loop that already ticks)::

        w = SegmentWatcher()
        w.tick()          # evaluates every segment, emits on transition

    Subscribe for in-process reactions on top of the dataflux emit::

        w.on_transition(lambda name, old, new, result: ...)
    """

    def __init__(self, confirm: int = DEFAULT_CONFIRM,
                 emit: bool = True, names: "tuple[str, ...] | None" = None) -> None:
        from gb_reactive import Signal, pairwise

        self._lock = threading.Lock()
        self._emit = emit
        self._subs: list = []
        self._signals: dict = {}
        self._last_result: dict = {}
        self._tick_count = 0
        self._last_tick_ts = 0.0

        self._confirm = max(1, confirm)
        self._Signal = Signal
        self._pairwise = pairwise

        import gb_semantics
        registry = gb_semantics.load()["segments"]
        wanted = names or tuple(registry.keys())
        # Names only. The Signal itself is created on the FIRST tick, primed
        # with the state actually observed then.
        #
        # Priming with a sentinel instead would be subtly wrong: Signal's
        # `confirm=N` debounce counts every changed set, so a sentinel->first
        # -real-state move consumes the confirm budget, and the first GENUINE
        # transition after it gets swallowed. Creating the Signal already
        # holding the observed baseline makes the first notification a real
        # change and nothing else.
        self._watched = tuple(n for n in wanted if n in registry)

    # ── subscription ─────────────────────────────────────────────────────────

    def on_transition(self, fn) -> None:
        """Register fn(name, old_state, new_state, result). Never raises out."""
        with self._lock:
            self._subs.append(fn)

    def _make_handler(self, name: str):
        def _handler(pair, _unused) -> None:
            old, new = pair
            # No baseline guard needed: the Signal is created already holding
            # the first observed state (see __init__), so every pair that
            # reaches here is a genuine change.
            result = self._last_result.get(name, {})
            self._dispatch(name, old, new, result)
        return _handler

    def _dispatch(self, name: str, old: str, new: str, result: dict) -> None:
        if self._emit:
            try:
                import gb_dataflux
                sev = (result.get("severity") or "info")
                gb_dataflux.emit({
                    "node": "host", "label": "gb_semantics",
                    "kind": "semantic_transition",
                    "segment": name, "from": old, "to": new,
                    "severity": sev,
                    # A violation becoming true is the case an operator must
                    # actually see; everything else is informational history.
                    "status": ("error" if (new == "matched"
                                           and sev in ("violation", "warning"))
                               else "ok"),
                    "evidence": result.get("evidence", [])[:4],
                })
            except Exception:
                pass   # telemetry must never break the watcher
        for fn in list(self._subs):
            try:
                fn(name, old, new, result)
            except Exception:
                pass

    # ── the tick ─────────────────────────────────────────────────────────────

    def tick(self) -> dict:
        """Re-evaluate every watched segment and push into its Signal.

        Returns {name: state} for the caller's own use. Cheap enough for a 5 s
        loop: every resolver underneath wraps an accessor that is already being
        read by the snapshot recorder on the same tick.
        """
        import gb_semantics
        states = {}
        for name in self._watched:
            try:
                result = gb_semantics.evaluate_segment(name)
            except Exception as e:
                result = {"segment": name, "error": str(e)}
            self._last_result[name] = result
            state = _state_of(result)
            states[name] = state
            sig = self._signals.get(name)
            if sig is None:
                # First observation: prime the Signal with it. pairwise() then
                # starts from this state, so the next change is transition #1.
                sig = self._Signal(state, name=f"segment:{name}",
                                   confirm=self._confirm)
                sig.pipe(self._pairwise()).subscribe(self._make_handler(name))
                self._signals[name] = sig
                continue
            sig.set(state)
        with self._lock:
            self._tick_count += 1
            self._last_tick_ts = time.time()
        return states

    # ── introspection ────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Current state of every watched segment, without re-evaluating."""
        return {
            "ticks": self._tick_count,
            "last_tick_ts": self._last_tick_ts,
            "segments": {n: {"state": s.get(),
                             "severity": (self._last_result.get(n, {})
                                          .get("severity", "info"))}
                         for n, s in self._signals.items()},
        }

    def matched(self) -> list:
        """Segments currently matched, worst-severity first , the short answer
        to "is anything wrong right now" without a full re-evaluation."""
        order = {"violation": 0, "warning": 1, "diagnosis": 2, "info": 3, "ok": 4}
        out = [(n, self._last_result.get(n, {}).get("severity", "info"))
               for n, s in self._signals.items() if s.get() == "matched"]
        out.sort(key=lambda t: order.get(t[1], 9))
        return [{"segment": n, "severity": sev} for n, sev in out]


# ── process-wide singleton (opt-in, mirrors gb_init's singleton style) ───────

_WATCHER: "SegmentWatcher | None" = None
_WATCHER_LOCK = threading.Lock()


def watcher(**kw) -> SegmentWatcher:
    """Shared SegmentWatcher for this process. Constructed on first use."""
    global _WATCHER
    with _WATCHER_LOCK:
        if _WATCHER is None:
            _WATCHER = SegmentWatcher(**kw)
        return _WATCHER


def tick() -> dict:
    """Convenience for a caller that just wants the shared watcher to advance."""
    return watcher().tick()

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_reactive.py , reactive-state primitives for GreenBoost.

Inspired by BehaviorSubject / operator composition patterns from
rxRust (~/Dev/rxRust), RxCpp (~/Dev/RxCpp), and RxGo (~/Dev/RxGo).
All operators are pure Python, stdlib only, thread-safe.

Classes
-------
Signal          , EWMA-smoothed observable with confirm debounce, hysteresis,
                  and on_enter/on_exit callbacks.  Also serves as the source
                  for any piped operator chain.

BehaviorSubject , Signal that immediately calls on_enter subscribers when the
                  signal is already in hysteresis at subscription time (the core
                  BehaviorSubject contract from rxRust/RxCpp).

FilteredSignal  , Derived observable: only propagates on_enter/on_exit from its
                  source when a runtime predicate is True.  Used for declarative
                  gate composition (replaces inline if-chains).

PairwiseSignal  , Derived observable: emits (previous, current) pairs on each
                  source notification (pairwise operator from rxRust).  Used for
                  transition detection (e.g. INFERENCE→IDLE phase gate).

WithLatestFrom  , Derived observable: when any of N sources fires, calls
                  subscriber with (trigger_val, *latest_from_each_other)
                  (with_latest_from operator from rxRust).

ScanSignal      , Derived signal: applies an accumulator function on each
                  source emission and emits the running result (scan operator).
                  EWMA inside Signal is a special case; ScanSignal generalises
                  to any fold function.

Computed        , Legacy derived Signal: recomputes fn(*deps) when any dep
                  notifies (kept for backward compat).

Operator factories (chainable via Signal.pipe)
----------------------------------------------
filter(predicate)                       → FilteredSignal
pairwise()                              → PairwiseSignal
with_latest_from(*others)               → WithLatestFrom
scan(fn, seed)                          → ScanSignal

Usage
-----
# Phase-transition detection (Loop J):
phase_sig.pipe(pairwise()).subscribe(on_transition)

# Declarative KV-grow gate (Loop C):
kv_sig.pipe(filter(lambda: not ecc and not throttled)).on_enter(on_kv_high)

# Accumulate free-MB across polls:
free_mb_sig.pipe(scan(lambda acc, v: acc + v, 0)).subscribe(on_total)

Design mirrors the g_t2_warn_adj controller discipline already in the shim:
  confirm   ←→ CONFIRM_POLLS=3 debounce      (gb_supervisor.py:78/640)
  hysteresis ←→ _prev_state flap-guard       (gb_supervisor.py:530)
  ewma_alpha ←→ EWMA α=1/8 controllers       (greenboost_cuda_shim.c:3133)

Importable by both gb_supervisor (root) and gb_init (per-process).
Thread-safety: one threading.Lock per Signal; subscriber exceptions are
swallowed so one bad subscriber cannot stall the loop.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional, Tuple, TypeVar

T = TypeVar("T")


class Signal:
    """
    Observable value with change-gated notifications.

    Parameters
    ----------
    value       Initial value.
    name        Human-readable label (appears in decisions/dump output).
    epsilon     Float-change threshold: |new-old| <= epsilon → no notification.
    confirm     Consecutive changed sets required before notifying.
                Mirrors CONFIRM_POLLS=3 (gb_supervisor.py:78).
    hysteresis  (enter_threshold, exit_threshold) for scalar values.
                on_enter fires when value crosses enter_threshold from below.
                on_exit  fires when value drops below exit_threshold.
                Use enter > exit to prevent flapping.
    ewma_alpha  Exponential moving-average coefficient 0 < alpha <= 1.
                .get() returns EMA; .raw returns last unsmoothed input.
                Mirrors α=1/8 EWMA in greenboost_cuda_shim.c:3133.
    """

    def __init__(
        self,
        value: Any = None,
        *,
        name: str = "",
        epsilon: float = 0.0,
        confirm: int = 1,
        hysteresis: Optional[Tuple[float, float]] = None,
        ewma_alpha: Optional[float] = None,
    ) -> None:
        self._lock       = threading.Lock()
        self._name       = name
        self._epsilon    = epsilon
        self._confirm    = max(1, confirm)
        self._hysteresis = hysteresis
        self._alpha      = ewma_alpha

        self._raw         = value
        self._ema         = float(value) if (ewma_alpha and value is not None) else value
        self._subs:       list[Callable] = []
        self._enter_subs: list[Callable] = []
        self._exit_subs:  list[Callable] = []
        self._confirm_count = 0
        self._prev_notified = value
        self._in_hysteresis = False
        self.last_set_ts: float = 0.0
        self.last_decision: str = ""
        self.last_actuated_ts: float = 0.0

    # ── read API ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    def get(self) -> Any:
        """Current value (EMA-smoothed if ewma_alpha is set)."""
        return self._ema

    @property
    def raw(self) -> Any:
        """Last unsmoothed input value."""
        return self._raw

    @property
    def in_hysteresis(self) -> bool:
        """True when value is above the enter threshold."""
        return self._in_hysteresis

    # ── write API ─────────────────────────────────────────────────────────────

    def set(self, v: Any) -> bool:
        """
        Update the value.  Returns True iff subscribers were notified.
        Applies EWMA smoothing, confirm debounce, and hysteresis crossing.
        """
        with self._lock:
            self._raw        = v
            self.last_set_ts = time.monotonic()

            # EWMA smoothing
            if self._alpha is not None:
                try:
                    prev_ema = self._ema if self._ema is not None else float(v)
                    self._ema = prev_ema * (1.0 - self._alpha) + float(v) * self._alpha
                except (TypeError, ValueError):
                    self._ema = v
            else:
                self._ema = v

            current  = self._ema
            notified = self._prev_notified

            changed = self._is_changed(current, notified)
            if not changed:
                self._confirm_count = 0
                return False

            self._confirm_count += 1
            if self._confirm_count < self._confirm:
                return False

            # Threshold crossed and confirmed , proceed
            self._confirm_count  = 0
            prev_for_subs        = self._prev_notified
            self._prev_notified  = current
            subs                 = list(self._subs)
            enters, exits        = self._check_hysteresis(current, prev_for_subs)

        # Fire callbacks outside lock to avoid deadlock with subscribing callbacks
        notified_any = False
        if subs:
            for cb in subs:
                try:
                    cb(current, prev_for_subs)
                except Exception:
                    pass
            notified_any = True
        for cb in enters:
            try:
                cb(current)
            except Exception:
                pass
            notified_any = True
        for cb in exits:
            try:
                cb(current)
            except Exception:
                pass
            notified_any = True

        return notified_any

    # ── operator pipeline (Rx-style) ─────────────────────────────────────────

    def pipe(self, *operators: Callable) -> Any:
        """
        Chain Rx-style operators, each taking the previous observable and
        returning a new one.  Mirrors rxRust/RxCpp .pipe() / operator| idiom.

        Example::

            sig.pipe(pairwise()).subscribe(on_pair)
            sig.pipe(filter(lambda: gate_ok)).on_enter(on_high)
        """
        result: Any = self
        for op in operators:
            result = op(result)
        return result

    # ── subscription API ──────────────────────────────────────────────────────

    def subscribe(self, fn: Callable) -> Callable:
        """Register fn(new, old) callback.  Returns an unsubscribe callable."""
        with self._lock:
            self._subs.append(fn)

        def _unsub() -> None:
            with self._lock:
                try:
                    self._subs.remove(fn)
                except ValueError:
                    pass
        return _unsub

    def on_enter(self, fn: Callable) -> Callable:
        """fn(value) fired when value crosses enter hysteresis threshold from below."""
        with self._lock:
            self._enter_subs.append(fn)

        def _unsub() -> None:
            with self._lock:
                try:
                    self._enter_subs.remove(fn)
                except ValueError:
                    pass
        return _unsub

    def on_exit(self, fn: Callable) -> Callable:
        """fn(value) fired when value drops below exit hysteresis threshold."""
        with self._lock:
            self._exit_subs.append(fn)

        def _unsub() -> None:
            with self._lock:
                try:
                    self._exit_subs.remove(fn)
                except ValueError:
                    pass
        return _unsub

    # ── debug ─────────────────────────────────────────────────────────────────

    def dump(self) -> dict:
        """Snapshot suitable for debug_dump() / vitals --json decisions block."""
        return {
            "name":             self._name,
            "raw":              self._raw,
            "value":            self._ema,
            "confirm_count":    self._confirm_count,
            "in_hysteresis":    self._in_hysteresis,
            "last_set_ts":      self.last_set_ts,
            "last_decision":    self.last_decision,
            "last_actuated_ts": self.last_actuated_ts,
        }

    # ── internals ─────────────────────────────────────────────────────────────

    def _is_changed(self, current: Any, previous: Any) -> bool:
        if previous is None and current is None:
            return False
        if previous is None or current is None:
            return True
        try:
            return abs(float(current) - float(previous)) > self._epsilon
        except (TypeError, ValueError):
            return current != previous

    def _check_hysteresis(
        self, current: Any, previous: Any
    ) -> "tuple[list[Callable], list[Callable]]":
        if not self._hysteresis:
            return [], []
        enter_thresh, exit_thresh = self._hysteresis
        enters: list[Callable] = []
        exits:  list[Callable] = []
        try:
            c = float(current)
            if not self._in_hysteresis and c >= enter_thresh:
                self._in_hysteresis = True
                enters = list(self._enter_subs)
            elif self._in_hysteresis and c < exit_thresh:
                self._in_hysteresis = False
                exits = list(self._exit_subs)
        except (TypeError, ValueError):
            pass
        return enters, exits


class Computed(Signal):
    """
    Derived Signal: recomputes fn(*deps) when any dep notifies.
    Automatically subscribes to each dep.
    """

    def __init__(
        self,
        fn: Callable,
        *deps: Signal,
        name: str = "",
        **kw: Any,
    ) -> None:
        initial = fn(*[d.get() for d in deps])
        super().__init__(initial, name=name or "computed", **kw)
        self._fn   = fn
        self._deps = deps
        for d in deps:
            d.subscribe(self._on_dep)

    def _on_dep(self, _new: Any, _old: Any) -> None:
        val = self._fn(*[d.get() for d in self._deps])
        self.set(val)


# ── BehaviorSubject ───────────────────────────────────────────────────────────

class BehaviorSubject(Signal):
    """
    Signal that immediately emits to new on_enter subscribers when already
    in hysteresis , the core BehaviorSubject contract from rxRust/RxCpp.

    Late subscribers don't miss the current state: if a new diagnostic loop
    registers on_enter after the metric that crossed the threshold, it still
    receives a synchronous call with the current value.

    All other Signal behaviour is unchanged.
    """

    def on_enter(self, fn: Callable) -> Callable:
        unsub = super().on_enter(fn)
        # BehaviorSubject: replay current state to the new subscriber
        with self._lock:
            already_in = self._in_hysteresis
            current    = self._ema
        if already_in and current is not None:
            try:
                fn(current)
            except Exception:
                pass
        return unsub


# ── FilteredSignal ────────────────────────────────────────────────────────────

class FilteredSignal:
    """
    Derived observable that gates on_enter/on_exit propagation from a source
    Signal behind a runtime predicate (filter operator, rxRust/RxCpp).

    The predicate is evaluated at call time (not subscription time) so it can
    close over mutable orchestrator state.  Replaces inline if-chains in
    callbacks with declarative composition::

        # Instead of 6 early-return guards in _on_kv_pressure_high:
        kv_sig.pipe(filter(all_gates_ok)).on_enter(self._do_kv_grow)

    Note: per-gate logging is preserved by keeping the existing callbacks and
    using FilteredSignal for future loops.
    """

    def __init__(self, source: Signal, predicate: Callable[[], bool]) -> None:
        self._pred        = predicate
        self._enter_subs: list[Callable] = []
        self._exit_subs:  list[Callable] = []
        self._subs:       list[Callable] = []
        source.on_enter(self._on_src_enter)
        source.on_exit(self._on_src_exit)
        source.subscribe(self._on_src_change)

    def _on_src_enter(self, val: Any) -> None:
        if not self._pred():
            return
        for cb in list(self._enter_subs):
            try:
                cb(val)
            except Exception:
                pass

    def _on_src_exit(self, val: Any) -> None:
        if not self._pred():
            return
        for cb in list(self._exit_subs):
            try:
                cb(val)
            except Exception:
                pass

    def _on_src_change(self, new: Any, old: Any) -> None:
        if not self._pred():
            return
        for cb in list(self._subs):
            try:
                cb(new, old)
            except Exception:
                pass

    def on_enter(self, fn: Callable) -> "FilteredSignal":
        self._enter_subs.append(fn)
        return self

    def on_exit(self, fn: Callable) -> "FilteredSignal":
        self._exit_subs.append(fn)
        return self

    def subscribe(self, fn: Callable) -> "FilteredSignal":
        self._subs.append(fn)
        return self


def filter(predicate: Callable[[], bool]) -> Callable[[Signal], FilteredSignal]:  # noqa: A001
    """Rx filter operator factory , use with Signal.pipe()."""
    return lambda source: FilteredSignal(source, predicate)


# ── PairwiseSignal ────────────────────────────────────────────────────────────

class PairwiseSignal:
    """
    Derived observable: emits (previous, current) tuples on each source
    notification (pairwise operator from rxRust / RxCpp).

    Used for transition detection without manual _last_phase tracking::

        phase_sig.pipe(pairwise()).subscribe(on_phase_transition)
        # on_phase_transition receives: (old_phase, new_phase), unused_old
    """

    def __init__(self, source: Signal) -> None:
        self._prev: Any   = source.get()
        self._subs: list[Callable] = []
        source.subscribe(self._on_src)

    def _on_src(self, new: Any, _old: Any) -> None:
        pair = (self._prev, new)
        self._prev = new
        for cb in list(self._subs):
            try:
                cb(pair, None)
            except Exception:
                pass

    def subscribe(self, fn: Callable) -> "PairwiseSignal":
        self._subs.append(fn)
        return self


def pairwise() -> Callable[[Signal], PairwiseSignal]:
    """Rx pairwise operator factory , use with Signal.pipe()."""
    return lambda source: PairwiseSignal(source)


# ── WithLatestFrom ────────────────────────────────────────────────────────────

class WithLatestFrom:
    """
    When the trigger Signal notifies, combines its value with the latest
    value from each of the other signals and calls subscribers with
    (trigger_val, other1_val, other2_val, ...) , with_latest_from from rxRust.

    Useful for gated combinations: "when KV pressure fires, combine with the
    latest clock state and phase to decide the action"::

        kv_sig.pipe(with_latest_from(clock_sig, phase_sig))
              .subscribe(on_kv_with_context)
    """

    def __init__(self, trigger: Signal, *others: Signal) -> None:
        self._others = others
        self._subs:  list[Callable] = []
        trigger.subscribe(self._on_trigger)
        # Also fire when trigger enters hysteresis
        trigger.on_enter(lambda v: self._dispatch(v))

    def _dispatch(self, trigger_val: Any) -> None:
        latest = tuple(s.get() for s in self._others)
        args = (trigger_val,) + latest
        for cb in list(self._subs):
            try:
                cb(*args)
            except Exception:
                pass

    def _on_trigger(self, new: Any, _old: Any) -> None:
        self._dispatch(new)

    def subscribe(self, fn: Callable) -> "WithLatestFrom":
        self._subs.append(fn)
        return self


def with_latest_from(*others: Signal) -> Callable[[Signal], WithLatestFrom]:
    """Rx with_latest_from operator factory , use with Signal.pipe()."""
    return lambda trigger: WithLatestFrom(trigger, *others)


# ── ScanSignal ────────────────────────────────────────────────────────────────

class ScanSignal(Signal):
    """
    Derived Signal that applies an accumulator function over each source
    emission and emits the running result (scan operator from rxRust / RxCpp).

    The built-in EWMA in Signal is a special case (acc = prev*(1-α)+v*α).
    ScanSignal generalises to any fold::

        # Running total of overflow bytes
        overflow_sig.pipe(scan(lambda acc, v: acc + v, 0)).subscribe(on_total)

        # Count consecutive high-pressure events
        kv_sig.pipe(scan(lambda n, v: n+1 if v > 0.85 else 0, 0)).subscribe(on_run)
    """

    def __init__(
        self,
        source: Signal,
        fn: Callable[[Any, Any], Any],
        seed: Any,
        *,
        name: str = "scan",
    ) -> None:
        super().__init__(seed, name=name)
        self._acc = seed
        self._fn  = fn
        source.subscribe(self._on_src)

    def _on_src(self, new: Any, _old: Any) -> None:
        self._acc = self._fn(self._acc, new)
        self.set(self._acc)


def scan(fn: Callable[[Any, Any], Any], seed: Any) -> Callable[[Signal], ScanSignal]:
    """Rx scan operator factory , use with Signal.pipe()."""
    return lambda source: ScanSignal(source, fn, seed)


class PidController:
    """Standard clamped PID controller (MuxFlow port — DI-6, see
    workflow/porting-reference.md §DI-6). Continuous, proportional-response
    complement to Signal's discrete enter/exit hysteresis: where a Signal
    decides WHETHER a loop reacts at all (bang-bang, debounced), a
    PidController decides HOW MUCH — e.g. scaling a KV-grow step or a
    prefetch-throttle duty cycle smoothly toward a setpoint instead of a
    fixed halve/restore jump.

    setpoint and every gain must be derived from MEASURED state at the call
    site (e.g. gb_orchestrator's `_sm_clock_max` ratchet), never a hardcoded
    absolute value (rule) — this class itself has no hardware knowledge, it
    only does the arithmetic.

    Anti-windup: the integral term is clamped to [out_min, out_max] / max(ki,
    epsilon) so a long-saturated error can't leave a huge integral that then
    overshoots once the process recovers (the classic PID windup failure).
    """

    def __init__(self, kp: float, ki: float, kd: float, setpoint: float,
                out_min: float = 0.0, out_max: float = 1.0) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint = setpoint
        self.out_min, self.out_max = out_min, out_max
        self._integral = 0.0
        self._prev_error: Optional[float] = None
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._integral = 0.0
            self._prev_error = None

    def update(self, measured: float, dt: float) -> float:
        """One control step. dt in seconds (must be > 0 — the caller's poll
        interval). Returns the clamped control output in [out_min, out_max]."""
        if dt <= 0:
            dt = 1e-6
        with self._lock:
            error = self.setpoint - measured
            self._integral += error * dt
            # Anti-windup clamp: bound the integral's own contribution to the
            # output range so it can never alone exceed [out_min, out_max].
            if self.ki > 0:
                i_bound = (self.out_max - self.out_min) / self.ki
                self._integral = max(-i_bound, min(i_bound, self._integral))
            derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
            self._prev_error = error
            output = self.kp * error + self.ki * self._integral + self.kd * derivative
            return max(self.out_min, min(self.out_max, output))

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_reactive.py — reactive-state primitive for GreenBoost.

Signal   — a change-gated observable value.  Generalises the existing
           TelemetryManager.add_callback fan-out (gb_telemetry.py:807) with
           explicit change detection, confirm-count debounce, EWMA smoothing,
           and hysteresis banding.  No external dependencies; stdlib only.

Computed — a derived Signal that recomputes when any dependency notifies.

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

            # Threshold crossed and confirmed — proceed
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

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_orchestrator.py — GreenBoost reactive signal-driven orchestrator.

Consumes the telemetry/flow signals that GreenBoost already publishes
and feeds decisions back into live tunables, closing three dead feedback
loops that previously generated signals but never acted on them:

  Loop A — ECC responder:       ecc_dbe_volatile delta → raise safety
                                 reserve + shrink KV reserve (one-way ratchet)
  Loop B — Workstation governor: non-GB VRAM growth → yield VRAM via
                                 workstation_reserve (hysteresis ±CONFIRM_POLLS)
  Loop C — Predictive KV grow:  kv_pressure approaching reserve AND phase in
                                 STEADY/INFERENCE → pre-emptively grow KV
  Loop D — Thermal governor:    sustained high temp → advisory safety raise

Default mode: DRY-RUN (observe + log + surface in vitals --json).
Set GB_ORCH_ACTUATE=1 to enable real lever moves.

Modes:
  "supervisor"  — root daemon; actuates sysfs/ioctl levers via GbControl.
  "process"     — inference process; may only write the control file
                  (workstation_reserve hint) and call tier eviction.

Self-fighting arbitration:
  1. ECC (A) vs KV-grow (C) on kv_reserve_mb: ecc_degraded flag gates Loop C.
  2. B + C jointly consuming T1: t1_budget accountant clamps grows.
  3. supervisor vs gb_init on system levers: structurally prevented (process
     mode writes control-file hints only; kernel CAP is the backstop).

Separated from gb_supervisor so that gb_init can import it without pulling
in root-daemon code.  gb_supervisor constructs ReactiveOrchestrator(mode="supervisor")
and feeds it in _tick; gb_init constructs mode="process" in _bootstrap.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from gb_reactive import Signal

if TYPE_CHECKING:
    from gb_telemetry import GpuMetrics
    from gb_control   import GbControl

log = logging.getLogger("gb_orchestrator")

# ── tunable constants (conservative defaults) ─────────────────────────────────
_KV_STEP_MB          = 512    # max KV grow/shrink per tick
_KV_FLOOR_MB         = 0      # minimum KV reserve
_KV_PRESSURE_ENTER   = 0.85   # on_enter: kv_used / kv_reserve > 85%
_KV_PRESSURE_EXIT    = 0.60   # on_exit: drop back below 60%
_KV_CONFIRM          = 2      # polls before kv_pressure enter fires
_SAFETY_STEP_GB      = 1      # max safety_reserve raise per ECC/thermal event
_SAFETY_MAX_GB       = 32     # never raise safety_reserve above this
_WS_STEP_MB          = 256    # workstation reserve grow/shrink step
_WS_FRACTION_ENTER   = 0.10   # enter when non_gb_mb > 10% of total VRAM
_WS_FRACTION_EXIT    = 0.07   # exit when non_gb_mb < 7% of total VRAM
_TEMP_ENTER_C        = 83.0   # thermal governor enter threshold
_TEMP_EXIT_C         = 75.0   # thermal governor exit threshold
_TEMP_CONFIRM        = 3      # sustained polls before thermal fires
_WEIGHTS_FLOOR_MB    = 2048   # minimum T1 VRAM reserved for weights

_ORCH_STATE_FILE = Path("/run/greenboost/orch_state.json")


class ReactiveOrchestrator:
    """
    Reactive closed-loop GreenBoost orchestrator.

    Parameters
    ----------
    mode        "supervisor" (root) or "process" (per-inference).
    control     GbControl instance.  If None, a default one is created.
    confirm_polls
                CONFIRM_POLLS for workstation governor (mirrors gb_supervisor.py:78).
    """

    def __init__(
        self,
        mode: str = "supervisor",
        control: "Optional[GbControl]" = None,
        confirm_polls: int = 3,
    ) -> None:
        self._mode  = mode
        self._actuate = os.environ.get("GB_ORCH_ACTUATE", "0") == "1"

        # Lazy import: gb_control may not be available in test environments
        if control is None:
            try:
                from gb_control import GbControl
                self._ctrl: "Optional[GbControl]" = GbControl()
            except ImportError:
                self._ctrl = None
        else:
            self._ctrl = control

        # ── global arbitration state ──────────────────────────────────────────
        self.ecc_degraded   = False   # Loop C disabled while ECC ratchet engaged
        self._ecc_seen      = 0       # monotone ratchet: never decreases
        self._total_vram_mb = 0       # learned from first metrics snapshot
        self._ws_reserve_mb = 0       # current workstation_reserve target

        # ── Loop A — ECC ratchet (monotone, confirm=1) ────────────────────────
        self._ecc_sig = Signal(0, name="ecc_dbe", confirm=1)
        self._ecc_sig.subscribe(self._on_ecc_change)

        # ── Loop B — workstation governor (EWMA, hysteresis, confirm) ─────────
        # Hysteresis thresholds are fractional; absolute MB derived once
        # total_vram_mb is known.  We feed raw non_gb_mb and compare to
        # the computed threshold in the subscriber.
        self._non_gb_sig = Signal(
            0.0,
            name="non_gb_mb",
            ewma_alpha=1.0 / 8.0,
            confirm=confirm_polls,
        )
        self._ws_enter_mb: int = 0   # set on first metrics with total_mb
        self._ws_exit_mb:  int = 0
        self._ws_above: bool = False
        self._non_gb_sig.subscribe(self._on_non_gb_change)

        # ── Loop C — predictive KV grow (hysteresis) ─────────────────────────
        self._kv_pressure_sig = Signal(
            0.0,
            name="kv_pressure",
            ewma_alpha=1.0 / 8.0,
            hysteresis=(_KV_PRESSURE_ENTER, _KV_PRESSURE_EXIT),
            confirm=_KV_CONFIRM,
        )
        self._kv_spilled_sig = Signal(False, name="kv_spilled", confirm=1)
        self._kv_pressure_sig.on_enter(self._on_kv_pressure_high)
        self._kv_pressure_sig.on_exit(self._on_kv_pressure_ok)
        self._kv_spilled_sig.subscribe(self._on_kv_spilled)

        # ── Loop D — thermal governor (EWMA, hysteresis, confirm) ────────────
        self._temp_sig = Signal(
            0.0,
            name="temp_c",
            ewma_alpha=1.0 / 8.0,
            hysteresis=(_TEMP_ENTER_C, _TEMP_EXIT_C),
            confirm=_TEMP_CONFIRM,
        )
        self._temp_sig.on_enter(self._on_temp_high)
        self._temp_sig.on_exit(self._on_temp_ok)

        # ── decision log (ring-buffer, last 64) ───────────────────────────────
        self._decisions: list[dict] = []

        # ── ECC degraded state persists across restarts ───────────────────────
        self._restore_ecc_degraded()

    # ── public feed methods ───────────────────────────────────────────────────

    def on_metrics(self, m: "GpuMetrics") -> None:
        """
        Called each telemetry poll (via telemetry.add_callback or supervisor _tick).
        Feeds all Signals from the GpuMetrics snapshot.
        """
        # Learn total VRAM once; derive workstation governor thresholds
        if m.fb_total_mb > 0 and self._total_vram_mb == 0:
            self._total_vram_mb = m.fb_total_mb
            self._ws_enter_mb   = int(m.fb_total_mb * _WS_FRACTION_ENTER)
            self._ws_exit_mb    = int(m.fb_total_mb * _WS_FRACTION_EXIT)
            log.info(
                "[gb_orchestrator] init: total_vram=%d MiB ws_enter=%d ws_exit=%d",
                self._total_vram_mb, self._ws_enter_mb, self._ws_exit_mb,
            )

        # Loop A: ECC — ratchet; only set when count rises
        ecc = m.ecc_dbe_volatile
        if ecc > self._ecc_seen:
            self._ecc_sig.set(ecc)
            # Do not update _ecc_seen here; the subscriber does it

        # Loop C: KV pressure (needs enriched GbPoolInfo; skip if absent)
        if m.gb is not None:
            kv_pressure = m.kv_pressure   # property: kv_used/kv_reserve (0..1+)
            kv_spilled  = m.kv_spilled    # property: kv_t2_mb > 0
            self._kv_pressure_sig.set(kv_pressure)
            if kv_spilled:
                self._kv_spilled_sig.set(True)

        # Loop D: thermal
        if m.temp_c > 0.0:
            self._temp_sig.set(m.temp_c)

    def feed_vram_state(self, non_gb_mb: float, total_mb: int) -> None:
        """
        Called by supervisor with the non-GB VRAM pressure data it already
        computes in _VramMonitor.poll().  Drives Loop B.
        """
        if total_mb > 0:
            if self._total_vram_mb == 0:
                self._total_vram_mb = total_mb
                self._ws_enter_mb   = int(total_mb * _WS_FRACTION_ENTER)
                self._ws_exit_mb    = int(total_mb * _WS_FRACTION_EXIT)
        self._non_gb_sig.set(non_gb_mb)

    # ── Loop A — ECC ratchet ──────────────────────────────────────────────────

    def _on_ecc_change(self, new: int, old: Any) -> None:
        """Fires when ecc_dbe_volatile count increases."""
        new_errors = int(new) - self._ecc_seen
        if new_errors <= 0:
            return
        self._ecc_seen      = int(new)
        self.ecc_degraded   = True

        decision = {
            "loop": "A_ecc_responder",
            "ts":   time.time(),
            "ecc_dbe": int(new),
            "new_errors": new_errors,
        }
        log.critical(
            "[gb_orchestrator] Loop A: ECC DBE +%d (total=%d) — raising safety reserve, shrinking KV reserve",
            new_errors, int(new),
        )

        if self._ctrl is not None:
            old_safety = self._ctrl._last.get("safety_reserve_gb", (1, 0))[0] or 1
            r_safety = self._ctrl.set_safety_reserve_gb(
                min(_SAFETY_MAX_GB, (old_safety or 1) + _SAFETY_STEP_GB),
                reason=f"ecc_dbe_delta={new_errors}",
            )
            old_kv = self._ctrl._last.get("kv_reserve_mb", (512, 0))[0] or 512
            r_kv = self._ctrl.set_kv_reserve_mb(
                max(_KV_FLOOR_MB, (old_kv or 512) - _KV_STEP_MB),
                reason=f"ecc_dbe_degraded",
            )
            decision.update({"safety_result": r_safety.reason, "kv_result": r_kv.reason})

        # Persist the degraded flag (read back on supervisor restart)
        self._persist_ecc_degraded()
        self._record_decision(decision)

    # ── Loop B — workstation governor ────────────────────────────────────────

    def _on_non_gb_change(self, new: float, old: Any) -> None:
        """Fires when EWMA of non-GB VRAM changes meaningfully."""
        if self._ws_enter_mb <= 0:
            return  # thresholds not yet known

        ema_mb = float(new)
        entered = (not self._ws_above) and (ema_mb >= self._ws_enter_mb)
        exited  = self._ws_above and (ema_mb < self._ws_exit_mb)

        if not entered and not exited:
            return

        decision: dict = {
            "loop": "B_ws_governor",
            "ts":   time.time(),
            "non_gb_mb_ema": round(ema_mb, 1),
        }

        if entered:
            self._ws_above      = True
            new_ws = min(
                self._total_vram_mb // 4,   # max 25% of total VRAM
                self._ws_reserve_mb + _WS_STEP_MB,
            )
            log.info(
                "[gb_orchestrator] Loop B: non-GB VRAM %.0f MiB > enter=%.0f — "
                "raising workstation_reserve to %d MiB",
                ema_mb, self._ws_enter_mb, new_ws,
            )
            if self._ctrl is not None:
                r = self._ctrl.set_workstation_reserve_mb(new_ws, reason="non_gb_vram_pressure")
                if r.applied:
                    self._ws_reserve_mb = new_ws
            decision["action"] = f"raise_ws_reserve_{new_ws}_mb"

        elif exited:
            self._ws_above = False
            new_ws = max(0, self._ws_reserve_mb - _WS_STEP_MB)
            log.info(
                "[gb_orchestrator] Loop B: non-GB VRAM %.0f MiB < exit=%.0f — "
                "restoring workstation_reserve to %d MiB",
                ema_mb, self._ws_exit_mb, new_ws,
            )
            if self._ctrl is not None:
                r = self._ctrl.set_workstation_reserve_mb(new_ws, reason="non_gb_vram_cleared")
                if r.applied:
                    self._ws_reserve_mb = new_ws
            decision["action"] = f"lower_ws_reserve_{new_ws}_mb"

        self._record_decision(decision)

    # ── Loop C — predictive KV grow ───────────────────────────────────────────

    def _on_kv_pressure_high(self, value: float) -> None:
        """Fires when kv_pressure EMA crosses 0.85."""
        if self.ecc_degraded:
            log.info(
                "[gb_orchestrator] Loop C: kv_pressure=%.2f BUT ecc_degraded — skipping KV grow",
                value,
            )
            return

        decision = {
            "loop": "C_predictive_kv",
            "ts":   time.time(),
            "kv_pressure": round(float(value), 3),
            "trigger": "pressure_high",
        }
        self._do_kv_grow("kv_pressure_approaching_reserve", decision)
        self._record_decision(decision)

    def _on_kv_pressure_ok(self, value: float) -> None:
        log.debug("[gb_orchestrator] Loop C: kv_pressure=%.2f — OK, no action", value)

    def _on_kv_spilled(self, new: bool, old: Any) -> None:
        """Fires when KV has already spilled to T2 — immediate grow."""
        if not new or self.ecc_degraded:
            return
        decision = {
            "loop": "C_predictive_kv",
            "ts":   time.time(),
            "trigger": "kv_spilled_to_t2",
        }
        self._do_kv_grow("kv_spilled_to_slow_t2", decision)
        self._record_decision(decision)

    def _do_kv_grow(self, reason: str, decision: dict) -> None:
        """Common KV grow logic, guarded by t1_budget accountant."""
        if self._ctrl is None:
            return
        old_kv = self._ctrl._last.get("kv_reserve_mb", (0, 0))[0] or 0
        new_kv = old_kv + _KV_STEP_MB

        # T1 budget guard: don't jointly over-commit T1 with Loop B
        headroom = self._t1_headroom_mb()
        if headroom > 0 and _KV_STEP_MB > headroom:
            clamped = max(0, headroom)
            log.info(
                "[gb_orchestrator] Loop C: T1 budget clamped KV grow %d→%d MiB "
                "(headroom=%d MiB)",
                _KV_STEP_MB, clamped, headroom,
            )
            new_kv = old_kv + clamped
            decision["clamped_to_headroom"] = clamped

        if new_kv <= old_kv:
            decision["action"] = "kv_grow_skipped_no_headroom"
            return

        r = self._ctrl.set_kv_reserve_mb(new_kv, reason=reason)
        decision["action"]     = f"kv_grow_{old_kv}→{new_kv}_mb"
        decision["kv_result"]  = r.reason

    def _t1_headroom_mb(self) -> int:
        """Conservative T1 headroom = total_vram - ws_reserve - kv_reserve - weights_floor."""
        if self._total_vram_mb <= 0:
            return 0
        kv  = self._ctrl._last.get("kv_reserve_mb", (0, 0))[0] if self._ctrl else 0
        ws  = self._ws_reserve_mb
        return self._total_vram_mb - (kv or 0) - ws - _WEIGHTS_FLOOR_MB

    # ── Loop D — thermal governor ─────────────────────────────────────────────

    def _on_temp_high(self, temp: float) -> None:
        decision = {
            "loop": "D_thermal",
            "ts":   time.time(),
            "temp_c_ema": round(float(temp), 1),
            "action": "advisory_safety_raise",
        }
        log.info(
            "[gb_orchestrator] Loop D: temp=%.1f°C (EMA) — advisory safety_reserve raise",
            temp,
        )
        if self._ctrl is not None:
            old = self._ctrl._last.get("safety_reserve_gb", (1, 0))[0] or 1
            r = self._ctrl.set_safety_reserve_gb(
                min(_SAFETY_MAX_GB, (old or 1) + _SAFETY_STEP_GB),
                reason=f"thermal_governor_temp={temp:.0f}",
            )
            decision["safety_result"] = r.reason
        self._record_decision(decision)

    def _on_temp_ok(self, temp: float) -> None:
        log.info(
            "[gb_orchestrator] Loop D: temp=%.1f°C — thermal OK, no action", temp
        )

    # ── ECC degraded persistence ──────────────────────────────────────────────

    def _persist_ecc_degraded(self) -> None:
        try:
            from gb_supervisor import ECC_DBE_FLAG
            ECC_DBE_FLAG.write_text(str(self._ecc_seen))
        except Exception:
            # Fallback path: write directly
            try:
                Path("/run/greenboost/ecc_dbe_flag").write_text(str(self._ecc_seen))
            except Exception:
                pass

    def _restore_ecc_degraded(self) -> None:
        """On startup, restore ECC degraded state from the flag file."""
        try:
            val = int(Path("/run/greenboost/ecc_dbe_flag").read_text().strip())
            if val > 0:
                self._ecc_seen    = val
                self.ecc_degraded = True
                log.warning(
                    "[gb_orchestrator] Restored ECC degraded state: seen=%d DBE",
                    self._ecc_seen,
                )
        except Exception:
            pass

    # ── decision log ─────────────────────────────────────────────────────────

    def _record_decision(self, decision: dict) -> None:
        self._decisions.append(decision)
        if len(self._decisions) > 64:
            self._decisions = self._decisions[-64:]
        # Write state file for vitals --json and other short-lived observers
        self._write_state_file()

    def _write_state_file(self) -> None:
        state = self.dump()
        try:
            tmp = _ORCH_STATE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=2))
            tmp.rename(_ORCH_STATE_FILE)
        except Exception:
            pass

    # ── dump / observability ─────────────────────────────────────────────────

    def dump(self) -> dict:
        """
        Full state snapshot for debug_dump() / vitals --json decisions block.
        Written to /run/greenboost/orch_state.json on every decision.
        """
        signals = [
            self._ecc_sig.dump(),
            self._non_gb_sig.dump(),
            self._kv_pressure_sig.dump(),
            self._kv_spilled_sig.dump(),
            self._temp_sig.dump(),
        ]
        for s in signals:
            # Add human-readable state hint
            if s["name"] == "ecc_dbe":
                s["state"] = "degraded" if self.ecc_degraded else "ok"
            elif s["name"] == "non_gb_mb":
                s["state"] = "pressure" if self._ws_above else "ok"
            elif s["name"] == "kv_pressure":
                s["state"] = "high" if s.get("in_hysteresis") else "ok"
            elif s["name"] == "temp_c":
                s["state"] = "hot" if s.get("in_hysteresis") else "ok"
            else:
                s["state"] = "ok"
        return {
            "mode":         self._mode,
            "actuate":      self._actuate,
            "ecc_degraded": self.ecc_degraded,
            "ecc_seen":     self._ecc_seen,
            "ws_above":     self._ws_above,
            "ws_reserve_mb": self._ws_reserve_mb,
            "total_vram_mb": self._total_vram_mb,
            "signals":      signals,
            "recent_decisions": self._decisions[-8:],
            "control":      self._ctrl.dump() if self._ctrl else {},
        }

    def stop(self) -> None:
        """Clean shutdown hook (called from gb_init._shutdown or supervisor)."""
        pass

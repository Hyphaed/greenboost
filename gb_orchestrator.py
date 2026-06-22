#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_orchestrator.py — GreenBoost reactive signal-driven orchestrator.

Consumes the telemetry/flow signals that GreenBoost already publishes
and feeds decisions back into live tunables, closing feedback loops:

  Loop A — ECC responder:       ecc_dbe_volatile delta → raise safety
                                 reserve + shrink KV reserve (one-way ratchet)
  Loop B — Workstation governor: non-GB VRAM growth → yield VRAM via
                                 workstation_reserve (hysteresis ±CONFIRM_POLLS)
  Loop C — Predictive KV grow:  kv_pressure approaching reserve AND phase in
                                 STEADY/INFERENCE → pre-emptively grow KV
  Loop D — Thermal governor:    sustained high temp → advisory safety raise
  Loop E — VRAM pressure:       fb_used_pct > 87% sustained → ModelTierManager
                                 auto_evict; clears when < 70%
  Loop G — Mem-BW stress:      mem_copy_util_pct > 85% sustained → suppress
                                 Loop C KV grow (mem BW already saturated)
  Loop H — ECC SBE monitor:    ecc_sbe_volatile delta → advisory warning;
                                 sets sbe_elevated flag for operator awareness
  Loop I — SM clock throttle:  sm_clock_mhz drops 12%+ from observed max →
                                 clock_throttled=True; gates Loop C KV grow.
                                 Mild zone (8–12%): halves KV step before hard gate.
  Loop J — Phase KV reclaim:   shim_phase INFERENCE → IDLE/DEEP_IDLE transition
                                 → shrink KV reserve to floor immediately, freeing
                                 VRAM for the next model load or weight prefetch.
  Loop K — Post-throttle restore: when Loop I clock_throttled clears AND
                                 kv_pressure_sig is still in hysteresis (grows were
                                 suppressed during throttle), fires one deferred KV
                                 grow so pressure relief isn't indefinitely delayed.

  -- Continuous OS tuning (supervisor mode + GB_OS_TUNE=1 only) --
  Loop O — Performance envelope: phase enters MODEL_LOAD/INFERENCE/STEADY →
                                 CPU governor=performance, EPP=performance,
                                 NUMA balancing off, swappiness low, GPU
                                 persistence + clock lock + power=TDP. Phase
                                 enters IDLE/DEEP_IDLE → restore_baseline().
  Loop P — Thermal/throttle power cap: thermal_stress or clock_throttled →
                                 step GPU power limit DOWN (trade clock for
                                 sustained throughput); clears → step back up.
  Loop Q — Memory-pressure VM tune: PSI mem "some" high → raise
                                 watermark_scale_factor, lower swappiness +
                                 dirty_background_ratio; relax on clear.
  Loop R — CPU/IO pressure assist: PSI cpu "some" high during a performance
                                 phase → reassert governor=performance; PSI io
                                 "some" high during T3 spill → advisory log.
  Loop S — PCIe saturation: pcie_saturated (sustained) → prefetch-throttle
                                 control-file hint; clears on recovery.

  All Loops O-S are no-ops unless mode=="supervisor" AND GB_OS_TUNE=1 AND
  gaming_mode is off (Gaming Suite owns governor/clocks while a game runs).

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

from gb_reactive import Signal, BehaviorSubject, pairwise, filter as rx_filter

if TYPE_CHECKING:
    from gb_telemetry import GpuMetrics
    from gb_control   import GbControl

log = logging.getLogger("gb_orchestrator")

# ── tunable constants (conservative defaults; topology overrides at runtime) ──
_KV_STEP_MB          = 512    # max KV grow/shrink per tick
_KV_FLOOR_MB         = 0      # minimum KV reserve
_KV_PRESSURE_ENTER   = 0.85   # on_enter: kv_used / kv_reserve > 85%
_KV_PRESSURE_EXIT    = 0.60   # on_exit: drop back below 60%
_KV_CONFIRM          = 2      # polls before kv_pressure enter fires
_SAFETY_STEP_GB      = 1      # max safety_reserve raise per ECC/thermal event
_SAFETY_MAX_GB       = 32     # absolute fallback ceiling for safety_reserve
_WS_STEP_MB          = 256    # workstation reserve grow/shrink step
_WS_FRACTION_ENTER   = 0.10   # enter when non_gb_mb > 10% of total VRAM
_WS_FRACTION_EXIT    = 0.07   # exit when non_gb_mb < 7% of total VRAM
_TEMP_ENTER_C        = 83.0   # thermal governor enter threshold
_TEMP_EXIT_C         = 75.0   # thermal governor exit threshold
_TEMP_CONFIRM        = 3      # sustained polls before thermal fires
_WEIGHTS_FLOOR_MB    = 2048   # minimum T1 VRAM reserved for weights
_VRAM_DEMOTE_ENTER   = 87.0   # Loop E: fb_used_pct enter threshold
_VRAM_DEMOTE_EXIT    = 70.0   # Loop E: fb_used_pct exit threshold
_VRAM_CONFIRM        = 3      # sustained polls before Loop E fires
_MEM_BW_ENTER        = 85.0   # Loop G: mem_copy_util_pct enter threshold
_MEM_BW_EXIT         = 65.0   # Loop G: mem_copy_util_pct exit threshold
_MEM_BW_CONFIRM      = 2      # sustained polls before Loop G fires
_SM_CLOCK_WARMUP     = 3      # warmup polls before Loop I ratio tracking starts
_SM_CLOCK_DROP_ENTER = 12.0   # Loop I: enter when SM clock drops 12%+ from max
_SM_CLOCK_DROP_EXIT  = 6.0    # Loop I: exit when drop recovers below 6%
_SM_CLOCK_DROP_MILD  = 8.0    # Loop I: mild zone — halve KV step (pre-throttle warning)
_SM_CLOCK_CONFIRM    = 2      # polls before Loop I fires
# ── Loop N — adaptive telemetry poll rate ─────────────────────────────────────
_POLL_MS_INFERENCE   = 250    # faster during active token generation
_POLL_MS_DEFAULT     = 500    # default / model-load / unknown phase
_POLL_MS_IDLE        = 1000   # slower during idle — save CPU between requests

# ── continuous OS tuning (Loops O-S; supervisor + GB_OS_TUNE=1 only) ──────────
_PERF_PHASES         = ("MODEL_LOAD", "INFERENCE", "STEADY")
_PSI_MEM_ENTER       = 10.0   # Loop Q: PSI mem "some" avg10 enter
_PSI_MEM_EXIT        = 4.0
_PSI_MEM_CONFIRM     = 2
_PSI_CPU_ENTER       = 20.0   # Loop R: PSI cpu "some" avg10 enter
_PSI_CPU_EXIT        = 8.0
_PSI_CPU_CONFIRM     = 2
_PSI_IO_ENTER        = 20.0   # Loop R: PSI io "some" avg10 enter
_PSI_IO_EXIT         = 8.0
_PSI_IO_CONFIRM      = 2
_GPU_POWER_STEP_W    = 15     # Loop P: power-limit step per thermal/throttle event
_GPU_POWER_FLOOR_FRAC = 0.5   # never step power limit below 50% of TDP

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
    tier_manager
                ModelTierManager instance for Loop E and Loop F eviction.
    cluster_tel
                ClusterTelemetryManager.  When provided, Loop E also fires when
                any feeder GPU is VRAM-pressured (B3 cluster-aware extension).
    """

    def __init__(
        self,
        mode: str = "supervisor",
        control: "Optional[GbControl]" = None,
        confirm_polls: int = 3,
        tier_manager=None,
        cluster_tel=None,
        tel_manager=None,
    ) -> None:
        self._mode  = mode
        self._actuate = os.environ.get("GB_ORCH_ACTUATE", "0") == "1"
        # Continuous OS tuning (Loops O-S): supervisor-only, opt-in via env,
        # deferred entirely while the Gaming Suite owns governor/clocks.
        self.os_tune_enabled = (mode == "supervisor") and (os.environ.get("GB_OS_TUNE", "0") == "1")
        self.gaming_mode: bool = False
        self._tier_manager = tier_manager
        self._cluster_tel  = cluster_tel      # B3: ClusterTelemetryManager or None
        self._cluster_pressure: bool = False  # B3: feeder-triggered pressure state
        self._tel_manager  = tel_manager      # N: TelemetryManager for adaptive poll rate
        self._pcie_degraded_logged: bool = False  # topology advisory — log once

        # ── topology-driven calibration (overrides module-level defaults) ────────
        try:
            from gb_topology import get_topology
            _topo = get_topology()
            self._kv_step_mb      = _topo.kv_step_mb       # kv_reserve_mb // 4
            self._safety_max_gb   = _topo.safety_max_gb    # safety_reserve_gb + 2
            self._kv_floor_mb     = _topo.kv_reserve_mb    # shrink floor = topology baseline
            self._weights_floor_mb = _WEIGHTS_FLOOR_MB
        except Exception:
            self._kv_step_mb      = _KV_STEP_MB
            self._safety_max_gb   = _SAFETY_MAX_GB
            self._kv_floor_mb     = _KV_FLOOR_MB
            self._weights_floor_mb = _WEIGHTS_FLOOR_MB

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
        self.thermal_stress = False   # Loop D→C gate: suppresses KV grow when GPU is hot
        self.mem_bw_stress  = False   # Loop G→C gate: suppresses KV grow when HBM BW saturated
        self.sbe_elevated   = False   # Loop H: SBE rate above zero — advisory warning only
        self.clock_throttled = False  # Loop I: SM clock dropped >12% from observed max
        self._ecc_seen      = 0       # monotone ratchet: never decreases
        self._sbe_seen      = 0       # SBE monotone counter (advisory; resets on driver restart)
        self._sm_clock_max  = 0       # ratcheting max observed SM clock (MHz)
        self._sm_clock_polls = 0      # warmup counter for Loop I
        self._total_vram_mb = 0       # learned from first metrics snapshot
        self._ws_reserve_mb = 0       # current workstation_reserve target
        self._last_metrics: "Optional[GpuMetrics]" = None

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

        # ── Loop E — VRAM pressure → model-tier demotion ──────────────────────
        self._vram_sig = Signal(
            0.0,
            name="fb_used_pct",
            ewma_alpha=1.0 / 6.0,
            hysteresis=(_VRAM_DEMOTE_ENTER, _VRAM_DEMOTE_EXIT),
            confirm=_VRAM_CONFIRM,
        )
        self._vram_sig.on_enter(self._on_vram_pressure_high)
        self._vram_sig.on_exit(self._on_vram_pressure_ok)
        self._vram_pressure: bool = False

        # ── Loop F — DCGM health → proactive tier eviction (opt-in) ──────────
        # Fires when health_ok transitions to False for confirm=2 polls.
        # Actuates only when GB_ORCH_HEALTH_EVICT=1 — default advisory-only
        # to avoid false-positive evictions on consumer GPUs (RTX 5070).
        self._health_sig = Signal(True, name="health_ok", confirm=2)
        self._health_sig.subscribe(self._on_health_change)
        self._health_evict_armed: bool = (
            os.environ.get("GB_ORCH_HEALTH_EVICT", "0") == "1"
        )

        # ── Loop G — memory-bandwidth stress → Loop C gate ───────────────────
        # mem_copy_util_pct > 85% means HBM is saturated.  Growing KV reserve
        # at that point increases bandwidth demand → higher power → thermal/power
        # throttle.  Same suppress-grow pattern as thermal_stress (Loop D→C).
        self._mem_bw_sig = Signal(
            0.0,
            name="mem_copy_util_pct",
            ewma_alpha=1.0 / 8.0,
            hysteresis=(_MEM_BW_ENTER, _MEM_BW_EXIT),
            confirm=_MEM_BW_CONFIRM,
        )
        self._mem_bw_sig.on_enter(self._on_mem_bw_high)
        self._mem_bw_sig.on_exit(self._on_mem_bw_ok)

        # ── Loop H — ECC SBE early-warning (advisory, no lever moves) ────────
        # SBEs are correctable single-bit errors. They precede DBEs and indicate
        # a degrading memory cell. Unlike the DBE ratchet (Loop A), we do NOT
        # move levers on SBEs — the hardware ECC correction handles them. But
        # surfacing them early gives operators visibility before a DBE ratchet fires.
        # confirm=2 debounces a transient SBE spike from cosmic-ray events.
        self._sbe_sig = Signal(0, name="ecc_sbe", confirm=2)
        self._sbe_sig.subscribe(self._on_sbe_change)

        # ── Loop I — SM clock throttle direct detector ────────────────────────
        # Tracks (1 - sm_clock/sm_clock_max)*100 = drop_pct. Signal fires on_enter
        # when drop rises above _SM_CLOCK_DROP_ENTER (inverted ratio convention).
        # After _SM_CLOCK_WARMUP polls, the ratcheting max is reliable enough.
        # Mild zone (8–12%): halve KV step in _do_kv_grow before hard block fires.
        self._clock_drop_sig = Signal(
            0.0,
            name="sm_clock_drop_pct",
            ewma_alpha=1.0 / 6.0,
            hysteresis=(_SM_CLOCK_DROP_ENTER, _SM_CLOCK_DROP_EXIT),
            confirm=_SM_CLOCK_CONFIRM,
        )
        self._clock_drop_sig.on_enter(self._on_clock_throttled)
        self._clock_drop_sig.on_exit(self._on_clock_ok)

        # ── Loop J — phase-transition detection via pairwise() (rxRust pattern) ─
        # _phase_sig holds the latest shim_phase string. pipe(pairwise()) turns
        # each change into a (prev, curr) pair so _on_phase_transition can detect
        # INFERENCE→IDLE without manual _last_phase bookkeeping in on_metrics.
        self._phase_sig = Signal("", name="shim_phase")
        self._phase_sig.pipe(pairwise()).subscribe(self._on_phase_transition)

        # ── Loop Q — host memory-pressure VM tune (PSI mem "some") ───────────
        self._mem_psi_sig = Signal(
            0.0, name="psi_mem_some", ewma_alpha=1.0 / 4.0,
            hysteresis=(_PSI_MEM_ENTER, _PSI_MEM_EXIT), confirm=_PSI_MEM_CONFIRM,
        )
        self._mem_psi_sig.on_enter(self._on_mem_psi_high)
        self._mem_psi_sig.on_exit(self._on_mem_psi_ok)

        # ── Loop R — host CPU/IO pressure assist (PSI cpu/io "some") ─────────
        self._cpu_psi_sig = Signal(
            0.0, name="psi_cpu_some", ewma_alpha=1.0 / 4.0,
            hysteresis=(_PSI_CPU_ENTER, _PSI_CPU_EXIT), confirm=_PSI_CPU_CONFIRM,
        )
        self._cpu_psi_sig.on_enter(self._on_cpu_psi_high)

        self._io_psi_sig = Signal(
            0.0, name="psi_io_some", ewma_alpha=1.0 / 4.0,
            hysteresis=(_PSI_IO_ENTER, _PSI_IO_EXIT), confirm=_PSI_IO_CONFIRM,
        )
        self._io_psi_sig.on_enter(self._on_io_psi_high)

        # ── Loop S — PCIe saturation → prefetch-throttle hint ─────────────────
        self._pcie_sat_sig = Signal(False, name="pcie_saturated", confirm=3)
        self._pcie_sat_sig.subscribe(self._on_pcie_saturation_change)

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
        self._last_metrics = m

        # Loop I: SM clock throttle direct detector — ratchet _sm_clock_max
        # before the phase-transition feed below, since Loop O's handler
        # (fired synchronously from _phase_sig.set when the phase changes)
        # reads _sm_clock_max to size the clock lock on phase entry.
        if m.sm_clock_mhz > 0:
            if m.sm_clock_mhz > self._sm_clock_max:
                self._sm_clock_max = m.sm_clock_mhz
            self._sm_clock_polls += 1
            if self._sm_clock_polls >= _SM_CLOCK_WARMUP and self._sm_clock_max > 0:
                drop_pct = (1.0 - m.sm_clock_mhz / self._sm_clock_max) * 100.0
                self._clock_drop_sig.set(drop_pct)

        # Loop J: feed shim_phase into _phase_sig — pairwise() subscriber handles
        # INFERENCE→IDLE transition detection (rxRust pairwise operator pattern).
        self._phase_sig.set(m.shim_phase)

        # Learn total VRAM once; derive workstation governor thresholds
        if m.fb_total_mb > 0 and self._total_vram_mb == 0:
            self._total_vram_mb = m.fb_total_mb
            self._ws_enter_mb   = int(m.fb_total_mb * _WS_FRACTION_ENTER)
            self._ws_exit_mb    = int(m.fb_total_mb * _WS_FRACTION_EXIT)
            log.info(
                "[gb_orchestrator] init: total_vram=%d MiB ws_enter=%d ws_exit=%d",
                self._total_vram_mb, self._ws_enter_mb, self._ws_exit_mb,
            )

        # Loop A: ECC DBE — ratchet; only set when count rises
        ecc = m.ecc_dbe_volatile
        if ecc > self._ecc_seen:
            self._ecc_sig.set(ecc)
            # Do not update _ecc_seen here; the subscriber does it

        # Loop H: ECC SBE — advisory monitor; only set when count rises
        sbe = m.ecc_sbe_volatile
        if sbe > self._sbe_seen:
            self._sbe_sig.set(sbe)

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

        # Loop G: memory-bandwidth stress gate
        if m.mem_copy_util_pct > 0.0 or m.gpu_util_pct > 0.0:
            self._mem_bw_sig.set(m.mem_copy_util_pct)

        # Loop E: VRAM pressure → tier demotion (local)
        if m.fb_total_mb > 0:
            self._vram_sig.set(m.fb_used_pct)

        # B3: Cluster-aware Loop E — fire demotion when any feeder GPU is pressured,
        # even before the local EMA crosses the 87% threshold.
        if self._cluster_tel is not None:
            cluster_demote = self._cluster_tel.any_should_demote()
            if cluster_demote and not self._cluster_pressure:
                self._cluster_pressure = True
                decision = {
                    "loop": "E_cluster_pressure",
                    "ts": time.time(),
                    "trigger": "any_feeder_should_demote",
                    "fb_used_pct_local": round(float(m.fb_used_pct), 1),
                }
                log.info(
                    "[gb_orchestrator] Loop E (cluster): feeder VRAM pressure "
                    "(local=%.1f%%) — triggering tier demotion", m.fb_used_pct,
                )
                if self._tier_manager is not None and self._last_metrics is not None:
                    try:
                        self._tier_manager.auto_evict(self._last_metrics)
                        decision["action"] = "tier_auto_evict"
                    except Exception as exc:
                        decision["action"] = f"tier_auto_evict_failed: {exc}"
                else:
                    decision["action"] = "no_tier_manager"
                self._record_decision(decision)
            elif not cluster_demote and self._cluster_pressure:
                self._cluster_pressure = False
                log.info("[gb_orchestrator] Loop E (cluster): feeder VRAM pressure cleared")

        # Loop F: DCGM health → proactive tier eviction (opt-in)
        self._health_sig.set(m.health_ok)

        # Loop O gate: gaming mode — Gaming Suite owns governor/clocks while active
        if m.gb is not None:
            self.gaming_mode = bool(m.gb.gaming_mode)

        # Loops Q/R: host OS pressure (SystemProvider; absent when /proc/pressure
        # is unavailable — supervisor-only PSI feed, never blocks the inference path)
        if m.sys is not None:
            self._mem_psi_sig.set(m.sys.psi_mem_some_avg10)
            self._cpu_psi_sig.set(m.sys.psi_cpu_some_avg10)
            self._io_psi_sig.set(m.sys.psi_io_some_avg10)

        # Loop S: PCIe saturation (property is False when topology unavailable)
        self._pcie_sat_sig.set(m.pcie_saturated)

        # Topology advisory — log once when PCIe slot runs below device max.
        # Does not gate any loop; purely an operator warning surfaced in logs.
        if m.topology is not None and m.topology.pcie_degraded and not self._pcie_degraded_logged:
            self._pcie_degraded_logged = True
            log.warning(
                "[gb_orchestrator] PCIe slot degraded: running gen%dx%d "
                "(device max gen%dx%d) — effective T2↔T1 bandwidth %.0f MB/s "
                "vs theoretical %.0f MB/s; B2 gate threshold adjusted automatically.",
                m.topology.pcie_gen_current, m.topology.pcie_width_current,
                m.topology.pcie_gen_max, m.topology.pcie_width_max,
                m.topology.pcie_bw_mb_s, m.topology.pcie_bw_max_mb_s,
            )

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
                min(self._safety_max_gb, (old_safety or 1) + _SAFETY_STEP_GB),
                reason=f"ecc_dbe_delta={new_errors}",
            )
            old_kv = self._ctrl._last.get("kv_reserve_mb", (512, 0))[0] or 512
            r_kv = self._ctrl.set_kv_reserve_mb(
                max(_KV_FLOOR_MB, (old_kv or 512) - self._kv_step_mb),
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
        if self.thermal_stress:
            log.info(
                "[gb_orchestrator] Loop C: kv_pressure=%.2f BUT thermal_stress — "
                "skipping KV grow to avoid adding VRAM pressure under thermal load",
                value,
            )
            return
        if self.mem_bw_stress:
            log.info(
                "[gb_orchestrator] Loop C: kv_pressure=%.2f BUT mem_bw_stress — "
                "skipping KV grow, HBM bandwidth already saturated",
                value,
            )
            return
        # G2: power-near-limit gate — growing KV when near TDP raises bandwidth
        # demand further, risking power-limit clock throttle.
        if self._last_metrics is not None and self._last_metrics.power_near_limit:
            log.info(
                "[gb_orchestrator] Loop C: kv_pressure=%.2f BUT power_near_limit "
                "(%.0f/%.0fW) — skipping KV grow to avoid power throttle",
                value,
                self._last_metrics.power_w,
                self._last_metrics.power_limit_w,
            )
            return
        # Loop I: SM clock throttle gate — the most direct throttle signal.
        # SM clock drop is the measured effect of throttle (not a leading indicator
        # like temp/power). When active, any KV grow would add bandwidth demand,
        # compounding the throttle. Hard block until clock recovers.
        if self.clock_throttled:
            log.info(
                "[gb_orchestrator] Loop C: kv_pressure=%.2f BUT clock_throttled "
                "(SM clock dropped >%.0f%% from max %d MHz) — skipping KV grow",
                value, _SM_CLOCK_DROP_ENTER, self._sm_clock_max,
            )
            return
        # Phase gate: skip KV grows during MODEL_LOAD (no active tokens — KV is empty)
        # and during IDLE/DEEP_IDLE (inference done — shrink, don't grow).
        # Empty shim_phase means shim absent → allow grow (conservative).
        if self._last_metrics is not None:
            phase = self._last_metrics.shim_phase
            if phase in ("MODEL_LOAD", "IDLE", "DEEP_IDLE"):
                log.debug(
                    "[gb_orchestrator] Loop C: kv_pressure=%.2f BUT shim_phase=%s "
                    "— skipping KV grow (no active tokens)",
                    value, phase,
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
        """KV pressure cleared — shrink KV reserve by one step to reclaim VRAM for weights."""
        if self._ctrl is None or self.ecc_degraded:
            return
        old_kv = self._ctrl._last.get("kv_reserve_mb", (0, 0))[0] or 0
        new_kv = max(self._kv_floor_mb, old_kv - self._kv_step_mb)
        if new_kv >= old_kv:
            log.debug(
                "[gb_orchestrator] Loop C: kv_pressure=%.2f OK — KV already at floor (%d MiB)",
                value, old_kv,
            )
            return
        log.info(
            "[gb_orchestrator] Loop C: kv_pressure=%.2f cleared — "
            "shrinking KV reserve %d→%d MiB (floor=%d)",
            value, old_kv, new_kv, self._kv_floor_mb,
        )
        r = self._ctrl.set_kv_reserve_mb(new_kv, reason="kv_pressure_cleared")
        decision = {
            "loop":        "C_predictive_kv",
            "ts":          time.time(),
            "kv_pressure": round(float(value), 3),
            "trigger":     "pressure_cleared",
            "action":      f"kv_shrink_{old_kv}→{new_kv}_mb",
            "kv_result":   r.reason,
        }
        self._record_decision(decision)

    def _on_kv_spilled(self, new: bool, old: Any) -> None:
        """Fires when KV has already spilled to T2 — immediate grow."""
        if not new or self.ecc_degraded:
            return
        # Skip during idle phases — KV spill from previous session, not active tokens
        if self._last_metrics is not None:
            phase = self._last_metrics.shim_phase
            if phase in ("IDLE", "DEEP_IDLE"):
                return
        decision = {
            "loop": "C_predictive_kv",
            "ts":   time.time(),
            "trigger": "kv_spilled_to_t2",
        }
        self._do_kv_grow("kv_spilled_to_slow_t2", decision)
        self._record_decision(decision)

    # ── Loop E — VRAM pressure → tier demotion ───────────────────────────────

    def _on_vram_pressure_high(self, value: float) -> None:
        """Fires when fb_used_pct EMA crosses 87% for _VRAM_CONFIRM polls."""
        self._vram_pressure = True
        decision = {
            "loop": "E_vram_pressure",
            "ts":   time.time(),
            "fb_used_pct_ema": round(float(value), 1),
            "trigger": "pressure_high",
        }
        log.info(
            "[gb_orchestrator] Loop E: VRAM %.1f%% > %.0f%% — triggering tier demotion",
            value, _VRAM_DEMOTE_ENTER,
        )
        if self._tier_manager is not None and self._last_metrics is not None:
            try:
                self._tier_manager.auto_evict(self._last_metrics)
                decision["action"] = "tier_auto_evict"
            except Exception as exc:
                decision["action"] = f"tier_auto_evict_failed: {exc}"
                log.warning("[gb_orchestrator] Loop E: auto_evict error: %s", exc)
        else:
            decision["action"] = "no_tier_manager"
        self._record_decision(decision)

    def _on_vram_pressure_ok(self, value: float) -> None:
        self._vram_pressure = False
        log.info(
            "[gb_orchestrator] Loop E: VRAM pressure cleared (%.1f%% < %.0f%%)",
            value, _VRAM_DEMOTE_EXIT,
        )

    # ── Loop F — DCGM health proactive eviction ───────────────────────────────

    def _on_health_change(self, new: Any, old: Any) -> None:
        """Fires when DCGM health_ok transitions (confirmed over 2 polls)."""
        if new:
            log.info("[gb_orchestrator] Loop F: DCGM health restored")
            return
        # Health degraded
        decision = {
            "loop": "F_health_degraded",
            "ts": time.time(),
            "health_evict_armed": self._health_evict_armed,
        }
        if self._health_evict_armed:
            log.warning(
                "[gb_orchestrator] Loop F: DCGM health_ok=False, "
                "GB_ORCH_HEALTH_EVICT=1 — proactive tier eviction",
            )
            if self._tier_manager is not None and self._last_metrics is not None:
                try:
                    self._tier_manager.auto_evict(self._last_metrics)
                    decision["action"] = "tier_auto_evict"
                except Exception as exc:
                    decision["action"] = f"tier_auto_evict_failed: {exc}"
                    log.warning("[gb_orchestrator] Loop F: auto_evict error: %s", exc)
            else:
                decision["action"] = "no_tier_manager"
        else:
            decision["action"] = "advisory_only"
            log.warning(
                "[gb_orchestrator] Loop F: DCGM health_ok=False (advisory — "
                "set GB_ORCH_HEALTH_EVICT=1 to enable proactive eviction)",
            )
        self._record_decision(decision)

    def _do_kv_grow(self, reason: str, decision: dict) -> None:
        """Common KV grow logic, guarded by t1_budget accountant."""
        if self._ctrl is None:
            return
        old_kv = self._ctrl._last.get("kv_reserve_mb", (0, 0))[0] or 0

        # Loop I: mild-throttle adaptive step — halve KV grow when SM clock is
        # in the 8–12% drop zone (pre-throttle warning). Full step resumes when
        # the clock recovers above _SM_CLOCK_DROP_MILD.
        step_mb = self._kv_step_mb
        drop_ema = float(self._clock_drop_sig.get() or 0.0)
        if drop_ema >= _SM_CLOCK_DROP_MILD:
            step_mb = max(step_mb // 2, 128)
            log.debug(
                "[gb_orchestrator] Loop I: SM clock mild throttle (drop=%.1f%%) — "
                "halving KV step %d→%d MiB",
                drop_ema, self._kv_step_mb, step_mb,
            )
        new_kv = old_kv + step_mb

        # T1 budget guard: don't jointly over-commit T1 with Loop B
        headroom = self._t1_headroom_mb()
        if headroom <= 0:
            # No T1 headroom at all — skip grow entirely
            decision["action"] = "kv_grow_skipped_no_headroom"
            return
        if step_mb > headroom:
            clamped = headroom
            log.info(
                "[gb_orchestrator] Loop C: T1 budget clamped KV grow %d→%d MiB "
                "(headroom=%d MiB)",
                step_mb, clamped, headroom,
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
        return self._total_vram_mb - (kv or 0) - ws - self._weights_floor_mb

    # ── Loop D — thermal governor ─────────────────────────────────────────────

    def _on_temp_high(self, temp: float) -> None:
        self.thermal_stress = True
        decision = {
            "loop": "D_thermal",
            "ts":   time.time(),
            "temp_c_ema": round(float(temp), 1),
            "action": "advisory_safety_raise",
        }
        log.info(
            "[gb_orchestrator] Loop D: temp=%.1f°C (EMA) — thermal_stress=True, "
            "advisory safety_reserve raise, Loop C KV grow suppressed",
            temp,
        )
        if self._ctrl is not None:
            old = self._ctrl._last.get("safety_reserve_gb", (1, 0))[0] or 1
            r = self._ctrl.set_safety_reserve_gb(
                min(self._safety_max_gb, (old or 1) + _SAFETY_STEP_GB),
                reason=f"thermal_governor_temp={temp:.0f}",
            )
            decision["safety_result"] = r.reason
        self._record_decision(decision)
        # Loop P: step GPU power limit down — trade clock for sustained throughput
        self._step_gpu_power_limit(down=True, reason=f"thermal_stress_temp={temp:.0f}")

    def _on_temp_ok(self, temp: float) -> None:
        self.thermal_stress = False
        log.info(
            "[gb_orchestrator] Loop D: temp=%.1f°C — thermal OK, thermal_stress=False, "
            "Loop C KV grow re-enabled", temp
        )
        # Loop P: step GPU power limit back up now that thermal stress cleared
        self._step_gpu_power_limit(down=False, reason="thermal_stress_cleared")

    # ── Loop G — memory-bandwidth stress gate ────────────────────────────────

    def _on_mem_bw_high(self, util_pct: float) -> None:
        self.mem_bw_stress = True
        log.info(
            "[gb_orchestrator] Loop G: mem_copy_util=%.1f%% (EMA) — mem_bw_stress=True, "
            "Loop C KV grow suppressed (HBM bandwidth saturated)",
            util_pct,
        )

    def _on_mem_bw_ok(self, util_pct: float) -> None:
        self.mem_bw_stress = False
        log.info(
            "[gb_orchestrator] Loop G: mem_copy_util=%.1f%% — mem_bw_stress=False, "
            "Loop C KV grow re-enabled",
            util_pct,
        )

    # ── Loop H — ECC SBE early-warning ───────────────────────────────────────

    def _on_sbe_change(self, new: int, old: Any) -> None:
        """Fires when ecc_sbe_volatile count increases.

        SBEs are correctable — no lever moves. We set sbe_elevated for
        operator visibility and log a warning so journal captures the trend.
        Two consecutive polls with rising SBE count required (confirm=2).
        """
        new_sbes = int(new) - self._sbe_seen
        if new_sbes <= 0:
            return
        self._sbe_seen = int(new)
        self.sbe_elevated = True
        decision = {
            "loop":     "H_sbe_monitor",
            "ts":       time.time(),
            "ecc_sbe":  int(new),
            "new_sbes": new_sbes,
            "action":   "advisory_sbe_warning",
        }
        log.warning(
            "[gb_orchestrator] Loop H: ECC SBE +%d (total=%d) — "
            "single-bit errors are correctable but indicate memory degradation. "
            "Monitor for DBE escalation (Loop A will fire if uncorrectable errors appear).",
            new_sbes, int(new),
        )
        self._record_decision(decision)

    # ── Loop I — SM clock throttle ───────────────────────────────────────────

    def _on_clock_throttled(self, drop_pct: float) -> None:
        """Fires when SM clock drops > _SM_CLOCK_DROP_ENTER% from observed max."""
        self.clock_throttled = True
        decision = {
            "loop":               "I_clock_throttle",
            "ts":                 time.time(),
            "sm_clock_drop_pct":  round(drop_pct, 1),
            "sm_clock_max_mhz":   self._sm_clock_max,
            "action":             "clock_throttled_gates_kv_grow",
        }
        log.warning(
            "[gb_orchestrator] Loop I: SM clock dropped %.1f%% from max %d MHz — "
            "clock_throttled=True, Loop C KV grow suppressed (inference throughput impacted)",
            drop_pct, self._sm_clock_max,
        )
        self._record_decision(decision)
        # Loop P: step GPU power limit down — clock drop is the measured
        # effect of throttle; relieving power pressure helps it recover.
        self._step_gpu_power_limit(down=True, reason=f"clock_throttled_drop={drop_pct:.0f}pct")

    def _on_clock_ok(self, drop_pct: float) -> None:
        """Fires when SM clock recovers above _SM_CLOCK_DROP_EXIT% drop threshold."""
        self.clock_throttled = False
        log.info(
            "[gb_orchestrator] Loop I: SM clock recovered (drop=%.1f%%) — "
            "clock_throttled=False, Loop C KV grow re-enabled",
            drop_pct,
        )
        # Loop P: step GPU power limit back up now that throttle cleared
        self._step_gpu_power_limit(down=False, reason="clock_throttled_cleared")
        # Loop K: post-throttle KV restore.
        # The kv_pressure Signal on_enter already fired while throttle was active,
        # but _on_kv_pressure_high returned early due to clock_throttled=True.
        # Signal won't re-fire on_enter until pressure exits and re-enters the band.
        # If pressure is still in hysteresis, trigger one deferred grow now.
        if (self._kv_pressure_sig.in_hysteresis
                and not self.ecc_degraded
                and self._last_metrics is not None
                and self._last_metrics.shim_phase not in ("MODEL_LOAD", "IDLE", "DEEP_IDLE")):
            kv_val = float(self._kv_pressure_sig.get() or 0.0)
            log.info(
                "[gb_orchestrator] Loop K: clock recovered, kv_pressure=%.2f still high "
                "— deferred KV grow", kv_val,
            )
            decision = {
                "loop":             "K_post_throttle_kv_restore",
                "ts":               time.time(),
                "sm_clock_drop_pct": round(drop_pct, 1),
                "kv_pressure_ema":  round(kv_val, 3),
                "trigger":          "clock_ok_pressure_still_high",
            }
            self._do_kv_grow("post_throttle_restore", decision)
            self._record_decision(decision)

    # ── Loop J — phase-transition KV reclaim (pairwise operator) ─────────────

    def _on_phase_transition(self, pair: tuple, _unused: Any) -> None:
        """
        Receives (previous_phase, current_phase) from the pairwise() subscriber
        on _phase_sig (rxRust pairwise operator pattern).

        Two actions:
        - Loop J: INFERENCE→IDLE/DEEP_IDLE → reclaim KV reserve to floor.
        - Loop N: any phase change → adapt telemetry poll rate for the new phase.
        """
        prev_phase, curr_phase = pair

        # Loop J: KV reclaim on inference→idle transition
        if (prev_phase == "INFERENCE"
                and curr_phase in ("IDLE", "DEEP_IDLE")
                and self._ctrl is not None
                and not self.ecc_degraded):
            self._reclaim_kv_for_idle()

        # Loop N: adaptive poll rate — speed up during token generation,
        # slow down during idle to save CPU between requests.
        self._adapt_poll_rate(curr_phase)

        # Loop O: continuous OS performance envelope (supervisor + GB_OS_TUNE only)
        self._apply_performance_envelope(curr_phase)

    def _adapt_poll_rate(self, phase: str) -> None:
        """Loop N: adjust TelemetryManager poll interval based on shim phase."""
        if self._tel_manager is None:
            return
        if phase == "INFERENCE":
            target_ms = _POLL_MS_INFERENCE
        elif phase in ("IDLE", "DEEP_IDLE"):
            target_ms = _POLL_MS_IDLE
        else:
            target_ms = _POLL_MS_DEFAULT
        current_ms = getattr(self._tel_manager, "poll_ms", target_ms)
        if current_ms != target_ms:
            log.info(
                "[gb_orchestrator] Loop N: shim_phase=%s → poll_ms %d→%d",
                phase, current_ms, target_ms,
            )
            self._tel_manager.set_poll_interval_ms(target_ms)

    def _reclaim_kv_for_idle(self) -> None:
        """Loop J: reclaim KV reserve to floor on INFERENCE→IDLE transition."""
        if self._ctrl is None:
            return
        old_kv = self._ctrl._last.get("kv_reserve_mb", (0, 0))[0] or 0
        new_kv = self._kv_floor_mb
        if new_kv >= old_kv:
            log.debug(
                "[gb_orchestrator] Loop J: INFERENCE→IDLE — KV already at floor (%d MiB)",
                old_kv,
            )
            return
        log.info(
            "[gb_orchestrator] Loop J: INFERENCE→IDLE — reclaiming KV reserve %d→%d MiB",
            old_kv, new_kv,
        )
        r = self._ctrl.set_kv_reserve_mb(new_kv, reason="phase_idle_reclaim")
        decision = {
            "loop":      "J_phase_transition",
            "ts":        time.time(),
            "trigger":   "inference_to_idle",
            "action":    f"kv_reclaim_{old_kv}_{new_kv}_mb",
            "kv_result": r.reason,
        }
        self._record_decision(decision)

    # ── continuous OS tuning gate (Loops O-S) ─────────────────────────────────

    def _os_tune_active(self) -> bool:
        """True when this orchestrator is allowed to touch OS-level tunables.

        Supervisor mode only (process mode is unprivileged inside Ollama),
        opt-in via GB_OS_TUNE=1, and deferred entirely while gaming_mode is
        active (CLAUDE.md gaming-coexistence rule: the Gaming Suite owns
        governor/clocks then).
        """
        return self.os_tune_enabled and not self.gaming_mode and self._ctrl is not None

    # ── Loop O — continuous performance envelope ──────────────────────────────

    def _apply_performance_envelope(self, curr_phase: str) -> None:
        """
        Flip the whole-system performance envelope as inference enters/leaves
        the latency-sensitive phases. Entering MODEL_LOAD/INFERENCE/STEADY
        engages a low-jitter, max-throughput floor (governor, EPP, NUMA
        balancing, swappiness, GPU persistence/clock-lock/power). Entering
        IDLE/DEEP_IDLE restores everything to its pre-tune baseline.
        """
        if not self._os_tune_active():
            return
        decision = {"loop": "O_performance_envelope", "ts": time.time(), "phase": curr_phase}

        if curr_phase in _PERF_PHASES:
            self._ctrl.set_cpu_governor("performance", reason=f"phase={curr_phase}")
            self._ctrl.set_energy_perf_pref("performance", reason=f"phase={curr_phase}")
            self._ctrl.set_numa_balancing(False, reason=f"phase={curr_phase}")
            self._ctrl.set_swappiness(10, reason=f"phase={curr_phase}")
            self._ctrl.set_gpu_persistence(True, reason=f"phase={curr_phase}")
            if self._sm_clock_max > 0:
                self._ctrl.lock_gpu_clocks(
                    max(300, self._sm_clock_max // 4), self._sm_clock_max,
                    reason=f"phase={curr_phase}",
                )
            if self._last_metrics is not None and self._last_metrics.power_limit_w > 0:
                self._ctrl.set_gpu_power_limit(
                    int(self._last_metrics.power_limit_w), reason=f"phase={curr_phase}",
                )
            decision["action"] = "performance_envelope_engaged"
            log.info(
                "[gb_orchestrator] Loop O: phase=%s — performance envelope engaged "
                "(governor=performance, NUMA balancing off, GPU clocks locked)",
                curr_phase,
            )
        elif curr_phase in ("IDLE", "DEEP_IDLE"):
            restored = self._ctrl.restore_baseline()
            decision["action"] = "performance_envelope_restored"
            decision["restored"] = restored
            log.info(
                "[gb_orchestrator] Loop O: phase=%s — performance envelope restored "
                "to pre-tune baseline", curr_phase,
            )
        else:
            return
        self._record_decision(decision)

    # ── Loop P — thermal/throttle GPU power cap ───────────────────────────────

    def _step_gpu_power_limit(self, down: bool, reason: str) -> None:
        """
        Step the GPU power limit down under thermal/clock-throttle stress
        (trading peak clock for sustained, jitter-free throughput) and back
        up once the stress clears. Floored at _GPU_POWER_FLOOR_FRAC of TDP.
        """
        if not self._os_tune_active() or self._last_metrics is None:
            return
        tdp = self._last_metrics.power_limit_w
        if tdp <= 0:
            return
        old_w = self._ctrl._last.get("gpu_power_limit_w", (tdp, 0))[0] or tdp
        delta = -_GPU_POWER_STEP_W if down else _GPU_POWER_STEP_W
        new_w = max(int(tdp * _GPU_POWER_FLOOR_FRAC), min(int(tdp), int(old_w) + delta))
        r = self._ctrl.set_gpu_power_limit(new_w, reason=reason)
        self._record_decision({
            "loop": "P_thermal_power_cap", "ts": time.time(),
            "direction": "down" if down else "up",
            "power_limit_w": new_w, "reason": reason, "result": r.reason,
        })

    # ── Loop Q — host memory-pressure VM tune ─────────────────────────────────

    def _on_mem_psi_high(self, value: float) -> None:
        """PSI mem "some" sustained high — tasks are stalling on reclaim.
        Tighten vm.* so DMA-BUF pinning has pinnable pages on hand."""
        if not self._os_tune_active():
            return
        self._ctrl.set_watermark_scale_factor(150, reason="psi_mem_pressure")
        self._ctrl.set_swappiness(5, reason="psi_mem_pressure")
        self._ctrl.set_dirty_background_ratio(5, reason="psi_mem_pressure")
        log.warning(
            "[gb_orchestrator] Loop Q: PSI mem-some=%.1f%% — tightening vm.* "
            "(watermark_scale_factor, swappiness, dirty_background_ratio)", value,
        )
        self._record_decision({
            "loop": "Q_mem_pressure_vm", "ts": time.time(),
            "psi_mem_some_avg10": round(value, 1), "action": "vm_tightened",
        })

    def _on_mem_psi_ok(self, value: float) -> None:
        if not self._os_tune_active():
            return
        self._ctrl.set_watermark_scale_factor(10, reason="psi_mem_cleared")
        self._ctrl.set_swappiness(10, reason="psi_mem_cleared")
        self._ctrl.set_dirty_background_ratio(10, reason="psi_mem_cleared")
        log.info("[gb_orchestrator] Loop Q: PSI mem pressure cleared — vm.* relaxed")
        self._record_decision({
            "loop": "Q_mem_pressure_vm", "ts": time.time(),
            "psi_mem_some_avg10": round(value, 1), "action": "vm_relaxed",
        })

    # ── Loop R — host CPU/IO pressure assist ──────────────────────────────────

    def _on_cpu_psi_high(self, value: float) -> None:
        """PSI cpu "some" sustained high during a performance phase — the
        governor may have slipped (e.g. external load); reassert it."""
        if not self._os_tune_active():
            return
        phase = self._last_metrics.shim_phase if self._last_metrics else ""
        if phase not in _PERF_PHASES:
            return
        self._ctrl.set_cpu_governor("performance", reason="psi_cpu_pressure")
        log.info(
            "[gb_orchestrator] Loop R: PSI cpu-some=%.1f%% during phase=%s — "
            "reasserting governor=performance", value, phase,
        )
        self._record_decision({
            "loop": "R_cpu_io_pressure", "ts": time.time(),
            "psi_cpu_some_avg10": round(value, 1), "action": "governor_reasserted",
        })

    def _on_io_psi_high(self, value: float) -> None:
        """PSI io "some" sustained high while T3 NVMe is under pressure —
        advisory only (no NVMe-device-path lever exists yet in GbControl)."""
        if not self._os_tune_active():
            return
        t3_pressured = bool(self._last_metrics and self._last_metrics.gb
                             and self._last_metrics.gb.t3_pressure >= 2)
        if not t3_pressured:
            return
        log.warning(
            "[gb_orchestrator] Loop R: PSI io-some=%.1f%% during T3 spill — "
            "NVMe IO pressure detected (advisory; no reactive NVMe lever yet)", value,
        )
        self._record_decision({
            "loop": "R_cpu_io_pressure", "ts": time.time(),
            "psi_io_some_avg10": round(value, 1), "action": "advisory_only",
        })

    # ── Loop S — PCIe saturation → prefetch-throttle hint ─────────────────────

    def _on_pcie_saturation_change(self, new: bool, old: Any) -> None:
        if not self._os_tune_active():
            return
        r = self._ctrl.set_prefetch_throttle(bool(new), reason="pcie_saturated" if new else "pcie_saturated_cleared")
        log.info(
            "[gb_orchestrator] Loop S: PCIe saturated=%s — prefetch_throttle %s",
            new, "engaged" if new else "cleared",
        )
        self._record_decision({
            "loop": "S_pcie_saturation", "ts": time.time(),
            "action": "prefetch_throttle_engaged" if new else "prefetch_throttle_cleared",
            "result": r.reason,
        })

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
            self._vram_sig.dump(),
            self._health_sig.dump(),       # B4
            self._mem_bw_sig.dump(),       # G1
            self._sbe_sig.dump(),          # H1
            self._clock_drop_sig.dump(),   # I1
            self._mem_psi_sig.dump(),      # Q
            self._cpu_psi_sig.dump(),      # R
            self._io_psi_sig.dump(),       # R
            self._pcie_sat_sig.dump(),     # S
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
            elif s["name"] == "fb_used_pct":
                s["state"] = "pressure" if self._vram_pressure else "ok"
            elif s["name"] == "health_ok":
                s["state"] = "ok" if s.get("value", True) else "degraded"
            elif s["name"] == "mem_copy_util_pct":
                s["state"] = "saturated" if self.mem_bw_stress else "ok"
            elif s["name"] == "ecc_sbe":
                s["state"] = "elevated" if self.sbe_elevated else "ok"
            elif s["name"] == "sm_clock_drop_pct":
                s["state"] = "throttled" if self.clock_throttled else "ok"
            elif s["name"] in ("psi_mem_some", "psi_cpu_some", "psi_io_some", "pcie_saturated"):
                s["state"] = "high" if s.get("in_hysteresis") or s.get("value") else "ok"
            else:
                s["state"] = "ok"
        return {
            "mode":              self._mode,
            "actuate":           self._actuate,
            "os_tune_enabled":   self.os_tune_enabled,         # Loops O-S
            "gaming_mode":       self.gaming_mode,              # Loops O-S gate
            "ecc_degraded":      self.ecc_degraded,
            "thermal_stress":    self.thermal_stress,         # Loop D→C gate
            "mem_bw_stress":     self.mem_bw_stress,          # Loop G→C gate
            "sbe_elevated":      self.sbe_elevated,           # Loop H: SBE advisory
            "sbe_seen":          self._sbe_seen,              # Loop H: SBE count
            "clock_throttled":   self.clock_throttled,        # Loop I: SM clock gate
            "sm_clock_max_mhz":  self._sm_clock_max,          # Loop I: ratcheting max
            "shim_phase":        (self._last_metrics.shim_phase if self._last_metrics else ""),
            "last_phase":        str(self._phase_sig.get() or ""),  # Loop J: pairwise source
            "ecc_seen":          self._ecc_seen,
            "ws_above":          self._ws_above,
            "ws_reserve_mb":     self._ws_reserve_mb,
            "vram_pressure":     self._vram_pressure,
            "cluster_pressure":  self._cluster_pressure,      # B3
            "health_ok":         bool(self._health_sig.get()), # B4
            "health_evict_armed": self._health_evict_armed,   # B4
            "poll_ms":           getattr(self._tel_manager, "poll_ms", _POLL_MS_DEFAULT),  # Loop N
            "topology":          self._topology_summary(),
            "total_vram_mb":     self._total_vram_mb,
            "signals":           signals,
            "recent_decisions":  self._decisions[-8:],
            "control":           self._ctrl.dump() if self._ctrl else {},
        }

    def _topology_summary(self) -> dict:
        """Compact topology snapshot for dump() — reads last_metrics.topology."""
        if self._last_metrics is None or self._last_metrics.topology is None:
            return {}
        t = self._last_metrics.topology
        return {
            "bdf":               t.bdf,
            "numa_node":         t.numa_node,
            "compute_capability": f"{t.compute_capability[0]}.{t.compute_capability[1]}",
            "pcie_gen_current":  t.pcie_gen_current,
            "pcie_width_current": t.pcie_width_current,
            "pcie_gen_max":      t.pcie_gen_max,
            "pcie_width_max":    t.pcie_width_max,
            "pcie_bw_mb_s":      round(t.pcie_bw_mb_s),
            "pcie_saturated":    self._last_metrics.pcie_saturated,
            "pcie_degraded":     t.pcie_degraded,
            "nvlink_count":      t.nvlink_count,
            "nvlink_peers":      list(t.nvlink_peer_ids),
            "p2p_devices":       list(t.p2p_device_ids),
        }

    def stop(self) -> None:
        """
        Clean shutdown hook (called from gb_init._shutdown or supervisor).
        Best-effort: reset any GPU clock lock so a stopped supervisor never
        leaves clocks pinned (the persisted sysctl/governor floor survives
        intentionally — only the volatile GPU clock lock is undone here).
        """
        if self._ctrl is not None:
            try:
                self._ctrl.reset_gpu_clocks(reason="orchestrator_shutdown")
            except Exception:
                pass

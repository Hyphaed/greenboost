#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
#
# Tier-thrashing detection for GB-Tiering.
#
# Algorithm ported from NVIDIA open-gpu-kernel-modules, MIT (dual MIT/GPLv2
# when linked as a kernel module):
#   kernel-open/nvidia-uvm/uvm_perf_thrashing.c
#   upstream commit e4a5faa2, driver 610.57.04 (2026-08-03)
#   Copyright (c) NVIDIA CORPORATION & AFFILIATES.
#
# UVM's model, and the four parameters that define it, transfer directly:
#
#   lapse       two accesses to the same page closer together than this are
#               one "thrashing event" (upstream: 500 us, a page-fault
#               timescale)
#   threshold   events before the page is declared thrashing (upstream: 3)
#   pin_threshold  throttling events before the page is pinned (upstream: 10)
#   epoch       quiet time after which the counters reset
#               (upstream: 2000 x lapse)
#   max_resets  how many times counters may reset before giving up
#               (upstream: 4)
#
# What changes here is granularity and timescale, and only those. UVM watches
# a 4 KB page bouncing between processors; GB-Tiering watches a model
# component - a UNet, a VAE, a text encoder - bouncing between T1 and T2. The
# counting logic is the same; the clock is not.
#
# TIMESCALE, MEASURED. `lapse_s` is not a guess. It was calibrated against a
# real workload on this box (2026-08-21): FLUX.2-klein-9B (33 GB cache, ~17 GB
# resident) through ai-forge's `gen_image.py` with GB_ORCH=1, so
# DiffusionOrchestrator drove real encode/denoise/decode phases on an 11.26 GB
# RTX 5070 - concurrently with a desktop session and a browser playing video,
# which is the realistic condition, not a confound. Three runs, two quality
# tiers. Every number came from `tier_move` dataflux events.
#
# Legitimate demote->promote reversals, by gb-quant quality tier:
#
#   fp8             21.48 s, 22.85 s      (fast path: no per-layer calibration)
#   near_lossless  272.97 s, 278.72 s     (slow path: 252+147 layer calibration)
#   both, later    264.41 s, 268.90 s     (denoise->decode handoff)
#
# The 12x gap between quality tiers is the whole point. An earlier value of
# 30 s - chosen by reasoning rather than measurement - sat ABOVE fp8's
# legitimate 21-23 s bounce, so a healthy fp8 pipeline scored a thrash event
# on every run. It escaped declaring thrashing only because `threshold` is 3;
# a 3-image fp8 batch would have falsely declared it. That is precisely the
# error a fixture cannot catch, because a fixture uses whatever interval its
# author already believed.
#
# 10 s sits 2.1x below the fastest observed legitimate reversal (21.48 s) and
# still catches genuine sub-10 s ping-pong. False positives are the worse
# failure for this detector: one that cries thrashing on healthy pipelines
# gets ignored, and is then worth nothing when something is actually wrong.
#
# What this still does NOT establish: whether real thrashing occurs in the
# 10-21 s band now excluded. Six reversals from one pipeline family cannot say.
# `stats()["params"]["calibration"]` carries the provenance so a reader can
# weigh it, and raw intervals keep accumulating for a better answer.
"""Detects a model component ping-ponging between tiers faster than it is used.

Observation only. Nothing here evicts, promotes, pins, or changes placement -
it counts, and it reports. Wiring its verdict into a decision is a separate,
later change that should not happen until the numbers below have been looked
at against a real workload.

The symptom this exists to name is already visible in GreenBoost's governed
layer as `rule1_underfilled`: VRAM sitting free while T2 spills. That is what
thrashing looks like from the outside; this is the thing that can say so.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Upstream names, GB-Tiering timescale.
LAPSE_S_DEFAULT = 10.0        # measured: 2.1x below fastest legitimate reversal
THRESHOLD_DEFAULT = 3         # upstream UVM_PERF_THRASHING_THRESHOLD_DEFAULT
PIN_THRESHOLD_DEFAULT = 10    # upstream UVM_PERF_THRASHING_PIN_THRESHOLD_DEFAULT
EPOCH_MULTIPLIER = 20         # upstream uses 2000x lapse; see _epoch_s note
MAX_RESETS_DEFAULT = 4        # upstream UVM_PERF_THRASHING_MAX_RESETS_DEFAULT

# Ring of recent intervals kept per component, for calibrating lapse_s later.
_INTERVAL_HISTORY = 32


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class ComponentThrashState:
    """Per-component counters. Mirrors UVM's page_thrashing_info_t."""

    name: str
    num_thrashing_events: int = 0
    num_resets: int = 0
    last_move_time: float = 0.0
    last_from_tier: Optional[str] = None
    last_to_tier: Optional[str] = None
    total_moves: int = 0
    #  Raw inter-move intervals, newest last. Kept so lapse_s can be replaced
    #  with a measured value instead of the provisional one.
    intervals: List[float] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "num_thrashing_events": self.num_thrashing_events,
            "num_resets": self.num_resets,
            "total_moves": self.total_moves,
            "last_from_tier": self.last_from_tier,
            "last_to_tier": self.last_to_tier,
            "median_interval_s": self.median_interval(),
            "min_interval_s": min(self.intervals) if self.intervals else None,
        }

    def median_interval(self) -> Optional[float]:
        if not self.intervals:
            return None
        ordered = sorted(self.intervals)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0


class ThrashDetector:
    """Counts tier ping-pong per component, using UVM's event/threshold model.

    Parameters keep upstream's names so the two can be compared. All are
    overridable by environment variable so a different box or workload can be
    re-calibrated without editing source , NOT because the default is a guess.
    This docstring used to say the timescale was provisional; that was left
    over from before the calibration described in the module header above,
    which measured legitimate fp8 reversals at 21.48/22.85 s and near_lossless
    at 272.97/278.72 s and set lapse_s to 10 s, 2.1x below the fastest of them.
    Re-confirmed 2026-08-22 from the tier_move history (21.5/22.8/278.7 s):

        GB_THRASH_LAPSE_S, GB_THRASH_THRESHOLD,
        GB_THRASH_PIN_THRESHOLD, GB_THRASH_MAX_RESETS
    """

    def __init__(
        self,
        lapse_s: Optional[float] = None,
        threshold: Optional[int] = None,
        pin_threshold: Optional[int] = None,
        max_resets: Optional[int] = None,
    ):
        self.lapse_s = lapse_s if lapse_s is not None else _env_float("GB_THRASH_LAPSE_S", LAPSE_S_DEFAULT)
        self.threshold = threshold if threshold is not None else _env_int("GB_THRASH_THRESHOLD", THRESHOLD_DEFAULT)
        self.pin_threshold = (
            pin_threshold if pin_threshold is not None
            else _env_int("GB_THRASH_PIN_THRESHOLD", PIN_THRESHOLD_DEFAULT)
        )
        self.max_resets = (
            max_resets if max_resets is not None
            else _env_int("GB_THRASH_MAX_RESETS", MAX_RESETS_DEFAULT)
        )
        self._state: Dict[str, ComponentThrashState] = {}

    @property
    def epoch_s(self) -> float:
        """Quiet time after which counters reset.

        Upstream uses 2000 x lapse, which at a 500 us lapse is one second. At
        this module's 10 s lapse the same multiplier would be a ~5.5 hour
        epoch: a counter that never resets inside a session, which defeats the
        point. The multiplier is rescaled to 20x, giving ~200 s.

        200 s is deliberately shorter than the measured legitimate reversal
        intervals (264-279 s on the slow path), so a pipeline whose phases are
        that far apart has its counter reset between them rather than slowly
        accumulating across an entire session. Genuine thrashing is sub-10 s
        ping-pong, whose bouts are far closer together than the epoch, so it
        survives the reset. This rescale is the one place the port departs
        from upstream's arithmetic rather than only its units.
        """
        return self.lapse_s * EPOCH_MULTIPLIER

    def record_move(self, name: str, from_tier: str, to_tier: str,
                    now: Optional[float] = None) -> ComponentThrashState:
        """Record one tier transition. Returns that component's live state.

        A move counts as a thrashing event when it happens within `lapse_s` of
        the previous move AND reverses it (T1->T2 then T2->T1). A one-way
        migration, however fast, is not thrashing - it is just a migration.
        That reversal check is the GB-Tiering equivalent of UVM requiring the
        two accesses to come from different processors.
        """
        now = time.time() if now is None else now
        st = self._state.get(name)
        if st is None:
            st = ComponentThrashState(name=name, first_seen=now)
            self._state[name] = st

        st.total_moves += 1

        if st.last_move_time:
            interval = now - st.last_move_time
            st.intervals.append(interval)
            if len(st.intervals) > _INTERVAL_HISTORY:
                del st.intervals[0]

            if interval > self.epoch_s:
                # Quiet for a whole epoch: forget what came before.
                if st.num_thrashing_events:
                    st.num_resets += 1
                st.num_thrashing_events = 0
            elif interval <= self.lapse_s and self._reverses(st, from_tier, to_tier):
                if st.num_resets < self.max_resets:
                    st.num_thrashing_events += 1

        st.last_move_time = now
        st.last_from_tier = from_tier
        st.last_to_tier = to_tier
        return st

    @staticmethod
    def _reverses(st: ComponentThrashState, from_tier: str, to_tier: str) -> bool:
        """True when this move undoes the previous one."""
        return (
            st.last_from_tier is not None
            and st.last_to_tier == from_tier
            and st.last_from_tier == to_tier
        )

    def is_thrashing(self, name: str) -> bool:
        st = self._state.get(name)
        return bool(st and st.num_thrashing_events >= self.threshold)

    def should_pin(self, name: str) -> bool:
        """Whether UVM would have pinned this by now.

        Advisory. Nothing acts on it yet. When something does, the natural
        actuator is gb_evict_policy priority - raising a thrashing component's
        priority is exactly 'stop choosing it as the victim'.
        """
        st = self._state.get(name)
        return bool(st and st.num_thrashing_events >= self.pin_threshold)

    def thrashing_components(self) -> List[str]:
        return [n for n in self._state if self.is_thrashing(n)]

    def stats(self) -> dict:
        """Everything the metric layer needs, in one read."""
        thrashing = self.thrashing_components()
        all_intervals = [i for st in self._state.values() for i in st.intervals]
        return {
            "tracked_components": len(self._state),
            "thrashing_count": len(thrashing),
            "thrashing_components": thrashing,
            "pin_recommended": [n for n in self._state if self.should_pin(n)],
            "total_moves": sum(st.total_moves for st in self._state.values()),
            "min_interval_s": min(all_intervals) if all_intervals else None,
            "params": {
                "lapse_s": self.lapse_s,
                "threshold": self.threshold,
                "pin_threshold": self.pin_threshold,
                "epoch_s": self.epoch_s,
                "max_resets": self.max_resets,
                "lapse_calibrated": True,
                "calibration": {
                    "measured_on": "2026-08-21",
                    "workload": "FLUX.2-klein-9B via ai-forge gen_image.py, GB_ORCH=1",
                    "gpu": "RTX 5070, 11264 MB T1",
                    "concurrent_load": "desktop + browser video (~598 MB VRAM, GPU time-sliced)",
                    "runs": 3,
                    "legitimate_reversal_s": {
                        "fp8": [21.48, 22.85],
                        "near_lossless": [272.97, 278.72],
                        "late_phase": [264.41, 268.90],
                    },
                    "fastest_legitimate_s": 21.48,
                    "margin_factor": 2.1,
                    "caveat": ("Six reversals from one pipeline family. Establishes that "
                               "10 s clears the fastest legitimate bounce (21.48 s); does "
                               "NOT establish whether real thrashing occurs in the 10-21 s "
                               "band this window now excludes."),
                },
            },
            "per_component": {n: st.as_dict() for n, st in self._state.items()},
        }

    def reset(self) -> None:
        self._state.clear()

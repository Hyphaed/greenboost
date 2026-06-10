#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
PR-WW: GreenBoost long-running stability monitor.

Polls /run/greenboost/shim_stats and /run/greenboost/metrics.json at a
configurable interval and watches for invariants that should hold over
a multi-hour inference workload:

  * Monotonic counters (kernel_dispatch_count, h2d_mb, d2h_mb,
    tier_*_lifetime_mb, etc.) must never decrease except when the shim
    PID changes - a decrease means either a wraparound bug or a stat
    reset that was forgotten.

  * Gauges that should converge back near zero when the model is unloaded
    (tier_*_cur_mb, kv_t1_tracked_mb) must drop below a baseline within
    a configurable cool-down window after the phase transitions away
    from INFERENCE - if they stay high we have a leak.

  * Pool fragmentation (t2_pool_frag_pct, kv_internal_frag_mb) must not
    trend upward indefinitely across the run; we flag a sustained climb.

  * Shim staleness: timestamp in shim_stats must update at least every
    GB_STALE_S seconds when the daemon is healthy.

Exit codes:
  0  clean run (no violations) - only meaningful when --duration is set
  1  at least one invariant violation observed (only set with --strict)
  2  argparse / setup error

Run with:
  python3 gb_stability_monitor.py --interval 30 --duration 86400 \
      --log /var/log/greenboost-stability.log
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

DEFAULT_SHIM_STATS = "/run/greenboost/shim_stats"
DEFAULT_METRICS_JSON = "/run/greenboost/metrics.json"

# Stats expected to be monotone non-decreasing while shim PID is stable.
# Each key is the field name in shim_stats (key=value format).
MONOTONE_FIELDS = (
    "path_a_count",
    "path_b_count",
    "kernel_dispatch_count",
    "remote_alloc_count",
    "remote_alloc_mb",
    "h2d_mb",
    "d2h_mb",
    "cold_epoch_evict_count",
    "kv_dedup_hits",
)

# Gauges that should drop below GAUGE_RELEASE_FRACTION × peak once the
# shim's reported phase transitions away from INFERENCE / STEADY for
# at least IDLE_COOLDOWN_S seconds.  Otherwise we suspect a tier leak.
GAUGE_LEAK_CHECK = (
    "kv_t1_tracked_mb",
    "tier_t1_local_cur_mb",
    "tier_t1_feeder_cur_mb",
    "tier_t2_local_cur_mb",
    "tier_t2_feeder_cur_mb",
)
GAUGE_RELEASE_FRACTION = 0.10   # gauge should fall ≥ 90 % below peak
IDLE_COOLDOWN_S = 120           # wait 2 min after phase leaves INFERENCE

# Trend detection - a gauge climbing for at least TREND_WINDOW_S
# consecutive samples without ever falling below its starting value
# counts as drift.
TREND_GAUGES = ("t2_pool_frag_pct", "kv_internal_frag_mb")
TREND_WINDOW = 30   # samples - at 30s interval = 15 min sustained climb
TREND_MIN_DELTA = 5  # minimum percentage-point / MB rise to flag

# Shim staleness threshold - if `timestamp` doesn't move for this many
# seconds we treat the shim as wedged.  The shim writes shim_stats on
# every cudaMalloc/Memcpy so during inference the timestamp lags rarely.
SHIM_STALE_S = 90


# ---- shim_stats parser -----------------------------------------------

def parse_shim_stats(path: str) -> dict[str, str] | None:
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return None
    except OSError as e:
        return {"_read_error": str(e)}
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def as_int(d: dict[str, str], k: str, default: int = 0) -> int:
    try:
        return int(d.get(k, default))
    except (TypeError, ValueError):
        return default


# ---- monitor state ---------------------------------------------------

class Violation(Exception):
    """Single invariant violation; carries a structured payload."""
    def __init__(self, kind: str, field: str, detail: str):
        super().__init__(f"{kind}: {field} - {detail}")
        self.kind = kind
        self.field = field
        self.detail = detail


class Monitor:
    def __init__(self, shim_path: str, metrics_path: str,
                 log_fp, json_out: bool):
        self.shim_path = shim_path
        self.metrics_path = metrics_path
        self.log_fp = log_fp
        self.json_out = json_out
        self.last_pid: int | None = None
        self.last_mono: dict[str, int] = {}
        self.last_ts: int = 0
        self.last_ts_wall: float = time.time()
        # Per-gauge peak tracking + sample window for trend detection.
        self.gauge_peak: dict[str, int] = {}
        self.gauge_window: dict[str, deque[int]] = {
            g: deque(maxlen=TREND_WINDOW) for g in TREND_GAUGES
        }
        self.phase_inference_last_seen: float = 0.0
        self.violations: list[Violation] = []
        self.samples = 0

    # ---- logging -----

    def _emit(self, level: str, msg: str, extra: dict | None = None) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if self.json_out:
            rec = {"ts": ts, "level": level, "msg": msg}
            if extra: rec.update(extra)
            self.log_fp.write(json.dumps(rec) + "\n")
        else:
            tag = f"[{level}]".ljust(7)
            self.log_fp.write(f"{ts} {tag} {msg}\n")
        self.log_fp.flush()

    # ---- per-sample checks -----

    def check_pid(self, d: dict[str, str]) -> None:
        pid = as_int(d, "pid")
        if pid == 0:
            return
        if self.last_pid is None:
            self.last_pid = pid
            return
        if pid != self.last_pid:
            self._emit("WARN", "shim PID changed - resetting monotone baselines",
                       {"old_pid": self.last_pid, "new_pid": pid})
            self.last_pid = pid
            self.last_mono.clear()
            self.gauge_peak.clear()
            for w in self.gauge_window.values():
                w.clear()

    def check_monotone(self, d: dict[str, str]) -> None:
        for f in MONOTONE_FIELDS:
            cur = as_int(d, f)
            prev = self.last_mono.get(f)
            if prev is not None and cur < prev:
                v = Violation("MONOTONE_REGRESSED", f,
                              f"value went {prev} → {cur}")
                self.violations.append(v)
                self._emit("ALERT", str(v))
            self.last_mono[f] = cur

    def check_staleness(self, d: dict[str, str]) -> None:
        ts = as_int(d, "timestamp")
        now = time.time()
        if ts == 0:
            return
        if ts != self.last_ts:
            self.last_ts = ts
            self.last_ts_wall = now
            return
        if now - self.last_ts_wall > SHIM_STALE_S:
            v = Violation("SHIM_STALE", "timestamp",
                          f"shim_stats timestamp frozen for "
                          f"{int(now - self.last_ts_wall)}s "
                          f"(> {SHIM_STALE_S}s threshold)")
            self.violations.append(v)
            self._emit("ALERT", str(v))
            self.last_ts_wall = now  # reset to avoid spam

    def check_gauge_leak(self, d: dict[str, str]) -> None:
        phase = d.get("phase", "")
        now = time.time()
        if phase in ("INFERENCE", "STEADY", "MODEL_LOAD"):
            self.phase_inference_last_seen = now
            for g in GAUGE_LEAK_CHECK:
                cur = as_int(d, g)
                if cur > self.gauge_peak.get(g, 0):
                    self.gauge_peak[g] = cur
            return
        # Phase is IDLE / DEEP_IDLE / INIT - check leak after cooldown.
        if self.phase_inference_last_seen == 0:
            return  # never seen inference yet
        if now - self.phase_inference_last_seen < IDLE_COOLDOWN_S:
            return
        for g in GAUGE_LEAK_CHECK:
            peak = self.gauge_peak.get(g, 0)
            if peak < 32:  # ignore tiny peaks (< 32 MB / blocks) - noise
                continue
            cur = as_int(d, g)
            limit = max(int(peak * GAUGE_RELEASE_FRACTION), 16)
            if cur > limit:
                v = Violation("GAUGE_LEAK", g,
                              f"phase=IDLE for "
                              f"{int(now - self.phase_inference_last_seen)}s "
                              f"but gauge stuck at {cur} (peak={peak}, "
                              f"expected ≤ {limit})")
                self.violations.append(v)
                self._emit("ALERT", str(v))
                # Reset peak so we don't keep firing on the same leak
                self.gauge_peak[g] = cur

    def check_trend(self, d: dict[str, str]) -> None:
        for g in TREND_GAUGES:
            v = as_int(d, g)
            w = self.gauge_window[g]
            w.append(v)
            if len(w) < TREND_WINDOW:
                continue
            # Climbing monotonically over the window AND total rise meets
            # the noise floor.
            if w[-1] - w[0] < TREND_MIN_DELTA:
                continue
            climbing = all(w[i] <= w[i + 1] for i in range(len(w) - 1))
            if climbing:
                vi = Violation("DRIFT", g,
                               f"value climbed {w[0]} → {w[-1]} over "
                               f"{TREND_WINDOW} samples without retreat")
                self.violations.append(vi)
                self._emit("ALERT", str(vi))
                # Clear the window so the next alert needs a fresh climb.
                w.clear()

    # ---- main loop -----

    def tick(self) -> None:
        d = parse_shim_stats(self.shim_path)
        if d is None:
            self._emit("WARN", f"{self.shim_path} not found - shim down?")
            return
        if "_read_error" in d:
            self._emit("WARN", f"read error: {d['_read_error']}")
            return
        self.samples += 1
        self.check_pid(d)
        self.check_monotone(d)
        self.check_staleness(d)
        self.check_gauge_leak(d)
        self.check_trend(d)

    def summary(self) -> str:
        return (f"samples={self.samples} violations={len(self.violations)} "
                f"kinds={sorted({v.kind for v in self.violations})}")


# ---- driver ----------------------------------------------------------

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--interval", type=int, default=30,
                    help="seconds between samples (default 30)")
    ap.add_argument("--duration", type=int, default=0,
                    help="run for N seconds then exit (0 = forever)")
    ap.add_argument("--shim-stats", default=DEFAULT_SHIM_STATS,
                    help=f"path to shim_stats (default {DEFAULT_SHIM_STATS})")
    ap.add_argument("--metrics-json", default=DEFAULT_METRICS_JSON,
                    help="path to metrics.json")
    ap.add_argument("--log", default="-",
                    help="output file ('-' = stdout)")
    ap.add_argument("--json", action="store_true",
                    help="emit one JSON record per line instead of plain text")
    ap.add_argument("--strict", action="store_true",
                    help="exit code 1 on first violation")
    args = ap.parse_args(argv)

    if args.interval < 1:
        print("--interval must be ≥ 1", file=sys.stderr)
        return 2

    if args.log == "-":
        log_fp = sys.stdout
    else:
        try:
            log_fp = open(args.log, "a", buffering=1)
        except OSError as e:
            print(f"cannot open --log {args.log}: {e}", file=sys.stderr)
            return 2

    mon = Monitor(args.shim_stats, args.metrics_json, log_fp, args.json)
    mon._emit("INFO", f"gb_stability_monitor starting "
              f"(interval={args.interval}s, duration={args.duration or 'forever'})")

    stop = {"flag": False}
    def _sig(*_):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    start = time.time()
    next_tick = start
    while not stop["flag"]:
        if args.duration and (time.time() - start) >= args.duration:
            break
        mon.tick()
        if args.strict and mon.violations:
            mon._emit("INFO", "--strict: exiting on first violation")
            log_fp.flush()
            return 1
        next_tick += args.interval
        sleep_for = next_tick - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            # Fell behind - skip the catch-up burst.
            next_tick = time.time()

    mon._emit("INFO", "stopping: " + mon.summary())
    log_fp.flush()
    return 1 if (args.strict and mon.violations) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

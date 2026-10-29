#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""gb_damon.py — host-side cold-page evidence from the kernel's DAMON_STAT.

The GPU half of GreenBoost's placement story is getting a measured signal
(GPU access counters, crosswalk P2-01). This is the host half: the kernel is
already sampling which pages are actually touched, for free, and nothing was
reading it.

WHERE THE DATA LIVES. Not `/sys/kernel/mm/damon/stat/` — that path does not
exist. `CONFIG_DAMON_STAT` exposes itself as module parameters:

    /sys/module/damon_stat/parameters/enabled
    /sys/module/damon_stat/parameters/memory_idle_ms_percentiles
    /sys/module/damon_stat/parameters/estimated_memory_bandwidth
    /sys/module/damon_stat/parameters/aggr_interval_us

All are mode 0400, root-only. That is why the reader lives here and is driven
by gb_supervisor (which runs as root) rather than by gb_tiering: the supervisor
reads and republishes to a world-readable file, and unprivileged consumers read
that. Routing around the permission would have meant either running the tiering
layer as root or shelling out to sudo, both worse.

READING THE PERCENTILES. `memory_idle_ms_percentiles` is 101 comma-separated
values, P0 through P100, of per-region idle time in milliseconds.

  * A NEGATIVE value means the region is being actively accessed, and the
    magnitude is how long it has been hot. It is not idle time at all.
  * A POSITIVE value is genuine idle time.

Measured on this box 2026-08-21 (61 GiB host, mid-session):

    P0            -9400000 ms   hottest region, hot for ~2.6 h
    P1..P20        -200000 ms   hot for ~200 s
    P21..P77       +200000 ms   idle ~200 s
    P78..P79       +400000 ms
    P80..P100      +800000 ms   idle ~800 s
    estimated_memory_bandwidth = 85996769 B/s

So ~21% of host memory was actively accessed and ~79% idle, with a fifth of it
idle for over 13 minutes.

WHAT THIS MODULE DELIBERATELY DOES NOT DO. It does not define "cold". There is
no single correct idle threshold — it depends on what the caller intends to do
with the answer, and inventing one here would bury a judgment call inside a
parser. `cold_pct_above(ms)` lets the caller draw the line and makes the choice
visible at the call site. The published summary reports the distribution.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

DAMON_STAT_DIR = Path("/sys/module/damon_stat/parameters")

_P_IDLE = "memory_idle_ms_percentiles"
_P_BW = "estimated_memory_bandwidth"
_P_ENABLED = "enabled"
_P_AGGR = "aggr_interval_us"


def available() -> bool:
    """Whether DAMON_STAT is present at all. Says nothing about readability."""
    return (DAMON_STAT_DIR / _P_IDLE).exists()


def readable() -> bool:
    """Whether this process can actually read the parameters (needs root)."""
    try:
        (DAMON_STAT_DIR / _P_IDLE).read_text()
        return True
    except OSError:
        return False


def _read_param(name: str) -> Optional[str]:
    try:
        return (DAMON_STAT_DIR / name).read_text().strip()
    except OSError:
        return None


def parse_idle_percentiles(raw: str) -> List[int]:
    """Parse the comma-separated P0..P100 idle-ms array.

    Kept separate from the file read so it can be exercised against captured
    real output without root.
    """
    vals = [int(v) for v in raw.strip().split(",") if v.strip()]
    if len(vals) != 101:
        raise ValueError(f"expected 101 percentiles (P0..P100), got {len(vals)}")
    return vals


@dataclass
class DamonSnapshot:
    """One read of DAMON_STAT, with the derived shares a caller usually wants."""

    enabled: bool
    idle_ms: List[int] = field(default_factory=list)
    est_bandwidth_bps: Optional[int] = None
    aggr_interval_us: Optional[int] = None
    error: Optional[str] = None

    # ── derived ──────────────────────────────────────────────────────────────

    @property
    def hot_pct(self) -> float:
        """Share of memory being actively accessed (negative idle)."""
        if not self.idle_ms:
            return 0.0
        return 100.0 * sum(1 for v in self.idle_ms if v < 0) / len(self.idle_ms)

    @property
    def idle_pct(self) -> float:
        return 100.0 - self.hot_pct if self.idle_ms else 0.0

    def cold_pct_above(self, idle_ms_threshold: int) -> float:
        """Share of memory idle for longer than `idle_ms_threshold`.

        The caller picks the threshold. There is no default, on purpose.
        """
        if not self.idle_ms:
            return 0.0
        return 100.0 * sum(1 for v in self.idle_ms if v > idle_ms_threshold) / len(self.idle_ms)

    def idle_percentile(self, p: int) -> Optional[int]:
        """Idle ms at percentile p (0..100). Negative means hot, not idle."""
        if not self.idle_ms or not 0 <= p <= 100:
            return None
        return self.idle_ms[p]

    @property
    def max_idle_ms(self) -> Optional[int]:
        return max(self.idle_ms) if self.idle_ms else None

    @property
    def hottest_ms(self) -> Optional[int]:
        """Most-negative value: how long the hottest region has been hot."""
        return min(self.idle_ms) if self.idle_ms else None

    def as_dict(self) -> dict:
        d = {
            "enabled": self.enabled,
            "hot_pct": round(self.hot_pct, 1),
            "idle_pct": round(self.idle_pct, 1),
            "idle_p50_ms": self.idle_percentile(50),
            "idle_p90_ms": self.idle_percentile(90),
            "max_idle_ms": self.max_idle_ms,
            "hottest_ms": self.hottest_ms,
            "est_bandwidth_bps": self.est_bandwidth_bps,
            "aggr_interval_us": self.aggr_interval_us,
            "source": "DAMON_STAT /sys/module/damon_stat/parameters",
        }
        if self.error:
            d["error"] = self.error
        return d


def read_snapshot() -> DamonSnapshot:
    """Read DAMON_STAT now. Requires root; returns an error-bearing snapshot
    rather than raising, so a caller polling in a loop cannot be killed by a
    transient permission or module-unload."""
    if not available():
        return DamonSnapshot(enabled=False, error="DAMON_STAT not present (CONFIG_DAMON_STAT)")

    raw_idle = _read_param(_P_IDLE)
    if raw_idle is None:
        return DamonSnapshot(
            enabled=False,
            error=f"cannot read {DAMON_STAT_DIR}/{_P_IDLE} (mode 0400, needs root)",
        )

    enabled_raw = _read_param(_P_ENABLED)
    enabled = (enabled_raw or "").strip().upper() in ("Y", "1", "TRUE")

    try:
        idle = parse_idle_percentiles(raw_idle)
    except ValueError as e:
        return DamonSnapshot(enabled=enabled, error=str(e))

    def _int_or_none(name: str) -> Optional[int]:
        v = _read_param(name)
        try:
            return int(v) if v is not None else None
        except ValueError:
            return None

    return DamonSnapshot(
        enabled=enabled,
        idle_ms=idle,
        est_bandwidth_bps=_int_or_none(_P_BW),
        aggr_interval_us=_int_or_none(_P_AGGR),
    )


# ── published summary (root writes, everyone reads) ───────────────────────────

PUBLISHED_FILE = Path("/var/lib/greenboost/damon_cold")


def render_published(snap: DamonSnapshot, cold_thresholds_ms=(200_000, 400_000, 800_000)) -> str:
    """key=value text for PUBLISHED_FILE, matching the ram_pressure precedent."""
    import time

    lines = [
        f"enabled={'1' if snap.enabled else '0'}",
        f"hot_pct={snap.hot_pct:.1f}",
        f"idle_pct={snap.idle_pct:.1f}",
    ]
    for t in cold_thresholds_ms:
        lines.append(f"cold_pct_above_{t}ms={snap.cold_pct_above(t):.1f}")
    lines += [
        f"idle_p50_ms={snap.idle_percentile(50)}",
        f"idle_p90_ms={snap.idle_percentile(90)}",
        f"max_idle_ms={snap.max_idle_ms}",
        f"hottest_ms={snap.hottest_ms}",
        f"est_bandwidth_bps={snap.est_bandwidth_bps}",
        f"aggr_interval_us={snap.aggr_interval_us}",
    ]
    if snap.error:
        lines.append(f"error={snap.error}")
    lines.append(f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%S')}")
    return "\n".join(lines) + "\n"


def read_published() -> dict:
    """Unprivileged read of what the root supervisor published.

    This is the entry point for gb_tiering and anything else running as the
    user. Returns {'available': False, ...} rather than raising when the
    supervisor has not written yet — a box with no DAMON evidence is a valid,
    actionable answer, not an error.
    """
    try:
        text = PUBLISHED_FILE.read_text()
    except OSError as e:
        return {"available": False, "reason": f"{PUBLISHED_FILE} unreadable: {e}"}

    out: dict = {"available": True, "source": str(PUBLISHED_FILE)}
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k in ("timestamp", "error", "source"):
            out[k] = v
            continue
        try:
            out[k] = float(v) if "." in v else int(v)
        except ValueError:
            out[k] = v
    return out


if __name__ == "__main__":
    import json
    import sys

    if "--published" in sys.argv:
        print(json.dumps(read_published(), indent=1))
    else:
        snap = read_snapshot()
        d = snap.as_dict()
        if snap.idle_ms:
            d["cold_pct_above"] = {
                f"{t}ms": round(snap.cold_pct_above(t), 1)
                for t in (200_000, 400_000, 800_000)
            }
        print(json.dumps(d, indent=1))
        if not snap.idle_ms and snap.error:
            print(f"\nhint: needs root — try: sudo python3 {__file__}", file=sys.stderr)

#!/usr/bin/env python3
"""GreenBoost Prometheus text-format exporter (bare-metal).

Reads /run/greenboost/metrics.json (written by the CUDA shim) and emits
Prometheus text exposition on stdout.  Intended for use with a cron-based
node_exporter textfile collector or a systemd timer.

Canonical metric reference: observability/METRICS.md
Use this exporter on bare-metal nodes. For Kubernetes, use the Go exporter
in cmd/greenboost-metrics-exporter/ (installed by the DRA Helm chart).
NOTE: memory metrics use MiB (not bytes) - see METRICS.md §Unit discrepancy.

Usage:
    python3 greenboost_exporter.py > /var/lib/node_exporter/textfile_collector/greenboost.prom
    # Or run as a tiny HTTP server on port 9742:
    python3 greenboost_exporter.py --serve [--port N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

METRICS_PATH      = os.environ.get("GREENBOOST_METRICS_JSON", "/run/greenboost/metrics.json")
SHIM_STATS_PATH   = "/run/greenboost/shim_stats"
SHIM_STATS_ALT    = "/tmp/greenboost_shim_stats"
SYSFS_BASE        = "/sys/class/greenboost/greenboost"
PREFIX            = "greenboost"

# Audit F-L5-12: allowlist for the `tier` label so a corrupted/extended
# metrics.json cannot blow up Prometheus storage with arbitrary label values.
ALLOWED_TIERS: tuple[str, ...] = ("T1", "T2", "T3")


def read_metrics() -> dict[str, Any]:
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def metrics_mtime() -> float:
    """Audit F-L5-13: when JSON parse fails, the operator still needs a way to
    distinguish a freshly-crashed shim from a never-running one.  Returning
    mtime lets emit() compute staleness even from a {} payload."""
    try:
        return os.stat(METRICS_PATH).st_mtime
    except OSError:
        return 0.0


def read_shim_stats() -> dict[str, Any]:
    """Parse /run/greenboost/shim_stats (key=value plaintext written by the CUDA shim)."""
    for path in [SHIM_STATS_PATH, SHIM_STATS_ALT]:
        try:
            with open(path) as f:
                content = f.read()
            result: dict[str, Any] = {}
            for line in content.splitlines():
                if "=" not in line:
                    continue
                k, _, v = line.partition("=")
                v = v.strip()
                try:
                    result[k.strip()] = int(v)
                except ValueError:
                    result[k.strip()] = v
            return result
        except OSError:
            continue
    return {}


def read_sysfs_stats() -> dict[str, int]:
    """Read machine-readable GreenBoost sysfs attributes.

    Uses individual attribute files and pool_brief (compact summary line):
      kv_reserve_mb  - /sys/class/greenboost/greenboost/kv_reserve_mb (single int)
      pool_brief     - /sys/class/greenboost/greenboost/pool_brief
                       format: "T1:12GB T2:1024/65536GB(1%) T3:0/0GB PRESSURE:ok KV_RSV:2048MB KV_T2:512MB"
    The human-readable `status` attribute is NOT parsed (it is not key=value).
    """
    import re as _re
    result: dict[str, int] = {}

    # kv_reserve_mb - dedicated single-integer sysfs attribute
    try:
        with open(f"{SYSFS_BASE}/kv_reserve_mb") as _f:
            v = _f.read().strip()
            if v.lstrip("-").isdigit():
                result["kv_reserve_mb"] = int(v)
    except OSError:
        pass

    # pool_brief - compact summary line, all other sysfs metrics
    try:
        with open(f"{SYSFS_BASE}/pool_brief") as _f:
            brief = _f.read().strip()
        # T2 allocated: "T2:<alloc_gb>/..." (pool_brief reports t2_alloc_mb/1024)
        _m = _re.search(r"T2:(\d+)/", brief)
        if _m:
            result["t2_allocated_mb"] = int(_m.group(1)) * 1024
        # Pressure string → int (kernel uses swap_pressure for both T2 DDR and T3 NVMe)
        _m = _re.search(r"PRESSURE:(\w+)", brief)
        if _m:
            _pval = {"ok": 0, "warn": 1, "CRITICAL": 2}.get(_m.group(1), 0)
            result["t2_pressure"]  = _pval
            result["swap_pressure"] = _pval
        # KV reserve from pool_brief (fallback if dedicated attr unavailable)
        _m = _re.search(r"KV_RSV:(\d+)MB", brief)
        if _m:
            result.setdefault("kv_reserve_mb", int(_m.group(1)))
        # KV spilled to T2 DDR
        _m = _re.search(r"KV_T2:(\d+)MB", brief)
        if _m:
            result["kv_t2_mb"] = int(_m.group(1))
    except OSError:
        pass

    return result


# Audit F-L5-10 / F-L5-29: distinguish counter from gauge.  Lines suffixed
# with _total or _count and metrics that only ever increase are counters per
# Prometheus convention; alerts that use rate() / increase() rely on this.
_COUNTER_NAMES: frozenset[str] = frozenset({
    "remote_alloc_count",
    "kernel_dispatch_count",
    "tier_alloc_count",
    "tier_lifetime_mb",
    "heartbeat_miss_total",
    "shim_h2d_mb",
    "shim_d2h_mb",
    "shim_cold_evicts",
    "shim_kv_dedup",
    "shim_remote_alloc_count",
    "shim_remote_alloc_mb",
    "fake_ptr_generation",
})


def emit(m: dict[str, Any] | None) -> str:
    lines: list[str] = []
    _declared: set[str] = set()

    def metric(name: str, value: Any, labels: str = "", help_text: str = "") -> None:
        if name not in _declared:
            if help_text:
                lines.append(f"# HELP {PREFIX}_{name} {help_text}")
            mtype = "counter" if name in _COUNTER_NAMES else "gauge"
            lines.append(f"# TYPE {PREFIX}_{name} {mtype}")
            _declared.add(name)
        label_str = f"{{{labels}}}" if labels else ""
        lines.append(f"{PREFIX}_{name}{label_str} {value}")

    if not m:
        metric("up", 0, help_text="1 if GreenBoost shim metrics are available")
        # Audit F-L5-13: still emit staleness so SREs can tell when the shim
        # actually died versus never started.
        mt = metrics_mtime()
        if mt > 0:
            metric("shim_staleness_s", max(0.0, time.time() - mt),
                   help_text="Seconds since metrics.json was last written")
        return "\n".join(lines) + "\n"

    metric("up", 1, help_text="1 if GreenBoost shim metrics are available")
    metric("local_t1_alloc_mb",     m.get("local_t1_alloc_mb", 0),     help_text="Local GPU VRAM in use (MB)")
    metric("remote_alloc_count",    m.get("remote_alloc_count", 0),    help_text="Allocations routed to feeder T1")
    metric("kernel_dispatch_count", m.get("kernel_dispatch_count", 0), help_text="Kernels dispatched to feeder GPU")
    metric("t2_pool_frag_pct",      m.get("t2_pool_frag_pct", 0),      help_text="T2 pool fragmentation 0-100")
    metric("t2_above_warn",         m.get("t2_above_warn", 0),         help_text="1 if T2 usage > 75% warn threshold")
    metric("timestamp",             m.get("timestamp", 0),             help_text="Unix timestamp of last shim stats write")

    # M1: new enhancement metrics (A2, A3, U18, U20)
    metric("fake_ptr_generation",   m.get("fake_ptr_generation", 0),
           help_text="A2 fake-pointer wraparound generation counter (increments on each wrap)")
    metric("double_buffer_enabled", m.get("double_buffer_enabled", 0),
           help_text="1 if U18 double-buffer T3→T2 prefetch is active")
    metric("kv_compress_enabled",   m.get("kv_compress_enabled", 0),
           help_text="1 if E1 K/V int8 absmax compression is active")

    # Always emit all known tiers so Prometheus never sees a missing series.
    # Merging shim data over zero defaults ensures dashboards and alerts that
    # expect greenboost_shim_tier_* to always exist get 0 rather than no data.
    _tier_defaults = {t: {"cur_mb": 0, "peak_mb": 0, "lifetime_mb": 0, "alloc_count": 0}
                      for t in ALLOWED_TIERS}
    _tiers = {**_tier_defaults, **m.get("tiers", {})}
    for tier, stats in _tiers.items():
        # Audit F-L5-12: allowlist the `tier` label.
        if tier not in ALLOWED_TIERS:
            continue
        lbl = f'tier="{tier}"'
        metric("tier_cur_mb",      stats.get("cur_mb", 0),      labels=lbl, help_text="Current MB allocated per tier")
        metric("tier_peak_mb",     stats.get("peak_mb", 0),     labels=lbl, help_text="Peak MB per tier")
        metric("tier_lifetime_mb", stats.get("lifetime_mb", 0), labels=lbl, help_text="Lifetime MB allocated per tier")
        metric("tier_alloc_count", stats.get("alloc_count", 0), labels=lbl, help_text="Total alloc calls per tier")

    # M1: per-feeder bandwidth and health metrics.  Audit F-L5-11: the feeder
    # label is bounded by the cluster size - fine in practice, but warn the
    # operator before it gets out of hand.
    feeders = m.get("feeders", [])
    if len(feeders) > 32:
        sys.stderr.write(
            f"[greenboost_exporter] WARN: {len(feeders)} feeder labels - "
            "high cardinality; consider aggregating\n"
        )
    for feeder in feeders:
        fdr = str(feeder.get("feeder", "unknown"))
        lbl = f'feeder="{fdr}"'
        metric("bw_ewma_mbs",           feeder.get("bw_measured_mbs", 0),  labels=lbl,
               help_text="U20 EWMA-measured PCIe bandwidth to feeder (MiB/s)")
        metric("heartbeat_miss_total",  feeder.get("heartbeat_miss_total", 0), labels=lbl,
               help_text="A3 consecutive heartbeat miss count per feeder")
        metric("feeder_health_state",   feeder.get("health_state", 0),     labels=lbl,
               help_text="Feeder health state: 0=HEALTHY 1=DEGRADED 2=UNHEALTHY 3=QUARANTINE 4=DISABLED")
        metric("feeder_throttled",      feeder.get("throttled", 0),        labels=lbl,
               help_text="1 if feeder GPU is clock-throttled")
        metric("feeder_t1_quarantined", feeder.get("t1_quarantined", 0),   labels=lbl,
               help_text="1 if feeder T1 VRAM is ECC-quarantined")
        metric("feeder_gpu_util_pct",   feeder.get("gpu_util_pct", 0),     labels=lbl,
               help_text="Feeder GPU utilization 0-100% from last heartbeat")

    # ── Shim stats (from /run/greenboost/shim_stats - richer real-time data) ────
    shim = read_shim_stats()
    if shim:
        metric("shim_h2d_mb",       shim.get("h2d_mb", 0),
               help_text="Host→Device DMA traffic since last shim reset (MB)")
        metric("shim_d2h_mb",       shim.get("d2h_mb", 0),
               help_text="Device→Host DMA traffic since last shim reset (MB)")
        metric("shim_headroom_mb",  shim.get("vram_headroom_mb", 0),
               help_text="Current VRAM headroom before T2 spillover begins (MB)")
        metric("shim_cold_evicts",  shim.get("cold_epoch_evict_count", 0),
               help_text="Cold epoch eviction count (buffers demoted T1→T2 by LRU)")
        metric("shim_kv_dedup",     shim.get("kv_dedup_hits", 0),
               help_text="KV cache dedup hits (prefix cache reuse)")
        metric("shim_kv_frag_mb",   shim.get("kv_internal_frag_mb", 0),
               help_text="KV cache internal fragmentation (MB)")
        metric("shim_t2_frag_pct",  shim.get("t2_warn_adj_pct", 0),
               help_text="T2 pool fragmentation-adjusted warning threshold (%)")
        metric("shim_remote_alloc_count", shim.get("remote_alloc_count", 0),
               help_text="Allocations offloaded to a T4 cluster feeder")
        metric("shim_remote_alloc_mb",    shim.get("remote_alloc_mb", 0),
               help_text="Total MB offloaded to cluster feeder(s)")
        ts = shim.get("timestamp", 0)
        if ts:
            # Audit F-L5-33/F-L5-25: clamp negative staleness; use float for
            # sub-second precision (avoids int() truncation on both sides).
            staleness = max(0.0, time.time() - float(ts))
            metric("shim_staleness_s", staleness,
                   help_text="Seconds since shim last wrote stats (>30 = no active process)")

    # ── Sysfs (kernel module live values from pool_brief + kv_reserve_mb) ───────
    # Note: kv_used_mb is not available in machine-readable sysfs; omitted here
    # to avoid emitting a misleading constant-zero series.
    sysfs = read_sysfs_stats()
    if sysfs:
        if "t2_pressure" in sysfs:
            metric("t2_pressure",    sysfs["t2_pressure"],
                   help_text="T2 DDR pressure level: 0=ok 1=warn 2=critical")
        if "swap_pressure" in sysfs:
            metric("swap_pressure",  sysfs["swap_pressure"],
                   help_text="T3 NVMe swap pressure level: 0=ok 1=warn 2=critical")
        if "kv_t2_mb" in sysfs:
            metric("kv_t2_mb",       sysfs["kv_t2_mb"],
                   help_text="KV cache spilled to T2 DDR (MB)")
        if "kv_reserve_mb" in sysfs:
            metric("kv_reserve_mb",  sysfs["kv_reserve_mb"],
                   help_text="KV cache T1 VRAM hard reserve (MB)")
        if "t2_allocated_mb" in sysfs:
            metric("t2_allocated_mb", sysfs["t2_allocated_mb"],
                   help_text="T2 DDR pool currently allocated (MB)")

    return "\n".join(lines) + "\n"


def serve(port: int = 9742, bind: str = "127.0.0.1") -> None:
    """Audit F-L5-19: use ThreadingHTTPServer with daemon threads so a slow
    read_metrics() doesn't block other scrapers.  Per-request CPU is small
    enough that we don't need an explicit cache here - a request thread that
    blocks on file I/O is fine.

    PR-J: default-bind to 127.0.0.1.  The previous default ("") bound the
    exporter on ALL interfaces and leaked per-tier allocation counts,
    feeder state, and GreenBoost build version to anyone with LAN reach.
    Operators who want LAN visibility should pass --bind 0.0.0.0
    explicitly (intentional opt-in, not a silent default). """
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = emit(read_metrics()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: Any) -> None:
            return None

    server = ThreadingHTTPServer((bind, port), Handler)
    server.daemon_threads = True
    print(f"[GreenBoost exporter] Listening on {bind}:{port}", file=sys.stderr)
    server.serve_forever()


def main() -> None:
    # Audit F-L5-26: parameterise port via env / CLI flag.
    p = argparse.ArgumentParser(description="GreenBoost Prometheus exporter")
    p.add_argument("--serve", action="store_true", help="Run as an HTTP server")
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("GREENBOOST_EXPORTER_PORT", "9742")),
                   help="HTTP port (default 9742, or $GREENBOOST_EXPORTER_PORT)")
    p.add_argument("--bind", type=str,
                   default=os.environ.get("GREENBOOST_EXPORTER_BIND", "127.0.0.1"),
                   help="Bind address (default 127.0.0.1; pass 0.0.0.0 for LAN exposure, "
                        "or $GREENBOOST_EXPORTER_BIND)")
    args = p.parse_args()
    if args.serve:
        serve(args.port, args.bind)
    else:
        sys.stdout.write(emit(read_metrics()))


if __name__ == "__main__":
    main()

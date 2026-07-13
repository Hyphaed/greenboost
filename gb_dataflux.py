#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_dataflux.py , GreenBoost cluster dataflux event log + web UI.

Records every unit of work gb_cluster.py's dispatch functions (cluster_map,
ClusterJobQueue, ensure_feeder_model, stage_bundle) push to the host or a
feeder: which script/label, which node, how many items, how long it took,
success/failure. A durable JSONL log (one event per line , no database
needed) is the single source of truth for "how did data flow through the
cluster" , read by this file's web UI, by `gb_dataflux_mcp.py` (so an LLM
can query it), and by any ad-hoc script.

Log location: ~/.local/share/greenboost/dataflux.jsonl (override with
GREENBOOST_DATAFLUX_LOG). Never raises on a logging failure , recording
history must never break a real dispatch.

CLI:
    python3 gb_dataflux.py serve [--port 8799] [--days 5]   # web UI
    python3 gb_dataflux.py summary [--days 5] [--llm]        # text/JSON summary
    python3 gb_dataflux.py critic [--days 1] [--llm]         # incident diagnosis
"""
from __future__ import annotations

import http.server
import json
import os
import socketserver
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_DEFAULT_LOG_PATH = str(Path.home() / ".local" / "share" / "greenboost" / "dataflux.jsonl")

DEFAULT_PORT = 8799
DEFAULT_DAYS = 5.0


def _log_path() -> Path:
    """Resolve the dataflux log path from GREENBOOST_DATAFLUX_LOG on every
    call (not a module-level constant) , so tests can redirect it reliably
    with monkeypatch.setenv, and so a long-lived process (the web server,
    a supervisor) picks up an env change without a restart."""
    return Path(os.environ.get("GREENBOOST_DATAFLUX_LOG", _DEFAULT_LOG_PATH))


class SnapshotRecorder:
    """Periodically records a GpuMetrics snapshot to the dataflux log , the
    continuous 'flight recorder' view of system state (VRAM used/free, GPU
    util%, KV cache pressure, T2/T3 pool occupancy) alongside the discrete
    dispatch/quantize/tier events emitted elsewhere. Works standalone on a
    single host , no cluster/feeder required, since it only reads the
    local telemetry snapshot.

    Throttled (default every 5s) via the telemetry callback , TelemetryManager
    polls every 500ms, but recording every poll would write ~172k events/day;
    5s keeps 5 days of continuous history to a reasonable JSONL size while
    still giving a real time-series, not just point-in-time snapshots.
    """

    def __init__(self, telemetry_manager, interval_s: float = 5.0, node: str | None = None):
        self._interval_s = interval_s
        self._last_emit = 0.0
        self._node = node
        # Discrete-transition state , greenboost's own orchestration decisions
        # (shim phase, T2 pressure, T3 spill) must be followable as EVENTS in the
        # dataflux MCP, not just inferred from the 5s snapshot time-series. See
        # CLAUDE.md "Observability Must-Rule" pt 4 (greenboost itself emits).
        self._prev_phase = None
        self._prev_t3_active = None
        self._prev_t2_bucket = None
        telemetry_manager.add_callback(self._on_metrics)

    def _bucket(self, pressure) -> str:
        try:
            p = float(pressure)
        except (TypeError, ValueError):
            return "ok"
        return "critical" if p >= 0.9 else "warn" if p >= 0.6 else "ok"

    def _detect_transitions(self, m, gb, node: str) -> None:
        """Emit a `shim_transition` event whenever an orchestration-relevant
        shim decision changes state. Runs every poll (not throttled) so fast
        transitions aren't lost; best-effort, never raises."""
        phase = (getattr(m, "shim_phase", "") or "").strip()
        t3_active = bool(gb.t3_used_mb) if gb else False
        t2_bucket = self._bucket(gb.t2_pressure if gb else 0)
        for field, prev_attr, frm, to, status in (
            ("shim_phase", "_prev_phase", self._prev_phase, phase, "ok"),
            ("t3_spill", "_prev_t3_active",
             self._prev_t3_active, t3_active,
             "warn" if t3_active else "ok"),
            ("t2_pressure", "_prev_t2_bucket", self._prev_t2_bucket, t2_bucket,
             "error" if t2_bucket == "critical" else
             "warn" if t2_bucket == "warn" else "ok"),
        ):
            if to == "" or to is None:
                continue
            if frm is None:                      # first observation , seed, no event
                setattr(self, prev_attr, to)
                continue
            if to != frm:
                setattr(self, prev_attr, to)
                emit({
                    "node": node, "label": "shim", "kind": "shim_transition",
                    "stage": field, "from": frm, "to": to,
                    "n_items": 0, "items": [], "duration_s": 0.0,
                    "status": status,
                    "t2_pressure": (gb.t2_pressure if gb else 0),
                    "t3_used_mb": (gb.t3_used_mb if gb else 0),
                    "fb_used_pct": round(getattr(m, "fb_used_pct", 0.0), 1),
                })

    def _on_metrics(self, m) -> None:
        now = time.time()
        gb = getattr(m, "gb", None)
        node = self._node if self._node is not None else (
            "host" if getattr(m, "device", 0) == 0 else f"gpu{m.device}")
        # Discrete orchestration transitions , every poll, unthrottled.
        self._detect_transitions(m, gb, node)
        # Continuous flight-recorder snapshot , throttled to interval_s.
        if now - self._last_emit < self._interval_s:
            return
        self._last_emit = now
        sys_m = getattr(m, "sys", None)
        node = self._node if self._node is not None else (
            "host" if getattr(m, "device", 0) == 0 else f"gpu{m.device}")
        ev = {
            "node": node,
            "label": "system_snapshot", "kind": "snapshot",
            "n_items": 0, "items": [], "duration_s": 0.0, "status": "ok",
            "fb_used_mb": getattr(m, "fb_used_mb", 0),
            "fb_free_mb": getattr(m, "fb_free_mb", 0),
            "fb_total_mb": getattr(m, "fb_total_mb", 0),
            "fb_used_pct": round(getattr(m, "fb_used_pct", 0.0), 1),
            "gpu_util_pct": round(getattr(m, "gpu_util_pct", 0.0), 1),
            "temp_c": round(getattr(m, "temp_c", 0.0), 1),
            "power_w": round(getattr(m, "power_w", 0.0), 1),
            "cpu_util_pct": round(sys_m.cpu_util_pct, 1) if sys_m else 0.0,
            "kv_used_mb": gb.kv_used_mb if gb else 0,
            "kv_reserve_mb": gb.kv_reserve_mb if gb else 0,
            "kv_t2_mb": gb.kv_t2_mb if gb else 0,
            "t2_allocated_mb": gb.t2_allocated_mb if gb else 0,
            "t2_available_mb": gb.t2_available_mb if gb else 0,
            "t2_pressure": gb.t2_pressure if gb else 0,
            "t3_used_mb": gb.t3_used_mb if gb else 0,
            "t3_pressure": gb.t3_pressure if gb else 0,
            "shim_phase": getattr(m, "shim_phase", ""),
        }
        # P0-C: compute/bandwidth/speed axes , added only when non-zero so
        # the event stays small and old consumers (summarize/critic use
        # .get()) keep working against pre-widening snapshots.
        topo = getattr(m, "topology", None)
        for key, val in (
            ("sm_clock_mhz", getattr(m, "sm_clock_mhz", 0)),
            ("mem_clock_mhz", getattr(m, "mem_clock_mhz", 0)),
            ("mem_copy_util_pct", round(getattr(m, "mem_copy_util_pct", 0.0), 1)),
            ("power_limit_w", round(getattr(m, "power_limit_w", 0.0), 1)),
            ("pcie_tx_mb_s", round(getattr(m, "pcie_tx_mb_s", 0.0), 1)),
            ("pcie_rx_mb_s", round(getattr(m, "pcie_rx_mb_s", 0.0), 1)),
            ("throttle_reasons", getattr(m, "throttle_reasons", 0)),
            ("ddr_speed_mts", getattr(m, "ddr_speed_mts", 0)),
            # Static-ish link identity: host from topology, feeder from the
            # GbPoolInfo filled out of metrics.json.
            ("pcie_gen", getattr(topo, "pcie_gen_current", 0) if topo
             else getattr(gb, "pcie_link_gen", 0) if gb else 0),
            ("pcie_width", getattr(topo, "pcie_width_current", 0) if topo
             else getattr(gb, "pcie_link_width", 0) if gb else 0),
            ("t2_speed_mts", getattr(gb, "t2_speed_mts", 0) if gb else 0),
            ("t3_speed_mbs", getattr(gb, "t3_speed_mbs", 0) if gb else 0),
            ("vram_bw_gbps", getattr(topo, "vram_bw_gbps", 0.0) if topo else 0.0),
        ):
            if val:
                ev[key] = val
        emit(ev)


def _feeder_node_label(feeder_idx: int) -> str:
    """'feeder0 (1.2.3.4)' when the feeder's IP is known from metrics.json,
    else plain 'feeder0'."""
    label = f"feeder{feeder_idx}"
    try:
        with open("/run/greenboost/metrics.json") as f:
            feeders = json.load(f).get("feeders", [])
        if feeder_idx < len(feeders):
            ip = feeders[feeder_idx].get("ip") or feeders[feeder_idx].get("host")
            if ip:
                label = f"{label} ({ip})"
    except Exception:
        pass
    return label


def start_snapshot_recorder(telemetry_manager=None, interval_s: float = 5.0):
    """Start recording periodic system snapshots to the dataflux log.

    Pass a TelemetryManager/ClusterTelemetryManager, or omit to use
    gb_init's active singleton (the common case , gb_init.py calls this
    automatically when GREENBOOST_ACTIVE=1). Returns the recorder(s), or
    None when no telemetry manager is available. Caller doesn't need to
    keep the reference alive , TelemetryManager's callback list holds it.

    Node labels: "host" for local device 0, "gpu<N>" for other local
    devices, "feeder<N> (ip)" for RemoteFeederProvider-backed managers , so
    the dataflux UI can tell which physical machine each row came from.
    """
    if telemetry_manager is None:
        try:
            import gb_init
            telemetry_manager = gb_init.get_telemetry()
        except Exception:
            return None
    if telemetry_manager is None:
        return None
    managers = getattr(telemetry_manager, "_managers", None) or [telemetry_manager]

    recorders = []
    feeder_idx = 0
    for m in managers:
        providers = getattr(m, "_providers", None) or []
        is_feeder = len(providers) == 1 and type(providers[0]).__name__ == "RemoteFeederProvider"
        if is_feeder:
            node = _feeder_node_label(feeder_idx)
            feeder_idx += 1
        else:
            node = "host" if getattr(m, "device", 0) == 0 else f"gpu{m.device}"
        recorders.append(SnapshotRecorder(m, interval_s=interval_s, node=node))
    return recorders[0] if len(recorders) == 1 else recorders


def emit(event: dict) -> None:
    """Append one structured event to the dataflux log.

    Best-effort , logging failures (disk full, permissions, or a
    non-JSON-serializable value in the event dict) are swallowed so a
    dispatch never fails because history couldn't be recorded.
    """
    try:
        log_path = _log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        event = dict(event)
        event.setdefault("ts", time.time())
        with open(log_path, "a") as f:
            f.write(json.dumps(event) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def read_events(since_hours: float | None = None) -> list[dict]:
    """All logged events, optionally filtered to the last `since_hours`."""
    log_path = _log_path()
    if not log_path.exists():
        return []
    cutoff = time.time() - since_hours * 3600 if since_hours else None
    out: list[dict] = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if cutoff is None or ev.get("ts", 0) >= cutoff:
                out.append(ev)
    out.sort(key=lambda e: e.get("ts", 0))
    return out


def summarize(events: list[dict]) -> dict:
    """Aggregate events: per-node item/duration totals, per-label item
    counts, per-run grouping, error count. Cheap , pure Python, no numpy."""
    nodes: dict[str, dict] = {}
    labels: dict[str, int] = {}
    runs: dict[str, dict] = {}
    tok_s: dict[str, dict] = {}
    stages: dict[str, dict] = {}
    errors = 0
    total_items = 0
    total_duration = 0.0

    for ev in events:
        # P1-D per-node attribution: feeder events key tok_s/stages as
        # "<node>:<model>" / "<node>:<stage>" so host and feeder rates never
        # blend.  Host events ("", "host", None node) keep their unprefixed
        # keys , existing host dashboards are unaffected.
        _ev_node = ev.get("node")
        _node_prefix = f"{_ev_node}:" if _ev_node not in ("", "host", None) else ""

        if ev.get("kind") == "tok_s_measured":
            model = _node_prefix + ev.get("model", "?")
            t = tok_s.setdefault(model, {"latest": 0.0, "samples": 0, "_sum": 0.0, "last_ts": 0})
            t["latest"] = ev.get("tok_s", 0.0)
            t["samples"] += 1
            t["_sum"] += ev.get("tok_s", 0.0)
            t["last_ts"] = ev.get("ts", 0)

        if ev.get("kind") == "stage_profile":
            stage = _node_prefix + ev.get("stage", "?")
            s = stages.setdefault(stage, {"count": 0, "_walls": [],
                                          "last_status": "", "last_ts": 0})
            s["count"] += 1
            s["_walls"].append(ev.get("duration_s", 0.0))
            s["last_status"] = ev.get("status", "")
            s["last_ts"] = ev.get("ts", 0)

        node = ev.get("node", "?")
        n = nodes.setdefault(node, {"items": 0, "duration_s": 0.0, "events": 0, "errors": 0})
        n["items"] += ev.get("n_items", 0)
        n["duration_s"] += ev.get("duration_s", 0.0)
        n["events"] += 1
        if ev.get("status") == "error":
            n["errors"] += 1
            errors += 1

        label = ev.get("label", "?")
        labels[label] = labels.get(label, 0) + ev.get("n_items", 0)

        run_id = ev.get("run_id", "?")
        r = runs.setdefault(run_id, {"label": label, "started": ev.get("ts", 0),
                                     "finished": ev.get("ts", 0), "items": 0,
                                     "nodes": set()})
        r["items"] += ev.get("n_items", 0)
        r["nodes"].add(node)
        r["started"] = min(r["started"], ev.get("ts", 0))
        r["finished"] = max(r["finished"], ev.get("ts", 0))

        total_items += ev.get("n_items", 0)
        total_duration += ev.get("duration_s", 0.0)

    for r in runs.values():
        r["nodes"] = sorted(r["nodes"])

    for t in tok_s.values():
        t["avg"] = round(t.pop("_sum") / t["samples"], 1) if t["samples"] else 0.0

    for s in stages.values():
        raw = s.pop("_walls")
        walls = sorted(raw)
        n = len(walls)
        s["mean_s"] = round(sum(walls) / n, 3) if n else 0.0
        # nearest-rank p95: ceil(0.95*n)-th order statistic, clamped
        s["p95_s"] = round(walls[min(n - 1, -(-(n * 95) // 100) - 1)], 3) if n else 0.0
        s["latest_s"] = round(raw[-1], 3) if n else 0.0  # arrival order, not max

    return {
        "event_count": len(events),
        "total_items": total_items,
        "total_duration_s": round(total_duration, 1),
        "errors": errors,
        "nodes": nodes,
        "labels": labels,
        "runs": runs,
        "tok_s": tok_s,
        "stages": stages,
    }


# ── Critic reports , snapshot-correlated incident diagnosis ──────────────────
# "What was happening at that moment": every error/warn event gets the nearest
# flight-recorder snapshots (±60s before/after) attached, plus the cluster
# activity in that window, plus rule-based diagnosis hints , so a human or an
# LLM can diagnose cluster inference from evidence instead of guessing.

_CRITIC_WINDOW_S = 60.0
_SNAP_VIEW_KEYS = ("node", "fb_used_mb", "fb_free_mb", "fb_total_mb",
                   "fb_used_pct", "gpu_util_pct", "cpu_util_pct", "shim_phase",
                   "kv_used_mb", "kv_reserve_mb", "t2_allocated_mb",
                   "t2_available_mb", "t2_pressure", "t3_used_mb", "t3_pressure")
_DISPATCH_KINDS = ("chunk_remote", "chunk_local", "job_remote", "job_local")


def _snap_view(snap: dict | None, incident_ts: float) -> dict | None:
    """Compact view of one snapshot event, with its offset from the incident."""
    if snap is None:
        return None
    d = {k: snap.get(k) for k in _SNAP_VIEW_KEYS}
    d["ts"] = snap.get("ts", 0)
    d["offset_s"] = round(snap.get("ts", 0) - incident_ts, 1)
    return d


def _bits_below_fp8(bits) -> bool:
    """gb-quant quality floor is fp8 , any 4-bit plan (int4/nf4/fp4/q4) is
    below it. int8/fp8/bf16 are at or above the floor."""
    s = str(bits).lower()
    return (any(t in s for t in ("int4", "nf4", "fp4", "4bit", "q4"))
            or s.strip() == "4")


def _is_incident(ev: dict) -> bool:
    if ev.get("status") == "error":
        return True
    return ev.get("kind") == "shim_transition" and ev.get("status") == "warn"


def _diagnosis_hints(ev: dict, before: dict | None, after: dict | None,
                     ctx: dict, window: list[dict]) -> list[str]:
    """Rule-based diagnosis strings for one incident. Pure; each rule reads
    the correlated snapshots/context, never the wall clock."""
    hints: list[str] = []
    snap = after or before   # "after" reflects the incident's consequences
    # shim_transition events carry their own t2/t3/VRAM numbers , usable as
    # the metric source when no snapshot landed within the window.
    if snap is None and ev.get("kind") == "shim_transition":
        snap = ev
    err_text = " ".join(str(ev.get(k, "")) for k in
                        ("error", "stage", "label", "kind")).lower()

    if snap:
        t3 = snap.get("t3_used_mb") or 0
        if t3 > 0:
            hints.append(
                f"placement-floor breach: T3 NVMe held {t3} MB at incident time "
                f", NVMe-tier reads are orders slower than T1/T2; re-run "
                f"quantize_to_fit with a tighter budget (never below fp8) or "
                f"enable TurboQuant KV so the working set fits T1/T2")
        fb = snap.get("fb_used_pct")
        t2 = snap.get("t2_allocated_mb") or 0
        if fb is not None and fb < 70 and t2 > 0:
            hints.append(
                f"Rule#1 underfill: T1 VRAM only {fb:.0f}% full while T2 DDR "
                f"held {t2} MB , weights that could live in VRAM are crossing "
                f"PCIe every access; front-load real VRAM chunks or use the "
                f"llama.cpp --rpc split so T1 fills first")
        util = snap.get("gpu_util_pct") or 0
        remote_items = ctx.get("chunk_remote", 0) + ctx.get("job_remote", 0)
        if remote_items == 0 and util > 90:
            hints.append(
                f"cluster rule: 0 feeder-side work items in ±{int(_CRITIC_WINDOW_S)}s "
                f"while the host GPU ran at {util:.0f}% , a connected feeder sat "
                f"idle; route multi-item batches through cluster_map/"
                f"ClusterJobQueue when a feeder is online")

    phase = str((before or {}).get("shim_phase")
                or (snap or {}).get("shim_phase") or "")
    is_oom = "oom" in err_text or "out of memory" in err_text
    if phase.upper().startswith("INIT") and ev.get("status") == "error":
        # Fires for explicit OOM text AND for stage errors whose event carries
        # no error string (e.g. forge StagedOOMError stage_profile rows) , the
        # INIT-phase correlation is the diagnostic signal either way.
        hints.append(
            ("OOM" if is_oom else "stage error") + " while shim_phase=INIT: "
            "the run failed before the shim reached steady state , likely a "
            "reinstall/restart collision (kernel module or netd not settled: "
            "`lsmod | grep greenboost`) or per-asset model reload pressure; "
            "stagger loads and confirm the kmod is loaded after any reinstall")
    elif is_oom:
        hints.append(
            "OOM in steady state: working set exceeds the planned budget , "
            "re-run quantize_to_fit with a tighter GB_QUANT_BUDGET_GB, "
            "keeping every component at or above the fp8 quality floor")

    low_q = [e for e in window
             if e.get("kind") in ("quantize", "quantize_to_fit")
             and _bits_below_fp8(e.get("bits", ""))]
    if low_q:
        bits = sorted({str(e.get("bits")) for e in low_q})
        hints.append(
            f"quality floor: {len(low_q)} quantize event(s) below fp8 in the "
            f"incident window (bits={bits}) , fp8 is the gb-quant floor for "
            f"cluster inference; a below-fp8 plan means the VRAM budget is too "
            f"tight, fix placement (feeder T1, front-load) instead of dropping "
            f"precision")

    if ev.get("kind") == "shim_transition":
        hints.append(
            f"shim transition {ev.get('stage')}: {ev.get('from')} → "
            f"{ev.get('to')} (t2_pressure={ev.get('t2_pressure')}, "
            f"t3_used_mb={ev.get('t3_used_mb')})")

    if any(t in err_text for t in ("broken pipe", "connection closed",
                                   "connection reset", "remotedisconnected")):
        hints.append(
            "feeder/link drop: the peer closed mid-operation , check "
            "greenboost-netd on the feeder (`greenboost feeders diag`), the "
            "SSH link, and whether an upgrade/restart raced the transfer")

    if not hints:
        hints.append("no rule matched , compare snapshot_before/after deltas "
                     "(VRAM, t2/t3, phase) manually")
    return hints


def critic_report(days: float = 1.0) -> dict:
    """Snapshot-correlated critic report over the dataflux log.

    For every incident (status=error anywhere; shim_transition warns) in the
    last `days` days, attach the nearest snapshot within ±60s before and
    after (VRAM/util/phase/T2/T3/KV = what was happening at that moment), the
    cluster activity in that window (was the feeder working?), and rule-based
    diagnosis_hints. Ends with merged recommendations (incident hints +
    gb_pilot.advise when importable) toward best cluster inference at ≥fp8
    quality. Pure stdlib; every section is best-effort."""
    events = read_events(since_hours=days * 24)
    snaps = [e for e in events if e.get("kind") == "snapshot"]
    snap_ts = [s.get("ts", 0) for s in snaps]   # events are ts-sorted

    import bisect
    incidents: list[dict] = []
    for ev in events:
        if not _is_incident(ev) or ev.get("kind") == "snapshot":
            continue
        ts = ev.get("ts", 0)
        i = bisect.bisect_right(snap_ts, ts)
        before = snaps[i - 1] if i > 0 and ts - snap_ts[i - 1] <= _CRITIC_WINDOW_S else None
        after = snaps[i] if i < len(snaps) and snap_ts[i] - ts <= _CRITIC_WINDOW_S else None

        window = [e for e in events
                  if abs(e.get("ts", 0) - ts) <= _CRITIC_WINDOW_S
                  and e.get("kind") != "snapshot"]
        ctx: dict = {k: 0 for k in _DISPATCH_KINDS}
        ctx["feeder_nodes"] = set()
        for w in window:
            k = w.get("kind")
            if k in _DISPATCH_KINDS:
                ctx[k] += 1
                if k in ("chunk_remote", "job_remote"):
                    ctx["feeder_nodes"].add(w.get("node", "?"))
        ctx["feeder_nodes"] = sorted(ctx["feeder_nodes"])
        ctx["feeder_active"] = bool(ctx["chunk_remote"] + ctx["job_remote"])

        incidents.append({
            "event": {k: ev.get(k) for k in
                      ("ts", "kind", "node", "label", "stage", "status",
                       "error", "items", "duration_s", "run_id")
                      if ev.get(k) is not None},
            "snapshot_before": _snap_view(before, ts),
            "snapshot_after": _snap_view(after, ts),
            "cluster_context": ctx,
            "diagnosis_hints": _diagnosis_hints(ev, before, after, ctx, window),
        })

    incidents = incidents[-50:]
    recommendations: list[str] = []
    seen: set[str] = set()
    for inc in incidents:
        for h in inc["diagnosis_hints"]:
            key = h.split(":", 1)[0]
            if key not in seen and not h.startswith("no rule matched"):
                seen.add(key)
                recommendations.append(h)
    try:
        import gb_pilot
        analysis = gb_pilot.analyze(events)
        for item in gb_pilot.advise(analysis):
            if item.get("severity") == "ok":
                continue
            rec = f"[pilot:{item['severity']}] {item['action']} (evidence: {item['evidence']})"
            if item.get("lever"):
                rec += f" [lever: {item['lever']}]"
            recommendations.append(rec)
    except Exception:
        pass
    if not recommendations:
        recommendations.append(
            "no incidents or pressure flags in the window , cluster inference "
            "nominal; keep quantization at or above the fp8 quality floor")

    return {
        "window_days": days,
        "events_scanned": len(events),
        "snapshots": len(snaps),
        "incident_count": len(incidents),
        "incidents": incidents,
        "recommendations": recommendations,
    }


def _print_critic_text(rep: dict) -> None:
    print(f"GreenBoost dataflux critic , last {rep['window_days']:g} day(s)  "
          f"({rep['events_scanned']} events, {rep['snapshots']} snapshots, "
          f"{rep['incident_count']} incidents)")
    for inc in rep["incidents"]:
        ev = inc["event"]
        print(f"\n  ── {_fmt_ts(ev.get('ts', 0))}  {ev.get('kind')}  "
              f"node={ev.get('node')}  label={ev.get('label', '')}"
              f"{'  stage=' + str(ev['stage']) if ev.get('stage') else ''}")
        if ev.get("error"):
            print(f"     error: {str(ev['error'])[:160]}")
        if ev.get("items"):
            print(f"     items: {', '.join(str(x) for x in ev['items'][:5])}")
        for tag in ("snapshot_before", "snapshot_after"):
            s = inc[tag]
            if s:
                print(f"     {tag} ({s['offset_s']:+.0f}s): "
                      f"VRAM {s.get('fb_used_pct', 0)}% "
                      f"({s.get('fb_used_mb', 0)}/{s.get('fb_total_mb', 0)} MB)  "
                      f"util {s.get('gpu_util_pct', 0)}%  "
                      f"phase={s.get('shim_phase', '')}  "
                      f"T2 {s.get('t2_allocated_mb', 0)} MB (P{s.get('t2_pressure', 0)})  "
                      f"T3 {s.get('t3_used_mb', 0)} MB  "
                      f"KV {s.get('kv_used_mb', 0)} MB")
            else:
                print(f"     {tag}: none within ±{int(_CRITIC_WINDOW_S)}s")
        ctx = inc["cluster_context"]
        print(f"     cluster: remote={ctx['chunk_remote'] + ctx['job_remote']} "
              f"local={ctx['chunk_local'] + ctx['job_local']} "
              f"feeders={ctx['feeder_nodes'] or '-'}")
        for h in inc["diagnosis_hints"]:
            print(f"     hint: {h}")
    print("\nRECOMMENDATIONS")
    for r in rep["recommendations"]:
        print(f"  * {r}")


# ── HTML rendering ───────────────────────────────────────────────────────────

def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _sparkline_svg(series: list[float], color: str, height: int = 40) -> str:
    """Minimal inline SVG polyline sparkline, 0-100 scale, no JS/CDN deps."""
    if not series:
        return ""
    width = max(len(series) * 3, 120)
    pts = " ".join(
        f"{i * width / max(len(series) - 1, 1):.1f},{height - (v / 100.0 * height):.1f}"
        for i, v in enumerate(series)
    )
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           f'preserveAspectRatio="none" style="display:block">'
           f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{pts}"/></svg>')


def _kv_cell(ev: dict) -> str:
    """kv_reserve_mb is 0 for any process that explicitly disabled KV
    reservation (GREENBOOST_KV_RESERVE_MB=0 , every diffusion/image/video
    workload profile, see gb_cluster.py's _WORKLOAD_PROFILES / forge/gpu.py's
    _GB_DIFFUSION_ENV: "diffusion has no KV cache; give all VRAM to
    weights"). Showing a bare "0/0 MB" there reads as broken telemetry
    instead of an accurate "not applicable" , this is the single most
    common support question this dashboard gets, so spell it out."""
    reserve = ev.get("kv_reserve_mb", 0)
    if not reserve:
        return "N/A (no KV workload)"
    return f"{ev.get('kv_used_mb', 0)}/{reserve} MB"


def _render_snapshot_section(snap_events: list[dict]) -> str:
    """Continuous system-state 'flight recorder' view: sparklines for VRAM
    used% and GPU util% over the window, plus a recent-samples table , the
    closest thing to "recording a video" of the system in a static HTML page."""
    if not snap_events:
        return ("<h2>System snapshots (VRAM / GPU util / KV pressure over time)</h2>"
               "<p style='opacity:.6'>No snapshots yet , starts automatically "
               "when GREENBOOST_ACTIVE=1 (gb_init.py), works with zero feeders.</p>")
    tail = snap_events[-300:]
    vram_series = [e.get("fb_used_pct", 0.0) for e in tail]
    util_series = [e.get("gpu_util_pct", 0.0) for e in tail]
    cpu_series = [e.get("cpu_util_pct", 0.0) for e in tail]
    last = snap_events[-1]
    rows = "".join(
        f"<tr><td>{_fmt_ts(ev.get('ts', 0))}</td><td>{_esc(ev.get('node', '?'))}</td>"
        f"<td>{ev.get('fb_used_mb', 0)}/{ev.get('fb_total_mb', 0)} MB "
        f"({ev.get('fb_used_pct', 0):.0f}%)</td>"
        f"<td>{ev.get('gpu_util_pct', 0):.0f}%</td>"
        f"<td>{ev.get('cpu_util_pct', 0):.0f}%</td>"
        f"<td>{ev.get('temp_c', 0):.0f}°C</td><td>{ev.get('power_w', 0):.0f}W</td>"
        f"<td>{_kv_cell(ev)}</td>"
        f"<td>{ev.get('t2_allocated_mb', 0)} MB (P{ev.get('t2_pressure', 0)})</td>"
        f"<td>{_esc(ev.get('shim_phase', ''))}</td></tr>"
        for ev in reversed(snap_events[-100:])
    )
    return f"""<h2>System snapshots (VRAM / GPU util / KV pressure over time)</h2>
<div class="cards">
  <div class="card"><div class="n">{last.get('fb_used_pct', 0):.0f}%</div><div class="l">VRAM used (latest)</div></div>
  <div class="card"><div class="n">{last.get('gpu_util_pct', 0):.0f}%</div><div class="l">GPU util (latest)</div></div>
  <div class="card"><div class="n">{last.get('cpu_util_pct', 0):.0f}%</div><div class="l">CPU util (latest)</div></div>
  <div class="card"><div class="n">{_kv_cell(last)}</div><div class="l">KV cache used</div></div>
  <div class="card"><div class="n">{last.get('t2_pressure', 0)}</div><div class="l">T2 pressure</div></div>
</div>
<div style="margin: 8px 0">
  <div style="opacity:.6;font-size:12px">VRAM used % (last {len(tail)} samples)</div>
  {_sparkline_svg(vram_series, "#e08030")}
  <div style="opacity:.6;font-size:12px;margin-top:6px">GPU util % (last {len(tail)} samples)</div>
  {_sparkline_svg(util_series, "#3080e0")}
  <div style="opacity:.6;font-size:12px;margin-top:6px">CPU util % (last {len(tail)} samples)</div>
  {_sparkline_svg(cpu_series, "#30b080")}
</div>
<div class="wrap"><table>
<tr><th>Time</th><th>Node</th><th>VRAM</th><th>GPU util</th><th>CPU util</th><th>Temp</th><th>Power</th>
<th>KV cache</th><th>T2 pool</th><th>Phase</th></tr>
{rows}
</table></div>"""


def render_html(events: list[dict], days: float) -> str:
    summary = summarize(events)
    def _rate(d: dict) -> str:
        return f"{d['items'] / d['duration_s']:.2f}" if d["duration_s"] > 0 else "-"

    node_rows = "".join(
        f"<tr><td>{_esc(node)}</td><td>{d['events']}</td><td>{d['items']}</td>"
        f"<td>{d['duration_s']:.1f}s</td><td>{_rate(d)}</td>"
        f"<td class='{'err' if d['errors'] else ''}'>{d['errors']}</td></tr>"
        for node, d in sorted(summary["nodes"].items())
    )

    label_rows = "".join(
        f"<tr><td>{_esc(label)}</td><td>{count}</td></tr>"
        for label, count in sorted(summary["labels"].items(), key=lambda kv: -kv[1])
    )

    run_rows = "".join(
        f"<tr><td>{_esc(run_id)}</td><td>{_esc(r['label'])}</td>"
        f"<td>{_fmt_ts(r['started'])}</td>"
        f"<td>{r['finished'] - r['started']:.1f}s</td>"
        f"<td>{r['items']}</td><td>{_esc(', '.join(r['nodes']))}</td></tr>"
        for run_id, r in sorted(summary["runs"].items(), key=lambda kv: -kv[1]["started"])
    )

    tok_s_rows = "".join(
        f"<tr><td>{_esc(model)}</td><td>{d['latest']:.1f}</td><td>{d['avg']:.1f}</td>"
        f"<td>{d['samples']}</td><td>{_fmt_ts(d['last_ts'])}</td></tr>"
        for model, d in sorted(summary["tok_s"].items(), key=lambda kv: -kv[1]["last_ts"])
    )

    quantize_events = [e for e in events if e.get("kind") in ("quantize", "quantize_to_fit")]
    quantize_rows = "".join(
        f"<tr><td>{_fmt_ts(ev.get('ts', 0))}</td><td>{_esc(ev.get('node', '?'))}</td>"
        f"<td>{_esc(ev.get('model_id', '?'))}</td>"
        f"<td>{_esc(', '.join(str(x) for x in ev.get('items', [])))}</td>"
        f"<td>{_esc(ev.get('bits', '?'))}</td>"
        f"<td>{ev.get('budget_gb', '')}</td><td>{ev.get('total_quant_gb', '')}</td>"
        f"<td>{ev.get('duration_s', 0):.2f}s</td></tr>"
        for ev in reversed(quantize_events[-200:])
    )

    tier_events = [e for e in events if e.get("kind") == "tier_move"]
    tier_rows = "".join(
        f"<tr><td>{_fmt_ts(ev.get('ts', 0))}</td><td>{_esc(ev.get('node', '?'))}</td>"
        f"<td>{_esc(', '.join(str(x) for x in ev.get('items', [])))}</td>"
        f"<td>{_esc(ev.get('from_tier', '?'))} → {_esc(ev.get('to_tier', '?'))}</td>"
        f"<td>{ev.get('size_gb', 0):.3f} GiB</td>"
        f"<td>{ev.get('duration_s', 0):.2f}s</td></tr>"
        for ev in reversed(tier_events[-200:])
    )

    tq_events = [e for e in events if e.get("kind") == "turboquant_activate"]
    tq_rows = "".join(
        f"<tr><td>{_fmt_ts(ev.get('ts', 0))}</td><td>{_esc(ev.get('node', '?'))}</td>"
        f"<td>{_esc(ev.get('mode', '?'))}</td>"
        f"<td>k={ev.get('k_bits', '?')} v={ev.get('v_bits', '?')}</td>"
        f"<td>{_esc(ev.get('device', '?'))}</td>"
        f"<td>{'yes' if ev.get('sparse_v') else 'no'}</td></tr>"
        for ev in reversed(tq_events[-200:])
    )

    snap_events = [e for e in events if e.get("kind") == "snapshot"]
    snapshot_section = _render_snapshot_section(snap_events)

    event_rows = "".join(
        f"<tr class='{_esc(ev.get('status', 'ok'))}'>"
        f"<td>{_fmt_ts(ev.get('ts', 0))}</td>"
        f"<td>{_esc(ev.get('node', '?'))}</td>"
        f"<td>{_esc(ev.get('label', '?'))}</td>"
        f"<td>{_esc(ev.get('kind', '?'))}</td>"
        f"<td>{ev.get('n_items', 0)}</td>"
        f"<td>{ev.get('duration_s', 0):.1f}s</td>"
        f"<td>{_esc(ev.get('status', 'ok'))}</td>"
        f"<td>{_esc(', '.join(str(x) for x in ev.get('items', [])[:8]))}"
        f"{'…' if len(ev.get('items', [])) > 8 else ''}</td>"
        f"<td>{_esc(ev.get('error', ''))}</td></tr>"
        for ev in reversed(events[-500:])  # most recent first, cap at 500 rows
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>GreenBoost Dataflux</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font: 14px/1.5 -apple-system, system-ui, sans-serif; margin: 0; padding: 24px;
        background: Canvas; color: CanvasText; }}
h1 {{ font-size: 20px; margin: 0 0 4px; }}
h2 {{ font-size: 15px; margin: 28px 0 8px; opacity: .85; }}
.sub {{ opacity: .6; margin-bottom: 24px; }}
.cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 8px; }}
.card {{ border: 1px solid color-mix(in srgb, CanvasText 20%, transparent);
         border-radius: 8px; padding: 12px 18px; min-width: 120px; }}
.card .n {{ font-size: 24px; font-weight: 600; }}
.card .l {{ opacity: .6; font-size: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 8px; font-size: 13px; }}
th, td {{ text-align: left; padding: 5px 10px; border-bottom: 1px solid
          color-mix(in srgb, CanvasText 12%, transparent); }}
th {{ opacity: .7; font-weight: 600; }}
tr.error td, td.err {{ color: #e05555; }}
.wrap {{ overflow-x: auto; }}
</style></head>
<body>
<div id="gb-root">
<h1>GreenBoost Dataflux</h1>
<div class="sub">Cluster dispatch activity , last {days:g} day(s) , <span id="gb-live-ts">live, updated {_fmt_ts(time.time())}</span></div>

<div class="cards">
  <div class="card"><div class="n">{summary['event_count']}</div><div class="l">events</div></div>
  <div class="card"><div class="n">{summary['total_items']}</div><div class="l">items processed</div></div>
  <div class="card"><div class="n">{summary['total_duration_s']:.0f}s</div><div class="l">total compute time</div></div>
  <div class="card"><div class="n">{len(summary['nodes'])}</div><div class="l">nodes active</div></div>
  <div class="card"><div class="n">{summary['errors']}</div><div class="l">errors</div></div>
</div>

<h2>Per-node throughput</h2>
<div class="wrap"><table>
<tr><th>Node</th><th>Events</th><th>Items</th><th>Compute time</th><th>Items/s</th><th>Errors</th></tr>
{node_rows or '<tr><td colspan=6>No events yet , run something through gb_cluster.cluster_map or ClusterJobQueue.</td></tr>'}
</table></div>

<h2>By script / label</h2>
<div class="wrap"><table>
<tr><th>Label</th><th>Items</th></tr>
{label_rows or '<tr><td colspan=2>-</td></tr>'}
</table></div>

<h2>Runs (one row per script invocation)</h2>
<div class="wrap"><table>
<tr><th>Run ID</th><th>Label</th><th>Started</th><th>Wall time</th><th>Items</th><th>Nodes used</th></tr>
{run_rows or '<tr><td colspan=6>-</td></tr>'}
</table></div>

{snapshot_section}

<h2>Measured throughput (gb_synapse.py) , real, client-observed tok/s</h2>
<div class="wrap"><table>
<tr><th>Model</th><th>Latest tok/s</th><th>Rolling avg</th><th>Samples</th><th>Last seen</th></tr>
{tok_s_rows or '<tr><td colspan=5>No measured tok/s samples yet , recorded by greenboost-cli after each gb-synapse turn.</td></tr>'}
</table></div>

<h2>Quantization (gb_quant.py)</h2>
<div class="wrap"><table>
<tr><th>Time</th><th>Node</th><th>Model</th><th>Component</th><th>Bits/quality</th>
<th>Budget GiB</th><th>Total GiB</th><th>Duration</th></tr>
{quantize_rows or '<tr><td colspan=8>No quantization events yet.</td></tr>'}
</table></div>

<h2>TurboQuant KV-cache activations (gb_attn.py)</h2>
<div class="wrap"><table>
<tr><th>Time</th><th>Node</th><th>Mode</th><th>K/V bits</th><th>Device</th><th>Sparse-V</th></tr>
{tq_rows or '<tr><td colspan=6>No TurboQuant activations yet.</td></tr>'}
</table></div>

<h2>Model tier movements , T1 (GPU) / T2 (DDR) / T3 (NVMe) (gb_model_tier.py)</h2>
<div class="wrap"><table>
<tr><th>Time</th><th>Node</th><th>Model</th><th>Move</th><th>Size</th><th>Duration</th></tr>
{tier_rows or '<tr><td colspan=6>No tier movements yet.</td></tr>'}
</table></div>

<h2>Recent events (max 500)</h2>
<div class="wrap"><table>
<tr><th>Time</th><th>Node</th><th>Label</th><th>Kind</th><th>Items</th><th>Duration</th><th>Status</th><th>Item slugs</th><th>Error</th></tr>
{event_rows or '<tr><td colspan=9>-</td></tr>'}
</table></div>
</div>
<script>
(function() {{
  var REFRESH_MS = 5000;
  async function tick() {{
    if (document.hidden) return;
    try {{
      var res = await fetch(location.href, {{cache: "no-store"}});
      var html = await res.text();
      var doc = new DOMParser().parseFromString(html, "text/html");
      var newRoot = doc.getElementById("gb-root");
      var root = document.getElementById("gb-root");
      if (newRoot && root) {{
        var sx = window.scrollX, sy = window.scrollY;
        root.innerHTML = newRoot.innerHTML;
        window.scrollTo(sx, sy);
      }}
    }} catch (e) {{ /* server restart / transient network blip , try again next tick */ }}
  }}
  setInterval(tick, REFRESH_MS);
}})();
</script>
</body></html>"""


# ── HTTP server ───────────────────────────────────────────────────────────

class _Handler(http.server.BaseHTTPRequestHandler):
    days = DEFAULT_DAYS

    def log_message(self, fmt, *args):  # quiet the default stderr access log
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        days = float(qs.get("days", [self.days])[0])
        events = read_events(since_hours=days * 24)

        if parsed.path == "/api/events.json":
            body = json.dumps(events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/summary.json":
            body = json.dumps(summarize(events)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = render_html(events, days).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Threaded + SO_REUSEADDR , so the page's own 5s auto-refresh fetch never
    serializes behind a slow request, and restarting the server doesn't hit
    TIME_WAIT ("Address already in use")."""
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        # A browser tab closed/reloaded mid-response (or the 5s auto-refresh
        # fetch got cancelled) is normal traffic, not a server bug , don't
        # spam a traceback to the console for it.
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def serve(port: int = DEFAULT_PORT, days: float = DEFAULT_DAYS) -> None:
    _Handler.days = days
    with _ThreadingHTTPServer(("127.0.0.1", port), _Handler) as httpd:
        print(f"[gb_dataflux] serving http://127.0.0.1:{port}  "
              f"(log: {_log_path()}, window: last {days:g} day(s))", flush=True)
        print("[gb_dataflux] Ctrl-C to stop", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[gb_dataflux] stopped", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="GreenBoost cluster dataflux log/UI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="launch the dataflux web UI")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--days", type=float, default=DEFAULT_DAYS)

    p = sub.add_parser("summary", help="print a text/JSON summary")
    p.add_argument("--days", type=float, default=DEFAULT_DAYS)
    p.add_argument("--llm", action="store_true", help="JSON output")

    p = sub.add_parser("critic", help="snapshot-correlated incident diagnosis")
    p.add_argument("--days", type=float, default=1.0)
    p.add_argument("--llm", action="store_true", help="JSON output")

    args = ap.parse_args()
    if args.cmd == "serve":
        serve(port=args.port, days=args.days)
        return 0

    if args.cmd == "critic":
        rep = critic_report(days=args.days)
        if args.llm:
            print(json.dumps(rep))
        else:
            _print_critic_text(rep)
        return 0

    events = read_events(since_hours=args.days * 24)
    s = summarize(events)
    if args.llm:
        print(json.dumps(s))
        return 0
    print(f"GreenBoost dataflux , last {args.days:g} day(s)  ({_log_path()})")
    print(f"  events={s['event_count']}  items={s['total_items']}  "
          f"compute_time={s['total_duration_s']:.0f}s  errors={s['errors']}")
    for node, d in sorted(s["nodes"].items()):
        rate = f"{d['items'] / d['duration_s']:.2f} items/s" if d["duration_s"] else "-"
        print(f"  {node:<24} events={d['events']:<4} items={d['items']:<4} "
              f"time={d['duration_s']:.1f}s  {rate}  errors={d['errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

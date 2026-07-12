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
        telemetry_manager.add_callback(self._on_metrics)

    def _on_metrics(self, m) -> None:
        now = time.time()
        if now - self._last_emit < self._interval_s:
            return
        self._last_emit = now
        gb = getattr(m, "gb", None)
        sys_m = getattr(m, "sys", None)
        node = self._node if self._node is not None else (
            "host" if getattr(m, "device", 0) == 0 else f"gpu{m.device}")
        emit({
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
        })


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
        if ev.get("kind") == "tok_s_measured":
            model = ev.get("model", "?")
            t = tok_s.setdefault(model, {"latest": 0.0, "samples": 0, "_sum": 0.0, "last_ts": 0})
            t["latest"] = ev.get("tok_s", 0.0)
            t["samples"] += 1
            t["_sum"] += ev.get("tok_s", 0.0)
            t["last_ts"] = ev.get("ts", 0)

        if ev.get("kind") == "stage_profile":
            stage = ev.get("stage", "?")
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

    args = ap.parse_args()
    if args.cmd == "serve":
        serve(port=args.port, days=args.days)
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

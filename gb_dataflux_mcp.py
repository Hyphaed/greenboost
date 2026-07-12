#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_dataflux_mcp.py , MCP server exposing GreenBoost's cluster dataflux log.

Lets an LLM (or any MCP client) query "how did data flow through the
greenboost cluster" without shelling out or scraping the web UI: same JSONL
log (gb_dataflux.py) that backs `greenboost dataflux-ui`, read-only.

Tools:
    dataflux_summary       , cheap aggregate overview (nodes/labels/runs/tok_s)
    dataflux_events         , raw events, filterable by node/label/kind/status
    dataflux_errors         , failed dispatches only
    dataflux_kinds          , event-kind breakdown (what actually happened)
    dataflux_tier_moves     , T1/T2/T3 promote/demote/evict events
    dataflux_quantization   , quantize / quantize_to_fit decisions
    dataflux_tok_s          , measured (not predicted) tokens/sec, per model
    greenboost_capabilities , installed/running shim feature manifest (gb_monitor)
    greenboost_status       , live tier/phase/KV-prefetch snapshot (gb_monitor)
    greenboost_pilot        , stage/tok_s trends + evidence-backed advice (gb_pilot)

Run standalone (stdio transport, for `.mcp.json` registration):
    python3 gb_dataflux_mcp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gb_dataflux as gdf  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("greenboost-dataflux")


@mcp.tool()
def dataflux_summary(days: float = 5.0) -> dict:
    """Aggregate summary of GreenBoost cluster dispatch activity over the
    last `days` days: total events/items/compute-time, per-node throughput
    and error counts, per-script (label) item counts, and one row per
    script invocation (run). Use this first , it's the cheap overview
    before pulling raw events."""
    events = gdf.read_events(since_hours=days * 24)
    return gdf.summarize(events)


@mcp.tool()
def dataflux_events(days: float = 5.0, node: str | None = None,
                    label: str | None = None, kind: str | None = None,
                    status: str | None = None, stage: str | None = None,
                    limit: int = 200) -> list[dict]:
    """Raw dataflux events over the last `days` days, most recent first.

    Filter by node (e.g. "host" or a feeder's hostname/ip), label (the
    script/workload name a caller passed to cluster_map/ClusterJobQueue,
    e.g. "gen_image_batch.py"), kind (the event category, e.g. "tier_move",
    "quantize", "quantize_to_fit", "turboquant_activate", "tok_s_measured",
    "chunk_local", "chunk_remote", "job_local", "job_remote", "model_push",
    "stage_bundle", "snapshot" , see `dataflux_kinds` for what's actually
    present), status ("ok" or "error"), and/or stage — the stage_profile
    stage name, substring match (e.g. "forge:image", "conduir:batch",
    "artloop"), the fast path to one pipeline stage's timing series.
    `limit` caps the number of events returned (default 200, most recent
    first).
    """
    events = gdf.read_events(since_hours=days * 24)
    if node:
        events = [e for e in events if e.get("node") == node]
    if label:
        events = [e for e in events if e.get("label") == label]
    if kind:
        events = [e for e in events if e.get("kind") == kind]
    if status:
        events = [e for e in events if e.get("status") == status]
    if stage:
        events = [e for e in events if stage in str(e.get("stage", ""))]
    return list(reversed(events))[:limit]


@mcp.tool()
def dataflux_errors(days: float = 5.0, limit: int = 50) -> list[dict]:
    """Every FAILED dispatch (remote exceptions, model-push failures, stage
    failures) in the last `days` days, most recent first , the fast path
    to "what broke recently and where"."""
    events = gdf.read_events(since_hours=days * 24)
    errors = [e for e in events if e.get("status") == "error"]
    return list(reversed(errors))[:limit]


@mcp.tool()
def dataflux_kinds(days: float = 5.0) -> dict:
    """Breakdown of how many events of each kind were logged in the last
    `days` days, with the most recent timestamp for each kind, sorted by
    count descending. The dataflux log holds several distinct kinds of
    activity (tier moves, quantization decisions, measured tok/s, cluster
    job chunks, model pushes, snapshots, ...); use this first to see what
    actually happened before drilling into any one kind with
    `dataflux_events(kind=...)`."""
    events = gdf.read_events(since_hours=days * 24)
    kinds: dict[str, dict] = {}
    for ev in events:
        k = ev.get("kind", "?")
        d = kinds.setdefault(k, {"count": 0, "last_ts": 0.0})
        d["count"] += 1
        d["last_ts"] = max(d["last_ts"], ev.get("ts", 0.0))
    return dict(sorted(kinds.items(), key=lambda kv: -kv[1]["count"]))


@mcp.tool()
def dataflux_tier_moves(days: float = 5.0, limit: int = 100) -> list[dict]:
    """Memory-tier movement events (T1/T2/T3 promote, demote, evict) emitted
    by `gb_model_tier.py` over the last `days` days, most recent first."""
    events = gdf.read_events(since_hours=days * 24)
    moves = [e for e in events if e.get("kind") == "tier_move"]
    return list(reversed(moves))[:limit]


@mcp.tool()
def dataflux_quantization(days: float = 5.0, limit: int = 100) -> list[dict]:
    """Quantization decisions emitted by `gb_quant.py` (component quantized,
    bits/quality chosen, budget vs. actual GiB) over the last `days` days,
    most recent first. Covers both per-component `quantize` events and
    whole-pipeline `quantize_to_fit` events."""
    events = gdf.read_events(since_hours=days * 24)
    q = [e for e in events if e.get("kind") in ("quantize", "quantize_to_fit")]
    return list(reversed(q))[:limit]


@mcp.tool()
def dataflux_tok_s(days: float = 5.0, model: str | None = None) -> dict:
    """Measured tokens/sec , the real, client-observed decode speed fed by
    `record_measured_tok_s()` after each turn, not a predicted or
    theoretical number.

    Without `model`: per-model latest/average/sample-count, one entry per
    model seen in the last `days` days (same rollup `dataflux_summary`
    reports under its "tok_s" key). With `model`: that rollup plus the raw
    `tok_s_measured` event series for just that model, most recent first ,
    useful for spotting a throughput regression over time rather than just
    the latest sample."""
    events = gdf.read_events(since_hours=days * 24)
    rollup = gdf.summarize(events)["tok_s"]
    if model is None:
        return rollup
    series = [e for e in events
              if e.get("kind") == "tok_s_measured" and e.get("model") == model]
    return {"model": model, "stats": rollup.get(model, {}),
            "series": list(reversed(series))}


@mcp.tool()
def greenboost_pilot(days: float = 5.0) -> dict:
    """The pilot's instrument panel (gb_pilot): per-stage wall-time trends
    (stage_profile events from ai-forge's jobqueue/seedlog), measured tok/s
    trends per model, latest memory-pressure picture, and evidence-backed
    advice mapping findings to GbControl levers or config changes. Read-only —
    advice is never auto-applied. Use this to decide how to redirect
    orchestration (re-quant, enable GB_TQ_ATTN, move a lever) from real data."""
    import gb_pilot
    events = gdf.read_events(since_hours=days * 24)
    analysis = gb_pilot.analyze(events)
    analysis["advice"] = gb_pilot.advise(analysis)
    return analysis


@mcp.tool()
def greenboost_capabilities() -> dict:
    """What the installed/running GreenBoost shim supports, via gb_monitor's
    capability manifest (runtime /run/greenboost/capabilities.json written by
    the shim at init → install manifest → binary sniff). Keys: shim_version,
    abi, source, features{gb_quant_cudart_rebind, expert_pool, cluster_fabric,
    gds, kv_compress, report_physical_vram}. Use before assuming a shim feature
    is present rather than sniffing the .so."""
    try:
        import gb_monitor
        return gb_monitor.capabilities()
    except Exception as e:
        return {"error": f"gb_monitor unavailable: {e}", "features": {}}


@mcp.tool()
def greenboost_status() -> dict:
    """Live read-only GreenBoost snapshot via gb_monitor: whether the kmod is
    loaded, GPU, the T1/T2/T3 tier pool (MB) + combined GB, pressure labels,
    OOM/gaming flags, the shim phase + active allocation path, and the
    Phase-4 KV-prefetch counters (kv_prefetch_* in the raw shim map). This is
    the same state `greenboost status --llm` reports."""
    try:
        import gb_monitor
        snap = gb_monitor.snapshot()
        d = snap.as_dict()
        # surface the KV-prefetch metering (shim_stats) alongside the pool view
        d["kv_prefetch"] = {k: v for k, v in snap.shim.items()
                            if k.startswith("kv_prefetch")}
        return d
    except Exception as e:
        return {"error": f"gb_monitor unavailable: {e}"}


if __name__ == "__main__":
    mcp.run()

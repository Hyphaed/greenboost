#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_dataflux_mcp.py , MCP server exposing GreenBoost's cluster dataflux log.

Lets an LLM (or any MCP client) query "how did data flow through the
greenboost cluster" without shelling out or scraping the web UI: same JSONL
log (gb_dataflux.py) that backs `greenboost dataflux-ui`, read-only.

Tools (21 — keep this list in sync with the @mcp.tool defs below):
    dataflux_summary       , cheap aggregate overview (nodes/labels/runs/tok_s/by_kind)
    dataflux_events         , raw events, filterable by node/label/kind/status/from_ts/to_ts/cursor
    dataflux_errors         , failed dispatches only
    dataflux_decisions      , shim tier-placement/spill decisions (kind=shim_decision)
    dataflux_actuations     , orchestrator lever moves (kind=actuation)
    a2a_status              , A2A gateway liveness + recent a2a_request rollup
    dataflux_critic         , snapshot-correlated incident diagnosis + recommendations
    dataflux_topology       , per-node hardware topology events (deduped latest)
    dataflux_kinds          , event-kind breakdown (what actually happened)
    dataflux_schema         , the event-kind REGISTRY , what every kind means (group/fields/incident rules)
    dataflux_group          , every event across a whole registry group (e.g. "shim", "quant") in one call
    dataflux_tier_moves     , T1/T2/T3 promote/demote/evict events
    dataflux_quantization   , quantize / quantize_to_fit decisions
    dataflux_tok_s          , measured (not predicted) tokens/sec, per model
    dataflux_candidates     , best-of-N candidate_selected rollup, per slug
    dataflux_models         , per-model call/rotation rollup + tok_s merged
    greenboost_pilot        , stage/tok_s trends + evidence-backed advice (gb_pilot)
    greenboost_capabilities , installed/running shim feature manifest (gb_monitor)
    greenboost_status       , live tier/phase/KV-prefetch snapshot (gb_monitor)
    synapse_status          , gb-synapse serving state
    tiering_status          , GB-Tiering live state (T1/T2/T3 pool + phase)

Note: greenboost_status/capabilities/pilot/synapse_status are also exposed on
the greenboost-orchestrator server (gb_mcp.py) for "one server suffices"
convenience — same underlying data, so either server answers identically.

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
    before pulling raw events. Canonical owner (shared impl in
    gb_mcp_common.py — also mirrored on greenboost-orchestrator)."""
    import gb_mcp_common
    return gb_mcp_common.dataflux_summary(days=days)


@mcp.tool()
def dataflux_events(days: float = 5.0, node: str | None = None,
                    label: str | None = None, kind: str | None = None,
                    status: str | None = None, stage: str | None = None,
                    limit: int = 200, from_ts: float | None = None,
                    to_ts: float | None = None, cursor: float | None = None) -> list[dict]:
    """Raw dataflux events over the last `days` days, most recent first.

    Filter by node (e.g. "host" or a feeder's hostname/ip), label (the
    script/workload name a caller passed to cluster_map/ClusterJobQueue,
    e.g. "gen_image_batch.py"), kind (the event category , see
    `dataflux_schema()` for what every known kind means, or `dataflux_kinds`
    for what's actually present in this window), status ("ok" or "error"),
    and/or stage — the stage_profile stage name, substring match (e.g.
    "forge:image", "conduir:batch", "artloop"), the fast path to one
    pipeline stage's timing series. `limit` caps the number of events
    returned (default 200, most recent first).

    `from_ts`/`to_ts` (unix seconds) narrow the window with exact bounds
    instead of the coarse `days`-back window , use when correlating against
    a known incident timestamp. `cursor` (a `ts` value from a previous
    call's last returned event) pages further back: pass the oldest event's
    `ts` from one call as `cursor` on the next to continue past `limit`
    without re-scanning from `days` again.
    """
    events = gdf.read_events(since_hours=days * 24)
    if node:
        # Match by canonical node, not the raw string , the SAME physical
        # machine can appear under more than one raw label (e.g. "host" vs
        # this host's real hostname, or a feeder's generic "feeder0"
        # SnapshotRecorder tag vs its real hostname), so a caller filtering
        # by "omen" would otherwise silently miss "feeder0"-tagged events
        # for that exact same feeder (found live 2026-07-14).
        _want = gdf.canonical_node(node)
        events = [e for e in events if gdf.canonical_node(e.get("node")) == _want]
    if label:
        events = [e for e in events if e.get("label") == label]
    if kind:
        events = [e for e in events if e.get("kind") == kind]
    if status:
        events = [e for e in events if e.get("status") == status]
    if stage:
        events = [e for e in events if stage in str(e.get("stage", ""))]
    if from_ts is not None:
        events = [e for e in events if e.get("ts", 0) >= from_ts]
    if to_ts is not None:
        events = [e for e in events if e.get("ts", 0) <= to_ts]
    if cursor is not None:
        events = [e for e in events if e.get("ts", 0) < cursor]
    return list(reversed(events))[:limit]


@mcp.tool()
def dataflux_schema(kind: str | None = None) -> dict:
    """The dataflux event-kind registry (gb_dataflux_kinds.py) , what every
    known kind MEANS: its group, expected fields, which numeric fields are
    worth trending, which statuses make it an incident, and whether it's
    emitted by GreenBoost itself or by a consumer repo (ai-forge). Without
    this, "queryable via dataflux_events(kind=...)" is technically true but
    practically useless , an LLM has to read source to know that e.g.
    "placement" carries `tensor_split`/`floor_bits`. Pass `kind` for one
    entry, omit for the full registry (~40 kinds across 9 groups)."""
    import gb_dataflux_kinds
    return gb_dataflux_kinds.schema(kind)


@mcp.tool()
def dataflux_group(group: str, days: float = 5.0, limit: int = 200,
                   from_ts: float | None = None, to_ts: float | None = None) -> dict:
    """Every event across ALL kinds in one registry group (see
    `dataflux_schema()` for the group each kind belongs to: placement, quant,
    synapse, cluster, shim, pipeline, health, agent, eval), plus a per-kind
    rollup (count/errors/last_ts) , the single tool that makes the ~22
    previously-orphaned kinds (shim_transition, turboquant_activate,
    tensor_split, capacity_fit, synapse_stall, pcie_degraded, health_transition,
    image_gen, and more) queryable without one dedicated tool per kind."""
    import gb_dataflux_kinds
    kinds_in_group = gb_dataflux_kinds.kinds_in_group(group)
    events = gdf.read_events(since_hours=days * 24)
    if from_ts is not None:
        events = [e for e in events if e.get("ts", 0) >= from_ts]
    if to_ts is not None:
        events = [e for e in events if e.get("ts", 0) <= to_ts]
    group_events = [e for e in events if e.get("kind") in kinds_in_group]
    by_kind = gdf.summarize(group_events)["by_kind"]
    return {
        "group": group, "kinds": list(kinds_in_group),
        "event_count": len(group_events), "by_kind": by_kind,
        "events": list(reversed(group_events))[:limit],
    }


@mcp.tool()
def dataflux_errors(days: float = 5.0, limit: int = 50) -> list[dict]:
    """Every FAILED dispatch (remote exceptions, model-push failures, stage
    failures) in the last `days` days, most recent first , the fast path
    to "what broke recently and where"."""
    events = gdf.read_events(since_hours=days * 24)
    errors = [e for e in events if e.get("status") == "error"]
    return list(reversed(errors))[:limit]


@mcp.tool()
def dataflux_decisions(days: float = 2.0, limit: int = 100) -> list[dict]:
    """Per-decision SHIM placement events (`shim_decision`) , every time the
    CUDA shim placed bytes off the fastest tier (T2 DDR spill, T3 NVMe spill,
    feeder T1, host-VMM fallback, path-B host-register), with the tier chosen,
    bytes, reason, and the physical VRAM fill % at that moment. Closes the
    former blind spot where shim allocation decisions were NVTX/log-only and
    invisible to the MCP. Most recent first."""
    events = gdf.read_events(since_hours=days * 24)
    decs = [e for e in events if e.get("kind") == "shim_decision"]
    return list(reversed(decs))[:limit]


@mcp.tool()
def dataflux_actuations(days: float = 2.0, limit: int = 100) -> list[dict]:
    """Reactive-orchestrator ACTUATION events (`actuation`) , every lever the
    orchestrator moved (kv_grow, tier auto-evict, clock/power cap, VM/PSI
    tuning), with the loop name (`lever`), the loop's fields, and whether it
    was gated (`GB_ORCH_ACTUATE`). This is the one true telemetry→decision
    loop, now correlatable with the snapshot that triggered it. Most recent
    first."""
    events = gdf.read_events(since_hours=days * 24)
    acts = [e for e in events if e.get("kind") == "actuation"]
    return list(reversed(acts))[:limit]


@mcp.tool()
def a2a_status(days: float = 1.0) -> dict:
    """GreenBoost A2A gateway state: whether the AgentCard/JSON-RPC endpoint is
    listening (GB_A2A_BIND, default 127.0.0.1:8790), the advertised skills, and
    a summary of recent `a2a_request` events (verb, gated/dry-run, outcome).
    Keeps the A2A control plane observable like every other subsystem."""
    import os
    import socket
    bind = os.environ.get("GB_A2A_BIND", "127.0.0.1:8790")
    host, _, port = bind.rpartition(":")
    listening = False
    try:
        with socket.create_connection((host or "127.0.0.1", int(port or "8790")),
                                      timeout=0.5):
            listening = True
    except OSError:
        pass
    skills = []
    try:
        import gb_actuation
        skills = list(gb_actuation.VERBS)
    except Exception:
        pass
    events = gdf.read_events(since_hours=days * 24)
    reqs = [e for e in events if e.get("kind") == "a2a_request"]
    by_outcome: dict[str, int] = {}
    for e in reqs:
        by_outcome[e.get("outcome", "?")] = by_outcome.get(e.get("outcome", "?"), 0) + 1
    return {"bind": bind, "listening": listening, "skills": skills,
            "requests_window": len(reqs), "by_outcome": by_outcome,
            "recent": list(reversed(reqs))[:20]}


@mcp.tool()
def dataflux_topology(days: float = 30.0, node: str | None = None) -> list[dict]:
    """node_topology events , the STATIC hardware identity of every node that
    has connected or (re)generated its profile: GPU model, VRAM GB, compute
    capability, P/E-core counts, RAM total/speed, PCIe gen/lanes, NVMe. Deduped
    to the latest event per node, most recent first. This is the event-LOG view
    (history of what hardware was seen, and when); for the live current-state
    view use the greenboost-cluster MCP's `cluster_topology` tool. Filter to one
    node with `node`. Default window is 30 days since topology changes rarely."""
    events = gdf.read_events(since_hours=days * 24)
    topo = [e for e in events if e.get("kind") == "node_topology"]
    if node:
        _want = gdf.canonical_node(node)
        topo = [e for e in topo if gdf.canonical_node(e.get("node")) == _want]
    latest: dict[str, dict] = {}
    for e in topo:
        n = gdf.canonical_node(e.get("node", "?"))
        if n not in latest or e.get("ts", 0.0) >= latest[n].get("ts", 0.0):
            latest[n] = e
    return sorted(latest.values(), key=lambda e: -e.get("ts", 0.0))


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
    """Memory-tier movement events, most recent first, from BOTH real sources
    of tier placement on this node (each event carries a `source` field so a
    caller can tell them apart):

    - `tier_move` , explicit Python-API moves via `gb_model_tier.py`'s
      manual T1/T2/T3 promote/demote/evict calls. NOT currently invoked by
      any ai-forge pipeline (verified 2026-07-14 via repo grep) , this half
      is usually empty and that is expected, not a recording bug.
    - `shim_decision` , the ACTIVE, automatic source: every time the CUDA
      shim's smart allocator placed a buffer off the fastest tier (T2 DDR
      spill, T3 NVMe spill, feeder T1, host-VMM fallback). This is where the
      real, high-volume tier-placement telemetry actually lives today.

    Previously this tool only queried `tier_move` and so returned `[]` on
    every real box (that mechanism has zero callers) even though thousands of
    real per-allocation tier decisions were being recorded every session
    under `shim_decision` , a misleading "nothing to see" result masking a
    real, rich data source. Fixed 2026-07-14 to surface both. Use
    `fb_phys_used_pct` on `shim_decision` rows to spot Rule #1 violations
    (a large `bytes_mb` spill to `t2_local` while `fb_phys_used_pct` is low
    means physical VRAM was under-filled before overflowing — see
    `GB_VRAM_FRONTLOAD` in `greenboost_cuda_shim.c`)."""
    events = gdf.read_events(since_hours=days * 24)
    moves = [dict(e, source="tier_move") for e in events if e.get("kind") == "tier_move"]
    decisions = [dict(e, source="shim_decision") for e in events if e.get("kind") == "shim_decision"]
    merged = sorted(moves + decisions, key=lambda e: e.get("ts", 0.0))
    return list(reversed(merged))[:limit]


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
def dataflux_models(days: float = 5.0) -> dict:
    """Per-model usage rollup over the last `days` days, from the
    kind="model_call" events forge pipelines emit (forge/gb_models.py) and
    the kind="model_rotation" phase events gb_rotator.py emits: call count,
    error count, mean duration and last-seen timestamp per model, with the
    measured tok/s rollup (same as `dataflux_tok_s`) merged in per model.
    The overnight-autonomy view: which models actually got used, which
    failed, and how fast they really decoded."""
    events = gdf.read_events(since_hours=days * 24)
    models: dict[str, dict] = {}
    for ev in events:
        if ev.get("kind") not in ("model_call", "model_rotation"):
            continue
        m = ev.get("model", "?")
        d = models.setdefault(m, {"calls": 0, "errors": 0,
                                  "total_duration_s": 0.0, "last_ts": 0.0})
        d["calls"] += 1
        if ev.get("status") == "error":
            d["errors"] += 1
        d["total_duration_s"] += ev.get("duration_s", 0.0) or 0.0
        d["last_ts"] = max(d["last_ts"], ev.get("ts", 0.0))
    tok_s = gdf.summarize(events)["tok_s"]
    for m, d in models.items():
        d["mean_duration_s"] = round(d["total_duration_s"] / d["calls"], 2)
        del d["total_duration_s"]
        if m in tok_s:
            d["tok_s"] = tok_s[m]
    return dict(sorted(models.items(), key=lambda kv: -kv[1]["calls"]))


@mcp.tool()
def dataflux_critic(days: float = 1.0) -> dict:
    """Critic report: snapshot-correlated diagnosis of what was happening
    during cluster inference. Every incident (error events, shim_transition
    warns) in the last `days` days is returned with the nearest flight-recorder
    snapshots within ±60s before/after (VRAM used/free %, GPU util, shim
    phase, T2/T3 pool state, KV), the cluster activity in that window
    (chunk_remote/chunk_local counts , was the feeder working?), and
    rule-based diagnosis_hints (T3 placement-floor breach, Rule#1 T1
    underfill, idle-feeder cluster rule, INIT-phase OOM collisions, below-fp8
    quality-floor breaches). Ends with merged `recommendations` (incident
    hints + gb_pilot advice) toward best cluster inference at ≥fp8 gb-quant
    quality. Use after any failed/slow run before tuning anything."""
    return gdf.critic_report(days=days)


@mcp.tool()
def dataflux_candidates(days: float = 5.0) -> dict:
    """Rollup of best-of-N `candidate_selected` events emitted by ai-forge's
    run_pilot best-of reroll (FORGE_BEST_OF). One row per slug over the last
    `days` days: how many best-of runs it took (attempts), total seed
    candidates tried, mean winner score, how many runs fully passed QC, total
    unresolved-text lines left for the repair/human pass, and the breakdown of
    chosen_reason (incl. "skipped_systemic_repeat" for slugs whose repeated
    failures were systemic and skipped). Use to see which slugs keep costing
    candidates and which are systemically stuck. Returns {} when no best-of
    events exist yet."""
    events = gdf.read_events(since_hours=days * 24)
    cs = [e for e in events if e.get("kind") == "candidate_selected"]
    if not cs:
        return {}
    per_slug: dict[str, dict] = {}
    for e in cs:
        slug = e.get("slug", "?")
        d = per_slug.setdefault(slug, {
            "attempts": 0, "candidates_tried": 0, "_score_sum": 0.0,
            "_scored": 0, "passed": 0, "unresolved_total": 0,
            "reasons": {}, "last_ts": 0.0})
        d["attempts"] += 1
        d["candidates_tried"] += e.get("n_tried", 0) or 0
        ws = e.get("winner_score")
        if ws is not None:
            d["_score_sum"] += ws
            d["_scored"] += 1
        if e.get("passed"):
            d["passed"] += 1
        d["unresolved_total"] += e.get("texts_unresolved_count", 0) or 0
        reason = e.get("chosen_reason", "?")
        d["reasons"][reason] = d["reasons"].get(reason, 0) + 1
        d["last_ts"] = max(d["last_ts"], e.get("ts", 0.0))
    for d in per_slug.values():
        d["mean_winner_score"] = (round(d.pop("_score_sum") / d["_scored"], 1)
                                  if d["_scored"] else None)
        del d["_scored"]
    return {"event_count": len(cs),
            "slugs": dict(sorted(per_slug.items(),
                                 key=lambda kv: -kv[1]["last_ts"]))}


@mcp.tool()
def greenboost_pilot(days: float = 5.0) -> dict:
    """The pilot's instrument panel (gb_pilot): per-stage wall-time trends
    (stage_profile events from ai-forge's jobqueue/seedlog), measured tok/s
    trends per model, latest memory-pressure picture, and evidence-backed
    advice mapping findings to GbControl levers or config changes. Read-only —
    advice is never auto-applied. Use this to decide how to redirect
    orchestration (re-quant, enable GB_TQ_ATTN, move a lever) from real data.
    Canonical owner (shared impl in gb_mcp_common.py — also mirrored on
    greenboost-orchestrator)."""
    import gb_mcp_common
    return gb_mcp_common.greenboost_pilot(days=days)


@mcp.tool()
def greenboost_capabilities() -> dict:
    """What the installed/running GreenBoost shim supports, via gb_monitor's
    capability manifest (runtime /run/greenboost/capabilities.json written by
    the shim at init → install manifest → binary sniff). Keys: shim_version,
    abi, source, features{gb_quant_cudart_rebind, expert_pool, cluster_fabric,
    gds, kv_compress, report_physical_vram}. Use before assuming a shim feature
    is present rather than sniffing the .so. Canonical owner (shared impl in
    gb_mcp_common.py — also mirrored on greenboost-orchestrator)."""
    import gb_mcp_common
    return gb_mcp_common.greenboost_capabilities()


@mcp.tool()
def greenboost_status() -> dict:
    """Live read-only GreenBoost snapshot via gb_monitor: whether the kmod is
    loaded, GPU, the T1/T2/T3 tier pool (MB) + combined GB, pressure labels,
    OOM/gaming flags, the shim phase + active allocation path, and the
    Phase-4 KV-prefetch counters (kv_prefetch_* in the raw shim map). This is
    the same state `greenboost status --llm` reports. Canonical owner (shared
    impl in gb_mcp_common.py — also mirrored on greenboost-orchestrator)."""
    import gb_mcp_common
    return gb_mcp_common.greenboost_status()


@mcp.tool()
def synapse_status() -> dict:
    """Gb-Synapse status: whether the llama.cpp `--rpc` engine is BUILT
    (llama-server + rpc-server present in ENGINE_DIR), its version, and whether
    a gb-synapse llama-server and/or the :11434 Ollama-compatible proxy are
    running now. Check this before routing pipeline inference through
    gb-synapse (preferred over raw ollama per CLAUDE.md): if `engine_built` is
    false the fallback is raw ollama and the fix is `gb_synapse build` (host +
    each feeder). Read-only. Canonical: greenboost-synapse. Mirrored here
    (shared impl in gb_mcp_common.py)."""
    import gb_mcp_common
    return gb_mcp_common.synapse_status()


@mcp.tool()
def tiering_status() -> dict:
    """GB-Tiering live state via gb_tiering (the T1/T2/T3 memory tier layer):
    per-tier pool occupancy (MB) + combined GB, pressure labels, shim phase +
    active allocation path, KV reserve, and OOM/gaming flags. This is the
    subsystem-named view of the same shim/kmod state `greenboost_status`
    exposes — use it to see whether VRAM (T1) is filled toward the ~90% target
    (Rule #1) or spilling to T2 DDR / T3 NVMe."""
    try:
        import gb_tiering
        return gb_tiering.tiering_status()
    except Exception as e:
        return {"error": f"gb_tiering unavailable: {e}"}


if __name__ == "__main__":
    mcp.run()

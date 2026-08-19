#!/usr/bin/env python3
"""gb_mcp.py — GreenBoost CENTRAL orchestrating MCP (server `greenboost-orchestrator`).

One server = full awareness of everything GreenBoost can do, plus the dynamic
orchestration flux between the kernel modules and the LLM:

    kernel/shim writes live state (/run/greenboost/shim_stats, metrics.json,
    dataflux events)  →  LLM reads it here  →  decides (rules + measured
    history)  →  acts via DOUBLE-GATED levers (GbControl, GB_ORCH_ACTUATE=1)
    →  kernel/shim reflects the change  →  LLM re-measures, iterates or rolls
    back.

GOAL (owner, verbatim intent): the speediest greenboost-cluster inference
possible while preserving AT LEAST fp8 quantization quality — dynamic
quantization is allowed, but quality is sought alongside speed: any move below
fp8 must surface an explicit quality tradeoff.

Complements (never duplicates) the other servers: `greenboost` (greenboost-cli:
rag/goals/factory), `greenboost-dataflux` (event log drill-down),
`greenboost-cluster` (live cluster state), `greenboost-synapse` (engine/serving
+ CLI bridge). `greenboost_overview()`'s taxonomy names which server owns what.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

mcp = FastMCP("greenboost-orchestrator")

# Per-tool MCP read-only declaration (AE-6). Judged one tool at a time, never
# swept across the file: this server also carries tools that actuate, serve,
# stop, dispatch or write policy, and mislabelling one of those as read-only
# would let a client run it concurrently with anything else. `readOnlyHint` is
# the protocol's own answer to "may this overlap"; a tool that does not carry
# it stays serial, which is the safe default and the previous behaviour.
_READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)

SHIM_STATS = "/run/greenboost/shim_stats"

# Precisions that satisfy the quality floor ("at least fp8").
_FLOOR_OK = {"16", "bf16", "fp16", "fp8"}

# GbControl methods optimize_inference's apply loop may dispatch a pilot lever
# to. Allowlist, not exec(): a lever names one of these by string, never an
# arbitrary Python expression. Keep in sync with gb_control.GbControl's public
# set_*/restore_baseline methods.
_GBCONTROL_LEVER_ALLOWLIST = {
    "set_kv_reserve_mb", "set_safety_reserve_gb", "set_workstation_reserve_mb",
    "set_virtual_vram_gb", "set_gaming_mode", "set_pool_cap_mb", "set_t3_cap_mb",
    "set_idle_cleanup_sec", "set_debug_mode", "set_kv_size_threshold_mb",
    "set_swa_window_mb", "set_prefetch_throttle", "set_phase_detect",
    "set_cpu_governor", "set_energy_perf_pref", "set_numa_balancing",
    "set_swappiness", "set_dirty_background_ratio", "set_watermark_scale_factor",
    "set_gpu_persistence", "set_gpu_power_limit", "set_proc_priority",
}

# `_RULES`/`_TAXONOMY` used to be hardcoded here. GB-Semantics (gb_semantics.py
# + semantics/*.yaml) is now the single source of truth: `_RULES`'s 4
# measurable rules are the docs of the corresponding semantics/segments.yaml
# entries (rule1_underfilled/below_quality_floor/weights_on_t3/
# feeder_idle_while_host_saturated , each is now an EXECUTABLE filter, not
# just prose); the 2 non-measurable policy statements (backend preference,
# actuation gating) stay as static strings in gb_semantics.rules(). Taxonomy
# moved verbatim to gb_semantics.TAXONOMY. See _rules_and_taxonomy() below.

def _rules_and_taxonomy() -> tuple[dict, dict]:
    """Lazy import (matches this file's own pattern for every other gb_*
    dependency) so a missing pyyaml doesn't break MCP server startup , falls
    back to an error marker instead of crashing greenboost_overview()."""
    try:
        import gb_semantics
        return gb_semantics.rules(), gb_semantics.TAXONOMY
    except Exception as e:
        return {"error": f"gb_semantics unavailable: {e}"}, {}



def _read_shim_stats(pid: "int | None" = None) -> dict:
    """Parsed KEY=VALUE map from the shim's live stats file + freshness.
    pid=None (default): the global file, unchanged behavior , every
    existing caller (greenboost_overview, flux_health, whole-system tools)
    wants that view. pid=<int>: that process's own per-PID snapshot
    (missing_features.md item (g)) when available, via
    gb_monitor.shim_stats_path_for , falls back to the global file's path
    otherwise (not its content; a mismatched pid= is the caller's problem
    to notice via the returned stats' own `pid=` field, same as
    gb_monitor.read_shim_stats(pid=)'s stricter contract elsewhere)."""
    out: dict = {"present": False, "fresh": False, "age_s": None}
    try:
        p = Path(SHIM_STATS)
        if pid is not None:
            try:
                import gb_monitor
                per_pid = gb_monitor.shim_stats_path_for(pid)
                if per_pid is not None:
                    p = per_pid
            except Exception:
                pass
        text = p.read_text()
        out["present"] = True
        kv = {}
        for line in text.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                kv[k.strip()] = v.strip()
        out["stats"] = kv
        ts = float(kv.get("timestamp", 0) or 0)
        if ts:
            out["age_s"] = round(time.time() - ts, 1)
            out["fresh"] = out["age_s"] <= 30.0
    except Exception as e:
        out["error"] = str(e)
    return out


def _read_orch_state() -> dict:
    """Parsed ReactiveOrchestrator.dump() JSON from its state file (DI-6:
    health_state/admits_overcommit live here — the orchestrator runs in the
    supervisor process, this MCP server can't read its in-memory state
    directly). Never raises; {} when absent/stale-format."""
    try:
        return json.loads(Path("/run/greenboost/orch_state.json").read_text())
    except Exception:
        return {}


@mcp.tool(annotations=_READ_ONLY)
def greenboost_overview() -> dict:
    """FULL GreenBoost awareness in one call: capabilities, live status, tier
    state, cluster state, synapse (engine/models/serving), dataflux topline,
    the subsystem taxonomy (which MCP server owns what), the operating rules
    (Rule #1 ~90% VRAM, fp8 quality floor, never-T3 weights, maximize
    cluster, prefer gb-synapse , now backed live by GB-Semantics' executable
    segments, not just prose), and a `semantics` verdict block (which named
    segments are currently matched , progressive disclosure: check verdicts
    here first, drill into `semantic_resolve`/a subsystem MCP only for the
    ones that matter). Start here when asked to 'use greenboost'."""
    rules, taxonomy = _rules_and_taxonomy()
    out: dict = {"taxonomy": taxonomy, "rules": rules}
    try:
        import gb_semantics
        out["semantics"] = {
            seg: gb_semantics.evaluate_segment(seg)["matched"]
            for seg in gb_semantics.load()["segments"]}
    except Exception as e:
        out["semantics_error"] = str(e)
    try:
        import gb_monitor
        out["capabilities"] = gb_monitor.capabilities()
        out["status"] = gb_monitor.snapshot().as_dict()
    except Exception as e:
        out["status_error"] = str(e)
    try:
        import gb_tiering
        out["tiering"] = gb_tiering.tiering_status()
    except Exception as e:
        out["tiering_error"] = str(e)
    try:
        import gb_cluster
        out["cluster"] = gb_cluster.status(probe=True)
    except Exception as e:
        out["cluster_error"] = str(e)
    try:
        import gb_synapse
        out["synapse"] = gb_synapse.status()
        out["synapse"]["models"] = [
            {"name": m.name, "quant": m.quant,
             "gib": round(m.n_bytes / 2**30, 2), "source": m.source,
             # Hybrid/pure-recurrent (Mamba/Mamba2/GDN) visibility at a
             # glance — 0/False for a plain transformer. See semantics'
             # ssm_state_gb/recurrent_layer_fraction for the governed form.
             "ssm_state_gb": round(gb_synapse._entry_ssm_gb(m), 3),
             "n_recurrent_layers": m.n_recurrent_layers,
             "is_recurrent_only": m.is_recurrent_only}
            for m in gb_synapse.list_models()]
        out["synapse"]["serving"] = gb_synapse.ps()
    except Exception as e:
        out["synapse_error"] = str(e)
    try:
        import gb_dataflux
        out["dataflux"] = gb_dataflux.summarize(
            gb_dataflux.read_events(since_hours=48))
    except Exception as e:
        out["dataflux_error"] = str(e)
    out["shim"] = _read_shim_stats()
    return out


@mcp.tool(annotations=_READ_ONLY)
def flux_health() -> dict:
    """Is the dynamic kernel⇄LLM orchestration loop actually CLOSED right now —
    and where is it broken if not? Checks: shim_stats fresh (<30s), dataflux
    log writable + recent, kernel module loaded, cluster fabric connected,
    :11434 proxy/ollama answering, GB_ORCH_ACTUATE gate state."""
    checks: dict = {}
    shim = _read_shim_stats()
    checks["shim_stats"] = {"ok": shim.get("fresh", False),
                            "age_s": shim.get("age_s"),
                            "present": shim.get("present", False)}
    try:
        import gb_dataflux
        ev = gb_dataflux.read_events(since_hours=1)
        checks["dataflux"] = {"ok": bool(ev), "events_last_hour": len(ev)}
    except Exception as e:
        checks["dataflux"] = {"ok": False, "error": str(e)}
    try:
        import gb_monitor
        s = gb_monitor.snapshot(probe_gpu=False)
        checks["kmod"] = {"ok": bool(getattr(s, "loaded", False))}
    except Exception as e:
        checks["kmod"] = {"ok": False, "error": str(e)}
    try:
        import gb_cluster
        checks["cluster_fabric"] = {"ok": gb_cluster.cluster_available()}
    except Exception as e:
        checks["cluster_fabric"] = {"ok": False, "error": str(e)}
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=4):
            checks["endpoint_11434"] = {"ok": True}
    except Exception as e:
        checks["endpoint_11434"] = {"ok": False, "error": str(e)[:120]}
    checks["actuation_gate"] = {
        "GB_ORCH_ACTUATE": os.environ.get("GB_ORCH_ACTUATE", "unset"),
        "levers_enabled": os.environ.get("GB_ORCH_ACTUATE") == "1"}
    # DI-6: GpuHealth composite admission-gate state (INIT/HEALTHY/UNHEALTHY/
    # OVERLIMIT/DISABLED) — read from the orchestrator's own state file since
    # it runs in the supervisor process, not this MCP server's.
    orch = _read_orch_state()
    checks["gpu_health"] = {"ok": orch.get("health_state") in (None, "HEALTHY"),
                            "state": orch.get("health_state", "unknown"),
                            "admits_overcommit": orch.get("admits_overcommit")}
    checks["loop_closed"] = all(
        c.get("ok") for k, c in checks.items()
        if k in ("shim_stats", "dataflux", "kmod", "endpoint_11434"))
    return checks


def _gguf_per_tensor_plan(source_gguf_path: str | None) -> dict:
    """missing_features.md item (j): the component-sensitivity-gated
    per-tensor GGUF plan (gb_gguf_plan.py), surfaced live through this MCP
    tool per the MCP-Tool-Gaps rule rather than only being reachable from a
    script.

    `source_gguf_path` MUST point at a Q8_0-or-better GGUF, never the
    manifest's own currently-served quant (typically Q4_K_M or lower) —
    gb_gguf_plan.plan_from_source() assumes every tensor starts at its
    SOURCE type and only ever compresses DOWNWARD; a Q4_K_M source cannot
    satisfy the ssm/mtp floor (q8_0), which needs a real upgrade from a
    higher-precision file (see gb_gguf_plan.py's module docstring for why
    no f16/bf16 GGUF exists upstream for the reference model family, and
    why a Q8_0 download is the correct source, not a shortcut)."""
    if not source_gguf_path:
        return {"error": "per_tensor_plan needs source_gguf_path pointing at a Q8_0-or-better "
                         "GGUF — the manifest's own currently-served quant is not a valid "
                         "source for this plan (see gb_gguf_plan.py's module docstring)"}
    import gb_gguf_plan
    import gb_synapse_backends
    try:
        inventory = gb_gguf_plan.read_gguf_tensor_inventory(source_gguf_path)
    except Exception as e:
        return {"error": f"failed to read {source_gguf_path}: {e}"}
    try:
        _host_free_mb, effective_free_mb, facts = gb_synapse_backends.effective_vram_budget_mb()
    except Exception as e:
        return {"error": f"failed to derive VRAM budget: {e}"}
    plan = gb_gguf_plan.plan_from_source(inventory, budget_bytes=int(effective_free_mb * 1024 ** 2))
    return {
        "source_gguf_path": source_gguf_path,
        "budget_mb": round(effective_free_mb, 1),
        "budget_facts": facts,
        "estimated_gb": round(plan.estimated_bytes / 2 ** 30, 3),
        "fits_budget": plan.fits_budget,
        "role_breakdown": plan.role_breakdown,
        "n_tensor_overrides": len(plan.tensor_types),
    }


@mcp.tool(annotations=_READ_ONLY)
def advisories(phase: str | None = None, severity_min: str | None = None) -> dict:
    """Query structural advisories from the orchestration layer.

    Advisories are generated by gb_pilot.collect_advisories() and other
    producers, unified with stable IDs, severity levels (fatal/blocking/
    warning/info/hint), and lifecycle phases (preflight/serve/runtime/etc).

    Args:
        phase: Optional phase name to filter by (e.g. "runtime.tier")
        severity_min: Optional minimum severity to include ("info", "warning", etc.)

    Returns:
        dict with advisories list and metadata (total_count, blocking_count, etc.)
    """
    import gb_pilot
    import gb_dataflux
    import gb_advisories

    advisories_list = []
    try:
        # Collect advisories from gb_pilot
        analysis = gb_pilot.analyze(gb_dataflux.read_events())
        advisories_list.extend(gb_pilot.collect_advisories(analysis))
    except Exception as e:
        pass  # Best-effort; missing gb_pilot advisories is not fatal

    # Filter by phase if requested
    if phase:
        advisories_list = [
            a for a in advisories_list if a.phase.value == phase
        ]

    # Filter by minimum severity if requested
    if severity_min:
        sev_order = ["hint", "info", "warning", "blocking", "fatal"]
        min_idx = sev_order.index(severity_min) if severity_min in sev_order else 0
        advisories_list = [
            a for a in advisories_list
            if sev_order.index(a.severity.value) >= min_idx
        ]

    # Convert to JSON-serializable dicts
    results = [a.to_dict() for a in advisories_list]

    # Count blocking advisories
    blocking_count = sum(
        1 for a in advisories_list
        if a.severity in (gb_advisories.AdvisorySeverity.FATAL,
                         gb_advisories.AdvisorySeverity.BLOCKING)
    )

    # Observability Must-Rule: a blocking/fatal advisory must leave a
    # dataflux trace, same as any other incident-worthy decision — the
    # KindSpec's incident_when=("blocking","fatal") is exactly this signal.
    for a in advisories_list:
        if a.severity in (gb_advisories.AdvisorySeverity.FATAL,
                          gb_advisories.AdvisorySeverity.BLOCKING):
            try:
                gb_dataflux.emit({
                    "kind": "advisory", "status": a.severity.value,
                    "advisory_id": a.id, "severity": a.severity.value,
                    "phase": a.phase.value,
                })
            except Exception:
                pass

    return {
        "advisories": results,
        "total_count": len(results),
        "blocking_count": blocking_count,
        "phases_present": sorted(set(a.phase.value for a in advisories_list)),
    }


@mcp.tool(annotations=_READ_ONLY)
def quant_advisor(model: str | None = None, ctx: int = 65536,
                  allow_below_fp8: bool = False,
                  per_tensor_plan: bool = False,
                  source_gguf_path: str | None = None) -> dict:
    """Dynamic-quantization intelligence with the quality floor: ranked options
    per model merging the gb-synapse fit predictions (VRAM/cluster placement,
    est/measured tok/s) with each option's position on the precision ladder
    (16 → fp8 → nvfp4 → 8 → 4). Options at or above fp8 rank first; below-fp8
    options are HIDDEN unless allow_below_fp8=True, and always carry an
    explicit `quality_tradeoff` that must be surfaced to the owner — we seek
    quality alongside speed, not only speed.

    per_tensor_plan=True (missing_features.md item (j)) additionally computes
    gb_gguf_plan's component-sensitivity-gated GGUF plan against
    `source_gguf_path` (a Q8_0-or-better file — see _gguf_per_tensor_plan's
    docstring) and returns it under the `per_tensor_plan` key."""
    import gb_synapse
    try:
        reports = gb_synapse.recommend(ctx=ctx, probe_feeders=True)
    except Exception as e:
        return {"error": f"recommend failed: {e}"}
    ladder = ["16", "bf16", "fp16", "fp8", "nvfp4", "8", "4", "tq3", "tq2"]

    def _prec_of(quant: str) -> str:
        q = (quant or "").lower()
        if "fp8" in q: return "fp8"
        if "bf16" in q or "f16" in q or "16" in q: return "bf16"
        if "nvfp4" in q: return "nvfp4"
        if "q8" in q or q == "8": return "8"
        if "q4" in q or "q5" in q or "q6" in q or q == "4": return "4"
        return q or "unknown"

    options = []
    for r in reports:
        if model and model.lower() not in r.name.lower():
            continue
        prec = _prec_of(r.quant)
        at_floor = prec in _FLOOR_OK
        opt = {"model": r.name, "quant": r.quant, "precision": prec,
               "meets_fp8_floor": at_floor,
               "total_gb": round(r.total_gb, 1), "ctx": r.ctx,
               "fits_vram": r.fits_vram,
               "est_tok_s": r.est_tok_s, "measured": r.measured,
               "note": r.note or ""}
        if not at_floor:
            opt["quality_tradeoff"] = (
                f"precision '{prec}' is BELOW the fp8 quality floor — faster/"
                f"smaller but measurably lossier; requires explicit owner "
                f"acknowledgment (dynamic quantization allowed, quality still a goal)")
            if not allow_below_fp8:
                continue
        options.append(opt)
    options.sort(key=lambda o: (not o["meets_fp8_floor"],
                                ladder.index(o["precision"]) if o["precision"] in ladder else 99,
                                -(o["est_tok_s"] or 0)))
    result = {"options": options, "floor": "fp8",
             "allow_below_fp8": allow_below_fp8,
             "guidance": "quality-first env: GB_QUALITY=near_lossless (3.0% "
                         "rel-err ceiling); per-node budgets on cluster runs; "
                         "weights never on T3."}
    if per_tensor_plan:
        result["per_tensor_plan"] = _gguf_per_tensor_plan(source_gguf_path)
    return result


@mcp.tool(annotations=_READ_ONLY)
def gb_plan(model: str, vram_fill_pct: float | None = None) -> dict:
    """Colibri-style tier-plan JSON for a GGUF MoE model (mirrors `coli plan`):
    parses the model's dense-vs-routed-expert byte split (gb_synapse.gguf_summary)
    and this node's live topology, and reports how the routed-expert bytes
    split across VRAM (hot) / T2 DDR (warm, --n-cpu-moe) / T3 (cold — never
    for speed-critical data, quality-first rule). Read-only: parses headers
    and reads topology only, allocates and starts nothing. Every budget term
    is %-derived from THIS node's real hardware (gb_topology), never a
    reference-box literal. Emits a `placement` dataflux event
    (kind=placement, runtime=expert_plan) on every call."""
    import gb_placement
    import gb_synapse
    try:
        entries = [e for e in gb_synapse.list_models() if model.lower() in e.name.lower()]
        if not entries:
            return {"error": f"no model matching {model!r} in the manifest"}
        entry = entries[0]
        summary = gb_synapse.gguf_summary(entry.path)
    except Exception as e:
        return {"error": f"gguf_summary failed: {e}"}
    dense_gb = summary.get("dense_bytes", 0) / (1024 ** 3)
    expert_gb = summary.get("expert_bytes", 0) / (1024 ** 3)
    kv_gb = gb_synapse.estimate_kv_gb(
        65536, entry.n_bytes, entry.quant, n_layers=summary.get("n_layers", 0),
        n_kv_heads=summary.get("n_kv_heads", 0), head_dim=summary.get("head_dim", 0))
    plan = gb_placement.plan_experts_and_emit(dense_gb=dense_gb, expert_gb=expert_gb,
                                              kv_gb=kv_gb, vram_fill_pct=vram_fill_pct)
    plan["model_name"] = entry.name
    plan["is_moe"] = summary.get("is_moe", False)
    return plan


@mcp.tool()
def optimize_inference(model: str | None = None, ctx: int = 65536,
                       days: float = 2.0, apply: bool = False,
                       measure: bool = False, settle_s: int = 20) -> dict:
    """Evolve the workflow along the optimized path: snapshot → constraint
    check (VRAM fill vs 90%, T3 spill, feeders idle, :11434 health) → pilot
    analyze/advise (measured dataflux history) → quant-floor audit →
    gb-synapse recommendation → ORDERED action plan.

    Read-only by default. apply=True executes ONLY GbControl levers and ONLY
    when GB_ORCH_ACTUATE=1 (double gate) — never server lifecycle, never
    re-quantization. measure=True re-reads tok/s after settle_s and AUTO ROLLS
    BACK (GbControl restore_baseline) if throughput regressed >10%."""
    result: dict = {"findings": [], "plan": [], "applied": []}
    plan: list = []

    # 1 · snapshot + constraint box
    try:
        import gb_monitor
        snap = gb_monitor.snapshot().as_dict()
        result["snapshot"] = {k: snap.get(k) for k in
                              ("loaded", "vram_physical_mb", "t2_pool_mb",
                               "t2_allocated_mb", "t3_used_mb", "shim_phase",
                               "swap_pressure", "t2_pressure") if k in snap}
        t3 = int(snap.get("t3_used_mb") or 0)
        if t3 > 0:
            result["findings"].append(
                {"severity": "critical", "topic": "t3_spill",
                 "evidence": f"t3_used_mb={t3} — weights/KV on NVMe breaks the "
                             f"speed goal (placement floor)"})
            plan.append({"action": "eliminate T3 spill: reduce footprint at "
                                   "quality (GB_QUALITY=near_lossless re-quant) "
                                   "or raise T2 pool", "why": "placement floor",
                         "lever": None, "auto_appliable": False})
    except Exception as e:
        result["snapshot_error"] = str(e)
    shim = _read_shim_stats()
    try:
        stats = shim.get("stats", {})
        phys = int(stats.get("physical_vram_mb", 0))
        t1 = int(stats.get("tier_t1_local_cur_mb", 0))
        if phys and shim.get("fresh"):
            fill = round(100.0 * t1 / phys, 1)
            result["t1_fill_pct"] = fill
            if fill < 70.0 and int(stats.get("tier_t2_local_cur_mb", 0)) > 0:
                result["findings"].append(
                    {"severity": "high", "topic": "vram_underfill",
                     "evidence": f"T1 {fill}% while T2 holds "
                                 f"{stats.get('tier_t2_local_cur_mb')} MB — "
                                 f"Rule #1 targets ~90%"})
                plan.append({"action": "front-load VRAM (GB_VRAM_FRONTLOAD=1 "
                                       "test) or serve via gb-synapse --rpc "
                                       "split", "why": "Rule #1 ~90% VRAM",
                             "lever": None, "auto_appliable": False})
    except Exception:
        pass
    try:
        import gb_cluster
        cs = gb_cluster.status(probe=True)
        idle = [f["hostname"] or f["ip"] for f in cs.get("feeders", [])
                if f.get("online") and not f.get("gpu_util_pct")]
        result["cluster"] = {"feeders_online": cs.get("feeders_online"),
                             "idle_feeders": idle}
        if idle:
            result["findings"].append(
                {"severity": "high", "topic": "idle_feeder",
                 "evidence": f"feeder(s) {idle} online but idle — cluster rule: "
                             f"both GPUs must work"})
            plan.append({"action": f"bring {idle} into the work: gb-synapse "
                                   f"--rpc tensor-split for LLM, cluster_map "
                                   f"for task-parallel gen",
                         "why": "maximize whole cluster", "lever": None,
                         "auto_appliable": False})
    except Exception as e:
        result["cluster_error"] = str(e)

    # 2 · pilot advice from measured history
    try:
        import gb_dataflux
        import gb_pilot
        events = gb_dataflux.read_events(since_hours=days * 24)
        analysis = gb_pilot.analyze(events)
        advice = gb_pilot.advise(analysis)
        result["pilot"] = {"advice": advice,
                           "models": analysis.get("models"),
                           "event_count": analysis.get("event_count")}
        for a in advice:
            lv = a.get("lever")
            plan.append({"action": a.get("action"), "why": a.get("topic"),
                         "evidence": a.get("evidence"),
                         "lever": lv,
                         # a structured lever is only auto-appliable when its
                         # args are concretely known (args is not None) — see
                         # gb_pilot.advise()'s docstring for why some levers
                         # deliberately carry args=None instead of a guess.
                         "auto_appliable": bool(lv) and lv.get("args") is not None})
        # 3 · quant floor audit over recent quantize decisions
        breaches = []
        for e in events:
            if e.get("kind") in ("quantize", "quantize_to_fit"):
                bits = str(e.get("bits", "")).lower()
                if bits and bits not in _FLOOR_OK:
                    breaches.append({"component": e.get("component"),
                                     "bits": bits, "ts": e.get("ts")})
        if breaches:
            result["findings"].append(
                {"severity": "medium", "topic": "quality_floor",
                 "evidence": f"{len(breaches)} quantize decision(s) below fp8 "
                             f"recently", "detail": breaches[:5]})
            plan.append({"action": "hold the floor: GB_QUALITY=near_lossless "
                                   "(fp8 ceiling; BF16 reservoir in T2 for "
                                   "outlier layers)", "why": "quality floor",
                         "lever": None, "auto_appliable": False})
    except Exception as e:
        result["pilot_error"] = str(e)

    # 4 · synapse recommendation for the target model
    try:
        import gb_synapse
        fits = gb_synapse.recommend(ctx=ctx, probe_feeders=True)
        if model:
            fits = [f for f in fits if model.lower() in f.name.lower()] or fits
        best = next((f for f in fits if f.fits_vram), fits[0] if fits else None)
        if best:
            result["synapse_pick"] = {
                "model": best.name, "quant": best.quant,
                "est_tok_s": best.est_tok_s, "measured": best.measured,
                "note": best.note or ""}
    except Exception as e:
        result["synapse_error"] = str(e)

    for i, p in enumerate(plan, 1):
        p["order"] = i
    result["plan"] = plan

    # 5 · gated apply + measured closed loop
    if apply:
        if os.environ.get("GB_ORCH_ACTUATE") != "1":
            result["apply_skipped"] = "GB_ORCH_ACTUATE!=1 (double gate)"
            return result
        try:
            import gb_dataflux
            from gb_control import GbControl
            ctl = GbControl()
            before = gb_dataflux.summarize(
                gb_dataflux.read_events(since_hours=1)).get("tok_s", {})
            for p in plan:
                lever = p.get("lever")
                if not lever or not p.get("auto_appliable"):
                    continue
                call = lever.get("call") if isinstance(lever, dict) else None
                args = lever.get("args") or [] if isinstance(lever, dict) else []
                kwargs = lever.get("kwargs") or {} if isinstance(lever, dict) else {}
                call_str = f"{call}({', '.join(map(repr, args))})"
                if call not in _GBCONTROL_LEVER_ALLOWLIST:
                    result["applied"].append(f"REFUSED {call_str}: not an allowlisted GbControl method")
                    continue
                try:
                    getattr(ctl, call)(*args, **kwargs)
                    result["applied"].append(call_str)
                except Exception as e:
                    result["applied"].append(f"FAILED {call_str}: {e}")
            if measure and result["applied"]:
                time.sleep(min(max(settle_s, 5), 60))
                after = gb_dataflux.summarize(
                    gb_dataflux.read_events(since_hours=1)).get("tok_s", {})
                result["measure"] = {"before": before, "after": after}
                try:
                    b = list(before.values())[0] if before else 0
                    a = list(after.values())[0] if after else 0
                    if b and a and a < b * 0.9:
                        ctl.restore_baseline()
                        result["rolled_back"] = (
                            f"tok/s regressed {b:.1f}→{a:.1f} (>10%) — "
                            f"restore_baseline() executed")
                except Exception as e:
                    result["rollback_error"] = str(e)
        except Exception as e:
            result["apply_error"] = str(e)
    return result


# ── gated actuation (shared impl in gb_actuation; A2A uses the same funcs) ──

@mcp.tool()
def set_quant_policy(budget_gb: float | None = None, quality: str | None = None,
                     confirm: bool = False) -> dict:
    """ACTUATE (double-gated): set the quant policy the NEXT pipeline run reads
    , GB_QUANT_BUDGET_GB and/or GB_QUALITY in the shared inference.env. Enforces
    the fp8 floor (below-fp8 quality is surfaced as an explicit tradeoff, never
    silent). This is the actuatable counterpart to quant_advisor's read-only
    recommendation. DRY-RUN unless confirm=True AND GB_ORCH_ACTUATE=1."""
    import gb_actuation
    return gb_actuation.set_quant_policy(budget_gb=budget_gb, quality=quality,
                                         confirm=confirm)


@mcp.tool()
def tier_actuate(lever: str, value: int, confirm: bool = False) -> dict:
    """ACTUATE (double-gated): move a GB-Tiering lever via GbControl , one of
    kv_reserve_mb, safety_reserve_gb, workstation_reserve_mb, virtual_vram_gb,
    pool_cap_mb. DRY-RUN unless confirm=True AND GB_ORCH_ACTUATE=1; emits an
    `actuation` event when applied. (Per-buffer promote/demote/evict is
    in-process only and not remotable.)"""
    import gb_actuation
    return gb_actuation.tier_actuate(lever, value, confirm=confirm)


@mcp.tool(annotations=_READ_ONLY)
def shim_env(workload: str = "diffusion", enabled: bool = True) -> dict:
    """QUERY (no gate): the LD_PRELOAD env overlay that turns GreenBoost on for a
    subprocess of `workload` type ("diffusion"|"llm"|...). Lets an agent obtain
    the exact env to launch a GreenBoost-accelerated run without a Python
    import , the missing "turn it on" surface for pipelines."""
    import gb_actuation
    return gb_actuation.shim_env(workload=workload, enabled=enabled)


@mcp.tool()
def run_under_greenboost(command: "list[str] | str", workload: str = "llm",
                         cwd: str = "", timeout_s: int = 900,
                         confirm: bool = False) -> dict:
    """ACTUATE (double-gated): run `command` as a subprocess with the
    shim_env(workload) overlay applied , closes discover -> configure ->
    execute -> observe entirely through MCP, no separate shell step needed.
    `command` always runs as an argv list (never through a shell , a string
    is tokenized with shlex, so pipes/redirects/`&&` have no special
    meaning). DRY-RUN unless confirm=True AND GB_ORCH_ACTUATE=1; emits
    kind="agent_run" (start + completion) either way for audit."""
    import gb_actuation
    return gb_actuation.run_under_greenboost(command, workload=workload, cwd=cwd,
                                             timeout_s=timeout_s, confirm=confirm)


@mcp.tool(annotations=_READ_ONLY)
def reclaim_plan(scope: str = "residue", kill_min_mb: int = 512) -> dict:
    """QUERY (no gate): classify every GreenBoost-held GPU/T2/T3 process into
    live/ambiguous/residue and report what scope ("residue"|"ambiguous"|
    "all") WOULD reclaim, without touching anything. "residue" (default) is
    orphaned processes only — a genuinely in-progress gb-synapse server
    never shows up until scope="all"."""
    import gb_actuation
    return gb_actuation.reclaim_plan(scope=scope, kill_min_mb=kill_min_mb)


@mcp.tool()
def reclaim_run(scope: str = "residue", kill_min_mb: int = 512,
                confirm: bool = False) -> dict:
    """ACTUATE (double-gated): reclaim GreenBoost-held GPU/T2/T3 memory at
    `scope`. Default scope="residue" only kills orphaned processes; a
    genuinely in-progress gb-synapse server is left running. scope="all"
    reproduces the old `greenboost clear memory-pool` nuke's full blast
    radius — never the default, explicit opt-in only (CLAUDE.md: never run
    against another genuinely-in-progress GreenBoost job without explicit
    authorization). DRY-RUN unless confirm=True AND GB_ORCH_ACTUATE=1."""
    import gb_actuation
    return gb_actuation.reclaim_run(scope=scope, kill_min_mb=kill_min_mb, confirm=confirm)


@mcp.tool()
def support_bundle(output: str = "", timeout_s: float = 60.0, confirm: bool = False) -> dict:
    """ACTUATE (gated): collect a redacted diagnostic tarball (readiness
    report, kmod/device state, nvidia-smi, memory, shim stats, dataflux
    tail, synapse run state, serving-recipe check, per-feeder state over
    SSH), NemoClaw round-2 item 11. Every value is redacted before being
    written (credential-shaped secrets via greenboost-cli's capture.py,
    private LAN IPs and /home/<user>/ paths always). DRY-RUN unless
    confirm=True, it writes a file, so it is actuation and must never sit
    inside `doctor`, whose `mutated: False` is schema-enforced. `output`
    defaults to `./greenboost-debug-<timestamp>.tar.gz`; review the bundle
    before sharing it regardless of the redaction pass."""
    import gb_debug_bundle
    if not output:
        output = f"./greenboost-debug-{time.strftime('%Y%m%d-%H%M%S')}.tar.gz"
    if not confirm:
        return {"dry_run": True, "would_write": os.path.expanduser(output),
                "note": "pass confirm=True to actually collect and write the bundle"}
    ok = gb_debug_bundle.collect_support_bundle(os.path.expanduser(output), timeout_s=timeout_s)
    return {"dry_run": False, "ok": ok,
            "path": os.path.expanduser(output) if ok else None}


@mcp.tool()
def a2a_gateway(action: str = "status", confirm: bool = False) -> dict:
    """A6: control the greenboost-a2a systemd unit (AgentCard + JSON-RPC
    actuation gateway). action="status" (default, always allowed): whether
    the unit is enabled/active per systemd, plus the same liveness probe as
    greenboost-dataflux's a2a_status (use that tool for the full recent-
    request rollup — this is a thin systemd-state complement, not a
    duplicate). action="restart" is DOUBLE-GATED (confirm=True AND
    GB_ORCH_ACTUATE=1) since it's a service-lifecycle action, not a policy
    write — deliberately NOT routed through gb_actuation.VERBS: restarting
    the A2A gateway FROM an A2A JSON-RPC call would kill the request's own
    connection mid-response, so this control only makes sense from MCP."""
    import subprocess
    unit = "greenboost-a2a.service"
    if action not in ("status", "restart"):
        return {"error": f"unknown action {action!r}; use 'status' or 'restart'"}
    if action == "restart":
        import gb_actuation
        gate = gb_actuation.actuation_gate(confirm)
        if not gate["allowed"]:
            return {"action": "restart", "unit": unit, "gate": gate,
                    "dry_run": f"would systemctl restart {unit}"}
        try:
            r = subprocess.run(["systemctl", "restart", unit],
                              capture_output=True, text=True, timeout=15)
            result = {"action": "restart", "unit": unit, "gate": gate,
                     "ok": r.returncode == 0, "stderr": r.stderr.strip()[:500]}
        except Exception as e:
            result = {"action": "restart", "unit": unit, "gate": gate, "ok": False, "error": str(e)}
        try:
            import gb_dataflux
            gb_dataflux.emit({"node": "host", "label": "a2a", "kind": "actuation",
                              "stage": "a2a_gateway_restart", "lever": "a2a_gateway",
                              "gated": True, "status": "ok" if result.get("ok") else "error"})
        except Exception:
            pass
        return result
    # action == "status"
    try:
        r = subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True, timeout=5)
        active = r.stdout.strip() == "active"
    except Exception:
        active = None
    try:
        r2 = subprocess.run(["systemctl", "is-enabled", unit], capture_output=True, text=True, timeout=5)
        enabled = r2.stdout.strip() == "enabled"
    except Exception:
        enabled = None
    return {"action": "status", "unit": unit, "active": active, "enabled": enabled,
           "note": "see greenboost-dataflux's a2a_status for the liveness probe + recent request rollup"}


# ── selective re-exports: one server suffices day-to-day ────────────────────

@mcp.tool(annotations=_READ_ONLY)
def greenboost_status() -> dict:
    """Live GreenBoost snapshot (kmod, GPU, T1/T2/T3 pools, pressure, phase) —
    same data as `greenboost status --llm`. Canonical: greenboost-dataflux.
    Mirrored here (identical shape, shared impl in gb_mcp_common.py) so this
    server alone suffices."""
    import gb_mcp_common
    return gb_mcp_common.greenboost_status()


@mcp.tool(annotations=_READ_ONLY)
def greenboost_capabilities() -> dict:
    """Installed/running shim capability manifest (features the deployed
    GreenBoost actually supports). Canonical: greenboost-dataflux. Mirrored
    here (shared impl in gb_mcp_common.py)."""
    import gb_mcp_common
    return gb_mcp_common.greenboost_capabilities()


@mcp.tool(annotations=_READ_ONLY)
def greenboost_pilot(days: float = 5.0) -> dict:
    """Pilot instrument panel: per-stage wall-time trends, measured tok/s per
    model, pressure picture, and evidence-backed advice (levers quoted, never
    auto-applied). Canonical: greenboost-dataflux. Mirrored here (shared impl
    in gb_mcp_common.py)."""
    import gb_mcp_common
    return gb_mcp_common.greenboost_pilot(days=days)


@mcp.tool(annotations=_READ_ONLY)
def cluster_status(probe: bool = True) -> dict:
    """Gb-Cluster headline: feeders online, per-node T1/T2/T3 + GPU telemetry.
    Canonical: greenboost-cluster. Mirrored here (shared impl in
    gb_mcp_common.py)."""
    import gb_mcp_common
    return gb_mcp_common.cluster_status(probe=probe)


@mcp.tool(annotations=_READ_ONLY)
def dataflux_summary(days: float = 5.0) -> dict:
    """GB-Dataflux headline rollup (events, per-node throughput, errors,
    tok/s). Canonical: greenboost-dataflux. Mirrored here (shared impl in
    gb_mcp_common.py)."""
    import gb_mcp_common
    return gb_mcp_common.dataflux_summary(days=days)


@mcp.tool(annotations=_READ_ONLY)
def synapse_status() -> dict:
    """Gb-Synapse engine/proxy status (built?, version, server + :11434 proxy
    running?). Canonical: greenboost-synapse. Mirrored here (shared impl in
    gb_mcp_common.py)."""
    import gb_mcp_common
    return gb_mcp_common.synapse_status()


# ── GB-Semantics , the governed semantic layer ──────────────────────────────
# MANDATORY DEFAULT PATH for any question about GreenBoost's own state: use
# these 4 tools BEFORE reading a raw telemetry/dataflux field directly. See
# gb_semantics.py's module docstring and .claude/skills/gb-semantics/SKILL.md
# for why , several raw fields in this codebase are traps (same name,
# different meaning, or a virtual number that looks like the real one).

@mcp.tool(annotations=_READ_ONLY)
def semantic_metrics(query: str = "") -> dict:
    """Discover governed metrics/segments by keyword (e.g. 'vram', 'rule 1',
    'throttl'). Returns names + docs + `never_use` traps , NOT resolved
    values, call `semantic_resolve`/`semantic_segments` for those. Empty
    query lists every governed metric name. This is the retrieval-failure
    fix: narrows GreenBoost's ~60-tool MCP surface to a handful of named
    answers before any other tool is called."""
    import gb_semantics
    if not query:
        L = gb_semantics.load()
        return {"metrics": sorted(L["metrics"].keys()),
                "segments": sorted(L["segments"].keys())}
    return {"matches": gb_semantics.discover(query, k=8)}


@mcp.tool(annotations=_READ_ONLY)
def semantic_resolve(metric: str, entity: "str | None" = None,
                      window_s: "float | None" = None) -> dict:
    """THE one governed number for `metric` (by name or alias) , value, unit,
    threshold verdict, and provenance (which raw field it came from, its
    owner, freshness). Use this instead of reading a dataflux/telemetry
    field directly; the response's `never_use` list names the specific
    fields that look plausible but are wrong for this metric."""
    import gb_semantics
    return gb_semantics.resolve(metric, entity_id=entity, window_s=window_s)


@mcp.tool(annotations=_READ_ONLY)
def semantic_segments(name: "str | None" = None) -> dict:
    """Evaluate named canonical filters (e.g. `rule1_underfilled`,
    `swap_thrash_not_gpu_throttle`, `kmod_missing_silent_degrade`) , each
    returns {matched, evidence}, evidence being the resolved metrics that
    fed the verdict. `name=None` evaluates every governed segment."""
    import gb_semantics
    if name:
        return gb_semantics.evaluate_segment(name)
    L = gb_semantics.load()
    return {seg: gb_semantics.evaluate_segment(seg) for seg in L["segments"]}


@mcp.tool(annotations=_READ_ONLY)
def semantic_answer(question: str) -> dict:
    """Full route: a natural-language question about GreenBoost's own state
    -> matching intent -> resolved metrics + evaluated segments + a
    provenance footer ('Source: semantic layer (...) · Owner: ... ·
    Governed: true'). Returns `governed: false` with a raw-tool suggestion
    when no route covers the question , that footer MUST be surfaced to the
    user verbatim per the gb-semantics SKILL.md's mandatory-default-path
    rule, never silently answered from a raw field instead."""
    import gb_semantics
    return gb_semantics.answer(question)


# ── live kernel⇄LLM resources (cheap polling surface) ───────────────────────

@mcp.resource("greenboost://shim-stats")
def res_shim_stats() -> str:
    """Live parsed shim stats (KEY=VALUE map + freshness) straight from the
    kernel-facing shim — the raw side of the dynamic loop."""
    return json.dumps(_read_shim_stats())


@mcp.resource("greenboost://rules")
def res_rules() -> str:
    """The operating constraint box (Rule #1, fp8 floor, placement floor,
    cluster rule, backend rule, actuation gates). Backed live by
    gb_semantics.rules() , see greenboost://semantics for the executable form."""
    rules, _ = _rules_and_taxonomy()
    return json.dumps(rules)


@mcp.resource("greenboost://taxonomy")
def res_taxonomy() -> str:
    """Subsystem taxonomy: what each GB-* module does and which MCP owns it."""
    _, taxonomy = _rules_and_taxonomy()
    return json.dumps(taxonomy)


@mcp.resource("greenboost://semantics")
def res_semantics() -> str:
    """Full GB-Semantics registry dump: every governed metric, segment,
    entity, and route , the machine-readable form of semantics/*.yaml, for
    an LLM client that wants the whole schema in one read instead of
    discovering it incrementally via `semantic_metrics`."""
    import gb_semantics
    L = gb_semantics.load()
    return json.dumps({
        "metrics": {n: m.__dict__ for n, m in L["metrics"].items()},
        "segments": {n: s.__dict__ for n, s in L["segments"].items()},
        "entities": {n: e.__dict__ for n, e in L["entities"].items()},
        "routes": [r.__dict__ for r in L["routes"]],
    }, default=str)


if __name__ == "__main__":
    mcp.run()

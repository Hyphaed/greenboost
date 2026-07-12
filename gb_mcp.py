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

mcp = FastMCP("greenboost-orchestrator")

SHIM_STATS = "/run/greenboost/shim_stats"

# Precisions that satisfy the quality floor ("at least fp8").
_FLOOR_OK = {"16", "bf16", "fp16", "fp8"}

_RULES = {
    "rule_1_vram": "Fill every GPU's physical VRAM to ~90% (10% headroom so the "
                   "system never collapses under pressure). GPU VRAM is the "
                   "fastest RAM — T2 DDR/T3 NVMe are overflow only.",
    "quality_floor": "Minimum quality = fp8 for hot (T1) weights. Quality-first "
                     "path: GB_QUALITY=near_lossless (rel-err ceiling 3.0%, fp8 "
                     "qualifies). Ladder 16 → fp8 → nvfp4 → 8 → 4. Dynamic "
                     "quantization is allowed, but any option below fp8 MUST "
                     "carry an explicit quality tradeoff surfaced to the owner.",
    "placement_floor": "Weights NEVER on T3 NVMe (quantize_to_fit refuses; "
                       "t3_used_mb>0 during inference is a critical finding).",
    "cluster": "Maximize the WHOLE cluster: host + every feeder toward ~90% VRAM "
               "AND compute (unified CUDA device). A connected idle feeder while "
               "the host bottlenecks is a rule violation.",
    "backend": "Prefer gb-synapse over raw ollama (cross-GPU --rpc tensor-split, "
               "gb-quant, proxy-side dataflux tok/s).",
    "actuation": "All levers double-gated: apply=True AND GB_ORCH_ACTUATE=1. "
                 "Server lifecycle is never auto-applied.",
}

_TAXONOMY = {
    "GB-Tiering": {"does": "T1 VRAM / T2 DDR / T3 NVMe as one virtual pool",
                   "module": "gb_tiering.py (engine: shim + greenboost.ko)",
                   "mcp": "greenboost-orchestrator: tiering via greenboost_status; "
                          "greenboost-dataflux: tiering_status, dataflux_tier_moves"},
    "GB-Quant": {"does": "weights (gb_quant) + KV cache (TurboQuant/gb_attn) "
                         "compression so models fit VRAM at quality",
                 "module": "gb_quant.py + gb_attn.py",
                 "mcp": "greenboost-orchestrator: quant_advisor; "
                        "greenboost-dataflux: dataflux_quantization"},
    "GB-Dataflux": {"does": "flight recorder: every subsystem emits events; "
                            "web UI via `greenboost dataflux-ui`",
                    "module": "gb_dataflux.py",
                    "mcp": "greenboost-dataflux (full drill-down); headline via "
                           "dataflux_summary here"},
    "GB-Cluster": {"does": "borrow LAN GPUs+RAM (feeders); unified virtual GPU; "
                           "cluster_map/ensure_feeder_ready dispatch",
                   "module": "gb_cluster.py (fabric: greenboost-netd)",
                   "mcp": "greenboost-cluster (live state); headline via "
                          "cluster_status here"},
    "GB-Synapse": {"does": "own model server: llama.cpp --rpc tensor-split "
                           "across cluster + Ollama/OpenAI proxy on :11434",
                   "module": "gb_synapse.py + gb_synapse_api.py",
                   "mcp": "greenboost-synapse (serve/stop/models + CLI bridge); "
                          "headline via synapse_status here"},
    "GB-CLI": {"does": "agentic terminal client (`gb`, `greenboost-cli`), "
               "always talks to gb-synapse :11434",
               "module": "greenboost-cli (installed by Full Install)",
               "mcp": "greenboost (rag/goals/factory); cli bridge in "
                      "greenboost-synapse"},
}


def _read_shim_stats() -> dict:
    """Parsed KEY=VALUE map from the shim's live stats file + freshness."""
    out: dict = {"present": False, "fresh": False, "age_s": None}
    try:
        p = Path(SHIM_STATS)
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


@mcp.tool()
def greenboost_overview() -> dict:
    """FULL GreenBoost awareness in one call: capabilities, live status, tier
    state, cluster state, synapse (engine/models/serving), dataflux topline,
    the subsystem taxonomy (which MCP server owns what), and the operating
    rules (Rule #1 ~90% VRAM, fp8 quality floor, never-T3 weights, maximize
    cluster, prefer gb-synapse). Start here when asked to 'use greenboost'."""
    out: dict = {"taxonomy": _TAXONOMY, "rules": _RULES}
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
             "gib": round(m.n_bytes / 2**30, 2), "source": m.source}
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


@mcp.tool()
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
    checks["loop_closed"] = all(
        c.get("ok") for k, c in checks.items()
        if k in ("shim_stats", "dataflux", "kmod", "endpoint_11434"))
    return checks


@mcp.tool()
def quant_advisor(model: str | None = None, ctx: int = 65536,
                  allow_below_fp8: bool = False) -> dict:
    """Dynamic-quantization intelligence with the quality floor: ranked options
    per model merging the gb-synapse fit predictions (VRAM/cluster placement,
    est/measured tok/s) with each option's position on the precision ladder
    (16 → fp8 → nvfp4 → 8 → 4). Options at or above fp8 rank first; below-fp8
    options are HIDDEN unless allow_below_fp8=True, and always carry an
    explicit `quality_tradeoff` that must be surfaced to the owner — we seek
    quality alongside speed, not only speed."""
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
    return {"options": options, "floor": "fp8",
            "allow_below_fp8": allow_below_fp8,
            "guidance": "quality-first env: GB_QUALITY=near_lossless (3.0% "
                        "rel-err ceiling); per-node budgets on cluster runs; "
                        "weights never on T3."}


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
                               "t2_allocated_mb", "t3_used_mb", "phase",
                               "pressure") if k in snap}
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
            plan.append({"action": a.get("action"), "why": a.get("topic"),
                         "evidence": a.get("evidence"),
                         "lever": a.get("lever"),
                         "auto_appliable": bool(a.get("lever"))})
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
                try:
                    exec(lever, {"GbControl": lambda: ctl})  # noqa: S102 — pilot-authored lever strings
                    result["applied"].append(lever)
                except Exception as e:
                    result["applied"].append(f"FAILED {lever}: {e}")
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


# ── selective re-exports: one server suffices day-to-day ────────────────────

@mcp.tool()
def greenboost_status() -> dict:
    """Live GreenBoost snapshot (kmod, GPU, T1/T2/T3 pools, pressure, phase) —
    same data as `greenboost status --llm`. Canonical drill-down owner:
    greenboost-dataflux MCP."""
    try:
        import gb_monitor
        return gb_monitor.snapshot().as_dict()
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def greenboost_capabilities() -> dict:
    """Installed/running shim capability manifest (features the deployed
    GreenBoost actually supports)."""
    try:
        import gb_monitor
        return gb_monitor.capabilities()
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def greenboost_pilot(days: float = 5.0) -> dict:
    """Pilot instrument panel: per-stage wall-time trends, measured tok/s per
    model, pressure picture, and evidence-backed advice (levers quoted, never
    auto-applied)."""
    try:
        import gb_dataflux
        import gb_pilot
        analysis = gb_pilot.analyze(gb_dataflux.read_events(since_hours=days * 24))
        analysis["advice"] = gb_pilot.advise(analysis)
        return analysis
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def cluster_status(probe: bool = True) -> dict:
    """Gb-Cluster headline: feeders online, per-node T1/T2/T3 + GPU telemetry.
    Canonical live-state owner: greenboost-cluster MCP."""
    try:
        import gb_cluster
        return gb_cluster.status(probe=probe)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def dataflux_summary(days: float = 5.0) -> dict:
    """GB-Dataflux headline rollup (events, per-node throughput, errors,
    tok/s). Canonical drill-down owner: greenboost-dataflux MCP."""
    try:
        import gb_dataflux
        return gb_dataflux.summarize(gb_dataflux.read_events(since_hours=days * 24))
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def synapse_status() -> dict:
    """Gb-Synapse engine/proxy status (built?, version, server + :11434 proxy
    running?). Serving control lives in the greenboost-synapse MCP."""
    try:
        import gb_synapse
        return gb_synapse.status()
    except Exception as e:
        return {"error": str(e)}


# ── live kernel⇄LLM resources (cheap polling surface) ───────────────────────

@mcp.resource("greenboost://shim-stats")
def res_shim_stats() -> str:
    """Live parsed shim stats (KEY=VALUE map + freshness) straight from the
    kernel-facing shim — the raw side of the dynamic loop."""
    return json.dumps(_read_shim_stats())


@mcp.resource("greenboost://rules")
def res_rules() -> str:
    """The operating constraint box (Rule #1, fp8 floor, placement floor,
    cluster rule, backend rule, actuation gates)."""
    return json.dumps(_RULES)


@mcp.resource("greenboost://taxonomy")
def res_taxonomy() -> str:
    """Subsystem taxonomy: what each GB-* module does and which MCP owns it."""
    return json.dumps(_TAXONOMY)


if __name__ == "__main__":
    mcp.run()

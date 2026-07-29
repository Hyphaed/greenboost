#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_semantics.py , GB-Semantics: the governed semantic layer for GreenBoost's
own agents.

Every `greenboost-*` MCP server (and every non-MCP consumer: GB-CLI,
ai-forge, optimal-claude in ~/Dev/claude_workflow) answers a "what's the
current X" or "is Y satisfied" question about GreenBoost's own state by
resolving through THIS module, not by reading a raw telemetry/dataflux field
directly. The reason: several raw fields in this codebase are traps , same
name, different meaning, or a virtual number that looks like the real one
(see semantics/metrics.yaml's `never_use` entries for the documented
incidents this has already caused: the shim-inflated `fb_used_pct`, the
`t2_pressure` float-vs-enum collision, the `n_layers`-vs-`n_kv_layers`
hybrid-architecture undercounting that clamped context to 2048 tokens).

Architecture: definitions live in human-owned `semantics/*.yaml`
(entities/metrics/segments/routes , diffable, reviewable without reading
Python, deliberately NOT LLM-authored wholesale, see Anthropic's own
finding: auto-generated metric definitions reproduce the exact ambiguity
they're meant to remove). This module COMPILES those definitions and binds
each one to a deterministic resolver function that wraps EXISTING GreenBoost
accessors (gb_monitor, gb_dataflux, gb_cluster, gb_synapse) , it adds no new
data path, only a governed name for data that already exists.

Public API:
    load()                          , compile + cache semantics/*.yaml
    resolve(metric, entity_id=None, window_s=None)  , THE one number
    discover(query, k=5)            , keyword/alias search over metrics+segments
    evaluate_segment(name)          , named canonical filter -> {matched, evidence}
    answer(question)                , full route: question -> metrics+segments+footer
    card()                          , bounded prompt block for non-MCP consumers

`checks/check_semantics_coverage.py` is the blocking CI pass that keeps this
honest: every metric needs a real `_res_*` resolver, every segment a real
`_seg_*` evaluator, every `never_use` field must still exist upstream.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError as e:  # pragma: no cover , environment misconfiguration
    raise ImportError(
        "gb_semantics requires pyyaml (already a greenboost-cli dependency; "
        "`pip install pyyaml` if running this module standalone). The "
        "semantic layer's definitions live in semantics/*.yaml, not Python "
        "literals , see Phase 1 of the GB-Semantics plan for why."
    ) from e

_SEMANTICS_DIR = Path(__file__).resolve().parent / "semantics"
_RESOLVER_PREFIX = "_res_"
_EVAL_PREFIX = "_seg_"


# ── definitions (compiled from YAML) ─────────────────────────────────────────

@dataclass(frozen=True)
class Metric:
    name: str
    entity: str = ""
    unit: str = ""
    doc: str = ""
    resolver: str = ""
    source_fields: tuple = ()
    source_tier: str = "measured"
    direction: str = "informational"
    threshold: dict = field(default_factory=dict)
    owner: str = ""
    never_use: tuple = ()
    aliases: tuple = ()


@dataclass(frozen=True)
class Segment:
    name: str
    doc: str = ""
    evaluator: str = ""
    reads: tuple = ()
    severity: str = "info"


@dataclass(frozen=True)
class Entity:
    name: str
    grain: str = ""
    identity_key: object = None
    doc: str = ""


@dataclass(frozen=True)
class Route:
    intent: str
    triggers: tuple = ()
    metrics: tuple = ()
    segments: tuple = ()
    raw_tool_fallback: str = ""


_CACHE: dict = {}


# ── subsystem taxonomy + operating rules ─────────────────────────────────────
# Moved verbatim from gb_mcp.py's old `_TAXONOMY`/`_RULES` module-level dicts
# (the ORIGINAL hardcoded glossary this whole layer replaces) , this is
# static descriptive metadata (which module/MCP owns which subsystem), not a
# resolvable metric, so it stays plain Python rather than forcing a fifth
# YAML file. `rules()` derives its 4 measurable entries from segments.yaml
# (the segments ARE the rules now, executable instead of prose); the 2
# non-measurable policy statements (backend preference, actuation gating)
# aren't conditions a resolver can evaluate, so they stay as static strings
# here, clearly marked as such.

TAXONOMY = {
    "GB-Tiering": {"does": "T1 VRAM / T2 DDR / T3 NVMe as one virtual pool",
                   "module": "gb_tiering.py (engine: shim + greenboost.ko)",
                   "mcp": "greenboost-orchestrator: tiering via greenboost_status; "
                          "greenboost-dataflux: tiering_status, dataflux_tier_moves"},
    "GB-Quant": {"does": "weights (gb_quant) + KV cache (TurboQuant/gb_attn) "
                         "compression so models fit VRAM at quality; "
                         "gb_placement.plan_experts (CB-3) plans the MoE "
                         "dense/hot-expert/warm-expert/cold-expert tier split",
                 "module": "gb_quant.py + gb_attn.py + gb_placement.py",
                 "mcp": "greenboost-orchestrator: quant_advisor, gb_plan; "
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
    "GB-Semantics": {"does": "governed metric/segment/route layer , the one "
                             "name per concept + never_use traps this whole "
                             "module implements",
                      "module": "gb_semantics.py + semantics/*.yaml",
                      "mcp": "greenboost-orchestrator: semantic_metrics, "
                             "semantic_resolve, semantic_segments, "
                             "semantic_answer; resource greenboost://semantics"},
}


def rules() -> dict:
    """Operating rules , the 4 measurable ones are pulled live from
    segments.yaml (so a rule and its enforcement can never drift apart
    again); the 2 policy statements aren't measurable conditions, so they
    stay as static strings, clearly labeled below."""
    L = load()
    seg = L["segments"]

    def _doc(name: str) -> str:
        return seg[name].doc if name in seg else f"(segment '{name}' missing)"

    return {
        "rule_1_vram": _doc("rule1_underfilled"),
        "quality_floor": _doc("below_quality_floor"),
        "placement_floor": _doc("weights_on_t3"),
        "cluster": _doc("feeder_idle_while_host_saturated"),
        # Not measurable conditions , policy statements, kept as static text.
        "backend": "Prefer gb-synapse over raw ollama (cross-GPU --rpc "
                   "tensor-split, gb-quant, proxy-side dataflux tok/s).",
        "actuation": "All levers double-gated: apply=True AND "
                     "GB_ORCH_ACTUATE=1. Server lifecycle is never auto-applied.",
    }


def _read_yaml_list(fname: str) -> list:
    p = _SEMANTICS_DIR / fname
    with open(p) as f:
        return yaml.safe_load(f) or []


def _read_yaml_map(fname: str) -> dict:
    p = _SEMANTICS_DIR / fname
    with open(p) as f:
        return yaml.safe_load(f) or {}


def load(force: bool = False) -> dict:
    """Compile semantics/*.yaml once and cache. A malformed YAML fails LOUD
    here (surfaced the first time any resolve()/answer() call touches this
    module), never silently , this is definitions-as-code, not best-effort
    telemetry."""
    if not force and _CACHE:
        return _CACHE

    metrics: dict[str, Metric] = {}
    for m in _read_yaml_list("metrics.yaml"):
        metrics[m["name"]] = Metric(
            name=m["name"], entity=m.get("entity", ""), unit=m.get("unit", ""),
            doc=(m.get("doc") or "").strip(), resolver=m.get("resolver", m["name"]),
            source_fields=tuple(m.get("source_fields", ())),
            source_tier=m.get("source_tier", "measured"),
            direction=m.get("direction", "informational"),
            threshold=m.get("threshold", {}) or {},
            owner=m.get("owner", ""),
            never_use=tuple(m.get("never_use", ()) or ()),
            aliases=tuple(m.get("aliases", ()) or ()),
        )

    segments: dict[str, Segment] = {}
    for s in _read_yaml_list("segments.yaml"):
        segments[s["name"]] = Segment(
            name=s["name"], doc=(s.get("doc") or "").strip(),
            evaluator=s.get("evaluator", s["name"]),
            reads=tuple(s.get("reads", ()) or ()), severity=s.get("severity", "info"))

    entities: dict[str, Entity] = {}
    for name, e in _read_yaml_map("entities.yaml").items():
        entities[name] = Entity(name=name, grain=(e.get("grain") or "").strip(),
                                 identity_key=e.get("identity_key"),
                                 doc=(e.get("doc") or "").strip())

    routes: list[Route] = []
    for r in _read_yaml_list("routes.yaml"):
        routes.append(Route(
            intent=r["intent"], triggers=tuple(r.get("triggers", ()) or ()),
            metrics=tuple(r.get("metrics", ()) or ()),
            segments=tuple(r.get("segments", ()) or ()),
            raw_tool_fallback=r.get("raw_tool_fallback", "")))

    _CACHE.clear()
    _CACHE.update({"metrics": metrics, "segments": segments,
                   "entities": entities, "routes": routes})
    return _CACHE


def _alias_lookup(name: str, metrics: dict) -> "Metric | None":
    for m in metrics.values():
        if name in m.aliases:
            return m
    return None


# ── shared read helpers (wrap EXISTING accessors, no new data path) ─────────

def _latest_event(kind: str, node: "str | None" = None,
                   max_age_s: "float | None" = None, days: float = 2.0) -> "dict | None":
    """Latest dataflux event of a given kind, optionally node- and
    freshness-gated. None if nothing recent enough exists , callers must
    treat that as "unavailable", never as "zero"."""
    try:
        import gb_dataflux
    except Exception:
        return None
    try:
        events = gb_dataflux.read_events(since_hours=days * 24)
    except Exception:
        return None
    cand = [e for e in events if e.get("kind") == kind]
    if node:
        cand = [e for e in cand if gb_dataflux.canonical_node(e.get("node")) == node]
    if not cand:
        return None
    ev = max(cand, key=lambda e: e.get("ts", 0))
    if max_age_s is not None and (time.time() - ev.get("ts", 0)) > max_age_s:
        return None
    return ev


def _free_swap_mb() -> "tuple[float, float] | tuple[None, None]":
    try:
        out = subprocess.run(["free", "-b"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if line.lower().startswith("swap"):
                parts = line.split()
                return float(parts[1]) / 1e6, float(parts[2]) / 1e6
    except Exception:
        pass
    return None, None


# ── resolvers , one per metric, thin wrappers over existing accessors ───────

def _res_vram_fill_pct(entity_id, window_s):
    node = entity_id or "host"
    ev = _latest_event("snapshot", node=node, max_age_s=window_s or 30.0)
    if ev and ev.get("fb_phys_used_pct") is not None:
        return {"value": ev["fb_phys_used_pct"], "unit": "percent",
                "raw_source": "dataflux.snapshot.fb_phys_used_pct",
                "freshness_s": round(time.time() - ev.get("ts", 0), 1)}
    try:
        import gb_monitor
        s = gb_monitor.snapshot(probe_gpu=True)
        if s.loaded and s.gpu_mem_total_mb:
            pct = round(100.0 * s.gpu_mem_used_mb / s.gpu_mem_total_mb, 1)
            return {"value": pct, "unit": "percent",
                    "raw_source": "gb_monitor.snapshot (NVML, physical)"}
    except Exception as e:
        return {"value": None, "unit": "percent", "raw_source": "unavailable", "error": str(e)}
    return {"value": None, "unit": "percent", "raw_source": "unavailable",
            "error": "no fresh snapshot event and gb_monitor probe unavailable"}


def _res_t2_pressure_level(entity_id, window_s):
    try:
        import gb_monitor
        s = gb_monitor.snapshot(probe_gpu=False)
        if s.loaded:
            return {"value": s.t2_pressure, "unit": "enum_0_1_2",
                    "raw_source": "gb_monitor.GbSnapshot.t2_pressure"}
    except Exception as e:
        return {"value": None, "unit": "enum_0_1_2", "raw_source": "unavailable", "error": str(e)}
    return {"value": None, "unit": "enum_0_1_2", "raw_source": "unavailable",
            "error": "kernel module not loaded"}


def _res_t2_pressure_fraction(entity_id, window_s):
    ev = _latest_event("snapshot", node=entity_id or "host", max_age_s=window_s or 30.0)
    if ev and ev.get("t2_pressure") is not None:
        return {"value": ev["t2_pressure"], "unit": "fraction_0_1",
                "raw_source": "dataflux.snapshot.t2_pressure"}
    return {"value": None, "unit": "fraction_0_1", "raw_source": "no recent snapshot event"}


def _res_t3_spill_active_mb(entity_id, window_s):
    try:
        import gb_monitor
        s = gb_monitor.snapshot(probe_gpu=False)
        if s.loaded:
            return {"value": s.t3_used_mb, "unit": "megabytes",
                    "raw_source": "gb_monitor.GbSnapshot.t3_used_mb"}
    except Exception:
        pass
    ev = _latest_event("snapshot", node=entity_id or "host", max_age_s=window_s or 30.0)
    if ev and ev.get("t3_used_mb") is not None:
        return {"value": ev["t3_used_mb"], "unit": "megabytes",
                "raw_source": "dataflux.snapshot.t3_used_mb"}
    return {"value": None, "unit": "megabytes", "raw_source": "unavailable"}


def _res_kv_resident_share(entity_id, window_s):
    ev = _latest_event("snapshot", node=entity_id or "host", max_age_s=window_s or 30.0)
    if not ev:
        return {"value": None, "unit": "fraction_0_1", "raw_source": "no recent snapshot event"}
    kv_t1 = ev.get("kv_used_mb", 0) or 0
    kv_t2 = ev.get("kv_t2_mb", 0) or 0
    total = kv_t1 + kv_t2
    share = round(kv_t1 / total, 3) if total else None
    return {"value": share, "unit": "fraction_0_1",
            "raw_source": "dataflux.snapshot.kv_used_mb/kv_t2_mb"}


def _res_effective_bits(entity_id, window_s):
    ev = _latest_event("kernel_backend", max_age_s=window_s)
    if ev and "bits" in ev:
        return {"value": ev["bits"], "unit": "bits", "raw_source": "dataflux.kernel_backend.bits"}
    return {"value": None, "unit": "bits", "raw_source": "no kernel_backend events yet"}


_FLOOR_OK = {"16", "bf16", "fp16", "fp8", 16, 8}


def _res_meets_fp8_floor(entity_id, window_s):
    bits_val = _res_effective_bits(entity_id, window_s)
    bits = bits_val.get("value")
    if bits is None:
        return {"value": None, "unit": "boolean", "raw_source": "effective_bits unavailable"}
    ok = str(bits) in _FLOOR_OK or (isinstance(bits, (int, float)) and bits >= 8)
    return {"value": ok, "unit": "boolean", "raw_source": "derived from effective_bits"}


def _res_kv_bits_by_layer(entity_id, window_s):
    ev = _latest_event("turboquant_activate", max_age_s=window_s)
    n_kv_layers = None
    try:
        import gb_synapse
        for e in gb_synapse.list_models():
            if entity_id in (None, e.name):
                # is_recurrent_only: n_kv_layers==0 is the CORRECT count (no
                # real attention layer at all) — `or e.n_layers` must not
                # paper over that with "unknown, assume every layer is real
                # attention" (the same trap this metric's own never_use entry
                # already warns about for n_layers itself).
                n_kv_layers = 0 if e.is_recurrent_only else (e.n_kv_layers or e.n_layers)
                break
    except Exception:
        pass
    return {"value": {"k_bits": ev.get("k_bits") if ev else None,
                       "v_bits": ev.get("v_bits") if ev else None,
                       "n_kv_layers": n_kv_layers},
            "unit": "bits_per_layer_map",
            "raw_source": "dataflux.turboquant_activate + gb_synapse.ModelEntry.n_kv_layers"}


def _res_ssm_state_gb(entity_id, window_s):
    """Selective-SSM recurrent-state footprint (GiB) for the named model —
    estimate_ssm_state_gb()'s own value, the counterpart to kv_bits_by_layer
    above for the layers n_kv_layers EXCLUDES. Constant in ctx (see the
    resolved metric's own never_use trap: unlike KV cache this does NOT grow
    with context length)."""
    try:
        import gb_synapse
        for e in gb_synapse.list_models():
            if entity_id in (None, e.name):
                return {"value": round(gb_synapse._entry_ssm_gb(e), 3), "unit": "gigabytes",
                        "raw_source": "gb_synapse.estimate_ssm_state_gb via ModelEntry"}
    except Exception:
        pass
    return {"value": None, "unit": "gigabytes", "raw_source": "no matching model in manifest"}


def _res_recurrent_layer_fraction(entity_id, window_s):
    """n_recurrent_layers / n_layers for the named model — 0.0 for a plain
    transformer, 1.0 for a pure-Mamba/Mamba2 architecture (is_recurrent_only),
    and the hybrid ratio (e.g. 0.75 for this reference workload's 3-GDN-to-
    1-attention mix) in between."""
    try:
        import gb_synapse
        for e in gb_synapse.list_models():
            if entity_id in (None, e.name) and e.n_layers > 0:
                return {"value": round(e.n_recurrent_layers / e.n_layers, 3), "unit": "fraction",
                        "raw_source": "gb_synapse.ModelEntry.n_recurrent_layers/n_layers"}
    except Exception:
        pass
    return {"value": None, "unit": "fraction", "raw_source": "no matching model in manifest"}


def _res_tok_s_decode(entity_id, window_s):
    try:
        import gb_dataflux
        events = gb_dataflux.read_events(since_hours=(window_s / 3600.0) if window_s else 48)
        summary = gb_dataflux.summarize(events)
        tok_s = summary.get("tok_s", {})
        if entity_id and entity_id in tok_s:
            t = tok_s[entity_id]
            return {"value": t["latest"], "unit": "tokens_per_second",
                     "raw_source": "dataflux.tok_s_measured", "samples": t.get("samples")}
        if tok_s:
            best = max(tok_s.items(), key=lambda kv: kv[1].get("last_ts", 0))
            return {"value": best[1]["latest"], "unit": "tokens_per_second",
                    "raw_source": "dataflux.tok_s_measured", "model": best[0]}
    except Exception as e:
        return {"value": None, "unit": "tokens_per_second", "raw_source": "unavailable", "error": str(e)}
    return {"value": None, "unit": "tokens_per_second", "raw_source": "no tok_s_measured events"}


def _res_ttft_ms(entity_id, window_s):
    ev = _latest_event("prompt_cache", node=entity_id, max_age_s=window_s)
    if ev and ev.get("ttft_ms") is not None:
        return {"value": ev["ttft_ms"], "unit": "milliseconds", "raw_source": "dataflux.prompt_cache.ttft_ms"}
    return {"value": None, "unit": "milliseconds", "raw_source": "no prompt_cache events yet"}


def _res_prompt_cache_hit_pct(entity_id, window_s):
    ev = _latest_event("prompt_cache", node=entity_id, max_age_s=window_s)
    if ev and ev.get("hit_pct") is not None:
        return {"value": ev["hit_pct"], "unit": "percent", "raw_source": "dataflux.prompt_cache.hit_pct"}
    return {"value": None, "unit": "percent", "raw_source": "no prompt_cache events yet"}


def _res_served_ctx(entity_id, window_s):
    try:
        import gb_synapse
        sessions = gb_synapse.ps()
        if entity_id:
            sessions = [s for s in sessions if s.get("model") == entity_id]
        if sessions:
            return {"value": sessions[0].get("ctx"), "unit": "tokens",
                    "raw_source": "gb_synapse.ps() run-state (server-reported at serve time)"}
    except Exception as e:
        return {"value": None, "unit": "tokens", "raw_source": "unavailable", "error": str(e)}
    return {"value": None, "unit": "tokens", "raw_source": "no serve session running"}


def _res_n_gpu_layers_effective(entity_id, window_s):
    try:
        import gb_synapse
        sessions = gb_synapse.ps()
        if entity_id:
            sessions = [s for s in sessions if s.get("model") == entity_id]
        if sessions:
            return {"value": sessions[0].get("n_gpu_layers"), "unit": "layer_count",
                    "raw_source": "gb_synapse.ps()"}
    except Exception as e:
        return {"value": None, "unit": "layer_count", "raw_source": "unavailable", "error": str(e)}
    return {"value": None, "unit": "layer_count", "raw_source": "no serve session running"}


def _res_cluster_vram_fill_pct(entity_id, window_s):
    try:
        import gb_cluster
        snap = gb_cluster.cluster_snapshot()
        nodes = snap.get("nodes", {}) if isinstance(snap, dict) else {}
        out = {}
        for name, info in (nodes or {}).items():
            total = info.get("vram_total_mb") or info.get("t1_vram_mb")
            used = info.get("vram_used_mb") or info.get("t1_used_mb")
            if total:
                out[name] = round(100.0 * used / total, 1)
        if entity_id:
            return {"value": out.get(entity_id), "unit": "percent", "raw_source": "gb_cluster.cluster_snapshot"}
        return {"value": out, "unit": "percent", "raw_source": "gb_cluster.cluster_snapshot"}
    except Exception as e:
        return {"value": None, "unit": "percent", "raw_source": "unavailable", "error": str(e)}


def _res_feeder_items(entity_id, window_s):
    try:
        import gb_dataflux
        events = gb_dataflux.read_events(since_hours=(window_s / 3600.0) if window_s else 24)
        summary = gb_dataflux.summarize(events)
        nodes = summary.get("nodes", {})
        if entity_id:
            n = nodes.get(entity_id)
            return {"value": (n or {}).get("items"), "unit": "count", "raw_source": "dataflux.summarize.nodes"}
        return {"value": {k: v.get("items") for k, v in nodes.items()}, "unit": "count",
                "raw_source": "dataflux.summarize.nodes"}
    except Exception as e:
        return {"value": None, "unit": "count", "raw_source": "unavailable", "error": str(e)}


def _res_link_bw_gbps(entity_id, window_s):
    ev = _latest_event("link_transfer", max_age_s=window_s)
    if ev and ev.get("seconds"):
        gbps = round((ev.get("nbytes", 0) * 8) / ev["seconds"] / 1e9, 3)
        return {"value": gbps, "unit": "gigabits_per_second", "raw_source": "dataflux.link_transfer"}
    return {"value": None, "unit": "gigabits_per_second", "raw_source": "no link_transfer events"}


def _res_quant_rel_error(entity_id, window_s):
    """gb_quant.plan_quality()'s measured mean/max relative error for the
    per-layer bit-assignment it actually chose , the real answer to "did
    quantizing this model to the target bit width cost meaningful accuracy",
    as opposed to assuming a nominal bit-width is automatically within
    tolerance (`quant_plan` can silently bf16-escalate individual layers,
    so the nominal target alone doesn't tell you the actual error either).
    `quant_plan` events are keyed by model in `items`, not `node` , no
    `node`-based filter fits, so this reads dataflux directly rather than
    going through `_latest_event`'s node-only filter."""
    try:
        import gb_dataflux
        events = gb_dataflux.read_events(since_hours=(window_s / 3600.0) if window_s else 48)
    except Exception as e:
        return {"value": None, "unit": "fraction", "raw_source": "unavailable", "error": str(e)}
    cand = [e for e in events if e.get("kind") == "quant_plan"]
    if entity_id:
        cand = [e for e in cand if entity_id in (e.get("items") or [])]
    if not cand:
        return {"value": None, "unit": "fraction",
                "raw_source": "no quant_plan events yet for this model" if entity_id
                else "no quant_plan events yet"}
    ev = max(cand, key=lambda e: e.get("ts", 0))
    return {
        "value": {"mean_rel_err": ev.get("mean_rel_err"), "max_rel_err": ev.get("max_rel_err")},
        "unit": "fraction", "raw_source": "dataflux.quant_plan",
        "target_bits": ev.get("target"), "error_ceiling": ev.get("error_ceiling"),
        "bf16_kept_layers": ev.get("bf16_kept"),
    }


def _res_swap_pressure(entity_id, window_s):
    total, used = _free_swap_mb()
    if total is None:
        return {"value": None, "unit": "percent", "raw_source": "unavailable"}
    pct = round(100.0 * used / total, 1) if total else 0.0
    return {"value": pct, "unit": "percent", "raw_source": "free -b (swap row)"}


def _res_iowait_pct(entity_id, window_s):
    try:
        out = subprocess.run(["vmstat", "1", "2"], capture_output=True, text=True, timeout=5).stdout
        lines = [l for l in out.splitlines() if l.strip() and l.strip()[0].isdigit()]
        if lines:
            wa = float(lines[-1].split()[-1])
            return {"value": wa, "unit": "percent", "raw_source": "vmstat 1 2 (wa column)"}
    except Exception as e:
        return {"value": None, "unit": "percent", "raw_source": "unavailable", "error": str(e)}
    return {"value": None, "unit": "percent", "raw_source": "vmstat returned no sample lines"}


def _res_kmod_loaded(entity_id, window_s):
    try:
        loaded = Path("/dev/greenboost").exists()
        if not loaded:
            out = subprocess.run(["lsmod"], capture_output=True, text=True, timeout=5).stdout
            loaded = "greenboost" in out
        return {"value": loaded, "unit": "boolean", "raw_source": "/dev/greenboost + lsmod"}
    except Exception as e:
        return {"value": None, "unit": "boolean", "raw_source": "unavailable", "error": str(e)}


def _res_shim_fresh(entity_id, window_s):
    try:
        p = Path("/run/greenboost/shim_stats")
        if not p.exists():
            return {"value": False, "unit": "boolean", "raw_source": "/run/greenboost/shim_stats missing"}
        age = time.time() - p.stat().st_mtime
        return {"value": age <= (window_s or 30.0), "unit": "boolean",
                "age_s": round(age, 1), "raw_source": "/run/greenboost/shim_stats mtime"}
    except Exception as e:
        return {"value": None, "unit": "boolean", "raw_source": "unavailable", "error": str(e)}


# ── the compile step: metric name -> value + provenance + threshold verdict ─

def _threshold_state(metric: Metric, value) -> str:
    th = metric.threshold or {}
    if value is None or not th:
        return "unknown"
    try:
        if "target_band" in th:
            lo, hi = th["target_band"]
            if value < lo:
                return "below_target"
            if value > hi:
                return "above_target"
            return "in_target"
        if "critical_at" in th and value >= th["critical_at"]:
            return "critical"
        if "warn_at" in th and value >= th["warn_at"]:
            return "warn"
        if "critical_above" in th and value > th["critical_above"]:
            return "critical"
        if "critical_below" in th and value < th["critical_below"]:
            return "critical"
        if "warn_above" in th and value > th["warn_above"]:
            return "warn"
        if "warn_below" in th and value < th["warn_below"]:
            return "warn"
        return "ok"
    except Exception:
        return "unknown"


def resolve(metric_name: str, entity_id: "str | None" = None,
            window_s: "float | None" = None) -> dict:
    """THE deterministic compile step: a governed metric name -> one value +
    provenance + threshold verdict. Never raises , failures fold into the
    returned dict (`error` key) so a caller always gets a well-formed answer,
    matching gb_dataflux.emit()'s own never-raise contract."""
    L = load()
    metric = L["metrics"].get(metric_name) or _alias_lookup(metric_name, L["metrics"])
    if metric is None:
        return {"metric": metric_name, "governed": False,
                "error": f"unknown metric '{metric_name}'",
                "known_metrics": sorted(L["metrics"].keys())}
    fn = globals().get(_RESOLVER_PREFIX + metric.resolver)
    if fn is None:
        return {"metric": metric.name, "governed": True,
                "error": f"resolver function missing: _res_{metric.resolver}"}
    t0 = time.time()
    try:
        raw = fn(entity_id, window_s)
    except Exception as e:
        raw = {"value": None, "unit": metric.unit, "raw_source": "resolver raised", "error": str(e)}
    value = raw.get("value")
    out = {
        "metric": metric.name,
        "value": value,
        "unit": raw.get("unit", metric.unit),
        "entity": metric.entity,
        "doc": metric.doc,
        "threshold_state": _threshold_state(metric, value),
        "governed": True,
        "provenance": {
            "source_tier": metric.source_tier,
            "raw_source": raw.get("raw_source", ""),
            "owner": metric.owner,
            "resolved_at": round(t0, 3),
            "freshness_s": raw.get("freshness_s"),
        },
        "never_use": [n["field"] for n in metric.never_use],
    }
    if "error" in raw:
        out["error"] = raw["error"]
    return out


def discover(query: str, k: int = 5) -> list[dict]:
    """Keyword+alias search over metric/segment docs , the retrieval-failure
    fix: narrows the ~60-tool surface to a handful of named answers BEFORE
    any tool call."""
    L = load()
    q = query.lower()
    words = q.split()
    scored = []
    for m in L["metrics"].values():
        haystack = " ".join([m.name, m.doc] + list(m.aliases)).lower()
        score = sum(1 for w in words if w in haystack)
        if score:
            scored.append((score, "metric", m.name, m.doc))
    for s in L["segments"].values():
        haystack = (s.name + " " + s.doc).lower()
        score = sum(1 for w in words if w in haystack)
        if score:
            scored.append((score, "segment", s.name, s.doc))
    scored.sort(key=lambda t: -t[0])
    return [{"kind": kind, "name": name, "doc": doc} for _, kind, name, doc in scored[:k]]


# ── segment evaluators , canonical filters, each {matched, evidence} ────────

def _seg_rule1_underfilled():
    vram = resolve("vram_fill_pct")
    t2f = resolve("t2_pressure_fraction")
    t3 = resolve("t3_spill_active_mb")
    v = vram.get("value")
    overflow_active = (t2f.get("value") or 0) > 0 or (t3.get("value") or 0) > 0
    matched = v is not None and v < 85.0 and overflow_active
    return matched, [vram, t2f, t3]


def _seg_weights_on_t3():
    t3 = resolve("t3_spill_active_mb")
    v = t3.get("value")
    return bool(v) and v > 0, [t3]


def _seg_feeder_idle_while_host_saturated():
    host_fill = resolve("vram_fill_pct", entity_id="host")
    feeders = resolve("feeder_items")
    v_host = host_fill.get("value")
    items = feeders.get("value") or {}
    idle_feeders = [n for n, c in items.items() if n != "host" and (c or 0) == 0]
    matched = bool(idle_feeders) and v_host is not None and v_host >= 85.0
    return matched, [host_fill, feeders]


def _seg_below_quality_floor():
    floor = resolve("meets_fp8_floor")
    bits = resolve("effective_bits")
    return floor.get("value") is False, [floor, bits]


def _seg_swap_thrash_not_gpu_throttle():
    swap = resolve("swap_pressure")
    iow = resolve("iowait_pct")
    sv, iv = swap.get("value"), iow.get("value")
    matched = (sv is not None and sv > 10.0) and (iv is not None and iv > 50.0)
    return matched, [swap, iow]


def _seg_kmod_missing_silent_degrade():
    kmod = resolve("kmod_loaded")
    fresh = resolve("shim_fresh")
    matched = kmod.get("value") is False and fresh.get("value") is True
    return matched, [kmod, fresh]


def _seg_prompt_cache_cold():
    hit = resolve("prompt_cache_hit_pct")
    v = hit.get("value")
    return (v is not None and v < 10.0), [hit]


def _seg_serve_healthy():
    kmod = resolve("kmod_loaded")
    fresh = resolve("shim_fresh")
    vram = resolve("vram_fill_pct")
    floor = resolve("meets_fp8_floor")
    matched = (kmod.get("value") is True and fresh.get("value") is True
               and (vram.get("value") or 0) >= 60.0 and floor.get("value") is not False)
    return matched, [kmod, fresh, vram, floor]


def evaluate_segment(name: str) -> dict:
    L = load()
    seg = L["segments"].get(name)
    if seg is None:
        return {"segment": name, "error": "unknown segment",
                "known_segments": sorted(L["segments"].keys())}
    fn = globals().get(_EVAL_PREFIX + seg.evaluator)
    if fn is None:
        return {"segment": seg.name, "error": f"evaluator missing: _seg_{seg.evaluator}"}
    try:
        matched, evidence = fn()
    except Exception as e:
        return {"segment": seg.name, "error": str(e), "matched": None, "evidence": []}
    return {"segment": seg.name, "doc": seg.doc, "severity": seg.severity,
            "matched": matched, "evidence": evidence}


# ── question routing + provenance footer ────────────────────────────────────

def _provenance_footer(metrics: dict) -> str:
    tiers = sorted({m.get("provenance", {}).get("source_tier", "?") for m in metrics.values()})
    owners = sorted({m.get("provenance", {}).get("owner", "?")
                     for m in metrics.values() if m.get("provenance", {}).get("owner")})
    return (f"Source: semantic layer ({'/'.join(tiers) or 'n/a'}) · "
            f"Owner: {'/'.join(owners) or 'n/a'} · Governed: true")


def answer(question: str) -> dict:
    """Full route: question -> matching route -> resolved metrics + evaluated
    segments + provenance footer. Falls back to `governed: false` with a
    named raw-tool suggestion when no route covers the question , that
    footer is the signal a consuming agent must surface to the user, per
    the gb-semantics SKILL.md's mandatory-default-path instruction."""
    L = load()
    q = question.lower()
    best = None
    for r in L["routes"]:
        if any(t in q for t in r.triggers):
            best = r
            break
    if best is None:
        return {"question": question, "governed": False,
                "note": "no governed route matched; try semantic_metrics(query=...) "
                        "to discover a metric by keyword, or fall back to a raw MCP "
                        "tool and say so in the answer",
                "footer": "Source: none · Confidence: low · Governed: false"}
    metrics = {m: resolve(m) for m in best.metrics}
    segments = {s: evaluate_segment(s) for s in best.segments}
    matched_segments = [s for s, v in segments.items() if v.get("matched")]
    return {
        "question": question, "intent": best.intent, "governed": True,
        "metrics": metrics, "segments": segments, "matched_segments": matched_segments,
        "raw_tool_fallback": best.raw_tool_fallback or None,
        "footer": _provenance_footer(metrics),
    }


# ── bounded prompt card for non-MCP consumers ───────────────────────────────

_CARD_METRICS = ("vram_fill_pct", "kmod_loaded", "shim_fresh", "meets_fp8_floor")
_CARD_SEGMENTS = ("rule1_underfilled", "kmod_missing_silent_degrade",
                  "swap_thrash_not_gpu_throttle", "weights_on_t3")


def card() -> str:
    """Bounded ~1.5KB prompt block for consumers without MCP access (GB-CLI,
    ai-forge, optimal-claude). Mirrors gb_monitor.context_summary()'s style:
    one status line + only-if-relevant warnings. Placed AFTER the existing
    tier banner by gb_monitor.context_summary() so it inherits that
    function's stable position in every consumer's prompt."""
    try:
        lines = ["\n- GB-Semantics governed layer active: resolve GreenBoost "
                 "state questions through it FIRST (semantic_resolve/"
                 "semantic_answer over MCP, `gb semantics answer \"...\"` "
                 "otherwise). Raw telemetry fields are the fallback only; a "
                 "raw-field answer must say so.\n"]
        for name in _CARD_METRICS:
            r = resolve(name)
            v = r.get("value")
            if v is None:
                continue
            state = r.get("threshold_state", "unknown")
            flag = "" if state in ("ok", "in_target", "unknown") else f" [{state.upper()}]"
            lines.append(f"  - {name}={v}{flag}\n")
        matched = [seg for seg in _CARD_SEGMENTS if evaluate_segment(seg).get("matched")]
        if matched:
            lines.append(f"  - ACTIVE SEGMENTS: {', '.join(matched)}\n")
        return "".join(lines)
    except Exception:
        return ""


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "answer":
        print(json.dumps(answer(" ".join(sys.argv[2:])), indent=2, default=str))
    elif len(sys.argv) > 1 and sys.argv[1] == "card":
        print(card())
    elif len(sys.argv) > 2 and sys.argv[1] == "resolve":
        print(json.dumps(resolve(sys.argv[2]), indent=2, default=str))
    else:
        print("usage: gb_semantics.py {answer '<question>'|resolve <metric>|card}")

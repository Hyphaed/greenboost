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

import os
import subprocess
import time
from dataclasses import dataclass, field
import re as _re
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


def _events_in_window(kind: str, node: "str | None" = None,
                      max_age_s: "float | None" = None,
                      days: float = 2.0) -> "list[dict]":
    """All events of a kind inside the freshness window, newest last.

    `_latest_event` answers "what happened to the last one", which is the wrong
    shape for any metric that is a RATE. Same contract otherwise: an empty list
    means unavailable, and a caller must not report it as a zero.
    """
    try:
        import gb_dataflux
    except Exception:
        return []
    try:
        events = gb_dataflux.read_events(since_hours=days * 24)
    except Exception:
        return []
    cand = [e for e in events if e.get("kind") == kind]
    if node:
        cand = [e for e in cand if gb_dataflux.canonical_node(e.get("node")) == node]
    if max_age_s is not None:
        now = time.time()
        cand = [e for e in cand if (now - e.get("ts", 0)) <= max_age_s]
    return sorted(cand, key=lambda e: e.get("ts", 0))


# Minimum GPU utilization (%) before a PCIe gen reading is trusted. Mirrors
# gb_telemetry._PCIE_ACTIVE_UTIL_PCT deliberately rather than importing it:
# this module wraps existing accessors and must not pull in the NVML stack
# just to read one threshold. Keep the two in step.
_PCIE_ACTIVE_UTIL_PCT = 25

# Below this GPU utilization a serve session counts as doing no work, so its
# resident VRAM is "held while idle" rather than "in use".
_GPU_BUSY_UTIL_PCT = 10

# VRAM fill (%) at which idle residency is worth surfacing. Below this the card
# still has room and there is nothing for the operator to decide.
# Rule #1's own words: "~90% occupancy (10% headroom so the system never
# collapses under memory pressure)". The reserve IS the rule, so the threshold
# is the rule's own number rather than a tuned one. A percentage of the card's
# own capacity, never an absolute MB figure , see CLAUDE.md's prohibition on
# hardware-shaped constants.
_VRAM_MIN_HEADROOM_PCT = 10.0
_IDLE_HOLD_FILL_PCT = 60

# Share of prompt_ms attributable to recurrent replay before it counts as
# dominating the turn's prefill. Half is the natural line: past it, most of
# what the operator waits for is work already done once.
_REPLAY_DOMINANT_SHARE = 0.5


def _latest_event_with_field(kind: str, field: str, node: "str | None" = None,
                             max_age_s: "float | None" = None,
                             days: float = 2.0) -> "dict | None":
    """Latest event of `kind` that actually CARRIES `field` (non-null).

    `_latest_event` takes the newest event of a kind and then checks the field
    on it. That is wrong for any field emitted conditionally: a single newer
    event without the field hides every older one that has it, and the metric
    reads as "no data" while the data exists. Same freshness contract —
    None means unavailable, never zero."""
    try:
        import gb_dataflux
    except Exception:
        return None
    try:
        events = gb_dataflux.read_events(since_hours=days * 24)
    except Exception:
        return None
    cand = [e for e in events if e.get("kind") == kind and e.get(field) is not None]
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


def _res_vram_headroom_pct(entity_id, window_s):
    """How much physical VRAM is NOT occupied , Rule #1's safety margin.

    Rule #1 is two clauses, and only the first one had a governed reading
    before this metric existed: "drive VRAM to ~90%" AND "10% headroom so the
    system never collapses under memory pressure". `vram_fill_pct` answers the
    first. This answers the second, which is the one that bites: a card at
    97.9% is not 8 points better than one at 90%, it is 8 points into the
    reserve that exists precisely so an allocation spike has somewhere to go.

    Measured 2026-08-20, three sessions of the SAME model at a byte-identical
    serve config (weights 15.85 GB, ctx 24576, q4_0, 65 layers, mtp_draft_n 4):
    93.8% fill decoded at 8.45 tok/s, 90.9% at 4.90, and 97.9% , 254 MB free
    on the whole card , at 3.00, with TTFT rising 506 ms to 6426 ms at an
    unchanged 99.7% prompt-cache hit. Correlation on n=3, not a proven cause;
    what is certain is that nothing in the governed layer said a word about it.
    """
    fill = resolve("vram_fill_pct", entity_id=entity_id, window_s=window_s)
    v = fill.get("value")
    prov = fill.get("provenance") or {}
    src = prov.get("raw_source") or fill.get("raw_source") or "unavailable"
    if v is None:
        return {"value": None, "unit": "percent", "raw_source": src,
                "error": "vram_fill_pct did not resolve: %s"
                         % (fill.get("error") or "no source")}
    return {"value": round(100.0 - float(v), 1), "unit": "percent",
            "raw_source": "derived: 100 - (%s)" % src,
            "freshness_s": prov.get("freshness_s")}


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


#: Share of tokens in the slow tail above which the tail, not the average, is
#: the thing to investigate. Dimensionless, so it carries across models and
#: boxes , this is a shape threshold, not a hardware one.
_TAIL_HEAVY_RATIO = 0.10


def _res_inter_token_p95_ms(entity_id, window_s):
    """95th-percentile gap between generated tokens, from the newest streamed
    sample that actually carries one.

    This is the signal `tok_s` cannot give: tok_s is a mean over a whole
    response, and with MTP speculative decode the real distribution is bimodal
    , near-zero inside an accepted draft batch, a full forward pass at the
    boundary. A user feels the boundary.
    """
    ev = _latest_event_with_field("tok_s_measured", "p95_ms", max_age_s=window_s)
    if ev is None:
        return {"value": None, "unit": "milliseconds",
                "raw_source": "no tok_s_measured event carrying 'p95_ms' yet "
                              "(non-streaming replies never carry one)"}
    return {"value": ev.get("p95_ms"), "unit": "milliseconds",
            "raw_source": "dataflux.tok_s_measured.p95_ms"}


def _res_slow_token_ratio(entity_id, window_s):
    """Share of tokens whose gap exceeded twice that response's median gap.

    Relative to the response's own median on purpose: what counts as slow
    depends on model, quant and how much of the model is streaming over PCIe.
    """
    ev = _latest_event_with_field("tok_s_measured", "slow_token_ratio",
                                  max_age_s=window_s)
    if ev is None:
        return {"value": None, "unit": "fraction_0_1",
                "raw_source": "no tok_s_measured event carrying "
                              "'slow_token_ratio' yet"}
    return {"value": ev.get("slow_token_ratio"), "unit": "fraction_0_1",
            "raw_source": "dataflux.tok_s_measured.slow_token_ratio"}


def _res_gaming_mode_active(entity_id, window_s):
    """greenboost.ko's `gaming_mode` flag, read live.

    This is the flag that parks inference T2 buffers at the LRU tail. It is
    SET by the Proton wrapper at launch and cleared at exit , so a session
    that died hard leaves it at 1 with no game running, and every inference
    allocation after that quietly loses its place in the queue. Nothing used
    to report that state; this metric is what makes it visible.
    """
    try:
        import gb_monitor
        snap = gb_monitor.snapshot(probe_gpu=False)
        if snap.loaded:
            return {"value": bool(snap.gaming_mode), "unit": "bool",
                    "raw_source": "gb_monitor.GbSnapshot.gaming_mode"}
    except Exception:
        pass
    # Kernel module not loaded / not readable through the ioctl , the sysfs
    # parameter is the same value by another door.
    try:
        with open("/sys/module/greenboost/parameters/gaming_mode") as fh:
            return {"value": fh.read().strip() not in ("0", "N", ""), "unit": "bool",
                    "raw_source": "/sys/module/greenboost/parameters/gaming_mode"}
    except OSError:
        pass
    return {"value": None, "unit": "bool",
            "raw_source": "unavailable , greenboost.ko not loaded"}


def _res_gaming_session_orphans(entity_id, window_s):
    """Processes that survived BOTH signals on the last stop request.

    `orphans` is emitted only on `action="terminated"` events, so this must
    search for the newest event that actually carries the field , taking the
    newest gaming_session event of any action would let one ordinary `start`
    hide every real teardown result (the exact shape of the 2026-08-18
    ttft_ms defect).
    """
    ev = _latest_event_with_field("gaming_session", "orphans", max_age_s=window_s, days=7.0)
    if ev is None:
        return {"value": None, "unit": "count",
                "raw_source": "no gaming_session event carrying 'orphans' yet"}
    return {"value": ev.get("orphans"), "unit": "count",
            "raw_source": "dataflux.gaming_session.orphans"}


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
    """Latest measured TTFT.

    Scans back for the most recent event that CARRIES ttft_ms rather than
    reading only the newest prompt_cache event: the field is emitted
    conditionally (gb_synapse.record_prompt_cache_sample omits it when the
    proxy could not latch a first-token timestamp), so one such event at the
    head of the log used to mask every good sample behind it.

    Deliberately does NOT fall back to engine_prompt_ms. They answer different
    questions and the gap between them is the diagnostic value — see
    _res_engine_prefill_ms's docstring. Conflating them would report queueing
    and contention as prefill cost.
    """
    ev = _latest_event_with_field("prompt_cache", "ttft_ms",
                                  node=entity_id, max_age_s=window_s)
    if ev is not None:
        return {"value": ev["ttft_ms"], "unit": "milliseconds",
                "raw_source": "dataflux.prompt_cache.ttft_ms"}
    # Say which of the two failure modes actually happened. Reporting "no
    # prompt_cache events yet" while 138 of them existed sent a 2026-08-18
    # audit looking in the wrong place; the real cause was that every one of
    # them came from a tool-call turn, whose SSE frames the proxy did not
    # recognise as generated content (fixed in gb_synapse_api.py).
    any_ev = _latest_event("prompt_cache", node=entity_id, max_age_s=window_s)
    reason = ("prompt_cache events exist but none carried ttft_ms"
              if any_ev is not None else "no prompt_cache events yet")
    return {"value": None, "unit": "milliseconds", "raw_source": reason}


def _res_engine_prefill_ms(entity_id, window_s):
    """The engine's own prefill time, the companion to ttft_ms.

    Kept as a separate metric rather than folded into ttft_ms because the two
    answer different questions and the difference between them is the whole
    diagnostic value: ttft_ms is what the client waited, engine_prefill_ms is
    what prefill actually cost, and the gap is queueing plus contention."""
    ev = _latest_event("prompt_cache", node=entity_id, max_age_s=window_s)
    if ev and ev.get("engine_prompt_ms") is not None:
        return {"value": ev["engine_prompt_ms"], "unit": "milliseconds",
                "raw_source": "dataflux.prompt_cache.engine_prompt_ms"}
    return {"value": None, "unit": "milliseconds",
            "raw_source": "no prompt_cache event carrying engine_prompt_ms yet"}


def _res_prompt_depth_tokens(entity_id, window_s):
    ev = _latest_event("prompt_cache", node=entity_id, max_age_s=window_s)
    if ev and ev.get("prompt_tokens") is not None:
        return {"value": ev["prompt_tokens"], "unit": "count",
                "raw_source": "dataflux.prompt_cache.prompt_tokens"}
    return {"value": None, "unit": "count",
            "raw_source": "no prompt_cache event carrying prompt_tokens yet"}


def _res_prompt_cache_hit_pct(entity_id, window_s):
    ev = _latest_event("prompt_cache", node=entity_id, max_age_s=window_s)
    if ev and ev.get("hit_pct") is not None:
        return {"value": ev["hit_pct"], "unit": "percent", "raw_source": "dataflux.prompt_cache.hit_pct"}
    return {"value": None, "unit": "percent", "raw_source": "no prompt_cache events yet"}


def _res_slot_pin_rate_pct(entity_id, window_s):
    """GB-1: how often a conversation got the slot that still held its KV.

    Counted over the window rather than read off the newest event, because one
    event answers "what happened to that request", not "is routing working".
    Returns None with the reason named when there are no events , an absent
    measurement is not a 0%, and reporting it as one would read as a total
    failure of a feature that simply has not been exercised yet.
    """
    evs = _events_in_window("cache_index", node=entity_id, max_age_s=window_s)
    if not evs:
        return {"value": None, "unit": "percent",
                "raw_source": "no cache_index events in window (slot pinning "
                              "may be off: GB_SYNAPSE_SLOT_PIN=0, or the engine "
                              "reports a single slot)"}
    kept = sum(1 for e in evs if str(e.get("decision", "")).startswith("pinned"))
    return {"value": round(100.0 * kept / len(evs), 1), "unit": "percent",
            "raw_source": f"dataflux.cache_index.decision over {len(evs)} events"}


def _res_last_edit_chunk(entity_id, window_s):
    """GB-1: which chunk the most recent turn changed, if any."""
    ev = _latest_event("cache_index", node=entity_id, max_age_s=window_s)
    if not ev:
        return {"value": None, "unit": "index",
                "raw_source": "no cache_index events yet"}
    if "changed_chunk" not in ev:
        return {"value": None, "unit": "index",
                "raw_source": "cache_index event carries no changed_chunk field"}
    return {"value": ev.get("changed_chunk"), "unit": "index",
            "raw_source": "dataflux.cache_index.changed_chunk"
                          + (" (null = appended only)" if ev.get("changed_chunk") is None else "")}


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


def _proxy_mem() -> "tuple[float | None, float | None, str]":
    """(rss_mb, addr_space_cap_mb, provenance) for the gb-synapse proxy.

    The proxy runs under an RLIMIT_AS cap (_proxy_resource_limits in
    gb_synapse.py) so a runaway allocation raises MemoryError in ONE request
    instead of inviting the OOM killer to take the whole session. That makes
    "how close is the proxy to its cap" a real operational question, and until
    2026-08-19 nothing could answer it: the proxy hit the cap, raised
    MemoryError mid-generation, and the only symptom the operator saw was
    "gb-synapse stopped responding while generating".
    """
    try:
        import glob as _glob
        for cmd in _glob.glob("/proc/[0-9]*/cmdline"):
            try:
                with open(cmd, "rb") as f:
                    argv = f.read().decode("utf-8", "replace")
            except OSError:
                continue
            # Match an argv TOKEN, not the raw cmdline. /proc cmdline is
            # NUL-separated, and a substring test over the joined string
            # matches any shell that merely MENTIONS the proxy , including
            # the grep that is looking for it, which is how this resolver
            # first reported 7.6 MB of "proxy" that was really a pipeline.
            tokens = [t for t in argv.split("\0") if t]
            if not tokens:
                continue
            # argv[0] must be a python interpreter AND some later token must be
            # the script path itself , a bare endswith() over tokens is not
            # enough, because a shell invoked as
            #   bash -c "pgrep -af gb_synapse_api.py"
            # carries the whole command as ONE token that also ends with the
            # script name. Requiring no embedded whitespace rules that out.
            if not os.path.basename(tokens[0]).startswith("python"):
                continue
            if not any(t.endswith("gb_synapse_api.py") and len(t.split()) == 1
                       for t in tokens[1:]):
                continue
            # Derive the process directory from the entry we actually
            # matched rather than indexing a fixed path component , the
            # latter silently reads the wrong pid the moment the glob root
            # is not literally /proc.
            pdir = os.path.dirname(cmd)
            if os.path.basename(pdir) == str(os.getpid()):
                continue
            rss = cap = None
            with open(os.path.join(pdir, "status")) as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = float(line.split()[1]) / 1024
            with open(os.path.join(pdir, "limits")) as f:
                for line in f:
                    if line.startswith("Max address space"):
                        tok = line.split()[3]
                        cap = None if tok == "unlimited" else float(tok) / (1024 * 1024)
            return rss, cap, f"{pdir}/{{status,limits}}"
        return None, None, "no gb_synapse_api process is running"
    except Exception as e:
        return None, None, f"could not read /proc: {e}"


def _res_proxy_rss_mb(entity_id, window_s):
    """Resident memory of the gb-synapse proxy, in MB."""
    rss, _cap, prov = _proxy_mem()
    return {"value": None if rss is None else round(rss, 1),
            "unit": "megabytes", "raw_source": prov}


def _res_proxy_mem_headroom_pct(entity_id, window_s):
    """How much of the proxy's address-space cap is still unused, in percent.

    Falls to None (never to 100) when the proxy is not running or carries no
    cap , "I cannot tell" and "there is plenty of room" are different answers,
    and reporting the second for the first is the failure this layer exists to
    prevent.
    """
    rss, cap, prov = _proxy_mem()
    if rss is None or not cap:
        why = prov if rss is None else "proxy runs with an unlimited address space"
        return {"value": None, "unit": "percent", "raw_source": why}
    return {"value": round(max(0.0, 100.0 * (1.0 - rss / cap)), 1),
            "unit": "percent", "raw_source": f"{prov} , VmRSS vs RLIMIT_AS"}


def _agent_events(kind: str, window_s=None, days: float = 2.0) -> list:
    """All events of an agent kind in the window, oldest first."""
    try:
        import gb_dataflux
        hours = (window_s / 3600.0) if window_s else days * 24
        return [e for e in gb_dataflux.read_events(since_hours=hours)
                if e.get("kind") == kind]
    except Exception:
        return []


def _res_agent_eval_overall(entity_id, window_s):
    """greenboost-cli's own agent benchmark score, 0-1 (AE-1).

    Comparable ONLY against a run of the same model: MCP-Bench puts
    gemma-3-27b-it at 0.582 and qwen3-30b-a3b at 0.627 against gpt-5 at 0.749,
    so a score means nothing without knowing which model produced it.
    """
    ev = _agent_events("agent_eval_run", window_s)
    if not ev:
        return {"value": None, "unit": "score",
                "raw_source": "no agent_eval_run events yet , "
                              "run greenboost_cli.bench.agent_eval"}
    last = ev[-1]
    v = last.get("overall")
    if v is None:
        return {"value": None, "unit": "score",
                "raw_source": "latest agent_eval_run carried no 'overall'"}
    return {"value": round(float(v), 3), "unit": "score",
            "raw_source": f"agent_eval_run.overall (model={last.get('model', '?')})"}


def _res_agent_hallucinated_tool_pct(entity_id, window_s):
    """Share of tool calls naming a tool the registry does not have (AE-5).

    PATool's finding is that small models fail tool use by SCHEMA
    MISALIGNMENT , they emit names learned in pretraining rather than the
    names offered. This repo has already seen it live: the model emitted
    `mcp__knowledge-rag__search_knowledge` against a registry that only knew
    the bare name. This is that failure rate, measured instead of assumed.
    """
    ev = _agent_events("agent_tool_schema_miss", window_s)
    if not ev:
        return {"value": None, "unit": "percent",
                "raw_source": "no agent_tool_schema_miss events yet"}
    missed = sum(1 for e in ev if e.get("outcome") not in ("resolved", "ok"))
    return {"value": round(100.0 * missed / len(ev), 1), "unit": "percent",
            "raw_source": f"agent_tool_schema_miss over {len(ev)} call(s)"}


def _res_agent_compaction_prefix_kept_pct(entity_id, window_s):
    """Share of compactions that preserved the reusable prompt prefix (AE-2).

    The whole point of prefix-stable compaction. A compaction that rewrites
    the prefix throws away the engine's KV reuse , dataflux has recorded a
    97.9% prompt-cache hit on a 64k window here, and at ~5 tok/s re-prefilling
    that is minutes of wall time per event.
    """
    ev = _agent_events("agent_context_edit", window_s)
    if not ev:
        return {"value": None, "unit": "percent",
                "raw_source": "no agent_context_edit events yet"}
    # head_kept > 0 is the ONLY thing that means the prefix survived.
    #
    # This used to accept `extended_prior is True` as equivalent, and that is
    # exactly backwards: extended_prior says a prior memory block was absorbed
    # out of the middle and folded into the new summary, which REWRITES it at a
    # new position. The prefix-destroying compaction found on 2026-08-21 emitted
    # {"head_kept": 0, "extended_prior": True} , scored as a success by this
    # resolver while it was throwing the whole prefix away. The 50% this
    # reported was therefore an over-estimate of a worse number.
    kept = sum(1 for e in ev if (e.get("head_kept") or 0) > 0)
    return {"value": round(100.0 * kept / len(ev), 1), "unit": "percent",
            "raw_source": f"agent_context_edit over {len(ev)} compaction(s)"}


def _res_kv_t2_resident_mb(entity_id, window_s):
    """KV bytes living in T2 right now , the only field that says whether
    lookahead KV prefetch has anything to prefetch.

    Read from the shim's `kv_t2_live_mb`. Until 2026-08-20 that counter was
    published only when GREENBOOST_KV_PREFETCH was already armed, so answering
    "is there KV in T2?" required turning on the feature the answer was meant
    to justify , and a reader seeing `kv_prefetch_t2_kv_mb=0` could not tell
    "no KV in T2" from "nobody measured". It is now written unconditionally.

    An older shim that predates that change returns None here, with the reason
    named, rather than a zero that would read as a clean bill of health.
    """
    fields, reason = _shim_stats()
    if fields is None:
        return {"value": None, "unit": "megabytes", "raw_source": reason}
    raw = fields
    if "kv_t2_live_mb" not in raw:
        return {"value": None, "unit": "megabytes",
                "raw_source": "this shim build does not publish kv_t2_live_mb "
                              "(added 2026-08-20) , rebuild and reinstall the "
                              "shim, or read kv_prefetch_t2_kv_mb with "
                              "GREENBOOST_KV_PREFETCH=stats armed"}
    try:
        return {"value": float(raw["kv_t2_live_mb"]), "unit": "megabytes",
                "raw_source": "shim_stats.kv_t2_live_mb"}
    except (TypeError, ValueError):
        return {"value": None, "unit": "megabytes",
                "raw_source": "shim_stats.kv_t2_live_mb was unparseable"}


def _res_kv_prefetch_opportunities(entity_id, window_s):
    """Ticks where the shim saw T2-resident KV AND room in the T1 reserve.

    Zero with the counter armed is a real, useful answer: it means the
    prefetch branch has no work on this workload. Null means the counter is
    not armed (GREENBOOST_KV_PREFETCH unset), which is the default , and the
    two must not be confused, because one closes the investigation and the
    other has not started it.
    """
    fields, reason = _shim_stats()
    if fields is None:
        return {"value": None, "unit": "count", "raw_source": reason}
    raw = fields
    mode = str(raw.get("kv_prefetch_mode", "0"))
    if mode in ("", "0"):
        return {"value": None, "unit": "count",
                "raw_source": "GREENBOOST_KV_PREFETCH is unset, so the shim's "
                              "opportunity counter never ticks , serve with "
                              "GREENBOOST_KV_PREFETCH=stats (measures only, "
                              "moves no bytes) to populate it"}
    try:
        return {"value": int(raw.get("kv_prefetch_opportunities") or 0),
                "unit": "count",
                "raw_source": f"shim_stats.kv_prefetch_opportunities "
                              f"(mode={mode}, ticks="
                              f"{raw.get('kv_prefetch_ticks')})"}
    except (TypeError, ValueError):
        return {"value": None, "unit": "count",
                "raw_source": "shim_stats.kv_prefetch_opportunities unparseable"}


def _res_draft_accept_pct(entity_id, window_s):
    """Share of speculatively-drafted tokens the target model accepted.

    The engine reports `draft_n` / `draft_n_accepted` on every response that
    drafted anything, so this is measured, not modelled. It is the state the
    2026-08-05 depth sweep was missing: that sweep found the tok/s curve is
    non-monotonic in depth (2:5.15, 3:5.58, 4:6.50, 6:4.40, 8:5.76), which is
    what it looks like when the right depth depends on something nobody is
    reading.

    Depth 0 is excluded rather than counted: with nothing drafted, nothing can
    be rejected, and the engine duly reports 100% , which reads as "drafting is
    working perfectly" when the truth is "drafting is off".
    """
    ev = [e for e in _agent_events("tok_s_measured", window_s)
          if (e.get("draft_n") or 0) > 0]
    if not ev:
        return {"value": None, "unit": "percent",
                "raw_source": "no tok_s_measured event carried draft_n , "
                              "either nothing was drafted (depth 0 / no MTP "
                              "head) or the proxy has not seen a response yet"}
    drafted = sum(int(e.get("draft_n") or 0) for e in ev)
    accepted = sum(int(e.get("draft_n_accepted") or 0) for e in ev)
    if not drafted:
        return {"value": None, "unit": "percent",
                "raw_source": "draft_n present but zero across the window"}
    return {"value": round(100.0 * accepted / drafted, 1), "unit": "percent",
            "raw_source": f"tok_s_measured.draft_n_accepted/draft_n over "
                          f"{len(ev)} response(s)"}


def _res_tuner_harvesting(entity_id, window_s):
    """Whether the control loop is currently holding a lever back.

    True means a harvest is in effect: something (today, the GPU power limit)
    has been given back because decode is bandwidth-bound and holding it buys
    no throughput. It is a state, not a fault , the fault would be sitting at
    full board power waiting on PCIe, which is what this box did before the
    loop existed.
    """
    ev = _agent_events("tuner_decision", window_s)
    if not ev:
        return {"value": None, "unit": "boolean",
                "raw_source": "no tuner_decision events yet , run tuner_tick"}
    last_move = next((e for e in reversed(ev)
                      if e.get("action") in ("harvest", "restore")), None)
    if last_move is None:
        return {"value": False, "unit": "boolean",
                "raw_source": f"tuner_decision over {len(ev)} tick(s), none "
                              f"moved a lever"}
    return {"value": last_move.get("action") == "harvest"
                     and bool(last_move.get("applied")),
            "unit": "boolean",
            "raw_source": f"latest tuner_decision action="
                          f"{last_move.get('action')} applied="
                          f"{last_move.get('applied')}"}


def _res_ctx_estimate_error_pct(entity_id, window_s):
    """How far the CLI's token estimate sits from the server's own count.

    Signed, and the sign is the whole point: a NEGATIVE value means the client
    predicted fewer tokens than the server charged, which is the dangerous
    direction , it reports headroom that is not there and lets a turn walk into
    a 400. Live 2026-08-20: a compaction retry was fired at 24654 tokens
    against a 24576-token window and the operator got the raw error.
    """
    ev = [e for e in _agent_events("agent_context_edit", window_s)
          if e.get("op") == "calibrate" and e.get("estimate_error_pct") is not None]
    if not ev:
        return {"value": None, "unit": "percent",
                "raw_source": "no agent_context_edit op=calibrate events yet "
                              "(the CLI has not completed a turn against a "
                              "server that reported usage.prompt_tokens)"}
    vals = sorted(float(e["estimate_error_pct"]) for e in ev)
    med = vals[len(vals) // 2]
    return {"value": round(med, 2), "unit": "percent",
            "raw_source": f"agent_context_edit op=calibrate, median of "
                          f"{len(vals)} sample(s)"}


def _res_ctx_hard_trims(entity_id, window_s):
    """Times the last-resort context eviction had to run.

    Every one of these is a turn that compaction alone could not save, because
    the bytes that did not fit were in the live tail. Nonzero is not a failure
    , the trim is what kept the turn alive , but a rising count means the
    served window is too small for how this session actually works.
    """
    ev = [e for e in _agent_events("agent_context_edit", window_s)
          if e.get("op") == "hard_trim"]
    if not ev:
        return {"value": 0, "unit": "count",
                "raw_source": "no agent_context_edit op=hard_trim events in window"}
    unmet = sum(1 for e in ev if e.get("met") is False)
    return {"value": len(ev), "unit": "count",
            "raw_source": f"agent_context_edit op=hard_trim, {unmet} of "
                          f"{len(ev)} could not reach the budget"}


def _res_agent_memory_recall_chars(entity_id, window_s):
    """Characters of recalled memory injected on the last recall (AE-9/10)."""
    ev = _agent_events("agent_memory_recall", window_s)
    if not ev:
        return {"value": None, "unit": "characters",
                "raw_source": "no agent_memory_recall events yet"}
    v = ev[-1].get("chars")
    if v is None:
        return {"value": None, "unit": "characters",
                "raw_source": "latest agent_memory_recall carried no 'chars'"}
    return {"value": int(v), "unit": "characters",
            "raw_source": "agent_memory_recall.chars"}


def _res_host_mem_available_gb(entity_id, window_s):
    """MemAvailable on the host, in GB.

    The number that decides whether the OOM killer runs. NOT MemFree: the
    kernel's own estimate of what a new allocation can actually get, which is
    what the T2 pool competes for.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = float(line.split()[1])
                    return {"value": round(kb / (1024 * 1024), 2), "unit": "gigabytes",
                            "raw_source": "/proc/meminfo MemAvailable"}
        return {"value": None, "unit": "gigabytes",
                "raw_source": "MemAvailable absent from /proc/meminfo"}
    except Exception as e:
        return {"value": None, "unit": "gigabytes",
                "raw_source": f"could not read /proc/meminfo: {e}"}


def _res_t2_oom_guard_state(entity_id, window_s):
    """The kernel T2 OOM guard's own verdict, from the pool's sysfs brief.

    greenboost.ko trips this guard when host MemAvailable falls under its
    reserve and starts evicting T2. Until now it announced that ONLY to the
    kernel log: `greenboost: T2 OOM guard TRIPPED - avail=4044MB < reserve=4GB`.
    It fired twice on 2026-08-18, roughly ten minutes before each of two OOM
    kills that took the user's terminal with them, and reached neither dataflux
    nor this layer , a direct Observability Must-Rule violation ("do not let
    shim decisions stay log-only").
    """
    try:
        with open("/sys/class/greenboost/greenboost/pool_brief") as f:
            brief = f.read().strip()
    except Exception as e:
        return {"value": None, "unit": "state",
                "raw_source": f"pool_brief unreadable (kmod loaded?): {e}"}
    for tok in brief.split():
        if tok.startswith("PRESSURE:"):
            return {"value": tok.split(":", 1)[1], "unit": "state",
                    "raw_source": "sysfs pool_brief PRESSURE",
                    "evidence": {"pool_brief": brief}}
    return {"value": None, "unit": "state",
            "raw_source": f"no PRESSURE field in pool_brief: {brief!r}"}


def _res_unregistered_cached_models(entity_id, window_s):
    """Models on disk in the HF cache that the manifest does not list.

    Counts GGUF files under the model store's `_hf_cache` whose path no
    manifest entry claims. A Full Install purge resets the manifest while
    leaving the cache alone, so this goes from 0 to "everything you ever
    downloaded" in one install, and the only symptom the operator sees is
    `no such model` when the CLI starts.
    """
    try:
        import glob as _glob
        import os as _os
        import gb_synapse
        cache = gb_synapse.MODEL_STORE_DIR / "_hf_cache"
        if not cache.exists():
            return {"value": 0, "unit": "count",
                    "raw_source": "no _hf_cache directory (nothing pulled from HF)"}
        known = {e.path for e in gb_synapse.list_models()}
        missing = []
        for path in _glob.glob(str(cache / "models--*" / "snapshots" / "*" / "*.gguf")):
            if path in known or _os.path.basename(path).startswith("mmproj"):
                continue
            real = _os.path.realpath(path)
            if _os.path.exists(real) and _os.path.getsize(real) >= 1_000_000:
                missing.append(_os.path.basename(path))
        return {"value": len(missing), "unit": "count",
                "raw_source": "_hf_cache/*.gguf vs gb_synapse.list_models()",
                "evidence": {"missing": sorted(missing)[:8]}}
    except Exception as e:
        # Never report 0 on failure — that is a clean bill inferred from an
        # error, the exact pattern the GB-Semantics rule forbids.
        return {"value": None, "unit": "count",
                "raw_source": f"could not read model cache: {e}"}


def _res_dataflux_event_count(entity_id, window_s):
    """How many events the flight recorder currently holds.

    Counts the whole retained log , live file plus rotated archive , not a
    window, because the question this answers ("is there any history at all?")
    is about the store, not about recent activity.

    Returns None, never 0, when the log cannot be read. A zero inferred from an
    exception is indistinguishable from a genuinely empty log, and those two
    demand opposite actions.
    """
    try:
        import gb_dataflux
        # since_hours large enough to span the retained archive; read_events
        # already merges the .1.gz archive ahead of the live file.
        events = gb_dataflux.read_events(since_hours=24 * 3650)
        n = len(events)
        newest = None
        if n:
            try:
                newest = max(float(e.get("ts") or 0) for e in events) or None
            except (TypeError, ValueError):
                newest = None
        return {"value": n, "unit": "count",
                "raw_source": "gb_dataflux.read_events (live log + rotated archive)",
                "evidence": {"newest_ts": newest}}
    except Exception as e:
        return {"value": None, "unit": "count",
                "raw_source": f"could not read the dataflux log: {e}"}


def _res_weights_gb(entity_id, window_s):
    """Weight footprint of the model currently being served.

    Read from the manifest entry (ModelEntry.n_bytes) matched against the live
    run-state's model+quant, so it reflects what is actually loaded rather than
    whatever the manifest happens to list first."""
    try:
        import gb_synapse
        sessions = gb_synapse.ps()
        if entity_id:
            sessions = [s for s in sessions if s.get("model") == entity_id]
        if not sessions:
            return {"value": None, "unit": "gigabytes",
                    "raw_source": "no serve session running"}
        s = sessions[0]
        for e in gb_synapse.list_models():
            if e.name == s.get("model") and (not s.get("quant") or e.quant == s.get("quant")):
                return {"value": round(e.n_bytes / (1024 ** 3), 2), "unit": "gigabytes",
                        "raw_source": "gb_synapse.list_models().n_bytes",
                        "model": e.name, "quant": e.quant}
        return {"value": None, "unit": "gigabytes",
                "raw_source": "served model not found in manifest"}
    except Exception as e:
        return {"value": None, "unit": "gigabytes", "raw_source": "unavailable", "error": str(e)}


def _res_inference_endpoint_is_local(entity_id, window_s):
    """Is every configured inference endpoint on hardware the owner controls?

    The executable half of CLAUDE.md's Local-First Must-Rule. GreenBoost exists
    so a 27B model runs on a 12 GB card instead of being rented by the token; a
    cloud endpoint does not merely leak data, it makes the whole stack pointless
    for that request. A prose rule cannot notice a config drifting, so this
    resolves the actual endpoints in play.

    Local means loopback, this box's own LAN, or a configured feeder. A remote
    `api_base` (integrate.api.nvidia.com, api.openai.com, ...) is not local even
    when it is fast or free.
    """
    remote = []
    checked = []
    try:
        import gb_synapse
        for st in gb_synapse.ps():
            bind = st.get("bind") or "127.0.0.1"
            checked.append(bind)
            if not _is_local_host(bind):
                remote.append(bind)
    except Exception as e:
        return {"value": None, "unit": "boolean", "raw_source": "unavailable",
                "error": str(e)}
    for var in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "FORGE_OLLAMA_URL",
                "GB_SYNAPSE_UPSTREAM", "ANTHROPIC_BASE_URL"):
        val = os.environ.get(var, "").strip()
        if not val:
            continue
        checked.append(f"{var}={val}")
        host = val.split("//")[-1].split("/")[0].split(":")[0]
        if not _is_local_host(host):
            remote.append(f"{var}={val}")
    return {"value": not remote, "unit": "boolean",
            "raw_source": "gb_synapse.ps().bind + inference base-url env vars",
            "remote_endpoints": remote, "checked": checked}


def _is_local_host(host: str) -> bool:
    """Loopback, this machine, or the owner's own LAN , all count as local.

    A feeder on the owner's LAN IS local: the cluster fabric is the whole point
    of the project, not an escape from it.
    """
    h = (host or "").strip().lower()
    if not h or h in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True
    if h.startswith(("10.", "192.168.", "127.")):
        return True
    if h.startswith("172."):
        try:
            return 16 <= int(h.split(".")[1]) <= 31      # 172.16/12
        except (IndexError, ValueError):
            return False
    return h.endswith((".local", ".lan", ".internal"))


def _res_recurrent_replay_ms(entity_id, window_s):
    """Estimated per-turn cost of re-running the recurrent layers over the whole
    prompt, from the most recent prompt_cache sample.

    THE governed number for the inference-speed program. This architecture is
    hybrid , 17 attention layers and 48 Gated DeltaNet recurrent layers , and a
    recurrent layer's state at position t only exists by running every token up
    to t. So the KV cache can hit 99.9% and the recurrent stack still replays
    the entire conversation. Measured 2026-08-18 across 29 warm turns: median
    6,937 ms of prefill for a median of 42 genuinely-new tokens, with cost per
    TOTAL token stable at 0.151 ms and cost per NEW token scattering 146-191.

    DERIVED, not measured: the engine reports one `prompt_ms` and does not break
    it down. New tokens are charged at this box's own cold-prefill rate and the
    remainder attributed to replay. The raw_source says so, because presenting a
    derived figure as a reading is how a plausible story outlives its evidence.
    """
    ev = _latest_event_with_field("prompt_cache", "engine_prompt_ms",
                                  node=entity_id, max_age_s=window_s)
    if ev is None:
        return {"value": None, "unit": "milliseconds",
                "raw_source": "no prompt_cache event carrying engine_prompt_ms"}
    try:
        import gb_bench_turn
        split = gb_bench_turn.split_prefill(
            int(ev.get("prompt_tokens") or 0),
            int(ev.get("reused_tokens") or 0),
            float(ev.get("engine_prompt_ms") or 0.0))
    except Exception as e:
        return {"value": None, "unit": "milliseconds",
                "raw_source": "unavailable", "error": str(e)}
    return {"value": split["recurrent_replay_ms_est"], "unit": "milliseconds",
            "raw_source": ("derived from dataflux.prompt_cache "
                           "(engine_prompt_ms minus new tokens at cold-prefill rate) "
                           ", ESTIMATE, the engine reports no breakdown"),
            "prompt_ms": split["prompt_ms"], "new_tokens": split["new_tokens"],
            "replay_share": split["replay_share"]}


def _res_idle_vram_held_gb(entity_id, window_s):
    """VRAM held by a serve session that is not currently doing any work.

    0 when the GPU is busy , this measures *dead* residency, not residency.
    The number the operator actually sees when they open a system monitor and
    find 87% of the card allocated at 0% utilization (measured 2026-08-18:
    10.4 GiB held, 787 MHz, 23 W). Governed so "should I pause?" has an
    answer that is not a screenshot."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        if not out:
            return {"value": None, "unit": "gigabytes",
                    "raw_source": "nvidia-smi returned no rows"}
        used_mb, util = [int(x.strip()) for x in out[0].split(",")]
        import gb_synapse
        sessions = gb_synapse.ps()
        if not sessions:
            return {"value": 0.0, "unit": "gigabytes",
                    "raw_source": "no serve session running",
                    "gpu_util_pct": util}
        held = round(used_mb / 1024.0, 2) if util < _GPU_BUSY_UTIL_PCT else 0.0
        return {"value": held, "unit": "gigabytes",
                "raw_source": "nvidia-smi memory.used gated on utilization.gpu",
                "gpu_util_pct": util, "vram_used_gb": round(used_mb / 1024.0, 2),
                "model": sessions[0].get("model")}
    except Exception as e:
        return {"value": None, "unit": "gigabytes", "raw_source": "unavailable",
                "error": str(e)}


def _res_pcie_link_gen_current(entity_id, window_s):
    """Negotiated PCIe generation, sampled NOW rather than at telemetry init.

    Deliberately a live read: GpuTopology probes the link once at NVML init and
    a GPU downtrains its link to Gen1/Gen2 whenever it is idle, so the cached
    value answers "what was the link doing at startup", which is almost never
    the question being asked."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=pcie.link.gen.current,pcie.link.gen.max,"
             "pcie.link.width.current,pcie.link.width.max,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        if not out:
            return {"value": None, "unit": "pcie_generation",
                    "raw_source": "nvidia-smi returned no rows"}
        gen_cur, gen_max, w_cur, w_max, util = [int(x.strip()) for x in out[0].split(",")]
        return {"value": gen_cur, "unit": "pcie_generation",
                "raw_source": "nvidia-smi pcie.link.gen.current (live)",
                "gen_max": gen_max, "width_current": w_cur, "width_max": w_max,
                "gpu_util_pct": util}
    except Exception as e:
        return {"value": None, "unit": "pcie_generation",
                "raw_source": "unavailable", "error": str(e)}


def _res_cluster_vram_fill_pct(entity_id, window_s):
    """Fixed 2026-07-30: this previously read snap.get("nodes", {}), a key
    gb_cluster.cluster_snapshot() has never produced — it returns
    {"host": {...}, "feeders": [...]}, each carrying t1_free_mb/t1_total_mb
    (see _host_metrics_dict/Feeder). The old code silently always resolved
    an empty dict regardless of real cluster state."""
    try:
        import gb_cluster
        snap = gb_cluster.cluster_snapshot()
    except Exception as e:
        return {"value": None, "unit": "percent", "raw_source": "unavailable", "error": str(e)}
    if not isinstance(snap, dict):
        return {"value": {}, "unit": "percent", "raw_source": "gb_cluster.cluster_snapshot"}

    def _fill_pct(total, free) -> "float | None":
        if not total:
            return None
        return round(100.0 * (total - (free or 0)) / total, 1)

    out: dict = {}
    host = snap.get("host") or {}
    v = _fill_pct(host.get("t1_total_mb"), host.get("t1_free_mb"))
    if v is not None:
        out["host"] = v
    for f in snap.get("feeders") or []:
        if not f.get("online"):
            continue
        name = f.get("hostname") or f.get("ip") or "?"
        v = _fill_pct(f.get("t1_total_mb"), f.get("t1_free_mb"))
        if v is not None:
            out[name] = v
    if entity_id:
        return {"value": out.get(entity_id), "unit": "percent", "raw_source": "gb_cluster.cluster_snapshot"}
    return {"value": out, "unit": "percent", "raw_source": "gb_cluster.cluster_snapshot"}


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


def _res_feeder_reachable(entity_id, window_s):
    try:
        import gb_cluster
        snap = gb_cluster.cluster_snapshot()
    except Exception as e:
        return {"value": None, "unit": "boolean", "raw_source": "unavailable", "error": str(e)}
    out: dict = {}
    for f in (snap or {}).get("feeders") or []:
        name = f.get("hostname") or f.get("ip") or "?"
        out[name] = bool(f.get("online"))
    if entity_id:
        return {"value": out.get(entity_id), "unit": "boolean",
                "raw_source": "gb_cluster.cluster_snapshot.feeders[].online"}
    return {"value": out, "unit": "boolean",
            "raw_source": "gb_cluster.cluster_snapshot.feeders[].online"}


def _res_gaming_session_active(entity_id, window_s):
    ev = _latest_event("gaming_session", max_age_s=window_s or 6 * 3600.0)
    if ev is None:
        return {"value": False, "unit": "boolean", "raw_source": "no gaming_session events yet"}
    return {"value": ev.get("action") == "start", "unit": "boolean",
            "raw_source": "dataflux.gaming_session (latest event)"}


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


def _res_kmod_loaded_version(entity_id, window_s):
    """Version the RUNNING module declares. None when it isn't loaded."""
    vf = Path("/sys/module/greenboost/version")
    try:
        if not vf.exists():
            return {"value": None, "unit": "version",
                    "raw_source": "/sys/module/greenboost/version absent "
                                  "(module not loaded)"}
        v = vf.read_text().strip()
        if not v:
            return {"value": None, "unit": "version",
                    "raw_source": "/sys/module/greenboost/version empty"}
        return {"value": v, "unit": "version",
                "raw_source": "/sys/module/greenboost/version"}
    except Exception as e:
        return {"value": None, "unit": "version",
                "raw_source": "/sys/module/greenboost/version unreadable",
                "error": str(e)}


def _journal_this_boot(grep: str, limit: int = 4000) -> "list[tuple[float, str]]":
    """(unix_ts, message) pairs from the CURRENT boot matching `grep`.

    Read-only, no root: journalctl is readable by members of systemd-journal
    and by the owner for their own boot. Returns [] on any failure , a caller
    that cannot read the journal must report "cannot tell", never "fine".
    """
    import subprocess
    try:
        out = subprocess.run(
            ["journalctl", "-b", "-o", "short-unix", "--no-pager", "-g", grep],
            capture_output=True, text=True, timeout=15)
    except Exception:
        return []
    if out.returncode != 0 or not out.stdout:
        return []
    rows = []
    for line in out.stdout.splitlines()[:limit]:
        head, _, rest = line.partition(" ")
        try:
            rows.append((float(head), rest))
        except ValueError:
            continue
    return rows


def _greenboost_load_blocks() -> "list[tuple[float, list[tuple[float, str]]]]":
    """This boot's greenboost module loads, newest last.

    Each module init prints exactly one `GreenBoost vX.Y - CUDA Memory ...`
    banner, so that line delimits one load. Returns
    [(banner_ts, [(ts, msg), ...]), ...] where the list is every greenboost
    kernel line belonging to that load.

    Why this exists: a boot can contain SEVERAL loads. This box booted v3.2
    out of the initramfs at 01:29 and was reloaded to v3.4 at 02:56. A
    resolver that reads the FIRST greenboost line of the boot answers a
    question about a module that is no longer running , which is a stale
    verdict presented as a current one, the exact failure this layer exists to
    prevent. Anything asking "what is true NOW" must read the LAST block.
    """
    rows = _journal_this_boot("greenboost: ")
    blocks: list = []
    for ts, msg in rows:
        if "GreenBoost v" in msg and "CUDA Memory" in msg:
            blocks.append((ts, []))
        if blocks:
            blocks[-1][1].append((ts, msg))
    return blocks


def _res_kmod_load_stage(entity_id, window_s):
    """WHERE the running greenboost module was loaded from: the initramfs, or
    the real root.

    This is the field that distinguishes a drift you can fix from one that
    comes back every boot. `initramfs.conf`'s MODULES=most makes
    initramfs-tools copy greenboost.ko into the boot image, and the initrd's
    own systemd-modules-load inserts it before switch-root. A reinstall
    rebuilds /lib/modules and never touches that image, so the machine keeps
    loading the frozen copy , `sudo greenboost load` fixes it until the next
    reboot and no further (incident 2026-08-21: v3.2 across several reboots
    while v3.4 was installed).

    Anything inserted BEFORE `initrd-switch-root.target` came from the
    initramfs. That marker is used rather than local-fs.target because there
    are two local-fs.targets per boot , one per systemd instance , and the
    initrd's own fires first, which would read as "real root" and invert the
    verdict.

    Returns None when the journal cannot be read or the boot has no switch-root
    marker (a system booted without an initrd at all). Unknown is not "fine".
    """
    if not Path("/sys/module/greenboost").exists():
        return {"value": None, "unit": "stage",
                "raw_source": "module not loaded , nothing to attribute"}
    blocks = _greenboost_load_blocks()
    if not blocks:
        return {"value": None, "unit": "stage",
                "raw_source": "journalctl returned no greenboost init banner for this boot"}
    sw = _journal_this_boot("Reached target initrd-switch-root.target|Switching root")
    if not sw:
        return {"value": None, "unit": "stage",
                "raw_source": "no switch-root marker this boot , cannot place the load "
                              "relative to the initrd (system may have booted without one)"}
    # LAST block: the module running right now, not the one the boot started with.
    load_ts, switch_ts = blocks[-1][0], sw[0][0]
    stage = "initramfs" if load_ts < switch_ts else "realroot"
    out = {"value": stage, "unit": "stage",
           "raw_source": f"journalctl -b: current load@{load_ts:.3f} vs switch-root@{switch_ts:.3f}",
           "loads_this_boot": len(blocks)}
    if len(blocks) > 1:
        out["note"] = (f"{len(blocks)} loads this boot; this is the newest. The "
                       f"first was at {blocks[0][0]:.3f} "
                       f"({'initramfs' if blocks[0][0] < switch_ts else 'realroot'}).")
    return out


# ── Gaming Suite: the launch path ──────────────────────────────────────────
#
# Added 2026-08-21 after a launch failure that the governed layer could say
# nothing at all about. Coverage was three metrics, none of them about
# launching, so "why will this game not start" was answerable only by reading
# the Steam console log by hand , which is what this layer exists to replace.
#
# Everything below is derived from files on disk (the wrapper's own logs,
# Steam's console log, compatibilitytools.d, the shader cache). No new
# plumbing, and nothing that requires the Suite to be running.

def _steam_root() -> "Path | None":
    from pathlib import Path as _P
    import os
    home = _P(os.path.expanduser("~"))
    for r in (home / ".local/share/Steam", home / ".steam/steam", home / ".steam/root",
              home / ".var/app/com.valvesoftware.Steam/data/Steam"):
        if r.is_dir():
            return r
    return None


def _steam_console_tail(max_lines: int = 6000) -> "list[str]":
    r = _steam_root()
    if r is None:
        return []
    for cand in (r / "logs/console-linux.txt", _P_home_logs()):
        try:
            if cand and cand.is_file():
                return cand.read_text(errors="replace").splitlines()[-max_lines:]
        except OSError:
            continue
    return []


def _P_home_logs():
    from pathlib import Path as _P
    import os
    p = _P(os.path.expanduser("~/.steam/steam/logs/console-linux.txt"))
    return p if p.is_file() else None


_GB_TOOL_RE = _re.compile(r"compatibilitytools\.d/(greenboost-proton[a-z-]*)")
_TS_RE = _re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")


def _launch_runs(window_s: float = 900.0) -> "list[tuple[float, str]]":
    """(epoch, tool_name) for each GreenBoost Proton wrapper start recently.

    One "Delegating to:" line per wrapper invocation, and the pre-flight line
    just above it names which compatibilitytools.d entry is running. Both come
    from Steam's console log, which is the only place with timestamps.
    """
    import time, datetime as _dt
    cutoff = time.time() - window_s
    # The tool path appears in the pre-flight line, which the wrapper writes
    # AFTER "Delegating to:", not before , so a run is opened on the
    # Delegating line and named by the next tool path that follows it.
    runs: list = []
    for ln in _steam_console_tail():
        if "[greenboost-proton] Delegating to:" in ln:
            mt = _TS_RE.match(ln)
            if not mt:
                continue
            try:
                ts = _dt.datetime.strptime(mt.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            except ValueError:
                continue
            runs.append([ts, None])
            continue
        m = _GB_TOOL_RE.search(ln)
        if m and runs and runs[-1][1] is None:
            runs[-1][1] = m.group(1)
    return [(ts, tool or "unknown") for ts, tool in runs if ts >= cutoff]


def _res_proton_tools_installed(entity_id, window_s):
    """How many GreenBoost Proton compat tools Steam can see.

    More than one is legal and useful (stable + experimental), and is also the
    precondition for the 2026-08-21 race: two of them started the SAME appid
    two seconds apart, into one wine prefix.
    """
    r = _steam_root()
    if r is None:
        return {"value": None, "unit": "count", "raw_source": "no Steam root found"}
    d = r / "compatibilitytools.d"
    if not d.is_dir():
        return {"value": 0, "unit": "count", "details": {"tools": []},
                "raw_source": f"{d} absent"}
    tools = sorted(e.name for e in d.iterdir()
                   if e.is_dir() and e.name.startswith("greenboost-proton"))
    return {"value": len(tools), "unit": "count", "details": {"tools": tools},
            "raw_source": str(d)}


def _res_game_launch_attempts(entity_id, window_s):
    """GreenBoost Proton wrapper starts in the last 15 minutes.

    One press of Launch should produce one. Repeated starts at a regular
    interval mean something is killing the game and Steam is retrying , the
    2026-08-21 shape was four starts about 62 s apart.
    """
    runs = _launch_runs(window_s or 900.0)
    return {"value": len(runs), "unit": "count",
            "details": {"tools": sorted({t for _, t in runs}),
                        "first_ts": runs[0][0] if runs else None,
                        "last_ts": runs[-1][0] if runs else None},
            "raw_source": "Steam console log , '[greenboost-proton] Delegating to:' lines"}


def _res_game_process_running(entity_id, window_s):
    """Is a wine/Proton game process actually alive right now.

    The question every launch failure turns on, and the one Steam's own
    "Launching" spinner does not answer: Steam reports that it accepted the
    request, not that anything started.
    """
    from pathlib import Path as _P
    try:
        names = []
        for pd in _P("/proc").iterdir():
            if not pd.name.isdigit():
                continue
            try:
                comm = (pd / "comm").read_text().strip()
            except OSError:
                continue
            if not (comm in ("wine64-preloader", "wine-preloader", "wineserver")
                    or comm.endswith(".exe")):
                continue
            # comm.endswith(".exe") is NOT sufficient, and the difference is
            # not academic: it matched an unrelated host binary named
            # `claude.exe` on 2026-08-21 and reported it as a running game.
            # Only processes wine actually started carry WINEPREFIX /
            # STEAM_COMPAT_DATA_PATH, so the environment is the real test.
            try:
                env = (pd / "environ").read_bytes()
            except OSError:
                continue
            if b"WINEPREFIX=" not in env and b"STEAM_COMPAT_DATA_PATH=" not in env:
                continue
            names.append(comm)
    except OSError as e:
        return {"value": None, "unit": "boolean", "raw_source": "/proc unreadable",
                "error": str(e)}
    # Proton's own furniture. Every one of these starts for ANY launch,
    # successful or not , `steam.exe` is Proton's Steamworks shim and
    # `wineserver` is the prefix's own daemon, so both are present for a
    # launch that produced no game at all. Counting them as "the game is
    # running" is precisely the misread this metric exists to prevent
    # (observed 2026-08-21: a 13-minute session whose entire process list was
    # this set, with the GPU at 0%).
    _WINE_INFRA = {
        "explorer.exe", "services.exe", "winedevice.exe", "plugplay.exe",
        "rpcss.exe", "svchost.exe", "conhost.exe", "tabtip.exe",
        "wineboot.exe", "start.exe", "steam.exe", "wineserver",
        "wine64-preloader", "wine-preloader", "winemenubuilder.exe",
        "wineconsole.exe", "cmd.exe", "iexplore.exe",
    }
    real = [n for n in names if n not in _WINE_INFRA]
    return {"value": bool(real), "unit": "boolean",
            "details": {"processes": sorted(set(real))[:8],
                        "wine_infrastructure_only": bool(names) and not real},
            "raw_source": "/proc/*/comm , wine processes, Proton's own helpers excluded"}


def _res_last_game_session_s(entity_id, window_s):
    """Duration of the most recent COMPLETED game session, in seconds.

    Read from the Proton wrapper's own `gaming_session` events, which are the
    only record that survives the game exiting. The wrapper writes them
    directly to ~/.local/share/greenboost/dataflux.jsonl when gb_dataflux is
    not importable from inside the Steam sandbox, which is the normal case.

    Exists because launch-attempt counting alone cries wolf. On 2026-08-21 six
    wrapper starts inside 29 s looked exactly like a failed retry loop, and
    five of them WERE aborted , but the sixth ran the game for 335 s and
    exited cleanly. A session that completed is the evidence that outranks the
    noise before it.
    """
    import json, os
    f = os.path.expanduser("~/.local/share/greenboost/dataflux.jsonl")
    best = None
    try:
        with open(f, errors="replace") as fh:
            try:
                fh.seek(max(0, os.path.getsize(f) - 2_000_000))
            except OSError:
                pass
            for line in fh:
                line = line.strip()
                if not line or '"gaming_session"' not in line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("kind") != "gaming_session" or d.get("action") != "stop":
                    continue
                if entity_id and str(d.get("appid")) != str(entity_id):
                    continue
                ts = d.get("ts") or 0
                if best is None or ts > best.get("ts", 0):
                    best = d
    except OSError as e:
        return {"value": None, "unit": "seconds", "raw_source": "wrapper dataflux unreadable",
                "error": str(e)}
    if best is None:
        return {"value": None, "unit": "seconds",
                "raw_source": "no gaming_session stop event recorded"}
    return {"value": round(float(best.get("elapsed_s") or 0.0), 1), "unit": "seconds",
            "details": {"appid": best.get("appid"),
                        "peak_vram_mb": best.get("peak_vram_mb"),
                        "avg_vram_mb": best.get("avg_vram_mb"),
                        "peak_t2_mb": best.get("peak_t2_mb"),
                        "vram_samples": best.get("vram_samples"),
                        "rc": best.get("rc"),
                        "ended_ts": best.get("ts")},
            "raw_source": "wrapper gaming_session events (~/.local/share/greenboost/dataflux.jsonl)"}


def _res_shader_cache_mb(entity_id, window_s):
    """Fossilize shader cache size for an appid (entity_id), in MB.

    Near-zero on a title that has been played means the pre-cache never ran,
    so the first session pays every pipeline compile in-frame. Fossilize
    reports no progress of its own , this is the only observable it leaves.
    """
    r = _steam_root()
    if r is None or not entity_id:
        return {"value": None, "unit": "megabytes",
                "raw_source": "need an appid as entity_id, and a Steam root"}
    d = r / "steamapps/shadercache" / str(entity_id)
    if not d.is_dir():
        return {"value": 0.0, "unit": "megabytes",
                "raw_source": f"{d} absent , nothing pre-cached for this appid"}
    total = 0
    for f in d.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return {"value": round(total / (1024 * 1024), 2), "unit": "megabytes",
            "raw_source": str(d)}


def _res_apparmor_profiles_contaminated(entity_id, window_s):
    """snapd-generated profiles carrying GreenBoost rules AFTER their final
    top-level closing brace.

    Counting is the whole point, and so is the direction of the reading. A rule
    past the closing brace is at top level, where apparmor_parser expects
    `profile NAME {`; it grants nothing AND makes the entire profile
    unparseable, which takes `snapd.apparmor.service` down and every snap with
    it (2026-07-27: Firefox, Chromium, snap-store, cups; reintroduced by
    gb_supervisor.py on 2026-08-20).

    Returns None when the directory cannot be read , an unknown state is never
    a healthy state, and this is a security-policy question.
    """
    from pathlib import Path as _P
    d = _P("/var/lib/snapd/apparmor/profiles")
    if not d.is_dir():
        return {"value": None, "unit": "count",
                "raw_source": f"{d} absent , snapd may not be installed"}
    bad = []
    try:
        for f in sorted(d.glob("snap-confine.snapd.*")):
            try:
                lines = f.read_text().splitlines()
            except OSError:
                return {"value": None, "unit": "count",
                        "raw_source": f"{f} unreadable (needs root?) , cannot tell"}
            last_close = max((i for i, ln in enumerate(lines) if ln.startswith("}")),
                             default=None)
            if last_close is None:
                continue
            if any("libgreenboost_audit" in ln for ln in lines[last_close + 1:]):
                bad.append(f.name)
    except OSError as e:
        return {"value": None, "unit": "count", "raw_source": f"{d} unreadable",
                "error": str(e)}
    return {"value": len(bad), "unit": "count", "profiles": bad,
            "raw_source": f"{d}/snap-confine.snapd.* , rules after the final top-level '}}'"}


def _res_system_degraded(entity_id, window_s):
    """systemd's own verdict, with the failed units as evidence."""
    import subprocess as _sp
    try:
        r = _sp.run(["systemctl", "is-system-running"], capture_output=True,
                    text=True, timeout=10)
        state = (r.stdout or "").strip()
    except Exception as e:
        return {"value": None, "unit": "boolean",
                "raw_source": "systemctl is-system-running unavailable", "error": str(e)}
    if not state:
        return {"value": None, "unit": "boolean",
                "raw_source": "systemctl is-system-running returned nothing"}
    failed = []
    try:
        r2 = _sp.run(["systemctl", "--failed", "--no-legend", "--no-pager", "--plain"],
                     capture_output=True, text=True, timeout=10)
        failed = [ln.split()[0] for ln in (r2.stdout or "").splitlines() if ln.strip()]
    except Exception:
        pass
    return {"value": state != "running", "unit": "boolean", "state": state,
            "failed_units": failed,
            "raw_source": f"systemctl is-system-running -> {state}"}


def _res_t3_capacity_configured_mb(entity_id, window_s):
    """T3's CONFIGURED cap , named for what it is, so it can never be read as
    available capacity.

    `GbSnapshot.t3_total_mb` is the module's `nvme_swap_total_mb`, which
    reports the cap set by the `t3_max_gb` module parameter whether or not the
    backing file ever opened. On 2026-08-21 this box reported 74752 MB with T3
    disabled. Pair every read of this with `t3_enabled`.
    """
    try:
        import gb_monitor
        snap = gb_monitor.snapshot(probe_gpu=False)
        if snap.loaded:
            return {"value": snap.t3_total_mb, "unit": "megabytes",
                    "raw_source": "gb_monitor.GbSnapshot.t3_total_mb (CONFIGURED cap)"}
    except Exception as e:
        return {"value": None, "unit": "megabytes", "raw_source": "unavailable", "error": str(e)}
    return {"value": None, "unit": "megabytes", "raw_source": "kernel module not loaded"}


def _res_t3_enabled(entity_id, window_s):
    """Whether Tier 3 actually OPENED its backing file , not whether one is
    configured.

    The two are routinely different and only one of them is visible in the
    obvious place. When the module loads before /var is mounted it logs
    `T3 backing file unavailable (...): -2 - T3 disabled` and runs with T3
    off, while `/sys/class/greenboost/greenboost/status` still prints
    `Tier 3  T3 backing file : 73 GB NVMe file [GreenBoost-managed,
    pre-allocated]` and `GbSnapshot.t3_total_mb` still reports 74752. Both are
    reporting the CONFIGURED cap. Reading either as capacity is how T3 stayed
    silently off for days (incident 2026-08-21).

    The kernel log is the only place the open is actually reported, so that is
    what this reads. None when the journal is unreadable.
    """
    if not Path("/sys/module/greenboost").exists():
        return {"value": None, "unit": "boolean",
                "raw_source": "module not loaded , T3 cannot be open"}
    blocks = _greenboost_load_blocks()
    if not blocks:
        return {"value": None, "unit": "boolean",
                "raw_source": "journalctl returned no greenboost init banner for this boot"}
    # Only the CURRENT load's lines. A "T3 disabled" from an earlier load that
    # has since been replaced is history, not state , reporting it as state is
    # how a fixed problem keeps being reported as broken.
    current = blocks[-1][1]
    t3_lines = [m for _, m in current if "T3" in m]
    for msg in t3_lines:
        if "T3 disabled" in msg or "T3 backing file unavailable" in msg:
            return {"value": False, "unit": "boolean",
                    "raw_source": f"kernel log (current load): {msg.split('greenboost: ')[-1][:120]}"}
    if not t3_lines:
        return {"value": None, "unit": "boolean",
                "raw_source": "current load logged no T3 line at all , cannot tell"}
    return {"value": True, "unit": "boolean",
            "raw_source": f"kernel log (current load): {t3_lines[0].split('greenboost: ')[-1][:120]}"}


def _res_core_build_version(entity_id, window_s):
    """Installed release, from the stamp the installer writes."""
    bi = Path("/etc/greenboost/build_info")
    try:
        if not bi.exists():
            return {"value": None, "unit": "version",
                    "raw_source": "/etc/greenboost/build_info absent "
                                  "(no install stamp on this node)"}
        for line in bi.read_text().splitlines():
            key, _, val = line.partition("=")
            if key.strip() == "BUILD_VERSION" and val.strip():
                return {"value": val.strip(), "unit": "version",
                        "raw_source": "/etc/greenboost/build_info BUILD_VERSION"}
        return {"value": None, "unit": "version",
                "raw_source": "/etc/greenboost/build_info has no BUILD_VERSION line"}
    except Exception as e:
        return {"value": None, "unit": "version",
                "raw_source": "/etc/greenboost/build_info unreadable",
                "error": str(e)}


def _shim_stats(window_s: "float | None" = None) -> "tuple[dict | None, str]":
    """Parse /run/greenboost/shim_stats, gated on the WRITER still being alive.

    Returns (fields, reason). `fields` is None when the file cannot be trusted,
    and `reason` says why in terms a caller can put in a raw_source.

    The gate used to be file mtime alone (stale after 30 s). That is wrong, and
    the mistake is easy to make: the shim writes this file from its CUDA hooks,
    so it stops being rewritten the moment inference goes idle — while every
    number in it stays exactly true, because nothing has moved. Weights sitting
    in T2 do not come back to VRAM just because no one asked a question.

    Measured on this box 2026-08-18, and the reason this helper exists: the file
    was 266 s old, so `t2_overflow_active_mb` resolved to None — while the PID
    that wrote it was alive, still in phase=INFERENCE, still holding
    t2_overflow_total_mb=11468 (11.2 GB crossing PCIe on every forward pass).
    The governed layer went blind at precisely the moment an operator looks at a
    system monitor and asks why the card is full.

    A dead writer is a genuinely different case and still rejected: those
    numbers describe a process that no longer exists.
    """
    p = Path("/run/greenboost/shim_stats")
    if not p.exists():
        return None, "/run/greenboost/shim_stats missing"
    try:
        fields = {}
        for line in p.read_text().splitlines():
            k, _, v = line.partition("=")
            if k:
                fields[k.strip()] = v.strip()
    except Exception as e:
        return None, f"/run/greenboost/shim_stats unreadable: {e}"

    age = time.time() - p.stat().st_mtime
    fields["_age_s"] = round(age, 1)

    pid = fields.get("pid")
    if pid and pid.isdigit():
        try:
            os.kill(int(pid), 0)
            fields["_writer_alive"] = "1"
            return fields, ("shim_stats (writer alive; file idle for "
                            f"{age:.0f}s, values unchanged)"
                            if age > (window_s or 30.0) else "shim_stats (live)")
        except (ProcessLookupError, PermissionError, OSError) as e:
            # PermissionError means the process EXISTS but is owned by another
            # user — alive for our purposes.
            if isinstance(e, PermissionError):
                fields["_writer_alive"] = "1"
                return fields, "shim_stats (writer alive, other uid)"
            fields["_writer_alive"] = "0"
            return None, f"shim_stats writer pid {pid} is gone (file {age:.0f}s old)"

    # No pid to check: fall back to the old mtime gate rather than trusting it.
    if age > (window_s or 30.0):
        return None, f"/run/greenboost/shim_stats stale ({age:.0f}s, no pid to verify)"
    return fields, "shim_stats (live, no pid field)"


# Measured bulk-DMA throughput of this box's PCIe link (host->VRAM, pinned),
# 24.43 GB/s on 2026-08-18 , about 94% of the Gen4 x16 usable ceiling. Used as a
# PHYSICAL LIMIT for sanity-checking, never as a target: no more than this many
# bytes can cross the bus per second, so any telemetry implying more is wrong.
_PCIE_BULK_DMA_GBS = float(os.environ.get("GB_PCIE_BULK_DMA_GBS", "24.43"))


def _overflow_resident_vs_streamed(overflow_mb: float) -> "tuple[bool | None, dict]":
    """Is all of this T2 allocation being read every token, or only part of it?

    NOT a correctness check on the counter , a check on how it is READ.
    `t2_overflow_active_mb` counts bytes ALLOCATED to T2. T2 also holds the KV
    cache and compute/scratch buffers, and those are not read in full on every
    forward pass. So the intuitive reading , "this many bytes cross PCIe per
    token" , is only true for the weights portion.

    The distinction has teeth. Measured 2026-08-18: 5,923 MB reported alongside
    an end-to-end 8.37 tok/s. Treating all of it as per-token traffic implies
    48.4 GB/s across a link measured at 24.43, which is impossible , so at most
    ~2,900 MB of that allocation can be weights actually streamed each token.
    The remainder is resident-but-not-hot.

    That inference was initially made in the wrong direction (concluding the
    counter over-reported by 2x). It does not: adds and subtracts in
    greenboost_cuda_shim.c both use the same size, and `alloc_bytesize ==
    bytesize` on the allocation path. The number is a faithful allocation
    figure; only the per-token interpretation was wrong.

    Returns (all_streamed, evidence). None means no decode rate to compare
    against, which is NOT the same as "all of it streams".
    """
    tok_s = None
    try:
        ev = _latest_event_with_field("tok_s_measured", "tok_s", max_age_s=3600)
        if ev:
            tok_s = float(ev.get("tok_s") or 0) or None
    except Exception:
        tok_s = None
    if not tok_s or not overflow_mb:
        return None, {}
    implied = (overflow_mb / 1024.0) * tok_s
    return implied <= _PCIE_BULK_DMA_GBS, {
        "implied_gbs_if_all_streamed": round(implied, 1),
        "link_gbs": _PCIE_BULK_DMA_GBS,
        "decode_tok_s": round(tok_s, 2),
        "max_streamed_mb": round(_PCIE_BULK_DMA_GBS / tok_s * 1024),
    }


def _res_t2_overflow_active_mb(entity_id, window_s):
    """Bytes the shim has actually routed to T2 DDR — "is overflow happening",
    read straight from the shim's own counter.

    This is NOT `t2_pressure_fraction`. Pressure is how full the T2 POOL is
    relative to its capacity; overflow is whether weights are being served from
    T2 at all. On a large pool the two diverge completely: measured 2026-08-17
    while serving a 15.85 GiB model on 11.26 GB of VRAM, the shim reported
    `t2_overflow_total_mb=14326` against a 43 GB pool whose pressure read 0.0.
    Because `rule1_underfilled` tested pressure, the Rule #1 tripwire could not
    fire in exactly the situation it exists to catch.

    Read from shim_stats directly rather than gb_monitor.GbSnapshot, which has
    no field for it (t2_pool/allocated/available/pressure only) — and note the
    kmod's own `t2_allocated_mb` read 0 in that same sample, so the shim counter
    is the trustworthy signal here.
    """
    try:
        fields, reason = _shim_stats(window_s)
        if fields is None:
            return {"value": None, "unit": "megabytes", "raw_source": reason}
        v = fields.get("t2_overflow_total_mb")
        if v is None:
            return {"value": None, "unit": "megabytes",
                    "raw_source": "shim_stats has no t2_overflow_total_mb"}
        val = float(v)
        all_streamed, ev = _overflow_resident_vs_streamed(val)
        out = {"value": val, "unit": "megabytes",
               "raw_source": f"shim_stats.t2_overflow_total_mb , {reason}",
               "age_s": fields.get("_age_s"),
               "shim_phase": fields.get("phase")}
        if all_streamed is False:
            # The allocation figure is correct; it simply is not all hot. Say
            # which part can be per-token traffic, so a reader does not multiply
            # the whole number by tok/s and conclude the bus is doing the
            # impossible.
            out["partly_resident"] = True
            out["streaming"] = ev
            out["raw_source"] += (
                f" , NOTE: only ~{ev['max_streamed_mb']} MB of this can be "
                f"streamed per token at {ev['decode_tok_s']} tok/s over a "
                f"{ev['link_gbs']} GB/s link; the rest is resident, not hot")
        elif all_streamed is True:
            out["streaming"] = ev
        return out
    except Exception as e:
        return {"value": None, "unit": "megabytes", "raw_source": "unavailable",
                "error": str(e)}


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
    # Optional evidence a resolver wants to carry with the number.
    #
    # resolve() returns a fixed schema on purpose, and everything not in it was
    # silently dropped , which meant a resolver could compute exactly the
    # detail a segment needed (which Proton tools fired, which processes are
    # alive) and have it discarded on the way out, leaving the segment to
    # report "cannot tell". Anything under `details` is passthrough: it is the
    # resolver's own structured evidence, never a second place to put `value`.
    if raw.get("details") is not None:
        out["details"] = raw["details"]
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

# A log this small cannot support any windowed metric , below it, "no data"
# is the honest answer for every event-reading resolver. Not a tuning knob:
# it only has to be small enough that a running recorder clears it within
# seconds (the SnapshotRecorder alone emits every 5s).
_DATAFLUX_RESET_MAX_EVENTS = 25


def _seg_decode_tail_heavy():
    """Decode is dominated by a slow tail rather than a slow average.

    Matters because the two have different fixes. A uniformly slow decode is a
    bandwidth or placement problem; a heavy tail with a fast median is a
    stall , a draft batch boundary, a slot eviction, or something else on the
    box competing. Reading the mean alone cannot tell them apart, which is why
    this segment exists.
    """
    p95 = resolve("inter_token_p95_ms")
    ratio = resolve("slow_token_ratio")
    if p95.get("value") is None or ratio.get("value") is None:
        return None, [{"why": p95.get("raw_source") or ratio.get("raw_source")}]
    # A tail is "heavy" when a meaningful share of tokens sit in it. One
    # outlier in a hundred is a hiccup, not a pattern.
    matched = ratio["value"] >= _TAIL_HEAVY_RATIO
    return matched, [{"inter_token_p95_ms": p95["value"],
                      "slow_token_ratio": ratio["value"],
                      "threshold": _TAIL_HEAVY_RATIO}]


def _wine_game_running() -> "bool | None":
    """Is a Wine/Proton game process alive right now?

    Deliberately a presence test over /proc, not a PID: the Suite's own
    `find_game_pid` returns ONE pid, which is not the tree and must never be
    read as one. None means /proc could not be read at all.
    """
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    for name in entries:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/comm") as fh:
                comm = fh.read().strip()
        except OSError:
            continue
        if comm.startswith("wine") and "preloader" in comm:
            return True
    return False


def _seg_gaming_mode_stuck():
    """gaming_mode left at 1 with no game running.

    Costs real throughput and nothing else reports it: while this is true,
    every inference T2 buffer is parked at the eviction queue's tail, so
    weights get evicted ahead of gaming buffers that no longer exist.
    Clears itself on the next clean game exit; `greenboost gaming-mode off`
    fixes it now.
    """
    mode = resolve("gaming_mode_active")
    active = mode.get("value")
    if active is None:
        # Cannot read the flag , say so. A clean bill of health inferred
        # from absent data is the failure mode this layer exists to prevent.
        return None, [{"gaming_mode_active": None, "why": mode.get("raw_source")}]
    running = _wine_game_running()
    if active and running is None:
        return None, [{"gaming_mode_active": True,
                       "why": "/proc unreadable , cannot tell if a game is running"}]
    matched = bool(active) and running is False
    orphans = resolve("gaming_session_orphans")
    return matched, [{"gaming_mode_active": bool(active),
                      "wine_game_running": running,
                      "last_stop_orphans": orphans.get("value")}]


def _seg_rule1_underfilled():
    vram = resolve("vram_fill_pct")
    t2o = resolve("t2_overflow_active_mb")
    t2f = resolve("t2_pressure_fraction")
    t3 = resolve("t3_spill_active_mb")
    v = vram.get("value")
    # "Overflow is active" means bytes are actually being served from T2/T3.
    # t2_pressure_fraction is POOL FULLNESS, not overflow, and on a large pool
    # it reads 0 while overflow is substantial — measured 2026-08-17,
    # t2_overflow_total_mb=14326 into a 43 GB pool at pressure 0.0 while VRAM
    # sat at 69%. Testing pressure alone made this tripwire unable to fire in
    # precisely the scenario Rule #1 exists to catch, so the shim's own overflow
    # counter leads and pressure is kept only as a secondary signal.
    overflow_active = ((t2o.get("value") or 0) > 0
                       or (t2f.get("value") or 0) > 0
                       or (t3.get("value") or 0) > 0)

    # `None` (unknown) and `0` (measured zero) are NOT the same answer, and
    # collapsing them silences this tripwire exactly when the shim goes quiet.
    # The shim rewrites /run/greenboost/shim_stats every 250 ms *from its CUDA
    # hooks*, so the file legitimately goes stale whenever inference is idle —
    # and t2_overflow_active_mb then resolves to None. Before this fix that
    # produced `matched: False`, i.e. a clean bill of health, from a state that
    # was actually a live violation. Measured 2026-08-17: vram_fill_pct=50.3
    # (below_target) with t2_overflow_total_mb=16312 sitting in the file the
    # resolver had just declined to trust.
    #
    # Return None = "cannot tell", which evaluate_segment already carries (it
    # uses the same value on its exception path). This still never ASSERTS a
    # violation on absent data — the rule this segment's tests encode — it only
    # stops asserting health on absent data, which is the other half of the same
    # principle. Only the decisive signal (the shim's own overflow counter)
    # triggers it; a known-zero overflow with quiet pressure/T3 stays a
    # definitive False.
    overflow_unknown = (t2o.get("value") is None
                        and not (t2f.get("value") or 0) > 0
                        and not (t3.get("value") or 0) > 0)
    if v is not None and v < 85.0 and overflow_unknown:
        return None, [vram, t2o, t2f, t3]

    # VRAM that is RESERVED for KV but not yet written is not "unfilled" in the
    # sense Rule #1 cares about. The shim holds kv_reserve_effective_mb from
    # the moment the model loads and consumes it as the conversation grows, so
    # a fresh session at a large ctx legitimately shows a gap.
    #
    # Measured 2026-08-18, a healthy serve caught by this tripwire: fill 79.4%,
    # kv_reserve_effective_mb=2009, kv_t1_tracked_mb=218 , about 1.8 GB of the
    # 2.5 GB "free" was reservation, not waste. Counting it as a violation
    # trains the operator to ignore the one tripwire Rule #1 depends on, which
    # is worse than the occasional real miss.
    #
    # Only the reserve that is NOT yet in use is added back. A reserve the
    # engine has actually filled already shows up in vram_fill_pct on its own,
    # and adding it twice would hide a genuine underfill.
    # _shim_stats() returns (fields, reason); fields is None when the file
    # cannot be trusted. Treat "cannot read" as "no reserve credit", never as
    # a reason to soften the verdict.
    _fields, _ = _shim_stats()
    _fields = _fields or {}
    held = _fields.get("kv_reserve_effective_mb")
    used = _fields.get("kv_t1_tracked_mb")
    effective = v
    reserve_note = None

    def _num(x):
        # shim_stats is a flat key=value text file, so every field arrives as a
        # STRING. An isinstance(x, (int, float)) check silently rejects all of
        # them, which is how the first version of this credit did nothing at
        # all while looking correct.
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    held_v, used_v = _num(held), _num(used)
    if v is not None and held_v is not None and used_v is not None:
        pending_mb = max(0.0, held_v - used_v)
        # Physical VRAM read live, never a stored constant (hardcoded-hardware
        # rule) , same source _seg_weights_dont_fit_vram uses.
        total = None
        try:
            import gb_monitor
            snap = gb_monitor.snapshot()
            total = (getattr(snap, "vram_physical_mb", None)
                     or getattr(snap, "gpu_mem_total_mb", None))
        except Exception:
            total = None
        if pending_mb > 0 and total:
            effective = min(100.0, v + 100.0 * pending_mb / float(total))
            reserve_note = {
                "metric": "kv_reserve_pending_mb",
                "value": round(pending_mb, 1),
                "unit": "megabytes",
                "doc": ("VRAM reserved for KV that the conversation has not "
                        "reached yet. Counted as filled for Rule #1: it is "
                        "committed, not wasted."),
                "vram_fill_pct_effective": round(effective, 1),
            }

    matched = effective is not None and effective < 85.0 and overflow_active
    ev = [vram, t2o, t2f, t3]
    if reserve_note:
        ev.append(reserve_note)
    return matched, ev


def _seg_weights_on_t3():
    t3 = resolve("t3_spill_active_mb")
    v = t3.get("value")
    return bool(v) and v > 0, [t3]


def _seg_proxy_memory_near_cap(entity_id=None, window_s=None):
    """The gb-synapse proxy is close to its own address-space cap.

    Past this line the next allocation inside a request raises MemoryError and
    that request dies, while the box itself stays healthy , which is exactly
    what makes the symptom confusing ("plenty of free RAM, yet generation
    stopped"). Returns None, never False, when the proxy is not running or has
    no cap: absence of a reading is not a clean bill of health.
    """
    rss = resolve("proxy_rss_mb")
    head = resolve("proxy_mem_headroom_pct")
    hv = head.get("value")
    if hv is None:
        return None, [rss, head]
    return hv <= 20.0, [rss, head]


def _seg_agent_compaction_broke_prefix(entity_id=None, window_s=None):
    """A compaction rewrote the prefix instead of appending to it.

    This is AE-2's failure mode, and it is expensive rather than incorrect:
    the answer is still right, it just costs a full re-prefill. Returns None,
    never False, when no compaction has happened , "nothing has gone wrong
    yet" and "nothing has happened yet" are different answers.
    """
    kept = resolve("agent_compaction_prefix_kept_pct")
    v = kept.get("value")
    if v is None:
        return None, [kept]
    return v < 100.0, [kept]


def _seg_kv_prefetch_has_no_target(entity_id=None, window_s=None):
    """There is no T2-resident KV to prefetch.

    The gate on the whole lookahead-KV-prefetch branch. Measured on this box
    2026-08-20 while serving the reference model at ctx 24576: KV sat entirely
    in T1 (427 MB tracked) and the 6687 MB in T2 was all weights , a prefetch
    mechanism there would move zero bytes. True is a legitimate, branch-ending
    answer, not a failure.

    Null when the shim is not reporting: "nobody measured" must stay
    distinguishable from "measured, and there is nothing there".
    """
    kv = resolve("kv_t2_resident_mb")
    v = kv.get("value")
    if v is None:
        return None, [kv]
    return v <= 0, [kv]


def _seg_bandwidth_bound_harvestable(entity_id=None, window_s=None):
    """The card is holding power it cannot use.

    Decode is bandwidth-bound (weights streaming from T2, GPU utilisation low)
    while the board still draws near its limit. Nothing is broken , throughput
    is exactly what the link allows , but the watts are being spent waiting,
    and the control loop can give them back without touching tok/s.

    Returns None, not False, when the power or overflow reading is missing:
    "the sensors did not answer" is not "there is nothing to harvest".
    """
    over = resolve("t2_overflow_active_mb")
    harvesting = resolve("tuner_harvesting")
    ov = over.get("value")
    if ov is None:
        return None, [over, harvesting]
    if ov <= 0:
        return False, [over, harvesting]
    # Already harvesting is not "harvestable" , the loop has it in hand.
    if harvesting.get("value") is True:
        return False, [over, harvesting]
    util = None
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi",
                            "--query-gpu=utilization.gpu,power.draw,power.limit",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        f = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        util, draw, limit = float(f[0]), float(f[1]), float(f[2])
    except Exception:
        return None, [over, harvesting]
    ev = [over, harvesting,
          {"metric": "gpu_util_pct_live", "value": util, "unit": "percent",
           "governed": False,
           "provenance": {"raw_source": "nvidia-smi utilization.gpu (live)"}},
          {"metric": "gpu_power_headroom_pct", "value": round(100.0 * draw / limit, 1)
           if limit else None, "unit": "percent", "governed": False,
           "provenance": {"raw_source": "nvidia-smi power.draw/power.limit (live)"}}]
    return (util < 60.0 and limit > 0 and draw >= limit * 0.80), ev


def _seg_ctx_estimate_undercounts(entity_id=None, window_s=None):
    """The client thinks requests are smaller than the server says they are.

    Only the under-count direction matters. Over-estimating compacts earlier
    than strictly needed, which costs some prefix reuse; under-estimating lets
    a request be assembled that the server then rejects, which costs the turn.
    Returns None, not False, when no turn has been graded yet.
    """
    err = resolve("ctx_estimate_error_pct")
    v = err.get("value")
    if v is None:
        return None, [err]
    return v <= -5.0, [err]


def _seg_agent_hallucinating_tool_names(entity_id=None, window_s=None):
    """The model is naming tools the registry does not have.

    Above a few percent this is schema misalignment, not bad luck, and PATool
    says the fix is to align the offered schema to what the model expects ,
    not to instruct the model harder.
    """
    pct = resolve("agent_hallucinated_tool_pct")
    v = pct.get("value")
    if v is None:
        return None, [pct]
    return v >= 5.0, [pct]


def _seg_agent_eval_below_reference(entity_id=None, window_s=None):
    """This harness scores below a 27B's published MCP-Bench reference.

    0.582 is where MCP-Bench puts gemma-3-27b-it. Scoring under that with a
    27B-class local model points at the harness, not the weights , which is
    the whole reason AE-1 exists before any other AE task may claim a win.
    """
    sc = resolve("agent_eval_overall")
    v = sc.get("value")
    if v is None:
        return None, [sc]
    return v < 0.582, [sc]


def _seg_host_oom_imminent():
    """Host RAM is close enough to exhaustion that the OOM killer is a real risk.

    This is the signal that was missing on 2026-08-18, when the OOM killer took
    the whole terminal twice , 12:30:46 killed a python3 holding 27.0 GB of
    anonymous RSS, and 07:52:32 killed one holding 31.7 GB. Both times
    greenboost.ko's T2 OOM guard had already tripped and said so in dmesg
    minutes earlier, where nothing was watching.

    Severity is `violation` rather than `diagnosis` because, unlike weights that
    do not fit, this one destroys work in progress: it does not slow the session
    down, it ends it.

    Returns None, never False, when either input is unreadable. "I cannot see
    host memory" must not be reported as "host memory is fine".
    """
    avail = resolve("host_mem_available_gb")
    guard = resolve("t2_oom_guard_state")
    av = avail.get("value")
    gv = guard.get("value")
    if av is None and gv is None:
        return None, [avail, guard]
    tripped = isinstance(gv, str) and gv.lower() not in ("ok", "none", "")
    # 4 GB mirrors the kmod's own reserve, which is what it compares
    # MemAvailable against before it starts evicting T2.
    low = av is not None and av < 4.0
    ev = {"host_mem_available_gb": av, "t2_oom_guard": gv,
          "kmod_reserve_gb": 4.0,
          "action": "greenboost clear memory-pool  (frees T1+T2 now)"}
    if av is None or gv is None:
        # One input missing: report only if the one we DO have is alarming,
        # otherwise say we cannot tell.
        return (True, [ev]) if (tripped or low) else (None, [avail, guard])
    return (tripped or low), [ev]


def _seg_models_wiped_from_manifest():
    """Weights on disk that GB-Synapse cannot serve because the manifest lost them.

    Severity is `violation`, not `diagnosis`: unlike a model that genuinely
    does not fit, nothing about the hardware makes this true. It is pure
    bookkeeping loss, it makes the CLI refuse to start, and it is fixed in
    seconds by `gb_synapse.recover_from_hf_cache()` without downloading a byte.

    Happened three times on 2026-08-18. Twice the operator's next action was
    to open the CLI and find it dead with `no such model:
    Qwen3.8-27B-Cold-Fusion-MTP-IQ4_XS` — a model whose 15.85 GiB of weights
    were sitting on the disk the whole time.
    """
    n = resolve("unregistered_cached_models")
    if n.get("value") is None:
        # Cannot tell — never False. A clean bill inferred from an unreadable
        # cache is the failure mode this layer exists to prevent.
        return None, [n]
    count = int(n["value"])
    ev = dict(n)
    ev["examples"] = (n.get("evidence") or {}).get("missing", [])
    ev["fix"] = "greenboost recover-models"
    return count > 0, [ev]


def _seg_dataflux_history_reset():
    """The log is (near) empty because it was cleared, not because it is broken.

    These two states are indistinguishable from any single metric: every
    resolver that reads recent events returns None either way. The operator's
    next action is opposite in each case , wait for the log to refill, or go
    find out why nothing is recording , so the layer has to be able to tell
    them apart rather than shrugging.

    The discriminator is the kernel module. A cleared log on a machine whose
    stack is up and loaded is a reset in progress; a near-empty log while the
    module is NOT loaded is the far more serious `kmod_missing_silent_degrade`
    case, which owns that verdict and must not be masked by this one.

    Added 2026-08-21 with `greenboost clear logs`, which empties the log by
    design and would otherwise leave the whole governed layer answering "no
    data" with no governed explanation for why.
    """
    n = resolve("dataflux_event_count")
    kmod = resolve("kmod_loaded")
    if n.get("value") is None:
        # Cannot read the log at all , that is not a reset, that is unknown.
        return None, [n, kmod]
    count = int(n["value"])
    if count > _DATAFLUX_RESET_MAX_EVENTS:
        return False, [n, kmod]
    # Near-empty. Only call it a reset when the stack is otherwise healthy;
    # otherwise defer to kmod_missing_silent_degrade rather than reporting a
    # reassuring verdict over a broken one.
    if kmod.get("value") is not True:
        return None, [n, kmod]
    return True, [n, kmod]


def _seg_vram_headroom_exhausted():
    """VRAM filled PAST the Rule #1 target into the 10% safety reserve.

    The mirror of `rule1_underfilled`, and it existed as a blind spot for as
    long as that one has existed: the tripwire watched the low side only, so a
    card at 97.9% , 254 MB free of 12227 , produced no verdict at all, while
    the identical card at 84.9% produced a `violation`. Rule #1 asks for ~90%
    WITH 10% headroom; both directions are departures from it, and the high
    side is the one that has nowhere to fail safely.

    Matched means: reduce what is resident (shorter ctx, a smaller quant, or
    stop a second process holding VRAM), do NOT read it as "excellent
    utilisation". Returns None when fill cannot be read , an unknown card is
    never a healthy card.
    """
    hr = resolve("vram_headroom_pct")
    fill = resolve("vram_fill_pct")
    v = hr.get("value")
    if v is None:
        return None, [hr, fill]
    return (v < _VRAM_MIN_HEADROOM_PCT), [hr, fill]


def _seg_weights_dont_fit_vram():
    """The served model's weights alone exceed physical VRAM.

    This is a CAPACITY FACT, not a fault, and it is the honest answer to "why
    is decode slow" whenever it matches: whatever does not fit crosses PCIe on
    every forward pass, which caps decode at (bytes over PCIe) / (link
    bandwidth) no matter how well everything else is tuned.

    It exists because nothing in the governed layer said this out loud.
    Measured on this box 2026-08-18: 15.86 GB of weights against 12227 MiB of
    VRAM, decode 2.7-3.5 tok/s — arithmetic that predicts the measurement
    almost exactly. What the operator actually saw instead was five
    `tok_s_drop` advisories built on n=3 populations, pointing at a KV
    threshold lever that cannot move a weights-overflow problem.

    Severity is `diagnosis`, not `violation`: the fix is a model/quant/ctx
    choice, not a bug to repair, and Rule #1 is not being broken — the tiering
    layer is doing exactly what it exists to do.
    """
    w = resolve("weights_gb")
    fill = resolve("vram_fill_pct")
    wv = w.get("value")
    if wv is None:
        return None, [w, fill]
    # Physical VRAM read live, never a stored constant (hardcoded-hardware rule).
    total_gb = None
    try:
        import gb_monitor
        snap = gb_monitor.snapshot()
        phys_mb = (getattr(snap, "vram_physical_mb", None)
                   or getattr(snap, "gpu_mem_total_mb", None))
        if phys_mb:
            total_gb = float(phys_mb) / 1024.0
    except Exception:
        total_gb = None
    if not total_gb:
        return None, [w, fill]
    overflow_gb = round(wv - total_gb, 2)
    ev = dict(w)
    ev["vram_physical_gb"] = round(total_gb, 2)
    ev["weights_overflow_gb"] = overflow_gb
    return overflow_gb > 0, [ev, fill]


def _seg_cloud_inference_in_use():
    """An inference endpoint is pointed at hardware the owner does not control.

    Severity `violation`: this is the one rule whose breach invalidates the
    entire stack for that request. Every tier, the shim, the kernel module and
    the cluster fabric exist so inference stays on this hardware.

    Returns None (cannot tell) rather than False when endpoints cannot be
    resolved , silence must never read as compliance.
    """
    m = resolve("inference_endpoint_is_local")
    v = m.get("value")
    if v is None:
        return None, [m]
    return (not v), [m]


def _seg_prefill_dominated_by_replay():
    """Most of this turn's prefill was re-running work the engine already did.

    Matches when the recurrent replay estimate is the majority of prompt_ms.
    A capacity/architecture FACT rather than a fault , the model is hybrid and
    llama.cpp has no way to resume a recurrent layer mid-prefix, which is also
    why it refuses `--cache-reuse` for this model and says so at startup.

    It is governed because it is the largest recoverable per-turn cost on this
    box and nothing named it: an operator watching a 99.9% cache-hit rate has
    every reason to believe prefill is already solved. Actionable via the
    sparse-prefix-checkpointing work (arXiv 2605.05219) on top of llama.cpp slot
    save/restore, which does carry recurrent state.
    """
    m = resolve("recurrent_replay_ms")
    v = m.get("value")
    if v is None:
        return None, [m]
    return (m.get("replay_share") or 0) >= _REPLAY_DOMINANT_SHARE, [m]


def _seg_idle_serve_holding_vram():
    """A serve session is holding significant VRAM while doing no work.

    Not a fault , llama-server keeps weights resident precisely so the next
    request is fast, and that is usually the right trade. It becomes a finding
    only when the operator wants the card back (a game, a training run, another
    model) and has no signal telling them the memory is reclaimable.

    Actionable by construction: `gb synapse pause <model>` saves the KV state
    and returns the VRAM, and `resume` brings the conversation back. Measured
    on this box: save 740 ms / restore 537 ms for 54,377 tokens.
    """
    m = resolve("idle_vram_held_gb")
    v = m.get("value")
    if v is None:
        return None, [m]
    fill = resolve("vram_fill_pct")
    # Only interesting once it is a meaningful share of the card, not for a
    # few hundred MB of context left over.
    f = fill.get("value")
    return (v > 0 and f is not None and f >= _IDLE_HOLD_FILL_PCT), [m, fill]


def _seg_pcie_link_degraded_under_load():
    """PCIe link below its ceiling WHILE the GPU is genuinely busy.

    Separate from the `pcie_degraded` dataflux event, which is sampled once at
    telemetry init and produced 13 false positives on this box (2026-08-18:
    every one reported gen2 while the link measures Gen4 x16 under real load).
    A GPU downtrains its link whenever it is idle, so a gen comparison is only
    meaningful alongside proof the card was working at the time — this reads
    both in the same call so the two cannot drift apart.
    """
    g = resolve("pcie_link_gen_current")
    gen = g.get("value")
    if gen is None:
        return None, [g]
    util = g.get("gpu_util_pct")
    if util is None or util < _PCIE_ACTIVE_UTIL_PCT:
        # Idle or unknown: refuse to answer rather than report a downtrained
        # link as a fault. "Cannot tell" is the honest verdict here.
        return None, [g]
    return (gen < (g.get("gen_max") or gen)
            or (g.get("width_current") or 0) < (g.get("width_max") or 0)), [g]


# Node labels in dataflux's per-node rollup that are NOT feeders. "host" is
# this box; "?" is the unknown-node bucket dataflux uses for events that
# carried no node label at all. Neither can be "a feeder sitting idle".
_NON_FEEDER_NODES = {"host", "?", "", None}


def _seg_feeder_idle_while_host_saturated():
    """Two false-positive sources fixed 2026-08-18, both observed live on this
    box at once (the segment reported matched=true with a feeder that does not
    exist and a feeder that was not connected):

    1. The `"?"` unknown-node bucket was being counted as a feeder. dataflux's
       per-node rollup returned {"host": 173, "?": 0}, and `"?" != "host"` with
       0 items read as "a feeder processed nothing". There is no feeder named
       "?" — it is where events with no node label land.

    2. An UNREACHABLE feeder was counted as idle. This segment's own doc says
       it "assumes the feeder IS connected and only checks whether it's doing
       work", and `feeder_unreachable` exists precisely to cover the other
       case. Firing both at once tells an agent there are two independent
       cluster violations when there is one condition with one fix, and the
       remediations differ: bring the feeder up, versus find out why a
       connected feeder is not being dispatched to.

    Idle-and-reachable stays a real violation and still matches."""
    host_fill = resolve("vram_fill_pct", entity_id="host")
    feeders = resolve("feeder_items")
    reach = resolve("feeder_reachable")
    v_host = host_fill.get("value")
    items = feeders.get("value") or {}
    reachable = reach.get("value") or {}

    idle_feeders = [
        n for n, c in items.items()
        if n not in _NON_FEEDER_NODES
        and (c or 0) == 0
        # Unknown reachability (feeder not in the map at all) is treated as
        # reachable: the pre-existing behaviour, so a cluster whose probe is
        # unavailable still surfaces a genuinely idle feeder rather than
        # silently passing.
        and reachable.get(n, True) is not False
    ]
    matched = bool(idle_feeders) and v_host is not None and v_host >= 85.0
    return matched, [host_fill, feeders, reach]


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


def _seg_kmod_version_drift():
    """Installed release != running module. None when either side is unknown.

    Deliberately returns None rather than False when the module isn't loaded
    or no build stamp exists: a clean bill of health inferred from absent data
    is the exact failure this layer exists to prevent, and "no drift" is a
    claim you can only make when you have read both numbers.
    """
    installed = resolve("core_build_version")
    running = resolve("kmod_loaded_version")
    iv, rv = installed.get("value"), running.get("value")
    if not iv or not rv:
        return None, [installed, running]
    return (iv != rv), [installed, running]


def _seg_proton_tool_race():
    """Two different GreenBoost Proton tools started the same game at once.

    Measured 2026-08-21: `greenboost-proton-experimental` at 02:57:48 and
    `greenboost-proton` at 02:57:50 , two seconds apart, both delegating to a
    DIFFERENT upstream Proton, both into wine prefix compatdata/2909400. One
    prefix cannot be brought up twice; each run's explorer.exe came up, the
    game never did, and the pair was force-killed about a minute later. Steam
    then retried, four times.

    Steam's CompatToolMapping named only ONE of the two, so this is not a
    user misconfiguration that shows up in Steam's UI , it needs the
    wrapper's own logs to see at all, which is why it is a segment.
    """
    runs = resolve("game_launch_attempts")
    tools = (runs.get("details") or {}).get("tools")
    if tools is None:
        return None, [runs]
    return (len(tools) > 1), [runs, resolve("proton_tools_installed")]


def _seg_game_launch_retry_loop(entity_id=None, window_s=None):
    """Repeated wrapper starts with no game process , a launch that is looping.

    Three or more starts inside the window means something kills the game and
    Steam retries; a spinner cannot distinguish that from a slow first run,
    and on 2026-08-21 it looked exactly like "still loading" for four minutes.
    """
    attempts = resolve("game_launch_attempts")
    running = resolve("game_process_running")
    last = resolve("last_game_session_s")
    n, r = attempts.get("value"), running.get("value")
    if n is None or r is None:
        return None, [attempts, running, last]
    # A completed session outranks the noise before it. Five aborted starts
    # followed by one 335-second clean run is a launch that WORKED, and
    # reporting it as a retry loop would be crying wolf on a success
    # (measured 2026-08-21). Only sessions that ended recently count , an
    # hours-old success says nothing about a loop happening now.
    import time as _t
    ended = (last.get("details") or {}).get("ended_ts") or 0
    secs = last.get("value") or 0
    recent_success = (secs >= 60 and ended and (_t.time() - ended) <= (window_s or 900.0))
    return (n >= 3 and not r and not recent_success), [attempts, running, last]


def _seg_launch_made_only_wine_infrastructure():
    """Proton came up, the game did not.

    wine's own helpers (explorer.exe, services.exe, wineboot) start for every
    launch whether or not the title does. Counting any .exe as "the game is
    running" reports a dead launch as a live one , this segment is matched
    precisely when the ONLY wine processes present are Proton's own.
    """
    running = resolve("game_process_running")
    infra = (running.get("details") or {}).get("wine_infrastructure_only")
    if running.get("value") is None:
        return None, [running]
    return bool(infra), [running]


def _seg_apparmor_contaminated_by_greenboost():
    """GreenBoost rules sitting past a snapd profile's closing brace.

    Matched means every snap on the machine is one `apparmor_parser` run away
    from refusing to launch, and that `snapd.apparmor.service` is already dead.

    The trap this encodes: `grep libgreenboost_audit <profile>` returning a hit
    reads like "the grant is active". It is the exact opposite. A rule past the
    closing brace grants nothing and invalidates the whole profile, so the
    presence of the string IS the failure signal. That misreading is why the
    same bug shipped twice.
    """
    n = resolve("apparmor_profiles_contaminated")
    v = n.get("value")
    if v is None:
        return None, [n]
    return (v >= 1), [n]


def _seg_kmod_stale_from_initramfs():
    """Version drift that a reload will NOT durably fix, because the stale
    module is coming out of the initramfs on every boot.

    This is deliberately a separate segment from `kmod_version_drift` rather
    than a refinement of it, because the two carry different instructions and
    handing out the wrong one costs a reboot to discover:

      kmod_version_drift          -> `sudo greenboost load` and you are done.
      kmod_stale_from_initramfs   -> `sudo greenboost load` fixes THIS boot and
                                     the next boot puts v-old back. The durable
                                     fix is to stop shipping the module in the
                                     boot image (`update-initramfs -u -k all`
                                     with the exclusion hook installed).

    Returns None when either input is unknown , the module isn't loaded, no
    build stamp exists, or the journal can't be read. "Cannot tell" is never
    "no drift", which is the whole reason this layer exists.
    """
    drift = evaluate_segment("kmod_version_drift")
    stage = resolve("kmod_load_stage")
    d, st = drift.get("matched"), stage.get("value")
    ev = [{"kmod_version_drift": d, "evidence": drift.get("evidence")}, stage]
    if d is None or st is None:
        return None, ev
    return (bool(d) and st == "initramfs"), ev


def _seg_t3_configured_but_disabled():
    """T3 has a configured cap and is not actually open.

    The failure this names is not "T3 is off" , it is that everything an
    operator would check says T3 is fine. The status page prints the cap,
    `t3_total_mb` reports 74752, and the only place the truth appears is one
    kernel log line at module init. Matched means: no byte will ever reach
    T3 in this boot, whatever the capacity readouts say.
    """
    enabled = resolve("t3_enabled")
    total = resolve("t3_capacity_configured_mb")
    e, t = enabled.get("value"), total.get("value")
    if e is None:
        return None, [enabled, total]
    return (e is False and bool(t)), [enabled, total]


def _seg_prompt_cache_cold():
    hit = resolve("prompt_cache_hit_pct")
    v = hit.get("value")
    return (v is not None and v < 10.0), [hit]


def _serve_sessions_with_dead_proxy() -> list:
    """Serve sessions whose engine is alive but whose proxy is gone.

    That combination is an outage that looks like health from every other
    angle: the kernel module is loaded, VRAM is full, the GPU is idle because
    nothing can reach it. Returns [] when gb_synapse cannot be consulted ,
    "cannot tell" must not be reported as "a proxy died", the same way it must
    not be reported as health.
    """
    try:
        import gb_synapse
        out = []
        for s in gb_synapse.ps():
            if s.get("proxy_error") and s.get("llama_pid"):
                out.append({"model": s.get("model"),
                            "llama_pid": s.get("llama_pid"),
                            "proxy_error": s.get("proxy_error"),
                            "port": s.get("port")})
        return out
    except Exception:
        return []


def _seg_serve_healthy():
    """Fixed 2026-07-30: this previously required shim_fresh unconditionally,
    so it could never match on a genuinely idle-but-healthy box (shim_fresh
    is only true while a CUDA process is actively writing shim_stats within
    the last 30s) — reproduced live on a box with kmod loaded and zero
    errors, just idle. shim_fresh/vram-in-band/quality-floor only make sense
    while something is actually being served; use a dataflux-event-sourced
    "is anything actively decoding" signal (tok_s_measured, NOT
    kmod_loaded/shim_fresh, which are OS-resolved and not fixture-
    controllable) so the idle branch stays deterministic under the eval
    fixture in tests/fixtures/semantics/events.py, whose scenario represents
    an ACTIVE Rule#1-violating session, not an idle one."""
    kmod = resolve("kmod_loaded")
    fresh = resolve("shim_fresh")
    vram = resolve("vram_fill_pct")
    floor = resolve("meets_fp8_floor")
    if kmod.get("value") is not True:
        return False, [kmod, fresh, vram, floor]

    # A serve session whose PROXY is dead is not healthy, however idle the box
    # looks. gb_synapse.ps() already reports `proxy_error: "proxy process is
    # gone"` in exactly this state , the information existed and this segment
    # ignored it.
    #
    # Overnight 2026-08-19: the proxy was OOM-killed at 01:47, llama-server
    # survived holding 10.5 GB of VRAM, every request failed with "Cannot
    # connect to http://localhost:11369/v1" , and this segment answered
    # MATCHED, a clean bill of health, for eight hours. Nothing else in the
    # layer contradicted it. That is the exact failure this layer exists to
    # prevent, one level up from a null metric.
    broken = _serve_sessions_with_dead_proxy()
    if broken:
        return False, [kmod, fresh, vram, floor, {
            "metric": "serve_proxy_dead",
            "value": True,
            "unit": "boolean",
            "doc": ("A serve session is running with no reachable proxy. The "
                    "engine still holds its VRAM; nothing can send it a "
                    "request."),
            "sessions": broken,
            "action": "greenboost synapse stop && greenboost synapse serve <model>",
        }]
    if _latest_event("tok_s_measured", max_age_s=60.0) is None:
        # No recent decode activity — idle GPU. Healthy iff kmod is loaded;
        # the VRAM target band / freshness gate / quality floor only apply
        # to an active serve session.
        return True, [kmod, fresh, vram, floor]
    matched = (fresh.get("value") is True
               and (vram.get("value") or 0) >= 60.0 and floor.get("value") is not False)
    return matched, [kmod, fresh, vram, floor]


def _seg_feeder_unreachable():
    reachable = resolve("feeder_reachable")
    values = reachable.get("value") or {}
    matched = any(v is False for v in values.values())
    return matched, [reachable]


def _seg_gaming_inference_contention():
    gaming = resolve("gaming_session_active")
    t2f = resolve("t2_pressure_fraction")
    matched = gaming.get("value") is True and (t2f.get("value") or 0) > 0.3
    return matched, [gaming, t2f]


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

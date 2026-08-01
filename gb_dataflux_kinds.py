#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_dataflux_kinds.py , the dataflux event-kind registry.

`workflow/dataflux-schema-additions.md`'s parity rule: "an event kind that's
emitted but not queryable, or documented as queryable but never emitted, is a
bug." Before this registry, 22 of ~40 known kinds had no dedicated MCP tool ,
`dataflux_events(kind=...)` was the only way to reach them, and nothing an
LLM client could discover without reading source. Adding a new per-kind tool
for each one doesn't scale (an LLM can't discover 40 tools' semantics either),
so this registry is the single source of truth that:

  1. Lets `gb_dataflux_mcp.py`'s `dataflux_group`/`dataflux_schema` tools
     answer for ANY kind, including all the previously-orphaned ones, with
     zero new tool code per kind.
  2. Drives `gb_dataflux.summarize()`'s per-kind rollup and
     `gb_dataflux.critic_report()`'s incident classification, so a newly
     registered kind is aggregated/incident-eligible automatically.
  3. Is asserted against by `checks/check_dataflux_coverage.py`'s blocking
     pass , every literal kind emitted anywhere in this repo (or the C shim)
     must be a key here, and every key must have a real emit site or be
     explicitly `planned` (emitted by a consumer repo like ai-forge, not by
     GreenBoost itself).

Stdlib-only (no gb_* imports) so `gb_dataflux.emit()` itself can import this
module for free without adding an import-time dependency to the hot path.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KindSpec:
    group: str
    doc: str
    # Fields a well-formed event of this kind is expected to carry, beyond
    # the universal ones emit() always stamps (ts, pid). Advisory only ,
    # nothing currently enforces this at emit time (emit() must never
    # raise), it's documentation for dataflux_schema() consumers.
    fields: tuple[str, ...] = ()
    # Numeric fields worth averaging/summing in a rollup (summarize()).
    numeric_fields: tuple[str, ...] = ()
    # Which event.status values make this kind an incident in
    # gb_dataflux.critic_report() BEYOND the universal status=="error" rule
    # (e.g. a "warn" status that's still worth surfacing as an incident for
    # this particular kind, even though "warn" isn't inherently an error
    # for every kind).
    incident_when: tuple[str, ...] = ()
    # Cluster-sync scope: "host" kinds are never re-emitted by
    # gb_cluster.sync_feeder_dataflux (host already has its own copy of
    # whatever this represents); "all" kinds ARE meaningful to sync from a
    # feeder (its own first-person telemetry that nothing else provides).
    # Advisory today (see gb_cluster.py's own note on why a blind kind
    # filter was reconsidered and NOT applied during the P4 pass) , kept
    # here so a future, carefully-scoped filter has a real field to read
    # instead of a hardcoded list.
    sync_scope: str = "all"
    # True when this kind is emitted by a CONSUMER repo (ai-forge) rather
    # than by GreenBoost itself , the coverage check accepts these without
    # requiring a local emit site, since grepping ai-forge is out of scope
    # for a check that lives in this repo.
    planned: bool = False


KINDS: dict[str, KindSpec] = {
    # ── placement ──────────────────────────────────────────────────────
    "placement": KindSpec(
        group="placement", doc="LLM tensor-split / MoE expert-tier placement plan.",
        fields=("tensor_split", "floor_bits"), sync_scope="host"),
    "tensor_split": KindSpec(
        group="placement", doc="gb-synapse's --rpc tensor-split computation across host+feeders.",
        fields=("split", "version"), sync_scope="host"),
    "capacity_fit": KindSpec(
        group="placement", doc="gb_placement.capacity_fit partial-offload advisory for a serve request.",
        sync_scope="host"),
    "synapse_engine_placement": KindSpec(
        group="placement", doc="Torch-engine (gLLM) placement validation vs Rule #1 (synapse_engine_backends._validate_placement).",
        incident_when=("warn",)),

    # ── quant ──────────────────────────────────────────────────────────
    "quantize": KindSpec(
        group="quant", doc="Per-component gb-quant decision (component, bits/quality, budget/actual GiB).",
        numeric_fields=("budget_gb", "actual_gb")),
    "quantize_to_fit": KindSpec(
        group="quant", doc="Whole-pipeline quantize-to-fit decision.",
        numeric_fields=("budget_gb", "actual_gb")),
    "quant_plan": KindSpec(
        group="quant", doc="gb_quant.plan_quality()'s per-layer bit-assignment plan (tier, ceiling, precision histogram, bf16-kept count).",
        fields=("target", "error_ceiling", "precision_histogram", "bf16_kept"),
        numeric_fields=("mean_rel_err", "max_rel_err")),
    "turboquant_activate": KindSpec(
        group="quant", doc="TurboQuant K/V attention compression activated (bits, mode, device).",
        fields=("k_bits", "v_bits", "mode")),
    "quant_budget_fallback": KindSpec(
        group="quant", doc="gb_quant._auto_budgets() fell back to the 2 GiB sentinel (topology detection failed).",
        incident_when=("warn",)),
    "kernel_backend": KindSpec(
        group="quant", doc="Per-layer quantized-kernel backend resolution in gb_quant._delegate_patch.",
        fields=("bits", "backend", "skipped_prequantized", "bf16_kept", "scalar_fallback")),

    # ── synapse ──────────────────────────────────────────────────────────
    "synapse_serve": KindSpec(
        group="synapse", doc="gb-synapse serve() attempt (ok/loading/proxy_error/error).",
        fields=("model", "engine", "quant")),
    "synapse_stall": KindSpec(
        group="synapse", doc="Streaming response sat with zero new output past the stall threshold.",
        incident_when=("warn", "ok")),
    "synapse_build_preflight": KindSpec(
        group="synapse", doc="gb-synapse engine build preflight tool-check result."),
    "synapse_engine_error": KindSpec(
        group="synapse", doc="Torch-engine (gLLM) runtime error, emitted from synapse_engine/gllm/async_llm_engine.py."),
    "bw_undetectable": KindSpec(
        group="synapse", doc="recommend()'s link-bandwidth probe could not determine feeder link speed.",
        incident_when=("warn",)),
    "cpu_spillover": KindSpec(
        group="synapse", doc="gb_synapse_backends.py LlamaCppBackend.serve() put some of a "
                             "model's layers on the CPU instead of GPU (dense partial-offload, "
                             "or an OOM retry backing off -ngl) — the state that used to be "
                             "invisible to dataflux, root cause of the 2026-08-01 6 tok/s "
                             "incident. status is one of dense_partial_offload/"
                             "vram_fragmentation_oom_retry/compute_graph_oom_retry, all of "
                             "which are the incident itself, not a side detail.",
        fields=("model", "engine"),
        numeric_fields=("weights_gb", "budget_gb", "n_gpu_layers", "n_layers"),
        incident_when=("dense_partial_offload", "vram_fragmentation_oom_retry",
                       "compute_graph_oom_retry")),
    "tok_s_measured": KindSpec(
        group="synapse", doc="Real, client-observed decode tokens/sec for one model.",
        fields=("model",), numeric_fields=("tok_s",)),
    "model_rotation": KindSpec(
        group="synapse", doc="gb_rotator.py overnight rotation phase event.",
        fields=("model", "phase")),
    "prompt_cache": KindSpec(
        group="synapse", doc="Proxy-observed llama.cpp host-memory prompt-cache "
                             "outcome for one request (--cache-reuse/--cache-ram): "
                             "TTFT and the reused-vs-total prompt token share. "
                             "Feeds GB-Semantics' ttft_ms/prompt_cache_hit_pct metrics.",
        fields=("model",), numeric_fields=("ttft_ms", "hit_pct", "reused_tokens")),

    # ── cluster ──────────────────────────────────────────────────────────
    "node_topology": KindSpec(
        group="cluster", doc="Static per-node hardware identity (GPU, VRAM, cores, RAM, PCIe, NVMe).",
        sync_scope="host"),
    "node_capabilities": KindSpec(
        group="cluster", doc="Shim/kmod feature capability manifest observed at process init."),
    "link_transfer": KindSpec(
        group="cluster", doc="A real host<->feeder data transfer (bytes, seconds, derived bandwidth).",
        numeric_fields=("nbytes", "seconds")),
    "chunk_local": KindSpec(group="cluster", doc="cluster_map/ClusterJobQueue chunk executed on the host."),
    "chunk_remote": KindSpec(group="cluster", doc="cluster_map/ClusterJobQueue chunk dispatched to a feeder."),
    "job_local": KindSpec(group="cluster", doc="ClusterJobQueue single item executed on the host."),
    "job_local_retry": KindSpec(group="cluster", doc="ClusterJobQueue item retried locally after a remote failure."),
    "job_remote": KindSpec(group="cluster", doc="ClusterJobQueue single item dispatched to a feeder."),
    "model_push": KindSpec(group="cluster", doc="ensure_feeder_model pushed/verified a model onto a feeder."),
    # Real emitted kind is "stage" (gb_cluster.py's stage_bundle() function
    # passes "stage_bundle" as the event's `label`, not its `kind` , caught
    # live by check_dataflux_coverage.py's kind-literal scan when this
    # registry was first built, which had assumed the function name and the
    # kind literal were the same string).
    "stage": KindSpec(group="cluster", doc="A one-shot torch stage run on a feeder over SSH "
                      "(gb_cluster.stage_bundle(); label carries the function name)."),
    "feeder_provision": KindSpec(
        group="cluster", doc="ensure_feeder_ready's provisioning gate (rsync/deps/model check) result.",
        incident_when=("warn", "error")),

    # ── shim ───────────────────────────────────────────────────────────
    "snapshot": KindSpec(
        group="shim", doc="Continuous flight-recorder snapshot (VRAM/GPU-util/KV/T2/T3/shim_phase).",
        numeric_fields=("fb_used_pct", "gpu_util_pct", "t2_pressure", "t3_pressure")),
    "shim_decision": KindSpec(
        group="shim", doc="Automatic per-allocation tier placement decision diffed from shim_stats counters.",
        fields=("tier", "reason"), numeric_fields=("bytes_mb",),
        incident_when=("warn",), sync_scope="host"),
    "shim_transition": KindSpec(
        group="shim", doc="Discrete shim_phase/t3_spill/t2_pressure/rule1_underfill transition.",
        incident_when=("warn",), sync_scope="host"),
    "tier_move": KindSpec(
        group="shim", doc="Explicit gb_model_tier.py promote/demote/evict call (rare; most placement is shim_decision)."),
    "mem_pool_trim": KindSpec(
        group="shim", doc="gb_mem_pool.MemPoolManager.trim()/trim_all() pool-reclaim result.",
        fields=("pool",), numeric_fields=("allocated_before_mb", "allocated_after_mb", "reclaimed_mb")),
    "reclaim": KindSpec(
        group="shim", doc="gb_reclaim.py run_reclaim() outcome — classified GPU/T2/T3 process "
                          "reclaim (scope=residue|ambiguous|all), the shared implementation "
                          "behind `greenboost clear memory-pool` and the greenboost-orchestrator "
                          "reclaim_run MCP tool (see docs/reclaim.md).",
        fields=("scope", "targets"),
        numeric_fields=("n_killed", "n_unloaded", "n_failed"),
        incident_when=("partial",)),
    "ssm_state": KindSpec(
        group="shim", doc="Selective-SSM (Mamba/Mamba2/hybrid Gated-DeltaNet) recurrent-"
                          "state cache-pool sizing decision (synapse_engine/gllm/"
                          "memory_manager.py's MemoryManager._clamp_ssm_pools) — working/"
                          "snapshot slot counts, and whether the snapshot pool (prefix-"
                          "state reuse) was clamped or disabled to fit the memory budget.",
        fields=("working_slots", "requested_snapshot_slots", "snapshot_slots"),
        numeric_fields=("per_slot_mb",), incident_when=("warn",)),

    # ── pipeline (mostly ai-forge-emitted; GreenBoost is the consumer/query side) ──
    "stage_profile": KindSpec(
        group="pipeline", doc="Per-pipeline-stage timing/status (ai-forge's mandatory per-stage convention).",
        planned=True),
    "image_gen": KindSpec(group="pipeline", doc="gb_diffusion_server.py image-generation request result."),
    "video_render": KindSpec(
        group="pipeline", doc="Per-render video pipeline event (ai-forge's LongLive runner).", planned=True),
    "candidate_selected": KindSpec(
        group="pipeline", doc="ai-forge best-of-N candidate selection rollup event.", planned=True),
    "model_call": KindSpec(
        group="pipeline", doc="ai-forge forge/gb_models.py per-call model usage event.", planned=True),
    # Found live 2026-07-30 (dataflux_kinds() showed real 14-day counts for
    # both — qc_summary=79, finish_summary=26 — with zero registry entry):
    # ai-forge's conduir_art_jobs pipeline emits its own QC-sweep/finish
    # summary rollups directly into the shared dataflux log.
    "qc_summary": KindSpec(
        group="pipeline", doc="ai-forge tools/conduir_art_jobs/qc_sweep.py quality-control sweep rollup.",
        planned=True),
    "finish_summary": KindSpec(
        group="pipeline", doc="ai-forge tools/conduir_art_jobs/finish_assets.py asset-finishing rollup.",
        planned=True),

    # ── gaming (emitted by greenboost_gaming, a separate repo, not GreenBoost itself) ──
    # Found live 2026-07-30 alongside qc_summary/finish_summary — same class
    # of gap: real events (gaming_session=1, gaming_vram_pressure=1 in the
    # 14-day window) with no registry entry. Correction 2026-07-30: NOT
    # live_stats.rs (that Rust file only READS dataflux.jsonl for the Live
    # view, e.g. get_dataflux_recent_impl) — the real emit site is the
    # Python Proton wrapper, greenboost_gaming/greenboost_proton/proton's
    # own _df_emit() helper (gaming_session at launch/exit, gaming_vram_pressure
    # from _check_t2t3_pressure() polling /sys/class/greenboost/greenboost/pool_brief).
    "gaming_session": KindSpec(
        group="gaming", doc="greenboost_gaming/greenboost_proton/proton (Python Proton wrapper): "
                            "a game session's start/stop lifecycle event.",
        planned=True),
    "gaming_vram_pressure": KindSpec(
        group="gaming", doc="greenboost_gaming/greenboost_proton/proton (Python Proton wrapper): "
                            "VRAM pressure observed during a game session (informs fan-daemon/"
                            "gaming_mode coexistence tuning).",
        planned=True),
    # Added 2026-07-30 (greenboost_gaming_polish.md): gb_gaming/fan_daemon.py
    # now emits this whenever it writes a new fan speed WHILE gaming_mode=1
    # (gated, not every idle-desktop fan tick) — lets dataflux correlate
    # thermal/fan behavior against gaming_vram_pressure events from the same
    # session.
    "gaming_fan_curve": KindSpec(
        group="gaming", doc="greenboost_gaming/gb_gaming/fan_daemon.py: a fan-speed change applied "
                            "during an active game session.",
        fields=("temp_c", "fan_pct", "held"), planned=True),
    # Added 2026-07-30: gb_gaming/gpu_profile.py's apply_profile() now emits
    # this on every successful (non-dry-run) clock/power/fan profile apply.
    "gaming_gpu_profile_applied": KindSpec(
        group="gaming", doc="greenboost_gaming/gb_gaming/gpu_profile.py: a GPU clock/power/fan "
                            "profile was applied (via nvidia-smi/nvidia-settings).",
        fields=("gpu_index", "power_limit_w", "core_offset_mhz", "mem_offset_mhz", "has_fan_curve"),
        planned=True),

    # ── health ─────────────────────────────────────────────────────────
    "pcie_degraded": KindSpec(
        group="health", doc="Reactive orchestrator detected a PCIe link running below its negotiated gen/width.",
        incident_when=("warn",)),
    "health_transition": KindSpec(
        group="health", doc="Reactive orchestrator health-state transition (thermal/ECC/clock).",
        incident_when=("warn",)),

    # ── agent ──────────────────────────────────────────────────────────
    "actuation": KindSpec(
        group="agent", doc="An orchestrator/MCP-actuation lever move (gated/dry-run status included).",
        fields=("lever", "gated")),
    "agent_run": KindSpec(
        group="agent", doc="run_under_greenboost() subprocess lifecycle (started/ok/error).",
        fields=("run_id", "status")),
    "a2a_request": KindSpec(group="agent", doc="A2A JSON-RPC gateway request (verb, gated/dry-run, outcome)."),

    # ── eval (model-quality verification gates) ───────────────────────
    "yarn_bake": KindSpec(group="eval", doc="gb_aviary YaRN context-extension bake result."),
    "niah_cert": KindSpec(group="eval", doc="gb_aviary needle-in-a-haystack quality certification result."),
    "smoke_gate": KindSpec(group="eval", doc="gb_aviary repetition-collapse smoke-gate result.",
                           incident_when=("warn", "error")),

    # ── bench (Speed Program Phase 0 measurement harness) ────────────────
    "bench_result": KindSpec(
        group="bench",
        doc="tests/bench/gb_pathbench and run_real_model.py before/after measurement "
            "(path bandwidth or end-to-end tok/s), keyed by config_hash so a later "
            "phase's change can be diffed against its own baseline row. Query via "
            "dataflux_group('bench') , no dedicated tool needed, that's the "
            "registry's whole point (see module docstring).",
        fields=("path", "config_hash", "model"),
        numeric_fields=("bandwidth_gb_s", "prefill_tok_s", "decode_tok_s"),
        incident_when=("error",)),
}


GROUPS: tuple[str, ...] = tuple(sorted({spec.group for spec in KINDS.values()}))


def schema(kind: "str | None" = None) -> dict:
    """Return the registry (or one kind's spec) as plain JSON-able dicts ,
    what `dataflux_schema()` hands back over MCP."""
    def _as_dict(k: str, s: KindSpec) -> dict:
        return {"kind": k, "group": s.group, "doc": s.doc, "fields": list(s.fields),
                "numeric_fields": list(s.numeric_fields),
                "incident_when": list(s.incident_when), "sync_scope": s.sync_scope,
                "planned": s.planned}
    if kind is not None:
        spec = KINDS.get(kind)
        return _as_dict(kind, spec) if spec else {}
    return {k: _as_dict(k, s) for k, s in KINDS.items()}


def kinds_in_group(group: str) -> "tuple[str, ...]":
    return tuple(k for k, s in KINDS.items() if s.group == group)

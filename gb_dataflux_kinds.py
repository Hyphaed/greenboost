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
        group="quant", doc="gb_quant.plan_quality()'s per-layer bit-assignment plan (tier, ceiling, precision histogram, bf16-kept count), or gb_quant.preflight_fit()'s side-effect-free component-level byte-count query (dry_run=True, no GPU allocation).",
        fields=("target", "error_ceiling", "precision_histogram", "bf16_kept", "dry_run", "planner"),
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
    "recipe_override": KindSpec(
        group="synapse",
        doc="A serving recipe's MEASURED value was discarded because the caller "
            "passed an explicit one. The precedence is deliberate, but the "
            "override used to be silent, and a measured value dropped without a "
            "word is how a model that fits ends up spilling: live 2026-08-18, "
            "greenboost-cli passed llamacpp_n_ctx=65536 from its config, the "
            "recipe's measured 16384 was dropped, and a model that fits this "
            "card at 16384 served with VRAM at 52.3% and 11.5 GB overflowing to "
            "T2. `rule1_underfilled` caught the RESULT; nothing named the CAUSE. "
            "Carries `field`, `recipe_value` and `caller_value`.",
        incident_when=("warn",)),
    "host_mem_pressure": KindSpec(
        group="health",
        doc="gb_supervisor._RamMonitor observed a host RAM/swap pressure "
            "TRANSITION (ok/warn/critical). Distinct from GreenBoost's own T2 "
            "pool accounting: CPU-offload streaming and any non-GreenBoost "
            "consumer take ordinary system RAM the shim never sees. Exists "
            "because this monitor already computed the state and wrote it only "
            "to the journal and a status file — on 2026-08-18 it went critical "
            "twice (MemAvailable 4044 MB = 6.6% of 61 GB, past its own 8% "
            "floor) about ten minutes before the OOM killer killed a 27 GB "
            "python3 and took the user's terminal with it, and nothing "
            "downstream could see either warning. Pairs with the governed "
            "segment host_oom_imminent. status: ok | warn | error(=critical).",
        incident_when=("warn", "error")),
    "cpu_spillover": KindSpec(
        group="synapse", doc="gb_synapse_backends.py LlamaCppBackend.serve() put some of a "
                             "model's layers on the CPU instead of GPU (dense partial-offload, "
                             "an OOM retry backing off -ngl, or a recipe's nGpuLayers override "
                             "asking for fewer than all layers on a dense model — NemoClaw "
                             "audit Phase 5/7 follow-up, same gate applies regardless of "
                             "whether the reduced layer count came from a live heuristic or a "
                             "YAML file) — the state that used to be invisible to dataflux, "
                             "root cause of the 2026-08-01 6 tok/s incident. status is one of "
                             "dense_partial_offload/vram_fragmentation_oom_retry/"
                             "compute_graph_oom_retry/recipe_dense_partial_offload, all of "
                             "which are the incident itself, not a side detail.",
        fields=("model", "engine"),
        numeric_fields=("weights_gb", "budget_gb", "n_gpu_layers", "n_layers"),
        incident_when=("dense_partial_offload", "vram_fragmentation_oom_retry",
                       "compute_graph_oom_retry", "recipe_dense_partial_offload")),
    "tok_s_measured": KindSpec(
        group="synapse", doc="Real, client-observed decode tokens/sec for one model. "
                             "completion_tokens (generation length) and prompt_tokens "
                             "(KV depth decoded at) are optional and omitted when the "
                             "caller didn't know them — absent is not zero. Without "
                             "both, samples are duration- and depth-blind: a 3-token "
                             "reply and a 600-token generation at opposite ends of the "
                             "context window weigh the same.",
        fields=("model", "completion_tokens", "prompt_tokens", "skip_reason"),
        numeric_fields=("tok_s",)),
    "model_rotation": KindSpec(
        group="synapse", doc="gb_rotator.py overnight rotation phase event.",
        fields=("model", "phase")),
    "prompt_cache": KindSpec(
        group="synapse", doc="Proxy-observed llama.cpp host-memory prompt-cache "
                             "outcome for one request (--cache-reuse/--cache-ram): "
                             "TTFT, the engine's own prefill duration, and the "
                             "reused-vs-total prompt token share at a known "
                             "prompt depth. ttft_ms minus engine_prompt_ms is "
                             "everything that is not prefill (queueing, proxy "
                             "overhead, contention on the box). "
                             "Feeds GB-Semantics' ttft_ms/prompt_cache_hit_pct metrics.",
        fields=("model", "prompt_tokens"),
        numeric_fields=("ttft_ms", "engine_prompt_ms", "hit_pct", "reused_tokens")),
    "kv_quality": KindSpec(
        group="synapse",
        doc="GB-4: what quantizing the KV cache costs, measured rather than "
            "assumed. `recall` is a NIAH score (secret codes planted at spread "
            "depths and retrieved), NOT a smoke-gate pass , that distinction "
            "is the point: quantized WEIGHTS fail by collapsing into "
            "repetition, which a smoke gate catches, while quantized KV stays "
            "fluent and quietly loses long-range retrieval, which only a "
            "recall gate sees. `kv_gb` is what the setting actually costs in "
            "VRAM, so the pair turns \"q8_0 is a budget config\" into a "
            "number: how much VRAM it frees against how much recall it loses.",
        fields=("kv_type", "kv_gb", "ctx", "recall", "found", "needles",
                "niah_tokens"),
        numeric_fields=("kv_gb", "recall", "found", "niah_tokens"),
        sync_scope="host"),
    "spec_decode": KindSpec(
        group="synapse", doc="GB-6: what speculative decoding (the model's own "
                             "MTP draft head) is actually paying at the depth "
                             "being served. `draft_n` is the depth, "
                             "`median_accept_rate` how often the engine kept a "
                             "drafted token, `median_tok_s` the decode rate "
                             "that produced. Depth is NON-monotonic on this "
                             "class of model , deeper drafts get rejected more "
                             "often and net out slower past a sweet spot , so "
                             "the depth is chosen from a sweep, not by picking "
                             "the largest. `accept_rate_source` names why a "
                             "rate is absent (depth 0, or no draft head) "
                             "rather than reporting it as zero.",
        fields=("model", "draft_n", "accept_rate_source"),
        numeric_fields=("median_tok_s", "mean_tok_s", "median_accept_rate", "samples")),
    "cache_index": KindSpec(
        group="synapse", doc="GB-1: which engine slot the proxy routed one "
                             "conversation to, and why. `decision` is "
                             "assigned/reassigned/pinned/pinned-edited; "
                             "`changed_chunk` is the index of the first "
                             "content-addressed chunk that differs from the "
                             "previous turn (null = pure append, which must "
                             "cost nothing), so a prompt_cache hit_pct drop is "
                             "attributable to a specific edit instead of being "
                             "a mystery. `conv` is a 16-char digest of the "
                             "conversation head, never message text.",
        fields=("model", "conv", "slot", "decision"),
        numeric_fields=("chunks", "chunks_before", "changed_chunk", "n_slots", "tracked")),
    "synapse_auth": KindSpec(
        group="synapse", doc="gb_synapse_api.py's proxy (:11369) rejected a "
                             "request with a missing/wrong Bearer token "
                             "(GB_SYNAPSE_TOKEN or /etc/greenboost/synapse_token "
                             "configured) — the audit trail for the auth gate "
                             "that refuses a non-loopback bind with no token. "
                             "Only rejections are emitted; a successful "
                             "authenticated (or loopback, no-auth) request "
                             "emits nothing here, same convention as gb_a2a.py.",
        fields=("path", "peer"), incident_when=("rejected",)),
    "allowlist_trust": KindSpec(
        group="cluster", doc="Trusted root-config-file validation (TCB-U1) before honoring "
                             "kernels.allow, cluster.conf, or synapse_token. decision is "
                             "trusted|rejected_owner|rejected_mode|rejected_symlink|"
                             "rejected_size|absent. Only rejections are emitted — a successful "
                             "trust validation and the routine 'no file configured' absent case "
                             "are both silent. Emitted from gb_synapse.py/gb_synapse_api.py's "
                             "synapse_token resolution (_trust_validate_root_file). The C-side "
                             "netd/netc checks on kernels.allow/cluster.conf fail closed and log "
                             "loudly via netd_log(), but do not yet bridge into this dataflux "
                             "kind — that bridge (tailing shim_stats or a similar mechanism) is "
                             "a follow-up, not done this pass.",
        fields=("path", "decision"), incident_when=(
            "rejected_owner", "rejected_mode", "rejected_symlink", "rejected_size")),
    "recipe_validate": KindSpec(
        group="synapse", doc="gb_synapse.load_recipe()'s lookup of a serving recipe "
                             "(NemoClaw audit, Phase 5e) for a model about to be "
                             "served — status is valid|invalid. A model with no "
                             "matching recipe file at all emits nothing here (no "
                             "news when there's nothing to report, same convention "
                             "as synapse_auth). An invalid recipe still lets serve() "
                             "proceed via the heuristic path, but this event is the "
                             "trace that a recipe existed and was rejected, rather "
                             "than silently ignored.",
        fields=("model", "path", "message"), incident_when=("invalid",)),
    "serve_probe": KindSpec(
        group="synapse", doc="gb_synapse.probe_serve_readiness_for()'s typed "
                             "pre-serve probe suite (NemoClaw audit, Phase 5d) "
                             "result for one step of one serve attempt — the "
                             "closed-set failure reason (see serving/probe.py's "
                             "PROBE_FAILURE_REASONS) instead of a bare crash log "
                             "to interpret by hand. status is ok|<one of the "
                             "closed-set reason strings>.",
        fields=("model", "step", "reason"), incident_when=(
            "gguf_unreadable", "gguf_malformed", "load_timeout", "load_crashed",
            "health_unreachable", "health_unhealthy", "completion_timeout",
            "completion_malformed", "completion_empty", "tool_call_unsupported",
            "tool_call_malformed", "probe_aborted",
        )),

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
        group="shim", doc="Explicit gb_model_tier.py promote/demote/evict call (rare; most placement is shim_decision). evict() may carry KV-cache tier-serde compression fields (missing_features.md item (e)) when the entry was registered with kv_bits.",
        fields=("from_tier", "to_tier", "kv_codec", "kv_skip_reasons"),
        numeric_fields=("size_gb", "disk_mb", "compress_ratio", "kv_bits", "kv_tensors",
                        "kv_tensors_skipped", "kv_ratio")),
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
    "route_decision": KindSpec(
        group="agent",
        doc="gb_router picked a model for one turn. Records the turn class "
            "(fast/deep), the model chosen and WHY, because a routing heuristic "
            "nobody can audit is worse than none , without the reason recorded, "
            "'the wrong model answered that' is unfalsifiable. The spread being "
            "routed across is real: measured 2026-08-18 on one box, MoE with "
            "expert offload 45.72 tok/s vs dense Qwen3.8-27B 2.7-3.5.",
        fields=("turn_class", "model", "reason", "tool_result_chars")),
    "session_audit": KindSpec(
        group="orchestration",
        doc="One gb_session_audit.py run , the record that a session WAS "
            "audited, what it concluded, and when. Exists so 'has anyone "
            "looked at that run, and what did they find' is answerable "
            "without re-deriving the whole audit; the audit itself is "
            "reconstructed from other kinds and stores nothing, so without "
            "this event a completed audit leaves no trace. `codes` carries "
            "the finding codes (cold_prefill_dominates, "
            "decode_below_baseline, cache_cold_repeat, ...) and "
            "worst_severity the highest one, so a trend across sessions is a "
            "query rather than a re-run.",
        fields=("model", "codes", "worst_severity"),
        numeric_fields=("session_started", "session_wall_s", "findings"),
        sync_scope="host"),
    "turn_bench": KindSpec(
        group="agent",
        doc="One gb_bench_turn.py benchmark run , the before/after record every "
            "inference-speed change is measured against. Splits a turn's cost "
            "into real prefill (tokens the engine has never seen) and estimated "
            "recurrent replay (re-running the recurrent layers over the whole "
            "prompt, which this hybrid architecture pays on every turn even at a "
            "99.9% KV hit rate). The replay figure is DERIVED, not read: the "
            "engine reports one prompt_ms and never breaks it down.",
        fields=("model", "n_turns", "median_prompt_ms", "median_replay_ms_est",
                "median_decode_tok_s", "median_new_tokens")),
    "semantic_transition": KindSpec(
        group="health",
        doc="A governed GB-Semantics segment changed verdict (matched/clear/"
            "unknown). Emitted ON TRANSITION ONLY by gb_semantics_watch, so a "
            "condition that stays true costs one event, not one per tick. "
            "Exists because evaluate_segment() is a pull API and nothing asked: "
            "on 2026-08-18 rule1_underfilled was matched while Rule #1's own "
            "text requires a tripwire to fire on exactly that condition, and "
            "weights_dont_fit_vram had been true for a whole session before "
            "anyone looked. `from`/`to` carry the three-valued verdict , "
            "`unknown` is tracked separately from `clear` because a telemetry "
            "failure must never read as a clean bill of health.",
        fields=("segment", "from", "to", "severity"),
        incident_when=("error",)),
    "serve_pause": KindSpec(
        group="health",
        doc="A serve session was paused (KV state saved to disk, engine stopped, "
            "VRAM returned) or resumed (engine re-served, KV state restored). "
            "Exists because an idle session holding most of the card is invisible "
            "otherwise: measured 2026-08-18, ~10.4 GiB of a 12 GB card held at 0% "
            "GPU utilization with nothing to show it. `action` is pause|resume; a "
            "pause carries vram_freed_mb and tokens_saved, a resume carries "
            "reload_s and restore_s so weight-loading cost is never confused with "
            "KV-restore cost.",
        fields=("model", "action", "tokens_saved", "vram_freed_mb",
                "tokens_restored", "reload_s", "restore_s"),
        incident_when=("error",)),

    # ── agent ──────────────────────────────────────────────────────────
    "actuation": KindSpec(
        group="agent", doc="An orchestrator/MCP-actuation lever move (gated/dry-run status included).",
        fields=("lever", "gated")),
    "agent_run": KindSpec(
        group="agent", doc="run_under_greenboost() subprocess lifecycle (started/ok/error).",
        fields=("run_id", "status")),
    "a2a_request": KindSpec(group="agent", doc="A2A JSON-RPC gateway request (verb, gated/dry-run, outcome)."),
    "support_bundle": KindSpec(
        group="agent", doc="Diagnostic support bundle collection result (path, sections collected, feeders queried, redaction status).",
        fields=("path", "sections", "feeders"), incident_when=("error",)),
    "agent_tool_schema_miss": KindSpec(
        group="agent",
        doc="greenboost-cli: the model called a tool by a name it was not "
            "shown (AE-5). outcome is one of unknown_instrument (resolved to "
            "nothing , the strongest hallucinated-tool signal), "
            "rescued_bare_name (a bare MCP tool name the registry recovered "
            "to its mcp__server__tool form), or rescued_other. This is the "
            "measurement behind schema alignment: small models fail tool use "
            "mainly by emitting pretraining-familiar names instead of the "
            "provided schema, and a rename can only be judged against a "
            "before/after rate.",
        fields=("requested", "resolved", "outcome", "known_tools"),
        numeric_fields=("known_tools",),
        incident_when=("error",), sync_scope="host"),
    "agent_memory_recall": KindSpec(
        group="agent",
        doc="greenboost-cli recalled project memory into a turn (AE-9/AE-10). "
            "`rules` counts standing corrections (recalled unconditionally , "
            "the ratchet that stops a mistake recurring), `scoped` counts "
            "memories surfaced because the turn touched their subsystem. "
            "`chars` is what recall spent of the context budget: memory that "
            "grows without bound is memory that crowds out the conversation.",
        fields=("n_items", "chars", "rules", "scoped"),
        numeric_fields=("n_items", "chars", "rules", "scoped"),
        sync_scope="host"),
    "agent_prefix_shift": KindSpec(
        group="agent",
        doc="greenboost-cli's assembled prompt changed somewhere OTHER than "
            "the trailing user turn (GB-1). `first_changed` names the earliest "
            "chunk that moved (system/tools/history/memory/plan) , that chunk "
            "invalidated every token after it, and reuse here is "
            "all-or-nothing because --cache-reuse is rejected for this hybrid "
            "architecture. Correlate with prompt_cache: a shift at `system` or "
            "`tools` should be rare and is worth chasing, since the 14-day "
            "baseline puts a lost prefix at ~166s time-to-first-token against "
            "~5.5s for a kept one. A shift at `user` is normal and is never "
            "emitted.",
        fields=("first_changed", "changed", "stable_prefix_chars",
                "invalidated_chars", "turns"),
        numeric_fields=("stable_prefix_chars", "invalidated_chars", "turns"),
        incident_when=("warn",), sync_scope="host"),
    "agent_context_edit": KindSpec(
        group="agent",
        doc="greenboost-cli edited its conversation context. `op` is either "
            "`microcompact` (cleared stale tool-result BODIES in place , "
            "message count, roles and order all unchanged, so the served "
            "prefix is untouched; `chars_freed`/`results_cleared` say how much "
            "room it bought) or `compact` (rewrote history into a summary , "
            "the expensive path microcompaction exists to postpone). For a "
            "compact (AE-2), "
            "head_kept is the pinned leading messages compaction never moves "
            "(what preserves the engine's reusable KV prefix), "
            "middle_compacted is how many were summarised, extended_prior is "
            "True when the previous memory block's bytes were reproduced and "
            "extended rather than rewritten. Correlate with prompt_cache: a "
            "compaction followed by a hit_pct collapse means the prefix moved "
            "when it should not have.",
        fields=("op", "head_kept", "middle_compacted", "tail_kept",
                "extended_prior", "summary_chars", "results_cleared",
                "results_kept", "chars_freed"),
        numeric_fields=("head_kept", "middle_compacted", "tail_kept",
                        "summary_chars", "results_cleared", "chars_freed"),
        sync_scope="host"),
    "agent_eval_run": KindSpec(
        group="agent",
        doc="greenboost-cli agent benchmark result (AE-1). One event per "
            "run of greenboost_cli.bench.agent_eval, scoring completion, "
            "tool_selection, efficiency and grounding in [0,1]. The "
            "before/after record any claim that a harness change helped has "
            "to be checked against.",
        fields=("overall", "completion", "tool_selection", "efficiency",
                "grounding", "total_tokens", "model"),
        numeric_fields=("overall", "completion", "tool_selection",
                        "efficiency", "grounding", "total_tokens"),
        sync_scope="host"),
    "cli_tool_call": KindSpec(
        group="agent", doc="greenboost-cli instruments/dispatcher.py's per-tool-call "
                            "audit trail (NemoClaw audit, Phase 3): every dispatch() "
                            "call emits one of these, allowed AND denied alike — a "
                            "declined call must leave a trace, same as one that ran. "
                            "decision is one of allowed|denied_policy|denied_user|"
                            "blocked_hook. `args` is captured through "
                            "instruments/capture.py's bounded_capture() (adapted from "
                            "NemoClaw's nemoclaw_observability.py, see "
                            "third_party/nemoclaw_patterns/NOTICE) — depth/size-bounded "
                            "and credential-shaped-value redacted before it ever "
                            "reaches this log.",
        fields=("name", "decision", "agent", "args"), sync_scope="host"),

    # ── eval (model-quality verification gates) ───────────────────────
    "yarn_bake": KindSpec(group="eval", doc="gb_aviary YaRN context-extension bake result."),
    "niah_cert": KindSpec(group="eval", doc="gb_aviary needle-in-a-haystack quality certification result."),
    "smoke_gate": KindSpec(group="eval", doc="gb_aviary repetition-collapse smoke-gate result.",
                                   # 2026-08-18: gained turn_id / duration_ms / outcome. Before that the
        # event said WHAT was called but not which turn it belonged to, how
        # long it took, or whether it worked , so "where did this 14-minute
        # turn go?" could only be answered by hand-reading the engine log.
        # Span-shaped correlation is the idea from NemoClaw's trace.ts
        # (trace_id / parent / duration_ms / status), carried on dataflux
        # rather than a second telemetry system. outcome is
        # ok|transient|semantic|denied, from handlers.classify_tool_failure.
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

    # ── install ────────────────────────────────────────────────────────
    "purge_action": KindSpec(
        group="install", doc="greenboost_setup.sh's do_purge/_rmmod_with_retry/"
                              "cmd_feed-stop PID-ownership proof (NemoClaw audit, "
                              "Phase 2): the decision made about one candidate PID "
                              "before a kill would have happened unconditionally. "
                              "decision is one of stopped|skipped_foreign|"
                              "skipped_no_match|failed — a purge that DECLINES to "
                              "kill something must leave a trace, same as one that "
                              "does, otherwise the next session re-diagnoses 'why "
                              "is the port still bound' from nothing.",
        fields=("target", "pid", "decision"), incident_when=("failed",),
        sync_scope="host"),

    # ── orchestration (advisories and decision-making) ──────────────────
    "advisory": KindSpec(
        group="orchestration",
        doc="Structured advisory notice from gb_pilot, gb_dataflux, gb_mcp, or other orchestration layer (id, severity, phase, title, reason).",
        fields=("advisory_id", "severity", "phase"),
        incident_when=("blocking", "fatal")),
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

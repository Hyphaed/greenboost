#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_mcp_common.py — shared implementations for MCP tools mirrored across
GreenBoost's servers (A5 — fixes F2: six tool names duplicated across
greenboost-orchestrator/greenboost-dataflux/greenboost-cluster/greenboost-
synapse with divergent copies, one of them (greenboost_status) an outright
shape bug — the dataflux copy added `kv_prefetch`, the orchestrator copy
didn't).

Every function here is the ONE canonical implementation for its concept.
Every server's `@mcp.tool` wrapper for a mirrored name calls straight
through — so no matter which server answers `greenboost_status` (or any of
the other five), the shape is identical. This module has NO `@mcp.tool`
decorators itself; it is imported by the servers, never registered as one.

Canonical owner per concept (matches the docstrings on the server-side
wrappers, and greenboost_overview()'s taxonomy):
    greenboost_status       -> greenboost-dataflux (richest: adds kv_prefetch)
    greenboost_capabilities -> greenboost-dataflux
    greenboost_pilot        -> greenboost-dataflux
    synapse_status          -> greenboost-synapse
    dataflux_summary        -> greenboost-dataflux
    cluster_status          -> greenboost-cluster

Do not add new logic to a server-side wrapper — add it here so every mirror
gets it for free, and add a test to tests/test_mcp_shapes.py asserting the
mirrors stay identical.
"""
from __future__ import annotations


def greenboost_status() -> dict:
    """Live read-only GreenBoost snapshot via gb_monitor: whether the kmod is
    loaded, GPU, the T1/T2/T3 tier pool (MB) + combined GB, pressure labels,
    OOM/gaming flags, the shim phase + active allocation path, and the
    Phase-4 KV-prefetch counters (kv_prefetch_* in the raw shim map). Same
    state `greenboost status --llm` reports."""
    try:
        import gb_monitor
        snap = gb_monitor.snapshot()
        d = snap.as_dict()
        d["kv_prefetch"] = {k: v for k, v in snap.shim.items()
                            if k.startswith("kv_prefetch")}
        return d
    except Exception as e:
        return {"error": f"gb_monitor unavailable: {e}"}


def greenboost_capabilities() -> dict:
    """Installed/running shim capability manifest (features the deployed
    GreenBoost actually supports), via gb_monitor's capability manifest
    (runtime /run/greenboost/capabilities.json written by the shim at init ->
    install manifest -> binary sniff). Keys: shim_version, abi, source,
    features{gb_quant_cudart_rebind, expert_pool, cluster_fabric, gds,
    kv_compress, report_physical_vram}."""
    try:
        import gb_monitor
        return gb_monitor.capabilities()
    except Exception as e:
        return {"error": f"gb_monitor unavailable: {e}", "features": {}}


def greenboost_pilot(days: float = 5.0) -> dict:
    """Pilot instrument panel (gb_pilot): per-stage wall-time trends
    (stage_profile events from ai-forge's jobqueue/seedlog), measured tok/s
    per model, latest memory-pressure picture, and evidence-backed advice
    mapping findings to GbControl levers or config changes. Read-only —
    advice is never auto-applied here."""
    try:
        import gb_dataflux
        import gb_pilot
        analysis = gb_pilot.analyze(gb_dataflux.read_events(since_hours=days * 24))
        analysis["advice"] = gb_pilot.advise(analysis)
        return analysis
    except Exception as e:
        return {"error": str(e)}


def synapse_status() -> dict:
    """Gb-Synapse status: engine built (llama-server + rpc-server) + version,
    and whether the server and the :11434 Ollama/OpenAI proxy are running.
    Check before routing pipeline inference through gb-synapse (preferred
    over raw ollama per CLAUDE.md): if engine_built is false, the fallback is
    raw ollama and the fix is `gb_synapse build` (host + each feeder)."""
    try:
        import gb_synapse
        return gb_synapse.status()
    except Exception as e:
        return {"error": f"gb_synapse unavailable: {e}", "engine_built": False}


def dataflux_summary(days: float = 5.0) -> dict:
    """GB-Dataflux headline rollup: events, per-node throughput, errors,
    tok/s, over the last `days` days."""
    try:
        import gb_dataflux
        return gb_dataflux.summarize(gb_dataflux.read_events(since_hours=days * 24))
    except Exception as e:
        return {"error": str(e)}


def cluster_status(probe: bool = True) -> dict:
    """Gb-Cluster overview: whether the shim is installed, whether a cluster
    is configured, how many feeders are online, each node's T1/T2/T3
    free/total MB, host GPU telemetry, and the workload/compute routing
    model. probe=False skips the network round-trip and returns last-known
    feeder state."""
    try:
        import gb_cluster
        return gb_cluster.status(probe=probe)
    except Exception as e:
        return {"error": str(e)}

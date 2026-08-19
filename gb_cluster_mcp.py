#!/usr/bin/env python3
"""gb_cluster_mcp.py — MCP server (FastMCP, stdio) exposing GreenBoost's
**Gb-Cluster** live telemetry read-only, so an LLM (and GreenBoost itself) can
ALWAYS know what is happening at cluster level: which feeders are online, each
node's VRAM / T2 / T3, GPU util / temp / power, measured link bandwidth, and
per-node throughput. Companion to gb_dataflux_mcp.py (which exposes the event
LOG); this one exposes the LIVE cluster STATE. Registered in this repo's
.mcp.json as 'greenboost-cluster'.

Design rule (CLAUDE.md): the cluster is a first-class subsystem, Gb-Cluster, and
its live telemetry must always be queryable — never a black box.
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gb_cluster as gc  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

mcp = FastMCP("greenboost-cluster")

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


@mcp.tool(annotations=_READ_ONLY)
def cluster_status(probe: bool = True) -> dict:
    """Gb-Cluster overview , the cheap first call. Reports whether the shim is
    installed, whether a cluster is configured, how many feeders are online,
    each node's T1/T2/T3 free/total MB, host GPU telemetry, and the
    workload/compute routing model. probe=False skips the network round-trip
    and returns last-known feeder state. Canonical owner (shared impl in
    gb_mcp_common.py — also mirrored on greenboost-orchestrator)."""
    import gb_mcp_common
    return gb_mcp_common.cluster_status(probe=probe)


@mcp.tool(annotations=_READ_ONLY)
def cluster_snapshot(force: bool = False) -> dict:
    """Live per-node telemetry snapshot for the whole cluster (host + every
    feeder): GPU util %, temp C, power W, VRAM/T2/T3 free & total MB, measured
    link MB/s, and throughput items/s. Cached ~1s so a hot poll loop doesn't
    re-probe the fabric every call; pass force=True to re-probe now."""
    return gc.cluster_snapshot(force=force)


@mcp.tool()
def cluster_sync_dataflux(max_lines: int = 2000) -> dict:
    """Pull every online feeder's OWN local dataflux log (its real
    first-person GPU/system telemetry — this can be richer than what the
    fabric's GB_MSG_FEEDER_STATUS wire protocol relays, e.g. sm_clock_mhz/
    mem_clock_mhz/power_limit_w) and merge new events into the HOST's own
    dataflux log, so dataflux_events/dataflux_summary/dataflux_critic (and
    every pipeline consuming them) see the WHOLE cluster's activity, not
    just what the host observed secondhand about a feeder.

    Real gap found 2026-07-14: a feeder's local log (confirmed live:
    thousands of lines, actively growing) was never read by anything on
    the host side — the per-node dataflux stores were completely disjoint.
    Dedup via a persisted per-feeder last-synced timestamp, so repeated
    calls only pull genuinely new events; bounded to a `tail -n max_lines`
    of the remote log per call, not a full pull. Returns
    {hostname_or_ip: n_events_synced}. Call this periodically (ai-forge's
    studio job queue already does, every ~30s) or on demand before a
    dataflux_critic/summary read when you need the freshest cluster-wide
    picture."""
    return gc.sync_cluster_dataflux(max_lines=max_lines)


@mcp.tool(annotations=_READ_ONLY)
def cluster_feeders(probe: bool = True) -> list[dict]:
    """Every configured feeder with its full live state: online flag, ssh
    target, T1/T2/T3 free/total MB, GPU util/temp/power, link + throughput
    EWMA, and the error string when a feeder is offline. probe=False returns
    last-known state without a network hit."""
    return [asdict(f) for f in gc.feeders(probe=probe)]


@mcp.tool()
def cluster_ensure_feeder_ready(feeder_ip: str, confirm: bool = False) -> dict:
    """ACTUATE (double-gated): provision + verify a feeder so a visibly-idle one
    can be put to work , rsyncs the pipeline code, ensures the model, import-
    checks deps. DRY-RUN (returns the plan) unless confirm=True AND the server
    has GB_ORCH_ACTUATE=1. Emits an `actuation` dataflux event when applied.
    The remediation counterpart to seeing an idle feeder in cluster_feeders."""
    import gb_actuation
    return gb_actuation.cluster_ensure_feeder_ready(feeder_ip, confirm=confirm)


@mcp.tool()
def cluster_dispatch(confirm: bool = False) -> dict:
    """ACTUATE (double-gated): prepare the cluster for dispatch , report which
    online feeders are eligible (not busy, link fast enough) and, when applied,
    provision them so the next pipeline fan-out uses them. DRY-RUN unless
    confirm=True AND GB_ORCH_ACTUATE=1. (Item-level dispatch needs the caller's
    own run_local/run_remote callables via gb_cluster.cluster_map; this readies
    the cluster, it does not fabricate work.)"""
    import gb_actuation
    return gb_actuation.cluster_dispatch_plan(confirm=confirm)


@mcp.tool(annotations=_READ_ONLY)
def cluster_available() -> bool:
    """True when at least one feeder is online and reachable right now , the
    fast yes/no gate before deciding to dispatch cluster work."""
    return gc.cluster_available()


@mcp.tool(annotations=_READ_ONLY)
def cluster_topology(force: bool = False) -> dict:
    """Full STATIC hardware topology of every cluster node (host + feeders) ,
    the complement to cluster_snapshot's live telemetry: GPU model/VRAM/compute
    capability, PCIe gen+lanes, CPU P/E-core counts, RAM total/speed/type, NVMe,
    NUMA nodes. Feeder topology arrives over the fabric (GB_MSG_TOPOLOGY) with an
    SSH fallback for older feeders; a node with no reachable topology reports {}.
    Cached per process (topology is static); force=True re-probes. Use it to make
    topology-aware placement/threading/budget decisions per node."""
    return gc.cluster_topology(force=force)


if __name__ == "__main__":
    mcp.run()

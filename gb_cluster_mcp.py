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

mcp = FastMCP("greenboost-cluster")


@mcp.tool()
def cluster_status(probe: bool = True) -> dict:
    """Gb-Cluster overview , the cheap first call. Reports whether the shim is
    installed, whether a cluster is configured, how many feeders are online,
    each node's T1/T2/T3 free/total MB, host GPU telemetry, and the
    workload/compute routing model. probe=False skips the network round-trip
    and returns last-known feeder state."""
    return gc.status(probe=probe)


@mcp.tool()
def cluster_snapshot(force: bool = False) -> dict:
    """Live per-node telemetry snapshot for the whole cluster (host + every
    feeder): GPU util %, temp C, power W, VRAM/T2/T3 free & total MB, measured
    link MB/s, and throughput items/s. Cached ~1s so a hot poll loop doesn't
    re-probe the fabric every call; pass force=True to re-probe now."""
    return gc.cluster_snapshot(force=force)


@mcp.tool()
def cluster_feeders(probe: bool = True) -> list[dict]:
    """Every configured feeder with its full live state: online flag, ssh
    target, T1/T2/T3 free/total MB, GPU util/temp/power, link + throughput
    EWMA, and the error string when a feeder is offline. probe=False returns
    last-known state without a network hit."""
    return [asdict(f) for f in gc.feeders(probe=probe)]


@mcp.tool()
def cluster_available() -> bool:
    """True when at least one feeder is online and reachable right now , the
    fast yes/no gate before deciding to dispatch cluster work."""
    return gc.cluster_available()


if __name__ == "__main__":
    mcp.run()

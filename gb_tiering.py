#!/usr/bin/env python3
"""gb_tiering.py — GB-Tiering: first-class module identity for GreenBoost's
T1/T2/T3 memory tiering subsystem (VRAM → DDR → NVMe as one virtual pool).

The tiering ENGINE is necessarily kernel-level and stays in C: allocation
placement lives in `greenboost_cuda_shim.c` and the pinned T2/T3 pools in
`greenboost.ko`. This module is the Python FACADE that unifies the tiering
surface into ONE first-class entry point — the same relationship `gb_cluster.py`
has to the `greenboost-netd` fabric. It re-exports the tier-move and pool APIs
and adds a live status read, so GB-Tiering reads as a subsystem peer to
GB-Quant / GB-Cluster / GB-Dataflux / GB-Synapse rather than scattered helpers.

Telemetry + MCP (subsystem taxonomy rule): tier moves emit `tier_move` and the
snapshot recorder emits tier_t1/t2/t3 occupancy + pressure to dataflux; live
state is queryable via `tiering_status()` here, the `tiering_status` MCP tool
(greenboost-dataflux), and `greenboost_status`.
"""
from __future__ import annotations

# One import for the whole GB-Tiering surface.
from gb_model_tier import ModelTierManager, Tier   # noqa: F401  tier moves T1<->T2<->T3
from gb_mem_pool import MemPoolManager, PoolStats   # noqa: F401  arena pools


def tiering_status(probe_gpu: bool = True) -> dict:
    """Live GB-Tiering state: T1/T2/T3 pool occupancy (MB) + combined GB,
    pressure labels, shim phase + active allocation path, KV reserve, and
    OOM/gaming flags. Reads the shim/kmod via gb_monitor (same data as
    `greenboost status --llm`). Best-effort; returns {'error': ...} when the
    shim/kmod state isn't available."""
    try:
        import gb_monitor
        return gb_monitor.snapshot(probe_gpu=probe_gpu).as_dict()
    except Exception as e:  # noqa: BLE001
        return {"error": f"gb_tiering: shim/kmod state unavailable: {e}"}


def t2_pool() -> dict:
    """Narrow, typed T2 DDR pool accessor: {total_mb, allocated_mb,
    available_mb, pressure}. The data already lives inside tiering_status()
    (via gb_monitor.snapshot()); this exists because ai-forge had to
    regex-scrape /sys/class/greenboost/greenboost/pool_brief directly in
    two places (forge/gpu.py, forge/runners/longlive.py) for exactly this
    number, duplicating what this module already computes. Returns all-
    zero fields (not an error) when the shim/kmod state is unavailable —
    a T2 pool of size 0 is itself a valid, actionable answer for a caller
    sizing a cgroup MemoryMax against it."""
    status = tiering_status(probe_gpu=False)
    if "error" in status:
        return {"total_mb": 0, "allocated_mb": 0, "available_mb": 0, "pressure": 0}
    return {
        "total_mb": status.get("t2_pool_mb", 0),
        "allocated_mb": status.get("t2_allocated_mb", 0),
        "available_mb": status.get("t2_available_mb", 0),
        "pressure": status.get("t2_pressure", 0),
    }


def manager():
    """The process-wide ModelTierManager singleton if gb_init started one
    (GREENBOOST_ACTIVE=1), else None. Prefer this over constructing your own so
    tier accounting stays single-sourced."""
    try:
        import gb_init
        for attr in ("get_model_tier", "model_tier", "get_tier_manager"):
            obj = getattr(gb_init, attr, None)
            if obj is not None:
                return obj() if callable(obj) else obj
    except Exception:
        pass
    return None


if __name__ == "__main__":
    import json
    import sys
    s = tiering_status()
    print(json.dumps(s) if "--llm" in sys.argv else s)

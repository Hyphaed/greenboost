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
from gb_evict_policy import (  # noqa: F401  prioritized-LRU victim selection
    PRIORITY_DEFAULT, PRIORITY_MAX, PRIORITY_MIN,
    PerTierEvictionController,
)


def host_cold_status() -> dict:
    """Host-side cold-page evidence from the kernel's DAMON_STAT, as
    republished by gb_supervisor.

    This is the host twin of the GPU-side access-counter work: DAMON already
    samples which pages are actually touched, for free, and until now nothing
    read it while tier placement guessed. Returns
    {'available': False, 'reason': ...} when the supervisor has not published
    yet or the kernel lacks CONFIG_DAMON_STAT , a box with no DAMON evidence
    is a valid answer, not an error.

    Observational. Nothing in the tiering path consumes this to make a
    placement decision yet; wiring it into one is a separate, later change.

    Key fields: hot_pct / idle_pct, cold_pct_above_<N>ms for N in
    (200000, 400000, 800000), idle_p50_ms, est_bandwidth_bps. A NEGATIVE idle
    value means actively accessed, not idle , see gb_damon's module docstring.
    """
    try:
        import gb_damon
        return gb_damon.read_published()
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"gb_tiering: DAMON state unavailable: {e}"}


def thrash_status() -> dict:
    """Tier-thrashing counters: components ping-ponging between tiers faster
    than they are used, per gb_thrash (ported from UVM's thrashing detector).

    Observation only , nothing reads this to make a placement decision. The
    `params.lapse_calibrated` field is False on purpose: the detection window
    is a provisional value at module granularity, not a measured one, and any
    reader should weigh the verdict accordingly."""
    try:
        from gb_model_tier import thrash_detector
        return thrash_detector().stats()
    except Exception as e:  # noqa: BLE001
        return {"error": f"gb_tiering: thrash state unavailable: {e}"}


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
    if "--thrash" in sys.argv:
        s = {"tiering": s, "thrash": thrash_status()}
    if "--cold" in sys.argv:
        s = {"tiering": s, "host_cold": host_cold_status()}
    print(json.dumps(s) if "--llm" in sys.argv else s)

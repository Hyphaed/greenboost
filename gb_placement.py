"""gb_placement — fp8-floor cluster-fit planner.

Owner directive: squeeze the whole cluster topology (host GPU + feeder GPUs +
DDR) for the fastest local inference while ALWAYS keeping quality at fp8 or
better. The mechanized rule: prefer *cluster placement* over dropping a model
below fp8. Only fall below fp8 when neither the local tiers nor the cluster can
hold the fp8 footprint.

This module arbitrates across BOTH runtimes and therefore lives on its own
(not in gb_quant, whose planner must stay CPU-only and import-light; not in
gb_synapse, which is GGUF/llama.cpp-only):
  - GGUF / llama.cpp  -> gb_synapse RPC `--tensor-split` across nodes.
  - PyTorch           -> gb_cluster.offload_tail_blocks (validated block split).

It lazy-imports gb_cluster only when it actually needs live feeder telemetry,
so `plan_placement(..., feeder_free_gb=[...])` is a pure function usable in
CPU tests. Everything is gated by GB_PLACEMENT=1 at the call sites; this module
itself is inert until called.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

# Strategy names (also the fp8-floor decision-ladder rungs, best first).
LOCAL_FP8 = "LOCAL_FP8"                    # fits local T1 VRAM at fp8
CLUSTER_SPLIT = "CLUSTER_SPLIT"            # fits aggregate cluster VRAM at fp8
LOCAL_TIERED_FP8 = "LOCAL_TIERED_FP8"      # fits local T1+T2 at fp8 (shim spill)
SUB_FP8_LAST_RESORT = "SUB_FP8_LAST_RESORT"  # nothing holds fp8 -> below fp8


@dataclass
class PlacementPlan:
    strategy: str
    floor_bits: str                        # "fp8" unless SUB_FP8_LAST_RESORT
    tensor_split: Optional[str] = None     # gguf: "h,f0,f1,..." ratio string
    tail_blocks: Optional[int] = None      # torch: blocks for offload_tail_blocks
    aggregate_vram_gb: float = 0.0
    feeders_online: int = 0
    notes: str = ""
    # informational echo of the inputs, for logs / dataflux
    weights_fp8_gb: float = 0.0
    kv_gb: float = 0.0
    local_t1_gb: float = 0.0

    def keeps_fp8(self) -> bool:
        return self.floor_bits == "fp8"


def _workspace_reserve_gb() -> float:
    """T1 held back for per-step compute workspace (GREENBOOST_T1_WORKSPACE_MB)."""
    try:
        return max(0, int(os.environ.get("GREENBOOST_T1_WORKSPACE_MB", "0"))) / 1024.0
    except ValueError:
        return 0.0


def _online_feeder_free_gb(timeout_s: float = 2.0) -> List[float]:
    """Live per-feeder free-VRAM (GiB) for online feeders. [] with no cluster."""
    try:
        import gb_cluster
        return [f.t1_free_mb / 1024.0
                for f in gb_cluster.feeders(probe=True, timeout_s=timeout_s)
                if getattr(f, "online", False) and f.t1_free_mb > 0]
    except Exception:
        return []


def _tensor_split_ratio(local_t1_gb: float, feeder_free_gb: List[float]) -> str:
    """Free-VRAM-proportioned split, host first. Mirrors gb_synapse's raw form;
    gb_synapse.serve computes the authoritative split (KV-aware, host-bias) at
    launch , this is the planner's estimate for logging/recommend."""
    shares = [max(local_t1_gb, 0.1)] + [max(g, 0.1) for g in feeder_free_gb]
    total = sum(shares)
    return ",".join(f"{s / total:.3f}" for s in shares)


def plan_placement(kind: str, *, weights_fp8_gb: float, kv_gb: float,
                   local_t1_gb: float, local_t2_gb: float = 0.0,
                   probe_feeders: bool = True,
                   feeder_free_gb: Optional[List[float]] = None,
                   per_block_fp8_gb: Optional[float] = None,
                   n_blocks: Optional[int] = None) -> PlacementPlan:
    """Decide where an fp8 model should live so quality never drops below fp8.

    kind: "gguf" (llama.cpp RPC split) or "torch" (block offload).
    feeder_free_gb: inject per-feeder free GiB to bypass live probing (tests).
      When None and probe_feeders, live feeders are queried.

    Ladder (first satisfied wins):
      1. fp8 + KV + workspace reserve fit local T1        -> LOCAL_FP8
      2. feeder online and fp8 + KV fit aggregate VRAM    -> CLUSTER_SPLIT
      3. fp8 + KV fit local T1 + T2 (shim spill)          -> LOCAL_TIERED_FP8
      4. otherwise                                        -> SUB_FP8_LAST_RESORT
    """
    if feeder_free_gb is None:
        feeder_free_gb = _online_feeder_free_gb() if probe_feeders else []
    feeders_online = len(feeder_free_gb)
    aggregate_vram_gb = local_t1_gb + sum(feeder_free_gb)
    footprint_gb = weights_fp8_gb + kv_gb

    def _mk(strategy, floor, **kw):
        return PlacementPlan(
            strategy=strategy, floor_bits=floor,
            aggregate_vram_gb=round(aggregate_vram_gb, 2),
            feeders_online=feeders_online,
            weights_fp8_gb=round(weights_fp8_gb, 2), kv_gb=round(kv_gb, 2),
            local_t1_gb=round(local_t1_gb, 2), **kw)

    # 1. Everything in local VRAM at fp8, leaving room for compute workspace.
    if footprint_gb + _workspace_reserve_gb() <= local_t1_gb:
        return _mk(LOCAL_FP8, "fp8", notes="fits local T1 at fp8")

    # 2. Cluster holds fp8 across host + feeder VRAM -> prefer split over quant.
    #    Feeder VRAM (~hundreds of GB/s local read) beats streaming weights over
    #    PCIe from T2, and RPC / block-offload move only per-step activations
    #    across the GbE wire.
    if feeders_online and footprint_gb <= aggregate_vram_gb:
        if kind == "gguf":
            return _mk(CLUSTER_SPLIT, "fp8",
                       tensor_split=_tensor_split_ratio(local_t1_gb, feeder_free_gb),
                       notes="cluster holds fp8 , RPC tensor-split over feeders")
        # torch: offload the tail blocks that overflow local T1.
        deficit_gb = max(0.0, footprint_gb - (local_t1_gb - _workspace_reserve_gb()))
        tail = None
        if per_block_fp8_gb and per_block_fp8_gb > 0:
            tail = int(math.ceil(deficit_gb / per_block_fp8_gb))
            if n_blocks:
                tail = min(tail, n_blocks - 1)  # keep >=1 block local
        return _mk(CLUSTER_SPLIT, "fp8", tail_blocks=tail,
                   notes=("cluster holds fp8 , offload_tail_blocks"
                          + ("" if tail is not None
                             else " (pass per_block_fp8_gb for block count)")))

    # 3. No usable feeder, but fp8 fits local VRAM + DDR (shim spill, still fp8).
    if footprint_gb <= local_t1_gb + local_t2_gb:
        return _mk(LOCAL_TIERED_FP8, "fp8",
                   notes="fits local T1+T2 at fp8 (shim overflow)")

    # 4. Nothing holds fp8 , delegate to the sub-fp8 tiered planner.
    return _mk(SUB_FP8_LAST_RESORT, "sub-fp8",
               notes="fp8 exceeds local tiers AND cluster , tiered plan_fit "
                     "(hot fp8, cold nvfp4/int4) is the fallback")


def proportional_split(total_units: int, weights: List[float],
                       mins: Optional[List[int]] = None) -> List[int]:
    """Apportion `total_units` integer units across len(weights) nodes in
    proportion to `weights`, with an exact-sum guarantee (largest-remainder /
    Hamilton method — port of exo's placement_utils.py). Each node gets
    floor(share), then the leftover units go to the nodes with the largest
    fractional remainder first. `mins[i]`, if given, is a floor for node i
    (e.g. every node gets >=1 layer); satisfied by stealing units from the
    node with the most units. Foundation for the Phase 3 cluster-placement
    tasks (capacity_fit, min_throughput_placement, cluster_map chunk sizing)
    — do not duplicate this arithmetic at those call sites.
    """
    n = len(weights)
    if n == 0 or total_units <= 0:
        return [0] * n
    total_w = sum(w for w in weights if w > 0) or 1.0
    shares = [total_units * max(w, 0.0) / total_w for w in weights]
    result = [int(math.floor(s)) for s in shares]
    remainders = [s - r for s, r in zip(shares, result)]
    leftover = total_units - sum(result)
    order = sorted(range(n), key=lambda i: remainders[i], reverse=True)
    for i in order[:max(0, leftover)]:
        result[i] += 1
    if mins:
        for i, m in enumerate(mins):
            while result[i] < m:
                donor = max(range(n), key=lambda j: result[j])
                if result[donor] <= m:
                    break  # nothing left to steal without going negative
                result[donor] -= 1
                result[i] += 1
    return result


def fits(assignment_gb: List[float], capacity_gb: List[float],
        reserve_gb: Optional[List[float]] = None) -> bool:
    """True iff every node's assignment fits its capacity minus reserve."""
    reserve_gb = reserve_gb or [0.0] * len(capacity_gb)
    return all(a <= c - r for a, c, r in zip(assignment_gb, capacity_gb, reserve_gb))


def plan_and_emit(kind: str, **kw) -> PlacementPlan:
    """plan_placement + best-effort dataflux emit (never raises)."""
    plan = plan_placement(kind, **kw)
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "kind": "placement", "runtime": kind, "strategy": plan.strategy,
            "floor_bits": plan.floor_bits, "feeders_online": plan.feeders_online,
            "aggregate_vram_gb": plan.aggregate_vram_gb,
            "weights_fp8_gb": plan.weights_fp8_gb, "kv_gb": plan.kv_gb,
            "tail_blocks": plan.tail_blocks, "tensor_split": plan.tensor_split,
        })
    except Exception:
        pass
    return plan


def _t2_pool_total_gb() -> float:
    """This node's real T2 DDR pool total in GB (0.0 if unknown). Same
    kmod pool_brief source + MemTotal*0.70 fallback as gb_quant._t2_pool_total_gb
    — duplicated here (not imported) because gb_placement must stay import-
    independent of gb_quant (see module docstring)."""
    try:
        import re as _re
        with open("/sys/class/greenboost/greenboost/pool_brief") as f:
            m = _re.search(r"T2:(\d+)/(\d+)GB", f.read())
        if m:
            return float(int(m.group(2)))
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024.0 * 1024.0) * 0.70, 1)
    except Exception:
        pass
    return 0.0


def plan_experts(*, dense_gb: float, expert_gb: float, kv_gb: float = 0.0,
                 vram_gb: Optional[float] = None, t2_gb: Optional[float] = None,
                 vram_fill_pct: Optional[float] = None) -> dict:
    """Colibri-style versioned tier plan for an MoE model's routed-expert
    weight bytes: how much lives hot in VRAM vs warm in T2 DDR vs stays cold
    (T3/disk), given THIS node's real topology. Read-only — allocates
    nothing, mirrors colibri's resource_plan.build_plan() output shape
    (tiers.{disk,ram,vram}, expected_bottleneck, decisions, warnings) for
    direct comparability, but every budget term is %-derived from
    gb_topology (never colibri's flat 1.2GB/2.5GB literals — see
    workflow/porting-reference.md §CB-3).

    Placement follows the Immutable Design Rules: dense weights + KV are
    resident in T1 FIRST (dense stays entirely in VRAM per the Reference
    Workload Rule's "-ngl 999"; KV is re-read every decode step); only the
    LEFTOVER VRAM after that is the routed-expert hot budget. `vram_gb`/
    `t2_gb` are injectable for tests; None probes gb_topology live.
    """
    try:
        import gb_topology
        topo = gb_topology.get_topology()
    except Exception:
        topo = None

    if vram_gb is None:
        vram_gb = float(topo.vram_gb) if topo and topo.vram_gb > 0 else 0.0
    if t2_gb is None:
        t2_gb = _t2_pool_total_gb()
    fill_pct = vram_fill_pct if vram_fill_pct is not None else 90.0  # Rule #1
    compute_reserve_gb = (gb_topology.compute_reserve_gb(vram_gb * 1024.0)
                          if topo and vram_gb > 0 else max(0.75, 0.08 * vram_gb))

    warnings: List[str] = []
    if vram_gb <= 0:
        warnings.append("VRAM undetectable on this node — plan is a 0-budget floor, not a real estimate")

    vram_for_experts = max(0.0, vram_gb * fill_pct / 100.0 - compute_reserve_gb - kv_gb - dense_gb)
    if dense_gb > vram_gb * fill_pct / 100.0:
        warnings.append(f"dense weights alone ({dense_gb:.1f} GB) exceed the VRAM fill target — "
                        f"KV and every routed expert will overflow")

    hot_gb = min(expert_gb, vram_for_experts)
    ram_for_experts = max(0.0, t2_gb)  # T2 is expert-cache-only here; KV/dense already accounted in T1
    warm_gb = min(max(0.0, expert_gb - hot_gb), ram_for_experts)
    cold_gb = max(0.0, expert_gb - hot_gb - warm_gb)

    if cold_gb > 0:
        bottleneck = "disk/NVMe expert misses (T3) — quality-first rule: T3 is never for speed-critical data"
        warnings.append(f"{cold_gb:.1f} GB of routed experts do not fit VRAM+T2 — "
                        f"raise T2 pool or use --n-cpu-moe / re-quantize before accepting T3 spill")
    elif warm_gb > 0:
        bottleneck = "CPU expert compute + DDR bandwidth (--n-cpu-moe offload)"
    else:
        bottleneck = "GPU compute and interconnect"

    return {
        "version": 1,
        "vram_fill_pct": fill_pct,
        "model": {"dense_gb": round(dense_gb, 2), "expert_gb": round(expert_gb, 2),
                  "kv_gb": round(kv_gb, 2)},
        "tiers": {
            "vram": {"role": "dense+kv+hot-experts", "budget_gb": round(vram_gb, 2),
                     "compute_reserve_gb": round(compute_reserve_gb, 2),
                     "hot_expert_gb": round(hot_gb, 2)},
            "t2": {"role": "warm-experts (--n-cpu-moe)", "budget_gb": round(t2_gb, 2),
                   "warm_expert_gb": round(warm_gb, 2)},
            "t3": {"role": "cold-backing (never speed-critical)", "cold_expert_gb": round(cold_gb, 2)},
        },
        "expected_bottleneck": bottleneck,
        "decisions": [
            {"target": "VRAM", "reason": "dense weights + KV resident first, then hottest routed experts"},
            {"target": "T2 (DDR)", "reason": "warm experts run via --n-cpu-moe, no quality loss"},
            {"target": "T3 (NVMe)", "reason": "last resort only — quality-first rule forbids T3 for speed-critical data"},
        ],
        "warnings": warnings,
    }


def plan_experts_and_emit(**kw) -> dict:
    """plan_experts + best-effort dataflux emit (never raises)."""
    plan = plan_experts(**kw)
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "kind": "placement", "runtime": "expert_plan",
            "vram_fill_pct": plan["vram_fill_pct"],
            "hot_expert_gb": plan["tiers"]["vram"]["hot_expert_gb"],
            "warm_expert_gb": plan["tiers"]["t2"]["warm_expert_gb"],
            "cold_expert_gb": plan["tiers"]["t3"]["cold_expert_gb"],
            "expected_bottleneck": plan["expected_bottleneck"],
        })
    except Exception:
        pass
    return plan


@dataclass
class CapacityFit:
    fraction: float                      # smallest f in (0,1] of free VRAM that fits n_offload layers
    per_device_layers: List[int]         # layer count per device, in the CALLER's original device order
    n_offload: int                       # total layers placed on GPUs (== all layers if everything fits)


def capacity_fit(layer_sizes_gb: List[float], device_free_gb: List[float],
                 device_reserve_gb: Optional[List[float]] = None,
                 order: Optional[List[int]] = None, iters: int = 16) -> CapacityFit:
    """Binary-search the smallest usable-VRAM fraction that still fits the
    maximum number of layers across devices (ollama `llm/server.go`'s
    findBestFit/greedyFit port — see workflow/porting-reference.md §DI-2).

    Why binary-search a fraction instead of just greedily packing at 100%:
    packing at the full free-VRAM figure lets one big device swallow almost
    everything and leaves smaller devices nearly empty; searching for the
    SMALLEST fraction that still achieves the maximum possible layer count
    forces an even balance across devices (a small per-device budget spills
    to the next device sooner), without giving up any offload capacity.

    `order` is the perf-ranked device visitation order (index into
    layer_sizes_gb/device_free_gb/device_reserve_gb) — default: as given.
    Pure function; caller supplies real free-VRAM figures (host + feeders).
    """
    ndev = len(device_free_gb)
    if ndev == 0 or not layer_sizes_gb:
        return CapacityFit(fraction=0.0, per_device_layers=[0] * ndev, n_offload=0)
    reserve = device_reserve_gb if device_reserve_gb is not None else [0.0] * ndev
    order = order if order is not None else list(range(ndev))

    def _greedy(frac: float) -> "tuple[int, List[int]]":
        free = [max(0.0, device_free_gb[order[i]] * frac - reserve[order[i]]) for i in range(ndev)]
        per_device = [0] * ndev
        dev_idx = 0
        remaining = free[0] if ndev else 0.0
        count = 0
        for size in layer_sizes_gb:
            while dev_idx < ndev and size > remaining:
                dev_idx += 1
                remaining = free[dev_idx] if dev_idx < ndev else 0.0
            if dev_idx >= ndev:
                break
            remaining -= size
            per_device[dev_idx] += 1
            count += 1
        return count, per_device

    max_count, max_per_device = _greedy(1.0)
    lo, hi = 0.0, 1.0
    best_frac, best_per_device = 1.0, max_per_device
    for _ in range(max(1, iters)):
        mid = (lo + hi) / 2.0
        count, per_device = _greedy(mid)
        if count >= max_count:
            hi, best_frac, best_per_device = mid, mid, per_device
        else:
            lo = mid

    result = [0] * ndev
    for i, d in enumerate(order):
        result[d] = best_per_device[i]
    return CapacityFit(fraction=round(best_frac, 4), per_device_layers=result, n_offload=max_count)


@dataclass
class NodeThroughput:
    key: str
    compute_tok_s: float     # this node's own decode-speed estimate/measurement
    network_tok_s: float     # 0.0 = no network hop needed (e.g. the host itself)

    @property
    def effective(self) -> float:
        """min(compute, network) — petals' throughput model: a fast GPU behind
        a slow link is still bottlenecked by the link. network_tok_s<=0 means
        no network hop applies (skip the min, e.g. for the local host)."""
        if self.network_tok_s > 0:
            return min(self.compute_tok_s, self.network_tok_s)
        return self.compute_tok_s


def min_throughput_placement(n_units: int, nodes: List[NodeThroughput],
                             mins: Optional[List[int]] = None) -> List[int]:
    """Assign n_units integer units (layers, chunks) across nodes proportional
    to each node's EFFECTIVE throughput (petals' block_selection port — see
    workflow/porting-reference.md §DI-3): the node with the lowest min(compute,
    network) throughput gets proportionally fewer units, so no single slow
    node becomes the pipeline's bottleneck. Delegates the actual integer
    apportionment to proportional_split (exact-sum guarantee)."""
    weights = [max(nd.effective, 0.01) for nd in nodes]  # floor: never fully starve a node
    return proportional_split(n_units, weights, mins=mins)


def should_rebalance(current_min: float, candidate_min: float,
                     balance_quality: float = 0.75) -> bool:
    """Petals' rebalance gate: only worth moving load if the CANDIDATE
    placement's min-throughput clears current_min / balance_quality (i.e.
    improves by more than the (1 - balance_quality) margin — default 0.75,
    petals' own default). Refuses non-improving or disconnecting candidates
    (candidate_min <= 0). This is the guard against moving load for a
    marginal, noise-level gain."""
    if candidate_min <= 0:
        return False
    if current_min <= 0:
        return True   # nothing currently placed usefully — any real throughput improves on that
    return candidate_min > current_min / balance_quality

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

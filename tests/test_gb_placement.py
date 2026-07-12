#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_placement: the fp8-floor cluster-fit planner.

CPU-only. Feeder telemetry is injected via feeder_free_gb, so no test probes a
live cluster. The load-bearing test is the fp8-floor invariant: whenever the
aggregate cluster (or local T1+T2) can hold the fp8 footprint, the plan must
never fall below fp8.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import gb_placement as gp


# ── ladder rung 1: LOCAL_FP8 ─────────────────────────────────────────────────

def test_fits_local_t1(monkeypatch):
    monkeypatch.delenv("GREENBOOST_T1_WORKSPACE_MB", raising=False)
    plan = gp.plan_placement("torch", weights_fp8_gb=6.0, kv_gb=1.0,
                             local_t1_gb=12.0, feeder_free_gb=[])
    assert plan.strategy == gp.LOCAL_FP8
    assert plan.keeps_fp8()


def test_workspace_reserve_pushes_off_local(monkeypatch):
    """A big workspace reserve can make an otherwise-fitting model spill."""
    monkeypatch.setenv("GREENBOOST_T1_WORKSPACE_MB", str(4 * 1024))
    plan = gp.plan_placement("torch", weights_fp8_gb=9.0, kv_gb=0.0,
                             local_t1_gb=12.0, local_t2_gb=40.0,
                             feeder_free_gb=[])
    # 9 + 4 reserve > 12 -> not LOCAL_FP8; fits T1+T2 -> LOCAL_TIERED_FP8
    assert plan.strategy == gp.LOCAL_TIERED_FP8
    assert plan.keeps_fp8()


# ── ladder rung 2: CLUSTER_SPLIT ─────────────────────────────────────────────

def test_cluster_split_gguf(monkeypatch):
    monkeypatch.delenv("GREENBOOST_T1_WORKSPACE_MB", raising=False)
    plan = gp.plan_placement("gguf", weights_fp8_gb=15.0, kv_gb=1.0,
                             local_t1_gb=11.0, local_t2_gb=40.0,
                             feeder_free_gb=[7.0])
    assert plan.strategy == gp.CLUSTER_SPLIT
    assert plan.keeps_fp8()
    assert plan.tensor_split is not None
    # host share first, then feeder; ratios sum ~1.
    parts = [float(x) for x in plan.tensor_split.split(",")]
    assert len(parts) == 2 and abs(sum(parts) - 1.0) < 1e-3
    assert parts[0] > parts[1]           # host has more free VRAM (11 vs 7)


def test_cluster_split_torch_tail_blocks(monkeypatch):
    monkeypatch.delenv("GREENBOOST_T1_WORKSPACE_MB", raising=False)
    # deficit = 15+1 - 11 = 5 GB; per block 1 GB -> 5 tail blocks
    plan = gp.plan_placement("torch", weights_fp8_gb=15.0, kv_gb=1.0,
                             local_t1_gb=11.0, local_t2_gb=40.0,
                             feeder_free_gb=[7.0],
                             per_block_fp8_gb=1.0, n_blocks=40)
    assert plan.strategy == gp.CLUSTER_SPLIT
    assert plan.tail_blocks == 5
    assert plan.keeps_fp8()


def test_cluster_preferred_over_local_tiered(monkeypatch):
    """With a feeder online AND fp8 fitting aggregate VRAM, cluster wins even
    though it would also fit local T1+T2 (feeder VRAM > PCIe T2 streaming)."""
    monkeypatch.delenv("GREENBOOST_T1_WORKSPACE_MB", raising=False)
    plan = gp.plan_placement("gguf", weights_fp8_gb=14.0, kv_gb=1.0,
                             local_t1_gb=11.0, local_t2_gb=40.0,
                             feeder_free_gb=[7.0])
    assert plan.strategy == gp.CLUSTER_SPLIT


# ── ladder rung 3: LOCAL_TIERED_FP8 (no feeder) ──────────────────────────────

def test_local_tiered_when_no_feeder(monkeypatch):
    monkeypatch.delenv("GREENBOOST_T1_WORKSPACE_MB", raising=False)
    plan = gp.plan_placement("torch", weights_fp8_gb=20.0, kv_gb=2.0,
                             local_t1_gb=11.0, local_t2_gb=40.0,
                             feeder_free_gb=[])
    assert plan.strategy == gp.LOCAL_TIERED_FP8
    assert plan.keeps_fp8()


def test_degrades_gracefully_without_feeder(monkeypatch):
    """No feeder: ladder skips rung 2, identical to a pure-local decision."""
    monkeypatch.delenv("GREENBOOST_T1_WORKSPACE_MB", raising=False)
    with_feeder = gp.plan_placement("gguf", weights_fp8_gb=6.0, kv_gb=1.0,
                                    local_t1_gb=12.0, feeder_free_gb=[7.0])
    no_feeder = gp.plan_placement("gguf", weights_fp8_gb=6.0, kv_gb=1.0,
                                  local_t1_gb=12.0, feeder_free_gb=[])
    assert with_feeder.strategy == no_feeder.strategy == gp.LOCAL_FP8


# ── ladder rung 4: SUB_FP8_LAST_RESORT ───────────────────────────────────────

def test_sub_fp8_only_when_nothing_holds_fp8(monkeypatch):
    monkeypatch.delenv("GREENBOOST_T1_WORKSPACE_MB", raising=False)
    plan = gp.plan_placement("torch", weights_fp8_gb=80.0, kv_gb=4.0,
                             local_t1_gb=11.0, local_t2_gb=40.0,
                             feeder_free_gb=[7.0])
    assert plan.strategy == gp.SUB_FP8_LAST_RESORT
    assert not plan.keeps_fp8()


# ── the fp8-floor invariant ──────────────────────────────────────────────────

@pytest.mark.parametrize("weights,kv,t1,t2,feeders", [
    (14.0, 1.0, 11.0, 40.0, [7.0]),
    (15.0, 2.0, 12.0, 0.0, [8.0]),
    (30.0, 3.0, 11.0, 40.0, []),
    (18.0, 1.0, 11.0, 8.0, [9.0, 9.0]),
    (6.0, 0.5, 12.0, 40.0, []),
])
def test_fp8_floor_invariant(monkeypatch, weights, kv, t1, t2, feeders):
    """If aggregate VRAM OR local T1+T2 can hold the fp8 footprint, the plan
    must keep fp8. Only a genuinely-too-big model may go sub-fp8."""
    monkeypatch.delenv("GREENBOOST_T1_WORKSPACE_MB", raising=False)
    footprint = weights + kv
    aggregate = t1 + sum(feeders)
    for kind in ("gguf", "torch"):
        plan = gp.plan_placement(kind, weights_fp8_gb=weights, kv_gb=kv,
                                 local_t1_gb=t1, local_t2_gb=t2,
                                 feeder_free_gb=feeders,
                                 per_block_fp8_gb=1.0, n_blocks=100)
        if footprint <= aggregate or footprint <= t1 + t2:
            assert plan.keeps_fp8(), (kind, plan.strategy)
        else:
            assert plan.strategy == gp.SUB_FP8_LAST_RESORT

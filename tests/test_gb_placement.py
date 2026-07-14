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


# ── proportional_split / fits: largest-remainder apportionment (DI-1) ───────

def test_proportional_split_exact_sum_equal_weights():
    result = gp.proportional_split(10, [1, 1, 1])
    assert sum(result) == 10


def test_proportional_split_exact_sum_adversarial_remainders():
    # weights chosen so floor(share) undershoots by more than one unit
    result = gp.proportional_split(7, [0.33, 0.33, 0.34])
    assert sum(result) == 7


def test_proportional_split_exact_sum_uneven_weights():
    result = gp.proportional_split(48, [7.5, 11.2, 3.1])
    assert sum(result) == 48


def test_proportional_split_zero_total_units():
    assert gp.proportional_split(0, [1, 2, 3]) == [0, 0, 0]


def test_proportional_split_empty_weights():
    assert gp.proportional_split(5, []) == []


def test_proportional_split_respects_mins():
    result = gp.proportional_split(3, [10, 1, 1], mins=[1, 1, 1])
    assert sum(result) == 3
    assert all(r >= 1 for r in result)


def test_fits_true_within_capacity():
    assert gp.fits([1.0, 2.0], [4.0, 4.0]) is True


def test_fits_false_over_capacity():
    assert gp.fits([1.0, 5.0], [4.0, 4.0]) is False


def test_fits_honors_reserve():
    assert gp.fits([3.0], [4.0], reserve_gb=[2.0]) is False


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


# ── plan_experts: colibri-style tier-plan JSON (CB-3) ────────────────────────

def test_plan_experts_all_hot_when_it_all_fits():
    p = gp.plan_experts(dense_gb=2.0, expert_gb=5.0, kv_gb=1.0, vram_gb=12.0, t2_gb=40.0)
    assert p["tiers"]["vram"]["hot_expert_gb"] == 5.0
    assert p["tiers"]["t2"]["warm_expert_gb"] == 0.0
    assert p["tiers"]["t3"]["cold_expert_gb"] == 0.0
    assert p["expected_bottleneck"] == "GPU compute and interconnect"
    assert p["warnings"] == []


def test_plan_experts_overflows_to_t2_when_experts_dont_fit_vram():
    p = gp.plan_experts(dense_gb=10.0, expert_gb=18.6, kv_gb=1.0, vram_gb=12.0, t2_gb=40.0)
    assert p["tiers"]["vram"]["hot_expert_gb"] == 0.0
    assert p["tiers"]["t2"]["warm_expert_gb"] == 18.6
    assert p["tiers"]["t3"]["cold_expert_gb"] == 0.0
    assert "CPU expert compute" in p["expected_bottleneck"]


def test_plan_experts_dense_and_kv_are_resident_first():
    # dense+kv alone exceed the vram fill target -> zero room for experts,
    # even though expert_gb would fit VRAM on its own.
    p = gp.plan_experts(dense_gb=11.0, expert_gb=1.0, kv_gb=1.0, vram_gb=12.0, t2_gb=40.0)
    assert p["tiers"]["vram"]["hot_expert_gb"] == 0.0
    assert any("exceed the VRAM fill target" in w for w in p["warnings"])


def test_plan_experts_cold_spill_when_t2_also_full():
    p = gp.plan_experts(dense_gb=1.0, expert_gb=50.0, kv_gb=0.0, vram_gb=12.0, t2_gb=8.0)
    assert p["tiers"]["t3"]["cold_expert_gb"] > 0.0
    assert "T3" in p["expected_bottleneck"] or "disk" in p["expected_bottleneck"].lower()
    assert any("T3 spill" in w or "do not fit" in w for w in p["warnings"])


def test_plan_experts_undetectable_vram_warns_and_zeroes_budget():
    p = gp.plan_experts(dense_gb=2.0, expert_gb=5.0, kv_gb=1.0, vram_gb=0.0, t2_gb=40.0)
    assert p["tiers"]["vram"]["budget_gb"] == 0.0
    assert any("undetectable" in w for w in p["warnings"])


def test_plan_experts_and_emit_never_raises_without_dataflux():
    # Should not raise even if gb_dataflux import/emit fails silently.
    plan = gp.plan_experts_and_emit(dense_gb=2.0, expert_gb=5.0, kv_gb=1.0,
                                    vram_gb=12.0, t2_gb=40.0)
    assert plan["tiers"]["vram"]["hot_expert_gb"] == 5.0


# ── capacity_fit: binary-search layer packing (ollama port, DI-2) ───────────

def test_capacity_fit_everything_fits_finds_minimal_balanced_fraction():
    cf = gp.capacity_fit([1.0] * 5, [10.0, 10.0])
    assert cf.n_offload == 5
    assert sum(cf.per_device_layers) == 5
    assert cf.fraction < 1.0  # doesn't need full capacity when it all fits easily


def test_capacity_fit_partial_offload_when_undersized():
    cf = gp.capacity_fit([1.0] * 10, [6.0, 3.0])
    assert cf.n_offload == 9   # only 9 GB of capacity across both devices
    assert sum(cf.per_device_layers) == 9
    assert cf.fraction == 1.0  # needs full capacity and still can't fit all 10


def test_capacity_fit_empty_devices_returns_zero():
    cf = gp.capacity_fit([1.0, 1.0], [])
    assert cf.n_offload == 0
    assert cf.per_device_layers == []


def test_capacity_fit_respects_device_reserve():
    # 10 GB free but 9 GB reserved leaves only 1 GB usable -> 1 layer.
    cf = gp.capacity_fit([1.0] * 5, [10.0], device_reserve_gb=[9.0])
    assert cf.n_offload == 1


# ── min_throughput_placement / should_rebalance (petals port, DI-3) ─────────

def test_min_throughput_placement_penalizes_slow_link():
    nodes = [gp.NodeThroughput("host", compute_tok_s=50, network_tok_s=0),
             gp.NodeThroughput("feeder", compute_tok_s=50, network_tok_s=5)]
    split = gp.min_throughput_placement(20, nodes)
    assert sum(split) == 20
    assert split[0] > split[1]   # host (no network bottleneck) gets more units


def test_min_throughput_placement_equal_nodes_split_evenly():
    nodes = [gp.NodeThroughput("a", compute_tok_s=10, network_tok_s=0),
             gp.NodeThroughput("b", compute_tok_s=10, network_tok_s=0)]
    split = gp.min_throughput_placement(10, nodes)
    assert split == [5, 5]


def test_node_throughput_effective_is_min_of_compute_and_network():
    nd = gp.NodeThroughput("x", compute_tok_s=100, network_tok_s=5)
    assert nd.effective == 5
    nd2 = gp.NodeThroughput("x", compute_tok_s=100, network_tok_s=0)
    assert nd2.effective == 100   # no network hop -> compute-bound only


def test_should_rebalance_clear_win():
    assert gp.should_rebalance(10, 20, balance_quality=0.75) is True


def test_should_rebalance_rejects_marginal_gain():
    assert gp.should_rebalance(10, 12, balance_quality=0.75) is False


def test_should_rebalance_rejects_disconnecting_candidate():
    assert gp.should_rebalance(10, 0, balance_quality=0.75) is False


def test_should_rebalance_accepts_any_gain_from_zero_baseline():
    assert gp.should_rebalance(0, 5, balance_quality=0.75) is True

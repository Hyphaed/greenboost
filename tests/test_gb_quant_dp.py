#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_quant_dp.py — the opt-in DP budget-optimal mixed-precision
planner (missing_features.md item (c)).

Pure Python, no torch/CUDA/GemLite required , the DP itself operates on
plain dicts (layer sizes, sensitivity, byte-per-param cost table), so every
test here hand-crafts small, brute-forceable inputs and checks the DP finds
the true budget-constrained optimum.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import gb_quant_dp


# Simple, clean cost table for hand-verifiable arithmetic , NOT gb_quant's
# real _BYTES_PER_PARAM (tests that need the real table import it directly).
# Values are bytes/param (not GB/param) , plan_bits_dp divides by 2**30
# itself, so realistic param counts (millions) below are what make the
# resulting cost units land in a sensibly-testable range.
_BPP = {16: 2.0, 8: 1.0, 4: 0.5}


# ── choose_bits_per_layer_with_path (the ported DP core) ──────────────────

def test_dp_core_picks_min_loss_within_budget():
    """Two independent layers, one option each fits, DP must pick the
    combination with the lowest total loss among all budget-feasible ones."""
    layers = {
        "a": [(16, 10, 0.0, ("a",)), (8, 5, 0.1, ("a",)), (4, 2, 0.5, ("a",))],
        "b": [(16, 10, 0.0, ("b",)), (8, 5, 0.2, ("b",)), (4, 2, 0.9, ("b",))],
    }
    # Budget 10: only combos summing to <=10 units.
    # a@4(2)+b@4(2)=4 loss=1.4 | a@8(5)+b@4(2)=7 loss=0.6 | a@4(2)+b@8(5)=7 loss=0.7
    # a@8(5)+b@8(5)=10 loss=0.3  <- best
    min_loss, path = gb_quant_dp.choose_bits_per_layer_with_path(layers, P=10)
    assert min_loss == pytest.approx(0.3)
    picked = dict((n, s) for names, s in path for n in names)
    assert picked == {"a": 8, "b": 8}


def test_dp_core_infeasible_returns_none():
    layers = {"a": [(16, 100, 0.0, ("a",))]}
    min_loss, path = gb_quant_dp.choose_bits_per_layer_with_path(layers, P=5)
    assert min_loss is None
    assert path is None


def test_dp_core_beam_width_still_finds_global_optimum_small_space():
    """max_states=1 forces aggressive pruning to a single state per layer,
    but the true optimum survives when it strictly dominates (lowest loss AT
    the tightest feasible params) after Pareto pruning."""
    layers = {
        "a": [(16, 1, 0.0, ("a",))],   # only one option , no ambiguity
    }
    min_loss, path = gb_quant_dp.choose_bits_per_layer_with_path(
        layers, P=10, max_states=1)
    assert min_loss == 0.0
    assert path == [(("a",), 16)]


def test_default_beam_width_scales_down_with_layer_count():
    small = gb_quant_dp._default_beam_width(4)
    large = gb_quant_dp._default_beam_width(4000)
    assert small >= large
    assert large >= 256   # floor from the implementation


def test_default_beam_width_env_override(monkeypatch):
    monkeypatch.setenv("GB_QUANT_DP_MAX_STATES", "7")
    assert gb_quant_dp._default_beam_width(100) == 7


def test_default_beam_width_ignores_invalid_env_override(monkeypatch):
    monkeypatch.setenv("GB_QUANT_DP_MAX_STATES", "not-a-number")
    # Falls back to the formula instead of raising.
    assert gb_quant_dp._default_beam_width(100) > 0


# ── plan_bits_dp (the layer-sensitivity -> bits assignment wrapper) ───────
#
# plan_bits_dp converts (n_params, bytes_per_param) -> GB via /2**30, then to
# integer cost UNITS via *1024 (see _UNITS_PER_GB). Realistic param counts
# (millions) are used below so this conversion doesn't round tiny toy sizes
# down to 0 units for every precision (which would make every option "free"
# and defeat the point of the test). With _BPP={16:2.0,8:1.0,4:0.5} bytes/param:
#   big   (n=100_000_000): bf16=191 units, int8=95 units, int4=48 units
#   small (n=  2_000_000): bf16=  4 units, int8= 2 units, int4= 1 unit
# (verified against the function's own round(gb*1024) arithmetic, not hand
# rounded , see the docstring of each test for the exact combo it targets.)

_BIG = 100_000_000
_SMALL = 2_000_000


def test_plan_bits_dp_prefers_bf16_when_budget_is_generous():
    layer_sizes = {"big": _BIG, "small": _SMALL}
    sensitivity = {"big": {8: 0.01, 4: 0.05}, "small": {8: 0.02, 4: 0.80}}
    # 200 units comfortably covers bf16+bf16 (191+4=195); zero-loss wins.
    result = gb_quant_dp.plan_bits_dp(
        sensitivity, layer_sizes, _BPP, budget_gb=200 / 1024,
        candidates=(16, 8, 4))
    assert result["feasible"]
    assert result["per_layer_bits"] == {"big": 16, "small": 16}
    assert result["total_loss"] == 0.0


def test_plan_bits_dp_trades_low_sensitivity_layer_down_for_budget():
    """The headline case item (c) exists for: at budget=100 units, no
    both-bf16 combo fits (195 units), but 'big' (int8 loss only 0.01, i.e.
    low-sensitivity) can drop to int8 (95 units) and still leave room for
    'small' (high-sensitivity: int4 loss 0.80) to stay at its safest bf16
    (95+4=99 <= 100). Total loss 0.01 beats every alternative combo that
    also fits (e.g. both-int8 at 97 units, loss 0.03) , DP finds this
    because it searches ALL feasible combos for min loss, not a per-layer
    ceiling walk that would treat each layer independently."""
    layer_sizes = {"big": _BIG, "small": _SMALL}
    sensitivity = {"big": {8: 0.01, 4: 0.05}, "small": {8: 0.02, 4: 0.80}}
    result = gb_quant_dp.plan_bits_dp(
        sensitivity, layer_sizes, _BPP, budget_gb=100 / 1024,
        candidates=(16, 8, 4))
    assert result["feasible"]
    assert result["per_layer_bits"] == {"big": 8, "small": 16}
    assert abs(result["total_loss"] - 0.01) < 1e-9


def test_plan_bits_dp_infeasible_when_even_the_floor_exceeds_budget():
    layer_sizes = {"big": _BIG}
    sensitivity = {"big": {4: 0.5}}
    # int4 alone costs 48 units; budget of 1 unit can't hold it.
    result = gb_quant_dp.plan_bits_dp(
        sensitivity, layer_sizes, _BPP, budget_gb=1 / 1024,
        candidates=(16, 8, 4))
    assert result["feasible"] is False
    assert result["per_layer_bits"] == {}
    assert result["total_loss"] is None


def test_plan_bits_dp_excluded_bits_removed_per_layer():
    """excluded mirrors the group_size int4-divisibility fallback , at a
    budget where int4 (48 units) is the ONLY option that fits (bf16=191,
    int8=95 both exceed 60), excluding bits=4 must turn a feasible plan
    into an infeasible one rather than silently picking int8 anyway."""
    layer_sizes = {"a": _BIG}
    sensitivity = {"a": {4: 0.001}}
    unrestricted = gb_quant_dp.plan_bits_dp(
        sensitivity, layer_sizes, _BPP, budget_gb=60 / 1024,
        candidates=(16, 8, 4))
    assert unrestricted["feasible"]
    assert unrestricted["per_layer_bits"]["a"] == 4

    restricted = gb_quant_dp.plan_bits_dp(
        sensitivity, layer_sizes, _BPP, budget_gb=60 / 1024,
        candidates=(16, 8, 4), excluded={"a": (4,)})
    assert restricted["feasible"] is False


def test_plan_bits_dp_missing_sensitivity_entry_costs_worst_case():
    """A candidate bits value absent from a layer's sensitivity dict costs
    loss=1.0 (mirrors plan_quality's greedy `layer_s.get(candidate, 1.0)`),
    so DP still terminates and treats it as maximally risky, not free."""
    layer_sizes = {"a": _BIG}
    sensitivity = {"a": {}}   # no calibration data at all
    result = gb_quant_dp.plan_bits_dp(
        sensitivity, layer_sizes, _BPP, budget_gb=1.0,
        candidates=(16, 8, 4))
    assert result["feasible"]
    # bf16 (loss 0.0) strictly beats any uncalibrated non-16 option (loss 1.0).
    assert result["per_layer_bits"]["a"] == 16


def test_plan_bits_dp_empty_layer_sizes_returns_infeasible_not_crash():
    result = gb_quant_dp.plan_bits_dp({}, {}, _BPP, budget_gb=10.0)
    assert result["feasible"] is False
    assert result["per_layer_bits"] == {}


def test_plan_bits_dp_total_gb_matches_assigned_bits():
    layer_sizes = {"a": _BIG}
    sensitivity = {"a": {8: 0.01}}   # no entry for 4 -> defaults to worst-case
    # 100 units: bf16 (191) doesn't fit, int8 (95) does -> forced to int8.
    result = gb_quant_dp.plan_bits_dp(
        sensitivity, layer_sizes, _BPP, budget_gb=100 / 1024,
        candidates=(16, 8, 4))
    assert result["feasible"]
    bits = result["per_layer_bits"]["a"]
    assert bits == 8
    expected_gb = round(layer_sizes["a"] * _BPP[bits] / 2 ** 30, 3)
    assert result["total_gb"] == expected_gb

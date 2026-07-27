#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_quant.plan_quality() and its supporting pieces (GpuProfile.
calibrated_precisions, QualityFitReport, _auto_budgets) — the P3 fix (plan
`bring-gb-synapse-gb-quant-and-async-nygaard.md`).

The verified bug: plan_quality's per-layer ladder walk iterated
`profile.precisions` in its stored highest-quality-first order and broke on
the FIRST candidate meeting the error ceiling, so it always picked the
highest-quality (lowest-error) precision — the opposite of the function's
own docstring ("select the LOWEST precision where rel_err <= error_ceil").
On Blackwell, fp8 is always first and almost always under both the 3% and
8% ceilings, so "balanced" (8% ceiling) was byte-identical to
"near_lossless" (3%) — dead code. Zero tests existed for plan_quality before
this file (P5.1-class test debt named directly in the plan).

No GemLite, no CUDA, no GPU hardware required , sensitivity is passed in
directly (bypasses gb_quant_calib entirely), same pattern the rest of this
suite (test_gb_quant.py) uses for gpu_profile()-shaped fakes.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn

import gb_quant


def _profile(precisions=(16, "fp8", 8, 4, "tq3", "tq2"),
            floor_default=4, quality_default="fp8"):
    return gb_quant.GpuProfile(
        family="blackwell", cc=(12, 0), precisions=precisions,
        floor_default=floor_default, quality_default=quality_default,
        t2_tolerance_gb=10.0,
    )


class _OneLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(64, 64, bias=False)


# ── GpuProfile.calibrated_precisions ──────────────────────────────────────

def test_calibrated_precisions_auto_derived_drops_tq_and_nvfp4():
    p = gb_quant.GpuProfile(
        family="blackwell", cc=(12, 0),
        precisions=(16, "fp8", "nvfp4", 8, 4, "tq3", "tq2"),
        floor_default=4, quality_default="fp8", t2_tolerance_gb=1.0,
    )
    assert p.calibrated_precisions == (16, "fp8", 8, 4)


def test_calibrated_precisions_explicit_override_kept():
    p = gb_quant.GpuProfile(
        family="blackwell", cc=(12, 0), precisions=(16, "fp8", 8, 4),
        floor_default=4, quality_default="fp8", t2_tolerance_gb=1.0,
        calibrated_precisions=(16, "fp8"),
    )
    assert p.calibrated_precisions == (16, "fp8")


# ── plan_quality: the core ladder-walk regression ─────────────────────────

def test_near_lossless_and_balanced_produce_different_histograms():
    """The headline bug: near_lossless (3% ceiling) and balanced (8%
    ceiling) must pick genuinely different precisions for a layer whose
    fp8 error is under 3% but whose int4 error is between 3% and 8%."""
    module = _OneLayer()
    sensitivity = {"a": {"fp8": 0.025, 8: 0.04, 4: 0.06}}
    profile = _profile()

    near_lossless = gb_quant.plan_quality(
        module, target="near_lossless", sensitivity=sensitivity,
        profile=profile, t1_budget_gb=100.0, t2_budget_gb=100.0, verbose=False)
    balanced = gb_quant.plan_quality(
        module, target="balanced", sensitivity=sensitivity,
        profile=profile, t1_budget_gb=100.0, t2_budget_gb=100.0, verbose=False)

    assert near_lossless.per_layer_bits["a"] == "fp8"
    # balanced's looser ceiling must pick the MOST compressed option that
    # still qualifies (int4, the lowest precision meeting the 8% ceiling) ,
    # not silently re-pick fp8 like the old highest-quality-first walk did.
    assert balanced.per_layer_bits["a"] == 4
    assert near_lossless.per_layer_bits != balanced.per_layer_bits


def test_ladder_walk_picks_most_compressed_precision_meeting_ceiling():
    """Direct check of "lowest precision where rel_err <= ceiling": with
    int8 ALSO under the ceiling, the walk must still prefer int4 (more
    compressed) over int8, not stop at the first (highest-quality) hit."""
    module = _OneLayer()
    sensitivity = {"a": {"fp8": 0.01, 8: 0.02, 4: 0.03}}
    profile = _profile()

    report = gb_quant.plan_quality(
        module, target="near_lossless", sensitivity=sensitivity,  # ceiling 3.0%
        profile=profile, t1_budget_gb=100.0, t2_budget_gb=100.0, verbose=False)

    assert report.per_layer_bits["a"] == 4


def test_ladder_walk_never_implicitly_picks_tq_precisions():
    """tq2/tq3 calibration is a plain per-row absmax approximation (measured
    ~6x the near_lossless ceiling's real error in the real kernel) , even
    a suspiciously-good calibrated number for tq2 must never be picked by
    the automatic walk. calibrated_precisions excludes it categorically."""
    module = _OneLayer()
    sensitivity = {"a": {"tq2": 0.0001, "tq3": 0.0002, "fp8": 0.025, 8: 0.5, 4: 0.5}}
    profile = _profile()

    report = gb_quant.plan_quality(
        module, target="near_lossless", sensitivity=sensitivity,
        profile=profile, t1_budget_gb=100.0, t2_budget_gb=100.0, verbose=False)

    assert report.per_layer_bits["a"] == "fp8"


def test_no_precision_meets_ceiling_keeps_bf16():
    module = _OneLayer()
    sensitivity = {"a": {"fp8": 0.5, 8: 0.5, 4: 0.5}}
    profile = _profile()

    report = gb_quant.plan_quality(
        module, target="near_lossless", sensitivity=sensitivity,
        profile=profile, t1_budget_gb=100.0, t2_budget_gb=100.0, verbose=False)

    assert report.per_layer_bits["a"] == 16
    assert report.precision_histogram.get("bf16", 0) == 1


# ── error accounting (mean_rel_err must reflect what actually happened) ───

def test_compact_tier_reports_real_error_not_zero():
    """Previously, compact/no-sensitivity-branch layers were excluded from
    layer_errs_used entirely, so an all-compact plan reported
    mean_rel_err=0.00000 as if lossless. With real calibration data present
    for the assigned (floor_default) bits, it must report the real number."""
    module = _OneLayer()
    sensitivity = {"a": {4: 0.12}}
    profile = _profile(floor_default=4)

    report = gb_quant.plan_quality(
        module, target="compact", sensitivity=sensitivity,
        profile=profile, t1_budget_gb=100.0, t2_budget_gb=100.0, verbose=False)

    assert report.per_layer_bits["a"] == 4
    assert report.mean_rel_err == 0.12
    assert report.max_rel_err == 0.12


def test_compact_tier_without_calibration_data_reports_zero_honestly():
    """No sensitivity entry at all for this layer/bits combo , nothing to
    report, so it's correctly excluded (never fabricated), not a bug."""
    module = _OneLayer()
    profile = _profile(floor_default=4)

    report = gb_quant.plan_quality(
        module, target="compact", sensitivity={},
        profile=profile, t1_budget_gb=100.0, t2_budget_gb=100.0, verbose=False)

    assert report.per_layer_bits["a"] == 4
    assert report.mean_rel_err == 0.0


def test_no_sensitivity_branch_uses_quality_default_and_is_unaccounted():
    """A layer plan_quality has no calibration data for at all (not in the
    sensitivity dict, non-compact target) falls back to quality_default and
    correctly contributes no error number , there is nothing calibrated to
    report."""
    module = _OneLayer()
    profile = _profile(quality_default="fp8")

    report = gb_quant.plan_quality(
        module, target="near_lossless", sensitivity={},
        profile=profile, t1_budget_gb=100.0, t2_budget_gb=100.0, verbose=False)

    assert report.per_layer_bits["a"] == "fp8"
    assert report.mean_rel_err == 0.0


# ── group_size fallback still works with the reversed walk ───────────────

def test_group_size_fallback_bumps_int4_to_next_precision():
    """in_features not divisible by group_size must fall back to the next-
    higher precision in profile.precisions (not the calibrated subset) ,
    unaffected by the ladder-walk direction fix."""
    class _Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(96, 32, bias=False)  # 96 not divisible by 64

    sensitivity = {"a": {"fp8": 0.01, 8: 0.02, 4: 0.03}}
    profile = _profile()

    report = gb_quant.plan_quality(
        _Odd(), target="near_lossless", sensitivity=sensitivity,
        profile=profile, group_size=64, t1_budget_gb=100.0, t2_budget_gb=100.0,
        verbose=False)

    # Would have picked int4 (see test above) but group_size=64 doesn't
    # divide in_features=96 , must bump to the next-higher entry (int8).
    assert report.per_layer_bits["a"] == 8


# ── _auto_budgets(): loud 2 GiB floor when detection genuinely fails ──────

def test_auto_budgets_falls_back_to_loud_2gib_floor(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda: (_ for _ in ()).throw(RuntimeError("no cuda")))
    # No /sys/class/greenboost/greenboost/pool_brief on this box in test mode,
    # and gb_topology unavailable , force both live probes to genuinely fail.
    with patch("builtins.open", side_effect=FileNotFoundError), \
         patch.dict(sys.modules, {"gb_topology": None}):
        t1, t2 = gb_quant._auto_budgets()
    assert t1 == 2.0


# ── normalize_bits_token() (shared by set_quant_policy + quant_cmds) ──────

@pytest.mark.parametrize("token,expected", [
    ("int4", 4), ("4", 4),
    ("int8", 8), ("8", 8),
    ("fp8", "fp8"), ("e4m3", "fp8"), ("FP8", "fp8"),
    ("nvfp4", "nvfp4"),
    ("tq3", "tq3"), ("3", "tq3"),
    ("tq2", "tq2"), ("2", "tq2"),
    ("bf16", 16),
    ("auto", "auto"),
    ("garbage-token", "fp8"),
])
def test_normalize_bits_token(token, expected):
    assert gb_quant.normalize_bits_token(token) == expected


def test_normalize_bits_token_bf16_is_the_real_passthrough_sentinel():
    """The actual bug this closes: quant_cmds' old copy mapped "bf16" to the
    STRING "bf16", which quantize_module's `if bits == 16: return module`
    no-op check never matches , it fell through into the real per-layer
    quantize path and crashed downstream. 16 (int) is what actually no-ops."""
    bits = gb_quant.normalize_bits_token("bf16")
    assert bits == 16
    mod = _OneLayer()
    result = gb_quant.quantize_module(mod, bits=bits, device="cpu")
    assert result is mod  # no-op passthrough, never reached _delegate_patch


# ── maybe_quantize_from_env(): GB_QUANT_BITS now shares the full vocabulary ──

def test_maybe_quantize_from_env_gb_quant_bits_nvfp4(monkeypatch):
    monkeypatch.setenv("GB_QUANT_BUDGET_GB", "4.0")
    monkeypatch.setenv("GB_QUANT_BITS", "nvfp4")
    monkeypatch.delenv("GB_QUALITY", raising=False)
    captured = {}

    def _fake_quantize_to_fit(obj, budget_gb, prefer_bits=4, verbose=True):
        captured["prefer_bits"] = prefer_bits
        return "sentinel"

    with patch.object(gb_quant, "quantize_to_fit", side_effect=_fake_quantize_to_fit):
        result = gb_quant.maybe_quantize_from_env(_OneLayer(), verbose=False)

    assert result == "sentinel"
    assert captured["prefer_bits"] == "nvfp4"


def test_maybe_quantize_from_env_gb_quant_bits_auto_resolves_via_profile(monkeypatch):
    """Previously "auto"/anything not in the legacy tq*/3/2/4/8 set was
    silently ignored and prefer_bits stayed at the hardcoded default (4).
    Now "auto" resolves through gpu_profile().quality_default."""
    monkeypatch.setenv("GB_QUANT_BUDGET_GB", "4.0")
    monkeypatch.setenv("GB_QUANT_BITS", "auto")
    monkeypatch.delenv("GB_QUALITY", raising=False)
    captured = {}

    def _fake_quantize_to_fit(obj, budget_gb, prefer_bits=4, verbose=True):
        captured["prefer_bits"] = prefer_bits
        return "sentinel"

    with patch.object(gb_quant, "gpu_profile", return_value=_profile(quality_default="fp8")), \
         patch.object(gb_quant, "quantize_to_fit", side_effect=_fake_quantize_to_fit):
        gb_quant.maybe_quantize_from_env(_OneLayer(), verbose=False)

    assert captured["prefer_bits"] == "fp8"

"""Tests for gb_gguf_plan.py — the GGUF per-tensor quantization planner
(missing_features.md item (j), GGUF/llama-quantize path).

Pure Python, no real GGUF file, no gguf-py import needed (plan_from_source
takes a plain [(name, n_bytes)] inventory) — the read_gguf_tensor_inventory
wrapper around gguf-py is exercised live in W4's A/B measurement pass, not
here.
"""
import pytest

import gb_gguf_plan as gp

GiB = 2 ** 30


def _inventory(ffn_tensors=2, ffn_gb=2.0, attn_tensors=2, attn_gb=2.0,
               ssm_tensors=0, ssm_gb=1.0):
    inv = []
    for i in range(ffn_tensors):
        inv.append((f"blk.{i}.ffn_gate.weight", int(ffn_gb * GiB)))
    for i in range(attn_tensors):
        inv.append((f"blk.{i}.attn_qkv.weight", int(attn_gb * GiB)))
    for i in range(ssm_tensors):
        inv.append((f"blk.{i}.ssm_out.weight", int(ssm_gb * GiB)))
    return inv


# ── Baseline / no-op cases ─────────────────────────────────────────────────

def test_plan_makes_no_changes_when_budget_is_generous():
    inv = _inventory()
    plan = gp.plan_from_source(inv, budget_bytes=8 * GiB)

    assert plan.tensor_types == {}
    assert plan.fits_budget is True
    assert plan.estimated_bytes == pytest.approx(8 * GiB, rel=1e-6)


# ── Priority ordering: ffn squeezed first, hardest ────────────────────────

def test_plan_squeezes_ffn_before_touching_attn():
    """At a budget only ffn's own max compression is needed to satisfy,
    attn must stay completely untouched (0 overrides for it)."""
    inv = _inventory()
    plan = gp.plan_from_source(inv, budget_bytes=int(6.0 * GiB))

    assert plan.fits_budget is True
    assert plan.role_breakdown["attn"]["bytes_gb"] == pytest.approx(4.0, abs=1e-3)
    assert not any("attn" in name for name in plan.tensor_types)
    assert plan.role_breakdown["ffn"]["bytes_gb"] < 4.0
    assert all("ffn" in name for name in plan.tensor_types)


def test_plan_touches_attn_only_after_ffn_is_maxed_out():
    """At a budget tight enough that ffn's OWN maximum compression still
    isn't enough, attn must start giving up bytes too — but ffn must be at
    its floor-of-the-ladder (max compression) first, never partially spared
    while attn is untouched, and never over-compressed further once
    already at max."""
    inv = _inventory()
    plan = gp.plan_from_source(inv, budget_bytes=int(4.5 * GiB))

    ffn_max_compressed_gb = 4.0 * gp._BYTES_PER_ELEM_GGUF["iq2_xxs"] / gp._BYTES_PER_ELEM_GGUF["q8_0"]
    assert plan.role_breakdown["ffn"]["bytes_gb"] == pytest.approx(ffn_max_compressed_gb, abs=1e-3)
    assert any("attn" in name for name in plan.tensor_types)  # attn had to give up bytes too
    assert plan.role_breakdown["attn"]["bytes_gb"] < 4.0


def test_plan_least_aggressive_type_that_meets_the_requirement():
    """Within a role, the chosen type must be the LEAST aggressive one that
    still satisfies the savings requirement — never skip past a type that
    would already have been enough."""
    inv = _inventory()
    plan = gp.plan_from_source(inv, budget_bytes=int(6.0 * GiB))

    assert plan.tensor_types["blk.0.ffn_gate.weight"] == "q3_k"
    q3k_gb = 4.0 * gp._BYTES_PER_ELEM_GGUF["q3_k"] / gp._BYTES_PER_ELEM_GGUF["q8_0"]
    q4k_gb = 4.0 * gp._BYTES_PER_ELEM_GGUF["q4_k"] / gp._BYTES_PER_ELEM_GGUF["q8_0"]
    # q4_k is less aggressive than q3_k but still wouldn't have met the 2.0 GiB
    # target this budget implies for ffn — confirms q3_k wasn't chosen out of
    # over-caution, it's the least aggressive type that actually fits.
    assert q4k_gb > 2.0 > q3k_gb


# ── Sensitive roles are never touched ──────────────────────────────────────

def test_plan_never_touches_ssm_regardless_of_budget_pressure():
    inv = _inventory(ssm_tensors=4, ssm_gb=0.25)  # 1.0 GiB total ssm
    tight_budget = int(1.0 * GiB)  # far below even max-compressed ffn+attn+ssm

    plan = gp.plan_from_source(inv, budget_bytes=tight_budget)

    assert not any("ssm" in name for name in plan.tensor_types)
    assert plan.role_breakdown["ssm"]["bytes_gb"] == pytest.approx(1.0, abs=1e-6)
    assert plan.role_breakdown["ssm"]["floor_type"] == "q8_0"


def test_plan_role_breakdown_reports_floor_for_every_role():
    inv = _inventory(ssm_tensors=2, ssm_gb=0.5)
    plan = gp.plan_from_source(inv, budget_bytes=8 * GiB)

    assert plan.role_breakdown["ssm"]["floor_type"] == "q8_0"
    assert plan.role_breakdown["ffn"]["floor_type"] is None
    assert plan.role_breakdown["attn"]["floor_type"] is None


# ── Infeasible budgets ─────────────────────────────────────────────────────

def test_plan_infeasible_budget_maxes_out_every_compressible_role():
    inv = _inventory()
    plan = gp.plan_from_source(inv, budget_bytes=int(1.0 * GiB))

    assert plan.fits_budget is False
    max_gb = 4.0 * gp._BYTES_PER_ELEM_GGUF["iq2_xxs"] / gp._BYTES_PER_ELEM_GGUF["q8_0"]
    assert plan.role_breakdown["ffn"]["bytes_gb"] == pytest.approx(max_gb, abs=1e-3)
    assert plan.role_breakdown["attn"]["bytes_gb"] == pytest.approx(max_gb, abs=1e-3)
    assert plan.tensor_types["blk.0.ffn_gate.weight"] == "iq2_xxs"
    assert plan.tensor_types["blk.0.attn_qkv.weight"] == "iq2_xxs"


# ── write_tensor_type_file ─────────────────────────────────────────────────

def test_write_tensor_type_file_format(tmp_path):
    plan = gp.GGUFQuantPlan(tensor_types={"blk.1.ffn_gate.weight": "q3_k",
                                          "blk.0.ffn_gate.weight": "iq2_xxs"})
    out = gp.write_tensor_type_file(plan, tmp_path / "tensor_types.txt")

    lines = out.read_text().splitlines()
    assert lines == ["blk.0.ffn_gate.weight iq2_xxs", "blk.1.ffn_gate.weight q3_k"]


def test_write_tensor_type_file_empty_plan_writes_empty_file(tmp_path):
    plan = gp.GGUFQuantPlan(tensor_types={})
    out = gp.write_tensor_type_file(plan, tmp_path / "tensor_types.txt")

    assert out.read_text() == "\n"

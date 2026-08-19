#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for serving/resolver.py — the preset requirements.all[] resolver
(NemoClaw audit, Phase 7): typed/scoped requirement evaluation, fail-closed
on a missing fact, and the hard ambiguous-selection error on a priority tie.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serving"))

import jsonschema
import pytest
import yaml

import resolver as rs


def _req(scope, kind, id_, operator=None, value=None):
    d = {"scope": scope, "kind": kind, "id": id_}
    if operator is not None:
        d["operator"] = operator
    if value is not None or operator is not None:
        d["value"] = value
    return d


def _preset(id_, priority, requirements, recipe_ref="x.yaml"):
    return {"id": id_, "priority": priority, "recipeRef": recipe_ref,
            "requirements": {"all": requirements}}


# ── requirement evaluation ─────────────────────────────────────────────

def test_controller_truthy_requirement_passes_when_true():
    facts = {"controller": {"kmod.loaded": True}, "nodes": {}}
    req = _req("controller", "observation", "kmod.loaded")
    assert rs.evaluate_requirement(req, facts) is True


def test_controller_truthy_requirement_fails_when_false():
    facts = {"controller": {"kmod.loaded": False}, "nodes": {}}
    req = _req("controller", "observation", "kmod.loaded")
    assert rs.evaluate_requirement(req, facts) is False


def test_missing_fact_fails_closed_not_assumed_true():
    facts = {"controller": {}, "nodes": {}}
    req = _req("controller", "capability", "capability.rpc_split")
    assert rs.evaluate_requirement(req, facts) is False


def test_comparison_gte_operator():
    facts = {"controller": {"gpu.cc": 12.0}, "nodes": {}}
    req = _req("controller", "comparison", "gpu.cc", operator="gte", value=12.0)
    assert rs.evaluate_requirement(req, facts) is True
    req2 = _req("controller", "comparison", "gpu.cc", operator="gte", value=13.0)
    assert rs.evaluate_requirement(req2, facts) is False


def test_version_at_least_operator():
    facts = {"controller": {"engine_version": "33a75f41c"}, "nodes": {}}
    # numeric-component extraction: "33a75f41c" -> (33, 75, 41)
    req = _req("controller", "comparison", "engine_version",
              operator="version-at-least", value="33")
    assert rs.evaluate_requirement(req, facts) is True


def test_every_node_requires_all_nodes_to_pass():
    facts = {"controller": {}, "nodes": {
        "host": {"gpu.cc": 12.0}, "feeder1": {"gpu.cc": 12.0},
    }}
    req = _req("everyNode", "comparison", "gpu.cc", operator="gte", value=12.0)
    assert rs.evaluate_requirement(req, facts) is True

    facts["nodes"]["feeder1"]["gpu.cc"] = 8.0
    assert rs.evaluate_requirement(req, facts) is False


def test_every_node_with_no_nodes_fails_closed():
    facts = {"controller": {}, "nodes": {}}
    req = _req("everyNode", "observation", "kmod.loaded")
    assert rs.evaluate_requirement(req, facts) is False


def test_any_node_passes_if_one_node_qualifies():
    facts = {"controller": {}, "nodes": {
        "host": {"gpu.cc": 8.0}, "feeder1": {"gpu.cc": 12.0},
    }}
    req = _req("anyNode", "comparison", "gpu.cc", operator="gte", value=12.0)
    assert rs.evaluate_requirement(req, facts) is True


def test_any_node_fails_if_no_node_qualifies():
    facts = {"controller": {}, "nodes": {"host": {"gpu.cc": 8.0}}}
    req = _req("anyNode", "comparison", "gpu.cc", operator="gte", value=12.0)
    assert rs.evaluate_requirement(req, facts) is False


def test_unknown_scope_raises():
    facts = {"controller": {}, "nodes": {}}
    req = {"scope": "bogus", "kind": "observation", "id": "x"}
    with pytest.raises(ValueError, match="unknown requirement scope"):
        rs.evaluate_requirement(req, facts)


# ── preset evaluation + resolution ─────────────────────────────────────

def test_evaluate_preset_matched_true_when_all_requirements_pass():
    facts = {"controller": {"kmod.loaded": True}, "nodes": {}}
    preset = _preset("p1", 10, [_req("controller", "observation", "kmod.loaded")])
    ev = rs.evaluate_preset(preset, facts)
    assert ev.matched is True
    assert ev.failed_requirements == []


def test_evaluate_preset_lists_failed_requirements():
    facts = {"controller": {"kmod.loaded": False}, "nodes": {}}
    req = _req("controller", "observation", "kmod.loaded")
    preset = _preset("p1", 10, [req])
    ev = rs.evaluate_preset(preset, facts)
    assert ev.matched is False
    assert ev.failed_requirements == [req]


def test_resolve_picks_highest_priority_eligible_preset():
    facts = {"controller": {"kmod.loaded": True}, "nodes": {"host": {"gpu.cc": 12.0}}}
    p_low = _preset("host-only", 10, [_req("controller", "observation", "kmod.loaded")])
    p_high = _preset("rpc-split", 20, [
        _req("controller", "observation", "kmod.loaded"),
        _req("anyNode", "comparison", "gpu.cc", operator="gte", value=12.0),
    ])
    selected, evaluations = rs.resolve([p_low, p_high], facts)
    assert selected["id"] == "rpc-split"
    assert {e.preset_id for e in evaluations} == {"host-only", "rpc-split"}


def test_resolve_falls_back_when_higher_priority_preset_is_ineligible():
    """The reference-workload scenario from the plan: host-only wins when
    no feeder qualifies, even though rpc-split has higher priority."""
    facts = {"controller": {"kmod.loaded": True}, "nodes": {"host": {"gpu.cc": 8.0}}}
    p_low = _preset("host-only", 10, [_req("controller", "observation", "kmod.loaded")])
    p_high = _preset("rpc-split", 20, [
        _req("controller", "observation", "kmod.loaded"),
        _req("anyNode", "comparison", "gpu.cc", operator="gte", value=12.0),
    ])
    selected, _ = rs.resolve([p_low, p_high], facts)
    assert selected["id"] == "host-only"


def test_resolve_raises_ambiguous_selection_on_priority_tie():
    facts = {"controller": {"kmod.loaded": True}, "nodes": {}}
    req = _req("controller", "observation", "kmod.loaded")
    p1 = _preset("a", 10, [req])
    p2 = _preset("b", 10, [req])
    with pytest.raises(rs.AmbiguousSelectionError) as exc_info:
        rs.resolve([p1, p2], facts)
    assert set(exc_info.value.tied_preset_ids) == {"a", "b"}
    assert exc_info.value.priority == 10


def test_resolve_raises_value_error_when_nothing_eligible():
    facts = {"controller": {"kmod.loaded": False}, "nodes": {}}
    req = _req("controller", "observation", "kmod.loaded")
    with pytest.raises(ValueError, match="no eligible preset"):
        rs.resolve([_preset("a", 10, [req])], facts)


def test_resolve_tie_only_among_eligible_presets_not_all():
    """Two presets share the top NUMERIC priority but only one is
    eligible — must resolve cleanly, not report a false ambiguity."""
    facts = {"controller": {"kmod.loaded": True}, "nodes": {}}
    p_eligible = _preset("a", 10, [_req("controller", "observation", "kmod.loaded")])
    p_ineligible = _preset("b", 10, [_req("controller", "observation", "nonexistent.fact")])
    selected, _ = rs.resolve([p_eligible, p_ineligible], facts)
    assert selected["id"] == "a"


# ── schema validation ──────────────────────────────────────────────────

def test_preset_schema_rejects_comparison_without_operator():
    schema = __import__("json").loads(
        (Path(__file__).resolve().parent.parent / "serving" / "preset.schema.json").read_text())
    preset = _preset("a", 10, [{"scope": "controller", "kind": "comparison", "id": "x"}])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(preset, schema)


def test_preset_schema_accepts_valid_preset():
    schema = __import__("json").loads(
        (Path(__file__).resolve().parent.parent / "serving" / "preset.schema.json").read_text())
    preset = _preset("a", 10, [
        _req("controller", "observation", "kmod.loaded"),
        _req("anyNode", "comparison", "gpu.cc", operator="gte", value=12.0),
    ])
    jsonschema.validate(preset, schema)  # must not raise


# ── the real shipped presets (host-only.yaml / rpc-split.yaml) ────────
# Regression guard for the plan's own stated verification scenarios: "the
# reference workload resolves to the host-only recipe with no feeder
# online and to the rpc-split recipe with one online."

def test_real_shipped_presets_load_and_validate():
    presets = rs.load_presets()
    assert {p["id"] for p in presets} == {"host-only", "rpc-split"}


def test_real_presets_resolve_to_host_only_with_no_feeder():
    presets = rs.load_presets()
    facts = {
        "controller": {"kmod.greenboost.loaded": True},
        "nodes": {"host": {"gpu.cc": 8.9, "synapse.engine_version": "33a75f4"}},
    }
    selected, _ = rs.resolve(presets, facts)
    assert selected["id"] == "host-only"


def test_real_presets_resolve_to_rpc_split_with_qualifying_feeder():
    presets = rs.load_presets()
    facts = {
        "controller": {"kmod.greenboost.loaded": True, "cluster.feeder_count": 1},
        "nodes": {
            "host": {"gpu.cc": 12.0, "synapse.engine_version": "33a75f4"},
            "feeder1": {"gpu.cc": 12.0, "synapse.engine_version": "33a75f41c"},
        },
    }
    selected, _ = rs.resolve(presets, facts)
    assert selected["id"] == "rpc-split"


def test_real_presets_resolve_to_host_only_when_feeder_count_zero_even_if_host_alone_qualifies():
    """Regression guard for the exact gap found live: a Blackwell-class
    HOST with zero configured feeders must not look eligible for
    rpc-split just because anyNode/everyNode trivially pass against the
    host counted as its own sole node."""
    presets = rs.load_presets()
    facts = {
        "controller": {"kmod.greenboost.loaded": True, "cluster.feeder_count": 0},
        "nodes": {"host": {"gpu.cc": 12.0, "synapse.engine_version": "e8f19cc0a"}},
    }
    selected, _ = rs.resolve(presets, facts)
    assert selected["id"] == "host-only"


def test_real_presets_no_eligible_preset_when_kmod_not_loaded():
    presets = rs.load_presets()
    facts = {"controller": {"kmod.greenboost.loaded": False}, "nodes": {"host": {}}}
    with pytest.raises(ValueError, match="no eligible preset"):
        rs.resolve(presets, facts)

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for gb_synapse.serve()'s use_preset=True dispatch (NemoClaw
audit, Phase 7 follow-up: the preset resolver now CAN drive serve()'s
live use_cluster decision, on explicit opt-in). Default (use_preset=False)
must stay byte-identical to before this parameter existed.

Mocks the fake backend the same way test_gb_synapse_recipe_probe.py does
(raise a sentinel right after capturing backend.serve()'s kwargs — no real
subprocess/GPU) plus serving/resolver.py's load_presets()/resolve() so the
resolution outcome is deterministic and doesn't depend on this box's real
live cluster/GPU state.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serving"))

import pytest
import yaml

import gb_synapse as gs
import resolver as rs


class _StopServe(Exception):
    pass


@pytest.fixture
def captured_backend_call(monkeypatch):
    calls = []

    class _FakeBackend:
        def serve(self, entry, port, **kwargs):
            calls.append({"entry": entry, "port": port, **kwargs})
            raise _StopServe

    import gb_synapse_backends
    monkeypatch.setattr(gb_synapse_backends, "select_backend", lambda entry: _FakeBackend())
    monkeypatch.setattr(gs, "_resolve_model", lambda model: gs.ModelEntry(
        name=model, path="", source="hf", engine="llama.cpp"))
    monkeypatch.setattr(gs, "_read_run_states", lambda: [])
    monkeypatch.setattr(gs, "load_recipe", lambda model_name: None)  # isolate from Phase 5e
    return calls


def _fake_preset(id_, priority, tier_intent, recipe_ref="fake.yaml"):
    return {"id": id_, "priority": priority, "recipeRef": recipe_ref,
            "requirements": {"all": []}, "_tierIntent": tier_intent}


@pytest.fixture
def resolved_preset(monkeypatch, tmp_path):
    """Deterministic resolver outcome: load_presets()/resolve() return a
    controlled result, and the resolved recipe file (read back by
    serve()'s use_preset branch to get tierIntent) is a real temp file."""
    def _configure(tier_intent, preset_id="p1"):
        recipes_dir = tmp_path / "serving" / "recipes"
        recipes_dir.mkdir(parents=True, exist_ok=True)
        recipe_path = recipes_dir / "fake.yaml"
        with open(recipe_path, "w", encoding="utf-8") as f:
            yaml.safe_dump({"tierIntent": tier_intent}, f)
        preset = _fake_preset(preset_id, 10, tier_intent)
        monkeypatch.setattr(rs, "load_presets", lambda: [preset])
        monkeypatch.setattr(gs, "_REPO_DIR", tmp_path)
        monkeypatch.setattr(gs, "build_serving_facts", lambda: {"controller": {}, "nodes": {}})
        return preset

    return _configure


def test_use_preset_false_is_byte_identical_to_before(captured_backend_call):
    """Regression guard: no use_preset argument at all, use_cluster passed
    explicitly — must reach backend.serve() completely unmodified."""
    with pytest.raises(_StopServe):
        gs.serve("test-model", use_cluster=True)
    assert captured_backend_call[0]["use_cluster"] is True

    with pytest.raises(_StopServe):
        gs.serve("test-model", use_cluster=False)
    assert captured_backend_call[-1]["use_cluster"] is False


def test_use_preset_true_rpcsplit_forces_use_cluster_true(
    captured_backend_call, resolved_preset,
):
    resolved_preset("rpcSplit")
    with pytest.raises(_StopServe):
        gs.serve("test-model", use_cluster=False, use_preset=True)  # caller says False
    assert captured_backend_call[0]["use_cluster"] is True  # preset overrides it


def test_use_preset_true_t1only_forces_use_cluster_false(
    captured_backend_call, resolved_preset,
):
    resolved_preset("t1Only")
    with pytest.raises(_StopServe):
        gs.serve("test-model", use_cluster=True, use_preset=True)  # caller says True
    assert captured_backend_call[0]["use_cluster"] is False  # preset overrides it


def test_use_preset_true_t2spill_forces_use_cluster_false(
    captured_backend_call, resolved_preset,
):
    resolved_preset("t2Spill")
    with pytest.raises(_StopServe):
        gs.serve("test-model", use_cluster=True, use_preset=True)
    assert captured_backend_call[0]["use_cluster"] is False


def test_use_preset_true_raises_on_ambiguous_selection(monkeypatch, captured_backend_call, tmp_path):
    p1 = _fake_preset("a", 10, "rpcSplit")
    p2 = _fake_preset("b", 10, "t1Only")
    monkeypatch.setattr(rs, "load_presets", lambda: [p1, p2])
    monkeypatch.setattr(gs, "build_serving_facts", lambda: {"controller": {}, "nodes": {}})

    def _raise_ambiguous(presets, facts):
        raise rs.AmbiguousSelectionError(["a", "b"], 10)

    monkeypatch.setattr(rs, "resolve", _raise_ambiguous)

    with pytest.raises(rs.AmbiguousSelectionError):
        gs.serve("test-model", use_preset=True)
    assert captured_backend_call == []  # never reached backend.serve()


def test_use_preset_true_raises_on_no_eligible_preset(monkeypatch, captured_backend_call):
    monkeypatch.setattr(rs, "load_presets", lambda: [])
    monkeypatch.setattr(gs, "build_serving_facts", lambda: {"controller": {}, "nodes": {}})

    def _raise_none(presets, facts):
        raise ValueError("no eligible preset: every preset's requirements.all[] failed")

    monkeypatch.setattr(rs, "resolve", _raise_none)

    with pytest.raises(ValueError, match="no eligible preset"):
        gs.serve("test-model", use_preset=True)
    assert captured_backend_call == []

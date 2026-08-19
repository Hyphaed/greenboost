#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for gb_synapse_mcp.py's serving-recipe MCP surface (NemoClaw
audit, Phase 5e): synapse_recommend's preview_recipe mode, synapse_doctor's
recipes coverage block, and synapse_serve's explicit recipe= parameter.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serving"))

from dataclasses import dataclass

import yaml

import check_recipes as cr
import gb_synapse
import gb_synapse_mcp


def _write_recipe(path, model_name="test-model", **overrides):
    recipe = {
        "schemaVersion": "1",
        "model": {"name": model_name, "revision": "a" * 40,
                  "files": [{"path": "m.gguf", "sha256": "b" * 64}]},
        "capabilities": {"toolCalls": True, "streaming": True, "mtpDraftHead": True,
                         "visionProjector": False, "recurrentState": False},
        "kvCache": {"key": "q8_0", "value": "q8_0"},
        "ctx": 45824, "nGpuLayers": "all", "tierIntent": "t2Spill", "mtpDraftN": 4,
    }
    recipe.update(overrides)
    recipe["contentDigest"] = cr.compute_content_digest(recipe)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(recipe, f)
    return recipe


@dataclass
class _FakeReport:
    model: str
    fits: bool = True


def test_synapse_doctor_reports_recipe_coverage(tmp_path, monkeypatch):
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    monkeypatch.setattr(cr, "RECIPES_DIR", recipes_dir)
    _write_recipe(recipes_dir / "good.yaml", model_name="good-model")

    monkeypatch.setattr("gb_synapse.doctor", lambda probe_feeders=True: {"engine_installed": True})
    out = gb_synapse_mcp.synapse_doctor()
    assert "recipes" in out
    assert len(out["recipes"]) == 1
    assert out["recipes"][0]["model"] == "good-model"
    assert out["recipes"][0]["valid"] is True


def test_synapse_doctor_flags_invalid_recipe(tmp_path, monkeypatch):
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    monkeypatch.setattr(cr, "RECIPES_DIR", recipes_dir)
    path = recipes_dir / "bad.yaml"
    recipe = _write_recipe(path, model_name="bad-model")
    recipe["ctx"] = 999999  # drift after digest computed
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(recipe, f)

    monkeypatch.setattr("gb_synapse.doctor", lambda probe_feeders=True: {})
    out = gb_synapse_mcp.synapse_doctor()
    assert out["recipes"][0]["model"] == "bad-model"
    assert out["recipes"][0]["valid"] is False
    assert "error" in out["recipes"][0]


def test_synapse_recommend_preview_recipe_attaches_matching_recipe(tmp_path, monkeypatch):
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    monkeypatch.setattr(cr, "RECIPES_DIR", recipes_dir)
    _write_recipe(recipes_dir / "r.yaml", model_name="model-a")

    monkeypatch.setattr("gb_synapse.recommend",
                        lambda ctx=65536, probe_feeders=True: [_FakeReport("model-a"), _FakeReport("model-b")])
    out = gb_synapse_mcp.synapse_recommend(preview_recipe=True)
    by_model = {r["model"]: r for r in out}
    assert by_model["model-a"]["recipe"]["ctx"] == 45824
    assert by_model["model-a"]["recipe"]["tierIntent"] == "t2Spill"
    assert by_model["model-b"]["recipe"] is None


def test_synapse_recommend_without_preview_recipe_has_no_recipe_key(monkeypatch):
    monkeypatch.setattr("gb_synapse.recommend",
                        lambda ctx=65536, probe_feeders=True: [_FakeReport("model-a")])
    out = gb_synapse_mcp.synapse_recommend()
    assert "recipe" not in out[0]


def test_synapse_serve_explicit_recipe_overrides_ctx_in_dry_run(tmp_path, monkeypatch):
    recipe_path = tmp_path / "explicit.yaml"
    _write_recipe(recipe_path, model_name="test-model")

    fake_entry = gb_synapse.ModelEntry(name="test-model", path="", source="hf", engine="llama.cpp")
    monkeypatch.setattr("gb_synapse.list_models", lambda: [fake_entry])
    monkeypatch.setattr("gb_synapse.ps", lambda: [])
    monkeypatch.setattr("gb_synapse.DEFAULT_PORT", 11369, raising=False)
    monkeypatch.setattr("gb_synapse_backends.select_backend",
                        lambda entry: type("B", (), {"name": "llama.cpp"})())

    out = gb_synapse_mcp.synapse_serve("test-model", ctx=8192, recipe=str(recipe_path))
    assert out["dry_run"] is True
    assert out["ctx"] == 45824  # recipe won, not the caller's 8192
    assert out["mtp_draft_n"] == 4
    assert out["recipe_applied"]["ctx"] == 45824


def test_synapse_serve_invalid_recipe_path_returns_error(monkeypatch):
    fake_entry = gb_synapse.ModelEntry(name="test-model", path="", source="hf", engine="llama.cpp")
    monkeypatch.setattr("gb_synapse.list_models", lambda: [fake_entry])
    out = gb_synapse_mcp.synapse_serve("test-model", recipe="does-not-exist.yaml")
    assert "error" in out
    assert "does-not-exist.yaml" in out["error"] or "failed to load" in out["error"]


def test_synapse_serve_without_recipe_param_unaffected(monkeypatch):
    fake_entry = gb_synapse.ModelEntry(name="test-model", path="", source="hf", engine="llama.cpp")
    monkeypatch.setattr("gb_synapse.list_models", lambda: [fake_entry])
    monkeypatch.setattr("gb_synapse.ps", lambda: [])
    monkeypatch.setattr("gb_synapse.DEFAULT_PORT", 11369, raising=False)
    monkeypatch.setattr("gb_synapse_backends.select_backend",
                        lambda entry: type("B", (), {"name": "llama.cpp"})())

    out = gb_synapse_mcp.synapse_serve("test-model", ctx=8192)
    assert out["ctx"] == 8192
    assert out.get("recipe_applied") is None

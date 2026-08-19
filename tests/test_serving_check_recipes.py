#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for serving/check_recipes.py — the schema + digest validation gate
for GreenBoost serving recipes (NemoClaw audit, Phase 5c).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serving"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import yaml

import check_recipes as cr

_VALID_RECIPE = {
    "schemaVersion": "1",
    "model": {
        "name": "test/model",
        "revision": "a" * 40,
        "files": [{"path": "model.gguf", "sha256": "b" * 64}],
    },
    "capabilities": {
        "toolCalls": True,
        "streaming": True,
        "mtpDraftHead": False,
        "visionProjector": False,
        "recurrentState": False,
    },
    "kvCache": {"key": "q8_0", "value": "q8_0"},
    "ctx": 65536,
    "nGpuLayers": "all",
    "tierIntent": "t2Spill",
}


def _write_recipe(path: Path, recipe: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(recipe, f)


def test_real_shipped_example_recipe_passes_check():
    """The repo's own serving/recipes/example.yaml, unmodified, must pass —
    a regression guard against this template silently drifting out of sync
    with the schema or its own digest."""
    schema = cr._load_schema()
    cr.check_recipe(cr.RECIPES_DIR / "example.yaml", schema)  # raises on failure


def test_valid_recipe_with_correct_digest_passes(tmp_path):
    recipe = dict(_VALID_RECIPE)
    recipe["contentDigest"] = cr.compute_content_digest(recipe)
    path = tmp_path / "r.yaml"
    _write_recipe(path, recipe)
    schema = cr._load_schema()
    cr.check_recipe(path, schema)  # must not raise


def test_wrong_digest_is_rejected(tmp_path):
    recipe = dict(_VALID_RECIPE)
    recipe["contentDigest"] = "sha256:" + "0" * 64  # deliberately wrong
    path = tmp_path / "r.yaml"
    _write_recipe(path, recipe)
    schema = cr._load_schema()
    with pytest.raises(cr.RecipeError, match="contentDigest mismatch"):
        cr.check_recipe(path, schema)


def test_content_change_without_digest_update_is_caught(tmp_path):
    """The exact scenario check_recipes.py exists to catch: someone edits
    ctx by hand and forgets to recompute the digest."""
    recipe = dict(_VALID_RECIPE)
    recipe["contentDigest"] = cr.compute_content_digest(recipe)
    recipe["ctx"] = 4096  # drift after the digest was computed
    path = tmp_path / "r.yaml"
    _write_recipe(path, recipe)
    schema = cr._load_schema()
    with pytest.raises(cr.RecipeError, match="contentDigest mismatch"):
        cr.check_recipe(path, schema)


def test_schema_violation_is_reported(tmp_path):
    recipe = dict(_VALID_RECIPE)
    recipe["kvCache"] = {"key": "not-a-real-kv-type", "value": "q8_0"}
    recipe["contentDigest"] = cr.compute_content_digest(recipe)
    path = tmp_path / "r.yaml"
    _write_recipe(path, recipe)
    schema = cr._load_schema()
    with pytest.raises(cr.RecipeError, match="schema validation failed"):
        cr.check_recipe(path, schema)


def test_revision_must_look_like_a_real_sha_not_a_branch_name(tmp_path):
    recipe = dict(_VALID_RECIPE)
    recipe["model"] = dict(recipe["model"])
    recipe["model"]["revision"] = "main"  # a branch name, not a SHA
    recipe["contentDigest"] = cr.compute_content_digest(recipe)
    path = tmp_path / "r.yaml"
    _write_recipe(path, recipe)
    schema = cr._load_schema()
    with pytest.raises(cr.RecipeError, match="schema validation failed"):
        cr.check_recipe(path, schema)


def test_visionprojector_capability_cannot_be_claimed_true(tmp_path):
    """capabilities.visionProjector is pinned const:false in the schema —
    no recipe can claim it until a real one is verified end-to-end."""
    recipe = dict(_VALID_RECIPE)
    recipe["capabilities"] = dict(recipe["capabilities"])
    recipe["capabilities"]["visionProjector"] = True
    recipe["contentDigest"] = cr.compute_content_digest(recipe)
    path = tmp_path / "r.yaml"
    _write_recipe(path, recipe)
    schema = cr._load_schema()
    with pytest.raises(cr.RecipeError, match="schema validation failed"):
        cr.check_recipe(path, schema)


def test_fix_recomputes_a_stale_digest(tmp_path):
    recipe = dict(_VALID_RECIPE)
    recipe["contentDigest"] = "sha256:" + "0" * 64
    path = tmp_path / "r.yaml"
    _write_recipe(path, recipe)
    schema = cr._load_schema()

    changed = cr.fix_recipe(path, schema)
    assert changed is True

    cr.check_recipe(path, schema)  # now passes


def test_fix_is_idempotent_when_digest_already_correct(tmp_path):
    recipe = dict(_VALID_RECIPE)
    recipe["contentDigest"] = cr.compute_content_digest(recipe)
    path = tmp_path / "r.yaml"
    _write_recipe(path, recipe)
    schema = cr._load_schema()

    changed = cr.fix_recipe(path, schema)
    assert changed is False


def test_run_check_over_a_directory_of_recipes(tmp_path, monkeypatch):
    good = dict(_VALID_RECIPE)
    good["contentDigest"] = cr.compute_content_digest(good)
    bad = dict(_VALID_RECIPE)
    bad["contentDigest"] = "sha256:" + "0" * 64

    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    _write_recipe(recipes_dir / "good.yaml", good)
    _write_recipe(recipes_dir / "bad.yaml", bad)

    monkeypatch.setattr(cr, "RECIPES_DIR", recipes_dir)
    errors = cr.run_check()
    assert len(errors) == 1
    assert "bad.yaml" in errors[0]


def test_run_check_over_empty_directory_reports_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "RECIPES_DIR", tmp_path / "does-not-exist")
    assert cr.run_check() == []

# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""Serving recipes may carry an optional `vision` and `proxy` block (schema only: the digest of existing recipes is unaffected)."""
import copy
import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")
yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "serving" / "recipe.schema.json").read_text())


def _recipe():
    return yaml.safe_load((ROOT / "serving" / "recipes" / "qwen38-35b-a3b-apex-miniplus.yaml").read_text())


def _errors(r):
    return list(jsonschema.Draft202012Validator(SCHEMA).iter_errors(r)) if "2020-12" in SCHEMA.get("$schema", "") \
        else list(jsonschema.Draft7Validator(SCHEMA).iter_errors(r))


def test_the_existing_recipe_still_validates_unchanged():
    assert _errors(_recipe()) == []


def test_vision_and_proxy_blocks_validate():
    r = copy.deepcopy(_recipe())
    r["vision"] = {"nCpuMoe": 32, "imageMinTokens": 1024, "imageMaxTokens": 1600}
    r["proxy"] = {"addressSpacePct": 20, "mallocArenaMax": 3}
    assert _errors(r) == []


@pytest.mark.parametrize("block,bad", [("vision", {"nCpuMoe": -1}), ("vision", {"imageMinTokens": 0}), ("vision", {"surprise": 1}),
                                       ("proxy", {"addressSpacePct": 0}), ("proxy", {"addressSpacePct": 150}), ("proxy", {"mallocArenaMax": 0}),
                                       ("proxy", {"extra": True})])
def test_bad_blocks_are_rejected(block, bad):
    r = copy.deepcopy(_recipe())
    r[block] = bad
    assert _errors(r), f"{block} {bad} should not validate"

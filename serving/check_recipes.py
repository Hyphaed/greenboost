#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""serving/check_recipes.py — validate + digest-check every serving recipe.

NemoClaw audit, Phase 5c. Mirrors NemoClaw's `generate-catalog.ts`'s
`--check` gate (design re-implemented, no code copied): every recipe under
`serving/recipes/*.yaml` must (1) validate against `recipe.schema.json`,
and (2) carry a `contentDigest` that matches `serving/digest.py`'s
recomputed digest of the recipe's own content (with `contentDigest` itself
excluded from what gets hashed — a self-referential field can't be part of
its own input). Same discipline `semantics/*.yaml`'s fail-loud-on-malformed
convention already has for GB-Semantics, applied to the serving layer.

Usage:
    python3 serving/check_recipes.py --check      # exit 1 on any failure
    python3 serving/check_recipes.py --fix         # recompute + rewrite
                                                    # every contentDigest
    python3 serving/check_recipes.py               # same as --check
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_REPO_DIR))

import digest as digest_mod  # noqa: E402

RECIPES_DIR = Path(__file__).resolve().parent / "recipes"
SCHEMA_PATH = Path(__file__).resolve().parent / "recipe.schema.json"


class RecipeError(Exception):
    """One recipe file failed validation or digest verification."""


def _load_schema() -> dict:
    import json
    return json.loads(SCHEMA_PATH.read_text())


def _load_recipe(path: Path) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RecipeError(f"{path}: recipe must be a YAML mapping, got {type(data).__name__}")
    return data


def compute_content_digest(recipe: dict) -> str:
    """The recipe's digest, computed over every field EXCEPT contentDigest
    itself (a field can't validate its own hash if it's part of the input
    to that hash)."""
    payload = {k: v for k, v in recipe.items() if k != "contentDigest"}
    return digest_mod.digest(payload)


def check_recipe(path: Path, schema: dict) -> None:
    """Raise RecipeError with a specific reason on any failure; return
    silently on success."""
    try:
        import jsonschema
    except ImportError as e:
        raise RecipeError(
            "the 'jsonschema' package is required to validate serving recipes "
            "(pip install jsonschema); it is already a declared GreenBoost "
            "dependency for GB-Semantics' YAML validation"
        ) from e

    recipe = _load_recipe(path)

    try:
        jsonschema.validate(recipe, schema)
    except jsonschema.ValidationError as e:
        raise RecipeError(f"{path}: schema validation failed: {e.message}") from e

    if "contentDigest" not in recipe:
        raise RecipeError(f"{path}: missing contentDigest (should have failed schema validation)")

    expected = compute_content_digest(recipe)
    actual = recipe["contentDigest"]
    if actual != expected:
        raise RecipeError(
            f"{path}: contentDigest mismatch — recipe content changed without "
            f"recomputing its digest. Expected {expected}, found {actual}. "
            f"Run `python3 serving/check_recipes.py --fix` to update it."
        )


def fix_recipe(path: Path, schema: dict) -> bool:
    """Recompute and rewrite path's contentDigest in place. Returns True if
    the file was changed. Still validates against the schema first (a
    structurally invalid recipe can't be "fixed" by recomputing a digest
    for content that's already wrong) — raises RecipeError on that."""
    import yaml

    recipe = _load_recipe(path)
    payload = {k: v for k, v in recipe.items() if k != "contentDigest"}
    try:
        import jsonschema
        # Validate the payload shape ignoring contentDigest's own presence
        # requirement isn't meaningful pre-fix — validate against the real
        # schema only AFTER the digest is filled in, at the end, so a
        # --fix run on a recipe missing everything else still reports the
        # real structural problem instead of "missing contentDigest".
        probe = dict(payload)
        probe["contentDigest"] = "sha256:" + "0" * 64
        jsonschema.validate(probe, schema)
    except jsonschema.ValidationError as e:
        raise RecipeError(f"{path}: schema validation failed: {e.message}") from e

    new_digest = digest_mod.digest(payload)
    if recipe.get("contentDigest") == new_digest:
        return False
    recipe["contentDigest"] = new_digest
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(recipe, f, default_flow_style=False, sort_keys=False)
    return True


def find_recipe_files() -> "list[Path]":
    if not RECIPES_DIR.is_dir():
        return []
    return sorted(RECIPES_DIR.glob("*.yaml"))


def run_check() -> "list[str]":
    """Return a list of error strings (empty = all recipes valid)."""
    schema = _load_schema()
    errors = []
    for path in find_recipe_files():
        try:
            check_recipe(path, schema)
        except RecipeError as e:
            errors.append(str(e))
    return errors


def run_fix() -> "list[Path]":
    """Recompute every recipe's digest in place; return the paths that
    actually changed."""
    schema = _load_schema()
    changed = []
    for path in find_recipe_files():
        if fix_recipe(path, schema):
            changed.append(path)
    return changed


def main(argv: "list[str] | None" = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--fix" in argv:
        changed = run_fix()
        if changed:
            for path in changed:
                print(f"updated: {path}")
        else:
            print("no recipe digests needed updating")
        # After fixing, still run the check — a schema problem doesn't go
        # away just because the digest was recomputed.
        errors = run_check()
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1 if errors else 0

    errors = run_check()
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(f"\n{len(errors)} recipe(s) failed.", file=sys.stderr)
        return 1
    n = len(find_recipe_files())
    print(f"{n} recipe(s) OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

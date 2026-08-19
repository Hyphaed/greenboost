#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""serving/resolver.py — preset requirements.all[] resolver.

NemoClaw audit, Phase 7. Design re-implemented from scratch (no code
copied) from the IDEAS in NemoClaw's `resolver.ts` + `preset.schema.json`:
five typed requirement kinds (qualification/observation/capability/
comparison/fact), each scoped `controller`/`everyNode`/`anyNode`
(GreenBoost's host/all-nodes/any-node distinction), and a HARD
`ambiguous-selection` error when two eligible presets tie at the top
priority instead of silently picking one.

Facts snapshot shape this module consumes (built by a caller — this module
has no I/O of its own, matching serving/probe.py's dependency-injection
style so it stays unit-testable without a live cluster):

    {
        "controller": {"kmod.greenboost.loaded": True, "gpu.cc": 12.0, ...},
        "nodes": {
            "host":    {"gpu.cc": 12.0, "synapse.engine_version": "33a75f4"},
            "feeder1": {"gpu.cc": 12.0, "synapse.engine_version": "33a75f41c"},
        },
    }

`controller`-scoped requirements check ONLY the `controller` dict.
`everyNode`/`anyNode`-scoped requirements check every entry in `nodes`
(which should include the host itself as one entry, same convention
gb_cluster.py already uses for its own snapshots) — a caller that wants a
requirement to also bind the host under `everyNode`/`anyNode` semantics
must include a `"host"` entry in `nodes`, this module does not add one
implicitly.

A missing fact id ALWAYS fails the requirement (fail closed) — never
treated as "assume true" or "assume false-but-ignore"; the whole point of
declared requirements is that an unknown fact can't silently pass.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PRESETS_DIR = Path(__file__).resolve().parent / "presets"
_SCHEMA_PATH = Path(__file__).resolve().parent / "preset.schema.json"


def find_preset_files() -> "list[Path]":
    if not PRESETS_DIR.is_dir():
        return []
    return sorted(PRESETS_DIR.glob("*.yaml"))


def load_presets(paths: "list[Path] | None" = None) -> "list[dict]":
    """Load and schema-validate every preset in `paths` (default: every
    file under PRESETS_DIR). Raises jsonschema.ValidationError immediately
    on a malformed preset — fail loud, same convention as
    serving/check_recipes.py, rather than silently skipping a broken file
    that would then never be eligible for anything."""
    import jsonschema
    import yaml

    schema = json.loads(_SCHEMA_PATH.read_text())
    presets = []
    for path in (paths if paths is not None else find_preset_files()):
        with open(path, encoding="utf-8") as f:
            preset = yaml.safe_load(f)
        jsonschema.validate(preset, schema)
        presets.append(preset)
    return presets


class AmbiguousSelectionError(Exception):
    """Raised when two or more eligible presets share the top priority.
    Carries the tied preset ids so the caller can report them, not just
    "something was ambiguous"."""

    def __init__(self, tied_preset_ids: "list[str]", priority: int):
        self.tied_preset_ids = tied_preset_ids
        self.priority = priority
        super().__init__(
            f"ambiguous-selection: presets {tied_preset_ids!r} all tie at "
            f"priority {priority} — no single preset can be chosen"
        )


@dataclass
class Evaluation:
    """Why one preset did or did not match — kept even for a passing
    preset so a caller can show its full reasoning, not just a boolean."""
    preset_id: str
    matched: bool
    failed_requirements: "list[dict]" = field(default_factory=list)


_COMPARATORS = {
    "eq": lambda a, b: a == b,
    "neq": lambda a, b: a != b,
    "gt": lambda a, b: a is not None and a > b,
    "gte": lambda a, b: a is not None and a >= b,
    "lt": lambda a, b: a is not None and a < b,
    "lte": lambda a, b: a is not None and a <= b,
}

_VERSION_COMPONENT_RE = re.compile(r"\d+")


def _version_tuple(v: str) -> "tuple[int, ...]":
    """Best-effort numeric-component extraction for version-at-least — good
    enough for the version strings GreenBoost's own facts actually carry
    (semver-ish, or a git short SHA's leading digits when nothing better
    exists); a caller with genuinely non-numeric versions should use a
    plain "eq"/"neq" comparison instead, not this operator."""
    nums = _VERSION_COMPONENT_RE.findall(str(v))
    return tuple(int(n) for n in nums) if nums else (0,)


def _version_at_least(actual: Any, minimum: Any) -> bool:
    if actual is None:
        return False
    return _version_tuple(actual) >= _version_tuple(minimum)


def _apply_operator(operator: str, actual: Any, expected: Any) -> bool:
    if operator == "version-at-least":
        return _version_at_least(actual, expected)
    cmp = _COMPARATORS.get(operator)
    if cmp is None:
        raise ValueError(f"unknown comparison operator {operator!r}")
    return cmp(actual, expected)


def _matches_one(requirement: dict, facts: dict) -> bool:
    """Evaluate `requirement` against a single flat facts dict (either the
    controller's, or one node's)."""
    fact_id = requirement["id"]
    if fact_id not in facts:
        return False  # fail closed on a missing fact
    actual = facts[fact_id]

    operator = requirement.get("operator")
    if operator is not None:
        return _apply_operator(operator, actual, requirement["value"])

    # No operator: qualification/observation/capability/fact all default to
    # a truthy check on the fact's own value.
    return bool(actual)


def evaluate_requirement(requirement: dict, facts_snapshot: dict) -> bool:
    """Evaluate one requirement against the full facts_snapshot, applying
    its `scope`."""
    scope = requirement["scope"]
    if scope == "controller":
        return _matches_one(requirement, facts_snapshot.get("controller", {}))
    nodes = facts_snapshot.get("nodes", {})
    if scope == "everyNode":
        # Vacuously true on an empty node set would let a requirement pass
        # with no evidence at all — fail closed instead: everyNode with no
        # nodes to check means the requirement cannot be shown to hold.
        return bool(nodes) and all(_matches_one(requirement, n) for n in nodes.values())
    if scope == "anyNode":
        return any(_matches_one(requirement, n) for n in nodes.values())
    raise ValueError(f"unknown requirement scope {scope!r}")


def evaluate_preset(preset: dict, facts_snapshot: dict) -> Evaluation:
    failed = []
    for req in preset["requirements"]["all"]:
        if not evaluate_requirement(req, facts_snapshot):
            failed.append(req)
    return Evaluation(preset_id=preset["id"], matched=not failed, failed_requirements=failed)


def resolve(presets: "list[dict]", facts_snapshot: dict) -> "tuple[dict, list[Evaluation]]":
    """Evaluate every preset, then select the highest-priority ELIGIBLE
    one. Returns (selected_preset, all_evaluations) — the evaluations list
    lets a caller show why every other preset was or wasn't in the running,
    not just which one won.

    Raises AmbiguousSelectionError if two or more eligible presets share
    the top priority — a silent pick here is exactly the failure mode this
    module exists to prevent (e.g. a feeder-absent box picking an
    rpc-split recipe because nothing broke the tie deliberately).

    Raises ValueError if no preset is eligible at all."""
    evaluations = [evaluate_preset(p, facts_snapshot) for p in presets]
    eligible_ids = {e.preset_id for e in evaluations if e.matched}
    eligible = [p for p in presets if p["id"] in eligible_ids]

    if not eligible:
        raise ValueError("no eligible preset: every preset's requirements.all[] failed")

    top_priority = max(p["priority"] for p in eligible)
    tied = [p for p in eligible if p["priority"] == top_priority]
    if len(tied) > 1:
        raise AmbiguousSelectionError([p["id"] for p in tied], top_priority)

    return tied[0], evaluations

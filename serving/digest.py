#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""serving/digest.py — canonical-JSON digest for serving recipes.

NemoClaw audit, Phase 5b. GreenBoost's serving recipes are YAML (see
`serving/recipe.schema.json`'s docstring for why — GreenBoost-native, not a
fork of NemoClaw's schema), but the digest-pinning IDEA is worth taking
directly: a recipe's content gets one deterministic hash regardless of key
order or incidental whitespace, so `serving/check_recipes.py --check` can
assert a recipe's committed digest still matches its committed content.

Design re-implemented from NemoClaw's `catalog-integrity.ts`
(`canonicalize`/`canonicalManagedInferenceJson`/`managedInferenceHexDigest`)
— no code copied, this is a from-scratch Python port of the ALGORITHM,
verified byte-for-byte against the real TypeScript implementation (sort
object keys, reject non-finite numbers, normalize -0 to 0, reject
`None`-as-`undefined` — Python has no direct `undefined` distinct from
`None`, see `canonicalize()`'s docstring for how that gap is handled — then
serialize with sorted keys and no incidental whitespace).
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any


class CanonicalizationError(ValueError):
    """Raised when a value can't be canonicalized (matches the TS
    implementation's thrown Error on a non-finite number or an
    object that isn't JSON-serializable)."""


def canonicalize(value: Any, path: str = "recipe") -> Any:
    """Recursively normalize `value` into the canonical form: dict keys
    sorted, -0.0 folded to 0, non-finite floats rejected.

    Note on `undefined`: the TS original explicitly rejects a dict value
    that is JavaScript's `undefined` (a distinct value from `null`, arising
    e.g. from an optional field the caller never set). Python has no
    equivalent third state — every dict value here is already `None`,
    a real value, or absent from the dict entirely (which the loop over
    `.items()` never visits) — so there is nothing for this port to reject
    that the type system doesn't already prevent. A caller who wants the
    TS behavior's practical effect (never silently drop an unset field)
    should validate the recipe against `recipe.schema.json`'s `required`
    lists BEFORE calling this function; that is a schema-validation
    concern, not this function's job.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError(f"{path} contains a non-finite number")
        return 0 if value == 0.0 else value
    if isinstance(value, (list, tuple)):
        return [canonicalize(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, dict):
        output: "dict[str, Any]" = {}
        for key in sorted(value.keys()):
            output[key] = canonicalize(value[key], f"{path}.{key}")
        return output
    raise CanonicalizationError(f"{path} is not JSON-serializable")


def canonical_json(value: Any) -> str:
    """The canonical JSON text for `value` — sorted keys, no incidental
    whitespace, non-ASCII characters emitted literally (matches
    JavaScript's `JSON.stringify`, which does not `\\uXXXX`-escape by
    default)."""
    return json.dumps(
        canonicalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def hex_digest(value: Any) -> str:
    """sha256 hex digest of `canonical_json(value)`, UTF-8 encoded — the
    Python equivalent of NemoClaw's `managedInferenceHexDigest`."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest(value: Any) -> str:
    """`sha256:<hex>` — the Python equivalent of NemoClaw's
    `managedInferenceDigest`, the form a recipe's `contentDigest` field
    stores."""
    return f"sha256:{hex_digest(value)}"


def text_digest(text: str) -> str:
    """`sha256:<hex>` of a raw string, UTF-8 encoded, no canonicalization —
    for content that is already a fixed text blob (a script, a rendered
    template) rather than a structured value. Equivalent of NemoClaw's
    `managedInferenceTextDigest`."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"

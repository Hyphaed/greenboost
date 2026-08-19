#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for serving/digest.py — the canonical-JSON digest routine ported
from NemoClaw's catalog-integrity.ts (NemoClaw audit, Phase 5b).

The four expected (canonical_json, hex_digest) pairs below were generated
by running the REAL TypeScript implementation
(`src/lib/inference/serving/catalog-integrity.ts`'s `canonicalize`/
`canonicalManagedInferenceJson`/`managedInferenceHexDigest`) under Node.js
against these exact same input values, then hardcoded here as a fixed
regression pin — this Python port must keep producing byte-identical
output even if Node isn't available to re-verify against at test time.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serving"))

import pytest

import digest as d

# (input, expected canonical JSON, expected sha256 hex) — verified against
# the real TS implementation via `node` on 2026-08-05.
_TS_VERIFIED_CASES = [
    (
        {"b": 2, "a": 1, "c": [3, 2, 1], "d": {"z": -0.0, "y": None, "x": True}},
        '{"a":1,"b":2,"c":[3,2,1],"d":{"x":true,"y":null,"z":0}}',
        "a89c95751eda1ed68ec7a701239ead61143878f02792b2d9c3767384a9be31ae",
    ),
    (
        {"name": "qwen3.6-27b", "revision": "abc123", "nested": {"arr": [1, 2, {"k": "v"}]}},
        '{"name":"qwen3.6-27b","nested":{"arr":[1,2,{"k":"v"}]},"revision":"abc123"}',
        "3093e664c8dfbe151daeac7efb4e44585b8071ec95874b044d09c89f62cb5a87",
    ),
    (
        {},
        "{}",
        "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    ),
    (
        {"unicode": "héllo wörld 日本語"},
        '{"unicode":"héllo wörld 日本語"}',
        "0a72e8baf00f5179f0047d739b5be8b2da386f34884f974f858dfb3ae812a39d",
    ),
]


@pytest.mark.parametrize("value,expected_json,expected_hex", _TS_VERIFIED_CASES)
def test_canonical_json_matches_real_typescript_output(value, expected_json, expected_hex):
    assert d.canonical_json(value) == expected_json


@pytest.mark.parametrize("value,expected_json,expected_hex", _TS_VERIFIED_CASES)
def test_hex_digest_matches_real_typescript_output(value, expected_json, expected_hex):
    assert d.hex_digest(value) == expected_hex


def test_digest_prefixes_sha256():
    assert d.digest({"a": 1}) == f"sha256:{d.hex_digest({'a': 1})}"


def test_canonicalize_sorts_keys_regardless_of_input_order():
    a = d.canonicalize({"z": 1, "a": 2})
    b = d.canonicalize({"a": 2, "z": 1})
    assert list(a.keys()) == list(b.keys()) == ["a", "z"]


def test_canonicalize_rejects_non_finite_float():
    with pytest.raises(d.CanonicalizationError):
        d.canonicalize(math.nan)
    with pytest.raises(d.CanonicalizationError):
        d.canonicalize(math.inf)


def test_canonicalize_folds_negative_zero_to_zero():
    assert d.canonicalize(-0.0) == 0


def test_canonicalize_recurses_into_nested_structures():
    value = {"a": [{"z": 1, "b": 2}, {"y": -0.0}]}
    result = d.canonicalize(value)
    assert list(result["a"][0].keys()) == ["b", "z"]
    assert result["a"][1]["y"] == 0


def test_text_digest_is_deterministic_and_prefixed():
    a = d.text_digest("hello world")
    b = d.text_digest("hello world")
    assert a == b
    assert a.startswith("sha256:")


def test_digest_is_deterministic_regardless_of_key_order():
    v1 = {"a": 1, "b": {"x": 1, "y": 2}}
    v2 = {"b": {"y": 2, "x": 1}, "a": 1}
    assert d.digest(v1) == d.digest(v2)

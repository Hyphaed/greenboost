"""Tests for instruments/capture.py — the bounded, redacting capture core
adapted from NemoClaw's nemoclaw_observability.py (NemoClaw audit, Phase 3d;
see third_party/nemoclaw_patterns/NOTICE). Exercises the behaviors that
motivate the port: secret-shaped-value redaction, credential-shaped key
redaction, and the size/depth/cycle bounds that keep a dataflux
`cli_tool_call` emit from becoming unbounded or leaking a pasted token.

Run with:  python -m pytest tests/ -v
"""
from __future__ import annotations

import json

from greenboost_cli.instruments.capture import bounded_capture


def test_bounded_capture_redacts_bearer_token_in_a_string_value():
    params = {"command": "curl -H 'Authorization: Bearer abcdefghijklmnop1234567890'"}
    result = bounded_capture(params)
    assert "abcdefghijklmnop1234567890" not in json.dumps(result)
    assert "Bearer <redacted-secret>" in result["command"]


def test_bounded_capture_redacts_known_provider_token_shape():
    params = {"env": "OPENAI_API_KEY=sk-ant-abcdefghijklmnopqrstuvwxyz1234567890"}
    result = bounded_capture(params)
    assert "abcdefghijklmnopqrstuvwxyz1234567890" not in json.dumps(result)


def test_bounded_capture_redacts_credential_shaped_key_regardless_of_value_shape():
    """A key literally named api_key/token/password/etc. is redacted
    outright — even a value with none of the recognized token shapes."""
    params = {"api_key": "just some ordinary looking text, no token shape at all"}
    result = bounded_capture(params)
    assert result["api_key"] == "<redacted>"


def test_bounded_capture_redacts_credential_shaped_key_camel_case():
    params = {"apiKey": "another ordinary-looking value"}
    result = bounded_capture(params)
    assert result["apiKey"] == "<redacted>"


def test_bounded_capture_preserves_non_sensitive_fields():
    params = {"file_path": "/tmp/example.txt", "limit": 100}
    result = bounded_capture(params)
    assert result == {"file_path": "/tmp/example.txt", "limit": 100}


def test_bounded_capture_handles_reference_cycle():
    cyclic: dict = {}
    cyclic["self"] = cyclic
    result = bounded_capture(cyclic)
    assert result["self"] == {"_omitted_reference": "shared_or_cycle"}


def test_bounded_capture_bounds_depth():
    deep: dict = {}
    cursor = deep
    for _ in range(20):
        cursor["x"] = {}
        cursor = cursor["x"]
    result = bounded_capture(deep)
    # Walk down exactly MAX_CAPTURE_DEPTH levels before hitting the marker.
    node = result
    depth = 0
    while isinstance(node, dict) and "x" in node:
        node = node["x"]
        depth += 1
    assert node == {"_omitted_at_depth": 8}
    assert depth == 8


def test_bounded_capture_bounds_item_count():
    big = {f"k{i}": i for i in range(200)}
    result = bounded_capture(big)
    assert result["_truncated_items"] == 200 - 50


def test_bounded_capture_never_raises_on_a_weird_object():
    class Weird:
        def __getattr__(self, name):
            raise RuntimeError("hostile __getattr__")

    result = bounded_capture({"payload": Weird()})
    assert result["payload"] == {"_omitted_type": "opaque"}


def test_bounded_capture_result_is_always_json_serializable():
    params = {"a": [1, 2, {"b": "x" * 100}], "c": None, "d": True, "e": 3.14}
    result = bounded_capture(params)
    json.dumps(result)  # must not raise

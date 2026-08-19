"""Tool-call arguments that don't parse must say WHY, and never be salvaged.

Live incident 2026-08-19: a model wrote a long Godot script through Write, its
streamed arguments were cut off by the token limit, and the CLI collapsed the
failure to {"_raw": ...}. Write then reported "invalid parameters
('file_path')" — a parameter name that was never the problem — so the model
retried the identical oversized call.
"""
import json

import pytest

from greenboost_cli.inference.adapters import (
    TOOL_ARG_PARSE_ERROR_KEY,
    parse_tool_arguments,
    _looks_truncated,
)


def test_valid_json_parses():
    inp, err = parse_tool_arguments('{"file_path": "/a/b.gd", "content": "x"}')
    assert err is None
    assert inp["file_path"] == "/a/b.gd"


def test_empty_arguments_are_an_empty_dict():
    for raw in ("", None):
        inp, err = parse_tool_arguments(raw)
        assert (inp, err) == ({}, None)


def test_literal_newlines_in_a_string_are_repaired_silently():
    """The #1 local-model failure: real newlines in generated code.

    The payload is COMPLETE, so this must repair without bothering the model.
    """
    raw = '{"file_path": "/a/b.gd", "content": "extends Node2D\nfunc _ready():\n\tpass"}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)                      # strict JSON genuinely rejects it
    inp, err = parse_tool_arguments(raw)
    assert err is None
    assert inp["file_path"] == "/a/b.gd"
    assert "extends Node2D" in inp["content"]
    assert TOOL_ARG_PARSE_ERROR_KEY not in inp


def test_truncated_arguments_are_reported_not_salvaged():
    """A salvaged Write writes half a file and reports success. Never do it."""
    raw = '{"file_path": "/a/b.gd", "content": "extends Node2D\\n## half a fi'
    inp, err = parse_tool_arguments(raw)
    assert err is not None
    assert "cut off" in err and "token limit" in err
    assert inp[TOOL_ARG_PARSE_ERROR_KEY] == err
    assert "content" not in inp          # nothing partial leaks to the handler
    assert inp["_raw"] == raw


def test_malformed_but_complete_json_reports_position():
    inp, err = parse_tool_arguments('{"file_path": , "content": "x"}')
    assert err is not None
    assert "not valid JSON" in err
    assert "cut off" not in err          # not misreported as truncation


def test_non_object_arguments_rejected():
    inp, err = parse_tool_arguments('["file_path", "/a/b.gd"]')
    assert err is not None and "must be a JSON object" in err


def test_braces_inside_string_literals_do_not_look_like_structure():
    """Generated code is full of braces; they must not fake an open depth."""
    assert not _looks_truncated('{"content": "func f() { return {1:2}; }"}')
    assert not _looks_truncated('{"content": "a \\" quoted brace {"}')
    assert _looks_truncated('{"content": "func f() {')


def test_dispatcher_reports_the_parse_failure_not_a_missing_parameter():
    from greenboost_cli.instruments import dispatcher as D
    raw = '{"file_path": "/a/b.gd", "content": "extends Node2D\\n## half a fi'
    params, err = parse_tool_arguments(raw)
    out = D.dispatch("Write", params, agent="main") if hasattr(D, "dispatch") else None
    if out is None:
        pytest.skip("dispatch() not exposed under that name")
    assert "could not be read" in out
    assert "invalid parameters" not in out

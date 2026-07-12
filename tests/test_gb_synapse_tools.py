"""Tests for gb_synapse_tools — tool-call injection lifted from greenboost-cli.
Pure/stdlib; the only external touch (`ollama show` in _ollama_parser_name) is
monkeypatched so nothing shells out.
"""
import pytest

import gb_synapse_tools as t


@pytest.fixture(autouse=True)
def _no_ollama(monkeypatch):
    # default: no ollama PARSER resolved → family detection uses name hints only
    monkeypatch.setattr(t, "_ollama_parser_name", lambda model: "")
    t._parser_name_cache.clear()


# ── should_inject_tools ─────────────────────────────────────────────────────
def test_tool_format_inject_forces_injection():
    assert t.should_inject_tools({"tool_format": "inject"}) is True


def test_tool_format_native_forces_native():
    assert t.should_inject_tools(
        {"tool_format": "native", "gb_synapse_native_fc": False}) is False


def test_native_fc_default_uses_native():
    assert t.should_inject_tools({}, "some-llama-gguf") is False


def test_native_fc_false_forces_injection():
    assert t.should_inject_tools({"gb_synapse_native_fc": False}) is True


def test_qwen35_family_always_injects():
    assert t.should_inject_tools({}, "qwen3.6-coder") is True


def test_parser_value_beats_name(monkeypatch):
    monkeypatch.setattr(t, "_ollama_parser_name", lambda m: "qwen3-coder")
    # even a non-qwen name is treated as XML family when PARSER says so
    assert t._is_qwen35_family("some-random-model") is True


# ── parse_tool_calls_from_text ──────────────────────────────────────────────
def test_parse_fenced_json():
    text = 'sure\n```json\n{"name": "read", "arguments": {"path": "/x"}}\n```'
    calls = t.parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "read"
    assert calls[0]["input"] == {"path": "/x"}
    assert "id" in calls[0]


def test_parse_hermes_tags():
    text = '<tool_call>{"name": "ls", "arguments": {}}</tool_call>'
    calls = t.parse_tool_calls_from_text(text)
    assert calls[0]["name"] == "ls"


def test_parse_glm_tokens():
    text = '<|tool_call_begin|>{"name": "grep", "arguments": {"q": "x"}}<|tool_call_end|>'
    calls = t.parse_tool_calls_from_text(text)
    assert calls[0]["name"] == "grep"
    assert calls[0]["input"]["q"] == "x"


def test_parse_bare_json_fallback():
    text = 'here: {"name": "run", "arguments": {"cmd": "ls"}} done'
    calls = t.parse_tool_calls_from_text(text)
    assert calls[0]["name"] == "run"


def test_parse_dedup_identical_blocks():
    block = '<tool_call>{"name": "ls", "arguments": {}}</tool_call>'
    calls = t.parse_tool_calls_from_text(block + block)
    assert len(calls) == 1


def test_parse_trailing_comma_repaired():
    text = '```json\n{"name": "ls", "arguments": {"a": 1,}}\n```'
    calls = t.parse_tool_calls_from_text(text)
    assert calls[0]["input"]["a"] == 1


def test_parse_none_when_absent():
    assert t.parse_tool_calls_from_text("just prose, no tools") == []


# ── Qwen XML parse + value coercion ─────────────────────────────────────────
def test_parse_qwen_xml():
    schemas = [{"name": "calc", "input_schema": {
        "properties": {"n": {"type": "integer"}, "flag": {"type": "boolean"}}}}]
    text = ("<tool_call><function=calc>"
            "<parameter=n>\n42\n</parameter>"
            "<parameter=flag>true</parameter>"
            "</function></tool_call>")
    calls = t.parse_tool_calls_from_text(text, schemas, model="qwen3.6")
    assert calls[0]["name"] == "calc"
    assert calls[0]["input"] == {"n": 42, "flag": True}


def test_coerce_qwen_value_types():
    assert t._coerce_qwen_value("null", "string") is None
    assert t._coerce_qwen_value("true", "boolean") is True
    assert t._coerce_qwen_value("7", "integer") == 7
    assert t._coerce_qwen_value("1.5", "number") == 1.5
    assert t._coerce_qwen_value("[1,2]", "array") == [1, 2]
    assert t._coerce_qwen_value("plain", None) == "plain"


# ── inject_tools_into_system ─────────────────────────────────────────────────
_SCHEMAS = [{"name": "read", "description": "read a file",
             "input_schema": {"properties": {"path": {"type": "string",
                                                      "description": "the path"}},
                              "required": ["path"]}}]


def test_inject_hermes_format():
    out = t.inject_tools_into_system("BASE", _SCHEMAS, model="llama")
    assert out.startswith("BASE")
    assert "## Tool Use" in out
    assert '"name": "read"' in out
    assert "*(required)*" in out


def test_inject_xml_format_for_qwen():
    out = t.inject_tools_into_system("BASE", _SCHEMAS, model="qwen3.6")
    assert "<tools>" in out
    assert "<function>" in out
    assert "<name>read</name>" in out


# ── messages_to_openai_for_injection ────────────────────────────────────────
def test_messages_tool_becomes_user():
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "name": "read", "content": "file body"}]
    out = t.messages_to_openai_for_injection(msgs)
    assert out[2]["role"] == "user"
    assert "Tool result" in out[2]["content"]
    assert "read" in out[2]["content"]


# ── clean_text_from_tool_blocks ─────────────────────────────────────────────
def test_clean_strips_blocks():
    text = ('answer here\n<tool_call>{"name":"x","arguments":{}}</tool_call>\n'
            'more text')
    cleaned = t.clean_text_from_tool_blocks(text)
    assert "tool_call" not in cleaned
    assert "answer here" in cleaned
    assert "more text" in cleaned

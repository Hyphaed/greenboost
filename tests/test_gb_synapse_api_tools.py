#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_synapse_api.py's server-side tool-calling (P6):

  * native mode (llama.cpp, --jinja): tools/tool_choice forwarded as-is on
    /v1/chat/completions (pure passthrough, tested in
    test_gb_synapse_api_upstream.py already); /api/chat translates the
    response — OpenAI's `arguments` is a JSON STRING, Ollama's is a parsed
    OBJECT.
  * emulate mode (torch/transformers/diffusers, no --jinja): inject tool
    definitions into the system message via gb_synapse_tools, force
    non-streaming upstream, parse tool calls out of the finished text,
    strip the tool-call markup from the visible content — for both the
    Ollama surface (/api/chat) and the OpenAI surface (/v1/chat/completions,
    which stops being a pure passthrough only in this one case).
"""
import asyncio
import json

import pytest

pytest.importorskip("aiohttp")

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import gb_synapse_api as api


def _run(coro):
    return asyncio.run(coro)


async def _start(app: web.Application) -> TestClient:
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def test_tools_mode_native_for_llama_cpp(monkeypatch):
    monkeypatch.setattr(api, "ENGINE", "llama.cpp")
    assert api._tools_mode() == "native"


def test_tools_mode_emulate_for_other_engines(monkeypatch):
    for engine in ("torch", "transformers", "diffusers", ""):
        monkeypatch.setattr(api, "ENGINE", engine)
        assert api._tools_mode() == "emulate"


def test_openai_tools_to_schemas():
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get the weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }}]
    schemas = api._openai_tools_to_schemas(tools)
    assert schemas == [{"name": "get_weather", "description": "Get the weather",
                        "input_schema": {"type": "object",
                                         "properties": {"city": {"type": "string"}}}}]


def test_openai_tools_to_schemas_empty():
    assert api._openai_tools_to_schemas(None) == []
    assert api._openai_tools_to_schemas([]) == []


def test_inject_tools_into_messages_creates_system_when_absent():
    messages = [{"role": "user", "content": "hi"}]
    schemas = [{"name": "get_weather", "description": "d", "input_schema": {}}]
    out = api._inject_tools_into_messages(messages, schemas, "test-model")
    assert out[0]["role"] == "system"
    assert "get_weather" in out[0]["content"]
    assert out[1] == {"role": "user", "content": "hi"}


def test_inject_tools_into_messages_appends_to_existing_system():
    messages = [{"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"}]
    schemas = [{"name": "get_weather", "description": "d", "input_schema": {}}]
    out = api._inject_tools_into_messages(messages, schemas, "test-model")
    assert len(out) == 2
    assert out[0]["role"] == "system"
    assert out[0]["content"].startswith("You are helpful.")
    assert "get_weather" in out[0]["content"]


def test_extract_tool_calls_parses_and_strips():
    text = 'Sure, let me check.\n<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    clean, calls = api._extract_tool_calls(text, [], "test-model")
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert calls[0]["input"] == {"city": "Paris"}
    assert "<tool_call>" not in clean


def test_extract_tool_calls_no_calls_returns_original_text():
    text = "Just a plain answer, no tools needed."
    clean, calls = api._extract_tool_calls(text, [], "test-model")
    assert calls == []
    assert clean == text


def test_calls_to_ollama_tool_calls_uses_object_arguments():
    calls = [{"id": "call_1", "name": "get_weather", "input": {"city": "Paris"}}]
    out = api._calls_to_ollama_tool_calls(calls)
    assert out == [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]


def test_calls_to_openai_tool_calls_uses_string_arguments():
    calls = [{"id": "call_1", "name": "get_weather", "input": {"city": "Paris"}}]
    out = api._calls_to_openai_tool_calls(calls)
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "get_weather"
    assert isinstance(out[0]["function"]["arguments"], str)
    assert json.loads(out[0]["function"]["arguments"]) == {"city": "Paris"}


def test_openai_tool_calls_to_ollama_converts_string_to_object():
    tcs = [{"id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]
    out = api._openai_tool_calls_to_ollama(tcs)
    assert out == [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]


def test_openai_tool_calls_to_ollama_handles_malformed_json():
    tcs = [{"function": {"name": "x", "arguments": "not json"}}]
    out = api._openai_tool_calls_to_ollama(tcs)
    assert out == [{"function": {"name": "x", "arguments": {}}}]


def test_openai_tool_calls_to_ollama_none_when_empty():
    assert api._openai_tool_calls_to_ollama(None) is None
    assert api._openai_tool_calls_to_ollama([]) is None


# ---------------------------------------------------------------------------
# Integration: /api/chat native mode (llama.cpp)
# ---------------------------------------------------------------------------

def test_ollama_chat_native_mode_forwards_tools_and_translates_response(monkeypatch):
    seen_upstream_req = {}

    async def _upstream(request):
        body = await request.json()
        seen_upstream_req.update(body)
        return web.json_response({
            "choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function",
                               "function": {"name": "get_weather",
                                            "arguments": '{"city": "Paris"}'}}],
            }}],
            "usage": {"completion_tokens": 5, "prompt_tokens": 10},
        })

    async def _go():
        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/chat/completions", _upstream)
        upstream = await _start(upstream_app)
        try:
            monkeypatch.setattr(api, "UPSTREAM", f"http://127.0.0.1:{upstream.port}")
            monkeypatch.setattr(api, "MODEL_NAME", "qwen3")
            monkeypatch.setattr(api, "ENGINE", "llama.cpp")
            proxy = await _start(api.build_app())
            try:
                resp = await proxy.post("/api/chat", json={
                    "model": "qwen3", "stream": False,
                    "messages": [{"role": "user", "content": "weather in paris?"}],
                    "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                })
                assert resp.status == 200
                data = await resp.json()
                assert seen_upstream_req["tools"]  # forwarded as-is
                tc = data["message"]["tool_calls"]
                assert tc == [{"function": {"name": "get_weather",
                                            "arguments": {"city": "Paris"}}}]
            finally:
                await proxy.close()
        finally:
            await upstream.close()

    _run(_go())


# ---------------------------------------------------------------------------
# Integration: /api/chat emulate mode (torch/transformers/diffusers)
# ---------------------------------------------------------------------------

def test_ollama_chat_emulate_mode_injects_and_parses(monkeypatch):
    seen_upstream_req = {}

    async def _upstream(request):
        body = await request.json()
        seen_upstream_req.update(body)
        # The "model" replies with a tool call embedded in plain text,
        # since this backend has no native tool-calling support.
        content = ('Let me check.\n<tool_call>{"name": "get_weather", '
                   '"arguments": {"city": "Paris"}}</tool_call>')
        return web.json_response({
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"completion_tokens": 5, "prompt_tokens": 10},
        })

    async def _go():
        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/chat/completions", _upstream)
        upstream = await _start(upstream_app)
        try:
            monkeypatch.setattr(api, "UPSTREAM", f"http://127.0.0.1:{upstream.port}")
            monkeypatch.setattr(api, "MODEL_NAME", "some-torch-model")
            monkeypatch.setattr(api, "ENGINE", "torch")
            proxy = await _start(api.build_app())
            try:
                resp = await proxy.post("/api/chat", json={
                    "model": "some-torch-model", "stream": True,  # must be forced non-stream
                    "messages": [{"role": "user", "content": "weather in paris?"}],
                    "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                })
                assert resp.status == 200
                # Must NOT be an ndjson stream — a single JSON object.
                data = await resp.json()
                assert seen_upstream_req["stream"] is False
                # System message must have been injected with tool defs.
                sys_msgs = [m for m in seen_upstream_req["messages"] if m["role"] == "system"]
                assert sys_msgs and "get_weather" in sys_msgs[0]["content"]
                # Response: tool_calls extracted, visible text cleaned.
                assert data["message"]["tool_calls"] == [
                    {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]
                assert "<tool_call>" not in data["message"]["content"]
            finally:
                await proxy.close()
        finally:
            await upstream.close()

    _run(_go())


# ---------------------------------------------------------------------------
# Integration: /v1/chat/completions emulate mode (the one non-passthrough case)
# ---------------------------------------------------------------------------

def test_openai_passthrough_emulate_mode_for_chat_completions(monkeypatch):
    async def _upstream(request):
        body = await request.json()
        assert body["stream"] is False
        assert "tools" not in body
        content = ('<tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}'
                   '</tool_call>')
        return web.json_response({
            "choices": [{"message": {"role": "assistant", "content": content},
                        "finish_reason": "stop"}],
        })

    async def _go():
        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/chat/completions", _upstream)
        upstream = await _start(upstream_app)
        try:
            monkeypatch.setattr(api, "UPSTREAM", f"http://127.0.0.1:{upstream.port}")
            monkeypatch.setattr(api, "MODEL_NAME", "some-torch-model")
            monkeypatch.setattr(api, "ENGINE", "transformers")
            proxy = await _start(api.build_app())
            try:
                resp = await proxy.post("/v1/chat/completions", json={
                    "model": "some-torch-model", "stream": False,
                    "messages": [{"role": "user", "content": "weather in rome?"}],
                    "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                })
                assert resp.status == 200
                data = await resp.json()
                choice = data["choices"][0]
                assert choice["finish_reason"] == "tool_calls"
                tc = choice["message"]["tool_calls"][0]
                assert tc["type"] == "function"
                assert json.loads(tc["function"]["arguments"]) == {"city": "Rome"}
            finally:
                await proxy.close()
        finally:
            await upstream.close()

    _run(_go())


def test_ollama_chat_native_streaming_buffers_tool_call_deltas(monkeypatch):
    """Ollama does not stream partial tool calls — the proxy must buffer
    OpenAI's per-index streamed tool_call deltas (name/arguments arrive in
    fragments across multiple SSE chunks) and emit ONE assembled tool_calls
    list on the final ndjson frame."""
    async def _upstream(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        frames = [
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"name": "get_weat"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"name": "her", "arguments": '{"city"'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": ': "Paris"}'}}]}}]},
            {"choices": [], "usage": {"completion_tokens": 3, "prompt_tokens": 5}},
        ]
        for f in frames:
            await resp.write(f"data: {json.dumps(f)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def _go():
        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/chat/completions", _upstream)
        upstream = await _start(upstream_app)
        try:
            monkeypatch.setattr(api, "UPSTREAM", f"http://127.0.0.1:{upstream.port}")
            monkeypatch.setattr(api, "MODEL_NAME", "qwen3")
            monkeypatch.setattr(api, "ENGINE", "llama.cpp")
            proxy = await _start(api.build_app())
            try:
                resp = await proxy.post("/api/chat", json={
                    "model": "qwen3", "stream": True,
                    "messages": [{"role": "user", "content": "weather in paris?"}],
                    "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                })
                assert resp.status == 200
                body = (await resp.read()).decode()
                lines = [ln for ln in body.splitlines() if ln.strip()]
                frames = [json.loads(ln) for ln in lines]
                # Exactly one frame carries the assembled tool call — Ollama
                # clients never see partial tool-call fragments.
                with_calls = [f for f in frames if "tool_calls" in f.get("message", {})]
                assert len(with_calls) == 1
                assert with_calls[0]["message"]["tool_calls"] == [
                    {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]
                assert with_calls[0]["done"] is True
            finally:
                await proxy.close()
        finally:
            await upstream.close()

    _run(_go())


def test_openai_passthrough_native_mode_is_pure_passthrough_even_with_tools(monkeypatch):
    """llama.cpp (native mode) must never route through the emulation
    layer — /v1/chat/completions stays a pure passthrough regardless of
    whether `tools` is present."""
    async def _upstream(request):
        body = await request.json()
        assert "tools" in body  # relayed byte-for-byte, not stripped
        return web.json_response({"choices": [{"message": {"role": "assistant",
                                                            "content": "no tool needed"}}]})

    async def _go():
        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/chat/completions", _upstream)
        upstream = await _start(upstream_app)
        try:
            monkeypatch.setattr(api, "UPSTREAM", f"http://127.0.0.1:{upstream.port}")
            monkeypatch.setattr(api, "MODEL_NAME", "qwen3")
            monkeypatch.setattr(api, "ENGINE", "llama.cpp")
            proxy = await _start(api.build_app())
            try:
                resp = await proxy.post("/v1/chat/completions", json={
                    "model": "qwen3", "stream": False,
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                })
                assert resp.status == 200
                data = await resp.json()
                assert data["choices"][0]["message"]["content"] == "no tool needed"
            finally:
                await proxy.close()
        finally:
            await upstream.close()

    _run(_go())

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_synapse_api's upstream error handling — the fix for the
"silent empty success on upstream error" class of bug: a 4xx/5xx (or a
transport failure) from llama-server used to be swallowed by _sse_lines
(which only ever yields `data:` lines) and turned into a clean
`{"done": true}` / empty-body 200 by the translation routes, and
openai_passthrough used to `prepare()` a 200 SSE response before even
opening the upstream connection.

Uses aiohttp.test_utils directly (no pytest-aiohttp plugin dependency) —
a real local TestServer stands in for llama-server so these exercise the
actual route handlers, not just the helper functions in isolation.
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
# _upstream_json / _upstream_stream — unit level
# ---------------------------------------------------------------------------

def test_upstream_json_raises_upstream_error_on_4xx():
    async def _handler(request):
        return web.json_response({"error": "bad request"}, status=400)

    async def _go():
        app = web.Application()
        app.router.add_post("/v1/completions", _handler)
        upstream = await _start(app)
        try:
            session_app = web.Application()
            session_app.on_startup.append(api._on_startup)
            session_app.on_cleanup.append(api._on_cleanup)
            client = await _start(session_app)
            try:
                url = f"http://127.0.0.1:{upstream.port}/v1/completions"
                with pytest.raises(api._UpstreamError) as exc_info:
                    await api._upstream_json("POST", url, "m", json={"prompt": "hi"})
                assert exc_info.value.status == 400
                assert b"bad request" in exc_info.value.body
            finally:
                await client.close()
        finally:
            await upstream.close()

    _run(_go())


def test_upstream_json_raises_upstream_error_on_transport_failure():
    async def _go():
        session_app = web.Application()
        session_app.on_startup.append(api._on_startup)
        session_app.on_cleanup.append(api._on_cleanup)
        client = await _start(session_app)
        try:
            # Nothing listens here — connection refused.
            url = "http://127.0.0.1:1/v1/completions"
            with pytest.raises(api._UpstreamError) as exc_info:
                await api._upstream_json("POST", url, "m", json={"prompt": "hi"})
            assert exc_info.value.status == 502
        finally:
            await client.close()

    _run(_go())


def test_upstream_stream_raises_before_yielding_on_4xx():
    async def _handler(request):
        return web.json_response({"error": "ctx overflow"}, status=400)

    async def _go():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", _handler)
        upstream = await _start(app)
        try:
            session_app = web.Application()
            session_app.on_startup.append(api._on_startup)
            session_app.on_cleanup.append(api._on_cleanup)
            client = await _start(session_app)
            try:
                url = f"http://127.0.0.1:{upstream.port}/v1/chat/completions"
                entered = False
                with pytest.raises(api._UpstreamError) as exc_info:
                    async with api._upstream_stream("POST", url, "m", json={}) as r:
                        entered = True
                assert not entered
                assert exc_info.value.status == 400
                assert b"ctx overflow" in exc_info.value.body
            finally:
                await client.close()
        finally:
            await upstream.close()

    _run(_go())


def test_upstream_status_maps_5xx_and_transport_failures_to_502():
    e_400 = api._UpstreamError(400, b"x", "m")
    e_404 = api._UpstreamError(404, b"x", "m")
    e_500 = api._UpstreamError(500, b"x", "m")
    assert api._upstream_status(e_400) == 400
    assert api._upstream_status(e_404) == 404
    assert api._upstream_status(e_500) == 502


def test_failure_body_uses_gb_synapse_failure_report(monkeypatch):
    monkeypatch.setattr("gb_synapse.failure_report", lambda model: f"engine for {model} is gone")
    body = api._failure_body("qwen3", detail="raw upstream bytes")
    assert body["error"] == "engine for qwen3 is gone"
    assert body["detail"] == "raw upstream bytes"


def test_failure_body_never_raises_when_gb_synapse_unavailable(monkeypatch):
    def _boom(model):
        raise RuntimeError("no run-state file")
    monkeypatch.setattr("gb_synapse.failure_report", _boom)
    body = api._failure_body("qwen3")
    assert "error" in body


# ---------------------------------------------------------------------------
# Integration: the translation routes must not turn an upstream error into a
# clean empty success.
# ---------------------------------------------------------------------------

def test_ollama_generate_nonstream_returns_error_not_empty_success(monkeypatch):
    async def _bad_upstream(request):
        return web.json_response({"error": "model still loading"}, status=503)

    async def _go():
        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/completions", _bad_upstream)
        upstream = await _start(upstream_app)
        try:
            monkeypatch.setattr(api, "UPSTREAM", f"http://127.0.0.1:{upstream.port}")
            monkeypatch.setattr(api, "MODEL_NAME", "qwen3")
            monkeypatch.setattr("gb_synapse.failure_report", lambda m: "still loading")
            proxy = await _start(api.build_app())
            try:
                resp = await proxy.post("/api/generate",
                                         json={"model": "qwen3", "prompt": "hi", "stream": False})
                assert resp.status >= 400
                data = await resp.json()
                assert data.get("response") != ""  # must not be the old empty-success shape
                assert "error" in data
            finally:
                await proxy.close()
        finally:
            await upstream.close()

    _run(_go())


def test_ollama_chat_stream_returns_error_before_any_frame(monkeypatch):
    """The core regression test: a bad upstream status must produce a
    non-200 JSON error with NOTHING prepared — not a 200 ndjson stream
    ending in {"done": true} with empty content."""
    async def _bad_upstream(request):
        return web.json_response({"error": "context length exceeded"}, status=400)

    async def _go():
        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/chat/completions", _bad_upstream)
        upstream = await _start(upstream_app)
        try:
            monkeypatch.setattr(api, "UPSTREAM", f"http://127.0.0.1:{upstream.port}")
            monkeypatch.setattr(api, "MODEL_NAME", "qwen3")
            monkeypatch.setattr("gb_synapse.failure_report", lambda m: "context length exceeded")
            proxy = await _start(api.build_app())
            try:
                resp = await proxy.post("/api/chat", json={
                    "model": "qwen3", "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                })
                assert resp.status == 400
                data = await resp.json()
                assert data["error"] == "context length exceeded"
            finally:
                await proxy.close()
        finally:
            await upstream.close()

    _run(_go())


def test_openai_passthrough_get_forwards_query_string_and_authorization():
    async def _echo(request):
        return web.json_response({
            "query": request.query_string,
            "auth": request.headers.get("Authorization", ""),
        })

    async def _go():
        upstream_app = web.Application()
        upstream_app.router.add_get("/v1/models", _echo)
        upstream = await _start(upstream_app)
        try:
            session_app_port = upstream.port
            import gb_synapse_api as api_mod
            api_mod.UPSTREAM = f"http://127.0.0.1:{session_app_port}"
            api_mod.MODEL_NAME = "qwen3"
            proxy = await _start(api_mod.build_app())
            try:
                resp = await proxy.get("/v1/models?foo=bar",
                                        headers={"Authorization": "Bearer xyz"})
                assert resp.status == 200
                data = await resp.json()
                assert data["query"] == "foo=bar"
                assert data["auth"] == "Bearer xyz"
            finally:
                await proxy.close()
        finally:
            await upstream.close()

    _run(_go())


def test_openai_passthrough_streaming_relays_upstream_error_status(monkeypatch):
    """Before the fix, this route `prepare()`d a 200 SSE response before
    even opening the upstream connection, so any upstream 4xx/5xx was
    relayed as a 200 with an empty body. Now status is checked first."""
    async def _bad(request):
        return web.json_response({"error": "bad tool schema"}, status=422)

    async def _go():
        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/chat/completions", _bad)
        upstream = await _start(upstream_app)
        try:
            monkeypatch.setattr(api, "UPSTREAM", f"http://127.0.0.1:{upstream.port}")
            monkeypatch.setattr(api, "MODEL_NAME", "qwen3")
            proxy = await _start(api.build_app())
            try:
                resp = await proxy.post("/v1/chat/completions",
                                         json={"model": "qwen3", "stream": True,
                                               "messages": [{"role": "user", "content": "hi"}]})
                assert resp.status == 422
                data = await resp.json()
                assert data["error"] == "bad tool schema"
            finally:
                await proxy.close()
        finally:
            await upstream.close()

    _run(_go())

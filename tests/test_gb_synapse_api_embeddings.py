#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_synapse_api.py's embeddings routes (P6) , the single largest
API hole for :11435 before this: no /v1/embeddings, /api/embed, or
/api/embeddings meant no RAG/reranker client could use gb-synapse at all.

_embed_upstream() reads gb_synapse's persisted run-state fresh on every
call (not a static CLI arg), so tests monkeypatch gb_synapse.ServerState /
_run_state_path / _pid_alive directly rather than needing a real second
llama-server process.
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


def _fake_run_state(tmp_path, monkeypatch, *, embed_port, embed_pid_alive=True):
    import gb_synapse as gs
    state_dir = tmp_path / "run"
    monkeypatch.setattr(gs, "RUN_DIR", state_dir)
    monkeypatch.setattr(gs, "_pid_alive", lambda pid: embed_pid_alive)
    st = gs.ServerState(model="qwen3", llama_pid=100, proxy_pid=200, port=11435,
                        internal_port=12435, tensor_split="", embed_pid=999,
                        embed_internal_port=embed_port)
    gs._write_run_state(st)


# ---------------------------------------------------------------------------
# _embed_upstream() / _embed_model_id()
# ---------------------------------------------------------------------------

def test_embed_upstream_none_when_no_run_state(tmp_path, monkeypatch):
    import gb_synapse as gs
    monkeypatch.setattr(gs, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(api, "MODEL_NAME", "qwen3")
    assert api._embed_upstream() is None


def test_embed_upstream_none_when_embed_pid_dead(tmp_path, monkeypatch):
    _fake_run_state(tmp_path, monkeypatch, embed_port=13435, embed_pid_alive=False)
    monkeypatch.setattr(api, "MODEL_NAME", "qwen3")
    assert api._embed_upstream() is None


def test_embed_upstream_resolves_when_configured_and_alive(tmp_path, monkeypatch):
    _fake_run_state(tmp_path, monkeypatch, embed_port=13435, embed_pid_alive=True)
    monkeypatch.setattr(api, "MODEL_NAME", "qwen3")
    assert api._embed_upstream() == "http://127.0.0.1:13435"


def test_embed_model_id_from_env(monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_EMBED_MODEL", "nomic-embed-text")
    assert api._embed_model_id() == "nomic-embed-text"


def test_embed_model_id_none_when_unset(monkeypatch):
    monkeypatch.delenv("GB_SYNAPSE_EMBED_MODEL", raising=False)
    assert api._embed_model_id() is None


# ---------------------------------------------------------------------------
# Routes — no embeddings engine configured
# ---------------------------------------------------------------------------

def test_v1_embeddings_returns_clear_error_when_unconfigured(monkeypatch):
    monkeypatch.setattr(api, "_embed_upstream", lambda: None)

    async def _go():
        proxy = await _start(api.build_app())
        try:
            resp = await proxy.post("/v1/embeddings", json={"input": "hello"})
            assert resp.status == 400
            data = await resp.json()
            assert "GB_SYNAPSE_EMBED_MODEL" in data["error"]
        finally:
            await proxy.close()

    _run(_go())


def test_api_embed_returns_clear_error_when_unconfigured(monkeypatch):
    monkeypatch.setattr(api, "_embed_upstream", lambda: None)

    async def _go():
        proxy = await _start(api.build_app())
        try:
            resp = await proxy.post("/api/embed", json={"input": "hello"})
            assert resp.status == 400
        finally:
            await proxy.close()

    _run(_go())


# ---------------------------------------------------------------------------
# Routes — embeddings engine configured, real upstream
# ---------------------------------------------------------------------------

async def _fake_embed_upstream_handler(request):
    body = await request.json()
    inputs = body["input"]
    if isinstance(inputs, str):
        inputs = [inputs]
    return web.json_response({
        "object": "list",
        "data": [{"object": "embedding", "index": i, "embedding": [0.1 * i, 0.2 * i]}
                 for i in range(len(inputs))],
        "model": body.get("model", ""),
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    })


def test_v1_embeddings_relays_openai_envelope(monkeypatch):
    async def _go():
        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/embeddings", _fake_embed_upstream_handler)
        upstream = await _start(upstream_app)
        try:
            monkeypatch.setattr(api, "_embed_upstream", lambda: f"http://127.0.0.1:{upstream.port}")
            monkeypatch.setattr(api, "_embed_model_id", lambda: "embed-model")
            proxy = await _start(api.build_app())
            try:
                resp = await proxy.post("/v1/embeddings",
                                        json={"input": ["a", "b"], "model": "embed-model"})
                assert resp.status == 200
                data = await resp.json()
                assert data["object"] == "list"
                assert len(data["data"]) == 2
            finally:
                await proxy.close()
        finally:
            await upstream.close()

    _run(_go())


def test_api_embed_returns_ollama_plural_shape(monkeypatch):
    async def _go():
        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/embeddings", _fake_embed_upstream_handler)
        upstream = await _start(upstream_app)
        try:
            monkeypatch.setattr(api, "_embed_upstream", lambda: f"http://127.0.0.1:{upstream.port}")
            monkeypatch.setattr(api, "_embed_model_id", lambda: "embed-model")
            proxy = await _start(api.build_app())
            try:
                resp = await proxy.post("/api/embed", json={"input": ["a", "b"]})
                assert resp.status == 200
                data = await resp.json()
                assert data["model"] == "embed-model"
                assert len(data["embeddings"]) == 2
                assert data["embeddings"][0] == [0.0, 0.0]
                assert data["embeddings"][1] == [0.1, 0.2]
            finally:
                await proxy.close()
        finally:
            await upstream.close()

    _run(_go())


def test_api_embeddings_legacy_returns_singular_shape(monkeypatch):
    async def _go():
        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/embeddings", _fake_embed_upstream_handler)
        upstream = await _start(upstream_app)
        try:
            monkeypatch.setattr(api, "_embed_upstream", lambda: f"http://127.0.0.1:{upstream.port}")
            monkeypatch.setattr(api, "_embed_model_id", lambda: "embed-model")
            proxy = await _start(api.build_app())
            try:
                resp = await proxy.post("/api/embeddings", json={"prompt": "hello"})
                assert resp.status == 200
                data = await resp.json()
                assert data == {"embedding": [0.0, 0.0]}
            finally:
                await proxy.close()
        finally:
            await upstream.close()

    _run(_go())


def test_api_embed_requires_input_field(monkeypatch):
    monkeypatch.setattr(api, "_embed_upstream", lambda: "http://127.0.0.1:1")

    async def _go():
        proxy = await _start(api.build_app())
        try:
            resp = await proxy.post("/api/embed", json={})
            assert resp.status == 400
        finally:
            await proxy.close()

    _run(_go())


def test_api_embeddings_legacy_requires_prompt_field(monkeypatch):
    monkeypatch.setattr(api, "_embed_upstream", lambda: "http://127.0.0.1:1")

    async def _go():
        proxy = await _start(api.build_app())
        try:
            resp = await proxy.post("/api/embeddings", json={})
            assert resp.status == 400
        finally:
            await proxy.close()

    _run(_go())


# ---------------------------------------------------------------------------
# /v1/models merge
# ---------------------------------------------------------------------------

def test_v1_models_merges_embed_model_when_configured(monkeypatch):
    async def _upstream_models(request):
        return web.json_response({"object": "list", "data": [{"id": "qwen3", "object": "model"}]})

    async def _go():
        upstream_app = web.Application()
        upstream_app.router.add_get("/v1/models", _upstream_models)
        upstream = await _start(upstream_app)
        try:
            monkeypatch.setattr(api, "UPSTREAM", f"http://127.0.0.1:{upstream.port}")
            monkeypatch.setattr(api, "MODEL_NAME", "qwen3")
            monkeypatch.setattr(api, "_embed_upstream", lambda: "http://127.0.0.1:9")
            monkeypatch.setattr(api, "_embed_model_id", lambda: "embed-model")
            proxy = await _start(api.build_app())
            try:
                resp = await proxy.get("/v1/models")
                assert resp.status == 200
                data = await resp.json()
                ids = {m["id"] for m in data["data"]}
                assert ids == {"qwen3", "embed-model"}
            finally:
                await proxy.close()
        finally:
            await upstream.close()

    _run(_go())


def test_v1_models_unchanged_when_no_embed_model(monkeypatch):
    async def _upstream_models(request):
        return web.json_response({"object": "list", "data": [{"id": "qwen3", "object": "model"}]})

    async def _go():
        upstream_app = web.Application()
        upstream_app.router.add_get("/v1/models", _upstream_models)
        upstream = await _start(upstream_app)
        try:
            monkeypatch.setattr(api, "UPSTREAM", f"http://127.0.0.1:{upstream.port}")
            monkeypatch.setattr(api, "MODEL_NAME", "qwen3")
            monkeypatch.setattr(api, "_embed_upstream", lambda: None)
            proxy = await _start(api.build_app())
            try:
                resp = await proxy.get("/v1/models")
                assert resp.status == 200
                data = await resp.json()
                assert [m["id"] for m in data["data"]] == ["qwen3"]
            finally:
                await proxy.close()
        finally:
            await upstream.close()

    _run(_go())

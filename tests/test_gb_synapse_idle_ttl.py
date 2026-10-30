#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Idle TTL for models served on demand (proxy side). Fakes only: no process is
started, no model is loaded."""
import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

import gb_synapse_api as api


def test_expiry_rules():
    assert not api._idle_expired(1000, 0, 0, 0)          # TTL 0 = pinned
    assert not api._idle_expired(100, 300, 0, 0)         # not yet
    assert api._idle_expired(301, 300, 0, 0)
    assert not api._idle_expired(1000, 300, 1, 0)        # in-flight request keeps it


def test_activity_middleware_tracks_inflight_and_resets(monkeypatch):
    seen = {}

    async def handler(req):
        seen["inflight"] = api._IDLE["inflight"]
        return web.Response(text="x")

    async def run():
        api._IDLE.update(inflight=0, last=0.0)
        await api._activity_middleware(make_mocked_request("POST", "/v1/chat/completions"), handler)
        return api._IDLE["inflight"], api._IDLE["last"]
    inflight, last = asyncio.run(run())
    assert seen["inflight"] == 1 and inflight == 0 and last > 0


def test_passive_paths_do_not_reset_timer():
    async def handler(req):
        return web.Response(text="ok")

    async def run():
        api._IDLE.update(inflight=0, last=5.0)
        await api._activity_middleware(make_mocked_request("GET", "/health"), handler)
        await api._activity_middleware(make_mocked_request("GET", "/api/ps"), handler)
    asyncio.run(run())
    assert api._IDLE["last"] == 5.0


def test_watch_unloads_once_via_stop_path(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "_idle_unload", lambda: calls.append(1))
    monkeypatch.setattr(api, "IDLE_TTL_S", 0.4)
    monkeypatch.setattr(api.time, "monotonic", lambda: 1000.0)
    api._IDLE.update(inflight=0, last=0.0)

    async def fast_sleep(_):
        return None
    monkeypatch.setattr(api.asyncio, "sleep", fast_sleep)
    asyncio.run(api._idle_watch(None))
    assert calls == [1]


def test_unload_spawns_detached_stop_for_this_model_only(monkeypatch):
    seen = {}
    monkeypatch.setattr(api, "MODEL_NAME", "m-x")
    monkeypatch.setattr(api.subprocess, "Popen", lambda cmd, **k: seen.update(cmd=cmd, k=k))
    api._idle_unload()
    assert "gb_synapse.stop('m-x')" in seen["cmd"][-1]
    assert seen["k"]["start_new_session"] is True


def test_no_watch_task_when_ttl_zero(monkeypatch):
    monkeypatch.setattr(api, "IDLE_TTL_S", 0.0)
    app = {}
    asyncio.run(api._start_idle_watch(app))
    assert "idle_task" not in app


def test_serve_passes_ttl_to_proxy_cmd(monkeypatch, tmp_path):
    import gb_synapse as gs
    captured = {}
    monkeypatch.setattr(gs.subprocess, "Popen", lambda cmd, **k: captured.setdefault("cmd", cmd) or type("P", (), {"pid": 1})())
    monkeypatch.setattr(gs, "_wait_proxy_ready", lambda *a, **k: None)
    monkeypatch.setattr(gs, "_run_log_path", lambda n: tmp_path / "log")
    entry = type("E", (), {"name": "m"})()
    gs._LOAD_CTX.idle_ttl_s = 300.0
    try:
        gs._start_proxy(entry, 1, 2, "llama.cpp")
    finally:
        gs._LOAD_CTX.idle_ttl_s = 0.0
    i = captured["cmd"].index("--idle-ttl-s")
    assert captured["cmd"][i + 1] == "300.0"

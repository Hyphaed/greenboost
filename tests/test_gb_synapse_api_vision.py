# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""The proxy accepts images of any size and brings them to the served model's token budget.

Before (2026-10-28): a request body over 1 MiB (aiohttp's default client_max_size) answered HTTP 413 with a bare
"Maximum request body size 1048576 exceeded", so every client grew its own resize, and got it wrong (575 tokens
made the vision model deny features that were in the picture).

A real local TestServer stands in for llama-server (same approach as test_gb_synapse_api_upstream.py).
"""
import asyncio
import base64
import io
import json

import pytest

pytest.importorskip("aiohttp")
pytest.importorskip("PIL")

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import gb_synapse_api as api
import gb_vision as gv


def _run(coro):
    return asyncio.run(coro)


def _budget():
    return gv.budget_from_profile(gv.profile_from_meta({"clip.projector_type": "qwen3vl_merger", "clip.vision.patch_size": 14,
                                                       "clip.vision.spatial_merge_size": 2}))


def _noisy_png(w, h):
    from PIL import Image
    buf = io.BytesIO()
    Image.effect_noise((w, h), 80).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _data_url(raw, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def _chat(url):
    return {"model": "m", "stream": False, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "describe"}, {"type": "image_url", "image_url": {"url": url}}]}]}


async def _start(app):
    c = TestClient(TestServer(app))
    await c.start_server()
    return c


async def _proxy_against(handler, monkeypatch, budget):
    """(proxy client, upstream client): the proxy forwards /v1/chat/completions to a recording upstream."""
    up_app = web.Application()
    up_app.router.add_post("/v1/chat/completions", handler)
    upstream = await _start(up_app)
    monkeypatch.setattr(api, "UPSTREAM", f"http://127.0.0.1:{upstream.port}")
    monkeypatch.setattr(api, "MODEL_NAME", "m")
    monkeypatch.setattr(api, "_vision_budget", lambda: budget)
    app = api.build_app()
    app.on_startup.append(api._on_startup)
    app.on_cleanup.append(api._on_cleanup)
    return await _start(app), upstream


def test_a_big_image_is_accepted_and_reaches_the_engine_at_the_token_budget(monkeypatch):
    seen = {}

    async def upstream(request):
        seen["body"] = await request.json()
        return web.json_response({"choices": [{"message": {"content": "ok"}}]})

    big = _noisy_png(1800, 1400)
    assert len(big) > (1 << 20), "the fixture must exceed the old 1 MiB limit"

    async def go():
        proxy, up = await _proxy_against(upstream, monkeypatch, _budget())
        try:
            r = await proxy.post("/v1/chat/completions", json=_chat(_data_url(big)))
            assert r.status == 200, await r.text()
        finally:
            await proxy.close(); await up.close()

    _run(go())
    from PIL import Image
    part = seen["body"]["messages"][0]["content"][1]["image_url"]["url"]
    assert part.startswith("data:image/jpeg;base64,")
    im = Image.open(io.BytesIO(base64.b64decode(part.split(",", 1)[1])))
    b = _budget()
    assert im.size[0] % 28 == 0 and im.size[1] % 28 == 0
    assert b.min_tokens <= gv.tokens_for(*im.size, b) <= b.max_tokens * 1.1
    assert seen["body"]["messages"][0]["content"][0] == {"type": "text", "text": "describe"}


def test_a_small_crop_is_upscaled_to_the_floor(monkeypatch):
    seen = {}

    async def upstream(request):
        seen["body"] = await request.json()
        return web.json_response({"choices": []})

    async def go():
        proxy, up = await _proxy_against(upstream, monkeypatch, _budget())
        try:
            r = await proxy.post("/v1/chat/completions", json=_chat(_data_url(_noisy_png(160, 380))))
            assert r.status == 200
        finally:
            await proxy.close(); await up.close()

    _run(go())
    from PIL import Image
    url = seen["body"]["messages"][0]["content"][1]["image_url"]["url"]
    im = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert gv.tokens_for(*im.size, _budget()) >= 1024


def test_a_text_only_model_gets_the_body_untouched(monkeypatch):
    seen = {}

    async def upstream(request):
        seen["raw"] = await request.read()
        return web.json_response({"choices": []})

    payload = json.dumps(_chat(_data_url(_noisy_png(200, 200)))).encode()

    async def go():
        proxy, up = await _proxy_against(upstream, monkeypatch, None)
        try:
            r = await proxy.post("/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"})
            assert r.status == 200
        finally:
            await proxy.close(); await up.close()

    _run(go())
    assert json.loads(seen["raw"]) == json.loads(payload)


def test_a_request_over_the_limit_gets_413_with_the_consequence(monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_MAX_BODY_MB", "1")

    async def upstream(request):
        raise AssertionError("an oversized request must never reach the engine")

    async def go():
        proxy, up = await _proxy_against(upstream, monkeypatch, _budget())
        try:
            r = await proxy.post("/v1/chat/completions", json=_chat(_data_url(_noisy_png(1800, 1400))))
            assert r.status == 413
            msg = (await r.json())["error"]["message"]
            assert "1 MiB" in msg and "nothing is broken" in msg and "GB_SYNAPSE_MAX_BODY_MB" in msg
        finally:
            await proxy.close(); await up.close()

    _run(go())


def test_normalise_returns_the_same_list_when_there_is_nothing_to_do(monkeypatch):
    monkeypatch.setattr(api, "_vision_budget", lambda: _budget())
    msgs = [{"role": "user", "content": "plain text"}]
    assert _run(api._normalize_images(msgs)) is msgs
    monkeypatch.setattr(api, "_vision_budget", lambda: None)
    img = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _data_url(_noisy_png(50, 50))}}]}]
    assert _run(api._normalize_images(img)) is img


def test_the_budget_comes_from_the_run_state_and_follows_its_mtime(tmp_path, monkeypatch):
    import gb_synapse as gs
    monkeypatch.setattr(api, "MODEL_NAME", "m")
    monkeypatch.setattr(gs, "_run_state_path", lambda model: tmp_path / "m.json")
    monkeypatch.setattr(api, "_BUDGET_CACHE", {"mtime": None, "budget": None})
    assert api._vision_budget() is None                                            # no run state yet
    (tmp_path / "m.json").write_text(json.dumps({"vision": {"budget": _budget().as_dict()}}))
    assert api._vision_budget() == _budget()
    (tmp_path / "m.json").write_text(json.dumps({"vision": {}}))
    import os, time
    os.utime(tmp_path / "m.json", (time.time() + 5, time.time() + 5))
    assert api._vision_budget() is None                                            # text-only now

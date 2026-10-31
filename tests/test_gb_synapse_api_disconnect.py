#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""A client that disconnects while the engine is still reading its prompt must reach the engine.

greenboost-cli closes its request on Esc/Ctrl-C. The proxy used to keep awaiting llama-server for a
client that was gone, so the engine finished the whole prompt first. With handler cancellation the
proxy's handler is cancelled on disconnect, which closes the upstream connection (an abort)."""
import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("aiohttp")

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gb_synapse_api as api


def test_run_server_turns_handler_cancellation_on(monkeypatch):
    seen = {}
    monkeypatch.setattr(web, "run_app", lambda app, **kw: seen.update(kw))
    api.run_server(object(), "127.0.0.1", 1234)
    assert seen["handler_cancellation"] is True and seen["port"] == 1234


def test_a_client_disconnect_during_prefill_cancels_the_upstream_request(monkeypatch):
    # aiohttp's TestServer cancels handlers on disconnect by default; run_server() (above)
    # is what turns the same on for the real server, where the default is off.
    async def scenario():
        upstream_cancelled = asyncio.Event()

        async def prefill(request):
            try:
                await asyncio.sleep(20)              # the engine reading a long prompt
            except asyncio.CancelledError:
                upstream_cancelled.set()
                raise
            return web.Response(text="late")

        up_app = web.Application()
        up_app.router.add_post("/v1/chat/completions", prefill)
        upstream = TestServer(up_app)
        await upstream.start_server()
        monkeypatch.setattr(api, "UPSTREAM", f"http://127.0.0.1:{upstream.port}")
        proxy = TestServer(api.build_app())
        await proxy.start_server()
        try:
            async with aiohttp.ClientSession() as s:
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        s.post(f"http://127.0.0.1:{proxy.port}/v1/chat/completions",
                               json={"model": "m", "stream": True,
                                     "messages": [{"role": "user", "content": "hi"}]}),
                        timeout=0.6)                 # the client gives up, i.e. Esc
            await asyncio.wait_for(upstream_cancelled.wait(), timeout=4)
        finally:
            await proxy.close()
            await upstream.close()

    asyncio.run(scenario())

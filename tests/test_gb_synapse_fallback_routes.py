#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_synapse_fallback.py's P6 fixes:

  * _flatten_content() degrades list-valued (multimodal) message content to
    text-only instead of crashing , the old `"\n".join(m.get("content",""))`
    in _apply_chat_template's except branch TypeErrored on a list, turning
    a vision-via-fallback request into an HTTP 500.
  * GET /v1/models , this was the one of the 4 gb-synapse backends missing
    it entirely.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("aiohttp")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gb_synapse_fallback as fb


def test_flatten_content_plain_string():
    assert fb._flatten_content("hello") == "hello"


def test_flatten_content_multimodal_list_extracts_text_only():
    content = [{"type": "text", "text": "what is in this image?"},
              {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,xyz"}}]
    assert fb._flatten_content(content) == "what is in this image?"


def test_flatten_content_multiple_text_parts_joined():
    content = [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]
    assert fb._flatten_content(content) == "part one\npart two"


def test_flatten_content_empty_or_none():
    assert fb._flatten_content(None) == ""
    assert fb._flatten_content([]) == ""


def test_apply_chat_template_does_not_crash_on_list_content(monkeypatch):
    """The actual regression: previously TypeErrored inside the except
    branch instead of degrading gracefully."""
    class _FakeTok:
        def apply_chat_template(self, *a, **kw):
            raise RuntimeError("no template for this tokenizer")

        def __call__(self, text, return_tensors=None):
            captured["text"] = text
            import torch
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    captured = {}
    monkeypatch.setattr(fb, "TOK", _FakeTok())
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,xyz"}},
    ]}]

    ids = fb._apply_chat_template(messages)  # must not raise

    assert captured["text"] == "describe this"
    import torch
    assert torch.is_tensor(ids)


def test_v1_models_route_registered():
    app = fb.build_app()
    routes = {r.resource.canonical for r in app.router.routes()}
    assert "/v1/models" in routes


def test_v1_models_response_shape(monkeypatch):
    import asyncio
    from aiohttp.test_utils import TestClient, TestServer

    monkeypatch.setattr(fb, "MODEL_NAME", "some-model")

    async def _go():
        client = TestClient(TestServer(fb.build_app()))
        await client.start_server()
        try:
            resp = await client.get("/v1/models")
            assert resp.status == 200
            data = await resp.json()
            assert data == {"object": "list", "data": [{"id": "some-model", "object": "model"}]}
        finally:
            await client.close()

    asyncio.run(_go())

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_synapse_api.py — thin translation proxy in front of a gb-synapse
llama-server instance.

Exposes three protocols on one port so existing tooling works unchanged:
  * Ollama-compatible:   /api/generate, /api/chat, /api/tags, /api/show, /api/ps
  * HuggingFace TGI:     /generate, /generate_stream
  * OpenAI (passthrough):/v1/*  — llama-server already speaks this natively,
                          so these routes are relayed byte-for-byte, streaming
                          included.
  * Slots (passthrough): /slots, /slots/{id}  — llama-server's native
                          prompt-cache save/restore API (not part of the
                          OpenAI surface), relayed the same way. Consumed by
                          greenboost-cli's /llamacache.

Launched as its own subprocess by gb_synapse.serve() (not imported — keeps
gb_synapse.py importable without aiohttp installed, and gives the proxy an
independent lifetime tracked in ServerState.proxy_pid).

Usage:
    python3 gb_synapse_api.py --port 11434 --upstream-port 12434 --model-name qwen3-coder
"""
from __future__ import annotations

import argparse
import json
import time

import aiohttp
from aiohttp import web

UPSTREAM = ""
MODEL_NAME = ""
SESSION: aiohttp.ClientSession | None = None


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _iso_from_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)) if ts else ""


async def _sse_lines(resp: aiohttp.ClientResponse):
    """Yield decoded `data: {...}` payloads from an upstream SSE stream,
    stopping at the terminal [DONE] sentinel."""
    async for raw in resp.content:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            return
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def _record_tok_s(model: str, t_first: "float | None", t_last: "float | None",
                  completion_tokens: int) -> None:
    """Compute client-observed decode tok/s from first→last token timing and
    the upstream's usage count, then hand it to gb_synapse.record_measured_tok_s
    (dataflux emit + rolling store). Proxy-side so ANY client on :11434 (curl,
    Zed, ai-forge's Ollama-compat steps) feeds recommend()'s fit estimator, not
    only greenboost-cli. Best-effort: never raises, silently skips incomplete
    samples (no usage, single token, zero interval)."""
    try:
        if completion_tokens <= 1 or t_first is None or t_last is None:
            return
        dt = t_last - t_first
        if dt <= 0:
            return
        # decode rate: tokens after the first, over the inter-token span
        tok_s = (completion_tokens - 1) / dt
        import gb_synapse
        gb_synapse.record_measured_tok_s(model, tok_s)
    except Exception:
        pass


def _usage_completion_tokens(chunk: dict, prev: int) -> int:
    """Pull completion_tokens from an OpenAI-style streamed chunk's `usage`
    (llama-server emits it in the final chunk when stream_options.include_usage
    is set). Keeps the last non-null value seen."""
    u = chunk.get("usage")
    if isinstance(u, dict) and u.get("completion_tokens") is not None:
        return int(u["completion_tokens"])
    return prev


# ---------------------------------------------------------------------------
# Ollama-compatible routes
# ---------------------------------------------------------------------------

def _ollama_opts(options: dict) -> dict:
    out = {}
    if "temperature" in options:
        out["temperature"] = options["temperature"]
    if "top_p" in options:
        out["top_p"] = options["top_p"]
    if "num_predict" in options:
        out["max_tokens"] = options["num_predict"]
    return out


def _ollama_format_to_response_format(fmt) -> "dict | None":
    """Translate Ollama's `format` request field into the OpenAI
    `response_format` field llama-server's /v1/* endpoints already natively
    support for grammar-constrained decoding (tools/server/server-common.cpp:
    response_format.type == "json_object" -> unconstrained JSON,
    "json_schema" -> response_format.json_schema.schema). Matches Ollama's
    own translation (llm/llama_server.go's llamaServerChatResponseFormat):
    "json" -> json_object, a JSON-Schema object -> json_schema. Without
    this, a `/api/generate`/`/api/chat` request with `format` silently got
    unconstrained output — the field was never read at all."""
    if not fmt:
        return None
    if fmt == "json":
        return {"type": "json_object"}
    if isinstance(fmt, dict):
        return {"type": "json_schema", "json_schema": {"name": "response", "schema": fmt}}
    return None


async def ollama_generate(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    model = body.get("model") or MODEL_NAME
    stream = body.get("stream", True)
    upstream_req = {"model": model, "prompt": body.get("prompt", ""), "stream": stream,
                     **_ollama_opts(body.get("options", {}) or {})}
    if stream:
        upstream_req["stream_options"] = {"include_usage": True}
    _response_format = _ollama_format_to_response_format(body.get("format"))
    if _response_format:
        upstream_req["response_format"] = _response_format
    url = f"{UPSTREAM}/v1/completions"

    if not stream:
        async with SESSION.post(url, json=upstream_req) as r:
            data = await r.json()
        text = (data.get("choices") or [{}])[0].get("text", "")
        usage = data.get("usage", {})
        return web.json_response({
            "model": model, "created_at": _iso_now(), "response": text, "done": True,
            "eval_count": usage.get("completion_tokens", 0),
            "prompt_eval_count": usage.get("prompt_tokens", 0),
        })

    resp = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
    await resp.prepare(request)
    t_first = t_last = None
    ctok = 0
    async with SESSION.post(url, json=upstream_req) as r:
        async for chunk in _sse_lines(r):
            piece = (chunk.get("choices") or [{}])[0].get("text", "")
            if piece:
                now = time.monotonic()
                if t_first is None:
                    t_first = now
                t_last = now
                await resp.write((json.dumps({"model": model, "created_at": _iso_now(),
                                               "response": piece, "done": False}) + "\n").encode())
            ctok = _usage_completion_tokens(chunk, ctok)
    await resp.write((json.dumps({"model": model, "created_at": _iso_now(),
                                   "response": "", "done": True}) + "\n").encode())
    await resp.write_eof()
    _record_tok_s(model, t_first, t_last, ctok)
    return resp


async def ollama_chat(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    model = body.get("model") or MODEL_NAME
    stream = body.get("stream", True)
    upstream_req = {"model": model, "messages": body.get("messages", []), "stream": stream,
                     **_ollama_opts(body.get("options", {}) or {})}
    if stream:
        upstream_req["stream_options"] = {"include_usage": True}
    _response_format = _ollama_format_to_response_format(body.get("format"))
    if _response_format:
        upstream_req["response_format"] = _response_format
    url = f"{UPSTREAM}/v1/chat/completions"

    if not stream:
        async with SESSION.post(url, json=upstream_req) as r:
            data = await r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return web.json_response({
            "model": model, "created_at": _iso_now(),
            "message": {"role": "assistant", "content": content},
            "done": True, "done_reason": "stop",
            "eval_count": usage.get("completion_tokens", 0),
            "prompt_eval_count": usage.get("prompt_tokens", 0),
        })

    resp = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
    await resp.prepare(request)
    t_first = t_last = None
    ctok = 0
    async with SESSION.post(url, json=upstream_req) as r:
        async for chunk in _sse_lines(r):
            piece = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content", "")
            if piece:
                now = time.monotonic()
                if t_first is None:
                    t_first = now
                t_last = now
                frame = {"model": model, "created_at": _iso_now(),
                         "message": {"role": "assistant", "content": piece}, "done": False}
                await resp.write((json.dumps(frame) + "\n").encode())
            ctok = _usage_completion_tokens(chunk, ctok)
    final = {"model": model, "created_at": _iso_now(),
             "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}
    await resp.write((json.dumps(final) + "\n").encode())
    await resp.write_eof()
    _record_tok_s(model, t_first, t_last, ctok)
    return resp


async def ollama_tags(request: web.Request) -> web.Response:
    import gb_synapse
    models = gb_synapse.list_models()
    return web.json_response({"models": [
        {"name": m.name, "model": m.name, "size": m.n_bytes, "digest": "",
         "modified_at": _iso_from_ts(m.added_ts),
         "details": {"family": "gguf", "quantization_level": m.quant}}
        for m in models
    ]})


async def ollama_show(request: web.Request) -> web.Response:
    body = await request.json()
    name = body.get("name") or body.get("model")
    import gb_synapse
    models = {m.name: m for m in gb_synapse.list_models()}
    m = models.get(name)
    if not m:
        return web.json_response({"error": f"model '{name}' not found"}, status=404)
    return web.json_response({
        "modelfile": "", "parameters": "", "template": "",
        "details": {"family": "gguf", "quantization_level": m.quant,
                    "parameter_size": f"{m.n_bytes / (1024 ** 3):.1f}GB"},
    })


async def ollama_ps(request: web.Request) -> web.Response:
    return web.json_response({"models": [
        {"name": MODEL_NAME, "model": MODEL_NAME, "size": 0, "expires_at": ""}
    ]})


# ---------------------------------------------------------------------------
# HuggingFace TGI-compatible routes
# ---------------------------------------------------------------------------

def _tgi_params(params: dict) -> dict:
    out = {}
    if "temperature" in params:
        out["temperature"] = params["temperature"]
    if "top_p" in params:
        out["top_p"] = params["top_p"]
    if "max_new_tokens" in params:
        out["max_tokens"] = params["max_new_tokens"]
    return out


async def tgi_generate(request: web.Request) -> web.Response:
    body = await request.json()
    upstream_req = {"model": MODEL_NAME, "prompt": body.get("inputs", ""), "stream": False,
                     **_tgi_params(body.get("parameters", {}) or {})}
    async with SESSION.post(f"{UPSTREAM}/v1/completions", json=upstream_req) as r:
        data = await r.json()
    text = (data.get("choices") or [{}])[0].get("text", "")
    return web.json_response({"generated_text": text})


async def tgi_generate_stream(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    upstream_req = {"model": MODEL_NAME, "prompt": body.get("inputs", ""), "stream": True,
                     **_tgi_params(body.get("parameters", {}) or {})}

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    acc = ""
    async with SESSION.post(f"{UPSTREAM}/v1/completions", json=upstream_req) as r:
        async for chunk in _sse_lines(r):
            piece = (chunk.get("choices") or [{}])[0].get("text", "")
            if not piece:
                continue
            acc += piece
            frame = {"token": {"text": piece, "special": False}, "generated_text": None}
            await resp.write(f"data:{json.dumps(frame)}\n\n".encode())
    await resp.write(f"data:{json.dumps({'token': None, 'generated_text': acc})}\n\n".encode())
    await resp.write_eof()
    return resp


# ---------------------------------------------------------------------------
# OpenAI /v1/* — raw passthrough (llama-server already speaks this natively)
# ---------------------------------------------------------------------------

async def openai_passthrough(request: web.Request) -> web.StreamResponse:
    path = request.match_info.get("path", "")
    url = f"{UPSTREAM}/v1/{path}"

    if request.method == "GET":
        async with SESSION.get(url) as r:
            data = await r.read()
            return web.Response(body=data, status=r.status, content_type=r.content_type)

    body = await request.read()
    headers = {"Content-Type": request.headers.get("Content-Type", "application/json")}
    try:
        stream = bool(json.loads(body).get("stream")) if body else False
    except json.JSONDecodeError:
        stream = False

    if not stream:
        async with SESSION.post(url, data=body, headers=headers) as r:
            data = await r.read()
            return web.Response(body=data, status=r.status, content_type=r.content_type)

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    async with SESSION.post(url, data=body, headers=headers) as r:
        async for raw_chunk in r.content.iter_any():
            await resp.write(raw_chunk)
    await resp.write_eof()
    return resp


# ---------------------------------------------------------------------------
# /slots — raw passthrough to llama-server's native prompt-cache slot API
# (GET /slots to list, POST /slots/{id}?action=save|restore|erase). Not part
# of the OpenAI surface, so it needs its own route alongside /v1/*.
# ---------------------------------------------------------------------------

async def slots_passthrough(request: web.Request) -> web.Response:
    tail = request.match_info.get("path", "")
    suffix = f"/{tail}" if tail else ""
    query = f"?{request.query_string}" if request.query_string else ""
    url = f"{UPSTREAM}/slots{suffix}{query}"

    if request.method == "GET":
        async with SESSION.get(url) as r:
            data = await r.read()
            return web.Response(body=data, status=r.status, content_type=r.content_type)

    body = await request.read()
    async with SESSION.post(url, data=body,
                             headers={"Content-Type": "application/json"}) as r:
        data = await r.read()
        return web.Response(body=data, status=r.status, content_type=r.content_type)


async def health(request: web.Request) -> web.Response:
    try:
        async with SESSION.get(f"{UPSTREAM}/health",
                                timeout=aiohttp.ClientTimeout(total=3)) as r:
            ok = r.status == 200
    except (aiohttp.ClientError, TimeoutError):
        ok = False
    return web.json_response({"status": "ok" if ok else "degraded",
                               "model": MODEL_NAME, "upstream": UPSTREAM})


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------

async def _on_startup(app: web.Application) -> None:
    global SESSION
    SESSION = aiohttp.ClientSession()


async def _on_cleanup(app: web.Application) -> None:
    if SESSION is not None:
        await SESSION.close()


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/api/generate", ollama_generate)
    app.router.add_post("/api/chat", ollama_chat)
    app.router.add_get("/api/tags", ollama_tags)
    app.router.add_post("/api/show", ollama_show)
    app.router.add_get("/api/ps", ollama_ps)
    app.router.add_post("/generate", tgi_generate)
    app.router.add_post("/generate_stream", tgi_generate_stream)
    app.router.add_route("*", "/v1/{path:.*}", openai_passthrough)
    app.router.add_route("*", "/slots", slots_passthrough)
    app.router.add_route("*", "/slots/{path:.*}", slots_passthrough)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


def main() -> None:
    global UPSTREAM, MODEL_NAME
    ap = argparse.ArgumentParser(description="gb-synapse API proxy")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--upstream-port", type=int, required=True)
    ap.add_argument("--model-name", required=True)
    args = ap.parse_args()

    UPSTREAM = f"http://127.0.0.1:{args.upstream_port}"
    MODEL_NAME = args.model_name
    web.run_app(build_app(), host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
    main()

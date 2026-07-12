#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_llm_server.py — minimal OpenAI-compatible server around gb_llm.py
(transformers + gb-quant). Used by gb_synapse.serve() as the "gbquant" engine
fallback when vLLM isn't installed.

Single global model, single-request-at-a-time — no continuous batching, that's
what vLLM is for (see gb_synapse._serve_vllm). Exposes just enough of the
OpenAI surface for gb_synapse_api.py's proxy to work unchanged: /health,
/v1/completions, /v1/chat/completions (both streaming and non-streaming).

Usage:
    python3 gb_llm_server.py --model org/repo --served-model-name name \
        --quant fp8 --host 127.0.0.1 --port 12434
"""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time

from aiohttp import web

import gb_llm

MODEL = None
TOK = None
MODEL_NAME = ""

# gb_quant._PRECISION_LADDER tokens for prefer_bits (16, "fp8", "nvfp4", 8, 4, "tq3", "tq2")
_QUANT_TO_PREFER_BITS = {"FP8": "fp8", "INT8": 8, "INT4": 4}


def _prefer_bits(quant: str):
    return _QUANT_TO_PREFER_BITS.get(quant.upper(), "fp8")


def _apply_chat_template(messages: list[dict]):
    import torch
    try:
        ids = TOK.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    except Exception:
        text = "\n".join(m.get("content", "") for m in messages)
        ids = TOK(text, return_tensors="pt")
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    return ids


def _generate_text(prompt_ids, max_tokens: int, temperature: float) -> tuple[str, int, int]:
    """Blocking single-shot generate — run via run_in_executor so it doesn't
    stall the event loop for other requests (e.g. /health) while it runs."""
    import torch
    prompt_ids = prompt_ids.to(MODEL.device)
    with torch.no_grad():
        out = MODEL.generate(prompt_ids, max_new_tokens=max_tokens,
                             do_sample=temperature > 0, temperature=max(temperature, 1e-4))
    completion_ids = out[0][prompt_ids.shape[1]:]
    text = TOK.decode(completion_ids, skip_special_tokens=True)
    return text, prompt_ids.shape[1], completion_ids.shape[0]


async def _async_stream_pieces(prompt_ids, max_tokens: int, temperature: float):
    """Bridges HF's thread-based TextIteratorStreamer into an async
    generator via an asyncio.Queue, so awaiting between pieces yields control
    back to the event loop instead of blocking it for the whole generation."""
    from transformers import TextIteratorStreamer
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    def _worker():
        streamer = TextIteratorStreamer(TOK, skip_special_tokens=True, skip_prompt=True)
        kwargs = dict(input_ids=prompt_ids.to(MODEL.device), max_new_tokens=max_tokens,
                      do_sample=temperature > 0, temperature=max(temperature, 1e-4),
                      streamer=streamer)
        error: "list[BaseException]" = []

        def _run_generate():
            try:
                MODEL.generate(**kwargs)
            except BaseException as exc:  # noqa: BLE001
                # generate() normally calls streamer.end() itself on completion;
                # on failure nothing does, so `for piece in streamer` below would
                # block forever on its internal queue.get() with no timeout.
                error.append(exc)
                streamer.end()

        gen_thread = threading.Thread(target=_run_generate)
        gen_thread.start()
        for piece in streamer:
            loop.call_soon_threadsafe(queue.put_nowait, piece)
        gen_thread.join()
        if error:
            loop.call_soon_threadsafe(queue.put_nowait, error[0])
        loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    threading.Thread(target=_worker, daemon=True).start()
    while True:
        piece = await queue.get()
        if piece is sentinel:
            break
        if isinstance(piece, BaseException):
            raise piece
        yield piece


def _sse_chat_chunk(model: str, piece: str, finish_reason: "str | None" = None) -> bytes:
    frame = {"id": "chatcmpl-gbquant", "object": "chat.completion.chunk",
             "created": int(time.time()), "model": model,
             "choices": [{"index": 0, "delta": ({"content": piece} if piece else {}),
                          "finish_reason": finish_reason}]}
    return f"data: {json.dumps(frame)}\n\n".encode()


def _sse_completion_chunk(model: str, piece: str, finish_reason: "str | None" = None) -> bytes:
    frame = {"id": "cmpl-gbquant", "object": "text_completion",
             "created": int(time.time()), "model": model,
             "choices": [{"index": 0, "text": piece, "finish_reason": finish_reason}]}
    return f"data: {json.dumps(frame)}\n\n".encode()


def _sse_usage_chunk(obj: str, model: str, prompt_tokens: int, completion_tokens: int) -> bytes:
    """Final usage-only chunk (choices: []), matching OpenAI's
    stream_options.include_usage shape — clients (greenboost-cli's tok/s
    footer) rely on this instead of falling back to a rough char-count
    estimate. TextIteratorStreamer only yields decoded text, not token
    counts, so the caller re-tokenizes the accumulated output once at the
    end to get an exact completion_tokens value."""
    frame = {"id": "chatcmpl-gbquant" if obj == "chat.completion.chunk" else "cmpl-gbquant",
             "object": obj, "created": int(time.time()), "model": model, "choices": [],
             "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                       "total_tokens": prompt_tokens + completion_tokens}}
    return f"data: {json.dumps(frame)}\n\n".encode()


async def chat_completions(request: web.Request) -> web.StreamResponse:
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return web.json_response({"error": {"message": "invalid JSON body"}}, status=400)
    max_tokens = int(body.get("max_tokens") or 256)
    temperature = float(body.get("temperature", 0.7))
    stream = bool(body.get("stream", False))
    prompt_ids = _apply_chat_template(body.get("messages", []))

    if not stream:
        loop = asyncio.get_event_loop()
        try:
            text, n_prompt, n_completion = await loop.run_in_executor(
                None, _generate_text, prompt_ids, max_tokens, temperature)
        except Exception as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=500)
        return web.json_response({
            "id": "chatcmpl-gbquant", "object": "chat.completion", "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_completion,
                      "total_tokens": n_prompt + n_completion},
        })

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    out_text = ""
    try:
        async for piece in _async_stream_pieces(prompt_ids, max_tokens, temperature):
            out_text += piece
            await resp.write(_sse_chat_chunk(MODEL_NAME, piece))
    except Exception as exc:
        await resp.write(_sse_chat_chunk(MODEL_NAME, f"[error: {exc}]", finish_reason="error"))
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp
    await resp.write(_sse_chat_chunk(MODEL_NAME, "", finish_reason="stop"))
    n_completion = len(TOK(out_text)["input_ids"]) if out_text else 0
    await resp.write(_sse_usage_chunk("chat.completion.chunk", MODEL_NAME,
                                       prompt_ids.shape[1], n_completion))
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


async def completions(request: web.Request) -> web.StreamResponse:
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return web.json_response({"error": {"message": "invalid JSON body"}}, status=400)
    max_tokens = int(body.get("max_tokens") or 256)
    temperature = float(body.get("temperature", 0.7))
    stream = bool(body.get("stream", False))
    prompt_ids = TOK(body.get("prompt", ""), return_tensors="pt")["input_ids"]

    if not stream:
        loop = asyncio.get_event_loop()
        try:
            text, n_prompt, n_completion = await loop.run_in_executor(
                None, _generate_text, prompt_ids, max_tokens, temperature)
        except Exception as exc:
            return web.json_response({"error": {"message": str(exc)}}, status=500)
        return web.json_response({
            "id": "cmpl-gbquant", "object": "text_completion", "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_completion,
                      "total_tokens": n_prompt + n_completion},
        })

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    out_text = ""
    try:
        async for piece in _async_stream_pieces(prompt_ids, max_tokens, temperature):
            out_text += piece
            await resp.write(_sse_completion_chunk(MODEL_NAME, piece))
    except Exception as exc:
        await resp.write(_sse_completion_chunk(MODEL_NAME, f"[error: {exc}]", finish_reason="error"))
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp
    await resp.write(_sse_completion_chunk(MODEL_NAME, "", finish_reason="stop"))
    n_completion = len(TOK(out_text)["input_ids"]) if out_text else 0
    await resp.write(_sse_usage_chunk("text_completion", MODEL_NAME,
                                       prompt_ids.shape[1], n_completion))
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok" if MODEL is not None else "loading", "model": MODEL_NAME})


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_post("/v1/completions", completions)
    return app


def main() -> None:
    global MODEL, TOK, MODEL_NAME
    ap = argparse.ArgumentParser(description="gb-synapse transformers+gb-quant server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-model-name", required=True)
    ap.add_argument("--quant", default="fp8")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()

    MODEL_NAME = args.served_model_name
    bits = _prefer_bits(args.quant)
    print(f"[gb_llm_server] loading {args.model} (gb-quant prefer_bits={bits!r})...", flush=True)
    MODEL, TOK = gb_llm.load_causal_lm(args.model, prefer_bits=bits)
    print(f"[gb_llm_server] ready, serving '{MODEL_NAME}' on {args.host}:{args.port}", flush=True)

    web.run_app(build_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()

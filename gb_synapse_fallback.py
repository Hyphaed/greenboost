#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_synapse_fallback.py — single-request OpenAI-compatible server for models
`SynapseTorchBackend` can't hand to the synapse torch engine (gLLM):
architecture not recognized, quantization already below the fp8 floor with
no matching engine loader yet, or the checkpoint simply doesn't fit the
live VRAM+T2 budget in bf16. Runs on the synapse torch engine's own venv
python (`greenboost.pth` there makes `gb_quant`/`gb_init` importable), so
no extra install step exists for this fallback — it's the same venv,
loaded on demand by `SynapseTorchBackend._torch_serve_mode()`.

Absorbs `gb_llm.py` (transformers + gb-quant quantize-to-fit: bf16 > int8 >
int4) and `gb_llm_server.py` (the aiohttp OpenAI surface) into one file, so
the fallback path has a single home instead of two separate modules whose
only consumer was each other. `gb_llm.py`/`gb_llm_server.py` themselves
stay in place until a later phase (external consumers still reference
them directly — see gb-synapse-unification workflow/task_to_develop.md).

Single global model, single-request-at-a-time — no continuous batching,
that's what the synapse torch engine (gLLM) is for. Exposes just enough of
the OpenAI surface for gb_synapse_api.py's proxy to work unchanged:
/health, /v1/completions, /v1/chat/completions (streaming and non-).

Usage:
    python3 gb_synapse_fallback.py --model org/repo --served-model-name name \
        --quant fp8 --host 127.0.0.1 --port 12434
"""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time

import torch
from aiohttp import web

import gb_quant

# Bootstrap all GreenBoost layers; no-op when not active.
try:
    import gb_init as _gb_init
except ImportError:
    _gb_init = None

MODEL = None
TOK = None
MODEL_NAME = ""

# gb_quant._PRECISION_LADDER tokens for prefer_bits (16, "fp8", "nvfp4", 8, 4, "tq3", "tq2")
_QUANT_TO_PREFER_BITS = {"FP8": "fp8", "INT8": 8, "INT4": 4}


def _prefer_bits(quant: str):
    return _QUANT_TO_PREFER_BITS.get(quant.upper(), "fp8")


def _auto_budget_gb() -> float:
    """Use gb_init unified budget (telemetry-first, torch fallback)."""
    if _gb_init is not None:
        return _gb_init.auto_budget_gb()
    try:
        free_b, _ = torch.cuda.mem_get_info()
        return free_b / 2**30 * 0.92
    except Exception:
        return 0.0


def load_causal_lm(model_id: str, budget_gb: "float | None" = None,
                   device: str = "cuda", dtype=torch.bfloat16,
                   cache_dir: "str | None" = None, trust_remote_code: bool = False,
                   prefer_bits: "int | str" = 4, verbose: bool = True, **hf_kwargs):
    """Load an HF causal LM quantized-to-fit through the gb-quant layer.

    Loads on CPU first (never OOMs the GPU on load), plans per-component
    precision against `budget_gb` (default: 92% of currently free VRAM), then
    realises the plan layer-by-layer onto `device`. Returns (model, tokenizer).

    `prefer_bits` is the floor precision plan_fit won't drop below (16,
    "fp8", "nvfp4", 8, 4, "tq3", "tq2" — see gb_quant._PRECISION_LADDER);
    default 4 matches gb_quant's own default.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Pre-inference check: ECC errors and VRAM headroom via telemetry singleton.
    if _gb_init is not None:
        _gb_init.pre_inference_check()

    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir,
                                        trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="cpu", low_cpu_mem_usage=True,
        cache_dir=cache_dir, trust_remote_code=trust_remote_code, **hf_kwargs,
    )
    if budget_gb is None:
        budget_gb = _auto_budget_gb()

    # NVTX range so the quantize+load span is visible in Nsight Systems.
    _gs = _gb_init.get_stream_sched() if _gb_init else None
    _nvtx_ctx = None
    if _gs is not None:
        try:
            import nvtx
            _nvtx_ctx = nvtx.annotate(
                message=f"gb:llm_load:{model_id.split('/')[-1]}",
                color="orange", domain="GreenBoost",
            )
            _nvtx_ctx.__enter__()
        except Exception:
            _nvtx_ctx = None

    gb_quant.quantize_to_fit(model, budget_gb=budget_gb, device=device,
                             dtype=dtype, prefer_bits=prefer_bits, verbose=verbose)
    # Move what the plan kept in bf16 (embeddings, norms, lm_head) to the GPU
    # on the transfer stream for non-blocking overlap.
    if _gs is not None:
        with _gs.on("transfer"):
            model.to(device, non_blocking=True)
        _gs.wait_for("transfer", on="gemm")
    else:
        model.to(device)
    model.eval()

    if _nvtx_ctx is not None:
        try:
            _nvtx_ctx.__exit__(None, None, None)
        except Exception:
            pass

    if verbose and _gb_init is not None:
        m = _gb_init.snapshot()
        if m is not None:
            print(
                f"[gb_synapse_fallback] loaded — VRAM {m.fb_used_mb}/{m.fb_total_mb} MB "
                f"({m.fb_used_pct:.0f}%)  pwr={m.power_w:.0f}W  "
                f"temp={m.temp_c:.0f}°C",
                flush=True,
            )
    return model, tok


def generate(model, tok, prompt: str, max_new_tokens: int = 128,
             temperature: float = 0.7, **gen_kwargs) -> str:
    """Small convenience wrapper for smoke tests and scripts."""
    messages = [{"role": "user", "content": prompt}]
    try:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt")
    except Exception:
        ids = tok(prompt, return_tensors="pt")
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    ids = ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens,
                             do_sample=temperature > 0, temperature=temperature,
                             **gen_kwargs)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def _flatten_content(content) -> str:
    """`content` is normally a plain string, but the proxy's vision
    translation (gb_synapse_api._ollama_messages_to_openai) always produces
    OpenAI's multimodal list-of-parts shape ([{"type":"text",...},
    {"type":"image_url",...}]) for any request carrying images. This
    fallback is a plain-text HF generate() path with no vision
    preprocessing at all, so an image part is dropped here (degrading to
    text-only) rather than crashing — the previous behavior TypeErrored
    inside apply_chat_template's own except branch ("\n".join() on a
    list), turning a vision-via-fallback request into an HTTP 500 instead
    of a degraded text-only answer."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") == "text")
    return str(content) if content else ""


def _apply_chat_template(messages: list[dict]):
    try:
        ids = TOK.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    except Exception:
        text = "\n".join(_flatten_content(m.get("content", "")) for m in messages)
        ids = TOK(text, return_tensors="pt")
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    return ids


def _generate_text(prompt_ids, max_tokens: int, temperature: float) -> tuple[str, int, int]:
    """Blocking single-shot generate — run via run_in_executor so it doesn't
    stall the event loop for other requests (e.g. /health) while it runs."""
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
    frame = {"id": "chatcmpl-gbfallback", "object": "chat.completion.chunk",
             "created": int(time.time()), "model": model,
             "choices": [{"index": 0, "delta": ({"content": piece} if piece else {}),
                          "finish_reason": finish_reason}]}
    return f"data: {json.dumps(frame)}\n\n".encode()


def _sse_completion_chunk(model: str, piece: str, finish_reason: "str | None" = None) -> bytes:
    frame = {"id": "cmpl-gbfallback", "object": "text_completion",
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
    frame = {"id": "chatcmpl-gbfallback" if obj == "chat.completion.chunk" else "cmpl-gbfallback",
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
            "id": "chatcmpl-gbfallback", "object": "chat.completion", "created": int(time.time()),
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
            "id": "cmpl-gbfallback", "object": "text_completion", "created": int(time.time()),
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


async def models(request: web.Request) -> web.Response:
    """GET /v1/models — this backend was the one of the 4 gb-synapse
    backends missing it entirely (gb_diffusion_server.py has the same
    route), so `:11369/v1/models` 404s exactly when the transformers
    fallback is the last-resort backend actually serving."""
    return web.json_response({"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_post("/v1/completions", completions)
    return app


def main() -> None:
    global MODEL, TOK, MODEL_NAME
    ap = argparse.ArgumentParser(description="gb-synapse torch-core fallback server "
                                             "(transformers + gb-quant, single-request)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-model-name", required=True)
    ap.add_argument("--quant", default="fp8")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()

    MODEL_NAME = args.served_model_name
    bits = _prefer_bits(args.quant)
    print(f"[gb_synapse_fallback] loading {args.model} (gb-quant prefer_bits={bits!r})...", flush=True)
    MODEL, TOK = load_causal_lm(args.model, prefer_bits=bits)
    print(f"[gb_synapse_fallback] ready, serving '{MODEL_NAME}' on {args.host}:{args.port}", flush=True)

    web.run_app(build_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()

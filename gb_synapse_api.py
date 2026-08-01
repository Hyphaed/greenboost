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
    python3 gb_synapse_api.py --port 11435 --upstream-port 12435 --model-name qwen3-coder
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import time

import aiohttp
from aiohttp import web

UPSTREAM = ""
MODEL_NAME = ""
ENGINE = ""
SESSION: aiohttp.ClientSession | None = None

# How long a streaming response may sit with zero output before it's
# flagged as a stall — see _StallWatch.
STALL_THRESHOLD_S = float(os.environ.get("GB_SYNAPSE_STALL_THRESHOLD_S", "120"))

# Idle-between-bytes limit for upstream requests (generation can legitimately
# run for many minutes; what must be bounded is silence, not wall-clock).
# GB_SYNAPSE_SOCK_READ_S overrides. total=None removes aiohttp's default 300s
# hard kill, which used to abort any generation (or still-loading model) past
# 5 minutes regardless of whether bytes were still arriving.
SOCK_READ_S = float(os.environ.get("GB_SYNAPSE_SOCK_READ_S", "900"))


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _iso_from_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)) if ts else ""


async def _sse_lines(resp: aiohttp.ClientResponse):
    """Yield decoded `data: {...}` payloads from an upstream SSE stream,
    stopping at the terminal [DONE] sentinel. Only ever called on a response
    whose status has already been verified <400 by the caller (_upstream_stream)
    — a mid-stream transport failure (connection drop) propagates out of the
    `async for` as an aiohttp exception rather than being swallowed here."""
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


class _UpstreamError(Exception):
    """Raised by _upstream_json/_upstream_stream when the upstream engine
    returned a non-2xx status or a transport-level failure occurred (refused
    connection, timeout, dropped mid-response). Callers convert this into a
    client-facing error body via _failure_body() instead of letting an empty
    "done": true success respond, or a bare aiohttp exception propagate."""

    def __init__(self, status: int, body: bytes, model: str):
        self.status = status
        self.body = body
        self.model = model
        super().__init__(f"upstream {status} for {model!r}: {body[:200]!r}")


def _failure_body(model: str, detail: str = "") -> dict:
    """Ollama-shaped error body sourced from gb_synapse.failure_report() —
    "the engine is alive but closed mid-response" / "the engine died" /
    "no running server", instead of an httpx/aiohttp exception name a client
    can't act on."""
    try:
        import gb_synapse
        reason = gb_synapse.failure_report(model)
    except Exception:
        reason = "gb-synapse proxy could not reach the upstream engine."
    body = {"error": reason}
    if detail:
        body["detail"] = detail
    return body


def _upstream_status(e: "_UpstreamError") -> int:
    """Map an upstream failure to the status returned to the client: relay
    a genuine 4xx as-is (bad request), fold everything else (5xx, transport
    failure) into 502 — we are a proxy, the upstream is what actually broke."""
    return e.status if 400 <= e.status < 500 else 502


async def _upstream_json(method: str, url: str, model: str, **kw) -> dict:
    """Non-streaming upstream call, used by the 4 translation (non-passthrough)
    routes. Raises _UpstreamError on status>=400 or a transport failure —
    never lets a caller construct a "done": true success body from an error."""
    try:
        async with SESSION.request(method, url, **kw) as r:
            raw = await r.read()
            if r.status >= 400:
                raise _UpstreamError(r.status, raw, model)
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise _UpstreamError(502, raw, model) from e
    except _UpstreamError:
        raise
    except (aiohttp.ClientError, TimeoutError) as e:
        raise _UpstreamError(502, str(e).encode(), model) from e


@contextlib.asynccontextmanager
async def _upstream_stream(method: str, url: str, model: str, **kw):
    """Open a streaming upstream request and only yield it once the response
    status is known to be <400 — status arrives with the HTTP headers,
    strictly before any SSE body, so this is available before the caller has
    touched the client-facing StreamResponse at all. That ordering is what
    lets a bad status become a clean JSON error response instead of a 200
    that was already `prepare()`d before the upstream was even known-good."""
    try:
        async with SESSION.request(method, url, **kw) as r:
            if r.status >= 400:
                raw = await r.read()
                raise _UpstreamError(r.status, raw, model)
            yield r
    except _UpstreamError:
        raise
    except (aiohttp.ClientError, TimeoutError) as e:
        raise _UpstreamError(502, str(e).encode(), model) from e


def _record_tok_s(model: str, t_first: "float | None", t_last: "float | None",
                  completion_tokens: int) -> None:
    """Compute client-observed decode tok/s from first→last token timing and
    the upstream's usage count, then hand it to gb_synapse.record_measured_tok_s
    (dataflux emit + rolling store). Proxy-side so ANY client on the gb-synapse
    port (curl, Zed, ai-forge's Ollama-compat steps) feeds recommend()'s fit
    estimator, not only greenboost-cli. Best-effort: never raises, silently
    skips incomplete samples (no usage, single token, zero interval)."""
    try:
        if completion_tokens <= 1 or t_first is None or t_last is None:
            return
        dt = t_last - t_first
        if dt <= 0:
            return
        # decode rate: tokens after the first, over the inter-token span
        tok_s = (completion_tokens - 1) / dt
        import gb_synapse
        gb_synapse.record_measured_tok_s(model, tok_s, source="proxy")
    except Exception:
        pass


def _record_prompt_cache(model: str, t_start: float, t_first: "float | None",
                         prompt_tokens: int, tokens_cached: "int | None") -> None:
    """Companion to _record_tok_s: TTFT (t_start->t_first) + reused-vs-total
    prompt token share, from llama-server's own `tokens_cached`/`timings`
    fields (see _cache_info_from_chunk). Feeds GB-Semantics' ttft_ms/
    prompt_cache_hit_pct metrics (semantics/metrics.yaml) — the measurement
    the --cache-ram/--cache-idle-slots actuation (gb_synapse_backends.py)
    exists to move. Best-effort, never raises."""
    try:
        ttft_ms = (t_first - t_start) * 1000 if t_first is not None else None
        hit_pct = (100.0 * tokens_cached / prompt_tokens
                   if tokens_cached is not None and prompt_tokens else None)
        import gb_synapse
        gb_synapse.record_prompt_cache_sample(model, ttft_ms, hit_pct, tokens_cached or 0)
    except Exception:
        pass


def _cache_info_from_chunk(chunk: dict, prev: "int | None") -> "int | None":
    """Pull the reused-prompt-token count from llama-server's own response
    fields when present (typically only the final SSE frame carries it) —
    best-effort, keeps the last non-null value seen, same shape as
    _usage_counts. Two real shapes, live-verified 2026-07-29 against this
    engine's actual OpenAI-compat streaming output:
      - top-level `tokens_cached` — llama-server's native /completion
        endpoint (server-task.cpp's to_json_non_oaicompat()).
      - `usage.prompt_tokens_details.cached_tokens` — the OpenAI-compat
        /v1/chat/completions shape GB-CLI's BACKEND_REGISTRY actually talks
        to (openai_passthrough(), not the Ollama-compat routes) — the one
        that matters for real GB-CLI traffic; this was checked-in without
        the nested lookup and silently produced reused_tokens=0/no hit_pct
        on every real request until this fix."""
    tc = chunk.get("tokens_cached")
    if tc is None:
        tc = (chunk.get("usage") or {}).get("prompt_tokens_details", {}).get("cached_tokens")
    return int(tc) if tc is not None else prev


def _parse_sse_telemetry_timed(
        chunks: "list[tuple[float, bytes]]") -> "tuple[int, int, int | None, float | None, float | None]":
    """Best-effort (completion_tokens, prompt_tokens, tokens_cached, t_first,
    t_last) from a list of (arrival_timestamp, raw_chunk) pairs — used by
    openai_passthrough(), which forwards raw_chunk bytes to the client
    UNCHANGED and separately observes the same bytes here purely for
    telemetry. A parse failure here must never affect what was already
    forwarded, so this never raises; malformed/partial lines are silently
    skipped. t_first/t_last are latched on the SSE event that actually
    carries generated content, not on whichever raw TCP chunk arrived
    first —
    actually carries generated content (delta.content or
    delta.reasoning_content — GB-CLI's own orchestrator.py counts both as
    "generation started", see StreamFragment/ReasoningFragment), not on
    whichever raw TCP chunk happened to arrive first at the transport layer.

    Real incident (2026-08-01): openai_passthrough()'s old t_first latched on
    ANY non-empty raw_chunk, which counts llama.cpp's role-only opening delta
    (`{"delta":{"role":"assistant"}}`, no content yet) as "decode started".
    For a turn with heavy prompt-eval (fills most of the request), that
    opening delta can arrive within the SAME first chunk as the very start of
    streaming — but the true decode span (last token minus FIRST real token)
    is what _record_tok_s needs. Live-verified against llama-server's own
    timing block for the incident that prompted this fix: engine truth was
    2.18 tok/s (43.2s / 94 tokens); the old latch reported 0.3 tok/s (~310s
    span) — reasoning/role frames arriving well before first real content
    inflated the measured span roughly 7x.

    Never raises — a parse failure just means (None, None) for the
    timestamps, same fallback shape _record_tok_s already handles."""
    ctok = ptok = 0
    tokens_cached = None
    t_first = t_last = None
    partial = ""
    for ts, raw in chunks:
        try:
            partial += raw.decode("utf-8", "ignore")
        except Exception:
            continue
        while "\n" in partial:
            line, partial = partial.split("\n", 1)
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]" or not payload:
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ctok, ptok = _usage_counts(chunk, (ctok, ptok))
            tokens_cached = _cache_info_from_chunk(chunk, tokens_cached)
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                if t_first is None:
                    t_first = ts
                t_last = ts
    return ctok, ptok, tokens_cached, t_first, t_last


def _usage_counts(chunk: dict, prev: "tuple[int, int]") -> "tuple[int, int]":
    """Pull (completion_tokens, prompt_tokens) from an OpenAI-style streamed
    chunk's `usage` (llama-server emits it in the final chunk when
    stream_options.include_usage is set). Keeps the last non-null values seen."""
    u = chunk.get("usage")
    if not isinstance(u, dict):
        return prev
    c = u.get("completion_tokens")
    p = u.get("prompt_tokens")
    return (int(c) if c is not None else prev[0], int(p) if p is not None else prev[1])


def _ollama_durations(t_start: float, t_first: "float | None", t_last: "float | None",
                       ctok: int, ptok: int) -> dict:
    """Real Ollama's streaming final frame carries eval_count/prompt_eval_count
    plus total/load/prompt_eval/eval durations in nanoseconds — clients that
    compute tok/s from eval_count/eval_duration (LangChain, OpenWebUI,
    ai-forge's own metrics) got zeros for all of these before, since the proxy
    only ever sent eval_count/prompt_eval_count on the non-streaming path."""
    t_end = time.monotonic()
    total_ns = int(max(t_end - t_start, 0.0) * 1e9)
    prompt_eval_ns = int(max((t_first - t_start), 0.0) * 1e9) if t_first is not None else 0
    eval_ns = int(max((t_last - t_first), 0.0) * 1e9) if t_first is not None and t_last is not None else 0
    return {
        "eval_count": ctok, "prompt_eval_count": ptok,
        "total_duration": total_ns, "load_duration": 0,
        "prompt_eval_duration": prompt_eval_ns, "eval_duration": eval_ns,
    }


class _StallWatch:
    """Closes the "HTTP already healthy, engine hung forever" gap
    (workflow/gb-synapse.md: torch-core bring-up, `/health` returned 200 but
    the first real request hung on a CUDA error during prefill — the client
    just sees a silent hang, never an error). A streaming request's own loop
    can't notice its own stall (it's what's stuck); this runs a background
    watchdog task alongside it that polls real idle time (time since the last
    `mark()`) and fires via gb_synapse.emit_stall whenever that idle time
    crosses STALL_THRESHOLD_S (default 120, GB_SYNAPSE_STALL_THRESHOLD_S).
    Unlike a single-shot sleep, this re-arms once output resumes — a stream
    that outputs, stalls, and never resumes is caught every time it happens,
    not just the first time in the request's lifetime. The watchdog only gets
    to run because asyncio yields control to it whenever the streaming loop
    blocks on upstream I/O — a plain single-threaded event loop, not a
    real thread.

    Usage: `async with _StallWatch(model) as watch:` around the upstream
    request/response loop, calling `watch.mark()` every time real output (a
    token, a non-empty chunk) arrives."""

    def __init__(self, model: str):
        self.model = model
        self._last_output_ts = time.monotonic()
        self._fired = False
        self._task: "asyncio.Task | None" = None

    def mark(self) -> None:
        self._last_output_ts = time.monotonic()
        self._fired = False  # output resumed — allow a fresh stall to fire again

    async def __aenter__(self) -> "_StallWatch":
        self._task = asyncio.create_task(self._watch())
        return self

    async def __aexit__(self, *exc) -> None:
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    async def _watch(self) -> None:
        # Poll finely enough to detect crossing the threshold promptly even
        # when a test shrinks STALL_THRESHOLD_S to sub-second values, without
        # busy-looping at the real 120s-default production threshold.
        poll_s = max(0.01, min(1.0, STALL_THRESHOLD_S / 10))
        while True:
            await asyncio.sleep(poll_s)
            idle_s = time.monotonic() - self._last_output_ts
            if idle_s >= STALL_THRESHOLD_S and not self._fired:
                self._fired = True
                try:
                    import gb_synapse
                    gb_synapse.emit_stall(self.model, ENGINE, idle_s)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Tool calling
#
# --jinja (already passed to llama-server, gb_synapse_backends.py) gives
# grammar-constrained, per-model-template-correct tool calls natively — it
# owns the per-model template quirks, so llama.cpp's /v1/chat/completions
# needs nothing beyond the passthrough fix above. The 3 non-llama.cpp
# backends (torch/transformers/diffusers) have no --jinja equivalent, so
# they get an EMULATED path via gb_synapse_tools.py (513 lines, 19 tests,
# previously consumed only by greenboost-cli's own injection.py — this is
# its second real consumer): inject tool definitions into the system
# message, force non-streaming upstream (a partial `<tool_call>` opener
# can't be reliably detected mid-stream across every model's variant
# syntax — a one-frame response is correct here, not a shortcut), parse
# calls out of the finished text, strip the tool-call markup from the
# visible content.
# ---------------------------------------------------------------------------

def _tools_mode() -> str:
    return "native" if ENGINE == "llama.cpp" else "emulate"


def _openai_tools_to_schemas(tools: "list | None") -> list:
    """OpenAI/Ollama tools ({"type":"function","function":{"name",
    "description","parameters"}}) -> gb_synapse_tools' own schema shape
    ({"name","description","input_schema"}, the same one greenboost-cli's
    injection.py already builds from Anthropic tool definitions)."""
    out = []
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        out.append({"name": fn.get("name", ""), "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {})})
    return out


def _inject_tools_into_messages(messages: list, tool_schemas: list, model: str) -> list:
    import gb_synapse_tools
    msgs = [dict(m) for m in messages]
    sys_idx = next((i for i, m in enumerate(msgs) if m.get("role") == "system"), None)
    if sys_idx is not None:
        msgs[sys_idx]["content"] = gb_synapse_tools.inject_tools_into_system(
            msgs[sys_idx].get("content", "") or "", tool_schemas, model)
    else:
        msgs.insert(0, {"role": "system",
                        "content": gb_synapse_tools.inject_tools_into_system("", tool_schemas, model)})
    return msgs


def _extract_tool_calls(text: str, tool_schemas: list, model: str) -> "tuple[str, list]":
    import gb_synapse_tools
    calls = gb_synapse_tools.parse_tool_calls_from_text(text, tool_schemas, model)
    clean = gb_synapse_tools.clean_text_from_tool_blocks(text) if calls else text
    return clean, calls


def _calls_to_ollama_tool_calls(calls: list) -> list:
    """gb_synapse_tools' parsed shape ({"id","name","input"}) -> Ollama's
    message.tool_calls ({"function": {"name","arguments"}}, arguments as a
    parsed OBJECT)."""
    return [{"function": {"name": c["name"], "arguments": c["input"]}} for c in calls]


def _calls_to_openai_tool_calls(calls: list) -> list:
    """gb_synapse_tools' parsed shape -> OpenAI's message.tool_calls
    (arguments as a JSON STRING, per the OpenAI/v1 spec)."""
    return [{"id": c.get("id") or f"call_{i}", "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["input"])}}
            for i, c in enumerate(calls)]


def _openai_tool_calls_to_ollama(tcs: "list | None") -> "list | None":
    """OpenAI's tool_calls carry `arguments` as a JSON STRING; Ollama's
    carry a parsed OBJECT — translate so /api/chat gets the shape its own
    clients expect regardless of what llama-server's native --jinja path
    produced (native mode only; the emulate path builds Ollama's shape
    directly via _calls_to_ollama_tool_calls)."""
    if not tcs:
        return None
    out = []
    for tc in tcs:
        fn = tc.get("function", {})
        args = fn.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        out.append({"function": {"name": fn.get("name", ""), "arguments": args}})
    return out


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
    if "stop" in options:
        out["stop"] = options["stop"]
    if "seed" in options:
        out["seed"] = options["seed"]
    if "presence_penalty" in options:
        out["presence_penalty"] = options["presence_penalty"]
    if "frequency_penalty" in options:
        out["frequency_penalty"] = options["frequency_penalty"]
    # llama-server's OpenAI-compatible endpoints accept these as extra,
    # non-standard sampling fields alongside the OpenAI-shaped body (it just
    # forwards unrecognized JSON keys into its own sampling params).
    if "top_k" in options:
        out["top_k"] = options["top_k"]
    if "repeat_penalty" in options:
        out["repeat_penalty"] = options["repeat_penalty"]
    if "min_p" in options:
        out["min_p"] = options["min_p"]
    # num_ctx intentionally NOT forwarded: llama-server's context size is
    # fixed at server launch (-c), not settable per-request — forwarding it
    # would silently no-op rather than resize anything.
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


def _ollama_messages_to_openai(messages: list) -> list:
    """Translate Ollama's per-message `images: [b64, ...]` into OpenAI's
    multimodal content parts.

    Ollama carries images beside the text ({"content": "...", "images": [b64]});
    OpenAI (and therefore llama-server's /v1 surface) carries them INSIDE
    content as image_url parts. Relaying the messages untouched dropped every
    image on the floor — and a VLM asked to describe an image it never received
    does not fail, it invents. Silent wrong answers are worse than errors, and
    this is the single translation an Ollama-API vision client (ai-forge's
    critic) needs from us.
    """
    out = []
    for m in messages:
        imgs = m.get("images")
        if not imgs:
            out.append(m)
            continue
        parts = []
        if m.get("content"):
            parts.append({"type": "text", "text": m["content"]})
        for img in imgs:
            # Accept a bare base64 payload (what Ollama clients send) or an
            # already-formed data URL, so both callers work unchanged.
            url = img if str(img).startswith("data:") else f"data:image/jpeg;base64,{img}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        out.append({k: v for k, v in m.items() if k not in ("images", "content")}
                   | {"content": parts})
    return out


def _prompt_to_messages(prompt: str) -> list:
    """/api/generate is prompt-shaped, not messages-shaped, but images still
    need the same content-parts translation _ollama_messages_to_openai does —
    reuse it by wrapping the single prompt as a one-message conversation."""
    return [{"role": "user", "content": prompt}]


async def ollama_generate(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    model = body.get("model") or MODEL_NAME
    stream = body.get("stream", True)
    images = body.get("images")
    if images:
        # /api/generate never had its own image translation — it silently
        # served vision requests text-only. Route through the same
        # messages-based translation /api/chat uses, then read the
        # resulting content back out of the /v1/chat/completions shape.
        upstream_req = {"model": model,
                         "messages": _ollama_messages_to_openai(
                             [{"content": body.get("prompt", ""), "images": images}]),
                         "stream": stream, **_ollama_opts(body.get("options", {}) or {})}
        url = f"{UPSTREAM}/v1/chat/completions"
        text_key = ("choices", 0, "message", "content")
        delta_key = ("choices", 0, "delta", "content")
    else:
        upstream_req = {"model": model, "prompt": body.get("prompt", ""), "stream": stream,
                         **_ollama_opts(body.get("options", {}) or {})}
        url = f"{UPSTREAM}/v1/completions"
        text_key = ("choices", 0, "text")
        delta_key = ("choices", 0, "text")
    if stream:
        upstream_req["stream_options"] = {"include_usage": True}
    _response_format = _ollama_format_to_response_format(body.get("format"))
    if _response_format:
        upstream_req["response_format"] = _response_format

    if not stream:
        try:
            data = await _upstream_json("POST", url, model, json=upstream_req)
        except _UpstreamError as e:
            return web.json_response(_failure_body(model, e.body.decode("utf-8", "ignore")),
                                      status=_upstream_status(e))
        choice = (data.get("choices") or [{}])[0]
        text = choice.get("message", {}).get("content", "") if images else choice.get("text", "")
        usage = data.get("usage", {})
        return web.json_response({
            "model": model, "created_at": _iso_now(), "response": text, "done": True,
            **_ollama_durations(time.monotonic(), None, None,
                                 usage.get("completion_tokens", 0), usage.get("prompt_tokens", 0)),
        })

    t_start = time.monotonic()
    try:
        async with _upstream_stream("POST", url, model, json=upstream_req) as r:
            resp = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
            await resp.prepare(request)
            t_first = t_last = None
            ctok = ptok = 0
            tokens_cached = None
            async with _StallWatch(model) as watch:
                async for chunk in _sse_lines(r):
                    choice = (chunk.get("choices") or [{}])[0]
                    piece = choice.get("delta", {}).get("content", "") if images else choice.get("text", "")
                    if piece:
                        watch.mark()
                        now = time.monotonic()
                        if t_first is None:
                            t_first = now
                        t_last = now
                        await resp.write((json.dumps({"model": model, "created_at": _iso_now(),
                                                       "response": piece, "done": False}) + "\n").encode())
                    ctok, ptok = _usage_counts(chunk, (ctok, ptok))
                    tokens_cached = _cache_info_from_chunk(chunk, tokens_cached)
    except _UpstreamError as e:
        return web.json_response(_failure_body(model, e.body.decode("utf-8", "ignore")),
                                  status=_upstream_status(e))
    await resp.write((json.dumps({"model": model, "created_at": _iso_now(), "response": "",
                                   "done": True,
                                   **_ollama_durations(t_start, t_first, t_last, ctok, ptok)}) + "\n").encode())
    await resp.write_eof()
    _record_tok_s(model, t_first, t_last, ctok)
    _record_prompt_cache(model, t_start, t_first, ptok, tokens_cached)
    return resp


async def ollama_chat(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    model = body.get("model") or MODEL_NAME
    stream = body.get("stream", True)
    tools = body.get("tools")
    mode = _tools_mode() if tools else "none"
    messages = _ollama_messages_to_openai(body.get("messages", []))
    tool_schemas: list = []
    if mode == "emulate":
        tool_schemas = _openai_tools_to_schemas(tools)
        messages = _inject_tools_into_messages(messages, tool_schemas, model)
        # A partial <tool_call>/XML opener can't be reliably detected
        # mid-stream across every model's variant syntax; a single-frame
        # response is correct here, not a shortcut.
        stream = False

    upstream_req = {"model": model, "messages": messages, "stream": stream,
                     **_ollama_opts(body.get("options", {}) or {})}
    if mode == "native":
        upstream_req["tools"] = tools
        if body.get("tool_choice") is not None:
            upstream_req["tool_choice"] = body["tool_choice"]
    if stream:
        upstream_req["stream_options"] = {"include_usage": True}
    _response_format = _ollama_format_to_response_format(body.get("format"))
    if _response_format:
        upstream_req["response_format"] = _response_format
    url = f"{UPSTREAM}/v1/chat/completions"

    if not stream:
        try:
            data = await _upstream_json("POST", url, model, json=upstream_req)
        except _UpstreamError as e:
            return web.json_response(_failure_body(model, e.body.decode("utf-8", "ignore")),
                                      status=_upstream_status(e))
        msg = (data.get("choices") or [{}])[0].get("message", {})
        content = msg.get("content", "") or ""
        usage = data.get("usage", {})
        out_message = {"role": "assistant", "content": content}
        if mode == "emulate":
            clean, calls = _extract_tool_calls(content, tool_schemas, model)
            out_message["content"] = clean
            if calls:
                out_message["tool_calls"] = _calls_to_ollama_tool_calls(calls)
        elif mode == "native":
            oc = _openai_tool_calls_to_ollama(msg.get("tool_calls"))
            if oc:
                out_message["tool_calls"] = oc
        return web.json_response({
            "model": model, "created_at": _iso_now(),
            "message": out_message,
            "done": True, "done_reason": "stop",
            **_ollama_durations(time.monotonic(), None, None,
                                 usage.get("completion_tokens", 0), usage.get("prompt_tokens", 0)),
        })

    # Streaming path — mode is "native" or "none" here ("emulate" forced
    # stream=False above).
    t_start = time.monotonic()
    accumulated_tool_calls: dict[int, dict] = {}
    try:
        async with _upstream_stream("POST", url, model, json=upstream_req) as r:
            resp = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
            await resp.prepare(request)
            t_first = t_last = None
            ctok = ptok = 0
            tokens_cached = None
            async with _StallWatch(model) as watch:
                async for chunk in _sse_lines(r):
                    delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                    piece = delta.get("content", "")
                    if piece:
                        watch.mark()
                        now = time.monotonic()
                        if t_first is None:
                            t_first = now
                        t_last = now
                        frame = {"model": model, "created_at": _iso_now(),
                                 "message": {"role": "assistant", "content": piece}, "done": False}
                        await resp.write((json.dumps(frame) + "\n").encode())
                    # Ollama does not stream partial tool calls — buffer the
                    # OpenAI-style per-index deltas and emit ONE assembled
                    # tool_calls list on the final frame below.
                    for tc_delta in delta.get("tool_calls") or []:
                        idx = tc_delta.get("index", 0)
                        entry = accumulated_tool_calls.setdefault(
                            idx, {"function": {"name": "", "arguments": ""}})
                        fn = tc_delta.get("function", {})
                        if fn.get("name"):
                            entry["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            entry["function"]["arguments"] += fn["arguments"]
                    ctok, ptok = _usage_counts(chunk, (ctok, ptok))
                    tokens_cached = _cache_info_from_chunk(chunk, tokens_cached)
    except _UpstreamError as e:
        return web.json_response(_failure_body(model, e.body.decode("utf-8", "ignore")),
                                  status=_upstream_status(e))
    final = {"model": model, "created_at": _iso_now(),
             "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
             **_ollama_durations(t_start, t_first, t_last, ctok, ptok)}
    if accumulated_tool_calls:
        merged = [accumulated_tool_calls[i] for i in sorted(accumulated_tool_calls)]
        oc = _openai_tool_calls_to_ollama(merged)
        if oc:
            final["message"]["tool_calls"] = oc
    await resp.write((json.dumps(final) + "\n").encode())
    await resp.write_eof()
    _record_tok_s(model, t_first, t_last, ctok)
    _record_prompt_cache(model, t_start, t_first, ptok, tokens_cached)
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

    param_b = gb_synapse._approx_param_count_b(m.n_bytes, m.quant)
    summary = {}
    try:
        summary = gb_synapse.gguf_summary(m.path)
    except Exception:
        pass
    arch = summary.get("arch") or m.arch or "unknown"
    mmproj = None
    try:
        mmproj = gb_synapse._find_mmproj(m)
    except Exception:
        pass
    capabilities = ["completion"]
    if mmproj:
        capabilities.append("vision")

    return web.json_response({
        "modelfile": "", "parameters": "", "template": "",
        "details": {"family": "gguf", "quantization_level": m.quant,
                    "parameter_size": f"{param_b:.1f}B"},
        "model_info": {
            "general.architecture": arch,
            "general.parameter_count": int(param_b * 1e9),
            f"{arch}.context_length": summary.get("ctx_length", m.ctx_length),
            f"{arch}.block_count": summary.get("n_layers", m.n_layers),
        },
        "capabilities": capabilities,
    })


async def ollama_ps(request: web.Request) -> web.Response:
    """Backed by gb_synapse.ps() (which prunes entries whose engine pid has
    died) instead of a hardcoded stub — previously this always reported one
    fake entry regardless of whether the engine behind it was alive, and
    gb_rotator uses this exact endpoint as its liveness probe."""
    import gb_synapse
    st = next((s for s in gb_synapse.ps() if s.get("model") == MODEL_NAME), None)
    if st is None:
        return web.json_response({"models": []})
    size = 0
    try:
        m = next((e for e in gb_synapse.list_models() if e.name == MODEL_NAME), None)
        size = m.n_bytes if m else 0
    except Exception:
        pass
    return web.json_response({"models": [
        {"name": MODEL_NAME, "model": MODEL_NAME, "size": size, "expires_at": ""}
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
    if "stop" in params:
        out["stop"] = params["stop"]
    if "seed" in params:
        out["seed"] = params["seed"]
    if "repetition_penalty" in params:
        out["repeat_penalty"] = params["repetition_penalty"]
    # `details` (per-token detail in the response) is a response-shape toggle,
    # not a sampling param — implementing it means restructuring
    # tgi_generate's return shape, left for when a TGI details consumer
    # actually needs it rather than speculatively now.
    return out


async def tgi_generate(request: web.Request) -> web.Response:
    body = await request.json()
    upstream_req = {"model": MODEL_NAME, "prompt": body.get("inputs", ""), "stream": False,
                     **_tgi_params(body.get("parameters", {}) or {})}
    try:
        data = await _upstream_json("POST", f"{UPSTREAM}/v1/completions", MODEL_NAME, json=upstream_req)
    except _UpstreamError as e:
        return web.json_response({"error": _failure_body(MODEL_NAME, e.body.decode("utf-8", "ignore"))["error"]},
                                  status=_upstream_status(e))
    text = (data.get("choices") or [{}])[0].get("text", "")
    return web.json_response({"generated_text": text})


async def tgi_generate_stream(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    upstream_req = {"model": MODEL_NAME, "prompt": body.get("inputs", ""), "stream": True,
                     **_tgi_params(body.get("parameters", {}) or {})}

    try:
        async with _upstream_stream("POST", f"{UPSTREAM}/v1/completions", MODEL_NAME,
                                     json=upstream_req) as r:
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            acc = ""
            async for chunk in _sse_lines(r):
                piece = (chunk.get("choices") or [{}])[0].get("text", "")
                if not piece:
                    continue
                acc += piece
                frame = {"token": {"text": piece, "special": False}, "generated_text": None}
                await resp.write(f"data:{json.dumps(frame)}\n\n".encode())
    except _UpstreamError as e:
        return web.json_response({"error": _failure_body(MODEL_NAME, e.body.decode("utf-8", "ignore"))["error"]},
                                  status=_upstream_status(e))
    await resp.write(f"data:{json.dumps({'token': None, 'generated_text': acc})}\n\n".encode())
    await resp.write_eof()
    return resp


# ---------------------------------------------------------------------------
# OpenAI /v1/* — raw passthrough (llama-server already speaks this natively)
# ---------------------------------------------------------------------------

def _forward_headers(request: web.Request) -> dict:
    headers = {}
    auth = request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth
    return headers


async def _emulate_openai_tool_call_completion(parsed: dict, model: str) -> web.Response:
    """/v1/chat/completions with `tools` present, on a non-llama.cpp backend
    with no native --jinja tool support. Mirrors ollama_chat's emulate
    branch but returns the OpenAI envelope (string-JSON `arguments`, `id`,
    `type: "function"`) instead of Ollama's. Forces non-streaming upstream
    for the same reason ollama_chat does: a partial tool-call opener can't
    be reliably detected mid-stream across every model's variant syntax."""
    tools = parsed.get("tools")
    tool_schemas = _openai_tools_to_schemas(tools)
    upstream_req = dict(parsed)
    upstream_req["messages"] = _inject_tools_into_messages(
        parsed.get("messages", []), tool_schemas, model)
    upstream_req["stream"] = False
    upstream_req.pop("tools", None)
    upstream_req.pop("tool_choice", None)
    try:
        data = await _upstream_json("POST", f"{UPSTREAM}/v1/chat/completions",
                                    model, json=upstream_req)
    except _UpstreamError as e:
        return web.json_response(_failure_body(model, e.body.decode("utf-8", "ignore")),
                                  status=_upstream_status(e))
    choice = (data.get("choices") or [{}])[0]
    msg = dict(choice.get("message", {}))
    content = msg.get("content", "") or ""
    clean, calls = _extract_tool_calls(content, tool_schemas, model)
    finish_reason = choice.get("finish_reason", "stop")
    msg["content"] = clean or None
    if calls:
        msg["tool_calls"] = _calls_to_openai_tool_calls(calls)
        finish_reason = "tool_calls"
    data["choices"] = [{**choice, "message": msg, "finish_reason": finish_reason}]
    return web.json_response(data)


async def openai_passthrough(request: web.Request) -> web.StreamResponse:
    path = request.match_info.get("path", "")
    query = f"?{request.query_string}" if request.query_string else ""
    url = f"{UPSTREAM}/v1/{path}{query}"
    headers = _forward_headers(request)

    if request.method == "GET":
        try:
            async with SESSION.get(url, headers=headers) as r:
                data = await r.read()
                # The embedding model must appear here — LangChain/llama-
                # index validate an embeddings model against /v1/models and
                # hard-fail otherwise, and it's served by a completely
                # separate process the primary /v1/models upstream knows
                # nothing about.
                embed_id = _embed_model_id() if path == "models" and _embed_upstream() else None
                if embed_id and r.status == 200:
                    try:
                        parsed_models = json.loads(data)
                        ids = {m.get("id") for m in parsed_models.get("data", [])}
                        if embed_id not in ids:
                            parsed_models.setdefault("data", []).append(
                                {"id": embed_id, "object": "model"})
                        return web.json_response(parsed_models)
                    except (json.JSONDecodeError, AttributeError):
                        pass
                return web.Response(body=data, status=r.status, content_type=r.content_type)
        except (aiohttp.ClientError, TimeoutError) as e:
            return web.json_response(_failure_body(MODEL_NAME, str(e)), status=502)

    body = await request.read()
    headers["Content-Type"] = request.headers.get("Content-Type", "application/json")
    req_model = MODEL_NAME
    try:
        parsed = json.loads(body) if body else {}
        stream = bool(parsed.get("stream"))
        req_model = parsed.get("model") or MODEL_NAME
    except json.JSONDecodeError:
        parsed = {}
        stream = False

    # This is the one narrow, mode-gated exception to "pure passthrough":
    # llama.cpp handles tools natively via --jinja (nothing to do above),
    # but the 3 non-llama.cpp backends have no equivalent, so a request
    # carrying `tools` on this exact path needs the emulation layer instead
    # of a byte-for-byte relay.
    if path == "chat/completions" and parsed.get("tools") and _tools_mode() == "emulate":
        return await _emulate_openai_tool_call_completion(parsed, req_model)

    if not stream:
        try:
            async with SESSION.post(url, data=body, headers=headers) as r:
                data = await r.read()
                return web.Response(body=data, status=r.status, content_type=r.content_type)
        except (aiohttp.ClientError, TimeoutError) as e:
            return web.json_response(_failure_body(req_model, str(e)), status=502)

    # This route is a genuine passthrough — llama-server's own error/JSON
    # shape is what an OpenAI-SDK client already expects on /v1/*, so on a
    # bad status we relay its real body/status rather than synthesizing a
    # failure_report(). The fix is ordering: check status BEFORE prepare()ing
    # the client-facing stream, not after — previously this route prepared a
    # 200 SSE response before even opening the upstream connection, so any
    # upstream 4xx/5xx was relayed as a 200 with an empty body.
    # Telemetry (tok_s_measured/prompt_cache) for THIS route: GB-CLI talks
    # /v1/* directly (BACKEND_REGISTRY's base_url), so this passthrough — not
    # ollama_generate/ollama_chat — carries its real traffic. raw_chunk is
    # still written to the client immediately and unmodified on every
    # iteration (true byte-for-byte passthrough, untouched); _telemetry_buf
    # is a SEPARATE accumulation of the same bytes, parsed only after the
    # response completes, so a parse failure can never affect what was
    # already forwarded.
    is_completions_path = path in ("chat/completions", "completions")
    t_start = time.monotonic()
    # (timestamp, raw_chunk) pairs, not a flat byte buffer — t_first/t_last
    # must be latched on the SSE event that actually carries generated
    # content, not on whichever raw TCP chunk arrives first (see
    # _parse_sse_telemetry_timed's docstring for the incident this fixes).
    # Still a pure observe-after-write: raw_chunk is forwarded unmodified
    # below regardless of what this list holds.
    _telemetry_chunks = [] if is_completions_path else None
    try:
        async with SESSION.post(url, data=body, headers=headers) as r:
            if r.status >= 400:
                data = await r.read()
                return web.Response(body=data, status=r.status, content_type=r.content_type)
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            async with _StallWatch(req_model) as watch:
                async for raw_chunk in r.content.iter_any():
                    if raw_chunk:
                        watch.mark()
                        if _telemetry_chunks is not None:
                            _telemetry_chunks.append((time.monotonic(), raw_chunk))
                    await resp.write(raw_chunk)
    except (aiohttp.ClientError, TimeoutError) as e:
        return web.json_response(_failure_body(req_model, str(e)), status=502)
    await resp.write_eof()
    if _telemetry_chunks is not None:
        ctok, ptok, tokens_cached, t_first, t_last = _parse_sse_telemetry_timed(_telemetry_chunks)
        _record_tok_s(req_model, t_first, t_last, ctok)
        _record_prompt_cache(req_model, t_start, t_first, ptok, tokens_cached)
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


async def ollama_root(request: web.Request) -> web.Response:
    """The `GET /` "Ollama is running" probe several clients use to decide
    whether an Ollama-compatible server is present at all before trying any
    real endpoint."""
    return web.Response(text="Ollama is running")


async def ollama_version(request: web.Request) -> web.Response:
    return web.json_response({"version": "gb-synapse"})


async def ollama_pull(request: web.Request) -> web.StreamResponse:
    """Exposes gb_synapse.pull() over the wire — the function already
    existed but nothing served it, so a client relying on Ollama's
    /api/pull to fetch a model had no equivalent here."""
    body = await request.json()
    name = body.get("model") or body.get("name")
    if not name:
        return web.json_response({"error": "model name required"}, status=400)
    resp = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
    await resp.prepare(request)
    try:
        import gb_synapse
        await asyncio.to_thread(gb_synapse.pull, name)
        await resp.write((json.dumps({"status": "success"}) + "\n").encode())
    except Exception as e:
        await resp.write((json.dumps({"status": "error", "error": str(e)}) + "\n").encode())
    await resp.write_eof()
    return resp


# ---------------------------------------------------------------------------
# Embeddings — the largest single API hole for :11435 before this: no
# /v1/embeddings, /api/embed, or /api/embeddings meant no RAG/reranker
# client could use gb-synapse at all. Backed by a SECOND, lazily-configured
# llama-server (GB_SYNAPSE_EMBED_MODEL, gb_synapse._maybe_serve_embedding)
# — a generation model and an embedding model are different processes,
# never the same one (llama-server's --embeddings flips it to pooled/
# non-causal attention, incompatible with the completion endpoints).
# ---------------------------------------------------------------------------

_NO_EMBED_ERROR = {
    "error": "no embeddings engine configured for this gb-synapse instance — "
            "set GB_SYNAPSE_EMBED_MODEL and (re)run `greenboost synapse serve`",
}


def _embed_upstream() -> "str | None":
    """Resolve the embeddings engine's upstream URL fresh on every call by
    reading gb_synapse's persisted run-state for THIS model — not a
    module-level constant — so a later serve() call that brings the
    embeddings engine up (or restarts it on a new port) is picked up on the
    next embeddings request without needing to restart this proxy."""
    try:
        import gb_synapse as gs
        path = gs._run_state_path(MODEL_NAME)
        if not path.is_file():
            return None
        st = gs.ServerState(**json.loads(path.read_text()))
        if st.embed_internal_port and gs._pid_alive(st.embed_pid):
            return f"http://127.0.0.1:{st.embed_internal_port}"
    except Exception:
        pass
    return None


def _embed_model_id() -> "str | None":
    return os.environ.get("GB_SYNAPSE_EMBED_MODEL", "").strip() or None


async def openai_embeddings(request: web.Request) -> web.Response:
    """POST /v1/embeddings — the exact OpenAI envelope, relayed almost
    byte-for-byte (llama-server's own /v1/embeddings already speaks it);
    the only translation is which upstream process answers."""
    upstream = _embed_upstream()
    if upstream is None:
        return web.json_response(_NO_EMBED_ERROR, status=400)
    body = await request.read()
    try:
        async with SESSION.post(f"{upstream}/v1/embeddings", data=body,
                                headers={"Content-Type": "application/json"}) as r:
            data = await r.read()
            return web.Response(body=data, status=r.status, content_type=r.content_type)
    except (aiohttp.ClientError, TimeoutError) as e:
        return web.json_response(_failure_body(_embed_model_id() or MODEL_NAME, str(e)), status=502)


async def ollama_embed(request: web.Request) -> web.Response:
    """POST /api/embed — Ollama's current (plural) embeddings endpoint:
    {"model":..., "input": str|list[str]} -> {"model":..., "embeddings":
    [[...], ...]}. llama-server handles a multi-prompt `input` list
    natively, so this passes it straight through rather than looping."""
    upstream = _embed_upstream()
    if upstream is None:
        return web.json_response(_NO_EMBED_ERROR, status=400)
    body = await request.json()
    inputs = body.get("input")
    if inputs is None:
        return web.json_response({"error": "input required"}, status=400)
    model = body.get("model") or _embed_model_id() or MODEL_NAME
    try:
        data = await _upstream_json("POST", f"{upstream}/v1/embeddings", model,
                                    json={"input": inputs, "model": model})
    except _UpstreamError as e:
        return web.json_response(_failure_body(model, e.body.decode("utf-8", "ignore")),
                                  status=_upstream_status(e))
    ordered = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
    return web.json_response({"model": model, "embeddings": [d.get("embedding", []) for d in ordered]})


async def ollama_embeddings_legacy(request: web.Request) -> web.Response:
    """POST /api/embeddings — Ollama's legacy, SINGULAR endpoint:
    {"model":..., "prompt": str} -> {"embedding": [...]}."""
    upstream = _embed_upstream()
    if upstream is None:
        return web.json_response(_NO_EMBED_ERROR, status=400)
    body = await request.json()
    prompt = body.get("prompt")
    if prompt is None:
        return web.json_response({"error": "prompt required"}, status=400)
    model = body.get("model") or _embed_model_id() or MODEL_NAME
    try:
        data = await _upstream_json("POST", f"{upstream}/v1/embeddings", model,
                                    json={"input": prompt, "model": model})
    except _UpstreamError as e:
        return web.json_response(_failure_body(model, e.body.decode("utf-8", "ignore")),
                                  status=_upstream_status(e))
    items = data.get("data") or [{}]
    return web.json_response({"embedding": items[0].get("embedding", [])})


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------

async def _on_startup(app: web.Application) -> None:
    global SESSION
    SESSION = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=SOCK_READ_S))


async def _on_cleanup(app: web.Application) -> None:
    if SESSION is not None:
        await SESSION.close()


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", ollama_root)
    app.router.add_get("/health", health)
    app.router.add_get("/api/version", ollama_version)
    app.router.add_post("/api/pull", ollama_pull)
    app.router.add_post("/api/generate", ollama_generate)
    app.router.add_post("/api/chat", ollama_chat)
    app.router.add_get("/api/tags", ollama_tags)
    app.router.add_post("/api/show", ollama_show)
    app.router.add_get("/api/ps", ollama_ps)
    app.router.add_post("/api/embed", ollama_embed)
    app.router.add_post("/api/embeddings", ollama_embeddings_legacy)
    app.router.add_post("/generate", tgi_generate)
    app.router.add_post("/generate_stream", tgi_generate_stream)
    # Registered before the /v1/{path:.*} wildcard so it wins the match.
    app.router.add_post("/v1/embeddings", openai_embeddings)
    app.router.add_route("*", "/v1/{path:.*}", openai_passthrough)
    app.router.add_route("*", "/slots", slots_passthrough)
    app.router.add_route("*", "/slots/{path:.*}", slots_passthrough)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


_TELEMETRY = None  # module-level ref, kept alive for the process lifetime


def _start_flight_recorder() -> None:
    """Best-effort: start the dataflux SnapshotRecorder (continuous VRAM/
    GPU-util/KV-pressure history) for the lifetime of this proxy process.

    gb_init.py normally owns this bootstrap, but gb_init also enforces
    PYTORCH_CUDA_ALLOC_CONF and monkeypatches torch.cuda.empty_cache — side
    effects that make no sense for this lightweight aiohttp proxy, which
    never touches torch. So this replicates only gb_init's telemetry-start
    step (gb_telemetry has no torch import at module scope), not the whole
    module. Real gap this closes (2026-08-01): a gb-synapse serve running
    llama-server (no Python process ever imports gb_init) produced ZERO
    dataflux `snapshot` events for the entire 12+ hour serve — the only
    process alive that outlives every request IS this proxy, so it's the
    natural owner. Opt-out: GREENBOOST_DATAFLUX=0, same convention as
    gb_init.py. Never blocks serving on failure."""
    global _TELEMETRY
    if os.environ.get("GREENBOOST_DATAFLUX", "1") == "0":
        return
    try:
        from gb_telemetry import TelemetryManager
        import gb_dataflux
        _TELEMETRY = TelemetryManager(device=0, poll_ms=500, enable_dcgm=True)
        _TELEMETRY.start()
        gb_dataflux.start_snapshot_recorder(_TELEMETRY, interval_s=5.0)
    except Exception as exc:
        print(f"[gb-synapse-api] flight recorder unavailable: {exc}", flush=True)


def main() -> None:
    global UPSTREAM, MODEL_NAME, ENGINE
    ap = argparse.ArgumentParser(description="gb-synapse API proxy")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--upstream-port", type=int, required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--engine", default="", help="backend name (llama.cpp/torch/"
                     "transformers/diffusers) — carried into synapse_stall events")
    args = ap.parse_args()

    UPSTREAM = f"http://127.0.0.1:{args.upstream_port}"
    MODEL_NAME = args.model_name
    ENGINE = args.engine
    _start_flight_recorder()
    web.run_app(build_app(), host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
    main()

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
    python3 gb_synapse_api.py --port 11369 --upstream-port 12435 --model-name qwen3-coder
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import json
import os
import stat
import time
from pathlib import Path

import aiohttp
from aiohttp import web

UPSTREAM = ""
MODEL_NAME = ""
ENGINE = ""
BIND = "127.0.0.1"
SESSION: aiohttp.ClientSession | None = None

# Same rule as gb_a2a.py (the A2A gateway's own auth): a non-loopback bind
# REQUIRES a token or the process refuses to start. Before this fix
# gb_synapse.py always launched this proxy on 0.0.0.0 with no auth at all —
# reachable from anything with network access to the host, not just the
# other local processes (greenboost-cli, ai-forge) that are its actual
# consumers. Every local consumer talks to this proxy over 127.0.0.1 (a
# feeder's own Ollama is reached via an SSH tunnel, not directly — see
# ai-forge's studio/server/gbcluster.py), so a loopback default with no
# token keeps every existing caller working unchanged; only a caller that
# needs LAN/container reach (e.g. exposing this to a sandboxed agent
# runtime) has to opt in with a token.
_TOKEN_ENV = "GB_SYNAPSE_TOKEN"
_TOKEN_FILE = Path("/etc/greenboost/synapse_token")


def _trust_validate_root_file(path: Path, max_size: int = 64 * 1024) -> str:
    """Audit TCB-U1 (2026-08-06): Validate that a root-owned config file is
    trustworthy before reading its contents. Returns one of "trusted",
    "absent", "rejected_symlink", "rejected_owner", "rejected_mode",
    "rejected_size" — matches gb_dataflux_kinds.KINDS["allowlist_trust"]'s
    documented decision enum exactly, so a caller can emit it verbatim.
    Logs nothing and never emits dataflux itself; "absent" is the routine
    "no file configured" case and callers should not treat it as a security
    event the way the other rejections are.
    """
    try:
        st = path.lstat()  # lstat to detect symlinks
    except (OSError, FileNotFoundError):
        return "absent"

    # Must be a regular file, not a symlink or other special file (a
    # symlink swap is the attack this whole check exists to close; any
    # other non-regular file at this path is equally "not what we expect
    # here" and gets the same bucket).
    if not stat.S_ISREG(st.st_mode):
        return "rejected_symlink"

    # Must be owned by root
    if st.st_uid != 0:
        return "rejected_owner"

    # Must not be writable by group or others
    if (st.st_mode & 0o022) != 0:
        return "rejected_mode"

    # Must be non-empty and within the size cap
    if st.st_size == 0 or st.st_size > max_size:
        return "rejected_size"

    return "trusted"


def _resolve_token() -> str:
    """GB_SYNAPSE_TOKEN env first (never on argv — /proc/<pid>/cmdline is
    world-readable), else the installer-managed token file (TCB-U1 validated),
    else no auth (loopback-only mode, same convention as GB_A2A_TOKEN)."""
    env = os.environ.get(_TOKEN_ENV, "").strip()
    if env:
        return env
    try:
        # TCB-U1: validate token file trust before reading
        decision = _trust_validate_root_file(_TOKEN_FILE, max_size=8192)
        if decision != "trusted":
            if decision != "absent":
                try:
                    import gb_dataflux
                    gb_dataflux.emit({"kind": "allowlist_trust",
                                       "path": str(_TOKEN_FILE), "decision": decision})
                except Exception:
                    pass
            # Not a regular root-owned file, insecure mode, or wrong size
            # — treat as if no file exists (loopback-only mode)
            return ""
        return _TOKEN_FILE.read_text().strip()
    except OSError:
        return ""


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")


@web.middleware
async def _auth_middleware(request: web.Request, handler):
    """Uniform Bearer-token check, no bypass for /health — an unauthenticated
    health probe still confirms the proxy is alive and serving, information
    a LAN-reachable deployment shouldn't hand out for free either."""
    token = _resolve_token()
    if not token:
        return await handler(request)  # loopback-only mode, enforced at startup
    auth = request.headers.get("Authorization", "")
    presented = auth[7:] if auth.startswith("Bearer ") else ""
    if not hmac.compare_digest(presented.encode(), token.encode()):
        try:
            import gb_dataflux
            gb_dataflux.emit({
                "kind": "synapse_auth", "status": "rejected",
                "path": request.path, "peer": request.remote,
            })
        except Exception:
            pass
        return web.json_response(
            {"error": {"message": "missing/invalid Bearer token"}}, status=401)
    return await handler(request)

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


# Minimum completion length before a turn counts as a decode-rate measurement.
#
# tok_s here is (completion_tokens - 1) / inter-token-span, so a 2-token reply is
# a single interval and a 5-token reply is four — dominated by fixed per-request
# overhead, not by decode throughput. The old guard only rejected 1 token, so
# ordinary short answers ("DONE", a tool call) were recorded as decode samples.
# Live effect measured 2026-08-17 on a model whose real rate is ~5 tok/s: the
# same 34-sample series held 137.6, 148.1 and 283.8 tok/s, pulling the mean to
# 30.6 and tripping gb_pilot's "decode degraded" advisory (which carries a
# set_kv_size_threshold_mb retune lever) purely from short replies.
#
# 24 tokens gives ~23 intervals, and with MTP speculative decoding emitting up
# to `mtp_draft_n` tokens per forward pass that is still several independent
# bursts rather than one. Env-overridable for deliberately short workloads.
_MIN_TOK_S_SAMPLE_TOKENS = max(2, int(os.environ.get("GB_SYNAPSE_MIN_TOK_S_TOKENS", "24")))


def _record_tok_s(model: str, t_first: "float | None", t_last: "float | None",
                  completion_tokens: int, prompt_tokens: int = 0) -> None:
    """Compute client-observed decode tok/s from first→last token timing and
    the upstream's usage count, then hand it to gb_synapse.record_measured_tok_s
    (dataflux emit + rolling store). Proxy-side so ANY client on the gb-synapse
    port (curl, Zed, ai-forge's Ollama-compat steps) feeds recommend()'s fit
    estimator, not only greenboost-cli. Best-effort: never raises, silently
    skips incomplete samples (no usage, single token, zero interval)."""
    try:
        if completion_tokens < _MIN_TOK_S_SAMPLE_TOKENS or t_first is None or t_last is None:
            # Countable, not averaged. See gb_synapse.record_tok_s_skipped for
            # why silence here was itself the bug.
            import gb_synapse
            gb_synapse.record_tok_s_skipped(
                model,
                "below_sample_floor" if completion_tokens < _MIN_TOK_S_SAMPLE_TOKENS
                else "no_token_timestamps",
                completion_tokens=completion_tokens, source="proxy")
            return
        dt = t_last - t_first
        if dt <= 0:
            return
        # decode rate: tokens after the first, over the inter-token span
        tok_s = (completion_tokens - 1) / dt
        import gb_synapse
        gb_synapse.record_measured_tok_s(model, tok_s, source="proxy",
                                         completion_tokens=completion_tokens,
                                         prompt_tokens=prompt_tokens)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# GB-1 — keep a conversation on the slot that still holds its KV
# ---------------------------------------------------------------------------
# Measured on this box (gb_bench_turn.py --edit-at, 3 runs, 2026-08-19): a warm
# turn prefills in ~1.77 s, the turn after an early edit costs ~8.8 s. Losing a
# slot to another conversation is the same cliff with nothing edited at all —
# 2026-08-05 established that --slot-prompt-similarity cannot prevent it,
# because any nonzero overlap with a recently-touched slot beats an idle slot's
# zero score. That needs identity, which gb_prompt_index provides.
#
# Lossless: identical tokens, identical order, on a slot that still has them.

_SLOT_INDEX = None            # gb_prompt_index.ConversationIndex, built lazily
_SLOT_COUNT_AT = 0.0
_SLOT_COUNT_TTL_S = 60.0      # slots only change on a re-serve, so this is cheap


async def _engine_slot_count() -> int:
    """How many slots the engine really has (--parallel), asked, not assumed.

    Returns 0 when the answer is unknown, and the caller must then send NO
    id_slot — guessing 4 and pinning to a slot that does not exist would be
    worse than letting the engine choose, since llama-server wraps an
    out-of-range id modulo the slot count and would silently collide two
    conversations.
    """
    global _SLOT_INDEX, _SLOT_COUNT_AT
    now = time.monotonic()
    if _SLOT_INDEX is not None and _SLOT_INDEX.n_slots and (now - _SLOT_COUNT_AT) < _SLOT_COUNT_TTL_S:
        return _SLOT_INDEX.n_slots
    n = 0
    try:
        if SESSION is not None:
            async with SESSION.get(f"{UPSTREAM}/slots",
                                   timeout=aiohttp.ClientTimeout(total=2)) as r:
                if r.status == 200:
                    slots = await r.json()
                    if isinstance(slots, list):
                        n = len(slots)
    except (aiohttp.ClientError, TimeoutError, ValueError, json.JSONDecodeError):
        n = 0
    import gb_prompt_index
    if _SLOT_INDEX is None:
        _SLOT_INDEX = gb_prompt_index.ConversationIndex(n_slots=n)
    elif n:
        _SLOT_INDEX.n_slots = n
    _SLOT_COUNT_AT = now
    return _SLOT_INDEX.n_slots


async def _pin_conversation_slot(parsed: dict, model: str,
                                 request: "web.Request | None" = None) -> "int | None":
    """Choose the slot for this chat request and record why.

    Best-effort in the strict sense: any failure returns None, the request goes
    upstream exactly as it arrived, and the engine picks a slot the way it
    always did. A cache optimisation must never be able to fail a request.
    """
    try:
        import gb_prompt_index
        messages = parsed.get("messages")
        if not isinstance(messages, list) or not messages:
            return None
        n = await _engine_slot_count()
        if n <= 1 or _SLOT_INDEX is None:
            return None
        # An explicit conversation id makes identity exact instead of inferred,
        # which is the difference between an edited first message keeping its
        # slot and paying to re-prefill the system prompt once. Accepted in the
        # body or as a header so a client that cannot touch the JSON (a proxy
        # chain, curl) can still supply it.
        import gb_prompt_index as _pi
        conv_id = str(parsed.pop(_pi.CONVERSATION_ID_FIELD, "") or "")
        if not conv_id and request is not None:
            conv_id = request.headers.get(_pi.CONVERSATION_ID_HEADER, "")
        a = _SLOT_INDEX.assign(messages, parsed.get("tools"),
                               conversation_id=conv_id[:128])
        pin = gb_prompt_index.pinning_enabled()
        try:
            import gb_dataflux
            gb_dataflux.emit({
                "node": "host", "label": "synapse", "kind": "cache_index",
                "model": model, "conv": a.key, "slot": a.slot,
                "decision": a.reason, "chunks": a.chunks,
                "identity": "explicit" if conv_id else "inferred",
                "chunks_before": a.chunks_before,
                "changed_chunk": a.changed_chunk,
                "n_slots": n, "tracked": len(_SLOT_INDEX),
                "applied": pin,
            })
        except Exception:
            pass
        # The index runs even when pinning is off: content-addressing the
        # conversation is what made the 2026-08-19 regression diagnosable, and
        # `changed_chunk` explains a prefill cost whether or not this proxy is
        # choosing slots. Only the id_slot itself is gated.
        return a.slot if pin else None
    except Exception:
        return None


def _record_prompt_cache(model: str, t_start: float, t_first: "float | None",
                         prompt_tokens: int, tokens_cached: "int | None",
                         engine_prompt_ms: "float | None" = None) -> None:
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
        gb_synapse.record_prompt_cache_sample(model, ttft_ms, hit_pct, tokens_cached or 0,
                                              engine_prompt_ms=engine_prompt_ms,
                                              prompt_tokens=prompt_tokens)
    except Exception:
        pass


def _record_non_stream_telemetry(model: str, raw: bytes) -> None:
    """Non-streaming counterpart to _record_tok_s/_record_prompt_cache.

    A `stream:false` request never enters openai_passthrough()'s SSE-parsing
    branch those two rely on, so it recorded NOTHING before this fix —
    confirmed live 2026-08-05: a plain curl with stream:false through
    openai_passthrough (the exact route GB-CLI's BACKEND_REGISTRY talks per
    _parse_sse_telemetry_timed's own docstring, not ollama_chat) left zero
    trace in dataflux_tok_s despite completing normally. No reconstruction
    needed to fix it: llama-server's non-streaming /v1/chat/completions
    response already carries its own `timings` block (predicted_per_second,
    prompt_ms, ...) same as its native /completion endpoint — just read it.
    Best-effort: never raises, silently skips a response with no
    timings/usage (other backends reachable via this same passthrough may
    not emit either)."""
    try:
        data = json.loads(raw)
        usage = data.get("usage") or {}
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        if completion_tokens <= 1:
            return
        timings = data.get("timings") or {}
        tok_s = timings.get("predicted_per_second")
        predicted_ms = timings.get("predicted_ms")
        if tok_s is None and predicted_ms:
            tok_s = timings.get("predicted_n", completion_tokens) / (predicted_ms / 1000.0)
        import gb_synapse
        # Same sample-quality floor the streaming path has always applied
        # (_record_tok_s → _MIN_TOK_S_SAMPLE_TOKENS). It was never carried over
        # when this non-streaming branch learned to record telemetry
        # (2026-08-05), so this path accepted 2-token replies as full-weight
        # samples: a handful of tokens timed over a few ms scores hundreds of
        # tok/s. That is the measured source of the 254.9 and 283.8 tok/s
        # samples that dragged the reference workload's mean to 33.2 against a
        # median of 13.9, and in turn produced a false "46% regression" alert.
        # Prompt-cache telemetry below deliberately stays on the looser >1
        # guard — TTFT/hit-rate are meaningful for a short reply; decode rate
        # is not.
        if tok_s and completion_tokens >= _MIN_TOK_S_SAMPLE_TOKENS:
            gb_synapse.record_measured_tok_s(model, tok_s, source="proxy",
                                             completion_tokens=completion_tokens,
                                             prompt_tokens=prompt_tokens)
        else:
            gb_synapse.record_tok_s_skipped(
                model,
                "below_sample_floor" if completion_tokens < _MIN_TOK_S_SAMPLE_TOKENS
                else "no_engine_rate",
                completion_tokens=completion_tokens, source="proxy")
        prompt_ms = timings.get("prompt_ms")
        tokens_cached = _cache_info_from_chunk(data, None)
        if prompt_ms is not None or tokens_cached is not None:
            hit_pct = (100.0 * tokens_cached / prompt_tokens
                       if tokens_cached is not None and prompt_tokens else None)
            # Non-streaming has no first-token timestamp, so there is no true
            # TTFT here — prompt_ms IS the engine's prefill time and is passed
            # in both slots deliberately, with the engine field making it
            # explicit which one is authoritative.
            gb_synapse.record_prompt_cache_sample(model, prompt_ms, hit_pct, tokens_cached or 0,
                                                  engine_prompt_ms=prompt_ms,
                                                  prompt_tokens=prompt_tokens)
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


class _SSETelemetry:
    """Incremental SSE telemetry , the same six scalars, without hoarding bytes.

    The streaming path used to append EVERY raw chunk of a response to a list
    and parse it after the response completed. That list is the only unbounded
    per-request buffer in the hot path, and the proxy is long-lived: on
    2026-08-18 `gb_synapse_api.py` was measured holding **37.5 GB** of
    anonymous memory (7.6 GB of it swapped) after twenty minutes of ordinary
    agentic traffic, on a 61 GB box, and the OOM killer took the user's whole
    terminal twice that day. RSS plateaued rather than climbing, which is the
    signature of a large transient peak that CPython's allocator never returns
    to the OS , so the peak itself has to go, not just its growth rate.

    Nothing downstream ever wanted the bytes. `_parse_sse_telemetry_timed()`
    reduces them to six numbers, so this consumes each chunk on arrival and
    keeps only those numbers plus whatever tail of a split line is still
    incomplete.

    Bytes are still forwarded to the client first and unmodified; this only
    observes. It must never raise , a telemetry parse failure cannot be allowed
    to affect a response that was already delivered.
    """

    #: A line this long with no newline is not SSE. Cap the carry-over so a
    #: misbehaving upstream cannot reintroduce unbounded growth by never
    #: sending a delimiter.
    MAX_PARTIAL = 1 << 20      # 1 MiB

    __slots__ = ("ctok", "ptok", "tokens_cached", "t_first", "t_last",
                 "engine_prompt_ms", "_partial")

    def __init__(self) -> None:
        self.ctok = self.ptok = 0
        self.tokens_cached = None
        self.t_first = self.t_last = None
        self.engine_prompt_ms = None
        self._partial = ""

    def feed(self, ts: float, raw: bytes) -> None:
        try:
            self._partial += raw.decode("utf-8", "ignore")
        except Exception:
            return
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            self._consume(ts, line.strip())
        if len(self._partial) > self.MAX_PARTIAL:
            self._partial = ""

    def _consume(self, ts: float, line: str) -> None:
        if not line.startswith("data:"):
            return
        payload = line[len("data:"):].strip()
        if payload == "[DONE]" or not payload:
            return
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            return
        # `data: null` and `data: 3` are valid JSON but not objects. Calling
        # .get() on them raises AttributeError, which would break the "never
        # raises" contract this whole path depends on.
        if not isinstance(chunk, dict):
            return
        self.ctok, self.ptok = _usage_counts(chunk, (self.ctok, self.ptok))
        self.tokens_cached = _cache_info_from_chunk(chunk, self.tokens_cached)
        _t = chunk.get("timings")
        if isinstance(_t, dict) and _t.get("prompt_ms") is not None:
            self.engine_prompt_ms = _t.get("prompt_ms")
        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
        # tool_calls counts as generated content , an agentic turn can emit
        # nothing else, and leaving it out is what made ttft_ms never resolve.
        if (delta.get("content") or delta.get("reasoning_content")
                or delta.get("tool_calls") or delta.get("function_call")):
            if self.t_first is None:
                self.t_first = ts
            self.t_last = ts

    def result(self) -> "tuple[int, int, int | None, float | None, float | None, float | None]":
        return (self.ctok, self.ptok, self.tokens_cached,
                self.t_first, self.t_last, self.engine_prompt_ms)


def _parse_sse_telemetry_timed(
        chunks: "list[tuple[float, bytes]]") -> "tuple[int, int, int | None, float | None, float | None, float | None]":
    """Best-effort (completion_tokens, prompt_tokens, tokens_cached, t_first,
    t_last, engine_prompt_ms) from a list of (arrival_timestamp, raw_chunk) pairs — used by
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
    # llama-server's own prefill duration, from the `timings` block it puts on
    # the final stream chunk. Distinct from the wall-clock TTFT the caller
    # computes: TTFT is (first content token - request arrival at the proxy)
    # and therefore includes queueing behind another slot, proxy overhead, and
    # any contention on the box. engine_prompt_ms is the engine's own view of
    # how long prefill actually took. Recording only the first made a slow TTFT
    # unattributable — 2026-08-18 a 68 s TTFT was measured that turned out to
    # have a 32-thread kernel build running concurrently, and nothing in the
    # telemetry could separate "prefill is slow" from "the box was busy".
    engine_prompt_ms = None
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
            if not isinstance(chunk, dict):
                continue        # `data: null` is valid JSON, not an object
            ctok, ptok = _usage_counts(chunk, (ctok, ptok))
            tokens_cached = _cache_info_from_chunk(chunk, tokens_cached)
            _t = chunk.get("timings")
            if isinstance(_t, dict) and _t.get("prompt_ms") is not None:
                engine_prompt_ms = _t.get("prompt_ms")
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            # tool_calls counts as generated content. Leaving it out is what
            # made an agentic turn invisible: a turn whose whole output is a
            # tool call emits `delta.tool_calls` frames and never a single
            # `delta.content` one, so t_first stayed None — _record_tok_s then
            # discarded the sample as "no_token_timestamps" and
            # _record_prompt_cache omitted ttft_ms entirely. Live-confirmed
            # 2026-08-18: every recent prompt_cache event carried
            # engine_prompt_ms but no ttft_ms, which is exactly this shape, and
            # GB-Semantics' ttft_ms metric had therefore never resolved.
            # Because tool-call turns dominate real GB-CLI traffic, this also
            # biased the surviving tok_s population toward prose-only turns.
            # `function_call` is the pre-tools OpenAI spelling, kept for any
            # backend reachable through this passthrough that still emits it.
            if (delta.get("content") or delta.get("reasoning_content")
                    or delta.get("tool_calls") or delta.get("function_call")):
                if t_first is None:
                    t_first = ts
                t_last = ts
    return ctok, ptok, tokens_cached, t_first, t_last, engine_prompt_ms


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
    _record_tok_s(model, t_first, t_last, ctok, ptok)
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
    # GB-1, same as openai_passthrough: an Ollama-API client's conversation
    # deserves its slot kept just as much as an OpenAI-API one's.
    # Carry a conversation id the Ollama-API client supplied through to the
    # index (ollama's own schema has no field for it, so it arrives either as
    # an extra body key or as the header).
    if body.get("gb_conversation"):
        upstream_req["gb_conversation"] = body["gb_conversation"]
    _slot = await _pin_conversation_slot(upstream_req, model, request)
    if _slot is not None:
        upstream_req["id_slot"] = _slot
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
    _record_tok_s(model, t_first, t_last, ctok, ptok)
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

    is_completions_path = path in ("chat/completions", "completions")

    # GB-1: route this conversation back to the slot that still holds its KV.
    # Only on the chat path (a bare /v1/completions has no message structure to
    # identify), only when the caller has not already chosen a slot itself, and
    # only when the engine told us how many slots exist. This is the one place
    # the body is no longer forwarded byte-for-byte — one added integer field,
    # and only when a slot was actually chosen.
    if path == "chat/completions" and parsed and "id_slot" not in parsed:
        # Re-serialize if we either added id_slot or consumed the client's
        # conversation-id field — the upstream engine should never see a field
        # invented by this proxy.
        _had_conv = "gb_conversation" in parsed
        _slot = await _pin_conversation_slot(parsed, req_model, request)
        if _slot is not None:
            parsed["id_slot"] = _slot
        if _slot is not None or _had_conv:
            body = json.dumps(parsed).encode()

    if not stream:
        try:
            async with SESSION.post(url, data=body, headers=headers) as r:
                data = await r.read()
                if r.status < 400 and is_completions_path:
                    _record_non_stream_telemetry(req_model, data)
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
    # already forwarded. is_completions_path is set above, shared with the
    # non-streaming branch's own telemetry gate.
    t_start = time.monotonic()
    # (timestamp, raw_chunk) pairs, not a flat byte buffer — t_first/t_last
    # must be latched on the SSE event that actually carries generated
    # content, not on whichever raw TCP chunk arrives first (see
    # _parse_sse_telemetry_timed's docstring for the incident this fixes).
    # Still a pure observe-after-write: raw_chunk is forwarded unmodified
    # below regardless of what this list holds.
    # Consume telemetry as it streams instead of accumulating the whole
    # response. See _SSETelemetry: the old list was the only unbounded
    # per-request buffer here, and this proxy is long-lived , measured holding
    # 37.5 GB of anonymous memory after twenty minutes of ordinary traffic.
    _telemetry = _SSETelemetry() if is_completions_path else None
    try:
        async with SESSION.post(url, data=body, headers=headers) as r:
            if r.status >= 400:
                data = await r.read()
                return web.Response(body=data, status=r.status, content_type=r.content_type)
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            async with _StallWatch(req_model) as watch:
                async for raw_chunk in r.content.iter_any():
                    # Forward FIRST, observe second: telemetry must never sit
                    # between the engine and the client.
                    await resp.write(raw_chunk)
                    if raw_chunk:
                        watch.mark()
                        if _telemetry is not None:
                            _telemetry.feed(time.monotonic(), raw_chunk)
    except (aiohttp.ClientError, TimeoutError) as e:
        return web.json_response(_failure_body(req_model, str(e)), status=502)
    await resp.write_eof()
    if _telemetry is not None:
        ctok, ptok, tokens_cached, t_first, t_last, engine_prompt_ms = _telemetry.result()
        _record_tok_s(req_model, t_first, t_last, ctok, ptok)
        _record_prompt_cache(req_model, t_start, t_first, ptok, tokens_cached,
                             engine_prompt_ms=engine_prompt_ms)
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


#: llama.cpp native routes this proxy forwards verbatim. Read-only and
#: side-effect free , they answer questions about the loaded model and change
#: nothing, so forwarding them is safe and the alternative is worse.
#:
#: Why this exists (found 2026-08-20): `gb_aviary.niah_certify()` , the ONLY
#: quality gate that can detect quantized-KV damage, and the evidence
#: CLAUDE.md's "never below fp8 without gate evidence" rule depends on , calls
#: `/tokenize` to size its haystack. It targets gb-synapse's own port because
#: certifying against a different port would certify a different serving path.
#: But the proxy forwarded `/v1/*` and `/slots` only, so `/tokenize` 404'd and
#: certification could not run AT ALL. CLAUDE.md's note that niah_certify "also
#: not yet run against this checkpoint" was not an oversight: it was unable to.
_LLAMA_NATIVE_ROUTES = ("/tokenize", "/detokenize", "/props")


async def llama_native_passthrough(request: web.Request) -> web.Response:
    """Forward a llama.cpp native route to the upstream engine, unchanged."""
    query = f"?{request.query_string}" if request.query_string else ""
    url = f"{UPSTREAM}{request.path}{query}"
    if request.method == "GET":
        async with SESSION.get(url) as r:
            data = await r.read()
            return web.Response(body=data, status=r.status,
                                content_type=r.content_type)
    body = await request.read()
    async with SESSION.post(url, data=body,
                            headers={"Content-Type": "application/json"}) as r:
        data = await r.read()
        return web.Response(body=data, status=r.status,
                            content_type=r.content_type)


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
# Embeddings — the largest single API hole for :11369 before this: no
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


#: Fraction of the address-space cap at which the proxy dumps a heap census.
#: Low enough to fire well before the cap turns allocations into MemoryError,
#: high enough that a healthy proxy (tens of MB) never triggers it.
_HEAP_WATCH_FRACTION = float(os.environ.get("GB_SYNAPSE_HEAP_WATCH_FRACTION", "0.45"))
_HEAP_WATCH_INTERVAL_S = float(os.environ.get("GB_SYNAPSE_HEAP_WATCH_INTERVAL_S", "60"))


def _own_rss_bytes() -> int:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return 0


def _heap_census(top: int = 12) -> str:
    """Which Python types are holding the heap, by count and by retained size.

    This exists because reading the code did not find the leak. The proxy grew
    to 5.4 GB in 25 anonymous arenas of ~500 MB , unmistakably Python objects ,
    with only 4 sockets open and no unbounded module-level container anywhere
    in this file. Two OOM kills and one address-space cap later, the honest
    conclusion was that a census beats another hypothesis.

    gc.get_objects() is expensive, which is exactly why this runs only once per
    threshold crossing rather than on a timer.
    """
    import gc
    import sys as _sys
    from collections import Counter

    try:
        gc.collect()
        objs = gc.get_objects()
        by_count: "Counter[str]" = Counter()
        by_size: "Counter[str]" = Counter()
        for o in objs:
            name = type(o).__name__
            by_count[name] += 1
            try:
                by_size[name] += _sys.getsizeof(o)
            except Exception:
                pass
        lines = [f"[heap] {len(objs):,} tracked objects, RSS {_own_rss_bytes()/1024**2:.0f} MB"]
        lines.append("[heap] by count: " + ", ".join(
            f"{n}={c:,}" for n, c in by_count.most_common(top)))
        lines.append("[heap] by size:  " + ", ".join(
            f"{n}={sz/1024**2:.0f}MB" for n, sz in by_size.most_common(top)))
        return "\n".join(lines)
    except Exception as e:
        return f"[heap] census failed: {e}"


async def _start_heap_watch(app) -> None:
    """Dump a heap census once the proxy's own RSS crosses a fraction of its cap.

    Fires ONCE per crossing, not per interval , a census under memory pressure
    must not itself become the problem it is reporting on.
    """
    import resource

    try:
        cap, _ = resource.getrlimit(resource.RLIMIT_AS)
    except Exception:
        cap = resource.RLIM_INFINITY
    if cap in (resource.RLIM_INFINITY, 0):
        return          # no cap set (dev run) , nothing to measure against

    threshold = int(cap * _HEAP_WATCH_FRACTION)

    async def _watch() -> None:
        fired = False
        while True:
            await asyncio.sleep(_HEAP_WATCH_INTERVAL_S)
            rss = _own_rss_bytes()
            if rss >= threshold and not fired:
                fired = True
                print(f"[gb-synapse] proxy RSS {rss/1024**3:.2f} GiB crossed "
                      f"{_HEAP_WATCH_FRACTION:.0%} of its {cap/1024**3:.2f} GiB cap "
                      f", heap census follows. This is a LEAK REPORT, not a crash.",
                      flush=True)
                print(_heap_census(), flush=True)
            elif rss < threshold * 0.8:
                fired = False       # dropped back , re-arm for the next climb

    app["_heap_watch"] = asyncio.create_task(_watch())


def build_app() -> web.Application:
    app = web.Application(middlewares=[_auth_middleware])
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
    for _native in _LLAMA_NATIVE_ROUTES:
        app.router.add_route("*", _native, llama_native_passthrough)
    app.router.add_route("*", "/slots", slots_passthrough)
    app.router.add_route("*", "/slots/{path:.*}", slots_passthrough)
    app.on_startup.append(_on_startup)
    app.on_startup.append(_start_heap_watch)
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
    global UPSTREAM, MODEL_NAME, ENGINE, BIND
    ap = argparse.ArgumentParser(description="gb-synapse API proxy")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--upstream-port", type=int, required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--engine", default="", help="backend name (llama.cpp/torch/"
                     "transformers/diffusers) — carried into synapse_stall events")
    ap.add_argument("--bind", default="127.0.0.1", help="listen address; a "
                     "non-loopback bind requires GB_SYNAPSE_TOKEN or "
                     "/etc/greenboost/synapse_token to be set (default: "
                     "127.0.0.1)")
    args = ap.parse_args()

    UPSTREAM = f"http://127.0.0.1:{args.upstream_port}"
    MODEL_NAME = args.model_name
    ENGINE = args.engine
    BIND = args.bind

    if not _is_loopback(BIND) and not _resolve_token():
        raise SystemExit(
            f"refusing to bind non-loopback {BIND}:{args.port} without a "
            f"token — set {_TOKEN_ENV} or populate {_TOKEN_FILE}, or bind "
            f"127.0.0.1.")

    print(f"[gb-synapse-api] listening on http://{BIND}:{args.port}/ "
          f"(auth={'token' if _resolve_token() else 'loopback-only'})",
          flush=True)
    _start_flight_recorder()
    web.run_app(build_app(), host=BIND, port=args.port, print=None)


if __name__ == "__main__":
    main()

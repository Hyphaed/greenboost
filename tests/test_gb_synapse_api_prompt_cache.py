"""Tests for proxy-owned prompt-cache telemetry in gb_synapse_api.

Regression guard for a real bug found via a live serve test against the
Fable-Fusion reference model (2026-07-29): `_cache_info_from_chunk` only
checked a top-level `tokens_cached` field (llama-server's native /completion
shape), but GB-CLI's actual traffic goes through `openai_passthrough()` ->
llama-server's OpenAI-compat /v1/chat/completions, which nests the same
information at `usage.prompt_tokens_details.cached_tokens` instead. Before
this fix, every real GB-CLI request produced a `prompt_cache` dataflux event
with `reused_tokens=0` and no `hit_pct` at all, even while the cache was
genuinely working (confirmed live: cold TTFT ~1866-2086ms, warm TTFT
~336-354ms, hit_pct 89.2% once the extraction was fixed).
"""
import pytest

pytest.importorskip("aiohttp")

import gb_synapse_api as api


def test_cache_info_from_chunk_top_level_shape():
    """llama-server's native /completion (non-oaicompat) shape."""
    assert api._cache_info_from_chunk({"tokens_cached": 42}, None) == 42
    assert api._cache_info_from_chunk({}, 42) == 42  # no field, keeps previous


def test_cache_info_from_chunk_openai_compat_nested_shape():
    """The REAL shape GB-CLI's traffic actually produces (BACKEND_REGISTRY
    talks /v1/chat/completions -> openai_passthrough) — this is the exact
    path that was silently broken before the 2026-07-29 fix."""
    chunk = {"usage": {"completion_tokens": 20, "prompt_tokens": 37,
                       "prompt_tokens_details": {"cached_tokens": 33}}}
    assert api._cache_info_from_chunk(chunk, None) == 33


def test_cache_info_from_chunk_top_level_wins_over_nested():
    chunk = {"tokens_cached": 10,
             "usage": {"prompt_tokens_details": {"cached_tokens": 99}}}
    assert api._cache_info_from_chunk(chunk, None) == 10


def test_cache_info_from_chunk_null_keeps_previous():
    chunk = {"usage": {"prompt_tokens_details": {}}}
    assert api._cache_info_from_chunk(chunk, 15) == 15


def test_parse_sse_telemetry_timed_realistic_stream():
    """A realistic OpenAI-compat SSE byte stream, as actually forwarded by
    openai_passthrough — a role-only opening delta (no content — must NOT
    move t_first, this is the 2026-08-01 regression this replaces
    _parse_sse_telemetry_buffer to fix), then content chunks, then a final
    usage/cache frame."""
    chunks = [
        (100.0, b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'),
        (100.2, b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'),
        (100.7, b'data: {"choices":[{"delta":{"content":" there"}}]}\n\n'),
        (100.9, b'data: {"choices":[],"usage":{"completion_tokens":8,"prompt_tokens":120,'
                b'"prompt_tokens_details":{"cached_tokens":115}}}\n\n'),
        (100.9, b'data: [DONE]\n\n'),
    ]
    (ctok, ptok, tokens_cached, t_first, t_last,
     engine_prompt_ms) = api._parse_sse_telemetry_timed(chunks)
    assert (ctok, ptok, tokens_cached) == (8, 120, 115)
    assert t_first == pytest.approx(100.2)   # NOT the 100.0 role-only frame
    assert t_last == pytest.approx(100.7)


def test_parse_sse_telemetry_timed_malformed_is_safe():
    """Malformed/partial lines must never raise — telemetry parsing runs
    AFTER the real bytes were already forwarded to the client unchanged."""
    assert api._parse_sse_telemetry_timed(
        [(1.0, b"data: {not json}\n\ngarbage\n\ndata: [DONE]\n\n")]) == (0, 0, None, None, None, None)
    assert api._parse_sse_telemetry_timed([]) == (0, 0, None, None, None, None)


def test_record_prompt_cache_emits_hit_pct(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "gb_synapse.record_prompt_cache_sample",
        lambda model, ttft_ms, hit_pct, reused_tokens, **kw: captured.update(
            model=model, ttft_ms=ttft_ms, hit_pct=hit_pct,
            reused_tokens=reused_tokens, **kw))
    # t_start=10.0, t_first=10.35 -> ttft_ms=350; 33/37 tokens reused
    api._record_prompt_cache("qwen3", t_start=10.0, t_first=10.35,
                             prompt_tokens=37, tokens_cached=33)
    assert captured["ttft_ms"] == pytest.approx(350.0, abs=0.1)
    assert captured["hit_pct"] == pytest.approx(100.0 * 33 / 37, abs=0.1)
    assert captured["reused_tokens"] == 33


def test_record_prompt_cache_never_raises(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("store down")
    monkeypatch.setattr("gb_synapse.record_prompt_cache_sample", _boom)
    api._record_prompt_cache("m", 1.0, 1.5, 100, 50)  # must swallow the error


# ── engine prefill time vs wall TTFT (2026-08-18) ─────────────────────────────

def test_engine_prompt_ms_is_harvested_from_the_stream():
    """llama-server puts its own `timings` block on the final stream chunk.
    Recording only the client-observed TTFT made a slow prefill unattributable:
    a 68 s TTFT measured this day turned out to have a 32-thread kernel build
    running concurrently, and nothing in the telemetry could separate "prefill
    is slow" from "the box was busy"."""
    chunks = [
        (1.0, b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'),
        (2.0, b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'),
        (2.5, b'data: {"choices":[],"usage":{"completion_tokens":2,"prompt_tokens":900},'
              b'"timings":{"prompt_ms":6100.5,"prompt_n":900}}\n\n'),
        (2.6, b"data: [DONE]\n\n"),
    ]
    (ctok, ptok, cached, t_first, t_last,
     engine_prompt_ms) = api._parse_sse_telemetry_timed(chunks)
    assert ctok == 2 and ptok == 900
    assert t_first == 1.0 and t_last == 2.0
    assert engine_prompt_ms == pytest.approx(6100.5)


def test_absent_timings_block_stays_none():
    """Backends reachable through the same passthrough may not emit timings.
    Absent must stay distinguishable from zero, same convention as
    completion_tokens/prompt_tokens on tok_s_measured."""
    chunks = [(1.0, b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n')]
    assert api._parse_sse_telemetry_timed(chunks)[5] is None


def test_prompt_cache_event_carries_both_timings_and_the_depth(monkeypatch):
    """The pair is the point: ttft_ms minus engine_prompt_ms is everything that
    is not prefill, and prompt_tokens is the depth it was measured at."""
    import gb_synapse
    events = []
    monkeypatch.setattr("gb_dataflux.emit", lambda ev: events.append(ev))
    gb_synapse.record_prompt_cache_sample(
        "qwen3", ttft_ms=68300.0, hit_pct=96.5, reused_tokens=27520,
        engine_prompt_ms=6100.5, prompt_tokens=28500)
    ev = events[-1]
    assert ev["ttft_ms"] == pytest.approx(68300.0)
    assert ev["engine_prompt_ms"] == pytest.approx(6100.5)
    assert ev["prompt_tokens"] == 28500
    assert ev["hit_pct"] == pytest.approx(96.5)


def test_engine_prompt_ms_alone_is_enough_to_emit(monkeypatch):
    """The old early-return needed ttft or hit_pct; a sample carrying only the
    engine's prefill time is still worth recording."""
    import gb_synapse
    events = []
    monkeypatch.setattr("gb_dataflux.emit", lambda ev: events.append(ev))
    gb_synapse.record_prompt_cache_sample("qwen3", None, None, 0,
                                          engine_prompt_ms=1234.5)
    assert events and events[-1]["engine_prompt_ms"] == pytest.approx(1234.5)

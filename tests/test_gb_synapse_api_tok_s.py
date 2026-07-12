"""Tests for proxy-owned tok/s measurement in gb_synapse_api.

Exercises the measurement helpers directly (no live aiohttp server), plus a
simulation of the SSE-chunk accumulation loop that ollama_chat/ollama_generate
run, asserting the correct decode rate is handed to gb_synapse.record_measured_
tok_s.
"""
import pytest

pytest.importorskip("aiohttp")

import gb_synapse_api as api


def test_usage_completion_tokens_extracts_and_keeps_last():
    assert api._usage_completion_tokens({}, 0) == 0
    assert api._usage_completion_tokens({"usage": {"completion_tokens": 50}}, 0) == 50
    # a chunk without usage keeps the previous value
    assert api._usage_completion_tokens({"choices": [{}]}, 50) == 50
    # null usage keeps previous
    assert api._usage_completion_tokens({"usage": {"completion_tokens": None}}, 50) == 50


def test_record_tok_s_computes_decode_rate(monkeypatch):
    captured = {}
    monkeypatch.setattr("gb_synapse.record_measured_tok_s",
                        lambda model, tok_s: captured.update(model=model, tok_s=tok_s))
    # 101 tokens, first→last span 1.0s → 100 tok after the first / 1.0s = 100 tok/s
    api._record_tok_s("qwen3", t_first=10.0, t_last=11.0, completion_tokens=101)
    assert captured["model"] == "qwen3"
    assert abs(captured["tok_s"] - 100.0) < 1e-6


def test_record_tok_s_skips_incomplete(monkeypatch):
    calls = []
    monkeypatch.setattr("gb_synapse.record_measured_tok_s",
                        lambda model, tok_s: calls.append((model, tok_s)))
    api._record_tok_s("m", None, 5.0, 10)          # no first-token time
    api._record_tok_s("m", 1.0, 1.0, 10)           # zero interval
    api._record_tok_s("m", 1.0, 2.0, 1)            # single token
    api._record_tok_s("m", 1.0, 2.0, 0)            # no usage
    assert calls == []


def test_record_tok_s_never_raises(monkeypatch):
    def _boom(model, tok_s):
        raise RuntimeError("store down")
    monkeypatch.setattr("gb_synapse.record_measured_tok_s", _boom)
    api._record_tok_s("m", 1.0, 2.0, 100)          # must swallow the error


def test_stream_loop_simulation(monkeypatch):
    """Mirror the ollama_chat accumulation: content chunks then a final usage
    chunk, and assert the recorded rate matches the observed span."""
    captured = {}
    monkeypatch.setattr("gb_synapse.record_measured_tok_s",
                        lambda model, tok_s: captured.update(model=model, tok_s=tok_s))

    chunks = [
        {"choices": [{"delta": {"content": "a"}}]},
        {"choices": [{"delta": {"content": "b"}}]},
        {"choices": [{"delta": {"content": "c"}}]},
        {"choices": [], "usage": {"completion_tokens": 3}},
    ]
    times = iter([100.0, 100.5, 101.0])   # monotonic() for the 3 content pieces
    monkeypatch.setattr(api.time, "monotonic", lambda: next(times))

    t_first = t_last = None
    ctok = 0
    for chunk in chunks:
        piece = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content", "")
        if piece:
            now = api.time.monotonic()
            if t_first is None:
                t_first = now
            t_last = now
        ctok = api._usage_completion_tokens(chunk, ctok)
    api._record_tok_s("qwen3", t_first, t_last, ctok)

    # (3-1) tokens / (101.0 - 100.0)s = 2 tok/s
    assert captured["tok_s"] == pytest.approx(2.0)

"""A turn that completes without producing a decode sample must still be
countable.

Measured 2026-08-18: two hours of heavy agentic use logged 7 completed turns
(prompt_cache events) and 2 tok_s_measured samples. The other 5 were rejected
by _MIN_TOK_S_SAMPLE_TOKENS, correctly — agentic turns are mostly short tool
calls, and a handful of tokens timed over milliseconds scores hundreds of
tok/s. But they vanished entirely, leaving an observer unable to distinguish
"throughput is unmeasured" from "nothing ran".
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("aiohttp")

import gb_dataflux
import gb_synapse
import gb_synapse_api as api


def test_short_streaming_turn_is_recorded_as_skipped(monkeypatch):
    events = []
    monkeypatch.setattr("gb_dataflux.emit", lambda ev: events.append(ev))
    monkeypatch.setattr("gb_synapse.record_measured_tok_s",
                        lambda *a, **k: pytest.fail("must not record a rate"))

    api._record_tok_s("qwen3", t_first=10.0, t_last=10.2, completion_tokens=5)

    assert len(events) == 1
    ev = events[0]
    assert ev["kind"] == "tok_s_measured"
    assert ev["status"] == "skipped"
    assert ev["skip_reason"] == "below_sample_floor"
    assert ev["completion_tokens"] == 5
    assert "tok_s" not in ev, "a skipped turn carries no rate"


def test_missing_timestamps_are_distinguished_from_a_short_turn(monkeypatch):
    events = []
    monkeypatch.setattr("gb_dataflux.emit", lambda ev: events.append(ev))
    api._record_tok_s("qwen3", t_first=None, t_last=None, completion_tokens=500)
    assert events[0]["skip_reason"] == "no_token_timestamps"


def test_short_non_streaming_turn_is_recorded_as_skipped(monkeypatch):
    events = []
    monkeypatch.setattr("gb_dataflux.emit", lambda ev: events.append(ev))
    monkeypatch.setattr("gb_synapse.record_prompt_cache_sample", lambda *a, **k: None)
    raw = json.dumps({
        "usage": {"completion_tokens": 4, "prompt_tokens": 900},
        "timings": {"predicted_per_second": 250.0},
    }).encode()
    api._record_non_stream_telemetry("qwen3", raw)
    skipped = [e for e in events if e.get("status") == "skipped"]
    assert skipped and skipped[0]["skip_reason"] == "below_sample_floor"


def test_skipped_samples_never_enter_the_rate_rollup(monkeypatch, tmp_path):
    """The mirror image of the outlier problem: averaging a valueless sample
    would drag every rate toward zero."""
    log = tmp_path / "dataflux.jsonl"
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(log))

    gb_synapse._df_emit_tok_s("qwen3", 10.0, "proxy", quant="q4", ctx=65536,
                              kv_type="f16", completion_tokens=100)
    gb_synapse._df_emit_tok_s("qwen3", 20.0, "proxy", quant="q4", ctx=65536,
                              kv_type="f16", completion_tokens=100)
    # Same serve config as the two rate samples above, so the skip count lands
    # on the SAME rollup key rather than a detached one.
    monkeypatch.setattr(gb_synapse, "_read_run_state",
                        lambda m: type("S", (), {"quant": "q4", "ctx": 65536,
                                                 "kv_type": "f16"})())
    for _ in range(5):
        gb_synapse.record_tok_s_skipped("qwen3", "below_sample_floor",
                                        completion_tokens=3, source="proxy")

    roll = gb_dataflux.summarize(gb_dataflux.read_events())["tok_s"]
    assert len(roll) == 1, f"skips must share the rate samples' key, got {list(roll)}"
    key = next(iter(roll))
    assert roll[key]["samples"] == 2, "skipped turns must not count as samples"
    assert roll[key]["avg"] == pytest.approx(15.0), "rate must be unaffected"
    assert roll[key]["skipped"] == 5, "but they must be countable"


def test_skipped_is_not_an_error(monkeypatch, tmp_path):
    """It is a normal outcome for a short turn, not an incident to page on."""
    log = tmp_path / "dataflux.jsonl"
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(log))
    gb_synapse.record_tok_s_skipped("qwen3", "below_sample_floor",
                                    completion_tokens=3, source="proxy")
    assert gb_dataflux.summarize(gb_dataflux.read_events())["errors"] == 0

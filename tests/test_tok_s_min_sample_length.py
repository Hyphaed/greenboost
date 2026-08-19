"""A turn too short to measure must not be recorded as a decode-rate sample.

`tok_s` is computed as (completion_tokens - 1) / inter-token-span, so a 2-token
reply is ONE interval and a 5-token reply is four — dominated by fixed
per-request overhead, not decode throughput. The only length guard used to be
`completion_tokens <= 1`, so ordinary short answers (a bare tool call, "DONE",
a one-word reply — abundant in an agentic session) were recorded as decode
measurements.

Live effect measured 2026-08-17 against a model whose real rate is ~5 tok/s: a
single 34-sample series contained 137.6, 148.1 and 283.8 tok/s, pulling the mean
to 30.6 and tripping gb_pilot's "decode degraded" advisory. That matters because
`recommend()` PREFERS measured history over its bandwidth heuristic, so junk
samples don't just misreport, they steer placement decisions.
"""
from __future__ import annotations

import pytest

import gb_synapse_api


@pytest.fixture
def recorded(monkeypatch):
    """Capture what reaches gb_synapse.record_measured_tok_s()."""
    calls: list = []

    class FakeSynapse:
        @staticmethod
        def record_measured_tok_s(model, tok_s, source="", **kw):
            calls.append({"model": model, "tok_s": tok_s, "source": source})

    import sys
    monkeypatch.setitem(sys.modules, "gb_synapse", FakeSynapse)
    return calls


def test_threshold_is_a_real_floor_not_one_token() -> None:
    assert gb_synapse_api._MIN_TOK_S_SAMPLE_TOKENS >= 8


def test_short_completion_is_not_recorded(recorded) -> None:
    # 3 tokens in 10ms reads as ~200 tok/s on a ~5 tok/s model.
    gb_synapse_api._record_tok_s("m", t_first=0.0, t_last=0.01, completion_tokens=3)
    assert recorded == []


def test_single_token_is_still_not_recorded(recorded) -> None:
    gb_synapse_api._record_tok_s("m", t_first=0.0, t_last=0.5, completion_tokens=1)
    assert recorded == []


def test_long_completion_is_recorded(recorded) -> None:
    # 101 tokens over 20s -> 5 tok/s, a real decode measurement.
    gb_synapse_api._record_tok_s("m", t_first=0.0, t_last=20.0, completion_tokens=101)
    assert len(recorded) == 1
    assert recorded[0]["tok_s"] == pytest.approx(5.0)
    assert recorded[0]["source"] == "proxy"


def test_missing_timings_are_skipped(recorded) -> None:
    gb_synapse_api._record_tok_s("m", t_first=None, t_last=1.0, completion_tokens=100)
    gb_synapse_api._record_tok_s("m", t_first=0.0, t_last=None, completion_tokens=100)
    gb_synapse_api._record_tok_s("m", t_first=1.0, t_last=1.0, completion_tokens=100)
    assert recorded == []


def test_threshold_is_env_overridable(monkeypatch) -> None:
    """Deliberately short workloads can opt out, but the default protects the
    common case."""
    import importlib
    monkeypatch.setenv("GB_SYNAPSE_MIN_TOK_S_TOKENS", "4")
    mod = importlib.reload(gb_synapse_api)
    try:
        assert mod._MIN_TOK_S_SAMPLE_TOKENS == 4
    finally:
        monkeypatch.delenv("GB_SYNAPSE_MIN_TOK_S_TOKENS", raising=False)
        importlib.reload(gb_synapse_api)

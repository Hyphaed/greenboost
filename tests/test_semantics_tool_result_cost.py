"""tool_result_prefill_ms — what carrying a tool's output costs.

Measured on a real session 2026-08-22: a Read of a 7.7 KB file executed in
6.3 ms and its result was ~1,938 tokens, about 45 s of prefill at this box's
rate. The cheap half was timed to the millisecond and the expensive half was
invisible. See casebook C-20.

Two things these pin, both learned by getting them wrong first:

* the rate must be MARGINAL, not overhead. One real turn missed 29 tokens for
  2,732 ms and implied 93 ms/token, because the ~1.3 s fixed per-request cost
  was divided over 29 tokens.
* the rate must come from RECENT turns. It is a function of context size
  (~4.4 ms/token at 1.6k, ~23 ms/token at 14-18k, because the model spills to
  T2 at that size), so a median over a long window describes neither.
"""
import pytest

import gb_semantics as S


def _events(monkeypatch, tool_calls, prompt_caches):
    def _r(**kw):
        return list(tool_calls) + list(prompt_caches)
    monkeypatch.setattr(S, "_read_events", _r)


def _tc(tokens, name="Read", ts=1):
    return {"kind": "cli_tool_call", "ts": ts, "name": name,
            "result_tokens_est": tokens}


def _pc(ts, reused, hit, ms):
    return {"kind": "prompt_cache", "ts": ts, "reused_tokens": reused,
            "hit_pct": hit, "engine_prompt_ms": ms}


def test_none_without_any_sized_tool_result(monkeypatch):
    _events(monkeypatch, [], [_pc(1, 10000, 90.0, 20000)])
    r = S.resolve("tool_result_prefill_ms")
    assert r["value"] is None
    assert "result size" in r["provenance"]["raw_source"]


def test_none_without_a_rate(monkeypatch):
    _events(monkeypatch, [_tc(1000)], [])
    r = S.resolve("tool_result_prefill_ms")
    assert r["value"] is None
    assert "ms/token rate" in r["provenance"]["raw_source"]


def test_cost_is_tokens_times_measured_rate(monkeypatch):
    # 10000 reused at 90% -> 11111 total -> 1111 missed; 22220 ms -> 20 ms/token
    _events(monkeypatch, [_tc(100)], [_pc(1, 10000, 90.0, 22220)])
    r = S.resolve("tool_result_prefill_ms")
    assert r["details"]["ms_per_token"] == pytest.approx(20.0, abs=0.5)
    assert r["value"] == pytest.approx(2000.0, rel=0.05)


def test_tiny_miss_counts_cannot_set_the_rate(monkeypatch):
    """29 missed tokens for 2732 ms is fixed cost, not 93 ms/token."""
    _events(monkeypatch, [_tc(1000)], [_pc(1, 5827, 99.5, 2732)])
    r = S.resolve("tool_result_prefill_ms")
    assert r["value"] is None, "a 29-token miss must not establish the rate"


def test_the_largest_result_is_the_one_reported(monkeypatch):
    _events(monkeypatch, [_tc(100, "Read", 1), _tc(5000, "Bash", 2)],
            [_pc(1, 10000, 90.0, 22220)])
    r = S.resolve("tool_result_prefill_ms")
    assert r["details"]["tool"] == "Bash"
    assert r["details"]["tokens"] == 5000


def test_recent_turns_win_over_old_ones(monkeypatch):
    """Old small-context turns must not dilute the current rate."""
    old = [_pc(i, 1000, 50.0, 2000) for i in range(1, 8)]      # 2 ms/token
    new = [_pc(100 + i, 10000, 90.0, 22220) for i in range(5)]  # 20 ms/token
    _events(monkeypatch, [_tc(100)], old + new)
    r = S.resolve("tool_result_prefill_ms")
    assert r["details"]["ms_per_token"] == pytest.approx(20.0, abs=1.0)


def test_segment_is_none_before_anything_is_sized(monkeypatch):
    _events(monkeypatch, [], [])
    assert S.evaluate_segment("tool_result_dominates_turn")["matched"] is None


@pytest.mark.parametrize("tokens,expected", [(100, False), (5000, True)])
def test_segment_fires_above_ten_seconds(monkeypatch, tokens, expected):
    _events(monkeypatch, [_tc(tokens)], [_pc(1, 10000, 90.0, 22220)])
    assert S.evaluate_segment("tool_result_dominates_turn")["matched"] is expected


def test_schema_declares_the_new_numerics():
    import gb_dataflux_kinds as k
    spec = k.KINDS["cli_tool_call"]
    for f in ("result_chars", "result_tokens_est"):
        assert f in spec.numeric_fields

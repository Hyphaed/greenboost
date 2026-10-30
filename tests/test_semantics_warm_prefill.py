"""warm_prefill_fixed_ms — the cost paid before the first output token.

This file replaces test_semantics_replay_share.py, and the reason is worth
keeping. That metric reported "66.7% of prefill is recurrent replay" by dividing
turn_bench's median_replay_ms_est by median_prompt_ms. But real_prefill_ms_est is
a CONSTANT (443.8 in every run on record), so replay_ms_est is prompt_ms minus a
constant, and the ratio was arithmetic wearing a measurement's clothes.

A length sweep killed it in three runs: across a 9.4x prompt-length range
prompt_ms moved 1273.1 -> 1283.6 ms. Whole-prompt recurrent replay would scale
nearly linearly with length. It does not happen on this engine.

What survives is the fixed per-request cost, which is real, large, and
actionable: ~1.3 s independent of prompt length AND output length.
"""
import pytest

import gb_semantics as S


def _bench(monkeypatch, events):
    monkeypatch.setattr(S, "_events_in_window",
                        lambda kind, node=None, max_age_s=None, days=2.0:
                        events if kind == "turn_bench" else [])


def _ev(ts, prompt_ms, model="m"):
    return {"kind": "turn_bench", "ts": ts, "model": model,
            "median_prompt_ms": prompt_ms}


def test_no_runs_is_none_and_names_the_fix(monkeypatch):
    _bench(monkeypatch, [])
    r = S.resolve("warm_prefill_fixed_ms")
    assert r["value"] is None
    assert "gb_bench_turn" in r["provenance"]["raw_source"]


def test_uses_the_latest_run(monkeypatch):
    _bench(monkeypatch, [_ev(1, 5000.0), _ev(2, 1300.0)])
    assert S.resolve("warm_prefill_fixed_ms")["value"] == 1300.0


def test_reports_measured_not_derived(monkeypatch):
    """The metric this replaced was derived and said it was measured."""
    _bench(monkeypatch, [_ev(1, 1337.8)])
    r = S.resolve("warm_prefill_fixed_ms")
    assert r["details"]["basis"] == "measured"
    assert r["value"] == 1337.8


def test_events_without_the_field_are_not_counted(monkeypatch):
    _bench(monkeypatch, [{"kind": "turn_bench", "ts": 1, "model": "m"}])
    r = S.resolve("warm_prefill_fixed_ms")
    assert r["value"] is None
    assert "none carrying" in r["provenance"]["raw_source"]


def test_zero_is_not_a_valid_measurement(monkeypatch):
    _bench(monkeypatch, [_ev(1, 0.0)])
    assert S.resolve("warm_prefill_fixed_ms")["value"] is None


def test_segment_is_none_before_any_benchmark(monkeypatch):
    _bench(monkeypatch, [])
    assert S.evaluate_segment("warm_prefill_dominates_turn")["matched"] is None


@pytest.mark.parametrize("ms,expected", [(400.0, False), (1337.8, True)])
def test_segment_fires_above_one_second(monkeypatch, ms, expected):
    _bench(monkeypatch, [_ev(1, ms)])
    assert S.evaluate_segment("warm_prefill_dominates_turn")["matched"] is expected


def test_the_withdrawn_metric_is_gone(monkeypatch):
    """recurrent_replay_share must not resolve. Leaving it registered would keep
    a disproven number answerable, which is worse than never having had it."""
    r = S.resolve("recurrent_replay_share")
    assert r.get("governed") is False


def test_turn_bench_declares_its_numerics_as_numeric():
    import gb_dataflux_kinds as k
    spec = k.KINDS["turn_bench"]
    for f in ("median_prompt_ms", "median_replay_ms_est"):
        assert f in spec.numeric_fields

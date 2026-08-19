"""The benchmark's cost split must be honest about what it measures.

Phase 0 of the inference-speed program. Its whole job is to make later phases
provable, so the split itself is the thing under test: the engine reports ONE
`prompt_ms` and never breaks it down, so the recurrent-replay figure is derived,
and a derived number presented as a measurement is how a plausible story
outlives its evidence.
"""
from __future__ import annotations

import pytest

import gb_bench_turn as b


def test_warm_turn_attributes_almost_everything_to_replay():
    """The real reference-workload shape: 50k prompt, 42 new, ~6.9s prefill."""
    s = b.split_prefill(prompt_tokens=50366, reused_tokens=50324, prompt_ms=6146)
    assert s["new_tokens"] == 42
    # 42 tokens at the cold rate is well under a second; the rest is replay.
    assert s["real_prefill_ms_est"] == pytest.approx(42 * 15.85, rel=0.01)
    assert s["recurrent_replay_ms_est"] > 5000
    assert s["replay_share"] > 0.85


def test_cold_turn_attributes_nothing_to_replay():
    """Zero reuse means every token is genuinely new — there is no replay to
    reclaim, and claiming otherwise would invent a saving that isn't there."""
    s = b.split_prefill(prompt_tokens=7906, reused_tokens=0, prompt_ms=125317)
    assert s["new_tokens"] == 7906
    # Not exactly zero: the calibration rate was derived from this very sample,
    # so it lands within rounding of the whole measured time. Negligible is the
    # honest assertion — asserting 0.0 would be pinning a rounding artifact.
    assert s["replay_share"] < 0.01
    assert s["recurrent_replay_ms_est"] < 100


def test_real_prefill_never_exceeds_measured_total():
    """If a box prefills FASTER than the calibration rate, the estimate must
    clamp rather than produce a negative replay figure."""
    s = b.split_prefill(prompt_tokens=1000, reused_tokens=0, prompt_ms=500)
    assert s["real_prefill_ms_est"] == 500
    assert s["recurrent_replay_ms_est"] == 0.0


def test_ms_per_new_token_is_none_when_nothing_is_new():
    """A fully-cached turn has no new tokens; the per-new-token rate is
    undefined, not zero, and reporting 0 would read as 'free'."""
    s = b.split_prefill(prompt_tokens=5000, reused_tokens=5000, prompt_ms=900)
    assert s["ms_per_new_token"] is None
    assert s["new_tokens"] == 0
    assert s["recurrent_replay_ms_est"] == 900


def test_ratio_is_the_diagnostic_that_exposed_the_bug():
    """ms-per-TOTAL-token stays flat across turns with very different new-token
    counts; ms-per-NEW-token scatters. That contrast is what made the replay
    cost visible, so it has to survive refactors."""
    a = b.split_prefill(49486, 49444, 6317)
    c = b.split_prefill(52882, 52843, 6471)
    assert abs(a["ms_per_total_token"] - c["ms_per_total_token"]) < 0.02
    assert abs(a["ms_per_new_token"] - c["ms_per_new_token"]) > 5


def test_zero_prompt_tokens_does_not_divide_by_zero():
    s = b.split_prefill(0, 0, 0)
    assert s["ms_per_total_token"] == 0.0
    assert s["replay_share"] == 0.0


def test_run_reports_a_clear_error_without_a_server(monkeypatch):
    """A benchmark that fails must say why, not emit an empty result that
    reads as 'no regression'."""
    import gb_synapse
    monkeypatch.setattr(gb_synapse, "ps", lambda: [])
    res = b.run(model="", emit=False)
    assert "error" in res and "no serve session" in res["error"]


# ── fit verification ─────────────────────────────────────────────────────────

def test_unknown_overflow_is_never_reported_as_a_fit(monkeypatch):
    """The failure this function exists to prevent.

    On 2026-08-05 a quant was compared on tok/s alone, concluded slower, and
    written off — when it had actually been spilling the whole time. Missing
    telemetry must read as "cannot tell", never as a clean fit.
    """
    import gb_semantics
    monkeypatch.setattr(gb_semantics, "resolve",
                        lambda name, **kw: {"value": None, "provenance": {}})
    out = b.verify_zero_spill()
    assert out["fits"] is None
    assert "cannot tell" in out["verdict"]


def test_spilling_model_is_called_out_as_pcie_bound(monkeypatch):
    import gb_semantics
    vals = {"t2_overflow_active_mb": 4600.0, "vram_fill_pct": 88.0}
    monkeypatch.setattr(gb_semantics, "resolve",
                        lambda name, **kw: {"value": vals.get(name), "provenance": {}})
    out = b.verify_zero_spill()
    assert out["fits"] is False
    assert "PCIe-bound" in out["verdict"]


def test_zero_overflow_is_a_fit(monkeypatch):
    import gb_semantics
    vals = {"t2_overflow_active_mb": 0.0, "vram_fill_pct": 91.0}
    monkeypatch.setattr(gb_semantics, "resolve",
                        lambda name, **kw: {"value": vals.get(name), "provenance": {}})
    out = b.verify_zero_spill()
    assert out["fits"] is True
    assert "VRAM-bound" in out["verdict"]


def test_summary_always_carries_the_fit_verdict(monkeypatch):
    """A tok/s number must never ship without saying whether the model fit."""
    import inspect
    src = inspect.getsource(b.run)
    assert 'summary["fit"] = verify_zero_spill(' in src


# ── physical plausibility of the spill figure ────────────────────────────────

def test_impossible_spill_is_caught():
    """The real case: 5,923 MB reported alongside a measured 8.37 tok/s implies
    48.4 GB/s across a link measured at 24.43. The decode rate is end-to-end and
    not in doubt, so it is the overflow figure that is wrong."""
    r = b.spill_is_physically_possible(5923, 8.37)
    assert r["possible"] is False
    assert r["implied_gbs"] > 24.43
    assert "over-reported" in r["note"]


def test_plausible_spill_passes_quietly():
    r = b.spill_is_physically_possible(39, 45.72)
    assert r["possible"] is True and r["note"] is None


def test_it_says_what_the_maximum_plausible_overflow_would_be():
    """Gives the reader a number to compare against, not just a complaint."""
    r = b.spill_is_physically_possible(5923, 8.37)
    assert 2900 < r["max_plausible_overflow_mb"] < 3100


def test_missing_inputs_skip_the_check_rather_than_guessing():
    for args in ((0, 8.37), (5923, 0), (None, 8.37), (5923, None)):
        assert b.spill_is_physically_possible(*args)["checked"] is False


def test_link_bandwidth_is_overridable_per_box():
    """24.43 GB/s is THIS board's measurement, not a universal constant."""
    slow = b.spill_is_physically_possible(5923, 8.37, link_gbs=60.0)
    assert slow["possible"] is True, "a faster link makes the same figure plausible"


def test_fit_verdict_surfaces_the_contradiction(monkeypatch):
    import gb_semantics
    vals = {"t2_overflow_active_mb": 5923.0, "vram_fill_pct": 89.0}
    monkeypatch.setattr(gb_semantics, "resolve",
                        lambda name, **kw: {"value": vals.get(name), "provenance": {}})
    out = b.verify_zero_spill(decode_tok_s=8.37)
    assert "BUT" in out["verdict"], "an impossible figure was reported as fact"


# ── GB-0: what an edited prefix costs ────────────────────────────────────────

def _fake_turn_factory(seen):
    """Record the messages each turn was asked to prefill, and charge a
    prefill cost that mimics a real server: cheap when the prefix is
    unchanged, expensive when it is not."""
    prev = {"prefix": None}

    def _turn(base_url, model, messages, max_tokens):
        text = "".join(m["content"] for m in messages)
        seen.append(text)
        warm = prev["prefix"] is not None and text.startswith(prev["prefix"])
        prev["prefix"] = text
        return {"prompt_tokens": 100, "new_tokens": 10,
                "prompt_ms": 100.0 if warm else 1000.0,
                "total_ms": 200.0, "decode_tok_s": 5.0,
                "reused_tokens": 90 if warm else 0,
                "recurrent_replay_ms_est": 50.0, "replay_share": 0.5,
                "ms_per_total_token": 2.0, "ms_per_new_token": 20.0}
    return _turn


def test_edit_at_actually_changes_an_early_message(monkeypatch):
    """The flag has to mutate the FRONT of the conversation. Appending a
    '(revised)' at the end would measure nothing — every turn already
    appends."""
    seen = []
    monkeypatch.setattr(b, "_turn", _fake_turn_factory(seen))
    monkeypatch.setattr(b, "verify_zero_spill", lambda tok_s: {})
    b.run(model="m", turns=4, edit_at=3, emit=False)
    assert "(revised)" not in seen[1]
    assert "(revised)" in seen[2]


def test_edit_cost_reports_the_penalty_against_the_warm_turns(monkeypatch):
    """The number GB-1 has to beat is a ratio, not a raw millisecond count —
    it must be stated, not left for the reader to eyeball."""
    monkeypatch.setattr(b, "_turn", _fake_turn_factory([]))
    monkeypatch.setattr(b, "verify_zero_spill", lambda tok_s: {})
    res = b.run(model="m", turns=4, edit_at=3, emit=False)
    ec = res["summary"]["edit_cost"]
    assert ec["edited_at_turn"] == 3
    assert ec["penalty_x"] == 10.0          # 1000 ms vs a 100 ms warm median
    assert ec["reused_tokens_after"] < ec["reused_tokens_before"]
    assert res["turns"][2]["edited_here"] is True


def test_no_edit_cost_block_when_no_edit_was_asked_for(monkeypatch):
    """A run that did not edit anything must not carry an edit verdict —
    an absent measurement is not a zero one."""
    monkeypatch.setattr(b, "_turn", _fake_turn_factory([]))
    monkeypatch.setattr(b, "verify_zero_spill", lambda tok_s: {})
    res = b.run(model="m", turns=3, emit=False)
    assert "edit_cost" not in res["summary"]
    assert all(not r.get("edited_here") for r in res["turns"])

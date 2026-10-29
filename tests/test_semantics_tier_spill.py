"""tier_spill_return_pct — did a demotion have to be undone?

Ported from TensorRT-LLM's KV-cache stats schema (crosswalk P5-01), which
counts pages offloaded to the host tier and dropped without ever coming back.
GreenBoost inverts which side is the loss, and these tests pin that inversion:
a demotion that never returns is the HEALTHY case here (the component was
finished and the VRAM was freed), while a round trip means the eviction policy
released something still live.

The contract that matters most is the same one every resolver in this layer
has: with no demotions it must answer "I cannot tell", never a reassuring 0%.
"""
import pytest

import gb_semantics as S


def _moves(monkeypatch, events):
    monkeypatch.setattr(S, "_events_in_window",
                        lambda kind, node=None, max_age_s=None, days=2.0:
                        events if kind == "tier_move" else [])


def _mv(ts, item, frm, to, gb=1.0, pid=1):
    return {"kind": "tier_move", "ts": ts, "pid": pid, "items": [item],
            "from_tier": frm, "to_tier": to, "size_gb": gb}


T1, T2 = "T1_HBM", "T2_DDR"


def test_no_events_is_none_not_zero(monkeypatch):
    _moves(monkeypatch, [])
    r = S.resolve("tier_spill_return_pct")
    assert r["value"] is None
    assert "no tier_move" in r["provenance"]["raw_source"]


def test_moves_without_a_demotion_is_none_and_says_so(monkeypatch):
    # A promotion with no prior demotion: something was moved, nothing spilled.
    _moves(monkeypatch, [_mv(1, "vae", T2, T1)])
    r = S.resolve("tier_spill_return_pct")
    assert r["value"] is None
    assert "no T1 demotion" in r["provenance"]["raw_source"]


def test_demotion_that_never_returns_scores_zero(monkeypatch):
    """The healthy case: freed VRAM, never needed back."""
    _moves(monkeypatch, [_mv(1, "vae", T1, T2, gb=2.0)])
    r = S.resolve("tier_spill_return_pct")
    assert r["value"] == 0.0
    assert r["details"]["never_returned"] == 1
    assert r["details"]["gb_never_returned"] == 2.0


def test_round_trip_scores_one_hundred(monkeypatch):
    _moves(monkeypatch, [_mv(1, "unet", T1, T2, gb=8.0), _mv(2, "unet", T2, T1)])
    r = S.resolve("tier_spill_return_pct")
    assert r["value"] == 100.0
    assert r["details"]["returned"] == 1
    assert r["details"]["gb_returned"] == 8.0


def test_matches_the_real_measured_shape(monkeypatch):
    """Three round trips of one component against nine one-way demotions of
    others — the pattern actually measured on this box (25%)."""
    ev = []
    t = 0
    for _ in range(3):
        t += 1; ev.append(_mv(t, "transformer", T1, T2, gb=16.91))
        t += 1; ev.append(_mv(t, "transformer", T2, T1, gb=16.91))
        for item in ("vae", "text_encoder", "extra"):
            t += 1; ev.append(_mv(t, item, T1, T2, gb=1.0))
    _moves(monkeypatch, ev)
    r = S.resolve("tier_spill_return_pct")
    assert r["value"] == 25.0
    assert r["details"]["demotions"] == 12
    assert r["details"]["returned"] == 3


def test_promotion_does_not_credit_a_different_process(monkeypatch):
    """Two processes can hold same-named components; a promotion in one says
    nothing about a demotion in the other."""
    _moves(monkeypatch, [_mv(1, "unet", T1, T2, pid=1), _mv(2, "unet", T2, T1, pid=2)])
    r = S.resolve("tier_spill_return_pct")
    assert r["value"] == 0.0
    assert r["details"]["never_returned"] == 1


def test_segment_is_none_when_nothing_was_spilled(monkeypatch):
    _moves(monkeypatch, [])
    assert S.evaluate_segment("tier_spill_returns_often")["matched"] is None


@pytest.mark.parametrize("returns,expected", [(1, False), (3, True)])
def test_segment_fires_only_above_half(monkeypatch, returns, expected):
    ev, t = [], 0
    for i in range(4):
        t += 1; ev.append(_mv(t, f"c{i}", T1, T2))
        if i < returns:
            t += 1; ev.append(_mv(t, f"c{i}", T2, T1))
    _moves(monkeypatch, ev)
    assert S.evaluate_segment("tier_spill_returns_often")["matched"] is expected


# ── the recorded basis, once gb_dataflux tags the move ───────────────────────

def _tagged(ts, item, outcome, resident=None, pid=1):
    e = _mv(ts, item, T2, T1 if outcome == "returned" else "T3_NVME")
    e["spill_outcome"] = outcome
    if resident is not None:
        e["spill_resident_s"] = resident
    return e


def test_recorded_outcome_beats_inference(monkeypatch):
    """Inference cannot tell "still in T2" from "evicted from T2". When the
    mover recorded which happened, that must win."""
    _moves(monkeypatch, [
        _mv(1, "unet", T1, T2), _tagged(2, "unet", "returned", resident=25.0),
        _mv(3, "vae", T1, T2), _tagged(4, "vae", "dropped", resident=80.0),
    ])
    r = S.resolve("tier_spill_return_pct")
    assert r["details"]["basis"] == "measured"
    assert r["value"] == 50.0
    assert r["details"]["dropped"] == 1
    assert "spill_outcome" in r["provenance"]["raw_source"]


def test_inferred_basis_is_labelled_as_such(monkeypatch):
    """Old events must still answer, but must not claim to be measured."""
    _moves(monkeypatch, [_mv(1, "unet", T1, T2), _mv(2, "unet", T2, T1)])
    r = S.resolve("tier_spill_return_pct")
    assert r["details"]["basis"] == "inferred"
    assert "INFERRED" in r["provenance"]["raw_source"]

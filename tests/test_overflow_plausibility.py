"""T2 allocation is not all per-token traffic, and readers must not assume it is.

Measured 2026-08-18: the shim reported t2_overflow_total_mb=5923 alongside an
end-to-end 8.37 tok/s. A dense model reads every weight every token, so that
implies 48.4 GB/s across a PCIe link measured at 24.43 — roughly 2x
over-reported. The decode rate is measured end to end and is not in doubt, so it
is the overflow that is wrong.

This is not a cosmetic issue. `rule1_underfilled` (the Rule #1 tripwire),
`weights_dont_fit_vram`, and every placement decision read this field. A layer
whose entire purpose is to stop agents reading trap fields must not itself hand
on a number that cannot be true.
"""
from __future__ import annotations

import pytest

import gb_semantics as gsem


def _with_tok_s(monkeypatch, tok_s):
    monkeypatch.setattr(gsem, "_latest_event_with_field",
                        lambda *a, **kw: {"tok_s": tok_s} if tok_s else None)


def test_the_real_impossible_case_is_flagged(monkeypatch):
    _with_tok_s(monkeypatch, 8.37)
    possible, ev = gsem._overflow_resident_vs_streamed(5923)
    assert possible is False
    assert ev["implied_gbs_if_all_streamed"] > ev["link_gbs"]
    assert 2900 < ev["max_streamed_mb"] < 3100


def test_a_plausible_overflow_passes(monkeypatch):
    _with_tok_s(monkeypatch, 45.72)
    possible, ev = gsem._overflow_resident_vs_streamed(39)
    assert possible is True
    assert ev["implied_gbs_if_all_streamed"] < ev["link_gbs"]


def test_no_decode_rate_means_cannot_tell_not_fine(monkeypatch):
    """Absence of a cross-check must never read as validation."""
    _with_tok_s(monkeypatch, None)
    possible, ev = gsem._overflow_resident_vs_streamed(5923)
    assert possible is None
    assert ev == {}


def test_zero_overflow_needs_no_check(monkeypatch):
    _with_tok_s(monkeypatch, 8.37)
    assert gsem._overflow_resident_vs_streamed(0)[0] is None


def test_resolver_marks_a_suspect_value_without_hiding_it(monkeypatch):
    """Keep the number — the direction of the error is known, the size is not —
    but make any reader see that it is a bound, not a measurement."""
    monkeypatch.setattr(gsem, "_shim_stats",
                        lambda w=None: ({"t2_overflow_total_mb": "5923",
                                         "_age_s": 1, "phase": "INFERENCE"}, "live"))
    _with_tok_s(monkeypatch, 8.37)
    r = gsem._res_t2_overflow_active_mb(None, None)
    assert r["value"] == 5923.0, "the raw reading is still reported"
    assert r["partly_resident"] is True
    assert "NOTE" in r["raw_source"]
    assert "resident, not hot" in r["raw_source"]


def test_resolver_leaves_a_plausible_value_unmarked(monkeypatch):
    monkeypatch.setattr(gsem, "_shim_stats",
                        lambda w=None: ({"t2_overflow_total_mb": "39",
                                         "_age_s": 1, "phase": "INFERENCE"}, "live"))
    _with_tok_s(monkeypatch, 45.72)
    r = gsem._res_t2_overflow_active_mb(None, None)
    assert "partly_resident" not in r
    assert "NOTE" not in r["raw_source"]


def test_link_bandwidth_is_this_box_not_a_constant():
    """24.43 GB/s is a measurement of THIS board's Gen4 x16 link."""
    assert 20 < gsem._PCIE_BULK_DMA_GBS < 30

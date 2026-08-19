"""Quant selection must optimise for zero spill, not for largest-that-fits.

The ordering principle comes from measurement, not preference. Same 21.27 GiB
model, same card, 2026-08-18:

    13,850 MB spilled  ->  18.81 tok/s
        39 MB spilled  ->  45.72 tok/s

2.4x purely from what crossed the bus. Against that, the quality gap between
adjacent quant levels is small — so a smaller quant that FITS beats a larger one
that streams. The old selection maximised file size against VRAM *total*, with
no term for KV, none for workspace, and no notion that spilling is catastrophic
rather than gradual.
"""
from __future__ import annotations

import pytest

import gb_synapse as gs


def _f(name: str, gib: float) -> dict:
    return {"filename": name, "size": int(gib * 1024 ** 3)}


CANDIDATES = [
    _f("model-UD-IQ2_XXS.gguf", 8.39),
    _f("model-UD-IQ2_M.gguf", 9.61),
    _f("model-UD-Q2_K_XL.gguf", 9.94),
    _f("model-UD-IQ3_XXS.gguf", 11.10),
    _f("model-UD-Q3_K_XL.gguf", 12.52),
    _f("model-IQ4_XS.gguf", 14.63),
    _f("model-Q8_0.gguf", 27.05),
]


def test_picks_the_largest_that_fits_with_zero_spill():
    """Best quality among the options that do not cross the bus."""
    pick = gs.select_quant_by_fit(CANDIDATES, budget_gb=11.5)
    assert pick["file"]["filename"] == "model-UD-IQ3_XXS.gguf"
    assert pick["fits"] is True


def test_a_tighter_budget_steps_down_rather_than_spilling():
    pick = gs.select_quant_by_fit(CANDIDATES, budget_gb=9.95)
    assert pick["file"]["filename"] == "model-UD-Q2_K_XL.gguf"
    assert pick["fits"] is True


def test_never_prefers_a_bigger_spilling_quant_over_a_fitting_one():
    """The regression this exists to prevent."""
    pick = gs.select_quant_by_fit(CANDIDATES, budget_gb=11.5)
    assert pick["file"]["size"] / 1024 ** 3 <= 11.5


def test_when_nothing_fits_it_says_so_instead_of_pretending():
    """A capacity fact the owner needs to act on, not a silent slow serve."""
    pick = gs.select_quant_by_fit(CANDIDATES, budget_gb=4.0)
    assert pick["fits"] is False
    assert pick["file"]["filename"] == "model-UD-IQ2_XXS.gguf", "smallest available"
    assert pick["overflow_gb"] > 0
    assert "spills" in pick["reason"]


def test_zero_sized_entries_are_ignored():
    """HF listings carry entries with no size; they must not win by accident."""
    files = CANDIDATES + [{"filename": "unsized.gguf", "size": None},
                          {"filename": "zero.gguf", "size": 0}]
    pick = gs.select_quant_by_fit(files, budget_gb=11.5)
    assert pick["file"]["filename"] == "model-UD-IQ3_XXS.gguf"


def test_empty_candidate_list_is_survivable():
    assert gs.select_quant_by_fit([], budget_gb=11.5) == {}
    assert gs.select_quant_by_fit([{"filename": "x", "size": 0}], budget_gb=11.5) == {}


def test_every_pick_explains_itself():
    for budget in (4.0, 9.95, 11.5, 99.0):
        assert gs.select_quant_by_fit(CANDIDATES, budget_gb=budget)["reason"]


def test_budget_subtracts_kv_and_workspace(monkeypatch):
    """VRAM total is not the weights budget: KV and graph workspace come first."""
    import gb_nvml
    class _M:
        def mem(self): return (0, 11.94 * 1024, 12.0 * 1024, 0)
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda i: _M())
    b = gs.zero_spill_weight_budget_gb(kv_gb=0.58)
    assert b < 11.94 - 0.58, "workspace allowance was not subtracted"
    assert b > 10.5, "allowance is implausibly large"


def test_budget_is_zero_when_the_gpu_cannot_be_read(monkeypatch):
    """Fail to 0 so the caller falls back rather than trusting a fabricated budget."""
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml",
                        lambda i: (_ for _ in ()).throw(RuntimeError("no nvml")))
    assert gs.zero_spill_weight_budget_gb() == 0.0

"""Calibration must measure REFERENCE attention, not GreenBoost's own.

gb_attn.patch_sdpa() replaces F.scaled_dot_product_attention globally. Running
calibration under it measures quantized attention against quantized attention,
so the per-layer sensitivities plan_quality() spends bits on are wrong — and
wrong silently, because the run still completes and still returns plausible
errors.

Adapted from unsloth's _compressed_quantize.py (study2 Q3-02), which gets the
same guarantee with a subprocess because its patches are import-time and
irreversible. GreenBoost's are reversible, so the guarantee costs a context
manager instead of a process.
"""
import torch.nn as nn
import pytest

import gb_attn
import gb_quant_calib as C


@pytest.fixture(autouse=True)
def _clean():
    gb_attn.unpatch_sdpa()
    yield
    gb_attn.unpatch_sdpa()


def test_context_reports_no_intervention_when_nothing_is_patched():
    with C.reference_attention() as intervened:
        assert intervened is False


def test_context_unpatches_an_active_sdpa_patch():
    gb_attn.patch_sdpa(k_bits=4, v_bits=3)
    assert gb_attn._original_sdpa is not None
    with C.reference_attention() as intervened:
        assert intervened is True
        assert gb_attn._original_sdpa is None, "reference attention must be live INSIDE"


def test_calibration_runs_against_reference_attention():
    gb_attn.patch_sdpa(k_bits=4, v_bits=3)
    m = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 32))
    res = C.calibrate_sensitivity(m, precisions=("fp8",), model_id="t_refattn",
                                  verbose=False, force_recompute=True)
    assert len(res) == 2
    assert gb_attn._original_sdpa is None


def test_intervention_is_recorded_not_silent(monkeypatch):
    """A calibration that had to unpatch is one whose caller believed something
    untrue. That must reach the flight recorder."""
    seen = []
    import gb_dataflux
    monkeypatch.setattr(gb_dataflux, "emit", lambda e: seen.append(e))
    gb_attn.patch_sdpa(k_bits=4, v_bits=3)
    with C.reference_attention():
        pass
    assert any(e.get("reason") == "calibration_unpatched_sdpa" for e in seen), seen
    assert any(e.get("status") == "warn" for e in seen)


def test_missing_gb_attn_degrades_quietly(monkeypatch):
    """Observation must never break calibration."""
    import builtins
    real = builtins.__import__

    def _boom(name, *a, **k):
        if name == "gb_attn":
            raise ImportError("simulated")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _boom)
    with C.reference_attention() as intervened:
        assert intervened is False

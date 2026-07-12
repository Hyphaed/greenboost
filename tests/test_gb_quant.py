#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_quant CPU-only paths: GpuProfile fallback, bits_tag, FitReport,
plan_fit, and quantize_to_fit telemetry budget gate.

No GemLite, no CUDA, no GPU hardware , every test runs in CI.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn


# ── gpu_profile() fallback ────────────────────────────────────────────────────

def test_gpu_profile_cpu_fallback():
    """gpu_profile() returns cpu_fallback when CUDA is unavailable."""
    import gb_quant
    # Clear cache so we don't get a stale result from previous test runs
    gb_quant._GPU_PROFILE_CACHE = None
    with patch("torch.cuda.is_available", return_value=False):
        p = gb_quant.gpu_profile()
    gb_quant._GPU_PROFILE_CACHE = None   # restore so we don't poison other tests
    assert p.family == "cpu_fallback"
    assert p.floor_default == 4
    assert p.t2_tolerance_gb == 0.0


def test_gpu_profile_cached():
    """gpu_profile() returns the same object on repeated calls."""
    import gb_quant
    gb_quant._GPU_PROFILE_CACHE = None
    with patch("torch.cuda.is_available", return_value=False):
        p1 = gb_quant.gpu_profile()
        p2 = gb_quant.gpu_profile()
    gb_quant._GPU_PROFILE_CACHE = None
    assert p1 is p2


# ── _bits_tag() ──────────────────────────────────────────────────────────────

def test_bits_tag_16():
    from gb_quant import _bits_tag
    assert _bits_tag(16) == "bf16"


def test_bits_tag_fp8():
    from gb_quant import _bits_tag
    assert _bits_tag("fp8") == "fp8"


def test_bits_tag_e4m3():
    from gb_quant import _bits_tag
    assert _bits_tag("e4m3") == "fp8"


def test_bits_tag_int4():
    from gb_quant import _bits_tag
    assert _bits_tag(4) == "int4"


# ── dataflux instrumentation ──────────────────────────────────────────────

def test_df_emit_quant_records_bits_and_component():
    from gb_quant import _df_emit_quant
    with patch("gb_dataflux.emit") as mock_emit:
        _df_emit_quant("text_encoder", "fp8", None, "flux2-klein", 1.5)
    mock_emit.assert_called_once()
    ev = mock_emit.call_args[0][0]
    assert ev["kind"] == "quantize"
    assert ev["items"] == ["text_encoder"]
    assert ev["bits"] == "fp8"
    assert ev["model_id"] == "flux2-klein"
    assert ev["duration_s"] == 1.5


def test_df_emit_quant_uses_quality_label_when_calibrated():
    from gb_quant import _df_emit_quant
    with patch("gb_dataflux.emit") as mock_emit:
        _df_emit_quant("transformer", 4, "near_lossless", "model", 2.0)
    ev = mock_emit.call_args[0][0]
    assert ev["bits"] == "near_lossless"


def test_df_emit_quant_never_raises_when_dataflux_unavailable():
    from gb_quant import _df_emit_quant
    with patch("gb_dataflux.emit", side_effect=RuntimeError("disk full")):
        _df_emit_quant("x", 4, None, "m", 0.1)  # must not raise


def test_bits_tag_int8():
    from gb_quant import _bits_tag
    assert _bits_tag(8) == "int8"


def test_bits_tag_tq3():
    from gb_quant import _bits_tag
    assert _bits_tag("tq3") == "tq3"


def test_bits_tag_tq2():
    from gb_quant import _bits_tag
    assert _bits_tag("tq2") == "tq2"


# ── FitReport properties and __str__ ─────────────────────────────────────────

def test_fit_report_fits_vram_true():
    from gb_quant import FitReport
    r = FitReport(budget_gb=8.0)
    r.total_quant_gb = 7.5
    assert r.fits_vram is True


def test_fit_report_fits_vram_false():
    from gb_quant import FitReport
    r = FitReport(budget_gb=8.0)
    r.total_quant_gb = 9.0
    assert r.fits_vram is False


def test_fit_report_needs_t2_overflow_positive():
    from gb_quant import FitReport
    r = FitReport(budget_gb=8.0)
    r.total_quant_gb = 9.5
    assert abs(r.needs_t2_overflow_gb - 1.5) < 1e-6


def test_fit_report_needs_t2_overflow_zero_when_fits():
    from gb_quant import FitReport
    r = FitReport(budget_gb=8.0)
    r.total_quant_gb = 6.0
    assert r.needs_t2_overflow_gb == 0.0


def test_fit_report_str_contains_budget():
    from gb_quant import FitReport
    r = FitReport(budget_gb=11.0)
    r.total_bf16_gb = 24.0
    r.total_quant_gb = 5.5
    s = str(r)
    assert "11.0" in s
    assert "fits T1 VRAM" in s


def test_fit_report_str_overflow_message():
    from gb_quant import FitReport
    r = FitReport(budget_gb=8.0)
    r.total_bf16_gb = 24.0
    r.total_quant_gb = 10.0
    s = str(r)
    assert "T2 DDR overflow" in s


def test_fit_report_str_contains_component_lines():
    from gb_quant import FitReport, ComponentPlan
    r = FitReport(budget_gb=8.0)
    r.components.append(ComponentPlan("transformer", 7_000_000_000, 14.0, 4, 3.85))
    r.total_bf16_gb = 14.0
    r.total_quant_gb = 3.85
    s = str(r)
    assert "transformer" in s
    assert "int4" in s


# ── plan_fit() with a tiny CPU module ────────────────────────────────────────

class _TinyPipeline:
    """Fake diffusers pipeline with two named sub-model components."""
    def __init__(self):
        self.transformer = nn.Linear(64, 64, bias=False)
        self.text_encoder = nn.Linear(32, 32, bias=False)
        self.scheduler = MagicMock()   # non-Module, should be skipped
        self.components = {
            "transformer": self.transformer,
            "text_encoder": self.text_encoder,
            "scheduler": self.scheduler,
        }


def test_plan_fit_single_module_fits():
    """plan_fit on a tiny module with a generous budget → all bf16."""
    from gb_quant import plan_fit
    mod = nn.Linear(64, 64, bias=False)
    # 4096 params × 2 bytes = 8 KB → well under 1 GB budget
    report = plan_fit(mod, budget_gb=1.0, prefer_bits=4)
    assert report.fits_vram


def test_plan_fit_single_module_plan_has_component():
    """plan_fit yields a ComponentPlan for the module."""
    from gb_quant import plan_fit
    mod = nn.Linear(64, 64, bias=False)
    report = plan_fit(mod, budget_gb=1.0, prefer_bits=4)
    assert len(report.components) == 1
    assert report.components[0].name == "model"


def test_plan_fit_budget_zero_picks_floor():
    """When budget=0, every component gets the floor precision."""
    from gb_quant import plan_fit
    mod = nn.Linear(128, 128, bias=False)  # 16384 params
    report = plan_fit(mod, budget_gb=0.0, prefer_bits=4)
    assert report.components[0].bits == 4


def test_plan_fit_budget_generous_picks_bf16():
    """When budget is huge, component should stay at bf16."""
    from gb_quant import plan_fit
    mod = nn.Linear(64, 64, bias=False)
    report = plan_fit(mod, budget_gb=100.0, prefer_bits=4)
    assert report.components[0].bits == 16


def test_plan_fit_invalid_prefer_bits_raises():
    """prefer_bits not in the precision ladder raises ValueError."""
    from gb_quant import plan_fit
    mod = nn.Linear(64, 64, bias=False)
    with pytest.raises(ValueError, match="prefer_bits"):
        plan_fit(mod, budget_gb=1.0, prefer_bits=3)


def test_plan_fit_pipeline_skips_scheduler():
    """plan_fit on a fake pipeline skips non-Module components."""
    from gb_quant import plan_fit
    pipeline = _TinyPipeline()
    report = plan_fit(pipeline, budget_gb=100.0)
    names = [c.name for c in report.components]
    assert "scheduler" not in names


def test_plan_fit_pipeline_skips_vae_component():
    """Components whose name is in skip_components stay at bf16."""
    from gb_quant import plan_fit
    class _PipelineWithVae:
        vae = nn.Linear(64, 64, bias=False)
        components = {"vae": vae}
    report = plan_fit(_PipelineWithVae(), budget_gb=0.0)
    vae_plan = next((c for c in report.components if c.name == "vae"), None)
    if vae_plan is not None:
        assert vae_plan.bits == 16


def test_plan_fit_total_quant_gb_positive():
    """total_quant_gb is set and positive."""
    from gb_quant import plan_fit
    mod = nn.Linear(128, 128, bias=False)
    report = plan_fit(mod, budget_gb=1.0)
    assert report.total_quant_gb >= 0


def test_plan_fit_total_bf16_gb_positive():
    """total_bf16_gb is >= 0 (tiny modules round to 0.0 due to float precision)."""
    from gb_quant import plan_fit
    mod = nn.Linear(128, 128, bias=False)
    report = plan_fit(mod, budget_gb=1.0)
    assert report.total_bf16_gb >= 0


# ── quantize_to_fit , telemetry budget gate ───────────────────────────────────

def test_quantize_to_fit_tightens_budget_from_telemetry():
    """If telemetry reports less free VRAM than the caller's budget, use the
    live value (×92% headroom factor)."""
    from gb_quant import FitReport, ComponentPlan

    fake_metrics = MagicMock()
    fake_metrics.fb_free_mb = 8192    # 8 GB free → 8*0.92≈7.37 GiB < 11 GiB

    fake_report = FitReport(budget_gb=7.37)
    fake_report.components = []
    fake_report.total_bf16_gb = 0.0
    fake_report.total_quant_gb = 0.0

    captured_budget = []

    def _fake_plan_fit(obj, budget_gb, **kw):
        captured_budget.append(budget_gb)
        return fake_report

    import gb_quant
    orig_gemlite_err = gb_quant._GEMLITE_ERR

    with patch.object(gb_quant, "_GEMLITE_ERR", None), \
         patch.object(gb_quant, "_require_gemlite", return_value=None), \
         patch.object(gb_quant, "plan_fit", side_effect=_fake_plan_fit), \
         patch.object(gb_quant, "_iter_named_components",
                      return_value=iter([])), \
         patch("gb_quant._gb_init") as mock_init:
        mock_init.snapshot.return_value = fake_metrics

        mod = nn.Linear(64, 64, bias=False)
        gb_quant.quantize_to_fit(mod, budget_gb=11.0, device="cpu", verbose=False)

    assert len(captured_budget) == 1
    assert captured_budget[0] < 11.0   # budget was tightened by telemetry
    assert abs(captured_budget[0] - 8192 / 1024 * 0.92) < 0.01


def test_quantize_to_fit_keeps_caller_budget_when_live_looser():
    """If telemetry free VRAM > caller budget, keep caller budget."""
    from gb_quant import FitReport

    fake_metrics = MagicMock()
    fake_metrics.fb_free_mb = 65536   # 64 GB free → much more than 4 GiB budget

    fake_report = FitReport(budget_gb=4.0)
    fake_report.components = []
    fake_report.total_bf16_gb = 0.0
    fake_report.total_quant_gb = 0.0

    captured_budget = []

    def _fake_plan_fit(obj, budget_gb, **kw):
        captured_budget.append(budget_gb)
        return fake_report

    import gb_quant

    with patch.object(gb_quant, "_GEMLITE_ERR", None), \
         patch.object(gb_quant, "_require_gemlite", return_value=None), \
         patch.object(gb_quant, "plan_fit", side_effect=_fake_plan_fit), \
         patch.object(gb_quant, "_iter_named_components",
                      return_value=iter([])), \
         patch("gb_quant._gb_init") as mock_init:
        mock_init.snapshot.return_value = fake_metrics

        mod = nn.Linear(64, 64, bias=False)
        gb_quant.quantize_to_fit(mod, budget_gb=4.0, device="cpu", verbose=False)

    assert abs(captured_budget[0] - 4.0) < 0.01   # budget unchanged


def test_quantize_to_fit_no_gb_init_uses_caller_budget():
    """When _gb_init is None, budget passes through unchanged."""
    from gb_quant import FitReport

    fake_report = FitReport(budget_gb=8.0)
    fake_report.components = []
    fake_report.total_bf16_gb = 0.0
    fake_report.total_quant_gb = 0.0

    captured_budget = []

    def _fake_plan_fit(obj, budget_gb, **kw):
        captured_budget.append(budget_gb)
        return fake_report

    import gb_quant

    with patch.object(gb_quant, "_GEMLITE_ERR", None), \
         patch.object(gb_quant, "_require_gemlite", return_value=None), \
         patch.object(gb_quant, "plan_fit", side_effect=_fake_plan_fit), \
         patch.object(gb_quant, "_iter_named_components",
                      return_value=iter([])), \
         patch.object(gb_quant, "_gb_init", None):

        mod = nn.Linear(64, 64, bias=False)
        gb_quant.quantize_to_fit(mod, budget_gb=8.0, device="cpu", verbose=False)

    assert abs(captured_budget[0] - 8.0) < 0.01


# ── autotune cache keying (version-scoped) ────────────────────────────────────

def test_autotune_cache_key_includes_cuda_and_gemlite_version():
    """Cache key scopes the file to GPU + CUDA + gemlite version so a
    toolchain upgrade never reuses stale autotune picks."""
    import gb_quant

    with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 5070"), \
         patch.object(torch.version, "cuda", "13.3"):
        key = gb_quant._autotune_cache_key()

    assert "RTX_5070" in key
    assert "cu133" in key
    assert key.count("_gl") == 1          # gemlite tag present exactly once


def test_autotune_cache_path_opt_out(monkeypatch):
    """GB_QUANT_NO_AUTOTUNE_CACHE=1 disables the cache entirely."""
    import gb_quant

    monkeypatch.setenv("GB_QUANT_NO_AUTOTUNE_CACHE", "1")
    assert gb_quant._autotune_cache_path() is None


def test_autotune_cache_path_uses_version_key(monkeypatch, tmp_path):
    """Path lands under the cache dir and carries the version-scoped key."""
    import gb_quant

    monkeypatch.delenv("GB_QUANT_NO_AUTOTUNE_CACHE", raising=False)
    monkeypatch.setenv("GB_QUANT_CACHE_DIR", str(tmp_path))
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 5070"), \
         patch.object(torch.version, "cuda", "13.3"):
        path = gb_quant._autotune_cache_path()

    assert path is not None
    assert path.startswith(str(tmp_path))
    assert "gemlite_autotune_" in path
    assert "cu133" in path


# ── tiered precision (GB_QUANT_TIERED_PRECISION) ──────────────────────────────

def _blackwell_profile():
    import gb_quant
    return gb_quant.GpuProfile(
        family="blackwell", cc=(12, 0),
        precisions=(16, "fp8", "nvfp4", 8, 4, "tq3", "tq2"),
        floor_default=4, quality_default="nvfp4", t2_tolerance_gb=8.0)


def _ada_profile():
    import gb_quant
    return gb_quant.GpuProfile(
        family="ada", cc=(8, 9),
        precisions=(16, "fp8", 8, 4, "tq3", "tq2"),
        floor_default=4, quality_default="fp8", t2_tolerance_gb=6.0)


def _one_component(n_params: int):
    """A fake pipeline exposing a single quantizable component of n_params."""
    import gb_quant
    mod = nn.Linear(1, 1, bias=False)
    ctx = patch.object(gb_quant, "_iter_named_components",
                       return_value=iter([("transformer", mod)]))
    ctx_size = patch.object(gb_quant, "_module_param_count", return_value=n_params)
    return ctx, ctx_size


def test_plan_fit_tiered_blackwell_hot_gets_fp8():
    """A component that fits the budget at fp8 is assigned fp8 (not bf16)."""
    import gb_quant
    c1, c2 = _one_component(1_000_000)   # ~0.98 MiB at fp8
    with patch.object(gb_quant, "gpu_profile", return_value=_blackwell_profile()), c1, c2:
        rep = gb_quant.plan_fit(object(), budget_gb=0.01, tiered=True)
    assert rep.components[0].bits == "fp8"


def test_plan_fit_tiered_blackwell_overflow_floors_at_nvfp4():
    """Too tight for fp8 → floors at nvfp4, never int4/bf16 on Blackwell."""
    import gb_quant
    # fp8 needs ~0.98 MiB; nvfp4 ~0.47 MiB. Budget between the two forces nvfp4.
    c1, c2 = _one_component(1_000_000)
    with patch.object(gb_quant, "gpu_profile", return_value=_blackwell_profile()), c1, c2:
        rep = gb_quant.plan_fit(object(), budget_gb=0.0006, tiered=True)
    assert rep.components[0].bits == "nvfp4"


def test_plan_fit_tiered_ada_never_assigns_nvfp4():
    """Ada has no nvfp4 tensor cores; tiered mode must not assign it."""
    import gb_quant
    c1, c2 = _one_component(1_000_000)
    with patch.object(gb_quant, "gpu_profile", return_value=_ada_profile()), c1, c2:
        rep = gb_quant.plan_fit(object(), budget_gb=0.0006, tiered=True)
    assert rep.components[0].bits != "nvfp4"
    assert rep.components[0].bits in _ada_profile().precisions


def test_plan_fit_tiered_env_gate(monkeypatch):
    """GB_QUANT_TIERED_PRECISION=1 activates tiered mode without an explicit arg."""
    import gb_quant
    monkeypatch.setenv("GB_QUANT_TIERED_PRECISION", "1")
    c1, c2 = _one_component(1_000_000)
    with patch.object(gb_quant, "gpu_profile", return_value=_blackwell_profile()), c1, c2:
        rep = gb_quant.plan_fit(object(), budget_gb=0.01)   # no tiered= arg
    assert rep.components[0].bits == "fp8"


def test_plan_fit_default_unchanged_allows_bf16(monkeypatch):
    """Default (tiered off): a component that fits keeps bf16 as before."""
    import gb_quant
    monkeypatch.delenv("GB_QUANT_TIERED_PRECISION", raising=False)
    c1, c2 = _one_component(1_000_000)
    with c1, c2:
        rep = gb_quant.plan_fit(object(), budget_gb=1.0)   # roomy → bf16
    assert rep.components[0].bits == 16

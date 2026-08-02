#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_quant_calib.calibrate_activations() / calibrate_diffusion()
(missing_features.md item (d)) , real-forward-pass per-layer sensitivity,
as opposed to calibrate_sensitivity's zero-data Frobenius weight proxy.

CPU-only, real forward passes: no mocking of the calibration math itself ,
these run genuine nn.Linear forward calls through real hooks and compare
real quantized-vs-unquantized outputs. Only calibrate_diffusion's `pipe`
object is a lightweight fake (a real diffusers pipeline needs GPU + real
weights, out of scope for this test tier; the fake exercises the WAN
dual-transformer iteration and the prompt-loop mechanics, which is exactly
what calibrate_diffusion adds beyond calibrate_activations itself).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn

import gb_quant_calib as gqc


class _TwoLayer(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.a = nn.Linear(64, 64, bias=False)
        self.b = nn.Linear(64, 64, bias=False)
        with torch.no_grad():
            self.a.weight.copy_(torch.randn(64, 64, generator=g))
            self.b.weight.copy_(torch.randn(64, 64, generator=g))

    def forward(self, x):
        return self.b(self.a(x))


@pytest.fixture(autouse=True)
def _isolated_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(gqc, "_CACHE_DIR", str(tmp_path))


# ── calibrate_activations: core mechanics ───────────────────────────────────

def test_calibrate_activations_returns_same_shape_as_calibrate_sensitivity():
    module = _TwoLayer()
    x = torch.randn(8, 64)
    result = gqc.calibrate_activations(
        module, lambda: module(x), precisions=("fp8", 4), model_id="t1", verbose=False)
    assert set(result.keys()) == {"a", "b"}
    for layer_errs in result.values():
        assert set(layer_errs.keys()) == {"fp8", 4}
        assert all(isinstance(v, float) for v in layer_errs.values())


def test_calibrate_activations_bits_16_is_always_zero():
    module = _TwoLayer()
    x = torch.randn(8, 64)
    result = gqc.calibrate_activations(
        module, lambda: module(x), precisions=(16, "fp8"), model_id="t2", verbose=False)
    assert result["a"][16] == 0.0


def test_calibrate_activations_layer_never_executed_is_omitted():
    """A layer whose forward hook never fires (dead branch under this
    particular calibration input) is silently omitted, not a KeyError or a
    fabricated zero."""
    module = _TwoLayer()

    class _OnlyA(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, x):
            return self.inner.a(x)   # never calls .b

    wrapper = _OnlyA(module)
    x = torch.randn(4, 64)
    result = gqc.calibrate_activations(
        wrapper, lambda: wrapper(x), precisions=("fp8",), model_id="t3", verbose=False)
    assert "inner.a" in result
    assert "inner.b" not in result


def test_calibrate_activations_more_bits_lower_error():
    """int8 must be at least as accurate as int4 on the SAME real
    activation , the whole point of measuring real output error instead of
    a weight-only proxy."""
    module = _TwoLayer(seed=1)
    x = torch.randn(16, 64)
    result = gqc.calibrate_activations(
        module, lambda: module(x), precisions=(8, 4), model_id="t4", verbose=False)
    assert result["a"][8] <= result["a"][4] + 1e-9


def test_calibrate_activations_caches_to_disk_and_reloads(tmp_path, monkeypatch):
    module = _TwoLayer()
    x = torch.randn(4, 64)
    calls = {"n": 0}

    def _run():
        calls["n"] += 1
        module(x)

    r1 = gqc.calibrate_activations(module, _run, precisions=("fp8",),
                                   model_id="cache_test", verbose=False)
    assert calls["n"] == 1

    # Second call with an EXPLODING run_calibration must never fire it , a
    # cache hit skips calling run_calibration entirely.
    def _boom():
        raise AssertionError("run_calibration should not be called on a cache hit")

    r2 = gqc.calibrate_activations(module, _boom, precisions=("fp8",),
                                   model_id="cache_test", verbose=False)
    assert r1 == r2


def test_calibrate_activations_force_recompute_bypasses_cache():
    module = _TwoLayer()
    x = torch.randn(4, 64)
    calls = {"n": 0}

    def _run():
        calls["n"] += 1
        module(x)

    gqc.calibrate_activations(module, _run, precisions=("fp8",),
                              model_id="force_test", verbose=False)
    gqc.calibrate_activations(module, _run, precisions=("fp8",),
                              model_id="force_test", verbose=False, force_recompute=True)
    assert calls["n"] == 2


def test_calibrate_activations_cache_namespace_separate_from_frobenius(monkeypatch, tmp_path):
    """The activation cache must never collide with calibrate_sensitivity's
    cache for the same model_id , different os.listdir entries."""
    module = _TwoLayer()
    x = torch.randn(4, 64)
    gqc.calibrate_sensitivity(module, precisions=("fp8",), model_id="ns_test", verbose=False)
    gqc.calibrate_activations(module, lambda: module(x), precisions=("fp8",),
                              model_id="ns_test", verbose=False)
    files = list(Path(gqc._CACHE_DIR).glob("sensitivity_ns_test*"))
    assert len(files) == 2
    assert any(".activation" in f.name for f in files)
    assert any(".activation" not in f.name for f in files)


def test_calibrate_activations_unsupported_bits_is_nan():
    module = _TwoLayer()
    x = torch.randn(4, 64)
    result = gqc.calibrate_activations(
        module, lambda: module(x), precisions=(3,), model_id="nan_test", verbose=False)
    import math
    assert math.isnan(result["a"][3])


def test_calibrate_activations_hooks_removed_after_call():
    """Forward hooks must be removed even on success , a second unrelated
    forward call must not re-trigger capture logic."""
    module = _TwoLayer()
    x = torch.randn(4, 64)
    gqc.calibrate_activations(module, lambda: module(x), precisions=("fp8",),
                              model_id="hook_test", verbose=False)
    for layer in (module.a, module.b):
        assert layer._forward_hooks == {}


def test_calibrate_activations_hooks_removed_on_exception():
    module = _TwoLayer()

    def _boom():
        raise RuntimeError("calibration input construction failed")

    with pytest.raises(RuntimeError):
        gqc.calibrate_activations(module, _boom, precisions=("fp8",),
                                  model_id="hook_exc_test", verbose=False)
    for layer in (module.a, module.b):
        assert layer._forward_hooks == {}


# ── calibrate_diffusion: WAN dual-transformer + prompt loop ────────────────

class _FakePipe:
    """Minimal stand-in for a diffusers pipeline: __call__ drives whichever
    transformer(s) are present, mirroring what a real denoising loop would
    do (call the transformer once per prompt , real pipelines call it once
    per inference step per prompt, but the calibration mechanics under test
    don't depend on how many times, just that hooks fire with real input)."""
    def __init__(self, has_transformer_2=False):
        self.transformer = _TwoLayer(seed=0)
        self.transformer_2 = _TwoLayer(seed=1) if has_transformer_2 else None
        self.calls = []

    def __call__(self, prompt, num_inference_steps, generator, **kwargs):
        self.calls.append((prompt, num_inference_steps, kwargs))
        x = torch.randn(2, 64, generator=generator)
        self.transformer(x)
        if self.transformer_2 is not None:
            self.transformer_2(x)


def test_calibrate_diffusion_single_transformer():
    pipe = _FakePipe(has_transformer_2=False)
    result = gqc.calibrate_diffusion(
        pipe, prompts=["a cat"], num_inference_steps=1, precisions=("fp8",),
        model_id="diff1", verbose=False)
    assert set(result.keys()) == {"transformer"}
    assert "a" in result["transformer"]


def test_calibrate_diffusion_wan_dual_transformer_both_present():
    pipe = _FakePipe(has_transformer_2=True)
    result = gqc.calibrate_diffusion(
        pipe, prompts=["a cat"], num_inference_steps=1, precisions=("fp8",),
        model_id="diff2", verbose=False)
    assert set(result.keys()) == {"transformer", "transformer_2"}


def test_calibrate_diffusion_skips_absent_transformer_2():
    pipe = _FakePipe(has_transformer_2=False)
    assert pipe.transformer_2 is None
    result = gqc.calibrate_diffusion(
        pipe, prompts=["a cat"], num_inference_steps=1, precisions=("fp8",),
        model_id="diff3", verbose=False)
    assert "transformer_2" not in result


def test_calibrate_diffusion_drives_all_prompts():
    pipe = _FakePipe(has_transformer_2=False)
    prompts = ["a cat", "a dog", "a tree"]
    gqc.calibrate_diffusion(
        pipe, prompts=prompts, num_inference_steps=2, precisions=("fp8",),
        model_id="diff4", verbose=False)
    assert [c[0] for c in pipe.calls] == prompts
    assert all(c[1] == 2 for c in pipe.calls)


def test_calibrate_diffusion_default_prompts_used_when_none_passed():
    pipe = _FakePipe(has_transformer_2=False)
    gqc.calibrate_diffusion(pipe, num_inference_steps=1, precisions=("fp8",),
                            model_id="diff5", verbose=False)
    assert len(pipe.calls) == len(gqc._DEFAULT_DIFFUSION_PROMPTS)


def test_calibrate_diffusion_pipe_call_kwargs_forwarded():
    pipe = _FakePipe(has_transformer_2=False)
    gqc.calibrate_diffusion(
        pipe, prompts=["a cat"], num_inference_steps=1, precisions=("fp8",),
        pipe_call_kwargs={"guidance_scale": 3.5}, model_id="diff6", verbose=False)
    assert pipe.calls[0][2].get("guidance_scale") == 3.5


def test_calibrate_diffusion_component_names_override():
    pipe = _FakePipe(has_transformer_2=True)
    result = gqc.calibrate_diffusion(
        pipe, component_names=("transformer",), prompts=["a cat"],
        num_inference_steps=1, precisions=("fp8",), model_id="diff7", verbose=False)
    assert set(result.keys()) == {"transformer"}


def test_calibrate_diffusion_result_slots_into_plan_quality():
    """The headline integration check: calibrate_diffusion's output shape
    must work as plan_quality's `sensitivity=` with ZERO adapter code."""
    import gb_quant
    pipe = _FakePipe(has_transformer_2=False)
    sensitivity = gqc.calibrate_diffusion(
        pipe, prompts=["a cat"], num_inference_steps=1,
        precisions=("fp8", 4), model_id="diff8", verbose=False)["transformer"]

    profile = gb_quant.GpuProfile(
        family="blackwell", cc=(12, 0), precisions=(16, "fp8", 8, 4),
        floor_default=4, quality_default="fp8", t2_tolerance_gb=10.0)
    report = gb_quant.plan_quality(
        pipe.transformer, target="near_lossless", sensitivity=sensitivity,
        profile=profile, t1_budget_gb=100.0, t2_budget_gb=100.0, verbose=False)
    assert set(report.per_layer_bits.keys()) <= {"a", "b"}

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for synapse_engine/gllm/layers/quantization/gptq.py — the GPTQ loader
for GreenBoost's vendored torch-core engine (gLLM). P5.1 shipped with zero
unit tests (unlike AWQ's 5, tests/test_synapse_engine_awq.py) — the v1/v2
qzeros zero-point offset bug this file's own docstring calls out
(_maybe_convert_v1_zeros) was found LIVE with no regression test guarding it.
This file exists to close that gap (P0, plan bring-gb-synapse-gb-quant-and-
async-nygaard.md).

Loaded by file path via importlib, same pattern test_synapse_engine_awq.py
uses (bypasses gllm/__init__.py, which eagerly imports the full engine —
only present in the dedicated synapse-torch-env venv). gptq.py's only
module-level import is torch; the gptqmodel dependency is lazy (imported
inside gptq_linear_method), so it's faked via sys.modules for the one test
that exercises that function, same as any other lazy-import test in this
repo (see tests/conftest.py's own use of the pattern elsewhere).
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_GPTQ_PATH = (Path(__file__).parent.parent
              / "synapse_engine" / "gllm" / "layers" / "quantization" / "gptq.py")


def _load_gptq():
    spec = importlib.util.spec_from_file_location("_gptq_standalone", _GPTQ_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gptq = _load_gptq()


class _FakeQuantConfig(dict):
    """gptq_create_weights reads layer.quant_config like a plain dict
    (.get()/[]) — a real gLLM layer's quant_config IS just the checkpoint's
    parsed quantize_config.json, no custom class."""


class _FakeLayer:
    def __init__(self, **quant_config):
        self.quant_config = _FakeQuantConfig(quant_config)
        self._params = {}

    def register_parameter(self, name, value):
        self._params[name] = value
        setattr(self, name, value)


# ── pack_factor ────────────────────────────────────────────────────────────

def test_pack_factor_2bit():
    assert gptq.pack_factor(2) == 16


def test_pack_factor_4bit():
    assert gptq.pack_factor(4) == 8


def test_pack_factor_8bit():
    assert gptq.pack_factor(8) == 4


# ── gptq_create_weights: guard paths ─────────────────────────────────────

def test_create_weights_rejects_unsupported_bits():
    layer = _FakeLayer(bits=3, group_size=128)
    with pytest.raises(NotImplementedError, match="3-bit"):
        gptq.gptq_create_weights(layer, input_size_per_partition=128,
                                 output_partition_sizes=[128], params_dtype=torch.float16)


def test_create_weights_registers_expected_shapes():
    in_features, out_features, group_size, bits = 128, 256, 64, 4
    layer = _FakeLayer(bits=bits, group_size=group_size)
    gptq.gptq_create_weights(layer, input_size_per_partition=in_features,
                             output_partition_sizes=[out_features], params_dtype=torch.float16)

    pf = gptq.pack_factor(bits)
    num_groups = in_features // group_size
    assert layer.qweight.shape == (in_features // pf, out_features)
    assert layer.qweight.dtype == torch.int32
    assert layer.qzeros.shape == (num_groups, out_features // pf)
    assert layer.scales.shape == (num_groups, out_features)
    assert layer.scales.dtype == torch.float16
    assert layer.g_idx.tolist() == [i // group_size for i in range(in_features)]
    # "weight" must exist (forward() always calls quant_method(input, self.weight, ...))
    # but must be None — GPTQ's real dequant reads qweight/qzeros/scales/g_idx instead.
    assert layer.weight is None


def test_create_weights_defaults_group_size_to_full_input_when_unset():
    """group_size <= 0 (or absent) means per-tensor quantization — one group
    spanning the whole input dimension, not a crash or a stray small group."""
    in_features, out_features, bits = 128, 64, 4
    layer = _FakeLayer(bits=bits)  # no group_size key at all
    gptq.gptq_create_weights(layer, input_size_per_partition=in_features,
                             output_partition_sizes=[out_features], params_dtype=torch.float16)
    assert layer.gptq_group_size == in_features
    assert layer.qzeros.shape[0] == 1


def test_create_weights_output_partition_sizes_are_summed():
    """Tensor-parallel column split: output_partition_sizes is a list of
    per-partition widths: total out_features is their sum, not just the
    first entry."""
    layer = _FakeLayer(bits=4, group_size=64)
    gptq.gptq_create_weights(layer, input_size_per_partition=128,
                             output_partition_sizes=[64, 64, 64], params_dtype=torch.float16)
    assert layer.scales.shape[1] == 192


# ── _maybe_convert_v1_zeros: the real, previously-unguarded incident ──────

def test_v1_zeros_get_offset_by_default():
    """No checkpoint_format field at all (the common case for public GPTQ
    checkpoints) must be treated as v1 and corrected — this is the exact
    incident: a v1 checkpoint served without this correction sampled
    garbage/out-of-vocab-range token ids."""
    layer = _FakeLayer(bits=4, group_size=64)
    gptq.gptq_create_weights(layer, input_size_per_partition=128,
                             output_partition_sizes=[64], params_dtype=torch.float16)
    before = layer.qzeros.data.clone()

    gptq._maybe_convert_v1_zeros(layer)

    expected_offset = gptq._V1_TO_V2_ZERO_OFFSET_INT32[4]
    assert torch.equal(layer.qzeros.data, before + expected_offset)
    assert layer._gptq_zeros_v1_converted is True


def test_v1_zeros_offset_is_per_bitwidth():
    for bits in (2, 4, 8):
        layer = _FakeLayer(bits=bits, group_size=64)
        gptq.gptq_create_weights(layer, input_size_per_partition=128,
                                 output_partition_sizes=[64], params_dtype=torch.float16)
        before = layer.qzeros.data.clone()
        gptq._maybe_convert_v1_zeros(layer)
        assert torch.equal(layer.qzeros.data,
                           before + gptq._V1_TO_V2_ZERO_OFFSET_INT32[bits])


@pytest.mark.parametrize("fmt", ["gptq_v2", "v2", "GPTQ_V2"])
def test_v2_checkpoints_are_left_alone(fmt):
    layer = _FakeLayer(bits=4, group_size=64, checkpoint_format=fmt)
    gptq.gptq_create_weights(layer, input_size_per_partition=128,
                             output_partition_sizes=[64], params_dtype=torch.float16)
    before = layer.qzeros.data.clone()

    gptq._maybe_convert_v1_zeros(layer)

    assert torch.equal(layer.qzeros.data, before)  # unchanged
    assert layer._gptq_zeros_v1_converted is True  # still marked done


def test_v1_zeros_conversion_runs_exactly_once():
    """A second call must be a no-op — applying the +offset twice would
    double-correct and produce a wrong zero-point just as surely as never
    correcting at all."""
    layer = _FakeLayer(bits=4, group_size=64)
    gptq.gptq_create_weights(layer, input_size_per_partition=128,
                             output_partition_sizes=[64], params_dtype=torch.float16)

    gptq._maybe_convert_v1_zeros(layer)
    once = layer.qzeros.data.clone()
    gptq._maybe_convert_v1_zeros(layer)  # must be a no-op (guarded by the flag)

    assert torch.equal(layer.qzeros.data, once)


# ── gptq_linear_method: dispatch wiring ───────────────────────────────────

def test_linear_method_calls_quant_matmul_with_layer_state(monkeypatch):
    """gptq_linear_method's real dependency (gptqmodel's triton dequant
    kernel) is lazily imported inside the function body — not installed in
    this test env, so it's faked via sys.modules, the same way the module's
    only non-torch dependency would be faked/mocked anywhere else in this
    repo's test suite."""
    captured = {}

    def _fake_quant_matmul(input_, qweight, scales, qzeros, g_idx, *, bits, pack_bits, maxq):
        captured.update(bits=bits, pack_bits=pack_bits, maxq=maxq)
        return torch.zeros(input_.shape[0], scales.shape[1], dtype=torch.float16)

    fake_dequant = types.ModuleType("gptqmodel.nn_modules.triton_utils.dequant")
    fake_dequant.quant_matmul = _fake_quant_matmul
    fake_triton_utils = types.ModuleType("gptqmodel.nn_modules.triton_utils")
    fake_triton_utils.dequant = fake_dequant
    fake_nn_modules = types.ModuleType("gptqmodel.nn_modules")
    fake_nn_modules.triton_utils = fake_triton_utils
    fake_gptqmodel = types.ModuleType("gptqmodel")
    fake_gptqmodel.nn_modules = fake_nn_modules
    monkeypatch.setitem(sys.modules, "gptqmodel", fake_gptqmodel)
    monkeypatch.setitem(sys.modules, "gptqmodel.nn_modules", fake_nn_modules)
    monkeypatch.setitem(sys.modules, "gptqmodel.nn_modules.triton_utils", fake_triton_utils)
    monkeypatch.setitem(sys.modules, "gptqmodel.nn_modules.triton_utils.dequant", fake_dequant)

    bits = 4
    layer = _FakeLayer(bits=bits, group_size=64, checkpoint_format="gptq_v2")
    gptq.gptq_create_weights(layer, input_size_per_partition=128,
                             output_partition_sizes=[64], params_dtype=torch.float16)
    x = torch.zeros(2, 128, dtype=torch.float16)

    out = gptq.gptq_linear_method(x, layer.weight, bias=None, layer=layer)

    assert captured == {"bits": bits, "pack_bits": 32, "maxq": (1 << bits) - 1}
    assert out.shape == (2, 64)
    assert out.dtype == x.dtype
    # v2-declared checkpoint: must have gone through the guard without
    # applying the v1 offset (dispatch order: convert-zeros then matmul).
    assert layer._gptq_zeros_v1_converted is True


def test_linear_method_adds_bias_when_given(monkeypatch):
    def _fake_quant_matmul(input_, qweight, scales, qzeros, g_idx, *, bits, pack_bits, maxq):
        return torch.zeros(input_.shape[0], scales.shape[1], dtype=torch.float16)

    fake_dequant = types.ModuleType("gptqmodel.nn_modules.triton_utils.dequant")
    fake_dequant.quant_matmul = _fake_quant_matmul
    fake_triton_utils = types.ModuleType("gptqmodel.nn_modules.triton_utils")
    fake_triton_utils.dequant = fake_dequant
    fake_nn_modules = types.ModuleType("gptqmodel.nn_modules")
    fake_nn_modules.triton_utils = fake_triton_utils
    fake_gptqmodel = types.ModuleType("gptqmodel")
    fake_gptqmodel.nn_modules = fake_nn_modules
    monkeypatch.setitem(sys.modules, "gptqmodel", fake_gptqmodel)
    monkeypatch.setitem(sys.modules, "gptqmodel.nn_modules", fake_nn_modules)
    monkeypatch.setitem(sys.modules, "gptqmodel.nn_modules.triton_utils", fake_triton_utils)
    monkeypatch.setitem(sys.modules, "gptqmodel.nn_modules.triton_utils.dequant", fake_dequant)

    layer = _FakeLayer(bits=4, group_size=64, checkpoint_format="gptq_v2")
    gptq.gptq_create_weights(layer, input_size_per_partition=128,
                             output_partition_sizes=[64], params_dtype=torch.float16)
    x = torch.zeros(2, 128, dtype=torch.float16)
    bias = torch.full((64,), 3.0, dtype=torch.float16)

    out = gptq.gptq_linear_method(x, layer.weight, bias=bias, layer=layer)

    assert torch.equal(out, bias.expand(2, 64))

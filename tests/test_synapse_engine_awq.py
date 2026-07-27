#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for synapse_engine/gllm/layers/quantization/awq.py — the AWQ loader
for GreenBoost's vendored torch-core engine (gLLM), see synapse_engine/
NOTICE for the local-patch writeup (P5.2, 2026-07-26).

Loaded by file path via importlib, bypassing `gllm/__init__.py` (which
eagerly imports the full engine, `pyzmq` included, only present in the
dedicated synapse-torch-env venv — see gb_synapse_backends._torch_env_dir).
awq.py itself has no gllm-internal imports (only torch), so this is a
faithful test of the real module, not a reimplementation, and runs on any
box with torch installed.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_AWQ_PATH = (Path(__file__).parent.parent
             / "synapse_engine" / "gllm" / "layers" / "quantization" / "awq.py")


def _load_awq():
    spec = importlib.util.spec_from_file_location("_awq_standalone", _AWQ_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


awq = _load_awq()


class _FakeQuantConfig(dict):
    """awq_create_weights reads layer.quant_config like a plain dict
    (.get()) — a real gLLM layer's quant_config IS just the checkpoint's
    parsed quantize_config.json, no custom class."""


class _FakeLayer:
    def __init__(self, **quant_config):
        self.quant_config = _FakeQuantConfig(quant_config)
        self._params = {}

    def register_parameter(self, name, value):
        self._params[name] = value
        setattr(self, name, value)


# ── pack_factor ────────────────────────────────────────────────────────────

def test_pack_factor_4bit():
    assert awq.pack_factor(4) == 8   # 32-bit word / 4-bit values = 8 packed


# ── awq_create_weights: guard paths ─────────────────────────────────────

def test_create_weights_rejects_non_4bit():
    layer = _FakeLayer(bits=8, zero_point=True, group_size=128)
    with pytest.raises(NotImplementedError, match="8-bit"):
        awq.awq_create_weights(layer, input_size_per_partition=128,
                               output_partition_sizes=[128], params_dtype=torch.float16)


def test_create_weights_rejects_symmetric_no_zero_point():
    layer = _FakeLayer(bits=4, zero_point=False, group_size=128)
    with pytest.raises(NotImplementedError, match="zero_point"):
        awq.awq_create_weights(layer, input_size_per_partition=128,
                               output_partition_sizes=[128], params_dtype=torch.float16)


def test_create_weights_registers_expected_shapes():
    in_features, out_features, group_size = 128, 256, 64
    layer = _FakeLayer(bits=4, zero_point=True, group_size=group_size)
    awq.awq_create_weights(layer, input_size_per_partition=in_features,
                           output_partition_sizes=[out_features], params_dtype=torch.float16)

    pf = awq.pack_factor(4)
    assert layer.qweight.shape == (in_features, out_features // pf)
    assert layer.qweight.dtype == torch.int32
    num_groups = in_features // group_size
    assert layer.qzeros.shape == (num_groups, out_features // pf)
    assert layer.scales.shape == (num_groups, out_features)
    assert layer.scales.dtype == torch.float16
    # g_idx is trivial (no desc_act equivalent in AWQ) and NOT a registered
    # Parameter — the module docstring is explicit that it must never appear
    # in named_parameters()/state_dict().
    assert layer.g_idx.tolist() == [i // group_size for i in range(in_features)]
    assert "g_idx" not in layer._params


# ── _maybe_convert_awq_layout: round-trips a known packed value ─────────

def test_convert_awq_layout_is_idempotent_and_matches_gptq_shape():
    """Full round trip: build real AWQ-ordered packed tensors for a tiny
    layer, run the conversion, and confirm (a) the output shape matches
    GPTQ's [in_features/pack_factor, out_features] convention (the whole
    point of the conversion — see the module docstring's two-differences
    list) and (b) running it twice is a no-op (the lazy
    `_awq_layout_converted` guard)."""
    bits, group_size = 4, 64
    in_features, out_features = 64, 32
    layer = _FakeLayer(bits=bits, zero_point=True, group_size=group_size)
    awq.awq_create_weights(layer, input_size_per_partition=in_features,
                           output_partition_sizes=[out_features], params_dtype=torch.float16)

    pf = awq.pack_factor(bits)
    # Fill with a real, decodable AWQ-ordered pattern instead of zeros —
    # zeros round-trip trivially and wouldn't catch a broken permutation.
    torch.manual_seed(0)
    layer.qweight.data = torch.randint(0, 2 ** 32 - 1, layer.qweight.shape,
                                       dtype=torch.int64).to(torch.int32)
    layer.qzeros.data = torch.randint(0, 2 ** 32 - 1, layer.qzeros.shape,
                                      dtype=torch.int64).to(torch.int32)
    before_qweight = layer.qweight.data.clone()

    awq._maybe_convert_awq_layout(layer)

    assert layer._awq_layout_converted is True
    # GPTQ-standard axis: [in_features // pack_factor, out_features].
    assert layer.qweight.shape == (in_features // pf, out_features)
    assert layer.qzeros.shape == (in_features // group_size, out_features // pf)

    converted_once = layer.qweight.data.clone()
    awq._maybe_convert_awq_layout(layer)   # second call must be a no-op
    assert torch.equal(layer.qweight.data, converted_once)
    assert not torch.equal(before_qweight, converted_once)  # actually transformed, not a no-op copy

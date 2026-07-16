#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for gb_synapse.read_quant_config() — checkpoint-truth quantization
detection from a local/remote config.json's quantization_config.

CPU-only. No CUDA, no real HF network calls (remote path is a separate live
check, not exercised here)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gb_synapse as gs


def _write_cfg(tmp_path, cfg):
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return str(tmp_path)


def test_gptq4(tmp_path):
    d = _write_cfg(tmp_path, {"quantization_config": {"quant_method": "gptq", "bits": 4}})
    assert gs.read_quant_config(d) == {"quant_method": "gptq", "quant_bits": 4}


def test_awq4(tmp_path):
    d = _write_cfg(tmp_path, {"quantization_config": {"quant_method": "awq", "bits": 4}})
    assert gs.read_quant_config(d) == {"quant_method": "awq", "quant_bits": 4}


def test_fp8_activation_scheme(tmp_path):
    d = _write_cfg(tmp_path, {"quantization_config": {
        "quant_method": "fp8", "activation_scheme": "dynamic"}})
    assert gs.read_quant_config(d) == {"quant_method": "fp8", "quant_bits": 8}


def test_fp8_weight_block_size(tmp_path):
    d = _write_cfg(tmp_path, {"quantization_config": {
        "quant_method": "fp8", "weight_block_size": [128, 128]}})
    assert gs.read_quant_config(d) == {"quant_method": "fp8", "quant_bits": 8}


def test_compressed_tensors_w4a16(tmp_path):
    d = _write_cfg(tmp_path, {"quantization_config": {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "int"}}}}})
    assert gs.read_quant_config(d) == {"quant_method": "compressed-tensors", "quant_bits": 4}


def test_compressed_tensors_w8a8(tmp_path):
    d = _write_cfg(tmp_path, {"quantization_config": {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "int"}}}}})
    assert gs.read_quant_config(d) == {"quant_method": "compressed-tensors", "quant_bits": 8}


def test_compressed_tensors_float_quantized_normalizes_to_fp8(tmp_path):
    d = _write_cfg(tmp_path, {"quantization_config": {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"type": "float"}}}}})
    assert gs.read_quant_config(d) == {"quant_method": "fp8", "quant_bits": 8}


def test_compressed_tensors_no_quant_method_key(tmp_path):
    """Some compressed-tensors exports omit quant_method entirely — infer
    from the presence of config_groups."""
    d = _write_cfg(tmp_path, {"quantization_config": {
        "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "int"}}}}})
    assert gs.read_quant_config(d) == {"quant_method": "compressed-tensors", "quant_bits": 4}


def test_bnb4(tmp_path):
    d = _write_cfg(tmp_path, {"quantization_config": {"load_in_4bit": True}})
    assert gs.read_quant_config(d) == {"quant_method": "bitsandbytes", "quant_bits": 4}


def test_bnb8(tmp_path):
    d = _write_cfg(tmp_path, {"quantization_config": {"load_in_8bit": True}})
    assert gs.read_quant_config(d) == {"quant_method": "bitsandbytes", "quant_bits": 8}


def test_nested_text_config(tmp_path):
    d = _write_cfg(tmp_path, {"text_config": {
        "quantization_config": {"quant_method": "gptq", "bits": 4}}})
    assert gs.read_quant_config(d) == {"quant_method": "gptq", "quant_bits": 4}


def test_absent_quantization_config(tmp_path):
    d = _write_cfg(tmp_path, {"model_type": "qwen3"})
    assert gs.read_quant_config(d) == {}


def test_malformed_json(tmp_path):
    (tmp_path / "config.json").write_text("{not json")
    assert gs.read_quant_config(str(tmp_path)) == {}


def test_missing_config_json(tmp_path):
    assert gs.read_quant_config(str(tmp_path)) == {}

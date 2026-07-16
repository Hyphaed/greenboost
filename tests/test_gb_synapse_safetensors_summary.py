#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for gb_synapse.safetensors_summary(): .bin byte counting + merged
quant_method/quant_bits keys (P1.3 of the gb-synapse unification).

CPU-only."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gb_synapse as gs


def test_bin_only_checkpoint_reports_nonzero_bytes(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3",
                                                       "num_hidden_layers": 4}))
    (tmp_path / "pytorch_model.bin").write_bytes(b"x" * 1000)
    (tmp_path / "training_args.bin").write_bytes(b"y" * 5000)
    summary = gs.safetensors_summary(str(tmp_path))
    assert summary["n_bytes"] == 1000
    assert summary["n_layers"] == 4


def test_safetensors_and_bin_bytes_both_counted(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (tmp_path / "model-00001.safetensors").write_bytes(b"a" * 300)
    (tmp_path / "model-00002.bin").write_bytes(b"b" * 200)
    summary = gs.safetensors_summary(str(tmp_path))
    assert summary["n_bytes"] == 500


def test_quant_config_merged_into_summary(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "quantization_config": {"quant_method": "gptq", "bits": 4}}))
    (tmp_path / "model.safetensors").write_bytes(b"z" * 10)
    summary = gs.safetensors_summary(str(tmp_path))
    assert summary["quant_method"] == "gptq"
    assert summary["quant_bits"] == 4


def test_no_quant_config_defaults_empty(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (tmp_path / "model.safetensors").write_bytes(b"z" * 10)
    summary = gs.safetensors_summary(str(tmp_path))
    assert summary["quant_method"] == ""
    assert summary["quant_bits"] == 0

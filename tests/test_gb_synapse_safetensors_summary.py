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


def _write_safetensors(path, tensors: dict):
    """Build a minimal real (not fake) .safetensors file: header + raw
    payload, byte-exact data_offsets, so _sum_tensor_bytes_by_name reads the
    same on-disk layout a real checkpoint has. `tensors` maps name -> byte
    length (content is arbitrary padding, only the span matters here)."""
    header = {}
    payload = bytearray()
    for name, nbytes in tensors.items():
        start = len(payload)
        payload += b"\x00" * nbytes
        header[name] = {"dtype": "F32", "shape": [nbytes // 4], "data_offsets": [start, len(payload)]}
    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(len(header_bytes).to_bytes(8, "little"))
        f.write(header_bytes)
        f.write(bytes(payload))


def test_qwen4exp_tensor_classes_summed_from_real_headers(tmp_path):
    """Regression: gb_synapse.py's Flash-Next (qwen4_exp) metadata plumbing
    added n_ngram_bytes/n_qsa_indexer_bytes/n_hyper_connection_bytes, summed
    byte-exact from safetensors headers' own data_offsets rather than
    reverse-engineered from config field arithmetic. Verified live 2026-08-27
    against the real Qwen/Qwen3.8-Flash-Next-FP8 checkpoint (47.75 GiB
    n_ngram_bytes on a partial pull) — this test pins the same tensor-name
    matching logic against a small synthetic checkpoint so it runs in CI."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    _write_safetensors(tmp_path / "model-00001-of-00002.safetensors", {
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": 4000,
        "model.language_model.layers.1.ple.norm_key.weight": 40,
        "model.language_model.layers.11.self_attn.indexer.index_qk_proj.weight": 800,
        "model.language_model.hyper_connection_mixer.hc_norm.weight": 80,
        "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight": 80,
        # Unrelated tensor — must NOT be swept into any of the three buckets.
        "model.language_model.layers.0.mlp.gate.weight": 1600,
    })
    _write_safetensors(tmp_path / "model-00002-of-00002.safetensors", {
        "model.language_model.layers.5.ple.norm_query.weight": 40,
    })
    summary = gs.safetensors_summary(str(tmp_path))
    assert summary["n_ngram_bytes"] == 4000 + 40 + 40
    assert summary["n_qsa_indexer_bytes"] == 800
    assert summary["n_hyper_connection_bytes"] == 80 + 80
    # n_bytes is real on-disk file size (header + payload), not a sum of
    # payload spans — matches what a real HF download's size represents.
    expected_n_bytes = sum((tmp_path / f).stat().st_size
                           for f in ("model-00001-of-00002.safetensors",
                                     "model-00002-of-00002.safetensors"))
    assert summary["n_bytes"] == expected_n_bytes


def test_safetensors_header_rejects_non_safetensors_file(tmp_path):
    """Regression, 2026-08-27: a garbage/fake .safetensors file (test
    fixtures elsewhere in this suite write plain padding bytes, and a
    still-downloading shard can be truncated) used to read its first 8
    bytes as a length prefix unconditionally — on non-safetensors content
    that's an arbitrary huge number, and json.loads(f.read(n)) raised
    MemoryError trying to honor it instead of returning {} like every other
    best-effort path in this function."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (tmp_path / "model.safetensors").write_bytes(b"not a real safetensors file" * 10)
    summary = gs.safetensors_summary(str(tmp_path))  # must not raise
    assert summary["n_ngram_bytes"] == 0
    assert summary["n_qsa_indexer_bytes"] == 0
    assert summary["n_hyper_connection_bytes"] == 0

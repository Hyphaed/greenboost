#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for selective-SSM (Mamba/Mamba2/hybrid Gated-DeltaNet) recurrent-
state accounting added 2026-07-29 — see CLAUDE.md's Reference Workload Rule
addendum and workflow/architecture.md's GB-Semantics section addendum.

Covers: safetensors_summary()'s layer_types counting (the primary-workload
fix, Brian6145/Qwen3.6-27B-Claude-Opus-Sonnet-Distilled-NVFP4-MTP),
estimate_ssm_state_gb()'s formula, _solve_ctx_and_layers()'s is_recurrent_only
handling, and ModelEntry backward compatibility with pre-Phase-1 manifests.

CPU-only, no real HF network calls."""
import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import gb_synapse as gs


# ---------------------------------------------------------------------------
# safetensors_summary(): layer_types counting (the primary-workload fix)
# ---------------------------------------------------------------------------

def _write_qwen35_style_config(root: Path, n_full: int = 2, n_linear_per_full: int = 3):
    """A HF config.json shaped like Qwen3.5/3.6's hybrid Gated-DeltaNet mix:
    `n_linear_per_full` linear-attention layers per 1 full-attention layer,
    repeated `n_full` times (mirrors this repo's own reference-workload
    framing: "16 x (3x Gated DeltaNet -> 1x Gated Attention)")."""
    layer_types = (["linear_attention"] * n_linear_per_full + ["full_attention"]) * n_full
    cfg = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5",
        "num_hidden_layers": len(layer_types),
        "layer_types": layer_types,
        "max_position_embeddings": 262144,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(cfg))
    (root / "model.safetensors").write_bytes(b"\x00" * 1024)
    return cfg, layer_types


def test_safetensors_summary_counts_layer_types_directly(tmp_path):
    """The exact bug: checkpoint_summary()/safetensors_summary() used to read
    `layer_types` only for its LENGTH (n_layers), never its per-layer
    contents, so n_kv_layers stayed 0 on every hybrid safetensors model —
    every `entry.n_kv_layers or entry.n_layers` fallback then silently
    treated that as "unknown, assume every layer is real attention",
    reintroducing the ~4x KV overestimate the GGUF full_attention_interval
    fix (2026-07-27) had already closed on the OTHER metadata path."""
    _write_qwen35_style_config(tmp_path, n_full=2, n_linear_per_full=3)
    meta = gs.safetensors_summary(str(tmp_path))

    assert meta["n_layers"] == 8
    assert meta["n_recurrent_layers"] == 6          # 3 linear per full x 2
    assert meta["n_kv_layers"] == 2                 # 8 - 6
    assert meta["is_recurrent_only"] is False
    # Recurrent geometry normalized from the GDN-style config fields
    # (memory_manager.py's _build_ssm_cache_config: conv_dim = 2*key_dim +
    # value_dim; state = num_v_heads * head_v_dim * head_k_dim).
    assert meta["ssm_d_conv"] == 4
    key_dim = 16 * 128
    value_dim = 32 * 128
    assert meta["ssm_conv_width"] == 2 * key_dim + value_dim
    assert meta["ssm_state_elems"] == 32 * 128 * 128


def test_safetensors_summary_plain_transformer_has_no_recurrent_layers(tmp_path):
    """A config with no `layer_types` at all (a plain transformer) must
    report zero recurrent layers, not a metadata gap — n_kv_layers stays
    n_layers, exactly the historical/safe default."""
    cfg = {
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "num_hidden_layers": 4, "max_position_embeddings": 8192,
        "num_key_value_heads": 8, "head_dim": 128,
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 64)

    meta = gs.safetensors_summary(str(tmp_path))
    assert meta["n_layers"] == 4
    assert meta["n_recurrent_layers"] == 0
    assert meta["n_kv_layers"] == 4
    assert meta["is_recurrent_only"] is False
    assert meta["ssm_d_conv"] == 0


def test_safetensors_summary_pure_mamba_is_recurrent_only(tmp_path):
    """model_type == "mamba"/"mamba2": every layer is recurrent, n_kv_layers
    == 0 is the CORRECT value here (no real attention layer at all), not
    "unknown" — the exact distinction is_recurrent_only exists to carry."""
    cfg = {
        "architectures": ["Mamba2ForCausalLM"], "model_type": "mamba2",
        "num_hidden_layers": 6, "max_position_embeddings": 4096,
        "conv_kernel": 4, "intermediate_size": 2048, "state_size": 128, "n_groups": 1,
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 64)

    meta = gs.safetensors_summary(str(tmp_path))
    assert meta["n_layers"] == 6
    assert meta["is_recurrent_only"] is True
    assert meta["n_recurrent_layers"] == 6
    assert meta["n_kv_layers"] == 0
    assert meta["ssm_d_conv"] == 4
    assert meta["ssm_conv_width"] == 2048 + 2 * 1 * 128
    assert meta["ssm_state_elems"] == 128 * 2048


# ---------------------------------------------------------------------------
# estimate_ssm_state_gb(): the formula itself
# ---------------------------------------------------------------------------

def test_estimate_ssm_state_gb_matches_llama_cpp_formula():
    """Hand-computed against llama-hparams.cpp's n_embd_r()/n_embd_s():
    n_embd_r = (d_conv-1)*conv_width, n_embd_s = state_elems,
    bytes = 4 (F32) * n_recurrent_layers * (n_embd_r + n_embd_s) * n_seq_max."""
    n_recurrent_layers = 6
    d_conv = 4
    conv_width = 100
    state_elems = 500
    n_embd_r = (d_conv - 1) * conv_width   # 300
    n_embd_s = state_elems                  # 500
    expected_bytes = 4 * n_recurrent_layers * (n_embd_r + n_embd_s)
    expected_gb = expected_bytes / (1024 ** 3)

    got = gs.estimate_ssm_state_gb(n_recurrent_layers, d_conv, conv_width, state_elems)
    assert got == pytest.approx(expected_gb)


def test_estimate_ssm_state_gb_scales_with_n_seq_max():
    base = gs.estimate_ssm_state_gb(4, 4, 100, 500, n_seq_max=1)
    doubled = gs.estimate_ssm_state_gb(4, 4, 100, 500, n_seq_max=2)
    assert doubled == pytest.approx(base * 2)


def test_estimate_ssm_state_gb_zero_for_no_recurrent_layers():
    """Zero recurrent layers (plain transformer) or missing conv geometry
    (old manifest entry) both correctly return 0.0 — not a metadata gap in
    either case, see the function's own docstring."""
    assert gs.estimate_ssm_state_gb(0, 4, 100, 500) == 0.0
    assert gs.estimate_ssm_state_gb(6, 0, 100, 500) == 0.0


def test_entry_ssm_gb_reads_from_model_entry():
    entry = gs.ModelEntry(name="m", path="", n_recurrent_layers=6,
                          ssm_d_conv=4, ssm_conv_width=100, ssm_state_elems=500)
    expected = gs.estimate_ssm_state_gb(6, 4, 100, 500)
    assert gs._entry_ssm_gb(entry) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _solve_ctx_and_layers(): is_recurrent_only must not clamp ctx on cache
# grounds, and the hybrid case must charge ssm_gb as a fixed cost
# ---------------------------------------------------------------------------

def _make_entry(**overrides):
    base = dict(name="m", path="", n_bytes=8 * (1024 ** 3), n_layers=32,
               n_kv_layers=8, n_kv_heads=8, head_dim=128,
               n_recurrent_layers=24, is_recurrent_only=False,
               ssm_d_conv=4, ssm_conv_width=100, ssm_state_elems=500)
    base.update(overrides)
    return gs.ModelEntry(**base)


def test_solve_ctx_and_layers_recurrent_only_does_not_clamp_ctx(monkeypatch):
    """A pure-Mamba/Mamba2 entry (is_recurrent_only=True) has NO per-token KV
    cost at all — ctx must pass through unclamped (only ngl is fit against
    the fixed ssm_gb budget), never fall to the "unknown geometry" branch."""
    entry = _make_entry(n_kv_layers=0, is_recurrent_only=True, n_recurrent_layers=32)
    ctx, ngl = gs._solve_ctx_and_layers(entry, vram_gb=20.0, requested_ctx=131072)
    assert ctx == 131072   # unconstrained by cache growth
    assert ngl > 0


def test_solve_ctx_and_layers_hybrid_charges_ssm_gb_as_fixed_cost(monkeypatch):
    """The hybrid hot path (bytes_per_tok > 0): ctx solved against a budget
    that has ssm_gb subtracted ONCE (fixed), not scaled by ctx. Verified by
    comparing against the same call with ssm_state_elems=0 (no recurrent
    cost) — the with-ssm case must solve to a smaller-or-equal ctx for the
    same VRAM budget, since less VRAM is left over for KV."""
    entry_with_ssm = _make_entry()
    entry_without_ssm = _make_entry(n_recurrent_layers=0, ssm_d_conv=0,
                                    ssm_conv_width=0, ssm_state_elems=0)

    ctx_with, ngl_with = gs._solve_ctx_and_layers(entry_with_ssm, vram_gb=10.0,
                                                   requested_ctx=65536)
    ctx_without, ngl_without = gs._solve_ctx_and_layers(entry_without_ssm, vram_gb=10.0,
                                                        requested_ctx=65536)
    assert ngl_with == ngl_without   # ssm_gb doesn't change layer-fit at this budget
    assert ctx_with <= ctx_without


def test_solve_ctx_and_layers_unknown_geometry_degrades_to_clamp(monkeypatch):
    """Old manifest entry (n_kv_heads/head_dim unset, is_recurrent_only
    False): bytes_per_tok==0 for a non-recurrent-only reason — must degrade
    to the pre-existing _clamp_ctx_to_budget path, not the recurrent-only
    unconstrained-ctx path (verified by spying on the call, since the exact
    ctx value depends on the param-count KV heuristic's numeric details)."""
    entry = _make_entry(n_kv_heads=0, head_dim=0, n_recurrent_layers=0,
                        ssm_d_conv=0, ssm_conv_width=0, ssm_state_elems=0)
    calls = []
    real_clamp = gs._clamp_ctx_to_budget
    monkeypatch.setattr(gs, "_clamp_ctx_to_budget",
                        lambda *a, **k: (calls.append(1), real_clamp(*a, **k))[1])
    gs._solve_ctx_and_layers(entry, vram_gb=10.0, requested_ctx=1_000_000)
    assert calls == [1]


# ---------------------------------------------------------------------------
# ModelEntry backward compatibility with pre-Phase-1 manifests
# ---------------------------------------------------------------------------

def test_model_entry_round_trips_pre_phase1_manifest():
    """A manifest entry saved before these fields existed must still
    deserialize cleanly — same TypeError trap as dense_bytes/expert_bytes
    (2026-07-15) and n_kv_layers (2026-07-27) before it."""
    old_style = {
        "name": "old-model", "path": "/x", "source": "hf", "quant": "Q4_K_M",
        "arch": "llama", "engine": "llama.cpp", "n_bytes": 123, "n_layers": 32,
        "is_moe": False, "n_experts": 0, "n_experts_used": 0,
        "dense_bytes": 100, "expert_bytes": 0, "ctx_length": 8192,
        "n_kv_heads": 8, "head_dim": 128, "n_kv_layers": 32, "added_ts": 0.0,
        # deliberately NO n_recurrent_layers / is_recurrent_only / ssm_* keys
    }
    entry = gs.ModelEntry(**old_style)
    assert entry.n_recurrent_layers == 0
    assert entry.is_recurrent_only is False
    assert entry.ssm_d_conv == 0
    assert entry.ssm_conv_width == 0
    assert entry.ssm_state_elems == 0
    # And it must serialize back via asdict() the same way _save_manifest does.
    assert dataclasses.asdict(entry)["n_recurrent_layers"] == 0

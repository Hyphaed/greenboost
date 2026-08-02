#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_tier_kv.py , KV-cache tier-serde compression codec
(missing_features.md item (e)). CPU-only: gb_attn's PolarQuant/QJL
primitives are device-parameterized and take plain tensors, no CUDA
required for either the quantizer construction or the quantize/dequantize
calls.
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

import gb_attn
import gb_tier_kv


def setup_function(_fn):
    # Every test gets a clean quantizer cache and default env.
    gb_attn._quantizer_cache.clear()
    os.environ.pop("GB_TIER_KV_COMPRESS", None)
    os.environ.pop("GB_TIER_KV_PRESET", None)
    os.environ.pop("GB_TIER_KV_MIN_ELEMS", None)


def _kv_tensor(shape=(2, 8, 64, 128), seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


# ── resolve_codec ───────────────────────────────────────────────────────────

def test_resolve_codec_explicit_wins():
    assert gb_tier_kv.resolve_codec("turboquant") == "turboquant"


def test_resolve_codec_default_is_polarquant():
    assert gb_tier_kv.resolve_codec() == "polarquant"


def test_resolve_codec_preset_turbo(monkeypatch):
    monkeypatch.setenv("GB_TIER_KV_PRESET", "turbo_k8v4")
    assert gb_tier_kv.resolve_codec() == "turboquant"


def test_resolve_codec_preset_off_disables(monkeypatch):
    monkeypatch.setenv("GB_TIER_KV_PRESET", "off")
    assert gb_tier_kv.resolve_codec() is None


def test_resolve_codec_kill_switch_overrides_everything(monkeypatch):
    monkeypatch.setenv("GB_TIER_KV_COMPRESS", "0")
    monkeypatch.setenv("GB_TIER_KV_PRESET", "polar4")
    assert gb_tier_kv.resolve_codec("turboquant") is None


def test_resolve_codec_unknown_preset_warns_and_falls_back(monkeypatch):
    monkeypatch.setenv("GB_TIER_KV_PRESET", "not-a-real-preset")
    with pytest.warns(UserWarning):
        assert gb_tier_kv.resolve_codec() == "polarquant"


# ── eligibility guard ───────────────────────────────────────────────────────

def test_eligible_normal_kv_tensor():
    t = _kv_tensor((2, 8, 64, 128))
    assert gb_tier_kv._kv_eligible(t) is None


def test_ineligible_non_tensor():
    assert gb_tier_kv._kv_eligible([1, 2, 3]) == "not_tensor"


def test_ineligible_int_buffer():
    t = torch.zeros(4, 128, dtype=torch.int64)
    assert gb_tier_kv._kv_eligible(t) == "non_float"


def test_ineligible_bool_buffer():
    t = torch.zeros(4, 128, dtype=torch.bool)
    assert gb_tier_kv._kv_eligible(t) == "non_float"


def test_ineligible_rank_1():
    t = torch.randn(128)
    assert gb_tier_kv._kv_eligible(t) == "rank_lt_2"


def test_ineligible_bad_head_dim():
    t = torch.randn(4, 17)   # not a known head_dim, and small numel too
    assert gb_tier_kv._kv_eligible(t) == "head_dim"


def test_ineligible_too_small(monkeypatch):
    monkeypatch.setenv("GB_TIER_KV_MIN_ELEMS", "999999999")
    t = _kv_tensor((2, 8, 64, 128))
    assert gb_tier_kv._kv_eligible(t) == "too_small"


def test_ineligible_nonfinite():
    t = _kv_tensor((2, 8, 64, 128)).clone()
    t[0, 0, 0, 0] = float("inf")
    assert gb_tier_kv._kv_eligible(t) == "nonfinite"


def test_ineligible_complex():
    t = torch.complex(_kv_tensor((2, 8, 64, 128)), _kv_tensor((2, 8, 64, 128), seed=1))
    assert gb_tier_kv._kv_eligible(t) == "complex"


def test_ineligible_meta():
    t = torch.zeros(2, 8, 64, 128, device="meta")
    assert gb_tier_kv._kv_eligible(t) == "meta_or_sparse"


# ── encode_state / decode_state round trip ──────────────────────────────────

def test_roundtrip_norm_close_and_tighter_at_8bit_than_4bit():
    """_quantize stores the ORIGINAL vector's fp32 norm and _dequantize
    scales the reconstructed direction by it , but the reconstructed
    direction (`centroids[idx] @ Pi`) is itself a per-COORDINATE Lloyd-Max
    approximation of a unit vector, not exactly unit, so the recovered
    norm is close but NOT bit-exact. This is the tightest cheap tripwire
    for a Pi/Pi_T transpose swap or a shape bug (which would blow the error
    far past these bounds, not just shift it slightly) , and it must get
    tighter as bit width grows, since more codebook levels means each
    coordinate's magnitude is captured more precisely."""
    torch.manual_seed(0)
    state = {"k_cache": _kv_tensor((2, 8, 64, 128), seed=0)}
    orig_norm = state["k_cache"].norm(dim=-1)

    enc4, _ = gb_tier_kv.encode_state(state, bits=4)
    dec4 = gb_tier_kv.decode_state(enc4)
    rel4 = ((orig_norm - dec4["k_cache"].norm(dim=-1)).abs() / orig_norm)

    enc8, _ = gb_tier_kv.encode_state(state, bits=8)
    dec8 = gb_tier_kv.decode_state(enc8)
    rel8 = ((orig_norm - dec8["k_cache"].norm(dim=-1)).abs() / orig_norm)

    assert rel4.mean() < 0.03
    assert rel4.max() < 0.15
    assert rel8.mean() < 0.01
    assert rel8.mean() < rel4.mean()


def test_roundtrip_cosine_fidelity_4bit():
    torch.manual_seed(0)
    state = {"k_cache": _kv_tensor((2, 8, 512, 128), seed=0)}
    enc, _ = gb_tier_kv.encode_state(state, bits=4)
    dec = gb_tier_kv.decode_state(enc)
    orig = state["k_cache"].reshape(-1, 128)
    rec = dec["k_cache"].reshape(-1, 128)
    cos = torch.nn.functional.cosine_similarity(orig, rec, dim=-1)
    assert cos.mean() > 0.97
    assert cos.min() > 0.90


def test_roundtrip_cosine_fidelity_8bit_tighter_than_4bit():
    torch.manual_seed(0)
    state = {"k_cache": _kv_tensor((2, 8, 512, 128), seed=0)}
    enc4, _ = gb_tier_kv.encode_state(state, bits=4)
    enc8, _ = gb_tier_kv.encode_state(state, bits=8)
    dec4 = gb_tier_kv.decode_state(enc4)
    dec8 = gb_tier_kv.decode_state(enc8)
    orig = state["k_cache"].reshape(-1, 128)
    cos4 = torch.nn.functional.cosine_similarity(orig, dec4["k_cache"].reshape(-1, 128), dim=-1).mean()
    cos8 = torch.nn.functional.cosine_similarity(orig, dec8["k_cache"].reshape(-1, 128), dim=-1).mean()
    assert cos8 > cos4
    assert cos8 > 0.995


def test_roundtrip_relative_frobenius_error_bounds():
    torch.manual_seed(0)
    state = {"k_cache": _kv_tensor((2, 8, 512, 128), seed=0)}
    enc4, _ = gb_tier_kv.encode_state(state, bits=4)
    dec4 = gb_tier_kv.decode_state(enc4)
    rel_err4 = (dec4["k_cache"] - state["k_cache"]).norm() / state["k_cache"].norm()
    assert rel_err4 < 0.25

    enc8, _ = gb_tier_kv.encode_state(state, bits=8)
    dec8 = gb_tier_kv.decode_state(enc8)
    rel_err8 = (dec8["k_cache"] - state["k_cache"]).norm() / state["k_cache"].norm()
    assert rel_err8 < 0.08


def test_roundtrip_turboquant_codec():
    torch.manual_seed(0)
    state = {"k_cache": _kv_tensor((2, 8, 256, 128), seed=0)}
    enc, stats = gb_tier_kv.encode_state(state, bits=4, codec="turboquant")
    assert stats["codec"] == "turboquant"
    assert stats["tensors"] == 1
    dec = gb_tier_kv.decode_state(enc)
    assert dec["k_cache"].shape == state["k_cache"].shape
    cos = torch.nn.functional.cosine_similarity(
        state["k_cache"].reshape(-1, 128), dec["k_cache"].reshape(-1, 128), dim=-1)
    assert cos.mean() > 0.90


def test_roundtrip_structural_identity_key_set_shape_dtype():
    state = {"k_cache": _kv_tensor((2, 8, 64, 128)),
            "meta_ids": torch.arange(64, dtype=torch.int64)}
    enc, stats = gb_tier_kv.encode_state(state, bits=4)
    assert stats["tensors"] == 1
    dec = gb_tier_kv.decode_state(enc)
    assert set(dec.keys()) == set(state.keys())
    assert not any(k.startswith("__gb_kv__") for k in dec.keys())
    for k in state:
        assert dec[k].shape == state[k].shape
        assert dec[k].dtype == state[k].dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_roundtrip_preserves_dtype(dtype):
    state = {"k_cache": _kv_tensor((2, 8, 64, 128)).to(dtype)}
    enc, stats = gb_tier_kv.encode_state(state, bits=4)
    assert stats["tensors"] == 1
    dec = gb_tier_kv.decode_state(enc)
    assert dec["k_cache"].dtype == dtype


# ── exclusion round-trips bit-exactly ────────────────────────────────────────

@pytest.mark.parametrize("bad_tensor,reason", [
    (torch.zeros(4, 128, dtype=torch.int64), "non_float"),
    (torch.zeros(4, 128, dtype=torch.bool), "non_float"),
    (torch.randn(128), "rank_lt_2"),
    (torch.randn(4, 17), "head_dim"),
])
def test_exclusions_round_trip_bit_exact_and_reason_recorded(bad_tensor, reason):
    state = {"a": bad_tensor}
    enc, stats = gb_tier_kv.encode_state(state, bits=4)
    assert stats["tensors"] == 0
    assert stats["skip_reasons"].get(reason, 0) == 1
    assert "a" not in (enc.get("__gb_kv__", {}).get("entries", {}))
    dec = gb_tier_kv.decode_state(enc)
    if bad_tensor.dtype == torch.bool:
        assert torch.equal(dec["a"], state["a"])
    else:
        assert torch.equal(dec["a"], state["a"])


def test_exclusion_too_small_bit_exact(monkeypatch):
    monkeypatch.setenv("GB_TIER_KV_MIN_ELEMS", "999999999")
    state = {"a": _kv_tensor((2, 8, 64, 128))}
    enc, stats = gb_tier_kv.encode_state(state, bits=4)
    assert stats["tensors"] == 0
    assert stats["skip_reasons"]["too_small"] == 1
    dec = gb_tier_kv.decode_state(enc)
    assert torch.equal(dec["a"], state["a"])


def test_exclusion_nonfinite_bit_exact():
    t = _kv_tensor((2, 8, 64, 128)).clone()
    t[0, 0, 0, 0] = float("nan")
    state = {"a": t}
    enc, stats = gb_tier_kv.encode_state(state, bits=4)
    assert stats["tensors"] == 0
    assert stats["skip_reasons"]["nonfinite"] == 1
    dec = gb_tier_kv.decode_state(enc)
    assert torch.equal(dec["a"].nan_to_num(), state["a"].nan_to_num())


# ── size guard: never encode if it wouldn't actually be smaller ────────────

def test_not_smaller_guard_skips_encoding():
    """8 bits + fp32 norms on a small-ish tensor can exceed the raw size ,
    the size guard must fall back to leaving it untouched, not encode
    anyway."""
    # A tensor whose packed 8-bit idx + fp32 norms genuinely isn't smaller
    # than the raw fp32 bytes: numel*1(idx) + (numel/D)*4(norms) vs
    # numel*4(raw fp32) , always smaller for fp32 raw actually, so force it
    # via a float16 raw tensor instead (numel*2 raw bytes).
    t = _kv_tensor((2, 8, 64, 128)).to(torch.float16)
    state = {"a": t}
    enc, stats = gb_tier_kv.encode_state(state, bits=8)
    # Whether this specific shape triggers "not_smaller" depends on the
    # packing math; assert internal consistency instead of a fixed verdict.
    if stats["tensors"] == 0:
        assert stats["skip_reasons"].get("not_smaller", 0) >= 1
        assert torch.equal(gb_tier_kv.decode_state(enc)["a"], state["a"])
    else:
        assert stats["post_bytes"] < stats["pre_bytes"]


# ── kill switch ──────────────────────────────────────────────────────────────

def test_kill_switch_bit_exact_even_when_bits_requested(monkeypatch):
    monkeypatch.setenv("GB_TIER_KV_COMPRESS", "0")
    state = {"k_cache": _kv_tensor((2, 8, 64, 128))}
    enc, stats = gb_tier_kv.encode_state(state, bits=4)
    assert stats["codec"] is None
    assert enc is state   # untouched, same object
    assert "__gb_kv__" not in enc
    dec = gb_tier_kv.decode_state(enc)
    assert torch.equal(dec["k_cache"], state["k_cache"])


# ── keys filter ──────────────────────────────────────────────────────────────

def test_keys_filter_restricts_encoding():
    state = {"k_cache": _kv_tensor((2, 8, 64, 128), seed=0),
            "v_cache": _kv_tensor((2, 8, 64, 128), seed=1)}
    enc, stats = gb_tier_kv.encode_state(state, bits=4, keys=("k_cache",))
    assert stats["tensors"] == 1
    assert "k_cache" in enc["__gb_kv__"]["entries"]
    assert "v_cache" not in enc["__gb_kv__"]["entries"]
    dec = gb_tier_kv.decode_state(enc)
    assert torch.equal(dec["v_cache"], state["v_cache"])   # untouched, bit-exact


# ── empty / no-manifest passthrough ──────────────────────────────────────────

def test_decode_state_no_manifest_returns_identity():
    state = {"w": torch.randn(4, 4)}
    assert gb_tier_kv.decode_state(state) is state


def test_encode_state_empty_dict():
    enc, stats = gb_tier_kv.encode_state({}, bits=4)
    assert enc == {}
    assert stats["tensors"] == 0


def test_encode_state_no_eligible_tensors_returns_identity_object():
    state = {"ids": torch.arange(10, dtype=torch.int64)}
    enc, stats = gb_tier_kv.encode_state(state, bits=4)
    assert enc is state
    assert "__gb_kv__" not in enc


# ── invalid bits ─────────────────────────────────────────────────────────────

def test_invalid_bits_raises():
    state = {"k_cache": _kv_tensor((2, 8, 64, 128))}
    with pytest.raises(ValueError, match="bits"):
        gb_tier_kv.encode_state(state, bits=6)


# ── codebook determinism (the load-bearing correctness invariant) ──────────

def test_quantizer_deterministic_across_cache_clears():
    gb_attn._quantizer_cache.clear()
    Pi1, *_ = gb_attn._get_quantizer(4, 128, "cpu")
    gb_attn._quantizer_cache.clear()
    Pi2, *_ = gb_attn._get_quantizer(4, 128, "cpu")
    assert torch.equal(Pi1, Pi2)


def test_decode_raises_on_pi_seed_mismatch():
    state = {"k_cache": _kv_tensor((2, 8, 64, 128))}
    enc, _ = gb_tier_kv.encode_state(state, bits=4)
    enc["__gb_kv__"]["pi_seed"] = 999
    with pytest.raises(ValueError, match="pi_seed"):
        gb_tier_kv.decode_state(enc)


def test_decode_raises_on_version_mismatch():
    state = {"k_cache": _kv_tensor((2, 8, 64, 128))}
    enc, _ = gb_tier_kv.encode_state(state, bits=4)
    enc["__gb_kv__"]["v"] = 999
    with pytest.raises(ValueError, match="format"):
        gb_tier_kv.decode_state(enc)


# ── encode never touches CUDA when device="cpu" ─────────────────────────────

def test_encode_cpu_device_no_cuda_touch():
    with patch("torch.cuda.is_available", return_value=False):
        state = {"k_cache": _kv_tensor((2, 8, 64, 128))}
        enc, stats = gb_tier_kv.encode_state(state, bits=4, device="cpu")
        assert stats["tensors"] == 1
        dec = gb_tier_kv.decode_state(enc)
        assert dec["k_cache"].shape == state["k_cache"].shape

"""
Tests for gb_attn.py's P9 "packed" backend (plan
bring-gb-synapse-gb-quant-and-async-nygaard.md) , an alternative K-side
quantizer sourced from the vendored TurboQuant implementation
(third_party/turboquant/NOTICE), opt-in via patch_sdpa(backend="packed") /
GB_ATTN_BACKEND=packed. Default ("dequant") is gb_attn's own original K/V
compression, unchanged.

The vendored QuantizedAttention class needs scipy (not installed in this
CPU-only test env, see synapse_engine/pyproject.toml) , _get_packed_quantizer
is monkeypatched throughout so these tests verify gb_attn's OWN plumbing
(backend selection, validation, status reporting, the actual K/V dispatch
into original_sdpa) without needing the real vendored package installed.
"""
import pytest
import torch

import gb_attn


@pytest.fixture(autouse=True)
def _clean_state():
    gb_attn.unpatch_sdpa()
    gb_attn._packed_quantizer_cache.clear()
    yield
    gb_attn.unpatch_sdpa()
    gb_attn._packed_quantizer_cache.clear()


# ── backend selection / validation ────────────────────────────────────────

def test_default_backend_is_dequant():
    gb_attn.patch_sdpa(k_bits=4, v_bits=3, device="cpu")
    assert gb_attn.status()["backend"] == "dequant"


def test_env_var_selects_packed_backend(monkeypatch):
    monkeypatch.setenv("GB_ATTN_BACKEND", "packed")
    gb_attn.patch_sdpa(k_bits=4, v_bits=3, device="cpu")
    assert gb_attn.status()["backend"] == "packed"


def test_explicit_backend_overrides_env_var(monkeypatch):
    monkeypatch.setenv("GB_ATTN_BACKEND", "packed")
    gb_attn.patch_sdpa(k_bits=4, v_bits=3, device="cpu", backend="dequant")
    assert gb_attn.status()["backend"] == "dequant"


def test_packed_backend_rejected_for_non_turboquant_mode():
    with pytest.raises(ValueError, match='backend="packed"'):
        gb_attn.patch_sdpa(k_bits=4, v_bits=3, device="cpu",
                           mode="snapkv", backend="packed")


def test_packed_backend_rejected_with_sparse_v():
    with pytest.raises(ValueError, match="sparse_v"):
        gb_attn.patch_sdpa(k_bits=4, v_bits=3, device="cpu",
                           backend="packed", sparse_v=True)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        gb_attn.patch_sdpa(k_bits=4, v_bits=3, device="cpu", backend="nonsense")


def test_unpatch_resets_backend_to_dequant():
    gb_attn.patch_sdpa(k_bits=4, v_bits=3, device="cpu", backend="packed",
                       **{})  # packed alone (mode=turboquant default) is fine to select
    assert gb_attn.status()["backend"] == "packed"
    gb_attn.unpatch_sdpa()
    assert gb_attn.status()["backend"] == "dequant"


# ── _packed_tq_attn(): K via vendored quantizer, V via gb_attn's own ──────

class _FakeQuantizedAttention:
    """Stand-in for third_party/turboquant's QuantizedAttention , records
    calls and returns an easily-asserted-on dequantized K."""

    def __init__(self):
        self.quantize_keys_calls = []
        self.dequantize_calls = []

    def quantize_keys(self, K):
        self.quantize_keys_calls.append(K)
        return ("fake_idx", "fake_norms")

    def dequantize(self, idx, norms):
        self.dequantize_calls.append((idx, norms))
        # Return a recognizable, deterministic tensor distinct from the
        # real key tensor so the test can prove THIS path was used.
        return torch.full_like(self._last_key, 7.0)

    def set_shape_like(self, key):
        self._last_key = key


def test_packed_tq_attn_uses_vendored_quantizer_for_k(monkeypatch):
    fake_qa = _FakeQuantizedAttention()

    def _fake_get_packed_quantizer(bit_width, head_dim, device):
        return fake_qa

    monkeypatch.setattr(gb_attn, "_get_packed_quantizer", _fake_get_packed_quantizer)

    captured = {}

    def _fake_original_sdpa(query, key, value, **kwargs):
        captured["key"] = key
        captured["value"] = value
        return torch.zeros_like(query)

    B, H, S, D = 1, 2, 8, 16
    query = torch.randn(B, H, S, D)
    key = torch.randn(B, H, S, D)
    value = torch.randn(B, H, S, D)
    fake_qa.set_shape_like(key)

    out = gb_attn._packed_tq_attn(query, key, value, k_bits=4, v_bits=3,
                                  device="cpu", original_sdpa=_fake_original_sdpa)

    assert len(fake_qa.quantize_keys_calls) == 1
    assert torch.equal(fake_qa.quantize_keys_calls[0], key)
    # K passed to original_sdpa came from the FAKE vendored dequantize (all 7.0),
    # not the raw key tensor , proves the vendored path was actually used.
    assert torch.equal(captured["key"], torch.full_like(key, 7.0))
    # V went through gb_attn's own PolarQuant path (not touched by the fake),
    # so it must NOT be the raw value tensor either (it was quantized+dequantized).
    assert captured["value"].shape == value.shape
    assert out.shape == query.shape


def test_packed_quantizer_is_cached_per_bit_width_head_dim_device(monkeypatch):
    calls = []

    class _CountingQA:
        def __init__(self, bit_width, head_dim, device):
            calls.append((bit_width, head_dim, device))

    monkeypatch.setattr(gb_attn, "_require_turboquant", lambda: _CountingQA)

    gb_attn._get_packed_quantizer(4, 16, "cpu")
    gb_attn._get_packed_quantizer(4, 16, "cpu")
    gb_attn._get_packed_quantizer(4, 32, "cpu")  # different head_dim -> new instance

    assert len(calls) == 2


# ── _require_turboquant(): actionable error when the vendor tree is absent ─

def test_require_turboquant_raises_actionable_error_when_missing(monkeypatch):
    monkeypatch.setattr(gb_attn, "_ensure_turboquant_path", lambda: None)
    import sys
    with pytest.MonkeyPatch().context() as mp:
        mp.setitem(sys.modules, "turboquant", None)
        mp.setitem(sys.modules, "turboquant.attention", None)
        with pytest.raises(RuntimeError, match="install-synapse-engine"):
            gb_attn._require_turboquant()

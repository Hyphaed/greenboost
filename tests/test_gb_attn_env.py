"""gb_attn.turboquant_attention_from_env — canonical GB_TQ_ATTN spec parsing
(promoted from ai-forge gen_image._turboquant_ctx). CPU-only: the underlying
turboquant_attention is monkeypatched to capture kwargs, never entered.
"""
import contextlib

import pytest

import gb_attn


@pytest.fixture
def capture(monkeypatch):
    calls = []

    def _fake(**kwargs):
        calls.append(kwargs)
        return contextlib.nullcontext()
    monkeypatch.setattr(gb_attn, "turboquant_attention", _fake)
    return calls


def test_unset_returns_nullcontext(monkeypatch, capture):
    monkeypatch.delenv("GB_TQ_ATTN", raising=False)
    ctx = gb_attn.turboquant_attention_from_env()
    assert isinstance(ctx, contextlib.nullcontext)
    assert capture == []


def test_asymmetric_k4v3(monkeypatch, capture):
    monkeypatch.setenv("GB_TQ_ATTN", "k4v3")
    gb_attn.turboquant_attention_from_env()
    assert capture == [{"k_bits": 4, "v_bits": 3.0}]


def test_asymmetric_float_v(monkeypatch, capture):
    monkeypatch.setenv("GB_TQ_ATTN", "k4v2.5")
    gb_attn.turboquant_attention_from_env()
    assert capture == [{"k_bits": 4, "v_bits": 2.5}]


def test_plain_bit_width(monkeypatch, capture):
    monkeypatch.setenv("GB_TQ_ATTN", "3")
    gb_attn.turboquant_attention_from_env()
    assert capture == [{"bit_width": 3.0}]


def test_garbage_degrades_to_nullcontext(monkeypatch, capture):
    monkeypatch.setenv("GB_TQ_ATTN", "banana")
    ctx = gb_attn.turboquant_attention_from_env()
    assert isinstance(ctx, contextlib.nullcontext)
    assert capture == []


def test_custom_var_name(monkeypatch, capture):
    monkeypatch.delenv("GB_TQ_ATTN", raising=False)
    monkeypatch.setenv("GB_TQ_ATTN_VIDEO", "k4v3")
    gb_attn.turboquant_attention_from_env("GB_TQ_ATTN_VIDEO")
    assert capture == [{"k_bits": 4, "v_bits": 3.0}]


def test_underlying_failure_degrades(monkeypatch):
    monkeypatch.setenv("GB_TQ_ATTN", "k4v3")

    def _boom(**kwargs):
        raise RuntimeError("no CUDA")
    monkeypatch.setattr(gb_attn, "turboquant_attention", _boom)
    ctx = gb_attn.turboquant_attention_from_env()
    assert isinstance(ctx, contextlib.nullcontext)

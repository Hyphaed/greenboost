"""gb_attn._sparse_v_max_seq — DI-7: %-derived sparse-V sequence limit.

CPU-only: torch.cuda.mem_get_info is monkeypatched, never touches real VRAM.
"""
import pytest

import gb_attn


@pytest.fixture(autouse=True)
def _clear_cache():
    gb_attn._sparse_v_max_seq_cache.clear()
    yield
    gb_attn._sparse_v_max_seq_cache.clear()


def test_derives_from_free_vram(monkeypatch):
    monkeypatch.delenv("GB_SPARSE_V_MAX_SEQ", raising=False)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (8 * 1024**3, 12 * 1024**3))
    max_seq = gb_attn._sparse_v_max_seq(batch=1, n_heads=32)
    assert max_seq > 0
    assert max_seq != 4096   # not the flat literal when VRAM is detectable


def test_more_free_vram_yields_larger_max_seq(monkeypatch):
    monkeypatch.delenv("GB_SPARSE_V_MAX_SEQ", raising=False)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (2 * 1024**3, 12 * 1024**3))
    small = gb_attn._sparse_v_max_seq(batch=1, n_heads=32)
    gb_attn._sparse_v_max_seq_cache.clear()
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (40 * 1024**3, 48 * 1024**3))
    large = gb_attn._sparse_v_max_seq(batch=1, n_heads=32)
    assert large > small


def test_more_heads_yields_smaller_max_seq(monkeypatch):
    monkeypatch.delenv("GB_SPARSE_V_MAX_SEQ", raising=False)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (8 * 1024**3, 12 * 1024**3))
    few_heads = gb_attn._sparse_v_max_seq(batch=1, n_heads=8)
    many_heads = gb_attn._sparse_v_max_seq(batch=1, n_heads=64)
    assert few_heads > many_heads


def test_cache_is_keyed_by_shape_not_global(monkeypatch):
    monkeypatch.delenv("GB_SPARSE_V_MAX_SEQ", raising=False)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (8 * 1024**3, 12 * 1024**3))
    a = gb_attn._sparse_v_max_seq(batch=1, n_heads=32)
    b = gb_attn._sparse_v_max_seq(batch=4, n_heads=32)
    assert a != b   # different shape must not silently reuse the first shape's cached answer


def test_env_override_takes_precedence(monkeypatch):
    monkeypatch.setenv("GB_SPARSE_V_MAX_SEQ", "777")
    assert gb_attn._sparse_v_max_seq(batch=1, n_heads=32) == 777


def test_falls_back_loudly_when_vram_undetectable(monkeypatch, capsys):
    monkeypatch.delenv("GB_SPARSE_V_MAX_SEQ", raising=False)
    def _boom():
        raise RuntimeError("no CUDA context")
    monkeypatch.setattr("torch.cuda.mem_get_info", _boom)
    max_seq = gb_attn._sparse_v_max_seq(batch=1, n_heads=32)
    assert max_seq == gb_attn.SPARSE_V_MAX_SEQ
    assert "undetectable" in capsys.readouterr().out


def test_result_bounded_by_floor_and_ceiling(monkeypatch):
    monkeypatch.delenv("GB_SPARSE_V_MAX_SEQ", raising=False)
    # Absurdly tiny VRAM -> floor kicks in, not zero/negative.
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (1, 12 * 1024**3))
    assert gb_attn._sparse_v_max_seq(batch=64, n_heads=128) >= 512
    gb_attn._sparse_v_max_seq_cache.clear()
    # Absurdly huge VRAM -> ceiling caps it, doesn't silently reintroduce
    # unbounded-memory risk.
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (10**15, 10**15))
    assert gb_attn._sparse_v_max_seq(batch=1, n_heads=1) <= 131072


def test_sparse_v_tau_env_override():
    import importlib
    import os
    os.environ["GB_SPARSE_V_TAU"] = "0.05"
    try:
        importlib.reload(gb_attn)
        assert gb_attn.SPARSE_V_THRESHOLD == 0.05
    finally:
        os.environ.pop("GB_SPARSE_V_TAU", None)
        importlib.reload(gb_attn)

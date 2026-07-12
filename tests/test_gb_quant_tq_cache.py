"""CPU-only tests for the TQ LUT-GEMM persistent autotune cache
(gb_quant_tq._tq_cache_path / _arm_tq_cache).  No GPU or kernel launch: the
Autotuner is faked and torch.cuda.is_available is monkeypatched.
"""
import json
import os

import pytest

import gb_quant_tq as tq


class _FakeKernel:
    """Stand-in for the triton Autotuner: only needs a .cache dict."""

    def __init__(self):
        self.cache = {}


class _FakeConfig:
    def __init__(self, kwargs, num_warps=4, num_stages=2, num_ctas=1):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.num_ctas = num_ctas


@pytest.fixture(autouse=True)
def _reset_armed(monkeypatch):
    monkeypatch.setattr(tq, "_tq_cache_armed", False)
    # Triton is required for the arm path; pretend it imported fine.
    monkeypatch.setattr(tq, "_TRITON_ERR", None)


def _use_fake_config(monkeypatch):
    """Route (de)serialization through _FakeConfig so no real triton needed."""
    monkeypatch.setattr(tq, "_dict_to_config",
                        lambda d: _FakeConfig(d["kwargs"],
                                              d.get("num_warps", 4),
                                              d.get("num_stages", 2),
                                              d.get("num_ctas", 1)))


def test_cache_path_disabled_env(monkeypatch):
    monkeypatch.setenv("GB_QUANT_NO_AUTOTUNE_CACHE", "1")
    monkeypatch.setattr(tq.torch.cuda, "is_available", lambda: True)
    assert tq._tq_cache_path() is None


def test_cache_path_no_cuda(monkeypatch):
    monkeypatch.delenv("GB_QUANT_NO_AUTOTUNE_CACHE", raising=False)
    monkeypatch.setattr(tq.torch.cuda, "is_available", lambda: False)
    assert tq._tq_cache_path() is None


def test_cache_path_shape(monkeypatch, tmp_path):
    monkeypatch.delenv("GB_QUANT_NO_AUTOTUNE_CACHE", raising=False)
    monkeypatch.setenv("GB_QUANT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tq.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tq, "_tq_cache_key", lambda: "TestGPU_cu130_tr370")
    p = tq._tq_cache_path()
    assert p == os.path.join(str(tmp_path),
                             "tq_autotune_TestGPU_cu130_tr370.json")


def test_arm_saves_and_reloads(monkeypatch, tmp_path):
    _use_fake_config(monkeypatch)
    path = str(tmp_path / "tq.json")
    monkeypatch.setattr(tq, "_tq_cache_path", lambda: path)

    # capture the atexit save callback instead of registering it globally
    saved_cbs = []
    monkeypatch.setattr("atexit.register", lambda fn: saved_cbs.append(fn))

    # first process: kernel autotuned two shapes
    k1 = _FakeKernel()
    k1.cache[(1024, 4096, 4096, 3)] = _FakeConfig({"BLOCK_M": 64}, 8, 3)
    k1.cache[(16, 4096, 4096, 2)] = _FakeConfig({"BLOCK_M": 16}, 4, 4, 2)
    monkeypatch.setattr(tq, "_tq_gemm_kernel", k1)
    tq._arm_tq_cache()
    assert saved_cbs, "arm should register an at-exit save"
    saved_cbs[0]()                      # simulate process exit
    assert os.path.isfile(path)

    # second process: fresh kernel, arm should pre-populate its cache
    monkeypatch.setattr(tq, "_tq_cache_armed", False)
    k2 = _FakeKernel()
    monkeypatch.setattr(tq, "_tq_gemm_kernel", k2)
    tq._arm_tq_cache()
    assert (1024, 4096, 4096, 3) in k2.cache
    assert (16, 4096, 4096, 2) in k2.cache
    assert k2.cache[(1024, 4096, 4096, 3)].kwargs == {"BLOCK_M": 64}
    assert k2.cache[(16, 4096, 4096, 2)].num_ctas == 2


def test_arm_tolerates_corrupt_file(monkeypatch, tmp_path):
    _use_fake_config(monkeypatch)
    path = str(tmp_path / "tq.json")
    with open(path, "w") as f:
        f.write("{not valid json")
    monkeypatch.setattr(tq, "_tq_cache_path", lambda: path)
    monkeypatch.setattr("atexit.register", lambda fn: None)
    k = _FakeKernel()
    monkeypatch.setattr(tq, "_tq_gemm_kernel", k)
    tq._arm_tq_cache()                  # must not raise
    assert k.cache == {}


def test_arm_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(tq, "_tq_cache_path", lambda: None)
    registered = []
    monkeypatch.setattr("atexit.register", lambda fn: registered.append(fn))
    k = _FakeKernel()
    monkeypatch.setattr(tq, "_tq_gemm_kernel", k)
    tq._arm_tq_cache()
    assert registered == []             # nothing to save when disabled


def test_arm_is_idempotent(monkeypatch, tmp_path):
    _use_fake_config(monkeypatch)
    path = str(tmp_path / "tq.json")
    monkeypatch.setattr(tq, "_tq_cache_path", lambda: path)
    calls = []
    monkeypatch.setattr("atexit.register", lambda fn: calls.append(fn))
    k = _FakeKernel()
    monkeypatch.setattr(tq, "_tq_gemm_kernel", k)
    tq._arm_tq_cache()
    tq._arm_tq_cache()                   # second call short-circuits
    assert len(calls) == 1


@pytest.mark.skipif(tq._TRITON_ERR is not None, reason="triton not importable")
def test_real_config_roundtrip():
    import triton
    cfg = triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=8,
                        num_stages=3)
    d = tq._config_to_dict(cfg)
    assert d["kwargs"] == {"BLOCK_M": 32, "BLOCK_N": 128}
    assert d["num_warps"] == 8 and d["num_stages"] == 3
    # survive a JSON round-trip and rebuild
    back = tq._dict_to_config(json.loads(json.dumps(d)))
    assert back.kwargs == cfg.kwargs
    assert back.num_warps == 8 and back.num_stages == 3

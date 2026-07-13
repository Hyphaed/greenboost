"""Contract test pinning gb_cluster.shim_env's diffusion profile.

This is the guard against the env-drift class that caused the 2026-07-07 fp8
OOM: ai-forge (and every feeder dispatch) depends on this exact key set, so a
silent profile edit must fail loudly here rather than in a live generation.
CPU-only: shim_supported + sysfs are monkeypatched, no /dev or /sys access.
"""
import pytest

import gb_cluster as gc


# The keys ai-forge's diffusion path relies on (forge/gpu.py). If any of these
# stops being emitted by shim_env("diffusion"), the consumer regresses.
_REQUIRED_DIFFUSION_KEYS = {
    "GREENBOOST_ACTIVE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
    "GREENBOOST_KV_RESERVE_MB": "0",
    "GREENBOOST_A0_DISABLE": "1",
    "GREENBOOST_DISABLE_T2_ON_BLACKWELL": "0",
    "GREENBOOST_CLUSTER": "0",
    "GREENBOOST_KV_COMPRESS": "1",
    "GREENBOOST_REPORT_PHYSICAL_VRAM": "1",
}


@pytest.fixture(autouse=True)
def _shim_present(monkeypatch):
    monkeypatch.setattr(gc, "shim_supported", lambda: True)
    monkeypatch.setattr(gc, "GREENBOOST_SHIM", "/usr/local/lib/libgreenboost_cuda.so")
    # default: no kernel pool visible (isolate T2_POOL sizing per-test)
    monkeypatch.setattr(gc, "_local_t2_pool_total_mb", lambda: None)


def _clean_base():
    # a base env with none of the greenboost keys set, so setdefault fills them
    return {"PATH": "/usr/bin", "HOME": "/tmp/testhome"}


def test_diffusion_profile_complete():
    env = gc.shim_env("diffusion", base_env=_clean_base())
    for k, v in _REQUIRED_DIFFUSION_KEYS.items():
        assert env.get(k) == v, f"diffusion profile lost {k}"
    assert env["LD_PRELOAD"].startswith("/usr/local/lib/libgreenboost_cuda.so")


def test_host_ram_safety_dynamic():
    # No longer a static "2048" literal: %-derived from MemTotal at call time
    # (max(2048, 3%)), so assert presence + the formula's floor, not a value.
    env = gc.shim_env("diffusion", base_env=_clean_base())
    assert int(env["GREENBOOST_HOST_RAM_SAFETY_MB"]) >= 2048


def test_torch_alias_matches_diffusion():
    a = gc.shim_env("torch", base_env=_clean_base())
    b = gc.shim_env("diffusion", base_env=_clean_base())
    for k in _REQUIRED_DIFFUSION_KEYS:
        assert a.get(k) == b.get(k)


def test_caller_env_wins():
    base = _clean_base()
    base["GREENBOOST_KV_COMPRESS"] = "0"      # explicit caller override
    env = gc.shim_env("diffusion", base_env=base)
    assert env["GREENBOOST_KV_COMPRESS"] == "0"


def test_disabled_strips_shim():
    base = _clean_base()
    base["GREENBOOST_ACTIVE"] = "1"
    base["LD_PRELOAD"] = "/usr/local/lib/libgreenboost_cuda.so:/other/lib.so"
    env = gc.shim_env("diffusion", enabled=False, base_env=base)
    assert "GREENBOOST_ACTIVE" not in env
    assert env["LD_PRELOAD"] == "/other/lib.so"


def test_disabled_drops_ld_preload_when_only_shim():
    base = _clean_base()
    base["LD_PRELOAD"] = "/usr/local/lib/libgreenboost_cuda.so"
    env = gc.shim_env("diffusion", enabled=False, base_env=base)
    assert "LD_PRELOAD" not in env


def test_t2_pool_sizing(monkeypatch):
    monkeypatch.setattr(gc, "_local_t2_pool_total_mb", lambda: 42 * 1024)
    env = gc.shim_env("diffusion", base_env=_clean_base())
    # 85% of 43008 MB
    assert env["GREENBOOST_T2_POOL_MB"] == str(43008 * 85 // 100)


def test_t2_pool_absent_leaves_unset(monkeypatch):
    monkeypatch.setattr(gc, "_local_t2_pool_total_mb", lambda: None)
    env = gc.shim_env("diffusion", base_env=_clean_base())
    assert "GREENBOOST_T2_POOL_MB" not in env


def test_cudart_path_set_for_torch():
    env = gc.shim_env("diffusion", base_env=_clean_base(),
                      cudart_path="/env/nvidia/cu13/lib/libcudart.so.13")
    assert env["GREENBOOST_CUDART_PATH"] == "/env/nvidia/cu13/lib/libcudart.so.13"


def test_cudart_and_t2_not_applied_to_llm(monkeypatch):
    monkeypatch.setattr(gc, "_local_t2_pool_total_mb", lambda: 42 * 1024)
    env = gc.shim_env("llm", base_env=_clean_base(),
                      cudart_path="/env/lib/libcudart.so.13")
    assert "GREENBOOST_T2_POOL_MB" not in env
    assert "GREENBOOST_CUDART_PATH" not in env
    assert env["GREENBOOST_CLUSTER"] == "1"


def test_unknown_workload_raises():
    with pytest.raises(ValueError):
        gc.shim_env("bogus", base_env=_clean_base())

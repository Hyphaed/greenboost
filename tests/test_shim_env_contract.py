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
    # Default: vmm_override absent, so existing LD_PRELOAD assertions below
    # (written before vmm_override was preloaded) stay deterministic
    # regardless of what's actually installed on the box running the tests.
    monkeypatch.setattr(gc, "GREENBOOST_VMM_OVERRIDE",
                        "/nonexistent/libgreenboost_vmm_override.so")
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


def test_llm_profile_has_frontload_and_cluster(monkeypatch):
    monkeypatch.setattr(gc, "_local_t2_pool_total_mb", lambda: None)
    env = gc.shim_env("llm", base_env=_clean_base())
    assert env["GREENBOOST_CLUSTER"] == "1"
    # Owner rule 2026-08-01 (T2 spill through the shim, never CPU offload):
    # llm gets the same Rule #1 front-load split "diffusion" already had —
    # a weights buffer bigger than free VRAM must fill VRAM to ~90% first,
    # not dump the whole buffer into T2 while T1 sits part-empty.
    assert env["GB_VRAM_FRONTLOAD"] == "1"


def test_ggml_alias_matches_llm():
    a = gc.shim_env("ggml", base_env=_clean_base())
    b = gc.shim_env("llm", base_env=_clean_base())
    assert a.get("GB_VRAM_FRONTLOAD") == b.get("GB_VRAM_FRONTLOAD") == "1"


def test_llm_joined_t2_pool_workloads(monkeypatch):
    """2026-08-01: an LLM spilling weights/KV to T2 through the shim needs
    the same %-derived safety/pool sizing diffusion and torch already get —
    llm was previously excluded, leaving the shim's flat defaults in place
    for every gb-synapse llama.cpp serve."""
    monkeypatch.setattr(gc, "_local_t2_pool_total_mb", lambda: 42 * 1024)
    env = gc.shim_env("llm", base_env=_clean_base(),
                      cudart_path="/env/lib/libcudart.so.13")
    assert env["GREENBOOST_T2_POOL_MB"] == str(43008 * 85 // 100)
    assert int(env["GREENBOOST_HOST_RAM_SAFETY_MB"]) >= 2048
    # cudart_path is accepted (harmless — gb-synapse's own llm serve calls
    # never pass it), not specifically excluded now that llm is T2-pooled.
    assert env["GREENBOOST_CUDART_PATH"] == "/env/lib/libcudart.so.13"


def test_llm_disables_pdl_by_default():
    """2026-08-01: ggml_cuda_kernel_can_use_pdl's cudaFuncGetAttributes probe
    aborts under this shim (split-brain libcudart — gb_shim_probe.py's
    docstring), confirmed on both Blackwell cc 12.0 and Ada cc 8.9. Default
    off so a fresh gb-synapse llama.cpp serve doesn't crash on first decode."""
    env = gc.shim_env("llm", base_env=_clean_base())
    assert env["GGML_CUDA_PDL"] == "0"


def test_llm_pdl_caller_override_wins():
    base = _clean_base()
    base["GGML_CUDA_PDL"] = "1"      # caller has verified PDL is safe here
    env = gc.shim_env("llm", base_env=base)
    assert env["GGML_CUDA_PDL"] == "1"


def test_vmm_override_preloaded_first_when_present(monkeypatch, tmp_path):
    vmm = tmp_path / "libgreenboost_vmm_override.so"
    vmm.write_bytes(b"")
    monkeypatch.setattr(gc, "GREENBOOST_VMM_OVERRIDE", str(vmm))

    env = gc.shim_env("llm", base_env=_clean_base())

    entries = env["LD_PRELOAD"].split(":")
    assert entries[0] == str(vmm)
    assert entries[1] == gc.GREENBOOST_SHIM


def test_vmm_override_absent_skipped(monkeypatch):
    monkeypatch.setattr(gc, "GREENBOOST_VMM_OVERRIDE",
                        "/nonexistent/libgreenboost_vmm_override.so")
    env = gc.shim_env("llm", base_env=_clean_base())
    assert env["LD_PRELOAD"] == gc.GREENBOOST_SHIM


def test_disabled_strips_vmm_override_too(monkeypatch, tmp_path):
    vmm = tmp_path / "libgreenboost_vmm_override.so"
    vmm.write_bytes(b"")
    monkeypatch.setattr(gc, "GREENBOOST_VMM_OVERRIDE", str(vmm))
    base = _clean_base()
    base["LD_PRELOAD"] = f"{vmm}:{gc.GREENBOOST_SHIM}:/other/lib.so"

    env = gc.shim_env("llm", enabled=False, base_env=base)

    assert env["LD_PRELOAD"] == "/other/lib.so"


def test_unknown_workload_raises():
    with pytest.raises(ValueError):
        gc.shim_env("bogus", base_env=_clean_base())

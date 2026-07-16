#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_synapse_backends.py: format detection, legacy-engine mapping,
and the %-derived VRAM+T2 budget helper (2026-07-16 vLLM/backend redesign).

CPU-only. No GGUF, no CUDA, no real HF network calls.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import gb_synapse_backends as gsb


# ── detect_format: local paths ────────────────────────────────────────────

def test_detect_format_local_gguf_file(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"")
    assert gsb.detect_format(str(f)) == "gguf"


def test_detect_format_local_gguf_dir(tmp_path):
    (tmp_path / "model-00001-of-00002.gguf").write_bytes(b"")
    assert gsb.detect_format(str(tmp_path)) == "gguf"


def test_detect_format_local_diffusers_dir(tmp_path):
    (tmp_path / "model_index.json").write_text("{}")
    (tmp_path / "unet").mkdir()
    assert gsb.detect_format(str(tmp_path)) == "diffusers"


def test_detect_format_local_safetensors_dir(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"")
    assert gsb.detect_format(str(tmp_path)) == "safetensors"


def test_detect_format_local_unknown_dir(tmp_path):
    (tmp_path / "readme.txt").write_text("hi")
    assert gsb.detect_format(str(tmp_path)) == "unknown"


def test_detect_format_local_bin_only_dir(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "pytorch_model.bin").write_bytes(b"")
    assert gsb.detect_format(str(tmp_path)) == "safetensors"


def test_detect_format_local_training_args_bin_only_is_unknown(tmp_path):
    """training_args.bin is a training-loop artifact, never a weight
    checkpoint — a dir with only that .bin file has no real weights."""
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "training_args.bin").write_bytes(b"")
    assert gsb.detect_format(str(tmp_path)) == "unknown"


def test_detect_format_remote_bin_only_repo(monkeypatch):
    class _FakeSibling:
        def __init__(self, name):
            self.rfilename = name

    class _FakeInfo:
        siblings = [_FakeSibling("config.json"), _FakeSibling("pytorch_model.bin")]

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def model_info(self, repo, files_metadata=False):
            return _FakeInfo()

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    import gb_synapse as gs
    monkeypatch.setattr(gs, "hf_token", lambda: None)
    assert gsb.detect_format("some/bin-only-repo") == "safetensors"


def test_detect_format_nonexistent_path_falls_back_to_hf_api(monkeypatch):
    """A path that doesn't exist locally is treated as a bare HF repo id —
    detect_format() should attempt the HfApi lookup and degrade to "unknown"
    when that's unavailable/fails, never raise."""
    assert gsb.detect_format("some/nonexistent-repo-id-xyz") == "unknown"


# ── select_backend: legacy "gbquant" + explicit engine routing ───────────

class _FakeEntry:
    def __init__(self, engine):
        self.engine = engine
        self.name = "fake"
        self.path = ""
        self.repo = ""
        self.quant = "FP8"
        self.arch = ""
        self.source = "hf"


def test_select_backend_llama_cpp_default():
    backend = gsb.select_backend(_FakeEntry("llama.cpp"))
    assert isinstance(backend, gsb.LlamaCppBackend)


def test_select_backend_diffusers_explicit():
    backend = gsb.select_backend(_FakeEntry("diffusers"))
    assert isinstance(backend, gsb.DiffusersBackend)


@pytest.mark.parametrize("engine", ["torch", "vllm", "gbquant", "transformers"])
def test_select_backend_routes_to_torch_when_engine_available(monkeypatch, engine):
    """"torch"/"vllm"/"gbquant"/"transformers" all route to
    SynapseTorchBackend now that it's the default (Phase 4 flip,
    2026-07-16) — as long as the torch engine venv is actually present."""
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: Path("/fake/torch-env"))
    backend = gsb.select_backend(_FakeEntry(engine))
    assert isinstance(backend, gsb.SynapseTorchBackend)


def test_select_backend_falls_back_to_vllm_when_torch_engine_unavailable(monkeypatch):
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: None)
    monkeypatch.setattr(gsb, "_find_vllm_bin", lambda: "/fake/vllm")
    backend = gsb.select_backend(_FakeEntry("torch"))
    assert isinstance(backend, gsb.VllmBackend)


def test_select_backend_falls_back_to_transformers_when_neither_available(monkeypatch):
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: None)
    monkeypatch.setattr(gsb, "_find_vllm_bin", lambda: None)
    backend = gsb.select_backend(_FakeEntry("vllm"))
    assert isinstance(backend, gsb.TransformersBackend)


def test_select_backend_gbquant_legacy_routes_to_torch(monkeypatch):
    """Legacy manifests carry engine=="gbquant" (pre-taxonomy) — select_backend
    must route it identically to "torch", not just gb_synapse._load_manifest's
    on-read normalization (belt-and-braces for any caller that constructs a
    ModelEntry directly without going through the manifest)."""
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: Path("/fake/torch-env"))
    backend = gsb.select_backend(_FakeEntry("gbquant"))
    assert isinstance(backend, gsb.SynapseTorchBackend)


def test_select_backend_force_vllm_escape_hatch(monkeypatch):
    """GB_SYNAPSE_FORCE_VLLM=1 + a real vllm binary present bypasses the
    torch-engine default entirely — a soak-window escape hatch, dies in
    Phase 6."""
    monkeypatch.setenv("GB_SYNAPSE_FORCE_VLLM", "1")
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: Path("/fake/torch-env"))
    monkeypatch.setattr(gsb, "_find_vllm_bin", lambda: "/fake/vllm")
    backend = gsb.select_backend(_FakeEntry("torch"))
    assert isinstance(backend, gsb.VllmBackend)


def test_select_backend_force_vllm_ignored_without_vllm_binary(monkeypatch):
    """GB_SYNAPSE_FORCE_VLLM=1 with no real vllm binary must not crash or
    silently serve nothing — falls through to the torch engine."""
    monkeypatch.setenv("GB_SYNAPSE_FORCE_VLLM", "1")
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: Path("/fake/torch-env"))
    monkeypatch.setattr(gsb, "_find_vllm_bin", lambda: None)
    backend = gsb.select_backend(_FakeEntry("torch"))
    assert isinstance(backend, gsb.SynapseTorchBackend)


# ── effective_vram_budget_mb: %-derived T2 sizing ─────────────────────────

def test_effective_budget_zero_fraction_is_shimless(monkeypatch):
    """GB_SYNAPSE_T2_FRACTION=0 must reproduce pre-shim (shimless) sizing
    exactly — the negative control in the T2 validation matrix."""
    monkeypatch.setenv("GB_SYNAPSE_T2_FRACTION", "0")

    class _FakeNvml:
        def mem(self):
            return (0, 8000.0, 12000.0, 0)

    monkeypatch.setitem(sys.modules, "gb_nvml", type(sys)("gb_nvml"))
    sys.modules["gb_nvml"].get_nvml = lambda *_a, **_k: _FakeNvml()

    real_read_text = Path.read_text

    def _fake_read_text(self, *a, **k):
        if str(self) == "/sys/class/greenboost/greenboost/pool_brief":
            return "T1:11GB T2:10/40GB(25%) T3:0/73GB\n"
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", _fake_read_text)

    host_free_mb, effective_free_mb, facts = gsb.effective_vram_budget_mb()
    assert host_free_mb == 8000.0
    assert effective_free_mb == host_free_mb   # T2 contributes nothing at frac=0
    assert facts["t2_fraction"] == 0.0
    assert facts["t2_free_mb"] == pytest.approx(30 * 1024)   # (40-10) GB free


def test_effective_budget_default_fraction_adds_half_t2_free(monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_T2_FRACTION", "0.5")

    class _FakeNvml:
        def mem(self):
            return (0, 8000.0, 12000.0, 0)

    monkeypatch.setitem(sys.modules, "gb_nvml", type(sys)("gb_nvml"))
    sys.modules["gb_nvml"].get_nvml = lambda *_a, **_k: _FakeNvml()

    real_read_text = Path.read_text

    def _fake_read_text(self, *a, **k):
        if str(self) == "/sys/class/greenboost/greenboost/pool_brief":
            return "T1:11GB T2:10/40GB(25%) T3:0/73GB\n"
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", _fake_read_text)

    host_free_mb, effective_free_mb, facts = gsb.effective_vram_budget_mb()
    t2_free_mb = 30 * 1024
    assert effective_free_mb == pytest.approx(host_free_mb + t2_free_mb * 0.5)


# ── _torch_env_dir: synapse torch engine venv search order ───────────────

def test_torch_env_dir_env_override_wins(tmp_path, monkeypatch):
    venv = tmp_path / "custom-torch-env"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("")
    monkeypatch.setenv("GB_SYNAPSE_TORCH_ENV", str(venv))
    assert gsb._torch_env_dir() == venv


def test_torch_env_dir_user_home_before_system(tmp_path, monkeypatch):
    """Env override unset, but a valid user-home venv exists — it must win
    over whatever the system path resolves to (real or absent), since the
    search order checks it first."""
    monkeypatch.delenv("GB_SYNAPSE_TORCH_ENV", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    user_venv = tmp_path / ".local/share/greenboost/synapse/torch-env"
    (user_venv / "bin").mkdir(parents=True)
    (user_venv / "bin" / "python").write_text("")
    assert gsb._torch_env_dir() == user_venv


def test_torch_env_dir_env_override_takes_priority_over_user_home(tmp_path, monkeypatch):
    override_venv = tmp_path / "override"
    (override_venv / "bin").mkdir(parents=True)
    (override_venv / "bin" / "python").write_text("")
    monkeypatch.setenv("GB_SYNAPSE_TORCH_ENV", str(override_venv))

    user_venv = tmp_path / ".local/share/greenboost/synapse/torch-env"
    (user_venv / "bin").mkdir(parents=True)
    (user_venv / "bin" / "python").write_text("")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert gsb._torch_env_dir() == override_venv


def test_find_torch_venv_lib_none_when_env_override_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_TORCH_ENV", str(tmp_path / "nonexistent"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "no-home-venv-here")
    if not Path("/usr/local/lib/greenboost/synapse-torch-env/bin/python").exists():
        assert gsb._find_torch_venv_lib("nvidia/cu12/lib/libcudart.so.12") is None


def test_cuda_home_for_vllm_is_alias_of_torch():
    assert gsb._cuda_home_for_vllm is gsb._cuda_home_for_torch


# ── quant_below_fp8_floor ─────────────────────────────────────────────────

class _FakeQuantEntry:
    def __init__(self, quant="", quant_method="", quant_bits=0):
        self.quant = quant
        self.quant_method = quant_method
        self.quant_bits = quant_bits


@pytest.mark.parametrize("method,bits,expected", [
    ("gptq", 4, True), ("gptq", 8, False),
    ("awq", 4, True), ("awq", 8, False),
    ("compressed-tensors", 4, True), ("compressed-tensors", 8, False),
    ("bitsandbytes", 4, True), ("bitsandbytes", 8, True),
    ("fp8", 8, False),
])
def test_quant_below_fp8_floor_checkpoint_truth(method, bits, expected):
    entry = _FakeQuantEntry(quant_method=method, quant_bits=bits)
    assert gsb._quant_below_fp8_floor(entry) is expected


def test_quant_below_fp8_floor_legacy_token_fallback():
    assert gsb._quant_below_fp8_floor(_FakeQuantEntry(quant="INT4")) is True
    assert gsb._quant_below_fp8_floor(_FakeQuantEntry(quant="FP8")) is False


# ── SynapseTorchBackend.serve() ───────────────────────────────────────────

class _FakeSynapseEntry:
    def __init__(self, name="fake-model", quant="BF16", quant_method="",
                quant_bits=0, engine="torch"):
        self.name = name
        self.path = ""
        self.repo = "org/fake-model"
        self.quant = quant
        self.quant_method = quant_method
        self.quant_bits = quant_bits
        self.engine = engine
        self.arch = ""
        self.source = "hf"


def _patch_torch_serve_common(monkeypatch, tmp_path, mode="gllm", reason=""):
    venv = tmp_path / "torch-env"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("")
    monkeypatch.setenv("GB_SYNAPSE_TORCH_ENV", str(venv))
    # _torch_serve_mode's own subprocess probe would fail against this fake
    # venv (bin/python is just an empty stub, not a real interpreter) —
    # short-circuit it so these tests exercise serve()'s OWN logic, not
    # the trigger logic (that's test_torch_serve_mode_*'s job).
    monkeypatch.setattr(gsb, "_torch_serve_mode", lambda entry: (mode, reason))

    import gb_synapse as gs
    import gb_cluster

    monkeypatch.setattr(gs, "hf_token", lambda: None)
    monkeypatch.setattr(gs, "_run_log_path", lambda name: tmp_path / f"{name}.log")

    captured = {}

    def _fake_launch(entry, proc, port, internal_port, engine="", **facts):
        captured.update(entry=entry, proc=proc, port=port,
                        internal_port=internal_port, engine=engine, facts=facts)
        return "sentinel-state"

    monkeypatch.setattr(gs, "_launch_proxy_and_record", _fake_launch)
    monkeypatch.setattr(gb_cluster, "shim_env",
                        lambda workload, enabled, base_env=None, cudart_path=None:
                        {"GREENBOOST_ACTIVE": "1"} if enabled else {})

    popen_calls = []

    class _FakeProc:
        pass

    def _fake_popen(cmd, env=None, stdout=None, stderr=None, start_new_session=None):
        popen_calls.append({"cmd": cmd, "env": env})
        return _FakeProc()

    monkeypatch.setattr(gsb.subprocess, "Popen", _fake_popen)
    return popen_calls, captured, venv


def test_synapse_torch_backend_serve_full_cmd(tmp_path, monkeypatch):
    popen_calls, captured, venv = _patch_torch_serve_common(monkeypatch, tmp_path)
    entry = _FakeSynapseEntry()

    backend = gsb.SynapseTorchBackend()
    result = backend.serve(entry, port=11435, ctx=8192)

    assert result == "sentinel-state"
    cmd = popen_calls[0]["cmd"]
    assert cmd[0] == str(venv / "bin" / "python")
    assert "-m" in cmd and "gllm.entrypoints.api_server" in cmd
    assert "--model-path" in cmd
    assert cmd[cmd.index("--model-path") + 1] == "org/fake-model"
    assert cmd[cmd.index("--port") + 1] == "12435"
    assert "--model-max-length" in cmd
    assert cmd[cmd.index("--model-max-length") + 1] == "8192"
    assert "--disable-cuda-graph" in cmd
    assert captured["engine"] == "torch"


def test_synapse_torch_backend_serve_util_clamped(tmp_path, monkeypatch):
    popen_calls, captured, venv = _patch_torch_serve_common(monkeypatch, tmp_path)
    monkeypatch.setenv("GB_SYNAPSE_TORCH_KV_UTIL", "2.0")  # way over 0.95
    entry = _FakeSynapseEntry()

    gsb.SynapseTorchBackend().serve(entry, port=11435)

    cmd = popen_calls[0]["cmd"]
    util_str = cmd[cmd.index("--gpu-memory-util") + 1]
    assert float(util_str) == pytest.approx(0.95)


def test_synapse_torch_backend_serve_defer_init_env(tmp_path, monkeypatch):
    popen_calls, captured, venv = _patch_torch_serve_common(monkeypatch, tmp_path)
    entry = _FakeSynapseEntry()

    gsb.SynapseTorchBackend().serve(entry, port=11435)

    env = popen_calls[0]["env"]
    assert env.get("GREENBOOST_DEFER_INIT") == "1"


def test_synapse_torch_backend_serve_cuda_graph_opt_in(tmp_path, monkeypatch):
    popen_calls, captured, venv = _patch_torch_serve_common(monkeypatch, tmp_path)
    monkeypatch.setenv("GB_SYNAPSE_TORCH_CUDA_GRAPH", "1")
    entry = _FakeSynapseEntry()

    gsb.SynapseTorchBackend().serve(entry, port=11435)

    cmd = popen_calls[0]["cmd"]
    assert "--disable-cuda-graph" not in cmd


def test_synapse_torch_backend_serve_never_passes_quant_flag_for_quantized(tmp_path, monkeypatch):
    """A checkpoint whose OWN quant_method is already set must never be
    re-quantized by the engine invocation — no requantization flag of any
    kind should appear in the launched command."""
    popen_calls, captured, venv = _patch_torch_serve_common(monkeypatch, tmp_path)
    entry = _FakeSynapseEntry(quant="GPTQ4", quant_method="gptq", quant_bits=4)

    gsb.SynapseTorchBackend().serve(entry, port=11435)

    cmd = popen_calls[0]["cmd"]
    joined = " ".join(cmd).lower()
    assert "quant" not in joined  # no --quantization/--quant-* flag at all
    assert captured["facts"]["quant_method"] == "gptq"
    assert captured["facts"]["quant_below_floor"] is True


def test_synapse_torch_backend_available_false_without_venv(tmp_path, monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_TORCH_ENV", str(tmp_path / "nonexistent"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "no-home-venv")
    if not Path("/usr/local/lib/greenboost/synapse-torch-env/bin/python").exists():
        assert gsb.SynapseTorchBackend().available() is False


def test_synapse_torch_backend_can_serve_legacy_engines():
    backend = gsb.SynapseTorchBackend()
    for engine in ("torch", "vllm", "transformers", "gbquant"):
        assert backend.can_serve(_FakeSynapseEntry(engine=engine))
    assert not backend.can_serve(_FakeSynapseEntry(engine="llama.cpp"))
    assert not backend.can_serve(_FakeSynapseEntry(engine="diffusers"))


def test_synapse_torch_backend_serve_fallback_mode_cmd(tmp_path, monkeypatch):
    popen_calls, captured, venv = _patch_torch_serve_common(
        monkeypatch, tmp_path, mode="fallback", reason="test reason")
    entry = _FakeSynapseEntry(quant="FP8")

    result = gsb.SynapseTorchBackend().serve(entry, port=11435)

    assert result == "sentinel-state"
    cmd = popen_calls[0]["cmd"]
    assert cmd[0] == str(venv / "bin" / "python")
    assert cmd[1] == str(gsb._REPO_DIR / "gb_synapse_fallback.py")
    assert "--model" in cmd
    assert cmd[cmd.index("--served-model-name") + 1] == "fake-model"
    assert cmd[cmd.index("--quant") + 1] == "FP8"
    assert cmd[cmd.index("--port") + 1] == "12435"
    # gllm-only flags must never appear in the fallback command.
    assert "--gpu-memory-util" not in cmd
    assert "--disable-cuda-graph" not in cmd
    assert captured["facts"]["degraded"] == "fallback-single-request"


# ── _torch_serve_mode: fallback trigger logic (pure-function cases) ──────

def test_torch_serve_mode_fallback_when_engine_not_importable(monkeypatch):
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: Path("/fake/torch-env"))
    monkeypatch.setattr(gsb, "_torch_engine_importable", lambda venv_py: False)
    mode, reason = gsb._torch_serve_mode(_FakeSynapseEntry())
    assert mode == "fallback"
    assert "import-broken" in reason or "missing" in reason


def test_torch_serve_mode_fallback_when_no_venv(monkeypatch):
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: None)
    mode, reason = gsb._torch_serve_mode(_FakeSynapseEntry())
    assert mode == "fallback"


def test_torch_serve_mode_fallback_when_arch_unsupported(monkeypatch):
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: Path("/fake/torch-env"))
    monkeypatch.setattr(gsb, "_torch_engine_importable", lambda venv_py: True)
    import gb_synapse as gs
    monkeypatch.setattr(gs, "_torch_engine_supported_archs",
                        lambda: {"Qwen3ForCausalLM"})
    entry = _FakeSynapseEntry()
    entry.arch = "SomeUnknownArchForCausalLM"
    mode, reason = gsb._torch_serve_mode(entry)
    assert mode == "fallback"
    assert "SomeUnknownArchForCausalLM" in reason


def test_torch_serve_mode_gllm_when_arch_supported(monkeypatch):
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: Path("/fake/torch-env"))
    monkeypatch.setattr(gsb, "_torch_engine_importable", lambda venv_py: True)
    import gb_synapse as gs
    monkeypatch.setattr(gs, "_torch_engine_supported_archs",
                        lambda: {"Qwen3ForCausalLM"})
    entry = _FakeSynapseEntry(quant="BF16")
    entry.arch = "Qwen3ForCausalLM"
    entry.n_bytes = 1  # trivially fits any budget
    mode, reason = gsb._torch_serve_mode(entry)
    assert mode == "gllm"
    assert reason == ""


def test_torch_serve_mode_fallback_when_bf16_exceeds_budget(monkeypatch):
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: Path("/fake/torch-env"))
    monkeypatch.setattr(gsb, "_torch_engine_importable", lambda venv_py: True)
    import gb_synapse as gs
    monkeypatch.setattr(gs, "_torch_engine_supported_archs", lambda: set())
    monkeypatch.setattr(gsb, "effective_vram_budget_mb", lambda: (1000.0, 1000.0, {}))
    entry = _FakeSynapseEntry(quant="BF16")
    entry.n_bytes = 999 * (1024 ** 3)  # 999 GB — way over any real budget
    mode, reason = gsb._torch_serve_mode(entry)
    assert mode == "fallback"
    assert "budget" in reason


def test_torch_serve_mode_fallback_when_requantize_token_requested(monkeypatch):
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: Path("/fake/torch-env"))
    monkeypatch.setattr(gsb, "_torch_engine_importable", lambda venv_py: True)
    import gb_synapse as gs
    monkeypatch.setattr(gs, "_torch_engine_supported_archs", lambda: set())
    monkeypatch.setattr(gsb, "effective_vram_budget_mb", lambda: (1e9, 1e9, {}))
    entry = _FakeSynapseEntry(quant="INT4")  # requested requantization token
    entry.n_bytes = 1
    mode, reason = gsb._torch_serve_mode(entry)
    assert mode == "fallback"
    assert "requantization" in reason


def test_torch_serve_mode_gllm_when_already_quantized_even_if_over_naive_budget(monkeypatch):
    """A checkpoint whose own quant_method is set is never routed to the
    fallback for budget/token reasons — its n_bytes already reflects the
    quantized size, and there's nothing to re-quantize."""
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: Path("/fake/torch-env"))
    monkeypatch.setattr(gsb, "_torch_engine_importable", lambda venv_py: True)
    import gb_synapse as gs
    monkeypatch.setattr(gs, "_torch_engine_supported_archs", lambda: set())
    monkeypatch.setattr(gsb, "effective_vram_budget_mb", lambda: (1.0, 1.0, {}))
    entry = _FakeSynapseEntry(quant="GPTQ4", quant_method="gptq", quant_bits=4)
    entry.n_bytes = 999 * (1024 ** 3)
    mode, reason = gsb._torch_serve_mode(entry)
    assert mode == "gllm"


def test_torch_engine_importable_caches_result(monkeypatch, tmp_path):
    gsb._TORCH_ENGINE_IMPORTABLE_CACHE.clear()
    calls = []

    def _fake_run(cmd, capture_output=None, timeout=None):
        calls.append(cmd)
        class _R:
            returncode = 0
        return _R()

    monkeypatch.setattr(gsb.subprocess, "run", _fake_run)
    venv_py = str(tmp_path / "python")
    assert gsb._torch_engine_importable(venv_py) is True
    assert gsb._torch_engine_importable(venv_py) is True
    assert len(calls) == 1  # second call served from cache, no new subprocess


def test_torch_engine_importable_false_on_exception(monkeypatch, tmp_path):
    gsb._TORCH_ENGINE_IMPORTABLE_CACHE.clear()

    def _fake_run(cmd, capture_output=None, timeout=None):
        raise FileNotFoundError("no such interpreter")

    monkeypatch.setattr(gsb.subprocess, "run", _fake_run)
    assert gsb._torch_engine_importable(str(tmp_path / "nonexistent-python")) is False

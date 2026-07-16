#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for gb_synapse.pull()'s routing matrix + the "never re-quantize an
already-quantized checkpoint" rule in _pull_torch (P1.6 of the gb-synapse
unification).

CPU-only. No real HF network calls — snapshot_download/list_repo_gguf/
_find_vllm_bin are all monkeypatched."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gb_synapse as gs
import gb_synapse_backends as gsb


def test_gguf_repo_routes_to_llamacpp(monkeypatch, tmp_path):
    """A repo with a real GGUF release never touches _pull_torch/
    _pull_diffusers — it downloads the GGUF directly and keeps the
    default "llama.cpp" engine."""
    monkeypatch.setattr(gs, "list_repo_gguf",
                        lambda repo: [{"filename": "model-Q4_K_M.gguf", "size": 1}])
    monkeypatch.setattr(gs, "_pull_torch",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("should not route to _pull_torch")))
    monkeypatch.setattr(gs, "_pull_diffusers",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("should not route to _pull_diffusers")))
    monkeypatch.setattr(gs, "doctor", lambda probe_feeders=True: {"aggregate_vram_mb": 999999})

    gguf_path = tmp_path / "model-Q4_K_M.gguf"
    gguf_path.write_bytes(b"")
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo_id, filename, token=None, cache_dir=None: str(gguf_path))
    monkeypatch.setattr(gs, "gguf_summary", lambda path: {"n_bytes": 1, "n_layers": 1,
                                                          "is_moe": False, "n_experts": 0,
                                                          "n_experts_used": 0,
                                                          "dense_bytes": 0, "expert_bytes": 0,
                                                          "ctx_length": 0, "quant": "Q4_K_M",
                                                          "arch": "", "n_kv_heads": 0,
                                                          "head_dim": 0})
    monkeypatch.setattr(gs, "_load_manifest", lambda: {})
    monkeypatch.setattr(gs, "_save_manifest", lambda m: None)

    entry = gs.pull("org/gguf-repo")
    assert entry.engine == "llama.cpp"


def test_fp8_token_routes_to_torch(monkeypatch):
    called = {}
    monkeypatch.setattr(gs, "_pull_torch",
                        lambda repo, quant, name, engine="torch": called.update(
                            repo=repo, quant=quant, engine=engine) or "sentinel")
    result = gs.pull("org/repo:FP8")
    assert result == "sentinel"
    assert called["engine"] == "torch"
    assert called["quant"] == "FP8"


def test_explicit_engine_vllm_maps_to_torch_with_deprecation_note(monkeypatch, capsys):
    called = {}
    monkeypatch.setattr(gs, "_pull_torch",
                        lambda repo, quant, name, engine="torch": called.update(
                            engine=engine) or "sentinel")
    result = gs.pull("org/repo", engine="vllm")
    assert result == "sentinel"
    assert called["engine"] == "torch"
    out = capsys.readouterr().out
    assert "deprecated" in out
    assert "torch" in out


def test_explicit_engine_transformers_maps_to_torch_with_deprecation_note(monkeypatch, capsys):
    called = {}
    monkeypatch.setattr(gs, "_pull_torch",
                        lambda repo, quant, name, engine="torch": called.update(
                            engine=engine) or "sentinel")
    gs.pull("org/repo", engine="transformers")
    assert called["engine"] == "torch"
    assert "deprecated" in capsys.readouterr().out


def test_engine_torch_explicit_routes_directly(monkeypatch):
    called = {}
    monkeypatch.setattr(gs, "_pull_torch",
                        lambda repo, quant, name, engine="torch": called.update(
                            engine=engine) or "sentinel")
    gs.pull("org/repo", engine="torch")
    assert called["engine"] == "torch"


def test_engine_diffusers_routes_directly(monkeypatch):
    called = {}
    monkeypatch.setattr(gs, "_pull_diffusers",
                        lambda repo, name: called.update(repo=repo) or "sentinel")
    result = gs.pull("org/repo", engine="diffusers")
    assert result == "sentinel"
    assert called["repo"] == "org/repo"


def test_token_less_safetensors_routes_to_torch_when_torch_engine_available(monkeypatch):
    monkeypatch.setattr(gs, "list_repo_gguf", lambda repo: [])
    monkeypatch.setattr(gsb, "detect_format", lambda repo: "safetensors")
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: Path("/fake/torch-env"))
    called = {}
    monkeypatch.setattr(gs, "_pull_torch",
                        lambda repo, quant, name=None: called.update(
                            repo=repo, quant=quant) or "sentinel")
    result = gs.pull("org/bare-safetensors-repo")
    assert result == "sentinel"
    assert called["quant"] == "fp8"


def test_token_less_safetensors_falls_back_to_gguf_convert_without_torch_engine(monkeypatch):
    monkeypatch.setattr(gs, "list_repo_gguf", lambda repo: [])
    monkeypatch.setattr(gsb, "detect_format", lambda repo: "safetensors")
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: None)
    called = {}
    monkeypatch.setattr(gs, "_pull_and_convert",
                        lambda repo, quant, name: called.update(repo=repo) or "sentinel")
    result = gs.pull("org/bare-safetensors-repo")
    assert result == "sentinel"
    assert called["repo"] == "org/bare-safetensors-repo"


# ── _pull_torch: never re-quantize checkpoint truth ───────────────────────

class _FakeHfHub:
    """Stand-in module object for the local `from huggingface_hub import
    snapshot_download` — monkeypatches the real module attribute so the
    function-local import picks it up."""


def _fake_snapshot_download(local_dir):
    def _f(repo_id, token=None, cache_dir=None, allow_patterns=None,
          ignore_patterns=None):
        return local_dir
    return _f


def test_pull_torch_gptq_checkpoint_ignores_conflicting_token(monkeypatch, tmp_path, capsys):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "quantization_config": {"quant_method": "gptq", "bits": 4}}))
    (tmp_path / "model.safetensors").write_bytes(b"x" * 10)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        _fake_snapshot_download(str(tmp_path)))
    monkeypatch.setattr(gs, "_load_manifest", lambda: {})
    monkeypatch.setattr(gs, "_save_manifest", lambda m: None)

    entry = gs._pull_torch("org/gptq-repo", "INT4", None)
    assert entry.quant == "GPTQ4"
    assert entry.quant_method == "gptq"
    assert entry.quant_bits == 4
    out = capsys.readouterr().out
    assert "already" in out
    assert "gptq" in out


def test_pull_torch_plain_checkpoint_uses_requested_token(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (tmp_path / "model.safetensors").write_bytes(b"x" * 10)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        _fake_snapshot_download(str(tmp_path)))
    monkeypatch.setattr(gs, "_load_manifest", lambda: {})
    monkeypatch.setattr(gs, "_save_manifest", lambda m: None)

    entry = gs._pull_torch("org/plain-repo", "fp8", None)
    assert entry.quant == "FP8"
    assert entry.quant_method == ""


def test_pull_torch_plain_checkpoint_no_token_defaults_bf16(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (tmp_path / "model.safetensors").write_bytes(b"x" * 10)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        _fake_snapshot_download(str(tmp_path)))
    monkeypatch.setattr(gs, "_load_manifest", lambda: {})
    monkeypatch.setattr(gs, "_save_manifest", lambda m: None)

    entry = gs._pull_torch("org/plain-repo", "", None)
    assert entry.quant == "BF16"

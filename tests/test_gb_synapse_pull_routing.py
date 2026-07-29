#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for gb_synapse.pull()'s routing matrix + the "never re-quantize an
already-quantized checkpoint" rule in _pull_torch (P1.6 of the gb-synapse
unification).

CPU-only. No real HF network calls — snapshot_download/list_repo_gguf/
_torch_env_dir are all monkeypatched."""
import json
import sys
from pathlib import Path

import pytest

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
    # _check_torch_engine_capability's pre-download gate (2026-07-28) calls
    # read_quant_config(repo) BEFORE snapshot_download — for a bare repo id
    # that resolves to hf_hub_download(filename="config.json"), which this
    # test previously left unmocked (only snapshot_download was), so it made
    # a real, unmocked HF Hub network call. Point it at the same local
    # config.json the rest of this test already sets up.
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo_id, filename, token=None, cache_dir=None:
                        str(tmp_path / filename))
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
    # See test_pull_torch_gptq_checkpoint_ignores_conflicting_token's comment:
    # the pre-download capability gate needs hf_hub_download mocked too, not
    # just snapshot_download, to stay hermetic.
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo_id, filename, token=None, cache_dir=None:
                        str(tmp_path / filename))
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
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo_id, filename, token=None, cache_dir=None:
                        str(tmp_path / filename))
    monkeypatch.setattr(gs, "_load_manifest", lambda: {})
    monkeypatch.setattr(gs, "_save_manifest", lambda m: None)

    entry = gs._pull_torch("org/plain-repo", "", None)
    assert entry.quant == "BF16"


# ---------------------------------------------------------------------------
# _manifest_lookup / _resolve_model — repo-aware, offline resolution
#
# Real incident, 2026-07-28: every HF-pulled manifest entry is keyed by the
# repo's BARE name (`name or repo.split("/")[-1]`), never by "org/name", but
# _resolve_model only ever checked `spec in manifest` — so typing a model's
# real "org/repo" spelling (exactly what `pull()`/`list`/the model card show)
# missed the manifest entirely and fell into the "/" in spec branch, which
# calls pull() — a network operation — for a model already on disk. A model
# pulled and served OK 18 times failed with a bare
# "No module named 'huggingface_hub'" the moment its org prefix was typed.
# ---------------------------------------------------------------------------

def _entry(name, repo="", quant="", **kw):
    return gs.ModelEntry(name=name, path=f"/fake/{name}", source="hf",
                         repo=repo, quant=quant, engine="llama.cpp", **kw)


def test_manifest_lookup_finds_entry_by_repo_field(monkeypatch):
    manifest = {
        "Qwen3.6-27B-Fable-Fusion-711-GGUF": _entry(
            "Qwen3.6-27B-Fable-Fusion-711-GGUF",
            repo="DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF",
            quant="LOW-MTP-IQ4_XS"),
    }
    hit = gs._manifest_lookup(
        "DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF",
        manifest)
    assert hit is not None
    assert hit.name == "Qwen3.6-27B-Fable-Fusion-711-GGUF"


def test_manifest_lookup_is_case_insensitive(monkeypatch):
    manifest = {"Foo-GGUF": _entry("Foo-GGUF", repo="Org/Foo-GGUF")}
    assert gs._manifest_lookup("org/foo-gguf", manifest) is not None
    assert gs._manifest_lookup("FOO-GGUF", manifest) is not None


def test_manifest_lookup_repo_quant_spec_matches_single_candidate():
    manifest = {"Foo-GGUF": _entry("Foo-GGUF", repo="org/Foo-GGUF", quant="Q4_K_M")}
    hit = gs._manifest_lookup("org/Foo-GGUF:Q4_K_M", manifest)
    assert hit is not None and hit.name == "Foo-GGUF"


def test_manifest_lookup_ambiguous_quant_raises_value_error():
    manifest = {
        "variant-a": _entry("variant-a", repo="org/Foo-GGUF", quant="MTP-Q4_K_M"),
        "variant-b": _entry("variant-b", repo="org/Foo-GGUF", quant="Q4_K_M"),
    }
    with pytest.raises(ValueError, match="matches more than one manifest entry"):
        gs._manifest_lookup("org/Foo-GGUF:Q4_K_M", manifest)


def test_manifest_lookup_no_match_returns_none():
    manifest = {"Foo-GGUF": _entry("Foo-GGUF", repo="org/Foo-GGUF")}
    assert gs._manifest_lookup("org/Bar-GGUF", manifest) is None


def test_resolve_model_org_prefixed_spec_resolves_offline(monkeypatch):
    """The exact reported failure: an already-pulled model's real
    "org/repo" spelling must resolve from the manifest ALONE — with
    huggingface_hub import forcibly broken, proving no network path is
    touched."""
    entry = _entry("Qwen3.6-27B-Fable-Fusion-711-GGUF",
                   repo="DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF",
                   quant="LOW-MTP-IQ4_XS")
    monkeypatch.setattr(gs, "_load_manifest", lambda: {entry.name: entry})

    def _boom():
        raise AssertionError("must not import huggingface_hub for an already-pulled model")
    monkeypatch.setattr(gs, "_require_huggingface_hub", _boom)

    hit = gs._resolve_model(
        "DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF")
    assert hit.name == entry.name


def test_resolve_model_unknown_bare_name_suggests_org_prefix(monkeypatch):
    monkeypatch.setattr(gs, "_load_manifest", lambda: {})
    monkeypatch.setattr(gs, "ollama_model_blob", lambda spec: None)
    with pytest.raises(KeyError, match="include the org prefix"):
        gs._resolve_model("Some-Model-Nobody-Has")


def test_resolve_model_unknown_name_hints_similar_entry(monkeypatch):
    entry = _entry("Qwen3.6-27B-Fable-Fusion-711-GGUF",
                   repo="DavidAU/Qwen3.6-27B-Fable-Fusion-711-GGUF")
    monkeypatch.setattr(gs, "_load_manifest", lambda: {entry.name: entry})
    monkeypatch.setattr(gs, "ollama_model_blob", lambda spec: None)
    with pytest.raises(KeyError, match="did you mean"):
        gs._resolve_model("Qwen3.6-27B-Fable-Fusion-711")


# ---------------------------------------------------------------------------
# _check_torch_engine_capability — refuse before download, not after
#
# gLLM's own layer dispatch (gllm/layers/linear.py) only has real code paths
# for quant_method in {None, "fp8", "gptq", "awq"} — anything else (verified
# live against Brian6145/Qwen3.6-27B-Claude-Opus-Sonnet-Distilled-NVFP4-MTP's
# raw config.json: quantization_config.quant_method == "compressed-tensors")
# hits gLLM's own `raise Exception("gLLM do not support quant_method ...")`.
# These tests point `read_quant_config` at a real local directory (its
# `Path(path_or_repo).is_dir()` branch) so nothing here touches the network.
# ---------------------------------------------------------------------------

def _write_config(tmp_path, quantization_config):
    (tmp_path / "config.json").write_text(json.dumps(
        {"model_type": "qwen3", "quantization_config": quantization_config}))
    return str(tmp_path)


def test_capability_gate_refuses_nvfp4_compressed_tensors(monkeypatch, tmp_path):
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: Path("/fake/torch-env"))
    repo_dir = _write_config(tmp_path, {
        "quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
        "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 4}}}})
    with pytest.raises(RuntimeError, match="nvfp4"):
        gs._check_torch_engine_capability(repo_dir, "torch")


def test_capability_gate_allows_real_fp8_compressed_tensors(monkeypatch, tmp_path):
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: Path("/fake/torch-env"))
    repo_dir = _write_config(tmp_path, {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 8}}}})
    gs._check_torch_engine_capability(repo_dir, "torch")  # must not raise


def test_capability_gate_allows_gptq(monkeypatch, tmp_path):
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: Path("/fake/torch-env"))
    repo_dir = _write_config(tmp_path, {"quant_method": "gptq", "bits": 4})
    gs._check_torch_engine_capability(repo_dir, "torch")  # must not raise


def test_capability_gate_skipped_when_torch_engine_not_installed(monkeypatch, tmp_path):
    """No gLLM installed -> select_backend would fall back to
    TransformersBackend, which may genuinely load compressed-tensors/nvfp4
    via `transformers`+`compressed_tensors` — untested, out of scope, so the
    gate must not block that fallback path."""
    monkeypatch.setattr(gsb, "_torch_env_dir", lambda: None)
    repo_dir = _write_config(tmp_path, {
        "quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
        "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 4}}}})
    gs._check_torch_engine_capability(repo_dir, "torch")  # must not raise


def test_read_quant_config_distinguishes_nvfp4_from_real_fp8(tmp_path):
    """Both nvfp4 and real fp8 compressed-tensors exports carry
    weights.type == "float" — only num_bits (4 vs 8) tells them apart. Before
    this fix, any float-typed group was bucketed as "fp8", which would have
    defeated the capability gate for exactly the nvfp4 checkpoint it exists
    to catch (confirmed live, 2026-07-28)."""
    nvfp4_dir = tmp_path / "nvfp4"
    nvfp4_dir.mkdir()
    _write_config(nvfp4_dir, {
        "quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
        "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 4}}}})
    assert gs.read_quant_config(str(nvfp4_dir)) == {"quant_method": "nvfp4", "quant_bits": 4}

    fp8_dir = tmp_path / "fp8"
    fp8_dir.mkdir()
    _write_config(fp8_dir, {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 8}}}})
    assert gs.read_quant_config(str(fp8_dir)) == {"quant_method": "fp8", "quant_bits": 8}

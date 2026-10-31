import json
from types import SimpleNamespace
from greenboost_cli.terminal.commands import _unregistered_hf_models


def _snap(root, repo, files, cfg=None):
    d = root / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / "abc"
    d.mkdir(parents=True)
    for f in files:
        (d / f).write_text("x")
    if cfg is not None:
        (d / "config.json").write_text(json.dumps(cfg))


def test_downloaded_mlx_model_is_reported_as_not_servable(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _snap(tmp_path, "Org/Qwen-MLX-APEX", ["model.safetensors", "chat_template.jinja"],
          {"architectures": ["Qwen3_5MoeForConditionalGeneration"]})
    _snap(tmp_path, "Org/Some-Embedding-0.6B", ["model.safetensors"], {"architectures": ["Qwen3ForCausalLM"]})
    out = dict(_unregistered_hf_models([]))
    assert "Apple Silicon only" in out["Org/Qwen-MLX-APEX"]
    assert "Org/Some-Embedding-0.6B" not in out


def test_registered_and_unregistered_gguf(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    _snap(tmp_path, "Org/A-GGUF", ["a.gguf"])
    _snap(tmp_path, "Org/B-GGUF", ["b.gguf"])
    out = dict(_unregistered_hf_models([SimpleNamespace(repo="Org/A-GGUF")]))
    assert "Org/A-GGUF" not in out and "gb synapse pull Org/B-GGUF" in out["Org/B-GGUF"]

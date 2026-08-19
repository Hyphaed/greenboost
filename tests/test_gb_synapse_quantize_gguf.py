"""Tests for gb_synapse._quantize_gguf()'s per-tensor override wiring
(missing_features.md item (j)) — tensor_types/output_tensor_type/
allow_requantize/dry_run must all default to the function's PRIOR
behavior (a flat, whole-file `quant` string, no extra flags) so the
existing _pull_and_convert() call site is unaffected; each new flag must
turn into the exact llama-quantize CLI flag it wraps when set.

No real llama-quantize binary needed — gb_synapse._run is monkeypatched to
capture the constructed command instead of executing it.
"""
from pathlib import Path

import gb_synapse as gs


def _fake_binary(tmp_path, monkeypatch):
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    (engine_dir / "llama-quantize").write_text("#!/bin/sh\n")
    monkeypatch.setattr(gs, "ENGINE_DIR", engine_dir)
    return engine_dir


def test_quantize_gguf_default_behavior_unchanged(tmp_path, monkeypatch):
    _fake_binary(tmp_path, monkeypatch)
    captured = {}
    monkeypatch.setattr(gs, "_run", lambda cmd, capture=False, **kw: captured.setdefault("cmd", cmd))

    gs._quantize_gguf(Path("/src.gguf"), tmp_path / "out" / "dst.gguf", "Q4_K_M")

    binary = str(gs.ENGINE_DIR / "llama-quantize")
    assert captured["cmd"] == [binary, "/src.gguf", str(tmp_path / "out" / "dst.gguf"), "Q4_K_M"]


def test_quantize_gguf_tensor_types_adds_tensor_type_file_flag(tmp_path, monkeypatch):
    _fake_binary(tmp_path, monkeypatch)
    captured = {}
    monkeypatch.setattr(gs, "_run", lambda cmd, capture=False, **kw: captured.setdefault("cmd", cmd))
    types_file = tmp_path / "tensor_types.txt"

    gs._quantize_gguf(Path("/src.gguf"), tmp_path / "dst.gguf", "Q4_K_M",
                      tensor_types=types_file)

    assert "--tensor-type-file" in captured["cmd"]
    idx = captured["cmd"].index("--tensor-type-file")
    assert captured["cmd"][idx + 1] == str(types_file)


def test_quantize_gguf_allow_requantize_and_output_tensor_type(tmp_path, monkeypatch):
    _fake_binary(tmp_path, monkeypatch)
    captured = {}
    monkeypatch.setattr(gs, "_run", lambda cmd, capture=False, **kw: captured.setdefault("cmd", cmd))

    gs._quantize_gguf(Path("/src.gguf"), tmp_path / "dst.gguf", "Q4_K_M",
                      allow_requantize=True, output_tensor_type="f16")

    assert "--allow-requantize" in captured["cmd"]
    assert "--output-tensor-type" in captured["cmd"]
    idx = captured["cmd"].index("--output-tensor-type")
    assert captured["cmd"][idx + 1] == "f16"


def test_quantize_gguf_dry_run_passes_flag_and_captures_output(tmp_path, monkeypatch):
    _fake_binary(tmp_path, monkeypatch)
    captured = {}

    class _FakeCompleted:
        stdout = "would write 12345 bytes\n"

    def _fake_run(cmd, capture=False, **kw):
        captured["cmd"] = cmd
        captured["capture"] = capture
        return _FakeCompleted()

    monkeypatch.setattr(gs, "_run", _fake_run)

    result = gs._quantize_gguf(Path("/src.gguf"), tmp_path / "dst.gguf", "Q4_K_M", dry_run=True)

    assert "--dry-run" in captured["cmd"]
    assert captured["capture"] is True
    assert result.stdout == "would write 12345 bytes\n"


def test_quantize_gguf_no_binary_raises_with_build_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(gs, "ENGINE_DIR", tmp_path / "no-engine-here")

    try:
        gs._quantize_gguf(Path("/src.gguf"), tmp_path / "dst.gguf", "Q4_K_M")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "build-engine" in str(e)

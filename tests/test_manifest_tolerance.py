"""One damaged manifest entry must not destroy every other registration.

Found 2026-08-18. `_load_manifest()` wrapped its whole parse loop in a single
try/except that caught TypeError and returned `{}`, so one entry carrying a
field this build's ModelEntry did not recognise silently discarded EVERY model.
`list_models()` then merged in its Ollama re-scan and re-persisted, meaning a
plain READ permanently erased every HF-pulled model from disk.

Two models were lost this way in one session, one of them a 21.27 GiB download
registered minutes earlier. Nothing was logged; the manifest's mtime was the
only evidence a read had rewritten it.
"""
from __future__ import annotations

import json

import pytest

import gb_synapse as gs


def _entry(name: str, **extra) -> dict:
    return dict({
        "name": name, "path": f"/tmp/{name}.gguf", "source": "hf", "repo": "org/repo",
        "quant": "Q4_K_M", "quant_method": "", "quant_bits": 0, "arch": "qwen",
        "engine": "llama.cpp", "n_bytes": 1234,
    }, **extra)


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    f = tmp_path / "manifest.json"
    monkeypatch.setattr(gs, "MANIFEST_FILE", f)
    return f


def test_unknown_field_skips_only_that_entry(manifest, capsys):
    """The exact shape that lost a 21 GiB download."""
    manifest.write_text(json.dumps({
        "good_a": _entry("good_a"),
        "from_newer_build": _entry("from_newer_build", field_this_build_lacks=True),
        "good_b": _entry("good_b"),
    }))
    out = gs._load_manifest()
    assert set(out) == {"good_a", "good_b"}, "a bad entry took healthy ones with it"
    assert "from_newer_build" in capsys.readouterr().err, "the skip was silent"


def test_non_dict_entry_is_survivable(manifest):
    manifest.write_text(json.dumps({"good": _entry("good"), "junk": "not a dict"}))
    assert set(gs._load_manifest()) == {"good"}


def test_missing_required_field_skips_only_that_entry(manifest):
    bad = _entry("bad"); bad.pop("path")
    manifest.write_text(json.dumps({"good": _entry("good"), "bad": bad}))
    assert set(gs._load_manifest()) == {"good"}


def test_legacy_engine_values_still_normalise(manifest):
    """The pre-torch-core normalisation must survive the refactor."""
    manifest.write_text(json.dumps({
        "v": _entry("v", engine="vllm"),
        "g": _entry("g", engine="gbquant"),
        "t": _entry("t", engine="transformers"),
    }))
    out = gs._load_manifest()
    assert {e.engine for e in out.values()} == {"torch"}


def test_unreadable_and_malformed_files_still_return_empty(manifest):
    assert gs._load_manifest() == {}          # file does not exist
    manifest.write_text("{not json")
    assert gs._load_manifest() == {}
    manifest.write_text(json.dumps(["a", "list"]))
    assert gs._load_manifest() == {}, "a non-dict document is not a manifest"


def test_a_read_never_shrinks_a_healthy_manifest(manifest, monkeypatch):
    """list_models() re-persists; with every entry healthy it must round-trip."""
    monkeypatch.setattr(gs, "index_ollama_models", lambda: [])
    manifest.write_text(json.dumps({"a": _entry("a"), "b": _entry("b")}))
    assert {e.name for e in gs.list_models()} == {"a", "b"}
    assert set(json.loads(manifest.read_text())) == {"a", "b"}

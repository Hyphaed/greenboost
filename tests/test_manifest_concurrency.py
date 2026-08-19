"""Concurrent manifest writers must not lose each other's registrations.

Three registrations disappeared on 2026-08-18 — twice breaking the CLI with
`no such model` for an alias registered minutes earlier — because the manifest
is edited by read-modify-write from several places at once (`pull()`,
`serve()`'s resolution, `list_models()`'s Ollama re-persist) with no lock.

The failure is a LOST UPDATE, not corruption: the later writer saves a dict it
loaded BEFORE the earlier writer's addition, and the addition vanishes. Nothing
is logged, because each process's own write genuinely succeeded. That is why it
took three occurrences to catch.
"""
from __future__ import annotations

import json
import multiprocessing as mp

import pytest

import gb_synapse as gs


def _entry(name: str) -> dict:
    return {"name": name, "path": f"/tmp/{name}.gguf", "source": "hf",
            "repo": "o/r", "quant": "Q4", "quant_method": "", "quant_bits": 0,
            "arch": "qwen", "engine": "llama.cpp", "n_bytes": 1}


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    f = tmp_path / "manifest.json"
    f.write_text(json.dumps({"pre-existing": _entry("pre-existing")}))
    monkeypatch.setattr(gs, "MANIFEST_FILE", f)
    return f


def test_update_manifest_preserves_untouched_entries(manifest):
    gs.update_manifest(lambda m: m.__setitem__("added", gs.ModelEntry(**_entry("added"))))
    on_disk = json.loads(manifest.read_text())
    assert set(on_disk) == {"pre-existing", "added"}


def test_update_manifest_sees_a_concurrent_addition(manifest):
    """The exact lost-update shape: B must not erase A's write.

    Simulated by writing A directly to disk after B has already been 'thinking',
    then having B commit through update_manifest — which re-reads under the lock.
    """
    manifest.write_text(json.dumps({
        "pre-existing": _entry("pre-existing"), "written-by-A": _entry("written-by-A")}))
    gs.update_manifest(lambda m: m.__setitem__("written-by-B",
                                               gs.ModelEntry(**_entry("written-by-B"))))
    on_disk = json.loads(manifest.read_text())
    assert "written-by-A" in on_disk, "B clobbered A's registration"
    assert "written-by-B" in on_disk


def _writer(path: str, name: str) -> None:
    import gb_synapse as g
    from pathlib import Path
    g.MANIFEST_FILE = Path(path)
    g.update_manifest(lambda m: m.__setitem__(name, g.ModelEntry(**_entry(name))))


def test_parallel_writers_all_survive(manifest):
    """The real reproduction: several processes registering at once."""
    names = [f"model-{i}" for i in range(8)]
    procs = [mp.Process(target=_writer, args=(str(manifest), n)) for n in names]
    for p in procs: p.start()
    for p in procs: p.join(timeout=60)
    on_disk = json.loads(manifest.read_text())
    missing = [n for n in names if n not in on_disk]
    assert not missing, f"lost registrations under concurrency: {missing}"
    assert "pre-existing" in on_disk


def test_rm_removes_only_its_own_key(manifest):
    manifest.write_text(json.dumps({"a": _entry("a"), "b": _entry("b")}))
    gs.rm("a")
    assert set(json.loads(manifest.read_text())) == {"b"}


def test_rm_on_a_missing_model_raises_without_touching_the_file(manifest):
    before = manifest.read_text()
    with pytest.raises(KeyError):
        gs.rm("not-here")
    assert manifest.read_text() == before


def test_lock_is_released_even_if_mutate_raises(manifest):
    """A crashed writer must not wedge every later one."""
    with pytest.raises(RuntimeError):
        gs.update_manifest(lambda m: (_ for _ in ()).throw(RuntimeError("boom")))
    gs.update_manifest(lambda m: m.__setitem__("after", gs.ModelEntry(**_entry("after"))))
    assert "after" in json.loads(manifest.read_text())

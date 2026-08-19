"""Tests for incremental RAG re-indexing (engine.update_folder / update_all).

These avoid loading torch / sentence-transformers by monkeypatching
``engine._embed`` with a deterministic stub, and redirect all store paths into
a tmp dir.  Run with:  python -m pytest tests/test_rag_update.py -v
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from greenboost_cli.rag import engine

BASELINE = 1_600_000_000.0  # fixed epoch used as the "last_indexed" pivot
DIM = 8


@pytest.fixture
def rag(tmp_path, monkeypatch):
    """Redirect the RAG store into tmp_path and stub the embedder."""
    rag_dir = tmp_path / "rag"
    monkeypatch.setattr(engine, "RAG_DIR", rag_dir)
    monkeypatch.setattr(engine, "EMBEDDINGS_FILE", rag_dir / "embeddings.npy")
    monkeypatch.setattr(engine, "METADATA_FILE", rag_dir / "metadata.json")
    monkeypatch.setattr(engine, "FOLDERS_FILE", rag_dir / "indexed_folders.yaml")
    monkeypatch.setattr(engine, "WEB_SOURCES_FILE", rag_dir / "web_sources.json")
    monkeypatch.setattr(engine, "STATS_FILE", rag_dir / "stats.json")
    monkeypatch.setattr(engine, "_embed",
                        lambda texts: np.ones((len(texts), DIM), dtype=np.float32))
    return rag_dir


def _write(path: Path, words: int = 12) -> None:
    path.write_text(" ".join(f"word{i}" for i in range(words)) + "\n")


def _set_mtime(path: Path, ts: float) -> None:
    os.utime(path, (ts, ts))


def _pin_last_indexed(folder: Path, ts: float) -> None:
    """Force a folder entry's last_indexed to a known timestamp."""
    folders = engine._load_folders()
    for e in folders:
        if e["folder"] == str(folder.resolve()):
            e["last_indexed"] = datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    engine._save_folders(folders)


def test_update_folder_incremental(rag, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    a, b, d = proj / "a.txt", proj / "b.txt", proj / "d.txt"
    for f in (a, b, d):
        _write(f)

    engine.index_folder(proj)              # full index: 3 files → 3 chunks
    emb, meta = engine._load_store()
    assert len(meta) == 3 and emb.shape[0] == 3

    _pin_last_indexed(proj, BASELINE)
    _set_mtime(a, BASELINE - 100)          # unchanged
    _set_mtime(b, BASELINE + 100)          # modified
    d.unlink()                             # removed
    c = proj / "c.txt"
    _write(c)
    _set_mtime(c, BASELINE + 100)          # new

    res = engine.update_folder(proj)

    assert res["reindexed_files"] == 2     # b + c
    assert res["removed_files"] == 1       # d
    assert res["unchanged_files"] == 1     # a
    assert res["chunks_added"] == 2        # b, c
    assert res["chunks_removed"] == 2      # old b + d
    assert res["forced"] is False

    emb, meta = engine._load_store()
    assert len(meta) == emb.shape[0] == 3  # a + b + c, store stays aligned
    files = {m["file"] for m in meta}
    assert str(d) not in files
    assert str(c) in files


def test_update_folder_no_changes_skips_embed(rag, tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    a = proj / "a.txt"
    _write(a)
    engine.index_folder(proj)

    # Files predate the pivot and nothing is removed → must not embed.
    _pin_last_indexed(proj, BASELINE)
    _set_mtime(a, BASELINE - 100)

    def _boom(_texts):
        raise AssertionError("_embed must not be called when nothing changed")
    monkeypatch.setattr(engine, "_embed", _boom)

    res = engine.update_folder(proj)
    assert res["reindexed_files"] == 0
    assert res["removed_files"] == 0
    assert res["unchanged_files"] == 1

    # last_indexed was bumped forward past the pivot.
    entry = next(e for e in engine._load_folders() if e["folder"] == str(proj.resolve()))
    assert datetime.fromisoformat(entry["last_indexed"]).timestamp() > BASELINE


def test_update_folder_force_full_reindex(rag, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    _write(proj / "a.txt")
    engine.index_folder(proj)
    _pin_last_indexed(proj, BASELINE)
    _set_mtime(proj / "a.txt", BASELINE - 100)   # would be "unchanged" incrementally

    res = engine.update_folder(proj, force=True)
    assert res["forced"] is True

    emb, meta = engine._load_store()
    assert len(meta) == emb.shape[0] == 1


def test_resolve_folder_entry_longest_match_and_skips_synthetic(rag, tmp_path):
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)

    engine._save_folders([
        {"folder": str(parent), "project": "parent", "chunk_count": 1,
         "last_indexed": "2020-01-01T00:00:00"},
        {"folder": str(child), "project": "child", "chunk_count": 1,
         "last_indexed": "2020-01-01T00:00:00"},
        {"folder": "qa:20200101-000000", "project": "qa_history",
         "chunk_count": 1, "last_indexed": "2020-01-01T00:00:00"},
        {"folder": "https://example.com/doc", "project": "web",
         "chunk_count": 1, "last_indexed": "2020-01-01T00:00:00"},
    ])

    entry = engine.resolve_folder_entry(child)
    assert entry is not None and entry["project"] == "child"   # longest match wins

    entry_parent = engine.resolve_folder_entry(parent)
    assert entry_parent["project"] == "parent"

    assert engine.resolve_folder_entry(tmp_path) is None       # outside any folder


def test_save_store_atomic(rag):
    emb = np.ones((3, DIM), dtype=np.float32)
    meta = [{"file": f"f{i}", "project": "p", "lines": [1, 2], "text": "x"} for i in range(3)]
    engine._save_store(emb, meta)

    # No temp files left behind.
    assert not (rag / "embeddings.npy.tmp").exists()
    assert not (rag / "metadata.json.tmp").exists()

    loaded_emb, loaded_meta = engine._load_store()
    assert loaded_emb.shape == (3, DIM)
    assert len(loaded_meta) == 3

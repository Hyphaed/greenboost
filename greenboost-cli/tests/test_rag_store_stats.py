"""store_stats() must be cheap AND never wrong.

`_load_store()` is called from ten places and several want nothing but
`len(metadata)`. On the reference box that meant parsing a 413 MB JSON —
measured 987 MB peak RSS, 0.91 s — to print "272,080 chunks" at every startup.

A cache that returns a stale count is worse than the slow path it replaced, so
the staleness rules get more test weight than the fast path does.
"""
import json
import os
import time

import numpy as np
import pytest

from greenboost_cli.rag import engine as E


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "RAG_DIR", tmp_path)
    monkeypatch.setattr(E, "METADATA_FILE", tmp_path / "metadata.json")
    monkeypatch.setattr(E, "EMBEDDINGS_FILE", tmp_path / "embeddings.npy")
    monkeypatch.setattr(E, "STATS_FILE", tmp_path / "stats.json")
    return tmp_path


def _write_meta(store, rows):
    (store / "metadata.json").write_text(json.dumps(rows))


def test_no_store_reports_zero_not_an_error(store):
    assert E.store_stats() == {"chunks": 0, "files": 0}


def test_counts_match_a_full_parse(store):
    rows = [{"file": "a.py", "text": "x"}, {"file": "a.py", "text": "y"},
            {"file": "b.py", "text": "z"}]
    _write_meta(store, rows)
    assert E.store_stats() == {"chunks": 3, "files": 2}


def test_sidecar_is_written_and_then_used(store, monkeypatch):
    """The fast path is taken only while the store is byte-for-byte unchanged."""
    _write_meta(store, [{"file": "a.py"}])
    assert not (store / "stats.json").exists()
    assert E.store_stats() == {"chunks": 1, "files": 1}
    assert (store / "stats.json").exists()

    # Prove the second call did NOT re-parse: make parsing fail loudly. The
    # sidecar is valid for this exact store, so it must be used instead.
    def _explode(*_a, **_kw):
        raise AssertionError("store_stats re-parsed a store it had a valid sidecar for")

    monkeypatch.setattr(E.json, "load", _explode)
    assert E.store_stats() == {"chunks": 1, "files": 1}


def test_a_stale_sidecar_is_ignored(store):
    """The failure mode that would make this worse than the slow path."""
    _write_meta(store, [{"file": "a.py"}])
    E.store_stats()
    _write_meta(store, [{"file": "a.py"}, {"file": "b.py"}, {"file": "c.py"}])
    os.utime(store / "metadata.json", None)          # store is now newer
    assert E.store_stats() == {"chunks": 3, "files": 3}


def test_a_corrupt_sidecar_falls_back_to_the_full_parse(store):
    _write_meta(store, [{"file": "a.py"}, {"file": "b.py"}])
    (store / "stats.json").write_text("{ truncated")
    os.utime(store / "stats.json", (time.time() + 10,) * 2)   # newer, but broken
    assert E.store_stats() == {"chunks": 2, "files": 2}


def test_a_sidecar_with_wrong_types_is_rejected(store):
    _write_meta(store, [{"file": "a.py"}])
    (store / "stats.json").write_text(json.dumps({"chunks": "many", "files": None}))
    os.utime(store / "stats.json", (time.time() + 10,) * 2)
    assert E.store_stats() == {"chunks": 1, "files": 1}


def test_an_unparseable_store_reports_zero(store):
    (store / "metadata.json").write_text("{ not json")
    assert E.store_stats() == {"chunks": 0, "files": 0}


def test_malformed_rows_do_not_raise(store):
    _write_meta(store, [{"file": "a.py"}, "a bare string", 42, {"no_file_key": 1}])
    stats = E.store_stats()
    assert stats["chunks"] == 4          # every row still counts as a chunk
    assert stats["files"] >= 1


def test_save_store_refreshes_the_sidecar(store):
    _write_meta(store, [{"file": "a.py"}])
    assert E.store_stats()["chunks"] == 1
    E._save_store(np.zeros((2, 4), dtype=np.float32),
                  [{"file": "a.py"}, {"file": "b.py"}])
    # No utime games: the sidecar must already be correct after a save.
    assert E.store_stats() == {"chunks": 2, "files": 2}


def test_the_cached_path_is_orders_of_magnitude_cheaper(store):
    _write_meta(store, [{"file": f"f{i}.py", "text": "x" * 200} for i in range(20000)])
    t = time.perf_counter(); E.store_stats(); cold = time.perf_counter() - t
    t = time.perf_counter(); E.store_stats(); warm = time.perf_counter() - t
    assert warm < cold / 10, f"sidecar gave no real saving (cold {cold:.3f}s warm {warm:.3f}s)"


def test_every_store_path_constant_is_redirectable(store):
    """A new path constant must not silently escape test isolation.

    STATS_FILE was added without updating test_rag_update.py's `rag` fixture,
    so `_save_store()` wrote a sidecar into the developer's REAL store — which
    then reported 3 chunks for a 272,080-chunk index. This fails the moment a
    future constant is added without being covered here too.
    """
    known = {"METADATA_FILE", "EMBEDDINGS_FILE", "STATS_FILE",
             "FOLDERS_FILE", "WEB_SOURCES_FILE"}
    from pathlib import Path

    found = {
        name for name, val in vars(E).items()
        if name.endswith("_FILE") and isinstance(val, Path)
    }
    missing = found - known
    assert not missing, (
        f"new RAG path constant(s) {sorted(missing)} — add them to this set AND "
        f"to the `rag` fixture in tests/test_rag_update.py, or tests will write "
        f"into the real store"
    )


def test_a_foreign_sidecar_is_not_believed(store):
    """Newer is not the same as 'describes this file'.

    A sidecar left by something else was newer than a 413 MB store it had
    never read, and the banner reported 3 chunks instead of 272,080.
    """
    _write_meta(store, [{"file": f"f{i}.py"} for i in range(9)])
    (store / "stats.json").write_text(json.dumps(
        {"chunks": 3, "files": 3, "fingerprint": "not-this-store"}))
    os.utime(store / "stats.json", (time.time() + 60,) * 2)   # decisively newer
    assert E.store_stats() == {"chunks": 9, "files": 9}


def test_a_sidecar_without_a_fingerprint_is_not_believed(store):
    """Sidecars written before fingerprinting existed must not be trusted."""
    _write_meta(store, [{"file": "a.py"}, {"file": "b.py"}])
    (store / "stats.json").write_text(json.dumps({"chunks": 99, "files": 99}))
    os.utime(store / "stats.json", (time.time() + 60,) * 2)
    assert E.store_stats() == {"chunks": 2, "files": 2}

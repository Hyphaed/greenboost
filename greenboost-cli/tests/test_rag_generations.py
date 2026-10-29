"""The RAG store writes immutable generations and switches between them with one
atomic rename. Covers: a crash mid-save, vectors/chunks that disagree, adopting a
pre-generations store, mixing embedding models, pruning, and mmap search."""
import json
import os

import numpy as np
import pytest

from greenboost_cli.rag import engine as E

DIM = 4


@pytest.fixture
def rag(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "RAG_DIR", tmp_path / "rag")
    yield_dir = tmp_path / "rag"
    assert str(yield_dir).startswith(str(tmp_path))        # isolation: never the real ~/.greenboost_cli
    monkeypatch.setattr(E, "STATS_FILE", tmp_path / "rag" / "stats.json")
    monkeypatch.setattr(E, "_active_embed_model", "model-A")
    return tmp_path / "rag"


def _vecs(n, fill=1.0, dim=DIM):
    return np.full((n, dim), fill, dtype=np.float32)


def _meta(n, tag="x"):
    return [{"file": f"/p/{tag}{i}.py", "project": "p", "lines": [1, 2], "text": f"{tag}{i}"} for i in range(n)]


def _generations(rag):
    return sorted(d.name for d in (rag / "generations").iterdir() if d.is_dir())


# ── a crash can never produce a mismatched pair ───────────────────────────────

def test_crash_before_the_switch_leaves_the_previous_generation_active(rag, monkeypatch):
    E._save_store(_vecs(3, 1.0), _meta(3, "old"))
    before = os.readlink(rag / "current")
    real_switch = E._switch_current

    def crash(gid):
        raise OSError("power cut")
    monkeypatch.setattr(E, "_switch_current", crash)
    with pytest.raises(OSError):
        E._save_store(_vecs(5, 2.0), _meta(5, "new"))
    # NOT monkeypatch.undo(): that would also revert the fixture's RAG_DIR and point
    # this test at the developer's real index.
    monkeypatch.setattr(E, "_switch_current", real_switch)

    assert os.readlink(rag / "current") == before
    emb, meta = E._load_store()
    assert emb.shape == (3, DIM) and meta[0]["text"] == "old0"        # old vectors AND old text
    assert len(_generations(rag)) == 1                                  # the half-written one was removed


def test_crash_while_writing_metadata_leaves_nothing_half_done(rag, monkeypatch):
    E._save_store(_vecs(2), _meta(2, "old"))
    real = json.JSONEncoder.iterencode

    def boom(self, o, *a, **k):
        yield "["
        raise MemoryError("died mid-write")
    monkeypatch.setattr(json.JSONEncoder, "iterencode", boom)
    with pytest.raises(MemoryError):
        E._save_store(_vecs(9), _meta(9, "new"))
    monkeypatch.setattr(json.JSONEncoder, "iterencode", real)
    emb, meta = E._load_store()
    assert emb.shape[0] == 2 and meta[1]["text"] == "old1"
    assert len(_generations(rag)) == 1


def test_a_save_with_vectors_and_chunks_that_disagree_is_refused(rag):
    with pytest.raises(E.RagIndexMismatch):
        E._save_store(_vecs(3), _meta(2))
    assert not (rag / "current").exists()


def test_a_store_whose_files_disagree_is_not_loaded_silently(rag):
    E._save_store(_vecs(3), _meta(3))
    (rag / "current" / "metadata.json").write_text(json.dumps(_meta(2)))   # torn pair
    with pytest.raises(E.RagIndexMismatch):
        E._load_store()


# ── manifest ──────────────────────────────────────────────────────────────────

def test_manifest_records_what_is_needed_to_trust_the_vectors(rag):
    E._save_store(_vecs(3), _meta(3))
    m = E.read_manifest()
    assert m["n_chunks"] == 3 and m["dims"] == DIM and m["embed_model"] == "model-A"
    assert m["chunker_version"] == E.CHUNKER_VERSION and len(m["metadata_sha256"]) == 64
    assert m["generation"] == os.readlink(rag / "current").split("/")[-1]


def test_the_fallback_embedding_model_is_recorded_not_passed_off_as_primary(rag, monkeypatch):
    monkeypatch.setattr(E, "_active_embed_model", "fallback-model")
    E._save_store(_vecs(1), _meta(1))
    assert E.read_manifest()["embed_model"] == "fallback-model"


def test_appending_vectors_from_a_different_model_is_refused(rag):
    E._save_store(_vecs(2), _meta(2))
    E._active_embed_model = "model-B"                       # same dimension, different model
    with pytest.raises(E.RagIndexMismatch, match="model-A"):
        E._guard_append(np.zeros((2, DIM), np.float32), _vecs(1))
    E._active_embed_model = "model-A"
    E._guard_append(np.zeros((2, DIM), np.float32), _vecs(1))             # same model: fine
    with pytest.raises(E.RagIndexMismatch, match="-dim"):
        E._guard_append(np.zeros((2, DIM), np.float32), _vecs(1, dim=DIM + 2))


# ── a store from before generations is adopted, not lost ──────────────────────

def test_legacy_store_is_adopted_as_the_first_generation(rag):
    rag.mkdir(parents=True)
    np.save(str(rag / "embeddings.npy"), _vecs(4, 3.0))
    (rag / "metadata.json").write_text(json.dumps(_meta(4, "legacy")))
    emb, meta = E._load_store()
    assert emb.shape == (4, DIM) and meta[3]["text"] == "legacy3"
    assert os.readlink(rag / "current").endswith("g-legacy")
    assert not (rag / "embeddings.npy").exists() and not (rag / "metadata.json").exists()
    assert E.read_manifest()["legacy"] is True and E.read_manifest()["embed_model"] is None
    E._guard_append(emb, _vecs(1))                           # unknown model: not blocked


def test_adopting_the_legacy_store_is_idempotent(rag):
    rag.mkdir(parents=True)
    np.save(str(rag / "embeddings.npy"), _vecs(2))
    (rag / "metadata.json").write_text(json.dumps(_meta(2)))
    E._migrate_legacy()
    E._migrate_legacy()
    assert E._load_store()[0].shape == (2, DIM)


def test_next_save_after_adoption_builds_on_the_adopted_data(rag):
    rag.mkdir(parents=True)
    np.save(str(rag / "embeddings.npy"), _vecs(2))
    (rag / "metadata.json").write_text(json.dumps(_meta(2, "legacy")))
    emb, meta = E._load_store()
    E._save_store(np.concatenate([emb, _vecs(1, 9.0)]), meta + _meta(1, "new"))
    emb2, meta2 = E._load_store()
    assert emb2.shape[0] == 3 and meta2[-1]["text"] == "new0"


# ── disk use is bounded, and a crashed save can't evict the real previous ─────

def test_old_generations_are_pruned(rag):
    for i in range(6):
        E._save_store(_vecs(i + 1), _meta(i + 1))
    assert len(_generations(rag)) <= E.KEEP_GENERATIONS
    assert E._load_store()[0].shape[0] == 6


def test_a_crashed_save_does_not_count_as_a_previous_generation(rag):
    E._save_store(_vecs(1), _meta(1, "a"))
    E._save_store(_vecs(2), _meta(2, "b"))
    partial = rag / "generations" / "g-crashed"
    partial.mkdir()
    (partial / "embeddings.npy").write_bytes(b"junk")        # no manifest: never finished
    E._save_store(_vecs(3), _meta(3, "c"))
    names = _generations(rag)
    assert "g-crashed" not in names and len(names) == E.KEEP_GENERATIONS


# ── search ────────────────────────────────────────────────────────────────────

def test_search_ranks_by_similarity_using_the_mmapped_matrix(rag, monkeypatch):
    emb = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]], dtype=np.float32)
    meta = [{"file": f"/p/f{i}.py", "project": "p", "lines": [1, 1], "text": f"chunk {i}"} for i in range(3)]
    E._save_store(emb, meta)
    monkeypatch.setattr(E, "_embed", lambda texts: np.array([[1, 0, 0, 0]], dtype=np.float32))
    hits = E.search("anything", top_k=2, min_score=0.1)
    assert [h["file"] for h in hits] == ["/p/f0.py", "/p/f2.py"]


def test_search_refuses_a_query_from_a_different_dimension(rag, monkeypatch):
    E._save_store(_vecs(2), _meta(2))
    monkeypatch.setattr(E, "_embed", lambda texts: np.ones((1, DIM + 3), dtype=np.float32))
    with pytest.raises(E.RagIndexMismatch):
        E.search("q")


def test_search_on_an_empty_store_returns_nothing(rag):
    assert E.search("q") == []

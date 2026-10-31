"""Embedding placement (queries on CPU, bulk on a free GPU) and the resumable re-embed."""
from __future__ import annotations

import numpy as np
import pytest

from greenboost_cli.rag import engine as E


class _Vec:
    def __init__(self, a): self._a = np.asarray(a, dtype="float32")
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self._a


class _FakeModel:
    prompts = {"query": "Instruct: find code\nQuery: "}
    def __init__(self, dim=6): self.dim, self.calls, self.max_seq_length = dim, [], 1024
    def encode(self, texts, **kw):
        self.calls.append((len(texts), kw.get("prompt_name") or kw.get("prompt")))
        return _Vec([[float(len(t) % 7 + 1)] * self.dim for t in texts])
    def parameters(self): return iter(())


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("GB_EMBED_DEVICE", raising=False)
    monkeypatch.delenv("GB_EMBED_BULK_MIN", raising=False)
    monkeypatch.setattr(E, "_models", {})


def _patch_gpu(monkeypatch, room: bool):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(E, "_gpu_has_room_for", lambda m: room)
    cpu, gpu = _FakeModel(), _FakeModel()
    monkeypatch.setattr(E, "_get_model", lambda device=None, name=None: cpu if device == "cpu" else gpu)
    return cpu, gpu


def test_queries_and_small_batches_stay_on_the_cpu(monkeypatch):
    cpu, gpu = _patch_gpu(monkeypatch, room=True)
    E._embed(["one query"], query=True)
    E._embed(["a"] * 10)
    assert [c[0] for c in cpu.calls] == [1, 10] and not gpu.calls
    assert cpu.calls[0][1].startswith("Instruct:") and "codebase" in cpu.calls[0][1]
    assert cpu.calls[1][1] is None                                    # never on the document side


def test_bulk_goes_to_the_gpu_when_there_is_room_else_the_cpu(monkeypatch):
    cpu, gpu = _patch_gpu(monkeypatch, room=True)
    E._embed(["x"] * 100)
    assert gpu.calls and not cpu.calls
    cpu2, gpu2 = _patch_gpu(monkeypatch, room=False)
    E._embed(["x"] * 100)
    assert cpu2.calls and not gpu2.calls


def test_gpu_embedder_is_released_after_bulk_work(monkeypatch):
    _patch_gpu(monkeypatch, room=True)
    released = []
    monkeypatch.setattr(E, "release_gpu_embedder", lambda: released.append(1))
    E._embed(["x"] * 100)
    assert released == [1]


def test_device_env_overrides(monkeypatch):
    cpu, gpu = _patch_gpu(monkeypatch, room=True)
    monkeypatch.setenv("GB_EMBED_DEVICE", "cpu")
    E._embed(["x"] * 100)
    assert cpu.calls and not gpu.calls


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "RAG_DIR", tmp_path / "rag")
    monkeypatch.setattr(E, "STATS_FILE", tmp_path / "rag" / "stats.json")
    meta = [{"file": f"/p/f{i}.py", "project": "p", "lines": [1, 2], "text": "x" * (10 + i * 3)} for i in range(25)]
    monkeypatch.setattr(E, "_active_embed_model", "old-model")
    E._save_store(np.ones((25, 4), dtype="float32"), meta)
    monkeypatch.setattr(E, "EMBED_MODEL", "new-model")
    monkeypatch.setattr(E, "_active_embed_model", "new-model")
    return meta


def test_reembed_swaps_a_new_generation_and_keeps_chunk_order(store, monkeypatch):
    _patch_gpu(monkeypatch, room=True)
    monkeypatch.setattr(E, "REEMBED_PART", 10)
    monkeypatch.setattr(E, "release_gpu_embedder", lambda: None)
    seen = []
    r = E.reembed_store(progress=lambda d, t, eta: seen.append((d, t)))
    assert r["status"] == "done" and r["chunks"] == 25 and r["dims"] == 6
    assert seen[-1] == (25, 25)
    emb, meta = E._load_store()
    assert emb.shape == (25, 6) and E.read_manifest()["embed_model"] == "new-model"
    # vector i must belong to chunk i (the fake encodes len(text) % 7 + 1)
    assert all(emb[i, 0] == float(len(meta[i]["text"]) % 7 + 1) for i in range(25))
    assert not (E.RAG_DIR / "reembed.tmp").exists()


def test_reembed_refuses_without_gpu_room_and_changes_nothing(store, monkeypatch):
    _patch_gpu(monkeypatch, room=False)
    with pytest.raises(RuntimeError, match="Not enough free VRAM"):
        E.reembed_store()
    assert E._load_store()[0].shape == (25, 4)           # the live index is untouched


def test_reembed_is_a_noop_when_already_current(store, monkeypatch):
    monkeypatch.setattr(E, "EMBED_MODEL", "old-model")
    assert E.reembed_store()["status"] == "current"


def test_search_embeds_the_query_with_the_model_that_built_the_index(store, monkeypatch):
    """After the default moves to a new model but before /rag-reembed, the old
    index must still be searched with ITS embedder, not rejected."""
    seen = {}
    monkeypatch.setattr(E, "_embed", lambda texts, **kw: seen.update(kw) or np.ones((1, 4), dtype="float32"))
    E.search("anything", top_k=3, min_score=0.0)
    assert seen.get("model_name") == "old-model" and seen.get("query") is True


def test_reembed_survives_an_out_of_memory_by_halving_the_batch(store, monkeypatch):
    """A real run climbed to 10.2 of 11.6 GiB because the longest chunks come last."""
    gpu = _FakeModel()
    seen = []
    real_encode = gpu.encode

    def encode(texts, **kw):
        seen.append(kw.get("batch_size"))
        if kw.get("batch_size", 0) > 8:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real_encode(texts, **kw)
    gpu.encode = encode
    cpu = _FakeModel()
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(E, "_gpu_has_room_for", lambda m: True)
    monkeypatch.setattr(E, "_get_model", lambda device=None, name=None: cpu if device == "cpu" else gpu)
    monkeypatch.setattr(E, "release_gpu_embedder", lambda: None)
    monkeypatch.setattr(E, "REEMBED_PART", 10)
    r = E.reembed_store(batch_size=32)
    assert r["status"] == "done" and E._load_store()[0].shape == (25, 6)
    assert seen[:3] == [32, 16, 8] and set(seen[3:]) == {8}        # halved until it fit, then stayed small


def test_a_non_oom_error_is_not_swallowed(store, monkeypatch):
    gpu = _FakeModel()
    gpu.encode = lambda texts, **kw: (_ for _ in ()).throw(ValueError("bad input"))
    cpu = _FakeModel()
    monkeypatch.setattr(E, "_gpu_has_room_for", lambda m: True)
    monkeypatch.setattr(E, "_get_model", lambda device=None, name=None: cpu if device == "cpu" else gpu)
    monkeypatch.setattr(E, "release_gpu_embedder", lambda: None)
    with pytest.raises(ValueError):
        E.reembed_store()


def test_a_big_batch_that_would_land_on_the_cpu_is_deferred_not_run(monkeypatch):
    cpu, gpu = _patch_gpu(monkeypatch, room=False)
    with pytest.raises(E.EmbedDeferred, match="GB_EMBED_CPU_MAX_CHUNKS"):
        E._embed(["x"] * 500, cpu_max=300)
    assert not cpu.calls and not gpu.calls                        # nothing was embedded
    E._embed(["x"] * 200, cpu_max=300)                            # small enough: runs on the CPU
    assert cpu.calls
    cpu.calls.clear()
    E._embed(["x"] * 500, cpu_max=0)                              # 0 = no limit, the owner's choice
    assert cpu.calls


def test_the_cap_never_blocks_a_batch_that_goes_to_the_gpu(monkeypatch):
    cpu, gpu = _patch_gpu(monkeypatch, room=True)
    E._embed(["x"] * 5000, cpu_max=300)
    assert gpu.calls and not cpu.calls


def test_the_cap_defaults_and_reads_the_environment(monkeypatch):
    monkeypatch.delenv("GB_EMBED_CPU_MAX_CHUNKS", raising=False)
    assert E._cpu_bulk_cap() == 300
    monkeypatch.setenv("GB_EMBED_CPU_MAX_CHUNKS", "0")
    assert E._cpu_bulk_cap() == 0


def test_query_instruction_by_task_override_and_models_without_prompts(monkeypatch):
    monkeypatch.delenv("GB_EMBED_QUERY_PROMPT", raising=False)
    m = _FakeModel()
    assert "codebase" in E._query_prompt(m, "code")["prompt"]
    assert "skill" in E._query_prompt(m, "skill")["prompt"]          # skill matching is not code search
    monkeypatch.setenv("GB_EMBED_QUERY_PROMPT", "Find the config key")
    assert E._query_prompt(m, "code")["prompt"] == "Instruct: Find the config key\nQuery: "
    assert "skill" in E._query_prompt(m, "skill")["prompt"]          # the override is for code search only
    monkeypatch.setenv("GB_EMBED_QUERY_PROMPT", "none")
    assert E._query_prompt(m, "code") == {"prompt_name": "query"}    # the model's own
    plain = _FakeModel(); plain.prompts = {}
    assert E._query_prompt(plain, "code") == {}                       # e.g. MiniLM: nothing to add


def test_releasing_the_gpu_embedder_really_frees_the_weights_before_the_cache_is_emptied(monkeypatch):
    """A /rag-update left ~1.5 GB of VRAM allocated to the CLI for the rest of the session,
    which took it from the model server's expert-cache budget (the cache then did not fit
    and decode fell from ~181 to ~54 tok/s). empty_cache() can only return weights that
    nothing references any more: not a local variable, not the module's `_model`."""
    import gc
    import weakref
    import torch

    class _Weights(_FakeModel):
        pass

    w = _Weights()
    ref = weakref.ref(w)
    monkeypatch.setattr(E, "_models", {"cuda": w})
    monkeypatch.setattr(E, "_model", w)
    monkeypatch.setattr(E, "_policy", lambda: "auto")
    alive_when_cache_emptied = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: alive_when_cache_emptied.append(ref() is not None))
    monkeypatch.setattr(E, "_pick_device", lambda n, b: "cuda")
    monkeypatch.setattr(E, "_get_model", lambda device=None, name=None: E._models["cuda"])   # no closure over the weights
    del w
    gc.collect()
    assert ref() is not None                                   # still held by the module, as it would be mid-use
    E._embed(["x"] * 3)                                        # a GPU batch; releases when done
    assert alive_when_cache_emptied == [False], alive_when_cache_emptied
    assert ref() is None and E._model is None and "cuda" not in E._models

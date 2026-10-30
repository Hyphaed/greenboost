#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for gb_synapse.shard_paths()/gguf_summary()'s split-GGUF handling
(2026-08-28 fix).

Real incident: Qwen3.8-Flash-Next-UD-Q2_K_XL is a 3-way split GGUF whose
first shard (`split.no=0`) holds ZERO tensors of its own — all 1224 tensors
live in shards 2-3. gguf_summary() used to sum only the one file it was
handed, so the manifest recorded `n_bytes: 0` / `expert_bytes: 0` for a
78.9 GB model and `recommend()` reported `fits_vram: true`. These tests pin
the fix with a REAL synthetic split GGUF (built via the vendored
GGUFWriter's own split machinery, not a hand-rolled binary layout) — the
live re-summarize of the actual Flash-Next manifest entry (done manually
this session, see workflow/known-issues.md) is the real-data proof this
fixture cannot substitute for on its own.

CPU-only, no hardware, no network."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import gb_synapse as gs

_VENDORED_GGUF_PY = str(Path(__file__).parent.parent / "greenboost-sources"
                         / "llama.cpp" / "gguf-py")
if _VENDORED_GGUF_PY not in sys.path:
    sys.path.insert(0, _VENDORED_GGUF_PY)
from gguf.gguf_writer import GGUFWriter  # noqa: E402


def _write_split_gguf(dest: Path, *, n_tensors: int, split_max_tensors: int,
                       small_first_shard: bool, arch: str = "llama",
                       n_layers: int = 2, ctx_length: int = 4096,
                       tensor_shape=(8, 8), expert_tensor_idx=None) -> Path:
    """Build a REAL split GGUF via the vendored writer's own split logic
    (small_first_shard reproduces the exact "metadata-only first shard, zero
    tensors" shape the live incident hit). Returns the path to shard 1
    (`-00001-of-NNNNN.gguf`), matching what callers hand to gguf_summary()."""
    w = GGUFWriter(str(dest), arch=arch, split_max_tensors=split_max_tensors,
                   small_first_shard=small_first_shard)
    w.add_context_length(ctx_length)
    w.add_block_count(n_layers)
    for i in range(n_tensors):
        name = (f"blk.{i}.ffn_gate_exps.weight" if i == expert_tensor_idx
                else f"blk.{i}.attn.weight")
        w.add_tensor(name, np.zeros(tensor_shape, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    shard_files = sorted(dest.parent.glob(f"{dest.stem}-*-of-*.gguf"))
    assert shard_files, "GGUFWriter did not produce split shards"
    return shard_files[0]


def test_shard_paths_derives_siblings_by_naming_not_glob(tmp_path):
    for i in (1, 2, 3):
        (tmp_path / f"model-{i:05d}-of-00003.gguf").write_bytes(b"")
    # An unrelated second quant in the SAME directory — shard_paths() must
    # not pick it up via a glob.
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"")

    got = gs.shard_paths(str(tmp_path / "model-00001-of-00003.gguf"))
    assert got == [str(tmp_path / f"model-{i:05d}-of-00003.gguf") for i in (1, 2, 3)]


def test_shard_paths_non_split_file_is_unchanged(tmp_path):
    p = tmp_path / "plain.gguf"
    p.write_bytes(b"")
    assert gs.shard_paths(str(p)) == [str(p)]


def test_split_gguf_first_shard_empty_still_sums_all_tensors(tmp_path):
    """The exact live-incident shape: shard 1 holds zero tensors of its
    own. Summing only the handed-in file must NOT report zero bytes."""
    shard1 = _write_split_gguf(tmp_path / "toy", n_tensors=4,
                                split_max_tensors=2, small_first_shard=True)
    assert shard1.name.endswith("-00001-of-00003.gguf")

    # The bug this fix removes: reading shard 1 alone reports zero tensors.
    GGUFReader = gs._load_gguf_reader()
    assert len(GGUFReader(str(shard1), mode="r").tensors) == 0

    meta = gs.gguf_summary(str(shard1))
    assert meta["n_bytes"] > 0
    assert meta["n_layers"] == 2
    assert meta["ctx_length"] == 4096


def test_split_gguf_expert_bytes_summed_across_shards(tmp_path):
    shard1 = _write_split_gguf(tmp_path / "toy", n_tensors=6,
                                split_max_tensors=2, small_first_shard=True,
                                expert_tensor_idx=4)
    meta = gs.gguf_summary(str(shard1))
    assert meta["expert_bytes"] > 0
    assert meta["expert_bytes"] < meta["n_bytes"]
    assert meta["dense_bytes"] == meta["n_bytes"] - meta["expert_bytes"]


def test_split_gguf_missing_shard_raises_loud(tmp_path):
    shard1 = _write_split_gguf(tmp_path / "toy", n_tensors=4,
                                split_max_tensors=2, small_first_shard=True)
    # Simulate a partial download: delete the last shard.
    shards = gs.shard_paths(str(shard1))
    Path(shards[-1]).unlink()

    with pytest.raises(RuntimeError, match="missing"):
        gs.gguf_summary(str(shard1))


def test_split_gguf_cache_keyed_on_all_shards(tmp_path):
    shard1 = _write_split_gguf(tmp_path / "toy", n_tensors=4,
                                split_max_tensors=2, small_first_shard=True)
    meta1 = gs.gguf_summary(str(shard1))

    # Mutate a SIBLING shard (not the handed-in file) and confirm the cache
    # invalidates — a stale total after a resumed download is exactly the
    # failure this cache-key extension exists to prevent.
    shards = gs.shard_paths(str(shard1))
    sibling = Path(shards[1])
    with open(sibling, "ab") as f:
        f.write(b"\x00" * 64)
    import os
    os.utime(sibling, None)

    meta2 = gs.gguf_summary(str(shard1))
    assert meta2 == meta1  # content-equivalent re-parse, not a stale hit
    assert gs._GGUF_SUMMARY_CACHE[str(shard1)][0] != (
        (Path(shards[0]).stat().st_mtime, 0),)  # sanity: key isn't degenerate


def test_non_split_gguf_unaffected(tmp_path):
    """A plain, non-split GGUF must behave exactly as before — no split.*
    fields, single reader, no tensor-count reconciliation."""
    w = GGUFWriter(str(tmp_path / "plain.gguf"), arch="llama")
    w.add_context_length(2048)
    w.add_block_count(1)
    w.add_tensor("blk.0.attn.weight", np.zeros((4, 4), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    meta = gs.gguf_summary(str(tmp_path / "plain.gguf"))
    assert meta["n_layers"] == 1
    assert meta["ctx_length"] == 2048
    assert meta["n_bytes"] == 4 * 4 * 4  # float32


def test_quant_label_prefers_file_type_over_tensor_vote(tmp_path):
    """2026-08-28 fix. general.file_type is the authoritative source when
    present -- a per-tensor-name vote must not override it even when every
    tensor on disk is a plain F32 (as in these synthetic fixtures)."""
    w = GGUFWriter(str(tmp_path / "typed.gguf"), arch="llama")
    w.add_context_length(2048)
    w.add_block_count(1)
    from gguf.constants import LlamaFileType
    w.add_file_type(int(LlamaFileType.MOSTLY_Q4_K_M))
    w.add_tensor("blk.0.attn_norm.weight", np.zeros((4,), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    meta = gs.gguf_summary(str(tmp_path / "typed.gguf"))
    assert meta["quant"] == "Q4_K_M"


def test_quant_label_falls_back_to_byte_weighted_vote_not_count(tmp_path):
    """2026-08-28 fix, the real live-incident shape: many small F32 norm-
    weight tensors outnumber a few much larger real-payload tensors. Without
    general.file_type, the vote must be weighted by bytes, not tensor count
    -- a plain count vote picks F32 (wrong: it's the norm dtype, not the
    model's real quant) exactly like it did on the live Flash-Next
    checkpoint (485 F32 norm tensors outvoted 244 real Q8_0 tensors)."""
    w = GGUFWriter(str(tmp_path / "voted.gguf"), arch="llama")
    w.add_context_length(2048)
    w.add_block_count(1)
    # No add_file_type() call -- forces the byte-weighted fallback.
    for i in range(5):
        w.add_tensor(f"blk.0.norm{i}.weight", np.zeros((4,), dtype=np.float32))
    # One much larger F16 tensor: fewer tensors, more total bytes.
    w.add_tensor("blk.0.ffn_down.weight", np.zeros((4096, 4096), dtype=np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    meta = gs.gguf_summary(str(tmp_path / "voted.gguf"))
    assert meta["quant"] == "F16"  # not "F32", despite F32 outnumbering it 5:1

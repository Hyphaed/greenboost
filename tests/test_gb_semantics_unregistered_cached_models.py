#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for gb_semantics._res_unregistered_cached_models's 2026-09-01
recursive-glob fix — the same bug as gb_synapse.recover_from_hf_cache's
(see tests/test_gb_synapse_recover_cache.py), but this is the copy that
directly backs the `unregistered_cached_models` metric and the
`models_wiped_from_manifest` segment: a shallow, single-level glob
undercounted models cached in Unsloth's UD-* per-quant-subfolder layout
(snapshots/<rev>/UD-Q2_K_XL/<file>.gguf) exactly like the recovery function
did, on the exact repo family this box has cached.

CPU-only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gb_semantics as S
import gb_synapse as gs


def _touch_gguf(path: Path, size: int = 1_100_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0" * size)


def test_finds_gguf_nested_in_per_quant_subfolder(tmp_path, monkeypatch):
    store = tmp_path / "store"
    monkeypatch.setattr(gs, "MODEL_STORE_DIR", store)
    monkeypatch.setattr(gs, "list_models", lambda: [])
    base = store / "_hf_cache" / "models--unsloth--Flash-GGUF" / "snapshots" / ("r" * 40) / "UD-Q2_K_XL"
    _touch_gguf(base / "Flash-UD-Q2_K_XL-00001-of-00003.gguf")
    _touch_gguf(base / "Flash-UD-Q2_K_XL-00002-of-00003.gguf")
    _touch_gguf(base / "Flash-UD-Q2_K_XL-00003-of-00003.gguf")

    out = S._res_unregistered_cached_models("host", 0)

    # A 3-shard download is ONE missing model, not three — counting every
    # shard would inflate models_wiped_from_manifest for a model that was
    # only ever going to register as a single manifest entry.
    assert out["value"] == 1


def test_known_model_is_not_counted_as_missing(tmp_path, monkeypatch):
    store = tmp_path / "store"
    monkeypatch.setattr(gs, "MODEL_STORE_DIR", store)
    path = store / "_hf_cache" / "models--org--repo" / "snapshots" / ("a" * 40) / "M.gguf"
    _touch_gguf(path)
    monkeypatch.setattr(gs, "list_models",
                        lambda: [gs.ModelEntry(name="M", path=str(path), source="hf")])

    out = S._res_unregistered_cached_models("host", 0)
    assert out["value"] == 0

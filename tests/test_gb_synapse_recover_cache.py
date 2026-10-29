#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for gb_synapse.recover_from_hf_cache() (2026-09-01 fixes).

Real incident this closes: this box's own HF cache held 8 GGUFs across 8 repo
dirs while the manifest carried only 4 entries after a purge, and the pre-fix
recovery function itself had three independent reasons it couldn't heal that:
a single-level glob that missed Unsloth's per-quant subfolder layout, no
collapsing of a multi-shard download into one servable entry, and no
disambiguation when two cached snapshot revisions of the same repo hold an
identically-named file. All three are exercised here against a real
filesystem tree (tmp_path), not a mock of the glob — the recursive-vs-single-
level distinction is exactly the kind of thing a mocked filesystem call would
paper over.

CPU-only. `gguf_summary` is monkeypatched to a stub — this file tests the
discovery/naming/collision logic, not GGUF header parsing (that has its own
tests, test_gb_synapse_*_summary.py)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serving"))

import yaml

import check_recipes as cr
import gb_synapse as gs


def _stub_gguf_summary(monkeypatch):
    monkeypatch.setattr(gs, "gguf_summary",
                        lambda path: {"n_bytes": 1_100_000, "quant": "Q4", "arch": "test"})


def _touch_gguf(path: Path, size: int = 1_100_000, mtime: float = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0" * size)
    if mtime is not None:
        import os
        os.utime(path, (mtime, mtime))


def _setup_store(tmp_path, monkeypatch):
    store = tmp_path / "store"
    manifest = store / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}")
    monkeypatch.setattr(gs, "MODEL_STORE_DIR", store)
    monkeypatch.setattr(gs, "MANIFEST_FILE", manifest)
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    monkeypatch.setattr(cr, "RECIPES_DIR", recipes_dir)
    _stub_gguf_summary(monkeypatch)
    return store, manifest, recipes_dir


def test_recovers_nested_per_quant_shard_layout(tmp_path, monkeypatch):
    """fix 1: Unsloth's UD-* layout nests shards one level below
    snapshots/<rev>/ (a per-quant subfolder) — a single-level glob misses
    them entirely."""
    store, manifest, _ = _setup_store(tmp_path, monkeypatch)
    base = store / "_hf_cache" / "models--unsloth--Flash-GGUF" / "snapshots" / ("r" * 40) / "UD-Q2_K_XL"
    for i in (1, 2, 3):
        _touch_gguf(base / f"Flash-UD-Q2_K_XL-{i:05d}-of-00003.gguf")

    result = gs.recover_from_hf_cache()

    assert len(result) == 1
    assert result[0].name == "Flash-UD-Q2_K_XL"
    assert result[0].path.endswith("-00001-of-00003.gguf")


def test_shard_set_collapses_to_one_entry_not_three(tmp_path, monkeypatch):
    """fix 2: N shards of one download must register as ONE ModelEntry
    (only the first shard is independently loadable), not N."""
    store, manifest, _ = _setup_store(tmp_path, monkeypatch)
    base = store / "_hf_cache" / "models--org--repo" / "snapshots" / ("a" * 40)
    for i in (1, 2, 3):
        _touch_gguf(base / f"Model-{i:05d}-of-00003.gguf")

    result = gs.recover_from_hf_cache()

    assert len(result) == 1
    saved = json.loads(manifest.read_text())
    assert len(saved) == 1


def test_mmproj_never_registered_as_its_own_model(tmp_path, monkeypatch):
    store, manifest, _ = _setup_store(tmp_path, monkeypatch)
    base = store / "_hf_cache" / "models--org--vl" / "snapshots" / ("b" * 40)
    _touch_gguf(base / "vl-model.gguf")
    _touch_gguf(base / "mmproj-vl-model.gguf")

    result = gs.recover_from_hf_cache()

    names = {e.name for e in result}
    assert "vl-model" in names
    assert not any("mmproj" in n for n in names)


def test_recipe_pinned_name_wins_over_filename_stem(tmp_path, monkeypatch):
    """fix 4 (this box's actual bug): a recipe pins an exact (revision,
    filename) -> a human-chosen model.name. Recovery must adopt that name
    instead of the raw, usually uglier, filename stem — otherwise the
    reference workload recovers under a name no recipe/serve() call
    recognizes."""
    store, manifest, recipes_dir = _setup_store(tmp_path, monkeypatch)
    rev = "c" * 40
    fname = "Ugly-Vendor-Internal-Name.gguf"
    base = store / "_hf_cache" / "models--vendor--repo" / "snapshots" / rev
    _touch_gguf(base / fname)

    recipe = {
        "model": {"name": "Clean-Pinned-Name", "revision": rev,
                  "files": [{"path": fname, "sha256": "d" * 64}]},
    }
    (recipes_dir / "r.yaml").write_text(yaml.safe_dump(recipe))

    result = gs.recover_from_hf_cache()

    assert len(result) == 1
    assert result[0].name == "Clean-Pinned-Name"


def test_no_recipe_match_falls_back_to_filename_stem(tmp_path, monkeypatch):
    store, manifest, recipes_dir = _setup_store(tmp_path, monkeypatch)
    base = store / "_hf_cache" / "models--vendor--repo" / "snapshots" / ("e" * 40)
    _touch_gguf(base / "Plain-Name.gguf")

    result = gs.recover_from_hf_cache()

    assert result[0].name == "Plain-Name"


def test_duplicate_revision_same_name_keeps_newer_and_reports_skip(tmp_path, monkeypatch):
    """fix 3: two snapshot revisions of the same repo holding an
    identically-named file must not both register — filesystem iteration
    order deciding which one gets served silently is the bug. The newer
    file (by mtime) wins; the other is named in `.skipped`, not dropped
    without a trace."""
    store, manifest, _ = _setup_store(tmp_path, monkeypatch)
    repo = store / "_hf_cache" / "models--org--repo" / "snapshots"
    now = time.time()
    old = repo / ("1" * 40) / "Dup-Model.gguf"
    new = repo / ("2" * 40) / "Dup-Model.gguf"
    _touch_gguf(old, mtime=now - 100_000)
    _touch_gguf(new, mtime=now)

    result = gs.recover_from_hf_cache()

    assert len(result) == 1
    assert result[0].path == str(new)
    assert len(result.skipped) == 1
    assert result.skipped[0]["path"] == str(old)


def test_recovery_result_stays_a_plain_list_for_existing_callers(tmp_path, monkeypatch):
    """greenboost_setup.sh's Full Install step does len(r)/r[:12]/for e in r
    on the return value — RecoveryResult must keep behaving exactly like the
    list it replaces for every caller that doesn't ask about `.skipped`."""
    store, manifest, _ = _setup_store(tmp_path, monkeypatch)
    base = store / "_hf_cache" / "models--org--repo" / "snapshots" / ("f" * 40)
    _touch_gguf(base / "Solo.gguf")

    result = gs.recover_from_hf_cache()

    assert isinstance(result, list)
    assert len(result) == 1
    assert [e.name for e in result[:12]] == ["Solo"]


def test_already_registered_model_is_left_alone(tmp_path, monkeypatch):
    """A path already present in the manifest must never be re-processed —
    recovery only ADDS missing entries, per its own safety contract."""
    store, manifest, _ = _setup_store(tmp_path, monkeypatch)
    base = store / "_hf_cache" / "models--org--repo" / "snapshots" / ("9" * 40)
    path = base / "Known.gguf"
    _touch_gguf(path)
    manifest.write_text(json.dumps({"Known": {"name": "Known", "path": str(path)}}))

    result = gs.recover_from_hf_cache()

    assert len(result) == 0
    assert len(result.skipped) == 0


def test_empty_cache_returns_empty_recovery_result(tmp_path, monkeypatch):
    _setup_store(tmp_path, monkeypatch)
    result = gs.recover_from_hf_cache()
    assert list(result) == []
    assert result.skipped == []

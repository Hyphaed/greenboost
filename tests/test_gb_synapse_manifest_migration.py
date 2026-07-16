#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for gb_synapse._load_manifest()'s engine-value migration:
{gbquant, vllm, transformers} -> "torch" (P1.5 of the gb-synapse unification).

CPU-only."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gb_synapse as gs


def _base_entry(**overrides):
    e = {"name": "fake-model", "path": "/tmp/fake"}
    e.update(overrides)
    return e


def test_gbquant_migrates_to_torch(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"m1": _base_entry(engine="gbquant")}))
    monkeypatch.setattr(gs, "MANIFEST_FILE", manifest)
    entries = gs._load_manifest()
    assert entries["m1"].engine == "torch"


def test_vllm_migrates_to_torch(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"m1": _base_entry(engine="vllm")}))
    monkeypatch.setattr(gs, "MANIFEST_FILE", manifest)
    entries = gs._load_manifest()
    assert entries["m1"].engine == "torch"


def test_transformers_migrates_to_torch(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"m1": _base_entry(engine="transformers")}))
    monkeypatch.setattr(gs, "MANIFEST_FILE", manifest)
    entries = gs._load_manifest()
    assert entries["m1"].engine == "torch"


def test_llamacpp_unaffected(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"m1": _base_entry(engine="llama.cpp")}))
    monkeypatch.setattr(gs, "MANIFEST_FILE", manifest)
    entries = gs._load_manifest()
    assert entries["m1"].engine == "llama.cpp"


def test_diffusers_unaffected(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"m1": _base_entry(engine="diffusers")}))
    monkeypatch.setattr(gs, "MANIFEST_FILE", manifest)
    entries = gs._load_manifest()
    assert entries["m1"].engine == "diffusers"


def test_old_manifest_without_new_quant_fields_still_loads(tmp_path, monkeypatch):
    """A manifest written before quant_method/quant_bits existed must still
    load — the dataclass defaults ("" / 0) fill in."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"m1": _base_entry(engine="llama.cpp")}))
    monkeypatch.setattr(gs, "MANIFEST_FILE", manifest)
    entries = gs._load_manifest()
    assert entries["m1"].quant_method == ""
    assert entries["m1"].quant_bits == 0


def test_round_trip_through_save_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"m1": _base_entry(engine="vllm")}))
    monkeypatch.setattr(gs, "MANIFEST_FILE", manifest)
    entries = gs._load_manifest()
    gs._save_manifest(entries)
    reloaded = json.loads(manifest.read_text())
    assert reloaded["m1"]["engine"] == "torch"

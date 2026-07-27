#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for DiffusionOrchestrator._emit_phase() , the P5 backfill fixing
gb_diffusion_orch.py (the phase choreographer ai-forge's production image
path runs under) never referencing gb_dataflux at all, despite being the
real T1/T2 promote/demote decision-maker for encode/denoise/decode phases.

Constructs a bare instance via object.__new__ (bypassing __init__, which
needs a real diffusers pipe/torch device) since _emit_phase only touches
its own arguments plus gb_dataflux.emit , no other instance state.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gb_diffusion_orch as gdo


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    p = tmp_path / "dataflux.jsonl"
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(p))
    import gb_dataflux
    gb_dataflux._READ_EVENTS_MEMO.clear()
    yield p
    gb_dataflux._READ_EVENTS_MEMO.clear()


def _bare_orch():
    return object.__new__(gdo.DiffusionOrchestrator)


def test_emit_phase_writes_placement_event(log_path):
    orch = _bare_orch()
    orch._emit_phase("encode", ("text_encoder",), ("transformer", "vae"))

    import gb_dataflux
    events = gb_dataflux.read_events()
    placements = [e for e in events if e.get("kind") == "placement"]
    assert len(placements) == 1
    ev = placements[0]
    assert ev["runtime"] == "diffusion_phase"
    assert ev["phase"] == "encode"
    assert ev["promoted"] == ["text_encoder"]
    assert ev["demoted"] == ["transformer", "vae"]


def test_emit_phase_never_raises_when_dataflux_unavailable(monkeypatch):
    orch = _bare_orch()
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **kw):
        if name == "gb_dataflux":
            raise ImportError("simulated")
        return real_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", _fake_import)

    orch._emit_phase("denoise", (), ())  # must not raise

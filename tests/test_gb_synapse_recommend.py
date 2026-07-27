#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_synapse.recommend()'s measured-vs-estimated tok/s substitution.

This is the "--measure" gap workflow/gb-synapse.md's `_estimate_tok_s`
docstring flags as future work: it's actually already shipped —
recommend() prefers a real, client-observed rolling average
(_measured_tok_s(), fed by gb_synapse_api.py's proxy via
record_measured_tok_s() after every turn) over the bandwidth-bound
heuristic whenever a sample exists for that model, and FitReport.measured
says which one it used. This file is the first direct test of that
substitution — CPU-only, no GGUF, no CUDA, no network.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gb_synapse as gs


def _fake_doctor(*, probe_feeders=True):
    return {
        "host_vram_total_mb": 12000, "host_vram_free_mb": 10000,
        "aggregate_vram_mb": 12000, "feeders": [],
    }


def _fake_entry(name="qwen3", n_bytes=4 * 1024 ** 3):
    return gs.ModelEntry(name=name, path="", quant="Q4_K_M", n_bytes=n_bytes,
                         n_layers=32, n_kv_heads=8, head_dim=128)


def test_recommend_uses_measured_tok_s_when_a_sample_exists(monkeypatch):
    monkeypatch.setattr(gs, "doctor", _fake_doctor)
    monkeypatch.setattr(gs, "list_models", lambda: [_fake_entry()])
    monkeypatch.setattr(gs, "_measured_tok_s", lambda model: 42.5)
    # If the heuristic ran, this would prove the substitution didn't happen.
    monkeypatch.setattr(gs, "_estimate_tok_s", lambda active_gb, budget_gb: 999.0)

    reports = gs.recommend(ctx=8192, probe_feeders=False)

    assert len(reports) == 1
    r = reports[0]
    assert r.measured is True
    assert r.est_tok_s == 42.5
    assert "measured" in r.note


def test_recommend_falls_back_to_estimate_without_a_sample(monkeypatch):
    monkeypatch.setattr(gs, "doctor", _fake_doctor)
    monkeypatch.setattr(gs, "list_models", lambda: [_fake_entry()])
    monkeypatch.setattr(gs, "_measured_tok_s", lambda model: None)
    monkeypatch.setattr(gs, "_estimate_tok_s", lambda active_gb, budget_gb: 17.0)

    reports = gs.recommend(ctx=8192, probe_feeders=False)

    assert len(reports) == 1
    r = reports[0]
    assert r.measured is False
    assert r.est_tok_s == 17.0
    assert "measured" not in r.note


def test_recommend_measured_flag_reaches_json_and_mcp_dict(monkeypatch):
    """The MCP tool (synapse_recommend) and the --llm CLI path both go
    through dataclasses.asdict(report) — confirm `measured` survives that,
    since it's the field a caller actually filters/sorts on."""
    from dataclasses import asdict

    monkeypatch.setattr(gs, "doctor", _fake_doctor)
    monkeypatch.setattr(gs, "list_models", lambda: [_fake_entry()])
    monkeypatch.setattr(gs, "_measured_tok_s", lambda model: 88.0)

    reports = gs.recommend(ctx=8192, probe_feeders=False)
    d = asdict(reports[0])

    assert d["measured"] is True
    assert d["est_tok_s"] == 88.0

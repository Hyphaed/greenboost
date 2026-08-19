#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_synapse_mcp.quality_gate() — the MCP invocation surface added
2026-08-05 for gb_aviary's smoke_gate()/niah_certify(). Before this, both
functions had zero callers anywhere in the repo (no CLI, no MCP tool), so
CLAUDE.md's "never below fp8 without gate evidence" policy had no
repeatable way to actually be exercised.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gb_synapse_mcp


def test_quality_gate_smoke_routes_to_smoke_gate(monkeypatch):
    captured = {}

    def _fake_smoke_gate(model):
        captured["model"] = model
        return {"verdict": "PASS", "kind": "smoke_gate", "model": model}

    monkeypatch.setattr("gb_aviary.smoke_gate", _fake_smoke_gate)
    out = gb_synapse_mcp.quality_gate("qwen3", gate="smoke")
    assert captured["model"] == "qwen3"
    assert out["verdict"] == "PASS"


def test_quality_gate_niah_routes_to_niah_certify_with_all_params(monkeypatch):
    captured = {}

    def _fake_niah_certify(model, tokens, needles=10, kv_type="unknown", seed=1337):
        captured.update(model=model, tokens=tokens, needles=needles,
                        kv_type=kv_type, seed=seed)
        return {"status": "ok", "score": needles, "needles": needles}

    monkeypatch.setattr("gb_aviary.niah_certify", _fake_niah_certify)
    out = gb_synapse_mcp.quality_gate("qwen3", gate="niah", tokens=32768,
                                      needles=5, kv_type="q8_0", seed=42)
    assert captured == {"model": "qwen3", "tokens": 32768, "needles": 5,
                        "kv_type": "q8_0", "seed": 42}
    assert out["status"] == "ok"


def test_quality_gate_unknown_gate_errors_without_calling_either_function(monkeypatch):
    calls = []
    monkeypatch.setattr("gb_aviary.smoke_gate", lambda *a, **k: calls.append("smoke"))
    monkeypatch.setattr("gb_aviary.niah_certify", lambda *a, **k: calls.append("niah"))
    out = gb_synapse_mcp.quality_gate("qwen3", gate="bogus")
    assert "error" in out
    assert calls == []


def test_quality_gate_never_raises_on_underlying_exception(monkeypatch):
    def _boom(model):
        raise RuntimeError("upstream unreachable")
    monkeypatch.setattr("gb_aviary.smoke_gate", _boom)
    out = gb_synapse_mcp.quality_gate("qwen3", gate="smoke")
    assert "error" in out

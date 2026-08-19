#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for gb_synapse.py's serving-recipe integration (NemoClaw audit,
Phase 5e) and probe-suite wiring (Phase 5d): load_recipe()'s lookup/
validation, serve()'s additive ctx/mtp_draft_n override, and
probe_serve_readiness_for()'s real (but bounded/mockable) step callables.

CPU-only. No GGUF, no CUDA, no real HF/network calls — every network or
subprocess boundary is monkeypatched to a fake.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serving"))

import pytest
import yaml

import gb_synapse as gs
import check_recipes as cr


def _write_recipe(path, **overrides):
    recipe = {
        "schemaVersion": "1",
        "model": {
            "name": "test-model",
            "revision": "a" * 40,
            "files": [{"path": "model.gguf", "sha256": "b" * 64}],
        },
        "capabilities": {
            "toolCalls": True, "streaming": True, "mtpDraftHead": True,
            "visionProjector": False, "recurrentState": False,
        },
        "kvCache": {"key": "q8_0", "value": "q8_0"},
        "ctx": 45824,
        "nGpuLayers": "all",
        "tierIntent": "t2Spill",
        "mtpDraftN": 4,
    }
    recipe.update(overrides)
    recipe["contentDigest"] = cr.compute_content_digest(recipe)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(recipe, f)
    return recipe


class _FakeEmit:
    def __init__(self):
        self.events = []

    def __call__(self, ev):
        self.events.append(ev)


@pytest.fixture
def recipes_dir(tmp_path, monkeypatch):
    d = tmp_path / "recipes"
    d.mkdir()
    monkeypatch.setattr(cr, "RECIPES_DIR", d)
    return d


@pytest.fixture
def fake_dataflux(monkeypatch):
    fake_emit = _FakeEmit()
    import types
    fake_module = types.SimpleNamespace(emit=fake_emit)
    monkeypatch.setitem(sys.modules, "gb_dataflux", fake_module)
    return fake_emit


def test_load_recipe_returns_none_when_no_matching_file(recipes_dir):
    assert gs.load_recipe("nonexistent-model") is None


def test_load_recipe_returns_matching_valid_recipe(recipes_dir, fake_dataflux):
    _write_recipe(recipes_dir / "r.yaml")
    recipe = gs.load_recipe("test-model")
    assert recipe is not None
    assert recipe["ctx"] == 45824
    assert any(e["kind"] == "recipe_validate" and e["status"] == "valid"
               for e in fake_dataflux.events)


def test_load_recipe_ignores_recipe_for_a_different_model(recipes_dir):
    _write_recipe(recipes_dir / "r.yaml")
    assert gs.load_recipe("some-other-model") is None


def test_load_recipe_rejects_invalid_recipe_and_emits_invalid(recipes_dir, fake_dataflux):
    path = recipes_dir / "r.yaml"
    recipe = _write_recipe(path)
    recipe["ctx"] = 999999999  # drift after digest computed -> stale digest
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(recipe, f)

    result = gs.load_recipe("test-model")
    assert result is None
    assert any(e["kind"] == "recipe_validate" and e["status"] == "invalid"
               for e in fake_dataflux.events)


def test_probe_serve_readiness_for_stops_at_first_failure(monkeypatch, fake_dataflux):
    entry = gs.ModelEntry(name="test-model", path="", source="hf")
    monkeypatch.setattr(gs, "_default_short_load_check", lambda e: True)
    monkeypatch.setattr(
        gs, "_http_probe",
        lambda url, timeout_s, payload=None: (False, "connection refused"),
    )
    results = gs.probe_serve_readiness_for(entry, 11369, require_tool_call=True)
    assert not results[-1].ok
    assert results[-1].step == "health"
    assert results[-1].reason == "health_unhealthy"
    # completion steps must never have run
    assert not any(r.step in ("one_token_completion", "tool_call_completion") for r in results)


def test_probe_serve_readiness_for_all_steps_pass_emits_ok_events(monkeypatch, fake_dataflux):
    entry = gs.ModelEntry(name="test-model", path="", source="hf")
    monkeypatch.setattr(gs, "_default_short_load_check", lambda e: True)
    monkeypatch.setattr(
        gs, "_http_probe",
        lambda url, timeout_s, payload=None: (True, '{"choices":[{"text":"ok"}]}'),
    )
    results = gs.probe_serve_readiness_for(entry, 11369, require_tool_call=True)
    assert len(results) == 5
    assert all(r.ok for r in results)
    probe_events = [e for e in fake_dataflux.events if e["kind"] == "serve_probe"]
    assert len(probe_events) == 5
    assert all(e["status"] == "ok" for e in probe_events)


def test_probe_serve_readiness_for_require_tool_call_none_reads_recipe(
    recipes_dir, monkeypatch, fake_dataflux,
):
    _write_recipe(recipes_dir / "r.yaml", capabilities={
        "toolCalls": False, "streaming": True, "mtpDraftHead": True,
        "visionProjector": False, "recurrentState": False,
    })
    entry = gs.ModelEntry(name="test-model", path="", source="hf")
    monkeypatch.setattr(gs, "_default_short_load_check", lambda e: True)
    monkeypatch.setattr(
        gs, "_http_probe",
        lambda url, timeout_s, payload=None: (True, "ok"),
    )
    results = gs.probe_serve_readiness_for(entry, 11369)  # require_tool_call=None
    assert [r.step for r in results] == [
        "gguf_header", "short_load", "health", "one_token_completion",
    ]


def test_probe_serve_readiness_for_gguf_header_step_skips_non_hf_models(monkeypatch, fake_dataflux):
    entry = gs.ModelEntry(name="test-model", path="", source="ollama")
    monkeypatch.setattr(gs, "_default_short_load_check", lambda e: True)
    monkeypatch.setattr(
        gs, "_http_probe",
        lambda url, timeout_s, payload=None: (True, "ok"),
    )
    results = gs.probe_serve_readiness_for(entry, 11369, require_tool_call=False)
    assert results[0].step == "gguf_header"
    assert results[0].ok


def test_probe_serve_readiness_for_gguf_header_exception_maps_to_malformed(
    monkeypatch, fake_dataflux,
):
    entry = gs.ModelEntry(name="test-model", path="/nonexistent/file.gguf", source="hf")
    results = gs.probe_serve_readiness_for(entry, 11369, require_tool_call=False)
    assert results[0].step == "gguf_header"
    assert not results[0].ok
    assert results[0].reason == "gguf_malformed"


# ── serve()'s additive recipe override (Phase 5e) ─────────────────────────

class _StopServe(Exception):
    """Sentinel raised by the fake backend so serve() stops right after the
    override logic, before touching anything else it would need mocking
    for (proxy launch, run-state persistence, ...) — this test is only
    about what ctx/mtp_draft_n backend.serve() gets CALLED with."""


@pytest.fixture
def captured_backend_call(monkeypatch):
    calls = []

    class _FakeBackend:
        def serve(self, entry, port, **kwargs):
            calls.append({"entry": entry, "port": port, **kwargs})
            raise _StopServe

    import gb_synapse_backends
    monkeypatch.setattr(gb_synapse_backends, "select_backend", lambda entry: _FakeBackend())
    monkeypatch.setattr(gs, "_resolve_model", lambda model: gs.ModelEntry(
        name=model, path="", source="hf", engine="llama.cpp"))
    monkeypatch.setattr(gs, "_read_run_states", lambda: [])
    return calls


def test_serve_ctx_sentinel_is_overridden_by_matching_recipe(
    recipes_dir, captured_backend_call, fake_dataflux,
):
    _write_recipe(recipes_dir / "r.yaml", ctx=45824, mtpDraftN=4)
    with pytest.raises(_StopServe):
        gs.serve("test-model", ctx=0, mtp_draft_n=None)
    assert captured_backend_call[0]["ctx"] == 45824
    assert captured_backend_call[0]["mtp_draft_n"] == 4


def test_serve_explicit_caller_ctx_wins_over_recipe(
    recipes_dir, captured_backend_call, fake_dataflux,
):
    _write_recipe(recipes_dir / "r.yaml", ctx=45824, mtpDraftN=4)
    with pytest.raises(_StopServe):
        gs.serve("test-model", ctx=8192, mtp_draft_n=None)
    assert captured_backend_call[0]["ctx"] == 8192  # caller's explicit value, not the recipe's
    assert captured_backend_call[0]["mtp_draft_n"] == 4  # still overridden — this one WAS a sentinel


def test_serve_no_matching_recipe_leaves_sentinels_untouched(
    recipes_dir, captured_backend_call, fake_dataflux,
):
    with pytest.raises(_StopServe):
        gs.serve("test-model", ctx=0, mtp_draft_n=None)
    assert captured_backend_call[0]["ctx"] == 0
    assert captured_backend_call[0]["mtp_draft_n"] is None


# ── placement overrides reach backend.serve() (Phase 5/7 follow-up) ──────

def test_serve_threads_ngpulayers_and_kvcache_from_recipe(
    recipes_dir, captured_backend_call, fake_dataflux,
):
    _write_recipe(recipes_dir / "r.yaml", nGpuLayers=42,
                  kvCache={"key": "q4_0", "value": "q4_0"})
    with pytest.raises(_StopServe):
        gs.serve("test-model")
    call = captured_backend_call[0]
    assert call["n_gpu_layers_override"] == 42
    assert call["kv_type_override"] == "q4_0"


def test_serve_threads_ngpulayers_all_sentinel_from_recipe(
    recipes_dir, captured_backend_call, fake_dataflux,
):
    _write_recipe(recipes_dir / "r.yaml", nGpuLayers="all")
    with pytest.raises(_StopServe):
        gs.serve("test-model")
    assert captured_backend_call[0]["n_gpu_layers_override"] == "all"


def test_serve_threads_ncpumoe_when_recipe_has_it(
    recipes_dir, captured_backend_call, fake_dataflux,
):
    _write_recipe(recipes_dir / "r.yaml", nCpuMoe=20)
    with pytest.raises(_StopServe):
        gs.serve("test-model")
    assert captured_backend_call[0]["n_cpu_moe_override"] == 20


def test_serve_asymmetric_kvcache_does_not_set_kv_type_override(
    recipes_dir, captured_backend_call, fake_dataflux,
):
    """LlamaCppBackend.serve() has no asymmetric --cache-type-k/-v support
    today — an asymmetric recipe must leave kv_type_override unset (falls
    back to the heuristic) rather than silently picking one side."""
    _write_recipe(recipes_dir / "r.yaml", kvCache={"key": "f16", "value": "q8_0"})
    with pytest.raises(_StopServe):
        gs.serve("test-model")
    assert captured_backend_call[0]["kv_type_override"] is None


def test_serve_no_matching_recipe_leaves_placement_overrides_none(
    recipes_dir, captured_backend_call, fake_dataflux,
):
    with pytest.raises(_StopServe):
        gs.serve("test-model")
    call = captured_backend_call[0]
    assert call["n_gpu_layers_override"] is None
    assert call["kv_type_override"] is None
    assert call["n_cpu_moe_override"] is None

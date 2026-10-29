#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Owner rule 2026-10-10: GreenBoost never loads a model without a prompt or an
explicit, named serve/run request. serve()/serve_gguf() are the only loaders
and refuse unless explicit=True with a trigger. No real process is started."""
import ast
import pathlib
import subprocess

import pytest

import gb_synapse as gs

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def raising_loader(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a model load was attempted")
    monkeypatch.setattr(gs, "_serve_impl", boom)
    monkeypatch.setattr(gs.gb_synapse_backends, "select_backend", boom)
    real_popen = subprocess.Popen

    def popen(cmd, *a, **k):
        if any("llama-server" in str(c) or "gb_synapse_api" in str(c)
               for c in (cmd if isinstance(cmd, (list, tuple)) else [cmd])):
            boom()
        return real_popen(cmd, *a, **k)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(gs, "engine_installed", lambda: True)


def test_serve_refuses_without_explicit(raising_loader):
    with pytest.raises(gs.LoadRefused):
        gs.serve("some-model")
    with pytest.raises(gs.LoadRefused):
        gs.serve("some-model", explicit=True)          # no trigger named
    with pytest.raises(gs.LoadRefused):
        gs.serve_gguf("/x.gguf", 1234)


@pytest.mark.parametrize("argv", [
    ["list"], ["status"], ["ps"], ["recommend"], ["doctor"], ["logs", "m"], ["stop", "m"],
])
def test_every_read_only_subcommand_loads_nothing(raising_loader, argv, monkeypatch, tmp_path):
    monkeypatch.setattr(gs, "RUN_DIR", tmp_path, raising=False)
    try:
        gs._cli_main(argv + ["--llm"])
    except AssertionError:
        raise
    except Exception:
        pass  # an environment problem is fine; a load attempt is not


def test_explicit_serve_reaches_loader_and_gets_ttl(monkeypatch):
    seen = {}

    def impl(model, *a, **k):
        seen["ttl"] = gs._LOAD_CTX.idle_ttl_s
        seen["loading"] = gs.load_in_progress()
        return "state"
    monkeypatch.setattr(gs, "_serve_impl", impl)
    monkeypatch.delenv("GB_SYNAPSE_IDLE_TTL_S", raising=False)
    assert gs.serve("m", explicit=True, trigger="prompt") == "state"
    assert seen == {"ttl": 300.0, "loading": True}
    assert not gs.load_in_progress()
    gs.serve("m", explicit=True, trigger="serve_cmd", keep=True)
    assert seen["ttl"] == 0.0
    monkeypatch.setenv("GB_SYNAPSE_IDLE_TTL_S", "0")
    gs.serve("m", explicit=True, trigger="prompt")
    assert seen["ttl"] == 0.0
    monkeypatch.setenv("GB_SYNAPSE_IDLE_TTL_S", "42")
    gs.serve("m", explicit=True, trigger="prompt")
    assert seen["ttl"] == 42.0


def test_shim_probe_never_loads_outside_a_gated_serve(monkeypatch, tmp_path):
    import gb_shim_probe as sp
    monkeypatch.delenv("GB_SYNAPSE_SHIM", raising=False)
    monkeypatch.setattr(sp, "_shim_path", lambda: tmp_path / "shim.so")
    monkeypatch.setattr(sp, "_cache_key", lambda d: {"k": 1})
    monkeypatch.setattr(sp, "_read_cache", lambda: None)
    monkeypatch.setattr(sp, "_run_probe", lambda *a: (_ for _ in ()).throw(AssertionError("probe loaded a model")))
    ok, why = sp.shim_works_for_llama(tmp_path)
    assert not ok and "skipped" in why


def _calls(path, names):
    tree = ast.parse(path.read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            nm = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if nm in names:
                yield n, nm


def test_every_loader_call_site_declares_explicit():
    files = [p for p in list(ROOT.glob("gb_*.py")) + list((ROOT / "greenboost-cli/greenboost_cli").rglob("*.py"))
             if p.name != "gb_synapse.py"]
    bad = []
    for p in files:
        for call, nm in _calls(p, {"serve", "serve_gguf"}):
            f = call.func
            owner = getattr(getattr(f, "value", None), "id", "")
            if nm == "serve" and owner not in ("gb_synapse", "gs"):
                continue
            kws = {k.arg for k in call.keywords}
            if "explicit" not in kws or "trigger" not in kws:
                bad.append(f"{p.relative_to(ROOT)}:{call.lineno}")
    assert not bad, f"loader call sites without explicit/trigger: {bad}"

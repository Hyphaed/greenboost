#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for LlamaCppBackend.serve_embedding() (P6) — the command-building and
port-derivation logic for the second, minimal llama-server this spawns for
embeddings. No real subprocess/GPU: subprocess.Popen and the readiness gate
are monkeypatched.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import gb_synapse_backends as gsb


class _FakeProc:
    pid = 1234

    def poll(self):
        return None


@pytest.fixture
def backend():
    return gsb.LlamaCppBackend()


@pytest.fixture
def entry():
    import gb_synapse as gs
    return gs.ModelEntry(name="embed-model", path="/models/embed.gguf", ctx_length=2048)


@pytest.fixture
def _stub_gb_synapse(monkeypatch, tmp_path):
    import gb_synapse as gs
    monkeypatch.setattr(gs, "engine_installed", lambda: True)
    monkeypatch.setattr(gs, "ENGINE_DIR", tmp_path / "engine")
    monkeypatch.setattr(gs, "SLOT_DIR", tmp_path / "slots")
    monkeypatch.setattr(gs, "_pcore_threads", lambda: 4)
    monkeypatch.setattr(gs, "_run_log_path", lambda label: tmp_path / f"{label}.log")
    monkeypatch.setattr(gs, "_wait_upstream_ready", lambda entry, proc, port, grace_s=60.0: True)


@pytest.fixture
def _stub_gb_cluster(monkeypatch):
    import gb_cluster
    monkeypatch.setattr(gb_cluster, "shim_env",
                        lambda workload, enabled: {"GREENBOOST_ACTIVE": "1", "LD_PRELOAD": "/x.so"})


def test_serve_embedding_port_is_primary_plus_2000(backend, entry, _stub_gb_synapse,
                                                     _stub_gb_cluster, monkeypatch):
    captured = {}

    def _fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        captured["env"] = kw.get("env")
        return _FakeProc()
    monkeypatch.setattr(gsb.subprocess, "Popen", _fake_popen)

    proc, internal_port = backend.serve_embedding(entry, primary_port=11435)

    assert internal_port == 11435 + 2000
    assert "--port" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--port") + 1] == str(internal_port)


def test_serve_embedding_cmd_has_embeddings_and_pooling_flags(backend, entry, _stub_gb_synapse,
                                                                _stub_gb_cluster, monkeypatch):
    captured = {}
    monkeypatch.setattr(gsb.subprocess, "Popen",
                        lambda cmd, **kw: captured.update(cmd=cmd) or _FakeProc())

    backend.serve_embedding(entry, primary_port=11435)

    cmd = captured["cmd"]
    assert "--embeddings" in cmd
    assert "--pooling" in cmd
    assert cmd[cmd.index("--pooling") + 1] == "mean"
    # No jinja/mmproj/spec-decode — this is deliberately not the generation cmd.
    assert "--jinja" not in cmd
    assert "--mmproj" not in cmd


def test_serve_embedding_ctx_capped_at_8192(backend, _stub_gb_synapse, _stub_gb_cluster, monkeypatch):
    import gb_synapse as gs
    big_ctx_entry = gs.ModelEntry(name="embed-model", path="/x", ctx_length=131072)
    captured = {}
    monkeypatch.setattr(gsb.subprocess, "Popen",
                        lambda cmd, **kw: captured.update(cmd=cmd) or _FakeProc())

    backend.serve_embedding(big_ctx_entry, primary_port=11435)

    cmd = captured["cmd"]
    assert cmd[cmd.index("--ctx-size") + 1] == "8192"


def test_serve_embedding_ngl_env_override(backend, entry, _stub_gb_synapse, _stub_gb_cluster,
                                           monkeypatch):
    monkeypatch.setenv("GB_SYNAPSE_EMBED_NGL", "20")
    captured = {}
    monkeypatch.setattr(gsb.subprocess, "Popen",
                        lambda cmd, **kw: captured.update(cmd=cmd) or _FakeProc())

    backend.serve_embedding(entry, primary_port=11435)

    cmd = captured["cmd"]
    assert cmd[cmd.index("-ngl") + 1] == "20"


def test_serve_embedding_strips_shim_when_disabled(backend, entry, _stub_gb_synapse,
                                                     _stub_gb_cluster, monkeypatch):
    monkeypatch.delenv("GB_SYNAPSE_SHIM", raising=False)
    captured = {}
    monkeypatch.setattr(gsb.subprocess, "Popen",
                        lambda cmd, **kw: captured.update(env=kw.get("env")) or _FakeProc())

    backend.serve_embedding(entry, primary_port=11435)

    assert "LD_PRELOAD" not in captured["env"]
    assert "GREENBOOST_ACTIVE" not in captured["env"]


def test_serve_embedding_raises_when_engine_not_built(backend, entry, monkeypatch):
    import gb_synapse as gs
    monkeypatch.setattr(gs, "engine_installed", lambda: False)
    with pytest.raises(RuntimeError, match="engine not built"):
        backend.serve_embedding(entry, primary_port=11435)


def test_serve_embedding_wraps_readiness_failure(backend, entry, _stub_gb_synapse,
                                                  _stub_gb_cluster, monkeypatch):
    import gb_synapse as gs
    monkeypatch.setattr(gs, "_wait_upstream_ready",
                        lambda entry, proc, port, grace_s=60.0: (_ for _ in ()).throw(
                            RuntimeError("engine exited during startup")))
    monkeypatch.setattr(gsb.subprocess, "Popen", lambda cmd, **kw: _FakeProc())

    with pytest.raises(RuntimeError, match="embeddings engine for 'embed-model' failed to start"):
        backend.serve_embedding(entry, primary_port=11435)

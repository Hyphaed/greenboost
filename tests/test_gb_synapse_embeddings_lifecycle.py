#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_synapse.py's embeddings lifecycle (P6):

  * _maybe_serve_embedding() is a no-op when GB_SYNAPSE_EMBED_MODEL is
    unset, when already running, or when the resolved backend has no
    serve_embedding method — and starts one otherwise, persisting
    embed_pid/embed_internal_port onto the ServerState.
  * stop() kills embed_pid alongside llama_pid/proxy_pid.
"""
import pytest

import gb_synapse as gs


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    d = tmp_path / "synapse_run"
    monkeypatch.setattr(gs, "RUN_DIR", d)
    return d


@pytest.fixture
def fake_alive(monkeypatch):
    alive = {}
    monkeypatch.setattr(gs, "_pid_alive", lambda pid: alive.get(pid, False))
    return alive


def _state(**kw):
    defaults = dict(model="qwen3", llama_pid=100, proxy_pid=200, port=11435,
                    internal_port=12435, tensor_split="")
    defaults.update(kw)
    return gs.ServerState(**defaults)


def test_maybe_serve_embedding_noop_when_env_unset(monkeypatch):
    monkeypatch.delenv("GB_SYNAPSE_EMBED_MODEL", raising=False)
    st = _state()

    def _boom(spec):
        raise AssertionError("must not resolve a model when unset")
    monkeypatch.setattr(gs, "_resolve_model", _boom)

    result = gs._maybe_serve_embedding(st)
    assert result is st
    assert result.embed_pid == 0


def test_maybe_serve_embedding_noop_when_already_running(monkeypatch, fake_alive):
    monkeypatch.setenv("GB_SYNAPSE_EMBED_MODEL", "embed-model")
    st = _state(embed_pid=999, embed_internal_port=13435)
    fake_alive[999] = True

    def _boom(spec):
        raise AssertionError("must not re-resolve when already running")
    monkeypatch.setattr(gs, "_resolve_model", _boom)

    result = gs._maybe_serve_embedding(st)
    assert result.embed_pid == 999
    assert result.embed_internal_port == 13435


def test_maybe_serve_embedding_starts_when_configured_and_not_running(
        monkeypatch, run_dir, fake_alive):
    monkeypatch.setenv("GB_SYNAPSE_EMBED_MODEL", "embed-model")
    st = _state()

    monkeypatch.setattr(gs, "_resolve_model",
                        lambda spec: gs.ModelEntry(name="embed-model", path="/x"))

    class _FakeBackend:
        def serve_embedding(self, entry, primary_port):
            class _Proc:
                pid = 4242
            return _Proc(), primary_port + 2000

    monkeypatch.setattr(gs.gb_synapse_backends, "select_backend", lambda entry: _FakeBackend())

    result = gs._maybe_serve_embedding(st)

    assert result.embed_pid == 4242
    assert result.embed_internal_port == 11435 + 2000
    # Persisted, not just in-memory.
    reloaded = gs._read_run_states()[0]
    assert reloaded.embed_pid == 4242


def test_maybe_serve_embedding_skips_when_backend_has_no_support(monkeypatch, run_dir):
    monkeypatch.setenv("GB_SYNAPSE_EMBED_MODEL", "embed-model")
    st = _state()

    monkeypatch.setattr(gs, "_resolve_model",
                        lambda spec: gs.ModelEntry(name="embed-model", path="/x"))

    class _FakeBackendNoEmbed:
        pass  # no serve_embedding method

    monkeypatch.setattr(gs.gb_synapse_backends, "select_backend",
                        lambda entry: _FakeBackendNoEmbed())

    result = gs._maybe_serve_embedding(st)
    assert result.embed_pid == 0


def test_maybe_serve_embedding_never_raises_on_failure(monkeypatch, run_dir):
    monkeypatch.setenv("GB_SYNAPSE_EMBED_MODEL", "embed-model")
    st = _state()

    def _boom(spec):
        raise RuntimeError("resolve failed")
    monkeypatch.setattr(gs, "_resolve_model", _boom)

    result = gs._maybe_serve_embedding(st)  # must not raise
    assert result.embed_pid == 0


def test_stop_kills_embed_pid_alongside_engine_and_proxy(run_dir, fake_alive, monkeypatch):
    st = _state(embed_pid=4242, embed_internal_port=13435)
    gs._write_run_state(st)
    fake_alive.update({100: True, 200: True, 4242: True})

    killed = []
    monkeypatch.setattr(gs, "_kill_process_group", lambda pid: killed.append(pid))
    monkeypatch.setattr(gs, "_teardown_feeders", lambda model, st: None)

    assert gs.stop("qwen3") is True
    assert set(killed) == {100, 200, 4242}


def test_stop_handles_zero_embed_pid_gracefully(run_dir, fake_alive, monkeypatch):
    """The common case: no embeddings engine was ever configured
    (embed_pid defaults to 0) — stop() must not try to kill pid 0."""
    st = _state()  # embed_pid defaults to 0
    gs._write_run_state(st)
    fake_alive.update({100: True, 200: True})

    killed = []
    monkeypatch.setattr(gs, "_kill_process_group", lambda pid: killed.append(pid))
    monkeypatch.setattr(gs, "_teardown_feeders", lambda model, st: None)

    assert gs.stop("qwen3") is True
    assert 0 not in killed
    assert set(killed) == {100, 200}

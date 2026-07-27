#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_synapse.py's process-lifecycle fixes:

  * ps() must prune a run-state only once BOTH the engine and the proxy
    pids are dead — previously it pruned on the engine pid alone, so a
    dead-proxy-live-engine server vanished from ps()/status() instead of
    being reported with proxy_error set.
  * status()'s server_running/proxy_running must be derived from ps()
    (pid-checked run-state) so it's correct for all 4 backends, not just
    llama.cpp (the old pgrep pattern only ever matched llama-server).
  * serve()'s reuse-check must restart ONLY the proxy when the engine is
    still alive but the proxy died — not spawn a second engine.
  * stop() must tear down feeder-side state: unconditionally for the torch
    (gLLM slave) engine, and for llama.cpp only when no other running
    serve still depends on that feeder's rpc-server.

No real subprocesses: _pid_alive is monkeypatched to a fake liveness table
keyed by pid, and RUN_DIR is redirected to tmp_path.
"""
import json

import pytest

import gb_synapse as gs


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    d = tmp_path / "synapse_run"
    monkeypatch.setattr(gs, "RUN_DIR", d)
    return d


@pytest.fixture
def fake_alive(monkeypatch):
    """A pid->bool liveness table _pid_alive reads instead of the real
    os.kill/procfs check."""
    alive = {}
    monkeypatch.setattr(gs, "_pid_alive", lambda pid: alive.get(pid, False))
    return alive


def _write_state(run_dir, **kw):
    defaults = dict(model="qwen3", llama_pid=100, proxy_pid=200, port=11435,
                     internal_port=12435, tensor_split="", feeders=[],
                     started_ts=0.0, engine="llama.cpp", ready=True, proxy_error=None)
    defaults.update(kw)
    st = gs.ServerState(**defaults)
    gs._write_run_state(st)
    return st


# ---------------------------------------------------------------------------
# ps() pruning semantics
# ---------------------------------------------------------------------------

def test_ps_keeps_entry_when_engine_alive_proxy_dead(run_dir, fake_alive):
    _write_state(run_dir, llama_pid=100, proxy_pid=200)
    fake_alive[100] = True
    fake_alive[200] = False

    running = gs.ps()

    assert len(running) == 1
    assert running[0]["proxy_error"] == "proxy process is gone"
    # the run-state file must survive — this is what serve()'s restart
    # path and stop()'s teardown both still need to find.
    assert gs._run_state_path("qwen3").is_file()


def test_ps_prunes_only_when_both_pids_dead(run_dir, fake_alive):
    _write_state(run_dir, llama_pid=100, proxy_pid=200)
    fake_alive[100] = False
    fake_alive[200] = False

    running = gs.ps()

    assert running == []
    assert not gs._run_state_path("qwen3").is_file()


def test_ps_reports_running_when_both_alive(run_dir, fake_alive):
    _write_state(run_dir, llama_pid=100, proxy_pid=200)
    fake_alive[100] = True
    fake_alive[200] = True

    running = gs.ps()

    assert len(running) == 1
    assert running[0]["proxy_error"] is None


# ---------------------------------------------------------------------------
# status() derivation
# ---------------------------------------------------------------------------

def test_status_server_running_true_for_non_llama_cpp_backend(run_dir, fake_alive, monkeypatch):
    """A gLLM (torch) serve must report server_running=True — the old
    pgrep-for-llama-server check always said False for this backend even
    while it was genuinely running."""
    _write_state(run_dir, model="gllm-model", llama_pid=100, proxy_pid=200, engine="torch")
    fake_alive[100] = True
    fake_alive[200] = True
    monkeypatch.setattr(gs, "engine_installed", lambda: False)
    monkeypatch.setattr(gs, "engine_version", lambda: None)
    monkeypatch.setattr(gs.gb_synapse_backends, "_torch_env_dir", lambda: None)

    s = gs.status()

    assert s["server_running"] is True
    assert s["proxy_running"] is True
    assert s["engines_running"] == ["torch"]


def test_status_server_running_false_when_nothing_running(run_dir, fake_alive, monkeypatch):
    monkeypatch.setattr(gs, "engine_installed", lambda: False)
    monkeypatch.setattr(gs, "engine_version", lambda: None)
    monkeypatch.setattr(gs.gb_synapse_backends, "_torch_env_dir", lambda: None)

    s = gs.status()

    assert s["server_running"] is False
    assert s["proxy_running"] is False
    assert s["engines_running"] == []


# ---------------------------------------------------------------------------
# serve()'s proxy-only restart
# ---------------------------------------------------------------------------

def test_serve_restarts_only_proxy_when_engine_alive_proxy_dead(run_dir, fake_alive, monkeypatch):
    st = _write_state(run_dir, model="qwen3", llama_pid=100, proxy_pid=200,
                       port=11435, internal_port=12435, engine="llama.cpp")
    fake_alive[100] = True   # engine still alive
    fake_alive[200] = False  # proxy died

    monkeypatch.setattr(gs, "_resolve_model", lambda spec: gs.ModelEntry(name="qwen3", path="/x"))
    calls = {}

    def _fake_start_proxy(entry, port, internal_port, engine):
        calls["args"] = (entry.name, port, internal_port, engine)
        class _P:
            pid = 999
        return _P(), None

    monkeypatch.setattr(gs, "_start_proxy", _fake_start_proxy)

    def _boom_select_backend(entry):
        raise AssertionError("must not spawn a fresh backend/engine when the "
                              "existing engine is still alive")
    monkeypatch.setattr(gs.gb_synapse_backends, "select_backend", _boom_select_backend)

    result = gs.serve("qwen3")

    # Reused the EXISTING engine's ports, not a freshly-requested one.
    assert calls["args"] == ("qwen3", 11435, 12435, "llama.cpp")
    assert result.proxy_pid == 999
    assert result.llama_pid == 100  # engine pid untouched — no second engine
    # And the persisted run-state reflects the new proxy pid.
    reloaded = gs._read_run_states()[0]
    assert reloaded.proxy_pid == 999


def test_serve_reuses_state_when_both_alive(run_dir, fake_alive, monkeypatch):
    _write_state(run_dir, model="qwen3", llama_pid=100, proxy_pid=200)
    fake_alive[100] = True
    fake_alive[200] = True
    monkeypatch.setattr(gs, "_resolve_model", lambda spec: gs.ModelEntry(name="qwen3", path="/x"))

    def _boom(entry):
        raise AssertionError("must not select a backend when already fully running")
    monkeypatch.setattr(gs.gb_synapse_backends, "select_backend", _boom)

    result = gs.serve("qwen3")

    assert result.llama_pid == 100
    assert result.proxy_pid == 200


def test_serve_spawns_fresh_backend_when_both_dead(run_dir, fake_alive, monkeypatch):
    _write_state(run_dir, model="qwen3", llama_pid=100, proxy_pid=200)
    fake_alive[100] = False
    fake_alive[200] = False
    monkeypatch.setattr(gs, "_resolve_model", lambda spec: gs.ModelEntry(name="qwen3", path="/x"))

    called = {}

    class _FakeBackend:
        def serve(self, entry, port, **kw):
            called["yes"] = True
            return gs.ServerState(model=entry.name, llama_pid=999, proxy_pid=888,
                                  port=port, internal_port=1, tensor_split="")

    monkeypatch.setattr(gs.gb_synapse_backends, "select_backend", lambda entry: _FakeBackend())

    result = gs.serve("qwen3")

    assert called.get("yes") is True
    assert result.llama_pid == 999


# ---------------------------------------------------------------------------
# stop()'s feeder teardown
# ---------------------------------------------------------------------------

class _FakeFeeder:
    def __init__(self, ip):
        self.ip = ip
        self.ssh_user = "root"


def test_stop_kills_gllm_slave_for_torch_engine(run_dir, fake_alive, monkeypatch):
    st = _write_state(run_dir, model="qwen3", llama_pid=100, proxy_pid=200,
                       engine="torch", feeders=["10.0.0.5"])
    fake_alive[100] = True
    fake_alive[200] = True
    monkeypatch.setattr(gs, "_kill_process_group", lambda pid: None)
    monkeypatch.setattr(gs.gb_cluster, "feeders", lambda probe=False: [_FakeFeeder("10.0.0.5")])

    killed = []
    monkeypatch.setattr(gs, "kill_feeder_gllm_slave", lambda feeder: killed.append(feeder.ip))

    assert gs.stop("qwen3") is True
    assert killed == ["10.0.0.5"]


def test_stop_skips_rpc_server_still_used_by_another_serve(run_dir, fake_alive, monkeypatch):
    """Two llama.cpp serves share feeder 10.0.0.5's rpc-server; stopping one
    must not kill it out from under the other."""
    _write_state(run_dir, model="qwen3-a", llama_pid=100, proxy_pid=200,
                 engine="llama.cpp", feeders=["10.0.0.5"])
    _write_state(run_dir, model="qwen3-b", llama_pid=101, proxy_pid=201,
                 engine="llama.cpp", feeders=["10.0.0.5"])
    fake_alive.update({100: True, 200: True, 101: True, 201: True})
    monkeypatch.setattr(gs, "_kill_process_group", lambda pid: None)
    monkeypatch.setattr(gs.gb_cluster, "feeders", lambda probe=False: [_FakeFeeder("10.0.0.5")])

    ssh_calls = []
    monkeypatch.setattr(gs.subprocess, "run",
                        lambda *a, **k: ssh_calls.append(a) or type("R", (), {"returncode": 0})())

    assert gs.stop("qwen3-a") is True
    assert ssh_calls == []  # the other serve still needs it — must not pkill


def test_stop_kills_rpc_server_when_last_serve_using_it(run_dir, fake_alive, monkeypatch):
    _write_state(run_dir, model="qwen3", llama_pid=100, proxy_pid=200,
                 engine="llama.cpp", feeders=["10.0.0.5"])
    fake_alive.update({100: True, 200: True})
    monkeypatch.setattr(gs, "_kill_process_group", lambda pid: None)
    monkeypatch.setattr(gs.gb_cluster, "feeders", lambda probe=False: [_FakeFeeder("10.0.0.5")])

    ssh_calls = []

    def _fake_run(cmd, **kw):
        ssh_calls.append(cmd)
        return type("R", (), {"returncode": 0})()
    monkeypatch.setattr(gs.subprocess, "run", _fake_run)

    assert gs.stop("qwen3") is True
    assert len(ssh_calls) == 1
    assert "rpc-server" in ssh_calls[0][-1]

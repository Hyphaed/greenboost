#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_synapse.py's P7 facade-backing additions: wait_ready(),
pull_model(), serve_gguf(), endpoints().
"""
import time
import urllib.request

import pytest

import gb_synapse as gs


# ---------------------------------------------------------------------------
# wait_ready()
# ---------------------------------------------------------------------------

def test_wait_ready_true_on_first_200(monkeypatch):
    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=5: _Resp())
    assert gs.wait_ready("http://x") is True


def test_wait_ready_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(url, timeout=5):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("not listening yet")
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(gs.time, "sleep", lambda s: None)
    assert gs.wait_ready("http://x", timeout_s=10.0) is True
    assert calls["n"] == 3


def test_wait_ready_false_on_timeout(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda url, timeout=5: (_ for _ in ()).throw(OSError("refused")))
    monkeypatch.setattr(gs.time, "sleep", lambda s: None)
    real_time = gs.time.time
    calls = {"n": 0}

    def _fake_time():
        calls["n"] += 1
        return real_time() + calls["n"] * 100  # advance fast past the deadline
    monkeypatch.setattr(gs.time, "time", _fake_time)
    assert gs.wait_ready("http://x", timeout_s=1.0) is False


def test_wait_ready_false_when_attempts_exhausted(monkeypatch):
    calls = {"n": 0}

    def _fake_urlopen(url, timeout=5):
        calls["n"] += 1
        raise OSError("refused")
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(gs.time, "sleep", lambda s: None)
    assert gs.wait_ready("http://x", timeout_s=1000.0, attempts=3) is False
    assert calls["n"] == 3


def test_wait_ready_non_200_status_keeps_polling(monkeypatch):
    class _Resp503:
        status = 503

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Resp200:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    calls = {"n": 0}

    def _fake_urlopen(url, timeout=5):
        calls["n"] += 1
        return _Resp503() if calls["n"] == 1 else _Resp200()
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(gs.time, "sleep", lambda s: None)
    assert gs.wait_ready("http://x") is True
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# pull_model()
# ---------------------------------------------------------------------------

def test_pull_model_calls_pull_and_progress_callbacks(monkeypatch):
    monkeypatch.setattr(gs, "pull", lambda name: gs.ModelEntry(name=name, path="/x"))
    events = []
    entry = gs.pull_model("org/repo", progress=lambda ev, name: events.append((ev, name)))
    assert entry.name == "org/repo"
    assert events == [("start", "org/repo"), ("done", "org/repo")]


def test_pull_model_no_progress_callback_required(monkeypatch):
    monkeypatch.setattr(gs, "pull", lambda name: gs.ModelEntry(name=name, path="/x"))
    entry = gs.pull_model("org/repo")
    assert entry.name == "org/repo"


# ---------------------------------------------------------------------------
# serve_gguf()
# ---------------------------------------------------------------------------

class _FakeProc:
    pid = 5555


def test_serve_gguf_raises_when_engine_not_built(monkeypatch):
    monkeypatch.setattr(gs, "engine_installed", lambda: False)
    with pytest.raises(RuntimeError, match="engine not built"):
        gs.serve_gguf("/models/x.gguf", 8081)


def test_serve_gguf_builds_expected_command(monkeypatch, tmp_path):
    monkeypatch.setattr(gs, "engine_installed", lambda: True)
    monkeypatch.setattr(gs, "ENGINE_DIR", tmp_path)
    monkeypatch.setattr(gs, "_run_log_path", lambda label: tmp_path / f"{label}.log")
    captured = {}

    def _fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        return _FakeProc()
    monkeypatch.setattr(gs.subprocess, "Popen", _fake_popen)

    result = gs.serve_gguf("/models/vision.gguf", 8081, mmproj="/models/mmproj.gguf", ctx=4096)

    assert result == {"pid": 5555, "port": 8081}
    cmd = captured["cmd"]
    assert "-m" in cmd and cmd[cmd.index("-m") + 1] == "/models/vision.gguf"
    assert "--port" in cmd and cmd[cmd.index("--port") + 1] == "8081"
    assert "--mmproj" in cmd and cmd[cmd.index("--mmproj") + 1] == "/models/mmproj.gguf"
    assert "--ctx-size" in cmd and cmd[cmd.index("--ctx-size") + 1] == "4096"


def test_serve_gguf_omits_optional_flags_when_not_given(monkeypatch, tmp_path):
    monkeypatch.setattr(gs, "engine_installed", lambda: True)
    monkeypatch.setattr(gs, "ENGINE_DIR", tmp_path)
    monkeypatch.setattr(gs, "_run_log_path", lambda label: tmp_path / f"{label}.log")
    captured = {}
    monkeypatch.setattr(gs.subprocess, "Popen",
                        lambda cmd, **kw: captured.update(cmd=cmd) or _FakeProc())

    gs.serve_gguf("/models/x.gguf", 8081)

    cmd = captured["cmd"]
    assert "--mmproj" not in cmd
    assert "--ctx-size" not in cmd


# ---------------------------------------------------------------------------
# endpoints()
# ---------------------------------------------------------------------------

def test_endpoints_delegates_to_gb_actuation_read_env_file(monkeypatch):
    import gb_actuation
    monkeypatch.setattr(gb_actuation, "_read_env_file",
                        lambda: {"FORGE_OLLAMA_URL": "http://127.0.0.1:11435"})
    assert gs.endpoints() == {"FORGE_OLLAMA_URL": "http://127.0.0.1:11435"}


def test_endpoints_never_raises_when_gb_actuation_unavailable(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **kw):
        if name == "gb_actuation":
            raise ImportError("simulated")
        return real_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert gs.endpoints() == {}

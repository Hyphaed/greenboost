"""greenboost-cli must release the GreenBoost memory pool however it exits.

Owner requirement, 2026-08-18: `greenboost clear memory-pool` runs on every
termination — /exit, /quit, Ctrl-D, SIGTERM, and SIGHUP when the terminal
window is closed out from under the process.

Two properties make this safe to put on an exit path, and both are asserted
here rather than assumed: it never blocks on a password (sudo is only ever
invoked with -n), and it never raises.
"""
from __future__ import annotations

import os

import pytest

from greenboost_cli.terminal import repl


@pytest.fixture(autouse=True)
def _reset_guard():
    repl._pool_released.clear()
    yield
    repl._pool_released.clear()


def _capture(monkeypatch):
    calls = []

    class _R:
        returncode = 0
        stdout = "Memory pool clear complete.\n"
        stderr = ""

    import subprocess
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: (calls.append((cmd, kw)), _R())[1])
    monkeypatch.setattr(repl.shutil, "which", lambda n: "/usr/local/bin/greenboost")
    return calls


def test_sudo_is_only_ever_non_interactive(monkeypatch):
    """A password prompt on exit would hang the terminal the user is closing."""
    calls = _capture(monkeypatch)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    repl.release_memory_pool("/exit")
    sudo_calls = [c for c, _ in calls if c[0] == "sudo"]
    assert sudo_calls, "should attempt the privileged clear"
    for c in sudo_calls:
        assert "-n" in c, f"sudo must be non-interactive, got {c}"


def test_every_invocation_is_bounded_by_a_timeout(monkeypatch):
    calls = _capture(monkeypatch)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    repl.release_memory_pool("/exit")
    for _, kw in calls:
        assert kw.get("timeout"), "an exit path must not wait forever"


def test_runs_directly_when_already_root(monkeypatch):
    calls = _capture(monkeypatch)
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    repl.release_memory_pool("SIGHUP")
    assert all(c[0] != "sudo" for c, _ in calls)


def test_success_is_silent(monkeypatch, capsys):
    """It runs on every exit route, so a confirmation line would be noise on
    the clean terminal /exit has just produced."""
    _capture(monkeypatch)
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    repl.release_memory_pool("/exit")
    assert capsys.readouterr().out == ""


def test_lack_of_root_is_also_silent(monkeypatch, capsys):
    """Owner decision: without root the GPU side is still released and the T2
    DDR is reclaimed by the kernel later under pressure. Not worth a message."""
    import subprocess

    class _Fail:
        returncode = 1
        stdout = stderr = ""

    class _Ok:
        returncode = 0
        stdout = "  (run as root for kernel-level buffer release)\n"
        stderr = ""

    seq = [_Fail(), _Ok()]
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: seq.pop(0))
    monkeypatch.setattr(repl.shutil, "which", lambda n: "/usr/local/bin/greenboost")
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    repl.release_memory_pool("/exit")
    assert capsys.readouterr().out == ""


def test_a_real_failure_is_reported(monkeypatch, capsys):
    import subprocess

    class _Fail:
        returncode = 1
        stdout = stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: _Fail())
    monkeypatch.setattr(repl.shutil, "which", lambda n: "/usr/local/bin/greenboost")
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    repl.release_memory_pool("/exit")
    out = capsys.readouterr().out
    assert "could not release the memory pool" in out
    assert "clear memory-pool" in out, "must say how to do it by hand"


def test_runs_at_most_once_per_process(monkeypatch):
    """/exit calls it directly and atexit calls it again; the pool must not be
    cleared twice."""
    calls = _capture(monkeypatch)
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    repl.release_memory_pool("/exit")
    repl.release_memory_pool("process exit")
    assert len(calls) == 1


def test_never_raises_when_the_binary_is_missing(monkeypatch):
    monkeypatch.setattr(repl.shutil, "which", lambda n: None)
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    repl.release_memory_pool("/exit")   # must not raise


def test_never_raises_when_the_subprocess_explodes(monkeypatch):
    import subprocess

    def _boom(cmd, **kw):
        raise OSError("fork failed")

    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(repl.shutil, "which", lambda n: "/usr/local/bin/greenboost")
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    repl.release_memory_pool("/exit")   # must not raise


def test_opt_out_env_var_skips_it(monkeypatch):
    calls = _capture(monkeypatch)
    monkeypatch.setenv("GB_CLI_NO_POOL_RELEASE", "1")
    repl.release_memory_pool("/exit")
    assert calls == []


def test_sigint_is_not_hijacked():
    """Ctrl-C already means 'cancel the turn'. Stealing it for shutdown would
    break that, so the handlers cover SIGTERM and SIGHUP only."""
    import inspect
    src = inspect.getsource(repl._install_termination_handlers)
    assert "SIGHUP" in src and "SIGTERM" in src
    assert 'signal.signal(getattr(signal, "SIGINT"' not in src
    assert '"SIGINT"' not in src.split("for sig in", 1)[1]

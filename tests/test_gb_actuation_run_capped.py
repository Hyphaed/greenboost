#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_actuation.run_capped() (P7) , the systemd-run --user wrapper
encoding 4 hard-won facts ai-forge's own version (forge/runners/longlive.py)
discovered the hard way: the env-delivery gap (--setenv only, not
subprocess env=), the HF_TOKEN /proc/cmdline leak via --setenv, MemoryMax
needing headroom above the pinned T2 pool, and cwd needing an explicit
--working-directory.

No real systemd-run: subprocess.run is monkeypatched to capture the built
command instead of executing it.
"""
import os
import subprocess

import pytest

import gb_actuation as ga


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="ok", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def captured_cmd(monkeypatch):
    captured = {}

    def _fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["kwargs"] = kw
        return _FakeCompleted()
    monkeypatch.setattr(ga.subprocess, "run", _fake_run)
    return captured


@pytest.fixture
def no_t2_pool(monkeypatch):
    """Most test environments have no gb_tiering/kmod state , make
    mem_max_mb's auto-derivation deterministic (uncapped) unless a test
    overrides it."""
    import gb_tiering
    monkeypatch.setattr(gb_tiering, "t2_pool", lambda: {"total_mb": 0})


def test_run_capped_uses_wait_collect_pipe(captured_cmd, no_t2_pool):
    ga.run_capped(["true"])
    cmd = captured_cmd["cmd"]
    assert cmd[0] == "systemd-run"
    assert "--user" in cmd
    assert "--wait" in cmd
    assert "--collect" in cmd
    assert "--pipe" in cmd


def test_run_capped_sets_explicit_working_directory(captured_cmd, no_t2_pool):
    ga.run_capped(["true"], cwd="/tmp/somewhere")
    cmd = captured_cmd["cmd"]
    assert "--working-directory=/tmp/somewhere" in cmd


def test_run_capped_defaults_cwd_to_getcwd_not_home(captured_cmd, no_t2_pool, monkeypatch):
    monkeypatch.setattr(os, "getcwd", lambda: "/current/dir")
    ga.run_capped(["true"])
    cmd = captured_cmd["cmd"]
    assert "--working-directory=/current/dir" in cmd


def test_run_capped_secrets_use_environment_file_not_setenv(captured_cmd, no_t2_pool, tmp_path):
    ga.run_capped(["true"], env={"HF_TOKEN": "hf_secret123"}, secrets_file=True)
    cmd = captured_cmd["cmd"]
    # The secret must NEVER appear as a literal --setenv (which would leak
    # into /proc/pid/cmdline for every local user).
    assert not any("HF_TOKEN=hf_secret123" in c for c in cmd if c.startswith("--setenv"))
    env_file_arg = next(c for c in cmd if c.startswith("--property=EnvironmentFile="))
    path = env_file_arg.split("=", 2)[2]
    # File must not survive the call (cleaned up in the `finally`).
    assert not os.path.exists(path)


def test_run_capped_environment_file_is_0600_while_it_exists(monkeypatch, no_t2_pool):
    captured_path = {}
    real_run = subprocess.run

    def _fake_run(cmd, **kw):
        env_file_arg = next(c for c in cmd if c.startswith("--property=EnvironmentFile="))
        path = env_file_arg.split("=", 2)[2]
        captured_path["mode"] = oct(os.stat(path).st_mode & 0o777)
        return _FakeCompleted()
    monkeypatch.setattr(ga.subprocess, "run", _fake_run)

    ga.run_capped(["true"], env={"HF_TOKEN": "x"})
    assert captured_path["mode"] == "0o600"


def test_run_capped_setenv_mode_when_secrets_file_false(captured_cmd, no_t2_pool):
    ga.run_capped(["true"], env={"FOO": "bar"}, secrets_file=False)
    cmd = captured_cmd["cmd"]
    assert any(c == "--setenv=FOO=bar" for c in cmd)
    assert not any(c.startswith("--property=EnvironmentFile=") for c in cmd)


def test_run_capped_mem_max_derived_above_t2_pool(captured_cmd, monkeypatch):
    import gb_tiering
    monkeypatch.setattr(gb_tiering, "t2_pool", lambda: {"total_mb": 10000})
    ga.run_capped(["true"])
    cmd = captured_cmd["cmd"]
    mem_arg = next(c for c in cmd if c.startswith("--property=MemoryMax="))
    mem_mb = int(mem_arg.split("=", 2)[2].rstrip("M"))
    assert mem_mb > 10000  # headroom ABOVE the pinned T2 pool, never at/below it


def test_run_capped_uncapped_when_t2_pool_zero(captured_cmd, no_t2_pool):
    ga.run_capped(["true"])
    cmd = captured_cmd["cmd"]
    assert not any(c.startswith("--property=MemoryMax=") for c in cmd)


def test_run_capped_explicit_mem_max_overrides_auto_derivation(captured_cmd, no_t2_pool):
    ga.run_capped(["true"], mem_max_mb=5000)
    cmd = captured_cmd["cmd"]
    assert "--property=MemoryMax=5000M" in cmd


def test_run_capped_returns_completed_process_shape(no_t2_pool, monkeypatch):
    monkeypatch.setattr(ga.subprocess, "run",
                        lambda cmd, **kw: _FakeCompleted(returncode=0, stdout="hi", stderr=""))
    result = ga.run_capped(["true"])
    assert result == {"returncode": 0, "stdout": "hi", "stderr": ""}


def test_run_capped_cleans_up_env_file_even_on_subprocess_exception(no_t2_pool, monkeypatch):
    written_path = {}

    def _boom(cmd, **kw):
        env_file_arg = next(c for c in cmd if c.startswith("--property=EnvironmentFile="))
        written_path["path"] = env_file_arg.split("=", 2)[2]
        raise subprocess.TimeoutExpired(cmd, 1)
    monkeypatch.setattr(ga.subprocess, "run", _boom)

    with pytest.raises(subprocess.TimeoutExpired):
        ga.run_capped(["true"], env={"X": "1"})

    assert not os.path.exists(written_path["path"])

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for gb_synapse._kill_process_group()/stop(): SIGTERM the whole
process group an engine leads, not just the tracked PID — real leak found
live 2026-07-16 (gLLM's detached multiprocessing worker survived a plain
os.kill on the tracked llama_pid, holding ~10 GB of VRAM after stop()
reported success).

CPU-only. No CUDA, no real subprocesses — os.killpg/os.kill are
monkeypatched."""
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gb_synapse as gs


def test_kill_process_group_uses_killpg(monkeypatch):
    calls = []
    monkeypatch.setattr(gs.os, "getpgid", lambda pid: 4242)
    monkeypatch.setattr(gs.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    gs._kill_process_group(1234)
    assert calls == [(4242, signal.SIGTERM)]


def test_kill_process_group_falls_back_to_single_pid_on_esrch(monkeypatch):
    def _raise_esrch(pid):
        raise OSError("No such process")

    calls = []
    monkeypatch.setattr(gs.os, "getpgid", _raise_esrch)
    monkeypatch.setattr(gs.os, "kill", lambda pid, sig: calls.append((pid, sig)))
    gs._kill_process_group(1234)
    assert calls == [(1234, signal.SIGTERM)]


def test_kill_process_group_never_raises_when_both_fail(monkeypatch):
    monkeypatch.setattr(gs.os, "getpgid", lambda pid: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(gs.os, "kill", lambda pid, sig: (_ for _ in ()).throw(OSError()))
    gs._kill_process_group(1234)  # must not raise

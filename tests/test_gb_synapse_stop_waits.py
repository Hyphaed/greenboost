#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""stop() must not return while the old server still holds the card.

Measured 2026-10-30: a restart planned against 3.1 and 6.9 GB free on a card that frees to 11.5 GB, so the
GPU expert cache looked impossible and the model ran at ~54 tok/s instead of 122-138."""
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gb_synapse as gs


class _Clock:
    def __init__(self): self.t = 1000.0
    def time(self): return self.t
    def sleep(self, s): self.t += s


def _patch(monkeypatch, alive_polls, free_series):
    clock = _Clock()
    monkeypatch.setattr(gs.time, "time", clock.time)
    monkeypatch.setattr(gs.time, "sleep", clock.sleep)
    polls = {"n": 0}

    def alive(pid):
        polls["n"] += 1
        return polls["n"] <= alive_polls
    monkeypatch.setattr(gs, "_pid_alive", alive)
    frees = iter(free_series)
    last = {"v": free_series[-1]}

    def mem():
        try:
            last["v"] = next(frees)
        except StopIteration:
            pass
        return (0, last["v"], 12227, 0)
    import gb_nvml
    monkeypatch.setattr(gb_nvml, "get_nvml", lambda i=0: types.SimpleNamespace(mem=mem))
    return clock, polls


def test_it_waits_for_the_process_and_for_the_free_vram_to_stop_rising(monkeypatch):
    clock, polls = _patch(monkeypatch, alive_polls=6, free_series=[3100, 6900, 9800, 11500, 11520, 11510])
    waited = gs._wait_for_release([1234])
    assert waited >= 2.0                                  # not the instant return stop() used to make
    assert polls["n"] > 6                                 # polled until the process was gone
    assert clock.t - 1000.0 < 20                          # and nowhere near the timeout


def test_a_server_that_will_not_die_is_killed_after_twenty_seconds(monkeypatch):
    clock, polls = _patch(monkeypatch, alive_polls=10**6, free_series=[5000])
    killed = []
    monkeypatch.setattr(os, "getpgid", lambda p: p)
    monkeypatch.setattr(os, "killpg", lambda g, sig: killed.append((g, sig)))
    gs._wait_for_release([4321], timeout_s=30.0)
    import signal
    assert (4321, signal.SIGKILL) in killed


def test_stop_calls_the_wait_after_signalling():
    import inspect
    src = inspect.getsource(gs.stop)
    assert "_wait_for_release(_killed)" in src and src.index("_kill_process_group(pid)") < src.index("_wait_for_release(_killed)")


def test_a_failure_inside_the_wait_never_fails_a_stop(monkeypatch):
    monkeypatch.setattr(gs, "_pid_alive", lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
    assert gs._wait_for_release([1]) >= 0.0

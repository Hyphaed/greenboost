#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_synapse_api._StallWatch — the "HTTP already healthy, engine
hung forever" watchdog (workflow/gb-synapse.md's torch-core bring-up notes:
`/health` returned 200 but the first real request hung on a CUDA error
during prefill, so a client only ever saw a silent hang). No live aiohttp
server: exercises the watchdog coroutine directly via asyncio.run.
"""
import asyncio

import pytest

pytest.importorskip("aiohttp")

import gb_synapse_api as api


def test_stall_watch_fires_when_nothing_marked(monkeypatch):
    monkeypatch.setattr(api, "STALL_THRESHOLD_S", 0.01)
    monkeypatch.setattr(api, "ENGINE", "torch")
    captured = {}
    monkeypatch.setattr(
        "gb_synapse.emit_stall",
        lambda model, engine, elapsed: captured.update(
            model=model, engine=engine, elapsed=elapsed),
    )

    async def _run():
        async with api._StallWatch("qwen3") as watch:
            await asyncio.sleep(0.05)   # outlast the (shrunk) threshold, never mark()

    asyncio.run(_run())

    assert captured["model"] == "qwen3"
    assert captured["engine"] == "torch"
    # Real idle time at the moment the watchdog fired — not a hardcoded
    # STALL_THRESHOLD_S echo (the fixed bug: the watchdog used to always
    # report the threshold itself, not how long the stall had actually run).
    assert captured["elapsed"] >= 0.01


def test_stall_watch_refires_after_resuming_then_stalling_again(monkeypatch):
    """A stream that outputs, stalls, resumes, then stalls again must be
    flagged twice — the old single-shot design only ever fired once per
    request lifetime, so a hang *after* the first token was never caught."""
    monkeypatch.setattr(api, "STALL_THRESHOLD_S", 0.02)
    calls = []
    monkeypatch.setattr(
        "gb_synapse.emit_stall",
        lambda model, engine, elapsed: calls.append(elapsed),
    )

    async def _run():
        async with api._StallWatch("qwen3") as watch:
            await asyncio.sleep(0.05)   # first stall: fires once
            watch.mark()                # output resumes
            await asyncio.sleep(0.05)   # second stall: must fire again

    asyncio.run(_run())

    assert len(calls) == 2


def test_stall_watch_does_not_fire_when_repeatedly_marked_within_threshold(monkeypatch):
    """Continuous output (marks spaced closer together than the threshold)
    must never trip the watchdog — only real, sustained silence should."""
    monkeypatch.setattr(api, "STALL_THRESHOLD_S", 0.1)
    calls = []
    monkeypatch.setattr("gb_synapse.emit_stall",
                        lambda model, engine, elapsed: calls.append(model))

    async def _run():
        async with api._StallWatch("qwen3") as watch:
            for _ in range(5):
                watch.mark()
                await asyncio.sleep(0.03)   # well under the 0.1s threshold each time

    asyncio.run(_run())

    assert calls == []


def test_stall_watch_fires_for_silence_after_a_single_mark(monkeypatch):
    """A single mark() does not grant permanent immunity — real, sustained
    silence *after* that mark must still be flagged (the bug this replaced:
    the old watchdog set `_got_output` permanently on the first token, so a
    hang later in the same stream was never caught)."""
    monkeypatch.setattr(api, "STALL_THRESHOLD_S", 0.02)
    calls = []
    monkeypatch.setattr("gb_synapse.emit_stall",
                        lambda model, engine, elapsed: calls.append(model))

    async def _run():
        async with api._StallWatch("qwen3") as watch:
            watch.mark()               # first token arrives immediately
            await asyncio.sleep(0.06)  # then genuine silence past the threshold

    asyncio.run(_run())

    assert calls == ["qwen3"]


def test_stall_watch_does_not_fire_when_request_finishes_quickly(monkeypatch):
    """The common case: the request completes well before the threshold —
    __aexit__ must cancel the watchdog task, not let it fire after the
    fact."""
    monkeypatch.setattr(api, "STALL_THRESHOLD_S", 10.0)
    calls = []
    monkeypatch.setattr("gb_synapse.emit_stall",
                        lambda model, engine, elapsed: calls.append(model))

    async def _run():
        async with api._StallWatch("qwen3") as watch:
            watch.mark()
        # __aexit__ already cancelled the background task by this point.

    asyncio.run(_run())

    assert calls == []

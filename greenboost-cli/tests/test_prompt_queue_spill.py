"""An accepted prompt is either consumed or reported, never silently dropped.

Measured 2026-08-22 on a real session: "build it inside a folder named
pinball_aug22_1530" was queued while turn 1 was still exploring. The session
ended before that turn finished, the in-memory queue went with it, and the work
went into pinball_aug19_1320 with nothing said.

The drain logic was never at fault , it dequeues correctly after every turn.
The turn never finished, because TTFT ran 20-107 s per turn on that session
(casebook C-20). So the fix is not to the drain; it is that exit must account
for what it is throwing away.
"""
import importlib
import json
import os
import tempfile

import pytest


@pytest.fixture()
def pq(monkeypatch):
    monkeypatch.setenv("GB_CLI_QUEUE_SPILL",
                       os.path.join(tempfile.mkdtemp(), "unconsumed.json"))
    import greenboost_cli.terminal.prompt_queue as m
    return importlib.reload(m)


def test_empty_queue_spills_nothing(pq):
    assert pq.spill_unconsumed(pq.PromptQueue()) == []
    assert not pq._SPILL.exists()


def test_unconsumed_prompts_are_returned_for_reporting(pq):
    q = pq.PromptQueue()
    q.enqueue("build it inside a folder named pinball_aug22_1530")
    assert pq.spill_unconsumed(q) == ["build it inside a folder named pinball_aug22_1530"]


def test_unconsumed_prompts_are_persisted(pq):
    q = pq.PromptQueue()
    q.enqueue("first")
    q.enqueue("second")
    pq.spill_unconsumed(q, "a session")
    rec = json.loads(pq._SPILL.read_text())[-1]
    assert rec["prompts"] == ["first", "second"]
    assert rec["session"] == "a session"


def test_spills_accumulate_across_sessions(pq):
    for name in ("s1", "s2"):
        q = pq.PromptQueue()
        q.enqueue(name)
        pq.spill_unconsumed(q, name)
    assert len(json.loads(pq._SPILL.read_text())) == 2


def test_spill_file_is_bounded(pq):
    for i in range(25):
        q = pq.PromptQueue()
        q.enqueue(f"p{i}")
        pq.spill_unconsumed(q, f"s{i}")
    assert len(json.loads(pq._SPILL.read_text())) == 20


def test_a_consumed_queue_spills_nothing(pq):
    """The normal path: everything was drained, so exit reports nothing."""
    q = pq.PromptQueue()
    q.enqueue("done")
    assert q.dequeue().text == "done"
    assert pq.spill_unconsumed(q) == []


def test_unwritable_spill_path_never_raises(pq, monkeypatch):
    """Losing the spill file must not also break the exit path."""
    monkeypatch.setattr(pq, "_SPILL", pq.Path("/proc/cannot/write/here.json"))
    q = pq.PromptQueue()
    q.enqueue("x")
    assert pq.spill_unconsumed(q) == ["x"]      # still reported to the user

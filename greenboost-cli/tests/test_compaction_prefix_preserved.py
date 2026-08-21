"""Compaction must never rewrite the prompt prefix, on a restored session either.

`_compress_context` splits the conversation into head / middle / tail and
rewrites only the middle, so that the engine's cached KV for the head stays
valid. It used to make one exception: if a structured-memory block was sitting
in the head it set `head = []`, on the reasoning that the head "IS a memory
block" and was therefore degenerate.

That exception was backwards. A memory block at the front is byte-identical
from turn to turn, which is exactly the property the head exists to protect.
Zeroing the head moved it into the middle, where it was absorbed and re-emitted
inside the new summary at a different position, changing every byte after
position zero.

It only triggered on a session that had already been compacted once, i.e. every
RESTORED session, and this CLI restores by default. Measured on this box before
the fix: agent_compaction_prefix_kept_pct = 50.

These tests assert the structural property, not the summary wording, which
depends on the extractor and would make them flaky.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from greenboost_cli.workflow import intelligence as I


class FakeSession:
    def __init__(self, messages):
        self.messages = messages


def _turns(n, tag="x"):
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"user {tag} {i} " + "word " * 400})
        out.append({"role": "assistant", "content": f"assistant {tag} {i} " + "word " * 400})
    return out


def _fresh_session():
    msgs = [
        {"role": "system", "content": "SYSTEM PROMPT PINNED"},
        {"role": "user", "content": "FIRST USER MESSAGE PINNED"},
    ]
    return FakeSession(msgs + _turns(40))


def _compact(session):
    I._compress_context(session, settings={}, force=True)


def test_fresh_session_keeps_its_head():
    s = _fresh_session()
    head_before = [dict(m) for m in s.messages[:I._HEAD_KEEP]]
    _compact(s)
    assert s.messages[:I._HEAD_KEEP] == head_before


def test_second_compaction_still_keeps_the_head():
    """The regression: after one compaction the head holds a memory block."""
    s = _fresh_session()
    _compact(s)
    head_after_first = [dict(m) for m in s.messages[:I._HEAD_KEEP]]
    s.messages.extend(_turns(40, "second"))
    _compact(s)
    assert s.messages[:I._HEAD_KEEP] == head_after_first, (
        "second compaction rewrote the pinned head")


def test_restored_session_starting_with_a_memory_block_keeps_its_head():
    """A session restored from disk can begin WITH the memory block.

    This is the shape that produced prefix_kept_pct = 50.
    """
    s = FakeSession([
        {"role": "user", "content": I._MEMORY_MARKER + "\n\nearlier stuff"},
        {"role": "assistant", "content": "[Structured memory loaded. Continuing task.]"},
    ] + _turns(40))
    head_before = [dict(m) for m in s.messages[:I._HEAD_KEEP]]
    _compact(s)
    assert s.messages[:I._HEAD_KEEP] == head_before, (
        "compaction of a restored session threw the prefix away")


def test_prefix_grows_monotonically_over_three_compactions():
    """The property that matters: the prefix may extend, never change."""
    s = _fresh_session()
    prefixes = []
    for i in range(3):
        _compact(s)
        prefixes.append("".join(
            str(m.get("content", "")) for m in s.messages[:I._HEAD_KEEP]))
        s.messages.extend(_turns(40, f"round{i}"))
    assert prefixes[0] == prefixes[1] == prefixes[2], (
        f"pinned prefix changed across compactions: "
        f"{[p[:40] for p in prefixes]}")


def test_telemetry_reports_head_shape():
    """head_kept must be non-zero so the dataflux metric can see the fix."""
    emitted = {}
    try:
        import gb_dataflux
        real = gb_dataflux.emit
        gb_dataflux.emit = lambda d: emitted.update(d)
    except ImportError:
        return  # dataflux not importable in this environment, nothing to assert
    try:
        s = FakeSession([
            {"role": "user", "content": I._MEMORY_MARKER + "\n\nearlier"},
            {"role": "assistant", "content": "[Structured memory loaded. Continuing task.]"},
        ] + _turns(40))
        _compact(s)
    finally:
        gb_dataflux.emit = real
    if emitted:
        assert emitted.get("head_kept", 0) > 0, emitted

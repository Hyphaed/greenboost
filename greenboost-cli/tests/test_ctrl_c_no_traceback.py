"""Ctrl-C must never dump a traceback and kill the session.

Reported 2026-08-18. The first Ctrl-C printed the intended
"Cancelling…  (stops at next token — Ctrl-C again to see status)". The second
produced:

    File ".../greenboost_cli/terminal/repl.py", line 1582, in run_interactive
        ev, data = _stdin_q.get(timeout=0.5)
    File "/usr/lib/python3.14/queue.py", line 210, in get
        self.not_empty.wait(remaining)
    File "/usr/lib/python3.14/threading.py", line 373, in wait
        gotit = waiter.acquire(True, timeout)
    KeyboardInterrupt

and the process died mid-turn. The interrupt design routes Ctrl-C through the
stdin reader thread as an ("interrupt", None) queue event, but Python delivers
SIGINT to the MAIN thread, which is parked in that get(). The loop caught only
queue.Empty, so the user did exactly what the message told them to and lost
the session.
"""
from __future__ import annotations

import queue

import pytest

from greenboost_cli.terminal.repl import next_stdin_event


class _RaisingQueue:
    def __init__(self, exc):
        self._exc = exc

    def get(self, timeout=None):
        raise self._exc


def test_keyboard_interrupt_becomes_an_interrupt_event():
    """The whole point: SIGINT on the main thread reaches the same branch the
    reader thread's event would have."""
    ev, data = next_stdin_event(_RaisingQueue(KeyboardInterrupt()))
    assert ev == "interrupt"
    assert data is None


def test_idle_timeout_is_distinguishable_from_an_event():
    ev, data = next_stdin_event(_RaisingQueue(queue.Empty()))
    assert ev is None and data is None


def test_a_real_event_passes_through_untouched():
    q = queue.Queue()
    q.put(("line", "hello world"))
    assert next_stdin_event(q) == ("line", "hello world")


def test_eof_still_reaches_the_loop():
    q = queue.Queue()
    q.put(("eof", None))
    assert next_stdin_event(q) == ("eof", None)


def test_other_exceptions_are_not_swallowed():
    """Only Empty and KeyboardInterrupt are ours. Hiding anything else would
    turn a real fault into an infinite quiet loop."""
    with pytest.raises(RuntimeError):
        next_stdin_event(_RaisingQueue(RuntimeError("queue broke")))


def test_the_loop_uses_the_helper_rather_than_a_bare_get():
    """Pin the call site: reverting to `_stdin_q.get(timeout=0.5)` inline would
    silently restore the traceback."""
    from pathlib import Path
    import greenboost_cli.terminal.repl as repl_mod
    src = Path(repl_mod.__file__).read_text()
    loop = src.split("# ── Main event loop", 1)[1][:400]
    assert "next_stdin_event(" in loop
    assert "_stdin_q.get(" not in loop


# ── cancel messaging ──────────────────────────────────────────────────────────

def test_first_cancel_during_decode_promises_a_quick_stop():
    from greenboost_cli.terminal.repl import cancel_message
    msg = cancel_message(None, still_prefilling=False)
    assert "next token" in msg
    assert "prompt" not in msg


def test_first_cancel_during_prefill_says_why_it_will_take_a_while():
    """Cancellation lands on the next yielded token. During prefill there is
    no token yet, and on a deep conversation the first one is tens of seconds
    away — 5.4 s at 13.8k prompt tokens rising to 34.2 s at 18.5k, measured
    2026-08-18. Promising "stops at next token" and then going silent for half
    a minute reads as a second bug."""
    from greenboost_cli.terminal.repl import cancel_message
    msg = cancel_message(None, still_prefilling=True)
    assert "reading the prompt" in msg
    assert "first token" in msg


def test_repeat_cancel_reports_how_long_it_has_been_pending():
    from greenboost_cli.terminal.repl import cancel_message
    assert "12s" in cancel_message(12.4, still_prefilling=False)
    assert "31s" in cancel_message(31.0, still_prefilling=True)


def test_repeat_cancel_distinguishes_prefill_from_decode():
    from greenboost_cli.terminal.repl import cancel_message
    assert "no token to stop at yet" in cancel_message(20.0, still_prefilling=True)
    assert "next token" in cancel_message(20.0, still_prefilling=False)

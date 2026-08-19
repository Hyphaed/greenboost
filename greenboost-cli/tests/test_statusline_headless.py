"""The animated statusline must not repaint when stdout is not a terminal.

Regression for a real headless-output blowup (2026-08-17): `StatusLine.start()`
spawned its animation thread unconditionally, repainting via a bare ``\\r`` every
0.08 s. On a TTY each frame overwrites the previous one; through a pipe the
carriage returns are just bytes, so every frame ACCUMULATES. One measured 72-second
`gb -p` turn emitted 259,358 bytes of spinner frames wrapping ~200 bytes of answer.

That is not merely ugly. The AI Factory captures agent output and feeds gate/retry
text back into the next prompt, so unbounded spinner bytes eat the scarce context
budget of a slow local model — the exact resource the factory is trying to conserve.
"""
from __future__ import annotations

import io
import sys
import time

from greenboost_cli.terminal.statusline import StatusLine


def _run_turn(seconds: float = 0.7) -> str:
    """Drive a statusline for `seconds` with stdout redirected to a buffer
    (a StringIO is not a tty, which is exactly the condition under test)."""
    buf = io.StringIO()
    real, sys.stdout = sys.stdout, buf
    try:
        sl = StatusLine()
        sl.start("Thinking")
        time.sleep(seconds)
        sl.update(in_tokens=4700, out_tokens=12)
        sl.stop()
    finally:
        sys.stdout = real
    return buf.getvalue()


def test_headless_output_is_a_single_line() -> None:
    out = _run_turn()
    assert out.count("\n") == 1, f"expected one summary line, got {out.count(chr(10))}"


def test_headless_emits_no_carriage_returns() -> None:
    # \r is the animation mechanism; its presence means frames are being repainted
    # into a stream that cannot overwrite them.
    assert "\r" not in _run_turn()


def test_headless_output_stays_small() -> None:
    # Pre-fix this was ~3.5 KB for 0.7 s and grew linearly with turn duration.
    # A single summary line is a few hundred bytes and is duration-independent.
    assert len(_run_turn()) < 1024


def test_headless_output_does_not_grow_with_turn_duration() -> None:
    """The load-bearing property: cost must be O(1) in turn length, not O(n).

    A 4 tok/s model routinely runs multi-minute turns, so a per-frame cost is
    what turned this from cosmetic into a context-budget problem.
    """
    short = len(_run_turn(0.3))
    long = len(_run_turn(1.5))
    assert long <= short + 64, f"output grew with duration: {short} -> {long}"


def test_headless_still_reports_the_turn_summary() -> None:
    # Suppressing animation must not suppress information — headless logs still
    # need the token counts.
    out = _run_turn()
    assert "4,700" in out and "12" in out

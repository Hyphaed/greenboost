"""The status line must never render wider than the terminal.

Regression for the artifact the owner reported with screenshots on 2026-08-18:
a "Processing" frame rendered wider than the terminal, wrapped onto a second
row, and every later repaint's ``\\r`` then landed on the *wrapped* row — so the
first row was never rewritten and stayed in the scrollback forever. The visible
symptom was a stranded head (``… · ∷ Proc``) above a live tail (``essing …``),
and status frames appearing inside finished tool-output blocks.

Two independent causes, both covered here:

1. ``statusline.py`` sized its frame with ``len()``. Every decorative glyph in
   the line (``·``, ``↑``, ``↓``) is East-Asian-Ambiguous and renders at two
   columns on a CJK-capable font, so the count undershot by ~5 against only 2
   columns of slack. See ``terminal/width.py``.
2. The status thread and the main thread both wrote to stdout with no lock, so
   a frame could be spliced into a half-written tool-output block.
"""
from __future__ import annotations

import io
import sys
import threading
import time

import pytest

from greenboost_cli.terminal import statusline as sl_mod
from greenboost_cli.terminal.statusline import StatusLine
from greenboost_cli.terminal.width import (
    TTY_LOCK,
    display_width,
    strip_ansi,
    truncate_to_width,
    tty_write,
)


# ── width primitives ─────────────────────────────────────────────────────────

def test_ansi_escapes_occupy_no_columns() -> None:
    assert display_width("\x1b[38;5;80mabc\x1b[0m") == 3
    assert strip_ansi("\x1b[2K\x1b[31mx\x1b[0m") == "x"


@pytest.mark.parametrize("glyph", ["·", "↑", "↓"])
def test_ambiguous_decoration_counts_wide(glyph: str) -> None:
    """The glyphs that actually caused the overflow must not count as 1."""
    assert display_width(glyph) == 2


def test_line_art_stays_narrow() -> None:
    """The `─` fill is Ambiguous too, but terminals render it at one column.

    Counting it wide would halve the visible status line — this carve-out is
    what makes the wide-Ambiguous default safe to apply at all.
    """
    assert display_width("─" * 40) == 40


def test_braille_spinner_is_one_column() -> None:
    for frame in sl_mod.SPINNER_THINK:
        assert display_width(frame) == 1, frame


def test_truncate_preserves_escapes_and_respects_budget() -> None:
    out = truncate_to_width("\x1b[31mabcdef\x1b[0m", 3)
    assert display_width(out) == 3
    assert out.startswith("\x1b[31m")


# ── the frame itself ─────────────────────────────────────────────────────────

def _frame(width: int, **state) -> str:
    """Render one frame as if the terminal were `width` columns wide."""
    sl = StatusLine()
    sl._start = time.monotonic() - state.pop("elapsed", 12.3)
    sl._first_out_ts = sl._start + 0.5
    sl._phase = state.pop("phase", "Processing")
    sl._in_tok = state.pop("in_tok", 41_666)
    sl._out_tok = state.pop("out_tok", 116)
    sl._ctx_pct = state.pop("ctx_pct", 0.28)
    sl._frame = state.pop("frame", 0)
    assert not state, f"unused: {state}"

    import shutil
    real = shutil.get_terminal_size
    shutil.get_terminal_size = lambda fallback=(80, 24): type(  # type: ignore[assignment]
        "S", (), {"columns": width, "lines": 24}
    )()
    try:
        return sl._render()
    finally:
        shutil.get_terminal_size = real  # type: ignore[assignment]


@pytest.mark.parametrize("width", [60, 72, 80, 100, 120, 160, 200, 220])
@pytest.mark.parametrize("frame", [0, 3, 7, 11])
def test_frame_never_exceeds_terminal_width(width: int, frame: int) -> None:
    """The core invariant. A frame wider than the terminal is what wrapped."""
    line = _frame(width, frame=frame)
    assert display_width(line) <= width - 2, (
        f"{display_width(line)} columns in a {width}-column terminal: "
        "this is the overflow that strands a row in the scrollback"
    )


@pytest.mark.parametrize("in_tok,out_tok", [(0, 0), (9, 1), (1_234_567, 890_123)])
def test_frame_fits_across_token_magnitudes(in_tok: int, out_tok: int) -> None:
    """Seven-digit counts widen the right side; the fill must absorb it."""
    line = _frame(100, in_tok=in_tok, out_tok=out_tok)
    assert display_width(line) <= 98


@pytest.mark.parametrize("phase", ["Thinking", "Processing", "Compacting context"])
def test_frame_fits_across_phase_labels(phase: str) -> None:
    assert display_width(_frame(90, phase=phase)) <= 88


def test_narrow_terminal_drops_fill_rather_than_overflowing() -> None:
    """The dash run has a floor of 4, so the clamp has to be able to go below it."""
    line = _frame(40)
    assert display_width(line) <= 38


def test_frame_carries_no_trailing_padding() -> None:
    """Padding was how the old code covered a wider previous frame.

    It is gone: \\033[2K erases the row instead, which is correct regardless of
    whether the width estimate was right.
    """
    line = _frame(120)
    assert not strip_ansi(line).endswith("  "), "trailing pad should be gone"


def test_frames_are_measured_wide_under_ambiguous_override(monkeypatch) -> None:
    """GB_CLI_AMBIGUOUS_WIDTH=1 narrows the estimate; the frame must still fit."""
    monkeypatch.setenv("GB_CLI_AMBIGUOUS_WIDTH", "1")
    assert display_width("·") == 1
    assert display_width(_frame(100)) <= 98


# ── serialized writes ────────────────────────────────────────────────────────

def test_repaint_erases_the_row_and_disables_wrapping() -> None:
    """Every repaint must erase, and must not let the terminal wrap the frame."""
    buf = io.StringIO()
    real, sys.stdout = sys.stdout, buf
    try:
        sl = StatusLine()
        sl._start = time.monotonic()
        sl._phase = "Processing"
        tty_write(sl_mod._WRAP_OFF + "\r" + sl_mod._ERASE_LINE + sl._render() + sl_mod._WRAP_ON)
    finally:
        sys.stdout = real
    out = buf.getvalue()
    assert "\033[2K" in out, "no erase-line: a shorter frame would leave a tail"
    assert "\033[?7l" in out and "\033[?7h" in out, "auto-wrap not suppressed"


def test_concurrent_writers_do_not_interleave() -> None:
    """A status frame must never land inside another writer's output.

    This is the half that put "Processing" inside finished tool-output blocks:
    both threads wrote to stdout with nothing serializing them.
    """
    chunks: list[str] = []

    class Recorder:
        def write(self, s: str) -> int:
            chunks.append(s)
            return len(s)

        def flush(self) -> None:
            pass

    real, sys.stdout = sys.stdout, Recorder()
    errors: list[BaseException] = []

    def painter(token: str) -> None:
        try:
            for _ in range(200):
                # One logical frame written in several parts, exactly like the
                # renderer composes its tool cards.
                with TTY_LOCK:
                    tty_write(f"<{token}")
                    tty_write("=" * 8)
                    tty_write(f"{token}>")
        except BaseException as e:  # pragma: no cover - surfaced via `errors`
            errors.append(e)

    try:
        threads = [threading.Thread(target=painter, args=(t,)) for t in ("A", "B", "C")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.stdout = real

    assert not errors, errors
    joined = "".join(chunks)
    for token in ("A", "B", "C"):
        assert joined.count(f"<{token}========{token}>") == 200, (
            f"frames for {token} were interleaved by another writer"
        )


def test_tty_lock_is_reentrant() -> None:
    """renderer.py composes nested writes; a plain Lock would self-deadlock."""
    with TTY_LOCK:
        with TTY_LOCK:
            assert True

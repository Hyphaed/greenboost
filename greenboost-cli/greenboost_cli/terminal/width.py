"""Terminal display-width measurement and serialized terminal writes.

Two primitives live here because they exist for the same reason: keeping the
animated status line from corrupting the scrollback above it. Both halves were
root causes of the same reported artifact (2026-08-18, owner screenshots), where
a "Processing" frame wrapped to a second row and left its own head stranded in
committed scrollback forever.

──────────────────────────────────────────────────────────────────────────────
Half 1 — display_width(): len() is not a column count
──────────────────────────────────────────────────────────────────────────────

statusline.py sized its frame with ``len()``. That counts codepoints, not
terminal columns, and every decorative glyph in the status line is East-Asian
**Ambiguous** width (``unicodedata.east_asian_width() == "A"``):

    ·  U+00B7 MIDDLE DOT          ↑  U+2191        ↓  U+2193
    ─  U+2500 BOX DRAWINGS LIGHT HORIZONTAL        ●  U+25CF

An Ambiguous glyph is one column in a Latin font and two in a CJK-capable one —
the standard leaves it to the font, so the same string is a different width on
two machines. A single status frame carries ~5 of them (three ``·`` separators
plus ``↑`` and ``↓``), and the frame budget only kept 2 columns of slack, so a
3-column undercount was enough to wrap the line. Once it wrapped, the ``\\r``
that starts the next frame returned to column 0 of the *wrapped* row, so the
first row was never rewritten again.

**The Box Drawing carve-out is deliberate and load-bearing.** U+2500 ``─`` is
Ambiguous too, but the status line uses ~50 of them as its fill run. Counting
those at 2 columns would halve the visible line on every terminal. Terminals
universally render Box Drawing and Block Elements at one column — that is the
entire point of line-art glyphs, they have to tile against ASCII — so this
module counts U+2500..U+259F narrow while counting other Ambiguous glyphs wide.
That combination is what actually matches the observed rendering.

Set ``GB_CLI_AMBIGUOUS_WIDTH=1`` to count all Ambiguous glyphs narrow, for a
terminal that renders them that way and shows a short status line as a result.

──────────────────────────────────────────────────────────────────────────────
Half 2 — tty_write(): four threads shared stdout with no lock
──────────────────────────────────────────────────────────────────────────────

The status thread repaints every 80 ms while the main thread streams model text
and tool output. Neither held a lock, so a status frame could land *inside* a
half-written tool-output block. That is why "Processing" appeared in finished
scrollback rather than only on the live bottom row.

``tty_write()`` serializes every writer through one reentrant lock. It is an
RLock because renderer.py composes multi-part writes that would otherwise
deadlock against themselves.
"""
from __future__ import annotations

import os
import contextlib
import re
import sys
import threading
import unicodedata

# CSI sequences (colour, erase, cursor moves) plus OSC strings. Zero columns.
_ANSI_RE = re.compile(
    r"""
      \x1b \[ [0-?]* [ -/]* [@-~]      # CSI  … final byte
    | \x1b \] .*? (?: \x07 | \x1b\\ )  # OSC  … BEL or ST
    | \x1b [@-Z\\-_]                   # two-character escapes
    """,
    re.VERBOSE | re.DOTALL,
)

# Box Drawing + Block Elements. Ambiguous by Unicode, always one column in
# practice — see the module docstring for why this carve-out has to exist.
_LINE_ART_FIRST = 0x2500
_LINE_ART_LAST = 0x259F


def _ambiguous_width() -> int:
    """Columns to charge for an East-Asian-Ambiguous glyph (2 unless overridden)."""
    raw = os.environ.get("GB_CLI_AMBIGUOUS_WIDTH", "").strip()
    return 1 if raw == "1" else 2


def strip_ansi(s: str) -> str:
    """Drop escape sequences, leaving only glyphs that occupy columns."""
    return _ANSI_RE.sub("", s)


def char_width(ch: str) -> int:
    """Columns one character occupies. Combining marks and controls are 0."""
    if unicodedata.combining(ch):
        return 0
    cat = unicodedata.category(ch)
    if cat in ("Mn", "Me", "Cf"):
        return 0
    if cat == "Cc":
        return 0
    eaw = unicodedata.east_asian_width(ch)
    if eaw in ("W", "F"):
        return 2
    if eaw == "A":
        if _LINE_ART_FIRST <= ord(ch) <= _LINE_ART_LAST:
            return 1
        return _ambiguous_width()
    return 1


def display_width(s: str) -> int:
    """Terminal columns `s` occupies, ignoring ANSI escapes.

    Use this anywhere a string is measured against the terminal width. ``len()``
    is wrong for every string in this package that contains a separator dot, an
    arrow, or a spinner glyph.
    """
    return sum(char_width(c) for c in strip_ansi(s))


def truncate_to_width(s: str, max_cols: int) -> str:
    """Cut `s` to at most `max_cols` columns, preserving ANSI escapes.

    Escapes are copied through without consuming budget, so colour state stays
    correct up to the cut. When the cut lands inside an open colour run, a
    reset is appended: otherwise the colour bleeds into whatever the next
    writer puts on that row, which on a shared status row means one tool's
    colour silently repainting another's. This mirrors reflow's truncate
    writer, which calls ResetAnsi() on the same condition
    (`greenboost-extras/tui-study/reflow/truncate/truncate.go`).
    """
    if max_cols <= 0:
        return ""
    out: list[str] = []
    used = 0
    i = 0
    open_sgr = False       # an SGR sequence other than a reset is in effect
    truncated = False
    while i < len(s):
        m = _ANSI_RE.match(s, i)
        if m:
            seq = m.group(0)
            out.append(seq)
            if seq.endswith("m"):
                open_sgr = seq not in ("\033[0m", "\033[m")
            i = m.end()
            continue
        w = char_width(s[i])
        if used + w > max_cols:
            truncated = True
            break
        out.append(s[i])
        used += w
        i += 1
    if truncated and open_sgr:
        out.append("\033[0m")
    return "".join(out)


# ── Serialized terminal writes ───────────────────────────────────────────────

#: Guards every write to the real terminal. Exposed for callers that need to
#: hold it across several writes that must not be interleaved.
TTY_LOCK = threading.RLock()


#: True when the last byte written to the terminal was a newline, i.e. the
#: cursor sits at column 0 and a live painter may safely own the row.
_at_line_start = True

#: Which painter currently owns the live row, or None when it is free.
#:
#: There is exactly ONE bottom row, and two independent painters want it: the
#: status line (its own thread, every 80 ms) and the tool action line (another
#: thread, every 80 ms). With no arbitration both claimed it and the terminal
#: ended up with TWO live rows carrying different timers , observed 2026-08-18,
#: "Processing … 0.6s" and "Processing … 81.0s" stacked on top of each other.
#:
#: Last claimant wins, and releasing hands the row back. The action line is the
#: shorter-lived of the two, so it naturally takes over during a tool call and
#: gives the row back to the status line afterwards.
_live_owner: "str | None" = None

#: True while the live region owns the current row and may repaint it with \r.
#: Distinct from _at_line_start on purpose: a painted frame leaves the cursor
#: mid-row, but that row is OURS, so the next frame overwrites rather than
#: scrolling. Any non-live write takes the row back.
_live_owns_row = False


def at_line_start() -> bool:
    """Whether the cursor is at column 0 of a fresh row.

    Serializing writes is NOT sufficient for a live status row. Streaming model
    text arrives as many small chunks, and the lock is released between them, so
    a status frame can land in the middle of a half-written line of prose. `\r`
    then returns to column 0 of a row that already holds someone else's text,
    the frame overruns, and the row is stranded in scrollback — which is exactly
    the artifact of four stacked half-drawn "Processing" rows, each carrying a
    different token count because each was a different frame in time.

    A live painter must therefore ask not only "am I alone?" but "is the row
    mine to take?".
    """
    with TTY_LOCK:
        return _at_line_start


#: Nesting depth of suspend_live() scopes. Non-zero means no live painter may
#: touch the screen at all.
_live_suspended = 0


def live_suspended() -> bool:
    """True while a multi-row block is being drawn and must not be interrupted."""
    with TTY_LOCK:
        return _live_suspended > 0


@contextlib.contextmanager
def suspend_live():
    """Stop every live painter for the duration of a multi-row block.

    at_line_start() alone is not enough for anything taller than one row. A
    permission card prints its rows one at a time, and BETWEEN two of them the
    cursor is legitimately at column 0 — so a status frame is free to paint
    right through the middle of the box. That is the mangled card in the
    2026-08-18 screenshots: the tool spinner was halted before the prompt, but
    the status line runs on its own thread and repaints every 80 ms.

    Anything that draws more than one row belongs inside this.
    """
    global _live_suspended
    with TTY_LOCK:
        _live_suspended += 1
    try:
        yield
    finally:
        with TTY_LOCK:
            _live_suspended -= 1


def drawn_as_block(fn):
    """Mark a function as drawing a multi-row block.

    Equivalent to wrapping the body in `suspend_live()`, but says so at the
    definition, where the next person editing the function will see it. Use it
    on anything that emits more than one row: a card, a banner, a table.
    """
    import functools

    @functools.wraps(fn)
    def _wrapped(*a, **kw):
        with suspend_live():
            return fn(*a, **kw)

    return _wrapped


def claim_live(owner: str) -> None:
    """Take the live row for `owner`. Frames from anyone else are dropped."""
    global _live_owner, _live_owns_row
    with TTY_LOCK:
        if _live_owner != owner:
            _live_owner = owner
            _live_owns_row = False      # the new owner must acquire its own row


def release_live(owner: str) -> None:
    """Hand the live row back, if `owner` still holds it."""
    global _live_owner, _live_owns_row
    with TTY_LOCK:
        if _live_owner == owner:
            _live_owner = None
            _live_owns_row = False


def tty_write(s: str, *, flush: bool = True, live: bool = False,
              owner: "str | None" = None) -> None:
    """Write to stdout under `TTY_LOCK`, so concurrent painters cannot interleave.

    `live=True` marks the write as a live-region repaint (a status frame, a
    spinner frame). Such a write is DROPPED when a multi-row block is being
    drawn or when the cursor does not own a fresh row.

    The check belongs here, inside the lock, rather than at the call site.
    A painter that tests `live_suspended()` and then calls tty_write has
    released the lock in between, so a block can start in that gap and the
    frame lands in the middle of it anyway — a check-then-act race that a
    concurrency test caught doing exactly that.
    """
    global _at_line_start, _live_owns_row
    with TTY_LOCK:
        if live:
            if _live_suspended > 0:
                return          # a multi-row block is being drawn , stay off
            if owner is not None and _live_owner is not None and _live_owner != owner:
                return          # another painter owns the row , do not stack
            # Three states, and conflating any two of them has now caused a bug
            # each:
            #
            #   owns the row      -> repaint in place with \r. The frame's own
            #                        bytes must NOT count as "someone dirtied
            #                        the row", or every frame newlines and you
            #                        get one "Processing" line per tick , 25 of
            #                        them in the owner's 2026-08-18 screenshot.
            #   clean row, unowned-> take it, and remember that we own it.
            #   dirty row         -> commit it with a newline first, then take
            #                        the fresh row below. Skipping instead (the
            #                        first attempt) froze the line dead: timer
            #                        stuck at 0.0s, spinner stopped.
            if not _live_owns_row and not _at_line_start:
                sys.stdout.write("\n")
                _at_line_start = True
            _live_owns_row = True
            sys.stdout.write(s)
            if flush:
                sys.stdout.flush()
            return
        # A non-live write is real output: it takes the row back from the live
        # region, which must then re-acquire one.
        _live_owns_row = False
        sys.stdout.write(s)
        if flush:
            sys.stdout.flush()
        if s:
            # Ignore trailing escape sequences: they move or style the cursor
            # but emit no glyph, so they do not change whether the row is dirty.
            visible = _ANSI_RE.sub("", s)
            if visible:
                _at_line_start = visible.endswith("\n") or visible.endswith("\r")

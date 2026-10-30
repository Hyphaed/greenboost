"""Pinned two-row footer that shows the working agent during a turn.

    ⠹  qwen3.8-35b-a3b · Writing · Bash nvidia-smi -q          15,116↑ 52↓ · 38 t/s · 12.4s
       esc interrupt · ctrl+p pause · ctrl+q compact · type to queue         T1 10.5/11.9G · T2 0.0/2.6G

The last two terminal rows are fenced off with a scroll region (DECSTBM), so
streamed text and tool blocks scroll ABOVE them and can never overwrite or
strand a frame of them. That replaces the old inline `Running ──` row, which
two painters (status line, tool spinner) fought over and which left a frozen
copy in the scrollback after every tool call.

Lifecycle: activate() when a turn starts animating, deactivate() when the turn
ends or a wizard needs the raw terminal. The prompt_toolkit idle prompt owns
the bottom only while no turn runs, so the two never paint together.
Non-TTY output and terminals under MIN_ROWS never activate.
"""
from __future__ import annotations

import shutil
import sys
import threading
import time

from greenboost_cli.terminal.theme import (
    ANSI_TEAL, ANSI_DIM, ANSI_GRAY, ANSI_RESET, ANSI_BOLD, ANSI_LIME, SPINNER_THINK,
)
from greenboost_cli.terminal.width import (
    TTY_LOCK, display_width, truncate_to_width, tty_overlay, tty_write,
)

ROWS = 2
MIN_ROWS = 10
_SAVE, _RESTORE = "\0337", "\0338"
_ERASE = "\033[2K"
_WRAP_OFF, _WRAP_ON = "\033[?7l", "\033[?7h"

_HINT = "esc interrupt  ·  ctrl+p pause  ·  ctrl+q compact  ·  type to queue"


def _fmt_elapsed(s: float) -> str:
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m{int(s % 60):02d}s"


class Footer:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active = False
        self._size = (0, 0)
        self._t0 = 0.0
        self._source = None          # StatusLine-like: .snapshot() -> dict
        self.agent = ""              # model / agent name shown on the left
        self.tool_label = lambda: ""  # callable -> current tool summary
        self.vitals = lambda: []      # callable -> [(text, style)] tier stats
        self._stop = threading.Event()
        self._thread: "threading.Thread | None" = None

    # ── lifecycle ────────────────────────────────────────────────────────────
    def is_active(self) -> bool:
        return self._active

    def activate(self, source=None) -> bool:
        with self._lock:
            if source is not None:
                self._source = source
            if self._active:
                return True
            try:
                if not sys.stdout.isatty():
                    return False
                cols, rows = shutil.get_terminal_size((80, 24))
            except Exception:
                return False
            if rows < MIN_ROWS:
                return False
            self._t0 = time.monotonic()
            self._size = (cols, rows)
            # Scroll the screen up so the footer rows are blank, fence them off,
            # park the cursor on the last text row and open a fresh line there.
            with TTY_LOCK:
                tty_write(
                    "\n" * ROWS
                    + f"\033[1;{rows - ROWS}r"
                    + f"\033[{rows - ROWS};1H\n"
                )
            self._active = True
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="gb-footer")
            self._thread.start()
            self._paint()
            return True

    def deactivate(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._stop.set()
        t = self._thread
        if t and t is not threading.current_thread():
            t.join(timeout=0.5)
        _, rows = self._size
        with TTY_LOCK:
            # Reset the region (this homes the cursor, hence save/restore), then
            # blank the rows the footer used so the idle prompt starts clean.
            tty_overlay(
                _SAVE + "\033[r"
                + "".join(f"\033[{rows - i};1H{_ERASE}" for i in range(ROWS - 1, -1, -1))
                + _RESTORE
            )

    # ── painting ─────────────────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop.wait(0.1):
            try:
                self._check_resize()
                self._paint()
            except Exception:
                pass            # a decoration must never take down a turn

    def _check_resize(self) -> None:
        cols, rows = shutil.get_terminal_size((80, 24))
        if (cols, rows) == self._size:
            return
        with TTY_LOCK:
            self._size = (cols, rows)
            tty_overlay(_SAVE + f"\033[1;{max(1, rows - ROWS)}r" + _RESTORE)

    def _paint(self) -> None:
        with self._lock:
            if not self._active:
                return
        cols, rows = self._size
        row1, row2 = self._rows(cols)
        tty_overlay(
            _SAVE + _WRAP_OFF
            + f"\033[{rows - 1};1H{_ERASE}{row1}{ANSI_RESET}"
            + f"\033[{rows};1H{_ERASE}{row2}{ANSI_RESET}"
            + _WRAP_ON + _RESTORE
        )

    def _rows(self, cols: int) -> tuple[str, str]:
        snap = {}
        try:
            if self._source is not None:
                snap = self._source.snapshot()
        except Exception:
            pass
        now = time.monotonic()
        elapsed = now - self._t0
        sp = SPINNER_THINK[int(now * 12) % len(SPINNER_THINK)]
        phase = snap.get("phase") or "Working"
        in_tok, out_tok = snap.get("in_tok", 0), snap.get("out_tok", 0)
        tps = snap.get("tok_s", 0.0)
        if tps <= 0 and out_tok and snap.get("first_out"):
            span = now - snap["first_out"]
            tps = out_tok / span if span > 0 else 0.0

        # Right: numbers, dim; the one live rate is the single accent colour.
        right_plain, right = [], []
        if in_tok or out_tok:
            s = f"{in_tok:,}↑ {out_tok:,}↓"
            right_plain.append(s)
            right.append(f"{ANSI_DIM}{s}{ANSI_RESET}")
        if tps > 0:
            s = f"{tps:.0f} t/s"
            right_plain.append(s)
            right.append(f"{ANSI_TEAL}{s}{ANSI_RESET}")
        s = _fmt_elapsed(elapsed)
        right_plain.append(s)
        right.append(f"{ANSI_GRAY}{s}{ANSI_RESET}")
        sep = f"  {ANSI_DIM}·{ANSI_RESET}  "
        right_s = sep.join(right)
        right_w = display_width("  ·  ".join(right_plain))

        # Left: spinner, agent, phase, current tool. Tool label gives way first.
        room = max(10, cols - 4 - right_w - 3)
        agent = self.agent.strip()
        left_plain_parts = ([agent] if agent else []) + [phase]
        tool = (self.tool_label() or "").strip()
        if tool:
            phase = "Running"
        left = f"  {ANSI_TEAL}{sp}{ANSI_RESET}  "
        used = 0
        if agent:
            left += f"{ANSI_BOLD}{ANSI_GRAY}{truncate_to_width(agent, max(8, room // 3))}{ANSI_RESET}{sep}"
            used += min(display_width(agent), max(8, room // 3)) + 5
        left += f"{ANSI_TEAL}{phase}{ANSI_RESET}"
        used += display_width(phase)
        if tool and room - used > 8:
            left += f"{sep}{ANSI_DIM}{truncate_to_width(tool, room - used - 5)}{ANSI_RESET}"
        pad = max(1, cols - 2 - display_width(_strip(left)) - right_w)
        row1 = truncate_to_width(left + " " * pad + right_s, cols - 1)

        # Row 2: key hints, tier vitals right-aligned. Dim: it is reference text.
        vit = ""
        try:
            segs = list(self.vitals() or [])
            vit = "  ·  ".join(t for t, _ in segs)
        except Exception:
            pass
        hint = f"     {_HINT}"
        gap = cols - 2 - display_width(hint) - display_width(vit)
        if vit and gap >= 2:
            row2 = f"{ANSI_DIM}{hint}{' ' * gap}{vit}"
        else:
            row2 = f"{ANSI_DIM}{truncate_to_width(hint, cols - 2)}"
        return row1, row2


def fit_hints(items: list[str], budget: int, sep: str = "  ·  ") -> str:
    """Join as many of `items` (priority order) as fit in `budget` columns."""
    shown: list[str] = []
    for item in items:
        need = display_width(item) + (display_width(sep) if shown else 0)
        if need > budget:
            break
        shown.append(item)
        budget -= need
    return sep.join(shown)


def _strip(s: str) -> str:
    from greenboost_cli.terminal.width import _ANSI_RE
    return _ANSI_RE.sub("", s)


_FOOTER = Footer()


def get_footer() -> Footer:
    return _FOOTER

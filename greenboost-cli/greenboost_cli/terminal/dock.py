"""Permanent bottom dock: one component owns the last rows of the terminal.

    <scrollback, scrolls above>
    <activity line: spinner + verb + elapsed + tokens + t/s while a turn runs>
    ──────────────────────────────────────────── session name ─   (rule)
    ❯ <input box, grows upward, stays live during a turn>
    ────────────────────────────────────────────────────────────  (rule)
    <status 1: permission mode · nonstop · model state · GPU owner · queue    ctx %>
    <status 2: hints (dim)                                  T1 · T2 · T3 tiers>

Ownership rule (see docs/terminal-dock.md): while the dock is live,
prompt_toolkit's Application is the ONLY writer of the bottom rows. Threads
change what it shows by mutating state and calling `app.invalidate()`, never by
writing escape sequences. While a wizard owns the raw terminal, `DockHold` is the
only writer: it fences the dock rows off with a scroll region and paints a static
copy, then restores everything when the wizard returns.

This module holds the pure layout functions (row allocation, truncation by display
width, hint fitting, status-row composition) so they are testable without a PTY.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import threading
from dataclasses import dataclass, field

from greenboost_cli.terminal.theme import (
    TEAL, VIOLET, LIME, AMBER, DIM, GRAY, CORAL, BOX_H,
)
from greenboost_cli.terminal.width import (
    TTY_LOCK, display_width, truncate_to_width, tty_overlay,
)

Frags = "list[tuple[str, str]]"

#: Terminals shorter than this get no dock hold during wizards (they flow as before).
MIN_HOLD_ROWS = 10
MAX_INPUT_ROWS = 8
MENU_ROWS = 8
SEP = "  ·  "
PROMPT_GLYPH = "❯ "


# ── pure layout ──────────────────────────────────────────────────────────────

def dock_mode(rows: int) -> str:
    """full (6 rows) from 12 up, compact (4) from 8, simple (2) below."""
    if rows >= 12:
        return "full"
    if rows >= 8:
        return "compact"
    return "simple"


def dock_rows(rows: int, input_rows: int = 1) -> int:
    """Rows the dock occupies for a terminal of `rows` rows (input grows upward)."""
    base = {"full": 5, "compact": 3, "simple": 1}[dock_mode(rows)]
    return base + max(1, min(input_rows, MAX_INPUT_ROWS))


def max_input_rows(rows: int) -> int:
    return max(1, min(MAX_INPUT_ROWS, rows // 3))


def frags_width(frags) -> int:
    return sum(display_width(t) for _, t in frags)


def truncate_frags(frags, width: int):
    """Cut a fragment list to `width` display columns (grapheme-safe via char_width)."""
    out, used = [], 0
    for style, text in frags:
        if used >= width:
            break
        w = display_width(text)
        if used + w <= width:
            out.append((style, text))
            used += w
        else:
            out.append((style, truncate_to_width(text, width - used)))
            break
    return out


def fit_parts(parts, budget: int, sep: str = SEP):
    """Join as many `parts` (each a fragment list, priority order) as fit in `budget`
    columns. A part that does not fit is dropped whole, with its separator."""
    out, used = [], 0
    sw = display_width(sep)
    for part in parts:
        w = frags_width(part)
        need = w + (sw if out else 0)
        if used + need > budget:
            continue            # a later, shorter part may still fit
        if out:
            out.append((f"fg:{DIM}", sep))
        out.extend(part)
        used += need
    return out


def compose_row(left, right, cols: int, gap_min: int = 2):
    """Left fragments, right fragments right-aligned; never wider than cols-1
    (the last column stays empty so the cursor never wraps). Right wins its space,
    left is truncated."""
    cols = max(1, cols - 1)
    rw = frags_width(right)
    if right and rw + gap_min + 1 > cols:
        right, rw = [], 0
    left = truncate_frags(left, max(0, cols - rw - (gap_min if right else 0)))
    pad = cols - frags_width(left) - rw
    if right:
        return left + [("", " " * max(gap_min, pad))] + right
    return left


def rule_frags(cols: int, name: str = ""):
    """Full-width rule; the session name sits right-aligned like a session label."""
    cols = max(2, cols - 1)
    name = (name or "").strip()
    if name:
        name = truncate_to_width(name, max(1, cols // 3))
        n = display_width(name)
        return [(f"fg:{DIM}", BOX_H * max(2, cols - n - 3) + " "),
                (f"fg:{GRAY}", name), (f"fg:{DIM}", " " + BOX_H)]
    return [(f"fg:{DIM}", BOX_H * cols)]


@dataclass
class DockState:
    """Everything the dock shows, as plain data (built by the REPL each repaint)."""
    name: str = ""
    activity: list = field(default_factory=list)     # fragments, [] when idle
    plan_mode: bool = False
    autonomous: bool = False
    nonstop: bool | None = None
    auto_answer: bool | None = None
    server: tuple | None = None                      # (style, text)
    gpu_owner: str = ""
    statusline: str = ""
    queued: int = 0
    ctx_pct: float | None = None
    tiers: list = field(default_factory=list)        # [(text, style)]
    hints: list = field(default_factory=list)        # priority order
    running: bool = False


def prefix_frags(st: DockState, continuation: bool = False):
    if continuation:
        return [(f"fg:{GRAY}", "  › ")]
    out = []
    if st.plan_mode:
        out.append((f"bold fg:{VIOLET}", "[PLAN] "))
    if st.autonomous:
        out.append((f"fg:{AMBER}", "[AUTO] "))
    if st.queued:
        out.append((f"fg:{AMBER}", f"[{st.queued}q] "))
    out.append((f"bold fg:{VIOLET}", PROMPT_GLYPH))
    return out


def status1_frags(st: DockState, cols: int):
    """Mode row: permission/plan mode, autonomy levers, model state, GPU owner, queue."""
    if st.plan_mode:
        mode = [(f"bold fg:{VIOLET}", "⏸ plan mode on"), (f"fg:{DIM}", " (shift+tab to cycle)")]
    elif st.autonomous:
        mode = [(f"fg:{AMBER}", "⏵⏵ auto-accept on"), (f"fg:{DIM}", " (shift+tab to cycle)")]
    else:
        mode = [(f"fg:{GRAY}", "⏵ ask before risky tools"), (f"fg:{DIM}", " (shift+tab to cycle)")]
    parts = [mode]
    if st.server:
        full = frags_width(mode) + display_width(st.server[1]) + 2 * display_width(SEP) + 20
        short = st.server[1].split(" · ")[0]
        parts.append([(st.server[0], st.server[1] if full < cols else short)])
    if st.nonstop:
        parts.append([(f"fg:{TEAL}", "nonstop")])
    if st.auto_answer:
        parts.append([(f"fg:{AMBER}", "autoanswer")])
    if st.gpu_owner:
        parts.append([(f"fg:{AMBER}", f"GPU: {st.gpu_owner}")])
    if st.queued:
        parts.append([(f"fg:{AMBER}", f"{st.queued} queued")])
    if st.statusline:
        parts.append([(f"fg:{TEAL}", st.statusline)])
    right = []
    if st.ctx_pct is not None:
        hot = st.ctx_pct >= 90
        warn = st.ctx_pct >= 75
        style = f"bold fg:{CORAL}" if hot else (f"fg:{AMBER}" if warn else f"fg:{DIM}")
        right = [(style, f"ctx {st.ctx_pct:.0f}%")]
    budget = cols - 1 - frags_width(right) - (2 if right else 0) - 1
    # The mode label is trimmed to its short form before anything else is dropped.
    left = fit_parts(parts, budget)
    if not left:
        left = fit_parts([[mode[0]]], budget)
    return compose_row([("", " ")] + left, right, cols)


def status2_frags(st: DockState, cols: int):
    """Hint row: dim hints on the left (dropped from the end), tier stats right."""
    right = []
    for i, (txt, style) in enumerate(st.tiers):
        if i:
            right.append((f"fg:{DIM}", " · "))
        right.append((style, txt))
    rw = frags_width(right)
    budget = cols - 1 - 1 - (rw + 3 if right else 0)
    hints = fit_parts([[(f"fg:{DIM}", h)] for h in st.hints], budget)
    return compose_row([("", " ")] + hints, right, cols)


def plan_hold(cursor_row: int, rows: int, dock_h: int) -> tuple[int, int, int]:
    """Row arithmetic for fencing the dock while a wizard runs (1-based rows).

    Returns (scroll_lines, region_bottom, park_row): emit `scroll_lines` newlines
    on the last screen row first (pushes content up when the cursor sits inside the
    rows the dock is about to take), then DECSTBM 1..region_bottom, then park the
    cursor at park_row."""
    region_bottom = max(1, rows - dock_h)
    if cursor_row <= region_bottom:
        return 0, region_bottom, cursor_row
    return cursor_row - region_bottom, region_bottom, region_bottom


# ── fragments → ANSI (for the static copy painted during a wizard) ───────────

_HEX = re.compile(r"fg:#([0-9a-fA-F]{6})")


def _color_enabled() -> bool:
    return not os.environ.get("NO_COLOR")


def frags_to_ansi(frags) -> str:
    out = []
    color = _color_enabled()
    for style, text in frags:
        sgr = []
        if "bold" in style:
            sgr.append("1")
        m = _HEX.search(style)
        if m and color:
            h = m.group(1)
            sgr.append(f"38;2;{int(h[0:2], 16)};{int(h[2:4], 16)};{int(h[4:6], 16)}")
        out.append((f"\033[{';'.join(sgr)}m" if sgr else "") + text + ("\033[0m" if sgr else ""))
    return "".join(out)


def static_dock_lines(st: DockState, cols: int, rows: int, note: str = "") -> list[str]:
    """The dock as plain ANSI rows (no input editor), for DockHold."""
    mode = dock_mode(rows)
    inp = [(f"bold fg:{VIOLET}", PROMPT_GLYPH), (f"fg:{DIM}", note or "answer above")]
    lines = []
    if mode == "full":
        lines.append([])
    if mode != "simple":
        lines.append(rule_frags(cols, st.name))
    lines.append(truncate_frags(inp, cols - 1))
    if mode != "simple":
        lines.append(rule_frags(cols))
    lines.append(status1_frags(st, cols))
    if mode == "full":
        lines.append(status2_frags(st, cols))
    return [frags_to_ansi(f) for f in lines]


# ── cursor row query (single reader, bounded wait) ───────────────────────────

def query_cursor_row(timeout: float = 0.8) -> int | None:
    """Ask the terminal where the cursor is (CPR). Only call when nothing else reads
    stdin (the input app is stopped). None when the terminal does not answer."""
    try:
        import select
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except Exception:
        return None
    answered = False
    try:
        tty.setcbreak(fd)
        with TTY_LOCK:
            sys.stdout.write("\033[6n")
            sys.stdout.flush()
        buf = ""
        import time
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            r, _, _ = select.select([fd], [], [], max(0.0, end - time.monotonic()))
            if not r:
                break
            buf += os.read(fd, 64).decode("latin1")
            m = re.search(r"\x1b\[(\d+);(\d+)R", buf)
            if m:
                answered = True
                return int(m.group(1))
        answered = False
        return None
    except Exception:
        return None
    finally:
        try:
            # A terminal that answers after the timeout (busy, slow link) must not find
            # echo on: its late reply would print as "^[[1;1R" on the prompt line.
            # restore_terminal() switches echo back on once it has drained the reply.
            if not answered:
                old[3] &= ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass


class DockHold:
    """Keeps the dock on screen while a wizard owns the raw terminal.

    hold(): fence the last rows with DECSTBM, paint a static dock there, leave the
    cursor in the region just below the scrollback. The wizard prints inside the
    region. release(): drop the fence, erase the static rows below the cursor so
    the live dock can be drawn again by the input app. Re-entrant (depth counted).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._depth = 0
        self._rows = 0
        self._dock_h = 0

    def active(self) -> bool:
        return self._depth > 0

    def hold(self, st: DockState, note: str = "") -> bool:
        with self._lock:
            self._depth += 1
            if self._depth > 1:
                return True
            try:
                if not sys.stdout.isatty():
                    return False
                cols, rows = shutil.get_terminal_size((80, 24))
            except Exception:
                return False
            if rows < MIN_HOLD_ROWS:
                return False
            lines = static_dock_lines(st, cols, rows, note)
            dock_h = len(lines)
            cur = query_cursor_row()
            if cur is None:
                cur = rows
            scroll, bottom, park = plan_hold(cur, rows, dock_h)
            seq = ""
            if scroll:
                seq += f"\033[{rows};1H" + "\n" * scroll
            seq += f"\033[1;{bottom}r"
            seq += "\0337"
            for i, ln in enumerate(lines):
                seq += f"\033[{rows - dock_h + 1 + i};1H\033[2K" + ln
            seq += "\0338" + f"\033[{park};1H"
            tty_overlay(seq)
            self._rows, self._dock_h = rows, dock_h
            return True

    def release(self) -> None:
        with self._lock:
            if self._depth == 0:
                return
            self._depth -= 1
            if self._depth > 0 or not self._dock_h:
                self._dock_h = 0
                return
            self._dock_h = 0
            cur = query_cursor_row()
            # `\033[r` homes the cursor; put it back, then clear what the static
            # dock occupied (the live dock repaints it).
            if cur is None:
                tty_overlay("\033[r")
                return
            tty_overlay(f"\033[r\033[{cur};1H\033[J")

    def force_reset(self) -> None:
        with self._lock:
            self._depth = 0
            self._dock_h = 0
        try:
            tty_overlay("\033[r")
        except Exception:
            pass


_HOLD = DockHold()


def get_hold() -> DockHold:
    return _HOLD


# ── prompt_toolkit application ───────────────────────────────────────────────

class DockSession:
    """Drop-in for PromptSession.prompt() with a permanent bottom dock layout.

    One Application is reused for every prompt() call. Layout (top to bottom):
    spacer (absorbs free rows so everything below sits on the last rows),
    activity, rule, input, completion menu, rule, status 1, status 2.
    """

    def __init__(self, *, history, completer, complete_while_typing, key_bindings,
                 style, get_state, refresh_interval: float = 1.0) -> None:
        from prompt_toolkit.application import Application, get_app
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.filters import Condition, has_completions, is_done
        from prompt_toolkit.formatted_text import FormattedText
        from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
        from prompt_toolkit.layout import (
            ConditionalContainer, Dimension, HSplit, Layout, Window,
        )
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
        from prompt_toolkit.layout.menus import CompletionsMenu
        from prompt_toolkit.layout.processors import (
            HighlightSelectionProcessor, HighlightIncrementalSearchProcessor,
        )

        self._get_state = get_state
        self._continuation = False
        self._ghost = False          # exit without echoing (wizard takes the terminal)
        self._draft = ""
        self._FT = FormattedText

        def _size():
            try:
                s = get_app().output.get_size()
                return s.columns, s.rows
            except Exception:
                c, r = shutil.get_terminal_size((80, 24))
                return c, r

        def st() -> DockState:
            try:
                return self._get_state()
            except Exception:
                return DockState()

        mode = lambda: dock_mode(_size()[1])           # noqa: E731
        done = Condition(lambda: bool(get_app().is_done))
        live = ~done
        full = Condition(lambda: mode() == "full")
        framed = Condition(lambda: mode() != "simple")
        ghost_done = Condition(lambda: bool(get_app().is_done) and (self._ghost or not self.buffer.text.strip()))

        def activity():
            cols = _size()[0]
            return FormattedText(compose_row([("", " ")] + list(st().activity), [], cols))

        def rule_top():
            return FormattedText(rule_frags(_size()[0], st().name))

        def rule_bottom():
            return FormattedText(rule_frags(_size()[0]))

        def line1():
            return FormattedText(status1_frags(st(), _size()[0]))

        def line2():
            return FormattedText(status2_frags(st(), _size()[0]))

        self.buffer = Buffer(
            history=history, completer=completer,
            complete_while_typing=complete_while_typing,
            enable_history_search=True, multiline=False,
            accept_handler=self._accept,
        )

        def line_prefix(line_no, wrap_count):
            p = prefix_frags(st(), self._continuation)
            if line_no == 0 and wrap_count == 0:
                return FormattedText(p)
            return FormattedText([("", " " * frags_width(p))])

        def input_height():
            return Dimension(min=1, max=max_input_rows(_size()[1]))

        self.input_window = Window(
            BufferControl(
                buffer=self.buffer,
                input_processors=[HighlightIncrementalSearchProcessor(),
                                  HighlightSelectionProcessor()],
                include_default_input_processors=True,
            ),
            height=input_height, wrap_lines=True, dont_extend_height=True,
            get_line_prefix=line_prefix,
        )

        def one(text_fn, visible, *, style=""):
            return ConditionalContainer(
                Window(FormattedTextControl(text_fn), height=1, dont_extend_height=True,
                       style=style),
                filter=visible & live)

        root = HSplit([
            ConditionalContainer(Window(height=Dimension(weight=1)), filter=live),
            one(activity, full),
            one(rule_top, framed),
            ConditionalContainer(self.input_window, filter=~ghost_done),
            ConditionalContainer(
                CompletionsMenu(max_height=MENU_ROWS, scroll_offset=1),
                filter=has_completions & live),
            one(rule_bottom, framed),
            one(line1, Condition(lambda: True)),
            one(line2, full),
            # Zero-information row kept only so a ghost exit has a non-empty layout.
            ConditionalContainer(Window(height=1), filter=ghost_done),
        ])
        self.layout = Layout(root, focused_element=self.input_window)

        kb = KeyBindings()
        default_focus = Condition(lambda: get_app().layout.has_focus(self.buffer))

        @kb.add("enter", filter=default_focus)
        def _accept_key(event):
            event.current_buffer.validate_and_handle()

        @kb.add("c-c", filter=default_focus)
        def _interrupt(event):
            event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

        @kb.add("c-d", filter=default_focus & Condition(lambda: self.buffer.text == ""))
        def _eof(event):
            event.app.exit(exception=EOFError, style="class:exiting")

        @kb.add("tab", filter=default_focus)
        def _tab(event):
            b = event.current_buffer
            if b.complete_state:
                b.complete_next()
            else:
                b.start_completion(select_first=False)

        self.app = Application(
            layout=self.layout,
            key_bindings=merge_key_bindings([kb, key_bindings] if key_bindings else [kb]),
            style=style, full_screen=False, refresh_interval=refresh_interval,
            erase_when_done=False,
        )

    # ── PromptSession-compatible surface ────────────────────────────────────
    def _accept(self, buff) -> bool:
        self.app.exit(result=buff.document.text)
        return True      # keep text; reset at the next prompt()

    def ghost_exit(self) -> None:
        """Mark the next exit as silent: no echoed input line (wizard suspend)."""
        self._ghost = True
        self._draft = self.buffer.text

    def prompt(self, *, continuation: bool = False, pre_run=None):
        from prompt_toolkit.document import Document
        self._continuation = continuation
        text = self._draft if not continuation else ""
        self._draft = ""
        self._ghost = False
        self.buffer.reset(Document(text, len(text)))
        return self.app.run(pre_run=pre_run)


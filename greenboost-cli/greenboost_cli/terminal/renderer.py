"""
All UI output for interactive sessions.

Visual language:
  ◈  GreenBoost  ──────────────────────────────────  claude-sonnet-4-6   response header
  ⠹  Thinking  ──────────────────────────────────────────────────  2.4s   statusline (animated)

  ╭─ › Bash  ──────────────────────────────────────────────────────────   tool card top
  │  git log --oneline -10                                                 tool input
  ◐  running  ──────────────────────────────────────────────────  0.0s   spinner (animated)
→ ╰─ ✓  10L output  ───────────────────────────────────────────  0.24s   result (overwrites)

    a3b2c1d fix auth bug                                                   output body
    d4e5f6g feat: user mgmt                                                (Bash/Grep/Glob)

  ──────────────────────────────────  ✓  4.2s  ·  1,234↑  ·  89↓         footer

Rules:
  - Tool spinner writes to stdout WITHOUT newline — spinner owns that line.
  - show_instrument_result() overwrites spinner line via \r then adds \n.
  - Rich markup used for static lines; raw ANSI used for \r-overwritten lines.
  - Bash/Grep/Glob output always shown; others shown in verbose mode only.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import threading
import time

from greenboost_cli.terminal.theme import (
    console,
    GRAY, AMBER, LIME, CYAN, DIM, TEAL,
    ICON_OK, ICON_FAIL, ICON_WARN, ICON_INFO, ICON_EXPAND,
    SEPARATOR,
    emit_ok, emit_err, emit_warn, emit_info,
    render_markdown, has_markdown,
    ANSI_GRAY, ANSI_AMBER, ANSI_RESET, ANSI_LIME,
    ANSI_VIOLET, ANSI_CYAN, ANSI_DIM, ANSI_BOLD, ANSI_TEAL,
    SPINNER_TOOL, BOX_H, BOX_V, BOX_TL, BOX_TR, BOX_BL, BOX_BR, BOX_ML, BOX_MR, TOOL_ICONS,
)

# ── Text buffer ────────────────────────────────────────────────────────────────
_text_buffer: list[str] = []
_block_start_t: float   = 0.0
_block_model: str       = ""

# ── Quiet-mode state (set per turn in open_response_block) ────────────────────
_quiet_mode: bool       = False
_turn_tool_names: list[str] = []   # accumulated per turn for tally footer

# ── Streaming line tracker (for cursor-up Markdown replacement) ────────────────
_stream_line_count: int = 0   # number of \n written via emit_text_fragment this turn
# True once replace_text_buffer() swaps the streamed buffer for cleaned text
# this turn (adapters.py does this whenever a tool call was parsed out of the
# streamed text) — finalize_response() uses this to know the raw stream and
# the final buffer can differ, which is exactly when the erase-and-repaint
# must run even if the final text is empty or non-markdown.
_buffer_was_replaced: bool = False

# ── Tool spinner state ─────────────────────────────────────────────────────────
_tool_t0: float                    = 0.0
_tool_stop_evt: threading.Event | None = None
_tool_thread: threading.Thread | None  = None

# ── prompt_toolkit mode ─────────────────────────────────────────────────────────
# Raw \r-overwrite animation is only safe when the pt app is NOT running.
# The pt app runs only at the idle prompt (inside _stdin_reader._pt_session.prompt()).
# During a model turn _stdin_reader blocks on _model_idle.wait(), so the pt app
# has already returned from prompt() → app.is_running == False.
# _pt_live() returns True only when the pt app is actually live (idle prompt).
# During turns it returns False → raw animation runs safely (patch_stdout inactive).
_PT_ACTIVE: bool = False   # kept for backward compat — use _pt_live() for decisions
_pt_live_probe = None       # type: ignore  # callable() -> bool, registered by repl.py


def set_pt_active(active: bool) -> None:
    global _PT_ACTIVE
    _PT_ACTIVE = active


def set_pt_live_probe(cb) -> None:
    """Register the live-probe callback from repl.py. Called once at startup."""
    global _pt_live_probe
    _pt_live_probe = cb


def _pt_live() -> bool:
    """True only when the pt PromptSession app is actively running.

    Returns False during every model turn → raw \\r animation and live token
    streaming are used. Returns True only at the idle prompt, but rendering
    functions (emit_text_fragment, show_instrument_start, etc.) are never
    called then, so the branch is effectively unreachable during idle."""
    if _pt_live_probe is None:
        return _PT_ACTIVE   # fallback: respect legacy set_pt_active() if no probe
    try:
        return bool(_pt_live_probe())
    except Exception:
        return False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _w() -> int:
    return shutil.get_terminal_size((80, 24)).columns


def _fmt_t(sec: float) -> str:
    return f"{sec * 1000:.0f}ms" if sec < 1.0 else f"{sec:.2f}s"


def _fmt_elapsed(sec: float) -> str:
    """Human-readable elapsed time: 450ms / 3.2s / 1m 23s."""
    if sec < 1.0:
        return f"{sec * 1000:.0f}ms"
    if sec >= 60:
        m, s = int(sec // 60), int(sec % 60)
        return f"{m}m {s}s"
    return f"{sec:.1f}s"


def _fmt_tokens(n: int) -> str:
    """Compact human-readable token count: 1.2m / 200k / 21k / 540."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}m"
    if n >= 10_000:
        return f"{n // 1000}k"
    if n >= 1_000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _dashes(used_visible: int, extra: int = 0) -> str:
    return BOX_H * max(2, _w() - used_visible + extra)


# ── Tool spinner ───────────────────────────────────────────────────────────────

_SPINNER_TIPS = [
    "/quiet  suppress tool cards",
    "/verbose  show full outputs",
    "/undo  remove last exchange",
    "/compact  free context space",
    "Ctrl+C  cancel this operation",
    "/retry  re-run last message",
    "/context  check window usage",
    "/gb-status  GPU · T2 · T3 stats",
    "/sessions search  find past work",
    "/goals  inject project context",
]


def _spinner_worker(stop: threading.Event, t0: float) -> None:
    frame = 0
    # Widest line painted so far this run. The >3s "tip" branch used to
    # append NO trailing pad at all (unlike the <3s branch), so crossing the
    # 3s threshold — or the tip text changing length on its 5s rotation —
    # could strand characters from a wider previous line at the right edge,
    # the same class of bug as statusline.py's breathing-dash artifact.
    prev_vis_len = 0
    while not stop.is_set():
        elapsed = time.monotonic() - t0
        t      = _fmt_t(elapsed)
        sp     = SPINNER_TOOL[frame % len(SPINNER_TOOL)]
        safe_w = _w() - 2

        if elapsed > 3.0:
            # After 3 seconds, replace some dashes with a rotating tip
            tip = _SPINNER_TIPS[int(elapsed / 5) % len(_SPINNER_TIPS)]
            left_d  = 4
            right_d = max(2, safe_w - 2 - 1 - 2 - len("running") - 2 - left_d - 2 - len(tip) - 2 - len(t) - 2)
            line = (
                f"\r  {ANSI_TEAL}{sp}{ANSI_RESET}"
                f"  {ANSI_DIM}running {'─' * left_d}  {tip}  {'─' * right_d}  {t}{ANSI_RESET}"
            )
            vis_len = 2 + 1 + 2 + len("running") + 2 + left_d + 2 + len(tip) + 2 + right_d + 2 + len(t)
        else:
            label_vis = 2 + 1 + 2 + len("running") + 2
            right_vis = len(t) + 2
            dashes    = max(2, safe_w - label_vis - right_vis)
            line = (
                f"\r  {ANSI_TEAL}{sp}{ANSI_RESET}"
                f"  {ANSI_DIM}running {'─' * dashes}  {t}{ANSI_RESET}"
            )
            vis_len = label_vis + dashes + right_vis

        pad_to = max(safe_w, prev_vis_len)
        prev_vis_len = vis_len
        line += " " * max(0, pad_to - vis_len)

        sys.stdout.write(line)
        sys.stdout.flush()
        stop.wait(0.08)
        frame += 1


def _start_tool_spinner() -> None:
    global _tool_t0, _tool_stop_evt, _tool_thread
    _tool_t0 = time.monotonic()
    if _pt_live():
        # pt app is live at the idle prompt — skip raw \r animation; toolbar
        # shows live status. (In practice this branch is unreachable during turns
        # because _pt_live() returns False then.)
        return
    _tool_stop_evt = threading.Event()
    _tool_thread   = threading.Thread(
        target=_spinner_worker,
        args=(_tool_stop_evt, _tool_t0),
        daemon=True,
        name="gb-tool",
    )
    _tool_thread.start()


def _stop_tool_spinner() -> float:
    global _tool_stop_evt, _tool_thread
    elapsed = time.monotonic() - _tool_t0
    if _tool_stop_evt:
        _tool_stop_evt.set()
    if _tool_thread:
        _tool_thread.join(timeout=0.5)
    _tool_stop_evt = None
    _tool_thread   = None
    return elapsed


def halt_tool_spinner() -> None:
    """Stop the tool spinner if running and erase its line. Safe to call when idle."""
    if _tool_stop_evt is None:
        return
    _stop_tool_spinner()
    if _pt_live():
        return
    w = _w()
    sys.stdout.write(f"\r{' ' * w}\r")
    sys.stdout.flush()


# ── Streaming ──────────────────────────────────────────────────────────────────

def emit_text_fragment(chunk: str) -> None:
    global _stream_line_count
    # Flush pending file-op group before the first text of a new prose block
    if _pending_file_ops > 0:
        _flush_file_group()
    if not _pt_live():
        # pt app not live (or no pt) — stream raw; also the path during every turn.
        if not _text_buffer:
            # First chunk of a new prose block — lead with the CC-style bullet.
            sys.stdout.write(f"  {ANSI_LIME}{_CC_BULLET}{ANSI_RESET}  ")
        sys.stdout.write(chunk)
        sys.stdout.flush()
        _stream_line_count += chunk.count("\n")
    _text_buffer.append(chunk)


def emit_reasoning(chunk: str, verbose: bool) -> None:
    if verbose:
        console.print(f"[dim {GRAY}]{chunk}[/]", end="")


def replace_text_buffer(clean_text: str) -> None:
    global _buffer_was_replaced
    _text_buffer.clear()
    if clean_text:
        _text_buffer.append(clean_text)
    _buffer_was_replaced = True


def finalize_response() -> None:
    global _stream_line_count, _group_tool, _buffer_was_replaced
    full = "".join(_text_buffer)
    _text_buffer.clear()
    if full.strip():
        _group_tool = ""  # assistant text between tool calls breaks the group

    # replace_text_buffer() was called this turn (adapters.py does this
    # whenever it parsed a tool call out of the streamed text) — the raw
    # stream and `full` can differ, so the erase-and-repaint below must run
    # even if `full` ends up EMPTY or non-markdown.
    needs_erase = _buffer_was_replaced
    _buffer_was_replaced = False

    if _pt_live():
        # pt app is live — nothing was streamed to stdout (buffered only);
        # just render once, no cursor-up erase needed.
        if full.strip():
            print()
            sys.stdout.write(f"  {ANSI_LIME}{_CC_BULLET}{ANSI_RESET}  ")
            if has_markdown(full):
                render_markdown(full)
            else:
                print(full)
        _stream_line_count = 0
        return

    if needs_erase or (full.strip() and has_markdown(full)):
        # Erase raw streamed text and re-render cleanly. The old gate here
        # was `full.strip() and has_markdown(full)`, which skipped this
        # entirely for a pure tool-call turn (full == "") — exactly the case
        # that left raw <tool_call>/<function=.../<parameter=... markup on
        # screen forever (confirmed live, before the incremental suppressor
        # in adapters.py existed; this stays the second line of defense for
        # whatever slips past it).
        # \x1b[nA  = move cursor up n lines
        # \r        = go to column 0
        # \x1b[J   = clear from cursor to end of screen
        if sys.stdout.isatty():
            if _stream_line_count > 0:
                sys.stdout.write(f"\x1b[{_stream_line_count}A")
            sys.stdout.write("\r\x1b[J")
            sys.stdout.flush()
        elif _stream_line_count > 0:
            print()
        if full.strip():
            print()
            sys.stdout.write(f"  {ANSI_LIME}{_CC_BULLET}{ANSI_RESET}  ")
            if has_markdown(full):
                render_markdown(full)
            else:
                print(full)
        _stream_line_count = 0
        return

    if full.strip():
        print()
    _stream_line_count = 0


# ── Tool display ───────────────────────────────────────────────────────────────

# File-op tools shown grouped as "Read N files (ctrl+o to expand)"
_FILE_TOOLS = {"Read", "Write", "Edit"}

# Running counter of tool calls in this turn (reset in open_response_block)
_tool_call_counter: int = 0
# Cached summary of the current tool's inputs (for result line)
_last_tool_summary: str = ""
_last_tool_badge:   str = ""

# Grouped file-op state (Read/Write/Edit defer to a single summary line)
_pending_file_ops:   int       = 0
_pending_file_names: list[str] = []
_pending_file_tools: list[str] = []   # which of Read/Write/Edit each entry was

# Last truncated tool result — the "(ctrl+o to expand)" hint printed by
# show_instrument_result() advertised this affordance since before it had any
# keybinding to trigger it. expand_last_result() (bound to ctrl+o in
# repl.py) reprints the lines that were cut.
_last_truncated: dict | None = None

# Consecutive Grep/Glob calls coalesce into one updating block
_GROUP_TOOLS  = {"Grep", "Glob"}
_group_tool:  str = ""   # name of the currently open group ("" = none)
_group_count: int = 0    # calls in the open group
_group_rows:  int = 0    # condensed child rows printed since the header

# Skills active this turn (set by open_response_block, shown in meta pill)
_active_loaded_skills: list[str] = []

# Claude Code visual constants
_CC_BULLET = "●"
_CC_TREE   = "└"

_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def _badge(n: int) -> str:
    return _CIRCLED[n - 1] if 1 <= n <= len(_CIRCLED) else f"({n})"


_FILE_VERBS = {"Read": "Read", "Write": "Wrote", "Edit": "Edited"}


def _flush_file_group() -> None:
    """Print the pending Read/Write/Edit summary line and reset state."""
    global _pending_file_ops, _pending_file_names, _pending_file_tools, _last_truncated
    if _pending_file_ops <= 0:
        return
    n     = _pending_file_ops
    names = _pending_file_names
    tools = _pending_file_tools
    _pending_file_ops   = 0
    _pending_file_names = []
    _pending_file_tools = []
    w = _w()
    # The verb used to be hardcoded "Read" even for a batch of Write/Edit
    # calls — derive it from what was actually in the group; "Modified" for
    # a mixed batch (Read+Edit in the same turn is common).
    distinct = set(tools)
    verb = _FILE_VERBS.get(next(iter(distinct)), "Modified") if len(distinct) == 1 else "Modified"
    label = f"{verb} {n} file{'s' if n > 1 else ''}"
    suffix = "  (ctrl+o to expand)"
    pad    = " " * max(0, w - len(label) - len(suffix))
    sys.stdout.write(
        f"\r  {ANSI_DIM}{label}  {ANSI_GRAY}(ctrl+o to expand){ANSI_RESET}{pad}\n"
    )
    sys.stdout.flush()
    _last_truncated = {"name": "Files", "lines": list(names), "is_diff": False, "shown": 0}


_TODO_STATUS_ICONS = {
    "pending":     (f"[{DIM}]○[/]",  f"[{GRAY}]"),
    "in_progress": (f"[{AMBER}]◐[/]", f"[{AMBER}]"),
    "completed":   (f"[{LIME}]✓[/]",  f"[{DIM}]"),
}
_TODO_PRIORITY_COLORS = {"high": AMBER, "medium": CYAN, "low": DIM}

# Snapshot of (content, status) from the last board render — lets consecutive
# TodoWrite calls in one turn print only the rows that actually changed
# instead of stacking N full copies of the same list.
_prev_todo_snapshot: list[tuple[str, str]] = []


def _format_todo_row(t: dict) -> str:
    st   = t.get("status", "pending")
    pri  = t.get("priority", "medium")
    icon, col = _TODO_STATUS_ICONS.get(st, _TODO_STATUS_ICONS["pending"])
    pcol = _TODO_PRIORITY_COLORS.get(pri, DIM)
    pri_tag = f"[{pcol}]{pri[:3]}[/]"
    content = t.get("content", "")[:80]
    strike = "~~" if st == "completed" else ""
    return f"  {icon}  {pri_tag}  {col}{strike}{content}{strike}[/]"


def _render_todo_board() -> None:
    """Print the in-session todo list in Claude Code task-board style.

    Consecutive TodoWrite calls within the same turn (the model commonly
    updates one task at a time) print only the changed rows instead of
    reprinting the entire list — avoids stacking duplicate-looking boards."""
    global _prev_todo_snapshot
    from greenboost_cli.instruments.handlers import _session_todos
    if not _session_todos:
        console.print(f"  [{DIM}]  {_CC_TREE}[/] [{GRAY}]Task list cleared[/]")
        _prev_todo_snapshot = []
        return

    cur_snapshot = [(t.get("content", ""), t.get("status", "pending")) for t in _session_todos]

    if len(_prev_todo_snapshot) == len(cur_snapshot) and _prev_todo_snapshot:
        changed = [i for i, (prev, cur) in enumerate(zip(_prev_todo_snapshot, cur_snapshot)) if prev != cur]
        if changed and len(changed) < len(cur_snapshot):
            for i in changed:
                console.print(_format_todo_row(_session_todos[i]))
            _prev_todo_snapshot = cur_snapshot
            return

    console.print()
    for t in _session_todos:
        console.print(_format_todo_row(t))
    console.print()
    _prev_todo_snapshot = cur_snapshot


def show_instrument_start(name: str, inputs: dict, verbose: bool) -> None:
    """Print Claude Code-style tool header and start the animated spinner.

    File tools (Read/Write/Edit) non-verbose: silent spinner, grouped summary.
    Group tools (Grep/Glob) consecutive: coalesce into one updating block.
    All other tools: `● ToolName(args)` bullet header + spinner below.
    Quiet mode: single overwritable spinner line.
    AskUserQuestion: interactive tool — wizard IS the UI, skip card + spinner.
    """
    global _tool_call_counter, _turn_tool_names, _last_tool_summary, _last_tool_badge
    global _group_tool, _group_count, _group_rows

    _tool_call_counter += 1
    _turn_tool_names.append(name)

    # AskUserQuestion is interactive — the wizard is its UI; no card or spinner needed.
    # TodoWrite/TodoRead render as the task board itself (show_instrument_result) —
    # no separate header/spinner, and no raw-repr leak from _summarize_instrument
    # (its inputs are a list of dicts, not a single short string).
    if name in ("AskUserQuestion", "TodoWrite", "TodoRead"):
        return

    summary = _summarize_instrument(name, inputs)
    _last_tool_summary = summary
    _last_tool_badge   = _badge(_tool_call_counter)

    # ── Consecutive Grep/Glob: coalesce into one updating block ───────────────
    if name in _GROUP_TOOLS:
        if name == _group_tool:
            # Continuation: just increment the count and restart the spinner
            _group_count += 1
            _start_tool_spinner()
            return
        # New group: flush any pending file-op group, print plain header (no args)
        _flush_file_group()
        _group_tool  = name
        _group_count = 1
        _group_rows  = 0
        if _quiet_mode:
            if not _pt_live():
                label = f"{name}  {summary[:_w() - 20]}"
                sys.stdout.write(f"  {ANSI_TEAL}◐{ANSI_RESET}  {ANSI_DIM}{label}{ANSI_RESET}  ")
                sys.stdout.flush()
            _start_tool_spinner()
            return
        console.print()
        console.print(f"  [{LIME}]{_CC_BULLET}[/] [{TEAL}]{name}[/]")
        _start_tool_spinner()
        return

    # Non-group tool: break any open group
    _group_tool  = ""
    _group_count = 0
    _group_rows  = 0

    # Flush any pending file-op group before a non-file tool
    if name not in _FILE_TOOLS:
        _flush_file_group()

    w = _w()

    if _quiet_mode:
        if not _pt_live():
            label = f"{name}  {summary[:w - 20]}"
            sys.stdout.write(f"  {ANSI_TEAL}◐{ANSI_RESET}  {ANSI_DIM}{label}{ANSI_RESET}  ")
            sys.stdout.flush()
        _start_tool_spinner()
        return

    # File tools non-verbose: defer to group summary, silent spinner on current line
    if name in _FILE_TOOLS and not verbose:
        _start_tool_spinner()
        return

    # ── Claude Code bullet header ──────────────────────────────────────────────
    from rich.markup import escape
    cmd_max  = max(20, w - len(name) - 10)
    cmd_disp = escape(summary[:cmd_max])

    console.print()
    console.print(f"  [{LIME}]{_CC_BULLET}[/] [{TEAL}]{name}[/][{DIM}]({cmd_disp})[/]")
    _start_tool_spinner()


def show_instrument_result(name: str, result: str, verbose: bool) -> None:
    """Stop spinner and display tool output in Claude Code tree style."""
    global _pending_file_ops, _pending_file_names, _pending_file_tools
    global _group_rows, _last_truncated
    elapsed  = _stop_tool_spinner()
    t        = _fmt_elapsed(elapsed)
    safe_w   = _w() - 2
    is_error = result.startswith("Error") or result.startswith("Denied")

    # Erase spinner line — during turns (pt app not live) a raw \r spinner was
    # drawn; clear it before printing the result. No-op if nothing was drawn.
    if not _pt_live():
        sys.stdout.write(f"\r{' ' * safe_w}\r")
        sys.stdout.flush()

    # ── Grep/Glob group: condensed child rows + in-place header count update ───
    if name in _GROUP_TOOLS and not _quiet_mode:
        from rich.markup import escape as _esc
        if is_error:
            oneliner = f"✗ {result[:80].replace(chr(10), ' ')}"
        elif not result.strip() or result.strip() == "No matches found":
            oneliner = "No matches"
        else:
            oneliner = _result_summary(name, result)

        # Rewrite header in-place to show running count (≥2nd call onward)
        if _group_count >= 2 and not _pt_live() and sys.stdout.isatty():
            lines_up = _group_rows + 1
            sys.stdout.write(
                f"\x1b[{lines_up}A"   # move up to header line
                f"\r\x1b[2K"          # clear it
                f"  {ANSI_LIME}{_CC_BULLET}{ANSI_RESET}"
                f" {ANSI_TEAL}{name}{ANSI_RESET}"
                f" {ANSI_DIM}×{_group_count}{ANSI_RESET}"
                f"\x1b[{lines_up}B\r" # move back down to current line
            )
            sys.stdout.flush()

        # Print condensed child row (first uses └ prefix, rest use spaces)
        prefix = f"  {_CC_TREE} " if _group_rows == 0 else "      "
        max_sum = max(10, safe_w - len(oneliner) - 6)
        summary_disp = _esc(_last_tool_summary[:max_sum])
        console.print(
            f"{prefix}[{DIM}]{summary_disp}[/]  [{GRAY}]{_esc(oneliner[:safe_w])}[/]"
        )
        _group_rows += 1
        return

    if _quiet_mode:
        if is_error:
            err_text = result[:80].replace("\n", " ")
            if _pt_live():
                from rich.markup import escape as _esc_q
                console.print(f"  [{AMBER}]✗  {name}  {_esc_q(err_text)}  [{t}][/]")
            else:
                sys.stdout.write(
                    f"  {ANSI_AMBER}✗  {name}  {err_text}  [{t}]{ANSI_RESET}\n"
                )
                sys.stdout.flush()
        return

    from rich.markup import escape

    # ── File tools (Read/Write/Edit) non-verbose: grouped summary ──────────────
    if name in _FILE_TOOLS and not verbose:
        if is_error:
            err_text = result[:80].replace("\n", " ")
            console.print(
                f"  [{AMBER}]✗ {name}[/] [{GRAY}]{escape(_last_tool_summary[:60])}[/]"
                f"  [{DIM}]{escape(err_text[:80])}[/]"
            )
        else:
            _pending_file_ops  += 1
            _pending_file_names.append(_last_tool_summary)
            _pending_file_tools.append(name)
        return

    if is_error:
        err_text = result[:80].replace("\n", " ")
        console.print(f"  [{AMBER}]  {_CC_TREE} ✗  {escape(err_text[:safe_w])}[/]")
        return

    # ── TodoWrite: render live task board ──────────────────────────────────────
    if name == "TodoWrite":
        _render_todo_board()
        return

    # ── TodoRead: silent (model reads it, user doesn't need to see the JSON) ──
    if name == "TodoRead":
        return

    # ── MemoryWrite: show which section was written and where ──────────────────
    if name == "MemoryWrite":
        from rich.markup import escape as _esc
        console.print(f"  [{LIME}]  {_CC_TREE} ◎  Memory updated[/]  [{DIM}]{_esc(result[:120])}[/]")
        return

    # ── MemoryRead: silent unless verbose (model reads it; verbose shows length) ─
    if name == "MemoryRead":
        if verbose:
            n_chars = len(result)
            console.print(f"  [{DIM}]  {_CC_TREE} ◇  memory read  ·  {n_chars:,} chars[/]")
        return

    # ── Tree-style output ──────────────────────────────────────────────────────
    lines = result.rstrip("\n").split("\n") if result.strip() else []

    if not lines:
        console.print(f"  [{DIM}]  {_CC_TREE}[/] [{GRAY}](No output)[/]")
        return

    limit = 4 if not verbose else 999
    if name in ("Glob", "Grep") and not verbose:
        limit = 10
    elif verbose:
        limit = 999

    shown     = lines[:limit]
    remaining = max(0, len(lines) - limit)

    _is_diff = any(
        ln.startswith(("diff --git", "--- ", "+++ ", "@@ "))
        for ln in shown[:5]
    )

    for i, line in enumerate(shown):
        tree_pfx = f"  {_CC_TREE} " if i == 0 else "     "
        console.print(_style_result_line(tree_pfx, name, line, _is_diff, safe_w))

    if remaining > 0:
        console.print(f"    [{DIM}]… +{remaining} lines (ctrl+o to expand)[/]")
        _last_truncated = {"name": name, "lines": lines, "is_diff": _is_diff, "shown": limit}
    else:
        _last_truncated = None

    # Timing for slow Bash (≥5s shown)
    if name == "Bash" and elapsed >= 5.0:
        console.print(f"  [{DIM}]  ({t})[/]")


def _style_result_line(tree_pfx: str, name: str, line: str, is_diff: bool, safe_w: int) -> str:
    """Rich-markup for one line of a tool result, shared by the initial
    (truncated) print and expand_last_result()'s reprint of the rest."""
    from rich.markup import escape
    safe_ln = escape(line[:safe_w])
    if is_diff:
        if line.startswith("+") and not line.startswith("+++"):
            return f"{tree_pfx}[{LIME}]{safe_ln}[/]"
        if line.startswith("-") and not line.startswith("---"):
            return f"{tree_pfx}[red]{safe_ln}[/]"
        if line.startswith("@@"):
            return f"{tree_pfx}[{CYAN}]{safe_ln}[/]"
        return f"{tree_pfx}[{DIM}]{safe_ln}[/]"
    if name == "Glob":
        return f"{tree_pfx}[{CYAN}]{safe_ln}[/]"
    if name == "Grep":
        parts = line.split(":", 2)
        if len(parts) >= 3:
            return (f"{tree_pfx}[{DIM}]{escape(parts[0])}:{escape(parts[1])}[/]"
                    f"  [{GRAY}]{escape(parts[2][:safe_w])}[/]")
        return f"{tree_pfx}[{GRAY}]{safe_ln}[/]"
    return f"{tree_pfx}[{GRAY}]{safe_ln}[/]"


def expand_last_result() -> None:
    """Reprint the lines truncated from the last tool result (ctrl+o).

    The truncation hint itself has always said "(ctrl+o to expand)" but no
    keybinding ever called this — confirmed no `c-o` binding existed
    anywhere in repl.py. Bound in repl.py alongside the other REPL
    shortcuts (c-j, escape, s-tab)."""
    global _last_truncated
    if _last_truncated is None:
        console.print(f"  [{DIM}](nothing to expand)[/]")
        return
    name    = _last_truncated["name"]
    lines   = _last_truncated["lines"]
    is_diff = _last_truncated["is_diff"]
    shown   = _last_truncated["shown"]
    rest    = lines[shown:]
    safe_w  = _w() - 2
    console.print(f"  [{DIM}]── expanded: {len(rest)} more line(s) ──[/]")
    for line in rest:
        console.print(_style_result_line("     ", name, line, is_diff, safe_w))
    _last_truncated = None



def _result_summary(name: str, result: str) -> str:
    if not result.strip():
        return "done"
    n_lines = result.count("\n") + 1
    n_bytes = len(result.encode())
    size    = f"{n_bytes / 1024:.1f}kb" if n_bytes >= 1024 else f"{n_bytes}b"
    if name in ("Glob", "Grep"):
        return f"{n_lines} result{'s' if n_lines != 1 else ''}"
    if name == "Bash":
        return f"{n_lines}L output"
    if name == "Read":
        return f"{n_lines} lines"
    if name == "Write":
        return f"wrote {n_lines} lines"
    if name == "Edit":
        return result.split("\n")[0][:60] if result else "edited"
    if name == "WebFetch":
        # Extract domain from result if possible; show char count
        try:
            domain = re.search(r"^#+ (.+)$", result, re.MULTILINE)
            if domain:
                return f"{domain.group(1)[:40]}  ·  {len(result):,} chars"
        except Exception:
            pass
        return f"{len(result):,} chars"
    if name == "WebSearch":
        n_results = result.count("\n\n") + 1
        return f"{n_results} result{'s' if n_results != 1 else ''}"
    return f"{n_lines}L · {size}"


def _summarize_instrument(name: str, inputs: dict) -> str:
    if name in ("Read", "Write", "Edit"):
        return inputs.get("file_path", "")
    if name == "Bash":
        return inputs.get("command", "")[:120]
    if name == "Glob":
        return inputs.get("pattern", "")
    if name == "Grep":
        pat  = inputs.get("pattern", "")
        path = inputs.get("path", "")
        return f"{pat}" + (f"  ·  {path}" if path else "")
    if name == "Semble":
        return inputs.get("query", "")[:80]
    if name in ("WebFetch", "WebSearch"):
        return inputs.get("url", inputs.get("query", ""))[:80]
    if name == "MemoryWrite":
        scope = inputs.get("scope", "project")
        return f"{inputs.get('key', '')}  [{scope}]"
    if name == "MemoryRead":
        return inputs.get("file", "project + global")[:80]
    vals = list(inputs.values())
    return str(vals[0])[:80] if vals else ""


# ── User message echo ──────────────────────────────────────────────────────────

def echo_user_message(text: str, n_attachments: int = 0) -> None:
    """Re-render the submitted prompt as a subtle full-width band (Claude-Code
    style), with a `└ [Image N]` sub-line when attachments are present."""
    import textwrap
    from rich.markup import escape

    text = (text or "").strip()
    if not text and not n_attachments:
        return

    w = _w()
    inner = max(10, w - 4)
    console.print()
    first = True
    for src_line in (text.splitlines() or [""]):
        for wrapped in (textwrap.wrap(src_line, inner) or [""]):
            lead = ">" if first else " "
            body = f"{lead}  {escape(wrapped)}"
            pad  = " " * max(0, w - len(body))
            console.print(f"[on grey11]{body}{pad}[/]")
            first = False

    if n_attachments:
        label = "[Image 1]" if n_attachments == 1 else f"[{n_attachments} images]"
        console.print(f"  [{DIM}]{_CC_TREE} {label}[/]")


# ── Response wrapper ───────────────────────────────────────────────────────────

def open_response_block(
    model: str = "",
    quiet: bool = False,
    loaded_skills: list | None = None,
) -> None:
    """Print the branded response header and set the start time."""
    global _block_start_t, _block_model, _tool_call_counter, _quiet_mode
    global _turn_tool_names, _stream_line_count, _last_tool_summary, _last_tool_badge
    global _pending_file_ops, _pending_file_names, _active_loaded_skills
    global _group_tool, _group_count, _group_rows, _prev_todo_snapshot
    global _buffer_was_replaced

    _block_start_t      = time.monotonic()
    _block_model        = model
    _tool_call_counter  = 0
    _quiet_mode         = quiet
    _turn_tool_names    = []
    _stream_line_count  = 0
    _buffer_was_replaced = False
    _last_tool_summary  = ""
    _last_tool_badge    = ""
    _pending_file_ops   = 0
    _pending_file_names = []
    _active_loaded_skills = list(loaded_skills) if loaded_skills else []
    _group_tool  = ""
    _group_count = 0
    _group_rows  = 0
    _prev_todo_snapshot = []   # new turn always shows the full board first

    console.print()
    sys.stdout.write(ANSI_GRAY)
    sys.stdout.flush()


def close_response_block(verbose: bool, in_tokens: int = 0, out_tokens: int = 0,
                          tok_s: float = 0.0) -> None:
    """Finalize streaming text, then print token/timing footer and optional tally.

    `tok_s`, when provided (from TurnComplete.tok_s), is the decode speed
    measured over just the final turn's actual generation span — preferred
    over deriving it from `elapsed` here, which spans the whole exchange
    since open_response_block() (every tool call + tool execution included),
    badly diluting tok/s for any answer that involved tool use."""
    _flush_file_group()   # flush any remaining Read/Write/Edit group
    sys.stdout.write(ANSI_RESET)
    sys.stdout.flush()
    finalize_response()

    elapsed = time.monotonic() - _block_start_t

    # Tool tally line — shown when tools ran, regardless of quiet mode
    if _turn_tool_names:
        from collections import Counter
        counts = Counter(_turn_tool_names)
        parts  = [f"{n}×{c}" if c > 1 else n for n, c in counts.most_common()]
        tally  = " · ".join(parts)
        console.print(f"  [{DIM}]{_CC_TREE} {tally}[/]")

    if in_tokens or out_tokens:
        t       = _fmt_elapsed(elapsed)
        in_s    = _fmt_tokens(in_tokens)
        out_s   = _fmt_tokens(out_tokens)
        tok_str = f"{in_s}↑  ·  {out_s}↓"
        tps_str = ""
        _tps = tok_s if tok_s > 0 else (out_tokens / elapsed if out_tokens > 0 and elapsed > 0 else 0.0)
        if _tps > 0:
            tps_str = f"  ·  {_tps:.0f}t/s"

        # A turn that burned tokens but produced neither text nor a tool call is
        # a silent failure (often: context too large for the backend, or a
        # truncated/garbled completion) — call it out instead of letting a bare
        # "0↓" footer pass as if nothing was wrong.
        if out_tokens == 0 and not _turn_tool_names:
            console.print(
                f"  [{AMBER}]⚠  Model returned no response this turn[/]"
                f"  [{DIM}]— try rephrasing, /retry, or check /backend connectivity.[/]"
            )

        # Compact "✻ Worked for Xm Xs · ↑ Nk · ↓ Nk" footer — no rule, every mode.
        console.print(
            f"  [{DIM}]✻  Worked for {t}  ·  {tok_str}{tps_str}[/]"
        )

    console.print()


# ── Permission prompt ──────────────────────────────────────────────────────────

def prompt_approval(description: str, settings: dict) -> bool:
    """Arrow-key interactive permission dialog (delegates to wizard_prompt)."""
    try:
        from greenboost_cli.terminal.wizard_prompt import run_approval_picker
        result = run_approval_picker(description)
        if result == "allow-all":
            settings["permission_mode"] = "accept-all"
            emit_ok("Accept-all mode — no more permission prompts this session.")
            return True
        return result == "allow"
    except Exception:
        # Fallback: plain numbered prompt if terminal doesn't support raw mode
        try:
            sys.stdout.write("\n")
            sys.stdout.flush()
            w = _w()
            inner = w - 4
            console.print(f"  [{DIM}]{BOX_TL}{BOX_H * inner}{BOX_TR}[/]")
            console.print(f"  [{DIM}]{BOX_V}[/]  [{AMBER}]⚠  Permission required[/]")
            console.print(f"  [{DIM}]{BOX_V}[/]  [{GRAY}]{description[:inner - 4]}[/]")
            console.print(f"  [{DIM}]{BOX_ML}{BOX_H * inner}{BOX_MR}[/]")
            console.print(f"  [{DIM}]{BOX_V}[/]  [{LIME}][1][/]  [{GRAY}]Allow once[/]")
            console.print(f"  [{DIM}]{BOX_V}[/]  [{AMBER}][2][/]  [{AMBER}]Allow for session[/]")
            console.print(f"  [{DIM}]{BOX_V}[/]  [{DIM}][3][/]  [{GRAY}]Deny[/]")
            console.print(f"  [{DIM}]{BOX_BL}{BOX_H * inner}{BOX_BR}[/]")
            console.print()
            while True:
                ans = input(f"  {ANSI_VIOLET}❯{ANSI_RESET}  ").strip().lower()
                if ans in ("1", "y", "yes"):
                    return True
                if ans in ("2", "a"):
                    settings["permission_mode"] = "accept-all"
                    emit_ok("Accept-all mode enabled.")
                    return True
                if ans in ("3", "n", "no", ""):
                    return False
                console.print(f"  [{AMBER}]Type 1 (allow), 2 (allow-all), or 3 (deny).[/]")
        except (KeyboardInterrupt, EOFError):
            print()
            return False


# ── Compaction progress ────────────────────────────────────────────────────────

def show_compact_progress(
    n_before: int,
    n_after: int,
    tokens_est: int,
    elapsed_s: int = 0,
) -> None:
    """Display compaction header + animated progress bar (Image #8 style)."""
    import time as _t

    w      = _w()
    tok_s  = _fmt_tokens(tokens_est)

    # Elapsed time in "Xm Ys" or "Xs" form
    if elapsed_s >= 60:
        m, s  = elapsed_s // 60, elapsed_s % 60
        time_s = f"{m}m {s}s"
    else:
        time_s = f"{elapsed_s}s"

    console.print(
        f"\n  [{TEAL}]✱[/] [{GRAY}]Compacting conversation…[/]"
        f"  [{DIM}]{n_before} → {n_after} turns  ·  ↑ {tok_s} tokens  ·  {time_s}[/]"
    )

    bar_w = min(40, w - 12)
    steps = [0.15, 0.35, 0.55, 0.72, 0.88, 1.0]
    for frac in steps:
        filled = int(bar_w * frac)
        empty  = bar_w - filled
        pct    = int(frac * 100)
        bar    = f"{'▓' * filled}{'░' * empty}"
        pad    = " " * max(0, w - bar_w - 10)
        sys.stdout.write(
            f"\r  {ANSI_DIM}{bar}{ANSI_RESET}"
            f"  {ANSI_GRAY}{pct}%{ANSI_RESET}{pad}"
        )
        sys.stdout.flush()
        if frac < 1.0:
            _t.sleep(0.025)

    sys.stdout.write("\n")
    sys.stdout.flush()


# ── Session banner ─────────────────────────────────────────────────────────────

def _rag_banner_line() -> str:
    try:
        from greenboost_cli.rag.engine import _load_store, _load_folders
        _, metadata = _load_store()
        folders     = _load_folders()
        if not metadata and not folders:
            return ""
        n_chunks = len(metadata) if metadata else 0
        n_files  = len({m["file"] for m in metadata}) if metadata else 0
        return f"{n_chunks:,} chunks  ·  {n_files} files"
    except Exception:
        return ""


def show_session_banner(settings: dict, session=None) -> None:
    """Startup banner: brand header + model + live system status."""
    model   = settings.get("model", "unknown")
    pmode   = settings.get("permission_mode", "auto")
    project = settings.get("active_project")
    w       = _w()
    sep_len = min(w - 4, 68)
    # Edge-to-edge rule, same look as the bottom input box (no side borders).
    full_rule = f"[{DIM}]{BOX_H * w}[/]"

    def _row(key: str, val: str, val_color: str = GRAY) -> None:
        console.print(f"  [{DIM}]{key:<14}[/][{val_color}]{val}[/]")

    console.print()
    # Brand header: show session name like Claude Code if session has one
    sname = getattr(session, "name", None) if session else None
    if sname:
        console.print(f"  [{TEAL}]◈[/]  [{TEAL}]GreenBoost CLI[/]  [{DIM}]·[/]  [{CYAN}]{sname}[/]")
    else:
        console.print(f"  [{TEAL}]◈[/]  [{TEAL}]GreenBoost CLI[/]")
    console.print(full_rule)

    _row("model", f"{model}  ·  gb-synapse", CYAN)
    _pmode_color = AMBER if pmode in ("accept-all", "autonomous") else GRAY
    _pmode_label = pmode
    if pmode == "autonomous":
        _auto_goal = settings.get("autonomous_goal", "")
        _pmode_label = "autonomous ⚠  /autonomous-coding off to disable"
        if _auto_goal:
            _pmode_label += f"  ·  goal: {_auto_goal[:50]}"
    _row("permissions", _pmode_label, _pmode_color)
    if project:
        _row("project", project, CYAN)

    # gb-synapse server status row
    try:
        from greenboost_cli.slash_commands.backend_cmds import llamacpp_server_status
        _vs = llamacpp_server_status(settings)
        if _vs == "running":
            _row("gb-synapse", "● running  ·  /llamaserve logs", LIME)
        elif _vs == "starting":
            _row("gb-synapse", "◌ loading model…  ·  /llamaserve logs", AMBER)
        else:
            _row("gb-synapse", "○ stopped  ·  /llamaserve", GRAY)
    except Exception:
        pass

    rag_line = _rag_banner_line()
    if rag_line:
        _row("rag", rag_line, GRAY)

    # gb-quant active status
    gb_quant_bits = os.environ.get("GB_QUANT_BITS", "")
    if gb_quant_bits:
        _row("gb-quant", f"active · {gb_quant_bits}", LIME)

    # Token usage today (per-project)
    try:
        from greenboost_cli.memory.brain import project_dir as _pdir
        from greenboost_cli.memory.token_tracker import format_header_line
        _tok_line = format_header_line(_pdir(settings.get("active_project")))
        if _tok_line:
            _row("tokens", _tok_line, DIM)
    except Exception:
        pass

    console.print(full_rule)
    console.print(
        f"  [{DIM}]/help  ·  /backend  ·  /gb-quant  ·  /doctor  ·  /quiet  ·  /dashboard[/]"
    )
    console.print()

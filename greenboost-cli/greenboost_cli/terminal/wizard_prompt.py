"""Interactive question wizard for the AskUserQuestion tool.

Full arrow-key/space-bar TUI picker — no third-party dependencies.

Controls:
  ↑ / ↓   — move cursor
  Space   — toggle option (multi-select) or select+advance (single)
  Enter   — confirm selection
  Esc     — cancel current question (returns None)
  Ctrl-C  — cancel entire wizard (returns None)

Public API
----------
run_question_wizard(questions) → list[dict] | None
    Returns one answer dict per question, or None on cancel.
"""
from __future__ import annotations

import os
import select
import shutil
import sys
import termios
import tty

from greenboost_cli.terminal.theme import (
    console,
    LIME, GRAY, AMBER, DIM, VIOLET,
    BOX_H, BOX_V, BOX_TL, BOX_TR, BOX_BL, BOX_BR, BOX_ML, BOX_MR,
    ANSI_VIOLET, ANSI_LIME, ANSI_AMBER, ANSI_GRAY, ANSI_DIM, ANSI_RESET,
    ANSI_BOLD,
)

_CANCELLED: object = object()

# ── ANSI helpers ──────────────────────────────────────────────────────────────
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_CLEAR_LINE  = "\033[2K\r"


def _term_w() -> int:
    return min(shutil.get_terminal_size((80, 24)).columns, 100)


# ── Raw keyboard reading ──────────────────────────────────────────────────────

def _getch() -> str:
    """Read one logical keypress (handles arrow-key escape sequences)."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1)
        if ch == b"\x1b":
            # Peek for CSI sequence with a short timeout
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                ch2 = os.read(fd, 1)
                if ch2 == b"[":
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if rlist:
                        ch3 = os.read(fd, 1)
                        if ch3 == b"A":
                            return "UP"
                        if ch3 == b"B":
                            return "DOWN"
                        if ch3 == b"C":
                            return "RIGHT"
                        if ch3 == b"D":
                            return "LEFT"
            return "ESC"
        decoded = ch.decode("utf-8", errors="replace")
        return decoded
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── Interactive picker ────────────────────────────────────────────────────────

def _strip_rich(s: str) -> str:
    """Remove Rich markup tags from a string."""
    import re
    return re.sub(r"\[/?[^\[\]]*\]", "", s)


def _picker(
    options: list[dict],
    multi: bool = False,
    title: str = "",
) -> list[int] | None:
    """Arrow-key + Space interactive picker.

    Returns list of 1-based indices on confirm, None on cancel.
    The last entry is always the 'Other / free text…' option.
    """
    n = len(options)
    cursor: int = 0
    selected: set[int] = set()   # 0-based
    w = _term_w()

    def _render_line(i: int) -> str:
        opt = options[i]
        label = _strip_rich(str(opt.get("label", ""))).strip()
        desc  = _strip_rich(str(opt.get("description", ""))).strip()
        is_cursor   = i == cursor
        is_selected = i in selected
        is_other    = i == n - 1   # last option is always "Other"

        # Cursor glyph
        cg = f"{ANSI_VIOLET}{ANSI_BOLD}❯{ANSI_RESET}" if is_cursor else " "

        # Tick glyph (multi-select)
        if multi:
            tg = f"{ANSI_LIME}✓{ANSI_RESET}" if is_selected else f"{ANSI_DIM}○{ANSI_RESET}"
            left = f"  {cg}  {tg}  "
        else:
            left = f"  {cg}  "

        # Label colour
        if is_other:
            label_s = f"{ANSI_AMBER}{label}{ANSI_RESET}"
        elif is_cursor:
            label_s = f"{ANSI_BOLD}{label}{ANSI_RESET}"
        else:
            label_s = f"{ANSI_GRAY}{label}{ANSI_RESET}"

        # Description (dim, truncated)
        avail_desc = w - len(_strip_rich(left)) - len(label) - 4
        desc_s = ""
        if desc and avail_desc > 8 and not is_other:
            desc_s = f"  {ANSI_DIM}{desc[:avail_desc]}{ANSI_RESET}"

        return _CLEAR_LINE + left + label_s + desc_s

    def _draw(initial: bool = False) -> None:
        out: list[str] = []
        if not initial:
            out.append(f"\033[{n}A")   # move cursor up N lines
        for i in range(n):
            out.append(_render_line(i) + "\n")
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    sys.stdout.write(_HIDE_CURSOR)
    try:
        _draw(initial=True)
        while True:
            key = _getch()

            if key == "UP":
                cursor = (cursor - 1) % n
                _draw()

            elif key == "DOWN":
                cursor = (cursor + 1) % n
                _draw()

            elif key == " ":
                if multi:
                    if cursor in selected:
                        selected.discard(cursor)
                    else:
                        selected.add(cursor)
                    _draw()
                else:
                    # Single-select: space = select + confirm
                    _draw()
                    return [cursor + 1]

            elif key in ("\r", "\n"):
                if multi:
                    if not selected:
                        selected.add(cursor)   # nothing ticked → take cursor
                    return sorted(i + 1 for i in selected)
                else:
                    return [cursor + 1]

            elif key in ("\x03", "\x04", "ESC"):
                return None

    finally:
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()


def _ask_freetext(prompt: str = "Free-text answer") -> str | object:
    """Plain line input after the picker returns."""
    # Flush stale bytes left over from raw-mode key reads
    try:
        import termios as _termios
        _termios.tcflush(sys.stdin.fileno(), _termios.TCIFLUSH)
    except Exception:
        pass
    sys.stdout.write(_SHOW_CURSOR)
    sys.stdout.write("\n")
    sys.stdout.flush()
    label = f"  {ANSI_VIOLET}❯{ANSI_RESET}  {ANSI_DIM}{prompt}:{ANSI_RESET}  "
    try:
        return input(label).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return _CANCELLED


# ── Box drawing ───────────────────────────────────────────────────────────────

def _wrap_lines(text: str, width: int) -> list[str]:
    """Greedy word-wrap *text* to *width* columns. Never cuts mid-word."""
    words = text.split()
    lines: list[str] = []
    line:  list[str] = []
    for word in words:
        if len(" ".join(line + [word])) > width:
            if line:
                lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    return lines or [""]


def _box_header(w: int, step: str, header: str) -> None:
    inner = w - 4
    step_s = f" {step} "
    h_s    = f" {header} "
    avail  = inner - len(step_s) - len(h_s) - 2
    filler = BOX_H * max(0, avail)
    console.print(
        f"  [{DIM}]{BOX_TL}{BOX_H}[/]"
        f"[{VIOLET}]{step_s}[/]"
        f"[{DIM}]{BOX_H}[/]"
        f"[{GRAY}]{h_s}[/]"
        f"[{DIM}]{filler}{BOX_TR}[/]"
    )


def _box_question(w: int, question: str) -> None:
    from rich.markup import escape
    inner = w - 4
    for ln in _wrap_lines(question, inner - 4):
        pad = max(0, inner - len(ln) - 2)
        console.print(
            f"  [{DIM}]{BOX_V}[/]  [{GRAY}]{escape(ln)}[/]" + " " * pad
        )


def _box_sep(w: int) -> None:
    console.print(f"  [{DIM}]{BOX_ML}{BOX_H * (w - 4)}{BOX_MR}[/]")


def _box_bot(w: int) -> None:
    console.print(f"  [{DIM}]{BOX_BL}{BOX_H * (w - 4)}{BOX_BR}[/]")


# ── Public API ────────────────────────────────────────────────────────────────

def run_question_wizard(questions: list[dict]) -> list[dict] | None:
    """Render a full stepped wizard for *questions*.

    Returns list of answer dicts (one per question), or None on cancel.
    """
    if not questions:
        return []

    total   = len(questions)
    results: list[dict] = []
    w       = _term_w()

    sys.stdout.write("\n")
    sys.stdout.flush()

    for i, q in enumerate(questions, 1):
        header   = q.get("header", f"Q{i}")
        question = q.get("question", "")
        options  = list(q.get("options", []))
        multi    = bool(q.get("multiSelect", False))

        # Build option list (always add Other at the end)
        picker_opts = list(options) + [{"label": "Other / free text…", "description": ""}]

        # ── Box ────────────────────────────────────────────────────────────
        _box_header(w, f"{i}/{total}", header)
        _box_question(w, question)
        _box_sep(w)

        # Hint row
        if multi:
            console.print(
                f"  [{DIM}]{BOX_V}[/]  [{DIM}]Space = toggle  ·  Enter = confirm  ·  Esc = cancel[/]"
            )
        else:
            console.print(
                f"  [{DIM}]{BOX_V}[/]  [{DIM}]↑↓ = navigate  ·  Enter / Space = select  ·  Esc = cancel[/]"
            )
        _box_bot(w)
        console.print()

        # ── Interactive picker ─────────────────────────────────────────────
        chosen = _picker(picker_opts, multi=multi, title=header)
        if chosen is None:
            return None

        # ── Resolve → answer strings ───────────────────────────────────────
        other_idx = len(picker_opts)   # 1-based index of Other
        answers: list[str] = []
        needs_freetext = other_idx in chosen

        for c in chosen:
            if c != other_idx:
                answers.append(options[c - 1]["label"])

        if needs_freetext:
            sys.stdout.write("\n")
            ft = _ask_freetext("Free-text answer")
            if ft is _CANCELLED:
                return None
            if str(ft).strip():
                answers.append(str(ft).strip())

        if not answers:
            answers = ["(no answer)"]

        # Print confirmation summary
        console.print(
            f"  [{LIME}]✓[/]  [{GRAY}]{header}:[/]  [{VIOLET}]{', '.join(answers)}[/]"
        )

        results.append({"header": header, "question": question, "answers": answers})

        if i < total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
    return results


# ── Public bare picker (reused by slash commands, e.g. /backend) ─────────────

def run_picker(options: list[dict], multi: bool = False, title: str = "") -> list[int] | None:
    """Public entry point for a bare arrow-key picker (no box/header chrome).
    Returns 1-based selected indices, or None on cancel."""
    return _picker(options, multi=multi, title=title)


# ── Approval picker (reused by renderer.prompt_approval) ─────────────────────

def _summarize_call(description: str) -> str:
    """Turn a raw `ToolName([[{'content': '...'}]])`-style call string into a
    clean `ToolName · <args summary>` line. Falls back to the raw string
    (truncated) if it doesn't look like a call."""
    name, _, rest = description.partition("(")
    if not rest:
        return description
    rest = rest.rstrip(")").strip()
    # Prefer a dict value (text after a `'key':`) — usually the most
    # meaningful piece (a file path, a content snippet, a command). Fall
    # back to the first quoted string of any kind.
    import re
    m = re.search(r"""['"]\s*:\s*['"]([^'"]{3,})['"]""", rest)
    if not m:
        m = re.search(r"""['"]([^'"]{3,})['"]""", rest)
    arg_s = m.group(1) if m else rest
    return f"{name.strip()} · {arg_s}"


def run_approval_picker(description: str) -> str:
    """Arrow-key approval dialog. Returns 'allow' | 'allow-all' | 'deny'."""
    opts = [
        {"label": "Allow once",                 "description": "permit this operation"},
        {"label": "Allow for this session",     "description": "no more prompts (accept-all)"},
        {"label": "Deny",                       "description": "skip this operation"},
    ]
    w = _term_w()

    sys.stdout.write("\n")
    sys.stdout.flush()

    inner = w - 4
    console.print(f"  [{DIM}]{BOX_TL}{BOX_H * inner}{BOX_TR}[/]")
    # Body rows previously printed only the LEFT border with no padding or
    # closing BOX_V, so the box rendered open-ended on the right (confirmed
    # in the screenshots — every "Permission required" row trails off with
    # no right edge). Pad each row to `inner` and close it, matching the top
    # rule's width (2 leading spaces + BOX_TL/BOX_V + inner + BOX_TR/BOX_V).
    header_text = "⚠  Permission required"
    header_pad  = " " * max(0, inner - 2 - len(header_text))
    console.print(
        f"  [{DIM}]{BOX_V}[/]  [{AMBER}]{header_text}[/]{header_pad}[{DIM}]{BOX_V}[/]"
    )
    from rich.markup import escape
    summary = _summarize_call(description)
    desc_lines = _wrap_lines(summary, inner - 4)
    shown, overflow = desc_lines[:3], len(desc_lines) > 3
    if overflow:
        shown[-1] = shown[-1].rstrip() + "…"
    for ln in shown:
        pad = " " * max(0, inner - 2 - len(ln))
        console.print(f"  [{DIM}]{BOX_V}[/]  [{GRAY}]{escape(ln)}[/]{pad}[{DIM}]{BOX_V}[/]")
    console.print(f"  [{DIM}]{BOX_BL}{BOX_H * inner}{BOX_BR}[/]")
    console.print()

    chosen = _picker(opts, multi=False, title="Permission")
    if chosen is None:
        return "deny"
    idx = chosen[0]   # 1-based
    if idx == 1:
        return "allow"
    if idx == 2:
        return "allow-all"
    return "deny"

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
from greenboost_cli.terminal.width import (
    display_width, suspend_live, truncate_to_width,
)

_CANCELLED: object = object()

# ── ANSI helpers ──────────────────────────────────────────────────────────────
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_CLEAR_LINE  = "\033[2K\r"


def _term_w() -> int:
    return min(shutil.get_terminal_size((80, 24)).columns, 100)


# ── Raw keyboard reading ──────────────────────────────────────────────────────

#: How long a question waits for a human before answering itself.
#: greenboost-cli is meant to run unattended for days (CLAUDE.md's
#: Unattended-For-Days Must-Rule): asking is fine and expected, but a question
#: nobody is there to answer must not hold the run until someone wanders back.
UNATTENDED_ANSWER_AFTER_S = 300.0

#: Returned by _getch when the wait elapsed with no keypress.
TIMEOUT = "TIMEOUT"


def _getch(timeout: "float | None" = None) -> str:
    """Read one logical keypress (handles arrow-key escape sequences).

    With `timeout`, returns TIMEOUT if nothing is pressed in that many seconds
    , the caller decides what unattended means.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        if timeout is not None:
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if not rlist:
                return TIMEOUT
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
    timeout_s: "float | None" = None,
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
            key = _getch(timeout_s)
            if key is TIMEOUT or key == TIMEOUT:
                return TIMEOUT          # nobody is here; the caller decides

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
    # display_width, not len: a Bash command can carry any glyph, and an
    # East-Asian-Ambiguous character counted as one column overflows the row
    # it was measured for. Same defect class as the status line fixed the same
    # day — see terminal/width.py.
    words = text.split()
    lines: list[str] = []
    line:  list[str] = []
    for word in words:
        # A single token longer than the row (a long path, a URL, a command
        # with no spaces) can never fit by wrapping between words. Break it
        # rather than emit a line that overflows the box.
        while display_width(word) > width:
            head = truncate_to_width(word, width)
            if not head:
                break
            if line:
                lines.append(" ".join(line))
                line = []
            lines.append(head)
            word = word[len(head):]
        if not word:
            continue
        if display_width(" ".join(line + [word])) > width:
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

def run_question_wizard(questions: list[dict],
                       timeout_s: "float | None" = UNATTENDED_ANSWER_AFTER_S,
                       ) -> list[dict] | None:
    """Render a full stepped wizard for *questions*.

    Returns list of answer dicts (one per question), or None on cancel.

    Asking is the expected workflow, especially at the start of a session. What
    is not acceptable is a question outliving the person it was asked of: after
    `timeout_s` with no keypress the wizard answers itself with the safe
    option and says so. Pass `timeout_s=None` to wait indefinitely.
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
        chosen = _picker(picker_opts, multi=multi, title=header,
                         timeout_s=timeout_s)
        if chosen is None:
            return None
        if chosen == TIMEOUT:
            # Unattended: answer it rather than hold the run. The choice is the
            # same safe-option logic /auto-answer uses, and it is recorded in
            # the journal with its reason so /session-report shows exactly what
            # was decided on the user's behalf and why.
            from greenboost_cli.core.autonomy import choose_answer, get_state
            idx, why = choose_answer(q)
            label = (options[idx]["label"] if 0 <= idx < len(options)
                     else "(no answer)")
            try:
                get_state().record("question", header=header, why=why,
                                   answer=label, unattended=True)
            except Exception:
                pass
            console.print(
                f"  [{AMBER}]no answer for "
                f"{int(timeout_s // 60)} min , continuing with "
                f"'{label}'[/]  [{DIM}]({why}; /session-report to review)[/]")
            results.append({"header": header, "question": question,
                            "answers": [label]})
            continue

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
    w = _term_w()

    # The card is many rows drawn one print at a time. Without this the status
    # line paints between them — that is the mangled box in the 2026-08-18
    # screenshots, where the borders land at whatever column the cursor was at.
    with suspend_live():
        return _draw_approval_card_and_pick(description, w)


def _draw_approval_card_and_pick(description: str, w: int) -> str:
    opts = [
        {"label": "Allow once",             "description": "permit this operation"},
        {"label": "Allow for this session", "description": "no more prompts (accept-all)"},
        {"label": "Deny",                   "description": "skip this operation"},
    ]
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
    # display_width: the warning sign U+26A0 is East-Asian-Ambiguous, so it
    # occupies two columns on a CJK-capable font while len() calls it one.
    # One column of over-pad pushes the right border past the margin, the
    # row wraps, and the card renders as the mangled box in the 2026-08-18
    # screenshots.
    header_pad  = " " * max(0, inner - 2 - display_width(header_text))
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
        pad = " " * max(0, inner - 2 - display_width(ln))
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

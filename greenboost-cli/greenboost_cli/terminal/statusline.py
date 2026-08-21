"""
Live status line — animated indicator during model inference.

Visual design:
  ⠹  Thinking  ─────────────────────────────────────────────  2.4s
  ●  Thinking  ──────────────────────────────────  1,234↑ 89↓  ·  2.4s

Phases:
  thinking  → teal braille spinner  ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏
  (final)   → static lime  ●  with full timing + tokens

Overwrites itself in-place via \r — never scrolls.
12 fps (80 ms per frame) — smooth without burning CPU.
Breathing dashes: sin-wave variation ±3 chars for subtle life.

Tier usage (T1/T2/T3) is NOT shown here — that's the idle bottom toolbar's
job (repl.py's _gb_stats_segs, right-aligned amid T1 and T3). A per-turn
tier badge used to duplicate that here too; removed 2026-08-05 (it read as
a second, inconsistent "T2" counter appearing near the bottom-left of the
screen alongside the real one).
"""
from __future__ import annotations

import math
import shutil
import sys
import threading
import time

from greenboost_cli.terminal.theme import (
    ANSI_TEAL, ANSI_VIOLET, ANSI_LIME, ANSI_GRAY, ANSI_DIM, ANSI_RESET,
    ANSI_AMBER,
    SPINNER_THINK, CTX_AMBER_PCT,
    TEAL, DIM,
)
from greenboost_cli.terminal.width import (
    at_line_start, display_width, live_suspended, truncate_to_width, tty_write,
)

# Erase-to-end-of-line. Paired with \r it makes each repaint overwrite the
# previous frame by construction, instead of relying on the new frame being
# padded at least as wide as the old one. See _render() for why padding alone
# was not enough.
_ERASE_LINE = "\033[2K"

# DECAWM off / on. With auto-wrap disabled the terminal TRUNCATES a too-long
# line at the right margin instead of continuing it on the next row. That makes
# the reported artifact structurally impossible: even if a glyph's width is
# still mis-guessed on some exotic font, the status line can no longer push the
# cursor onto a second row and strand its own head in the scrollback.
_WRAP_OFF = "\033[?7l"
_WRAP_ON = "\033[?7h"

_H = "─"

# ── prompt_toolkit integration ───────────────────────────────────────────────
# When a prompt_toolkit bottom toolbar owns the screen, the statusline must not
# write raw \r/ANSI to stdout (it gets escaped/garbled by patch_stdout). Instead
# it feeds its live fields to the toolbar via toolbar_status_fragments(), and
# asks pt to repaint via the registered invalidate callback.
_pt_invalidate = None   # type: ignore  # callable() -> None, set by repl.py
# The StatusLine instance currently live, whichever render path it is using.
# Set on BOTH the pt-toolbar path and the raw \r path: it used to be assigned
# only in the pt branch, which meant nothing tracked the status line during an
# actual model turn (turns run the raw path by construction, since the pt app
# has already returned from prompt()). That is why repl.py's Ctrl-C handler had
# no way to reach it and crashed with `NameError: name 'sl' is not defined`
# (owner report 2026-08-18).
_live_sl: "StatusLine | None" = None
_pt_is_live = None      # type: ignore  # callable() -> bool, set by repl.py

# Set by repl._suspend_pt_for_wizard()/_resume_pt_after_wizard() while a wizard
# (approval picker / AskUserQuestion) owns the raw terminal. While True, the
# background repaint loop below must not call app.invalidate() — a still-ticking
# invalidate can race the wizard's raw-ANSI redraw and double its rendered rows.
_wizard_active = False


def set_pt_mode(invalidate_cb) -> None:
    """Register the prompt_toolkit invalidate callback. Call once at startup."""
    global _pt_invalidate
    _pt_invalidate = invalidate_cb


def set_pt_live_probe(cb) -> None:
    """Register a callback that returns True only when the pt app is actively
    running (the idle input prompt is live). During model turns the app is not
    running and the callback returns False, so raw \\r animation is used instead
    of the toolbar repaint path — which is a no-op during turns anyway."""
    global _pt_is_live
    _pt_is_live = cb


def _use_pt() -> bool:
    """True only when the pt app is live and can process invalidate calls.

    During a model turn _pt_session.app.is_running == False (the prompt()
    call already returned when the user submitted their input). Raw \\r
    animation is safe then because patch_stdout is only installed inside
    prompt(). Returning False here routes the statusline to the raw _run
    loop, giving live animated feedback for every turn."""
    if _pt_invalidate is None or _pt_is_live is None:
        return False
    try:
        return bool(_pt_is_live())
    except Exception:
        return False


def _is_tty() -> bool:
    """True only when stdout is a real terminal.

    The animated statusline repaints via a bare ``\\r`` every 0.08 s, which only
    overwrites in place on a TTY. When stdout is a pipe or a file — ``gb -p``,
    the AI Factory's ``_invoke_agent``, any CI capture — the carriage returns are
    just bytes, so every frame ACCUMULATES: one 72-second turn measured 259 KB of
    spinner frames around ~200 bytes of actual answer. That pollutes captured
    output and, worse, inflates the gate/retry text the factory feeds back into
    the next prompt, which is scarce context on a slow local model."""
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def renders_in_toolbar() -> bool:
    """True when the live status is painted into prompt_toolkit's bottom
    toolbar rather than written to stdout with a bare carriage return.

    The distinction decides whether the status line can coexist with streamed
    model output. In raw mode it cannot: both write to the same line and the
    result is torn. In toolbar mode it can, because the toolbar is a separate,
    pt-owned row that printed text scrolls above.

    repl.py uses this so a long, slow response keeps showing elapsed time and
    token counts instead of going completely silent the moment the first token
    lands. At the decode rates this box reaches on a deep conversation, that
    silence lasts minutes and is indistinguishable from a hang."""
    return _use_pt()


def set_wizard_active(active: bool) -> None:
    """Toggle wizard-active state — see _wizard_active docstring above."""
    global _wizard_active
    _wizard_active = active


def is_prefilling() -> "bool | None":
    """True while the model is still prefilling, False once it is decoding.

    None means there is no live status line to ask — the caller must treat that
    as "unknown" rather than assuming either state.

    Exists so a caller outside the turn function can ask this without reaching
    for a local variable that is not in its scope. repl.py's Ctrl-C handler did
    exactly that (`getattr(sl, "_first_out_ts", None)`, with `sl` defined in a
    different function) and crashed the whole CLI with a NameError on the first
    Ctrl-C during a turn — taking its memory-pool cleanup down with it.
    """
    sl = _live_sl
    if sl is None:
        return None
    return getattr(sl, "_first_out_ts", None) is None


def toolbar_status_fragments() -> list[tuple[str, str]]:
    """Return (style, text) fragments for the live status, or [] when idle."""
    sl = _live_sl
    if sl is None:
        return []
    return sl._toolbar_fragments()


class StatusLine:
    """
    Animated status bar during model inference.

    Usage::

        sl = StatusLine()
        sl.start("Thinking")
        sl.update(in_tokens=1234, out_tokens=89)  # from TurnComplete
        sl.stop()                          # final static line + newline
    """

    def __init__(self) -> None:
        self._phase: str    = "Thinking"
        self._in_tok: int   = 0
        self._out_tok: int  = 0
        self._ctx_pct: float | None = None   # 0.0–1.0 context fill estimate
        self._start: float  = 0.0
        self._frame: int    = 0
        self._lock  = threading.Lock()
        # Monotonic time the first output token arrived, or None while the model
        # is still prefilling. Decode rate must be measured from HERE, not from
        # turn start, or a long prefill silently divides the number down. See
        # the tps computation in _render().
        self._first_out_ts: "float | None" = None
        # Server-measured decode rate from the last TurnComplete. Preferred over
        # anything computed here: gb_synapse measures first-token-to-last, which
        # is the real decode span, while this process only ever sees the totals
        # after the fact. Survives start(), because a turn that calls three tools
        # is still one turn and its decode rate did not change when a tool ran.
        self._tok_s: float = 0.0
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._prev_vis_len: int = 0   # widest frame painted so far this run — see _render

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self, phase: str = "Thinking") -> None:
        self._phase    = phase
        self._start    = time.monotonic()
        self._frame    = 0
        self._prev_vis_len = 0
        self._first_out_ts = None
        # _tok_s is NOT cleared here. start() runs again after every tool call
        # (_restart_sl), and clearing _first_out_ts there is what left the
        # post-tool blocks with a token count and no rate to divide it by: the
        # decode already happened, before the tool ran. Keeping the measured
        # rate is what puts a t/s on every block instead of only the last one.
        self._stop_evt.clear()
        if not _is_tty():
            # Non-interactive: no animation thread at all. stop() still emits one
            # final static line, so headless logs keep the turn summary.
            return
        global _live_sl
        _live_sl = self
        if _use_pt():
            # pt app is live at the idle prompt — drive via toolbar repaint.
            self._thread = threading.Thread(target=self._run_pt, daemon=True, name="gb-sl")
            self._thread.start()
            return
        # pt app is not running (turn in progress) or not configured — use raw
        # \r animation directly on stdout (patch_stdout is inactive during turns).
        self._thread = threading.Thread(target=self._run, daemon=True, name="gb-sl")
        self._thread.start()

    def update(
        self,
        phase: str | None       = None,
        in_tokens: int | None   = None,
        out_tokens: int | None  = None,
        ctx_pct: float | None   = None,
        tok_s: float | None     = None,
    ) -> None:
        with self._lock:
            if phase     is not None: self._phase    = phase
            if in_tokens is not None: self._in_tok   = in_tokens
            if out_tokens is not None:
                if out_tokens > 0 and self._first_out_ts is None:
                    self._first_out_ts = time.monotonic()
                self._out_tok = out_tokens
            if ctx_pct   is not None: self._ctx_pct  = ctx_pct
            if tok_s     is not None and tok_s > 0: self._tok_s = tok_s

    def stop(self) -> None:
        self._stop_evt.set()
        self._forget_live()
        if self._thread:
            self._thread.join(timeout=0.5)
        if not _is_tty():
            # One plain summary line, no \r and no hint — safe to capture.
            tty_write(self._render(final=True) + "\n")
            return
        if _use_pt():
            self._pt_deactivate()
            return
        self._clear_hint()
        # Erase before the final frame too: the static line inherits the row the
        # animation was using, so anything wider left over there must go.
        tty_write(
            _WRAP_OFF + "\r" + _ERASE_LINE + self._render(final=True) + _WRAP_ON + "\n"
        )

    def _forget_live(self) -> None:
        """Drop the module-level live reference if it still points at us.

        Called from every termination route, not just the pt one. Without this
        a finished turn leaves a stale instance behind and is_prefilling()
        answers about a turn that already ended."""
        global _live_sl
        if _live_sl is self:
            _live_sl = None

    def cancel(self) -> None:
        """Stop the animation and erase the line without printing a final static line."""
        self._stop_evt.set()
        self._forget_live()
        if self._thread:
            self._thread.join(timeout=0.5)
        if not _is_tty():
            return  # nothing was painted, so there is nothing to erase
        if _use_pt():
            self._pt_deactivate()
            return
        self._clear_hint()
        tty_write("\r" + _ERASE_LINE)

    # ── prompt_toolkit mode ───────────────────────────────────────────────────

    def _pt_deactivate(self) -> None:
        global _live_sl
        if _live_sl is self:
            _live_sl = None
        if _pt_invalidate is not None:
            try:
                _pt_invalidate()
            except Exception:
                pass

    def _run_pt(self) -> None:
        """pt mode: no stdout writes — just bump the frame and ask pt to repaint."""
        while not self._stop_evt.is_set():
            if _pt_invalidate is not None and not _wizard_active:
                try:
                    _pt_invalidate()
                except Exception:
                    pass
            self._stop_evt.wait(0.25)
            with self._lock:
                self._frame += 1

    def _toolbar_fragments(self) -> list[tuple[str, str]]:
        """(style, text) fragments for the live status, for the pt bottom toolbar."""
        with self._lock:
            phase   = self._phase
            in_tok  = self._in_tok
            out_tok = self._out_tok
            frame   = self._frame
            measured_tps = self._tok_s
            first_out    = self._first_out_ts
        elapsed = time.monotonic() - self._start
        sp_char = SPINNER_THINK[frame % len(SPINNER_THINK)]

        frags: list[tuple[str, str]] = [
            (f"fg:{TEAL}", f"{sp_char}  {phase}"),
        ]
        if in_tok or out_tok:
            frags.append((f"fg:{DIM}", f"  ·  {in_tok:,}↑  {out_tok:,}↓"))
            span = (time.monotonic() - first_out) if first_out else 0.0
            tps = measured_tps if measured_tps > 0 else (
                out_tok / span if out_tok > 0 and span > 0 else 0.0)
            if tps > 0:
                frags.append((f"fg:{TEAL}", f"  ·  {tps:.0f}t/s"))
        frags.append((f"fg:{DIM}", f"  ·  {elapsed:.1f}s"))
        return frags

    # ── Bottom hint bar ────────────────────────────────────────────────────

    def _show_hint(self) -> None:
        """Print a one-line keyboard hint below the status line."""
        # The hint carries a `·` too, so it was mis-measured the same way the
        # status line was — and it padded to the full width, which pushed it one
        # column over and cost the \033[1A its row. Erase instead of pad.
        # ctrl+p is listed here too, not just on the idle toolbar: mid-turn is
        # exactly when an operator decides they want the GPU back for something
        # else, and a lever they cannot see is a lever they will not use.
        hint = ("  esc to interrupt  ·  ctrl+p pause/resume  ·  ctrl+i compact  ·  "
                "type to queue next prompt")
        tty_write(
            f"\n{_ERASE_LINE}{ANSI_DIM}{hint}{ANSI_RESET}"
            f"\033[1A"   # cursor up — keep status line owning the bottom row
        )

    def _clear_hint(self) -> None:
        """Erase the hint line printed below the status line."""
        tty_write(f"\n{_ERASE_LINE}\033[1A")

    # ── Internal ───────────────────────────────────────────────────────────

    def _run(self) -> None:
        _hint_shown = False
        while not self._stop_evt.is_set():
            # live=True: dropped, atomically under the lock, if a multi-row
            # block is being drawn or the cursor does not own a fresh row.
            tty_write(_WRAP_OFF + "\r" + _ERASE_LINE + self._render() + _WRAP_ON,
                      live=True, owner="statusline")
            # Show hint line after first 0.5 s so it doesn't flash on fast responses
            if not _hint_shown and (time.monotonic() - self._start) > 0.5:
                self._show_hint()
                _hint_shown = True
            self._stop_evt.wait(0.08)
            with self._lock:
                self._frame += 1

    def _render(self, final: bool = False) -> str:
        with self._lock:
            phase   = self._phase
            in_tok  = self._in_tok
            out_tok = self._out_tok
            ctx_pct = self._ctx_pct
            frame   = self._frame
            first_out = self._first_out_ts
            measured_tps = self._tok_s

        width   = shutil.get_terminal_size((80, 24)).columns
        safe_w  = width - 2
        elapsed = time.monotonic() - self._start

        # ── Right side: ctx% · tokens · tps · elapsed ────────────────────────
        right_parts: list[str] = []
        tps_str = ""
        ttft_str = ""
        ctx_str = ""
        if ctx_pct is not None and ctx_pct > 0.05:
            ctx_str = f"{int(ctx_pct * 100)}%"
            right_parts.append(ctx_str)
        if in_tok or out_tok:
            right_parts.append(f"{in_tok:,}↑  {out_tok:,}↓")
            # DECODE rate, measured over the decode span only.
            #
            # This used to be out_tok / elapsed, i.e. divided by the WHOLE turn
            # including prefill, and that is not a decode rate — it is a
            # wall-clock average that collapses whenever the prompt is long.
            # Measured on this box 2026-08-18: a turn with 27,438 prompt tokens
            # took 73.5 s, of which 68.3 s was time-to-first-token and ~5 s was
            # decode of 81 tokens. The old formula displayed "1t/s". The real
            # decode rate was ~16 tok/s. Reading 1t/s for months is what made
            # decode look like the problem when prefill was.
            #
            # gb_synapse's own proxy-side tok_s_measured has always measured the
            # correct span (first token to last); this display was the outlier.
            decode_span = (time.monotonic() - first_out) if first_out else 0.0
            tps = 0.0
            if measured_tps > 0:
                tps = measured_tps
            elif out_tok > 0 and decode_span > 0:
                tps = out_tok / decode_span
            if tps > 0:
                tps_str = f"{tps:.0f}t/s"
                right_parts.append(tps_str)
            # Surface prefill explicitly rather than letting it hide inside the
            # elapsed figure — on a long conversation it is the dominant cost.
            if first_out is not None:
                ttft = first_out - self._start
                if ttft >= 1.0:
                    ttft_str = f"ttft {ttft:.0f}s"
                    right_parts.append(ttft_str)
        right_parts.append(f"{elapsed:.1f}s")
        right_plain = "  ·  ".join(right_parts)

        # Build ANSI-colored right side (for \r context)
        right_ansi_parts: list[str] = []
        if ctx_str:
            ctx_color = ANSI_AMBER if (ctx_pct or 0) >= CTX_AMBER_PCT else ANSI_DIM
            right_ansi_parts.append(f"{ctx_color}{ctx_str}{ANSI_RESET}")
        if in_tok or out_tok:
            right_ansi_parts.append(f"{ANSI_DIM}{in_tok:,}↑  {out_tok:,}↓{ANSI_RESET}")
            if tps_str:
                right_ansi_parts.append(f"{ANSI_TEAL}{tps_str}{ANSI_RESET}")
            if ttft_str:
                # Amber: on a deep conversation this is the number that hurts,
                # and it should not read as neutral decoration.
                right_ansi_parts.append(f"{ANSI_AMBER}{ttft_str}{ANSI_RESET}")
        right_ansi_parts.append(f"{ANSI_GRAY}{elapsed:.1f}s{ANSI_RESET}")
        right_ansi = f"  {ANSI_DIM}·{ANSI_RESET}  ".join(right_ansi_parts)

        # ── Spinner + phase label ──────────────────────────────────────────
        if final:
            sp_char  = "●"
            sp_color = ANSI_LIME
            lbl_color = ANSI_LIME
        else:
            sp_char  = SPINNER_THINK[frame % len(SPINNER_THINK)]
            sp_color = ANSI_TEAL
            lbl_color = ANSI_TEAL

        # ── Dash fill (breathing sine-wave, DOWNWARD only, when not final) ──
        # Breathing by +/-3 let a frame render WIDER than safe_w; the next,
        # narrower frame then padded only to safe_w, stranding 2-3 chars of
        # the wider frame's tail at the right edge (the "1.8s8s"/"2.5sss"
        # artifact, confirmed live) — and stop()'s final render used
        # breath=0, so the widest frame's leftover tail could survive into
        # the STATIC line too. Breathing only ever shrinks the dash run now,
        # so a frame is never wider than base_dashes; padding against
        # max(safe_w, prev_vis_len) is the belt-and-braces case where a
        # shrinking frame still needs to overwrite a wider previous one.
        # Measure in terminal COLUMNS, not codepoints. `·`, `↑` and `↓` are
        # East-Asian-Ambiguous and render at two columns on a CJK-capable font,
        # so the old len()-based count undershot by ~5 on a frame that only kept
        # 2 columns of slack. The frame then wrapped, and every later repaint's
        # \r landed on the wrapped row, stranding the first row in scrollback
        # permanently (owner screenshots, 2026-08-18). display_width() counts
        # those wide while keeping the `─` fill narrow — see width.py.
        fixed_vis = (
            2 + display_width(sp_char) + 2 + display_width(phase)
            + 2 + 2 + display_width(right_plain)
        )
        base_dashes = max(4, safe_w - fixed_vis)
        if final:
            dashes = base_dashes
        else:
            breath = -abs(int(math.sin(frame * 0.35) * 3))
            dashes = max(4, min(base_dashes, base_dashes + breath))

        line = (
            f"  {sp_color}{sp_char}{ANSI_RESET}"
            f"  {lbl_color}{phase}{ANSI_RESET}"
            f"  {ANSI_DIM}{'─' * dashes}{ANSI_RESET}"
            f"  {right_ansi}"
        )
        # Hard clamp, in two steps. `dashes` has a floor of 4, so a narrow
        # terminal or a long phase label can still overflow the arithmetic
        # above. Drop the fill run first, since that is the part with no
        # information in it.
        overflow = display_width(line) - safe_w
        if overflow > 0 and dashes > 0:
            dashes = max(0, dashes - overflow)
            line = (
                f"  {sp_color}{sp_char}{ANSI_RESET}"
                f"  {lbl_color}{phase}{ANSI_RESET}"
                + (f"  {ANSI_DIM}{'─' * dashes}{ANSI_RESET}" if dashes else "")
                + f"  {right_ansi}"
            )
        # With the fill fully gone the fixed content (spinner + label + the
        # right-hand readout) can still be wider than the terminal — a ~60
        # column window is enough. Cut it rather than let the terminal wrap it.
        if display_width(line) > safe_w:
            line = truncate_to_width(line, safe_w) + ANSI_RESET
        # No trailing padding: \033[2K in _run() and stop() already erased the
        # row, so overwriting a wider previous frame no longer needs the frame
        # itself to be wide. _prev_vis_len is kept only for the pt toolbar path.
        self._prev_vis_len = display_width(line)
        return line

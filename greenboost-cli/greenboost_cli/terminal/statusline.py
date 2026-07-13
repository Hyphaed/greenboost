"""
Live status line — animated indicator during model inference.

Visual design:
  ⠹  Thinking  ─────────────────────────────────────────────  2.4s
  ●  Thinking  ──────────────────────────────────  1,234↑ 89↓  ·  T1  ·  2.4s

Phases:
  thinking  → teal braille spinner  ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏
  (final)   → static lime  ●  with full timing + tokens + tier

Overwrites itself in-place via \r — never scrolls.
12 fps (80 ms per frame) — smooth without burning CPU.
Breathing dashes: sin-wave variation ±3 chars for subtle life.
Tier badge color: T1=teal · T2=lavender · T3=coral
"""
from __future__ import annotations

import math
import shutil
import sys
import threading
import time

from greenboost_cli.terminal.theme import (
    ANSI_TEAL, ANSI_VIOLET, ANSI_LIME, ANSI_GRAY, ANSI_DIM, ANSI_RESET,
    ANSI_T1, ANSI_T2, ANSI_T3, ANSI_AMBER,
    SPINNER_THINK, CTX_AMBER_PCT,
    TEAL, DIM, LAVENDER, CORAL,
)

_H = "─"

# ── prompt_toolkit integration ───────────────────────────────────────────────
# When a prompt_toolkit bottom toolbar owns the screen, the statusline must not
# write raw \r/ANSI to stdout (it gets escaped/garbled by patch_stdout). Instead
# it feeds its live fields to the toolbar via toolbar_status_fragments(), and
# asks pt to repaint via the registered invalidate callback.
_pt_invalidate = None   # type: ignore  # callable() -> None, set by repl.py
_pt_active_sl: "StatusLine | None" = None   # the StatusLine instance currently live, for the toolbar to read
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


def set_wizard_active(active: bool) -> None:
    """Toggle wizard-active state — see _wizard_active docstring above."""
    global _wizard_active
    _wizard_active = active


def toolbar_status_fragments() -> list[tuple[str, str]]:
    """Return (style, text) fragments for the live status, or [] when idle."""
    sl = _pt_active_sl
    if sl is None:
        return []
    return sl._toolbar_fragments()


def _tier_ansi(label: str) -> str:
    """Map a GB tier label to its ANSI color code."""
    upper = label.upper()
    if upper.startswith("T1"):
        return ANSI_T1
    if upper.startswith("T2"):
        return ANSI_T2
    if upper.startswith("T3"):
        return ANSI_T3
    return ANSI_TEAL


class StatusLine:
    """
    Animated status bar during model inference.

    Usage::

        sl = StatusLine()
        sl.start("Thinking")
        sl.update(gb_tier="T2")           # add GB info
        sl.update(in_tokens=1234, out_tokens=89)  # from TurnComplete
        sl.stop()                          # final static line + newline
    """

    def __init__(self) -> None:
        self._phase: str    = "Thinking"
        self._in_tok: int   = 0
        self._out_tok: int  = 0
        self._gb_tier: str | None = None
        self._ctx_pct: float | None = None   # 0.0–1.0 context fill estimate
        self._start: float  = 0.0
        self._frame: int    = 0
        self._lock  = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self, phase: str = "Thinking") -> None:
        self._phase    = phase
        self._start    = time.monotonic()
        self._frame    = 0
        self._stop_evt.clear()
        if _use_pt():
            # pt app is live at the idle prompt — drive via toolbar repaint.
            global _pt_active_sl
            _pt_active_sl = self
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
        gb_tier: str | None     = None,
        ctx_pct: float | None   = None,
    ) -> None:
        with self._lock:
            if phase     is not None: self._phase    = phase
            if in_tokens is not None: self._in_tok   = in_tokens
            if out_tokens is not None: self._out_tok = out_tokens
            if gb_tier   is not None: self._gb_tier  = gb_tier
            if ctx_pct   is not None: self._ctx_pct  = ctx_pct

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        if _use_pt():
            self._pt_deactivate()
            return
        self._clear_hint()
        sys.stdout.write("\r" + self._render(final=True) + "\n")
        sys.stdout.flush()

    def cancel(self) -> None:
        """Stop the animation and erase the line without printing a final static line."""
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        if _use_pt():
            self._pt_deactivate()
            return
        self._clear_hint()
        w = shutil.get_terminal_size((80, 24)).columns
        sys.stdout.write("\r" + " " * w + "\r")
        sys.stdout.flush()

    # ── prompt_toolkit mode ───────────────────────────────────────────────────

    def _pt_deactivate(self) -> None:
        global _pt_active_sl
        if _pt_active_sl is self:
            _pt_active_sl = None
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
            gb_tier = self._gb_tier
            frame   = self._frame
        elapsed = time.monotonic() - self._start
        sp_char = SPINNER_THINK[frame % len(SPINNER_THINK)]

        frags: list[tuple[str, str]] = [
            (f"fg:{TEAL}", f"{sp_char}  {phase}"),
        ]
        if in_tok or out_tok:
            frags.append((f"fg:{DIM}", f"  ·  {in_tok:,}↑  {out_tok:,}↓"))
            if out_tok > 0 and elapsed > 0:
                tps = out_tok / elapsed
                frags.append((f"fg:{TEAL}", f"  ·  {tps:.0f}t/s"))
        if gb_tier:
            tier_style = {"T1": f"fg:{TEAL}", "T2": f"fg:{LAVENDER}", "T3": f"fg:{CORAL}"}.get(
                gb_tier.upper()[:2], f"fg:{TEAL}"
            )
            frags.append((tier_style, f"  ·  {gb_tier}"))
        frags.append((f"fg:{DIM}", f"  ·  {elapsed:.1f}s"))
        return frags

    # ── Bottom hint bar ────────────────────────────────────────────────────

    def _show_hint(self) -> None:
        """Print a one-line keyboard hint below the status line."""
        w = shutil.get_terminal_size((80, 24)).columns
        hint = "  esc to interrupt  ·  type to queue next prompt"
        padded = hint + " " * max(0, w - len(hint))
        sys.stdout.write(
            f"\n{ANSI_DIM}{padded}{ANSI_RESET}"
            f"\033[1A"   # cursor up — keep status line owning the bottom row
        )
        sys.stdout.flush()

    def _clear_hint(self) -> None:
        """Erase the hint line printed below the status line."""
        w = shutil.get_terminal_size((80, 24)).columns
        sys.stdout.write(f"\n{' ' * w}\033[1A")
        sys.stdout.flush()

    # ── Internal ───────────────────────────────────────────────────────────

    def _run(self) -> None:
        _hint_shown = False
        while not self._stop_evt.is_set():
            sys.stdout.write("\r" + self._render())
            sys.stdout.flush()
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
            gb_tier = self._gb_tier
            ctx_pct = self._ctx_pct
            frame   = self._frame

        width   = shutil.get_terminal_size((80, 24)).columns
        safe_w  = width - 2
        elapsed = time.monotonic() - self._start

        # ── Right side: ctx% · tokens · tps · tier · elapsed ─────────────────
        right_parts: list[str] = []
        tps_str = ""
        ctx_str = ""
        if ctx_pct is not None and ctx_pct > 0.05:
            ctx_str = f"{int(ctx_pct * 100)}%"
            right_parts.append(ctx_str)
        if in_tok or out_tok:
            right_parts.append(f"{in_tok:,}↑  {out_tok:,}↓")
            if out_tok > 0 and elapsed > 0:
                tps = out_tok / elapsed
                tps_str = f"{tps:.0f}t/s"
                right_parts.append(tps_str)
        if gb_tier:
            right_parts.append(f"{gb_tier}")
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
        if gb_tier:
            tier_col = _tier_ansi(gb_tier)
            right_ansi_parts.append(f"{tier_col}{gb_tier}{ANSI_RESET}")
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

        # ── Dash fill (breathing sine-wave ±3 when not final) ─────────────
        fixed_vis = 2 + 1 + 2 + len(phase) + 2 + 2 + len(right_plain)
        base_dashes = max(4, safe_w - fixed_vis)
        if final:
            dashes = base_dashes
        else:
            breath = int(math.sin(frame * 0.35) * 3)
            dashes = max(4, base_dashes + breath)

        line = (
            f"  {sp_color}{sp_char}{ANSI_RESET}"
            f"  {lbl_color}{phase}{ANSI_RESET}"
            f"  {ANSI_DIM}{'─' * dashes}{ANSI_RESET}"
            f"  {right_ansi}"
        )
        vis_len = fixed_vis + dashes
        return line + " " * max(0, safe_w - vis_len)

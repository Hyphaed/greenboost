"""
Brand color palette, icon system, spinners, and emit helpers for GreenBoost CLI.

Colors match the web dashboard CSS vars exactly — cohesive look across CLI + browser.
All terminal output goes through this module; never use raw ANSI codes elsewhere.
"""
from __future__ import annotations

from rich.console import Console
from rich.theme import Theme
from rich.markdown import Markdown

# ── Brand palette ──────────────────────────────────────────────────────────────
TEAL     = "#3ab0a0"   # primary: GreenBoost brand, response headers, banner accent
VIOLET   = "#7c6dff"   # interactive: REPL prompt glyph, thinking spinner
LIME     = "#a3e635"   # success: ok states, streaming active, result indicators
CYAN     = "#22d3ee"   # tools: file paths, data highlights, tool card accents
GRAY     = "#d4dbe8"   # body text, secondary info, streaming response
AMBER    = "#fbbf24"   # warnings, errors, wizard prompts
RED      = "#f87171"   # error states (hard failures)
DIM      = "#8b97b3"   # muted: dashes, timestamps, decorators

# Tier-specific semantic colors — match dashboard CSS vars
LAVENDER = "#a78bfa"   # T2 RAM tier
CORAL    = "#d4604a"   # T3 NVMe tier

# ── Rich theme ────────────────────────────────────────────────────────────────
_THEME = Theme({
    "teal":     f"bold {TEAL}",
    "violet":   f"bold {VIOLET}",
    "lime":     LIME,
    "cyan":     CYAN,
    "gray":     GRAY,
    "amber":    AMBER,
    "red":      RED,
    "dim":      f"dim {DIM}",
    "muted":    f"dim {GRAY}",
    "hdr":      f"bold {TEAL}",
    "ok":       LIME,
    "warn":     AMBER,
    "err":      RED,
    "tool":     f"bold {CYAN}",
})

console = Console(theme=_THEME, highlight=False)

# ── Raw ANSI helpers (for \r overwrite contexts where Rich can't be used) ────
def _hex_to_ansi_fg(h: str) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"\033[38;2;{r};{g};{b}m"

ANSI_TEAL    = _hex_to_ansi_fg(TEAL)
ANSI_VIOLET  = _hex_to_ansi_fg(VIOLET)
ANSI_LIME    = _hex_to_ansi_fg(LIME)
ANSI_CYAN    = _hex_to_ansi_fg(CYAN)
ANSI_GRAY    = _hex_to_ansi_fg(GRAY)
ANSI_AMBER   = _hex_to_ansi_fg(AMBER)
ANSI_DIM     = _hex_to_ansi_fg(DIM)
ANSI_RESET   = "\033[0m"
ANSI_BOLD    = "\033[1m"
ANSI_FAINT   = "\033[2m"

# Tier ANSI colors — for statusline tier badge
ANSI_T1 = ANSI_TEAL                     # T1 VRAM  — teal
ANSI_T2 = _hex_to_ansi_fg(LAVENDER)     # T2 RAM   — lavender
ANSI_T3 = _hex_to_ansi_fg(CORAL)        # T3 NVMe  — coral

# ── Spinner frame arrays ──────────────────────────────────────────────────────
SPINNER_THINK = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_TOOL  = ["◐", "◓", "◑", "◒"]

# ── Context-fill thresholds ───────────────────────────────────────────────────
# Single source of truth — referenced by repl.py (emit_warn) and statusline.py (colour).
CTX_WARN_PCT  = 0.85   # emit_warn when context fills past this fraction
CTX_AMBER_PCT = 0.75   # statusline context bar turns amber at this fraction

# ── Box drawing ───────────────────────────────────────────────────────────────
BOX_H  = "─"
BOX_V  = "│"
BOX_TL = "╭"
BOX_TR = "╮"
BOX_BL = "╰"
BOX_BR = "╯"
BOX_ML = "├"
BOX_MR = "┤"

# ── Tool icons — one character per tool, unique per type ──────────────────────
TOOL_ICONS: dict[str, str] = {
    "Read":      "▷",    # file read
    "Write":     "◉",    # write / commit
    "Edit":      "◈",    # edit / modify
    "Bash":      "›",    # shell
    "Glob":      "✦",    # pattern match
    "Grep":      "⌕",    # search
    "WebFetch":  "↓",    # fetch / download
    "Screenshot": "▤",   # UI capture
    "WebSearch": "⊹",    # web search
    "TodoWrite":   "☑",   # task list write
    "TodoRead":    "☐",   # task list read
    "MemoryWrite": "◎",   # persistent memory write
    "MemoryRead":  "◇",   # persistent memory read
}

# ── Icon constants ────────────────────────────────────────────────────────────
ICON_OK       = f"[{LIME}]✓[/]"
ICON_FAIL     = f"[{AMBER}]✗[/]"
ICON_WARN     = f"[{AMBER}]⚠[/]"
ICON_INFO     = f"[{TEAL}]◈[/]"
ICON_ACTIVE   = f"[{LIME}]●[/]"
ICON_INACTIVE = f"[{GRAY}]○[/]"
ICON_PROMPT   = f"[{VIOLET}]❯[/]"
ICON_SYSTEM   = f"[{GRAY}]◎[/]"
ICON_LOADING  = f"[{GRAY}]⟳[/]"
ICON_EXPAND   = f"[{CYAN}]▸[/]"

SEPARATOR = f"[{DIM}]{'─' * 60}[/]"


# ── Layout helpers ────────────────────────────────────────────────────────────

def emit_header(title: str, subtitle: str = "") -> None:
    console.print(f"[bold white]{title}[/]")
    console.print(SEPARATOR)
    if subtitle:
        console.print(f"[{GRAY}]{subtitle}[/]")


def emit_section(title: str) -> None:
    console.print()
    console.print(f"[hdr]{title}[/]")
    console.print(SEPARATOR)
    console.print()


def emit_step(n: int, total: int, label: str) -> None:
    console.print(f"[{LIME}][{n}/{total}][/] [{GRAY}]{label}[/]")


# ── Status helpers ────────────────────────────────────────────────────────────

def emit_ok(msg: str) -> None:
    console.print(f"{ICON_OK} [{GRAY}]{msg}[/]")


def emit_warn(msg: str) -> None:
    console.print(f"{ICON_WARN} [{AMBER}]{msg}[/]")


def emit_err(msg: str) -> None:
    console.print(f"{ICON_FAIL} [{AMBER}]{msg}[/]")


def emit_info(msg: str) -> None:
    console.print(f"{ICON_INFO} [{GRAY}]{msg}[/]")


def emit_muted(msg: str) -> None:
    console.print(f"[muted]{msg}[/]")


# ── Markdown rendering ────────────────────────────────────────────────────────

def render_markdown(text: str) -> None:
    if text.strip():
        console.print(Markdown(text))


def has_markdown(text: str) -> bool:
    return any(c in text for c in ("#", "*", "`", "_", "[", "\n-"))

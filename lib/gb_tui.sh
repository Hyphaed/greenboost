# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
#
# lib/gb_tui.sh - TUI helpers shared by every GreenBoost interactive command.
#
# All UI commands (vitals, cluster, logs, inference-logs, nvtx-logs, build-info,
# etc.) follow the project's "UI Command Paradigm":
#   1. _cmd_<name>_snapshot()  - render a single static frame
#   2. Interactive TUI loop    - alt-screen + 5-s refresh + Ctrl+C exit
#   3. Non-interactive mode    - snapshot once and exit (--llm or piped)
#   4. --llm flag              - machine-readable, ANSI-stripped
#
# This file factors out the canonical interactive-TUI-loop helper so commands
# don't each carry a ~25-line copy of the alt-screen + stty + trap dance.
#
# Sourced by greenboost_setup.sh near the top of the script.  See PR-BB / PR-GG.

# _gb_run_tui_loop <snapshot_fn> [refresh_sec]
#
# Run an interactive TUI: enter the alt-screen, hide the cursor, render the
# snapshot every `refresh_sec` seconds (default 5), and listen for Ctrl+C
# (exit) or Ctrl+S (immediate refresh).  Restores stty + cursor on every exit
# path (signal or normal completion).
_gb_run_tui_loop() {
    local _snapshot_fn="$1"
    local _refresh="${2:-5}"
    local _saved_stty; _saved_stty=$(stty -g 2>/dev/null || true)
    stty -ixon 2>/dev/null || true
    printf '\033[?1049h'   # enter alternate-screen buffer
    printf '\033[?25l'     # hide cursor
    trap 'printf "\033[?25h\033[?1049l"; [[ -n "$_saved_stty" ]] && stty "$_saved_stty" 2>/dev/null || true; exit 0' INT TERM EXIT

    local _key=""
    while true; do
        printf '\033[H'             # home cursor
        "$_snapshot_fn"             # call the per-command snapshot renderer
        printf '\033[J'             # clear to end of screen
        if read -t "$_refresh" -s -n 1 _key 2>/dev/null; then
            case "$_key" in
                $'\x03') break ;;   # Ctrl+C
                $'\x13') ;;         # Ctrl+S (refresh): just loop
                *) ;;
            esac
        fi
    done

    printf '\033[?25h\033[?1049l'
    [[ -n "$_saved_stty" ]] && stty "$_saved_stty" 2>/dev/null || true
    trap - INT TERM EXIT
}

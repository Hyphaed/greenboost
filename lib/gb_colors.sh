# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
#
# lib/gb_colors.sh - ANSI color palette used by every GreenBoost UI command.
#
# Matches synapse_cli color scheme (#6C71C4 violet, #E6FF3C lime, etc.).
# All variables degrade to 16-color ANSI when COLORTERM is unset/8bit.
#
# Sourced by greenboost_setup.sh near the top of the script.  See PR-GG.

_gb_truecolor() { [[ "${COLORTERM:-}" =~ ^(truecolor|24bit)$ ]]; }

if _gb_truecolor; then
    C_VIOLET=$'\033[38;2;108;113;196m'    # #6C71C4 - brand accent
    C_LIME=$'\033[38;2;230;255;60m'       # #E6FF3C - success / active
    C_GRAY=$'\033[38;2;208;207;204m'      # #D0CFCC - body text
    C_CYAN=$'\033[38;2;48;200;255m'       # #30C8FF - section headers
    C_AMBER=$'\033[38;2;255;191;0m'       # #FFBF00 - prompt ❯ / warnings
    C_PURPLE=$'\033[38;2;167;139;250m'    # #a78bfa - wizard headers
    C_RED=$'\033[38;2;255;92;50m'         # #FF5C32 - error / critical
    C_WHITE=$'\033[38;2;255;255;255m'     # #FFFFFF - code block text
else
    C_VIOLET=$'\033[0;34m'
    C_LIME=$'\033[0;32m'
    C_GRAY=$'\033[0;37m'
    C_CYAN=$'\033[0;36m'
    C_AMBER=$'\033[1;33m'
    C_PURPLE=$'\033[0;35m'
    C_RED=$'\033[0;31m'
    C_WHITE=$'\033[1;37m'
fi
C_BOLD=$'\033[1m'
C_DIM=$'\033[2m'
C_RESET=$'\033[0m'

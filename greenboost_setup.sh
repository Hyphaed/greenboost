#!/usr/bin/env bash
# GreenBoost v2.8.2 — Setup & installation script (Ubuntu / Debian and derivatives)
# Supports: Ubuntu, Debian, Pop!_OS, Mint, and other apt-based distros.
# Delegates to greenboost_setup_rocky.sh on RHEL/Fedora and
# greenboost_setup_arch.sh on Arch-based systems.
#
# Hardware is detected at runtime: CPU topology, GPU VRAM, RAM, kernel version.
# No hardware-specific values are hard-coded.
#
# USAGE:
#   (no args)                                    — interactive wizard
#   sudo ./greenboost_setup.sh setup             — full install (prompts for mode)
#   sudo ./greenboost_setup.sh module-only       — kernel module only (safe on any machine)
#   sudo ./greenboost_setup.sh install           — build + install module + shim
#   sudo ./greenboost_setup.sh uninstall         — remove module + all config
#   sudo ./greenboost_setup.sh load              — insmod with default params
#   sudo ./greenboost_setup.sh unload            — rmmod
#   sudo ./greenboost_setup.sh tune              — runtime tuning (governor, NVMe, sysctl)
#   sudo ./greenboost_setup.sh tune-all          — run all tune-* commands
#        ./greenboost_setup.sh status            — show pool info + system state
#        ./greenboost_setup.sh benchmark         — cuda memory pool bandwidth benchmark (T1/T2/T3)
#        ./greenboost_setup.sh help              — show common commands
#        ./greenboost_setup.sh help --all        — show all commands + env vars
#
# GLOBAL FLAGS (before or after the command):
#   --skip-update-check  Skip GitLab version check (offline workstations)
#
# ENVIRONMENT (for load command — all values auto-detected at runtime):
#   GPU_PHYS_GB    physical VRAM in GB       (detected via nvidia-smi)
#   VIRT_VRAM_GB   system RAM pool size in GB (80% of total RAM)
#   RESERVE_GB     minimum free system RAM to always maintain
#   NVME_SWAP_GB   total NVMe swap capacity  (auto-detected)
#   NVME_POOL_GB   GreenBoost soft cap on T3 allocations

DRIVER_NAME="greenboost"
SHIM_LIB="libgreenboost_cuda.so"
AUDIT_LIB="libgreenboost_audit.so"
AUDIT_LIB32="libgreenboost_audit32.so"
VULKAN_LAYER_LIB="libVkLayer_greenboost.so"
VULKAN_LAYER_MANIFEST="VkLayer_greenboost.json"
VULKAN_IMPLICIT_LAYER_DIR="/etc/vulkan/implicit_layer.d"
SHIM_DEST="/usr/local/lib"
MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GB_PROFILES_DIR="/etc/greenboost/profiles"
GB_ACTIVE_PROFILE_LINK="/etc/greenboost/active_profile.md"

# Dynamic swapfile — GreenBoost provisions this automatically when no adequate swap exists
GB_SWAP_FILE="/var/lib/greenboost/swapfile"
GB_SWAP_MIN_GB=8      # minimum existing swap to consider adequate (skip provisioning)
GB_SWAP_MAX_GB=120    # cap for auto-provisioned swapfile

GB_VERSION="2.8.2"
GB_REPO_API="https://gitlab.com/api/v4/projects/IsolatedOctopi%2Fgreenboost/repository/tags"

# Colours
RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
BLU='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${GRN}[GreenBoost]${NC} $*"; }
warn()  { echo -e "${YLW}[GreenBoost] WARN:${NC} $*"; }
die()   { echo -e "${RED}[GreenBoost] ERROR:${NC} $*" >&2; exit 1; }

# ---- Brand palette + UI primitives -------------------------------------
# Matches synapse_cli color scheme (#6C71C4 violet, #E6FF3C lime, etc.)
# All functions degrade to 16-color ANSI when COLORTERM is unset/8bit.

_gb_truecolor() { [[ "${COLORTERM:-}" =~ ^(truecolor|24bit)$ ]]; }

if _gb_truecolor; then
    C_VIOLET=$'\033[38;2;108;113;196m'    # #6C71C4 — brand accent
    C_LIME=$'\033[38;2;230;255;60m'       # #E6FF3C — success / active
    C_GRAY=$'\033[38;2;208;207;204m'      # #D0CFCC — body text
    C_CYAN=$'\033[38;2;48;200;255m'       # #30C8FF — section headers
    C_AMBER=$'\033[38;2;255;191;0m'       # #FFBF00 — prompt ❯ / warnings
    C_PURPLE=$'\033[38;2;167;139;250m'    # #a78bfa — wizard headers
    C_RED=$'\033[38;2;255;92;50m'         # #FF5C32 — error / critical
    C_WHITE=$'\033[38;2;255;255;255m'     # #FFFFFF — code block text
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

GB_SPIN_FRAMES=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")

# Default log paths (override via env)
GB_STATUS_LOG="${GB_STATUS_LOG:-$HOME/.local/share/greenboost/status.log}"
GB_INFER_TEST_LOG="${GB_INFER_TEST_LOG:-$HOME/.local/share/greenboost/inference-test-latest.log}"

# gb_header — branded box header
gb_header() {
    local cols; cols=$(tput cols 2>/dev/null || echo 64)
    local title=" GreenBoost v${GB_VERSION} — CUDA Memory Orchestrator for NVidia GPUs"
    echo -e ""
    echo -e "${C_VIOLET}${C_BOLD}  ╔$(printf '═%.0s' $(seq 1 $((cols - 4))))╗${C_RESET}"
    echo -e "${C_VIOLET}${C_BOLD}  ║${C_RESET} ${C_GRAY}${C_BOLD}${title}$(printf ' %.0s' $(seq 1 $((cols - 4 - ${#title} - 1))))${C_VIOLET}${C_BOLD}║${C_RESET}"
    echo -e "${C_VIOLET}${C_BOLD}  ╚$(printf '═%.0s' $(seq 1 $((cols - 4))))╝${C_RESET}"
    echo -e ""
}

# gb_separator — full-width thin line
gb_separator() {
    local cols; cols=$(tput cols 2>/dev/null || echo 64)
    echo -e "${C_DIM}$(printf '─%.0s' $(seq 1 $((cols - 2))))${C_RESET}"
}

# gb_step N M "description"
gb_step() {
    echo -e ""
    echo -e "${C_CYAN}${C_BOLD}  [$1/$2]${C_RESET} ${C_GRAY}${C_BOLD}$3${C_RESET}"
}

# gb_ok / gb_fail / gb_warn — status messages
gb_ok()   { echo -e "  ${C_LIME}✓${C_RESET}  ${C_GRAY}$*${C_RESET}"; }
gb_fail() { echo -e "  ${C_RED}✗${C_RESET}  $*"; }
gb_warn_ui() { echo -e "  ${C_AMBER}⚠${C_RESET}  $*"; }

# gb_spin PID "message" — braille spinner until PID exits
gb_spin() {
    local pid=$1 msg="$2" i=0 start_time
    start_time=$(date +%s)
    while kill -0 "$pid" 2>/dev/null; do
        local elapsed=$(( $(date +%s) - start_time ))
        local time_str
        if (( elapsed >= 60 )); then
            time_str=$(printf "%d:%02d" $((elapsed / 60)) $((elapsed % 60)))
        else
            time_str="${elapsed}s"
        fi
        printf "\r  ${C_LIME}%s${C_RESET}  ${C_DIM}%s${C_RESET} ${C_GRAY}[%s]${C_RESET}   " \
            "${GB_SPIN_FRAMES[$((i % ${#GB_SPIN_FRAMES[@]}))]}" "$msg" "$time_str"
        sleep 0.08
        (( i++ )) || true
    done
    printf "\r  ${C_LIME}✓${C_RESET}  ${C_GRAY}%s${C_RESET}                             \n" "$msg"
}

# gb_info — replaces info() with brand style (keeps [GreenBoost] prefix for log compat)
gb_info() { echo -e "  ${C_VIOLET}◈${C_RESET}  ${C_GRAY}$*${C_RESET}"; }

# ---- Panel primitives (used by cmd_status) ----------------------------
# gb_panel_top "Title" — violet box top border with embedded cyan title
gb_panel_top() {
    local title="$1"
    local cols; cols=$(tput cols 2>/dev/null || echo 80)
    local inner=$(( cols - 6 ))          # space inside the box borders (2 indent + 2 borders + 2 spaces)
    local tlen=${#title}
    local dashes=$(( inner - tlen - 3 )) # ══ + space + title + space + dashes
    (( dashes < 1 )) && dashes=1
    printf '%b' "  ${C_VIOLET}${C_BOLD}╔══ ${C_CYAN}${title}${C_VIOLET} "
    printf '═%.0s' $(seq 1 $dashes)
    printf '%b\n' "╗${C_RESET}"
}

# gb_panel_bottom — violet box bottom border
gb_panel_bottom() {
    local cols; cols=$(tput cols 2>/dev/null || echo 80)
    local inner=$(( cols - 4 ))
    printf '%b' "  ${C_VIOLET}${C_BOLD}╚"
    printf '═%.0s' $(seq 1 $inner)
    printf '%b\n' "╝${C_RESET}"
}

# gb_panel_row "content" — bordered content row, ANSI-aware padding
gb_panel_row() {
    local cols; cols=$(tput cols 2>/dev/null || echo 80)
    local inner=$(( cols - 6 ))   # usable chars between ║ + space and space + ║
    # strip ANSI codes to compute visible length
    local visible
    visible=$(printf '%b' "$1" | sed 's/\x1b\[[0-9;]*m//g')
    local vlen=${#visible}
    local pad=$(( inner - vlen ))
    (( pad < 0 )) && pad=0
    printf '%b' "  ${C_VIOLET}${C_BOLD}║${C_RESET} "
    printf '%b' "$1"
    printf '%*s' "$pad" ''
    printf '%b\n' " ${C_VIOLET}${C_BOLD}║${C_RESET}"
}

# gb_panel_empty — empty bordered spacer row
gb_panel_empty() { gb_panel_row ""; }

# gb_bar pct fill_color empty_color width — inline progress bar (no newline)
gb_bar() {
    local pct=$1 fill_c="$2" empty_c="$3" width=${4:-30}
    local filled=$(( pct * width / 100 ))
    (( filled > width )) && filled=$width
    local empty=$(( width - filled ))
    printf '%b' "${fill_c}"
    (( filled > 0 )) && printf '█%.0s' $(seq 1 $filled)
    printf '%b' "${empty_c}"
    (( empty > 0 )) && printf '░%.0s' $(seq 1 $empty)
    printf '%b' "${C_RESET}"
}

# gb_tier_color pct — echo ANSI color for a utilization percentage
gb_tier_color() {
    local pct=$1
    if   (( pct >= 90 )); then printf '%b' "${C_RED}"
    elif (( pct >= 75 )); then printf '%b' "${C_AMBER}"
    else                       printf '%b' "${C_LIME}"
    fi
}

# gb_section "Title" — purple bold section header + separator (mirrors unlocker ui_section)
gb_section() {
    echo -e ""
    echo -e "  ${C_PURPLE}${C_BOLD}$1${C_RESET}"
    gb_separator
}

# gb_col_width — usable character width of each column in a 2-col layout
gb_col_width() {
    local cols; cols=$(tput cols 2>/dev/null || echo 80)
    local w=$(( (cols - 7) / 2 ))
    (( w < 36 )) && w=36
    echo "$w"
}

# gb_strip_ansi — remove ANSI color/SGR codes from stdin (for visible-width counting)
gb_strip_ansi() {
    sed 's/\x1b\[[0-9;]*[mK]//g'
}

# _trunc "string" max_len — truncates string to max_len chars, adding ".." if cut
_trunc() { local s="$1" n="$2"; (( ${#s} > n )) && s="${s:0:$((n-2))}.."; echo "$s"; }

# gb_render_2col left_arr right_arr
# Prints paired lines side-by-side with a dim │ divider.
# Args are bash array variable names (namereferences, requires bash ≥ 4.3).
gb_render_2col() {
    local -n _gc2_L="$1"
    local -n _gc2_R="$2"
    local cw; cw=$(gb_col_width)
    local max=$(( ${#_gc2_L[@]} > ${#_gc2_R[@]} ? ${#_gc2_L[@]} : ${#_gc2_R[@]} ))
    local i
    for (( i = 0; i < max; i++ )); do
        local L="${_gc2_L[$i]:-}"
        local R="${_gc2_R[$i]:-}"
        local vis; vis=$(printf '%s' "$L" | gb_strip_ansi)
        local vlen=${#vis}
        local pad=$(( cw - vlen ))
        (( pad < 0 )) && pad=0
        printf '%s%*s  %b│%b  %s\n' "$L" "$pad" "" "${C_DIM}" "${C_RESET}" "$R"
    done
}

# gb_menu_item num "Label" "Description" [root] [full]
# Fixed 24-char label column so all descriptions align; optional ⚠ root / ◈ full install badges.
gb_menu_item() {
    local num="$1" label="$2" desc="$3"
    shift 3
    local root_tag="" full_tag=""
    for flag in "$@"; do
        [[ "$flag" == "root" ]] && root_tag="  ${C_DIM}${C_RED}⚠ root${C_RESET}"
        [[ "$flag" == "full" ]] && full_tag="  ${C_DIM}${C_CYAN}◈ full install${C_RESET}"
    done
    printf "  ${C_LIME}${C_BOLD}[%s]${C_RESET}  ${C_BOLD}${C_GRAY}%-24s${C_RESET}${C_DIM}${C_GRAY}%s${C_RESET}" \
        "$num" "$label" "$desc"
    printf "%b\n" "${root_tag}${full_tag}"
}

# gb_prompt "label" — amber ❯ prompt; result in $REPLY
gb_prompt() {
    printf "\n  ${C_AMBER}${C_BOLD}❯${C_RESET}  ${C_GRAY}${1:-Choice}${C_RESET}: "
    read -r REPLY
}

# gb_confirm "Question" — amber ❯ Y/n; returns 0 if yes
gb_confirm() {
    printf "  ${C_AMBER}${C_BOLD}❯${C_RESET}  ${C_GRAY}$1${C_RESET} ${C_DIM}[Y/n]${C_RESET}: "
    read -r _confirm_reply
    [[ "${_confirm_reply:-Y}" =~ ^[Yy]$ ]]
}

# gb_press_enter — pause prompt at the end of a wizard action
gb_press_enter() {
    echo -e ""
    printf "  ${C_DIM}${C_GRAY}Press Enter to return to menu…${C_RESET}"
    read -r
}

# ── Install mode helpers ──────────────────────────────────────────────────────
GB_INSTALL_MODE="module"   # default: safe (no system tuning)

# gb_select_install_mode — interactive or flag-driven mode selection.
# Sets GB_INSTALL_MODE to "module" or "full".
# Pass --module-only or --full-install to skip the prompt (for scripts/CI).
gb_select_install_mode() {
    for _a in "$@"; do
        [[ "$_a" == "--module-only"  ]] && { GB_INSTALL_MODE="module"; return; }
        [[ "$_a" == "--full-install" ]] && { GB_INSTALL_MODE="full";   return; }
    done
    gb_separator
    echo ""
    printf '%b\n' "  ${C_VIOLET}${C_BOLD}GreenBoost — Choose Installation Mode${C_RESET}"
    echo ""
    printf '%b\n' "  ${C_LIME}${C_BOLD}[1]${C_RESET}  ${C_BOLD}${C_GRAY}Kernel module only${C_RESET}  ${C_DIM}DKMS install — no sysctl, GRUB, or service changes${C_RESET}"
    printf '%b\n' "  ${C_LIME}${C_BOLD}[2]${C_RESET}  ${C_BOLD}${C_GRAY}Full system setup ${C_RESET}  ${C_DIM}Module + AI libs + sysctl + GRUB + systemd services${C_RESET}"
    echo ""
    printf '%b\n' "  ${C_DIM}Mode [1] is safe on any machine. Mode [2] tunes sysctl, GRUB, and${C_RESET}"
    printf '%b\n' "  ${C_DIM}enables system services — recommended only on a dedicated workstation.${C_RESET}"
    echo ""
    printf '%b' "  ${C_AMBER}${C_BOLD}❯${C_RESET}  ${C_GRAY}Mode [1/2] (default: 1): ${C_RESET}"
    read -r _mode_reply
    case "${_mode_reply:-1}" in
        2) GB_INSTALL_MODE="full" ;;
        *) GB_INSTALL_MODE="module" ;;
    esac
}

# gb_consent_gate "description" — per-phase confirmation before a system change.
# Returns 0 if the user confirms, 1 if they decline.
gb_consent_gate() {
    echo ""
    printf '%b\n' "  ${C_AMBER}${C_BOLD}⚠  System change:${C_RESET}  ${C_GRAY}$1${C_RESET}"
    gb_confirm "Apply this change?" && return 0 || return 1
}

# ---- Pool data parsers (used by cmd_status) ---------------------------
# Strip non-digit chars from a named variable and default to 0 if empty.
# Usage: _sanitize_int VARNAME  (no subshell, uses printf -v nameref)
_sanitize_int() { local _v="${!1}"; _v="${_v%%[^0-9]*}"; printf -v "$1" '%s' "${_v:-0}"; }

# Sets PB_* vars from pool_brief sysfs one-liner
# Format: T1:12GB T2:0/51GB(0%) T3:0/64GB PRESSURE:ok KV_RSV:2048MB KV_T2:0MB
parse_pool_brief() {
    PB_T1_GB=0; PB_T2_USED_GB=0; PB_T2_MAX_GB=0; PB_T2_PCT=0
    PB_T3_USED_GB=0; PB_T3_MAX_GB=0; PB_PRESSURE="—"
    PB_KV_RSV_MB=0; PB_KV_T2_MB=0
    local brief_f="/sys/class/greenboost/greenboost/pool_brief"
    [[ -r "$brief_f" ]] || return 1
    local brief; brief=$(cat "$brief_f" 2>/dev/null) || return 1
    PB_T1_GB=$(echo "$brief"     | grep -oP 'T1:\K[0-9]+'         | head -1 || echo 0)
    PB_T2_USED_GB=$(echo "$brief"| grep -oP 'T2:\K[0-9]+'         | head -1 || echo 0)
    PB_T2_MAX_GB=$(echo "$brief" | grep -oP 'T2:[0-9]+/\K[0-9]+'  | head -1 || echo 0)
    PB_T2_PCT=$(echo "$brief"    | grep -oP 'T2:[^(]+\(\K[0-9]+'  | head -1 || echo 0)
    PB_T3_USED_GB=$(echo "$brief"| grep -oP 'T3:\K[0-9]+'         | head -1 || echo 0)
    PB_T3_MAX_GB=$(echo "$brief" | grep -oP 'T3:[0-9]+/\K[0-9]+'  | head -1 || echo 0)
    PB_PRESSURE=$(echo "$brief"  | grep -oP 'PRESSURE:\K\S+' || echo "—")
    PB_KV_RSV_MB=$(echo "$brief" | grep -oP 'KV_RSV:\K[0-9]+'     | head -1 || echo 0)
    PB_KV_T2_MB=$(echo "$brief"  | grep -oP 'KV_T2:\K[0-9]+'      | head -1 || echo 0)
    _sanitize_int PB_T1_GB; _sanitize_int PB_T2_USED_GB; _sanitize_int PB_T2_MAX_GB
    _sanitize_int PB_T2_PCT; _sanitize_int PB_T3_USED_GB; _sanitize_int PB_T3_MAX_GB
    _sanitize_int PB_KV_RSV_MB; _sanitize_int PB_KV_T2_MB
}

# Sets PI_* vars from status sysfs (RAM/T2 detail, KV placement, swap)
parse_pool_info() {
    PI_RAM_TOTAL_MB=0; PI_RAM_FREE_MB=0; PI_SAFETY_RSV_MB=0
    PI_T2_ALLOC_MB=0; PI_T2_AVAIL_MB=0; PI_ACTIVE_BUFS=0
    PI_OOM_GUARD="no"; PI_PAGE_MODE="—"
    PI_KV_T1_RSV_MB=0; PI_KV_T2_MB=0; PI_KV_T3_MB=0; PI_KV_TOTAL_MB=0
    PI_KV_PLACEMENT="—"; PI_TURBO_STATUS="disabled"
    PI_SWAP_TOTAL_MB=0; PI_SWAP_USED_MB=0; PI_SWAP_FREE_MB=0; PI_T3_ALLOC_MB=0
    local pool_f="/sys/class/greenboost/greenboost/status"
    [[ -r "$pool_f" ]] || return 1
    local info; info=$(cat "$pool_f" 2>/dev/null) || return 1
    PI_RAM_TOTAL_MB=$(echo "$info"  | grep -oP 'Total RAM\s*:\s*\K[0-9]+'              | head -1 || echo 0)
    PI_RAM_FREE_MB=$(echo "$info"   | grep -oP 'Avail RAM[^:]*:\s*\K[0-9]+'            | head -1 || echo 0)
    PI_SAFETY_RSV_MB=$(echo "$info" | grep -oP 'Safety reserve\s*:\s*\K[0-9]+'         | head -1 || echo 0)
    PI_T2_ALLOC_MB=$(echo "$info"   | grep -oP 'T2 allocated\s*:\s*\K[0-9]+'           | head -1 || echo 0)
    PI_T2_AVAIL_MB=$(echo "$info"   | grep -oP 'T2 available\s*:\s*\K[0-9]+'           | head -1 || echo 0)
    PI_ACTIVE_BUFS=$(echo "$info"   | grep -oP 'Active DMA-BUF objects\s*:\s*\K[0-9]+' | head -1 || echo 0)
    PI_OOM_GUARD=$(echo "$info"     | grep -oP 'OOM guard\s*:\s*\K\S+'                || echo "no")
    PI_PAGE_MODE=$(echo "$info"     | grep -oP 'Page mode\s*:\s*\K[^\n]+'  | head -1 | sed 's/[[:space:]]*$//')
    PI_PAGE_MODE="${PI_PAGE_MODE:-—}"
    PI_KV_T1_RSV_MB=$(echo "$info"  | grep -oP 'KV in T1.*reserve:\s*\K[0-9]+'        | head -1 || echo 0)
    PI_KV_T2_MB=$(echo "$info"      | grep -oP 'KV in T2[^:]*:\s*\K[0-9]+'            | head -1 || echo 0)
    PI_KV_T3_MB=$(echo "$info"      | grep -oP 'KV in T3[^:]*:\s*\K[0-9]+'            | head -1 || echo 0)
    PI_KV_TOTAL_MB=$(echo "$info"   | grep -oP 'KV tagged total[^:]*:\s*\K[0-9]+'      | head -1 || echo 0)
    PI_KV_PLACEMENT=$(echo "$info"  | grep -oP 'KV cache placement\s*:\s*\K[^\n]+' | head -1 | sed 's/[[:space:]]*$//')
    PI_KV_PLACEMENT="${PI_KV_PLACEMENT:-—}"
    PI_TURBO_STATUS=$(echo "$info"  | grep -oP 'TurboQuant Compression\s*:\s*\K[^\n]+' | head -1 | sed 's/[[:space:]]*$//')
    PI_TURBO_STATUS="${PI_TURBO_STATUS:-disabled}"
    PI_SWAP_TOTAL_MB=$(echo "$info" | grep -oP 'Swap total\s*:\s*\K[0-9]+'             | head -1 || echo 0)
    PI_SWAP_USED_MB=$(echo "$info"  | grep -oP 'Swap used\s*:\s*\K[0-9]+'              | head -1 || echo 0)
    PI_SWAP_FREE_MB=$(echo "$info"  | grep -oP 'Swap free\s*:\s*\K[0-9]+'              | head -1 || echo 0)
    PI_T3_ALLOC_MB=$(echo "$info"   | grep -oP 'T3 GreenBoost alloc\s*:\s*\K[0-9]+'   | head -1 || echo 0)
    _sanitize_int PI_RAM_TOTAL_MB; _sanitize_int PI_RAM_FREE_MB; _sanitize_int PI_SAFETY_RSV_MB
    _sanitize_int PI_T2_ALLOC_MB;  _sanitize_int PI_T2_AVAIL_MB; _sanitize_int PI_ACTIVE_BUFS
    _sanitize_int PI_KV_T1_RSV_MB; _sanitize_int PI_KV_T2_MB; _sanitize_int PI_KV_T3_MB
    _sanitize_int PI_KV_TOTAL_MB
    _sanitize_int PI_SWAP_TOTAL_MB; _sanitize_int PI_SWAP_USED_MB; _sanitize_int PI_SWAP_FREE_MB
    _sanitize_int PI_T3_ALLOC_MB
}

# ---- Shim stats reader (used by cmd_status) ---------------------------
# Sets SS_* vars from shim_stats (written by the CUDA shim).
# Primary location: /run/greenboost/shim_stats
# Fallback location: /tmp/greenboost_shim_stats (used when Ollama can't write to /run/greenboost/)
# SS_ACTIVE_PATH: A0 | A | B | C | none | unknown
# SS_STALE: 1 if the file is older than 30 s (shim not running or crashed)
parse_shim_stats() {
    SS_ACTIVE_PATH="unknown"; SS_STALE=1
    SS_PATH_A0=0; SS_PATH_A=0; SS_PATH_B=0; SS_PATH_C=0
    SS_PHASE=""; SS_KV_RSV_NOM=0; SS_KV_RSV_EFF=0; SS_KV_T1_MB=0; SS_HEADROOM_MB=0
    local stats_f="/run/greenboost/shim_stats"
    [[ -r "$stats_f" ]] || stats_f="/tmp/greenboost_shim_stats"
    [[ -r "$stats_f" ]] || return 1
    local content; content=$(cat "$stats_f" 2>/dev/null) || return 1
    local ts; ts=$(echo "$content" | grep -oP 'timestamp=\K[0-9]+' | head -1 || echo 0)
    local now; now=$(date +%s)
    (( now - ts <= 30 )) && SS_STALE=0
    SS_ACTIVE_PATH=$(echo "$content"  | grep -oP 'active_path=\K\S+'              | head -1 || echo "unknown")
    SS_PATH_A0=$(echo "$content"      | grep -oP 'path_a0_count=\K[0-9]+'         | head -1 || echo 0)
    SS_PATH_A=$(echo "$content"       | grep -oP 'path_a_count=\K[0-9]+'          | head -1 || echo 0)
    SS_PATH_B=$(echo "$content"       | grep -oP 'path_b_count=\K[0-9]+'          | head -1 || echo 0)
    SS_PATH_C=$(echo "$content"       | grep -oP 'path_c_count=\K[0-9]+'          | head -1 || echo 0)
    SS_PHASE=$(echo "$content"        | grep -oP 'phase=\K\S+'                    | head -1 || echo "")
    SS_KV_RSV_NOM=$(echo "$content"   | grep -oP 'kv_reserve_nominal_mb=\K[0-9]+' | head -1 || echo 0)
    SS_KV_RSV_EFF=$(echo "$content"   | grep -oP 'kv_reserve_effective_mb=\K[0-9]+' | head -1 || echo 0)
    SS_KV_T1_MB=$(echo "$content"     | grep -oP 'kv_t1_tracked_mb=\K[0-9]+'     | head -1 || echo 0)
    SS_HEADROOM_MB=$(echo "$content"  | grep -oP 'vram_headroom_mb=\K[0-9]+'      | head -1 || echo 0)
}

# ---- Live GPU VRAM query (used by cmd_status) -------------------------
# Sets GPU_VRAM_USED_MB, GPU_VRAM_TOTAL_MB, GPU_VRAM_PCT from nvidia-smi.
# Silently falls back to zeros if nvidia-smi is unavailable or times out.
query_gpu_vram() {
    GPU_VRAM_USED_MB=0; GPU_VRAM_TOTAL_MB=0; GPU_VRAM_PCT=0
    command -v nvidia-smi &>/dev/null || return 0
    local raw
    raw=$(timeout 2 nvidia-smi --query-gpu=memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null | head -1) || return 0
    [[ -z "$raw" ]] && return 0
    GPU_VRAM_USED_MB=$(echo "$raw" | cut -d, -f1 | tr -dc '0-9')
    GPU_VRAM_TOTAL_MB=$(echo "$raw" | cut -d, -f2 | tr -dc '0-9')
    GPU_VRAM_USED_MB="${GPU_VRAM_USED_MB:-0}"
    GPU_VRAM_TOTAL_MB="${GPU_VRAM_TOTAL_MB:-0}"
    if (( GPU_VRAM_TOTAL_MB > 0 )); then
        GPU_VRAM_PCT=$(( GPU_VRAM_USED_MB * 100 / GPU_VRAM_TOTAL_MB ))
    fi
}

# ---- Live system swap query (used by cmd_status) ----------------------
# Sets LIVE_SWAP_TOTAL_MB, LIVE_SWAP_USED_MB, LIVE_SWAP_PCT from /proc/swaps.
# Always reflects actual kernel swap accounting, not kernel module parameters.
query_live_swap() {
    LIVE_SWAP_TOTAL_KB=0; LIVE_SWAP_USED_KB=0
    LIVE_SWAP_TOTAL_MB=0; LIVE_SWAP_USED_MB=0; LIVE_SWAP_PCT=0
    [[ -r /proc/swaps ]] || return 0
    local _sf _stype _sw_kb _used_kb _prio
    while read -r _sf _stype _sw_kb _used_kb _prio; do
        [[ "$_sf" == "Filename" ]] && continue
        (( LIVE_SWAP_TOTAL_KB += _sw_kb )) || true
        (( LIVE_SWAP_USED_KB  += _used_kb )) || true
    done < /proc/swaps
    LIVE_SWAP_TOTAL_MB=$(( LIVE_SWAP_TOTAL_KB / 1024 ))
    LIVE_SWAP_USED_MB=$(( LIVE_SWAP_USED_KB  / 1024 ))
    if (( LIVE_SWAP_TOTAL_MB > 0 )); then
        LIVE_SWAP_PCT=$(( LIVE_SWAP_USED_MB * 100 / LIVE_SWAP_TOTAL_MB ))
    fi
}

# ---- Ollama loaded-model query (used by cmd_status) -------------------
# Sets OL_MODEL, OL_PARAM_COUNT, OL_CTX_SIZE, OL_VRAM_MB.
# Falls back to empty strings if Ollama is not running or no model is loaded.
query_ollama_ps() {
    OL_MODEL=""; OL_PARAM_COUNT=""; OL_CTX_SIZE=""; OL_VRAM_MB=""
    command -v curl &>/dev/null || return 0
    local raw
    raw=$(curl -sf --max-time 2 http://localhost:11434/api/ps 2>/dev/null) || return 0
    [[ -z "$raw" ]] && return 0
    command -v python3 &>/dev/null || return 0
    local parsed
    # Pass JSON via stdin to avoid shell injection when $raw contains quotes.
    parsed=$(printf '%s' "$raw" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    models = d.get('models', [])
    if not models:
        sys.exit(0)
    m = models[0]
    name = m.get('name','')
    # param count from model name (e.g. 'qwen3:32b' -> '32B')
    tag = name.split(':')[-1].upper() if ':' in name else ''
    param = ''
    for p in ['120B','72B','70B','32B','14B','13B','8B','7B','3B','1.5B','1B']:
        if p in tag or p in name.upper():
            param = p; break
    det = m.get('details', {})
    if not param:
        pf = det.get('parameter_size','')
        param = pf if pf else ''
    vram_b = m.get('size_vram', 0)
    vram_mb = vram_b // (1024*1024) if vram_b else 0
    # context_length: try top-level first (Ollama >=0.6), then model_info, then details
    num_ctx = int(m.get('context_length', 0) or 0)
    if not num_ctx:
        mi = m.get('model_info', {})
        for k,v in mi.items():
            if 'context_length' in k or 'num_ctx' in k:
                try: num_ctx = int(v)
                except: pass
    if not num_ctx:
        num_ctx = int(det.get('context_length', 0) or 0)
    print(f'{name}|{param}|{num_ctx}|{vram_mb}')
except Exception as e:
    pass
" 2>/dev/null) || return 0
    [[ -z "$parsed" ]] && return 0
    OL_MODEL=$(echo "$parsed"      | cut -d'|' -f1)
    OL_PARAM_COUNT=$(echo "$parsed"| cut -d'|' -f2)
    OL_CTX_SIZE=$(echo "$parsed"   | cut -d'|' -f3)
    OL_VRAM_MB=$(echo "$parsed"    | cut -d'|' -f4)
}

# ---- KV size estimator (used by cmd_status) ---------------------------
# estimate_kv_mb ctx_size param_count_str
# Returns estimated KV cache size in MB, accounting for q8_0 quantization.
estimate_kv_mb() {
    local ctx=${1:-0} param_str="${2:-}"
    local kv_per_tok_bytes=512   # default: ~7B class
    # Extract leading integer from param string (handles "31.6B", "70B", "120B", etc.)
    local param_int
    param_int=$(echo "$param_str" | grep -oP '^\d+' | head -1 || echo 0)
    if   (( param_int >= 100 )); then kv_per_tok_bytes=3584   # ~120B
    elif (( param_int >= 60  )); then kv_per_tok_bytes=2560   # ~70B
    elif (( param_int >= 25  )); then kv_per_tok_bytes=1331   # ~32B
    elif (( param_int >= 10  )); then kv_per_tok_bytes=819    # ~13B
    elif (( param_int >= 5   )); then kv_per_tok_bytes=512    # ~7B
    elif (( param_int >= 2   )); then kv_per_tok_bytes=256    # ~3B
    fi
    # q8_0 KV cache halves the size vs FP16 (OLLAMA_KV_CACHE_TYPE=q8_0)
    kv_per_tok_bytes=$(( kv_per_tok_bytes / 2 ))
    local mb=$(( ctx * kv_per_tok_bytes / 1024 / 1024 ))
    # Minimum 1 MB to avoid hiding real (but small) estimates
    (( mb < 1 && ctx > 0 )) && mb=1
    echo "$mb"
}

# ---- Memory flow event log (used by cmd_status) -----------------------
# Patterns that indicate a tier transition or notable orchestration event
_GB_FLOW_PAT='KV spill|Evicted.*cold|T2 auto-evict|T2 WARN:|T2 CRITICAL:|T3 CRITICAL:|KV cache T2 full|T3 safety-net|KV reserve set|New process PID|Process PID.*exited|T2 DDR pool|T3 NVMe swap CRITICAL|T2 OOM guard|overflow|Phase →|KV reserve auto|VRAM:.*OVERFLOW|kv_allocated_t1|T3 cap'

# gather_flow_events N — emit last N tier-transition log lines from kernel
gather_flow_events() {
    local n=${1:-12}
    dmesg 2>/dev/null \
        | grep -E "greenboost.*($( echo "$_GB_FLOW_PAT" | tr '|' '|'))" | tail -"$n"
}

# format_flow_event "raw log line" — emit a branded, colored status line
format_flow_event() {
    local line="$1"
    # Extract just the message (after journalctl prefix or kernel timestamp)
    local msg
    msg=$(printf '%s' "$line" | sed 's/.*greenboost[d]*[^:]*: //' | sed 's/.*\] //')
    # Extract short timestamp HH:MM:SS — handles journalctl ISO and dmesg [NNN.NN] formats
    local ts
    ts=$(printf '%s' "$line" | grep -oP 'T\d\d:\d\d:\d\d' | head -1)
    if [[ -z "$ts" ]]; then
        ts=$(printf '%s' "$line" | grep -oP '\d\d:\d\d:\d\d' | head -1)
    fi
    if [[ -z "$ts" ]]; then
        # dmesg kernel uptime: [NNNNN.NNN] — convert to wall-clock HH:MM:SS
        local _ksec; _ksec=$(printf '%s' "$line" | grep -oP '^\[\s*\K[0-9]+' | head -1)
        if [[ -n "$_ksec" ]]; then
            local _uptime_s; _uptime_s=$(awk '{print int($1)}' /proc/uptime 2>/dev/null || echo 0)
            local _now_s;    _now_s=$(date +%s 2>/dev/null || echo 0)
            ts=$(date -d "@$(( _now_s - _uptime_s + _ksec ))" '+%H:%M:%S' 2>/dev/null || echo "")
        fi
    fi

    # Fixed-width columns: timestamp(10) + tag(7) + message
    local ts_fmt; ts_fmt=$(printf '%-10s' "${ts:-?}")
    local prefix="${C_DIM}${ts_fmt}${C_RESET}"

    if   printf '%s' "$msg" | grep -q 'KV spill'; then
        printf '%b\n' "  ${prefix}  ${C_CYAN}$(printf '%-7s' 'T1→T2')${C_RESET}  ${C_CYAN}KV${C_RESET}  ${C_GRAY}${msg}${C_RESET}"
    elif printf '%s' "$msg" | grep -qE 'auto-evict|Evicted.*cold'; then
        printf '%b\n' "  ${prefix}  ${C_AMBER}$(printf '%-7s' 'T2→T3')${C_RESET}  ${C_AMBER}evict${C_RESET}  ${C_GRAY}${msg}${C_RESET}"
    elif printf '%s' "$msg" | grep -q 'safety-net triggered'; then
        printf '%b\n' "  ${prefix}  ${C_RED}$(printf '%-7s' 'T2→T3')${C_RESET}  ${C_RED}spill${C_RESET}  ${C_GRAY}${msg}${C_RESET}"
    elif printf '%s' "$msg" | grep -q 'KV cache T2 full'; then
        printf '%b\n' "  ${prefix}  ${C_RED}$(printf '%-7s' 'KV')${C_RESET}  ${C_RED}BLOCKED${C_RESET}  ${C_GRAY}${msg}${C_RESET}"
    elif printf '%s' "$msg" | grep -q 'KV reserve set'; then
        printf '%b\n' "  ${prefix}  ${C_CYAN}$(printf '%-7s' 'KV RSV')${C_RESET}  ${C_GRAY}${msg}${C_RESET}"
    elif printf '%s' "$msg" | grep -q 'CRITICAL'; then
        printf '%b\n' "  ${prefix}  ${C_RED}$(printf '%-7s' '⚠')${C_RESET}  ${C_RED}${msg}${C_RESET}"
    elif printf '%s' "$msg" | grep -q 'WARN'; then
        printf '%b\n' "  ${prefix}  ${C_AMBER}$(printf '%-7s' '⚠')${C_RESET}  ${C_AMBER}${msg}${C_RESET}"
    elif printf '%s' "$msg" | grep -q 'New process'; then
        printf '%b\n' "  ${prefix}  ${C_LIME}$(printf '%-7s' '+proc')${C_RESET}  ${C_LIME}${msg}${C_RESET}"
    elif printf '%s' "$msg" | grep -q 'exited'; then
        printf '%b\n' "  ${prefix}  ${C_DIM}$(printf '%-7s' '-proc')${C_RESET}  ${C_DIM}${msg}${C_RESET}"
    else
        printf '%b\n' "  ${prefix}  ${C_DIM}$(printf '%-7s' '·')${C_RESET}  ${C_GRAY}${msg}${C_RESET}"
    fi
}

# ---- Update check -----------------------------------------------------
# Queries the GitLab tags API; prints a warning if a newer tag exists.
# If stdin is a TTY, offers to git pull and re-exec the installer.
# Non-blocking: 5-second timeout; silently skips if network is unavailable.

check_update() {
    local raw latest

    # Fetch the tag list — try curl first, then wget
    if command -v curl &>/dev/null; then
        raw=$(curl -fsSL --max-time 5 "$GB_REPO_API" 2>/dev/null)
    elif command -v wget &>/dev/null; then
        raw=$(wget -qO- --timeout=5 "$GB_REPO_API" 2>/dev/null)
    else
        return   # no HTTP client available — skip silently
    fi

    [[ -z "$raw" ]] && return   # network unavailable or empty response

    # Extract the highest version tag (e.g. "v2.6" → "2.6")
    # Tags are returned newest-first by the API; we sort anyway for safety.
    latest=$(echo "$raw" \
        | grep -oP '"name"\s*:\s*"v?\K[0-9]+\.[0-9]+(?:\.[0-9]+)?"' \
        | tr -d '"' \
        | sort -V \
        | tail -1)

    [[ -z "$latest" ]] && return   # no version tags found

    # Compare: if latest > GB_VERSION, warn and optionally update
    local newer
    newer=$(printf '%s\n%s\n' "$GB_VERSION" "$latest" | sort -V | tail -1)
    if [[ "$newer" != "$GB_VERSION" ]]; then
        echo ""
        gb_panel_top "Update available"
        gb_panel_row "  ${C_AMBER}A newer GreenBoost version is available:${C_RESET}  ${C_LIME}${C_BOLD}v${latest}${C_RESET}"
        gb_panel_row "  ${C_GRAY}You are running:${C_RESET}                          ${C_DIM}v${GB_VERSION}${C_RESET}"
        gb_panel_bottom
        echo ""

        # Only prompt when stdin is an interactive terminal
        if [[ -t 0 ]]; then
            if gb_confirm "Fetch the update and restart the installer?"; then
                info "Pulling latest release..."
                # Run git pull as the original (non-root) user when invoked via sudo
                local _pull_ok=0
                if [[ -n "${SUDO_USER:-}" ]]; then
                    sudo -u "$SUDO_USER" git -C "$MODULE_DIR" pull origin main \
                        && _pull_ok=1
                else
                    git -C "$MODULE_DIR" pull origin main \
                        && _pull_ok=1
                fi
                if [[ $_pull_ok -eq 1 ]]; then
                    info "Update complete — restarting installer..."
                    exec "$0" "${GB_ORIG_ARGS[@]}"
                else
                    warn "git pull failed — continuing with v${GB_VERSION}."
                fi
            fi
        fi
    else
        info "Version check: v${GB_VERSION} is the latest release."
    fi
}

# Compute T2 DDR pool cap in GB.
# < 64 GB total RAM -> 70%;  >= 64 GB -> 80%.  Minimum 4 GB.
gb_calc_ddr_cap_gb() {
    local total_gb=$1
    local cap_gb
    if (( total_gb >= 64 )); then
        cap_gb=$(( total_gb * 80 / 100 ))
    else
        cap_gb=$(( total_gb * 70 / 100 ))
    fi
    [[ $cap_gb -lt 4 ]] && cap_gb=4
    echo "$cap_gb"
}

# ---- Hardware detection -----------------------------------------------
# Populates GB_PHYS, GB_VIRT, GB_RESERVE, GB_NVME_SWAP, GB_NVME_POOL,
# GB_PCORES_MAX, GB_GOLDEN_MIN, GB_GOLDEN_MAX, GB_PCORES_ONLY,
# RAM_TYPE, RAM_SPEED_MT, GPU_NAME, CPU_NAME, NVME_SIZE_GB,
# PCIE_GEN, PCIE_WIDTH, PCIE_BW_GBS, PCIE_MAX_GEN, PCIE_MAX_BW_GBS.
# Safe to call multiple times — idempotent.

detect_hardware() {
    [[ "${_GB_HW_DETECTED:-0}" -eq 1 ]] && return 0
    # ── GPU ──────────────────────────────────────────────────────────────
    # nvidia-smi can hang when the driver is wedged (suspend/resume, Xid fault).
    # One combined query avoids two separate NVML cold-init round trips.
    # 30 s covers the worst-case first-call-as-root NVML initialization latency.
    if command -v nvidia-smi &>/dev/null; then
        local smi_line smi_ec
        smi_line=$(timeout 30 nvidia-smi \
            --query-gpu=name,memory.total \
            --format=csv,noheader,nounits 2>/dev/null | head -1)
        smi_ec=${PIPESTATUS[0]}
        if [[ $smi_ec -eq 124 ]]; then
            warn "nvidia-smi timed out (>30 s) — GPU driver may be wedged."
            warn "  Try: sudo systemctl restart nvidia-persistenced && sudo nvidia-smi"
            GPU_NAME="Unknown (nvidia-smi timeout)"
            GB_PHYS=12
        elif [[ -z "$smi_line" ]]; then
            warn "nvidia-smi returned no output — check: sudo nvidia-smi"
            GPU_NAME="Unknown (nvidia-smi no output)"
            GB_PHYS=12
        else
            GPU_NAME=$(echo "$smi_line" | cut -d, -f1 | sed 's/^[[:space:]]*//')
            local gpu_mem_mib; gpu_mem_mib=$(echo "$smi_line" | cut -d, -f2 | tr -d ' ')
            GB_PHYS=$(( gpu_mem_mib / 1024 ))
            [[ $GB_PHYS -lt 1 ]] && GB_PHYS=1
        fi
    else
        GPU_NAME="Unknown (nvidia-smi not found)"
        GB_PHYS=8
        warn "nvidia-smi not found — assuming 8 GB VRAM. Install NVIDIA driver."
    fi

    # ── System RAM ───────────────────────────────────────────────────────
    local total_ram_kb
    total_ram_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    local total_ram_gb=$(( total_ram_kb / 1024 / 1024 ))

    local dmi_mem
    dmi_mem=$(dmidecode -t memory 2>/dev/null)
    RAM_TYPE=$(echo "$dmi_mem" \
        | awk '/^\s*Type:/{t=$2} /^\s*Configured Memory Speed:/{if(t && t!="Unknown") {print t; exit}}')
    [[ -z "$RAM_TYPE" || "$RAM_TYPE" == "Unknown" ]] && RAM_TYPE="DDR"
    RAM_SPEED_MT=$(echo "$dmi_mem" \
        | awk '/^\s*Configured Memory Speed:/{print $4; exit}')
    [[ -z "$RAM_SPEED_MT" || "$RAM_SPEED_MT" == "Unknown" ]] && RAM_SPEED_MT="?"

    # ── PCIe link (GPU slot) ──────────────────────────────────────────────
    # current_link_speed is unreliable — PCIe ASPM downclock to Gen 1 when the
    # GPU is in idle/P8 power state.  Use the parent bridge max_link_speed for
    # the slot capability, and the GPU sysfs max_link_speed for GPU hardware max.
    PCIE_GEN=0; PCIE_WIDTH="x?"; PCIE_BW_GBS=0
    PCIE_MAX_GEN=0; PCIE_MAX_BW_GBS=0
    local gpu_pci
    gpu_pci=$(ls /proc/driver/nvidia/gpus/ 2>/dev/null | head -1)
    if [[ -n "$gpu_pci" ]]; then
        local gpu_sysfs parent_sysfs
        gpu_sysfs=$(readlink -f "/sys/bus/pci/devices/$gpu_pci")
        parent_sysfs=$(readlink -f "$gpu_sysfs/..")

        # helper: map GT/s speed string ("16.0 GT/s PCIe") → PCIe gen number
        _gts_to_gen() {
            local gts; gts=$(echo "$1" | grep -oP '^[0-9]+'); gts="${gts:-0}"
            if   [[ $gts -ge 64 ]]; then echo 6
            elif [[ $gts -ge 32 ]]; then echo 5
            elif [[ $gts -ge 16 ]]; then echo 4
            elif [[ $gts -ge 8  ]]; then echo 3
            elif [[ $gts -ge 5  ]]; then echo 2
            elif [[ $gts -ge 1  ]]; then echo 1
            else echo 0; fi
        }
        # helper: x16 bandwidth in GB/s (128b/130b encoding, Gen3+)
        _pcie_bw() { echo $(( $1 * $2 * 128 / 130 / 8 )); }

        # Slot capability: parent bridge max_link_speed + max_link_width
        local slot_spd slot_w
        slot_spd=$(cat "$parent_sysfs/max_link_speed" 2>/dev/null)
        slot_w=$(  cat "$parent_sysfs/max_link_width"  2>/dev/null)
        slot_w="${slot_w//[^0-9]/}"; slot_w="${slot_w:-0}"
        if [[ -n "$slot_spd" ]]; then
            PCIE_GEN=$(_gts_to_gen "$slot_spd")
            PCIE_WIDTH="x${slot_w:-?}"
            [[ $slot_w -gt 0 ]] && \
                PCIE_BW_GBS=$(_pcie_bw "$(echo "$slot_spd" | grep -oP '^[0-9]+')" "$slot_w")
        fi

        # GPU hardware max: GPU device max_link_speed
        local gpu_max_spd gpu_max_w
        gpu_max_spd=$(cat "$gpu_sysfs/max_link_speed" 2>/dev/null)
        gpu_max_w=$(  cat "$gpu_sysfs/max_link_width"  2>/dev/null)
        gpu_max_w="${gpu_max_w//[^0-9]/}"; gpu_max_w="${gpu_max_w:-0}"
        if [[ -n "$gpu_max_spd" ]]; then
            PCIE_MAX_GEN=$(_gts_to_gen "$gpu_max_spd")
            [[ $gpu_max_w -gt 0 ]] && \
                PCIE_MAX_BW_GBS=$(_pcie_bw "$(echo "$gpu_max_spd" | grep -oP '^[0-9]+')" "$gpu_max_w")
        fi
    fi

    # safety reserve: 8% of RAM, min 4 GB, max 10 GB
    GB_RESERVE=$(( total_ram_gb * 8 / 100 ))
    [[ $GB_RESERVE -lt 4  ]] && GB_RESERVE=4
    [[ $GB_RESERVE -gt 10 ]] && GB_RESERVE=10

    # virtual VRAM pool: 70% of total RAM (< 64 GB) or 80% (>= 64 GB).
    # The safety_reserve_gb is enforced dynamically by the kernel watchdog at
    # allocation time — do NOT subtract it here or T2 is under-provisioned.
    GB_VIRT=$(gb_calc_ddr_cap_gb "$total_ram_gb")

    # ── CPU topology ─────────────────────────────────────────────────────
    CPU_NAME=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^[[:space:]]*//')
    local total_cpus; total_cpus=$(nproc)

    # Detect Intel hybrid P/E core split.
    # Primary:   lscpu -p — count CPUs per physical core; ≥2 = P-core (HT), 1 = E-core.
    #            Always available (util-linux); survives containers and newer kernels.
    # Secondary: thread_siblings_list — "N-M" range = P-core, bare integer = E-core.
    # Tertiary:  core_type sysfs integer (1=P, 0=E) — kernel 5.17+ Intel-only.
    # Fallback:  non-hybrid path (AMD, older Intel, containers without sysfs).
    local has_ecores=0
    local -a p_cpu_list=() e_cpu_list=()
    local _n _sib _ct _f _fpath

    # ── Primary: lscpu -p ────────────────────────────────────────────────
    local _lscpu_p_count=0 _lscpu_e_count=0
    local _lscpu_p_cpus="" _lscpu_e_cpus=""
    if lscpu -p &>/dev/null; then
        local _lscpu_awk_out
        _lscpu_awk_out=$(lscpu -p 2>/dev/null | grep -v '^#' | awk -F, '
            { count[$2]++; cpulist[$2]=(cpulist[$2]=="" ? $1 : cpulist[$2]","$1) }
            END {
                pc=0; ec=0; pu=""; eu=""
                for (c in count) {
                    if (count[c]>=2) { pc++; pu=(pu==""?cpulist[c]:pu","cpulist[c]) }
                    else             { ec++; eu=(eu==""?cpulist[c]:eu","cpulist[c]) }
                }
                print pc" "ec" "pu" "eu
            }')
        read -r _lscpu_p_count _lscpu_e_count _lscpu_p_cpus _lscpu_e_cpus <<< "$_lscpu_awk_out"
        if [[ ${_lscpu_e_count:-0} -gt 0 && ${_lscpu_p_count:-0} -gt 0 ]]; then
            has_ecores=1
            IFS=',' read -ra p_cpu_list <<< "$_lscpu_p_cpus"
            IFS=',' read -ra e_cpu_list <<< "$_lscpu_e_cpus"
        fi
    fi

    # ── Secondary: thread_siblings_list ─────────────────────────────────
    if [[ $has_ecores -eq 0 && -d /sys/devices/system/cpu/cpu0/topology ]]; then
        p_cpu_list=(); e_cpu_list=()
        for _n in $(seq 0 $(( total_cpus - 1 ))); do
            _sib=$(cat "/sys/devices/system/cpu/cpu${_n}/topology/thread_siblings_list" 2>/dev/null)
            [[ -z "$_sib" ]] && continue
            if [[ "$_sib" == *"-"* ]]; then
                p_cpu_list+=("$_n")
            else
                e_cpu_list+=("$_n")
            fi
        done
        [[ ${#e_cpu_list[@]} -gt 0 && ${#p_cpu_list[@]} -gt 0 ]] && has_ecores=1
    fi

    # ── Tertiary: core_type sysfs integer (kernel 5.17+ Intel) ───────────
    if [[ $has_ecores -eq 0 ]] && compgen -G "/sys/devices/system/cpu/cpu*/topology/core_type" &>/dev/null; then
        p_cpu_list=(); e_cpu_list=()
        for _f in /sys/devices/system/cpu/cpu*/topology/core_type; do
            [[ -r "$_f" ]] || continue
            _ct=$(cat "$_f" 2>/dev/null)
            _n=${_f##*cpu}; _n=${_n%%/*}
            if   [[ "$_ct" == "1" ]]; then p_cpu_list+=("$_n")
            elif [[ "$_ct" == "0" ]]; then e_cpu_list+=("$_n")
            fi
        done
        [[ ${#e_cpu_list[@]} -gt 0 && ${#p_cpu_list[@]} -gt 0 ]] && has_ecores=1
    fi

    if [[ ${#e_cpu_list[@]} -gt 0 && ${#p_cpu_list[@]} -gt 0 ]]; then
        has_ecores=1
        GB_PCORES_MAX=0
        for _n in "${p_cpu_list[@]}"; do
            [[ $_n -gt $GB_PCORES_MAX ]] && GB_PCORES_MAX=$_n
        done
        GB_PCORES_ONLY=1
        # Golden core detection: P-cores with highest cpuinfo_max_freq
        local _max_freq=0 _gfreq _gold_min=$GB_PCORES_MAX _gold_max=0
        for _n in "${p_cpu_list[@]}"; do
            _fpath="/sys/devices/system/cpu/cpu${_n}/cpufreq/cpuinfo_max_freq"
            [[ -r "$_fpath" ]] || continue
            _gfreq=$(< "$_fpath")
            [[ $_gfreq -gt $_max_freq ]] && _max_freq=$_gfreq
        done
        if [[ $_max_freq -gt 0 ]]; then
            for _n in "${p_cpu_list[@]}"; do
                _fpath="/sys/devices/system/cpu/cpu${_n}/cpufreq/cpuinfo_max_freq"
                [[ -r "$_fpath" ]] || continue
                _gfreq=$(< "$_fpath")
                if [[ $_gfreq -eq $_max_freq ]]; then
                    [[ $_n -lt $_gold_min ]] && _gold_min=$_n
                    [[ $_n -gt $_gold_max ]] && _gold_max=$_n
                fi
            done
            GB_GOLDEN_MIN=$_gold_min
            GB_GOLDEN_MAX=$_gold_max
        else
            # No cpufreq — fall back to arithmetic heuristic
            local _pcores_count=$(( GB_PCORES_MAX + 1 ))
            GB_GOLDEN_MIN=$(( _pcores_count / 4 ))
            GB_GOLDEN_MAX=$(( _pcores_count / 4 + 3 ))
        fi
    else
        # Non-hybrid (AMD or older Intel): all CPUs are uniform
        GB_PCORES_MAX=$(( total_cpus - 1 ))
        GB_PCORES_ONLY=0
        GB_GOLDEN_MIN=0
        GB_GOLDEN_MAX=$(( total_cpus > 4 ? 3 : total_cpus - 1 ))
    fi

    # ── NVMe ─────────────────────────────────────────────────────────────
    NVME_SIZE_GB=0
    local name size
    while read -r name size; do
        local sz_gb=$(( size / 1024 / 1024 / 1024 ))
        [[ $sz_gb -gt $NVME_SIZE_GB ]] && NVME_SIZE_GB=$sz_gb
    done < <(lsblk -b -d -o NAME,SIZE 2>/dev/null | awk '$1~/^nvme/{print $1,$2}')
    [[ $NVME_SIZE_GB -eq 0 ]] && NVME_SIZE_GB=128  # fallback

    # ── System swap detection (kept for backward compat / GB_NVME_SWAP) ─────
    local _nvme_swap_kb=0 _other_swap_kb=0
    local _sf _stype _sw_kb _used_kb _prio
    while read -r _sf _stype _sw_kb _used_kb _prio; do
        [[ "$_sf" == "Filename" ]] && continue
        local _bd; _bd=$(df --output=source "$_sf" 2>/dev/null | tail -1)
        if [[ "$_bd" == /dev/nvme* || "$_bd" == /dev/mapper/* ]]; then
            (( _nvme_swap_kb += _sw_kb )) || true
        else
            (( _other_swap_kb += _sw_kb )) || true
        fi
    done < /proc/swaps
    local _total_swap_kb=$(( _nvme_swap_kb > 0 ? _nvme_swap_kb : _other_swap_kb ))
    GB_NVME_SWAP=$(( _total_swap_kb / 1024 / 1024 ))
    [[ $GB_NVME_SWAP -lt 1 ]] && GB_NVME_SWAP=0

    # ── T3 pool: sized to guarantee 120B-class models (120 GB) load ─────────
    # Formula: max(0, 120 − T1 − T2) + 10% buffer, capped to 80% of free disk.
    local _t3_target=120
    local _t3_needed=$(( _t3_target - GB_PHYS - GB_VIRT ))
    [[ $_t3_needed -lt 0 ]] && _t3_needed=0
    _t3_needed=$(( _t3_needed * 110 / 100 ))          # 10% buffer
    [[ $_t3_needed -gt 0 && $_t3_needed -lt 32 ]] && _t3_needed=32
    local _t3_disk_gb
    # Fall back through parent directories when /var/lib/greenboost doesn't exist yet
    # (fresh install), otherwise GB_NVME_POOL would be capped to 0.
    _t3_disk_gb=$(df -BG /var/lib/greenboost 2>/dev/null | awk 'NR==2{gsub("G",""); print $4}')
    if [[ -z "$_t3_disk_gb" ]]; then
        _t3_disk_gb=$(df -BG /var/lib 2>/dev/null | awk 'NR==2{gsub("G",""); print $4}')
    fi
    if [[ -z "$_t3_disk_gb" ]]; then
        _t3_disk_gb=$(df -BG / 2>/dev/null | awk 'NR==2{gsub("G",""); print $4}')
    fi
    _t3_disk_gb="${_t3_disk_gb:-0}"
    local _t3_disk_cap=$(( _t3_disk_gb * 80 / 100 ))
    [[ $_t3_needed -gt $_t3_disk_cap ]] && _t3_needed=$_t3_disk_cap
    GB_NVME_POOL=$_t3_needed

    # ── Ollama CTX based on T1+T2 KV headroom (T3 NVMe excluded) ────────
    # KV cache never spills to T3; use T1 VRAM + half T2 DDR as budget.
    # Half of T2 is left for model weights; the other half is KV headroom.
    local kv_pool_gb=$(( GB_PHYS + GB_VIRT / 2 ))
    if   [[ $kv_pool_gb -ge 32 ]]; then GB_OLLAMA_CTX=131072
    elif [[ $kv_pool_gb -ge 16 ]]; then GB_OLLAMA_CTX=65536
    elif [[ $kv_pool_gb -ge 8  ]]; then GB_OLLAMA_CTX=32768
    elif [[ $kv_pool_gb -ge 4  ]]; then GB_OLLAMA_CTX=16384
    else                                 GB_OLLAMA_CTX=8192
    fi

    GB_HUGEPAGES=1

    # ── Extra fields for profile generation ──────────────────────────────
    DET_GPU_COUNT=$(timeout 10 nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)
    DET_CC=$(timeout 10 nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "0.0")
    DET_PCIE_GEN_SMI=$(timeout 10 nvidia-smi --query-gpu=pcie.link.gen.max --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "$PCIE_MAX_GEN")
    DET_PCIE_LANES_SMI=$(timeout 10 nvidia-smi --query-gpu=pcie.link.width.max --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "16")
    DET_DRIVER=$(timeout 10 nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "")
    DET_CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | grep -oP '[0-9]+\.[0-9]+' | head -1 || echo "")
    DET_NVLINK=false
    timeout 10 nvidia-smi topo -m 2>/dev/null | grep -q "NV[0-9]" && DET_NVLINK=true
    DET_RAM_ECC=false
    dmidecode -t memory 2>/dev/null | grep -qi "Error Correction.*Multi" && DET_RAM_ECC=true
    DET_RAM_CHANNELS=$(dmidecode -t memory 2>/dev/null | grep -c "^\s*Size:.*GB" || echo 2)
    DET_TOTAL_LOGICAL_CPUS=$(nproc 2>/dev/null || echo 0)
    DET_NUMA=$(lscpu 2>/dev/null | grep "^NUMA node(s):" | awk '{print $3}' || echo 1)
    DET_L3_MB=$(lscpu 2>/dev/null | grep "^L3 cache:" | grep -oP '[0-9]+(?=\s*[MG]iB)' | head -1 || echo 0)
    # If GiB, convert to MiB
    if lscpu 2>/dev/null | grep "^L3 cache:" | grep -q "GiB"; then
        DET_L3_MB=$(( ${DET_L3_MB:-0} * 1024 ))
    fi
    DET_OS=$(lsb_release -d 2>/dev/null | cut -d: -f2 | xargs || echo "Unknown")
    DET_KERNEL=$(uname -r)
    # Use lsblk (always available) as primary; nvme CLI as secondary for model if lsblk has none
    DET_NVME_COUNT=$(lsblk -d -o NAME,TYPE 2>/dev/null | awk '$2=="disk" && $1~/^nvme/{c++} END{print c+0}')
    DET_NVME_0_MODEL=$(lsblk -d -o NAME,MODEL 2>/dev/null \
        | awk '$1~/^nvme/{$1=""; gsub(/^[[:space:]]+/,""); print; exit}' | xargs)
    [[ -z "$DET_NVME_0_MODEL" ]] && \
        DET_NVME_0_MODEL=$(nvme list 2>/dev/null | awk 'NR==2{$1=""; print $0}' | xargs || true)
    DET_IB_SPEED=0
    DET_IB_ADAPTER=""
    if command -v ibstat &>/dev/null; then
        DET_IB_ADAPTER=$(ibstat 2>/dev/null | grep "CA '" | head -1 | cut -d"'" -f2 || echo "")
        DET_IB_SPEED=$(ibstat 2>/dev/null | grep "Rate:" | head -1 | awk '{print $2}' || echo 0)
    fi
    DET_MPI_ENABLED=false
    command -v mpirun &>/dev/null && DET_MPI_ENABLED=true
    DET_HAS_PE_SPLIT=false
    DET_P_CORES=0
    DET_E_CORES=0
    DET_P_CORE_CPUS=""
    DET_E_CORE_CPUS=""
    if [[ $has_ecores -eq 1 ]]; then
        DET_HAS_PE_SPLIT=true
        # Physical P-core count: unique thread_siblings_list groups among P-core CPUs
        local -A _seen_sibs=()
        local _phys_p=0
        for _n in "${p_cpu_list[@]}"; do
            _sib=$(cat "/sys/devices/system/cpu/cpu${_n}/topology/thread_siblings_list" 2>/dev/null)
            [[ -z "$_sib" ]] && _sib="$_n"
            if [[ -z "${_seen_sibs[$_sib]+x}" ]]; then
                _seen_sibs[$_sib]=1
                (( _phys_p++ )) || true
            fi
        done
        DET_P_CORES=$_phys_p
        DET_E_CORES=${#e_cpu_list[@]}
        DET_P_CORE_CPUS=$(IFS=,; echo "${p_cpu_list[*]}")
        DET_E_CORE_CPUS=$(IFS=,; echo "${e_cpu_list[*]}")
    fi
    _GB_HW_DETECTED=1
}

print_detected_hardware() {
    local pcie_info
    if [[ $PCIE_GEN -eq 0 ]]; then
        pcie_info="unknown (GPU PCI address not found)"
    else
        pcie_info="Gen ${PCIE_GEN} ${PCIE_WIDTH}  (~${PCIE_BW_GBS} GB/s)"
        if [[ $PCIE_MAX_GEN -gt $PCIE_GEN ]]; then
            pcie_info+="  |  GPU max: Gen ${PCIE_MAX_GEN} x${PCIE_WIDTH#x}  (~${PCIE_MAX_BW_GBS} GB/s if slot supports it)"
        fi
    fi
    info "Detected hardware:"
    info "  GPU   : ${GPU_NAME}  (${GB_PHYS} GB VRAM)"
    info "  PCIe  : ${pcie_info}"
    info "  RAM   : ${RAM_TYPE}-${RAM_SPEED_MT} MT/s  ->  pool ${GB_VIRT} GB  (reserve ${GB_RESERVE} GB)"
    info "  CPU   : ${CPU_NAME}"
    local _t3_store="/var/lib/greenboost/t3_store"
    if [[ -f "$_t3_store" ]]; then
        local _t3_gb=$(( $(stat -c%s "$_t3_store" 2>/dev/null || echo 0) / 1073741824 ))
        info "  NVMe  : ${NVME_SIZE_GB} GB  ->  T3 backing ${_t3_gb} GB"
        info "  Pool  : T1=${GB_PHYS}GB + T2=${GB_VIRT}GB + T3=${_t3_gb}GB = $(( GB_PHYS + GB_VIRT + _t3_gb )) GB"
    else
        info "  NVMe  : ${NVME_SIZE_GB} GB  ->  T3 backing ${GB_NVME_POOL} GB (created on first use)"
        info "  Pool  : T1=${GB_PHYS}GB + T2=${GB_VIRT}GB + T3=${GB_NVME_POOL}GB = $(( GB_PHYS + GB_VIRT + GB_NVME_POOL )) GB"
    fi
    info "  CTX   : OLLAMA_NUM_CTX=${GB_OLLAMA_CTX}"
}

# ---- Profile write ----------------------------------------------------
# write_profile <output_path> <generated_by>
# Writes a Markdown profile file from the detect_hardware() populated vars.
# Must call detect_hardware() first.

write_profile() {
    local out="$1" generated_by="${2:-autodetect}"
    local now; now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local profile_name; profile_name=$(hostname -s 2>/dev/null || echo "local")
    local profile_type="workstation"
    [[ ${DET_GPU_COUNT:-1} -gt 1 ]] && profile_type="server"
    [[ "${DET_IB_SPEED:-0}" -gt 0 ]] && profile_type="cluster_node"

    # Recompute virtual pool size directly from live /proc/meminfo to avoid
    # stale values from a prior cached detect_hardware() call.
    local _ram_kb; _ram_kb=$(awk '/^MemTotal/{print $2}' /proc/meminfo)
    local _total_gb=$(( _ram_kb / 1024 / 1024 ))
    GB_VIRT=$(gb_calc_ddr_cap_gb "$_total_gb")

    mkdir -p "$(dirname "$out")"
    cat > "$out" << PROFILE_EOF
---
profile_version: "1.0"
profile_name: "${profile_name}"
profile_type: "${profile_type}"
created: "${now}"
generated_by: "${generated_by}"
greenboost_version: "${GB_VERSION}"
---

## Hardware

### CPU
cpu_model: "${CPU_NAME:-Unknown}"
cpu_arch: x86_64
logical_cpus: ${DET_TOTAL_LOGICAL_CPUS:-0}
p_cores: ${DET_P_CORES:-0}
e_cores: ${DET_E_CORES:-0}
p_core_cpus: "${DET_P_CORE_CPUS:-}"
e_core_cpus: "${DET_E_CORE_CPUS:-}"
pcores_max_cpu: ${GB_PCORES_MAX:-0}
golden_cpu_min: ${GB_GOLDEN_MIN:-0}
golden_cpu_max: ${GB_GOLDEN_MAX:-0}
l3_cache_mb: ${DET_L3_MB:-0}
numa_nodes: ${DET_NUMA:-1}
has_pe_split: ${DET_HAS_PE_SPLIT:-false}

### GPU
gpu_count: ${DET_GPU_COUNT:-1}
gpu_model: "${GPU_NAME:-Unknown}"
vram_gb: ${GB_PHYS:-0}
compute_capability: "${DET_CC:-}"
pcie_gen: ${DET_PCIE_GEN_SMI:-${PCIE_MAX_GEN:-4}}
pcie_lanes: ${DET_PCIE_LANES_SMI:-16}
nvlink: ${DET_NVLINK:-false}
driver_version: "${DET_DRIVER:-}"
cuda_version: "${DET_CUDA_VER:-}"

### RAM
ram_total_gb: $(($(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024))
ram_type: ${RAM_TYPE:-DDR4}
ram_speed_mt: ${RAM_SPEED_MT:-0}
ram_ecc: ${DET_RAM_ECC:-false}

### Storage
nvme_count: ${DET_NVME_COUNT:-0}
nvme_0_model: "${DET_NVME_0_MODEL:-}"
nvme_0_capacity_tb: $(awk "BEGIN {printf \"%.1f\", ${NVME_SIZE_GB:-0}/1000}")

### OS
os_distro: "${DET_OS:-}"
kernel_version: "${DET_KERNEL:-}"

## GreenBoost Parameters

physical_vram_gb: ${GB_PHYS:-0}
virtual_vram_gb: ${GB_VIRT:-0}
safety_reserve_gb: ${GB_RESERVE:-0}
nvme_swap_gb: ${GB_NVME_SWAP:-0}
nvme_pool_gb: ${GB_NVME_POOL:-0}
use_hugepages: ${GB_HUGEPAGES:-1}
ecores_only: ${GB_PCORES_ONLY:-0}
debug_mode: 0
tier3_backend: nvme

## Ollama / Inference Runtime

ollama_flash_attention: 1
ollama_kv_cache_type: q8_0
ollama_num_ctx: ${GB_OLLAMA_CTX:-131072}
kv_reserve_mb: ${GB_KV_RESERVE_MB:-2048}

## Profile Notes

Auto-detected profile. Run 'greenboost_setup.sh profile diff' to compare vs live hardware.
PROFILE_EOF

    if [[ "${DET_IB_SPEED:-0}" -gt 0 ]]; then
        cat >> "$out" << NET_EOF

### Networking
ib_adapter: "${DET_IB_ADAPTER:-}"
ib_speed_gbps: ${DET_IB_SPEED:-0}
ib_protocol: InfiniBand
mpi_enabled: ${DET_MPI_ENABLED:-false}
NET_EOF
    fi

    info "Profile written: $out"
}

# parse_profile_field <file> <field>
# Extracts a YAML field value from a profile file (key: value format).
parse_profile_field() {
    local file="$1" field="$2"
    grep -m1 "^${field}:" "$file" 2>/dev/null | cut -d: -f2- | sed 's/^[[:space:]]*//' | tr -d '"'
}

# load_profile_values <file>
# Loads profile parameter fields into PROF_* variables.
load_profile_values() {
    local file="$1"
    [[ -f "$file" ]] || return 1
    PROF_PHYS=$(parse_profile_field "$file" physical_vram_gb)
    PROF_VIRT=$(parse_profile_field "$file" virtual_vram_gb)
    PROF_RESERVE=$(parse_profile_field "$file" safety_reserve_gb)
    PROF_NVME_SWAP=$(parse_profile_field "$file" nvme_swap_gb)
    PROF_NVME_POOL=$(parse_profile_field "$file" nvme_pool_gb)
    PROF_HUGEPAGES=$(parse_profile_field "$file" use_hugepages)
    PROF_PCORES_ONLY=$(parse_profile_field "$file" ecores_only)
    PROF_PCORES_MAX=$(parse_profile_field "$file" pcores_max_cpu)
    PROF_GOLDEN_MIN=$(parse_profile_field "$file" golden_cpu_min)
    PROF_GOLDEN_MAX=$(parse_profile_field "$file" golden_cpu_max)
    PROF_OLLAMA_CTX=$(parse_profile_field "$file" ollama_num_ctx)
    PROF_TIER3=$(parse_profile_field "$file" tier3_backend)
    PROF_NAME=$(parse_profile_field "$file" profile_name)
    PROF_TYPE=$(parse_profile_field "$file" profile_type)
    PROF_VRAM_GB=$(parse_profile_field "$file" vram_gb)
    PROF_NVLINK=$(parse_profile_field "$file" nvlink)
    PROF_CC=$(parse_profile_field "$file" compute_capability)
}

# resolve_profile <user_profile_file>
# Cross-checks user profile against auto-detected hardware.
# Writes resolved profile to /etc/greenboost/profiles/resolved_<timestamp>.md
# Updates active_profile.md symlink to resolved profile.
# Sets GB_PHYS, GB_VIRT, GB_RESERVE, GB_NVME_SWAP, GB_NVME_POOL, GB_PCORES_ONLY
# from the resolved values so cmd_load() picks them up.

resolve_profile() {
    local user_file="$1"
    [[ -f "$user_file" ]] || die "Profile file not found: $user_file"

    info "Resolving profile: $user_file"
    detect_hardware
    load_profile_values "$user_file"

    local conflicts=""
    local resolved_phys=$GB_PHYS
    local resolved_virt=$GB_VIRT
    local resolved_reserve=$GB_RESERVE
    local resolved_nvme_sw=$GB_NVME_SWAP
    local resolved_nvme_pool=$GB_NVME_POOL
    local resolved_pcores=${PROF_PCORES_ONLY:-$GB_PCORES_ONLY}
    local resolved_hugepages=${PROF_HUGEPAGES:-1}
    local _mem_kb; _mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}'); _mem_kb="${_mem_kb:-0}"
    local total_ram_gb=$(( _mem_kb / 1024 / 1024 ))

    # Rule: physical_vram_gb — always use detected; ignore if profile claims more
    if [[ -n "$PROF_VRAM_GB" && "$PROF_VRAM_GB" -gt "$GB_PHYS" ]]; then
        conflicts+="physical_vram_gb overridden ${PROF_VRAM_GB}→${GB_PHYS} GB (physical limit); "
    fi

    # Rule: virtual_vram_gb — use profile value if <= 95% RAM, else cap
    if [[ -n "$PROF_VIRT" ]]; then
        local max_virt=$(( total_ram_gb * 95 / 100 ))
        if [[ "$PROF_VIRT" -le "$max_virt" ]]; then
            resolved_virt=$PROF_VIRT
        else
            resolved_virt=$max_virt
            conflicts+="virtual_vram_gb capped ${PROF_VIRT}→${resolved_virt} GB (95% of ${total_ram_gb} GB RAM); "
        fi
    fi

    # Rule: safety_reserve_gb — use max(profile, 6% RAM)
    if [[ -n "$PROF_RESERVE" ]]; then
        local min_reserve=$(( total_ram_gb * 6 / 100 ))
        [[ $min_reserve -lt 4 ]] && min_reserve=4
        if [[ "$PROF_RESERVE" -ge "$min_reserve" ]]; then
            resolved_reserve=$PROF_RESERVE
        else
            resolved_reserve=$min_reserve
            conflicts+="safety_reserve_gb raised ${PROF_RESERVE}→${resolved_reserve} GB (6% RAM minimum); "
        fi
    fi

    # Rule: nvme_swap_gb — use profile if <= free NVMe space, else cap at 90% free
    if [[ -n "$PROF_NVME_SWAP" ]]; then
        local nvme_free_gb; nvme_free_gb=$(df -BG "${GB_NVME_SWAPFILE:-/}" 2>/dev/null | awk 'NR==2{gsub("G",""); print $4}' || echo 0)
        nvme_free_gb="${nvme_free_gb:-0}"
        local nvme_cap=$(( nvme_free_gb * 90 / 100 ))
        [[ $nvme_cap -lt 1 ]] && nvme_cap=$GB_NVME_SWAP
        if [[ "$PROF_NVME_SWAP" -le "${NVME_SIZE_GB:-9999}" ]]; then
            resolved_nvme_sw=$PROF_NVME_SWAP
        else
            resolved_nvme_sw=$nvme_cap
            conflicts+="nvme_swap_gb capped ${PROF_NVME_SWAP}→${resolved_nvme_sw} GB (NVMe capacity); "
        fi
    fi

    # Rule: nvme_pool_gb — min(profile, nvme_swap * 0.89)
    if [[ -n "$PROF_NVME_POOL" ]]; then
        local max_pool=$(( resolved_nvme_sw * 89 / 100 ))
        if [[ "$PROF_NVME_POOL" -le "$max_pool" ]]; then
            resolved_nvme_pool=$PROF_NVME_POOL
        else
            resolved_nvme_pool=$max_pool
            conflicts+="nvme_pool_gb capped ${PROF_NVME_POOL}→${resolved_nvme_pool} GB (89% of nvme_swap); "
        fi
    fi

    # Rule: ecores_only — warn if P/E split not detected
    if [[ "$resolved_pcores" == "1" && "${DET_HAS_PE_SPLIT}" == "false" ]]; then
        warn "ecores_only=1 in profile but no P/E-core split detected — module will use all CPUs"
    fi

    # Rule: nvlink_pool warning
    local prof_nvlink_pool; prof_nvlink_pool=$(parse_profile_field "$user_file" nvlink_pool)
    if [[ "$prof_nvlink_pool" == "true" && "${DET_NVLINK}" == "false" ]]; then
        warn "nvlink_pool=true in profile but no NVLink topology detected — overriding to false"
        prof_nvlink_pool="false"
        conflicts+="nvlink_pool overridden true→false (no NVLink detected); "
    fi

    # Apply resolved values back to GB_* so cmd_load() picks them up
    GB_PHYS=$resolved_phys
    GB_VIRT=$resolved_virt
    GB_RESERVE=$resolved_reserve
    GB_NVME_SWAP=$resolved_nvme_sw
    GB_NVME_POOL=$resolved_nvme_pool
    GB_PCORES_ONLY=$resolved_pcores
    [[ -n "${PROF_HUGEPAGES}" ]] && GB_HUGEPAGES=$resolved_hugepages
    [[ -n "${PROF_OLLAMA_CTX}" ]] && GB_OLLAMA_CTX=$PROF_OLLAMA_CTX

    # Write resolved profile
    mkdir -p "$GB_PROFILES_DIR"
    local ts; ts=$(date +"%Y%m%d_%H%M%S")
    local resolved_file="${GB_PROFILES_DIR}/resolved_${ts}.md"

    cp "$user_file" "$resolved_file"

    local summary="All user values accepted."
    [[ -n "$conflicts" ]] && summary="Resolved: ${conflicts%%; }"

    cat >> "$resolved_file" << RESOLVED_EOF

## Resolution Notes (${ts})
${summary}
RESOLVED_EOF

    # Update symlink
    ln -sf "$resolved_file" "$GB_ACTIVE_PROFILE_LINK"

    if [[ -n "$conflicts" ]]; then
        warn "GreenBoost profile resolved: ${conflicts%%; }"
    else
        info "Profile accepted without changes."
    fi
    info "Active profile: $resolved_file"
}

# ---- Profile subcommands -----------------------------------------------

cmd_profile_create() {
    need_root "profile create"
    _GB_HW_DETECTED=0   # force fresh detection — discard any cached state
    detect_hardware
    mkdir -p "$GB_PROFILES_DIR"
    local out="${GB_PROFILES_DIR}/default.md"
    write_profile "$out" "autodetect"
    ln -sf "$out" "$GB_ACTIVE_PROFILE_LINK"
    info "Active profile set to: $out"
}

cmd_profile_show() {
    local file="${1:-}"
    if [[ -z "$file" ]]; then
        [[ -L "$GB_ACTIVE_PROFILE_LINK" || -f "$GB_ACTIVE_PROFILE_LINK" ]] \
            || die "No active profile. Run: sudo $0 profile create"
        file=$(readlink -f "$GB_ACTIVE_PROFILE_LINK")
    fi
    [[ -f "$file" ]] || die "Profile not found: $file"
    info "Profile: $file"
    echo ""
    cat "$file"
}

cmd_profile_list() {
    [[ -d "$GB_PROFILES_DIR" ]] || { info "No profiles directory ($GB_PROFILES_DIR). Run: sudo $0 profile create"; return; }
    local active=""
    [[ -L "$GB_ACTIVE_PROFILE_LINK" ]] && active=$(readlink -f "$GB_ACTIVE_PROFILE_LINK")
    info "Profiles in $GB_PROFILES_DIR:"
    local f
    for f in "$GB_PROFILES_DIR"/*.md; do
        [[ -f "$f" ]] || continue
        local mark="  "
        [[ "$f" == "$active" ]] && mark="* "
        echo "  ${mark}$(basename "$f")"
    done
    echo ""
    [[ -n "$active" ]] && info "Active: $active"
}

cmd_profile_activate() {
    need_root "profile activate"
    local file="$1"
    [[ -f "$file" ]] || die "Profile not found: $file"
    local abs; abs=$(readlink -f "$file")
    if [[ "$abs" != ${GB_PROFILES_DIR}/* ]]; then
        mkdir -p "$GB_PROFILES_DIR"
        cp "$abs" "${GB_PROFILES_DIR}/$(basename "$abs")"
        abs="${GB_PROFILES_DIR}/$(basename "$abs")"
    fi
    ln -sf "$abs" "$GB_ACTIVE_PROFILE_LINK"
    info "Active profile: $abs"
}

cmd_profile_diff() {
    local file="${1:-}"
    if [[ -z "$file" ]]; then
        [[ -L "$GB_ACTIVE_PROFILE_LINK" || -f "$GB_ACTIVE_PROFILE_LINK" ]] \
            || die "No active profile. Provide a file or run: sudo $0 profile create"
        file=$(readlink -f "$GB_ACTIVE_PROFILE_LINK")
    fi
    [[ -f "$file" ]] || die "Profile file not found: $file"
    detect_hardware

    info "Comparing profile '$file' vs auto-detected hardware:"
    echo ""

    _diff_field() {
        local label="$1" prof_val="$2" det_val="$3"
        if [[ "$prof_val" == "$det_val" ]]; then
            printf "  %-30s profile=%-20s detected=%-20s  %s\n" "$label" "$prof_val" "$det_val" "OK"
        else
            printf "  %-30s profile=%-20s detected=%-20s  %s\n" "$label" "$prof_val" "$det_val" "DIFF"
        fi
    }

    load_profile_values "$file"
    _diff_field "physical_vram_gb"  "${PROF_VRAM_GB:-?}"      "$GB_PHYS"
    _diff_field "virtual_vram_gb"   "${PROF_VIRT:-?}"          "$GB_VIRT"
    _diff_field "safety_reserve_gb" "${PROF_RESERVE:-?}"       "$GB_RESERVE"
    _diff_field "nvme_swap_gb"      "${PROF_NVME_SWAP:-?}"     "$GB_NVME_SWAP"
    _diff_field "nvme_pool_gb"      "${PROF_NVME_POOL:-?}"     "$GB_NVME_POOL"
    _diff_field "ecores_only"       "${PROF_PCORES_ONLY:-?}"   "$GB_PCORES_ONLY"
    _diff_field "vram (gpu hw)"     "${PROF_VRAM_GB:-?}"       "$GB_PHYS"
    echo ""
}

cmd_profile() {
    local sub="${1:-show}"
    shift 2>/dev/null || true
    case "$sub" in
        create)   cmd_profile_create ;;
        show)     cmd_profile_show "$@" ;;
        list)     cmd_profile_list ;;
        activate) cmd_profile_activate "$@" ;;
        diff)     cmd_profile_diff "$@" ;;
        *) die "Unknown profile subcommand: '$sub'. Use: create|show|list|activate|diff" ;;
    esac
}

# ── cmd_profile_wizard — interactive profile management sub-menu ──────────
cmd_profile_wizard() {
    while true; do
        clear
        # ── Header ──────────────────────────────────────────────────────
        local cols; cols=$(tput cols 2>/dev/null || echo 72)
        local title=" GreenBoost v${GB_VERSION} — Profile Management"
        echo -e ""
        echo -e "${C_VIOLET}${C_BOLD}  ╔$(printf '═%.0s' $(seq 1 $((cols - 4))))╗${C_RESET}"
        echo -e "${C_VIOLET}${C_BOLD}  ║${C_RESET} ${C_GRAY}${C_BOLD}${title}$(printf ' %.0s' $(seq 1 $((cols - 4 - ${#title} - 1))))${C_VIOLET}${C_BOLD}║${C_RESET}"
        echo -e "${C_VIOLET}${C_BOLD}  ╚$(printf '═%.0s' $(seq 1 $((cols - 4))))╝${C_RESET}"
        echo -e ""

        # ── Active profile panel ─────────────────────────────────────────
        if [[ -L "$GB_ACTIVE_PROFILE_LINK" || -f "$GB_ACTIVE_PROFILE_LINK" ]]; then
            local active_path; active_path=$(readlink -f "$GB_ACTIVE_PROFILE_LINK" 2>/dev/null)
            local active_name; active_name=$(basename "$active_path" .md)
            local p_gpu;   p_gpu=$(parse_profile_field  "$active_path" gpu_model)
            local p_phys;  p_phys=$(parse_profile_field "$active_path" physical_vram_gb)
            local p_virt;  p_virt=$(parse_profile_field "$active_path" virtual_vram_gb)
            local p_ram;   p_ram=$(parse_profile_field  "$active_path" ram_total_gb)
            local p_nvme;  p_nvme=$(parse_profile_field "$active_path" nvme_swap_gb)
            local p_ctx;   p_ctx=$(parse_profile_field  "$active_path" ollama_num_ctx)
            gb_panel_top "Active profile"
            gb_panel_row "${C_LIME}${C_BOLD}✓  ${active_name}${C_RESET}  ${C_DIM}${C_GRAY}${active_path}${C_RESET}"
            gb_panel_empty
            gb_panel_row "  ${C_CYAN}GPU${C_RESET}   ${C_GRAY}${p_gpu:-unknown}${C_RESET}"
            gb_panel_row "  ${C_CYAN}VRAM${C_RESET}  ${C_GRAY}T1 ${p_phys:-?} GB physical  →  T1+T2 ${C_LIME}$(( ${p_phys:-0} + ${p_virt:-0} )) GB virtual${C_RESET}"
            gb_panel_row "  ${C_CYAN}DDR${C_RESET}   ${C_GRAY}${p_ram:-?} GB total RAM  ·  T2 pool ${p_virt:-?} GB${C_RESET}"
            gb_panel_row "  ${C_CYAN}NVMe${C_RESET}  ${C_GRAY}T3 swap ${p_nvme:-?} GB  ·  ctx ${p_ctx:-?} tokens${C_RESET}"
            gb_panel_bottom
        else
            echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_AMBER}${C_BOLD}No active profile${C_RESET}  ${C_DIM}— select [1] to auto-detect hardware and create one${C_RESET}"
        fi
        echo -e ""

        # ── Profile list ─────────────────────────────────────────────────
        if [[ -d "$GB_PROFILES_DIR" ]]; then
            local active_abs=""
            [[ -L "$GB_ACTIVE_PROFILE_LINK" ]] && active_abs=$(readlink -f "$GB_ACTIVE_PROFILE_LINK" 2>/dev/null)
            local profiles=( "$GB_PROFILES_DIR"/*.md )
            if [[ -f "${profiles[0]}" ]]; then
                echo -e "  ${C_DIM}${C_GRAY}Saved profiles:${C_RESET}"
                local pf
                for pf in "${profiles[@]}"; do
                    [[ -f "$pf" ]] || continue
                    local pname; pname=$(basename "$pf" .md)
                    if [[ "$pf" == "$active_abs" ]]; then
                        echo -e "    ${C_LIME}${C_BOLD}●  ${pname}${C_RESET}  ${C_DIM}${C_GRAY}(active)${C_RESET}"
                    else
                        echo -e "    ${C_DIM}${C_GRAY}○  ${pname}${C_RESET}"
                    fi
                done
                echo -e ""
            fi
        fi

        # ── Menu ─────────────────────────────────────────────────────────
        gb_section "Actions"
        gb_menu_item 1 "Create / regenerate" "Auto-detect hardware → overwrite default.md"  root
        gb_menu_item 2 "Show active"         "Print full active profile to terminal"
        gb_menu_item 3 "Activate"            "Switch active profile (choose from list)"      root
        gb_menu_item 4 "Diff vs hardware"    "Compare active profile against live detection"
        gb_separator
        echo -e "  ${C_DIM}${C_GRAY}[B]  Back to main menu${C_RESET}"
        echo -e ""

        if [[ "$(id -u)" != "0" ]]; then
            gb_warn_ui "Not root — options marked ${C_RED}⚠ root${C_AMBER} require sudo."
            echo -e ""
        fi

        gb_prompt "Choice"
        local choice="$REPLY"
        echo -e ""

        case "$choice" in
            1)
                cmd_profile_create
                gb_press_enter
                ;;
            2)
                cmd_profile_show
                gb_press_enter
                ;;
            3)
                # Build numbered list of available profiles for selection
                if [[ ! -d "$GB_PROFILES_DIR" ]]; then
                    gb_warn_ui "No profiles directory — run option 1 first."
                    gb_press_enter; continue
                fi
                local profiles=( "$GB_PROFILES_DIR"/*.md )
                if [[ ! -f "${profiles[0]}" ]]; then
                    gb_warn_ui "No profiles found — run option 1 first."
                    gb_press_enter; continue
                fi
                echo -e "  ${C_CYAN}${C_BOLD}Available profiles:${C_RESET}"
                echo -e ""
                local idx=1
                local pf
                for pf in "${profiles[@]}"; do
                    [[ -f "$pf" ]] || continue
                    gb_menu_item "$idx" "$(basename "$pf" .md)" "$pf"
                    (( idx++ ))
                done
                echo -e ""
                gb_prompt "Profile number to activate"
                local sel="$REPLY"
                if [[ "$sel" =~ ^[0-9]+$ ]] && (( sel >= 1 && sel < idx )); then
                    local target="${profiles[$(( sel - 1 ))]}"
                    cmd_profile_activate "$target"
                else
                    gb_warn_ui "Invalid selection."
                fi
                gb_press_enter
                ;;
            4)
                cmd_profile_diff
                gb_press_enter
                ;;
            b|B|"")
                return
                ;;
            *)
                gb_warn_ui "Unknown option."
                sleep 1
                ;;
        esac
    done
}

# Owner-workstation preset — hard-coded optimal for known hardware:
# ASRock B760M-ITX/D4 | i9-14900KF | RTX 5070 OC 12GB | 64GB DDR4-3600 dual-ch | Samsung 990 EVO Plus 4TB

set_owner_workstation_params() {
    GPU_NAME="ASUS RTX 5070 OC (GB205)"
    CPU_NAME="Intel Core i9-14900KF (8Px2HT + 16E = 32 logical, golden CPU 4-7 @ 6GHz)"
    RAM_TYPE="DDR4"
    RAM_SPEED_MT="3600"
    PCIE_GEN=4
    PCIE_WIDTH="x16"
    PCIE_BW_GBS=32
    PCIE_MAX_GEN=5
    PCIE_MAX_BW_GBS=63
    NVME_SIZE_GB=4000
    GB_PHYS=12
    GB_VIRT=51
    GB_RESERVE=12
    GB_NVME_SWAP=128
    GB_NVME_POOL=114
    GB_PCORES_MAX=15
    GB_GOLDEN_MIN=4
    GB_GOLDEN_MAX=7
    GB_PCORES_ONLY=1
    GB_HUGEPAGES=1
    GB_OLLAMA_CTX=131072
    info "Owner-workstation preset applied (i9-14900KF | RTX 5070 12GB | 64GB DDR4-3600 | PCIe 4 x16 | 4TB NVMe, 128 GB swap)"
}

# ---- Helpers -----------------------------------------------------------

need_root() {
    [[ $EUID -eq 0 ]] || die "Root required. Use: sudo $0 $1"
}

check_deps() {
    info "Checking build prerequisites..."
    # Provide distro-specific install hints for missing tools
    local _os_id_hint; _os_id_hint=$(grep -oP '^ID=\K.*' /etc/os-release 2>/dev/null | tr -d '"')
    local _make_hint _gcc_hint _hdr_hint
    if [[ -f /etc/arch-release ]] || printf '%s' "$_os_id_hint" | grep -qiE '^(arch|manjaro|endeavouros|cachyos|garuda)$'; then
        _make_hint="pacman -S base-devel"; _gcc_hint="pacman -S gcc"
        _hdr_hint="pacman -S linux-headers (or linux-zen-headers / linux-lts-headers)"
    elif printf '%s' "$_os_id_hint" | grep -qiE '^(fedora|rhel|centos|rocky|almalinux|ol)$'; then
        _make_hint="dnf install make gcc kernel-devel"; _gcc_hint="dnf install gcc"
        _hdr_hint="dnf install kernel-devel-$(uname -r)"
    else
        _make_hint="apt install build-essential"; _gcc_hint="apt install gcc"
        _hdr_hint="apt install linux-headers-$(uname -r)"
    fi
    command -v make >/dev/null || die "make not found — install with: sudo ${_make_hint}"
    command -v gcc  >/dev/null || die "gcc not found — install with: sudo ${_gcc_hint}"

    local kdir="/lib/modules/$(uname -r)/build"
    [[ -d "$kdir" ]] || die "Kernel headers not found at $kdir
    Install with: sudo ${_hdr_hint}"
    info "Kernel headers : $kdir  ✓"

    if lsmod | grep -q "^nvidia "; then
        info "NVIDIA driver  : loaded  ✓"
    else
        warn "NVIDIA driver not loaded — run: sudo modprobe nvidia"
    fi

    if lsmod | grep -q "^nvidia_uvm "; then
        info "NVIDIA UVM     : loaded  ✓  (managed memory / system RAM overflow ready)"
    else
        warn "nvidia_uvm not loaded — CUDA UVM overflow unavailable"
        warn "Fix: sudo modprobe nvidia_uvm"
    fi
}

# ---- Commands ----------------------------------------------------------

# ---------------------------------------------------------------------------
# cmd_install_recovery — install greenboost-recovery + greenboost-sentinel
#   systemd services and the /usr/local/sbin/greenboost-recover script.
#
# greenboost-sentinel.service  writes a sentinel file when GreenBoost is
#   running and renames it on clean shutdown via ExecStop.  If ExecStop never
#   runs (hard reset, kernel panic), the sentinel persists on the next boot
#   and is detected by greenboost-recovery.service.
#
# greenboost-recovery.service  runs early in boot (before ollama.service)
#   and repairs any damage left by a dirty shutdown: verifies the kernel
#   module is loaded, re-activates the NVMe swap file if it is missing or
#   has a corrupt header, ensures /etc/fstab has the swap entry, and clears
#   the kernel module's OOM guard via GB_IOCTL_RESET.
# ---------------------------------------------------------------------------
cmd_install_recovery() {
    local STATE_DIR="/var/lib/greenboost"
    mkdir -p "$STATE_DIR"

    # ── 1. Sentinel service ─────────────────────────────────────────────────
    cat > /etc/systemd/system/greenboost-sentinel.service << 'SENTINELEOF'
[Unit]
Description=GreenBoost dirty-shutdown sentinel
Documentation=man:greenboost_setup.sh(1)
DefaultDependencies=no
After=greenboost-recovery.service
Before=ollama.service shutdown.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStartPre=/bin/mkdir -p /var/lib/greenboost
ExecStart=/bin/touch /var/lib/greenboost/running
ExecStop=/bin/mv -f /var/lib/greenboost/running /var/lib/greenboost/sentinel

[Install]
WantedBy=multi-user.target
SENTINELEOF

    # ── 2. Recovery service ─────────────────────────────────────────────────
    cat > /etc/systemd/system/greenboost-recovery.service << 'RECOVERYEOF'
[Unit]
Description=GreenBoost crash recovery and pre-flight check
Documentation=man:greenboost_setup.sh(1)
DefaultDependencies=no
After=local-fs.target systemd-modules-load.service
Before=ollama.service multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/greenboost-recover
StandardOutput=journal
StandardError=journal
TimeoutStartSec=120
Restart=no

[Install]
WantedBy=multi-user.target
RequiredBy=ollama.service
RECOVERYEOF

    # ── 3. Recovery script ──────────────────────────────────────────────────
    # Embedded color helpers (subset of greenboost_setup.sh, no dependency).
    # Uses the same brand palette: Violet/Lime/Cyan/Amber and braille spinner.
    cat > /usr/local/sbin/greenboost-recover << 'RECOVEREOF'
#!/usr/bin/env bash
# greenboost-recover — pre-boot crash recovery for GreenBoost
# Called by greenboost-recovery.service before ollama.service starts.
# Safe to run manually: sudo greenboost-recover
set -euo pipefail

# ── Minimal color helpers ────────────────────────────────────────────────────
C_LIME=$'\033[38;2;230;255;60m'
C_RED=$'\033[38;2;220;50;47m'
C_AMBER=$'\033[38;2;255;191;0m'
C_CYAN=$'\033[38;2;48;200;255m'
C_VIOLET=$'\033[38;2;108;113;196m'
C_RESET=$'\033[0m'
C_BOLD=$'\033[1m'

_ok()   { echo -e "  ${C_LIME}✓${C_RESET} $*"; }
_fail() { echo -e "  ${C_RED}✗${C_RESET} $*" >&2; }
_warn() { echo -e "  ${C_AMBER}⚠${C_RESET} $*"; }
_info() { echo -e "  ${C_VIOLET}◈${C_RESET} $*"; }
_step() { local n=$1 t=$2; shift 2; echo -e "\n${C_CYAN}${C_BOLD}[$n/$t]${C_RESET} $*"; }

STATE_DIR="/var/lib/greenboost"
DRIVER_NAME="greenboost"
DEV_NODE="/dev/greenboost"
# GB_IOCTL_RESET = _IO('G', 3) = 0x4703
IOCTL_RESET="0x4703"

_header() {
    echo -e "\n${C_VIOLET}${C_BOLD}  GreenBoost Recovery${C_RESET}  ${C_LIME}v2.7${C_RESET}\n"
}

_header

# ── 0/3 Dirty-shutdown detection ────────────────────────────────────────────
_step 0 3 "Checking shutdown state..."
_dirty=0
if [[ -f "$STATE_DIR/sentinel" || -f "$STATE_DIR/running" ]]; then
    _dirty=1
    _warn "Dirty shutdown detected — running full recovery sequence"
else
    _info "No dirty-shutdown sentinel — last boot was clean"
fi

# ── 1/3 Kernel module ───────────────────────────────────────────────────────
_step 1 3 "Verifying kernel module..."
if lsmod | grep -q "^${DRIVER_NAME} "; then
    _ok "greenboost.ko loaded"
else
    _warn "Module not loaded — attempting modprobe..."
    if modprobe "$DRIVER_NAME" 2>/dev/null; then
        _ok "greenboost.ko loaded via modprobe"
    else
        _fail "modprobe greenboost failed — DKMS build may be incomplete"
        _fail "Run: sudo ./greenboost_setup.sh build && sudo make load"
        exit 1
    fi
fi

# ── 2/3 OOM guard reset ─────────────────────────────────────────────────────
_step 2 3 "Clearing kernel OOM guard (GB_IOCTL_RESET)..."
if [[ -c "$DEV_NODE" ]]; then
    if python3 - <<PYEOF 2>/dev/null
import fcntl, os, sys
try:
    fd = os.open("$DEV_NODE", os.O_RDWR)
    fcntl.ioctl(fd, $IOCTL_RESET)
    os.close(fd)
    sys.exit(0)
except Exception as e:
    print(f"  ioctl error: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
    then
        _ok "OOM guard cleared"
    else
        _warn "GB_IOCTL_RESET failed — module may still be initializing; non-fatal"
    fi
else
    _warn "$DEV_NODE not present — skipping OOM reset"
fi

# ── 3/3 Sentinel cleanup ────────────────────────────────────────────────────
_step 3 3 "Cleaning up sentinel files..."
rm -f "$STATE_DIR/sentinel" "$STATE_DIR/running"
mkdir -p "$STATE_DIR"
date -Iseconds > "$STATE_DIR/last_clean_boot"
_ok "Recovery complete — sentinel cleared"
echo ""
RECOVEREOF

    chmod 755 /usr/local/sbin/greenboost-recover

    # ── 4. Physical VRAM pressure watchdog ─────────────────────────────────
    # Polls nvidia-smi every 3 s and logs when non-GreenBoost GPU processes
    # (browser NVDEC, desktop compositor, etc.) consume enough physical VRAM
    # to crowd out the T1 pool.  This is what caused the 2026-03-30 crash:
    # browser hardware video decode + Ollama both competing for the same 12 GB
    # of physical VRAM with only 1 GB of headroom configured.
    #
    # The watchdog does NOT modify allocations — it only logs and writes a
    # pressure file so 'greenboost status' can surface it.  All thresholds are auto-detected fractions of physical
    # VRAM — no hard-coded values.
    cat > /usr/local/sbin/greenboost-vram-watchdog << 'VWEOF'
#!/usr/bin/env bash
# greenboost-vram-watchdog — physical VRAM pressure monitor
# Detects when non-GreenBoost GPU consumers (browser NVDEC, compositor, etc.)
# reduce the headroom available for the T1 pool.
set -euo pipefail

STATE_DIR="/var/lib/greenboost"
PRESSURE_FILE="$STATE_DIR/vram_pressure"
POLL_INTERVAL=3   # seconds between nvidia-smi polls

# Thresholds expressed as percentages of physical VRAM — hardware-agnostic.
# WARN fires when non-GreenBoost GPU usage > WARN_PCT% of physical VRAM.
# CRITICAL fires when free VRAM < CRIT_FREE_PCT% of physical VRAM.
WARN_PCT="${GREENBOOST_VRAM_WATCHDOG_WARN_PCT:-10}"      # 10% = 1.2 GB on 12 GB GPU
CRIT_FREE_PCT="${GREENBOOST_VRAM_WATCHDOG_CRIT_PCT:-8}"  # 8%  = 0.96 GB on 12 GB GPU

mkdir -p "$STATE_DIR"

# Detect total physical VRAM once at startup.
_total_mib=0
if command -v nvidia-smi &>/dev/null; then
    _total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheaders,nounits \
                 2>/dev/null | head -1 | tr -d ' ') || _total_mib=0
fi
if [[ "$_total_mib" -le 0 ]]; then
    echo "greenboost-vram-watchdog: nvidia-smi unavailable — exiting" >&2
    exit 0
fi

_warn_mib=$(( _total_mib * WARN_PCT / 100 ))
_crit_free_mib=$(( _total_mib * CRIT_FREE_PCT / 100 ))

echo "greenboost-vram-watchdog: started — GPU ${_total_mib} MiB, warn threshold ${_warn_mib} MiB non-GB usage, crit free ${_crit_free_mib} MiB"

_prev_state="ok"

while true; do
    sleep "$POLL_INTERVAL"

    # Read current VRAM used + free from nvidia-smi
    _line=$(nvidia-smi --query-gpu=memory.used,memory.free \
            --format=csv,noheaders,nounits 2>/dev/null | head -1) || continue
    _used_mib=$(echo "$_line" | awk -F',' '{gsub(/ /,"",$1); print $1}')
    _free_mib=$(echo "$_line" | awk -F',' '{gsub(/ /,"",$2); print $2}')
    [[ -z "$_used_mib" || -z "$_free_mib" ]] && continue

    # Approximate GreenBoost T1 allocation from sysfs (bytes → MiB)
    _gb_t1_mib=0
    if [[ -f /sys/class/greenboost/greenboost/status ]]; then
        _t1_bytes=$(grep -oP 't1_used_bytes:\s*\K\d+' \
                    /sys/class/greenboost/greenboost/status 2>/dev/null || echo 0)
        _gb_t1_mib=$(( _t1_bytes / 1048576 ))
    fi

    # Non-GreenBoost GPU usage = total used − what GreenBoost placed in T1
    _non_gb_mib=$(( _used_mib - _gb_t1_mib ))
    [[ "$_non_gb_mib" -lt 0 ]] && _non_gb_mib=0

    _state="ok"
    if [[ "$_free_mib" -lt "$_crit_free_mib" ]]; then
        _state="critical"
    elif [[ "$_non_gb_mib" -gt "$_warn_mib" ]]; then
        _state="warn"
    fi

    if [[ "$_state" != "$_prev_state" ]]; then
        case "$_state" in
            critical)
                logger -t greenboost-vram-watchdog -p kern.crit \
                    "CRITICAL: free VRAM ${_free_mib} MiB < threshold ${_crit_free_mib} MiB — non-GreenBoost consumers: ${_non_gb_mib} MiB (browser NVDEC / compositor?). Consider closing GPU-heavy apps or increasing GREENBOOST_VRAM_HEADROOM_MB."
                ;;
            warn)
                logger -t greenboost-vram-watchdog -p kern.warning \
                    "WARN: non-GreenBoost GPU usage ${_non_gb_mib} MiB > ${_warn_mib} MiB threshold — free VRAM ${_free_mib} MiB. Running browser hardware video decode alongside inference reduces T1 headroom."
                ;;
            ok)
                logger -t greenboost-vram-watchdog -p kern.info \
                    "OK: physical VRAM pressure cleared — used ${_used_mib} MiB, free ${_free_mib} MiB"
                ;;
        esac
        _prev_state="$_state"
    fi

    # Write pressure file for 'greenboost status' to read
    printf 'state=%s\nused_mib=%s\nfree_mib=%s\nnon_gb_mib=%s\ntotal_mib=%s\ntimestamp=%s\n' \
        "$_state" "$_used_mib" "$_free_mib" "$_non_gb_mib" "$_total_mib" \
        "$(date -Iseconds)" > "$PRESSURE_FILE"
done
VWEOF
    chmod 755 /usr/local/sbin/greenboost-vram-watchdog

    cat > /etc/systemd/system/greenboost-vram-watchdog.service << 'VWSEOF'
[Unit]
Description=GreenBoost physical VRAM pressure watchdog
Documentation=man:greenboost_setup.sh(1)
After=nvidia-persistenced.service ollama.service
PartOf=ollama.service

[Service]
Type=simple
ExecStart=/usr/local/sbin/greenboost-vram-watchdog
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
VWSEOF

    # ── Enable services ─────────────────────────────────────────────────────
    systemctl daemon-reload
    systemctl enable greenboost-recovery.service \
                     greenboost-sentinel.service \
                     greenboost-vram-watchdog.service 2>/dev/null || true
    gb_ok "greenboost-recovery + greenboost-sentinel + greenboost-vram-watchdog installed and enabled"
}

cmd_install_sys_configs() {
    need_root install-sys-configs
    detect_hardware

    info "Installing GreenBoost v2.8.2 system configuration files..."

    # 1. Ollama service — inject GreenBoost env vars + LD_PRELOAD (always refresh)
    local svc="/etc/systemd/system/ollama.service"
    if [[ -f "$svc" ]]; then
        # Remove any previously injected GreenBoost lines first (idempotent upgrade)
        sed -i '/OLLAMA_FLASH_ATTENTION/d
/OLLAMA_KV_CACHE_TYPE/d
/OLLAMA_NUM_CTX/d
/OLLAMA_MAX_LOADED_MODELS/d
/OLLAMA_KEEP_ALIVE/d
/OLLAMA_NUM_GPU/d
/GREENBOOST_/d
/libgreenboost/d' "$svc"
        # Inject fresh v2.7 env vars
        sed -i "/^\[Service\]/a Environment=\"OLLAMA_FLASH_ATTENTION=1\"\nEnvironment=\"OLLAMA_KV_CACHE_TYPE=q8_0\"\nEnvironment=\"OLLAMA_NUM_CTX=${GB_OLLAMA_CTX}\"\nEnvironment=\"OLLAMA_MAX_LOADED_MODELS=1\"\nEnvironment=\"OLLAMA_KEEP_ALIVE=-1\"\nEnvironment=\"OLLAMA_NUM_GPU=999\"\nEnvironment=\"GREENBOOST_VIRTUAL_VRAM_MB=$((GB_VIRT * 1024))\"\nEnvironment=\"GREENBOOST_DEBUG=0\"\nEnvironment=\"GREENBOOST_ACTIVE=1\"\nEnvironment=\"LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so\"" "$svc"
        systemctl daemon-reload
        info "Ollama service: GreenBoost v2.8.2 env vars injected (refreshed)"
        gb_ok "Ollama context cap set to ${GB_OLLAMA_CTX} tokens (T1: ${GB_PHYS} GB, T2: ${GB_VIRT} GB)"
    else
        warn "Ollama service not found at $svc — skipping"
    fi

    # 1b. Drop-in override — write 99-greenboost.conf so GreenBoost vars always win
    # over any third-party drop-ins (boost.conf, override.conf, etc.) that may set
    # conflicting values for OLLAMA_NUM_CTX, LD_PRELOAD, or OLLAMA_GPU_OVERHEAD.
    local dropin_dir="/etc/systemd/system/ollama.service.d"
    mkdir -p "$dropin_dir"

    # Remove conflicting entries from other drop-in files before writing 99-greenboost.conf
    local gb_vars=(OLLAMA_NUM_CTX LD_PRELOAD OLLAMA_GPU_OVERHEAD OLLAMA_FLASH_ATTENTION
                   OLLAMA_KV_CACHE_TYPE OLLAMA_MAX_LOADED_MODELS OLLAMA_KEEP_ALIVE OLLAMA_NUM_GPU)
    local _conflict_found=0
    for _f in "$dropin_dir"/*.conf; do
        [[ -f "$_f" ]] || continue
        [[ "$(basename "$_f")" == "99-greenboost.conf" ]] && continue
        for _var in "${gb_vars[@]}"; do
            if grep -q "$_var" "$_f" 2>/dev/null; then
                sed -i "/[[:space:]]*Environment.*${_var}=/d" "$_f"
                gb_info "Removed conflicting ${_var} from $(basename "$_f")"
                _conflict_found=1
            fi
        done
    done

    # Write 99-greenboost.conf — alphabetically last, so it always wins
    cat > "$dropin_dir/99-greenboost.conf" << DROPINEOF
# GreenBoost v2.8.2 — managed file, do not edit manually
# Re-generated by: sudo ./greenboost_setup.sh install-sys-configs
[Unit]
After=greenboost-recovery.service greenboost-sentinel.service
Requires=greenboost-recovery.service

[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_NUM_CTX=${GB_OLLAMA_CTX}"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_NUM_GPU=999"
Environment="GREENBOOST_VIRTUAL_VRAM_MB=$((GB_VIRT * 1024))"
Environment="GREENBOOST_DEBUG=0"
Environment="GREENBOOST_ACTIVE=1"
Environment="LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so"
DROPINEOF
    systemctl daemon-reload
    if [[ $_conflict_found -eq 1 ]]; then
        gb_ok "99-greenboost.conf written — conflicting entries removed from other drop-ins"
    else
        gb_ok "99-greenboost.conf written (no conflicts detected)"
    fi

    # 1c. Other known inference tool services — inject LD_PRELOAD + GREENBOOST_ACTIVE
    # if the service file exists (idempotent: skip if already patched).
    local shim_path="$SHIM_DEST/$SHIM_LIB"
    local _reloaded=0
    for svc in /etc/systemd/system/vllm.service \
               /etc/systemd/system/text-generation-inference.service; do
        [[ -f "$svc" ]] || continue
        sed -i '/GREENBOOST_/d
/libgreenboost/d' "$svc"
        sed -i "/^\[Service\]/a Environment=\"GREENBOOST_ACTIVE=1\"\nEnvironment=\"LD_PRELOAD=${shim_path}\"" "$svc"
        _reloaded=1
        info "$(basename "$svc"): GreenBoost env vars injected"
    done
    [[ $_reloaded -eq 1 ]] && systemctl daemon-reload

    # 2a. GreenBoost device udev rule — allow video group (includes ollama) access
    cat > /etc/udev/rules.d/99-greenboost.rules << 'UDEVEOF'
# GreenBoost kernel module — allow video group (includes ollama) to access /dev/greenboost
KERNEL=="greenboost", GROUP="video", MODE="0660"
UDEVEOF
    info "GreenBoost device udev rule installed: /etc/udev/rules.d/99-greenboost.rules"
    # Apply rule immediately to existing /dev/greenboost (if module already loaded)
    udevadm control --reload-rules 2>/dev/null || true
    udevadm trigger --name-match=greenboost 2>/dev/null \
        || udevadm trigger --subsystem-match=greenboost 2>/dev/null \
        || true
    udevadm settle 2>/dev/null || true

    # 2b. NVMe udev rule — scheduler=none, read_ahead=4096, nr_requests=1023
    # ENV{DEVTYPE}=="disk" restricts rules to whole-disk nodes only (nvme0n1, nvme1n1 …),
    # excluding partition nodes (nvme0n1p1 …) which have no queue/ sysfs directory.
    # nr_requests capped at 1023 — Samsung 990 EVO Plus hardware limit (max_hw_sectors_kb=512).
    cat > /etc/udev/rules.d/99-nvme-greenboost.rules << 'UDEVEOF'
# GreenBoost v2.8.2 — NVMe tuning for T3 swap performance
ACTION=="add|change", KERNEL=="nvme[0-9]n[0-9]", ENV{DEVTYPE}=="disk", ATTR{queue/scheduler}="none"
ACTION=="add|change", KERNEL=="nvme[0-9]n[0-9]", ENV{DEVTYPE}=="disk", ATTR{queue/read_ahead_kb}="4096"
ACTION=="add|change", KERNEL=="nvme[0-9]n[0-9]", ENV{DEVTYPE}=="disk", ATTR{queue/nr_requests}="1023"
UDEVEOF
    udevadm control --reload-rules && udevadm trigger || true
    info "NVMe udev rule installed: /etc/udev/rules.d/99-nvme-greenboost.rules"

    # 3. CPU governor service — P-cores only (E-cores stay on powersave)
    cat > /etc/systemd/system/cpu-perf.service << CPUEOF
[Unit]
Description=GreenBoost CPU performance governor (P-cores 0-${GB_PCORES_MAX})
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c 'for cpu in \$(seq 0 ${GB_PCORES_MAX}); do echo performance > /sys/devices/system/cpu/cpu\${cpu}/cpufreq/scaling_governor; done; cset set -c ${GB_GOLDEN_MIN}-${GB_GOLDEN_MAX} -s inference_set || true; cset proc -m -f root -t inference_set -k || true'
ExecStop=/bin/bash  -c 'for cpu in \$(seq 0 ${GB_PCORES_MAX}); do echo powersave  > /sys/devices/system/cpu/cpu\${cpu}/cpufreq/scaling_governor; done; cset set -d inference_set || true'

[Install]
WantedBy=multi-user.target
CPUEOF
    systemctl daemon-reload
    systemctl enable --now cpu-perf.service
    info "CPU governor service installed and started (includes cset for golden cores)"

    # 4. THP sysfs.d — transparent hugepages for compaction + THP performance
    # NOTE: gb_alloc_buf() uses alloc_pages(GFP_KERNEL|__GFP_COMP, order=9) which draws
    # from the BUDDY ALLOCATOR, NOT the HugeTLB pool.  Pre-allocating HugeTLB pages
    # (vm.nr_hugepages=26112) locks 51 GB in the HugeTLB pool, leaving <12 GB free RAM,
    # which triggers the OOM guard and makes T2 unavailable.  Keep nr_hugepages=0.
    mkdir -p /etc/sysfs.d
    cat > /etc/sysfs.d/greenboost-hugepages.conf << 'HPEOF'
# GreenBoost v2.8.2 — THP config (no HugeTLB pre-allocation: gb_alloc_buf uses buddy allocator)
kernel/mm/transparent_hugepage/enabled = always
HPEOF
    info "THP sysfs conf: /etc/sysfs.d/greenboost-hugepages.conf"

    # Release any previously locked HugeTLB pages and free them back to buddy allocator
    if [[ "$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages 2>/dev/null)" != "0" ]]; then
        echo 0 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages 2>/dev/null \
            && info "HugeTLB pages released: freed $(( $(cat /proc/meminfo | grep HugePages_Total | awk '{print $2}') * 2 )) kB back to buddy allocator" \
            || warn "Could not set nr_hugepages=0 — reboot to apply"
    else
        info "HugeTLB: already 0 (correct — GreenBoost T2 uses buddy allocator)"
    fi

    # 5. VM sysctl — handled by cmd_tune_sysctl (99-zzz-greenboost.conf)
    # The definitive sysctl file is written by tune-sysctl/tune-all and uses
    # the 99-zzz- prefix so it sorts last and wins over any other conf.
    # No separate 99-greenboost.conf sysctl block here to avoid conflicting values.

    # 6. Install LD_AUDIT library + AppArmor abstraction
    # The audit library (< 5 KB, no CUDA code) sits in /etc/ld.so.preload and
    # injects the full CUDA shim ONLY into processes that load libcuda.so or
    # libcudart.so.  PAM helpers, GDM, snap-confine, and every other non-CUDA
    # process are never touched — AppArmor blast radius is one r/mr permission
    # for the tiny audit stub, not the full shim.
    local audit_src="$MODULE_DIR/$AUDIT_LIB"
    local audit_dest="$SHIM_DEST/$AUDIT_LIB"
    if [[ -f "$audit_src" ]]; then
        cp "$audit_src" "$audit_dest"
        ldconfig 2>/dev/null || true
        info "LD_AUDIT library installed: $audit_dest"
    else
        warn "LD_AUDIT library not found at $audit_src — run 'make audit' first"
    fi

    # AppArmor abstraction — allows confined profiles to open the audit stub
    # (and the full shim, but only in CUDA processes).
    #
    # What this does (in plain language):
    #   Linux's security sandbox (AppArmor) controls which files each program can open.
    #   We're telling it: "it's OK for GPU apps like Ollama to load the GreenBoost library."
    #   Without this step, AppArmor would silently block the shim in sandboxed processes
    #   and the GPU memory expansion would do nothing.
    #
    # Strategy (two layers for complete coverage):
    #   A) Inject into abstractions/base  — any profile that includes <abstractions/base>
    #      (the vast majority) automatically inherits the permission.  New profiles
    #      installed by apt/snap are covered without re-running setup.
    #   B) Dynamic scan of all profiles in /etc/apparmor.d/ — catches profiles that
    #      don't use <abstractions/base> (e.g. ubuntu_pro_*, wsdd, lsblk, lsusb).
    #      snap-confine is skipped: snapd overwrites its profile on every update.
    local aa_dir="/etc/apparmor.d/abstractions"
    local aa_src="$MODULE_DIR/apparmor/abstractions/greenboost-audit"
    local aa_dest="$aa_dir/greenboost-audit"
    if [[ -d "$aa_dir" && -f "$aa_src" ]]; then
        echo -e ""
        echo -e "  ${C_VIOLET}◈${C_RESET}  ${C_GRAY}${C_BOLD}Security sandbox (AppArmor)${C_RESET}"
        echo -e "  ${C_DIM}  Granting GPU apps permission to use GreenBoost.${C_RESET}"
        echo -e "  ${C_DIM}  This is safe — we only add the minimum read permission needed.${C_RESET}"
        echo -e ""

        cp "$aa_src" "$aa_dest"
        gb_ok "AppArmor abstraction installed"

        # Layer A — inject into abstractions/base for global coverage
        local base_abs="/etc/apparmor.d/abstractions/base"
        if [[ -f "$base_abs" ]] && ! grep -q "greenboost-audit" "$base_abs"; then
            sed -i '/^  abi <abi\//a\  #include <abstractions\/greenboost-audit>' "$base_abs"
            gb_ok "Added to abstractions/base — all new apps auto-covered"
        else
            gb_info "abstractions/base already includes greenboost-audit (skip)"
        fi

        # Layer B — dynamic scan: patch profiles that don't inherit from base.
        # Skip: sub-directories, abstractions, tunables, disable, force-complain,
        #        local overrides, abi files, ldd (inlines base), and snap-confine
        #        (snapd manages and overwrites it on every snapd upgrade).
        #
        # Collect profile list first so we can show N/total progress.
        local _profiles=()
        while IFS= read -r _pf; do
            _profiles+=("$_pf")
        done < <(find /etc/apparmor.d/ -maxdepth 1 -type f \
            | grep -vE '/(abstractions|tunables|disable|force-complain|local|abi|ldd)$' \
            | sort)

        local _total="${#_profiles[@]}"
        local _patched=0 _idx=0
        for profile in "${_profiles[@]}"; do
            (( _idx++ )) || true
            local bname
            bname=$(basename "$profile")
            [[ "$bname" == "usr.lib.snapd.snap-confine.real" ]] && continue
            [[ -d "$profile" ]] && continue
            grep -q "greenboost-audit" "$profile" 2>/dev/null && continue
            grep -qE '^\s*profile\s|^\S.*\{$' "$profile" 2>/dev/null || continue
            # Show progress bar: [N/M] ████████░░░ pct%
            local _pct=$(( _idx * 100 / _total ))
            local _filled=$(( _pct / 2 )) _empty=$(( 50 - _pct / 2 ))
            printf "\r  ${C_GRAY}[%d/%d]${C_RESET} ${C_LIME}%s${C_GRAY}%s${C_RESET} %3d%%  ${C_DIM}%-30s${C_RESET}" \
                "$_idx" "$_total" \
                "$(printf '█%.0s' $(seq 1 $_filled 2>/dev/null || true))" \
                "$(printf '░%.0s' $(seq 1 $_empty  2>/dev/null || true))" \
                "$_pct" "$bname"
            sed -i '/^}$/i\  #include <abstractions\/greenboost-audit>' "$profile"
            (( _patched++ )) || true
        done
        # Clear progress line
        printf "\r%*s\r" "$(tput cols 2>/dev/null || echo 80)" ""

        if [[ $_patched -gt 0 ]]; then
            gb_ok "Patched $_patched AppArmor profiles"
        else
            gb_info "All AppArmor profiles already patched (skip)"
        fi

        # Reload AppArmor with spinner
        apparmor_parser -r /etc/apparmor.d/ 2>/dev/null &
        gb_spin $! "Reloading AppArmor rules..."
    else
        [[ -d "$aa_dir" ]] || gb_info "AppArmor not active on this system — skipping"
    fi

    # 7b. Gaming PAM limits — allow video group to nice -5 (game process elevation)
    # Written here so the Proton wrapper's os.setpriority(PRIO_PROCESS, 0, -5) succeeds
    # without requiring the game to run as root.  Users in the video group (which GPU
    # users already are for /dev/greenboost access) inherit this limit automatically.
    mkdir -p /etc/security/limits.d
    cat > /etc/security/limits.d/99-greenboost-gaming.conf << 'LIMITSEOF'
# GreenBoost gaming process priority — allow video group members to elevate games
# to nice -5 so the game process runs above nice-0 background tasks.
# The GreenBoost Proton wrapper calls os.setpriority(PRIO_PROCESS, 0, -5) which
# requires the effective nice ceiling set here.
@video  hard  nice  -5
@video  soft  nice  -5
LIMITSEOF
    gb_ok "Gaming PAM limits: /etc/security/limits.d/99-greenboost-gaming.conf"

    # 8. Gaming / Vulkan shader boost service
    cmd_install_shader_boost

    # 8b. Idle memory reclaim daemon
    cmd_install_idle_reclaim

    # 9. Vulkan implicit layer (VK_LAYER_GREENBOOST_memory)
    cmd_install_vulkan_layer

    # 10. Crash recovery services (greenboost-recovery + greenboost-sentinel)
    cmd_install_recovery

    # 11. TurboQuant KV cache compression daemon
    local _tq_src="$MODULE_DIR/cmd/greenboost-turboquant"
    if [[ -d "$_tq_src" ]]; then
        local _tq_dest="/usr/local/lib/greenboost/cmd/greenboost-turboquant"
        mkdir -p "$_tq_dest"
        cp "$_tq_src/"*.py "$_tq_dest/" 2>/dev/null || true
        install -m 755 "$_tq_src/greenboost-turboquant" /usr/local/bin/greenboost-turboquant
        install -m 644 "$_tq_src/greenboost-turboquant.service" /etc/systemd/system/greenboost-turboquant.service
        systemctl daemon-reload
        systemctl enable greenboost-turboquant.service 2>/dev/null || true
        if lsmod | grep -q "^greenboost "; then
            systemctl restart greenboost-turboquant.service 2>/dev/null || true
        fi
        gb_ok "TurboQuant daemon installed + enabled"
    else
        gb_info "TurboQuant daemon source not found at $_tq_src — skipping"
    fi

    echo ""
    info "System config installation complete."
    warn "Restart Ollama to pick up new env vars: sudo systemctl restart ollama"
}

cmd_install_shader_boost() {
    # Gaming / Vulkan Shader Compilation Boost
    # ─────────────────────────────────────────
    # Steam launches fossilize_replay (Proton VKD3D pipeline pre-compilation)
    # at nice 10 (low priority), causing 168k-shader precompiles to take 60+ minutes.
    # This daemon detects fossilize_replay worker processes and boosts them:
    #   renice -5   → elevate above nice 0 background tasks
    #   ionice -c2 -n0 → best-effort I/O, highest level
    #   taskset 0-<pcores_max> → pin to P-cores (avoids E-core cache thrash)
    # Safe to run root: only affects fossilize_replay owned by any user.
    local boost_bin="/usr/local/bin/greenboost-shader-boost"
    local boost_svc="/etc/systemd/system/greenboost-shader-boost.service"

    cat > "$boost_bin" << 'SHADERBOOSTEOF'
#!/bin/bash
# GreenBoost Vulkan Shader Compilation Boost Daemon
# Monitors for fossilize_replay workers (Steam/Proton pipeline pre-compilation)
# and boosts them: renice -5, ionice best-effort/0, pin to P-cores.
# Managed by greenboost-shader-boost.service (runs as root).

# Detect highest P-core CPU number (Intel hybrid topology).
# Primary: thread_siblings_list — CPU with "N-M" range = P-core (HyperThreaded).
# Secondary: core_type integer 1=P when primary finds no P-cores.
# Falls back to half of total CPUs for AMD or uniform Intel systems.
pcores_max=0
total_cpus_boost=$(nproc)
for (( _i=0; _i < total_cpus_boost; _i++ )); do
    _sib=$(cat "/sys/devices/system/cpu/cpu${_i}/topology/thread_siblings_list" 2>/dev/null)
    [[ -z "$_sib" ]] && continue
    [[ "$_sib" == *"-"* ]] || continue
    (( _i > pcores_max )) && pcores_max=$_i
done
if (( pcores_max == 0 )); then
    # Try core_type integer fallback (1=P-core)
    for f in /sys/devices/system/cpu/cpu*/topology/core_type; do
        [[ -r "$f" ]] || continue
        [[ "$(< "$f")" == "1" ]] || continue
        n=${f##*cpu}; n=${n%%/*}
        (( n > pcores_max )) && pcores_max=$n
    done
fi
(( pcores_max == 0 )) && pcores_max=$(( total_cpus_boost / 2 - 1 ))
cpuset="0-${pcores_max}"

declare -A seen

while true; do
    new_pids=()
    while IFS= read -r pid; do
        [[ -z "$pid" || -n "${seen[$pid]}" ]] && continue
        seen[$pid]=1
        new_pids+=("$pid")
        renice    -n -5    -p "$pid" >/dev/null 2>&1
        ionice    -c 2 -n 0 -p "$pid" >/dev/null 2>&1
        taskset   -acp "$cpuset" "$pid" >/dev/null 2>&1
    done < <(pgrep -f fossilize_replay 2>/dev/null)

    if (( ${#new_pids[@]} > 0 )); then
        logger -t greenboost-shader-boost "Detected fossilize_replay — boosting PIDs: ${new_pids[*]}"
        logger -t greenboost-shader-boost "Applied: renice -5, ionice best-effort/0, taskset 0-${pcores_max}"
    fi

    # Prune dead PIDs from seen map
    for pid in "${!seen[@]}"; do
        [[ -d /proc/$pid ]] || unset "seen[$pid]"
    done

    sleep 1
done
SHADERBOOSTEOF
    chmod +x "$boost_bin"

    cat > "$boost_svc" << 'SHADERUNITEOF'
[Unit]
Description=GreenBoost Vulkan Shader Compilation Boost
Documentation=https://gitlab.com/IsolatedOctopi/greenboost
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/local/bin/greenboost-shader-boost
Restart=always
RestartSec=2
Nice=0

[Install]
WantedBy=multi-user.target
SHADERUNITEOF

    systemctl daemon-reload
    systemctl enable --now greenboost-shader-boost.service 2>/dev/null \
        && gb_ok "Gaming boost service installed and running (fossilize_replay → renice -5 + P-cores)" \
        || gb_warn_ui "Gaming boost service installed but failed to start — check: systemctl status greenboost-shader-boost"
}

cmd_install_idle_reclaim() {
    # GreenBoost Idle Memory Reclaim Daemon
    # ──────────────────────────────────────
    # Framework-agnostic: works for Ollama, vLLM, llama.cpp, ExLlamaV3, TGI,
    # or any CUDA app that uses GreenBoost T2/T3 memory.
    #
    # Why SIGTERM works universally:
    #   Every process that has /dev/greenboost open holds T2 DMA-BUF allocations.
    #   When that process exits (cleanly via SIGTERM or abruptly via SIGKILL),
    #   the kernel fires gb_close() → gb_release_pid_buffers() → all T2 DMA-BUF
    #   slots are freed automatically. The NVIDIA driver releases T1 VRAM once
    #   the CUDA context reference count drops to zero.
    #   Systemd-managed services (Restart=always) restart automatically on demand.
    #
    # Note: abrupt termination (user kills the process, crash, Ctrl+C) is ALREADY
    #   handled by the kernel — gb_close() fires regardless. This daemon only adds
    #   the idle case where the process is still running but not being used.
    local reclaim_bin="/usr/local/bin/greenboost-idle-reclaim"
    local reclaim_svc="/etc/systemd/system/greenboost-idle-reclaim.service"

    cat > "$reclaim_bin" << 'RECLAIMEOF'
#!/bin/bash
# GreenBoost Idle Memory Reclaim Daemon
#
# Framework-agnostic idle memory reclaim for any CUDA inference app.
#
# How it works:
#   1. Watches /run/greenboost/phase (written by CUDA shim every 30 s).
#   2. On DEEP_IDLE: confirms GPU compute utilization is 0% (avoids false triggers).
#   3. Finds every process with /dev/greenboost open via fuser — these are the
#      only processes with active GreenBoost T2/T3 allocations.
#   4. Sends SIGTERM. The process exits, kernel gb_close() fires automatically,
#      all T2 DMA-BUF released. NVIDIA driver frees T1 VRAM (KV cache).
#   Ollama special-case: prefer REST API unload (service stays alive + auto-reloads).
#
# Abrupt termination (user kills app, crash) is already handled by the kernel.
# This daemon only covers idle processes that are still running but not being used.

PHASE_FILE="/run/greenboost/phase"
GB_DEV="/dev/greenboost"
OLLAMA_BASE="${GREENBOOST_OLLAMA_URL:-http://127.0.0.1:11434}"
POLL_SEC=30
GPU_CONFIRM_COUNT=3   # consecutive 0% GPU polls required before acting

log() { logger -t greenboost-idle-reclaim "$*"; }

# Returns current GPU compute utilization percent (0-100), or 100 on error.
gpu_util_pct() {
    nvidia-smi --query-gpu=utilization.compute --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d ' ' || echo 100
}

# Returns PIDs of every process with /dev/greenboost open.
# These are the only processes that hold active GreenBoost T2/T3 allocations.
gb_active_pids() {
    fuser "$GB_DEV" 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' || true
}

# Ollama-preferred path: REST API unload keeps the service alive and auto-reloads.
ollama_unload_all() {
    local models
    models=$(curl -sf --max-time 5 "${OLLAMA_BASE}/api/ps" 2>/dev/null \
        | grep -oP '"name"\s*:\s*"\K[^"]+') || return 1
    [[ -z "$models" ]] && return 0
    while IFS= read -r model; do
        [[ -z "$model" ]] && continue
        local code
        code=$(curl -sf --max-time 10 -o /dev/null -w "%{http_code}" \
            -X DELETE "${OLLAMA_BASE}/api/delete" \
            -H "Content-Type: application/json" \
            -d "{\"name\":\"${model}\",\"keep_alive\":0}" 2>/dev/null)
        log "Ollama: unloaded '${model}' → HTTP ${code}"
    done <<< "$models"
    return 0
}

do_reclaim() {
    local idle_ms="$1"
    local pids
    pids=$(gb_active_pids)

    if [[ -z "$pids" ]]; then
        log "DEEP_IDLE (${idle_ms} ms): no active GreenBoost processes — memory already free"
        return
    fi

    log "DEEP_IDLE (${idle_ms} ms): reclaiming T1 VRAM + T2 RAM — PIDs: $(echo "$pids" | tr '\n' ' ')"

    for pid in $pids; do
        [[ -d /proc/$pid ]] || continue
        local comm
        comm=$(cat /proc/"$pid"/comm 2>/dev/null || echo unknown)

        # Ollama: prefer REST API — service stays alive and auto-reloads on next request
        if [[ "$comm" == "ollama"* ]]; then
            log "PID ${pid} (${comm}): using Ollama REST API for graceful unload"
            if ollama_unload_all; then
                continue
            fi
            log "PID ${pid} (${comm}): REST API failed — falling back to SIGTERM"
        fi

        # Universal path: SIGTERM → process exits → kernel gb_close() runs automatically
        # → gb_release_pid_buffers() frees all T2 DMA-BUF for this PID
        # → NVIDIA driver frees T1 VRAM when CUDA context refcount → 0
        # Systemd-managed processes (Restart=always) restart when next request arrives.
        if kill -TERM "$pid" 2>/dev/null; then
            log "PID ${pid} (${comm}): SIGTERM sent — kernel will free T2 on exit"
        else
            log "PID ${pid} (${comm}): already exited or no permission (OK)"
        fi
    done
}

last_phase=""
gpu_zero_streak=0

while true; do
    sleep "$POLL_SEC"

    [[ -f "$PHASE_FILE" ]] || { gpu_zero_streak=0; continue; }
    phase=$(grep -oP '(?<=phase=)\S+' "$PHASE_FILE" 2>/dev/null)
    [[ -z "$phase" ]] && continue

    case "$phase" in
        DEEP_IDLE)
            # Confirm GPU is genuinely idle (not just a momentary gap between tokens)
            gpu_util=$(gpu_util_pct)
            if [[ "$gpu_util" =~ ^[0-9]+$ && "$gpu_util" -eq 0 ]]; then
                (( gpu_zero_streak++ ))
            else
                gpu_zero_streak=0
            fi

            if [[ "$last_phase" != "DEEP_IDLE" && "$gpu_zero_streak" -ge "$GPU_CONFIRM_COUNT" ]]; then
                idle_ms=$(grep -oP '(?<=idle_ms=)\d+' "$PHASE_FILE" 2>/dev/null || echo 0)
                do_reclaim "$idle_ms"
                last_phase="DEEP_IDLE"
            fi
            ;;
        IDLE)
            gpu_zero_streak=0
            [[ "$last_phase" != "IDLE" ]] \
                && log "IDLE phase — monitoring (reclaim at DEEP_IDLE after GPU confirms 0%)"
            last_phase="IDLE"
            ;;
        MODEL_LOAD|INFERENCE|STEADY)
            gpu_zero_streak=0
            [[ "$last_phase" == "DEEP_IDLE" || "$last_phase" == "IDLE" ]] \
                && log "Activity resumed (phase=${phase}) — standing by"
            last_phase="$phase"
            ;;
        INIT)
            gpu_zero_streak=0
            last_phase="INIT"
            ;;
    esac
done
RECLAIMEOF
    chmod +x "$reclaim_bin"

    cat > "$reclaim_svc" << 'RECLAIMUNITEOF'
[Unit]
Description=GreenBoost Idle Memory Reclaim
Documentation=https://gitlab.com/IsolatedOctopi/greenboost
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/greenboost-idle-reclaim
Restart=always
RestartSec=5
Environment=GREENBOOST_OLLAMA_URL=http://127.0.0.1:11434

[Install]
WantedBy=multi-user.target
RECLAIMUNITEOF

    systemctl daemon-reload
    systemctl enable --now greenboost-idle-reclaim.service 2>/dev/null \
        && gb_ok "Idle reclaim daemon installed and running (T1+T2 freed after ~17 min idle, all frameworks)" \
        || gb_warn_ui "Idle reclaim daemon installed but failed to start — check: systemctl status greenboost-idle-reclaim"
}

# ── clean-memory ──────────────────────────────────────────────────────────────
# Force-release T1 VRAM + T2 RAM + T3 NVMe immediately.
# Framework-agnostic: finds every process with /dev/greenboost open via fuser,
# cmd_gaming_mode — stop/start Ollama to free T2 DDR before/after gaming.
# When OLLAMA_KEEP_ALIVE=-1 a loaded model stays pinned in T2 DDR indefinitely.
# Combined RAM pressure (model + game + OS) can push free RAM below safety_reserve_gb,
# triggering the OOM guard and crashing the game.  Stopping Ollama releases all T2 pages.
cmd_gaming_mode() {
    local action="${1:-}"
    case "$action" in
        enable|start)
            need_root gaming-mode
            gb_header
            gb_info "Gaming mode: suspending AI inference..."
            systemctl start greenboost-gaming.service
            gb_ok "Gaming mode active — Ollama suspended, T2 DDR freed for gaming"
            echo ""
            gb_info "GreenBoost Proton activates this automatically on game launch."
            gb_info "Run ${C_LIME}sudo greenboost gaming-mode disable${C_RESET} when done."
            ;;
        disable|stop)
            need_root gaming-mode
            gb_header
            gb_info "Gaming mode: restoring AI inference..."
            systemctl stop greenboost-gaming.service
            gb_ok "AI inference restored — Ollama running"
            ;;
        *)
            gb_header
            echo -e "  ${C_BOLD}Usage:${C_RESET}"
            echo -e "    ${C_LIME}sudo greenboost gaming-mode enable${C_RESET}   — suspend Ollama before gaming"
            echo -e "    ${C_LIME}sudo greenboost gaming-mode disable${C_RESET}  — restore Ollama after gaming"
            echo ""
            gb_info "GreenBoost Proton triggers this automatically on game launch."
            gb_info "When Ollama is suspended, pinned T2 DDR pages are freed, giving"
            gb_info "the game full access to system RAM without OOM guard pressure."
            ;;
    esac
}

# prefers Ollama REST API when Ollama is among them, sends SIGTERM to everything else.
# The kernel's gb_close() → gb_release_pid_buffers() frees T2 DMA-BUF on exit.
cmd_t3_memory() {
    # Usage: greenboost t3-memory <size>   e.g. 100GB, 50GB, 0 (disk-limited)
    local raw="${1:-}"
    if [[ -z "$raw" ]]; then
        printf "  ${C_AMBER}⚠${C_RESET}  Usage: greenboost t3-memory <size>  (e.g. 100GB, 50GB, 0 for unlimited)\n"
        return 1
    fi

    # Parse value — strip suffix, allow G/GB/g
    local cap_gb
    cap_gb=$(echo "$raw" | sed 's/[Gg][Bb]*$//' | tr -d ' ')
    if ! [[ "$cap_gb" =~ ^[0-9]+$ ]]; then
        die "Invalid size '$raw' — expected a number in GB (e.g. 100GB or 100)"
    fi
    local cap_mb=$(( cap_gb * 1024 ))

    if [[ ! -e /dev/greenboost ]]; then
        die "GreenBoost kernel module not loaded — run: sudo greenboost load"
    fi

    gb_header

    # Write T3 cap via IOCTL using Python helper (avoids C tool dependency)
    local prev_mb
    prev_mb=$(python3 - <<PYEOF
import fcntl, struct, os, sys
IOCTL_SET_T3_CAP = 0xc010471f   # _IOWR('G', 17, 2*u64) on x86-64
cap_mb = $cap_mb
try:
    fd = os.open("/dev/greenboost", os.O_RDWR)
    buf = struct.pack("QQ", cap_mb, 0)
    res = fcntl.ioctl(fd, IOCTL_SET_T3_CAP, bytearray(buf))
    _, prev = struct.unpack("QQ", res)
    os.close(fd)
    print(prev)
except Exception as e:
    print("ERR:" + str(e), file=sys.stderr)
    sys.exit(1)
PYEOF
) || die "Failed to set T3 cap — are you root? (run with sudo)"

    if [[ "$cap_gb" -eq 0 ]]; then
        gb_ok "T3 pool set to disk-limited (unlimited); was ${prev_mb} MB"
    else
        gb_ok "T3 pool set to ${cap_gb} GB (${cap_mb} MB); was ${prev_mb} MB"
    fi

    # Refresh virtual VRAM report in shim: update the systemd env var
    local svc_drop="/etc/systemd/system/ollama.service.d/99-greenboost.conf"
    if [[ -f "$svc_drop" ]] && grep -q "GREENBOOST_VIRTUAL_VRAM_MB" "$svc_drop" 2>/dev/null; then
        # Recompute: T2 (virtual_vram_gb from sysfs) + T3 (new cap)
        local t2_gb
        t2_gb=$(cat /sys/module/greenboost/parameters/virtual_vram_gb 2>/dev/null || echo 0)
        local total_virtual_mb=$(( (t2_gb + cap_gb) * 1024 ))
        sed -i "s/GREENBOOST_VIRTUAL_VRAM_MB=[0-9]*/GREENBOOST_VIRTUAL_VRAM_MB=${total_virtual_mb}/" "$svc_drop"
        gb_info "Updated GREENBOOST_VIRTUAL_VRAM_MB=${total_virtual_mb} MB in Ollama service drop-in"
        systemctl daemon-reload
        gb_info "Restart Ollama to apply: sudo systemctl restart ollama"
    fi

    # Show new T3 stats
    if [[ -r /sys/class/greenboost/greenboost/status ]]; then
        echo ""
        grep -E "T3|NVMe|t3|nvme" /sys/class/greenboost/greenboost/status 2>/dev/null | \
            sed "s/^/  ${C_CYAN}◈${C_RESET}  /" || true
    fi
}

cmd_clean_memory() {
    local ollama_base="${GREENBOOST_OLLAMA_URL:-http://127.0.0.1:11434}"

    gb_header
    gb_step "Scanning for active inference processes..."

    # Show memory state before
    local vram_before=""
    if command -v nvidia-smi &>/dev/null; then
        vram_before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")
        printf "  ${C_CYAN}◈${C_RESET}  T1 VRAM before : %s MiB used\n" "$vram_before"
    fi
    if [[ -r /sys/class/greenboost/greenboost/pool_brief ]]; then
        local brief
        brief=$(cat /sys/class/greenboost/greenboost/pool_brief 2>/dev/null)
        printf "  ${C_CYAN}◈${C_RESET}  GreenBoost pool : %s\n" "$brief"
    fi
    echo ""

    # Discover active inference processes via fuser /dev/greenboost
    local pids=""
    if [[ -e /dev/greenboost ]]; then
        pids=$(fuser /dev/greenboost 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' || true)
    fi

    if [[ -z "$pids" ]]; then
        gb_info "No active GreenBoost processes found"
        if [[ $(cat /sys/class/greenboost/greenboost/active_buffers 2>/dev/null) == "0" ]] \
            || [[ ! -e /dev/greenboost ]]; then
            gb_info "T2 pool is empty — memory already free"
            return 0
        fi
    fi

    echo ""
    gb_step "Releasing T1 VRAM + T2 RAM + T3 NVMe:"

    local reclaimed=0
    for pid in $pids; do
        [[ -d /proc/$pid ]] || continue
        local comm
        comm=$(cat /proc/"$pid"/comm 2>/dev/null || echo unknown)

        # Ollama: REST API unload — cleaner, service stays alive, auto-reloads on next request
        if [[ "$comm" == "ollama"* ]]; then
            local models
            models=$(curl -sf --max-time 5 "${ollama_base}/api/ps" 2>/dev/null \
                | grep -oP '"name"\s*:\s*"\K[^"]+') || true
            if [[ -n "$models" ]]; then
                while IFS= read -r model; do
                    [[ -z "$model" ]] && continue
                    local code
                    code=$(curl -sf --max-time 15 -o /dev/null -w "%{http_code}" \
                        -X DELETE "${ollama_base}/api/delete" \
                        -H "Content-Type: application/json" \
                        -d "{\"name\":\"${model}\",\"keep_alive\":0}" 2>/dev/null)
                    if [[ "$code" == "200" || "$code" == "204" || "$code" == "404" ]]; then
                        gb_ok "Ollama: unloaded '${model}' (service stays running)"
                        (( reclaimed++ ))
                    else
                        gb_warn_ui "Ollama: could not unload '${model}' (HTTP ${code})"
                    fi
                done <<< "$models"
                continue
            fi
        fi

        # Universal: SIGTERM → process exits → kernel frees all its T2 DMA-BUF automatically
        if kill -TERM "$pid" 2>/dev/null; then
            gb_ok "Stopped: ${comm} (PID ${pid}) — kernel freeing T2 DMA-BUF"
            (( reclaimed++ ))
        else
            gb_warn_ui "Could not signal PID ${pid} (${comm}) — may need sudo"
        fi
    done

    if [[ $reclaimed -eq 0 && -z "$pids" ]]; then
        gb_info "Nothing to reclaim"
        return 0
    fi

    # Brief pause for processes to exit and kernel to release DMA-BUF
    sleep 1

    # Show memory state after
    echo ""
    gb_step "Memory state after reclaim:"
    if command -v nvidia-smi &>/dev/null; then
        local vram_after
        vram_after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")
        local freed=""
        if [[ "$vram_before" =~ ^[0-9]+$ && "$vram_after" =~ ^[0-9]+$ ]]; then
            freed="  (freed $(( vram_before - vram_after )) MiB)"
        fi
        local vram_total
        vram_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")
        printf "  ${C_CYAN}◈${C_RESET}  T1 VRAM  : %s / %s MiB%s\n" "$vram_after" "$vram_total" "$freed"
    fi
    if [[ -r /sys/class/greenboost/greenboost/pool_brief ]]; then
        local brief
        brief=$(cat /sys/class/greenboost/greenboost/pool_brief 2>/dev/null)
        printf "  ${C_CYAN}◈${C_RESET}  GreenBoost pool : %s\n" "$brief"
    fi
    echo ""
    gb_ok "Memory reclaim complete — inference apps reload on next request"
}

cmd_install_vulkan_layer() {
    # VK_LAYER_GREENBOOST_memory — implicit Vulkan layer
    # ───────────────────────────────────────────────────
    # Inflates the device-local heap reported to Vulkan games to match the
    # virtual VRAM the CUDA shim reports (auto-detected from kernel module params),
    # so games choose maximum quality presets and never self-limit based on VRAM.
    # On VK_ERROR_OUT_OF_DEVICE_MEMORY (allocs ≥ 64 MB), attempts a fallback
    # through GreenBoost T2 DDR via DMA-BUF import (VK_KHR_external_memory_fd).
    #
    # Activated on demand: GREENBOOST_VULKAN=1 %command% in Steam launch options.
    # Zero cost for all other Vulkan applications when the env var is absent.

    local layer_src="$MODULE_DIR/$VULKAN_LAYER_LIB"
    local layer_dest="$SHIM_DEST/$VULKAN_LAYER_LIB"
    local manifest_src="$MODULE_DIR/$VULKAN_LAYER_MANIFEST"
    local manifest_dest="$VULKAN_IMPLICIT_LAYER_DIR/$VULKAN_LAYER_MANIFEST"

    if [[ ! -f "$layer_src" ]]; then
        gb_warn_ui "Vulkan layer not built — run 'make vulkan' first, then re-run install-sys-configs"
        return 0
    fi

    mkdir -p "$VULKAN_IMPLICIT_LAYER_DIR"
    cp "$layer_src" "$layer_dest"
    cp "$manifest_src" "$manifest_dest"
    ldconfig

    gb_ok "Vulkan layer installed — use GREENBOOST_VULKAN=1 %command% in Steam launch options"
}

cmd_fix_steam() {
    need_root fix-steam

    # ── 1. System-level: Steam launch wrapper (GTK2/XIM SIGBUS fix) ──────────
    local wrapper="/usr/local/bin/greenboost-steam"
    gb_step "Installing Steam launch wrapper → $wrapper"

    cat > "$wrapper" << 'STEAMEOF'
#!/usr/bin/env bash
# GreenBoost Steam wrapper — prevents SIGBUS from broken GTK2 theme + XIM failure.
# GTK2_RC_FILES=/dev/null  : bypasses Yaru-purple-dark (contains GTK3 syntax at main.rc:775)
# XMODIFIERS=""            : suppresses XOpenIM() so GTK2 does not enter degraded state
export GTK2_RC_FILES=/dev/null
export XMODIFIERS=""
# Forward WAYLAND_DISPLAY when launched from a .desktop entry (session manager
# may not export it).  Without this, PROTON_ENABLE_WAYLAND=1 silently falls
# back to X11 because winewayland.drv checks WAYLAND_DISPLAY at init time.
if [[ -z "${WAYLAND_DISPLAY:-}" ]]; then
    for _wd in /run/user/"$(id -u)"/wayland-0 /run/user/"$(id -u)"/wayland-1; do
        [[ -S "$_wd" ]] && export WAYLAND_DISPLAY="${_wd##*/}" && break
    done
fi
exec /usr/bin/steam "$@"
STEAMEOF
    chmod 755 "$wrapper"
    gb_ok "Wrapper installed: greenboost-steam"

    echo ""
    info "Launch Steam with: ${C_CYAN}greenboost-steam${C_RESET}"
}

cmd_proton_clean() {
    # Removes every entry from Steam's compatibilitytools.d except greenboost-proton
    # and LegacyRuntime (the Steam-internal entry).  Stale custom builds can cause
    # Steam to fail loading manifests and appear as ghost entries in the compat tool list.
    local _removed=0
    for _compat_dir in \
        "$HOME/.local/share/Steam/compatibilitytools.d" \
        "$HOME/.steam/root/compatibilitytools.d"
    do
        [[ -d "$_compat_dir" ]] || continue
        for _tool in "$_compat_dir"/*/; do
            [[ -d "$_tool" ]] || continue
            local _name
            _name=$(basename "$_tool")
            case "$_name" in
                greenboost-proton|LegacyRuntime) continue ;;
            esac
            rm -rf "$_tool"
            gb_ok "Removed: ${C_DIM}$_name${C_RESET}"
            _removed=1
        done
    done
    (( _removed )) || gb_info "Nothing to remove — only greenboost-proton present."
    gb_info "Restart Steam to apply changes."
}

# ── cmd_install_proton ────────────────────────────────────────────────
# Gaming wizard entry point.  Runs as root (need_root) so it can write system
# files, then drops back to the real user for the Steam-level symlink.
#
# What it does:
#   1. Removes all previous GreenBoost Proton versions (legacy + current names)
#      from both Steam compat tool directories.
#   2. Installs the 32-bit audit stub to the i386 multiarch path and updates
#      /etc/ld.so.preload to use the library name (not a hard path) so that
#      the dynamic linker resolves the correct ELF class per process automatically
#      — eliminating the ELFCLASS64 "ignored" spam from 32-bit Steam components.
#   3. Installs GreenBoost Proton as a Steam compat tool symlink.
#      GREENBOOST_VULKAN=1 is set automatically by the proton script on every
#      launch — no Steam launch options are required.
cmd_install_proton() {
    need_root install-proton

    local real_user="${SUDO_USER:-$USER}"
    local user_home
    user_home="$(eval echo "~${real_user}")"

    gb_section "GreenBoost Proton"

    # ── 1. Full uninstall of any previous version ─────────────────────────────
    # Run the complete uninstall routine so that system-level artifacts (audit
    # libs, /etc/ld.so.preload entry, systemd unit, polkit rule) and user-level
    # compat-tool directories are all removed before the fresh install begins.
    gb_step "Running pre-install cleanup of any previous GreenBoost Proton version..."
    cmd_uninstall_proton

    # ── 2. Fix libgreenboost_audit.so ELFCLASS mismatch ──────────────────────
    # /etc/ld.so.preload processes every spawned process (including 32-bit ones).
    # Using a bare library name lets glibc's ldconfig cache resolve the right
    # ELF class per architecture — no more ELFCLASS64 errors in 32-bit processes.
    gb_step "Installing multiarch audit stubs..."

    # Remove any existing audit preload entry NOW — before spawning cp — so that
    # child processes don't inherit a stale/mismatched libgreenboost_audit.so via
    # /etc/ld.so.preload.  A broken preloaded library causes every spawned child
    # (including cp itself) to SIGSEGV before it can do any useful work.
    # The new $LIB-based entry is written back below, after both copies succeed.
    if [[ -f /etc/ld.so.preload ]]; then
        sed -i '/libgreenboost_audit/d' /etc/ld.so.preload
    fi

    local audit64_src="$MODULE_DIR/$AUDIT_LIB"
    local audit32_src="$MODULE_DIR/$AUDIT_LIB32"
    local dest64="/usr/local/lib/x86_64-linux-gnu"
    local dest32="/usr/local/lib/i386-linux-gnu"

    if [[ -f "$audit64_src" ]]; then
        mkdir -p "$dest64"
        local _tmp64="${dest64}/libgreenboost_audit.so.new"
        if cp "$audit64_src" "$_tmp64" && mv "$_tmp64" "${dest64}/libgreenboost_audit.so"; then
            gb_info "64-bit: ${C_DIM}${dest64}/libgreenboost_audit.so${C_RESET}"
        else
            rm -f "$_tmp64"
            gb_warn_ui "Could not install 64-bit audit library — run 'make audit' and retry"
        fi
    fi

    if [[ -f "$audit32_src" ]]; then
        mkdir -p "$dest32"
        local _tmp32="${dest32}/libgreenboost_audit.so.new"
        if cp "$audit32_src" "$_tmp32" && mv "$_tmp32" "${dest32}/libgreenboost_audit.so"; then
            gb_info "32-bit: ${C_DIM}${dest32}/libgreenboost_audit.so${C_RESET}"
        else
            rm -f "$_tmp32"
            gb_warn_ui "32-bit stub not found or copy failed — run 'make audit32' and retry"
        fi
    else
        gb_warn_ui "32-bit stub not found at $audit32_src — run 'make audit32' to build it"
    fi

    # Refresh ldconfig so both arch versions are indexed
    ldconfig 2>/dev/null || true

    # Update /etc/ld.so.preload: use $LIB token so glibc expands to the correct
    # multiarch path per ELF class before open():
    #   64-bit → /usr/local/lib/x86_64-linux-gnu/libgreenboost_audit.so
    #   32-bit → /usr/local/lib/i386-linux-gnu/libgreenboost_audit.so
    # Both libraries are installed (make audit + make audit32). $LIB is resolved
    # to a full absolute path by the dynamic linker before open(), so it works in
    # AppArmor-confined namespaces without relying on ldconfig.
    # Single-quotes prevent shell expansion here — $LIB must appear literally in
    # ld.so.preload for glibc to expand it at process-load time.
    echo '/usr/local/$LIB/libgreenboost_audit.so' >> /etc/ld.so.preload
    gb_ok "ld.so.preload updated (\$LIB multiarch)"

    # Sync updated AppArmor abstraction so the multiarch path is allowed.
    # install-sys-configs deploys the abstraction from the repo, but a user who
    # ran install-sys-configs before install-steam has the old file on disk
    # (only /usr/local/lib/ path, missing x86_64-linux-gnu/).  Redeploy and
    # reload so confined processes (unix-chkpwd, lsblk, dig, …) stop getting
    # DENIED in the audit log.
    local aa_src="$MODULE_DIR/apparmor/abstractions/greenboost-audit"
    local aa_dest="/etc/apparmor.d/abstractions/greenboost-audit"
    if [[ -f "$aa_src" && -d /etc/apparmor.d/abstractions ]]; then
        cp "$aa_src" "$aa_dest"
        apparmor_parser -r /etc/apparmor.d/ 2>/dev/null &
        gb_spin $! "Reloading AppArmor rules (multiarch paths)..."
        gb_ok "AppArmor abstraction updated — multiarch paths now allowed"
    fi

    # ── 3. Install GreenBoost Proton compat tool ─────────────────────
    local proton_installer="$MODULE_DIR/greenboost_proton/install.sh"
    if [[ ! -f "$proton_installer" ]]; then
        gb_warn_ui "Installer not found: $proton_installer"
        return 1
    fi

    gb_step "Installing GreenBoost Proton..."
    # Run as the real user so the symlink lands in their Steam directory
    sudo -u "$real_user" bash "$proton_installer"

    # ── 4. Install greenboost-gaming.service + polkit rule ────────────────────
    # greenboost-gaming.service stops Ollama on start and restores it on stop.
    # The proton script calls systemctl start/stop without sudo — the polkit rule
    # grants any active local user permission to control this specific service.
    gb_step "Installing gaming mode service and polkit rule..."

    cat > /etc/systemd/system/greenboost-gaming.service << 'GAMINGEOF'
[Unit]
Description=GreenBoost Gaming Mode — suspend AI inference during gaming session
After=ollama.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/systemctl stop ollama.service
ExecStop=/usr/bin/systemctl start ollama.service

[Install]
WantedBy=multi-user.target
GAMINGEOF

    mkdir -p /etc/polkit-1/rules.d
    cat > /etc/polkit-1/rules.d/50-greenboost-gaming.rules << 'POLKITEOF'
/* GreenBoost: allow any active local user to start/stop greenboost-gaming.service
 * without a password prompt.  This lets the Proton script trigger gaming mode
 * automatically on game launch without requiring sudo. */
polkit.addRule(function(action, subject) {
    if (action.id == "org.freedesktop.systemd1.manage-units" &&
        action.lookup("unit") == "greenboost-gaming.service" &&
        subject.local && subject.active) {
        return polkit.Result.YES;
    }
});
POLKITEOF

    systemctl daemon-reload
    gb_ok "Gaming mode service installed (auto gaming-mode on Proton launch)"
    cmd_fix_steam
    cmd_steam_launch_info

}

cmd_uninstall_proton() {
    local real_user="${SUDO_USER:-$USER}"
    local user_home
    user_home="$(eval echo "~${real_user}")"

    # ── 1. Remove user-level compat tool directory ────────────────────────────
    local proton_installer="$MODULE_DIR/greenboost_proton/install.sh"
    if [[ -f "$proton_installer" ]]; then
        sudo -u "$real_user" bash "$proton_installer" --uninstall
    else
        # Fallback: remove known install paths directly
        local _removed=0
        for _d in \
            "${user_home}/.local/share/Steam/compatibilitytools.d/greenboost-proton" \
            "${user_home}/.steam/root/compatibilitytools.d/greenboost-proton" \
            "${user_home}/.local/share/Steam/compatibilitytools.d/greenboost-proton-wayland" \
            "${user_home}/.steam/root/compatibilitytools.d/greenboost-proton-wayland"
        do
            if [[ -L "$_d" || -d "$_d" ]]; then
                rm -rf "$_d"
                gb_ok "Removed: ${C_DIM}$_d${C_RESET}"
                _removed=1
            fi
        done
        (( _removed )) || gb_info "No compat tool directories found."
    fi

    # ── 1b. Remove user-local Vulkan layer files (installed by install-proton) ──
    for _vkf in \
        "${user_home}/.local/share/vulkan/libVkLayer_greenboost.so" \
        "${user_home}/.local/share/vulkan/implicit_layer.d/VkLayer_greenboost.json"
    do
        if [[ -f "$_vkf" ]]; then
            rm -f "$_vkf"
            gb_ok "Removed: ${C_DIM}$_vkf${C_RESET}"
        fi
    done

    # ── 2. Remove system-level artifacts installed by install-proton ──
    # gaming service + polkit rule
    for _f in \
        /etc/systemd/system/greenboost-gaming.service \
        /etc/polkit-1/rules.d/50-greenboost-gaming.rules
    do
        if [[ -f "$_f" ]]; then
            local _svcname; _svcname=$(basename "$_f")
            # Disable service before removing
            [[ "$_f" == *.service ]] && systemctl disable --now "$_svcname" 2>/dev/null || true
            rm -f "$_f"
            gb_ok "Removed: ${C_DIM}$_f${C_RESET}"
        fi
    done
    systemctl daemon-reload 2>/dev/null || true

    # multiarch audit libraries (installed into multiarch paths by install-proton)
    # Remove the $LIB-based preload entry first — unconditionally — so that no
    # child process inherits a stale reference even if the .so files are already gone.
    if [[ -f /etc/ld.so.preload ]] && grep -q 'libgreenboost_audit' /etc/ld.so.preload; then
        sed -i '/libgreenboost_audit/d' /etc/ld.so.preload
    fi

    local _audit_removed=0
    for _lib in \
        /usr/local/lib/x86_64-linux-gnu/libgreenboost_audit.so \
        /usr/local/lib/i386-linux-gnu/libgreenboost_audit.so
    do
        if [[ -f "$_lib" ]]; then
            rm -f "$_lib"
            (( _audit_removed++ )) || true
            gb_info "Removed audit lib: ${C_DIM}$_lib${C_RESET}"
        fi
    done
    ldconfig 2>/dev/null || true
    gb_ok "Audit libraries removed and ldconfig refreshed"

    gb_ok "GreenBoost Proton uninstalled. Restart Steam to apply."
}

cmd_install_llama_configs() {
    need_root install-llama-configs

    local audit_path="$SHIM_DEST/$AUDIT_LIB"

    # /etc/ld.so.preload — place the tiny LD_AUDIT library (NOT the full shim).
    # The audit library fires la_objopen() and injects the full shim only into
    # processes that load libcuda.so or libcudart.so — zero AppArmor blast radius
    # on non-CUDA confined processes (GDM, PAM, snap-confine).
    if [[ ! -f "$audit_path" ]]; then
        warn "LD_AUDIT library not found at $audit_path — run 'make audit' and re-run this command"
        return 1
    fi

    # Remove any legacy full-shim entry from prior installs
    if [[ -f /etc/ld.so.preload ]] && grep -q "$SHIM_LIB" /etc/ld.so.preload; then
        sed -i "/$SHIM_LIB/d" /etc/ld.so.preload
        info "ld.so.preload: removed legacy full-shim entry"
    fi
    if ! grep -q "$AUDIT_LIB" /etc/ld.so.preload 2>/dev/null; then
        echo "$audit_path" >> /etc/ld.so.preload
        info "ld.so.preload: added $audit_path (LD_AUDIT CUDA-aware injection)"
    else
        info "ld.so.preload: already contains $AUDIT_LIB (skip)"
    fi

    # Remove any stale i386 companion entry — ld.so.preload has no architecture
    # filter, so the 32-bit lib generates ELFCLASS32 errors in every 64-bit
    # process.  The single ELFCLASS64 warning from Steam's 32-bit bootstrap
    # (ubuntu12_32/steam) is harmless and marked "ignored" by ld.so.
    if grep -qF "i386-linux-gnu" /etc/ld.so.preload 2>/dev/null; then
        sed -i '/i386-linux-gnu/d' /etc/ld.so.preload
        info "ld.so.preload: removed i386 companion (was causing ELFCLASS32 spam in every 64-bit process)"
    fi

    echo ""
    info "ld.so.preload configured — GreenBoost shim injects automatically into CUDA processes."
}

# ---- do_purge — remove ALL previously installed GreenBoost artifacts -----
# Internal helper (no root check — callers must ensure root).
# Called by cmd_uninstall and cmd_full_install.
do_purge() {
    # restart_after=1 → restart stopped services at the end (cmd_uninstall).
    # restart_after=0 → leave them stopped (cmd_full_install handles restart after fresh install).
    local restart_after="${1:-0}"

    # 1. Stop services that hold /dev/greenboost open (prevents rmmod EBUSY).
    #    Ollama and llama-server are NOT uninstalled — only stopped temporarily.
    local _stopped_svcs=""
    for svc in ollama llama-server; do
        if systemctl is-active --quiet "$svc.service" 2>/dev/null; then
            systemctl stop --wait "$svc" 2>/dev/null || true
            _stopped_svcs="$_stopped_svcs $svc"
            GB_STOPPED_SERVICES="$GB_STOPPED_SERVICES $svc"
        fi
    done
    [[ -n "$_stopped_svcs" ]] && gb_ok "Services stopped:${_stopped_svcs}"

    # Kill any remaining process with the device open.
    if [[ -e /dev/greenboost ]]; then
        fuser -k /dev/greenboost 2>/dev/null || true
        local waited=0
        while [[ -e /dev/greenboost ]] && fuser /dev/greenboost &>/dev/null; do
            sleep 0.2
            (( waited++ ))
            [[ $waited -ge 25 ]] && break
        done
    fi

    # 2. Remove kernel module from ALL installed kernel versions (system-agnostic).
    #    Using find covers multi-kernel setups and non-current kernels.
    local _ko_removed=0
    while IFS= read -r _ko_file; do
        rm -f "$_ko_file" && (( _ko_removed++ )) || true
    done < <(find /lib/modules -name "greenboost.ko*" 2>/dev/null)
    # Remove DKMS state so re-install can always proceed from scratch.
    # Use both dkms remove (clean bookkeeping) and direct rm (handles corrupted state).
    if command -v dkms &>/dev/null; then
        dkms remove greenboost/"${GB_VERSION}" --all &>/dev/null || true
    fi
    rm -rf /var/lib/dkms/greenboost 2>/dev/null || true
    rm -rf /usr/src/greenboost-* 2>/dev/null || true
    { depmod -a 2>/dev/null || true; } &
    gb_spin $! "Rebuilding module dependency tree..."

    # 3. Remove GreenBoost entries from /etc/ld.so.preload FIRST — before deleting
    #    the .so files (prevents "cannot be preloaded" linker errors on forked procs).
    if [[ -f /etc/ld.so.preload ]] && grep -q "libgreenboost" /etc/ld.so.preload; then
        sed -i '/libgreenboost/d' /etc/ld.so.preload
        [[ -s /etc/ld.so.preload ]] || rm -f /etc/ld.so.preload
        gb_ok "Removed from /etc/ld.so.preload"
    fi

    # 4. Remove CUDA shim and LD_AUDIT library
    local _libs_removed=0
    for lib in "$SHIM_DEST/$SHIM_LIB" "$SHIM_DEST/$AUDIT_LIB" "$SHIM_DEST/$VULKAN_LAYER_LIB"; do
        [[ -f "$lib" ]] && rm -f "$lib" && (( _libs_removed++ )) || true
    done
    rm -f "$VULKAN_IMPLICIT_LAYER_DIR/$VULKAN_LAYER_MANIFEST" 2>/dev/null || true
    { ldconfig 2>/dev/null || true; } &
    gb_spin $! "Refreshing dynamic linker cache..."
    [[ $_libs_removed -gt 0 ]] && gb_ok "CUDA shim + audit library + Vulkan layer removed"

    # 4b. Remove AppArmor abstraction + all injected includes
    #     Progress bar over profile list — no per-profile noise.
    if [[ -f /etc/apparmor.d/abstractions/greenboost-audit ]]; then
        rm -f /etc/apparmor.d/abstractions/greenboost-audit

        # Layer A — remove from abstractions/base
        local base_abs="/etc/apparmor.d/abstractions/base"
        if [[ -f "$base_abs" ]] && grep -q "greenboost-audit" "$base_abs"; then
            sed -i '/greenboost-audit/d' "$base_abs"
        fi

        # Layer B — scan all profiles; show progress bar, suppress per-profile output
        local _aa_profiles=()
        while IFS= read -r _pf; do
            _aa_profiles+=("$_pf")
        done < <(find /etc/apparmor.d/ -maxdepth 1 -type f | sort)

        local _aa_total="${#_aa_profiles[@]}"
        local _aa_cleaned=0 _aa_idx=0
        for profile in "${_aa_profiles[@]}"; do
            (( _aa_idx++ )) || true
            [[ -d "$profile" ]] && continue
            grep -q "greenboost-audit" "$profile" 2>/dev/null || continue
            sed -i '/greenboost-audit/d' "$profile"
            (( _aa_cleaned++ )) || true
            # Progress bar: overwrite same line
            local _pct=$(( _aa_idx * 100 / _aa_total ))
            local _filled=$(( _pct / 2 )) _empty=$(( 50 - _pct / 2 ))
            printf "\r  ${C_GRAY}[%d/%d]${C_RESET} ${C_AMBER}%s${C_GRAY}%s${C_RESET} %3d%%  ${C_DIM}Reverting AppArmor permissions...${C_RESET}" \
                "$_aa_idx" "$_aa_total" \
                "$(printf '█%.0s' $(seq 1 "$_filled" 2>/dev/null || true))" \
                "$(printf '░%.0s' $(seq 1 "$_empty"  2>/dev/null || true))" \
                "$_pct"
        done
        printf "\r%*s\r" "$(tput cols 2>/dev/null || echo 80)" ""

        if [[ $_aa_cleaned -gt 0 ]]; then
            gb_ok "AppArmor: reverted $_aa_cleaned profiles"
        else
            gb_info "AppArmor: no profiles needed reverting"
        fi

        apparmor_parser -r /etc/apparmor.d/ 2>/dev/null &
        gb_spin $! "Reloading AppArmor rules..."
    fi

    # 5. Remove static config files (silent batch — no per-file noise)
    local _cfg_removed=0
    for f in \
        /etc/modprobe.d/greenboost.conf \
        /etc/modprobe.d/greenboost.conf.bak \
        /etc/profile.d/greenboost.sh \
        /usr/local/bin/greenboost \
        /usr/local/bin/greenboost-steam \
        /etc/modules-load.d/greenboost.conf \
        /etc/udev/rules.d/99-greenboost.rules \
        /etc/udev/rules.d/99-nvme-greenboost.rules \
        /etc/sysctl.d/99-greenboost.conf \
        /etc/sysctl.d/99-zzz-greenboost.conf \
        /etc/sysfs.d/greenboost-hugepages.conf \
        /etc/security/limits.d/99-greenboost-gaming.conf; do
        [[ -f "$f" ]] && rm -f "$f" && (( _cfg_removed++ )) || true
    done
    udevadm control --reload-rules 2>/dev/null || true
    [[ $_cfg_removed -gt 0 ]] && gb_ok "Config files removed ($_cfg_removed files)"
    rm -rf /etc/greenboost

    # 6. Disable + remove ALL GreenBoost systemd services — generic glob catches any
    #    service file installed by any version of full-install, regardless of name.
    local _svcs_removed=0
    for _svc_file in /etc/systemd/system/greenboost*.service \
                     /etc/systemd/system/greenboostd.service \
                     /etc/systemd/system/cpu-perf.service; do
        [[ -f "$_svc_file" ]] || continue
        local _svc_name; _svc_name=$(basename "$_svc_file")
        systemctl disable --now "$_svc_name" 2>/dev/null || true
        rm -f "$_svc_file"
        (( _svcs_removed++ )) || true
    done
    [[ $_svcs_removed -gt 0 ]] && gb_ok "$_svcs_removed GreenBoost service(s) removed"
    # Gaming mode polkit rule
    rm -f /etc/polkit-1/rules.d/50-greenboost-gaming.rules
    # Ollama drop-in override
    rm -f /etc/systemd/system/ollama.service.d/99-greenboost.conf
    rmdir --ignore-fail-on-non-empty /etc/systemd/system/ollama.service.d/ 2>/dev/null || true
    # TurboQuant daemon
    systemctl disable greenboost-turboquant.service 2>/dev/null || true
    rm -f /usr/local/bin/greenboost-turboquant \
          /usr/local/lib/libgreenboost_tq.so \
          /etc/systemd/system/greenboost-turboquant.service
    rm -rf /usr/local/lib/greenboost

    # Installed daemon scripts and CLI wrappers (all known names across all versions)
    rm -f /usr/local/sbin/greenboost-recover \
          /usr/local/sbin/greenboost-vram-watchdog \
          /usr/local/sbin/greenboostd \
          /usr/local/bin/greenboost-shader-boost \
          /usr/local/bin/greenboost-idle-reclaim \
          /usr/local/bin/greenboost-run \
          /usr/local/bin/greenboost-run-tgi \
          /usr/local/bin/greenboost-run-unsloth \
          /usr/local/bin/greenboost-run-vllm
    # Documentation directory and stray Python hooks
    rm -rf /usr/local/share/greenboost/
    rm -f /usr/local/lib/greenboost_*.py
    # State files
    rm -f /var/lib/greenboost/sentinel \
          /var/lib/greenboost/running \
          /var/lib/greenboost/last_clean_boot \
          /var/lib/greenboost/vram_pressure
    rmdir /var/lib/greenboost 2>/dev/null || true
    systemctl daemon-reload 2>/dev/null || true

    # 7. Strip GreenBoost env lines from service unit files
    local _svc_stripped=0
    local ollama_svc="/etc/systemd/system/ollama.service"
    if [[ -f "$ollama_svc" ]] && grep -qE "GREENBOOST_|libgreenboost" "$ollama_svc"; then
        sed -i '/OLLAMA_FLASH_ATTENTION/d
/OLLAMA_KV_CACHE_TYPE/d
/OLLAMA_NUM_CTX/d
/OLLAMA_MAX_LOADED_MODELS/d
/OLLAMA_KEEP_ALIVE/d
/GREENBOOST_/d
/libgreenboost/d' "$ollama_svc"
        (( _svc_stripped++ )) || true
    fi
    if [[ -f /etc/systemd/system/llama-server.service ]] && \
       grep -q "libgreenboost" /etc/systemd/system/llama-server.service; then
        sed -i '/libgreenboost/d
/GREENBOOST_/d' /etc/systemd/system/llama-server.service
        (( _svc_stripped++ )) || true
    fi
    systemctl daemon-reload 2>/dev/null || true
    [[ $_svc_stripped -gt 0 ]] && gb_ok "Service unit files cleaned ($_svc_stripped files)"

    # 8. Remove /opt/greenboost if present (legacy venv location)
    if [[ -d /opt/greenboost ]]; then
        rm -rf /opt/greenboost
        gb_ok "Removed /opt/greenboost"
    fi

    # 8b. Remove /swap_nvme.img if present — created by older GreenBoost versions.
    #     Only this specific file is touched; the system's regular swap (/swap.img,
    #     /swapfile, swap partitions) is never modified.
    if [[ -f /swap_nvme.img ]]; then
        swapoff /swap_nvme.img 2>/dev/null || true
        rm -f /swap_nvme.img
        # Remove the fstab entry added by the old setup
        if grep -q '/swap_nvme.img' /etc/fstab 2>/dev/null; then
            sed -i '\|/swap_nvme\.img|d' /etc/fstab
            gb_ok "Removed /swap_nvme.img and its /etc/fstab entry"
        else
            gb_ok "Removed /swap_nvme.img (no fstab entry found)"
        fi
    fi

    # 8b2. Remove greenboost_swap.img if present — T3 swap file created by v2.9+
    local _gb_swap="/var/lib/greenboost/greenboost_swap.img"
    if [[ -f "$_gb_swap" ]]; then
        swapoff "$_gb_swap" 2>/dev/null || true
        rm -f "$_gb_swap"
        sed -i '\|greenboost_swap\.img|d' /etc/fstab
        gb_ok "Removed T3 swap file (${_gb_swap})"
    fi

    # 8c. Remove T3 backing file — created by GreenBoost v2.8+ as a sparse file.
    #     The module closes the file before rmmod, so it is safe to remove.
    if [[ -f /var/lib/greenboost/t3_store ]]; then
        rm -f /var/lib/greenboost/t3_store
        gb_ok "Removed T3 backing file (/var/lib/greenboost/t3_store)"
    fi
    rmdir /var/lib/greenboost 2>/dev/null || true

    # 9. Unload kernel module — done last so all consumers are gone.
    if lsmod | grep -q "^${DRIVER_NAME} "; then
        if rmmod "$DRIVER_NAME" 2>/dev/null; then
            gb_ok "Kernel module unloaded"
        else
            fuser -k /dev/greenboost 2>/dev/null || true
            sleep 1
            if rmmod "$DRIVER_NAME" 2>/dev/null; then
                gb_ok "Kernel module unloaded (after retry)"
            else
                gb_warn_ui "rmmod failed — module still loaded."
                gb_warn_ui "Run manually: sudo rmmod ${DRIVER_NAME}"
            fi
        fi
    fi

    # 10. Restart services that were stopped before the purge.
    #     Only done on standalone uninstall (restart_after=1).
    #     On full-install (restart_after=0) services stay stopped — cmd_full_install
    #     restarts them once at the very end, after the fresh install completes.
    if [[ $restart_after -eq 1 ]]; then
        for svc in $GB_STOPPED_SERVICES; do
            systemctl start "$svc" 2>/dev/null \
                && gb_ok "$svc restarted" \
                || gb_warn_ui "$svc failed to restart — check: journalctl -u $svc"
        done
    fi
}

cmd_build() {
    [[ -d /usr/local/cuda/bin ]] && export PATH="/usr/local/cuda/bin:$PATH"
    make -C "$MODULE_DIR" all &>/tmp/gb_build.log &
    gb_spin $! "Compiling kernel module + CUDA shim..."
    if ! wait $!; then
        gb_fail() { echo -e "  ${RED}✗${C_RESET}  Build failed:"; }
        cat /tmp/gb_build.log >&2
        die "Build failed — see output above"
    fi
    gb_ok "Build complete  (greenboost.ko · ${SHIM_LIB} · ${AUDIT_LIB} · ${VULKAN_LAYER_LIB})"
}

cmd_install() {
    need_root install
    detect_hardware
    check_deps

    # Step 0: clean previous installation to guarantee a fresh install
    # Skip when called from cmd_full_install which already ran do_purge at step 0/5.
    if [[ "${GB_SKIP_INSTALL_PURGE:-0}" -ne 1 ]]; then
        gb_step "Removing previous GreenBoost installation (if any)..."
        do_purge 0
        gb_ok "Previous installation removed"
    fi

    cmd_build

    local _install_log
    _install_log=$(mktemp /tmp/gb_install.XXXXX.log)
    { make -C "$MODULE_DIR" install >"$_install_log" 2>&1; } &
    gb_spin $! "Installing kernel module..."
    if ! wait $!; then
        echo "" >&2
        cat "$_install_log" >&2
        rm -f "$_install_log"
        die "Module install failed — see output above"
    fi
    rm -f "$_install_log"
    gb_ok "Kernel module installed"

    cp "$MODULE_DIR/$SHIM_LIB" "$SHIM_DEST/"
    [[ -f "$MODULE_DIR/$AUDIT_LIB" ]] && cp "$MODULE_DIR/$AUDIT_LIB" "$SHIM_DEST/"
    [[ -f "$MODULE_DIR/$VULKAN_LAYER_LIB" ]] && cp "$MODULE_DIR/$VULKAN_LAYER_LIB" "$SHIM_DEST/"
    { ldconfig 2>/dev/null; } &
    gb_spin $! "Installing CUDA shim + LD_AUDIT library + Vulkan layer..."
    gb_ok "Libraries installed to $SHIM_DEST/"

    # modprobe defaults
    # Ensure T3 backing store directory exists before writing modprobe.conf
    mkdir -p /var/lib/greenboost

    cat > /etc/modprobe.d/greenboost.conf << MODEOF
# GreenBoost — cuda memory pool (auto-configured for detected hardware)
# GPU   : ${GPU_NAME}  (${GB_PHYS} GB VRAM)
# RAM   : ${RAM_TYPE}-${RAM_SPEED_MT}  (pool ${GB_VIRT} GB, reserve ${GB_RESERVE} GB)
# T3    : file-backed (/var/lib/greenboost/t3_store, cap ${GB_NVME_POOL} GB)
options greenboost physical_vram_gb=${GB_PHYS} virtual_vram_gb=${GB_VIRT} safety_reserve_gb=${GB_RESERVE} nvme_pool_gb=${GB_NVME_POOL} t3_max_gb=${GB_NVME_POOL} t3_file_path=/var/lib/greenboost/t3_store pcores_max_cpu=${GB_PCORES_MAX} golden_cpu_min=${GB_GOLDEN_MIN} golden_cpu_max=${GB_GOLDEN_MAX} ecores_only=${GB_PCORES_ONLY}
MODEOF

    # profile.d — auto-activate GreenBoost for all CUDA inference tools launched
    # from a login shell (terminal, SSH).  GREENBOOST_ACTIVE=1 is exported globally
    # so vLLM, PyTorch scripts, TGI, Transformers, etc. all work without any wrapper.
    # The greenboost() function remains available as a fallback for non-login contexts.
    cat > /etc/profile.d/greenboost.sh << PROFEOF
# GreenBoost v2.8.2 — auto-activation for CUDA inference tools
export GREENBOOST_ACTIVE=1
export GREENBOOST_SHIM="$SHIM_DEST/$SHIM_LIB"
# 'greenboost run' is a fallback for non-login contexts (cron, Docker entrypoints)
greenboost() {
    case "\$1" in
        run) shift; GREENBOOST_ACTIVE=1 LD_PRELOAD="\$GREENBOOST_SHIM" "\$@" ;;
        *) echo "Usage: greenboost run <app> [args...]" >&2; return 1 ;;
    esac
}
export -f greenboost
PROFEOF

    # Standalone wrapper — fallback for non-login contexts where profile.d is not sourced
    cat > /usr/local/bin/greenboost << WRAPEOF
#!/usr/bin/env bash
# GreenBoost CLI wrapper
# Usage: greenboost <command> [args...]
# Run 'greenboost help' for the full command reference.
GB_SETUP="$MODULE_DIR/greenboost_setup.sh"
case "\$1" in
    status)       exec "\$GB_SETUP" status "\${@:2}" ;;
    clean)        exec "\$GB_SETUP" clean "\${@:2}" ;;
    clean-memory) exec "\$GB_SETUP" clean-memory ;;
    benchmark)    exec "\$GB_SETUP" benchmark "\${@:2}" ;;
    profile)      exec "\$GB_SETUP" profile "\${@:2}" ;;
    load)         exec "\$GB_SETUP" load "\${@:2}" ;;
    unload)       exec "\$GB_SETUP" unload ;;
    tune)         exec "\$GB_SETUP" tune "\${@:2}" ;;
    tune-grub)    exec "\$GB_SETUP" tune-grub ;;
    tune-sysctl)  exec "\$GB_SETUP" tune-sysctl ;;
    tune-libs)    exec "\$GB_SETUP" tune-libs ;;
    tune-all)     exec "\$GB_SETUP" tune-all ;;
    setup|install|full-install) exec "\$GB_SETUP" "\$@" ;;
    steam-launch-guide) exec "\$GB_SETUP" steam-launch-guide ;;
    fix-steam)          exec "\$GB_SETUP" fix-steam ;;
    remove-proton)      exec "\$GB_SETUP" remove-proton ;;
    uninstall-mangohud) exec "\$GB_SETUP" uninstall-mangohud ;;
    vulkan)       exec "\$GB_SETUP" vulkan "\${@:2}" ;;
    logs)            exec "\$GB_SETUP" logs "\${@:2}" ;;
    proton-logs)     exec "\$GB_SETUP" proton-logs "\${@:2}" ;;
    inference-logs)  exec "\$GB_SETUP" inference-logs "\${@:2}" ;;
    clear)           exec "\$GB_SETUP" clear "\${@:2}" ;;
    clean-logs)      exec "\$GB_SETUP" clean-logs ;;
    run)          shift; GREENBOOST_ACTIVE=1 LD_PRELOAD="$SHIM_DEST/$SHIM_LIB" "\$@" ;;
    help|--help|-h|"") exec "\$GB_SETUP" show-commands ;;
    *)            echo "Unknown command: '\$1'  — run: greenboost help" >&2; exit 1 ;;
esac
WRAPEOF
    chmod +x /usr/local/bin/greenboost

    # Install commands reference so 'greenboost help' works from any path
    mkdir -p /usr/local/share/greenboost
    if [[ -f "$MODULE_DIR/GREENBOOST_COMMANDS.md" ]]; then
        install -m 644 "$MODULE_DIR/GREENBOOST_COMMANDS.md" /usr/local/share/greenboost/GREENBOOST_COMMANDS.md
    fi

    gb_ok "Installation complete"
    gb_info "Load:    sudo modprobe greenboost"
    gb_info "Status:  greenboost status"
}

cmd_load() {
    need_root load
    detect_hardware

    # Load active profile values (lowest priority — env vars and CLI flags override)
    if [[ -n "$GB_PROFILE_FILE" ]]; then
        resolve_profile "$GB_PROFILE_FILE"
    elif [[ -f "$GB_ACTIVE_PROFILE_LINK" ]]; then
        load_profile_values "$GB_ACTIVE_PROFILE_LINK"
        [[ -n "$PROF_PHYS"       ]] && GB_PHYS=$PROF_PHYS
        [[ -n "$PROF_VIRT"       ]] && GB_VIRT=$PROF_VIRT
        [[ -n "$PROF_RESERVE"    ]] && GB_RESERVE=$PROF_RESERVE
        [[ -n "$PROF_NVME_SWAP"  ]] && GB_NVME_SWAP=$PROF_NVME_SWAP
        [[ -n "$PROF_NVME_POOL"  ]] && GB_NVME_POOL=$PROF_NVME_POOL
        [[ -n "$PROF_PCORES_ONLY" ]] && GB_PCORES_ONLY=$PROF_PCORES_ONLY
        [[ -n "$PROF_OLLAMA_CTX" ]] && GB_OLLAMA_CTX=$PROF_OLLAMA_CTX
    fi

    local phys="${GPU_PHYS_GB:-${GB_PHYS}}"
    local virt="${VIRT_VRAM_GB:-${GB_VIRT}}"
    local res="${RESERVE_GB:-${GB_RESERVE}}"
    local nvme_sw="${NVME_SWAP_GB:-${GB_NVME_SWAP}}"
    local nvme_pool="${NVME_POOL_GB:-${GB_NVME_POOL}}"
    local pcores_max="${PROF_PCORES_MAX:-${GB_PCORES_MAX}}"
    local golden_min="${PROF_GOLDEN_MIN:-${GB_GOLDEN_MIN}}"
    local golden_max="${PROF_GOLDEN_MAX:-${GB_GOLDEN_MAX}}"
    local ecores_only="${GB_PCORES_ONLY}"

    # KV cache reserve: auto-scale from OLLAMA_NUM_CTX or profile value.
    # Larger contexts need more headroom so weights overflow to T2 sooner
    # and KV cache stays in fast VRAM.
    if [[ -z "${GB_KV_RESERVE_MB:-}" ]]; then
        local _ctx="${OLLAMA_NUM_CTX:-${PROF_OLLAMA_CTX:-32768}}"
        if   [[ $_ctx -le 8192   ]]; then GB_KV_RESERVE_MB=1024
        elif [[ $_ctx -le 32768  ]]; then GB_KV_RESERVE_MB=2048
        elif [[ $_ctx -le 65536  ]]; then GB_KV_RESERVE_MB=4096
        elif [[ $_ctx -le 131072 ]]; then GB_KV_RESERVE_MB=6144
        else                              GB_KV_RESERVE_MB=8192
        fi
        info "KV reserve auto-set to ${GB_KV_RESERVE_MB} MB (OLLAMA_NUM_CTX=${_ctx})"
    fi

    if lsmod | grep -q "^${DRIVER_NAME} "; then
        warn "Module already loaded — reloading..."
        # Stop consumers before rmmod to avoid EBUSY
        for svc in ollama llama-server; do
            systemctl is-active --quiet "$svc" 2>/dev/null && \
                systemctl stop "$svc" 2>/dev/null || true
        done
        [[ -e /dev/greenboost ]] && { fuser -k /dev/greenboost 2>/dev/null || true; sleep 0.5; }
        rmmod "$DRIVER_NAME" || die "Failed to unload existing module"
    fi

    local ko="$MODULE_DIR/greenboost.ko"
    [[ -f "$ko" ]] || die "greenboost.ko not found — run: make  or  $0 build"

    insmod "$ko" \
        physical_vram_gb="$phys"  \
        virtual_vram_gb="$virt"   \
        safety_reserve_gb="$res"  \
        nvme_swap_gb="$nvme_sw"   \
        nvme_pool_gb="$nvme_pool" \
        pcores_max_cpu="$pcores_max" \
        golden_cpu_min="$golden_min" \
        golden_cpu_max="$golden_max" \
        ecores_only="$ecores_only" \
        kv_reserve_mb="${GB_KV_RESERVE_MB:-2048}" \
        active_profile_name="${PROF_NAME:-autodetect}" \
        || die "insmod failed — check: dmesg | tail -20"

    info "GreenBoost v2.8.2 loaded — cuda memory pool active!"
    info ""
    info "  T1 ${GPU_NAME} : ${phys} GB  [hot layers]"
    info "  T2 ${RAM_TYPE} pool         : ${virt} GB  [cold layers]"
    info "  T3 NVMe swap               : ${nvme_sw} GB  [frozen pages]"
    info "  ─────────────────────────────────────────"
    info "  Combined view              : $(( phys + virt + nvme_sw )) GB total model capacity"
    info ""
    info "Status     : greenboost status"
    info "Kernel log : dmesg | grep greenboost"
    echo ""
    dmesg | grep greenboost | tail -8 | sed 's/^/  /'
}

cmd_unload() {
    need_root unload
    if lsmod | grep -q "^${DRIVER_NAME} "; then
        rmmod "$DRIVER_NAME" && info "GreenBoost unloaded" \
            || die "rmmod failed — check: dmesg | tail -5"
    else
        info "GreenBoost is not loaded"
    fi
}

cmd_uninstall() {
    need_root uninstall
    GB_STOPPED_SERVICES=""

    info "============================================================"
    info " GreenBoost — Uninstall"
    info "============================================================"
    info ""
    info "What will be removed:"
    info "  - Kernel module (greenboost.ko) from /lib/modules"
    info "  - CUDA shim (libgreenboost_cuda.so) from /usr/local/lib"
    info "  - /etc/ld.so.preload entries for GreenBoost"
    info "  - /etc/modprobe.d/greenboost.conf"
    info "  - /etc/modules-load.d/greenboost.conf"
    info "  - /etc/profile.d/greenboost.sh"
    info "  - /etc/sysctl.d/99-greenboost.conf"
    info "  - /etc/sysfs.d/greenboost-hugepages.conf"
    info "  - /etc/udev/rules.d/99-greenboost*.rules"
    info "  - cpu-perf.service (GreenBoost CPU performance service)"
    info "  - GreenBoost env lines from ollama.service / llama-server.service"
    info "  - AppArmor abstractions for GreenBoost audit library"
    info "  - /usr/local/bin/greenboost (CLI wrapper)"
    info ""
    info "What will NOT be touched:"
    info "  - Ollama itself (not uninstalled — only GreenBoost env vars removed)"
    info "  - llama-server itself (not uninstalled — only GreenBoost env vars removed)"
    info "  - /etc/ld.so.preload entries not related to GreenBoost"
    info "  - NVIDIA drivers, CUDA toolkit, Steam, Wine, Proton"
    info "  - Any user data, models, or application configs"
    info "  - System swap (/swap.img, swap partitions) — system swap is never touched"
    info "  - /swap_nvme.img (old GreenBoost swap) — removed if present"
    info ""
    info "Starting purge..."
    info ""

    do_purge 1

    info ""
    info "============================================================"
    info " GreenBoost uninstalled cleanly."
    info ""
    info " Ollama and llama-server (if installed) have been restarted"
    info " without GreenBoost — they will use native VRAM only."
    info ""
    info " To reinstall GreenBoost at any time:"
    info "   sudo ./greenboost_setup.sh full-install"
    info "============================================================"
}

cmd_tune() {
    need_root tune
    detect_hardware

    info "Tuning workstation for GreenBoost / LLM workloads..."
    info "Hardware: ${CPU_NAME} | ${GPU_NAME} | ${RAM_TYPE}-${RAM_SPEED_MT} MT/s | PCIe Gen ${PCIE_GEN} ${PCIE_WIDTH} (~${PCIE_BW_GBS} GB/s) | ${NVME_SIZE_GB} GB NVMe"
    echo ""

    # ── CPU governor → performance (P-cores run at max boost, not idle) ──
    local changed=0
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [[ -w "$gov" ]] && echo performance > "$gov" && changed=1
    done
    [[ $changed -eq 1 ]] && info "CPU governor      : performance ($(nproc) CPUs)" \
                          || warn "CPU governor      : could not set (check cpufreq driver)"

    # ── NVMe scheduler → none (best latency for Samsung 990 EVO Plus) ──
    for sched in /sys/block/nvme*/queue/scheduler; do
        [[ -w "$sched" ]] && echo none > "$sched" 2>/dev/null || true
    done
    info "NVMe scheduler    : none (was: mq-deadline)"

    # ── NVMe read-ahead: scale to PCIe gen ───────────────────────────────
    # PCIe 5 (64 GB/s) can sustain 8 MB sequential prefetch without stalling;
    # PCIe 4 (32 GB/s) saturates at 4 MB; older gens use 2 MB.
    local ra_kb=2048
    [[ $PCIE_GEN -ge 4 ]] && ra_kb=4096
    [[ $PCIE_GEN -ge 5 ]] && ra_kb=8192
    for ra in /sys/block/nvme*/queue/read_ahead_kb; do
        [[ -w "$ra" ]] && echo $ra_kb > "$ra"
    done
    info "NVMe read_ahead   : ${ra_kb} KB  (scaled to PCIe Gen ${PCIE_GEN})"

    # ── NVMe nr_requests → 1023 (Samsung 990 EVO Plus hardware limit) ────
    for nr in /sys/block/nvme*/queue/nr_requests; do
        [[ -w "$nr" ]] && echo 1023 > "$nr"
    done
    info "NVMe nr_requests  : 1023"

    # ── PCIe runtime PM → on (disable ASPM for GPU slot) ─────────────────
    # With ASPM active the GPU link downclock to Gen 1 (2.5 GT/s) at idle.
    # Forcing runtime PM "on" keeps the link at the negotiated gen during
    # inference, preventing latency spikes on the first DMA transfer.
    local gpu_pci_rt
    gpu_pci_rt=$(ls /proc/driver/nvidia/gpus/ 2>/dev/null | head -1)
    if [[ -n "$gpu_pci_rt" ]]; then
        local pm_ctl="/sys/bus/pci/devices/${gpu_pci_rt}/power/control"
        if [[ -w "$pm_ctl" ]]; then
            echo on > "$pm_ctl"
            info "PCIe GPU ASPM     : disabled (power/control=on, link stays at Gen ${PCIE_GEN})"
        fi
    fi

    # ── PCIe MRRS → 4096 bytes (maximises H2D DMA burst size) ───────────
    # Max Read Request Size controls the burst length of each PCIe read
    # transaction.  The default (256–512 B) forces many small round-trips
    # per DMA transfer.  Setting 4096 B reduces transaction overhead and
    # raises sustained H2D throughput by ~2–5 GB/s on PCIe 4.0 x16 links.
    # Hardware-agnostic: reads CAP_EXP+8.w, clears bits 14:12, sets 101b.
    if command -v setpci &>/dev/null && [[ -n "$gpu_pci_rt" ]]; then
        local mrrs_cur
        mrrs_cur=$(setpci -s "$gpu_pci_rt" CAP_EXP+8.w 2>/dev/null)
        if [[ -n "$mrrs_cur" ]]; then
            local mrrs_new
            mrrs_new=$(printf '%04x' $(( (16#${mrrs_cur} & 0x8FFF) | 0x5000 )))
            setpci -s "$gpu_pci_rt" CAP_EXP+8.w="${mrrs_new}" 2>/dev/null && \
                info "PCIe MRRS         : 4096 B (was 0x${mrrs_cur} → 0x${mrrs_new})" || \
                gb_warn_ui "PCIe MRRS         : setpci write failed (non-fatal)"
        fi
    fi

    # ── THP → always (GreenBoost 2 MB hugepage pool requires it) ─────────
    echo always > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
    info "THP               : always"

    # ── vm.swappiness → 10 (prefer RAM; only page to NVMe under pressure) ─
    sysctl -qw vm.swappiness=10
    info "vm.swappiness     : 10 (was default 60)"

    # ── vm.dirty_ratio / background_ratio (reduce write stalls) ──────────
    sysctl -qw vm.dirty_ratio=40
    sysctl -qw vm.dirty_background_ratio=10
    info "vm.dirty_ratio    : 40 / background: 10"

    echo ""
    gb_ok "Runtime tuning applied (active until next reboot)"

    # ── Persist settings unconditionally ──────────────────────────────────
    _tune_persist_sysctl
    _tune_persist_nvme "$ra_kb"
    gb_ok "Settings saved — will apply automatically on every boot"
}

# Write sysctl tunables to the drop-in file (also called by tune-sysctl)
_tune_persist_sysctl() {
    local conf="/etc/sysctl.d/99-zzz-greenboost.conf"
    cat > "$conf" << 'SCTL'
# GreenBoost — persistent sysctl tunables
# Written by: greenboost_setup.sh tune
vm.swappiness            = 10
vm.dirty_ratio           = 40
vm.dirty_background_ratio = 10
SCTL
    sysctl -p "$conf" -q 2>/dev/null || true
    gb_ok "sysctl: wrote $conf"
}

# Write NVMe scheduler + THP udev rule and sysfs.d conf
_tune_persist_nvme() {
    local ra_kb="${1:-4096}"

    # udev rule — fires on every NVMe block device add/change
    local udev_rule="/etc/udev/rules.d/99-nvme-greenboost.rules"
    cat > "$udev_rule" << UDEV
# GreenBoost — NVMe tuning applied at boot via udev
ACTION=="add|change", KERNEL=="nvme[0-9]n[0-9]", ENV{DEVTYPE}=="disk", ATTR{queue/scheduler}="none", ATTR{queue/read_ahead_kb}="${ra_kb}", ATTR{queue/nr_requests}="1023"
UDEV
    udevadm control --reload-rules 2>/dev/null || true
    gb_ok "udev rule: wrote $udev_rule"

    # THP via sysfs.d (applied by sysfsutils at boot)
    local sysfs_conf="/etc/sysfs.d/greenboost-hugepages.conf"
    cat > "$sysfs_conf" << 'SYSFS'
# GreenBoost — Transparent Huge Pages must stay 'always' for T2 pool
kernel/mm/transparent_hugepage/enabled = always
SYSFS
    gb_ok "sysfs.d: wrote $sysfs_conf (THP=always on boot)"
}

# ---- tune-grub ---------------------------------------------------------
# Validate each candidate flag against the kernel config, then apply.
# Strategy: read-only checks first; only write GRUB if all checks pass.
# Security mitigations are NEVER touched.

_grub_has()  { grep -qw "$1" /proc/cmdline; }
_kcfg_has()  { grep -q "^${1}=y" /boot/config-"$(uname -r)" 2>/dev/null; }

_grub_check_flag() {
    local flag="$1" desc="$2" kcfg="$3"
    if _grub_has "$flag"; then
        info "  [skip]    $flag  — already active"
        return 1
    fi
    if [[ -n "$kcfg" ]] && ! _kcfg_has "$kcfg"; then
        warn "  [skip]    $flag  — kernel not built with $kcfg"
        return 1
    fi
    info "  [add]     $flag  — $desc"
    return 0
}

cmd_tune_grub() {
    need_root tune-grub
    detect_hardware

    local grub_file="/etc/default/grub"
    local kver; kver="$(uname -r)"
    local kcfg="/boot/config-${kver}"

    [[ -f "$grub_file" ]] || die "GRUB config not found: $grub_file"

    info "Validating GRUB flags for: ${CPU_NAME} | ${GPU_NAME} | ${RAM_TYPE}-${RAM_SPEED_MT} MT/s | PCIe Gen ${PCIE_GEN} ${PCIE_WIDTH}"
    info "Kernel: $kver"
    echo ""

    # Read current GRUB cmdline value
    local current_line
    current_line=$(grep '^GRUB_CMDLINE_LINUX_DEFAULT=' "$grub_file" | head -1 | sed 's/^GRUB_CMDLINE_LINUX_DEFAULT=//;s/^"//;s/"$//')

    local new_flags=""

    # ── Flag: transparent_hugepage=always ───────────────────────────────
    # GreenBoost T2 pool allocates 2MB compound pages. THP=always ensures
    # the kernel tries to satisfy those allocs from huge pages at boot.
    # Currently set to 'madvise' in GRUB — change to 'always'.
    if _kcfg_has "CONFIG_TRANSPARENT_HUGEPAGE"; then
        if echo "$current_line" | grep -q "transparent_hugepage=madvise"; then
            info "  [fix]     transparent_hugepage=madvise -> always  (GreenBoost T2 hugepage pool)"
            current_line="${current_line/transparent_hugepage=madvise/transparent_hugepage=always}"
        elif ! _grub_has "transparent_hugepage=always"; then
            info "  [add]     transparent_hugepage=always  (GreenBoost T2 hugepage pool)"
            new_flags="$new_flags transparent_hugepage=always"
        else
            info "  [skip]    transparent_hugepage=always  — already active"
        fi
    fi

    # ── Flag: skew_tick=1 ───────────────────────────────────────────────
    # Staggers per-CPU timer ticks on the i9-14900KF hybrid topology
    # (8 P-cores + 16 E-cores). Reduces lock contention when all 32 CPUs
    # fire timer interrupts simultaneously.
    # Runtime test: always safe — no kernel config dependency.
    _grub_check_flag "skew_tick=1" \
        "stagger timer ticks — reduces lock contention on hybrid P/E cores" "" \
        && new_flags="$new_flags skew_tick=1"

    # ── Flag: rcu_nocbs=<ecores> ─────────────────────────────────────────
    # Offloads RCU (Read-Copy-Update) callback processing to E-cores or
    # non-golden CPUs, freeing the high-frequency P-cores for inference.
    # Range is derived from detected CPU topology (GB_PCORES_MAX).
    if [[ $GB_PCORES_ONLY -eq 1 ]]; then
        local ecores_start=$(( GB_PCORES_MAX + 1 ))
        local ecores_end=$(( $(nproc) - 1 ))
        if [[ $ecores_start -le $ecores_end ]]; then
            _grub_check_flag "rcu_nocbs=${ecores_start}-${ecores_end}" \
                "offload RCU callbacks to E-cores (CPU ${ecores_start}-${ecores_end}), freeing P-cores for inference" \
                "CONFIG_RCU_NOCB_CPU" \
                && new_flags="$new_flags rcu_nocbs=${ecores_start}-${ecores_end}"
        fi
    fi

    # ── Flag: nohz_full=<golden> ─────────────────────────────────────────
    # Makes the highest-frequency cores tick-less when they have exactly
    # one runnable thread. Eliminates timer interrupts during dense matrix
    # multiplications — reduces LLM token latency.
    # Range derived from detected golden-core topology (GB_GOLDEN_MIN/MAX).
    if [[ $GB_PCORES_ONLY -eq 1 && $GB_GOLDEN_MIN -lt $GB_GOLDEN_MAX ]]; then
        _grub_check_flag "nohz_full=${GB_GOLDEN_MIN}-${GB_GOLDEN_MAX}" \
            "tick-less golden P-cores (CPU ${GB_GOLDEN_MIN}-${GB_GOLDEN_MAX}) during single-thread inference" \
            "CONFIG_NO_HZ_FULL" \
            && new_flags="$new_flags nohz_full=${GB_GOLDEN_MIN}-${GB_GOLDEN_MAX}"
    fi

    # ── Flag: numa_balancing=disable ────────────────────────────────────
    # This workstation has a single NUMA node (all CPUs on node 0).
    # The kernel's automatic NUMA balancing task wastes cycles scanning
    # pages that will never need to move. Already disabled at runtime
    # via sysctl, this makes it persistent across reboots.
    _grub_check_flag "numa_balancing=disable" \
        "single NUMA node — disable page-migration scanning overhead" "" \
        && new_flags="$new_flags numa_balancing=disable"

    # ── Flag: workqueue.power_efficient=0 ──────────────────────────────
    # Kernel workqueues (DMA, NVMe completion, etc.) use power-efficient
    # mode by default, routing work to whichever CPU happens to be awake.
    # Disabling this routes workqueue items to the fastest available CPU
    # (P-core), which matters for DMA-BUF completion and NVMe IRQ paths.
    _grub_check_flag "workqueue.power_efficient=0" \
        "route kernel workqueues to P-cores instead of any-idle CPU" "" \
        && new_flags="$new_flags workqueue.power_efficient=0"

    # ── Fix: deduplicate nvidia-drm.modeset=1 ──────────────────────────
    local count; count=$(echo "$current_line" | grep -o "nvidia-drm.modeset=1" | wc -l)
    if [[ "$count" -gt 1 ]]; then
        info "  [fix]     nvidia-drm.modeset=1 appears ${count}× — deduplicating"
        # Remove all occurrences then add one back
        current_line=$(echo "$current_line" | sed 's/nvidia-drm\.modeset=1//g' | tr -s ' ')
        current_line="$current_line nvidia-drm.modeset=1"
    fi

    # ── Build new cmdline ───────────────────────────────────────────────
    local new_line="${current_line}${new_flags}"
    # Normalise multiple spaces
    new_line=$(echo "$new_line" | tr -s ' ' | sed 's/^ //;s/ $//')

    if [[ "$new_line" == "$current_line" ]] && [[ -z "$new_flags" ]]; then
        info ""
        info "GRUB is already fully optimised — nothing to change."
        return 0
    fi

    echo ""
    info "Current GRUB cmdline:"
    echo "  $current_line" | fold -s -w 100 | sed 's/^/  /'
    echo ""
    info "New GRUB cmdline:"
    echo "  $new_line" | fold -s -w 100 | sed 's/^/  /'
    echo ""

    # ── Backup + write ──────────────────────────────────────────────────
    local bak="${grub_file}.bak.$(date +%Y%m%d_%H%M%S)"
    cp "$grub_file" "$bak"
    info "Backup saved: $bak"

    # Replace the GRUB_CMDLINE_LINUX_DEFAULT line
    sed -i "s|^GRUB_CMDLINE_LINUX_DEFAULT=.*|GRUB_CMDLINE_LINUX_DEFAULT=\"${new_line}\"|" "$grub_file"

    info "Running update-grub..."
    update-grub 2>&1 | grep -v "^$" | sed 's/^/  /'

    echo ""
    info "GRUB updated. Changes take effect on next reboot."
    warn "Reboot when ready: sudo reboot"
}

# ---- tune-sysctl -------------------------------------------------------
# Consolidate conflicting sysctl files and add missing compute settings.
# Writes /etc/sysctl.d/99-zzz-greenboost.conf — loaded last, wins all
# conflicts. Previous files are left untouched (history/documentation).

cmd_tune_sysctl() {
    need_root tune-sysctl
    detect_hardware

    local dest="/etc/sysctl.d/99-zzz-greenboost.conf"

    info "Writing definitive sysctl config: $dest"
    info "Hardware: ${CPU_NAME} | ${GPU_NAME} | ${RAM_TYPE}-${RAM_SPEED_MT} MT/s | PCIe Gen ${PCIE_GEN} ${PCIE_WIDTH} (~${PCIE_BW_GBS} GB/s) | ${NVME_SIZE_GB} GB NVMe"
    info "This file loads last (99-zzz) and wins over all conflicting files."
    echo ""

    # Show conflicts found
    info "Conflicts resolved:"
    info "  vm.swappiness       : multiple files set 10/20 → final: 10"
    info "  vm.dirty_ratio      : 15 vs 40 → final: 40"
    info "  vm.dirty_background_ratio: 5 vs 10 → final: 10"
    info "  kernel.sched_autogroup_enabled: 1 → 0 (bad for compute, groups by session)"
    info "New settings added:"
    if [[ -e /proc/sys/kernel/sched_migration_cost_ns ]]; then
        info "  kernel.sched_migration_cost_ns: 5000000 (5ms — keep threads on P-cores)"
        info "  kernel.sched_min_granularity_ns: 10000000 (10ms — better for large tasks)"
        info "  kernel.sched_wakeup_granularity_ns: 15000000 (reduces spurious wakeups)"
    else
        info "  CFS sched knobs: skipped (kernel $(uname -r) uses EEVDF, not CFS)"
    fi
    echo ""

    # Write the header line with detected hardware (variables don't expand in 'HEREDOC')
    printf '# GreenBoost v2.8.2 — Definitive sysctl config\n' > "$dest"
    printf '# Hardware: %s | %s | %s-%s MT/s | PCIe Gen %s %s (~%s GB/s) | %s GB NVMe\n' \
        "${CPU_NAME}" "${GPU_NAME}" "${RAM_TYPE}" "${RAM_SPEED_MT}" \
        "${PCIE_GEN}" "${PCIE_WIDTH}" "${PCIE_BW_GBS}" "${NVME_SIZE_GB}" >> "$dest"
    printf '# Loaded last (99-zzz) — wins all conflicts with earlier sysctl.d files.\n' >> "$dest"
    printf '# Do NOT edit other sysctl.d files; make changes here instead.\n' >> "$dest"

    # vm.nr_overcommit_hugepages: scale to T2 pool size so the kernel can back
    # the full pool with 2 MB THP pages.  GB_VIRT GB × 512 pages/GB = pages.
    local overcommit_hp=$(( GB_VIRT * 512 ))

    cat >> "$dest" << 'SYSCTL_EOF'

# ── Swap / memory pressure ───────────────────────────────────────────────
# Keep LLM weights in system RAM (T2); only spill to NVMe (T3) under real pressure.
vm.swappiness = 10

# ── Write-back (Samsung 990 EVO Plus sustains 6,300 MB/s writes) ─────────
# Allow up to 40% dirty pages before throttling writes (~25 GB at 64 GB RAM).
# Background flush at 10% (~6.4 GB) — keeps NVMe busy without stalling allocs.
vm.dirty_ratio = 40
vm.dirty_background_ratio = 10
vm.dirty_writeback_centisecs = 1500
vm.dirty_expire_centisecs = 3000

# ── Memory allocation ─────────────────────────────────────────────────────
# LLM frameworks (llama.cpp, PyTorch, JAX) mmap model files and pre-reserve
# large virtual address ranges. Without overcommit the kernel rejects these.
vm.overcommit_memory = 1

# Max VMA regions: transformer models (70B+) split across thousands of mmap
# file segments. Default 65530 is too low; 2M covers any realistic case.
vm.max_map_count = 2147483642

# Always keep 512 MB free — prevents latency spikes under allocation storms.
vm.min_free_kbytes = 524288

# Proactive compaction: GreenBoost T2 needs contiguous 2 MB hugepage ranges.
# Value 20 = moderate background compaction (0=off, 100=aggressive).
vm.compaction_proactiveness = 20

# Keep inode/dentry caches alive — LLM loaders open thousands of weight files.
vm.vfs_cache_pressure = 50
SYSCTL_EOF

    # Overcommit hugepage pool: sized to cover the full T2 System DDR pool.
    # Formula: GB_VIRT × 512 = number of 2 MB pages.  RAM speed ${RAM_SPEED_MT} MT/s.
    printf '# Overcommit hugepage pool: %d × 2 MB = %d GB — covers T2 pool (%d GB, %s-%s MT/s)\n' \
        "$overcommit_hp" "$(( overcommit_hp * 2 / 1024 ))" "$GB_VIRT" "$RAM_TYPE" "$RAM_SPEED_MT" >> "$dest"
    printf 'vm.nr_overcommit_hugepages = %d\n' "$overcommit_hp" >> "$dest"

    cat >> "$dest" << 'SYSCTL_EOF'

# Disable zone reclaim: single NUMA node — cross-zone reclaim wastes cycles.
vm.zone_reclaim_mode = 0

# ── CPU scheduler (i9-14900KF P-core / E-core hybrid) ────────────────────
# Disable session-based task grouping. sched_autogroup groups shell tasks
# together which is good for desktop but HURTS inference: Ollama worker
# threads (long-running compute) compete with short interactive tasks for
# scheduler time-slices in the same group.
kernel.sched_autogroup_enabled = 0

# CFS scheduler knobs below (sched_migration_cost_ns, sched_min_granularity_ns,
# sched_wakeup_granularity_ns) are appended conditionally after this heredoc:
# they were removed in kernel 6.6 when EEVDF replaced CFS and do not exist
# on kernel 7.0+. Attempting to set them produces harmless but noisy errors.

# ── NUMA ──────────────────────────────────────────────────────────────────
# Single-socket i9-14900KF — all CPUs are on NUMA node 0. Automatic NUMA
# balancing scans pages and attempts cross-node migrations that will never
# happen. Disable to remove the page-scanning overhead.
kernel.numa_balancing = 0

# ── File system ───────────────────────────────────────────────────────────
# GGUF model files (70B+ = thousands of weight tensors) require many open
# file descriptors and inotify watches during model loading.
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288
fs.inotify.max_user_instances = 1024

# ── Network (Ollama API / distributed inference) ──────────────────────────
# Large buffers for Ollama HTTP streaming API and any future multi-GPU setup.
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_fastopen = 3

# ── Perf / profiling access ───────────────────────────────────────────────
# Allow nsys / perf / CUDA Nsight without sudo (needed for GPU profiling).
kernel.perf_event_paranoid = 1
kernel.kptr_restrict = 0
SYSCTL_EOF

    # CFS scheduler params: only present on kernels < 6.6 (CFS).
    # Kernel 6.6+ uses EEVDF — these knobs were removed entirely.
    _sysctl_if_exists() {
        local key="$1" val="$2" proc_path="$3"
        if [[ -e "$proc_path" ]]; then
            printf '\n# CFS scheduler (kernel < 6.6 only)\n%s = %s\n' "$key" "$val" >> "$dest"
        fi
    }
    _sysctl_if_exists kernel.sched_migration_cost_ns   5000000  /proc/sys/kernel/sched_migration_cost_ns
    _sysctl_if_exists kernel.sched_min_granularity_ns  10000000 /proc/sys/kernel/sched_min_granularity_ns
    _sysctl_if_exists kernel.sched_wakeup_granularity_ns 15000000 /proc/sys/kernel/sched_wakeup_granularity_ns

    sysctl -p "$dest" 2>&1 | grep -v "^$" | sed 's/^/  /' || true
    echo ""
    info "sysctl applied and persistent (survives reboot via $dest)."
}

# ---- tune-libs ---------------------------------------------------------
# Install missing libraries and kernel modules for AI/compute workloads.
# All packages chosen for AVX2/FMA/VNNI capabilities on i9-14900KF.

cmd_tune_libs() {
    need_root tune-libs
    detect_hardware

    info "Installing missing AI/compute libraries for: ${CPU_NAME} | ${GPU_NAME}"
    echo ""

    # ── APT packages ──────────────────────────────────────────────────────
    local pkgs=(
        # BLAS/LAPACK — OpenBLAS compiled with AVX2/FMA or equivalent
        libopenblas-dev
        libblas-dev
        liblapack-dev

        # OpenMP — multi-threaded CPU inference (llama.cpp uses this heavily)
        libomp-dev

        # hwloc — hardware topology library used by Ollama/llama.cpp for
        # CPU pinning; without it Ollama uses a generic thread affinity model
        hwloc
        libhwloc-dev

        # libnuma — NUMA-aware memory allocation (single node but still used
        # by CUDA and some ML runtimes for memory locality hints)
        libnuma-dev

        # OpenCL — GPU compute via OpenCL API (some inference backends use it)
        ocl-icd-opencl-dev

        # nvtop — real-time GPU + CPU monitor (shows all 3 tiers at a glance)
        nvtop
    )

    # CPU frequency tools — package names differ between Debian and Ubuntu/others
    local os_id
    os_id=$(grep -oP '^ID=\K.*' /etc/os-release | tr -d '"')
    if [[ "$os_id" == "debian" ]]; then
        pkgs+=(linux-cpupower linux-perf psmisc)
    else
        pkgs+=(cpufrequtils linux-tools-generic psmisc)
    fi

    # CPU vendor-specific microcode
    local cpu_vendor
    cpu_vendor=$(grep -m1 "vendor_id" /proc/cpuinfo | awk '{print $3}')
    if [[ "$cpu_vendor" == "GenuineIntel" ]]; then
        pkgs+=(intel-microcode)
        info "CPU vendor: Intel — adding intel-microcode"
    elif [[ "$cpu_vendor" == "AuthenticAMD" ]]; then
        pkgs+=(amd64-microcode)
        info "CPU vendor: AMD — adding amd64-microcode"
    fi

    info "Packages to install:"
    local to_install=()
    for pkg in "${pkgs[@]}"; do
        if dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
            info "  [ok]      $pkg"
        else
            info "  [install] $pkg"
            to_install+=("$pkg")
        fi
    done
    echo ""

    if [[ ${#to_install[@]} -eq 0 ]]; then
        info "All packages already installed."
    else
        apt-get install -y "${to_install[@]}" 2>&1 | tail -5
        info "Packages installed."
    fi

    echo ""

    # ── Kernel modules ────────────────────────────────────────────────────
    info "Kernel modules:"

    # cpuid — lets userspace read CPUID leaves directly. Used by turbostat,
    # CUDA diagnostics, and intel-microcode update verification.
    if lsmod | grep -q "^cpuid "; then
        info "  [ok]      cpuid  (loaded)"
    else
        modprobe cpuid && info "  [loaded]  cpuid" || warn "  cpuid: modprobe failed"
    fi

    # Ensure cpuid + msr auto-load at boot
    local ml_conf="/etc/modules-load.d/ai-workstation.conf"
    if ! grep -q "^cpuid" "$ml_conf" 2>/dev/null; then
        echo "cpuid" >> "$ml_conf"
        info "  [add]     cpuid -> $ml_conf (auto-load at boot)"
    else
        info "  [ok]      cpuid already in $ml_conf"
    fi

    echo ""

    # ── OpenBLAS CPU target selection ─────────────────────────────────────
    # Make sure the system BLAS points to OpenBLAS (AVX2/FMA optimised)
    # rather than the reference BLAS implementation.
    if command -v update-alternatives &>/dev/null && dpkg -l libopenblas-dev 2>/dev/null | grep -q "^ii"; then
        update-alternatives --set libblas.so.3-x86_64-linux-gnu \
            /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 2>/dev/null \
            && info "BLAS alternative: set to OpenBLAS (AVX2/FMA)" \
            || info "BLAS alternative: already set or path differs — check manually"
    fi

    echo ""
    info "tune-libs complete."
    info "  Turbostat (P/E core monitoring): sudo turbostat --quiet --Summary"
    info "  nvtop (GPU + CPU):               nvtop"
    local os_id_tip
    os_id_tip=$(grep -oP '^ID=\K.*' /etc/os-release | tr -d '"')
    if [[ "$os_id_tip" == "debian" ]]; then
        info "  CPU frequency info:              cpupower frequency-info"
    else
        info "  CPU frequency info:              cpufreq-info"
    fi
}

# ---- tune-all ----------------------------------------------------------

cmd_tune_all() {
    need_root tune-all
    info "Running full system tuning for GreenBoost v2.8.2..."
    echo ""
    cmd_tune
    echo ""
    cmd_tune_sysctl
    echo ""
    cmd_tune_grub
    echo ""
    cmd_tune_libs
    echo ""
    info "All tuning complete."
    info "Reboot to activate GRUB changes: sudo reboot"
}

cmd_clear_logs() {
    need_root "clear logs"
    gb_header
    echo -e "  ${C_CYAN}${C_BOLD}Clear All GreenBoost Logs${C_RESET}"
    echo -e "  ${C_DIM}Clears kernel ring buffer, journal, GreenBoost log files, Proton game logs, and Wine coredumps.${C_RESET}"
    echo -e "  ${C_DIM}Steam's own logs (compat_log.txt, pressure-vessel) are Steam-managed and are not deleted.${C_RESET}"
    echo -e ""
    local choice
    read -r -p "  Clear all GreenBoost-related logs now? [Y/n] " choice
    if [[ "$choice" =~ ^[Nn] ]]; then
        gb_info "Skipping log cleanup."
        return
    fi

    local _real_user="${SUDO_USER:-$USER}"
    local _real_home
    _real_home="$(getent passwd "$_real_user" | cut -d: -f6)"
    [[ -z "$_real_home" ]] && _real_home="$HOME"

    dmesg -c > /dev/null 2>&1 || true
    journalctl --rotate > /dev/null 2>&1 || true
    journalctl --vacuum-time=1s > /dev/null 2>&1 || true
    rm -rf /var/log/greenboost/* 2>/dev/null || true
    rm -f /tmp/greenboost*.log 2>/dev/null || true
    local _gbproton_dir="$_real_home/.local/share/greenboost/proton-logs"
    rm -f "$_gbproton_dir"/steam-*.log 2>/dev/null || true
    rm -f "$_real_home"/steam-*.log 2>/dev/null || true
    if command -v coredumpctl &>/dev/null; then
        coredumpctl delete wine64 2>/dev/null || true
        coredumpctl delete wineserver 2>/dev/null || true
        coredumpctl delete winedevice 2>/dev/null || true
    fi
    gb_ok "All GreenBoost logs (dmesg, journal, /var/log/greenboost, Proton, Wine coredumps) cleared."
}

cmd_clear_proton_logs() {
    gb_header
    echo -e "  ${C_CYAN}${C_BOLD}Clear Proton Logs${C_RESET}"
    echo -e "  ${C_DIM}Clears GreenBoost Proton game logs and Wine coredumps. No root required.${C_RESET}"
    echo -e ""
    local choice
    read -r -p "  Clear Proton logs now? [Y/n] " choice
    if [[ "$choice" =~ ^[Nn] ]]; then
        gb_info "Skipping."
        return
    fi
    local _gbproton_dir="$HOME/.local/share/greenboost/proton-logs"
    rm -f "$_gbproton_dir"/steam-*.log 2>/dev/null || true
    rm -f "$HOME"/steam-*.log 2>/dev/null || true
    if command -v coredumpctl &>/dev/null; then
        coredumpctl delete wine64 2>/dev/null || true
        coredumpctl delete wineserver 2>/dev/null || true
        coredumpctl delete winedevice 2>/dev/null || true
    fi
    gb_ok "Proton logs (${_gbproton_dir}/, ~/steam-*.log, Wine coredumps) cleared."
}

cmd_clear_inference_logs() {
    need_root "clear inference-logs"
    gb_header
    echo -e "  ${C_CYAN}${C_BOLD}Clear Inference Logs${C_RESET}"
    echo -e "  ${C_DIM}Clears kernel ring buffer, systemd journal, /var/log/greenboost, and /tmp/greenboost*.log.${C_RESET}"
    echo -e ""
    local choice
    read -r -p "  Clear inference logs now? [Y/n] " choice
    if [[ "$choice" =~ ^[Nn] ]]; then
        gb_info "Skipping."
        return
    fi
    dmesg -c > /dev/null 2>&1 || true
    journalctl --rotate > /dev/null 2>&1 || true
    journalctl --vacuum-time=1s > /dev/null 2>&1 || true
    rm -rf /var/log/greenboost/* 2>/dev/null || true
    rm -f /tmp/greenboost*.log 2>/dev/null || true
    gb_ok "Inference logs (dmesg, journal, /var/log/greenboost, /tmp/greenboost*.log) cleared."
}

# Backward-compat alias — kept so existing scripts/bookmarks still work.
cmd_clean_logs() { cmd_clear_logs; }

# ════════════════════════════════════════════════════════════════════════
# _logs_llm — compact, token-efficient log output for LLM/AI tools.
# No ANSI colors, no human-readable decoration, deduplicated lines.
# Format: key=value header, labeled sections, [Nx] for repeated lines.
_logs_llm() {
    local _strip='s/\x1b\[[0-9;]*[mKHJsu]//g'

    # Header
    local _mod="MISSING"
    lsmod 2>/dev/null | grep -q '^greenboost' && _mod="loaded"
    local _layer="MISSING"
    [[ -f /etc/vulkan/implicit_layer.d/VkLayer_greenboost.json ]] && _layer="ok"
    local _proton="MISSING"
    for _pp in \
        "$HOME/.steam/root/compatibilitytools.d/greenboost-proton-wayland" \
        "$HOME/.local/share/Steam/compatibilitytools.d/greenboost-proton-wayland" \
        "$HOME/.steam/root/compatibilitytools.d/greenboost-proton" \
        "$HOME/.local/share/Steam/compatibilitytools.d/greenboost-proton"
    do [[ -f "$_pp/proton" ]] && _proton="installed" && break; done
    local _vram="?"
    _vram=$(journalctl -u ollama --no-pager -q -n 50 2>/dev/null \
        | grep -oP 'total="\K[^"]+' | tail -1)
    [[ -z "$_vram" ]] && _vram="?"
    echo "gb_logs v=${GB_VERSION} ts=$(date '+%Y-%m-%dT%H:%M') mod=${_mod} vram=${_vram} layer=${_layer} proton=${_proton}"

    # Helper: dedup + limit a block of text, keep only actionable lines
    _llm_section() {
        local _label="$1" _text="$2" _limit="${3:-15}"
        local _filtered
        _filtered=$(printf '%s\n' "$_text" \
            | grep -iE 'err|warn|fail|oom|crash|denied|evict|tier|transition|alloc|free|loaded|unload|timeout|reset|panic|CRITICAL' \
            | sed "$_strip" \
            | sed 's/^[[:space:]]*//' \
            | sort | uniq -c | sort -rn \
            | awk '{c=$1; $1=""; sub(/^ /,""); if(c>1) printf "[%dx] %s\n",c,$0; else print $0}' \
            | head -"$_limit")
        local _n=0; [[ -n "$_filtered" ]] && _n=$(printf '%s\n' "$_filtered" | wc -l)
        echo "[${_label}] ${_n} actionable events"
        [[ -n "$_filtered" ]] && printf '%s\n' "$_filtered" | sed 's/^/  /'
    }

    # 1. Kernel
    local _kern; _kern=$(gather_flow_events 50 2>/dev/null || \
        dmesg 2>/dev/null | grep -E 'greenboost' | tail -50)
    _llm_section "kernel" "$_kern"

    # 2. Services — inference only (ollama, idle-reclaim, sentinel, recovery)
    local _svc
    _svc=$(journalctl --since "1 hour ago" --no-pager -q \
        -u ollama -u greenboost-idle-reclaim \
        -u greenboost-sentinel -u greenboost-recovery 2>/dev/null | tail -60)
    _llm_section "services" "$_svc"

    # 3. Vulkan layer
    local _vkev; _vkev=$(gather_vulkan_events 30 2>/dev/null || \
        journalctl --since "1 hour ago" --no-pager -q 2>/dev/null \
            | grep 'VK_LAYER_GREENBOOST' | tail -30)
    _llm_section "vulkan" "$_vkev"

    # 4. Proton / VKD3D
    query_dx12_status 2>/dev/null || true
    local _proton_ev; _proton_ev=$(gather_proton_log_events 30 2>/dev/null)
    _llm_section "proton" "$_proton_ev"

    # 5. AppArmor
    local _aa
    _aa=$(journalctl -k --no-pager -q -n 200 2>/dev/null \
        | grep -iE 'apparmor="DENIED".*greenboost|greenboost.*apparmor="DENIED"' | tail -10)
    _llm_section "apparmor" "$_aa"

    # Diagnostic summary
    local _errs=() _warns=() _ok=()
    [[ "$_mod" == "loaded" ]] && _ok+=("module=loaded") || _errs+=("module=MISSING")
    [[ "$_layer" == "ok" ]]   && _ok+=("vulkan_layer=ok") || _warns+=("vulkan_layer=MISSING")
    [[ "$_proton" == "installed" ]] && _ok+=("proton=installed") || _warns+=("proton=MISSING")
    [[ -z "${WAYLAND_DISPLAY:-}" ]] && _warns+=("WAYLAND_DISPLAY=unset")
    local _ec=${#_errs[@]} _wc=${#_warns[@]} _oc=${#_ok[@]}
    echo "[diag] errors=${_ec} warns=${_wc} ok=${_oc}"
    for _e in "${_errs[@]}";  do echo "  ERR: ${_e}"; done
    for _w in "${_warns[@]}"; do echo "  WARN: ${_w}"; done
    for _o in "${_ok[@]}";    do echo "  OK: ${_o}"; done
}

cmd_logs() {
    local _llm_mode=0
    for _a in "$@"; do [[ "$_a" == "--llm" ]] && _llm_mode=1; done
    if (( _llm_mode )); then _logs_llm; return; fi

    # ── Compact status header (LLM-friendly: no banner, info-dense 1-liner) ──
    local _mod_status="MISSING"
    lsmod 2>/dev/null | grep -q '^greenboost' && _mod_status="loaded"
    local _vram_str="?"
    _vram_str=$(journalctl -u ollama --no-pager -q -n 50 2>/dev/null \
        | grep -oP 'total="\K[^"]+' | tail -1)
    [[ -z "$_vram_str" ]] && _vram_str="?"
    local _layer_status="MISSING"
    [[ -f /etc/vulkan/implicit_layer.d/VkLayer_greenboost.json ]] && _layer_status="ok"
    local _proton_install_status="MISSING"
    for _pp in \
        "$HOME/.steam/root/compatibilitytools.d/greenboost-proton-wayland" \
        "$HOME/.local/share/Steam/compatibilitytools.d/greenboost-proton-wayland" \
        "$HOME/.steam/root/compatibilitytools.d/greenboost-proton" \
        "$HOME/.local/share/Steam/compatibilitytools.d/greenboost-proton"
    do
        [[ -f "$_pp/proton" ]] && _proton_install_status="installed" && break
    done
    local _ts; _ts=$(date '+%Y-%m-%dT%H:%M')
    echo -e ""
    echo -e "  ${C_VIOLET}${C_BOLD}GreenBoost v${GB_VERSION} logs${C_RESET}  ${C_DIM}·  ${_ts}  ·  module:${_mod_status}  vram:${_vram_str}  layer:${_layer_status}  proton:${_proton_install_status}${C_RESET}"
    echo -e ""

    # Track findings for DIAGNOSTIC SUMMARY
    local _diag_errors=() _diag_warns=() _diag_ok=()
    [[ "$_mod_status" == "loaded" ]] \
        && _diag_ok+=("kernel module loaded — VRAM ${_vram_str} reported to apps") \
        || _diag_errors+=("kernel module NOT loaded — no T2/T3 memory available")
    [[ "$_layer_status" == "ok" ]] \
        && _diag_ok+=("Vulkan layer: /etc/vulkan/implicit_layer.d/VkLayer_greenboost.json") \
        || _diag_warns+=("Vulkan layer JSON missing — games won't see inflated virtual VRAM")
    [[ "$_proton_install_status" == "installed" ]] \
        && _diag_ok+=("GreenBoost Proton: compat tool installed") \
        || _diag_warns+=("GreenBoost Proton not found in compatibilitytools.d")
    [[ -z "${WAYLAND_DISPLAY:-}" ]] \
        && _diag_warns+=("WAYLAND_DISPLAY not set — PROTON_ENABLE_WAYLAND=1 may silently fall back to X11")

    # ── 1. Kernel module events ─────────────────────────────────────────────
    local _flow; _flow=$(gather_flow_events 25)
    local _flow_n=0; [[ -n "$_flow" ]] && _flow_n=$(printf '%s\n' "$_flow" | wc -l)
    gb_section "Kernel Module (dmesg)  (${_flow_n} events)"
    if [[ -n "$_flow" ]]; then
        while IFS= read -r _fline; do
            [[ -z "$_fline" ]] && continue
            echo -e "  ${C_DIM}${_fline}${C_RESET}"
        done <<< "$_flow"
    else
        echo -e "  ${C_DIM}(empty — dmesg grep: greenboost tier-transitions)${C_RESET}"
    fi
    echo ""

    # ── 2. GreenBoost service journal ───────────────────────────────────────
    local _svc
    _svc=$(journalctl --since "1 hour ago" --no-pager -q \
        -u ollama -u greenboost-idle-reclaim -u greenboost-shader-boost \
        -u greenboost-sentinel -u greenboost-recovery 2>/dev/null | tail -25)
    local _svc_n=0; [[ -n "$_svc" ]] && _svc_n=$(printf '%s\n' "$_svc" | wc -l)
    gb_section "Services (last 1h)  (${_svc_n} events)"
    if [[ -n "$_svc" ]]; then
        while IFS= read -r _sline; do
            [[ -z "$_sline" ]] && continue
            echo -e "  ${C_GRAY}${_sline}${C_RESET}"
        done <<< "$_svc"
    else
        echo -e "  ${C_DIM}(empty — units: ollama greenboost-idle-reclaim greenboost-shader-boost greenboost-sentinel greenboost-recovery)${C_RESET}"
    fi
    echo ""

    # ── 3. Vulkan layer events ───────────────────────────────────────────────
    local _vkev; _vkev=$(gather_vulkan_events 25)
    local _vk_n=0; [[ -n "$_vkev" ]] && _vk_n=$(printf '%s\n' "$_vkev" | wc -l)
    gb_section "Vulkan Layer (VK_LAYER_GREENBOOST)  (${_vk_n} events)"
    if [[ -n "$_vkev" ]]; then
        while IFS= read -r _vkline; do
            [[ -z "$_vkline" ]] && continue
            format_vulkan_event "$_vkline"
        done <<< "$_vkev"
    else
        echo -e "  ${C_DIM}(empty — checked: journalctl VK_LAYER_GREENBOOST, /var/log/syslog)${C_RESET}"
        echo -e "  ${C_DIM}(also checked: ~/.local/share/greenboost/proton-logs/vulkan-layer.log)${C_RESET}"
        _diag_warns+=("no Vulkan layer events — GreenBoost Proton may not be selected for this game")
    fi
    echo ""

    # ── 4. Proton / VKD3D log events ────────────────────────────────────────
    # Populate DX12_APPID so gather_proton_log_events can locate steam-<id>.log
    query_dx12_status
    local _proton; _proton=$(gather_proton_log_events 20)
    local _proton_n=0; [[ -n "$_proton" ]] && _proton_n=$(printf '%s\n' "$_proton" | wc -l)
    local _gbdir="$HOME/.local/share/greenboost/proton-logs"
    gb_section "Proton / VKD3D Logs  (${_proton_n} events)"
    if [[ -n "$_proton" ]]; then
        while IFS= read -r _pline; do
            [[ -z "$_pline" ]] && continue
            format_proton_event "$_pline"
        done <<< "$_proton"
    else
        echo -e "  ${C_DIM}(empty — checked: ${_gbdir}/ ~/steam-*.log PROTON_LOG_DIR)${C_RESET}"
        _diag_warns+=("no Proton log files found — game did not reach Wine logging stage")
    fi
    echo ""

    # ── 5. Steam client logs ─────────────────────────────────────────────────
    local _stcl; _stcl=$(gather_steam_client_logs 15)
    local _stcl_n=0; [[ -n "$_stcl" ]] && _stcl_n=$(printf '%s\n' "$_stcl" | wc -l)
    gb_section "Steam Client Logs  (${_stcl_n} events)"
    if [[ -n "$_stcl" ]]; then
        while IFS= read -r _sl; do
            [[ -z "$_sl" ]] && continue
            if printf '%s' "$_sl" | grep -qiE 'error|exception|traceback|fail|crash'; then
                echo -e "  ${C_RED}✗${C_RESET}  ${C_RED}${_sl:0:140}${C_RESET}"
                _diag_errors+=("steam: ${_sl:0:100}")
            else
                echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}${_sl:0:140}${C_RESET}"
            fi
        done <<< "$_stcl"
    else
        echo -e "  ${C_DIM}(empty — checked: ~/.local/share/Steam/logs/compat_log.txt content_log.txt ~/.steam/steam.log)${C_RESET}"
    fi
    echo ""

    # ── 6. Pressure vessel / SteamLinuxRuntime ───────────────────────────────
    local _pvl; _pvl=$(gather_pressure_vessel_logs 10)
    local _pvl_n=0; [[ -n "$_pvl" ]] && _pvl_n=$(printf '%s\n' "$_pvl" | wc -l)
    gb_section "Pressure Vessel / SteamLinuxRuntime  (${_pvl_n} events)"
    if [[ -n "$_pvl" ]]; then
        while IFS= read -r _pvline; do
            [[ -z "$_pvline" ]] && continue
            if printf '%s' "$_pvline" | grep -qiE 'error|fail|crash'; then
                echo -e "  ${C_RED}✗${C_RESET}  ${C_RED}${_pvline:0:140}${C_RESET}"
                _diag_errors+=("pressure-vessel: ${_pvline:0:100}")
            else
                echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}${_pvline:0:140}${C_RESET}"
            fi
        done <<< "$_pvl"
    else
        echo -e "  ${C_DIM}(empty — checked: ~/.local/share/Steam/steamapps/common/SteamLinuxRuntime_*/var/slr/log/ /tmp/pressure-vessel-*)${C_RESET}"
    fi
    echo ""

    # ── 7. Wine / Proton crashes ─────────────────────────────────────────────
    local _wcl; _wcl=$(gather_wine_crash_logs 8)
    local _wcl_n=0; [[ -n "$_wcl" ]] && _wcl_n=$(printf '%s\n' "$_wcl" | wc -l)
    gb_section "Wine / Proton Crashes  (${_wcl_n} events)"
    if [[ -n "$_wcl" ]]; then
        while IFS= read -r _wcline; do
            [[ -z "$_wcline" ]] && continue
            echo -e "  ${C_RED}✗${C_RESET}  ${C_RED}${_wcline:0:140}${C_RESET}"
        done <<< "$_wcl"
        _diag_errors+=("wine64/wineserver crash detected — see Wine / Proton Crashes section above")
    else
        echo -e "  ${C_DIM}(empty — checked: coredumpctl wine64/wineserver/winedevice, compatdata/*.dmp)${C_RESET}"
    fi
    echo ""

    # ── 8. AppArmor denials ──────────────────────────────────────────────────
    local _aa
    _aa=$(journalctl -k --no-pager -q -n 200 2>/dev/null \
        | grep -iE 'apparmor="DENIED".*greenboost|greenboost.*apparmor="DENIED"' | tail -10)
    if [[ -n "$_aa" ]]; then
        local _aa_n; _aa_n=$(printf '%s\n' "$_aa" | wc -l)
        gb_section "AppArmor Denials  (${_aa_n} events)"
        while IFS= read -r _aaline; do
            [[ -z "$_aaline" ]] && continue
            echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_RED}${_aaline}${C_RESET}"
            _diag_errors+=("apparmor denied: ${_aaline:0:80}")
        done <<< "$_aa"
        echo ""
    fi

    # ── DXR crash detection ──────────────────────────────────────────────────
    # Check whether the last GreenBoost session had DXR active AND the most
    # recently modified game log in compatdata contains the d3d12core NULL ptr
    # crash signature.  If so, surface the GREENBOOST_NO_DXR=1 hint.
    local _dxr_crash_appid=""
    local _last_vkd3d_line
    _last_vkd3d_line=$(grep '\[greenboost-proton\] VKD3D_CONFIG=' \
        "$HOME/.local/share/Steam/logs/console-linux.txt" 2>/dev/null | tail -1)
    if [[ "$_last_vkd3d_line" == *"dxr"* ]]; then
        local _dxr_candidate
        _dxr_candidate=$(find "$HOME/.local/share/Steam/steamapps/compatdata" \
            -maxdepth 8 -name '*.log' 2>/dev/null \
            | xargs -r ls -t 2>/dev/null | head -10 \
            | while IFS= read -r _f; do
                grep -ql 'EXCEPTION_ACCESS_VIOLATION' "$_f" 2>/dev/null \
                    && grep -ql 'd3d12core' "$_f" 2>/dev/null \
                    && echo "$_f" && break
              done)
        if [[ -n "$_dxr_candidate" ]]; then
            _dxr_crash_appid=$(grep -oP 'compatdata/\K[0-9]+' <<< "$_dxr_candidate" | head -1)
        fi
    fi

    # ── Diagnostic Summary ───────────────────────────────────────────────────
    gb_section "Diagnostic Summary"

    if [[ -n "$_dxr_crash_appid" ]]; then
        echo -e "  ${C_RED}✗${C_RESET}  ${C_RED}${C_BOLD}DXR crash detected${C_RESET}  ${C_RED}— NULL pointer in d3d12core.dll (AppID ${_dxr_crash_appid})${C_RESET}"
        echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}This game crashes when DXR (ray tracing) is active via VKD3D-Proton.${C_RESET}"
        echo -e "  ${C_VIOLET}◈${C_RESET}  ${C_GRAY}Fix: add to Steam launch options for this game:${C_RESET}"
        echo -e "  ${C_VIOLET}◈${C_RESET}  ${C_CYAN}${C_BOLD}  GREENBOOST_NO_DXR=1 %command%${C_RESET}"
        echo -e ""
        _diag_errors+=("DXR crash in AppID ${_dxr_crash_appid}: add GREENBOOST_NO_DXR=1 %command% to Steam launch options")
    fi
    for _e in "${_diag_errors[@]}"; do
        echo -e "  ${C_RED}✗${C_RESET}  ${C_RED}${_e}${C_RESET}"
    done
    for _w in "${_diag_warns[@]}"; do
        echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}${_w}${C_RESET}"
    done
    for _o in "${_diag_ok[@]}"; do
        echo -e "  ${C_LIME}✓${C_RESET}  ${C_GRAY}${_o}${C_RESET}"
    done
    if [[ ${#_diag_errors[@]} -eq 0 && ${#_diag_warns[@]} -eq 0 ]]; then
        echo -e "  ${C_LIME}✓${C_RESET}  ${C_GRAY}All checks passed — no errors or warnings detected.${C_RESET}"
    fi
    echo ""
}

# cmd_proton_logs — focused view: Proton/VKD3D, Steam client, pressure-vessel,
# Wine crashes, and DXR crash detection.  No kernel/inference sections.
cmd_proton_logs() {
    local _llm_mode=0
    for _a in "$@"; do [[ "$_a" == "--llm" ]] && _llm_mode=1; done
    if (( _llm_mode )); then
        # LLM mode: gaming-only compact output
        local _strip='s/\x1b\[[0-9;]*[mKHJsu]//g'
        _llm_section() {
            local _label="$1" _text="$2" _limit="${3:-15}"
            local _filtered
            _filtered=$(printf '%s\n' "$_text" \
                | grep -iE 'err|warn|fail|crash|denied|dxr|vkd3d|vulkan|dx12' \
                | sed "$_strip" | sed 's/^[[:space:]]*//' \
                | sort | uniq -c | sort -rn \
                | awk '{c=$1; $1=""; sub(/^ /,""); if(c>1) printf "[%dx] %s\n",c,$0; else print $0}' \
                | head -"$_limit")
            local _n=0; [[ -n "$_filtered" ]] && _n=$(printf '%s\n' "$_filtered" | wc -l)
            echo "[${_label}] ${_n} events"
            [[ -n "$_filtered" ]] && printf '%s\n' "$_filtered" | sed 's/^/  /'
        }
        echo "gb_proton_logs v=${GB_VERSION} ts=$(date '+%Y-%m-%dT%H:%M')"
        query_dx12_status 2>/dev/null || true
        _llm_section "proton_vkd3d" "$(gather_proton_log_events 40 2>/dev/null)"
        _llm_section "vulkan_layer" "$(gather_vulkan_events 20 2>/dev/null)"
        _llm_section "steam_client" "$(gather_steam_client_logs 20 2>/dev/null)"
        _llm_section "pressure_vessel" "$(gather_pressure_vessel_logs 15 2>/dev/null)"
        _llm_section "wine_crashes" "$(gather_wine_crash_logs 10 2>/dev/null)"
        return
    fi

    local _ts; _ts=$(date '+%Y-%m-%dT%H:%M')
    echo -e ""
    echo -e "  ${C_VIOLET}${C_BOLD}GreenBoost Proton logs${C_RESET}  ${C_DIM}·  ${_ts}${C_RESET}"
    echo -e ""

    local _diag_errors=() _diag_warns=() _diag_ok=()

    # ── Proton / VKD3D log events ────────────────────────────────────────
    query_dx12_status
    local _proton; _proton=$(gather_proton_log_events 40)
    local _proton_n=0; [[ -n "$_proton" ]] && _proton_n=$(printf '%s\n' "$_proton" | wc -l)
    local _gbdir="$HOME/.local/share/greenboost/proton-logs"
    gb_section "Proton / VKD3D Logs  (${_proton_n} events)"
    if [[ -n "$_proton" ]]; then
        while IFS= read -r _pline; do
            [[ -z "$_pline" ]] && continue
            format_proton_event "$_pline"
        done <<< "$_proton"
    else
        echo -e "  ${C_DIM}(empty — checked: ${_gbdir}/ ~/steam-*.log PROTON_LOG_DIR)${C_RESET}"
        _diag_warns+=("no Proton log files found — game did not reach Wine logging stage")
    fi
    echo ""

    # ── GreenBoost Vulkan layer events ────────────────────────────────────
    local _vkev; _vkev=$(gather_vulkan_events 20)
    local _vkev_n=0; [[ -n "$_vkev" ]] && _vkev_n=$(printf '%s\n' "$_vkev" | wc -l)
    gb_section "GreenBoost Vulkan Layer  (${_vkev_n} events)"
    if [[ -n "$_vkev" ]]; then
        while IFS= read -r _vkline; do
            [[ -z "$_vkline" ]] && continue
            format_vulkan_event "$_vkline"
        done <<< "$_vkev"
        _diag_ok+=("Vulkan layer active — ${_vkev_n} events recorded")
    else
        echo -e "  ${C_DIM}(empty — checked: journalctl VK_LAYER_GREENBOOST, /var/log/syslog)${C_RESET}"
        _diag_warns+=("no Vulkan layer events — GreenBoost Proton may not be selected for this game")
    fi
    echo ""

    # ── Steam client logs ─────────────────────────────────────────────────
    local _stcl; _stcl=$(gather_steam_client_logs 20)
    local _stcl_n=0; [[ -n "$_stcl" ]] && _stcl_n=$(printf '%s\n' "$_stcl" | wc -l)
    gb_section "Steam Client Logs  (${_stcl_n} events)"
    if [[ -n "$_stcl" ]]; then
        local _has_vdf_error=0
        while IFS= read -r _sl; do
            [[ -z "$_sl" ]] && continue
            if printf '%s' "$_sl" | grep -qiE 'error|exception|traceback|fail|crash'; then
                echo -e "  ${C_RED}✗${C_RESET}  ${C_RED}${_sl:0:140}${C_RESET}"
                _diag_errors+=("steam: ${_sl:0:100}")
                # Detect libraryfolders.vdf load failures — Steam can't find its game library.
                if printf '%s' "$_sl" | grep -q 'libraryfolders.vdf' && _has_vdf_error=0; then
                    _has_vdf_error=1
                fi
            else
                echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}${_sl:0:140}${C_RESET}"
            fi
        done <<< "$_stcl"
        if (( _has_vdf_error )); then
            _diag_errors+=("Steam library config unreadable (libraryfolders.vdf) — run: steam steam://validate or delete and re-login to Steam")
        fi
    else
        echo -e "  ${C_DIM}(empty — checked: ~/.local/share/Steam/logs/ ~/.steam/steam.log)${C_RESET}"
    fi
    echo ""

    # ── Pressure vessel / SteamLinuxRuntime ───────────────────────────────
    local _pvl; _pvl=$(gather_pressure_vessel_logs 10)
    local _pvl_n=0; [[ -n "$_pvl" ]] && _pvl_n=$(printf '%s\n' "$_pvl" | wc -l)
    gb_section "Pressure Vessel / SteamLinuxRuntime  (${_pvl_n} events)"
    if [[ -n "$_pvl" ]]; then
        while IFS= read -r _pvline; do
            [[ -z "$_pvline" ]] && continue
            if printf '%s' "$_pvline" | grep -qiE 'error|fail|crash'; then
                echo -e "  ${C_RED}✗${C_RESET}  ${C_RED}${_pvline:0:140}${C_RESET}"
                _diag_errors+=("pressure-vessel: ${_pvline:0:100}")
            else
                echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}${_pvline:0:140}${C_RESET}"
            fi
        done <<< "$_pvl"
    else
        echo -e "  ${C_DIM}(empty — checked: SteamLinuxRuntime_*/var/slr/log/ /tmp/pressure-vessel-*)${C_RESET}"
    fi
    echo ""

    # ── Wine / Proton crashes ─────────────────────────────────────────────
    local _wcl; _wcl=$(gather_wine_crash_logs 8)
    local _wcl_n=0; [[ -n "$_wcl" ]] && _wcl_n=$(printf '%s\n' "$_wcl" | wc -l)
    gb_section "Wine / Proton Crashes  (${_wcl_n} events)"
    if [[ -n "$_wcl" ]]; then
        while IFS= read -r _wcline; do
            [[ -z "$_wcline" ]] && continue
            echo -e "  ${C_RED}✗${C_RESET}  ${C_RED}${_wcline:0:140}${C_RESET}"
        done <<< "$_wcl"
        _diag_errors+=("wine64/wineserver crash detected — see Wine / Proton Crashes section above")
    else
        echo -e "  ${C_DIM}(empty — checked: coredumpctl wine64/wineserver/winedevice, compatdata/*.dmp)${C_RESET}"
    fi
    echo ""

    # ── DXR crash detection ──────────────────────────────────────────────
    local _dxr_crash_appid=""
    local _last_vkd3d_line
    _last_vkd3d_line=$(grep '\[greenboost-proton\] VKD3D_CONFIG=' \
        "$HOME/.local/share/Steam/logs/console-linux.txt" 2>/dev/null | tail -1)
    if [[ "$_last_vkd3d_line" == *"dxr"* ]]; then
        local _dxr_candidate
        _dxr_candidate=$(find "$HOME/.local/share/Steam/steamapps/compatdata" \
            -maxdepth 8 -name '*.log' 2>/dev/null \
            | xargs -r ls -t 2>/dev/null | head -10 \
            | while IFS= read -r _f; do
                grep -ql 'EXCEPTION_ACCESS_VIOLATION' "$_f" 2>/dev/null \
                    && grep -ql 'd3d12core' "$_f" 2>/dev/null \
                    && echo "$_f" && break
              done)
        if [[ -n "$_dxr_candidate" ]]; then
            _dxr_crash_appid=$(grep -oP 'compatdata/\K[0-9]+' <<< "$_dxr_candidate" | head -1)
        fi
    fi

    # ── GreenBoost stack health checks ──────────────────────────────────
    # 1. Kernel module
    if lsmod 2>/dev/null | grep -q '^greenboost'; then
        local _phys _virt
        _phys=$(cat /sys/module/greenboost/parameters/physical_vram_gb 2>/dev/null || echo "?")
        _virt=$(cat /sys/module/greenboost/parameters/virtual_vram_gb 2>/dev/null || echo "?")
        _diag_ok+=("kernel module loaded — T1=${_phys} GB physical + T2 cap=${_virt} GB virtual")
    else
        _diag_errors+=("kernel module NOT loaded — T2 pool unavailable (run: sudo greenboost load)")
    fi
    # 2. GreenBoost Proton install
    if [[ "${GB_PROTON_INSTALLED:-0}" == "1" ]]; then
        _diag_ok+=("GreenBoost Proton: installed")
    else
        _diag_errors+=("GreenBoost Proton NOT installed — run: cd ~/Dev/greenboost/greenboost_proton && ./install.sh  then restart Steam")
    fi
    # 3. GREENBOOST_VULKAN=1 in running DX12 game
    if (( DX12_ACTIVE )); then
        local _gb_vk=0
        for _pid in $(pgrep wine64 2>/dev/null | head -10); do
            if tr '\0' '\n' < "/proc/$_pid/environ" 2>/dev/null \
                    | grep -q '^GREENBOOST_VULKAN=1'; then
                _gb_vk=1; break
            fi
        done
        if (( _gb_vk )); then
            _diag_ok+=("GREENBOOST_VULKAN=1 active in running game — virtual VRAM pool enabled")
        else
            _diag_errors+=("GREENBOOST_VULKAN=1 NOT set in running game — T2 pool inactive. In Steam: right-click game → Properties → Compatibility → select 'GreenBoost Proton'")
        fi
    fi

    # ── Diagnostic Summary ───────────────────────────────────────────────
    gb_section "Diagnostic Summary"
    if [[ -n "$_dxr_crash_appid" ]]; then
        echo -e "  ${C_RED}✗${C_RESET}  ${C_RED}${C_BOLD}DXR crash detected${C_RESET}  ${C_RED}— NULL pointer in d3d12core.dll (AppID ${_dxr_crash_appid})${C_RESET}"
        echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}This game crashes when DXR (ray tracing) is active via VKD3D-Proton.${C_RESET}"
        echo -e "  ${C_VIOLET}◈${C_RESET}  ${C_GRAY}Fix: add to Steam launch options for this game:${C_RESET}"
        echo -e "  ${C_VIOLET}◈${C_RESET}  ${C_CYAN}${C_BOLD}  GREENBOOST_NO_DXR=1 %command%${C_RESET}"
        echo -e ""
        _diag_errors+=("DXR crash in AppID ${_dxr_crash_appid}: add GREENBOOST_NO_DXR=1 %command% to Steam launch options")
    fi
    for _e in "${_diag_errors[@]}"; do
        echo -e "  ${C_RED}✗${C_RESET}  ${C_RED}${_e}${C_RESET}"
    done
    for _w in "${_diag_warns[@]}"; do
        echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}${_w}${C_RESET}"
    done
    for _o in "${_diag_ok[@]}"; do
        echo -e "  ${C_LIME}✓${C_RESET}  ${C_GRAY}${_o}${C_RESET}"
    done
    if [[ ${#_diag_errors[@]} -eq 0 && ${#_diag_warns[@]} -eq 0 ]]; then
        echo -e "  ${C_LIME}✓${C_RESET}  ${C_GRAY}All checks passed — no errors or warnings detected.${C_RESET}"
    fi
    echo ""
}

# cmd_inference_logs — focused view: kernel tier-transitions, GreenBoost services,
# Vulkan layer.  No Proton/Steam/Wine sections.
cmd_inference_logs() {
    local _llm_mode=0
    for _a in "$@"; do [[ "$_a" == "--llm" ]] && _llm_mode=1; done
    if (( _llm_mode )); then
        # LLM mode: inference-only compact output (kernel + AI services, no gaming)
        local _strip='s/\x1b\[[0-9;]*[mKHJsu]//g'
        _llm_section() {
            local _label="$1" _text="$2" _limit="${3:-15}"
            local _filtered
            _filtered=$(printf '%s\n' "$_text" \
                | grep -iE 'err|warn|fail|oom|crash|denied|evict|tier|transition|alloc|timeout|CRITICAL' \
                | sed "$_strip" | sed 's/^[[:space:]]*//' \
                | sort | uniq -c | sort -rn \
                | awk '{c=$1; $1=""; sub(/^ /,""); if(c>1) printf "[%dx] %s\n",c,$0; else print $0}' \
                | head -"$_limit")
            local _n=0; [[ -n "$_filtered" ]] && _n=$(printf '%s\n' "$_filtered" | wc -l)
            echo "[${_label}] ${_n} events"
            [[ -n "$_filtered" ]] && printf '%s\n' "$_filtered" | sed 's/^/  /'
        }
        local _mod="MISSING"
        lsmod 2>/dev/null | grep -q '^greenboost' && _mod="loaded"
        echo "gb_inference_logs v=${GB_VERSION} ts=$(date '+%Y-%m-%dT%H:%M') mod=${_mod}"
        _llm_section "kernel" "$(gather_flow_events 50 2>/dev/null)"
        _llm_section "services" "$(journalctl --since '1 hour ago' --no-pager -q \
            -u ollama -u greenboost-idle-reclaim \
            -u greenboost-sentinel -u greenboost-recovery 2>/dev/null | tail -60)"
        return
    fi

    local _ts; _ts=$(date '+%Y-%m-%dT%H:%M')
    echo -e ""
    echo -e "  ${C_VIOLET}${C_BOLD}GreenBoost inference logs${C_RESET}  ${C_DIM}·  ${_ts}${C_RESET}"
    echo -e ""

    local _diag_errors=() _diag_warns=() _diag_ok=()

    # Check module status for summary
    local _mod_status="MISSING"
    lsmod 2>/dev/null | grep -q '^greenboost' && _mod_status="loaded"
    [[ "$_mod_status" == "loaded" ]] \
        && _diag_ok+=("kernel module loaded") \
        || _diag_errors+=("kernel module NOT loaded — no T2/T3 memory available")

    # ── Kernel module events ──────────────────────────────────────────────
    local _flow; _flow=$(gather_flow_events 40)
    local _flow_n=0; [[ -n "$_flow" ]] && _flow_n=$(printf '%s\n' "$_flow" | wc -l)
    gb_section "Kernel Module (dmesg)  (${_flow_n} events)"
    if [[ -n "$_flow" ]]; then
        while IFS= read -r _fline; do
            [[ -z "$_fline" ]] && continue
            echo -e "  ${C_DIM}${_fline}${C_RESET}"
        done <<< "$_flow"
    else
        echo -e "  ${C_DIM}(empty — dmesg grep: greenboost tier-transitions)${C_RESET}"
    fi
    echo ""

    # ── AI inference service journal (no gaming services) ─────────────────
    local _svc
    _svc=$(journalctl --since "1 hour ago" --no-pager -q \
        -u ollama -u greenboost-idle-reclaim \
        -u greenboost-sentinel -u greenboost-recovery 2>/dev/null | tail -40)
    local _svc_n=0; [[ -n "$_svc" ]] && _svc_n=$(printf '%s\n' "$_svc" | wc -l)
    gb_section "AI Inference Services (last 1h)  (${_svc_n} events)"
    if [[ -n "$_svc" ]]; then
        while IFS= read -r _sline; do
            [[ -z "$_sline" ]] && continue
            echo -e "  ${C_GRAY}${_sline}${C_RESET}"
        done <<< "$_svc"
    else
        echo -e "  ${C_DIM}(empty — units: ollama greenboost-idle-reclaim greenboost-sentinel greenboost-recovery)${C_RESET}"
    fi
    echo ""

    # ── OOM / memory-pressure detection ──────────────────────────────────
    local _oom_events
    _oom_events=$(printf '%s\n' "$_svc" \
        | grep -iE 'oom.kill|OOM killer|Failed.*oom-kill|killed.*OOM')
    if [[ -n "$_oom_events" ]]; then
        local _mem_peak
        _mem_peak=$(printf '%s\n' "$_svc" \
            | grep -oP '[0-9]+(\.[0-9]+)?[GMKT]i? memory peak' | head -1 \
            | grep -oP '^[0-9]+(\.[0-9]+)?[GMKT]i?')
        local _peak_note=""
        [[ -n "$_mem_peak" ]] && _peak_note=" (peak: ${_mem_peak})"
        _diag_errors+=("OOM kill detected in ollama service${_peak_note} — KV cache + model overflow exceeded available system RAM")
        _diag_errors+=("  Likely cause: virtual VRAM total included NVMe T3 pool, inflating context length")
        _diag_errors+=("  Fix: sudo greenboost set-kv-reserve <MB>  or  set OLLAMA_NUM_CTX=<smaller value>")
    fi

    # Warn if ollama chose a very large default context (can lead to OOM KV allocation)
    local _num_ctx
    _num_ctx=$(printf '%s\n' "$_svc" | grep -oP 'default_num_ctx=\K[0-9]+' | head -1)
    if [[ -n "$_num_ctx" ]] && (( _num_ctx >= 131072 )); then
        _diag_warns+=("Ollama default_num_ctx=${_num_ctx} — KV cache will be very large; verify T2 has enough headroom (sudo greenboost status)")
    fi

    # ── Diagnostic Summary ────────────────────────────────────────────────
    gb_section "Diagnostic Summary"
    for _e in "${_diag_errors[@]}"; do
        echo -e "  ${C_RED}✗${C_RESET}  ${C_RED}${_e}${C_RESET}"
    done
    for _w in "${_diag_warns[@]}"; do
        echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}${_w}${C_RESET}"
    done
    for _o in "${_diag_ok[@]}"; do
        echo -e "  ${C_LIME}✓${C_RESET}  ${C_GRAY}${_o}${C_RESET}"
    done
    if [[ ${#_diag_errors[@]} -eq 0 && ${#_diag_warns[@]} -eq 0 ]]; then
        echo -e "  ${C_LIME}✓${C_RESET}  ${C_GRAY}All checks passed — no errors or warnings detected.${C_RESET}"
    fi
    echo ""
}

# ════════════════════════════════════════════════════════════════════════
# inference-test — live benchmark that verifies GreenBoost path + perf
# ════════════════════════════════════════════════════════════════════════

# _infer_test_pick_model [--model NAME]
# Queries Ollama for pulled models and presents a numbered selection wizard.
# Prints chosen model name to stdout.
_infer_test_pick_model() {
    local _forced_model="$1"
    [[ -n "$_forced_model" ]] && { echo "$_forced_model"; return 0; }

    local _default="glm4:latest"
    # Prefer glm-4 variants
    local _pref_patterns=("glm-4" "glm4" "flash")

    # Query Ollama /api/tags for pulled models
    local _raw _models=()
    if command -v curl &>/dev/null; then
        _raw=$(curl -sf --max-time 4 http://localhost:11434/api/tags 2>/dev/null) || true
    fi
    if [[ -n "$_raw" ]] && command -v python3 &>/dev/null; then
        local _parsed
        _parsed=$(printf '%s' "$_raw" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    for m in d.get('models', []):
        n = m.get('name','')
        if n: print(n)
except: pass
" 2>/dev/null) || true
        while IFS= read -r _m; do
            [[ -n "$_m" ]] && _models+=("$_m")
        done <<< "$_parsed"
    fi

    # Determine default selection
    local _default_idx=0
    local _i=0
    for _m in "${_models[@]}"; do
        for _pat in "${_pref_patterns[@]}"; do
            if [[ "${_m,,}" == *"$_pat"* ]]; then
                _default_idx=$_i
                break 2
            fi
        done
        (( _i++ ))
    done

    echo ""
    echo -e "  ${C_CYAN}${C_BOLD}Select model for inference test${C_RESET}"
    echo ""

    if [[ ${#_models[@]} -eq 0 ]]; then
        echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}No pulled models found in Ollama.${C_RESET}"
        echo -e "  ${C_DIM}Defaulting to: ${C_CYAN}${_default}${C_RESET}"
        echo -e "  ${C_DIM}(pull it first with: ollama pull ${_default})${C_RESET}"
        echo ""
        echo "$_default"
        return 0
    fi

    local _idx=1
    for _m in "${_models[@]}"; do
        local _marker=""
        (( _idx - 1 == _default_idx )) && _marker=" ${C_LIME}← default${C_RESET}"
        echo -e "  ${C_GRAY}[${_idx}]${C_RESET}  ${C_CYAN}${_m}${C_RESET}${_marker}"
        (( _idx++ ))
    done
    echo ""
    echo -e -n "  ${C_AMBER}❯${C_RESET}  ${C_GRAY}Model [$(( _default_idx + 1 ))]: ${C_RESET}"

    local _choice=""
    read -r _choice 2>/dev/null || true
    _choice="${_choice// /}"

    if [[ -z "$_choice" ]]; then
        echo "${_models[$_default_idx]}"
    elif [[ "$_choice" =~ ^[0-9]+$ ]] && (( _choice >= 1 && _choice <= ${#_models[@]} )); then
        echo "${_models[$(( _choice - 1 ))]}"
    else
        echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}Invalid selection — using default.${C_RESET}" >&2
        echo "${_models[$_default_idx]}"
    fi
}

# _infer_test_run_prompt MODEL
# Runs a fixed benchmark prompt via Ollama streaming API.
# Sets: IT_TOK_TOTAL, IT_TOK_PER_SEC, IT_FIRST_TOKEN_MS, IT_TOTAL_MS, IT_ERROR
_infer_test_run_prompt() {
    local _model="$1"
    IT_TOK_TOTAL=0; IT_TOK_PER_SEC=0; IT_FIRST_TOKEN_MS=0; IT_TOTAL_MS=0; IT_ERROR=""

    local _prompt="Explain in exactly 3 sentences why memory bandwidth matters for large language model inference throughput."
    local _t0 _t1 _tfirst=0 _token_count=0 _first_done=0

    _t0=$(date +%s%3N)

    local _resp
    if ! command -v curl &>/dev/null; then
        IT_ERROR="curl not available"; return 1
    fi

    # Stream NDJSON from /api/generate; collect tokens and timing
    local _tmp; _tmp=$(mktemp)
    curl -sf --max-time 60 \
        -X POST http://localhost:11434/api/generate \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${_model}\",\"prompt\":\"${_prompt}\",\"stream\":true}" \
        > "$_tmp" 2>/dev/null
    local _curl_rc=$?

    _t1=$(date +%s%3N)
    IT_TOTAL_MS=$(( _t1 - _t0 ))

    if [[ $_curl_rc -ne 0 || ! -s "$_tmp" ]]; then
        IT_ERROR="Ollama API request failed (is ollama running? is '${_model}' pulled?)"
        rm -f "$_tmp"; return 1
    fi

    # Parse NDJSON: count tokens, find first-token time from eval_duration
    if command -v python3 &>/dev/null; then
        local _parsed
        _parsed=$(python3 -c "
import json, sys
lines = open('$_tmp').readlines()
tok = 0
first_ms = 0
total_ms = 0
for line in lines:
    line = line.strip()
    if not line: continue
    try:
        d = json.loads(line)
        if d.get('response'): tok += 1
        if d.get('prompt_eval_duration'):
            first_ms = d['prompt_eval_duration'] // 1_000_000
        if d.get('eval_duration') and d.get('eval_count'):
            dur_s = d['eval_duration'] / 1e9
            total_ms = int(d['eval_duration'] // 1_000_000)
            tps = d['eval_count'] / dur_s if dur_s > 0 else 0
            tok = d.get('eval_count', tok)
            print(f'{tok}|{first_ms}|{total_ms}|{tps:.1f}')
            sys.exit(0)
    except: pass
print(f'{tok}|{first_ms}|{total_ms}|0')
" 2>/dev/null) || true

        if [[ -n "$_parsed" ]]; then
            IT_TOK_TOTAL=$(echo "$_parsed"      | cut -d'|' -f1)
            IT_FIRST_TOKEN_MS=$(echo "$_parsed" | cut -d'|' -f2)
            IT_TOTAL_MS=$(echo "$_parsed"       | cut -d'|' -f3)
            IT_TOK_PER_SEC=$(echo "$_parsed"    | cut -d'|' -f4)
            [[ -z "$IT_TOTAL_MS" || "$IT_TOTAL_MS" == "0" ]] && IT_TOTAL_MS=$(( _t1 - _t0 ))
        fi
    fi
    rm -f "$_tmp"
}

# _infer_test_write_llm_report — writes structured report to GB_INFER_TEST_LOG
# and optionally to stdout (when _print=1).
_infer_test_write_llm_report() {
    local _model="$1" _print="${2:-0}"
    local _dir; _dir=$(dirname "$GB_INFER_TEST_LOG")
    mkdir -p "$_dir" 2>/dev/null || true

    # path verdict
    local _verdict="UNKNOWN"
    case "$SS_ACTIVE_PATH" in
        A0) _verdict="OPTIMAL" ;;
        A)  _verdict="GOOD" ;;
        B)  _verdict="FALLBACK" ;;
        C)  _verdict="LAST_RESORT" ;;
    esac

    # kernel last 5 greenboost dmesg lines
    local _kern5
    _kern5=$(dmesg 2>/dev/null | grep -i greenboost | tail -5 | tr '\n' '|' | sed 's/|$//')
    [[ -z "$_kern5" ]] && _kern5="none"

    local _report
    _report=$(cat <<REPORTEOF
greenboost inference-test v=${GB_VERSION} ts=$(date -Iseconds) model=${_model}
[path]
active=${SS_ACTIVE_PATH:-unknown}  a0_delta=${IT_A0_DELTA:-0}  a_delta=${IT_A_DELTA:-0}  b_delta=${IT_B_DELTA:-0}  c_delta=${IT_C_DELTA:-0}  verdict=${_verdict}
[perf]
tok_per_sec=${IT_TOK_PER_SEC:-0}  first_token_ms=${IT_FIRST_TOKEN_MS:-0}  total_ms=${IT_TOTAL_MS:-0}  tokens=${IT_TOK_TOTAL:-0}
[memory]
vram_used_mb=${GPU_VRAM_USED_MB:-0}  vram_total_mb=${GPU_VRAM_TOTAL_MB:-0}  t2_alloc_mb=${PI_T2_ALLOC_MB:-0}  t3_alloc_mb=${PI_T3_ALLOC_MB:-0}
[module]
loaded=$(lsmod 2>/dev/null | grep -c "^${DRIVER_NAME} " || echo 0)  version=${GB_VERSION}  kv_reserve_mb=${PB_KV_RSV_MB:-0}
[ollama]
model=${OL_MODEL:--}  ctx_size=${OL_CTX_SIZE:-0}  param_count=${OL_PARAM_COUNT:--}  vram_mb=${OL_VRAM_MB:-0}
[warnings]
${IT_ERROR:-none}
[kernel_last5]
${_kern5}
REPORTEOF
)

    printf '%s\n' "$_report" > "$GB_INFER_TEST_LOG" 2>/dev/null || true
    if [[ "$_print" == "1" ]]; then
        echo ""
        echo -e "  ${C_CYAN}${C_BOLD}━━━ LLM Report (${GB_INFER_TEST_LOG}) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
        echo ""
        while IFS= read -r _l; do
            echo -e "  ${C_DIM}${_l}${C_RESET}"
        done <<< "$_report"
        echo ""
    fi
}

# cmd_inference_test [--model NAME] [--llm] [--no-load]
# Interactive inference test: model wizard → pre-flight → run → report.
cmd_inference_test() {
    local _forced_model="" _print_llm=0 _no_load=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)  _forced_model="$2"; shift 2 ;;
            --llm)    _print_llm=1; shift ;;
            --no-load) _no_load=1; shift ;;
            *) shift ;;
        esac
    done

    gb_header
    echo -e "  ${C_CYAN}${C_BOLD}Inference Test${C_RESET}  ${C_DIM}— verifies GreenBoost path, perf, and memory${C_RESET}"
    echo ""

    # ── 1. Pre-flight ────────────────────────────────────────────────────
    detect_hardware 2>/dev/null || true
    parse_shim_stats || true

    local _pf_errors=() _pf_warns=()

    if ! lsmod 2>/dev/null | grep -q "^${DRIVER_NAME} "; then
        _pf_warns+=("GreenBoost kernel module not loaded — paths B/C only")
    fi
    if ! command -v ollama &>/dev/null && ! curl -sf --max-time 2 http://localhost:11434/api/tags &>/dev/null; then
        _pf_errors+=("Ollama not running — start with: ollama serve")
    fi

    for _e in "${_pf_errors[@]}"; do
        echo -e "  ${C_RED}✗${C_RESET}  ${C_RED}${_e}${C_RESET}"
    done
    for _w in "${_pf_warns[@]}"; do
        echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}${_w}${C_RESET}"
    done
    if [[ ${#_pf_errors[@]} -gt 0 ]]; then
        echo ""
        echo -e "  ${C_RED}Pre-flight failed — fix the above before running inference-test.${C_RESET}"
        echo ""
        return 1
    fi

    # ── 2. Model selection ───────────────────────────────────────────────
    local _model
    if [[ -t 0 ]]; then
        _model=$(_infer_test_pick_model "$_forced_model")
    else
        _model="${_forced_model:-glm4:latest}"
    fi
    _model="${_model//[$'\n\r']/}"
    [[ -z "$_model" ]] && _model="glm4:latest"

    echo ""
    echo -e "  ${C_VIOLET}◈${C_RESET}  ${C_GRAY}Testing model: ${C_CYAN}${C_BOLD}${_model}${C_RESET}"
    echo ""

    # ── 3. Snapshot BEFORE ───────────────────────────────────────────────
    parse_pool_brief; parse_pool_info; parse_shim_stats; query_gpu_vram; query_ollama_ps
    local _a0_before=${SS_PATH_A0:-0} _a_before=${SS_PATH_A:-0}
    local _b_before=${SS_PATH_B:-0}  _c_before=${SS_PATH_C:-0}
    local _vram_before=${GPU_VRAM_USED_MB:-0}

    # ── 4. Run inference (with spinner) ──────────────────────────────────
    echo -e "  ${C_DIM}Running benchmark prompt…${C_RESET}"
    IT_ERROR=""
    _infer_test_run_prompt "$_model" &
    local _infer_pid=$!
    gb_spin "$_infer_pid" "Running inference on ${_model}"
    wait "$_infer_pid" 2>/dev/null || true

    # ── 5. Snapshot AFTER ────────────────────────────────────────────────
    parse_shim_stats; query_gpu_vram; query_ollama_ps
    IT_A0_DELTA=$(( ${SS_PATH_A0:-0} - _a0_before ))
    IT_A_DELTA=$(( ${SS_PATH_A:-0}  - _a_before ))
    IT_B_DELTA=$(( ${SS_PATH_B:-0}  - _b_before ))
    IT_C_DELTA=$(( ${SS_PATH_C:-0}  - _c_before ))

    # ── 6. Result panel ──────────────────────────────────────────────────
    gb_separator
    echo ""
    echo -e "  ${C_CYAN}${C_BOLD}Results${C_RESET}"
    echo ""

    # Path verdict
    local _verdict_color _verdict_label
    case "$SS_ACTIVE_PATH" in
        A0) _verdict_color="$C_LIME";  _verdict_label="OPTIMAL  (A0 — fastest path)" ;;
        A)  _verdict_color="$C_LIME";  _verdict_label="GOOD     (A — fast path)" ;;
        B)  _verdict_color="$C_AMBER"; _verdict_label="FALLBACK (B — no kernel module)" ;;
        C)  _verdict_color="$C_RED";   _verdict_label="DEGRADED (C — last resort)" ;;
        *)  _verdict_color="$C_GRAY";  _verdict_label="UNKNOWN  (shim stats unavailable)" ;;
    esac
    echo -e "  ${C_BOLD}Path${C_RESET}       ${_verdict_color}${C_BOLD}${_verdict_label}${C_RESET}"

    # Performance
    local _tps_color="$C_LIME"
    (( ${IT_TOK_PER_SEC%.*} < 5 )) 2>/dev/null && _tps_color="$C_RED"
    echo -e "  ${C_BOLD}Speed${C_RESET}      ${_tps_color}${IT_TOK_PER_SEC:-0} tok/s${C_RESET}  ${C_DIM}first-token: ${IT_FIRST_TOKEN_MS:-0} ms  total: ${IT_TOTAL_MS:-0} ms  tokens: ${IT_TOK_TOTAL:-0}${C_RESET}"

    # Memory
    local _vram_delta=$(( ${GPU_VRAM_USED_MB:-0} - _vram_before ))
    echo -e "  ${C_BOLD}VRAM${C_RESET}       ${C_CYAN}${GPU_VRAM_USED_MB:-0}${C_DIM}/${GPU_VRAM_TOTAL_MB:-0} MB${C_RESET}  ${C_DIM}(delta: +${_vram_delta} MB)${C_RESET}"
    if (( ${PI_T2_ALLOC_MB:-0} > 0 )); then
        echo -e "  ${C_BOLD}T2 spill${C_RESET}   ${C_AMBER}${PI_T2_ALLOC_MB} MB${C_RESET} ${C_DIM}(VRAM overflow → DDR)${C_RESET}"
    else
        echo -e "  ${C_BOLD}T2 spill${C_RESET}   ${C_LIME}none${C_RESET}"
    fi
    if (( ${PI_T3_ALLOC_MB:-0} > 0 )); then
        echo -e "  ${C_BOLD}T3 spill${C_RESET}   ${C_AMBER}${PI_T3_ALLOC_MB} MB${C_RESET} ${C_DIM}(DDR overflow → NVMe)${C_RESET}"
    fi

    # Alloc delta
    echo -e "  ${C_BOLD}Allocs${C_RESET}     ${C_DIM}A0:+${IT_A0_DELTA}  A:+${IT_A_DELTA}  B:+${IT_B_DELTA}  C:+${IT_C_DELTA}${C_RESET}"

    # Errors
    if [[ -n "$IT_ERROR" ]]; then
        echo ""
        echo -e "  ${C_RED}✗${C_RESET}  ${C_RED}${IT_ERROR}${C_RESET}"
    fi

    echo ""
    gb_separator

    # ── 7. LLM report ────────────────────────────────────────────────────
    _infer_test_write_llm_report "$_model" "$_print_llm"
    echo -e "  ${C_DIM}Full report saved: ${C_GRAY}${GB_INFER_TEST_LOG}${C_RESET}"
    echo -e "  ${C_DIM}Feed to LLM:  ${C_CYAN}cat ${GB_INFER_TEST_LOG} | greenboost inference-test --llm${C_RESET}"
    echo ""
}

# ════════════════════════════════════════════════════════════════════════
# Vulkan dashboard — data helpers + renderer + entry point
# ════════════════════════════════════════════════════════════════════════

# query_vulkan_layer — check layer installation state, query device/driver info.
# Sets: VKL_MANIFEST_OK (0/1), VKL_SO_OK (0/1), VKL_MANIFEST_PATH, VKL_SO_PATH,
#       VKL_GPU_DEVICE, VKL_DRIVER_VERSION.
query_vulkan_layer() {
    VKL_MANIFEST_OK=0; VKL_SO_OK=0
    VKL_MANIFEST_PATH="$VULKAN_IMPLICIT_LAYER_DIR/$VULKAN_LAYER_MANIFEST"
    VKL_SO_PATH="$SHIM_DEST/$VULKAN_LAYER_LIB"
    VKL_GPU_DEVICE="—"; VKL_DRIVER_VERSION="—"

    [[ -f "$VKL_MANIFEST_PATH" ]] && VKL_MANIFEST_OK=1
    [[ -f "$VKL_SO_PATH"       ]] && VKL_SO_OK=1

    # Try vulkaninfo --summary first (fastest; available when vulkan-tools installed).
    if command -v vulkaninfo &>/dev/null; then
        local _vksum
        _vksum=$(timeout 5 vulkaninfo --summary 2>/dev/null) || _vksum=""
        if [[ -n "$_vksum" ]]; then
            local _dev _drv
            _dev=$(printf '%s' "$_vksum" \
                | grep -m1 'deviceName' | grep -oP '=\s*\K.+' | sed 's/[[:space:]]*$//')
            _drv=$(printf '%s' "$_vksum" \
                | grep -m1 -iE 'driverVersion|driverInfo' \
                | grep -oP '=\s*\K.+' | sed 's/[[:space:]]*$//')
            [[ -n "$_dev" ]] && VKL_GPU_DEVICE="$_dev"
            [[ -n "$_drv" ]] && VKL_DRIVER_VERSION="$_drv"
        fi
    fi

    # Fallback to GPU_NAME (detect_hardware) for device name.
    if [[ "$VKL_GPU_DEVICE" == "—" && -n "${GPU_NAME:-}" ]]; then
        VKL_GPU_DEVICE="$GPU_NAME"
    fi

    # Fallback to nvidia-smi for driver version.
    if [[ "$VKL_DRIVER_VERSION" == "—" ]] && command -v nvidia-smi &>/dev/null; then
        local _drv2
        _drv2=$(timeout 2 nvidia-smi --query-gpu=driver_version \
            --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
        [[ -n "$_drv2" ]] && VKL_DRIVER_VERSION="$_drv2"
    fi
}

# query_vulkan_processes — detect live Vulkan/Proton processes and their VRAM usage.
# Sets: VKP_PROCS (newline-separated "pid name vram_mb"), VKP_COUNT.
query_vulkan_processes() {
    VKP_PROCS=""; VKP_COUNT=0

    local _pids
    _pids=$(pgrep -f 'wine64|wine|proton|fossilize_replay|SteamLinuxRuntime' \
        2>/dev/null | sort -u)
    [[ -z "$_pids" ]] && return 0

    # Single nvidia-smi call to get per-process VRAM.
    local _nvsmi_procs=""
    if command -v nvidia-smi &>/dev/null; then
        _nvsmi_procs=$(timeout 3 nvidia-smi \
            --query-compute-apps=pid,used_gpu_memory \
            --format=csv,noheader,nounits 2>/dev/null) || _nvsmi_procs=""
    fi

    local _count=0 _out=""
    while IFS= read -r _pid; do
        [[ -z "$_pid" ]] && continue
        local _name
        _name=$(ps -p "$_pid" -o comm= 2>/dev/null | head -1)
        [[ -z "$_name" ]] && continue  # process vanished

        local _vram="?"
        if [[ -n "$_nvsmi_procs" ]]; then
            local _hit
            _hit=$(printf '%s' "$_nvsmi_procs" \
                | grep -E "^[[:space:]]*${_pid}[[:space:]]*,")
            if [[ -n "$_hit" ]]; then
                local _v; _v=$(printf '%s' "$_hit" | cut -d, -f2 | tr -dc '0-9')
                [[ -n "$_v" ]] && _vram="$_v"
            fi
        fi

        _out+="${_pid} ${_name} ${_vram}"$'\n'
        (( _count++ )) || true
    done <<< "$_pids"

    VKP_PROCS="${_out%$'\n'}"
    VKP_COUNT=$_count
}

# query_shader_boost — query shader-boost service state and fossilize_replay scheduling.
# Sets: SB_ACTIVE (active|inactive|failed|unknown), SB_FOSSIL_PIDS,
#       SB_FOSSIL_NICE, SB_FOSSIL_IONICE, SB_FOSSIL_AFFINITY.
query_shader_boost() {
    SB_ACTIVE="unknown"
    SB_FOSSIL_PIDS=""
    SB_FOSSIL_NICE="—"; SB_FOSSIL_IONICE="—"; SB_FOSSIL_AFFINITY="—"

    if command -v systemctl &>/dev/null; then
        local _a
        _a=$(systemctl is-active greenboost-shader-boost 2>/dev/null || echo "unknown")
        SB_ACTIVE="${_a:-unknown}"
    fi

    local _fpids
    _fpids=$(pgrep -f fossilize_replay 2>/dev/null | tr '\n' ' ')
    _fpids="${_fpids%% }"
    SB_FOSSIL_PIDS="$_fpids"
    [[ -z "$_fpids" ]] && return 0

    local _fpid; _fpid=$(echo "$_fpids" | awk '{print $1}')
    [[ -z "$_fpid" ]] && return 0

    local _n
    _n=$(ps -p "$_fpid" -o nice= 2>/dev/null | tr -d ' ')
    [[ -n "$_n" ]] && SB_FOSSIL_NICE="$_n"

    if command -v ionice &>/dev/null; then
        local _iout
        _iout=$(ionice -p "$_fpid" 2>/dev/null)
        if [[ -n "$_iout" ]]; then
            local _cls _pri
            _cls=$(printf '%s' "$_iout" | awk -F: '{print $1}' | tr -d ' ')
            _pri=$(printf '%s' "$_iout" | grep -oP 'prio \K[0-9]+' || echo "?")
            SB_FOSSIL_IONICE="${_cls}/${_pri}"
        fi
    fi

    if command -v taskset &>/dev/null; then
        local _aff
        _aff=$(taskset -acp "$_fpid" 2>/dev/null \
            | grep -oP 'current affinity list: \K.+' || echo "")
        [[ -n "$_aff" ]] && SB_FOSSIL_AFFINITY="$_aff"
    fi
}

# gather_vulkan_events N — emit last N [VK_LAYER_GREENBOOST] syslog lines.
# Tries journalctl first, then /var/log/syslog, then /var/log/messages.
gather_vulkan_events() {
    local n=${1:-10}
    local _pat='VK_LAYER_GREENBOOST'

    local _gbvk_log="$HOME/.local/share/greenboost/proton-logs/vulkan-layer.log"
    if [[ -f "$_gbvk_log" ]]; then
        local _fout
        _fout=$(grep "$_pat" "$_gbvk_log" | tail -"$n")
        if [[ -n "$_fout" ]]; then
            printf '%s\n' "$_fout"
            return 0
        fi
    fi

    if command -v journalctl &>/dev/null; then
        local _jout
        _jout=$(journalctl --since "1 hour ago" --no-pager -q 2>/dev/null \
            | grep "$_pat" | tail -"$n")
        if [[ -n "$_jout" ]]; then
            printf '%s\n' "$_jout"
            return 0
        fi
    fi

    local _slog=""
    for _sf in /var/log/syslog /var/log/messages; do
        [[ -r "$_sf" ]] && _slog="$_sf" && break
    done
    [[ -n "$_slog" ]] && grep "$_pat" "$_slog" | tail -"$n"
    return 0
}

# format_vulkan_event "raw log line" — emit a branded, colored Vulkan event line.
format_vulkan_event() {
    local line="$1"
    local msg
    msg=$(printf '%s' "$line" \
        | sed 's/.*\[VK_LAYER_GREENBOOST\][[:space:]]*//')

    local ts=""
    ts=$(printf '%s' "$line" | grep -oP '\d\d:\d\d:\d\d' | head -1)
    if [[ -z "$ts" ]]; then
        ts=$(printf '%s' "$line" | grep -oP 'T\K\d\d:\d\d:\d\d' | head -1)
    fi
    local ts_fmt; ts_fmt=$(printf '%-10s' "${ts:-?}")
    local prefix="${C_DIM}${ts_fmt}${C_RESET}"

    if   printf '%s' "$msg" | grep -q 'CreateInstance: hooked'; then
        printf '%b\n' "  ${prefix}  ${C_LIME}$(printf '%-8s' 'hook')${C_RESET}  ${C_LIME}✓${C_RESET}  ${C_GRAY}${msg}${C_RESET}"
    elif printf '%s' "$msg" | grep -q 'DMA-BUF OK'; then
        printf '%b\n' "  ${prefix}  ${C_CYAN}$(printf '%-8s' 'T2-dma')${C_RESET}  ${C_CYAN}◈${C_RESET}  ${C_GRAY}${msg}${C_RESET}"
    elif printf '%s' "$msg" | grep -qE 'all tiers failed|returning OOM'; then
        printf '%b\n' "  ${prefix}  ${C_RED}$(printf '%-8s' 'OOM')${C_RESET}  ${C_RED}✗${C_RESET}  ${C_RED}${msg}${C_RESET}"
    elif printf '%s' "$msg" | grep -q 'FreeMemory: closed DMA-BUF'; then
        printf '%b\n' "  ${prefix}  ${C_DIM}$(printf '%-8s' 'free')${C_RESET}  ${C_DIM}·${C_RESET}  ${C_DIM}${msg}${C_RESET}"
    else
        printf '%b\n' "  ${prefix}  ${C_DIM}$(printf '%-8s' '·')${C_RESET}  ${C_DIM}·${C_RESET}  ${C_GRAY}${msg}${C_RESET}"
    fi
}

# query_dx12_status — detect running DX12 game and VKD3D-Proton environment.
# Sets: DX12_ACTIVE (0/1), DX12_APPID, DX12_GAME_NAME, DX12_PROTON_VER,
#       DX12_VKD3D_DEBUG, DX12_VKD3D_CONFIG, DX12_VKD3D_SHADER_DEBUG,
#       DX12_T2_OK, DX12_T2_FAIL, DX12_T2_MB, GB_PROTON_INSTALLED (0/1).
query_dx12_status() {
    DX12_ACTIVE=0; DX12_APPID=""; DX12_GAME_NAME=""; DX12_PROTON_VER="—"
    DX12_VKD3D_DEBUG="—"; DX12_VKD3D_CONFIG="—"; DX12_VKD3D_SHADER_DEBUG="—"
    DX12_T2_OK=0; DX12_T2_FAIL=0; DX12_T2_MB=0
    GB_PROTON_INSTALLED=0

    # Check if greenboost-proton is installed as a Steam compat tool.
    local _gbp_paths=(
        "$HOME/.steam/root/compatibilitytools.d/greenboost-proton"
        "$HOME/.local/share/Steam/compatibilitytools.d/greenboost-proton"
        "$HOME/Dev/greenboost_proton/greenboost-proton"
    )
    for _p in "${_gbp_paths[@]}"; do
        [[ -f "$_p/proton" ]] && GB_PROTON_INSTALLED=1 && break
    done

    # Find wine64 process with vkd3d-proton loaded in its maps (DX12 game).
    local _dx12_pid=""
    for _pid in $(pgrep wine64 2>/dev/null | head -10); do
        if [[ -r "/proc/$_pid/maps" ]] && grep -ql 'vkd3d' /proc/"$_pid"/maps 2>/dev/null; then
            _dx12_pid="$_pid"
            DX12_ACTIVE=1
            break
        fi
    done

    # Gather T2 stats from layer log regardless of game state.
    local _all_ev; _all_ev=$(gather_vulkan_events 200)
    if [[ -n "$_all_ev" ]]; then
        DX12_T2_OK=$(printf '%s\n' "$_all_ev" | grep -c 'T2 DMA-BUF OK' || true)
        DX12_T2_FAIL=$(printf '%s\n' "$_all_ev" | grep -c 'all tiers failed\|returning OOM' || true)
        DX12_T2_MB=$(printf '%s\n' "$_all_ev" \
            | grep -oP 'T2 DMA-BUF OK — \K[0-9]+' \
            | awk '{s+=$1}END{print s+0}')
    fi

    [[ -z "$_dx12_pid" ]] && return 0

    # Read env vars from the running game process.
    if [[ -r "/proc/$_dx12_pid/environ" ]]; then
        local _env
        _env=$(tr '\0' '\n' < "/proc/$_dx12_pid/environ" 2>/dev/null)

        DX12_APPID=$(printf '%s' "$_env" \
            | grep '^SteamAppId=' | cut -d= -f2- | head -1)
        local _vd; _vd=$(printf '%s' "$_env" \
            | grep '^VKD3D_DEBUG=' | cut -d= -f2- | head -1)
        [[ -n "$_vd" ]] && DX12_VKD3D_DEBUG="$_vd"
        local _vc; _vc=$(printf '%s' "$_env" \
            | grep '^VKD3D_CONFIG=' | cut -d= -f2- | head -1)
        [[ -n "$_vc" ]] && DX12_VKD3D_CONFIG="$_vc"
        local _vs; _vs=$(printf '%s' "$_env" \
            | grep '^VKD3D_SHADER_DEBUG=' | cut -d= -f2- | head -1)
        [[ -n "$_vs" ]] && DX12_VKD3D_SHADER_DEBUG="$_vs"

        # Detect Proton version from compat tool path env vars.
        local _compat=""
        _compat=$(printf '%s' "$_env" \
            | grep '^STEAM_COMPAT_TOOL_PATHS=\|^STEAM_COMPAT_INSTALL_PATH=' \
            | head -1 | cut -d= -f2-)
        if printf '%s' "$_compat" | grep -qi 'greenboost.proton\|greenboost-proton'; then
            DX12_PROTON_VER="greenboost-proton"
        elif printf '%s' "$_compat" | grep -qi 'proton_experimental\|Proton Experimental'; then
            DX12_PROTON_VER="Proton Experimental"
        elif [[ -n "$_compat" ]]; then
            DX12_PROTON_VER=$(printf '%s' "$_compat" \
                | tr ':' '\n' | grep -iv 'pressure.vessel\|sniper\|steam-runtime' \
                | xargs -I{} basename {} 2>/dev/null | head -1)
            [[ -z "$DX12_PROTON_VER" ]] && DX12_PROTON_VER="—"
        fi
    fi

    # Resolve game name from appmanifest.
    if [[ -n "$DX12_APPID" ]]; then
        local _acf
        _acf=$(find "$HOME/.local/share/Steam/steamapps" \
            -maxdepth 1 -name "appmanifest_${DX12_APPID}.acf" 2>/dev/null | head -1)
        if [[ -n "$_acf" && -r "$_acf" ]]; then
            local _gn; _gn=$(grep -oP '"name"\s*"\K[^"]+' "$_acf" | head -1)
            [[ -n "$_gn" ]] && DX12_GAME_NAME="$_gn"
        fi
    fi
}

# gather_proton_log_events N — emit last N VKD3D/DXVK warn/error log lines.
# Reads from greenboost-proton log dir, then standard Proton log, then journalctl.
gather_proton_log_events() {
    local n=${1:-8}
    local _log=""

    # greenboost-proton routes PROTON_LOG_DIR here
    local _gbdir="$HOME/.local/share/greenboost/proton-logs"

    if [[ -n "${DX12_APPID:-}" ]]; then
        local _candidate
        for _candidate in \
            "$_gbdir/steam-${DX12_APPID}.log" \
            "$HOME/steam-${DX12_APPID}.log"
        do
            [[ -r "$_candidate" ]] && _log="$_candidate" && break
        done
    fi

    # Also try SteamGameId — Proton's setup_logging() uses SteamGameId for the
    # filename, not SteamAppId.  They differ for non-standard/shortcut launches.
    local _sgid="${STEAM_GAMEID:-${SteamGameId:-}}"
    if [[ -n "$_sgid" && -z "$_log" ]]; then
        local _candidate
        for _candidate in \
            "$_gbdir/steam-${_sgid}.log" \
            "$HOME/steam-${_sgid}.log"
        do
            [[ -r "$_candidate" ]] && _log="$_candidate" && break
        done
    fi

    # Fallback: most recently modified steam-*.log
    if [[ -z "$_log" ]]; then
        local _recent
        _recent=$(find "$_gbdir" "$HOME" -maxdepth 1 \
            -name 'steam-*.log' 2>/dev/null \
            | xargs -r ls -t 2>/dev/null | head -1) || true
        [[ -n "$_recent" && -r "$_recent" ]] && _log="$_recent"
    fi

    # Also check PROTON_LOG_DIR from any running wine64 process environ
    if [[ -z "$_log" ]]; then
        local _pid
        for _pid in $(pgrep -x wine64 2>/dev/null | head -5); do
            local _pld
            _pld=$(tr '\0' '\n' < "/proc/$_pid/environ" 2>/dev/null \
                | grep '^PROTON_LOG_DIR=' | cut -d= -f2- | head -1)
            if [[ -n "$_pld" ]]; then
                local _pld_recent
                _pld_recent=$(find "$_pld" -maxdepth 1 -name 'steam-*.log' 2>/dev/null \
                    | xargs -r ls -t 2>/dev/null | head -1) || true
                [[ -n "$_pld_recent" && -r "$_pld_recent" ]] && _log="$_pld_recent" && break
            fi
        done
    fi

    [[ -z "$_log" ]] && return 0

    # Filter for events relevant to GreenBoost: VKD3D/DXVK errors, warnings,
    # and GreenBoost Proton wrapper status lines.
    grep -E \
        'ERR[^O]|vkd3d.*warn|WARN|fixme.*d3d|out.of.mem|OOM|alloc.*fail|VK_ERROR|\[greenboost-proton\]' \
        "$_log" 2>/dev/null \
        | grep -vE 'suppress|ignore|stub|trivial' \
        | tail -"$n"
}

# format_proton_event "raw line" — emit a branded, colored Proton log line.
format_proton_event() {
    local line="$1"
    local msg; msg=$(printf '%s' "$line" | cut -c1-140)
    if   printf '%s' "$msg" | grep -q '\[greenboost-proton\]'; then
        printf '%b\n' "  ${C_LIME}$(printf '%-8s' 'gb')${C_RESET}  ${C_LIME}◈${C_RESET}  ${C_GRAY}${msg}${C_RESET}"
    elif printf '%s' "$msg" | grep -qiE '\bERR\b|VK_ERROR|alloc.*fail'; then
        printf '%b\n' "  ${C_RED}$(printf '%-8s' 'vkd3d')${C_RESET}  ${C_RED}✗${C_RESET}  ${C_RED}${msg}${C_RESET}"
    elif printf '%s' "$msg" | grep -qiE 'WARN|fixme|warn'; then
        printf '%b\n' "  ${C_AMBER}$(printf '%-8s' 'vkd3d')${C_RESET}  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}${msg}${C_RESET}"
    else
        printf '%b\n' "  ${C_DIM}$(printf '%-8s' 'proton')${C_RESET}  ${C_DIM}·${C_RESET}  ${C_DIM}${msg}${C_RESET}"
    fi
}

# gather_steam_client_logs N — emit last N error/warning lines from Steam's own logs.
# Primary source for compat-tool launch failures and Python exceptions in the proton script.
gather_steam_client_logs() {
    local n=${1:-15}
    local _pat='fail|error|exception|traceback|exit code [1-9]|crash|cannot|not found|no such|permission denied'
    local _found=""

    # compat_log.txt — records every compat tool invocation and exit code
    for _f in \
        "$HOME/.local/share/Steam/logs/compat_log.txt" \
        "$HOME/.steam/root/logs/compat_log.txt"
    do
        if [[ -r "$_f" ]]; then
            _found=$(grep -iE "$_pat" "$_f" 2>/dev/null | grep -iv 'no error' | tail -"$n")
            [[ -n "$_found" ]] && printf '%s\n' "$_found" && return 0
        fi
    done

    # content_log.txt — app launch/download errors
    for _f in \
        "$HOME/.local/share/Steam/logs/content_log.txt" \
        "$HOME/.steam/root/logs/content_log.txt"
    do
        if [[ -r "$_f" ]]; then
            _found=$(grep -iE "$_pat" "$_f" 2>/dev/null | grep -iv 'no error' | tail -"$n")
            [[ -n "$_found" ]] && printf '%s\n' "$_found" && return 0
        fi
    done

    # main steam.log as last resort
    for _f in \
        "$HOME/.steam/steam.log" \
        "$HOME/.local/share/Steam/steam.log"
    do
        if [[ -r "$_f" ]]; then
            _found=$(grep -iE "$_pat" "$_f" 2>/dev/null | grep -iv 'no error' | tail -"$n")
            [[ -n "$_found" ]] && printf '%s\n' "$_found" && return 0
        fi
    done
    return 0
}

# gather_pressure_vessel_logs N — emit last N error/warning lines from SteamLinuxRuntime.
gather_pressure_vessel_logs() {
    local n=${1:-10}
    local _pat='error|warn|fail|cannot|missing|not found|permission'
    local _found=""

    # SteamLinuxRuntime Sniper log directory
    local _slr_base="$HOME/.local/share/Steam/steamapps/common"
    local _slr_log
    _slr_log=$(find "$_slr_base" -maxdepth 4 \
        -path '*/SteamLinuxRuntime*/var/slr/log/*.log' 2>/dev/null \
        | xargs -r ls -t 2>/dev/null | head -1) || true
    if [[ -n "$_slr_log" && -r "$_slr_log" ]]; then
        _found=$(grep -iE "$_pat" "$_slr_log" 2>/dev/null | tail -"$n")
        [[ -n "$_found" ]] && printf '%s\n' "$_found" && return 0
    fi

    # Pressure vessel temp files (written during container setup)
    local _pv_tmp
    _pv_tmp=$(find /tmp -maxdepth 1 -name 'pressure-vessel-*' -newer /proc/1/exe 2>/dev/null \
        | xargs -r ls -t 2>/dev/null | head -1) || true
    if [[ -n "$_pv_tmp" && -r "$_pv_tmp" ]]; then
        _found=$(grep -iE "$_pat" "$_pv_tmp" 2>/dev/null | tail -"$n")
        [[ -n "$_found" ]] && printf '%s\n' "$_found"
    fi
    return 0
}

# gather_wine_crash_logs N — emit recent wine64/wineserver crash evidence.
gather_wine_crash_logs() {
    local n=${1:-8}
    local _found=""

    # systemd coredumps for Wine processes
    if command -v coredumpctl &>/dev/null; then
        _found=$(coredumpctl list --no-pager 2>/dev/null \
            | grep -E 'wine64|wineserver|winedevice|proton' | tail -"$n")
        [[ -n "$_found" ]] && printf '%s\n' "$_found"
    fi

    # Wine crash dump files in compatdata (written by Wine's built-in crash handler)
    local _dumps
    _dumps=$(find "$HOME/.local/share/Steam/steamapps/compatdata" \
        -maxdepth 4 -name '*.dmp' -newer /proc/1/exe 2>/dev/null \
        | head -5) || true
    if [[ -n "$_dumps" ]]; then
        while IFS= read -r _d; do
            printf 'crash-dump: %s  (%s)\n' "$_d" \
                "$(stat -c '%y' "$_d" 2>/dev/null | cut -c1-19)"
        done <<< "$_dumps"
    fi
    return 0
}

# _cmd_vulkan_snapshot — render the Vulkan dashboard (5 panels).
_cmd_vulkan_snapshot() {
    # Gather all data
    query_vulkan_layer
    query_vulkan_processes
    query_shader_boost
    query_gpu_vram
    query_dx12_status

    # ════════════════════════════════════════════════════════════════════
    # Panel 1 — Vulkan Device & Layer
    # ════════════════════════════════════════════════════════════════════
    gb_panel_top "Vulkan Device & Layer"
    gb_panel_empty

    if [[ -n "${VKL_GPU_DEVICE:-}" && "$VKL_GPU_DEVICE" != "—" ]]; then
        gb_panel_row "${C_GRAY}◎  GPU Device        ${C_RESET}${C_GRAY}${VKL_GPU_DEVICE}${C_RESET}"
    else
        gb_panel_row "${C_GRAY}◎  GPU Device        ${C_DIM}(unavailable — install vulkan-tools or nvidia-smi)${C_RESET}"
    fi
    gb_panel_row "   ${C_GRAY}Driver              ${C_RESET}${C_GRAY}${VKL_DRIVER_VERSION:-—}${C_RESET}"
    gb_panel_empty

    # Manifest status
    local _mi _mc _ml
    if [[ "$VKL_MANIFEST_OK" -eq 1 ]]; then
        _mi="${C_LIME}✓${C_RESET}"; _mc="$C_LIME"; _ml="installed"
    else
        _mi="${C_RED}✗${C_RESET}"; _mc="$C_RED"; _ml="NOT FOUND"
    fi
    gb_panel_row "${_mi}  ${C_GRAY}Layer manifest      ${_mc}${_ml}${C_RESET}${C_DIM}   ${VKL_MANIFEST_PATH}${C_RESET}"

    # Library status
    local _si _sc _sl
    if [[ "$VKL_SO_OK" -eq 1 ]]; then
        _si="${C_LIME}✓${C_RESET}"; _sc="$C_LIME"; _sl="present"
    else
        _si="${C_RED}✗${C_RESET}"; _sc="$C_RED"; _sl="NOT FOUND"
    fi
    gb_panel_row "${_si}  ${C_GRAY}Layer library       ${_sc}${_sl}${C_RESET}${C_DIM}   ${VKL_SO_PATH}${C_RESET}"
    gb_panel_empty

    # Virtual VRAM + real VRAM bar
    local _vk_total_gb=$(( ${GB_PHYS:-0} + ${GB_VIRT:-0} ))
    local _vk_total_str="${_vk_total_gb} GB"
    (( _vk_total_gb == 0 )) && _vk_total_str="auto-detected"
    gb_panel_row "${C_VIOLET}◈${C_RESET}  ${C_GRAY}Virtual VRAM        ${C_RESET}${C_CYAN}${_vk_total_str}${C_RESET}${C_DIM}   (reported to apps when GREENBOOST_VULKAN=1)${C_RESET}"

    if (( ${GPU_VRAM_TOTAL_MB:-0} > 0 )); then
        local _vt1_used_gb _vt1_total_gb _vt1_color _vt1_bar
        _vt1_used_gb=$(awk "BEGIN{printf \"%.1f\", ${GPU_VRAM_USED_MB:-0}/1024}" 2>/dev/null || echo "?")
        _vt1_total_gb=$(( GPU_VRAM_TOTAL_MB / 1024 ))
        _vt1_color=$(gb_tier_color "${GPU_VRAM_PCT:-0}")
        _vt1_bar=$(gb_bar "${GPU_VRAM_PCT:-0}" "$_vt1_color" "${C_DIM}" 32)
        gb_panel_row "   ${C_GRAY}VRAM usage          ${_vt1_color}${_vt1_used_gb}/${_vt1_total_gb} GB${C_RESET}"
        gb_panel_row "   ${_vt1_bar}  ${_vt1_color}${GPU_VRAM_PCT}%${C_RESET}"
    else
        gb_panel_row "   ${C_GRAY}VRAM usage          ${C_DIM}(nvidia-smi unavailable)${C_RESET}"
    fi
    gb_panel_empty

    if [[ "$VKL_MANIFEST_OK" -eq 1 && "$VKL_SO_OK" -eq 1 ]]; then
        gb_panel_row "   ${C_DIM}Activate: add ${C_GRAY}GREENBOOST_VULKAN=1 %command%${C_DIM} in Steam launch options${C_RESET}"
    else
        gb_panel_row "   ${C_AMBER}⚠${C_RESET}  ${C_AMBER}Layer not installed — run: sudo greenboost install-sys-configs${C_RESET}"
    fi
    gb_panel_empty
    gb_panel_bottom

    # ════════════════════════════════════════════════════════════════════
    # Panel 2 — Active Vulkan Processes
    # ════════════════════════════════════════════════════════════════════
    gb_panel_top "Active Vulkan Processes"
    gb_panel_empty

    if (( VKP_COUNT > 0 )); then
        gb_panel_row "${C_DIM}$(printf '%-8s' 'PID')  $(printf '%-22s' 'Process')  VRAM MB${C_RESET}"
        gb_panel_empty
        while IFS= read -r _proc_line; do
            [[ -z "$_proc_line" ]] && continue
            local _ppid _pname _pvram
            _ppid=$(echo "$_proc_line"  | awk '{print $1}')
            _pname=$(echo "$_proc_line" | awk '{print $2}')
            _pvram=$(echo "$_proc_line" | awk '{print $3}')
            local _vram_str
            if [[ "$_pvram" == "?" ]]; then
                _vram_str="${C_DIM}?${C_RESET}"
            else
                _vram_str="${C_CYAN}${_pvram} MB${C_RESET}"
            fi
            gb_panel_row "${C_LIME}$(printf '%-8s' "$_ppid")${C_RESET}  ${C_GRAY}$(printf '%-22s' "$_pname")${C_RESET}  ${_vram_str}"
        done <<< "$VKP_PROCS"
        gb_panel_empty
    else
        gb_panel_row "  ${C_DIM}◎  No active Vulkan/Proton processes${C_RESET}"
        gb_panel_empty
    fi

    # Shader boost service status
    local _sb_icon _sb_color _sb_label
    case "$SB_ACTIVE" in
        active)   _sb_icon="${C_LIME}✓${C_RESET}"; _sb_color="$C_LIME";   _sb_label="active"   ;;
        inactive) _sb_icon="${C_AMBER}⚠${C_RESET}"; _sb_color="$C_AMBER"; _sb_label="inactive" ;;
        failed)   _sb_icon="${C_RED}✗${C_RESET}";  _sb_color="$C_RED";    _sb_label="failed"   ;;
        *)        _sb_icon="${C_DIM}·${C_RESET}";  _sb_color="$C_DIM";    _sb_label="unknown"  ;;
    esac
    gb_panel_row "${_sb_icon}  ${C_GRAY}shader-boost svc    ${_sb_color}${_sb_label}${C_RESET}"

    if [[ -n "$SB_FOSSIL_PIDS" ]]; then
        gb_panel_row "   ${C_GRAY}fossilize_replay    ${C_RESET}${C_CYAN}PIDs: ${SB_FOSSIL_PIDS}${C_RESET}"
        gb_panel_row "   ${C_DIM}nice ${SB_FOSSIL_NICE}   ionice ${SB_FOSSIL_IONICE}   affinity ${SB_FOSSIL_AFFINITY}${C_RESET}"
    else
        gb_panel_row "   ${C_DIM}fossilize_replay    not running${C_RESET}"
    fi
    gb_panel_empty
    gb_panel_bottom

    # ════════════════════════════════════════════════════════════════════
    # Panel 3 — DX12 / VKD3D-Proton
    # ════════════════════════════════════════════════════════════════════
    gb_panel_top "DX12 / VKD3D-Proton"
    gb_panel_empty

    if (( DX12_ACTIVE )); then
        # Running game info
        if [[ -n "$DX12_GAME_NAME" ]]; then
            gb_panel_row "${C_VIOLET}◈${C_RESET}  ${C_GRAY}Game               ${C_RESET}${C_CYAN}${DX12_GAME_NAME}${C_DIM}${DX12_APPID:+   (AppID ${DX12_APPID})}${C_RESET}"
        else
            gb_panel_row "${C_VIOLET}◈${C_RESET}  ${C_GRAY}Game               ${C_DIM}AppID ${DX12_APPID:-?}${C_RESET}"
        fi

        # Proton version — highlight if using greenboost-proton
        local _pv_color="$C_GRAY"
        [[ "$DX12_PROTON_VER" == "greenboost-proton" ]] && _pv_color="$C_LIME"
        gb_panel_row "   ${C_GRAY}Proton             ${_pv_color}${DX12_PROTON_VER}${C_RESET}"
        gb_panel_empty

        # VKD3D-Proton env vars from the live process
        local _dbg_color="$C_GRAY"
        [[ "$DX12_VKD3D_DEBUG" == "none" || "$DX12_VKD3D_DEBUG" == "—" ]] && _dbg_color="$C_DIM"
        gb_panel_row "   ${C_GRAY}VKD3D_DEBUG        ${_dbg_color}${DX12_VKD3D_DEBUG}${C_RESET}"
        gb_panel_row "   ${C_GRAY}VKD3D_SHADER_DEBUG ${C_DIM}${DX12_VKD3D_SHADER_DEBUG}${C_RESET}"

        if [[ "$DX12_VKD3D_CONFIG" != "—" && -n "$DX12_VKD3D_CONFIG" ]]; then
            gb_panel_row "   ${C_GRAY}VKD3D_CONFIG       ${C_CYAN}${DX12_VKD3D_CONFIG}${C_RESET}"
        else
            gb_panel_row "   ${C_GRAY}VKD3D_CONFIG       ${C_DIM}<defaults>${C_RESET}"
        fi
        gb_panel_empty
    else
        gb_panel_row "  ${C_DIM}◎  No DX12 game active${C_RESET}"
        gb_panel_empty
    fi

    # T2 DMA-BUF allocation statistics (from last hour of layer logs)
    if (( DX12_T2_OK > 0 || DX12_T2_FAIL > 0 )); then
        local _t2_icon _t2_color
        if (( DX12_T2_FAIL > 0 )); then
            _t2_icon="${C_AMBER}⚠${C_RESET}"; _t2_color="$C_AMBER"
        else
            _t2_icon="${C_LIME}✓${C_RESET}"; _t2_color="$C_LIME"
        fi
        gb_panel_row "${_t2_icon}  ${C_GRAY}T2 DMA-BUF (1 hr)  ${_t2_color}${DX12_T2_OK} OK${C_RESET}${C_DIM}  →  ${C_RESET}${C_CYAN}${DX12_T2_MB:-0} MB${C_RESET}${C_DIM}   ${DX12_T2_FAIL} failed${C_RESET}"
    else
        gb_panel_row "   ${C_DIM}T2 DMA-BUF (1 hr)  no overflow allocs recorded${C_RESET}"
    fi

    # greenboost-proton install hint
    gb_panel_empty
    if (( GB_PROTON_INSTALLED )); then
        gb_panel_row "   ${C_DIM}Compat tool        ${C_LIME}greenboost-proton installed${C_RESET}"
    else
        gb_panel_row "   ${C_DIM}Compat tool        ${C_AMBER}greenboost-proton not installed${C_RESET}${C_DIM}  →  greenboost_proton/install.sh${C_RESET}"
    fi
    gb_panel_empty
    gb_panel_bottom

    # ════════════════════════════════════════════════════════════════════
    # Panel 4 — GreenBoost Vulkan Activity
    # ════════════════════════════════════════════════════════════════════
    gb_panel_top "GreenBoost Vulkan Activity"
    gb_panel_empty

    local _vk_events
    _vk_events=$(gather_vulkan_events 8)
    if [[ -n "$_vk_events" ]]; then
        while IFS= read -r _vkline; do
            [[ -z "$_vkline" ]] && continue
            gb_panel_row "$(format_vulkan_event "$_vkline")"
        done <<< "$_vk_events"
    else
        gb_panel_row "  ${C_DIM}(no VK_LAYER_GREENBOOST events in the last hour)${C_RESET}"
        gb_panel_row "  ${C_DIM}Start a game with GREENBOOST_VULKAN=1 to see activity here.${C_RESET}"
    fi

    # VKD3D/DXVK log events from greenboost-proton log dir (if available)
    local _proton_events
    _proton_events=$(gather_proton_log_events 5)
    if [[ -n "$_proton_events" ]]; then
        gb_panel_empty
        gb_panel_row "  ${C_DIM}── VKD3D / DXVK ──────────────────────────────────────────────${C_RESET}"
        while IFS= read -r _pline; do
            [[ -z "$_pline" ]] && continue
            gb_panel_row "$(format_proton_event "$_pline")"
        done <<< "$_proton_events"
    fi
    gb_panel_empty
    gb_panel_bottom

    # ════════════════════════════════════════════════════════════════════
    # Panel 5 — Issues
    # ════════════════════════════════════════════════════════════════════
    gb_panel_top "Issues"
    gb_panel_empty

    local _all_events _issue_count=0
    _all_events=$(gather_vulkan_events 50)

    if [[ -n "$_all_events" ]]; then
        local _fail_lines
        _fail_lines=$(printf '%s\n' "$_all_events" \
            | grep -E 'DMA-BUF fallback failed|returning OOM')
        if [[ -n "$_fail_lines" ]]; then
            gb_panel_row "${C_RED}✗${C_RESET}  ${C_RED}T2 DMA-BUF fallback failures (last hour):${C_RESET}"
            while IFS= read -r _fl; do
                [[ -z "$_fl" ]] && continue
                gb_panel_row "$(format_vulkan_event "$_fl")"
                (( _issue_count++ )) || true
            done <<< "$_fail_lines"
            gb_panel_empty
        fi
    fi

    # DX12 game running but VKD3D_DEBUG=none → diagnostics blind
    if (( DX12_ACTIVE )) && \
       [[ "$DX12_VKD3D_DEBUG" == "none" || "$DX12_VKD3D_DEBUG" == "—" ]]; then
        gb_panel_row "${C_AMBER}⚠${C_RESET}  ${C_AMBER}VKD3D_DEBUG=none — use greenboost-proton for automatic warn logging${C_RESET}"
        (( _issue_count++ )) || true
        gb_panel_empty
    fi

    # DX12 game active but not using greenboost-proton
    if (( DX12_ACTIVE )) && \
       [[ "$DX12_PROTON_VER" != "greenboost-proton" ]] && (( GB_PROTON_INSTALLED )); then
        gb_panel_row "${C_AMBER}⚠${C_RESET}  ${C_AMBER}DX12 game running without greenboost-proton${C_RESET}"
        gb_panel_row "   ${C_DIM}Switch in Steam: Properties → Compatibility → GreenBoost Proton${C_RESET}"
        (( _issue_count++ )) || true
        gb_panel_empty
    fi

    # Shader boost warnings from journalctl
    if command -v journalctl &>/dev/null; then
        local _boost_issues
        _boost_issues=$(journalctl -t greenboost-shader-boost \
            --since "1 hour ago" --no-pager -q 2>/dev/null \
            | grep -iE 'error|fail|warn' | tail -5)
        if [[ -n "$_boost_issues" ]]; then
            gb_panel_row "${C_AMBER}⚠${C_RESET}  ${C_AMBER}Shader boost warnings:${C_RESET}"
            while IFS= read -r _bl; do
                [[ -z "$_bl" ]] && continue
                local _bmsg
                _bmsg=$(printf '%s' "$_bl" | sed 's/.*greenboost-shader-boost[^:]*: //')
                gb_panel_row "   ${C_DIM}${_bmsg}${C_RESET}"
                (( _issue_count++ )) || true
            done <<< "$_boost_issues"
            gb_panel_empty
        fi
    fi

    # Layer not installed is itself an issue
    if [[ "$VKL_MANIFEST_OK" -eq 0 || "$VKL_SO_OK" -eq 0 ]]; then
        gb_panel_row "${C_RED}✗${C_RESET}  ${C_RED}Vulkan layer not fully installed${C_RESET}"
        [[ "$VKL_MANIFEST_OK" -eq 0 ]] && \
            gb_panel_row "   ${C_DIM}Missing: ${VKL_MANIFEST_PATH}${C_RESET}"
        [[ "$VKL_SO_OK" -eq 0 ]] && \
            gb_panel_row "   ${C_DIM}Missing: ${VKL_SO_PATH}${C_RESET}"
        gb_panel_row "   ${C_AMBER}Fix: sudo greenboost install-sys-configs${C_RESET}"
        (( _issue_count++ )) || true
        gb_panel_empty
    fi

    if (( _issue_count == 0 )); then
        gb_panel_row "  ${C_LIME}✓${C_RESET}  ${C_GRAY}No issues detected in the last hour${C_RESET}"
        gb_panel_empty
    fi

    gb_panel_bottom
    echo ""
}

# cmd_vulkan — Vulkan layer + gaming dashboard entry point.
cmd_vulkan() {
    detect_hardware 2>/dev/null || true

    # Non-interactive (pipe/script): single snapshot, no prompts
    if [[ ! -t 0 ]]; then
        gb_header
        _cmd_vulkan_snapshot
        return
    fi

    # Interactive: continuous loop with alternate screen
    printf '\033[?1049h'
    trap 'printf "\033[?1049l"; exit 0' INT TERM EXIT

    while true; do
        local _t=$SECONDS
        printf '\033[H'
        gb_header
        echo -e "  ${C_DIM}Vulkan · DX12 · VKD3D-Proton dashboard — updating every 5s — Ctrl+C to exit\033[K${C_RESET}"
        echo ""
        _cmd_vulkan_snapshot
        printf '\033[J'
        local _s=$(( 5 - (SECONDS - _t) )); (( _s < 1 )) && _s=1
        sleep $_s
    done
}

# get_tier_bw_label <tier>
# Returns a short bandwidth label for a memory tier, derived from detected
# hardware where available. Never uses hardcoded model-specific values.
#
# Usage:
#   T1_LABEL=$(get_tier_bw_label t1)   # GPU VRAM — "GDDR6X" / "GDDR7" / "VRAM"
#   T2_LABEL=$(get_tier_bw_label t2)   # PCIe — "PCIe gen N" / "PCIe DMA"
#   T3_LABEL=$(get_tier_bw_label t3)   # NVMe — "NVMe" always
get_tier_bw_label() {
    local tier="${1:-t1}"
    case "$tier" in
        t1)
            # Use GPU memory type from nvidia-smi -q -d MEMORY (works across all driver versions).
            # The CSV --query-gpu=memory.type field is not available on all drivers.
            local mem_type
            mem_type=$(nvidia-smi -q -d MEMORY 2>/dev/null \
                | grep -iP '^\s*Memory Type\s*:' | head -1 \
                | sed 's/.*:\s*//' | xargs)
            if [[ -n "$mem_type" && "$mem_type" != "N/A" && ! "$mem_type" =~ [Ff]ield ]]; then
                echo "$mem_type"
            else
                echo "VRAM"
            fi
            ;;
        t2)
            # PCIe gen from detected hardware
            local gen="${DET_PCIE_GEN_SMI:-${PCIE_MAX_GEN:-0}}"
            if (( gen > 0 )); then
                echo "PCIe gen${gen} DMA"
            else
                echo "PCIe DMA"
            fi
            ;;
        t3)
            echo "NVMe"
            ;;
    esac
}

# ════════════════════════════════════════════════════════════════════════
# ---- Single-shot status snapshot (no prompts, no loop) ----------------
# Called by cmd_status when non-interactive, and by the monitor loop each iteration.
_cmd_status_snapshot() {
    local _ts; _ts=$(date '+%Y-%m-%dT%H:%M')
    local prof_f="/sys/class/greenboost/greenboost/active_profile"
    local prof_name="default"
    [[ -r "$prof_f" ]] && prof_name=$(cat "$prof_f")

    # Gather data from sysfs and live queries
    parse_pool_brief
    parse_pool_info
    parse_shim_stats
    query_gpu_vram
    query_live_swap
    query_ollama_ps
    local kv_est_mb=0
    if [[ -n "$OL_CTX_SIZE" && -n "$OL_PARAM_COUNT" ]] && (( ${OL_CTX_SIZE:-0} > 0 )); then
        kv_est_mb=$(estimate_kv_mb "$OL_CTX_SIZE" "$OL_PARAM_COUNT")
    fi

    local _diag_errors=() _diag_warns=() _diag_ok=()

    # ── Title (1 line) ────────────────────────────────────────────────
    echo -e "  ${C_VIOLET}${C_BOLD}GreenBoost${C_RESET} ${C_DIM}v${GB_VERSION} · ${_ts}${C_RESET}"

    # ── ROW 1 (4 lines): System | AI Inference ────────────────────────
    local -a _sys_lines=() _ai_lines=()

    # Left: System
    _sys_lines+=("  ${C_BOLD}System${C_RESET}")
    if lsmod | grep -q "^${DRIVER_NAME} "; then
        _sys_lines+=("  ${C_LIME}✓${C_RESET}  ${C_GRAY}Module ${C_LIME}v${GB_VERSION}${C_RESET}")
        _diag_ok+=("kernel module loaded")
    else
        _sys_lines+=("  ${C_RED}✗${C_RESET}  ${C_GRAY}Module ${C_RED}not loaded${C_RESET}")
        _diag_errors+=("kernel module not loaded — T2/T3 memory unavailable")
    fi
    _sys_lines+=("  ${C_VIOLET}◈${C_RESET}  ${C_GRAY}$(_trunc "${prof_name}" 20)${C_RESET}")
    local _gpu_short; _gpu_short=$(_trunc "${GPU_NAME:-Unknown}" 20)
    _sys_lines+=("  ${C_GRAY}◎${C_RESET}  ${C_GRAY}${_gpu_short}${C_DIM}  ${GB_PHYS:-?} GB${C_RESET}")
    # Active alloc path badge in System column
    if [[ "$SS_STALE" == "0" && -n "$SS_ACTIVE_PATH" ]]; then
        local _sys_path_badge
        case "$SS_ACTIVE_PATH" in
            A0) _sys_path_badge="${C_LIME}${C_BOLD}●A0${C_RESET} ${C_DIM}optimal path${C_RESET}" ;;
            A)  _sys_path_badge="${C_LIME}●A${C_RESET}  ${C_DIM}fast path${C_RESET}" ;;
            B)  _sys_path_badge="${C_AMBER}${C_BOLD}⚠B${C_RESET}  ${C_DIM}fallback path${C_RESET}" ;;
            C)  _sys_path_badge="${C_RED}${C_BOLD}✗C${C_RESET}  ${C_DIM}last resort${C_RESET}" ;;
            *)  _sys_path_badge="${C_GRAY}path:${SS_ACTIVE_PATH}${C_RESET}" ;;
        esac
        _sys_lines+=("  ${C_DIM}Path${C_RESET}  ${_sys_path_badge}")
    fi

    # Right: AI Inference — Phase and TQ shown here
    local _phase_label="${SS_PHASE:-—}"
    local _tq_short
    if [[ "$PI_TURBO_STATUS" == "disabled" || -z "$PI_TURBO_STATUS" ]]; then
        _tq_short="${C_DIM}TQ: off${C_RESET}"
    else
        _tq_short="${C_LIME}TQ: ${PI_TURBO_STATUS}${C_RESET}"
    fi

    _ai_lines+=("  ${C_BOLD}AI Inference${C_RESET}")
    if [[ -n "$OL_MODEL" ]]; then
        local _model_short; _model_short=$(_trunc "${OL_MODEL}" 22)
        local _param_str=""; [[ -n "$OL_PARAM_COUNT" ]] && _param_str=" ${C_DIM}(${OL_PARAM_COUNT})${C_RESET}"
        if [[ -n "$OL_CTX_SIZE" ]] && (( ${OL_CTX_SIZE:-0} > 0 )); then
            _ai_lines+=("  ${C_LIME}✓${C_RESET}  ${C_GRAY}${_model_short}${_param_str}")
            local _ctx_k=$(( ${OL_CTX_SIZE:-0} / 1024 ))
            local _kv_str=""
            (( ${kv_est_mb:-0} > 0 )) && _kv_str=" · KV ~${kv_est_mb} MB q8_0"
            _ai_lines+=("     ${C_GRAY}Ctx ${_ctx_k}K${_kv_str}${C_RESET}")
            _diag_ok+=("Ollama: ${OL_MODEL}")
        else
            _ai_lines+=("  ${C_LIME}✓${C_RESET}  ${C_GRAY}${_model_short}${_param_str}")
            _ai_lines+=("     ${C_AMBER}⚠ Loading model...${C_RESET}")
        fi
        # Phase + TurboQuant on one line
        local _phase_color="${C_DIM}"
        case "${SS_PHASE:-}" in INFERENCE|STEADY) _phase_color="$C_LIME" ;; MODEL_LOAD) _phase_color="$C_AMBER" ;; esac
        _ai_lines+=("     ${_phase_color}${_phase_label}${C_RESET}  ${_tq_short}")
    else
        _ai_lines+=("  ${C_DIM}◎  no active model — idle${C_RESET}")
        _ai_lines+=("")
        _ai_lines+=("")
    fi

    gb_render_2col _sys_lines _ai_lines
    gb_separator

    # ── ROW 2 (4 lines): Memory Tiers (1 line each + combined) ────────
    local _combined_gb=$(( ${PB_T1_GB:-0} + ${PB_T2_MAX_GB:-0} + ${PB_T3_MAX_GB:-0} ))

    # T1 — GPU VRAM
    if (( ${GPU_VRAM_TOTAL_MB:-0} > 0 )); then
        local _t1_used_f; _t1_used_f=$(awk "BEGIN{printf \"%.1f\", ${GPU_VRAM_USED_MB:-0}/1024}" 2>/dev/null || echo "${PB_T1_GB:-0}")
        local _t1_tot=$(( ${GPU_VRAM_TOTAL_MB:-0} / 1024 ))
        local _t1_col; _t1_col=$(gb_tier_color "$GPU_VRAM_PCT")
        local _t1_bar; _t1_bar=$(gb_bar "$GPU_VRAM_PCT" "$_t1_col" "${C_DIM}" 28)
        echo -e "  ${C_CYAN}T1${C_RESET}  ${C_GRAY}GPU VRAM   ${C_RESET}${_t1_bar}  ${_t1_col}${_t1_used_f}/${_t1_tot} GB  ${GPU_VRAM_PCT}%${C_RESET}"
        (( GPU_VRAM_PCT >= 90 )) && _diag_warns+=("T1 VRAM at ${GPU_VRAM_PCT}%")
    else
        echo -e "  ${C_CYAN}T1${C_RESET}  ${C_GRAY}GPU VRAM   ${C_LIME}${PB_T1_GB} GB${C_RESET}"
    fi

    # T2 — System RAM pool
    local _t2_pct="${PB_T2_PCT:-0}"
    local _t2_col; _t2_col=$(gb_tier_color "$_t2_pct")
    local _t2_bar; _t2_bar=$(gb_bar "$_t2_pct" "$_t2_col" "${C_DIM}" 28)
    echo -e "  ${C_CYAN}T2${C_RESET}  ${C_GRAY}System RAM ${C_RESET}${_t2_bar}  ${_t2_col}${PB_T2_USED_GB}/${PB_T2_MAX_GB} GB  ${_t2_pct}%${C_RESET}"
    if [[ "$PI_OOM_GUARD" == "YES" ]]; then
        _diag_errors+=("OOM guard — T2 full; system RAM at safety limit")
    fi

    # T3 — NVMe swap
    local _t3_pct=0
    (( ${PB_T3_MAX_GB:-0} > 0 )) && _t3_pct=$(( ${PB_T3_USED_GB:-0} * 100 / ${PB_T3_MAX_GB:-1} ))
    local _t3_col; _t3_col=$(gb_tier_color "$_t3_pct")
    local _t3_bar; _t3_bar=$(gb_bar "$_t3_pct" "$_t3_col" "${C_DIM}" 28)
    echo -e "  ${C_CYAN}T3${C_RESET}  ${C_GRAY}NVMe swap  ${C_RESET}${_t3_bar}  ${_t3_col}${PB_T3_USED_GB}/${PB_T3_MAX_GB} GB  ${_t3_pct}%${C_RESET}"
    (( _t3_pct >= 90 )) && _diag_errors+=("T3 NVMe swap near cap (${_t3_pct}%) — cold weights may not evict")
    (( _t3_pct >= 75 && _t3_pct < 90 )) && _diag_warns+=("T3 NVMe swap at ${_t3_pct}%")

    echo -e "  ${C_DIM}Combined: ${C_GRAY}${C_BOLD}${_combined_gb} GB${C_RESET}"
    gb_separator

    # ── ROW 3 (4 lines): KV Cache | Overflow ─────────────────────────
    local -a _kv_lines=() _ov_lines=()

    # Left: KV Cache
    _kv_lines+=("  ${C_BOLD}KV Cache${C_RESET}")
    if (( ${PI_KV_T2_MB:-0} > 0 || ${PI_KV_T3_MB:-0} > 0 )); then
        local _kv_icon _kv_pc
        if (( ${PI_KV_T3_MB:-0} > 0 )); then
            _kv_icon="${C_RED}●${C_RESET}"; _kv_pc="${C_RED}"
            _diag_errors+=("KV in T3 (${PI_KV_T3_MB} MB) — generation speed degraded")
        else
            _kv_icon="${C_AMBER}●${C_RESET}"; _kv_pc="${C_AMBER}"
            _diag_warns+=("KV spilled to T2 DDR (${PI_KV_T2_MB} MB) — consider increasing kv_reserve_mb")
        fi
        _kv_lines+=("  ${_kv_icon}  ${C_GRAY}Placement ${_kv_pc}${PI_KV_PLACEMENT}${C_RESET}")
    else
        _kv_lines+=("  ${C_LIME}●${C_RESET}  ${C_GRAY}Placement ${C_LIME}T1 VRAM${C_RESET}")
        _diag_ok+=("KV cache in T1 VRAM (no spill)")
    fi
    _kv_lines+=("    ${C_GRAY}Rsv ${C_CYAN}${PB_KV_RSV_MB} MB${C_RESET}")
    local _kv_t1_mb=$(( (${PI_KV_TOTAL_MB:-0} > ${PI_KV_T2_MB:-0}) ? (${PI_KV_TOTAL_MB:-0} - ${PI_KV_T2_MB:-0}) : 0 ))
    _kv_lines+=("    ${C_DIM}T1:${_kv_t1_mb} T2:${PI_KV_T2_MB} T3:${PI_KV_T3_MB}${C_RESET}")

    # Right: Overflow
    local _alloc_str=""
    if [[ "$SS_STALE" == "0" ]]; then
        local _path_badge
        case "$SS_ACTIVE_PATH" in
            A0) _path_badge="${C_LIME}${C_BOLD}●A0${C_RESET} ${C_DIM}(optimal)${C_RESET}" ;;
            A)  _path_badge="${C_LIME}●A${C_RESET}  ${C_DIM}(good)${C_RESET}" ;;
            B)  _path_badge="${C_AMBER}${C_BOLD}⚠B${C_RESET}  ${C_DIM}(fallback)${C_RESET}"
                _diag_warns+=("CUDA shim on fallback B path") ;;
            C)  _path_badge="${C_RED}${C_BOLD}✗C${C_RESET}  ${C_DIM}(last resort)${C_RESET}"
                _diag_warns+=("CUDA shim on fallback C path") ;;
            *)  _path_badge="${C_GRAY}path:${SS_ACTIVE_PATH:-?}${C_RESET}" ;;
        esac
        _alloc_str="${_path_badge}  ${C_DIM}A0:${SS_PATH_A0} A:${SS_PATH_A} B:${SS_PATH_B} C:${SS_PATH_C}${C_RESET}"
    else
        _alloc_str="${C_DIM}(shim stats unavailable)${C_RESET}"
    fi
    local _oom_str
    if [[ "$PI_OOM_GUARD" == "YES" ]]; then
        _oom_str="${C_RED}OOM: active${C_RESET}"
    else
        _oom_str="${C_LIME}OOM: no${C_RESET}"
    fi

    _ov_lines+=("  ${C_BOLD}Overflow${C_RESET}")
    _ov_lines+=("  ${C_GRAY}${_alloc_str}${C_RESET}")
    _ov_lines+=("  ${_oom_str}${C_DIM} · Bufs: ${PI_ACTIVE_BUFS}${C_RESET}")
    _ov_lines+=("  ${C_DIM}T2 free: ${PI_T2_AVAIL_MB} MB · Safety: ${PI_SAFETY_RSV_MB} MB${C_RESET}")

    gb_render_2col _kv_lines _ov_lines
    gb_separator

    # ── Diagnostics (max 4 lines: ≤3 issues + 1 ok summary) ──────────
    # AppArmor: collapse to single line
    local _aa_count=0
    if [[ -d /etc/apparmor.d ]] && command -v journalctl &>/dev/null; then
        local _aa_denials
        _aa_denials=$(journalctl -k -q --no-pager -n 200 2>/dev/null \
            | grep -cE 'apparmor="DENIED".*greenboost|greenboost.*apparmor="DENIED"' || true)
        _aa_count=${_aa_denials:-0}
        (( _aa_count > 0 )) && _diag_errors+=("AppArmor denials (${_aa_count}) — run: sudo greenboost install-steam")
    fi

    # Remove blank placeholder ok entries
    local _clean_ok=()
    for _o in "${_diag_ok[@]}"; do [[ -n "$_o" ]] && _clean_ok+=("$_o"); done

    # Print max 3 error/warn lines
    local _shown=0
    for _e in "${_diag_errors[@]}"; do
        (( _shown >= 3 )) && break
        echo -e "  ${C_RED}✗${C_RESET}  ${C_RED}${_e}${C_RESET}"
        (( _shown++ ))
    done
    for _w in "${_diag_warns[@]}"; do
        (( _shown >= 3 )) && break
        echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}${_w}${C_RESET}"
        (( _shown++ ))
    done
    local _total_issues=$(( ${#_diag_errors[@]} + ${#_diag_warns[@]} ))
    if (( _total_issues > 3 && _shown >= 3 )); then
        echo -e "  ${C_DIM}… and $(( _total_issues - 3 )) more (Ctrl+L for details)${C_RESET}"
    elif (( ${#_clean_ok[@]} > 0 )); then
        # Show compact ok summary
        local _ok_short="${_clean_ok[0]}"
        (( ${#_clean_ok[@]} > 1 )) && _ok_short+=" · ${_clean_ok[1]}"
        echo -e "  ${C_LIME}✓${C_RESET}  ${C_GRAY}${_ok_short}${C_RESET}"
    fi
    echo ""

    # Append one structured line to the status log for Ctrl+L log view
    _status_log_append
}


# ---- _status_log_append — append one line to the status log -----------
# Called at end of every _cmd_status_snapshot invocation.
_status_log_append() {
    local _dir; _dir=$(dirname "$GB_STATUS_LOG")
    mkdir -p "$_dir" 2>/dev/null || return
    # Rotate if > 1 MB
    if [[ -f "$GB_STATUS_LOG" ]] && (( $(stat -c%s "$GB_STATUS_LOG" 2>/dev/null || echo 0) > 1048576 )); then
        tail -n 2000 "$GB_STATUS_LOG" > "${GB_STATUS_LOG}.tmp" 2>/dev/null \
            && mv "${GB_STATUS_LOG}.tmp" "$GB_STATUS_LOG" 2>/dev/null || true
    fi
    printf '%s path=%s vram=%s/%s t2=%s t3=%s a0=%s a=%s b=%s c=%s phase=%s model=%s\n' \
        "$(date -Iseconds)" \
        "${SS_ACTIVE_PATH:-?}" \
        "${GPU_VRAM_USED_MB:-0}" "${GPU_VRAM_TOTAL_MB:-0}" \
        "${PI_T2_ALLOC_MB:-0}" "${PI_T3_ALLOC_MB:-0}" \
        "${SS_PATH_A0:-0}" "${SS_PATH_A:-0}" "${SS_PATH_B:-0}" "${SS_PATH_C:-0}" \
        "${SS_PHASE:-?}" \
        "${OL_MODEL:--}" \
        >> "$GB_STATUS_LOG" 2>/dev/null || true
}

# ---- _show_log_view — Ctrl+L log view within the status alternate screen
# Displays GB_STATUS_LOG; Ctrl+S or q returns to live status.
_show_log_view() {
    local _log="${1:-$GB_STATUS_LOG}"
    while true; do
        printf '\033[H\033[J'   # cursor home + clear screen
        local _rows; _rows=$(tput lines 2>/dev/null || echo 24)
        local _cols; _cols=$(tput cols  2>/dev/null || echo 80)
        echo -e "  ${C_CYAN}${C_BOLD}GreenBoost Status Log${C_RESET}  ${C_DIM}Ctrl+S: back to status  q: back  Ctrl+C: exit\033[K${C_RESET}"
        echo -e "  ${C_DIM}Log: ${C_GRAY}${_log}\033[K${C_RESET}"
        gb_separator
        local _content_rows=$(( _rows - 5 ))
        (( _content_rows < 5 )) && _content_rows=5
        if [[ -f "$_log" ]]; then
            tail -n "$_content_rows" "$_log" | while IFS= read -r _line; do
                # Colour-code by path: A0→lime, A→lime-dim, B→amber, C→red
                local _coloured="$_line"
                if [[ "$_line" =~ path=A0 ]]; then
                    _coloured="${C_LIME}${_line}${C_RESET}"
                elif [[ "$_line" =~ path=A[^0] ]]; then
                    _coloured="${C_GRAY}${_line}${C_RESET}"
                elif [[ "$_line" =~ path=B ]]; then
                    _coloured="${C_AMBER}${_line}${C_RESET}"
                elif [[ "$_line" =~ path=C ]]; then
                    _coloured="${C_RED}${_line}${C_RESET}"
                fi
                echo -e "  ${_coloured}\033[K"
            done
        else
            echo ""
            echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}No log yet — status refreshes write entries here.${C_RESET}"
            echo -e "  ${C_DIM}Log will appear at: ${C_GRAY}${_log}${C_RESET}"
        fi
        printf '\033[J'   # erase trailing lines
        local _k=""
        read -t 2 -s -n 1 _k 2>/dev/null || true
        case "$_k" in
            $'\x13'|q) return ;;   # Ctrl+S or q → back to status
        esac
    done
}

cmd_status() {
    detect_hardware 2>/dev/null || true

    # Non-interactive: single snapshot, no prompts, no infinite loop
    # Safe for use from scripts, pipes, and synapse CLI bash tool
    if [[ ! -t 0 ]]; then
        _cmd_status_snapshot
        return
    fi

    # Save terminal settings and disable XON/XOFF so Ctrl+S is capturable
    local _saved_stty
    _saved_stty=$(stty -g 2>/dev/null || true)
    stty -ixon 2>/dev/null || true

    # Enter alternate screen so terminal history is preserved; restore on exit
    printf '\033[?1049h'
    trap 'stty "$_saved_stty" 2>/dev/null || true; printf "\033[?1049l"; exit 0' INT TERM EXIT

    while true; do
        local _t_start=$SECONDS

        printf '\033[H'   # cursor home — no erase, no flash
        echo -e "  ${C_DIM}Updating every 5s — ${C_GRAY}Ctrl+S${C_DIM}: refresh  ${C_GRAY}Ctrl+L${C_DIM}: log view  ${C_GRAY}Ctrl+C${C_DIM}: exit\033[K${C_RESET}"
        echo ""

        _cmd_status_snapshot
        printf '\033[J'   # erase any trailing lines from a previous taller draw

        local _elapsed=$(( SECONDS - _t_start ))
        local _sleep=$(( 5 - _elapsed ))
        (( _sleep < 1 )) && _sleep=1

        local _key=""
        read -t "$_sleep" -s -n 1 _key 2>/dev/null || true

        case "$_key" in
            $'\x13')  # Ctrl+S: immediate status refresh
                ;;
            $'\x0c')  # Ctrl+L: switch to log view (Ctrl+S inside to return)
                _show_log_view "$GB_STATUS_LOG"
                # Return to status: clear and restart the draw loop
                printf '\033[H\033[J'
                ;;
        esac
    done
}


# ---- show-commands (greenboost help) -----------------------------------
# Displays GREENBOOST_COMMANDS.md as a styled terminal reference.
# Accessible via: greenboost help  /  greenboost_setup.sh show-commands
# and from the wizard "GreenBoost Commands" menu entry.

cmd_show_commands() {
    local doc=""
    for loc in \
        "$MODULE_DIR/GREENBOOST_COMMANDS.md" \
        "$(dirname "$(realpath "$0")")/GREENBOOST_COMMANDS.md" \
        "/usr/local/share/greenboost/GREENBOOST_COMMANDS.md"; do
        [[ -f "$loc" ]] && doc="$loc" && break
    done

    gb_header
    echo -e "  ${C_CYAN}${C_BOLD}GreenBoost Command Reference${C_RESET}  ${C_DIM}— run ${C_GRAY}greenboost help${C_DIM} to see this anytime${C_RESET}"
    echo -e ""

    if [[ -z "$doc" ]]; then
        gb_warn_ui "GREENBOOST_COMMANDS.md not found — showing built-in summary"
        echo -e ""
        printf "  ${C_LIME}%-28s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "greenboost status"          "cuda memory pool + system health"
        printf "  ${C_LIME}%-28s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "greenboost vulkan"          "Vulkan layer, active games, DMA-BUF fallback activity, shader boost"
        printf "  ${C_LIME}%-28s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "greenboost clean memory"    "Force-release T1 VRAM + T2 RAM + T3 now"
        printf "  ${C_LIME}%-28s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "greenboost run <app>"       "Force-activate shim for one app (non-login shells/scripts)"
        printf "  ${C_LIME}%-28s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "sudo greenboost setup"      "Full first-time install"
        printf "  ${C_LIME}%-28s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "sudo greenboost tune"       "Apply CPU governor + NVMe + sysctl tuning"
        printf "  ${C_LIME}%-28s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "greenboost profile"         "Open interactive profile wizard"
        printf "  ${C_LIME}%-28s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "greenboost benchmark"       "cuda memory pool bandwidth benchmark (T1 VRAM / T2 DDR / T3 NVMe)"
        echo -e ""
        gb_info "Idle reclaim: T2 RAM released after 2 min idle. T1 VRAM (KV cache) released after 17 min idle."
        gb_info "Manual:       greenboost clean memory"
        return 0
    fi

    # Pretty-print the markdown with section headers styled in Cyan and commands in Lime
    local _in_code=0
    while IFS= read -r line; do
        if [[ "$line" =~ ^\`\`\` ]]; then
            _in_code=$(( 1 - _in_code ))
        elif [[ $_in_code -eq 1 ]]; then
            # Code block content — full white for contrast
            [[ -n "$line" ]] && echo -e "  ${C_WHITE}${line}${C_RESET}" || echo ""
        elif [[ "$line" =~ ^##[[:space:]]+(.*) ]]; then
            echo -e ""
            echo -e "  ${C_CYAN}${C_BOLD}${BASH_REMATCH[1]}${C_RESET}"
            echo -e "  ${C_DIM}$(printf '─%.0s' {1..56})${C_RESET}"
        elif [[ "$line" =~ ^\|[[:space:]]*\`([^\`]+)\`[[:space:]]*\|(.*)\|$ ]]; then
            # Table row with a command in backticks: highlight command Lime, desc Gray
            printf "  ${C_LIME}%-38s${C_RESET} ${C_GRAY}%s${C_RESET}\n" \
                "${BASH_REMATCH[1]}" "$(echo "${BASH_REMATCH[2]}" | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')"
        elif [[ "$line" =~ ^#[[:space:]] ]]; then
            : # skip H1 title (shown in header already)
        elif [[ "$line" =~ ^\|.*\| ]]; then
            : # skip table separator rows and header rows (non-command lines)
        elif [[ "$line" =~ ^--- ]]; then
            echo -e ""
        else
            # Normal text — dim gray
            [[ -n "$line" ]] && echo -e "  ${C_DIM}${line}${C_RESET}" || echo ""
        fi
    done < "$doc"

    echo -e ""
    gb_info "Full document: $doc"
}

# ── cmd_turboquant — KV cache TurboQuant compression control ────────────
# Usage: greenboost turboquant [status|enable [2|3|4]|disable]
#
# Reads /run/greenboost/turboquant.conf and /dev/greenboost for status.
# enable/disable write the conf file and issue GB_IOCTL_SET_TURBOQUANT.
cmd_turboquant() {
    local subcmd="${1:-status}"

    local conf_path="/run/greenboost/turboquant.conf"
    local dev_path="/dev/greenboost"

    case "$subcmd" in
        status|"")
            gb_section "TurboQuant KV Cache Compression"

            # Read conf file
            local enabled=0 bits=0 head_dim=128 ratio=1.0
            if [[ -f "$conf_path" ]]; then
                while IFS='=' read -r key val; do
                    case "$key" in
                        enabled)  enabled="$val"  ;;
                        bits)     bits="$val"      ;;
                        head_dim) head_dim="$val"  ;;
                        ratio)    ratio="$val"     ;;
                    esac
                done < "$conf_path"
            fi

            if [[ "$enabled" == "1" && "$bits" -gt 0 ]]; then
                gb_ok "TurboQuant:   turbo${bits} active (${ratio}× compression, head_dim=${head_dim})"
            else
                gb_info "TurboQuant:   disabled (KV cache stored uncompressed)"
            fi

            # Show sysfs stats if module is loaded
            local sysfs="/sys/class/greenboost/greenboost/status"
            if [[ -r "$sysfs" ]]; then
                local kv_cmp_mb kv_cmp_bits
                kv_cmp_mb=$(grep -oP 'KV compressed\s*:\s*\K\d+' "$sysfs" 2>/dev/null || echo "0")
                kv_cmp_bits=$(grep -oP 'compression bits\s*:\s*\K\d+' "$sysfs" 2>/dev/null || echo "0")
                [[ "$kv_cmp_mb" -gt 0 ]] && gb_info "Saved by TQ:  ${kv_cmp_mb} MB"
            fi

            gb_info "Daemon:       greenboost-turboquant.service (auto-selects bit width)"
            gb_info "Manual ctl:   greenboost turboquant enable [2|3|4]"
            gb_info "Conf file:    ${conf_path}"
            ;;

        enable)
            local req_bits="${2:-0}"
            if [[ "$req_bits" == "0" ]]; then
                # Auto: let daemon pick; just start/restart it
                gb_step 1 2 "Enabling TurboQuant (auto bit selection)"
                if systemctl is-active --quiet greenboost-turboquant.service 2>/dev/null; then
                    systemctl restart greenboost-turboquant.service
                    gb_ok "Restarted greenboost-turboquant daemon (auto mode)"
                else
                    systemctl start greenboost-turboquant.service 2>/dev/null && \
                        gb_ok "Started greenboost-turboquant daemon" || \
                        gb_warn_ui "greenboost-turboquant.service not installed — run full-install first"
                fi
            else
                if [[ "$req_bits" != "2" && "$req_bits" != "3" && "$req_bits" != "4" ]]; then
                    gb_fail "Invalid bit width: ${req_bits}. Use 2, 3, or 4."
                    return 1
                fi
                gb_step 1 2 "Enabling TurboQuant turbo${req_bits}"
                local ratio
                case "$req_bits" in
                    4) ratio="3.90" ;;
                    3) ratio="4.60" ;;
                    2) ratio="6.40" ;;
                esac

                # Write conf file atomically
                mkdir -p /run/greenboost
                {
                    echo "enabled=1"
                    echo "bits=${req_bits}"
                    echo "head_dim=128"
                    echo "ratio=${ratio}"
                    echo "seed=42"
                } > "${conf_path}.tmp" && mv "${conf_path}.tmp" "$conf_path"
                gb_ok "Wrote ${conf_path} (turbo${req_bits}, ratio ${ratio}×)"

                # Send IOCTL if /dev/greenboost is available
                if [[ -c "$dev_path" ]]; then
                    gb_step 2 2 "Sending GB_IOCTL_SET_TURBOQUANT"
                    python3 -c "
import fcntl, struct, os
_IOC_WRITE = 1
def _iow(m, n, s): return (_IOC_WRITE<<30)|(s<<16)|(m<<8)|n
GB_IOCTL_SET_TURBOQUANT = _iow(ord('G'), 10, 16)
req = struct.pack('IIII', 1, ${req_bits}, 128, 42)
fd = os.open('${dev_path}', os.O_RDWR | os.O_CLOEXEC)
fcntl.ioctl(fd, GB_IOCTL_SET_TURBOQUANT, req)
os.close(fd)
print('[GreenBoost] GB_IOCTL_SET_TURBOQUANT sent: enabled=1 bits=${req_bits}')
" 2>&1 && gb_ok "IOCTL sent to ${dev_path}" || gb_warn_ui "IOCTL failed — conf file is active, shim will pick up on next refresh"
                fi
            fi
            ;;

        disable)
            gb_step 1 2 "Disabling TurboQuant"
            rm -f "$conf_path"
            gb_ok "Removed ${conf_path}"

            if [[ -c "$dev_path" ]]; then
                gb_step 2 2 "Sending disable IOCTL"
                python3 -c "
import fcntl, struct, os
_IOC_WRITE = 1
def _iow(m, n, s): return (_IOC_WRITE<<30)|(s<<16)|(m<<8)|n
GB_IOCTL_SET_TURBOQUANT = _iow(ord('G'), 10, 16)
req = struct.pack('IIII', 0, 0, 128, 42)
fd = os.open('${dev_path}', os.O_RDWR | os.O_CLOEXEC)
fcntl.ioctl(fd, GB_IOCTL_SET_TURBOQUANT, req)
os.close(fd)
" 2>&1 && gb_ok "IOCTL sent: TurboQuant disabled" || gb_warn_ui "IOCTL failed — conf file removed, shim will disable on next refresh"
            fi
            ;;

        *)
            gb_warn_ui "Unknown sub-command: ${subcmd}"
            gb_info "Usage: greenboost turboquant [status|enable [2|3|4]|disable]"
            return 1
            ;;
    esac
}

# ---- wizard (default interactive mode) --------------------------------
# Shown when no arguments are given and stdin is a TTY.

cmd_wizard() {
    while true; do
        clear
        gb_header
        echo -e "  ${C_DIM}${C_GRAY}GreenBoost was built to improve local AI inference by orchestrating a CUDA memory pool.${C_RESET}"
        echo -e "  ${C_DIM}${C_GRAY}The Vulkan/gaming layer is a secondary byproduct.${C_RESET}"
        echo ""

        gb_section "Core"
        gb_menu_item  1  "Full install"             "DKMS module + AI libs + sysctl + GRUB + systemd services (hardware-agnostic)"  root
        gb_menu_item  2  "Light install"            "DKMS module + hardware-tuned build — no system changes (hardware-agnostic)"  root
        gb_menu_item  3  "Status"                   "Show cuda memory pool + system state"
        gb_menu_item  4  "Benchmark"                "Measure T1/T2/T3 bandwidth"

        gb_section "Configuration"
        gb_menu_item  5  "Profile management"       "Interactive wizard: create, activate, diff profiles"

        gb_section "Gaming"
        gb_menu_item  6  "Install GB Proton"        "T1+T2 CUDA memory pool for Steam games. Select GreenBoost Proton on Steam"  root
        gb_menu_item  7  "Remove GB Proton"         "Uninstall GreenBoost Proton from Steam compat tools"
        gb_menu_item  8  "Clean custom Protons"     "Remove all non-GreenBoost custom Proton builds from Steam compat tools"
        gb_menu_item  9  "Install MangoHud"         "Build MangoHud from source + GreenBoost overlay for benchmarking"  root
        gb_menu_item  10 "Remove MangoHud"          "Uninstall MangoHud binaries and MangoHud.conf"  root

        gb_section "Maintenance"
        gb_menu_item  11 "GreenBoost Commands"     "All commands reference (also: greenboost help)"
        gb_menu_item  12 "Clear logs"               "Clear dmesg, journal, Proton logs, and Wine coredumps"
        gb_menu_item  13 "Uninstall"                "Remove GreenBoost (module + all config)"        root

        gb_separator
        echo -e "  ${C_DIM}${C_GRAY}[Q]  Quit${C_RESET}"
        echo -e ""

        if [[ "$(id -u)" != "0" ]]; then
            gb_warn_ui "Not running as root — options marked ${C_RED}⚠ root${C_AMBER} will fail. Use sudo."
            echo -e ""
        fi

        gb_prompt "Choice"
        local choice="$REPLY"
        echo -e ""

        case "$choice" in
            1)  cmd_full_install;              gb_press_enter ;;
            2)  bash "$MODULE_DIR/install_module.sh"; gb_press_enter ;;
            3)  cmd_status ;;
            4)  cmd_benchmark;                 gb_press_enter ;;
            5)  cmd_profile_wizard ;;
            6)  cmd_install_proton;             gb_press_enter ;;
            7)  cmd_uninstall_proton;          gb_press_enter ;;
            8)  cmd_proton_clean;              gb_press_enter ;;
            9)  cmd_install_mangohud;          gb_press_enter ;;
            10) cmd_uninstall_mangohud;        gb_press_enter ;;
            11) cmd_show_commands;             gb_press_enter ;;
            12) cmd_clear_logs;                gb_press_enter ;;
            13) cmd_uninstall;                 gb_press_enter ;;
            q|Q|"") exit 0 ;;
            *) gb_warn_ui "Unknown option."; sleep 1 ;;
        esac
    done
}

# ── cmd_benchmark — workstation bandwidth benchmark ──────────────────────
# Usage: greenboost benchmark [--skip-bandwidth] [--json]
#
# cuda memory pool bandwidth test (T1 VRAM / T2 DDR / T3 NVMe)
# Results logged to /var/log/greenboost/benchmark-<timestamp>.log
cmd_benchmark() {
    local skip_bw=0 json_out=0
    for arg in "$@"; do
        case "$arg" in
            --skip-bandwidth)  skip_bw=1  ;;
            --json)            json_out=1 ;;
        esac
    done

    # ── Log setup ──────────────────────────────────────────────────────
    local log_dir="/var/log/greenboost"
    local ts; ts=$(date +%Y%m%d_%H%M%S)
    local log_file="${log_dir}/benchmark-${ts}.log"
    mkdir -p "$log_dir" 2>/dev/null
    # Symlink latest for easy access
    ln -sfn "${log_file}" "${log_dir}/benchmark-latest.log" 2>/dev/null

    _bm_log() { printf '%s\n' "$*" | tee -a "$log_file" >/dev/null; }
    _bm_log "# GreenBoost benchmark  ts=${ts}  version=${GB_VERSION}"

    # ── Header ─────────────────────────────────────────────────────────
    gb_header
    echo -e "  ${C_GRAY}Log: ${C_DIM}${log_file}${C_RESET}"
    echo -e ""

    # ══════════════════════════════════════════════════════════════════
    # CUDA Memory Pool Bandwidth
    # ══════════════════════════════════════════════════════════════════
    if [[ $skip_bw -eq 0 ]]; then
        gb_section "Memory Bandwidth Benchmark"

        local bench_script=""
        for loc in \
            "$MODULE_DIR/tools/gb_workstation_bench.py" \
            "$MODULE_DIR/../greenboost_python/gb_workstation_bench.py"; do
            [[ -f "$loc" ]] && bench_script="$loc" && break
        done

        if [[ -z "$bench_script" ]]; then
            gb_warn_ui "Bandwidth benchmark script not found — skipping"
            gb_warn_ui "Expected: $MODULE_DIR/tools/gb_workstation_bench.py"
            _bm_log "phase=bandwidth status=skipped reason=script_not_found"
        else
            local py="python3"

            gb_panel_top "Memory Bandwidth Benchmark"
            gb_panel_empty

            # Detect hardware for display
            detect_hardware 2>/dev/null
            gb_panel_row "  ${C_VIOLET}◈${C_RESET}  ${C_GRAY}Hardware: ${C_RESET}${C_CYAN}${GPU_NAME:-Unknown}${C_RESET}  ${C_DIM}·  ${GB_PHYS:-?} GB T1  ·  ${GB_VIRT:-?} GB T2  ·  ${GB_NVME_SWAP:-?} GB T3${C_RESET}"
            gb_panel_empty

            local bw_args=()
            [[ $json_out -eq 1 ]] && bw_args+=("--json")

            if [[ $json_out -eq 0 ]]; then
                # Run with live spinner output; also write JSON to a temp file for parsing
                gb_panel_row "  ${C_VIOLET}◈${C_RESET}  ${C_GRAY}Measuring tier bandwidths (this takes ~30s)...${C_RESET}"
                gb_panel_empty

                local bw_tmpout; bw_tmpout=$(mktemp)
                local bw_json_tmp; bw_json_tmp=$(mktemp --suffix=.json)
                "$py" "$bench_script" --output "$bw_json_tmp" >"$bw_tmpout" 2>&1 &
                local bw_pid=$!
                local bw_i=0 bw_start; bw_start=$(date +%s)
                while kill -0 "$bw_pid" 2>/dev/null; do
                    local bw_elapsed=$(( $(date +%s) - bw_start ))
                    local bw_ts
                    if (( bw_elapsed >= 60 )); then
                        bw_ts=$(printf "%d:%02d" $((bw_elapsed/60)) $((bw_elapsed%60)))
                    else
                        bw_ts="${bw_elapsed}s"
                    fi
                    printf "\r  ${C_LIME}%s${C_RESET}  ${C_DIM}Running bandwidth test...${C_RESET} ${C_GRAY}[%s]${C_RESET}   " \
                        "${GB_SPIN_FRAMES[$((bw_i % ${#GB_SPIN_FRAMES[@]}))]}" "$bw_ts"
                    sleep 0.08
                    (( bw_i++ )) || true
                done
                printf "\r  ${C_LIME}✓${C_RESET}  ${C_GRAY}Bandwidth test complete${C_RESET}                        \n"

                # Parse results from JSON output (avoids ANSI-code interference in regex)
                local bw_output; bw_output=$(cat "$bw_tmpout")
                rm -f "$bw_tmpout"
                _bm_log "phase=bandwidth status=complete"

                # Extract bandwidth values (GB/s) and methods from JSON
                local t1_bw t2_bw t3_bw t1_method t2_method
                t1_bw=$(python3     -c "import json; d=json.load(open('$bw_json_tmp')); v=d.get('t1',{}).get('bandwidth_gbs'); print(v if v is not None else '')" 2>/dev/null || true)
                t2_bw=$(python3     -c "import json; d=json.load(open('$bw_json_tmp')); v=d.get('t2',{}).get('bandwidth_gbs'); print(v if v is not None else '')" 2>/dev/null || true)
                t3_bw=$(python3     -c "import json; d=json.load(open('$bw_json_tmp')); v=d.get('t3',{}).get('seq_read_gbs'); print(v if v is not None else '')" 2>/dev/null || true)
                t1_method=$(python3 -c "import json; d=json.load(open('$bw_json_tmp')); print(d.get('t1',{}).get('method',''))" 2>/dev/null || true)
                t2_method=$(python3 -c "import json; d=json.load(open('$bw_json_tmp')); print(d.get('t2',{}).get('method',''))" 2>/dev/null || true)
                rm -f "$bw_json_tmp"

                # Fallback: show raw output if parse fails
                if [[ -z "$t1_bw" && -z "$t2_bw" && -z "$t3_bw" ]]; then
                    gb_panel_bottom
                    echo -e ""
                    printf '%s\n' "$bw_output"
                    _bm_log "phase=bandwidth raw_output=${bw_output}"
                else
                    # Render results as bars relative to T1
                    local t1_ref="${t1_bw:-336}"
                    local _render_bw_row
                    _render_bw_row() {
                        local tier="$1" bw="$2" label="$3" ref="$4"
                        local pct=0
                        if [[ -n "$bw" && -n "$ref" && "$ref" != "0" ]]; then
                            pct=$(( ${bw%%.*} * 100 / ${ref%%.*} ))
                            (( pct > 100 )) && pct=100
                        fi
                        local bar_c; bar_c=$(gb_tier_color "$pct")
                        local bar; bar=$(gb_bar "$pct" "$bar_c" "${C_DIM}" 28)
                        gb_panel_row "  ${C_LIME}✓${C_RESET}  ${C_CYAN}${C_BOLD}${tier}${C_RESET}  ${label}  ${bar_c}${bw:-N/A} GB/s${C_RESET}  ${bar}  ${C_DIM}${pct}%${C_RESET}"
                    }
                    # Label estimated measurements distinctly
                    local _t1_lbl="${C_DIM}VRAM   ${C_RESET}"
                    local _t2_lbl="${C_DIM}DDR    ${C_RESET}"
                    [[ "$t1_method" == "theoretical_lookup" || "$t1_method" == "theoretical_gddr7" ]] && \
                        _t1_lbl="${C_DIM}VRAM ${C_AMBER}~est${C_RESET} "
                    [[ "$t2_method" == "numpy_memcopy" ]] && \
                        _t2_lbl="${C_DIM}DDR ${C_AMBER}CPU  ${C_RESET}"
                    _render_bw_row "T1" "$t1_bw" "$_t1_lbl" "$t1_ref"
                    _render_bw_row "T2" "$t2_bw" "$_t2_lbl" "$t1_ref"
                    _render_bw_row "T3" "$t3_bw" "${C_DIM}NVMe   ${C_RESET}" "$t1_ref"
                    gb_panel_empty

                    # Speedup ratios
                    if [[ -n "$t1_bw" && -n "$t2_bw" && "${t2_bw%%.*}" -gt 0 ]]; then
                        local r12=$(( ${t1_bw%%.*} / ${t2_bw%%.*} ))
                        gb_panel_row "  ${C_GRAY}T1→T2 ratio  ${C_RESET}${C_AMBER}${r12}×${C_RESET} faster"
                    fi
                    if [[ -n "$t1_bw" && -n "$t3_bw" && "${t3_bw%%.*}" -gt 0 ]]; then
                        local r13=$(( ${t1_bw%%.*} / ${t3_bw%%.*} ))
                        gb_panel_row "  ${C_GRAY}T1→T3 ratio  ${C_RESET}${C_AMBER}${r13}×${C_RESET} faster"
                    fi
                    gb_panel_empty
                    _bm_log "phase=bandwidth t1_bw=${t1_bw} t2_bw=${t2_bw} t3_bw=${t3_bw}"
                fi
                gb_panel_bottom
            else
                # JSON mode: pass through directly
                "$py" "$bench_script" "${bw_args[@]}" | tee -a "$log_file"
            fi
        fi
        echo -e ""
    fi

    # ── Footer ─────────────────────────────────────────────────────────
    gb_separator
    echo -e "  ${C_LIME}✓${C_RESET}  ${C_GRAY}Benchmark complete${C_RESET}"
    echo -e "  ${C_DIM}Results: ${log_file}${C_RESET}"
    echo -e ""
}

cmd_help() {
    local show_all=0
    for arg in "$@"; do [[ "$arg" == "--all" ]] && show_all=1; done

    detect_hardware 2>/dev/null
    gb_header

    echo -e "  ${C_GRAY}${CPU_NAME:-Unknown CPU}  ·  ${GPU_NAME:-Unknown GPU} ${GB_PHYS} GB  ·  ${RAM_TYPE:-DDR4}-${RAM_SPEED_MT} MT/s  ·  ${NVME_SIZE_GB} GB NVMe${C_RESET}"
    echo -e "  ${C_DIM}T1 ${GB_PHYS} GB VRAM  |  T2 ${GB_VIRT} GB DDR  |  T3 ${GB_NVME_SWAP} GB NVMe  =  $((GB_PHYS + GB_VIRT + GB_NVME_SWAP)) GB combined${C_RESET}"
    echo -e ""
    gb_separator
    echo -e ""

    if [[ $show_all -eq 0 ]]; then
        echo -e "  ${C_CYAN}${C_BOLD}USAGE:${C_RESET}  ${C_GRAY}sudo ./greenboost_setup.sh <command>${C_RESET}"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}COMMON COMMANDS:${C_RESET}"
        echo -e ""
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "setup"       "Full install — deps, module, tune, Python inference tools"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "status"      "Show pool info + system state"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "benchmark"   "cuda memory pool bandwidth benchmark (T1 VRAM / T2 DDR / T3 NVMe)  [--skip-bandwidth] [--json]"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "tune"        "Runtime tuning (CPU governor, NVMe, THP, sysctl)"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "load"        "Load kernel module with cuda memory pool params"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "unload"      "Unload module"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "uninstall"   "Remove module + all config"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "profile"     "Interactive profile wizard (create / activate / diff)"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clean memory" "Force-release T1 VRAM + T2 RAM + T3 now (unloads inference models)"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "logs"         "Snapshot all GreenBoost log sources (kernel, Ollama, Vulkan, Proton)"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clear logs"   "Clear all GreenBoost logs for a fresh diagnostic baseline"
        echo -e ""
        echo -e "  ${C_DIM}Run ${C_GRAY}greenboost help${C_DIM} for the full command reference (always available after install).${C_RESET}"
        echo -e "  ${C_DIM}Run without arguments for the interactive wizard (includes ${C_GRAY}GreenBoost Commands${C_DIM} entry).${C_RESET}"
        echo -e "  ${C_DIM}Run ${C_GRAY}help --all${C_DIM} for environment variables and advanced flags.${C_RESET}"
    else
        echo -e "  ${C_CYAN}${C_BOLD}USAGE:${C_RESET}  ${C_GRAY}sudo ./greenboost_setup.sh <command>${C_RESET}"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}MAIN COMMANDS:${C_RESET}"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "setup"           "Full install (deps + module + shim + configs + tune + Python tools)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "install"         "Build and install module + CUDA shim system-wide"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "uninstall"       "Unload, remove module + all config files"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "build"           "Build only (no system install)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "load"            "Load module with default cuda memory pool parameters"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "unload"          "Unload module (keeps installed files)"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}TUNING:${C_RESET}"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "tune"            "Runtime tuning (governor, NVMe, THP, sysctl)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "tune-grub"       "Fix GRUB boot params (THP=always, rcu_nocbs, nohz_full…)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "tune-sysctl"     "Consolidate sysctl + apply compute-optimized knobs"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "tune-libs"       "Install missing AI/compute libraries (OpenBLAS, hwloc…)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "tune-all"        "Run tune + tune-grub + tune-sysctl + tune-libs"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}DIAGNOSTICS:${C_RESET}"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "status"          "Show module status and cuda memory pool"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clean memory"    "Force-release T1 VRAM + T2 RAM + T3 immediately (unloads models)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "benchmark [flags]" "cuda memory pool bandwidth benchmark (--skip-bandwidth --json)"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}LOGGING:${C_RESET}"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "logs"                  "Snapshot all GreenBoost log sources (kernel, Ollama, Vulkan, Proton)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "proton-logs"           "Show Proton/VKD3D game logs"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "inference-logs"        "Show Ollama/inference logs"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "inference-test"        "Benchmark inference + verify fastest path (A0/A); --llm for LLM report"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clear logs"            "Clear all GreenBoost logs"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clear proton-logs"     "Clear Proton logs"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clear inference-logs"  "Clear inference logs"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}GAMING:${C_RESET}"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "gaming-mode enable|disable" "Suspend/restore Ollama before/after gaming"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "install-proton" "Install GreenBoost Proton for Steam"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "remove-proton"          "Remove GreenBoost Proton"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "steam-launch-guide"     "Show Steam launch options + Proton setup guide"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "install-mangohud"       "Build MangoHud from source + install GreenBoost config"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "uninstall-mangohud"     "Remove MangoHud"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "fix-steam"              "Install Steam launch wrapper (GTK2/XIM SIGBUS fix)"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}━━━ Required: select the Proton version in Steam ━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
        echo -e ""
        echo -e "  ${C_VIOLET}◈${C_RESET}  Right-click the game → ${C_BOLD}Properties → Compatibility${C_RESET}"
        echo -e "     Check ${C_BOLD}\"Force the use of a specific Steam Play compatibility tool\"${C_RESET}"
        echo -e "     and select ${C_LIME}${C_BOLD}GreenBoost Proton${C_RESET}"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}━━━ Enable per game in Steam launch options ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
        echo -e ""
        echo -e "  ${C_AMBER}▸ With HDR:${C_RESET}"
        echo -e "  ${C_LIME}PROTON_ENABLE_HDR=1 %command%${C_RESET}"
        echo -e ""
        echo -e "  ${C_AMBER}▸ With DLSS Super Resolution preset override:${C_RESET}"
        echo -e ""
        echo -e "    ${C_DIM}  Preset       Quality                   Best for${C_RESET}"
        echo -e "    ${C_DIM}  ──────────   ────────────────────────  ──────────────────${C_RESET}"
        echo -e "    ${C_DIM}  M (Heavier)  Highest quality           RTX 40/50 series${C_RESET}"
        echo -e "    ${C_DIM}  L (Balanced) Good quality/perf balance  Any RTX${C_RESET}"
        echo -e "    ${C_DIM}  K (Lighter)  Better performance         RTX 20/30 series${C_RESET}"
        echo -e ""
        echo -e "  ${C_AMBER}  Example — preset M:${C_RESET}"
        echo -e "  ${C_LIME}DXVK_NVAPI_DRS_NGX_DLSS_SR_OVERRIDE_RENDER_PRESET_SELECTION=render_preset_m %command%${C_RESET}"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}ADVANCED:${C_RESET}"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "install-sys-configs"   "Ollama env, NVMe udev, CPU governor, hugepages, LD_AUDIT"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "install-llama-configs" "LD_AUDIT library → ld.so.preload"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "profile [sub]"         "Interactive wizard; or: create / show / list / activate / diff"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}GLOBAL FLAGS:${C_RESET}"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "--skip-update-check"   "Bypass GitLab version check (offline)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "--profile <file>"      "Load module parameters from profile file"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}ENVIRONMENT (load command):${C_RESET}"
        printf "  ${C_DIM}%-26s${C_RESET} ${C_DIM}%s${C_RESET}\n" "GPU_PHYS_GB=${GB_PHYS}"     "Physical VRAM in GB (detected: ${GPU_NAME})"
        printf "  ${C_DIM}%-26s${C_RESET} ${C_DIM}%s${C_RESET}\n" "VIRT_VRAM_GB=${GB_VIRT}"    "System RAM pool size in GB"
        printf "  ${C_DIM}%-26s${C_RESET} ${C_DIM}%s${C_RESET}\n" "RESERVE_GB=${GB_RESERVE}"   "System RAM to keep free"
        printf "  ${C_DIM}%-26s${C_RESET} ${C_DIM}%s${C_RESET}\n" "NVME_SWAP_GB=${GB_NVME_SWAP}" "NVMe swap capacity in GB"
        printf "  ${C_DIM}%-26s${C_RESET} ${C_DIM}%s${C_RESET}\n" "NVME_POOL_GB=${GB_NVME_POOL}" "GreenBoost T3 soft cap in GB"
        echo -e ""
        echo -e "  ${C_DIM}Example: sudo VIRT_VRAM_GB=48 NVME_SWAP_GB=64 ./greenboost_setup.sh load${C_RESET}"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}MONITORING:${C_RESET}"
        echo -e "  ${C_DIM}greenboost status${C_RESET}"
        echo -e "  ${C_DIM}dmesg | grep greenboost | tail -20${C_RESET}"
        echo -e "  ${C_DIM}watch -n1 free -h   # T2 RAM pressure${C_RESET}"
        echo -e "  ${C_DIM}watch -n1 swapon --show   # T3 NVMe usage${C_RESET}"
    fi
    echo -e ""
}

# ---- install-mangohud --------------------------------------------------
# Clone MangoHud from GitHub and build it with GreenBoost integration.
# Source: https://github.com/flightlessmango/MangoHud
# Also installs MangoHud.conf with T1/T2/T3 GreenBoost pool metrics.

MANGOHUD_REPO="https://github.com/flightlessmango/MangoHud"
MANGOHUD_SRC="/opt/MangoHud"
MANGOHUD_BUILD_DIR="$MANGOHUD_SRC/build"
MANGOHUD_PREFIX="/usr/local"

# Detecta PCI bus ID del primer GPU NVIDIA para pci_dev en MangoHud.conf
_mangohud_pci_dev() {
    nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null \
        | head -1 \
        | sed 's/^00000000://'   # strip PCI domain prefix (00000000:)
}

cmd_install_mangohud() {
    need_root install-mangohud

    local real_user="${SUDO_USER:-$USER}"
    local real_home
    real_home="$(getent passwd "$real_user" | cut -d: -f6)"

    gb_header
    echo -e "  ${C_GRAY}MangoHud — build from source with GreenBoost integration${C_RESET}"
    echo ""

    # 0. Pre-clean: uninstall any previous GreenBoost-built MangoHud so that
    #    stale installed binaries/libraries don't linger if source files change.
    #    Run before touching the source tree so ninja can still find build.ninja.
    if [[ -d "$MANGOHUD_BUILD_DIR" ]]; then
        gb_step "Removing previous MangoHud install..."
        { ninja -C "$MANGOHUD_BUILD_DIR" uninstall >/dev/null 2>&1; } &
        gb_spin $! "Uninstalling previous MangoHud binaries..."
        wait $! || true  # non-fatal: may fail if already partially removed
        rm -rf "$MANGOHUD_BUILD_DIR"
        gb_ok "Previous MangoHud build cleaned"
    fi

    # 1. Clone or update MangoHud from GitHub
    gb_step "Fetching MangoHud from GitHub..."
    if [[ -d "$MANGOHUD_SRC/.git" ]]; then
        { git -C "$MANGOHUD_SRC" pull origin master >/dev/null 2>&1; } &
        gb_spin $! "Updating repository ($MANGOHUD_SRC)..."
        wait $! || gb_warn_ui "git pull failed — continuing with local version"
        gb_ok "Repository updated"
    else
        rm -rf "$MANGOHUD_SRC"
        { git clone --depth=1 "$MANGOHUD_REPO" "$MANGOHUD_SRC" >/dev/null 2>&1; } &
        gb_spin $! "Cloning $MANGOHUD_REPO..."
        if ! wait $!; then
            die "git clone failed — check GitHub connectivity"
        fi
        gb_ok "Repository cloned to $MANGOHUD_SRC"
    fi

    # 2. Remove system-packaged MangoHud if present
    gb_step "Removing system-packaged MangoHud (if any)..."
    if dpkg -l mangohud 2>/dev/null | grep -q '^ii'; then
        { apt-get remove -y mangohud 2>/dev/null; } &
        gb_spin $! "Uninstalling mangohud (apt)..."
        wait $! || true
        gb_ok "System mangohud removed"
    else
        gb_info "No system-packaged mangohud found (skip)"
    fi

    # 3. Build dependencies
    gb_step "Installing MangoHud build dependencies..."
    {
        apt-get install -y -qq \
            meson ninja-build \
            glslang-tools glslang-dev \
            libvulkan-dev \
            libxnvctrl-dev \
            libwayland-dev \
            libx11-dev libxrandr-dev \
            libdbus-1-dev \
            pkg-config \
            python3 python3-mako \
            cmake 2>/dev/null
    } &
    gb_spin $! "Installing dependencies..."
    wait $! || gb_warn_ui "Some packages failed — continuing anyway"
    gb_ok "Dependencies installed"

    # Detect whether XNVCtrl dev headers were successfully installed.
    # libxnvctrl-dev may not exist in newer distro repos; fall back gracefully.
    local _xnvctrl_flag="-Dwith_xnvctrl=disabled"
    if find /usr/include -name "NVCtrl.h" 2>/dev/null | grep -q .; then
        _xnvctrl_flag="-Dwith_xnvctrl=enabled"
    fi

    # python3-mako is required by meson; apt may silently fail on PEP-668-enforced systems.
    if ! python3 -c "import mako" 2>/dev/null; then
        gb_step "python3-mako not available — installing via pip..."
        { pip3 install --break-system-packages --quiet mako 2>/dev/null \
            || pip3 install --user --quiet mako 2>/dev/null; } &
        gb_spin $! "pip install mako..."
        if ! wait $! || ! python3 -c "import mako" 2>/dev/null; then
            die "python3 mako is required for MangoHud. Install manually: pip3 install mako"
        fi
        gb_ok "mako installed via pip"
    fi

    # 4. Configure build (always fresh: build dir was wiped in step 0)
    gb_step "Configuring MangoHud (meson)..."
    local _meson_log
    _meson_log=$(mktemp /tmp/gb_meson.XXXXX.log)
    { meson setup \
            --prefix="$MANGOHUD_PREFIX" \
            -Dwith_nvml=enabled \
            -Dwith_x11=enabled \
            -Dwith_wayland=enabled \
            "$_xnvctrl_flag" \
            "$MANGOHUD_BUILD_DIR" "$MANGOHUD_SRC" >"$_meson_log" 2>&1; } &
    gb_spin $! "Configuring with meson..."
    if ! wait $!; then
        echo "" >&2
        tail -30 "$_meson_log" >&2
        rm -f "$_meson_log"
        die "meson configuration failed — see output above"
    fi
    rm -f "$_meson_log"
    gb_ok "meson configuration OK"

    # 5. Compile
    gb_step "Compiling MangoHud..."
    local _ninja_log
    _ninja_log=$(mktemp /tmp/gb_ninja.XXXXX.log)
    local _cpus
    _cpus=$(nproc)
    { ninja -C "$MANGOHUD_BUILD_DIR" -j"$_cpus" >"$_ninja_log" 2>&1; } &
    gb_spin $! "Compiling with ninja -j${_cpus}..."
    if ! wait $!; then
        echo "" >&2
        tail -40 "$_ninja_log" >&2
        rm -f "$_ninja_log"
        die "ninja build failed — see output above"
    fi
    rm -f "$_ninja_log"
    gb_ok "Build complete"

    # 6. Install
    gb_step "Installing MangoHud to $MANGOHUD_PREFIX..."
    { ninja -C "$MANGOHUD_BUILD_DIR" install >/dev/null 2>&1; } &
    gb_spin $! "Installing..."
    wait $! || die "MangoHud install failed"
    { ldconfig 2>/dev/null; } &
    gb_spin $! "Updating ldconfig..."
    wait $! || true
    gb_ok "MangoHud installed to $MANGOHUD_PREFIX"

    # 7. Install MangoHud.conf with GreenBoost integration
    gb_step "Installing MangoHud.conf with GreenBoost integration..."
    local _conf_dir="$real_home/.config/MangoHud"

    mkdir -p "$_conf_dir"
    cat > "$_conf_dir/MangoHud.conf" << CONFEOF
# MangoHud config — GreenBoost integration
# Generated by: sudo ./greenboost_setup.sh install-mangohud

# GPU stats (NVML reports GreenBoost virtual pool size when shim is active)
gpu_stats
gpu_load_change
gpu_temp
gpu_power
gpu_core_clock
gpu_mem_clock
vram

# CPU
cpu_stats
cpu_temp

# Framerate
fps
frametime
frame_timing

# Overlay background
background_alpha=0.5

# GreenBoost: Vulkan layer status, T1/T2 pool config, live T2/T3 usage
exec=[ "\${GREENBOOST_VULKAN:-0}" = "1" ] && { P=\$(cat /sys/module/greenboost/parameters/physical_vram_gb 2>/dev/null||echo ?); V=\$(cat /sys/module/greenboost/parameters/virtual_vram_gb 2>/dev/null||echo ?); printf "GB Layer ON  T1=%sGB T2=%sGB" "\$P" "\$V"; } || printf "GB Layer OFF"
exec=awk 'BEGIN{u="?";a="?"} /T2 allocated/{match(\$0,/: *([0-9]+)/,x);u=x[1]} /T2 available/{match(\$0,/: *([0-9]+)/,x);a=x[1]} END{printf "GB T2: %s used / %s MB avail",u,a}' /sys/class/greenboost/greenboost/status 2>/dev/null || printf "GB T2: N/A"
exec=awk '/T3 allocated/{match(\$0,/: *([0-9]+)/,a); printf "GB T3: %s MB NVMe",a[1]; exit}' /sys/class/greenboost/greenboost/status 2>/dev/null
CONFEOF

    chown -R "$real_user:$real_user" "$_conf_dir"
    gb_ok "MangoHud.conf installed to $_conf_dir/"

    # 8. Show Steam launch commands + MangoHud OSD hint
    cmd_steam_launch_info
    _mangohud_steam_hint
}

cmd_uninstall_mangohud() {
    need_root uninstall-mangohud

    local real_user="${SUDO_USER:-$USER}"
    local real_home
    real_home="$(getent passwd "$real_user" | cut -d: -f6)"

    gb_header
    echo -e "  ${C_CYAN}${C_BOLD}Remove MangoHud${C_RESET}"
    echo -e "  ${C_DIM}Removes MangoHud binaries, libraries, and GreenBoost MangoHud.conf.${C_RESET}"
    echo -e ""
    local choice
    read -r -p "  Remove MangoHud now? [Y/n] " choice
    if [[ "$choice" =~ ^[Nn] ]]; then
        gb_info "Skipping."
        return
    fi

    # Uninstall via ninja if the build directory is still present
    if [[ -d "$MANGOHUD_BUILD_DIR" ]]; then
        gb_step "Uninstalling MangoHud (ninja uninstall)..."
        { ninja -C "$MANGOHUD_BUILD_DIR" uninstall >/dev/null 2>&1; } &
        gb_spin $! "Removing installed files..."
        wait $! || gb_warn_ui "ninja uninstall reported errors — continuing"
        gb_ok "MangoHud binaries/libraries removed"
    else
        gb_warn_ui "Build directory $MANGOHUD_BUILD_DIR not found — skipping ninja uninstall"
    fi

    # Remove source tree and build directory
    if [[ -d "$MANGOHUD_SRC" ]]; then
        rm -rf "$MANGOHUD_SRC"
        gb_ok "Removed source tree: $MANGOHUD_SRC"
    fi
    if [[ -d "$MANGOHUD_BUILD_DIR" ]]; then
        rm -rf "$MANGOHUD_BUILD_DIR"
        gb_ok "Removed build directory: $MANGOHUD_BUILD_DIR"
    fi

    # Remove MangoHud.conf installed by GreenBoost
    local _conf="$real_home/.config/MangoHud/MangoHud.conf"
    if [[ -f "$_conf" ]]; then
        rm -f "$_conf"
        gb_ok "Removed MangoHud.conf: $_conf"
    fi

    ldconfig 2>/dev/null || true
    gb_ok "MangoHud uninstalled."
}

cmd_steam_launch_info() {
    echo ""
    echo -e "  ${C_CYAN}${C_BOLD}━━━ GreenBoost Proton — install location ━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
    echo ""
    local _gpw_path=""
    for _d in \
        "$HOME/.local/share/Steam/compatibilitytools.d/greenboost-proton" \
        "$HOME/.steam/root/compatibilitytools.d/greenboost-proton" \
        "$HOME/.local/share/Steam/compatibilitytools.d/greenboost-proton-wayland" \
        "$HOME/.steam/root/compatibilitytools.d/greenboost-proton-wayland"
    do
        [[ -d "$_d" ]] && _gpw_path="$_d" && break
    done
    if [[ -n "$_gpw_path" ]]; then
        echo -e "  ${C_LIME}✓${C_RESET}  Installed at:  ${C_LIME}${C_BOLD}$_gpw_path${C_RESET}"
    else
        echo -e "  ${C_AMBER}⚠${C_RESET}  Not installed — run: ${C_CYAN}greenboost_proton/install.sh${C_RESET}"
    fi
    echo ""
    echo -e "  ${C_CYAN}${C_BOLD}━━━ Required: select the Proton version in Steam ━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
    echo ""
    echo -e "  ${C_VIOLET}◈${C_RESET}  Right-click the game → ${C_BOLD}Properties → Compatibility${C_RESET}"
    echo -e "     Check ${C_BOLD}\"Force the use of a specific Steam Play compatibility tool\"${C_RESET}"
    echo -e "     and select ${C_LIME}${C_BOLD}GreenBoost Proton${C_RESET}"
    echo ""
    echo -e "  ${C_CYAN}${C_BOLD}━━━ Enable per game in Steam launch options ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
    echo ""
    echo -e "  ${C_AMBER}▸ With HDR:${C_RESET}"
    echo -e "  ${C_LIME}PROTON_ENABLE_HDR=1 %command%${C_RESET}"
    echo ""
    echo -e "  ${C_AMBER}▸ With DLSS Super Resolution preset override:${C_RESET}"
    echo ""
    echo -e "    ${C_DIM}${C_GRAY}  Preset       Quality                   Best for${C_RESET}"
    echo -e "    ${C_DIM}${C_GRAY}  ──────────   ────────────────────────  ──────────────────${C_RESET}"
    echo -e "    ${C_DIM}${C_GRAY}  M (Heavier)  Highest quality           RTX 40/50 series${C_RESET}"
    echo -e "    ${C_DIM}${C_GRAY}  L (Balanced) Good quality/perf balance  Any RTX${C_RESET}"
    echo -e "    ${C_DIM}${C_GRAY}  K (Lighter)  Better performance         RTX 20/30 series${C_RESET}"
    echo ""
    echo -e "  ${C_AMBER}  Example — preset M:${C_RESET}"
    echo -e "  ${C_LIME}DXVK_NVAPI_DRS_NGX_DLSS_SR_OVERRIDE_RENDER_PRESET_SELECTION=render_preset_m %command%${C_RESET}"
    echo ""
}

# ---- MangoHud OSD hint -------------------------------------------------
# Shown after install-mangohud and via steam-launch-guide.
_mangohud_steam_hint() {
    echo ""
    echo -e "  ${C_CYAN}${C_BOLD}━━━ Show MangoHud benchmark OSD in Steam ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${C_RESET}"
    echo ""
    echo -e "  ${C_AMBER}▸ Use this command in Steam launch options to show the MangoHud OSD:${C_RESET}"
    echo ""
    echo -e "  ${C_LIME}${C_BOLD}MANGOHUD=1 %command%${C_RESET}"
    echo ""
    echo -e "  ${C_DIM}Combined with HDR:${C_RESET}"
    echo -e "  ${C_LIME}MANGOHUD=1 PROTON_ENABLE_HDR=1 %command%${C_RESET}"
    echo ""
    echo -e "  ${C_DIM}Right-click the game → Properties → General → Launch Options${C_RESET}"
    echo ""
}

# ---- install-deps ------------------------------------------------------
# Install all Ubuntu packages needed for GreenBoost v2.8.2 + ExLlamaV3

cmd_install_build_deps() {
    need_root install-deps

    printf "  ${C_CYAN}❯${C_RESET}  ${C_DIM}Updating package lists...${C_RESET}"
    apt-get update -qq 2>/dev/null
    printf "\r%*s\r" "$(tput cols 2>/dev/null || echo 80)" ""

    # Minimal packages required to build and DKMS-register the kernel module.
    # No AI libraries, no Python, no CUDA — those go in cmd_install_optional_pkgs.
    local _pkgs=(
        build-essential gcc gcc-multilib make git curl wget
        "linux-headers-$(uname -r)"
        pkg-config sysfsutils liburing-dev
        kmod dkms
    )
    # CPU vendor-specific microcode (safe to install on any machine)
    local _cpu_vendor
    _cpu_vendor=$(grep -m1 "vendor_id" /proc/cpuinfo | awk '{print $3}')
    if [[ "$_cpu_vendor" == "GenuineIntel" ]]; then
        _pkgs+=(intel-microcode)
    elif [[ "$_cpu_vendor" == "AuthenticAMD" ]]; then
        _pkgs+=(amd64-microcode)
    fi

    # Install with live APT progress bar via APT::Status-Fd
    local _fifo
    _fifo=$(mktemp -u /tmp/gb_apt_XXXXXX)
    mkfifo "$_fifo"

    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        -o "APT::Status-Fd=3" \
        "${_pkgs[@]}" 2>/dev/null 3>"$_fifo" &
    local _apt_pid=$!

    local _pct=0 _pkg=""
    while IFS= read -r _line; do
        if [[ "$_line" == pmstatus:* ]]; then
            IFS=: read -r _ _pkg _pct _msg <<< "$_line"
            _pct=${_pct%%.*}
            local _filled=$(( _pct * 40 / 100 )) _empty=$(( 40 - _pct * 40 / 100 ))
            printf "\r  ${C_GRAY}[%3d%%]${C_RESET} ${C_LIME}%s${C_GRAY}%s${C_RESET}  ${C_DIM}%-28s${C_RESET}" \
                "$_pct" \
                "$(printf '█%.0s' $(seq 1 "$_filled" 2>/dev/null || true))" \
                "$(printf '░%.0s' $(seq 1 "$_empty"  2>/dev/null || true))" \
                "$_pkg"
        fi
    done < "$_fifo"

    wait "$_apt_pid" || gb_warn_ui "Some packages failed — check: apt-get install ${_pkgs[*]}"
    rm -f "$_fifo"
    printf "\r%*s\r" "$(tput cols 2>/dev/null || echo 80)" ""

    # Ensure cpuid module loads at boot
    if ! grep -q cpuid /etc/modules-load.d/*.conf 2>/dev/null; then
        echo cpuid > /etc/modules-load.d/ai-workstation.conf
    fi

    gb_ok "Build dependencies installed"
}

cmd_install_optional_pkgs() {
    need_root install-optional-pkgs

    printf "  ${C_CYAN}❯${C_RESET}  ${C_DIM}Updating package lists...${C_RESET}"
    apt-get update -qq 2>/dev/null
    printf "\r%*s\r" "$(tput cols 2>/dev/null || echo 80)" ""

    # Optional AI/compute libraries and tools. Not required to build the kernel
    # module — only install these for a full AI workstation setup.
    local _pkgs=(
        python3 python3-pip python3-dev python3-venv
        libopenblas-dev libblas-dev liblapack-dev
        libhwloc-dev hwloc libnuma-dev libomp-dev
        ocl-icd-opencl-dev
        nvidia-cuda-toolkit
    )
    local _os_id
    _os_id=$(grep -oP '^ID=\K.*' /etc/os-release 2>/dev/null | tr -d '"')
    if [[ "$_os_id" == "debian" ]]; then
        _pkgs+=(linux-cpupower linux-perf nvtop)
    else
        _pkgs+=(cpufrequtils linux-tools-generic nvtop "linux-tools-$(uname -r)")
    fi

    local _fifo
    _fifo=$(mktemp -u /tmp/gb_apt_XXXXXX)
    mkfifo "$_fifo"

    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        -o "APT::Status-Fd=3" \
        "${_pkgs[@]}" 2>/dev/null 3>"$_fifo" &
    local _apt_pid=$!

    local _pct=0 _pkg=""
    while IFS= read -r _line; do
        if [[ "$_line" == pmstatus:* ]]; then
            IFS=: read -r _ _pkg _pct _msg <<< "$_line"
            _pct=${_pct%%.*}
            local _filled=$(( _pct * 40 / 100 )) _empty=$(( 40 - _pct * 40 / 100 ))
            printf "\r  ${C_GRAY}[%3d%%]${C_RESET} ${C_LIME}%s${C_GRAY}%s${C_RESET}  ${C_DIM}%-28s${C_RESET}" \
                "$_pct" \
                "$(printf '█%.0s' $(seq 1 "$_filled" 2>/dev/null || true))" \
                "$(printf '░%.0s' $(seq 1 "$_empty"  2>/dev/null || true))" \
                "$_pkg"
        fi
    done < "$_fifo"

    wait "$_apt_pid" || gb_warn_ui "Some packages failed — check: apt-get install ${_pkgs[*]}"
    rm -f "$_fifo"
    printf "\r%*s\r" "$(tput cols 2>/dev/null || echo 80)" ""

    gb_ok "Optional AI/compute libraries installed"
    gb_info "Note: NVIDIA driver 580+ and CUDA 13 must be installed separately"
}

# cmd_install_deps — installs all dependencies (build + optional).
# Called when running 'install-deps' directly; full-install uses
# cmd_install_build_deps (always) + cmd_install_optional_pkgs (gated).
cmd_install_deps() {
    cmd_install_build_deps
    cmd_install_optional_pkgs
}

# ---- full-install ------------------------------------------------------
# Complete fresh-OS install — run this after a clean Ubuntu install.
# Covers: OS deps, kernel module, CUDA shim, all system configs,
# sysctl tuning, GRUB params, and optional ExLlamaV3 with GreenBoost patches.
# NVMe swap (T3 tier) is intentionally NOT touched — configure manually before running.

cmd_full_install() {
    need_root full-install
    GB_STOPPED_SERVICES=""

    # ── Mode selection — cmd_full_install always targets full system setup.
    # Only --module-only overrides this (for scripts / CI / direct invocation).
    GB_INSTALL_MODE="full"
    for _a in "$@"; do
        [[ "$_a" == "--module-only" ]] && { GB_INSTALL_MODE="module"; break; }
    done

    # Build the mode flag to forward to delegated distro scripts
    local _mode_flag="--module-only"
    [[ "$GB_INSTALL_MODE" == "full" ]] && _mode_flag="--full-install"

    # ── Distro detection: delegate to Rocky/RHEL script on Red Hat-based systems ──
    if [[ -f /etc/redhat-release ]] || \
       grep -qiE '^ID_LIKE=.*rhel|^ID_LIKE=.*fedora|^ID=.*rhel|^ID=.*rocky|^ID=.*almalinux|^ID=.*centos|^ID=.*fedora' \
           /etc/os-release 2>/dev/null; then
        local rocky_script="$MODULE_DIR/greenboost_setup_rocky.sh"
        [[ -x "$rocky_script" ]] || die "Red Hat-based system detected but $rocky_script not found or not executable."
        info "Red Hat-based system detected — delegating to greenboost_setup_rocky.sh"
        exec "$rocky_script" full-install "$_mode_flag" "$@"
    fi

    # ── Distro detection: delegate to Arch script on Arch-based systems ──────
    if [[ -f /etc/arch-release ]] || \
       grep -qiE '^ID=arch$|^ID=manjaro|^ID=endeavouros|^ID=cachyos|^ID_LIKE=.*arch' \
           /etc/os-release 2>/dev/null; then
        local arch_script="$MODULE_DIR/greenboost_setup_arch.sh"
        [[ -x "$arch_script" ]] || die "Arch-based system detected but $arch_script not found or not executable."
        info "Arch-based system detected — delegating to greenboost_setup_arch.sh"
        exec "$arch_script" full-install "$_mode_flag" "$@"
    fi

    # ── Hardware preset: --owner-workstation only for full mode ─────────
    local _owner_ws=0
    for arg in "$@"; do
        [[ "$arg" == "--owner-workstation" ]] && _owner_ws=1
    done

    if [[ $_owner_ws -eq 1 && "$GB_INSTALL_MODE" == "full" ]]; then
        set_owner_workstation_params
        gb_header
    else
        detect_hardware
        gb_header
        print_detected_hardware
    fi
    echo ""

    # ── SHARED PATH: kernel module (runs for both module-only and full) ──────

    # 0 — Purge any previous GreenBoost install to guarantee a clean slate
    gb_step 0 5 "Purging previous GreenBoost installation (if any)..."
    do_purge 0
    gb_ok "Previous installation purged"

    # 1 — Build dependencies (minimal — just what's needed to compile the module)
    gb_step 1 5 "Installing build dependencies..."
    cmd_install_build_deps
    gb_ok "Build dependencies installed"

    # 2 — Build + install kernel module + CUDA shim
    gb_step 2 5 "Building and installing kernel module + CUDA shim..."
    GB_SKIP_INSTALL_PURGE=1 cmd_install
    gb_ok "Kernel module + CUDA shim installed"

    # 3 — Load kernel module
    gb_step 3 5 "Loading kernel module..."
    cmd_load
    gb_ok "Kernel module loaded"

    if [[ "$GB_INSTALL_MODE" == "module" ]]; then
        echo ""
        gb_separator
        echo ""
        gb_ok "Kernel module installed and loaded."
        gb_info "Run 'sudo ./greenboost_setup.sh status' to verify."
        gb_info "For full system tuning, re-run and choose option [2]."
        echo ""
        gb_separator
        echo ""
        return 0
    fi

    # ── FULL ONLY PATH: all system changes applied automatically (user consented by choosing full mode) ───

    # 4 — System configs: Ollama, udev rules, CPU governor service, sysctl, llama.cpp
    gb_step 4 5 "Installing system configuration files..."
    gb_info "Applying: Ollama/inference service config (drop-ins, TurboQuant, udev, NVMe, cpu-perf)"
    cmd_install_sys_configs
    cmd_install_llama_configs
    gb_ok "System configuration installed"

    # 4b — Optional AI/compute libraries (CUDA toolkit, OpenBLAS, Python, nvtop, etc.)
    gb_info "Applying: optional packages (cuda-toolkit, openblas, python3-pip, nvtop, cpufrequtils, 32-bit compat)"
    cmd_install_optional_pkgs
    gb_ok "Optional AI/compute libraries installed"


    # 5 — System tuning (sysctl + NVMe + CPU governor + THP)
    gb_step 5 5 "Applying system tuning..."
    gb_info "Applying: sysctl + NVMe/THP/CPU governor tuning"
    cmd_tune_sysctl
    cmd_tune
    gb_ok "sysctl and runtime tuning applied"

    # 5b — GRUB boot parameters (requires reboot)
    gb_info "Applying: GRUB boot parameters (transparent_hugepage, rcu_nocbs, nohz_full)"
    cmd_tune_grub
    gb_ok "GRUB updated"

    # Always regenerate profile on full-install to ensure it reflects current hardware.
    gb_info "Regenerating hardware profile..."
    cmd_profile_create

    # Restart only services that were stopped during purge
    for svc in $GB_STOPPED_SERVICES; do
        info "Restarting $svc (was running before install)..."
        systemctl restart "$svc" 2>/dev/null \
            && info "$svc restarted." \
            || warn "$svc restart failed — run: sudo systemctl restart $svc"
    done

    echo ""
    gb_separator
    echo -e ""
    gb_ok "${C_BOLD}Full install complete!"
    echo -e ""
    echo -e "  ${C_AMBER}${C_BOLD}⚠  REBOOT REQUIRED${C_RESET} ${C_GRAY}to activate GRUB params + hugepage pre-allocation${C_RESET}"
    echo -e ""
    gb_info "GreenBoost Proton and MangoHud are gaming tools — not part of full install."
    gb_info "Install them separately from the interactive menu (options [6]–[9])."
    echo ""
    gb_separator
    echo ""
}


# ---- Entry point -------------------------------------------------------

# Strip global flags before command dispatch.
# --skip-update-check  Bypass the GitLab version check (use on offline workstations).
# --profile <file>     Load module parameters from a profile file.
GB_ORIG_ARGS=("$@")   # preserved for exec-restart after git pull
GB_SKIP_UPDATE=0
GB_PROFILE_FILE=""    # path to user-supplied profile file (empty = use active profile)
_ARGS=()
_expect_profile=0
for _arg in "$@"; do
    case "$_arg" in
        --skip-update-check) GB_SKIP_UPDATE=1 ;;
        --profile)           _expect_profile=1 ;;
        *)
            if [[ "${_expect_profile}" -eq 1 ]]; then
                GB_PROFILE_FILE="$_arg"
                _expect_profile=0
            else
                _ARGS+=("$_arg")
            fi
            ;;
    esac
done
set -- "${_ARGS[@]}"
unset _ARGS _arg _expect_profile

COMMAND="${1:-}"

# ---------------------------------------------------------------------------
# cmd_recover — manual crash-recovery entrypoint.
# Calls /usr/local/sbin/greenboost-recover directly with output on the
# terminal.  Equivalent to what greenboost-recovery.service runs on boot,
# but usable after a hard reset without rebooting.
# ---------------------------------------------------------------------------
cmd_recover() {
    need_root recover
    local script="/usr/local/sbin/greenboost-recover"
    if [[ ! -x "$script" ]]; then
        gb_warn_ui "Recovery script not installed. Running: sudo $0 install-sys-configs"
        cmd_install_recovery
    fi
    exec "$script"
}

# Check for a newer release before any install operation.
# Skipped for non-modifying commands (status, unload, help, build).
# Suppressed when --skip-update-check is passed (offline workstations).
case "$COMMAND" in
    install|setup|full-install|install-sys-configs|install-llama-configs|install-mangohud|recover|fix-steam)
        if [[ $GB_SKIP_UPDATE -eq 0 ]]; then check_update; fi
        echo "" ;;
esac

case "$COMMAND" in
    # Primary commands
    install)             cmd_install            ;;
    uninstall)           cmd_uninstall          ;;
    build)               cmd_build              ;;
    load)                cmd_load               ;;
    unload)              cmd_unload             ;;
    setup|full-install)  cmd_full_install "--full-install" "$@"  ;;
    module-only)         GB_INSTALL_MODE="module" cmd_full_install "--module-only" "$@" ;;
    tune)                cmd_tune               ;;
    tune-grub)           cmd_tune_grub          ;;
    tune-sysctl)         cmd_tune_sysctl        ;;
    tune-libs)           cmd_tune_libs          ;;
    tune-all)            cmd_tune_all           ;;
    status)              cmd_status             ;;
    vulkan)              cmd_vulkan             ;;
    benchmark)           cmd_benchmark "${@:2}" ;;
    profile)             cmd_profile "${@:2}"   ;;
    gaming-mode)         cmd_gaming_mode "${@:2}" ;;
    t3-memory)           cmd_t3_memory "${2:-}" ;;
    clean)
        case "${2:-}" in
            memory)  cmd_clean_memory ;;
            *) die "Usage: greenboost clean memory" ;;
        esac
        ;;
    clean-memory)        cmd_clean_memory       ;;
    logs)                cmd_logs               "${@:2}" ;;
    proton-logs)         cmd_proton_logs        "${@:2}" ;;
    inference-logs)      cmd_inference_logs     "${@:2}" ;;
    clear)
        case "${2:-}" in
            logs)            cmd_clear_logs ;;
            proton-logs)     cmd_clear_proton_logs ;;
            inference-logs)  cmd_clear_inference_logs ;;
            *) die "Usage: greenboost clear logs|proton-logs|inference-logs" ;;
        esac
        ;;
    clean-logs)          cmd_clean_logs         ;;
    turboquant)          cmd_turboquant "${@:2}" ;;
    show-commands)       cmd_show_commands      ;;
    help|--help|-h)      cmd_help "${@:2}"      ;;
    # Advanced (kept for compat)
    recover)                cmd_recover               ;;
    install-sys-configs)    cmd_install_sys_configs   ;;
    install-llama-configs)  cmd_install_llama_configs ;;
    install-vulkan-layer)   cmd_install_vulkan_layer  ;;
    steam-launch-guide)     cmd_steam_launch_info; _mangohud_steam_hint ;;
    fix-steam)              cmd_fix_steam             ;;
    proton-clean)           cmd_proton_clean          ;;
    install-proton) cmd_install_proton ;;
    remove-proton)          cmd_uninstall_proton      ;;
    install-mangohud)       cmd_install_mangohud      ;;
    uninstall-mangohud)     cmd_uninstall_mangohud    ;;
    inference-test)         cmd_inference_test        "${@:2}" ;;
    # Default: interactive wizard
    "")
        cmd_wizard ;;
    *) die "Unknown command: '$COMMAND'  — run: $0 help" ;;
esac

#!/usr/bin/env bash
# install.sh — GreenBoost Proton Wayland installer
#
# Usage:
#   ./install.sh           Install to Steam compatibilitytools.d
#   ./install.sh --uninstall   Remove installation
#   ./install.sh --help
#
# All hardware values auto-detected at runtime — nothing hard-coded.

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Brand colors (truecolor) ──────────────────────────────────────────────────
C_RESET=$'\033[0m'
C_VIOLET=$'\033[38;2;108;113;196m'
C_LIME=$'\033[38;2;230;255;60m'
C_CYAN=$'\033[38;2;48;200;255m'
C_AMBER=$'\033[38;2;255;191;0m'
C_GRAY=$'\033[38;2;208;207;204m'
C_RED=$'\033[38;2;220;50;47m'
C_DIM=$'\033[2m'
C_BOLD=$'\033[1m'

gb_ok()   { printf '%b\n' "  ${C_LIME}✓${C_RESET}  $*"; }
gb_fail() { printf '%b\n' "  ${C_RED}✗${C_RESET}  $*"; }
gb_warn() { printf '%b\n' "  ${C_AMBER}⚠${C_RESET}  $*"; }
gb_info() { printf '%b\n' "  ${C_VIOLET}◈${C_RESET}  $*"; }
gb_step() { printf '%b\n' "  ${C_CYAN}❯${C_RESET}  $*"; }

gb_header() {
    printf '\n%b\n' "${C_VIOLET}${C_BOLD}  GreenBoost Proton — Installer${C_RESET}"
    printf '%b\n\n' "${C_DIM}  Thin wrapper around Proton Experimental — virtual-VRAM, Wayland, DXR, NVAPI, shader-cache${C_RESET}"
}

# Canonical install directory — shared Steam compat tools folder
INSTALL_DIR="$HOME/.local/share/Steam/compatibilitytools.d/greenboost-proton"

# ── Argument parsing ──────────────────────────────────────────────────────────
UNINSTALL=0
for _arg in "$@"; do
    case "$_arg" in
        --uninstall) UNINSTALL=1 ;;
        --help|-h)
            gb_header
            printf '%b\n' "  ${C_GRAY}Usage:${C_RESET}"
            printf '%b\n' "    ${C_CYAN}./install.sh${C_RESET}             ${C_DIM}Install to Steam compat tools${C_RESET}"
            printf '%b\n' "    ${C_CYAN}./install.sh --uninstall${C_RESET}  ${C_DIM}Remove installation${C_RESET}"
            printf '%b\n' ""
            printf '%b\n' "  ${C_GRAY}Install path:${C_RESET}"
            printf '%b\n' "    ${C_DIM}$INSTALL_DIR${C_RESET}"
            echo ""
            exit 0
            ;;
    esac
done

gb_header

# ── Hardware summary (informational — never stored, always re-detected) ───────
gb_step "${C_GRAY}Detecting hardware...${C_RESET}"

GPU_NAME=""; GPU_VRAM_MB=0; GPU_IS_NVIDIA=0; GPU_SUPPORTS_RT=0
if command -v nvidia-smi &>/dev/null; then
    _info=$(timeout 3 nvidia-smi \
        --query-gpu=name,memory.total \
        --format=csv,noheader 2>/dev/null | head -1) || true
    if [[ -n "$_info" ]]; then
        GPU_NAME=$(printf '%s' "$_info" | cut -d, -f1 \
            | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        _v=$(printf '%s' "$_info" | cut -d, -f2 | grep -oP '[0-9]+')
        [[ -n "$_v" ]] && GPU_VRAM_MB="$_v"
        GPU_IS_NVIDIA=1
    fi
fi
if [[ -z "$GPU_NAME" ]] && command -v lspci &>/dev/null; then
    GPU_NAME=$(lspci 2>/dev/null \
        | grep -iE 'VGA compatible|3D controller|Display controller' \
        | head -1 | sed 's/.*: //;s/ (.*)//')
    printf '%s' "${GPU_NAME,,}" | grep -q 'nvidia' && GPU_IS_NVIDIA=1
fi
if (( GPU_IS_NVIDIA )); then
    printf '%s' "${GPU_NAME,,}" \
        | grep -qiE 'rtx|a[0-9]{3,4}[^d]|h[0-9]{2,3}|l[0-9]{2,3}|b[0-9]{3}|blackwell|lovelace|ampere|turing' \
        && GPU_SUPPORTS_RT=1 || true
fi

gb_info "${C_GRAY}GPU:${C_RESET} ${C_CYAN}${GPU_NAME:-unknown}${C_RESET}${C_DIM}  ${GPU_VRAM_MB} MiB  NVIDIA:${GPU_IS_NVIDIA}  RT:${GPU_SUPPORTS_RT}${C_RESET}"

# GreenBoost virtual VRAM check
if [[ -c /dev/greenboost ]]; then
    gb_ok "GreenBoost kernel module: ${C_LIME}active${C_RESET}"
else
    gb_warn "GreenBoost kernel module not loaded — T2 DMA-BUF fallback unavailable"
    gb_info "  Run: ${C_CYAN}sudo greenboost_setup.sh load${C_RESET}"
fi
echo ""

# ── Uninstall ─────────────────────────────────────────────────────────────────
if (( UNINSTALL )); then
    gb_step "${C_GRAY}Removing GreenBoost Proton installation...${C_RESET}"
    _removed=0
    for _d in \
        "$INSTALL_DIR" \
        "$HOME/.steam/root/compatibilitytools.d/greenboost-proton" \
        "$HOME/.steam/root/compatibilitytools.d/greenboost-proton-wayland" \
        "$HOME/.local/share/Steam/compatibilitytools.d/greenboost-proton-wayland"
    do
        if [[ -L "$_d" || -d "$_d" ]]; then
            rm -rf "$_d"
            gb_ok "Removed: ${C_DIM}$_d${C_RESET}"
            _removed=1
        fi
    done
    (( _removed )) || gb_info "Nothing to remove."
    gb_ok "Uninstalled. Restart Steam to apply."
    exit 0
fi

# ── Verify dependencies ────────────────────────────────────────────────────────
gb_step "${C_GRAY}Checking dependencies...${C_RESET}"

_proton_exp_found=0
for _pd in \
    "$HOME/.local/share/Steam/steamapps/common/Proton - Experimental" \
    "$HOME/.steam/root/steamapps/common/Proton - Experimental"
do
    [[ -f "$_pd/proton" ]] && _proton_exp_found=1 && break
done
if (( _proton_exp_found )); then
    gb_ok "Proton Experimental: installed"
else
    gb_warn "Proton Experimental not found — GreenBoost Proton requires it"
    gb_info "  In Steam: Library → Tools → Proton Experimental → Install"
fi

_sniper_found=0
for _sd in \
    "$HOME/.steam/root/steamapps/common/SteamLinuxRuntime_sniper" \
    "$HOME/.local/share/Steam/steamapps/common/SteamLinuxRuntime_sniper"
do
    [[ -d "$_sd" ]] && _sniper_found=1 && break
done
if (( _sniper_found )); then
    gb_ok "SteamLinuxRuntime Sniper: installed"
else
    gb_warn "SteamLinuxRuntime Sniper not found — install via Steam (AppID 1628350)"
    gb_info "  In Steam: Library → search 'Steam Linux Runtime 3.0 (sniper)' → Install"
fi

# ── Remove legacy symlinks / old installations ────────────────────────────────
for _legacy in \
    "$HOME/.steam/root/compatibilitytools.d/greenboost-proton-wayland" \
    "$HOME/.steam/root/compatibilitytools.d/greenboost-proton" \
    "$HOME/.local/share/Steam/compatibilitytools.d/greenboost-proton-wayland"
do
    if [[ -L "$_legacy" || -d "$_legacy" ]]; then
        rm -rf "$_legacy"
        gb_info "Removed old entry: ${C_DIM}$_legacy${C_RESET}"
    fi
done

# ── Copy tool to shared Steam compat tools directory ─────────────────────────
echo ""
gb_step "${C_GRAY}Installing to shared Steam compat tools folder...${C_RESET}"

mkdir -p "$INSTALL_DIR"

# Only the wrapper script and Steam VDF manifests are installed.
_steam_items=(
    proton
    compatibilitytool.vdf
    toolmanifest.vdf
    version
)

if command -v rsync &>/dev/null; then
    for _item in "${_steam_items[@]}"; do
        [[ -e "$SELF_DIR/$_item" ]] || { gb_warn "Missing: $SELF_DIR/$_item — skipping"; continue; }
        rsync -a --delete "$SELF_DIR/$_item" "$INSTALL_DIR/" 2>/dev/null \
            || rsync -a "$SELF_DIR/$_item" "$INSTALL_DIR/"
    done
else
    for _item in "${_steam_items[@]}"; do
        [[ -e "$SELF_DIR/$_item" ]] || { gb_warn "Missing: $SELF_DIR/$_item — skipping"; continue; }
        rm -rf "${INSTALL_DIR:?}/$_item"
        cp -r "$SELF_DIR/$_item" "$INSTALL_DIR/"
    done
fi

chmod +x "$INSTALL_DIR/proton"
gb_ok "proton script: executable"

# ── Report installation path ──────────────────────────────────────────────────
echo ""
printf '%b\n' "${C_VIOLET}${C_BOLD}  Installation complete${C_RESET}"
echo ""
gb_ok "Installed at:  ${C_LIME}$INSTALL_DIR${C_RESET}"
echo ""
gb_info "GreenBoost Proton wraps ${C_CYAN}Proton Experimental${C_RESET} — as Valve updates it, you get the update automatically."
echo ""
gb_info "Next steps:"
gb_info "  1. Restart Steam"
gb_info "  2. Right-click a game → Properties → Compatibility"
gb_info "  3. Select ${C_LIME}GreenBoost Proton${C_RESET}"
gb_info ""
gb_info "To update GreenBoost itself, re-run this script."
gb_info "Optional launch options:"
printf '%b\n' "         ${C_CYAN}PROTON_ENABLE_WAYLAND=0 %command%${C_RESET}  ${C_DIM}(fall back to XWayland)${C_RESET}"
printf '%b\n' "         ${C_CYAN}GREENBOOST_NO_DXR=1 %command%${C_RESET}    ${C_DIM}(disable DXR injection — for games that crash with dxr/dxr11)${C_RESET}"
echo ""
gb_info "Monitor: ${C_CYAN}greenboost vulkan${C_RESET}"
echo ""

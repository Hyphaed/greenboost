#!/usr/bin/env bash
# install.sh — GreenBoost Proton installer
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
    printf '%b\n\n' "${C_DIM}  Two Steam entries: wraps latest stable Proton and Proton Experimental — virtual-VRAM, Wayland, DXR, NVAPI${C_RESET}"
}

# Canonical install directories — shared Steam compat tools folder
COMPAT_DIR="$HOME/.local/share/Steam/compatibilitytools.d"
INSTALL_DIR="$COMPAT_DIR/greenboost-proton"
INSTALL_DIR_EXP="$COMPAT_DIR/greenboost-proton-experimental"

# ── Argument parsing ──────────────────────────────────────────────────────────
UNINSTALL=0
FORCE=0
for _arg in "$@"; do
    case "$_arg" in
        --uninstall) UNINSTALL=1 ;;
        --force)     FORCE=1 ;;
        --help|-h)
            gb_header
            printf '%b\n' "  ${C_GRAY}Usage:${C_RESET}"
            printf '%b\n' "    ${C_CYAN}./install.sh${C_RESET}             ${C_DIM}Install to Steam compat tools${C_RESET}"
            printf '%b\n' "    ${C_CYAN}./install.sh --uninstall${C_RESET}  ${C_DIM}Remove installation${C_RESET}"
            printf '%b\n' "    ${C_CYAN}./install.sh --force${C_RESET}      ${C_DIM}Skip hard-fail checks (e.g. Proton Experimental missing)${C_RESET}"
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
        "$INSTALL_DIR_EXP" \
        "$HOME/.steam/root/compatibilitytools.d/greenboost-proton" \
        "$HOME/.steam/root/compatibilitytools.d/greenboost-proton-experimental" \
        "$HOME/.steam/root/compatibilitytools.d/greenboost-proton-wayland" \
        "$HOME/.local/share/Steam/compatibilitytools.d/greenboost-proton-wayland" \
        "$HOME/.local/share/Steam/compatibilitytools.d/proton-ryzen5-5070" \
        "$HOME/.steam/root/compatibilitytools.d/proton-ryzen5-5070"
    do
        if [[ -L "$_d" || -d "$_d" ]]; then
            rm -rf "$_d"
            gb_ok "Removed: ${C_DIM}$_d${C_RESET}"
            _removed=1
        fi
    done
    (( _removed )) || gb_info "Nothing to remove."

    # Remove user-local Vulkan layer files
    for _vkf in \
        "$HOME/.local/share/vulkan/libVkLayer_greenboost.so" \
        "$HOME/.local/share/vulkan/implicit_layer.d/VkLayer_greenboost.json"
    do
        if [[ -f "$_vkf" ]]; then
            rm -f "$_vkf"
            gb_ok "Removed: ${C_DIM}$_vkf${C_RESET}"
        fi
    done

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
    gb_warn "Proton Experimental not found — GreenBoost Proton Experimental will warn at launch"
    gb_info "  In Steam: Library → Tools → Proton Experimental → Install"
fi

_proton_stable_found=0
_proton_stable_name=""
for _base in \
    "$HOME/.local/share/Steam/steamapps/common" \
    "$HOME/.steam/root/steamapps/common"
do
    [[ -d "$_base" ]] || continue
    _best_ver="-1.-1"
    for _pd in "$_base"/Proton\ [0-9]*; do
        [[ -f "$_pd/proton" ]] || continue
        _vstr="$(basename "$_pd" | grep -oP '\d+\.\d+')"
        [[ -z "$_vstr" ]] && continue
        _vmaj="${_vstr%%.*}"; _vmin="${_vstr##*.}"
        _best_maj="${_best_ver%%.*}"; _best_min="${_best_ver##*.}"
        if (( _vmaj > _best_maj || (_vmaj == _best_maj && _vmin > _best_min) )); then
            _best_ver="$_vmaj.$_vmin"
            _proton_stable_name="$(basename "$_pd")"
            _proton_stable_found=1
        fi
    done
    (( _proton_stable_found )) && break
done
if (( _proton_stable_found )); then
    gb_ok "Proton stable: ${_proton_stable_name}"
else
    gb_warn "No stable Proton version found — GreenBoost Proton will warn at launch"
    gb_info "  In Steam: Library → Games → search 'Proton 10.0' (or newer) → Install"
fi

if (( ! _proton_exp_found && ! _proton_stable_found )); then
    if (( FORCE )); then
        gb_warn "Neither Proton Experimental nor stable Proton found — continuing due to --force"
    else
        gb_fail "No Proton version found — cannot install a useful GreenBoost Proton entry"
        gb_info "  Install at least one Proton version in Steam, then re-run."
        gb_info "  Or pass ${C_CYAN}--force${C_RESET} to install anyway."
        exit 1
    fi
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
    "$HOME/.steam/root/compatibilitytools.d/greenboost-proton-experimental" \
    "$HOME/.local/share/Steam/compatibilitytools.d/greenboost-proton-wayland" \
    "$HOME/.local/share/Steam/compatibilitytools.d/proton-ryzen5-5070" \
    "$HOME/.steam/root/compatibilitytools.d/proton-ryzen5-5070"
do
    if [[ -L "$_legacy" || -d "$_legacy" ]]; then
        rm -rf "$_legacy"
        gb_info "Removed old entry: ${C_DIM}$_legacy${C_RESET}"
    fi
done

# Remove any other stale GreenBoost Proton copies (any dir whose
# compatibilitytool.vdf claims our display_name but isn't the canonical dir).
for _base in \
    "$HOME/.local/share/Steam/compatibilitytools.d" \
    "$HOME/.steam/root/compatibilitytools.d"
do
    [[ -d "$_base" ]] || continue
    for _vdf in "$_base"/*/compatibilitytool.vdf; do
        [[ -f "$_vdf" ]] || continue
        _stale_dir="$(dirname "$_vdf")"
        case "$(basename "$_stale_dir")" in
            greenboost-proton|greenboost-proton-experimental) continue ;;
        esac
        if grep -qE '"display_name"[[:space:]]+"GreenBoost Proton( Experimental)?"' "$_vdf" 2>/dev/null; then
            rm -rf "$_stale_dir"
            gb_info "Removed stale GreenBoost copy: ${C_DIM}$_stale_dir${C_RESET}"
        fi
    done
done

# ── Helper: install one Steam compatibility tool entry ────────────────────────
# Args: $1=install_dir  $2=display_name  $3=vdf_tool_key  $4=channel (stable|experimental)
_install_entry() {
    local _dir="$1" _name="$2" _key="$3" _chan="$4"
    gb_step "${C_GRAY}Installing '${_name}'...${C_RESET}  ${C_DIM}$_dir${C_RESET}"
    mkdir -p "$_dir"

    # Copy shared runtime files (proton script, toolmanifest.vdf, version)
    local _items=(proton toolmanifest.vdf version)
    if command -v rsync &>/dev/null; then
        for _item in "${_items[@]}"; do
            [[ -e "$SELF_DIR/$_item" ]] || { gb_warn "Missing: $SELF_DIR/$_item — skipping"; continue; }
            rsync -a --delete "$SELF_DIR/$_item" "$_dir/" 2>/dev/null \
                || rsync -a "$SELF_DIR/$_item" "$_dir/"
        done
    else
        for _item in "${_items[@]}"; do
            [[ -e "$SELF_DIR/$_item" ]] || { gb_warn "Missing: $SELF_DIR/$_item — skipping"; continue; }
            rm -rf "${_dir:?}/$_item"
            cp -r "$SELF_DIR/$_item" "$_dir/"
        done
    fi
    chmod +x "$_dir/proton"

    # Channel sidecar — tells the proton wrapper which upstream to use
    printf '%s\n' "$_chan" > "$_dir/channel"

    # Generate compatibilitytool.vdf for this entry
    cat > "$_dir/compatibilitytool.vdf" << VDFEOF
"compatibilitytools"
{
  "compat_tools"
  {
    "${_key}"
    {
      "install_path" "."
      "display_name" "${_name}"
      "from_oslist"  "windows"
      "to_oslist"    "linux"
    }
  }
}
VDFEOF
    gb_ok "'${_name}': installed"
}

# ── Install both Steam compatibility tool entries ─────────────────────────────
echo ""
gb_step "${C_GRAY}Installing Steam compatibility tool entries...${C_RESET}"

_install_entry \
    "$INSTALL_DIR" \
    "GreenBoost Proton" \
    "greenboost-proton" \
    "stable"

_install_entry \
    "$INSTALL_DIR_EXP" \
    "GreenBoost Proton Experimental" \
    "greenboost-proton-experimental" \
    "experimental"

# ── User-local Vulkan layer (accessible inside pressure-vessel) ───────────────
# The system-wide layer at /etc/vulkan/implicit_layer.d/ is not bindmounted into
# SteamLinuxRuntime Sniper's pressure-vessel container.  Install a user-local copy
# under ~/.local/share/vulkan/ which pressure-vessel does see (home is bindmounted),
# with library_path updated to the user-local .so location.
echo ""
gb_step "${C_GRAY}Installing user-local Vulkan layer (pressure-vessel accessible)...${C_RESET}"

_vk_src_lib="/usr/local/lib/libVkLayer_greenboost.so"
_vk_user_dir="$HOME/.local/share/vulkan"
_vk_user_lib="$_vk_user_dir/libVkLayer_greenboost.so"
_vk_user_manifest_dir="$_vk_user_dir/implicit_layer.d"
_vk_user_manifest="$_vk_user_manifest_dir/VkLayer_greenboost.json"

if [[ -f "$_vk_src_lib" ]]; then
    mkdir -p "$_vk_user_manifest_dir"
    mkdir -p "$HOME/.local/share/greenboost/proton-logs"
    cp "$_vk_src_lib" "$_vk_user_lib"
    cat > "$_vk_user_manifest" << VKEOF
{
    "file_format_version": "1.0.0",
    "layer": {
        "name": "VK_LAYER_GREENBOOST_memory",
        "type": "GLOBAL",
        "library_path": "${_vk_user_lib}",
        "api_version": "1.3.0",
        "implementation_version": "1",
        "description": "GreenBoost virtual VRAM — inflates device-local heap and routes overflow to T2/T3 DDR via DMA-BUF",
        "enable_environment": {
            "GREENBOOST_VULKAN": "1"
        }
    }
}
VKEOF
    gb_ok "Vulkan layer: ${C_DIM}$_vk_user_manifest${C_RESET}"
else
    gb_warn "Vulkan layer library not found at $_vk_src_lib"
    gb_info "  Run: ${C_CYAN}sudo greenboost_setup.sh install-sys-configs${C_RESET} first"
fi

# ── Report installation path ──────────────────────────────────────────────────
echo ""
printf '%b\n' "${C_VIOLET}${C_BOLD}  Installation complete${C_RESET}"
echo ""
gb_ok "Stable entry:       ${C_LIME}$INSTALL_DIR${C_RESET}"
gb_ok "Experimental entry: ${C_LIME}$INSTALL_DIR_EXP${C_RESET}"
echo ""
gb_info "Entries and their upstream Proton:"
gb_info "  ${C_LIME}GreenBoost Proton${C_RESET}              → wraps latest stable Proton (${_proton_stable_name:-not found})"
gb_info "  ${C_LIME}GreenBoost Proton Experimental${C_RESET} → wraps Proton Experimental"
echo ""
gb_info "Next steps:"
gb_info "  1. Restart Steam"
gb_info "  2. Right-click a game → Properties → Compatibility"
gb_info "  3. Select ${C_LIME}GreenBoost Proton${C_RESET} (stable) or ${C_LIME}GreenBoost Proton Experimental${C_RESET}"
gb_info ""
gb_info "To update, re-run this script."
gb_info "Optional launch options:"
printf '%b\n' "         ${C_CYAN}PROTON_ENABLE_WAYLAND=0 %command%${C_RESET}  ${C_DIM}(fall back to XWayland)${C_RESET}"
printf '%b\n' "         ${C_CYAN}GREENBOOST_NO_DXR=1 %command%${C_RESET}    ${C_DIM}(disable DXR injection)${C_RESET}"
echo ""
gb_info "Monitor: ${C_CYAN}greenboost vulkan${C_RESET}"
echo ""

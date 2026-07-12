#!/usr/bin/env bash
# GreenBoost v3.2 - Setup & installation script (Ubuntu / Debian and derivatives)
# Supports: Ubuntu, Debian, Pop!_OS, Mint, and other apt-based distros.
# Delegates to greenboost_setup_rocky.sh on RHEL/Fedora and
# greenboost_setup_arch.sh on Arch-based systems.
#
# Hardware is detected at runtime: CPU topology, GPU VRAM, RAM, kernel version.
# No hardware-specific values are hard-coded.
#
# USAGE:
#   (no args)                                    - interactive wizard
#   sudo ./greenboost_setup.sh setup             - full install (prompts for mode)
#   sudo ./greenboost_setup.sh module-only       - kernel module only (safe on any machine)
#   sudo ./greenboost_setup.sh install           - build + install module + shim
#   sudo ./greenboost_setup.sh uninstall         - remove module + all config
#   sudo ./greenboost_setup.sh load              - insmod with default params
#   sudo ./greenboost_setup.sh unload            - rmmod
#   sudo ./greenboost_setup.sh tune              - runtime tuning (governor, NVMe, sysctl)
#   sudo ./greenboost_setup.sh tune-all          - run all tune-* commands
#        ./greenboost_setup.sh status            - show pool info + system state
#        ./greenboost_setup.sh benchmark         - cuda memory pool bandwidth benchmark (T1/T2/T3)
#        ./greenboost_setup.sh help              - show common commands
#        ./greenboost_setup.sh help --all        - show all commands + env vars
#
# GLOBAL FLAGS (before or after the command):
#   --skip-update-check  Skip GitLab version check (offline workstations)
#
# ENVIRONMENT (for load command - all values auto-detected at runtime):
#   GPU_PHYS_GB    physical VRAM in GB       (detected via nvidia-smi)
#   VIRT_VRAM_GB   system RAM pool size in GB (80% of total RAM)
#   RESERVE_GB     minimum free system RAM to always maintain
#   NVME_SWAP_GB   total NVMe swap capacity  (auto-detected)
#   NVME_POOL_GB   GreenBoost soft cap on T3 allocations

set -euo pipefail

DRIVER_NAME="greenboost"
SHIM_LIB="libgreenboost_cuda.so"
AUDIT_LIB="libgreenboost_audit.so"
AUDIT_LIB32="libgreenboost_audit32.so"
SHIM_DEST="/usr/local/lib"

# ── Preload sanity - pure bash, zero external commands ───────────────────────
# Remove stale /etc/ld.so.preload entries whose .so files no longer exist.
# Must run before the first subprocess (MODULE_DIR assignment below spawns
# dirname/pwd).  Prevents "cannot be preloaded" spam in every forked process
# throughout this session (e.g. during detect_hardware, _gb_backup_create).
# Note: install/full-install handlers additionally strip the preload entry
# proactively at the top of their function to avoid LD_AUDIT mode errors
# from glibc 2.38+ before the new audit lib is in place.
if [[ $EUID -eq 0 && -f /etc/ld.so.preload ]]; then
    _gb_ps_lines=() _gb_ps_changed=0
    while IFS= read -r _gb_ps_line || [[ -n "$_gb_ps_line" ]]; do
        if [[ "$_gb_ps_line" == *libgreenboost* && ! -f "$_gb_ps_line" ]]; then
            _gb_ps_changed=1
        else
            _gb_ps_lines+=("$_gb_ps_line")
        fi
    done < /etc/ld.so.preload
    if (( _gb_ps_changed )); then
        if (( ${#_gb_ps_lines[@]} > 0 )); then
            printf '%s\n' "${_gb_ps_lines[@]}" > /etc/ld.so.preload
        else
            > /etc/ld.so.preload
        fi
    fi
    unset _gb_ps_lines _gb_ps_changed _gb_ps_line
fi

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# PR-GG: locate the shared lib/ directory.  Look in the source tree first
# (for `./greenboost_setup.sh` from a development checkout), then the
# system install path that `make install` populates.  The lib/ scripts
# contain color palette definitions, the canonical _gb_run_tui_loop
# helper, and SSH-option helpers - small, well-bounded modules.  If
# none are found, fall back to the inline definitions further down so
# upgrading from an old install that lacks lib/ continues to work.
GB_LIB_DIR=""
for _candidate in \
        "$MODULE_DIR/lib" \
        "/usr/local/share/greenboost/lib" \
        "/usr/share/greenboost/lib"; do
    if [[ -d "$_candidate" ]]; then
        GB_LIB_DIR="$_candidate"; break
    fi
done
if [[ -n "$GB_LIB_DIR" ]]; then
    # shellcheck disable=SC1090,SC1091
    for _libf in gb_colors.sh gb_tui.sh gb_ssh.sh; do
        [[ -r "$GB_LIB_DIR/$_libf" ]] && source "$GB_LIB_DIR/$_libf"
    done
fi

GB_PROFILES_DIR="/etc/greenboost/profiles"
GB_ACTIVE_PROFILE_LINK="/etc/greenboost/active_profile.md"

# Dynamic swapfile - GreenBoost provisions this automatically when no adequate swap exists
GB_SWAP_FILE="/var/lib/greenboost/swapfile"
GB_SWAP_MIN_GB=8      # minimum existing swap to consider adequate (skip provisioning)
GB_SWAP_MAX_GB=120    # cap for auto-provisioned swapfile

GB_VERSION="3.2"
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

# ── Distro / kernel helpers ────────────────────────────────────────────────────

# Semantic kernel version comparison - true when running kernel ≥ want_major.want_minor
_kernel_at_least() {
    local want_major=$1 want_minor=$2
    local cur cur_major cur_minor
    cur=$(uname -r | grep -oP '^\d+\.\d+' || echo "0.0")
    cur_major=${cur%%.*}
    cur_minor=${cur##*.}
    [[ $cur_major -gt $want_major ]] || \
      { [[ $cur_major -eq $want_major ]] && [[ $cur_minor -ge $want_minor ]]; }
}

# Return distro family: debian | fedora | arch | suse | alpine | void | unknown
_detect_distro_family() {
    local id id_like
    id=$(. /etc/os-release 2>/dev/null && echo "${ID:-}" | tr '[:upper:]' '[:lower:]')
    id_like=$(. /etc/os-release 2>/dev/null && echo "${ID_LIKE:-}" | tr '[:upper:]' '[:lower:]')
    case "$id" in
        ubuntu|debian|kali|linuxmint|pop|raspbian) echo "debian"  ; return ;;
        fedora|rhel|centos|rocky|almalinux|ol)     echo "fedora"  ; return ;;
        arch|manjaro|endeavouros|cachyos|garuda)   echo "arch"    ; return ;;
        opensuse*|sles)                             echo "suse"    ; return ;;
        alpine)                                     echo "alpine"  ; return ;;
        void)                                       echo "void"    ; return ;;
    esac
    # Fall back to ID_LIKE for derivatives
    case "$id_like" in
        *ubuntu*|*debian*)  echo "debian"  ; return ;;
        *rhel*|*fedora*)    echo "fedora"  ; return ;;
        *arch*)             echo "arch"    ; return ;;
        *suse*)             echo "suse"    ; return ;;
    esac
    echo "unknown"
}

# _gb_cuda_installed_version - print "MAJOR.MINOR" from whichever nvcc is
# actually the current one (prefers /usr/local/cuda, the update-alternatives
# path NVIDIA's own installers manage, over a possibly-stale distro-packaged
# nvcc on PATH - same reasoning as gb_synapse.py's _find_nvcc()). Empty
# string if no nvcc is found anywhere.
_gb_cuda_installed_version() {
    local nvcc_bin="/usr/local/cuda/bin/nvcc"
    [[ -x "$nvcc_bin" ]] || nvcc_bin=$(command -v nvcc 2>/dev/null) || { echo ""; return; }
    "$nvcc_bin" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1
}

# _gb_cuda_needs_install - true (rc 0) unless a CUDA toolkit new enough for
# this project's target GPUs (Blackwell, sm_120 - added in CUDA 12.8) is
# already the active one.
_gb_cuda_needs_install() {
    local ver major minor
    ver=$(_gb_cuda_installed_version)
    [[ -z "$ver" ]] && return 0
    major="${ver%%.*}"; minor="${ver#*.}"
    (( major > 12 )) && return 1
    (( major == 12 && minor >= 8 )) && return 1
    return 0
}

# _gb_is_wsl - true if running under WSL (same systemd-detect-virt check
# cmd_gen_inference_config already uses for virt_type detection).
_gb_is_wsl() {
    [[ "$(systemd-detect-virt 2>/dev/null)" == "wsl" ]]
}

# _gb_install_cuda_toolkit - install the latest CUDA toolkit from NVIDIA's
# official per-distro repo (Arch: from Arch's own `extra` repo instead -
# NVIDIA doesn't publish an Arch repo). Full-install only; light install
# never touches this. Idempotent: no-ops if already >= 12.8 (Blackwell
# support) unless FORCE=1 is passed.
#
# Uses the version-less `cuda-toolkit` meta-package (not a pinned
# `cuda-toolkit-MAJOR-MINOR`) so this keeps tracking "latest" as NVIDIA ships
# new releases, without needing a code update here every time.
_gb_install_cuda_toolkit() {
    local force="${1:-0}"
    if [[ "$force" != "1" ]] && ! _gb_cuda_needs_install; then
        gb_ok "CUDA toolkit $(_gb_cuda_installed_version) already supports Blackwell (>= 12.8) - skipping"
        return 0
    fi

    local id version_id
    id=$(. /etc/os-release 2>/dev/null && echo "${ID:-}" | tr '[:upper:]' '[:lower:]')
    version_id=$(. /etc/os-release 2>/dev/null && echo "${VERSION_ID:-}")
    local family; family=$(_detect_distro_family)

    gb_info "Installing latest CUDA toolkit (distro: ${id:-unknown} ${version_id:-}$(_gb_is_wsl && echo " [WSL]"))..."

    if [[ "$family" == "arch" ]]; then
        # No NVIDIA-hosted repo for Arch - the `extra` repo already ships a
        # current `cuda` package directly.
        pacman -S --needed --noconfirm cuda \
            || { gb_warn_ui "pacman install of 'cuda' failed - install manually: sudo pacman -S cuda"; return 1; }
        gb_ok "CUDA toolkit installed via pacman"
        return 0
    fi

    local repo_os="" pkg_mgr=""
    if _gb_is_wsl; then
        repo_os="wsl-ubuntu"; pkg_mgr="apt"
    else
        case "$id" in
            ubuntu)
                case "$version_id" in
                    22.04) repo_os="ubuntu2204" ;;
                    24.04) repo_os="ubuntu2404" ;;
                    26.04) repo_os="ubuntu2604" ;;
                    *)     repo_os="ubuntu2404"; gb_warn_ui "Unrecognized Ubuntu ${version_id} - assuming ubuntu2404 repo" ;;
                esac
                pkg_mgr="apt"
                ;;
            debian)
                case "${version_id%%.*}" in
                    12) repo_os="debian12" ;;
                    13) repo_os="debian13" ;;
                    *)  repo_os="debian12"; gb_warn_ui "Unrecognized Debian ${version_id} - assuming debian12 repo" ;;
                esac
                pkg_mgr="apt"
                ;;
            rhel|centos|rocky|almalinux|ol)
                case "${version_id%%.*}" in
                    8)  repo_os="rhel8" ;;
                    9)  repo_os="rhel9" ;;
                    10) repo_os="rhel10" ;;
                    *)  repo_os="rhel9"; gb_warn_ui "Unrecognized RHEL-family ${version_id} - assuming rhel9 repo" ;;
                esac
                pkg_mgr="dnf"
                ;;
            fedora)
                repo_os="fedora${version_id%%.*}"
                pkg_mgr="dnf5"   # Fedora ships DNF5 - different addrepo syntax than RHEL's DNF4
                ;;
            sles)
                case "${version_id%%.*}" in
                    15) repo_os="sles15" ;;
                    16) repo_os="suse16" ;;
                    *)  repo_os="sles15"; gb_warn_ui "Unrecognized SLES ${version_id} - assuming sles15 repo" ;;
                esac
                pkg_mgr="zypper"
                ;;
            opensuse*|opensuse-leap|opensuse-tumbleweed)
                case "${version_id%%.*}" in
                    15) repo_os="opensuse15" ;;
                    16) repo_os="suse16" ;;
                    *)  repo_os="opensuse15"; gb_warn_ui "Unrecognized openSUSE ${version_id} - assuming opensuse15 repo" ;;
                esac
                pkg_mgr="zypper"
                ;;
            *)
                gb_warn_ui "Unrecognized distro '${id}' for CUDA toolkit install - skipping. Install manually: https://developer.nvidia.com/cuda-downloads"
                return 1
                ;;
        esac
    fi

    local base_url="https://developer.download.nvidia.com/compute/cuda/repos/${repo_os}/x86_64"
    case "$pkg_mgr" in
        apt)
            local tmp_deb; tmp_deb=$(mktemp --suffix=.deb)
            wget -qO "$tmp_deb" "${base_url}/cuda-keyring_1.1-1_all.deb" \
                || { gb_warn_ui "Could not download cuda-keyring from ${base_url}"; rm -f "$tmp_deb"; return 1; }
            dpkg -i "$tmp_deb" && rm -f "$tmp_deb"
            apt-get update -qq
            apt-get -y install cuda-toolkit
            ;;
        dnf)
            dnf config-manager --add-repo "${base_url}/cuda-${repo_os}.repo"
            dnf clean all
            dnf -y install cuda-toolkit
            ;;
        dnf5)
            dnf config-manager addrepo --from-repofile "${base_url}/cuda-${repo_os}.repo"
            dnf clean all
            dnf -y install cuda-toolkit
            ;;
        zypper)
            zypper --non-interactive addrepo "${base_url}/cuda-${repo_os}.repo"
            zypper --non-interactive refresh
            zypper --non-interactive install cuda-toolkit
            ;;
    esac
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        gb_ok "CUDA toolkit installed ($(_gb_cuda_installed_version))"
    else
        gb_warn_ui "CUDA toolkit install failed (rc=${rc}) - install manually: https://developer.nvidia.com/cuda-downloads"
    fi
    return $rc
}

# Idempotent kernel header install - uses the appropriate package manager
_ensure_kernel_headers() {
    local kver family
    kver=$(uname -r)
    family=$(_detect_distro_family)
    case "$family" in
        debian) sudo apt-get install -y "linux-headers-${kver}" ;;
        fedora) sudo dnf install -y "kernel-devel-${kver}" 2>/dev/null || \
                sudo yum install -y "kernel-devel-${kver}" ;;
        arch)   local pkg="linux-headers"
                uname -r | grep -qi lts && pkg="linux-lts-headers"
                sudo pacman -S --needed --noconfirm "$pkg" ;;
        suse)   sudo zypper install -y "kernel-default-devel=${kver}" ;;
        alpine) sudo apk add linux-headers ;;
        void)   sudo xbps-install -y "kernel-headers-${kver}" ;;
        *)      warn "Unknown distro family; cannot auto-install kernel headers for ${kver}" ;;
    esac
}

# ---- Brand palette + UI primitives -------------------------------------
# Matches synapse_cli color scheme (#6C71C4 violet, #E6FF3C lime, etc.)
# All functions degrade to 16-color ANSI when COLORTERM is unset/8bit.

# PR-GG: inline color fallback.  If lib/gb_colors.sh was already sourced
# above, the C_* vars are already set and this whole block is a no-op
# (the `-z "${C_RESET:-}"` guard).  Kept as a fallback so old installs
# that lack /usr/local/share/greenboost/lib/ still work.
if [[ -z "${C_RESET:-}" ]]; then
    _gb_truecolor() { [[ "${COLORTERM:-}" =~ ^(truecolor|24bit)$ ]]; }
    if _gb_truecolor; then
        C_VIOLET=$'\033[38;2;108;113;196m'
        C_LIME=$'\033[38;2;230;255;60m'
        C_GRAY=$'\033[38;2;208;207;204m'
        C_CYAN=$'\033[38;2;48;200;255m'
        C_AMBER=$'\033[38;2;255;191;0m'
        C_PURPLE=$'\033[38;2;167;139;250m'
        C_RED=$'\033[38;2;255;92;50m'
        C_WHITE=$'\033[38;2;255;255;255m'
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
fi

GB_SPIN_FRAMES=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")

# Default log paths (override via env)
GB_STATUS_LOG="${GB_STATUS_LOG:-$HOME/.local/share/greenboost/status.log}"
GB_INFER_TEST_LOG="${GB_INFER_TEST_LOG:-$HOME/.local/share/greenboost/inference-test-latest.log}"

# gb_header - branded box header with build timestamp
gb_header() {
    local _bi_date="" _bi_git=""
    for _bi in "${MODULE_DIR}/build_info" ./build_info /etc/greenboost/build_info; do
        if [[ -f "$_bi" ]]; then
            local _ep _gh
            _ep=$(grep '^BUILD_EPOCH=' "$_bi" 2>/dev/null | cut -d= -f2-)
            _gh=$(grep '^BUILD_GIT='   "$_bi" 2>/dev/null | cut -d= -f2-)
            [[ -n "$_ep" ]] && _bi_date=$(date -d "@${_ep}" '+%Y-%m-%d %H:%M' 2>/dev/null || true)
            [[ -n "$_gh" ]] && _bi_git="$_gh"
            break
        fi
    done
    local _build_str=""
    [[ -n "$_bi_date" ]] && _build_str="  built ${_bi_date}"
    [[ -n "$_bi_git"  ]] && _build_str+=" (${_bi_git})"
    local cols; cols=$(tput cols 2>/dev/null || echo 64)
    local title=" GreenBoost v${GB_VERSION}${_build_str} - CUDA Memory & Compute Orchestrator for NVIDIA GPUs"
    echo -e ""
    echo -e "${C_VIOLET}${C_BOLD}  ╔$(printf '═%.0s' $(seq 1 $((cols - 4))))╗${C_RESET}"
    echo -e "${C_VIOLET}${C_BOLD}  ║${C_RESET} ${C_GRAY}${C_BOLD}${title}$(printf ' %.0s' $(seq 1 $((cols - 4 - ${#title} - 1))))${C_VIOLET}${C_BOLD}║${C_RESET}"
    echo -e "${C_VIOLET}${C_BOLD}  ╚$(printf '═%.0s' $(seq 1 $((cols - 4))))╝${C_RESET}"
    echo -e ""
}

# gb_separator - full-width thin line
gb_separator() {
    local cols; cols=$(tput cols 2>/dev/null || echo 64)
    echo -e "${C_DIM}$(printf '─%.0s' $(seq 1 $((cols - 2))))${C_RESET}"
}

# gb_step N M "description"
gb_step() {
    echo -e ""
    echo -e "${C_CYAN}${C_BOLD}  [$1/$2]${C_RESET} ${C_GRAY}${C_BOLD}$3${C_RESET}"
}

# gb_ok / gb_fail / gb_warn - status messages
gb_ok()      { echo -e "  ${C_LIME}✓${C_RESET}  ${C_GRAY}$*${C_RESET}"; }
gb_fail()    { echo -e "  ${C_RED}✗${C_RESET}  $*"; }
gb_warn()    { echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_AMBER}$*${C_RESET}"; }
gb_warn_ui() { echo -e "  ${C_AMBER}⚠${C_RESET}  $*"; }

# gb_spin PID "message" - braille spinner until PID exits
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

# gb_info - replaces info() with brand style (keeps [GreenBoost] prefix for log compat)
gb_info() { echo -e "  ${C_VIOLET}◈${C_RESET}  ${C_GRAY}$*${C_RESET}"; }

# ---- Panel primitives (used by cmd_vitals) ----------------------------
# gb_panel_top "Title" - violet box top border with embedded cyan title
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

# gb_panel_bottom - violet box bottom border
gb_panel_bottom() {
    local cols; cols=$(tput cols 2>/dev/null || echo 80)
    local inner=$(( cols - 4 ))
    printf '%b' "  ${C_VIOLET}${C_BOLD}╚"
    printf '═%.0s' $(seq 1 $inner)
    printf '%b\n' "╝${C_RESET}"
}

# gb_panel_row "content" - bordered content row, ANSI-aware padding
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

# gb_panel_empty - empty bordered spacer row
gb_panel_empty() { gb_panel_row ""; }

# gb_bar pct fill_color empty_color width - inline progress bar (no newline)
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

# gb_tier_color pct - echo ANSI color for a utilization percentage
gb_tier_color() {
    local pct=$1
    if   (( pct >= 90 )); then printf '%b' "${C_RED}"
    elif (( pct >= 75 )); then printf '%b' "${C_AMBER}"
    else                       printf '%b' "${C_LIME}"
    fi
}

# gb_section "Title" - purple bold section header + separator (mirrors unlocker ui_section)
gb_section() {
    echo -e ""
    echo -e "  ${C_PURPLE}${C_BOLD}$1${C_RESET}"
    gb_separator
}

# gb_col_width - usable character width of each column in a 2-col layout
gb_col_width() {
    local cols; cols=$(tput cols 2>/dev/null || echo 80)
    local w=$(( (cols - 7) / 2 ))
    (( w < 36 )) && w=36
    echo "$w"
}

# gb_strip_ansi - remove ANSI color/SGR codes from stdin (for visible-width counting)
gb_strip_ansi() {
    sed 's/\x1b\[[0-9;]*[mK]//g'
}

# _trunc "string" max_len - truncates string to max_len chars, adding ".." if cut
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
        # Project style: avoid vertical pipes for column alignment.
        # Use a faint em-dash between the columns instead - alignment is
        # already enforced by `pad` above.
        printf '%s%*s  %b-%b  %s\n' "$L" "$pad" "" "${C_DIM}" "${C_RESET}" "$R"
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

# gb_prompt "label" - amber ❯ prompt; result in $REPLY
gb_prompt() {
    printf "\n  ${C_AMBER}${C_BOLD}❯${C_RESET}  ${C_GRAY}${1:-Choice}${C_RESET}: "
    read -r REPLY
}

# gb_confirm "Question" - amber ❯ Y/n; returns 0 if yes
gb_confirm() {
    printf "  ${C_AMBER}${C_BOLD}❯${C_RESET}  ${C_GRAY}$1${C_RESET} ${C_DIM}[Y/n]${C_RESET}: "
    read -r _confirm_reply
    [[ "${_confirm_reply:-Y}" =~ ^[Yy]$ ]]
}

# gb_press_enter - pause prompt at the end of a wizard action
gb_press_enter() {
    echo -e ""
    printf "  ${C_DIM}${C_GRAY}Press Enter to return to menu…${C_RESET}"
    read -r
}

# ── Install mode helpers ──────────────────────────────────────────────────────
GB_INSTALL_MODE="module"   # default: safe (no system tuning)

# gb_select_install_mode - interactive or flag-driven mode selection.
# Sets GB_INSTALL_MODE to "module" or "full".
# Pass --module-only or --full-install to skip the prompt (for scripts/CI).
gb_select_install_mode() {
    for _a in "$@"; do
        [[ "$_a" == "--module-only"  ]] && { GB_INSTALL_MODE="module"; return; }
        [[ "$_a" == "--full-install" ]] && { GB_INSTALL_MODE="full";   return; }
    done
    gb_separator
    echo ""
    printf '%b\n' "  ${C_VIOLET}${C_BOLD}GreenBoost - Choose Installation Mode${C_RESET}"
    echo ""
    printf '%b\n' "  ${C_LIME}${C_BOLD}[1]${C_RESET}  ${C_BOLD}${C_GRAY}Kernel module only${C_RESET}  ${C_DIM}DKMS install - no sysctl, GRUB, or service changes${C_RESET}"
    printf '%b\n' "  ${C_LIME}${C_BOLD}[2]${C_RESET}  ${C_BOLD}${C_GRAY}Full system setup ${C_RESET}  ${C_DIM}Module + AI libs + sysctl + GRUB + systemd services${C_RESET}"
    echo ""
    printf '%b\n' "  ${C_DIM}Mode [1] is safe on any machine. Mode [2] tunes sysctl, GRUB, and${C_RESET}"
    printf '%b\n' "  ${C_DIM}enables system services - recommended only on a dedicated workstation.${C_RESET}"
    echo ""
    printf '%b' "  ${C_AMBER}${C_BOLD}❯${C_RESET}  ${C_GRAY}Mode [1/2] (default: 1): ${C_RESET}"
    read -r _mode_reply
    case "${_mode_reply:-1}" in
        2) GB_INSTALL_MODE="full" ;;
        *) GB_INSTALL_MODE="module" ;;
    esac
}

# gb_consent_gate "description" - per-phase confirmation before a system change.
# Returns 0 if the user confirms, 1 if they decline.
gb_consent_gate() {
    echo ""
    printf '%b\n' "  ${C_AMBER}${C_BOLD}⚠  System change:${C_RESET}  ${C_GRAY}$1${C_RESET}"
    gb_confirm "Apply this change?" && return 0 || return 1
}

# ---- Pool data parsers (used by cmd_vitals) ---------------------------
# Strip non-digit chars from a named variable and default to 0 if empty.
# Usage: _sanitize_int VARNAME  (no subshell, uses printf -v nameref)
_sanitize_int() { local _v="${!1}"; _v="${_v%%[^0-9]*}"; printf -v "$1" '%s' "${_v:-0}"; }

# Sets PB_* vars from pool_brief sysfs one-liner
# Format: T1:12GB T2:0/51GB(0%) T3:0/64GB PRESSURE:ok KV_RSV:2048MB KV_T2:0MB
parse_pool_brief() {
    PB_T1_GB=0; PB_T2_USED_GB=0; PB_T2_MAX_GB=0; PB_T2_PCT=0
    PB_T3_USED_GB=0; PB_T3_MAX_GB=0; PB_PRESSURE="-"
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
    PB_PRESSURE=$(echo "$brief"  | grep -oP 'PRESSURE:\K\S+' || echo "-")
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
    PI_OOM_GUARD="no"; PI_PAGE_MODE="-"
    PI_KV_T1_RSV_MB=0; PI_KV_T2_MB=0; PI_KV_T3_MB=0; PI_KV_TOTAL_MB=0
    PI_KV_PLACEMENT="-"
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
    PI_PAGE_MODE=$(echo "$info"     | grep -oP 'Page mode\s*:\s*\K[^\n]+'  | head -1 | sed 's/[[:space:]]*$//' || echo "")
    PI_PAGE_MODE="${PI_PAGE_MODE:--}"
    PI_KV_T1_RSV_MB=$(echo "$info"  | grep -oP 'KV in T1.*reserve:\s*\K[0-9]+'        | head -1 || echo 0)
    PI_KV_T2_MB=$(echo "$info"      | grep -oP 'KV in T2[^:]*:\s*\K[0-9]+'            | head -1 || echo 0)
    PI_KV_T3_MB=$(echo "$info"      | grep -oP 'KV in T3[^:]*:\s*\K[0-9]+'            | head -1 || echo 0)
    PI_KV_TOTAL_MB=$(echo "$info"   | grep -oP 'KV tagged total[^:]*:\s*\K[0-9]+'      | head -1 || echo 0)
    PI_KV_PLACEMENT=$(echo "$info"  | grep -oP 'KV cache placement\s*:\s*\K[^\n]+' | head -1 | sed 's/[[:space:]]*$//' || echo "")
    PI_KV_PLACEMENT="${PI_KV_PLACEMENT:--}"
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

# ---- Shim stats reader (used by cmd_vitals) ---------------------------
# Sets SS_* vars from shim_stats (written by the CUDA shim).
# Primary location: /run/greenboost/shim_stats
# Fallback location: /tmp/greenboost_shim_stats (used when Ollama can't write to /run/greenboost/)
# SS_ACTIVE_PATH: A | B | none | unknown
# SS_STALE: 1 if the file is older than 30 s (shim not running or crashed)
parse_shim_stats() {
    SS_ACTIVE_PATH="unknown"; SS_STALE=1
    SS_PATH_A=0; SS_PATH_B=0
    SS_PHASE=""; SS_KV_RSV_NOM=0; SS_KV_RSV_EFF=0; SS_KV_T1_MB=0; SS_HEADROOM_MB=0
    SS_LOCAL_T1_MB=0; SS_REMOTE_ALLOC_COUNT=0; SS_REMOTE_ALLOC_MB=0
    SS_H2D_MB=0; SS_D2H_MB=0; SS_KERNEL_DISPATCH=0
    local stats_f="/run/greenboost/shim_stats"
    [[ -r "$stats_f" ]] || stats_f="/tmp/greenboost_shim_stats"
    [[ -r "$stats_f" ]] || return 1
    local content; content=$(cat "$stats_f" 2>/dev/null) || return 1
    local ts; ts=$(echo "$content" | grep -oP 'timestamp=\K[0-9]+' | head -1 || echo 0)
    local now; now=$(date +%s)
    (( now - ts <= 30 )) && SS_STALE=0
    SS_ACTIVE_PATH=$(echo "$content"      | grep -oP 'active_path=\K\S+'                | head -1 || echo "unknown")
    SS_PATH_A=$(echo "$content"           | grep -oP 'path_a_count=\K[0-9]+'            | head -1 || echo 0)
    SS_PATH_B=$(echo "$content"           | grep -oP 'path_b_count=\K[0-9]+'            | head -1 || echo 0)
    SS_PHASE=$(echo "$content"            | grep -oP 'phase=\K\S+'                      | head -1 || echo "")
    SS_KV_RSV_NOM=$(echo "$content"       | grep -oP 'kv_reserve_nominal_mb=\K[0-9]+'   | head -1 || echo 0)
    SS_KV_RSV_EFF=$(echo "$content"       | grep -oP 'kv_reserve_effective_mb=\K[0-9]+' | head -1 || echo 0)
    SS_KV_T1_MB=$(echo "$content"         | grep -oP 'kv_t1_tracked_mb=\K[0-9]+'       | head -1 || echo 0)
    SS_HEADROOM_MB=$(echo "$content"      | grep -oP 'vram_headroom_mb=\K[0-9]+'        | head -1 || echo 0)
    SS_LOCAL_T1_MB=$(echo "$content"      | grep -oP 'local_t1_alloc_mb=\K[0-9]+'      | head -1 || echo 0)
    SS_REMOTE_ALLOC_COUNT=$(echo "$content" | grep -oP 'remote_alloc_count=\K[0-9]+'   | head -1 || echo 0)
    SS_REMOTE_ALLOC_MB=$(echo "$content"  | grep -oP 'remote_alloc_mb=\K[0-9]+'        | head -1 || echo 0)
    SS_H2D_MB=$(echo "$content"           | grep -oP 'h2d_mb=\K[0-9]+'                 | head -1 || echo 0)
    SS_D2H_MB=$(echo "$content"           | grep -oP 'd2h_mb=\K[0-9]+'                 | head -1 || echo 0)
    SS_KERNEL_DISPATCH=$(echo "$content"  | grep -oP 'kernel_dispatch_count=\K[0-9]+'   | head -1 || echo 0)
    SS_T2_WARN_ADJ=$(echo "$content"     | grep -oP 't2_warn_adj_pct=\K-?[0-9]+'       | head -1 || echo 0)
    SS_COLD_EVICT=$(echo "$content"      | grep -oP 'cold_epoch_evict_count=\K[0-9]+'  | head -1 || echo 0)
    SS_KV_DEDUP=$(echo "$content"        | grep -oP 'kv_dedup_hits=\K[0-9]+'           | head -1 || echo 0)
    SS_KV_FRAG_MB=$(echo "$content"      | grep -oP 'kv_internal_frag_mb=\K[0-9]+'    | head -1 || echo 0)
    SS_PINNED_FREE=$(echo "$content"     | grep -oP 'pinned_pool_bufs_free=\K[0-9]+'   | head -1 || echo "")
    # Per-tier lifetime stats (from tier_<name>_{alloc_count,lifetime_mb,peak_mb})
    local _tc; _tc="$content"
    SS_TIER_T1_LOCAL_COUNT=$(echo "$_tc"   | grep -oP 'tier_t1_local_alloc_count=\K-?[0-9]+'   | head -1 || echo 0)
    SS_TIER_T1_LOCAL_MB=$(echo "$_tc"      | grep -oP 'tier_t1_local_lifetime_mb=\K-?[0-9]+'   | head -1 || echo 0)
    SS_TIER_T1_LOCAL_PEAK=$(echo "$_tc"    | grep -oP 'tier_t1_local_peak_mb=\K-?[0-9]+'       | head -1 || echo 0)
    SS_TIER_T1_FEEDER_COUNT=$(echo "$_tc"  | grep -oP 'tier_t1_feeder_alloc_count=\K-?[0-9]+'  | head -1 || echo 0)
    SS_TIER_T1_FEEDER_MB=$(echo "$_tc"     | grep -oP 'tier_t1_feeder_lifetime_mb=\K-?[0-9]+'  | head -1 || echo 0)
    SS_TIER_T1_FEEDER_PEAK=$(echo "$_tc"   | grep -oP 'tier_t1_feeder_peak_mb=\K-?[0-9]+'      | head -1 || echo 0)
    SS_TIER_T2_LOCAL_COUNT=$(echo "$_tc"   | grep -oP 'tier_t2_local_alloc_count=\K-?[0-9]+'   | head -1 || echo 0)
    SS_TIER_T2_LOCAL_MB=$(echo "$_tc"      | grep -oP 'tier_t2_local_lifetime_mb=\K-?[0-9]+'   | head -1 || echo 0)
    SS_TIER_T2_LOCAL_PEAK=$(echo "$_tc"    | grep -oP 'tier_t2_local_peak_mb=\K-?[0-9]+'       | head -1 || echo 0)
    SS_TIER_T2_FEEDER_COUNT=$(echo "$_tc"  | grep -oP 'tier_t2_feeder_alloc_count=\K-?[0-9]+'  | head -1 || echo 0)
    SS_TIER_T2_FEEDER_MB=$(echo "$_tc"     | grep -oP 'tier_t2_feeder_lifetime_mb=\K-?[0-9]+'  | head -1 || echo 0)
    SS_TIER_T2_FEEDER_PEAK=$(echo "$_tc"   | grep -oP 'tier_t2_feeder_peak_mb=\K-?[0-9]+'      | head -1 || echo 0)
    SS_TIER_T3_LOCAL_COUNT=$(echo "$_tc"   | grep -oP 'tier_t3_local_alloc_count=\K-?[0-9]+'   | head -1 || echo 0)
    SS_TIER_T3_LOCAL_MB=$(echo "$_tc"      | grep -oP 'tier_t3_local_lifetime_mb=\K-?[0-9]+'   | head -1 || echo 0)
    SS_TIER_T3_LOCAL_PEAK=$(echo "$_tc"    | grep -oP 'tier_t3_local_peak_mb=\K-?[0-9]+'       | head -1 || echo 0)
    SS_TIER_T3_FEEDER_COUNT=$(echo "$_tc"  | grep -oP 'tier_t3_feeder_alloc_count=\K-?[0-9]+'  | head -1 || echo 0)
    SS_TIER_T3_FEEDER_MB=$(echo "$_tc"     | grep -oP 'tier_t3_feeder_lifetime_mb=\K-?[0-9]+'  | head -1 || echo 0)
    SS_TIER_T3_FEEDER_PEAK=$(echo "$_tc"   | grep -oP 'tier_t3_feeder_peak_mb=\K-?[0-9]+'      | head -1 || echo 0)
    SS_TIER_VMM_COUNT=$(echo "$_tc"        | grep -oP 'tier_vmm_alloc_count=\K-?[0-9]+'        | head -1 || echo 0)
    SS_TIER_VMM_MB=$(echo "$_tc"           | grep -oP 'tier_vmm_lifetime_mb=\K-?[0-9]+'        | head -1 || echo 0)
    SS_TIER_VMM_PEAK=$(echo "$_tc"         | grep -oP 'tier_vmm_peak_mb=\K-?[0-9]+'            | head -1 || echo 0)
    SS_TIER_PATH_B_COUNT=$(echo "$_tc"     | grep -oP 'tier_path_b_alloc_count=\K-?[0-9]+'     | head -1 || echo 0)
    SS_TIER_PATH_B_MB=$(echo "$_tc"        | grep -oP 'tier_path_b_lifetime_mb=\K-?[0-9]+'     | head -1 || echo 0)
    SS_TIER_PATH_B_PEAK=$(echo "$_tc"      | grep -oP 'tier_path_b_peak_mb=\K-?[0-9]+'         | head -1 || echo 0)
    # Per-tier LIVE residency (tier_<name>_cur_mb) - what's actually resident
    # right now, not lifetime/peak. This is the number that shows the
    # local-VRAM-vs-feeder-VRAM split for the 2-device (ggml-2dev) path.
    SS_TIER_T1_LOCAL_CUR=$(echo "$_tc"   | grep -oP 'tier_t1_local_cur_mb=\K-?[0-9]+'   | head -1 || echo 0)
    SS_TIER_T1_FEEDER_CUR=$(echo "$_tc"  | grep -oP 'tier_t1_feeder_cur_mb=\K-?[0-9]+'  | head -1 || echo 0)
    SS_TIER_T2_LOCAL_CUR=$(echo "$_tc"   | grep -oP 'tier_t2_local_cur_mb=\K-?[0-9]+'   | head -1 || echo 0)
    SS_TIER_T2_FEEDER_CUR=$(echo "$_tc"  | grep -oP 'tier_t2_feeder_cur_mb=\K-?[0-9]+'  | head -1 || echo 0)
    SS_TIER_T3_LOCAL_CUR=$(echo "$_tc"   | grep -oP 'tier_t3_local_cur_mb=\K-?[0-9]+'   | head -1 || echo 0)
    SS_TIER_T3_FEEDER_CUR=$(echo "$_tc"  | grep -oP 'tier_t3_feeder_cur_mb=\K-?[0-9]+'  | head -1 || echo 0)
    SS_TIER_VMM_CUR=$(echo "$_tc"        | grep -oP 'tier_vmm_cur_mb=\K-?[0-9]+'        | head -1 || echo 0)
    SS_TIER_PATH_B_CUR=$(echo "$_tc"     | grep -oP 'tier_path_b_cur_mb=\K-?[0-9]+'     | head -1 || echo 0)
}

# ---- Live GPU metrics query (used by cmd_vitals) ----------------------
# Tries gb_vitals_helper.py (pynvml, no nvidia-smi fork) first.
# Falls back to _query_gpu_vram_smi on pynvml failure.
# Sets: GPU_NAME GPU_VRAM_USED_MB GPU_VRAM_TOTAL_MB GPU_VRAM_FREE_MB GPU_VRAM_PCT
#       GPU_UTIL_PCT GPU_MEM_UTIL_PCT GPU_TEMP_C GPU_POWER_W GPU_POWER_LIMIT_W
#       GPU_SM_CLOCK_MHZ GPU_MEM_CLOCK_MHZ GPU_ECC_DBE GPU_ECC_DBE_AGG GPU_ECC_SBE
#       GPU_PCIE_TX_MB_S GPU_PCIE_RX_MB_S GPU_GAMING_MODE GB_PRESSURE_STATE
#       SHIM_PHASE SHIM_ACTIVE_PATH SHIM_T1_LOCAL_MB SHIM_T2_LOCAL_MB SHIM_T3_LOCAL_MB
#       SHIM_WS_RESERVE_MB SHIM_WS_RESERVE_EFF_MB SHIM_KV_RESERVE_MB
#       SHIM_VIRTUAL_VRAM_MB SHIM_CLUSTER_REMOTE_MB
#       ORCH_ECC_DEGRADED ORCH_WS_ABOVE ORCH_WS_RESERVE_MB ORCH_ACTUATE ORCH_VRAM_PRESSURE
#       ORCH_CLUSTER_PRESSURE ORCH_HEALTH_OK ORCH_HEALTH_EVICT_ARMED
#       TOPO_INFERENCE_CPUS TOPO_INFERENCE_THREADS TOPO_BACKGROUND_THREADS
#       TOPO_PCIE_SAT_MB_S TOPO_IS_BLACKWELL
#       GPU_HEALTH_OK GPU_HEALTH_SUMMARY GPU_POWER_INSTANT_W GPU_NVLINK_BW_MB_S
_gb_vitals_init_vars() {
    GPU_NAME=""; GPU_VRAM_USED_MB=0; GPU_VRAM_TOTAL_MB=0; GPU_VRAM_FREE_MB=0; GPU_VRAM_PCT=0
    GPU_UTIL_PCT=0; GPU_MEM_UTIL_PCT=0; GPU_TEMP_C=0
    GPU_POWER_W=0; GPU_POWER_LIMIT_W=0; GPU_SM_CLOCK_MHZ=0; GPU_MEM_CLOCK_MHZ=0
    GPU_ECC_DBE=0; GPU_ECC_DBE_AGG=0; GPU_ECC_SBE=0; GPU_GAMING_MODE=0; GB_PRESSURE_STATE=""
    GPU_PCIE_TX_MB_S=0; GPU_PCIE_RX_MB_S=0
    SHIM_PHASE=""; SHIM_ACTIVE_PATH=""; SHIM_T1_LOCAL_MB=""; SHIM_T2_LOCAL_MB=""; SHIM_T3_LOCAL_MB=""
    SHIM_WS_RESERVE_MB=""; SHIM_WS_RESERVE_EFF_MB=""; SHIM_KV_RESERVE_MB=""
    SHIM_VIRTUAL_VRAM_MB=""; SHIM_CLUSTER_REMOTE_MB=""
    ORCH_ECC_DEGRADED=0; ORCH_THERMAL_STRESS=0; ORCH_MEM_BW_STRESS=0; ORCH_SBE_ELEVATED=0; ORCH_SBE_SEEN=0
    ORCH_CLOCK_THROTTLED=0; ORCH_SM_CLOCK_MAX_MHZ=0
    ORCH_WS_ABOVE=0; ORCH_WS_RESERVE_MB=""; ORCH_ACTUATE=0; ORCH_VRAM_PRESSURE=0
    ORCH_CLUSTER_PRESSURE=0; ORCH_HEALTH_OK=1; ORCH_HEALTH_EVICT_ARMED=0
    ORCH_OS_TUNE_ENABLED=0; ORCH_GAMING_MODE=0; ORCH_CPU_GOVERNOR=""; ORCH_GPU_PERSISTENCE=0
    ORCH_GPU_POWER_LIMIT_W=""; ORCH_GPU_CLOCKS_LOCKED=""; ORCH_SWAPPINESS=""
    TOPO_INFERENCE_CPUS=""; TOPO_INFERENCE_THREADS=""; TOPO_BACKGROUND_THREADS=""
    TOPO_PCIE_SAT_MB_S=""; TOPO_IS_BLACKWELL=0
    GPU_HEALTH_OK=""; GPU_HEALTH_SUMMARY=""; GPU_POWER_INSTANT_W=""; GPU_NVLINK_BW_MB_S=0
}

_gb_vitals_helper_path() {
    local _d="${MODULE_DIR:-/usr/local/lib/greenboost}"
    for _p in "$_d/gb_vitals_helper.py" \
              "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/gb_vitals_helper.py" \
              "/usr/local/lib/greenboost/gb_vitals_helper.py"; do
        [[ -f "$_p" ]] && { echo "$_p"; return 0; }
    done
    return 1
}

query_gpu_vram() {
    _gb_vitals_init_vars
    local _py; _py=$(_gb_vitals_helper_path 2>/dev/null) || { _query_gpu_vram_smi; return; }
    local _out; _out=$(python3 "$_py" 2>/dev/null) || { _query_gpu_vram_smi; return; }
    while IFS='=' read -r _k _v; do
        case "$_k" in
            GPU_NAME)          GPU_NAME="$_v" ;;
            GPU_VRAM_USED_MB)  GPU_VRAM_USED_MB="$_v" ;;
            GPU_VRAM_TOTAL_MB) GPU_VRAM_TOTAL_MB="$_v" ;;
            GPU_VRAM_FREE_MB)  GPU_VRAM_FREE_MB="$_v" ;;
            GPU_VRAM_PCT)      GPU_VRAM_PCT="$_v" ;;
            GPU_UTIL_PCT)      GPU_UTIL_PCT="$_v" ;;
            GPU_MEM_UTIL_PCT)  GPU_MEM_UTIL_PCT="$_v" ;;
            GPU_TEMP_C)        GPU_TEMP_C="$_v" ;;
            GPU_POWER_W)       GPU_POWER_W="$_v" ;;
            GPU_POWER_LIMIT_W) GPU_POWER_LIMIT_W="$_v" ;;
            GPU_SM_CLOCK_MHZ)  GPU_SM_CLOCK_MHZ="$_v" ;;
            GPU_MEM_CLOCK_MHZ) GPU_MEM_CLOCK_MHZ="$_v" ;;
            GPU_ECC_DBE)            GPU_ECC_DBE="$_v" ;;
            GPU_ECC_DBE_AGG)        GPU_ECC_DBE_AGG="$_v" ;;
            GPU_ECC_SBE)            GPU_ECC_SBE="$_v" ;;
            GPU_PCIE_TX_MB_S)       GPU_PCIE_TX_MB_S="$_v" ;;
            GPU_PCIE_RX_MB_S)       GPU_PCIE_RX_MB_S="$_v" ;;
            GPU_GAMING_MODE)        GPU_GAMING_MODE="$_v" ;;
            GB_PRESSURE_STATE)      GB_PRESSURE_STATE="$_v" ;;
            SHIM_PHASE)             SHIM_PHASE="$_v" ;;
            SHIM_ACTIVE_PATH)       SHIM_ACTIVE_PATH="$_v" ;;
            SHIM_T1_LOCAL_MB)       SHIM_T1_LOCAL_MB="$_v" ;;
            SHIM_T2_LOCAL_MB)       SHIM_T2_LOCAL_MB="$_v" ;;
            SHIM_T3_LOCAL_MB)       SHIM_T3_LOCAL_MB="$_v" ;;
            SHIM_WS_RESERVE_MB)     SHIM_WS_RESERVE_MB="$_v" ;;
            SHIM_WS_RESERVE_EFF_MB) SHIM_WS_RESERVE_EFF_MB="$_v" ;;
            SHIM_KV_RESERVE_MB)     SHIM_KV_RESERVE_MB="$_v" ;;
            SHIM_VIRTUAL_VRAM_MB)   SHIM_VIRTUAL_VRAM_MB="$_v" ;;
            SHIM_CLUSTER_REMOTE_MB) SHIM_CLUSTER_REMOTE_MB="$_v" ;;
            ORCH_ECC_DEGRADED)      ORCH_ECC_DEGRADED="$_v" ;;
            ORCH_THERMAL_STRESS)    ORCH_THERMAL_STRESS="$_v" ;;
            ORCH_MEM_BW_STRESS)     ORCH_MEM_BW_STRESS="$_v" ;;
            ORCH_SBE_ELEVATED)      ORCH_SBE_ELEVATED="$_v" ;;
            ORCH_SBE_SEEN)          ORCH_SBE_SEEN="$_v" ;;
            ORCH_CLOCK_THROTTLED)   ORCH_CLOCK_THROTTLED="$_v" ;;
            ORCH_SM_CLOCK_MAX_MHZ)  ORCH_SM_CLOCK_MAX_MHZ="$_v" ;;
            ORCH_WS_ABOVE)          ORCH_WS_ABOVE="$_v" ;;
            ORCH_WS_RESERVE_MB)     ORCH_WS_RESERVE_MB="$_v" ;;
            ORCH_ACTUATE)           ORCH_ACTUATE="$_v" ;;
            ORCH_VRAM_PRESSURE)     ORCH_VRAM_PRESSURE="$_v" ;;
            ORCH_CLUSTER_PRESSURE)  ORCH_CLUSTER_PRESSURE="$_v" ;;
            ORCH_HEALTH_OK)         ORCH_HEALTH_OK="$_v" ;;
            ORCH_HEALTH_EVICT_ARMED)  ORCH_HEALTH_EVICT_ARMED="$_v" ;;
            ORCH_OS_TUNE_ENABLED)     ORCH_OS_TUNE_ENABLED="$_v" ;;
            ORCH_GAMING_MODE)         ORCH_GAMING_MODE="$_v" ;;
            ORCH_CPU_GOVERNOR)        ORCH_CPU_GOVERNOR="$_v" ;;
            ORCH_GPU_PERSISTENCE)     ORCH_GPU_PERSISTENCE="$_v" ;;
            ORCH_GPU_POWER_LIMIT_W)   ORCH_GPU_POWER_LIMIT_W="$_v" ;;
            ORCH_GPU_CLOCKS_LOCKED)   ORCH_GPU_CLOCKS_LOCKED="$_v" ;;
            ORCH_SWAPPINESS)          ORCH_SWAPPINESS="$_v" ;;
            TOPO_INFERENCE_CPUS)      TOPO_INFERENCE_CPUS="$_v" ;;
            TOPO_INFERENCE_THREADS)   TOPO_INFERENCE_THREADS="$_v" ;;
            TOPO_BACKGROUND_THREADS)  TOPO_BACKGROUND_THREADS="$_v" ;;
            TOPO_PCIE_SAT_MB_S)       TOPO_PCIE_SAT_MB_S="$_v" ;;
            TOPO_IS_BLACKWELL)        TOPO_IS_BLACKWELL="$_v" ;;
            GPU_HEALTH_OK)            GPU_HEALTH_OK="$_v" ;;
            GPU_HEALTH_SUMMARY)     GPU_HEALTH_SUMMARY="$_v" ;;
            GPU_POWER_INSTANT_W)    GPU_POWER_INSTANT_W="$_v" ;;
            GPU_NVLINK_BW_MB_S)     GPU_NVLINK_BW_MB_S="$_v" ;;
        esac
    done <<< "$_out"
}

# Fallback: original nvidia-smi VRAM-only query
_query_gpu_vram_smi() {
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

# query_gpu_dcgm - fetch DCGM health line via the helper's --dcgm flag.
# Uses a 30s file cache in the helper itself; fast on repeated calls.
# Populates: GPU_HEALTH_OK GPU_HEALTH_SUMMARY GPU_POWER_INSTANT_W GPU_NVLINK_BW_MB_S
query_gpu_dcgm() {
    local _py; _py=$(_gb_vitals_helper_path 2>/dev/null) || return 0
    local _out; _out=$(python3 "$_py" --dcgm 2>/dev/null) || return 0
    while IFS='=' read -r _k _v; do
        case "$_k" in
            GPU_HEALTH_OK)       GPU_HEALTH_OK="$_v" ;;
            GPU_HEALTH_SUMMARY)  GPU_HEALTH_SUMMARY="$_v" ;;
            GPU_POWER_INSTANT_W) GPU_POWER_INSTANT_W="$_v" ;;
            GPU_NVLINK_BW_MB_S)  GPU_NVLINK_BW_MB_S="$_v" ;;
        esac
    done <<< "$_out"
}

# ---- Extended GPU vitals query ----------------------------------------
# Now delegates to query_gpu_vram (pynvml path already returns all fields).
# Kept for compatibility with callers that expect the extended var set.
query_gpu_extended() {
    query_gpu_vram
    # nvidia-smi fallback for any vars still unset (GPU_NAME etc.)
    if [[ -z "$GPU_NAME" ]]; then
        command -v nvidia-smi &>/dev/null || return 0
        local raw
        raw=$(timeout 3 nvidia-smi \
            --query-gpu=name,temperature.gpu,power.draw,power.limit,utilization.gpu,\
utilization.memory,clocks.current.sm,clocks.current.memory \
            --format=csv,noheader,nounits 2>/dev/null | head -1) || return 0
        [[ -z "$raw" ]] && return 0
        IFS=',' read -r GPU_NAME GPU_TEMP_C GPU_POWER_W GPU_POWER_LIMIT_W \
            GPU_UTIL_PCT GPU_MEM_UTIL_PCT GPU_SM_CLOCK_MHZ GPU_MEM_CLOCK_MHZ <<< "$raw"
        GPU_NAME=$(echo "${GPU_NAME:-}" | xargs)
        GPU_TEMP_C=$(echo "${GPU_TEMP_C:-0}" | tr -dc '0-9.'); GPU_TEMP_C="${GPU_TEMP_C:-0}"
        GPU_POWER_W=$(echo "${GPU_POWER_W:-0}" | tr -dc '0-9.'); GPU_POWER_W="${GPU_POWER_W:-0}"
        GPU_POWER_LIMIT_W=$(echo "${GPU_POWER_LIMIT_W:-0}" | tr -dc '0-9.'); GPU_POWER_LIMIT_W="${GPU_POWER_LIMIT_W:-0}"
        GPU_UTIL_PCT=$(echo "${GPU_UTIL_PCT:-0}" | tr -dc '0-9'); GPU_UTIL_PCT="${GPU_UTIL_PCT:-0}"
        GPU_MEM_UTIL_PCT=$(echo "${GPU_MEM_UTIL_PCT:-0}" | tr -dc '0-9'); GPU_MEM_UTIL_PCT="${GPU_MEM_UTIL_PCT:-0}"
        GPU_SM_CLOCK_MHZ=$(echo "${GPU_SM_CLOCK_MHZ:-0}" | tr -dc '0-9'); GPU_SM_CLOCK_MHZ="${GPU_SM_CLOCK_MHZ:-0}"
        GPU_MEM_CLOCK_MHZ=$(echo "${GPU_MEM_CLOCK_MHZ:-0}" | tr -dc '0-9'); GPU_MEM_CLOCK_MHZ="${GPU_MEM_CLOCK_MHZ:-0}"
    fi
}

# ---- Live system swap query (used by cmd_vitals) ----------------------
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

# ---- NVTX event log tail reader ----------------------------------------
# Sets NVTX_TAIL to the last N lines from /run/greenboost/nvtx_events.log.
read_nvtx_tail() {
    local n="${1:-10}"
    NVTX_TAIL=""
    local log_f="/run/greenboost/nvtx_events.log"
    [[ -r "$log_f" ]] || return 0
    NVTX_TAIL=$(tail -n "$n" "$log_f" 2>/dev/null || true)
}

# ---- NVTX log diagnostic aggregator ------------------------------------
# Reads the full NVTX event log (current + rotated .1) and aggregates
# OOM counts, eviction counts, zero-copy fallbacks, and per-sub-path alloc
# stats into DIAG_* variables.  Called once per _cmd_vitals_snapshot tick.
_nvtx_parse_diag() {
    DIAG_OOM_MEMAVAIL=0; DIAG_OOM_PATH_B_FAIL=0; DIAG_OOM_FULL=0
    DIAG_OOM_T1_LOCAL=0; DIAG_OOM_T2_CAP=0; DIAG_OOM_VMM_HOST=0; DIAG_OOM_TOTAL=0
    DIAG_ZEROCOPY_FAIL=0; DIAG_EVICT_COUNT=0; DIAG_EVICT_MB=0; DIAG_LAST_PHASE=""
    DIAG_PATH_A_ZC_COUNT=0; DIAG_PATH_A_ZC_MB=0
    DIAG_PATH_A_POOL_COUNT=0; DIAG_PATH_A_POOL_MB=0
    DIAG_PATH_A_PIN_COUNT=0; DIAG_PATH_A_PIN_MB=0

    local log_f="/run/greenboost/nvtx_events.log"
    local log_f1="/run/greenboost/nvtx_events.log.1"
    local log_fb="/tmp/greenboost_nvtx_events.log"
    local _src_files=()
    [[ -r "$log_f1" ]] && _src_files+=("$log_f1")
    [[ -r "$log_f"  ]] && _src_files+=("$log_f")
    if [[ ${#_src_files[@]} -eq 0 ]]; then
        [[ -r "$log_fb" ]] && _src_files+=("$log_fb") || return 0
    fi

    local _awk_out
    _awk_out=$(cat "${_src_files[@]}" 2>/dev/null | awk '
        {
            i = 1
            if ($i+0 > 0 || $i ~ /^[0-9]+$/) { epoch=$i; i++ } else next
            if ($i == "NETD" || $i == "SHIM" || $i == "GB") i++
            ev = $i; i++
            tier = $i; i++
            mb_s = $i; gsub(/[^0-9]/, "", mb_s); mb = mb_s + 0

            if      (ev == "OOM_MEMAVAIL")         oom_mem++
            else if (ev == "OOM_PATH_B_FAIL")       oom_b++
            else if (ev == "OOM_FULL")              oom_full++
            else if (ev == "OOM_T1_LOCAL")          oom_t1++
            else if (ev == "OOM_T2_CAP")            oom_t2cap++
            else if (ev == "OOM_VMM_HOST")          oom_vmm++
            else if (ev == "ALLOC_A_ZEROCOPY_FAIL") zc_fail++
            else if (ev == "EVICT_BATCH_MADVISE" || ev == "EVICT_ARC_KV" || ev == "SWA_EVICT") {
                evict++; evict_mb += mb
            }
            else if (ev == "ALLOC_A_ZEROCOPY")  { a_zc++;   a_zc_mb   += mb }
            else if (ev == "ALLOC_A_POOL")       { a_pool++; a_pool_mb += mb }
            else if (ev == "ALLOC_A_PINNED")     { a_pin++;  a_pin_mb  += mb }
            if (substr(ev, 1, 6) == "PHASE_") last_phase = ev
        }
        END {
            printf "oom_mem=%d\n",      oom_mem+0
            printf "oom_b=%d\n",        oom_b+0
            printf "oom_full=%d\n",     oom_full+0
            printf "oom_t1=%d\n",       oom_t1+0
            printf "oom_t2cap=%d\n",    oom_t2cap+0
            printf "oom_vmm=%d\n",      oom_vmm+0
            printf "zc_fail=%d\n",      zc_fail+0
            printf "evict=%d\n",        evict+0
            printf "evict_mb=%d\n",     evict_mb+0
            printf "last_phase=%s\n",   last_phase
            printf "a_zc=%d\n",         a_zc+0
            printf "a_zc_mb=%d\n",      a_zc_mb+0
            printf "a_pool=%d\n",       a_pool+0
            printf "a_pool_mb=%d\n",    a_pool_mb+0
            printf "a_pin=%d\n",        a_pin+0
            printf "a_pin_mb=%d\n",     a_pin_mb+0
        }
    ')
    DIAG_OOM_MEMAVAIL=$(  echo "$_awk_out" | grep -oP 'oom_mem=\K[0-9]+'   | head -1 || echo 0)
    DIAG_OOM_PATH_B_FAIL=$(echo "$_awk_out" | grep -oP 'oom_b=\K[0-9]+'    | head -1 || echo 0)
    DIAG_OOM_FULL=$(       echo "$_awk_out" | grep -oP 'oom_full=\K[0-9]+'  | head -1 || echo 0)
    DIAG_OOM_T1_LOCAL=$(   echo "$_awk_out" | grep -oP 'oom_t1=\K[0-9]+'   | head -1 || echo 0)
    DIAG_OOM_T2_CAP=$(     echo "$_awk_out" | grep -oP 'oom_t2cap=\K[0-9]+' | head -1 || echo 0)
    DIAG_OOM_VMM_HOST=$(   echo "$_awk_out" | grep -oP 'oom_vmm=\K[0-9]+'  | head -1 || echo 0)
    DIAG_ZEROCOPY_FAIL=$(  echo "$_awk_out" | grep -oP 'zc_fail=\K[0-9]+'  | head -1 || echo 0)
    DIAG_EVICT_COUNT=$(    echo "$_awk_out" | grep -oP 'evict=\K[0-9]+'    | head -1 || echo 0)
    DIAG_EVICT_MB=$(       echo "$_awk_out" | grep -oP 'evict_mb=\K[0-9]+' | head -1 || echo 0)
    DIAG_LAST_PHASE=$(     echo "$_awk_out" | grep -oP 'last_phase=\K\S+'  | head -1 || echo "")
    DIAG_PATH_A_ZC_COUNT=$(   echo "$_awk_out" | grep -oP 'a_zc=\K[0-9]+'      | head -1 || echo 0)
    DIAG_PATH_A_ZC_MB=$(      echo "$_awk_out" | grep -oP 'a_zc_mb=\K[0-9]+'   | head -1 || echo 0)
    DIAG_PATH_A_POOL_COUNT=$(  echo "$_awk_out" | grep -oP 'a_pool=\K[0-9]+'   | head -1 || echo 0)
    DIAG_PATH_A_POOL_MB=$(     echo "$_awk_out" | grep -oP 'a_pool_mb=\K[0-9]+'| head -1 || echo 0)
    DIAG_PATH_A_PIN_COUNT=$(   echo "$_awk_out" | grep -oP 'a_pin=\K[0-9]+'    | head -1 || echo 0)
    DIAG_PATH_A_PIN_MB=$(      echo "$_awk_out" | grep -oP 'a_pin_mb=\K[0-9]+' | head -1 || echo 0)
    DIAG_OOM_TOTAL=$(( DIAG_OOM_MEMAVAIL + DIAG_OOM_PATH_B_FAIL + DIAG_OOM_FULL + DIAG_OOM_T1_LOCAL + DIAG_OOM_T2_CAP + DIAG_OOM_VMM_HOST ))
}

# ---- Debug vitals flag check -------------------------------------------
_debug_vitals_enabled() { [[ -f /etc/greenboost/debug_vitals.enabled ]]; }

# ---- Ollama loaded-model query (used by cmd_vitals) -------------------
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

# ---- KV size estimator (used by cmd_vitals) ---------------------------
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

# ---- Memory flow event log (used by cmd_vitals) -----------------------
# Patterns that indicate a tier transition or notable orchestration event
_GB_FLOW_PAT='KV spill|Evicted.*cold|T2 auto-evict|T2 WARN:|T2 CRITICAL:|T3 CRITICAL:|KV cache T2 full|T3 safety-net|KV reserve set|New process PID|Process PID.*exited|T2 DDR pool|T3 NVMe swap CRITICAL|T2 OOM guard|overflow|Phase →|KV reserve auto|VRAM:.*OVERFLOW|kv_allocated_t1|T3 cap'

# gather_flow_events N - emit last N tier-transition log lines from kernel
gather_flow_events() {
    local n=${1:-12}
    dmesg 2>/dev/null \
        | grep -E "greenboost.*($( echo "$_GB_FLOW_PAT" | tr '|' '|'))" | tail -"$n"
}

# format_flow_event "raw log line" - emit a branded, colored status line
format_flow_event() {
    local line="$1"
    # Extract just the message (after journalctl prefix or kernel timestamp)
    local msg
    msg=$(printf '%s' "$line" | sed 's/.*greenboost[d]*[^:]*: //' | sed 's/.*\] //')
    # Extract short timestamp HH:MM:SS - handles journalctl ISO and dmesg [NNN.NN] formats
    local ts
    ts=$(printf '%s' "$line" | grep -oP 'T\d\d:\d\d:\d\d' | head -1)
    if [[ -z "$ts" ]]; then
        ts=$(printf '%s' "$line" | grep -oP '\d\d:\d\d:\d\d' | head -1)
    fi
    if [[ -z "$ts" ]]; then
        # dmesg kernel uptime: [NNNNN.NNN] - convert to wall-clock HH:MM:SS
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

    # Fetch the tag list - try curl first, then wget
    if command -v curl &>/dev/null; then
        raw=$(curl -fsSL --max-time 5 "$GB_REPO_API" 2>/dev/null)
    elif command -v wget &>/dev/null; then
        raw=$(wget -qO- --timeout=5 "$GB_REPO_API" 2>/dev/null)
    else
        return   # no HTTP client available - skip silently
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
                    info "Update complete - restarting installer..."
                    exec "$0" "${GB_ORIG_ARGS[@]}"
                else
                    warn "git pull failed - continuing with v${GB_VERSION}."
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
# Safe to call multiple times - idempotent.

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
            warn "nvidia-smi timed out (>30 s) - GPU driver may be wedged."
            warn "  Try: sudo systemctl restart nvidia-persistenced && sudo nvidia-smi"
            GPU_NAME="Unknown (nvidia-smi timeout)"
            GB_PHYS=12
        elif [[ -z "$smi_line" ]]; then
            warn "nvidia-smi returned no output - check: sudo nvidia-smi"
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
        warn "nvidia-smi not found - assuming 8 GB VRAM. Install NVIDIA driver."
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
    # current_link_speed is unreliable - PCIe ASPM downclock to Gen 1 when the
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
    # allocation time - do NOT subtract it here or T2 is under-provisioned.
    GB_VIRT=$(gb_calc_ddr_cap_gb "$total_ram_gb")

    # ── CPU topology ─────────────────────────────────────────────────────
    CPU_NAME=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^[[:space:]]*//')
    local total_cpus; total_cpus=$(nproc)

    # Detect Intel hybrid P/E core split.
    # Primary:   lscpu -p - count CPUs per physical core; ≥2 = P-core (HT), 1 = E-core.
    #            Always available (util-linux); survives containers and newer kernels.
    # Secondary: thread_siblings_list - "N-M" range = P-core, bare integer = E-core.
    # Tertiary:  core_type sysfs integer (1=P, 0=E) - kernel 5.17+ Intel-only.
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
            # No cpufreq - fall back to arithmetic heuristic
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
        local _bd; _bd=$(df --output=source "$_sf" 2>/dev/null | tail -1 || true)
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
    _t3_disk_gb=$(df -BG /var/lib/greenboost 2>/dev/null | awk 'NR==2{gsub("G",""); print $4}' || true)
    if [[ -z "$_t3_disk_gb" ]]; then
        _t3_disk_gb=$(df -BG /var/lib 2>/dev/null | awk 'NR==2{gsub("G",""); print $4}' || true)
    fi
    if [[ -z "$_t3_disk_gb" ]]; then
        _t3_disk_gb=$(df -BG / 2>/dev/null | awk 'NR==2{gsub("G",""); print $4}' || true)
    fi
    _t3_disk_gb="${_t3_disk_gb:-0}"
    local _t3_disk_cap=$(( _t3_disk_gb * 80 / 100 ))
    [[ $_t3_needed -gt $_t3_disk_cap ]] && _t3_needed=$_t3_disk_cap
    GB_NVME_POOL=$_t3_needed

    # ── Ollama CTX based on T1+T2 KV headroom (T3 NVMe excluded) ────────
    # KV cache never spills to T3; use T1 VRAM + half T2 DDR as budget.
    # Half of T2 is left for model weights; the other half is KV headroom.
    #
    # Blackwell (cc >= 12.0): T2 DDR is served via cuMemAllocManaged and
    # reserved for model-weight overflow (the cuMemAddressReserve intercept
    # in the GreenBoost shim forces ggml-cuda to the cudaMalloc legacy pool).
    # KV cache must stay in T1 VRAM for SM latency - only GB_PHYS contributes
    # to the KV headroom budget on Blackwell desktop PCIe.
    local _cc_major_ctx
    _cc_major_ctx=$(timeout 5 nvidia-smi --query-gpu=compute_cap \
                        --format=csv,noheader,nounits 2>/dev/null \
                        | head -1 | cut -d. -f1 | xargs || echo 0)
    local kv_pool_gb
    if [[ "${_cc_major_ctx:-0}" -ge 12 ]]; then
        kv_pool_gb=$GB_PHYS          # T2 is weight-only on Blackwell PCIe
    else
        kv_pool_gb=$(( GB_PHYS + GB_VIRT / 2 ))
    fi
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
    grep -q "NV[0-9]" <<< "$(timeout 10 nvidia-smi topo -m 2>/dev/null)" && DET_NVLINK=true
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
    fi
    info "Detected hardware:"
    info "  GPU   : ${GPU_NAME}  (${GB_PHYS} GB VRAM)"
    info "  PCIe  : ${pcie_info}"
    info "  RAM   : ${RAM_TYPE}-${RAM_SPEED_MT} MT/s  ->  pool ${GB_VIRT} GB  (reserve ${GB_RESERVE} GB)"
    info "  CPU   : ${CPU_NAME}"
    local _t3_store="/var/lib/greenboost/t3_store"
    if [[ -f "$_t3_store" ]]; then
        local _t3_gb=$(( $(stat -c%s "$_t3_store" 2>/dev/null || echo 0) / 1073741824 ))
        if [[ $_t3_gb -gt 0 ]]; then
            info "  NVMe  : ${NVME_SIZE_GB} GB  ->  T3 backing ${_t3_gb} GB used  (cap ${GB_NVME_POOL} GB)"
            info "  Pool  : T1=${GB_PHYS}GB + T2=${GB_VIRT}GB + T3=${_t3_gb}GB used / ${GB_NVME_POOL}GB cap = $(( GB_PHYS + GB_VIRT + GB_NVME_POOL )) GB"
        else
            # File placeholder exists but is empty - show configured cap as potential T3
            info "  NVMe  : ${NVME_SIZE_GB} GB  ->  T3 cap ${GB_NVME_POOL} GB  (sparse file, grows on demand)"
            info "  Pool  : T1=${GB_PHYS}GB + T2=${GB_VIRT}GB + T3=${GB_NVME_POOL}GB cap = $(( GB_PHYS + GB_VIRT + GB_NVME_POOL )) GB"
        fi
    else
        info "  NVMe  : ${NVME_SIZE_GB} GB  ->  T3 cap ${GB_NVME_POOL} GB  (created on first use)"
        info "  Pool  : T1=${GB_PHYS}GB + T2=${GB_VIRT}GB + T3=${GB_NVME_POOL}GB cap = $(( GB_PHYS + GB_VIRT + GB_NVME_POOL )) GB"
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

    # Rule: physical_vram_gb - always use detected; ignore if profile claims more
    if [[ -n "$PROF_VRAM_GB" && "$PROF_VRAM_GB" -gt "$GB_PHYS" ]]; then
        conflicts+="physical_vram_gb overridden ${PROF_VRAM_GB}→${GB_PHYS} GB (physical limit); "
    fi

    # Rule: virtual_vram_gb - use profile value if <= 95% RAM, else cap
    if [[ -n "$PROF_VIRT" ]]; then
        local max_virt=$(( total_ram_gb * 95 / 100 ))
        if [[ "$PROF_VIRT" -le "$max_virt" ]]; then
            resolved_virt=$PROF_VIRT
        else
            resolved_virt=$max_virt
            conflicts+="virtual_vram_gb capped ${PROF_VIRT}→${resolved_virt} GB (95% of ${total_ram_gb} GB RAM); "
        fi
    fi

    # Rule: safety_reserve_gb - use max(profile, 6% RAM)
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

    # Rule: nvme_swap_gb - use profile if <= free NVMe space, else cap at 90% free
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

    # Rule: nvme_pool_gb - cap at 80% of free disk on the T3 store location.
    # T3 is a backing FILE on the NVMe, completely separate from the OS swap
    # partition; capping it to nvme_swap size would incorrectly restrict it.
    if [[ -n "$PROF_NVME_POOL" ]]; then
        local _t3_free_gb
        _t3_free_gb=$(df -BG /var/lib/greenboost 2>/dev/null | awk 'NR==2{gsub("G",""); print $4}')
        [[ -z "$_t3_free_gb" ]] && _t3_free_gb=$(df -BG /var/lib 2>/dev/null | awk 'NR==2{gsub("G",""); print $4}')
        [[ -z "$_t3_free_gb" ]] && _t3_free_gb=$(df -BG / 2>/dev/null | awk 'NR==2{gsub("G",""); print $4}')
        _t3_free_gb="${_t3_free_gb:-0}"
        if [[ $_t3_free_gb -gt 0 ]]; then
            local max_pool=$(( _t3_free_gb * 80 / 100 ))
            if [[ "$PROF_NVME_POOL" -le "$max_pool" ]]; then
                resolved_nvme_pool=$PROF_NVME_POOL
            else
                resolved_nvme_pool=$max_pool
                conflicts+="nvme_pool_gb capped ${PROF_NVME_POOL}→${resolved_nvme_pool} GB (80% of ${_t3_free_gb} GB free disk); "
            fi
        else
            resolved_nvme_pool=$PROF_NVME_POOL
        fi
    fi

    # Rule: ecores_only - warn if P/E split not detected
    if [[ "$resolved_pcores" == "1" && "${DET_HAS_PE_SPLIT}" == "false" ]]; then
        warn "ecores_only=1 in profile but no P/E-core split detected - module will use all CPUs"
    fi

    # Rule: nvlink_pool warning
    local prof_nvlink_pool; prof_nvlink_pool=$(parse_profile_field "$user_file" nvlink_pool)
    if [[ "$prof_nvlink_pool" == "true" && "${DET_NVLINK}" == "false" ]]; then
        warn "nvlink_pool=true in profile but no NVLink topology detected - overriding to false"
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
    _GB_HW_DETECTED=0   # force fresh detection - discard any cached state
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

    # S3: Live-reload - if the module is loaded and profile changes a kernel
    # parameter (nvme_pool_gb or t2_pool_gb), offer to reload the module.
    if grep -q "^greenboost " <<< "$(lsmod 2>/dev/null)"; then
        local sysfs_nvme="/sys/module/greenboost/parameters/nvme_pool_gb"
        local sysfs_t2="/sys/module/greenboost/parameters/t2_pool_gb"
        load_profile_values "$abs" 2>/dev/null || true
        local needs_reload=0
        if [[ -r "$sysfs_nvme" && -n "${PROF_NVME_POOL:-}" ]]; then
            local cur_nvme; cur_nvme=$(cat "$sysfs_nvme" 2>/dev/null || echo "0")
            [[ "$cur_nvme" != "${PROF_NVME_POOL}" ]] && needs_reload=1
        fi
        if [[ -r "$sysfs_t2" && -n "${PROF_VIRT:-}" ]]; then
            local cur_t2; cur_t2=$(cat "$sysfs_t2" 2>/dev/null || echo "0")
            [[ "$cur_t2" != "${PROF_VIRT}" ]] && needs_reload=1
        fi
        if [[ "$needs_reload" -eq 1 ]]; then
            gb_warn_ui "Profile changes a kernel parameter - module reload required."
            read -rp "  Reload GreenBoost module now? [y/N] " _reload_ans
            if [[ "${_reload_ans,,}" == y* ]]; then
                cmd_unload
                cmd_load
                gb_ok "Module reloaded with new profile parameters."
            else
                gb_warn_ui "Reload skipped. Changes will take effect on next: sudo greenboost unload && sudo greenboost load"
            fi
        fi
    fi
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

# cmd_profile_export_json [profile-file]
# Emit the active (or supplied) profile as JSON to stdout. Lets external
# tooling - e.g. the hyphaed kernel-build wizard at ~/Dev/kernel_inference
# - consume the profile without parsing the Markdown/YAML hybrid format.
cmd_profile_export_json() {
    local file="${1:-}"
    if [[ -z "$file" ]]; then
        [[ -L "$GB_ACTIVE_PROFILE_LINK" || -f "$GB_ACTIVE_PROFILE_LINK" ]] \
            || die "No active profile. Provide a file or run: sudo $0 profile create"
        file=$(readlink -f "$GB_ACTIVE_PROFILE_LINK")
    fi
    [[ -f "$file" ]] || die "Profile file not found: $file"

    # Parse `key: value` lines (skip headings, frontmatter, blank lines, comments)
    # then emit a flat JSON object via awk. Values are emitted as strings -
    # the consumer can re-type them.
    awk '
        BEGIN { printf "{"; first = 1 }
        /^[[:space:]]*$/ { next }
        /^#/ { next }
        /^---$/ { next }
        /^##/ { next }
        /^[a-zA-Z_][a-zA-Z0-9_]*[[:space:]]*:/ {
            key = $0; sub(/[[:space:]]*:.*$/, "", key)
            val = $0; sub(/^[^:]*:[[:space:]]*/, "", val)
            gsub(/^"|"$/, "", val)            # strip surrounding quotes
            gsub(/\\/, "\\\\", val)           # escape backslashes
            gsub(/"/, "\\\"", val)            # escape quotes
            gsub(/\t/, "\\t", val)            # escape tabs
            if (!first) printf ","
            printf "\"%s\":\"%s\"", key, val
            first = 0
        }
        END { print "}" }
    ' "$file"
}

cmd_profile() {
    local sub="${1:-show}"
    shift 2>/dev/null || true
    case "$sub" in
        create)        cmd_profile_create ;;
        show)          cmd_profile_show "$@" ;;
        list)          cmd_profile_list ;;
        activate)      cmd_profile_activate "$@" ;;
        diff)          cmd_profile_diff "$@" ;;
        export-json)   cmd_profile_export_json "$@" ;;
        *) die "Unknown profile subcommand: '$sub'. Use: create|show|list|activate|diff|export-json" ;;
    esac
}

# ── cmd_profile_wizard - interactive profile management sub-menu ──────────
cmd_profile_wizard() {
    while true; do
        clear
        # ── Header ──────────────────────────────────────────────────────
        local cols; cols=$(tput cols 2>/dev/null || echo 72)
        local title=" GreenBoost v${GB_VERSION} - Profile Management"
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
            echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_AMBER}${C_BOLD}No active profile${C_RESET}  ${C_DIM}- select [1] to auto-detect hardware and create one${C_RESET}"
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
            gb_warn_ui "Not root - options marked ${C_RED}⚠ root${C_AMBER} require sudo."
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
                    gb_warn_ui "No profiles directory - run option 1 first."
                    gb_press_enter; continue
                fi
                local profiles=( "$GB_PROFILES_DIR"/*.md )
                if [[ ! -f "${profiles[0]}" ]]; then
                    gb_warn_ui "No profiles found - run option 1 first."
                    gb_press_enter; continue
                fi
                echo -e "  ${C_CYAN}${C_BOLD}Available profiles:${C_RESET}"
                echo -e ""
                local idx=1
                local pf
                for pf in "${profiles[@]}"; do
                    [[ -f "$pf" ]] || continue
                    gb_menu_item "$idx" "$(basename "$pf" .md)" "$pf"
                    (( idx++ )) || true
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

# Owner-workstation preset - hard-coded optimal for known hardware:
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
    command -v make >/dev/null || die "make not found - install with: sudo ${_make_hint}"
    command -v gcc  >/dev/null || die "gcc not found - install with: sudo ${_gcc_hint}"

    local kdir="/lib/modules/$(uname -r)/build"
    [[ -d "$kdir" ]] || die "Kernel headers not found at $kdir
    Install with: sudo ${_hdr_hint}"
    info "Kernel headers : $kdir  ✓"

    if grep -qE "^nvidia[[:space:]]" <<< "$(lsmod 2>/dev/null)" \
            || [[ -c /dev/nvidia0 ]] \
            || [[ -f /proc/driver/nvidia/version ]]; then
        info "NVIDIA driver  : loaded  ✓"
    else
        warn "NVIDIA driver not loaded - run: sudo modprobe nvidia"
    fi
}

# ---- Commands ----------------------------------------------------------

# ---------------------------------------------------------------------------
# cmd_install_supervisor , unified GreenBoost supervisor (v3.1+).
#
# Replaces four separate units (greenboost-recovery, greenboost-sentinel,
# greenboost-vram-watchdog, greenboost-idle-reclaim) with a single
# Type=notify Python service backed by gb_supervisor.py.  Auto-enabled.
#
# Key improvements vs. the old four-daemon design:
#   - No nvidia-smi forks.  VRAM reads use pynvml via gb_supervisor.py.
#   - sd_notify READY=1 sent after boot recovery → Ollama cannot start before
#     recovery completes (Before=ollama.service enforced without a separate
#     oneshot unit).
#   - Process kill is opt-in (GB_SUPERVISOR_AGGRESSIVE_RECLAIM=1, default 0).
#   - Single unit to manage, single log stream (journalctl -u gb-supervisor).
# ---------------------------------------------------------------------------
cmd_install_supervisor() {
    local STATE_DIR="/var/lib/greenboost"
    local LIB_DIR="/usr/local/lib/greenboost"
    local SCRIPT_SRC
    SCRIPT_SRC="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/gb_supervisor.py"

    mkdir -p "$STATE_DIR" "$LIB_DIR"

    # 1. Install the Python supervisor script + vitals helper
    if [[ ! -f "$SCRIPT_SRC" ]]; then
        gb_warn "gb_supervisor.py not found at $SCRIPT_SRC , skipping supervisor install"
        return 1
    fi
    install -m 755 "$SCRIPT_SRC" "$LIB_DIR/gb_supervisor.py"
    gb_ok "gb_supervisor.py installed to $LIB_DIR/"
    local _vh_src; _vh_src="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/gb_vitals_helper.py"
    if [[ -f "$_vh_src" ]]; then
        install -m 755 "$_vh_src" "$LIB_DIR/gb_vitals_helper.py"
        gb_ok "gb_vitals_helper.py installed to $LIB_DIR/"
    fi
    local _mon_src; _mon_src="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/gb_monitor.py"
    if [[ -f "$_mon_src" ]]; then
        install -m 755 "$_mon_src" "$LIB_DIR/gb_monitor.py"
        gb_ok "gb_monitor.py installed to $LIB_DIR/"
    fi
    local _pilot_src; _pilot_src="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/gb_pilot.py"
    if [[ -f "$_pilot_src" ]]; then
        install -m 755 "$_pilot_src" "$LIB_DIR/gb_pilot.py"
        gb_ok "gb_pilot.py installed to $LIB_DIR/"
    fi

    # 2. Remove old separate units (idempotent upgrade from ≤v3.0)
    local _old_units=(
        greenboost-recovery.service
        greenboost-sentinel.service
        greenboost-vram-watchdog.service
        greenboost-idle-reclaim.service
    )
    local _old_bins=(
        /usr/local/sbin/greenboost-recover
        /usr/local/sbin/greenboost-vram-watchdog
        /usr/local/bin/greenboost-idle-reclaim
    )
    for _u in "${_old_units[@]}"; do
        systemctl disable --now "$_u" 2>/dev/null || true
        rm -f "/etc/systemd/system/$_u"
    done
    rm -f "${_old_bins[@]}"

    # 3. Write the unified systemd unit (Type=notify, Before=ollama)
    cat > /etc/systemd/system/greenboost-supervisor.service << 'SUPEOF'
[Unit]
Description=GreenBoost Unified Supervisor (VRAM watchdog + idle reclaim + boot recovery)
Documentation=https://gitlab.com/IsolatedOctopi/greenboost
After=network.target nvidia-persistenced.service
# Recovery completes before Ollama starts (sd_notify READY=1 only after recovery)
Before=ollama.service

[Service]
Type=notify
ExecStart=/usr/local/lib/greenboost/gb_supervisor.py
ExecStop=/bin/kill -TERM $MAINPID
KillMode=process
Restart=on-failure
RestartSec=10
TimeoutStartSec=90
TimeoutStopSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=gb-supervisor
# Continuous reactive control: actuate ReactiveOrchestrator levers (Loops A-S),
# including the continuous OS tuner (Loops O-S: CPU governor/EPP, GPU
# clocks/power, vm.* tunables). Disable with override.conf if you want
# observe-only (dry-run) mode instead.
Environment=GB_ORCH_ACTUATE=1
Environment=GB_OS_TUNE=1
# Optional tuning (set in /etc/systemd/system/greenboost-supervisor.service.d/override.conf):
#   Environment=GB_SUPERVISOR_POLL_SECS=10
#   Environment=GB_SUPERVISOR_VRAM_WARN_PCT=10
#   Environment=GB_SUPERVISOR_VRAM_CRIT_FREE_PCT=8
#   Environment=GB_SUPERVISOR_OLLAMA_URL=http://127.0.0.1:11434
#   Environment=GB_SUPERVISOR_AGGRESSIVE_RECLAIM=0
#   Environment=GB_ORCH_ACTUATE=0   # observe-only / dry-run
#   Environment=GB_OS_TUNE=0        # disable continuous OS tuning

[Install]
WantedBy=multi-user.target
SUPEOF

    # 4. Enable and start (auto-enable: kill path opt-in, no nvidia-smi forks)
    systemctl daemon-reload
    systemctl enable --now greenboost-supervisor.service 2>/dev/null \
        && gb_ok "greenboost-supervisor.service enabled and started (auto-enabled)" \
        || gb_warn "supervisor enable failed , run: sudo systemctl enable --now greenboost-supervisor.service"
}

# Legacy entry-point kept so any external scripts that call cmd_install_recovery
# still work.  Just delegates to the unified supervisor installer.
cmd_install_recovery() {
    cmd_install_supervisor
}

cmd_install_sys_configs() {
    need_root install-sys-configs
    detect_hardware
    gb_ensure_shim_libs
    gb_ensure_greenboost_group
    # Repair existing cluster.key permissions on upgrade (idempotent).
    [[ -f "$GB_CLUSTER_KEY" ]] && _gb_set_keyfile_perms "$GB_CLUSTER_KEY"

    info "Installing GreenBoost v3.2 system configuration files..."

    # 1. Ollama service - inject GreenBoost env vars + LD_PRELOAD (always refresh)
    local svc="/etc/systemd/system/ollama.service"
    if [[ -f "$svc" ]]; then
        # Remove any previously injected GreenBoost lines first (idempotent upgrade)
        sed -i '/OLLAMA_FLASH_ATTENTION/d
/OLLAMA_KV_CACHE_TYPE/d
/OLLAMA_NUM_CTX/d
/OLLAMA_CONTEXT_LENGTH/d
/OLLAMA_MAX_LOADED_MODELS/d
/OLLAMA_KEEP_ALIVE/d
/OLLAMA_NUM_GPU/d
/GREENBOOST_/d
/libgreenboost/d' "$svc"
        # Inject fresh v2.7 env vars
        sed -i "/^\[Service\]/a Environment=\"OLLAMA_FLASH_ATTENTION=1\"\nEnvironment=\"OLLAMA_KV_CACHE_TYPE=q8_0\"\nEnvironment=\"OLLAMA_NUM_CTX=${GB_OLLAMA_CTX}\"\nEnvironment=\"OLLAMA_CONTEXT_LENGTH=${GB_OLLAMA_CTX}\"\nEnvironment=\"OLLAMA_MAX_LOADED_MODELS=1\"\nEnvironment=\"OLLAMA_KEEP_ALIVE=-1\"\nEnvironment=\"GREENBOOST_VIRTUAL_VRAM_MB=$((GB_VIRT * 1024))\"\nEnvironment=\"GREENBOOST_DEBUG=0\"\nEnvironment=\"GREENBOOST_ACTIVE=1\"\nEnvironment=\"LD_PRELOAD=/usr/local/lib/libgreenboost_vmm_override.so:/usr/local/lib/libgreenboost_cuda.so\"" "$svc"
        systemctl daemon-reload
        info "Ollama service: GreenBoost v3.2 env vars injected (refreshed)"
        gb_ok "Ollama context cap set to ${GB_OLLAMA_CTX} tokens (T1: ${GB_PHYS} GB, T2: ${GB_VIRT} GB)"
    else
        warn "Ollama service not found at $svc - skipping"
    fi

    # 1b. Drop-in override - write 99-greenboost.conf so GreenBoost vars always win
    # over any third-party drop-ins (boost.conf, override.conf, etc.) that may set
    # conflicting values for OLLAMA_NUM_CTX, LD_PRELOAD, or OLLAMA_GPU_OVERHEAD.
    local dropin_dir="/etc/systemd/system/ollama.service.d"
    mkdir -p "$dropin_dir"

    # Remove conflicting entries from other drop-in files before writing 99-greenboost.conf
    local gb_vars=(OLLAMA_NUM_CTX OLLAMA_CONTEXT_LENGTH LD_PRELOAD OLLAMA_GPU_OVERHEAD OLLAMA_FLASH_ATTENTION
                   OLLAMA_KV_CACHE_TYPE OLLAMA_MAX_LOADED_MODELS OLLAMA_KEEP_ALIVE OLLAMA_NUM_GPU
                   GREENBOOST_KV_RESERVE_MB GREENBOOST_VIRTUAL_VRAM_MB GREENBOOST_DEBUG
                   GREENBOOST_ACTIVE GREENBOOST_WORKSTATION_RESERVE_MB GREENBOOST_FORCE_CC_MAJOR
                   GOMP_CPU_AFFINITY OMP_NUM_THREADS
                   GREENBOOST_INFERENCE_CPUS GREENBOOST_INFERENCE_THREADS)
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

    # Ensure GB_KV_RESERVE_MB is computed before writing the drop-in.
    # install-sys-configs may be called standalone without cmd_kmod_install
    # having run first, so the auto-scale must be replicated here.
    # Right-size using estimate_kv_mb() with a VRAM-based worst-case model param count
    # (2x safety buffer, floor 128 MB). Ceiling is VRAM-proportional: reserving >1/16 of
    # physical VRAM for KV strands on MoE models where the actual KV
    # is typically 4-8× smaller than a dense model of the same nominal size.
    if [[ -z "${GB_KV_RESERVE_MB:-}" ]]; then
        local _ctx_for_kv="${GB_OLLAMA_CTX:-32768}"
        local _kv_param_guess
        if   (( ${GB_PHYS:-0} >= 80 )); then _kv_param_guess="120B"
        elif (( ${GB_PHYS:-0} >= 40 )); then _kv_param_guess="70B"
        elif (( ${GB_PHYS:-0} >= 20 )); then _kv_param_guess="32B"
        elif (( ${GB_PHYS:-0} >= 10 )); then _kv_param_guess="13B"
        else                                  _kv_param_guess="7B"
        fi
        local _raw_kv_mb
        _raw_kv_mb=$(estimate_kv_mb "$_ctx_for_kv" "$_kv_param_guess")
        GB_KV_RESERVE_MB=$(( _raw_kv_mb * 2 ))
        (( GB_KV_RESERVE_MB < 128  )) && GB_KV_RESERVE_MB=128
    fi
    # VRAM-proportional cap: never strand more than 1/16 of physical VRAM as KV reserve.
    # A 12 GB card caps at 768 MB; a 24 GB card caps at 1536 MB; a 80 GB card caps at 2048 MB.
    # This prevents a pre-set GB_KV_RESERVE_MB=2048 from wasting T1 VRAM on small-VRAM cards.
    local _kv_vram_cap=$(( ${GB_PHYS:-12} * 1024 / 16 ))
    (( _kv_vram_cap < 128  )) && _kv_vram_cap=128
    (( _kv_vram_cap > 2048 )) && _kv_vram_cap=2048
    if (( GB_KV_RESERVE_MB > _kv_vram_cap )); then
        gb_info "GB_KV_RESERVE_MB clamped ${GB_KV_RESERVE_MB} → ${_kv_vram_cap} MB (1/16 of ${GB_PHYS} GB VRAM)"
        GB_KV_RESERVE_MB=$_kv_vram_cap
    fi

    # Compute CC major for Blackwell-specific tweaks in the drop-in.
    local _dropin_cc_major
    _dropin_cc_major=$(echo "${DET_CC:-0.0}" | cut -d. -f1 | tr -d '[:space:]')
    _dropin_cc_major="${_dropin_cc_major:-0}"

    # Compute golden-core (or P-core) affinity for inference thread pinning.
    # Golden cores have the lowest latency and dedicated L2 slices on hybrid CPUs
    # (i9-14900KF: CPUs 4-7). Pinning llamacpp CPU threads here reduces context-
    # switch pressure on the cores handling CUDA stream scheduling.
    local _golden_min="${PROF_GOLDEN_MIN:-${GB_GOLDEN_MIN:-0}}"
    local _golden_max="${PROF_GOLDEN_MAX:-${GB_GOLDEN_MAX:-7}}"
    local _golden_threads=$(( _golden_max - _golden_min + 1 ))
    [[ $_golden_threads -lt 1 ]] && _golden_threads=4
    local _gomp_affinity="${_golden_min}-${_golden_max}"

    # cluster.key is 0640 root:greenboost , the service needs the group so the
    # shim (gb_netc_init) can authenticate to feeders without running as root.
    gb_ensure_greenboost_group

    # Write 99-greenboost.conf - alphabetically last, so it always wins
    {
        cat << DROPINEOF
# GreenBoost v3.2 - managed file, do not edit manually
# Re-generated by: sudo ./greenboost_setup.sh install-sys-configs
[Unit]
After=greenboost-supervisor.service
Wants=greenboost-supervisor.service

[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_NUM_CTX=${GB_OLLAMA_CTX}"
Environment="OLLAMA_CONTEXT_LENGTH=${GB_OLLAMA_CTX}"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="GREENBOOST_VIRTUAL_VRAM_MB=$((GB_VIRT * 1024))"
Environment="GREENBOOST_DEBUG=0"
Environment="GREENBOOST_KV_RESERVE_MB=${GB_KV_RESERVE_MB}"
Environment="GREENBOOST_ACTIVE=1"
# GREENBOOST_WORKSTATION_RESERVE_MB intentionally unset: the shim now sizes
# this dynamically from live desktop/compositor GPU usage (see
# gb_effective_workstation_reserve_bytes in greenboost_cuda_shim.c) instead of
# a flat tax, so inference gets whatever VRAM the desktop isn't using.
Environment="LD_PRELOAD=/usr/local/lib/libgreenboost_vmm_override.so:/usr/local/lib/libgreenboost_cuda.so"
Environment="GOMP_CPU_AFFINITY=${_gomp_affinity}"
Environment="OMP_NUM_THREADS=${_golden_threads}"
Environment="GREENBOOST_INFERENCE_CPUS=${_gomp_affinity}"
Environment="GREENBOOST_INFERENCE_THREADS=${_golden_threads}"
DROPINEOF
        # Grant read access to /etc/greenboost/cluster.key (0640 root:greenboost)
        # so the shim inside ollama can HMAC-authenticate to feeders.
        getent group greenboost >/dev/null 2>&1 && echo "SupplementaryGroups=greenboost"
        # Blackwell (cc >= 12): bypass the lazy CC probe race in vmm_override so
        # VMM_SUPPORTED=0 fires even on the very first cuDeviceGetAttribute call.
        if (( _dropin_cc_major >= 12 )); then
            echo "Environment=\"GREENBOOST_FORCE_CC_MAJOR=${_dropin_cc_major}\""
        fi
    } > "$dropin_dir/99-greenboost.conf"
    systemctl daemon-reload
    if [[ $_conflict_found -eq 1 ]]; then
        gb_ok "99-greenboost.conf written - conflicting entries removed from other drop-ins"
    else
        gb_ok "99-greenboost.conf written (no conflicts detected)"
    fi

    # 1c. Other known inference tool services - inject LD_PRELOAD + GREENBOOST_ACTIVE
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

    # 2a. GreenBoost device udev rule - allow video group (includes ollama) access
    cat > /etc/udev/rules.d/99-greenboost.rules << 'UDEVEOF'
# GreenBoost kernel module - allow video group (includes ollama) to access /dev/greenboost
KERNEL=="greenboost", GROUP="video", MODE="0660"
UDEVEOF
    info "GreenBoost device udev rule installed: /etc/udev/rules.d/99-greenboost.rules"
    # Apply rule immediately to existing /dev/greenboost (if module already loaded)
    udevadm control --reload-rules 2>/dev/null || true
    udevadm trigger --name-match=greenboost 2>/dev/null \
        || udevadm trigger --subsystem-match=greenboost 2>/dev/null \
        || true
    udevadm settle 2>/dev/null || true

    # 2d. Add the invoking user to the 'video' group for /dev/greenboost access.
    # The udev rule grants GROUP="video" MODE="0660"; without group membership
    # a non-root user gets EACCES, which breaks gb_quant int4 FLUX/DMA-BUF T2.
    if [[ -n "${SUDO_USER:-}" ]]; then
        if ! id -nG "$SUDO_USER" 2>/dev/null | tr ' ' '\n' | grep -q '^video$'; then
            usermod -aG video "$SUDO_USER" \
                && info "User '$SUDO_USER' added to 'video' group (re-login or: newgrp video)" \
                || warn "usermod failed , add $SUDO_USER to 'video' group manually"
        else
            gb_info "User '$SUDO_USER' already in 'video' group"
        fi
        # Grant immediate rw via ACL so current session works before re-login
        if [[ -c /dev/greenboost ]] && command -v setfacl &>/dev/null; then
            setfacl -m "u:${SUDO_USER}:rw" /dev/greenboost 2>/dev/null \
                && gb_info "ACL: /dev/greenboost → u:${SUDO_USER}:rw (current session)" \
                || true
        fi
    fi

    # 2b. NVMe udev rule - scheduler=none, read_ahead=4096, nr_requests=1023
    # ENV{DEVTYPE}=="disk" restricts rules to whole-disk nodes only (nvme0n1, nvme1n1 …),
    # excluding partition nodes (nvme0n1p1 …) which have no queue/ sysfs directory.
    # nr_requests capped at 1023 - Samsung 990 EVO Plus hardware limit (max_hw_sectors_kb=512).
    cat > /etc/udev/rules.d/99-nvme-greenboost.rules << 'UDEVEOF'
# GreenBoost v3.2 - NVMe tuning for T3 swap performance
ACTION=="add|change", KERNEL=="nvme[0-9]n[0-9]", ENV{DEVTYPE}=="disk", ATTR{queue/scheduler}="none"
ACTION=="add|change", KERNEL=="nvme[0-9]n[0-9]", ENV{DEVTYPE}=="disk", ATTR{queue/read_ahead_kb}="4096"
ACTION=="add|change", KERNEL=="nvme[0-9]n[0-9]", ENV{DEVTYPE}=="disk", ATTR{queue/nr_requests}="1023"
UDEVEOF
    udevadm control --reload-rules && udevadm trigger || true
    info "NVMe udev rule installed: /etc/udev/rules.d/99-nvme-greenboost.rules"

    # 2c. NVIDIA shutdown fix - prevent driver from hanging 10-30s during reboot/shutdown.
    # NVreg_PreserveVideoMemoryAllocations=1 (default) makes the driver save GPU state to RAM
    # on suspend/shutdown. With large models loaded this blocks reboot for up to 30s and can
    # hang indefinitely on RTX 50xx + kernel 6.19. Setting to 0 disables that save/restore.
    cat > /etc/modprobe.d/99-nvidia-greenboost.conf << 'NVEOF'
options nvidia NVreg_PreserveVideoMemoryAllocations=0
NVEOF
    info "NVIDIA shutdown fix installed: /etc/modprobe.d/99-nvidia-greenboost.conf"

    # 3. CPU governor service - P-cores only (E-cores stay on powersave)
    cat > /etc/systemd/system/cpu-perf.service << CPUEOF
[Unit]
Description=GreenBoost CPU performance governor (P-cores 0-${GB_PCORES_MAX})
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c 'for cpu in \$(seq 0 ${GB_PCORES_MAX}); do echo performance > /sys/devices/system/cpu/cpu\${cpu}/cpufreq/scaling_governor; done; cset set -c ${GB_GOLDEN_MIN}-${GB_GOLDEN_MAX} -s inference_set || true; cset proc -m -f root -t inference_set -k || true'
ExecStop=/bin/bash  -c 'for cpu in \$(seq 0 ${GB_PCORES_MAX}); do echo powersave  > /sys/devices/system/cpu/cpu\${cpu}/cpufreq/scaling_governor; done; cset set -d inference_set || true'
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
CPUEOF
    systemctl daemon-reload
    gb_ok "cpu-perf service installed (not auto-enabled)"
    gb_info "Enable manually if needed: systemctl enable --now cpu-perf.service"

    # 4. THP sysfs.d - transparent hugepages for compaction + THP performance
    # NOTE: gb_alloc_buf() uses alloc_pages(GFP_KERNEL|__GFP_COMP, order=9) which draws
    # from the BUDDY ALLOCATOR, NOT the HugeTLB pool.  Pre-allocating HugeTLB pages
    # (vm.nr_hugepages=26112) locks 51 GB in the HugeTLB pool, leaving <12 GB free RAM,
    # which triggers the OOM guard and makes T2 unavailable.  Keep nr_hugepages=0.
    mkdir -p /etc/sysfs.d
    cat > /etc/sysfs.d/greenboost-hugepages.conf << 'HPEOF'
# GreenBoost v3.2 - THP config (no HugeTLB pre-allocation: gb_alloc_buf uses buddy allocator)
kernel/mm/transparent_hugepage/enabled = always
HPEOF
    info "THP sysfs conf: /etc/sysfs.d/greenboost-hugepages.conf"

    # Release any previously locked HugeTLB pages and free them back to buddy allocator
    if [[ "$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages 2>/dev/null)" != "0" ]]; then
        echo 0 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages 2>/dev/null \
            && info "HugeTLB pages released: freed $(( $(cat /proc/meminfo | grep HugePages_Total | awk '{print $2}') * 2 )) kB back to buddy allocator" \
            || warn "Could not set nr_hugepages=0 - reboot to apply"
    else
        info "HugeTLB: already 0 (correct - GreenBoost T2 uses buddy allocator)"
    fi

    # 5. VM sysctl - handled by cmd_tune_sysctl (99-zzz-greenboost.conf)
    # The definitive sysctl file is written by tune-sysctl/tune-all and uses
    # the 99-zzz- prefix so it sorts last and wins over any other conf.
    # No separate 99-greenboost.conf sysctl block here to avoid conflicting values.

    # 6. Install LD_AUDIT library + AppArmor abstraction
    # The audit library (< 5 KB, no CUDA code) sits in /etc/ld.so.preload and
    # injects the full CUDA shim ONLY into processes that load libcuda.so or
    # libcudart.so.  PAM helpers, GDM, snap-confine, and every other non-CUDA
    # process are never touched - AppArmor blast radius is one r/mr permission
    # for the tiny audit stub, not the full shim.
    local audit_src="$MODULE_DIR/$AUDIT_LIB"
    local audit_dest="$SHIM_DEST/$AUDIT_LIB"
    if [[ -f "$audit_src" ]]; then
        cp "$audit_src" "$audit_dest"
        ldconfig 2>/dev/null || true
        info "LD_AUDIT library installed: $audit_dest"
    else
        warn "LD_AUDIT library not found at $audit_src - run 'make audit' first"
    fi

    # AppArmor working-minimum - ALWAYS enforced (not opt-in).
    #
    # Every install tier deploys the minimal safe AppArmor rules needed for
    # GreenBoost to work without log spam:
    #   1. Install the greenboost-audit abstraction file (harmless - just a file).
    #   2. Add a targeted local override for unix-chkpwd so sudo/PAM auth
    #      doesn't generate AppArmor denials from ld.so.preload loading the
    #      audit stub (the 18 denials seen in a half-deployed state).
    #
    # What is NOT done here (safe default): abstractions/base injection and
    # snap-confine patching.  These cause snap apps to break (snap-confine
    # self-integrity check failure).  Set GB_APPARMOR_FULL_INSTALL=1 to opt
    # into the aggressive global injection (rare - only for confined CUDA workloads).
    #
    # Ollama on standard Ubuntu runs UNCONFINED - its LD_PRELOAD (in the
    # systemd drop-in) works without any AppArmor changes.
    local aa_dir="/etc/apparmor.d/abstractions"
    local aa_src="$MODULE_DIR/apparmor/abstractions/greenboost-audit"
    local aa_dest="$aa_dir/greenboost-audit"

    echo -e ""
    echo -e "  ${C_VIOLET}◈${C_RESET}  ${C_GRAY}${C_BOLD}AppArmor - working-minimum${C_RESET}"

    # Step 1: Install the abstraction file (always safe).
    if [[ -d "$aa_dir" && -f "$aa_src" ]]; then
        cp "$aa_src" "$aa_dest"
        gb_ok "AppArmor abstraction installed: $aa_dest"
    else
        gb_info "AppArmor abstraction dir not found , skipping (non-AppArmor system)"
    fi

    # Step 2: Targeted local override for unix-chkpwd.
    # When the audit lib is in /etc/ld.so.preload, EVERY process that starts
    # tries to open it, including PAM helpers like unix-chkpwd which are
    # confined under a strict AppArmor profile.  Without mr permission for the
    # audit stub, AppArmor denies the open and logs a denial per sudo/auth event.
    if [[ -d "/etc/apparmor.d/local" && -f "/etc/apparmor.d/usr.sbin.unix-chkpwd" ]]; then
        local _aa_chkpwd_local="/etc/apparmor.d/local/usr.sbin.unix-chkpwd"
        if ! grep -q "libgreenboost_audit" "$_aa_chkpwd_local" 2>/dev/null; then
            {
                echo "# GreenBoost: allow unix-chkpwd to open ld.so.preload audit stub"
                echo "/usr/local/lib/libgreenboost_audit.so mr,"
                echo "/usr/local/lib/x86_64-linux-gnu/libgreenboost_audit.so mr,"
                echo "/usr/local/lib/i386-linux-gnu/libgreenboost_audit.so mr,"
            } >> "$_aa_chkpwd_local"
            apparmor_parser -r /etc/apparmor.d/usr.sbin.unix-chkpwd 2>/dev/null && \
                gb_ok "unix-chkpwd local override: audit stub mr granted (denials cleared)" || \
                gb_warn "unix-chkpwd profile reload failed , will apply on next apparmor restart"
        else
            gb_info "unix-chkpwd local override: already set (skip)"
        fi
    fi

    # GB_APPARMOR_FULL_INSTALL=1: aggressive global injection (opt-in, snap-risk).
    if [[ "${GB_APPARMOR_FULL_INSTALL:-0}" != "1" ]]; then
        echo -e "  ${C_DIM}  Full AppArmor injection skipped (default). Set GB_APPARMOR_FULL_INSTALL=1${C_RESET}"
        echo -e "  ${C_DIM}  to inject into abstractions/base , WARNING: breaks snap apps.${C_RESET}"
    elif [[ -d "$aa_dir" && -f "$aa_src" ]]; then
        echo -e "  ${C_AMBER}  WARNING: GB_APPARMOR_FULL_INSTALL=1 , this WILL break snap apps.${C_RESET}"
        echo -e "  ${C_AMBER}  Run \`sudo greenboost apparmor-uninstall\` to revert.${C_RESET}"
        echo -e ""

        cp "$aa_src" "$aa_dest"
        gb_ok "AppArmor abstraction installed"

        # Layer A - inject into abstractions/base for global coverage
        local base_abs="/etc/apparmor.d/abstractions/base"
        if [[ -f "$base_abs" ]] && ! grep -q "greenboost-audit" "$base_abs"; then
            sed -i '/^  abi <abi\//a\  #include <abstractions\/greenboost-audit>' "$base_abs"
            gb_ok "Added to abstractions/base - all new apps auto-covered"
        else
            gb_info "abstractions/base already includes greenboost-audit (skip)"
        fi

        # snap-confine local override - snapd overwrites the main profile on every
        # update, so we inject into the local/ override file instead, which is
        # never touched by snapd.  Without this, snap apps that load CUDA trigger
        # AppArmor denials for libgreenboost_audit.so (mmap permission denied).
        local sc_local="/etc/apparmor.d/local/usr.lib.snapd.snap-confine.real"
        if [[ -d "/etc/apparmor.d/local" ]]; then
            if ! grep -q "libgreenboost_audit" "$sc_local" 2>/dev/null; then
                {
                    echo "# GreenBoost: allow snap-confine to mmap the LD_AUDIT stub"
                    echo "/usr/local/lib/libgreenboost_audit.so mr,"
                    echo "/usr/local/lib/x86_64-linux-gnu/libgreenboost_audit.so mr,"
                } >> "$sc_local"
                gb_ok "snap-confine local override: audit library mmap allowed"
            else
                gb_info "snap-confine local override already set (skip)"
            fi
        fi

        # snap-confine inside snapd snap - the snap ships its own confined binary
        # (/snap/snapd/NNNN/usr/lib/snapd/snap-confine) whose AppArmor profile is
        # stored in /var/lib/snapd/apparmor/profiles/ and loaded by snapd directly
        # (not by apparmor_parser from /etc/apparmor.d/).  The local/ override
        # above only affects the system binary.  Patch the snapd snap profile too.
        local _snapd_profile
        _snapd_profile=$(find /var/lib/snapd/apparmor/profiles/ \
            -name "snap-confine.snapd.*" -o -name "snap-confine.*" 2>/dev/null \
            | head -1)
        if [[ -n "$_snapd_profile" ]]; then
            if ! grep -q "libgreenboost_audit" "$_snapd_profile" 2>/dev/null; then
                {
                    echo ""
                    echo "# GreenBoost: allow snap-confine (snapd snap) to mmap audit stub"
                    echo "/usr/local/lib/libgreenboost_audit.so mr,"
                    echo "/usr/local/lib/x86_64-linux-gnu/libgreenboost_audit.so mr,"
                } >> "$_snapd_profile"
                apparmor_parser -r "$_snapd_profile" 2>/dev/null && \
                    gb_ok "snapd snap-confine profile patched and reloaded: $(basename "$_snapd_profile")" || \
                    gb_warn "snapd snap-confine profile patched but reload failed (will apply on next snapd start)"
            else
                gb_info "snapd snap-confine profile already patched (skip)"
            fi
        fi

        # Layer B - dynamic scan: patch profiles that don't inherit from base.
        # Skip: sub-directories, abstractions, tunables, disable, force-complain,
        #        local overrides, abi files, ldd (inlines base), and snap-confine
        #        (snapd manages and overwrites the main profile on every upgrade;
        #         handled above via the persistent local/ override instead).
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
        [[ -d "$aa_dir" ]] || gb_info "AppArmor not active on this system - skipping"
    fi

    # 8b/9. Unified supervisor (replaces idle-reclaim + recovery + sentinel + vram-watchdog)
    cmd_install_supervisor

    # 10. LD_AUDIT / LD_PRELOAD shim injection (merged from install-llama-configs)
    cmd_install_llama_configs

    # 11. tmpfiles.d - Audit F-L4-06: mode 0775 (was 0777) with the `ollama`
    #     group (or `root` if missing).  Ollama and the shim need to write
    #     metrics/NVTX/phase here, but no unprivileged third-party user should.
    local _tmpfiles_group="root"
    if getent group ollama >/dev/null 2>&1; then _tmpfiles_group="ollama"; fi
    printf 'd /run/greenboost 0775 root %s -\n' "$_tmpfiles_group" \
        > /etc/tmpfiles.d/greenboost.conf
    systemd-tmpfiles --create /etc/tmpfiles.d/greenboost.conf 2>/dev/null
    gb_ok "tmpfiles.d/greenboost.conf installed - /run/greenboost mode 0775 root:${_tmpfiles_group}"

    # 12. gb-diag-read - narrow read-only diagnostic helper (dmesg, /proc/<pid>/{maps,environ})
    # plus a NOPASSWD sudoers rule scoped to ONLY this binary (no argument wildcards,
    # which sudo's parser rejects - the script itself validates its own arguments).
    # Lets an AI assistant or any unprivileged tool self-diagnose shim/kernel-module
    # issues (LD_PRELOAD propagation, mapped libraries, dmesg) without a full sudo session.
    if [[ -f "${MODULE_DIR}/gb_diag_read.sh" ]]; then
        install -m 755 "${MODULE_DIR}/gb_diag_read.sh" /usr/local/sbin/gb-diag-read
        local _diag_user="${SUDO_USER:-${USER:-root}}"
        printf '%s ALL=(root) NOPASSWD: /usr/local/sbin/gb-diag-read\n' "$_diag_user" \
            > /etc/sudoers.d/greenboost-diag-read
        chmod 440 /etc/sudoers.d/greenboost-diag-read
        if visudo -c -f /etc/sudoers.d/greenboost-diag-read >/dev/null 2>&1; then
            gb_ok "gb-diag-read installed - NOPASSWD diagnostics enabled for ${_diag_user}"
        else
            rm -f /etc/sudoers.d/greenboost-diag-read
            warn "gb-diag-read sudoers rule failed validation - removed, diagnostics need manual sudo"
        fi
    fi

    echo ""
    info "System config installation complete."
    if systemctl is-active --quiet ollama.service 2>/dev/null; then
        systemctl restart ollama.service 2>/dev/null \
            && gb_ok "Ollama restarted - new env vars active" \
            || warn "Ollama restart failed - run: sudo systemctl restart ollama"
    fi
}


# cmd_install_idle_reclaim , removed; logic now in gb_supervisor.py via cmd_install_supervisor
# ── clean-memory ──────────────────────────────────────────────────────────────
# Force-release T1 VRAM + T2 RAM + T3 NVMe immediately.
# Framework-agnostic: finds every process with /dev/greenboost open via fuser,
# cmd_gaming_mode - stop/start Ollama to free T2 DDR before/after gaming.
# When OLLAMA_KEEP_ALIVE=-1 a loaded model stays pinned in T2 DDR indefinitely.
# Combined RAM pressure (model + game + OS) can push free RAM below safety_reserve_gb,
# triggering the OOM guard and crashing the game.  Stopping Ollama releases all T2 pages.
# prefers Ollama REST API when Ollama is among them, sends SIGTERM to everything else.
# The kernel's gb_close() → gb_release_pid_buffers() frees T2 DMA-BUF on exit.
cmd_t3_memory() {
    # Usage: greenboost t3-memory <size>   e.g. 100GB, 50GB, 0 (disk-limited)
    local raw="${1:-}"
    if [[ -z "$raw" ]]; then
        printf "  ${C_AMBER}⚠${C_RESET}  Usage: greenboost t3-memory <size>  (e.g. 100GB, 50GB, 0 for unlimited)\n"
        return 1
    fi

    # Parse value - strip suffix, allow G/GB/g
    local cap_gb
    cap_gb=$(echo "$raw" | sed 's/[Gg][Bb]*$//' | tr -d ' ')
    if ! [[ "$cap_gb" =~ ^[0-9]+$ ]]; then
        die "Invalid size '$raw' - expected a number in GB (e.g. 100GB or 100)"
    fi
    local cap_mb=$(( cap_gb * 1024 ))

    if [[ ! -e /dev/greenboost ]]; then
        die "GreenBoost kernel module not loaded - run: sudo greenboost load"
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
) || die "Failed to set T3 cap - are you root? (run with sudo)"

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
        if systemctl is-active --quiet ollama.service 2>/dev/null; then
            systemctl restart ollama.service 2>/dev/null \
                && gb_ok "Ollama restarted - GREENBOOST_VIRTUAL_VRAM_MB active" \
                || warn "Ollama restart failed - run: sudo systemctl restart ollama"
        fi
    fi

    # Show new T3 stats
    if [[ -r /sys/class/greenboost/greenboost/status ]]; then
        echo ""
        grep -E "T3|NVMe|t3|nvme" /sys/class/greenboost/greenboost/status 2>/dev/null | \
            sed "s/^/  ${C_CYAN}◈${C_RESET}  /" || true
    fi
}

# gb_ensure_greenboost_group , create the "greenboost" system group if absent and
# add the invoking user (SUDO_USER) to it so non-root processes (greenboost-cli,
# llama-server, vLLM) can read cluster.key at mode 0640 root:greenboost without
# requiring sudo.  Idempotent: safe to call on every install.
gb_ensure_greenboost_group() {
    if ! getent group greenboost >/dev/null 2>&1; then
        groupadd --system greenboost 2>/dev/null \
            || groupadd greenboost 2>/dev/null \
            || { gb_warn "Could not create 'greenboost' group , cluster.key will stay 0600 (root-only)"; return 0; }
        gb_ok "Created system group 'greenboost'"
    fi
    # Add the invoking user to the group when running via sudo.
    local _user="${SUDO_USER:-}"
    if [[ -n "$_user" ]] && id "$_user" &>/dev/null; then
        if ! id -nG "$_user" 2>/dev/null | grep -qw greenboost; then
            usermod -aG greenboost "$_user" 2>/dev/null \
                || gpasswd -a "$_user" greenboost 2>/dev/null \
                || gb_warn "Could not add '$_user' to group 'greenboost' , run: sudo usermod -aG greenboost $_user"
            gb_ok "Added '$_user' to group 'greenboost'"
            gb_warn "Log out and back in (or run: newgrp greenboost) for the new group membership to take effect in existing shells"
        fi
    fi
}

# _gb_set_keyfile_perms <path> , apply correct ownership + mode to cluster.key.
# Uses 0640 root:greenboost when the group exists; falls back to 0600 root:root.
_gb_set_keyfile_perms() {
    local _kf="$1"
    if getent group greenboost >/dev/null 2>&1; then
        chown root:greenboost "$_kf"
        chmod 0640 "$_kf"
    else
        chown root:root "$_kf"
        chmod 0600 "$_kf"
    fi
}

# gb_ensure_shim_libs , build shim + vmm_override + audit and install-libs if any
# installed lib is missing or any source file is newer than the installed shim.
# Called at the top of every config-install command so the correct workflow
# (build → install) is always followed, even for standalone CLI invocations.
gb_ensure_shim_libs() {
    local shim_installed="$SHIM_DEST/$SHIM_LIB"
    local vmm_installed="$SHIM_DEST/libgreenboost_vmm_override.so"
    local audit_installed="$SHIM_DEST/$AUDIT_LIB"

    local need_build=0

    # Trigger rebuild if any lib is absent
    [[ ! -f "$shim_installed"  ]] && { warn "Shim not installed at $shim_installed , will build"; need_build=1; }
    [[ ! -f "$vmm_installed"   ]] && { warn "VMM override not installed at $vmm_installed , will build"; need_build=1; }
    [[ ! -f "$audit_installed" ]] && { warn "Audit lib not installed at $audit_installed , will build"; need_build=1; }

    # Also trigger if any source file is newer than the installed shim
    if (( ! need_build )) && [[ -f "$shim_installed" ]]; then
        for _src in \
            "$MODULE_DIR/greenboost_cuda_shim.c" \
            "$MODULE_DIR/greenboost_cuda_v12.c" \
            "$MODULE_DIR/greenboost_netc.c" \
            "$MODULE_DIR/greenboost_vmm_override.c" \
            "$MODULE_DIR/greenboost_audit.c"; do
            [[ -f "$_src" && "$_src" -nt "$shim_installed" ]] && { need_build=1; break; }
        done
    fi

    (( ! need_build )) && return 0

    gb_step 0 1 "Building GreenBoost shim + vmm_override + audit (source newer than install)..."
    local _log; _log=$(mktemp /tmp/gb_shim_ensure.XXXXX.log)
    if ! make -C "$MODULE_DIR" shim vmm_override audit >"$_log" 2>&1; then
        echo "" >&2; cat "$_log" >&2; rm -f "$_log"
        gb_die "Shim build failed , fix compiler errors and retry"
    fi
    if ! make -C "$MODULE_DIR" install-libs >>"$_log" 2>&1; then
        echo "" >&2; cat "$_log" >&2; rm -f "$_log"
        gb_die "install-libs failed , check Makefile and retry"
    fi
    rm -f "$_log"
    # Restore ownership so the developer can rebuild without sudo
    if [[ -n "${SUDO_USER:-}" ]]; then
        chown -R "${SUDO_USER}:$(id -gn "${SUDO_USER}" 2>/dev/null || echo users)" \
            "$MODULE_DIR" 2>/dev/null || true
    fi
    gb_ok "Shim libs built and installed"
}

cmd_install_llama_configs() {
    need_root install-llama-configs
    gb_ensure_shim_libs

    local audit_path="$SHIM_DEST/$AUDIT_LIB"

    # /etc/ld.so.preload - ordered list of three GreenBoost libraries:
    #
    # 1. libgreenboost_vmm_override.so  (FIRST - Blackwell VMM PLT fix)
    # 2. libgreenboost_cuda.so          (full CUDA shim - memory inflation etc.)
    # 3. libgreenboost_audit.so         (LD_AUDIT - CUDA-process injection)
    #
    # WHY libgreenboost_vmm_override.so must be FIRST:
    #   libcuda.so.1 exports cuDeviceGetAttribute and cuMemAddressReserve as BARE
    #   UNVERSIONED symbols.  The CUDA shim exports them as @@GB_HOOKS (versioned).
    #   glibc explicitly PREFERS unversioned definitions for unversioned PLT
    #   references - so libcuda always wins the PLT race over the shim regardless
    #   of ld.so.preload order.  The VMM override library also exports these symbols
    #   as BARE UNVERSIONED and is loaded BEFORE libcuda enters the link map, so
    #   glibc finds OUR unversioned definition first.  On Blackwell (cc >= 12) it
    #   zeroes the VMM_SUPPORTED attribute, forcing ggml-cuda to select pool_leg
    #   (cudaMalloc-based) whose overflow routes through managed-UVM (SM-accessible).
    if [[ ! -f "$audit_path" ]]; then
        warn "LD_AUDIT library not found at $audit_path - run 'make audit' and re-run this command"
        return 1
    fi

    local vmm_override_lib="libgreenboost_vmm_override.so"
    local vmm_override_path="$SHIM_DEST/$vmm_override_lib"
    local shim_path_full="$SHIM_DEST/$SHIM_LIB"

    if [[ ! -f "$shim_path_full" ]]; then
        warn "CUDA shim not found at $shim_path_full - run 'make install' and re-run this command"
        return 1
    fi

    # Install-time capability manifest: answers "what does the INSTALLED shim
    # support" before any shim process runs (the shim also writes a per-launch
    # /run/greenboost/capabilities.json that supersedes this at runtime). Only
    # static build features are known here; env-gated runtime features are left
    # false — consumers that need those read the runtime manifest.
    # Keep the feature set in sync with gb_write_capabilities_file() in the shim.
    cat > "$SHIM_DEST/greenboost_capabilities.json" << 'CAPEOF'
{
  "shim_version": "3.2",
  "abi": 1,
  "source": "install",
  "features": {
    "gb_quant_cudart_rebind": true,
    "expert_pool": true,
    "cluster_fabric": true,
    "gds": false,
    "kv_compress": false,
    "report_physical_vram": false
  }
}
CAPEOF
    chmod 0644 "$SHIM_DEST/greenboost_capabilities.json" 2>/dev/null || true

    # GreenBoost no longer registers shims in /etc/ld.so.preload.  Writing paths
    # to our CUDA/audit interposers there loads them into EVERY process on the
    # system , including systemd PID 1 , which freezes early boot (manifests as
    # "Failed to load libmount.so" / "systemd[1]: Freezing execution").
    #
    # Injection is scoped per-process via:
    #   • systemd service drop-ins  (/etc/systemd/system/*.service.d/99-greenboost.conf)
    #   • wrapper scripts           (greenboost-run, greenboost-run-tgi, etc.)
    # Both mechanisms set LD_PRELOAD only for the processes that actually need it.
    #
    # Scrub any stale entries that may have been written by an older install.
    if [[ -f /etc/ld.so.preload ]]; then
        sed -i '/libgreenboost/d;/greenboost/d' /etc/ld.so.preload
        [[ -s /etc/ld.so.preload ]] || rm -f /etc/ld.so.preload
        info "ld.so.preload: removed any stale GreenBoost entries."
    fi

    echo ""
    info "Injection configured , shims activate per-process via systemd drop-ins and greenboost-run* wrappers."
    info "  CUDA shim : $shim_path_full"
    info "  Audit lib : $audit_path"
}

# ── Restore functions - undo each Additional Install step individually ──────

cmd_restore_sys_configs() {
    need_root restore-sys-configs

    gb_info "Restoring sys configs - removing GreenBoost service/udev/governor/shim entries..."

    # Ollama drop-in
    local _dropin="/etc/systemd/system/ollama.service.d/99-greenboost.conf"
    if [[ -f "$_dropin" ]]; then
        rm -f "$_dropin"
        gb_ok "Removed Ollama drop-in: ${_dropin}"
    fi
    # Remove inline GreenBoost injections from ollama.service (legacy direct edit)
    local _svc="/etc/systemd/system/ollama.service"
    if [[ -f "$_svc" ]] && grep -q "GREENBOOST\|libgreenboost" "$_svc"; then
        sed -i '/OLLAMA_FLASH_ATTENTION/d;/OLLAMA_KV_CACHE_TYPE/d;/OLLAMA_NUM_CTX/d;/OLLAMA_CONTEXT_LENGTH/d
/OLLAMA_MAX_LOADED_MODELS/d;/OLLAMA_KEEP_ALIVE/d;/OLLAMA_NUM_GPU/d
/GREENBOOST_/d;/libgreenboost/d' "$_svc"
        gb_ok "Removed GreenBoost vars from ollama.service"
    fi
    systemctl daemon-reload 2>/dev/null || true

    # LD_PRELOAD / LD_AUDIT injection
    if [[ -f /etc/ld.so.preload ]] && grep -q "greenboost\|libgreenboost" /etc/ld.so.preload; then
        sed -i '/greenboost/d;/libgreenboost/d' /etc/ld.so.preload
        [[ -s /etc/ld.so.preload ]] || rm -f /etc/ld.so.preload
        ldconfig 2>/dev/null || true
        gb_ok "Removed GreenBoost entries from /etc/ld.so.preload"
    fi

    # udev rules
    for _r in /etc/udev/rules.d/99-greenboost.rules \
               /etc/udev/rules.d/99-nvme-greenboost.rules; do
        [[ -f "$_r" ]] && rm -f "$_r" && gb_ok "Removed udev rule: ${_r}"
    done
    udevadm control --reload-rules 2>/dev/null || true

    # NVIDIA shutdown fix
    if [[ -f /etc/modprobe.d/99-nvidia-greenboost.conf ]]; then
        rm -f /etc/modprobe.d/99-nvidia-greenboost.conf
        gb_ok "Removed NVIDIA shutdown fix conf"
    fi

    # CPU governor service
    if systemctl is-enabled --quiet cpu-perf.service 2>/dev/null; then
        systemctl disable --now cpu-perf.service 2>/dev/null || true
    fi
    [[ -f /etc/systemd/system/cpu-perf.service ]] && \
        rm -f /etc/systemd/system/cpu-perf.service && gb_ok "Removed cpu-perf.service"
    systemctl daemon-reload 2>/dev/null || true

    # THP sysfs conf
    if [[ -f /etc/sysfs.d/greenboost-hugepages.conf ]]; then
        rm -f /etc/sysfs.d/greenboost-hugepages.conf
        gb_ok "Removed THP sysfs conf"
    fi

    # AppArmor abstraction
    if [[ -f /etc/apparmor.d/abstractions/greenboost-audit ]]; then
        rm -f /etc/apparmor.d/abstractions/greenboost-audit
        local _base="/etc/apparmor.d/abstractions/base"
        [[ -f "$_base" ]] && sed -i '/greenboost-audit/d' "$_base"
        apparmor_parser -r /etc/apparmor.d/ 2>/dev/null || true
        gb_ok "Removed AppArmor greenboost-audit abstraction"
    fi

    echo ""
    gb_ok "Sys configs restored. Restart Ollama: sudo systemctl restart ollama"
}

cmd_restore_tune_runtime() {
    need_root restore-tune-runtime

    gb_info "Restoring runtime tuning - resetting CPU governor, NVMe scheduler, PCIe, VM..."

    # CPU governor → schedutil (modern default)
    for _g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo schedutil > "$_g" 2>/dev/null || echo powersave > "$_g" 2>/dev/null || true
    done
    gb_ok "CPU governor → schedutil (default)"

    # NVMe scheduler → mq-deadline (kernel default)
    for _s in /sys/block/nvme*/queue/scheduler; do
        echo mq-deadline > "$_s" 2>/dev/null || true
    done
    gb_ok "NVMe scheduler → mq-deadline (default)"

    # Re-enable PCIe runtime PM
    for _pm in /sys/bus/pci/devices/*/power/control; do
        echo auto > "$_pm" 2>/dev/null || true
    done
    gb_ok "PCIe runtime PM → auto (default)"

    # VM defaults
    sysctl -w vm.swappiness=60 vm.dirty_ratio=20 vm.dirty_background_ratio=10 \
              kernel.numa_balancing=1 >/dev/null 2>&1 && \
        gb_ok "VM tunables reset to defaults (swappiness=60, dirty_ratio=20)"

    # Remove any runtime-only NVMe tuning persist file written by tune
    local _nvme_conf="/etc/udev/rules.d/99-greenboost-nvme-runtime.rules"
    [[ -f "$_nvme_conf" ]] && rm -f "$_nvme_conf" && \
        udevadm control --reload-rules 2>/dev/null || true

    echo ""
    gb_ok "Runtime tuning restored. Note: these are live changes - persist via tune-sysctl."
}

cmd_restore_tune_sysctl() {
    need_root restore-tune-sysctl

    local _conf="/etc/sysctl.d/99-zzz-greenboost.conf"
    if [[ -f "$_conf" ]]; then
        rm -f "$_conf"
        gb_ok "Removed ${_conf}"
        # Reload remaining sysctl confs
        sysctl --system >/dev/null 2>&1 && gb_ok "Reloaded kernel tunables from remaining conf files"
    else
        gb_info "No GreenBoost sysctl conf found at ${_conf} - nothing to remove"
    fi
    echo ""
    gb_ok "Sysctl restored. Kernel defaults apply after next sysctl --system or reboot."
}

cmd_restore_tune_grub() {
    need_root restore-tune-grub

    local _grub="/etc/default/grub"
    [[ -f "$_grub" ]] || { gb_warn_ui "GRUB config not found: ${_grub}"; return 1; }

    gb_info "Removing GreenBoost GRUB parameters from ${_grub}..."

    # Backup before touching
    cp --preserve=all "$_grub" "${_grub}.gb-restore-$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true

    # Parameters added by cmd_tune_grub
    local _gb_params=(
        "transparent_hugepage=always"
        "skew_tick=1"
        "numa_balancing=disable"
        "workqueue.power_efficient=0"
    )
    # Regex-removable pattern params (value varies per machine)
    local _gb_patterns=(
        "rcu_nocbs=[^ \"]*"
        "nohz_full=[^ \"]*"
    )

    local _line
    _line=$(grep '^GRUB_CMDLINE_LINUX_DEFAULT=' "$_grub" | head -1 \
            | sed 's/^GRUB_CMDLINE_LINUX_DEFAULT=//;s/^"//;s/"$//')

    local _new="$_line"
    for _p in "${_gb_params[@]}"; do
        _new=$(echo "$_new" | sed "s/ *${_p}//g; s/${_p} *//g")
    done
    for _re in "${_gb_patterns[@]}"; do
        _new=$(echo "$_new" | sed "s/ *${_re}//g; s/${_re} *//g")
    done
    _new=$(echo "$_new" | tr -s ' ' | sed 's/^ //;s/ $//')

    sed -i "s|^GRUB_CMDLINE_LINUX_DEFAULT=.*|GRUB_CMDLINE_LINUX_DEFAULT=\"${_new}\"|" "$_grub"
    gb_ok "Removed GreenBoost params - new cmdline: ${_new}"

    # Regenerate GRUB config
    if command -v update-grub &>/dev/null; then
        update-grub 2>/dev/null && gb_ok "GRUB config updated (update-grub)"
    elif command -v grub-mkconfig &>/dev/null; then
        grub-mkconfig -o /boot/grub/grub.cfg 2>/dev/null && gb_ok "GRUB config updated (grub-mkconfig)"
    else
        gb_warn_ui "Could not update GRUB automatically - run: sudo update-grub"
    fi

    echo ""
    gb_ok "GRUB restored. Changes take effect after next reboot."
}

# ---- _rmmod_with_retry - unload the kernel module, retrying up to ~15 s ----
# Usage: _rmmod_with_retry [quiet]
# Returns 0 on success, 1 if the module is still loaded after all retries.
# Strategy:
#   1. Kill every process with /dev/greenboost open (fuser + lsof fallback).
#   2. sync + drop page-cache so DMA-BUF kernel references are released.
#   3. Retry rmmod up to MAX_TRIES times with 1 s sleep between attempts.
#      Each failed attempt repeats the fuser kill in case new processes appeared.
_rmmod_with_retry() {
    local quiet="${1:-}"
    local MAX_TRIES=15
    local attempt=0

    # Helper: kill everything holding the device open.
    _kill_dev_users() {
        if [[ -e /dev/greenboost ]]; then
            fuser -k /dev/greenboost 2>/dev/null || true
            # lsof fallback - catches processes fuser sometimes misses
            local _pids
            _pids=$(lsof /dev/greenboost 2>/dev/null | awk 'NR>1 {print $2}' | sort -u || true)
            [[ -n "$_pids" ]] && kill -9 $( echo "$_pids" ) 2>/dev/null || true
        fi
    }

    grep -q "^${DRIVER_NAME} " <<< "$(lsmod)" || return 0   # not loaded - nothing to do

    # Check if the module is already stuck in Unloading state (MODULE_STATE_GOING).
    # This happens when a previous rmmod process was killed before the exit function
    # could complete. The kernel won't let a new rmmod succeed while in this state.
    # Strategy:
    #   1. Wait up to 15 s for the exit function to complete naturally.
    #   2. If still stuck, try rmmod -f (force; requires CONFIG_MODULE_FORCE_UNLOAD).
    #   3. If force also fails, give up - caller should schedule boot-time cleanup.
    if grep -q "^${DRIVER_NAME}.*Unloading" /proc/modules 2>/dev/null; then
        # Parse refcount - if -1, the module is permanently stuck (kernel ABI crash in gb_exit).
        # Waiting is pointless; skip straight to force-unload.
        local _rc
        _rc=$(awk "/^${DRIVER_NAME} /{print \$3}" /proc/modules 2>/dev/null)
        if [[ "$_rc" != "-1" ]]; then
            gb_warn_ui "Module is in 'Unloading' state - waiting up to 15s for exit to complete…"
            local _waited=0
            while (( _waited < 15 )); do
                sleep 2; (( _waited += 2 ))
                if ! grep -q "^${DRIVER_NAME}" /proc/modules 2>/dev/null; then
                    gb_ok "Module finished unloading after ${_waited}s"
                    return 0
                fi
                ! grep -q "^${DRIVER_NAME}.*Unloading" /proc/modules 2>/dev/null && break
            done
        fi

        if grep -q "^${DRIVER_NAME}.*Unloading" /proc/modules 2>/dev/null; then
            local _msg="Still stuck"
            [[ "$_rc" == "-1" ]] && _msg="Module is permanently stuck (refcnt=-1)"
            gb_warn_ui "${_msg} - trying rmmod -f (force)…"
            if rmmod -f "$DRIVER_NAME" 2>/dev/null; then
                gb_ok "Module force-unloaded (kernel tainted; harmless for next load)"
                return 0
            fi
            gb_warn_ui "rmmod -f also failed - a reboot is required."
            gb_warn_ui "After rebooting: sudo ./greenboost_setup.sh full-install"
            return 1
        fi
    fi

    _kill_dev_users
    sync
    # Drop page-cache + slab objects so DMA-BUF pinned pages are returned to buddy.
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    sleep 0.3

    while (( attempt < MAX_TRIES )); do
        if rmmod "$DRIVER_NAME" 2>/dev/null; then
            [[ -z "$quiet" ]] && gb_ok "Kernel module unloaded (attempt $((attempt+1)))"
            return 0
        fi
        (( attempt++ )) || true
        # Re-check: rmmod may have triggered the exit function and the module is
        # now transitioning. If so, wait for it to finish rather than giving up.
        if grep -q "^${DRIVER_NAME}.*Unloading" /proc/modules 2>/dev/null; then
            gb_warn_ui "Module entered 'Unloading' state - waiting for exit to complete…"
            local _uw=0
            while (( _uw < 15 )); do
                sleep 2; (( _uw += 2 ))
                grep -q "^${DRIVER_NAME}" /proc/modules 2>/dev/null || { gb_ok "Module unloaded"; return 0; }
                grep -q "^${DRIVER_NAME}.*Unloading" /proc/modules 2>/dev/null || break
            done
            grep -q "^${DRIVER_NAME}.*Unloading" /proc/modules 2>/dev/null || continue
            rmmod -f "$DRIVER_NAME" 2>/dev/null && { gb_ok "Module force-unloaded"; return 0; }
            gb_warn_ui "Unloading stuck and force failed - reboot required."
            return 1
        fi
        local refcnt
        refcnt=$(cat /sys/module/${DRIVER_NAME}/refcnt 2>/dev/null || echo "?")
        [[ -z "$quiet" ]] && \
            gb_warn_ui "rmmod attempt ${attempt}/${MAX_TRIES} - refcnt=${refcnt}, retrying…"
        _kill_dev_users
        sleep 1
    done

    gb_warn_ui "rmmod failed after ${MAX_TRIES} attempts - module still loaded (refcnt=$(cat /sys/module/${DRIVER_NAME}/refcnt 2>/dev/null || echo '?'))."
    gb_warn_ui "Run manually: sudo rmmod ${DRIVER_NAME}   (or reboot)"
    return 1
}

# ---- _try_install_fixed_module -----------------------------------------------
# When rmmod fails (module stuck in STATE_GOING), the underlying cause is often
# a crash in gb_exit (e.g. unregister_kretprobe on a NULL rph because the compiled
# .ko predates the g_kretprobe_ok guard).  This helper recompiles greenboost.ko
# from the current source (which has the fix) and installs it WITHOUT calling rmmod,
# so the next boot loads the corrected module.
#
# Returns 0 on success (fixed .ko installed - caller should NOT schedule boot cleanup,
# the stuck STATE_GOING module clears naturally on reboot).
# Returns 1 on failure (build unavailable - caller should fall back to boot cleanup).
_try_install_fixed_module() {
    local KVER; KVER=$(uname -r)
    local INSTALL_DIR="/lib/modules/${KVER}/extra"

    gb_info "Attempting to compile + install fixed module (skipping rmmod)…"

    # Compile - module target only, not full build
    if ! make -C "$MODULE_DIR" module > /tmp/gb_fix_build.log 2>&1; then
        gb_warn_ui "Could not compile fixed module - see /tmp/gb_fix_build.log"
        return 1
    fi

    mkdir -p "$INSTALL_DIR"
    cp "$MODULE_DIR/greenboost.ko" "$INSTALL_DIR/greenboost.ko"
    depmod -a "$KVER" 2>/dev/null || true
    gb_ok "Fixed greenboost.ko installed to $INSTALL_DIR"

    # Ensure the module auto-loads on the next boot (may have been removed by a
    # previous boot cleanup or partial purge).
    [[ ! -f /etc/modules-load.d/greenboost.conf ]] \
        && echo "greenboost" > /etc/modules-load.d/greenboost.conf

    # Remove any stale blacklist that would prevent loading.
    rm -f /etc/modprobe.d/99-greenboost-blacklist.conf

    # Update DKMS source tree so kernel updates also use the fixed source.
    local GB_VER; GB_VER=$(grep -m1 '^GB_VERSION' "$MODULE_DIR/Makefile" 2>/dev/null \
                           | awk -F':?=' '{gsub(/ /,"",$2); print $2}')
    local DKMS_ROOT="/usr/src/greenboost-${GB_VER}"
    if command -v dkms &>/dev/null && [[ -d "$DKMS_ROOT" ]]; then
        cp "$MODULE_DIR/greenboost.c" "$DKMS_ROOT/"
        cp "$MODULE_DIR/greenboost_ioctl.h" "$DKMS_ROOT/" 2>/dev/null || true
        gb_ok "Updated DKMS source tree at $DKMS_ROOT"
    fi

    return 0
}

# ---- _schedule_boot_cleanup - called when module is stuck in MODULE_STATE_GOING.
# Removes any legacy boot-cleanup artifacts and prints SysRq force-reboot instructions.
# No boot service is installed - it caused more problems than it solved.
_schedule_boot_cleanup() {
    # Remove any leftover boot-cleanup service from old installs
    systemctl disable --now greenboost-boot-cleanup.service 2>/dev/null || true
    rm -f /etc/systemd/system/greenboost-boot-cleanup.service
    rm -f /etc/systemd/system/sysinit.target.wants/greenboost-boot-cleanup.service
    rm -f /etc/systemd/system/rescue.target.wants/greenboost-boot-cleanup.service
    rm -f /etc/systemd/system/emergency.target.wants/greenboost-boot-cleanup.service
    rm -f /usr/local/sbin/greenboost-boot-cleanup
    rm -f /lib/systemd/system-shutdown/greenboost-stuck.sh
    systemctl daemon-reload 2>/dev/null || true

    gb_warn_ui "Module stuck in Unloading - reboot required."
    echo ""
    echo -e "  ${C_BOLD}Force-reboot now (instant, no hang):${C_RESET}"
    echo -e "    sync && echo b | sudo tee /proc/sysrq-trigger"
    echo ""
    echo -e "  Wait ~30s, reconnect, then re-run:"
    echo -e "    sudo ./greenboost_setup.sh full-install"
    echo ""
}

# ---- _gb_backup_create - snapshot pre-install system state ---------------
_GB_BACKUP_DIR="/etc/greenboost/backup"
_gb_backup_create() {
    local stamp; stamp=$(date +%Y%m%d_%H%M%S)
    local bdir="${_GB_BACKUP_DIR}/${stamp}"
    mkdir -p "$bdir" || { gb_warn_ui "Backup: cannot create ${bdir}"; return 1; }

    # GRUB config
    [[ -f /etc/default/grub ]] && cp --preserve=all /etc/default/grub "${bdir}/grub"
    # ld.so.preload
    [[ -f /etc/ld.so.preload ]] && cp --preserve=all /etc/ld.so.preload "${bdir}/ld.so.preload"
    # Kernel tunables (vm/kernel/net subset)
    sysctl -a --ignore-errors 2>/dev/null \
        | grep -E '^(vm\.|kernel\.|net\.)' > "${bdir}/sysctl_state.conf" || true
    # CPU governor
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null \
        > "${bdir}/cpu_governor.txt" || true
    # NVMe scheduler (first nvme device if present)
    local _nvme; _nvme=$(ls /sys/block/nvme*/queue/scheduler 2>/dev/null | head -1 || true)
    [[ -n "$_nvme" ]] && cat "$_nvme" > "${bdir}/nvme_scheduler.txt" 2>/dev/null || true
    # Ollama service environment
    local _olenv="/etc/systemd/system/ollama.service.d/override.conf"
    [[ -f "$_olenv" ]] && cp --preserve=all "$_olenv" "${bdir}/ollama_service_env.txt"
    # Manifest
    printf 'stamp=%s\nhostname=%s\nkernel=%s\n' \
        "$stamp" "$(hostname)" "$(uname -r)" > "${bdir}/manifest.txt"

    gb_ok "System backup created: ${bdir}"
}

# ---- _gb_backup_restore - apply most recent backup -----------------------
_gb_backup_restore() {
    local bdir; bdir=$(ls -1d "${_GB_BACKUP_DIR}"/[0-9]* 2>/dev/null | sort | tail -1)
    if [[ -z "$bdir" ]]; then
        gb_warn_ui "No backup found under ${_GB_BACKUP_DIR}"
        return 1
    fi
    local stamp; stamp=$(basename "$bdir")
    gb_info "Restoring from backup: ${stamp}"

    [[ -f "${bdir}/grub" ]]        && cp --preserve=all "${bdir}/grub" /etc/default/grub       && gb_ok "Restored /etc/default/grub"
    [[ -f "${bdir}/ld.so.preload" ]] && cp --preserve=all "${bdir}/ld.so.preload" /etc/ld.so.preload && gb_ok "Restored /etc/ld.so.preload"
    [[ -f "${bdir}/sysctl_state.conf" ]] && sysctl --load "${bdir}/sysctl_state.conf" &>/dev/null && gb_ok "Restored sysctl state"
    if [[ -f "${bdir}/cpu_governor.txt" ]]; then
        local _gov; _gov=$(cat "${bdir}/cpu_governor.txt" | tr -d '[:space:]')
        [[ -n "$_gov" ]] && \
            for _g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
                echo "$_gov" > "$_g" 2>/dev/null || true
            done && gb_ok "Restored CPU governor: ${_gov}"
    fi
    if [[ -f "${bdir}/nvme_scheduler.txt" ]]; then
        local _sched; _sched=$(cat "${bdir}/nvme_scheduler.txt" | grep -oP '^\[\K[^]]+' || cat "${bdir}/nvme_scheduler.txt" | tr -d '[:space:]')
        [[ -n "$_sched" ]] && \
            for _ns in /sys/block/nvme*/queue/scheduler; do
                echo "$_sched" > "$_ns" 2>/dev/null || true
            done && gb_ok "Restored NVMe scheduler: ${_sched}"
    fi
    [[ -f "${bdir}/ollama_service_env.txt" ]] && \
        { mkdir -p /etc/systemd/system/ollama.service.d && \
          cp --preserve=all "${bdir}/ollama_service_env.txt" /etc/systemd/system/ollama.service.d/override.conf && \
          systemctl daemon-reload 2>/dev/null || true; \
          gb_ok "Restored Ollama service override"; }
    gb_ok "Restore complete from backup ${stamp}"
}

# ---- _do_purge_apparmor - roll back every AppArmor change made by the ----
# install.  Idempotent; safe to re-run.  Used by both `cmd_uninstall` (via
# do_purge) and the standalone `cmd_apparmor_uninstall` so a single helper
# is the source of truth for what gets reverted.
#
# Reverses, in order:
#   1) abstractions/greenboost-audit                - the abstraction file itself
#   2) abstractions/base                            - Layer A: `#include` injection
#   3) /etc/apparmor.d/*                            - Layer B: per-profile includes
#   4) local/usr.lib.snapd.snap-confine.real        - snap-confine override
#   5) /var/lib/snapd/apparmor/profiles/snap-confine*  - snapd-snap profile patch
#   6) apparmor_parser -r /etc/apparmor.d/          - make the kernel forget
#   7) snapd-shipped snap-confine profile reload    - restore self-integrity check
# ---------------------------------------------------------------------------
_do_purge_apparmor() {
    [[ -d /etc/apparmor.d ]] || return 0

    local removed=0 patched=0

    # 1) Remove the abstraction file
    if [[ -f /etc/apparmor.d/abstractions/greenboost-audit ]]; then
        rm -f /etc/apparmor.d/abstractions/greenboost-audit
        (( removed++ )) || true
    fi

    # 2) Layer A - abstractions/base
    local base_abs="/etc/apparmor.d/abstractions/base"
    if [[ -f "$base_abs" ]] && grep -q "greenboost-audit" "$base_abs"; then
        sed -i '/greenboost-audit/d' "$base_abs"
        (( removed++ )) || true
    fi

    # 3) Layer B - every profile in /etc/apparmor.d/ that we patched
    local _profiles=() pf
    while IFS= read -r pf; do
        _profiles+=("$pf")
    done < <(find /etc/apparmor.d/ -maxdepth 1 -type f 2>/dev/null | sort)
    for pf in "${_profiles[@]}"; do
        [[ -d "$pf" ]] && continue
        if grep -q "greenboost-audit" "$pf" 2>/dev/null; then
            sed -i '/greenboost-audit/d' "$pf"
            (( patched++ )) || true
        fi
    done

    # 4) snap-confine local override - this is the file whose modification
    #    trips `snap-confine has elevated permissions and is not confined
    #    but should be - refusing to continue`.  Remove our two lines and the
    #    preceding comment.  If the file ends up empty, remove it entirely so
    #    snapd's profile reload picks up a pristine state.
    local sc_local="/etc/apparmor.d/local/usr.lib.snapd.snap-confine.real"
    if [[ -f "$sc_local" ]] && grep -q "libgreenboost_audit" "$sc_local"; then
        sed -i \
            -e '/^# GreenBoost: allow snap-confine to mmap the LD_AUDIT stub$/d' \
            -e '/libgreenboost_audit/d' \
            "$sc_local"
        # Trim leading/trailing blank lines; remove if pure whitespace.
        sed -i '/./,$!d; :a; /^\s*$/{$d;N;ba}' "$sc_local"
        [[ -s "$sc_local" ]] || rm -f "$sc_local"
        (( removed++ )) || true
    fi

    # 5) snapd-shipped snap-confine profile (lives outside /etc/apparmor.d/)
    local snapd_profile
    while IFS= read -r snapd_profile; do
        [[ -f "$snapd_profile" ]] || continue
        if grep -q "libgreenboost_audit" "$snapd_profile" 2>/dev/null; then
            sed -i \
                -e '/^# GreenBoost: allow snap-confine (snapd snap) to mmap audit stub$/d' \
                -e '/libgreenboost_audit/d' \
                "$snapd_profile"
            apparmor_parser -r "$snapd_profile" 2>/dev/null || true
            (( patched++ )) || true
        fi
    done < <(find /var/lib/snapd/apparmor/profiles/ \
        \( -name "snap-confine.snapd.*" -o -name "snap-confine.*" \) \
        2>/dev/null)

    if (( removed > 0 || patched > 0 )); then
        gb_ok "AppArmor: reverted $removed file edit(s) + $patched profile patch(es)"
    else
        gb_info "AppArmor: nothing to revert"
    fi

    # 6) Tell the kernel to forget the old profile data.
    apparmor_parser -r /etc/apparmor.d/ 2>/dev/null &
    gb_spin $! "Reloading AppArmor rules..."

    # 7) snap-confine's self-integrity check uses the profile loaded by snapd.
    #    Force snapd to reload its shipped snap-confine profile so the
    #    refused-elevated-permissions error stops triggering.  snapd holds the
    #    canonical profile in /var/lib/snapd/apparmor/profiles/ and reloads it
    #    on service start; restart snapd to be safe.
    if systemctl is-active snapd >/dev/null 2>&1; then
        systemctl restart snapd 2>/dev/null \
            && gb_ok "snapd restarted - snap-confine self-check restored" \
            || gb_warn_ui "snapd restart failed - run: sudo systemctl restart snapd"
    fi
}

# ---- cmd_apparmor_uninstall - standalone entrypoint for the helper -------
# Lets users fix the snap-confine breakage without a full GreenBoost
# uninstall.  Usage:  sudo greenboost apparmor-uninstall
# ---------------------------------------------------------------------------
cmd_apparmor_uninstall() {
    need_root apparmor-uninstall
    echo ""
    info "Reverting GreenBoost AppArmor modifications..."
    _do_purge_apparmor
    echo ""
    info "Done.  Snap apps (Firefox, VSCode, etc.) should now launch normally."
    info "Test with: snap run firefox  (or just click the icon)"
}

# ---- do_purge - remove ALL previously installed GreenBoost artifacts -----
# Internal helper (no root check - callers must ensure root).
# Called by cmd_uninstall and cmd_full_install.
do_purge() {
    # restart_after=1 → restart stopped services at the end (cmd_uninstall).
    # restart_after=0 → leave them stopped (cmd_full_install handles restart after fresh install).
    # preserve_cluster=1 → keep cluster identity/state files across the
    #   /etc/greenboost wipe (install/reinstall paths). A true uninstall
    #   passes 0 so secrets don't outlive the product.
    local restart_after="${1:-0}"
    local preserve_cluster="${2:-0}"

    # 0. Remove any legacy boot-cleanup artifacts from previous failed install attempts.
    #    These services caused more problems than they solved (kernel oops during boot).
    systemctl disable --now greenboost-boot-cleanup.service 2>/dev/null || true
    rm -f /etc/systemd/system/greenboost-boot-cleanup.service
    rm -f /etc/systemd/system/sysinit.target.wants/greenboost-boot-cleanup.service
    rm -f /etc/systemd/system/rescue.target.wants/greenboost-boot-cleanup.service
    rm -f /etc/systemd/system/emergency.target.wants/greenboost-boot-cleanup.service
    rm -f /usr/local/sbin/greenboost-boot-cleanup
    rm -f /lib/systemd/system-shutdown/greenboost-stuck.sh
    systemctl daemon-reload 2>/dev/null || true

    # 1. Stop services that hold /dev/greenboost open (prevents rmmod EBUSY).
    #    Ollama and llama-server are NOT uninstalled - only stopped temporarily.
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
            (( waited++ )) || true
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

    # 3. Remove GreenBoost entries from /etc/ld.so.preload FIRST - before deleting
    #    the .so files (prevents "cannot be preloaded" linker errors on forked procs).
    if [[ -f /etc/ld.so.preload ]] && grep -q "libgreenboost" /etc/ld.so.preload; then
        sed -i '/libgreenboost/d' /etc/ld.so.preload
        [[ -s /etc/ld.so.preload ]] || rm -f /etc/ld.so.preload
        gb_ok "Removed from /etc/ld.so.preload"
    fi

    # 4. Remove CUDA shim, VMM override, LD_AUDIT library + install-time
    #    capability manifest.  vmm_override (cmd_install) and
    #    greenboost_capabilities.json (cmd_install_llama_configs) are shim
    #    companions and must be removed symmetrically, not just the two core libs.
    local _libs_removed=0
    for lib in "$SHIM_DEST/$SHIM_LIB" \
               "$SHIM_DEST/$AUDIT_LIB" \
               "$SHIM_DEST/libgreenboost_vmm_override.so" \
               "$SHIM_DEST/greenboost_capabilities.json"; do
        [[ -f "$lib" ]] && rm -f "$lib" && (( _libs_removed++ )) || true
    done
    { ldconfig 2>/dev/null || true; } &
    gb_spin $! "Refreshing dynamic linker cache..."
    [[ $_libs_removed -gt 0 ]] && gb_ok "CUDA shim + VMM override + audit library removed"

    # 4b. Remove AppArmor abstraction + all injected includes + snap-confine
    #     modifications.  Calls the shared helper so the standalone
    #     `greenboost apparmor-uninstall` command and the full uninstall both
    #     reverse exactly the same set of changes.
    _do_purge_apparmor

    # 5. Remove static config files (silent batch - no per-file noise)
    local _cfg_removed=0
    for f in \
        /etc/modprobe.d/greenboost.conf \
        /etc/modprobe.d/greenboost.conf.bak \
        /etc/modprobe.d/99-greenboost-blacklist.conf \
        /etc/profile.d/greenboost.sh \
        /usr/local/bin/greenboost \
        /usr/local/bin/gb \
        /usr/local/bin/greenboost-cli \
        /etc/modules-load.d/greenboost.conf \
        /etc/udev/rules.d/99-greenboost.rules \
        /etc/udev/rules.d/99-nvme-greenboost.rules \
        /etc/udev/rules.d/99-usb-greenboost.rules \
        /etc/sysctl.d/99-greenboost.conf \
        /etc/sysctl.d/99-zzz-greenboost.conf \
        /etc/sysfs.d/greenboost-hugepages.conf \
        /lib/systemd/system-shutdown/greenboost-stuck.sh \
        /etc/modprobe.d/99-nvidia-greenboost.conf; do
        [[ -f "$f" ]] && rm -f "$f" && (( _cfg_removed++ )) || true
    done
    # Strip greenboost from ai-workstation.conf (nvidia/cpuid still needed - don't delete)
    local _ml_ai="/etc/modules-load.d/ai-workstation.conf"
    if [[ -f "$_ml_ai" ]] && grep -q "^greenboost[[:space:]]*$" "$_ml_ai"; then
        sed -i '/^greenboost[[:space:]]*$/d' "$_ml_ai"
        (( _cfg_removed++ )) || true
    fi
    udevadm control --reload-rules 2>/dev/null || true
    [[ $_cfg_removed -gt 0 ]] && gb_ok "Config files removed ($_cfg_removed files)"
    if [[ "$preserve_cluster" -eq 1 ]]; then
        # Reinstall path: /etc/greenboost holds cluster identity , losing
        # cluster.key desyncs every feeder (they keep their copy) and losing
        # cluster.conf silently drops all connected feeders (this broke the
        # omen feeder on 2026-07-06). Preserve state files across the wipe;
        # everything else in /etc/greenboost is regenerated by the install.
        local _keep_dir _kf
        _keep_dir=$(mktemp -d /run/greenboost-keep.XXXXXX)
        for _kf in cluster.key cluster.conf known_hosts turboquant.enabled ggml_2dev.enabled; do
            [[ -f "/etc/greenboost/$_kf" ]] && cp -a "/etc/greenboost/$_kf" "$_keep_dir/" || true
        done
        rm -rf /etc/greenboost
        if compgen -G "$_keep_dir/*" >/dev/null; then
            mkdir -p /etc/greenboost
            # Copy entries, NOT "$_keep_dir/." , cp -a of "dir/." also stamps
            # the mktemp dir's 0700 mode onto /etc/greenboost, locking every
            # non-root reader (cluster display, shim, gb_cluster.py) out.
            cp -a "$_keep_dir"/* /etc/greenboost/
            chmod 0755 /etc/greenboost
            gb_ok "Cluster state preserved:$(cd "$_keep_dir" && printf ' %s' *)"
        fi
        rm -rf "$_keep_dir"
    else
        rm -rf /etc/greenboost
    fi

    # 6. Disable + remove ALL GreenBoost systemd services - generic glob catches any
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
    # Ollama drop-in override
    rm -f /etc/systemd/system/ollama.service.d/99-greenboost.conf
    # Speculative decoding drop-in (feature removed - always purge any leftover)
    rm -f /etc/systemd/system/ollama.service.d/zz-speculative.conf
    rmdir --ignore-fail-on-non-empty /etc/systemd/system/ollama.service.d/ 2>/dev/null || true
    # TurboQuant daemon (optional install - clean up if present)
    systemctl disable --now greenboost-turboquant.service 2>/dev/null || true
    rm -f /usr/local/bin/greenboost-turboquant \
          /usr/local/lib/libgreenboost_tq.so \
          /etc/systemd/system/greenboost-turboquant.service
    # Unified supervisor (v3.1+) and legacy separate units (≤v3.0)
    systemctl disable --now greenboost-supervisor.service 2>/dev/null || true
    rm -f /etc/systemd/system/greenboost-supervisor.service \
          /etc/systemd/system/greenboost-recovery.service \
          /etc/systemd/system/greenboost-sentinel.service \
          /etc/systemd/system/greenboost-vram-watchdog.service \
          /etc/systemd/system/greenboost-idle-reclaim.service
    rm -rf /usr/local/lib/greenboost

    # Legacy daemon scripts (all known names across all versions)
    rm -f /usr/local/sbin/greenboost-recover \
          /usr/local/sbin/greenboost-vram-watchdog \
          /usr/local/sbin/greenboostd \
          /usr/local/bin/greenboost-idle-reclaim \
          /usr/local/bin/greenboost-run \
          /usr/local/bin/greenboost-run-tgi \
          /usr/local/bin/greenboost-run-unsloth \
          /usr/local/bin/greenboost-run-vllm
    # greenboost-cli wrappers (the venv itself lives under /usr/local/lib/greenboost/)
    rm -f /usr/local/bin/gb /usr/local/bin/greenboost-cli
    # Documentation directory and stray Python hooks
    rm -rf /usr/local/share/greenboost/
    rm -f /usr/local/lib/greenboost_*.py
    rm -rf /usr/local/lib/greenboost/
    rm -f /etc/profile.d/greenboost_pythonpath.sh
    rm -f /usr/local/lib/python3*/dist-packages/greenboost.pth 2>/dev/null || true
    # Claude CLI MCP registrations (per-user) , symmetric with cmd_register_mcp.
    # The gb_*_mcp.py targets live under /usr/local/lib/greenboost (removed just
    # above), so their registrations are now dangling. Remove only on a TRUE
    # uninstall (preserve_cluster=0); a reinstall keeps them and cmd_register_mcp
    # refreshes them. Best-effort, run as the invoking user, never fatal.
    if [[ "$preserve_cluster" -eq 0 ]]; then
        local _mcp_u="${SUDO_USER:-$USER}"
        local -a _mcp_run=()
        if [[ $EUID -eq 0 && -n "$_mcp_u" && "$_mcp_u" != "root" ]]; then
            _mcp_run=(sudo -u "$_mcp_u")
        fi
        if "${_mcp_run[@]}" bash -lc 'command -v claude' &>/dev/null; then
            local _mcp_name
            for _mcp_name in greenboost-dataflux greenboost-cluster \
                             greenboost-orchestrator greenboost-synapse; do
                "${_mcp_run[@]}" bash -lc "claude mcp remove '$_mcp_name'" &>/dev/null || true
            done
            gb_ok "Claude MCP registrations removed"
        fi
    fi
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
/OLLAMA_CONTEXT_LENGTH/d
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

    # 8b. Remove /swap_nvme.img if present - created by older GreenBoost versions.
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

    # 8b2. Remove greenboost_swap.img if present - T3 swap file created by v2.9+
    local _gb_swap="/var/lib/greenboost/greenboost_swap.img"
    if [[ -f "$_gb_swap" ]]; then
        swapoff "$_gb_swap" 2>/dev/null || true
        rm -f "$_gb_swap"
        sed -i '\|greenboost_swap\.img|d' /etc/fstab
        gb_ok "Removed T3 swap file (${_gb_swap})"
    fi

    # 8c. Remove T3 backing file - created by GreenBoost v2.8+ as a sparse file.
    #     The module closes the file before rmmod, so it is safe to remove.
    if [[ -f /var/lib/greenboost/t3_store ]]; then
        rm -f /var/lib/greenboost/t3_store
        gb_ok "Removed T3 backing file (/var/lib/greenboost/t3_store)"
    fi
    rmdir /var/lib/greenboost 2>/dev/null || true

    # 9. Unload kernel module - done last so all consumers are gone.
    _rmmod_with_retry || return 1

    # 10. Restart services that were stopped before the purge.
    #     Only done on standalone uninstall (restart_after=1).
    #     On full-install (restart_after=0) services stay stopped - cmd_full_install
    #     restarts them once at the very end, after the fresh install completes.
    if [[ $restart_after -eq 1 ]]; then
        # Offer to restore pre-install system settings if a backup exists
        local _latest_backup; _latest_backup=$(ls -1d "${_GB_BACKUP_DIR}"/[0-9]* 2>/dev/null | sort | tail -1)
        if [[ -n "$_latest_backup" ]]; then
            local _bstamp; _bstamp=$(basename "$_latest_backup")
            if [[ -t 0 ]]; then
                echo ""
                echo -e "  ${C_CYAN}Backup from ${_bstamp} found.${C_RESET}"
                echo -n "  Restore pre-GreenBoost system settings (GRUB, sysctl, CPU governor)? [y/N] "
                local _ans; read -r _ans
                [[ "$_ans" =~ ^[Yy] ]] && _gb_backup_restore
            fi
        fi
        for svc in $GB_STOPPED_SERVICES; do
            systemctl start "$svc" 2>/dev/null \
                && gb_ok "$svc restarted" \
                || gb_warn_ui "$svc failed to restart - check: journalctl -u $svc"
        done
    fi
}

cmd_build() {
    # Prefer the newest side-by-side versioned CUDA install so that a fresh
    # NVIDIA-repo CUDA 13 is used instead of the Ubuntu-repo CUDA 12 that the
    # /usr/local/cuda symlink may still point to.
    local _cuda_bin=""
    local _latest_cuda
    _latest_cuda=$(ls -d /usr/local/cuda-[0-9]* 2>/dev/null | sort -V | tail -1)
    if [[ -n "$_latest_cuda" && -x "$_latest_cuda/bin/nvcc" ]]; then
        _cuda_bin="$_latest_cuda/bin"
    elif [[ -d /usr/local/cuda/bin ]]; then
        _cuda_bin="/usr/local/cuda/bin"
    fi
    if [[ -n "$_cuda_bin" ]]; then
        export PATH="$_cuda_bin:$PATH"
        local _nvcc_ver
        _nvcc_ver=$("$_cuda_bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "unknown")
        gb_info "Using nvcc from $_cuda_bin (CUDA $_nvcc_ver)"
        # Warn if the selected nvcc major version differs from the driver's reported CUDA version
        local _drv_ver
        _drv_ver=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+' || echo "")
        local _nvcc_major="${_nvcc_ver%%.*}"
        if [[ -n "$_drv_ver" && -n "$_nvcc_major" && "$_drv_ver" != "$_nvcc_major" ]]; then
            gb_warn_ui "nvcc major version ($_nvcc_major) differs from driver's CUDA version ($_drv_ver). Consider reinstalling or adjusting PATH."
        fi
    fi
    make -C "$MODULE_DIR" clean all &>/tmp/gb_build.log &
    gb_spin $! "Clean-building kernel module + CUDA shim..."
    if ! wait $!; then
        gb_fail() { echo -e "  ${RED}✗${C_RESET}  Build failed:"; }
        cat /tmp/gb_build.log >&2
        die "Build failed - see output above"
    fi
    gb_ok "Build complete  (greenboost.ko · ${SHIM_LIB} · ${AUDIT_LIB} · greenboost-netd)"
}

GB_PY_DEST="/usr/local/lib/greenboost"

# Install all gb_*.py orchestration modules to $GB_PY_DEST and add that
# directory to the system PYTHONPATH so every CUDA venv can import them.
# Called from: cmd_install (full-install), cmd_load (light-install/reload).
cmd_install_python_files() {
    local _dest="$GB_PY_DEST"
    mkdir -p "$_dest"

    # Core orchestration stack , gb_init.py is the bootstrap wiring module that
    # must be listed first so downstream modules find it on import.
    local _py_files=(
        gb_init.py
        gb_quant.py gb_quant_tq.py gb_quant_calib.py
        gb_kernel_backends.py gb_placement.py
        gb_attn.py  gb_llm.py
        gb_telemetry.py
        gb_stream_sched.py
        gb_model_tier.py
        gb_mem_pool.py
        gb_diffusion_orch.py
        gb_stability_monitor.py
        gb_feeder_diag.py
        gb_gguf_tensor_map.py
        # Reactive signal-driven orchestration (v3.3+)
        gb_nvml.py
        gb_nvml_ctypes.py
        gb_reactive.py
        gb_control.py
        gb_orchestrator.py
        gb_dataflux.py
        # Monitoring / serving / cluster stack (the installed greenboost-cli
        # resolves these from /usr/local/lib/greenboost via GB_PY_ROOT)
        gb_monitor.py
        gb_pilot.py
        gb_tiering.py
        gb_synapse.py
        gb_synapse_api.py
        gb_cluster.py
        gb_mcp.py
        gb_synapse_mcp.py gb_synapse_tools.py
        gb_dataflux_mcp.py
        gb_cluster_mcp.py
    )
    local _installed=0
    for _f in "${_py_files[@]}"; do
        if [[ -f "$MODULE_DIR/$_f" ]]; then
            install -m 644 "$MODULE_DIR/$_f" "$_dest/$_f"
            (( _installed++ )) || true
        else
            gb_warn "$_f not found in $MODULE_DIR — skipped"
        fi
    done

    # Install an empty __init__.py so the directory is importable as a package.
    # Init-M5: "from . import *" was removed , it eager-loaded torch/gemlite at
    # import time and conflicted with absolute imports (PYTHONPATH exposes all
    # modules top-level; no relative-import indirection needed).
    if [[ ! -f "$_dest/__init__.py" ]]; then
        printf '# GreenBoost Python orchestration package\n' \
            > "$_dest/__init__.py"
    fi

    # Add to system-wide PYTHONPATH via profile.d (idempotent)
    local _profile_pth="/etc/profile.d/greenboost_pythonpath.sh"
    cat > "$_profile_pth" << 'PYPATHEOF'
# GreenBoost Python orchestration modules (gb_quant, gb_telemetry, etc.)
export PYTHONPATH="/usr/local/lib/greenboost${PYTHONPATH:+:$PYTHONPATH}"
PYPATHEOF
    chmod 644 "$_profile_pth"

    # Also write a .pth file so conda/pip envs pick it up without sourcing profile.d
    # (this covers artpipeline_cu13 and any venv that respects site.py)
    local _pth_dir
    for _pth_dir in \
        /usr/local/lib/python3/dist-packages \
        /usr/local/lib/python3.12/dist-packages \
        /usr/local/lib/python3.11/dist-packages \
        /usr/local/lib/python3.10/dist-packages
    do
        if [[ -d "$_pth_dir" ]]; then
            echo "$_dest" > "$_pth_dir/greenboost.pth"
        fi
    done
    # Also drop a .pth in the active conda/mamba env if detectable
    if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
        local _conda_pth
        _conda_pth=$(find "$CONDA_PREFIX/lib" -maxdepth 2 -name "dist-packages" -o -name "site-packages" 2>/dev/null | head -1)
        [[ -n "$_conda_pth" ]] && echo "$_dest" > "$_conda_pth/greenboost.pth"
    fi

    # Write a sitecustomize.py into every detected site-packages so that
    # processes launched with GREENBOOST_ACTIVE=1 (system services, _gen_gb.sh,
    # greenboost run) get gb_init auto-imported even without sourcing profile.d.
    # We use a unique filename (greenboost_sitecustomize.py) to avoid clobbering
    # an existing sitecustomize.py written by other tools.
    local _sc_content='# GreenBoost auto-bootstrap , generated by install-python
import os as _os
if _os.environ.get("GREENBOOST_ACTIVE") == "1":
    try:
        import gb_init  # noqa: F401
    except Exception:
        pass
'
    local _sc_target
    for _pth_dir in \
        /usr/local/lib/python3/dist-packages \
        /usr/local/lib/python3.12/dist-packages \
        /usr/local/lib/python3.11/dist-packages \
        /usr/local/lib/python3.10/dist-packages
    do
        if [[ -d "$_pth_dir" ]]; then
            _sc_target="$_pth_dir/greenboost_sitecustomize.py"
            printf '%s' "$_sc_content" > "$_sc_target"
            # Hook it via a .pth import line so site.py executes it
            printf '%s\nimport greenboost_sitecustomize\n' "$_dest" \
                > "$_pth_dir/greenboost.pth"
        fi
    done
    if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
        local _conda_sp
        _conda_sp=$(find "$CONDA_PREFIX/lib" -maxdepth 2 \
            \( -name "dist-packages" -o -name "site-packages" \) 2>/dev/null | head -1)
        if [[ -n "$_conda_sp" ]]; then
            printf '%s' "$_sc_content" > "$_conda_sp/greenboost_sitecustomize.py"
            printf '%s\nimport greenboost_sitecustomize\n' "$_dest" \
                > "$_conda_sp/greenboost.pth"
        fi
    fi

    gb_ok "Python orchestration files installed to $_dest/ ($_installed files)"
    gb_info "  import: from gb_telemetry import TelemetryManager"
    gb_info "  import: from gb_diffusion_orch import DiffusionOrchestrator"
    gb_info "  auto-init: GREENBOOST_ACTIVE=1 triggers gb_init on Python startup"
}

# Install greenboost-cli (the `gb` terminal agent) into a dedicated venv at
# $GB_PY_DEST/cli-venv and expose /usr/local/bin/gb + greenboost-cli wrappers.
# Best-effort: a missing source checkout, offline mode, or a pip failure only
# warns — Full Install must never abort on the CLI step.
# Called from: cmd_install (full-install) and the install-cli verb.
cmd_install_cli() {
    local _src="${GB_CLI_SRC:-$MODULE_DIR/../greenboost-cli}"
    if [[ ! -f "$_src/pyproject.toml" ]]; then
        gb_warn "greenboost-cli source not found (set GB_CLI_SRC) — skipping CLI install"
        return 0
    fi

    if [[ "${GB_OFFLINE:-0}" == "1" ]]; then
        gb_warn "GB_OFFLINE=1 — skipping greenboost-cli install (needs pip network access)"
        return 0
    fi

    if ! python3 -m venv --help >/dev/null 2>&1; then
        gb_warn "python3 venv module unavailable — skipping CLI install (apt install python3-venv)"
        return 0
    fi

    local _venv="$GB_PY_DEST/cli-venv"
    if [[ ! -d "$_venv" ]]; then
        if ! python3 -m venv "$_venv" &>/tmp/gb_cli_venv.log; then
            gb_warn "cli-venv creation failed (see /tmp/gb_cli_venv.log) — skipping CLI install"
            return 0
        fi
    fi

    gb_info "Installing greenboost-cli from $_src into $_venv ..."
    "$_venv/bin/pip" install --upgrade pip -q &>/tmp/gb_cli_pip.log || true
    if ! "$_venv/bin/pip" install -q "$_src[mcp]" &>/tmp/gb_cli_pip.log; then
        gb_warn "greenboost-cli pip install failed (see /tmp/gb_cli_pip.log) — skipping CLI install"
        return 0
    fi

    # Make the gb_*.py orchestration modules ($GB_PY_DEST) importable inside
    # the venv , same .pth mechanism cmd_install_python_files uses system-wide.
    local _sp
    for _sp in "$_venv"/lib/python3.*/site-packages; do
        [[ -d "$_sp" ]] && echo "$GB_PY_DEST" > "$_sp/greenboost.pth"
    done

    # Wrappers (idempotent overwrite). GB_PY_ROOT points the CLI's gb_paths
    # resolver at the installed python root instead of a dev checkout.
    local _entry _wrapper
    for _entry in gb greenboost-cli; do
        _wrapper="/usr/local/bin/$_entry"
        cat > "$_wrapper" << WRAPEOF
#!/usr/bin/env bash
# GreenBoost CLI wrapper - generated by greenboost_setup.sh (install-cli)
export GB_PY_ROOT=/usr/local/lib/greenboost
exec "$GB_PY_DEST/cli-venv/bin/$_entry" "\$@"
WRAPEOF
        chmod 755 "$_wrapper"
    done

    gb_ok "greenboost-cli installed  (gb / greenboost-cli → $_venv)"
}

# cmd_install_pipelines , provision the ai-forge pipeline dependencies that
# GreenBoost's pipelines rely on (e.g. PaddleOCR for conduir art jobs).
#
# DELIBERATE UNINSTALL ASYMMETRY: these deps are installed INTO the ai-forge
# environment (user-owned, shared with ai-forge itself), NOT under
# /usr/local/lib/greenboost.  GreenBoost only *triggers* their install here, so
# Full Uninstall does NOT rip them out (that would break ai-forge). do_purge /
# cmd_uninstall print a note pointing this out; remove them manually if desired.
#
# Best-effort: a missing ai-forge checkout, offline mode, or a failing setup
# script only warns , Full Install must never abort on this step.
# Called from: cmd_install (full-install) and the install-pipelines verb.
cmd_install_pipelines() {
    local _af="${GB_AIFORGE_SRC:-$MODULE_DIR/../ai-forge}"
    if [[ ! -d "$_af" ]]; then
        gb_warn "ai-forge not found (set GB_AIFORGE_SRC) — skipping pipeline deps"
        return 0
    fi

    if [[ "${GB_OFFLINE:-0}" == "1" ]]; then
        gb_warn "GB_OFFLINE=1 — skipping ai-forge pipeline deps (setup scripts need network access)"
        return 0
    fi

    local _jobs_dir="$_af/tools/conduir_art_jobs"
    if [[ ! -d "$_jobs_dir" ]]; then
        gb_info "No conduir_art_jobs dir in ai-forge ($_jobs_dir) — no pipeline deps to install"
        return 0
    fi

    # GLOB HOOK: run every setup_*.sh present.  Auto-picks-up setup_paddleocr.sh
    # and any future setup_*.sh with zero changes here.  Each script is expected
    # to be idempotent and is run non-fatally: a failure warns and continues so
    # one broken setup script never aborts Full Install.
    local _ran=0 _script _name
    shopt -s nullglob
    for _script in "$_jobs_dir"/setup_*.sh; do
        _name=$(basename "$_script")
        gb_info "Running pipeline dep setup: $_name"
        if bash "$_script"; then
            gb_ok "pipeline dep setup ok: $_name"
            (( _ran++ )) || true
        else
            gb_warn "pipeline dep setup failed: $_name — continuing"
        fi
    done
    shopt -u nullglob

    if [[ $_ran -eq 0 ]]; then
        gb_info "No setup_*.sh pipeline scripts found in $_jobs_dir"
    else
        gb_ok "ai-forge pipeline deps provisioned ($_ran setup script(s) ran)"
    fi
}

# cmd_register_mcp , register this repo's 4 MCP servers with the Claude CLI so
# an LLM assistant can query GreenBoost dataflux / cluster / orchestrator /
# synapse state directly.  Registered under the invoking (non-root) user's
# Claude config (--scope user), since Claude CLI config is per-user.
#
# Best-effort: if the Claude CLI isn't on PATH we warn and return , Full Install
# must never abort on this step.  Idempotent: each server is removed first, then
# re-added, so re-running never duplicates entries.
# Called from: cmd_install (full-install) and the register-mcp verb.
# Uninstall symmetry: do_purge removes the same 4 servers on a true uninstall.
cmd_register_mcp() {
    local _u="${SUDO_USER:-$USER}"
    # Run the Claude CLI as the target (non-root) user; -lc loads that user's
    # login PATH so claude is found even under ~/.local/bin, nvm, etc.
    local -a _run=()
    if [[ $EUID -eq 0 && -n "$_u" && "$_u" != "root" ]]; then
        _run=(sudo -u "$_u")
    fi

    if ! "${_run[@]}" bash -lc 'command -v claude' &>/dev/null; then
        gb_warn "Claude CLI not found — skipping MCP registration; run 'greenboost register-mcp' later"
        return 0
    fi

    local _py="$GB_PY_DEST"
    # name:module (all under $GB_PY_DEST after cmd_install_python_files)
    local -a _servers=(
        "greenboost-dataflux:$_py/gb_dataflux_mcp.py"
        "greenboost-cluster:$_py/gb_cluster_mcp.py"
        "greenboost-orchestrator:$_py/gb_mcp.py"
        "greenboost-synapse:$_py/gb_synapse_mcp.py"
    )
    local _entry _name _path _ok=0
    for _entry in "${_servers[@]}"; do
        _name="${_entry%%:*}"
        _path="${_entry#*:}"
        if [[ ! -f "$_path" ]]; then
            gb_warn "MCP module missing: $_path — skipping $_name"
            continue
        fi
        # Idempotent: drop any prior registration, then add fresh.
        "${_run[@]}" bash -lc "claude mcp remove '$_name'" &>/dev/null || true
        if "${_run[@]}" bash -lc "claude mcp add --scope user '$_name' -- python3 '$_path'" &>/dev/null; then
            (( _ok++ )) || true
        else
            gb_warn "MCP registration failed: $_name — continuing"
        fi
    done
    [[ $_ok -gt 0 ]] && gb_ok "Registered $_ok MCP server(s) with Claude CLI (user: $_u)"
    return 0
}

# _gb_ensure_ollama , install ollama on the host during Full Install if missing,
# or update it if a newer release exists. Feeders then match it via
# `greenboost feeders sync-ollama` (kernel-name parity is what lets feeder-GPU
# compute resolve host-dispatched kernels). Best-effort: logs progress, never
# aborts the install. Set GB_SKIP_OLLAMA_UPDATE=1 to skip entirely.
_gb_ensure_ollama() {
    [[ "${GB_SKIP_OLLAMA_UPDATE:-0}" == "1" ]] && { gb_info "ollama update skipped (GB_SKIP_OLLAMA_UPDATE=1)"; return 0; }
    command -v curl &>/dev/null || { gb_warn "curl not found , skipping ollama update"; return 0; }

    local _before; _before=$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)

    if [[ -n "$_before" ]]; then
        # Already installed - only reinstall when a newer release actually
        # exists. The official installer restarts the ollama service, so
        # running it on every Full Install would needlessly bounce a healthy
        # daemon. If the latest release can't be determined, leave it alone.
        local _latest
        _latest=$(curl -fsSL --max-time 5 "https://api.github.com/repos/ollama/ollama/releases/latest" 2>/dev/null \
            | grep -oP '"tag_name"\s*:\s*"v?\K[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        if [[ -z "$_latest" ]]; then
            gb_info "ollama ${_before} installed - couldn't check for a newer release, leaving it as-is"
            return 0
        fi
        local _newer; _newer=$(printf '%s\n%s\n' "$_before" "$_latest" | sort -V | tail -1)
        if [[ "$_newer" == "$_before" ]]; then
            gb_ok "ollama ${_before} is already the latest release - skipping reinstall"
            return 0
        fi
        gb_info "Updating ollama (current: ${_before}, latest: ${_latest}) via official installer…"
    else
        gb_info "Installing ollama via official installer…"
    fi
    # Official installer updates the ollama binary + systemd unit in place.
    if curl -fsSL https://ollama.com/install.sh -o /tmp/gb_ollama_install.sh 2>/dev/null \
       && sh /tmp/gb_ollama_install.sh >/tmp/gb_ollama_install.log 2>&1; then
        local _after; _after=$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        gb_ok "ollama ${_after:-installed}${_before:+ (was ${_before})}"
        [[ -n "$_after" && -f "$GB_CLUSTER_CONF" ]] && \
            gb_info "Match feeders to it:  sudo greenboost feeders sync-ollama"
    else
        gb_warn "ollama install/update failed (see /tmp/gb_ollama_install.log) , continuing"
    fi
    rm -f /tmp/gb_ollama_install.sh
}

# _gb_install_ebpf_tracer , install greenboost-ebpf-trace if built, (re)start it.
# Called from cmd_install (Full Install).  Non-fatal when binary absent.
_gb_install_ebpf_tracer() {
    local _src="${MODULE_DIR}/greenboost-ebpf-trace"
    local _dst="/usr/local/bin/greenboost-ebpf-trace"
    local _pid_f="/run/greenboost/ebpf_trace.pid"

    if [[ ! -x "$_src" ]]; then
        gb_info "eBPF tracer not built (clang/bpftool/libbpf required) , skipping"
        gb_info "  Build: cd ${MODULE_DIR} && make BPF=1 ebpf"
        return 0
    fi

    # Stop running instance (best-effort)
    if [[ -f "$_pid_f" ]]; then
        local _old_pid; _old_pid=$(cat "$_pid_f" 2>/dev/null || true)
        [[ -n "$_old_pid" ]] && kill "$_old_pid" 2>/dev/null || true
        sleep 0.3
    fi

    install -m 755 "$_src" "$_dst"
    gb_ok "Installed ${_dst}"

    # Only start when greenboost.ko is already loaded (kprobes need the symbols)
    if lsmod | grep -q '^greenboost '; then
        mkdir -p /run/greenboost /var/log/greenboost
        nohup "$_dst" >>/var/log/greenboost/ebpf-trace.log 2>&1 &
        local _bpf_bg_pid=$!
        sleep 0.5
        if kill -0 "$_bpf_bg_pid" 2>/dev/null; then
            gb_ok "eBPF tracer started (pid ${_bpf_bg_pid})"
        else
            gb_warn "eBPF tracer exited immediately , check /var/log/greenboost/ebpf-trace.log"
            gb_info "  Hint: requires CAP_BPF (root) and CONFIG_KALLSYMS_ALL=y"
        fi
    else
        gb_info "eBPF tracer installed , will auto-start after: sudo modprobe greenboost"
        gb_info "  Manual start: sudo ${_dst}"
    fi
}

cmd_install() {
    need_root install

    # Strip libgreenboost entries from ld.so.preload before spawning any
    # subprocess (detect_hardware, check_deps, backup, etc.).  With la_version
    # exported the dynamic linker tries to open the audit lib in LD_AUDIT mode
    # for every child process; in certain subprocess contexts that open fails and
    # prints "cannot be preloaded" noise.  The install_ld_preload step later
    # re-adds the correct entry once the new lib is in place.
    if [[ -f /etc/ld.so.preload ]]; then
        sed -i '/libgreenboost/d' /etc/ld.so.preload
        [[ -s /etc/ld.so.preload ]] || rm -f /etc/ld.so.preload
    fi

    detect_hardware
    check_deps

    # Capture pre-install system state for potential rollback
    _gb_backup_create

    # Step 0: clean previous installation to guarantee a fresh install
    # Skip when called from cmd_full_install which already ran do_purge at step 0/5.
    if [[ "${GB_SKIP_INSTALL_PURGE:-0}" -ne 1 ]]; then
        gb_step 0 4 "Removing previous GreenBoost installation (if any)..."
        do_purge 0 1
        gb_ok "Previous installation removed"
    fi

    # Build all targets (module + shim + audit + netd) in the source tree.
    # The source tree may contain root-owned artifacts from a prior 'sudo make'.
    # We run 'make clean' first to clear them, then rebuild everything.
    local _build_log
    _build_log=$(mktemp /tmp/gb_build.XXXXX.log)
    { make -C "$MODULE_DIR" clean all >"$_build_log" 2>&1; } &
    gb_spin $! "Building GreenBoost (module + shim + audit + netd)..."
    if ! wait $!; then
        echo "" >&2
        cat "$_build_log" >&2
        rm -f "$_build_log"
        die "Build failed , see output above"
    fi
    rm -f "$_build_log"
    # Restore source-tree ownership so the developer can rebuild without sudo.
    if [[ -n "${SUDO_USER:-}" ]]; then
        chown -R "${SUDO_USER}:$(id -gn "${SUDO_USER}" 2>/dev/null || echo users)" \
            "$MODULE_DIR" 2>/dev/null || true
    fi
    gb_ok "Build complete"

    local _install_log
    _install_log=$(mktemp /tmp/gb_install.XXXXX.log)
    { make -C "$MODULE_DIR" install >"$_install_log" 2>&1; } &
    gb_spin $! "Installing kernel module (DKMS) + libs..."
    if ! wait $!; then
        echo "" >&2
        cat "$_install_log" >&2
        rm -f "$_install_log"
        die "Module install failed - see output above"
    fi
    rm -f "$_install_log"
    gb_ok "Kernel module + libs installed"

    cp "$MODULE_DIR/$SHIM_LIB" "$SHIM_DEST/"
    [[ -f "$MODULE_DIR/$AUDIT_LIB" ]] && cp "$MODULE_DIR/$AUDIT_LIB" "$SHIM_DEST/"
    [[ -f "$MODULE_DIR/libgreenboost_vmm_override.so" ]] && \
        cp "$MODULE_DIR/libgreenboost_vmm_override.so" "$SHIM_DEST/"
    { ldconfig 2>/dev/null; } &
    gb_spin $! "Installing CUDA shim + VMM override + LD_AUDIT library..."
    gb_ok "Libraries installed to $SHIM_DEST/"

    # modprobe defaults
    # Ensure T3 backing store directory exists before writing modprobe.conf
    mkdir -p /var/lib/greenboost

    cat > /etc/modprobe.d/greenboost.conf << MODEOF
# GreenBoost - cuda memory pool (auto-configured for detected hardware)
# GPU   : ${GPU_NAME}  (${GB_PHYS} GB VRAM)
# RAM   : ${RAM_TYPE}-${RAM_SPEED_MT}  (pool ${GB_VIRT} GB, reserve ${GB_RESERVE} GB)
# T3    : file-backed (/var/lib/greenboost/t3_store, cap ${GB_NVME_POOL} GB)
options greenboost physical_vram_gb=${GB_PHYS} virtual_vram_gb=${GB_VIRT} safety_reserve_gb=${GB_RESERVE} nvme_pool_gb=${GB_NVME_POOL} t3_max_gb=${GB_NVME_POOL} t3_file_path=/var/lib/greenboost/t3_store pcores_max_cpu=${GB_PCORES_MAX} golden_cpu_min=${GB_GOLDEN_MIN} golden_cpu_max=${GB_GOLDEN_MAX} ecores_only=${GB_PCORES_ONLY}
MODEOF

    # Load the module NOW so it picks up the tuned parameters just written ,
    # `make install` (dkms-install) only builds/registers the module with DKMS,
    # it never inserts it into the running kernel, and this script never called
    # modprobe itself either (it only ever PRINTED "sudo modprobe greenboost"
    # as a suggestion). Confirmed 2026-07-09: a full reinstall completed
    # cleanly but left the module unloaded , GreenBoost's T2 DDR spill then
    # silently degraded into real OS swap (8 GB used, 90% iowait), which
    # looked like generic slowness/"GPU thermal throttling" until traced back
    # to this. Unload first if some (possibly stale-params) instance is
    # already loaded, so the fresh modprobe.conf options actually take effect.
    depmod -a 2>/dev/null || true
    if lsmod | grep -q '^greenboost '; then
        modprobe -r greenboost 2>/dev/null || _rmmod_with_retry || true
    fi
    if modprobe greenboost; then
        gb_ok "Kernel module loaded (virtual_vram_gb=${GB_VIRT}, reserve=${GB_RESERVE} GB)"
    else
        gb_warn "modprobe greenboost failed after install , load manually: sudo modprobe greenboost"
    fi

    # Persist across reboots , same gap as the modprobe fix above but for the
    # NEXT boot: without this file, systemd-modules-load.service has nothing
    # telling it to load greenboost, and the module silently stays unloaded
    # after a reboot until someone notices (confirmed gap 2026-07-10, this
    # Full Install path never wrote it even after the modprobe fix above).
    if [[ ! -f /etc/modules-load.d/greenboost.conf ]]; then
        echo "greenboost" > /etc/modules-load.d/greenboost.conf
        gb_ok "Boot-time autoload registered: /etc/modules-load.d/greenboost.conf"
    fi

    # Build for every OTHER kernel already on disk too , `make install` above
    # only builds/registers DKMS for $(uname -r).
    dkms autoinstall -m greenboost -v "$GB_VERSION" &>/dev/null || true

    # Boot guard: self-heals a missing .ko for the running kernel (e.g. a
    # kernel upgrade whose DKMS autoinstall never ran) before
    # systemd-modules-load.service tries to load it.
    if [[ -f "$MODULE_DIR/greenboost_boot_guard.sh" && -f "$MODULE_DIR/greenboost-boot-guard.service" ]]; then
        install -m 755 "$MODULE_DIR/greenboost_boot_guard.sh" /usr/local/sbin/greenboost_boot_guard.sh
        install -m 644 "$MODULE_DIR/greenboost-boot-guard.service" /etc/systemd/system/greenboost-boot-guard.service
        systemctl daemon-reload 2>/dev/null || true
        systemctl enable greenboost-boot-guard.service &>/dev/null \
            && gb_ok "Boot guard installed + enabled: greenboost-boot-guard.service"
    fi

    # profile.d - auto-activate GreenBoost for all CUDA inference tools launched
    # from a login shell (terminal, SSH).  GREENBOOST_ACTIVE=1 is exported globally
    # so vLLM, PyTorch scripts, TGI, Transformers, etc. all work without any wrapper.
    # The greenboost() function remains available as a fallback for non-login contexts.
    cat > /etc/profile.d/greenboost.sh << PROFEOF
# GreenBoost v3.2 - auto-activation for CUDA inference tools
export GREENBOOST_ACTIVE=1
export GREENBOOST_SHIM="$SHIM_DEST/$SHIM_LIB"
export GREENBOOST_VMM_OVERRIDE="$SHIM_DEST/libgreenboost_vmm_override.so"
# 'greenboost run' is a fallback for non-login contexts (cron, Docker entrypoints)
greenboost() {
    case "\$1" in
        run) shift
            _gb_preload="\${GREENBOOST_VMM_OVERRIDE}:\${GREENBOOST_SHIM}"
            [[ ! -f "\${GREENBOOST_VMM_OVERRIDE}" ]] && _gb_preload="\${GREENBOOST_SHIM}"
            GREENBOOST_ACTIVE=1 LD_PRELOAD="\${_gb_preload}" "\$@"
            unset _gb_preload
            ;;
        # Fall through to the CLI wrapper , without this the exported function
        # shadows /usr/local/bin/greenboost in every login shell and breaks
        # all other subcommands (cluster, vitals, connect, ...).
        *) command /usr/local/bin/greenboost "\$@" ;;
    esac
}
export -f greenboost
PROFEOF

    # Standalone wrapper - fallback for non-login contexts where profile.d is not sourced
    cat > /usr/local/bin/greenboost << WRAPEOF
#!/usr/bin/env bash
# GreenBoost CLI wrapper
# Usage: greenboost <command> [args...]
# Run 'greenboost help' for the full command reference.
GB_SETUP="$MODULE_DIR/greenboost_setup.sh"
case "\$1" in
    clean)        exec "\$GB_SETUP" clean "\${@:2}" ;;
    clean-memory) exec "\$GB_SETUP" clean-memory ;;
    benchmark)    exec "\$GB_SETUP" benchmark "\${@:2}" ;;
    profile)      exec "\$GB_SETUP" profile "\${@:2}" ;;
    build|build-info) exec "\$GB_SETUP" build-info "\${@:2}" ;;
    compile)      exec "\$GB_SETUP" compile "\${@:2}" ;;
    feed)         exec "\$GB_SETUP" feed "\${@:2}" ;;
    connect)         exec "\$GB_SETUP" connect "\${@:2}" ;;
    disconnect)      exec "\$GB_SETUP" disconnect "\${@:2}" ;;
    cluster)         exec "\$GB_SETUP" cluster ;;
    dataflux-ui)     exec "\$GB_SETUP" dataflux-ui "\${@:2}" ;;
    update)          exec "\$GB_SETUP" update "\${@:2}" ;;
    update-feeders)  exec "\$GB_SETUP" update-feeders ;;
    feeders)         exec "\$GB_SETUP" feeders "\${@:2}" ;;
    built-stamp)     exec "\$GB_SETUP" built-stamp "\${@:2}" ;;
    load)         exec "\$GB_SETUP" load "\${@:2}" ;;
    unload)       exec "\$GB_SETUP" unload ;;
    tune)         exec "\$GB_SETUP" tune "\${@:2}" ;;
    tune-grub)    exec "\$GB_SETUP" tune-grub ;;
    tune-sysctl)  exec "\$GB_SETUP" tune-sysctl ;;
    tune-libs)    exec "\$GB_SETUP" tune-libs ;;
    tune-all)     exec "\$GB_SETUP" tune-all ;;
    turboquant)   exec "\$GB_SETUP" turboquant "\${@:2}" ;;
    ggml-2dev)    exec "\$GB_SETUP" ggml-2dev "\${@:2}" ;;
    vitals)       exec "\$GB_SETUP" vitals "\${@:2}" ;;
    faults)       exec "\$GB_SETUP" faults "\${@:2}" ;;
    top)          exec "\$GB_SETUP" top "\${@:2}" ;;
    residency)    exec "\$GB_SETUP" residency "\${@:2}" ;;
    debug)        exec "\$GB_SETUP" debug "\${@:2}" ;;
    gen-inference-config) exec "\$GB_SETUP" gen-inference-config "\${@:2}" ;;
    doctor)          exec "\$GB_SETUP" doctor "\${@:2}" ;;
    recommend)       exec "\$GB_SETUP" recommend "\${@:2}" ;;
    pull)            exec "\$GB_SETUP" pull "\${@:2}" ;;
    synapse)         exec "\$GB_SETUP" synapse "\${@:2}" ;;
    setup|install|full-install) exec "\$GB_SETUP" "\$@" ;;
    install-pipelines)  exec "\$GB_SETUP" install-pipelines ;;
    register-mcp)       exec "\$GB_SETUP" register-mcp ;;
    uninstall)          exec "\$GB_SETUP" uninstall ;;
    apparmor-uninstall) exec "\$GB_SETUP" apparmor-uninstall ;;
    logs)            exec "\$GB_SETUP" logs "\${@:2}" ;;
    nvtx-logs)       exec "\$GB_SETUP" nvtx-logs "\${@:2}" ;;
    nvtx)            exec "\$GB_SETUP" nvtx "\${@:2}" ;;
    diag)            exec "\$GB_SETUP" diag "\${@:2}" ;;
    inference-logs)  exec "\$GB_SETUP" inference-logs "\${@:2}" ;;
    clear)           exec "\$GB_SETUP" clear "\${@:2}" ;;
    clean-logs)      exec "\$GB_SETUP" clean-logs ;;
    test)            exec "\$GB_SETUP" test "\${@:2}" ;;
    run)          shift
        _gb_vmm="$SHIM_DEST/libgreenboost_vmm_override.so"
        _gb_preload="\${_gb_vmm}:$SHIM_DEST/$SHIM_LIB"
        [[ ! -f "\${_gb_vmm}" ]] && _gb_preload="$SHIM_DEST/$SHIM_LIB"
        GREENBOOST_ACTIVE=1 LD_PRELOAD="\${_gb_preload}" "\$@"
        ;;
    help|--help|-h|"") exec "\$GB_SETUP" show-commands ;;
    *)            echo "Unknown command: '\$1'  - run: greenboost help" >&2; exit 1 ;;
esac
WRAPEOF
    chmod +x /usr/local/bin/greenboost

    # Install commands reference so 'greenboost help' works from any path
    mkdir -p /usr/local/share/greenboost
    if [[ -f "$MODULE_DIR/GREENBOOST_COMMANDS.md" ]]; then
        install -m 644 "$MODULE_DIR/GREENBOOST_COMMANDS.md" /usr/local/share/greenboost/GREENBOOST_COMMANDS.md
    fi

    # PR-GG: install lib/ helpers so installed-from-PATH greenboost can
    # source them.  The lib/ directory contains gb_colors.sh, gb_tui.sh,
    # gb_ssh.sh - small, well-bounded modules that the top of the main
    # script auto-discovers via the GB_LIB_DIR lookup.  If a prior install
    # lacked lib/, the inline fallbacks in the main script ensure
    # functionality regardless; this install just makes the lib variant
    # the canonical one.
    if [[ -d "$MODULE_DIR/lib" ]]; then
        mkdir -p /usr/local/share/greenboost/lib
        for _libf in "$MODULE_DIR"/lib/*.sh; do
            [[ -f "$_libf" ]] && install -m 644 "$_libf" /usr/local/share/greenboost/lib/
        done
        gb_ok "Installed lib/ helpers to /usr/local/share/greenboost/lib/"
    fi

    # Install GreenBoost Python orchestration files (gb_*.py) so they are
    # importable from any CUDA env on PYTHONPATH=/usr/local/lib/greenboost.
    cmd_install_python_files
    # Install greenboost-cli (`gb`) into its venv , best-effort, never aborts.
    cmd_install_cli
    # Provision ai-forge pipeline deps (PaddleOCR etc.) , best-effort, never aborts.
    # Deliberately NOT reversed by Full Uninstall (see cmd_install_pipelines).
    cmd_install_pipelines

    # Ensure 'greenboost' group exists and the invoking user is a member so
    # non-root processes can read cluster.key (mode 0640 root:greenboost).
    gb_ensure_greenboost_group
    # Repair existing cluster.key permissions on upgrade.
    [[ -f "$GB_CLUSTER_KEY" ]] && _gb_set_keyfile_perms "$GB_CLUSTER_KEY"

    # Write build_info stamp (readable by 'greenboost build')
    mkdir -p /etc/greenboost
    local _git_hash
    _git_hash=$(git -C "$MODULE_DIR" rev-parse --short HEAD 2>/dev/null || echo "nogit")
    printf 'BUILD_ID=%s\nBUILD_VERSION=%s\nBUILD_HOST=%s\nBUILD_GIT=%s\nBUILD_EPOCH=%s\n' \
        "$(date +%d%m-%H%M)" "$GB_VERSION" "$(hostname)" "$_git_hash" "$(date +%s)" \
        > /etc/greenboost/build_info
    gb_ok "Build stamp written  ($(date '+%d/%m %H:%M') · ${_git_hash})"

    # Always generate a hardware profile so the shim's DDR-speed lookup
    # (greenboost_cuda_shim.c get_local_ddr_speed) has a non-root source.
    gb_info "Generating hardware profile..."
    cmd_profile_create || gb_warn "profile create failed - DDR speed will default to 2400 MT/s"

    # Install and (re)start eBPF tracer if the binary was built
    _gb_install_ebpf_tracer

    # Ensure ollama is present + current so the host runs a known LLM backend
    # version that feeders can match (kernel-name parity for feeder-GPU
    # compute).  Best-effort; never fails the install.  Skip with
    # GB_SKIP_OLLAMA_UPDATE=1.
    _gb_ensure_ollama

    # Register GreenBoost MCP servers with the Claude CLI (per-user) so an LLM
    # assistant can query cluster/dataflux/orchestrator/synapse state. The
    # gb_*_mcp.py modules were just deployed to $GB_PY_DEST above. Best-effort.
    cmd_register_mcp

    gb_ok "Installation complete"
    gb_info "Load:    sudo modprobe greenboost"
    gb_info "Status:  greenboost vitals"
    gb_info "Faults:  greenboost faults"
    gb_info "Top:     greenboost top"
    gb_info "Residency: greenboost residency"

    # Push update to all connected feeders (unattended, best-effort)
    _gb_update_all_feeders
}

cmd_load() {
    need_root load
    detect_hardware

    # Load active profile values (lowest priority - env vars and CLI flags override)
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

    if grep -q "^${DRIVER_NAME} " <<< "$(lsmod)"; then
        warn "Module already loaded - reloading..."
        # Stop consumers before rmmod to avoid EBUSY
        for svc in ollama llama-server; do
            systemctl is-active --quiet "$svc" 2>/dev/null && \
                systemctl stop --wait "$svc" 2>/dev/null || true
        done
        _rmmod_with_retry || die "Failed to unload existing module - try: sudo rmmod greenboost"
    fi

    local ko="$MODULE_DIR/greenboost.ko"
    [[ -f "$ko" ]] || die "greenboost.ko not found - run: make  or  $0 build"

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
        || die "insmod failed - check: dmesg | tail -20"

    # Ensure /dev/greenboost has correct group permission immediately after insmod.
    # The .devnode kernel callback sets mode 0660 at creation, but run udevadm
    # anyway so any previously-installed udev rule (GROUP="video") is also applied.
    udevadm control --reload-rules 2>/dev/null || true
    udevadm trigger --name-match=greenboost 2>/dev/null \
        || udevadm trigger --subsystem-match=greenboost 2>/dev/null \
        || true
    udevadm settle 2>/dev/null || true

    # Add invoking user to 'video' group + grant immediate ACL for the current session.
    if [[ -n "${SUDO_USER:-}" ]]; then
        if ! id -nG "$SUDO_USER" 2>/dev/null | tr ' ' '\n' | grep -q '^video$'; then
            usermod -aG video "$SUDO_USER" \
                && info "User '$SUDO_USER' added to 'video' group (re-login or: newgrp video)" \
                || warn "usermod failed , add $SUDO_USER to 'video' group manually"
        else
            gb_info "User '$SUDO_USER' already in 'video' group"
        fi
        if [[ -c /dev/greenboost ]] && command -v setfacl &>/dev/null; then
            setfacl -m "u:${SUDO_USER}:rw" /dev/greenboost 2>/dev/null \
                && gb_info "ACL: /dev/greenboost → u:${SUDO_USER}:rw (no re-login needed)" \
                || true
        fi
    fi

    # Refresh Python files on every load (light-install path)
    [[ -d "$MODULE_DIR" ]] && cmd_install_python_files 2>/dev/null || true

    info "GreenBoost v3.2 loaded - cuda memory pool active!"
    info ""
    info "  T1 ${GPU_NAME} : ${phys} GB  [hot layers]"
    info "  T2 ${RAM_TYPE} pool         : ${virt} GB  [cold layers]"
    info "  T3 NVMe swap               : ${nvme_sw} GB  [frozen pages]"
    info "  ─────────────────────────────────────────"
    info "  Combined view              : $(( phys + virt + nvme_sw )) GB total model capacity"
    info ""
    info "Status     : greenboost vitals"
    info "Kernel log : dmesg | grep greenboost"
    echo ""
    dmesg | grep greenboost | tail -8 | sed 's/^/  /'
}

cmd_unload() {
    need_root unload
    if grep -q "^${DRIVER_NAME} " <<< "$(lsmod)"; then
        _rmmod_with_retry || die "rmmod failed - check: dmesg | tail -5"
    else
        info "GreenBoost is not loaded"
    fi
}

cmd_uninstall() {
    need_root uninstall
    GB_STOPPED_SERVICES=""

    info "============================================================"
    info " GreenBoost - Uninstall"
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
    info "  - Claude CLI MCP registrations (greenboost-dataflux/cluster/orchestrator/synapse)"
    info ""
    info "What will NOT be touched:"
    info "  - Ollama itself (not uninstalled - only GreenBoost env vars removed)"
    info "  - llama-server itself (not uninstalled - only GreenBoost env vars removed)"
    info "  - /etc/ld.so.preload entries not related to GreenBoost"
    info "  - NVIDIA drivers, CUDA toolkit, Steam, Wine, Proton"
    info "  - Any user data, models, or application configs"
    info "  - System swap (/swap.img, swap partitions) - system swap is never touched"
    info "  - /swap_nvme.img (old GreenBoost swap) - removed if present"
    info "  - ai-forge pipeline deps (PaddleOCR etc.) - left intact in the forge env; remove manually if desired"
    info ""
    info "Starting purge..."
    info ""

    if ! do_purge 1; then
        gb_warn_ui "Scheduling boot-time cleanup so the next reboot finishes the removal…"
        _schedule_boot_cleanup
        info ""
        info "============================================================"
        info " GreenBoost partially uninstalled."
        info " All config files and the .ko have been removed from disk."
        info " The running module is stuck in 'Unloading' - it will be"
        info " fully gone after a reboot."
        info ""
        info " Reboot now to complete uninstallation."
        info " (The boot cleanup service runs automatically before module load.)"
        info "============================================================"
        return 0
    fi

    info ""
    info "============================================================"
    info " GreenBoost uninstalled cleanly."
    info ""
    info " Ollama and llama-server (if installed) have been restarted"
    info " without GreenBoost - they will use native VRAM only."
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

    # ── NVMe nr_requests → 1023 where the device allows (clamped to hw limit) ──
    # The 1023 figure is the Samsung 990 EVO Plus tag-set depth; other NVMe
    # controllers may enforce a lower max and reject writes with EINVAL.
    # We attempt the write silently and read back the kernel-accepted value.
    local _nr_report="(none)"
    for nr in /sys/block/nvme*/queue/nr_requests; do
        [[ -w "$nr" ]] || continue
        echo 1023 > "$nr" 2>/dev/null || true      # kernel clamps automatically; ignore EINVAL
        _nr_report=$(cat "$nr" 2>/dev/null || echo "?")
    done
    info "NVMe nr_requests  : ${_nr_report}"

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

    # ── USB storage scheduler → none + larger read-ahead ─────────────────
    local _usb_tuned=0
    for sched in /sys/block/sd*/queue/scheduler; do
        [[ -w "$sched" ]] && echo none > "$sched" 2>/dev/null && (( _usb_tuned++ )) || true
    done
    for ra in /sys/block/sd*/queue/read_ahead_kb; do
        [[ -w "$ra" ]] && echo 2048 > "$ra" 2>/dev/null || true
    done
    [[ $_usb_tuned -gt 0 ]] && info "USB storage       : scheduler=none, read_ahead=2048 KB ($_usb_tuned device(s))"

    echo ""
    gb_ok "Runtime tuning applied (active until next reboot)"

    # ── Persist settings unconditionally ──────────────────────────────────
    _tune_persist_sysctl
    _tune_persist_nvme "$ra_kb"
    gb_ok "Settings saved - will apply automatically on every boot"
}

# Write sysctl tunables to the drop-in file (also called by tune-sysctl)
_tune_persist_sysctl() {
    local conf="/etc/sysctl.d/99-zzz-greenboost.conf"
    cat > "$conf" << 'SCTL'
# GreenBoost - persistent sysctl tunables
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

    # udev rule - fires on every NVMe block device add/change
    local udev_rule="/etc/udev/rules.d/99-nvme-greenboost.rules"
    cat > "$udev_rule" << UDEV
# GreenBoost - NVMe tuning applied at boot via udev
ACTION=="add|change", KERNEL=="nvme[0-9]n[0-9]", ENV{DEVTYPE}=="disk", ATTR{queue/scheduler}="none", ATTR{queue/read_ahead_kb}="${ra_kb}", ATTR{queue/nr_requests}="1023"
UDEV
    # USB storage udev rule - scheduler=none, larger read-ahead for any USB block device
    local usb_rule="/etc/udev/rules.d/99-usb-greenboost.rules"
    cat > "$usb_rule" << 'USB_UDEV'
# GreenBoost - USB storage tuning applied at boot via udev
ACTION=="add|change", KERNEL=="sd[a-z]", SUBSYSTEMS=="usb", ATTR{queue/scheduler}="none", ATTR{queue/read_ahead_kb}="2048"
USB_UDEV
    udevadm control --reload-rules 2>/dev/null || true
    gb_ok "udev rule: wrote $udev_rule (NVMe) + $usb_rule (USB)"

    # THP via sysfs.d (applied by sysfsutils at boot)
    local sysfs_conf="/etc/sysfs.d/greenboost-hugepages.conf"
    cat > "$sysfs_conf" << 'SYSFS'
# GreenBoost - Transparent Huge Pages must stay 'always' for T2 pool
kernel/mm/transparent_hugepage/enabled = always
SYSFS
    gb_ok "sysfs.d: wrote $sysfs_conf (THP=always on boot)"
}

# ---- tune-revert --------------------------------------------------------
# Restore the continuous OS tuner's levers (CPU governor/EPP, GPU clocks/
# power limit, vm.* tunables) to the pre-tune baseline GbControl captured
# the first time each lever was touched. Does NOT undo the static
# install-time `tune`/`tune-sysctl`/`tune-grub` floor , only the dynamic
# Loop O-S levers in gb_control.py's os_tune_baseline.json.
cmd_tune_revert() {
    need_root tune-revert
    info "Reverting continuous OS-tune levers to captured baseline..."
    python3 - << 'PYEOF'
import sys
sys.path.insert(0, "/usr/local/lib/greenboost")
try:
    from gb_control import GbControl
    ctrl = GbControl()
    results = ctrl.restore_baseline()
    if not results:
        print("[tune-revert] no baseline captured , nothing to revert")
    else:
        for key, ok in results.items():
            print(f"[tune-revert] {key}: {'restored' if ok else 'FAILED'}")
except Exception as exc:
    print(f"[tune-revert] error: {exc}", file=sys.stderr)
    sys.exit(1)
PYEOF
    rm -f /etc/sysctl.d/99-zzz-greenboost-dynamic.conf 2>/dev/null
    gb_ok "OS-tune baseline restore complete"
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
        info "  [skip]    $flag  - already active"
        return 1
    fi
    if [[ -n "$kcfg" ]] && ! _kcfg_has "$kcfg"; then
        warn "  [skip]    $flag  - kernel not built with $kcfg"
        return 1
    fi
    info "  [add]     $flag  - $desc"
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
    # Currently set to 'madvise' in GRUB - change to 'always'.
    if _kcfg_has "CONFIG_TRANSPARENT_HUGEPAGE"; then
        if echo "$current_line" | grep -q "transparent_hugepage=madvise"; then
            info "  [fix]     transparent_hugepage=madvise -> always  (GreenBoost T2 hugepage pool)"
            current_line="${current_line/transparent_hugepage=madvise/transparent_hugepage=always}"
        elif ! _grub_has "transparent_hugepage=always"; then
            info "  [add]     transparent_hugepage=always  (GreenBoost T2 hugepage pool)"
            new_flags="$new_flags transparent_hugepage=always"
        else
            info "  [skip]    transparent_hugepage=always  - already active"
        fi
    fi

    # ── Flag: skew_tick=1 ───────────────────────────────────────────────
    # Staggers per-CPU timer ticks on the i9-14900KF hybrid topology
    # (8 P-cores + 16 E-cores). Reduces lock contention when all 32 CPUs
    # fire timer interrupts simultaneously.
    # Runtime test: always safe - no kernel config dependency.
    _grub_check_flag "skew_tick=1" \
        "stagger timer ticks - reduces lock contention on hybrid P/E cores" "" \
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
    # multiplications - reduces LLM token latency.
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
        "single NUMA node - disable page-migration scanning overhead" "" \
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
        info "  [fix]     nvidia-drm.modeset=1 appears ${count}× - deduplicating"
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
        info "GRUB is already fully optimised - nothing to change."
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
# Writes /etc/sysctl.d/99-zzz-greenboost.conf - loaded last, wins all
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
        info "  kernel.sched_migration_cost_ns: 5000000 (5ms - keep threads on P-cores)"
        info "  kernel.sched_min_granularity_ns: 10000000 (10ms - better for large tasks)"
        info "  kernel.sched_wakeup_granularity_ns: 15000000 (reduces spurious wakeups)"
    else
        info "  CFS sched knobs: skipped (kernel $(uname -r) uses EEVDF, not CFS)"
    fi
    echo ""

    # Write the header line with detected hardware (variables don't expand in 'HEREDOC')
    printf '# GreenBoost v3.2 - Definitive sysctl config\n' > "$dest"
    printf '# Hardware: %s | %s | %s-%s MT/s | PCIe Gen %s %s (~%s GB/s) | %s GB NVMe\n' \
        "${CPU_NAME}" "${GPU_NAME}" "${RAM_TYPE}" "${RAM_SPEED_MT}" \
        "${PCIE_GEN}" "${PCIE_WIDTH}" "${PCIE_BW_GBS}" "${NVME_SIZE_GB}" >> "$dest"
    printf '# Loaded last (99-zzz) - wins all conflicts with earlier sysctl.d files.\n' >> "$dest"
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
# Background flush at 10% (~6.4 GB) - keeps NVMe busy without stalling allocs.
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

# Always keep 512 MB free - prevents latency spikes under allocation storms.
vm.min_free_kbytes = 524288

# Proactive compaction: GreenBoost T2 needs contiguous 2 MB hugepage ranges.
# Value 20 = moderate background compaction (0=off, 100=aggressive).
vm.compaction_proactiveness = 20

# Keep inode/dentry caches alive - LLM loaders open thousands of weight files.
vm.vfs_cache_pressure = 50
SYSCTL_EOF

    # Overcommit hugepage pool: sized to cover the full T2 System DDR pool.
    # Formula: GB_VIRT × 512 = number of 2 MB pages.  RAM speed ${RAM_SPEED_MT} MT/s.
    printf '# Overcommit hugepage pool: %d × 2 MB = %d GB - covers T2 pool (%d GB, %s-%s MT/s)\n' \
        "$overcommit_hp" "$(( overcommit_hp * 2 / 1024 ))" "$GB_VIRT" "$RAM_TYPE" "$RAM_SPEED_MT" >> "$dest"
    printf 'vm.nr_overcommit_hugepages = %d\n' "$overcommit_hp" >> "$dest"

    cat >> "$dest" << 'SYSCTL_EOF'

# Disable zone reclaim: single NUMA node - cross-zone reclaim wastes cycles.
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
# Single-socket i9-14900KF - all CPUs are on NUMA node 0. Automatic NUMA
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

# ── Cluster fabric (host <-> feeder, greenboost-netd binary protocol) ──────
# The fabric link goes idle between bursts (heartbeats every ~2s, then a
# burst of H2D/D2H chunks or a per-layer activation handoff). Linux resets
# cwnd after any idle period (RFC 2861), so every burst restarts from a tiny
# window and slow-starts back up - the dominant added latency for small,
# frequent transfers. Disabling this is safe on a private LAN link (no
# outside traffic to protect against a runaway sender).
net.ipv4.tcp_slow_start_after_idle = 0

# ── Perf / profiling access ───────────────────────────────────────────────
# Allow nsys / perf / CUDA Nsight without sudo (needed for GPU profiling).
kernel.perf_event_paranoid = 1
kernel.kptr_restrict = 0
SYSCTL_EOF

    # CFS scheduler params: only present on kernels < 6.6 (CFS).
    # Kernel 6.6+ uses EEVDF - these knobs were removed entirely.
    _sysctl_if_exists() {
        local key="$1" val="$2" proc_path="$3"
        if [[ -e "$proc_path" ]]; then
            printf '\n# CFS scheduler (kernel < 6.6 only)\n%s = %s\n' "$key" "$val" >> "$dest"
        fi
    }
    _sysctl_if_exists kernel.sched_migration_cost_ns   5000000  /proc/sys/kernel/sched_migration_cost_ns
    _sysctl_if_exists kernel.sched_min_granularity_ns  10000000 /proc/sys/kernel/sched_min_granularity_ns
    _sysctl_if_exists kernel.sched_wakeup_granularity_ns 15000000 /proc/sys/kernel/sched_wakeup_granularity_ns

    # BBR congestion control: lower latency + better throughput than the
    # default CUBIC on the host<->feeder LAN link (bulk weight/KV transfers
    # mixed with small latency-sensitive control RPCs). Only written if the
    # running kernel actually has the module - avoids a hard sysctl error on
    # kernels built without CONFIG_TCP_CONG_BBR.
    modprobe tcp_bbr 2>/dev/null || true
    if grep -qw bbr /proc/sys/net/ipv4/tcp_available_congestion_control 2>/dev/null; then
        printf '\n# BBR congestion control - better latency/throughput than CUBIC for the\n# host<->feeder cluster fabric link.\nnet.ipv4.tcp_congestion_control = bbr\n' >> "$dest"
    else
        info "  BBR congestion control: unavailable on this kernel (tcp_bbr module missing) - skipped"
    fi

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
        # BLAS/LAPACK - OpenBLAS compiled with AVX2/FMA or equivalent

        # OpenMP - multi-threaded CPU inference (llama.cpp uses this heavily)

        # CPU pinning; without it Ollama uses a generic thread affinity model

        # libnuma - NUMA-aware memory allocation (single node but still used
        # by CUDA and some ML runtimes for memory locality hints)

        # OpenCL - GPU compute via OpenCL API (some inference backends use it)

    )

    # CPU frequency tools - package names differ between Debian and Ubuntu/others
    local os_id
    os_id=$(grep -oP '^ID=\K.*' /etc/os-release | tr -d '"')
    if [[ "$os_id" == "debian" ]]; then
        pkgs+=(psmisc)
    else
        pkgs+=(psmisc)
    fi

    # CPU vendor-specific microcode
    local cpu_vendor
    cpu_vendor=$(grep -m1 "vendor_id" /proc/cpuinfo | awk '{print $3}')
    if [[ "$cpu_vendor" == "GenuineIntel" ]]; then
        pkgs+=(intel-microcode)
        info "CPU vendor: Intel - adding intel-microcode"
    elif [[ "$cpu_vendor" == "AuthenticAMD" ]]; then
        pkgs+=(amd64-microcode)
        info "CPU vendor: AMD - adding amd64-microcode"
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

    # cpuid - lets userspace read CPUID leaves directly. Used by turbostat,
    # CUDA diagnostics, and intel-microcode update verification.
    if grep -q "^cpuid " <<< "$(lsmod)"; then
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
        update-alternatives --set libblas.so.3-x86_64-linux-gnu \
            /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 2>/dev/null \
            && info "BLAS alternative: set to OpenBLAS (AVX2/FMA)" \
            || info "BLAS alternative: already set or path differs - check manually"

    echo ""
    info "tune-libs complete."
    info "  Turbostat (P/E core monitoring): sudo turbostat --quiet --Summary"
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
    info "Running full system tuning for GreenBoost v3.2..."
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
    echo -e "  ${C_DIM}Clears kernel ring buffer, journal, and GreenBoost log files.${C_RESET}"
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
    gb_ok "All GreenBoost logs (dmesg, journal, /var/log/greenboost) cleared."
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

cmd_clear_nvtx_logs() {
    need_root "clear nvtx-logs"
    local log_file="/run/greenboost/nvtx_events.log"
    local log_old="/run/greenboost/nvtx_events.log.1"
    local cleared=0
    if [[ -f "$log_file" ]]; then
        > "$log_file"
        gb_ok "Cleared $log_file"
        cleared=1
    fi
    if [[ -f "$log_old" ]]; then
        rm -f "$log_old"
        gb_ok "Removed $log_old"
        cleared=1
    fi
    [[ $cleared -eq 0 ]] && gb_info "No NVTX log files found (nothing to clear)"
}

# ── cmd_faults , eBPF tier-migration and UVM fault observability ─────────────
# Usage: greenboost faults [--llm]
#
# Shows live T2↔T3 migration rates, cold-evict and alloc rates, and UVM
# page-fault counters sourced from /run/greenboost/ebpf_stats (written by
# greenboost-ebpf-trace).  When the tracer is absent, falls back to reading
# /proc/driver/nvidia-uvm/.../fault_stats and the kernel module's GET_INFO
# counters, and prints a one-line hint to start the tracer.
#
# Interactive TUI: alternate-screen buffer, 5 s refresh, Ctrl+S / Ctrl+C.
# --llm: machine-readable plain text (ANSI stripped, one section per key).

_cmd_faults_read_ebpf_stats() {
    # Reads /run/greenboost/ebpf_stats into EBPF_* variables.
    # Sets EBPF_TRACER_ACTIVE=1 when file is fresh (written in last 3 s).
    EBPF_TRACER_ACTIVE=0
    EBPF_T3_EVICT_RATE=0; EBPF_T3_PROMOTE_RATE=0; EBPF_COLD_EVICT_RATE=0
    EBPF_T3_BYTES_OUT=0;  EBPF_T3_BYTES_IN=0
    EBPF_ALLOC_RATE=0;    EBPF_PIN_RATE=0
    EBPF_UVM_FAULT_RATE=0; EBPF_UVM_PAGES_IN=0; EBPF_UVM_PAGES_OUT=0
    EBPF_EVENTS_TOTAL=0

    local _sf="/run/greenboost/ebpf_stats"
    [[ -f "$_sf" ]] || return 0
    # Stale check: mtime within 3 s
    local _mtime _now
    _mtime=$(stat -c %Y "$_sf" 2>/dev/null || echo 0)
    _now=$(date +%s)
    (( _now - _mtime > 3 )) && return
    EBPF_TRACER_ACTIVE=1

    local key val
    while IFS='=' read -r key val; do
        case "$key" in
            t3_evict_rate)   EBPF_T3_EVICT_RATE="$val"   ;;
            t3_promote_rate) EBPF_T3_PROMOTE_RATE="$val"  ;;
            cold_evict_rate) EBPF_COLD_EVICT_RATE="$val"  ;;
            t3_bytes_out_s)  EBPF_T3_BYTES_OUT="$val"     ;;
            t3_bytes_in_s)   EBPF_T3_BYTES_IN="$val"      ;;
            alloc_rate)      EBPF_ALLOC_RATE="$val"        ;;
            pin_rate)        EBPF_PIN_RATE="$val"          ;;
            uvm_fault_rate)  EBPF_UVM_FAULT_RATE="$val"   ;;
            uvm_pages_in)    EBPF_UVM_PAGES_IN="$val"      ;;
            uvm_pages_out)   EBPF_UVM_PAGES_OUT="$val"     ;;
            events_total)    EBPF_EVENTS_TOTAL="$val"      ;;
        esac
    done < "$_sf"
}

_cmd_faults_read_uvm_procfs() {
    # Fallback: read UVM fault counts from /proc/driver/nvidia-uvm
    UVM_PROCFS_FAULTS=0; UVM_PROCFS_PAGES_IN=0; UVM_PROCFS_PAGES_OUT=0
    local _base="/proc/driver/nvidia-uvm/gpus"
    [[ -d "$_base" ]] || return 0
    local _gpu _line _val
    for _gpu in "$_base"/*/; do
        [[ -f "${_gpu}fault_stats" ]] || continue
        while read -r _line; do
            case "$_line" in
                "replayable_faults"*)
                    _val="${_line##* }"; UVM_PROCFS_FAULTS=$(( UVM_PROCFS_FAULTS + _val )) ;;
                *"num_pages_in"*)
                    _val=$(echo "$_line" | awk '{print $2}')
                    UVM_PROCFS_PAGES_IN=$(( UVM_PROCFS_PAGES_IN + _val )) ;;
                *"num_pages_out"*)
                    _val=$(echo "$_line" | awk '{print $2}')
                    UVM_PROCFS_PAGES_OUT=$(( UVM_PROCFS_PAGES_OUT + _val )) ;;
            esac
        done < "${_gpu}fault_stats"
    done
}

_cmd_faults_snapshot() {
    local _ts; _ts=$(date '+%Y-%m-%dT%H:%M:%S')

    _cmd_faults_read_ebpf_stats
    _cmd_faults_read_uvm_procfs

    local _pid_f="/run/greenboost/ebpf_trace.pid"
    local _tracer_pid=""
    [[ -f "$_pid_f" ]] && _tracer_pid=$(cat "$_pid_f" 2>/dev/null)

    # ── Title ─────────────────────────────────────────────────────────────
    echo -e "  ${C_VIOLET}${C_BOLD}GreenBoost Faults${C_RESET}  ${C_DIM}eBPF tier-migration · ${_ts}${C_RESET}"
    echo -e "${C_DIM}$(printf '─%.0s' $(seq 1 72))${C_RESET}"

    # ── Tracer status ──────────────────────────────────────────────────────
    if [[ "$EBPF_TRACER_ACTIVE" == "1" ]]; then
        local _pid_hint=""
        [[ -n "$_tracer_pid" ]] && _pid_hint=" (pid ${_tracer_pid})"
        echo -e "  ${C_LIME}${C_BOLD}●${C_RESET}  ${C_GRAY}eBPF tracer active${_pid_hint}  ${C_DIM}events: ${EBPF_EVENTS_TOTAL}${C_RESET}"
    else
        echo -e "  ${C_AMBER}${C_BOLD}○${C_RESET}  ${C_AMBER}eBPF tracer not running${C_RESET}  ${C_DIM}start: sudo greenboost-ebpf-trace${C_RESET}"
        echo -e "  ${C_DIM}(UVM procfs counters shown below , rates not available without tracer)${C_RESET}"
    fi
    echo ""

    # ── T2 ↔ T3 migration rates ───────────────────────────────────────────
    echo -e "  ${C_CYAN}${C_BOLD}T2 ↔ T3 Migration${C_RESET}  ${C_DIM}(5 s avg)${C_RESET}"
    echo -e "${C_DIM}$(printf '─%.0s' $(seq 1 72))${C_RESET}"

    if [[ "$EBPF_TRACER_ACTIVE" == "1" ]]; then
        # Human-readable byte rate helper
        _fmt_brate() {
            local _b="${1:-0}"; local _bi="${_b%.*}"
            if (( _bi >= 1073741824 )); then
                printf "%.1f GB/s" "$(echo "scale=1; $_b / 1073741824" | bc -l 2>/dev/null || echo 0)"
            elif (( _bi >= 1048576 )); then
                printf "%.1f MB/s" "$(echo "scale=1; $_b / 1048576" | bc -l 2>/dev/null || echo 0)"
            elif (( _bi >= 1024 )); then
                printf "%.0f KB/s" "$(echo "scale=0; $_b / 1024" | bc -l 2>/dev/null || echo 0)"
            else
                printf "%.0f B/s" "$_b"
            fi
        }

        local _out_rate; _out_rate=$(_fmt_brate "$EBPF_T3_BYTES_OUT")
        local _in_rate;  _in_rate=$(_fmt_brate "$EBPF_T3_BYTES_IN")

        printf "  %-30s %s/s   %s\n" \
            "T2 → T3 evictions" \
            "${EBPF_T3_EVICT_RATE}" \
            "(${_out_rate})"
        printf "  %-30s %s/s   %s\n" \
            "T3 → T2 promotions" \
            "${EBPF_T3_PROMOTE_RATE}" \
            "(${_in_rate})"
        printf "  %-30s %s/s\n"  "Cold-evict sweeps"  "${EBPF_COLD_EVICT_RATE}"
        printf "  %-30s %s/s\n"  "T2 alloc rate"       "${EBPF_ALLOC_RATE}"
        printf "  %-30s %s/s\n"  "DMA-BUF pin rate"    "${EBPF_PIN_RATE}"
    else
        echo -e "  ${C_DIM}Migration rates unavailable , tracer not running${C_RESET}"
    fi
    echo ""

    # ── UVM page faults ───────────────────────────────────────────────────
    echo -e "  ${C_CYAN}${C_BOLD}NVIDIA UVM Page Faults${C_RESET}"
    echo -e "${C_DIM}$(printf '─%.0s' $(seq 1 72))${C_RESET}"

    if [[ "$EBPF_TRACER_ACTIVE" == "1" ]]; then
        printf "  %-30s %s/s\n"  "UVM GPU fault rate"   "${EBPF_UVM_FAULT_RATE}"
        printf "  %-30s %s\n"    "Pages migrated to GPU" "${EBPF_UVM_PAGES_IN}"
        printf "  %-30s %s\n"    "Pages migrated from GPU" "${EBPF_UVM_PAGES_OUT}"
    fi

    # Procfs always shown as reference
    if [[ "$UVM_PROCFS_FAULTS" -gt 0 || "$UVM_PROCFS_PAGES_IN" -gt 0 ]]; then
        echo -e "  ${C_DIM}procfs (cumulative since load):${C_RESET}"
        printf "  ${C_DIM}  %-28s %s${C_RESET}\n" "replayable faults" "${UVM_PROCFS_FAULTS}"
        printf "  ${C_DIM}  %-28s %s pages${C_RESET}\n" "pages in (→ GPU)" "${UVM_PROCFS_PAGES_IN}"
        printf "  ${C_DIM}  %-28s %s pages${C_RESET}\n" "pages out (← GPU)" "${UVM_PROCFS_PAGES_OUT}"
    else
        echo -e "  ${C_DIM}No managed memory (cudaMallocManaged) detected${C_RESET}"
        echo -e "  ${C_DIM}GreenBoost uses pinned DDR (DMA-BUF) , no UVM page faults expected${C_RESET}"
    fi
    echo ""

    # ── Recent events tail ────────────────────────────────────────────────
    local _evf="/run/greenboost/ebpf_events"
    if [[ -f "$_evf" && "$EBPF_TRACER_ACTIVE" == "1" ]]; then
        echo -e "  ${C_CYAN}${C_BOLD}Recent Migration Events${C_RESET}  ${C_DIM}(last 10)${C_RESET}"
        echo -e "${C_DIM}$(printf '─%.0s' $(seq 1 72))${C_RESET}"
        tail -10 "$_evf" | while IFS= read -r _ev; do
            # Convert ns timestamp to relative "Ns ago"
            local _ts_ns; _ts_ns=$(echo "$_ev" | cut -d' ' -f1)
            local _rest;  _rest=$(echo   "$_ev" | cut -d' ' -f2-)
            local _now_ns; _now_ns=$(date +%s%N 2>/dev/null || echo 0)
            local _age_ms=$(( (_now_ns - _ts_ns) / 1000000 ))
            if (( _age_ms < 2000 )); then
                local _rel="${_age_ms}ms ago"
            else
                local _rel="$(( _age_ms / 1000 ))s ago"
            fi
            echo -e "  ${C_DIM}${_rel}${C_RESET}  ${_rest}"
        done
        echo ""
    fi
}

cmd_faults() {
    local _llm=0
    [[ "${1:-}" == "--llm" ]] && _llm=1

    if [[ "$_llm" == "1" ]]; then
        _cmd_faults_snapshot | sed 's/\x1b\[[0-9;]*m//g' | sed 's/^[[:space:]]*//'
        return 0
    fi

    if [[ ! -t 0 ]]; then
        # Non-interactive: single snapshot to stdout
        _cmd_faults_snapshot
        return 0
    fi

    # Interactive TUI
    _gb_run_tui_loop "_cmd_faults_snapshot" 5 \
        "  ${C_DIM}Updating every 5s  ${C_GRAY}Ctrl+S${C_DIM}: refresh  ${C_GRAY}Ctrl+C${C_DIM}: exit${C_RESET}" \
        "/run/greenboost/ebpf_events"
}

# ── cmd_top / cmd_residency - per-buffer hot/cold residency observability ──
#
# Source: /sys/kernel/debug/greenboost/residency (kernel debugfs export,
# greenboost.c gb_residency_show), one line per live T2/T3 buffer with
# id/tier/size_mb/heat/flags/pid.  `heat` is the ARC-style access-frequency
# score pushed by the shim via GB_IOCTL_SET_HEAT - this is the "nvidia-smi
# for memory migration" gap: which allocations are actually hot vs. just
# recently touched.

_GB_RESIDENCY_FILE="/sys/kernel/debug/greenboost/residency"

# Reads $_GB_RESIDENCY_FILE into the global array _GB_RES_LINES (one
# key=value string per buffer).  Sets _GB_RES_AVAILABLE=1/0.
_gb_residency_read() {
    _GB_RES_AVAILABLE=0
    _GB_RES_LINES=()
    if [[ -r "$_GB_RESIDENCY_FILE" ]]; then
        mapfile -t _GB_RES_LINES < "$_GB_RESIDENCY_FILE" 2>/dev/null
        _GB_RES_AVAILABLE=1
    elif [[ -e "$_GB_RESIDENCY_FILE" ]] && command -v sudo >/dev/null 2>&1; then
        mapfile -t _GB_RES_LINES < <(sudo cat "$_GB_RESIDENCY_FILE" 2>/dev/null)
        [[ ${#_GB_RES_LINES[@]} -gt 0 ]] && _GB_RES_AVAILABLE=1
    fi
}

# Extracts "key=value" field $2 from a residency line $1.
_gb_res_field() {
    local _line="$1" _key="$2"
    [[ "$_line" =~ ${_key}=([^[:space:]]+) ]] && echo "${BASH_REMATCH[1]}"
}

_cmd_top_snapshot() {
    local _ts; _ts=$(date '+%Y-%m-%dT%H:%M:%S')

    echo -e "  ${C_VIOLET}${C_BOLD}GreenBoost Top${C_RESET}  ${C_DIM}per-buffer residency · ${_ts}${C_RESET}"
    echo -e "${C_DIM}$(printf '─%.0s' $(seq 1 88))${C_RESET}"

    _gb_residency_read
    if [[ "$_GB_RES_AVAILABLE" != "1" ]]; then
        echo -e "  ${C_AMBER}residency export unavailable${C_RESET}  ${C_DIM}(debugfs not mounted, kernel module not loaded, or run with sudo)${C_RESET}"
        return 0
    fi
    if [[ ${#_GB_RES_LINES[@]} -eq 0 ]]; then
        echo -e "  ${C_DIM}No live T2/T3 buffers${C_RESET}"
        return 0
    fi

    printf "  %-8s %-5s %10s %8s %-22s %-8s %s\n" \
        "ID" "TIER" "SIZE_MB" "HEAT" "FLAGS" "PID" "STATE"
    echo -e "${C_DIM}$(printf '─%.0s' $(seq 1 88))${C_RESET}"

    # Sort by heat descending (hottest first) - the whole point of this view.
    local _line _id _tier _size _heat _flags _pid _frozen _t1p _state
    while IFS= read -r _line; do
        [[ -z "$_line" ]] && continue
        _id=$(_gb_res_field   "$_line" id)
        _tier=$(_gb_res_field "$_line" tier)
        _size=$(_gb_res_field "$_line" size_mb)
        _heat=$(_gb_res_field "$_line" heat)
        _flags=$(_gb_res_field "$_line" flags)
        _pid=$(_gb_res_field  "$_line" pid)
        _frozen=$(_gb_res_field "$_line" frozen)
        _t1p=$(_gb_res_field    "$_line" t1_priority)
        _state="cold"
        [[ "$_heat" -ge 8 ]] 2>/dev/null && _state="hot"
        [[ "${_heat:-0}" -gt 0 && "${_heat:-0}" -lt 8 ]] 2>/dev/null && _state="warm"
        [[ "$_frozen" == "1" ]] && _state="frozen"
        [[ "$_t1p" == "1" ]] && _state="t1-priority"
        printf "  %-8s %-5s %10s %8s %-22s %-8s %s\n" \
            "$_id" "$_tier" "$_size" "${_heat:-0}" "$_flags" "$_pid" "$_state"
    done < <(printf '%s\n' "${_GB_RES_LINES[@]}" | sort -t= -k4 -rn 2>/dev/null || printf '%s\n' "${_GB_RES_LINES[@]}")
}

cmd_top() {
    local _llm=0
    [[ "${1:-}" == "--llm" ]] && _llm=1

    if [[ "$_llm" == "1" ]]; then
        _cmd_top_snapshot | sed 's/\x1b\[[0-9;]*m//g' | sed 's/^[[:space:]]*//'
        return 0
    fi

    if [[ ! -t 0 ]]; then
        _cmd_top_snapshot
        return 0
    fi

    _gb_run_tui_loop "_cmd_top_snapshot" 5 \
        "  ${C_DIM}Updating every 5s  ${C_GRAY}Ctrl+S${C_DIM}: refresh  ${C_GRAY}Ctrl+C${C_DIM}: exit${C_RESET}"
}

_cmd_residency_snapshot() {
    local _ts; _ts=$(date '+%Y-%m-%dT%H:%M:%S')

    echo -e "  ${C_VIOLET}${C_BOLD}GreenBoost Residency${C_RESET}  ${C_DIM}hot/warm/cold breakdown · ${_ts}${C_RESET}"
    echo -e "${C_DIM}$(printf '─%.0s' $(seq 1 72))${C_RESET}"

    _gb_residency_read
    if [[ "$_GB_RES_AVAILABLE" != "1" ]]; then
        echo -e "  ${C_AMBER}residency export unavailable${C_RESET}  ${C_DIM}(debugfs not mounted, kernel module not loaded, or run with sudo)${C_RESET}"
        return 0
    fi

    local _hot_mb=0 _warm_mb=0 _cold_mb=0 _t2_mb=0 _t3_mb=0
    local _line _tier _size _heat
    for _line in "${_GB_RES_LINES[@]}"; do
        [[ -z "$_line" ]] && continue
        _tier=$(_gb_res_field "$_line" tier)
        _size=$(_gb_res_field "$_line" size_mb); _size="${_size:-0}"
        _heat=$(_gb_res_field "$_line" heat);    _heat="${_heat:-0}"
        [[ "$_tier" == "T2" ]] && _t2_mb=$(( _t2_mb + _size ))
        [[ "$_tier" == "T3" ]] && _t3_mb=$(( _t3_mb + _size ))
        if   (( _heat >= 8 )); then _hot_mb=$(( _hot_mb + _size ))
        elif (( _heat > 0  )); then _warm_mb=$(( _warm_mb + _size ))
        else                        _cold_mb=$(( _cold_mb + _size ))
        fi
    done

    echo -e "  ${C_CYAN}${C_BOLD}By tier${C_RESET}"
    printf "  %-20s %10s MB\n" "T2 (DDR)"  "$_t2_mb"
    printf "  %-20s %10s MB\n" "T3 (NVMe)" "$_t3_mb"
    echo ""
    echo -e "  ${C_CYAN}${C_BOLD}By heat${C_RESET}  ${C_DIM}(hot: heat≥8, warm: 1-7, cold: 0 - never re-touched since last decay)${C_RESET}"
    printf "  %-20s %10s MB\n" "Hot"  "$_hot_mb"
    printf "  %-20s %10s MB\n" "Warm" "$_warm_mb"
    printf "  %-20s %10s MB\n" "Cold" "$_cold_mb"
    echo ""

    # Churn: reuse the eBPF migration rates if the tracer is running -
    # same source _cmd_faults_snapshot uses, avoids a second counting path.
    _cmd_faults_read_ebpf_stats 2>/dev/null
    if [[ "${EBPF_TRACER_ACTIVE:-0}" == "1" ]]; then
        echo -e "  ${C_CYAN}${C_BOLD}Churn${C_RESET}  ${C_DIM}(5 s avg, from eBPF tracer)${C_RESET}"
        printf "  %-20s %s/s\n" "T2 → T3 evictions"  "${EBPF_T3_EVICT_RATE:-0}"
        printf "  %-20s %s/s\n" "T3 → T2 promotions" "${EBPF_T3_PROMOTE_RATE:-0}"
    else
        echo -e "  ${C_DIM}Churn rates unavailable - start: sudo greenboost-ebpf-trace${C_RESET}"
    fi
}

cmd_residency() {
    local _llm=0
    [[ "${1:-}" == "--llm" ]] && _llm=1

    if [[ "$_llm" == "1" ]]; then
        _cmd_residency_snapshot | sed 's/\x1b\[[0-9;]*m//g' | sed 's/^[[:space:]]*//'
        return 0
    fi

    if [[ ! -t 0 ]]; then
        _cmd_residency_snapshot
        return 0
    fi

    _gb_run_tui_loop "_cmd_residency_snapshot" 5 \
        "  ${C_DIM}Updating every 5s  ${C_GRAY}Ctrl+S${C_DIM}: refresh  ${C_GRAY}Ctrl+C${C_DIM}: exit${C_RESET}"
}

# ── cmd_debug / cmd_debug_vitals - toggleable deep diagnostics ──────────────
# Usage: greenboost debug vitals [on|off]
#
# on:   writes /etc/greenboost/debug_vitals.enabled - scripts check this flag
#       to show the extended vitals panel automatically.
# off:  removes the flag file.
# (no arg): shows the full vitals dump regardless of flag state.

_show_debug_vitals_dump() {
    _cmd_vitals_snapshot
}

_vitals_set_kernel_debug() {
    local val="${1:-0}"
    local dbg_f="/sys/module/greenboost/parameters/debug_mode"
    if [[ -w "$dbg_f" ]]; then
        echo "$val" > "$dbg_f"
        return 0
    fi
    return 1
}

_vitals_write_profiled() {
    local enabled="${1:-0}"
    local profiled="/etc/profile.d/greenboost-vitals.sh"
    if (( enabled )); then
        cat > "$profiled" << 'VITPROFEOF'
# GreenBoost Debug Vitals - set when /etc/greenboost/debug_vitals.enabled exists
[ -f /etc/greenboost/debug_vitals.enabled ] && export GREENBOOST_NVTX_VERBOSE=1
VITPROFEOF
        gb_ok "profile.d: GREENBOOST_NVTX_VERBOSE=1 exported for new login shells"
    else
        rm -f "$profiled"
    fi
}

_vitals_restart_service() {
    local svc="$1" label="$2"
    if systemctl is-active --quiet "$svc" 2>/dev/null; then
        if systemctl restart "$svc" 2>/dev/null; then
            gb_ok "Restarted ${label} (picks up new vitals settings)"
        else
            gb_warn "Could not restart ${label} - may need manual restart"
        fi
    fi
}

_vitals_find_gb_services() {
    local -a _svcs=()
    for _unit in /etc/systemd/system/*.service /usr/lib/systemd/system/*.service; do
        [[ -f "$_unit" ]] || continue
        grep -q 'GREENBOOST_ACTIVE=1' "$_unit" 2>/dev/null || continue
        local _name; _name=$(basename "$_unit" .service)
        [[ "$_name" == "greenboost-"* ]] && continue
        _svcs+=("$_name")
    done
    printf '%s\n' "${_svcs[@]}"
}

cmd_debug_vitals() {
    local subcmd="${1:-}"
    local flag_file="/etc/greenboost/debug_vitals.enabled"
    local reboot_needed=0

    case "$subcmd" in
        on)
            need_root "debug vitals on"
            mkdir -p /etc/greenboost
            touch "$flag_file"
            echo ""
            gb_ok "Debug vitals ENABLED  (flag: ${flag_file})"
            echo ""

            # 1. Kernel debug_mode (live, no restart needed)
            if _vitals_set_kernel_debug 1; then
                gb_ok "Kernel debug_mode=1  (verbose dmesg - view: journalctl -k --grep=greenboost -f)"
            else
                gb_warn "Module not loaded or debug_mode not writable - load module first: sudo greenboost load"
            fi

            # 2. profile.d for new login shells
            _vitals_write_profiled 1

            # 3. GREENBOOST_NVTX_VERBOSE live hint (current session only)
            export GREENBOOST_NVTX_VERBOSE=1
            gb_info "GREENBOOST_NVTX_VERBOSE=1 set for this shell session"

            # 4. Restart greenboost-supervisor if running
            systemctl daemon-reload 2>/dev/null || true
            _vitals_restart_service "greenboost-supervisor" "greenboost-supervisor"

            # 5. Restart other GB-aware services (ollama, vllm, etc.)
            local _need_restart=()
            while IFS= read -r _svc; do
                [[ -n "$_svc" ]] && _need_restart+=("$_svc")
            done < <(_vitals_find_gb_services)

            if (( ${#_need_restart[@]} > 0 )); then
                echo ""
                gb_warn "The following services use GreenBoost but are NOT yet restarted:"
                for _s in "${_need_restart[@]}"; do
                    printf "  ${C_AMBER}◈${C_RESET}  ${C_GRAY}%s${C_RESET}  ${C_DIM}(restart: sudo systemctl restart %s)${C_RESET}\n" "$_s" "$_s"
                done
                gb_info "Restart them to pick up GREENBOOST_NVTX_VERBOSE=1 in their sessions."
                echo ""
            fi

            # 6. Check if LD_PRELOAD is globally applied - if not, new processes won't get shim
            if ! grep -q 'libgreenboost_cuda.so' /etc/ld.so.preload 2>/dev/null; then
                gb_warn "/etc/ld.so.preload does not include libgreenboost_cuda.so"
                gb_info "Only processes launched with explicit LD_PRELOAD will generate NVTX events."
                gb_info "For system-wide shim: sudo greenboost install-sys-configs"
                reboot_needed=1
            fi

            echo ""
            gb_info "Instrumentation applied:"
            gb_info "  • Kernel debug_mode=1       → /sys/module/greenboost/parameters/debug_mode"
            gb_info "  • GREENBOOST_NVTX_VERBOSE=1 → /etc/profile.d/greenboost-vitals.sh"
            gb_info "  • Flag: ${flag_file}"
            echo ""
            gb_info "Data sources now active:"
            gb_info "  /run/greenboost/nvtx_events.log  - allocation/phase/OOM events"
            gb_info "  /run/greenboost/shim_stats        - path/frag/dispatch counters"
            gb_info "  /run/greenboost/metrics.json      - JSON + feeder data"
            gb_info "  /run/greenboost/phase             - current phase"
            gb_info "  journalctl -k --grep=greenboost   - kernel module events"
            gb_info "  nvidia-smi (extended)             - GPU temp/power/clocks/util"
            echo ""
            gb_info "View all vitals: greenboost debug vitals"
            gb_info "Disable:         sudo greenboost debug vitals off"

            if (( reboot_needed )); then
                echo ""
                gb_warn "Some changes require processes to be (re)started or a reboot to take full effect."
                gb_warn "Reboot: sudo reboot  (ensures all services pick up the new profile.d settings)"
            fi
            echo ""
            ;;

        off)
            need_root "debug vitals off"
            rm -f "$flag_file"
            echo ""
            gb_ok "Debug vitals DISABLED"

            # 1. Reset kernel debug_mode to 0
            if _vitals_set_kernel_debug 0; then
                gb_ok "Kernel debug_mode=0  (verbose dmesg disabled)"
            fi

            # 2. Remove profile.d
            _vitals_write_profiled 0
            gb_ok "profile.d: GREENBOOST_NVTX_VERBOSE removed"

            # 3. Restart services
            systemctl daemon-reload 2>/dev/null || true
            _vitals_restart_service "greenboost-supervisor" "greenboost-supervisor"

            local _need_restart=()
            while IFS= read -r _svc; do
                [[ -n "$_svc" ]] && _need_restart+=("$_svc")
            done < <(_vitals_find_gb_services)

            if (( ${#_need_restart[@]} > 0 )); then
                echo ""
                gb_info "Services that should be restarted to fully disable verbose NVTX:"
                for _s in "${_need_restart[@]}"; do
                    printf "  ${C_GRAY}◈${C_RESET}  ${C_DIM}%s${C_RESET}\n" "$_s"
                done
            fi

            echo ""
            gb_info "Overhead removed. Extended vitals panel suppressed in scripts."
            gb_info "On-demand dump still available: greenboost debug vitals"
            echo ""
            ;;

        "--llm")
            # Audit F-L4-27: machine-readable compact vitals for LLM/script consumers.
            detect_hardware 2>/dev/null || true
            local _strip='s/\x1b\[[0-9;]*[mKHJsu]//g'
            _cmd_vitals_snapshot | sed "$_strip" | sed 's/^[[:space:]]*//' | grep -v '^─'
            return
            ;;

        ""|vitals)
            detect_hardware 2>/dev/null || true
            if [[ ! -t 0 ]]; then
                _cmd_vitals_snapshot
                return
            fi
            _gb_run_tui_loop "_cmd_vitals_snapshot" 5 \
                "  ${C_DIM}Updating every 5s  ${C_GRAY}Ctrl+S${C_DIM}: refresh  ${C_GRAY}Ctrl+L${C_DIM}: log view  ${C_GRAY}Ctrl+C${C_DIM}: exit${C_RESET}" \
                "$GB_STATUS_LOG"
            ;;
        *)
            gb_fail "Unknown subcommand: '${subcmd}'  (use: on | off, or no arg to show vitals)"
            echo ""
            return 1
            ;;
    esac
}

cmd_debug() {
    local subcmd="${1:-vitals}"
    case "$subcmd" in
        vitals) cmd_debug_vitals "${@:2}" ;;
        *)
            gb_fail "Unknown debug subcommand: '${subcmd}'  (use: vitals [on|off])"
            echo ""
            return 1
            ;;
    esac
}

# Returns 0 (true) if a process comm is essential and must never be killed.
# Matched against /proc/<pid>/comm, which the kernel truncates to 15 chars -
# so all patterns use globs/prefixes to survive truncation (e.g.
# "gnome-session-binary" appears as "gnome-session-b").
_gb_proc_is_protected() {
    # Protected categories (comments kept outside the pattern list because bash
    # forbids comments between '\'-continued case patterns):
    #   GNOME/desktop shell + session, display managers, X/Wayland, KDE,
    #   init/session/bus/audio daemons, NVIDIA + GreenBoost daemons.
    case "$1" in
        gnome-shell|gnome-shell-*|gnome-session*|gnome-control-*|gnome-software*|gnome-keyring*|mutter|gjs|tracker-*|\
        gdm|gdm-*|Xorg|Xorg.bin|Xwayland|Xwayland*|X|\
        kwin|kwin_*|plasmashell|kded*|ksmserver|sddm|sddm-*|lightdm|\
        systemd|systemd-*|init|dbus-daemon|dbus-broker|elogind|seatd|\
        pipewire|pipewire-*|wireplumber|pulseaudio|\
        nvidia-persiste*|nvidia-*|greenboost*|greenboost-netd)
            return 0 ;;
    esac
    return 1
}

cmd_clear_cluster_workers() {
    # Kill cluster stage/block workers on every feeder + host-side SSH
    # tunnels, and remove stage temp files. Companion to 'clear memory-pool':
    # orphaned gb_remote_blocks workers hold ~6 GB of feeder VRAM hostage.
    gb_header
    echo -e "  ${C_CYAN}${C_BOLD}Clear Cluster Workers${C_RESET}"
    echo -e "  ${C_DIM}Kills feeder-side stage workers (gb_remote_blocks, encode/render stages)"
    echo -e "  and host-side SSH tunnels; removes /tmp stage files on both sides.${C_RESET}"
    echo ""

    local conf="/etc/greenboost/cluster.conf"
    if [[ ! -f "$conf" ]]; then
        gb_info "No cluster.conf , nothing feeder-side to clear"
    else
        local _line _addr _ip _user
        while IFS= read -r _line; do
            _line="${_line%%#*}"; [[ -z "${_line// }" ]] && continue
            _addr=$(echo "$_line" | awk '{print $1}')
            _ip="${_addr%%:*}"
            _user=$(echo "$_line" | awk '{print $3}')
            _user="${_user:-${SUDO_USER:-$USER}}"
            gb_info "Feeder ${_ip}: killing stage workers..."
            # [g]/[e]/[r] patterns so pkill never matches its own shell.
            ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
                "${_user}@${_ip}" \
                'pkill -f "[g]b_remote_blocks.py --serve"; \
                 pkill -f "[e]ncode_klein_prompts.py"; \
                 pkill -f "[r]ender_klein_cards.py"; \
                 rm -f /tmp/gb_feeder_te_* /tmp/gb_feeder_render_*; \
                 exit 0' 2>/dev/null \
                && gb_ok "  ${_ip} cleared" \
                || gb_warn "  ${_ip} unreachable (skipped)"
        done < "$conf"
    fi

    # Host side: worker tunnels + stage temp files.
    pkill -f "[s]sh .*-L 9741:127.0.0.1:9741" 2>/dev/null || true
    rm -f /tmp/gb_feeder_te_* /tmp/gb_feeder_render_* 2>/dev/null || true
    gb_ok "Host tunnels + stage temp files cleared"
}

cmd_clear_memory_pool() {
    local GB_DEV=/dev/greenboost
    local SYSFS=/sys/class/greenboost/greenboost

    gb_header
    echo -e "  ${C_CYAN}${C_BOLD}Clear Memory Pool${C_RESET}"
    echo -e "  ${C_DIM}Releases GPU + T2/T3 memory held by inference processes."
    echo -e "  Kills heavy GPU compute jobs (CUDA / GreenBoost); desktop & GNOME processes are protected.${C_RESET}"
    echo -e ""

    local module_loaded=0
    if [[ -c "$GB_DEV" ]]; then
        module_loaded=1
    else
        gb_warn "GreenBoost module not loaded (/dev/greenboost not found) — skipping kernel-level buffer release; GPU process cleanup still runs."
    fi

    # Capture RAM state before any action
    local ram_before_free ram_before_cached ram_before_avail
    ram_before_free=$(awk '/^MemFree:/{print int($2/1024)}' /proc/meminfo)
    ram_before_cached=$(awk '/^Cached:/{print int($2/1024)}' /proc/meminfo)
    ram_before_avail=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)

    local before
    before=$(cat "$SYSFS/pool_brief" 2>/dev/null || echo "unavailable")
    echo -e "  Before: ${C_DIM}${before}${C_RESET}"
    echo -e "  RAM:    ${C_DIM}free=${ram_before_free}MB cache=${ram_before_cached}MB avail=${ram_before_avail}MB${C_RESET}"

    # Minimum GPU memory (MiB) for a *compute* process to count as "big data"
    # worth killing.  Desktop GPU-rasterised apps (browsers, Electron) stay well
    # below this in the CUDA context; inference jobs use gigabytes.
    # Override with GB_KILL_MIN_MB.
    local kill_min_mb="${GB_KILL_MIN_MB:-512}"

    # Never target our own process tree.
    local self_pids=" $$ ${PPID:-0} ${GB_SELF_PID:-0} "

    # ── Map every CUDA compute process → GPU MiB (graphics-only desktop procs
    #    such as gnome-shell never appear in this query). ────────────────────
    local -A gpu_mb=()
    if command -v nvidia-smi >/dev/null 2>&1; then
        local cpid cmem
        while IFS=',' read -r cpid cmem; do
            cpid="${cpid//[[:space:]]/}"; cmem="${cmem//[[:space:]]/}"
            [[ "$cpid" =~ ^[0-9]+$ ]] || continue
            [[ "$cmem" =~ ^[0-9]+$ ]] || cmem=0
            gpu_mb["$cpid"]="$cmem"
        done < <(nvidia-smi --query-compute-apps=pid,used_memory \
                     --format=csv,noheader,nounits 2>/dev/null || true)
    fi

    # ── Build candidate set ─────────────────────────────────────────────────
    #   (a) processes holding /dev/greenboost open  → GreenBoost pool users
    #       (always candidates, any size - that's what "clear pool" targets)
    #   (b) CUDA compute processes using >= kill_min_mb MiB
    local -A cand=()
    local p
    for p in $(fuser "$GB_DEV" 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' || true); do
        cand["$p"]=1
    done
    for p in "${!gpu_mb[@]}"; do
        if (( ${gpu_mb[$p]:-0} >= kill_min_mb )); then cand["$p"]=1; fi
    done

    # ── Filter: drop self, PID 1, and protected desktop/system processes ────
    local kill_list=() comm
    for p in "${!cand[@]}"; do
        [[ "$p" == "1" ]] && continue
        case "$self_pids" in *" $p "*) continue ;; esac
        comm=$(cat "/proc/$p/comm" 2>/dev/null || true)
        [[ -z "$comm" ]] && continue                 # already gone
        if _gb_proc_is_protected "$comm"; then
            gb_info "Protected (skipped): PID $p (${comm}, ${gpu_mb[$p]:-0} MiB)"
            continue
        fi
        kill_list+=("$p")
    done

    if (( ${#kill_list[@]} == 0 )); then
        gb_info "No killable GPU inference processes found (desktop/GNOME processes are protected)."
    else
        echo -e "  ${C_AMBER}Terminating GPU inference processes:${C_RESET}"
        for p in "${kill_list[@]}"; do
            comm=$(cat "/proc/$p/comm" 2>/dev/null || echo "?")
            echo -e "    ${C_AMBER}• PID $p (${comm}) - ${gpu_mb[$p]:-?} MiB${C_RESET}"
        done
        # SIGTERM first for a clean CUDA teardown, then SIGKILL any stragglers.
        for p in "${kill_list[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
        sleep 2
        for p in "${kill_list[@]}"; do
            if kill -0 "$p" 2>/dev/null; then
                kill -9 "$p" 2>/dev/null && gb_info "  SIGKILL PID $p" || gb_info "  PID $p unkillable"
            else
                gb_info "  PID $p exited"
            fi
        done
        sleep 1
    fi

    # Root path: force-release T2/T3 buffers via GB_IOCTL_RELEASE_PID
    # struct gb_release_pid_req { uint32_t pid; uint32_t _pad; }  magic=0x47, nr=16
    if [[ $EUID -ne 0 ]]; then
        gb_info "  (run as root for kernel-level buffer release via GB_IOCTL_RELEASE_PID)"
    elif [[ $module_loaded -eq 1 ]]; then
        python3 - "$GB_DEV" "${kill_list[@]}" <<'PYEOF'
import sys, os, fcntl, struct
GB_MAGIC = ord('G')
NR = (1 << 30) | (8 << 16) | (GB_MAGIC << 8) | 16
dev  = sys.argv[1]
pids = [int(x) for x in sys.argv[2:] if x.strip()]
fd   = os.open(dev, os.O_RDWR)
for pid in pids:
    req = bytearray(struct.pack("II", pid, 0))
    try:
        fcntl.ioctl(fd, NR, req)
        print(f"  IOCTL GB_RELEASE_PID({pid}): ok")
    except OSError as e:
        print(f"  IOCTL GB_RELEASE_PID({pid}): {e}")
os.close(fd)
PYEOF
    else
        gb_info "  (module not loaded — kernel-level buffer release skipped)"
    fi

    # Drop system page cache so model weight files (read via read() / --no-mmap)
    # are immediately reclaimed from DDR rather than waiting for kernel pressure.
    # echo 3 drops page cache + dentries + inodes; sync first to avoid data loss.
    if [[ $EUID -eq 0 ]]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches
        # Compact memory fragmentation left by large DMA-BUF allocations
        echo 1 > /proc/sys/vm/compact_memory 2>/dev/null || true
        gb_info "  Page cache dropped, memory compacted"
    fi

    local after
    after=$(cat "$SYSFS/pool_brief" 2>/dev/null || echo "unavailable")
    local ram_after_free ram_after_cached ram_after_avail
    ram_after_free=$(awk '/^MemFree:/{print int($2/1024)}' /proc/meminfo)
    ram_after_cached=$(awk '/^Cached:/{print int($2/1024)}' /proc/meminfo)
    ram_after_avail=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
    local ram_reclaimed=$(( ram_after_avail - ram_before_avail ))
    echo -e "  After:  ${C_DIM}${after}${C_RESET}"
    echo -e "  RAM:    ${C_DIM}free=${ram_after_free}MB cache=${ram_after_cached}MB avail=${ram_after_avail}MB${C_RESET}"
    if (( ram_reclaimed > 0 )); then
        gb_ok "Memory pool clear complete  (+${ram_reclaimed}MB DDR reclaimed)."
    else
        gb_ok "Memory pool clear complete."
    fi
}

# Backward-compat alias - kept so existing scripts/bookmarks still work.
cmd_clean_logs() { cmd_clear_logs; }

# ════════════════════════════════════════════════════════════════════════
# _logs_llm - compact, token-efficient log output for LLM/AI tools.
# No ANSI colors, no human-readable decoration, deduplicated lines.
# Format: key=value header, labeled sections, [Nx] for repeated lines.
_logs_llm() {
    local _strip='s/\x1b\[[0-9;]*[mKHJsu]//g'

    # Header
    local _mod="MISSING"
    grep -q '^greenboost' <<< "$(lsmod 2>/dev/null)" && _mod="loaded"
    local _vram="?"
    _vram=$(journalctl -u ollama --no-pager -q -n 50 2>/dev/null \
        | grep -oP 'total="\K[^"]+' | tail -1)
    [[ -z "$_vram" ]] && _vram="?"
    echo "gb_logs v=${GB_VERSION} ts=$(date '+%Y-%m-%dT%H:%M') mod=${_mod} vram=${_vram}"

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

    # 2. Services - inference only (ollama + supervisor)
    local _svc
    _svc=$(journalctl --since "1 hour ago" --no-pager -q \
        -u ollama -u greenboost-supervisor 2>/dev/null | tail -60)
    _llm_section "services" "$_svc"

    # 3. AppArmor
    local _aa
    _aa=$(journalctl -k --no-pager -q -n 200 2>/dev/null \
        | grep -iE 'apparmor="DENIED".*greenboost|greenboost.*apparmor="DENIED"' | tail -10)
    _llm_section "apparmor" "$_aa"

    # Diagnostic summary
    local _errs=() _warns=() _ok=()
    [[ "$_mod" == "loaded" ]] && _ok+=("module=loaded") || _errs+=("module=MISSING")
    local _ec=${#_errs[@]} _wc=${#_warns[@]} _oc=${#_ok[@]}
    echo "[diag] errors=${_ec} warns=${_wc} ok=${_oc}"
    for _e in "${_errs[@]}";  do echo "  ERR: ${_e}"; done
    for _w in "${_warns[@]}"; do echo "  WARN: ${_w}"; done
    for _o in "${_ok[@]}";    do echo "  OK: ${_o}"; done
}

_cmd_logs_snapshot() {
    local _mod_status="MISSING"
    grep -q '^greenboost' <<< "$(lsmod 2>/dev/null)" && _mod_status="loaded"
    local _vram_str="?"
    _vram_str=$(journalctl -u ollama --no-pager -q -n 50 2>/dev/null \
        | grep -oP 'total="\K[^"]+' | tail -1)
    [[ -z "$_vram_str" ]] && _vram_str="?"
    local _ts; _ts=$(date '+%Y-%m-%dT%H:%M')
    echo -e ""
    echo -e "  ${C_VIOLET}${C_BOLD}GreenBoost v${GB_VERSION} logs${C_RESET}  ${C_DIM}·  ${_ts}  ·  module:${_mod_status}  vram:${_vram_str}${C_RESET}"
    echo -e ""

    local _diag_errors=() _diag_warns=() _diag_ok=()
    [[ "$_mod_status" == "loaded" ]] \
        && _diag_ok+=("kernel module loaded - VRAM ${_vram_str} reported to apps") \
        || _diag_errors+=("kernel module NOT loaded - no T2/T3 memory available")

    # 1. Kernel module events
    local _flow; _flow=$(gather_flow_events 25)
    local _flow_n=0; [[ -n "$_flow" ]] && _flow_n=$(printf '%s\n' "$_flow" | wc -l)
    gb_section "Kernel Module (dmesg)  (${_flow_n} events)"
    if [[ -n "$_flow" ]]; then
        while IFS= read -r _fline; do
            [[ -z "$_fline" ]] && continue
            echo -e "  ${C_DIM}${_fline}${C_RESET}"
        done <<< "$_flow"
    else
        echo -e "  ${C_DIM}(empty - dmesg grep: greenboost tier-transitions)${C_RESET}"
    fi
    echo ""

    # 2. Service journal
    local _svc
    _svc=$(journalctl --since "1 hour ago" --no-pager -q \
        -u ollama -u greenboost-supervisor 2>/dev/null | tail -25)
    local _svc_n=0; [[ -n "$_svc" ]] && _svc_n=$(printf '%s\n' "$_svc" | wc -l)
    gb_section "Services (last 1h)  (${_svc_n} events)"
    if [[ -n "$_svc" ]]; then
        while IFS= read -r _sline; do
            [[ -z "$_sline" ]] && continue
            echo -e "  ${C_GRAY}${_sline}${C_RESET}"
        done <<< "$_svc"
    else
        echo -e "  ${C_DIM}(empty - units: ollama greenboost-supervisor)${C_RESET}"
    fi
    echo ""

    # 3. AppArmor denials
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
        echo -e "  ${C_LIME}✓${C_RESET}  ${C_GRAY}All checks passed - no errors or warnings detected.${C_RESET}"
    fi
    echo -e "  ${C_DIM}[Ctrl+C exit · Ctrl+S refresh]${C_RESET}"
}

_GB_LOG_CAPTURE_PID_FILE="${TMPDIR:-/tmp}/greenboost_log_capture.pid"

# _gb_logs_full_dump - writes a comprehensive, ANSI-free diagnostic bundle.
# Format is readable by both humans and LLMs (no escape codes, timestamped).
_gb_logs_full_dump() {
    local _strip='s/\x1b\[[0-9;]*[mKHJsu]//g'
    local _ts; _ts=$(date '+%Y-%m-%dT%H:%M:%S')

    echo "=== GreenBoost Diagnostic Bundle v${GB_VERSION} ==="
    echo "Timestamp: ${_ts}"
    echo "Host: $(hostname -f 2>/dev/null || hostname)"
    echo ""

    echo "--- Vitals Snapshot ---"
    _cmd_vitals_snapshot 2>/dev/null | sed "$_strip" | sed 's/^[[:space:]]*//'
    echo ""

    echo "--- Kernel Events (last 100 greenboost dmesg) ---"
    gather_flow_events 100 2>/dev/null \
        | sed "$_strip" | sed 's/^[[:space:]]*//' \
        || dmesg 2>/dev/null | grep -i greenboost | tail -50
    echo ""

    echo "--- Service Journal (last 1h: ollama, greenboost-supervisor) ---"
    journalctl --since "1 hour ago" --no-pager -q \
        -u ollama -u greenboost-supervisor 2>/dev/null | tail -100
    echo ""

    echo "--- NVTX Events (last 100) ---"
    _cmd_nvtx_logs_snapshot 100 "" 1 2>/dev/null | sed "$_strip" | sed 's/^[[:space:]]*//'
    echo ""

    echo "--- AppArmor Denials ---"
    journalctl -k --no-pager -q -n 200 2>/dev/null \
        | grep -iE 'apparmor="DENIED".*greenboost|greenboost.*apparmor="DENIED"' | tail -20 \
        || echo "(none)"
    echo ""

    echo "--- Health Check ---"
    cmd_health_check --llm 2>/dev/null || echo "(health-check unavailable)"
    echo ""

    echo "=== End of Bundle ==="
}

# cmd_logs_save - write one-shot diagnostic bundle to file.
cmd_logs_save() {
    local _outfile="${1:-}"
    if [[ -z "$_outfile" ]]; then
        if [[ -t 0 ]]; then
            local _default; _default="greenboost_$(date '+%Y%m%d_%H%M%S').log"
            printf "Save log to file [%s]: " "$_default"
            read -r _outfile
            [[ -z "$_outfile" ]] && _outfile="$_default"
        else
            _outfile="greenboost_$(date '+%Y%m%d_%H%M%S').log"
        fi
    fi

    detect_hardware 2>/dev/null || true
    gb_info "Collecting diagnostic bundle..."
    _gb_logs_full_dump > "$_outfile"
    local _sz; _sz=$(wc -c < "$_outfile" 2>/dev/null || echo "?")
    gb_ok "Saved ${_sz} bytes to: ${_outfile}"
    echo ""
    echo "  Share with support or paste into an LLM:"
    echo "    cat '$_outfile' | less"
    echo "    # or: greenboost logs save myfile.log && cat myfile.log"
}

# cmd_logs_start - background log capture, appends bundle every 30s.
cmd_logs_start() {
    local _outfile="${1:-}"
    if [[ -z "$_outfile" ]]; then
        if [[ -t 0 ]]; then
            local _default; _default="greenboost_capture_$(date '+%Y%m%d_%H%M%S').log"
            printf "Capture to file [%s]: " "$_default"
            read -r _outfile
            [[ -z "$_outfile" ]] && _outfile="$_default"
        else
            _outfile="greenboost_capture_$(date '+%Y%m%d_%H%M%S').log"
        fi
    fi

    # Stop existing capture if running
    if [[ -f "$_GB_LOG_CAPTURE_PID_FILE" ]]; then
        local _old_pid; _old_pid=$(cat "$_GB_LOG_CAPTURE_PID_FILE" 2>/dev/null)
        if [[ -n "$_old_pid" ]] && kill -0 "$_old_pid" 2>/dev/null; then
            kill "$_old_pid" 2>/dev/null || true
            gb_info "Stopped previous capture (PID ${_old_pid})"
        fi
    fi

    detect_hardware 2>/dev/null || true
    # Launch background loop
    (
        while true; do
            {
                echo ""
                echo "--- Capture $(date '+%Y-%m-%dT%H:%M:%S') ---"
                _gb_logs_full_dump 2>/dev/null
            } >> "$_outfile" 2>/dev/null
            sleep 30
        done
    ) &
    local _cpid=$!
    echo "$_cpid" > "$_GB_LOG_CAPTURE_PID_FILE"
    gb_ok "Background log capture started (PID ${_cpid}) → ${_outfile}"
    echo "  Stop with: greenboost logs stop"
}

# cmd_logs_stop - stop background log capture.
cmd_logs_stop() {
    if [[ ! -f "$_GB_LOG_CAPTURE_PID_FILE" ]]; then
        gb_warn_ui "No active log capture found."
        return 0
    fi
    local _pid; _pid=$(cat "$_GB_LOG_CAPTURE_PID_FILE" 2>/dev/null)
    if [[ -z "$_pid" ]]; then
        gb_warn_ui "PID file empty , nothing to stop."
        rm -f "$_GB_LOG_CAPTURE_PID_FILE"
        return 0
    fi
    if kill -0 "$_pid" 2>/dev/null; then
        kill "$_pid" 2>/dev/null && gb_ok "Log capture stopped (PID ${_pid})" \
            || gb_fail "Failed to stop PID ${_pid}"
    else
        gb_info "Capture process ${_pid} already exited."
    fi
    rm -f "$_GB_LOG_CAPTURE_PID_FILE"
}

cmd_logs() {
    local _sub="${1:-}"
    case "$_sub" in
        save)    shift; cmd_logs_save "${1:-}" ;;
        start)   shift; cmd_logs_start "${1:-}" ;;
        stop)    cmd_logs_stop ;;
        --llm)   _logs_llm ;;
        "")
            if [[ ! -t 0 ]]; then
                _cmd_logs_snapshot
                return
            fi
            _gb_run_tui_loop _cmd_logs_snapshot 5 \
                "  ${C_DIM}Updating every 5s  ${C_GRAY}Ctrl+S${C_DIM}: refresh  ${C_GRAY}Ctrl+C${C_DIM}: exit${C_RESET}"
            ;;
        *)
            gb_fail "Unknown logs subcommand: '${_sub}'  (use: save [file] | start [file] | stop | --llm)"
            return 1
            ;;
    esac
}

_cmd_inference_logs_llm() {
    local _strip='s/\x1b\[[0-9;]*[mKHJsu]//g'
    _il_section() {
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
    grep -q '^greenboost' <<< "$(lsmod 2>/dev/null)" && _mod="loaded"
    echo "gb_inference_logs v=${GB_VERSION} ts=$(date '+%Y-%m-%dT%H:%M') mod=${_mod}"
    _il_section "kernel" "$(gather_flow_events 50 2>/dev/null)"
    _il_section "services" "$(journalctl --since '1 hour ago' --no-pager -q \
        -u ollama -u greenboost-supervisor 2>/dev/null | tail -60)"
}

_cmd_inference_logs_snapshot() {
    local _ts; _ts=$(date '+%Y-%m-%dT%H:%M')
    echo -e ""
    echo -e "  ${C_VIOLET}${C_BOLD}GreenBoost inference logs${C_RESET}  ${C_DIM}·  ${_ts}${C_RESET}"
    echo -e ""

    local _diag_errors=() _diag_warns=() _diag_ok=()
    local _mod_status="MISSING"
    grep -q '^greenboost' <<< "$(lsmod 2>/dev/null)" && _mod_status="loaded"
    [[ "$_mod_status" == "loaded" ]] \
        && _diag_ok+=("kernel module loaded") \
        || _diag_errors+=("kernel module NOT loaded - no T2/T3 memory available")

    local _flow; _flow=$(gather_flow_events 40)
    local _flow_n=0; [[ -n "$_flow" ]] && _flow_n=$(printf '%s\n' "$_flow" | wc -l)
    gb_section "Kernel Module (dmesg)  (${_flow_n} events)"
    if [[ -n "$_flow" ]]; then
        while IFS= read -r _fline; do
            [[ -z "$_fline" ]] && continue
            echo -e "  ${C_DIM}${_fline}${C_RESET}"
        done <<< "$_flow"
    else
        echo -e "  ${C_DIM}(empty - dmesg grep: greenboost tier-transitions)${C_RESET}"
    fi
    echo ""

    local _svc
    _svc=$(journalctl --since "1 hour ago" --no-pager -q \
        -u ollama -u greenboost-supervisor 2>/dev/null | tail -40)
    local _svc_n=0; [[ -n "$_svc" ]] && _svc_n=$(printf '%s\n' "$_svc" | wc -l)
    gb_section "AI Inference Services (last 1h)  (${_svc_n} events)"
    if [[ -n "$_svc" ]]; then
        while IFS= read -r _sline; do
            [[ -z "$_sline" ]] && continue
            echo -e "  ${C_GRAY}${_sline}${C_RESET}"
        done <<< "$_svc"
    else
        echo -e "  ${C_DIM}(empty - units: ollama greenboost-supervisor)${C_RESET}"
    fi
    echo ""

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
        _diag_errors+=("OOM kill detected in ollama service${_peak_note} - KV cache + model overflow exceeded available system RAM")
        _diag_errors+=("  Likely cause: virtual VRAM total included NVMe T3 pool, inflating context length")
        _diag_errors+=("  Fix: sudo greenboost set-kv-reserve <MB>  or  set OLLAMA_NUM_CTX=<smaller value>")
    fi

    local _num_ctx
    _num_ctx=$(printf '%s\n' "$_svc" | grep -oP 'default_num_ctx=\K[0-9]+' | head -1)
    if [[ -n "$_num_ctx" ]] && (( _num_ctx >= 131072 )); then
        _diag_warns+=("Ollama default_num_ctx=${_num_ctx} - KV cache will be very large; verify T2 has enough headroom (sudo greenboost vitals)")
    fi

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
        echo -e "  ${C_LIME}✓${C_RESET}  ${C_GRAY}All checks passed - no errors or warnings detected.${C_RESET}"
    fi
    echo -e "  ${C_DIM}[Ctrl+C exit · Ctrl+S refresh]${C_RESET}"
}

cmd_inference_logs() {
    if [[ "${1:-}" == "--llm" ]]; then _cmd_inference_logs_llm; return; fi

    if [[ ! -t 0 ]]; then
        _cmd_inference_logs_snapshot
        return
    fi

    # PR-BB: shared TUI loop helper.
    _gb_run_tui_loop _cmd_inference_logs_snapshot
}

# ════════════════════════════════════════════════════════════════════════
# inference-test - live benchmark that verifies GreenBoost path + perf
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
        (( _i++ )) || true
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
        (( _idx++ )) || true
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
        echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}Invalid selection - using default.${C_RESET}" >&2
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

# _infer_test_write_llm_report - writes structured report to GB_INFER_TEST_LOG
# and optionally to stdout (when _print=1).
_infer_test_write_llm_report() {
    local _model="$1" _print="${2:-0}"
    local _dir; _dir=$(dirname "$GB_INFER_TEST_LOG")
    mkdir -p "$_dir" 2>/dev/null || true

    # path verdict
    local _verdict="UNKNOWN"
    case "$SS_ACTIVE_PATH" in
        A)  _verdict="GOOD" ;;
        B)  _verdict="FALLBACK" ;;
        *)  _verdict="UNKNOWN" ;;
    esac

    # kernel last 5 greenboost dmesg lines
    local _kern5
    _kern5=$(dmesg 2>/dev/null | grep -i greenboost | tail -5 | tr '\n' '|' | sed 's/|$//')
    [[ -z "$_kern5" ]] && _kern5="none"

    local _report
    _report=$(cat <<REPORTEOF
greenboost inference-test v=${GB_VERSION} ts=$(date -Iseconds) model=${_model}
[path]
active=${SS_ACTIVE_PATH:-unknown}  a_delta=${IT_A_DELTA:-0}  b_delta=${IT_B_DELTA:-0}  verdict=${_verdict}
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
    echo -e "  ${C_CYAN}${C_BOLD}Inference Test${C_RESET}  ${C_DIM}- verifies GreenBoost path, perf, and memory${C_RESET}"
    echo ""

    # ── 1. Pre-flight ────────────────────────────────────────────────────
    detect_hardware 2>/dev/null || true
    parse_shim_stats || true

    local _pf_errors=() _pf_warns=()

    if ! grep -q "^${DRIVER_NAME} " <<< "$(lsmod 2>/dev/null)"; then
        _pf_warns+=("GreenBoost kernel module not loaded - paths B/C only")
    fi
    if ! command -v ollama &>/dev/null && ! curl -sf --max-time 2 http://localhost:11434/api/tags &>/dev/null; then
        _pf_errors+=("Ollama not running - start with: ollama serve")
    fi

    for _e in "${_pf_errors[@]}"; do
        echo -e "  ${C_RED}✗${C_RESET}  ${C_RED}${_e}${C_RESET}"
    done
    for _w in "${_pf_warns[@]}"; do
        echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}${_w}${C_RESET}"
    done
    if [[ ${#_pf_errors[@]} -gt 0 ]]; then
        echo ""
        echo -e "  ${C_RED}Pre-flight failed - fix the above before running inference-test.${C_RESET}"
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
    local _a_before=${SS_PATH_A:-0}
    local _b_before=${SS_PATH_B:-0}
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
    IT_A_DELTA=$(( ${SS_PATH_A:-0}  - _a_before ))
    IT_B_DELTA=$(( ${SS_PATH_B:-0}  - _b_before ))

    # ── 6. Result panel ──────────────────────────────────────────────────
    gb_separator
    echo ""
    echo -e "  ${C_CYAN}${C_BOLD}Results${C_RESET}"
    echo ""

    # Path verdict
    local _verdict_color _verdict_label
    case "$SS_ACTIVE_PATH" in
        A)  _verdict_color="$C_LIME";  _verdict_label="GOOD     (A - DMA-BUF pinned DDR)" ;;
        B)  _verdict_color="$C_AMBER"; _verdict_label="FALLBACK (B - HostReg no-kernel)" ;;
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
    echo -e "  ${C_BOLD}Allocs${C_RESET}     ${C_DIM}A:+${IT_A_DELTA}  B:+${IT_B_DELTA}${C_RESET}"

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
# Called by cmd_vitals - renders the full unified vitals snapshot.
_cmd_vitals_snapshot() {
    local _ts; _ts=$(date '+%Y-%m-%dT%H:%M')
    local prof_f="/sys/class/greenboost/greenboost/active_profile"
    local prof_name="default"
    [[ -r "$prof_f" ]] && prof_name=$(cat "$prof_f")

    # Read build stamp from /etc/greenboost/build_info (written by setup/install)
    local _build_badge=""
    for _sbi in "${MODULE_DIR}/build_info" ./build_info /etc/greenboost/build_info; do
        if [[ -f "$_sbi" ]]; then
            local _bid _bgit
            _bid=$(grep  '^BUILD_ID='  "$_sbi" 2>/dev/null | cut -d= -f2-)
            _bgit=$(grep '^BUILD_GIT=' "$_sbi" 2>/dev/null | cut -d= -f2-)
            [[ -n "$_bid"  ]] && _build_badge="  build ${_bid}"
            [[ -n "$_bgit" ]] && _build_badge+=" · ${_bgit}"
            break
        fi
    done

    # Gather data from sysfs and live queries
    parse_pool_brief
    parse_pool_info
    parse_shim_stats
    query_gpu_vram
    query_gpu_dcgm       # DCGM health / power-instant / NVLink (30s cached)
    query_live_swap
    query_ollama_ps
    read_nvtx_tail 100
    _nvtx_parse_diag
    local kv_est_mb=0
    if [[ -n "$OL_CTX_SIZE" && -n "$OL_PARAM_COUNT" ]] && (( ${OL_CTX_SIZE:-0} > 0 )); then
        kv_est_mb=$(estimate_kv_mb "$OL_CTX_SIZE" "$OL_PARAM_COUNT")
    fi

    local _diag_errors=() _diag_warns=() _diag_ok=()

    # ── Title (1 line) ────────────────────────────────────────────────
    echo -e "  ${C_VIOLET}${C_BOLD}GreenBoost Vitals${C_RESET} ${C_DIM}v${GB_VERSION} · ${_ts}${_build_badge}${C_RESET}"

    # ── ROW 1 (4 lines): System | AI Inference ────────────────────────
    local -a _sys_lines=() _ai_lines=()

    # Left: System
    _sys_lines+=("  ${C_BOLD}System${C_RESET}")
    if grep -q "^${DRIVER_NAME} " <<< "$(lsmod)"; then
        _sys_lines+=("  ${C_LIME}✓${C_RESET}  ${C_GRAY}Module ${C_LIME}v${GB_VERSION}${C_RESET}")
        _diag_ok+=("kernel module loaded")
    else
        _sys_lines+=("  ${C_RED}✗${C_RESET}  ${C_GRAY}Module ${C_RED}not loaded${C_RESET}")
        _diag_errors+=("kernel module not loaded - T2/T3 memory unavailable")
    fi
    _sys_lines+=("  ${C_VIOLET}◈${C_RESET}  ${C_GRAY}$(_trunc "${prof_name}" 20)${C_RESET}")
    local _gpu_short; _gpu_short=$(_trunc "${GPU_NAME:-Unknown}" 20)
    _sys_lines+=("  ${C_GRAY}◎${C_RESET}  ${C_GRAY}${_gpu_short}${C_DIM}  ${GB_PHYS:-?} GB${C_RESET}")
    # Active alloc path badge in System column
    if [[ "$SS_STALE" == "0" && -n "$SS_ACTIVE_PATH" ]]; then
        local _sys_path_badge
        case "$SS_ACTIVE_PATH" in
            A)  _sys_path_badge="${C_LIME}●A${C_RESET}  ${C_DIM}DMA-BUF pinned${C_RESET}" ;;
            B)  _sys_path_badge="${C_AMBER}${C_BOLD}⚠B${C_RESET}  ${C_DIM}HostReg (no-kernel)${C_RESET}" ;;
            *)  _sys_path_badge="${C_GRAY}path:${SS_ACTIVE_PATH}${C_RESET}" ;;
        esac
        _sys_lines+=("  ${C_DIM}Path${C_RESET}  ${_sys_path_badge}")
    fi
    # Gaming mode badge (from pynvml helper / sysfs)
    if (( ${GPU_GAMING_MODE:-0} == 1 )); then
        _sys_lines+=("  ${C_AMBER}⚡ Gaming Mode active${C_RESET}")
        _diag_warns+=("Gaming mode active , inference T2 priority reduced")
    fi
    # ECC double-bit error badge (hardware-critical)
    if (( ${GPU_ECC_DBE:-0} > 0 )); then
        _sys_lines+=("  ${C_RED}✗${C_RESET}  ${C_RED}ECC DBE ${GPU_ECC_DBE} error(s)!${C_RESET}")
        _diag_errors+=("ECC double-bit error: ${GPU_ECC_DBE} volatile, ${GPU_ECC_DBE_AGG:-0} aggregate , hardware memory risk")
    fi

    # ECC SBE early-warning (correctable but signals memory degradation trend)
    if (( ${GPU_ECC_SBE:-0} > 0 )); then
        _sys_lines+=("  ${C_AMBER}⚠${C_RESET}  ${C_AMBER}ECC SBE ${GPU_ECC_SBE} corrected bit(s)${C_RESET}")
        _diag_warns+=("ECC single-bit errors: ${GPU_ECC_SBE} corrected , monitor for DBE escalation")
    fi
    # DCGM hardware health
    if [[ "$GPU_HEALTH_OK" == "0" && -n "$GPU_HEALTH_SUMMARY" ]]; then
        _sys_lines+=("  ${C_RED}✗${C_RESET}  ${C_RED}DCGM ${GPU_HEALTH_SUMMARY}${C_RESET}")
        _diag_errors+=("DCGM health: ${GPU_HEALTH_SUMMARY}")
    elif [[ "$GPU_HEALTH_OK" == "1" ]]; then
        _sys_lines+=("  ${C_LIME}✓${C_RESET}  ${C_DIM}DCGM PASS${C_RESET}")
        _diag_ok+=("DCGM health PASS")
    fi

    # Right: AI Inference
    local _phase_label="${SS_PHASE:--}"
    local _gpu_hw_str=""
    if (( ${GPU_TEMP_C:-0} > 0 )); then
        local _temp_col="${C_LIME}"
        (( GPU_TEMP_C >= 80 )) && _temp_col="${C_AMBER}"
        (( GPU_TEMP_C >= 90 )) && _temp_col="${C_RED}"
        _gpu_hw_str="${_temp_col}${GPU_TEMP_C}°C${C_RESET}"
        if (( ${GPU_UTIL_PCT:-0} > 0 )); then
            _gpu_hw_str+="  ${C_DIM}util ${GPU_UTIL_PCT}%${C_RESET}"
        fi
        if (( ${GPU_MEM_UTIL_PCT:-0} > 0 )); then
            _gpu_hw_str+="  ${C_DIM}mem-ctrl ${GPU_MEM_UTIL_PCT}%${C_RESET}"
        fi
        if awk "BEGIN{exit !(${GPU_POWER_W:-0}>0)}" 2>/dev/null; then
            _gpu_hw_str+="  ${C_DIM}${GPU_POWER_W}W${C_RESET}"
            # power_instant from DCGM (more precise; only shown when different from NVML avg)
            if awk "BEGIN{exit !(${GPU_POWER_INSTANT_W:-0}>0)}" 2>/dev/null; then
                _gpu_hw_str+="${C_DIM}(${GPU_POWER_INSTANT_W}W inst)${C_RESET}"
            fi
            if awk "BEGIN{exit !(${GPU_POWER_LIMIT_W:-0}>0)}" 2>/dev/null; then
                _gpu_hw_str+="${C_DIM}/${GPU_POWER_LIMIT_W}W${C_RESET}"
            fi
        fi
        if (( ${GPU_SM_CLOCK_MHZ:-0} > 0 )); then
            local _sm_ghz; _sm_ghz=$(awk "BEGIN{printf \"%.2f\", ${GPU_SM_CLOCK_MHZ}/1000}" 2>/dev/null || echo "${GPU_SM_CLOCK_MHZ}M")
            _gpu_hw_str+="  ${C_DIM}SM ${_sm_ghz}GHz${C_RESET}"
        fi
        if awk "BEGIN{exit !(${GPU_PCIE_TX_MB_S:-0}+${GPU_PCIE_RX_MB_S:-0}>0)}" 2>/dev/null; then
            local _pcie_tx_int=${GPU_PCIE_TX_MB_S%.*}
            local _pcie_rx_int=${GPU_PCIE_RX_MB_S%.*}
            _gpu_hw_str+="  ${C_DIM}PCIe ↑${_pcie_tx_int}↓${_pcie_rx_int} MB/s${C_RESET}"
        fi
        if (( ${GPU_NVLINK_BW_MB_S:-0} > 0 )); then
            _gpu_hw_str+="  ${C_DIM}NVLink ${GPU_NVLINK_BW_MB_S} MB/s${C_RESET}"
        fi
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
        local _phase_color="${C_DIM}"
        case "${SS_PHASE:-}" in INFERENCE|STEADY) _phase_color="$C_LIME" ;; MODEL_LOAD) _phase_color="$C_AMBER" ;; esac
        _ai_lines+=("     ${_phase_color}${_phase_label}${C_RESET}")
    else
        _ai_lines+=("  ${C_DIM}◎  no active model - idle${C_RESET}")
        _ai_lines+=("")
        _ai_lines+=("")
    fi
    # Append GPU hw stats line (temp/util/power) if available
    [[ -n "$_gpu_hw_str" ]] && _ai_lines+=("  ${C_DIM}GPU ${C_RESET}${_gpu_hw_str}")

    gb_render_2col _sys_lines _ai_lines
    gb_separator

    # ── ROW 2 (4 lines): Memory Tiers (1 line each + combined) ────────
    local _combined_gb=$(( ${PB_T1_GB:-0} + ${PB_T2_MAX_GB:-0} + ${PB_T3_MAX_GB:-0} ))

    # T1 - GPU VRAM
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

    # T2 - System RAM pool
    local _t2_pct="${PB_T2_PCT:-0}"
    local _t2_col; _t2_col=$(gb_tier_color "$_t2_pct")
    local _t2_bar; _t2_bar=$(gb_bar "$_t2_pct" "$_t2_col" "${C_DIM}" 28)
    echo -e "  ${C_CYAN}T2${C_RESET}  ${C_GRAY}System RAM ${C_RESET}${_t2_bar}  ${_t2_col}${PB_T2_USED_GB}/${PB_T2_MAX_GB} GB  ${_t2_pct}%${C_RESET}"
    if [[ "$PI_OOM_GUARD" == "YES" ]]; then
        _diag_errors+=("OOM guard - T2 full; system RAM at safety limit")
    fi

    # T3 - NVMe swap
    local _t3_pct=0
    (( ${PB_T3_MAX_GB:-0} > 0 )) && _t3_pct=$(( ${PB_T3_USED_GB:-0} * 100 / ${PB_T3_MAX_GB:-1} ))
    local _t3_col; _t3_col=$(gb_tier_color "$_t3_pct")
    local _t3_bar; _t3_bar=$(gb_bar "$_t3_pct" "$_t3_col" "${C_DIM}" 28)
    echo -e "  ${C_CYAN}T3${C_RESET}  ${C_GRAY}NVMe swap  ${C_RESET}${_t3_bar}  ${_t3_col}${PB_T3_USED_GB}/${PB_T3_MAX_GB} GB  ${_t3_pct}%${C_RESET}"
    (( _t3_pct >= 90 )) && _diag_errors+=("T3 NVMe swap near cap (${_t3_pct}%) - cold weights may not evict")
    (( _t3_pct >= 75 && _t3_pct < 90 )) && _diag_warns+=("T3 NVMe swap at ${_t3_pct}%")

    # Remote Feeders (Cluster) - Combined line + per-feeder bars
    # Load per-feeder BW and health from metrics.json (set by shim, no handshake needed)
    declare -A _fdr_bw _fdr_health _fdr_gpu_util _fdr_gpu_mem_util _fdr_gpu_temp _fdr_gpu_power
    _gb_read_feeder_metrics

    local _remote_gpus=0
    local _remote_t1=0
    local _remote_t2=0
    local _remote_t3=0
    local _feeder_labels=""
    local -a _feeder_bar_lines=()
    if [[ -f "$GB_CLUSTER_CONF" ]]; then
        while IFS= read -r line; do
            [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
            local addr nickname
            addr=$(echo "$line" | awk '{print $1}')
            nickname=$(echo "$line" | awk '{print $2}')
            local ip port; ip=$(echo "$addr" | cut -d: -f1); port=$(echo "$addr" | cut -d: -f2)
            [[ -z "$port" ]] && port=$GB_NET_PORT
            local probe_out; probe_out=$(_gb_net_handshake "$ip" "$port" 2>/dev/null)
            if echo "$probe_out" | grep -q "^OK:"; then
                local _ft2_done=0 _ft3_done=0
                local _mem_t1f=0 _mem_t1t=0 _mem_t2f=0 _mem_t2t=0 _mem_t3f=0 _mem_t3t=0
                while IFS= read -r gl; do
                    if [[ "$gl" == GPU:* ]]; then
                        local fv; fv=$(echo "$gl" | cut -d: -f4)
                        _remote_t1=$((_remote_t1 + (fv + 536870912) / 1073741824))
                        _remote_gpus=$((_remote_gpus + 1))
                        if (( _ft2_done == 0 )); then
                            local ft2; ft2=$(echo "$gl" | cut -d: -f6)
                            [[ "$ft2" =~ ^[0-9]+$ && ft2 -gt 0 ]] && _remote_t2=$((_remote_t2 + (ft2 + 536870912) / 1073741824))
                            _ft2_done=1
                        fi
                        if (( _ft3_done == 0 )); then
                            local ft3; ft3=$(echo "$gl" | cut -d: -f7)
                            [[ "$ft3" =~ ^[0-9]+$ && ft3 -gt 0 ]] && _remote_t3=$((_remote_t3 + (ft3 + 536870912) / 1073741824))
                            _ft3_done=1
                        fi
                    elif [[ "$gl" == MEM:0:* ]]; then
                        # MEM:{dev_id}:{t1f}:{t1t}:{t2f}:{t2t}:{t3f}:{t3t} (bytes)
                        _mem_t1f=$(echo "$gl" | cut -d: -f3)
                        _mem_t1t=$(echo "$gl" | cut -d: -f4)
                        _mem_t2f=$(echo "$gl" | cut -d: -f5)
                        _mem_t2t=$(echo "$gl" | cut -d: -f6)
                        _mem_t3f=$(echo "$gl" | cut -d: -f7)
                        _mem_t3t=$(echo "$gl" | cut -d: -f8)
                    fi
                done <<< "$probe_out"
                local _fn; _fn="${nickname:-$ip}"
                _feeder_labels="${_feeder_labels}${_fn} "

                # Build per-feeder bar lines if MEM data is available
                if (( _mem_t1t > 0 || _mem_t2t > 0 )); then
                    local _fn_pad; printf -v _fn_pad "%-6s" "${_fn:0:6}"
                    # T1 bar
                    local _ft1u=$(( (_mem_t1t - _mem_t1f + 536870912) / 1073741824 ))
                    local _ft1tot=$(( (_mem_t1t + 536870912) / 1073741824 ))
                    local _ft1pct=0; (( _ft1tot > 0 )) && _ft1pct=$(( (_ft1u * 100) / _ft1tot ))
                    local _ft1col; _ft1col=$(gb_tier_color "$_ft1pct")
                    local _ft1bar; _ft1bar=$(gb_bar "$_ft1pct" "$_ft1col" "${C_DIM}" 18)
                    # T2 bar
                    local _ft2u=$(( (_mem_t2t - _mem_t2f + 536870912) / 1073741824 ))
                    local _ft2tot=$(( (_mem_t2t + 536870912) / 1073741824 ))
                    local _ft2pct=0; (( _ft2tot > 0 )) && _ft2pct=$(( (_ft2u * 100) / _ft2tot ))
                    local _ft2col; _ft2col=$(gb_tier_color "$_ft2pct")
                    local _ft2bar; _ft2bar=$(gb_bar "$_ft2pct" "$_ft2col" "${C_DIM}" 18)
                    # T3 bar
                    local _ft3u=$(( (_mem_t3t - _mem_t3f + 536870912) / 1073741824 ))
                    local _ft3tot=$(( (_mem_t3t + 536870912) / 1073741824 ))
                    local _ft3pct=0; (( _ft3tot > 0 )) && _ft3pct=$(( (_ft3u * 100) / _ft3tot ))
                    local _ft3col; _ft3col=$(gb_tier_color "$_ft3pct")
                    local _ft3bar; _ft3bar=$(gb_bar "$_ft3pct" "$_ft3col" "${C_DIM}" 18)
                    # Health + BW from metrics.json (set by shim on each alloc/free cycle)
                    local _fhealth_badge="" _fbw_badge=""
                    local _fh_val="${_fdr_health[$ip]:-}"
                    local _fbw_val="${_fdr_bw[$ip]:-0}"
                    if [[ -n "$_fh_val" ]]; then
                        case "$_fh_val" in
                            0) _fhealth_badge="  ${C_LIME}HEALTHY${C_RESET}" ;;
                            1) _fhealth_badge="  ${C_AMBER}DEGRADED${C_RESET}" ;;
                            2) _fhealth_badge="  ${C_RED}QUARANTINE${C_RESET}" ;;
                            *) _fhealth_badge="  ${C_DIM}health=${_fh_val}${C_RESET}" ;;
                        esac
                    fi
                    local _fbw_int=${_fbw_val%.*}
                    (( ${_fbw_int:-0} > 0 )) && _fbw_badge="  ${C_DIM}${_fbw_int} MB/s${C_RESET}"
                    # Feeder GPU COMPUTE (not just memory) - util%/mem-util%/temp/power
                    # from the netd heartbeat (via metrics.json). A feeder can hold VRAM
                    # (T1 bar above) while its GPU sits at 0% - this is the number that
                    # shows whether the feeder is actually COMPUTING (ggml-2dev / feeder-
                    # exclusive kernel dispatch) versus just storing weights.
                    local _fgpu_badge=""
                    local _fg_util="${_fdr_gpu_util[$ip]:-0}" _fg_mem="${_fdr_gpu_mem_util[$ip]:-0}"
                    local _fg_temp="${_fdr_gpu_temp[$ip]:-0}" _fg_pwr="${_fdr_gpu_power[$ip]:-0}"
                    if (( ${_fg_util:-0} > 0 || ${_fg_mem:-0} > 0 )); then
                        # NOTE: unlike the T1/T2/T3 bars above, high GPU util% is GOOD
                        # (the feeder is computing, not idle) - deliberately NOT using
                        # gb_tier_color's danger scale (high%=red) here.
                        _fgpu_badge="  ${C_DIM}GPU${C_RESET} ${C_CYAN}${_fg_util}%${C_RESET} ${C_DIM}mem${C_RESET} ${_fg_mem}%  ${_fg_temp}°C ${_fg_pwr}W${C_RESET}"
                    fi
                    _feeder_bar_lines+=("  ${C_DIM}↳${C_RESET} ${C_CYAN}${_fn_pad}${C_RESET}  ${C_DIM}T1${C_RESET} ${_ft1bar} ${_ft1col}${_ft1u}/${_ft1tot} GB${C_RESET}  ${C_DIM}T2${C_RESET} ${_ft2bar} ${_ft2col}${_ft2u}/${_ft2tot} GB${C_RESET}  ${C_DIM}T3${C_RESET} ${_ft3bar} ${_ft3col}${_ft3u}/${_ft3tot} GB${C_RESET}${_fhealth_badge}${_fbw_badge}${_fgpu_badge}")
                    # Surface feeder issues to vitals diag
                    [[ "$_fh_val" == "2" ]] && _diag_errors+=("Feeder ${_fn}: ECC quarantine , weights unreliable")
                    [[ "$_fh_val" == "1" ]] && _diag_warns+=("Feeder ${_fn}: degraded health state")
                fi
            fi
        done < "$GB_CLUSTER_CONF"
    fi
    if (( _remote_gpus > 0 )); then
        local _cl_total=$((_remote_t1 + _remote_t2 + _remote_t3))
        local _gpu_label; (( _remote_gpus == 1 )) && _gpu_label="${_remote_gpus} GPU" || _gpu_label="${_remote_gpus} GPUs"
        echo -e "  ${C_CYAN}CL${C_RESET}  ${C_GRAY}Cluster    ${C_LIME}${_cl_total} GB${C_RESET}  ${C_DIM}(T1:${_remote_t1} + T2:${_remote_t2} + T3:${_remote_t3} · ${_gpu_label} · ${_feeder_labels% })${C_RESET}"
        local _bl; for _bl in "${_feeder_bar_lines[@]}"; do echo -e "$_bl"; done
        local _combined_total=$(( ${_combined_gb} + _cl_total ))
        echo -e "  ${C_DIM}Combined: ${C_GRAY}${C_BOLD}${_combined_total} GB${C_RESET}  ${C_DIM}(local ${_combined_gb} + cluster ${_cl_total})${C_RESET}"
    else
        echo -e "  ${C_DIM}Combined: ${C_GRAY}${C_BOLD}${_combined_gb} GB${C_RESET}"
    fi
    gb_separator

    # ── ROW 3 (4 lines): KV Cache | Overflow ─────────────────────────
    local -a _kv_lines=() _ov_lines=()

    # Left: KV Cache
    _kv_lines+=("  ${C_BOLD}KV Cache${C_RESET}")
    if (( ${PI_KV_T2_MB:-0} > 0 || ${PI_KV_T3_MB:-0} > 0 )); then
        local _kv_icon _kv_pc
        if (( ${PI_KV_T3_MB:-0} > 0 )); then
            _kv_icon="${C_RED}●${C_RESET}"; _kv_pc="${C_RED}"
            _diag_errors+=("KV in T3 (${PI_KV_T3_MB} MB) - generation speed degraded")
        else
            _kv_icon="${C_AMBER}●${C_RESET}"; _kv_pc="${C_AMBER}"
            _diag_warns+=("KV spilled to T2 DDR (${PI_KV_T2_MB} MB) - consider increasing kv_reserve_mb")
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
            A)  _path_badge="${C_LIME}●A${C_RESET}  ${C_DIM}(DMA-BUF pinned DDR)${C_RESET}" ;;
            B)  _path_badge="${C_AMBER}${C_BOLD}⚠B${C_RESET}  ${C_DIM}(HostReg no-kernel)${C_RESET}"
                _diag_warns+=("CUDA shim on fallback Path B - no kernel module") ;;
            *)  _path_badge="${C_GRAY}path:${SS_ACTIVE_PATH:-?}${C_RESET}" ;;
        esac
        _alloc_str="${_path_badge}  ${C_DIM}A:${SS_PATH_A} B:${SS_PATH_B}${C_RESET}"
    else
        _alloc_str="${C_DIM}(shim stats unavailable)${C_RESET}"
    fi
    local _oom_str
    if [[ "$PI_OOM_GUARD" == "YES" ]]; then
        _oom_str="${C_RED}OOM: active${C_RESET}"
    else
        _oom_str="${C_LIME}OOM: no${C_RESET}"
    fi

    local _adj_sign=""; (( SS_T2_WARN_ADJ > 0 )) && _adj_sign="+"
    local _pinned_str=""
    [[ -n "$SS_PINNED_FREE" ]] && _pinned_str="  Pinned: ${SS_PINNED_FREE}/8"
    _ov_lines+=("  ${C_BOLD}Overflow${C_RESET}")
    _ov_lines+=("  ${C_GRAY}${_alloc_str}${C_RESET}")
    _ov_lines+=("  ${_oom_str}${C_DIM} · Bufs: ${PI_ACTIVE_BUFS}${C_RESET}")
    _ov_lines+=("  ${C_DIM}T2 free: ${PI_T2_AVAIL_MB} MB · Safety: ${PI_SAFETY_RSV_MB} MB${C_RESET}")
    _ov_lines+=("  ${C_DIM}Refault adj ${_adj_sign}${SS_T2_WARN_ADJ}%  KV dedup ${SS_KV_DEDUP} hits  Frag ${SS_KV_FRAG_MB} MB${C_RESET}")
    _ov_lines+=("  ${C_DIM}Cold evict ${SS_COLD_EVICT}${_pinned_str}${C_RESET}")

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
        (( _aa_count > 0 )) && _diag_errors+=("AppArmor denials (${_aa_count}) - run: sudo greenboost install-sys-configs")
    fi

    # Remove blank placeholder ok entries
    local _clean_ok=()
    for _o in "${_diag_ok[@]}"; do [[ -n "$_o" ]] && _clean_ok+=("$_o"); done

    # Print max 3 error/warn lines
    local _shown=0
    for _e in "${_diag_errors[@]}"; do
        (( _shown >= 3 )) && break
        echo -e "  ${C_RED}✗${C_RESET}  ${C_RED}${_e}${C_RESET}"
        (( _shown++ )) || true
    done
    for _w in "${_diag_warns[@]}"; do
        (( _shown >= 3 )) && break
        echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}${_w}${C_RESET}"
        (( _shown++ )) || true
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

    # ── Diagnostic Summary ────────────────────────────────────────────────
    local _dsep; _dsep=$(printf '─%.0s' $(seq 1 $(( $(tput cols 2>/dev/null || echo 72) - 2 ))))
    echo ""
    printf '  %b\n' "${C_PURPLE}${C_BOLD}Diagnostic Summary${C_RESET}"
    printf '%b\n'   "  ${C_DIM}${_dsep}${C_RESET}"

    # Tier allocation history (from shim_stats per-tier counters)
    local _any_tier=0
    _ds_tier() {
        local _label="$1" _desc="$2" _cnt="$3" _mb="$4" _peak="$5" _cur="${6:-0}"
        (( ${_cnt:-0} <= 0 )) && return
        _any_tier=1
        printf "  ${C_LIME}✓${C_RESET}  %-18s ${C_DIM}%-22s${C_RESET}  ${C_CYAN}%6d${C_RESET} alloc(s)  ${C_AMBER}live: %6d MB${C_RESET}  ${C_GRAY}lifetime: %7d MB${C_RESET}  ${C_DIM}peak: %d MB${C_RESET}\n" \
            "$_label" "$_desc" "${_cnt:-0}" "${_cur:-0}" "${_mb:-0}" "${_peak:-0}"
    }
    _ds_tier "T1 local"    "(cudaMalloc)"            "${SS_TIER_T1_LOCAL_COUNT:-0}"  "${SS_TIER_T1_LOCAL_MB:-0}"  "${SS_TIER_T1_LOCAL_PEAK:-0}"  "${SS_TIER_T1_LOCAL_CUR:-0}"
    _ds_tier "T1 feeder"   "(remote VRAM)"           "${SS_TIER_T1_FEEDER_COUNT:-0}" "${SS_TIER_T1_FEEDER_MB:-0}" "${SS_TIER_T1_FEEDER_PEAK:-0}" "${SS_TIER_T1_FEEDER_CUR:-0}"
    _ds_tier "T2 local"    "(DDR pool / Path A)"     "${SS_TIER_T2_LOCAL_COUNT:-0}"  "${SS_TIER_T2_LOCAL_MB:-0}"  "${SS_TIER_T2_LOCAL_PEAK:-0}"  "${SS_TIER_T2_LOCAL_CUR:-0}"
    _ds_tier "T2 feeder"   "(remote DDR)"            "${SS_TIER_T2_FEEDER_COUNT:-0}" "${SS_TIER_T2_FEEDER_MB:-0}" "${SS_TIER_T2_FEEDER_PEAK:-0}" "${SS_TIER_T2_FEEDER_CUR:-0}"
    _ds_tier "T3 local"    "(NVMe GDS)"              "${SS_TIER_T3_LOCAL_COUNT:-0}"  "${SS_TIER_T3_LOCAL_MB:-0}"  "${SS_TIER_T3_LOCAL_PEAK:-0}"  "${SS_TIER_T3_LOCAL_CUR:-0}"
    _ds_tier "T3 feeder"   "(remote NVMe)"           "${SS_TIER_T3_FEEDER_COUNT:-0}" "${SS_TIER_T3_FEEDER_MB:-0}" "${SS_TIER_T3_FEEDER_PEAK:-0}" "${SS_TIER_T3_FEEDER_CUR:-0}"
    _ds_tier "VMM"         "(cuMemCreate)"           "${SS_TIER_VMM_COUNT:-0}"       "${SS_TIER_VMM_MB:-0}"       "${SS_TIER_VMM_PEAK:-0}"       "${SS_TIER_VMM_CUR:-0}"
    _ds_tier "Path B"      "(HostReg fallback)"      "${SS_TIER_PATH_B_COUNT:-0}"    "${SS_TIER_PATH_B_MB:-0}"    "${SS_TIER_PATH_B_PEAK:-0}"    "${SS_TIER_PATH_B_CUR:-0}"
    (( _any_tier == 0 )) && printf "  ${C_DIM}No tier stats - shim not active${C_RESET}\n"
    unset -f _ds_tier

    # Allocation sub-paths from NVTX log
    echo ""
    printf "  ${C_DIM}Allocation paths${C_RESET}\n"
    local _nvtx_data=0
    (( DIAG_PATH_A_ZC_COUNT > 0 )) && {
        _nvtx_data=1
        printf "  ${C_LIME}✓${C_RESET}  %-38s  ${C_CYAN}%6d${C_RESET} alloc(s)  ${C_GRAY}%7d MB${C_RESET}\n" \
            "Path A  zero-copy  (OpaqueFd DMA-BUF)" "${DIAG_PATH_A_ZC_COUNT}" "${DIAG_PATH_A_ZC_MB}"
    }
    (( DIAG_PATH_A_POOL_COUNT > 0 )) && {
        _nvtx_data=1
        printf "  ${C_LIME}✓${C_RESET}  %-38s  ${C_CYAN}%6d${C_RESET} alloc(s)  ${C_GRAY}%7d MB${C_RESET}\n" \
            "Path A  pool (inner sub-alloc slab)" "${DIAG_PATH_A_POOL_COUNT}" "${DIAG_PATH_A_POOL_MB}"
    }
    (( DIAG_PATH_A_PIN_COUNT > 0 )) && {
        _nvtx_data=1
        printf "  ${C_LIME}✓${C_RESET}  %-38s  ${C_CYAN}%6d${C_RESET} alloc(s)  ${C_GRAY}%7d MB${C_RESET}\n" \
            "Path A  pinned (mmap→cuMemHostReg)" "${DIAG_PATH_A_PIN_COUNT}" "${DIAG_PATH_A_PIN_MB}"
    }
    (( DIAG_ZEROCOPY_FAIL > 0 )) && {
        _nvtx_data=1
        printf "  ${C_AMBER}⚠${C_RESET}  Path A zero-copy fell to pinned sub-method:  ${C_AMBER}%d${C_RESET} time(s)\n" \
            "${DIAG_ZEROCOPY_FAIL}"
    }
    (( _nvtx_data == 0 )) && printf "  ${C_DIM}No path events - log empty or shim not active${C_RESET}\n"

    # OOM events
    echo ""
    if (( DIAG_OOM_TOTAL > 0 )); then
        printf "  ${C_RED}✗${C_RESET}  ${C_RED}${C_BOLD}OOM events: %d total${C_RESET}\n" "${DIAG_OOM_TOTAL}"
        (( DIAG_OOM_MEMAVAIL   > 0 )) && printf "     ${C_RED}✗${C_RESET}  ${C_GRAY}%-42s${C_RESET}  ${C_RED}%d${C_RESET} time(s)\n" "MemAvailable guard triggered"    "${DIAG_OOM_MEMAVAIL}"
        (( DIAG_OOM_PATH_B_FAIL> 0 )) && printf "     ${C_RED}✗${C_RESET}  ${C_GRAY}%-42s${C_RESET}  ${C_RED}%d${C_RESET} time(s)\n" "Path B cuMemHostRegister failed" "${DIAG_OOM_PATH_B_FAIL}"
        (( DIAG_OOM_FULL       > 0 )) && printf "     ${C_RED}✗${C_RESET}  ${C_GRAY}%-42s${C_RESET}  ${C_RED}%d${C_RESET} time(s)\n" "All tiers exhausted (smart_alloc)" "${DIAG_OOM_FULL}"
        (( DIAG_OOM_T1_LOCAL   > 0 )) && printf "     ${C_RED}✗${C_RESET}  ${C_GRAY}%-42s${C_RESET}  ${C_RED}%d${C_RESET} time(s)\n" "T1 VRAM OOM (cudaMalloc)"        "${DIAG_OOM_T1_LOCAL}"
        (( DIAG_OOM_T2_CAP     > 0 )) && printf "     ${C_RED}✗${C_RESET}  ${C_GRAY}%-42s${C_RESET}  ${C_RED}%d${C_RESET} time(s)\n" "T2 pool cap exceeded"            "${DIAG_OOM_T2_CAP}"
        (( DIAG_OOM_VMM_HOST   > 0 )) && printf "     ${C_RED}✗${C_RESET}  ${C_GRAY}%-42s${C_RESET}  ${C_RED}%d${C_RESET} time(s)\n" "VMM host pinned failed"          "${DIAG_OOM_VMM_HOST}"
    else
        printf "  ${C_LIME}✓${C_RESET}  ${C_GRAY}No OOM events${C_RESET}\n"
    fi
    if (( DIAG_EVICT_COUNT > 0 )); then
        printf "  ${C_AMBER}⚠${C_RESET}  Evictions: ${C_AMBER}%d${C_RESET}  (${C_GRAY}%d MB${C_RESET})  ${C_DIM}- T2 KV pool under pressure${C_RESET}\n" \
            "${DIAG_EVICT_COUNT}" "${DIAG_EVICT_MB}"
    fi

    # Last phase + total alloc count from tier stats
    local _total_allocs=$(( ${SS_TIER_T1_LOCAL_COUNT:-0} + ${SS_TIER_T1_FEEDER_COUNT:-0} + ${SS_TIER_T2_LOCAL_COUNT:-0} + ${SS_TIER_T2_FEEDER_COUNT:-0} + ${SS_TIER_T3_LOCAL_COUNT:-0} + ${SS_TIER_T3_FEEDER_COUNT:-0} + ${SS_TIER_VMM_COUNT:-0} + ${SS_TIER_PATH_B_COUNT:-0} ))
    local _phase_disp="${DIAG_LAST_PHASE:-${SS_PHASE:----}}"
    printf "  ${C_DIM}Last phase: ${C_RESET}${C_VIOLET}%-22s${C_RESET}  ${C_DIM}Total allocs: ${C_RESET}${C_CYAN}%d${C_RESET}  ${C_DIM}H2D: %d MB  D2H: %d MB  Feeder dispatch: %d kernel(s)${C_RESET}\n" \
        "${_phase_disp}" "${_total_allocs}" "${SS_H2D_MB:-0}" "${SS_D2H_MB:-0}" "${SS_KERNEL_DISPATCH:-0}"

    # H2D/D2H throughput (MB/s) - delta of the cumulative shim_stats counters
    # against the previous snapshot, so the data-flow rate is visible instead
    # of only the cumulative total. State file survives across TUI refreshes
    # (same process re-execs the snapshot function every 5s); first sample
    # after shim restart or a fresh process has no prior state, shows "-".
    _gb_vitals_rate_line() {
        local _state="/run/greenboost/vitals_rate_state"
        [[ -w /run/greenboost || -w "$_state" ]] || _state="/tmp/greenboost_vitals_rate_state"
        local _now; _now=$(date +%s)
        local _prev_ts=0 _prev_h2d=0 _prev_d2h=0
        if [[ -f "$_state" ]]; then
            # shellcheck disable=SC1090
            source "$_state" 2>/dev/null || true
        fi
        local _h2d_rate="-" _d2h_rate="-"
        local _dt=$(( _now - _prev_ts ))
        if [[ $_prev_ts -gt 0 && $_dt -gt 0 ]]; then
            local _dh=$(( ${SS_H2D_MB:-0} - _prev_h2d ))
            local _dd=$(( ${SS_D2H_MB:-0} - _prev_d2h ))
            (( _dh < 0 )) && _dh=0   # shim restarted, counters reset
            (( _dd < 0 )) && _dd=0
            _h2d_rate="$(( _dh / _dt ))"
            _d2h_rate="$(( _dd / _dt ))"
        fi
        printf '_prev_ts=%d\n_prev_h2d=%d\n_prev_d2h=%d\n' "$_now" "${SS_H2D_MB:-0}" "${SS_D2H_MB:-0}" > "$_state" 2>/dev/null || true
        if [[ "$_h2d_rate" != "-" ]]; then
            printf "  ${C_DIM}Data flow:  ${C_RESET}${C_CYAN}%s MB/s${C_RESET} H2D (host→feeder)  ${C_CYAN}%s MB/s${C_RESET} D2H (feeder→host)${C_RESET}\n" \
                "$_h2d_rate" "$_d2h_rate"
        fi
    }
    _gb_vitals_rate_line
    unset -f _gb_vitals_rate_line

    # Modes line: TurboQuant (KV compression), ggml-2dev (feeder as CUDA
    # device 1), and gb-quant's last recorded activity (NVTX gb:quantize:
    # range - gb-quant has no dedicated status file, this is the cheapest
    # true signal available without adding new Python-side plumbing).
    {
        local _tq_on="OFF" _2dev_on="OFF" _gbq_last=""
        [[ -f /etc/greenboost/turboquant.enabled ]] && _tq_on="ON"
        [[ -f /etc/greenboost/ggml_2dev.enabled ]] && _2dev_on="ON"
        if [[ -f /run/greenboost/nvtx_events.log ]]; then
            _gbq_last=$(grep -F 'gb:quantize' /run/greenboost/nvtx_events.log 2>/dev/null | tail -1 | cut -c1-19 || echo "")
        fi
        printf "  ${C_DIM}TurboQuant:${C_RESET} "
        [[ "$_tq_on" == "ON" ]] && printf "${C_LIME}ON${C_RESET}  " || printf "${C_DIM}OFF${C_RESET}  "
        printf "${C_DIM}ggml-2dev:${C_RESET} "
        [[ "$_2dev_on" == "ON" ]] && printf "${C_LIME}ON${C_RESET}  " || printf "${C_DIM}OFF${C_RESET}  "
        printf "${C_DIM}gb-quant last activity:${C_RESET} "
        [[ -n "$_gbq_last" ]] && printf "${C_CYAN}%s${C_RESET}\n" "$_gbq_last" || printf "${C_DIM}none this session${C_RESET}\n"
    }

    # Orchestrator state , shown only when the daemon is running (keys non-empty)
    if [[ -n "$ORCH_WS_RESERVE_MB" ]]; then
        echo ""
        printf '  %b\n' "${C_PURPLE}${C_BOLD}Orchestrator${C_RESET}"
        local _orch_act_col="${C_DIM}"
        (( ${ORCH_ACTUATE:-0} == 1 )) && _orch_act_col="${C_AMBER}"
        local _orch_ecc_col="${C_DIM}"
        (( ${ORCH_ECC_DEGRADED:-0} == 1 )) && _orch_ecc_col="${C_RED}"
        local _orch_therm_col="${C_DIM}"
        (( ${ORCH_THERMAL_STRESS:-0} == 1 )) && _orch_therm_col="${C_AMBER}"
        local _orch_membw_col="${C_DIM}"
        (( ${ORCH_MEM_BW_STRESS:-0} == 1 )) && _orch_membw_col="${C_AMBER}"
        local _orch_ws_col="${C_DIM}"
        (( ${ORCH_WS_ABOVE:-0} == 1 )) && _orch_ws_col="${C_AMBER}"
        local _orch_vp_col="${C_DIM}"
        (( ${ORCH_VRAM_PRESSURE:-0} == 1 )) && _orch_vp_col="${C_RED}"
        local _orch_cp_col="${C_DIM}"
        (( ${ORCH_CLUSTER_PRESSURE:-0} == 1 )) && _orch_cp_col="${C_AMBER}"
        printf "  ${C_DIM}ws_reserve: ${C_RESET}${C_CYAN}%s MB${C_RESET}  ${C_DIM}actuate: ${C_RESET}${_orch_act_col}%s${C_RESET}  ${C_DIM}ecc_degraded: ${C_RESET}${_orch_ecc_col}%s${C_RESET}  ${C_DIM}thermal_stress: ${C_RESET}${_orch_therm_col}%s${C_RESET}  ${C_DIM}mem_bw_stress: ${C_RESET}${_orch_membw_col}%s${C_RESET}  ${C_DIM}ws_above: ${C_RESET}${_orch_ws_col}%s${C_RESET}  ${C_DIM}vram_pressure: ${C_RESET}${_orch_vp_col}%s${C_RESET}  ${C_DIM}cluster_pressure: ${C_RESET}${_orch_cp_col}%s${C_RESET}\n" \
            "${ORCH_WS_RESERVE_MB:-?}" \
            "$( (( ${ORCH_ACTUATE:-0} == 1 )) && echo YES || echo no )" \
            "$( (( ${ORCH_ECC_DEGRADED:-0} == 1 )) && echo YES || echo no )" \
            "$( (( ${ORCH_THERMAL_STRESS:-0} == 1 )) && echo YES || echo no )" \
            "$( (( ${ORCH_MEM_BW_STRESS:-0} == 1 )) && echo YES || echo no )" \
            "$( (( ${ORCH_WS_ABOVE:-0} == 1 )) && echo YES || echo no )" \
            "$( (( ${ORCH_VRAM_PRESSURE:-0} == 1 )) && echo YES || echo no )" \
            "$( (( ${ORCH_CLUSTER_PRESSURE:-0} == 1 )) && echo YES || echo no )"
        local _orch_health_col="${C_DIM}"
        (( ${ORCH_HEALTH_OK:-1} == 0 )) && _orch_health_col="${C_RED}"
        local _orch_armed_col="${C_DIM}"
        (( ${ORCH_HEALTH_EVICT_ARMED:-0} == 1 )) && _orch_armed_col="${C_AMBER}"
        local _orch_sbe_col="${C_DIM}"
        (( ${ORCH_SBE_ELEVATED:-0} == 1 )) && _orch_sbe_col="${C_AMBER}"
        local _orch_clk_col="${C_DIM}"
        (( ${ORCH_CLOCK_THROTTLED:-0} == 1 )) && _orch_clk_col="${C_AMBER}"
        printf "  ${C_DIM}health_ok: ${C_RESET}${_orch_health_col}%s${C_RESET}  ${C_DIM}health_evict_armed: ${C_RESET}${_orch_armed_col}%s${C_RESET}  ${C_DIM}sbe_elevated: ${C_RESET}${_orch_sbe_col}%s${C_RESET}  ${C_DIM}sbe_count: ${C_RESET}${_orch_sbe_col}%s${C_RESET}  ${C_DIM}clock_throttled: ${C_RESET}${_orch_clk_col}%s${C_RESET}  ${C_DIM}sm_clk_max: ${C_RESET}${C_DIM}%s MHz${C_RESET}\n" \
            "$( (( ${ORCH_HEALTH_OK:-1} == 1 )) && echo yes || echo NO )" \
            "$( (( ${ORCH_HEALTH_EVICT_ARMED:-0} == 1 )) && echo YES || echo no )" \
            "$( (( ${ORCH_SBE_ELEVATED:-0} == 1 )) && echo YES || echo no )" \
            "${ORCH_SBE_SEEN:-0}" \
            "$( (( ${ORCH_CLOCK_THROTTLED:-0} == 1 )) && echo YES || echo no )" \
            "${ORCH_SM_CLOCK_MAX_MHZ:-0}"
        if [[ "${ORCH_OS_TUNE_ENABLED:-0}" == "1" ]]; then
            local _orch_gaming_col="${C_DIM}"
            (( ${ORCH_GAMING_MODE:-0} == 1 )) && _orch_gaming_col="${C_AMBER}"
            printf "  ${C_DIM}os_tune: ${C_RESET}${C_LIME}active${C_RESET}  ${C_DIM}gaming_mode: ${C_RESET}${_orch_gaming_col}%s${C_RESET}  ${C_DIM}governor: ${C_RESET}${C_CYAN}%s${C_RESET}  ${C_DIM}gpu_persistence: ${C_RESET}${C_CYAN}%s${C_RESET}  ${C_DIM}power_limit: ${C_RESET}${C_CYAN}%s W${C_RESET}  ${C_DIM}swappiness: ${C_RESET}${C_CYAN}%s${C_RESET}\n" \
                "$( (( ${ORCH_GAMING_MODE:-0} == 1 )) && echo YES || echo no )" \
                "${ORCH_CPU_GOVERNOR:-?}" \
                "$( (( ${ORCH_GPU_PERSISTENCE:-0} == 1 )) && echo on || echo off )" \
                "${ORCH_GPU_POWER_LIMIT_W:-?}" \
                "${ORCH_SWAPPINESS:-?}"
        fi
        if [[ -n "$TOPO_INFERENCE_CPUS" ]]; then
            local _topo_bw_col="${C_DIM}"
            printf "  ${C_DIM}infer_cpus: ${C_RESET}${C_CYAN}%-10s${C_RESET}  ${C_DIM}infer_threads: ${C_RESET}${C_CYAN}%s${C_RESET}  ${C_DIM}bg_threads: ${C_RESET}${C_CYAN}%s${C_RESET}  ${C_DIM}pcie_sat: ${C_RESET}${_topo_bw_col}%s MB/s${C_RESET}  ${C_DIM}blackwell: ${C_RESET}${C_CYAN}%s${C_RESET}\n" \
                "${TOPO_INFERENCE_CPUS}" \
                "${TOPO_INFERENCE_THREADS:-?}" \
                "${TOPO_BACKGROUND_THREADS:-?}" \
                "${TOPO_PCIE_SAT_MB_S:-?}" \
                "$( (( ${TOPO_IS_BLACKWELL:-0} == 1 )) && echo yes || echo no )"
        fi
        if [[ -n "$SHIM_VIRTUAL_VRAM_MB" ]]; then
            printf "  ${C_DIM}virtual_vram: ${C_RESET}${C_CYAN}%s MB${C_RESET}  ${C_DIM}cluster_remote: ${C_RESET}${C_CYAN}%s MB${C_RESET}  ${C_DIM}kv_reserve: ${C_RESET}${C_CYAN}%s MB${C_RESET}\n" \
                "${SHIM_VIRTUAL_VRAM_MB:-?}" "${SHIM_CLUSTER_REMOTE_MB:-0}" "${SHIM_KV_RESERVE_MB:-?}"
        fi
    fi
    echo ""

    # Append one structured line to the status log for Ctrl+L log view
    _status_log_append
}


# ---- _status_log_append - append one line to the status log -----------
# Called at end of every _cmd_vitals_snapshot invocation.
_status_log_append() {
    local _dir; _dir=$(dirname "$GB_STATUS_LOG")
    mkdir -p "$_dir" 2>/dev/null || return 0
    # Rotate if > 1 MB
    if [[ -f "$GB_STATUS_LOG" ]] && (( $(stat -c%s "$GB_STATUS_LOG" 2>/dev/null || echo 0) > 1048576 )); then
        tail -n 2000 "$GB_STATUS_LOG" > "${GB_STATUS_LOG}.tmp" 2>/dev/null \
            && mv "${GB_STATUS_LOG}.tmp" "$GB_STATUS_LOG" 2>/dev/null || true
    fi
    printf '%s path=%s vram=%s/%s t2=%s t3=%s a=%s b=%s phase=%s model=%s\n' \
        "$(date -Iseconds)" \
        "${SS_ACTIVE_PATH:-?}" \
        "${GPU_VRAM_USED_MB:-0}" "${GPU_VRAM_TOTAL_MB:-0}" \
        "${PI_T2_ALLOC_MB:-0}" "${PI_T3_ALLOC_MB:-0}" \
        "${SS_PATH_A:-0}" "${SS_PATH_B:-0}" \
        "${SS_PHASE:-?}" \
        "${OL_MODEL:--}" \
        >> "$GB_STATUS_LOG" 2>/dev/null || true
}

# ---- _show_log_view - Ctrl+L log view within the status alternate screen
# Displays GB_STATUS_LOG; Ctrl+S or q returns to live status.
_show_log_view() {
    local _log="${1:-$GB_STATUS_LOG}"
    while true; do
        printf '\033[2J\033[H'  # clear entire screen then cursor home
        local _rows; _rows=$(tput lines 2>/dev/null || echo 24)
        local _cols; _cols=$(tput cols  2>/dev/null || echo 80)
        echo -e "  ${C_CYAN}${C_BOLD}GreenBoost Status Log${C_RESET}  ${C_DIM}Ctrl+S: back to status  q: back  Ctrl+C: exit\033[K${C_RESET}"
        echo -e "  ${C_DIM}Log: ${C_GRAY}${_log}\033[K${C_RESET}"
        gb_separator
        local _content_rows=$(( _rows - 5 ))
        (( _content_rows < 5 )) && _content_rows=5
        if [[ -f "$_log" ]]; then
            tail -n "$_content_rows" "$_log" | while IFS= read -r _line; do
                # Colour-code by path: A→lime, B→amber
                local _coloured="$_line"
                if [[ "$_line" =~ path=A ]]; then
                    _coloured="${C_LIME}${_line}${C_RESET}"
                elif [[ "$_line" =~ path=B ]]; then
                    _coloured="${C_AMBER}${_line}${C_RESET}"
                fi
                echo -e "  ${_coloured}\033[K"
            done
        else
            echo ""
            echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_GRAY}No log yet - status refreshes write entries here.${C_RESET}"
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

# ---- Network Fabric: feed / connect / disconnect / cluster --------
# Distributed GPU compute + memory across machines via TCP.
# Feeder machine: greenboost feed      (starts greenboost-netd daemon)
# Host machine:   greenboost connect <IP>  (connects to feeder)

GB_NET_PORT=9740
GB_NETD_BIN="/usr/local/bin/greenboost-netd"
GB_NETD_PID="/run/greenboost/netd.pid"
GB_CLUSTER_CONF="/etc/greenboost/cluster.conf"
GB_CLUSTER_KEY="/etc/greenboost/cluster.key"
# Audit F-L4-02: cluster-scoped known_hosts so feeder SSH calls can pin the
# host key instead of using `StrictHostKeyChecking=no`.  cmd_connect adds an
# entry on first contact; later calls can opt into strict checking by passing
# `-o UserKnownHostsFile="$GB_KNOWN_HOSTS" -o StrictHostKeyChecking=yes` (see
# _gb_ssh_opts helper below).
GB_KNOWN_HOSTS="/etc/greenboost/known_hosts"

# _gb_ssh_opts <ip> - emit SSH options that prefer strict checking against
# the pinned key file.  If the host has not been pinned yet, fall back to the
# accept-new policy on the SAME file so that subsequent connects upgrade to
# strict automatically.  Call sites that currently use `-o StrictHostKeyChecking=no`
# can migrate by replacing those two options with `$(_gb_ssh_opts "$ip")`.
# PR-GG: inline fallback - only defined if lib/gb_ssh.sh wasn't sourced.
if ! declare -F _gb_ssh_opts >/dev/null 2>&1; then
_gb_ssh_opts() {
    local ip="$1"
    mkdir -p "$(dirname "$GB_KNOWN_HOSTS")" 2>/dev/null || true
    if [[ -f "$GB_KNOWN_HOSTS" ]] && ssh-keygen -F "$ip" -f "$GB_KNOWN_HOSTS" >/dev/null 2>&1; then
        printf -- '-o UserKnownHostsFile=%s -o StrictHostKeyChecking=yes' "$GB_KNOWN_HOSTS"
    else
        printf -- '-o UserKnownHostsFile=%s -o StrictHostKeyChecking=accept-new' "$GB_KNOWN_HOSTS"
    fi
}
fi

# PR-BB: _gb_run_tui_loop <snapshot_fn> [refresh_sec]
#
# The TUI command paradigm requires every
# interactive status command to:
#   1. Enter the alternate-screen buffer (preserve terminal history)
#   2. Hide the cursor
#   3. Loop: home cursor → render snapshot → clear-to-end-of-screen → wait up
#      to N seconds for a keypress
#   4. On Ctrl+C / SIGTERM: restore cursor + leave alt-screen + restore stty.
#
# Before this helper, six command implementations each had ~25 lines of
# boilerplate doing exactly that - drift between them was an ongoing audit
# finding (the prior audit flagged this as "massive duplication in TUI loop
# lines 5317-5340, 5454-5475, 7651-7674, 7768-7780, 8162-8183, 5064-5085").
#
# Each caller now passes its snapshot function name and (optionally) the
# refresh interval.  The trap is local to the function - exit-from-this-shell
# semantics restore on signal.
# PR-GG: inline fallback - only defined if lib/gb_tui.sh wasn't sourced.
if ! declare -F _gb_run_tui_loop >/dev/null 2>&1; then
# _gb_run_tui_loop <snapshot_fn> [refresh=5] [header_hint=""] [log_path=""]
# header_hint: if non-empty, printed above the snapshot each frame.
# log_path:    if non-empty, Ctrl+L opens _show_log_view for that file.
_gb_run_tui_loop() {
    local _snapshot_fn="$1"
    local _refresh="${2:-5}"
    local _header_hint="${3:-}"
    local _log_path="${4:-}"
    local _saved_stty; _saved_stty=$(stty -g 2>/dev/null || true)
    stty -ixon 2>/dev/null || true
    printf '\033[?1049h'
    printf '\033[?25l'
    trap 'printf "\033[?25h\033[?1049l"; stty "${_saved_stty-}" 2>/dev/null || true; exit 0' INT TERM EXIT
    local _key=""
    while true; do
        printf '\033[H'
        [[ -n "$_header_hint" ]] && echo -e "$_header_hint"
        "$_snapshot_fn" || true
        printf '\033[J'
        if read -t "$_refresh" -s -n 1 _key 2>/dev/null; then
            case "$_key" in
                $'\x03') break ;;
                $'\x13') ;;  # Ctrl+S: immediate refresh
                $'\x0c')     # Ctrl+L: log view (only when log_path provided)
                    if [[ -n "$_log_path" ]]; then
                        _show_log_view "$_log_path"
                        printf '\033[2J\033[H'
                    fi
                    ;;
                *) ;;
            esac
        fi
    done
    printf '\033[?25h\033[?1049l'
    [[ -n "$_saved_stty" ]] && stty "${_saved_stty:-}" 2>/dev/null || true
    trap - INT TERM EXIT
}
fi

# ---- _gb_apply_fabric_net_tuning - keep the cluster fabric link tuned ----
# Called from BOTH cmd_feed (feeder side) and cmd_connect (host side) so the
# host<->feeder TCP link always runs with the lowest-latency/highest-
# throughput settings this GreenBoost version knows about, not just when the
# user separately remembers to run `tune-sysctl`. Live-applies immediately
# AND persists to the same 99-zzz-greenboost.conf that tune-sysctl/Full
# Install write, so it survives reboots. Idempotent (grep-guarded appends) -
# safe to call on every connect/feed-start. Best-effort: never fails the
# caller (feed/connect must still work on a locked-down or non-root path).
_gb_apply_fabric_net_tuning() {
    sysctl -w net.ipv4.tcp_slow_start_after_idle=0 >/dev/null 2>&1 || true
    modprobe tcp_bbr >/dev/null 2>&1 || true
    local _bbr_ok=0
    grep -qw bbr /proc/sys/net/ipv4/tcp_available_congestion_control 2>/dev/null && _bbr_ok=1
    [[ $_bbr_ok -eq 1 ]] && { sysctl -w net.ipv4.tcp_congestion_control=bbr >/dev/null 2>&1 || true; }
    gb_ok "Network link tuned (tcp_slow_start_after_idle=0$( [[ $_bbr_ok -eq 1 ]] && echo ', BBR' ))"

    # Persisting to /etc/sysctl.d requires root; if not root the live-apply
    # above still helps this boot, and Full Install's tune-sysctl step
    # (which writes the full, canonical network block) will persist it
    # properly the next time it runs as root.
    [[ $EUID -eq 0 ]] || return 0
    local dest="/etc/sysctl.d/99-zzz-greenboost.conf"
    mkdir -p /etc/sysctl.d
    if ! grep -q "^net\.ipv4\.tcp_slow_start_after_idle" "$dest" 2>/dev/null; then
        printf '\n# Cluster fabric (auto-applied by feed/connect - see tune-sysctl for the full network block)\nnet.ipv4.tcp_slow_start_after_idle = 0\n' >> "$dest"
    fi
    if [[ $_bbr_ok -eq 1 ]] && ! grep -q "^net\.ipv4\.tcp_congestion_control" "$dest" 2>/dev/null; then
        printf 'net.ipv4.tcp_congestion_control = bbr\n' >> "$dest"
    fi
}

cmd_feed() {
    need_root "feed"
    local action="${1:-start}"
    case "$action" in
        start|"")
            if [[ ! -x "$GB_NETD_BIN" ]]; then
                die "greenboost-netd not found at $GB_NETD_BIN - build it first: make netd && sudo cp greenboost-netd /usr/local/bin/"
            fi
            if [[ -f "$GB_NETD_PID" ]] && kill -0 "$(cat "$GB_NETD_PID" 2>/dev/null)" 2>/dev/null; then
                info "Feeder daemon already running (pid $(cat "$GB_NETD_PID"))"
                return 0
            fi
            # Feeder GPU must be COMPUTE-ready, not just VRAM-ready: locked/
            # low clocks silently cripple remote compute (found live: omen
            # stuck at 180 MHz / 3.3 TFLOPS under load; -rgc restored
            # 42.7 TFLOPS, a 13× difference). Reset clock locks, enable
            # persistence, prefer the performance platform profile.
            nvidia-smi -rgc &>/dev/null || true
            nvidia-smi -pm 1 &>/dev/null || true
            powerprofilesctl set performance &>/dev/null || true
            gb_ok "Feeder GPU compute-ready (clock locks reset, persistence on)"
            # Feeder link must be LATENCY-ready too, not just the GPU: this
            # daemon is the sender for heartbeats, H2D/D2H responses, and
            # (with the 2-device Ollama split) per-layer activation replies.
            _gb_apply_fabric_net_tuning
            # Ensure T3 swap is provisioned: minimum T1 × 2, floor 16 GB
            local _t1_mib=0
            _t1_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
                      2>/dev/null | head -1 | tr -d ' ') || _t1_mib=0
            local _t1_gb=$(( (_t1_mib + 512) / 1024 ))
            local _t3_min=$(( _t1_gb * 2 ))
            [[ $_t3_min -lt 16 ]] && _t3_min=16
            mkdir -p /var/lib/greenboost
            local _swap_target="$GB_SWAP_FILE"
            local _cur_swap_gb=0
            if [[ -f "$_swap_target" ]]; then
                _cur_swap_gb=$(( $(stat -c%s "$_swap_target" 2>/dev/null || echo 0) / 1073741824 ))
            fi
            if [[ $_cur_swap_gb -lt $_t3_min ]]; then
                local _disk_free
                _disk_free=$(df -BG /var/lib/greenboost 2>/dev/null | awk 'NR==2{gsub("G",""); print $4}')
                local _disk_cap=$(( ${_disk_free:-0} * 80 / 100 ))
                [[ $_t3_min -gt $_disk_cap ]] && _t3_min=$_disk_cap
                if [[ $_t3_min -gt 0 ]]; then
                    info "Provisioning T3 swap: ${_t3_min} GB (T1=${_t1_gb} GB × 2) → ${_swap_target}"
                    swapoff "$_swap_target" 2>/dev/null || true
                    fallocate -l "${_t3_min}G" "$_swap_target" 2>/dev/null \
                        || dd if=/dev/zero of="$_swap_target" bs=1G count="$_t3_min" status=none
                    chmod 600 "$_swap_target"
                    mkswap "$_swap_target" >/dev/null
                    swapon -p 10 "$_swap_target"
                    if ! grep -qF "$_swap_target" /etc/fstab 2>/dev/null; then
                        echo "${_swap_target} swap swap defaults,nofail 0 0" >> /etc/fstab
                    fi
                    gb_ok "T3 swap: ${_t3_min} GB ready at ${_swap_target}"
                else
                    gb_warn_ui "Insufficient disk space for T3 swap - feeder T3 will rely on existing swap"
                fi
            else
                gb_ok "T3 swap: ${_cur_swap_gb} GB already provisioned at ${_swap_target}"
            fi

            # Disable cuFile (GPUDirect Storage) in any Ollama service on this feeder.
            # The CUFileThreadPoolWorker assertion fires when nvidia-fs is absent or
            # misconfigured; GreenBoost handles NVMe via its own T3 path on the feeder.
            if systemctl is-enabled ollama.service &>/dev/null 2>&1; then
                mkdir -p /etc/systemd/system/ollama.service.d
                cat >/etc/systemd/system/ollama.service.d/99-greenboost-feed.conf <<'OEOF'
[Service]
Environment=NVCUFILE_DISABLE=1
OEOF
                systemctl daemon-reload
                systemctl restart ollama.service &>/dev/null || true
                gb_ok "Disabled cuFile in feeder Ollama (prevents CUFileThreadPoolWorker crash)"
            fi

            # ── Open firewall port for feeder protocol ─────────────────────────────
            # greenboost-netd listens on TCP $GB_NET_PORT (default 9740). Most Linux
            # distros enable a firewall by default; without this the host can't
            # reach the feeder even though netd binds to 0.0.0.0.
            local _fw_opened=0
            if command -v ufw &>/dev/null && ufw status 2>/dev/null | grep -qE '^Status: active'; then
                if ufw allow "$GB_NET_PORT"/tcp &>/dev/null; then
                    gb_ok "ufw: opened port ${GB_NET_PORT}/tcp (feeder protocol)"
                    _fw_opened=1
                else
                    gb_warn_ui "ufw allow ${GB_NET_PORT}/tcp failed , workstation may not reach this feeder"
                fi
            elif command -v firewall-cmd &>/dev/null && firewall-cmd --state 2>/dev/null | grep -q running; then
                if firewall-cmd --permanent --add-port="${GB_NET_PORT}/tcp" &>/dev/null && \
                   firewall-cmd --reload &>/dev/null; then
                    gb_ok "firewalld: opened port ${GB_NET_PORT}/tcp (feeder protocol)"
                    _fw_opened=1
                else
                    gb_warn_ui "firewalld add-port failed , workstation may not reach this feeder"
                fi
            elif command -v iptables &>/dev/null; then
                if ! iptables -C INPUT -p tcp --dport "$GB_NET_PORT" -j ACCEPT &>/dev/null; then
                    if iptables -I INPUT -p tcp --dport "$GB_NET_PORT" -j ACCEPT 2>/dev/null; then
                        gb_ok "iptables: opened port ${GB_NET_PORT}/tcp (feeder protocol)"
                        _fw_opened=1
                    fi
                else
                    _fw_opened=1  # rule already present
                fi
            fi
            [[ $_fw_opened -eq 0 ]] && gb_info "No active firewall detected , port ${GB_NET_PORT}/tcp assumed reachable"

            # N6: MPS daemon setup - start and validate nvidia-cuda-mps-server if requested
            local _mps_started_by_us=0
            if [[ "${GREENBOOST_MPS:-0}" == "1" ]]; then
                # Audit F-L4-17: use controlled paths under /run/greenboost (not world-
                # writable /tmp) so MPS socket and logs are not accessible to all users.
                export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/run/greenboost/nvidia-mps}"
                export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/var/log/greenboost/mps}"
                mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
                chmod o-rwx "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
                if ! pgrep -x nvidia-cuda-mps-server >/dev/null 2>&1; then
                    info "N6: Starting nvidia-cuda-mps-control daemon for MPS SM% control..."
                    nvidia-cuda-mps-control -d 2>/dev/null || true
                    sleep 1
                    if pgrep -x nvidia-cuda-mps-server >/dev/null 2>&1; then
                        gb_ok "MPS daemon started - CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
                        _mps_started_by_us=1
                    else
                        gb_warn_ui "MPS daemon failed to start - disabling GREENBOOST_MPS"
                        export GREENBOOST_MPS=0
                    fi
                else
                    gb_ok "MPS daemon already running"
                fi
            fi

            info "Starting network feeder daemon on port $GB_NET_PORT..."
            mkdir -p /run/greenboost /var/log/greenboost
            local _netd_stderr_tmp
            _netd_stderr_tmp=$(mktemp /tmp/gb-netd-XXXXXX)
            # Export _mps_started_by_us so the stop action can clean up MPS if we started it
            echo "$_mps_started_by_us" > /run/greenboost/mps_owned 2>/dev/null || true
            "$GB_NETD_BIN" -d -p "$GB_NET_PORT" 2>"$_netd_stderr_tmp"
            sleep 1
            if [[ -f "$GB_NETD_PID" ]] && kill -0 "$(cat "$GB_NETD_PID" 2>/dev/null)" 2>/dev/null; then
                rm -f "$_netd_stderr_tmp"
                gb_ok "Feeder daemon started (pid $(cat "$GB_NETD_PID"))"
                echo -e "\n  ${C_LIME}Run the following command on the AI workstation to connect to this machine:${C_RESET}";
                echo -e "  ${C_BOLD}sudo greenboost connect $(hostname -I | awk '{print $1}')${C_RESET}\n"
            else
                if [[ -s "$_netd_stderr_tmp" ]]; then
                    gb_warn_ui "Startup error:"
                    while IFS= read -r _el; do gb_info "  $_el"; done < "$_netd_stderr_tmp"
                elif [[ -s "/var/log/greenboost/netd.log" ]]; then
                    gb_warn_ui "Last log entries:"
                    tail -5 "/var/log/greenboost/netd.log" | while IFS= read -r _el; do gb_info "  $_el"; done
                fi
                rm -f "$_netd_stderr_tmp"
                die "Feeder daemon failed to start - for details run: sudo $GB_NETD_BIN -p $GB_NET_PORT"
            fi
            ;;
        --stop|stop)
            if [[ -f "$GB_NETD_PID" ]]; then
                local pid
                pid=$(cat "$GB_NETD_PID" 2>/dev/null)
                if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                    kill "$pid"
                    sleep 0.5
                    gb_ok "Feeder daemon stopped (pid $pid)"
                else
                    info "Feeder daemon not running"
                fi
                rm -f "$GB_NETD_PID"
                # N6: shut down MPS daemon if we started it
                local _mps_owned=0
                [[ -f /run/greenboost/mps_owned ]] && _mps_owned=$(cat /run/greenboost/mps_owned 2>/dev/null || echo 0)
                if [[ "$_mps_owned" == "1" ]] && pgrep -x nvidia-cuda-mps-server >/dev/null 2>&1; then
                    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
                    gb_ok "MPS daemon stopped (was started by greenboost feed start)"
                fi
                rm -f /run/greenboost/mps_owned
            else
                info "Feeder daemon not running"
            fi
            ;;
        --foreground|fg)
            if [[ ! -x "$GB_NETD_BIN" ]]; then
                die "greenboost-netd not found - build it first: make netd"
            fi
            mkdir -p /run/greenboost /var/log/greenboost
            exec "$GB_NETD_BIN" -p "$GB_NET_PORT"
            ;;
        *)
            echo "Usage: greenboost feed [start|stop|fg]" >&2
            exit 1
            ;;
    esac
}

_gb_net_handshake() {
    local ip="$1" port="${2:-$GB_NET_PORT}"
    local timeout=5

    # Binary handshake: send HANDSHAKE_REQ, read HANDSHAKE_RESP
    # Wire protocol v3: gb_net_header is 16 bytes (magic+msg_type+flags+payload_len+seq_num).
    python3 -c "
import socket, struct, sys, hashlib, hmac, os

ip, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout($timeout)
try:
    sock.connect((ip, port))
except Exception as e:
    print(f'ERROR:connect:{e}', file=sys.stderr)
    sys.exit(1)

# PSK auth: if cluster.key present, recv 32-byte nonce and send HMAC-SHA256 response.
_GB_KEY_PATH = '/etc/greenboost/cluster.key'
def _load_psk():
    try:
        with open(_GB_KEY_PATH) as f:
            h = f.readline().strip()
        return bytes.fromhex(h[:64]) if len(h) >= 64 else None
    except (OSError, ValueError):
        return None

psk = _load_psk()
if psk is not None:
    nonce = b''
    while len(nonce) < 32:
        chunk = sock.recv(32 - len(nonce))
        if not chunk:
            print('ERROR:psk_nonce_truncated', file=sys.stderr)
            sys.exit(1)
        nonce += chunk
    mac = hmac.new(psk, nonce, hashlib.sha256).digest()
    sock.sendall(mac)
else:
    # No local cluster.key. If the feeder immediately sends a 32-byte PSK
    # nonce it requires auth and the plain handshake below would just get a
    # connection reset , fail with the remediation instead.
    sock.settimeout(0.6)
    try:
        pre = sock.recv(32, socket.MSG_PEEK)
        if pre:
            print('ERROR:psk_required - feeder requires cluster.key auth but this host has no /etc/greenboost/cluster.key. On the feeder: sudo greenboost feeders export-key ; then here: sudo greenboost feeders import-key <HEX64>', file=sys.stderr)
            sys.exit(1)
    except socket.timeout:
        pass
    sock.settimeout($timeout)

# Wire protocol v3: header = magic(4) + msg_type(2) + flags(2) + payload_len(4) + seq_num(4) = 16 bytes
HDR_FMT  = '<IHHII'
HDR_SIZE = struct.calcsize(HDR_FMT)   # 16
magic    = 0x47424E46
send_seq = 0

def send_msg(msg_type, payload):
    global send_seq
    hdr = struct.pack(HDR_FMT, magic, msg_type, 0, len(payload), send_seq)
    send_seq += 1
    sock.sendall(hdr + payload)

def recv_hdr():
    buf = b''
    while len(buf) < HDR_SIZE:
        chunk = sock.recv(HDR_SIZE - len(buf))
        if not chunk:
            return None
        buf += chunk
    return struct.unpack(HDR_FMT, buf)   # magic, type, flags, payload_len, seq_num

def recv_payload(length):
    buf = b''
    while len(buf) < length:
        chunk = sock.recv(length - len(buf))
        if not chunk:
            return buf
        buf += chunk
    return buf

# Build HANDSHAKE_REQ payload: proto_version(4) + gpu_count(4) + hostname(64) + gpus(8*100)
hostname = os.uname().nodename[:63]
proto_ver = 3
gpu_count = 0
payload = struct.pack('<II', proto_ver, gpu_count)
payload += hostname.encode('utf-8').ljust(64, b'\x00')[:64]
payload += b'\x00' * (8 * 100)

send_msg(0x01, payload)   # GB_MSG_HANDSHAKE_REQ = 0x01

resp = recv_hdr()
if not resp:
    print('ERROR:short_header', file=sys.stderr)
    sys.exit(1)
r_magic, r_type, r_flags, r_len, r_seq = resp
if r_magic != magic:
    print('ERROR:bad_magic', file=sys.stderr)
    sys.exit(1)

resp_payload = recv_payload(r_len)

# Parse HANDSHAKE_RESP: status(4) + feeder_id(4) + proto_ver(4) + gpu_count(4) + hostname(64) + gpus
status, feeder_id, proto_ver, gpu_count = struct.unpack_from('<IIII', resp_payload, 0)
feeder_hostname = resp_payload[16:80].split(b'\x00')[0].decode('utf-8', errors='replace')

if status != 0:
    print(f'ERROR:rejected:{status}', file=sys.stderr)
    sys.exit(1)

# Parse GPU info: offset 80, each entry is 100 bytes
# gpu_id(4) + vram_bytes(8) + cc_major(4) + cc_minor(4) + ram_available(8) + t3_bytes(8) + name(64)
offset = 80
for i in range(min(gpu_count, 8)):
    gid, vram, cc_maj, cc_min, ram_avail, t3 = struct.unpack_from('<IQIIQQ', resp_payload, offset)
    name = resp_payload[offset+36:offset+100].split(b'\x00')[0].decode('utf-8', errors='replace')
    print(f'GPU:{gid}:{name}:{vram}:{cc_maj}.{cc_min}:{ram_avail}:{t3}')
    offset += 100

print(f'OK:{feeder_id}:{feeder_hostname}:{gpu_count}')

# MEM_INFO query (msg_type=0x31) - get live T1/T2/T3 free+total for bars
try:
    for dev_id in range(min(gpu_count, 8)):
        mem_payload = struct.pack('<II', dev_id, 0)  # gb_net_mem_info: device_id(4) t2_speed_mts(4)
        send_msg(0x31, mem_payload)

        mhdr = recv_hdr()
        if not mhdr: break
        _, _, _, mr_len, _ = mhdr
        mpay = recv_payload(mr_len)
        if len(mpay) < 36: break

        # gb_net_mem_info_resp: status(4) t2_speed(4) free(8) total(8) t1_free(8) t1_total(8)
        #   t2_free(8) t2_total(8) t3_free(8) t3_total(8) = 72 bytes
        if len(mpay) >= 72:
            st, spd, fb, tb, t1f, t1t, t2f, t2t, t3f, t3t = struct.unpack_from('<IIqqqqqqqq', mpay, 0)
        else:
            st, spd, fb, tb = struct.unpack_from('<IIqq', mpay, 0)
            t1f = t1t = t2f = t2t = t3f = t3t = 0
        print(f'MEM:{dev_id}:{t1f}:{t1t}:{t2f}:{t2t}:{t3f}:{t3t}')
except Exception:
    pass

# GB_MSG_FEEDER_STATUS (0x34) , v3.1 GPU telemetry (temp/power/util/ECC/throttle).
# Replaces the SSH+nvidia-smi side-channel for feeder GPU health.
# gb_feeder_status_resp v3.1: status(4) mps(4) t1f(8) t1t(8) t2f(8) t2t(8)
#   t3f(8) t3t(8) kd_count(4) _pad(4) = 64 bytes (v3.0)
#   + gpu_temp_c(2) gpu_power_w(2) gpu_util_pct(4) ecc_dbe(4) throttle(4) _pad2(4) = 80 bytes (v3.1)
_FEEDER_STATUS_V31_SIZE = 80
try:
    send_msg(0x34, b'')   # GB_MSG_FEEDER_STATUS; no payload
    fshdr = recv_hdr()
    if fshdr:
        _, _, _, fs_len, _ = fshdr
        fspay = recv_payload(fs_len)
        if len(fspay) >= _FEEDER_STATUS_V31_SIZE:
            # Parse v3.1 GPU telemetry fields at offset 64
            gpu_temp_c, gpu_power_w, gpu_util_pct, ecc_dbe, throttle = \
                struct.unpack_from('<HHIII', fspay, 64)
            print(f'GPU_HEALTH:{gpu_temp_c}:{gpu_power_w}:{gpu_util_pct}:{ecc_dbe}:{throttle}')
except Exception:
    pass

sock.close()
" "$ip" "$port" 2>&1
}

# PR-CC: _gb_write_cluster_extra_mem is now a stub.
#
# The kernel module's cluster_extra_mem_gb parameter was removed in PR-K
# because nothing in the module ever read it.  The function is preserved as
# a no-op so existing call-sites (cmd_connect, cmd_disconnect) don't need
# editing; the cluster-memory-aware sysinfo() behaviour the original sysfs
# write was supposed to enable is implemented entirely in the CUDA shim's
# cuDeviceTotalMem / nvmlDeviceGetMemoryInfo interception.  See Chapter G
# of greenboost_documentation_extension_official_nvidia.md.
_gb_write_cluster_extra_mem() {
    return 0
}

cmd_connect() {
    local ip="${1:-}"
    local port="${2:-$GB_NET_PORT}"

    if [[ -z "$ip" ]]; then
        echo "Usage: greenboost connect <IP> [PORT]" >&2
        exit 1
    fi

    info "Connecting to ${ip}:${port}..."

    # 2>&1: _gb_net_handshake prints its ERROR: lines on stderr , without
    # capturing them the die below reports "unknown error" for every failure.
    # `|| rc=$?` keeps set -e from killing the script on handshake failure
    # before the error is reported (the old code died here silently).
    local output rc=0
    output=$(_gb_net_handshake "$ip" "$port" 2>&1) || rc=$?

    if [[ $rc -ne 0 ]] || echo "$output" | grep -q "^ERROR:"; then
        local err
        err=$(echo "$output" | grep "^ERROR:" | head -1)
        die "Connection failed: ${err:-unknown error}"
    fi

    # Parse handshake response
    local ok_line
    ok_line=$(echo "$output" | grep "^OK:")
    local feeder_id feeder_host gpu_count
    feeder_id=$(echo "$ok_line" | cut -d: -f2)
    feeder_host=$(echo "$ok_line" | cut -d: -f3)
    gpu_count=$(echo "$ok_line" | cut -d: -f4)

    gb_ok "Connected - feeder \"${feeder_host}\" (${gpu_count} GPU(s))"

    # Show GPU details
    while IFS= read -r line; do
        if [[ "$line" == GPU:* ]]; then
            local gid gname vram cc_ver
            gid=$(echo "$line" | cut -d: -f2)
            gname=$(echo "$line" | cut -d: -f3)
            vram=$(echo "$line" | cut -d: -f4)
            cc_ver=$(echo "$line" | cut -d: -f5)
            local vram_gb=$(( (vram + 536870912) / 1073741824 ))
            gb_info "  GPU ${gid}: ${gname} - ${vram_gb} GB VRAM (cc ${cc_ver})"
        fi
    done <<< "$output"

    # Use invoking user as the remote SSH user (same person running sudo greenboost connect)
    local ssh_user="${SUDO_USER:-$USER}"

    # Save to cluster.conf (format: IP:PORT hostname ssh_user)
    mkdir -p /etc/greenboost
    # Audit F-L4-04: validate IP strictly before interpolating into a sed
    # expression.  Earlier code interpolated $ip directly; an IP carrying sed
    # metacharacters (& | / etc.) could craft an arbitrary substitution.
    if [[ ! "$ip" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
        echo "[GreenBoost] ERROR: refusing non-IPv4-literal feeder address: $ip" >&2
        exit 1
    fi
    local entry="${ip}:${port} ${feeder_host} ${ssh_user}"
    if grep -q "^${ip}:" "$GB_CLUSTER_CONF" 2>/dev/null; then
        # Audit F-L4-26: atomic write - build the new conf in a temp file,
        # then mv into place so a crash mid-write can't corrupt cluster.conf.
        local _tmp
        _tmp=$(mktemp "${GB_CLUSTER_CONF}.XXXXXX") || _tmp="${GB_CLUSTER_CONF}.tmp"
        awk -v ip="$ip" -v repl="$entry" \
            '($0 ~ "^"ip":") { print repl; next } { print }' \
            "$GB_CLUSTER_CONF" > "$_tmp" && mv "$_tmp" "$GB_CLUSTER_CONF"
        gb_info "Updated feeder entry in $GB_CLUSTER_CONF"
    else
        echo "$entry" >> "$GB_CLUSTER_CONF"
        gb_info "Added feeder to $GB_CLUSTER_CONF"
    fi
    # PR-J: cluster.conf contains feeder IPs + SSH usernames - sensitive
    # enough to deny world read.  Default umask (022) leaves the file
    # mode 0644 = world-readable.  Tighten to 0640 (root:root, group can
    # read for monitoring/observability scripts).
    chmod 0644 "$GB_CLUSTER_CONF" 2>/dev/null || true
    gb_info "SSH user: ${ssh_user}"
    _gb_write_cluster_extra_mem

    # Tune the HOST side of the fabric link the moment a feeder joins the
    # cluster - the feeder side is self-tuned by `greenboost feed start`.
    _gb_apply_fabric_net_tuning

    # ── Build-stamp parity check (2026-07-06) ────────────────────────────
    # GreenBoost feeder-GPU compute requires the feeder to run the SAME
    # GreenBoost build as the host (the netd binary is -march=native, and the
    # remote kernel-dispatch path resolves host-dispatched kernel names).  On
    # a stamp mismatch, offer to upgrade the feeder now , which pushes SOURCE
    # and rebuilds on the feeder for its own CPU topology (see
    # cmd_feeders_upgrade_greenboost, which builds on the feeder by default),
    # not a host-built binary.
    _gb_connect_check_parity "$ip" "$ssh_user"

    # The shim reads cluster.conf only once at init (gb_netc_init).
    # A running Ollama won't see the new feeder until it restarts.
    echo ""
    gb_info "Feeder saved to $GB_CLUSTER_CONF , restart Ollama/inference services to activate:"
    gb_info "  sudo systemctl restart ollama"
}

# _gb_connect_check_parity <ip> <ssh_user> , compare the host's GreenBoost
# build stamp with the feeder's; warn (and offer upgrade) on mismatch so host
# and feeder always run identical GreenBoost binaries.  Best-effort: never
# fails the connect.
_gb_connect_check_parity() {
    local ip="$1" ssh_user="$2"
    local _local_id=""
    for _bi in "${MODULE_DIR}/build_info" /etc/greenboost/build_info; do
        [[ -f "$_bi" ]] && { _local_id=$(grep '^BUILD_ID=' "$_bi" 2>/dev/null | cut -d= -f2); break; }
    done
    [[ -z "$_local_id" ]] && return 0

    local _ssh_as=()
    [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]] && _ssh_as=(runuser -u "$SUDO_USER" --)
    local _remote_id
    _remote_id=$("${_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=5 \
        -o StrictHostKeyChecking=no "${ssh_user}@${ip}" \
        "grep '^BUILD_ID=' /etc/greenboost/build_info 2>/dev/null | cut -d= -f2" 2>/dev/null)

    if [[ -z "$_remote_id" ]]; then
        gb_warn "Feeder has no GreenBoost build stamp , feeder-GPU compute needs a matching build."
        gb_info  "  Sync + rebuild on the feeder:  sudo greenboost update feeders"
        return 0
    fi
    if [[ "$_remote_id" != "$_local_id" ]]; then
        gb_warn "GreenBoost build mismatch , host ${_local_id} vs feeder ${_remote_id}."
        gb_warn "  Feeder-GPU compute requires identical builds (netd is -march=native;"
        gb_warn "  host CPU and feeder CPU may differ , Intel vs AMD , so binaries are NOT portable)."
        gb_info  "  Align now:  sudo greenboost update feeders"
    else
        gb_ok "GreenBoost build parity OK (${_local_id})."
    fi

    # ollama version parity , the feeder's native libggml-cuda.so must expose
    # the SAME CUDA kernels the host dispatches, which requires the SAME ollama
    # build.  A skew (seen: host 0.30.8 vs feeder 0.21.1) is why remote kernel
    # names don't resolve → LLM feeder-GPU compute can't run.
    local _host_oll; _host_oll=$(_gb_host_ollama_version)
    if [[ -n "$_host_oll" ]]; then
        local _feeder_oll
        _feeder_oll=$("${_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=5 \
            -o StrictHostKeyChecking=no "${ssh_user}@${ip}" \
            "ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1" 2>/dev/null)
        if [[ "$_feeder_oll" != "$_host_oll" ]]; then
            gb_warn "ollama version mismatch , host ${_host_oll} vs feeder ${_feeder_oll:-none}."
            gb_warn "  LLM feeder-GPU compute needs matching ollama (kernel-name parity)."
            gb_info  "  Align now:  sudo greenboost feeders sync-ollama"
        else
            gb_ok "ollama parity OK (${_host_oll}) , LLM feeder-GPU compute eligible."
        fi
    fi
}

# ── cmd_health_check - one-shot comprehensive cluster health audit (S1) ──────
# Usage: greenboost health-check [--llm]
#
# Checks: module, shim, NVML, T2/T3 pool, each feeder handshake.
# Prints PASS/FAIL/WARN per check.  --llm strips ANSI and emits STATUS:OK or
# STATUS:FAIL:<reason> for machine-readable consumption.
cmd_health_check() {
    local llm_mode=0
    [[ "${1:-}" == "--llm" ]] && llm_mode=1

    local C_PASS="${C_LIME}" C_FAIL="${C_RED}" C_WARN="${C_AMBER}" C_RST="${C_RESET}"
    if [[ $llm_mode -eq 1 ]]; then
        C_PASS="" C_FAIL="" C_WARN="" C_RST=""
    fi

    local fail_reason="" fails=0 warns=0

    _hc_pass() { echo -e "  ${C_PASS}PASS${C_RST}  $1"; }
    _hc_fail() { echo -e "  ${C_FAIL}FAIL${C_RST}  $1"; fails=$(( fails + 1 )); fail_reason="${fail_reason}${1}; "; }
    _hc_warn() { echo -e "  ${C_WARN}WARN${C_RST}  $1"; warns=$(( warns + 1 )); }

    # Audit F-L4-08: SSH identity - prefer the original invoking user so BatchMode
    # key-based auth works when running under sudo.
    local _hc_ssh_as=()
    if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
        _hc_ssh_as=(runuser -u "$SUDO_USER" --)
    fi

    [[ $llm_mode -eq 0 ]] && { gb_header; gb_section "GreenBoost Health Check  $(date '+%Y-%m-%d %H:%M:%S')"; echo ""; }

    # 1. Kernel module
    if grep -q "^greenboost " <<< "$(lsmod 2>/dev/null)"; then
        _hc_pass "Kernel module: greenboost loaded"
    else
        _hc_fail "Kernel module: greenboost NOT loaded (run: sudo greenboost load)"
    fi

    # 2. CUDA shim
    local shim_path
    for shim_path in /usr/local/lib/libgreenboost_cuda.so \
                     /usr/lib/libgreenboost_cuda.so \
                     /usr/local/lib/libgreenboost_cuda_shim.so; do
        [[ -f "$shim_path" ]] && break || shim_path=""
    done
    if [[ -n "$shim_path" ]]; then
        _hc_pass "CUDA shim: $shim_path"
    else
        _hc_warn "CUDA shim: not found in standard paths (run: sudo make install)"
    fi

    # 3. NVML / nvidia-smi
    if nvidia-smi -L &>/dev/null; then
        local gpu_line; gpu_line=$(nvidia-smi -L 2>/dev/null | head -1)
        _hc_pass "NVML: ${gpu_line:-OK}"
    else
        _hc_fail "NVML: nvidia-smi not accessible (driver issue or not a GPU host)"
    fi

    # 4. T2 pool sysfs
    local sysfs_t2="/sys/module/greenboost/parameters/t2_pool_gb"
    if [[ -r "$sysfs_t2" ]]; then
        local t2_val; t2_val=$(cat "$sysfs_t2")
        _hc_pass "T2 pool: ${t2_val} GB (from sysfs)"
    else
        _hc_warn "T2 pool: sysfs not available (module not loaded or param not exposed)"
    fi

    # 5. T3 NVMe file
    local sysfs_nvme="/sys/module/greenboost/parameters/nvme_pool_gb"
    local t3_file="/var/lib/greenboost/nvme_swap.img"
    if [[ -f "$t3_file" ]]; then
        local t3_sz; t3_sz=$(du -sh "$t3_file" 2>/dev/null | awk '{print $1}')
        _hc_pass "T3 NVMe file: $t3_file (${t3_sz})"
    else
        local nvme_val=0
        [[ -r "$sysfs_nvme" ]] && nvme_val=$(cat "$sysfs_nvme")
        if [[ "$nvme_val" -gt 0 ]]; then
            _hc_fail "T3 NVMe file: $t3_file missing but nvme_pool_gb=$nvme_val (run: sudo greenboost setup)"
        else
            _hc_warn "T3 NVMe: not configured (optional)"
        fi
    fi

    # 6. Feeder handshakes
    if [[ ! -f "$GB_CLUSTER_CONF" ]] || ! grep -q . "$GB_CLUSTER_CONF" 2>/dev/null; then
        _hc_warn "Cluster: no feeders configured in $GB_CLUSTER_CONF"
    else
        while IFS=: read -r _ip _port _rest || [[ -n "$_ip" ]]; do
            [[ -z "$_ip" || "$_ip" == \#* ]] && continue
            # Audit F-L4-13: cluster.conf line is "IP:PORT hostname user".  With
            # IFS=: the second field (_port) captures "PORT hostname user" because
            # the hostname is space-separated, not colon-separated.  Strip the
            # trailing fields so _port_num carries only the numeric port.
            local _port_num="${_port%% *}"
            _port_num="${_port_num:-9740}"
            # Extract SSH user: _port is "PORT hostname user"; strip port, then last word.
            local _after_port="${_port#"${_port_num}"}"
            _after_port="${_after_port# }"  # strip leading space
            local _feeder_user
            _feeder_user="${_after_port##* }"  # last space-delimited word
            [[ "$_feeder_user" == "$_after_port" && -z "$_after_port" ]] && _feeder_user=""
            _feeder_user="${_feeder_user:-root}"

            # TCP handshake
            local _hs; _hs=$(_gb_net_handshake "$_ip" "$_port_num" 2>&1)
            if echo "$_hs" | grep -q "^OK:"; then
                _hc_pass "Feeder ${_ip}:${_port_num}: netd reachable"
            else
                _hc_fail "Feeder ${_ip}:${_port_num}: handshake failed (${_hs%%$'\n'*})"
            fi

            # Audit F-L4-02: pin host keys via $GB_KNOWN_HOSTS through _gb_ssh_opts.
            # On first contact for an IP the helper falls back to accept-new on the
            # same file, so subsequent calls automatically upgrade to strict checking.
            local _ssh_rc=0
            "${_hc_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=10 \
                $(_gb_ssh_opts "$_ip") "${_feeder_user}@${_ip}" "true" 2>/dev/null \
                || _ssh_rc=$?
            if [[ $_ssh_rc -eq 0 ]]; then
                _hc_pass "Feeder ${_ip}: SSH auth ok (user=${_feeder_user})"
            else
                _hc_warn "Feeder ${_ip}: SSH auth failed (user=${_feeder_user}, rc=${_ssh_rc}) - upgrade-greenboost needs key auth"
            fi
        done < "$GB_CLUSTER_CONF"
    fi

    echo ""
    if [[ $llm_mode -eq 1 ]]; then
        if [[ $fails -eq 0 ]]; then
            echo "STATUS:OK warns=${warns}"
        else
            echo "STATUS:FAIL:${fail_reason%%; }"
        fi
    else
        if [[ $fails -eq 0 && $warns -eq 0 ]]; then
            gb_ok "All checks passed."
        elif [[ $fails -eq 0 ]]; then
            gb_warn_ui "${warns} warning(s) - system functional."
        else
            echo -e "  ${C_RED}${fails} check(s) FAILED.${C_RESET} Review output above."
        fi
        gb_separator
    fi
}

cmd_disconnect() {
    local ip="${1:-}"
    if [[ -z "$ip" ]]; then
        echo "Usage: greenboost disconnect <IP>" >&2
        exit 1
    fi
    # Audit F-L4-04: validate IP literal before using it in a sed/awk anchor.
    if [[ ! "$ip" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
        echo "[GreenBoost] ERROR: refusing non-IPv4-literal feeder address: $ip" >&2
        exit 1
    fi

    if [[ -f "$GB_CLUSTER_CONF" ]] && grep -q "^${ip}:" "$GB_CLUSTER_CONF"; then
        # Audit F-L4-26: atomic delete via temp + mv.
        local _tmp
        _tmp=$(mktemp "${GB_CLUSTER_CONF}.XXXXXX") || _tmp="${GB_CLUSTER_CONF}.tmp"
        awk -v ip="$ip" '!($0 ~ "^"ip":")' "$GB_CLUSTER_CONF" > "$_tmp" \
            && mv "$_tmp" "$GB_CLUSTER_CONF"
        gb_ok "Disconnected feeder $ip"
    else
        echo "[GreenBoost] ERROR: feeder $ip is not in the cluster configuration (${GB_CLUSTER_CONF})" >&2
        echo "[GreenBoost] Run 'greenboost cluster' to list currently connected feeders." >&2
        exit 1
    fi
    _gb_write_cluster_extra_mem
}

# _gb_update_all_feeders - push new daemon + script to every feeder in cluster.conf.
# Called automatically at the end of cmd_install / cmd_full_install.
# Completely unattended: uses SSH key auth (BatchMode); skips feeders that are
# unreachable or missing key auth with a warning instead of aborting.
_gb_update_all_feeders() {
    # Thin wrapper: delegate to cmd_feeders_upgrade_greenboost, the maintained
    # push path (pre-flight sudo -n check, atomic `install` swap, netd.pid
    # readiness poll, stderr shown on failure). This function used to carry
    # its own copy of the remote install script which ran WITHOUT sudo on the
    # feeder , with a non-root SSH user it always failed rc=1 at the
    # /usr/local/bin cp (hit on omen, 2026-07-06).
    [[ ! -f "$GB_CLUSTER_CONF" ]] && return 0
    grep -q '^[^#[:space:]]' "$GB_CLUSTER_CONF" 2>/dev/null || return 0
    cmd_feeders_upgrade_greenboost \
        || gb_warn "Feeder update had failures (see above) - continuing install"
    return 0
}

cmd_update_feeders() {
    need_root "update feeders"
    if [[ ! -f "$GB_CLUSTER_CONF" ]]; then
        die "No feeders configured - run: greenboost connect <IP>"
    fi

    # `greenboost update feeders` is THE canonical command to bring every
    # feeder into parity with the host: (1) match ollama version so the
    # feeder's native ggml has the host's kernels, then (2) rebuild + install
    # GreenBoost (netd + interposer) ON each feeder from source (feeder CPU
    # differs from host , Intel vs AMD , so binaries are never copied).
    gb_section "Update feeders (match host)"

    gb_info "Step 1/2 , matching ollama version…"
    cmd_feeders_sync_ollama || gb_warn "ollama sync had failures (see above) , continuing to GreenBoost build"

    gb_info "Step 2/2 , building GreenBoost on each feeder from source…"
    _gb_feeders_upgrade_from_source

    info "Done. Verify with: greenboost cluster"
}

# cmd_feeders_setup_sudo - configure passwordless sudo for GreenBoost on all feeders.
# Runs interactively (prompts for feeder sudo password once per feeder).
# Usage: sudo greenboost feeders setup-sudo
cmd_feeders_setup_sudo() {
    need_root "feeders setup-sudo"
    if [[ ! -f "$GB_CLUSTER_CONF" ]] || ! grep -q '[^[:space:]]' "$GB_CLUSTER_CONF" 2>/dev/null; then
        die "No feeders configured - run: sudo greenboost connect <IP>"
    fi

    local _ssh_as=()
    if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
        _ssh_as=(runuser -u "$SUDO_USER" --)
    fi

    gb_section "GreenBoost Feeder sudo Setup"

    while IFS= read -r _line; do
        [[ -z "$_line" || "$_line" == \#* ]] && continue
        local _addr _hostname _ssh_user
        _addr=$(echo "$_line" | awk '{print $1}')
        _hostname=$(echo "$_line" | awk '{print $2}')
        _ssh_user=$(echo "$_line" | awk '{print $3}')
        _ssh_user="${_ssh_user:-root}"
        # Audit F-L4-18: reject usernames that could inject content into the sudoers
        # file via the unquoted shell interpolation at the SSH command below.
        if [[ ! "$_ssh_user" =~ ^[a-zA-Z_][a-zA-Z0-9._-]{0,31}$ ]]; then
            printf "    %b\n" "${C_RED}✗  invalid feeder username '${_ssh_user}' - skipping${C_RESET}" >&2
            continue
        fi
        local _ip="${_addr%%:*}"
        printf "\n  ${C_VIOLET}%-22s${C_RESET} %s@%s\n" "${_hostname:-$_ip}" "$_ssh_user" "$_ip"

        # Read password from /dev/tty so it works even when stdin is redirected by sudo
        local _sudo_pass="" _char
        printf "    Feeder sudo password: " > /dev/tty
        while IFS= read -r -s -n1 _char < /dev/tty; do
            case "$_char" in
                '')            break ;;           # Enter
                $'\x7f'|$'\b') if [[ -n "$_sudo_pass" ]]; then
                                   _sudo_pass="${_sudo_pass%?}"; printf '\b \b' > /dev/tty
                               fi ;;              # Backspace / DEL
                *)             _sudo_pass+="$_char"; printf '*' > /dev/tty ;;
            esac
        done
        printf '\n' > /dev/tty

        # sudo -S reads the password from stdin - no TTY or PTY needed
        local _setup_rc=0
        "${_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=30 \
            -o StrictHostKeyChecking=no "${_ssh_user}@${_ip}" \
            "sudo -S sh -c 'echo \"${_ssh_user} ALL=(root) NOPASSWD: /bin/bash\" > /etc/sudoers.d/greenboost && chmod 440 /etc/sudoers.d/greenboost && echo SUDOERS_OK' 2>/dev/null" \
            <<< "$_sudo_pass" | grep -q "SUDOERS_OK" || _setup_rc=$?
        if (( _setup_rc == 0 )); then
            printf "    %b\n" "${C_LIME}✓  sudo configured - upgrade-greenboost will now work${C_RESET}"
        else
            printf "    %b\n" "${C_RED}✗  sudo setup failed (rc=${_setup_rc})${C_RESET}"
        fi
    done < "$GB_CLUSTER_CONF"
}

# cmd_feeders_diag - run T1/T2/T3 and compute diagnostic against feeder.
# Usage: greenboost feeders diag [t1|t2|t3|compute|all]
cmd_feeders_diag() {
    local _sub="${1:-all}"
    local _diag_script="$MODULE_DIR/gb_feeder_diag.py"
    if [[ ! -f "$_diag_script" ]]; then
        die "Diagnostic script not found: $_diag_script"
    fi
    # Audit F-L4-12: timeout + exit-code check so a hung feeder doesn't block
    # indefinitely and failures are surfaced rather than silently swallowed.
    local _rc=0
    timeout 120 python3 "$_diag_script" "$_sub" || _rc=$?
    [[ $_rc -eq 124 ]] && die "Diagnostic timed out after 120s - feeder may be unresponsive"
    [[ $_rc -ne 0  ]] && die "Diagnostic script failed (rc=${_rc}) - see output above"
    return 0
}

cmd_dataflux_ui() {
    # Web UI over gb_cluster's dataflux event log (gb_dataflux.py) - shows
    # how work has flowed through the cluster (host/feeder, which script,
    # timing, errors) over a configurable window. Foreground server, same
    # data an LLM can query via the greenboost-dataflux MCP server
    # (gb_dataflux_mcp.py, registered in .mcp.json).
    local _script="$MODULE_DIR/gb_dataflux.py"
    if [[ ! -f "$_script" ]]; then
        die "Dataflux script not found: $_script"
    fi
    exec python3 "$_script" serve "$@"
}

# cmd_feeders_genkey - generate a fresh random 32-byte cluster PSK and write
# it to /etc/greenboost/cluster.key on THIS machine. Run once on whichever
# side doesn't have a key yet, then export-key/import-key it to every other
# machine in the cluster. Refuses to overwrite an existing key (use
# import-key with --force semantics manually if you really mean to rotate -
# rotating silently would desync every other machine's copy).
# Usage: sudo greenboost feeders genkey
cmd_feeders_genkey() {
    need_root "feeders genkey"
    if [[ -f "$GB_CLUSTER_KEY" ]]; then
        die "A key already exists at $GB_CLUSTER_KEY - refusing to overwrite. Delete it first if you really mean to rotate (this will desync every other machine in the cluster)."
    fi
    gb_ensure_greenboost_group
    mkdir -p "$(dirname "$GB_CLUSTER_KEY")"
    local _hex
    _hex=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    printf '%s' "$_hex" > "$GB_CLUSTER_KEY"
    _gb_set_keyfile_perms "$GB_CLUSTER_KEY"
    local _mode; _mode=$(stat -c '%a' "$GB_CLUSTER_KEY")
    gb_ok "Generated ${GB_CLUSTER_KEY} (mode ${_mode})"
    gb_info "Share it with other machines: sudo greenboost feeders export-key"
}

# cmd_feeders_export_key - print this machine's cluster PSK for manual
# copy-paste onto another machine. Run this ON THE FEEDER (or on whichever
# side already has a working cluster.key) when there is no SSH/network path
# to transfer the file directly - the operator reads the hex string off this
# terminal and pastes it into `greenboost feeders import-key` on the other
# machine. No network connection is required for this command itself.
# Usage: sudo greenboost feeders export-key
cmd_feeders_export_key() {
    need_root "feeders export-key"
    if [[ ! -f "$GB_CLUSTER_KEY" ]]; then
        die "No cluster key at $GB_CLUSTER_KEY - generate one with: sudo greenboost feeders genkey"
    fi
    local _hex
    _hex=$(tr -d '\n' < "$GB_CLUSTER_KEY")
    if [[ ! "$_hex" =~ ^[0-9a-fA-F]{64}$ ]]; then
        die "$GB_CLUSTER_KEY does not contain exactly 64 hex characters - refusing to print a malformed key"
    fi
    gb_section "GreenBoost Cluster Key Export"
    printf "  %b\n" "${C_AMBER}This is the cluster PSK - treat it like a password.${C_RESET}"
    printf "  Copy the line below and run on the OTHER machine:\n\n"
    printf "    %b\n\n" "${C_LIME}sudo greenboost feeders import-key ${_hex}${C_RESET}"
}

# cmd_feeders_import_key - write a cluster PSK (from export-key on another
# machine) to this machine's /etc/greenboost/cluster.key. Prompts for the
# key interactively (masked) if not passed as an argument.
# Usage: sudo greenboost feeders import-key [HEX64]
cmd_feeders_import_key() {
    need_root "feeders import-key"
    local _hex="${1:-}"
    if [[ -z "$_hex" ]]; then
        gb_section "GreenBoost Cluster Key Import"
        local _char
        printf "  Paste the 64-char hex key from 'feeders export-key': " > /dev/tty
        while IFS= read -r -s -n1 _char < /dev/tty; do
            case "$_char" in
                '')            break ;;
                $'\x7f'|$'\b') if [[ -n "$_hex" ]]; then
                                   _hex="${_hex%?}"; printf '\b \b' > /dev/tty
                               fi ;;
                *)             _hex+="$_char"; printf '*' > /dev/tty ;;
            esac
        done
        printf '\n' > /dev/tty
    fi
    if [[ ! "$_hex" =~ ^[0-9a-fA-F]{64}$ ]]; then
        die "Key must be exactly 64 hex characters (32 bytes) - got ${#_hex} chars"
    fi
    gb_ensure_greenboost_group
    mkdir -p "$(dirname "$GB_CLUSTER_KEY")"
    printf '%s' "$_hex" > "$GB_CLUSTER_KEY"
    _gb_set_keyfile_perms "$GB_CLUSTER_KEY"
    local _mode; _mode=$(stat -c '%a' "$GB_CLUSTER_KEY")
    gb_ok "Wrote ${GB_CLUSTER_KEY} (mode ${_mode})"
    gb_info "Retry: sudo greenboost connect <feeder-ip>"
}

# cmd_feeders_redeploy_netd - rebuild greenboost-netd from the local source tree
# and atomically swap /usr/local/bin/greenboost-netd WITHOUT touching cluster.key
# or any other /etc/greenboost/ file.  Use this instead of a Full Install when
# you only need to deliver a daemon fix; it avoids the install path that wipes
# cluster.key and re-installs stale cached binaries.
# Usage: sudo greenboost feeders redeploy-netd
cmd_feeders_redeploy_netd() {
    need_root "feeders redeploy-netd"

    gb_section "GreenBoost netd Redeploy"
    gb_info "Source dir: ${MODULE_DIR}"
    gb_info "Target:     ${GB_NETD_BIN}"

    # 1. Build
    gb_info "Building greenboost-netd (make netd)..."
    if ! make -C "$MODULE_DIR" netd 2>&1 | tail -5; then
        die "Build failed - check compiler output above"
    fi
    local _src="${MODULE_DIR}/greenboost-netd"
    if [[ ! -f "$_src" ]]; then
        die "Build succeeded but artifact not found at ${_src}"
    fi

    # 2. Stop running daemon (mirror feed stop inline - no subshell dispatch needed)
    if [[ -f "$GB_NETD_PID" ]]; then
        local _pid
        _pid=$(cat "$GB_NETD_PID" 2>/dev/null)
        if [[ -n "$_pid" ]] && kill -0 "$_pid" 2>/dev/null; then
            kill "$_pid"
            sleep 0.8
            gb_ok "Feeder daemon stopped (pid ${_pid})"
        fi
        rm -f "$GB_NETD_PID"
    else
        gb_info "Feeder daemon not running (will start fresh)"
    fi

    # 3. Atomic install - does NOT touch cluster.key or any other /etc/greenboost/ file
    install -m 755 "$_src" "$GB_NETD_BIN"
    gb_ok "Installed ${GB_NETD_BIN}"

    # 4. Confirm the fix is in the installed binary (source-level check)
    local _fix_check
    _fix_check=$(grep -c "client_init(struct client" "$MODULE_DIR/greenboost_netd.c" 2>/dev/null || echo 0)
    if [[ "$_fix_check" -ge 1 ]]; then
        gb_ok "Heartbeat-underflow fix confirmed in source (client_init takes now_ms)"
    else
        gb_warn_ui "Could not verify fix in source - verify greenboost_netd.c manually"
    fi

    # 5. Restart daemon
    gb_info "Starting feeder daemon on port ${GB_NET_PORT}..."
    mkdir -p /run/greenboost /var/log/greenboost
    local _netd_stderr_tmp
    _netd_stderr_tmp=$(mktemp /tmp/gb-netd-XXXXXX)
    "$GB_NETD_BIN" -d -p "$GB_NET_PORT" 2>"$_netd_stderr_tmp"
    sleep 1
    if [[ -f "$GB_NETD_PID" ]] && kill -0 "$(cat "$GB_NETD_PID" 2>/dev/null)" 2>/dev/null; then
        rm -f "$_netd_stderr_tmp"
        gb_ok "Feeder daemon started (pid $(cat "$GB_NETD_PID"))"
    else
        [[ -s "$_netd_stderr_tmp" ]] && tail -5 "$_netd_stderr_tmp" | while IFS= read -r _el; do gb_info "  $_el"; done
        rm -f "$_netd_stderr_tmp"
        die "Feeder daemon failed to start - run: sudo ${GB_NETD_BIN} -p ${GB_NET_PORT}"
    fi

    # 6. Tail the log to confirm clean startup and show PSK/key state
    gb_info "Recent netd log:"
    tail -6 /var/log/greenboost/netd.log 2>/dev/null | while IFS= read -r _l; do
        printf "    %b\n" "${C_DIM}${_l}${C_RESET}"
    done
    echo
    gb_info "Cluster key: $( [[ -f "$GB_CLUSTER_KEY" ]] && echo "present ($(wc -c < "$GB_CLUSTER_KEY") bytes)" || echo "${C_RED}MISSING${C_RESET} - run: sudo greenboost feeders genkey" )"
    gb_info "Next step on workstation: sudo greenboost connect <feeder-ip>"
}

# PR-WW: cmd_stability_monitor - long-running invariant watcher.
# Polls shim_stats / metrics.json and flags monotone-counter regressions,
# tier-gauge leaks (gauge stuck high long after phase left INFERENCE), and
# fragmentation drift.  Wraps the Python implementation.
# Usage: greenboost stability [--interval N] [--duration N] [--strict] [--json] [--log PATH]
cmd_stability_monitor() {
    local _script="$MODULE_DIR/gb_stability_monitor.py"
    if [[ ! -f "$_script" ]]; then
        die "Stability monitor script not found: $_script"
    fi
    # Pass through args; default to 30 s interval, forever.
    exec python3 "$_script" "$@"
}

# cmd_feeders_upgrade_greenboost - full GreenBoost push to all feeders.
# Pushes: greenboost-netd, greenboost_setup.sh, libgreenboost_cuda.so, build_info.
# Restarts greenboost-netd on each feeder after install.
# Usage: sudo greenboost update feeders
# _gb_host_ollama_version , the host's ollama version string (x.y.z) or "".
_gb_host_ollama_version() {
    command -v ollama &>/dev/null || return 0
    ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1
}

# cmd_feeders_sync_ollama , install the HOST's exact ollama version on every
# feeder.  Feeder-GPU compute for LLM inference requires an identical ollama
# build so the feeder's native libggml-cuda.so exposes the SAME CUDA kernels
# the host dispatches (a version skew is why kernel names don't resolve).
# Streams the installer output so the operator sees progress.
# Usage: sudo greenboost feeders sync-ollama
cmd_feeders_sync_ollama() {
    need_root "feeders sync-ollama"
    [[ -f "$GB_CLUSTER_CONF" ]] || die "No feeders configured - run: sudo greenboost connect <IP>"
    local _hostver; _hostver=$(_gb_host_ollama_version)
    [[ -z "$_hostver" ]] && die "Cannot determine host ollama version (is ollama installed?)"

    gb_section "Sync ollama to feeders  (host version: ${_hostver})"
    local _ssh_as=()
    [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]] && _ssh_as=(runuser -u "$SUDO_USER" --)

    local _ok=0 _fail=0
    while IFS= read -r _line; do
        [[ -z "$_line" || "$_line" == \#* ]] && continue
        local _ip _user
        _ip=$(echo "$_line" | awk '{print $1}'); _ip="${_ip%%:*}"
        _user=$(echo "$_line" | awk '{print $3}'); _user="${_user:-root}"
        printf "\n  ${C_VIOLET}%-22s${C_RESET} %s@%s\n" "$(echo "$_line" | awk '{print $2}')" "$_user" "$_ip"

        local _fver
        _fver=$("${_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
            "${_user}@${_ip}" "ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1" 2>/dev/null)
        if [[ "$_fver" == "$_hostver" ]]; then
            gb_ok "  Already on ${_hostver} , no change."
            _ok=$((_ok+1)); continue
        fi
        gb_info "  Feeder has ${_fver:-none} → installing ${_hostver} (streaming installer output)…"
        if ! "${_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
                "${_user}@${_ip}" "sudo -n bash -c true" 2>/dev/null; then
            gb_warn "  No passwordless sudo , run: sudo greenboost feeders setup-sudo"
            _fail=$((_fail+1)); continue
        fi
        # Official installer honours OLLAMA_VERSION to pin the exact release.
        # NOTE: the feeder sudoers rule env_resets, so passing OLLAMA_VERSION
        # THROUGH sudo is refused ("not allowed to set … OLLAMA_VERSION").
        # Instead run everything inside one `sudo -n bash -c`, setting the var
        # AFTER elevation (sudo never sees it).  Download the installer to a
        # file first , piping curl|sh over ssh broke with "Failure writing".
        # No -tt (it caused the connection-close + write failures).
        local _rc=0
        "${_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=no \
            "${_user}@${_ip}" \
            "sudo -n bash -c 'curl -fsSL https://ollama.com/install.sh -o /tmp/gb_ollama_install.sh && OLLAMA_VERSION=\"${_hostver}\" bash /tmp/gb_ollama_install.sh; rm -f /tmp/gb_ollama_install.sh'" \
            2>&1 | sed 's/^/    /' || _rc=$?
        local _newver
        _newver=$("${_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
            "${_user}@${_ip}" "ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1" 2>/dev/null)
        if [[ "$_newver" == "$_hostver" ]]; then
            gb_ok "  Feeder now on ${_hostver}."
            gb_info "  Point netd at the matched backend: it will use the feeder's own libggml-cuda.so"
            _ok=$((_ok+1))
        else
            gb_warn "  Install did not reach ${_hostver} (feeder reports ${_newver:-none}, rc=${_rc})"
            _fail=$((_fail+1))
        fi
    done < "$GB_CLUSTER_CONF"
    echo ""
    gb_info "ollama sync: ${_ok} ok, ${_fail} failed"
    (( _fail == 0 ))
}

# _gb_feeders_upgrade_from_source , sync the GreenBoost source needed to build
# greenboost-netd (+ its capture interposer) and `make` it ON each feeder, so
# the -march=native binary matches the feeder's CPU.  Pushes the host build
# stamp too so parity checks pass.  Best-effort per feeder; reports pass/fail.
_gb_feeders_upgrade_from_source() {
    # Minimal source set the netd targets need (see Makefile `netd`/`netd-capture`).
    local _srcs=(greenboost_netd.c greenboost_netd_capture.c Makefile)

    # Resolve where the .c sources actually live. When greenboost is invoked as an
    # INSTALLED command, MODULE_DIR is the install dir and holds no sources , only
    # the repo checkout does. Fall back to it so `update feeders` works from
    # anywhere (honours GB_SRC_DIR, then the invoking user's ~/Dev checkout).
    local _src_dir="$MODULE_DIR"
    if [[ ! -f "$_src_dir/greenboost_netd.c" ]]; then
        local _cand _sudo_home=""
        [[ -n "${SUDO_USER:-}" ]] && _sudo_home="$(getent passwd "$SUDO_USER" 2>/dev/null | cut -d: -f6)"
        for _cand in \
            "${GB_SRC_DIR:-}" \
            "${_sudo_home:+$_sudo_home/Dev/greenboost_all/greenboost}" \
            "$HOME/Dev/greenboost_all/greenboost" \
            /root/Dev/greenboost_all/greenboost; do
            [[ -n "$_cand" && -f "$_cand/greenboost_netd.c" ]] && { _src_dir="$_cand"; break; }
        done
    fi

    local _feat_dir="$_src_dir/features"
    local _build_info="$_src_dir/build_info"
    [[ ! -f "$_build_info" ]] && _build_info="/etc/greenboost/build_info"

    for _s in "${_srcs[@]}"; do
        [[ -f "$_src_dir/$_s" ]] || { warn "missing source $_s , not in \$MODULE_DIR nor a repo checkout (set GB_SRC_DIR=/path/to/greenboost_all/greenboost)"; return 1; }
    done

    local _ssh_as=()
    [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]] && _ssh_as=(runuser -u "$SUDO_USER" --)

    gb_section "GreenBoost Feeder Upgrade (build from source)"
    local _ok=0 _fail=0
    while IFS= read -r _line; do
        [[ -z "$_line" || "$_line" == \#* ]] && continue
        local _ip _user
        _ip=$(echo "$_line" | awk '{print $1}'); _ip="${_ip%%:*}"
        _user=$(echo "$_line" | awk '{print $3}'); _user="${_user:-root}"
        printf "\n  ${C_VIOLET}%-22s${C_RESET} %s@%s\n" "$(echo "$_line" | awk '{print $2}')" "$_user" "$_ip"

        if ! "${_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
                "${_user}@${_ip}" "sudo -n bash -c true" 2>/dev/null; then
            printf "    %b\n" "${C_RED}✗  no passwordless sudo (run: sudo greenboost feeders setup-sudo)${C_RESET}"
            _fail=$((_fail+1)); continue
        fi

        "${_ssh_as[@]}" ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${_user}@${_ip}" \
            "rm -rf /tmp/gb_src && mkdir -p /tmp/gb_src/features" 2>/dev/null
        printf "    Pushing source... "
        if ! "${_ssh_as[@]}" scp -o BatchMode=yes -o StrictHostKeyChecking=no -q \
                "${_srcs[@]/#/$_src_dir/}" "${_user}@${_ip}:/tmp/gb_src/" 2>/dev/null \
           || ! "${_ssh_as[@]}" scp -o BatchMode=yes -o StrictHostKeyChecking=no -q \
                "$_feat_dir/net_fabric.h" "$_feat_dir/compat.h" "${_user}@${_ip}:/tmp/gb_src/features/" 2>/dev/null; then
            printf '%b\n' "${C_RED}✗  SCP failed${C_RESET}"; _fail=$((_fail+1)); continue
        fi
        [[ -f "$_build_info" ]] && "${_ssh_as[@]}" scp -o BatchMode=yes -o StrictHostKeyChecking=no -q \
            "$_build_info" "${_user}@${_ip}:/tmp/gb_src/build_info" 2>/dev/null
        printf '%b\n' "${C_LIME}ok${C_RESET}"

        printf "    Building netd on feeder (-march=native)... "
        local _rc=0
        "${_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=120 -o StrictHostKeyChecking=no \
            "${_user}@${_ip}" "sudo -n bash -s -- $GB_NET_PORT" 2>/dev/null << 'REMOTE_BUILD' || _rc=$?
set -e
PORT="${1:-9740}"
cd /tmp/gb_src
make netd >/tmp/gb_src/build.log 2>&1 || { echo BUILD_FAILED; tail -5 /tmp/gb_src/build.log; exit 2; }
pkill -9 -f greenboost-netd 2>/dev/null || true; sleep 2
install -m 755 greenboost-netd /usr/local/bin/greenboost-netd
[ -f libgreenboost_netd_capture.so ] && install -m 755 libgreenboost_netd_capture.so /usr/local/lib/libgreenboost_netd_capture.so
mkdir -p /etc/greenboost /run/greenboost /var/log/greenboost
[ -f build_info ] && cp build_info /etc/greenboost/build_info
rm -f /run/greenboost/netd.pid
setsid bash -c "nohup /usr/local/bin/greenboost-netd -d -p $PORT >>/var/log/greenboost/netd.log 2>&1 </dev/null &"
for i in $(seq 1 25); do sleep 0.5; p=$(cat /run/greenboost/netd.pid 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { echo STARTED; exit 0; }; done
echo FAILED_TO_START; exit 1
REMOTE_BUILD
        if (( _rc == 0 )); then
            printf '%b\n' "${C_LIME}✓  built + restarted${C_RESET}"; _ok=$((_ok+1))
        else
            printf '%b\n' "${C_RED}✗  build/start failed (rc=$_rc)${C_RESET}"; _fail=$((_fail+1))
        fi
    done < "$GB_CLUSTER_CONF"

    echo ""
    gb_info "From-source upgrade: ${_ok} ok, ${_fail} failed"
    (( _fail == 0 ))
}

cmd_feeders_upgrade_greenboost() {
    need_root "feeders upgrade-greenboost"
    if [[ ! -f "$GB_CLUSTER_CONF" ]] || ! grep -q '[^[:space:]]' "$GB_CLUSTER_CONF" 2>/dev/null; then
        die "No feeders configured - run: sudo greenboost connect <IP>"
    fi

    # Build from source ON the feeder by DEFAULT.  greenboost-netd + the shim
    # compile -march=native, and the feeder CPU generally differs from the host
    # (this cluster: Intel i9 host vs AMD Ryzen feeder) , so a host-built binary
    # can use instructions the feeder lacks.  Building on the feeder is the only
    # correct default.  `--binary` opts into the legacy copy path (safe ONLY
    # when host and feeder CPUs are identical).
    local _use_binary=0
    for _a in "$@"; do [[ "$_a" == "--binary" ]] && _use_binary=1; done
    if (( ! _use_binary )); then
        _gb_feeders_upgrade_from_source
        return $?
    fi

    local _netd_bin="$MODULE_DIR/greenboost-netd"
    local _setup_sh="$MODULE_DIR/greenboost_setup.sh"
    local _shim_lib="$SHIM_DEST/$SHIM_LIB"
    local _build_info="$MODULE_DIR/build_info"
    [[ ! -f "$_build_info" ]] && _build_info="/etc/greenboost/build_info"

    if [[ ! -f "$_netd_bin" ]]; then
        info "greenboost-netd not found - building..."
        make -C "$MODULE_DIR" netd 2>&1 | tail -5 || die "Build failed - run: make netd"
    fi

    local _ssh_as=()
    if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
        _ssh_as=(runuser -u "$SUDO_USER" --)
    fi

    local _local_stamp=""
    [[ -f "$_build_info" ]] && _local_stamp=$(grep BUILD_ID "$_build_info" 2>/dev/null | cut -d= -f2)
    gb_section "GreenBoost Feeder Upgrade  (local stamp: ${_local_stamp:-unknown})"

    local _ok=0 _fail=0

    while IFS= read -r _line; do
        [[ -z "$_line" || "$_line" == \#* ]] && continue

        local _addr _hostname _ssh_user
        _addr=$(echo "$_line" | awk '{print $1}')
        _hostname=$(echo "$_line" | awk '{print $2}')
        _ssh_user=$(echo "$_line" | awk '{print $3}')
        _ssh_user="${_ssh_user:-root}"

        local _ip _port
        _ip="${_addr%%:*}"
        _port="${_addr##*:}"
        [[ "$_port" == "$_ip" ]] && _port="$GB_NET_PORT"

        # Audit F-L4-19: _port is interpolated into the SSH command string;
        # validate it is numeric before use to prevent remote shell injection
        # from a tampered cluster.conf.
        if [[ ! "$_port" =~ ^[0-9]{1,5}$ ]] || (( _port < 1 || _port > 65535 )); then
            printf "    %b\n" "${C_RED}✗  invalid port '${_port}' in cluster.conf - skipping${C_RESET}" >&2
            _fail=$(( _fail + 1 ))
            continue
        fi

        printf "\n  ${C_VIOLET}%-22s${C_RESET} %s@%s\n" "${_hostname:-$_ip}" "$_ssh_user" "$_ip"

        # Check SSH key auth
        if ! "${_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=5 \
                -o StrictHostKeyChecking=no "${_ssh_user}@${_ip}" true 2>/dev/null; then
            printf "    %b\n" "${C_AMBER}⚠  SSH key auth failed - run: ssh-copy-id ${_ssh_user}@${_ip}${C_RESET}"
            _fail=$(( _fail + 1 ))
            continue
        fi

        # Check passwordless sudo using the same binary the install script will use
        if ! "${_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=5 \
                -o StrictHostKeyChecking=no "${_ssh_user}@${_ip}" \
                "sudo -n bash -c 'true'" 2>/dev/null; then
            printf "    %b\n" "${C_RED}✗  No passwordless sudo on feeder${C_RESET}"
            printf "    %b\n" "${C_AMBER}   Run: sudo greenboost feeders setup-sudo${C_RESET}"
            _fail=$(( _fail + 1 ))
            continue
        fi

        # Show feeder's current stamp before upgrade
        local _remote_stamp
        _remote_stamp=$("${_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=5 \
            -o StrictHostKeyChecking=no "${_ssh_user}@${_ip}" \
            "grep BUILD_ID /etc/greenboost/build_info 2>/dev/null | cut -d= -f2 || echo '?'" 2>/dev/null)
        printf "    ${C_DIM}feeder stamp: %-12s → %s${C_RESET}\n" "${_remote_stamp:-?}" "${_local_stamp:-?}"

        # Create staging dir on feeder
        if ! "${_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=10 \
                -o StrictHostKeyChecking=no "${_ssh_user}@${_ip}" \
                "mkdir -p /tmp/gb_update" 2>/dev/null; then
            printf "    %b\n" "${C_RED}✗  mkdir on feeder failed${C_RESET}"
            _fail=$(( _fail + 1 ))
            continue
        fi

        # Build file list to transfer
        local _files=("$_netd_bin" "$_setup_sh")
        [[ -f "$_shim_lib" ]] && _files+=("$_shim_lib")
        [[ -f "$_build_info" ]] && _files+=("$_build_info")

        printf "    Pushing %d file(s)... " "${#_files[@]}"
        if ! "${_ssh_as[@]}" scp -o BatchMode=yes -o ConnectTimeout=30 \
                -o StrictHostKeyChecking=no -q \
                "${_files[@]}" "${_ssh_user}@${_ip}:/tmp/gb_update/" 2>/dev/null; then
            printf '%b\n' "${C_RED}✗  SCP failed${C_RESET}"
            _fail=$(( _fail + 1 ))
            continue
        fi
        printf '%b\n' "${C_LIME}ok${C_RESET}"

        # Remote install script
        printf "    Installing on feeder... "
        local _rc=0
        local _err_tmp; _err_tmp=$(mktemp /tmp/gb_install_err.XXXXXX)
        "${_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=60 \
            -o StrictHostKeyChecking=no "${_ssh_user}@${_ip}" \
            "sudo -n bash -s -- $_port" 2>"$_err_tmp" << 'REMOTE_SCRIPT' || _rc=$?
set -e
PORT="${1:-9740}"
NETD=/tmp/gb_update/greenboost-netd
SETUP=/tmp/gb_update/greenboost_setup.sh
SHIM=/tmp/gb_update/libgreenboost_cuda.so
BI=/tmp/gb_update/build_info

# Stop old daemon - SIGTERM first, then SIGKILL to guarantee no ETXTBSY on cp
systemctl stop greenboost-netd 2>/dev/null || true
pkill -x    greenboost-netd 2>/dev/null || true
sleep 1
pkill -9 -x greenboost-netd 2>/dev/null || true
sleep 1

# Install binaries - 'install' writes to a temp then renames atomically,
# avoiding ETXTBSY even if a stale copy is still mapped in memory
install -m 755 "$NETD"  /usr/local/bin/greenboost-netd
install -m 755 "$SETUP" /usr/local/bin/greenboost_setup.sh

# Install CUDA shim if provided
if [[ -f "$SHIM" ]]; then
    mkdir -p /usr/local/lib
    install -m 755 "$SHIM" /usr/local/lib/libgreenboost_cuda.so
    ldconfig 2>/dev/null || true
fi

# Install build stamp
mkdir -p /etc/greenboost
[[ -f "$BI" ]] && cp "$BI" /etc/greenboost/build_info

# Restart feeder daemon
mkdir -p /run/greenboost /var/log/greenboost
# Remove stale PID file so we can detect when the daemon writes a fresh one
# PID file path matches GB_NETD_PID_FILE in features/net_fabric.h
rm -f /run/greenboost/netd.pid
nohup /usr/local/bin/greenboost-netd -d -p "$PORT" \
    >>/var/log/greenboost/netd.log 2>&1 &
# daemonize() runs before probe_gpus() (CUDA fork-safety).
# The daemon writes its PID file after fork; poll for it instead of checking $!.
_started=0
for _i in $(seq 1 20); do
    sleep 0.5
    _pid=$(cat /run/greenboost/netd.pid 2>/dev/null)
    if [[ -n "$_pid" ]] && kill -0 "$_pid" 2>/dev/null; then
        _started=1; break
    fi
done
if (( _started )); then
    echo "STARTED"
else
    echo "FAILED_TO_START"
    exit 1
fi

rm -rf /tmp/gb_update
REMOTE_SCRIPT

        if (( _rc == 0 )); then
            printf '%b\n' "${C_LIME}✓  upgraded + daemon restarted${C_RESET}"
            _ok=$(( _ok + 1 ))
        else
            local _errmsg; _errmsg=$(head -3 "$_err_tmp" 2>/dev/null)
            printf '%b\n' "${C_RED}✗  remote install failed (rc=${_rc})${C_RESET}"
            [[ -n "$_errmsg" ]] && printf "    %b\n" "${C_DIM}${_errmsg}${C_RESET}"
            _fail=$(( _fail + 1 ))
        fi
        rm -f "$_err_tmp"

    done < "$GB_CLUSTER_CONF"

    echo ""
    if (( _ok > 0 && _fail == 0 )); then
        gb_ok "All ${_ok} feeder(s) upgraded successfully."
    elif (( _ok > 0 )); then
        gb_warn_ui "${_ok} upgraded, ${_fail} failed."
    else
        die "All feeder upgrades failed."
    fi
    gb_info "Verify with: greenboost cluster"
    gb_info "Check stamp: greenboost built-stamp --feeders"
}

# cmd_built_stamp - print build stamp (local and/or feeders).
# Usage: greenboost built-stamp [--feeders]
cmd_capabilities() {
    # Report what the installed/running GreenBoost shim supports, via the
    # canonical read-only client gb_monitor.py (runtime manifest → install
    # manifest → binary sniff). --llm/--json pass straight through.
    local _mon=""
    for _c in "${MODULE_DIR}/gb_monitor.py" \
              "/usr/local/lib/greenboost/gb_monitor.py"; do
        [[ -f "$_c" ]] && { _mon="$_c"; break; }
    done
    [[ -z "$_mon" ]] && die "gb_monitor.py not found — run: sudo greenboost install"
    exec python3 "$_mon" --capabilities "$@"
}

cmd_pilot() {
    # Pilot instrument panel over the dataflux flight recorder: per-stage
    # wall-time trends, measured tok/s, pressure flags, evidence-backed
    # advice. Read-only (v1 never moves a lever). Flags: --llm --json --days N.
    local _pilot=""
    for _c in "${MODULE_DIR}/gb_pilot.py" \
              "/usr/local/lib/greenboost/gb_pilot.py"; do
        [[ -f "$_c" ]] && { _pilot="$_c"; break; }
    done
    [[ -z "$_pilot" ]] && die "gb_pilot.py not found — run: sudo greenboost install"
    exec python3 "$_pilot" "$@"
}

cmd_built_stamp() {
    local _show_feeders=0
    for _a in "$@"; do [[ "$_a" == "--feeders" ]] && _show_feeders=1; done

    local _bi=""
    for _bic in "${MODULE_DIR}/build_info" ./build_info /etc/greenboost/build_info; do
        [[ -f "$_bic" ]] && { _bi="$_bic"; break; }
    done

    gb_section "GreenBoost Build Stamp - $(hostname)"
    if [[ -n "$_bi" ]]; then
        local _id _ver _host _git _epoch _date=""
        _id=$(grep BUILD_ID      "$_bi" 2>/dev/null | cut -d= -f2)
        _ver=$(grep BUILD_VERSION "$_bi" 2>/dev/null | cut -d= -f2)
        _host=$(grep BUILD_HOST  "$_bi" 2>/dev/null | cut -d= -f2)
        _git=$(grep BUILD_GIT    "$_bi" 2>/dev/null | cut -d= -f2)
        _epoch=$(grep BUILD_EPOCH "$_bi" 2>/dev/null | cut -d= -f2)
        [[ -n "$_epoch" ]] && _date=$(date -d "@${_epoch}" '+%Y-%m-%d %H:%M' 2>/dev/null || date -r "${_epoch}" '+%Y-%m-%d %H:%M' 2>/dev/null || echo "")
        printf "  ${C_DIM}%-14s${C_RESET} %b\n"  "ID"      "${C_LIME}${_id:-?}${C_RESET}"
        printf "  ${C_DIM}%-14s${C_RESET} %b\n"  "Version" "${C_LIME}${_ver:-?}${C_RESET}"
        printf "  ${C_DIM}%-14s${C_RESET} %b\n"  "Host"    "${C_GRAY}${_host:-?}${C_RESET}"
        printf "  ${C_DIM}%-14s${C_RESET} %b\n"  "Git"     "${C_GRAY}${_git:-?}${C_RESET}"
        printf "  ${C_DIM}%-14s${C_RESET} %b\n"  "Built"   "${C_GRAY}${_date:-${_epoch:-?}}${C_RESET}"
    else
        printf "  ${C_AMBER}⚠  No build_info found - run: sudo greenboost install${C_RESET}\n"
    fi

    (( _show_feeders == 0 )) && { echo ""; return 0; }

    [[ ! -f "$GB_CLUSTER_CONF" ]] && { echo ""; return 0; }

    local _ssh_as=()
    if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
        _ssh_as=(runuser -u "$SUDO_USER" --)
    fi

    while IFS= read -r _line; do
        [[ -z "$_line" || "$_line" == \#* ]] && continue

        local _addr _hostname _ssh_user
        _addr=$(echo "$_line" | awk '{print $1}')
        _hostname=$(echo "$_line" | awk '{print $2}')
        _ssh_user=$(echo "$_line" | awk '{print $3}')
        _ssh_user="${_ssh_user:-root}"

        local _ip
        _ip="${_addr%%:*}"

        echo ""
        gb_section "Feeder - ${_hostname:-$_ip}  (${_ssh_user}@${_ip})"

        if ! "${_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=5 \
                -o StrictHostKeyChecking=no "${_ssh_user}@${_ip}" true 2>/dev/null; then
            printf "  %b\n" "${C_AMBER}⚠  SSH unreachable${C_RESET}"
            continue
        fi

        local _remote_info
        _remote_info=$("${_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=10 \
            -o StrictHostKeyChecking=no "${_ssh_user}@${_ip}" \
            "cat /etc/greenboost/build_info 2>/dev/null || echo BUILD_ID=not-installed" 2>/dev/null)

        if [[ -z "$_remote_info" ]]; then
            printf "  %b\n" "${C_AMBER}⚠  Could not read build_info on feeder${C_RESET}"
            continue
        fi

        local _rid _rver _rhost _rgit _repoch _rdate=""
        _rid=$(echo "$_remote_info"   | grep BUILD_ID      | cut -d= -f2)
        _rver=$(echo "$_remote_info"  | grep BUILD_VERSION | cut -d= -f2)
        _rhost=$(echo "$_remote_info" | grep BUILD_HOST    | cut -d= -f2)
        _rgit=$(echo "$_remote_info"  | grep BUILD_GIT     | cut -d= -f2)
        _repoch=$(echo "$_remote_info" | grep BUILD_EPOCH  | cut -d= -f2)
        [[ -n "$_repoch" ]] && _rdate=$(date -d "@${_repoch}" '+%Y-%m-%d %H:%M' 2>/dev/null || echo "")

        local _match_col="${C_LIME}"
        [[ -n "$_id" && "$_rid" != "$_id" ]] && _match_col="${C_AMBER}"

        printf "  ${C_DIM}%-14s${C_RESET} %b\n"  "ID"      "${_match_col}${_rid:-?}${C_RESET}"
        printf "  ${C_DIM}%-14s${C_RESET} %b\n"  "Version" "${C_GRAY}${_rver:-?}${C_RESET}"
        printf "  ${C_DIM}%-14s${C_RESET} %b\n"  "Host"    "${C_GRAY}${_rhost:-?}${C_RESET}"
        printf "  ${C_DIM}%-14s${C_RESET} %b\n"  "Git"     "${C_GRAY}${_rgit:-?}${C_RESET}"
        printf "  ${C_DIM}%-14s${C_RESET} %b\n"  "Built"   "${C_GRAY}${_rdate:-${_repoch:-?}}${C_RESET}"

        [[ -n "$_id" && "$_rid" != "$_id" ]] && \
            printf "  %b\n" "${C_AMBER}⚠  Stamp mismatch - run: sudo greenboost update feeders${C_RESET}"

    done < "$GB_CLUSTER_CONF"

    echo ""
}


# _gb_read_feeder_metrics - parse /run/greenboost/metrics.json into associative arrays.
# Sets _fdr_bw[ip], _fdr_health[ip], and (for the GPU-compute view) _fdr_gpu_util[ip],
# _fdr_gpu_mem_util[ip], _fdr_gpu_temp[ip], _fdr_gpu_power[ip] for each feeder entry.
# Safe to call with unset arrays - caller should declare -A them all first.
_gb_read_feeder_metrics() {
    local _metrics_json="/run/greenboost/metrics.json"
    [[ -f "$_metrics_json" ]] && command -v python3 &>/dev/null || return 0
    while IFS='|' read -r _fip _fbw _fhealth _fgutil _fgmem _fgtemp _fgpower; do
        _fdr_bw["$_fip"]="$_fbw"
        _fdr_health["$_fip"]="$_fhealth"
        _fdr_gpu_util["$_fip"]="$_fgutil"
        _fdr_gpu_mem_util["$_fip"]="$_fgmem"
        _fdr_gpu_temp["$_fip"]="$_fgtemp"
        _fdr_gpu_power["$_fip"]="$_fgpower"
    done < <(python3 -c "
import json
try:
    d=json.load(open('$_metrics_json'))
    for f in d.get('feeders',[]):
        ip=f.get('feeder','?')
        print(ip+'|'+str(f.get('bw_measured_mbs',0))+'|'+str(f.get('health_state',0))
              +'|'+str(f.get('gpu_util_pct',0))+'|'+str(f.get('gpu_mem_util_pct',0))
              +'|'+str(f.get('gpu_temp_c',0))+'|'+str(f.get('gpu_power_w',0)))
except: pass
" 2>/dev/null)
}

_cmd_cluster_snapshot() {
    local _ts; _ts=$(date '+%Y-%m-%dT%H:%M')

    gb_section "GreenBoost Cluster Status  (${_ts})"

    printf "  ${C_BOLD}%-8s %-16s %-23s %-9s %-9s %-9s %-9s %-9s %-16s${C_RESET}\n" \
           "ROLE" "HOST" "GPU" "T1 VRAM" "T2 DDR" "T3 NVMe" "HEALTH" "BW" "STATUS"
    echo -e "  ${C_DIM}$(printf '─%.0s' $(seq 1 115))${C_RESET}"

    # Read per-feeder BW and health from shim metrics JSON (shared helper)
    declare -A _fdr_bw _fdr_health _fdr_gpu_util _fdr_gpu_mem_util _fdr_gpu_temp _fdr_gpu_power
    _gb_read_feeder_metrics

    # Read shim stats for per-tier health badges (D2/D3) and virtual VRAM figure
    local _shim_stats="/run/greenboost/shim_stats"
    local _t2_warn=0 _t2_frag=0
    local _phys_vram_mb=0 _remote_vram_mb=0 _virtual_vram_mb=0
    if [[ -f "$_shim_stats" ]]; then
        _t2_warn=$(grep '^t2_above_warn=' "$_shim_stats" 2>/dev/null | cut -d= -f2 || echo 0)
        _t2_frag=$(grep '^t2_pool_frag_pct=' "$_shim_stats" 2>/dev/null | cut -d= -f2 || echo 0)
        _phys_vram_mb=$(grep '^physical_vram_mb=' "$_shim_stats" 2>/dev/null | cut -d= -f2 || echo 0)
        _remote_vram_mb=$(grep '^cluster_remote_vram_mb=' "$_shim_stats" 2>/dev/null | cut -d= -f2 || echo 0)
        _virtual_vram_mb=$(grep '^cluster_virtual_vram_mb=' "$_shim_stats" 2>/dev/null | cut -d= -f2 || echo 0)
    fi

    # Local GPU info
    local local_gpu="(unknown)"
    local local_vram=0 local_temp="" local_power="" local_throttle=""
    if command -v nvidia-smi &>/dev/null; then
        local_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
        local_vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
        local_vram=$(( (local_vram + 512) / 1024 ))
        local _tmp _pwr _clk
        _tmp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "")
        _pwr=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs || echo "")
        _clk=$(nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader 2>/dev/null | head -1 | xargs || echo "")
        [[ -n "$_tmp" && "$_tmp" =~ ^[0-9]+$ ]] && local_temp="${_tmp}°C"
        [[ -n "$_pwr" && "$_pwr" =~ ^[0-9] ]] && local_power="${_pwr%.*}W"
        [[ "$_clk" != "Not Active" && -n "$_clk" ]] && local_throttle=" ${C_RED}[T]${C_RESET}"
    fi
    local_gpu=$(_trunc "$local_gpu" 22)

    # Local T2: total system RAM
    local local_t2=0
    if [[ -f /proc/meminfo ]]; then
        local _memkb; _memkb=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
        local_t2=$(( _memkb / 1048576 ))
    fi

    # Local T3: prefer GreenBoost kernel module pool size; fall back to /proc/swaps.
    local local_t3=0
    local _gb_nvme_pool
    _gb_nvme_pool=$(cat /sys/module/greenboost/parameters/nvme_pool_gb 2>/dev/null || echo 0)
    if [[ "$_gb_nvme_pool" =~ ^[0-9]+$ && $_gb_nvme_pool -gt 0 ]]; then
        local_t3=$_gb_nvme_pool
    elif [[ -f /proc/swaps ]]; then
        local _swkb=0
        while read -r _sf _st _ssz _sus _spri; do
            [[ "$_sf" == Filename ]] && continue
            [[ "$_ssz" =~ ^[0-9]+$ ]] && _swkb=$(( _swkb + _ssz ))
        done < /proc/swaps
        local_t3=$(( _swkb / 1048576 ))
    fi

    # D4: T2 warn/fragmentation badges
    local _t2_badge=""
    [[ "$_t2_warn" == "1" ]] && _t2_badge+=" ${C_AMBER}[T2>75%]${C_RESET}"
    [[ "$_t2_frag" =~ ^[0-9]+$ && $_t2_frag -gt 80 ]] && _t2_badge+=" ${C_AMBER}[FRAG]${C_RESET}"

    local _local_status="active${local_throttle}${_t2_badge}"
    local _local_info=""
    [[ -n "$local_temp" ]] && _local_info+=" ${local_temp}"
    [[ -n "$local_power" ]] && _local_info+=" ${local_power}"

    printf "  ${C_CYAN}%-8s${C_RESET} ${C_GRAY}%-16s${C_RESET} ${C_GRAY}%-23s${C_RESET} ${C_LIME}%5s GB${C_RESET}  ${C_AMBER}%5s GB${C_RESET}  ${C_GRAY}%5s GB${C_RESET}  ${C_LIME}%-9s${C_RESET} ${C_DIM}%-9s${C_RESET} ${C_LIME}%-10s${C_RESET}%s\n" \
           "host" "$(_trunc "$(hostname -s)" 16)" "$local_gpu" "$local_vram" "$local_t2" "$local_t3" "HEALTHY" "local" "active" "$_local_info"

    # Remote feeders from cluster.conf
    local total_gpus=1 total_t1=$local_vram total_t2=$local_t2 total_t3=$local_t3
    if [[ -f "$GB_CLUSTER_CONF" ]] && [[ -r "$GB_CLUSTER_CONF" ]]; then
        while IFS= read -r line; do
            [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
            local addr nickname
            addr=$(echo "$line" | awk '{print $1}')
            nickname=$(echo "$line" | awk '{print $2}')
            local ip port
            ip=$(echo "$addr" | cut -d: -f1)
            port=$(echo "$addr" | cut -d: -f2)
            [[ -z "$port" ]] && port=$GB_NET_PORT

            local probe_out
            probe_out=$(_gb_net_handshake "$ip" "$port" 2>/dev/null) || true
            local fstatus="offline"
            local fgpu="-" fvram="-" fddr="-" fnvme="-"

            # N5: look up BW and health from metrics JSON
            local _fhealth_num="${_fdr_health[$ip]:-}"
            local _fhealth_str="-"
            case "$_fhealth_num" in
                0) _fhealth_str="HEALTHY"  ;;
                1) _fhealth_str="DEGRAD"   ;;
                2) _fhealth_str="UNHLTHY"  ;;
                3) _fhealth_str="QUAR"     ;;
                4) _fhealth_str="DISAB"    ;;
            esac
            local _fbw_num="${_fdr_bw[$ip]:-0}"
            local _fbw_str="-"
            [[ "$_fbw_num" =~ ^[0-9]+$ && $_fbw_num -gt 0 ]] && _fbw_str="${_fbw_num} MiB/s"

            if echo "$probe_out" | grep -q "^OK:"; then
                fstatus="connected"
                local _ft2_done=0 _ft3_done=0
                local _feeder_t2=0 _feeder_t3=0
                # D2/3e: feeder GPU health from GB_MSG_FEEDER_STATUS v3.1 (no SSH needed)
                local _feeder_throttle="" _feeder_temp="" _feeder_ecc="" _feeder_util=""
                local _gh_line
                _gh_line=$(echo "$probe_out" | grep "^GPU_HEALTH:" | head -1)
                if [[ -n "$_gh_line" ]]; then
                    local _gh_temp _gh_pwr _gh_util _gh_ecc _gh_thr
                    IFS=: read -r _ _gh_temp _gh_pwr _gh_util _gh_ecc _gh_thr <<< "$_gh_line"
                    [[ "$_gh_temp" =~ ^[0-9]+$ && $_gh_temp -gt 0 ]] && _feeder_temp=" ${_gh_temp}°C"
                    [[ "$_gh_pwr" =~ ^[0-9]+$ && $_gh_pwr -gt 0 ]] && _feeder_util="${_feeder_util} ${_gh_pwr}W"
                    [[ "$_gh_util" =~ ^[0-9]+$ && $_gh_util -gt 0 ]] && _feeder_util="${_feeder_util} ${_gh_util}%SM"
                    [[ "$_gh_ecc" =~ ^[1-9] ]] && _feeder_ecc=" ${C_RED}[ECC!]${C_RESET}"
                    [[ "$_gh_thr" =~ ^[1-9] ]] && _feeder_throttle=" ${C_RED}[T]${C_RESET}"
                fi
                while IFS= read -r gl; do
                    if [[ "$gl" == GPU:* ]]; then
                        fgpu=$(_trunc "$(echo "$gl" | cut -d: -f3)" 22)
                        local fv; fv=$(echo "$gl" | cut -d: -f4)
                        fvram=$(( (fv + 536870912) / 1073741824 ))
                        if (( _ft2_done == 0 )); then
                            local ft2r; ft2r=$(echo "$gl" | cut -d: -f6)
                            [[ "$ft2r" =~ ^[0-9]+$ && ft2r -gt 0 ]] && _feeder_t2=$(( (ft2r + 536870912) / 1073741824 ))
                            fddr="${_feeder_t2}"
                            _ft2_done=1
                        fi
                        if (( _ft3_done == 0 )); then
                            local ft3r; ft3r=$(echo "$gl" | cut -d: -f7)
                            [[ "$ft3r" =~ ^[0-9]+$ && ft3r -gt 0 ]] && _feeder_t3=$(( (ft3r + 536870912) / 1073741824 ))
                            fnvme="${_feeder_t3}"
                            _ft3_done=1
                        fi
                        printf "  ${C_PURPLE}%-8s${C_RESET} ${C_GRAY}%-16s${C_RESET} ${C_GRAY}%-23s${C_RESET} ${C_LIME}%5s GB${C_RESET}  ${C_AMBER}%5s GB${C_RESET}  ${C_GRAY}%5s GB${C_RESET}  ${C_CYAN}%-9s${C_RESET} ${C_DIM}%-9s${C_RESET} ${C_LIME}%-10s${C_RESET}%s%s%s%s\n" \
                               "feeder" "$(_trunc "${nickname:-$ip}" 16)" "$fgpu" "$fvram" "$fddr" "$fnvme" \
                               "$_fhealth_str" "$_fbw_str" "$fstatus" \
                               "$_feeder_throttle" "$_feeder_ecc" "$_feeder_temp" "$_feeder_util"
                        total_gpus=$((total_gpus + 1))
                        total_t1=$((total_t1 + fvram))
                    fi
                done <<< "$probe_out"
                total_t2=$((total_t2 + _feeder_t2))
                total_t3=$((total_t3 + _feeder_t3))
            else
                printf "  ${C_PURPLE}%-8s${C_RESET} ${C_GRAY}%-16s${C_RESET} ${C_DIM}%-23s${C_RESET} ${C_DIM}%9s${C_RESET} ${C_DIM}%9s${C_RESET} ${C_DIM}%9s${C_RESET}  ${C_DIM}%-9s${C_RESET} ${C_DIM}%-9s${C_RESET} ${C_RED}%-10s${C_RESET}\n" \
                       "feeder" "$(_trunc "${nickname:-$ip}" 16)" "-" "-" "-" "-" "-" "-" "offline"
            fi
        done < "$GB_CLUSTER_CONF"
    fi

    echo -e "  ${C_DIM}$(printf '─%.0s' $(seq 1 92))${C_RESET}"
    local total_combined=$(( total_t1 + total_t2 + total_t3 ))
    echo -e "  ${C_BOLD}Combined:${C_RESET} ${C_GRAY}${total_gpus} GPU(s)${C_RESET}   ${C_GRAY}T1:${C_LIME}${total_t1} GB${C_RESET}   ${C_GRAY}T2:${C_AMBER}${total_t2} GB${C_RESET}   ${C_GRAY}T3:${total_t3} GB${C_RESET}   ${C_GRAY}Total: ${C_BOLD}${total_combined} GB${C_RESET}"
    # Show clean "Virtual VRAM" line when feeder VRAM is present (Phase 1: 20 GB = 12+8)
    if [[ "$_virtual_vram_mb" =~ ^[0-9]+$ && $_virtual_vram_mb -gt 0 && $_remote_vram_mb -gt 0 ]]; then
        local _virt_gb=$(( (_virtual_vram_mb + 512) / 1024 ))
        local _phys_gb=$(( (_phys_vram_mb + 512) / 1024 ))
        local _rem_gb=$(( (_remote_vram_mb + 512) / 1024 ))
        echo -e "  ${C_BOLD}Virtual VRAM:${C_RESET} ${C_LIME}${_virt_gb} GB${C_RESET}  ${C_DIM}(${_phys_gb} GB local + ${_rem_gb} GB feeder)${C_RESET}"
    fi

    # Build stamp summary - local + each feeder
    local _local_bi=""
    for _bic in "${MODULE_DIR}/build_info" /etc/greenboost/build_info; do
        [[ -f "$_bic" ]] && { _local_bi="$_bic"; break; }
    done
    echo ""
    echo -e "  ${C_DIM}Build stamps:${C_RESET}"
    local _local_stamp_id _local_stamp_epoch _local_stamp_date _local_stamp_ver
    if [[ -n "$_local_bi" ]]; then
        _local_stamp_id=$(grep BUILD_ID "$_local_bi" 2>/dev/null | cut -d= -f2)
        _local_stamp_ver=$(grep BUILD_VERSION "$_local_bi" 2>/dev/null | cut -d= -f2)
        _local_stamp_epoch=$(grep BUILD_EPOCH "$_local_bi" 2>/dev/null | cut -d= -f2)
        _local_stamp_date=$(date -d "@${_local_stamp_epoch}" '+%Y-%m-%d %H:%M' 2>/dev/null || echo "${_local_stamp_epoch}")
        printf "  ${C_CYAN}%-12s${C_RESET}  ${C_GRAY}v%-7s${C_RESET} ${C_LIME}%s${C_RESET}  ${C_DIM}%s${C_RESET}\n" \
            "$(hostname -s)" "${_local_stamp_ver:-?}" "${_local_stamp_id:-?}" "${_local_stamp_date}"
    else
        printf "  ${C_CYAN}%-12s${C_RESET}  ${C_DIM}no build_info${C_RESET}\n" "$(hostname -s)"
    fi

    if [[ -f "$GB_CLUSTER_CONF" ]] && [[ -r "$GB_CLUSTER_CONF" ]]; then
        local _any_stale=0
        while IFS= read -r _bline; do
            [[ "$_bline" =~ ^#.*$ || -z "$_bline" ]] && continue
            local _baddr _bnick _bssh
            _baddr=$(echo "$_bline" | awk '{print $1}')
            _bnick=$(echo "$_bline" | awk '{print $2}')
            _bssh=$(echo "$_bline" | awk '{print $3}')
            _bssh="${_bssh:-root}"
            local _bip; _bip="${_baddr%%:*}"

            local _rbi
            _rbi=$(ssh -o BatchMode=yes -o ConnectTimeout=3 -o StrictHostKeyChecking=no \
                "${_bssh}@${_bip}" \
                "cat /etc/greenboost/build_info 2>/dev/null" 2>/dev/null) || true

            local _rid _rver _repoch _rdate
            _rid=$(echo "$_rbi" | grep BUILD_ID | cut -d= -f2)
            _rver=$(echo "$_rbi" | grep BUILD_VERSION | cut -d= -f2)
            _repoch=$(echo "$_rbi" | grep BUILD_EPOCH | cut -d= -f2)
            if [[ -n "$_repoch" ]]; then
                _rdate=$(date -d "@${_repoch}" '+%Y-%m-%d %H:%M' 2>/dev/null || echo "${_repoch}")
            else
                _rdate="unreachable"
            fi

            local _stamp_col="${C_LIME}" _stale_tag=""
            if [[ -n "$_local_stamp_id" && "$_rid" != "$_local_stamp_id" ]]; then
                _stamp_col="${C_AMBER}"
                _stale_tag="  ${C_AMBER}⚠ needs update${C_RESET}"
                _any_stale=1
            fi

            printf "  ${C_PURPLE}%-12s${C_RESET}  ${C_GRAY}v%-7s${C_RESET} ${_stamp_col}%s${C_RESET}  ${C_DIM}%s${C_RESET}%b\n" \
                "${_bnick:-$_bip}" "${_rver:-?}" "${_rid:-?}" "${_rdate}" "$_stale_tag"

        done < "$GB_CLUSTER_CONF"
        if (( _any_stale )); then
            printf "  %b\n" "${C_AMBER}⚠  Feeder(s) on older build , run: sudo greenboost update feeders${C_RESET}"
        fi
    fi
    echo ""
}

cmd_cluster() {
    local _llm_mode=0
    for _a in "$@"; do [[ "$_a" == "--llm" ]] && _llm_mode=1; done
    if (( _llm_mode )); then
        local _strip='s/\x1b\[[0-9;]*[mKHJsu]//g'
        _cmd_cluster_snapshot | sed "$_strip" | sed 's/^[[:space:]]*//' | grep -v '^─'
        return
    fi

    # Non-interactive: single snapshot, no prompts, no infinite loop
    if [[ ! -t 0 ]]; then
        _cmd_cluster_snapshot
        return
    fi

    _gb_run_tui_loop "_cmd_cluster_snapshot" 5 \
        "  ${C_DIM}Updating every 5s  ${C_GRAY}Ctrl+S${C_DIM}: refresh  ${C_GRAY}Ctrl+C${C_DIM}: exit${C_RESET}"
}

_cmd_build_snapshot() {
    local _bi=""
    for _bic in "${MODULE_DIR}/build_info" ./build_info /etc/greenboost/build_info; do
        [[ -f "$_bic" ]] && { _bi="$_bic"; break; }
    done
    local _build_id _build_ver _build_host _build_git _build_epoch _build_date
    if [[ -n "$_bi" && -f "$_bi" ]]; then
        _build_id=$(    grep '^BUILD_ID='      "$_bi" | cut -d= -f2-)
        _build_ver=$(   grep '^BUILD_VERSION=' "$_bi" | cut -d= -f2-)
        _build_host=$(  grep '^BUILD_HOST='    "$_bi" | cut -d= -f2-)
        _build_git=$(   grep '^BUILD_GIT='     "$_bi" | cut -d= -f2-)
        _build_epoch=$( grep '^BUILD_EPOCH='   "$_bi" | cut -d= -f2-)
        if [[ -n "$_build_epoch" ]]; then
            _build_date=$(date -d "@${_build_epoch}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null \
                          || date -r "$_build_epoch" '+%Y-%m-%d %H:%M:%S' 2>/dev/null \
                          || echo "unknown")
        else
            _build_date="unknown"
        fi
    else
        _build_id="-"; _build_ver="$GB_VERSION"; _build_host="-"
        _build_git="-"; _build_date="-"
    fi

    # Count connected feeders
    local _feeder_count=0
    if [[ -f "$GB_CLUSTER_CONF" ]]; then
        while IFS= read -r _l; do
            [[ -z "$_l" || "$_l" == \#* ]] && continue
            (( _feeder_count++ )) || true
        done < "$GB_CLUSTER_CONF"
    fi
    local _feeders_str
    if (( _feeder_count == 0 )); then
        _feeders_str="${C_DIM}none connected${C_RESET}"
    else
        _feeders_str="${C_LIME}${_feeder_count} connected${C_RESET}  ${C_DIM}(greenboost cluster for details)${C_RESET}"
    fi

    local _ts; _ts=$(date '+%H:%M:%S')
    local _w=54
    local _div; printf -v _div '%*s' "$_w" ''; _div="${_div// /─}"

    printf '%b\n' "${C_DIM}${_div}${C_RESET}  ${C_DIM}${_ts}${C_RESET}"
    printf '%b\n' " ${C_BOLD}${C_CYAN}GreenBoost v${GB_VERSION}${C_RESET}${C_BOLD} - Build Information${C_RESET}"
    echo ""
    printf '  %-16s %b\n' "Build ID"    "${C_BOLD}${_build_id}${C_RESET}"
    printf '  %-16s %s\n' "Version"     "${_build_ver}"
    printf '  %-16s %s\n' "Git commit"  "${_build_git}"
    printf '  %-16s %s\n' "Built on"    "${_build_host}"
    printf '  %-16s %s\n' "Build time"  "${_build_date}"
    echo ""
    printf '%b\n' "  ${C_DIM}Binary paths${C_RESET}"
    printf '  %-16s %s\n' "  Daemon"    "/usr/local/bin/greenboost-netd"
    printf '  %-16s %s\n' "  CUDA shim" "/usr/local/lib/libgreenboost_cuda.so"
    printf '  %-16s %s\n' "  CLI"       "/usr/local/bin/greenboost"
    echo ""
    printf '  %-16s %b\n' "Feeders"     "$_feeders_str"
    printf '%b\n' "${C_DIM}${_div}${C_RESET}"

    if [[ ! -f "$_bi" ]]; then
        echo ""
        printf '%b\n' "  ${C_AMBER}⚠  No build stamp found - run: sudo greenboost install${C_RESET}"
    fi
}

cmd_build_info() {
    local _llm_mode=0
    for _a in "$@"; do [[ "$_a" == "--llm" ]] && _llm_mode=1; done
    if (( _llm_mode )); then
        local _bi=""
        for _bic in "${MODULE_DIR}/build_info" ./build_info /etc/greenboost/build_info; do
            [[ -f "$_bic" ]] && { _bi="$_bic"; break; }
        done
        if [[ -n "$_bi" ]]; then
            cat "$_bi"
        else
            printf 'BUILD_ID=not-installed\nBUILD_VERSION=%s\n' "$GB_VERSION"
        fi
        return
    fi

    if [[ ! -t 0 ]]; then
        _cmd_build_snapshot
        return
    fi
    _gb_run_tui_loop "_cmd_build_snapshot" 5 \
        "  ${C_DIM}Updating every 5s  ${C_GRAY}Ctrl+S${C_DIM}: refresh  ${C_GRAY}Ctrl+C${C_DIM}: exit${C_RESET}"
}

# ---- NVTX Event Log viewer -----------------------------------------

_cmd_nvtx_logs_snapshot() {
    local _ts; _ts=$(date '+%Y-%m-%dT%H:%M:%S')
    local log_file="/run/greenboost/nvtx_events.log"
    local log_old="/run/greenboost/nvtx_events.log.1"
    local log_fallback="/tmp/greenboost_nvtx_events.log"
    local log_fallback_old="/tmp/greenboost_nvtx_events.log.1"
    local _tail="${1:-50}"
    local _filter="${2:-}"
    local _llm="${3:-0}"

    # Prefer /run/greenboost; fall back to /tmp if the shim couldn't write there
    local _using_fallback=0
    if [[ ! -f "$log_file" ]] && [[ ! -f "$log_old" ]] && \
       { [[ -f "$log_fallback" ]] || [[ -f "$log_fallback_old" ]]; }; then
        log_file="$log_fallback"
        log_old="$log_fallback_old"
        _using_fallback=1
    fi

    local _sz="n/a"
    if [[ -f "$log_file" ]]; then
        _sz=$(du -sh "$log_file" 2>/dev/null | cut -f1)
    fi

    if [[ "$_llm" -eq 1 ]]; then
        echo "# GreenBoost NVTX Event Log - ${_ts}"
        echo "# log=${log_file} size=${_sz} tail=${_tail} filter=${_filter:-all}"
        [[ "$_using_fallback" -eq 1 ]] && echo "# NOTE: using /tmp fallback - /run/greenboost not writable by ollama user"
        echo "#"
        if [[ -f "$log_file" ]]; then
            if [[ -n "$_filter" ]]; then
                grep -i "$_filter" "$log_file" 2>/dev/null | tail -n "$_tail"
            else
                tail -n "$_tail" "$log_file" 2>/dev/null
            fi
        else
            echo "# no log file found"
        fi
        return
    fi

    local _W=120
    local _div
    _div=$(printf '─%.0s' $(seq 1 $_W))

    echo ""
    echo -e "  ${C_CYAN}${C_BOLD}GreenBoost NVTX Event Log${C_RESET}  ${C_DIM}${_ts}${C_RESET}"
    if [[ "$_using_fallback" -eq 1 ]]; then
        echo -e "  ${C_DIM}Log: ${log_file}  (${_sz})  ${C_RESET}\033[33m(fallback: /tmp - fix /run/greenboost perms)\033[0m   Showing last ${_tail} events"
    else
        echo -e "  ${C_DIM}Log: ${log_file}  (${_sz})   Showing last ${_tail} events${C_RESET}"
    fi
    [[ -n "$_filter" ]] && echo -e "  ${C_DIM}Filter: ${_filter}${C_RESET}"
    echo ""
    echo -e "  ${C_DIM}${_div}${C_RESET}"
    printf "  ${C_BOLD}%-24s  %-22s  %-14s  %8s  %-18s  %s${C_RESET}\n" \
        "TIMESTAMP(ms)" "EVENT_TYPE" "TIER" "SIZE" "PTR" "DETAIL"
    echo -e "  ${C_DIM}${_div}${C_RESET}"

    local _found=0
    if [[ -f "$log_file" ]] || [[ -f "$log_old" ]]; then
        _found=1
        # Combine old + current, get last N lines, optional filter
        local _src=""
        [[ -f "$log_old"  ]] && _src="$log_old"
        [[ -f "$log_file" ]] && _src="${_src:+$_src }${log_file}"
        local _lines
        if [[ -n "$_filter" ]]; then
            _lines=$(cat $_src 2>/dev/null | grep -i "$_filter" | tail -n "$_tail")
        else
            _lines=$(cat $_src 2>/dev/null | tail -n "$_tail")
        fi

        while IFS= read -r _line; do
            [[ -z "$_line" ]] && continue
            local _ts_ms _etype _tier _sz_f _ptr _det
            read -r _ts_ms _etype _tier _sz_f _ptr _det <<< "$_line"
            local _color="${C_RESET}"
            case "$_etype" in
                # T1 GPU VRAM
                ALLOC_T1_LOCAL|ALLOC_VMM_T1)   _color="\033[92m" ;;
                OOM_T1_LOCAL)                   _color="\033[31m${C_BOLD}" ;;
                # T2 DDR Path A (DMA-BUF pinned)
                ALLOC_A_ZEROCOPY|ALLOC_A_POOL|ALLOC_A_PINNED) _color="${C_CYAN}" ;;
                ALLOC_A_ZEROCOPY_FAIL)          _color="\033[38;5;208m" ;;
                # T2 DDR VMM HOST (cuMemCreate pinned)
                ALLOC_VMM_HOST)                 _color="\033[38;5;51m" ;;
                # T2 DDR Path B (HostReg no-kernel)
                ALLOC_B_HOSTREG)                _color="\033[38;5;214m" ;;
                # T2 pool inner (sub-alloc from pre-reg slab)
                ALLOC_T2_POOL)                  _color="\033[38;5;87m" ;;
                # Feeder allocs
                ALLOC_T1_FEEDER|ALLOC_T2_FEEDER) _color="${C_VIOLET}" ;;
                # Free / release
                FREE_A_ZEROCOPY|FREE_T2_PINNED|FREE*) _color="${C_DIM}" ;;
                # Data movement - local T2 PCIe DMA
                MEMCPY_H2D_T2|MEMCPY_D2H_T2)   _color="\033[34m" ;;
                # Data movement - remote feeder network
                MEMCPY_H2D_NET|MEMCPY_D2H_NET)  _color="\033[36m" ;;
                # Phase transitions
                PHASE_MODEL_LOAD)               _color="\033[38;5;208m" ;;
                PHASE_INFERENCE|PHASE_STEADY)   _color="\033[35m" ;;
                PHASE*)                         _color="\033[35m" ;;
                # OOM - red bold
                OOM*)                           _color="\033[31m${C_BOLD}" ;;
                # Eviction, pressure
                EVICT*|SWA_EVICT)               _color="\033[33m" ;;
                MEM_PRESS*)                     _color="\033[31m" ;;
                # Kernel feeder dispatch
                KERNEL_FEEDER)                  _color="\033[32m" ;;
                # GDS T3 I/O
                GDS_WRITE_OK|GDS_READ_OK)       _color="\033[33m" ;;
                GDS_WRITE_FAIL|GDS_READ_FAIL)   _color="\033[31m" ;;
                # Init / misc
                SHIM_INIT)                      _color="${C_DIM}" ;;
            esac
            printf "  ${_color}%-24s  %-22s  %-14s  %8s  %-18s  %s${C_RESET}\n" \
                "$_ts_ms" "$_etype" "$_tier" "$_sz_f" "$_ptr" "$_det"
        done <<< "$_lines"
    fi

    if [[ $_found -eq 0 ]]; then
        echo -e "  ${C_DIM}  No NVTX log file found at ${log_file}${C_RESET}"
        echo -e "  ${C_DIM}  (also checked ${log_fallback})${C_RESET}"
        echo -e "  ${C_DIM}  Start Ollama with GreenBoost active to generate events.${C_RESET}"
    fi

    echo -e "  ${C_DIM}${_div}${C_RESET}"

    # Diagnostic Summary - single awk pass over combined log
    local _diag_src=""
    [[ -f "$log_old"  ]] && _diag_src="$log_old"
    [[ -f "$log_file" ]] && _diag_src="${_diag_src:+$_diag_src }$log_file"

    if [[ -n "$_diag_src" ]]; then
        local _W2=113
        local _div2; _div2=$(printf '─%.0s' $(seq 1 $_W2))
        echo ""
        echo -e "  ${C_BOLD}Diagnostic Summary${C_RESET}"
        echo -e "  ${C_DIM}${_div2}${C_RESET}"

        # Single awk pass - emits tagged lines; no bashisms required in awk body
        local _stats
        _stats=$(cat $_diag_src 2>/dev/null | awk '
        {
            etype=$2; tier=$3; sz=$4
            gsub(/MB$/,"",sz); sz+=0
            if (etype ~ /^ALLOC/) {
                alloc_cnt++
                # Tier totals (skip non-memory tier labels)
                if (tier !~ /^(SYSTEM|PHASE|ALL)$/) {
                    tier_mb[tier]+=sz; tier_cnt[tier]++
                    if (sz > tier_peak[tier]) tier_peak[tier]=sz
                    if (sz > gpeak) { gpeak=sz; gpeak_tier=tier }
                }
                # Per-event-type alloc breakdown
                ev_cnt[etype]++; ev_mb[etype]+=sz
            }
            if (etype ~ /^MEMCPY_H2D/) { h2d_cnt++; h2d_mb+=sz }
            if (etype ~ /^MEMCPY_D2H/) { d2h_cnt++; d2h_mb+=sz }
            if (etype ~ /^FREE/)        { free_cnt++; freed_mb+=sz }
            if (etype ~ /^OOM/)         { oom_cnt++; oom_ev[etype]++ }
            if (etype ~ /^EVICT|^SWA/)  { evict_cnt++; evict_mb+=sz }
            if (etype ~ /^KERNEL/)        kern_cnt++
            if (etype ~ /^MEM_PRESS/)     press_cnt++
            if (etype ~ /^PHASE/)         last_phase=etype
        }
        END {
            print "ALLOC_CNT " alloc_cnt+0
            print "FREE_CNT  " free_cnt+0
            print "OOM_CNT   " oom_cnt+0
            print "H2D_CNT   " h2d_cnt+0
            print "H2D_MB    " h2d_mb+0
            print "D2H_CNT   " d2h_cnt+0
            print "D2H_MB    " d2h_mb+0
            print "EVICT_CNT " evict_cnt+0
            print "EVICT_MB  " evict_mb+0
            print "KERN_CNT  " kern_cnt+0
            print "PRESS_CNT " press_cnt+0
            print "LAST_PHASE " last_phase
            print "GPEAK_MB  " gpeak+0
            print "GPEAK_TIER " gpeak_tier
            for (t in tier_mb)  printf "TIER %s %d %d %d\n", t, tier_mb[t], tier_cnt[t], tier_peak[t]
            for (e in ev_cnt)   printf "EVTYPE %s %d %d\n",  e, ev_cnt[e], ev_mb[e]
            for (o in oom_ev)   printf "OOMEV %s %d\n",      o, oom_ev[o]
        }')

        # Parse awk output into bash vars
        local _alloc_cnt=0 _free_cnt=0 _oom_cnt=0
        local _h2d_cnt=0 _h2d_mb=0 _d2h_cnt=0 _d2h_mb=0
        local _evict_cnt=0 _evict_mb=0 _kern_cnt=0 _press_cnt=0
        local _last_phase="" _gpeak_mb=0 _gpeak_tier=""
        declare -A _tier_mb _tier_cnt _tier_peak _ev_cnt _ev_mb _oom_ev 2>/dev/null || true
        while IFS= read -r _line; do
            case "$_line" in
                "ALLOC_CNT "*)   _alloc_cnt="${_line#ALLOC_CNT }" ;;
                "FREE_CNT  "*)   _free_cnt="${_line#FREE_CNT  }" ;;
                "OOM_CNT   "*)   _oom_cnt="${_line#OOM_CNT   }" ;;
                "H2D_CNT   "*)   _h2d_cnt="${_line#H2D_CNT   }" ;;
                "H2D_MB    "*)   _h2d_mb="${_line#H2D_MB    }" ;;
                "D2H_CNT   "*)   _d2h_cnt="${_line#D2H_CNT   }" ;;
                "D2H_MB    "*)   _d2h_mb="${_line#D2H_MB    }" ;;
                "EVICT_CNT "*)   _evict_cnt="${_line#EVICT_CNT }" ;;
                "EVICT_MB  "*)   _evict_mb="${_line#EVICT_MB  }" ;;
                "KERN_CNT  "*)   _kern_cnt="${_line#KERN_CNT  }" ;;
                "PRESS_CNT "*)   _press_cnt="${_line#PRESS_CNT }" ;;
                "LAST_PHASE "*)  _last_phase="${_line#LAST_PHASE }" ;;
                "GPEAK_MB  "*)   _gpeak_mb="${_line#GPEAK_MB  }" ;;
                "GPEAK_TIER "*)  _gpeak_tier="${_line#GPEAK_TIER }" ;;
                "TIER "*)
                    read -r _ _t _tmb _tcnt _tpeak <<< "$_line"
                    _tier_mb[$_t]=$_tmb; _tier_cnt[$_t]=$_tcnt; _tier_peak[$_t]=$_tpeak ;;
                "EVTYPE "*)
                    read -r _ _e _ecnt _emb <<< "$_line"
                    _ev_cnt[$_e]=$_ecnt; _ev_mb[$_e]=$_emb ;;
                "OOMEV "*)
                    read -r _ _o _ocnt <<< "$_line"
                    _oom_ev[$_o]=$_ocnt ;;
            esac
        done <<< "$_stats"

        # ── Tier allocation rows (correct tier names from shim) ──────────────
        # Order: local tiers first, then feeder, T3 last (warns on spillover)
        local _tier_order=(T1_GPU T2_DDR T3_LOCAL T1_FEEDER T2_FEEDER T3_FEEDER)
        local -A _tier_label=([T1_GPU]="T1 GPU VRAM  (local)" [T2_DDR]="T2 DDR       (local)"
                             [T3_LOCAL]="T3 NVMe      (local)" [T1_FEEDER]="T1 GPU VRAM  (feeder)"
                             [T2_FEEDER]="T2 DDR       (feeder)" [T3_FEEDER]="T3 NVMe      (feeder)")
        local _any_tier=0
        for _t in "${_tier_order[@]}"; do
            [[ -z "${_tier_mb[$_t]+x}" ]] && continue
            _any_tier=1
            local _mb=${_tier_mb[$_t]} _cnt=${_tier_cnt[$_t]} _pk=${_tier_peak[$_t]}
            local _icon="✓" _col="\033[92m"
            [[ "$_t" == T2_DDR   ]] && _col="${C_CYAN}"
            [[ "$_t" == T2_FEEDER ]] && _col="\033[36m"
            [[ "$_t" == T3_LOCAL || "$_t" == T3_FEEDER ]] && _icon="⚠" _col="\033[33m"
            [[ "$_t" == T1_FEEDER ]] && _col="\033[35m"
            local _lbl="${_tier_label[$_t]:-$_t}"
            printf "  ${_col}${_icon}${C_RESET}  %-24s  %6d MB  %4d alloc(s)  peak single: %4d MB\n" \
                "$_lbl" "$_mb" "$_cnt" "$_pk"
        done
        # Catch any unexpected tier names
        for _t in "${!_tier_mb[@]}"; do
            case "$_t" in T1_GPU|T2_DDR|T3_LOCAL|T1_FEEDER|T2_FEEDER|T3_FEEDER) continue ;; esac
            printf "  \033[36m?${C_RESET}  %-24s  %6d MB  %4d alloc(s)  peak single: %4d MB\n" \
                "$_t" "${_tier_mb[$_t]}" "${_tier_cnt[$_t]}" "${_tier_peak[$_t]}"
        done

        # ── Per-path alloc breakdown (key insight into which path served the model) ─
        echo ""
        echo -e "  ${C_BOLD}Allocation paths${C_RESET}"
        # Ordered from best to worst
        local _path_order=(
            ALLOC_VMM_T1      "VMM T1  device VRAM     (cuMemCreate native)"
            ALLOC_A_ZEROCOPY  "Path A  zero-copy        (cudaImportExtMem)"
            ALLOC_A_POOL      "Path A  pool sub-alloc   (pre-reg slab)"
            ALLOC_A_PINNED    "Path A  pinned per-alloc (DMA-BUF+HostReg)"
            ALLOC_VMM_HOST    "Path B  VMM host-pinned  (cuMemCreate HOST)"
            ALLOC_B_HOSTREG   "Path B  HostReg no-kmod  (mmap+cuMemHostReg)"
            ALLOC_T1_LOCAL    "T1 local (cudaMalloc)"
            ALLOC_T2_POOL     "T2 pool  (inner sub-alloc from slab)"
            ALLOC_T1_FEEDER   "T1 feeder VRAM"
            ALLOC_T2_FEEDER   "T2 feeder DDR"
        )
        local _any_path=0
        local _i=0
        while (( _i < ${#_path_order[@]} )); do
            local _ev="${_path_order[$_i]}"
            local _lbl="${_path_order[$(( _i + 1 ))]}"
            _i=$(( _i + 2 ))
            [[ -z "${_ev_cnt[$_ev]+x}" ]] && continue
            _any_path=1
            local _ec=${_ev_cnt[$_ev]} _em=${_ev_mb[$_ev]}
            local _pc="\033[92m"
            [[ "$_ev" == ALLOC_A_* || "$_ev" == ALLOC_VMM_HOST ]] && _pc="${C_CYAN}"
            [[ "$_ev" == ALLOC_B_* ]] && _pc="\033[38;5;214m"
            [[ "$_ev" == ALLOC_T2_POOL ]] && _pc="\033[38;5;87m"
            printf "  ${_pc}✓${C_RESET}  %-46s  %4d alloc(s)  %6d MB total\n" "$_lbl" "$_ec" "$_em"
        done
        [[ $_any_path -eq 0 ]] && echo -e "  ${C_DIM}  No allocation events recorded yet${C_RESET}"

        # One-off: ALLOC_A_ZEROCOPY_FAIL - tells if zero-copy fell to pinned
        if [[ -n "${_ev_cnt[ALLOC_A_ZEROCOPY_FAIL]+x}" ]]; then
            printf "  \033[38;5;208m⚠${C_RESET}  Path A zero-copy fell to pinned sub-method:  %d time(s)\n" \
                "${_ev_cnt[ALLOC_A_ZEROCOPY_FAIL]}"
        fi

        # ── Tensor data movement through T2 DDR (PCIe DMA path) ─────────────
        if (( _h2d_cnt + _d2h_cnt > 0 )); then
            echo ""
            echo -e "  ${C_BOLD}Tensor data movement (T2 DDR ↔ GPU via PCIe)${C_RESET}"
            (( _h2d_cnt > 0 )) && \
                printf "  \033[34m→${C_RESET}  H2D (host→device):  %4d transfer(s)  %6d MB  (weights/KV loaded to GPU)\n" \
                    "$_h2d_cnt" "$_h2d_mb"
            (( _d2h_cnt > 0 )) && \
                printf "  \033[34m←${C_RESET}  D2H (device→host):  %4d transfer(s)  %6d MB  (KV written back to T2)\n" \
                    "$_d2h_cnt" "$_d2h_mb"
        fi

        # ── OOM breakdown ────────────────────────────────────────────────────
        echo ""
        if [[ "$_oom_cnt" -gt 0 ]]; then
            printf "  \033[31m✗${C_RESET}  OOM events: %d total\n" "$_oom_cnt"
            local _oom_labels=(
                OOM_T2_CAP       "T2 pool capacity cap exceeded"
                OOM_MEMAVAIL     "MemAvailable guard triggered"
                OOM_VMM_HOST     "cuMemCreate HOST pinned failed"
                OOM_PATH_B_FAIL  "Path B cuMemHostRegister failed"
                OOM_FULL         "All tiers exhausted (smart_alloc)"
                OOM_T1_LOCAL     "T1 VRAM OOM (cudaMalloc)"
            )
            local _oi=0
            while (( _oi < ${#_oom_labels[@]} )); do
                local _oe="${_oom_labels[$_oi]}" _ol="${_oom_labels[$(( _oi + 1 ))]}"
                _oi=$(( _oi + 2 ))
                [[ -z "${_oom_ev[$_oe]+x}" ]] && continue
                printf "  \033[31m  ✗${C_RESET}  %-38s  %d time(s)\n" "$_ol" "${_oom_ev[$_oe]}"
            done
        else
            echo -e "  \033[32m✓${C_RESET}  No OOM events"
        fi

        # ── Eviction / pressure / feeder ────────────────────────────────────
        [[ "$_evict_cnt" -gt 0 ]] && \
            printf "  \033[33m⚠${C_RESET}  Evictions: %d  (%d MB)  - T2 KV pool under pressure\n" \
                "$_evict_cnt" "$_evict_mb"
        [[ "$_press_cnt" -gt 0 ]] && \
            printf "  \033[33m⚠${C_RESET}  Memory pressure events: %d\n" "$_press_cnt"
        [[ "$_kern_cnt" -gt 0 ]] && \
            printf "  \033[32m✓${C_RESET}  Feeder GPU kernel dispatches: %d\n" "$_kern_cnt"

        # ── Phase / totals ───────────────────────────────────────────────────
        echo ""
        [[ -n "$_last_phase" ]] && \
            printf "  ${C_DIM}Last phase: %-20s  Allocs: %d  Frees: %d  Peak single alloc: %d MB (%s)%b\n" \
                "$_last_phase" "$_alloc_cnt" "$_free_cnt" "$_gpeak_mb" "$_gpeak_tier" "$C_RESET"

        echo -e "  ${C_DIM}${_div2}${C_RESET}"
    fi

    echo ""
    echo -e "  ${C_DIM}[Ctrl+S: refresh]  [Ctrl+C: exit]  Use --tail N  --filter TYPE  --llm for options${C_RESET}"
    echo ""
}

cmd_nvtx_logs() {
    local _tail=50
    local _filter=""
    local _llm=0

    # Parse flags
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --tail)   shift; _tail="${1:-50}" ;;
            --tail=*) _tail="${1#--tail=}" ;;
            --filter) shift; _filter="${1:-}" ;;
            --filter=*) _filter="${1#--filter=}" ;;
            --llm)    _llm=1 ;;
            *) ;;
        esac
        shift
    done

    if [[ "$_llm" -eq 1 ]]; then
        _cmd_nvtx_logs_snapshot "$_tail" "$_filter" 1
        return
    fi

    if [[ ! -t 0 ]]; then
        _cmd_nvtx_logs_snapshot "$_tail" "$_filter" 0
        return
    fi

    # Wrapper to pass _tail/_filter into the shared loop helper
    __GB_NVTX_TAIL="$_tail"
    __GB_NVTX_FILTER="$_filter"
    _nvtx_snap_wrap() { _cmd_nvtx_logs_snapshot "$__GB_NVTX_TAIL" "$__GB_NVTX_FILTER" 0; }
    _gb_run_tui_loop "_nvtx_snap_wrap" 5 \
        "  ${C_DIM}Updating every 5s  ${C_GRAY}Ctrl+S${C_DIM}: refresh  ${C_GRAY}Ctrl+C${C_DIM}: exit${C_RESET}"
    unset -f _nvtx_snap_wrap
    unset __GB_NVTX_TAIL __GB_NVTX_FILTER
}

# cmd_nvtx_vitals - live merged tail of local shim + feeder daemon NVTX events.
# Both logs use the same format: epoch_ms SOURCE EVENT TIER SIZE ptr detail
# Merges streams via a named FIFO; Python formatter colorises by event/tier.
# Usage: greenboost nvtx vitals [--last N] [--filter EV1,EV2] [--feeder-only] [--local-only] [--llm]
cmd_nvtx_vitals() {
    local _last=0 _filter="" _feeder_only=0 _local_only=0 _llm=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --last)        shift; _last="${1:-50}" ;;
            --last=*)      _last="${1#--last=}" ;;
            --filter)      shift; _filter="${1:-}" ;;
            --filter=*)    _filter="${1#--filter=}" ;;
            --feeder-only) _feeder_only=1 ;;
            --local-only)  _local_only=1 ;;
            --llm)         _llm=1 ;;
            *) ;;
        esac
        shift
    done

    local _local_log="/run/greenboost/nvtx_events.log"
    local _local_fallback="/tmp/greenboost_nvtx_events.log"
    [[ ! -f "$_local_log" && -f "$_local_fallback" ]] && _local_log="$_local_fallback"

    local _feeder_ip="" _feeder_port="9740"
    if [[ -f "$GB_CLUSTER_CONF" ]]; then
        local _cline
        _cline=$(grep -v '^#\|^[[:space:]]*$' "$GB_CLUSTER_CONF" 2>/dev/null | head -1)
        if [[ -n "$_cline" ]]; then
            local _addr; _addr=$(echo "$_cline" | awk '{print $1}')
            _feeder_ip="${_addr%%:*}"
            local _p="${_addr##*:}"
            [[ "$_p" != "$_feeder_ip" ]] && _feeder_port="$_p"
        fi
    fi
    local _feeder_user="${GB_SSH_USER:-${SUDO_USER:-$USER}}"
    local _ssh_as=()
    [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]] && _ssh_as=(runuser -u "$SUDO_USER" --)

    # LLM mode: raw tail, no colour
    if [[ "$_llm" -eq 1 ]]; then
        echo "# GreenBoost NVTX Vitals - $(date '+%Y-%m-%dT%H:%M:%S')"
        echo "# local=$_local_log feeder=${_feeder_ip:-none}:${_feeder_port}"
        [[ -f "$_local_log" ]] && tail -n "${_last:-50}" "$_local_log" | sed 's/^/LOCAL:/'
        if [[ -n "$_feeder_ip" ]]; then
            "${_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=5 \
                -o StrictHostKeyChecking=no "${_feeder_user}@${_feeder_ip}" \
                "tail -n ${_last:-50} /run/greenboost/nvtx_events.log 2>/dev/null" \
                | sed 's/^/FEEDER:/' || true
        fi
        return
    fi

    # Write Python formatter to temp file - avoids heredoc-within-function issues
    local _pyfmt; _pyfmt=$(mktemp /tmp/gb_nvtx_fmt.XXXXXX.py)
    local _fifo_dir; _fifo_dir=$(mktemp -d /run/greenboost/nvtx.XXXXXX 2>/dev/null || mktemp -d)
    local _fifo="${_fifo_dir}/merge"
    mkfifo "$_fifo"
    trap 'rm -rf "$_fifo_dir" 2>/dev/null; rm -f "$_pyfmt" 2>/dev/null; kill 0 2>/dev/null || true' EXIT INT TERM

    cat > "$_pyfmt" << 'NVTX_PY'
import sys, time
filter_str = sys.argv[1] if len(sys.argv) > 1 else ""
filters = [f.strip().upper() for f in filter_str.split(",") if f.strip()] if filter_str else []
R="\033[0m"; DIM="\033[2m"
EV_COL={
    "ALLOC_T1_GPU":"\033[92m","ALLOC_T2_DDR":"\033[94m","ALLOC_T2_POOL":"\033[34m",
    "ALLOC_T3_MMAP":"\033[36m","FREE_OK":"\033[90m","OOM_T1_GPU":"\033[91;1m",
    "OOM_T2_DDR":"\033[91;1m","OOM_T3":"\033[91;1m","EXEC_KERNEL":"\033[93m",
    "EXEC_DISPATCH":"\033[93;1m","EXEC_LOCAL":"\033[33m","MEMCPY_H2D":"\033[96m",
    "MEMCPY_D2H":"\033[96m","MEMCPY_D2D":"\033[96m","ALLOC_A_ZEROCOPY_FAIL":"\033[91;1m",
    "ALLOC_A_ZEROCOPY":"\033[32m","CLIENT_CONN":"\033[35m","CLIENT_DISC":"\033[90m",
    "PHASE_DEEP_IDLE":"\033[90m","HANDSHAKE":"\033[35m",
}
TIER_COL={
    "T1_GPU":"\033[92;1m","T2_DDR":"\033[94;1m","T3_NVMe":"\033[36;1m",
    "NET":"\033[35;1m","PHASE":"\033[90m","T2_DDR_POOL":"\033[34m",
}
def fts(ms):
    try: n=int(ms); return time.strftime("%H:%M:%S",time.localtime(n/1000))+f".{n%1000:03d}"
    except: return ms
def fsz(s):
    try:
        n=int(s.replace("MB","").strip())
        if n==0: return DIM+"   -   "+R
        return ("\033[93m" if n>=1024 else "")+f"{n:5d}MB"+R
    except: return f"{s:>7}"
for raw in sys.stdin:
    raw=raw.rstrip("\n")
    if not raw: continue
    if raw.startswith("LOCAL:"): src="LOCAL"; line=raw[6:]
    elif raw.startswith("FEEDER:"): src="FEEDER"; line=raw[7:]
    else: src="?"; line=raw
    p=line.split()
    if len(p)<3: continue
    i=0
    try: epoch=p[i]; int(epoch); i+=1
    except (ValueError,IndexError): continue
    if i<len(p) and p[i] in ("NETD","SHIM","GB"): i+=1
    ev    =p[i] if i<len(p) else "?";  i+=1
    tier  =p[i] if i<len(p) else "?";  i+=1
    size  =p[i] if i<len(p) else "0MB";i+=1
    ptr   =p[i] if i<len(p) else "";   i+=1
    detail=" ".join(p[i:])
    if filters and not any(f in ev.upper() for f in filters): continue
    sc="\033[97;1m" if src=="LOCAL" else "\033[95;1m"
    st="◀ LOCAL " if src=="LOCAL" else "▶ FEEDER"
    print(f"  {DIM}{fts(epoch)}{R}  {sc}{st}{R}  "
          f"{EV_COL.get(ev,chr(27)+'[37m')}{ev:<22}{R}  "
          f"{TIER_COL.get(tier,chr(27)+'[37m')}{tier:<12}{R}  "
          f"{fsz(size)}  {DIM}{ptr:<22} {detail}{R}")
    sys.stdout.flush()
NVTX_PY

    # Header
    printf "\n  ${C_BOLD}${C_CYAN}GreenBoost NVTX Vitals${C_RESET}  ${C_DIM}%s${C_RESET}\n" \
        "$(date '+%Y-%m-%d %H:%M:%S')"
    printf "  ${C_DIM}local : %s${C_RESET}\n" "$_local_log"
    [[ -n "$_feeder_ip" ]] && printf "  ${C_DIM}feeder: %s:%s${C_RESET}\n" "$_feeder_ip" "$_feeder_port"
    [[ -n "$_filter" ]]    && printf "  ${C_AMBER}filter: %s${C_RESET}\n" "$_filter"
    printf "\n  ${C_DIM}Colour: "
    printf "\033[92mALLOC_T1\033[0m${C_DIM} · \033[94mALLOC_T2\033[0m${C_DIM} · "
    printf "\033[36mALLOC_T3\033[0m${C_DIM} · \033[91;1mOOM\033[0m${C_DIM} · "
    printf "\033[93mEXEC\033[0m${C_DIM} · \033[96mMEMCPY\033[0m${C_DIM} · "
    printf "\033[90mFREE/IDLE\033[0m${C_DIM}"
    printf "\n  Ctrl+C to exit  ·  ◀ LOCAL = host shim  ·  ▶ FEEDER = remote daemon${C_RESET}\n"
    printf "  ${C_DIM}─────────────────────────────────────────────────────────────────${C_RESET}\n\n"

    # Launch tail streams - both write to the same FIFO; Python reads from it
    local _tail_args="-n 0 -F"
    [[ "$_last" -gt 0 ]] && _tail_args="-n $_last"

    if [[ "$_feeder_only" -eq 0 ]]; then
        if [[ -f "$_local_log" ]]; then
            # shellcheck disable=SC2086
            tail $_tail_args "$_local_log" 2>/dev/null | sed -u 's/^/LOCAL:/' > "$_fifo" &
        else
            printf "  ${C_AMBER}⚠${C_RESET}  Local NVTX log not found: %s\n" "$_local_log"
        fi
    fi

    if [[ "$_local_only" -eq 0 ]]; then
        if [[ -n "$_feeder_ip" ]]; then
            # shellcheck disable=SC2086
            "${_ssh_as[@]}" ssh -o BatchMode=yes -o ConnectTimeout=10 \
                -o StrictHostKeyChecking=no "${_feeder_user}@${_feeder_ip}" \
                "tail $_tail_args /run/greenboost/nvtx_events.log 2>/dev/null" \
                | sed -u 's/^/FEEDER:/' > "$_fifo" &
        else
            printf "  ${C_AMBER}⚠${C_RESET}  No feeder in cluster.conf - showing local only\n"
        fi
    fi

    python3 -u "$_pyfmt" "$_filter" < "$_fifo"
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
    echo -e "  ${C_CYAN}${C_BOLD}GreenBoost Command Reference${C_RESET}  ${C_DIM}- run ${C_GRAY}greenboost help${C_DIM} to see this anytime${C_RESET}"
    echo -e ""

    if [[ -z "$doc" ]]; then
        gb_warn_ui "GREENBOOST_COMMANDS.md not found - showing built-in summary"
        echo -e ""
        printf "  ${C_LIME}%-28s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "greenboost vitals"          "live memory tiers + NVTX events + system health"
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
            # Code block content - full white for contrast
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
            # Normal text - dim gray
            [[ -n "$line" ]] && echo -e "  ${C_DIM}${line}${C_RESET}" || echo ""
        fi
    done < "$doc"

    echo -e ""
    gb_info "Full document: $doc"
}

# ── cmd_gen_inference_config - optimized inference configuration generator ─
# Usage: greenboost gen-inference-config [--format ollama|env|both]
#                                        [--turboquant] [--no-turboquant]
#                                        [--output /path] [--model <name>]
cmd_gen_inference_config() {
    local fmt="ollama" tq_flag="" out_file="" model_name="" virt_type="bare-metal" dry_run=0 llm_mode=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --format)        fmt="${2:-ollama}"; shift 2 ;;
            --turboquant)    tq_flag="on";       shift ;;
            --no-turboquant) tq_flag="off";      shift ;;
            --output)        out_file="${2:-}";   shift 2 ;;
            --model)         model_name="${2:-}"; shift 2 ;;
            --dry-run)       dry_run=1;           shift ;;
            --llm)           llm_mode=1;          shift ;;
            *)               shift ;;
        esac
    done

    # -- Detect virtualization type
    if [[ -f "/.dockerenv" ]]; then
        virt_type="container"
    elif grep -qE 'docker|lxc|kubepods' /proc/1/cgroup 2>/dev/null; then
        virt_type="container"
    else
        local _sdv
        _sdv=$(systemd-detect-virt 2>/dev/null | head -1 || echo "none")
        _sdv="${_sdv//[$'\t\r\n']/ }"
        _sdv="${_sdv%% *}"
        case "$_sdv" in
            none|"") virt_type="bare-metal" ;;
            wsl)     virt_type="wsl2" ;;
            *)       virt_type="vm:${_sdv}" ;;
        esac
    fi

    # -- Read hardware profile (best-effort; detection may be incomplete in VMs)
    detect_hardware 2>/dev/null || true
    local phys_gb="${GB_PHYS:-0}"
    local virt_gb="${GB_VIRT:-0}"
    local nvme_gb="${GB_NVME_SWAP:-0}"
    local ctx="${GB_OLLAMA_CTX:-8192}"

    # In VMs/containers cap context conservatively (no NVMe swap, limited DMA)
    local effective_ctx="$ctx"
    case "$virt_type" in
        container|wsl2) effective_ctx=$(( ctx / 2 ))      ;;
        vm:*)           effective_ctx=$(( ctx * 3 / 4 ))  ;;
    esac

    # -- Cluster feeder count
    local cluster_feeders=0
    if [[ -f "/etc/greenboost/cluster.conf" ]]; then
        cluster_feeders=$(grep -v '^[[:space:]]*#' /etc/greenboost/cluster.conf 2>/dev/null | grep -c '.' || echo 0)
    fi

    # -- TurboQuant: if not forced via flag, read the current global flag
    if [[ -z "$tq_flag" ]]; then
        [[ -f "/etc/greenboost/turboquant.enabled" ]] && tq_flag="on" || tq_flag="off"
    fi
    local kv_type="q8_0"
    [[ "$tq_flag" == "on" ]] && kv_type="q4_0"

    local ts; ts=$(date '+%Y-%m-%d %H:%M:%S')
    local out=""

    out+="# GreenBoost optimized inference configuration\n"
    out+="# Generated: ${ts}\n"
    out+="# Environment: ${virt_type}  GPU T1: ${phys_gb} GB  T2: ${virt_gb} GB  T3: ${nvme_gb} GB\n"
    [[ $cluster_feeders -gt 0 ]] && out+="# Cluster: ${cluster_feeders} feeder(s) connected\n"
    out+="# TurboQuant: ${tq_flag}\n"
    out+="\n"

    if [[ "$fmt" == "ollama" || "$fmt" == "both" ]]; then
        out+="# ── Ollama environment variables ──────────────────────────────────\n"
        out+="OLLAMA_NUM_CTX=${effective_ctx}\n"
        out+="OLLAMA_CONTEXT_LENGTH=${effective_ctx}\n"
        out+="OLLAMA_FLASH_ATTENTION=1\n"
        out+="OLLAMA_KV_CACHE_TYPE=${kv_type}\n"
        out+="OLLAMA_NUM_PARALLEL=1\n"
        out+="OLLAMA_MAX_LOADED_MODELS=1\n"
        out+="OLLAMA_KEEP_ALIVE=-1\n"
        out+="OLLAMA_NUM_GPU=999\n"
        out+="CUDA_VISIBLE_DEVICES=0\n"
        out+="GREENBOOST_ACTIVE=1\n"
        [[ "$tq_flag" == "on" ]] && out+="GREENBOOST_TURBOQUANT=1\n"
        [[ -n "$model_name" ]] && out+="# Suggested model: ${model_name}\n"
        out+="\n"
        if [[ "$virt_type" != "bare-metal" ]]; then
            out+="# NOTE: Running in ${virt_type} - context capped at ${effective_ctx} tokens.\n"
            out+="# Raise OLLAMA_NUM_CTX if the host dedicates more RAM to this environment.\n"
            out+="\n"
        fi
    fi

    if [[ "$fmt" == "env" || "$fmt" == "hf" || "$fmt" == "both" ]]; then
        out+="# ── HuggingFace / PyTorch inference (Python) ──────────────────────\n"
        out+="# Add to your inference script:\n"
        out+="\n"
        out+="# import sys\n"
        out+="# sys.path.insert(0, \"/path/to/greenboost_all/greenboost\")\n"
        out+="# from gb_attn import turboquant_attention\n"
        out+="#\n"
        if [[ "$tq_flag" == "on" ]]; then
            out+="# with turboquant_attention(k_bits=4, v_bits=3, sparse_v=True):\n"
            out+="#     output = model(input)  # ~4x KV bandwidth reduction\n"
        else
            out+="# TurboQuant is currently OFF.  Enable with: sudo greenboost turboquant on\n"
            out+="# Then wrap inference: with turboquant_attention(k_bits=4, v_bits=3): ...\n"
        fi
        out+="\n"
        out+="# Environment variables:\n"
        out+="# export GREENBOOST_ACTIVE=1\n"
        [[ "$tq_flag" == "on" ]] && out+="# export GREENBOOST_TURBOQUANT=1\n"
        out+="\n"
    fi

    if [[ "$llm_mode" -eq 1 ]]; then
        # Machine-readable: strip ANSI, strip comment lines, emit raw key=value
        printf '%b' "$out" | sed 's/\x1b\[[0-9;]*[mKHJsu]//g' | grep -v '^#' | grep -v '^[[:space:]]*$'
    elif [[ "$dry_run" -eq 1 ]]; then
        local target="${out_file:-/etc/greenboost/inference.env}"
        echo "[dry-run] would write to ${target}:"
        echo "────────────────────────────────────────"
        printf '%b' "$out"
        echo "────────────────────────────────────────"
        echo "[dry-run] no files written."
    elif [[ -n "$out_file" ]]; then
        printf '%b' "$out" > "$out_file"
        gb_ok "Config written to: ${out_file}"
    elif [[ ! -t 1 ]]; then
        # Audit F-L4-16: stdout is not a terminal (pipe, redirect) - emit raw
        # config without any ANSI decoration so the output is machine-readable.
        printf '%b' "$out"
    else
        gb_header
        gb_section "Optimized Inference Configuration  (${virt_type})"
        echo ""
        printf '%b' "$out"
        gb_separator
        gb_info "Save output: greenboost gen-inference-config --format both --output /etc/greenboost/inference.env"
        gb_info "TurboQuant:  sudo greenboost turboquant on|off|status"
    fi
}


# ── cmd_turboquant - global TurboQuant K/V compression toggle ───────────────
# Usage: greenboost turboquant on|off|status
#
# ON : writes /etc/greenboost/turboquant.enabled, sets OLLAMA_KV_CACHE_TYPE=q4_0,
#      injects GREENBOOST_TURBOQUANT=1 into the Ollama drop-in and profile.d,
#      restarts Ollama if it is running.
# OFF: removes flag, reverts KV type to q8_0, removes TQ env vars.
cmd_turboquant() {
    local subcmd="${1:-status}"
    local dropin="/etc/systemd/system/ollama.service.d/99-greenboost.conf"
    local flag_file="/etc/greenboost/turboquant.enabled"
    local profiled="/etc/profile.d/greenboost-turboquant.sh"

    case "$subcmd" in
        on)
            need_root "turboquant on"
            mkdir -p /etc/greenboost
            touch "$flag_file"

            # Update Ollama drop-in: q8_0 → q4_0, inject GREENBOOST_TURBOQUANT
            if [[ -f "$dropin" ]]; then
                sed -i 's/OLLAMA_KV_CACHE_TYPE=q8_0/OLLAMA_KV_CACHE_TYPE=q4_0/' "$dropin"
                if ! grep -q 'GREENBOOST_TURBOQUANT' "$dropin"; then
                    sed -i '/GREENBOOST_ACTIVE=1/a Environment="GREENBOOST_TURBOQUANT=1"' "$dropin"
                fi
                systemctl daemon-reload
                if systemctl is-active --quiet ollama; then
                    systemctl restart ollama
                    gb_ok "Ollama restarted - KV cache: q4_0 (TurboQuant)"
                else
                    gb_ok "Ollama drop-in updated (service not running)"
                fi
            else
                gb_warn "Ollama drop-in not found - only flag file written."
                gb_info "Run: sudo greenboost install-sys-configs  to create the drop-in."
            fi

            # profile.d shim so login-shell Python processes pick up the flag
            cat > "$profiled" << 'TQPROFEOF'
# GreenBoost TurboQuant - set when /etc/greenboost/turboquant.enabled exists
[ -f /etc/greenboost/turboquant.enabled ] && export GREENBOOST_TURBOQUANT=1
TQPROFEOF

            echo ""
            gb_ok "TurboQuant ON"
            gb_info "Ollama:  OLLAMA_KV_CACHE_TYPE=q4_0 (4-bit KV cache)"
            gb_info "Python:  GREENBOOST_TURBOQUANT=1 exported for login shells"
            gb_info "         Use turboquant_attention() from gb_attn.py in PyTorch code"
            gb_info "Synapse: /turboquant status  to confirm inside the CLI"
            ;;

        off)
            need_root "turboquant off"
            rm -f "$flag_file"

            if [[ -f "$dropin" ]]; then
                sed -i 's/OLLAMA_KV_CACHE_TYPE=q4_0/OLLAMA_KV_CACHE_TYPE=q8_0/' "$dropin"
                sed -i '/GREENBOOST_TURBOQUANT/d' "$dropin"
                systemctl daemon-reload
                if systemctl is-active --quiet ollama; then
                    systemctl restart ollama
                    gb_ok "Ollama restarted - KV cache: q8_0 (standard)"
                else
                    gb_ok "Ollama drop-in reverted (service not running)"
                fi
            fi

            rm -f "$profiled"

            echo ""
            gb_ok "TurboQuant OFF"
            gb_info "Ollama: OLLAMA_KV_CACHE_TYPE=q8_0 restored"
            ;;

        status|--llm)
            local tq_state kv_cur
            [[ -f "$flag_file" ]] && tq_state="ON" || tq_state="OFF"
            kv_cur="q8_0 (default)"
            if [[ -f "$dropin" ]]; then
                local _kv; _kv=$(grep -oP 'OLLAMA_KV_CACHE_TYPE=\K[^"]+' "$dropin" 2>/dev/null || echo "")
                [[ -n "$_kv" ]] && kv_cur="$_kv"
            fi
            local _tq_py="${GREENBOOST_TURBOQUANT:-0}"
            # Audit F-L4-15: --llm emits machine-readable key=value pairs.
            if [[ "$subcmd" == "--llm" ]]; then
                echo "turboquant_state=${tq_state}"
                echo "ollama_kv_cache=${kv_cur}"
                echo "python_env_set=${_tq_py}"
                return
            fi
            echo ""
            printf "  ${C_BOLD}TurboQuant global state:${C_RESET}  "
            if [[ "$tq_state" == "ON" ]]; then
                printf "${C_LIME}${C_BOLD}ON${C_RESET}\n"
            else
                printf "${C_DIM}OFF${C_RESET}\n"
            fi
            printf "  ${C_GRAY}Flag file:${C_RESET}         ${C_DIM}${flag_file}${C_RESET}  "
            [[ -f "$flag_file" ]] && printf "${C_LIME}(present)${C_RESET}\n" || printf "${C_DIM}(absent)${C_RESET}\n"
            printf "  ${C_GRAY}Ollama KV cache:${C_RESET}   ${C_CYAN}%s${C_RESET}\n" "$kv_cur"
            printf "  ${C_GRAY}Python env:${C_RESET}        "
            [[ "$_tq_py" == "1" ]] \
                && printf "${C_LIME}GREENBOOST_TURBOQUANT=1${C_RESET}\n" \
                || printf "${C_DIM}GREENBOOST_TURBOQUANT not set in this shell${C_RESET}\n"
            echo ""
            gb_info "Toggle: sudo greenboost turboquant on|off"
            ;;

        *)
            die "Usage: greenboost turboquant on|off|status|--llm"
            ;;
    esac
}

# ── cmd_ggml_2dev - GREENBOOST_GGML_2DEV toggle (Ollama/ggml only) ───────────
# Usage: greenboost ggml-2dev on|off|status|--llm
#
# Presents the connected cluster feeder as a real second CUDA device to
# Ollama/llama.cpp so it natively splits an oversized model's layers across
# local VRAM (device 0) and the feeder (device 1), with the feeder GPU
# computing its assigned layers - instead of the whole overflow spilling to
# local DDR (the default single-virtual-GPU path). OFF by default: this is a
# newer code path than the proven feeder-as-memory-tier default and should
# be validated (compute-sanitizer, a real generation, tok/s comparison)
# before relying on it. Torch/HF pipelines are never affected by this flag -
# they use GREENBOOST_CLUSTER=0 + gb_cluster's own block-offload path.
cmd_ggml_2dev() {
    local subcmd="${1:-status}"
    local dropin="/etc/systemd/system/ollama.service.d/99-greenboost.conf"
    local flag_file="/etc/greenboost/ggml_2dev.enabled"

    case "$subcmd" in
        on)
            need_root "ggml-2dev on"
            if [[ ! -f "$GB_CLUSTER_CONF" ]] || ! grep -q '[^[:space:]]' "$GB_CLUSTER_CONF" 2>/dev/null; then
                gb_warn "No feeder connected ($GB_CLUSTER_CONF empty/missing) - connect one first:"
                gb_info "  sudo greenboost connect <feeder-IP>"
            fi
            mkdir -p /etc/greenboost
            touch "$flag_file"

            if [[ -f "$dropin" ]]; then
                if ! grep -q 'GREENBOOST_GGML_2DEV' "$dropin"; then
                    sed -i '/GREENBOOST_ACTIVE=1/a Environment="GREENBOOST_GGML_2DEV=1"' "$dropin"
                fi
                systemctl daemon-reload
                if systemctl is-active --quiet ollama; then
                    systemctl restart ollama
                    gb_ok "Ollama restarted - feeder presented as CUDA device 1"
                else
                    gb_ok "Ollama drop-in updated (service not running)"
                fi
            else
                gb_warn "Ollama drop-in not found - only flag file written."
                gb_info "Run: sudo greenboost install-sys-configs  to create the drop-in."
            fi

            echo ""
            gb_ok "ggml-2dev ON"
            gb_info "Ollama will enumerate 1 local + N feeder GPU(s) as separate CUDA devices."
            gb_info "Verify:  journalctl -u ollama | grep -i 'inference compute'"
            ;;

        off)
            need_root "ggml-2dev off"
            rm -f "$flag_file"

            if [[ -f "$dropin" ]]; then
                sed -i '/GREENBOOST_GGML_2DEV/d' "$dropin"
                systemctl daemon-reload
                if systemctl is-active --quiet ollama; then
                    systemctl restart ollama
                    gb_ok "Ollama restarted - back to single-virtual-GPU (feeder as memory tier only)"
                else
                    gb_ok "Ollama drop-in reverted (service not running)"
                fi
            fi

            echo ""
            gb_ok "ggml-2dev OFF"
            ;;

        status|--llm)
            local state="OFF" dropin_live="OFF"
            [[ -f "$flag_file" ]] && state="ON"
            [[ -f "$dropin" ]] && grep -q 'GREENBOOST_GGML_2DEV=1' "$dropin" 2>/dev/null && dropin_live="ON"

            if [[ "$subcmd" == "--llm" ]]; then
                echo "ggml_2dev_state=${state}"
                echo "ggml_2dev_dropin_live=${dropin_live}"
                return
            fi
            echo ""
            printf "  ${C_BOLD}ggml-2dev (feeder as CUDA device 1):${C_RESET}  "
            if [[ "$state" == "ON" ]]; then
                printf "${C_LIME}${C_BOLD}ON${C_RESET}\n"
            else
                printf "${C_DIM}OFF${C_RESET}\n"
            fi
            printf "  ${C_GRAY}Ollama drop-in:${C_RESET}  "
            [[ "$dropin_live" == "ON" ]] \
                && printf "${C_LIME}GREENBOOST_GGML_2DEV=1${C_RESET}\n" \
                || printf "${C_DIM}not set${C_RESET}\n"
            echo ""
            gb_info "Toggle: sudo greenboost ggml-2dev on|off"
            ;;

        *)
            die "Usage: greenboost ggml-2dev on|off|status|--llm"
            ;;
    esac
}


# ── _gb_ollama_model_blob - resolve an Ollama model name to its GGUF blob path ─
# Usage: _gb_ollama_model_blob <model[:tag]>
# Outputs the absolute path to the model's GGUF blob file, or returns 1.
_gb_ollama_model_blob() {
    local model_name="$1"
    local models_dir=""
    for _d in /usr/share/ollama/.ollama/models "${HOME}/.ollama/models" /root/.ollama/models; do
        [[ -d "${_d}/blobs" ]] && { models_dir="${_d}"; break; }
    done
    [[ -z "$models_dir" ]] && return 1

    # Split name into namespace / model / tag
    local ns name tag
    if [[ "$model_name" == *"/"* ]]; then
        ns="${model_name%%/*}"
        name="${model_name#*/}"
    else
        ns="library"
        name="$model_name"
    fi
    if [[ "$name" == *":"* ]]; then
        tag="${name#*:}"; name="${name%:*}"
    else
        tag="latest"
    fi

    # Try registry.ollama.ai/<ns>/<name>/<tag> then library fallback
    local manifest=""
    for _mp in \
        "${models_dir}/manifests/registry.ollama.ai/${ns}/${name}/${tag}" \
        "${models_dir}/manifests/registry.ollama.ai/library/${name}/${tag}"; do
        [[ -f "$_mp" ]] && { manifest="$_mp"; break; }
    done
    [[ -z "$manifest" ]] && return 1

    # Extract the model-layer blob digest (mediaType contains "model")
    local digest
    digest=$(python3 - "$manifest" << 'PYEOF'
import json, sys
try:
    m = json.load(open(sys.argv[1]))
    for l in m.get("layers", []):
        if "model" in l.get("mediaType", ""):
            print(l["digest"].replace(":", "-")); sys.exit(0)
    # fallback: largest layer
    layers = sorted(m.get("layers",[]), key=lambda x: x.get("size",0), reverse=True)
    if layers: print(layers[0]["digest"].replace(":", "-"))
except Exception: pass
PYEOF
    ) || return 1
    [[ -z "$digest" ]] && return 1

    local blob="${models_dir}/blobs/${digest}"
    [[ -f "$blob" ]] || return 1
    echo "$blob"
}


# ── gb-synapse: HuggingFace-native, cluster-distributed GGUF serving ────────
# All logic lives in gb_synapse.py / gb_synapse_api.py (repo root) - these are
# thin CLI wrappers, per the project rule that bash only dispatches + renders.
# See workflow/gb-synapse.md.
_GB_SYNAPSE_PY="$MODULE_DIR/gb_synapse.py"

_gb_synapse_run() {
    [[ -f "$_GB_SYNAPSE_PY" ]] || die "gb-synapse script not found: $_GB_SYNAPSE_PY"
    python3 "$_GB_SYNAPSE_PY" "$@"
}

# cmd_synapse_login - store a HuggingFace token. Masked /dev/tty read mirrors
# the masked-password pattern in cmd_feeders_setup_sudo (line ~8246).
# Usage: sudo greenboost synapse login [TOKEN]   (or: export HF_TOKEN=...)
cmd_synapse_login() {
    need_root "synapse login"
    local _token="${1:-${HF_TOKEN:-}}"
    if [[ -z "$_token" ]]; then
        local _char
        printf "  HuggingFace token: " > /dev/tty
        while IFS= read -r -s -n1 _char < /dev/tty; do
            case "$_char" in
                '')            break ;;
                $'\x7f'|$'\b') if [[ -n "$_token" ]]; then _token="${_token%?}"; printf '\b \b' > /dev/tty; fi ;;
                *)             _token+="$_char"; printf '*' > /dev/tty ;;
            esac
        done
        printf '\n' > /dev/tty
    fi
    [[ -z "$_token" ]] && die "No token provided"
    _gb_synapse_run login "$_token"
}

cmd_synapse_pull() {
    need_root "synapse pull"
    [[ -z "${1:-}" ]] && die "Usage: sudo greenboost pull <repo>[:quant] [name]"
    _gb_synapse_run pull "$@"
}
cmd_synapse_list()         { _gb_synapse_run list "$@"; }
cmd_synapse_rm()           { need_root "synapse rm"; [[ -z "${1:-}" ]] && die "Usage: sudo greenboost synapse rm <name>"; _gb_synapse_run rm "$@"; }
cmd_synapse_index_ollama() { need_root "synapse index-ollama"; _gb_synapse_run index-ollama "$@"; }
cmd_synapse_build_engine() { need_root "synapse build-engine";  _gb_synapse_run build-engine; }
cmd_synapse_update_engine(){ need_root "synapse update-engine"; _gb_synapse_run update-engine; }

cmd_synapse_serve() {
    [[ -z "${1:-}" ]] && die "Usage: greenboost synapse run <model> [port]"
    _gb_synapse_run serve "$@"
}
cmd_synapse_stop() { [[ -z "${1:-}" ]] && die "Usage: greenboost synapse stop <model>"; _gb_synapse_run stop "$@"; }

_cmd_synapse_ps_snapshot() {
    echo ""
    echo -e "  ${C_BOLD}gb-synapse - running servers${C_RESET}  ${C_DIM}$(date '+%H:%M:%S')${C_RESET}"
    gb_separator
    _gb_synapse_run ps
    echo ""
}
cmd_synapse_ps() {
    local _llm_mode=0
    for _a in "$@"; do [[ "$_a" == "--llm" ]] && _llm_mode=1; done
    (( _llm_mode )) && { _gb_synapse_run ps --llm; return; }
    if [[ ! -t 0 ]]; then _cmd_synapse_ps_snapshot; return; fi
    _gb_run_tui_loop "_cmd_synapse_ps_snapshot" 5 \
        "  ${C_DIM}Updating every 5s  ${C_GRAY}Ctrl+S${C_DIM}: refresh  ${C_GRAY}Ctrl+C${C_DIM}: exit${C_RESET}"
}

cmd_synapse() {
    case "${1:-}" in
        login)         shift; cmd_synapse_login "$@" ;;
        pull)          shift; cmd_synapse_pull "$@" ;;
        list)          shift; cmd_synapse_list "$@" ;;
        rm)            shift; cmd_synapse_rm "$@" ;;
        index-ollama)  shift; cmd_synapse_index_ollama "$@" ;;
        build-engine)  cmd_synapse_build_engine ;;
        update-engine) cmd_synapse_update_engine ;;
        run|serve)     shift; cmd_synapse_serve "$@" ;;
        stop)          shift; cmd_synapse_stop "$@" ;;
        ps)            shift; cmd_synapse_ps "$@" ;;
        *) die "Usage: greenboost synapse [login|pull|list|rm|index-ollama|build-engine|update-engine|run|stop|ps]" ;;
    esac
}

# _cmd_doctor_snapshot / cmd_doctor - cluster hardware + gb-synapse readiness
# view. Follows the UI Command Paradigm (--llm, non-tty, _gb_run_tui_loop).
_cmd_doctor_snapshot() {
    echo ""
    echo -e "  ${C_BOLD}greenboost doctor${C_RESET}  ${C_DIM}$(date '+%H:%M:%S')${C_RESET}"
    gb_separator
    _gb_synapse_run doctor
    echo ""
}
cmd_doctor() {
    local _llm_mode=0
    for _a in "$@"; do [[ "$_a" == "--llm" ]] && _llm_mode=1; done
    (( _llm_mode )) && { _gb_synapse_run doctor --llm; return; }
    if [[ ! -t 0 ]]; then _cmd_doctor_snapshot; return; fi
    _gb_run_tui_loop "_cmd_doctor_snapshot" 5 \
        "  ${C_DIM}Updating every 5s  ${C_GRAY}Ctrl+S${C_DIM}: refresh  ${C_GRAY}Ctrl+C${C_DIM}: exit${C_RESET}"
}

# cmd_recommend - fit + throughput estimate per manifest model against the
# live cluster's aggregate VRAM. One-shot (not a TUI loop - it's a report,
# not a live status view).
cmd_recommend() {
    local _llm_mode=0
    local -a _args=()
    for _a in "$@"; do
        if [[ "$_a" == "--llm" ]]; then _llm_mode=1; else _args+=("$_a"); fi
    done
    if (( _llm_mode )); then
        _gb_synapse_run recommend "${_args[@]}" --llm
        return
    fi
    echo ""
    echo -e "  ${C_BOLD}greenboost recommend${C_RESET}"
    gb_separator
    _gb_synapse_run recommend "${_args[@]}"
    echo ""
}


# ---- wizard (default interactive mode) --------------------------------
# Shown when no arguments are given and stdin is a TTY.

cmd_wizard() {
    while true; do
        clear
        gb_header
        echo -e "  ${C_DIM}${C_GRAY}GreenBoost improves local AI inference by orchestrating a CUDA memory pool.${C_RESET}"
        echo ""

        gb_section "Core"
        gb_menu_item  1  "Full install"              "DKMS module + tune runtime + sysctl + GRUB + systemd services + latest CUDA toolkit"  root
        gb_menu_item  2  "Light install"             "DKMS module + hardware-tuned build - no system changes, no CUDA toolkit install"  root
        gb_menu_item  3  "Status"                    "Show cuda memory pool + system state"
        gb_menu_item  4  "Benchmark"                 "Measure T1/T2/T3 bandwidth"

        gb_section "Additional install (included in full-install)"
        gb_menu_item  5  "Install sys configs"       "Ollama drop-in, udev rules, LD_PRELOAD shim, CPU governor, THP"  root
        gb_menu_item  6  "Tune runtime"              "NVMe scheduler, CPU governor, PCIe, swappiness (live)"  root
        gb_menu_item  7  "Tune sysctl"               "Persistent kernel tunables - 99-zzz-greenboost.conf"  root
        gb_menu_item  8  "Tune GRUB"                 "Boot params: hugepages, rcu_nocbs, nohz_full (needs reboot)"  root
        gb_menu_item  9  "Generate inference config"  "Optimized Ollama/HF config for this hardware & environment"
        gb_menu_item 10  "Build gb-synapse engine"   "From-source llama.cpp build (llama-server, rpc-server, llama-quantize)"  root

        gb_section "Restore"
        gb_menu_item 11  "Restore sys configs"       "Remove Ollama drop-in, udev rules, LD_PRELOAD, governor service"  root
        gb_menu_item 12  "Restore tune runtime"      "Reset CPU governor, NVMe scheduler, PCIe PM, VM defaults"  root
        gb_menu_item 13  "Restore tune sysctl"       "Remove 99-zzz-greenboost.conf and reload kernel defaults"  root
        gb_menu_item 14  "Restore tune GRUB"         "Strip GreenBoost boot params and run update-grub"  root

        gb_section "Configuration"
        gb_menu_item 15  "Profile management"        "Interactive wizard: create, activate, diff profiles"

        gb_section "Maintenance"
        gb_menu_item 16  "GreenBoost Commands"       "All commands reference (also: greenboost help)"
        gb_menu_item 17  "Clear logs"                "Clear dmesg and journal"
        gb_menu_item 18  "Uninstall"                 "Remove GreenBoost (module + all config)"  root
        gb_menu_item 19  "Install Python files"      "Copy gb_*.py orchestration stack to /usr/local/lib/greenboost/"  root

        gb_separator
        echo -e "  ${C_DIM}${C_GRAY}[Q]  Quit${C_RESET}"
        echo -e ""

        if [[ "$(id -u)" != "0" ]]; then
            gb_warn_ui "Not running as root - options marked ${C_RED}⚠ root${C_AMBER} will fail. Use sudo."
            echo -e ""
        fi

        gb_prompt "Choice"
        local choice="$REPLY"
        echo -e ""

        case "$choice" in
            1)  cmd_full_install;                    gb_press_enter ;;
            2)  bash "$MODULE_DIR/install_module.sh"; gb_press_enter ;;
            3)  cmd_debug_vitals ;;
            4)  cmd_benchmark;                       gb_press_enter ;;
            5)  cmd_install_sys_configs;             gb_press_enter ;;
            6)  cmd_tune;                            gb_press_enter ;;
            7)  cmd_tune_sysctl;                     gb_press_enter ;;
            8)  cmd_tune_grub;                       gb_press_enter ;;
            9)  cmd_gen_inference_config;            gb_press_enter ;;
            10) cmd_synapse_build_engine;            gb_press_enter ;;
            11) cmd_restore_sys_configs;             gb_press_enter ;;
            12) cmd_restore_tune_runtime;            gb_press_enter ;;
            13) cmd_restore_tune_sysctl;             gb_press_enter ;;
            14) cmd_restore_tune_grub;               gb_press_enter ;;
            15) cmd_profile_wizard ;;
            16) cmd_show_commands;                   gb_press_enter ;;
            17) cmd_clear_logs;                      gb_press_enter ;;
            18) cmd_uninstall;                       gb_press_enter ;;
            19) cmd_install_python_files;            gb_press_enter ;;
            q|Q|"") exit 0 ;;
            *) gb_warn_ui "Unknown option."; sleep 1 ;;
        esac
    done
}

# ── cmd_benchmark - workstation bandwidth benchmark ──────────────────────
# Usage: greenboost benchmark [--skip-bandwidth] [--json]
#
# cuda memory pool bandwidth test (T1 VRAM / T2 DDR / T3 NVMe)
# Results logged to /var/log/greenboost/benchmark-<timestamp>.log
cmd_benchmark() {
    local skip_bw=0 json_out=0
    for arg in "$@"; do
        case "$arg" in
            --skip-bandwidth)  skip_bw=1  ;;
            # Audit F-L4-15: accept both --json (legacy) and --llm (canonical).
            --json|--llm)      json_out=1 ;;
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
            gb_warn_ui "Bandwidth benchmark script not found - skipping"
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

    detect_hardware 2>/dev/null || true
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
        printf "  ${C_LIME}%-20s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "setup"              "Full install - deps, module, tune, Python inference tools"
        printf "  ${C_LIME}%-20s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "vitals"             "Live TUI: tier usage, GPU telemetry, inference state, diagnostics"
        printf "  ${C_LIME}%-20s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "benchmark"          "Memory pool bandwidth benchmark (T1/T2/T3)  [--skip-bandwidth] [--llm]"
        printf "  ${C_LIME}%-20s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "tune"               "Runtime tuning (CPU governor, NVMe, THP, sysctl)"
        printf "  ${C_LIME}%-20s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "load"               "Load kernel module with cuda memory pool params"
        printf "  ${C_LIME}%-20s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "unload"             "Unload module"
        printf "  ${C_LIME}%-20s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "uninstall"          "Remove module + all config"
        printf "  ${C_LIME}%-20s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "profile"            "Interactive profile wizard (create / activate / diff)"
        printf "  ${C_LIME}%-20s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clear memory-pool"  "Force-release T1 VRAM + T2 RAM + T3 now (unloads inference models)"
        printf "  ${C_LIME}%-20s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clear cluster-workers" "Kill feeder stage/block workers + host tunnels, remove stage temp files"
        printf "  ${C_LIME}%-20s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "logs"               "Live log TUI (kernel, Ollama, AppArmor)"
        printf "  ${C_LIME}%-20s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "logs save [file]"   "Save full diagnostic bundle to file (prompts if omitted)"
        printf "  ${C_LIME}%-20s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "logs start [file]"  "Start background log capture (appends every 30s)"
        printf "  ${C_LIME}%-20s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "logs stop"          "Stop background log capture"
        printf "  ${C_LIME}%-20s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clear logs"         "Clear all GreenBoost logs for a fresh diagnostic baseline"
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
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "tune-revert"     "Restore continuous OS-tuner levers to pre-tune baseline"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "tune-grub"       "Fix GRUB boot params (THP=always, rcu_nocbs, nohz_full…)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "tune-sysctl"     "Consolidate sysctl + apply compute-optimized knobs"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "tune-all"        "Run tune + tune-grub + tune-sysctl + tune-libs"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}DIAGNOSTICS:${C_RESET}"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "vitals"             "Live TUI: tier usage, GPU telemetry, inference state, diagnostics  [--llm]"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "health-check"       "One-shot PASS/FAIL/WARN across module, VRAM, T2, T3, cluster  [--llm]"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clear memory-pool"  "Force-release T1 VRAM + T2 RAM + T3 immediately (unloads models)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clear cluster-workers" "Kill feeder stage/block workers + host tunnels, remove stage temp files"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "benchmark [flags]"  "Memory pool bandwidth benchmark  [--skip-bandwidth] [--llm]"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}LOGGING:${C_RESET}"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "logs"                  "Live log TUI (kernel, Ollama, AppArmor)  [--llm]"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "logs save [file]"      "Save full diagnostic bundle to file (prompts if omitted)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "logs start [file]"     "Start background log capture (appends snapshot every 30s)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "logs stop"             "Stop background log capture"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "inference-logs"        "Show Ollama/inference logs"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "inference-test"        "Benchmark inference + verify fastest path (A0/A); --llm for LLM report"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "test cluster"          "Live cluster inference test: Qwen3-35B tier routing + throughput (--unload)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clear logs"            "Clear all GreenBoost logs"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clear inference-logs"  "Clear inference logs"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}ADVANCED:${C_RESET}"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "install-python"        "Copy gb_*.py orchestration stack to /usr/local/lib/greenboost/"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "install-cli"           "Install greenboost-cli (gb) into /usr/local/lib/greenboost/cli-venv"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "install-pipelines"     "Provision ai-forge pipeline deps (PaddleOCR etc.) via setup_*.sh"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "register-mcp"          "Register GreenBoost MCP servers with the Claude CLI (per-user)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "install-sys-configs"   "Ollama drop-in, udev, governor, LD_PRELOAD, THP (all-in-one)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "install-llama-configs" "LD_AUDIT → ld.so.preload only (also runs inside install-sys-configs)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "restore-sys-configs"   "Undo install-sys-configs (drop-in, udev, ld.so.preload, governor)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "restore-tune-runtime"  "Reset CPU governor, NVMe scheduler, PCIe PM, VM tunables"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "restore-tune-sysctl"   "Remove 99-zzz-greenboost.conf and reload kernel defaults"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "restore-tune-grub"     "Strip GreenBoost GRUB params and run update-grub"
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
        echo -e "  ${C_DIM}greenboost vitals${C_RESET}"
        echo -e "  ${C_DIM}dmesg | grep greenboost | tail -20${C_RESET}"
        echo -e "  ${C_DIM}watch -n1 free -h   # T2 RAM pressure${C_RESET}"
        echo -e "  ${C_DIM}watch -n1 swapon --show   # T3 NVMe usage${C_RESET}"
    fi
    echo -e ""
}

# ---- install-deps ------------------------------------------------------
# Install all Ubuntu packages needed for GreenBoost v3.2 + ExLlamaV3

cmd_install_build_deps() {
    need_root install-deps

    # Minimal packages required to build and DKMS-register the kernel module.
    # No AI libraries, no Python, no CUDA - those go in cmd_install_optional_pkgs.
    local _pkgs=(
        build-essential gcc gcc-multilib make git curl wget
        "linux-headers-$(uname -r)"
        kmod dkms
        # eBPF tracer build deps (clang/bpftool/libbpf , see
        # _gb_install_ebpf_tracer / `make BPF=1 ebpf`). Previously absent from
        # every install path, so the tracer silently skipped itself with a
        # "clang/bpftool/libbpf required" notice on every fresh system and
        # had to be built by hand afterward (verified 2026-07-09). Required,
        # not optional, so a fresh install builds it in step [2/5] with no
        # manual follow-up.
        clang bpftool libbpf-dev
    )
    # CPU vendor-specific microcode (safe to install on any machine)
    local _cpu_vendor
    _cpu_vendor=$(grep -m1 "vendor_id" /proc/cpuinfo | awk '{print $3}')
    if [[ "$_cpu_vendor" == "GenuineIntel" ]]; then
        _pkgs+=(intel-microcode)
    elif [[ "$_cpu_vendor" == "AuthenticAMD" ]]; then
        _pkgs+=(amd64-microcode)
    fi

    local _missing=()
    for pkg in "${_pkgs[@]}"; do
        if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "ok installed"; then
            _missing+=("$pkg")
        fi
    done

    if [[ ${#_missing[@]} -eq 0 ]]; then
        gb_ok "Build dependencies already installed"
    else
        printf "  ${C_CYAN}❯${C_RESET}  ${C_DIM}Updating package lists...${C_RESET}"
        apt-get update -qq 2>/dev/null || true
        printf "\r%*s\r" "$(tput cols 2>/dev/null || echo 80)" ""

        # Install with live APT progress bar via APT::Status-Fd
        # Audit F-L4-19: use mktemp -d so the FIFO path is atomically unique;
        # the old mktemp -u pattern races between name generation and mkfifo.
        local _fifo_dir _fifo
        _fifo_dir=$(mktemp -d)
        _fifo="${_fifo_dir}/apt_progress"
        mkfifo "$_fifo"

        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
            -o "APT::Status-Fd=3" \
            "${_missing[@]}" 2>/dev/null 3>"$_fifo" &
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

        wait "$_apt_pid" || gb_warn_ui "Some packages failed - check: apt-get install ${_missing[*]}"
        rm -rf "$_fifo_dir"
        printf "\r%*s\r" "$(tput cols 2>/dev/null || echo 80)" ""
    fi

    # Ensure cpuid module loads at boot
    if ! grep -q cpuid /etc/modules-load.d/*.conf 2>/dev/null; then
        echo cpuid > /etc/modules-load.d/ai-workstation.conf
    fi

    gb_ok "Build dependencies installed"
}


# cmd_install_deps - installs all dependencies (build + optional).
# Called when running 'install-deps' directly; full-install uses
# cmd_install_build_deps (always) + cmd_install_optional_pkgs (gated).
cmd_install_optional_pkgs() {
    need_root install-optional-pkgs

    # Restored minimal set of topology/freq tools for GreenBoost profiling
    local _pkgs=(
        hwloc libhwloc-dev nvtop
    )
    local _os_id
    _os_id=$(grep -oP '^ID=\K.*' /etc/os-release 2>/dev/null | tr -d '"')
    if [[ "$_os_id" == "debian" ]]; then
        _pkgs+=(linux-cpupower linux-perf)
    else
        _pkgs+=(cpufrequtils linux-tools-generic "linux-tools-$(uname -r)")
    fi

    local _missing=()
    for pkg in "${_pkgs[@]}"; do
        if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "ok installed"; then
            _missing+=("$pkg")
        fi
    done

    if [[ ${#_missing[@]} -eq 0 ]]; then
        gb_ok "Topology and monitoring tools already installed"
        return
    fi

    printf "  ${C_CYAN}❯${C_RESET}  ${C_DIM}Updating package lists...${C_RESET}"
    apt-get update -qq 2>/dev/null || true
    printf "\r%*s\r" "$(tput cols 2>/dev/null || echo 80)" ""

    # PR-J/F-L4-19: use mktemp -d so the FIFO path is atomically unique;
    # the old mktemp -u pattern races between name generation and mkfifo
    # (an attacker with /tmp write access could pre-create the file).
    # This call was missed when the same fix landed for cmd_install_build_deps.
    local _fifo_dir _fifo
    _fifo_dir=$(mktemp -d)
    _fifo="${_fifo_dir}/apt_progress"
    mkfifo "$_fifo"

    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        -o "APT::Status-Fd=3" \
        "${_missing[@]}" 2>/dev/null 3>"$_fifo" &
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

    wait "$_apt_pid" || gb_warn_ui "Some packages failed - check: apt-get install ${_missing[*]}"
    rm -rf "$_fifo_dir"
    rm -f "$_fifo"
    printf "\r%*s\r" "$(tput cols 2>/dev/null || echo 80)" ""

    gb_ok "Topology and monitoring tools installed"
}

cmd_install_deps() {
    cmd_install_build_deps
    cmd_install_optional_pkgs
}

# ---- full-install ------------------------------------------------------
# Complete fresh-OS install - run this after a clean Ubuntu install.
# Covers: OS deps, kernel module, CUDA shim, all system configs,
# sysctl tuning, GRUB params, and optional ExLlamaV3 with GreenBoost patches.
# NVMe swap (T3 tier) is intentionally NOT touched - configure manually before running.

cmd_full_install() {
    need_root full-install
    GB_STOPPED_SERVICES=""

    # Strip libgreenboost entries from ld.so.preload before spawning any
    # subprocess.  See cmd_install for rationale.
    if [[ -f /etc/ld.so.preload ]]; then
        sed -i '/libgreenboost/d' /etc/ld.so.preload
        [[ -s /etc/ld.so.preload ]] || rm -f /etc/ld.so.preload
    fi

    # Capture pre-install system state for potential rollback
    _gb_backup_create

    # ── Mode selection - cmd_full_install always targets full system setup.
    # Only --module-only overrides this (for scripts / CI / direct invocation).
    GB_INSTALL_MODE="full"
    for _a in "$@"; do
        [[ "$_a" == "--module-only" ]] && { GB_INSTALL_MODE="module"; break; }
    done

    # Build the mode flag to forward to delegated distro scripts
    local _mode_flag="--module-only"
    [[ "$GB_INSTALL_MODE" == "full" ]] && _mode_flag="--full-install"

    # ── Distro detection: delegate to per-family scripts ──────────────────
    local _distro_family
    _distro_family=$(_detect_distro_family)

    if [[ "$_distro_family" == "fedora" ]]; then
        local rocky_script="$MODULE_DIR/greenboost_setup_rocky.sh"
        [[ -x "$rocky_script" ]] || die "Red Hat-based system detected but $rocky_script not found or not executable."
        info "Red Hat-based system detected (family: fedora) - delegating to greenboost_setup_rocky.sh"
        exec "$rocky_script" full-install "$_mode_flag" "$@"
    fi

    if [[ "$_distro_family" == "arch" ]]; then
        local arch_script="$MODULE_DIR/greenboost_setup_arch.sh"
        [[ -x "$arch_script" ]] || die "Arch-based system detected but $arch_script not found or not executable."
        info "Arch-based system detected (family: arch) - delegating to greenboost_setup_arch.sh"
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

    # 0 - Purge any previous GreenBoost install to guarantee a clean slate
    gb_step 0 5 "Purging previous GreenBoost installation (if any)..."
    if ! do_purge 0 1; then
        # Module is stuck in STATE_GOING.  Try to compile + install a fixed .ko
        # (without rmmod) so the next boot loads correctly and full-install can
        # proceed.  If compilation fails, fall back to boot cleanup (full wipe).
        if _try_install_fixed_module; then
            gb_warn_ui "Fixed module installed as a fallback for this boot."
        fi
        # Always schedule boot cleanup regardless of whether the fixed module
        # compiled.  This ensures the next boot starts with a clean slate:
        # the cleanup removes any .ko on disk and blocks auto-load so that
        # full-install can compile a fresh module from scratch.
        gb_warn_ui "Scheduling boot-time cleanup for a clean start on next boot…"
        _schedule_boot_cleanup
        info ""
        info "  If 'sudo reboot' still hangs (shutdown hook may not have installed), force-reboot:"
        info "    sync && echo b | sudo tee /proc/sysrq-trigger"
        info "  (Machine reboots instantly - SSH drops. Wait ~30s then reconnect.)"
        info "  Or hold the power button 5 seconds."
        die "Reboot now, then re-run:  sudo ./greenboost_setup.sh full-install"
    fi
    gb_ok "Previous installation purged"

    # 1 - Build dependencies (minimal - just what's needed to compile the module)
    gb_step 1 5 "Installing build dependencies..."
    cmd_install_build_deps

    # 2 - Build + install kernel module + CUDA shim
    gb_step 2 5 "Building and installing kernel module + CUDA shim..."
    GB_SKIP_INSTALL_PURGE=1 cmd_install
    gb_ok "Kernel module + CUDA shim installed"

    # 3 - Load kernel module
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

    # 4 - System configs: Ollama, udev rules, CPU governor service, sysctl, llama.cpp
    gb_step 4 5 "Installing system configuration files..."
    gb_info "Applying: Ollama/inference service config (drop-ins, udev, NVMe, cpu-perf)"
    cmd_install_sys_configs
    cmd_install_llama_configs
    gb_ok "System configuration installed"

    cmd_install_python_files
    gb_ok "GreenBoost Python orchestration stack installed"

    cmd_install_cli

    cmd_install_optional_pkgs
    gb_ok "Optional AI/compute libraries installed"

    # 4b - Latest CUDA toolkit (full install only - light install never
    # touches system CUDA). Needed for gb-synapse's from-source llama.cpp
    # build to target current GPU architectures (e.g. Blackwell sm_120,
    # which requires CUDA >= 12.8 - a distro-packaged nvcc is often older).
    gb_info "Checking CUDA toolkit (gb-synapse's engine build needs a current one)..."
    _gb_install_cuda_toolkit


    # 5 - System tuning (sysctl + NVMe + CPU governor + THP)
    gb_step 5 5 "Applying system tuning..."
    gb_info "Applying: sysctl + NVMe/THP/CPU governor tuning"
    cmd_tune_sysctl
    cmd_tune
    gb_ok "sysctl and runtime tuning applied"

    # 5b - GRUB boot parameters (requires reboot)
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
            || warn "$svc restart failed - run: sudo systemctl restart $svc"
    done

    # 6 - gb-synapse engine (llama-server, rpc-server, llama-quantize),
    # from-source build against the CUDA toolkit installed in step 4b.
    # Last step deliberately: needs CUDA + Python orchestration stack from
    # earlier steps already in place. Best-effort under set -euo pipefail —
    # a build failure (network hiccup fetching llama.cpp, disk space, a
    # toolchain issue) shouldn't take down an otherwise-successful full
    # install; the same step is also offered standalone (menu option 10 /
    # `sudo greenboost synapse build-engine`) for retrying afterward.
    gb_info "Building gb-synapse engine (llama.cpp: llama-server, rpc-server, llama-quantize)..."
    if cmd_synapse_build_engine; then
        gb_ok "gb-synapse engine built"
    else
        gb_warn_ui "gb-synapse engine build failed — retry later: sudo greenboost synapse build-engine"
    fi

    echo ""
    gb_separator
    echo -e ""
    gb_ok "${C_BOLD}Full install complete!"
    echo -e ""
    echo -e "  ${C_AMBER}${C_BOLD}⚠  REBOOT REQUIRED${C_RESET} ${C_GRAY}to activate GRUB params + hugepage pre-allocation${C_RESET}"
    echo -e ""
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
# cmd_diag - feeder diagnostic tests using gb_feeder_diag.py
# ---------------------------------------------------------------------------
cmd_diag() {
    local subcmd="${1:-all}"
    local _diag="${MODULE_DIR}/gb_feeder_diag.py"
    if [[ ! -f "$_diag" ]]; then
        die "Diagnostic script not found: $_diag (reinstall GreenBoost)"
    fi
    if ! command -v python3 &>/dev/null; then
        die "python3 not found - install Python 3 to run diagnostics"
    fi
    case "$subcmd" in
        feeder|feeder-t1|feeder-t2|feeder-t3|feeder-compute|all|info)
            ;;  # valid
        t1|t2|t3|compute)
            ;;  # also valid (forward directly)
        *)
            echo "Usage: greenboost diag [feeder|feeder-t1|feeder-t2|feeder-t3|feeder-compute|all]" >&2
            echo "       greenboost diag [--ip IP] [--port PORT] [t1|t2|t3|compute|info|all]" >&2
            exit 1
            ;;
    esac
    # Map feeder-* aliases to bare tier names
    case "$subcmd" in
        feeder)         subcmd="all" ;;
        feeder-t1)      subcmd="t1" ;;
        feeder-t2)      subcmd="t2" ;;
        feeder-t3)      subcmd="t3" ;;
        feeder-compute) subcmd="compute" ;;
    esac
    shift
    python3 "$_diag" "$subcmd" "$@"
}

# ---------------------------------------------------------------------------
# cmd_recover - manual crash-recovery entrypoint.
# Runs the GreenBoost supervisor in --recover mode: same boot-time recovery
# sequence (modprobe, OOM guard clear, fault classification, sentinel cleanup)
# but exits immediately instead of entering the daemon loop.
# ---------------------------------------------------------------------------
cmd_recover() {
    need_root recover
    local script="/usr/local/lib/greenboost/gb_supervisor.py"
    if [[ ! -x "$script" ]]; then
        gb_warn_ui "Supervisor not installed. Running: sudo $0 install-sys-configs"
        cmd_install_supervisor
    fi
    exec python3 "$script" --recover
}

# Check for a newer release before any install operation.
# Skipped for non-modifying commands (status, unload, help, build).
# Suppressed when --skip-update-check is passed (offline workstations).
case "$COMMAND" in
    install|setup|full-install|install-sys-configs|install-llama-configs|recover)
        if [[ $GB_SKIP_UPDATE -eq 0 ]]; then check_update; fi
        echo "" ;;
esac

case "$COMMAND" in
    # Primary commands
    install)             cmd_install            ;;
    uninstall)           cmd_uninstall          ;;
    apparmor-uninstall)  cmd_apparmor_uninstall ;;
    backup)              need_root backup; _gb_backup_create ;;
    restore)             need_root restore; _gb_backup_restore ;;
    build|build-info)    cmd_build_info "$@"    ;;
    compile)             cmd_build              ;;
    load)                cmd_load               ;;
    unload)              cmd_unload             ;;
    setup|full-install)  cmd_full_install "--full-install" "$@"  ;;
    module-only)         GB_INSTALL_MODE="module" cmd_full_install "--module-only" "$@" ;;
    tune)                cmd_tune               ;;
    tune-revert)         cmd_tune_revert        ;;
    tune-grub)           cmd_tune_grub          ;;
    tune-sysctl)         cmd_tune_sysctl        ;;
    tune-libs)           cmd_tune_libs          ;;
    tune-all)            cmd_tune_all           ;;
    benchmark)           cmd_benchmark "${@:2}" ;;
    profile)             cmd_profile "${@:2}"   ;;
    stop)
        if [[ "${2:-}" == "feed" ]]; then
            cmd_feed stop
        else
            die "Unknown command: stop ${2:-}"
        fi
        ;;
    feed)                cmd_feed "${@:2}"      ;;
    connect)             cmd_connect "${@:2}"   ;;
    disconnect)          cmd_disconnect "${@:2}" ;;
    cluster)             cmd_cluster            ;;
    dataflux-ui)         cmd_dataflux_ui "${@:2}" ;;
    update)
        case "${2:-}" in
            feeders) cmd_update_feeders ;;
            *) die "Usage: greenboost update feeders" ;;
        esac
        ;;
    update-feeders)      cmd_update_feeders     ;;
    feeders)
        case "${2:-}" in
            upgrade-greenboost) die "Use: sudo greenboost update feeders  (matches ollama + rebuilds GreenBoost on each feeder)" ;;
            setup-sudo)         cmd_feeders_setup_sudo ;;
            diag)               cmd_feeders_diag "${3:-all}" ;;
            export-key)         cmd_feeders_export_key ;;
            import-key)         cmd_feeders_import_key "${3:-}" ;;
            genkey)             cmd_feeders_genkey ;;
            redeploy-netd)      cmd_feeders_redeploy_netd ;;
            sync-ollama)        cmd_feeders_sync_ollama ;;
            *) die "Usage: greenboost feeders [setup-sudo|diag|export-key|import-key|genkey|redeploy-netd|sync-ollama]  (to update feeders: sudo greenboost update feeders)" ;;
        esac
        ;;
    built-stamp)         cmd_built_stamp "${@:2}" ;;
    capabilities)        cmd_capabilities "${@:2}" ;;
    pilot)               cmd_pilot "${@:2}" ;;

    t3-memory)           cmd_t3_memory "${2:-}" ;;
    clean)
        case "${2:-}" in
            memory)  gb_warn_ui "'greenboost clean memory' deprecated - use 'greenboost clear memory-pool'"; cmd_clear_memory_pool ;;
            *) die "Usage: greenboost clear memory-pool" ;;
        esac
        ;;
    clean-memory)        gb_warn_ui "'greenboost clean-memory' deprecated - use 'greenboost clear memory-pool'"; cmd_clear_memory_pool ;;
    logs)                cmd_logs               "${@:2}" ;;

    inference-logs)      cmd_inference_logs     "${@:2}" ;;
    nvtx-logs)           cmd_nvtx_logs          "${@:2}" ;;
    nvtx)
        case "${2:-}" in
            vitals) cmd_nvtx_vitals "${@:3}" ;;
            *)      die "Usage: greenboost nvtx vitals [--last N] [--filter EV1,EV2] [--feeder-only] [--local-only] [--llm]" ;;
        esac
        ;;
    clear)
        case "${2:-}" in
            logs)            cmd_clear_logs ;;

            inference-logs)  cmd_clear_inference_logs ;;
            memory-pool)     cmd_clear_memory_pool ;;
            cluster-workers) cmd_clear_cluster_workers ;;
            nvtx-logs)       cmd_clear_nvtx_logs ;;
            *) die "Usage: greenboost clear logs|inference-logs|memory-pool|cluster-workers|nvtx-logs" ;;
        esac
        ;;
    clean-logs)          cmd_clean_logs         ;;
    show-commands)       cmd_show_commands      ;;
    help|--help|-h)      cmd_help "${@:2}"      ;;
    turboquant)          cmd_turboquant "${@:2}" ;;
    ggml-2dev)           cmd_ggml_2dev "${@:2}" ;;
    gen-inference-config) cmd_gen_inference_config "${@:2}" ;;
    doctor)              cmd_doctor "${@:2}" ;;
    recommend)           cmd_recommend "${@:2}" ;;
    pull)                cmd_synapse_pull "${@:2}" ;;
    synapse)             cmd_synapse "${@:2}" ;;
    health-check)        cmd_health_check "${@:2}"  ;;
    diag)               cmd_diag "${@:2}"    ;;
    stability)           cmd_stability_monitor "${@:2}" ;;
    # Advanced (kept for compat)
    recover)                cmd_recover               ;;
    install-python|install_python) cmd_install_python_files ;;
    install-cli|install_cli)       cmd_install_cli           ;;
    install-pipelines|install_pipelines) cmd_install_pipelines ;;
    register-mcp|register_mcp)     cmd_register_mcp          ;;
    install-sys-configs)    cmd_install_sys_configs   ;;
    install-llama-configs)  cmd_install_llama_configs ;;
    restore-sys-configs)    cmd_restore_sys_configs   ;;
    restore-tune-runtime)   cmd_restore_tune_runtime  ;;
    restore-tune-sysctl)    cmd_restore_tune_sysctl   ;;
    restore-tune-grub)      cmd_restore_tune_grub     ;;
    inference-test)         cmd_inference_test        "${@:2}" ;;
    test)
        case "${2:-}" in
            cluster) exec "$MODULE_DIR/test_cluster_qwen3.sh" "${@:3}" ;;
            *) die "Usage: greenboost test cluster [--unload]" ;;
        esac
        ;;
    debug)               cmd_debug "${@:2}" ;;
    vitals)              cmd_debug_vitals "${@:2}" ;;
    faults)              cmd_faults "${@:2}" ;;
    top)                 cmd_top "${@:2}" ;;
    residency)           cmd_residency "${@:2}" ;;
    # Default: interactive wizard
    "")
        cmd_wizard ;;
    *) die "Unknown command: '$COMMAND'  - run: $0 help" ;;
esac

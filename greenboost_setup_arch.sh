#!/usr/bin/env bash
# GreenBoost - Setup & installation script for Arch Linux
# Supports: Arch, Manjaro, EndeavourOS, CachyOS, and any Arch-based distro.
#
# This script mirrors greenboost_setup.sh's Arch-specific parts: package-manager
# bootstrap (pacman) and the menu UI. Feature logic shared with the main script
# (cluster connect, vitals/pool-info parsing, turboquant, etc.) is NOT duplicated
# here - it execs into greenboost_setup.sh, so patches to that logic land here
# automatically and never need manual porting. Only Arch-specific bootstrap/UI
# changes need to be made in this file directly.
#
# USAGE:
#   sudo ./greenboost_setup_arch.sh full-install     - full install (prompts for mode)
#   sudo ./greenboost_setup_arch.sh module-only      - kernel module only (safe on any machine)
#   sudo ./greenboost_setup_arch.sh install          - build + install system-wide
#   sudo ./greenboost_setup_arch.sh uninstall        - remove module + all config
#   sudo ./greenboost_setup_arch.sh load             - insmod with detected params
#   sudo ./greenboost_setup_arch.sh unload           - rmmod
#        ./greenboost_setup_arch.sh status           - show pool info + system state
#        ./greenboost_setup_arch.sh                  - open interactive wizard

set -euo pipefail

DRIVER_NAME="greenboost"
SHIM_LIB="libgreenboost_cuda.so"
AUDIT_LIB="libgreenboost_audit.so"
VULKAN_LAYER_LIB="libVkLayer_greenboost.so"
VULKAN_LAYER_MANIFEST="VkLayer_greenboost.json"
VULKAN_IMPLICIT_LAYER_DIR="/etc/vulkan/implicit_layer.d"
SHIM_DEST="/usr/local/lib"
MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GB_VERSION="3.2"
GB_PROFILES_DIR="/etc/greenboost/profiles"
GB_ACTIVE_PROFILE_LINK="/etc/greenboost/active_profile.md"
GB_STOPPED_SERVICES=""

# ---- Basic helpers (kept for backward-compat with older function calls) -----
RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GRN}[GreenBoost]${NC} $*"; }
warn()  { echo -e "${YLW}[GreenBoost] WARN:${NC} $*"; }
die()   { echo -e "${RED}[GreenBoost] ERROR:${NC} $*" >&2; exit 1; }

# ---- Brand palette + UI primitives (mirrors greenboost_setup.sh) -----------
_gb_truecolor() { [[ "${COLORTERM:-}" =~ ^(truecolor|24bit)$ ]]; }

if _gb_truecolor; then
    C_VIOLET='\033[38;2;108;113;196m'
    C_LIME='\033[38;2;230;255;60m'
    C_GRAY='\033[38;2;208;207;204m'
    C_CYAN='\033[38;2;48;200;255m'
    C_AMBER='\033[38;2;255;191;0m'
    C_PURPLE='\033[38;2;167;139;250m'
    C_RED='\033[38;2;255;92;50m'
    C_WHITE='\033[38;2;255;255;255m'
else
    C_VIOLET='\033[0;34m'
    C_LIME='\033[0;32m'
    C_GRAY='\033[0;37m'
    C_CYAN='\033[0;36m'
    C_AMBER='\033[1;33m'
    C_PURPLE='\033[0;35m'
    C_RED='\033[0;31m'
    C_WHITE='\033[1;37m'
fi
C_BOLD='\033[1m'
C_DIM='\033[2m'
C_RESET='\033[0m'

GB_SPIN_FRAMES=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")

gb_header() {
    local cols; cols=$(tput cols 2>/dev/null || echo 64)
    local title=" GreenBoost v${GB_VERSION} - CUDA Memory & Compute Orchestrator for NVIDIA GPUs (Arch)"
    echo -e ""
    echo -e "${C_VIOLET}${C_BOLD}  ╔$(printf '═%.0s' $(seq 1 $((cols - 4))))╗${C_RESET}"
    echo -e "${C_VIOLET}${C_BOLD}  ║${C_RESET} ${C_GRAY}${C_BOLD}${title}$(printf ' %.0s' $(seq 1 $((cols - 4 - ${#title} - 1))))${C_VIOLET}${C_BOLD}║${C_RESET}"
    echo -e "${C_VIOLET}${C_BOLD}  ╚$(printf '═%.0s' $(seq 1 $((cols - 4))))╝${C_RESET}"
    echo -e ""
}
gb_separator() {
    local cols; cols=$(tput cols 2>/dev/null || echo 64)
    echo -e "${C_DIM}$(printf '─%.0s' $(seq 1 $((cols - 2))))${C_RESET}"
}
gb_step() {
    echo -e ""
    echo -e "${C_CYAN}${C_BOLD}  [$1/$2]${C_RESET} ${C_GRAY}${C_BOLD}$3${C_RESET}"
}
gb_ok()      { echo -e "  ${C_LIME}✓${C_RESET}  ${C_GRAY}$*${C_RESET}"; }
gb_fail()    { echo -e "  ${C_RED}✗${C_RESET}  $*"; }
gb_warn_ui() { echo -e "  ${C_AMBER}⚠${C_RESET}  $*"; }
gb_info()    { echo -e "  ${C_VIOLET}◈${C_RESET}  ${C_GRAY}$*${C_RESET}"; }
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
gb_section() {
    echo -e ""
    echo -e "  ${C_PURPLE}${C_BOLD}$1${C_RESET}"
    gb_separator
}
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
gb_prompt() {
    printf "\n  ${C_AMBER}${C_BOLD}❯${C_RESET}  ${C_GRAY}${1:-Choice}${C_RESET}: "
    read -r REPLY
}
gb_press_enter() {
    echo -e ""
    printf "  ${C_DIM}${C_GRAY}Press Enter to return to menu…${C_RESET}"
    read -r
}

# gb_confirm "Question" - amber ❯ Y/n; returns 0 if yes
gb_confirm() {
    printf "  ${C_AMBER}${C_BOLD}❯${C_RESET}  ${C_GRAY}$1${C_RESET} ${C_DIM}[Y/n]${C_RESET}: "
    read -r _confirm_reply
    [[ "${_confirm_reply:-Y}" =~ ^[Yy]$ ]]
}

# ── Install mode helpers ──────────────────────────────────────────────────────
GB_INSTALL_MODE="module"   # default: safe (no system tuning)

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

gb_consent_gate() {
    echo ""
    printf '%b\n' "  ${C_AMBER}${C_BOLD}⚠  System change:${C_RESET}  ${C_GRAY}$1${C_RESET}"
    gb_confirm "Apply this change?" && return 0 || return 1
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

# ---- Hardware detection ---------------------------------------------------
# Populates: GB_PHYS, GB_VIRT, GB_RESERVE, GB_NVME_SWAP, GB_NVME_POOL,
#            GB_PCORES_MAX, GB_GOLDEN_MIN, GB_GOLDEN_MAX, GB_PCORES_ONLY,
#            GPU_NAME, CPU_NAME, NVME_SIZE_GB, RAM_TYPE, RAM_SPEED_MT.
detect_hardware() {
    # GPU
    if command -v nvidia-smi &>/dev/null; then
        local gpu_mem_mib
        gpu_mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
            2>/dev/null | head -1 | tr -d ' ')
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 \
            | sed 's/^[[:space:]]*//')
        GB_PHYS=$(( ${gpu_mem_mib:-8192} / 1024 ))
        [[ $GB_PHYS -lt 1 ]] && GB_PHYS=1
    else
        GPU_NAME="Unknown (nvidia-smi not found)"
        GB_PHYS=8
        warn "nvidia-smi not found - assuming 8 GB VRAM. Install NVIDIA driver."
    fi

    # System RAM
    local total_kb; total_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
    local total_gb=$(( total_kb / 1024 / 1024 ))
    # virtual VRAM pool: 70% of total RAM (< 64 GB) or 80% (>= 64 GB).
    GB_VIRT=$(gb_calc_ddr_cap_gb "$total_gb")
    GB_RESERVE=$(( total_gb * 8 / 100 ))
    [[ $GB_RESERVE -lt 4  ]] && GB_RESERVE=4
    [[ $GB_RESERVE -gt 10 ]] && GB_RESERVE=10

    # RAM type/speed (best-effort via dmidecode)
    RAM_TYPE="DDR"
    RAM_SPEED_MT="auto"
    if command -v dmidecode &>/dev/null; then
        RAM_TYPE=$(dmidecode -t 17 2>/dev/null | awk '/Type:/ && !/Detail/ {print $2; exit}' \
            | grep -E 'DDR[0-9]' || echo "DDR")
        RAM_SPEED_MT=$(dmidecode -t 17 2>/dev/null | awk '/Speed:.*MT/ {print $2; exit}' \
            || echo "auto")
    fi

    # CPU
    CPU_NAME=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "Unknown CPU")

    # P-core / E-core detection (Intel hybrid) - graceful fallback for AMD/uniform
    GB_PCORES_MAX=$(nproc 2>/dev/null || echo 4)
    GB_GOLDEN_MIN=0
    GB_GOLDEN_MAX=$(( GB_PCORES_MAX - 1 ))
    GB_PCORES_ONLY=0

    local pcore_count=0
    while IFS= read -r _ctype_file; do
        local _ct; _ct=$(cat "$_ctype_file" 2>/dev/null || echo "")
        [[ "$_ct" == "performance" || "$_ct" == "1" ]] && (( pcore_count++ )) || true
    done < <(find /sys/devices/system/cpu/cpu*/topology -name "core_type" 2>/dev/null)

    if [[ $pcore_count -gt 0 ]]; then
        GB_PCORES_MAX=$(( pcore_count * 2 - 1 ))  # logical CPUs (HT)
        GB_GOLDEN_MIN=4
        GB_GOLDEN_MAX=$(( pcore_count - 1 ))
        GB_PCORES_ONLY=1
    fi

    # NVMe - largest NVMe device
    NVME_SIZE_GB=0
    while IFS= read -r blk; do
        local sz; sz=$(lsblk -bdn -o SIZE "/dev/$blk" 2>/dev/null | head -1 | tr -d ' ')
        local sz_gb=$(( ${sz:-0} / 1024 / 1024 / 1024 ))
        [[ $sz_gb -gt $NVME_SIZE_GB ]] && NVME_SIZE_GB=$sz_gb
    done < <(lsblk -dn -o NAME,TYPE 2>/dev/null | awk '$2=="disk" && $1~/nvme/ {print $1}')

    # ── System swap detection (T3) ───────────────────────────────────────
    # T3 uses existing system swap - GreenBoost does not create swap files.
    # Read /proc/swaps and prefer NVMe-backed entries; fall back to any swap.
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
    _t3_disk_gb=$(df -BG /var/lib/greenboost 2>/dev/null | awk 'NR==2{gsub("G",""); print $4}')
    _t3_disk_gb="${_t3_disk_gb:-0}"
    local _t3_disk_cap=$(( _t3_disk_gb * 80 / 100 ))
    [[ $_t3_needed -gt $_t3_disk_cap ]] && _t3_needed=$_t3_disk_cap
    GB_NVME_POOL=$_t3_needed

    # PCIe bandwidth (GPU slot)
    PCIE_GEN=0; PCIE_WIDTH="x?"; PCIE_BW_GBS=0
    if command -v lspci &>/dev/null; then
        _pcie_bw() { echo $(( $1 * $2 * 128 / 130 / 8 )); }
        _gts_to_gen() { case "$1" in 2.5*) echo 1;; 5*) echo 2;; 8*) echo 3;; 16*) echo 4;; 32*) echo 5;; *) echo 0;; esac; }
        local slot_spd; slot_spd=$(lspci -vvv 2>/dev/null | grep -A20 "VGA\|3D controller" \
            | awk '/LnkSta:/ && /Speed/ {for(i=1;i<=NF;i++) if($i~/Speed/) {gsub(/[^0-9.GT]/,"",$i); print $i; exit}}' | head -1)
        local slot_w_raw; slot_w_raw=$(lspci -vvv 2>/dev/null | grep -A20 "VGA\|3D controller" \
            | awk '/LnkSta:/ && /Width/ {for(i=1;i<=NF;i++) if($i~/Width/) {gsub(/Width,?/,"",$i); print $i; exit}}' | head -1)
        local slot_w=${slot_w_raw//x/}
        if [[ -n "$slot_spd" && -n "$slot_w" && "$slot_w" -gt 0 ]] 2>/dev/null; then
            PCIE_GEN=$(_gts_to_gen "$slot_spd")
            PCIE_WIDTH="x${slot_w}"
            PCIE_BW_GBS=$(_pcie_bw "$(echo "$slot_spd" | grep -oP '^[0-9]+')" "$slot_w")
        fi
    fi

    # Ollama context - T1+T2 KV headroom only (T3 NVMe excluded).
    # KV cache never spills to T3; use T1 VRAM + half T2 DDR as budget.
    # Half of T2 is left for model weights; the other half is KV headroom.
    local kv_pool_gb=$(( GB_PHYS + GB_VIRT / 2 ))
    if   [[ $kv_pool_gb -ge 32 ]]; then GB_OLLAMA_CTX=131072
    elif [[ $kv_pool_gb -ge 16 ]]; then GB_OLLAMA_CTX=65536
    elif [[ $kv_pool_gb -ge 8  ]]; then GB_OLLAMA_CTX=32768
    elif [[ $kv_pool_gb -ge 4  ]]; then GB_OLLAMA_CTX=16384
    else                                 GB_OLLAMA_CTX=8192
    fi
}

print_detected_hardware() {
    gb_info "GPU  : ${GPU_NAME}  (${GB_PHYS} GB VRAM → virtual ${GB_VIRT} GB T2 pool)"
    gb_info "RAM  : ${RAM_TYPE}-${RAM_SPEED_MT}  (pool ${GB_VIRT} GB, reserve ${GB_RESERVE} GB)"
    local _t3_store="/var/lib/greenboost/t3_store"
    if [[ -f "$_t3_store" ]]; then
        local _t3_gb=$(( $(stat -c%s "$_t3_store" 2>/dev/null || echo 0) / 1073741824 ))
        gb_info "NVMe : T3 backing ${_t3_gb} GB"
    else
        gb_info "NVMe : T3 backing ${GB_NVME_POOL} GB (created on first use)"
    fi
    gb_info "CPU  : ${CPU_NAME}"
    gb_info "Ctx  : ${GB_OLLAMA_CTX} tokens"
}

# ---- Arch-specific kernel headers detection --------------------------------
_detect_kernel_headers_pkg() {
    local kver; kver=$(uname -r)
    if [[ "$kver" == *zen* ]]; then
        echo "linux-zen-headers"
    elif [[ "$kver" == *lts* ]]; then
        echo "linux-lts-headers"
    elif [[ "$kver" == *hardened* ]]; then
        echo "linux-hardened-headers"
    elif [[ "$kver" == *cachyos* ]]; then
        echo "linux-cachyos-headers"
    else
        echo "linux-headers"
    fi
}

# ---- Misc helpers ----------------------------------------------------------
need_root() {
    [[ "$(id -u)" == "0" ]] || die "Command '$1' requires root. Run: sudo $0 $1"
}

check_deps() {
    local missing=()
    for cmd in make gcc modprobe ldconfig; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    [[ ${#missing[@]} -eq 0 ]] || die "Missing tools: ${missing[*]}"
}

# ---- do_purge - remove ALL previously installed GreenBoost artifacts --------
# Internal helper (no root check - callers must ensure root).
# Called by cmd_uninstall and cmd_full_install.
do_purge() {
    # restart_after=1 → restart stopped services at the end (cmd_uninstall).
    # restart_after=0 → leave them stopped (cmd_full_install handles restart after fresh install).
    local restart_after="${1:-0}"

    # 1. Stop services that hold /dev/greenboost open (prevents rmmod EBUSY).
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
    local _ko_removed=0
    while IFS= read -r _ko_file; do
        rm -f "$_ko_file" && (( _ko_removed++ )) || true
    done < <(find /lib/modules -name "greenboost.ko" 2>/dev/null)
    if command -v dkms &>/dev/null; then
        dkms remove greenboost/"${GB_VERSION}" --all &>/dev/null || true
    fi
    { depmod -a 2>/dev/null || true; } &
    gb_spin $! "Rebuilding module dependency tree..."

    # 3. Remove GreenBoost entries from /etc/ld.so.preload FIRST.
    if [[ -f /etc/ld.so.preload ]] && grep -q "libgreenboost" /etc/ld.so.preload; then
        sed -i '/libgreenboost/d' /etc/ld.so.preload
        [[ -s /etc/ld.so.preload ]] || rm -f /etc/ld.so.preload
        gb_ok "Removed from /etc/ld.so.preload"
    fi

    # 4. Remove CUDA shim, LD_AUDIT library, Vulkan layer
    local _libs_removed=0
    for lib in "$SHIM_DEST/$SHIM_LIB" "$SHIM_DEST/$AUDIT_LIB" "$SHIM_DEST/$VULKAN_LAYER_LIB"; do
        [[ -f "$lib" ]] && rm -f "$lib" && (( _libs_removed++ )) || true
    done
    rm -f "$VULKAN_IMPLICIT_LAYER_DIR/$VULKAN_LAYER_MANIFEST" 2>/dev/null || true
    { ldconfig 2>/dev/null || true; } &
    gb_spin $! "Refreshing dynamic linker cache..."
    [[ $_libs_removed -gt 0 ]] && gb_ok "CUDA shim + audit library + Vulkan layer removed"

    # 4b. AppArmor cleanup (Arch: optional, skip if not present)
    if [[ -f /etc/apparmor.d/abstractions/greenboost-audit ]]; then
        rm -f /etc/apparmor.d/abstractions/greenboost-audit
        local base_abs="/etc/apparmor.d/abstractions/base"
        [[ -f "$base_abs" ]] && sed -i '/greenboost-audit/d' "$base_abs" || true
        find /etc/apparmor.d/ -maxdepth 1 -type f -exec grep -l "greenboost-audit" {} \; \
            | xargs -I{} sed -i '/greenboost-audit/d' {} 2>/dev/null || true
        apparmor_parser -r /etc/apparmor.d/ 2>/dev/null || true
        gb_ok "AppArmor: reverted GreenBoost rules"
    fi

    # 5. Remove static config files
    local _cfg_removed=0
    for f in \
        /etc/modprobe.d/greenboost.conf \
        /etc/modprobe.d/greenboost.conf.bak \
        /etc/profile.d/greenboost.sh \
        /usr/local/bin/greenboost \
        /etc/modules-load.d/greenboost.conf \
        /etc/udev/rules.d/99-greenboost.rules \
        /etc/udev/rules.d/99-nvme-greenboost.rules \
        /etc/sysctl.d/99-greenboost.conf \
        /etc/sysctl.d/99-zzz-greenboost.conf \
        /etc/sysfs.d/greenboost-hugepages.conf; do
        [[ -f "$f" ]] && rm -f "$f" && (( _cfg_removed++ )) || true
    done
    udevadm control --reload-rules 2>/dev/null || true
    [[ $_cfg_removed -gt 0 ]] && gb_ok "Config files removed ($_cfg_removed files)"

    # 6. Disable + remove ALL GreenBoost systemd services - generic glob.
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
    # TurboQuant daemon (optional install - clean up if present)
    systemctl disable --now greenboost-turboquant.service 2>/dev/null || true
    rm -f /usr/local/bin/greenboost-turboquant \
          /usr/local/lib/libgreenboost_tq.so \
          /etc/systemd/system/greenboost-turboquant.service
    rm -rf /usr/local/lib/greenboost/cmd/greenboost-turboquant

    # Daemon scripts and CLI wrappers
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

    # 8b. Remove /swap_nvme.img if present - created by older GreenBoost versions.
    #     Only this specific file is touched; the system's regular swap is never modified.
    if [[ -f /swap_nvme.img ]]; then
        swapoff /swap_nvme.img 2>/dev/null || true
        rm -f /swap_nvme.img
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
    if [[ -f /var/lib/greenboost/t3_store ]]; then
        rm -f /var/lib/greenboost/t3_store
        gb_ok "Removed T3 backing file (/var/lib/greenboost/t3_store)"
    fi
    rmdir /var/lib/greenboost 2>/dev/null || true

    # 9. Unload kernel module - done last so all consumers are gone.
    if lsmod | grep -q "^${DRIVER_NAME} "; then
        if rmmod "$DRIVER_NAME" 2>/dev/null; then
            gb_ok "Kernel module unloaded"
        else
            fuser -k /dev/greenboost 2>/dev/null || true
            sleep 1
            if rmmod "$DRIVER_NAME" 2>/dev/null; then
                gb_ok "Kernel module unloaded (after retry)"
            else
                gb_warn_ui "rmmod failed - module still loaded."
                gb_warn_ui "Run manually: sudo rmmod ${DRIVER_NAME}"
            fi
        fi
    fi

    # 10. Restart services that were stopped before the purge.
    #     Only done on standalone uninstall (restart_after=1).
    #     On full-install (restart_after=0) services stay stopped - cmd_full_install
    #     restarts them once at the very end, after the fresh install completes.
    if [[ $restart_after -eq 1 ]]; then
        for svc in $GB_STOPPED_SERVICES; do
            systemctl start "$svc" 2>/dev/null \
                && gb_ok "$svc restarted" \
                || gb_warn_ui "$svc failed to restart - check: journalctl -u $svc"
        done
    fi
}

cmd_uninstall() {
    need_root uninstall
    gb_header
    gb_info "Removing all GreenBoost artifacts from this system..."
    echo ""
    GB_STOPPED_SERVICES=""
    do_purge 1
    echo ""
    gb_ok "GreenBoost uninstalled cleanly."
}

# ---- Build / Install -------------------------------------------------------
cmd_build() {
    local _build_log
    _build_log=$(mktemp /tmp/gb_build_XXXXXX.log)
    make -C "$MODULE_DIR" all &>"$_build_log" &
    gb_spin $! "Compiling kernel module + CUDA shim..."
    if ! wait $!; then
        cat "$_build_log" >&2
        rm -f "$_build_log"
        die "Build failed - see output above"
    fi
    rm -f "$_build_log"
    gb_ok "Build complete  (greenboost.ko · ${SHIM_LIB} · ${AUDIT_LIB} · ${VULKAN_LAYER_LIB})"
}

cmd_install() {
    need_root install
    detect_hardware
    check_deps
    cmd_build

    make -C "$MODULE_DIR" install || die "Module install failed"
    gb_ok "Kernel module installed"

    cp "$MODULE_DIR/$SHIM_LIB" "$SHIM_DEST/"
    [[ -f "$MODULE_DIR/$AUDIT_LIB"        ]] && cp "$MODULE_DIR/$AUDIT_LIB"        "$SHIM_DEST/"
    [[ -f "$MODULE_DIR/$VULKAN_LAYER_LIB" ]] && cp "$MODULE_DIR/$VULKAN_LAYER_LIB" "$SHIM_DEST/"
    ldconfig
    mkdir -p /run/greenboost
    chmod 0775 /run/greenboost
    chgrp ollama /run/greenboost 2>/dev/null || true
    gb_ok "CUDA shim + libraries installed to $SHIM_DEST/"

    cat > /etc/modprobe.d/greenboost.conf << MODEOF
# GreenBoost - cuda memory pool (auto-configured for detected hardware)
# GPU   : ${GPU_NAME}  (${GB_PHYS} GB VRAM)
# RAM   : ${RAM_TYPE}-${RAM_SPEED_MT}  (pool ${GB_VIRT} GB, reserve ${GB_RESERVE} GB)
# NVMe  : swap ${GB_NVME_SWAP} GB
options greenboost physical_vram_gb=${GB_PHYS} virtual_vram_gb=${GB_VIRT} safety_reserve_gb=${GB_RESERVE} nvme_pool_gb=${GB_NVME_POOL} t3_max_gb=${GB_NVME_POOL} t3_file_path=/var/lib/greenboost/t3_store pcores_max_cpu=${GB_PCORES_MAX} golden_cpu_min=${GB_GOLDEN_MIN} golden_cpu_max=${GB_GOLDEN_MAX} ecores_only=${GB_PCORES_ONLY}
MODEOF

    cat > /etc/profile.d/greenboost.sh << PROFEOF
# GreenBoost - shell helpers
export GREENBOOST_SHIM="$SHIM_DEST/$SHIM_LIB"
greenboost-run() { LD_PRELOAD="\$GREENBOOST_SHIM" "\$@"; }
export -f greenboost-run
PROFEOF

    cat > /usr/local/bin/greenboost-run << WRAPEOF
#!/usr/bin/env bash
LD_PRELOAD="$SHIM_DEST/$SHIM_LIB" "\$@"
WRAPEOF
    chmod +x /usr/local/bin/greenboost-run
    gb_ok "Installation complete"
}

# ---- Load / Unload ---------------------------------------------------------
cmd_load() {
    need_root load
    detect_hardware

    if [[ -n "${GB_PROFILE_FILE:-}" ]]; then
        resolve_profile "$GB_PROFILE_FILE"
    elif [[ -f "$GB_ACTIVE_PROFILE_LINK" ]]; then
        load_profile_values "$GB_ACTIVE_PROFILE_LINK"
        [[ -n "$PROF_PHYS"        ]] && GB_PHYS=$PROF_PHYS
        [[ -n "$PROF_VIRT"        ]] && GB_VIRT=$PROF_VIRT
        [[ -n "$PROF_RESERVE"     ]] && GB_RESERVE=$PROF_RESERVE
        [[ -n "$PROF_NVME_SWAP"   ]] && GB_NVME_SWAP=$PROF_NVME_SWAP
        [[ -n "$PROF_NVME_POOL"   ]] && GB_NVME_POOL=$PROF_NVME_POOL
        [[ -n "$PROF_PCORES_ONLY" ]] && GB_PCORES_ONLY=$PROF_PCORES_ONLY
        [[ -n "$PROF_OLLAMA_CTX"  ]] && GB_OLLAMA_CTX=$PROF_OLLAMA_CTX
    fi

    local phys="${GPU_PHYS_GB:-${GB_PHYS}}"
    local virt="${VIRT_VRAM_GB:-${GB_VIRT}}"
    local res="${RESERVE_GB:-${GB_RESERVE}}"
    local nvme_sw="${NVME_SWAP_GB:-${GB_NVME_SWAP}}"
    local nvme_pool="${NVME_POOL_GB:-${GB_NVME_POOL}}"

    if lsmod | grep -q "^${DRIVER_NAME} "; then
        warn "Module already loaded - reloading..."
        for svc in ollama llama-server; do
            systemctl is-active --quiet "$svc" 2>/dev/null && \
                systemctl stop "$svc" 2>/dev/null || true
        done
        [[ -e /dev/greenboost ]] && { fuser -k /dev/greenboost 2>/dev/null || true; sleep 0.5; }
        rmmod "$DRIVER_NAME" || die "Failed to unload existing module"
    fi

    local ko="$MODULE_DIR/greenboost.ko"
    [[ -f "$ko" ]] || die "greenboost.ko not found - run: make  or  $0 build"

    insmod "$ko" \
        physical_vram_gb="$phys"  \
        virtual_vram_gb="$virt"   \
        safety_reserve_gb="$res"  \
        nvme_swap_gb="$nvme_sw"   \
        nvme_pool_gb="$nvme_pool" \
        pcores_max_cpu="${GB_PCORES_MAX}" \
        golden_cpu_min="${GB_GOLDEN_MIN}" \
        golden_cpu_max="${GB_GOLDEN_MAX}" \
        ecores_only="${GB_PCORES_ONLY}"   \
        active_profile_name="${PROF_NAME:-autodetect}" \
        || die "insmod failed - check: dmesg | tail -20"

    gb_ok "GreenBoost loaded - cuda memory pool active!"
    gb_info "  T1 ${GPU_NAME:-GPU} VRAM : ${phys} GB   [hot layers]"
    gb_info "  T2 System DDR pool      : ${virt} GB  (~${PCIE_BW_GBS:-auto} GB/s PCIe)  [cold layers]"
    gb_info "  T3 NVMe swap            : ${nvme_sw} GB  [frozen pages]"
    gb_info "  Combined view           : $(( phys + virt + nvme_sw )) GB total capacity"
}

cmd_unload() {
    need_root unload
    if lsmod | grep -q "^${DRIVER_NAME} "; then
        for svc in ollama llama-server; do
            systemctl is-active --quiet "$svc" 2>/dev/null && \
                systemctl stop "$svc" 2>/dev/null || true
        done
        [[ -e /dev/greenboost ]] && { fuser -k /dev/greenboost 2>/dev/null || true; sleep 0.5; }
        rmmod "$DRIVER_NAME" && gb_ok "Kernel module unloaded" || die "rmmod failed"
    else
        gb_warn_ui "Module not loaded"
    fi
}

# ---- OS dependencies (Arch/pacman) -----------------------------------------
cmd_install_build_deps() {
    need_root install-deps

    local headers_pkg; headers_pkg=$(_detect_kernel_headers_pkg)

    printf "  ${C_CYAN}❯${C_RESET}  ${C_DIM}Updating package database...${C_RESET}"
    pacman -Sy --noconfirm -q 2>/dev/null || true
    printf "\r%*s\r" "$(tput cols 2>/dev/null || echo 80)" ""

    # Minimal packages required to build and DKMS-register the kernel module.
    local _groups=( "Build tools + kernel headers" "DKMS + io_uring" "Microcode" )
    local _total=${#_groups[@]} _idx=0

    _dep_bar() {
        local _i=$1 _name="$2"
        local _pct=$(( _i * 100 / _total ))
        local _filled=$(( _pct * 40 / 100 )) _empty=$(( 40 - _pct * 40 / 100 ))
        printf "\r  ${C_GRAY}[%d/%d]${C_RESET} ${C_LIME}%s${C_GRAY}%s${C_RESET} %3d%%  ${C_DIM}%-32s${C_RESET}" \
            "$_i" "$_total" \
            "$(printf '█%.0s' $(seq 1 "$_filled" 2>/dev/null || true))" \
            "$(printf '░%.0s' $(seq 1 "$_empty"  2>/dev/null || true))" \
            "$_pct" "$_name"
    }

    (( _idx++ )); _dep_bar $_idx "${_groups[$_idx-1]}"
    pacman -S --noconfirm --needed -q \
        base-devel cmake "${headers_pkg}" git curl wget pkg-config sysfsutils kmod 2>/dev/null

    (( _idx++ )); _dep_bar $_idx "${_groups[$_idx-1]}"
    pacman -S --noconfirm --needed -q dkms liburing 2>/dev/null || true

    (( _idx++ )); _dep_bar $_idx "${_groups[$_idx-1]}"
    local _cpu_vendor
    _cpu_vendor=$(grep -m1 "vendor_id" /proc/cpuinfo | awk '{print $3}')
    if [[ "$_cpu_vendor" == "GenuineIntel" ]]; then
        pacman -S --noconfirm --needed -q intel-ucode 2>/dev/null || true
    elif [[ "$_cpu_vendor" == "AuthenticAMD" ]]; then
        pacman -S --noconfirm --needed -q amd-ucode 2>/dev/null || true
    fi

    printf "\r%*s\r" "$(tput cols 2>/dev/null || echo 80)" ""

    # Ensure cpuid module loads at boot
    if ! grep -q cpuid /etc/modules-load.d/*.conf 2>/dev/null; then
        echo cpuid > /etc/modules-load.d/ai-workstation.conf
    fi

    gb_ok "Build dependencies installed"
}

cmd_install_optional_pkgs() {
    need_root install-optional-pkgs

    # Optional AI/compute libraries and tools. Not required to build the kernel
    # module - only install these for a full AI workstation setup.
    local _groups=(
        "gcc-multilib (32-bit)"
        "Python runtime"
        "Monitoring tools"
        "AI compute libraries"
    )
    local _total=${#_groups[@]} _idx=0

    _dep_bar() {
        local _i=$1 _name="$2"
        local _pct=$(( _i * 100 / _total ))
        local _filled=$(( _pct * 40 / 100 )) _empty=$(( 40 - _pct * 40 / 100 ))
        printf "\r  ${C_GRAY}[%d/%d]${C_RESET} ${C_LIME}%s${C_GRAY}%s${C_RESET} %3d%%  ${C_DIM}%-32s${C_RESET}" \
            "$_i" "$_total" \
            "$(printf '█%.0s' $(seq 1 "$_filled" 2>/dev/null || true))" \
            "$(printf '░%.0s' $(seq 1 "$_empty"  2>/dev/null || true))" \
            "$_pct" "$_name"
    }

    (( _idx++ )); _dep_bar $_idx "${_groups[$_idx-1]}"
    pacman -S --noconfirm --needed -q multilib-devel lib32-gcc-libs 2>/dev/null || true

    (( _idx++ )); _dep_bar $_idx "${_groups[$_idx-1]}"
    pacman -S --noconfirm --needed -q python python-pip python-virtualenv 2>/dev/null

    (( _idx++ )); _dep_bar $_idx "${_groups[$_idx-1]}"
    pacman -S --noconfirm --needed -q cpupower nvtop 2>/dev/null || true

    (( _idx++ )); _dep_bar $_idx "${_groups[$_idx-1]}"
    pacman -S --noconfirm --needed -q \
        openblas hwloc numactl openmp opencl-icd-loader cuda 2>/dev/null || true

    printf "\r%*s\r" "$(tput cols 2>/dev/null || echo 80)" ""

    gb_ok "Optional AI/compute libraries installed"
    gb_info "Note: NVIDIA driver (nvidia/nvidia-dkms) and CUDA must be installed separately"
}

# cmd_install_deps - installs all dependencies (build + optional).
# Called when running 'install-deps' directly; full-install uses
# cmd_install_build_deps (always) + cmd_install_optional_pkgs (gated).
cmd_install_deps() {
    cmd_install_build_deps
    cmd_install_optional_pkgs
}

# ---- NVMe swap setup -------------------------------------------------------
cmd_setup_swap() {
    # GreenBoost uses the system's existing swap for T3 - no swap file creation.
    # This command reports detected system swap and the resulting T3 configuration.
    detect_hardware 2>/dev/null
    gb_info "T3 uses system swap - GreenBoost does not create swap files."
    echo ""
    if swapon --show --noheadings 2>/dev/null | grep -q .; then
        gb_ok "System swap detected:"
        swapon --show | sed 's/^/  /'
        echo ""
        gb_ok "T3 (nvme_swap_gb) set to: ${GB_NVME_SWAP:-0} GB"
    else
        gb_warn_ui "No active system swap detected. T3 is disabled (nvme_swap_gb=0)."
        gb_info "To enable T3: configure a swap partition or swap file via your OS tools"
        gb_info "  (e.g. /etc/fstab with a swap partition, or fallocate + mkswap + swapon)"
        gb_info "  then re-run 'greenboost full-install' to update the kernel module parameters."
    fi
}

# ---- System configs --------------------------------------------------------
cmd_install_sys_configs() {
    need_root install-sys-configs
    detect_hardware

    # Ollama drop-in
    local dropin_dir="/etc/systemd/system/ollama.service.d"
    mkdir -p "$dropin_dir"
    cat > "$dropin_dir/99-greenboost.conf" << DROPIN
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
DROPIN
    gb_ok "Ollama drop-in written: $dropin_dir/99-greenboost.conf"
    gb_ok "Ollama context cap set to ${GB_OLLAMA_CTX} tokens (T1: ${GB_PHYS} GB, T2: ${GB_VIRT} GB)"

    # GreenBoost no longer writes to /etc/ld.so.preload , doing so loads the
    # CUDA/audit interposers into every process including systemd PID 1 and
    # freezes boot ("Failed to load libmount.so").  Injection is per-process
    # via the systemd drop-in written above and the greenboost-run* wrappers.
    # Scrub any stale entry left by an older install.
    if [[ -f /etc/ld.so.preload ]]; then
        sed -i '/libgreenboost/d;/greenboost/d' /etc/ld.so.preload
        [[ -s /etc/ld.so.preload ]] || rm -f /etc/ld.so.preload
    fi

    # udev rules
    cat > /etc/udev/rules.d/99-greenboost.rules << UDEV
SUBSYSTEM=="greenboost", MODE="0660", GROUP="video"
UDEV
    cat > /etc/udev/rules.d/99-nvme-greenboost.rules << NVME
# GreenBoost NVMe tuning - auto-detected PCIe bandwidth
ACTION=="add|change", KERNEL=="nvme[0-9]n[0-9]", ATTR{queue/scheduler}="none"
ACTION=="add|change", KERNEL=="nvme[0-9]n[0-9]", ATTR{queue/read_ahead_kb}="4096"
ACTION=="add|change", KERNEL=="nvme[0-9]n[0-9]", ATTR{queue/nr_requests}="1023"
NVME
    udevadm control --reload-rules 2>/dev/null || true
    gb_ok "udev rules installed"

    # sysfs - THP
    mkdir -p /etc/sysfs.d/
    echo "kernel/mm/transparent_hugepage/enabled = always" \
        > /etc/sysfs.d/greenboost-hugepages.conf
    echo always > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
    gb_ok "THP configured"

    # Gaming PAM limits - allow video group to nice -5 (game process elevation)
    mkdir -p /etc/security/limits.d
    cat > /etc/security/limits.d/99-greenboost-gaming.conf << 'LIMITSEOF'
# GreenBoost gaming process priority - allow video group members to elevate games
# to nice -5 so the game process runs above nice-0 background tasks.
# The GreenBoost Proton wrapper calls os.setpriority(PRIO_PROCESS, 0, -5) which
# requires the effective nice ceiling set here.
@video  hard  nice  -5
@video  soft  nice  -5
LIMITSEOF
    gb_ok "Gaming PAM limits: /etc/security/limits.d/99-greenboost-gaming.conf"

    # Vulkan layer
    cmd_install_vulkan_layer

    systemctl daemon-reload 2>/dev/null || true
    gb_ok "System configs installed"

}

cmd_install_vulkan_layer() {
    [[ -f "$MODULE_DIR/$VULKAN_LAYER_LIB" ]] || { gb_warn_ui "Vulkan layer .so not built - skipping"; return 0; }
    mkdir -p "$VULKAN_IMPLICIT_LAYER_DIR"
    cp "$MODULE_DIR/$VULKAN_LAYER_LIB" "$SHIM_DEST/"
    local manifest="$MODULE_DIR/VkLayer_greenboost.json"
    [[ -f "$manifest" ]] && cp "$manifest" "$VULKAN_IMPLICIT_LAYER_DIR/" || true
    gb_ok "Vulkan layer installed"
}

# ---- Tuning ----------------------------------------------------------------
cmd_tune() {
    need_root tune
    detect_hardware

    gb_info "Tuning workstation for GreenBoost / LLM workloads..."
    gb_info "Hardware: ${GPU_NAME:-auto-detected} | ${CPU_NAME:-auto-detected}"
    echo ""

    # CPU governor
    if command -v cpupower &>/dev/null; then
        cpupower frequency-set -g performance &>/dev/null && \
            gb_ok "CPU governor: performance (cpupower)" || \
            gb_warn_ui "cpupower failed - check cpufreq driver"
    else
        local changed=0
        for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            [[ -w "$gov" ]] && echo performance > "$gov" && changed=1
        done
        [[ $changed -eq 1 ]] && gb_ok "CPU governor: performance" || \
            gb_warn_ui "Could not set CPU governor"
    fi

    # NVMe scheduler
    for sched in /sys/block/nvme*/queue/scheduler; do
        [[ -w "$sched" ]] && echo none > "$sched" 2>/dev/null || true
    done
    gb_ok "NVMe scheduler: none"

    # NVMe read-ahead
    for ra in /sys/block/nvme*/queue/read_ahead_kb; do
        [[ -w "$ra" ]] && echo 4096 > "$ra"
    done
    gb_ok "NVMe read_ahead: 4096 KB"

    # NVMe nr_requests
    for nr in /sys/block/nvme*/queue/nr_requests; do
        [[ -w "$nr" ]] && echo 1023 > "$nr"
    done
    gb_ok "NVMe nr_requests: 1023"

    # THP
    echo always > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
    gb_ok "THP: always"

    # sysctl
    sysctl -qw vm.swappiness=10
    sysctl -qw vm.dirty_ratio=40
    sysctl -qw vm.dirty_background_ratio=10
    gb_ok "sysctl: vm.swappiness=10, dirty_ratio=40"
}

cmd_tune_sysctl() {
    need_root tune-sysctl
    cat > /etc/sysctl.d/99-zzz-greenboost.conf << SYSCTL
# GreenBoost - compute-optimized sysctl
vm.swappiness = 10
vm.dirty_ratio = 40
vm.dirty_background_ratio = 10
kernel.sched_migration_cost_ns = 5000000
kernel.sched_min_granularity_ns = 10000000
kernel.sched_wakeup_granularity_ns = 15000000
SYSCTL
    sysctl --system &>/dev/null || true
    gb_ok "Persistent sysctl conf written: /etc/sysctl.d/99-zzz-greenboost.conf"
}

cmd_tune_grub() {
    need_root tune-grub
    local grub_file="/etc/default/grub"
    [[ -f "$grub_file" ]] || { gb_warn_ui "GRUB config not found - skipping"; return 0; }

    local kver; kver=$(uname -r)
    gb_info "Validating GRUB flags for detected hardware"
    gb_info "Kernel: $kver"

    cp "$grub_file" "${grub_file}.bak.$(date +%Y%m%d_%H%M%S)"

    local current_line
    current_line=$(grep '^GRUB_CMDLINE_LINUX_DEFAULT=' "$grub_file" | head -1)
    local current_args; current_args=$(echo "$current_line" | sed 's/^GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/\1/')

    local new_args="$current_args"
    local _add=()

    echo "$new_args" | grep -q "nvidia-drm.modeset=1" || _add+=("nvidia-drm.modeset=1")
    echo "$new_args" | grep -q "iommu=pt"             || _add+=("iommu=pt")
    echo "$new_args" | grep -q "numa_balancing=disable" || _add+=("numa_balancing=disable")
    echo "$new_args" | grep -q "workqueue.power_efficient=0" || _add+=("workqueue.power_efficient=0")

    if grep -qi "intel" /proc/cpuinfo 2>/dev/null; then
        echo "$new_args" | grep -q "intel_iommu=on" || _add+=("intel_iommu=on,igfx_off")
    elif grep -qi "amd" /proc/cpuinfo 2>/dev/null; then
        echo "$new_args" | grep -q "amd_iommu=pt" || _add+=("amd_iommu=pt")
    fi

    if [[ ${#_add[@]} -gt 0 ]]; then
        new_args="$new_args ${_add[*]}"
        sed -i "s|^GRUB_CMDLINE_LINUX_DEFAULT=.*|GRUB_CMDLINE_LINUX_DEFAULT=\"${new_args}\"|" "$grub_file"
        grub-mkconfig -o /boot/grub/grub.cfg &>/dev/null || \
            gb_warn_ui "grub-mkconfig failed - run manually"
        gb_ok "GRUB updated: added ${_add[*]}"
    else
        gb_ok "GRUB: no changes needed"
    fi
}

cmd_tune_libs() {
    need_root tune-libs
    gb_info "Installing missing AI/compute libraries..."
    pacman -S --noconfirm --needed \
        openblas blas lapack \
        hwloc numactl openmp \
        opencl-icd-loader clblast 2>/dev/null || true
    gb_ok "AI/compute libraries installed"
}

cmd_tune_all() {
    cmd_tune
    cmd_tune_grub
    cmd_tune_sysctl
    cmd_tune_libs
    gb_ok "All tuning steps complete"
}

# ---- Status ----------------------------------------------------------------
cmd_status() {
    local main_script="$MODULE_DIR/greenboost_setup.sh"
    if [[ -x "$main_script" ]]; then
        exec "$main_script" status "${@}"
    fi
    # Fallback: raw sysfs output when Ubuntu script is unavailable
    gb_header
    local pool_f="/sys/class/greenboost/greenboost/status"
    if lsmod | grep -q "^${DRIVER_NAME} "; then
        gb_ok "Kernel module loaded"
    else
        gb_fail "Kernel module not loaded"
    fi
    if [[ -r "$pool_f" ]]; then
        echo ""
        cat "$pool_f" | sed 's/^/  /'
    fi
    echo ""
    dmesg | grep greenboost | tail -8 | sed 's/^/  /'
}

# ---- Benchmark -------------------------------------------------------------
cmd_benchmark() {
    local bench_py="$MODULE_DIR/tools/gb_workstation_bench.py"
    [[ -f "$bench_py" ]] || die "Benchmark script not found: $bench_py"
    python3 "$bench_py" "$@"
}

# ---- Gaming mode -----------------------------------------------------------
# cmd_gaming_mode - stop/start Ollama to free T2 DDR before/after gaming.

cmd_t3_memory() {
    local main_script="$(dirname "$(realpath "$0")")/greenboost_setup.sh"
    [[ -f "$main_script" ]] || die "Main setup script not found: $main_script"
    exec bash "$main_script" t3-memory "${1:-}"
}

# ---- Clean memory ----------------------------------------------------------
cmd_clean_memory() {
    need_root "clean memory"
    sync
    echo 3 > /proc/sys/vm/drop_caches
    gb_ok "Page cache, dentries and inodes dropped"
    swapon --show
}

# ---- Show commands ---------------------------------------------------------
cmd_show_commands() {
    local cmds_file="$MODULE_DIR/GREENBOOST_COMMANDS.md"
    if [[ -f "$cmds_file" ]]; then
        cat "$cmds_file"
    else
        gb_info "GREENBOOST_COMMANDS.md not found - run: $0 help"
    fi
}

# ---- Help ------------------------------------------------------------------
cmd_help() {
    echo ""
    echo -e "${C_VIOLET}${C_BOLD}GreenBoost v${GB_VERSION} - CUDA Memory & Compute Orchestrator for NVIDIA GPUs (Arch Linux)${C_RESET}"
    echo ""
    echo "USAGE:  sudo ./greenboost_setup_arch.sh <command>"
    echo ""
    echo "COMMANDS:"
    echo "  full-install         Complete first-time setup (auto-detects hardware)"
    echo "  install              Build + install module + CUDA shim"
    echo "  uninstall            Remove module + all config files"
    echo "  build                Build only (no system install)"
    echo "  load                 Load module with detected parameters"
    echo "  unload               Unload module"
    echo "  install-sys-configs  Install Ollama env, udev rules, hugepages, sysctl"
    echo "  install-deps         Install Arch packages (build tools, headers, CUDA)"
    echo "  setup-swap [GB]      Create/activate NVMe swap file"
    echo "  tune                 Tune system for LLM workloads"
    echo "  tune-grub            Add GRUB boot parameters"
    echo "  tune-sysctl          Write persistent sysctl conf"
    echo "  tune-libs            Install AI/compute libraries"
    echo "  tune-all             Run all tune-* commands"
    echo "  status               Show pool info + swap + kernel messages"
    echo "  benchmark            Measure T1/T2/T3 bandwidth"
    echo "  profile [sub]        Manage hardware profiles"
    echo "  clean memory         Drop page cache"
    echo "  show-commands        Full command reference"
    echo "  wizard               Open interactive menu"
    echo "  help                 Show this help"
}

# ---- Profile stubs (delegate to main script profile commands) ---------------
cmd_profile() {
    local sub="${1:-show}"
    local main_script="$MODULE_DIR/greenboost_setup.sh"
    if [[ -x "$main_script" ]]; then
        exec "$main_script" profile "$@"
    else
        gb_warn_ui "Profile management requires greenboost_setup.sh"
    fi
}

cmd_profile_wizard() { cmd_profile wizard; }

# ---- Steam / MangoHud (delegate to main script) ----------------------------


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
    dmesg -c > /dev/null 2>&1 || true
    journalctl --rotate > /dev/null 2>&1 || true
    journalctl --vacuum-time=1s > /dev/null 2>&1 || true
    rm -rf /var/log/greenboost/* 2>/dev/null || true
    rm -f /tmp/greenboost*.log 2>/dev/null || true
    local _gbproton_dir="$HOME/.local/share/greenboost/proton-logs"
    rm -f "$_gbproton_dir"/steam-*.log 2>/dev/null || true
    rm -f "$HOME"/steam-*.log 2>/dev/null || true
    if command -v coredumpctl &>/dev/null; then
        coredumpctl delete wine64 2>/dev/null || true
        coredumpctl delete wineserver 2>/dev/null || true
        coredumpctl delete winedevice 2>/dev/null || true
    fi
    gb_ok "All GreenBoost logs (dmesg, journal, /var/log/greenboost, Proton, Wine coredumps) cleared."
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

# Backward-compat alias
cmd_clean_logs() { cmd_clear_logs; }

# Focused proton/inference log views and proton/mangohud uninstall - delegate to Ubuntu script

# ---- Wizard (interactive menu) ---------------------------------------------
cmd_wizard() {
    while true; do
        clear
        gb_header
        echo -e "  ${C_DIM}${C_GRAY}GreenBoost was built to improve local AI inference by orchestrating a CUDA memory pool.${C_RESET}"
        echo -e "  ${C_DIM}${C_GRAY}The Vulkan/gaming layer is a secondary byproduct.${C_RESET}"
        echo ""

        gb_section "Core"
        gb_menu_item  1  "Full install"       "First-time setup: deps + module + tune"  root
        gb_menu_item  2  "Status"             "Show cuda memory pool + system state"
        gb_menu_item  3  "Benchmark"          "Measure T1/T2/T3 bandwidth"

        gb_section "Configuration"
        gb_menu_item  4  "Profile management" "Interactive wizard: create, activate, diff profiles"

        gb_section "Maintenance"
        gb_menu_item  5  "GreenBoost Commands" "All commands reference (also: greenboost help)"
        gb_menu_item  6  "Clear logs"          "Clear dmesg, journal, Proton logs, and Wine coredumps"
        gb_menu_item  7  "Uninstall"           "Remove GreenBoost (module + all config)"        root

        gb_separator
        echo -e "  ${C_DIM}${C_GRAY}Gaming (Proton, DLSS, Vulkan/OpenGL layers, MangoHud) is now installed"
        echo -e "  ${C_DIM}${C_GRAY}separately through GreenBoost Gaming Suite:${C_RESET}"
        echo -e "  ${C_DIM}${C_CYAN}https://gitlab.com/IsolatedOctopi/greenboost_gaming${C_RESET}"

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
            1)  cmd_full_install;              gb_press_enter ;;
            2)  cmd_status ;;
            3)  cmd_benchmark;                 gb_press_enter ;;
            4)  cmd_profile_wizard ;;
            5)  cmd_show_commands;             gb_press_enter ;;
            6)  cmd_clear_logs;                gb_press_enter ;;
            7)  cmd_uninstall;                 gb_press_enter ;;
            q|Q|"") exit 0 ;;
            *) gb_warn_ui "Unknown option."; sleep 1 ;;
        esac
    done
}

# ---- Full install ----------------------------------------------------------
cmd_full_install() {
    need_root full-install
    GB_STOPPED_SERVICES=""

    # ── Mode selection - cmd_full_install always targets full system setup.
    # Only --module-only overrides this (for scripts / CI / direct invocation).
    GB_INSTALL_MODE="full"
    for _a in "$@"; do
        [[ "$_a" == "--module-only" ]] && { GB_INSTALL_MODE="module"; break; }
    done

    detect_hardware
    gb_header
    print_detected_hardware
    echo ""

    # ── SHARED PATH: kernel module (runs for both module-only and full) ──────

    # 0 - Purge any previous GreenBoost install to guarantee a clean slate
    gb_step 0 5 "Purging previous GreenBoost installation (if any)..."
    do_purge 0
    gb_ok "Previous installation purged"

    # 1 - Build dependencies (minimal - just what's needed to compile the module)
    gb_step 1 5 "Installing Arch Linux build dependencies..."
    cmd_install_build_deps
    gb_ok "Build dependencies installed"

    # 2 - Build + install kernel module + CUDA shim
    gb_step 2 5 "Building and installing kernel module + CUDA shim..."
    cmd_install
    gb_ok "Kernel module + CUDA shim installed"

    # 3 - Load kernel module
    gb_step 3 5 "Loading kernel module..."
    cmd_load
    gb_ok "Kernel module loaded"

    if [[ "$GB_INSTALL_MODE" == "module" ]]; then
        echo ""
        gb_ok "Kernel module installed and loaded."
        gb_info "Run 'sudo ./greenboost_setup_arch.sh status' to verify."
        gb_info "For full system tuning, re-run and choose option [2]."
        echo ""
        return 0
    fi

    # ── FULL ONLY PATH: all system changes applied automatically (user consented by choosing full mode) ───

    # 4 - System configs
    gb_step 4 5 "Installing system configuration files..."
    gb_info "Applying: Ollama/inference service config (drop-ins, udev, cpu-perf)"
    cmd_install_sys_configs
    gb_ok "System configuration installed"

    # 4b - Optional AI/compute libraries
    gb_info "Applying: optional packages (cuda, openblas, python, nvtop, cpupower, multilib-devel, lib32-gcc-libs)"
    cmd_install_optional_pkgs
    gb_ok "Optional AI/compute libraries installed"

    # 5 - sysctl tuning
    gb_step 5 5 "Applying system tuning..."
    gb_info "Applying: sysctl tuning"
    cmd_tune_sysctl
    gb_ok "sysctl tuning applied"

    # 5b - GRUB boot parameters (requires reboot)
    gb_info "Applying: GRUB boot parameters (nvidia-drm.modeset, iommu, numa_balancing)"
    cmd_tune_grub
    gb_ok "GRUB updated"

    # Generate hardware profile
    local main_script="$MODULE_DIR/greenboost_setup.sh"
    if [[ ! -f "${GB_ACTIVE_PROFILE_LINK}" ]]; then
        [[ -x "$main_script" ]] && "$main_script" profile create 2>/dev/null || true
        gb_ok "Hardware profile generated"
    fi

    # 5c - Python orchestration stack + MCP + gb-synapse (delegated to the
    # main, distro-agnostic setup script — this Arch installer only ever
    # covered the kernel module + CUDA shim + system tuning above; the whole
    # Python/MCP/CLI/gb-quant/gb-dataflux/gb-synapse layer was silently
    # absent on Arch until now. Each step mirrors greenboost_setup.sh's own
    # Full Install: best-effort, never aborts the install on failure — a
    # missing/incompatible piece here should warn and let the user retry it
    # standalone, not fail the whole Arch install.
    if [[ -x "$main_script" ]]; then
        gb_info "Applying: Python orchestration stack (gb-quant/gb-dataflux/gb-synapse/CLI)"
        "$main_script" install-python \
            || gb_warn "Python file install had failures — retry: sudo ./greenboost_setup.sh install-python"
        "$main_script" install-cli \
            || gb_warn "greenboost-cli install had failures — retry: sudo ./greenboost_setup.sh install-cli"
        "$main_script" register-mcp \
            || gb_warn "MCP registration had failures — retry: greenboost register-mcp"
        if [[ "${GB_INSTALL_SYNAPSE_ENGINE:-1}" != "0" ]]; then
            "$main_script" install-synapse-engine \
                || gb_warn "gb-synapse torch engine install had failures — retry: sudo greenboost install-synapse-engine"
            "$main_script" synapse build-engine \
                || gb_warn "gb-synapse llama.cpp engine build had failures — retry: sudo greenboost synapse build-engine"
        fi
        gb_ok "Python orchestration stack + gb-synapse installed"
    else
        gb_warn "greenboost_setup.sh not found next to this script — skipping Python/MCP/gb-synapse install"
    fi

    echo ""
    gb_ok "GreenBoost v${GB_VERSION} full install complete on Arch Linux!"
    gb_info "Reboot recommended to activate GRUB parameters."
    gb_info "Then verify: greenboost status"
    gb_info "GreenBoost Proton and MangoHud are gaming tools - not part of full install."
    gb_info "Install them separately from the interactive menu (options [6]–[9])."
}

# ---- Entry point -----------------------------------------------------------
GB_PROFILE_FILE=""
_ARGS=()
_expect_profile=0
for _arg in "$@"; do
    case "$_arg" in
        --profile) _expect_profile=1 ;;
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

COMMAND="${1:-wizard}"
case "$COMMAND" in
    install)             cmd_install            ;;
    uninstall)           cmd_uninstall          ;;
    build)               cmd_build              ;;
    load)                cmd_load               ;;
    unload)              cmd_unload             ;;
    install-sys-configs) cmd_install_sys_configs  ;;
    install-vulkan-layer)cmd_install_vulkan_layer ;;
    install-deps)        cmd_install_deps         ;;
    steam-launch-guide)     cmd_steam_launch_info    ;;
    setup-swap)          cmd_setup_swap "$@"    ;;
    full-install|setup)  cmd_full_install "$@"  ;;
    module-only)         GB_INSTALL_MODE="module" cmd_full_install "--module-only" "$@" ;;
    tune)                cmd_tune               ;;
    tune-grub)           cmd_tune_grub          ;;
    tune-sysctl)         cmd_tune_sysctl        ;;
    tune-libs)           cmd_tune_libs          ;;
    tune-all)            cmd_tune_all           ;;
    status)              cmd_status             ;;
    benchmark)           cmd_benchmark          ;;
    profile)             cmd_profile "${@:2}"   ;;
     
    t3-memory)           cmd_t3_memory "${2:-}" ;;
    clean)
        case "${2:-}" in
            memory)  cmd_clean_memory ;;
            *) die "Usage: greenboost clean memory" ;;
        esac
        ;;
    clean-memory)        cmd_clean_memory       ;;
    show-commands)       cmd_show_commands      ;;
    logs)                cmd_logs               "${@:2}" ;;
     
    inference-logs)      cmd_inference_logs     "${@:2}" ;;
    clear)
        case "${2:-}" in
            logs)            cmd_clear_logs ;;
             
            inference-logs)  cmd_clear_inference_logs ;;
            *) die "Usage: greenboost clear logs|proton-logs|inference-logs" ;;
        esac
        ;;
    clean-logs)          cmd_clean_logs         ;;
     
     
    wizard)              cmd_wizard             ;;
    help|--help|-h)      cmd_help               ;;
    *) die "Unknown command: '$COMMAND'  - use: $0 help" ;;
esac

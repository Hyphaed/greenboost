#!/usr/bin/env bash
# GreenBoost v3.4 - Setup & installation script for Red Hat / Fedora / Rocky Linux
# Supports: Rocky Linux, AlmaLinux, RHEL, Fedora, CentOS Stream.
#
# Hardware-detected at runtime: CPU topology, GPU VRAM, RAM, kernel version.
# No hardware-specific values are hard-coded.
#
# USAGE:
#   sudo ./greenboost_setup_rocky.sh full-install    - full install (prompts for mode)
#   sudo ./greenboost_setup_rocky.sh module-only     - kernel module only (safe on any machine)
#   sudo ./greenboost_setup_rocky.sh install         - build + install system-wide
#   sudo ./greenboost_setup_rocky.sh uninstall       - remove module + all config
#   sudo ./greenboost_setup_rocky.sh load            - insmod with default params
#   sudo ./greenboost_setup_rocky.sh unload          - rmmod
#   sudo ./greenboost_setup_rocky.sh tune            - runtime tuning (governor, NVMe, sysctl)
#   sudo ./greenboost_setup_rocky.sh tune-grub       - GRUB/boot parameter optimization
#   sudo ./greenboost_setup_rocky.sh tune-sysctl     - consolidate + enhance sysctl (persistent)
#   sudo ./greenboost_setup_rocky.sh tune-libs       - install missing AI/compute libraries
#   sudo ./greenboost_setup_rocky.sh tune-all        - run all tune-* commands
#        ./greenboost_setup_rocky.sh status          - show pool info + system state
#        ./greenboost_setup_rocky.sh help            - show this help
#
# ENVIRONMENT (for load command - all values auto-detected at runtime):
#   GPU_PHYS_GB    physical VRAM in GB       (detected via nvidia-smi)
#   VIRT_VRAM_GB   system RAM pool size in GB (80% of total RAM)
#   RESERVE_GB     minimum free system RAM to always maintain
#   NVME_SWAP_GB=64    total NVMe swap capacity  (auto-detected; 64 GB default)
#   NVME_POOL_GB=58    GreenBoost soft cap on T3 allocations

set -euo pipefail

DRIVER_NAME="greenboost"
SHIM_LIB="libgreenboost_cuda.so"
AUDIT_LIB="libgreenboost_audit.so"
VULKAN_LAYER_LIB="libVkLayer_greenboost.so"
VULKAN_LAYER_MANIFEST="VkLayer_greenboost.json"
VULKAN_IMPLICIT_LAYER_DIR="/etc/vulkan/implicit_layer.d"
SHIM_DEST="/usr/local/lib"
MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GB_VERSION="3.4"
GB_PROFILES_DIR="/etc/greenboost/profiles"
GB_ACTIVE_PROFILE_LINK="/etc/greenboost/active_profile.md"

# Colours
RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
BLU='\033[0;34m'
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
    local title=" GreenBoost v${GB_VERSION} - CUDA Memory & Compute Orchestrator for NVIDIA GPUs (Red Hat)"
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

# ---- Hardware detection -----------------------------------------------
# Populates GB_PHYS, GB_VIRT, GB_RESERVE, GB_NVME_SWAP, GB_NVME_POOL,
# GB_PCORES_MAX, GB_GOLDEN_MIN, GB_GOLDEN_MAX, GB_PCORES_ONLY,
# RAM_TYPE, RAM_SPEED_MT, GPU_NAME, CPU_NAME, NVME_SIZE_GB,
# PCIE_GEN, PCIE_WIDTH, PCIE_BW_GBS, PCIE_MAX_GEN, PCIE_MAX_BW_GBS.
# Safe to call multiple times - idempotent.

detect_hardware() {
    # ── GPU ──────────────────────────────────────────────────────────────
    if command -v nvidia-smi &>/dev/null; then
        local gpu_mem_mib
        gpu_mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
            2>/dev/null | head -1 | tr -d ' ')
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 \
            | sed 's/^[[:space:]]*//')
        GB_PHYS=$(( ${gpu_mem_mib:-12288} / 1024 ))
        [[ $GB_PHYS -lt 1 ]] && GB_PHYS=1
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
    PCIE_GEN=0; PCIE_WIDTH="x?"; PCIE_BW_GBS=0
    PCIE_MAX_GEN=0; PCIE_MAX_BW_GBS=0
    local gpu_pci
    gpu_pci=$(ls /proc/driver/nvidia/gpus/ 2>/dev/null | head -1)
    if [[ -n "$gpu_pci" ]]; then
        local gpu_sysfs parent_sysfs
        gpu_sysfs=$(readlink -f "/sys/bus/pci/devices/$gpu_pci")
        parent_sysfs=$(readlink -f "$gpu_sysfs/..")

        _gts_to_gen() {
            local gts; gts=$(echo "$1" | grep -oP '^[0-9]+')
            if   [[ $gts -ge 64 ]]; then echo 6
            elif [[ $gts -ge 32 ]]; then echo 5
            elif [[ $gts -ge 16 ]]; then echo 4
            elif [[ $gts -ge 8  ]]; then echo 3
            elif [[ $gts -ge 5  ]]; then echo 2
            elif [[ $gts -ge 1  ]]; then echo 1
            else echo 0; fi
        }
        _pcie_bw() { echo $(( $1 * $2 * 128 / 130 / 8 )); }

        local slot_spd slot_w
        slot_spd=$(cat "$parent_sysfs/max_link_speed" 2>/dev/null)
        slot_w=$(  cat "$parent_sysfs/max_link_width"  2>/dev/null)
        if [[ -n "$slot_spd" ]]; then
            PCIE_GEN=$(_gts_to_gen "$slot_spd")
            PCIE_WIDTH="x${slot_w:-?}"
            [[ -n "$slot_w" && $slot_w -gt 0 ]] && \
                PCIE_BW_GBS=$(_pcie_bw "$(echo "$slot_spd" | grep -oP '^[0-9]+')" "$slot_w")
        fi

        local gpu_max_spd gpu_max_w
        gpu_max_spd=$(cat "$gpu_sysfs/max_link_speed" 2>/dev/null)
        gpu_max_w=$(  cat "$gpu_sysfs/max_link_width"  2>/dev/null)
        if [[ -n "$gpu_max_spd" ]]; then
            PCIE_MAX_GEN=$(_gts_to_gen "$gpu_max_spd")
            [[ -n "$gpu_max_w" && $gpu_max_w -gt 0 ]] && \
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
    # Primary: thread_siblings_list - CPU with "N-M" range = P-core (HyperThreaded);
    #          bare integer = E-core (no HT). Works on kernel 2.6+ always.
    # Secondary: core_type integer (1=P, 0=E) - used only when primary finds no split.
    # Fallback: non-hybrid path (AMD, older Intel, containers without sysfs).
    local has_ecores=0
    local -a p_cpu_list=() e_cpu_list=()
    local _n _sib _ct _f _fpath

    if [[ -d /sys/devices/system/cpu/cpu0/topology ]]; then
        for _n in $(seq 0 $(( total_cpus - 1 ))); do
            _sib=$(< "/sys/devices/system/cpu/cpu${_n}/topology/thread_siblings_list" 2>/dev/null) || continue
            if [[ "$_sib" == *"-"* ]]; then
                p_cpu_list+=("$_n")
            else
                e_cpu_list+=("$_n")
            fi
        done
    fi

    # If thread_siblings_list gave no P/E split, try core_type integer (secondary)
    if [[ ${#e_cpu_list[@]} -eq 0 ]] && compgen -G "/sys/devices/system/cpu/cpu*/topology/core_type" &>/dev/null; then
        p_cpu_list=()
        for _f in /sys/devices/system/cpu/cpu*/topology/core_type; do
            [[ -r "$_f" ]] || continue
            _ct=$(< "$_f")
            _n=${_f##*cpu}; _n=${_n%%/*}
            if   [[ "$_ct" == "1" ]]; then p_cpu_list+=("$_n")
            elif [[ "$_ct" == "0" ]]; then e_cpu_list+=("$_n")
            fi
        done
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

    # ── Ollama CTX based on available pool ───────────────────────────────
    # Heuristic: large context needs ~12GB KV cache; use 131K if pool >= 40 GB
    local total_pool=$(( GB_PHYS + GB_VIRT + GB_NVME_POOL ))
    if   [[ $total_pool -ge 40 ]]; then GB_OLLAMA_CTX=131072
    elif [[ $total_pool -ge 24 ]]; then GB_OLLAMA_CTX=65536
    elif [[ $total_pool -ge 16 ]]; then GB_OLLAMA_CTX=32768
    else                                 GB_OLLAMA_CTX=16384
    fi
}

print_detected_hardware() {
    info "Detected hardware:"
    info "  GPU   : ${GPU_NAME}  (${GB_PHYS} GB VRAM)"
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
logical_cpus: ${GB_PCORES_MAX:-0}
pcores_max_cpu: ${GB_PCORES_MAX:-0}
golden_cpu_min: ${GB_GOLDEN_MIN:-0}
golden_cpu_max: ${GB_GOLDEN_MAX:-0}

### GPU
gpu_count: 1
gpu_model: "${GPU_NAME:-Unknown}"
vram_gb: ${GB_PHYS:-0}
pcie_gen: ${PCIE_GEN:-4}

### RAM
ram_total_gb: $(($(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024))
ram_type: ${RAM_TYPE:-DDR}
ram_speed_mt: ${RAM_SPEED_MT:-0}

### Storage
nvme_capacity_gb: ${NVME_SIZE_GB:-0}

### OS
kernel_version: "$(uname -r)"

## GreenBoost Parameters

physical_vram_gb: ${GB_PHYS:-0}
virtual_vram_gb: ${GB_VIRT:-0}
safety_reserve_gb: ${GB_RESERVE:-0}
nvme_swap_gb: ${GB_NVME_SWAP:-0}
nvme_pool_gb: ${GB_NVME_POOL:-0}
use_hugepages: 1
ecores_only: ${GB_PCORES_ONLY:-0}
debug_mode: 0
tier3_backend: nvme

## Ollama / Inference Runtime

ollama_flash_attention: 1
ollama_kv_cache_type: q8_0
ollama_num_ctx: ${GB_OLLAMA_CTX:-131072}

## Profile Notes

Auto-detected profile. Run 'greenboost_setup_rocky.sh profile diff' to compare vs live hardware.
PROFILE_EOF

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
    PROF_OLLAMA_CTX=$(parse_profile_field "$file" ollama_num_ctx)
    PROF_TIER3=$(parse_profile_field "$file" tier3_backend)
    PROF_NAME=$(parse_profile_field "$file" profile_name)
    PROF_TYPE=$(parse_profile_field "$file" profile_type)
    PROF_VRAM_GB=$(parse_profile_field "$file" vram_gb)
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
    local _mem_kb; _mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}'); _mem_kb="${_mem_kb:-0}"
    local total_ram_gb=$(( _mem_kb / 1024 / 1024 ))

    # Rule: physical_vram_gb - always use detected; ignore if profile claims more
    if [[ -n "$PROF_VRAM_GB" && "$PROF_VRAM_GB" -gt "$GB_PHYS" ]]; then
        conflicts+="physical_vram_gb overridden ${PROF_VRAM_GB}→${GB_PHYS} GB (physical limit); "
    fi

    # Rule: virtual_vram_gb - use profile value if <= 90% RAM, else cap
    if [[ -n "$PROF_VIRT" ]]; then
        local max_virt=$(( total_ram_gb * 90 / 100 ))
        if [[ "$PROF_VIRT" -le "$max_virt" ]]; then
            resolved_virt=$PROF_VIRT
        else
            resolved_virt=$max_virt
            conflicts+="virtual_vram_gb capped ${PROF_VIRT}→${resolved_virt} GB (90% of ${total_ram_gb} GB RAM); "
        fi
    fi

    # Rule: safety_reserve_gb - use max(profile, 10% RAM)
    if [[ -n "$PROF_RESERVE" ]]; then
        local min_reserve=$(( total_ram_gb * 10 / 100 ))
        [[ $min_reserve -lt 8 ]] && min_reserve=8
        if [[ "$PROF_RESERVE" -ge "$min_reserve" ]]; then
            resolved_reserve=$PROF_RESERVE
        else
            resolved_reserve=$min_reserve
            conflicts+="safety_reserve_gb raised ${PROF_RESERVE}→${resolved_reserve} GB (10% RAM minimum); "
        fi
    fi

    # Rule: nvme_swap_gb - use profile if <= NVMe capacity, else cap at 90% free
    if [[ -n "$PROF_NVME_SWAP" ]]; then
        local nvme_free_gb; nvme_free_gb=$(df -BG / 2>/dev/null | awk 'NR==2{gsub("G",""); print $4}' || echo 0)
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

    # Rule: nvme_pool_gb - min(profile, nvme_swap * 0.89)
    if [[ -n "$PROF_NVME_POOL" ]]; then
        local max_pool=$(( resolved_nvme_sw * 89 / 100 ))
        if [[ "$PROF_NVME_POOL" -le "$max_pool" ]]; then
            resolved_nvme_pool=$PROF_NVME_POOL
        else
            resolved_nvme_pool=$max_pool
            conflicts+="nvme_pool_gb capped ${PROF_NVME_POOL}→${resolved_nvme_pool} GB (89% of nvme_swap); "
        fi
    fi

    # Apply resolved values back to GB_* so cmd_load() picks them up
    GB_PHYS=$resolved_phys
    GB_VIRT=$resolved_virt
    GB_RESERVE=$resolved_reserve
    GB_NVME_SWAP=$resolved_nvme_sw
    GB_NVME_POOL=$resolved_nvme_pool
    GB_PCORES_ONLY=$resolved_pcores
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
    GB_NVME_SWAP=64
    GB_NVME_POOL=58
    GB_PCORES_MAX=15
    GB_GOLDEN_MIN=4
    GB_GOLDEN_MAX=7
    GB_PCORES_ONLY=1
    GB_OLLAMA_CTX=131072
    info "Owner-workstation preset applied (i9-14900KF | RTX 5070 12GB | 64GB DDR4-3600 | PCIe 4 x16 | 4TB NVMe)"
}

# ---- Helpers -----------------------------------------------------------

need_root() {
    [[ $EUID -eq 0 ]] || die "Root required. Use: sudo $0 $1"
}

check_deps() {
    info "Checking build prerequisites..."
    command -v make >/dev/null || die "make not found (dnf groupinstall 'Development Tools')"
    command -v gcc  >/dev/null || die "gcc not found (dnf install gcc)"


    local kdir="/lib/modules/$(uname -r)/build"
    [[ -d "$kdir" ]] || die "Kernel headers not found at $kdir
    Install with: sudo apt install kernel-headers-$(uname -r)"
    info "Kernel headers : $kdir  ✓"

    if lsmod | grep -q "^nvidia "; then
        info "NVIDIA driver  : loaded  ✓"
    else
        warn "NVIDIA driver not loaded - run: sudo modprobe nvidia"
    fi

    if lsmod | grep -q "^nvidia_uvm "; then
        info "NVIDIA UVM     : loaded  ✓  (managed memory / System DDR overflow ready)"
    else
        warn "nvidia_uvm not loaded - CUDA UVM overflow unavailable"
        warn "Fix: sudo modprobe nvidia_uvm"
    fi
}

# ---- Commands ----------------------------------------------------------

cmd_install_sys_configs() {
    need_root install-sys-configs
    detect_hardware

    info "Installing GreenBoost v3.4 system configuration files..."

    # 1. Ollama service - inject GreenBoost env vars + LD_PRELOAD
    local svc="/etc/systemd/system/ollama.service"
    if [[ -f "$svc" ]]; then
        # Add environment lines if not already present
        if ! grep -q "GREENBOOST_VIRTUAL_VRAM_MB" "$svc"; then
            sed -i "/^\[Service\]/a Environment=\"OLLAMA_FLASH_ATTENTION=1\"\nEnvironment=\"OLLAMA_KV_CACHE_TYPE=q8_0\"\nEnvironment=\"OLLAMA_NUM_CTX=${GB_OLLAMA_CTX}\"\nEnvironment=\"OLLAMA_MAX_LOADED_MODELS=1\"\nEnvironment=\"OLLAMA_KEEP_ALIVE=-1\"\nEnvironment=\"GREENBOOST_VIRTUAL_VRAM_MB=$((GB_VIRT * 1024))\"\nEnvironment=\"GREENBOOST_DEBUG=0\"\nEnvironment=\"LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so\"" "$svc"
            info "Ollama service: GreenBoost env vars injected"
        else
            info "Ollama service: already configured (skip)"
        fi
        systemctl daemon-reload
        info "Ollama service: daemon-reload done"
    else
        warn "Ollama service not found at $svc - skipping"
    fi

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

    # 2b. NVMe udev rule - scheduler=none, read_ahead=4096, nr_requests=2048
    cat > /etc/udev/rules.d/99-nvme-greenboost.rules << 'UDEVEOF'
# GreenBoost v3.4 - NVMe tuning for T3 swap performance
ACTION=="add|change", KERNEL=="nvme[0-9]n[0-9]", ATTR{queue/scheduler}="none"
ACTION=="add|change", KERNEL=="nvme[0-9]n[0-9]", ATTR{queue/read_ahead_kb}="4096"
ACTION=="add|change", KERNEL=="nvme[0-9]n[0-9]", ATTR{queue/nr_requests}="2048"
UDEVEOF
    udevadm control --reload-rules && udevadm trigger || true
    info "NVMe udev rule installed: /etc/udev/rules.d/99-nvme-greenboost.rules"

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

[Install]
WantedBy=multi-user.target
CPUEOF
    systemctl daemon-reload
    warn "cpu-perf.service sets CPU governor to 'performance' on P-cores and pins inference cpuset."
    warn "This may increase power consumption on shared/laptop systems."
    if [[ -t 0 ]]; then
        read -r -p "  Enable cpu-perf.service now? [y/N] " _cpu_ans </dev/tty
        if [[ "${_cpu_ans,,}" == "y" ]]; then
            systemctl enable --now cpu-perf.service
            info "CPU governor service enabled and started"
        else
            info "cpu-perf.service installed but NOT enabled - run: sudo systemctl enable --now cpu-perf.service"
        fi
    else
        info "Non-interactive mode: cpu-perf.service installed but NOT auto-enabled"
        info "To enable: sudo systemctl enable --now cpu-perf.service"
    fi

    # 4. THP sysfs.d - transparent hugepages for compaction + THP performance
    # NOTE: gb_alloc_buf() uses alloc_pages(GFP_KERNEL|__GFP_COMP, order=9) which draws
    # from the BUDDY ALLOCATOR, NOT the HugeTLB pool.  Pre-allocating HugeTLB pages
    # (vm.nr_hugepages=26112) locks 51 GB in the HugeTLB pool, leaving <12 GB free RAM,
    # which triggers the OOM guard and makes T2 unavailable.  Keep nr_hugepages=0.
    mkdir -p /etc/sysfs.d
    cat > /etc/sysfs.d/greenboost-hugepages.conf << 'HPEOF'
# GreenBoost v3.4 - THP config (no HugeTLB pre-allocation: gb_alloc_buf uses buddy allocator)
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

    # 5. VM sysctl - reduce swap pressure, tune write-back
    cat > /etc/sysctl.d/99-greenboost.conf << 'SYSCTLEOF'
# GreenBoost v3.4 - VM tuning for cuda memory pool
vm.swappiness = 5
vm.dirty_ratio = 20
vm.dirty_background_ratio = 5
SYSCTLEOF
    sysctl -p /etc/sysctl.d/99-greenboost.conf 2>&1 | sed 's/^/  /'
    info "sysctl conf installed: /etc/sysctl.d/99-greenboost.conf"

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

    # Gaming / Vulkan shader boost service
    cmd_install_shader_boost

    # Vulkan implicit layer (VK_LAYER_GREENBOOST_memory)
    cmd_install_vulkan_layer

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
        gb_info "TurboQuant daemon source not found at $_tq_src - skipping"
    fi

    echo ""
    info "System config installation complete."
    warn "Restart Ollama to pick up new env vars: sudo systemctl restart ollama"
}

cmd_install_shader_boost() {
    local boost_bin="/usr/local/bin/greenboost-shader-boost"
    local boost_svc="/etc/systemd/system/greenboost-shader-boost.service"

    cat > "$boost_bin" << 'SHADERBOOSTEOF'
#!/bin/bash
# GreenBoost Vulkan Shader Compilation Boost Daemon
# Monitors for fossilize_replay workers (Steam/Proton pipeline pre-compilation)
# and boosts them: renice -5, ionice best-effort/0, pin to P-cores.
# Managed by greenboost-shader-boost.service (runs as root).

pcores_max=0
for f in /sys/devices/system/cpu/cpu*/topology/core_type; do
    [[ -r "$f" ]] || continue
    ct=$(< "$f")
    [[ "$ct" == "1" ]] || continue
    n=${f##*cpu}; n=${n%%/*}
    (( n > pcores_max )) && pcores_max=$n
done
(( pcores_max == 0 )) && pcores_max=$(( $(nproc) / 2 - 1 ))
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
        logger -t greenboost-shader-boost "Detected fossilize_replay - boosting PIDs: ${new_pids[*]}"
        logger -t greenboost-shader-boost "Applied: renice -5, ionice best-effort/0, taskset 0-${pcores_max}"
    fi

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
        && info "Gaming boost service installed and running (fossilize_replay → renice -5 + P-cores)" \
        || warn "Gaming boost service installed but failed to start - check: systemctl status greenboost-shader-boost"
}

cmd_install_vulkan_layer() {
    local layer_src="$MODULE_DIR/$VULKAN_LAYER_LIB"
    local layer_dest="$SHIM_DEST/$VULKAN_LAYER_LIB"
    local manifest_src="$MODULE_DIR/$VULKAN_LAYER_MANIFEST"
    local manifest_dest="$VULKAN_IMPLICIT_LAYER_DIR/$VULKAN_LAYER_MANIFEST"

    if [[ ! -f "$layer_src" ]]; then
        warn "Vulkan layer not built - run 'make vulkan' first, then re-run install-sys-configs"
        return 0
    fi

    mkdir -p "$VULKAN_IMPLICIT_LAYER_DIR"
    cp "$layer_src" "$layer_dest"
    cp "$manifest_src" "$manifest_dest"
    ldconfig

    info "Vulkan layer installed - use GREENBOOST_VULKAN=1 %command% in Steam launch options"
}


cmd_build() {
    info "Building GreenBoost v3.4 (cuda memory pool: VRAM + System DDR + NVMe)..."
    make -C "$MODULE_DIR" all || die "Build failed - check output above"
    info "Build complete:"
    info "  Kernel module : $MODULE_DIR/greenboost.ko"
    info "  CUDA shim     : $MODULE_DIR/$SHIM_LIB"
    info "  LD_AUDIT lib  : $MODULE_DIR/$AUDIT_LIB"
    info "  Vulkan layer  : $MODULE_DIR/$VULKAN_LAYER_LIB"
}

cmd_install() {
    need_root install
    detect_hardware
    check_deps
    cmd_build

    info "Installing kernel module..."
    make -C "$MODULE_DIR" install || die "Module install failed"

    info "Installing CUDA shim + LD_AUDIT library + Vulkan layer to $SHIM_DEST/..."
    cp "$MODULE_DIR/$SHIM_LIB" "$SHIM_DEST/"
    [[ -f "$MODULE_DIR/$AUDIT_LIB" ]] && cp "$MODULE_DIR/$AUDIT_LIB" "$SHIM_DEST/"
    [[ -f "$MODULE_DIR/$VULKAN_LAYER_LIB" ]] && cp "$MODULE_DIR/$VULKAN_LAYER_LIB" "$SHIM_DEST/"
    # F-L4-31: Set persistent SELinux file context on the CUDA shim so
    # ld.so.preload injection works on SELinux-enforcing Rocky/RHEL systems.
    # semanage sets the persistent policy (survives restorecon -R); chcon sets
    # the immediate label; together they handle both live and rebooted systems.
    local _shim_path="$SHIM_DEST/$SHIM_LIB"
    if selinuxenabled 2>/dev/null; then
        if command -v semanage &>/dev/null; then
            semanage fcontext -a -t lib_t "${_shim_path}" 2>/dev/null \
                || semanage fcontext -m -t lib_t "${_shim_path}" 2>/dev/null || true
            restorecon -v "${_shim_path}" 2>/dev/null || true
        elif command -v chcon &>/dev/null; then
            chcon -t lib_t "${_shim_path}" 2>/dev/null || true
        fi
    fi
    ldconfig

    # modprobe defaults
    info "Writing /etc/modprobe.d/greenboost.conf ..."
    cat > /etc/modprobe.d/greenboost.conf << MODEOF
# GreenBoost - cuda memory pool (auto-configured for detected hardware)
# GPU   : ${GPU_NAME}  (${GB_PHYS} GB VRAM)
# RAM   : ${RAM_TYPE}-${RAM_SPEED_MT}  (pool ${GB_VIRT} GB, reserve ${GB_RESERVE} GB)
# NVMe  : swap ${GB_NVME_SWAP} GB
options greenboost physical_vram_gb=${GB_PHYS} virtual_vram_gb=${GB_VIRT} safety_reserve_gb=${GB_RESERVE} nvme_pool_gb=${GB_NVME_POOL} t3_max_gb=${GB_NVME_POOL} t3_file_path=/var/lib/greenboost/t3_store pcores_max_cpu=${GB_PCORES_MAX} golden_cpu_min=${GB_GOLDEN_MIN} golden_cpu_max=${GB_GOLDEN_MAX} ecores_only=${GB_PCORES_ONLY}
MODEOF

    # profile.d helper
    cat > /etc/profile.d/greenboost.sh << PROFEOF
# GreenBoost v3.4 - shell helpers
export GREENBOOST_SHIM="$SHIM_DEST/$SHIM_LIB"
greenboost-run() { LD_PRELOAD="\$GREENBOOST_SHIM" "\$@"; }
export -f greenboost-run
PROFEOF

    # Standalone wrapper
    cat > /usr/local/bin/greenboost-run << WRAPEOF
#!/usr/bin/env bash
# Run a CUDA application with GreenBoost System DDR overflow enabled
LD_PRELOAD="$SHIM_DEST/$SHIM_LIB" "\$@"
WRAPEOF
    chmod +x /usr/local/bin/greenboost-run

    info ""
    info "Installation complete!"
    info "  Load module    : sudo modprobe greenboost"
    info "  Run CUDA app   : greenboost-run your_cuda_app"
    info "  Pool status    : cat /sys/class/greenboost/greenboost/status"
}

cmd_load() {
    need_root load
    detect_hardware

    # Load active profile values (lowest priority - env vars and CLI flags override)
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
    local pcores_max="${GB_PCORES_MAX}"
    local golden_min="${GB_GOLDEN_MIN}"
    local golden_max="${GB_GOLDEN_MAX}"
    local ecores_only="${GB_PCORES_ONLY}"

    if lsmod | grep -q "^${DRIVER_NAME} "; then
        warn "Module already loaded - reloading..."
        # Stop consumers before rmmod to avoid EBUSY
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
        pcores_max_cpu="$pcores_max" \
        golden_cpu_min="$golden_min" \
        golden_cpu_max="$golden_max" \
        ecores_only="$ecores_only" \
        active_profile_name="${PROF_NAME:-autodetect}" \
        || die "insmod failed - check: dmesg | tail -20"

    info "GreenBoost loaded - cuda memory pool active!"
    info ""
    info "  T1 ${GPU_NAME:-GPU} VRAM : ${phys} GB   [hot layers]"
    info "  T2 System DDR pool     : ${virt} GB  (~${PCIE_BW_GBS:-auto} GB/s PCIe)  [cold layers]"
    info "  T3 NVMe swap     : ${nvme_sw} GB  [frozen pages]"
    info "  ─────────────────────────────────────────"
    info "  Combined view    : $(( phys + virt + nvme_sw )) GB total model capacity"
    info ""
    info "Pool info  : cat /sys/class/greenboost/greenboost/status"
    info "Kernel log : dmesg | grep greenboost"
    echo ""
    dmesg | grep greenboost | tail -8 | sed 's/^/  /'
}

cmd_unload() {
    need_root unload
    if lsmod | grep -q "^${DRIVER_NAME} "; then
        rmmod "$DRIVER_NAME" && info "GreenBoost unloaded" \
            || die "rmmod failed - check: dmesg | tail -5"
    else
        info "GreenBoost is not loaded"
    fi
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

    # 3. Remove GreenBoost entries from /etc/ld.so.preload FIRST - before deleting
    #    the .so files (prevents "cannot be preloaded" linker errors on forked procs).
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

    # 4b. SELinux context cleanup (RHEL/Rocky - no AppArmor)
    if command -v semanage &>/dev/null && command -v restorecon &>/dev/null; then
        semanage fcontext -d "/usr/local/lib/libgreenboost.*" 2>/dev/null || true
        restorecon -R /usr/local/lib/ 2>/dev/null || true
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
    # TurboQuant daemon
    systemctl disable greenboost-turboquant.service 2>/dev/null || true
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

cmd_tune() {
    need_root tune
    detect_hardware

    info "Tuning workstation for GreenBoost / LLM workloads..."
    info "Hardware: ${GPU_NAME:-auto-detected} | ${CPU_NAME:-auto-detected}"
    echo ""

    # ── CPU governor → performance (P-cores run at 6 GHz, not 800 MHz) ──
    local changed=0
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [[ -w "$gov" ]] && echo performance > "$gov" && changed=1
    done
    [[ $changed -eq 1 ]] && info "CPU governor      : performance (all 32 CPUs)" \
                          || warn "CPU governor      : could not set (check cpufreq driver)"

    # ── NVMe scheduler → none (best latency for Samsung 990 EVO Plus) ──
    for sched in /sys/block/nvme*/queue/scheduler; do
        [[ -w "$sched" ]] && echo none > "$sched" 2>/dev/null || true
    done
    info "NVMe scheduler    : none (was: mq-deadline)"

    # ── NVMe read-ahead → 4 MB (large sequential model weight loading) ──
    for ra in /sys/block/nvme*/queue/read_ahead_kb; do
        [[ -w "$ra" ]] && echo 4096 > "$ra"
    done
    info "NVMe read_ahead   : 4096 KB = 4 MB (was: 128 KB)"

    # ── NVMe nr_requests → 1024 ──────────────────────────────────────────
    for nr in /sys/block/nvme*/queue/nr_requests; do
        [[ -w "$nr" ]] && echo 1024 > "$nr"
    done
    info "NVMe nr_requests  : 1024"

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
    info "Tuning done. To make permanent add to /etc/sysctl.conf:"
    info "  vm.swappiness=10"
    info "  vm.dirty_ratio=40"
    info "  vm.dirty_background_ratio=10"
    info "And add to /etc/rc.local for NVMe + THP settings."
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

    info "Validating GRUB flags for: ${CPU_NAME} | ${GPU_NAME}"
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
    # Staggers per-CPU timer ticks to reduce lock contention when all CPUs
    # fire timer interrupts simultaneously (common on hybrid P/E core designs).
    # Runtime test: always safe - no kernel config dependency.
    _grub_check_flag "skew_tick=1" \
        "stagger timer ticks - reduces lock contention on hybrid P/E cores" "" \
        && new_flags="$new_flags skew_tick=1"

    # ── Flag: rcu_nocbs=<ecores> ─────────────────────────────────────────
    # Offloads RCU (Read-Copy-Update) callback processing to E-cores,
    # freeing high-frequency P-cores for inference hot paths.
    # Range is derived from detected CPU topology at runtime.
    if [[ $GB_PCORES_ONLY -eq 1 ]]; then
        local ecores_start=$(( GB_PCORES_MAX + 1 ))
        local ecores_end=$(( $(nproc) - 1 ))
        if [[ $ecores_start -le $ecores_end ]]; then
            _grub_check_flag "rcu_nocbs=${ecores_start}-${ecores_end}" \
                "offload RCU callbacks to E-cores (CPU ${ecores_start}-${ecores_end}), freeing P-cores for inference" \
                "CONFIG_RCU_NOCB_CPU" \
                && new_flags="$new_flags rcu_nocbs=${ecores_start}-${ecores_end}"
        fi
    else
        info "  [skip]    rcu_nocbs - no E-cores detected (non-hybrid CPU)"
    fi

    # ── Flag: nohz_full=<golden> ─────────────────────────────────────────
    # Makes the highest-frequency cores tick-less when they have exactly
    # one runnable thread. Eliminates timer interrupts during dense matrix
    # multiplications - reduces LLM token latency.
    # Range derived from detected golden-core topology at runtime.
    if [[ $GB_PCORES_ONLY -eq 1 && $GB_GOLDEN_MIN -lt $GB_GOLDEN_MAX ]]; then
        _grub_check_flag "nohz_full=${GB_GOLDEN_MIN}-${GB_GOLDEN_MAX}" \
            "tick-less golden P-cores (CPU ${GB_GOLDEN_MIN}-${GB_GOLDEN_MAX}) during single-thread inference" \
            "CONFIG_NO_HZ_FULL" \
            && new_flags="$new_flags nohz_full=${GB_GOLDEN_MIN}-${GB_GOLDEN_MAX}"
    else
        info "  [skip]    nohz_full - no golden cores detected or CONFIG_NO_HZ_FULL absent"
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
    python3 -c "
import re, sys
txt = open('$grub_file').read()
txt = re.sub(r'^GRUB_CMDLINE_LINUX_DEFAULT=.*', 'GRUB_CMDLINE_LINUX_DEFAULT=\"' + sys.argv[1] + '\"', txt, flags=re.MULTILINE)
open('$grub_file', 'w').write(txt)
" "$new_line"

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

    local dest="/etc/sysctl.d/99-zzz-greenboost.conf"

    info "Writing definitive sysctl config: $dest"
    info "This file loads last (99-zzz) and wins over all conflicting files."
    echo ""

    # Show conflicts found
    info "Conflicts resolved:"
    info "  vm.swappiness       : multiple files set 10/20 → final: 10"
    info "  vm.dirty_ratio      : 15 vs 40 → final: 40 (Samsung 990 sustains 7 GB/s)"
    info "  vm.dirty_background_ratio: 5 vs 10 → final: 10"
    info "  kernel.sched_autogroup_enabled: 1 → 0 (bad for compute, groups by session)"
    info "New settings added:"
    info "  kernel.sched_migration_cost_ns: 5000000 (5ms - keep threads on P-cores)"
    info "  kernel.sched_min_granularity_ns: 10000000 (10ms - better for large tasks)"
    info "  kernel.sched_wakeup_granularity_ns: 15000000 (reduces spurious wakeups)"
    echo ""

    cat > "$dest" << 'SYSCTL_EOF'
# GreenBoost v3.4 - Definitive sysctl config
# Tuned for GreenBoost inference workloads - edit if your hardware differs.
# Loaded last (99-zzz) - wins all conflicts with earlier sysctl.d files.
# Do NOT edit other sysctl.d files; make changes here instead.

# ── Swap / memory pressure ───────────────────────────────────────────────
# Keep LLM weights in System DDR (T2); only spill to NVMe (T3) under real pressure.
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

SYSCTL_EOF

    # Derived, not fixed , same reasoning as greenboost_setup.sh: 512 MB is
    # 0.8% of a 64 GB box and 8% of an 8 GB feeder, and the literal silently
    # overrode linux-kernel-inference's own 95-greenboost-t2.conf because this
    # file sorts last and wins every conflict.
    local _memtotal_kb
    _memtotal_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
    local _min_free_kb=$(( _memtotal_kb * 16 / 1000 ))    # ~1.6% of RAM
    (( _min_free_kb < 65536 )) && _min_free_kb=65536      # floor: 64 MB
    printf '\n# Free-memory floor: ~1.6%% of this box'"'"'s %s kB of RAM (derived).\n' "$_memtotal_kb" >> "$dest"
    printf 'vm.min_free_kbytes = %d\n' "$_min_free_kb" >> "$dest"

    cat >> "$dest" << 'SYSCTL_EOF'

# Proactive compaction: GreenBoost T2 needs contiguous 2 MB hugepage ranges.
# Value 20 = moderate background compaction (0=off, 100=aggressive).
vm.compaction_proactiveness = 20

# Overcommit hugepage pool: 10240 × 2 MB = 20 GB pre-reserved for THP allocs.
vm.nr_overcommit_hugepages = 10240

# Keep inode/dentry caches alive - LLM loaders open thousands of weight files.
vm.vfs_cache_pressure = 50

# Disable zone reclaim: single NUMA node - cross-zone reclaim wastes cycles.
vm.zone_reclaim_mode = 0

# ── CPU scheduler (i9-14900KF P-core / E-core hybrid) ────────────────────
# Disable session-based task grouping. sched_autogroup groups shell tasks
# together which is good for desktop but HURTS inference: Ollama worker
# threads (long-running compute) compete with short interactive tasks for
# scheduler time-slices in the same group.
kernel.sched_autogroup_enabled = 0

# Raise migration cost threshold: scheduler avoids migrating tasks to
# a different CPU unless the cache-miss penalty exceeds this value (5 ms).
# Effect: LLM inference threads stay on their assigned P-cores rather than
# bouncing between P-cores and E-cores (which have different cache hierarchies).
kernel.sched_migration_cost_ns = 5000000

# Minimum scheduling granularity (10 ms): gives large compute tasks
# (matrix multiplications, attention heads) more uninterrupted runtime
# before the scheduler can preempt them.
kernel.sched_min_granularity_ns = 10000000

# Wakeup granularity (15 ms): a waking task only preempts the current task
# if it has been sleeping for more than this long. Prevents short-lived
# system tasks from constantly interrupting inference threads.
kernel.sched_wakeup_granularity_ns = 15000000

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

# ── Perf / profiling access ───────────────────────────────────────────────
# Allow nsys / perf / CUDA Nsight without sudo (needed for GPU profiling).
kernel.perf_event_paranoid = 1
kernel.kptr_restrict = 0
SYSCTL_EOF

    sysctl -p "$dest" 2>&1 | grep -v "^$" | sed 's/^/  /' || true
    echo ""
    info "sysctl applied and persistent (survives reboot via $dest)."
}

# ---- tune-libs ---------------------------------------------------------
# Install missing libraries and kernel modules for AI/compute workloads.
# All packages chosen for AVX2/FMA/VNNI capabilities on i9-14900KF.

cmd_tune_libs() {
    need_root tune-libs

    info "Installing missing AI/compute libraries..."
    echo ""

    # ── APT packages ──────────────────────────────────────────────────────
    local pkgs=(
        # BLAS/LAPACK - OpenBLAS compiled with AVX2+FMA for CPU inference
        openblas-devel
        blas-devel
        lapack-devel

        # OpenMP - multi-threaded CPU inference (llama.cpp uses this heavily)
        libomp-devel

        # hwloc - hardware topology library used by Ollama/llama.cpp for
        # CPU pinning; without it Ollama uses a generic thread affinity model
        hwloc
        hwloc-devel

        # libnuma - NUMA-aware memory allocation (single node but still used
        # by CUDA and some ML runtimes for memory locality hints)
        numa-devel

        # OpenCL - GPU compute via OpenCL API (some inference backends use it)
        ocl-icd-devel

        # nvtop - real-time GPU + CPU monitor (shows all 3 tiers at a glance)
        nvtop

        # cpufrequtils - userspace CPU frequency tools (cpufreq-info, etc.)
        cpufrequtils

        # kernel-tools - perf, turbostat (monitors P/E core frequencies + C-states)
        kernel-tools

        # microcode_ctl - latest CPU microcode (fixes + performance patches
        # for i9-14900KF Raptor Lake stepping)
        microcode_ctl
    )

    info "Packages to install:"
    local to_install=()
    for pkg in "${pkgs[@]}"; do
        if rpm -q "$pkg" &>/dev/null; then
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
        dnf install -y "${to_install[@]}" 2>&1 | tail -5
        info "Packages installed."
    fi

    echo ""

    # ── Kernel modules ────────────────────────────────────────────────────
    info "Kernel modules:"

    # cpuid - lets userspace read CPUID leaves directly. Used by turbostat,
    # CUDA diagnostics, and microcode_ctl update verification.
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
    if rpm -q openblas-devel &>/dev/null; then
        if command -v alternatives &>/dev/null; then
            alternatives --set libblas.so.3 \
                /usr/lib64/libopenblas.so.0 2>/dev/null \
                && info "BLAS alternative: set to OpenBLAS (AVX2/FMA)" \
                || info "BLAS alternative: already set or path differs - check manually"
        fi
    fi

    echo ""
    info "tune-libs complete."
    info "  Turbostat (P/E core monitoring): sudo turbostat --quiet --Summary"
    info "  nvtop (GPU + CPU):               nvtop"
    info "  CPU frequency info:              cpufreq-info"
}

# ---- tune-all ----------------------------------------------------------

cmd_tune_all() {
    need_root tune-all
    info "Running full system tuning for GreenBoost v2.8..."
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
    dmesg | grep greenboost | tail -10 | sed 's/^/  /'
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
        echo -e "  ${C_CYAN}${C_BOLD}USAGE:${C_RESET}  ${C_GRAY}sudo ./greenboost_setup_rocky.sh <command>${C_RESET}"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}COMMON COMMANDS:${C_RESET}"
        echo -e ""
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "setup"        "Full install - deps, module, tune, Python inference tools"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "status"       "Show pool info + system state"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "benchmark"    "cuda memory pool bandwidth benchmark (T1 VRAM / T2 DDR / T3 NVMe)"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "tune"         "Runtime tuning (CPU governor, NVMe, THP, sysctl)"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "load"         "Load kernel module with cuda memory pool params"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "unload"       "Unload module"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "uninstall"    "Remove module + all config"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "profile"      "Interactive profile wizard (create / activate / diff)"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clean memory" "Force-release T1 VRAM + T2 RAM + T3 now (unloads inference models)"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "logs"         "Snapshot all GreenBoost log sources (kernel, Ollama, Vulkan, Proton)"
        printf "  ${C_LIME}%-18s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clear logs"   "Clear all GreenBoost logs for a fresh diagnostic baseline"
        echo -e ""
        echo -e "  ${C_DIM}Run ${C_GRAY}greenboost help${C_DIM} for the full command reference (always available after install).${C_RESET}"
        echo -e "  ${C_DIM}Run without arguments for the interactive wizard (includes ${C_GRAY}GreenBoost Commands${C_DIM} entry).${C_RESET}"
        echo -e "  ${C_DIM}Run ${C_GRAY}help --all${C_DIM} for environment variables and advanced flags.${C_RESET}"
    else
        echo -e "  ${C_CYAN}${C_BOLD}USAGE:${C_RESET}  ${C_GRAY}sudo ./greenboost_setup_rocky.sh <command>${C_RESET}"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}MAIN COMMANDS:${C_RESET}"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "setup"           "Full install (deps + module + shim + configs + tune + Python tools)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "install"         "Build and install module + CUDA shim system-wide"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "uninstall"       "Unload, remove module + all config files"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "build"           "Build only (no system install)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "load"            "Load module with default cuda memory pool parameters"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "unload"          "Unload module (keeps installed files)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "status"          "Show module status and cuda memory pool"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "profile [sub]"   "Interactive wizard; or: create / show / list / activate / diff"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "benchmark"       "cuda memory pool bandwidth benchmark (T1 VRAM / T2 DDR / T3 NVMe)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clean memory"    "Force-release T1 VRAM + T2 RAM + T3 immediately (unloads models)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "setup-swap [GB]" "Create/activate NVMe swap (auto-sized if omitted)"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}TUNING:${C_RESET}"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "tune"            "Runtime tuning (governor, NVMe, THP, sysctl)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "tune-grub"       "Fix GRUB boot params (THP=always, rcu_nocbs, nohz_full…)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "tune-sysctl"     "Consolidate sysctl + apply compute-optimized knobs"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "tune-libs"       "Install missing AI/compute libraries (OpenBLAS, hwloc…)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "tune-all"        "Run tune + tune-grub + tune-sysctl + tune-libs"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}LOGGING:${C_RESET}"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "logs"                  "Snapshot all GreenBoost log sources (kernel, Ollama, Vulkan, Proton)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "proton-logs"           "Show Proton/VKD3D game logs"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "inference-logs"        "Show Ollama/inference logs"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clear logs"            "Clear all GreenBoost logs"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clear proton-logs"     "Clear Proton logs"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "clear inference-logs"  "Clear inference logs"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}GAMING:${C_RESET}"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "gaming-mode enable|disable" "Suspend/restore Ollama before/after gaming"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "steam-launch-guide"     "Show Steam launch options + Proton setup guide"
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
        echo -e "  ${C_AMBER}  Example - preset M:${C_RESET}"
        echo -e "  ${C_LIME}DXVK_NVAPI_DRS_NGX_DLSS_SR_OVERRIDE_RENDER_PRESET_SELECTION=render_preset_m %command%${C_RESET}"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}ADVANCED:${C_RESET}"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "install-sys-configs"   "Ollama env, NVMe udev, CPU governor, hugepages, LD_AUDIT"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "install-deps"          "Install all Rocky OS packages (build + CUDA + AI libs)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "install-vulkan-layer"  "Install GreenBoost Vulkan layer"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}GLOBAL FLAGS:${C_RESET}"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "--skip-update-check"   "Bypass GitLab version check (offline)"
        printf "  ${C_LIME}%-26s${C_RESET} ${C_GRAY}%s${C_RESET}\n" "--profile <file>"      "Load module parameters from profile file"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}ENVIRONMENT (load command):${C_RESET}"
        printf "  ${C_DIM}%-26s${C_RESET} ${C_DIM}%s${C_RESET}\n" "GPU_PHYS_GB=${GB_PHYS}"      "Physical VRAM in GB (detected: ${GPU_NAME})"
        printf "  ${C_DIM}%-26s${C_RESET} ${C_DIM}%s${C_RESET}\n" "VIRT_VRAM_GB=${GB_VIRT}"     "System RAM pool size in GB"
        printf "  ${C_DIM}%-26s${C_RESET} ${C_DIM}%s${C_RESET}\n" "RESERVE_GB=${GB_RESERVE}"    "System RAM to keep free"
        printf "  ${C_DIM}%-26s${C_RESET} ${C_DIM}%s${C_RESET}\n" "NVME_SWAP_GB=${GB_NVME_SWAP}" "NVMe swap capacity in GB"
        printf "  ${C_DIM}%-26s${C_RESET} ${C_DIM}%s${C_RESET}\n" "NVME_POOL_GB=${GB_NVME_POOL}" "GreenBoost T3 soft cap in GB"
        echo -e ""
        echo -e "  ${C_DIM}Example: sudo VIRT_VRAM_GB=48 NVME_SWAP_GB=64 ./greenboost_setup_rocky.sh load${C_RESET}"
        echo -e ""
        echo -e "  ${C_CYAN}${C_BOLD}MONITORING:${C_RESET}"
        echo -e "  ${C_DIM}greenboost status${C_RESET}"
        echo -e "  ${C_DIM}dmesg | grep greenboost | tail -20${C_RESET}"
        echo -e "  ${C_DIM}watch -n1 free -h   # T2 RAM pressure${C_RESET}"
        echo -e "  ${C_DIM}watch -n1 swapon --show   # T3 NVMe usage${C_RESET}"
    fi
    echo -e ""
}

# ---- install-deps ------------------------------------------------------
# Install all Rocky packages needed for GreenBoost v3.4 + ExLlamaV3

cmd_install_build_deps() {
    need_root install-deps

    printf "  ${C_CYAN}❯${C_RESET}  ${C_DIM}Updating package database...${C_RESET}"
    dnf update -qq -y 2>/dev/null || true
    printf "\r%*s\r" "$(tput cols 2>/dev/null || echo 80)" ""

    # Minimal packages required to build and DKMS-register the kernel module.
    local _groups=( "Development Tools" "kernel headers + build tools" "DKMS + io_uring" "Microcode" )
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
    dnf -y groupinstall 'Development Tools' -q 2>/dev/null

    (( _idx++ )); _dep_bar $_idx "${_groups[$_idx-1]}"
    dnf install -y -q \
        gcc gcc-c++ make cmake git curl wget \
        "kernel-headers-$(uname -r)" \
        pkg-config sysfsutils kmod 2>/dev/null || true

    (( _idx++ )); _dep_bar $_idx "${_groups[$_idx-1]}"
    dnf install -y -q dkms liburing-devel 2>/dev/null || true

    (( _idx++ )); _dep_bar $_idx "${_groups[$_idx-1]}"
    dnf install -y -q microcode_ctl 2>/dev/null || true

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
    dnf install -y -q glibc-devel.i686 libgcc.i686 2>/dev/null || true

    (( _idx++ )); _dep_bar $_idx "${_groups[$_idx-1]}"
    dnf install -y -q python3 python3-pip python3-devel python3-virtualenv 2>/dev/null

    (( _idx++ )); _dep_bar $_idx "${_groups[$_idx-1]}"
    dnf install -y -q cpufrequtils kernel-tools nvtop 2>/dev/null || true
    dnf install -y -q "kernel-tools-$(uname -r)" 2>/dev/null || true

    (( _idx++ )); _dep_bar $_idx "${_groups[$_idx-1]}"
    dnf install -y -q \
        openblas-devel blas-devel lapack-devel \
        hwloc-devel hwloc numactl-devel libomp-devel \
        ocl-icd-devel 2>/dev/null || true
    # cuda-toolkit requires NVIDIA repo: https://developer.nvidia.com/cuda-downloads
    dnf install -y -q cuda-toolkit 2>/dev/null || true

    printf "\r%*s\r" "$(tput cols 2>/dev/null || echo 80)" ""

    gb_ok "Optional AI/compute libraries installed"
    gb_info "Note: NVIDIA driver 580+ and CUDA 13 must be installed separately"
}

# cmd_install_deps - installs all dependencies (build + optional).
# Called when running 'install-deps' directly; full-install uses
# cmd_install_build_deps (always) + cmd_install_optional_pkgs (gated).
cmd_install_deps() {
    cmd_install_build_deps
    cmd_install_optional_pkgs
}

# ---- setup-swap --------------------------------------------------------
# Create NVMe swap file (T3 tier). Safe to re-run - idempotent.

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
        gb_info "  (e.g. systemd-swap, /etc/fstab with a swap partition, or fallocate + mkswap)"
        gb_info "  then re-run 'greenboost full-install' to update the kernel module parameters."
    fi
}

# ---- full-install ------------------------------------------------------
# Complete fresh-OS install - run this after a clean Rocky install.
# Covers: OS deps, NVMe swap, kernel module, CUDA shim, all system configs,
# sysctl tuning, GRUB params, and optional ExLlamaV3 with GreenBoost patches.

cmd_full_install() {
    need_root full-install
    GB_STOPPED_SERVICES=""

    # ── Mode selection - cmd_full_install always targets full system setup.
    # Only --module-only overrides this (for scripts / CI / direct invocation).
    GB_INSTALL_MODE="full"
    for _a in "$@"; do
        [[ "$_a" == "--module-only" ]] && { GB_INSTALL_MODE="module"; break; }
    done

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
    do_purge 0
    gb_ok "Previous installation purged"

    # 1 - Build dependencies (minimal - just what's needed to compile the module)
    gb_step 1 5 "Installing Rocky/RHEL build dependencies..."
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
        gb_info "Run 'sudo ./greenboost_setup_rocky.sh status' to verify."
        gb_info "For full system tuning, re-run and choose option [2]."
        echo ""
        return 0
    fi

    # ── FULL ONLY PATH: all system changes applied automatically (user consented by choosing full mode) ───

    # 4 - System configs: Ollama, NVMe udev, CPU governor, hugepages, sysctl
    gb_step 4 5 "Installing system configuration files..."
    gb_info "Applying: Ollama/inference service config (drop-ins, TurboQuant, udev, NVMe, cpu-perf)"
    cmd_install_sys_configs
    gb_ok "System configuration installed"

    # 4b - Optional AI/compute libraries (CUDA toolkit, OpenBLAS, Python, nvtop, etc.)
    gb_info "Applying: optional packages (cuda-toolkit, openblas, python3-pip, nvtop, cpufrequtils, 32-bit compat)"
    cmd_install_optional_pkgs
    gb_ok "Optional AI/compute libraries installed"

    # 5 - sysctl tuning
    gb_step 5 5 "Applying system tuning..."
    gb_info "Applying: sysctl + NVMe/THP/CPU governor tuning"
    cmd_tune_sysctl
    gb_ok "sysctl tuning applied"

    # 5b - GRUB boot parameters (requires reboot)
    gb_info "Applying: GRUB boot parameters (transparent_hugepage, rcu_nocbs, nohz_full)"
    cmd_tune_grub
    gb_ok "GRUB updated"

    # Generate or regenerate profile: create on first install, or when version changed.
    local _prof_ver=""
    [[ -f "${GB_ACTIVE_PROFILE_LINK}" ]] && \
        _prof_ver=$(grep -m1 '^greenboost_version:' "${GB_ACTIVE_PROFILE_LINK}" 2>/dev/null \
                    | awk -F'"' '{print $2}')
    if [[ ! -f "${GB_ACTIVE_PROFILE_LINK}" ]]; then
        info "Generating hardware profile..."
        cmd_profile_create
    elif [[ "$_prof_ver" != "$GB_VERSION" ]]; then
        warn "Profile version mismatch (${_prof_ver} → ${GB_VERSION}) - regenerating..."
        cmd_profile_create
    fi

    # 5c - Python orchestration stack + MCP + gb-synapse (delegated to the
    # main, distro-agnostic setup script — this Rocky/RHEL installer only
    # ever covered the kernel module + CUDA shim + system tuning above; the
    # whole Python/MCP/CLI/gb-quant/gb-dataflux/gb-synapse layer was
    # silently absent on Rocky until now. Each step mirrors
    # greenboost_setup.sh's own Full Install: best-effort, never aborts the
    # install on failure.
    local main_script="$MODULE_DIR/greenboost_setup.sh"
    if [[ -x "$main_script" ]]; then
        info "Applying: Python orchestration stack (gb-quant/gb-dataflux/gb-synapse/CLI)"
        "$main_script" install-python \
            || warn "Python file install had failures — retry: sudo ./greenboost_setup.sh install-python"
        "$main_script" install-cli \
            || warn "greenboost-cli install had failures — retry: sudo ./greenboost_setup.sh install-cli"
        "$main_script" register-mcp \
            || warn "MCP registration had failures — retry: greenboost register-mcp"
        if [[ "${GB_INSTALL_SYNAPSE_ENGINE:-1}" != "0" ]]; then
            "$main_script" install-synapse-engine \
                || warn "gb-synapse torch engine install had failures — retry: sudo greenboost install-synapse-engine"
            "$main_script" synapse build-engine \
                || warn "gb-synapse llama.cpp engine build had failures — retry: sudo greenboost synapse build-engine"
        fi
        gb_ok "Python orchestration stack + gb-synapse installed"
    else
        warn "greenboost_setup.sh not found next to this script — skipping Python/MCP/gb-synapse install"
    fi

    info "╔══════════════════════════════════════════════════════════════╗"
    info "║  Full install complete!                                      ║"
    info "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    warn "REBOOT REQUIRED to activate GRUB params + hugepage pre-allocation"
    echo ""
    info "For live Ollama logs: journalctl -u ollama -f"
    gb_info "GreenBoost Proton and MangoHud are gaming tools - not part of full install."
    gb_info "Install them separately from the interactive menu (options [6]–[9])."
}

# cmd_gaming_mode - stop/start Ollama to free T2 DDR before/after gaming.

cmd_clear_logs() {
    need_root clear-logs
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

# ---- Wizard support functions (required by all 9 wizard options) ------------

# Option 3 - Benchmark
cmd_benchmark() {
    local bench_py="$MODULE_DIR/tools/gb_workstation_bench.py"
    [[ -f "$bench_py" ]] || die "Benchmark script not found: $bench_py"
    python3 "$bench_py" "$@"
}

cmd_t3_memory() {
    local main_script="$MODULE_DIR/greenboost_setup.sh"
    [[ -f "$main_script" ]] || die "Main setup script not found: $main_script"
    exec bash "$main_script" t3-memory "${1:-}"
}

# Option 4 - Profile management interactive sub-menu
cmd_profile_wizard() {
    while true; do
        clear
        local cols; cols=$(tput cols 2>/dev/null || echo 72)
        local title=" GreenBoost v${GB_VERSION} - Profile Management"
        echo -e ""
        echo -e "${C_VIOLET}${C_BOLD}  ╔$(printf '═%.0s' $(seq 1 $((cols - 4))))╗${C_RESET}"
        echo -e "${C_VIOLET}${C_BOLD}  ║${C_RESET} ${C_GRAY}${C_BOLD}${title}$(printf ' %.0s' $(seq 1 $((cols - 4 - ${#title} - 1))))${C_VIOLET}${C_BOLD}║${C_RESET}"
        echo -e "${C_VIOLET}${C_BOLD}  ╚$(printf '═%.0s' $(seq 1 $((cols - 4))))╝${C_RESET}"
        echo -e ""

        # Show active profile
        if [[ -L "$GB_ACTIVE_PROFILE_LINK" || -f "$GB_ACTIVE_PROFILE_LINK" ]]; then
            local active_path; active_path=$(readlink -f "$GB_ACTIVE_PROFILE_LINK" 2>/dev/null)
            local active_name; active_name=$(basename "$active_path" .md)
            echo -e "  ${C_LIME}${C_BOLD}✓  Active:${C_RESET}  ${C_CYAN}${active_name}${C_RESET}  ${C_DIM}${C_GRAY}${active_path}${C_RESET}"
        else
            echo -e "  ${C_AMBER}⚠${C_RESET}  ${C_AMBER}${C_BOLD}No active profile${C_RESET}  ${C_DIM}- select [1] to auto-detect hardware and create one${C_RESET}"
        fi

        # List saved profiles
        if [[ -d "$GB_PROFILES_DIR" ]]; then
            local active_abs=""
            [[ -L "$GB_ACTIVE_PROFILE_LINK" ]] && active_abs=$(readlink -f "$GB_ACTIVE_PROFILE_LINK" 2>/dev/null)
            local profiles=( "$GB_PROFILES_DIR"/*.md )
            if [[ -f "${profiles[0]}" ]]; then
                echo -e ""
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
            fi
        fi
        echo -e ""

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


# Option 6 - Install MangoHud (dnf / build-from-source)
MANGOHUD_REPO="https://github.com/flightlessmango/MangoHud"
MANGOHUD_SRC="/opt/MangoHud"
MANGOHUD_BUILD_DIR="$MANGOHUD_SRC/build"
MANGOHUD_PREFIX="/usr/local"


# Option 7 - GreenBoost Commands reference
cmd_show_commands() {
    local cmds_file="$MODULE_DIR/GREENBOOST_COMMANDS.md"
    if [[ -f "$cmds_file" ]]; then
        cat "$cmds_file"
    else
        gb_info "GREENBOOST_COMMANDS.md not found - run: $0 help"
    fi
}

# ── cmd_turboquant - KV cache TurboQuant compression control (Rocky/RHEL variant) ──
# Delegates status/enable/disable to the same logic as the Ubuntu variant.
# Usage: greenboost turboquant [status|enable [2|3|4]|disable]
cmd_turboquant() {
    local subcmd="${1:-status}"
    local conf_path="/run/greenboost/turboquant.conf"
    local dev_path="/dev/greenboost"

    case "$subcmd" in
        status|"")
            gb_section "TurboQuant KV Cache Compression"
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
            local sysfs="/sys/class/greenboost/greenboost/status"
            if [[ -r "$sysfs" ]]; then
                local kv_cmp_mb
                kv_cmp_mb=$(grep -oP 'KV compressed\s*:\s*\K\d+' "$sysfs" 2>/dev/null || echo "0")
                [[ "$kv_cmp_mb" -gt 0 ]] && gb_info "Saved by TQ:  ${kv_cmp_mb} MB"
            fi
            gb_info "Daemon:       greenboost-turboquant.service (auto-selects bit width)"
            gb_info "Manual ctl:   greenboost turboquant enable [2|3|4]"
            gb_info "Conf file:    ${conf_path}"
            ;;
        enable)
            local req_bits="${2:-0}"
            if [[ "$req_bits" == "0" ]]; then
                systemctl restart greenboost-turboquant.service 2>/dev/null && \
                    gb_ok "Restarted greenboost-turboquant daemon (auto mode)" || \
                    gb_warn_ui "greenboost-turboquant.service not installed - run full-install first"
            else
                if [[ "$req_bits" != "2" && "$req_bits" != "3" && "$req_bits" != "4" ]]; then
                    gb_fail "Invalid bit width: ${req_bits}. Use 2, 3, or 4."; return 1
                fi
                local ratio
                case "$req_bits" in
                    4) ratio="3.90" ;; 3) ratio="4.60" ;; 2) ratio="6.40" ;;
                esac
                mkdir -p /run/greenboost
                { echo "enabled=1"; echo "bits=${req_bits}"; echo "head_dim=128"; echo "ratio=${ratio}"; echo "seed=42"; } > "${conf_path}.tmp" && mv "${conf_path}.tmp" "$conf_path"
                gb_ok "Wrote ${conf_path} (turbo${req_bits}, ratio ${ratio}×)"
                if [[ -c "$dev_path" ]]; then
                    python3 -c "
import fcntl, struct, os
_IOC_WRITE=1
def _iow(m,n,s): return (_IOC_WRITE<<30)|(s<<16)|(m<<8)|n
GB_IOCTL_SET_TURBOQUANT=_iow(ord('G'),10,16)
req=struct.pack('IIII',1,${req_bits},128,42)
fd=os.open('${dev_path}',os.O_RDWR|os.O_CLOEXEC)
fcntl.ioctl(fd,GB_IOCTL_SET_TURBOQUANT,req)
os.close(fd)
" 2>&1 && gb_ok "IOCTL sent to ${dev_path}" || gb_warn_ui "IOCTL failed - conf file is active"
                fi
            fi
            ;;
        disable)
            rm -f "$conf_path"
            gb_ok "Removed ${conf_path}"
            if [[ -c "$dev_path" ]]; then
                python3 -c "
import fcntl, struct, os
_IOC_WRITE=1
def _iow(m,n,s): return (_IOC_WRITE<<30)|(s<<16)|(m<<8)|n
GB_IOCTL_SET_TURBOQUANT=_iow(ord('G'),10,16)
req=struct.pack('IIII',0,0,128,42)
fd=os.open('${dev_path}',os.O_RDWR|os.O_CLOEXEC)
fcntl.ioctl(fd,GB_IOCTL_SET_TURBOQUANT,req)
os.close(fd)
" 2>&1 && gb_ok "IOCTL sent: TurboQuant disabled" || gb_warn_ui "IOCTL failed - conf file removed"
            fi
            ;;
        *)
            gb_warn_ui "Unknown sub-command: ${subcmd}"
            gb_info "Usage: greenboost turboquant [status|enable [2|3|4]|disable]"
            return 1
            ;;
    esac
}

# ---- Wizard (interactive menu - mirrors greenboost_setup.sh cmd_wizard) -----
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

# ---- Entry point -------------------------------------------------------

# Strip global flags before command dispatch.
# --profile <file>  Load module parameters from a profile file.
GB_ORIG_ARGS=("$@")   # preserved for exec-restart after update
GB_PROFILE_FILE=""    # path to user-supplied profile file (empty = use active profile)
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
    steam-launch-guide)  cmd_steam_launch_info    ;;
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
     
     
    turboquant)          cmd_turboquant "${@:2}" ;;
    show-commands)       cmd_show_commands      ;;
    wizard)              cmd_wizard             ;;
    help|--help|-h)      cmd_help "${@:2}"      ;;
    *) die "Unknown command: '$COMMAND'  - use: $0 help" ;;
esac

#!/usr/bin/env bash
# install_module.sh - GreenBoost kernel module - standalone DKMS installer
#
# Hardware-agnostic: works on AMD, Intel, any GPU, or no GPU.
# Installs only the kernel module via DKMS - no sysctl, no GRUB, no CUDA.
# The module auto-rebuilds on kernel upgrades (AUTOINSTALL=yes in dkms.conf).
#
# Usage:
#   sudo ./install_module.sh           - install via DKMS
#   sudo ./install_module.sh --uninstall - remove DKMS entry + unload module
#   sudo ./install_module.sh --status   - show DKMS registration state
#   sudo ./install_module.sh --help

set -euo pipefail

GB_VERSION="3.2"
DRIVER_NAME="greenboost"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DKMS_SRC="/usr/src/${DRIVER_NAME}-${GB_VERSION}"

# When sourced by a setup script to reuse DKMS functions, skip the entry point.
[[ "${GB_SOURCE_MODE:-0}" == "1" ]] && return 0 2>/dev/null || true

# ── Colour palette ────────────────────────────────────────────────────────────
if [[ "${COLORTERM:-}" =~ ^(truecolor|24bit)$ ]]; then
    C_VIOLET='\033[38;2;108;113;196m'
    C_LIME='\033[38;2;230;255;60m'
    C_CYAN='\033[38;2;48;200;255m'
    C_AMBER='\033[38;2;255;191;0m'
    C_RED='\033[38;2;255;92;50m'
    C_GRAY='\033[38;2;208;207;204m'
    C_DIM='\033[2m'
    C_BOLD='\033[1m'
else
    C_VIOLET='\033[0;34m'; C_LIME='\033[0;32m'; C_CYAN='\033[0;36m'
    C_AMBER='\033[1;33m';  C_RED='\033[0;31m';  C_GRAY='\033[0;37m'
    C_DIM='\033[2m';       C_BOLD='\033[1m'
fi
C_RESET='\033[0m'

gb_ok()   { printf '%b\n' "  ${C_LIME}✓${C_RESET}  ${C_GRAY}$*${C_RESET}"; }
gb_fail() { printf '%b\n' "  ${C_RED}✗${C_RESET}  $*" >&2; }
gb_warn() { printf '%b\n' "  ${C_AMBER}⚠${C_RESET}  $*"; }
gb_info() { printf '%b\n' "  ${C_VIOLET}◈${C_RESET}  ${C_GRAY}$*${C_RESET}"; }
gb_step() { printf '%b\n' "  ${C_CYAN}❯${C_RESET}  ${C_GRAY}$*${C_RESET}"; }
gb_die()  { gb_fail "$*"; exit 1; }

gb_header() {
    local cols; cols=$(tput cols 2>/dev/null || echo 64)
    local title=" GreenBoost v${GB_VERSION} - Kernel Module Installer"
    printf '\n%b\n' "${C_VIOLET}${C_BOLD}  ╔$(printf '═%.0s' $(seq 1 $((cols - 4))))╗${C_RESET}"
    printf '%b\n'   "${C_VIOLET}${C_BOLD}  ║${C_RESET} ${C_GRAY}${C_BOLD}${title}$(printf ' %.0s' $(seq 1 $((cols - 4 - ${#title} - 1))))${C_VIOLET}${C_BOLD}║${C_RESET}"
    printf '%b\n\n' "${C_VIOLET}${C_BOLD}  ╚$(printf '═%.0s' $(seq 1 $((cols - 4))))╝${C_RESET}"
}

need_root() { [[ $EUID -eq 0 ]] || gb_die "Root required. Use: sudo $0 $*"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
MODE="install"
for _arg in "$@"; do
    case "$_arg" in
        --uninstall|-u)  MODE="uninstall" ;;
        --status|-s)     MODE="status"    ;;
        --help|-h)
            gb_header
            printf '%b\n' "  ${C_GRAY}Usage:${C_RESET}"
            printf '%b\n' "    ${C_CYAN}sudo ./install_module.sh${C_RESET}            ${C_DIM}Install kernel module via DKMS${C_RESET}"
            printf '%b\n' "    ${C_CYAN}sudo ./install_module.sh --uninstall${C_RESET} ${C_DIM}Remove DKMS entry + unload module${C_RESET}"
            printf '%b\n' "    ${C_CYAN}sudo ./install_module.sh --status${C_RESET}    ${C_DIM}Show DKMS registration state${C_RESET}"
            printf '\n%b\n' "  ${C_DIM}This installer is hardware-agnostic: no GPU, CUDA, sysctl, or GRUB changes.${C_RESET}"
            echo ""
            exit 0
            ;;
    esac
done

gb_header

# ── Distro detection ──────────────────────────────────────────────────────────
_os_id=""
_os_like=""
if [[ -f /etc/os-release ]]; then
    _os_id=$(  grep -oP '^ID=\K.*'      /etc/os-release | tr -d '"')
    _os_like=$(grep -oP '^ID_LIKE=\K.*' /etc/os-release | tr -d '"' || true)
fi

_is_debian=0; _is_rhel=0; _is_arch=0
case "$_os_id" in
    ubuntu|debian|linuxmint|pop|elementary|zorin|kali) _is_debian=1 ;;
    rhel|centos|rocky|almalinux|fedora|ol)             _is_rhel=1   ;;
    arch|manjaro|endeavouros|cachyos|garuda)           _is_arch=1   ;;
    *)
        # fall back to ID_LIKE
        if printf '%s' "$_os_like" | grep -qiE 'debian|ubuntu'; then _is_debian=1
        elif printf '%s' "$_os_like" | grep -qiE 'rhel|fedora'; then _is_rhel=1
        elif printf '%s' "$_os_like" | grep -qi 'arch';          then _is_arch=1
        else gb_warn "Unknown distro '${_os_id}' - attempting generic install"; _is_debian=1
        fi
        ;;
esac

# ── STATUS ────────────────────────────────────────────────────────────────────
if [[ "$MODE" == "status" ]]; then
    gb_step "DKMS status for ${DRIVER_NAME}/${GB_VERSION}"
    if command -v dkms &>/dev/null; then
        dkms status "${DRIVER_NAME}/${GB_VERSION}" 2>/dev/null || gb_info "Not registered with DKMS"
    else
        gb_warn "dkms not installed"
    fi
    echo ""
    if lsmod 2>/dev/null | grep -q "^${DRIVER_NAME}\b"; then
        gb_ok "Module loaded: $(lsmod | grep "^${DRIVER_NAME}\b")"
    else
        gb_info "Module not currently loaded"
    fi
    [[ -c /dev/greenboost ]] && gb_ok "/dev/greenboost present" || gb_info "/dev/greenboost not found"
    exit 0
fi

# ── UNINSTALL ─────────────────────────────────────────────────────────────────
if [[ "$MODE" == "uninstall" ]]; then
    need_root --uninstall

    gb_step "Unloading kernel module (if loaded)..."
    if lsmod 2>/dev/null | grep -q "^${DRIVER_NAME}\b"; then
        rmmod "$DRIVER_NAME" 2>/dev/null && gb_ok "Module unloaded" || gb_warn "rmmod failed - reboot to complete removal"
    else
        gb_info "Module not loaded"
    fi

    gb_step "Removing DKMS entry..."
    if command -v dkms &>/dev/null && dkms status 2>/dev/null | grep -q "^${DRIVER_NAME}"; then
        rm -rf /var/lib/dkms/greenboost 2>/dev/null && gb_ok "DKMS entries removed"
    else
        gb_info "DKMS entry not found - skipping"
    fi

    gb_step "Removing source trees from /usr/src..."
    _any_src_removed=0
    for _src_tree in /usr/src/greenboost-*; do
        [[ -d "$_src_tree" ]] || continue
        rm -rf "$_src_tree" && (( _any_src_removed++ )) || true
    done
    (( _any_src_removed > 0 )) && gb_ok "Removed $_any_src_removed source tree(s)" \
                                || gb_info "No source trees found - skipping"

    gb_step "Removing modprobe configuration..."
    if [[ -f /etc/modprobe.d/greenboost.conf ]]; then
        rm -f /etc/modprobe.d/greenboost.conf
        gb_ok "Removed /etc/modprobe.d/greenboost.conf"
    fi

    echo ""
    gb_ok "Uninstall complete."
    exit 0
fi

# ── INSTALL ───────────────────────────────────────────────────────────────────
need_root

# 0/4 - Purge all previous GreenBoost residues before doing anything else.
#        This prevents stale DKMS state, old source trees, and leftover kernel
#        modules from causing failures on reinstall.
gb_step "[0/4] Cleaning previous GreenBoost installation residues..."
if lsmod 2>/dev/null | grep -q "^${DRIVER_NAME}\b"; then
    rmmod "$DRIVER_NAME" 2>/dev/null && gb_ok "Module unloaded" \
        || gb_warn "rmmod failed - continuing (reboot if module stays stuck)"
fi
rm -rf /var/lib/dkms/greenboost 2>/dev/null || true
for _old_src in /usr/src/greenboost-*; do
    [[ -d "$_old_src" ]] && rm -rf "$_old_src" || true
done
find /lib/modules -name "greenboost.ko*" -delete 2>/dev/null || true
{ depmod -a 2>/dev/null || true; }
gb_ok "Residues cleared"

# 1/4 - Install minimal build dependencies
gb_step "[1/4] Installing build dependencies..."

if (( _is_debian )); then
    apt-get update -qq 2>/dev/null
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        dkms build-essential "linux-headers-$(uname -r)" kmod 2>/dev/null \
        || gb_die "apt-get failed - check your package manager"

elif (( _is_rhel )); then
    _kernel_devel="kernel-devel-$(uname -r)"
    if (( $(rpm -E '%{rhel}' 2>/dev/null || echo 0) >= 8 )) || \
       grep -qiE '^ID=fedora' /etc/os-release 2>/dev/null; then
        dnf install -y -q dkms gcc make "$_kernel_devel" kmod 2>/dev/null \
            || dnf install -y -q dkms gcc make kernel-devel kmod 2>/dev/null \
            || gb_die "dnf failed - check your package manager"
    else
        yum install -y -q dkms gcc make "$_kernel_devel" kmod 2>/dev/null \
            || yum install -y -q dkms gcc make kernel-devel kmod 2>/dev/null \
            || gb_die "yum failed - check your package manager"
    fi

elif (( _is_arch )); then
    # Detect running kernel variant (linux, linux-lts, linux-zen, etc.)
    _kernel_pkg="linux-headers"
    _running_ver=$(uname -r)
    if printf '%s' "$_running_ver" | grep -q '\-lts'; then
        _kernel_pkg="linux-lts-headers"
    elif printf '%s' "$_running_ver" | grep -q '\-zen'; then
        _kernel_pkg="linux-zen-headers"
    elif printf '%s' "$_running_ver" | grep -q '\-hardened'; then
        _kernel_pkg="linux-hardened-headers"
    fi
    pacman -Sy --noconfirm --needed dkms base-devel "$_kernel_pkg" 2>/dev/null \
        || gb_die "pacman failed - check your package manager"
fi

gb_ok "Build dependencies ready"

# 2/4 - Copy source tree to /usr/src
gb_step "[2/4] Copying source to ${DKMS_SRC}..."

# Required files for module build
_src_files=(
    greenboost.c
    Kbuild
    dkms.conf
    greenboost_ioctl.h
)
_src_dirs=(
    features
)

# Verify source files exist
for _f in "${_src_files[@]}"; do
    [[ -f "$SRC_DIR/$_f" ]] || gb_die "Missing source file: $SRC_DIR/$_f"
done
for _d in "${_src_dirs[@]}"; do
    [[ -d "$SRC_DIR/$_d" ]] || gb_die "Missing source directory: $SRC_DIR/$_d"
done

mkdir -p "$DKMS_SRC"
for _f in "${_src_files[@]}"; do
    cp "$SRC_DIR/$_f" "$DKMS_SRC/"
done
for _d in "${_src_dirs[@]}"; do
    cp -r "$SRC_DIR/$_d" "$DKMS_SRC/"
done

gb_ok "Source copied to $DKMS_SRC"

# 3/4 - Register, build, and install via DKMS
gb_step "[3/4] Registering and building kernel module via DKMS..."

# Idempotent add - wipe all stale entries (any version) before registering
gb_info "Removing stale DKMS entry..."
rm -rf /var/lib/dkms/greenboost 2>/dev/null || true

dkms add "${DRIVER_NAME}/${GB_VERSION}" \
    || gb_die "dkms add failed"
gb_ok "DKMS entry registered"

dkms build "${DRIVER_NAME}/${GB_VERSION}" \
    || gb_die "dkms build failed - check kernel headers for $(uname -r)"
gb_ok "Module built"

dkms install "${DRIVER_NAME}/${GB_VERSION}" \
    || gb_die "dkms install failed"
gb_ok "Module installed"

# Also build for every OTHER kernel already present on disk (e.g. a second
# kernel installed via ~/Dev/kernel_inference but not yet booted) , the calls
# above only ever cover $(uname -r), so any other installed kernel would stay
# silently unbuilt until someone notices and reruns dkms for it by hand.
# Best-effort: never fail the install over this.
gb_info "Building for any other installed kernels..."
dkms autoinstall -m "${DRIVER_NAME}" -v "${GB_VERSION}" 2>&1 | while read -r _l; do gb_info "$_l"; done || true

# 4/4 - Detect hardware and load module with tuned parameters
gb_step "[4/4] Detecting hardware and loading kernel module..."

# GPU VRAM - nvidia-smi when available, otherwise 0 (module uses its own default)
_phys_gb=0
if command -v nvidia-smi &>/dev/null; then
    _vram_mib=$(timeout 3 nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null \
        | head -1 | grep -oP '[0-9]+' || true)
    if [[ -n "$_vram_mib" ]] && (( _vram_mib > 0 )); then
        _phys_gb=$(( (_vram_mib + 512) / 1024 ))   # round to nearest GB
        gb_info "GPU VRAM detected: ${_phys_gb} GB"
    fi
fi
if (( _phys_gb == 0 )); then
    gb_info "No NVIDIA GPU detected - physical_vram_gb left at module default"
fi

# System RAM → virtual_vram_gb follows gb_calc_ddr_cap_gb() in setup.sh:
#   < 64 GB → 70%, >= 64 GB → 80%.  safety_reserve = 8% (min 4, max 10).
_total_ram_kb=$(grep -m1 '^MemTotal:' /proc/meminfo | awk '{print $2}')
_total_ram_gb=$(( (_total_ram_kb + 524288) / 1048576 ))   # round to nearest GB
if (( _total_ram_gb >= 64 )); then
    _virt_gb=$(( _total_ram_gb * 80 / 100 ))
else
    _virt_gb=$(( _total_ram_gb * 70 / 100 ))
fi
_reserve_gb=$(( _total_ram_gb * 8 / 100 ))
(( _virt_gb   < 4  )) && _virt_gb=4
(( _reserve_gb < 4  )) && _reserve_gb=4
(( _reserve_gb > 10 )) && _reserve_gb=10
gb_info "System RAM: ${_total_ram_gb} GB  →  pool ${_virt_gb} GB, reserve ${_reserve_gb} GB"

# Write hardware-tuned modprobe config
mkdir -p /etc/modprobe.d
{
    printf '# GreenBoost - hardware-tuned parameters (auto-generated by light install)\n'
    printf '# CPU: %s\n' "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
    if (( _phys_gb > 0 )); then
        printf 'options %s physical_vram_gb=%d virtual_vram_gb=%d safety_reserve_gb=%d\n' \
            "$DRIVER_NAME" "$_phys_gb" "$_virt_gb" "$_reserve_gb"
    else
        printf 'options %s virtual_vram_gb=%d safety_reserve_gb=%d\n' \
            "$DRIVER_NAME" "$_virt_gb" "$_reserve_gb"
    fi
} > /etc/modprobe.d/greenboost.conf
gb_ok "modprobe config written: /etc/modprobe.d/greenboost.conf"

modprobe "$DRIVER_NAME" \
    || gb_die "modprobe failed - check dmesg for errors"
gb_ok "Module loaded"

# Persist across reboots , a successful modprobe here only affects the current
# boot. Without this file, systemd-modules-load.service has nothing telling it
# to load greenboost on the NEXT boot, and the module silently stays unloaded
# until someone notices and modprobes it by hand (confirmed gap 2026-07-10:
# this file was missing here even though the module loaded fine at install time).
if [[ ! -f /etc/modules-load.d/greenboost.conf ]]; then
    echo "$DRIVER_NAME" > /etc/modules-load.d/greenboost.conf
    gb_ok "Boot-time autoload registered: /etc/modules-load.d/greenboost.conf"
else
    gb_info "Boot-time autoload already registered: /etc/modules-load.d/greenboost.conf"
fi

# Boot guard: self-heals a missing .ko for the running kernel (e.g. a kernel
# upgrade whose DKMS autoinstall never ran) BEFORE systemd-modules-load.service
# tries to load it, so the module keeps loading at boot even across a kernel
# bump that this install didn't know about.
if [[ -f "$SRC_DIR/greenboost_boot_guard.sh" && -f "$SRC_DIR/greenboost-boot-guard.service" ]]; then
    install -m 755 "$SRC_DIR/greenboost_boot_guard.sh" /usr/local/sbin/greenboost_boot_guard.sh
    install -m 644 "$SRC_DIR/greenboost-boot-guard.service" /etc/systemd/system/greenboost-boot-guard.service
    systemctl daemon-reload 2>/dev/null || true
    systemctl enable greenboost-boot-guard.service &>/dev/null \
        && gb_ok "Boot guard installed + enabled: greenboost-boot-guard.service" \
        || gb_warn "Could not enable greenboost-boot-guard.service (no systemd?)"
else
    gb_warn "greenboost_boot_guard.sh / .service not found next to this script , skipping boot guard"
fi

# ── Device permissions: udev rule + group membership ─────────────────────────
# /dev/greenboost defaults to 0600 root:root without a udev rule.
# Add the udev rule so any member of the 'video' group can open the device
# (required for gb_quant int4 FLUX inference and DMA-BUF T2 overflow).
gb_step "Installing /dev/greenboost udev rule..."
mkdir -p /etc/udev/rules.d
cat > /etc/udev/rules.d/99-greenboost.rules << 'UDEVEOF'
# GreenBoost kernel module - allow video group to access /dev/greenboost
KERNEL=="greenboost", GROUP="video", MODE="0660"
UDEVEOF
udevadm control --reload-rules 2>/dev/null || true
udevadm trigger --name-match=greenboost 2>/dev/null \
    || udevadm trigger --subsystem-match=greenboost 2>/dev/null \
    || true
udevadm settle 2>/dev/null || true
gb_ok "udev rule installed: /etc/udev/rules.d/99-greenboost.rules"

# Add the invoking user (via SUDO_USER) to the 'video' group.
# Without this the udev rule's GROUP="video" does nothing for the user.
if [[ -n "${SUDO_USER:-}" ]]; then
    if ! id -nG "$SUDO_USER" 2>/dev/null | tr ' ' '\n' | grep -q '^video$'; then
        usermod -aG video "$SUDO_USER" \
            && gb_ok "User '$SUDO_USER' added to 'video' group (re-login or: newgrp video)" \
            || gb_warn "usermod failed , add $SUDO_USER to 'video' group manually"
    else
        gb_info "User '$SUDO_USER' already in 'video' group"
    fi
    # Immediate ACL so current shell can open /dev/greenboost without re-login
    if [[ -c /dev/greenboost ]] && command -v setfacl &>/dev/null; then
        setfacl -m "u:${SUDO_USER}:rw" /dev/greenboost 2>/dev/null \
            && gb_info "ACL: /dev/greenboost → u:${SUDO_USER}:rw (current session)" \
            || true
    fi
fi

# ── Write build_info stamp ────────────────────────────────────────────────────
mkdir -p /etc/greenboost
_git_hash=$(git -C "$SRC_DIR" rev-parse --short HEAD 2>/dev/null || echo "nogit")
printf 'BUILD_ID=%s\nBUILD_VERSION=%s\nBUILD_HOST=%s\nBUILD_GIT=%s\nBUILD_EPOCH=%s\n' \
    "$(date +%d%m-%H%M)" "$GB_VERSION" "$(hostname)" "$_git_hash" "$(date +%s)" \
    > /etc/greenboost/build_info
gb_ok "Build stamp written: /etc/greenboost/build_info"

# ── Install greenboost_setup.sh to a stable system path ─────────────────────
# F-L2-05/F-L2-06: the wrapper must not embed build-time SRC_DIR. Install the
# setup script to /usr/local/share/greenboost/ so the wrapper can reference it
# via a fixed path that survives source-tree moves or deletions.
install -d /usr/local/share/greenboost
install -m 755 "$SRC_DIR/greenboost_setup.sh" /usr/local/share/greenboost/greenboost_setup.sh
gb_ok "greenboost_setup.sh installed: /usr/local/share/greenboost/greenboost_setup.sh"

# ── Install greenboost wrapper script ────────────────────────────────────────
cat > /usr/local/bin/greenboost << 'WRAPEOF'
#!/usr/bin/env bash
# GreenBoost CLI wrapper (installed by install_module.sh)
GB_SETUP=/usr/local/share/greenboost/greenboost_setup.sh
GB_SHIM=${GREENBOOST_SHIM:-/usr/local/lib/libgreenboost_cuda.so}
case "$1" in
    status)       exec "$GB_SETUP" status "${@:2}" ;;
    benchmark)    exec "$GB_SETUP" benchmark "${@:2}" ;;
    load)         exec "$GB_SETUP" load "${@:2}" ;;
    unload)       exec "$GB_SETUP" unload ;;
    feed)         exec "$GB_SETUP" feed "${@:2}" ;;
    connect)      exec "$GB_SETUP" connect "${@:2}" ;;
    disconnect)   exec "$GB_SETUP" disconnect "${@:2}" ;;
    cluster)      exec "$GB_SETUP" cluster ;;
    diag)         exec "$GB_SETUP" diag "${@:2}" ;;
    build|build-info) exec "$GB_SETUP" build-info "${@:2}" ;;
    turboquant)   exec "$GB_SETUP" turboquant "${@:2}" ;;
    setup|install|full-install) exec "$GB_SETUP" "$@" ;;
    run)          shift; GREENBOOST_ACTIVE=1 LD_PRELOAD="$GB_SHIM" "$@" ;;
    help|--help|-h|"") exec "$GB_SETUP" show-commands ;;
    *)            exec "$GB_SETUP" "$@" ;;
esac
WRAPEOF
chmod +x /usr/local/bin/greenboost
gb_ok "greenboost wrapper installed: /usr/local/bin/greenboost"

# ── Build and install CUDA shim + native libs ────────────────────────────────
# DKMS only handles the kernel module (.ko).  The LD_PRELOAD shim
# (libgreenboost_cuda.so) and any other native libs require a separate
# 'make all install-libs' pass in the source tree.
# We clean first to avoid root-owned stale artifacts from a prior sudo make,
# then chown the tree back to the calling developer so rebuilds don't require sudo.
gb_step "Building and installing GreenBoost CUDA shim + native libs..."
_shim_log=$(mktemp /tmp/gb_shim_build.XXXXX.log)
make -C "$SRC_DIR" clean all install-libs >"$_shim_log" 2>&1 \
    || { echo "" >&2; cat "$_shim_log" >&2; rm -f "$_shim_log"; \
         gb_die "Shim build failed , see output above"; }
rm -f "$_shim_log"
# Restore ownership so the developer can rebuild without sudo.
if [[ -n "${SUDO_USER:-}" ]]; then
    chown -R "${SUDO_USER}:$(id -gn "${SUDO_USER}" 2>/dev/null || echo users)" \
        "$SRC_DIR" 2>/dev/null || true
fi
gb_ok "CUDA shim + native libs installed"

# ── Install GreenBoost Python orchestration stack ────────────────────────────
# The Python stack (gb_telemetry, gb_quant, gb_diffusion_orch, etc.) must be
# importable system-wide , by the active conda env, any venv, and system Python ,
# without the user sourcing profile.d manually.  We install:
#   /usr/local/lib/greenboost/   , the package directory
#   /etc/profile.d/greenboost_pythonpath.sh , PYTHONPATH for login shells
#   greenboost.pth               , dist-packages + conda pth drop-in
gb_step "Installing GreenBoost Python orchestration stack..."
GB_PY_DEST="/usr/local/lib/greenboost"
mkdir -p "$GB_PY_DEST"
# Deliberately a LIBRARY-ONLY subset, not gb-synapse's HTTP-serving stack —
# this is the standalone kernel-module installer, so it ships just enough for
# a consumer script to `import gb_quant`/`gb_attn`/etc. directly. Each of
# these files is self-contained against exactly this list (verified:
# gb_synapse_fallback.py imports only gb_quant + stdlib/torch/aiohttp, so it
# works standalone here as a library AND as its own runnable single-request
# server — it is NOT missing gb_synapse.py/_api/_backends/_mcp/gb_rotator.py/
# gb_diffusion_server.py by omission, those simply aren't this script's job).
# `greenboost synapse serve` / the :11435 proxy / cluster RPC split need the
# full stack — that's `greenboost_setup.sh`'s Full Install, not this script.
GB_PY_FILES=(
    gb_init.py
    gb_quant.py gb_quant_tq.py gb_attn.py gb_synapse_fallback.py
    gb_telemetry.py gb_stream_sched.py gb_model_tier.py
    gb_mem_pool.py gb_diffusion_orch.py
    gb_stability_monitor.py gb_feeder_diag.py
)
_installed=0
for _f in "${GB_PY_FILES[@]}"; do
    if [[ -f "$SRC_DIR/$_f" ]]; then
        install -m 644 "$SRC_DIR/$_f" "$GB_PY_DEST/$_f"
        (( _installed++ )) || true
    fi
done
# Package __init__.py
if [[ ! -f "$GB_PY_DEST/__init__.py" ]]; then
    printf '# GreenBoost Python orchestration stack , auto-generated\n' \
        > "$GB_PY_DEST/__init__.py"
fi
# profile.d export
cat > /etc/profile.d/greenboost_pythonpath.sh << 'PROFEOF'
# GreenBoost Python orchestration stack
if [[ -d /usr/local/lib/greenboost ]]; then
    export PYTHONPATH="/usr/local/lib/greenboost${PYTHONPATH:+:$PYTHONPATH}"
fi
PROFEOF
chmod 644 /etc/profile.d/greenboost_pythonpath.sh
# Drop a .pth + sitecustomize into every system dist-packages.
# The .pth adds the path AND imports greenboost_sitecustomize so that
# processes launched with GREENBOOST_ACTIVE=1 get gb_init auto-imported.
_GB_SC_CONTENT='# GreenBoost auto-bootstrap , generated by install_module.sh
import os as _os
if _os.environ.get("GREENBOOST_ACTIVE") == "1":
    try:
        import gb_init  # noqa: F401
    except Exception:
        pass
'
for _dp in /usr/local/lib/python3*/dist-packages; do
    if [[ -d "$_dp" ]]; then
        printf '%s' "$_GB_SC_CONTENT" > "$_dp/greenboost_sitecustomize.py"
        printf '/usr/local/lib/greenboost\nimport greenboost_sitecustomize\n' \
            > "$_dp/greenboost.pth"
    fi
done
# Drop into active conda env if present
if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
    for _dp in "$CONDA_PREFIX"/lib/python3*/site-packages; do
        if [[ -d "$_dp" ]]; then
            printf '%s' "$_GB_SC_CONTENT" > "$_dp/greenboost_sitecustomize.py"
            printf '/usr/local/lib/greenboost\nimport greenboost_sitecustomize\n' \
                > "$_dp/greenboost.pth"
        fi
    done
fi
gb_ok "Python stack installed: $_installed files → $GB_PY_DEST"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
printf '%b\n' "  ${C_VIOLET}${C_BOLD}Installation complete${C_RESET}"
echo ""
if [[ -c /dev/greenboost ]]; then
    gb_ok "/dev/greenboost is present - module active"
else
    gb_warn "/dev/greenboost not found - check: dmesg | grep greenboost"
fi
echo ""
gb_info "DKMS will automatically rebuild the module on kernel upgrades."
gb_info "Module parameters: /etc/modprobe.d/greenboost.conf"
gb_info "Run diagnostics: greenboost diag feeder"
echo ""

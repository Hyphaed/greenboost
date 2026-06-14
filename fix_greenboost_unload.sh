#!/usr/bin/env bash
# fix_greenboost_unload.sh
# Fixes stuck greenboost module caused by missing kretprobe guard in compiled .ko.
#
# Root cause: greenboost.ko was compiled before the g_kretprobe_ok guard was added
# to gb_exit(). Without it, gb_exit() calls unregister_kretprobe() unconditionally;
# when the kretprobe failed to register at boot, gb_sysinfo_krp.rph is NULL, causing
# a kernel NULL-ptr oops in rethook_free(). rmmod is killed, module stays in
# MODULE_STATE_GOING forever - blocking every subsequent rmmod and every reboot.
#
# This script recompiles greenboost.ko from the current source (which has the fix)
# and installs it so the next boot loads the corrected module.
#
# Usage: sudo bash fix_greenboost_unload.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="greenboost"
GB_VERSION="3.0"
KVER=$(uname -r)
DKMS_ROOT="/usr/src/greenboost-${GB_VERSION}"

C_GREEN='\033[0;32m'; C_YELLOW='\033[1;33m'; C_RED='\033[0;31m'
C_CYAN='\033[0;36m'; C_BOLD='\033[1m'; C_RESET='\033[0m'
ok()   { echo -e "${C_GREEN}  ✓${C_RESET}  $*"; }
warn() { echo -e "${C_YELLOW}  ⚠${C_RESET}  $*"; }
err()  { echo -e "${C_RED}  ✗${C_RESET}  $*"; exit 1; }
step() { echo -e "\n${C_CYAN}${C_BOLD}[$1]${C_RESET}  $2"; }

[[ $EUID -eq 0 ]] || err "Run as root:  sudo bash $0"

echo ""
echo -e "${C_BOLD}GreenBoost Stuck-Module Fix${C_RESET}"
echo    "────────────────────────────────────────────────────────────────"
echo    "Root cause: gb_exit() in the compiled .ko calls unregister_kretprobe()"
echo    "unconditionally. When the kretprobe fails to register at boot,"
echo    "gb_sysinfo_krp.rph is NULL → rethook_free() oops → rmmod killed"
echo    "→ module stuck in MODULE_STATE_GOING → reboot hangs."
echo    ""
echo    "The guard (g_kretprobe_ok check) already exists in greenboost.c."
echo    "This script recompiles and installs the fixed module."
echo    "────────────────────────────────────────────────────────────────"

# ── Current state ────────────────────────────────────────────────────────────
step "0/4" "Current module state"
if grep -q "^${DRIVER}.*Unloading" /proc/modules 2>/dev/null; then
    warn "Module is stuck in MODULE_STATE_GOING (Unloading) - will clear on reboot"
elif lsmod | grep -q "^${DRIVER} "; then
    warn "Module is currently loaded"
else
    ok "Module is not loaded"
fi
[[ -f /etc/modprobe.d/99-greenboost-blacklist.conf ]] \
    && warn "Stale blacklist conf found - will be removed"

# ── Step 1: Compile ───────────────────────────────────────────────────────────
step "1/4" "Compiling fixed module (skipping rmmod - module is stuck)"
# Pre-flight: verify this is a real GreenBoost source tree before invoking make
# as root.  A script placed in a foreign directory would otherwise execute an
# unrelated Makefile with root privileges.
[[ -f "$SCRIPT_DIR/greenboost.c" ]]   || err "greenboost.c not found in $SCRIPT_DIR - wrong directory?"
[[ -f "$SCRIPT_DIR/Kbuild" ]]          || err "Kbuild not found in $SCRIPT_DIR - wrong directory?"
[[ -d "/lib/modules/${KVER}/build" ]] \
    || [[ -d "/usr/src/kernels/${KVER}" ]] \
    || err "Kernel headers not found for ${KVER} - install kernel-devel and retry"
# Build only the kernel module, not shim/libs - those are not the problem.
make -C "$SCRIPT_DIR" module 2>&1 \
    || err "Compilation failed - see errors above"
ok "greenboost.ko compiled with kretprobe guard"

# ── Step 2: Install fixed .ko ─────────────────────────────────────────────────
step "2/4" "Installing fixed .ko to /lib/modules"
INSTALL_DIR="/lib/modules/${KVER}/extra"
mkdir -p "$INSTALL_DIR"
cp "${SCRIPT_DIR}/greenboost.ko" "${INSTALL_DIR}/greenboost.ko"
depmod -a "$KVER"
ok "Installed to ${INSTALL_DIR}/greenboost.ko"
ok "depmod -a complete"

# Also update the DKMS source tree so kernel updates pick up the fix.
if command -v dkms &>/dev/null && [[ -d "$DKMS_ROOT" ]]; then
    cp "$SCRIPT_DIR/greenboost.c" "$DKMS_ROOT/"
    cp "$SCRIPT_DIR/greenboost_ioctl.h" "$DKMS_ROOT/" 2>/dev/null || true
    if [[ -d "$SCRIPT_DIR/features" ]]; then
        cp -r "$SCRIPT_DIR/features" "$DKMS_ROOT/"
    fi
    ok "Updated DKMS source tree at $DKMS_ROOT"
fi

# ── Step 3: Clear stale config blockers ──────────────────────────────────────
step "3/4" "Clearing stale config"

# Remove the blacklist that was written by a previous boot cleanup - it would
# block modprobe/modules-load from loading the freshly installed module.
if rm -f /etc/modprobe.d/99-greenboost-blacklist.conf 2>/dev/null; then
    ok "Removed /etc/modprobe.d/99-greenboost-blacklist.conf"
fi

# Ensure modules-load.d entry exists so the module auto-loads on next boot.
if [[ ! -f /etc/modules-load.d/greenboost.conf ]]; then
    echo "greenboost" > /etc/modules-load.d/greenboost.conf
    ok "Restored /etc/modules-load.d/greenboost.conf"
else
    ok "/etc/modules-load.d/greenboost.conf is in place"
fi

# If the CUDA shim is missing but ld.so.preload still references it, remove
# the stale entry to prevent 'cannot be preloaded' errors at boot.
if [[ -f /etc/ld.so.preload ]] && grep -q "libgreenboost" /etc/ld.so.preload; then
    if [[ ! -f /usr/local/lib/libgreenboost_cuda.so ]]; then
        sed -i '/libgreenboost/d' /etc/ld.so.preload
        warn "Removed stale LD_PRELOAD entry (shim missing - will be restored by full-install)"
    fi
fi

# ── Step 4: Summary ───────────────────────────────────────────────────────────
step "4/4" "Done"
echo ""
echo -e "${C_BOLD}════════════════════════════════════════════════════════════════"
echo    " Fixed module ready.  Next steps:"
echo    "════════════════════════════════════════════════════════════════${C_RESET}"
echo    ""
echo    " 1.  REBOOT - the stuck module clears on boot, then the fixed"
echo    "     greenboost.ko loads automatically."
echo    ""
echo    "     If 'sudo reboot' hangs (the stuck module may block soft reboot),"
echo    "     use one of these instead:"
echo    ""
echo    "     a) Hold the power button 5 s  (cleanest, hardware forced off)"
echo    ""
echo    "     b) Emergency SysRq reboot (syncs filesystems first - safe):"
echo -e "          ${C_CYAN}echo s > /proc/sysrq-trigger${C_RESET}  # sync all filesystems"
echo -e "          ${C_CYAN}echo u > /proc/sysrq-trigger${C_RESET}  # remount read-only"
echo -e "          ${C_CYAN}echo b > /proc/sysrq-trigger${C_RESET}  # immediate reboot"
echo    ""
echo    " 2.  After boot, run full-install:"
echo -e "          ${C_CYAN}sudo ./greenboost_setup.sh full-install${C_RESET}"
echo    ""
echo    "════════════════════════════════════════════════════════════════"
echo    ""

#!/usr/bin/env bash
# configure_boot_machine.sh - apply greenboost's boot-persistence fixes to THIS
# machine right now, and install the boot-time self-heal guard.
#
# Context (2026-07-10): the running kernel (7.1.3-hyphaed, installed via
# ~/Dev/kernel_inference) had no matching DKMS-built greenboost.ko , only
# 7.1.2-hyphaed did , and /etc/modules-load.d/greenboost.conf was missing
# entirely, so greenboost never loaded on this boot. Root cause: none of
# greenboost's three install paths (install_module.sh, `make install`,
# greenboost_setup.sh's Full Install) ever wrote the modules-load.d file, and
# their `dkms install` calls only ever cover $(uname -r), silently skipping
# any OTHER kernel already on disk. All three are now fixed in-repo; this
# script applies the same fixes to the machine's CURRENT state without
# requiring a full reinstall, and installs a boot-time self-heal guard as a
# second line of defense for future kernel upgrades.
#
# Usage: sudo ./configure_boot_machine.sh

set -uo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GB_VERSION="3.2"
DRIVER_NAME="greenboost"

C_LIME='\033[0;32m'; C_RED='\033[0;31m'; C_AMBER='\033[1;33m'; C_CYAN='\033[0;36m'; C_RESET='\033[0m'
ok()   { printf "  ${C_LIME}✓${C_RESET}  %s\n" "$*"; }
warn() { printf "  ${C_AMBER}⚠${C_RESET}  %s\n" "$*"; }
fail() { printf "  ${C_RED}✗${C_RESET}  %s\n" "$*"; }
step() { printf "\n${C_CYAN}❯ %s${C_RESET}\n" "$*"; }

[[ $EUID -eq 0 ]] || { fail "Root required. Use: sudo $0"; exit 1; }

step "[1/5] Building greenboost for every installed kernel"
if ! command -v dkms &>/dev/null; then
    warn "dkms not found , install_module.sh installs it, running that first"
fi
if ! dkms status "${DRIVER_NAME}" 2>/dev/null | grep -q .; then
    warn "greenboost not registered with DKMS yet , running install_module.sh first"
    [[ -x "$SRC_DIR/install_module.sh" ]] || { fail "install_module.sh not found in $SRC_DIR"; exit 1; }
    "$SRC_DIR/install_module.sh" || { fail "install_module.sh failed , see output above"; exit 1; }
    ok "install_module.sh complete"
fi
dkms autoinstall -m "${DRIVER_NAME}" -v "${GB_VERSION}" 2>&1 | sed 's/^/    /'
ok "dkms autoinstall complete"

step "[2/5] Verifying kernel coverage"
_missing=0
for _kdir in /lib/modules/*/; do
    _kver="$(basename "$_kdir")"
    [[ -d "${_kdir}kernel" ]] || continue   # skip anything without a real module tree
    if find "$_kdir" -name 'greenboost.ko*' 2>/dev/null | grep -q .; then
        ok "greenboost.ko present for $_kver"
    else
        warn "no greenboost.ko for $_kver (no kernel headers installed for it? check: dkms build ${DRIVER_NAME}/${GB_VERSION} -k $_kver)"
        _missing=$((_missing + 1))
    fi
done
(( _missing == 0 )) || warn "$_missing kernel(s) still missing a build , usually means missing headers for that kernel"

step "[3/5] Boot-time autoload (modules-load.d)"
if [[ ! -f /etc/modules-load.d/greenboost.conf ]]; then
    echo "$DRIVER_NAME" > /etc/modules-load.d/greenboost.conf
    ok "Created /etc/modules-load.d/greenboost.conf"
else
    ok "/etc/modules-load.d/greenboost.conf already present"
fi

step "[4/5] Installing boot-time self-heal guard"
if [[ -f "$SRC_DIR/greenboost_boot_guard.sh" && -f "$SRC_DIR/greenboost-boot-guard.service" ]]; then
    install -m 755 "$SRC_DIR/greenboost_boot_guard.sh" /usr/local/sbin/greenboost_boot_guard.sh
    install -m 644 "$SRC_DIR/greenboost-boot-guard.service" /etc/systemd/system/greenboost-boot-guard.service
    systemctl daemon-reload
    systemctl enable greenboost-boot-guard.service &>/dev/null \
        && ok "greenboost-boot-guard.service installed + enabled" \
        || fail "systemctl enable failed , check systemctl status greenboost-boot-guard.service"
else
    fail "greenboost_boot_guard.sh / .service not found in $SRC_DIR , skipping"
fi

step "[5/5] Loading the module now (this boot)"
depmod -a "$(uname -r)" 2>/dev/null || true
if lsmod | grep -q '^greenboost\b'; then
    ok "greenboost already loaded for $(uname -r)"
elif modprobe "$DRIVER_NAME" 2>&1 | sed 's/^/    /'; then
    ok "greenboost loaded for $(uname -r)"
else
    fail "modprobe failed , check dmesg | grep greenboost"
fi

step "Final status"
echo "  dkms status:"
dkms status "${DRIVER_NAME}" 2>/dev/null | sed 's/^/    /'
echo "  modules-load.d:"
cat /etc/modules-load.d/greenboost.conf 2>/dev/null | sed 's/^/    /'
echo "  lsmod:"
lsmod | grep '^greenboost' | sed 's/^/    /' || echo "    (not loaded)"
echo "  boot guard service:"
systemctl is-enabled greenboost-boot-guard.service 2>/dev/null | sed 's/^/    enabled: /'
echo
ok "Done. This machine now rebuilds + loads greenboost for every installed kernel,"
ok "and self-heals at boot if a future kernel upgrade slips through DKMS autoinstall."

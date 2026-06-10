#!/usr/bin/env bash
# fix_usb.sh - full system sanity check + USB mass-storage fix.
#
# Addresses all known issues on this system:
#   1. USB drive detected (lsusb) but no /dev/sdX appears
#      Root cause: kernel tried to auto-load usb_storage for the SanDisk but
#      the load got stuck in MODULE_STATE_LOADING; every subsequent modprobe
#      blocks waiting for it - same class of bug as the greenboost unload hang.
#      Fix: bypass modprobe; bind the usb-storage driver via sysfs directly.
#
#   2. AppArmor profile integrity after GreenBoost injection
#      GreenBoost injects into every AppArmor profile; checks for DENIED
#      events on kmod, udev, systemd that would silently block operations.
#
#   3. GreenBoost module stuck in MODULE_STATE_GOING
#      Causes rmmod failure and reboot hang. Detected here; fix_greenboost_unload.sh
#      handles the actual repair.
#
#   4. Reboot always hangs
#      Caused by greenboost stuck module and/or NVIDIA driver saving GPU state.
#      Verifies that both fixes are installed.
#
# Usage:
#   sudo bash fix_usb.sh
#   sudo bash fix_usb.sh --force-rebind    # also rebind xhci_hcd (full USB reset)

set -euo pipefail

C_RED='\033[0;31m'; C_GREEN='\033[0;32m'; C_YELLOW='\033[1;33m'
C_CYAN='\033[0;36m'; C_BOLD='\033[1m'; C_DIM='\033[2m'; C_RESET='\033[0m'
C_VIOLET='\033[0;35m'

ok()    { echo -e "${C_GREEN}  ✓${C_RESET}  $*"; }
warn()  { echo -e "${C_YELLOW}  ⚠${C_RESET}  $*"; }
bad()   { echo -e "${C_RED}  ✗${C_RESET}  $*"; ISSUES=$(( ISSUES + 1 )); }
fixed() { echo -e "${C_GREEN}  ✓ FIXED${C_RESET}  $*"; FIXED=$(( FIXED + 1 )); }
step()  { echo -e "\n${C_CYAN}${C_BOLD}[$1]${C_RESET}  $2"; }
die()   { echo -e "${C_RED}  ✗${C_RESET}  $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Run as root:  sudo bash $0"

FORCE_REBIND=0
for _arg in "$@"; do [[ "$_arg" == "--force-rebind" ]] && FORCE_REBIND=1; done

ISSUES=0
FIXED=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo -e "${C_VIOLET}${C_BOLD}System Sanity Check + USB Fix${C_RESET}"
echo    "────────────────────────────────────────────────────────────────"

# ═══════════════════════════════════════════════════════════════
# PART A - SYSTEM CHECKS
# ═══════════════════════════════════════════════════════════════

# ── A1: GreenBoost module state ───────────────────────────────────────────────
step "A1" "GreenBoost kernel module state"

gb_state=""
if grep -q "^greenboost " /proc/modules 2>/dev/null; then
    gb_state=$(awk '/^greenboost /{print $5}' /proc/modules 2>/dev/null || echo "Unknown")
    if [[ "$gb_state" == "Unloading" ]]; then
        bad "greenboost module is STUCK in MODULE_STATE_GOING (Unloading)"
        echo -e "  ${C_DIM}This also causes reboot hangs - the stuck module blocks soft reboot.${C_RESET}"
        echo -e "  ${C_DIM}Fix:  sudo bash ${SCRIPT_DIR}/fix_greenboost_unload.sh${C_RESET}"
    elif [[ "$gb_state" == "Loading" ]]; then
        bad "greenboost module is STUCK in MODULE_STATE_LOADING"
        echo -e "  ${C_DIM}Fix:  sudo bash ${SCRIPT_DIR}/fix_greenboost_unload.sh${C_RESET}"
    else
        ok "greenboost module loaded (state: ${gb_state:-Live})"
    fi
elif lsmod 2>/dev/null | grep -q "^greenboost "; then
    ok "greenboost module loaded"
else
    ok "greenboost module not loaded (not expected - normal if not running Ollama)"
fi

# Check if any module is stuck
stuck_mods=$(awk '$5 == "Unloading" || $5 == "Loading" {print $1 " (" $5 ")"}' /proc/modules 2>/dev/null || true)
if [[ -n "$stuck_mods" ]]; then
    bad "Stuck kernel module(s): $stuck_mods"
    echo -e "  ${C_DIM}Stuck modules block reboot - sudo reboot will hang.${C_RESET}"
    echo -e "  ${C_DIM}Safe reboot via SysRq if needed:${C_RESET}"
    echo -e "  ${C_DIM}  echo s > /proc/sysrq-trigger && echo u > /proc/sysrq-trigger && echo b > /proc/sysrq-trigger${C_RESET}"
fi

# ── A2: NVIDIA reboot-hang fix ────────────────────────────────────────────────
step "A2" "NVIDIA shutdown fix (prevents reboot hang)"

nvidia_conf="/etc/modprobe.d/99-nvidia-greenboost.conf"
if [[ -f "$nvidia_conf" ]] && grep -q "NVreg_PreserveVideoMemoryAllocations=0" "$nvidia_conf"; then
    ok "NVIDIA shutdown fix in place ($nvidia_conf)"
else
    bad "NVIDIA shutdown fix MISSING - reboot will hang when GPU memory is in use"
    echo -e "  ${C_DIM}Applying fix now...${C_RESET}"
    cat > "$nvidia_conf" << 'NVEOF'
# Prevents NVIDIA driver from hanging during reboot/shutdown.
# Default (1) causes the driver to save GPU state to RAM - with large models
# this takes 30+ seconds and can hang indefinitely on kernel 6.19+.
options nvidia NVreg_PreserveVideoMemoryAllocations=0
NVEOF
    fixed "NVIDIA fix written to $nvidia_conf (takes effect after next modprobe nvidia)"
fi

# ── A3: AppArmor integrity ────────────────────────────────────────────────────
step "A3" "AppArmor profile integrity"

aa_abstraction="/etc/apparmor.d/abstractions/greenboost-audit"
aa_base="/etc/apparmor.d/abstractions/base"

# Check abstraction file is installed and valid
if [[ -f "$aa_abstraction" ]]; then
    ok "greenboost-audit abstraction installed"
else
    warn "greenboost-audit abstraction not installed (may be uninstalled - OK if GreenBoost removed)"
fi

# Check for DENIED events - any denial hitting kmod/modprobe/udev is a blocker
if journalctl -k --since "24h ago" >/dev/null 2>&1; then
    aa_denials=$(journalctl -k --since "24h ago" 2>/dev/null \
        | grep -iE 'apparmor="DENIED"' \
        | grep -iE 'kmod|modprobe|udev|systemd|insmod|rmmod' \
        | tail -5 || true)
else
    aa_denials=$(dmesg 2>/dev/null \
        | grep -iE 'apparmor.*DENIED' \
        | grep -iE 'kmod|modprobe|udev|systemd|insmod|rmmod' \
        | tail -5 || true)
fi
if [[ -n "$aa_denials" ]]; then
    bad "AppArmor DENIED events on kernel-module or udev tools (last 24h):"
    echo "$aa_denials" | sed 's/^/    /'
    echo ""
    echo -e "  ${C_DIM}This can silently break modprobe, rmmod, and udev hotplug.${C_RESET}"
    echo -e "  ${C_DIM}To see all denials:  journalctl -k | grep apparmor${C_RESET}"
else
    ok "No AppArmor DENIED events on kmod/udev tools (last 24h)"
fi

# Check for any general AppArmor parse errors in the journal
if journalctl -u apparmor --since "48h ago" >/dev/null 2>&1; then
    aa_errors=$(journalctl -u apparmor --since "48h ago" 2>/dev/null \
        | grep -iE 'error|fail|invalid|parse' | tail -5 || true)
else
    aa_errors=$(dmesg 2>/dev/null \
        | grep -iE 'apparmor.*(error|fail|invalid|parse)' | tail -5 || true)
fi
if [[ -n "$aa_errors" ]]; then
    bad "AppArmor service errors (last 48h):"
    echo "$aa_errors" | sed 's/^/    /'
    echo -e "  ${C_DIM}Profile parse errors cause all affected profiles to be skipped,${C_RESET}"
    echo -e "  ${C_DIM}which may leave processes unconfined or blocked.${C_RESET}"
    echo -e "  ${C_DIM}Try:  sudo apparmor_parser --skip-bad-cache -r /etc/apparmor.d/${C_RESET}"
else
    ok "No AppArmor parse errors (last 48h)"
fi

# Validate that the greenboost injection in abstractions/base is syntactically safe
if [[ -f "$aa_base" ]] && grep -q "greenboost-audit" "$aa_base"; then
    ok "greenboost-audit injected in abstractions/base"
    # Check it has a matching abstraction file (dangling include = parse error)
    if [[ ! -f "$aa_abstraction" ]]; then
        bad "DANGLING include: abstractions/base includes greenboost-audit but the file is MISSING"
        echo -e "  ${C_DIM}This breaks ALL AppArmor profiles - every profile fails to parse.${C_RESET}"
        echo -e "  ${C_DIM}Fix:  sudo ./greenboost_setup.sh install-sys-configs${C_RESET}"
        echo -e "  ${C_DIM}  OR:  sudo sed -i '/greenboost-audit/d' $aa_base${C_RESET}"
    fi
elif [[ -f "$aa_base" ]]; then
    ok "abstractions/base: no greenboost injection (not installed or already removed)"
fi

# ── A4: udisks2 (Disks app backend) ──────────────────────────────────────────
step "A4" "udisks2 - Disks app and auto-mount backend"

if systemctl is-active --quiet udisks2 2>/dev/null; then
    ok "udisks2 is running"
else
    bad "udisks2 is NOT running - Disks app and auto-mount are broken"
    systemctl start udisks2 2>/dev/null \
        && fixed "udisks2 started" \
        || warn "Could not start udisks2 - install: sudo apt-get install -y udisks2"
fi

# ═══════════════════════════════════════════════════════════════
# PART B - USB FIX
# ═══════════════════════════════════════════════════════════════

step "B1" "USB baseline"

echo ""
echo -e "  ${C_BOLD}lsusb:${C_RESET}"
USB_BEFORE=$(lsusb 2>/dev/null || true)
echo "${USB_BEFORE:-  (nothing)}" | sed 's/^/    /'

echo ""
echo -e "  ${C_BOLD}Block devices (/dev/sd*):${C_RESET}"
SD_BEFORE=$(ls /dev/sd* 2>/dev/null | tr '\n' ' ' || true)
echo "${SD_BEFORE:-  (none)}" | sed 's/^/    /'

echo ""
echo -e "  ${C_BOLD}usb_storage module state:${C_RESET}"
# Check sysfs directly - lsmod -2/-2 means module is present in kernel's list
# but its sysfs files can't be read (stuck MODULE_STATE_LOADING)
if [[ -d /sys/module/usb_storage ]]; then
    usb_stor_state=$(cat /sys/module/usb_storage/initstate 2>/dev/null || echo "unknown")
    echo -e "    /sys/module/usb_storage exists - state: ${C_BOLD}${usb_stor_state}${C_RESET}"
    if [[ "$usb_stor_state" == "live" ]]; then
        ok "usb_storage is live - driver should bind; problem is elsewhere"
    else
        bad "usb_storage initstate='${usb_stor_state}' - module stuck, modprobe will hang"
        echo -e "  ${C_DIM}Bypassing modprobe; will use direct sysfs driver bind below.${C_RESET}"
    fi
else
    warn "usb_storage not in /sys/module - not loaded at all; will load via insmod"
fi

# ── B2: Load modules safely (never modprobe a stuck module) ──────────────────
step "B2" "Loading USB storage modules safely"

# xhci_hcd: safe to modprobe (it IS loaded and live)
modprobe xhci_hcd 2>/dev/null && ok "xhci_hcd confirmed" || true

# usb_storage: ONLY load if NOT already in /sys/module
if [[ ! -d /sys/module/usb_storage ]]; then
    # Not in kernel at all - safe to insmod directly to bypass module-lock
    stor_ko=$(find /lib/modules/"$(uname -r)" -name 'usb-storage.ko*' 2>/dev/null | head -1)
    if [[ -n "$stor_ko" ]]; then
        insmod "$stor_ko" 2>/dev/null && fixed "usb_storage loaded via insmod ($stor_ko)" \
            || warn "insmod usb-storage.ko failed - will try sysfs bind anyway"
    else
        warn "usb-storage.ko not found in /lib/modules - already built-in or missing"
    fi
else
    # Module already present in kernel (possibly stuck) - do NOT call modprobe
    ok "usb_storage already in kernel - skipping modprobe to avoid hang"
fi

# uas: same guard
if [[ ! -d /sys/module/uas ]]; then
    uas_ko=$(find /lib/modules/"$(uname -r)" -name 'uas.ko*' 2>/dev/null | head -1)
    [[ -n "$uas_ko" ]] && insmod "$uas_ko" 2>/dev/null && ok "uas loaded" || true
else
    ok "uas already in kernel"
fi

# usbhid: safe - already live
modprobe usbhid 2>/dev/null || true

# ── B3: Disable autosuspend ───────────────────────────────────────────────────
step "B3" "Disabling USB autosuspend"

if [[ -f /sys/module/usbcore/parameters/autosuspend ]]; then
    echo -1 > /sys/module/usbcore/parameters/autosuspend
    ok "usbcore autosuspend → -1 (disabled globally)"
fi
for dev_path in /sys/bus/usb/devices/*/; do
    [[ -f "${dev_path}power/autosuspend_delay_ms" ]] && echo 0    > "${dev_path}power/autosuspend_delay_ms" 2>/dev/null || true
    [[ -f "${dev_path}power/control"              ]] && echo "on" > "${dev_path}power/control"              2>/dev/null || true
done
ok "All USB devices set to power/control=on"

# ── B4: Direct sysfs driver bind for all USB mass-storage devices ─────────────
step "B4" "Direct sysfs driver bind (bypasses stuck modprobe)"

bound_any=0
for dev_path in /sys/bus/usb/devices/*/; do
    [[ -d "$dev_path" ]] || continue
    vid=$(cat "${dev_path}idVendor"  2>/dev/null || echo "")
    pid=$(cat "${dev_path}idProduct" 2>/dev/null || echo "")
    cls=$(cat "${dev_path}bDeviceClass" 2>/dev/null || echo "")
    devname=$(basename "$dev_path")

    # Class 0x00 = device-class defined per interface, 0x08 = mass storage
    # Also target SanDisk specifically by VID:PID
    is_storage=0
    [[ "$cls" == "00" || "$cls" == "08" ]] && is_storage=1
    [[ "$vid" == "0781" && "$pid" == "55c3" ]] && is_storage=1   # SanDisk 3.2Gen1
    [[ $is_storage -eq 0 ]] && continue

    echo -e "  ${C_BOLD}Found storage device: ${devname} (${vid}:${pid})${C_RESET}"

    # Iterate over interfaces (e.g. 2-5:1.0 for Bus2, Device5, Config1, Interface0)
    for iface_path in "${dev_path}"*/; do
        iface_name=$(basename "$iface_path" 2>/dev/null || true)
        # Interface dirs are named BUSNUM-PORTPATH:CONFIG.IFACE
        [[ "$iface_name" =~ ^[0-9]+-[0-9.]+:[0-9]+\.[0-9]+$ ]] || continue

        # Check if already bound to a driver
        if [[ -e "${iface_path}driver" ]]; then
            current_drv=$(basename "$(readlink -f "${iface_path}driver")" 2>/dev/null || echo "unknown")
            ok "  Interface ${iface_name} already bound to: ${current_drv}"
            [[ "$current_drv" == "usb-storage" || "$current_drv" == "uas" ]] && bound_any=1
            continue
        fi

        echo -e "  ${C_DIM}  Interface ${iface_name} has no driver - attempting bind...${C_RESET}"

        # Try uas first (faster for USB 3.x)
        if echo "$iface_name" > /sys/bus/usb/drivers/uas/bind 2>/dev/null; then
            fixed "  ${iface_name} bound to uas"
            bound_any=1
        # Fallback: usb-storage
        elif echo "$iface_name" > /sys/bus/usb/drivers/usb-storage/bind 2>/dev/null; then
            fixed "  ${iface_name} bound to usb-storage"
            bound_any=1
        else
            warn "  Could not bind ${iface_name} to uas or usb-storage"
            echo -e "  ${C_DIM}  This usually means the driver module itself is stuck.${C_RESET}"
            echo -e "  ${C_DIM}  Try the SanDisk-specific quirk below, then unplug/replug.${C_RESET}"
        fi
    done
done

if [[ $bound_any -eq 0 ]]; then
    warn "No storage interfaces bound - driver may be stuck in MODULE_STATE_LOADING"
fi

# ── B5: udev rescan ───────────────────────────────────────────────────────────
step "B5" "udev rescan + settle"

udevadm control --reload-rules 2>/dev/null && ok "udev rules reloaded" || true
udevadm trigger --subsystem-match=usb   --action=add 2>/dev/null || true
udevadm trigger --subsystem-match=block --action=add 2>/dev/null || true
udevadm settle --timeout=10 2>/dev/null && ok "udev settled" || warn "udevadm settle timed out"
sleep 2

# Optional: full xhci_hcd rebind
if [[ $FORCE_REBIND -eq 1 ]]; then
    echo ""
    warn "--force-rebind: This will immediately disconnect ALL USB devices - including keyboard/mouse. Only proceed if using SSH or PS/2 keyboard. Continue? [y/N]"
    read -r _ans
    if [[ "${_ans,,}" == "y" ]]; then
        for pci_dev in /sys/bus/pci/drivers/xhci_hcd/????:??:??.?; do
            [[ -e "$pci_dev" ]] || continue
            dev_id=$(basename "$pci_dev")
            echo "$dev_id" > /sys/bus/pci/drivers/xhci_hcd/unbind 2>/dev/null && sleep 0.5 || continue
            if echo "$dev_id" > /sys/bus/pci/drivers/xhci_hcd/bind 2>/dev/null; then
                fixed "Rebound xhci_hcd controller: $dev_id"
            else
                warn "Rebind failed for $dev_id - attempting drivers_probe recovery..."
                echo "$dev_id" > /sys/bus/pci/drivers_probe 2>/dev/null || true
                bad "Could not rebind $dev_id. USB may be dead. Reboot required." || true
                break  # stop unbinding more controllers in this session (F-L7-09)
            fi
        done
        sleep 2; udevadm trigger --subsystem-match=usb --action=add 2>/dev/null || true
        udevadm settle --timeout=10 2>/dev/null || true
    fi
fi

# ═══════════════════════════════════════════════════════════════
# RESULT
# ═══════════════════════════════════════════════════════════════

step "★" "Summary"

sleep 1
SD_AFTER=$(ls /dev/sd* 2>/dev/null | tr '\n' ' ' || true)
new_devs=()
for dev in $(ls /dev/sd? 2>/dev/null); do
    echo "$SD_BEFORE" | grep -qF "$dev" || new_devs+=("$dev")
done

echo ""
if [[ ${#new_devs[@]} -gt 0 ]]; then
    fixed "${#new_devs[@]} new block device(s): ${new_devs[*]}"
    echo ""
    for dev in "${new_devs[@]}"; do
        echo -e "  ${C_BOLD}${dev}:${C_RESET}"
        lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINT "$dev" 2>/dev/null | sed 's/^/    /' || true
    done
    echo ""
    echo -e "  ${C_DIM}Mount with:  udisksctl mount -b /dev/sdX1${C_RESET}"
    echo -e "  ${C_DIM}Disks app should now show the drive.${C_RESET}"
elif [[ -n "$SD_AFTER" ]]; then
    ok "Block devices present ($SD_AFTER)- already there before fix"
else
    bad "Still no block device for USB drive"
    echo ""
    echo -e "  ${C_BOLD}The usb_storage module is stuck in MODULE_STATE_LOADING.${C_RESET}"
    echo -e "  Unplug the SanDisk, run this, then replug:"
    echo ""
    echo -e "    ${C_DIM}# Force usb_storage quirk mode (disables UAS, uses legacy protocol):${C_RESET}"
    echo -e "    echo 'options usb_storage quirks=0781:55c3:u' | sudo tee /etc/modprobe.d/sandisk-no-uas.conf"
    echo -e "    ${C_DIM}# Then reboot - the stuck module will clear and the quirk loads fresh:${C_RESET}"
    echo -e "    sudo bash fix_usb.sh  # after reboot"
fi

echo ""
echo "────────────────────────────────────────────────────────────────"
echo -e "  Issues found: ${ISSUES}   Fixed this run: ${FIXED}"
echo "────────────────────────────────────────────────────────────────"

# Reboot guidance if stuck modules found
if [[ -n "$stuck_mods" ]]; then
    echo ""
    echo -e "  ${C_YELLOW}${C_BOLD}Reboot warning:${C_RESET} stuck module(s) will cause reboot to hang."
    echo -e "  Use SysRq for a safe reboot that syncs filesystems first:"
    echo -e "    ${C_DIM}echo s > /proc/sysrq-trigger   # sync${C_RESET}"
    echo -e "    ${C_DIM}echo u > /proc/sysrq-trigger   # remount ro${C_RESET}"
    echo -e "    ${C_DIM}echo b > /proc/sysrq-trigger   # reboot${C_RESET}"
fi
echo ""

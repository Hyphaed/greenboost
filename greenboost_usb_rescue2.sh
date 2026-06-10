#!/usr/bin/env bash
# greenboost_usb_rescue.sh - run from Ubuntu Live USB to completely remove GreenBoost
# from a LUKS-encrypted Ubuntu installation.
#
# Usage (from Ubuntu Live USB terminal):
#   sudo bash greenboost_usb_rescue.sh
#
# Requirements: cryptsetup, lvm2, chroot (standard on Ubuntu Live)

set -euo pipefail

C_RED='\033[0;31m'; C_GREEN='\033[0;32m'; C_YELLOW='\033[1;33m'
C_CYAN='\033[0;36m'; C_BOLD='\033[1m'; C_DIM='\033[2m'; C_RESET='\033[0m'
C_VIOLET='\033[0;35m'

MOUNT_ROOT="/mnt/rescue"
LUKS_NAME="greenboost_rescue"
LUKS_DEV=""
ROOT_DEV=""

info()    { echo -e "${C_CYAN}  ▸ $*${C_RESET}"; }
success() { echo -e "${C_GREEN}  ✓ $*${C_RESET}"; }
warn()    { echo -e "${C_YELLOW}  ⚠ $*${C_RESET}"; }
error()   { echo -e "${C_RED}  ✗ $*${C_RESET}" >&2; }
header()  { echo -e "\n${C_VIOLET}${C_BOLD}══ $* ══${C_RESET}"; }
die()     { error "$*"; cleanup_and_exit 1; }

# Ensure /dev/fd and /dev/pts are available for scripts using process substitution or sudo PTY
ensure_dev_bindpoints() {
    # If running on Live USB (outside chroot), /dev already ok. If running inside chroot, bind-mounts may be needed.
    if [[ ! -e /dev/fd/0 ]]; then
        mkdir -p /dev/fd
        mount --bind /proc/self/fd /dev/fd 2>/dev/null || true
    fi
    if ! mountpoint -q /dev/pts 2>/dev/null; then
        mkdir -p /dev/pts
        mount -t devpts devpts /dev/pts -o gid=5,mode=620 2>/dev/null || true
    fi
}

# ── dependency check ──────────────────────────────────────────────────────────
check_deps() {
    local missing=()
    for cmd in cryptsetup lsblk blkid mount chroot depmod sed find; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    if (( ${#missing[@]} > 0 )); then
        error "Missing tools: ${missing[*]}"
        echo "Install with: sudo apt-get install -y cryptsetup lvm2"
        exit 1
    fi
}

# ── safety: warn if running on installed system ───────────────────────────────
check_live_environment() {
    # Heuristic: live systems have 'casper' or 'live' in /proc/cmdline
    if ! grep -qiE 'casper|live|toram' /proc/cmdline 2>/dev/null; then
        warn "This script is designed for a Ubuntu Live USB environment."
        warn "Running on an installed system may cause issues."
        echo -ne "${C_YELLOW}  Continue anyway? [y/N] ${C_RESET}"
        read -r ans
        [[ "${ans,,}" == "y" ]] || { echo "Aborted."; exit 0; }
    fi
}

# ── detect LUKS partitions ────────────────────────────────────────────────────
detect_luks_partitions() {
    header "Detecting LUKS partitions"
    echo ""
    lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT 2>/dev/null || true
    echo ""

    mapfile -t LUKS_PARTS < <(blkid -t TYPE=crypto_LUKS -o device 2>/dev/null | sort)

    if (( ${#LUKS_PARTS[@]} == 0 )); then
        die "No LUKS partitions found. Is the encrypted disk connected?"
    fi

    if (( ${#LUKS_PARTS[@]} == 1 )); then
        LUKS_DEV="${LUKS_PARTS[0]}"
        info "Found LUKS partition: ${C_BOLD}${LUKS_DEV}${C_RESET}"
        echo -ne "${C_CYAN}  Use this partition? [Y/n] ${C_RESET}"
        read -r ans
        [[ "${ans,,}" != "n" ]] || select_luks_partition
    else
        select_luks_partition
    fi
}

select_luks_partition() {
    echo -e "\n${C_BOLD}  Available LUKS partitions:${C_RESET}"
    local i=1
    for p in "${LUKS_PARTS[@]}"; do
        local size
        size=$(lsblk -dno SIZE "$p" 2>/dev/null || echo "?")
        echo "    ${C_BOLD}${i})${C_RESET} ${p}  (${size})"
        (( i++ ))
    done
    echo ""
    echo -ne "${C_CYAN}  Enter number [1-${#LUKS_PARTS[@]}]: ${C_RESET}"
    read -r choice
    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#LUKS_PARTS[@]} )); then
        LUKS_DEV="${LUKS_PARTS[$((choice-1))]}"
    else
        die "Invalid selection."
    fi
}

# ── open LUKS container ───────────────────────────────────────────────────────
open_luks() {
    header "Opening LUKS container"
    cryptsetup close "$LUKS_NAME" 2>/dev/null || true

    info "Opening ${LUKS_DEV} as /dev/mapper/${LUKS_NAME}"
    echo "  (Enter your disk encryption passphrase below)"
    if ! cryptsetup luksOpen "$LUKS_DEV" "$LUKS_NAME"; then
        die "Failed to open LUKS container. Wrong passphrase?"
    fi
    success "LUKS container opened: /dev/mapper/${LUKS_NAME}"
}

# ── activate LVM if present ───────────────────────────────────────────────────
activate_lvm() {
    if ! command -v vgscan &>/dev/null; then
        warn "lvm2 not installed - skipping LVM activation. Install: sudo apt-get install -y lvm2"
        return
    fi

    local vg_count
    vg_count=$(vgscan --mknodes 2>/dev/null | grep -c 'Found volume group' || echo 0)
    if (( vg_count > 0 )); then
        info "Found ${vg_count} LVM volume group(s) - activating"
        # Determine the VG that lives on this specific LUKS PV to avoid
        # activating unrelated VGs from other disks (F-L7-02).
        target_vg=$(pvs --noheadings -o vg_name "/dev/mapper/${LUKS_NAME}" 2>/dev/null | tr -d ' ' | head -1)
        if [[ -z "$target_vg" ]]; then
            bad "Could not determine VG name for /dev/mapper/${LUKS_NAME}"
        fi
        vgchange -ay "$target_vg" 2>/dev/null || true
        sleep 1
        success "LVM volumes activated (VG: ${target_vg})"
    else
        info "No LVM inside LUKS (plain ext4 or btrfs)"
    fi
}

# ── find root filesystem ──────────────────────────────────────────────────────
find_root_volume() {
    header "Finding root filesystem"

    local candidates=()

    if command -v lvs &>/dev/null; then
        mapfile -t lv_devs < <(lvs --noheadings -o lv_path 2>/dev/null | tr -d ' ')
        for dev in "${lv_devs[@]}"; do
            [[ -b "$dev" ]] && candidates+=("$dev")
        done
    fi

    [[ -b "/dev/mapper/${LUKS_NAME}" ]] && candidates+=("/dev/mapper/${LUKS_NAME}")

    while IFS= read -r dev; do
        [[ -b "$dev" ]] && candidates+=("$dev")
    done < <(ls /dev/mapper/ 2>/dev/null | grep -v "^${LUKS_NAME}$" | grep -v control | sed 's|^|/dev/mapper/|')

    local tmp_mnt
    tmp_mnt=$(mktemp -d)
    trap 'umount "$tmp_mnt" 2>/dev/null || true; rmdir "$tmp_mnt" 2>/dev/null || true' RETURN
    for dev in "${candidates[@]}"; do
        if mount -o ro "$dev" "$tmp_mnt" 2>/dev/null; then
            if [[ -f "$tmp_mnt/etc/os-release" ]] && [[ -d "$tmp_mnt/etc/systemd" ]]; then
                ROOT_DEV="$dev"
                umount "$tmp_mnt"
                rmdir "$tmp_mnt"
                success "Root volume: ${ROOT_DEV}"
                return
            fi
            umount "$tmp_mnt" 2>/dev/null || true
        fi
    done
    rmdir "$tmp_mnt" 2>/dev/null || true

    echo ""
    warn "Could not auto-detect root volume."
    echo -e "  ${C_DIM}Available block devices:${C_RESET}"
    lsblk -o NAME,SIZE,TYPE,FSTYPE | grep -v loop || true
    echo ""
    echo -ne "${C_CYAN}  Enter root device path (e.g. /dev/mapper/ubuntu--vg-ubuntu--lv): ${C_RESET}"
    read -r ROOT_DEV
    [[ -b "$ROOT_DEV" ]] || die "Device not found: ${ROOT_DEV}"
}

# ── mount system for chroot ───────────────────────────────────────────────────
mount_system() {
    header "Mounting system"
    mkdir -p "$MOUNT_ROOT"

    if ! mount "$ROOT_DEV" "$MOUNT_ROOT"; then
        die "Failed to mount root filesystem: ${ROOT_DEV}"
    fi
    success "Root mounted at ${MOUNT_ROOT}"

    local boot_dev boot_spec
    boot_spec=$(awk '$2 == "/boot" {print $1; exit}' "$MOUNT_ROOT/etc/fstab" 2>/dev/null || true)
    case "$boot_spec" in
        UUID=*)
            boot_dev=$(blkid -t "${boot_spec}" -o device 2>/dev/null) ;;
        LABEL=*)
            boot_dev=$(blkid -L "${boot_spec#LABEL=}" 2>/dev/null) ;;
        /dev/*)
            boot_dev="$boot_spec" ;;
        *)
            boot_dev="" ;;
    esac
    if [[ -n "$boot_dev" && -b "$boot_dev" ]]; then
        mkdir -p "$MOUNT_ROOT/boot"
        mount "$boot_dev" "$MOUNT_ROOT/boot" 2>/dev/null || warn "Could not mount /boot ($boot_dev)"
    fi

    local efi_dev
    efi_dev=$(grep -E 'vfat|EFI|efi' "$MOUNT_ROOT/etc/fstab" 2>/dev/null | grep '/boot/efi' | awk '{print $1}' || true)
    if [[ -z "$efi_dev" ]]; then
        local parent_disk
        parent_disk=$(lsblk -no PKNAME "$LUKS_DEV" 2>/dev/null | head -1)
        if [[ -n "$parent_disk" ]]; then
            efi_dev=$(blkid -t TYPE=vfat "/dev/${parent_disk}"* 2>/dev/null | grep -iE 'EFI|esp' | awk -F: '{print $1}' | head -1 || true)
            if [[ -z "$efi_dev" ]]; then
                efi_dev=$(fdisk -l "/dev/${parent_disk}" 2>/dev/null | grep -i 'EFI' | awk '{print $1}' | head -1 || true)
            fi
        fi
    fi
    if [[ -n "$efi_dev" && -b "$efi_dev" ]]; then
        mkdir -p "$MOUNT_ROOT/boot/efi"
        mount "$efi_dev" "$MOUNT_ROOT/boot/efi" 2>/dev/null && success "EFI partition mounted: ${efi_dev}" || warn "Could not mount EFI partition ($efi_dev) - GRUB update may fail"
    else
        warn "EFI partition not found - GRUB update may fail (GRUB params still cleaned from /etc/default/grub)"
    fi

    # Bind mounts for chroot (ensure /dev/fd and /dev/pts too)
    for dir in proc sys dev dev/pts run; do
        mkdir -p "$MOUNT_ROOT/$dir"
        mount --bind "/$dir" "$MOUNT_ROOT/$dir" 2>/dev/null || warn "Could not bind-mount /$dir"
    done

    # Also try to provide /dev/fd in the chroot
    mkdir -p "$MOUNT_ROOT/dev/fd"
    mount --bind /proc/self/fd "$MOUNT_ROOT/dev/fd" 2>/dev/null || true

    if [[ -d /sys/firmware/efi/efivars ]]; then
        mkdir -p "$MOUNT_ROOT/sys/firmware/efi/efivars"
        mount --bind /sys/firmware/efi/efivars "$MOUNT_ROOT/sys/firmware/efi/efivars" 2>/dev/null || true
    fi

    success "Bind mounts ready"
}

# ── run cleanup inside chroot ─────────────────────────────────────────────────
run_cleanup() {
    header "Removing GreenBoost (chroot)"
    info "Running cleanup inside chroot..."

    chroot "$MOUNT_ROOT" /bin/bash << 'CHROOT_EOF'
set -euo pipefail

echo "  [1/10] Disabling GreenBoost services..."
for svc in greenboost-recovery greenboost-sentinel greenboost-vram-watchdog \
           greenboost-boot-cleanup cpu-perf greenboost-turboquant \
           greenboost-idle-reclaim greenboost-netd greenboost-shader-boost; do
    systemctl disable "${svc}.service" 2>/dev/null || true
done
rm -f /etc/systemd/system/greenboost*.service
rm -f /etc/systemd/system/cpu-perf.service
rm -f /etc/systemd/system/sysinit.target.wants/greenboost-boot-cleanup.service
rm -f /etc/systemd/system/rescue.target.wants/greenboost-boot-cleanup.service
rm -f /etc/systemd/system/emergency.target.wants/greenboost-boot-cleanup.service
rm -f /lib/systemd/system-shutdown/greenboost-stuck.sh
echo "     Done."

echo "  [2/10] Cleaning Ollama configuration..."
rm -f /etc/systemd/system/ollama.service.d/99-greenboost.conf
rmdir /etc/systemd/system/ollama.service.d/ 2>/dev/null || true
if [[ -f /etc/systemd/system/ollama.service ]]; then
    sed -i '/GREENBOOST\|libgreenboost/d' /etc/systemd/system/ollama.service
fi
echo "     Done."

echo "  [3/10] Removing kernel module..."
find /lib/modules -name "greenboost.ko*" -delete 2>/dev/null || true
rm -f /etc/modules-load.d/greenboost.conf
rm -f /etc/modprobe.d/greenboost.conf
rm -f /etc/modprobe.d/99-greenboost-blacklist.conf
target_kver=$(ls /lib/modules/ 2>/dev/null | sort -V | tail -1)
if [[ -n "$target_kver" ]]; then
    depmod -a "$target_kver" 2>/dev/null || true
fi
echo "     Done."

echo "  [4/10] Removing CUDA shim and libraries..."
rm -f /usr/local/lib/libgreenboost_cuda.so
rm -f /usr/local/lib/libgreenboost_tq.so
rm -f /usr/local/lib/libgreenboost_audit.so
rm -f /usr/local/sbin/greenboost-recover
rm -f /usr/local/sbin/greenboost-vram-watchdog
rm -f /usr/local/sbin/greenboostd
rm -f /usr/local/bin/greenboost
rm -f /usr/local/bin/greenboost-turboquant
rm -f /usr/local/bin/greenboost-idle-reclaim
rm -f /usr/local/bin/greenboost-run
rm -f /usr/local/bin/greenboost-run-tgi
rm -f /usr/local/bin/greenboost-run-unsloth
rm -f /usr/local/bin/greenboost-run-vllm
rm -rf /usr/local/lib/greenboost
rm -rf /usr/local/share/greenboost
rm -rf /var/lib/greenboost
echo "     Done."

echo "  [5/10] Removing system configuration files..."
rm -f /etc/udev/rules.d/99-greenboost.rules
rm -f /etc/udev/rules.d/99-nvme-greenboost.rules
rm -f /etc/sysctl.d/99-greenboost.conf
rm -f /etc/sysctl.d/99-zzz-greenboost.conf
rm -f /etc/sysfs.d/greenboost-hugepages.conf
rm -f /etc/greenboost 2>/dev/null || true
rm -rf /etc/greenboost/ 2>/dev/null || true
echo "     Done."

echo "  [6/10] Cleaning ld.so.preload..."
if [[ -f /etc/ld.so.preload ]]; then
    sed -i '/libgreenboost/d' /etc/ld.so.preload
    [[ -s /etc/ld.so.preload ]] || rm -f /etc/ld.so.preload
fi
echo "     Done."

echo "  [7/10] Removing DKMS state..."
rm -rf /var/lib/dkms/greenboost 2>/dev/null || true
rm -rf /usr/src/greenboost-* 2>/dev/null || true
echo "     Done."

echo "  [8/10] Cleaning AppArmor..."
rm -f /etc/apparmor.d/abstractions/greenboost-audit 2>/dev/null || true
for profile in /etc/apparmor.d/*; do
    [[ -f "$profile" ]] && sed -i '/greenboost/d' "$profile" 2>/dev/null || true
done
echo "     Done."

echo "  [9/10] Cleaning GRUB parameters..."
if [[ -f /etc/default/grub ]]; then
    sed -i \
        -e 's/ transparent_hugepage=always//g' \
        -e 's/ skew_tick=1//g' \
        -e 's/ rcu_nocbs=[^ "]*//g' \
        -e 's/ nohz_full=[^ "]*//g' \
        -e 's/ numa_balancing=disable//g' \
        -e 's/ workqueue\.power_efficient=0//g' \
        /etc/default/grub
fi
echo "     Done."

echo "  [10/10] Applying NVIDIA shutdown fix..."
cat > /etc/modprobe.d/nvidia-shutdown-fix.conf << 'NVEOF'
# Prevents NVIDIA driver from hanging during reboot/shutdown.
options nvidia NVreg_PreserveVideoMemoryAllocations=0
NVEOF
echo "     Done."

echo ""
echo "  GreenBoost components removed. Running final systemd reload..."
systemctl daemon-reload 2>/dev/null || true
CHROOT_EOF

    success "Chroot cleanup complete"
}

# ── update GRUB ───────────────────────────────────────────────────────────────
update_grub() {
    header "Updating GRUB"
    info "Running update-grub inside chroot..."

    if chroot "$MOUNT_ROOT" which update-grub &>/dev/null; then
        if chroot "$MOUNT_ROOT" update-grub 2>&1 | sed 's/^/  /'; then
            :  # update-grub succeeded
        else
            warn "update-grub FAILED - verify /boot is mounted before rebooting"
        fi
    elif chroot "$MOUNT_ROOT" which grub2-mkconfig &>/dev/null; then
        if chroot "$MOUNT_ROOT" grub2-mkconfig -o /boot/grub2/grub.cfg 2>&1 | sed 's/^/  /'; then
            :  # grub2-mkconfig succeeded
        else
            warn "grub2-mkconfig FAILED - verify /boot is mounted before rebooting"
        fi
    else
        warn "Neither update-grub nor grub2-mkconfig found in chroot - skipping GRUB regeneration"
        warn "GRUB parameters were already cleaned from /etc/default/grub; they take effect on next update-grub run."
    fi

    success "GRUB update done"
}

# ── unmount and close ─────────────────────────────────────────────────────────
unmount_system() {
    header "Unmounting"
    info "Unmounting bind mounts and root..."

    for dir in sys/firmware/efi/efivars dev/fd dev/pts dev run sys proc boot/efi boot ""; do
        local target="$MOUNT_ROOT/$dir"
        if mountpoint -q "$target" 2>/dev/null; then
            umount "$target" 2>/dev/null || umount -l "$target" 2>/dev/null || true
        fi
    done

    if command -v vgchange &>/dev/null; then
        if [[ -n "${target_vg:-}" ]]; then
            vgchange -an "$target_vg" 2>/dev/null || true
        else
            vgchange -an 2>/dev/null || true
        fi
    fi

    if [[ -b "/dev/mapper/${LUKS_NAME}" ]]; then
        cryptsetup close "$LUKS_NAME" 2>/dev/null && success "LUKS container closed" || warn "Could not close LUKS (may still be busy)"
    fi

    success "Unmount complete"
}

cleanup_and_exit() {
    local code="${1:-1}"
    echo ""
    warn "Cleaning up..."
    unmount_system 2>/dev/null || true
    exit "$code"
}

trap 'cleanup_and_exit 1' ERR INT TERM

# ── main ──────────────────────────────────────────────────────────────────────
main() {
    echo ""
    echo -e "${C_VIOLET}${C_BOLD}╔══════════════════════════════════════════════════════╗${C_RESET}"
    echo -e "${C_VIOLET}${C_BOLD}║         GreenBoost USB Rescue - LUKS Edition         ║${C_RESET}"
    echo -e "${C_VIOLET}${C_BOLD}╚══════════════════════════════════════════════════════╝${C_RESET}"
    echo ""
    echo -e "  This script completely removes GreenBoost from your"
    echo -e "  LUKS-encrypted Ubuntu installation."
    echo -e "  ${C_DIM}Run from an Ubuntu Live USB. Requires root.${C_RESET}"
    echo ""

    if [[ $EUID -ne 0 ]]; then
        error "Must run as root: sudo bash $0"
        exit 1
    fi

    ensure_dev_bindpoints
    check_deps
    check_live_environment
    detect_luks_partitions
    open_luks
    activate_lvm
    find_root_volume
    mount_system
    read -r -p "  → Proceed with cleanup on $ROOT_DEV? [y/N] " _confirm
    [[ "${_confirm,,}" == "y" ]] || { warn "Aborted by user."; exit 1; }
    run_cleanup
    update_grub
    unmount_system

    echo ""
    echo -e "${C_GREEN}${C_BOLD}══════════════════════════════════════════════════════${C_RESET}"
    echo -e "${C_GREEN}${C_BOLD}  GreenBoost fully removed.${C_RESET}"
    echo ""
    echo -e "  Next steps:"
    echo -e "  ${C_BOLD}1.${C_RESET} Remove the Live USB and reboot the system."
    echo -e "  ${C_BOLD}2.${C_RESET} The system should boot cleanly without GreenBoost."
    echo -e "  ${C_BOLD}3.${C_RESET} Ollama will start normally using real GPU VRAM."
    echo -e "  ${C_BOLD}4.${C_RESET} To reinstall GreenBoost: ${C_DIM}sudo ./greenboost_setup.sh full-install${C_RESET}"
    echo ""
    echo -e "  ${C_DIM}NVIDIA shutdown fix applied: reboots should no longer hang.${C_RESET}"
    echo -e "${C_GREEN}${C_BOLD}══════════════════════════════════════════════════════${C_RESET}"
    echo ""
}
main "$@"



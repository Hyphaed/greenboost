#!/usr/bin/env bash
# greenboost_boot_guard.sh - last-resort boot-time self-heal for the greenboost
# kernel module.
#
# Runs as greenboost-boot-guard.service, ordered BEFORE systemd-modules-load.service
# (which is what actually loads greenboost via /etc/modules-load.d/greenboost.conf).
# modules-load.d only knows how to *load* an already-built .ko; if the running
# kernel doesn't have one yet (DKMS's own autoinstall hook didn't run, failed
# silently, or the module was never rebuilt after a kernel upgrade), the load
# silently no-ops and greenboost just stays absent for that boot. This script
# closes that gap: if no .ko exists for `uname -r`, rebuild one via DKMS first.
#
# Must NEVER fail/block boot , every exit path is 0, all output goes to a log
# file plus the journal (via logger), never to a boot console.

set -uo pipefail   # no -e: every step is best-effort

KVER="$(uname -r)"
LOG_FILE=/var/log/greenboost-boot-guard.log
TAG="greenboost-boot-guard"

log() {
    printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG_FILE" 2>/dev/null
    command -v logger &>/dev/null && logger -t "$TAG" "$*"
}

if lsmod 2>/dev/null | grep -q '^greenboost\b'; then
    log "greenboost already loaded for $KVER , nothing to do"
    exit 0
fi

KO="$(find "/lib/modules/${KVER}" -name 'greenboost.ko*' 2>/dev/null | head -1)"
if [[ -z "$KO" ]]; then
    log "no greenboost.ko present for $KVER , attempting DKMS self-heal build"
    GB_VER="$(dkms status greenboost 2>/dev/null | head -1 | grep -oP 'greenboost/\K[0-9.]+')"
    if [[ -n "$GB_VER" ]] && command -v dkms &>/dev/null; then
        if dkms install "greenboost/${GB_VER}" -k "$KVER" --force >> "$LOG_FILE" 2>&1; then
            log "DKMS self-heal build succeeded for $KVER (version $GB_VER)"
        else
            log "DKMS self-heal build FAILED for $KVER , see $LOG_FILE"
            exit 0
        fi
    else
        log "greenboost not registered in DKMS , nothing to self-heal from, skipping"
        exit 0
    fi
fi

depmod -a "$KVER" 2>/dev/null || true
if modprobe greenboost >> "$LOG_FILE" 2>&1; then
    log "greenboost loaded for $KVER"
else
    log "modprobe greenboost FAILED for $KVER , see $LOG_FILE and dmesg"
fi

exit 0

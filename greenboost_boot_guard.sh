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
# It also closes a second, nastier gap , IDENTITY, not just presence.
#
#   Incident 2026-08-21: the box booted GreenBoost v3.2 for days while v3.4 was
#   the installed build. `initramfs.conf` carries MODULES=most, so initramfs-tools
#   copies updates/dkms/greenboost.ko INTO the initrd, and the initrd's own
#   systemd-modules-load inserts it before the real root is even mounted. A
#   reinstall rebuilds /lib/modules/... and never touches the initrd, so every
#   subsequent boot loaded the frozen copy. Rebooting was the one action
#   guaranteed not to fix it. Two consequences: the running module was two
#   versions stale, and because /var is not mounted that early, T3 failed open
#   with "T3 backing file unavailable (...): -2 - T3 disabled" on every boot.
#
#   This guard used to check `lsmod | grep greenboost` and exit 0 on a hit. A
#   wrong-version module IS loaded, so the guard that exists precisely for this
#   reported success. Presence is not identity. It now compares the loaded
#   module's srcversion against the on-disk .ko's and reloads on a mismatch ,
#   which, running after local-fs.target, also gets T3 its backing file.
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

# One dataflux event, best-effort, never fails the caller. Mirrors
# gb_dataflux.py's per-call GREENBOOST_DATAFLUX_LOG resolution; the installer
# bakes the owner's path into the unit so a root-run guard lands in the log the
# MCP actually reads instead of /root's.
gb_dataflux_emit() {
    local _obj="$1" _log
    _log="${GREENBOOST_DATAFLUX_LOG:-${HOME:-/root}/.local/share/greenboost/dataflux.jsonl}"
    mkdir -p "$(dirname "$_log")" 2>/dev/null || return 0
    printf '%s\n' "$_obj" >> "$_log" 2>/dev/null || true
    return 0
}

emit_drift() {
    # emit_drift <action> <loaded_ver> <ondisk_ver> <loaded_src> <ondisk_src> <refcnt> <detail>
    gb_dataflux_emit "$(printf '{"kind":"kmod_version_drift","ts":"%s","action":"%s","loaded_version":"%s","ondisk_version":"%s","loaded_srcversion":"%s","ondisk_srcversion":"%s","refcnt":"%s","kver":"%s","detail":"%s","source":"greenboost_boot_guard.sh"}' \
        "$(date -Is)" "$1" "$2" "$3" "$4" "$5" "$6" "$KVER" "$7")"
}

loaded_field() { cat "/sys/module/greenboost/$1" 2>/dev/null; }
ondisk_field() { modinfo -F "$1" greenboost 2>/dev/null | head -1; }

# Reload an already-loaded module whose identity does not match the installed
# .ko. Refuses to unload while anything holds it , a version number is never
# worth killing a live CUDA process for.
reload_stale_module() {
    local why="$1" lver="$2" over="$3" lsrc="$4" osrc="$5"
    local refcnt
    refcnt="$(loaded_field refcnt)"
    refcnt="${refcnt:-unknown}"

    if [[ "$refcnt" =~ ^[0-9]+$ ]] && (( refcnt > 0 )); then
        log "DRIFT: $why , but the module is IN USE (refcnt=$refcnt); refusing to unload."
        log "       The running module keeps working exactly as before; you are simply not"
        log "       getting anything that changed in v${over}. To pick it up: stop whatever"
        log "       holds /dev/greenboost, then run 'sudo greenboost load', or reboot."
        emit_drift "deferred_in_use" "$lver" "$over" "$lsrc" "$osrc" "$refcnt" "$why"
        return 0
    fi

    log "DRIFT: $why , reloading (refcnt=$refcnt)"
    if modprobe -r greenboost >> "$LOG_FILE" 2>&1 && modprobe greenboost >> "$LOG_FILE" 2>&1; then
        local now_ver now_src
        now_ver="$(loaded_field version)"; now_src="$(loaded_field srcversion)"
        if [[ "$now_src" == "$osrc" || "$now_ver" == "$over" ]]; then
            log "reloaded: now running v${now_ver} (${now_src}) , matches the installed build"
            emit_drift "reloaded" "$lver" "$over" "$lsrc" "$osrc" "$refcnt" "$why"
        else
            log "reloaded but STILL mismatched: v${now_ver} (${now_src}) != installed v${over} (${osrc})"
            log "       something outside /lib/modules/${KVER} is supplying the module , check the initrd:"
            log "       lsinitramfs /boot/initrd.img-${KVER} | grep greenboost"
            emit_drift "reloaded_still_mismatched" "$now_ver" "$over" "$now_src" "$osrc" "$refcnt" "$why"
        fi
    else
        log "reload FAILED , see $LOG_FILE and dmesg. Still running v${lver}."
        emit_drift "reload_failed" "$lver" "$over" "$lsrc" "$osrc" "$refcnt" "$why"
    fi
    return 0
}

# ── already loaded: check IDENTITY, not just presence ───────────────────────
if lsmod 2>/dev/null | grep -q '^greenboost\b'; then
    LOADED_VER="$(loaded_field version)"
    LOADED_SRC="$(loaded_field srcversion)"
    ONDISK_VER="$(ondisk_field version)"
    ONDISK_SRC="$(ondisk_field srcversion)"

    if [[ -z "$ONDISK_VER" && -z "$ONDISK_SRC" ]]; then
        log "greenboost v${LOADED_VER:-unknown} is loaded but no .ko is resolvable for $KVER ,"
        log "       nothing to compare against, leaving the running module alone"
        exit 0
    fi

    # srcversion is the real identity (a content hash of the source). Prefer it;
    # fall back to the version string when either side doesn't publish one.
    if [[ -n "$LOADED_SRC" && -n "$ONDISK_SRC" ]]; then
        if [[ "$LOADED_SRC" == "$ONDISK_SRC" ]]; then
            log "greenboost v${LOADED_VER} is the installed build (${LOADED_SRC}) , nothing to do"
            exit 0
        fi
        reload_stale_module \
            "loaded v${LOADED_VER} (${LOADED_SRC}) != installed v${ONDISK_VER} (${ONDISK_SRC})" \
            "$LOADED_VER" "$ONDISK_VER" "$LOADED_SRC" "$ONDISK_SRC"
        exit 0
    fi

    if [[ -n "$LOADED_VER" && -n "$ONDISK_VER" && "$LOADED_VER" == "$ONDISK_VER" ]]; then
        log "greenboost v${LOADED_VER} matches the installed version (no srcversion to compare) , nothing to do"
        exit 0
    fi
    reload_stale_module \
        "loaded v${LOADED_VER:-unknown} != installed v${ONDISK_VER:-unknown} (no srcversion available)" \
        "${LOADED_VER:-unknown}" "${ONDISK_VER:-unknown}" "${LOADED_SRC:-}" "${ONDISK_SRC:-}"
    exit 0
fi

# ── not loaded at all: the original DKMS self-heal path ─────────────────────
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
    log "greenboost loaded for $KVER (v$(loaded_field version))"
else
    log "modprobe greenboost FAILED for $KVER , see $LOG_FILE and dmesg"
fi

exit 0

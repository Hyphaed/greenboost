# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
#
# lib/gb_ssh.sh - SSH helpers for feeder cluster operations.
#
# Pinned host-key SSH options.  All feeder SSH/SCP callsites should use
# `$(_gb_ssh_opts "$_ip")` instead of hardcoded `-o StrictHostKeyChecking=no`
# so the first contact uses `accept-new` (which writes to $GB_KNOWN_HOSTS)
# and every subsequent contact uses `strict` against the pinned key.
#
# Sourced by greenboost_setup.sh near the top of the script.  See PR-J / PR-GG.

# _gb_ssh_opts <ip> - emit SSH options that prefer strict checking against
# the pinned key file.  If the host has not been pinned yet, fall back to the
# accept-new policy on the SAME file so that subsequent connects upgrade to
# strict automatically.
_gb_ssh_opts() {
    local ip="$1"
    mkdir -p "$(dirname "$GB_KNOWN_HOSTS")" 2>/dev/null || true
    if [[ -f "$GB_KNOWN_HOSTS" ]] && ssh-keygen -F "$ip" -f "$GB_KNOWN_HOSTS" >/dev/null 2>&1; then
        printf -- '-o UserKnownHostsFile=%s -o StrictHostKeyChecking=yes' "$GB_KNOWN_HOSTS"
    else
        printf -- '-o UserKnownHostsFile=%s -o StrictHostKeyChecking=accept-new' "$GB_KNOWN_HOSTS"
    fi
}

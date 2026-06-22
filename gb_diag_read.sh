#!/usr/bin/env bash
# Read-only diagnostic helper for inspecting GreenBoost-related processes.
# Installed to /usr/local/sbin/gb-diag-read by greenboost_setup.sh so that
# an AI assistant (or any unprivileged tool) can read /proc/<pid>/{maps,environ}
# and dmesg for diagnosing GreenBoost shim/kernel-module issues without a full
# interactive sudo session. Intentionally narrow: no write, no process control.
set -euo pipefail

usage() { echo "usage: gb-diag-read dmesg | gb-diag-read <pid> maps|environ" >&2; exit 2; }

[[ $# -ge 1 ]] || usage

if [[ "$1" == "dmesg" ]]; then
    exec dmesg
fi

[[ $# -eq 2 ]] || usage
pid="$1"; what="$2"

[[ "$pid" =~ ^[0-9]+$ ]] || { echo "gb-diag-read: pid must be numeric" >&2; exit 2; }
[[ "$what" == "maps" || "$what" == "environ" ]] || usage
[[ -d "/proc/$pid" ]] || { echo "gb-diag-read: no such pid $pid" >&2; exit 1; }

exec cat "/proc/$pid/$what"

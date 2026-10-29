#!/usr/bin/env bash
# Shared helpers for the installer tests. No sudo, no real /etc, no real /boot:
# every test works inside a fake root created under $GB_TEST_TMP (default: a
# fresh `mktemp -d`). Nothing here deletes anything; the caller owns the sandbox.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETUP="$REPO/greenboost_setup.sh"
PASS=0; FAIL=0

new_sandbox() { mktemp -d "${GB_TEST_TMP:-${TMPDIR:-/tmp}}/gbinst.XXXXXX"; }

ok()   { PASS=$((PASS+1)); printf '  ok   %s\n' "$*"; }
bad()  { FAIL=$((FAIL+1)); printf '  FAIL %s\n' "$*"; }
check() { local d="$1"; shift; if "$@"; then ok "$d"; else bad "$d"; fi; }

# load_fn <file> <function>...: define the REAL function text from the script
# under test (from `name() {` to the first column-0 `}`), without running the
# script's dispatcher (the script has no source guard).
load_fn() {
    local file="$1" fn body; shift
    for fn in "$@"; do
        body="$(awk -v f="$fn" '$0 ~ "^"f"\\(\\) *\\{" {p=1} p{print} p&&/^}/{exit}' "$file")"
        [[ -n "$body" ]] || { echo "load_fn: $fn not found in $file" >&2; return 1; }
        eval "$body"
    done
}

# Minimal stand-ins for the script's own message helpers.
info() { :; }
warn() { printf '    (warn) %s\n' "$*" >&2; }

summary() { printf '%s: %d passed, %d failed\n' "$(basename "$0")" "$PASS" "$FAIL"; [[ $FAIL -eq 0 ]]; }

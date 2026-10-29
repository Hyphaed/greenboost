#!/usr/bin/env bash
# End-of-install reboot verdict: only say a reboot is needed when one would change something.
# Before: every Full Install ended with "REBOOT REQUIRED to activate GRUB params + hugepage
# pre-allocation", although the drop-in matched the running command line (2026-10-28) and the
# installer holds vm.nr_hugepages at 0 on purpose. These tests run the REAL function text.
source "$(dirname "$0")/lib.sh"
SB="$(new_sandbox)"; echo "sandbox: $SB"
C_AMBER=""; C_BOLD=""; C_RESET=""; C_GRAY=""
gb_ok() { printf 'OK %s\n' "$*"; }
load_fn "$SETUP" gb_cmdline_missing_params gb_report_reboot_need || exit 2

export GB_GRUB_DROPIN="$SB/99-greenboost.cfg" GB_CMDLINE_FILE="$SB/cmdline"
cat > "$GB_GRUB_DROPIN" <<'DROPIN'
# comment line
GRUB_CMDLINE_LINUX_DEFAULT="${GRUB_CMDLINE_LINUX_DEFAULT} transparent_hugepage=always skew_tick=1 nohz_full=4-7"
DROPIN

verdict() { GB_REBOOT_REASONS=("${PRESET[@]}"); gb_report_reboot_need 2>&1; }

# 1. everything already on the running command line
echo "quiet splash transparent_hugepage=always skew_tick=1 nohz_full=4-7 iommu=pt" > "$GB_CMDLINE_FILE"
PRESET=()
check "complete command line: nothing missing" test -z "$(gb_cmdline_missing_params)"
out="$(verdict)"
check "complete command line: says no reboot is needed" grep -q 'No reboot needed' <<<"$out"
check "complete command line: no warning banner" test -z "$(grep -E 'Reboot needed|REBOOT' <<<"$out" || true)"
check "the no-reboot message says hugepages stay at 0 on purpose" grep -q 'nr_hugepages stays 0' <<<"$out"

# 2. one parameter missing
echo "quiet splash transparent_hugepage=always nohz_full=4-7" > "$GB_CMDLINE_FILE"
check "one missing parameter is named exactly" test "$(gb_cmdline_missing_params)" = "skew_tick=1"
out="$(verdict)"
check "missing parameter: reboot needed and the parameter is listed" bash -c 'grep -q "Reboot needed" <<<"$1" && grep -q "skew_tick=1" <<<"$1"' _ "$out"
check "missing parameter: reassures nothing is broken" grep -q 'nothing is broken' <<<"$out"

# 3. whole-word match: a longer value must not satisfy a shorter one
echo "transparent_hugepage=always skew_tick=1 nohz_full=4-77" > "$GB_CMDLINE_FILE"
check "nohz_full=4-77 does not count as nohz_full=4-7" test "$(gb_cmdline_missing_params)" = "nohz_full=4-7"

# 4. a stuck module forces the banner even when the command line is complete
echo "transparent_hugepage=always skew_tick=1 nohz_full=4-7" > "$GB_CMDLINE_FILE"
PRESET=("the GreenBoost kernel module is stuck unloading")
out="$(verdict)"
check "stuck module: reboot needed with its reason" bash -c 'grep -q "Reboot needed" <<<"$1" && grep -q "stuck unloading" <<<"$1"' _ "$out"

# 5. unreadable drop-in or command line must not fail the install under set -e
( set -euo pipefail; GB_GRUB_DROPIN="$SB/missing.cfg"; gb_cmdline_missing_params ) >/dev/null 2>&1
check "missing drop-in returns 0 (safe under set -e)" test $? -eq 0
PRESET=()
out="$( set -euo pipefail; GB_GRUB_DROPIN="$SB/missing.cfg"; verdict )"
check "missing drop-in: no false reboot demand" grep -q 'No reboot needed' <<<"$out"

# 6. static: the obsolete text is gone and the existing "reboot required" sites record reasons
check "old unconditional banner text is gone" test -z "$(grep -n 'hugepage pre-allocation' "$SETUP" | grep -v '^[0-9]*:[[:space:]]*#' || true)"
check "both stuck-module sites record a reason" test "$(grep -c 'GB_REBOOT_REASONS+=(.*stuck unloading' "$SETUP")" -ge 2
check "installer still parses (bash -n)" bash -n "$SETUP"
summary

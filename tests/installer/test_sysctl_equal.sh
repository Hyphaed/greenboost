#!/usr/bin/env bash
# R2-02 / round 4: install (tune-sysctl), `tune` and tune-sysctl write the SAME sysctl file;
# no definitive-block tunables survive; INS-17 modprobe file is no longer written.
source "$(dirname "$0")/lib.sh"
SB="$(new_sandbox)"; echo "sandbox: $SB"; mkdir -p "$SB/etc/sysctl.d"
need_root() { :; }; gb_ok() { :; }; sysctl() { :; }; detect_hardware() { :; }
# Point the real function text at the fake root.
for fn in _tune_persist_sysctl cmd_tune_sysctl; do
    body="$(awk -v f="$fn" '$0 ~ "^"f"\\(\\) *\\{" {p=1} p{print} p&&/^}/{exit}' "$SETUP")"
    [[ -n "$body" ]] || { echo "$fn not found"; exit 2; }
    eval "${body//\/etc\//$SB/etc/}"
done
F="$SB/etc/sysctl.d/99-zzz-greenboost.conf"
cmd_tune_sysctl >/dev/null; cp "$F" "$SB/from_tune_sysctl"
rm -f "$F"
_tune_persist_sysctl >/dev/null; cp "$F" "$SB/from_tune"
check "tune-sysctl file equals tune file byte for byte" cmp -s "$SB/from_tune_sysctl" "$SB/from_tune"
cmd_tune_sysctl >/dev/null; _tune_persist_sysctl >/dev/null
check "install order (tune-sysctl then tune) leaves the same bytes" cmp -s "$F" "$SB/from_tune"
check "file is the thin 172-byte measured one" test "$(stat -c %s "$F")" = 172
check "no vm.overcommit_memory / max_map_count / new tunable written" \
    test -z "$(grep -E 'overcommit|max_map_count|compaction|vfs_cache' "$F" || true)"
check "installer text has no 2147483642 or overcommit_memory = 1" \
    test -z "$(grep -nE '2147483642|overcommit_memory *= *1' "$SETUP" | grep -v '^[0-9]*:[[:space:]]*#' || true)"
check "installer no longer writes NVreg_PreserveVideoMemoryAllocations=0 (INS-17)" \
    test -z "$(grep -n 'NVreg_PreserveVideoMemoryAllocations' "$SETUP" | grep -v '^[0-9]*:[[:space:]]*#' || true)"
check "uninstall still removes an older 99-nvidia-greenboost.conf" \
    test "$(grep -c '99-nvidia-greenboost.conf' "$SETUP")" -ge 2
summary

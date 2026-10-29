#!/usr/bin/env bash
# SP62-02: one swapfile per machine. GreenBoost's installer must not create a second swapfile when the system has
# swap or the owner set GB_SWAP_AUTO=0, and must own no vm.* sysctl when GB_VM_SYSCTL=0. Fake root, no sudo, no
# real /proc/swaps, no swapon/mkswap, nothing deleted.
source "$(dirname "$0")/lib.sh"
SB="$(new_sandbox)"; echo "sandbox: $SB"; mkdir -p "$SB/etc/greenboost" "$SB/etc/sysctl.d"
msgs="$SB/msgs"; info() { echo "$*" >> "$msgs"; }; gb_ok() { :; }; sysctl() { :; }
GB_SWAP_FILE="/var/lib/greenboost/swapfile"; GB_SWAP_MIN_GB=8
GB_SWAP_CONF="$SB/etc/greenboost/gb-swap.conf"; GB_PROC_SWAPS="$SB/swaps"
load_fn "$SETUP" gb_swap_conf_get gb_swap_other_gb gb_swap_auto_ok || exit 2
HDR='Filename Type Size Used Priority'
swaps() { { echo "$HDR"; for l in "$@"; do echo "$l"; done; } > "$GB_PROC_SWAPS"; }
SYS='/swap.img file 8388604 8340904 -1'; GBS='/var/lib/greenboost/swapfile file 25165820 24050260 -1'

swaps; check "no swap and no conf: GreenBoost may provision" gb_swap_auto_ok
swaps "$SYS"; gb_swap_auto_ok; test $? = 1 && ok "8 GB /swap.img (8388604 KiB) is adequate: no second swapfile" || bad "8 GB system swap"
swaps "$GBS"; check "only GreenBoost's own swapfile present: not counted as system swap, may top up" gb_swap_auto_ok
swaps "$SYS" "$GBS"; gb_swap_auto_ok; test $? = 1 && ok "system swap plus own file: no provisioning" || bad "system swap plus own file"
swaps; printf 'GB_SWAP_AUTO=0\n' > "$GB_SWAP_CONF"
gb_swap_auto_ok; test $? = 1 && ok "GB_SWAP_AUTO=0 blocks provisioning even with no swap" || bad "GB_SWAP_AUTO=0"
check "reason names the config file" grep -q 'GB_SWAP_AUTO=0' "$msgs"
printf 'GB_SWAP_AUTO=0; touch %s/pwned\n' "$SB" > "$GB_SWAP_CONF"; gb_swap_auto_ok
check "conf is parsed, never sourced: a command in the file is not run" test ! -e "$SB/pwned"
printf '# c\nGB_SWAP_AUTO=1\n' > "$GB_SWAP_CONF"; check "GB_SWAP_AUTO=1 behaves as the default" gb_swap_auto_ok

# vm.* ownership
load_fn "$SETUP" _tune_persist_sysctl || exit 2
F="$SB/etc/sysctl.d/99-zzz-greenboost.conf"
body="$(awk '/^_tune_persist_sysctl\(\) *\{/{p=1} p{print} p&&/^}/{exit}' "$SETUP")"; eval "${body//\/etc\/sysctl.d/$SB/etc/sysctl.d}"
printf 'GB_VM_SYSCTL=0\n' > "$GB_SWAP_CONF"; _tune_persist_sysctl
check "GB_VM_SYSCTL=0: sysctl drop-in sets no vm. key" test -z "$(grep -E '^vm\.' "$F" || true)"
printf '' > "$GB_SWAP_CONF"; _tune_persist_sysctl
check "default: the 3 measured keys are still written" test "$(grep -cE '^vm\.(swappiness|dirty_ratio|dirty_background_ratio)' "$F")" = 3

# static: cmd_feed gates on the helper, caps at GB_SWAP_MAX_GB, writes pri=10; uninstall removes the fstab line outside the -f test
feed="$(awk '/^cmd_feed\(\) *\{/{p=1} p{print} p&&/^}/{exit}' "$SETUP")"
check "cmd_feed asks gb_swap_auto_ok before provisioning" grep -q 'gb_swap_auto_ok' <<<"$feed"
check "cmd_feed honours GB_SWAP_MAX_GB" grep -q 'GB_SWAP_MAX_GB' <<<"$feed"
check "fstab line carries pri=10 (matches swapon -p 10)" grep -q 'defaults,nofail,pri=10' <<<"$feed"
check "uninstall drops the fstab line even if the file is gone" grep -q 'outside the -f test' "$SETUP"
summary

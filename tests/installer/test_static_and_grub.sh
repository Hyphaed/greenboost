#!/usr/bin/env bash
# Static regression checks on the four scripts plus the GRUB drop-in round trip.
source "$(dirname "$0")/lib.sh"
SCRIPTS=("$REPO/greenboost_setup.sh" "$REPO/greenboost_setup_arch.sh" "$REPO/configure_boot_machine.sh" "$REPO/greenboost_boot_guard.sh")

for f in "${SCRIPTS[@]}"; do check "bash -n $(basename "$f")" bash -n "$f"; done
if command -v shellcheck >/dev/null 2>&1; then
    for f in "${SCRIPTS[@]}"; do
        check "shellcheck -S error $(basename "$f")" shellcheck -S error -e SC1090,SC1091 "$f"
    done
else
    echo "  skip shellcheck: not installed on this machine (bash -n and manual review only)"
fi

# One-line invariants that each fix an audited defect. A regression re-opens it.
check "no OLLAMA_KEEP_ALIVE=-1 written anywhere in the installer (INS-01)" \
    test -z "$(grep -n 'OLLAMA_KEEP_ALIVE=-1' "$SETUP" | grep -v '^[0-9]*:[[:space:]]*#' || true)"
check "no 'eval echo ~user' (INS-08)" test -z "$(grep -n 'eval echo "~' "$SETUP" || true)"
check "preloaded audit library is never copied with a bare cp (INS-02)" test -z "$(grep -n 'cp "\$audit_src"' "$SETUP" || true)"
check "ollama installer not run from a fixed /tmp name, locally or on feeders (INS-09)" test -z "$(grep -n '/tmp/gb_ollama_install.sh' "$SETUP" | grep -v '^[0-9]*:\s*#' || true)"
check "no 'lsmod | grep -q' (also with 2>/dev/null) and no 'dkms status | grep -q' under pipefail in any script (INS-10)" \
    test -z "$(grep -nE '^[^#]*lsmod[^|#]*\| *grep -q' "$REPO"/*.sh || true)$(grep -nE '^[^#]*dkms status[^|#]*\| *grep -q' "$SETUP" "$REPO/install_module.sh" "$REPO/greenboost_setup_arch.sh" "$REPO/greenboost_setup_rocky.sh" || true)"
check "boot guard checks the module through /sys/module, not a pipe (INS-10)" \
    grep -q '^if \[\[ -d /sys/module/greenboost \]\]' "$REPO/greenboost_boot_guard.sh"
check "no fixed /tmp/gb_*.log or *.\$\$ file name left in the local install path (INS-15)" \
    test -z "$(grep -nE '/tmp/gb_[a-z_]+(\.log|\.\$\$)' "$SETUP" | grep -vE 'mktemp|gb_src|gb_update|gb_feeder|gb_ollama\.' || true)"
check "uninstall takes --dry-run and never prompts (INS-05, no y/n in do_purge)" \
    bash -c "grep -q -- '--dry-run) _gb_uninstall_plan' '$SETUP' && ! sed -n '/^do_purge()/,/^}/p' '$SETUP' | grep -qE 'read -r _ans|read -rp'"
check "no bare \$USER or 'pipeline || echo N' counter that appends a second line (set -u / pipefail)" \
    test -z "$(grep -nE '\$USER\b' "$SETUP" | grep -v 'USER:-' || true)$(grep -nE '(wc -l|grep -c[^|]*)\|\| *echo' "$SETUP" || true)"
check "Arch and Rocky uninstall hand over to the main script (INS-11)" \
    bash -c "grep -q 'exec \"\$MODULE_DIR/greenboost_setup.sh\" --skip-update-check uninstall' '$REPO/greenboost_setup_arch.sh' '$REPO/greenboost_setup_rocky.sh'"
check "sudoers rules are validated before they are renamed into place (INS-07)" \
    test "$(grep -c 'visudo -c -f' "$SETUP")" -ge 2 -a -z "$(grep -n '> /etc/sudoers.d/greenboost &&' "$SETUP" || true)"

check "reinstall purge keeps /etc/greenboost/backup so the pre-install snapshot survives (INS-14)" grep -q 'synapse_token backup .*; do' "$SETUP"

# Every gb_*.py that an installed module imports must be in cmd_install_python_files' list (round 4: gb_lease_client.py was missing).
check "every imported gb_*.py module is in the installer's python file list (checks/check_python_manifest.py)" \
    bash -c "cd '$REPO' && python3 checks/check_python_manifest.py"

# The cuda-keyring pin for this distro must equal the copy shipped in the repo.
pin="$(grep -o '\[ubuntu2604\]=[0-9a-f]\{64\}' "$SETUP" | cut -d= -f2)"
have="$(sha256sum "$REPO/cuda-keyring_1.1-1_all.deb" 2>/dev/null | cut -d' ' -f1)"
check "cuda-keyring pin for ubuntu2604 matches the keyring .deb in the repo" test -n "$pin" -a "$pin" = "$have"

# GRUB drop-in: render, validate, write, rewrite is idempotent, bad text is rejected.
load_fn "$SETUP" _gb_grub_dropin_render gb_managed_write || exit 2
SB="$(new_sandbox)"; echo "sandbox: $SB"
flags='transparent_hugepage=always skew_tick=1 numa_balancing=disable'
_gb_grub_dropin_render "$flags" | bash -n /dev/stdin; check "grub drop-in: rendered text is valid shell" test $? -eq 0
_gb_grub_dropin_render "$flags" | gb_managed_write "$SB/99-greenboost.cfg" 644
r1="$(sed '/^# gb-managed-sha256: /d' "$SB/99-greenboost.cfg")"
check "grub drop-in: content equals the render (stamp stripped), so the script's equality test sees 'nothing to change'" \
    test "$r1" = "$(_gb_grub_dropin_render "$flags")"
ino="$(stat -c %i "$SB/99-greenboost.cfg")"
_gb_grub_dropin_render "$flags" | gb_managed_write "$SB/99-greenboost.cfg" 644
check "grub drop-in: second write is a no-op" test "$(stat -c %i "$SB/99-greenboost.cfg")" = "$ino"
( GRUB_CMDLINE_LINUX_DEFAULT=""; . "$SB/99-greenboost.cfg"; true ); check "grub drop-in: sourcing it (as grub-mkconfig does) succeeds" test $? -eq 0
GRUB_CMDLINE_LINUX_DEFAULT="quiet"; . "$SB/99-greenboost.cfg"
check "grub drop-in: sourced result appends to the existing command line" test "$GRUB_CMDLINE_LINUX_DEFAULT" = "quiet $flags"
_gb_grub_dropin_render 'bad"flag $(touch '"$SB"'/injected)' | bash -n /dev/stdin 2>/dev/null
check "grub drop-in: a flag that breaks the quoting is caught by bash -n" test $? -ne 0
check "grub drop-in: nothing was executed by the check" test ! -e "$SB/injected"
summary

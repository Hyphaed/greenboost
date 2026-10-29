#!/usr/bin/env bash
# Run every installer test. No sudo. Sandboxes are left under $GB_TEST_TMP
# (default $TMPDIR or /tmp) for the caller to remove.
cd "$(dirname "$0")" || exit 2
rc=0
for t in test_write_primitives.sh test_initramfs.sh test_static_and_grub.sh test_sysctl_equal.sh test_single_swapfile.sh test_ollama_ctx.sh test_shim_dbgfile.sh test_reboot_banner.sh; do
    echo "== $t"; ./"$t" || rc=1
done
exit $rc

#!/bin/bash
# 99greenboost-exclude , dracut module.
#
# Companion to initramfs/zz-greenboost-exclude (the initramfs-tools hook). Same
# job, different generator: keep GreenBoost out of early userspace.
#
# Why a MODULE and not just omit_drivers (incident 2026-08-21, this box):
#
#   /etc/dracut.conf.d/99-greenboost-exclude.conf's omit_drivers keeps
#   greenboost.ko out of the image, and that part works. But omit_drivers only
#   omits KERNEL MODULES. dracut's own 00systemd module copies /etc/tmpfiles.d
#   and /etc/modules-load.d into the image wholesale, so the *config fragments*
#   went in regardless, and the initrd's systemd-tmpfiles then failed on every
#   boot with:
#
#       /etc/tmpfiles.d/greenboost-gaming.conf:14:
#           Failed to resolve group 'greenboost': Unknown group
#
#   The group is not missing , the initrd is a different root filesystem with
#   its own /etc/group, and 'greenboost' is not in it. There is nothing for the
#   rule to act on that early either: it chmods /sys/module/greenboost/, which
#   cannot exist before the module is loaded. The fix is to keep the fragment
#   out of the boot image, which is what install() below does.
#
#   Do NOT try to fix this with a sysusers.d fragment on the real root , it is
#   not in the initrd either, so it cannot help.
#
# The 99 prefix is load-bearing: dracut runs module install() hooks in numeric
# order, so this must sort after 00systemd or there is nothing there to remove.

check() {
    return 0
}

depends() {
    echo systemd
    return 0
}

install() {
    rm -f "$initdir"/etc/tmpfiles.d/greenboost.conf \
          "$initdir"/etc/tmpfiles.d/greenboost-gaming.conf \
          "$initdir"/usr/lib/tmpfiles.d/greenboost.conf \
          "$initdir"/usr/lib/tmpfiles.d/greenboost-gaming.conf \
          "$initdir"/etc/modules-load.d/greenboost.conf \
          "$initdir"/usr/lib/modules-load.d/greenboost.conf \
        2>/dev/null
    return 0
}

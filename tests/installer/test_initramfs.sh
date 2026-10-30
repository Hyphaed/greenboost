#!/usr/bin/env bash
# Boot-image safety: gb_initramfs_regen_one against a fake /boot with fake generators.
# Proves: the previous image is backed up off /boot, a failed or truncated
# regeneration is rolled back, a good one is kept, and *.gb-prev style names
# never count as kernels.
source "$(dirname "$0")/lib.sh"
load_fn "$SETUP" gb_initramfs_image_for gb_initramfs_kernels gb_initramfs_image_sane gb_initramfs_regen_one gb_initramfs_verify_clean || exit 2
SB="$(new_sandbox)"; echo "sandbox: $SB"
export GB_BOOT_DIR="$SB/boot" GB_MODULES_DIR="$SB/lib/modules" GB_INITRD_BACKUP_DIR="$SB/var/initrd-prev"
export _GB_INITRAMFS_GEN=initramfs-tools
mkdir -p "$GB_BOOT_DIR" "$GB_MODULES_DIR/6.9.9-test" "$SB/bin"; export PATH="$SB/bin:$PATH"
gb_initramfs_generator() { printf '%s' "$_GB_INITRAMFS_GEN"; }
KV=6.9.9-test; IMG="$GB_BOOT_DIR/initrd.img-$KV"

# fake archive lister: one "entry" per line of the file
for t in lsinitramfs lsinitrd; do printf '#!/bin/sh\ncat "$1"\n' > "$SB/bin/$t"; chmod +x "$SB/bin/$t"; done  # stubs shadow any real tool
# fake generator whose behaviour the test selects with GEN_MODE
cat > "$SB/bin/update-initramfs" <<'GEN'
#!/bin/sh
img="$GB_BOOT_DIR/initrd.img-$3"
case "$GEN_MODE" in
  good)     printf 'usr/lib/modules/nvme.ko\nusr/lib/modules/ext4.ko\nNEW\n' > "$img" ;;
  truncate) : > "$img" ;;
  fail)     echo "boom" >&2; exit 1 ;;
  garbage)  : > "$img" ;;
esac
GEN
chmod +x "$SB/bin/update-initramfs"

printf 'usr/lib/modules/nvme.ko\nOLD\n' > "$IMG"

GEN_MODE=good gb_initramfs_regen_one "$KV"; rc=$?
check "good regeneration returns 0" test $rc -eq 0
check "good regeneration: new image is in place" grep -q NEW "$IMG"
check "backup of the previous image kept OFF /boot" grep -q OLD "$GB_INITRD_BACKUP_DIR/initrd.img-$KV"
check "nothing but the image itself added to /boot" test "$(ls "$GB_BOOT_DIR" | wc -l)" -eq 1

printf 'usr/lib/modules/nvme.ko\nGOODBEFORE\n' > "$IMG"
GEN_MODE=truncate gb_initramfs_regen_one "$KV" 2>"$SB/w1"; rc=$?
check "truncated image: returns non-zero" test $rc -ne 0
check "truncated image: previous image restored" grep -q GOODBEFORE "$IMG"
check "truncated image: warning says it was restored" grep -q 'previous image restored' "$SB/w1"
check "restore leaves no .new file in /boot" test -z "$(ls "$GB_BOOT_DIR" | grep '\.new$' || true)"

printf 'usr/lib/modules/nvme.ko\nGOODBEFORE2\n' > "$IMG"
GEN_MODE=fail gb_initramfs_regen_one "$KV" 2>/dev/null; rc=$?
check "generator failure: returns non-zero" test $rc -ne 0
check "generator failure: image still the old good one" grep -q GOODBEFORE2 "$IMG"

# no previous image at all (new kernel): failure must not invent one
rm_target="$GB_BOOT_DIR/initrd.img-6.9.8-new"; mkdir -p "$GB_MODULES_DIR/6.9.8-new"
GEN_MODE=fail gb_initramfs_regen_one 6.9.8-new 2>"$SB/w2"; rc=$?
check "no previous image: failure reported, nothing fabricated" test $rc -ne 0 -a ! -e "$rm_target"

# kernel listing ignores backup-like names
printf 'x\n' > "$GB_BOOT_DIR/initrd.img-$KV.gb-prev"; printf 'x\n' > "$GB_BOOT_DIR/initrd.img-$KV.dpkg-bak"
mkdir -p "$GB_MODULES_DIR/$KV.gb-prev"
check "kernel list ignores .gb-prev and .dpkg-bak names" test "$(gb_initramfs_kernels | tr '\n' ' ')" = "$KV "

# verify_clean: module file present -> 1; the exclusion machinery's own name is not a module
printf 'usr/lib/modules/6.9.9/updates/dkms/greenboost.ko\n' > "$IMG"; gb_initramfs_verify_clean "$KV"; rc=$?
check "verify_clean: stale greenboost.ko is reported (1)" test $rc -eq 1
printf 'usr/lib/dracut/modules.d/99greenboost-exclude/module-setup.sh\n' > "$IMG"; gb_initramfs_verify_clean "$KV"; rc=$?
check "verify_clean: the exclusion module's name is not a false positive (0)" test $rc -eq 0
summary

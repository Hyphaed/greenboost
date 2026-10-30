#!/usr/bin/env bash
# gb_atomic_install, gb_backup_once, gb_managed_write against a fake root.
source "$(dirname "$0")/lib.sh"
load_fn "$SETUP" gb_atomic_install gb_backup_once gb_managed_write gb_user_home || exit 2
SB="$(new_sandbox)"; echo "sandbox: $SB"

# --- gb_atomic_install -------------------------------------------------------
printf 'v1\n' > "$SB/src1"; printf 'v2\n' > "$SB/src2"
gb_atomic_install "$SB/src1" "$SB/lib.so" 755
check "atomic: creates dest with mode 755" test "$(stat -c %a "$SB/lib.so")" = 755
exec 7< "$SB/lib.so"; ino1="$(stat -c %i "$SB/lib.so")"
gb_atomic_install "$SB/src2" "$SB/lib.so" 755
check "atomic: dest content replaced" test "$(cat "$SB/lib.so")" = v2
check "atomic: new inode (rename, not in-place truncate)" test "$(stat -c %i "$SB/lib.so")" != "$ino1"
read -r held <&7; exec 7<&-
check "atomic: a reader that had the old file open still sees the OLD content" test "$held" = v1
check "atomic: no temp files left behind" test -z "$(ls "$SB" | grep gbnew || true)"
GB_DRY_RUN=1 gb_atomic_install "$SB/src1" "$SB/lib.so" 755
check "atomic: GB_DRY_RUN=1 changes nothing" test "$(cat "$SB/lib.so")" = v2
gb_atomic_install "$SB/nonexistent" "$SB/lib.so" 755 2>/dev/null
check "atomic: failed install leaves dest intact and no temp" test "$(cat "$SB/lib.so")" = v2 -a -z "$(ls "$SB" | grep gbnew || true)"

# --- gb_backup_once ----------------------------------------------------------
printf 'original\n' > "$SB/unit.service"
gb_backup_once "$SB/unit.service"
printf 'rewritten by us\n' > "$SB/unit.service"
gb_backup_once "$SB/unit.service"
check "backup_once: keeps the FIRST copy, not the latest" test "$(cat "$SB/unit.service.gb-orig")" = original
gb_backup_once "$SB/missing"; check "backup_once: missing file is not an error" test $? -eq 0

# --- gb_managed_write --------------------------------------------------------
D="$SB/ollama.service.d"; mkdir -p "$D"; F="$D/99-greenboost.conf"
printf '[Service]\nEnvironment="A=1"\n' | gb_managed_write "$F" 644
check "managed: first write is stamped" grep -q '^# gb-managed-sha256: [0-9a-f]\{64\}$' "$F"
check "managed: mode 644" test "$(stat -c %a "$F")" = 644
ino="$(stat -c %i "$F")"; mt="$(stat -c %Y.%y "$F")"
printf '[Service]\nEnvironment="A=1"\n' | gb_managed_write "$F" 644
check "managed: identical content is a no-op (same inode and mtime)" test "$(stat -c %i "$F")" = "$ino" -a "$(stat -c %Y.%y "$F")" = "$mt"
check "managed: no backup when nothing was hand-edited" test -z "$(ls "$D" | grep gb-prev || true)"
printf '[Service]\nEnvironment="A=2"\n' | gb_managed_write "$F" 644
check "managed: generator change without hand edit overwrites silently" test -z "$(ls "$D" | grep gb-prev || true)"
check "managed: new content landed" grep -q 'A=2' "$F"
# hand edit: change a value, keep the stamp
sed -i 's/A=2/A=HAND/' "$F"
printf '[Service]\nEnvironment="A=3"\n' | gb_managed_write "$F" 644 2>"$SB/warn.txt"
bak="$(ls "$D" | grep '\.gb-prev-' | head -1)"
check "managed: hand edit is detected (a gb-prev copy exists)" test -n "$bak"
check "managed: the kept copy holds the hand edit" grep -q 'A=HAND' "$D/$bak"
check "managed: the warning names the backup" grep -q 'gb-prev-' "$SB/warn.txt"
check "managed: new content still lands after a hand edit" grep -q 'A=3' "$F"
# legacy file with no stamp and different content
G="$D/legacy.conf"; printf 'old installer output\n' > "$G"
printf 'new\n' | gb_managed_write "$G" 644 2>/dev/null
check "managed: unstamped legacy file is backed up before replacement" test -n "$(ls "$D" | grep '^legacy.conf.gb-prev-' || true)"
# systemd and sudo ignore non-.conf suffixes: backups must not end in .conf
check "managed: backup names do not end in .conf" test -z "$(ls "$D" | grep 'gb-prev.*\.conf$' || true)"
printf 'x\n' | GB_DRY_RUN=1 gb_managed_write "$SB/never" 644
check "managed: GB_DRY_RUN=1 writes nothing" test ! -e "$SB/never"

# --- gb_user_home ------------------------------------------------------------
check "user_home: root resolves without eval" test "$(gb_user_home root)" = /root
check "user_home: metacharacters are data, not code" test -z "$(gb_user_home 'x;touch '"$SB"'/pwned')" -a ! -e "$SB/pwned"
summary

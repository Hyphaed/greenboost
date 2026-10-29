#!/usr/bin/env bash
# Runs INSIDE the throwaway container (see container_gate.sh). The repo is mounted
# read-only at /repo. Real: the installer, its uninstall, the file system. Stubbed:
# everything that needs a kernel, a GPU, a compiler or systemd as PID 1 (see $STUBS).
set -u
OUT=/out; STUBS=/opt/stubs; REPO=/repo
export DEBIAN_FRONTEND=noninteractive
mkdir -p "$OUT"

# ---- base image: only what a stock server has plus what the installer calls ----
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq python3 sudo kmod rsync curl iproute2 procps pciutils ca-certificates udev systemd initramfs-tools >/dev/null 2>&1 \
    || { echo "GATE-ERROR: apt prerequisites failed"; exit 3; }

# ---- stubs: no kernel module, no GPU, no compiler, no systemd as PID 1 ----
mkdir -p "$STUBS"
for c in systemctl depmod update-initramfs update-grub grub-mkconfig udevadm dkms ldconfig sysctl apt-get \
         dpkg-reconfigure mokutil gcc cc clang sync setpci; do
    printf '#!/bin/sh\necho "stub %s $*" >> /var/tmp/stub.log\nexit 0\n' "$c" > "$STUBS/$c"
done
# kernel-module trio: flag file stands in for "loaded"
cat > "$STUBS/lsmod" <<'S'
#!/bin/sh
echo "Module                  Size  Used by"
[ -f /var/tmp/gb_loaded ] && echo "greenboost            1048576  0"
exit 0
S
for c in modprobe insmod; do
    printf '#!/bin/sh\necho "stub %s $*" >> /var/tmp/stub.log\ncase "$*" in *-r*) rm -f /var/tmp/gb_loaded;; *) touch /var/tmp/gb_loaded;; esac\nexit 0\n' "$c" > "$STUBS/$c"
done
printf '#!/bin/sh\necho "stub rmmod $*" >> /var/tmp/stub.log\nrm -f /var/tmp/gb_loaded\nexit 0\n' > "$STUBS/rmmod"
# make: no compiler. Only the file placement of `install-libs` is emulated, from the PREBUILT binaries in the repo.
cat > "$STUBS/make" <<'S'
#!/bin/sh
echo "stub make $*" >> /var/tmp/stub.log
case " $* " in *" install-libs "*)
  R=/repo
  install -d /usr/local/lib /usr/local/bin /usr/local/lib/i386-linux-gnu
  for f in libgreenboost_cuda.so libgreenboost_audit.so libgreenboost_vmm_override.so; do [ -f $R/$f ] && cp $R/$f /usr/local/lib/$f.new && mv /usr/local/lib/$f.new /usr/local/lib/$f; done
  [ -f $R/greenboost-netd ] && cp $R/greenboost-netd /usr/local/bin/
  [ -f $R/libgreenboost_audit32.so ] && cp $R/libgreenboost_audit32.so /usr/local/lib/i386-linux-gnu/libgreenboost_audit.so
  ;;
esac
exit 0
S
chmod +x "$STUBS"/*
mkdir -p "/lib/modules/$(uname -r)/build" && : > "/lib/modules/$(uname -r)/build/Makefile"   # "kernel headers"
: > /dev/greenboost                                                                           # "the device node"
export PATH="$STUBS:$PATH"
# The dataflux event log is the owner's telemetry history, kept by design on uninstall: point it outside the compared tree.
export GREENBOOST_DATAFLUX_LOG=/var/tmp/dataflux.jsonl

snap() {   # path|type|mode|uid|gid|size|link|sha256 for every file the installer could touch
    find / -xdev \( -path /proc -o -path /sys -o -path /dev -o -path /run -o -path /tmp -o -path /var/tmp -o -path /repo -o -path /out -o -name __pycache__ \) -prune -o -path /etc/group- -prune -o -path /etc/gshadow- -prune -o -path /etc/passwd- -prune -o -path /etc/shadow- -prune -o -print0 2>/dev/null \
      | sort -z | while IFS= read -r -d '' p; do
        t=$(stat -c '%F|%a|%u|%g|%s|%N' "$p" 2>/dev/null) || continue
        h=-; [ -f "$p" ] && [ ! -L "$p" ] && h=$(sha256sum "$p" | cut -c1-16)
        printf '%s|%s|%s\n' "$p" "$t" "$h"
    done
}
G=0; res() { printf '%-62s %s\n' "$1" "$2" | tee -a "$OUT/result.txt"; [ "$2" = PASS ] || G=1; }
SETUP="$REPO/greenboost_setup.sh"
run() { local tag="$1"; shift; timeout 1500 "$SETUP" --skip-update-check "$@" </dev/null >"$OUT/$tag.log" 2>&1; echo $? > "$OUT/$tag.rc"; cat "$OUT/$tag.rc"; }

snap > "$OUT/0-before.txt"
echo "files before: $(wc -l < "$OUT/0-before.txt")" | tee "$OUT/result.txt"

# 1. dry run on a clean tree changes nothing
rc=$(run dry0 uninstall --dry-run); snap > "$OUT/1.txt"
[ "$rc" = 0 ] && cmp -s "$OUT/0-before.txt" "$OUT/1.txt" && res "uninstall --dry-run on a clean tree: rc 0, tree unchanged" PASS || res "uninstall --dry-run on a clean tree" FAIL

# 2. install (up to the stubs)
rc=$(run install_module_only module-only); echo "module-only rc=$rc" | tee -a "$OUT/result.txt"
rc2=$(run install_full full-install); echo "full-install rc=$rc2" | tee -a "$OUT/result.txt"
snap > "$OUT/2-installed.txt"
n_new=$(comm -13 <(cut -d'|' -f1 "$OUT/0-before.txt" | sort) <(cut -d'|' -f1 "$OUT/2-installed.txt" | sort) | wc -l)
echo "paths added by install: $n_new" | tee -a "$OUT/result.txt"
[ "$n_new" -gt 50 ] && res "install wrote a real tree (>50 new paths)" PASS || res "install wrote a real tree" FAIL

# 3. dry run on the installed tree: prints a plan, changes nothing
rc=$(run dry1 uninstall --dry-run); snap > "$OUT/3.txt"
[ "$rc" = 0 ] && cmp -s "$OUT/2-installed.txt" "$OUT/3.txt" && grep -q 'would remove' "$OUT/dry1.log" \
    && res "uninstall --dry-run on the installed tree: plan printed, tree unchanged" PASS || res "uninstall --dry-run on the installed tree" FAIL

# 4. real uninstall, no prompts (stdin is /dev/null), then compare with the tree before
rc=$(run uninstall uninstall); snap > "$OUT/4-after-uninstall.txt"
[ "$rc" = 0 ] && res "uninstall exits 0 with no prompt" PASS || res "uninstall exits 0 with no prompt (rc=$rc)" FAIL
diff "$OUT/0-before.txt" "$OUT/4-after-uninstall.txt" > "$OUT/tree.diff"
[ -s "$OUT/tree.diff" ] && res "tree diff install+uninstall vs before is EMPTY ($(grep -c '^[<>]' "$OUT/tree.diff") lines differ)" FAIL \
                        || res "tree diff install+uninstall vs before is EMPTY" PASS

# 5. idempotent: a second uninstall succeeds and changes nothing
rc=$(run uninstall2 uninstall); snap > "$OUT/5.txt"
[ "$rc" = 0 ] && cmp -s "$OUT/4-after-uninstall.txt" "$OUT/5.txt" && res "second uninstall: rc 0, tree unchanged" PASS || res "second uninstall idempotent (rc=$rc)" FAIL

# 7. the standalone DKMS installer (install_module.sh): install, preview, uninstall, tree compare
IM="$REPO/install_module.sh"
snap > "$OUT/7-before-im.txt"
timeout 600 "$IM" </dev/null >"$OUT/im_install.log" 2>&1; echo $? > "$OUT/im_install.rc"
snap > "$OUT/7-im-installed.txt"
[ "$(cat "$OUT/im_install.rc")" = 0 ] && res "install_module.sh installs (dkms/make/modprobe stubbed): rc 0" PASS || res "install_module.sh installs (rc=$(cat "$OUT/im_install.rc"))" FAIL
timeout 600 "$IM" --uninstall --dry-run </dev/null >"$OUT/im_dry.log" 2>&1; im_rc=$?; snap > "$OUT/7-im-dry.txt"
[ "$im_rc" = 0 ] && cmp -s "$OUT/7-im-installed.txt" "$OUT/7-im-dry.txt" && grep -q 'would remove' "$OUT/im_dry.log" \
    && res "install_module.sh --uninstall --dry-run: plan printed, tree unchanged" PASS || res "install_module.sh --uninstall --dry-run" FAIL
timeout 600 "$IM" --uninstall </dev/null >"$OUT/im_un.log" 2>&1; im_rc=$?; snap > "$OUT/7-im-after.txt"
diff "$OUT/7-before-im.txt" "$OUT/7-im-after.txt" > "$OUT/im_tree.diff"
[ "$im_rc" = 0 ] && [ ! -s "$OUT/im_tree.diff" ] && res "install_module.sh install+uninstall: rc 0, tree diff EMPTY" PASS \
    || res "install_module.sh install+uninstall (rc=$im_rc, $(grep -c '^[<>]' "$OUT/im_tree.diff") lines differ)" FAIL
timeout 600 "$IM" --uninstall </dev/null >"$OUT/im_un2.log" 2>&1; im_rc=$?; snap > "$OUT/7-im-after2.txt"
[ "$im_rc" = 0 ] && cmp -s "$OUT/7-im-after.txt" "$OUT/7-im-after2.txt" && res "install_module.sh second uninstall: rc 0, tree unchanged" PASS || res "install_module.sh second uninstall" FAIL
# 6. the manifest read on the installed tree agrees with what the install really wrote
python3 -I - "$OUT" "$REPO/tools/manifest/manifest.json" <<'PY' | tee -a "$OUT/result.txt"
import json, sys
out, mf = sys.argv[1], sys.argv[2]
before = {l.split("|")[0] for l in open(out + "/0-before.txt")}
inst = {l.split("|")[0] for l in open(out + "/2-installed.txt")}
m = json.load(open(mf)); known = {e.get("path") or e.get("file") for e in m["entries"]}
new = sorted(inst - before)
unk = [p for p in new if p not in known and not any(p.startswith(k.rstrip("/") + "/") for k in known if k)]
print("installed-but-not-in-manifest paths: %d of %d" % (len(unk), len(new)))
open(out + "/unlisted.txt", "w").write("\n".join(unk) + "\n")
PY
grep -E '^(stub|GATE)' /var/tmp/stub.log 2>/dev/null | sort | uniq -c | sort -rn | head -0
cut -d' ' -f2 /var/tmp/stub.log 2>/dev/null | sort | uniq -c | sort -rn > "$OUT/stubs-called.txt"
echo "GATE-EXIT=$G"; exit $G

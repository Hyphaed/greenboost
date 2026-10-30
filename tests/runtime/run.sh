#!/usr/bin/env bash
# Userspace tests for the runtime audit (GA2).  Compiles small harnesses with gcc, loads nothing
# into any process, touches no GPU, no module, no root.   Usage: tests/runtime/run.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/.build"
command -v gcc >/dev/null || { echo "SKIP: no gcc"; exit 0; }
mkdir -p "$OUT"
fail=0
ok()  { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }

SHIM="$REPO/greenboost_cuda_shim.c"

# 1. the testable region exists exactly once and is extracted verbatim
[ "$(grep -c 'GB_TESTABLE_BEGIN gb_report_logic' "$SHIM")" = 1 ] && [ "$(grep -c 'GB_TESTABLE_END gb_report_logic' "$SHIM")" = 1 ] \
  && ok "region markers present once" || bad "region markers"
sed -n '/GB_TESTABLE_BEGIN gb_report_logic/,/GB_TESTABLE_END gb_report_logic/p' "$SHIM" > "$OUT/report_logic.inc"

# 2. unit tests of the extracted logic
gcc -std=gnu11 -Wall -Wextra -Werror -I"$OUT" -o "$OUT/test_report_logic" "$HERE/test_report_logic.c"
"$OUT/test_report_logic" && ok "report logic unit tests" || bad "report logic unit tests"

# 3. the shim is wired to that logic (no private copy of the old inflation left)
[ "$(grep -c 'gb_nvml_inflate(g_report_physical_vram' "$SHIM")" = 3 ] && ok "3 NVML hooks use gb_nvml_inflate" || bad "NVML hooks not wired"
! grep -q 'memory->total += ' "$SHIM" && ok "no raw NVML total inflation left" || bad "raw NVML inflation still present"
grep -q 'physical_vram_source=' "$SHIM" && grep -q 'gb_phys_mem_now(&_cached_free' "$SHIM" && ok "stats use gb_phys_mem_now" || bad "stats not wired"
grep -q 'gb_report_profile_resolve(prof' "$SHIM" && ok "profile resolved at init" || bad "profile not wired"

# 4. syntax checks (nothing is linked or loaded)
gcc -std=gnu11 -fPIC -fsyntax-only -I"$REPO" "$SHIM" && ok "shim syntax" || bad "shim syntax"
gcc -std=gnu11 -fPIC -fsyntax-only "$REPO/greenboost_vmm_override.c" && ok "vmm override syntax" || bad "vmm override syntax"

# 5b. proposed patch 0003 (NOT applied to the repository) still applies cleanly to the current
# source and the patched copy compiles; nothing in the repository is changed.
P3="$REPO/docs/greenboost-audit/patches/0003-shim-free-ordering-passthrough-oom-evict-fork.patch"
if [ -e "$P3" ] && command -v patch >/dev/null; then
  mkdir -p "$OUT/p3" && cp "$SHIM" "$OUT/p3/greenboost_cuda_shim.c"
  if patch -s -p1 -d "$OUT/p3" < "$P3" && gcc -std=gnu11 -fPIC -fsyntax-only -I"$REPO" "$OUT/p3/greenboost_cuda_shim.c"; then
    ok "patch 0003 applies to a scratch copy and compiles"
  else bad "patch 0003 is stale or does not compile"; fi
fi

# 5. information only: which dlsym/dlopen version each interpreter binds to (RT-07)
echo "--- info: dlsym versions (the shim exports dlsym@@GLIBC_2.2.5 only)"
for b in /usr/bin/python3 /usr/local/bin/ollama; do
  [ -e "$b" ] && echo "$b: $(nm -D --undefined-only "$(readlink -f "$b")" 2>/dev/null | awk '/ dlsym@/{print $2}' | tr '\n' ' ')"
done
[ -e "$REPO/libgreenboost_cuda.so" ] && echo "shim exports: $(nm -D "$REPO/libgreenboost_cuda.so" | awk '/ T dlsym/{print $3}')"
exit $fail

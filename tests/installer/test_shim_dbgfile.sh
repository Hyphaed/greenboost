#!/usr/bin/env bash
# SP62-02b: the shim's per-pid debug file is gated by GREENBOOST_DBGFILE=1 (default /dev/null).
# The REAL gbdbg_fp text is cut out of greenboost_cuda_shim.c, its /tmp path is redirected into a sandbox,
# compiled with ASan and run with and without the variable. Nothing is created in /tmp, nothing deleted.
source "$(dirname "$0")/lib.sh"
SB="$(new_sandbox)"; echo "sandbox: $SB"; mkdir -p "$SB/out"
SHIM="$REPO/greenboost_cuda_shim.c"
fn="$(awk '/^static FILE \*gbdbg_fp\(void\)/{p=1} p{print} p&&/^}/{exit}' "$SHIM")"
check "gbdbg_fp found in the shim" test -n "$fn"
check "only one /tmp/gbdbg_ path in code (snprintf)" test "$(grep -c 'snprintf.*"/tmp/gbdbg_' "$SHIM")" = 1
check "gate reads GREENBOOST_DBGFILE" grep -q 'getenv("GREENBOOST_DBGFILE")' <<<"$fn"
{
  printf '#include <stdio.h>\n#include <stdlib.h>\n#include <unistd.h>\n#include <string.h>\n'
  printf '%s\n' "${fn//\/tmp\/gbdbg_/$SB/out/gbdbg_}"
  printf 'int main(void){ fprintf(gbdbg_fp(), "[GBDBG] probe\\n"); fprintf(gbdbg_fp(), "second\\n"); return 0; }\n'
} > "$SB/t.c"
if ! cc -std=gnu11 -Wall -Werror -fsanitize=address,undefined -g -o "$SB/t" "$SB/t.c" 2> "$SB/cc.err"; then
    cat "$SB/cc.err"; bad "compile with ASan"; summary; exit 1; fi
ok "compiles clean with -Wall -Werror and ASan/UBSan"
count() { find "$SB/out" -type f | wc -l; }
env -u GREENBOOST_DBGFILE "$SB/t"; check "unset: runs, no file created" test "$(count)" = 0
GREENBOOST_DBGFILE=0 "$SB/t";      check "=0: no file created" test "$(count)" = 0
GREENBOOST_DBGFILE= "$SB/t";       check "empty: no file created" test "$(count)" = 0
GREENBOOST_DBGFILE=1 "$SB/t";      check "=1: exactly one per-pid file" test "$(count)" = 1
f="$(find "$SB/out" -type f | head -1)"
check "=1: file holds both lines" test "$(wc -l < "$f")" = 2
summary

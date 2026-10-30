#!/usr/bin/env bash
# SP62-02b: ollama context-by-model patch. Server default 8192 in the drop-in, ladder value in ollama-ctx.conf,
# zz-model-slots.conf never touched by the conflict sweep. Fake root, no sudo, no systemctl, nothing deleted.
source "$(dirname "$0")/lib.sh"
SB="$(new_sandbox)"; echo "sandbox: $SB"; mkdir -p "$SB/etc/greenboost" "$SB/dropin"
gb_info() { :; }; gb_ok() { :; }
load_fn "$SETUP" gb_ollama_default_ctx gb_managed_write gb_backup_once || exit 2

unset GB_OLLAMA_EMBED_CTX GB_OLLAMA_SERVER_CTX GB_OLLAMA_CTX
check "no env: default 8192" test "$(gb_ollama_default_ctx)" = 8192
GB_OLLAMA_CTX=32768; check "ladder 32768: server default stays 8192 (embedding cap)" test "$(gb_ollama_default_ctx)" = 8192
GB_OLLAMA_CTX=4096;  check "ladder 4096 below the cap: default follows the ladder" test "$(gb_ollama_default_ctx)" = 4096
GB_OLLAMA_CTX=32768 GB_OLLAMA_SERVER_CTX=16384; check "GB_OLLAMA_SERVER_CTX overrides" test "$(gb_ollama_default_ctx)" = 16384
unset GB_OLLAMA_SERVER_CTX

# the real printf line that writes ollama-ctx.conf, pointed into the sandbox
GB_OLLAMA_CTX=32768
line="$(grep -F '| gb_managed_write /etc/greenboost/ollama-ctx.conf' "$SETUP")"
check "setup writes ollama-ctx.conf through gb_managed_write" test -n "$line"
# join the printf continuation (previous line) with the pipe line
stmt="$(grep -B1 -F '| gb_managed_write /etc/greenboost/ollama-ctx.conf' "$SETUP" | sed 's/\\$//' | tr '\n' ' ')"
eval "${stmt//\/etc\/greenboost/$SB/etc/greenboost}"
C="$SB/etc/greenboost/ollama-ctx.conf"
check "ollama-ctx.conf has chat=32768 (ladder)" grep -qx 'chat=32768' "$C"
check "ollama-ctx.conf has embedding=8192" grep -qx 'embedding=8192' "$C"
check "ollama-ctx.conf has default=8192" grep -qx 'default=8192' "$C"

# drop-in text: both server variables take the default function, never the ladder
check "drop-in OLLAMA_CONTEXT_LENGTH uses the default function" grep -q 'Environment="OLLAMA_CONTEXT_LENGTH=\$(gb_ollama_default_ctx)"' "$SETUP"
check "drop-in OLLAMA_NUM_CTX uses the default function" grep -q 'Environment="OLLAMA_NUM_CTX=\$(gb_ollama_default_ctx)"' "$SETUP"
check "drop-in never writes the raw ladder into OLLAMA_CONTEXT_LENGTH" test -z "$(grep -E 'Environment="OLLAMA_(CONTEXT_LENGTH|NUM_CTX)=\$\{?GB_OLLAMA_CTX' "$SETUP")"

# conflict sweep: run the REAL loop on a sandbox drop-in dir
loop="$(awk '/for _f in "\$dropin_dir"\/\*\.conf; do/{p=1} p{print} p&&/^    done$/{exit}' "$SETUP")"
check "conflict sweep loop found" test -n "$loop"
D="$SB/dropin"
printf '[Service]\nEnvironment="OLLAMA_NUM_CTX=2048"\nEnvironment="OLLAMA_MAX_LOADED_MODELS=1"\n' > "$D/zz-model-slots.conf"
printf '[Service]\nEnvironment="OLLAMA_NUM_CTX=2048"\nEnvironment="OLLAMA_CONTEXT_LENGTH=2048"\n' > "$D/boost.conf"
cp "$D/zz-model-slots.conf" "$SB/zz.before"
dropin_dir="$D"; _conflict_found=0
gb_vars=(OLLAMA_NUM_CTX OLLAMA_CONTEXT_LENGTH OLLAMA_MAX_LOADED_MODELS)
eval "$loop"
check "zz-model-slots.conf still exists" test -f "$D/zz-model-slots.conf"
check "zz-model-slots.conf byte-identical after the sweep" cmp -s "$D/zz-model-slots.conf" "$SB/zz.before"
check "other drop-in loses the conflicting variable" test -z "$(grep OLLAMA_NUM_CTX "$D/boost.conf")"
check "other drop-in backed up once (.gb-orig)" test -f "$D/boost.conf.gb-orig"
summary

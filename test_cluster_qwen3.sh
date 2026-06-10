#!/usr/bin/env bash
# test_cluster_qwen3.sh - Live cluster inference test for Qwen3-35B-A3B
#
# Exercises memory tier priority:
#   T1_local + T1_feeder → GPU compute local + GPU compute feeder → T2 → T3
#
# Usage:
#   bash test_cluster_qwen3.sh           # full test suite
#   bash test_cluster_qwen3.sh --unload  # force-unload model before starting

set -euo pipefail

MODEL="qwen3.6:latest"
OLLAMA_URL="http://localhost:11434"
FEEDER_IP="192.168.50.246"
FEEDER_PORT="9740"
# Override OLLAMA_CTX_SIZE env var to change context window (default 4096).
# OLLAMA_NUM_CTX=131072 in the systemd service would require ~17 GB KV cache,
# exhausting the 43 GB virtual VRAM; a small ctx keeps KV << 1 GB.
CTX_SIZE="${OLLAMA_CTX_SIZE:-4096}"

# ── UI ────────────────────────────────────────────────────────────────────────

C_RESET='\033[0m';  C_BOLD='\033[1m';    C_DIM='\033[2m'
C_CYAN='\033[36m';  C_LIME='\033[92m';   C_AMBER='\033[93m'
C_RED='\033[91m';   C_VIOLET='\033[35m'; C_GRAY='\033[90m'
C_BLUE='\033[94m';  C_WHITE='\033[97m'
DIV="───────────────────────────────────────────────────────────────────────"

gb_section() { printf "\n${C_BOLD}${C_CYAN}%s${C_RESET}\n${C_DIM}${DIV}${C_RESET}\n" "$1"; }
gb_ok()      { printf "  ${C_LIME}✓${C_RESET}  %s\n" "$1"; }
gb_info()    { printf "  ${C_CYAN}◈${C_RESET}  ${C_DIM}%s${C_RESET}\n" "$1"; }
gb_warn()    { printf "  ${C_AMBER}⚠${C_RESET}  %s\n" "$1"; }
gb_fail()    { printf "  ${C_RED}✗${C_RESET}  %s\n" "$1" >&2; }
gb_label()   { printf "  ${C_BOLD}%-26s${C_RESET}${C_WHITE}%s${C_RESET}\n" "$1" "$2"; }

# ── Tier snapshot helpers ─────────────────────────────────────────────────────

# Prints local T1/T2/T3 used/total via nvidia-smi + pool_brief
snap_local() {
    local t1_used=0 t1_total=0
    if command -v nvidia-smi &>/dev/null; then
        IFS=',' read -r t1_used t1_total < <(
            nvidia-smi --query-gpu=memory.used,memory.total \
                       --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
        ) || true
    fi

    local brief
    brief=$(cat /sys/class/greenboost/greenboost/pool_brief 2>/dev/null || echo "")
    local t2_used t2_total t3_used t3_total
    # pool_brief: "T1:11GB T2:0/42GB(0%) T3:0/73GB ..."
    t2_used=$(  echo "$brief" | grep -oP '\bT2:\K\d+' 2>/dev/null | head -1 || echo "?")
    t2_total=$( echo "$brief" | grep -oP 'T2:\d+/\K\d+' 2>/dev/null || echo "?")
    t3_used=$(  echo "$brief" | grep -oP 'T3:\K\d+' 2>/dev/null || echo "?")
    t3_total=$( echo "$brief" | grep -oP 'T3:\d+/\K\d+' 2>/dev/null || echo "?")

    local kv_rsv kv_t2
    kv_rsv=$(echo "$brief" | grep -oP 'KV_RSV:\K\S+' 2>/dev/null || echo "?")
    kv_t2=$( echo "$brief" | grep -oP 'KV_T2:\K\S+' 2>/dev/null || echo "?")

    printf "  ${C_BOLD}${C_VIOLET}Local  (ncore)${C_RESET}\n"
    printf "    ${C_CYAN}T1 VRAM ${C_RESET}  %s / %s MiB\n" "$t1_used" "$t1_total"
    printf "    ${C_CYAN}T2 DDR  ${C_RESET}  %s / %s GB used\n" "$t2_used" "$t2_total"
    printf "    ${C_CYAN}T3 NVMe ${C_RESET}  %s / %s GB used\n" "$t3_used" "$t3_total"
    printf "    ${C_DIM}KV reserve: %s   KV in T2: %s${C_RESET}\n" "$kv_rsv" "$kv_t2"
}

# Prints feeder T1/T2/T3 free/total via net_fabric GB_MSG_MEM_INFO
snap_feeder() {
    python3 - "$FEEDER_IP" "$FEEDER_PORT" <<'PYEOF'
import sys, socket, struct

ip, port = sys.argv[1], int(sys.argv[2])

C_RESET = "\033[0m"; C_BOLD = "\033[1m"; C_DIM = "\033[2m"
C_CYAN = "\033[36m"; C_VIOLET = "\033[35m"

GB_NET_MAGIC         = 0x47424E46
GB_NET_HDR_SIZE      = 12
GB_MSG_HANDSHAKE_REQ = 0x01
GB_MSG_MEM_INFO      = 0x31
GB_NET_MAX_HOSTNAME  = 64
GB_GPU_INFO_SIZE     = 100
GB_NET_MAX_GPUS      = 8

def recvall(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c: raise EOFError("connection closed")
        buf += c
    return buf

def send_msg(sock, msg_type, payload=b""):
    hdr = struct.pack("<IHHI", GB_NET_MAGIC, msg_type, 0, len(payload))
    sock.sendall(hdr + payload)

def recv_msg(sock):
    raw = recvall(sock, GB_NET_HDR_SIZE)
    magic, msg_type, flags, plen = struct.unpack("<IHHI", raw)
    return msg_type, flags, recvall(sock, plen) if plen else b""

def mb(b): return b >> 20

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    sock.connect((ip, port))

    hs = struct.pack("<II", 1, 0) + b"gb_monitor\x00".ljust(GB_NET_MAX_HOSTNAME, b"\x00") \
         + b"\x00" * (GB_NET_MAX_GPUS * GB_GPU_INFO_SIZE)
    send_msg(sock, GB_MSG_HANDSHAKE_REQ, hs)
    recv_msg(sock)

    send_msg(sock, GB_MSG_MEM_INFO, struct.pack("<II", 0, 0))
    _, _, resp = recv_msg(sock)
    sock.close()

    if len(resp) >= 72:
        (status, t2_spd, free_b, total_b,
         t1_free, t1_total, t2_free, t2_total, t3_free, t3_total,
         t3_spd, _pad) = struct.unpack_from("<IIQQQQQQQQIi", resp, 0)
    elif len(resp) >= 40:
        status, t2_spd, free_b, total_b, t1_free, t1_total, t2_free, t2_total = \
            struct.unpack_from("<IIQQQQqq", resp, 0)
        t3_free = t3_total = t3_spd = 0
    else:
        print("  Feeder MEM_INFO response too short")
        sys.exit(0)

    t1_used = mb(t1_total) - mb(t1_free)
    t2_used = mb(t2_total) - mb(t2_free)
    t3_used = mb(t3_total) - mb(t3_free)

    print(f"  {C_BOLD}{C_VIOLET}Feeder (omen){C_RESET}")
    print(f"    {C_CYAN}T1 VRAM {C_RESET}  {t1_used} / {mb(t1_total)} MB used")
    print(f"    {C_CYAN}T2 DDR  {C_RESET}  {t2_used} / {mb(t2_total)} MB used  [{t2_spd} MT/s]")
    print(f"    {C_CYAN}T3 NVMe {C_RESET}  {t3_used} / {mb(t3_total)} MB used  [{t3_spd} MB/s]")

except Exception as e:
    print(f"  Feeder query failed: {e}")
PYEOF
}

snap_all() {
    snap_local
    echo ""
    snap_feeder
}

# ── Ollama helpers ────────────────────────────────────────────────────────────

ollama_generate() {
    local prompt="$1" stream="${2:-false}" timeout="${3:-120}" np="${4:-512}"
    curl -s --max-time "$timeout" "$OLLAMA_URL/api/generate" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"prompt\":$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$prompt"),\"stream\":$stream,\"options\":{\"num_predict\":$np,\"num_ctx\":$CTX_SIZE}}"
}

ollama_chat() {
    local prompt="$1" timeout="${2:-180}"
    curl -s --max-time "$timeout" "$OLLAMA_URL/api/chat" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$prompt")}],\"stream\":false,\"options\":{\"num_predict\":512,\"num_ctx\":$CTX_SIZE}}"
}

model_unload() {
    curl -s "$OLLAMA_URL/api/generate" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"keep_alive\":\"0s\"}" >/dev/null 2>&1 || true
    sleep 2
}

# ── GB journal scraper ────────────────────────────────────────────────────────

show_gb_logs() {
    local since="$1"
    echo ""
    printf "  ${C_BOLD}GreenBoost shim log (since load):${C_RESET}\n"
    journalctl -u ollama --since "$since" --no-pager -q 2>/dev/null \
        | grep -i "GreenBoost\|feeder T1\|feeder T2\|feeder T3\|data-driven dispatch\|cluster\|fake=\|cudaMalloc\|overflow" \
        | tail -20 \
        | while IFS= read -r line; do printf "    ${C_DIM}%s${C_RESET}\n" "$line"; done \
        || true
}

# ── Preflight checks ──────────────────────────────────────────────────────────

gb_section "Preflight Checks"

# Ollama running
if ! curl -s --max-time 3 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    gb_fail "Ollama not responding at $OLLAMA_URL"
    exit 1
fi
gb_ok "Ollama is running"

# GreenBoost kernel module
if [[ -d /sys/class/greenboost/greenboost ]]; then
    gb_ok "GreenBoost kernel module loaded"
else
    gb_warn "GreenBoost sysfs not found - T2/T3 tier tracking unavailable"
fi

# Feeder reachable
if python3 -c "import socket; s=socket.socket(); s.settimeout(3); s.connect(('$FEEDER_IP',$FEEDER_PORT)); s.close()" 2>/dev/null; then
    gb_ok "Feeder reachable at $FEEDER_IP:$FEEDER_PORT"
else
    gb_fail "Cannot reach feeder at $FEEDER_IP:$FEEDER_PORT - run: sudo greenboost feeders upgrade-greenboost"
    exit 1
fi

# Model present
if curl -s "$OLLAMA_URL/api/tags" | python3 -c "
import sys,json
models=[m['name'] for m in json.load(sys.stdin).get('models',[])]
exit(0 if '$MODEL' in models else 1)
" 2>/dev/null; then
    gb_ok "Model found: $MODEL"
else
    gb_fail "Model not found in Ollama. Pull it first:"
    gb_info  "ollama pull $MODEL"
    exit 1
fi

# GreenBoost shim placement check - confirm oversize model weights use pinned DDR (Path A/B).
# The shim routes cudaMalloc overflow to T2 DDR via pinned paths (GPU DMA-accessible).
# Path C (UVM) has been removed; GGML_CUDA_NO_VMM=1 is no longer required but safe to leave.
SHIM_PATH="/usr/local/lib/libgreenboost_cuda.so"
if [[ -f "$SHIM_PATH" ]]; then
    SHIM_MTIME=$(stat -c '%Y' "$SHIM_PATH" 2>/dev/null || echo 0)
    if systemctl cat ollama 2>/dev/null | grep -q "GGML_CUDA_NO_VMM=1"; then
        gb_ok "GGML_CUDA_NO_VMM=1 set - VMM bypassed, using cudaMalloc → Path B (T2 DDR)"
    else
        gb_ok "GreenBoost shim present - UVM oversize uses CPU-preferred placement (T3 swap)"
    fi
else
    gb_warn "GreenBoost shim not found at $SHIM_PATH - LD_PRELOAD will fail"
fi

# Optional unload before test
if [[ "${1:-}" == "--unload" ]]; then
    gb_info "Unloading model (--unload flag)"
    model_unload
fi

# ── Baseline snapshot ─────────────────────────────────────────────────────────

gb_section "Tier State - Baseline (model not loaded)"
snap_all

# ── Model load ────────────────────────────────────────────────────────────────

gb_section "Loading Model into Cluster"
printf "  Model  : ${C_BOLD}%s${C_RESET}\n" "$MODEL"
printf "  Target : ${C_DIM}T1_local(11 GB) + T1_feeder(8 GB) → T2_local if overflow${C_RESET}\n\n"

LOAD_START=$(date +%s)
LOAD_SINCE=$(date --iso-8601=seconds)

printf "  Warming up model (first token)... "
LOAD_RESP=$(ollama_generate "Hello" "false" 180)
LOAD_END=$(date +%s)
LOAD_TIME=$(( LOAD_END - LOAD_START ))

LOAD_ERR=$(echo "$LOAD_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error',''))" 2>/dev/null || echo "")
if [[ -n "$LOAD_ERR" ]]; then
    printf "${C_RED}failed${C_RESET} (${LOAD_TIME}s)\n"
    gb_fail "Ollama error: $LOAD_ERR"
    echo ""
    printf "  ${C_BOLD}Ollama runner log (last 30 lines):${C_RESET}\n"
    journalctl -u ollama --since "$LOAD_SINCE" --no-pager -q -n 30 2>/dev/null \
        | grep -v "^\[GIN\]\|GreenBoost.*Cluster\|GreenBoost.*Network" \
        | while IFS= read -r line; do printf "    ${C_DIM}%s${C_RESET}\n" "$line"; done \
        || true
elif echo "$LOAD_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('response') else 1)" 2>/dev/null; then
    printf "${C_LIME}done${C_RESET} (${LOAD_TIME}s)\n"
else
    printf "${C_AMBER}slow/partial${C_RESET} (${LOAD_TIME}s)\n"
fi

# ── Post-load snapshot ────────────────────────────────────────────────────────

gb_section "Tier State - After Model Load"
snap_all
show_gb_logs "$LOAD_SINCE"

# Path-selection verdict from shim_stats.
# Path A (DMA-BUF pinned DDR) and Path B (HostReg no-kernel) are the only paths.
# Path C (UVM) has been removed - all overflow routes to GPU-DMA-accessible pinned DDR.
if [[ -f /run/greenboost/shim_stats ]]; then
    _PATH_A=$(grep "^path_a_count=" /run/greenboost/shim_stats 2>/dev/null | cut -d= -f2 || echo 0)
    _PATH_B=$(grep "^path_b_count=" /run/greenboost/shim_stats 2>/dev/null | cut -d= -f2 || echo 0)
    echo ""
    gb_label "GreenBoost Path A (T2 pinned DDR, DMA-BUF):" "${_PATH_A:-0} allocs"
    gb_label "GreenBoost Path B (T2 pinned DDR, HostReg):" "${_PATH_B:-0} allocs"
    if [[ $(( ${_PATH_A:-0} + ${_PATH_B:-0} )) -gt 0 ]]; then
        gb_ok "Model overflow routed to GPU-DMA-accessible pinned DDR - GPU compute active"
    else
        gb_info "No GreenBoost overflow detected - model may fit in T1 VRAM or use cuMemCreate VMM path"
    fi
fi

# ── Inference test 1 - short reasoning (MoE token throughput) ────────────────

gb_section "Test 1 - Short Reasoning  (MoE active-param throughput)"
PROMPT1="What is 1234567 × 9876543? Compute step by step. /nothink"

printf "  Prompt: %s\n\n" "$PROMPT1"
T1_SINCE=$(date --iso-8601=seconds)
T1_START=$(date +%s%N)

RESP1=$(ollama_generate "$PROMPT1" "false" 120)

T1_END=$(date +%s%N)
T1_MS=$(( (T1_END - T1_START) / 1000000 ))

OLLAMA_ERR1=$(echo "$RESP1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error',''))" 2>/dev/null || echo "")
EVAL_COUNT=$(echo "$RESP1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_count',0))" 2>/dev/null || echo 0)
EVAL_RATE=$(echo "$RESP1" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('eval_rate',0); print(f'{r:.1f}')" 2>/dev/null || echo "?")
RESPONSE1=$(echo "$RESP1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('response','[no response]')[:500])" 2>/dev/null || echo "[error]")
[[ -n "$OLLAMA_ERR1" ]] && gb_fail "Ollama error: $OLLAMA_ERR1"

printf "  %s\n\n" "$RESPONSE1"
gb_label "Tokens generated:"  "$EVAL_COUNT"
gb_label "Throughput:"        "$EVAL_RATE tok/s"
gb_label "Wall time:"         "${T1_MS} ms"

# ── Inference test 2 - long creative generation ───────────────────────────────

gb_section "Test 2 - Extended Generation  (feeder compute dispatch)"
PROMPT2="Write a detailed 400-word story about a guy who discovers a hidden island ruled by an ancient AI oracle. Include dialogue, action, and vivid setting descriptions. /nothink"

printf "  Prompt: %s\n\n" "$PROMPT2"
T2_SINCE=$(date --iso-8601=seconds)
T2_START=$(date +%s%N)

RESP2=$(ollama_generate "$PROMPT2" "false" 240 1024)

T2_END=$(date +%s%N)
T2_MS=$(( (T2_END - T2_START) / 1000000 ))

OLLAMA_ERR2=$(echo "$RESP2" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error',''))" 2>/dev/null || echo "")
EVAL_COUNT2=$(echo "$RESP2" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('eval_count',0))" 2>/dev/null || echo 0)
EVAL_RATE2=$(echo "$RESP2" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('eval_rate',0); print(f'{r:.1f}')" 2>/dev/null || echo "?")
RESPONSE2=$(echo "$RESP2" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('response','[no response]')[:800])" 2>/dev/null || echo "[error]")
[[ -n "$OLLAMA_ERR2" ]] && gb_fail "Ollama error: $OLLAMA_ERR2"

printf "  %s\n\n" "$RESPONSE2"
gb_label "Tokens generated:"  "$EVAL_COUNT2"
gb_label "Throughput:"        "$EVAL_RATE2 tok/s"
gb_label "Wall time:"         "${T2_MS} ms"

# ── Post-inference tier snapshot ─────────────────────────────────────────────

gb_section "Tier State - After Inference"
snap_all

# ── GreenBoost dispatch evidence ──────────────────────────────────────────────

gb_section "GreenBoost Cluster Evidence"
printf "  ${C_BOLD}Feeder T1 / compute dispatch log entries:${C_RESET}\n"
journalctl -u ollama --since "$LOAD_SINCE" --no-pager -q 2>/dev/null \
    | grep -iE "feeder T1|feeder T2|feeder T3|data-driven dispatch|cluster.*feeder|remote.*MB|fake=0x" \
    | tail -30 \
    | while IFS= read -r line; do printf "    ${C_DIM}%s${C_RESET}\n" "$line"; done ─\
    || true

printf "\n  ${C_BOLD}Feeder resource consumption delta:${C_RESET}\n"
snap_feeder

# ── Summary ───────────────────────────────────────────────────────────────────

gb_section "Summary"
printf "  %-30s %s\n" "Model" "$MODEL"
printf "  %-30s %s\n" "Load time" "${LOAD_TIME}s"
printf "\n"
printf "  %-30s %s tok/s  (%s tokens)\n" "Test 1 - Short reasoning" "$EVAL_RATE"  "$EVAL_COUNT"
printf "  %-30s %s tok/s  (%s tokens)\n" "Test 2 - Extended generation" "$EVAL_RATE2" "$EVAL_COUNT2"
printf "\n"
printf "  ${C_DIM}Tier priority expected: T1_local → T1_feeder → T2_local${C_RESET}\n"
printf "  ${C_DIM}Compute dispatch: local GPU first, feeder GPU for remote-pointer layers${C_RESET}\n"
printf "\n"
printf "  ${C_DIM}Check feeder compute logs:${C_RESET}\n"
printf "  ${C_DIM}  journalctl -u ollama | grep 'data-driven dispatch'${C_RESET}\n"
printf "  ${C_DIM}  journalctl -u ollama | grep 'feeder T1'${C_RESET}\n"
printf "  ${C_DIM}  ssh %s 'tail -40 /var/log/greenboost/netd.log'${C_RESET}\n" "$FEEDER_IP"

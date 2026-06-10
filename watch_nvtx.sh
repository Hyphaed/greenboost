#!/usr/bin/env bash
# watch_nvtx.sh - Real-time GreenBoost NVTX vital-signal monitor
#
# Shows a live merged stream of local shim + feeder daemon NVTX events,
# colour-coded by tier and event type. Both logs use the same format:
#
#   epoch_ms  SOURCE  EVENT_TYPE  TIER  SIZE  ptr=0xADDR  detail
#
# Usage:
#   bash watch_nvtx.sh                     # follow mode (live tail)
#   bash watch_nvtx.sh --last 200          # show last N events and exit
#   bash watch_nvtx.sh --filter EXEC,MEMCPY  # show only matching event types
#   bash watch_nvtx.sh --feeder-only       # feeder events only
#   bash watch_nvtx.sh --local-only        # local shim events only

set -euo pipefail

LOCAL_LOG="/run/greenboost/nvtx_events.log"
FEEDER_IP="192.168.50.246"
FEEDER_LOG="/run/greenboost/nvtx_events.log"

FOLLOW=1
LAST=0
FILTER=""
FEEDER_ONLY=0
LOCAL_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --last)        FOLLOW=0; LAST="${2:-100}"; shift 2 ;;
        --filter)      FILTER="$2"; shift 2 ;;
        --feeder-only) FEEDER_ONLY=1; shift ;;
        --local-only)  LOCAL_ONLY=1; shift ;;
        -h|--help)
            sed -n '3,12p' "$0" | sed 's/^# //'
            exit 0 ;;
        *) shift ;;
    esac
done

# ── Python formatter ──────────────────────────────────────────────────────────
# Reads lines from stdin and writes colorised output.
# Each line is prefixed with a SOURCE tag injected by the bash driver below.
python3 - "$FILTER" <<'PYEOF'
import sys, time, re, os

filter_str = sys.argv[1] if len(sys.argv) > 1 else ""
filters = [f.strip().upper() for f in filter_str.split(",") if f.strip()] if filter_str else []

ESC = "\033["
R   = ESC + "0m"
DIM = ESC + "2m"
BOLD= ESC + "1m"

COLORS = {
    # allocation events
    "ALLOC_T1_GPU":    ESC + "92m",   # bright green
    "ALLOC_T2_DDR":    ESC + "94m",   # bright blue
    "ALLOC_T2_POOL":   ESC + "34m",   # blue (dim)
    "ALLOC_T3_MMAP":   ESC + "36m",   # cyan
    # free events
    "FREE_OK":         ESC + "90m",   # dark grey
    # OOM
    "OOM_T1_GPU":      ESC + "91m",   # bright red
    "OOM_T2_DDR":      ESC + "91m",
    "OOM_T3":          ESC + "91m",
    # compute / memcpy
    "EXEC_KERNEL":     ESC + "93m",   # yellow
    "EXEC_DISPATCH":   ESC + "93m",
    "EXEC_LOCAL":      ESC + "33m",   # amber
    "MEMCPY_H2D":      ESC + "96m",   # bright cyan
    "MEMCPY_D2H":      ESC + "96m",
    "MEMCPY_D2D":      ESC + "96m",
    # path / pool
    "PATH_A0_FAIL":    ESC + "91m",   # red
    "PATH_A0_OK":      ESC + "32m",
    "PATH_A_FAIL":     ESC + "91m",
    "PATH_B_OK":       ESC + "32m",
    # network / client
    "CLIENT_CONN":     ESC + "35m",   # violet
    "CLIENT_DISC":     ESC + "90m",   # grey
    # phases / misc
    "PHASE_DEEP_IDLE": ESC + "90m",
    "HANDSHAKE":       ESC + "35m",
}

TIER_COL = {
    "T1_GPU":      ESC + "92;1m",
    "T2_DDR":      ESC + "94;1m",
    "T3_NVMe":     ESC + "36;1m",
    "NET":         ESC + "35;1m",
    "PHASE":       ESC + "90m",
    "T2_DDR_POOL": ESC + "34;1m",
}

SOURCE_COL = {
    "LOCAL":  ESC + "97;1m",    # white bold
    "FEEDER": ESC + "95;1m",    # magenta bold
    "NETD":   ESC + "95;1m",    # treat NETD as feeder origin
}

def fmt_epoch(ms):
    t = float(ms) / 1000.0
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(ms)%1000:03d}"

def fmt_size(s):
    try:
        n = int(s.replace("MB","").strip())
        if n == 0:    return DIM + "    -   " + R
        if n >= 1024: return ESC + "93m" + f"{n:5d}MB" + R
        return f"{n:5d}MB"
    except Exception:
        return s

DIV_LOCAL  = f"{ESC}97;1m{'─'*4} LOCAL  {'─'*4}{R}"
DIV_FEEDER = f"{ESC}95;1m{'─'*4} FEEDER {'─'*4}{R}"

last_source = None

for raw in sys.stdin:
    raw = raw.rstrip("\n")
    if not raw:
        continue

    # Lines are prefixed "LOCAL:" or "FEEDER:" by the bash driver
    if raw.startswith("LOCAL:"):
        source = "LOCAL"
        line = raw[6:]
    elif raw.startswith("FEEDER:"):
        source = "FEEDER"
        line = raw[7:]
    else:
        line = raw
        source = "?"

    # Parse NVTX line format:
    #   epoch_ms  [SOURCE]  EVENT_TYPE  TIER  SIZE  ptr=0xADDR  detail
    # The [SOURCE] field (NETD/SHIM) may or may not be present.
    parts = line.split()
    if len(parts) < 4:
        continue

    idx = 0
    try:
        epoch_ms = parts[idx]; idx += 1
        int(epoch_ms)  # must be numeric
    except ValueError:
        continue

    # Optional source tag (NETD, SHIM)
    if idx < len(parts) and parts[idx] in ("NETD", "SHIM", "GB"):
        idx += 1

    if idx >= len(parts): continue
    event   = parts[idx]; idx += 1
    tier    = parts[idx] if idx < len(parts) else "?"    ; idx += 1
    size    = parts[idx] if idx < len(parts) else "0MB"  ; idx += 1
    ptr     = parts[idx] if idx < len(parts) else ""     ; idx += 1
    detail  = " ".join(parts[idx:]) if idx < len(parts) else ""

    # Apply filter
    if filters and not any(f in event.upper() for f in filters):
        continue

    ev_col   = COLORS.get(event, ESC + "37m")
    tier_col = TIER_COL.get(tier, ESC + "37m")
    src_col  = SOURCE_COL.get(source, "")

    ts = fmt_epoch(epoch_ms)
    sz = fmt_size(size)

    # Source separator on change
    if source != last_source:
        sep = DIV_FEEDER if source in ("FEEDER","NETD") else DIV_LOCAL
        print(f"\n{sep}")
        last_source = source

    print(
        f"  {DIM}{ts}{R}  "
        f"{src_col}{source:<6}{R}  "
        f"{ev_col}{event:<22}{R}  "
        f"{tier_col}{tier:<10}{R}  "
        f"{sz}  "
        f"{DIM}{ptr:<20}{R}  "
        f"{DIM}{detail}{R}"
    )
    sys.stdout.flush()
PYEOF
exit 0
PYEOF

# ── Header ────────────────────────────────────────────────────────────────────
printf "\033[1m\033[36mGreenBoost NVTX Vital-Signal Monitor\033[0m"
[[ -n "$FILTER" ]] && printf "  \033[93mfilter: %s\033[0m" "$FILTER"
printf "\n\033[2m%s\033[0m\n\n" "$(date '+%Y-%m-%d %H:%M:%S')  local=$LOCAL_LOG  feeder=$FEEDER_IP:$FEEDER_LOG"

printf "\033[2m  Event colour key:\033[0m\n"
printf "  \033[92mALLOC_T1_GPU\033[0m  \033[94mALLOC_T2_DDR\033[0m  \033[36mALLOC_T3\033[0m  "
printf "\033[91mOOM\033[0m  \033[93mEXEC\033[0m  \033[96mMEMCPY\033[0m  \033[90mFREE / IDLE\033[0m\n"
printf "\033[2m%s\033[0m\n\n" "────────────────────────────────────────────────────────────────────────────"

# ── Build tail commands ───────────────────────────────────────────────────────
# Each line is prefixed "LOCAL:" or "FEEDER:" before being piped to Python.

TMPDIR_NVTX=$(mktemp -d /tmp/gb_nvtx.XXXXXX)
trap 'rm -rf "$TMPDIR_NVTX"; kill 0 2>/dev/null' EXIT INT TERM

LOCAL_PIPE="$TMPDIR_NVTX/local.pipe"
FEEDER_PIPE="$TMPDIR_NVTX/feeder.pipe"
mkfifo "$LOCAL_PIPE" "$FEEDER_PIPE"

# ── Local log tail ────────────────────────────────────────────────────────────
if [[ "$FEEDER_ONLY" -eq 0 ]]; then
    if [[ -f "$LOCAL_LOG" ]]; then
        if [[ "$FOLLOW" -eq 1 ]]; then
            tail -n 0 -F "$LOCAL_LOG" 2>/dev/null | sed 's/^/LOCAL:/' > "$LOCAL_PIPE" &
        else
            tail -n "$LAST" "$LOCAL_LOG" 2>/dev/null | sed 's/^/LOCAL:/' > "$LOCAL_PIPE" &
        fi
    else
        printf "LOCAL:# local NVTX log not found: %s\n" "$LOCAL_LOG" > "$LOCAL_PIPE" &
    fi
else
    (while true; do sleep 60; done) > "$LOCAL_PIPE" &
fi

# ── Feeder log tail ───────────────────────────────────────────────────────────
if [[ "$LOCAL_ONLY" -eq 0 ]]; then
    if ssh -o BatchMode=yes -o ConnectTimeout=3 -o StrictHostKeyChecking=no \
            "ferran@$FEEDER_IP" "test -f '$FEEDER_LOG'" 2>/dev/null; then
        if [[ "$FOLLOW" -eq 1 ]]; then
            ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
                "ferran@$FEEDER_IP" \
                "tail -n 0 -F '$FEEDER_LOG' 2>/dev/null" \
                | sed 's/^/FEEDER:/' > "$FEEDER_PIPE" &
        else
            ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
                "ferran@$FEEDER_IP" \
                "tail -n '$LAST' '$FEEDER_LOG' 2>/dev/null" \
                | sed 's/^/FEEDER:/' > "$FEEDER_PIPE" &
        fi
    else
        printf "FEEDER:# feeder NVTX log unreachable\n" > "$FEEDER_PIPE" &
    fi
else
    (while true; do sleep 60; done) > "$FEEDER_PIPE" &
fi

# ── Merge and feed to formatter ───────────────────────────────────────────────
# Interleave both pipes using a background cat to the formatter.
# We use `cat` on named pipes in a subshell to merge streams.
(
    cat "$LOCAL_PIPE" &
    cat "$FEEDER_PIPE" &
    wait
) | python3 -u - "$FILTER" <<'PYEOF'
import sys, time, re

filter_str = sys.argv[1] if len(sys.argv) > 1 else ""
filters = [f.strip().upper() for f in filter_str.split(",") if f.strip()] if filter_str else []

ESC = "\033["
R   = ESC + "0m"
DIM = ESC + "2m"
BOLD= ESC + "1m"

COLORS = {
    "ALLOC_T1_GPU":    ESC + "92m",
    "ALLOC_T2_DDR":    ESC + "94m",
    "ALLOC_T2_POOL":   ESC + "34m",
    "ALLOC_T3_MMAP":   ESC + "36m",
    "FREE_OK":         ESC + "90m",
    "OOM_T1_GPU":      ESC + "91m",
    "OOM_T2_DDR":      ESC + "91m",
    "OOM_T3":          ESC + "91m",
    "EXEC_KERNEL":     ESC + "93m",
    "EXEC_DISPATCH":   ESC + "93m",
    "EXEC_LOCAL":      ESC + "33m",
    "MEMCPY_H2D":      ESC + "96m",
    "MEMCPY_D2H":      ESC + "96m",
    "MEMCPY_D2D":      ESC + "96m",
    "PATH_A0_FAIL":    ESC + "91;1m",
    "PATH_A0_OK":      ESC + "32m",
    "PATH_A_FAIL":     ESC + "91m",
    "PATH_B_OK":       ESC + "32m",
    "CLIENT_CONN":     ESC + "35m",
    "CLIENT_DISC":     ESC + "90m",
    "PHASE_DEEP_IDLE": ESC + "90m",
    "HANDSHAKE":       ESC + "35m",
}

TIER_COL = {
    "T1_GPU":      ESC + "92;1m",
    "T2_DDR":      ESC + "94;1m",
    "T3_NVMe":     ESC + "36;1m",
    "NET":         ESC + "35;1m",
    "PHASE":       ESC + "90m",
    "T2_DDR_POOL": ESC + "34;1m",
}

SOURCE_COL = {
    "LOCAL":  ESC + "97;1m",
    "FEEDER": ESC + "95;1m",
    "NETD":   ESC + "95;1m",
}

def fmt_epoch(ms):
    t = float(ms) / 1000.0
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(ms)%1000:03d}"

def fmt_size(s):
    try:
        n = int(s.replace("MB","").strip())
        if n == 0:    return DIM + "    -   " + R
        if n >= 1024: return ESC + "93m" + f"{n:5d}MB" + R
        return f"{n:5d}MB"
    except Exception:
        return f"{s:>7}"

counts = {"LOCAL": {}, "FEEDER": {}}

for raw in sys.stdin:
    raw = raw.rstrip("\n")
    if not raw:
        continue

    if raw.startswith("LOCAL:"):
        source = "LOCAL"; line = raw[6:]
    elif raw.startswith("FEEDER:"):
        source = "FEEDER"; line = raw[7:]
    else:
        source = "?"; line = raw

    parts = line.split()
    if len(parts) < 4:
        continue

    idx = 0
    try:
        epoch_ms = parts[idx]; idx += 1
        int(epoch_ms)
    except ValueError:
        continue

    if idx < len(parts) and parts[idx] in ("NETD", "SHIM", "GB"):
        idx += 1

    if idx >= len(parts): continue
    event  = parts[idx]; idx += 1
    tier   = parts[idx] if idx < len(parts) else "?"   ; idx += 1
    size   = parts[idx] if idx < len(parts) else "0MB" ; idx += 1
    ptr    = parts[idx] if idx < len(parts) else ""    ; idx += 1
    detail = " ".join(parts[idx:]) if idx < len(parts) else ""

    if filters and not any(f in event.upper() for f in filters):
        continue

    ev_col   = COLORS.get(event, ESC + "37m")
    tier_col = TIER_COL.get(tier, ESC + "37m")
    src_col  = SOURCE_COL.get(source, "")

    # Track counts per source/event
    counts.setdefault(source, {})
    counts[source][event] = counts[source].get(event, 0) + 1

    ts = fmt_epoch(epoch_ms)
    sz = fmt_size(size)

    src_badge = f"{src_col}{'▶ FEEDER' if source == 'FEEDER' else '◀ LOCAL ':8}{R}"

    print(
        f"  {DIM}{ts}{R}  "
        f"{src_badge}  "
        f"{ev_col}{event:<22}{R}  "
        f"{tier_col}{tier:<12}{R}  "
        f"{sz}  "
        f"{DIM}{ptr:<22}{R}  "
        f"{DIM}{detail}{R}"
    )
    sys.stdout.flush()
PYEOF

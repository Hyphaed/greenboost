# GreenBoost Commands

## Core commands

- `greenboost help` ; Show this command reference 
- `greenboost clear memory-pool` ; Force-release T1 VRAM + T2 RAM + T3 immediately by unloading all inference models 
- `sudo greenboost load` ; Load the GreenBoost kernel module with auto-detected parameters 
- `sudo greenboost unload` ; Unload the kernel module (does not remove installed files) 
- `greenboost run <app>` ; Force-activate shim on environments that do not use the kernel module, under virtualization environments 

---

## Network cluster (distributed GPU)

Pool GPU compute + memory across multiple machines on the local network.
The **feeder** machine exposes its GPU; the **host** machine connects and uses it.

- `sudo greenboost feed` ; Start the feeder daemon, exposes local GPU(s) to the network (port 9740) 
- `sudo greenboost feed stop` ; Stop the feeder daemon 
- `sudo greenboost feed fg` ; Run feeder in foreground (for debugging) 
- `sudo greenboost connect <IP>` ; Connect to a feeder machine and add it to the cluster 
- `sudo greenboost disconnect <IP>` ; Remove a feeder from the cluster 
- `greenboost diag feeder` - run all feeder checks
- `greenboost diag feeder-t1` - T1 VRAM alloc/free test
- `greenboost diag feeder-t2` - T2 DDR alloc/free test
- `greenboost diag feeder-t3` - T3 NVMe alloc/free test
- `greenboost diag feeder-compute` - kernel dispatch test (send `GB_MSG_CUDA_EXEC` and check response)
- `greenboost cluster` ; Show cluster status: all connected GPUs, VRAM, and connection state 
- `sudo greenboost feeders setup-sudo` ; One-time: grant NOPASSWD sudo on each feeder so upgrade-greenboost can run unattended
- `sudo greenboost feeders upgrade-greenboost` ; Push full GreenBoost update (netd + shim + setup script + build stamp) to all configured feeders and restart their daemon. Run after building a new version on the host.
- `greenboost feeders diag [t1|t2|t3|compute|all]` ; Run T1/T2/T3 alloc + compute diagnostic against each feeder via net_fabric protocol; wraps gb_feeder_diag.py
- `greenboost built-stamp` ; Show local build stamp (BUILD_ID, version, git hash, build date)
- `greenboost built-stamp --feeders` ; Show build stamp on local host AND each connected feeder side-by-side; highlights mismatched stamps in amber

---

## Diagnostics
- `greenboost vitals` ; Live TUI: T1/T2/T3 pool gauges, GPU metrics, cluster status, pressure alerts (5s refresh)
- `greenboost status` ; Show cuda memory pool (T1 VRAM / T2 RAM / T3 NVMe), module state, and system health 
- `greenboost benchmark` ; cuda memory pool bandwidth benchmark (T1 VRAM / T2 DDR / T3 NVMe) 
- `greenboost inference-test [--model NAME] [--llm]` ; End-to-end inference benchmark - verifies fastest compute path (A/B) and measures tok/s
- `greenboost logs [--llm]` ; Aggregated log snapshot: kernel module events, services, AppArmor denials (TUI loop when interactive)
- `greenboost inference-logs [--llm]` ; Show inference service logs (Ollama, idle-reclaim, shader-boost) 
- `greenboost nvtx-logs [--llm]` ; Live NVTX event log - allocation events, phase transitions, OOM events
- `greenboost nvtx vitals [--last N] [--filter EV] [--feeder-only] [--local-only] [--llm]` ; NVTX event timeline view
- `greenboost health-check [--llm]` ; One-shot comprehensive cluster health audit (module, shim, NVML, feeder handshakes)
- `greenboost build-info [--llm]` ; Show build metadata (version, git hash, build date, CUDA version)
- `sudo greenboost clear logs` ; Clear all GreenBoost log sources for a fresh diagnostic baseline (dmesg, journal, log files) 
- `sudo greenboost clear inference-logs` ; Clear inference service logs only 
- `sudo greenboost clear nvtx-logs` ; Rotate NVTX event log (rename current → .1, start fresh)
- `greenboost diag all` ; Run full local diagnostic suite (T1/T2/T3 alloc tests + compute)

---

## TurboQuant K/V compression

- `sudo greenboost turboquant on` ; Enable TurboQuant globally: sets OLLAMA_KV_CACHE_TYPE=q4_0 and GREENBOOST_TURBOQUANT=1, restarts Ollama
- `sudo greenboost turboquant off` ; Disable TurboQuant: reverts KV cache to q8_0
- `greenboost turboquant status` ; Show current TurboQuant state (flag file + Ollama drop-in)
- `greenboost turboquant --llm` ; Machine-readable TurboQuant state for scripts/AI tools

---

## Inference configuration

- `greenboost gen-inference-config` ; Auto-generate optimized Ollama env vars for the current hardware (virt type, T1/T2/T3, TurboQuant)
- `greenboost gen-inference-config --format env` ; Emit systemd-compatible Environment= lines instead of KEY=VALUE
- `greenboost gen-inference-config --format both` ; Emit both Ollama env vars and systemd drop-in lines
- `greenboost gen-inference-config --turboquant` ; Force TurboQuant on in the generated config
- `greenboost gen-inference-config --output FILE` ; Write config to FILE instead of stdout
- `greenboost gen-inference-config --llm` ; Machine-readable key=value output (no ANSI decoration)

---

## Live debug signals

GreenBoost exposes three layers of live telemetry readable at any time, even mid-inference.

### 1. sysfs kernel interface (always available when module is loaded)

```bash
# One-liner pool summary (T1/T2/T3 + pressure) - ideal for watch
watch -n2 'cat /sys/class/greenboost/greenboost/pool_brief'
# → T1:11GB T2:23/42GB(56%) T3:0/73GB PRESSURE:ok KV_RSV:0MB KV_T2:0MB

# Full tier status (T1/T2/T3 sizes, KV cache placement, T3 pressure)
watch -n2 'cat /sys/class/greenboost/greenboost/status'

# Hardware topology (CPU/GPU/RAM/NVMe, NUMA, kthread affinity)
cat /sys/class/greenboost/greenboost/hw_info

# Active DMA-BUF object count (number of live tensor handles)
cat /sys/class/greenboost/greenboost/active_buffers

# KV cache reserve in MB (0 = all T2 free for weights)
cat /sys/class/greenboost/greenboost/kv_reserve_mb

# Writeable parameters (require sudo tee)
echo 0 | sudo tee /sys/module/greenboost/parameters/kv_reserve_mb   # set KV reserve
```

### 2. `/run/greenboost/` runtime files (written by the LD_PRELOAD shim)

These files are created when a process runs with `LD_PRELOAD=libgreenboost_cuda.so`.
> **Note:** If a previous process (e.g. Ollama) wrote these files, they may be owned by that user. Run `rm -f /run/greenboost/shim_stats /run/greenboost/metrics.json` before starting a new generation so the shim can create fresh, writable files.

| File | Format | What it shows |
|------|--------|---------------|
| `shim_stats` | `key=value` | Full shim counters: tier allocations, kernel dispatch count, feeder alloc MB, T2 frag %, path A/B/C counts, phase |
| `metrics.json` | JSON | Same data structured as JSON with a `feeders[]` array per connected feeder |
| `nvtx_events.log` | TSV | Every allocation, phase change, OOM event with timestamp/tier/MB/pointer |
| `phase` | plain text | Current phase: `MODEL_LOAD` / `INFERENCE` / `STEADY` / `OOM_RECOVERY` |

**Key fields to watch in `shim_stats`:**

```bash
watch -n2 'grep -E "phase|tier_t1_feeder_cur|tier_t2_feeder_cur|kernel_dispatch|remote_alloc|t2_pool_frag|t2_above_warn|vram_headroom" /run/greenboost/shim_stats'
```

| Field | Meaning | Healthy value |
|-------|---------|---------------|
| `phase` | Memory manager phase | `STEADY` or `INFERENCE` |
| `tier_t1_feeder_cur_mb` | Feeder T1 VRAM in use | > 0 when feeder is pooling memory |
| `kernel_dispatch_count` | CUDA kernels dispatched to feeder | > 0 when feeder compute is active |
| `remote_alloc_count` | Allocations placed on feeder | > 0 when feeder T1 is used |
| `t2_pool_frag_pct` | T2 DDR fragmentation | < 30 (> 80 = pressure) |
| `t2_above_warn` | T2 pool above 75% threshold | 0 (1 = warn) |
| `vram_headroom_mb` | Free T1 VRAM headroom | > 200 |
| `path_a_count` | Allocations via pinned DDR (Path A - with kernel module) | increases during inference |
| `path_b_count` | Allocations via pinned DDR (Path B - no kernel module) | 0 on bare-metal |

**Check feeder health from metrics.json:**

```bash
python3 -c "
import json
d = json.load(open('/run/greenboost/metrics.json'))
print('Phase:', d.get('phase'))
print('Feeders:', d.get('feeders', []))
print('Kernel dispatches:', d.get('kernel_dispatch_count'))
print('Feeder T1 cur:', d['tiers']['t1_feeder']['cur_mb'], 'MB')
"
```

### 3. NVTX event log (chronological allocation trace)

```bash
# Live tail - see allocations and phase transitions as they happen
tail -f /run/greenboost/nvtx_events.log

# Same via greenboost CLI (interactive, auto-scrolling)
greenboost nvtx-logs

# Clear log for a fresh session baseline
sudo greenboost clear nvtx-logs
```

**Event format:** `<epoch_ns>  <EVENT_TYPE>  <TIER>  <MB>  ptr=<hex>  <message>`

Key event types: `SHIM_INIT`, `PHASE_MODEL_LOAD`, `PHASE_INFERENCE`, `ALLOC_T1_VRAM`, `ALLOC_T2_POOL`, `ALLOC_T3_UVM`, `FREE_*`, `OOM_EVICT`, `FEEDER_ALLOC`, `KERNEL_DISPATCH`

### 4. Web dashboard (greenboost-cli)

```bash
# Start the web dashboard (served at http://localhost:7821)
greenboost-cli   # then navigate to /greenboost

# Or run directly:
python -m greenboost_cli  # serves localhost:7821/greenboost
```

The `/greenboost` page shows:
- **SHIM STATISTICS** - live poll of `shim_stats` every 2 s: path counters, tier allocations (local + feeder), kernel dispatch count, T2 fragmentation, T1 headroom
- **FEEDER VITALS** - per-feeder T1/T2/T3 current and peak, remote alloc count, connection status
- **NVTX EVENT LOG** - scrollable log of the last N NVTX events

---

## Debug vitals (toggleable deep diagnostics)

GreenBoost debug vitals aggregate **9 data sources** into a single comprehensive view
for troubleshooting, performance analysis, and enhancement work.

```bash
# Show full vitals dump (always available, regardless of flag state)
greenboost debug vitals

# Enable persistent vitals - restarts services, sets kernel debug_mode=1, exports
# GREENBOOST_NVTX_VERBOSE=1 via /etc/profile.d/greenboost-vitals.sh
sudo greenboost debug vitals on

# Disable persistent vitals - reverts debug_mode to 0, removes profile.d, restarts services
sudo greenboost debug vitals off
```

**What "vitals on" activates:**

| Change | Mechanism | Requires restart? |
|--------|-----------|-------------------|
| Kernel verbose dmesg | `debug_mode=1` via sysfs | No - live |
| Verbose NVTX shim events | `GREENBOOST_NVTX_VERBOSE=1` in profile.d | New processes only |
| Extended vitals panel in scripts | `/etc/greenboost/debug_vitals.enabled` flag | No |
| greenboost-idle-reclaim | `systemctl restart` | Auto |
| Other GB-aware services | Listed on screen | Manual |

If `LD_PRELOAD` is not global (no `/etc/ld.so.preload` entry), a reboot warning is shown.

**Data sources exposed by `greenboost debug vitals`:**

| Source | What it shows |
|--------|--------------|
| IOCTL `GB_IOCTL_GET_INFO` | T1/T2/T3/KV/pressure/phase_reset_seq (24 fields) |
| `/run/greenboost/shim_stats` | Path A0/A/B/C counts, H2D/D2H MB, kernel_dispatch, T2 frag%, headroom, KV dedup, remote allocs |
| `/run/greenboost/phase` | Current phase (MODEL_LOAD / INFERENCE / STEADY / OOM_RECOVERY) |
| `/run/greenboost/nvtx_events.log` | Last 10 allocation/phase/OOM events with timestamps |
| `/run/greenboost/metrics.json` | JSON structured view + feeder T1/T2/T3 + BW |
| `/sys/class/greenboost/greenboost/` | pool_brief, status, active_buffers, active_profile |
| `/sys/module/greenboost/parameters/debug_mode` | Verbose dmesg state |
| `nvidia-smi` extended | Temp, power, power limit, GPU util%, mem util%, SM clock, mem clock |
| `journalctl -k --grep=greenboost` | Kernel module events (when debug_mode=1) |

**From greenboost-cli (Python):**

```bash
/gb-vitals          # Full Rich-rendered vitals in the CLI
/gb-vitals on       # Enable (same as sudo greenboost debug vitals on)
/gb-vitals off      # Disable
```

---

## Profile management

Hardware profiles store auto-detected parameters in `/etc/greenboost/profiles/`.
The active profile drives module parameters (`virtual_vram_gb`, `safety_reserve_gb`, etc.).

Running `greenboost profile` (no sub-command) opens the **interactive wizard** with a live
hardware panel and a numbered menu. Individual sub-commands are also available directly:


- `greenboost profile` ; Open Interactive wizard: create, activate, diff profiles 
- `sudo greenboost profile create` ; Auto-detect hardware and create a new profile 
- `greenboost profile list` ; List all available profiles 
- `greenboost profile show` ; Show the currently active profile 
- `greenboost profile show <file>` ; Show a specific profile file 
- `sudo greenboost profile activate <file>` ; Switch active profile 
- `greenboost profile diff <file>` ; Cross-check a profile against live hardware 

---

## Maintenance


- `sudo greenboost install-sys-configs` ; (Re-)install Ollama env, NVMe udev rules, CPU governor, hugepages, LD_AUDIT, idle-reclaim daemon 
- `sudo greenboost recover` ; Attempt automatic recovery after a failed install or module load error 

---

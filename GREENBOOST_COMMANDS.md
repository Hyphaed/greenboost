# GreenBoost Commands

*Full CLI reference for GreenBoost , the CUDA Memory & Compute Orchestrator
(memory tiering · gb-quant compression · cluster mode) , v3.2. See
[README.md](README.md) for install/quick-start and [CHANGELOG.md](CHANGELOG.md)
for what's new.*

Every command below also accepts `--llm` where noted, which strips ANSI
color codes and prints a compact `key=value` line , built for feeding a
command's output straight to an AI assistant or a log pipeline.

## Contents

- [🧩 Core commands](#-core-commands)
- [🌐 Network cluster (distributed GPU)](#-network-cluster-distributed-gpu)
- [🩺 Diagnostics](#-diagnostics)
- [📦 TurboQuant K/V compression](#-turboquant-kv-compression)
- [🖥️ ggml-2dev (feeder as CUDA device 1)](#️-ggml-2dev-feeder-as-cuda-device-1)
- [🧠 gb-synapse (HuggingFace-native, cluster-distributed GGUF serving)](#-gb-synapse-huggingface-native-cluster-distributed-gguf-serving)
- [⚡ Inference configuration](#-inference-configuration)
- [📡 Live debug signals](#-live-debug-signals)
- [🫀 Debug vitals (toggleable deep diagnostics)](#-debug-vitals-toggleable-deep-diagnostics)
- [📁 Profile management](#-profile-management)
- [🧰 Maintenance](#-maintenance)

---

## 🧩 Core commands

| Command | Description |
|---|---|
| `greenboost help` | Show this command reference |
| `greenboost clear memory-pool` | Force-release T1 VRAM + T2 RAM + T3 immediately by unloading all inference models |
| `sudo greenboost load` | Load the GreenBoost kernel module with auto-detected parameters |
| `sudo greenboost unload` | Unload the kernel module (does not remove installed files) |
| `greenboost run <app>` | Force-activate the shim on environments that don't use the kernel module, e.g. under virtualization |

---

## 🌐 Network cluster (distributed GPU)

Pool GPU memory , and, for basic ggml kernels, compute , across machines on
the local network. The **feeder** machine exposes its GPU; the **host**
connects to it and folds it into its own virtual device. See
[README.md § Cluster mode](README.md#cluster-mode---put-a-second-machines-gpu-to-work)
for what this actually delivers today.

| Command | Description |
|---|---|
| `sudo greenboost feed` | Start the feeder daemon, exposing local GPU(s) to the network (port 9740) |
| `sudo greenboost feed stop` | Stop the feeder daemon |
| `sudo greenboost feed fg` | Run the feeder daemon in the foreground (for debugging) |
| `sudo greenboost connect <IP>` | Connect to a feeder machine and add it to the cluster |
| `sudo greenboost disconnect <IP>` | Remove a feeder from the cluster |
| `greenboost cluster` | Show cluster status: all connected GPUs, VRAM, and connection state |
| `sudo greenboost feeders setup-sudo` | One-time: grant NOPASSWD sudo on each feeder so `update feeders` can run unattended |
| `sudo greenboost update feeders` (alias: `update-feeders`) | Push a full GreenBoost update (netd + shim + setup script + build stamp) to every configured feeder and restart its daemon. Run after building a new version on the host |
| `sudo greenboost feeders sync-ollama` | Install the host's exact Ollama version on every feeder , kernel-name parity is required for feeder GPU compute to resolve host-dispatched kernels |
| `sudo greenboost feeders redeploy-netd` | Rebuild the feeder daemon from source and swap the binary in place, without touching the cluster key , lighter than a full feeder update for daemon-only fixes |
| `sudo greenboost feeders genkey` | Generate a fresh cluster pre-shared key on this machine (refuses to overwrite an existing key) |
| `sudo greenboost feeders export-key` | Print this machine's cluster pre-shared key for copying to another feeder |
| `sudo greenboost feeders import-key [HEX64]` | Write a cluster pre-shared key received via `export-key` onto this machine |
| `greenboost feeders diag [t1\|t2\|t3\|compute\|all]` | Run T1/T2/T3 alloc + compute diagnostics against each feeder via the net_fabric protocol; wraps `gb_feeder_diag.py` |
| `greenboost diag feeder` | Run all feeder checks |
| `greenboost diag feeder-t1` | T1 VRAM alloc/free test |
| `greenboost diag feeder-t2` | T2 DDR alloc/free test |
| `greenboost diag feeder-t3` | T3 NVMe alloc/free test |
| `greenboost diag feeder-compute` | Kernel dispatch test (sends `GB_MSG_CUDA_EXEC` and checks the response) |
| `greenboost built-stamp` | Show the local build stamp (BUILD_ID, version, git hash, build date) |
| `greenboost built-stamp --feeders` | Show the build stamp on the local host **and** every connected feeder side by side; highlights mismatched stamps in amber |

---

## 🩺 Diagnostics

| Command | Description |
|---|---|
| `greenboost vitals` | Live TUI: T1/T2/T3 pool gauges, GPU metrics, cluster status, pressure alerts (5 s refresh) |
| `greenboost top [--llm]` | Live per-buffer residency view, sorted hottest-first (VRAM by tier and by heat: hot/warm/cold) |
| `greenboost residency [--llm]` | Aggregate hot/warm/cold byte breakdown + churn across all live buffers |
| `greenboost faults [--llm]` | Tier-migration and memory-fault activity (eBPF tracer when running, kernel counters otherwise) |
| `sudo greenboost tune-revert` | Undo everything the reactive orchestrator's continuous OS tuner changed (CPU governor, GPU clocks/power limit, sysctls) , restores the pre-tune baseline |
| `greenboost status` | Show the cuda memory pool (T1 VRAM / T2 RAM / T3 NVMe), module state, and system health |
| `greenboost benchmark` | Cuda memory pool bandwidth benchmark (T1 VRAM / T2 DDR / T3 NVMe) |
| `greenboost inference-test [--model NAME] [--llm]` | End-to-end inference benchmark , verifies the fastest compute path (A/B) and measures tok/s |
| `greenboost logs [--llm]` | Aggregated log snapshot: kernel module events, services, AppArmor denials (TUI loop when interactive) |
| `greenboost inference-logs [--llm]` | Show inference service logs (Ollama, idle-reclaim, shader-boost) |
| `greenboost nvtx-logs [--llm]` | Live NVTX event log , allocation events, phase transitions, OOM events |
| `greenboost nvtx vitals [--last N] [--filter EV] [--feeder-only] [--local-only] [--llm]` | NVTX event timeline view |
| `greenboost health-check [--llm]` | One-shot comprehensive cluster health audit (module, shim, NVML, feeder handshakes) |
| `greenboost capabilities [--llm]` | Print the installed/running shim's feature manifest (`gb_monitor.py`), also written to `/run/greenboost/capabilities.json` |
| `greenboost pilot [--llm]` | Read-only dataflux advisor (`gb_pilot.py`): evidence-backed advice naming the exact `gb_control` lever to pull next, never actuates anything itself |
| `greenboost build-info [--llm]` | Show build metadata (version, git hash, build date, CUDA version) |
| `sudo greenboost clear logs` | Clear all GreenBoost log sources for a fresh diagnostic baseline (dmesg, journal, log files) |
| `sudo greenboost clear inference-logs` | Clear inference service logs only |
| `sudo greenboost clear nvtx-logs` | Rotate the NVTX event log (rename current → `.1`, start fresh) |
| `greenboost diag all` | Run the full local diagnostic suite (T1/T2/T3 alloc tests + compute) |

---

## 📦 TurboQuant K/V compression

| Command | Description |
|---|---|
| `sudo greenboost turboquant on` | Enable TurboQuant globally: sets `OLLAMA_KV_CACHE_TYPE=q4_0` and `GREENBOOST_TURBOQUANT=1`, restarts Ollama |
| `sudo greenboost turboquant off` | Disable TurboQuant: reverts the KV cache to `q8_0` |
| `greenboost turboquant status` | Show the current TurboQuant state (flag file + Ollama drop-in) |
| `greenboost turboquant --llm` | Machine-readable TurboQuant state for scripts/AI tools |

---

## 🖥️ ggml-2dev (feeder as CUDA device 1)

| Command | Description |
|---|---|
| `sudo greenboost ggml-2dev on` | Present the connected feeder as a real second CUDA device to Ollama/llama.cpp: oversized models split layers across local VRAM (device 0) and the feeder (device 1), with the feeder GPU computing its assigned layers. Ollama-only - torch/HF pipelines are unaffected. OFF by default; requires a connected feeder (`sudo greenboost connect <IP>`) |
| `sudo greenboost ggml-2dev off` | Revert to the default single-virtual-GPU path (feeder as memory tier only) |
| `greenboost ggml-2dev status` | Show whether the flag file and Ollama drop-in have `GREENBOOST_GGML_2DEV=1` set |
| `greenboost ggml-2dev --llm` | Machine-readable ggml-2dev state for scripts/AI tools |

---

## 🧠 gb-synapse (HuggingFace-native, cluster-distributed GGUF serving)

**New in v3.2.** Ollama only serves models from its own registry. gb-synapse
pulls GGUFs straight from any HuggingFace repo (gated or public, given a
token), serves them with a matched llama.cpp build on every cluster node
using llama.cpp's own `--rpc` backend for the cross-node split (real
layer-granular tensor split, only activations cross the wire), and puts an
Ollama + OpenAI + HuggingFace-TGI compatible API on one port (`:11435`) so
existing tooling doesn't notice the difference. The GreenBoost shim still
runs on every node underneath it, extending each node's own VRAM slice into
that node's local RAM/disk , RPC owns the boundary *between* machines, the
shim owns the tiers *within* each one.

| Command | Description |
|---|---|
| `greenboost doctor [--llm]` | Cluster hardware view: host + feeder GPU/RAM, aggregate VRAM/RAM, engine build status, HF token status |
| `greenboost recommend [ctx] [--llm]` | Fit + throughput estimate for every pulled model against the live cluster VRAM budget |
| `sudo greenboost synapse login [TOKEN]` | Store a HuggingFace token (masked prompt if omitted, or set `HF_TOKEN`) |
| `sudo greenboost pull <repo>[:quant] [name]` | Download a GGUF from HuggingFace; auto-picks the largest quant that fits the cluster if none given |
| `greenboost synapse list [--llm]` | List all known models (HuggingFace-pulled + indexed Ollama blobs) |
| `sudo greenboost synapse rm <name>` | Remove an HF-pulled model (refuses Ollama-managed blobs - use `ollama rm`) |
| `sudo greenboost synapse index-ollama` | Register every GGUF Ollama has already downloaded so `synapse list`/`recommend` see them |
| `sudo greenboost synapse build-engine` | Build llama-server/llama-cli/rpc-server from the vendored `greenboost-sources/llama.cpp` (CUDA+RPC). Must run on every cluster node - binaries are not portable across CPU vendors |
| `sudo greenboost synapse update-engine` | Fetch the latest `ggml-org/llama.cpp` and rebuild |
| `greenboost synapse run <model> [port]` | Serve a model: starts feeder `rpc-server`(s) if online, computes `--tensor-split` from real free VRAM, launches `llama-server --rpc` + the API proxy (Ollama + OpenAI + HuggingFace TGI endpoints) on `:11435` |
| `greenboost synapse stop <model>` | Stop a running gb-synapse server |
| `greenboost synapse ps [--llm]` | Running gb-synapse servers (TUI loop when interactive) |

---

## ⚡ Inference configuration

| Command | Description |
|---|---|
| `greenboost gen-inference-config` | Auto-generate optimized Ollama env vars for the current hardware (virt type, T1/T2/T3, TurboQuant) |
| `greenboost gen-inference-config --format env` | Emit systemd-compatible `Environment=` lines instead of `KEY=VALUE` |
| `greenboost gen-inference-config --format both` | Emit both Ollama env vars and systemd drop-in lines |
| `greenboost gen-inference-config --turboquant` | Force TurboQuant on in the generated config |
| `greenboost gen-inference-config --output FILE` | Write the config to `FILE` instead of stdout |
| `greenboost gen-inference-config --llm` | Machine-readable `key=value` output (no ANSI decoration) |

---

## 📡 Live debug signals

GreenBoost exposes three layers of live telemetry, readable at any time , even mid-inference.

### 1. sysfs kernel interface (always available when the module is loaded)

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
|---|---|---|
| `shim_stats` | `key=value` | Full shim counters: tier allocations, kernel dispatch count, feeder alloc MB, T2 frag %, path A/B counts, phase |
| `metrics.json` | JSON | Same data structured as JSON, with a `feeders[]` array per connected feeder |
| `nvtx_events.log` | TSV | Every allocation, phase change, and OOM event, with timestamp/tier/MB/pointer |
| `phase` | plain text | Current phase: `MODEL_LOAD` / `INFERENCE` / `STEADY` / `OOM_RECOVERY` |

**Key fields to watch in `shim_stats`:**

```bash
watch -n2 'grep -E "phase|tier_t1_feeder_cur|tier_t2_feeder_cur|kernel_dispatch|remote_alloc|t2_pool_frag|t2_above_warn|vram_headroom" /run/greenboost/shim_stats'
```

| Field | Meaning | Healthy value |
|---|---|---|
| `phase` | Memory manager phase | `STEADY` or `INFERENCE` |
| `tier_t1_feeder_cur_mb` | Feeder T1 VRAM in use | > 0 when a feeder is pooling memory |
| `kernel_dispatch_count` | CUDA kernels dispatched to a feeder | > 0 when feeder compute is active |
| `remote_alloc_count` | Allocations placed on a feeder | > 0 when feeder T1 is used |
| `t2_pool_frag_pct` | T2 DDR fragmentation | < 30 (> 80 = pressure) |
| `t2_above_warn` | T2 pool above the 75% threshold | 0 (1 = warn) |
| `vram_headroom_mb` | Free T1 VRAM headroom | > 200 |
| `path_a_count` | Allocations via pinned DDR (Path A , with kernel module) | increases during inference |
| `path_b_count` | Allocations via pinned DDR (Path B , no kernel module) | 0 on bare metal |

**Check feeder health from `metrics.json`:**

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
- **SHIM STATISTICS** , live poll of `shim_stats` every 2 s: path counters, tier allocations (local + feeder), kernel dispatch count, T2 fragmentation, T1 headroom
- **FEEDER VITALS** , per-feeder T1/T2/T3 current and peak, remote alloc count, connection status
- **NVTX EVENT LOG** , scrollable log of the last N NVTX events

---

## 🫀 Debug vitals (toggleable deep diagnostics)

GreenBoost debug vitals aggregate **9 data sources** into a single comprehensive view for troubleshooting, performance analysis, and enhancement work.

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
|---|---|---|
| Kernel verbose dmesg | `debug_mode=1` via sysfs | No , live |
| Verbose NVTX shim events | `GREENBOOST_NVTX_VERBOSE=1` in profile.d | New processes only |
| Extended vitals panel in scripts | `/etc/greenboost/debug_vitals.enabled` flag | No |
| `greenboost-idle-reclaim` | `systemctl restart` | Auto |
| Other GB-aware services | Listed on screen | Manual |

If `LD_PRELOAD` is not global (no `/etc/ld.so.preload` entry), a reboot warning is shown.

**Data sources exposed by `greenboost debug vitals`:**

| Source | What it shows |
|---|---|
| IOCTL `GB_IOCTL_GET_INFO` | T1/T2/T3/KV/pressure/phase_reset_seq (24 fields) |
| `/run/greenboost/shim_stats` | Path A/B counts, H2D/D2H MB, kernel_dispatch, T2 frag %, headroom, KV dedup, remote allocs |
| `/run/greenboost/phase` | Current phase (`MODEL_LOAD` / `INFERENCE` / `STEADY` / `OOM_RECOVERY`) |
| `/run/greenboost/nvtx_events.log` | Last 10 allocation/phase/OOM events with timestamps |
| `/run/greenboost/metrics.json` | JSON structured view + feeder T1/T2/T3 + bandwidth |
| `/sys/class/greenboost/greenboost/` | `pool_brief`, `status`, `active_buffers`, `active_profile` |
| `/sys/module/greenboost/parameters/debug_mode` | Verbose dmesg state |
| `nvidia-smi` extended | Temp, power, power limit, GPU util%, mem util%, SM clock, mem clock |
| `journalctl -k --grep=greenboost` | Kernel module events (when `debug_mode=1`) |

**From greenboost-cli (Python):**

```bash
/gb-vitals          # Full Rich-rendered vitals in the CLI
/gb-vitals on       # Enable (same as sudo greenboost debug vitals on)
/gb-vitals off      # Disable
```

---

## 📁 Profile management

Hardware profiles store auto-detected parameters in `/etc/greenboost/profiles/`.
The active profile drives module parameters (`virtual_vram_gb`, `safety_reserve_gb`, etc.).

Running `greenboost profile` (no sub-command) opens the **interactive wizard**
with a live hardware panel and a numbered menu. Individual sub-commands are
also available directly:

| Command | Description |
|---|---|
| `greenboost profile` | Open the interactive wizard: create, activate, diff profiles |
| `sudo greenboost profile create` | Auto-detect hardware and create a new profile |
| `greenboost profile list` | List all available profiles |
| `greenboost profile show` | Show the currently active profile |
| `greenboost profile show <file>` | Show a specific profile file |
| `sudo greenboost profile activate <file>` | Switch the active profile |
| `greenboost profile diff <file>` | Cross-check a profile against live hardware |

---

## 🧰 Maintenance

| Command | Description |
|---|---|
| `sudo greenboost install-sys-configs` | (Re-)install the Ollama env, NVMe udev rules, CPU governor, hugepages, LD_AUDIT, and idle-reclaim daemon |
| `sudo greenboost recover` | Attempt automatic recovery after a failed install or module load error |

## GB-CLI (greenboost-cli, installed by Full Install)

| Command | Description |
|---|---|
| `gb` (or `greenboost-cli`) | Open the agentic terminal client — gb-synapse-only backend on `:11435` |
| `gb -p "<prompt>" [-m model]` | One-shot prompt through gb-synapse (cross-GPU split when serving with `--rpc`) |
| `gb <headless-subcommand>` | Script-friendly JSON subcommands: `rag-search`, `rag-status`, `tokens`, `skill-list`, `plan-list`, `convert`, `compress`, … |
| `sudo greenboost install-cli` | (Re-)install just the CLI (venv under `/usr/local/lib/greenboost/cli-venv` + `/usr/local/bin/gb`) |
| `python3 gb_synapse.py status` | GB-Synapse engine/proxy status (also `synapse_status` in the MCPs) |
| `python3 gb_rotator.py run <queue.json>` | Overnight multi-model rotation (serve → run work → stop, resumable) |

### GB-CLI workflow (offline, whole cluster)

1. `greenboost_overview()` / `flux_health()` (greenboost-orchestrator MCP) — know the system, confirm the loop is closed.
2. Pick model + precision: `quant_advisor()` or `greenboost synapse recommend` — fp8 quality floor; below-fp8 only with the surfaced tradeoff.
3. Serve on the cluster: `greenboost synapse serve <model>` (llama.cpp `--rpc` tensor-split: host GPU + feeder GPU, feeder share backed by feeder VRAM→DDR via the feeder shim).
4. Query from the host, fully offline: `gb -p "…"`, any ollama client, or ai-forge pipelines via `FORGE_OLLAMA_URL=http://127.0.0.1:11435`.
5. Watch the dataflux: `greenboost dataflux-ui` or the `greenboost-dataflux` MCP (`dataflux_tok_s`, `dataflux_models`).

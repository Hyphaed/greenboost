# GreenBoost CLI — Architecture

## System Overview

```
User Input (terminal REPL / slash commands)
        │
        ▼
  terminal/repl.py              readline loop, process_query()
        │
        ├─► workflow/intelligence.py   pre_process_query()
        │       RAG inject → goals inject → compression → cache sentinel
        │
        ├─► core/orchestrator.py       execute_turn() generator
        │       │
        │       ├─► inference/router.py         generate() → resolve_backend()
        │       │       ├─► inference/adapters.py    Anthropic / OpenAI streams
        │       │       └─► inference/injection.py   tool-inject for Ollama/vLLM
        │       │
        │       └─► instruments/dispatcher.py   dispatch() → handlers.py
        │               Read / Write / Edit / Bash / Glob / Grep / WebFetch / WebSearch
        │
        └─► terminal/renderer.py       streaming output, tool display, statusline
```

---

## GreenBoost Memory Tier Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Virtual VRAM seen by CUDA process: ~126 GB             │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  T1 — VRAM   │  │  T2 — DDR    │  │  T3 — NVMe   │  │
│  │    11 GB     │  │    42 GB     │  │    73 GB     │  │
│  │  RTX 5070    │  │ cuMemHost-   │  │ GreenBoost   │  │
│  │  Blackwell   │  │ Register     │  │ backing file │  │
│  │  CC 12.0     │  │ Path B       │  │ ~100× slower │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│        hot layers        cold layers     emergency only  │
└─────────────────────────────────────────────────────────┘
         ▲                    ▲
         │                    │ libgreenboost_cuda.so shim
         │               (LD_PRELOAD intercepts cudaMalloc,
         │                cuMemGetInfo_v2, cuMemHostRegister)
         │
   CUDA process (vLLM, FLUX diffusion, sentence-transformers)
```

**Allocation routing:**
- T1 fills first (native CUDA cudaMalloc)
- When T1 headroom < threshold → shim routes overflow to T2 (cuMemHostRegister, Path B)
- When T2 exhausted → shim routes to T3 NVMe (slow — avoid)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` required so shim can intercept cudaMalloc (expandable_segments uses cuMemCreate which bypasses the hook)

---

## Monitor Detection Chain

`greenboost/monitor.py:GreenBoostMonitor._detect()`:

```
1. ioctl /dev/greenboost (GB_IOCTL_GET_INFO)
       │  success                  │  fail (EPERM / no device)
       ▼                           ▼
   fill from ioctl struct     2. sysfs /sys/module/greenboost/ exists?
   (max_pool_mb may be 0)            │  yes
       │                            ▼
       └──────────────────►  _read_sysfs_stats()
                             _read_sysfs_class_status()   (key=value format)
                                     │
                                     ▼
                    3. Both paths → _fill_from_sysfs_status()
                       reads /sys/class/greenboost/greenboost/status
                       (human-readable "Label  :  value" format)
                       fills: ram_pool_mb, ram_available_mb,
                              nvme_swap_total_mb, nvme_swap_used_mb
                       detects stale kv_t3 kernel counter

                    4. Fallback: dkms status (module present but not loaded)
```

**Key sysfs file:** `/sys/class/greenboost/greenboost/status`
```
Tier 2  System RAM pool    :   42 GB   PCIe DMA  [cold layers]
  T2 allocated             :    0 MB  (0%)
  T2 available             : 50993 MB
Tier 3  T3 backing file    :   73 GB  NVMe file
  T3 allocated             :    0 MB
  KV in T3 (NVMe swap)     :    0 MB
```

**Key sysfs file:** `/sys/class/greenboost/greenboost/pool_brief` (one-line summary)
```
T1:11GB T2:0/42GB(0%) T3:0/73GB PRESSURE:ok KV_RSV:0MB KV_T2:0MB
```

---

## GreenBoostStatus Dataclass

Primary fields populated by `_detect()`:

| Field | Source | Description |
|-------|--------|-------------|
| `loaded` | ioctl / sysfs | Module is present and active |
| `vram_total_mb` | nvidia-smi | T1 GPU VRAM total |
| `vram_used_mb` | nvidia-smi | T1 GPU VRAM used |
| `ram_pool_mb` | sysfs status | T2 DDR pool total (GB × 1024) |
| `ram_allocated_mb` | ioctl / sysfs | T2 DDR currently allocated |
| `ram_available_mb` | sysfs status | T2 DDR free for allocation |
| `nvme_swap_total_mb` | sysfs status | T3 NVMe backing file size |
| `nvme_swap_used_mb` | sysfs / ioctl | T3 NVMe currently in use |
| `kv_used_mb` | sysfs class | KV cache total (T1+T2+T3) |
| `kv_t2_mb` | sysfs class | KV cache in T2 DDR |
| `_kv_t3_stale` | sysfs analysis | True = kernel counter not reset on exit |
| `shim_phase` | /run/greenboost/shim_stats | INIT / STEADY / INFERENCE / OOM |
| `shim_active_path` | shim_stats | A0 / B / none |
| `nvtx_events` | /run/greenboost/nvtx_events.log | Last N NVTX events |

---

## Dashboard Server

`dashboard/server.py` — stdlib-only HTTP server on port 7821 (configurable).

**API endpoints used by the live dashboard:**
- `GET /api/status` → `s.as_dict()` (all tier stats)
- `GET /api/logs` → last 200 lines of `~/.greenboost_cli/session.log`
- `GET /api/vitals` → diffuser_vitals.json contents

The browser polls `/api/status` and `/api/logs` every 3 seconds via `pollStatus()` and
`pollLogs()` in the DOMContentLoaded handler. T2/T3 pool sizes were previously showing
0/0 GB because `as_dict()` returned `ram_pool_mb: 0` when ioctl `max_pool_mb` was unset.
Fixed by `_fill_from_sysfs_status()` in both ioctl and sysfs detection branches.

---

## Diffusion Pipeline Integration

FLUX pipelines in `curse_of_the_seas` scripts write live state to
`/run/greenboost/diffuser_vitals.json` via `gb_vitals.py`:

```python
# Pattern used in gen_art.py, gen_digital_assets.py, gen_manual_art.py
gb_vitals.write("loading", model=model_version, pipeline="FLUX")
with gb_vitals.nvtx_range(f"pipeline_load/{model_version}"):
    pipe = load_pipeline_nf4(...)
gb_vitals.write("ready", model=model_version, pipeline="FLUX")

# In generation loop:
gb_vitals.write("generating", model=..., pipeline="FLUX",
                step=0, total_steps=steps, prompt=card["art_prompt"][:120])
image = pipe(...)
gb_vitals.write("ready", model=..., elapsed_s=elapsed, last_image=str(out_path))
```

States visible in `gb vitals` terminal panel and dashboard `/greenboost` page:
`loading` → `ready` → `generating` → `ready`

---

## vLLM Integration

vLLM is **not bundled** in the greenboost-cli venv. It must be installed separately:
```bash
$HOME/.local/share/greenboost-cli/env/bin/pip install vllm
```

`slash_commands/backend_cmds.py:_find_vllm_bin()` search order:
1. Same venv as the running Python (sibling of `sys.executable`)
2. `~/.local/share/greenboost-cli/env/bin/vllm` (hardcoded GB venv path)
3. `shutil.which("vllm")` (system PATH)

`_build_vllm_env()` injects the GreenBoost env vars into the vLLM subprocess:
```
LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so
GREENBOOST_ACTIVE=1
GREENBOOST_VLLM_COMPAT=1        # VCM-01 deferred init
GREENBOOST_VIRTUAL_VRAM_MB=43008
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
```

### LMCache (`/lmcache`)

Opt-in KV-cache reuse connector for the `vllm/` backend, toggled via
`/lmcache [on|off|status]` (`slash_commands/greenboost_cmds.py:cmd_lmcache`,
mirrors `/turboquant`). State is stored in CLI settings (`lmcache_enabled`),
not a root flag file.

- `_lmcache_installed()` / `_ensure_lmcache_installed()` (`backend_cmds.py`)
  check/install the `lmcache` package into the same venv as `vllm` (resolved
  via `_find_vllm_bin()`), so LMCache never pulls its own torch.
- When enabled, `_build_vllm_cmd()` appends
  `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`.
- When enabled, `_build_vllm_env()` sets `LMCACHE_LOCAL_CPU=True`,
  `LMCACHE_MAX_LOCAL_CPU_SIZE=8` (GB, host RAM — kept below the GreenBoost T2
  pool size, never derived from it), `LMCACHE_LOCAL_DISK=file://~/.local/share/greenboost-cli/lmcache/`,
  `LMCACHE_MAX_LOCAL_DISK_SIZE=40` (GB), `LMCACHE_CHUNK_SIZE=256`.
- If enabled but install fails, vLLM still launches without the connector
  (warning only, never a hard failure).

**Scope:** LMCache only accelerates the `vllm/` backend. It has no
llama.cpp/Ollama connector (Ollama KV speedup stays on `/turboquant`) and no
applicability to diffusion pipelines (ComfyUI / FLUX 2 Klein + LoRAs — no
autoregressive KV cache exists there to reuse; those already use GreenBoost's
T2/T3 VRAM tiering described above). See Chapter G.14 in the parent repo's
`greenboost_documentation_extension_official_nvidia.md` for the full
coexistence rationale.

---

## Key Invariants

- **`expandable_segments:False`** — set by `apply_gb_torch_env()` before any torch import.
  Without this, PyTorch's expandable allocator uses `cuMemCreate` which bypasses the shim's
  `cudaMalloc` hook and T2 routing breaks silently.
- **ioctl `max_pool_mb`** — returns 0 when no explicit cap set via `GB_IOCTL_SET_POOL_CAP`.
  Never interpret 0 as "no T2 pool"; always fall back to sysfs for pool totals.
- **KV T3 counter** — kernel module's `kv_t3_tracked_mb` not reset on process exit.
  Python guard: if `KV in T3 > 0` AND `T3 allocated == 0`, treat as stale.

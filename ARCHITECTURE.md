# GreenBoost Architecture - v2.9

**GreenBoost** is a CUDA Memory Orchestrator. It presents every CUDA process with a single virtual GPU device backed by three physical memory tiers, eliminating CPU layer spillover in LLM inference without requiring application code changes.

---

## Component Map

```
┌─────────────────────────────────────────────────────────────────┐
│  User space                                                     │
│                                                                 │
│  ┌─────────────────────────┐   ┌──────────────────────────┐    │
│  └────────────┬────────────┘   └──────────┬───────────────┘    │
│               │ cudaMalloc / cuMemAlloc    │ vkAllocateMemory   │
│  ┌────────────▼────────────┐   ┌──────────▼───────────────┐    │
│  │  libgreenboost_cuda.so  │   │  libVkLayer_greenboost.so │    │
│  └────────────┬────────────┘   └──────────────────────────┘    │
│               │ ioctl / mmap                                    │
│  ┌────────────▼────────────┐                                    │
│  │  /dev/greenboost        │   ◄── LD_AUDIT gatekeeper          │
│  │  (char device)          │       libgreenboost_audit.so       │
│  └────────────┬────────────┘       /etc/ld.so.audit             │
└───────────────┼─────────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────────┐
│  Kernel space                                                   │
│                                                                 │
│  greenboost.ko                                                  │
│  ├── DMA-BUF exporter (T2: pinned System DDR hugepages)         │
│  ├── Watchdog kthread  (safety_reserve + swap pressure)         │
│  ├── IDR buffer tracker (struct gb_buf per live allocation)     │
│  ├── sysfs: status / hw_info / active_buffers / active_profile  │
│  └── features/nvlink_pool  (V100 cluster T1 aggregation)        │
│                                                                 │
│  T3: GreenBoost backing file → /var/lib/greenboost/t3_store     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Memory Tier Hierarchy

| Tier | Physical device | Bandwidth | Role |
|------|----------------|-----------|------|
| T1 | GPU VRAM | ~336 GB/s | Hot computation - full GPU bandwidth, native CUDA execution |
| T2 | System DDR (pinned, DMA-BUF) | ~50 GB/s | Cold weights - transferred over PCIe on demand |
| T3 | NVMe (backing file) | ~1.8 GB/s CPU / ~7 GB/s GDS | Frozen pages - `/var/lib/greenboost/t3_store`; GDS path bypasses CPU |
| T4 | Remote cluster GPU + DDR | ~1 GB/s (GbE) / ~10 GB/s (10GbE) | Pipeline-parallel layers on feeder nodes |

All tiers are invisible to the application. CUDA apps see a single virtual VRAM device (T1+T2+T3 locally) plus additional CUDA device IDs for each cluster feeder (T4). The IOCTL `GB_IOCTL_GET_INFO` field `total_combined_mb` covers T1–T3; `greenboost cluster` shows the full T1–T4 view.

### TurboQuant K/V Compression Layer

`gb_attn.py` adds a bandwidth-reduction layer on top of the memory tiers: when K/V tensors
live in T2 DDR, reading them back to T1 for each attention call is the main throughput
bottleneck. TurboQuant+ compresses K/V in-place before the DMA.

**Global toggle (v2.9+):** `sudo greenboost turboquant on|off|status` activates TurboQuant
system-wide in a single command.  It manages two distinct paths simultaneously:

| Inference path | TurboQuant mechanism | Activation |
|---|---|---|
| Ollama (Go binary) | `OLLAMA_KV_CACHE_TYPE=q4_0` + `OLLAMA_FLASH_ATTENTION=1` in Ollama systemd drop-in | Ollama restarted automatically |
| Python apps (PyTorch / Transformers / Diffusers / vLLM) | `gb_attn.py` patches `F.scaled_dot_product_attention`; `GREENBOOST_TURBOQUANT=1` exported via `/etc/profile.d/greenboost-turboquant.sh` | Active for all subsequent login shells |

State is persisted in `/etc/greenboost/turboquant.enabled` (flag file) and in the
Ollama drop-in `/etc/systemd/system/ollama.service.d/99-greenboost.conf`.


| Method | Bits | Compression | PPL impact |
|--------|------|-------------|------------|
| PolarQuant only (V) | v_bits | 3–6× | ~+0.06 % |
| TurboQuant (K) | k_bits | 3–8× | ~+0.23 % |
| K=4bit / V=3bit (recommended) | asymmetric | ~4× | +0.23 % |
| K=3bit / V=2bit (memory pressure) | asymmetric | ~6× | +1.06 % |

K tensors use **PolarQuant (k_bits−1) + QJL 1-bit residual** to preserve inner products for Q·Kᵀ.
V tensors use **PolarQuant v_bits only** (MSE-optimal; inner products not required).
The C-shim additionally applies absmax int8 quantisation at eviction time (Phase 2 of `tq_plan.md`)
for a further 2× reduction on the DMA path, orthogonal to gb_attn.py.

**0 % CPU spillover mandate.** GreenBoost intercepts `cuDeviceTotalMem_v2`, `cuMemGetInfo`, and `nvmlDeviceGetMemoryInfo` to return the full virtual VRAM size. Ollama is further protected by `OLLAMA_NUM_GPU=999`, which forces all layers onto GPU regardless of Ollama's internal VRAM accounting. Computation always executes on the GPU; only data residency moves between tiers.

---

## Debug Vitals Layer (v2.9+)

A toggleable deep-instrumentation subsystem that aggregates **9 data sources** into a single view.
Designed to always know what GreenBoost is doing - essential for debugging, enhancement, and issue diagnosis.

```
Toggle:   sudo greenboost debug vitals on   →  /etc/greenboost/debug_vitals.enabled
          sudo greenboost debug vitals off  →  removes flag, resets instrumentation

Show:     greenboost debug vitals           →  full dump, always available
          /gb-vitals                         →  same, from greenboost-cli (Rich)
```

**Data sources aggregated:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Debug Vitals Layer                                                     │
│                                                                         │
│  Kernel (via IOCTL + sysfs):                                            │
│    GB_IOCTL_GET_INFO        → 24 fields: T1/T2/T3/KV/pressure/phase     │
│    /sys/class/greenboost/   → pool_brief, status, active_buffers        │
│    /sys/module/greenboost/  → debug_mode, kv_reserve_mb, version        │
│                                                                         │
│  Shim (via /run/greenboost/ runtime files):                             │
│    shim_stats      → path A0/A/B, H2D/D2H MB, dispatch, frag, dedup    │
│    phase           → MODEL_LOAD / INFERENCE / STEADY / OOM_RECOVERY     │
│    nvtx_events.log → per-allocation events with timestamps              │
│    metrics.json    → structured JSON + feeder T1/T2/T3 + bandwidth      │
│                                                                         │
│  Hardware (via nvidia-smi extended):                                    │
│    temp, power, power_limit, GPU util%, mem util%, SM clock, mem clock  │
│                                                                         │
│  Kernel journal (when debug_mode=1):                                    │
│    journalctl -k --grep=greenboost  → module events, pressure changes   │
└─────────────────────────────────────────────────────────────────────────┘
```

**What "vitals on" changes:**

| Setting | Mechanism | Applied how |
|---------|-----------|-------------|
| Kernel verbose dmesg | `debug_mode=1` sysfs write | Live, no restart |
| Verbose NVTX shim events | `GREENBOOST_NVTX_VERBOSE=1` in `/etc/profile.d/greenboost-vitals.sh` | New processes |
| Extended script panels | `/etc/greenboost/debug_vitals.enabled` flag file | Immediately |
| greenboost-idle-reclaim | `systemctl restart` | Auto |
| Other GB-aware services | Listed on screen for manual restart | Manual |

Reboot is required only if `LD_PRELOAD` shim is not globally applied (`/etc/ld.so.preload`).

---

## Cluster Architecture (T4 - Network Fabric)

### Topology (v3.1 - Single Virtual GPU)

GreenBoost presents **exactly ONE CUDA device (device 0)** to Ollama regardless of how many
feeders are connected. All cluster memory is aggregated into that single device's memory reports.
Ollama sees one huge GPU and places all model layers on it.

```
┌──────────────────── Host (B760M) ─────────────────────────────┐
│                                                               │
│  Ollama process                                               │
│    cudaGetDeviceCount() → 1  (single virtual GPU)            │
│    cuDeviceTotalMem(0)  → local T1+T2+T3 + remote memory     │
│    cuMemGetInfo(0)      → aggregated free across all tiers    │
│                                                               │
│  libgreenboost_cuda.so  ←──── LD_PRELOAD ──────────────────  │
│    ├── gb_shim_init → gb_netc_init → TCP 9740 → feeders      │
│    ├── cudaMalloc overflow order: T1_feeder → T2_local        │
│    │     → T2_feeder → T3_local → T3_feeder                  │
│    ├── cudaMemcpy(fake 0xAA…ptr) → gb_netc_memcpy_h2d/d2h   │
│    └── cuLaunchKernel → data-driven dispatch: if any arg      │
│          is a feeder fake-ptr → GB_MSG_CUDA_EXEC → feeder     │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                          │ TCP 9740
                          │ binary wire protocol (proto v3)
                          │ gb_net_header (16 bytes, little-endian)
                          ▼
┌──────────────────── Feeder (omen) ────────────────────────────┐
│                                                               │
│  greenboost-netd                                              │
│    ├── GB_MSG_CUDA_MALLOC  → cudaMalloc on feeder VRAM/DDR   │
│    ├── GB_MSG_CUDA_MEMCPY  → data transfer local ↔ feeder    │
│    ├── GB_MSG_CUDA_EXEC    → dlsym kernel + cuLaunchKernel   │
│    ├── GB_MSG_CUDA_REGISTER_FN → register kernel on feeder   │
│    └── GB_MSG_HEARTBEAT    → live VRAM/RAM stats             │
│                                                               │
│  RTX 5070  7 GB  +  DDR5  20 GB                              │
└───────────────────────────────────────────────────────────────┘
```

### Memory allocation order within the single virtual device
Overflow allocation is attempted in this exact order:
```
T1 host  (local GPU VRAM - real cudaMalloc, fastest)
T1 feeder (feeder GPU VRAM - fake ptr 0xAA00…, faster than local DDR)
T2 host  (local DDR, DMA-BUF pinned)
T2 feeder (feeder DDR)
T3 host  (local NVMe swap)
T3 feeder (feeder NVMe swap)
```

### Compute dispatch (data-driven)
`cuLaunchKernel` and `cuLaunchCooperativeKernel` scan the kernel's argument buffer for remote
fake pointers (`0xAA00…`). If any argument lives on a feeder, the kernel is dispatched to that
feeder via `GB_MSG_CUDA_EXEC` over TCP - no manual device selection required.

### Wire protocol (`features/net_fabric.h`)
Every message starts with a 16-byte `gb_net_header` (packed, little-endian, proto v3):
```c
struct gb_net_header {
    gb_u32 magic;       /* 0x47424E46 "GBNF"             */
    gb_u16 msg_type;    /* GB_MSG_* enum                  */
    gb_u16 flags;       /* GB_NET_FLAG_*                  */
    gb_u32 payload_len; /* bytes after header             */
    gb_u32 seq_num;     /* per-connection sequence number */
};
```
Key message types: `HANDSHAKE_REQ/RESP` (connect), `CUDA_MALLOC/FREE`, `CUDA_MEMCPY_H2D/D2H/D2D`,
`CUDA_LAUNCH`, `CUDA_REGISTER_FN`, `HEARTBEAT`, `GPU_QUERY`, `MEM_INFO`.

**PSK authentication (optional):** If `/etc/greenboost/cluster.key` exists (32-byte hex), the
feeder sends a 32-byte random nonce immediately after `accept()`. The host responds with
`HMAC-SHA256(psk, nonce)` (32 bytes). The feeder closes the connection on MAC mismatch.
Authentication runs before the `HANDSHAKE_REQ` message exchange.

### Adding a new feeder (laptop, NAS, etc.)
```bash
# On the feeder machine - build + install netd
git clone /path/to/greenboost && cd greenboost
make netd && sudo make install-libs
sudo systemctl enable --now greenboost-netd

# On the host B760M
sudo greenboost connect <feeder_IP>
sudo systemctl restart ollama   # picks up new feeder
greenboost cluster              # verify all nodes visible
```

---

## Components

### 1. Kernel Module - `greenboost.c` → `greenboost.ko`

**Responsibilities:**
- Registers the `/dev/greenboost` character device with an IOCTL interface.
- **Tier 2 allocator:** allocates pinned physical pages (2 MB hugepages by default, 4 K optional) and exports them as DMA-BUF file descriptors (`GB_IOCTL_ALLOC`). The GPU imports these via `cudaImportExternalMemory` without a CPU round-trip.
- **Tier 3 monitor:** reads and writes cold buffers directly to the backing file at `/var/lib/greenboost/t3_store` (configurable via `t3_file_path` module parameter or `GB_IOCTL_SET_T3_CAP`). Disabled by default (`nvme_pool_gb=0`); enabled on systems with NVMe.
- **Watchdog kthread:** monitors free RAM against `safety_reserve_gb` and NVMe swap pressure. Signals userspace via eventfd (`GB_IOCTL_POLL_FD`) when pressure levels change.
- **IDR tracker:** uses Linux's integer ID allocator to track all live `gb_buf` objects. A mutex + spinlock pair protects the pool.
- **sysfs interface** under `/sys/class/greenboost/greenboost/`:
  - `status` - human-readable pool summary
  - `hw_info` - detected hardware
  - `active_buffers` - live DMA-BUF count
  - `active_profile` - name of the loaded profile
- **Reboot/panic notifiers** for graceful teardown.
- **NVLink pool** (`features/nvlink_pool.c`) - optional V100 cluster extension that aggregates multiple GPU VRAMs into a unified T1 pool (see §6 below).

**Key module parameters** (all auto-detected or profile-driven; none hard-coded):

| Parameter | Purpose |
|-----------|---------|
| `physical_vram_gb` | T1 usable VRAM (physical − headroom) |
| `virtual_vram_gb` | T2 DDR pool cap (~80 % of free RAM) |
| `safety_reserve_gb` | Minimum free RAM always kept |
| `nvme_swap_gb` | T3 NVMe swap capacity monitored for pressure |
| `nvme_pool_gb` | GreenBoost T3 allocation soft-cap (0 = disabled) |
| `use_hugepages` | 2 MB compound pages for T2 |
| `kv_reserve_mb` | T1 VRAM reserved for KV cache |
| `pcores_max_cpu` | Last P-core logical CPU (watchdog affinity) |
| `golden_cpu_min` | First high-frequency golden-core CPU (-1 = disabled; Intel hybrid CPUs) |
| `golden_cpu_max` | Last high-frequency golden-core CPU (-1 = disabled) |
| `t3_file_path` | Path to GreenBoost T3 backing file (default: `/var/lib/greenboost/t3_store`) |
| `idle_cleanup_sec` | Seconds between watchdog dead-PID buffer reap (0 = disabled, default: 30) |
| `debug_mode` | Verbose `dmesg` output |
| `active_profile_name` | Set by installer at `insmod` time |

---

### 2. CUDA Shim - `greenboost_cuda_shim.c` → `libgreenboost_cuda.so`

Loaded system-wide via `/etc/ld.so.preload`. A two-stage constructor (`RTLD_NOLOAD` probe) keeps it inert in non-CUDA processes (GDM, shells, systemd helpers, PAM).

#### Intercepted symbols

| Symbol | Purpose |
|--------|---------|
| `cudaMalloc` / `cudaMallocAsync` | Route overflow to T2/T3 |
| `cuMemAllocAsync` | Route overflow (compute capability ≥ 8.0 gate) |
| `cudaFree` / `cuMemFree_v2` / `cuMemFreeAsync` | Release T2/T3 buffers |
| `cuDeviceTotalMem_v2` | Return virtual VRAM size |
| `nvmlDeviceGetMemoryInfo` / `_v3` | Return virtual VRAM stats |
| `cuMemCreate` | Intercept CUDA VMM allocations (Ollama 0.18+ ggml backend); retries with host-pinned VMM on T1 OOM |
| `dlsym` | Intercept Ollama's runtime GPU API lookups |
| `dlopen` | Strip `RTLD_DEEPBIND` so hooks stay active inside bundled CUDA libs |

#### Overflow allocation paths (tried in order)

| Path | Mechanism | Requirement |
|------|-----------|-------------|
| A0 | `GB_IOCTL_ALLOC` → `cudaImportExternalMemory(OpaqueFd)` | `greenboost.ko` + libcudart ≥ 10.0 |
| A | `mmap` → `GB_IOCTL_PIN_USER_PTR` → `cuMemHostRegister(DEVICEMAP)` | `greenboost.ko` |
| B | `mmap` (2 MB huge) → `cuMemHostRegister(DEVICEMAP)` | No kernel module needed (containers, VMs, WSL2) |
| - | Path C (`cuMemAllocManaged`) **removed** - UVM caused GPU page-fault stalls and CPU-preferred demand-paging. `OOM` is returned when T2 is exhausted; llama.cpp handles it correctly. | - |

Path A0 is true zero-copy: CUDA drives the IOMMU mapping directly from the kernel DMA-BUF scatter-gather table built from 2 MB hugepages.

#### Phase detector

A temporal state machine classifies allocs to place KV cache in T1 and let weights spill to T2:

```
INIT → MODEL_LOAD → INFERENCE → STEADY → IDLE → DEEP_IDLE
```

| Phase | KV reserve | Rationale |
|-------|-----------|-----------|
| INIT / MODEL_LOAD | zeroed | Weights fill T1 maximally |
| INFERENCE / STEADY | active | KV tensors compete for VRAM |
| IDLE | active | No recent alloc; idle timeout running |
| DEEP_IDLE | active | Signals idle-reclaim daemon to unload models |

T1 utilization rises from ~33 % to ~75–83 % on a 12 GB device because the KV reserve only turns on once inference starts.

#### Adaptive KV reserve

```
effective_reserve = kv_reserve_bytes − g_kv_allocated_t1_bytes
```

If KV cache already occupies T1, the effective reserve collapses proportionally - eliminating double-counting so weights are not unnecessarily pushed to T2.

#### Internal data structures

- **Open-addressed hash map** (131 072 slots, 8 MB): maps `devPtr → gb_buf` for O(1) free-path lookup.
- `g_kv_allocated_t1_bytes` atomic: tracks KV bytes resident in T1.
- `g_phase` atomic: current phase enum.

#### Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GREENBOOST_VRAM_HEADROOM_MB` | 512 | Free VRAM kept before overflowing to T2 |
| `GREENBOOST_KV_RESERVE_MB` | kernel param | T1 VRAM reserved for KV cache |
| `GREENBOOST_PHASE_DETECT` | 1 | Phase detector on/off |
| `GREENBOOST_KV_SIZE_THRESHOLD_MB` | 64 | Min alloc size classified as KV in inference phase |
| `GREENBOOST_KV_OVERFLOW` | 0 | Force all overflow allocs to `GB_ALLOC_KV_CACHE` |
| `GREENBOOST_IDLE_TIMEOUT_MS` | 120 000 | ms idle in STEADY before → IDLE |
| `GREENBOOST_DEEP_IDLE_TIMEOUT_MS` | 900 000 | ms idle in IDLE before → DEEP_IDLE |
| `GREENBOOST_DISABLE` | - | Set to 1 to opt out a single process |
| `GREENBOOST_DEBUG` | 0 | Verbose stderr logging |

---

### 3. LD_AUDIT Gatekeeper - `greenboost_audit.c` → `libgreenboost_audit.so`

Installed to `/etc/ld.so.audit`. Intercepts `la_objopen()` at the dynamic linker level.

Triggers shim injection only when `libcuda.so.*` or `libcudart.so.*` appears in the process link map. All other processes (GDM, PAM, cups, snap-confine, shells) are left completely untouched - the AppArmor blast radius is a single read-only open of the audit library.

**Exclusion rules** (cached after first call):
- `GREENBOOST_DISABLE=1` - explicit opt-out.


---



**Two hooks:**

| Hook | Effect |
|------|--------|
| `vkAllocateMemory` | On `VK_ERROR_OUT_OF_DEVICE_MEMORY` for allocations ≥ 64 MB, retries from T2 DDR via DMA-BUF import (`VK_KHR_external_memory_fd`). Best-effort - silently returns the original error if `/dev/greenboost` is absent |


---

### 5. IOCTL Interface - `greenboost_ioctl.h`

Single header usable from both kernel and userspace. Magic byte: `'G'`.

| IOCTL | Cmd | Direction | Purpose |
|-------|-----|-----------|---------|
| `GB_IOCTL_ALLOC` | 1 | IOWR | Allocate pinned T2 buffer → DMA-BUF fd |
| `GB_IOCTL_GET_INFO` | 2 | IOR | Read pool statistics (`struct gb_info`) |
| `GB_IOCTL_RESET` | 3 | IO | Reset internal state |
| `GB_IOCTL_MADVISE` | 4 | IOW | Advise eviction priority (cold/hot/freeze/T1-prefer) |
| `GB_IOCTL_EVICT` | 5 | IOW | Force-push T2 buffer to T3 swap |
| *(cmd 6)* | 6 | - | ABI gap - do not reuse |
| `GB_IOCTL_POLL_FD` | 7 | IOW | Register eventfd for pressure notifications |
| `GB_IOCTL_PIN_USER_PTR` | 8 | IOWR | Pin existing userspace VA → DMA-BUF fd (Path A) |
| `GB_IOCTL_SET_KV_RESERVE` | 9 | IOW | Update T1 KV reserve at runtime |
| *(cmd 10)* | 10 | - | ABI gap - SET_TURBOQUANT removed, do not reuse |
| `GB_IOCTL_SET_POOL_CAP` | 11 | IOWR | Resize T2 pool cap dynamically |
| *(cmd 12–13)* | 12–13 | - | ABI gaps - do not reuse |
| `GB_IOCTL_GET_POOL_INFO_V3` | 14 | IOR | Machine-readable pool info for k8s DRA / Prometheus |
| `GB_IOCTL_RESET_PHASE` | 15 | IO | Increment phase_reset_seq (Synapse CLI uses before model swap) |
| `GB_IOCTL_RELEASE_PID` | 16 | IOW | Release all T2/T3 buffers owned by a PID |
| `GB_IOCTL_SET_T3_CAP` | 17 | IOWR | Live T3 backing file resize; 0 = disk-limited; requires CAP_SYS_ADMIN |
| `GB_IOCTL_SESSION_IDLE` | 18 | IOW | Move all PID's T2 buffers to LRU tail (preferred eviction candidates) |
| `GB_IOCTL_SESSION_ACTIVE` | 19 | IOW | Move all PID's T2 buffers to LRU head (evicted last under pressure) |

**Allocation flags** (`gb_alloc_req.flags`):

| Flag | Meaning |
|------|---------|
| `GB_ALLOC_WEIGHTS` | Model weight tensor |
| `GB_ALLOC_KV_CACHE` | KV cache - never spills to T3, auto-frozen in T2 LRU |
| `GB_ALLOC_ACTIVATIONS` | Ephemeral activation buffer |
| `GB_ALLOC_FROZEN` | Never evict from T2 |
| `GB_ALLOC_NO_HUGEPAGE` | Force 4 K pages |
| `GB_ALLOC_T1_PRIORITY` | KV-like; moved to LRU head - weight bufs evicted first |
| *(bit 6)* | Reserved - GB_ALLOC_KV_COMPRESSED removed, do not reuse |
| `GB_ALLOC_SESSION_PROTECTED` | Skipped by auto-eviction until all unprotected candidates are exhausted |

---

### 6. NVLink Pool - `features/nvlink_pool.{h,c}`

Optional extension for multi-GPU V100 HBM2 clusters. Aggregates all GPU VRAMs in an NVLink fabric into a single logical T1 pool.

- Kernel: `gb_nvlink_pool_init()` queries NVLink fabric state via NVML.
- `gb_nvlink_set_ready()` is called by the kubelet plugin after P2P verification; updates `total_vram_gb` and `gpu_count`.
- `GB_IOCTL_GET_POOL_INFO_V3` exposes `t1_nvlink_total_mb` and `nvlink_ready` for the Prometheus exporter.
- The CUDA shim's virtual VRAM is updated to `sum(per-GPU VRAM)` when the pool is active.

---

## Data Flow: CUDA Allocation

```
cudaMalloc(size)
    │
    ├── fits in T1 free VRAM?  ──yes──► NVIDIA driver handles normally
    │
    └── no (overflow)
         │
         ├─ [Phase check] KV reserve active? Does alloc fit in effective headroom?
         │
         ├─ Path A0: GB_IOCTL_ALLOC (DMA-BUF, 2 MB hugepages)
         │           cudaImportExternalMemory(OpaqueFd) → CUdeviceptr   ← preferred
         │
         ├─ Path A:  mmap + GB_IOCTL_PIN_USER_PTR + cuMemHostRegister
         │
         ├─ Path B:  mmap (2 MB huge) + cuMemHostRegister (no kernel module)
         │
         └─ T2 exhausted → OOM returned  (Path C / UVM removed; llama.cpp handles OOM correctly)
```

When a T2 buffer is freed (`cudaFree`), `GB_IOCTL_RELEASE_PID` or process exit, the kernel releases the DMA-BUF and unpins the pages. Cold T2 pages may be evicted to T3 by the watchdog or via `GB_IOCTL_EVICT` before free.

---

## Profile System

Profiles are Markdown files with YAML frontmatter that store auto-detected hardware values and workload parameters. The active profile is a symlink at `/etc/greenboost/active_profile.md`.

**Resolution priority:**
```
CLI flags > environment variables > active profile > compiled-in defaults
```

The `active_profile_name` module parameter is set at `insmod` time by the installer so the kernel sysfs `active_profile` attribute reflects the loaded profile.

**Storage layout:**
```
/etc/greenboost/
├── profiles/
│   ├── default.md            # auto-generated on first install
│   └── resolved_<ts>.md      # written when --profile has conflicts
└── active_profile.md         # symlink → profiles/<active>.md
```

---

## Installer & Configuration System (v2.9)

### Install menu structure

`sudo greenboost` (interactive wizard) exposes a two-tier install layout:

```
━━ Core ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [1]  Full install           - everything below, in order
  [2]  Light install          - module + hardware build only

━━ Additional install (included in full-install) ━━━
  [5]  Install sys configs    - Ollama drop-in, udev, CPU governor
  [6]  Install llama configs  - LD_AUDIT shim injection
  [7]  Tune runtime           - NVMe scheduler, swappiness, PCIe
  [8]  Tune sysctl            - persistent kernel tunables
  [9]  Tune GRUB              - hugepages, rcu_nocbs, nohz_full
  [10] Generate inference config - hardware-aware Ollama/HF config
```

Steps [5]–[10] are idempotent and can be run individually at any time.

### Inference configuration generator

`greenboost gen-inference-config` detects the runtime environment (bare-metal / VM / container / WSL2) and emits an optimised set of environment variables for Ollama and/or systemd drop-in lines.  Flags: `--format ollama|env|both`, `--turboquant`, `--no-turboquant`, `--output <path>`, `--llm`.

---

## Synapse CLI Integration (v2.9)

Synapse CLI (`greenboost_synapse_cli_new/`) is the companion multi-provider terminal AI assistant. As of v2.9 it is GreenBoost-aware:

| Feature | File | Detail |
|---|---|---|
| Startup banner GB/TQ badges | `synapse/terminal/renderer.py` | Shows `GB`, `TQ`, feeder count |
| System prompt injection | `synapse/environment/context_builder.py` | `_greenboost_context()` appended to every request |
| `/turboquant` slash command | `synapse/terminal/greenboost_cmds.py` | Calls `sudo greenboost turboquant on\|off\|status` |
| Qwen3.6-35B model entry | `synapse/inference/registry.py` | Registered in Ollama backend |

---

## Build Artifacts

| Source | Output | Install path |
|--------|--------|--------------|
| `greenboost.c` | `greenboost.ko` | DKMS / `/lib/modules/…` |
| `greenboost_cuda_shim.c` | `libgreenboost_cuda.so` | `/usr/local/lib/` |
| `greenboost_audit.c` | `libgreenboost_audit.so` | `/etc/ld.so.audit` |

The shim and audit library are compiled with `-march=native -mtune=native -O3 -flto -fvisibility=hidden` targeting the build host CPU. The resulting binaries are intentionally non-portable.

---

## Framework Compatibility

GreenBoost supports 26 tested inference frameworks via the LD_PRELOAD shim. All use the Path A0 → A → B → C fallback chain automatically.

**LLM servers:** Ollama, ExLlamaV3, vLLM, TGI, LM Studio
**Image generation:** Diffusers, SD A1111, InvokeAI, ComfyUI, FLUX.1, SD 3.5, ControlNet, AnimateDiff
**Speech:** whisper.cpp, OpenAI Whisper, Coqui TTS, Piper TTS (ONNX)
**Vision/detection:** YOLOv8, OpenCV, Detectron2
**Quantization/training:** GPTQ, bitsandbytes, DeepSpeed ZeRO-Inference
**Monitoring:** NVIDIA DMS

---

## Kubernetes / Cluster Deployment

For multi-node V100 clusters, GreenBoost integrates with the NVIDIA k8s-dra-driver-gpu via:

- **DRA kubelet plugin** - calls `GB_IOCTL_GET_POOL_INFO_V3` (struct `gb_pool_info_v3`) for machine-readable tier stats consumed by the Prometheus exporter.
- **NVLink pool** - `gb_nvlink_set_ready()` is called by the kubelet plugin after NVML P2P verification; the kernel module updates `total_vram_gb` and exposes it via sysfs.
- **T3 on Lustre** - on cluster nodes, the T3 backing file (`/var/lib/greenboost/t3_store`) is replaced by a Lustre parallel filesystem mount for aggregate petabyte-scale T3 capacity.

See `k8s-deployment/INSTALL_CLUSTER.md` for full cluster installation steps.

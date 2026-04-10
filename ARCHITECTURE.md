# GreenBoost Architecture

**GreenBoost** is a CUDA Memory Orchestrator. It presents every CUDA process with a single virtual GPU device backed by three physical memory tiers, eliminating CPU layer spillover in LLM inference without requiring application code changes.

---

## Component Map

```
┌─────────────────────────────────────────────────────────────────┐
│  User space                                                     │
│                                                                 │
│  ┌─────────────────────────┐   ┌──────────────────────────┐    │
│  │  CUDA app (Ollama, vLLM │   │  Vulkan game (Steam/     │    │
│  │  ExLlamaV3, PyTorch…)   │   │  Proton / native)        │    │
│  └────────────┬────────────┘   └──────────┬───────────────┘    │
│               │ cudaMalloc / cuMemAlloc    │ vkAllocateMemory   │
│  ┌────────────▼────────────┐   ┌──────────▼───────────────┐    │
│  │  libgreenboost_cuda.so  │   │  libVkLayer_greenboost.so │    │
│  │  (CUDA LD_PRELOAD shim) │   │  (implicit Vulkan layer)  │    │
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
│  T3: kernel swap subsystem → /swap_nvme.img                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Memory Tier Hierarchy

| Tier | Physical device | Role |
|------|----------------|------|
| T1 | GPU VRAM | Hot computation — full GPU bandwidth, native CUDA execution |
| T2 | System DDR (pinned, DMA-BUF) | Cold weights — transferred over PCIe on demand |
| T3 | NVMe swap | Frozen pages — evicted by kernel swap subsystem |

All three tiers are invisible to the application. CUDA apps see a single virtual VRAM device whose size is `physical_vram_gb + virtual_vram_gb` (T1 + T2). The IOCTL `GB_IOCTL_GET_INFO` field `total_combined_mb` covers all three.

**0 % CPU spillover mandate.** GreenBoost intercepts `cuDeviceTotalMem_v2` and `nvmlDeviceGetMemoryInfo` to return the full virtual VRAM size. Ollama is further protected by `OLLAMA_NUM_GPU=999`, which forces all layers onto GPU regardless of Ollama's internal VRAM accounting. Computation always executes on the GPU; only data residency moves between tiers.

---

## Components

### 1. Kernel Module — `greenboost.c` → `greenboost.ko`

**Responsibilities:**
- Registers the `/dev/greenboost` character device with an IOCTL interface.
- **Tier 2 allocator:** allocates pinned physical pages (2 MB hugepages by default, 4 K optional) and exports them as DMA-BUF file descriptors (`GB_IOCTL_ALLOC`). The GPU imports these via `cudaImportExternalMemory` without a CPU round-trip.
- **Tier 3 monitor:** allocates swappable 4 K pages; the kernel swap subsystem handles eviction to `/swap_nvme.img`. Disabled by default (`nvme_pool_gb=0`); enabled on systems with NVMe.
- **Watchdog kthread:** monitors free RAM against `safety_reserve_gb` and NVMe swap pressure. Signals userspace via eventfd (`GB_IOCTL_POLL_FD`) when pressure levels change.
- **IDR tracker:** uses Linux's integer ID allocator to track all live `gb_buf` objects. A mutex + spinlock pair protects the pool.
- **sysfs interface** under `/sys/class/greenboost/greenboost/`:
  - `status` — human-readable pool summary
  - `hw_info` — detected hardware
  - `active_buffers` — live DMA-BUF count
  - `active_profile` — name of the loaded profile
- **Reboot/panic notifiers** for graceful teardown.
- **NVLink pool** (`features/nvlink_pool.c`) — optional V100 cluster extension that aggregates multiple GPU VRAMs into a unified T1 pool (see §6 below).

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
| `debug_mode` | Verbose `dmesg` output |
| `active_profile_name` | Set by installer at `insmod` time |

---

### 2. CUDA Shim — `greenboost_cuda_shim.c` → `libgreenboost_cuda.so`

Loaded system-wide via `/etc/ld.so.preload`. A two-stage constructor (`RTLD_NOLOAD` probe) keeps it inert in non-CUDA processes (GDM, shells, systemd helpers, PAM).

#### Intercepted symbols

| Symbol | Purpose |
|--------|---------|
| `cudaMalloc` / `cudaMallocAsync` | Route overflow to T2/T3 |
| `cuMemAllocAsync` | Route overflow (compute capability ≥ 8.0 gate) |
| `cudaFree` / `cuMemFree_v2` / `cuMemFreeAsync` | Release T2/T3 buffers |
| `cuDeviceTotalMem_v2` | Return virtual VRAM size |
| `nvmlDeviceGetMemoryInfo` / `_v3` | Return virtual VRAM stats |
| `dlsym` | Intercept Ollama's runtime GPU API lookups |
| `dlopen` | Strip `RTLD_DEEPBIND` so hooks stay active inside bundled CUDA libs |

#### Overflow allocation paths (tried in order)

| Path | Mechanism | Requirement |
|------|-----------|-------------|
| A0 | `GB_IOCTL_ALLOC` → `cudaImportExternalMemory(OpaqueFd)` | `greenboost.ko` + libcudart ≥ 10.0 |
| A | `mmap` → `GB_IOCTL_PIN_USER_PTR` → `cuMemHostRegister(DEVICEMAP)` | `greenboost.ko` |
| B | `mmap` (2 MB huge) → `cuMemHostRegister(DEVICEMAP)` | No kernel module needed (containers, VMs, WSL2) |
| C | `cuMemAllocManaged` + `cuMemAdvise` prefetch hints | Last resort / UVM |

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

If KV cache already occupies T1, the effective reserve collapses proportionally — eliminating double-counting so weights are not unnecessarily pushed to T2.

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
| `GREENBOOST_DISABLE` | — | Set to 1 to opt out a single process |
| `GREENBOOST_DEBUG` | 0 | Verbose stderr logging |

---

### 3. LD_AUDIT Gatekeeper — `greenboost_audit.c` → `libgreenboost_audit.so`

Installed to `/etc/ld.so.audit`. Intercepts `la_objopen()` at the dynamic linker level.

Triggers shim injection only when `libcuda.so.*` or `libcudart.so.*` appears in the process link map. All other processes (GDM, PAM, cups, snap-confine, shells) are left completely untouched — the AppArmor blast radius is a single read-only open of the audit library.

**Exclusion rules** (cached after first call):
- `GREENBOOST_DISABLE=1` — explicit opt-out.
- `PRESSURE_VESSEL_RUNTIME` set — nested LD_AUDIT inside Pressure Vessel (Proton sandbox); the outer `wine64` process already has the shim.

Steam / Proton / Wine processes are **not** excluded. Wine's `nvcuda.dll` calls host `libcuda.so` via the PE/ELF bridge; the shim intercepts at the Linux ELF level.

---

### 4. Vulkan Layer — `greenboost_vulkan_layer.c` → `libVkLayer_greenboost.so`

Implicit Vulkan layer, activated by `GREENBOOST_VULKAN=1`. Install manifest at `/etc/vulkan/implicit_layer.d/VkLayer_greenboost.json`.

**Two hooks:**

| Hook | Effect |
|------|--------|
| `vkGetPhysicalDeviceMemoryProperties[2[KHR]]` | Inflates device-local heap to the same virtual VRAM size as the CUDA shim, so Vulkan games choose the correct quality presets and texture budgets |
| `vkAllocateMemory` | On `VK_ERROR_OUT_OF_DEVICE_MEMORY` for allocations ≥ 64 MB, retries from T2 DDR via DMA-BUF import (`VK_KHR_external_memory_fd`). Best-effort — silently returns the original error if `/dev/greenboost` is absent |

Thread-safety: the internal mutex guards only the dispatch-table arrays; all Vulkan calls are made outside the lock to prevent deadlock.

---

### 5. IOCTL Interface — `greenboost_ioctl.h`

Single header usable from both kernel and userspace. Magic byte: `'G'`.

| IOCTL | Cmd | Direction | Purpose |
|-------|-----|-----------|---------|
| `GB_IOCTL_ALLOC` | 1 | IOWR | Allocate pinned T2 buffer → DMA-BUF fd |
| `GB_IOCTL_GET_INFO` | 2 | IOR | Read pool statistics (`struct gb_info`) |
| `GB_IOCTL_RESET` | 3 | IO | Reset internal state |
| `GB_IOCTL_MADVISE` | 4 | IOW | Advise eviction priority (cold/hot/freeze/T1-prefer) |
| `GB_IOCTL_EVICT` | 5 | IOW | Force-push T2 buffer to T3 swap |
| *(cmd 6)* | 6 | — | ABI gap — do not reuse |
| `GB_IOCTL_POLL_FD` | 7 | IOW | Register eventfd for pressure notifications |
| `GB_IOCTL_PIN_USER_PTR` | 8 | IOWR | Pin existing userspace VA → DMA-BUF fd (Path A) |
| `GB_IOCTL_SET_KV_RESERVE` | 9 | IOW | Update T1 KV reserve at runtime |
| `GB_IOCTL_SET_TURBOQUANT` | 10 | IOW | Configure TurboQuant KV compression |
| `GB_IOCTL_SET_POOL_CAP` | 11 | IOWR | Resize T2 pool cap dynamically |
| *(cmd 12–13)* | 12–13 | — | ABI gaps — do not reuse |
| `GB_IOCTL_GET_POOL_INFO_V3` | 14 | IOR | Machine-readable pool info for k8s DRA / Prometheus |
| `GB_IOCTL_RESET_PHASE` | 15 | IO | Increment phase_reset_seq (Synapse CLI uses before model swap) |
| `GB_IOCTL_RELEASE_PID` | 16 | IOW | Release all T2/T3 buffers owned by a PID |

**Allocation flags** (`gb_alloc_req.flags`):

| Flag | Meaning |
|------|---------|
| `GB_ALLOC_WEIGHTS` | Model weight tensor |
| `GB_ALLOC_KV_CACHE` | KV cache — never spills to T3, auto-frozen in T2 LRU |
| `GB_ALLOC_ACTIVATIONS` | Ephemeral activation buffer |
| `GB_ALLOC_FROZEN` | Never evict from T2 |
| `GB_ALLOC_NO_HUGEPAGE` | Force 4 K pages |
| `GB_ALLOC_T1_PRIORITY` | KV-like; moved to LRU head — weight bufs evicted first |
| `GB_ALLOC_KV_COMPRESSED` | TurboQuant-compressed KV buffer |

---

### 6. NVLink Pool — `features/nvlink_pool.{h,c}`

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
         └─ Path C:  cuMemAllocManaged (UVM)   ← last resort
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

## Gaming / GreenBoost Proton

### GreenBoost Proton — Steam Compatibility Tool (`greenboost_proton_wayland/`)

GreenBoost Proton is a **Steam compatibility tool**, not a wrapper. It is a full Proton build registered in `~/.steam/root/compatibilitytools.d/` where:

- `files/` and `protonfixes/` are **symlinks** to the upstream pre-built Proton Experimental binaries at `~/Dev/Proton Experimental/`. This means it always runs the latest upstream Wine, VKD3D-Proton, and DXVK without requiring a separate GreenBoost Proton update.
- The `proton` Python script is **patched**: after `init_session()`, it calls `gb_detect_nvidia()` to probe the GPU at runtime (via `nvidia-smi`, `lspci`, `vulkaninfo`) and then `gb_apply_greenboost()` to inject the environment variables below.

| Variable | Value | Effect |
|----------|-------|--------|
| `GREENBOOST_VULKAN` | `1` | Activates `VK_LAYER_GREENBOOST_memory` — inflates reported VRAM to the virtual size and routes overflow to T2 DDR via DMA-BUF |
| `PROTON_ENABLE_WAYLAND` | `1` | Native Wayland window (no XWayland overhead) |
| `VKD3D_DEBUG` | `warn` | VKD3D-Proton warnings routed to `greenboost vulkan` dashboard |
| `VKD3D_CONFIG` | `dxr,dxr11` | DirectX Raytracing — **enabled only if GPU reports RT support** |
| `DXVK_ENABLE_NVAPI` | `1` | DLSS, Frame Generation, Reflex — **enabled only on NVIDIA** |
| `PROTON_LOG` + `PROTON_LOG_DIR` | `1` + `~/.local/share/greenboost/proton-logs` | Log routing for the Vulkan dashboard |

Nothing is hard-coded — all GPU capability checks run on every game launch.

**Installation:**
```bash
cd ~/Dev/greenboost/greenboost_proton_wayland
./install.sh
```
Creates the symlink in `compatibilitytools.d/`, then: Steam → restart → right-click game → Properties → Compatibility → **GreenBoost Proton**.

---

## Gaming / Vulkan Shader Boost

`greenboost-shader-boost.service` — a root daemon that polls every second for `fossilize_replay` workers (Steam Proton shader pre-compilation) and applies three boosts on first detection:

| Action | Effect |
|--------|--------|
| `renice -5` | Elevates above all nice-0 background tasks |
| `ionice -c2 -n0` | Best-effort I/O at highest priority |
| `taskset 0-<pcores_max>` | Pins to P-cores (auto-detected from sysfs `core_type`) |

This is orthogonal to the CUDA shim — `fossilize_replay` is a Vulkan process and is not hooked by the shim.

---

## Build Artifacts

| Source | Output | Install path |
|--------|--------|--------------|
| `greenboost.c` | `greenboost.ko` | DKMS / `/lib/modules/…` |
| `greenboost_cuda_shim.c` | `libgreenboost_cuda.so` | `/usr/local/lib/` |
| `greenboost_audit.c` | `libgreenboost_audit.so` | `/etc/ld.so.audit` |
| `greenboost_vulkan_layer.c` | `libVkLayer_greenboost.so` | `/usr/local/lib/` |

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
**Steam/Proton:** Supported via Wine PE/ELF bridge; CUDA calls intercepted at the Linux `libcuda.so` level

---

## Kubernetes / Cluster Deployment

For multi-node V100 clusters, GreenBoost integrates with the NVIDIA k8s-dra-driver-gpu via:

- **DRA kubelet plugin** — calls `GB_IOCTL_GET_POOL_INFO_V3` (struct `gb_pool_info_v3`) for machine-readable tier stats consumed by the Prometheus exporter.
- **NVLink pool** — `gb_nvlink_set_ready()` is called by the kubelet plugin after NVML P2P verification; the kernel module updates `total_vram_gb` and exposes it via sysfs.
- **T3 on Lustre** — on cluster nodes, the NVMe swap (`/swap_nvme.img`) is replaced by a Lustre parallel filesystem mount for aggregate petabyte-scale T3 capacity.

See `k8s-deployment/INSTALL_CLUSTER.md` for full cluster installation steps.

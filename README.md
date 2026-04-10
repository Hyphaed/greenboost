## 🧬 GreenBoost : CUDA Memory Orchestrator for NVidia GPUs

**Author:** Ferran Duarri
**License:** GPL v2 (opensource)
**Version:** 2.8
**Changelog:** [changelog.md](changelog.md)
**Documentation:** [documentation.md](documentation.md)

<div align="center">

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/greenboost)

</div>

---

> **Disclaimer:** GreenBoost is an independent open-source project and is not affiliated with, endorsed by, or sponsored by NVIDIA Corporation. NVIDIA, CUDA, GeForce, and RTX are trademarks of NVIDIA Corporation.

---

## 📦 Installation

```bash
git clone https://gitlab.com/IsolatedOctopi/greenboost.git
cd greenboost
sudo ./greenboost_setup.sh
```
---

**GreenBoost does not replace or modify the official NVIDIA kernel drivers**. It loads as a completely independent kernel module alongside them and works at the CUDA allocation layer.

Thanks to all the contributors and to the opensource community. Without them GreenBoost would not be possible.

---

## 🛠️ How it works

Greenboost orchestrates memory flow and extends GPU's addressable memory transparently, no application changes required.

## 🏗️ Kernel module : `greenboost.ko`

Allocates pinned DDR pages using the buddy allocator (2 MB compound pages for efficiency)
and exports them as DMA-BUF file descriptors. The GPU imports these pages as CUDA external
memory via `cudaImportExternalMemory`. From CUDA's perspective, those pages look like
device-accessible memory, it does not know they live in system RAM. The PCIe link handles
the actual data movement (~32 GB/s on PCIe 4.0 x16, ~64 GB/s on PCIe 5.0 x16).

Run `greenboost status` to monitor usage live. A watchdog kernel thread monitors RAM and NVMe pressure and signals userspace before
things get dangerous.

## 🔗 CUDA shim : `libgreenboost_cuda.so`

Injected via `LD_PRELOAD`. Intercepts `cudaMalloc`, `cudaMallocAsync`, `cuMemAllocAsync`,
`cudaFree`, and `cuMemFree`. Small allocations pass straight through to the real CUDA
runtime. Large ones, KV cache, model weights overflowing VRAM, are redirected to the
kernel module and imported back as CUDA device pointers.

One tricky part: Ollama resolves GPU symbols via `dlopen` + `dlsym` internally, which
bypasses `LD_PRELOAD` entirely for those symbols. The shim also intercepts `dlsym` itself
and returns hooked versions of `cuDeviceTotalMem_v2` and `nvmlDeviceGetMemoryInfo`. Without
this, Ollama sees only GPU VRAM and schedules the spillover layers that do not fit into GPU VRAM to be computed by CPU + System RAM.

## 🤖 Heuristic KV cache detection and prioritization

The shim contains a built-in mechanism that classifies every large allocation automatically:

```
GB_PHASE_INIT → GB_PHASE_MODEL_LOAD → GB_PHASE_INFERENCE → GB_PHASE_STEADY
```

During `INFERENCE` and `STEADY` phases, large allocations (≥ 64 MB) receive `GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY` flags. The kernel module auto-freezes them in T2 LRU and refuses T3 spill, keeping the KV cache in the fastest available tier.

The shim tracks how much KV cache has already landed in T1 VRAM (`g_kv_allocated_t1_bytes`) and collapses the pre-allocation reserve proportionally as KV fills VRAM. 

This mechanism can be tuned further by sending flags to Greenboost to indicate which are KVCache memory chunks. Tests are being done through a developer tool under the name "synapse-cli", this tool hasn't been opensourced yet.

## 📊 Memory Pool, memory tiers

*examples based on my hardware;

T1 GPU VRAM : NVidia RTX GPU GDDR7 : 12 GB : ~336 GB/s : Hot layers, active compute 

Holds the hot layers, the phase detector keeps KV cache here whenever VRAM has room, collapsing the reserve as allocations land. 

T2 System DDR : System RAM pool : 53 GB : ~32 GB/s ~64 GB/s PCIe 4/5 : KV cache, cold weights 

Holds the rest of the model and KV cache overflow at long contexts. 

T3 NVMe : NVMe swap : auto-sized : ~1.8 GB/s : Safety overflow (rarely hit) 

Is a safety net, at normal working context lengths the model fits comfortably in T1+T2 and T3 is rarely touched.

## 🧬 What GreenBoost is not

- It is **not** a replacement for the NVIDIA driver. `nvidia.ko`, `nvidia-uvm.ko`, and all NVIDIA official modules continue to run exactly as normal. GreenBoost loads beside them.
- It is **not** a virtual GPU. It does not expose a new GPU device or change how compute works. It only affects how CUDA memory allocations are routed.
- It is **not** a hack around driver restrictions. The DMA-BUF + external memory import path it uses is a documented CUDA feature.
- It does **not** work without the NVIDIA driver installed.

---

## 🐳 Using GreenBoost inside containers, VMs, and WSL2

On bare metal, the greenboost kernel module  handles everything. For environments where `greenboost.ko` cannot be loaded; Docker, LXC, KVM guests, WSL2, shared HPC clusters, the CUDA shim automatically falls back to Path B: `cuMemHostRegister(DEVICEMAP)`, which pins anonymous system RAM pages directly through the CUDA driver with no kernel module required. Check Jerry Nguyen's contribution (see Contributors), for full explanation of path selection, bandwidth trade-offs, and per-environment instructions see **[container_vm_mode.md](container_vm_mode.md)**.

---

## 🛡️ GreenBoost Memory Orchestration Vs CPU Spillover Offload

I really like the concept of the CPU handling some tensor computation, the idea is elegant.
The problem is that the compute power simply isn’t there, current CPUs we find on workstations were not made specifically for tensor computation.

GreenBoost creates and orchestrates a memory pool, T1 (GPU VRAM) + T2 (pinned system DDR) + T3 (NVMe safe fallback) into a single coherent address space. 
The CUDA shim intercepts every cudaMalloc/cudaFree, so any CUDA process sees the combined GPU VRAM + pinned system DDR memory and can address weights directly via DMA-BUF, allowing the GPU to use the entire memory pool and perform all computation on tensor cores.

CPU spillover offload, by contrast, runs only inside the LLM process. The OS still reports a GPU VRAM ceiling, and spilled pages remain hidden behind a per-application paging policy.
CPU spillover offload must perform those same operations on CPU AVX units, which are not designed for massive batch tensor computation and quickly become the bottleneck.

---

## 🎮 Gaming — Vulkan Layer

Grenboost was meant to let large language models run entirely on the GPU with zero CPU spillover. 
Gaming support had been added in v2.8 as an early beta feature.S

GreenBoost reports an inflated VRAM total (T1 GPU VRAM + T2 System DDR) to Vulkan. 
Games that would otherwise hit a VRAM wall, triggering texture pop-in, reduced quality presets, or crashes, can now use the extra headroom.

Then in Steam: right-click a game → **Properties → Compatibility → select "GreenBoost Proton"**.


---

## 🧠 Contributors

- **Alan Sill** ([@alansill](https://gitlab.com/alansill)), contributed `greenboost_setup_rocky.sh`,
  a setup script for Red Hat-based systems (Rocky Linux, AlmaLinux, RHEL).

- **Jerry Nguyen** ([@phubao](https://gitlab.com/phubao)), contributed the
  kernel-module-free overflow path, the `cuMemHostRegister(DEVICEMAP)` approach that
  enables GreenBoost VRAM extension inside containers, VMs, WSL2, and HPC clusters without
  requiring `greenboost.ko`. Integrated as Path B of the blended shim in v2.5.

- **Giuseppe Marco Randazzo** ([@gmrandazzo](https://gitlab.com/gmrandazzo)), contributed
  Debian Trixie support in `greenboost_setup.sh` (v2.6): package dependency mapping for
  Debian testing/unstable and a build-time patch for kernel 6.12+ `MODULE_IMPORT_NS(DMA_BUF)`.

- **Alexey Masolov** ([@alexeymasolov](https://gitlab.com/alexeymasolov)), contributed the
  `cuMemHostGetDevicePointer_v2` fix (v2.6): resolves `CUDA_ERROR_INVALID_CONTEXT (201)` with
  primary-context frameworks (PyTorch, vLLM), restoring the fast PCIe DMA path on modern ML stacks.

---

## 📜 License

GPL v2, open-source. Attribution to Ferran Duarri is required in all forks, derivatives,
and any documentation that references this work.

---

```
Copyright (C) 2026 Ferran Duarri
```
---
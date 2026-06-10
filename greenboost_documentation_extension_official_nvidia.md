# Chapter G. GreenBoost, Host-RAM Tiering for CUDA

> **Disclaimer.** This document is **not** an official NVIDIA publication. It is a
> third-party supplement to the *CUDA C++ Programming Guide*.
> It documents **GreenBoost**, an experimental,
> open-source Linux kernel module and CUDA `LD_PRELOAD` shim that extends the
> tiered-memory model described in Programming Guide §3.2 and §4.1 onto host
> system RAM and (optionally) remote machines. GreenBoost is created by Ferran
> Duarri and is opensource licensed (GPL v2 ), see `LICENSE` in the
> source tree. The CUDA, NVIDIA, NVML, NVTX, Hopper, Grace, Ada, Blackwell, and
> Tegra names are trademarks of NVIDIA Corporation. GreenBoost is not affiliated
> with, endorsed by, or supported by NVIDIA.

This chapter is intended to be readable in the same register as Programming
Guide chapters 3 (CUDA Programming Interface) and 4 (CUDA Features). Where the
Programming Guide describes an NVIDIA-published primitive that GreenBoost
relies on, the relevant section number is cross-referenced. Where GreenBoost
introduces behavior that is **not** documented by NVIDIA, typically because it
combines documented primitives in ways NVIDIA never described, or alters what
the runtime *reports* about devices, that fact is called out explicitly.

---

## G.1. Overview

A CUDA application on a desktop GPU is bounded by VRAM. A 12 GB RTX 5070,
 a 24 GB RTX 4090: when a model or dataset exceeds those sizes,
the application either refuses to load (`cudaErrorMemoryAllocation` /
`CUDA_ERROR_OUT_OF_MEMORY`) or falls back to a host-CPU code path. The
host-CPU fallback runs tensor math on AVX/AVX2/AVX-512 instructions inside the
CPU, leaving the GPU idle for those layers and capping throughput at
single-digit-tok/s for typical LLM workloads.

GreenBoost replaces that fallback. It presents every CUDA process with **a
single virtual GPU device** whose reported total memory equals
`VRAM + system_DDR_pool (+ NVMe pool) (+ remote-cluster pool)`. All allocations
that exceed physical VRAM are transparently routed onto host system RAM (and
optionally NVMe or other machines), but in a way that keeps **all tensor
computation on the GPU's streaming multiprocessors (SMs)**. The CPU never
executes tensor math.

The trick is to take three documented NVIDIA primitives and combine them in a
configuration the Programming Guide does not describe:

| Path | Primitive (documented in Programming Guide) | GreenBoost role |
|---|---|---|
| A | `cuImportExternalMemory(OpaqueFd)` + DMA-BUF (§3.2.7) | Pinned host DDR exported by kernel module, imported by CUDA |
| B | `cuMemHostRegister` + `cuMemHostGetDevicePointer` (§3.2.5) | Pinned mapped host memory, used when no kernel module is present |
| C | `cuMemAllocManaged` + `cuMemAdvise` (§4.1, §4.1.4) | Managed memory anchored to CPU by hint |

The combination, host-RAM-backed allocations that the application receives via
the standard `cudaMalloc` / `cuMemAlloc_v2` APIs, transparently, with the
runtime reporting an inflated `cuDeviceTotalMem`, is what is not documented
by NVIDIA. The individual building blocks are.

Programmer perspective: an application calls `cudaMalloc(&p, 33 GB)` on a
12 GB GPU and the call succeeds. The returned pointer `p` is a valid CUDA
device pointer. Subsequent `cudaMemcpy`, `cuLaunchKernel`, `nvmlDeviceGetMemoryInfo`
all behave consistently with that allocation. No application code changes.

---

## G.2. The Problem GreenBoost Solves

### G.2.1. Why CPU spillover is slow

Two widely deployed inference servers, Ollama (via `n_gpu_layers`) and vLLM
(via `gpu_memory_utilization` + `cpu_offload_gb`), handle VRAM exhaustion by
moving some model layers onto the **CPU**. This is not the same thing as
GreenBoost does, and the performance gap is large.

**Ollama with partial CPU offload (`n_gpu_layers=N` where `N` is less than the
model's total layer count):**

- Layers `0..N-1` run on GPU SMs.
- Layers `N..L-1` run on CPU using AVX2/AVX-512 matmul kernels in `llama.cpp`.
- Per-token latency is bounded by the slowest tier, and the CPU tier is the
  slowest by orders of magnitude.
- A 12 GB GPU running a 70B-parameter Q4_K_M model (~40 GB) with the bottom
  ~12 GB of layers on GPU and the top ~28 GB on CPU typically lands at
  **3–8 tok/s on a modern desktop CPU** (Ryzen 9 / i9-class).
- CPU power draw at sustained inference exceeds 100 W; thermal throttling
  becomes a factor.

**vLLM with `cpu_offload_gb`:**

- KV cache and a portion of model weights live in pinned host memory.
- Layer execution per token requires a `cudaMemcpyAsync(D2H)` for the
  prior layer's outputs and a `cudaMemcpyAsync(H2D)` of the spilled weights
  for the next layer.
- Effective batch size collapses because each request that touches a spilled
  layer pays the round-trip on every token.
- Tensor parallelism across two GPUs is supported, but only across GPUs, the
  CPU spillover path is single-threaded by design.

In both cases, the **GPU sits idle waiting for the CPU** for the spilled
fraction of every token. The PCIe bus is mostly unused. System DDR is read
sequentially by `memcpy` rather than by the GPU's parallel memory controllers.

### G.2.2. What GreenBoost does instead

GreenBoost keeps all tensor execution on the GPU and uses one of the three
paths described in §G.1 to make host RAM accessible to GPU SMs directly:

- Allocation: `cudaMalloc(&p, 33 GB)` returns a device pointer that points
  into pinned host RAM (Path A or B) or into managed UVM hinted to host
  (Path C). VRAM headroom is unchanged.
- Kernel launch: `cuLaunchKernel` receives this pointer in its argument
  buffer and runs the SMs against it. The GPU's L2 cache and memory
  controllers issue PCIe reads to host RAM transparently.
- PCIe bandwidth on a PCIe 4.0 x16 link is ~32 GB/s theoretical, ~25 GB/s
  sustained. PCIe 5.0 x16 doubles that. The GPU reads DDR at PCIe rates,
  not at memcpy rates.
- The CPU executes zero tensor math.

The result on the same 12 GB-GPU / 70B-Q4 workload is typically **25–40 tok/s
versus Ollama's 3–8 tok/s**, a 4–6× speedup, with CPU sitting at idle and the
GPU sustaining near-VRAM-bound throughput because the working set rotates
through VRAM via PCIe rather than being computed by AVX.

> Models with attention-heavy decode (very long contexts) bottleneck on KV
> cache bandwidth and benefit less. Models with weight-heavy MLPs benefit
> most. The TurboQuant K/V compression layer (`gb_attn.py`) further reduces
> the attention-bandwidth bottleneck and is independently composable.

---

## G.3. Architecture

GreenBoost is split across kernel and userspace:

- **`greenboost.ko`**, Linux kernel module. Exports DMA-BUF descriptors backed
  by `alloc_pages`-allocated huge pages, owns a 3-tier (T1 VRAM bookkeeping /
  T2 DDR pool / T3 NVMe backing file) bookkeeping IDR, and exposes ioctls and
  sysfs. **This is a Linux kernel module; it is not an NVIDIA module and does
  not link against any NVIDIA headers. It interacts with the NVIDIA driver
  only through the documented DMA-BUF import API on the CUDA side.**
- **`libgreenboost_cuda.so`**, `LD_PRELOAD` shim that intercepts the CUDA
  Driver API (`libcuda.so.1`) and CUDA Runtime API (`libcudart.so.12+`).
  Intercepted symbols include `cudaMalloc`, `cuMemAlloc_v2`,
  `cuMemHostRegister`, `cuMemAllocManaged`, `cuMemFree_v2`, `cuLaunchKernel`,
  `cuMemPrefetchAsync`, `cuMemGetInfo_v2`, `cuDeviceTotalMem`, and the
  `nvmlDeviceGetMemoryInfo` family.
- **`libgreenboost_audit.so`**, `LD_AUDIT` gatekeeper that injects the shim
  cleanly into multi-process apps without `LD_PRELOAD` propagation issues.
- **`greenboost-netd`**, optional TCP daemon for cluster mode. Each remote
  machine running this daemon contributes its own VRAM + DDR + NVMe to a
  shared "device 0" pool from the perspective of the host application.

The application's view of the GPU comes entirely from the intercepted
`cuDeviceTotalMem` and `cuMemGetInfo_v2` calls. From the application's
perspective:

```
nvidia-smi:           shows physical 12 GB
GreenBoost banner:    shows 12 GB physical + 49 GB T2 + 32 GB T3 = 93 GB virtual
cuMemGetInfo_v2:      returns free=92 GB, total=93 GB
nvmlDeviceGetMemoryInfo: total=93 GB
ollama log:           "GPU device 0: 93 GB VRAM"
```

The application schedules onto a "93 GB GPU" and the model fits.

---

## G.4. The Three Allocation Paths

GreenBoost tries paths in order of expected performance and falls back on
failure. Path selection is driven by hardware detection at shim init
(`cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)`) and by
the presence of `/dev/greenboost`.

### G.4.1. Path A, DMA-BUF Pinned DDR via `greenboost.ko`

**When used.** Hopper (cc 9.x), Grace, server-SKU Blackwell, and any
non-PCIe-isolated GPU where the IOMMU permits the GPU to access pinned host
RAM via the kernel's DMA-BUF facility. Also pre-Hopper desktop SKUs (Ampere,
Ada) where Path B works but Path A gives lower latency on large allocations.

**How it works.** The shim's `cudaMalloc(33 GB)` path:

1. Issues `ioctl(/dev/greenboost, GB_IOCTL_ALLOC, {size: 33 GB, flags})`.
2. The kernel module allocates 33 GB of host RAM as hugepages (or 4 KB pages
   if hugepages are exhausted), wraps it in a `struct dma_buf` via
   `dma_buf_export`, and returns a file descriptor.
3. The shim imports the fd into CUDA:
   - **G.4.1.1 Sub-method 1, zero-copy via `cuImportExternalMemory(OpaqueFd)`.**
     This is the documented NVIDIA API for DMA-BUF interop (Programming Guide
     §3.2.7 "External Resource Interoperability"). The CUDA driver builds its
     own IOMMU mapping from the fd's `sg_table`. No `mmap` round-trip. Skipped
     on Blackwell (cc ≥ 12) where the driver currently refuses `OpaqueFd`
     fds backed by anonymous host pages.
   - **G.4.1.2 Sub-method 2, pinned via `cuMemHostRegister(DEVICEMAP)`.**
     The shim `mmap`s the fd, calls `ioctl(GB_IOCTL_PIN_USER_PTR)` to take
     a kernel reference on the pages, then registers the mapped range with
     `cuMemHostRegister`. Fallback when sub-method 1 is unavailable, and
     primary for small allocations (< 4 MB) where DMA-BUF import overhead
     dominates.
4. Returns the CUDA device pointer to the caller.

**Reference to NVIDIA documentation.** Programming Guide §3.2.7
("External Resource Interoperability") documents `cuImportExternalMemory` and
its allowed handle types (`OpaqueFd`, `OpaqueWin32`, `NvSciBuf`,
`OpaqueFdDmabuf` on certain platforms). GreenBoost uses `OpaqueFd` with a
DMA-BUF the kernel module exported. NVIDIA documents the API surface; what
is undocumented is **using it to expose pinned anonymous host pages backed
by a non-NVIDIA Linux kernel module** as a general-purpose extension of
device VRAM.

**Failure mode and detection.** On Blackwell consumer PCIe the
`cuImportExternalMemory` import returns `CUDA_ERROR_NOT_SUPPORTED` for
anonymous-page-backed fds. The shim falls back to Path C.

### G.4.2. Path B, `cuMemHostRegister` Direct

**When used.** Container runtimes that do not pass `/dev/greenboost` through
to the container; virtual machines without IOMMU passthrough of the GPU;
WSL2; and any system where the kernel module cannot be loaded but the GPU
supports pinned host memory mapping (which is essentially every CUDA GPU
since cc 2.0 when `canMapHostMemory == 1`).

**How it works.** The shim allocates pinned host memory directly without the
kernel module:

```c
posix_memalign(&h_ptr, 2 MB, size);             // hugepage-aligned
mlock(h_ptr, size);                              // page-pin via mlock2 if available
cuMemHostRegister(h_ptr, size,
                  CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_PORTABLE);
cuMemHostGetDevicePointer(&d_ptr, h_ptr, 0);     // GPU-visible device VA
```

The returned `d_ptr` is a CUDA device pointer. GPU SMs access it over PCIe,
identical to Path A from the application's perspective.

**Reference to NVIDIA documentation.** Programming Guide §3.2.5
("Page-Locked Host Memory" → "Mapped Memory") documents `cuMemHostRegister`,
`cuMemHostGetDevicePointer`, and the `canMapHostMemory` device attribute.
This is the most directly-documented of GreenBoost's three paths. What is
undocumented is **using it at 30+ GB scale for general overflow**, NVIDIA
documents `cuMemHostRegister` as a tool for zero-copy data transfer of
*specific* user buffers, not as a backing store for `cudaMalloc`.

### G.4.3. Path C, Managed UVM with `PREFERRED_LOCATION=CPU`

**When used.** Blackwell desktop / consumer PCIe (cc ≥ 12.0), specifically the
RTX 5070/5080/5090 family. On these SKUs Paths A and B - and also
`cuMemCreate(HOST / HOST_NUMA_CURRENT) + cuMemMap + cuMemSetAccess` - return
device pointers that are **DMA-only**, usable by the copy engines but rejected by
the SMs (`CUDA_ERROR_INVALID_RESOURCE_HANDLE`) for direct kernel access. Path C
sidesteps that constraint.

**ggml-cuda VMM pool compatibility.** Ollama 0.18+ uses ggml-cuda's VMM pool
(`ggml_cuda_pool_vmm`) which allocates via `cuMemCreate + cuMemMap` rather than
`cudaMalloc`. On Blackwell, this pool's HOST_NUMA_CURRENT fallback is DMA-only
and crashes on the first kernel dispatch (e.g. `IM2COL_3D` during model load).
GreenBoost works around this by **intercepting `cuMemAddressReserve`** and
returning `CUDA_ERROR_NOT_SUPPORTED` on cc ≥ 12 (unless
`GREENBOOST_BLACKWELL_ALLOW_VMM=1` is set). `ggml_cuda_vmm_available()` tests
`cuMemAddressReserve` during backend init; on failure it selects the legacy
`ggml_cuda_pool_leg` pool which uses `cudaMalloc`. GreenBoost's `cudaMalloc`
hook then routes T1-overflow through `gb_vmm_t2_alloc_blackwell_managed()` -
the only SM-accessible T2 path on desktop Blackwell PCIe. Three additional
guards enforce this on any remaining call sites:
- `cuMemAddressReserve` hook: `CUDA_ERROR_NOT_SUPPORTED` on cc ≥ 12 (primary)
- `gb_overflow_alloc_ex`: zerocopy/hostnuma fallbacks return clean OOM on
  cc ≥ 12 when `GREENBOOST_BLACKWELL_ALLOW_VMM` is unset (defence-in-depth)
- `cuMemCreate` hook: HOST_NUMA_CURRENT fallback returns OOM on cc ≥ 12 when
  `GREENBOOST_BLACKWELL_ALLOW_VMM` is unset (defence-in-depth)

**ATS-capable Blackwell server SKUs (Blackwell H100, GH200):** set
`GREENBOOST_BLACKWELL_ALLOW_VMM=1` to re-enable all VMM paths; HOST_NUMA_CURRENT
IS SM-accessible on ATS-equipped interconnects.

**How it works.** The shim allocates managed memory and pins it to host RAM
via documented hints:

```c
cuMemAllocManaged(&d_ptr, size, CU_MEM_ATTACH_GLOBAL);
cuMemAdvise(d_ptr, size, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, CU_DEVICE_CPU);
cuMemAdvise(d_ptr, size, CU_MEM_ADVISE_SET_ACCESSED_BY,        device);
```

The returned `d_ptr` is a managed-memory pointer that GPU SMs can dereference
directly. The `SET_PREFERRED_LOCATION=CPU` advice anchors the physical pages
to host RAM; `SET_ACCESSED_BY=device` pre-creates GPU page-table mappings so
the first SM touch does not stall on a fault.

**Reference to NVIDIA documentation.** Programming Guide §4.1
("Unified Memory") documents managed memory; §4.1.4 ("Performance Tuning")
documents `cuMemAdvise` and the full set of advice flags including
`SET_PREFERRED_LOCATION`, `SET_ACCESSED_BY`, and `SET_READ_MOSTLY`. The
Programming Guide explicitly states that `cudaMemPrefetchAsync` can
**override** the preferred location:

> *"cudaMemPrefetchAsync may override [the preferred location] and allow the
> memory to migrate."* , Programming Guide §4.1.4.2

This single sentence is load-bearing for GreenBoost Path C correctness. PyTorch,
diffusers, and most modern CUDA-using libraries call `cuMemPrefetchAsync`
implicitly during model loading and per-step warm-up; if those prefetches
target `dst=device`, the host-pinned pages will migrate onto the GPU and the
design collapses. GreenBoost's shim therefore **intercepts both
`cuMemPrefetchAsync` and `cudaMemPrefetchAsync`** and short-circuits the call
on managed-UVM allocations it owns:

```c
CUresult cuMemPrefetchAsync(CUdeviceptr dptr, size_t count, CUdevice dst, CUstream s) {
    if (bvmm_ht_peek_type(dptr, &type) && type == BVMM_TYPE_MANAGED)
        return CUDA_SUCCESS;        // no-op: pages stay on host
    return real_cuMemPrefetchAsync(dptr, count, dst, s);
}
```

The application sees `CUDA_SUCCESS`; the pages do not move.

**Why not just rely on the hint without intercepting prefetch?** Because the
Programming Guide itself warrants that prefetch overrides the hint. Relying on
the hint alone is reading half of §4.1.4.2.

**Detection at allocation time.** The shim now verifies the `cuMemAdvise`
return value for `SET_PREFERRED_LOCATION`. If the driver rejects the hint -
which can happen on devices with `concurrentManagedAccess=0`, the
allocation is freed and the call falls back to Path A/B. This means Path C
allocations that succeed are *known* to be anchored.

---

## G.5. Why GreenBoost Is Better Than Ollama / vLLM CPU Spillover

The mechanism difference is the throughput difference. The table below uses a
representative workload (a 70B-parameter Q4 model on a 12 GB GPU + 64 GB
host DDR; PCIe 4.0 x16; Ryzen 7950X CPU):

| Configuration | Tensor compute device | DDR access bandwidth | Typical tok/s |
|---|---|---|---|
| Ollama `n_gpu_layers=15` (spillover) | CPU for layers 16–80 | ~30 GB/s AVX-512 memcpy | 3–8 |
| vLLM `cpu_offload_gb=28` | CPU layers + D2H/H2D per step | ~30 GB/s memcpy + PCIe round-trip | 2–6 |
| GreenBoost (Path A on Hopper, Path C on Blackwell) | **GPU SMs**, reading DDR via PCIe | ~22 GB/s PCIe 4.0 x16 effective | **25–40** |

The PCIe-effective bandwidth of Path A/B/C (~22 GB/s) is comparable to AVX-512
memcpy bandwidth (~30 GB/s), but the multiplier comes from **GPU SMs being
~80× faster at the matmul itself** than the CPU's AVX units. The CPU's
bandwidth advantage is irrelevant because the CPU is bottlenecked on FLOPS,
not bytes/s, on this workload.

Two further qualitative advantages:

- **Power.** With CPU spillover the CPU pulls 100–150 W sustained. With
  GreenBoost the CPU sits at idle (~10–20 W) and only the GPU draws full
  power. Total system wattage is ~30% lower.
- **Concurrent CPU workloads.** With CPU spillover the inference job
  monopolizes the CPU. With GreenBoost the CPU is free for the rest of the
  system, running a UI, compiling code, serving other requests.

### G.5.3. The same calculus applied to image generation (`diffusers`)

The CPU-spillover analogue in image-generation land is
`diffusers.enable_model_cpu_offload()` and
`diffusers.enable_sequential_cpu_offload()`. Both work by leaving the
inactive pipeline components in host pinned memory and `cudaMemcpyAsync`-ing
them onto the GPU when their turn comes, then evicting them after. On a
Flux / SDXL / SD3-class pipeline that needs ~25–35 GB of BF16 weights on
a 12 GB GPU:

- `enable_model_cpu_offload`: one full-component swap per step. UNet,
  VAE, and text encoders each take ~150–400 ms to host->device-copy on
  PCIe 4.0 x16. With 4-step Klein this dominates total latency; effective
  step rate drops to ~1 step/s.
- `enable_sequential_cpu_offload`: layer-granular swapping. Worse per-step
  latency, lower peak VRAM. Useful only when peak-VRAM is critical.

With GreenBoost loaded, the same pipeline does `pipe.to("cuda")` once at
load time; the working set lives in host RAM via Path C; per-step GPU
kernels read the relevant weights via PCIe directly into the SM caches.
No `cudaMemcpy`. No component swapping. The 4-step Klein generation
finishes in roughly **the same wall-clock as the pure-VRAM baseline on
a 24 GB workstation GPU**, because PCIe-into-cache and VRAM-into-cache
have similar bandwidth characteristics for the SDPA + GEMM workloads
that dominate a denoising step.

---

## G.6. Use Cases

### G.6.1. Image generation with HuggingFace `diffusers`

`art_wizard.sh` is a real work pipeline (used by Ferran Duarri) that exercises
Path C end-to-end against the **HuggingFace `diffusers` library**. It
generates illustrations on a 12 GB RTX 5070, without any application-side code that knows GreenBoost
is present.

The relevant `diffusers` API surface that interacts with GreenBoost:

```python
from diffusers import *
import torch

pipe = *.from_pretrained(
    "youmodelhere",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")                    # ← triggers GreenBoost overflow routing
img = pipe(prompt="...", num_inference_steps=4).images[0]
```

What happens at the CUDA layer when running under `LD_PRELOAD=libgreenboost_cuda.so`:

1. `pipe.from_pretrained(...)` materialises BF16 weights into CPU tensors.
2. `pipe.to("cuda")` walks the pipeline's `_components` (UNet, VAE, T5 text
   encoder, CLIP) and calls `.to("cuda")` on each, which translates to
   `torch.cuda.Tensor` allocations via `cudaMalloc`. Total ~33 GB.
3. Each `cudaMalloc` reaches the shim's hook.
4. The first ~9 GB lands in real T1 VRAM (`real_cudaMalloc` succeeds).
5. Subsequent allocations overflow to Path C: managed UVM hinted to host RAM.
6. **`diffusers` calls `cuMemPrefetchAsync` implicitly** during the
   `.to("cuda")` walk for some components - and this is exactly the
   load-bearing case for Path C's prefetch-interception (§G.4.3). Without
   it, those prefetches would migrate the managed-UVM pages onto the GPU
   and the design collapses to OOM.
7. The diffusion sampling loop (`pipe(...)`) runs entirely on GPU SMs.
   PCIe reads stream the UNet weights for each denoising step from host
   RAM into the GPU's L2 / SM register file.
8. Step latency on Blackwell is ~1.3× the pure-VRAM baseline; the
   dominant cost is PCIe weight streaming, not migration or page faults.

#### `diffusers` features that compose well with GreenBoost

The following `diffusers` patterns work as documented under GreenBoost and
are the recommended way to use the library on a memory-constrained GPU:

- `pipe.to("cuda")` - direct allocation. Works with `art_wizard.sh`. Best
  for inference where the full pipeline fits in `physical_VRAM + host_T2`.
- `pipe.enable_attention_slicing()` - orthogonal to GreenBoost; reduces
  per-step attention working set. Composable.
- `pipe.enable_vae_slicing()` / `enable_vae_tiling()` - same; reduce VAE
  peak memory. Composable.
- `pipe.enable_xformers_memory_efficient_attention()` - composable.
- Component-by-component `.to("cuda")` dispatch (the
  `_pipe_to_cuda_meta_dispatch` pattern used in `art_wizard.sh`'s
  helpers): each component is moved to CUDA in isolation, so each
  triggers its own overflow decision and the peak working set never
  exceeds the GPU. Best for huge pipelines (LTX-Video, Hunyuan-Video).

#### `diffusers` features that are unnecessary under GreenBoost

These are CPU-offload knobs from `diffusers` that exist to work around
VRAM limits and become **redundant** when GreenBoost is loaded - and
should be **disabled** because they actively interfere:

- `pipe.enable_model_cpu_offload()` - moves whole components to/from
  host RAM between steps via `cudaMemcpy(D2H/H2D)`. With GreenBoost the
  components already live in host RAM and the GPU SMs read them via
  PCIe; an extra `cudaMemcpy` round-trip just wastes time. **Disable
  this when running under GreenBoost.**
- `pipe.enable_sequential_cpu_offload()` - worse: moves at the
  layer level, doubling the memcpy traffic. **Disable.**
- `accelerate`'s `device_map="auto"` / `"balanced"` for a *single*-GPU
  case - solves the same problem GreenBoost solves, at higher overhead.
  Set `device_map="cuda"` instead and let GreenBoost handle overflow.

Without GreenBoost the ~25–35 GB model on a 12 GB GPU requires
either `enable_model_cpu_offload()` (~3-5× slower per step due to
component swapping) or aggressive quantization (Q8 weights through
`bitsandbytes`, with quality loss). With GreenBoost, neither is needed -
the pipeline runs in its native BF16 representation and the steps land
near the pure-VRAM-baseline latency.

#### Compatibility notes (`diffusers` ↔ GreenBoost)

- `diffusers` ≥ 0.30 has been tested. Earlier versions work but use
  legacy memcpy patterns that hit slower fall-through paths in the shim.
- `transformers` text encoders loaded by `diffusers` (T5, CLIP) overflow
  through the same paths and require no special handling.
- `safetensors` mmap-loading: orthogonal - GreenBoost intercepts the
  CUDA-side `cudaMalloc`, not the safetensors host-side mmap.
- `torch.compile(pipe.unet)` works and benefits GreenBoost setups
  because Torch-compiled kernels reduce launch overhead, which matters
  more when each kernel reads its weights from host RAM over PCIe.

### G.6.2. Local big-model LLM inference

GreenBoost is designed to make consumer GPUs run models that would otherwise
require workstation hardware:

| Model | Q4 size | Fits on RTX 5070 (12 GB) alone? | Fits with GreenBoost + 64 GB DDR? |
|---|---|---|---|
| Llama-3.1-8B | ~5 GB | yes | yes |
| Mistral-Large-2 (123B) | ~70 GB | no | yes (with NVMe T3 spill) |
| **GLM-4.7 (~106B)** | ~60 GB | no | **yes** |
| **Qwen-3.6 (480B MoE)** | ~280 GB | no | only with cluster (Path A/B/C + T4) |
| **Nemotron-Cascade-2** | ~24 GB cascade | no (alone) | **yes** |
| DeepSeek-V3-Coder | ~370 GB | no | only with cluster |

For the models marked *yes*, GreenBoost makes the difference between "won't
load" and "runs at 20–35 tok/s". For models marked *only with cluster*, the
T4 tier aggregates DDR/VRAM across multiple machines (see §G.6.3).

### G.6.3. Cluster mode, distributed inference across machines

GreenBoost extends the local 3-tier hierarchy onto remote machines running
`greenboost-netd`. From the application's perspective there is still **one**
CUDA device 0; that device's total memory now equals
`local_T1 + local_T2 + local_T3 + Σ feeder_(T1+T2+T3)`.

Key design point: only **one** virtual device is presented to the
application, never multiple. The reason is that Ollama's and llama.cpp's
multi-GPU schedulers assume NVLink/PCIe peer GPUs that can each independently
issue kernel dispatches. A TCP-remoted feeder cannot service direct kernel
dispatch at the latency a peer GPU does. So GreenBoost aggregates feeder
memory into device 0 and uses **data-driven kernel dispatch**: when
`cuLaunchKernel` is called, the shim scans the argument buffer for fake
remote pointers (range `0xAA00…`) and forwards the launch to the feeder that
owns the data.

Practical examples:

- **5070 + 1080 Ti as one logical GPU.** The 5070 contributes 12 GB T1 + 32 GB
  T2; the 1080 Ti contributes 11 GB feeder T1 + 16 GB feeder T2. The shim
  reports 71 GB total. A 60 GB model loads with weights split across both
  machines.
- **Two-node Qwen-3.6 480B Q4.** Two consumer machines, each with 12 GB GPU
  and 128 GB DDR, total ~280 GB aggregated. The model fits.

### G.6.4. Background workloads on consumer hardware

Because CPU usage stays low under GreenBoost (§G.5), the system remains
responsive during inference. This makes it viable to run long-running
batch jobs, like generating 80+ pieces of game art over several hours -
without the desktop becoming unusable. CPU-spillover-based inference
typically blocks all other useful work on the same machine.

---

## G.7. Documented vs. Undocumented Behavior

GreenBoost is built on documented NVIDIA primitives but combines them in
ways NVIDIA does not document. This section makes the boundary explicit.

### G.7.1. Built on documented NVIDIA APIs

The following APIs are used as their NVIDIA documentation specifies. Bugs
should be filed against GreenBoost, not against NVIDIA:

- `cuMemAlloc_v2`, `cuMemFree_v2`, `cudaMalloc`, `cudaFree`
- `cuMemHostRegister`, `cuMemHostUnregister`, `cuMemHostGetDevicePointer`
  (Programming Guide §3.2.5)
- `cuMemAllocManaged`, `cuMemAdvise` with `SET_PREFERRED_LOCATION`,
  `SET_ACCESSED_BY`, `SET_READ_MOSTLY` (Programming Guide §4.1, §4.1.4)
- `cuImportExternalMemory(CUmemAllocationHandleType::OpaqueFd)`
  (Programming Guide §3.2.7)
- `cuLaunchKernel`, `cuLaunchKernelEx`, `cuLaunchCooperativeKernel`
  (Programming Guide §3.2.10)
- `cuMemPrefetchAsync` and `cudaMemPrefetchAsync` (intercepted only;
  unmodified for non-GreenBoost allocations)
- `cuDeviceTotalMem`, `cuMemGetInfo_v2`, `nvmlDeviceGetMemoryInfo` (intercepted
  to report virtual size; unmodified for non-GreenBoost callers if
  `GREENBOOST_DISABLE=1`)
- `LD_PRELOAD`, `LD_AUDIT`, `dlsym(RTLD_NEXT, …)`, standard glibc
  facilities for library interposition.

### G.7.2. Undocumented behavior (works empirically; may break with driver updates)

The following GreenBoost behaviors rely on observed runtime behavior of the
NVIDIA driver in current shipping versions. NVIDIA does not contractually
guarantee any of these and is free to break them:

- **Reporting an inflated `cuDeviceTotalMem` and observing the runtime
  accept allocations up to that size.** The Programming Guide does not say
  the driver enforces a `total_mem` ceiling on `cuMemAlloc`; in practice
  current drivers do not. Future drivers may.
- **DMA-BUF anonymous-page imports yielding SM-accessible device pointers
  on pre-Blackwell hardware.** NVIDIA documents DMA-BUF interop in the
  context of camera/codec interop (e.g. V4L2, OpenMAX); using it as a
  general extension of host-RAM-as-VRAM is empirical.
- **Managed-UVM `SET_PREFERRED_LOCATION=CPU` keeping pages on host
  through the lifetime of an allocation under PCIe-attached Blackwell
  consumer SKUs.** The Programming Guide hints at this (§4.1.4.2) and the
  shim verifies the advice was accepted at runtime, but the hint is not a
  guarantee.
- **`cuLaunchKernel` argument-buffer pointer scan finding fake remote
  pointers reliably.** GreenBoost's data-driven cluster dispatch reads
  the application's argument buffer to identify pointers in the
  `0xAA00…` reserved range. This relies on the application packing
  pointer arguments contiguously in the buffer, which is the convention
  but not a contractual requirement.
- **`cuGetProcAddress` hookability.** Future drivers may resolve hot
  paths through a non-`dlsym`-visible mechanism, bypassing the shim.

### G.7.3. Not undocumented but officially "your problem if you break it"

- LD_PRELOAD into proprietary applications (Steam games, certain ML SaaS
  clients) may violate the application's EULA. GreenBoost is a tool;
  application of the tool is the user's responsibility.

---

## G.8. Detection, Safety, and Telemetry

Because GreenBoost relies on partially-undocumented behavior, the shim
maintains runtime invariants and exposes telemetry so deployments can
detect regressions:

- **Allocation-path bookkeeping.** Every overflow allocation is tagged with
  its origin (`BVMM_TYPE_MANAGED`, `BVMM_TYPE_ZEROCOPY`, `BVMM_TYPE_VMM`,
  Path A pinned, Path A zero-copy, Path B). On free, the corresponding
  cleanup path runs.
- **Path C hint verification.** `cuMemAdvise(SET_PREFERRED_LOCATION=CPU)`
  return value is checked. If rejected, the allocation is freed and the
  caller falls back to Path A/B.
- **Prefetch interception** (§G.4.3) prevents PyTorch/diffusers from
  silently migrating managed-UVM pages onto the GPU.
- **MemAvailable safety floor** (default 6 GB; tunable via
  `GREENBOOST_HOST_RAM_SAFETY_MB`). When `/proc/meminfo`'s `MemAvailable`
  drops below the floor, new T2 allocations are refused. Prevents the
  kernel OOM-killer from firing on legitimate workloads.
- **Shim statistics** at `/run/greenboost/shim_stats`. Path counters,
  per-tier allocated bytes, T2 fragmentation %, KV-cache tracking. The
  `greenboost vitals` and `greenboost cluster` commands render this.
- **NVTX events** for each allocation, free, prefetch-skip, and
  kernel-dispatch event. Visible in Nsight Systems traces if NVTX
  capture is enabled.
- **Prometheus exporter** (`greenboost_exporter.py`) emits per-tier
  bandwidth, allocation counts, T2 pool fragmentation, feeder GPU
  utilization, and PSK auth state for ingestion into Grafana.

---

## G.9. Limitations and Caveats

### G.9.1. What GreenBoost does not change

- **Peer-to-peer (NVLink) bandwidth.** Multi-GPU NVLink topologies are
  unaffected. GreenBoost intercepts host-RAM-spill paths only.
- **Compute capability requirements of the application.** A library that
  requires `sm_90` will not run on `sm_86` because GreenBoost provided
  memory.
- **Kernel correctness.** GreenBoost never substitutes kernel implementations;
  it only routes allocations.

### G.9.2. Known constraints

- **Path C is Blackwell-only.** On pre-Blackwell desktop GPUs Paths A/B are
  used. The shim selects automatically.
- **Path A requires `/dev/greenboost`** and therefore root access at install
  time. Containers and unprivileged VMs use Path B.
- **PCIe bandwidth is the throughput ceiling.** GreenBoost cannot make a
  PCIe 3.0 x4 system as fast as a PCIe 5.0 x16 system. The ceiling is
  fundamental.
- **TurboQuant K/V compression** (`gb_attn.py`) trades a controlled amount
  of quality (typically +0.23% PPL at 4-bit K, 3-bit V) for bandwidth. It
  is opt-in and per-process.
- **Cluster mode (T4)** does not support tensor parallelism within a single
  kernel launch. Layer-parallel and data-parallel patterns work; intra-kernel
  AllReduce across feeders is not implemented.

### G.9.3. Operational risks

- **Driver updates.** A future NVIDIA driver could reject one of the three
  paths. The shim attempts all three in order and falls back; if all three
  fail the application sees `CUDA_ERROR_OUT_OF_MEMORY` (which is correct
  behavior for OOM) and inference workloads that previously fit will fail
  to load.
- **Kernel updates.** `greenboost.ko` is DKMS-managed and rebuilt against
  each new kernel. Kernel APIs around DMA-BUF, kprobes, and hugepages
  have shifted between 5.x and 7.x; the `features/compat.h` shim covers
  4.15–7.x at the time of writing.
- **Application updates.** Libraries that switch from documented to
  undocumented CUDA entry points (e.g. internal `__cudaXxx` symbols) may
  bypass the shim. The shim hooks the exposed `libcudart`/`libcuda`
  surface only.

---

## G.10. Compatibility Matrix

| Component | Minimum | Tested | Notes |
|---|---|---|---|
| CUDA Driver | 525 | 555, 560 | 5070 (Blackwell) needs 555+ |
| CUDA Runtime | 11.8 | 12.4, 12.6, 13.x | Both ABI versions hooked |
| Linux kernel | 4.15 | 5.15, 6.x, 7.0 | `features/compat.h` shims |
| Distros | Debian, Ubuntu, Fedora, RHEL/Rocky, Arch, openSUSE | Ubuntu 24.04, Rocky 9, Arch | Multi-distro via `greenboost_builder.py` |
| Ollama | 0.2 | 0.5+ | Picks up inflated `cuDeviceTotalMem` automatically |
| vLLM | 0.6 | 0.7+ | Works without `cpu_offload_gb`; GreenBoost replaces that knob |
| HuggingFace `diffusers` | 0.27 | 0.30+ | Disable `enable_model_cpu_offload()`; use plain `pipe.to("cuda")` |
| HuggingFace `transformers` | 4.40 | 4.45+ | Works as-is; `device_map="cuda"` recommended over `"auto"` |
| PyTorch | 2.1 | 2.4+ | Caching allocator (`expandable_segments`) interacts cleanly |
| `safetensors` | 0.4 | 0.4+ | Orthogonal; host-side mmap, GreenBoost intercepts CUDA side |
| Containers | Docker, Podman | NVIDIA Container Toolkit | Use Path B (no `/dev/greenboost`) |
| WSL2 | yes | yes | Path B only |

---

## G.11. Licensing and Attribution

GreenBoost is dual-licensed under **GPL v2** (open-source / personal /
research use) and a **Commercial License** (proprietary integration, SaaS,
embedded products). See `LICENSE` in the source tree for full terms.

All derivative works, forks, and citations must include the attribution:

> GreenBoost, created by Ferran Duarri.
> https://gitlab.com/IsolatedOctopi/greenboost

This requirement applies under both license options.

GreenBoost is **not** affiliated with NVIDIA Corporation. The CUDA, NVIDIA,
NVML, NVTX, Hopper, Grace, Ada, Blackwell, and Tegra names referenced in
this document are trademarks of NVIDIA Corporation. The *CUDA C++
Programming Guide* is © NVIDIA Corporation.

---

## G.12. Cross-Reference Index to the *CUDA C++ Programming Guide*

For readers who want to cross-check claims in this chapter against NVIDIA's
own documentation:

| Topic | Programming Guide section | GreenBoost section |
|---|---|---|
| Page-Locked Host Memory | §3.2.5 | §G.4.2 (Path B) |
| Portable + Mapped Memory | §3.2.5.1 / 3.2.5.3 | §G.4.2 |
| External Resource Interoperability | §3.2.7 | §G.4.1 (Path A) |
| Unified Memory Programming | §4.1 | §G.4.3 (Path C) |
| Unified Memory on cc 6.x+ Linux | §4.1.2 | §G.4.3 |
| Data Usage Hints (`cuMemAdvise`) | §4.1.4 | §G.4.3 |
| `SET_PREFERRED_LOCATION` semantics | §4.1.4.2 | §G.4.3 |
| `cudaMemPrefetchAsync` override behavior | §4.1.4.2 | §G.4.3 (load-bearing) |
| Querying Data Usage Attributes | §4.1.4.4 | §G.8 |
| Kernel Launch | §3.2.10 | §G.6.3 |
| Stream-Ordered Memory Allocator | §3.2.6 | §G.7.1 (hooked, untouched semantics) |

For the load-bearing claim about `cudaMemPrefetchAsync` overriding
`PREFERRED_LOCATION`, see Programming Guide §4.1.4.2 verbatim; that
sentence is the formal NVIDIA documentation that justifies GreenBoost's
prefetch-interception design.

---

*End of Chapter G.*

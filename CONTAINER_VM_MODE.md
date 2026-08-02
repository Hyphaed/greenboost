# 📦 GreenBoost in Containers, VMs, and WSL2

*Applies to GreenBoost v3.3. See [README.md](README.md) for the general
overview and [DOCUMENTATION.md](DOCUMENTATION.md) for the full architecture.*

> **TL;DR:** Can't load a kernel module (Docker, LXC, KVM, WSL2, shared HPC)?
> GreenBoost falls back automatically to **Path B** , no `greenboost.ko`
> needed, no config required.

---

## Contents

- [Overview](#overview)
- [What was contributed by Jerry Nguyen (MR !3)](#what-was-contributed-by-jerry-nguyen-mr-3)
- [What GreenBoost added on top of Path B](#what-greenboost-added-on-top-of-path-b)
- [When to use Path A0/A (bare metal, the default)](#when-to-use-path-a0a-bare-metal-the-default)
- [When to use Path B (no kernel, containers/VMs)](#when-to-use-path-b-no-kernel-containersvms)
- [Path B limitations compared to Path A0/A](#path-b-limitations-compared-to-path-a0a)
- [Environment variable reference](#environment-variable-reference)
- [License and attribution](#license-and-attribution)

---

## Overview

This document covers the memory-tiering pillar of GreenBoost under Path B ,
gb-quant compression and cluster mode work the same inside a container as on
bare metal; see [README.md](README.md) for the full picture.

The GreenBoost CUDA shim (`libgreenboost_cuda.so`) routes VRAM-overflow
allocations through one of **three** paths, selected automatically at
runtime , no flags to set, no config to write:

| Path | Name | Requires kernel module | Best for |
|---|---|---|---|
| **A0** | `cudaImportExternalMemory` (zero-copy) | Yes , `greenboost.ko` | Bare-metal Linux, best bandwidth (tried first) |
| **A** | DMA-BUF + `cuMemHostRegister` | Yes , `greenboost.ko` | Bare-metal Linux, fallback from A0 |
| **B** | `cuMemHostRegister`, no kernel module | No | Containers, VMs, Docker, LXC, KVM, WSL2 |

The shim tries **A0 → A → B**, in order. If `/dev/greenboost` is
unavailable (no kernel module loaded , the normal case inside a container
or VM), it falls through to Path B automatically, with zero configuration
needed.

> **What happened to Path C?** Earlier GreenBoost releases had a fourth
> path , managed UVM (`cuMemAllocManaged`) , as a universal last-resort
> fallback once system RAM (T2) was exhausted. It was **removed**: UVM's
> fault-driven demand-paging caused GPU stalls and silently pulled compute
> onto the CPU, which violates GreenBoost's one rule (compute always stays
> on the GPU). Today, once T2 is exhausted, GreenBoost returns a clean
> out-of-memory error instead , the same thing your framework would see on
> a GPU with no overflow at all, and something llama.cpp and most
> inference frameworks already handle correctly. See `CHANGELOG.md` v3.2
> for the full reasoning. (Managed UVM is still used internally for one
> narrow, unrelated purpose , a Blackwell-specific pinning trick , see
> `DOCUMENTATION.md` and the NVIDIA extension doc for details; it is not a
> general-purpose spill tier.)

---

## What was contributed by Jerry Nguyen (MR !3)

Jerry Nguyen's merge request introduced a **kernel-module-free** approach to
VRAM overflow using the standard CUDA driver API:

1. `mmap(MAP_ANONYMOUS)` to allocate anonymous system RAM pages.
2. `cuMemHostRegister(ptr, size, CU_MEMHOSTREGISTER_DEVICEMAP)` to pin those
   pages and make them GPU-accessible without any kernel driver involvement.
3. `cuMemHostGetDevicePointer_v2(&dev_ptr, ptr, 0)` to obtain a CUDA device
   pointer the application can use normally.

This is the complete core of Path B. The v2 variant of
`cuMemHostGetDevicePointer` is used (switched from v1 , fixes CUDA error 201
on newer drivers). No other changes were made to this mechanism; it works
exactly as Jerry proposed. The key insight from MR !3 is that the CUDA
driver itself, via `cuMemHostRegister`, can pin host memory and expose it to
the GPU without a custom kernel module , making the overflow approach
viable inside containers and virtual machines where `greenboost.ko` cannot
be loaded.

---

## What GreenBoost added on top of Path B

### 2 MB hugepage preference
Before falling back to 4 K pages, Path B attempts `mmap` with
`MAP_HUGETLB | MAP_HUGE_2MB`. This reduces TLB pressure for large model
allocations (tens of gigabytes) and can improve PCIe DMA throughput by
~10–15% on kernel ≥ 5.14 with `vm.nr_hugepages` available. A compile-time
`MAP_HUGE_2MB` availability guard ensures the build succeeds on kernels that
don't define this flag.

### Hash-table tracking
Every Path B allocation is recorded in the same open-addressed
131,072-slot hash map used by Path A. This ensures `cuMemFree_v2` /
`cudaFree` correctly unregisters the host memory with
`cuMemHostUnregister` and calls `munmap` , no leak, even for apps that free
allocations out of order.

### Transparent fallback ordering
Path B sits between Path A and the OOM-on-exhaustion behavior in
`gb_overflow_alloc()`. The calling code (`cuMemAlloc_v2`,
`cuMemAllocAsync`, `cudaMalloc`, etc.) sees a single function call and is
unaware which path succeeded.

### Container detection caching
The shim detects at startup whether it's running inside a container
(Docker, LXC, Kubernetes pod) by inspecting `/proc/1/cgroup` and the
presence of `/.dockerenv`. This detection result is cached at constructor
time , the check doesn't repeat on every allocation, removing a hot-path
syscall that was measurable under high-frequency inference.

### `GREENBOOST_NO_HOSTREG=1` escape hatch
Skips Path B entirely , useful for profiling or debugging. With Path B
skipped, allocations that don't fit T1/local behavior return OOM once
Path A0/A are exhausted.

### Debug banner
`GREENBOOST_DEBUG=1` prints the status of every path at startup:
```
[GreenBoost] Path A0 (cudaImportExtMem) : enabled - cudaImportExternalMemory (best bandwidth)
[GreenBoost] Path A  (DMA-BUF+kernel)   : enabled - mmap+GB_IOCTL_PIN_USER_PTR+HostReg
[GreenBoost] Path B  (HostReg/no-kmod)  : available - mmap+cuMemHostRegister (containers/VMs, no greenboost.ko needed)
[GreenBoost] Path C  (UVM)              : REMOVED - CPU compute forbidden; OOM returned when T2 exhausted
```

### Everything else stays active on Path B
- Two-stage `RTLD_NOLOAD` constructor guard (safe system-wide injection via `/etc/ld.so.preload`)
- `dlsym` hook intercepting Ollama's `dlopen`+`dlsym` GPU API lookups
- `dlopen` hook stripping `RTLD_DEEPBIND`
- Virtual VRAM inflation hooks (`cuDeviceTotalMem_v2`, `nvmlDeviceGetMemoryInfo`, `nvmlDeviceGetMemoryInfo_v3`)
- `cuMemGetInfo` hook (reports physical VRAM + system RAM pool to CUDA)
- `cuMemFreeAsync` hook (ensures async-allocated buffers are correctly released and removed from the hash map)
- `cuMemAllocAsync` compute-capability gate (Path B async allocs are only attempted on cc ≥ 8.0 devices; older GPUs fall back to synchronous paths)
- `GREENBOOST_ACTIVE` opt-in for lazy-dlopen apps (Ollama, vLLM, PyTorch)
- Async prefetch worker thread with `MADV_WILLNEED`
- `gb_atoll()` GLIBC_2.38 snap compatibility fix

---

## When to use Path A0/A (bare metal, the default)

**Use Path A0/A (the default) on bare-metal Linux.**

Path A0 → A is the recommended configuration for all standard installations.
It provides the best possible bandwidth and the most robust integration:

- **Pinned hugepages via the kernel buddy allocator.** `greenboost.ko`
  allocates 2 MB compound pages that are physically contiguous and
  guaranteed not to be swapped or migrated by the kernel pager , the best
  possible access pattern for the GPU's DMA engine over PCIe.
- **Kernel-controlled memory pressure watchdog.** A kthread monitors system
  RAM and NVMe swap pressure and signals userspace via eventfd before things
  become dangerous , something a pure-userspace path cannot do.
- **T3 NVMe tier.** The kernel module manages the Tier 3 NVMe overflow pool
  (`/var/lib/greenboost/t3_store`), extending effective memory to hundreds
  of gigabytes. Path B has no T3 equivalent , when system DDR is exhausted,
  it returns OOM.
- **Live sysfs monitoring.** `watch -n1 'cat /sys/class/greenboost/greenboost/status'`
  gives instant visibility into how much system DDR the model is consuming.

**Summary: on bare metal, always run `sudo ./greenboost_setup.sh` (Full
Install) and use Path A0/A. Path B exists for environments where A0/A is
impossible.**

---

## When to use Path B (no kernel, containers/VMs)

Use Path B when you cannot load a custom kernel module:

### Docker / Podman containers
Docker containers run in a shared kernel namespace , you cannot `insmod` a
custom module inside one (the host kernel doesn't allow it, and the module
ABI may differ anyway). Path B lets GreenBoost work inside CUDA-enabled
Docker images without any host-side kernel change beyond the NVIDIA
Container Toolkit being present.

```bash
# Inside a Docker container - Path A is impossible, Path B activates automatically
docker run --gpus all \
  -e GREENBOOST_ACTIVE=1 \
  -v /usr/local/lib/libgreenboost_cuda.so:/usr/local/lib/libgreenboost_cuda.so \
  -e LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
  my-llm-image python run_model.py
```

### LXC / LXD unprivileged containers
Same constraint as Docker , custom kernel modules cannot be loaded from
within an unprivileged container.

### KVM / QEMU virtual machines with GPU passthrough
In a KVM guest with PCI passthrough of an NVIDIA GPU, the guest kernel can
load `nvidia.ko` and `nvidia-uvm.ko`, but `greenboost.ko` would need to be
built against the *guest* kernel , a separate build step most users want to
avoid. Path B provides VRAM overflow without needing to build and load a
guest kernel module.

### WSL2 (Windows Subsystem for Linux 2)
WSL2 runs a Microsoft-patched Linux kernel that doesn't support arbitrary
third-party kernel modules via `insmod`. The NVIDIA CUDA driver for WSL2 is
provided by the host Windows driver and exposed through `/dev/dxg`. Path
B's `cuMemHostRegister` path works correctly on WSL2 CUDA because it only
calls into the CUDA driver API.

### Shared HPC / cloud GPU instances
On shared HPC clusters or cloud GPU VMs (AWS, GCP, Azure) you typically
don't have root access to load kernel modules. Path B lets GreenBoost's
VRAM overflow work inside `srun`/`sbatch` jobs without any sysadmin
involvement.

---

## Path B limitations compared to Path A0/A

| Feature | Path A0/A (DMA-BUF) | Path B (HostReg) |
|---|---|---|
| Hugepages | Kernel-allocated, guaranteed 2 MB | Best-effort `MAP_HUGETLB` (may fall back to 4 K) |
| T3 NVMe tier | Yes , kernel watchdog + swap subsystem | No , system DDR only; OOM once exhausted |
| Live sysfs monitoring | `/sys/class/greenboost/greenboost/status` | Not available |
| Memory pressure watchdog | Kernel kthread, eventfd signal | Not available |
| Physical page pinning | Hard-pinned by the kernel (cannot be evicted) | Soft-pinned by the CUDA driver (may be re-registered) |
| PCIe bandwidth (typical) | ~50 GB/s (dual-channel DDR, pinned hugepages) | ~32–45 GB/s (depends on page size and TLB miss rate) |
| Container detection | N/A | Cached at startup , no per-alloc overhead |
| Requires `greenboost.ko` | Yes | No |
| Works in Docker/VM/WSL2 | No | Yes |

---

## Environment variable reference

| Variable | Default | Effect |
|---|---|---|
| `GREENBOOST_USE_DMA_BUF` | `1` | `0` = skip Path A0 and A entirely (jump straight to B) |
| `GREENBOOST_NO_HOSTREG` | `0` | `1` = skip Path B (OOM returned once T2 is exhausted) |
| `GREENBOOST_ACTIVE` | unset | Set to `1` for lazy-dlopen apps (Ollama, vLLM, PyTorch) |
| `GREENBOOST_DEBUG` | `0` | `1` = verbose per-path status at startup + per-alloc log |
| `GREENBOOST_VRAM_HEADROOM_MB` | `512` | Keep this many MB free in VRAM before overflowing |
| `GREENBOOST_VIRTUAL_VRAM_MB` | auto | Override the reported virtual VRAM size |
| `GREENBOOST_CUDART_PATH` | auto | Explicit `libcudart.so` path |
| `GREENBOOST_DISABLE` | `0` | `1` = disable the shim entirely for one process |

---

## License and attribution

GreenBoost is GPL v2. Attribution to Ferran Duarri is required in all forks
and derivatives.

Path B was originally proposed by **Jerry Nguyen** (GitLab
[@jerry.nguyen](https://gitlab.com/jerry.nguyen)) in merge request
[!3](https://gitlab.com/IsolatedOctopi/greenboost/-/merge_requests/3). The
`cuMemHostRegister(DEVICEMAP)` mechanism, the kernel-module-free concept,
and the container/VM use-case rationale all originate from Jerry's
contribution. The hugepage preference, hash-table integration, fallback
ordering, env var controls, debug banner, container detection caching, Path
A0 (`cudaImportExternalMemory`), the `cuMemHostGetDevicePointer_v2` switch,
and the later Path B compute-capability/async hooks were added by Ferran
Duarri when integrating MR !3 into the blended shim.

```
Copyright (C) 2026 Ferran Duarri
Path B concept: Copyright (C) 2026 Jerry Nguyen
```

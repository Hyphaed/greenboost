# GreenBoost Changelog

---

## v2.8 : 2026-04-10

On Friday, April 3rd, I had a bicycle accident while jumping some hills.
Four stitches on the eyebrow, a fractured clavicle, and two broken ribs.
I really shouldn't be near a keyboard right now.

---

⚖️ **Trademark compliance — project renamed**

This release was made to comply with a trademark notice concerning the use of the NVIDIA wordmark in the project title. The tool has been renamed:

> **GreenBoost : CUDA Memory Orchestrator for NVidia GPUs**

"NVidia" is used here only as a hardware descriptor, not as a brand identifier.
GreenBoost is an independent open-source project and is not affiliated with, endorsed by, or sponsored by NVIDIA Corporation. NVIDIA, CUDA, GeForce, and RTX are trademarks of NVIDIA Corporation.

---

🎮 **GreenBoost now works with games**
A new Vulkan layer exposes the complete GreenBoost CUDA memory pool (T1 GPU VRAM + T2 System DDR RAM) to Proton and games this is an an early beta feature, greenboost was not created for gaming, this is a byproduct of the project.

GreenBoost reports an inflated VRAM total (T1 GPU VRAM + T2 System DDR) to Vulkan. 
Games that would otherwise hit a VRAM wall, triggering texture pop-in, reduced quality presets, or crashes, can now use the extra headroom.

Then in Steam: right-click a game → Properties → Compatibility → select **GreenBoost Proton**.

**Enable per game in Steam launch options:**

With DLSS Super Resolution preset override:

| Preset | Quality | Best for |
|--------|---------|----------|
| M (Heavier) | Highest quality | RTX 40/50 series |
| L (Balanced) | Good quality/perf balance | Any RTX |
| K (Lighter) | Better performance | RTX 20/30 series |

Example for preset M:
```
DXVK_NVAPI_DRS_NGX_DLSS_SR_OVERRIDE_RENDER_PRESET_SELECTION=render_preset_m %command%
```

With HDR:
```
PROTON_ENABLE_HDR=1 DXVK_NVAPI_DRS_NGX_DLSS_SR_OVERRIDE_RENDER_PRESET_SELECTION=render_preset_m %command%
```

For benchmarking setup (MangoHud configure for greenboost), go to the installation wizard → Gaming → Install MangoHud.
                                                                                                               
📚 **Added documentation**
documentation.md
greenboost_commands.md

🧹 **Code mantainance + cleanup**
Internal code refactor, dead code paths removed, and general improvements to the
shim and setup scripts. 
Updated RedHat based OS installation script.
Added Arch based OS installation script.
Model tuning tools had been moved into a separate project, this in development at the moment...

🐛 **Stability fixes**
- Enhancements on the agnostic hardware dinamyc calculations.
- Fixed a potential crash when requesting very large memory allocations at the edge of available capacity.

---

## v2.7 : 2026-03-29 same-day version update, develop branch merged

🤖 **Heuristic KV cache detection and prioritization**
The CUDA shim identifies and prioritizes KV cache allocations in T1 VRAM
entirely on its own. Two complementary mechanisms:

1. **Phase detector**: temporal state machine (INIT → MODEL_LOAD → INFERENCE → STEADY)
   classifies every overflow alloc as weights, KV cache, or activations. During
   INFERENCE/STEADY, large allocs (≥ 64 MB, down from 256 MB) receive
   `GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY` — the kernel auto-freezes them in T2
   LRU and refuses T3 spill.

2. **Adaptive KV reserve**: the shim reserves T1 VRAM headroom for upcoming KV allocs
   while weights are loading. Once KV has been allocated in T1, the reserve collapses
   proportionally — eliminating the previous double-counting bug where `cuMemGetInfo`
   already reflected the KV allocation but the full reserve was still subtracted.

Set `GREENBOOST_KV_RESERVE_MB` or `GREENBOOST_KV_SIZE_THRESHOLD_MB` to tune.

🎛️ **Adjustable RAM usage limit**
You can now change how much system RAM GreenBoost is allowed to use while running
(`GB_IOCTL_SET_POOL_CAP`). Avoids conflicts with VMs, Docker, or other memory-hungry
workloads running alongside inference.

🛠️ **Profiles**
Hardware profile system for persistent configuration across reboots and for cluster
deployments. A profile is auto-generated on first install from detected hardware.
V100 cluster support: 8-GPU NVLink unified T1 pool (128 GB), Lustre T3 storage,
per-node safety reserve. See `profiles/v100_cluster_node.md`.

🏗️ **Native hardware-targeting build**
The kernel module and CUDA shim are now compiled with flags that target the exact CPU
microarchitecture of the machine where GreenBoost is installed:

```
-march=native -mtune=native -O3 -funroll-loops
-flto -fvisibility=hidden -ffunction-sections -fdata-sections
```
AVX2 is detected automatically at build time (`/proc/cpuinfo`) and enabled if present.
Link-time optimization (`-flto`) and dead-code elimination (`--gc-sections --as-needed`)
produce leaner, faster shared libraries. The shim is built with `-fvisibility=hidden` so
only the hooked symbols are exported — nothing internal is exposed to the linker.

This is not a portable binary. It is compiled for the workstation it runs on, maximising
throughput on the host CPU during shim bookkeeping (hash table probes, phase-detector
timestamp comparisons, atomic KV-reserve accounting).

---

## v2.6 : 2026-03-29

🎨 **New Installer**
New installer UI

📊 **Benchmark and status tools**
New benchmark and status tools accesible trough the wizard.
Status;  Show cuda memory pool + system state
Benchmark; Measure T1/T2/T3 bandwidth

🐛 **`cuMemHostGetDevicePointer_v2` — primary context fix (MR !5)**
The CUDA shim was resolving the v1 symbol `cuMemHostGetDevicePointer`, which returns
`CUDA_ERROR_INVALID_CONTEXT (201)` when the application uses the primary context model
(`cuDevicePrimaryCtxRetain`). PyTorch, vLLM, and most modern ML frameworks use primary
contexts by default, so this caused a silent fallback to UVM (Path C), losing the fast
PCIe DMA path entirely. The fix is a single-line change in `gb_shim_init()` resolving
`cuMemHostGetDevicePointer_v2` instead, which handles both explicit and primary contexts
correctly. Paths A and B now work as intended on PyTorch/vLLM.

🐧 **Debian Trixie support + kernel 6.12+ DMA_BUF build patch (MR !4)**
The setup script now detects Debian (including Trixie / testing / unstable) and installs
the correct packages for that distribution: `linux-cpupower` and `linux-perf` in place of
the Ubuntu-specific `cpufrequtils` and `linux-tools-generic`. The build step also
automatically patches `greenboost.c` with `MODULE_IMPORT_NS(DMA_BUF);` if the line is
absent, which is required for the kernel module to compile on Linux 6.12 and above.

🧠 **Contributors**
2 new contributors:

- **Giuseppe Marco Randazzo** ([@gmrandazzo](https://gitlab.com/gmrandazzo)) — Debian Trixie
  support in `greenboost_setup.sh`; package dependency mapping for Debian testing/unstable
  (`linux-cpupower`, `linux-perf` in place of Ubuntu-specific equivalents); kernel 6.12+
  `MODULE_IMPORT_NS(DMA_BUF)` build-time patch.

- **Alexey Masolov** ([@alexeymasolov](https://gitlab.com/alexeymasolov)) — fix for
  `cuMemHostGetDevicePointer_v2` primary context compatibility; ensures Path A/B work
  correctly with PyTorch, vLLM, and all frameworks that use `cuDevicePrimaryCtxRetain`.
  Without this, `CUDA_ERROR_INVALID_CONTEXT (201)` caused silent fallback to UVM (Path C),
  losing ~32 GB/s PCIe DMA performance. Validated on Quadro RTX 5000 with vLLM 0.18.0.

---

## v2.5 : first open-source release
The v2.5 tag on GitLab marks the first public release of GreenBoost.

🛠️ Path B integrated
Now greenboost can work on virtualized enviornments thanks to Jerry Nguyen.

🧭 Installation script gets broader support
Now greenboost can be installed on RedHat based OS thanks to Alan Sill.


🧠 Contributors
First external contributors joined in this version

- **Alan Sill** ([@alansill](https://gitlab.com/alansill)) — contributed
  `greenboost_setup_rocky.sh`, a setup script for Red Hat-based systems (Rocky Linux,
  AlmaLinux, RHEL). The Ubuntu script now delegates automatically to Alan's script when a
  Red Hat-based OS is detected at runtime.

- **Jerry Nguyen** ([@phubao](https://gitlab.com/phubao)) — contributed the
  kernel-module-free overflow path (MR !3): `cuMemHostRegister(DEVICEMAP)` enables
  GreenBoost VRAM extension inside containers, VMs, WSL2, and HPC clusters without requiring
  `greenboost.ko`. Integrated as Path B of the blended shim.

---

## v2.4 : last private release
v2.4 was the last version developed privately before the project was open-sourced.
There were no external contributors at this stage — all work was done by the author alone.
The core cuda memory pool (VRAM + DDR DMA-BUF + NVMe swap), CUDA shim, and Ollama integration
were functional at this point.

---

## <v2.4 : earlier private development
All versions prior to v2.4 were non-public development releases used exclusively on the
author's own workstation (i9-14900KF / RTX 5070 / 64 GB DDR4 / Samsung 990 EVO Plus 4 TB).
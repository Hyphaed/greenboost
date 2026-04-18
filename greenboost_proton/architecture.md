# GreenBoost Proton — Architecture

## Overview

`greenboost-proton-wayland` is a standalone Steam compatibility tool built on top of
Proton Experimental. It is NOT a wrapper script — it IS a full Proton runtime, with `files/`
(Wine, VKD3D-Proton, DXVK binaries) symlinked from the upstream pre-built release.

GreenBoost integration is achieved by patching the Proton Experimental Python orchestrator script
(`proton`) at two injection points, requiring no source recompilation.

---

## Execution chain

```
Steam (launch game)
  │
  └─ greenboost-proton-wayland/proton %verb%      ← Python 3 orchestrator
       │
       ├─ [patched] gb_detect_nvidia()             auto-detects GPU, VRAM, RT support
       ├─ [upstream] g_session.init_session(...)   sets Proton Experimental defaults
       ├─ [patched] gb_apply_greenboost(session)   injects GreenBoost env vars
       │    ├─ GREENBOOST_VULKAN=1
       │    ├─ PROTON_ENABLE_WAYLAND=1
       │    ├─ VKD3D_DEBUG=warn (overrides init_session's "none")
       │    ├─ VKD3D_CONFIG=dxr,dxr11  (if GPU supports RT)
       │    ├─ DXVK_ENABLE_NVAPI=1  (if NVIDIA)
       │    └─ PROTON_LOG=1 → ~/.local/share/greenboost/proton-logs
       │
       └─ files/ (Proton Experimental binaries — Wine + VKD3D-Proton + DXVK)
            ├─ Wine (wine64, wineserver)
            │    └─ Windows API → Linux syscalls
            ├─ VKD3D-Proton (d3d12.dll.so)      DX12 → Vulkan
            │    └─ Vulkan API calls
            ├─ DXVK (d3d11.dll.so, ...)         DX9/10/11 → Vulkan
            │    └─ Vulkan API calls
            └─ Vulkan loader (libvulkan.so)
                 │
                 └─ VK_LAYER_GREENBOOST_memory   ← GreenBoost implicit Vulkan layer
                      │  inflates device-local heap to 65 GB
                      │  on VK_ERROR_OUT_OF_DEVICE_MEMORY ≥64 MB:
                      │    opens /dev/greenboost → ioctl GB_IOCTL_ALLOC
                      │    imports DMA-BUF via VK_EXT_external_memory_dma_buf
                      │
                      └─ GPU Vulkan driver
                           ├─ T1: physical VRAM        ~336 GB/s
                           └─ T2: system DDR (DMA-BUF) ~32 GB/s (PCIe)
```

---

## Memory tiers as seen by a DX12 game

| Tier | Backed by | Vulkan heap | Bandwidth |
|------|-----------|-------------|-----------|
| T1 | GPU VRAM | device-local (real) | ~336 GB/s |
| T2 | System DDR (pinned, DMA-BUF) | device-local (virtual, inflated) | ~32 GB/s |
| T3 | NVMe swap | — (kernel swap; not exposed to Vulkan) | ~1.8 GB/s |

The game and VKD3D-Proton see a single 65 GB device-local heap. Allocations that would
overflow real VRAM are caught by `VK_LAYER_GREENBOOST_memory` and routed to T2 DDR.

**Zero CPU spillover**: VKD3D-Proton sees 65 GB and never falls back to CPU-side execution.
Tensor computation (shader dispatch) always executes on the GPU.

---

## VKD3D-Proton configuration

Applied by `gb_apply_greenboost()` (auto-detected at runtime):

| Option | Applied when | Effect |
|--------|-------------|--------|
| `dxr` | GPU supports RT | Enable DirectX Raytracing |
| `dxr11` | GPU supports RT | Enable DXR via D3D11 path |

User can append more via `GREENBOOST_VKD3D_CONFIG=option1,option2`.

`VKD3D_DEBUG=warn` is set (overriding Proton Experimental's default of `none`) so the
`greenboost vulkan` dashboard can surface VKD3D-Proton warnings.

---

## DXVK configuration

| Variable | Value | Reason |
|----------|-------|--------|
| `DXVK_ENABLE_NVAPI` | `1` | DLSS, Frame Generation, Reflex (NVIDIA) |
| `DXVK_LOG_LEVEL` | `warn` | Visible in `greenboost vulkan` dashboard |
| `DXVK_LOG_PATH` | `~/.local/share/greenboost/proton-logs` | Centralised log |
| `DXVK_STATE_CACHE` | `1` | Persistent pipeline state |
| `DXVK_STATE_CACHE_PATH` | `~/.local/share/greenboost/proton-cache/dxvk-state` | Stable path |

---

## Wayland

`PROTON_ENABLE_WAYLAND=1` is set by default. Proton Experimental's native Wayland backend is used,
eliminating XWayland overhead. HDR requires Wayland: add `PROTON_ENABLE_HDR=1 %command%`.

---

## Logging and observability

| Log | Location | Content |
|-----|----------|---------|
| VK_LAYER_GREENBOOST | syslog/journalctl | T2 DMA-BUF allocs, OOM events |
| VKD3D-Proton | `~/.local/share/greenboost/proton-logs/steam-<appid>.log` | D3D12 warnings |
| DXVK | `~/.local/share/greenboost/proton-logs/dxvk.log` | D3D11/9 errors, NVAPI |

All logs are aggregated by `greenboost vulkan` into the live dashboard (Panel 3 + Panel 4).

---

## File layout

```
greenboost-proton-wayland/
├── proton                   Python 3 orchestrator (patched Proton Experimental)
├── install.sh               Steam compat tool installer
├── compatibilitytool.vdf    Steam registration
├── toolmanifest.vdf         Steam invocation spec
├── version                  Version string
├── files/                   (Wine+VKD3D+DXVK)
├── protonfixes/          
├── filelock.py           
├── architecture.md
└── documentation.md
```

---

## Integration with GreenBoost status commands

```bash
greenboost vulkan          # live dashboard: device, DX12 game, T2 stats, issues
```
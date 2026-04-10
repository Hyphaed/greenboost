# GreenBoost Proton Wayland — Beginner's Guide

  The patched proton script auto-detects your GPU on every launch via gb_detect_nvidia() and calls gb_apply_greenboost() after Proton Experimental's init_session() to inject:     
  GREENBOOST_VULKAN=1, PROTON_ENABLE_WAYLAND=1, VKD3D_DEBUG=warn, VKD3D_CONFIG=dxr,dxr11 (if RT detected), DXVK_ENABLE_NVAPI=1 (NVIDIA), and log routing to              
  ~/.local/share/greenboost/proton-logs for greenboost vulkan dashboard.        
  
This document explains how a Windows game runs on Linux, how DirectX translates to
Vulkan, and where GreenBoost fits in. No prior Linux or graphics experience assumed.

---

## 1. Why do Windows games need Proton?

Windows games are written for the Windows operating system. They call Windows APIs
(application programming interfaces) — functions like "draw this triangle" or "play this
sound" that Windows provides. Linux does not have these APIs natively.

**Proton** is a compatibility layer developed by Valve that makes Linux pretend to be
Windows for a game. Internally it uses:

- **Wine** — translates Windows API calls (file system, memory, threads) to Linux equivalents.
- **VKD3D-Proton** — translates Direct3D 12 (Microsoft's graphics API) to Vulkan.
- **DXVK** — translates Direct3D 9, 10, and 11 to Vulkan.

The game never knows it is on Linux.

---

## 2. The graphics stack for a DX12 game

Here is what happens from the moment a DX12 game calls "draw this frame":

```
Game (re9.exe — Windows process)
  │
  │  calls Direct3D 12 functions
  ▼
VKD3D-Proton (d3d12.dll.so)
  │
  │  translates D3D12 calls into Vulkan calls
  ▼
Vulkan Loader (libvulkan.so)
  │
  │  passes calls through Vulkan layers (like GreenBoost's)
  ▼
GPU Driver (NVIDIA libvulkan_nvidia.so)
  │
  │  sends commands to the physical GPU
  ▼
GPU Hardware (renders the frame)
```

---

## 3. What is a Vulkan layer?

A **Vulkan layer** is a library that intercepts Vulkan API calls and can modify their
behaviour — without the application knowing.

```
VKD3D-Proton
  ↓  vkAllocateMemory(device, 40 GB, ...)
VK_LAYER_GREENBOOST_memory          ← intercepts the call
  │  40 GB fits in real VRAM? → pass through
  │  40 GB > real VRAM?       → allocate from system RAM via DMA-BUF, return success
  ↓
NVIDIA Vulkan driver
```

The game and VKD3D-Proton think they got 40 GB of GPU memory. Whether that memory
physically lives on the GPU or in system RAM is invisible to them.

---

## 4. How GreenBoost extends VRAM for gaming

The real GPU has N GB of VRAM. Without GreenBoost, a DX12 game that tries to allocate
more fails with an out-of-memory error and either crashes or reduces quality.

GreenBoost solves this in two steps:

### Step 1 — Tell the game there is 65 GB of VRAM

The GreenBoost Vulkan layer hooks `vkGetPhysicalDeviceMemoryProperties`. Every time
VKD3D-Proton asks "how much GPU memory is there?", the layer returns 65 GB instead of
the real amount.

### Step 2 — Satisfy overflow allocations from system RAM

When VKD3D-Proton tries to allocate more than the real VRAM and the GPU driver returns
`VK_ERROR_OUT_OF_DEVICE_MEMORY`, the GreenBoost layer catches this error and instead:

1. Opens `/dev/greenboost` (the GreenBoost kernel module).
2. Sends an ioctl request: "give me a DMA-BUF for N GB of pinned system RAM".
3. Imports that file descriptor into Vulkan as GPU-accessible memory using
   `VK_EXT_external_memory_dma_buf`.
4. Returns `VK_SUCCESS` to VKD3D-Proton.

The game and VKD3D-Proton think they got VRAM. In reality it is system RAM accessed
over PCIe. For cold data (weights not currently being computed), this is acceptable
because the GPU prefetches data before it is needed.

**Tensor computation always stays on the GPU** — only data residency moves to system RAM.

---

## 5. What GreenBoost Proton Wayland adds

| Feature | Vanilla Proton | Proton Experimental | GreenBoost Proton Wayland |
|---------|---------------|-----------|--------------------------|
| DX12 → Vulkan | VKD3D-Proton | VKD3D-Proton | VKD3D-Proton |
| Wayland support | Limited | Yes | **Yes (default on)** |
| GreenBoost Vulkan layer | Manual | Manual | **Automatic** |
| VKD3D_DEBUG=warn | Off | Off | **On** (for diagnostics) |
| Ray tracing (DXR) | Off | Off | **Auto-enabled if GPU supports** |
| DLSS / Frame Gen (NVAPI) | Off | Off | **Auto-enabled on NVIDIA** |
| Log routing for `greenboost vulkan` | No | No | **Yes** |

GreenBoost Proton Wayland does NOT replace VKD3D-Proton or DXVK — it configures them
optimally for GreenBoost and automates settings that would otherwise require per-game
Steam launch options.

---

## 6. Ray tracing (DXR)

Ray tracing simulates light rays as they bounce off surfaces, producing realistic shadows,
reflections, and global illumination.

DX12 exposes ray tracing through the DirectX Raytracing (DXR) API. VKD3D-Proton
translates DXR calls to `VK_KHR_ray_tracing_pipeline`. The GPU must have dedicated
ray-tracing hardware (RT cores on NVIDIA RTX GPUs).

GreenBoost Proton Wayland enables DXR automatically when the GPU supports it:

```bash
VKD3D_CONFIG=dxr,dxr11
```

With GreenBoost's 65 GB virtual VRAM, games no longer run out of VRAM for ray-tracing
resources.

---

## 7. NVAPI — DLSS, Frame Generation, Reflex

NVIDIA's proprietary API (NVAPI) provides DLSS (AI upscaling), Frame Generation
(interpolated frames), and Reflex (latency reduction).

Under Proton/Linux, `dxvk-nvapi` implements the NVAPI interface. GreenBoost Proton
Wayland enables this automatically on NVIDIA hardware:

```bash
DXVK_ENABLE_NVAPI=1
```

---

## 8. Native Wayland

`PROTON_ENABLE_WAYLAND=1` is set by default. The game window appears as a native Wayland
surface (no XWayland translation layer), reducing input latency and enabling:

- HDR (add `PROTON_ENABLE_HDR=1 %command%` to Steam launch options)
- Variable Refresh Rate (VRR) with compatible displays

To revert to XWayland for a specific game: `PROTON_ENABLE_WAYLAND=0 %command%`.

---

## 9. How to monitor everything

The `greenboost vulkan` command provides a live dashboard:

```
greenboost vulkan
```

It shows:
- **Vulkan Device & Layer** — GPU name, driver version, layer install status
- **Active Vulkan Processes** — running game processes and their VRAM usage
- **DX12 / VKD3D-Proton** — detected game, Proton version, T2 allocation stats
- **GreenBoost Vulkan Activity** — T2 DMA-BUF allocations and VKD3D warnings
- **Issues** — actionable problems (OOM failures, missing layer, etc.)

---

## 10. Common launch options

Add these to Steam → Properties → Launch Options:

| Option | Effect |
|--------|--------|
| `PROTON_ENABLE_HDR=1 %command%` | Enable HDR (requires Wayland) |
| `GREENBOOST_DISABLE=1 %command%` | Disable GreenBoost for this game |
| `GREENBOOST_VULKAN=0 %command%` | Disable only the Vulkan layer |
| `VKD3D_CONFIG=no_dxr %command%` | Disable ray tracing if causing issues |
| `DXVK_HUD=fps,memory %command%` | On-screen FPS + memory usage overlay |
| `PROTON_ENABLE_WAYLAND=0 %command%` | Use XWayland instead |

---

## 11. Troubleshooting

### Game crashes on launch
1. Check `greenboost vulkan` Issues panel for T2 DMA-BUF failures.
2. Check `journalctl | grep VK_LAYER_GREENBOOST` for layer errors.
3. Try `GREENBOOST_VULKAN=0 %command%` to rule out the layer.
4. Try `VKD3D_CONFIG=no_dxr %command%` to disable ray tracing.

### Game runs but VRAM shows only real VRAM in-game
- Check `greenboost vulkan` Panel 1: should show "65 GB (virtual)".
- Ensure `GREENBOOST_VULKAN=1` is active (default with this Proton).
- Verify GreenBoost kernel module is loaded: `ls /dev/greenboost`.

### Low FPS compared to Windows
- T2 DMA-BUF allocations are slower than VRAM (~32 GB/s vs ~336 GB/s).
- Check T2 MB count in `greenboost vulkan` Panel 3.
- Run `greenboost status` to check RAM pressure (`safety_reserve_gb`).

### DLSS not available
- Ensure `DXVK_ENABLE_NVAPI=1` is set (GreenBoost Proton Wayland does this automatically).
- Some games also require NVAPI driver settings: check `greenboost vulkan` Issues panel.

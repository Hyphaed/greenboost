# GreenBoost Proton

GreenBoost Proton is a Steam compatibility tool that wraps **Proton Experimental** (Valve's official cutting-edge Proton build) and injects GreenBoost's optimizations automatically for every game.

## Why it exists

When you select GreenBoost Proton in Steam, the game runs through Proton Experimental — so you get Valve's latest Wine, VKD3D-Proton, DXVK, and protonfixes — but before Wine starts, GreenBoost's wrapper script sets a precise set of environment variables that activate the virtual-VRAM layer, enable Wayland, tune VKD3D for ray tracing and shader caching, and coordinate the gaming-mode service. Without this wrapper you would have to set all of these manually in Steam's launch options on every game.

Because it wraps Proton Experimental rather than bundling its own Wine stack, **any update Valve ships to Proton Experimental is automatically available** — no reinstall of GreenBoost Proton needed when a new version of Proton is released (unless this carrying big changes, then GreenBoost wrapper will get an update).

## What the wrapper actually does

`greenboost_proton_wayland/proton` is ~170 lines of Python:

### 1. Detects hardware at runtime

- **GPU** — tries `nvidia-smi`, then `/proc/driver/nvidia/gpus/*/information`, then `lspci`. Determines GPU name, VRAM (MiB), whether it is NVIDIA, and whether it supports ray tracing (RTX/Ampere/Ada/Blackwell detected by name; Vulkan extension `VK_KHR_ray_tracing_pipeline` used as authoritative fallback).
- **CPU** — reads `/proc/cpuinfo` for model name; reads `core_type` sysfs to count P-cores vs E-cores on hybrid Intel CPUs; falls back to half of `os.cpu_count()` for non-hybrid.

Nothing is hard-coded. The same script works on any hardware.

### 2. Injects GreenBoost environment variables

All variables are written to `os.environ` so Proton Experimental inherits them when it starts.

| Variable | Value set | Effect |
|----------|-----------|--------|
| `GREENBOOST_VULKAN` | `1` | Activates `VK_LAYER_GREENBOOST_memory` — inflates the Vulkan device-local heap to T1 (GPU VRAM) + T2 (System DDR RAM); overflow allocations use T2 DDR via DMA-BUF |
| `PROTON_ENABLE_WAYLAND` | `1` | Proton Experimental uses the native Wayland backend instead of XWayland |
| `VKD3D_CONFIG` | `dxr,dxr11,pipeline_library_app_cache` | Enables DirectX Raytracing 1.0 + 1.1 (NVIDIA RTX only, skipped for games in the DXR blocklist); enables VKD3D-Proton's per-app pipeline cache for faster subsequent launches |
| `VKD3D_DEBUG` | `warn` | Routes VKD3D-Proton log output so `greenboost logs` can surface errors |
| `DXVK_ENABLE_NVAPI` | `1` | Enables DXVK's NVAPI layer — required for DLSS, Frame Generation, and NVIDIA Reflex in DX11/DX12 games |
| `DXVK_NUM_COMPILER_THREADS` | P-core count (max 16) | Shader pipeline compilation uses P-cores only, matching GreenBoost's shader-boost service |
| `PROTON_LOG` + `PROTON_LOG_DIR` | `1` + `~/.local/share/greenboost/proton-logs/` | Enables Proton's Wine debug log and routes it to the GreenBoost log directory so `greenboost logs` picks it up |
| `DXVK_LOG_PATH` | `~/.local/share/greenboost/proton-logs/` | DXVK error log also lands in the GreenBoost log directory |
| `__GL_SHADER_DISK_CACHE*` | `~/.local/share/greenboost/proton-cache/gl-shaders/` (8 GiB) | NVIDIA OpenGL shader disk cache — persists compiled shaders across game sessions |
| `DXVK_STATE_CACHE*` | `~/.local/share/greenboost/proton-cache/dxvk-state/` | DXVK pipeline state cache |
| `VKD3D_SHADER_CACHE_PATH` | `~/.local/share/greenboost/proton-cache/vkd3d-shader/` | VKD3D-Proton SPIR-V shader cache |
| `MESA_SHADER_CACHE_DIR` | `~/.local/share/greenboost/proton-cache/mesa-shader/` | Mesa driver shader cache |

### 3. Applies per-game overrides

Some games are incompatible with specific VKD3D options. The wrapper applies these overrides **before** Proton Experimental starts, so they take effect at VKD3D initialization time:

| AppID | Game | Override |
|-------|------|----------|
| 2358720 | Black Myth: Wukong | DXR disabled — NULL pointer crash in `d3d12core.dll` with `dxr`/`dxr11` active |

Add entries to `_NO_DXR_GAMES` in `proton` for other games with DXR crashes.

### 4. Starts/stops the GreenBoost gaming service

Before launching Proton Experimental, the wrapper calls:
```
systemctl start greenboost-gaming.service
```

This signals GreenBoost to shift VRAM priority from Ollama/LLM inference to the game (reduces KV-cache T1 reservation so the GPU is fully available for rendering). After the game exits, `greenboost-gaming.service` is stopped in a `try/finally` block, restoring normal inference priority. `systemctl` unavailability (e.g. inside pressure-vessel) is silently ignored.

### 5. Delegates to Proton Experimental

The wrapper calls `subprocess.run([proton_exp] + sys.argv[1:])`. Because `os.environ` was modified in place, Proton Experimental inherits all GreenBoost variables automatically without needing an explicit `env=` argument. Proton Experimental then handles the full Wine session: prefix setup, wineboot, VKD3D-Proton, DXVK, protonfixes, and the game itself.

Proton Experimental's install path is detected at runtime from `STEAM_COMPAT_CLIENT_INSTALL_PATH`, `STEAM_COMPAT_LIBRARY_PATHS`, and known Steam paths — never hard-coded.

## Architecture

```
Steam → SteamLinuxRuntime Sniper (container)
           └─ greenboost-proton/proton             ← GreenBoost wrapper (this script)
                ├─ _detect_nvidia()                auto-detect GPU, VRAM, RT support
                ├─ _detect_cpu()                   auto-detect P-cores, model
                ├─ inject env vars into os.environ  GREENBOOST_VULKAN, VKD3D_CONFIG, etc.
                ├─ per-game overrides               _NO_DXR_GAMES, etc.
                ├─ systemctl start gaming-service
                └─ subprocess.run(Proton Experimental/proton)
                        └─ Wine + VKD3D-Proton + DXVK    (from Proton Experimental)
                                └─ Vulkan loader
                                        └─ VK_LAYER_GREENBOOST_memory
                                                └─ NVIDIA driver
                                                        ├─ T1: VRAM (~336 GB/s)
                                                        └─ T2: DDR via DMA-BUF (~32 GB/s)
```

## Do you need it?

If you play games on Steam without Ollama/LLMs running, the three variables you'd lose without GreenBoost Proton are `GREENBOOST_VULKAN=1`, `PROTON_ENABLE_WAYLAND=1`, and `VKD3D_CONFIG=dxr,dxr11`. You could set these per-game in Steam's launch options. But you would also lose:

- Automatic log routing to `~/.local/share/greenboost/proton-logs/` (breaks `greenboost logs`)
- Shader cache consolidation under `~/.local/share/greenboost/proton-cache/`
- Gaming-service coordination (Ollama keeps consuming VRAM while you game)
- P-core-aware compiler thread tuning
- Per-game DXR blocklist

## GREENBOOST_NO_DXR — disabling ray tracing for crashing games

GreenBoost injects `VKD3D_CONFIG=dxr,dxr11` for RTX GPUs to enable DirectX Raytracing via VKD3D-Proton. Proton Experimental on its own does **not** set these flags. This means DXR-related crashes are caused by GreenBoost's own injection, not by Proton Experimental.

Some games crash when VKD3D-Proton initializes DXR, typically with a `EXCEPTION_ACCESS_VIOLATION` writing to address `0x0000000000000000` in `d3d12core.dll` on the render thread. This is a NULL pointer dereference triggered by a VKD3D-Proton DXR code path the game doesn't fully support.

**To fix a DXR crash for a specific game**, add this to Steam launch options:

```
GREENBOOST_NO_DXR=1 %command%
```

This tells the GreenBoost wrapper to skip injecting `dxr`/`dxr11` into `VKD3D_CONFIG` for that game. All other GreenBoost optimizations (virtual VRAM, Wayland, NVAPI, shader caches) remain active.

**Known DXR-incompatible games** are also permanently blocklisted in `_NO_DXR_GAMES` inside the `proton` wrapper script — these games never receive DXR injection regardless of user flags. Currently:

| AppID | Game |
|-------|------|
| 2358720 | Black Myth: Wukong |

If you notice `greenboost logs` reporting a DXR crash, it will show the exact `GREENBOOST_NO_DXR=1 %command%` fix directly in the Diagnostic Summary.

## Runtime env vars

| Variable | Default | Effect |
|----------|---------|--------|
| `GREENBOOST_DISABLE` | `0` | Set to `1` to skip all GreenBoost injection and run bare Proton Experimental |
| `GREENBOOST_VULKAN` | `1` | Enable/disable the Vulkan layer |
| `GREENBOOST_NO_DXR` | `0` | Set to `1` to disable DXR injection (`dxr`/`dxr11`) — use for games that crash on render thread with d3d12core.dll NULL pointer |
| `GREENBOOST_VKD3D_CONFIG` | `""` | Extra options appended to VKD3D_CONFIG (comma-separated) |
| `PROTON_ENABLE_WAYLAND` | `1` | Override to `0` to use XWayland instead |
| `VKD3D_DEBUG_OVERRIDE` | `""` | Override the VKD3D log level (default: `warn`) |

## Installation

```bash
cd ~/Dev/greenboost_main_branch/greenboost_proton_wayland
./install.sh
```

Installs to `~/.local/share/Steam/compatibilitytools.d/greenboost-proton/`. Restart Steam, then select **GreenBoost Proton** in a game's Properties → Compatibility.

Re-run after any GreenBoost upgrade. Proton Experimental updates apply automatically — no reinstall needed.

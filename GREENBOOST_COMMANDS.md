# GreenBoost Commands

Quick reference for all `greenboost` CLI commands.
Run `greenboost help` at any time to see this reference in your terminal.

---

## Core commands

| Command | Description |
|---------|-------------|
| `greenboost help` | Show this command reference |
| `greenboost status` | Show cuda memory pool (T1 VRAM / T2 RAM / T3 NVMe), module state, and system health |
| `greenboost clean memory` | Force-release T1 VRAM + T2 RAM + T3 immediately by unloading all inference models |
| `sudo greenboost load` | Load the GreenBoost kernel module with auto-detected parameters |
| `sudo greenboost unload` | Unload the kernel module (does not remove installed files) |
| `greenboost run <app>` | Force-activate shim on environments that do not use the kernel module, under virtualization environments |
---

## Diagnostics

| Command | Description |
|---------|-------------|
| `greenboost benchmark` | cuda memory pool bandwidth benchmark (T1 VRAM / T2 DDR / T3 NVMe) |
| `greenboost logs` | Aggregated log snapshot: kernel module events, services, Vulkan layer, Proton/VKD3D errors, AppArmor denials |
| `greenboost proton-logs` | Show Proton/VKD3D game logs from `~/.local/share/greenboost/proton-logs/` |
| `greenboost inference-logs` | Show inference service logs (Ollama, idle-reclaim, shader-boost) |
| `sudo greenboost clear logs` | Clear all GreenBoost log sources for a fresh diagnostic baseline (dmesg, journal, log files, Proton logs) |
| `sudo greenboost clear proton-logs` | Clear Proton/VKD3D game logs only |
| `sudo greenboost clear inference-logs` | Clear inference service logs only |
| `greenboost show-commands` | Display this CLI command reference in the terminal |

---

## Profile management

Hardware profiles store auto-detected parameters in `/etc/greenboost/profiles/`.
The active profile drives module parameters (`virtual_vram_gb`, `safety_reserve_gb`, etc.).

Running `greenboost profile` (no sub-command) opens the **interactive wizard** with a live
hardware panel and a numbered menu. Individual sub-commands are also available directly:

| Command | Description |
|---------|-------------|
| `greenboost profile` | Open Interactive wizard: create, activate, diff profiles |
| `sudo greenboost profile create` | Auto-detect hardware and create a new profile |
| `greenboost profile list` | List all available profiles |
| `greenboost profile show` | Show the currently active profile |
| `greenboost profile show <file>` | Show a specific profile file |
| `sudo greenboost profile activate <file>` | Switch active profile |
| `greenboost profile diff <file>` | Cross-check a profile against live hardware |

---

## Maintenance

| Command | Description |
|---------|-------------|
| `sudo greenboost install-sys-configs` | (Re-)install Ollama env, NVMe udev rules, CPU governor, hugepages, LD_AUDIT, idle-reclaim daemon |
| `sudo greenboost recover` | Attempt automatic recovery after a failed install or module load error |

---

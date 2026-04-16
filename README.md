# GreenBoost : CUDA Memory Orchestrator for NVidia GPUs

**Author:** Ferran Duarri  
**License:** GPL v2 (open-source)  
**Version:** 2.8.1

<div align="center">
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/greenboost)
</div>

---

**Disclaimer:** GreenBoost is an independent open-source project and is not affiliated with, endorsed by, or sponsored by NVIDIA Corporation. NVIDIA, CUDA, GeForce, and RTX are trademarks of NVIDIA Corporation.


**Important:** GreenBoost works alongside your existing NVIDIA drivers — it doesn't replace or modify them.

Thanks to all the contributors and the open-source community. GreenBoost wouldn't exist without them.

---

## 📦 Installation

```bash
git clone https://gitlab.com/IsolatedOctopi/greenboost.git
cd greenboost
sudo ./greenboost_setup.sh
```

The installer detects your hardware at runtime and presents a mode choice before making any changes: 
Full Install; (kernel module + system tuning).
Light Install; (kernel module only).

---

## 📚 Documentation

| Document | Purpose |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Technical architecture reference |
| [DOCUMENTATION.md](DOCUMENTATION.md) | Integration guide for AI inference tools |
| [CONTAINER_VM_MODE.md](CONTAINER_VM_MODE.md) | Docker, VM, WSL2, HPC setup |
| [GREENBOOST_COMMANDS.md](GREENBOOST_COMMANDS.md) | Full CLI reference |
| [GREENBOOST_PROTON.md](GREENBOOST_PROTON.md) | Gaming with GreenBoost Proton |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## 🛠️ How It Works

GreenBoost creates a unified memory pool from three physical tiers and presents it to every CUDA application as a single large GPU device, no application changes required.

GreenBoost intercepts CUDA memory calls so applications see a memory pool with GPU VRAM + System DDR RAM + SSD fallback instead of just GPU VRAM. All computation tensor computation runs on GPU. 

### Components

**`greenboost.ko`**, the kernel module. 
Allocates pinned system DDR pages (2 MB hugepages) and exports them as DMA-BUF file descriptors. The GPU imports these pages as CUDA external memory, zero CPU involvement in data movement. A watchdog thread monitors RAM and NVMe pressure.

**`libgreenboost_cuda.so`**, the CUDA shim. 
Intercepts CUDA calls. 
Small allocations pass through to the NVIDIA driver. 
Large ones that overflow T1 VRAM are redirected to T2 DDR via the kernel module. 
A phase detector (`INIT → MODEL_LOAD → INFERENCE → STEADY`) automatically classifies large allocations as KV cache during inference and keeps them in T1 at full GPU bandwidth.

**`libVkLayer_greenboost.so`**, the Vulkan layer. Reports the inflated T1+T2 VRAM total to Vulkan games and routes overflow allocations to T2 DDR via DMA-BUF when T1 is full.

---

## 🐳 Using GreenBoost in Containers, VMs, and WSL2

For environments where greenboost kernel module cannot be loaded; Docker, LXC, KVM guests, WSL2, shared HPC clusters, we have the Path B, which pins anonymous system RAM pages directly through the CUDA driver with no kernel module required. 

Jerry Nguyen's contribution (see Contributors), brought the solution for those scenarios.

See **[CONTAINER_VM_MODE.md](CONTAINER_VM_MODE.md)** for setup instructions specific to your environment.

---

## 🛡️ GreenBoost vs. CPU Offload: Why This Approach Is Better

Some AI tools try to speed up by offloading computations to your CPU when the GPU runs out of memory. Sounds good, but not a proper solution:

Modern CPUs aren't built for heavy AI math. They have a few specialized units (AVX), but they're designed for general-purpose work. Trying to run tensor computation on them creates a massive bottleneck, your CPU becomes the limiting factor, local AI inference slows down.

GreenBoost takes a different approach: instead of moving computation to the CPU, it moves memory around and lets the GPU do all the tensor computation. Since GPUs have specialized cores to perform those tasks. The GPU simply reaches across your PCIe connection to grab data from system RAM as needed, much faster than waiting for the CPU to do the work.

---

## 🎮 Gaming with GreenBoost (Beta)

Starting in version 2.8, GreenBoost can extend VRAM for games too. If a game normally hits VRAM limits (causing texture pop-in or crashes), GreenBoost reports a larger pool so the game thinks it has more memory to work with.


**To enable it for a Steam game:**
1. Right-click the game → **Properties**
2. Go to **Compatibility**
3. Select **"GreenBoost Proton"**

This is an early beta feature — performance gains depend on your game and hardware. See [GREENBOOST_PROTON.md](GREENBOOST_PROTON.md) for DLSS preset overrides, HDR, and MangoHud setup.

---

## 🧠 Contributors
- **Alan Sill** — setup scripts for Red Hat–based systems (Rocky Linux, AlmaLinux, RHEL)
- **Jerry Nguyen** — kernel-module-free path for containers and VMs
- **Giuseppe Marco Randazzo** — Debian Trixie support and Linux 6.12+ compatibility
- **Alexey Masolov** — fixes for PyTorch and vLLM on modern systems

---

## 📜 License

**GPL v2, open-source.** If you fork, modify, or reference this project, please credit Ferran Duarri.

```
Copyright (C) 2026 Ferran Duarri
```
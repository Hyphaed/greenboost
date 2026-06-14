# GreenBoost - CUDA Memory Orchestrator for NVidia GPUs

**Author:** Ferran Duarri
**License:** GPL v2 (open-source) + Commercial (dual licensed)
**Version:** 2.10

<div align="center">

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/greenboost)

</div>

---

**Disclaimer:** GreenBoost is an independent open-source project and is not
affiliated with, endorsed by, or sponsored by NVIDIA Corporation. NVIDIA,
CUDA, GeForce, and RTX are trademarks of NVIDIA Corporation.

**Important:** GreenBoost works alongside your existing NVIDIA drivers - it
doesn't replace or modify them.

Thanks to all the contributors and the open-source community. GreenBoost
wouldn't exist without them.

---

## What is GreenBoost?

> Your GPU runs out of VRAM, your model crashes, you buy a bigger GPU.
> GreenBoost is the third option.

In plain English: GreenBoost tricks CUDA into thinking your **GPU VRAM +
System RAM + NVMe** are all one giant pool of GPU memory. Your model loads,
inference runs on the GPU at full speed, and the parts that don't fit in
VRAM live in your system RAM - fetched on demand over PCIe.

Nothing in your model code changes. No retraining. No quantization (unless
you want it). It just works with Ollama, llama.cpp, vLLM, PyTorch, and
anything else that calls `cudaMalloc()`.

### Who is this for?

- **Newcomers to local LLMs:** you have a 12 GB or 16 GB GPU and want to
  run a 30 B+ model that needs 24 GB. Install GreenBoost, point Ollama at
  it, done.
- **Inference engineers:** you want to push context length or batch size
  past VRAM, without paying a 100× CPU offload penalty. GreenBoost keeps
  compute on the GPU; only memory crosses PCIe.
- **Cluster operators:** you have a few workstations with idle VRAM.
  GreenBoost's cluster mode turns them into "feeders" so one host can
  borrow VRAM from them over TCP.

If your workload is small enough to fit entirely in VRAM, GreenBoost adds
no benefit - and adds no overhead either, since the shim only intercepts
the allocations that overflow.

---

## Quick install

Works on **CUDA 12 and 13** (side-by-side installs are handled automatically)
and on both GCC- and Clang-built kernels (CachyOS, Arch/clang — no manual
`LLVM=1` needed).

```bash
git clone https://gitlab.com/IsolatedOctopi/greenboost.git
cd greenboost
sudo ./greenboost_setup.sh
```

The installer detects your hardware and asks which mode to use:

- **Full Install** - kernel module + system tuning (NVMe scheduler, swap,
  THP, hugepages). Best on a dedicated AI/ML workstation.
- **Light Install** - kernel module only. Safer on a daily-driver desktop
  where you don't want sysctls changed.

If you're inside a container, a VM, or WSL2 (no kernel module possible),
GreenBoost auto-falls back to **Path B** (no-kmod mode). See
[CONTAINER_VM_MODE.md](CONTAINER_VM_MODE.md).

---

## Documentation map

| Document | When to read it |
|---|---|
| [DOCUMENTATION.md](DOCUMENTATION.md) | You want the long-form story — the five layers, architecture, tiers, cluster, observability, all in one place |
| [greenboost_documentation_extension_official_nvidia.md](greenboost_documentation_extension_official_nvidia.md) | You are integrating GreenBoost into a new framework and need to know exactly where the shim departs from NVIDIA's documented CUDA behaviour (Chapter G, written in the style of the CUDA Programming Guide) |
| [CONTAINER_VM_MODE.md](CONTAINER_VM_MODE.md) | Docker, LXC, KVM, WSL2, HPC, Kubernetes |
| [GREENBOOST_COMMANDS.md](GREENBOOST_COMMANDS.md) | "What does `greenboost cluster` do again?" - full CLI reference |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## How it works

GreenBoost stitches three physical storage tiers into one "virtual VRAM"
that CUDA applications see as a single huge GPU:

```
┌──────────────────────────────────────────────────────────────┐
│   What your application sees: ONE giant CUDA device          │
└──────────────────────────────────────────────────────────────┘
   ▲ cudaMalloc / cuMemAlloc / cuLaunchKernel
   │
┌──┴────────────────────────────────────────────────────────────┐
│ libgreenboost_cuda.so   (LD_PRELOAD shim)                     │
│  • small allocs → pass through to the NVIDIA driver           │
│  • large allocs → overflow handler                            │
└──┬────────────────────────────────────────────────────────────┘
   │
   ▼
 ┌─────────────┐   ┌────────────────────────┐   ┌────────────┐
 │  T1: VRAM   │ → │  T2: System DDR RAM    │ → │  T3: NVMe  │
 │  (cudaMalloc│   │ (DMA-BUF pinned pages, │   │ (swap as a │
 │   real)     │   │  GPU reads over PCIe)  │   │  last fall-│
 └─────────────┘   └────────────────────────┘   └────────────┘
```

The kernel module (`greenboost.ko`) is the trick: it pins 2 MB hugepages
of system RAM and hands them to CUDA via `cuImportExternalMemory`
(zero-copy) or `cuMemHostRegister` (host-mapped). The GPU's PCIe engine
reads tensors straight from DDR; the CPU never touches the data.

**Two big things make this practical:**
1. The shim has a *phase detector* (`INIT → MODEL_LOAD → INFERENCE → STEADY`)
   that learns when KV cache is being allocated and pins it in T1 so
   attention runs at full GPU bandwidth.
2. Computation is **always on the GPU.** GreenBoost moves memory, never
   compute. CPU offload is what other tools do; CPU offload turns a 50
   tok/s setup into a 2 tok/s setup. GreenBoost stays at ~95 % of native
   GPU speed for the parts that fit, and degrades gracefully for the rest.

---

## Containers, VMs, WSL2: Path B

Some environments don't let you load kernel modules - Docker without
`--privileged`, KVM guests, WSL2, shared HPC nodes. In those, GreenBoost
runs in **Path B** mode: it skips `greenboost.ko` entirely and pins host
memory through `cuMemHostRegister`. Slightly higher per-allocation cost
(no zero-copy import) but otherwise the same behaviour.

Jerry Nguyen contributed this path. See
[CONTAINER_VM_MODE.md](CONTAINER_VM_MODE.md).

---

## Cluster mode - borrow VRAM from other machines

Got a couple of workstations with idle VRAM?  Each one runs:

```bash
sudo greenboost feed start
```

On your "host" (the one doing inference), each remote machine becomes a
**feeder**:

```bash
sudo greenboost connect 192.168.1.42
sudo greenboost connect 192.168.1.43
greenboost cluster        # interactive TUI showing all feeders + status
```

The shim treats the local VRAM + every feeder's VRAM + every feeder's DDR
as **one virtual device.** Layer weights that overflow are placed on the
fastest tier available - feeder GPU VRAM beats local DDR. Kernel launches
are dispatched to whichever feeder owns the data ("data-driven dispatch")
so compute happens close to memory.

The cluster fabric is secured with:
- Pre-shared key (PSK) auth + HKDF-derived session keys
- Per-message MAC (proto v4) to prevent tampering
- LAN-only bind by default; you opt in to WAN explicitly
- AppArmor profiles for the daemon

Full security model: [DOCUMENTATION.md § Cluster security](DOCUMENTATION.md).

---

## GreenBoost vs CPU offload - why the choice matters

Some tools (llama.cpp `-ngl`, accelerate `device_map="auto"`) handle VRAM
overflow by running parts of the model on the CPU. That works but it's
*slow* - typically 20-50× slower than the GPU portion. Inference becomes
CPU-bound.

GreenBoost goes the other way: **compute stays on the GPU, only memory
moves.** When a kernel needs a weight that lives in DDR, the GPU reads it
over PCIe (≈25 GB/s on PCIe 4.0 x16, ≈55 GB/s on PCIe 5.0). The CPU is
not in the data path.

End-to-end, you get something close to "GPU with 2-4× more VRAM" rather
than "GPU + CPU painfully sharing the work."

---

## gb-quant - quantize-to-fit (the fastest tier is the one you fit in)

Memory overflow gives you *capacity*, not *bandwidth*: a weight living in
system RAM is read at PCIe speed, ~12× slower than VRAM. For models that
are 1.5-3× your VRAM, GreenBoost's **gb-quant** layer is usually the better
answer: quantize the weights so the whole working set fits T1 VRAM and runs
at full GPU bandwidth.

```python
import gb_quant
report = gb_quant.quantize_to_fit(pipe_or_model, budget_gb=11.0)
```

- **Quality-first planner:** every component gets the *highest* precision
  that still fits the budget (bf16 > int8 > int4). Nothing is quantized
  that didn't need to be.
- **Self-contained:** the low-bit Triton GEMM kernels (Apache-2.0, in
  `third_party/`) and quantizer ship inside GreenBoost - your venv installs
  nothing extra.
- **Works with**: diffusers pipelines (two-phase text-encoder recipe
  included), HF causal LLMs (`gb_llm.py`), and pipelines you don't own via
  the `GB_QUANT_BUDGET_GB` environment hook. vLLM is served by the bundled
  plugin (`--quantization gemlite`).
- Measured on an RTX 5070 12 GB: a 9 B image model that needed ~7 min/image
  through DDR overflow runs at ~5 s/image quantized into VRAM, with no
  visible quality loss; a 12 B LLM (22.7 GiB bf16) fits in 6.2 GiB.

gb-quant and the memory tiers are complementary: quantize to fit first, and
let T2 DDR absorb only what genuinely exceeds the quantized footprint.

---

## Contributors

- **Alan Sill** ([@alansill](https://gitlab.com/alansill)) - setup scripts
  for Red Hat–based systems (Rocky Linux, AlmaLinux, RHEL).
- **Jerry Nguyen** ([@phubao](https://gitlab.com/phubao)) - kernel-
  module-free path for containers and VMs.
- **Giuseppe Marco Randazzo** ([@gmrandazzo](https://gitlab.com/gmrandazzo)) -
  Debian Trixie support and Linux 6.12+ compatibility fixes.
- **Alexey Masolov** ([@alexeymasolov](https://gitlab.com/alexeymasolov)) -
  PyTorch and vLLM compatibility fixes on modern systems.

---

## License

**GPL v2** - same licensing model as NVIDIA's official open-source
kernel modules (`github.com/NVIDIA/open-gpu-kernel-modules`).

Individual source files are MIT-licensed; when linked together into a
Linux kernel module the resulting binary is dual MIT / GPLv2.  See
`LICENSE` for the full text.

If you fork, modify, or reference this project, please credit Ferran
Duarri.

```
Copyright (C) 2026 Ferran Duarri
```

GreenBoost is an independent open-source project and is not affiliated
with, endorsed by, or sponsored by NVIDIA Corporation. NVIDIA, CUDA,
GeForce, and RTX are trademarks of NVIDIA Corporation.

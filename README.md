<div align="center">

# ⚡ 🧠 ⚡ GreenBoost ⚡ 🧠 ⚡

### CUDA Memory & Compute Orchestrator for NVIDIA GPUs

![Version](https://img.shields.io/badge/version-3.2-6C4FF6?style=flat-square)
![License](https://img.shields.io/badge/license-GPLv2%20%2B%20Commercial-blue?style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-12%20%7C%2013-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-supported-333333?style=flat-square&logo=linux)
![Status](https://img.shields.io/badge/status-daily%20driver-success?style=flat-square)

## Run bigger models on the GPU you already own.

**Turn GPU VRAM + System RAM + NVMe + idle LAN GPUs into one larger CUDA device.**

**No retraining. No code changes.** Just install **GreenBoost** and keep using **Ollama**, **llama.cpp**, **vLLM**, **PyTorch**, and **Diffusers** and/or step further, use directly **GB-Synapse + Greenboost MCP** for your pipelines.

</div>


<div align="center">

| Subsystem | Description |
|---------|-------------|
| 📦 **GB-Tiering** | Extend GPU memory using system RAM and NVMe. |
| 🗜️ **GB-Quant** | Compress model weights and the KV cache so larger models fit into available VRAM. |
| 📡 **GB-Dataflux** | Live telemetry exposed through the GreenBoost Dataflux MCP server and a web dashboard. Launch the UI with `greenboost dataflux-ui`. |
| 🌐 **GB-Cluster** | Borrow idle GPU and RAM resources from other machines on your local network. |
| 🔗 **GB-Synapse** | Translation proxy exposing Ollama-compatible endpoints (`/api/generate`, `/api/chat`, `/api/tags`, `/api/ps`) and the OpenAI-compatible `/v1/*` API on port `11434`. It sits in front of `gb-synapse` (`llama-server --rpc`), enabling cross-GPU RPC execution, GB-Quant, and proxy-side Dataflux telemetry (including tokens/sec). A genuine drop-in replacement for Ollama. |
| 🖥️ **GB-CLI** | Agentic terminal client, installed by Full Install — no separate setup. Open it with `gb` (or `greenboost-cli`); one-shot prompts with `gb -p "…"`; headless JSON subcommands for scripts (`gb rag-search …`). Always talks to GB-Synapse on `:11434` (Ollama **and** HuggingFace models via `greenboost synapse pull` / `index-ollama`). |
</div>

<div align="center">

**MCP servers** (LLM-facing): `greenboost-orchestrator` (central , full awareness via `greenboost_overview`, `optimize_inference`, `quant_advisor`, `flux_health`) · `greenboost-dataflux` (event log) · `greenboost-cluster` (live cluster state) · `greenboost-synapse` (serving control + CLI bridge) · `greenboost` (GB-CLI: rag/goals/factory)

</div>


<div align="center">

[Quick Start](#-quick-install) ·
[Documentation](#-documentation-map) ·
[Architecture](#-how-it-works) ·
[GB-Tiering](#-gb-tiering) ·
[GB-Quant](#-gb-quant) ·
[GB-Dataflux](#-gb-dataflux) ·
[GB-Cluster](#-gb-cluster) ·
[GB-Synapse](#-gb-synapse) ·
[GB-CLI](#-gb-cli) ·
[Changelog](CHANGELOG.md)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/greenboost)

</div>

---

> **Disclaimer:** GreenBoost is an independent open-source project and is
> not affiliated with, endorsed by, or sponsored by NVIDIA Corporation.
> NVIDIA, CUDA, GeForce, and RTX are trademarks of NVIDIA Corporation.
>
> **Important:** GreenBoost works alongside your existing NVIDIA drivers ,
> it doesn't replace or modify them.

Thanks to all the contributors and the open-source community. GreenBoost
wouldn't exist without them.

---

## What is GreenBoost?

**Your GPU ran out of VRAM.**
**Most tools tell you to buy a bigger GPU.**
**GreenBoost is the third option.**

It extends CUDA memory with system RAM, NVMe, model compression, and even
idle GPUs on your local network - allowing models much larger than your
card's VRAM to keep running, while computation stays on the GPU.

Under the hood it is six subsystems you mix and match: **GB-Tiering**
(VRAM → DDR → NVMe as one virtual pool), **GB-Quant** (compress the model's
weights at load and its KV cache on every decode step), **GB-Cluster**
(borrow an idle GPU and its RAM over LAN), **GB-Synapse** (GreenBoost's own
cluster-aware model server, drop-in for Ollama), **GB-Dataflux** (live
telemetry for all of the above), and **GB-CLI** (the agentic terminal client
on top). Each has its own section below, in that order.

Nothing in your model code changes. No retraining required. It just works
with Ollama, llama.cpp, vLLM, PyTorch, and anything else that calls
`cudaMalloc()`.

GreenBoost grew out of running local AI at home and constantly hitting the
same wall: not enough VRAM. My answer was never "buy a bigger GPU" - it was
"find the memory somewhere else and keep using what I have." DDR and NVMe
were the first places to look; a second GPU on the network (GB-Cluster,
below) is the next. It's the same idea DLSS proved years ago: software
squeezing more out of hardware that's already there isn't a workaround, it's
legitimate engineering. Every part of GreenBoost gets exercised daily against
real workloads on my own machines - a range of local LLMs served through
Ollama (dense and mixture-of-experts, small and large) and Hugging Face
diffusion pipelines for image and video - not just synthetic benchmarks.

### Who is this for?

- **Newcomers to local LLMs:** you have a 12 GB or 16 GB GPU and want to
  run a 30 B+ model that needs 24 GB. Install GreenBoost, point Ollama at
  it, done.
- **Inference engineers:** you want to push context length or batch size
  past VRAM, without paying a 100× CPU offload penalty. GreenBoost keeps
  compute on the GPU; only memory crosses PCIe.
- **Quality-conscious users:** your model is 1.5-3× your VRAM. GB-Quant
  quantizes both branches - weights at load, KV cache on every decode step -
  to fit it in VRAM with near-zero quality loss, and no offload penalty at
  all, because the whole working set runs at full GPU bandwidth.
- **Cluster operators:** you have a few workstations with idle VRAM.
  GB-Cluster turns them into "feeders" so one host can borrow VRAM and
  compute from them over TCP.

If your workload is small enough to fit entirely in VRAM, GreenBoost adds
no benefit - and adds no overhead either, since the shim only intercepts
the allocations that overflow.

---

## ⚡ Quick install

Works on **CUDA 12 and 13** (side-by-side installs are handled automatically)
and on both GCC- and Clang-built kernels (CachyOS, Arch/clang , no manual
`LLVM=1` needed).

```bash
git clone https://gitlab.com/IsolatedOctopi/greenboost.git
cd greenboost
sudo ./greenboost_setup.sh
```

The installer detects your hardware and asks which mode to use:

- **Full Install** - kernel module + system tuning (NVMe scheduler, swap,
  THP, hugepages) + GB-CLI + the MCP servers. Best on a dedicated AI/ML
  workstation.
- **Light Install** - kernel module only. Safer on a daily-driver desktop
  where you don't want sysctls changed.

If you're inside a container, a VM, or WSL2 (no kernel module possible),
GreenBoost auto-falls back to **Path B** (no-kmod mode). See
[CONTAINER_VM_MODE.md](CONTAINER_VM_MODE.md).

---

## 📚 Documentation map

| Document | When to read it |
|---|---|
| [DOCUMENTATION.md](DOCUMENTATION.md) | You want the long-form story , the subsystems, architecture, tiers, cluster, observability, all in one place |
| [greenboost_documentation_extension_official_nvidia.md](greenboost_documentation_extension_official_nvidia.md) | You are integrating GreenBoost into a new framework and need to know exactly where the shim departs from NVIDIA's documented CUDA behaviour (Chapter G, written in the style of the CUDA Programming Guide) |
| [CONTAINER_VM_MODE.md](CONTAINER_VM_MODE.md) | Docker, LXC, KVM, WSL2, HPC, Kubernetes |
| [GREENBOOST_COMMANDS.md](GREENBOOST_COMMANDS.md) | "What does `greenboost cluster` do again?" - full CLI reference |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## 🔧 How it works

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

### Containers, VMs, WSL2: Path B

Some environments don't let you load kernel modules - Docker without
`--privileged`, KVM guests, WSL2, shared HPC nodes. In those, GreenBoost
runs in **Path B** mode: it skips `greenboost.ko` entirely and pins host
memory through `cuMemHostRegister`. Slightly higher per-allocation cost
(no zero-copy import) but otherwise the same behaviour.

Jerry Nguyen contributed this path. See
[CONTAINER_VM_MODE.md](CONTAINER_VM_MODE.md).

---

## 📦 GB-Tiering

**Extend GPU memory using system RAM and NVMe , the T1/T2/T3 tier layer that
every other subsystem allocates through.**

GB-Tiering is the engine behind the diagram above: the LD_PRELOAD shim
(`libgreenboost_cuda.so`) plus the kernel module (`greenboost.ko`) place each
allocation in the fastest tier that still has room, and spill only what
doesn't fit.

| Tier | Backing | Used for |
|---|---|---|
| **T1** | GPU VRAM (real `cudaMalloc`) | KV cache first, then as many weights as fit |
| **T2** | System DDR, DMA-BUF pinned hugepages | Weights that overflow VRAM, read by the GPU over PCIe |
| **T3** | NVMe | Last-resort overflow |

Placement is driven by **access frequency, not allocation order**: the KV
cache is re-read on every decode step, so it is reserved in T1 first and the
weights fill whatever VRAM is left. A single allocation larger than free VRAM
is *split* rather than dumped wholesale into DDR - VRAM is driven toward ~90 %
occupancy, and only the genuine remainder spills.

```bash
greenboost vitals         # live TUI: tiers, VRAM, pressure, phase
greenboost health-check   # one-shot PASS/FAIL/WARN across module/VRAM/T2/T3/cluster
greenboost capabilities   # installed/running shim feature manifest
```

From Python, `gb_tiering.py` is the one import for the tier layer
(`ModelTierManager`, `Tier`, `MemPoolManager`, and `tiering_status()` for live
state); over MCP, `tiering_status` / `greenboost_status` expose the same thing
to an assistant.


GreenBoost way: **compute stays on the GPU, only memory moves.** 
When a kernel needs a weight that lives in DDR, the GPU reads it over PCIe (≈25 GB/s on PCIe 4.0 x16, ≈55 GB/s on PCIe 5.0). 
The CPU is not in the data path.

End-to-end, you get something close to "GPU with 2-4× more VRAM" rather than "GPU + CPU painfully sharing the work." Yet System DDR is slower than GPU VRAM.

GreenBoost applies techniques to avoid CPU spillover always that harms ai inference speed and/or quality. It provdies a direct VRAM <> System RAM + GPU transfer of data, without need for a copy at CPU which only adds more latency. 

---

## 🗜️ GB-Quant

**Compress model weights and the KV cache so larger models fit into available
VRAM , the fastest tier is the one you fit in.**

GB-Quant has **two branches**, used alone or together:

- the **weights branch** shrinks the model's weights once, at load
  (`gb_quant.py`, planner + low-bit GEMM kernels);
- the **KV branch , TurboQuant** shrinks the KV cache continuously, on every
  decode step (`gb_attn.py`, and the system-wide `greenboost turboquant`
  toggle).

They solve different bottlenecks, which is why both exist under one subsystem.

### Weights branch , quantize-to-fit

thanks to https://github.com/dropbox/gemlite

Memory overflow gives you *capacity*, not *bandwidth*: a weight living in
system RAM is read at PCIe speed, ~12× slower than VRAM. For models that
are 1.5-3× your VRAM, quantizing is usually the better answer than spilling:
shrink the weights so the whole working set fits T1 VRAM and runs at full
GPU bandwidth.

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

The low-bit GEMM kernel underneath (GemLite, `scaled_mm` fp8, bf16
passthrough) is itself pluggable via `gb_kernel_backends.py` and
`GB_KERNEL_BACKEND`.

### KV branch , TurboQuant

The KV cache is re-read on *every* decode step, so its bandwidth cost
compounds over a whole generation - a different bottleneck than the one-time
weight load. TurboQuant quantizes the K and V tensors as attention runs,
freeing PCIe/VRAM bandwidth for the rest of the model without materially
changing output quality.

System-wide, zero code changes:

```bash
sudo greenboost turboquant on     # Ollama KV cache -> q4_0, GREENBOOST_TURBOQUANT=1
greenboost turboquant status
sudo greenboost turboquant off    # back to q8_0
```

Or fine-grained control from Python (`gb_attn.py`):

```python
from gb_attn import turboquant_attention

# Asymmetric: K keeps more precision (needed for Q·Kᵀ), V needs less
with turboquant_attention(k_bits=4, v_bits=3):
    output = model(input)
```

- **Asymmetric by design:** K tensors get full TurboQuant (PolarQuant +
  a 1-bit QJL residual, preserving inner products for attention scores);
  V tensors get PolarQuant only (MSE-optimal, no inner-product needed).
- **Bandwidth:** 3-8× reduction depending on bit widths; `k_bits=4,
  v_bits=3` gets ~4× with +0.23% PPL - effectively free.
- **Layer-adaptive:** the first two and last two layers automatically get
  +1 bit of K precision, where quality is most sensitive.

GB-Quant and GB-Tiering are complementary: quantize weights to fit first,
compress the KV cache so attention doesn't reopen the bandwidth problem, and
let T2 DDR / T3 NVMe absorb only what genuinely exceeds the quantized
footprint.

---

## 📡 GB-Dataflux

**Live telemetry exposed through the GreenBoost Dataflux MCP server and a web
dashboard , a flight recorder for your inference.**

New in v3.2. GreenBoost continuously logs what your whole setup is doing ,
VRAM used/free, GPU and CPU load, temperature and power, KV-cache pressure,
memory-tier moves, quantization decisions, measured tokens/sec, and (once a
feeder is connected) per-machine cluster throughput , so you can look back at
what actually happened, not just guess from a single snapshot.

```bash
greenboost dataflux-ui        # opens a live web page at :8799
```

![GreenBoost dataflux web UI](greenboost_dataflux_ui.png)

It works standalone on a single machine, no feeder required, and auto-refreshes
every 5 seconds. Full reference: [DOCUMENTATION.md § dataflux](DOCUMENTATION.md).

`gb_dataflux_mcp.py` exposes the same log read-only over MCP
(`dataflux_summary`, `dataflux_events`, `dataflux_errors`, `dataflux_tok_s`,
`greenboost_status`/`capabilities`/`pilot`), so any MCP-compatible AI
assistant can query live capability state and dataflux history directly
instead of you opening the web UI. Every other subsystem emits into this one
log, which is what makes a placement or quantization decision traceable after
the fact.

```bash
greenboost pilot          # dataflux-driven advice on what lever to pull next
```

---

## 🌐 GB-Cluster

**Borrow idle GPU and RAM resources from other machines on your local network.**

💻 Got a laptop lying around collecting dust with an idle NVIDIA GPU inside?
Put it to work: start `greenboost cluster` and watch it in `greenboost
dataflux-ui` +  `greenboost dataflux MCP` (direct dataflux connection for your inference pipelines).

Cluster mode has existed since v2.9, but **v3.2 is the first release where
it's genuinely working**, not just alpha-stage.
GB-Cluster brings local AI inference to the next level, because now you can
orchestrate the hardware you already have on your own network, instead of an
idle laptop GPU just sitting there. It's also been polished through real daily
use, mostly Hugging Face diffusion pipelines, not just synthetic benchmarks.

Start a feeder on the idle machine:

```bash
sudo greenboost feed start
```

On your "host" (the one doing inference), each remote machine becomes a
**feeder**:

```bash
sudo greenboost connect 192.168.1.42
greenboost cluster        # interactive TUI showing all feeders + status
```

The shim treats the local VRAM + every feeder's VRAM + every feeder's DDR
+ every feeder's NVMe as **one virtual device** - a model that doesn't fit
on your card can now overflow into a whole second machine, not just your
own RAM.

Where this pays off *today*, measurably: `gb_cluster.py` gives any PyTorch
pipeline a simple API to hand off a whole stage (e.g. text encoding) or the
tail of a model to a feeder's GPU (`offload_tail_blocks`, backed by
`gb_remote_blocks.py` - a persistent, model-agnostic tensor-RPC worker that
holds the tail transformer blocks in feeder VRAM so only activations cross
the wire on every step). On a real diffusion pipeline at home, combining
that with a few related fixes took a generation from ~5.5 minutes down to
~42 seconds - about **7.8× faster**, from putting a second consumer GPU to
work instead of leaving it idle. For GGUF models, `gb_placement.py` picks
between that split and llama.cpp's own RPC tensor-split (GB-Synapse, below)
automatically, always preferring a cluster feeder over silently dropping
below fp8 precision (`GB_PLACEMENT=1`).

Live cluster state is queryable at any moment over the `greenboost-cluster`
MCP server (`cluster_status`, `cluster_snapshot`, `cluster_feeders`), and every
dispatch leaves a GB-Dataflux event, so a feeder is never a black box.

The cluster fabric is secured with:
- Pre-shared key (PSK) auth + HKDF-derived session keys (`feeders genkey`,
  `feeders export-key` / `import-key` to distribute one across machines)
- Per-message MAC (proto v4) to prevent tampering
- LAN-only bind by default; you opt in to WAN explicitly
- AppArmor profiles for the daemon

Full security model: [DOCUMENTATION.md § Cluster security](DOCUMENTATION.md).

---

## 🔗 GB-Synapse

**GreenBoost's own model server and Ollama-compatible proxy on `:11434`,
spread across the cluster.**

New in v3.2. Ollama only serves models from its own registry. GB-Synapse
pulls GGUFs straight from any HuggingFace repository, gated or public, given
a token, and also indexes GGUFs Ollama already downloaded, so one tool sees
both. For clustering, it hands the cross-machine split to llama.cpp's own
RPC backend, real layer-granular tensor split, only activations cross the
wire, while the GreenBoost shim keeps extending each node's own share into
that node's local RAM/disk underneath it.

```bash
sudo greenboost synapse login              # store a HuggingFace token
sudo greenboost pull <repo>[:quant]        # download a GGUF
greenboost synapse run <model>             # serve it, cluster-aware, on :11434
```

The proxy in front of `llama-server` speaks Ollama's API (`/api/generate`,
`/api/chat`, `/api/tags`, `/api/show`, `/api/ps`), OpenAI's `/v1/*`, and
HuggingFace's TGI API on the same port, so existing tools don't need to know
anything changed - a genuine drop-in replacement for Ollama. It measures
tokens/sec proxy-side and emits it to GB-Dataflux, which closes the loop
between an orchestration decision and its real, client-observed throughput.

Serving control is also exposed over the `greenboost-synapse` MCP server
(`synapse_status`, `synapse_models`, `synapse_ps`, `synapse_recommend`,
`synapse_doctor`; `synapse_serve`/`synapse_stop` are confirmation-gated so an
assistant can't yank a live `:11434` out from under you).

Full reference: [GREENBOOST_COMMANDS.md § gb-synapse](GREENBOOST_COMMANDS.md#-gb-synapse-huggingface-native-cluster-distributed-gguf-serving).

When vLLM isn't installed, GB-Synapse falls back to `gb_llm_server.py`, a
minimal OpenAI-compatible server built directly on `gb_llm.py` - same API
surface, no extra dependency.

---

## 🖥️ GB-CLI

**Agentic terminal client, installed by Full Install , no separate setup.**

```bash
gb                        # interactive agent in your terminal
gb -p "summarize this repo"   # one-shot prompt
gb rag-search "kv cache"      # headless JSON subcommand, for scripts
```

GB-CLI always talks to **GB-Synapse on `:11434`**, so whatever you served
(Ollama-indexed or HuggingFace-pulled GGUF, single-GPU or RPC-split across the
cluster) is what the agent runs on, with GB-Quant and GB-Tiering underneath and
every turn's measured tokens/sec landing in GB-Dataflux.

Full Install deploys it into `/usr/local/lib/greenboost/cli-venv` and puts
`gb` + `greenboost-cli` on your PATH; the `greenboost` MCP server exposes its
rag/goals/factory surface to other assistants.


---

## 🙌 Contributors

- **Alan Sill** ([@alansill](https://gitlab.com/alansill)) - setup scripts
  for Red Hat–based systems (Rocky Linux, AlmaLinux, RHEL).
- **Jerry Nguyen** ([@phubao](https://gitlab.com/phubao)) - kernel-
  module-free path for containers and VMs.
- **Giuseppe Marco Randazzo** ([@gmrandazzo](https://gitlab.com/gmrandazzo)) -
  Debian Trixie support and Linux 6.12+ compatibility fixes.
- **Alexey Masolov** ([@alexeymasolov](https://gitlab.com/alexeymasolov)) -
  PyTorch and vLLM compatibility fixes on modern systems.

---

## 🔗 Non direct contributors

- **Mobius Labs** thanks to https://github.com/dropbox/gemlite, big part of gb-quant is based on it

---

## 💡 Inspirational sources

[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).


## 📄 License

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
</content>
</invoke>

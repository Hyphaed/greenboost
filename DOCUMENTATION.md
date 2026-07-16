# 📘 GreenBoost v3.2 , integration guide for inference tools

> **GreenBoost** is a CUDA Memory & Compute Orchestrator for NVIDIA GPUs: it
> tiers system RAM/NVMe into VRAM, quantizes models to fit (gb-quant), and pools
> LAN machines into one virtual device. This guide shows how to wire each
> inference tool into it.

Check [GREENBOOST_COMMANDS.md](GREENBOOST_COMMANDS.md) for all the available commands.
Check [CHANGELOG.md](CHANGELOG.md) for what changed in each version.

<details>
<summary><strong>📑 Table of contents</strong> (click to expand)</summary>

**Getting oriented**
- [Reading this document](#reading-this-document)
- [What's new in v3.2](#whats-new-in-v32)
- [What's new in v3.1](#whats-new-in-v31)
- [What's new in v3.0](#whats-new-in-v30)
- [Build compatibility](#build-compatibility)
- [How GreenBoost hooks in](#how-greenboost-hooks-in)
- [The five GreenBoost layers](#the-five-greenboost-layers)
- [The official CUDA extension document](#the-official-cuda-extension-document)

**Framework integration guides**
- [Ollama](#ollama)
- [vLLM](#vllm)
- [PyTorch scripts](#pytorch-scripts)
- [text-generation-inference (TGI)](#text-generation-inference-tgi)
- [CTranslate2](#ctranslate2)
- [Hugging Face Transformers](#hugging-face-transformers)
- [ExLlamaV3](#exllamav3)
- [TensorFlow / Keras](#tensorflow--keras)

**Direct / advanced control**
- [PyTorch - direct tensor allocation control](#pytorch---direct-tensor-allocation-control)
- [Hugging Face Transformers - direct KV cache control](#hugging-face-transformers---direct-kv-cache-control)
- [vLLM - direct KV cache placement](#vllm---direct-kv-cache-placement)
- [Ollama 0.18+ and the CUDA VMM path](#ollama-018-and-the-cuda-vmm-path)
- [KV Cache Fine-Tuning - Bypassing the Phase Heuristic](#kv-cache-fine-tuning---bypassing-the-phase-heuristic)
- [Session Priority Management](#session-priority-management)

**Debugging & telemetry**
- [Verify GreenBoost is active](#verify-greenboost-is-active)
- [Live debug signals - real-time telemetry during inference](#live-debug-signals---real-time-telemetry-during-inference)
- [Debug Vitals](#debug-vitals)

**Performance & features**
- [TurboQuant - global toggle](#turboquant---global-toggle-v29)
- [gb-quant - weight quantize-to-fit](#gb-quant---weight-quantize-to-fit-v30)
- [Optimized inference configuration generator](#optimized-inference-configuration-generator-v29)
- [Cluster - connect remote machines as GPU memory and compute feeders](#cluster---connect-remote-machines-as-gpu-memory-and-compute-feeders)
- [gb-synapse - HuggingFace-native, cluster-distributed GGUF serving](#gb-synapse--huggingface-native-cluster-distributed-gguf-serving-v32)
- [dataflux - the flight recorder](#dataflux--the-flight-recorder-port-8799)
- [Synapse CLI - terminal AI assistant with GreenBoost integration](#synapse-cli---terminal-ai-assistant-with-greenboost-integration-v29)
- [Diffusion Models - Best Practices](#diffusion-models---best-practices)
- [Long-running stability monitor](#long-running-stability-monitor-v30)
- [Worker pool](#worker-pool-v30-opt-in)

**Other**
- [Sister project: GreenBoost Gaming Suite](#sister-project-greenboost-gaming-suite)

</details>

---

## Reading this document

Two audiences read this file:

- **Newcomers** who installed GreenBoost and want to make a specific
  inference tool work. Jump straight to your tool's section
  ([Ollama](#ollama), [vLLM](#vllm), [PyTorch](#pytorch-scripts),
  [Hugging Face](#hugging-face-transformers), …). The first paragraph
  of each section is a copy-pasteable recipe; everything after that is
  optional tuning.
- **Engineers integrating GreenBoost into infrastructure** who want
  the architectural model - phase detector, KV-cache pinning, cluster
  fabric, security. Read [§ How GreenBoost hooks in](#how-greenboost-hooks-in)
  first, then the [§ Cluster](#cluster--connect-remote-machines-as-gpu-memory-and-compute-feeders)
  and [§ Debug Vitals](#debug-vitals) sections.

If you only want to *use* GreenBoost with one inference tool, you can
skip the architecture sections entirely. The shim is transparent - your
existing Python script does not change.

---

## What's new in v3.2

The 3.2 cycle's headline is `greenboost cluster`: the first release where it
works well enough to actually recommend, not just experiment with. See
`CHANGELOG.md` for the full writeup and a real measured result (a diffusion
pipeline going ~7.8× faster by putting a second GPU to work). Everything else
this cycle builds on the v3.1 telemetry/orchestrator foundation:

**Cluster (`greenboost_netd.c`, `greenboost_netc.c`, `gb_cluster.py`, new)**
- Feeder link reliability overhaul: dedicated heartbeat/reconnect thread,
  chunked large transfers, zombie-daemon cleanup, node-parity checks so a
  host and feeder running mismatched builds are caught at `connect` time.
- `gb_cluster.py` - a stable Python API (`shim_env`, `feeders`,
  `run_stage_on_feeder`, `offload_tail_blocks`, `parallel_map_with_feeder`)
  any PyTorch pipeline can import to use a feeder's GPU for a whole
  stage or the tail of a model.
- New key-management and daemon commands: `feeders genkey`,
  `feeders export-key` / `import-key`, `feeders redeploy-netd`.
- Running a CUDA kernel directly on a feeder against data that already lives
  there has been demonstrated for a couple of basic operations, but it isn't
  reliable enough for real inference workloads yet , that stays
  research-in-progress and is not part of what this release ships.

**gb-synapse , HuggingFace-native, cluster-distributed GGUF serving
(`gb_synapse.py`, `gb_synapse_api.py`, new)**
- Pulls GGUFs directly from any HuggingFace repo (gated or public, given a
  token), indexes GGUFs Ollama already downloaded, and serves either through
  a matched llama.cpp build on every cluster node.
- Cross-node split rides llama.cpp's own `--rpc` backend (layer-granular,
  only activations cross the wire) while the GreenBoost shim keeps extending
  each node's own share into that node's local RAM/disk underneath it.
- One port (`:11435`) speaks Ollama's API, OpenAI's API, and HuggingFace TGI
  at once, so existing tooling doesn't need to change. See § gb-synapse below
  and [GREENBOOST_COMMANDS.md](GREENBOOST_COMMANDS.md#-gb-synapse-huggingface-native-cluster-distributed-gguf-serving).

**`greenboost-dataflux` MCP server (`gb_dataflux_mcp.py`), grown from 3 tools
to 8**
- Read-only queries over the dataflux log for any MCP-compatible AI
  assistant: `dataflux_summary`, `dataflux_events` (now filterable by event
  kind), `dataflux_errors`, plus four new ones , `dataflux_kinds`,
  `dataflux_tier_moves`, `dataflux_quantization`, `dataflux_tok_s`. See §
  dataflux below.

**Reactive orchestration (`gb_orchestrator.py`, `gb_control.py`, new
`gb_reactive.py`, `gb_topology.py`)**
- Expanded from the 3 feedback loops introduced in v3.1 to cover thermal
  stress, memory-bandwidth stress, clock throttling, and a predictive
  KV-cache grower gated behind six independent safety checks.
- Opt-in continuous OS tuning for the root daemon (CPU governor, GPU
  clock/power limits, kernel memory tunables), reversible with the new
  `greenboost tune-revert`. Off by default (`GB_ORCH_ACTUATE=1` to enable) -
  GreenBoost only observes and reports until you turn it on.

**dataflux , a flight recorder for the whole system (`gb_dataflux.py`, new)**
- Continuous local event log covering VRAM/GPU/CPU/temp/power, KV-cache
  pressure, memory-tier movement, quantization decisions, and per-machine
  cluster throughput once a feeder is connected.
- `greenboost dataflux-ui` (or `python3 gb_dataflux.py serve`) opens a live,
  auto-refreshing web page at `:8799`; works standalone, no feeder required.
  See § dataflux below.

**Telemetry (`gb_telemetry.py`), polished through daily use**
- PCIe link-health tracking, host-level pressure metrics (CPU/memory/I/O via
  the kernel's pressure-stall accounting), feeder-aware telemetry so a
  connected feeder's GPU health rolls into the same picture as the local one,
  a no-pip-dependency GPU-metrics fallback, and a measured-tokens-per-second
  signal that closes the loop between orchestration decisions and the actual
  speed a model delivered.

**Hot/cold residency + observability**
- The kernel evictor now prefers genuinely cold buffers over pure LRU order
  (see full writeup in `CHANGELOG.md`), with a new eBPF-based tracer and
  three new commands: `greenboost top`, `greenboost residency`,
  `greenboost faults`.

**Two experimental, honestly-reported performance layers**
- `gb_prefetch.py` - overlaps loading a transformer's next layer with
  computing the current one. Hasn't beaten the no-prefetch baseline on a
  gb-quant-placed model in testing so far; useful for models gb-quant hasn't
  already placed.
- `gb_diffcache.py` - a TeaCache/DeepCache-style activation cache for
  diffusion models that skips recomputing a step when little has changed.

**Fixes**
- Full Install no longer reinstalls Ollama on every run - it checks the
  latest release first and only updates when actually behind.

**AI memory OS tranche + polish roundup**
- `gb_monitor.py` - canonical read-only telemetry/capability client unifying
  the shim_stats parsers and ioctl mirrors that had grown independently.
  Backs `greenboost capabilities` (§ below), which prints the installed/
  running shim's feature manifest.
- `gb_pilot.py` - a read-only "instrument panel" over the dataflux log;
  turns recorded trends into evidence-backed advice naming the exact
  `gb_control` lever to pull, never actuates anything itself. New
  `greenboost pilot` command and MCP tool.
- `gb_dataflux_mcp.py` gained `greenboost_status`, `greenboost_capabilities`,
  and `greenboost_pilot` tools alongside the existing dataflux-log tools.
- `gb_synapse_tools.py` - text-based tool-call injection for GGUFs that
  don't emit native OpenAI `tool_calls`, shared by gb-synapse serving.
- `gb_llm_server.py` - minimal OpenAI-compatible server on `gb_llm.py`;
  the "gbquant" engine fallback gb-synapse uses when vLLM isn't installed.
- `gb_placement.py` - fp8-floor cluster-fit planner (`GB_PLACEMENT=1`);
  prefers a connected feeder over dropping below fp8, and arbitrates
  between GGUF RPC tensor-split and PyTorch `offload_tail_blocks`.
- `gb_kernel_backends.py` - pluggable low-bit GEMM backend selector for
  gb-quant (`GB_KERNEL_BACKEND`: GemLite default, `scaled_mm` fp8, bf16
  passthrough, CUTLASS reserved).
- `gb_nvml_ctypes.py` - pynvml-compatible NVML binding over `ctypes`, so
  telemetry works without `nvidia-ml-py` pip-installed.
- Cluster fabric zstd compression on host-to-device payloads, negotiated at
  handshake (protocol v3, backward compatible with v2 feeders).
- Phase-aware KV prefetch stats mode (`GREENBOOST_KV_PREFETCH`), exposing
  `kv_prefetch_*` telemetry counters.
- CUTLASS sm_120a NVFP4 scaffold (`third_party/gb_cutlass`,
  `GB_CUTLASS_ENABLE`) - groundwork only, not in the default kernel path.
- TurboQuant Triton autotune persistence + `turboquant_attention_from_env`.
- Proxy-owned tok/s measurement on gb-synapse's Ollama-surface streaming.
- Four default-safe efficiency wins: version-scoped gb-quant autotune cache
  (opt-out `GB_QUANT_NO_AUTOTUNE_CACHE=1`), transparent zstd for T3
  checkpoints (opt-out `GB_T3_COMPRESS=0`), phase-aware T1 workspace
  reserve (`GREENBOOST_T1_WORKSPACE_MB`, default off), opt-in tiered
  precision (`GB_QUANT_TIERED_PRECISION=1` - fp8 stays the default).

821 tests green.

---

## What's new in v3.1

The 3.1 cycle stitches a reactive telemetry layer into the CUDA shim and replaces
four installed-but-inert systemd daemons with a single unified Python runtime:

**Runtime stack (new modules)**
- `gb_supervisor` , one long-running daemon that subsumes `greenboost-recovery`,
  `greenboost-sentinel`, `greenboost-vram-watchdog`, and `greenboost-idle-reclaim`.
  One process, one log stream, zero redundant pynvml connections.
- `gb_orchestrator` , signal-driven actuator that closes three feedback loops
  previously left open: ECC responder, VRAM pressure ratchet, idle-reclaim gate.
- `gb_control` , the single place that mutates GreenBoost runtime state, backed by
  sysfs → ioctl → `/run/greenboost/control` in that priority order.
- `gb_nvml` , unified pynvml singleton shared by all of the above (was duplicated
  across four modules with divergent query sets).
- `gb_reactive` , stdlib-only Signal/Computed primitives (debounce, EWMA, hysteresis)
  used by the orchestrator; no external dependencies.
- `gb_vitals_helper` , in-process pynvml for the `vitals`/`cluster` commands,
  removing the nvidia-smi subprocess fork that previously ran on every refresh tick.

**Shim**
- `greenboost_cuda_shim.c` extended to publish live ECC, power, and PCIe health
  signals at 500 ms resolution , previously these were logged only, never acted on.

**Fixes**
- `vmm_override` PLT symbol resolution fixed for Blackwell (cc 12.x) targets.
- `greenboost_setup.sh` refactored: duplicate code consolidated, TUI refresh loop
  de-spammed, `feeders upgrade-greenboost` atomic install path hardened.

---

## What's new in v3.0

The 3.0 cycle brought a multi-month hardening campaign on the cluster
fabric and the LD_PRELOAD shim, a full Python quantization stack (gb-quant),
and embedded DCGM telemetry. No new user-visible APIs to learn for existing
workflows, but plenty of fixes you'll notice in production:

**Security**
- Cluster fabric upgraded from proto v3 to **proto v4** with mutual
  authentication: both the host *and* each feeder prove knowledge of
  the PSK. A man-in-the-middle who only replays a prior handshake can
  no longer impersonate either side.
- Every message after the handshake carries a truncated HMAC-SHA256
  computed over (header ‖ payload) keyed by an HKDF-derived session
  key. Tampering is detected at receive time, not after parsing.
- A v4 host still talks to a v3 feeder if `GREENBOOST_PSK_V4=0` is set
  (so you can roll feeders one at a time); default is v4-only.

**Robustness**
- 117-test pytest suite covering wire-protocol, PSK loading, LAN
  filter, HKDF, exported symbols, and end-to-end handshake.
- Kernel-module DMA path holds a `dma_resv` reservation across every
  device mapping, with `active_mappings` reference counting so the
  eviction worker can never free memory the GPU is reading.
- `cuFuncGetParamInfo` (CUDA 12.3+) probes kernels for argument
  arity, so `cuLaunchKernel` argument scanning has correct bounds
  instead of guessing.
- The hot allocation path is fully big-endian-safe - every wire field
  is read via `GB_LE_U16/U32/U64` accessors. The full BE port still
  requires opt-in (`-DGREENBOOST_BE_INCOMPLETE=1`).
- **CUDA 13 / cu130 compatibility**: the shim no longer fails with
  `cudaErrorDeviceUninitialized` (998) when frameworks call
  `cuMemGetInfo_v2` or `cuDeviceTotalMem_v2` before the first
  `cudaSetDevice`. On `INVALID_CONTEXT` (201) the shim falls back to
  the runtime API which retains the primary context transparently.
- **Clang-built kernel support**: the Makefile auto-detects
  `CONFIG_CC_IS_CLANG=y` and passes `LLVM=1` to the kernel sub-make,
  fixing build failures on CachyOS, Arch/clang, and Gentoo/clang
  kernels. `dkms.conf` now sets explicit `MAKE`/`CLEAN` so DKMS
  auto-rebuilds on kernel updates also use the correct toolchain.

**New tools**
- `greenboost stability` - a Python long-running monitor that watches
  `/run/greenboost/shim_stats` for monotone-counter regressions,
  tier-gauge leaks (memory still allocated after idle), and
  fragmentation drift over hours of inference.
- `greenboost feeders diag` - host-side T1/T2/T3 + compute test
  against a remote feeder.
- A worker-pool scaffold inside `greenboost-netd` (opt in with
  `GREENBOOST_WORKERS=N`) so a slow CUDA op on one client no longer
  blocks the whole reactor.

**Decoupled**
- The experimental Vulkan layer that let games use the GreenBoost
  pool has been moved into a sister project: **GreenBoost Gaming
  Suite**. The CUDA-inference path stays small and auditable; gamers
  get a GTK4 GUI instead of CLI. See [§ Sister project: GreenBoost
  Gaming Suite](#sister-project-greenboost-gaming-suite).

---

## MCP & A2A surface (current — supersedes tool counts quoted in per-version notes above)

GreenBoost exposes five MCP servers (each registered in `.mcp.json` for a
fresh clone, or via `greenboost register-mcp` for the installed path) plus
one A2A gateway. Per-version bullets above (e.g. "grown from 3 tools to 8")
describe that release's delta only — this section is the live count.

| Server | Entry file | Tools |
|---|---|---|
| `greenboost-dataflux` | `gb_dataflux_mcp.py` | 19 — event-log queries (`dataflux_*`), plus mirrored `greenboost_status`/`greenboost_capabilities`/`greenboost_pilot`/`synapse_status`/`tiering_status` |
| `greenboost-orchestrator` | `gb_mcp.py` | 16 — `greenboost_overview`, `optimize_inference`, `quant_advisor`, `gb_plan` (CB-3 tier-plan), gated actuation (`tier_actuate`, `set_quant_policy`, `run_under_greenboost`, `a2a_gateway`), `shim_env`, plus mirrors of the status/capabilities/pilot/synapse_status/cluster_status/dataflux_summary tools (shared impl in `gb_mcp_common.py`) for "one server suffices" |
| `greenboost-cluster` | `gb_cluster_mcp.py` | 7 — live feeder/VRAM/T2/T3 state, gated `cluster_dispatch`/`cluster_ensure_feeder_ready` |
| `greenboost-synapse` | `gb_synapse_mcp.py` | 10 — serving control, `synapse_recommend`, gated `serve_and_repoint`, CLI bridge |
| `greenboost` (CLI) | `greenboost-cli/greenboost_cli/mcp/server.py` | 16 — RAG/goals/history/factory |

**A2A gateway** (`gb_a2a.py`, installed + enabled by Full Install as the
`greenboost-a2a.service` systemd unit — loopback-only, actuation gate off,
until an operator opts in): JSON-RPC 2.0 over HTTP, `GET
/.well-known/agent.json` (legacy AgentCard shape) AND `GET
/.well-known/agent-card.json` (A2A protocol v0.3 shape — interop with
ai-forge's `studio/server/a2a_gateway.py`, which serves the same path; see
`docs/a2a-interop.md`). Loopback-bound by default (`GB_A2A_TOKEN` required
for a LAN bind). Verbs map 1:1 to `gb_actuation.VERBS` (now including
`run_under_greenboost`) so MCP and A2A share the same double-gated
(`confirm=True` + `GB_ORCH_ACTUATE=1`) dispatch — A2A can never actuate
anything an MCP tool couldn't. Liveness + recent requests queryable via the
`a2a_status` tool on `greenboost-dataflux`; unit-level status/restart via
`a2a_gateway` on `greenboost-orchestrator`.

Keep this table current when a server's tool count changes — see
`checks/check_mcp_parity.py` (once landed) for the mechanical version of this
check.

---

## Build compatibility

GreenBoost builds without manual flags on all supported configurations.
The build system auto-detects the environment:

| Environment | Detection | Behaviour |
|---|---|---|
| CUDA 12 and 13 side-by-side | `ls /usr/local/cuda-[0-9]* | sort -V` | Picks the highest versioned install; warns if nvcc major ≠ driver CUDA version |
| Clang-built kernels (CachyOS, Arch/clang, Gentoo/clang) | `CONFIG_CC_IS_CLANG=y` in `$(KDIR)/.config` | Appends `LLVM=1` to all kernel sub-make targets (Makefile + DKMS) |
| GCC-built kernels | no `CONFIG_CC_IS_CLANG` | Standard GCC path, unchanged |
| No versioned CUDA dirs | fallback | `/usr/local/cuda` → `/usr/cuda` → `/opt/cuda` |

If you have both CUDA 12 and CUDA 13 installed and want to force a specific
version, set `CUDA_DIR=/usr/local/cuda-12 make` before building.

---

## How GreenBoost hooks in

GreenBoost works by intercepting `cudaMalloc` and related CUDA symbols via `LD_PRELOAD`, plus intercepting `dlsym` so that the virtual VRAM total (T1 VRAM + T2 DDR pool, computed from detected hardware) is returned to any app that queries VRAM size at runtime through `dlopen`+`dlsym`.

The shim is injected **per-process** , never system-wide via `/etc/ld.so.preload`.
Using `/etc/ld.so.preload` forces the CUDA interposer into every process on the
system including **systemd PID 1**, which freezes early boot. Injection happens
via two safe mechanisms instead:

* **systemd service drop-ins** , `Environment="LD_PRELOAD=…"` in
  `/etc/systemd/system/<service>.d/99-greenboost.conf`
* **Wrapper scripts** , `greenboost-run`, `greenboost-run-tgi`, etc. export
  `LD_PRELOAD` only for the target command.

`GREENBOOST_ACTIVE=1` still gates whether the shim does any work. In
**interactive login shells** (terminal, SSH), `/etc/profile.d/greenboost.sh`
exports `GREENBOOST_ACTIVE=1` automatically , no wrapper needed. Just run your
script directly:

```bash
python your_script.py               # shim already active in login shells
python -m vllm.entrypoints.openai.api_server ...
```

In **non-login contexts** (cron jobs, Docker entrypoints, sudo scripts, spawned subshells)
profile.d is not sourced. Use the wrapper or set the variable explicitly:

```bash
# Wrapper - sets GREENBOOST_ACTIVE=1 for one command
greenboost run python your_script.py

# Or inline
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so python your_script.py

# Or in a systemd unit / Docker environment
Environment="GREENBOOST_ACTIVE=1"
Environment="LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so"
```

---

## The five GreenBoost layers

GreenBoost is **not** a single library , it is five composable layers, each solving
one specific problem in the memory-limited GPU inference stack. A beginner AI engineer
can adopt them one at a time:

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5 · gb_diffusion_orch.py , diffusion pipeline orchestrator │
│  (multi-component management: VAE, CLIP, UNet/DiT)               │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4 · gb_llm.py , LLM inference quantization               │
│  (HuggingFace Transformers + vLLM, apply gb-quant at load time)  │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3 · gb_quant.py , weight quantize-to-fit                  │
│  (shrink weights to fit VRAM: bf16 → int8 → int4 → tq3 → tq2)   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2 · gb_init.py + gb_telemetry.py , bootstrap + telemetry  │
│  (single import wires all layers; ECC guard; GPU metrics)         │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1 · CUDA shim + kernel module , memory tier extension      │
│  (transparent cudaMalloc overflow to DDR/NVMe/cluster)            │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 1 , CUDA shim + kernel module

**What it solves:** Your model is 33 GB, your VRAM is 12 GB. `cudaMalloc` would
fail and the framework falls back to slow CPU math.

**How it works:** `libgreenboost_cuda.so` is loaded via `LD_PRELOAD` before your
Python process starts. It intercepts `cudaMalloc` and, when the allocation would
exceed VRAM, routes it to one of three hardware paths (in order of speed):

- **Path A , DMA-BUF pinned DDR** (requires `greenboost.ko`): the kernel module
  exports hugepages as a DMA-BUF, which CUDA imports as a device pointer via
  `cuImportExternalMemory`. The GPU reads DDR at PCIe speeds (~25 GB/s). The CPU
  executes zero tensor math.
- **Path B , `cuMemHostRegister`** (no kernel module): works inside Docker, VMs,
  WSL2. Slightly higher latency than Path A but portable.
- **Path C was removed.** UVM/managed memory caused GPU page-fault stalls.

The framework (Ollama, vLLM, PyTorch) sees a single GPU with inflated total
memory: `VRAM + T2_DDR_pool + T3_NVMe_pool`. It schedules the full model onto
that virtual device. No code changes required in your Python script.

> **Under the hood:** The shim uses `.symver` trampolines (`greenboost_cuda_v12.c`)
> to export both the `@@libcudart.so.12` default and `@libcudart.so.12` non-default
> symbols, so frameworks linked against CUDA 12 or CUDA 13 both resolve correctly.
> See [greenboost_documentation_extension_official_nvidia.md](greenboost_documentation_extension_official_nvidia.md)
> for the full technical explanation of where GreenBoost departs from NVIDIA's
> documented behaviour (inflated `cuDeviceTotalMem`, DMA-BUF interop details,
> data-driven feeder dispatch).

### Layer 2 , `gb_init.py` + `gb_telemetry.py` (bootstrap + telemetry)

**What it solves:** Starting a Python inference session requires wiring together
GPU metrics, an ECC error guard, and the memory-tier singletons. Without a
single entry point each project re-invents this boilerplate.

**`gb_init.py`** is a one-import bootstrap:

```python
import gb_init  # that's it

# What happens automatically:
# • torch.cuda.empty_cache → no-op (prevents conflict with the DynamicVRAM allocator)
# • TelemetryManager started (500 ms poll)
# • ECC DBE callback installed (stderr warning on uncorrectable memory error,
#   before inference silently corrupts outputs)
# • gb_stream_sched / gb_model_tier / gb_mem_pool singletons exposed
```

**`gb_telemetry.py`** uses a provider chain:

| Provider | Requires | What it reports |
|---|---|---|
| `NVMLProvider` | pynvml | power, temp, VRAM free/total, GPU utilisation |
| `DCGMProvider` | dcgm Python bindings | ECC single/double-bit errors (FI 310/311/313), NVLink BW (FI 449), health check (PCIe / SM / Memory / Thermal / …) |
| `GreenBoostProvider` | `/dev/greenboost` | T1/T2/T3 usage via ioctl |
| `TorchFallbackProvider` | torch.cuda | last-resort VRAM numbers |

The embedded DCGM mode (`dcgmStartEmbedded`) means no dcgmd daemon is needed on
bare-metal workstations or cluster nodes.

`gb_init.pre_inference_check()` blocks inference if an ECC double-bit error has
occurred since last boot , GPU memory is corrupted, output is wrong, better to
know before wasting 10 minutes of generation.

### Layer 3 , `gb_quant.py` (weight quantize-to-fit)

**What it solves:** Even with T2 DDR overflow, streaming large weights over PCIe
each token costs bandwidth. If the weights fit entirely in VRAM at a lower
precision, every token is fast.

**How it works:** `quantize_module` shrinks model weights using a quality-first
planner , each component gets the highest precision that still fits the VRAM
budget:

```
bf16  (16 bits) → highest quality, largest
int8  ( 8 bits)
int4  ( 4 bits) ← default sweet spot, HQQ backend
tq3   ( 3 bits) ← TurboQuant: Triton LUT-GEMM, rotated activations
tq2   ( 2 bits) ← TurboQuant: most compact, useful for huge models
```

The low-bit GEMM backend (GemLite + HQQ) ships inside GreenBoost
(`third_party/`, Apache-2.0). Your venv needs nothing extra.

```python
from gb_quant import quantize_module

model = load_your_model()
model = quantize_module(model)   # planner picks the right precision for each layer
```

`GB_QUANT_BUDGET_GB` overrides the VRAM budget. A warm-kernel autotune cache
(`~/.cache/greenboost/`) means Triton compilation only happens once.

### Layer 4 , `gb_llm.py` (LLM inference)

**What it solves:** Wiring gb-quant into HuggingFace Transformers or vLLM with
the right hooks and profiler annotations.

```python
from gb_llm import load_model_gb

# Loads a HF model, auto-selects quantization, pins KV cache to the right tier
model, tokenizer = load_model_gb("mistralai/Mistral-7B-v0.3")
```

NVTX range `gb:llm_load:<model>` is emitted around the load phase so Nsight
Systems traces show exactly how long quantization + tier placement took.

### Layer 5 , `gb_diffusion_orch.py` (diffusion pipeline orchestrator)

**What it solves:** Diffusion models (Stable Diffusion, FLUX, LTX-Video) have
three or four sub-models that each allocate GPU memory: a VAE, a text encoder
(CLIP / T5), and the main denoiser (UNet or DiT). Managing them as separate
`quantize_module` calls leaves memory fragmentation gaps and misses the chance
to tune per-component budgets.

`gb_diffusion_orch.py` treats the whole pipeline as one budget:

```python
from gb_diffusion_orch import DiffusionOrchestrator

orch = DiffusionOrchestrator(pipeline)
orch.fit()      # quantizes each component to the right precision
orch.run(prompt="a cat", steps=20)
```

Uses the `gb_init` singletons for telemetry and only stops the telemetry thread
if it was the one that created it (`_tel_owned` flag), so stacking multiple
orchestrators in one session works correctly.

---

## The official CUDA extension document

[`greenboost_documentation_extension_official_nvidia.md`](greenboost_documentation_extension_official_nvidia.md)
is GreenBoost's technical companion to the CUDA Programming Guide.
It is written in the same register as NVIDIA's Programming Guide chapters 3
and 4, and explicitly calls out every place where GreenBoost combines
documented NVIDIA primitives in ways NVIDIA never described:

- Inflating `cuDeviceTotalMem` / `nvmlDeviceGetMemoryInfo` to include DDR + NVMe pools
- `cuImportExternalMemory(OpaqueFd)` + DMA-BUF zero-copy pinned DDR (Path A)
- `cuMemHostRegister(DEVICEMAP)` over `GB_IOCTL_PIN_USER_PTR` (Path A pinned sub-method)
- Virtual device aggregation across TCP-connected feeder machines
- `cuLaunchKernel` data-driven remote dispatch via fake pointer scanning
- CUDA 13 invalid-context runtime fallback in `cuMemGetInfo_v2` / `cuDeviceTotalMem_v2`

Read it if you are integrating GreenBoost into a new framework and need to
understand precisely what the shim reports vs. what the NVIDIA driver would report.

---

## Ollama

Handled automatically by `install-sys-configs`, which injects the shim and
`GREENBOOST_ACTIVE=1` into the systemd unit. No manual wrapper needed.

```bash
ollama run glm-4.7-flash:q8_0   # GreenBoost is transparent
```

If running Ollama outside systemd:

```bash
greenboost run ollama serve
```

---

## vLLM

vLLM loads libcuda lazily through PyTorch (`torch.cuda` → `ctypes.CDLL`).

```bash
greenboost run python -m vllm.entrypoints.openai.api_server \
    --model /opt/models/glm-4.7-flash-hf \
    --dtype float16 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95
```

As a systemd service, add to the unit:

```ini
[Service]
Environment="LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so"
Environment="GREENBOOST_ACTIVE=1"
```

---

## PyTorch scripts

Any `import torch; torch.cuda.*` call triggers lazy loading of libcuda via ctypes.

```bash
greenboost run python your_inference_script.py
```

Or inline without the wrapper:

```bash
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    python your_script.py
```

---

## text-generation-inference (TGI)

TGI uses a PyTorch backend - same lazy-CUDA pattern:

```bash
greenboost run text-generation-launcher \
    --model-id /opt/models/glm-4.7-flash-hf \
    --num-shard 1 \
    --max-total-tokens 131072
```

As a systemd service:

```ini
[Service]
Environment="LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so"
Environment="GREENBOOST_ACTIVE=1"
```

---

## CTranslate2

CTranslate2 loads libcudart via ctypes in its Python bindings:

```bash
greenboost run python your_ctranslate2_script.py
```

---

## Hugging Face Transformers

Transformers loads CUDA through PyTorch - same lazy-CUDA pattern as vLLM and TGI.

```bash
greenboost run python your_transformers_script.py
```

With `device_map="auto"`, Transformers queries available VRAM before placing layers.
GreenBoost reports the detected T1+T2 total, so the full model is placed on the "GPU"
(T1+T2) instead of being split to CPU:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "/opt/models/glm-4.7-flash-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",          # sees T1+T2 total - loads entire model onto T1+T2
)
tokenizer = AutoTokenizer.from_pretrained("/opt/models/glm-4.7-flash-hf")

messages = [{"role": "user", "content": "Hello!"}]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
).to(model.device)

output = model.generate(input_ids, max_new_tokens=300)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Pipeline API works the same way:

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="/opt/models/glm-4.7-flash-hf",
    torch_dtype="bfloat16",
    device_map="auto",
)
print(pipe("Hello!", max_new_tokens=200)[0]["generated_text"])
```

Downloading a model then running it in one shot:

```bash
greenboost run python - <<'EOF'
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

snapshot_download("THUDM/glm-4.7-flash-hf", local_dir="/opt/models/glm-4.7-flash-hf")

model = AutoModelForCausalLM.from_pretrained(
    "/opt/models/glm-4.7-flash-hf", torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/opt/models/glm-4.7-flash-hf")
ids = tokenizer("Hello!", return_tensors="pt").input_ids.to(model.device)
print(tokenizer.decode(model.generate(ids, max_new_tokens=100)[0]))
EOF
```

When using the GreenBoost venv:

```bash
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    /opt/greenboost/venv/bin/python your_transformers_script.py
```

---

## ExLlamaV3

ExLlamaV3 loads CUDA through PyTorch:

```bash
greenboost run python your_exllama_script.py
```

When using the GreenBoost venv:

```bash
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    /opt/greenboost/venv/bin/python your_exllama_script.py
```

The GreenBoost-patched KV cache layer (`CacheLayer_greenboost`) allocates directly
from `/dev/greenboost` and does not go through the shim's cudaMalloc path for cache
tensors - `GREENBOOST_ACTIVE=1` is still needed for virtual VRAM reporting.

---

## PyTorch - direct tensor allocation control

PyTorch loads libcuda lazily via `ctypes.CDLL` when `torch.cuda` is first accessed.
GreenBoost intercepts from that point forward.

```bash
greenboost run python your_script.py
# or inline:
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so python your_script.py
```

### Explicit KV cache flagging from Python

For custom inference loops where you know exactly which tensors are KV cache, use
`GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY` via the MADVISE IOCTL so the kernel
freezes them in T2's LRU and refuses T3 spill:

```python
import fcntl, struct, os
from pathlib import Path

# GB_IOCTL_MADVISE = _IOW('G', 4, struct gb_madvise_req)  - 8 bytes
_IOC_MADVISE     = (1 << 30) | (ord('G') << 8) | 4 | (8 << 16)
GB_MADVISE_T1_PREFER = 3   # mark as T1-priority: freeze in T2 LRU, refuse T3 spill

def gb_mark_kv(buf_fd: int) -> None:
    """
    Tell GreenBoost that buf_fd (DMA-BUF fd from GB_IOCTL_ALLOC) is a KV cache buffer.
    Kernel will: freeze in T2 LRU, refuse T3 spill, set t1_priority.
    buf_fd is the fd returned by GB_IOCTL_ALLOC (stored in gb_alloc_req.fd).
    """
    dev = Path("/dev/greenboost")
    if not dev.exists():
        return
    # struct gb_madvise_req { s32 buf_id; u32 advise; }
    buf = struct.pack("iI", buf_fd, GB_MADVISE_T1_PREFER)
    fd  = os.open(str(dev), os.O_RDWR)
    try:
        fcntl.ioctl(fd, _IOC_MADVISE, buf)
    finally:
        os.close(fd)

# Usage with PyTorch (via ExLlamaV3 native IOCTL path)
# buf_fd is obtained from GB_IOCTL_ALLOC, not from cudaMalloc.
# For pure-PyTorch workflows, use GREENBOOST_KV_OVERFLOW=1 or the
# phase detector (automatic) instead of calling this directly.
```

### Optimal loading order

GreenBoost's phase detector expects: **weights first, KV cache second**.
Respect this order to get maximum T1 utilization:

```python
import torch

# 1. Load model weights (phase = MODEL_LOAD - reserve inactive, weights fill T1)
model = MyModel().cuda()
model.load_state_dict(torch.load("model.pt", map_location="cuda"))

# 2. Allocate KV cache (phase = INFERENCE - reserve activates, KV lands in T1)
kv_cache = torch.zeros(batch, n_layers, seq_len, d_model, device="cuda")
# ↑ GreenBoost auto-classifies this as KV via quiet-gap heuristic
#   OR call gb_mark_kv() above for certainty

# 3. Run inference
output = model.generate(input_ids, past_key_values=kv_cache)
```

If your script loads KV cache before or interleaved with weights, set
`GREENBOOST_KV_OVERFLOW=1` to disable the heuristic.

---

## Hugging Face Transformers - direct KV cache control

Transformers uses PyTorch's CUDA allocator; GreenBoost intercepts all `cudaMalloc`
calls. `device_map="auto"` queries VRAM size - GreenBoost reports the detected T1+T2
total, so the full model is placed on T1+T2 instead of being split to CPU.

```bash
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    python inference.py
```

### Pinning the KV cache to T1

Transformers' `DynamicCache` allocates keys/values lazily during generation.
To ensure they land in T1 rather than T2, set a generous KV reserve **before**
generation starts and tell GreenBoost the phase is INFERENCE:

```python
import os
os.environ["GREENBOOST_KV_RESERVE_MB"] = "4096"   # 4 GB reserved for KV
os.environ["GREENBOOST_KV_OVERFLOW"]   = "0"       # use phase detector (default)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "/opt/models/glm-4.7-flash-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",          # sees T1+T2 total - all layers on GPU
    attn_implementation="flash_attention_2",   # requires OLLAMA_FLASH_ATTENTION=1 equivalent
)
tokenizer = AutoTokenizer.from_pretrained("/opt/models/glm-4.7-flash-hf")

# All tokens below will have KV cache allocated in T1 (phase detector: INFERENCE)
out = model.generate(
    tokenizer("Hello!", return_tensors="pt").input_ids.cuda(),
    max_new_tokens=512,
    use_cache=True,   # ensure KV cache is used
)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### Mark Transformers' KV cache explicitly (advanced)

If the phase detector misclassifies activations as KV, disable it and mark manually:

```python
import os
os.environ["GREENBOOST_PHASE_DETECT"] = "0"    # disable auto-classification

from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(..., device_map="auto")

# Monkey-patch _update_causal_mask to mark KV after each alloc
_orig_forward = model.forward

def _patched_forward(*args, **kwargs):
    out = _orig_forward(*args, **kwargs)
    # Walk past_key_values and mark each tensor as KV
    if out.past_key_values:
        for layer_kv in out.past_key_values:
            for tensor in layer_kv:
                if tensor.is_cuda:
                    gb_mark_kv(tensor.data_ptr(), tensor.nbytes)
    return out

model.forward = _patched_forward
```

### Context-window KV reserve sizing

| OLLAMA_NUM_CTX / max_new_tokens | Recommended GREENBOOST_KV_RESERVE_MB |
|----------------------------------|---------------------------------------|
| ≤ 8 K tokens                    | 1024 MB                               |
| 8 K – 32 K tokens               | 2048 MB (default)                     |
| 32 K – 64 K tokens              | 4096 MB                               |
| 64 K – 128 K tokens             | 6144 MB                               |
| > 128 K tokens                  | 8192 MB                               |

---

## vLLM - direct KV cache placement

vLLM manages its own block-based KV cache allocator (PagedAttention). It pre-allocates
the entire KV cache up front from GPU memory before inference begins.

```bash
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    python -m vllm.entrypoints.openai.api_server \
        --model /opt/models/glm-4.7-flash-hf \
        --max-model-len 131072 \
        --gpu-memory-utilization 0.95 \
        --enforce-eager
```

### Why `--gpu-memory-utilization 0.95` with GreenBoost

vLLM queries `torch.cuda.mem_get_info()` to calculate how many KV blocks it can
allocate. GreenBoost intercepts this and returns T1 free space. At 0.95 utilization,
vLLM will try to allocate ~11.4 GB of KV cache from "GPU" memory - GreenBoost
routes any T1 overflow to T2 via DMA-BUF.

**Increase KV reserve before launching vLLM** so that vLLM's pre-allocated KV cache
gets priority over leftover weight fragments in T1:

```bash
# Set 8 GB reserve before starting vLLM (131K context needs ~7–9 GB KV)
GREENBOOST_KV_RESERVE_MB=8192 \
GREENBOOST_KV_OVERFLOW=1 \
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    python -m vllm.entrypoints.openai.api_server \
        --model /opt/models/glm-4.7-flash-hf \
        --max-model-len 131072
```

`GREENBOOST_KV_OVERFLOW=1` is recommended for vLLM because PagedAttention allocs
KV blocks in a dedicated phase that doesn't follow the weight-then-KV timing
the phase detector expects.

### vLLM systemd service

```ini
[Service]
Environment="GREENBOOST_ACTIVE=1"
Environment="GREENBOOST_KV_OVERFLOW=1"
Environment="GREENBOOST_KV_RESERVE_MB=8192"
Environment="LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so"
ExecStart=python -m vllm.entrypoints.openai.api_server --model ...
```

---

## Ollama 0.18+ and the CUDA VMM path

Ollama 0.18+ switched its ggml CUDA backend from `cudaMalloc` to the CUDA Virtual Memory
Management API (`cuMemCreate` → `cuMemMap` → `cuMemSetAccess`). GreenBoost intercepts
`cuMemCreate` transparently - no configuration change is needed. When a `cuMemCreate`
call fails due to T1 being full, the shim retries with `CU_MEM_LOCATION_TYPE_HOST`, which
creates a pinned host-memory (system DDR) allocation the GPU accesses over PCIe - the
same effective result as Path B, but initiated from within the VMM code path.

---

## TensorFlow / Keras

TensorFlow loads `libcuda.so` via `ctypes` when `tf.config.list_physical_devices("GPU")`
is called. GreenBoost intercepts from that moment.

```bash
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    python your_tf_script.py
```

### Memory growth vs pre-allocation

TensorFlow defaults to pre-allocating all GPU memory on startup. GreenBoost reports
the detected T1+T2 total, which TF would try to allocate entirely - set memory growth instead:

```python
import os
os.environ["GREENBOOST_ACTIVE"] = "1"
# LD_PRELOAD must be set before Python starts, not here

import tensorflow as tf

# Prevent TF from trying to allocate the entire T1+T2 virtual pool at once
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# From this point forward cudaMalloc calls are GreenBoost-managed:
# weights → T1, KV-like buffers → T1-priority (if phase heuristic fires)
model = tf.keras.models.load_model("/opt/models/my_model.keras")
```

### Manual KV cache marking for TF custom training

```python
import ctypes, struct, os

def gb_mark_tensor_kv(tf_tensor) -> None:
    """Mark a TensorFlow GPU tensor as KV cache in GreenBoost."""
    # Get raw device pointer via experimental C API
    from tensorflow.python.framework import ops
    ptr = ctypes.cast(
        tf_tensor._handle(),  # internal: raw device pointer
        ctypes.c_void_p
    ).value
    if ptr:
        gb_mark_kv(ptr, tf_tensor.nbytes)
```

### TF with XLA and GreenBoost

XLA compilations may pre-allocate scratch buffers that confuse the phase detector.
Disable the heuristic for XLA workloads:

```bash
GREENBOOST_PHASE_DETECT=0 GREENBOOST_KV_OVERFLOW=0 \
GREENBOOST_ACTIVE=1 LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
    python xla_training.py
```

---

## KV Cache Fine-Tuning - Bypassing the Phase Heuristic

GreenBoost auto-classifies CUDA allocations via a temporal phase detector
(`INIT → MODEL_LOAD → INFERENCE → STEADY`). This works transparently for Ollama and
llama.cpp, but some engines or custom workloads need explicit control.

### When the heuristic needs help

| Scenario | Symptom | Solution |
|----------|---------|----------|
| ExLlamaV3 (native IOCTL path) | KV allocs don't follow weight-then-KV timing | `GREENBOOST_KV_OVERFLOW=1` |
| Batched inference servers | Multiple parallel KV allocs confuse phase transitions | `GREENBOOST_KV_OVERFLOW=1` |
| Very small models (<8K ctx) | 2 GB default reserve wastes T1 for tiny KV | `GREENBOOST_KV_RESERVE_MB=512` |
| Long-context (128K ctx) | Default 2 GB reserve undersized; KV spills to T2 | `GREENBOOST_KV_RESERVE_MB=6144` |
| Custom training loops | Activation buffers wrongly classified as KV | `GREENBOOST_PHASE_DETECT=0` |
| Verify KV location | Want to confirm KV is in T1, not T2 | Check `kv_t2_mb` in sysfs |

### Environment variable reference

All variables are read by the shim at startup and refreshed every 64 allocations from
the kernel module. They can be set in the Ollama service drop-in, in shell exports, or
prepended to any command.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GREENBOOST_KV_RESERVE_MB` | integer | from kernel (2048) | MB of T1 VRAM reserved exclusively for KV cache. Weights overflow to T2 sooner; KV lands in T1 at 336 GB/s instead of T2 at 32 GB/s. Auto-scaled from `OLLAMA_NUM_CTX` if unset: 8K→1 GB · 32K→2 GB · 64K→4 GB · 128K→6 GB · >128K→8 GB. Capped at T1−2 GB. |
| `GREENBOOST_KV_OVERFLOW` | `0`/`1` | `0` | **Master bypass**: when `1`, every overflow allocation is flagged `GB_ALLOC_KV_CACHE \| GB_ALLOC_T1_PRIORITY` - the phase detector is skipped entirely. Use for ExLlamaV3 or any engine whose overflow allocs are predominantly KV. |
| `GREENBOOST_PHASE_DETECT` | `0`/`1` | `1` | Disable the temporal phase classifier. All overflow allocs use generic flags. Combine with `GREENBOOST_KV_OVERFLOW=1` for full manual control. |
| `GREENBOOST_KV_SIZE_THRESHOLD_MB` | integer | `64` | Minimum alloc size (MB) to classify as KV during the INFERENCE phase. Raise to `256` if activation buffers are being wrongly pinned as KV. |
| `GREENBOOST_VRAM_HEADROOM_MB` | integer | `512` | MB kept free in T1 as a safety buffer. Reduce to `256` to squeeze more weights into T1. |
| `GREENBOOST_VIRTUAL_VRAM_MB` | integer | computed | Override the virtual VRAM total reported to CUDA apps (default: T1 + T2 in MB). |
| `GREENBOOST_ACTIVE` | `0`/`1` | `1` if shim loaded | Must be `1` for the shim to intercept allocations. Set to `0` to disable without unloading. |
| `GREENBOOST_DEBUG` | `0`/`1` | `0` | Enable verbose shim logging to stderr. Each alloc decision is logged: phase, flags, tier placement. |

### Quick recipes

```bash
# Ollama with a 128K context - increase KV reserve to 6 GB
GREENBOOST_KV_RESERVE_MB=6144 ollama run nemotron-3-super:120b

# ExLlamaV3 - bypass phase detector, all overflow is KV
GREENBOOST_KV_OVERFLOW=1 GREENBOOST_ACTIVE=1 \
  LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
  python your_exllama_script.py

# Debug KV placement (stderr shows each allocation decision)
GREENBOOST_DEBUG=1 ollama run glm-4.7-flash:q8_0 2>&1 | grep -E "KV|Phase|VRAM"

# Disable KV heuristic entirely for a custom training script
GREENBOOST_PHASE_DETECT=0 greenboost run python train.py

# Small model, shrink reserve to reclaim 1.5 GB of T1
GREENBOOST_KV_RESERVE_MB=512 ollama run llama3.2:3b
```

### Runtime KV reserve update via IOCTL

The KV reserve can be changed at runtime without restarting Ollama. The shim polls
`GB_IOCTL_GET_INFO` every 64 allocations and picks up the new value:

```bash
# Via Synapse CLI slash command (recommended)
/kv-reserve 4096

# Via greenboost_setup.sh (reads current then patches)
sudo ./greenboost_setup.sh tune-kv-reserve 4096

# Via Python (see synapse/greenboost.py)
from synapse.greenboost import set_kv_reserve
set_kv_reserve(4096)   # returns True on success
```

### `GB_ALLOC_*` flags - programmatic allocation control

These flags are used by ExLlamaV3's `CacheLayer_greenboost` when calling
`GB_IOCTL_ALLOC` directly on `/dev/greenboost`. They bypass the shim entirely and
give the inference engine precise control over tier placement.

```c
/* from greenboost_ioctl.h */
#define GB_ALLOC_WEIGHTS       (1u << 0)  /* model weight tensor             */
#define GB_ALLOC_KV_CACHE      (1u << 1)  /* KV cache - never spills to T3;
                                           * auto-frozen in T2 LRU           */
#define GB_ALLOC_ACTIVATIONS   (1u << 2)  /* ephemeral activation buffer     */
#define GB_ALLOC_FROZEN        (1u << 3)  /* never evict from T2             */
#define GB_ALLOC_NO_HUGEPAGE   (1u << 4)  /* force 4K pages (T3-spillable)   */
#define GB_ALLOC_T1_PRIORITY   (1u << 5)  /* KV-like; moved to LRU head,
                                           * weight bufs evicted first        */
#define GB_ALLOC_KV_COMPRESSED (1u << 6)  /* TurboQuant-compressed KV;
                                           * 3-7× more KV history in T2/T3  */
```

**Flag semantics:**

- `GB_ALLOC_KV_CACHE` - marks the buffer as KV cache. The kernel auto-freezes it in
  the T2 LRU (so it is never evicted to T3) and refuses T3 spill with `ENOSPC`. KV
  bandwidth at T3 (~1.8 GB/s) would collapse generation to unusable speeds.

- `GB_ALLOC_T1_PRIORITY` - combined with `GB_ALLOC_KV_CACHE` it sets `t1_priority=1`
  on the buffer, making weight buffers preferred for eviction over this one. Always
  use together: `GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY`.

- `GB_ALLOC_KV_COMPRESSED` - marks the buffer as TurboQuant-compressed KV. The kernel
  tracks compression savings separately (`kv_compressed_mb`, `kv_compression_bits`
  in pool_info). Set via `GB_IOCTL_SET_TURBOQUANT` first.

- `GB_ALLOC_FROZEN` - prevents eviction from T2 regardless of type. Use for buffers
  that must never be paged out (e.g. LoRA adapter weights during inference).

**Usage pattern (ExLlamaV3 native):**

```c
struct gb_alloc_req req = {
    .size_bytes = kv_size,
    .flags      = GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY,
    .tier_hint  = 1,   /* prefer T1 */
};
if (ioctl(gb_fd, GB_IOCTL_ALLOC, &req) == 0)
    kv_handle = req.handle;
```

### Diagnosing KV spill

If KV cache spills from T1 to T2, the shim logs a warning to stderr and
`kv_t2_mb` in sysfs becomes non-zero:

```bash
# Live monitoring
watch -n1 'grep kv /sys/class/greenboost/greenboost/status'

# One-liner: show KV location
python3 -c "
data = {k.strip(): int(v.strip())
        for line in open('/sys/class/greenboost/greenboost/status')
        if '=' in line for k, v in [line.split('=', 1)]}
kv_t2 = data.get('kv_t2_mb', 0)
kv_used = data.get('kv_used_mb', 0)
print(f'KV used: {kv_used} MB  |  KV in T2: {kv_t2} MB  |  KV in T1: {kv_used - kv_t2} MB')
print('✓ KV fully in T1' if kv_t2 == 0 else f'⚠ KV spilling to T2 - increase GREENBOOST_KV_RESERVE_MB')
"
```

---

## Session Priority Management

For multi-model deployments where several inference processes share T2 DDR, GreenBoost
provides two IOCTLs to control which process's buffers get evicted first when T2 runs low:

| IOCTL | Cmd | Effect |
|-------|-----|--------|
| `GB_IOCTL_SESSION_IDLE` | 18 | Move all caller-PID T2 buffers to LRU tail - first to be evicted under pressure |
| `GB_IOCTL_SESSION_ACTIVE` | 19 | Move all caller-PID T2 buffers to LRU head - last to be evicted |

Call `SESSION_IDLE` when a model goes idle (no pending requests) and `SESSION_ACTIVE` when
a new request arrives. This lets the active model keep its weights in T2 while the idle
model's weights become eviction candidates.

```python
import fcntl, struct, os

# GB_IOCTL_SESSION_IDLE   = _IOW('G', 18, struct gb_session_req)  - 8 bytes
# GB_IOCTL_SESSION_ACTIVE = _IOW('G', 19, struct gb_session_req)  - 8 bytes
_IOC_SESSION_IDLE   = (1 << 30) | (ord('G') << 8) | 18 | (8 << 16)
_IOC_SESSION_ACTIVE = (1 << 30) | (ord('G') << 8) | 19 | (8 << 16)

def gb_session_idle() -> None:
    """Mark this process's T2 buffers as low priority (idle model)."""
    _gb_session_ioctl(_IOC_SESSION_IDLE)

def gb_session_active() -> None:
    """Mark this process's T2 buffers as high priority (active model)."""
    _gb_session_ioctl(_IOC_SESSION_ACTIVE)

def _gb_session_ioctl(cmd: int) -> None:
    dev = "/dev/greenboost"
    if not os.path.exists(dev):
        return
    # struct gb_session_req { u32 pid; u32 reserved; }  - pid=0 means caller's PID
    buf = struct.pack("II", 0, 0)
    fd  = os.open(dev, os.O_RDWR)
    try:
        fcntl.ioctl(fd, cmd, buf)
    finally:
        os.close(fd)
```

---

## Verify GreenBoost is active

```bash
# Should show T2 pool in use (non-zero) after loading a model larger than 12 GB:
cat /sys/class/greenboost/greenboost/status

# Confirm virtual VRAM is visible (should report T1+T2 total, not just physical VRAM):
# From an interactive terminal - GREENBOOST_ACTIVE=1 is already set by profile.d
python -c "
import torch
print('VRAM reported:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"

# From a non-login shell (cron, Docker, sudo), use the wrapper:
# greenboost run python -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"

# Check shim is loaded into a running process (e.g. Ollama):
grep greenboost /proc/$(pgrep ollama)/maps | head -3
```

---

## Capabilities, pilot & health-check

Three read-only commands for "what's actually running right now, and what
should I do about it":

```bash
greenboost capabilities   # installed/running shim feature manifest
greenboost pilot           # dataflux-driven advice on the next tuning step
greenboost health-check    # one-shot PASS/FAIL/WARN across module/VRAM/T2/T3/cluster
```

- **`greenboost capabilities`** is backed by `gb_monitor.py`, the canonical
  telemetry/capability client that replaced four independently-grown
  `shim_stats` parsers. It reads the shim's feature manifest, written at
  install time and refreshed at runtime to `/run/greenboost/capabilities.json`.
- **`greenboost pilot`** is backed by `gb_pilot.py`. It reads the dataflux
  log (§ dataflux below) and turns recorded trends into concrete,
  evidence-backed suggestions - each one names the exact `gb_control` lever
  to pull. It never actuates anything itself; you decide whether to act.
- Both are also exposed as MCP tools (`greenboost_capabilities`,
  `greenboost_pilot`) by `gb_dataflux_mcp.py`, alongside `greenboost_status`,
  so an MCP-connected AI assistant can query the same state directly.

---

## Live debug signals - real-time telemetry during inference

GreenBoost exposes three layers of live telemetry readable at any time, even mid-inference.

### Layer 1: sysfs kernel interface

The kernel module exports a sysfs tree at `/sys/class/greenboost/greenboost/`. These files are always readable while the module is loaded, regardless of whether any app is running.

```bash
# One-liner pool summary - ideal for continuous monitoring
watch -n2 'cat /sys/class/greenboost/greenboost/pool_brief'
# → T1:11GB T2:23/42GB(56%) T3:0/73GB PRESSURE:ok KV_RSV:0MB KV_T2:0MB

# Full tier status with KV cache breakdown and T3 pressure
watch -n2 'cat /sys/class/greenboost/greenboost/status'

# Hardware topology (CPU/GPU/RAM/NVMe, NUMA, kthread affinity) - static
cat /sys/class/greenboost/greenboost/hw_info

# Count of live DMA-BUF tensor handles
watch -n1 'cat /sys/class/greenboost/greenboost/active_buffers'

# KV cache reserve in MB (0 = no reserve, all T2 free for model weights)
cat /sys/class/greenboost/greenboost/kv_reserve_mb
echo 0 | sudo tee /sys/module/greenboost/parameters/kv_reserve_mb   # set live
```

### Layer 2: `/run/greenboost/` runtime files (LD_PRELOAD shim)

These files are written by any process running with `LD_PRELOAD=libgreenboost_cuda.so`. They reflect the *most recent* process that ran with the shim active.

> **Ownership note:** If a previous process (e.g. Ollama running as `ollama`) wrote these files, they will be owned by that user and a new process cannot overwrite them. Fix: `rm -f /run/greenboost/shim_stats /run/greenboost/metrics.json` before starting your process. The `/run/greenboost/` directory is world-writable so any user can unlink these files.

| File | Format | Content |
|------|--------|---------|
| `shim_stats` | `key=value` | Full shim counters - see field table below |
| `metrics.json` | JSON | Same data in JSON with a `feeders[]` array per connected feeder |
| `nvtx_events.log` | TSV | Chronological event trace: allocations, phase transitions, OOM events |
| `phase` | plain text | Current phase: `MODEL_LOAD` / `INFERENCE` / `STEADY` / `OOM_RECOVERY` |

**Key `shim_stats` fields and what they mean:**

```bash
# Watch the most diagnostic fields live during inference
watch -n2 'grep -E "^phase=|^tier_t1_feeder_cur|^tier_t2_feeder_cur|^kernel_dispatch|^remote_alloc_count|^t2_pool_frag|^t2_above_warn|^vram_headroom|^path_[abc]_count" /run/greenboost/shim_stats'
```

| Field | Healthy value | Problem if... |
|-------|--------------|---------------|
| `phase` | `STEADY` or `INFERENCE` | `OOM_RECOVERY` = memory crisis |
| `vram_headroom_mb` | > 200 | < 100 = T1 pressure |
| `tier_t1_feeder_cur_mb` | > 0 when feeder pooled | 0 = feeder not contributing memory |
| `kernel_dispatch_count` | > 0 when feeder pooled | 0 = feeder compute idle despite pooling |
| `remote_alloc_count` | > 0 when feeder pooled | 0 = all allocations staying local |
| `path_a_count` | increases during inference | - (normal T2 DDR path) |
| `path_c_count` | 0 | > 0 = T3 NVMe spill → 10× slowdown |
| `t2_pool_frag_pct` | < 30 | > 80 = fragmentation pressure |
| `t2_above_warn` | 0 | 1 = T2 pool > 75% full |
| `kv_dedup_hits` | increases over time | 0 = KV dedup inactive |

**Inspect feeder connectivity from `metrics.json`:**

```bash
python3 -c "
import json, sys
d = json.load(open('/run/greenboost/metrics.json'))
print(f'Phase:             {d[\"phase\"]}')
print(f'Kernel dispatches: {d[\"kernel_dispatch_count\"]}')
print(f'T1 local cur:      {d[\"tiers\"][\"t1_local\"][\"cur_mb\"]} MB')
print(f'T1 feeder cur:     {d[\"tiers\"][\"t1_feeder\"][\"cur_mb\"]} MB')
print(f'T2 feeder cur:     {d[\"tiers\"][\"t2_feeder\"][\"cur_mb\"]} MB')
for f in d.get('feeders', []):
    print(f'Feeder: {f[\"feeder\"]}  health={f[\"health_state\"]}  bw={f[\"bw_measured_mbs\"]} MiB/s  dispatched={d[\"kernel_dispatch_count\"]}')
"
```

### Layer 3: NVTX event log

The NVTX log is a chronological trace of every significant event the shim handles. It persists across process restarts (appended, not overwritten).

```bash
# Live tail - see allocations and phase transitions as they happen
tail -f /run/greenboost/nvtx_events.log

# Same via greenboost CLI (interactive pager, auto-scrolling)
greenboost nvtx-logs

# Clear log for a fresh diagnostic session
sudo greenboost clear nvtx-logs
```

**Log format:** `<epoch_ns>  <EVENT_TYPE>  <TIER>  <MB>  ptr=<hex>  <message>`

Key event types to look for:

| Event type | Meaning |
|------------|---------|
| `SHIM_INIT` | Process attached to GreenBoost shim |
| `PHASE_MODEL_LOAD` | Model weights loading phase started |
| `PHASE_INFERENCE` | Model loaded, inference phase started |
| `ALLOC_T1_VRAM` | Allocation placed in GPU VRAM (fast path) |
| `ALLOC_T2_POOL` | Allocation placed in T2 DDR pool (path A) |
| `ALLOC_T3_UVM` | Allocation spilled to T3 NVMe (path C) - investigate if frequent |
| `FEEDER_ALLOC` | Allocation placed on a remote feeder GPU |
| `KERNEL_DISPATCH` | CUDA kernel dispatched to feeder for remote execution |
| `OOM_EVICT` | OOM pressure - tensor evicted from T1 to T2/T3 |
| `FREE_T2_POOL` | T2 tensor freed (deallocation) |

### Layer 4: Web dashboard

```bash
# Start greenboost-cli dashboard server (served at http://localhost:7821)
greenboost-cli
# or: python -m greenboost_cli

# Navigate to: http://localhost:7821/greenboost
```

The `/greenboost` page provides:
- **SHIM STATISTICS** - auto-polls `shim_stats` every 2 s: path A/B/C counters, per-tier allocations (local + feeder peaks), kernel dispatch count, T2 fragmentation, T1 headroom
- **FEEDER VITALS** - per-feeder T1/T2/T3 current and peak MB, remote alloc count, connection status badge
- **NVTX EVENT LOG** - scrollable live log of the last N NVTX events with timestamps

### dataflux , the flight recorder (port 8799)

This is a different, complementary page from the Layer-4 dashboard above.
Where the `:7821` dashboard shows live shim counters right now, dataflux
(`gb_dataflux.py`) keeps a continuous local log of the whole system over time
, VRAM used/free, GPU/CPU load, temperature/power, KV-cache pressure,
memory-tier movement, quantization decisions, and (once a feeder is
connected) per-machine cluster throughput , plus real, client-measured
tokens-per-second, not just what GreenBoost predicted.

```bash
greenboost dataflux-ui                            # opens the web UI, port 8799
python3 gb_dataflux.py serve [--port 8799] [--days 5]

# headless snapshot instead of the browser UI
greenboost dataflux-ui --llm
python3 gb_dataflux.py summary
```

Works standalone on a single machine, no feeder required. The web page
auto-refreshes every 5 seconds and breaks down activity by script/label, one
row per script run, and rolling VRAM/GPU-util/CPU-util/KV-pressure sparklines
plus a raw snapshot table:

![GreenBoost dataflux web UI](greenboost_dataflux_ui.png)

The same log is also exposed read-only over MCP
(`gb_dataflux_mcp.py`, registered as `greenboost-dataflux`), so any
MCP-compatible client can query cluster and inference history directly
instead of opening the web UI. Seven tools, each answering a specific
question:

| Tool | Answers |
|---|---|
| `dataflux_summary` | Cheap overview , totals, per-node/label/run rollups, tok/s |
| `dataflux_events` | Raw events, filterable by node, label, **kind**, and status |
| `dataflux_errors` | What broke recently, and where |
| `dataflux_kinds` | What kinds of activity actually happened (tier moves, quantization, tok/s, cluster chunks, ...) before drilling in |
| `dataflux_tier_moves` | T1/T2/T3 promote/demote/evict events from `gb_model_tier.py` |
| `dataflux_quantization` | Quantization decisions from `gb_quant.py` , component, bits, budget vs. actual GiB |
| `dataflux_tok_s` | Measured, real tokens/sec per model, with the full time series when you ask for one model by name |

### Diagnosing "feeder shows zero activity" (pooled mode)

If `tier_t1_feeder_cur_mb=0` and `kernel_dispatch_count=0` despite `greenboost cluster` showing the feeder connected:

```bash
# 1. Check shim_stats is writable by the current user
ls -la /run/greenboost/shim_stats
# If owned by another user → delete and re-run:
rm -f /run/greenboost/shim_stats /run/greenboost/metrics.json

# 2. Check feeder daemon is actually running on the feeder
ssh <user>@<feeder-ip> 'ss -tlnp | grep 9740'
# If not listening → start it:
ssh <user>@<feeder-ip> 'sudo systemctl start greenboost-netd || sudo nohup /usr/local/bin/greenboost-netd -d -p 9740 &'
# Then re-register:
sudo greenboost connect <feeder-ip>

# 3. Check T3 spillover (NVMe path C) - causes dramatic slowdown
cat /sys/class/greenboost/greenboost/pool_brief
# If T3 > 0 during active inference:
echo 0 | sudo tee /sys/module/greenboost/parameters/kv_reserve_mb

# 4. One-command feeder redeploy if daemon is stale
sudo greenboost feeders upgrade-greenboost
```

---

## TurboQuant - global toggle (v2.9+)

TurboQuant compresses the K/V cache during attention computation, reducing the bandwidth needed to move tensors between T2 DDR and T1 VRAM.

### Enable / disable system-wide

```bash
# Enable for Ollama (q4_0 KV cache) + Python inference apps
sudo greenboost turboquant on

# Disable and revert to q8_0 KV cache
sudo greenboost turboquant off

# Check current state
greenboost turboquant status
```

`turboquant on` does two things simultaneously:

1. **Ollama** - updates `/etc/systemd/system/ollama.service.d/99-greenboost.conf` to set `OLLAMA_KV_CACHE_TYPE=q4_0` and `OLLAMA_FLASH_ATTENTION=1`, then restarts the Ollama service.
2. **Python apps** - writes `/etc/greenboost/turboquant.enabled` and creates `/etc/profile.d/greenboost-turboquant.sh` which exports `GREENBOOST_TURBOQUANT=1` for all future login shells.

### Using TurboQuant in Python code

When `GREENBOOST_TURBOQUANT=1` is set (or the flag file exists), Python inference scripts can opt in:

```python
import sys, os
sys.path.insert(0, "/path/to/greenboost_all/greenboost")
from gb_attn import turboquant_attention

# Recommended: asymmetric K=4bit / V=3bit (~4× bandwidth reduction, +0.23% PPL)
with turboquant_attention(k_bits=4, v_bits=3, sparse_v=True):
    output = model(input)
```

Or check the flag and apply conditionally:

```python
from pathlib import Path
if Path("/etc/greenboost/turboquant.enabled").exists() \
        or os.environ.get("GREENBOOST_TURBOQUANT") == "1":
    from gb_attn import turboquant_attention
    ctx = turboquant_attention(k_bits=4, v_bits=3)
else:
    from contextlib import nullcontext
    ctx = nullcontext()

with ctx:
    output = model(input)
```

---

## gb-quant - weight quantize-to-fit (v3.0+)

The fastest tier is the one you fit in. `gb_quant.py` shrinks model weights
so the working set lives in T1 VRAM at full GPU bandwidth instead of
streaming over PCIe from T2. The planner is quality-first: every component
gets the HIGHEST precision that still fits the budget
(bf16 > int8 > int4 > tq3 > tq2); T3 is never used for weights. The low-bit
Triton GEMM backend ships inside GreenBoost (`third_party/`, Apache-2.0) -
nothing to install in your venv.

Sub-4-bit modes `"tq3"` / `"tq2"` (3.1 / 2.1 bits per weight effective) are
TurboQuant weight quantization (`gb_quant_tq.py`: random rotation +
Lloyd-Max codebook + a Triton LUT-GEMM kernel). Below 4 bits TurboQuant
beats HQQ on quality, so the sub-int4 ladder rungs are TQ; int4-HQQ stays
the default floor. Select per call (`quantize_module(m, bits="tq3")`,
`quantize_to_fit(pipe, prefer_bits="tq3")`) or via env (`GB_QUANT_BITS=tq3`
next to `GB_QUANT_BUDGET_GB`) - the env value is a precision FLOOR, the
planner still gives every component the highest precision that fits.

```python
import sys; sys.path.insert(0, "/path/to/greenboost")
import gb_quant

# any diffusers pipeline or nn.Module:
report = gb_quant.quantize_to_fit(pipe, budget_gb=11.0)

# text-conditioned pipelines, two-phase (encoder freed before denoiser):
embeds = gb_quant.encode_then_quantize(pipe, prompts)

# pipelines you don't own - env hook, zero code:
#   GB_QUANT_BUDGET_GB=11   (or "fit")   then call:
gb_quant.maybe_quantize_from_env(model)
```

LLMs go through `gb_llm.py` (`load_causal_lm` for transformers; vLLM via
the bundled plugin: `vllm serve <model> --quantization gemlite`). Ollama
runs pre-quantized GGUF - there the lever is the shim plus the quant level
you pull.

gb-quant and the tiers are complementary: quantize to fit FIRST, then let
T2 absorb only what genuinely exceeds the quantized footprint. Measured on
an RTX 5070 12 GB: FLUX.2-klein-9B went from ~7 min/image (BF16 via T2
overflow) to ~5 s/image steady state (int4 in T1, peak 8.2 GiB, quality
preserved); gemma-3-12b (22.7 GiB bf16) fits in 6.2 GiB int4.

Note: gb-quant currently runs WITHOUT the LD_PRELOAD shim (its backend
cannot initialise under it yet); for fits-after-quantization models the
shim is unnecessary anyway.

### New tuning knobs (v3.2 AI memory OS tranche)

All default-safe (off, or the previous behavior, unless set):

| Variable | Default | Description |
|----------|---------|-------------|
| `GB_PLACEMENT` | `0` | Enable the `gb_placement.py` fp8-floor cluster-fit planner: prefer a connected feeder over dropping model precision below fp8. |
| `GB_KERNEL_BACKEND` | `gemlite` | Low-bit GEMM backend gb-quant uses: `gemlite`, `scaled_mm` (fp8), `bf16` (passthrough). CUTLASS reserved for future use. |
| `GB_QUANT_NO_AUTOTUNE_CACHE` | `0` | Disable the version-scoped (GPU+CUDA+gemlite) persisted Triton autotune cache. |
| `GB_T3_COMPRESS` | `1` | Transparent zstd compression for models evicted to T3 NVMe. Set `0` to store plain. |
| `GB_QUANT_TIERED_PRECISION` | `0` | Hot (fitting) weights stay fp8; cold overflow tail drops to nvfp4 (Blackwell) or int4. fp8 remains the default when unset. |
| `GREENBOOST_T1_WORKSPACE_MB` | unset (off) | Reserve T1 VRAM for per-step compute workspace during model load, released at inference. |
| `GREENBOOST_KV_PREFETCH` | `0` | Enable phase-aware KV prefetch stats mode, exposing `kv_prefetch_*` telemetry counters. |
| `GB_CUTLASS_ENABLE` | `0` | Build/enable the CUTLASS sm_120a NVFP4 GEMM scaffold (`third_party/gb_cutlass`). Groundwork only. |

---

## Optimized inference configuration generator (v2.9+)

`greenboost gen-inference-config` emits a configuration block tuned for the current machine and environment. Useful when setting up Ollama or a Python inference service on a machine that may be a VM, container, or WSL2 instance.

```bash
# Print Ollama env block + Python snippet to stdout
greenboost gen-inference-config --format both

# Save to a file, with TurboQuant explicitly enabled
greenboost gen-inference-config --format ollama \
    --turboquant \
    --output /etc/greenboost/inference.env

# Then source it before starting Ollama manually
export $(grep -v '^#' /etc/greenboost/inference.env | xargs)
ollama serve
```

Flags:

| Flag | Default | Effect |
|------|---------|--------|
| `--format ollama\|hf\|both` | `ollama` | Which config blocks to emit |
| `--turboquant` | (reads flag file) | Force TurboQuant settings on |
| `--no-turboquant` | - | Force TurboQuant settings off |
| `--output <path>` | stdout | Write to file instead of printing |
| `--model <name>` | - | Include a model name hint in the output |

---

## Cluster - connect remote machines as GPU memory and compute feeders

GreenBoost can use the VRAM, RAM, and **GPU compute** of other machines on your network. The host machine (running Ollama) aggregates all feeder resources into a single virtual GPU. Model layers that overflow into a feeder's VRAM are not just stored there - they are also computed there. GreenBoost automatically dispatches CUDA kernels to whichever machine owns the data, with no manual configuration.

### Set up a feeder machine

```bash
# On the feeder machine - start the daemon
sudo greenboost feed start

# The daemon listens on TCP port 9740
```

### Connect from the host

```bash
# Connect to a feeder (saves to /etc/greenboost/cluster.conf)
sudo greenboost connect 192.168.1.50

# Connect to multiple feeders
sudo greenboost connect 192.168.1.51
sudo greenboost connect 192.168.1.52

# View live cluster status - updates every 5 seconds
greenboost cluster

# Disconnect a feeder
sudo greenboost disconnect 192.168.1.50
```

### What `greenboost cluster` shows

`greenboost cluster` is the live interactive view of all connected feeders. It displays:
- Each feeder's IP, VRAM available, DDR available, connection latency
- Total memory contributed by the cluster
- Per-tier breakdown (T1 feeder / T2 feeder / T3 feeder)

Press `Ctrl+S` to refresh immediately, `Ctrl+C` to exit.

### Upgrading GreenBoost on feeders

After building or updating GreenBoost on the host, push the new binaries to all feeders in one command:

```bash
# Push greenboost-netd + CUDA shim + setup script + build stamp to all feeders,
# then restart their daemon automatically.
sudo greenboost feeders upgrade-greenboost

# Check that all machines are on the same build:
greenboost built-stamp --feeders
```

`built-stamp` shows the `BUILD_ID`, version, git hash, and build date for the local machine.
With `--feeders` it also SSHes to every configured feeder and prints their stamp side-by-side.
Mismatched stamps are highlighted in amber as a reminder to upgrade.

```bash
# Check only the local stamp:
greenboost built-stamp
```

### How the cluster works with Ollama

GreenBoost presents **one single virtual GPU** to Ollama, regardless of how many feeders are connected. Ollama sees one large GPU with total memory = local T1+T2+T3 + all feeder T1+T2+T3. All memory is pooled into device 0 - there is no multi-GPU scheduling.

**Memory allocation order** when VRAM overflows:
```
Local T1 VRAM → Feeder T1 VRAM → Local T2 DDR → Feeder T2 DDR → Local T3 NVMe → Feeder T3 NVMe
```

**Compute dispatch (data-driven, automatic):**
When a CUDA kernel is launched, GreenBoost inspects the kernel's arguments. If any tensor lives on a feeder (identified by its fake pointer in the `0xAA00…` range), the entire kernel is sent to that feeder's GPU over TCP to be executed there. The result is returned to the host. This means feeder GPUs are not idle storage - they actively run inference for the layers stored in their VRAM. No configuration is needed; dispatch follows the data automatically.

---

## gb-synapse , HuggingFace-native, cluster-distributed GGUF serving (v3.2)

> Not to be confused with the "Synapse CLI" terminal assistant below , that's
> a different, older, largely dormant project. **gb-synapse** is GreenBoost's
> own model-serving layer, new in v3.2.

Ollama pulls models from its own registry only. `gb-synapse` (`gb_synapse.py`)
goes straight to the source instead: it pulls GGUF models directly from any
HuggingFace repository, gated or public, given a token, and it also indexes
whatever GGUFs Ollama already downloaded, so one command sees both.

For clustering, gb-synapse doesn't reinvent the wheel , it hands the
cross-node split to llama.cpp's own `--rpc` backend: real layer-granular
tensor split, only activations cross the network, not the weights. The
GreenBoost shim still runs on every node underneath it (`GREENBOOST_CLUSTER=0`
on each), extending that node's own layer share into its local RAM/disk if it
doesn't fit VRAM. In short: llama.cpp's RPC owns the split *between* machines,
GreenBoost owns the tiers *within* each one.

```bash
greenboost doctor                          # cluster hardware view: host + feeders
greenboost recommend                       # which pulled models fit the live cluster budget
sudo greenboost synapse login [TOKEN]      # store a HuggingFace token
sudo greenboost pull <repo>[:quant] [name] # download a GGUF
sudo greenboost synapse build-engine       # build llama-server/llama-cli/rpc-server (run on every node)
greenboost synapse run <model> [port]      # serve it, cluster-aware, default :11435
greenboost synapse ps                      # what's running
greenboost synapse stop <model>
```

`gb_synapse_api.py` is the thin proxy `synapse run` launches in front of the
llama-server it starts: one port speaks Ollama's API (`/api/generate`,
`/api/chat`, `/api/tags`, `/api/show`, `/api/ps`), OpenAI's API (`/v1/*`,
relayed byte-for-byte including streaming), and HuggingFace TGI
(`/generate`, `/generate_stream`) at once, so existing tooling doesn't need
to know anything changed.

Full command reference: [GREENBOOST_COMMANDS.md § gb-synapse](GREENBOOST_COMMANDS.md#-gb-synapse-huggingface-native-cluster-distributed-gguf-serving).

---

## Synapse CLI - terminal AI assistant with GreenBoost integration (v2.9+)

> **Note:** this is the older terminal-assistant project, distinct from
> **gb-synapse** above (the current, cluster-distributed model server, new in
> v3.2). Kept here for reference.

Synapse CLI is the companion terminal assistant at `greenboost_synapse_cli_new/`. Install it once, then run `synapse-cli` or `python synapse_cli.py`.

### GreenBoost features inside Synapse

**Startup banner** shows live GreenBoost state:
```
🐙 Synapse CLI
──────────────────────────────────────────────────
  Model        nutboy02/Qwen3.6-35B-...  (ollama)
  Permissions  auto
  GreenBoost   GB  TQ  +1 feeder(s)
  /backend · /model · /turboquant · /help
```
`GB` = GreenBoost installed · `TQ` = TurboQuant on · `+N feeder(s)` = cluster connected

**`/turboquant` slash command:**
```
/turboquant status    show current TurboQuant state
/turboquant on        sudo greenboost turboquant on
/turboquant off       sudo greenboost turboquant off
```

**System prompt context** - every AI request includes a short GreenBoost paragraph so the model knows it has extended memory available and can be asked to work with large contexts.

### Running the Qwen3.6-35B model

```bash
cd greenboost_synapse_cli_new
python synapse_cli.py --model "qwen3.6:latest"
```

This routes through the Ollama backend (`localhost:11435`). GreenBoost provides the extended memory (T2+T3+cluster) so the 35B model fits even when local VRAM is smaller than the model size. Enable TurboQuant before loading for best throughput:

```bash
sudo greenboost turboquant on
python synapse_cli.py --model "qwen3.6:latest"
```

---

## Diffusion Models - Best Practices

FLUX.1-dev and FLUX.2-klein (image generation) have different memory and performance characteristics from LLM models. Key recommendations for running diffusion pipelines through GreenBoost:

### Model selection and speed

| Model | Mode | VRAM | Est. speed | GreenBoost required |
|---|---|---|---|---|
| FLUX.2-klein | gb-quant int4 | ~8 GB T1 only | **~5 s/image** ★ | No (shimless) |
| FLUX.2-klein | FP8 + TurboQuant | ~16 GB T1+T2 | 4–6 s/image | Optional |
| FLUX.2-klein | BF16 + TurboQuant | ~22 GB T1+T2 | 5–8 s/image | Required |
| FLUX.1-dev | NF4 + TurboQuant | ~8 GB T1 | ~2 min/image | Optional |
| FLUX.1-dev | BF16 + TurboQuant | ~23 GB T1+T2 | ~4 min/image | Required |

★ **Recommended default** (gb-quant) , zero overflow, fits a 12 GB card alone; first image adds ~28 s Triton autotune.

### Environment variables for diffusion pipelines

```bash
# Required for GreenBoost aggregation
export GREENBOOST_ACTIVE=1
export LD_PRELOAD='/usr/local/lib/libgreenboost_cuda.so'

# Diffusion models have no KV cache - set to 0 to give all T2 to model weights
export GREENBOOST_KV_RESERVE_MB=0

# Enable T1↔T2 bandwidth compression (reduces PCIe congestion during denoising steps)
export GREENBOOST_KV_COMPRESS=1

# Keep Path A0 (DMA-BUF zero-copy) disabled - cudaImportExternalMemory not
# supported on RTX 5070 Laptop + current driver combination
export GREENBOOST_A0_DISABLE=1
```

### Why KV reserve = 0 for diffusion

Unlike LLMs, diffusion transformers perform attention over spatial latent tokens - there is no autoregressive KV cache to accumulate across denoising steps. Each step is a full forward pass with no persistent KV state. Setting `GREENBOOST_KV_RESERVE_MB=0` dedicates all T2 DDR to model weight storage, which is where diffusion models need the memory.

### TurboQuant with diffusion models

TurboQuant compresses attention tensors during the forward pass. For diffusion models:
- Effective for BF16 modes: large per-step attention tensors benefit from 4-bit K / 3-bit V compression
- Less critical for FP8 modes: FP8 weights are already compact; TurboQuant still helps on T1↔T2 transfers
- Enable with a `--turboquant` flag in your generation pipeline's script,
  wired to call `gb_attn.turboquant_attention()` / `patch_sdpa()` (this repo
  only provides the layer; personalized pipelines are the ones that reference
  it at their call sites)

### T3 spillover detection

T3 (system RAM via PCIe) causes dramatic slowdown (~10× slower than T2). Monitor during generation:

```bash
watch -n2 'cat /sys/class/greenboost/greenboost/status'
# T3 used_mb should stay at 0 during normal klein/flux generation
```

If T3 > 0 appears, the model did not fit in T1+T2. Switch to a smaller quantization mode or reduce batch size.

---


## Debug Vitals

GreenBoost provides a comprehensive, toggleable debug vitals system that surfaces
**9 data sources** in a single command. Essential for diagnosing issues, measuring
performance, and guiding enhancement work.

### Quick start

```bash
# Show full vitals dump immediately (no flag needed)
greenboost debug vitals

# Enable persistent vitals mode - activates verbose instrumentation
sudo greenboost debug vitals on

# Disable persistent vitals - restores minimal-overhead settings
sudo greenboost debug vitals off
```

### What vitals on/off does

**`sudo greenboost debug vitals on`:**
1. Sets kernel `debug_mode=1` via sysfs (live, no restart required)
2. Writes `/etc/profile.d/greenboost-vitals.sh` → exports `GREENBOOST_NVTX_VERBOSE=1`
3. Creates `/etc/greenboost/debug_vitals.enabled` flag file
4. Restarts `greenboost-idle-reclaim` daemon
5. Lists other GB-aware services that need manual restart
6. Warns if a reboot is needed for full shim injection

**`sudo greenboost debug vitals off`:**
1. Resets `debug_mode=0`
2. Removes profile.d file
3. Removes flag file and restarts services

### Data sources table

| Source | Path | What it provides |
|--------|------|-----------------|
| IOCTL | `/dev/greenboost` | T1/T2/T3/KV/pressure, 24 fields real-time |
| Sysfs brief | `/sys/class/greenboost/greenboost/pool_brief` | One-liner T1/T2/T3 for polling |
| Sysfs status | `/sys/class/greenboost/greenboost/status` | Full text status, KV placement |
| Shim stats | `/run/greenboost/shim_stats` | Path counters, H2D/D2H, dispatch, frag, dedup |
| Phase file | `/run/greenboost/phase` | Current CUDA phase string |
| NVTX log | `/run/greenboost/nvtx_events.log` | Per-allocation events with timestamps |
| Metrics JSON | `/run/greenboost/metrics.json` | Structured JSON + feeder vitals |
| nvidia-smi ext | nvidia-smi extended query | Temp, power, util%, SM/mem clocks |
| Kernel journal | `journalctl -k` | Module events (when debug_mode=1) |

### From greenboost-cli

```bash
/gb-vitals          # Full vitals in the Python CLI (Rich rendering)
/gb-vitals on       # Enable (calls sudo greenboost debug vitals on)
/gb-vitals off      # Disable
```

### In scripts

Your own generation scripts and wizards can show the extended vitals panel
when the flag file exists — check for `/etc/greenboost/debug_vitals.enabled`
and, if present, render an extra panel (e.g. after your normal T1/T2/T3
progress bars) by reading `/run/greenboost/shim_stats` or
`/run/greenboost/metrics.json` at your own checkpoints (before/after each
generation stage is a common choice). This is a convention personalized
pipelines can opt into — GreenBoost itself doesn't require any particular
script structure.

### Typical use cases

| Scenario | What to look for |
|----------|-----------------|
| T3 spillover | `t3_used_mb > 0` or Path C count increasing |
| KV cache pressure | `kv_t2_mb > 0` means KV spilling into DDR |
| Memory fragmentation | `shim_frag_pct > 30` indicates T2 fragmentation |
| Feeder not being used | `shim_remote_alloc_count == 0` despite feeder connected |
| Phase stuck | `shim_phase` stays `MODEL_LOAD` after model should be ready |
| OOM guard triggered | `oom_active = true` in IOCTL data |
| Kernel events | `journalctl -k --grep=greenboost` when debug_mode=1 |

---

## Long-running stability monitor (v3.0+)

Once you have a workload running for hours or days, you want a sanity
check that nothing is drifting. That's what `greenboost stability` does.

### What it watches

The script (`gb_stability_monitor.py`) polls
`/run/greenboost/shim_stats` at a configurable interval and flags four
invariant violations:

1. **Counter regressed.** Fields like `kernel_dispatch_count`,
   `h2d_mb`, `d2h_mb`, `remote_alloc_count`, `tier_*_lifetime_mb`
   should only ever go up. If one decreases without an accompanying
   PID change, that's either a wraparound bug or a reset that wasn't
   declared.
2. **Tier-gauge leak.** Gauges like `kv_t1_tracked_mb`,
   `tier_t1_local_cur_mb`, `tier_t2_local_cur_mb` should drop close to
   zero once the shim's reported phase leaves `INFERENCE` or `STEADY`
   and a cool-down period passes. If the gauge stays above 10 % of its
   peak after 2 minutes of idle, that's a memory leak somewhere.
3. **Fragmentation drift.** `t2_pool_frag_pct` and
   `kv_internal_frag_mb` climbing monotonically across 30 consecutive
   samples (≈15 min at the default 30-second interval) without ever
   retreating is reported as drift.
4. **Shim wedged.** If `timestamp` in `shim_stats` doesn't move for 90
   seconds while the file still exists, the shim has stopped writing -
   usually a sign of a hang inside an LD_PRELOAD hook.

A shim PID change resets the monotone baselines automatically - it's
not a leak, just a fresh process.

### Running it

```bash
# Short check - useful in CI:
greenboost stability --interval 5 --duration 60 --strict

# Day-long observation, JSON output for log shippers:
greenboost stability --interval 30 --duration 86400 --json \
    --log /var/log/greenboost-stability.log

# Until you Ctrl-C, plain text to stdout:
greenboost stability
```

Exit codes:
- `0` clean run, or `--strict` not used.
- `1` at least one violation observed (only when `--strict`).
- `2` invalid arguments / setup error.

### Use with Prometheus

If you already scrape `greenboost_exporter.py`, the stability monitor
covers a different angle: the exporter tells you *what the system
currently looks like*, the stability monitor tells you *whether that
view has been internally consistent over time*.

A workflow that has worked in practice: scrape metrics into Prometheus
for short-term graphs and alerts, run the stability monitor with
`--json --log /var/log/greenboost-stability.log` for the long-running
invariant trail.

---

## Worker pool (v3.0+, opt-in)

`greenboost-netd` is single-threaded by default. A slow CUDA op on one
client (a big `cudaMalloc`, a large `cudaMemcpy`) blocks the reactor
and every other client waits behind it. For most setups that's fine -
clusters are usually 1–4 feeders serving one host.

If you're running a larger cluster or a multi-tenant feeder, set:

```bash
GREENBOOST_WORKERS=8 sudo greenboost feed start
```

That spawns up to 32 worker threads behind a FIFO queue of 256 jobs.
When the queue is full, `gb_workpool_submit()` returns `-1` and the
caller falls back to inline execution - no work is dropped silently.

At the time of v3.0 the scaffold is in place but only diagnostic
handlers use it. Migration of `handle_cuda_memcpy_h2d/d2h` and
`handle_cuda_exec` onto the pool is gated on a lock-discipline audit
(several globals - `g_alloc_lock`, `g_inflight_ops`, `ra_count` -
currently assume single-threaded callers).

You can verify the pool is alive by watching netd logs for
`PR-VV: worker pool active (N threads)` at startup and the drain line
at shutdown reporting submitted/completed/rejected counts.

---

## Sister project: GreenBoost Gaming Suite

Through v2.9 the main GreenBoost repository shipped an experimental
Vulkan layer (`libVkLayer_greenboost.so`) that let Steam games via
Proton see the GreenBoost memory pool. That layer touched a different
graphics API stack (Vulkan, not CUDA), required different testing, and
attracted a different audience (gamers vs. ML engineers).

In v3.0 the Vulkan layer was **extracted** into its own repository:

> **[GreenBoost Gaming Suite](../greenboost_gaming/)**

What you'll find there:

- The **GVM Vulkan implicit layer** that inflates each game's reported
  device-local heap by routing overflow allocations through the
  GreenBoost CUDA pool. Vulkan apps see VRAM that doesn't fit on the
  card.
- A **GTK4 desktop application** (shows up in the GNOME app grid) that
  scans your Steam library, lists installed games with their current
  Proton/DLSS versions, and applies per-game optimal settings.
- A **DLSS / FSR / XeSS updater** that finds the relevant DLLs inside
  each Steam Proton prefix and bumps them to the latest version.
- A **GPU profile editor** (clocks, power limit, fan curve) for
  sustained gaming sessions.
- The **GreenBoost Proton wrapper** that injects the environment
  variables games need to activate the layer.

### Installation pre-req

The Gaming Suite **requires GreenBoost (this project) to be installed
first**. The Vulkan layer is just a frontend onto the same CUDA pool
that your inference workflow already uses, so the kernel module and
shim must be in place. The Gaming Suite installer detects whether
GreenBoost is present and refuses with a clear instruction if not.

### Do I need it?

- Doing AI inference only → **no.** You can ignore the Gaming Suite
  entirely.
- Mostly gaming, no AI work → install GreenBoost (this repo) **and**
  the Gaming Suite. The CUDA inference parts of GreenBoost stay idle;
  it's the kernel module + memory pool that the Vulkan layer needs.
- Both → install both. GreenBoost provides the pool, the Gaming Suite
  exposes it to Vulkan games, and your inference workflow continues
  to work unchanged.

---

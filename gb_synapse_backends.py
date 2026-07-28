#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_synapse_backends.py — engine backend abstraction for gb-synapse.

gb_synapse.py owns model sourcing (pull/manifest), GGUF metadata, and the
shared serve-tail (proxy launch, run-state, dataflux `synapse_serve` events).
This module owns HOW each engine actually gets launched: llama.cpp (the
default, cluster --rpc split), the synapse torch engine (gb-quant
safetensors, cluster PP split — see SynapseTorchBackend), transformers (its
fallback when the torch venv isn't installed), and diffusers (HF image-gen
pipelines).

`gb_synapse.serve()` calls `select_backend(entry).serve(...)` — one call
site, format/engine detection decides which backend runs. Every backend
implements the same four-method surface (`EngineBackend`) so gb_synapse
never branches on engine name itself.

Import direction is one-way: gb_synapse.py imports this module at the top;
this module imports gb_synapse (and gb_cluster) LAZILY inside methods to
avoid a circular top-level import.
"""
from __future__ import annotations

import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_DIR))


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _has_weight_files(names) -> bool:
    """True if `names` (an iterable of filenames) contains at least one real
    weight file: `*.safetensors`, or `*.bin` other than `training_args.bin`
    (a training-loop artifact, never a weight checkpoint, that HF repos and
    local `transformers.Trainer` output dirs commonly ship alongside the
    real `pytorch_model.bin`/`pytorch_model-NNNNN-of-NNNNN.bin` shards)."""
    for n in names:
        if n.endswith(".safetensors"):
            return True
        if n.endswith(".bin") and Path(n).name != "training_args.bin":
            return True
    return False


def detect_format(path_or_repo: str) -> str:
    """"gguf" | "safetensors" | "diffusers" | "unknown" for a local
    directory/file OR a bare "org/repo" HF repo id (queried via the HF API
    when `path_or_repo` isn't a path that exists locally). "safetensors" also
    covers `.bin`-only checkpoints (older HF exports, pre-safetensors) — the
    return value name stays "safetensors" since both route to the same
    torch-core engine backend."""
    p = Path(path_or_repo)
    if p.exists():
        if p.is_dir():
            if (p / "model_index.json").is_file():
                return "diffusers"
            if any(p.glob("*.gguf")):
                return "gguf"
            names = [f.name for f in p.iterdir() if f.is_file()]
            if (p / "config.json").is_file() and _has_weight_files(names):
                return "safetensors"
            return "unknown"
        return "gguf" if p.suffix == ".gguf" else "unknown"

    try:
        from huggingface_hub import HfApi
        import gb_synapse as gs
        api = HfApi(token=gs.hf_token())
        info = api.model_info(path_or_repo, files_metadata=False)
        names = {s.rfilename for s in info.siblings}
    except Exception:
        return "unknown"
    if "model_index.json" in names:
        return "diffusers"
    if any(n.endswith(".gguf") for n in names):
        return "gguf"
    if "config.json" in names and _has_weight_files(names):
        return "safetensors"
    return "unknown"


# Below the fp8 quality floor — served, but flagged in serve_facts so the
# fp8-floor policy (gb_mcp.py quant_advisor, gb_actuation.set_quant_policy)
# has a truthful signal for THIS backend too, not just gb_quant's own budget
# planner. Legacy token set for GGUF/llama.cpp-routed entries (llama.cpp
# quant is a filename token, not parsed checkpoint metadata — no
# quant_method/quant_bits to check).
_BELOW_FP8_QUANTS = {"INT8", "INT4"}

# Checkpoint-truth below-fp8-floor bit widths per quant_method (Phase 2's
# read_quant_config/_pull_torch) — a method/bits check, not a fixed token
# set, since e.g. GPTQ2/GPTQ3/GPTQ4 all count regardless of the exact bit
# width, and new bit widths shouldn't need a new set entry.
_BELOW_FP8_METHOD_BITS = {
    "gptq": (2, 3, 4),
    "awq": (4,),
    "compressed-tensors": (4,),
    "bitsandbytes": (4, 8),
}


def _quant_below_fp8_floor(entry) -> bool:
    """True if `entry`'s quantization is below the fp8 quality floor.
    Prefers checkpoint truth (`entry.quant_method`/`entry.quant_bits`,
    Phase 2) when set; falls back to the legacy `_BELOW_FP8_QUANTS` token
    check for GGUF/llama.cpp-routed entries, which carry no quant_method at
    all (GGUF quant is a filename token, not parsed checkpoint metadata)."""
    if entry.quant_method:
        return entry.quant_bits in _BELOW_FP8_METHOD_BITS.get(entry.quant_method, ())
    return entry.quant.upper() in _BELOW_FP8_QUANTS


def _cuda_home_for_torch() -> "str | None":
    """FlashInfer (used by the synapse torch engine) JIT-compiles Blackwell
    (SM 12.x) kernels and requires CUDA >= 12.9 to do it — it finds nvcc via
    CUDA_HOME, which defaults to /usr (a stale distro-packaged nvcc, often
    < 12.9) when unset. Same root cause as gb_synapse._find_nvcc(): the
    distro package and /usr/local/cuda's update-alternatives-managed install
    can diverge. FlashInfer swallows the resulting version-check exception
    internally and misreports "GPU too old" instead of "CUDA too old" —
    confirmed live on an RTX 5070 (cc 12.0) with CUDA 13.3 at
    /usr/local/cuda but 12.4 at /usr/bin/nvcc."""
    alt = Path("/usr/local/cuda")
    return str(alt) if alt.exists() else None


def _torch_env_dir() -> "Path | None":
    """The synapse torch engine's venv directory: explicit override, the
    user-home dev venv (`gb-synapse install-synapse-engine`-equivalent run
    by hand), then the root-owned Full-Install venv
    ($GB_PY_DEST/synapse-torch-env — see cmd_install_synapse_engine in
    greenboost_setup.sh). Returns the first candidate that actually has a
    `bin/python` (a bare empty dir from a half-finished install doesn't
    count as "available")."""
    override = os.environ.get("GB_SYNAPSE_TORCH_ENV")
    candidates = ([Path(override)] if override else []) + [
        Path.home() / ".local/share/greenboost/synapse/torch-env",
        Path("/usr/local/lib/greenboost/synapse-torch-env"),
    ]
    for c in candidates:
        if (c / "bin/python").exists():
            return c
    return None


def _find_torch_venv_lib(rel_glob: str) -> "Path | None":
    """A library bundled inside the synapse torch engine's venv
    site-packages (e.g. torch's own pinned nvidia-cuXX wheel's libcudart),
    for shim_env's cudart_path."""
    venv = _torch_env_dir()
    if not venv:
        return None
    matches = list(venv.glob(f"lib/python*/site-packages/{rel_glob}"))
    return matches[0] if matches else None


# Cached per-venv-path: is `import gllm, sgl_kernel` actually importable in
# this venv? A real subprocess probe (imports have real side effects — CUDA
# context, etc. — not safe to do in-process from gb_synapse's own
# interpreter), so cache the result rather than re-probing on every serve.
_TORCH_ENGINE_IMPORTABLE_CACHE: dict = {}


def _torch_engine_importable(venv_py: str) -> bool:
    if venv_py in _TORCH_ENGINE_IMPORTABLE_CACHE:
        return _TORCH_ENGINE_IMPORTABLE_CACHE[venv_py]
    try:
        r = subprocess.run([venv_py, "-c", "import gllm, sgl_kernel"],
                           capture_output=True, timeout=30)
        ok = r.returncode == 0
    except Exception:
        ok = False
    _TORCH_ENGINE_IMPORTABLE_CACHE[venv_py] = ok
    return ok


def _torch_serve_mode(entry) -> "tuple[str, str]":
    """("gllm"|"fallback", reason) — pure function (the one subprocess probe
    it makes, _torch_engine_importable, is itself cached, so repeated calls
    have no side effects). Decides whether SynapseTorchBackend hands `entry`
    to the real synapse torch engine (gLLM) or to the single-request
    transformers+gb-quant fallback server (gb_synapse_fallback.py), in
    order:

    (a) the venv doesn't actually have a working gllm/sgl_kernel import
        (missing/broken install) — fallback.
    (b) `entry.arch` isn't one gLLM's own model_loader recognizes
        (`gb_synapse._torch_engine_supported_archs()`) — fallback.
    (c) the checkpoint is NOT already quantized (`quant_method == ""`) AND
        either its bf16 byte size exceeds the live VRAM+T2 budget, or the
        caller explicitly requested a below-fp8 requantization token
        (`:INT8`/`:INT4` — gLLM has no on-the-fly quantize-to-fit the way
        the fallback's gb_quant does) — fallback.
    Reason is a short human-readable string for the "prints WHY" caller
    contract; empty when mode is "gllm"."""
    venv = _torch_env_dir()
    if not venv or not _torch_engine_importable(str(venv / "bin" / "python")):
        return "fallback", "synapse torch engine venv missing or import-broken"

    import gb_synapse as gs
    supported = gs._torch_engine_supported_archs()
    if entry.arch and supported and entry.arch not in supported:
        return "fallback", (f"architecture '{entry.arch}' not recognized by "
                            f"the vendored synapse torch engine")

    if not entry.quant_method:
        _, effective_free_mb, _ = effective_vram_budget_mb()
        bf16_mb = entry.n_bytes / (1024 ** 2)
        if bf16_mb > effective_free_mb:
            return "fallback", (f"{bf16_mb / 1024:.1f} GB bf16 exceeds the "
                                f"{effective_free_mb / 1024:.1f} GB VRAM+T2 budget "
                                f"(gLLM has no quantize-to-fit)")
        if entry.quant.upper() in _BELOW_FP8_QUANTS:
            return "fallback", (f"requested {entry.quant} requantization — gLLM has "
                                f"no on-the-fly quantize-to-fit, only the fallback's "
                                f"gb_quant does")

    return "gllm", ""


def _gbquant_model_source(entry) -> str:
    """Local snapshot dir if `_pull_gbquant()`/`_pull_diffusers()` already
    downloaded one (avoids the torch engine/transformers/diffusers
    redundantly re-fetching the same repo into their own default HF cache
    instead of reusing gb-synapse's own MODEL_STORE_DIR/_hf_cache copy), else
    the bare repo id so the runtime can fetch it itself."""
    return entry.path if entry.path and os.path.isdir(entry.path) else entry.repo


# ---------------------------------------------------------------------------
# %-derived VRAM+T2 budget sizing (2.3 — kills the old hardcoded 0.5/0.85)
# ---------------------------------------------------------------------------

def effective_vram_budget_mb() -> "tuple[float, float, dict]":
    """(host_free_mb, effective_free_mb, facts) for sizing a torch-workload
    engine's memory-utilization flag against the LIVE budget.

    effective_free_mb = host_free_mb + t2_free_mb * GB_SYNAPSE_T2_FRACTION
    (default 0.5, env-overridable) — T2 genuinely extends usable capacity
    when the shim is active, so sizing against physical VRAM alone
    under-fills T1. Setting
    GB_SYNAPSE_T2_FRACTION=0 reproduces pre-shim (shimless) sizing exactly —
    used as the negative control in the T2 validation matrix.

    `facts` feeds `serve_facts()` for dataflux (workflow/gb-synapse.md T2
    validation procedure)."""
    try:
        frac = float(os.environ.get("GB_SYNAPSE_T2_FRACTION", "0.5"))
    except ValueError:
        frac = 0.5

    t2_free_mb = 0.0
    try:
        brief = Path("/sys/class/greenboost/greenboost/pool_brief").read_text()
        m = re.search(r"T2:(\d+)/(\d+)GB", brief)
        if m:
            t2_free_mb = float((int(m.group(2)) - int(m.group(1))) * 1024)
    except OSError:
        pass

    host_free_mb = host_total_mb = 0.0
    try:
        from gb_nvml import get_nvml
        _, host_free_mb, host_total_mb, _ = get_nvml(0).mem()
    except Exception:
        pass
    if host_total_mb <= 0:
        # pynvml not installed — gb_nvml returns all zeros by design in that
        # case. torch is already a hard requirement of the torch engine/
        # transformers, so this import is free on any path that reaches
        # this function.
        try:
            import torch
            free_b, total_b = torch.cuda.mem_get_info()
            host_free_mb, host_total_mb = free_b / (1024 ** 2), total_b / (1024 ** 2)
        except Exception:
            host_free_mb = host_total_mb = 0.0

    effective_free_mb = host_free_mb + t2_free_mb * frac
    facts = {"t2_free_mb": round(t2_free_mb, 1), "t2_fraction": frac,
             "host_free_mb": round(host_free_mb, 1), "host_total_mb": round(host_total_mb, 1)}
    return host_free_mb, effective_free_mb, facts


# ---------------------------------------------------------------------------
# T2 validation + Rule #1 tripwire (2.5)
# ---------------------------------------------------------------------------

_SHIM_STATS_PATH = Path("/run/greenboost/shim_stats")


def _read_shim_stats() -> dict:
    stats: dict[str, str] = {}
    try:
        for line in _SHIM_STATS_PATH.read_text().splitlines():
            k, _, v = line.partition("=")
            if k:
                stats[k] = v
    except OSError:
        pass
    return stats


def _validate_placement(entry, util: float, budget_facts: dict, engine: str) -> None:
    """After a synapse-torch-engine serve, confirm the weights genuinely
    landed partly in T2 rather than the whole overflow silently sitting in
    T2 while T1 stayed under-filled (Rule #1). Reads
    /run/greenboost/shim_stats (t2 bytes, defer_init) + live NVML
    fb_used_pct; emits a `synapse_engine_placement` dataflux event with
    `rule1_warning=true` when t2_bytes>0 while vram_used_pct is below
    GB_SYNAPSE_RULE1_PCT (default 85). Best-effort, never raises.

    Renamed from `_validate_genuine_t2` / event kind renamed from
    `synapse_vllm_placement` (2026-07-16, Phase 4 backend wiring) — this
    validation is engine-agnostic; `engine` says which backend actually
    served (the vLLM backend this originally validated was retired in
    Phase 6)."""
    try:
        stats = _read_shim_stats()
        t2_bytes = (int(stats.get("tier_t2_local_cur_mb", "0") or 0)
                    + int(stats.get("tier_t2_feeder_cur_mb", "0") or 0)) * 1024 * 1024

        from gb_nvml import get_nvml
        _, free_mb, total_mb, _ = get_nvml(0).mem()
        vram_used_pct = round((1 - free_mb / total_mb) * 100, 1) if total_mb else 0.0

        try:
            rule1_pct = float(os.environ.get("GB_SYNAPSE_RULE1_PCT", "85"))
        except ValueError:
            rule1_pct = 85.0
        rule1_warning = t2_bytes > 0 and vram_used_pct < rule1_pct

        # defer_init is the new shim_stats key; vllm_compat is read as a
        # fallback for a shim built before the C-side rename (P4.6).
        defer_init = stats.get("defer_init", stats.get("vllm_compat", "0")) == "1"

        import gb_dataflux
        gb_dataflux.emit({
            "kind": "synapse_engine_placement", "status": "warn" if rule1_warning else "ok",
            "model": entry.name, "engine": engine,
            "vram_used_pct": vram_used_pct, "t2_bytes": t2_bytes,
            "t2_fraction": budget_facts.get("t2_fraction"),
            "gpu_mem_util_flag": round(util, 2), "quant": entry.quant,
            "defer_init": defer_init,
            "rule1_warning": rule1_warning,
        })
    except Exception:
        pass


# ---------------------------------------------------------------------------
# EngineBackend
# ---------------------------------------------------------------------------

class EngineBackend:
    name = "base"

    def available(self) -> bool:
        raise NotImplementedError

    def can_serve(self, entry) -> bool:
        raise NotImplementedError

    def serve(self, entry, port, ctx=0, use_cluster=True, n_slots=-1, extra_args="", cuda_graph=None):
        """n_slots=-1 (default) means "let the engine decide" — for
        LlamaCppBackend this passes straight through to llama.cpp's own
        `-np -1` auto mode (n_parallel=4, kv_unified=true, a single shared
        KV pool split across slots, not 4x the memory), rather than
        GreenBoost overriding it with a hardcoded 1 and silently disabling
        upstream's own continuous-batching default (Speed Program audit
        Finding 10, 2026-07-26: n_slots defaulted to 1 everywhere, disabling
        batching entirely with no one ever passing a real value). Backends
        that don't support concurrent slots (Transformers/Diffusers) ignore
        this parameter regardless of its value.

        cuda_graph=None (default) means "use the GB_SYNAPSE_TORCH_CUDA_GRAPH
        env var" (SynapseTorchBackend only — CUDA graph capture reserves
        extra warmup buffers on top of the KV cache and can OOM on small
        cards, off by default; see that backend's serve() for the full
        rationale). An explicit True/False overrides the env var for this
        one serve call — added so callers (the synapse_serve MCP tool
        included) can opt a specific serve into graphs without mutating
        process-wide environment state. Backends without a CUDA-graph
        concept ignore this parameter regardless of its value."""
        raise NotImplementedError

    def serve_facts(self, entry) -> dict:
        """Extra dataflux fields for this serve — see gb_synapse._emit_serve."""
        return {}


class LlamaCppBackend(EngineBackend):
    """The default, cluster-aware engine — llama.cpp with an explicit
    --rpc/--tensor-split across online feeders, and the shim's local T2/T3
    tiers absorbing whatever's left over per node. See gb_synapse.serve()'s
    former docstring (now here) for the full flag rationale."""
    name = "llama.cpp"

    def available(self) -> bool:
        import gb_synapse as gs
        return gs.engine_installed()

    def can_serve(self, entry) -> bool:
        return entry.engine == "llama.cpp"

    def serve(self, entry, port, ctx=0, use_cluster=True, n_slots=-1, extra_args="", cuda_graph=None):
        import gb_synapse as gs
        import gb_cluster
        from gb_nvml import get_nvml

        if not gs.engine_installed():
            raise RuntimeError("engine not built — run: greenboost synapse build-engine")

        supported = gs._engine_supported_archs()
        if entry.arch and supported and entry.arch not in supported:
            hint = (f"run it directly with Ollama instead: ollama run {entry.name}"
                    if entry.source == "ollama" else
                    "wait for upstream llama.cpp to add support, or pick a different quant/repo")
            raise RuntimeError(
                f"'{entry.name}' uses architecture '{entry.arch}', which this gb-synapse engine "
                f"build (llama.cpp {gs.engine_version()}) doesn't recognize — likely a custom/newer "
                f"arch only another runtime supports; {hint}."
            )

        _, host_free_mb, _, _ = get_nvml(0).mem()

        online_feeders, rpc_args = [], []
        if use_cluster:
            for i, f in enumerate(gb_cluster.feeders(probe=True)):
                if not f.online:
                    continue
                rpc_port = gs.RPC_PORT_BASE + i
                # Only a feeder we can actually REACH may influence the budget and
                # the split. Degrading to a host-only serve is always better than
                # handing llama.cpp a device it can't talk to, which fails the load
                # outright instead of just being slower.
                if not gs.ensure_feeder_rpc(f, rpc_port):
                    continue
                rpc_args.append(f"{f.ip}:{rpc_port}")
                online_feeders.append(f)
        budget_gb = host_free_mb / 1024 + sum(f.t1_free_mb for f in online_feeders) / 1024

        internal_port = port + 1000
        env = gb_cluster.shim_env(workload="llm", enabled=True)
        env["GREENBOOST_CLUSTER"] = "0"  # RPC owns cross-node split; shim = local tiers only
        # Pin the loader to the engine dir: the binary's RUNPATH points at the
        # BUILD TREE, so without this a rebuilt tree silently swaps the ggml libs
        # under an older binary (confirmed by crash backtraces referencing
        # build-synapse/bin paths) — two engine versions in one process.
        env["LD_LIBRARY_PATH"] = str(gs.ENGINE_DIR) + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")

        # The shim currently aborts llama.cpp's CUDA backend on this host:
        # LD_PRELOAD of libgreenboost_cuda.so makes cudaFuncGetAttributes return
        # "invalid device function" inside ggml_cuda_kernel_can_use_pdl (reproduced
        # with GREENBOOST_ACTIVE=0, so it is the interposition itself, not a
        # GreenBoost code path; a plain cudart-12 test binary is unaffected, so the
        # trigger is ggml's late-dlopen'ed backend). Serving through a shim that
        # kills the engine is strictly worse than serving without it: RPC still
        # splits across the cluster and gb-quant still holds quality, we only lose
        # per-node T2/T3 spill. Probe once, then decide — never assume.
        if os.environ.get("GB_SYNAPSE_SHIM", "0") != "1":
            for k in ("LD_PRELOAD", "GREENBOOST_ACTIVE"):
                env.pop(k, None)
            shim_note = "off (known llama.cpp/CUDA incompatibility)"
        else:
            shim_note = "on (GB_SYNAPSE_SHIM=1)"

        # -ngl 999 ("every layer on the GPU") is right only when the weights can
        # actually LAND there. Without the shim inflating VRAM, forcing all layers
        # onto a card that cannot hold them doesn't spill, it fails the load
        # outright ("unable to allocate CUDA0 buffer of size 21398561280"). When the
        # weights exceed the VRAM budget, let llama.cpp fit the layers itself: it
        # puts as many on the GPU as fit and keeps the rest in host RAM. That is
        # slower, and it is what the cluster (RPC) and the shim (T2) exist to avoid,
        # so we say plainly which one we got.
        weights_gb = entry.n_bytes / (1024 ** 3)
        fits_vram = weights_gb <= budget_gb or "LD_PRELOAD" in env
        cpu_quirk = (not fits_vram and entry.arch in gs.ARCH_CPU_SPLIT_BROKEN
                     and os.environ.get("GB_SYNAPSE_FORCE_SPLIT") != "1")

        # ctx=0 means "whatever the model itself was certified for" — a 1M YaRN bake
        # exists precisely so it can be USED, and a hardcoded 64K default silently
        # discarded 94% of it. The budget clamp below still decides what actually
        # fits, so asking for the model's full context is never dangerous, just
        # honest about the ceiling.
        #
        # Which budget, though: when `cpu_quirk` is about to force this model onto
        # the CPU-only arch-split-crash path, weights AND KV cache live in plain
        # host+feeder RAM, not VRAM — clamping ctx/KV-type against the (already
        # weights-insufficient) VRAM budget starved ctx down to a few thousand
        # tokens even with 60+ GB of RAM sitting idle. Use the real RAM budget then.
        ctx_kv_budget_gb = budget_gb
        if cpu_quirk:
            ram_free_mb = gs._read_ram_available_mb()
            ctx_kv_budget_gb = max(
                budget_gb,
                ram_free_mb / 1024 + sum(f.t2_free_mb for f in online_feeders) / 1024)

        if ctx <= 0:
            ctx = entry.ctx_length or 65536
        ctx = gs._clamp_ctx_to_budget(ctx, entry, ctx_kv_budget_gb)

        # KV precision is a QUALITY tier, not a memory knob, so it is chosen against
        # the budget rather than hardcoded: f16 is certification grade, q8_0 is a
        # budget config. Take f16 whenever it fits; drop to q8_0 only to make the
        # context reachable at all.
        kv_type = gs._pick_kv_type(ctx, entry, ctx_kv_budget_gb)

        kv_total_gb = gs.estimate_kv_gb(ctx, entry.n_bytes, entry.quant,
                                        n_layers=(entry.n_kv_layers or entry.n_layers),
                                        n_kv_heads=entry.n_kv_heads,
                                        head_dim=entry.head_dim,
                                        kv_bytes_per_elem=2.0 if kv_type == "f16" else 1.0)
        tensor_split = gs._compute_tensor_split(host_free_mb, online_feeders, kv_total_gb)

        gs.SLOT_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [str(gs.ENGINE_DIR / "llama-server"), "-m", entry.path,
               "--host", "127.0.0.1", "--port", str(internal_port),
               "--ctx-size", str(ctx),
               "--slot-save-path", str(gs.SLOT_DIR), "--no-webui",
               "--flash-attn", "auto",
               "--cache-reuse", "256",
               "--no-mmap",  # DMA-BUF pinning (Path A) is incompatible with mmap-backed pages
               # n_slots defaults to -1 (EngineBackend.serve's docstring):
               # passed straight through, llama.cpp's own arg parser treats
               # -1 as "auto" (n_parallel=4, kv_unified=true) — no GreenBoost
               # override needed here.
               "-np", str(n_slots), "--threads", str(gs._pcore_threads()),
               "--cache-type-k", kv_type, "--cache-type-v", kv_type,
               # llama.cpp's own `-fit` auto-placement (default ON, common/fit.cpp)
               # exists to size -ngl/--tensor-split itself when the caller leaves
               # them unset. GreenBoost always sets -ngl explicitly (this function's
               # whole job), which `-fit`'s own pre-flight check treats as "the user
               # already decided, abort" the moment it needs to change ANYTHING —
               # confirmed live 2026-07-27 against an RPC feeder split: it rejected
               # every -ngl value tried (999 down to 11/65) with the identical
               # "n_gpu_layers already set by user to N, abort", because the
               # rejection fires on ANY explicit -ngl once the multi-device margin
               # check wants an adjustment, independent of what N actually is (see
               # common/fit.cpp's early-return-if-already-fits path vs. the
               # unconditional throw once that path is missed). GreenBoost already
               # owns this decision (_fit_gpu_layers/_compute_tensor_split), so
               # llama.cpp's own redundant veto is disabled rather than raced.
               "-fit", "off",
               "--jinja"]

        n_cpu_moe = 0
        n_gpu = None
        if fits_vram:
            cmd += ["-ngl", "999"]        # all layers on GPU — the fast path
        elif cpu_quirk:
            # Upstream llama.cpp CUDA regression: for this arch ANY CPU/GPU split
            # aborts at the first batched prompt. -ngl 0 alone is NOT CPU-only:
            # llama.cpp still op-offloads batched matmuls above ~32 tokens, so hide
            # the GPU entirely. See workflow/known-issues.md.
            cmd += ["-ngl", "0", "--no-op-offload"]
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif entry.is_moe:
            # Every layer still runs on the GPU; only the experts of the first
            # n_cpu_moe layers live in host DDR.
            n_cpu_moe = gs._fit_cpu_moe_layers(entry, budget_gb, kv_total_gb,
                                               1 + len(online_feeders))
            cmd += ["-ngl", "999", "--n-cpu-moe", str(n_cpu_moe)]
        else:
            n_gpu = gs._fit_gpu_layers(entry, budget_gb, kv_total_gb, 1 + len(online_feeders))
            cmd += ["-ngl", str(n_gpu)]

        # No mmap: DMA-BUF pinning needs owned pages. Without the shim there is
        # nothing to pin, and mmap lets the page cache serve a 20 GB GGUF instead of
        # re-reading it from disk on every restart.
        if "LD_PRELOAD" not in env:
            cmd.remove("--no-mmap")

        # Vision: a GGUF's projector is a SEPARATE file. Without --mmproj a VLM is
        # served text-only and every image is silently dropped.
        mmproj = gs._find_mmproj(entry)
        if mmproj:
            cmd += ["--mmproj", str(mmproj)]

        # MTP: models carrying a grafted multi-token-prediction layer decode ~34%
        # faster through llama.cpp's speculative path, with identical output.
        if gs._has_mtp(entry):
            cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(gs.MTP_DRAFT_N)]

        if rpc_args:
            cmd += ["--rpc", ",".join(rpc_args), "--tensor-split", tensor_split]
        if extra_args:
            cmd += shlex.split(extra_args)

        kv_grade = "certification-grade" if kv_type == "f16" else "budget"
        if fits_vram:
            placement = "all-GPU"
        elif cpu_quirk:
            placement = "CPU-ONLY (arch CUDA-split quirk — see known-issues.md)"
        else:
            placement = "PARTIAL CPU OFFLOAD (slow)"
        print(f"  [gb-synapse] {entry.name}: ctx={ctx} kv={kv_type} ({kv_grade}) "
              f"weights={weights_gb:.1f}GB/{budget_gb:.1f}GB budget → {placement}, "
              f"shim={shim_note}"
              f"{' +mtp' if gs._has_mtp(entry) else ''}{' +vision' if mmproj else ''}"
              f"{' rpc=' + ','.join(rpc_args) if rpc_args else ''}", flush=True)
        if cpu_quirk:
            print(f"  [gb-synapse] {weights_gb:.1f} GB of weights vs {budget_gb:.1f} GB VRAM "
                  f"AND arch '{entry.arch}' cannot survive a CPU/GPU split on this llama.cpp "
                  f"build (upstream CUDA regression) — serving CPU-ONLY. Measured 11.2 tok/s "
                  f"decode (3B-active MoE). Re-test the split with GB_SYNAPSE_FORCE_SPLIT=1 "
                  f"after an engine upgrade.", flush=True)
        elif not fits_vram and n_cpu_moe:
            print(f"  [gb-synapse] {weights_gb:.1f} GB of weights vs {budget_gb:.1f} GB VRAM: "
                  f"ALL {entry.n_layers} layers stay on GPU (attention + KV); experts of "
                  f"{n_cpu_moe}/{entry.n_layers} layers held in DDR "
                  f"({gs._moe_expert_gb_per_layer(entry):.2f} GB/layer). Only 8 of "
                  f"{entry.n_experts} experts fire per token, so this is the cheapest "
                  f"weight in the model to keep out of VRAM.", flush=True)
        elif not fits_vram:
            _fit = gs._fit_gpu_layers(entry, budget_gb, kv_total_gb, 1 + len(online_feeders))
            print(f"  [gb-synapse] {weights_gb:.1f} GB of weights do not fit "
                  f"{budget_gb:.1f} GB of VRAM: {_fit}/{entry.n_layers} layers on GPU, the "
                  f"rest on CPU. CPU layers cost far more than any memory transfer — add "
                  f"VRAM (another feeder) or a smaller quant to close the gap.", flush=True)

        llama_log = open(gs._run_log_path(entry.name), "ab")
        llama_proc = subprocess.Popen(cmd, env=env, stdout=llama_log, stderr=subprocess.STDOUT,
                                       start_new_session=True)

        try:
            return gs._launch_proxy_and_record(entry, llama_proc, port, internal_port,
                                               tensor_split=tensor_split,
                                               feeders=[f.ip for f in online_feeders],
                                               engine=self.name,
                                               ctx=ctx, kv_type=kv_type, placement=placement,
                                               mtp=bool(gs._has_mtp(entry)), vision=bool(mmproj))
        except RuntimeError as e:
            err_lower = str(e).lower()
            is_oom = "out of memory" in err_lower
            # A desktop with many long-running GPU contexts can fragment VRAM badly
            # enough that a plain cudaMalloc for a buffer well UNDER the nominally-
            # free byte count still fails. The shim (which would otherwise absorb
            # this via T2 spill) is unusable with llama.cpp on this build — so a raw
            # -ngl 999 fragmentation OOM has no other net. Retry ONCE at whatever
            # layer count actually fits without a single giant weights allocation.
            if fits_vram and is_oom:
                fallback_ngl = gs._fit_gpu_layers(entry, budget_gb * 0.7, kv_total_gb,
                                                  1 + len(online_feeders))
                print(f"  [gb-synapse] {entry.name}: all-GPU load hit a VRAM "
                     f"fragmentation OOM (weights fit the nominal budget but no "
                     f"contiguous block was actually free) — retrying at "
                     f"-ngl {fallback_ngl}/{entry.n_layers} (partial GPU offload).",
                     flush=True)
                retry_cmd = list(cmd)
                retry_cmd[retry_cmd.index("-ngl") + 1] = str(fallback_ngl)
                llama_log = open(gs._run_log_path(entry.name), "ab")
                llama_proc = subprocess.Popen(retry_cmd, env=env, stdout=llama_log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
                return gs._launch_proxy_and_record(
                    entry, llama_proc, port, internal_port,
                    tensor_split=tensor_split, feeders=[f.ip for f in online_feeders],
                    engine=self.name, ctx=ctx, kv_type=kv_type,
                    placement="PARTIAL CPU OFFLOAD (VRAM fragmentation fallback)",
                    mtp=bool(gs._has_mtp(entry)), vision=bool(mmproj))
            # The dense partial-offload branch (`_fit_gpu_layers`) sizes -ngl against
            # the %-derived compute-graph reserve (`_compute_reserve_gb`), but that
            # reserve is a flat per-device estimate tuned for plain attention-only
            # architectures. A hybrid-attention arch (e.g. Gated Delta Net layers,
            # confirmed live 2026-07-27 against the Qwen3.6-27B-Fable-Fusion
            # reference model: `graph_reserve: failed to allocate compute buffers`
            # at -ngl 25/65, OOM on a 183 MB allocation) can need more graph
            # workspace than that estimate provides. Same fix as the fragmentation
            # case above: back off the layer count and retry once rather than
            # failing the load outright over a margin that's cheap to give back.
            if n_gpu and is_oom:
                fallback_ngl = max(0, n_gpu - max(1, math.ceil(n_gpu * 0.15)))
                print(f"  [gb-synapse] {entry.name}: partial-GPU load (-ngl {n_gpu}) "
                     f"still hit a VRAM OOM allocating compute-graph buffers on top "
                     f"of the weights (hybrid-attention archs need more workspace "
                     f"than the flat compute-reserve estimate) — retrying at "
                     f"-ngl {fallback_ngl}/{entry.n_layers}.", flush=True)
                retry_cmd = list(cmd)
                retry_cmd[retry_cmd.index("-ngl") + 1] = str(fallback_ngl)
                llama_log = open(gs._run_log_path(entry.name), "ab")
                llama_proc = subprocess.Popen(retry_cmd, env=env, stdout=llama_log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
                return gs._launch_proxy_and_record(
                    entry, llama_proc, port, internal_port,
                    tensor_split=tensor_split, feeders=[f.ip for f in online_feeders],
                    engine=self.name, ctx=ctx, kv_type=kv_type,
                    placement="PARTIAL CPU OFFLOAD (compute-buffer OOM fallback)",
                    mtp=bool(gs._has_mtp(entry)), vision=bool(mmproj))
            raise

    def serve_embedding(self, entry, primary_port: int) -> "tuple[subprocess.Popen, int]":
        """Launch a SECOND, minimal llama-server in --embeddings mode for
        `entry`, on an internal port distinct from the primary engine's own
        (primary uses `port + 1000`; this uses `port + 2000`). Host-only,
        deliberately simpler than serve() above: an embedding model is
        typically small, and a RAG client's latency budget doesn't benefit
        from a cluster RPC split the way a large generation model does, so
        this skips the feeder/tensor-split/MTP/vision logic entirely.

        No VRAM-budget check against the primary model here — Rule #1 says
        VRAM belongs to the primary model until proven otherwise, but
        enforcing that split precisely needs the primary's own live
        footprint, which serve()'s caller has already computed and this
        method doesn't receive. Left as -ngl 999 (fits the common case: a
        small embedding model website alongside a quantized generation
        model); GB_SYNAPSE_EMBED_NGL overrides for a tighter box.
        """
        import gb_synapse as gs
        import gb_cluster

        if not gs.engine_installed():
            raise RuntimeError("engine not built — run: greenboost synapse build-engine")

        internal_port = primary_port + 2000
        env = gb_cluster.shim_env(workload="llm", enabled=True)
        env["GREENBOOST_CLUSTER"] = "0"
        env["LD_LIBRARY_PATH"] = str(gs.ENGINE_DIR) + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        if os.environ.get("GB_SYNAPSE_SHIM", "0") != "1":
            for k in ("LD_PRELOAD", "GREENBOOST_ACTIVE"):
                env.pop(k, None)

        ctx = min(entry.ctx_length or 8192, 8192)  # embeddings never need a huge window
        ngl = os.environ.get("GB_SYNAPSE_EMBED_NGL", "999")
        gs.SLOT_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [str(gs.ENGINE_DIR / "llama-server"), "-m", entry.path,
               "--host", "127.0.0.1", "--port", str(internal_port),
               "--ctx-size", str(ctx), "--embeddings", "--pooling", "mean",
               "-np", "1", "--threads", str(gs._pcore_threads()),
               "-ngl", ngl, "--no-webui", "--no-mmap"]

        log = open(gs._run_log_path(entry.name + "_embed"), "ab")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        try:
            gs._wait_upstream_ready(entry, proc, internal_port, grace_s=60.0)
        except RuntimeError as e:
            raise RuntimeError(f"embeddings engine for '{entry.name}' failed to start: {e}") from e
        return proc, internal_port


class SynapseTorchBackend(EngineBackend):
    """GreenBoost's own torch-core inference engine (vendored gLLM, see
    synapse_engine/NOTICE) — the safetensors-checkpoint backend the
    gb-synapse unification is converging on. Routes every checkpoint gLLM
    can load through its own `api_server` (OpenAI-compatible, streaming
    included, same passthrough gb_synapse_api.py already uses for
    llama.cpp); `_torch_serve_mode()` decides per-checkpoint whether that's
    possible, and `serve()` falls back to `gb_synapse_fallback.py` (the
    same single-request server `TransformersBackend` uses when there's no
    torch venv at all) under this venv's own python when it isn't.

    Cluster PP (2026-07-24): unlike llama.cpp's --rpc (a from-scratch
    protocol GreenBoost wired itself), gLLM already ships its own
    multi-host launch mode (`--launch-mode master|slave` + `--ranks`) and
    real TCP rendezvous (`dist.init_process_group` with a routable
    --master-addr) — this backend just orchestrates it, the same shape
    ensure_feeder_rpc() already established for llama.cpp: SSH-launch the
    remote side, degrade to host-only on any failure.

    Deliberately PP-only, TP fixed at 1 — two reasons, both from reading
    the vendored source, not assumption: gLLM's frontend↔rank-0 scheduling
    sockets are hardcoded `ipc://` (Unix domain, can't cross machines) so
    rank 0 + the HTTP server must stay on the host regardless; and gLLM's
    TP fast-path all-reduce (CustomAllreduce) uses same-host-only CUDA IPC
    handles, unconditionally attempted whenever tp_size>1 — never
    constructed when tp_size==1, so this sidesteps it entirely rather than
    working around it. PP also just suits the hardware better: the feeder
    link is 1GbE (~0.11-0.12 GB/s, measured separately) and PP only
    exchanges activations at pipeline-stage boundaries, where TP would need
    a full all-reduce every layer."""
    name = "torch"

    def available(self) -> bool:
        return _torch_env_dir() is not None

    def can_serve(self, entry) -> bool:
        return entry.engine in ("torch", "vllm", "transformers", "gbquant")

    def serve_facts(self, entry) -> dict:
        _, _, facts = effective_vram_budget_mb()
        facts["quant_method"] = entry.quant_method
        facts["quant_below_floor"] = _quant_below_fp8_floor(entry)
        return facts

    def serve(self, entry, port, ctx=0, use_cluster=True, n_slots=-1, extra_args="", cuda_graph=None):
        import gb_synapse as gs
        import gb_cluster

        venv = _torch_env_dir()
        if not venv:
            raise RuntimeError(
                f"'{entry.name}' needs the synapse torch engine but no venv was found — "
                f"install it (Full Install default-on, or: sudo greenboost "
                f"install-synapse-engine) or set GB_SYNAPSE_TORCH_ENV."
            )
        venv_py = str(venv / "bin" / "python")
        internal_port = port + 1000

        # Same "torch" shim profile already validated for FLUX (ai-forge
        # CLAUDE.md RULE #1): real T2 (DDR) VRAM extension without CPU
        # compute participation.
        torch_cudart = str(_find_torch_venv_lib("nvidia/cu13/lib/libcudart.so.13")
                           or _find_torch_venv_lib("nvidia/cu12/lib/libcudart.so.12") or "")
        shim_on = os.environ.get("GB_SYNAPSE_TORCH_SHIM", "1") != "0"
        env = gb_cluster.shim_env(workload="torch", enabled=shim_on,
                                  cudart_path=torch_cudart or None)
        if shim_on:
            # Deferred-shim-init env var (greenboost_cuda_shim.c) — the
            # canonical name for the mechanism the retired VllmBackend set
            # as GREENBOOST_VLLM_COMPAT; the shim still accepts either name
            # (see the C shim's env check).
            env.setdefault("GREENBOOST_DEFER_INIT", "1")
        if gs.hf_token():
            env["HF_TOKEN"] = gs.hf_token()
        cuda_home = _cuda_home_for_torch()
        if cuda_home:
            env["CUDA_HOME"] = cuda_home

        facts = self.serve_facts(entry)
        mode, reason = _torch_serve_mode(entry)
        # Queryable via dataflux (dataflux_events kind=synapse_serve), not
        # just stdout — "why did this land on the fallback instead of the
        # real engine" was previously only answerable by reading the print()
        # below or grepping the log file directly.
        facts["torch_serve_mode"] = mode
        facts["torch_serve_reason"] = reason

        if mode == "fallback":
            print(f"  [gb-synapse] {entry.name}: using the single-request fallback "
                 f"server instead of the synapse torch engine — {reason}.", flush=True)
            # gb_synapse_fallback.py needs torch/aiohttp/transformers, which
            # _torch_serve_mode()'s "fallback" reasons don't call into
            # question (its only import probe is `import gllm, sgl_kernel` —
            # see _torch_engine_importable — so a venv that fails THAT can
            # still have a perfectly good torch/transformers install). Prefer
            # this venv's own python for that reason; only fall back to
            # sys.executable in the genuinely broken case where venv_py
            # itself isn't even a real executable (corrupted/partial venv).
            fallback_py = venv_py if Path(venv_py).is_file() else sys.executable
            cmd = [fallback_py, str(_REPO_DIR / "gb_synapse_fallback.py"),
                   "--model", _gbquant_model_source(entry),
                   "--served-model-name", entry.name,
                   "--quant", entry.quant or "fp8",
                   "--host", "127.0.0.1", "--port", str(internal_port)]
            facts["degraded"] = "fallback-single-request"
            online_feeders = []
        else:
            try:
                util = float(os.environ.get("GB_SYNAPSE_TORCH_KV_UTIL", "0.90"))
            except ValueError:
                util = 0.90
            util = max(0.3, min(0.95, util))

            model_path = _gbquant_model_source(entry)
            cmd = [venv_py, "-m", "gllm.entrypoints.api_server",
                   "--model-path", model_path,
                   "--host", "127.0.0.1", "--port", str(internal_port),
                   "--gpu-memory-util", f"{util:.2f}"]
            if ctx and ctx > 0:
                cmd += ["--model-max-length", str(ctx)]

            # gLLM's own --maxp (max prefill tokens per batch, used to size
            # profile_run()'s dummy warmup sequence) defaults to 8192
            # regardless of --model-max-length/the checkpoint's real
            # max_position_embeddings — confirmed live, 2026-07-26:
            # TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ (max_position_embeddings
            # =2048) crashed profile_run() with "size of tensor a (2048)
            # must match tensor b (8192)" before ever reaching a forward
            # pass, on every quant method (reproduces identically for bf16/
            # GPTQ/AWQ — the crash is in generic input-buffer sizing, not
            # anything quant-specific). Clamp --maxp to whatever's actually
            # known to be smaller than gLLM's default: the requested ctx, or
            # failing that the checkpoint's own ctx_length from its
            # config.json (safetensors_summary/gguf_summary already parse
            # this). Leave gLLM's default alone for large-context models —
            # this only ever shrinks the dummy prefill, never grows it.
            effective_max_len = ctx if (ctx and ctx > 0) else entry.ctx_length
            if effective_max_len and effective_max_len < 8192:
                cmd += ["--maxp", str(effective_max_len)]

            # CUDA graph capture reserves its own warmup buffers on top of
            # the KV cache — confirmed live 2026-07-16 (P3.9 bring-up) to
            # OOM on a 12 GB card with this engine's default KV-cache
            # sizing. Off by default until that sizing is revisited
            # (workflow/gb-synapse.md); GB_SYNAPSE_TORCH_CUDA_GRAPH=1
            # re-enables it process-wide, or pass cuda_graph=True to this
            # one serve() call (see EngineBackend.serve()'s docstring —
            # added 2026-07-28 so the synapse_serve MCP tool can opt in
            # per-call instead of needing an env var no MCP tool exposed).
            use_cuda_graph = (
                cuda_graph if cuda_graph is not None
                else os.environ.get("GB_SYNAPSE_TORCH_CUDA_GRAPH", "0") == "1"
            )
            if not use_cuda_graph:
                cmd += ["--disable-cuda-graph"]

            # Cluster PP — see class docstring for why PP-only/TP=1. Host is
            # always pp_rank 0 (gLLM's frontend↔rank-0 sockets are
            # hardcoded ipc://, can't move); each online feeder becomes the
            # next PP stage, same enumeration pattern as LlamaCppBackend's
            # RPC_PORT_BASE + i.
            #
            # All-or-nothing launch, deliberately: every rank's --pp commits
            # it to a specific world_size before dist.init_process_group()
            # runs, so a world_size that shrinks mid-launch (because feeder
            # 2 of 3 failed to start) would leave the survivors waiting for
            # a peer that's never coming — there's no such thing as a
            # partial rendezvous to fall back to. If any candidate fails,
            # kill whichever slaves already launched and drop to host-only
            # entirely rather than serve with a wrong world_size.
            online_feeders = []
            if use_cluster:
                candidates = [f for f in gb_cluster.feeders(probe=True) if f.online]
                if candidates:
                    master_ip = gb_cluster._local_ip_toward(candidates[0].ip)
                    master_port = gs.GLLM_MASTER_PORT_BASE
                    pp_size = 1 + len(candidates)
                    if not master_ip:
                        print(f"  [gb-synapse] couldn't resolve this host's own routable "
                              f"IP toward {candidates[0].ip} — skipping cluster PP", flush=True)
                    else:
                        launched = []
                        all_ok = True
                        for i, f in enumerate(candidates):
                            if gs.ensure_feeder_gllm_slave(f, master_ip, master_port,
                                                            pp_rank=i + 1, pp_size=pp_size,
                                                            model_path=model_path):
                                launched.append(f)
                            else:
                                all_ok = False
                                break
                        if all_ok:
                            online_feeders = launched
                            cmd += ["--launch-mode", "master",
                                    "--ranks", "0", "--pp", str(pp_size), "--tp", "1",
                                    "--master-addr", master_ip, "--master-port", str(master_port)]
                        else:
                            for f in launched:
                                gs.kill_feeder_gllm_slave(f)

        if extra_args:
            cmd += shlex.split(extra_args)

        log = open(gs._run_log_path(entry.name), "ab")
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)

        if online_feeders:
            # Cluster rendezvous needs a much larger readiness budget than a
            # single-node load (cold GPU CUDA init on the feeder + the NCCL
            # handshake itself) — gLLM's own dist.init_process_group() call
            # passes no timeout=, so a feeder that never joins just hangs
            # rather than failing fast; this is the GreenBoost-side bound
            # standing in for the one gLLM doesn't provide. On expiry, tear
            # down both sides and retry host-only rather than leaving a
            # half-joined NCCL world holding the GPU.
            try:
                ready = gs._wait_upstream_ready(entry, proc, internal_port,
                                                 grace_s=gs.GLLM_CLUSTER_READY_TIMEOUT_S)
            except RuntimeError:
                ready = False
            if not ready:
                print(f"  [gb-synapse] {entry.name}: cluster PP rendezvous did not "
                      f"complete within {gs.GLLM_CLUSTER_READY_TIMEOUT_S:.0f}s — "
                      f"tearing down and retrying host-only.", flush=True)
                proc.kill()
                for f in online_feeders:
                    gs.kill_feeder_gllm_slave(f)
                return self.serve(entry, port, ctx=ctx, use_cluster=False,
                                   n_slots=n_slots, extra_args=extra_args)

        state = gs._launch_proxy_and_record(entry, proc, port, internal_port,
                                            feeders=[f.ip for f in online_feeders],
                                            engine=self.name, **facts)
        _, _, budget_facts = effective_vram_budget_mb()
        _validate_placement(entry, util if mode == "gllm" else 0.0, budget_facts, self.name)
        return state


class TransformersBackend(EngineBackend):
    """Fallback when the synapse torch engine isn't installed at all (no
    torch venv found — see `_torch_env_dir()`): runs `gb_synapse_fallback.py`
    (in-process transformers + gb-quant quantize-to-fit) under `sys.executable`
    — the same interpreter running gb_synapse.py itself, since there's no
    dedicated venv to defer to here. gb_synapse_fallback.py's own imports
    (torch/aiohttp/transformers/gb_quant) resolve the same way whichever
    interpreter runs it, script-directory auto-insertion covers gb_quant/
    gb_init and the rest are ordinary pip deps expected on any host that runs
    gb_synapse.py at all. No continuous batching — single-request-at-a-time
    — so prefer the torch engine when it's available; this exists so
    ':fp8'-style models are never a hard dead end just because that venv
    isn't set up yet. SynapseTorchBackend's own fallback branch (torch venv
    present but gLLM/sgl_kernel import broken) uses this same script under
    the torch venv's python instead — see its serve()."""
    name = "transformers"

    def available(self) -> bool:
        return True   # always a valid fallback — no external binary needed

    def can_serve(self, entry) -> bool:
        return entry.engine in ("transformers", "gbquant")

    def serve(self, entry, port, ctx=0, use_cluster=True, n_slots=-1, extra_args="", cuda_graph=None):
        import gb_synapse as gs

        internal_port = port + 1000
        env = os.environ.copy()
        env["GREENBOOST_ACTIVE"] = "1"
        if gs.hf_token():
            env["HF_TOKEN"] = gs.hf_token()
        cmd = [sys.executable, str(_REPO_DIR / "gb_synapse_fallback.py"),
               "--model", _gbquant_model_source(entry), "--served-model-name", entry.name,
               "--quant", entry.quant or "fp8",
               "--host", "127.0.0.1", "--port", str(internal_port)]
        log = open(gs._run_log_path(entry.name), "ab")
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        return gs._launch_proxy_and_record(entry, proc, port, internal_port, engine=self.name)


class DiffusersBackend(EngineBackend):
    """HF diffusers image-generation pipelines (FLUX, SDXL, LTX, ...), served
    through gb_diffusion_server.py behind the same :11434 proxy — the proxy's
    raw /v1/{path:.*} passthrough needs zero changes for
    /v1/images/generations. ctx/n_slots are meaningless for image gen and are
    ignored."""
    name = "diffusers"

    def available(self) -> bool:
        try:
            import diffusers  # noqa: F401
            return True
        except ImportError:
            return False

    def can_serve(self, entry) -> bool:
        return entry.engine == "diffusers"

    def serve(self, entry, port, ctx=0, use_cluster=True, n_slots=-1, extra_args="", cuda_graph=None):
        import gb_synapse as gs
        import gb_cluster

        internal_port = port + 1000
        # Same "torch" shim posture as SynapseTorchBackend/TransformersBackend:
        # torch workload, no cluster fabric, real T2.
        env = gb_cluster.shim_env(workload="torch", enabled=True)
        if gs.hf_token():
            env["HF_TOKEN"] = gs.hf_token()
        cmd = [sys.executable, str(_REPO_DIR / "gb_diffusion_server.py"),
               "--model", _gbquant_model_source(entry), "--served-model-name", entry.name,
               "--quant", entry.quant or "fp8",
               "--host", "127.0.0.1", "--port", str(internal_port)]
        log = open(gs._run_log_path(entry.name), "ab")
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        return gs._launch_proxy_and_record(entry, proc, port, internal_port, engine=self.name)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_backend(entry) -> EngineBackend:
    """Legacy `entry.engine` values "gbquant"/"vllm"/"transformers" (all
    normalized to "torch" on load by gb_synapse._load_manifest — see that
    function's docstring) plus the current "torch" value itself all route
    to SynapseTorchBackend by default — live-verified end-to-end (Phase 3/
    P3.9 bring-up, 2026-07-16: the vendored gLLM engine, with its local
    SDPA-fallback patch, serves bf16 safetensors checkpoints correctly on
    this hardware). Falls back to TransformersBackend only if the torch
    engine isn't installed (VllmBackend retired — Phase 6). "diffusers" is
    explicit — no silent fallback. Anything else (including the "llama.cpp"
    default) gets the llama.cpp engine."""
    if entry.engine in ("torch", "vllm", "gbquant", "transformers"):
        t = SynapseTorchBackend()
        return t if t.available() else TransformersBackend()
    if entry.engine == "diffusers":
        return DiffusersBackend()
    return LlamaCppBackend()

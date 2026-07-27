#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""gb_api.py — GreenBoost's public facade for external consumers (ai-forge,
or any other pipeline that wants GPU-memory-tiering, quantization, cluster,
and inference-serving primitives without importing GreenBoost's internal
modules directly).

Why a facade at all, given every function here is a thin wrapper: the
friction it removes is DISCOVERY + a single stable import, not missing
logic. ai-forge alone has 30+ `sys.path.insert`/`PYTHONPATH` sites reaching
into 8 different gb_*.py modules, and duplicates real logic GreenBoost
already has (a 70-line cudart resolver, VRAM/contention nvidia-smi
shelling, a `pool_brief` regex scrape, an 8-attempt endpoint-readiness
backoff, a systemd-run wrapper with 4 hard-won env/security fixes) because
there was no single, documented entry point to reach it through instead.

This module deliberately implements NOTHING itself — every function below
lazily imports and delegates to the module that actually owns the data or
decision (gb_cluster, gb_tiering, gb_monitor, gb_synapse, gb_actuation,
gb_dataflux), so dataflux/MCP coverage checks and existing tests for those
modules stay meaningful. `gb_tiering.py`'s own docstring already documents
this exact facade relationship for the tiering subsystem — this is the
same pattern applied one level up, not a new architecture.

No torch import at module level (ground rule: importing the backend must
never touch the GPU) — every function imports its target module inside its
own body.

Usage:
    import greenboost_bootstrap   # puts the GreenBoost root on sys.path
    import gb_api

    env = gb_api.workload_env("diffusion", python=sys.executable)
    pool = gb_api.t2_pool()
    gb_api.wait_for_vram(free_mb=4000, timeout_s=30)
"""
from __future__ import annotations


def workload_env(workload: str = "diffusion", *, enabled: bool = True,
                 python: "str | None" = None,
                 extra: "dict[str, str] | None" = None) -> dict:
    """Subprocess env overlay for a GreenBoost-accelerated workload , the
    ONE call that replaces ai-forge's forge/gpu.py 4-key re-layer (
    GREENBOOST_KV_COMPRESS, GREENBOOST_REPORT_PHYSICAL_VRAM,
    GREENBOOST_T2_POOL_MB, GREENBOOST_CUDART_PATH) plus its 12-key
    shimless fallback profile.

    `python`, if given, auto-resolves GREENBOOST_CUDART_PATH via
    cudart_for() so the caller doesn't have to discover its own venv's
    cudart separately (gb_cluster.shim_env()'s own docstring names this
    exact gap: "the consumer discovers it"). `extra` merges additional
    env vars on top (applied last, so a caller can still override
    anything this profile sets)."""
    import gb_cluster
    cudart_path = cudart_for(python) if python else None
    env = gb_cluster.shim_env(workload=workload, enabled=enabled, cudart_path=cudart_path)
    if extra:
        env.update(extra)
    return env


def cudart_for(python_exe: str) -> "str | None":
    """The CUDA runtime shared library (libcudart.so) a given Python
    interpreter's own environment would load. See gb_cluster.cudart_for
    for the search order and the production incidents this replaces."""
    import gb_cluster
    return gb_cluster.cudart_for(python_exe)


def t2_pool() -> dict:
    """{total_mb, allocated_mb, available_mb, pressure} for the T2 DDR
    pool — replaces regex-scraping /sys/class/greenboost/greenboost/
    pool_brief directly (ai-forge did this in two separate files)."""
    import gb_tiering
    return gb_tiering.t2_pool()


def gpu_state() -> dict:
    """{used_mb, total_mb, free_mb, util_pct} for the local GPU — replaces
    3 separate nvidia-smi subprocess calls ai-forge's forge/gpu.py hand-
    rolled for the same data."""
    import gb_monitor
    return gb_monitor.gpu_state()


def wait_for_vram(free_mb: float, timeout_s: float = 60.0, poll_s: float = 1.0) -> bool:
    """Block until at least `free_mb` VRAM is free or timeout_s elapses.
    Replaces ai-forge's forge/gpu.py `guard()` polling loop (added after
    free-VRAM-alone missed a real GPU-contention case where another
    process held ~99% utilization with memory nominally free)."""
    import gb_monitor
    return gb_monitor.wait_for_vram(free_mb, timeout_s=timeout_s, poll_s=poll_s)


def wait_ready(url: str, *, path: str = "/health", timeout_s: float = 120.0,
              attempts: "int | None" = None) -> bool:
    """Poll `url + path` until HTTP 200 or timeout_s elapses — a generic
    endpoint-readiness gate for any OpenAI/Ollama-compatible server, not
    just gb-synapse's own. Replaces the ad hoc readiness polling ai-forge
    hand-rolls per endpoint (forge/gb_models.py's 8-attempt/60s backoff,
    forge/ocr_vl.py's own health-poll loop)."""
    import gb_synapse
    return gb_synapse.wait_ready(url, path=path, timeout_s=timeout_s, attempts=attempts)


def pull_model(name: str, progress=None):
    """Pull an HF-hosted GGUF/safetensors model into gb-synapse's model
    store. `progress`, if given, is called with ("start", name) and
    ("done", name) — see gb_synapse.pull_model for the honest limits of
    what "progress" means here (no byte-level percentage yet)."""
    import gb_synapse
    return gb_synapse.pull_model(name, progress=progress)


def run_capped(argv: list[str], *, mem_max_mb: "float | None" = None,
              env: "dict[str, str] | None" = None, cwd: "str | None" = None,
              secrets_file: bool = True, timeout_s: int = 3600) -> dict:
    """Run `argv` under a memory-capped `systemd-run --user` unit, with the
    subprocess env delivered correctly and secrets kept out of
    /proc/pid/cmdline. See gb_actuation.run_capped for the 4 hard-won
    facts this encodes (replaces ai-forge's forge/runners/longlive.py
    capping wrapper, whose own history includes a real incident where the
    whole env silently vanished and produced a run of misdiagnosed
    torch.OutOfMemoryErrors)."""
    import gb_actuation
    return gb_actuation.run_capped(argv, mem_max_mb=mem_max_mb, env=env, cwd=cwd,
                                   secrets_file=secrets_file, timeout_s=timeout_s)


def stage(name: str, **fields):
    """Context manager: emit a `stage_profile` dataflux event on success
    (status=ok) or before re-raising on failure (status=error), both with
    real duration_s. Replaces ai-forge's 4 independently hand-built
    versions of this exact pattern (build_jobs_exams.py's df_emit/_vlm,
    run_all_exams.py's build_dir, forge/dataflux.py, forge/seedlog.py) with
    one shared implementation.

    Usage: `with gb_api.stage("forge:image", label="gen_art"): ...`"""
    import gb_dataflux
    return gb_dataflux.stage(name, **fields)


def serve_gguf(model_path: str, port: int, mmproj: "str | None" = None, ctx: int = 0) -> dict:
    """Serve an arbitrary GGUF file directly via llama-server on `port` ,
    no gb-synapse Ollama proxy, no manifest resolution. Replaces ai-forge's
    forge/ocr_vl.py, which reimplements gb_synapse._resolve_engine_dir by
    its own docstring's admission, then hand-launches + health-polls
    llama-server itself for its OCR-VL model."""
    import gb_synapse
    return gb_synapse.serve_gguf(model_path, port, mmproj=mmproj, ctx=ctx)


def endpoints() -> dict:
    """The known inference endpoints this box exposes
    (/etc/greenboost/inference.env) — replaces ai-forge's forge/config.py
    hand-parsing the same file for its 4 independent endpoints
    (gb-synapse :11435, OCR-VL :8081, OCR-GPU :8082, AI-tools :8083)."""
    import gb_synapse
    return gb_synapse.endpoints()


def cluster() -> "object":
    """The gb_cluster module itself, for callers that need something this
    facade doesn't wrap yet (e.g. feeders()/cluster_map()). Public,
    replacing the private `forge.gpu._gb_cluster_module` coupling
    studio/server/cluster_sched.py relied on."""
    import gb_cluster
    return gb_cluster


def ssh_opts(connect_timeout: int = 10, compress: bool = False) -> list:
    """Public alias for gb_cluster's feeder SSH option list — replaces the
    private `gb_cluster._ssh_opts()` coupling
    tools/cluster_runner/express_gen.py relied on."""
    import gb_cluster
    return gb_cluster._ssh_opts(connect_timeout=connect_timeout, compress=compress)

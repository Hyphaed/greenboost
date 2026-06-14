#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_init.py — GreenBoost Python layer bootstrap.

Single-import wiring module: idempotent, safe to import multiple times.
All downstream GreenBoost modules (gb_quant, gb_diffusion_orch, gb_llm,
gb_attn) import this at their top level when GREENBOOST_ACTIVE=1.

What this does on first import when GREENBOOST_ACTIVE=1:
  1. Enforces PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False (mandatory).
  2. Monkey-patches torch.cuda.empty_cache → no-op (DynamicVRAM contract).
  3. Creates the module-level TelemetryManager singleton (device 0) and
     starts its background thread.  For cluster setups (2+ GPUs detected),
     also creates ClusterTelemetryManager.
  4. Installs an ECC DBE error callback that prints a loud warning to stderr
     (double-bit errors = uncorrectable hardware memory corruption).
  5. Exposes the gb_stream_sched, gb_model_tier, gb_mem_pool singletons via
     get_stream_sched(), get_tier_manager(), get_mem_pools() accessors.
  6. Registers an atexit handler that cleanly stops telemetry on exit.

When GREENBOOST_ACTIVE != "1": import is a no-op (zero overhead for non-GB
venvs, unit tests, CI environments).

Usage in downstream modules:
    # At the top of gb_quant.py, gb_llm.py, etc.:
    import gb_init  # activates all layers when GREENBOOST_ACTIVE=1

    # Read the telemetry snapshot (always fast, no I/O):
    m = gb_init.telemetry.snapshot()
    if m.should_demote:
        gb_init.get_tier_manager().auto_evict(m)

    # Use named CUDA streams:
    with gb_init.gs.on("quant"):
        quantize_layer(...)

    # Manually trigger pre-inference sanity check:
    gb_init.pre_inference_check()
"""
from __future__ import annotations

import atexit
import os
import sys

# ── Activation guard ─────────────────────────────────────────────────────────
ACTIVE = os.environ.get("GREENBOOST_ACTIVE") == "1"

# Module-level singletons (None until _bootstrap() runs)
telemetry       = None   # TelemetryManager (device 0)
cluster_tel     = None   # ClusterTelemetryManager (multi-GPU, or None)
gs              = None   # gb_stream_sched module
_tier_manager   = None
_mem_pools      = None

_initialized    = False
_torch_patched  = False


def _bootstrap():
    """Run once on first import when GREENBOOST_ACTIVE=1."""
    global telemetry, cluster_tel, gs, _tier_manager, _mem_pools
    global _initialized, _torch_patched

    if _initialized:
        return
    _initialized = True

    # 1. Enforce expandable_segments:False — GreenBoost DynamicVRAM requirement.
    # Setting this after torch is imported has no effect (allocator is already
    # initialized), but it documents intent and guards env for child processes.
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            ("expandable_segments:False," + alloc_conf).rstrip(",")
        )

    # 2. Patch torch.cuda.empty_cache → no-op.
    # Under GreenBoost DynamicVRAM, calling empty_cache raises
    # "CUDA error: invalid argument" because the allocator is controlled by
    # the kernel module, not by the PyTorch caching allocator.  We replace it
    # globally so every call site (gemlite, hqq, diffusers callbacks, user
    # scripts) is covered without per-callsite changes.
    try:
        import torch
        if not _torch_patched:
            _real_empty_cache = torch.cuda.empty_cache.__func__ \
                if hasattr(torch.cuda.empty_cache, "__func__") \
                else torch.cuda.empty_cache
            torch.cuda.empty_cache = lambda: None
            _torch_patched = True
    except Exception:
        pass

    # 3. Start telemetry singleton (device 0).
    try:
        from gb_telemetry import TelemetryManager, ClusterTelemetryManager, \
            _detect_device_count, _detect_feeder_count
        # 500ms poll: aggressive enough for tier decisions, light enough not to
        # interfere with Triton kernel launches or DCGM embedded-mode overhead.
        telemetry = TelemetryManager(device=0, poll_ms=500, enable_dcgm=True)
        telemetry.start()

        # Cluster mode: start cluster manager when 2+ GPUs are visible,
        # counting both local GPUs and connected feeder GPUs from metrics.json.
        n_dev = _detect_device_count()
        n_feeders = _detect_feeder_count()
        if n_dev + n_feeders > 1:
            cluster_tel = ClusterTelemetryManager(poll_ms=500, enable_dcgm=True)
            cluster_tel.start()
    except Exception as exc:
        print(f"[gb_init] telemetry unavailable: {exc}", file=sys.stderr)

    # 4. ECC DBE callback — loud stderr warning on hardware memory errors.
    # Registered on the local manager AND each feeder manager so both local
    # and remote double-bit errors are caught (Phase 3d: feeder ECC guard).
    if telemetry is not None:
        def _ecc_guard(m):
            if m.ecc_dbe_volatile > 0:
                where = "local" if m.device == 0 else f"feeder GPU {m.device}"
                print(
                    f"\n[gb_init] *** ECC DOUBLE-BIT ERROR on {where} "
                    f"(device {m.device}): "
                    f"{m.ecc_dbe_volatile} uncorrectable error(s) detected. "
                    f"Hardware memory may be corrupted. "
                    f"Aggregate DBE: {m.ecc_dbe_aggregate}. ***\n",
                    file=sys.stderr, flush=True,
                )
        telemetry.add_callback(_ecc_guard)
        # Also attach to each feeder manager created by the cluster manager
        if cluster_tel is not None:
            for _mgr in cluster_tel._managers[1:]:  # skip local (already covered)
                _mgr.add_callback(_ecc_guard)

    # 5. Stream scheduler singleton (lazy: only needs torch, no daemon).
    try:
        import gb_stream_sched as _gs
        gs = _gs
    except Exception as exc:
        print(f"[gb_init] stream scheduler unavailable: {exc}", file=sys.stderr)

    # 6. ModelTierManager singleton.
    try:
        from gb_model_tier import ModelTierManager
        _tier_manager = ModelTierManager(hbm_headroom_mb=1500)
    except PermissionError:
        # Expected when running as non-root — model_pages is a privileged path.
        pass
    except Exception as exc:
        print(f"[gb_init] tier manager unavailable: {exc}", file=sys.stderr)

    # 7. MemPoolManager singleton.
    try:
        from gb_mem_pool import MemPoolManager
        _mem_pools = MemPoolManager()
    except Exception as exc:
        print(f"[gb_init] mem pool manager unavailable: {exc}", file=sys.stderr)

    # 8. Clean atexit shutdown.
    atexit.register(_shutdown)

    print("[gb_init] GreenBoost Python layers active", flush=True)


def _shutdown():
    """Stop background threads on process exit."""
    global telemetry, cluster_tel
    try:
        if telemetry is not None:
            telemetry.stop()
        if cluster_tel is not None:
            cluster_tel.stop()
    except Exception:
        pass


# ── Public accessors ──────────────────────────────────────────────────────────

def get_telemetry():
    """Return the TelemetryManager singleton (device 0), or None if unavailable."""
    return telemetry


def get_cluster_telemetry():
    """Return the ClusterTelemetryManager singleton, or None if single-GPU."""
    return cluster_tel


def get_stream_sched():
    """Return the gb_stream_sched module, or None if unavailable."""
    return gs


def get_tier_manager():
    """Return the ModelTierManager singleton, or None if unavailable."""
    return _tier_manager


def get_mem_pools():
    """Return the MemPoolManager singleton, or None if unavailable."""
    return _mem_pools


def snapshot():
    """Return the latest cached GpuMetrics, or None if telemetry unavailable."""
    if telemetry is not None:
        return telemetry.snapshot()
    return None


def pre_inference_check(run_dcgm_diag: bool = False) -> bool:
    """
    Run a pre-inference sanity check.  Returns True if all checks pass.

    Checks performed:
      - ECC DBE errors (fails if any uncorrectable errors present)
      - DCGM health check (uses cached snapshot; no extra I/O)
      - VRAM headroom (warns if < 1 GB free)
      - Optional: DCGM diagnostic level 1 (set run_dcgm_diag=True)

    Safe to call before any generation pipeline; takes <1 ms (reads cache).
    """
    ok = True
    m = snapshot()
    if m is None:
        return True   # Telemetry not available — proceed optimistically

    if m.ecc_dbe_volatile > 0:
        print(
            f"[gb_init] PRE-INFERENCE FAIL: ECC DBE errors={m.ecc_dbe_volatile} "
            f"on GPU {m.device}. Hardware memory risk.",
            file=sys.stderr, flush=True,
        )
        ok = False

    if not m.health_ok:
        print(
            f"[gb_init] PRE-INFERENCE WARN: DCGM health not OK: {m.health_summary}",
            file=sys.stderr, flush=True,
        )
        # Health warn doesn't block — DCGM may report false positives on
        # consumer GPUs for subsystems like NVLink that aren't present.

    if m.fb_free_mb < 1024:
        print(
            f"[gb_init] PRE-INFERENCE WARN: low VRAM headroom "
            f"({m.fb_free_mb} MB free). Quantize-to-fit recommended.",
            file=sys.stderr, flush=True,
        )

    if run_dcgm_diag and telemetry is not None:
        result = telemetry.run_diag(level=1)
        print(f"[gb_init] {result}", flush=True)

    return ok


# ── Auto-activate ─────────────────────────────────────────────────────────────
if ACTIVE:
    _bootstrap()

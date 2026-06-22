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
_orchestrator   = None   # ReactiveOrchestrator (process mode)

_initialized    = False
_torch_patched  = False


def _bootstrap():
    """Run once on first import when GREENBOOST_ACTIVE=1."""
    global telemetry, cluster_tel, gs, _tier_manager, _mem_pools, _orchestrator
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
    # Primary fix: PR-FREE-1 in greenboost_cuda_shim.c tolerates
    # cudaErrorInvalidValue (=1) in cudaFree fallthrough paths (CUDA context
    # teardown race during DEEP_IDLE reclaim), so the error no longer surfaces
    # to Python callers in most cases.
    # Defense-in-depth: keep the no-op for call sites that still reach
    # empty_cache from gemlite, hqq, diffusers callbacks, or user scripts
    # that expect it to be a reliable flushing mechanism.  Removing this
    # patch would restore the original behavior if the shim fix proves
    # sufficient in practice — test before removing.
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

    # 9. Reactive orchestrator (process mode — only writes control-file hints;
    #    kernel/sysfs levers belong to gb_supervisor in supervisor mode).
    #    Feeds on the same telemetry add_callback bus used by _ecc_guard above.
    try:
        from gb_orchestrator import ReactiveOrchestrator
        _orchestrator = ReactiveOrchestrator(
            mode="process",
            tier_manager=_tier_manager,
            cluster_tel=cluster_tel,   # B3: cluster-aware Loop E
            tel_manager=telemetry,     # N: adaptive poll rate
        )
        if telemetry is not None:
            telemetry.add_callback(_orchestrator.on_metrics)
    except Exception as exc:
        print(f"[gb_init] orchestrator unavailable: {exc}", file=sys.stderr)

    print("[gb_init] GreenBoost Python layers active", flush=True)


def _shutdown():
    """Stop background threads on process exit."""
    global telemetry, cluster_tel
    try:
        if _orchestrator is not None:
            _orchestrator.stop()
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


def get_orchestrator():
    """Return the ReactiveOrchestrator singleton (process mode), or None."""
    return _orchestrator


def make_layer_prefetcher(model, keep_resident: int = 2, lookahead: int = 1):
    """Construct a gb_prefetch.LayerPrefetcher for `model`, reusing the
    shared ModelTierManager singleton (get_tier_manager()) instead of each
    caller creating its own. One LayerPrefetcher per loaded model - not a
    singleton itself, unlike the other accessors here, since a process may
    load more than one dense model. Returns None if gb_prefetch or the
    shared tier manager is unavailable."""
    try:
        from gb_prefetch import LayerPrefetcher
    except Exception as exc:
        print(f"[gb_init] layer prefetcher unavailable: {exc}", file=sys.stderr)
        return None
    return LayerPrefetcher(model, tm=_tier_manager,
                            keep_resident=keep_resident, lookahead=lookahead)


def auto_budget_gb(headroom: float = 0.92) -> float:
    """
    Unified free-VRAM budget estimate in GB.

    Telemetry first: fb_free_mb reflects GreenBoost virtual memory (T1+T2+T3)
    so it correctly accounts for the expanded address space.
    Torch fallback: raw cuMemGetInfo — real VRAM only, misses T2/T3 overflow.

    Downstream callers (gb_quant, gb_llm) should call this instead of querying
    torch.cuda.mem_get_info() directly so all budget decisions use the same view.
    """
    m = snapshot()
    if m is not None and m.fb_free_mb > 0:
        return m.fb_free_mb / 1024.0 * headroom
    try:
        import torch
        free_b, _ = torch.cuda.mem_get_info()
        return free_b / (1 << 30) * headroom
    except Exception:
        return 0.0


def debug_dump() -> dict:
    """
    Full state snapshot for LLM diagnostics and `vitals --json`.
    Merges telemetry snapshot, enriched pool info, shim_stats file,
    and orchestrator decisions (per-signal raw/value/state/last_decision).
    """
    from pathlib import Path
    result: dict = {
        "active":      ACTIVE,
        "initialized": _initialized,
        "auto_budget_gb": auto_budget_gb(),
    }
    m = snapshot()
    if m is not None:
        result["metrics"] = {
            "fb_used_mb":       m.fb_used_mb,
            "fb_free_mb":       m.fb_free_mb,
            "fb_total_mb":      m.fb_total_mb,
            "fb_used_pct":      round(m.fb_used_pct, 1),
            "temp_c":           round(m.temp_c, 1),
            "power_w":          round(m.power_w, 1),
            "gpu_util_pct":     round(m.gpu_util_pct, 1),
            "ecc_dbe_volatile": m.ecc_dbe_volatile,
            "kv_pressure":      round(m.kv_pressure, 3),
            "kv_spilled":       m.kv_spilled,
            "health_ok":        m.health_ok,
        }
        if m.gb is not None:
            result["pool"] = {
                "t1_vram_mb":        m.gb.t1_vram_mb,
                "t2_allocated_mb":   m.gb.t2_allocated_mb,
                "t2_available_mb":   m.gb.t2_available_mb,
                "kv_reserve_mb":     m.gb.kv_reserve_mb,
                "kv_used_mb":        m.gb.kv_used_mb,
                "kv_t2_mb":          m.gb.kv_t2_mb,
                "safety_reserve_mb": m.gb.safety_reserve_mb,
                "gaming_mode":       m.gb.gaming_mode,
                "t2_pressure":       m.gb.t2_pressure,
                "t3_pressure":       m.gb.t3_pressure,
                "oom_active":        m.gb.oom_active,
                "phase_reset_seq":   m.gb.phase_reset_seq,
            }
    try:
        stats_path = Path("/run/greenboost/shim_stats")
        if stats_path.exists():
            result["shim_stats"] = dict(
                line.split("=", 1) for line in stats_path.read_text().splitlines()
                if "=" in line
            )
    except Exception:
        pass
    if _orchestrator is not None:
        result["orchestrator"] = _orchestrator.dump()
    return result


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

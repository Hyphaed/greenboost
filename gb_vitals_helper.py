#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_vitals_helper.py , unified GPU + GreenBoost flow metrics helper.

Called by greenboost_setup.sh vitals/cluster commands to replace
nvidia-smi subprocess forks with in-process pynvml calls, and to surface
the shim's live allocation flow state for debugging.

Modes (--json, --flow, --dcgm, default KEY=VALUE):
  default   KEY=VALUE pairs (bash IFS='=' read parseable; unknown keys ignored)
  --json    JSON object merging GPU metrics + shim flow state; LLM/human debug
  --flow    compact one-line shim flow summary (phase, path, tier stats)
  --dcgm    append DCGM health / power-instant / NVLink BW to KEY=VALUE output
            (uses a 30s file cache at /run/greenboost/dcgm_health.json)

Sources (consumed in order; degrade gracefully when unavailable):
  1. gb_nvml.get_nvml()             , GPU hardware metrics via shared singleton
  2. /run/greenboost/shim_stats     , shim flow state (phase, tiers, ws_reserve)
  3. /sys/module/greenboost/…       , gaming_mode sysfs
  4. /var/lib/greenboost/vram_pressure , supervisor pressure state
  5. /run/greenboost/orch_state.json , orchestrator decisions (written by daemon)
  6. GB_IOCTL_GET_INFO (ctypes)     , kernel module pool state (optional)

Exit 0: success (pynvml available)
Exit 1: pynvml unavailable , caller should fall back to nvidia-smi
"""
from __future__ import annotations
import json
import os
import sys
import time
from typing import Any

_DCGM_CACHE = "/run/greenboost/dcgm_health.json"
_DCGM_TTL   = 30  # seconds , reuse cached result within this window


def _probe_dcgm(device: int = 0) -> dict:
    """Run one DCGM health + FI sample. Returns dict; empty on any error.

    Uses a file cache at _DCGM_CACHE so that the heavy dcgmStartEmbedded()
    call runs at most once per _DCGM_TTL seconds even when vitals refreshes
    every 5 s.
    """
    # Cache hit?
    try:
        st = os.stat(_DCGM_CACHE)
        if (time.time() - st.st_mtime) < _DCGM_TTL:
            with open(_DCGM_CACHE) as fh:
                return json.load(fh)
    except Exception:
        pass

    result: dict = {}
    try:
        from gb_telemetry import DCGMProvider  # type: ignore
        prov = DCGMProvider(device)
        from gb_telemetry import GpuMetrics   # type: ignore
        m = GpuMetrics()
        prov.sample(m)
        result = {
            "health_ok":       m.health_ok,
            "health_summary":  m.health_summary,
            "power_instant_w": round(getattr(m, "power_instant_w", 0.0) or 0.0, 1),
            "nvlink_bw_mb_s":  round(getattr(m, "nvlink_bw_mb_s", 0.0) or 0.0, 1),
        }
        try:
            prov.close()
        except Exception:
            pass
    except Exception:
        pass  # DCGM not available , caller uses empty dict

    # Write cache (best effort; may fail in containers)
    try:
        os.makedirs(os.path.dirname(_DCGM_CACHE), exist_ok=True)
        with open(_DCGM_CACHE, "w") as fh:
            json.dump(result, fh)
    except Exception:
        pass

    return result


# ── sysfs helpers ─────────────────────────────────────────────────────────────

def _sysfs_str(path: str, default: str = "") -> str:
    try:
        return open(path).read().strip()
    except Exception:
        return default


def _sysfs_int(path: str, default: int = 0) -> int:
    try:
        return int(open(path).read().strip())
    except Exception:
        return default


# ── shim stats reader ─────────────────────────────────────────────────────────

def sample_shim_flow(stats_path: str = "/run/greenboost/shim_stats") -> dict[str, str]:
    """
    Parse /run/greenboost/shim_stats (KEY=VALUE) into a dict.
    Returns empty dict if the file is absent or unreadable (shim not active).
    Keys are prefixed with SHIM_ in the KEY=VALUE output for namespace clarity.
    """
    try:
        with open(stats_path) as f:
            content = f.read()
    except Exception:
        return {}
    try:
        from gb_monitor import parse_shim_stats  # canonical KEY=VALUE grammar
        return parse_shim_stats(content)
    except Exception:
        result: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            if "=" in line:
                k, _, v = line.partition("=")
                result[k.strip()] = v.strip()
        return result


# ── orchestrator state reader ─────────────────────────────────────────────────

def _read_orch_state(path: str = "/run/greenboost/orch_state.json") -> dict:
    """
    Read the orchestrator decision state file (written by ReactiveOrchestrator
    on every decision; short-lived vitals helper can't hold the live singleton).
    Returns empty dict if absent (daemon not running or never triggered).
    """
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    mode = "kv"
    args = sys.argv[1:]
    want_dcgm = "--dcgm" in args
    if "--json" in args:
        mode = "json"
    elif "--flow" in args:
        mode = "flow"

    # ── GPU metrics via gb_nvml singleton (fixes missing nvmlShutdown) ────────
    try:
        from gb_nvml import get_nvml
        nvml = get_nvml(0)
    except Exception:
        sys.exit(1)

    if not nvml.ok:
        sys.exit(1)

    name              = nvml.device_name()
    used_mb, free_mb, total_mb, pct_f = nvml.mem()
    pct               = int(pct_f)
    gpu_util, mem_util = nvml.util()
    gpu_util          = int(gpu_util)
    mem_util          = int(mem_util)
    temp_c            = int(nvml.temp_c())
    power_w           = nvml.power_w()
    power_limit_w     = nvml.power_limit_w()
    sm_clock, mem_clock = nvml.clocks_mhz()
    ecc_dbe           = nvml.ecc_dbe_volatile()
    ecc_dbe_agg       = nvml.ecc_dbe_aggregate()
    ecc_sbe           = nvml.ecc_sbe_volatile()
    pcie_tx, pcie_rx  = nvml.pcie_mb_s()

    # ── GreenBoost sources (degrade gracefully) ──────────────────────────────
    gaming_mode = _sysfs_int("/sys/module/greenboost/parameters/gaming_mode", 0)

    pressure_state = ""
    try:
        for line in open("/var/lib/greenboost/vram_pressure"):
            if line.startswith("state="):
                pressure_state = line.split("=", 1)[1].strip()
                break
    except Exception:
        pass

    # Shim flow state , zero-cost when shim is absent
    flow = sample_shim_flow()

    # Orchestrator decisions , read from state file written by daemon
    orch = _read_orch_state()

    # ── Output modes ─────────────────────────────────────────────────────────

    if mode == "flow":
        # Compact one-liner for humans: phase, active path, T1/T2 current MB
        phase      = flow.get("phase", "?")
        apath      = flow.get("active_path", "?")
        t1_cur     = flow.get("tier_t1_local_cur_mb", "?")
        t2_cur     = flow.get("tier_t2_local_cur_mb", "?")
        ws_eff     = flow.get("workstation_reserve_eff_mb", "?")
        print(f"phase={phase} path={apath} t1={t1_cur}MB t2={t2_cur}MB ws_rsv={ws_eff}MB "
              f"gpu={gpu_util}% {used_mb}/{total_mb}MB {temp_c}°C")
        sys.exit(0)

    if mode == "json":
        obj: dict[str, Any] = {
            "gpu_name":        name,
            "vram_used_mb":    used_mb,
            "vram_total_mb":   total_mb,
            "vram_free_mb":    free_mb,
            "vram_pct":        pct,
            "gpu_util_pct":    gpu_util,
            "mem_util_pct":    mem_util,
            "temp_c":          temp_c,
            "power_w":         round(power_w, 1),
            "power_limit_w":   round(power_limit_w, 1),
            "sm_clock_mhz":    sm_clock,
            "mem_clock_mhz":   mem_clock,
            "ecc_dbe":         ecc_dbe,
            "ecc_dbe_agg":     ecc_dbe_agg,
            "gaming_mode":     gaming_mode,
            "pressure_state":  pressure_state,
            # Shim flow fields (empty dict when shim inactive , no KeyError)
            "shim_flow":       flow,
        }
        # Orchestrator decisions block , surfaces per-signal state + recent actions
        if orch:
            decisions_summary = []
            for sig in orch.get("signals", []):
                decisions_summary.append({
                    "signal":        sig.get("name", ""),
                    "value":         sig.get("value"),
                    "state":         sig.get("state", ""),
                    "last_decision": sig.get("last_decision", ""),
                    "in_hysteresis": sig.get("in_hysteresis", False),
                })
            obj["decisions"] = {
                "mode":            orch.get("mode", ""),
                "actuate":         orch.get("actuate", False),
                "ecc_degraded":    orch.get("ecc_degraded", False),
                "ws_above":        orch.get("ws_above", False),
                "ws_reserve_mb":   orch.get("ws_reserve_mb", 0),
                "signals":         decisions_summary,
                "recent":          orch.get("recent_decisions", []),
            }
        print(json.dumps(obj, indent=2))
        sys.exit(0)

    # Default: KEY=VALUE (bash-parseable; query_gpu_vram ignores unknown keys)
    lines = [
        f"GPU_NAME={name}",
        f"GPU_VRAM_USED_MB={used_mb}",
        f"GPU_VRAM_TOTAL_MB={total_mb}",
        f"GPU_VRAM_FREE_MB={free_mb}",
        f"GPU_VRAM_PCT={pct}",
        f"GPU_UTIL_PCT={gpu_util}",
        f"GPU_MEM_UTIL_PCT={mem_util}",
        f"GPU_TEMP_C={temp_c}",
        f"GPU_POWER_W={power_w:.1f}",
        f"GPU_POWER_LIMIT_W={power_limit_w:.1f}",
        f"GPU_SM_CLOCK_MHZ={sm_clock}",
        f"GPU_MEM_CLOCK_MHZ={mem_clock}",
        f"GPU_ECC_DBE={ecc_dbe}",
        f"GPU_ECC_DBE_AGG={ecc_dbe_agg}",
        f"GPU_ECC_SBE={ecc_sbe}",
        f"GPU_PCIE_TX_MB_S={pcie_tx:.1f}",
        f"GPU_PCIE_RX_MB_S={pcie_rx:.1f}",
        f"GPU_GAMING_MODE={gaming_mode}",
        f"GB_PRESSURE_STATE={pressure_state}",
        # Shim flow keys (new; bash parser ignores unknowns , back-compat)
        f"SHIM_PHASE={flow.get('phase', '')}",
        f"SHIM_ACTIVE_PATH={flow.get('active_path', '')}",
        f"SHIM_T1_LOCAL_MB={flow.get('tier_t1_local_cur_mb', '')}",
        f"SHIM_T2_LOCAL_MB={flow.get('tier_t2_local_cur_mb', '')}",
        f"SHIM_T3_LOCAL_MB={flow.get('tier_t3_local_cur_mb', '')}",
        f"SHIM_WS_RESERVE_MB={flow.get('workstation_reserve_mb', '')}",
        f"SHIM_WS_RESERVE_EFF_MB={flow.get('workstation_reserve_eff_mb', '')}",
        f"SHIM_KV_RESERVE_MB={flow.get('kv_reserve_effective_mb', '')}",
        f"SHIM_VIRTUAL_VRAM_MB={flow.get('virtual_vram_mb', '')}",
        f"SHIM_CLUSTER_REMOTE_MB={flow.get('cluster_remote_vram_mb', '')}",
        # Orchestrator state (when daemon is running)
        f"ORCH_ECC_DEGRADED={int(orch.get('ecc_degraded', False))}",
        f"ORCH_THERMAL_STRESS={int(orch.get('thermal_stress', False))}",        # Loop D→C gate
        f"ORCH_MEM_BW_STRESS={int(orch.get('mem_bw_stress', False))}",         # Loop G→C gate
        f"ORCH_SBE_ELEVATED={int(orch.get('sbe_elevated', False))}",          # Loop H advisory
        f"ORCH_SBE_SEEN={orch.get('sbe_seen', 0)}",                           # Loop H count
        f"ORCH_CLOCK_THROTTLED={int(orch.get('clock_throttled', False))}",    # Loop I SM clock
        f"ORCH_SM_CLOCK_MAX_MHZ={orch.get('sm_clock_max_mhz', 0)}",          # Loop I reference
        f"ORCH_SHIM_PHASE={orch.get('shim_phase', '')}",                      # current shim phase
        f"ORCH_WS_ABOVE={int(orch.get('ws_above', False))}",
        f"ORCH_WS_RESERVE_MB={orch.get('ws_reserve_mb', '')}",
        f"ORCH_ACTUATE={int(orch.get('actuate', False))}",
        f"ORCH_VRAM_PRESSURE={int(orch.get('vram_pressure', False))}",
        f"ORCH_CLUSTER_PRESSURE={int(orch.get('cluster_pressure', False))}",   # B3
        f"ORCH_HEALTH_OK={int(bool(orch.get('health_ok', True)))}",            # B4
        f"ORCH_HEALTH_EVICT_ARMED={int(orch.get('health_evict_armed', False))}", # B4
        # Continuous OS tuner (Loops O-S)
        f"ORCH_OS_TUNE_ENABLED={int(orch.get('os_tune_enabled', False))}",
        f"ORCH_GAMING_MODE={int(orch.get('gaming_mode', False))}",
        f"ORCH_CPU_GOVERNOR={orch.get('control', {}).get('cpu_governor', {}).get('value', '')}",
        f"ORCH_GPU_PERSISTENCE={int(bool(orch.get('control', {}).get('gpu_persistence', {}).get('value', False)))}",
        f"ORCH_GPU_POWER_LIMIT_W={orch.get('control', {}).get('gpu_power_limit_w', {}).get('value', '')}",
        f"ORCH_GPU_CLOCKS_LOCKED={orch.get('control', {}).get('gpu_clocks_locked', {}).get('value', '')}",
        f"ORCH_SWAPPINESS={orch.get('control', {}).get('swappiness', {}).get('value', '')}",
    ]
    # Topology constants (C0) , inference CPU pinning hints for the shell layer
    try:
        from gb_topology import get_topology
        _topo = get_topology()
        _inf_cpus = ",".join(str(c) for c in _topo.inference_cpus)
        lines += [
            f"TOPO_INFERENCE_CPUS={_inf_cpus}",
            f"TOPO_INFERENCE_THREADS={_topo.inference_threads}",
            f"TOPO_BACKGROUND_THREADS={_topo.background_threads}",
            f"TOPO_PCIE_SAT_MB_S={_topo.pcie_saturation_mb_s:.0f}",
            f"TOPO_IS_BLACKWELL={int(_topo.is_blackwell)}",
        ]
    except Exception:
        pass
    # DCGM health block , only probed when --dcgm flag passed (30s cache)
    if want_dcgm:
        dcgm = _probe_dcgm(0)
        lines += [
            f"GPU_HEALTH_OK={int(dcgm.get('health_ok', True))}",
            f"GPU_HEALTH_SUMMARY={dcgm.get('health_summary', '')}",
            f"GPU_POWER_INSTANT_W={dcgm.get('power_instant_w', '')}",
            f"GPU_NVLINK_BW_MB_S={dcgm.get('nvlink_bw_mb_s', 0)}",
        ]
    print("\n".join(lines))
    sys.exit(0)


if __name__ == "__main__":
    main()

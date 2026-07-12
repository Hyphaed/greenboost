#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_monitor.py — canonical, read-only, cross-process GreenBoost telemetry client.

One place that knows how to read GreenBoost's runtime state, replacing the four
independent shim_stats parsers (greenboost-cli monitor.py, gb_vitals_helper.py,
greenboost_exporter.py, greenboost_setup.sh bash) and the drifting `_GbInfo`
ioctl mirrors (the CLI mirror had phantom kv_compression fields not in
greenboost_ioctl.h; the struct here matches the header exactly).

Scope: **read-only**. This module never actuates — tier moves, KV reserve, pool
cap, turboquant all stay in gb_control.GbControl (GB_ORCH_ACTUATE-gated). The one
mutation here is reset_stale_tracker(), which only deletes a stats file left by a
dead PID (lifted from ai-forge forge/gpu.py so the consumer stops owning it).

Design constraints:
  * No torch / no pynvml import — importable in any process, incl. CPU-only CI.
    (gb_telemetry stays the heavyweight in-process NVML/DCGM stack.)
  * Degrades to a not-loaded snapshot when nothing is present; never raises.

Sources (in order, each optional):
  1. /dev/greenboost           GB_IOCTL_GET_INFO — richest kernel pool state
  2. /sys/module/greenboost    version + params fallback
  3. /run/greenboost/shim_stats  shim flow state (phase, active path, tiers, pid)
  4. /run/greenboost/capabilities.json  shim feature manifest (see gb write path)
  5. nvidia-smi                GPU name + memory (best-effort, probe_gpu=True)

CLI:  gb_monitor.py [--json | --llm | --capabilities]
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


# ── paths ───────────────────────────────────────────────────────────────────
DEVICE_PATH        = Path("/dev/greenboost")
SYS_MODULE_PATH    = Path("/sys/module/greenboost")
SHIM_STATS_PATH    = Path("/run/greenboost/shim_stats")
SHIM_STATS_ALT     = Path("/tmp/greenboost_shim_stats")
METRICS_JSON_PATH  = Path("/run/greenboost/metrics.json")
PHASE_PATH         = Path("/run/greenboost/phase")
RUNTIME_CAPS_PATH  = Path("/run/greenboost/capabilities.json")
INSTALL_CAPS_PATH  = Path("/usr/local/lib/greenboost_capabilities.json")
SHIM_LIB_PATH      = Path("/usr/local/lib/libgreenboost_cuda.so")

_STALE_SECS = 30


# ── ioctl (mirrors struct gb_info in greenboost_ioctl.h — authoritative) ─────
def _IOR(t: int, nr: int, size: int) -> int:
    return (2 << 30) | (size << 16) | (t << 8) | nr


class GbInfo(ctypes.Structure):
    _fields_ = [
        ("vram_physical_mb",     ctypes.c_uint64),
        ("total_ram_mb",         ctypes.c_uint64),
        ("free_ram_mb",          ctypes.c_uint64),
        ("allocated_mb",         ctypes.c_uint64),
        ("max_pool_mb",          ctypes.c_uint64),
        ("safety_reserve_mb",    ctypes.c_uint64),
        ("available_mb",         ctypes.c_uint64),
        ("active_buffers",       ctypes.c_uint32),
        ("oom_active",           ctypes.c_uint32),
        ("nvme_swap_total_mb",   ctypes.c_uint64),
        ("nvme_swap_used_mb",    ctypes.c_uint64),
        ("nvme_swap_free_mb",    ctypes.c_uint64),
        ("nvme_t3_allocated_mb", ctypes.c_uint64),
        ("swap_pressure",        ctypes.c_uint32),
        ("kv_reserve_mb",        ctypes.c_uint32),
        ("kv_used_mb",           ctypes.c_uint32),
        ("kv_t2_mb",             ctypes.c_uint32),
        ("t2_pressure",          ctypes.c_uint32),
        ("phase_reset_seq",      ctypes.c_uint32),
        ("gaming_mode",          ctypes.c_uint32),
        ("_pad",                 ctypes.c_uint32),
        ("total_combined_mb",    ctypes.c_uint64),
    ]


_GB_IOCTL_GET_INFO = _IOR(ord("G"), 2, ctypes.sizeof(GbInfo))

_PRESSURE = {0: "ok", 1: "warn", 2: "critical"}


# ── shim_stats parsing (THE canonical parser) ────────────────────────────────
def parse_shim_stats(text: str) -> dict:
    """Parse KEY=VALUE shim_stats text into a dict of strings. Unknown keys are
    kept; malformed lines skipped. This is the single implementation the
    exporter, vitals helper, and CLI monitor all delegate to."""
    out: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def read_shim_stats(path: "str | os.PathLike | None" = None) -> dict:
    """Read + parse the shim_stats file (with the /tmp fallback), adding two
    derived keys: `_stale` (bool) and `_pid` (int|None). Empty dict if absent."""
    p = Path(path) if path is not None else (
        SHIM_STATS_PATH if SHIM_STATS_PATH.exists() else SHIM_STATS_ALT)
    try:
        text = Path(p).read_text(errors="replace")
    except OSError:
        return {}
    d = parse_shim_stats(text)
    if not d:
        return {}
    ts = d.get("timestamp") or d.get("ts") or "0"
    try:
        tsv = int(ts)
        d["_stale"] = (tsv == 0) or ((time.time() - tsv) > _STALE_SECS)
    except (ValueError, TypeError):
        d["_stale"] = True
    try:
        d["_pid"] = int(d["pid"]) if "pid" in d else None
    except (ValueError, TypeError):
        d["_pid"] = None
    return d


def _to_int(v) -> int:
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return 0


# ── snapshot ─────────────────────────────────────────────────────────────────
@dataclass
class GbSnapshot:
    loaded: bool = False
    version: str = ""
    gpu_name: str = ""
    # kernel pool (MB)
    vram_physical_mb: int = 0
    t2_pool_mb: int = 0
    t2_allocated_mb: int = 0
    t2_available_mb: int = 0
    t3_total_mb: int = 0
    t3_used_mb: int = 0
    total_combined_mb: int = 0
    active_buffers: int = 0
    oom_active: bool = False
    swap_pressure: int = 0
    t2_pressure: int = 0
    kv_reserve_mb: int = 0
    kv_used_mb: int = 0
    kv_t2_mb: int = 0
    gaming_mode: bool = False
    # GPU live (nvidia-smi, best-effort)
    gpu_mem_used_mb: int = 0
    gpu_mem_total_mb: int = 0
    # shim flow (shim_stats)
    shim_stale: bool = True
    shim_phase: str = ""
    shim_active_path: str = ""
    shim_pid: "int | None" = None
    shim: dict = field(default_factory=dict)
    error: str = ""

    @property
    def pressure_label(self) -> str:
        return _PRESSURE.get(self.swap_pressure, "?")

    @property
    def t2_pressure_label(self) -> str:
        return _PRESSURE.get(self.t2_pressure, "?")

    @property
    def total_combined_gb(self) -> float:
        return round(self.total_combined_mb / 1024, 1)

    @property
    def indicator(self) -> str:
        if self.error:
            return "GB ✗"
        if self.loaded:
            return "GB ✓" + (f" {self.gpu_name}" if self.gpu_name else "")
        return "GB ○"

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in (
            "loaded", "version", "gpu_name", "vram_physical_mb", "t2_pool_mb",
            "t2_allocated_mb", "t2_available_mb", "t3_total_mb", "t3_used_mb",
            "total_combined_mb", "active_buffers", "oom_active", "swap_pressure",
            "t2_pressure", "kv_reserve_mb", "kv_used_mb", "kv_t2_mb",
            "gaming_mode", "gpu_mem_used_mb", "gpu_mem_total_mb", "shim_stale",
            "shim_phase", "shim_active_path", "shim_pid", "error")}
        d["total_combined_gb"] = self.total_combined_gb
        d["indicator"] = self.indicator
        return d


def _try_ioctl(s: GbSnapshot) -> bool:
    if not DEVICE_PATH.exists():
        return False
    try:
        fd = os.open(str(DEVICE_PATH), os.O_RDWR)
    except OSError:
        return False
    try:
        import fcntl
        info = GbInfo()
        try:
            fcntl.ioctl(fd, _GB_IOCTL_GET_INFO, info)
        except OSError:
            return False
        s.loaded = True
        s.vram_physical_mb = info.vram_physical_mb
        s.t2_pool_mb = info.max_pool_mb
        s.t2_allocated_mb = info.allocated_mb
        s.t2_available_mb = info.available_mb
        s.t3_total_mb = info.nvme_swap_total_mb
        s.t3_used_mb = info.nvme_swap_used_mb
        s.total_combined_mb = info.total_combined_mb
        s.active_buffers = info.active_buffers
        s.oom_active = bool(info.oom_active)
        s.swap_pressure = info.swap_pressure
        s.t2_pressure = info.t2_pressure
        s.kv_reserve_mb = info.kv_reserve_mb
        s.kv_used_mb = info.kv_used_mb
        s.kv_t2_mb = info.kv_t2_mb
        s.gaming_mode = bool(info.gaming_mode)
        return True
    except Exception:
        return False
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _read_version(s: GbSnapshot) -> None:
    vf = SYS_MODULE_PATH / "version"
    try:
        s.version = vf.read_text().strip()
    except OSError:
        pass


def _probe_gpu(s: GbSnapshot) -> None:
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return
        parts = [p.strip() for p in r.stdout.strip().split("\n")[0].split(",")]
        if parts:
            s.gpu_name = parts[0]
        if len(parts) >= 3:
            s.gpu_mem_used_mb = _to_int(parts[1])
            s.gpu_mem_total_mb = _to_int(parts[2])
            if not s.vram_physical_mb:
                s.vram_physical_mb = s.gpu_mem_total_mb
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def snapshot(probe_gpu: bool = True) -> GbSnapshot:
    """Collect a full read-only GreenBoost state snapshot. Never raises."""
    s = GbSnapshot()
    if not _try_ioctl(s) and SYS_MODULE_PATH.exists():
        s.loaded = True
    _read_version(s)

    d = read_shim_stats()
    if d:
        s.shim = d
        s.shim_stale = bool(d.get("_stale", True))
        s.shim_phase = d.get("phase", "")
        s.shim_active_path = d.get("active_path", "")
        s.shim_pid = d.get("_pid")
        # kernel ioctl is authoritative for pool; fill gaps from shim only
        if not s.loaded and d:
            s.loaded = True

    if probe_gpu:
        _probe_gpu(s)
    return s


def tier_stats() -> "dict | None":
    """Compatibility shape for greenboost-cli get_tier_stats(): None when the
    module isn't loaded, else a flat MB/GB dict for banners + warnings."""
    s = snapshot()
    if not s.loaded:
        return None
    return {
        "loaded": s.loaded,
        "gpu_name": s.gpu_name,
        "t1_vram_mb": s.gpu_mem_total_mb or s.vram_physical_mb,
        "t1_used_mb": s.gpu_mem_used_mb,
        "t2_pool_mb": s.t2_pool_mb,
        "t2_allocated_mb": s.t2_allocated_mb,
        "t2_available_mb": s.t2_available_mb,
        "t3_swap_total_mb": s.t3_total_mb,
        "t3_swap_used_mb": s.t3_used_mb,
        "total_combined_gb": s.total_combined_gb,
        "t2_pressure": s.t2_pressure,
        "swap_pressure": s.swap_pressure,
        "oom_active": s.oom_active,
    }


def context_summary(stats: "dict | None" = None) -> str:
    """The GreenBoost block injected into an LLM's system prompt: a one-line
    tier banner plus behaviour-shaping warnings. Empty string when not loaded.
    Lifted from greenboost-cli context_builder._greenboost_context so every
    consumer (CLI, ai-forge, future agents) emits the same guidance."""
    if stats is None:
        stats = tier_stats()
    if not stats:
        return ""
    t1_used = round(stats.get("t1_used_mb", 0) / 1024, 1)
    t1_total = round(stats.get("t1_vram_mb", 0) / 1024, 1)
    t2 = round(stats.get("t2_available_mb", 0) / 1024, 1)
    t3 = round(stats.get("t3_swap_total_mb", 0) / 1024, 1)
    total = stats.get("total_combined_gb", 0)
    gpu = stats.get("gpu_name", "")
    t1_str = f"{t1_used}/{t1_total}GB" if t1_used else f"{t1_total}GB"
    line = (f"\n- GreenBoost active: T1={t1_str} VRAM  T2={t2}GB RAM  "
            f"T3={t3}GB NVMe  total={total}GB")
    if gpu:
        line += f"  GPU:{gpu}"
    line += "\n"

    warnings: list = []
    if stats.get("oom_active"):
        warnings.append(
            "WARNING: GreenBoost OOM recovery is ACTIVE — memory critically low.")
    t3_used = stats.get("t3_swap_used_mb", 0)
    if t3_used > 0:
        warnings.append(
            f"WARNING: T3 NVMe spillover active ({round(t3_used / 1024, 1)} GB "
            "on disk swap) — inference is ~100× slower than normal. Avoid large "
            "context expansions.")
    t2p = stats.get("t2_pressure", 0)
    if t2p == 2:
        warnings.append(
            "WARNING: T2 DDR pressure is CRITICAL — limit large tool outputs.")
    elif t2p == 1:
        warnings.append("NOTICE: T2 DDR pressure is elevated (warn level).")
    if warnings:
        line += "".join(f"- {w}\n" for w in warnings)
    return line


# ── capability manifest (merge chain) ─────────────────────────────────────────
def _load_caps_file(p: Path) -> "dict | None":
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def capabilities() -> dict:
    """What the GreenBoost shim supports, resolved as: fresh runtime manifest
    (/run/greenboost/capabilities.json, written by the shim at init) → install
    manifest (/usr/local/lib/greenboost_capabilities.json, written by
    greenboost_setup.sh) → binary sniff of the installed .so. Always returns a
    dict with a `features` map and a `source` tag; empty features if unknown."""
    rt = _load_caps_file(RUNTIME_CAPS_PATH)
    if rt is not None and isinstance(rt.get("features"), dict):
        rt.setdefault("source", "runtime")
        return rt
    inst = _load_caps_file(INSTALL_CAPS_PATH)
    if inst is not None and isinstance(inst.get("features"), dict):
        inst.setdefault("source", "install")
        return inst
    # last resort: sniff the shim binary for the cudart-rebind fix literal
    feats: dict = {}
    try:
        blob = SHIM_LIB_PATH.read_bytes()
        feats["gb_quant_cudart_rebind"] = b"cudart rebind" in blob
    except OSError:
        pass
    return {"features": feats, "source": "sniff" if feats else "none"}


# ── stale-tracker cleanup (lifted from ai-forge forge/gpu.py) ─────────────────
def reset_stale_tracker() -> bool:
    """Delete shim_stats/metrics.json left behind by a crashed prior run (a
    dead-PID stats file misleads the shim's headroom calc on the next launch).
    Returns True iff a stale file was removed. Silent no-op / never raises when
    nothing is stale, the PID is alive, or the files are absent."""
    try:
        text = SHIM_STATS_PATH.read_text()
    except OSError:
        return False
    m = re.search(r"^pid=(\d+)", text, re.MULTILINE)
    if not m:
        return False
    pid = int(m.group(1))
    try:
        os.kill(pid, 0)
        return False            # alive — not stale
    except ProcessLookupError:
        pass
    except OSError:
        return False
    removed = False
    for f in (SHIM_STATS_PATH, METRICS_JSON_PATH):
        try:
            f.unlink()
            removed = True
        except OSError:
            pass
    return removed


# ── CLI ───────────────────────────────────────────────────────────────────────
def _print_llm(s: GbSnapshot) -> None:
    caps = capabilities()
    print(f"loaded={int(s.loaded)}")
    print(f"version={s.version}")
    print(f"gpu={s.gpu_name}")
    print(f"phase={s.shim_phase}")
    print(f"active_path={s.shim_active_path}")
    print(f"shim_stale={int(s.shim_stale)}")
    print(f"t1_vram_mb={s.gpu_mem_total_mb or s.vram_physical_mb}")
    print(f"t1_used_mb={s.gpu_mem_used_mb}")
    print(f"t2_pool_mb={s.t2_pool_mb}")
    print(f"t2_available_mb={s.t2_available_mb}")
    print(f"t3_total_mb={s.t3_total_mb}")
    print(f"t3_used_mb={s.t3_used_mb}")
    print(f"total_combined_gb={s.total_combined_gb}")
    print(f"t2_pressure={s.t2_pressure_label}")
    print(f"swap_pressure={s.pressure_label}")
    print(f"oom_active={int(s.oom_active)}")
    print(f"gaming_mode={int(s.gaming_mode)}")
    print(f"caps_source={caps.get('source')}")
    for k, v in sorted(caps.get("features", {}).items()):
        print(f"cap_{k}={int(bool(v))}")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="GreenBoost read-only monitor")
    ap.add_argument("--json", action="store_true", help="full snapshot as JSON")
    ap.add_argument("--llm", action="store_true", help="compact key=value")
    ap.add_argument("--capabilities", action="store_true",
                    help="shim capability manifest")
    args = ap.parse_args()

    if args.capabilities:
        caps = capabilities()
        if args.json:
            print(json.dumps(caps, indent=2))
        else:
            print(f"source={caps.get('source')}")
            for k, v in sorted(caps.get("features", {}).items()):
                print(f"{k}={int(bool(v))}")
        return 0

    s = snapshot()
    if args.json:
        print(json.dumps(s.as_dict(), indent=2))
    else:
        _print_llm(s)          # --llm is also the default compact view
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

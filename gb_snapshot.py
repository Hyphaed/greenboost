#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
gb_snapshot.py , read-only, versioned telemetry snapshot for other programs.

Written atomically by the supervisor loop to a JSON file that any user can
read without root (default ~/.cache/greenboost/snapshot.json, override with
GB_SNAPSHOT_PATH). Version 1 follows the field names of the Trading project's
resource contract: vram{physical,used,free,virtual,reserve}_mib, tiers_mib,
lease. A field this repo cannot read is null, never 0 and never a guess.

What is NOT known here and therefore null: the lease unless the budgeted
lease arbiter (gb_lease.py) has been initialised (`python3 gb_lease.py init`),
and the reserve when no live shim has published one. With the arbiter in use
the lease block carries held, holders, budget_mib, waiters, last_wait_s and the
optional extras queue_max and rejects (version 1 readers ignore extras).

Rules for the numbers:
  * physical_mib / free_mib come from NVML (the real card), never from the
    shim's cached value (that read 0 before the first allocation).
  * used_mib = physical_mib - free_mib, so used + free == physical exactly
    (NVML's own "used" leaves out the driver-reserved slice).
  * tiers are reported per LIVE process (pid, name, tier occupancy). Dead
    processes leave metrics.<pid>.json behind; they are skipped. While a
    process is in phase INIT the shim has not counted yet and prints 0, so
    its tiers are null.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

SNAPSHOT_VERSION = 1
RUN_DIR = Path(os.environ.get("GB_STATS_DIR", "/run/greenboost"))
_TIER_GROUPS = {"t1": ("t1_local", "t1_feeder"),
                "t2": ("t2_local", "t2_feeder"),
                "t3": ("t3_local", "t3_feeder")}


def snapshot_path() -> Path:
    p = os.environ.get("GB_SNAPSHOT_PATH")
    return Path(p) if p else Path.home() / ".cache" / "greenboost" / "snapshot.json"


def _pid_name(pid: int) -> Optional[str]:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return None


def _read_kv(path: Path) -> dict:
    kv: dict = {}
    try:
        for line in path.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep:
                kv[k.strip()] = v.strip()
    except OSError:
        pass
    return kv


def _group(tiers: Optional[dict]) -> Optional[dict]:
    if not tiers:
        return None
    return {g: sum((tiers.get(m) or {}).get("cur_mib") or 0 for m in ms)
            for g, ms in _TIER_GROUPS.items()}


def collect_processes(run_dir: Path = RUN_DIR, now: Optional[float] = None) -> list:
    """One entry per live process that wrote metrics.<pid>.json."""
    now = time.time() if now is None else now
    out = []
    for f in sorted(run_dir.glob("metrics.*.json")):
        stem = f.name[len("metrics."):-len(".json")]
        if not stem.isdigit():
            continue
        pid = int(stem)
        name = _pid_name(pid)
        if name is None:          # dead process: stale file, skip
            continue
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        phase = d.get("phase")
        tiers = None
        if phase != "INIT" and isinstance(d.get("tiers"), dict):
            tiers = {}
            for tname, t in d["tiers"].items():
                if isinstance(t, dict):
                    tiers[tname] = {"cur_mib": t.get("cur_mb"), "peak_mib": t.get("peak_mb")}
        ts = d.get("timestamp")
        out.append({
            "pid": pid,
            "name": name,
            "phase": phase,
            "age_s": round(now - ts, 1) if isinstance(ts, (int, float)) else None,
            "tiers_mib": tiers,
            "tier_mib": _group(tiers),
        })
    return out


def sum_tiers(procs: list) -> Optional[dict]:
    """t1/t2/t3 summed over processes that have counted; None if none have."""
    counted = [p["tiers_mib"] for p in procs if p.get("tiers_mib")]
    if not counted:
        return None
    res = {}
    for g, members in _TIER_GROUPS.items():
        res[g] = sum((t.get(m) or {}).get("cur_mib") or 0 for t in counted for m in members)
    return res


def _virtual_and_reserve(run_dir: Path, procs: list) -> tuple:
    """From the newest live process's shim_stats.<pid>; None when absent."""
    for p in sorted(procs, key=lambda p: p["age_s"] if p["age_s"] is not None else 1e12):
        kv = _read_kv(run_dir / f"shim_stats.{p['pid']}")
        v = kv.get("virtual_vram_mb")
        r = kv.get("workstation_reserve_eff_mb")
        if v is not None or r is not None:
            return (int(v) if v and v.isdigit() else None,
                    int(r) if r and r.isdigit() else None)
    return None, None


_NO_LEASE = {"held": None, "holders": None, "budget_mib": None,
             "waiters": None, "last_wait_s": None}


def lease_block(lease_path: Optional[Path] = None, now: Optional[float] = None) -> dict:
    """The arbiter's state, read without locking or writing; all-null when it is not in use
    (that null is how the engine learns it must keep its own file lock)."""
    try:
        import gb_lease
        blk = gb_lease.read_block(lease_path, now)
    except Exception:
        blk = None
    return blk if blk is not None else dict(_NO_LEASE)


def build_snapshot(total_mb: int, free_mb: int, run_dir: Path = RUN_DIR,
                   now: Optional[float] = None, lease_path: Optional[Path] = None) -> dict:
    """total_mb/free_mb: the real card as NVML reports it (0 total = unread)."""
    now = time.time() if now is None else now
    procs = collect_processes(run_dir, now)
    virt, reserve = _virtual_and_reserve(run_dir, procs)
    if total_mb > 0:
        vram = {"physical_mib": total_mb, "used_mib": total_mb - free_mb, "free_mib": free_mb}
    else:
        vram = {"physical_mib": None, "used_mib": None, "free_mib": None}
    vram["virtual_mib"] = virt
    vram["reserve_mib"] = reserve
    return {
        "version": SNAPSHOT_VERSION,
        "ts": now,
        "source": "greenboost",
        "vram": vram,
        "tiers_mib": sum_tiers(procs),
        "tier_processes": procs,
        "lease": lease_block(lease_path, now),
    }


def write_snapshot(snap: dict, path: Optional[Path] = None) -> Path:
    """Atomic and symlink-safe (SEC-02): unpredictable temp name made with O_EXCL|O_NOFOLLOW in the same
    directory, mode 0644 set on the descriptor, renamed over the target. A target that is a symlink is refused."""
    import gb_lease
    path = path or snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return gb_lease.safe_atomic_write(path, json.dumps(snap, indent=1), 0o644)


if __name__ == "__main__":      # python3 gb_snapshot.py [--write]: one-shot, read-only unless --write
    import sys
    try:
        from gb_nvml import get_nvml
        _u, _f, _t, _ = get_nvml(0).mem()
    except Exception:
        _f = _t = 0
    s = build_snapshot(_t, _f)
    if "--write" in sys.argv:
        print(write_snapshot(s))
    else:
        print(json.dumps(s, indent=1))

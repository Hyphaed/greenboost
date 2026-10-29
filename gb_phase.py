#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""gb_phase.py - GPU phase lease: one heavy phase on the card at a time, across every consumer.

The card is shared by a vision LLM (Qwen3.8 + projector, ~10.6 GB), a FLUX batch (~9.9 GiB peak), the browser's GPU raster and the
video encoder. Two heavy phases at once do not degrade, they die (the engine's vision encoder ran out of CUDA memory next to a FLUX
load). Each consumer used to invent its own wait: the studio's TypeScript hub has a lock file, ai-forge has a VRAM wait, gb-synapse has
its own load gate. This module is the one lease they can all take, and it speaks the studio's lock format so they interoperate today.

Lock file (same as svg-animation-studio/packages/art/src/lock.ts)
    path     $SVG_STUDIO_GPU_LOCK, else <tmpdir>/svg-studio-gpu.lock (machine wide)
    format   JSON {"class","pid","owner","job","since","heartbeat","token"}
    created  O_EXCL: two processes cannot both win
    stale    the pid is dead, or the heartbeat is older than 120 s (the holder refreshes it every 10 s): the next acquirer removes it
    release  only by the token that took it

What a lease adds to the lock
    * need_pct: % of the card's PHYSICAL VRAM that must be free before the phase starts (derived by the caller from the model's real
      footprint; never a literal GB figure in here).
    * when VRAM is short and the phase is not itself an LLM, gb-synapse's own served models are PAUSED (KV saved, VRAM returned),
      never stopped, never killed. Foreign processes are reported by name and left alone.
    * a dataflux `gpu_phase` event per phase with its wall time and peak VRAM, so "where did the GPU time go" has an answer.

    with gb_phase.gpu_phase("flux-batch", need_pct=75, owner="ai-forge", wait_s=120):
        run_the_batch()
"""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

HEARTBEAT_S = 10.0
STALE_S = 120.0
_PEAK_SAMPLE_S = 2.0
_LLM_CLASSES = ("llm", "vlm")


class PhaseBusy(RuntimeError):
    """The card is not available for this phase. Carries the holder and the numbers; nothing was started or changed."""

    def __init__(self, message: str, holder: "dict | None" = None, free_pct: "float | None" = None, need_pct: float = 0.0):
        super().__init__(message)
        self.holder, self.free_pct, self.need_pct = holder, free_pct, need_pct


def lock_path() -> Path:
    env = os.environ.get("SVG_STUDIO_GPU_LOCK")
    return Path(env).resolve() if env else Path(tempfile.gettempdir()) / "svg-studio-gpu.lock"


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError):
        return False


def _iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int((t % 1) * 1000):03d}Z"


def _parse_iso(s: "str | None") -> "float | None":
    try:
        base, _, frac = (s or "").rstrip("Z").partition(".")
        import calendar
        return calendar.timegm(time.strptime(base, "%Y-%m-%dT%H:%M:%S")) + (float("0." + frac) if frac else 0.0)
    except Exception:
        return None


def read_lock(path: "Path | None" = None) -> "dict | None":
    try:
        return json.loads((path or lock_path()).read_text())
    except Exception:
        return None


def is_stale(info: "dict | None", stale_s: float = STALE_S, now: "float | None" = None) -> bool:
    if not info:
        return True
    if not _pid_alive(info.get("pid")):
        return True
    hb = _parse_iso(info.get("heartbeat") or info.get("since"))
    return hb is not None and ((now if now is not None else time.time()) - hb) > stale_s


def vram_mib() -> "tuple[int, int] | None":
    """(used, total) MiB of the first GPU, or None without NVML/nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        used, total = (int(x) for x in out[0].split(","))
        return used, total
    except Exception:
        return None


def _gpu_processes() -> list:
    """Names and VRAM of the processes on the card, for the 'who holds it' part of a refusal. Read only."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        rows = []
        for line in out:
            pid, name, mem = [x.strip() for x in line.split(",")]
            rows.append({"pid": int(pid), "name": os.path.basename(name), "mib": int(mem)})
        return rows
    except Exception:
        return []


def _free_pct(vram_fn) -> "float | None":
    v = vram_fn()
    if not v or not v[1]:
        return None
    return 100.0 * (v[1] - v[0]) / v[1]


def phase_status(need_pct: float = 0.0, *, vram_fn=vram_mib, path: "Path | None" = None, stale_s: float = STALE_S) -> dict:
    """What a phase asking for need_pct would meet right now. Read only."""
    holder = read_lock(path)
    v = vram_fn()
    free = _free_pct(vram_fn)
    stale = is_stale(holder, stale_s) if holder else False
    return {"lock_path": str(path or lock_path()), "holder": holder, "holder_stale": stale,
            "vram_used_mib": v[0] if v else None, "vram_total_mib": v[1] if v else None,
            "free_pct": round(free, 1) if free is not None else None,
            "need_pct": need_pct, "would_start": (holder is None or stale) and (free is None or free >= need_pct),
            "gpu_processes": _gpu_processes()}


def _emit(event: dict) -> None:
    try:
        import gb_dataflux
        gb_dataflux.emit({"node": "host", "label": "gb_phase", "kind": "gpu_phase", **event})
    except Exception:
        pass


def _synapse_served() -> list:
    try:
        import gb_synapse
        return [str(s.get("model")) for s in gb_synapse.ps() if s.get("model")]
    except Exception:
        return []


def _synapse_pause(model: str) -> dict:
    import gb_synapse
    return gb_synapse.pause(model)


class Lease:
    """A held GPU phase. release() is idempotent and safe from any thread."""

    def __init__(self, name: str, path: Path, info: dict, need_pct: float, vram_fn):
        self.name, self.path, self.info, self.need_pct = name, path, info, need_pct
        self.token = info["token"]
        self.t0 = time.time()
        self.peak_mib = 0
        self.paused: list = []
        self._vram_fn = vram_fn
        self._stop = threading.Event()
        self._released = False
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"gb-phase-{name}")
        self._thread.start()

    def _run(self) -> None:
        last_hb = 0.0
        while not self._stop.wait(_PEAK_SAMPLE_S):
            v = self._vram_fn()
            if v:
                self.peak_mib = max(self.peak_mib, v[0])
            if time.time() - last_hb >= HEARTBEAT_S:
                last_hb = time.time()
                cur = read_lock(self.path)
                if cur and cur.get("token") == self.token:
                    cur["heartbeat"] = _iso(time.time())
                    try:
                        self.path.write_text(json.dumps(cur))
                    except OSError:
                        pass

    def release(self) -> dict:
        if self._released:
            return {"released": False}
        self._released = True
        self._stop.set()
        v = self._vram_fn()
        if v:
            self.peak_mib = max(self.peak_mib, v[0])
        removed = False
        cur = read_lock(self.path)
        if cur and cur.get("token") == self.token:
            try:
                self.path.unlink()
                removed = True
            except OSError:
                pass
        seconds = round(time.time() - self.t0, 1)
        _emit({"status": "end", "phase": self.name, "owner": self.info.get("owner"), "class": self.info.get("class"),
               "seconds": seconds, "peak_vram_mib": self.peak_mib, "paused_models": self.paused})
        return {"released": removed, "seconds": seconds, "peak_vram_mib": self.peak_mib}

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def _refusal(name: str, need_pct: float, free_pct: "float | None", holder: "dict | None", reason: str) -> PhaseBusy:
    procs = _gpu_processes()
    who = ", ".join(f"{p['name']} (pid {p['pid']}, {p['mib']} MiB)" for p in procs[:4]) or "no process was listed"
    holder_txt = (f" The lease is held by {holder.get('owner')} (job {holder.get('job') or 'unnamed'}, pid {holder.get('pid')})."
                  if holder else "")
    free_txt = f"{free_pct:.0f}% of the card is free" if free_pct is not None else "the free VRAM could not be read"
    return PhaseBusy(
        f"The GPU phase '{name}' did not start: {reason}.{holder_txt} {free_txt} and it needs {need_pct:.0f}%. "
        f"Nothing was started and nothing was changed, and no other process was touched. On the card now: {who}. "
        f"Wait for it to finish, or ask gb-synapse to hand its model back: gb synapse pause <model>.",
        holder=holder, free_pct=free_pct, need_pct=need_pct)


def acquire_phase(name: str, need_pct: float = 0.0, *, wait_s: float = 0.0, owner: str = "greenboost", cls: str = "image",
                  job: str = "", pause_synapse: "bool | None" = None, keep_models: "tuple | list" = (),
                  stale_s: float = STALE_S, poll_s: float = 1.0, path: "Path | None" = None,
                  vram_fn=vram_mib, pause_fn=None, served_fn=None, sleep=time.sleep, now=time.time) -> Lease:
    """Take the lease for a heavy GPU phase, waiting up to wait_s. Raises PhaseBusy (with the reason in words) when it cannot.

    need_pct: % of the card's physical VRAM that must be free; the caller derives it from the phase's real footprint.
    pause_synapse: None = pause gb-synapse's served models when VRAM is short, unless the phase is itself an LLM/VLM (cls llm|vlm).
    keep_models: models never paused. Foreign processes are never touched."""
    path = path or lock_path()
    pause_fn = pause_fn or _synapse_pause
    served_fn = served_fn or _synapse_served
    if pause_synapse is None:
        pause_synapse = cls not in _LLM_CLASSES
    deadline = now() + max(0.0, wait_s)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(8)
    t = now()
    info = {"class": cls, "pid": os.getpid(), "owner": owner, "job": job or name, "since": _iso(t), "heartbeat": _iso(t), "token": token}

    while True:                                            # 1. the lock
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(info))
            break
        except FileExistsError:
            holder = read_lock(path)
            if is_stale(holder, stale_s):
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            if now() >= deadline:
                raise _refusal(name, need_pct, _free_pct(vram_fn), holder, "another phase holds the card")
            sleep(poll_s)

    paused: list = []
    try:                                                   # 2. the VRAM
        asked = False
        while need_pct > 0:
            free = _free_pct(vram_fn)
            if free is None or free >= need_pct:
                break
            if pause_synapse and not asked:
                asked = True
                for m in served_fn():
                    if m in set(keep_models):
                        continue
                    r = pause_fn(m)
                    if isinstance(r, dict) and r.get("ok"):
                        paused.append(m)
                if paused:
                    continue
            if now() >= deadline:
                raise _refusal(name, need_pct, free, None, "not enough VRAM is free")
            sleep(poll_s)
    except BaseException:
        cur = read_lock(path)
        if cur and cur.get("token") == token:
            try:
                path.unlink()
            except OSError:
                pass
        raise
    lease = Lease(name, path, info, need_pct, vram_fn)
    lease.paused = paused
    free = _free_pct(vram_fn)
    _emit({"status": "start", "phase": name, "owner": owner, "class": cls, "need_pct": need_pct,
           "free_pct": round(free, 1) if free is not None else None, "paused_models": paused})
    return lease


@contextmanager
def gpu_phase(name: str, need_pct: float = 0.0, **kw):
    """Context manager over acquire_phase(): the lease is released on exit, errors included."""
    lease = acquire_phase(name, need_pct, **kw)
    try:
        yield lease
    finally:
        lease.release()

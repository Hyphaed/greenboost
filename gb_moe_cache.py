"""gb_moe_cache: size llama.cpp's GPU expert cache (`--moe-cache-mib`, upstream PR 29887) from the free VRAM, and switch a MoE model to "all experts on the host + a hot-expert cache on the GPU".

Why (measured 2026-10-29, RTX 5070 12 GB with the desktop holding 3.5 GB, Qwen3.8-35B-A3B APEX, stock upstream engine b11999 + cuBLAS prefill, no MTP, decode tok/s):
  today's placement  --n-cpu-moe 30                     54 tok/s   engine VRAM ~8.3 GB
  all experts on host + cache 3000 MiB                  74 tok/s   engine VRAM  5.0 GB
  all experts on host + cache 4500 MiB                  93 tok/s   engine VRAM  6.6 GB
  all experts on host + cache 5500 MiB                 95-101 tok/s engine VRAM  7.6 GB
  all experts on host + cache 6500 MiB                 13 to 96 tok/s (card full, 0 MiB spare: unstable)
MoE routing is skewed, so a few GB of hot experts on the GPU serve most tokens; every token that hits skips a PCIe upload. The engine's fixed cost is the dense weights
(attention, norms: 1.65 GB here) + KV + compute buffers (~0.4 GB at ubatch 512); everything left, minus a safety margin, is cache.
The cache only serves batches of up to 32 tokens (decode and short speculative drafts); prompt processing is unchanged.

`plan()` is pure arithmetic (testable without a GPU). `engine_supports()` asks the engine binary, so nothing is emitted for an engine without the flag.
Env: GB_MOE_CACHE = auto (default) | 0/off | <MiB>; GB_MOE_CACHE_MIN_GB (3.0), GB_MOE_CACHE_SAFETY_FRAC (0.16 of the free budget, at least 0.5 GB).
"""
from __future__ import annotations

import functools
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

# The MTP draft model is a second llama context (own KV and compute buffers). Measured 2026-10-30 as engine VRAM minus the same engine without MTP at the same cache size: 1.5 GB at draft
# depth 2, 2.0 GB at depth 4 (ubatch 512). The old 0.6 GB (one compute buffer) let the default depth-4 serve run out of memory at load and left depth 2 only 0.5 GB spare.
MTP_CONTEXT_GB = 1.5
# Draft depth while the cache is in use. Measured (cache auto-sized, ubatch 512, 3 prompts x 200 tokens): depth 2 = 181 tok/s (accept 0.74), depth 3 = 160, depth 4 does not load: the verify
# batch of 1+depth tokens touches more experts and the extra experts cross PCIe. Cache-only was 128 tok/s on the same card.
MTP_DRAFT_N = 2
# Safety fraction 0.11 -> 0.16 (2026-10-30): with 0.11 the engine crashed with CUDA out of memory on the first real CLI request (a second concurrent slot opened with 289 MiB free; the
# desktop had grown by ~0.5 GB since sizing). Measured on the 5070: cache 3534 MiB = 108 tok/s with 987 MiB spare; 3000 MiB = 106 tok/s (-2%) with 1721 MiB spare; 2600 MiB = 98 tok/s (-9%).
COMPUTE_BUF_GB_AT_512 = 0.4          # measured: engine fixed cost 2.05 GB = 1.65 dense + ~0.4 KV and compute buffers at ubatch 512
# Below this the cache is not worth switching the placement for (3 GB gave 74 vs 54 tok/s, so the real floor is lower, but the sweep stopped there).
DEFAULT_MIN_GB = 3.0


@dataclass
class MoeCachePlan:
    cache_mib: int
    n_cpu_moe: int
    detail: str


@functools.lru_cache(maxsize=8)
def _help(path: str, mtime: float) -> str:
    try:
        env = {**os.environ, "LD_LIBRARY_PATH": os.path.dirname(path) + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
        out = subprocess.run([path, "--help"], capture_output=True, text=True, timeout=60, env=env)
        return out.stdout + out.stderr
    except Exception:  # noqa: BLE001
        return ""


def engine_help(server_path) -> str:
    """The engine's --help text (cached per binary and mtime); "" when it cannot be run."""
    p = str(server_path)
    try:
        return _help(p, os.path.getmtime(p))
    except OSError:
        return ""


def engine_supports(server_path) -> bool:
    """True when this llama-server accepts --moe-cache-mib."""
    return "--moe-cache-mib" in engine_help(server_path)


def engine_supports_static(engine_dir) -> bool:
    """Like engine_supports but WITHOUT running anything (status paths must not touch the GPU): the flag text lives in llama-server or in libllama-common.so inside the engine directory."""
    import glob
    needle = b"--moe-cache-mib"
    for f in [os.path.join(str(engine_dir), "llama-server")] + sorted(glob.glob(os.path.join(str(engine_dir), "libllama-common.so*"))):
        try:
            with open(f, "rb") as fh:
                tail = b""
                while True:
                    chunk = fh.read(8 << 20)
                    if not chunk:
                        break
                    if needle in tail + chunk[:len(needle) - 1] or needle in chunk:
                        return True
                    tail = chunk[-(len(needle) - 1):]
        except OSError:
            continue
    return False


def compat_flags(cmd: list, server_path) -> list:
    """Adapt flags a newer upstream engine renamed: `--no-mmap` became `--load-mode none`. Leaves cmd alone for an engine that still knows the old flag (or cannot be probed)."""
    if "--no-mmap" not in cmd:
        return cmd
    h = engine_help(server_path)
    if h and "--no-mmap" not in h and "--load-mode" in h:
        i = cmd.index("--no-mmap")
        return cmd[:i] + ["--load-mode", "none"] + cmd[i + 1:]
    return cmd


def _setting() -> str:
    return os.environ.get("GB_MOE_CACHE", "auto").strip().lower()


def plan(*, n_layers: int, n_cpu_moe: int, dense_gb: float, expert_gb_total: float, kv_gb: float, vision_gb: float, budget_gb: float, ubatch: Optional[int] = None, mtp: bool = False) -> Optional[MoeCachePlan]:
    """The all-host + cache placement for this model and this much free VRAM, or None to keep the placement as it is.

    budget_gb = free physical VRAM (what serve() already uses); n_cpu_moe = layers whose experts the current placement keeps on the host (0 = everything fits: no cache)."""
    s = _setting()
    if s in ("0", "off", "false", "no") or n_cpu_moe <= 0 or n_layers <= 0:
        return None
    ub = ubatch or 512
    compute = COMPUTE_BUF_GB_AT_512 * max(1.0, ub / 512.0)             # linear in ubatch: a conservative guess above 512, measured only at 512
    safety = max(0.5, float(os.environ.get("GB_MOE_CACHE_SAFETY_FRAC", "0.16")) * budget_gb)
    draft = MTP_CONTEXT_GB if mtp else 0.0
    room = budget_gb - dense_gb - kv_gb - vision_gb - compute - draft - safety
    if s.isdigit():
        mib = int(s)
    else:
        if room < float(os.environ.get("GB_MOE_CACHE_MIN_GB", str(DEFAULT_MIN_GB))):
            return None
        mib = int(min(room, expert_gb_total) * 1024)
    if mib <= 0:
        return None
    return MoeCachePlan(mib, n_layers, f"free {budget_gb:.1f} GB - dense {dense_gb:.1f} - kv {kv_gb:.1f} - vision {vision_gb:.1f} - compute {compute:.1f} - mtp {draft:.1f} - safety {safety:.1f} = {room:.1f} GB of cache")


def engine_info(server_path, moecache_dir, vendored_dir=None) -> dict:
    """Which engine this llama-server is: kind ("moecache" | "vendored" | "other"), directory, version file content, and whether it accepts --moe-cache-mib. `engine_supports` runs the binary (cached)."""
    d = os.path.dirname(str(server_path))
    kind = "moecache" if os.path.realpath(d) == os.path.realpath(str(moecache_dir)) else ("vendored" if vendored_dir and os.path.realpath(d) == os.path.realpath(str(vendored_dir)) else "other")
    try:
        with open(os.path.join(d, "engine.version"), encoding="utf-8") as f:
            version = f.read().strip()
    except OSError:
        version = ""
    return {"kind": kind, "dir": d, "version": version, "supports_cache": engine_supports(server_path)}


UNDER_GB_WARN = 1.0      # a cache this small keeps almost nothing hot: the placement change is not worth its cost


def decision_event(*, model: str, plan_: Optional[MoeCachePlan], inputs: dict, engine: dict, n_cpu_moe: int, is_moe: bool, mtp: bool, mtp_draft_n: Optional[int], mtp_reason: str = "") -> dict:
    """The fields of the dataflux `moe_cache_plan` event for one serve (emit_decision adds the kind). Pure. status=warn for: a cache under UNDER_GB_WARN GB, or a MoE model with experts on the host served by an engine without the flag."""
    skip = ""
    if plan_ is None:
        if not is_moe or not n_cpu_moe:
            skip = "no_experts_on_host"
        elif _setting() in ("0", "off", "false", "no"):
            skip = "disabled"
        elif not engine.get("supports_cache"):
            skip = "engine_without_flag"
        else:
            skip = "not_enough_room"
    status, warn = "ok", ""
    if plan_ is not None and plan_.cache_mib < UNDER_GB_WARN * 1024:
        status, warn = "warn", f"expert cache only {plan_.cache_mib} MiB (< {UNDER_GB_WARN:g} GB): few hot experts stay resident"
    elif skip == "engine_without_flag":
        status, warn = "warn", "MoE experts are on the host and this engine has no --moe-cache-mib: sudo greenboost synapse build-moecache-engine && sudo greenboost synapse use-engine moecache"
    ev = {"node": "host", "label": "gb_moe_cache", "n_items": 1, "items": [model], "duration_s": 0.0, "status": status,
          "model": model, "cache_mib": plan_.cache_mib if plan_ else 0, "n_cpu_moe": int(n_cpu_moe or 0), "mtp": bool(mtp), "mtp_draft_n": int(mtp_draft_n or 0), "mtp_reason": mtp_reason,
          "engine_kind": engine.get("kind", ""), "engine_version": engine.get("version", ""), "engine_supports_cache": bool(engine.get("supports_cache")),
          "skip_reason": skip, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in inputs.items()}}
    if plan_ is not None:
        ev["detail"] = plan_.detail
    if warn:
        ev["warning"] = warn
    return ev


def emit_decision(**kw) -> None:
    """Emit the moe_cache_plan dataflux event (best effort, never raises). Arguments as decision_event."""
    try:
        import gb_dataflux
        gb_dataflux.emit({"kind": "moe_cache_plan", **decision_event(**kw)})
    except Exception:  # noqa: BLE001
        pass


LAST_FIELDS = ("ts", "model", "cache_mib", "n_cpu_moe", "mtp", "mtp_draft_n", "mtp_reason", "engine_kind", "engine_version", "skip_reason", "status", "budget_gb")


def last_decision(since_hours: float = 72.0) -> Optional[dict]:
    """The most recent moe_cache_plan event as a compact dict (None when no serve has logged one). Reads the dataflux log only."""
    try:
        import gb_dataflux
        evs = [e for e in gb_dataflux.read_events(since_hours=since_hours) if e.get("kind") == "moe_cache_plan"]
    except Exception:  # noqa: BLE001
        return None
    return {k: evs[-1][k] for k in LAST_FIELDS if k in evs[-1]} if evs else None

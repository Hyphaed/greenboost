"""Retry a model load that ran out of VRAM, one lighter rung at a time.

Why this exists. The Qwen3.8 recipe pins micro-batch 2048 and 28 MoE layers on
the CPU, measured at 11.2 GB of the 12 GB card (92%) with a light desktop. With
a browser, a VM and a chat app holding another ~1 GB, the engine's prompt-
processing buffers no longer fit: `OOM in smart_alloc ... failed to allocate
compute pp buffers`, and the engine dies ~40 s into the load, after /llamaserve
already printed "started". Nothing then told the user why, or tried less.

The rungs come from the recipe's own measurements, not from new VRAM math:
micro-batch 512 uses ~0.7 GB less than 2048, and each step up in `--n-cpu-moe`
trades speed for VRAM (recipe table: 28 → 49.7 tok/s, 32 → 46.8, 40 → 38.3).
This is a stop-gap in the CLI: serving-fit logic belongs in gb_synapse, which
has uncommitted edits, so it was not changed there."""
from __future__ import annotations

import time
from pathlib import Path

#: (ubatch_size, n_cpu_moe or None to keep the recipe's), lightest change first.
LADDER: "list[tuple[int, int | None]]" = [(1024, None), (512, None), (512, 32), (512, 40)]

OOM_SIGNATURES = ("OOM in smart_alloc", "failed to allocate compute", "failed to allocate CUDA0 buffer",
                  "cudaMalloc failed: out of memory")


def log_shows_oom(text: str) -> bool:
    return any(sig in text for sig in OOM_SIGNATURES)


def describe(rung: "tuple[int, int | None]") -> str:
    ub, moe = rung
    return f"micro-batch {ub}" + (f", {moe} expert layers on the CPU" if moe else "")


def _read_tail(path: Path, limit: int = 60_000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - limit))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def watch_and_retry(*, status_fn, log_file: Path, relaunch, say, poll: float = 2.0,
                    max_wait: float = 1500.0, ladder=None, sleep=time.sleep, now=time.monotonic) -> str:
    """Watch a load that is in progress. Returns "running", "gone" (stopped for a reason
    other than VRAM), "exhausted" (every rung ran out of VRAM) or "timeout".

    `relaunch(ubatch, n_cpu_moe)` stops the dead engine and starts a new one; it may raise."""
    rungs = list(LADDER if ladder is None else ladder)
    used = 0
    t0 = now()
    launched_at = time.time()
    while now() - t0 < max_wait:
        st = status_fn()
        if st == "running":
            return "running"
        if st == "stopped":
            fresh = log_file.exists() and log_file.stat().st_mtime >= launched_at - 5
            if not (fresh and log_shows_oom(_read_tail(log_file))):
                return "gone"
            if used >= len(rungs):
                say("The model still does not fit with the lightest settings. Something else is holding the "
                    "card: /gpu shows who. Close it (nothing is killed here) and run /llamaserve load again.")
                return "exhausted"
            rung = rungs[used]
            used += 1
            say(f"The load ran out of VRAM (the desktop and other apps share this card). "
                f"Retrying lighter: {describe(rung)} ({used}/{len(rungs)}).")
            try:
                relaunch(*rung)
            except Exception as e:
                say(f"Retry failed: {str(e)[:160]}")
                return "gone"
            launched_at = time.time()
            t0 = now()
        sleep(poll)
    return "timeout"

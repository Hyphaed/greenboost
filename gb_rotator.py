#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_rotator.py — sequential model-rotation runner for overnight autonomy.

A 12+8 GB cluster serves ONE model at a time; multi-model overnight work is
therefore a rotation, not a scheduler: serve model → run the pipeline that
consumes it (via the gb-synapse Ollama-compatible endpoint, default :11435)
→ unload → next. This module is that rotation, dataflux-recorded end to end
so the whole night is followable through the dataflux MCP (`dataflux_models`).

Dual serve mode (honest about what's installed):
    gb-synapse engine built  → gb_synapse.serve(model, ctx=..., use_cluster=True)
    engine NOT built         → ask the running ollama to load the model
                               (POST /api/generate {"keep_alive": "10m"})
Either way the pipeline's `work` command talks to the resolved endpoint
(FORGE_OLLAMA_URL → gb-synapse's own port → raw Ollama's legacy :11434) as
usual.

Resumable: the queue file records per-job status (pending/done/failed);
re-running the same queue skips jobs already done. Aborts after 3
consecutive failures — an overnight run that keeps failing is telling you
something (dead endpoint, full disk), not asking for 20 more attempts.

CLI:
    python3 gb_rotator.py run <queue.json> [--llm]
    python3 gb_rotator.py example > queue.json

Env:
    GB_ROTATOR_DRY=1   skip serve/unload (work still runs, events still emit)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gb_dataflux  # noqa: E402

MAX_CONSECUTIVE_FAILURES = 3


def _candidate_urls() -> list[str]:
    """Priority order for the Ollama-compatible endpoint this rotation talks
    to: an explicit FORGE_OLLAMA_URL override, then gb-synapse's own proxy
    port (GB_SYNAPSE_PORT, default 11435), then raw Ollama's legacy :11434
    as a last-resort fallback for a box still mid-migration off it."""
    urls = []
    env = os.environ.get("FORGE_OLLAMA_URL")
    if env:
        urls.append(env.rstrip("/"))
    try:
        import gb_synapse
        port = gb_synapse.DEFAULT_PORT
    except Exception:
        port = int(os.environ.get("GB_SYNAPSE_PORT", "11435"))
    synapse_url = f"http://127.0.0.1:{port}"
    if synapse_url not in urls:
        urls.append(synapse_url)
    if "http://127.0.0.1:11434" not in urls:
        urls.append("http://127.0.0.1:11434")
    return urls


def _resolve_ollama_url() -> "str | None":
    """Probe `_candidate_urls()` in priority order, return the first that
    answers /api/ps. None if nothing is alive. Resolved fresh on every call
    (never a frozen constant) so a box mid-migration between gb-synapse and
    raw ollama always talks to whichever is actually up right now."""
    for url in _candidate_urls():
        try:
            with urllib.request.urlopen(f"{url}/api/ps", timeout=5):
                return url
        except Exception:
            continue
    return None


@dataclass
class RotationJob:
    model: str                    # manifest/ollama name, served exclusively
    work: str                     # shell command run while the model is up
    timeout_s: int = 7200
    ctx: int = 16384
    status: str = "pending"       # pending | done | failed (queue-file state)


def _dry() -> bool:
    return os.environ.get("GB_ROTATOR_DRY") == "1"


def _emit(model: str, phase: str, status: str, duration_s: float,
          **extra) -> None:
    """One dataflux event per rotation phase — best-effort, never raises."""
    gb_dataflux.emit({"run_id": f"rotator-{os.getpid()}", "node": "host",
                      "label": "gb_rotator", "kind": "model_rotation",
                      "stage": f"rotator:{model}", "model": model,
                      "phase": phase, "status": status,
                      "duration_s": round(duration_s, 2), **extra})


def _endpoint_alive() -> bool:
    return _resolve_ollama_url() is not None


def _engine_built() -> bool:
    try:
        import gb_synapse
        return gb_synapse.engine_installed()
    except Exception:
        return False


def _ollama_keep_alive(model: str, keep_alive: str) -> None:
    """Ask the resolved Ollama-compatible endpoint to (un)load `model` —
    empty prompt, no tokens. keep_alive "10m" loads/refreshes; "0" unloads
    immediately."""
    url = _resolve_ollama_url() or _candidate_urls()[-1]
    body = json.dumps({"model": model, "keep_alive": keep_alive,
                       "stream": False}).encode()
    req = urllib.request.Request(f"{url}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()


def _serve(job: RotationJob) -> str:
    """Bring the job's model up; returns the mode used ("synapse"/"ollama")."""
    if _engine_built():
        import gb_synapse
        gb_synapse.serve(job.model, ctx=job.ctx, use_cluster=True)
        return "synapse"
    _ollama_keep_alive(job.model, "10m")
    return "ollama"


def _unload(job: RotationJob, mode: str) -> None:
    if mode == "synapse":
        import gb_synapse
        gb_synapse.stop(job.model)
    else:
        _ollama_keep_alive(job.model, "0")


def _run_job(job: RotationJob) -> bool:
    """serve → work → unload for one job, dataflux event per phase."""
    # Pre-gate (flux_health style): no candidate endpoint alive AND no
    # gb-synapse engine means there is nothing that can serve the model —
    # skip honestly.
    if not _dry() and not _endpoint_alive() and not _engine_built():
        _emit(job.model, "gate", "error", 0.0,
              error=f"none of {_candidate_urls()} reachable and gb-synapse "
                    "engine not built")
        return False

    mode = "dry"
    if not _dry():
        t0 = time.time()
        try:
            mode = _serve(job)
            _emit(job.model, "serve", "ok", time.time() - t0, mode=mode)
        except Exception as e:
            _emit(job.model, "serve", "error", time.time() - t0,
                  error=f"{type(e).__name__}: {e}")
            return False

    t0 = time.time()
    try:
        proc = subprocess.run(job.work, shell=True, timeout=job.timeout_s,
                              capture_output=True, text=True)
        ok = proc.returncode == 0
        _emit(job.model, "work", "ok" if ok else "error", time.time() - t0,
              returncode=proc.returncode,
              **({} if ok else {"error": (proc.stderr or proc.stdout)[-500:]}))
    except subprocess.TimeoutExpired:
        ok = False
        _emit(job.model, "work", "error", time.time() - t0,
              error=f"timeout after {job.timeout_s}s")

    if not _dry():
        t0 = time.time()
        try:
            _unload(job, mode)
            _emit(job.model, "unload", "ok", time.time() - t0, mode=mode)
        except Exception as e:
            # Unload failure doesn't fail the job — the work already ran.
            _emit(job.model, "unload", "error", time.time() - t0,
                  error=f"{type(e).__name__}: {e}")
    return ok


def _save_queue(jobs: list[RotationJob], queue_file: "str | None") -> None:
    if not queue_file:
        return
    tmp = Path(queue_file).with_suffix(".tmp")
    tmp.write_text(json.dumps({"jobs": [asdict(j) for j in jobs]}, indent=2))
    tmp.replace(queue_file)


def load_queue(queue_file: str) -> list[RotationJob]:
    data = json.loads(Path(queue_file).read_text())
    return [RotationJob(**j) for j in data["jobs"]]


def run_rotation(jobs: list[RotationJob],
                 queue_file: "str | None" = None) -> dict:
    """Run each job in sequence (one served model at a time). Jobs already
    marked done in `queue_file` are skipped; status is persisted after every
    job so a crashed/killed run resumes where it stopped."""
    consecutive = 0
    counts = {"done": 0, "failed": 0, "skipped": 0}
    for job in jobs:
        if job.status == "done":
            counts["skipped"] += 1
            continue
        if consecutive >= MAX_CONSECUTIVE_FAILURES:
            _emit(job.model, "abort", "error", 0.0,
                  error=f"{consecutive} consecutive failures — rotation aborted")
            break
        ok = _run_job(job)
        job.status = "done" if ok else "failed"
        counts["done" if ok else "failed"] += 1
        consecutive = 0 if ok else consecutive + 1
        _save_queue(jobs, queue_file)
    return {"jobs": len(jobs), **counts,
            "aborted": consecutive >= MAX_CONSECUTIVE_FAILURES}


_EXAMPLE_QUEUE = {"jobs": [
    # Reference Workload Rule (CLAUDE.md, 2026-07-13): satgeze/qwen36-35b-
    # uncensored-1m via GB-Synapse replaces qwen36-coder:studio / qwen35-
    # claude-coder:9b for every agentic/coding roster slot.
    {"model": "satgeze/qwen36-35b-uncensored-1m",
     "work": "echo 'coder batch here — pipeline consumes the gb-synapse port'",
     "timeout_s": 7200, "ctx": 16384, "status": "pending"},
    {"model": "qwen3-vl:30b",
     "work": "echo 'vision batch here — pipeline consumes the gb-synapse port'",
     "timeout_s": 7200, "ctx": 16384, "status": "pending"},
]}


def _cli_main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    verb, rest = argv[0], [a for a in argv[1:] if a != "--llm"]
    llm = "--llm" in argv[1:]

    if verb == "run":
        if not rest:
            print("usage: gb_rotator.py run <queue.json> [--llm]", file=sys.stderr)
            return 2
        jobs = load_queue(rest[0])
        result = run_rotation(jobs, queue_file=rest[0])
        print(json.dumps(result) if llm else
              f"rotation: {result['done']} done, {result['failed']} failed, "
              f"{result['skipped']} skipped" +
              ("  [ABORTED — consecutive failures]" if result["aborted"] else ""))
        return 1 if result["failed"] or result["aborted"] else 0
    elif verb == "example":
        print(json.dumps(_EXAMPLE_QUEUE, indent=2))
    else:
        print(f"unknown verb: {verb}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main(sys.argv[1:]))

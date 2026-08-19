"""Long-running commands that do not block the turn.

Why this earns its place HERE specifically: this CLI drives a local model at
roughly 5-15 tokens/second, so a single turn already costs tens of seconds of
real time. A `make`, a test run or a model pull that takes 90 seconds is not
an inconvenience on top of that , it is the whole turn, spent watching. Started
in the background, the same 90 seconds overlap with thinking that was going to
happen anyway.

The contract is deliberately small:

    start()   launch, return an id immediately
    output()  what it has printed so far, and whether it is still running
    stop()    kill it

Output is captured to a file rather than a pipe, because a pipe that nobody
drains will fill its buffer and block the very process this module exists to
keep unblocked , the failure mode is that the "background" job silently stops
making progress at ~64KB of output, which looks exactly like a hung build.

Reads are bounded by the same context budget the foreground path uses: a
background job is not a licence to spend the window.
"""
from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

#: Live jobs, per process. Not persisted: a job belongs to the session that
#: started it, and a pid from a previous run is not ours to report on or kill.
_JOBS: dict = {}

#: Hard ceiling on captured output per job, independent of the context budget.
#: A runaway `yes` must not fill the disk.
MAX_CAPTURE_BYTES = 8 * 1024 * 1024


@dataclass
class Job:
    id: str
    command: str
    proc: object
    log: Path
    started: float = field(default_factory=time.time)
    read_offset: int = 0

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    @property
    def exit_code(self):
        return self.proc.poll()


#: Finished jobs kept for later collection. Unattended-For-Days Must-Rule: a
#: dict that gains an entry per backgrounded command is unbounded over days,
#: and each entry pins a log file open. Running jobs are NEVER evicted , only
#: finished ones, oldest first, because a job still running is a job whose
#: output has not been collected yet.
MAX_FINISHED_JOBS = 200


def _reap() -> None:
    finished = [j for j in _JOBS.values() if not j.running]
    if len(finished) <= MAX_FINISHED_JOBS:
        return
    for j in sorted(finished, key=lambda j: j.started)[:len(finished) - MAX_FINISHED_JOBS]:
        try:
            j.log.unlink(missing_ok=True)
        except OSError:
            pass
        _JOBS.pop(j.id, None)


def start(command: str, cwd: str | None = None) -> str:
    """Launch `command` detached. Returns the job id."""
    jid = uuid.uuid4().hex[:8]
    log = Path(tempfile.gettempdir()) / f"gb_bg_{jid}.log"
    fh = open(log, "wb")
    proc = subprocess.Popen(
        command, shell=True, executable="/bin/bash",
        stdout=fh, stderr=subprocess.STDOUT,
        cwd=cwd or os.getcwd(),
        # Own process group, so stop() can kill the whole pipeline rather than
        # just the shell that spawned it and leaving orphans behind.
        start_new_session=True,
    )
    _JOBS[jid] = Job(id=jid, command=command, proc=proc, log=log)
    _reap()
    return jid


def _read_tail(job: Job, max_bytes: int) -> str:
    try:
        size = job.log.stat().st_size
    except OSError:
        return ""
    if size > MAX_CAPTURE_BYTES:
        size = MAX_CAPTURE_BYTES
    start_at = job.read_offset
    if size - start_at > max_bytes:
        start_at = size - max_bytes            # newest wins on a huge burst
    with job.log.open("rb") as fh:
        fh.seek(start_at)
        data = fh.read(size - start_at)
    job.read_offset = size
    return data.decode("utf-8", errors="replace")


def output(job_id: str, max_chars: int = 0) -> str:
    """New output since the last read, plus the job's state."""
    job = _JOBS.get(job_id)
    if job is None:
        known = ", ".join(sorted(_JOBS)) or "none"
        return (f"Error: no background job '{job_id}' in this session "
                f"(known: {known})")
    if not max_chars:
        try:
            from greenboost_cli.instruments.handlers import _ctx_char_budget
            max_chars = _ctx_char_budget(20_000)
        except Exception:
            max_chars = 20_000
    text = _read_tail(job, max_chars)
    if job.running:
        head = f"[{job_id} still running, {time.time() - job.started:.0f}s so far]"
    else:
        code = job.exit_code
        head = (f"[{job_id} finished after {time.time() - job.started:.0f}s, "
                f"exit code {code}]")
    return f"{head}\n{text}" if text else f"{head}\n(no new output)"


def stop(job_id: str) -> str:
    job = _JOBS.get(job_id)
    if job is None:
        return f"Error: no background job '{job_id}' in this session"
    if not job.running:
        return f"[{job_id} had already finished, exit code {job.exit_code}]"
    try:
        os.killpg(os.getpgid(job.proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            job.proc.terminate()
        except Exception:
            return f"Error: could not stop {job_id}"
    return f"[{job_id} stopped]"


def list_jobs() -> str:
    if not _JOBS:
        return "No background jobs in this session."
    rows = []
    for j in _JOBS.values():
        state = "running" if j.running else f"exit {j.exit_code}"
        rows.append(f"{j.id}  {state:<9} {time.time() - j.started:5.0f}s  {j.command[:70]}")
    return "\n".join(rows)

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
gb_lease.py , a VRAM-budget lease arbiter (userland, no root, no daemon).

Jobs that share one card declare a budget (VRAM, host RAM, CPU threads) and ask for a lease.
Jobs whose budgets fit in the card's PHYSICAL free memory may hold it together; a job that needs
tiering (virtual memory over the physical card) runs alone. The state is one JSON file that every
operation reads and rewrites under an flock, so there is nothing to start, restart or lose:
a crashed client leaves a record that the next operation prunes.

    arb = LeaseArbiter()                    # dir: GB_LEASE_DIR or ~/.cache/greenboost/lease
    r = arb.acquire("tsfm-train", vram_mib=4096, ram_mib=8000, cpu=4, priority=0,
                    idempotency_key="tsfm-train:1234:ab12", ttl=300)
    # r = {granted, rejected, lease_id, state, position, reason}; poll the same call until granted
    arb.release(r["lease_id"])

Rules (docs/room-architecture/budgeted-lease.md in the Trading repo is the design):
  * Admission is strict head-of-line by effective priority (priority + aging per minute waited,
    then arrival order). No backfill: a big job at the head is never starved by small ones.
  * Fit = sum(granted budgets) + request <= physical - reserve, AND request <= real free VRAM minus
    what granted jobs have not allocated yet. The real card only; the shim's virtual size is never used.
    A card that cannot be read admits one job at a time.
  * The queue is bounded (GB_LEASE_QUEUE_MAX, default 64). A full queue, or a budget that can never
    fit on the card, is rejected at once with a reason (copied from the model server's 503 behaviour).
  * Liveness: every record holds pid + /proc start time. A record whose pid is gone, or was reused by
    another process, is dropped before anything is decided. A holder that is alive is NEVER evicted,
    even past its ttl (it is reported `overdue`); a waiter that stops polling for `ttl` seconds is dropped.
  * cancel() is cooperative: it writes one word (stop or pause) into the holder's control file
    (the engine's jobs.py reads it at a safe point) and reports `cancelling`. No signal, no SIGSTOP.
  * Models a holder paused (the local model server) are recorded here so the LAST holder to leave
    resumes them, and no new grant is made while a holder is resuming them.

Arbiter round 2 (docs/room-architecture/gpu-arbiter.md in the Trading repo): one arbiter for every client.
  * `client` names the caller (`<repo>.<component>`); the TABLE below (CLIENTS) gives its class and base priority. A client cannot
    claim a priority. Unknown names get P9 and are logged (refused with UNKNOWN_CLIENT when GB_LEASE_STRICT_CLIENTS=1).
  * Tiers: interactive (P0) > scheduled (P1, P2) > batch (P3, P4, P9). Effective priority = class base + aging per minute waited, capped just
    below the next class up, so aging orders waiters inside a class and never lets a lower class pass a higher one. Anti-starvation: a waiter
    is promoted one class per GB_LEASE_STARVE_S (default 1800 s) waited, up to P1 (never into interactive), so nothing waits for ever.
  * P0 and P1 are solo classes: at most one holder of each at a time.
  * Fencing: every grant carries a monotonic `token`. renew/release/check with a token that is no longer the holder's are refused (STALE_TOKEN).
  * Per-client memory budgets (GB_LEASE_CLIENT_BUDGETS='{"invest.engine.job": 8192}' or the constructor): a request above its client's budget
    is rejected (CLIENT_BUDGET), one that would push the client's total over it waits (head-of-line is kept, so it blocks those behind it).
  * Every answer carries `code` (CODES) and, for a refusal, `retry_after_s`. `wait=False` is for background callers: they are never queued behind a holder.

Hardware figures are never hardcoded: the reserve is a percentage of the card (GB_LEASE_RESERVE_PCT,
default 5) with a small floor, the aging rate and queue length are env-overridable.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import errno
import os
import secrets
import stat as _stat
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

LEASE_VERSION = 1

# --- the client table (greenboost-coexistence.md 5.1): name -> (class, base priority, tier) --------------------------------------
CLIENTS = {
    "owner.interactive": ("P0", 100, "interactive"),
    "invest.engine.pass": ("P1", 80, "scheduled"),
    "invest.engine.job": ("P2", 50, "scheduled"),
    "invest.engine.embed": ("P2", 50, "scheduled"),
    "invest.engine.reader": ("P3", 40, "batch"),
    "optimal-claude.rag": ("P3", 40, "batch"),
    "greenboost.cli": ("P3", 40, "batch"),
    "optimal-claude.hook": ("P4", 10, "batch"),
    "greenboost.mcp": ("P4", 10, "batch"),
    "greenboost.bench": ("P4", 10, "batch"),
    # SP41-39: the other clients name themselves (docs/arbiter-clients.md)
    "greenboost.synapse.serve": ("P0", 100, "interactive"),   # a model served because the owner or a tool asked now
    "greenboost.cli.prompt": ("P0", 100, "interactive"),      # `gb -p` / the cli_prompt MCP tool: a prompt the owner typed
    "greenboost.rag.query": ("P0", 100, "interactive"),       # embedding of a search query the owner typed
    "greenboost.rag.index": ("P2", 50, "scheduled"),          # RAG indexing: embeds many chunks
    "greenboost.rotator": ("P2", 50, "scheduled"),            # gb_rotator serving a queued job's model
    "optimal-claude-history": ("P4", 10, "batch"),            # optimal-claude's batch history summariser
}
UNKNOWN_CLIENT = ("P9", 0, "batch")
SOLO_CLASSES = {"P0", "P1"}              # one holder at a time per class
_LEVELS = sorted({v[1] for v in CLIENTS.values()} | {UNKNOWN_CLIENT[1]}, reverse=True)

# The documented refusal codes. Every response has `code`; `granted` and `rejected` stay for version 1 readers.
CODES = {
    "GRANTED": "the lease is held; `token` is its fencing token",
    "QUEUED": "waiting in the queue; poll with the same idempotency_key",
    "BUSY_HOLDER": "wait=False and a holder (or the queue) is in the way; `retry_after_s` says when to ask again",
    "NO_FIT": "the budget can never fit the card; declare tiering to run alone",
    "QUEUE_FULL": "the bounded queue is full",
    "RAM_SHORT": "host RAM does not fit (raised by the engine client, which admits RAM itself)",
    "BLOCKED": "the owner's block flag (file `block` in the lease directory) is set",
    "UNKNOWN_CLIENT": "the client name is not in the table and GB_LEASE_STRICT_CLIENTS=1",
    "CLIENT_BUDGET": "the request is larger than the client's own memory budget",
    "BAD_REQUEST": "negative budget, non-positive ttl, or the pid is not running",
    "STALE_TOKEN": "the lease or fencing token is no longer the holder's (expired, dropped, replaced); stop at the next safe point",
    "STATE_UNREADABLE": "the state file is damaged; admission is refused until `python3 gb_lease.py repair`",
    "NO_LEASE": "a model load asked for outside any lease (the load gate's code)",
}
SETTLE_S = 120.0          # a grant with no per-pid reading counts as "not yet allocated" for this long


def lease_dir() -> Path:
    p = os.environ.get("GB_LEASE_DIR")
    return Path(p) if p else Path.home() / ".cache" / "greenboost" / "lease"


def proc_starttime(pid: int) -> Optional[int]:
    """Field 22 of /proc/<pid>/stat (clock ticks since boot): with the pid it names one process for good."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        return int(raw.rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def _env_f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def default_card() -> dict:
    """{physical_mib, free_mib, used_by_pid} from NVML; physical None when the card cannot be read."""
    try:
        from gb_nvml import get_nvml
        _u, free, total, _ = get_nvml(0).mem()
        return {"physical_mib": int(total) or None, "free_mib": int(free), "used_by_pid": None}
    except Exception:
        return {"physical_mib": None, "free_mib": None, "used_by_pid": None}


# --- safe file writes (SEC-02 / SEC-03): used by this module, gb_snapshot and gb_control -------------------------------
def safe_read_text(path: Path, limit: int = 1 << 22) -> Optional[str]:
    """Read a regular file without following a symlink at the last component; None when absent, a symlink or not a regular file."""
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        if not _stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        with os.fdopen(fd, "rb", closefd=False) as fh:
            return fh.read(limit).decode("utf-8", "replace")
    finally:
        os.close(fd)


def safe_atomic_write(dest, data, mode: int = 0o644) -> Path:
    """Write `data` (str or bytes) to `dest` so that a hostile neighbour of the directory cannot redirect it.

    The temp file has an unpredictable name and is created in the same directory with O_CREAT|O_EXCL|O_NOFOLLOW
    (an existing name or a symlink at that name is an error, never followed); the mode is set on the descriptor
    (fchmod), the data is fsynced, then renamed over `dest` (atomic on one filesystem). A `dest` that is a symlink
    or anything but a regular file is refused with OSError: rename would replace the link itself, but a planted
    link means someone else controls that name, so the writer stops and says so. Parent directories are not
    created here and a symlinked parent is allowed (a user may link ~/.cache elsewhere); the final name is what is checked."""
    dest = Path(dest)
    name = dest.name
    raw = data if isinstance(data, bytes) else str(data).encode()
    dfd = os.open(str(dest.parent), os.O_RDONLY | os.O_DIRECTORY)
    tmp_name = None
    try:
        try:
            if not _stat.S_ISREG(os.lstat(name, dir_fd=dfd).st_mode):
                raise OSError(errno.ELOOP, f"refusing to write {dest}: it is a symlink or not a regular file")
        except FileNotFoundError:
            pass
        for _ in range(8):
            cand = f".{name}.{secrets.token_hex(8)}.tmp"
            try:
                fd = os.open(cand, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, mode, dir_fd=dfd)
            except FileExistsError:
                continue
            tmp_name = cand
            break
        else:
            raise OSError(errno.EEXIST, f"could not create a temp file next to {dest}")
        try:
            os.fchmod(fd, mode)
            view = memoryview(raw)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.rename(tmp_name, name, src_dir_fd=dfd, dst_dir_fd=dfd)
        tmp_name = None
        return dest
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name, dir_fd=dfd)
            except OSError:
                pass
        os.close(dfd)


def read_block(path: Optional[Path] = None, now: Optional[float] = None,
               starttime: Optional[Callable[[int], Optional[int]]] = None) -> Optional[dict]:
    """The snapshot's `lease` block, read WITHOUT locking or writing (the snapshot writer must never
    change the arbiter's files). None when the arbiter has never been used (state file absent):
    that is how a reader learns the lease is not offered. Dead records are filtered in memory."""
    path = Path(path) if path else lease_dir() / "state.json"
    try:
        text = path.read_text()
    except OSError:
        return None
    try:
        st = json.loads(text)
        if not isinstance(st, dict):
            raise ValueError("not an object")
    except ValueError as exc:
        # present but unparseable: the state is UNKNOWN, not empty. All fields stay null (a reader keeps its own lock)
        # and `error` says why; version 1 readers ignore the extra key.
        return {"held": None, "holders": None, "budget_mib": None, "waiters": None, "last_wait_s": None,
                "error": f"lease state unreadable ({exc}); admission refused until `python3 gb_lease.py repair`"}
    now = time.time() if now is None else now
    starttime = starttime or proc_starttime
    live = [e for e in st.get("entries", []) if starttime(e.get("pid", -1)) == e.get("starttime")]
    granted = [e for e in live if e["state"] == "granted"]
    queued = sorted((e for e in live if e["state"] == "queued"), key=lambda e: e["seq"])
    return {"held": bool(granted), "budget_mib": sum(e["vram_mib"] for e in granted),
            "holders": [{"lease_id": e["lease_id"], "job": e["job"], "pid": e["pid"], "vram_mib": e["vram_mib"], "ram_mib": e.get("ram_mib"),
                         "since_s": round(now - e["t_grant"], 1), "tiering": e["tiering"], "overdue": now - e["seen"] > e["ttl"], "cancel": e.get("cancel")} for e in granted],
            "waiters": [{"job": e["job"], "vram_mib": e["vram_mib"], "priority": e["priority"], "wait_s": round(now - e["t_enq"], 1)} for e in queued],
            "last_wait_s": st.get("last_wait_s"), "queue_max": st.get("queue_max"), "rejects": st.get("rejects", 0), "version": LEASE_VERSION}


class LeaseStateUnreadable(Exception):
    """state.json exists but cannot be parsed (or read): the arbiter does not know who holds the card."""


def _unknown_state(fallback: Callable[[str], dict]):
    """Public operations answer a damaged state file with a refusal that carries the reason, never an exception and never a guess."""
    def deco(fn):
        import functools
        @functools.wraps(fn)
        def wrapper(self, *a, **kw):
            try:
                return fn(self, *a, **kw)
            except LeaseStateUnreadable as exc:
                return fallback(str(exc))
        return wrapper
    return deco


class LeaseArbiter:
    """All operations are short critical sections under one flock; `card`, `now` and `starttime` are injectable (tests use a fake card and fake processes)."""

    def __init__(self, path: Optional[Path] = None, card: Optional[Callable[[], dict]] = None, now: Optional[Callable[[], float]] = None,
                 starttime: Callable[[int], Optional[int]] = proc_starttime, queue_max: Optional[int] = None, aging_per_min: Optional[float] = None, client_budgets: Optional[dict] = None,
                 starve_s: Optional[float] = None):
        self.dir = Path(path) if path else lease_dir()
        self.card = card or default_card
        self.now = now or time.time
        self.starttime = starttime
        self.queue_max = int(queue_max if queue_max is not None else _env_f("GB_LEASE_QUEUE_MAX", 64))
        self.aging = aging_per_min if aging_per_min is not None else _env_f("GB_LEASE_AGING_PER_MIN", 1.0)
        self.starve_s = starve_s if starve_s is not None else _env_f("GB_LEASE_STARVE_S", 1800.0)
        if client_budgets is None:
            try:
                client_budgets = json.loads(os.environ.get("GB_LEASE_CLIENT_BUDGETS", "") or "{}")
            except ValueError:
                client_budgets = {}
        self.client_budgets = {str(k): int(v) for k, v in client_budgets.items()}

    # --- storage ---------------------------------------------------------------------------------------------------
    @contextlib.contextmanager
    def _txn(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "lock", "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                try:
                    st = json.loads((self.dir / "state.json").read_text())
                    if not isinstance(st, dict):
                        raise ValueError("not a JSON object")
                except FileNotFoundError:
                    st = {}
                except (OSError, ValueError) as exc:
                    # Never read a damaged file as "empty": that would grant the card to a second job while a holder may be alive.
                    raise LeaseStateUnreadable(f"lease state {self.dir / 'state.json'} is unreadable ({exc}); "
                                               f"admission refused, fix it or run `python3 gb_lease.py repair`") from exc
                st.setdefault("entries", [])
                st.setdefault("seq", 0)
                st.setdefault("paused", [])
                st.setdefault("events", [])
                st.setdefault("rejects", 0)
                st.setdefault("token", 0)
                st["queue_max"] = self.queue_max
                self._prune(st)
                yield st
                self._admit(st)
                safe_atomic_write(self.dir / "state.json", json.dumps(st, indent=1))
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _event(self, st: dict, text: str) -> None:
        st["events"] = (st["events"] + [{"t": round(self.now(), 1), "text": text}])[-20:]

    def _alive(self, e: dict) -> bool:
        return self.starttime(e.get("pid", -1)) == e.get("starttime")

    def _prune(self, st: dict) -> None:
        now, keep = self.now(), []
        for e in st["entries"]:
            if not self._alive(e):
                self._event(st, f"dropped {e['job']} ({e['state']}): pid {e['pid']} is gone or was reused")
            elif e["state"] == "queued" and now - e["seen"] > e["ttl"]:
                self._event(st, f"dropped waiting {e['job']}: no poll for {e['ttl']} s")
            else:
                keep.append(e)
        st["entries"] = keep
        if not any(e["state"] == "granted" for e in keep):
            if st["paused"]:
                self._event(st, f"models left paused with no holder: {st['paused']} (resume with gb synapse)")
            st["paused"] = []

    # --- admission -------------------------------------------------------------------------------------------------
    def _reserve(self, physical: int) -> int:
        return max(256, int(physical * _env_f("GB_LEASE_RESERVE_PCT", 5.0) / 100))

    def _eff(self, e: dict, now: float) -> float:
        waited = now - e["t_enq"]
        if e.get("cls") is None:                                   # a legacy caller with a numeric priority: unchanged
            return e["priority"] + self.aging * waited / 60.0
        i = _LEVELS.index(e["base"])
        if e["base"] < _LEVELS[0]:                                 # promotion: one class up per starve_s waited, never into the interactive class
            i = max(i - int(waited // self.starve_s), 1)
        eff = _LEVELS[i] + self.aging * waited / 60.0
        return min(eff, _LEVELS[i - 1] - 0.001) if i > 0 else eff

    def _fit(self, st: dict, e: dict, card: dict, now: float) -> Optional[str]:
        """None when `e` may start now, else the reason it waits."""
        granted = [g for g in st["entries"] if g["state"] == "granted"]
        if any(g.get("releasing") for g in granted):
            return "a holder is still giving the card back (resuming the model server)"
        if not granted:
            return None
        if e.get("cls") in SOLO_CLASSES:
            same = next((g for g in granted if g.get("cls") == e["cls"]), None)
            if same:
                return f"class {e['cls']} runs one holder at a time; held by {same['job']}"
        cb = self.client_budgets.get(e.get("client") or "")
        if cb is not None:
            used = sum(g["vram_mib"] for g in granted if g.get("client") == e["client"])
            if used + e["vram_mib"] > cb:
                return f"client {e['client']} budget is {cb} MiB and it already holds {used} MiB"
        if e["tiering"]:
            return f"needs tiering, which runs alone; held by {', '.join(g['job'] for g in granted)}"
        if any(g["tiering"] for g in granted):
            return f"{next(g['job'] for g in granted if g['tiering'])} uses tiering and runs alone"
        phys, free = card.get("physical_mib"), card.get("free_mib")
        if not phys:
            return "the card cannot be read, so it is shared by nobody (one job at a time)"
        room = phys - self._reserve(phys) - sum(g["vram_mib"] for g in granted)
        if e["vram_mib"] > room:
            return f"needs {e['vram_mib']} MiB, {max(room, 0)} MiB of the budget is unclaimed"
        if free is not None:
            by = card.get("used_by_pid")
            pending = sum(max(0, g["vram_mib"] - (by.get(g["pid"], 0) if by is not None else 0)) for g in granted
                          if by is not None or now - g["t_grant"] < SETTLE_S)
            if e["vram_mib"] > free - pending:
                return f"needs {e['vram_mib']} MiB, {max(free - pending, 0)} MiB is really free after what granted jobs have yet to allocate"
        return None

    def _admit(self, st: dict) -> None:
        now = self.now()
        queued = sorted((e for e in st["entries"] if e["state"] == "queued"), key=lambda e: (-self._eff(e, now), e["seq"]))
        if not queued:
            return
        card = self.card()
        blocked = None
        for e in queued:
            if blocked:
                e["reason"] = f"behind {blocked['job']}"
                continue
            why = self._fit(st, e, card, now)
            if why is None:
                st["token"] = st.get("token", 0) + 1
                e.update(state="granted", t_grant=now, reason=None, token=st["token"])
                st["last_wait_s"] = round(now - e["t_enq"], 1)
                self._event(st, f"granted {e['job']} {e['vram_mib']} MiB after {st['last_wait_s']} s")
                if e["tiering"]:
                    card = {**card, "physical_mib": None}            # nothing else is granted in this pass
            else:
                e["reason"], blocked = why, e

    @staticmethod
    def _view(st: dict, e: dict) -> dict:
        queued = sorted((x for x in st["entries"] if x["state"] == "queued"), key=lambda x: x["seq"])
        pos = 1 + [x["lease_id"] for x in queued].index(e["lease_id"]) if e["state"] == "queued" else 0
        return {"granted": e["state"] == "granted", "rejected": False, "lease_id": e["lease_id"], "state": e["state"], "position": pos, "reason": e.get("reason"),
                "code": "GRANTED" if e["state"] == "granted" else "QUEUED", "token": e.get("token")}

    # --- the contract: acquire, release, status ---------------------------------------------------------------------
    @_unknown_state(lambda why: {"granted": False, "rejected": True, "lease_id": None, "state": "rejected", "position": 0, "reason": why, "error": why, "code": "STATE_UNREADABLE", "token": None})
    def acquire(self, job: str, vram_mib: int, ram_mib: int = 0, cpu: int = 0, priority: float = 0, idempotency_key: Optional[str] = None,
                ttl: float = 300.0, pid: Optional[int] = None, tiering: bool = False, control_file: Optional[str] = None,
                client: Optional[str] = None, purpose: Optional[str] = None, model: Optional[str] = None, wait: bool = True,
                max_hold_s: Optional[float] = None) -> dict:
        """Grant, queue or reject. The same idempotency_key returns the same ticket (and counts as a poll), so a client polls by repeating the call.
        `client` picks the class and base priority from CLIENTS (then `priority` is ignored); `wait=False` never queues behind a holder (BUSY_HOLDER)."""
        pid = os.getpid() if pid is None else pid
        with self._txn() as st:
            if idempotency_key:
                for e in st["entries"]:
                    if e.get("key") == idempotency_key:
                        e["seen"] = self.now()
                        self._admit(st)
                        return self._view(st, e)
            def reject(code: str, why: str, retry: Optional[float] = None, holder: Optional[dict] = None) -> dict:
                st["rejects"] += 1
                self._event(st, f"rejected {client or job}: {code} {why}")
                return {"granted": False, "rejected": True, "lease_id": None, "state": "rejected", "position": 0, "reason": why, "code": code,
                        "retry_after_s": retry, "token": None, "holder": holder}
            if vram_mib < 0 or ttl <= 0:
                return reject("BAD_REQUEST", "budget and ttl must be positive")
            start = self.starttime(pid)
            if start is None:
                return reject("BAD_REQUEST", f"pid {pid} is not running")
            cls = base = tier = None
            if client:
                known = CLIENTS.get(client)
                if known is None and os.environ.get("GB_LEASE_STRICT_CLIENTS") == "1":
                    return reject("UNKNOWN_CLIENT", f"{client} is not in the client table")
                if known is None:
                    self._event(st, f"unknown client {client}: class P9")
                cls, base, tier = known or UNKNOWN_CLIENT
            if (self.dir / "block").exists() and client != "owner.interactive":
                return reject("BLOCKED", "the owner's block flag is set", 300.0)
            if sum(e["state"] == "queued" for e in st["entries"]) >= self.queue_max:
                return reject("QUEUE_FULL", f"queue is full ({self.queue_max} waiting)", 60.0)
            cb = self.client_budgets.get(client or "")
            if cb is not None and vram_mib > cb:
                return reject("CLIENT_BUDGET", f"{vram_mib} MiB is above the {cb} MiB budget of {client}")
            phys = (self.card() or {}).get("physical_mib")
            if phys and not tiering and vram_mib > phys - self._reserve(phys):
                return reject("NO_FIT", f"{vram_mib} MiB can never fit the card ({phys} MiB, {self._reserve(phys)} MiB reserved): declare tiering to run it alone")
            st["seq"] += 1
            now = self.now()
            e = {"lease_id": uuid.uuid4().hex[:12], "seq": st["seq"], "job": job, "vram_mib": int(vram_mib), "ram_mib": int(ram_mib), "cpu": int(cpu), "priority": base if cls else priority, "key": idempotency_key,
                 "pid": pid, "starttime": start, "ttl": float(ttl), "tiering": bool(tiering), "control_file": control_file, "state": "queued", "t_enq": now, "t_grant": None, "seen": now, "reason": None,
                 "client": client, "cls": cls, "base": base, "tier": tier, "purpose": purpose, "model": model, "max_hold_s": max_hold_s, "token": None}
            st["entries"].append(e)
            self._admit(st)
            if e["state"] != "granted" and not wait:
                st["entries"] = [x for x in st["entries"] if x is not e]
                held = next((g for g in st["entries"] if g["state"] == "granted"), None)
                left = (held["max_hold_s"] - (now - held["t_grant"])) if held and held.get("max_hold_s") else None
                return reject("BUSY_HOLDER", f"{(held or {}).get('client') or (held or {}).get('job') or 'a holder'} is in the way: {e.get('reason')}", round(max(30.0, left), 0) if left is not None else 60.0,
                              {"client": held.get("client") or held["job"], "lease_id": held["lease_id"]} if held else None)
            return self._view(st, e)

    @_unknown_state(lambda why: {"ok": False, "code": "STATE_UNREADABLE", "error": why})
    def check(self, lease_id: str, token: int) -> dict:
        """Fencing check for a worker: ok only while this lease is still granted AND `token` is the one it was granted. A dropped, released or replaced lease answers STALE_TOKEN."""
        with self._txn() as st:
            ok = any(e["lease_id"] == lease_id and e["state"] == "granted" and e.get("token") == token for e in st["entries"])
        return {"ok": ok, "code": "GRANTED" if ok else "STALE_TOKEN"}

    @_unknown_state(lambda why: {"ok": False, "error": why})
    def renew(self, lease_id: str, token: Optional[int] = None) -> dict:
        """The holder says it is alive (resets `overdue`). With `token`, a stale one is refused. Returns {ok, code}."""
        with self._txn() as st:
            for e in st["entries"]:
                if e["lease_id"] == lease_id:
                    if token is not None and e.get("token") != token:
                        return {"ok": False, "code": "STALE_TOKEN"}
                    e["seen"] = self.now()
                    return {"ok": True, "code": "GRANTED" if e["state"] == "granted" else "QUEUED"}
        return {"ok": False, "code": "STALE_TOKEN"}

    @_unknown_state(lambda why: {"resume": [], "error": why})
    def begin_release(self, lease_id: str) -> dict:
        """First half of a release for a holder that must give models back: {resume: [models]} is non-empty only for the LAST holder.
        While it is resuming, nothing new is granted. Call release() afterwards."""
        with self._txn() as st:
            me = next((e for e in st["entries"] if e["lease_id"] == lease_id and e["state"] == "granted"), None)
            if me is None:
                return {"resume": []}
            me["releasing"] = True
            others = [e for e in st["entries"] if e["state"] == "granted" and not e.get("releasing")]
            models = [] if others else list(st["paused"])
            if not others:
                st["paused"] = []
            return {"resume": models}

    @_unknown_state(lambda why: {"ok": False, "resume": [], "error": why})
    def release(self, lease_id: str, token: Optional[int] = None) -> dict:
        """Frees the budget (granted or queued); unknown ids are fine. Returns {ok, resume}: `resume` is what begin_release would have said, for a caller that did not call it."""
        resume: list = []
        with self._txn() as st:
            me = next((e for e in st["entries"] if e["lease_id"] == lease_id), None)
            if me is None:
                return {"ok": True, "resume": []}
            if token is not None and me.get("token") != token:
                return {"ok": False, "resume": [], "code": "STALE_TOKEN"}
            st["entries"] = [e for e in st["entries"] if e["lease_id"] != lease_id]
            if me["state"] == "granted" and not me.get("releasing") and not any(e["state"] == "granted" for e in st["entries"]):
                resume, st["paused"] = list(st["paused"]), []
        return {"ok": True, "resume": resume}

    @_unknown_state(lambda why: None)
    def note_paused(self, lease_id: str, models: list) -> None:
        """Record models this holder paused, so the last holder resumes them."""
        with self._txn() as st:
            if any(e["lease_id"] == lease_id for e in st["entries"]):
                st["paused"] = sorted(set(st["paused"]) | set(models))

    @_unknown_state(lambda why: {"ok": False, "state": why, "released": False, "error": why})
    def cancel(self, lease_id: str, kind: str = "stop", wait_s: float = 0.0, sleep: Callable[[float], None] = time.sleep) -> dict:
        """Cooperative. A waiter is removed. A holder is asked to `stop` or `pause` through its control file at its next safe point; the budget stays
        held until it releases or dies. Never a signal, never SIGSTOP. With wait_s it polls for the release. Returns {ok, state, released}."""
        if kind not in ("stop", "pause"):
            raise ValueError("kind is stop or pause")
        with self._txn() as st:
            me = next((e for e in st["entries"] if e["lease_id"] == lease_id), None)
            if me is None:
                return {"ok": False, "state": "unknown", "released": True}
            if me["state"] == "queued":
                st["entries"] = [e for e in st["entries"] if e["lease_id"] != lease_id]
                return {"ok": True, "state": "cancelled", "released": True}
            cf = me.get("control_file")
            if not cf:
                return {"ok": False, "state": "no control file: the holder cannot be asked, wait for it or stop it with `invest jobs stop`", "released": False}
            safe_atomic_write(cf, kind)
            me["cancel"] = kind
            self._event(st, f"asked {me['job']} to {kind} through its control file")
        t0 = self.now()
        while wait_s > 0 and self.now() - t0 < wait_s:
            sleep(1.0)
            if not any(e["lease_id"] == lease_id for e in (self.status().get("holders") or [])):
                return {"ok": True, "state": "released", "released": True}
        return {"ok": True, "state": "cancelling", "released": False}

    @_unknown_state(lambda why: {"held": None, "budget_mib": None, "holders": None, "waiters": None, "error": why, "version": LEASE_VERSION})
    def status(self) -> dict:
        with self._txn() as st:
            now = self.now()
            queued = sorted((e for e in st["entries"] if e["state"] == "queued"), key=lambda e: (-self._eff(e, now), e["seq"]))
            card = self.card()
            return {"held": any(e["state"] == "granted" for e in st["entries"]), "budget_mib": sum(e["vram_mib"] for e in st["entries"] if e["state"] == "granted"),
                    "holders": [{"lease_id": e["lease_id"], "job": e["job"], "pid": e["pid"], "vram_mib": e["vram_mib"], "ram_mib": e["ram_mib"], "since_s": round(now - e["t_grant"], 1),
                                 "client": e.get("client"), "class": e.get("cls"), "token": e.get("token"), "model": e.get("model"), "purpose": e.get("purpose"), "tiering": e["tiering"], "overdue": now - e["seen"] > e["ttl"], "cancel": e.get("cancel")} for e in st["entries"] if e["state"] == "granted"],
                    "waiters": [{"lease_id": e["lease_id"], "job": e["job"], "client": e.get("client"), "class": e.get("cls"), "vram_mib": e["vram_mib"], "priority": e["priority"], "effective": round(self._eff(e, now), 3), "wait_s": round(now - e["t_enq"], 1), "reason": e.get("reason")} for e in queued],
                    "last_wait_s": st.get("last_wait_s"), "queue_max": self.queue_max, "rejects": st["rejects"], "paused": list(st["paused"]), "token": st.get("token", 0), "events": st["events"][-5:],
                    "physical_mib": card.get("physical_mib"), "reserve_mib": self._reserve(card["physical_mib"]) if card.get("physical_mib") else None, "version": LEASE_VERSION}


if __name__ == "__main__":        # python3 gb_lease.py init|status|cancel <lease_id> [stop|pause]
    import sys
    a = LeaseArbiter()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "repair":            # a damaged state.json is moved aside (kept for inspection) and the arbiter starts empty
        sp = a.dir / "state.json"
        if sp.exists():
            aside = a.dir / f"state.json.damaged-{int(time.time())}"
            os.rename(sp, aside)
            print(f"moved {sp} to {aside}")
        print("repaired: the next operation starts with an empty state")
    elif cmd == "init":              # create the (empty) state file: from then on the snapshot says the budgeted lease is offered
        with a._txn():
            pass
        print(a.dir / "state.json")
    elif cmd == "cancel" and len(sys.argv) > 2:
        print(json.dumps(a.cancel(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "stop"), indent=1))
    else:
        print(json.dumps(a.status(), indent=1))

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
gb_lease_client.py , the one way a GreenBoost-side or sibling-repo client takes the card through gb_lease.LeaseArbiter.

    with gb_lease_client.lease("greenboost.rag.index", purpose="embed 40 chunks", wait_s=120):
        ...load or call the model...

A refusal raises LeaseRefused (a RuntimeError) whose text is a plain sentence: what is in the way, what it costs, when to ask again.
A lease is released on exit, also on an exception. Docs: the maintainer notes

  * wait_s=0 never queues behind a holder (the arbiter's wait=False): for background callers, which skip when the card is busy.
  * wait_s>0 polls the same ticket once a second and gives up with BUSY_HOLDER after wait_s.
  * Inside a lease already held by this process tree (env GB_LEASE_HELD) nothing is asked again: a child process or a nested call does not deadlock on its parent.
  * GB_LEASE_OFF=1 skips the arbiter (a test or an owner who wants the old behaviour). An arbiter that cannot be imported or whose state is unreadable
    is never a reason to block a model call that worked before; the fallback is logged to stderr. A real refusal always raises.
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
import uuid
from typing import Callable, Optional

HELD_ENV = "GB_LEASE_HELD"

PLAIN = {
    "BUSY_HOLDER": "the GPU is busy with {who}; nothing was loaded and nothing is wrong. Ask again in about {retry} s.",
    "QUEUE_FULL": "too many jobs are already waiting for the GPU; nothing was loaded. Ask again in about {retry} s.",
    "BLOCKED": "the owner has kept the GPU free (block flag in the lease directory); nothing was loaded. Delete the file `block` there to lift it.",
    "NO_FIT": "this job can never fit the card next to the reserve; nothing was loaded. Declare tiering or pick a smaller model.",
    "CLIENT_BUDGET": "this client asked for more VRAM than its own budget allows (GB_LEASE_CLIENT_BUDGETS); nothing was loaded.",
    "UNKNOWN_CLIENT": "the client name is not in the arbiter table (gb_lease.CLIENTS); nothing was loaded.",
    "STATE_UNREADABLE": "the arbiter's state file is damaged; nothing was loaded. Run `python3 gb_lease.py repair`.",
    "BAD_REQUEST": "the arbiter refused the request as malformed; nothing was loaded.",
}


class LeaseRefused(RuntimeError):
    def __init__(self, code: str, message: str, retry_after_s: Optional[float] = None, holder: Optional[dict] = None):
        super().__init__(message)
        self.code, self.retry_after_s, self.holder = code, retry_after_s, holder


def plain(r: dict) -> str:
    code = r.get("code") or "REFUSED"
    who = (r.get("holder") or {}).get("client") or "another job"
    retry = int(r.get("retry_after_s") or 60)
    t = PLAIN.get(code)
    return f"GPU lease refused ({code}): " + (t.format(who=who, retry=retry) if t else f"{r.get('reason') or 'no reason given'}; nothing was loaded.")


def _arbiter():
    import gb_lease
    return gb_lease.LeaseArbiter()


@contextlib.contextmanager
def lease(client: str, purpose: str = "", *, vram_mib: int = 0, wait_s: float = 0.0, max_hold_s: Optional[float] = None, model: Optional[str] = None,
          arbiter=None, sleep: Optional[Callable[[float], None]] = None, now: Optional[Callable[[], float]] = None):
    """Hold the card for the `with` body. Yields {lease_id, token, skipped}. Raises LeaseRefused. `vram_mib` 0 means "budget not declared": the class rules (solo P0/P1, tiering, block flag) still apply."""
    if os.environ.get("GB_LEASE_OFF") == "1" or os.environ.get(HELD_ENV):
        yield {"lease_id": None, "token": None, "skipped": "held by a parent" if os.environ.get(HELD_ENV) else "GB_LEASE_OFF"}
        return
    try:
        arb = arbiter or _arbiter()
    except Exception as exc:                                  # arbiter missing: log and carry on, as before this wiring
        print(f"gb lease: arbiter unavailable ({exc}); {client} runs unleased", file=sys.stderr)
        yield {"lease_id": None, "token": None, "skipped": "arbiter unavailable"}
        return
    sleep, now = sleep or time.sleep, now or time.monotonic
    key = f"{client}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    ask = dict(job=client, vram_mib=vram_mib, idempotency_key=key, client=client, purpose=purpose or None, model=model, wait=wait_s > 0, max_hold_s=max_hold_s)
    t0 = now()
    r = arb.acquire(**ask)
    while not r.get("granted") and not r.get("rejected") and now() - t0 < wait_s:
        sleep(1.0)
        r = arb.acquire(**ask)
    if r.get("code") == "STATE_UNREADABLE":
        print(f"gb lease: {r.get('reason')}; {client} runs unleased", file=sys.stderr)
        yield {"lease_id": None, "token": None, "skipped": "state unreadable"}
        return
    if not r.get("granted"):
        if r.get("lease_id"):
            arb.release(r["lease_id"])                        # we stopped waiting: leave the queue
            r = {**r, "code": "BUSY_HOLDER", "retry_after_s": r.get("retry_after_s") or 60}
        raise LeaseRefused(r.get("code") or "REFUSED", plain(r), r.get("retry_after_s"), r.get("holder"))
    lid, token = r["lease_id"], r.get("token")
    prev = os.environ.get(HELD_ENV)
    os.environ[HELD_ENV] = lid                                # children (subprocess env) and nested calls see it
    try:
        yield {"lease_id": lid, "token": token, "skipped": None}
    finally:
        if prev is None:
            os.environ.pop(HELD_ENV, None)
        else:
            os.environ[HELD_ENV] = prev
        try:
            arb.release(lid, token=token)
        except Exception as exc:
            print(f"gb lease: release of {lid} failed ({exc}); the arbiter drops it when this process exits", file=sys.stderr)


def unload_blocked(model: str, *, arbiter=None) -> Optional[dict]:
    """The holder that makes unloading `model` wrong right now, or None when the unload may go ahead. An unload path (the supervisor's idle reclaim, gb_reclaim, an engine
    pre-flight) asks this before it sends keep_alive=0 or kills a server: a live lease (not overdue) that names the model, or names none (it is using the card), blocks it.
    Fails open like lease(): GB_LEASE_OFF=1, a lease held by this process tree, an arbiter that cannot be read, all return None, so an unload that worked before still works."""
    if os.environ.get("GB_LEASE_OFF") == "1" or os.environ.get(HELD_ENV) or not model:
        return None
    try:
        holders = (arbiter or _arbiter()).status().get("holders", [])
    except Exception as exc:
        print(f"gb lease: cannot read the arbiter ({exc}); unload of {model} not checked", file=sys.stderr)
        return None
    want = model.split(":")[0].lower()
    for h in holders:
        if h.get("overdue"):
            continue
        held = (h.get("model") or "").split(":")[0].lower()
        if not held or held == want:
            return h
    return None

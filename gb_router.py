#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_router.py , pick the cheapest LOCAL model that can serve a turn.

Why this exists
---------------
Measured on this box 2026-08-18, same hardware, same day:

    Qwen3.6-35B-A3B (MoE) + expert offload    45.72 tok/s
    Qwen3.8-27B     (dense)                    2.7-3.5 tok/s

That is a ~15x spread, and it is not a quality ranking , it is an architecture
one. The MoE activates 8 of 256 experts per token so most of its weight never
crosses PCIe; the dense model reads every parameter every token and is pinned by
the bus. Qwen3.8-27B is genuinely the stronger model (SWE-bench Pro 61.7 vs
49.5, Terminal-Bench 73.0 vs 51.5), and there is no Qwen3.8 MoE to have both.

So the choice is real and it recurs every turn. Most agentic turns , run a
command, read a file, apply an edit, emit a tool call , do not need the stronger
model. Planning, architecture and debugging arguments do. Routing per turn
avoids picking one model for a whole session and paying for that choice on every
turn where it was wrong.

Modelled on NemoClaw's `router/pool-config.yaml`, with two deliberate changes:

  * **Every pool entry must be local.** NemoClaw's own pool points at
    `integrate.api.nvidia.com`; the mechanism is worth copying, the targets are
    not. `check_pool_is_local()` refuses a remote entry outright, per CLAUDE.md's
    Local-First Must-Rule.
  * **Heuristics before a learned router.** NemoClaw runs a trained 0.8B encoder
    over the prompt. That is a model on the critical path of every request, and
    on a box where the whole problem is that models are expensive, adding one
    before proving routing pays would be backwards. The signals below are cheap
    and inspectable; swap in a classifier once the logs show heuristics losing.

Nothing here dispatches. `classify()` and `route()` are pure functions over the
request, so they are testable without a GPU and cannot themselves become a
latency cost.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# Turn classes, cheapest first.
FAST = "fast"     # tool calls, edits, short factual answers
DEEP = "deep"     # planning, architecture, debugging, long-horizon reasoning

# Words that genuinely indicate a reasoning turn rather than a mechanical one.
# Deliberately narrow: a false DEEP costs ~15x throughput, so the bar to escalate
# is "the user is asking for judgement", not "the text is long".
_DEEP_PATTERNS = re.compile(
    r"\b(why|design|architect(ure)?|trade[- ]?off|compare|evaluate|refactor|"
    r"root cause|debug|diagnos|explain how|strategy|plan\b|approach|"
    r"should (i|we)|pros and cons|alternatives?)\b", re.I)

# Mechanical intents. Present tense imperatives dominate agentic tool turns.
_FAST_PATTERNS = re.compile(
    r"\b(run|list|show|cat|read|open|grep|find|rename|move|copy|delete|"
    r"install|format|lint|commit|status|print|add|append)\b", re.I)


@dataclass
class Pool:
    """Local models available to route between, cheapest first."""
    fast: str = ""
    deep: str = ""
    endpoints: dict = field(default_factory=dict)   # model -> base_url


def _is_local_url(url: str) -> bool:
    """Reuse the semantic layer's own definition so the two cannot drift.

    A feeder on the owner's LAN counts as local , the cluster fabric is the
    point of the project, not an escape from it.
    """
    try:
        import gb_semantics
        host = (url or "").split("//")[-1].split("/")[0].split(":")[0]
        return gb_semantics._is_local_host(host)
    except Exception:
        # Fail CLOSED: an unverifiable endpoint is treated as remote rather
        # than waved through, because the cost of a wrong "local" is that the
        # rule silently stops applying.
        return False


def check_pool_is_local(pool: Pool) -> list:
    """Endpoints in `pool` that are not local. Empty list means compliant."""
    return [f"{m}={u}" for m, u in (pool.endpoints or {}).items()
            if not _is_local_url(u)]


def classify(messages: list, tools: "list | None" = None,
             tool_result_chars: int = 0) -> str:
    """FAST or DEEP for this turn, from the request alone.

    Ordered so the cheapest decisive signal wins. The bias is toward FAST:
    escalating costs ~15x throughput, while a FAST turn that needed DEEP is
    usually visible immediately and can be re-asked.
    """
    last_user = ""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content")
            last_user = c if isinstance(c, str) else " ".join(
                p.get("text", "") for p in (c or []) if isinstance(p, dict))
            break

    # An explicit reasoning request wins outright — this is the user asking for
    # judgement, which is exactly what the stronger model is for.
    if _DEEP_PATTERNS.search(last_user):
        return DEEP

    # A turn whose input is dominated by tool output is a continuation: the model
    # is reading a file or command result, not reasoning from scratch. These are
    # the bulk of an agentic session and the reason routing pays at all.
    if tool_result_chars > 2000 and not _DEEP_PATTERNS.search(last_user):
        return FAST

    if _FAST_PATTERNS.search(last_user) and len(last_user) < 240:
        return FAST

    # A long prompt with no reasoning cue is usually pasted context, not a hard
    # question. Length alone is a poor proxy for difficulty and is used last.
    if len(last_user) > 1200:
        return DEEP

    return FAST


def route(messages: list, pool: "Pool | None" = None, tools: "list | None" = None,
          tool_result_chars: int = 0) -> dict:
    """Choose a model. Returns the decision AND why, so a bad route is debuggable.

    Falls back to whichever model the pool actually has: a pool with only one
    model is valid and routes everything to it rather than failing.
    """
    pool = pool or pool_from_env()
    turn = classify(messages, tools, tool_result_chars)
    remote = check_pool_is_local(pool)
    if remote:
        # Refuse rather than silently serving from a cloud endpoint.
        raise ValueError(
            f"router pool contains non-local endpoints: {', '.join(remote)}. "
            f"CLAUDE.md's Local-First Must-Rule forbids routing inference off "
            f"this hardware; fix the pool or remove those entries.")
    chosen = (pool.deep or pool.fast) if turn == DEEP else (pool.fast or pool.deep)
    return {"turn_class": turn, "model": chosen,
            "endpoint": (pool.endpoints or {}).get(chosen, ""),
            "reason": _why(turn, messages, tool_result_chars)}


def _why(turn: str, messages: list, tool_result_chars: int) -> str:
    if turn == DEEP:
        return "reasoning cue in the request, or long prompt with no mechanical intent"
    if tool_result_chars > 2000:
        return "continuation over tool output — reading, not reasoning"
    return "mechanical intent, short request"


def pool_from_env() -> Pool:
    """Pool from env, defaulting both roles to whatever is served now.

    Env-driven rather than a config file for the first cut: routing has to prove
    it pays before it earns a schema.
    """
    fast = os.environ.get("GB_ROUTER_FAST", "").strip()
    deep = os.environ.get("GB_ROUTER_DEEP", "").strip()
    base = os.environ.get("GB_ROUTER_ENDPOINT", "http://127.0.0.1:11369/v1").strip()
    if not (fast or deep):
        try:
            import gb_synapse
            served = [s["model"] for s in gb_synapse.ps()]
            fast = deep = served[0] if served else ""
        except Exception:
            pass
    ep = {m: base for m in (fast, deep) if m}
    return Pool(fast=fast, deep=deep, endpoints=ep)


def route_and_emit(messages: list, pool: "Pool | None" = None,
                   tools: "list | None" = None, tool_result_chars: int = 0) -> dict:
    """route() plus a dataflux event , the Observability Must-Rule's shape.

    Kept separate from route() on purpose: route() is a pure function over the
    request so it stays testable without a GPU or a log, matching
    gb_placement.plan_and_emit()/plan_experts_and_emit(). Emitting is
    best-effort and never raises, so telemetry cannot break a turn.

    The event is what makes a bad route debuggable after the fact. Without
    `turn_class` and `reason` recorded, "the wrong model answered that" is
    unfalsifiable, and a routing heuristic nobody can audit is worse than none.
    """
    decision = route(messages, pool=pool, tools=tools,
                     tool_result_chars=tool_result_chars)
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "node": "host", "label": "gb_router", "kind": "route_decision",
            "turn_class": decision["turn_class"], "model": decision["model"],
            "reason": decision["reason"],
            "tool_result_chars": tool_result_chars,
        })
    except Exception:
        pass
    return decision

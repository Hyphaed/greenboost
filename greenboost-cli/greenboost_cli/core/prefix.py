"""Which part of the prompt moved, and what it cost.

The measurement this exists to serve: over 14 days of real sessions on this
box, a turn that reused >=99% of its prompt prefix reached first token in
~5.5 s, and one that reused under 50% took ~166 s. Same model, same hardware,
30x apart, decided entirely by whether the leading tokens were byte-identical
to the previous request.

The engine cannot help here. `--cache-reuse` , llama.cpp's partial-prefix
reuse, which tolerates a gap , is confirmed rejected for this hybrid
architecture (CLAUDE.md; `gb_synapse._CACHE_REUSE_REJECT`), so reuse is strictly
all-or-nothing from the first differing token. Position-independent caching as
the literature describes it (EPIC, CacheBlend) is a change to the serving
system, not something a client can bolt on.

What a client CAN do is stop causing the invalidation, and that needs the cost
to be attributable. Before this module, a compaction, a re-rendered system
block and a changed tool schema were indistinguishable: all three showed up as
"the turn was slow". This splits the assembled prompt into named chunks, hashes
each one, and reports WHICH chunk moved first , because that chunk is the one
that threw away everything after it.

Cheap by construction: hashing a few kilobytes per turn against a turn that
costs seconds to minutes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

#: Chunks in the order they reach the engine. The first one that changes
#: invalidates every chunk after it, so order IS the diagnosis.
CHUNK_ORDER = ("system", "tools", "history", "memory", "plan", "user")


def _h(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", "replace")).hexdigest()[:16]


@dataclass
class PrefixState:
    """Last turn's chunk fingerprints, and what changed since."""
    chunks: dict = field(default_factory=dict)
    turns: int = 0

    def observe(self, chunks: dict) -> dict:
        """Fingerprint this turn's chunks and report the first that moved.

        Returns a verdict dict; `first_changed` is None on the first turn (no
        previous request to compare against , not the same thing as "nothing
        changed") and on a turn where the prefix held.
        """
        new = {name: _h(chunks.get(name, "")) for name in CHUNK_ORDER}
        sizes = {name: len(chunks.get(name, "") or "") for name in CHUNK_ORDER}
        prev, self.chunks = self.chunks, new
        self.turns += 1
        if not prev:
            return {"first_turn": True, "first_changed": None,
                    "stable_prefix_chars": 0, "invalidated_chars": 0,
                    "chunks": list(CHUNK_ORDER)}

        first = None
        stable = 0
        for name in CHUNK_ORDER:
            if prev.get(name) != new[name]:
                first = name
                break
            stable += sizes[name]
        # "Invalidated" means chars that WERE reusable and now are not. A
        # changed trailing user turn is new content that was never in the cache
        # , counting it here would report a cost on every healthy turn and make
        # the number meaningless.
        if first == CHUNK_ORDER[-1]:
            invalidated = 0
        else:
            invalidated = sum(sizes[n] for n in CHUNK_ORDER) - stable - sizes[CHUNK_ORDER[-1]]
        changed = [n for n in CHUNK_ORDER if prev.get(n) != new[n]]
        return {
            "first_turn": False,
            "first_changed": first,
            "changed": changed,
            "stable_prefix_chars": stable,
            "invalidated_chars": max(0, invalidated) if first else 0,
            "chunks": list(CHUNK_ORDER),
        }


_state = PrefixState()


def observe(chunks: dict, emit: bool = True) -> dict:
    """Module-level entry point. Emits `agent_prefix_shift` when a chunk moved.

    Only emits when something ACTUALLY moved and it was not the trailing user
    turn , a new user message at the end is the normal, healthy case and
    invalidates nothing before it. Emitting on that would bury the real signal
    in noise at one event per turn.
    """
    verdict = _state.observe(chunks)
    first = verdict.get("first_changed")
    if emit and first and first != CHUNK_ORDER[-1]:
        try:
            import gb_dataflux
            gb_dataflux.emit({
                "kind": "agent_prefix_shift", "status": "warn",
                "first_changed": first,
                "changed": ",".join(verdict.get("changed", [])),
                "stable_prefix_chars": verdict["stable_prefix_chars"],
                "invalidated_chars": verdict["invalidated_chars"],
                "turns": _state.turns,
            })
        except Exception:
            pass
    return verdict


def reset() -> None:
    global _state
    _state = PrefixState()


def explain(verdict: dict) -> str:
    """One line a human can act on, or "" when there is nothing to say."""
    first = verdict.get("first_changed")
    if verdict.get("first_turn") or not first:
        return ""
    if first == CHUNK_ORDER[-1]:
        return ""                      # a new user turn: normal, costs nothing
    return (f"prefix shifted at '{first}' , {verdict['invalidated_chars']:,} "
            f"chars after it must be re-prefilled ({verdict['stable_prefix_chars']:,} "
            f"still reusable)")

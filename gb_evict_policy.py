#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
#
# Eviction policies for GB-Tiering.
#
# Algorithm ported from NVIDIA TensorRT-LLM, Apache-2.0:
#   tensorrt_llm/runtime/kv_cache_manager_v2/_eviction_controller/
#       _eviction_controller.py
#   upstream commit 828e765a78 (2026-08-21)
#   Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
# Ported, not copied, and the difference is deliberate. Upstream is
# page-granular: thousands of KV pages, an `llist` C-extension doubly-linked
# list for O(1) removal, mypyc `__slots__` annotations, and LifeCycleId /
# PoolGroupIndex concepts that only mean something for KV pages. GB-Tiering is
# module-granular - UNet, VAE, text encoders, a few dozen entries at most - so
# the linked list buys nothing, the external dependency costs something, and
# the pool-group machinery has no analogue. What transfers is the part that
# matters: priority as a first-class dimension alongside recency, a swappable
# EvictionPolicy protocol, and one controller per tier.
#
# Priority constants keep upstream's numeric values (0..100, default 35) so
# the semantics stay diffable against the source if it is ever re-synced.
#
# LOWER priority is evicted FIRST. Within one priority, least-recently-used is
# evicted first. With every entry left at PRIORITY_DEFAULT - which is what the
# default configuration does - this degenerates to exactly the plain LRU order
# that ModelTierManager._lru_in_tier() produced before, by construction. That
# equivalence is what makes this safe to land ahead of any priority assignment,
# and it is asserted in tests/test_gb_evict_policy.py.
"""GB-Tiering eviction policies: prioritized LRU, per tier.

Replaces the single-dimension `sorted(last_used)[0]` selection that
`ModelTierManager` used, which could only ever answer "what was touched
longest ago" and had no way to express "this is the KV cache, evict it last".
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Protocol

# Upstream numeric values, kept so the two are comparable.
PRIORITY_MIN = 0
PRIORITY_MAX = 100
PRIORITY_DEFAULT = 35


class Evictable(Protocol):
    """What a policy needs from a thing it may evict.

    Deliberately narrower than upstream's EvictablePage: no cache_level (the
    controller already knows its tier), no life_cycle (no pool groups here),
    no PageStatus (module entries are not partially committed).
    """

    name: str
    tier: str
    last_used: float
    priority: int


class EvictionPolicy(Protocol):
    """Swappable selection strategy for one (tier, priority) bucket."""

    def push(self, item: Evictable, evict_first: bool = False) -> None: ...

    def pop(self) -> Evictable: ...

    def remove(self, item: Evictable) -> Evictable: ...

    def peek(self) -> "Evictable | None": ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[Evictable]: ...


class LRUEvictionPolicy:
    """Least-recently-used first.

    List-backed rather than linked-list-backed. Upstream uses `llist` because
    it removes arbitrary nodes from a queue of thousands; here the queue holds
    a handful of model components, so a list keeps the dependency count at zero
    and the cost immeasurable.

    Order is resolved at pop time from each item's live `last_used`, not frozen
    at push time. That matters because callers mutate `last_used` directly on
    the entry (ModelTierManager does, on every promote), and a policy that
    cached insertion order would silently go stale.
    """

    __slots__ = ("_items", "_evict_first")

    def __init__(self) -> None:
        self._items: List[Evictable] = []
        self._evict_first: List[Evictable] = []

    def push(self, item: Evictable, evict_first: bool = False) -> None:
        if any(i is item for i in self._items) or any(i is item for i in self._evict_first):
            return
        (self._evict_first if evict_first else self._items).append(item)

    def peek(self) -> "Evictable | None":
        # evict_first items jump the queue regardless of recency - upstream's
        # appendleft, expressed as a separate bucket so recency ordering inside
        # the normal queue stays untouched.
        if self._evict_first:
            return min(self._evict_first, key=lambda i: i.last_used)
        if not self._items:
            return None
        return min(self._items, key=lambda i: i.last_used)

    def pop(self) -> Evictable:
        victim = self.peek()
        if victim is None:
            raise IndexError("pop from empty eviction policy")
        return self.remove(victim)

    def remove(self, item: Evictable) -> Evictable:
        for bucket in (self._evict_first, self._items):
            for idx, existing in enumerate(bucket):
                if existing is item:
                    return bucket.pop(idx)
        raise KeyError(f"{getattr(item, 'name', item)!r} not tracked by this policy")

    def __len__(self) -> int:
        return len(self._items) + len(self._evict_first)

    def __iter__(self) -> Iterator[Evictable]:
        yield from self._evict_first
        yield from self._items


class PrioritizedEvictionPolicy:
    """Buckets items by priority; evicts from the lowest-priority bucket first.

    Upstream keeps `_policies` sorted by re-sorting the dict on insert. Same
    idea here, expressed as a sorted lookup at read time - a handful of
    priority levels makes the difference academic, and it removes a mutation
    that was easy to forget.
    """

    __slots__ = ("_policy_creator", "_policies")

    def __init__(self, policy_creator: Callable[[int], EvictionPolicy]) -> None:
        self._policy_creator = policy_creator
        self._policies: Dict[int, EvictionPolicy] = {}

    def get_policy(self, priority: int) -> EvictionPolicy:
        if priority not in self._policies:
            self._policies[priority] = self._policy_creator(priority)
        return self._policies[priority]

    def _ordered(self) -> List[int]:
        return sorted(self._policies)

    def push(self, item: Evictable, evict_first: bool = False) -> None:
        self.get_policy(item.priority).push(item, evict_first)

    def peek(self) -> "Evictable | None":
        for priority in self._ordered():
            candidate = self._policies[priority].peek()
            if candidate is not None:
                return candidate
        return None

    def pop(self) -> Evictable:
        victim = self.peek()
        if victim is None:
            raise IndexError("pop from empty eviction policy")
        return self.remove(victim)

    def remove(self, item: Evictable) -> Evictable:
        policy = self._policies.get(item.priority)
        if policy is None:
            raise KeyError(f"no policy for priority {item.priority}")
        removed = policy.remove(item)
        if not len(policy):
            self._policies.pop(item.priority, None)
        return removed

    def __len__(self) -> int:
        return sum(len(p) for p in self._policies.values())

    def __iter__(self) -> Iterator[Evictable]:
        for priority in self._ordered():
            yield from self._policies[priority]


class PrioritizedLRUEvictionPolicy(PrioritizedEvictionPolicy):
    """Priority first, recency within priority. The default policy."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(lambda _priority: LRUEvictionPolicy())


class PerTierEvictionController:
    """One eviction policy per tier.

    Upstream's PerLevelEvictionController holds one controller per cache level
    and sub-partitions by pool group; GB-Tiering has no pool groups, so a tier
    maps straight to a policy. Named for tiers rather than levels because that
    is the vocabulary everywhere else in this codebase.
    """

    __slots__ = ("_policies", "_policy_factory")

    def __init__(self, policy_factory: Callable[[], EvictionPolicy] = PrioritizedLRUEvictionPolicy):
        self._policy_factory = policy_factory
        self._policies: Dict[str, EvictionPolicy] = {}

    def policy_for(self, tier: str) -> EvictionPolicy:
        if tier not in self._policies:
            self._policies[tier] = self._policy_factory()
        return self._policies[tier]

    def schedule_for_eviction(self, item: Evictable, evict_first: bool = False) -> None:
        self.policy_for(item.tier).push(item, evict_first)

    def victim(self, tier: str) -> "Evictable | None":
        """Next thing that would be evicted from `tier`, without evicting it."""
        return self.policy_for(tier).peek()

    def evict(self, tier: str) -> Evictable:
        return self.policy_for(tier).pop()

    def forget(self, item: Evictable, tier: "str | None" = None) -> None:
        """Stop tracking an item. Tolerates an untracked item on purpose: the
        caller is often unwinding an error path and should not have to know
        whether the push happened."""
        try:
            self.policy_for(tier or item.tier).remove(item)
        except (KeyError, IndexError):
            pass

    def retier(self, item: Evictable, old_tier: str, new_tier: str,
               evict_first: bool = False) -> None:
        """Move an item's tracking between tiers. Callers set `item.tier`
        themselves; this keeps the controller in step."""
        self.forget(item, old_tier)
        self.policy_for(new_tier).push(item, evict_first)

    def counts(self) -> Dict[str, int]:
        return {tier: len(p) for tier, p in self._policies.items()}

    def __len__(self) -> int:
        return sum(len(p) for p in self._policies.values())


@dataclass
class _SimpleEvictable:
    """Minimal Evictable, for tests and for callers with no entry object."""

    name: str
    tier: str
    priority: int = PRIORITY_DEFAULT
    last_used: float = field(default_factory=time.time)


def clamp_priority(priority: Optional[int]) -> int:
    """Coerce anything caller-supplied into the valid band.

    None means "no opinion" and maps to the default, which is what keeps
    unannotated callers on the pre-existing plain-LRU behavior.
    """
    if priority is None:
        return PRIORITY_DEFAULT
    return max(PRIORITY_MIN, min(PRIORITY_MAX, int(priority)))

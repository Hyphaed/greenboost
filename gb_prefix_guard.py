#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
#
# Prefix-reuse accounting over content-addressed conversation chunks.
#
# Concept ported from NVIDIA TensorRT-LLM, Apache-2.0:
#   tensorrt_llm/runtime/kv_cache_manager_v2/_block_radix_tree.py
#   upstream commit 828e765a78 (2026-08-21)
#   Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
# What transfers is the shape of the idea - a radix tree keyed on a token
# prefix, answering "how much of this request is already covered" as a tree
# lookup rather than a similarity score (upstream: ReuseScope, ReuseMatch,
# Block.page_coverage). What does NOT transfer is upstream's purpose: choosing
# which KV pages to keep resident.
#
# WHY NOT THAT PURPOSE, MEASURED. The obvious port was "use the tree to pick a
# better engine slot". That is disproven on this configuration and the evidence
# is already in this repo. gb_prompt_index.pinning_enabled() records an A/B
# from 2026-08-19 (4 conversations x 3 turns, gb_bench_turn.py):
#
#     without id_slot   median warm reuse 40.2%   prefill 33.8 s
#     with    id_slot   median warm reuse  0.0%   prefill 28.0 s (full)
#
# Routing was correct - four conversations landed on four distinct slots, every
# later turn a pure append - and reuse still collapsed, because llama.cpp runs
# here with kv_unified and cache_idle_slots, which SAVE each idle slot to the
# host RAM prompt cache and then CLEAR it. A slot does not hold your
# conversation by the time you return to it. Reuse is recovered from that host
# cache instead. So a cleverer slot chooser optimises an identity the engine
# discards.
#
# WHAT THE EVIDENCE DOES SUPPORT , and a correction, 2026-08-22.
#
# This module was first written against a governed VIOLATION raised on
# 2026-08-21: `agent_compaction_broke_prefix`, agent_compaction_prefix_kept_pct
# = 0, recorded as casebook C-09. Wiring it up meant reading the events under
# that number, and they do not say what the case said. The log held exactly ONE
# `agent_context_edit` event in its whole history, and it was an `op=calibrate`.
# Four ops share that kind and the resolver had no `op` filter, so a
# calibration - which edits nothing and therefore carries no head_kept - was
# scored as a compaction that kept nothing. No compaction had happened. C-09's
# prefix finding is retracted there.
#
# The module survives that correction, for a reason worth stating plainly
# rather than quietly: the number was NOT MEASURABLE. Before this, the only
# available signal was `head_kept > 0` counted across events - a yes/no
# standing in for a percentage, and one that could not distinguish a destroyed
# prefix from an event that was never a compaction. That is exactly how a
# calibration came to be read as a lost cache for a day. A real per-compaction
# share, computed before the rewrite is applied, is what makes the difference
# visible.
#
# The cost being avoided is still real and still measured, just not by C-09:
# dataflux has recorded a 97.9% prompt-cache hit on a 64k window on this box,
# and gb_prompt_index's 2026-08-19 A/B measured warm reuse at 40.2% against
# 0.0%. At ~5 tok/s, discarding a prefix of that size is minutes of wall time.
# What was missing was not the motive but the measurement.
"""Answer "how much reusable prefix does this rewrite destroy?" before paying for it.

Accounting only. Nothing here edits a request, chooses a slot, or changes
placement - it reports a number and a verdict, and the caller decides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# A rewrite that keeps less than this share of the prior reusable prefix is
# reported as breaking it.
#
# Provisional, and labelled as such. It was originally justified by C-09's 0.0%
# reading, which was retracted on 2026-08-22 (see the header), so what is left
# supporting it is structural rather than measured: every compaction moves
# SOMETHING - the summary has to go somewhere - so any line at or near 100
# would fire on the normal, cheap case and mean nothing, while a rewrite that
# keeps under half the prefix is unambiguously the expensive path. Halfway is a
# defensible place to stand and not a derived one. Re-derive it from a real
# distribution of prefix_kept_pct once enough compactions have been recorded;
# until then callers who want a different line pass one, and there is no hidden
# default inside the accounting itself.
BROKE_PREFIX_PCT = 50.0


def common_prefix_len(a: Sequence[str], b: Sequence[str]) -> int:
    """Number of leading chunk digests `a` and `b` share.

    This is the whole reuse question at chunk granularity: llama.cpp keeps the
    prefix up to the first divergence and recomputes everything after it, so
    the shared prefix IS what survives.
    """
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@dataclass
class ReuseMatch:
    """What a lookup found. Mirrors upstream's ReuseMatch, minus KV specifics."""

    key: str                     # conversation whose prefix matched best
    matched_chunks: int          # how many leading chunks are shared
    matched_tokens: int          # ...and roughly how many tokens that is
    total_chunks: int            # chunks in the query
    total_tokens: int

    @property
    def coverage_pct(self) -> float:
        """Share of the QUERY already covered by a known prefix."""
        return 100.0 * self.matched_tokens / self.total_tokens if self.total_tokens else 0.0

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "matched_chunks": self.matched_chunks,
            "matched_tokens": self.matched_tokens,
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "coverage_pct": round(self.coverage_pct, 1),
        }


class PrefixTree:
    """Radix tree over chunk-digest sequences.

    Upstream indexes KV pages by token-block hash; this indexes conversations
    by chunk digest, which is the same operation one level up. Kept small and
    explicit rather than clever: the population is bounded by
    gb_prompt_index.MAX_CONVERSATIONS (512), so the interesting property is
    that longest-prefix is exact and cheap to reason about, not that it is
    asymptotically optimal.
    """

    __slots__ = ("_children", "_keys", "_tokens")

    def __init__(self) -> None:
        # digest -> subtree
        self._children: Dict[str, "PrefixTree"] = {}
        # conversation keys whose sequence ENDS at this node
        self._keys: List[str] = []
        # token count of the chunk that leads INTO this node
        self._tokens: int = 0

    def insert(self, key: str, digests: Sequence[str],
               tokens: "Sequence[int] | None" = None) -> None:
        node = self
        for i, d in enumerate(digests):
            child = node._children.get(d)
            if child is None:
                child = PrefixTree()
                node._children[d] = child
            if tokens is not None and i < len(tokens):
                child._tokens = tokens[i]
            node = child
        if key not in node._keys:
            node._keys.append(key)

    def longest_match(self, digests: Sequence[str],
                      tokens: "Sequence[int] | None" = None) -> ReuseMatch:
        """Deepest known prefix of `digests`, with the key that owns it."""
        node = self
        matched = matched_tokens = 0
        best_key = ""
        for i, d in enumerate(digests):
            child = node._children.get(d)
            if child is None:
                break
            node = child
            matched = i + 1
            matched_tokens += (tokens[i] if tokens is not None and i < len(tokens)
                               else child._tokens)
            if node._keys:
                best_key = node._keys[0]
        if not best_key:
            best_key = _first_key_below(node)
        total_tokens = sum(tokens) if tokens is not None else _sum_tokens(self, digests)
        return ReuseMatch(key=best_key, matched_chunks=matched,
                          matched_tokens=matched_tokens,
                          total_chunks=len(digests), total_tokens=total_tokens)

    def remove(self, key: str, digests: Sequence[str]) -> None:
        """Forget one conversation. Prunes nodes that nothing else needs."""
        path: List[Tuple["PrefixTree", str]] = []
        node = self
        for d in digests:
            child = node._children.get(d)
            if child is None:
                return
            path.append((node, d))
            node = child
        if key in node._keys:
            node._keys.remove(key)
        for parent, d in reversed(path):
            child = parent._children.get(d)
            if child is not None and not child._children and not child._keys:
                del parent._children[d]
            else:
                break

    def __len__(self) -> int:
        return len(self._keys) + sum(len(c) for c in self._children.values())


def _first_key_below(node: "PrefixTree") -> str:
    if node._keys:
        return node._keys[0]
    for child in node._children.values():
        k = _first_key_below(child)
        if k:
            return k
    return ""


def _sum_tokens(root: "PrefixTree", digests: Sequence[str]) -> int:
    node, total = root, 0
    for d in digests:
        child = node._children.get(d)
        if child is None:
            break
        total += child._tokens
        node = child
    return total


# ── the compaction question ───────────────────────────────────────────────────

@dataclass
class CompactionVerdict:
    """What a proposed rewrite does to the reusable prefix."""

    prefix_kept_pct: float
    kept_chunks: int
    prev_chunks: int
    kept_tokens: int
    prev_tokens: int
    broke_prefix: bool
    threshold_pct: float
    #  Tokens that will have to be recomputed because the prefix moved.
    tokens_lost: int = 0

    def as_dict(self) -> dict:
        return {
            "prefix_kept_pct": round(self.prefix_kept_pct, 1),
            "kept_chunks": self.kept_chunks,
            "prev_chunks": self.prev_chunks,
            "kept_tokens": self.kept_tokens,
            "prev_tokens": self.prev_tokens,
            "tokens_lost": self.tokens_lost,
            "broke_prefix": self.broke_prefix,
            "threshold_pct": self.threshold_pct,
        }

    def explain(self) -> str:
        """One line, cost first. Mechanism only if it changes what to do."""
        if self.prev_chunks == 0:
            return "no prior turn to compare against , nothing to preserve"
        if not self.broke_prefix:
            return (f"prefix preserved: {self.prefix_kept_pct:.0f}% of the reusable "
                    f"prefix survives ({self.kept_chunks}/{self.prev_chunks} chunks)")
        return (f"this rewrite keeps {self.prefix_kept_pct:.0f}% of the reusable prefix "
                f"({self.kept_chunks}/{self.prev_chunks} chunks); ~{self.tokens_lost} tokens "
                f"must be re-prefilled that the cache already held")


def assess_compaction(prev_digests: Sequence[str], new_digests: Sequence[str],
                      prev_tokens: "Sequence[int] | None" = None,
                      threshold_pct: float = BROKE_PREFIX_PCT) -> CompactionVerdict:
    """How much of the previous reusable prefix survives a proposed rewrite.

    This is the pre-flight form of the `agent_compaction_prefix_kept_pct`
    metric, which on 2026-08-21 fired at 0 AFTER the cost had been paid. Called
    before the request goes out, the same number is actionable.
    """
    kept = common_prefix_len(prev_digests, new_digests)
    prev_n = len(prev_digests)
    if prev_tokens is None:
        prev_tokens = [1] * prev_n
    prev_tok_total = sum(prev_tokens[:prev_n])
    kept_tok = sum(prev_tokens[:kept])
    pct = (100.0 * kept / prev_n) if prev_n else 100.0
    return CompactionVerdict(
        prefix_kept_pct=pct,
        kept_chunks=kept,
        prev_chunks=prev_n,
        kept_tokens=kept_tok,
        prev_tokens=prev_tok_total,
        tokens_lost=max(0, prev_tok_total - kept_tok),
        broke_prefix=bool(prev_n) and pct < threshold_pct,
        threshold_pct=threshold_pct,
    )


def _chunk_tokens(c) -> int:
    """Fallback token count for a chunk that does not carry one."""
    return getattr(c, "tokens", 0) or getattr(c, "n_tokens", 0) or 1


def verdict_from_chunks(prev_chunks, new_chunks,
                        threshold_pct: float = BROKE_PREFIX_PCT,
                        tokens_of=_chunk_tokens) -> CompactionVerdict:
    """Convenience over gb_prompt_index.Chunk sequences.

    `gb_prompt_index.Chunk` carries `chars`, not tokens, so the default
    `tokens_of` degrades to one-per-chunk and the token fields then report
    CHUNKS under a token name. Callers that know the tokenizer - or, better, a
    chars-per-token ratio calibrated against the server's own
    `usage.prompt_tokens` - pass `tokens_of` and get real token counts. The
    accounting does not guess a ratio on the caller's behalf; a made-up
    divisor here would be indistinguishable from a measured one downstream.
    """
    pd = [c.digest for c in prev_chunks]
    nd = [c.digest for c in new_chunks]
    pt = [tokens_of(c) for c in prev_chunks]
    return assess_compaction(pd, nd, pt, threshold_pct)

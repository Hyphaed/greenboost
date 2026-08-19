#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_prompt_index.py , GB-1: content-address a conversation in CHUNKS and keep
it on its own engine slot.

Why this exists, in one measurement. On this box, a warm turn prefills in
~1.77 s. The turn after an early message is edited costs ~8.8 s , 4.98x , and
cached tokens fall from 3113 to 2553 (`gb_bench_turn.py --edit-at`, three runs,
2026-08-19). Reuse does not drop to zero: llama.cpp keeps the prefix up to the
edit point and recomputes everything after it. So the cost of an edit is
exactly "how much conversation follows it", and the cost of losing a slot to
another conversation is "all of it".

This module owns the bookkeeping half of that problem, and only the lossless
half:

  * `chunk_messages()` splits a chat request the way the KV cache actually
    ages , a stable head (system prompt + tool schemas) followed by one chunk
    per message , and content-addresses each chunk. That makes "which chunk
    changed" answerable, which is what turns a reuse drop from a mystery into
    an attributable cost.

  * `ConversationIndex` remembers which engine slot a conversation was last
    served on and asks the engine for that same slot again (`id_slot`, honored
    on the OAI completions path , `server-context.cpp` reads it from the same
    request JSON before applying its own LCP similarity search).

The second point is the actual win, and it fixes a limitation this repo
already documented and left open on 2026-08-05: with slot selection driven
only by `--slot-prompt-similarity`, two concurrent conversations that share a
system prompt both converge on whichever slot matched that shared prefix, so
each one repeatedly evicts the other's tail. Any nonzero overlap with a
recently-touched slot beats an idle slot's zero score, which is not a
threshold that can be tuned around , it needs identity, which is this file.

What this module deliberately does NOT do: reuse a chunk's KV at a different
position. That is position-independent caching (GB-2, MiniPIC/EPIC), it is
approximate, and it is gated on a measured quality check. Everything here is
bookkeeping , the same tokens, in the same order, on a slot that still has
them.

Unattended-For-Days rule: the index is bounded (`MAX_CONVERSATIONS`) with
oldest-first eviction, and holds fixed-size digests, never message text.
"""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass, field

# A conversation's identity is its head, not its whole history. Using the full
# message list would mint a new identity on every turn, which is precisely the
# opposite of what a slot map is for.
#
# Two chunks is the smallest head that distinguishes real conversations on this
# box: chunk 0 is the system prompt + tool schemas, which GB-CLI makes
# byte-identical across every invocation (2026-08-05's prefix-stability work
# did that on purpose), so chunk 0 alone would map every conversation in the
# system onto one identity. Chunk 1 is the first user message, which is what
# actually differs.
IDENTITY_CHUNKS = 2

# An edit inside the identity window (a compaction rewriting the first user
# message) mints a NEW key, because the key is derived from exactly those
# chunks. That is not a bug to heuristic away, and an earlier draft of this
# module tried: "same head plus a long history" was supposed to mean "the same
# conversation, edited". It does not. A conversation that differs from another
# only in its first message is structurally identical to that same conversation
# with its first message edited , no amount of prefix, length or
# recency-matching separates them, and getting it wrong reintroduces exactly
# the cross-conversation contamination this module exists to prevent.
#
# So identity is exact or it is absent:
#
#   * exact  , the client supplied a conversation id (CONVERSATION_ID_FIELD),
#              or the head + first message match something already seen;
#   * absent , the request gets a slot the index has not mapped yet, and the
#              engine's own LCP similarity search does what it is good at.
#
# The cost of that honesty is bounded and known: the turn right after an early
# edit may land on a slot that does not hold its system-prompt head, and pays
# to re-prefill it once (2553 tokens, ~1.7 s on this box, measured 2026-08-19).
# A client that wants that turn free supplies the conversation id.
CONVERSATION_ID_FIELD = "gb_conversation"
CONVERSATION_ID_HEADER = "X-GB-Conversation"

# Bounded per the Unattended-For-Days must-rule. Each entry is a 16-char digest
# plus an int, so 512 entries is a few tens of KB and still far more than the
# engine has slots (typically 4) , the cap exists to stop unbounded growth over
# a multi-day run, not to ration a scarce resource.
MAX_CONVERSATIONS = 512

_DIGEST_LEN = 16


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:_DIGEST_LEN]


def _content_text(content) -> str:
    """OpenAI content is either a string or a list of typed parts.

    A part that is not text (an image_url, an audio chunk) still has to
    contribute to the digest , two conversations differing only by an attached
    image are not the same conversation , so unknown parts fall back to their
    repr rather than being skipped.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                out.append(part["text"])
            else:
                out.append(repr(part))
        return "\x1f".join(out)
    if content is None:
        return ""
    return repr(content)


@dataclass(frozen=True)
class Chunk:
    """One content-addressed span of a request, in prefill order."""
    idx: int
    role: str
    digest: str
    chars: int
    label: str = ""


def chunk_messages(messages: list, tools: "list | None" = None) -> "list[Chunk]":
    """Split a chat request into the spans whose KV ages independently.

    Chunk 0 is the stable head: the leading system message(s) plus the tool
    schemas, which is what a coding agent re-sends verbatim on every turn.
    They are ONE chunk because the engine can only reuse a prefix, so a change
    anywhere in that head invalidates all of it regardless of how it is
    counted here , splitting it would suggest a granularity the engine does
    not have.

    Every later message is its own chunk, in order. The tool result that
    follows an assistant tool call is an ordinary message here; it is a
    separate chunk because it is separately editable, which is the property
    that matters.
    """
    chunks: list[Chunk] = []
    msgs = list(messages or [])

    head_parts: list[str] = []
    if tools:
        # Tool schemas prefix the head: they are rendered into the prompt
        # ahead of the conversation by the chat template, so a change to a
        # tool definition invalidates at least as much as a system-prompt
        # change does.
        head_parts.append(repr(tools))
    consumed = 0
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") != "system":
            break
        head_parts.append(_content_text(m.get("content")))
        consumed += 1

    head_text = "\x1e".join(head_parts)
    if head_parts:
        chunks.append(Chunk(idx=0, role="system", digest=_digest(head_text),
                            chars=len(head_text),
                            label="system+tools" if tools else "system"))

    for m in msgs[consumed:]:
        role = m.get("role", "?") if isinstance(m, dict) else "?"
        text = _content_text(m.get("content") if isinstance(m, dict) else m)
        # An assistant turn that only called tools carries no content but is
        # still real prefill, so fold the calls into the digest.
        if isinstance(m, dict) and m.get("tool_calls"):
            text = f"{text}\x1f{m['tool_calls']!r}"
        chunks.append(Chunk(idx=len(chunks), role=role, digest=_digest(text),
                            chars=len(text)))
    return chunks


def conversation_key(chunks: "list[Chunk]") -> str:
    """Stable identity for the conversation these chunks belong to."""
    head = "".join(c.digest for c in chunks[:IDENTITY_CHUNKS])
    return _digest(head) if head else ""


def first_divergence(prev: "list[Chunk]", cur: "list[Chunk]") -> "int | None":
    """Index of the first chunk that changed, or None if `cur` only appended.

    This is the number that makes a reuse drop attributable: a pure append
    returns None and must cost nothing, while an edit returns the chunk index
    whose KV , and every token after it , the engine has to recompute.
    """
    for i, (a, b) in enumerate(zip(prev, cur)):
        if a.digest != b.digest:
            return i
    return None


@dataclass
class Assignment:
    """What the index decided for one request."""
    slot: "int | None"
    key: str
    reason: str
    chunks: int
    changed_chunk: "int | None" = None
    chunks_before: int = 0

    @property
    def appended_only(self) -> bool:
        return self.changed_chunk is None and self.chunks >= self.chunks_before


@dataclass
class _Entry:
    slot: int
    chunks: "list[Chunk]" = field(default_factory=list)


class ConversationIndex:
    """Maps conversations to engine slots, bounded and oldest-first.

    `n_slots` is the engine's real slot count (`--parallel`), read from the
    server rather than assumed; when it is unknown the index refuses to guess
    and `assign()` returns `slot=None`, which the caller must treat as "send
    no id_slot and let the engine choose", never as slot 0.
    """

    def __init__(self, n_slots: int = 0, max_conversations: int = MAX_CONVERSATIONS):
        self.n_slots = int(n_slots or 0)
        self._max = int(max_conversations)
        self._by_key: "OrderedDict[str, _Entry]" = OrderedDict()
        # What each engine slot last held, for divergence reporting. Bounded by
        # the slot count, which is the engine's own --parallel.
        self._slot_chunks: "dict[int, list[Chunk]]" = {}

    # -- introspection -----------------------------------------------------
    def __len__(self) -> int:
        return len(self._by_key)

    def slot_of(self, key: str) -> "int | None":
        e = self._by_key.get(key)
        return e.slot if e else None

    def _slots_in_use(self) -> set:
        return {e.slot for e in self._by_key.values()}

    def _unmapped_slot(self) -> "int | None":
        """A slot this index has not routed a conversation to yet."""
        used = {e.slot for e in self._by_key.values()}
        for s in range(self.n_slots):
            if s not in used:
                return s
        return None

    def _lru_slot(self) -> int:
        for key in self._by_key:                       # OrderedDict: oldest first
            return self._by_key[key].slot
        return 0

    def assign(self, messages: list, tools: "list | None" = None,
               conversation_id: str = "") -> Assignment:
        chunks = chunk_messages(messages, tools)
        key = _digest(conversation_id) if conversation_id else conversation_key(chunks)
        if not key or self.n_slots <= 1:
            # One slot means there is nothing to route between, and pinning to
            # slot 0 would only add a field to every request for no effect.
            return Assignment(slot=None, key=key, reason="no-routing",
                              chunks=len(chunks))

        entry = self._by_key.get(key)
        if entry is None:
            slot = self._unmapped_slot()
            reason = "assigned"
            if slot is None:
                slot = self._lru_slot()
                reason = "reassigned"
            self._by_key[key] = _Entry(slot=slot, chunks=chunks)
            self._evict()
        else:
            slot = entry.slot
            entry.chunks = chunks
            self._by_key.move_to_end(key)
            reason = "pinned"

        # What changed is measured against what that SLOT last held, not
        # against what the conversation last sent. The engine compares the
        # incoming tokens with the slot's tokens, so this is the comparison
        # that actually predicts the prefill about to be paid , and it stays
        # meaningful even when a conversation was reassigned.
        held = self._slot_chunks.get(slot, [])
        changed = first_divergence(held, chunks) if held else None
        before = len(held)
        self._slot_chunks[slot] = chunks
        if reason == "pinned" and changed is not None:
            reason = "pinned-edited"
        return Assignment(slot=slot, key=key, reason=reason, chunks=len(chunks),
                          changed_chunk=changed, chunks_before=before)

    def _evict(self) -> None:
        while len(self._by_key) > self._max:
            self._by_key.popitem(last=False)


def pinning_enabled() -> bool:
    """OFF by default. Measured, and it made things worse.

    A/B on this box, 2026-08-19, 4 conversations x 3 turns sharing one system
    prompt (`gb_bench_turn.py --conversations 4`), same model and ctx:

        without id_slot   median warm reuse 40.2%   prefill 33.8 s
        with    id_slot   median warm reuse  0.0%   prefill 28.0 s (full)

    The index itself did exactly what it was built to do , the `cache_index`
    events show four conversations on four distinct slots, every later turn
    `pinned`, every one a pure append. Routing was correct and reuse still
    collapsed, because on this engine configuration per-slot KV is not where
    reuse comes from:

      * `gb_synapse` launches llama-server with `-np -1` (auto), and
        `tools/server/server.cpp:146` resolves that to `n_parallel = 4` AND
        `kv_unified = true`.
      * With `kv_unified` and the default `cache_idle_slots = true`, every new
        task SAVES each idle slot to the host RAM prompt cache and then CLEARS
        it (`server-context.cpp`, [TAG_IDLE_SLOT_CLEAR]). A slot therefore does
        not hold your conversation by the time you come back to it.
      * Reuse is recovered instead by `prompt_load()` from that host cache ,
        but `get_available_slot()` only calls it when `update_cache` is true,
        which happens when the slot was picked by LRU, or by similarity with
        `f_keep < 0.5`. Passing an explicit `id_slot` satisfies neither: `ret`
        is already set, so the restore never runs.

    So pinning cannot help here (the slot is cleared regardless) and it
    actively disables the mechanism that was helping. It stays in the tree
    because the finding is configuration-specific , an engine built without
    `kv_unified`, or a future one that restores the cache for an explicitly
    requested slot, flips the conclusion , and because turning it on is how
    that gets re-measured. `GB_SYNAPSE_SLOT_PIN=1` opts in.

    The chunk index around it is NOT gated on this: content-addressing the
    conversation is what made the failure diagnosable in the first place, and
    it keeps running to emit `cache_index`.
    """
    return os.environ.get("GB_SYNAPSE_SLOT_PIN", "0") in ("1", "true", "yes")

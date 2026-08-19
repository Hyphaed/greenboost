#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""GB-1: conversation identity, chunking, and slot routing.

The property under test throughout is that identity survives a turn but not an
edit, because that is the difference between keeping a conversation's KV and
silently recomputing it.
"""
import gb_prompt_index as pi

SYS = {"role": "system", "content": "You are a coding agent. " * 50}


def _conv(user_msgs):
    out = [SYS]
    for i, u in enumerate(user_msgs):
        out.append({"role": "user", "content": u})
        out.append({"role": "assistant", "content": f"reply {i}"})
    return out


# ── chunking ────────────────────────────────────────────────────────────────

def test_system_and_tools_are_one_chunk():
    """The engine can only reuse a prefix, so splitting the head would
    advertise a granularity that does not exist."""
    chunks = pi.chunk_messages([SYS, {"role": "user", "content": "hi"}],
                               tools=[{"name": "read_file"}])
    assert chunks[0].role == "system" and chunks[0].label == "system+tools"
    assert len(chunks) == 2


def test_a_tool_schema_change_changes_the_head():
    """Tool definitions are rendered ahead of the conversation, so editing one
    invalidates at least as much as editing the system prompt."""
    a = pi.chunk_messages([SYS], tools=[{"name": "read_file"}])
    b = pi.chunk_messages([SYS], tools=[{"name": "read_file", "extra": 1}])
    assert a[0].digest != b[0].digest


def test_tool_calls_count_even_with_empty_content():
    """An assistant turn that only called tools is real prefill; digesting it
    as empty would make two different histories look identical."""
    base = [SYS, {"role": "assistant", "content": ""}]
    withcall = [SYS, {"role": "assistant", "content": "",
                      "tool_calls": [{"id": "1", "function": {"name": "ls"}}]}]
    assert pi.chunk_messages(base)[1].digest != pi.chunk_messages(withcall)[1].digest


def test_multimodal_parts_are_not_silently_dropped():
    text_only = [SYS, {"role": "user", "content": [{"type": "text", "text": "look"}]}]
    with_image = [SYS, {"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]
    assert pi.chunk_messages(text_only)[1].digest != pi.chunk_messages(with_image)[1].digest


# ── identity ────────────────────────────────────────────────────────────────

def test_identity_survives_appending_turns():
    """The whole point of a slot map: turn 5 of a conversation must be the same
    conversation as turn 1."""
    k1 = pi.conversation_key(pi.chunk_messages(_conv(["a"])))
    k5 = pi.conversation_key(pi.chunk_messages(_conv(["a", "b", "c", "d", "e"])))
    assert k1 == k5


def test_a_shared_system_prompt_is_not_a_shared_identity():
    """GB-CLI makes the system prompt byte-identical across invocations on
    purpose (2026-08-05). If identity stopped at chunk 0, every conversation on
    the box would collapse onto one slot , the exact bug this fixes."""
    k1 = pi.conversation_key(pi.chunk_messages(_conv(["question one"])))
    k2 = pi.conversation_key(pi.chunk_messages(_conv(["question two"])))
    assert k1 != k2


def test_editing_the_head_is_a_different_conversation():
    edited = _conv(["a"])
    edited[0] = {"role": "system", "content": "different prompt"}
    assert pi.conversation_key(pi.chunk_messages(_conv(["a"]))) != \
           pi.conversation_key(pi.chunk_messages(edited))


# ── divergence ──────────────────────────────────────────────────────────────

def test_pure_append_reports_no_divergence():
    """A turn that only appends must cost nothing, and must be REPORTED as
    costing nothing , otherwise every turn looks like an edit."""
    prev = pi.chunk_messages(_conv(["a"]))
    cur = pi.chunk_messages(_conv(["a", "b"]))
    assert pi.first_divergence(prev, cur) is None


def test_an_early_edit_is_reported_at_its_chunk():
    prev = pi.chunk_messages(_conv(["a", "b", "c"]))
    edited = _conv(["a", "b", "c"])
    edited[1]["content"] = "a (revised)"        # first user message
    assert pi.first_divergence(prev, pi.chunk_messages(edited)) == 1


# ── routing ─────────────────────────────────────────────────────────────────

def test_two_conversations_get_two_slots():
    """The measured failure this fixes: both conversations converging on the
    slot that matched their shared system prompt."""
    idx = pi.ConversationIndex(n_slots=4)
    a = idx.assign(_conv(["question one"]))
    b = idx.assign(_conv(["question two"]))
    assert a.slot != b.slot
    assert (a.reason, b.reason) == ("assigned", "assigned")


def test_a_returning_conversation_gets_its_slot_back():
    idx = pi.ConversationIndex(n_slots=4)
    first = idx.assign(_conv(["question one"]))
    idx.assign(_conv(["question two"]))
    again = idx.assign(_conv(["question one", "more"]))
    assert again.slot == first.slot
    assert again.reason == "pinned"
    assert again.changed_chunk is None


def test_an_edit_downstream_of_the_head_keeps_the_slot_and_names_the_chunk():
    """Editing anything after the identity window is the common case (a tool
    result rewritten, a turn re-rendered). Identity holds, so the slot holds,
    and the report says which chunk the engine will have to recompute from."""
    idx = pi.ConversationIndex(n_slots=4)
    first = idx.assign(_conv(["a", "b", "c"]))
    edited = _conv(["a", "b", "c"])
    edited[3]["content"] = "b (revised)"          # second user message
    after = idx.assign(edited)
    assert after.slot == first.slot
    assert after.reason == "pinned-edited"
    assert after.changed_chunk == 3


def test_an_edit_inside_the_identity_window_is_a_new_identity():
    """Deliberate, and documented in the module: a conversation whose first
    message was rewritten is structurally indistinguishable from a different
    conversation with the same system prompt. The index refuses to guess, so
    the turn gets its own slot and the engine's LCP search does the rest. A
    client that wants that turn free supplies a conversation id (test below)."""
    idx = pi.ConversationIndex(n_slots=4)
    first = idx.assign(_conv(["a", "b"]))
    edited = _conv(["a", "b"])
    edited[1]["content"] = "a (revised)"          # first user message
    after = idx.assign(edited)
    assert after.reason == "assigned"
    assert after.slot != first.slot


def test_a_conversation_id_makes_an_early_edit_free():
    """The exact-identity path: with an id, even a rewritten first message
    keeps its slot, and the changed chunk is named."""
    idx = pi.ConversationIndex(n_slots=4)
    first = idx.assign(_conv(["a", "b"]), conversation_id="session-7")
    edited = _conv(["a", "b"])
    edited[1]["content"] = "a (revised)"
    after = idx.assign(edited, conversation_id="session-7")
    assert after.slot == first.slot
    assert after.reason == "pinned-edited"
    assert after.changed_chunk == 1


def test_divergence_is_reported_against_what_the_slot_holds():
    """Not against what the conversation last sent. The engine compares the
    incoming tokens with the SLOT's tokens, so that is the comparison that
    predicts the prefill about to be paid."""
    idx = pi.ConversationIndex(n_slots=1 + 1)      # 2 slots
    idx.assign(_conv(["one"]), conversation_id="A")
    b = idx.assign(_conv(["two"]), conversation_id="B")
    assert b.chunks_before == 0                    # slot 1 held nothing
    assert b.changed_chunk is None                 # nothing to diverge from


def test_more_conversations_than_slots_reassigns_the_oldest():
    idx = pi.ConversationIndex(n_slots=2)
    a = idx.assign(_conv(["one"]))
    idx.assign(_conv(["two"]))
    c = idx.assign(_conv(["three"]))
    assert c.reason == "reassigned"
    assert c.slot == a.slot          # oldest conversation's slot is the one reused


def test_unknown_slot_count_refuses_to_guess():
    """llama-server wraps an out-of-range id_slot modulo the slot count, so a
    guess does not fail loudly , it silently collides two conversations."""
    idx = pi.ConversationIndex(n_slots=0)
    assert idx.assign(_conv(["a"])).slot is None
    assert pi.ConversationIndex(n_slots=1).assign(_conv(["a"])).slot is None


def test_index_is_bounded_for_a_multi_day_run():
    """Unattended-For-Days rule: nothing per-turn may grow without a cap."""
    idx = pi.ConversationIndex(n_slots=4, max_conversations=8)
    for i in range(200):
        idx.assign(_conv([f"conversation {i}"]))
    assert len(idx) <= 8


def test_no_message_text_is_retained():
    """The index holds digests so a dataflux event can carry `conv` without
    leaking the conversation."""
    idx = pi.ConversationIndex(n_slots=4)
    secret = "SECRET-TOKEN-VALUE"
    idx.assign(_conv([secret]))
    assert secret not in repr(idx.__dict__)


def test_pinning_is_opt_in_because_it_measured_worse(monkeypatch):
    """Measured 2026-08-19: pinning took warm reuse from 40.2% to 0.0% on this
    engine configuration, because `-np -1` implies kv_unified, idle slots are
    cleared on every new task, and an explicit id_slot skips the host-cache
    restore that was doing the actual work. Default off; the flag exists to
    re-measure it on an engine build where that does not hold."""
    monkeypatch.delenv("GB_SYNAPSE_SLOT_PIN", raising=False)
    assert pi.pinning_enabled() is False
    monkeypatch.setenv("GB_SYNAPSE_SLOT_PIN", "1")
    assert pi.pinning_enabled() is True

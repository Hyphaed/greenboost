"""
Core turn execution loop.

Drives the multi-turn, multi-provider agentic cycle:
  user message → stream model → execute instruments → loop until no more tool calls

Loop guards (important for chatty local models like qwen36-coder):
  - Hard turn cap: max_tool_turns (default 50) — prevents infinite loops.
  - Repeat-limit: same tool+args key repeated ≥3 times → abort (model is stuck).
  - Consecutive-error abort: ≥4 tool results starting with Denied/ERROR/Error → abort.
All three yield a LoopGuardTriggered event before stopping.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Generator

from greenboost_cli.core.session import ConversationSession
from greenboost_cli.instruments.schemas import INSTRUMENT_DEFINITIONS
from greenboost_cli.instruments.dispatcher import dispatch, _describe_operation
from greenboost_cli.instruments.safety import is_readonly_command, is_autonomous_safe
from greenboost_cli.inference.router import generate, StreamFragment, ReasoningFragment, CompletedResponse

# Guard thresholds (override via settings)
_DEFAULT_MAX_TURNS   = 50
_REPEAT_LIMIT        = 3   # same tool+args repeated this many times → stop
_CONSEC_ERROR_LIMIT  = 4   # this many consecutive Denied/ERROR results → stop
#: How many TRANSIENT tool failures (timeout, device busy, connection reset)
#: one turn may absorb before they start counting toward the guard above.
#: Small on purpose: a genuinely wedged resource should still stop the turn,
#: but a couple of hiccups should not end a long agentic run.
_TRANSIENT_ERROR_LIMIT = 2
_INTENT_NUDGE_CAP    = 2   # at most this many "you said you would, now do it" nudges

# Matches a model announcing a build/task ("I'll build X", "Let me create Y")
# without having called any tool yet. Narrow on purpose: two-part match
# (first-person forward-looking phrase, THEN an action verb within a short
# span) so it doesn't fire on prose that merely mentions "build" or
# "create" in passing. Real incident this targets (2026-08-01): a turn
# whose entire output was an intent paragraph ("I'll build a complete Snake
# game in Godot 4...") with zero tool calls — the loop treated it as a
# finished answer and stopped, having written nothing.
#
# Widened 2026-08-02: the verb list originally only covered *productive*
# verbs (build/create/write/...), so it missed the investigative hand-off
# that actually caused the next incident — turn 3 of a Godot Snake-game
# session ended on "I'll check what's in the godot directory and find how
# to run Godot on this machine." (30 tokens, clean EOS per llama-server's
# own log, no tool call) and the loop accepted it as a finished answer. Added
# check/find/look/inspect/examine/explore/read/see/verify/search. The gate
# below was also widened from turn_count==1-only to any turn (bounded by a
# short-reply length check + the nudge cap above), since this incident fired
# on turn 3, not turn 1.
_INTENT_RE = re.compile(
    r"\b(I'?ll|I will|Let me|I'?m going to|I am going to)\b"
    r"[^.\n]{0,120}\b(build|create|write|implement|develop|generate|"
    r"set up|scaffold|start (building|working|coding)|"
    r"check|find|look|inspect|examine|explore|read|see|verify|search)\b",
    re.IGNORECASE,
)

# A terse forward-looking sentence ("I'll check X.") is a hand-off with
# nothing done yet; a long reply that happens to contain "I'll check" partway
# through is very likely a real, substantive answer. Gates the any-turn
# intent-nudge below so it can't misfire on legitimate long-form answers now
# that the verb list includes common narrative words like "check"/"look".
_INTENT_NUDGE_MAX_LEN = 400


# Hard byte ceilings, independent of the live context window — NemoClaw
# audit, Phase 3e. The fraction-of-context budget below (`budget_chars`)
# already shrinks for a small window, but had no UPPER bound at all: on a
# huge served ctx, `int(ctx * 0.15 * 4)` could still let an enormous MCP
# schema block through uncapped. These two constants close that, matching
# NemoClaw's progressive_tool_disclosure.py's own byte-budget shape
# (design re-implemented from scratch — no NemoClaw code copied here,
# unlike instruments/capture.py). MAX_SINGLE_TOOL_SCHEMA_BYTES additionally
# protects against ONE pathological MCP server's oversized schema forcing
# every OTHER server's tools to degrade too, even when the aggregate would
# otherwise fit.
MAX_VISIBLE_DISCOVERED_SCHEMA_BYTES = 128 * 1024
MAX_SINGLE_TOOL_SCHEMA_BYTES = 16 * 1024
MAX_SEARCH_QUERY_LENGTH = 256
MAX_SEARCH_OUTPUT_BYTES = 8 * 1024


def _light_schema(schema: dict) -> dict:
    """Name + one sentence, no parameters , the degraded form.

    Deliberately carries NO "call ToolSearch for the real schema" sentence.
    That instruction is identical for every tool and is already stated once, in
    the ToolSearch builtin's own description ("most MCP tools are advertised to
    you with only a name and short description ... call ToolSearch first").
    Repeating it per tool bought nothing and cost a great deal: measured on a
    live 238-tool session, 116 chars x 238 = 27,608 chars of byte-identical
    text, about 6,900 prompt tokens re-sent on EVERY request.

    That is not a rounding error on this hardware. The reference architecture
    rejects `--cache-reuse`. That does NOT mean every turn re-pays prefill ,
    same-slot prefix reuse still works and measured 99.7% hit on this model.
    It does mean the FIRST turn after any restart pays in full, and that turn
    was measured at 30-46 tok/s while weights spill to T2 , roughly 197 seconds
    of it spent re-reading the same sentence 238 times. An empty
    `properties` is already the per-tool signal that the schema was withheld.
    """
    name = schema.get("name", "")
    desc = (schema.get("description") or "").strip()
    first_sentence = desc.split(". ")[0][:160]
    return {
        "name": name,
        "description": first_sentence,
        "input_schema": {"type": "object", "properties": {}},
    }


def _capped_mcp_schemas(mcp_schemas: list, settings: dict) -> list:
    """Cap the MCP portion of the tool-schema block to a fraction of the
    LIVE served context window, instead of sending every connected server's
    full tool schema on every turn unconditionally.

    Before this, a 233-tool session (10 MCP servers) sent ~64k tokens of
    tool schemas as a FIXED PREFIX on every turn — confirmed live 2026-08-01
    via a prompt_cache dataflux event (64,144 reused tokens, 97.9% hit,
    against a 65,536-token window: ~50 tokens left to think or generate in).
    History compaction (_compress_context) only rewrites session.messages,
    never this prefix, so once the prefix alone approached the window the
    turn could never produce more than a few dozen output tokens no matter
    how many times it "auto-compacted" — the empty_response loop guard that
    silently killed an 88-minute unattended build session with zero useful
    output.

    Small MCP surfaces (few servers connected) keep full schemas — never
    shrink for a session that can afford them, same convention as
    instruments/handlers.py's `_ctx_char_budget`. Only once the full set
    would exceed ~15% of the live window (that fraction chosen to match
    `_ctx_char_budget`'s existing precedent), OR the hard
    MAX_VISIBLE_DISCOVERED_SCHEMA_BYTES ceiling above, do entries degrade to
    name + first-sentence description, no input_schema — the model must
    call the ToolSearch builtin to fetch the real parameters before calling
    an unfamiliar mcp__ tool with arguments, the same deferred-schema
    pattern Claude Code's own harness uses for its MCP tool surface. This
    never changes what a tool call actually DOES — the real MCP server
    still gets whatever arguments the model sends regardless of the schema
    shown to it, so a model that guesses right on an obvious tool still
    works fine.

    Independently of the aggregate budget, any SINGLE schema over
    MAX_SINGLE_TOOL_SCHEMA_BYTES degrades on its own — one pathological MCP
    server's giant schema must not force every other server's tools to
    degrade too."""
    if not mcp_schemas:
        return []
    try:
        from greenboost_cli.environment.settings import gb_synapse_ctx
        ctx = gb_synapse_ctx(settings)
    except Exception:
        ctx = 0
    budget_chars = max(4_000, int(ctx * 0.15 * 4)) if ctx else 30_000
    budget_chars = min(budget_chars, MAX_VISIBLE_DISCOVERED_SCHEMA_BYTES)

    full_chars = sum(len(json.dumps(s)) for s in mcp_schemas)
    if full_chars <= budget_chars:
        # Aggregate fits — still enforce the per-tool ceiling individually.
        return [
            schema if len(json.dumps(schema)) <= MAX_SINGLE_TOOL_SCHEMA_BYTES
            else _light_schema(schema)
            for schema in mcp_schemas
        ]

    light = [_light_schema(schema) for schema in mcp_schemas]
    light_chars = sum(len(json.dumps(x)) for x in light)
    if light_chars <= budget_chars:
        return light

    # Degrading every schema was not enough: the NAMES alone still overflow.
    # Measured live 2026-08-18 , ten MCP servers, 238 tools , degraded schemas
    # still carried ~17k prompt tokens, and this architecture rejects
    # --cache-reuse, so the first turn after a restart pays full prefill
    # (later turns reuse the prefix via slots , measured 99.7% hit). Worse,
    # on that first turn prefill
    # cost is super-linear in prompt length (attention is quadratic): the
    # observed per-2048-token chunk time grew by a steady +9.1 s, so a 23,273
    # token prompt took a projected 837 s BEFORE the first output token.
    #
    # So cap the COUNT too, not just the bytes. This is exactly what ToolSearch
    # exists for , every withheld tool stays reachable by name or keyword, and
    # its own description already tells the model to use it.
    #
    # Round-robin across servers rather than truncating the list: a plain
    # head-slice would let whichever server sorts first consume the entire
    # budget and leave whole servers invisible, which is a far worse failure
    # than showing fewer tools from each.
    by_server: "dict[str, list]" = {}
    for x in light:
        name = x.get("name", "")
        server = name.split("__")[1] if name.startswith("mcp__") and "__" in name[5:] else ""
        by_server.setdefault(server, []).append(x)

    kept, used, queues = [], 0, list(by_server.values())
    while queues:
        for q in list(queues):
            if not q:
                queues.remove(q)
                continue
            cand = q.pop(0)
            cost = len(json.dumps(cand)) + 1
            if used + cost > budget_chars:
                queues.clear()
                break
            kept.append(cand)
            used += cost
    withheld = len(light) - len(kept)
    if withheld:
        kept.append({
            "name": "_mcp_tools_withheld",
            "description": (
                f"{withheld} more MCP tools are connected but not listed here, to "
                f"keep the prompt small. Call ToolSearch with a keyword or an "
                f"exact mcp__server__tool name to fetch any of them."
            ),
            "input_schema": {"type": "object", "properties": {}},
        })
    return kept


def _tool_search(mcp_registry, query: str, max_results: int = 6) -> str:
    """Handler for the ToolSearch builtin: look up full MCP tool schemas
    (from the registry's uncapped list, unaffected by _capped_mcp_schemas)
    by name or description keyword.

    This is a CONTEXT OPTIMIZATION, not an authorization boundary (repeating
    NemoClaw's own caveat on progressive_tool_disclosure.py verbatim, since
    the failure mode it warns against is real here too): a model that
    guesses a withheld tool's exact name and calls it directly, without ever
    going through ToolSearch, still has that call evaluated by
    dispatcher.py's policy gate like any other tool call. Narrowing what's
    VISIBLE in the schema block never narrows what's PERMITTED — those are
    two independent layers, and this function only ever touches the first."""
    if mcp_registry is None:
        return "No MCP servers connected."
    q = (query or "").lower().strip()[:MAX_SEARCH_QUERY_LENGTH]
    if not q:
        return "ToolSearch needs a non-empty query (a tool name or keyword)."
    scored = []
    for schema in mcp_registry.tool_schemas:
        name = schema.get("name", "")
        desc = schema.get("description", "") or ""
        if q in name.lower():
            scored.append((0, schema))
        elif q in desc.lower():
            scored.append((1, schema))
    scored.sort(key=lambda pair: pair[0])
    total_matched = len(scored)
    matches = [s for _, s in scored[:max_results]]
    if not matches:
        return (
            f"No tools matched '{query}'. Try a shorter or different keyword "
            "(e.g. part of the server name, like 'forge3d' or 'greenboost')."
        )
    # Tell the model when results were withheld and why (NemoClaw audit,
    # Phase 3e) — a silent top-N truncation looks identical to "these are
    # all the matches", which can lead the model to conclude a tool doesn't
    # exist when it was simply outside the shown slice.
    withheld = total_matched - len(matches)
    encoded = json.dumps(matches, indent=2)
    if len(encoded) > MAX_SEARCH_OUTPUT_BYTES:
        # Re-encode compactly first; only fall back to a hard truncation
        # (rare — one match's schema would have to be enormous) if even
        # that doesn't fit.
        encoded = json.dumps(matches)
        if len(encoded) > MAX_SEARCH_OUTPUT_BYTES:
            encoded = encoded[:MAX_SEARCH_OUTPUT_BYTES] + \
                "...[truncated — narrow your query for a smaller result set]"
    if withheld > 0:
        encoded += (
            f"\n\n({withheld} more match(es) not shown — refine your query "
            "to see them.)"
        )
    return encoded


# ── Event types yielded to the caller ─────────────────────────────────────

@dataclass
class InstrumentInvoked:
    """Emitted when the model calls a tool, before execution."""
    name:   str
    inputs: dict

@dataclass
class InstrumentResult:
    """Emitted after a tool finishes executing."""
    name:      str
    result:    str
    permitted: bool = True

@dataclass
class TurnComplete:
    """Emitted at the end of each model turn."""
    input_tokens:  int
    output_tokens: int
    tok_s:         float = 0.0    # decode speed for this turn's generation span
    elapsed_s:     float = 0.0
    is_final:      bool  = True   # False for intermediate tool-calling turns

@dataclass
class ApprovalNeeded:
    """Emitted when an operation requires user permission.

    The caller sets .granted = True/False; execution continues based on that.
    """
    description: str
    granted:     bool = False

@dataclass
class QuestionsAsked:
    """Emitted when the model calls AskUserQuestion.

    The caller fills .answers with a list of answer dicts:
      [{"header": str, "question": str, "answers": [str, ...]}, ...]
    A None value means the user cancelled (Ctrl-C / EOF).
    """
    questions: list
    answers:   list | None = None

@dataclass
class LoopGuardTriggered:
    """Emitted when a loop-safety guard aborts the agentic loop early.

    reason is one of: 'max_turns' | 'repeat' | 'consec_errors'
    """
    reason:  str
    message: str


# ── Main loop ──────────────────────────────────────────────────────────────


#: Upper bound on tools run at once. Matches Claude Code's default; the work is
#: I/O-bound (file reads, greps, fetches), so this is about not drowning the
#: filesystem, not about CPU.
_MAX_TOOL_CONCURRENCY = 10


def _prefetch_safe_batches(tool_calls, session, settings):
    """Run consecutive concurrency-safe tool calls ahead of the event loop.

    Returns {index_in_tool_calls: result_string} for calls that were executed
    here; the main loop yields their events in the original order and picks the
    result up instead of dispatching again.

    A call is only eligible when it would have run unattended anyway , not an
    intercept, not blocked by plan mode, and already auto-approved. Anything
    that would have shown the user an approval card is left to the serial path,
    so this can never execute something the user has not agreed to, and can
    never reorder a prompt.
    """
    from concurrent.futures import ThreadPoolExecutor
    from greenboost_cli.instruments.concurrency import partition_tool_calls

    _INTERCEPTS = ("AskUserQuestion", "ToolSearch")
    out: dict[int, str] = {}
    index = {id(tc): i for i, tc in enumerate(tool_calls)}

    def _eligible(tc):
        if tc["name"] in _INTERCEPTS:
            return False
        if _plan_mode_block(tc, session) is not None:
            return False
        try:
            _is_mcp = bool(session.mcp_registry
                           and session.mcp_registry.has_tool(tc["name"]))
            return bool(_is_auto_approved(tc, settings, is_mcp=_is_mcp))
        except Exception:
            return False

    for safe, batch in partition_tool_calls(
            tool_calls, mcp_registry=getattr(session, "mcp_registry", None)):
        if not safe or len(batch) < 2:
            continue                      # one call gains nothing from a thread
        runnable = [tc for tc in batch if _eligible(tc)]
        if len(runnable) < 2:
            continue

        def _run(tc):
            return dispatch(
                tc["name"], tc["input"],
                approval_mode="accept-all",     # eligibility checked above
                policy=settings.get("_tool_policy"),
                agent=settings.get("_agent_label", "main"),
            )

        with ThreadPoolExecutor(max_workers=min(len(runnable),
                                                _MAX_TOOL_CONCURRENCY)) as pool:
            for tc, res in zip(runnable, pool.map(_run, runnable)):
                out[index[id(tc)]] = res
    return out


def _touched_paths(messages: list) -> list:
    """Files this session has actually worked on, newest first.

    Drives scope-matched memory recall (AE-10): a memory about the CUDA shim
    should arrive when the shim is being edited and stay silent otherwise.
    """
    out: list = []
    for m in reversed(messages[-40:]):
        for tc in (m.get("tool_calls") or ()):
            fp = (tc.get("input") or {}).get("file_path")
            if fp and fp not in out:
                out.append(fp)
        if m.get("role") == "tool" and m.get("name") in ("Read", "Write", "Edit"):
            c = str(m.get("content", ""))[:200]
            for tok in c.split():
                if "/" in tok and tok not in out:
                    out.append(tok)
                    break
    return out[:20]


def _with_recalled_memory(messages: list, user_text: str) -> list:
    """Append memories that apply to this turn , same placement rule as the
    pinned plan: at the END, where changing content costs no cache reuse."""
    try:
        from greenboost_cli.memory.store import recall, render_block
        mems = recall(user_text, touched_paths=_touched_paths(messages))
        block = render_block(mems)
    except Exception:
        return messages
    if not block:
        return messages
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "kind": "agent_memory_recall", "status": "ok",
            "n_items": len(mems), "chars": len(block),
            "rules": sum(1 for m in mems if m.type == "rule"),
            "scoped": sum(1 for m in mems if m.scope),
        })
    except Exception:
        pass
    return list(messages) + [{"role": "user", "content": block}]


def _with_pinned_todos(messages: list) -> list:
    """Append the live todo list as the last thing the model sees.

    A plan is a context-time object, not something the model retains. Measured
    externally (arXiv 2606.22953), an agent's plan decays several-fold out of
    hidden state within ONE action-observation cycle, and evicting it cost
    34.7 percentage points of task success , which re-surfacing it later did
    not recover. So the todo list has to be IN the request, every request, or
    it is not in the run at all.

    Two placement rules, and the second is the one that is easy to get wrong:

    1. It goes at the END, not the front. Pinning does not mean "put it first".
       This block changes whenever a todo changes, and the engine's KV cache is
       only reusable up to the first token that differs , at the front it would
       invalidate the entire conversation on every status flip. At the end it
       invalidates nothing before it.
    2. It is never written into `session.messages`. It is rendered fresh per
       request from the live store, so it cannot go stale, be compacted away,
       or accumulate copies of itself in history.

    Returns `messages` unchanged when there are no todos , an empty pinned
    block is pure prefix churn for no information.
    """
    try:
        from greenboost_cli.instruments.handlers import _session_todos
    except Exception:
        return messages
    todos = [t for t in (_session_todos or []) if isinstance(t, dict)]
    if not todos:
        return messages
    lines = []
    for t in todos:
        status = str(t.get("status", "pending")).lower()
        mark = {"completed": "x", "in_progress": ">", "in-progress": ">"}.get(status, " ")
        lines.append(f"- [{mark}] {t.get('content') or t.get('task') or ''}".rstrip())
    open_n = sum(1 for t in todos
                 if str(t.get("status", "pending")).lower() not in ("completed", "cancelled"))
    block = ("<pinned-plan>\n"
             "Your current todo list (live, re-rendered every turn , not history):\n"
             + "\n".join(lines)
             + f"\n{open_n} item(s) still open. Continue with the next open item.\n"
               "</pinned-plan>")
    return list(messages) + [{"role": "user", "content": block}]


def execute_turn(
    user_message: str,
    session: ConversationSession,
    settings: dict,
    system_context: str,
) -> Generator:
    """
    Process one user message through the full agentic loop.

    Yields: StreamFragment | ReasoningFragment | InstrumentInvoked |
            InstrumentResult | ApprovalNeeded | QuestionsAsked |
            TurnComplete | LoopGuardTriggered
    """
    session.messages.append({"role": "user", "content": user_message})

    max_turns    = int(settings.get("max_tool_turns", _DEFAULT_MAX_TURNS))
    turn_count   = 0         # tool-execution turns this invocation
    last_key     = ""        # fingerprint of last (tool, args) pair
    repeat_count = 0         # consecutive repeats of last_key
    consec_errors = 0        # consecutive Denied/ERROR results
    _auto_compact_done = False  # guard: at most one auto-compact per invocation
    _empty_retries = 0           # spurious-empty nudge retries (cap: 2)
    _intent_nudges = 0           # "you said you would, now do it" nudges (cap: _INTENT_NUDGE_CAP)

    while True:
        # ── Turn cap ──────────────────────────────────────────────────────
        turn_count += 1
        if turn_count > max_turns:
            msg = f"Loop guard: reached {max_turns}-turn limit — stopping to prevent runaway execution."
            yield LoopGuardTriggered("max_turns", msg)
            break

        session.turn_count += 1
        completed: CompletedResponse | None = None

        # A long idle gap means the engine's cache for this conversation is
        # gone regardless, so this is the one moment when clearing stale tool
        # output costs nothing. Doing it here means the next active stretch
        # starts small rather than compacting under pressure mid-task.
        try:
            from greenboost_cli.workflow.intelligence import microcompact_if_cold
            microcompact_if_cold(session, getattr(session, "_last_turn_ts", 0.0))
        except Exception:
            pass
        session._last_turn_ts = time.time()

        tool_schemas = list(INSTRUMENT_DEFINITIONS)
        if session.mcp_registry is not None:
            # active_tool_schemas(), not tool_schemas: a server made dormant
            # for this session keeps running and stays reachable via
            # ToolSearch, but its schemas are kept OUT of the prompt , which is
            # where the cost actually is.
            _adv = session.mcp_registry.active_tool_schemas()
            tool_schemas.extend(_capped_mcp_schemas(_adv, settings))

        _turn_t0 = time.monotonic()
        _first_token_t = None  # set on the first streamed fragment — marks TTFT boundary
        try:
            _sent = _with_pinned_todos(
                _with_recalled_memory(session.messages, user_message))
            # GB-1: name the chunk that moved. Reuse is all-or-nothing from the
            # first differing token (--cache-reuse is rejected for this hybrid
            # architecture), so the FIRST chunk to change is the one that threw
            # away everything after it , and until this ran, a compaction, a
            # re-rendered system block and a changed tool schema were
            # indistinguishable slow turns.
            try:
                from greenboost_cli.core.prefix import observe as _observe_prefix
                _observe_prefix({
                    "system": system_context,
                    "tools": "".join(str(t.get("name", "")) for t in tool_schemas),
                    "history": "".join(str(m.get("content", ""))
                                       for m in session.messages[:-1]),
                    "memory": "".join(str(m.get("content", "")) for m in _sent
                                      if str(m.get("content", "")).startswith("<recalled-memory>")),
                    "plan": "".join(str(m.get("content", "")) for m in _sent
                                    if str(m.get("content", "")).startswith("<pinned-plan>")),
                    "user": str(session.messages[-1].get("content", "")) if session.messages else "",
                })
            except Exception:
                pass
            for event in generate(
                model=settings["model"],
                system=system_context,
                messages=_sent,
                tool_schemas=tool_schemas,
                settings=settings,
            ):
                if isinstance(event, (StreamFragment, ReasoningFragment)):
                    if _first_token_t is None:
                        _first_token_t = time.monotonic()
                    yield event
                elif isinstance(event, CompletedResponse):
                    completed = event
        except Exception as _gen_err:
            # A context-overflow 400 used to become a generic RuntimeError
            # with no retry, dead-ending the turn (confirmed live: "request
            # exceeds the available context size"). The near-overflow guard
            # below only catches it BEFORE it happens (a short/empty 200
            # response) — this is the "it already happened" counterpart:
            # force-compact once and retry the same user turn.
            from greenboost_cli.inference.adapters import ContextOverflowError
            # Compaction can only shrink HISTORY. When the prompt's fixed
            # overhead (system prompt + every connected MCP server's tool
            # schemas) already exceeds the window on its own, compacting frees
            # a few hundred tokens, the retry fails identically, and the
            # operator gets a raw 400 for their trouble. Live 2026-08-18: ten
            # MCP servers, 238 tools, 29,507 prompt tokens against a 16,384
            # window for a thirty-word message and an empty history.
            #
            # Detect that case from the server's own numbers and explain it
            # instead of burning the one retry on something that cannot work.
            if isinstance(_gen_err, ContextOverflowError):
                _pt = getattr(_gen_err, "prompt_tokens", 0)
                _nc = getattr(_gen_err, "n_ctx", 0)
                _hist_chars = sum(len(str(m.get("content", "")))
                                  for m in session.messages)
                _hist_tokens = _hist_chars // 4
                # If dropping the ENTIRE history still would not fit, the
                # overhead is the problem and no compaction reaches it.
                if _pt and _nc and (_pt - _hist_tokens) >= _nc:
                    _fixed = _pt - _hist_tokens
                    raise ContextOverflowError(
                        f"This request needs {_pt:,} tokens but the server is "
                        f"serving a {_nc:,}-token window.\n\n"
                        f"Roughly {_fixed:,} of those are FIXED overhead , the "
                        f"system prompt plus the tool schemas of every "
                        f"connected MCP server , paid on every request no "
                        f"matter how short your message is. Compacting the "
                        f"conversation cannot reduce it, which is why "
                        f"retrying would fail the same way.\n\n"
                        f"Nothing is broken and nothing was lost , the turn "
                        f"simply never started.\n\n"
                        f"Fix it with either:\n"
                        f"  /llamaserve restart --ctx {max(32768, 1 << (_pt - 1).bit_length())}\n"
                        f"  /mcp disconnect <server>   (fewer tool schemas)",
                        prompt_tokens=_pt, n_ctx=_nc,
                    ) from _gen_err
            if isinstance(_gen_err, ContextOverflowError) and not _auto_compact_done:
                try:
                    from greenboost_cli.environment.settings import invalidate_gb_synapse_ctx_cache
                    invalidate_gb_synapse_ctx_cache()
                except Exception:
                    pass
                if (session.messages
                        and session.messages[-1].get("role") == "assistant"):
                    session.messages.pop()  # drop the failed assistant turn, if any
                from greenboost_cli.workflow.intelligence import _compress_context
                _compress_context(session, settings, force=True)
                _auto_compact_done = True
                yield LoopGuardTriggered(
                    "auto_compact",
                    "Request exceeded the server's context window — "
                    "auto-compacted history and retrying.",
                )
                continue
            raise
        _elapsed = time.monotonic() - _turn_t0

        if completed is None:
            break

        session.messages.append({
            "role":       "assistant",
            "content":    completed.text,
            "tool_calls": completed.tool_calls,
        })

        # For local backends (Ollama, vLLM) that don't return token usage,
        # estimate from text length (~4 chars per token).
        in_tok  = completed.in_tokens
        out_tok = completed.out_tokens
        if not in_tok and not out_tok:
            ctx_chars  = sum(len(str(m.get("content", ""))) for m in session.messages[:-1])
            in_tok     = ctx_chars // 4
            out_tok    = len(completed.text) // 4

        session.total_input_tokens  += in_tok
        session.total_output_tokens += out_tok
        # tok_s must measure DECODE speed only (this dataclass's own field doc:
        # "decode speed for this turn's generation span") — dividing out_tok by
        # _elapsed (turn start → completion) folds prompt-eval/TTFT into the
        # denominator. For a huge or barely-cached prompt, TTFT can be 30s+ on
        # its own; a turn that then emits only a handful of tokens collapses to
        # a near-zero tok_s that looks like a GPU/shim performance problem when
        # it is actually a prompt-processing cost (confirmed live 2026-08-01:
        # dataflux recorded tok_s=0.2 during a context-overflow turn whose real
        # decode throughput, measured directly against the backend, was ~4.9
        # tok/s — a measurement artifact, not a slow serve). Use the generation
        # span (first streamed token → completion) when we saw any streaming;
        # fall back to full _elapsed only for non-streamed/tool-only turns
        # where no generation span was observed.
        _decode_elapsed = (
            (time.monotonic() - _first_token_t) if _first_token_t is not None else _elapsed
        )
        tok_s = out_tok / _decode_elapsed if _decode_elapsed > 0 and out_tok > 0 else 0.0
        yield TurnComplete(in_tok, out_tok, tok_s=tok_s, elapsed_s=_elapsed,
                            is_final=not completed.tool_calls)

        if not completed.tool_calls:
            # ── Context-overflow detection ─────────────────────────────────
            # Detect two signatures of context overflow:
            #   1. Empty response (0 output tokens) — llama-server returns 200
            #      OK with nothing when the prompt fills the -c window exactly.
            #   2. Truncated response (< 60 output tokens while near the context
            #      limit) — the prompt was truncated at the server's -c limit,
            #      the model generated a fragment and stopped (e.g. 13 tokens
            #      "Now I have a thorough picture. Let me write the plan.").
            #      Confirmed by `truncated = 1` in llama-server's slot logs.
            # Both cases: auto-compact history once and retry; on second failure
            # surface an actionable error instead of stopping silently.
            _is_empty = not completed.text.strip()

            # Determine effective context window for near-overflow check
            _ctx_oc = 0
            try:
                from greenboost_cli.environment.settings import gb_synapse_ctx as _gb_ctx_oc
                _ctx_oc = _gb_ctx_oc(settings)
            except Exception:
                pass

            _is_near_overflow = (
                not _is_empty
                and out_tok > 0 and out_tok < 60
                and _ctx_oc > 0
                and in_tok > int(_ctx_oc * 0.80)
            )

            # ── Genuine context overflow (near-limit) ─────────────────────
            if _is_near_overflow:
                if not _auto_compact_done:
                    if (session.messages
                            and session.messages[-1].get("role") == "assistant"
                            and not session.messages[-1].get("tool_calls")):
                        _tail_len = len(
                            (session.messages[-1].get("content") or "").strip()
                        )
                        if _tail_len < 120:
                            session.messages.pop()
                    try:
                        from greenboost_cli.workflow.intelligence import (
                            _compress_context as _ic,
                        )
                        _ic(session, settings, force=True)
                    except Exception:
                        pass
                    _auto_compact_done = True
                    yield LoopGuardTriggered(
                        "auto_compact",
                        f"Response was truncated (~{out_tok} tokens out, "
                        f"~{in_tok:,}/{_ctx_oc:,} context). "
                        f"Auto-compacted history and retrying.",
                    )
                    continue
                # Already compacted — surface overflow error.
                yield LoopGuardTriggered(
                    "empty_response",
                    f"Response still truncated after compaction "
                    f"(~{in_tok:,}/{_ctx_oc:,} tokens). "
                    f"Try /clear to start fresh, or /rag-status to reduce RAG size.",
                )
                break

            # ── Spurious empty turn (context is NOT near the limit) ────────
            # Local finetunes sometimes return an empty turn mid-task (Ollama
            # slot flush, thinking-only output, etc.).  Nudge the model back
            # rather than aborting with a misleading "context full" message.
            if _is_empty:
                _EMPTY_RETRY_CAP = 2
                if _empty_retries < _EMPTY_RETRY_CAP:
                    _empty_retries += 1
                    # Drop the empty assistant turn so the retry sees a clean tail.
                    if (session.messages
                            and session.messages[-1].get("role") == "assistant"
                            and not session.messages[-1].get("tool_calls")
                            and not (session.messages[-1].get("content") or "").strip()):
                        session.messages.pop()
                    # Nudge: short user message asking the model to continue.
                    session.messages.append({
                        "role": "user",
                        "content": (
                            "(Your last reply was empty — no text and no tool call. "
                            "Please continue: either call a tool or give your final answer.)"
                        ),
                    })
                    yield LoopGuardTriggered(
                        "auto_compact",   # re-use "non-fatal" reason so UI just shows "Retrying"
                        f"Empty model turn (retry {_empty_retries}/{_EMPTY_RETRY_CAP}) — nudging.",
                    )
                    continue
                # Exhausted nudge retries — give an accurate diagnosis.
                ctx_hint = (
                    f" (~{in_tok:,}/{_ctx_oc:,} tokens)"
                    if _ctx_oc else f" (~{in_tok:,} tokens in)"
                )
                yield LoopGuardTriggered(
                    "empty_response",
                    f"Model returned empty turns {_EMPTY_RETRY_CAP + 1} times in a row"
                    f"{ctx_hint}. "
                    f"Try enabling thinking (/config thinking=true) or switching model.",
                )
                break

            # ── Intent without action ────────────────────────────────────
            # A model turn reads as "I'll build/create/check/find X" but
            # called no tool. Bounded by three independent guards (see
            # _INTENT_RE / _INTENT_NUDGE_CAP / _INTENT_NUDGE_MAX_LEN above)
            # so ordinary answers are never touched: the phrasing must match,
            # the reply must be short (a genuine long-form answer that happens
            # to contain "I'll check" isn't a stalled hand-off), and this can
            # fire at most _INTENT_NUDGE_CAP times per invocation. Originally
            # gated to turn_count == 1 only — widened 2026-08-02 after a real
            # stall on turn 3 (see _INTENT_RE's comment).
            if (_intent_nudges < _INTENT_NUDGE_CAP
                    and len(completed.text.strip()) < _INTENT_NUDGE_MAX_LEN
                    and _INTENT_RE.search(completed.text)):
                _intent_nudges += 1
                session.messages.append({
                    "role": "user",
                    "content": (
                        "(You described a plan but didn't call any tool. "
                        "Start now: call the tool for the first concrete "
                        "step instead of describing it.)"
                    ),
                })
                yield LoopGuardTriggered(
                    "auto_compact",   # non-fatal reason, UI shows "Retrying"
                    "Model stated intent without acting — nudging to start.",
                )
                continue

            # ── Non-empty, no tool calls: genuine final answer ─────────────
            # Also handle the legacy context-full case: empty + context was
            # already compacted once (shouldn't reach here now, kept as safety).
            if _auto_compact_done and _is_empty:
                ctx_hint = (
                    f" (~{in_tok:,} tokens vs {_ctx_oc:,} limit)"
                    if _ctx_oc else f" (~{in_tok:,} tokens)"
                )
                yield LoopGuardTriggered(
                    "empty_response",
                    f"Model output too short even after auto-compaction{ctx_hint}. "
                    f"Try /clear to start fresh, or /rag-status to reduce RAG size.",
                )
            break

        # ── Execute each requested tool ──────────────────────────────────
        _guard_triggered = False
        # Overlap the read-only calls in this batch before replaying the events
        # in order. Failure here is never fatal: an empty map just means every
        # call takes the serial path it always did.
        try:
            _prefetched = _prefetch_safe_batches(
                completed.tool_calls, session, settings)
        except Exception:
            _prefetched = {}
        for _tc_idx, tc in enumerate(completed.tool_calls):
            # ── Repeat guard ──────────────────────────────────────────────
            try:
                call_key = f"{tc['name']}:{json.dumps(tc['input'], sort_keys=True)}"
            except (TypeError, ValueError):
                call_key = tc["name"]
            if call_key == last_key:
                repeat_count += 1
            else:
                last_key     = call_key
                repeat_count = 1
            if repeat_count >= _REPEAT_LIMIT:
                msg = (
                    f"Loop guard: tool '{tc['name']}' called with identical arguments "
                    f"{repeat_count} times in a row — stopping to prevent a stuck loop."
                )
                yield LoopGuardTriggered("repeat", msg)
                _guard_triggered = True
                break

            yield InstrumentInvoked(tc["name"], tc["input"])

            # ── AskUserQuestion intercept ──────────────────────────────────
            # This tool is handled entirely by the REPL (interactive wizard);
            # it never goes through dispatch().
            if tc["name"] == "AskUserQuestion":
                q_event = QuestionsAsked(tc["input"].get("questions", []))
                yield q_event
                if q_event.answers is None:
                    tool_result = "User cancelled the question — no answer received."
                else:
                    lines = []
                    for ans in q_event.answers:
                        hdr = ans.get("header", "")
                        qtext = ans.get("question", "")
                        av  = ans.get("answers", [])
                        lines.append(f"{hdr}: {qtext}\n  → {', '.join(av)}")
                    tool_result = "\n".join(lines)
                yield InstrumentResult(tc["name"], tool_result, True)
                session.messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.get("id", "ask"),
                    "name":         tc["name"],
                    "content":      tool_result,
                })
                # Questions don't count as errors; reset guards so they never
                # trip the repeat / consecutive-error limits.
                consec_errors = 0
                transient_errors = 0   # timeouts/busy , see the guard below
                last_key      = ""
                repeat_count  = 0
                continue

            # ── ToolSearch intercept ────────────────────────────────────────
            # Local lookup against the registry's full/uncapped schemas
            # (_capped_mcp_schemas only shrinks what's shown to the model,
            # never the registry itself) — never goes through dispatch() or
            # session.mcp_registry.call_tool(), it isn't a real MCP call.
            if tc["name"] == "ToolSearch":
                tool_result = _tool_search(
                    session.mcp_registry, tc["input"].get("query", ""))
                yield InstrumentResult(tc["name"], tool_result, True)
                session.messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.get("id", "toolsearch"),
                    "name":         tc["name"],
                    "content":      tool_result,
                })
                consec_errors = 0
                last_key      = ""
                repeat_count  = 0
                continue

            _plan_violation = _plan_mode_block(tc, session)
            if _plan_violation is not None:
                result, permitted = _plan_violation, False
            else:
                _is_mcp = bool(session.mcp_registry and session.mcp_registry.has_tool(tc["name"]))
                permitted = _is_auto_approved(tc, settings, is_mcp=_is_mcp)
                if not permitted:
                    request = ApprovalNeeded(description=_describe_operation(tc["name"], tc["input"]))
                    yield request
                    permitted = request.granted

                if not permitted:
                    result = "Denied: user rejected this operation"
                elif _tc_idx in _prefetched:
                    result = _prefetched[_tc_idx]      # already ran, concurrently
                elif session.mcp_registry and session.mcp_registry.has_tool(tc["name"]):
                    result = session.mcp_registry.call_tool(tc["name"], tc["input"])
                else:
                    result = dispatch(
                        tc["name"],
                        tc["input"],
                        approval_mode="accept-all",   # already gate-checked above
                        # NemoClaw audit, Phase 3b: settings["_tool_policy"]/
                        # ["_agent_label"] are private conventions (not
                        # public settings), set by run_subagent()/
                        # _invoke_agent() before calling execute_turn(s)ync)
                        # — absent for the interactive REPL, so this stays a
                        # no-op there (policy=None -> dispatch()'s own
                        # PERMISSIVE default).
                        policy=settings.get("_tool_policy"),
                        agent=settings.get("_agent_label", "main"),
                    )

            # ── Consecutive-error guard ───────────────────────────────────
            _is_error = (
                not permitted
                or result.startswith("ERROR")
                or result.startswith("Error")
                or result.startswith("error:")
            )
            if _is_error:
                # Not every failure is the same kind of failure. A Bash timeout
                # or a busy device may well succeed on the next attempt; "file
                # not found" will fail identically forever. Counting both the
                # same trips the loop guard on a run of transient hiccups while
                # letting a genuinely stuck sequence run just as long.
                #
                # Classification adapted from NemoClaw's validation-recovery
                # (see instruments/handlers.classify_tool_failure).
                from greenboost_cli.instruments.handlers import classify_tool_failure
                _kind, _retryable = classify_tool_failure(result)
                if _retryable:
                    # Transient: do not advance the guard, but do not spin
                    # either , cap how many of these one turn may absorb.
                    transient_errors += 1
                    if transient_errors > _TRANSIENT_ERROR_LIMIT:
                        consec_errors += 1
                else:
                    consec_errors += 1
            else:
                consec_errors = 0
                transient_errors = 0
            if consec_errors >= _CONSEC_ERROR_LIMIT:
                yield InstrumentResult(tc["name"], result, permitted)
                session.messages.append({
                    "role": "tool", "tool_call_id": tc["id"],
                    "name": tc["name"], "content": result,
                })
                msg = (
                    f"Loop guard: {consec_errors} consecutive tool errors/denials — "
                    "stopping to avoid repeating a broken sequence."
                )
                yield LoopGuardTriggered("consec_errors", msg)
                _guard_triggered = True
                break

            yield InstrumentResult(tc["name"], result, permitted)

            session.messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "name":         tc["name"],
                "content":      result,
            })

            # ── Mid-turn budget check ─────────────────────────────────────
            # Tool results were appended verbatim with no size check and no
            # re-estimate before this — a single wide result (a 71-line
            # `find`, confirmed live) could push the NEXT request straight
            # past the server's context window with nothing in between to
            # catch it. Re-estimate after every tool result, not just at the
            # top of the next generate() call, and compact proactively
            # before that request is even attempted — the 400-triggered
            # retry above is the safety net for whatever this misses (a
            # single result alone exceeding the post-compaction floor).
            if not _auto_compact_done:
                try:
                    from greenboost_cli.environment.settings import gb_synapse_ctx
                    from greenboost_cli.workflow.intelligence import (
                        _estimate_tokens, _compress_context,
                    )
                    _ctx_budget = int(settings.get("context_window", 0)) or gb_synapse_ctx(settings)
                    if _ctx_budget and _estimate_tokens(session) > int(_ctx_budget * 0.85):
                        _compress_context(session, settings, force=True)
                except Exception:
                    pass

        if _guard_triggered:
            break


def execute_turn_sync(
    user_message: str,
    session: ConversationSession,
    settings: dict,
    system_context: str,
) -> str:
    """Blocking wrapper around execute_turn — returns the final assistant text."""
    last_text = ""
    for event in execute_turn(user_message, session, settings, system_context):
        if isinstance(event, StreamFragment):
            last_text += event.text
        elif isinstance(event, TurnComplete):
            pass  # ignore per-turn token counts
    return last_text.strip()


# ── Permission helpers ─────────────────────────────────────────────────────

# Plan mode permits read-only investigation plus todo/memory bookkeeping —
# everything else (Bash mutations, Write/Edit outside the plan file) is
# blocked outright rather than prompted, so the model self-corrects within
# the turn instead of wandering into a long-running command. This is the
# actual enforcement behind `plan_mode_directive()`'s prompt-level rules in
# workflow/intelligence.py — that text alone does not stop a model from
# ignoring it and running Bash anyway.
_PLAN_MODE_READONLY = {
    "Read", "Glob", "Grep", "WebFetch", "WebSearch",
    "TodoWrite", "TodoRead", "MemoryRead", "MemoryWrite", "Semble",
}


def _plan_mode_block(tc: dict, session: ConversationSession) -> str | None:
    """Return a denial message if *tc* violates active plan-mode
    restrictions, else None (call allowed to proceed to normal permission
    checks). Read-only Bash is still permitted (parity with "auto" mode)."""
    if not getattr(session, "plan_mode", False):
        return None

    name = tc["name"]
    if name in _PLAN_MODE_READONLY:
        return None
    if name == "Bash":
        if is_readonly_command(tc["input"].get("command", "")):
            return None
        return (
            "Blocked by plan mode: Bash commands that change state are not "
            "allowed. Plan mode is read-only — investigate, then write your "
            "findings to the plan file instead of executing changes."
        )

    plan_file = getattr(session, "plan_file", None)
    if name in ("Write", "Edit"):
        if plan_file and str(plan_file) == tc["input"].get("file_path"):
            return None
        return (
            f"Blocked by plan mode: only the plan file ({plan_file or 'not set'}) "
            "may be written. Write your plan there, then end the turn."
        )

    return (
        f"Blocked by plan mode: '{name}' is not a read-only operation. "
        "Finish writing the plan instead of executing changes."
    )


def _is_auto_approved(tc: dict, settings: dict, is_mcp: bool = False) -> bool:
    """Return True if this operation is automatically approved.

    Tiers:
      accept-all  — approve everything (factory agents, subagents)
      autonomous  — approve reads + writes + coding commands (test/build/lint/git)
      auto        — approve read-only tool calls only (default)
      manual      — approve nothing (always prompt)
    """
    perm_mode = settings.get("permission_mode", "auto")

    if perm_mode == "accept-all":
        return True

    if perm_mode == "manual":
        return False

    name = tc["name"]

    # Session-local bookkeeping — never destructive, never worth a prompt,
    # regardless of permission tier.
    if name in ("TodoWrite", "TodoRead"):
        return True

    if perm_mode == "autonomous":
        # Reads, writes, edits all auto-approved in autonomous mode
        if name in ("Read", "Glob", "Grep", "WebFetch", "WebSearch", "Screenshot", "Write", "Edit"):
            return True
        if name == "Bash":
            return is_autonomous_safe(tc["input"].get("command", ""))
        if is_mcp:
            return True   # MCP tools auto-approved when user has enabled autonomous mode
        return False

    # "auto" mode: read-only operations only, ask for everything else
    if name in ("Read", "Glob", "Grep", "WebFetch", "WebSearch", "Screenshot"):
        return True
    if name == "Bash":
        return is_readonly_command(tc["input"].get("command", ""))
    return False   # Write, Edit → ask

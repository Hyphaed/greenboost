"""Workflow intelligence middleware.

pre_process_query() runs before every model call:
  1. auto_rag_inject   — silently search RAG and prepend top results
  2. inject_goals      — always prepend project goals block
  3. compress_context  — if conversation > 60k tokens, summarise old turns

This ensures local tools are used when a RAG index or project memory is
available, and keeps the system prompt lean for gb-synapse's PCIe-bound
local inference.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from greenboost_cli.core.session import ConversationSession


# Threshold at which we compress old conversation turns (estimated tokens)
_COMPRESS_THRESHOLD = 60_000
# Max RAG chunks auto-injected per turn (keep context lean)
_AUTO_RAG_TOP_K = 3
# Min RAG score to auto-inject (higher threshold than manual search)
_AUTO_RAG_MIN_SCORE = 0.25
# Max combined size (chars) of auto-loaded skill bodies appended to the system
# prompt. Bounds prompt-injection blast radius from third-party SKILL.md files.
_MAX_SKILL_INJECT_CHARS = 4000
# How many skills can be matched per turn at most
_MAX_SKILLS_PER_TURN = 3


#: Characters per token assumed by the estimator when nothing better is known.
#: 4.0 is the usual English prose figure. Code, JSON and file paths tokenize
#: nearer 3, which is why the estimate can pass while the real request 400s ,
#: see `note_real_prompt_tokens()`, which replaces this guess with the server's
#: own count as soon as one is available.
_CHARS_PER_TOKEN_DEFAULT = 4.0

#: Live calibration, learned from `usage.prompt_tokens`. Clamped: a ratio
#: outside this band means the sample did not describe the same bytes the
#: estimator counted (a fresh session, a tool-schema change), and trusting it
#: would make the estimate worse than the fixed guess it replaced.
_CHARS_PER_TOKEN_MIN = 2.0
_CHARS_PER_TOKEN_MAX = 6.0

#: EMA weight for a new sample. Low enough that one unusual turn cannot swing
#: the estimator, high enough that a model or prompt change is tracked within a
#: handful of turns.
_CALIBRATION_ALPHA = 0.3

_chars_per_token = _CHARS_PER_TOKEN_DEFAULT
_calibration_samples = 0


def calibration() -> dict:
    """Current estimator calibration , for /status, tests and telemetry."""
    return {"chars_per_token": round(_chars_per_token, 3),
            "samples": _calibration_samples,
            "source": "server usage.prompt_tokens" if _calibration_samples
                      else "default (no server sample yet)"}


def reset_calibration() -> None:
    """Forget what was learned. Used by tests, and by anything that changes the
    tokenizer under us (a model swap)."""
    global _chars_per_token, _calibration_samples
    _chars_per_token = _CHARS_PER_TOKEN_DEFAULT
    _calibration_samples = 0


def note_real_prompt_tokens(actual: int, estimated: int) -> "float | None":
    """Teach the estimator what the server actually counted.

    `actual` is the server's own `usage.prompt_tokens`; `estimated` is what
    this module predicted for the SAME request (history + fixed overhead).
    Returns the resulting chars-per-token, or None when the sample was
    unusable.

    This exists because the estimate is the only thing standing between a turn
    and a 400, and an estimate that is systematically low is worse than no
    estimate at all: it reports headroom that is not there. Live 2026-08-20:
    a compaction retry was fired against a 24576-token window and the real
    request came back at 24654 , a 78-token miss that ended the turn.
    """
    global _chars_per_token, _calibration_samples
    if actual <= 0 or estimated <= 0:
        return None
    # `estimated` was produced at the current ratio, so the implied character
    # count is estimated * ratio. Dividing by the real token count gives the
    # ratio that WOULD have been right.
    implied_chars = estimated * _chars_per_token
    observed = implied_chars / actual
    if not (_CHARS_PER_TOKEN_MIN <= observed <= _CHARS_PER_TOKEN_MAX):
        return None
    _chars_per_token = ((1.0 - _CALIBRATION_ALPHA) * _chars_per_token
                        + _CALIBRATION_ALPHA * observed)
    _calibration_samples += 1
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "kind": "agent_context_edit", "status": "ok", "op": "calibrate",
            "actual_prompt_tokens": int(actual),
            "estimated_prompt_tokens": int(estimated),
            "estimate_error_pct": round(100.0 * (estimated - actual) / actual, 2),
            "chars_per_token": round(_chars_per_token, 3),
            "samples": _calibration_samples,
        })
    except Exception:
        pass
    return _chars_per_token


def _estimate_tokens(session: "ConversationSession") -> int:
    """Token estimate for the session's messages, at the live calibration.

    The chars-per-token divisor is not a constant: it starts at 4.0 and is
    replaced by whatever the server's own `usage.prompt_tokens` implies (see
    `note_real_prompt_tokens`), because 4.0 describes prose and this CLI's
    context is mostly code, JSON and paths.

    Previously counted only `content`, so an assistant message carrying a
    tool_calls list and empty text (the common case — the model called a
    tool and said nothing) counted as ZERO tokens, even though the
    tool_calls JSON (name + full argument payload) is exactly what gets sent
    to the server on the next request. This under-count is one of the
    reasons compaction never fired before the real request overflowed the
    server's context (confirmed live: request hit 8324 tokens against a
    7680-token window). Does NOT include the system prompt or tool
    schemas — callers that assemble those (see intelligence.py's
    pre_process_query, orchestrator.py's mid-turn check) pass them via
    `extra_tokens` to _compress_context instead, since they aren't part of
    session.messages."""
    import json as _json
    chars = 0
    for msg in session.messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    chars += len(block.get("text", ""))
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            try:
                chars += len(_json.dumps(tool_calls))
            except (TypeError, ValueError):
                pass
    return int(chars / _chars_per_token)


def estimate_request_tokens(system: str, tool_schemas: list, messages: list) -> int:
    """Estimate the WHOLE request , system prompt, tool schemas and messages.

    `_estimate_tokens()` only sees `session.messages`, but the fixed overhead is
    most of a local request: ten MCP servers' schemas were measured at ~29.5k
    tokens on this box. A budget check that ignores them is checking the small
    half. Uses the same live calibration, so this number can be compared
    directly against the server's `usage.prompt_tokens`.
    """
    import json as _json
    chars = len(system or "")
    try:
        chars += len(_json.dumps(tool_schemas or []))
    except (TypeError, ValueError):
        pass
    for m in messages or []:
        c = m.get("content", "")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    chars += len(b.get("text", ""))
        tc = m.get("tool_calls")
        if tc:
            try:
                chars += len(_json.dumps(tc))
            except (TypeError, ValueError):
                pass
    return int(chars / _chars_per_token)


def _auto_rag_inject(user_input: str, settings: dict) -> str:
    """Search RAG silently and return injected context prefix, or empty string.

    Search is strictly scoped to the current working directory's indexed folder.
    When cwd is not inside any registered folder, nothing is injected — never
    fall back to the global index so foreign-project content can't leak in.
    """
    try:
        from greenboost_cli.rag.engine import (
            search, format_for_claude, METADATA_FILE, resolve_folder_entry,
        )

        if not METADATA_FILE.exists():
            return ""

        # Strict scope: resolve the registered folder for cwd.
        entry = resolve_folder_entry()
        if entry is None:
            # cwd is not inside any indexed folder — inject nothing.
            return ""

        results = search(
            user_input,
            top_k=_AUTO_RAG_TOP_K,
            min_score=_AUTO_RAG_MIN_SCORE,
            path_prefix=entry["folder"],
        )
        if not results:
            return ""

        context = format_for_claude(results, user_input)
        return context + "\n\n"
    except Exception:
        return ""


def _inject_goals(settings: dict) -> str:
    """Return goals block for the current project, or empty string."""
    try:
        from greenboost_cli.memory.brain import get_goals_summary
        return get_goals_summary()
    except Exception:
        return ""


def compress_text(text: str, target_chars: int = 6000) -> str:
    """Heuristically shrink arbitrary prose/markdown text toward target_chars.

    Strategy:
      1. If already small enough, return as-is.
      2. Split on markdown headers (## / ###). For each section, keep the header
         plus a leading slice; replace the rest with "[…]".
      3. If still too long, repeat with a tighter per-section budget.

    Used by `gb compress` and by optimal-claude's resume flow to shrink
    plan_session.md before injecting it as system context. Deliberately
    LLM-free so it works offline and adds no latency.
    """
    if not text:
        return text
    if len(text) <= target_chars:
        return text

    import re
    header_re = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

    # Collect (start_offset, level, header_line) tuples
    headers = [(m.start(), len(m.group(1)), m.group(0)) for m in header_re.finditer(text)]
    if not headers:
        # No headers — just truncate with an ellipsis marker
        return text[:target_chars - 8].rstrip() + "\n[…]"

    # Build section boundaries
    boundaries = [h[0] for h in headers] + [len(text)]
    sections = []
    for i, (start, _level, header_line) in enumerate(headers):
        end = boundaries[i + 1]
        body = text[start + len(header_line):end]
        sections.append((header_line, body))

    # Allocate budget per section
    overhead = sum(len(h) + 1 for h, _ in sections) + 16  # headers + ellipses
    body_budget = max(target_chars - overhead, len(sections) * 80)
    per_section = max(80, body_budget // len(sections))

    out = []
    for header, body in sections:
        out.append(header)
        body = body.strip("\n")
        if len(body) <= per_section:
            out.append(body)
        else:
            head = body[:per_section].rstrip()
            out.append(head + "\n[…]")
        out.append("")

    result = "\n".join(out).rstrip() + "\n"

    # Second pass if we overshot (rare — happens when budget math under-allocates)
    if len(result) > target_chars * 1.1:
        per_section = max(60, per_section // 2)
        out = []
        for header, body in sections:
            out.append(header)
            body = body.strip("\n")
            if len(body) <= per_section:
                out.append(body)
            else:
                out.append(body[:per_section].rstrip() + " […]")
        result = "\n".join(out).rstrip() + "\n"

    return result


def _msg_text(msg: dict) -> str:
    """Extract plain text from any message format."""
    content = msg.get("content", "")
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content or "")


def _summarize_turn_pair(user_msg: dict, asst_msg: dict | None) -> str:
    """Produce a compact summary of one user↔assistant exchange."""
    user_text = _msg_text(user_msg).strip()
    # Trim the user message to the first meaningful line + a tail
    user_lines = [l for l in user_text.splitlines() if l.strip()]
    if len(user_lines) <= 2:
        user_summary = user_text[:200]
    else:
        user_summary = user_lines[0][:120] + " … " + user_lines[-1][:80]

    if asst_msg is None:
        return f"User: {user_summary}"

    asst_text = _msg_text(asst_msg).strip()
    # Keep first non-empty line + last line (captures conclusion/result)
    asst_lines = [l for l in asst_text.splitlines() if l.strip()]
    tool_calls = asst_msg.get("tool_calls", [])
    tool_names = [tc["name"] for tc in tool_calls] if tool_calls else []

    if asst_lines:
        if len(asst_lines) == 1:
            asst_summary = asst_lines[0][:150]
        else:
            asst_summary = asst_lines[0][:100] + " … " + asst_lines[-1][:80]
    else:
        asst_summary = "(no text)"

    if tool_names:
        asst_summary += f"  [tools: {', '.join(tool_names)}]"

    return f"User: {user_summary}\nAssistant: {asst_summary}"


#: Messages at the FRONT of the history that compaction never moves or
#: rewrites. The opening exchange is the task framing , the cheapest tokens in
#: the session to keep and the ones every later turn is conditioned on.
#: Keeping them pinned in place is what makes the served prompt's leading
#: tokens byte-identical from turn to turn, which is the precondition for the
#: engine reusing its KV cache instead of re-prefilling.
_HEAD_KEEP = 2

#: Marker opening a compaction block. Also how a later compaction RECOGNISES
#: the previous one, so summaries extend instead of being rebuilt , an
#: extended block keeps its existing bytes and therefore its cache reuse.
_MEMORY_MARKER = "[Structured session memory , earlier conversation compacted]"


#: Tool results whose CONTENT can be cleared once it is stale. A file read
#: from twenty steps ago is re-readable; the fact that it was read is what the
#: conversation still needs. Deliberately excludes anything whose output is a
#: decision the run depends on (AskUserQuestion, TodoRead) rather than a
#: lookup.
_COMPACTABLE_TOOLS = frozenset({
    "Read", "Bash", "Grep", "Glob", "Semble", "WebFetch", "WebSearch",
    "Edit", "Write",
})

#: How many of the most recent compactable results keep their content.
_MICROCOMPACT_KEEP = 4

#: What replaces cleared content. Short, and honest about what happened , the
#: model must be able to tell "this was dropped" from "this was empty".
_CLEARED = "[Old tool result content cleared , re-run the tool if you need it again]"

#: Below this many characters a result is not worth clearing.
_MICROCOMPACT_MIN_CHARS = 400


def _microcompact(session, keep: int = _MICROCOMPACT_KEEP) -> int:
    """Clear stale tool-result CONTENT in place. Returns characters freed.

    The cheap alternative to compaction, and it should be tried first.

    Full compaction rewrites the conversation into a summary: it changes
    message count, message order and message text, and everything downstream of
    the first changed token has to be re-prefilled. Measured on this box, that
    is the difference between a ~5.5s turn and a ~140s one.

    Microcompaction changes none of that. Message count, roles and order stay
    identical; only the BODY of tool results the run has moved past is
    replaced. Those bodies are where context actually goes , one file read or
    one `grep -r` is worth more characters than a dozen turns of dialogue , so
    clearing them buys most of the room a compaction would, at a fraction of
    the disruption, and it postpones the expensive event rather than
    performing it.

    What is never cleared: the most recent `keep` results (the ones the model
    is still reasoning about), anything small enough not to matter, and any
    tool not on the compactable list.
    """
    msgs = getattr(session, "messages", None) or []
    idxs = [i for i, m in enumerate(msgs)
            if m.get("role") == "tool"
            and m.get("name") in _COMPACTABLE_TOOLS
            and isinstance(m.get("content"), str)
            and not m["content"].startswith(_CLEARED)
            and len(m["content"]) >= _MICROCOMPACT_MIN_CHARS]
    if len(idxs) <= keep:
        return 0
    freed = 0
    for i in idxs[:-keep] if keep else idxs:
        freed += len(msgs[i]["content"]) - len(_CLEARED)
        msgs[i]["content"] = _CLEARED
    if freed:
        try:
            import gb_dataflux
            gb_dataflux.emit({
                "kind": "agent_context_edit", "status": "ok",
                "op": "microcompact",
                "results_cleared": len(idxs) - keep,
                "results_kept": keep,
                "chars_freed": freed,
            })
        except Exception:
            pass
    return freed


#: Idle gap after which the serving engine's KV cache for this conversation is
#: assumed gone. Past it, editing history costs nothing that was not already
#: lost , which turns the usual rule on its head: normally we protect the
#: prefix, but there is no prefix left to protect.
_COLD_CACHE_GAP_S = 600.0


def microcompact_if_cold(session, last_turn_ts: float) -> int:
    """Clear aggressively when the cache is already cold. Returns chars freed.

    The expensive thing about editing history is losing cache reuse. After a
    long idle gap there is no reuse left to lose, so this is the one moment
    when clearing almost everything is free , and doing it here means the next
    active stretch starts small instead of compacting under pressure mid-task.
    """
    import time as _t
    if not last_turn_ts or (_t.time() - last_turn_ts) < _COLD_CACHE_GAP_S:
        return 0
    return _microcompact(session, keep=1)


#: A tool result trimmed by the last-resort path keeps this much of its head.
#: Enough to see what the command was and how its output started; small enough
#: that trimming a handful of results actually reaches a hard token target.
_HARD_TRIM_KEEP_CHARS = 800

#: Marker left behind by the last-resort path. Distinct from `_CLEARED` because
#: the two mean different things: microcompaction drops a result the run has
#: moved past, this one truncates a result the run may still want, and the
#: model needs to be able to tell those apart.
_HARD_TRIMMED = "\n[… truncated to fit the server's context window , re-run the tool if you need the rest]"


def hard_trim_to_fit(session, budget_tokens: int) -> dict:
    """Last-resort eviction: bring the session under `budget_tokens`.

    Compaction summarises the MIDDLE of a conversation and leaves the last few
    messages verbatim. That is the right default , the tail is what the model
    is still reasoning about , but it means a single large tool result sitting
    in the tail is uncompactable, and no amount of re-compaction moves it.
    Live 2026-08-20: a turn 400'd, auto-compaction ran, the retry 400'd again
    at 24654 tokens against a 24576-token window, and the operator got the raw
    error because the one permitted retry had been spent on a request that
    could not fit.

    Escalates in cost order, stopping as soon as the budget is met:
      1. microcompaction with no keep-window (stale results first),
      2. truncating the largest message bodies, largest first,
      3. dropping the oldest non-pinned messages.

    The pinned head, any structured-memory block and the final user message are
    never touched , losing the question the turn is answering would make the
    retry pointless.

    Returns {"needed", "freed", "met", "steps"} , `met=False` is a real answer
    and the caller must not fire a request on the back of it.
    """
    msgs = getattr(session, "messages", None) or []
    before = _estimate_tokens(session)
    if before <= budget_tokens or not msgs:
        return {"needed": 0, "freed": 0, "met": True, "steps": []}

    steps: list[str] = []

    if _microcompact(session, keep=0):
        steps.append("microcompact")
    if _estimate_tokens(session) <= budget_tokens:
        return _hard_trim_result(session, before, budget_tokens, steps)

    # Protected: the pinned head, any memory block, and the last user message.
    last_user = max((i for i, m in enumerate(msgs)
                     if m.get("role") == "user"), default=-1)

    def _protected(i: int, m: dict) -> bool:
        if i < _HEAD_KEEP or i == last_user:
            return True
        c = m.get("content")
        return isinstance(c, str) and c.startswith(_MEMORY_MARKER)

    while _estimate_tokens(session) > budget_tokens:
        cand = [(len(m["content"]), i) for i, m in enumerate(msgs)
                if isinstance(m.get("content"), str)
                and not _protected(i, m)
                and len(m["content"]) > _HARD_TRIM_KEEP_CHARS + len(_HARD_TRIMMED)]
        if not cand:
            break
        _, i = max(cand)
        msgs[i]["content"] = msgs[i]["content"][:_HARD_TRIM_KEEP_CHARS] + _HARD_TRIMMED
        steps.append(f"truncate[{i}]")

    while _estimate_tokens(session) > budget_tokens:
        drop = next((i for i, m in enumerate(msgs) if not _protected(i, m)), None)
        if drop is None:
            break
        # A tool result must not outlive the assistant message that called it,
        # and vice versa , an orphaned tool message is a malformed request on
        # every backend, which would turn a context problem into a 400 of a
        # different kind.
        msgs.pop(drop)
        while (drop < len(msgs) and msgs[drop].get("role") == "tool"):
            msgs.pop(drop)
        steps.append("drop_oldest")

    return _hard_trim_result(session, before, budget_tokens, steps)


def _hard_trim_result(session, before: int, budget_tokens: int,
                      steps: list) -> dict:
    after = _estimate_tokens(session)
    out = {"needed": max(0, before - budget_tokens),
           "freed": max(0, before - after),
           "met": after <= budget_tokens,
           "steps": steps}
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "kind": "agent_context_edit",
            "status": "ok" if out["met"] else "error",
            "op": "hard_trim",
            "budget_tokens": int(budget_tokens),
            "before_tokens": int(before),
            "after_tokens": int(after),
            "freed_tokens": int(out["freed"]),
            "met": out["met"],
            "steps": len(steps),
        })
    except Exception:
        pass
    return out


def _compress_context(
    session: "ConversationSession", settings: dict, force: bool = False,
    extra_tokens: int = 0,
) -> None:
    """Collapse old turns into a structured summary when context exceeds threshold.

    Strategy:
      - Keep the last 8 messages (4 exchanges) intact.
      - Pair older user/assistant messages and produce a one-line summary per pair.
      - Tool messages (role=tool) are absorbed into the assistant summary.
      - Injects the summary as a user+assistant pair so the conversation
        structure stays valid for all backends.
    Pass force=True to bypass the token threshold (used by /compact and auto-compact).
    Pass extra_tokens to include fixed overhead (system context, RAG injections) in the
    threshold comparison so compaction fires before the real assembled prompt overflows.
    """
    # force=True MUST bypass this floor too — it exists to avoid summarizing a
    # session that's still short, but the auto-compact callers (orchestrator's
    # near-overflow retry, the mid-turn budget check) pass force=True
    # precisely because a request is ABOUT to fail regardless of message
    # count. A turn that grows to 8000+ tokens across 6 messages (one big
    # tool result) used to be uncompactable no matter what, because this
    # check ran unconditionally before force was even consulted.
    if len(session.messages) < 10 and not force:
        return

    if not force:
        estimated = _estimate_tokens(session)
        # Scale threshold to the model's actual context window so local 8k models
        # auto-compress at the right time (not a cloud-calibrated default).
        try:
            from greenboost_cli.environment.settings import gb_synapse_ctx
            ctx_win = int(settings.get("context_window", 0)) or gb_synapse_ctx(settings)
            # Compress at 0.50 so there is room for tool results to accumulate
            # within a turn before the window overflows. A single research turn
            # can read 10+ files and double the context mid-turn; compacting at
            # 50% gives headroom for that growth.
            threshold = max(4096, int(ctx_win * 0.50))
        except Exception:
            threshold = _COMPRESS_THRESHOLD
        if (estimated + extra_tokens) < threshold:
            return

    # Cheapest first. If clearing stale tool-result bodies gets the session
    # back under threshold, the conversation is never rewritten at all , and a
    # rewrite is the thing that costs a re-prefill.
    if _microcompact(session) and not force:
        if (_estimate_tokens(session) + extra_tokens) < threshold:
            return

    keep_count   = 8
    # Three regions, not two. `head` is pinned at the front and never moves;
    # `tail` is the live recent context; only the middle is compacted. The
    # previous shape sliced into two and then `insert(0, ...)`-ed the summary,
    # which put freshly-generated text at token position zero on every single
    # compaction , the worst possible placement for prefix reuse, because the
    # engine's KV cache can only reuse tokens up to the first byte that
    # changed. Measured over 14 days of this box's own sessions, a turn whose
    # prefix was reused >=99% reached first token in ~5.5s; a turn in the
    # 50-90% band took ~140s. Where the summary goes is worth minutes.
    head = session.messages[:_HEAD_KEEP]
    prior_memory = ""
    for m in head:
        if isinstance(m.get("content"), str) and m["content"].startswith(_MEMORY_MARKER):
            head = []                     # degenerate: head IS a memory block
            break
    middle = session.messages[len(head):-keep_count]
    # A memory block already in the middle is absorbed verbatim, so its bytes
    # survive into the new block unchanged rather than being re-summarised
    # into different words that invalidate the cache all over again.
    kept_middle = []
    for m in middle:
        c = m.get("content")
        if isinstance(c, str) and c.startswith(_MEMORY_MARKER):
            prior_memory = c
        elif isinstance(c, str) and c.startswith("[Structured memory loaded"):
            continue
        else:
            kept_middle.append(m)
    old_messages = kept_middle
    if not old_messages and not prior_memory:
        # force=True can fire on a session that's short in MESSAGE COUNT but
        # still over threshold in TOKENS (e.g. one huge tool result) — but
        # when there are fewer than keep_count messages total, there is
        # nothing older than the kept tail to summarize. Injecting the
        # "[Earlier conversation compressed]" pair here would GROW the
        # session instead of compacting it (a real regression this exact
        # early return exists to prevent).
        return
    tail = session.messages[-keep_count:]

    # Build structured memory from old messages:
    #   Files modified, architecture decisions, completed tasks, open TODOs,
    #   key facts, important tool outputs — compress 30K → 3–6K.
    files_modified:   list[str] = []
    arch_decisions:   list[str] = []
    completed_tasks:  list[str] = []
    open_todos:       list[str] = []
    key_facts:        list[str] = []
    tool_outputs:     list[str] = []

    i = 0
    while i < len(old_messages):
        msg  = old_messages[i]
        role = msg.get("role", "")

        if role == "assistant":
            text  = _msg_text(msg).strip()
            tools = msg.get("tool_calls", [])

            # Collect file modifications from tool calls
            for tc in tools:
                tname = tc.get("name", "")
                tinp  = tc.get("input", {})
                if tname in ("Write", "Edit"):
                    fp = tinp.get("file_path", "")
                    if fp and fp not in files_modified:
                        files_modified.append(fp)
                elif tname == "Bash":
                    cmd = tinp.get("command", "")
                    if any(kw in cmd for kw in ("git commit", "git add", "npm install", "pip install")):
                        key_facts.append(f"Ran: {cmd[:80]}")

            # Extract decisions and tasks from assistant text
            for line in text.splitlines():
                ls = line.strip()
                if not ls:
                    continue
                low = ls.lower()
                if any(low.startswith(k) for k in ("decision:", "chose", "approach:", "architecture:")):
                    arch_decisions.append(ls[:120])
                elif any(low.startswith(k) for k in ("todo:", "- [ ]", "next:", "remaining:")):
                    open_todos.append(ls[:100])
                elif any(low.startswith(k) for k in ("done:", "completed:", "fixed:", "✓", "✅")):
                    completed_tasks.append(ls[:100])

            # Keep first non-empty line as a general fact when substantive
            first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
            if 20 < len(first_line) < 200 and not tools:
                key_facts.append(first_line[:140])

        elif role == "tool":
            # Keep brief tool outputs that look important (errors, key values)
            content = str(msg.get("content", ""))[:200]
            if any(kw in content.lower() for kw in ("error", "warning", "success", "created", "found")):
                tname = msg.get("name", "tool")
                tool_outputs.append(f"[{tname}]: {content[:100]}")

        i += 1

    # Assemble structured memory block (target: 3–6K tokens = 12–24K chars)
    sections: list[str] = []
    if files_modified:
        sections.append("## Files Modified\n" + "\n".join(f"- {f}" for f in files_modified[:30]))
    if completed_tasks:
        sections.append("## Completed Tasks\n" + "\n".join(f"- {t}" for t in completed_tasks[:20]))
    if open_todos:
        sections.append("## Open TODOs\n" + "\n".join(f"- {t}" for t in open_todos[:15]))
    if arch_decisions:
        sections.append("## Architecture Decisions\n" + "\n".join(f"- {d}" for d in arch_decisions[:10]))
    if key_facts:
        sections.append("## Key Facts\n" + "\n".join(f"- {f}" for f in key_facts[:20]))
    if tool_outputs:
        sections.append("## Notable Tool Outputs\n" + "\n".join(f"- {o}" for o in tool_outputs[:10]))

    if sections:
        new_block = "\n\n".join(sections)
        if prior_memory:
            # Append-only: the earlier block's bytes are reproduced exactly and
            # the new material is added after them. A rewritten summary would
            # change tokens the engine had already cached; an extended one only
            # adds tokens after them, so everything before the extension point
            # is still reusable.
            summary = prior_memory.rstrip() + "\n\n" + new_block
        else:
            summary = _MEMORY_MARKER + "\n\n" + new_block
    else:
        # Fallback to simple pair summaries when nothing structured was extracted
        pair_parts: list[str] = []
        j = 0
        while j < len(old_messages):
            m = old_messages[j]
            if m.get("role") == "user":
                asst = old_messages[j + 1] if j + 1 < len(old_messages) else None
                if asst and asst.get("role") == "assistant":
                    pair_parts.append(_summarize_turn_pair(m, asst))
                    j += 2
                    while j < len(old_messages) and old_messages[j].get("role") == "tool":
                        j += 1
                else:
                    pair_parts.append(_summarize_turn_pair(m, None))
                    j += 1
            else:
                j += 1
        body = "\n\n".join(pair_parts)
        summary = (prior_memory.rstrip() + "\n\n" + body) if prior_memory else (
            _MEMORY_MARKER + "\n\n" + body)

    # Reassemble in place: pinned head, then the memory block, then the live
    # tail. The head's tokens are byte-identical to the previous request's, so
    # the engine reuses them instead of re-prefilling from zero.
    session.messages = list(head) + [
        {"role": "user", "content": summary},
        {"role": "assistant", "content": "[Structured memory loaded. Continuing task.]"},
    ] + list(tail)

    try:
        import gb_dataflux
        gb_dataflux.emit({
            "kind": "agent_context_edit",
            "status": "ok",
            "op": "compact",
            "head_kept": len(head),
            "middle_compacted": len(old_messages),
            "tail_kept": len(tail),
            "extended_prior": bool(prior_memory),
            "summary_chars": len(summary),
        })
    except Exception:
        pass


def _auto_load_skills(user_input: str, settings: dict) -> str:
    """Match skills against the user's turn and return injected SKILL.md bodies.

    Scans all auto-discovered skill directories (Claude Code accounts, global,
    claude_workflow/commands) plus any user-configured skills_dir.
    Stores matched skill names in settings["_loaded_skills"] for UI display.
    Returns "" when no dirs found, no skill matches, or dependencies unavailable.
    """
    if settings:
        settings["_loaded_skills"] = []
    try:
        from pathlib import Path as _P
        from greenboost_cli.skill.router import (
            discover_all_skill_dirs, match_skills_multi, load_skill_body,
        )
        dirs = discover_all_skill_dirs(settings)
        if not dirs:
            return ""
        matches = match_skills_multi(
            user_input,
            dirs,
            top_k=_MAX_SKILLS_PER_TURN,
            min_score=0.20,
        )
        if not matches:
            return ""
        budget = _MAX_SKILL_INJECT_CHARS
        chunks: list[str] = []
        loaded_names: list[str] = []
        for entry in matches:
            if budget <= 0:
                break
            body = load_skill_body(_P(entry.path), max_chars=min(budget, 2000))
            if not body:
                continue
            header = f"\n\n### SKILL: {entry.name}\n_{entry.description}_\n\n"
            block = header + body
            if len(block) > budget:
                block = block[:budget].rstrip() + "\n[...truncated...]"
            chunks.append(block)
            loaded_names.append(entry.name)
            budget -= len(block)
        if not chunks:
            return ""
        if settings is not None:
            settings["_loaded_skills"] = loaded_names
        return (
            "\n\n[Auto-loaded skills based on this turn]\n"
            + "".join(chunks)
            + "\n"
        )
    except Exception:
        return ""


def plan_mode_directive(settings: dict, session) -> str:
    """Return a plan-mode instruction block to append to the system prompt.

    Activates when the session is marked plan_mode=True. Tells the model to
    only edit the designated plan file and to refrain from other writes.
    """
    if not getattr(session, "plan_mode", False):
        return ""
    plan_file = getattr(session, "plan_file", None)
    if not plan_file:
        return ""
    return (
        "\n\n[PLAN MODE — ACTIVE]\n"
        "You are in plan mode. STRICT RULES:\n"
        "1. Write ONLY to the plan file. Do NOT execute commands, edit source files, "
        "or run tests.\n"
        "2. Do NOT self-approve or exit plan mode. You CANNOT say 'Approved', "
        "'I'll leave plan mode', or anything implying you are exiting. Only the user "
        "exits plan mode by typing /plan-approve or /plan-exit.\n"
        "3. If you have questions before proceeding, call AskUserQuestion — do not "
        "write prose questions.\n"
        f"Plan file: {plan_file}\n"
        "When your plan is ready, end with: 'Plan written. Review with /plan-list, "
        "then /plan-approve when satisfied.'\n"
    )


def autonomous_coding_directive(settings: dict) -> str:
    """Return a system-prompt block for autonomous-coding mode.

    Injected when permission_mode == 'autonomous'. Instructs the model to work
    methodically without pausing for user confirmation, run tests, commit
    incrementally, and write progress notes so the user can audit the work.
    """
    if settings.get("permission_mode") != "autonomous":
        return ""
    goal = settings.get("autonomous_goal", "").strip()
    goal_line = f"\nCurrent objective: {goal}" if goal else ""
    return (
        "\n\n[AUTONOMOUS CODING MODE]\n"
        "The user has stepped away and explicitly consented to unattended execution.\n"
        "You have elevated permissions for coding commands (tests, builds, git commits).\n"
        "Follow this workflow strictly:\n"
        "  1. Understand the objective fully before writing any code.\n"
        "  2. Plan the work in small, verifiable steps. Write a brief plan to PROGRESS.md.\n"
        "  3. Implement one step at a time. Run tests after every non-trivial change.\n"
        "  4. Fix test failures before proceeding to the next step.\n"
        "  5. Commit working increments frequently (git add + git commit). Do NOT git push.\n"
        "  6. Append a one-line progress note to PROGRESS.md at each milestone.\n"
        "  7. If you encounter an irrecoverable blocker (missing credentials, ambiguous\n"
        "     spec, broken environment), write the blocker to PROGRESS.md and stop.\n"
        "Do NOT pause to ask for confirmation — work to completion independently.\n"
        "Hard limits still apply: no git push, no rm -rf, no sudo rm, no DB drops.\n"
        f"{goal_line}\n"
    )


def pre_process_query(
    user_input: str,
    session: "ConversationSession",
    settings: dict,
    system_context: str,
) -> tuple[str, str]:
    """Run all middleware and return (augmented_user_input, augmented_system_context).

    Call this before execute_turn():
        user_input, system_context = pre_process_query(user_input, session, settings, system_context)
        for event in execute_turn(user_input, session, settings, system_context):
            ...
    """
    # 1. Check for pending RAG inject (from /rag-inject command)
    pending = ""
    try:
        from greenboost_cli.slash_commands.rag_cmds import get_pending_inject
        pending = get_pending_inject() or ""
    except ImportError:
        pass

    # 2. Auto-RAG inject — skipped during plan mode. Plan mode already directs
    #    the model to investigate via its own Read/Grep/Glob calls (read-only
    #    gated — see orchestrator._plan_mode_block); blind top-k similarity
    #    search against a broad "audit the whole code"-style prompt returns
    #    near-random snippets (e.g. matching the literal substring "audit"
    #    inside an unrelated apparmor profile path) that pollute context
    #    before the model's own systematic investigation even starts.
    if not pending and not getattr(session, "plan_mode", False):
        pending = _auto_rag_inject(user_input, settings)

    if pending:
        user_input = pending + user_input

    # 3. Goals injection (already in system_context via context_builder, but
    #    re-inject here if goals were added mid-session)
    # Goals are already in system_context from assemble_system_context() — skip.

    # 4. Skill auto-load (trigger-gated) — append SKILL.md bodies to system_context
    skill_block = _auto_load_skills(user_input, settings)
    if skill_block:
        system_context = system_context + skill_block

    # 5. Plan-mode directive — restricts the model to plan-file edits only
    plan_block = plan_mode_directive(settings, session)
    if plan_block:
        system_context = system_context + plan_block

    # 5b. Autonomous-coding directive — expanded permissions + workflow instructions
    auto_block = autonomous_coding_directive(settings)
    if auto_block:
        system_context = system_context + auto_block

    # 5c. Qwen3 no-think prefix — prepend /nothink so the model skips chain-of-thought.
    # think:false in the API body is the hard switch; /nothink in the prompt is belt-and-suspenders.
    _model = settings.get("model", "").lower()
    if (
        ("qwen3" in _model or "qwen36" in _model or "claude-coder" in _model)
        and not settings.get("qwen_thinking", False)
    ):
        system_context = "/nothink\n\n" + system_context

    # 5d. System context compression.
    # The full system prompt (base + two CLAUDE.md files) can exceed 20k chars.
    # On a PCIe-bound MoE at 60-128 tok/s, every 4k extra chars adds ~15 s to the
    # first-token latency. compress_text() is header-aware so structural sections
    # (tool descriptions, operating principles) stay readable; only verbose Notes
    # sections get trimmed. gb-synapse is always local, so this always applies.
    try:
        _sys_limit = int(settings.get("local_sys_ctx_chars", 10000))
        if len(system_context) > _sys_limit:
            system_context = compress_text(system_context, target_chars=_sys_limit)
    except Exception:
        pass

    # 6. Context compression — include assembled system context in the budget so
    # compaction fires before the real prompt (system + messages) hits the window.
    # _estimate_tokens counts only session.messages; system_context (CLAUDE.md, git,
    # RAG summary, goals) can add 10–20k tokens that the estimate previously missed.
    _sys_extra_tokens = len(system_context) // 4
    _compress_context(session, settings, extra_tokens=_sys_extra_tokens)

    return user_input, system_context

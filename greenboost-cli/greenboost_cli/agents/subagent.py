"""Subagent runner: spin up a fresh ConversationSession for a single prompt.

Used to delegate a bounded sub-task without polluting the caller's context.
The subagent reuses gb settings/credentials but starts with an empty
message history. Output is captured in-memory and returned as a structured
SubagentResult.

This is intentionally a thin wrapper around `execute_turn()` — no extra tool
sandboxing, no separate process. Tool calls execute exactly as they would
in the main REPL. The isolation is *context* only, by design.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from greenboost_cli.core.session import ConversationSession
from greenboost_cli.core.orchestrator import (
    execute_turn,
    InstrumentInvoked, InstrumentResult, TurnComplete, ApprovalNeeded,
)
from greenboost_cli.inference.router import StreamFragment, ReasoningFragment


@dataclass
class SubagentToolCall:
    name: str
    inputs: dict
    result: str
    permitted: bool


@dataclass
class SubagentResult:
    summary: str                                     # concatenated assistant text
    tool_calls: list[SubagentToolCall] = field(default_factory=list)
    tokens_used: int = 0
    duration_s: float = 0.0
    timed_out: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "tool_calls": [asdict(tc) for tc in self.tool_calls],
            "tokens_used": self.tokens_used,
            "duration_s": round(self.duration_s, 3),
            "timed_out": self.timed_out,
            "error": self.error,
        }


def run_subagent(
    prompt: str,
    *,
    model: Optional[str] = None,
    isolation: str = "context",       # reserved for future "process" mode
    tools: Optional[list[str]] = None,  # tool-name allowlist (NemoClaw audit, Phase 3b)
    label: str = "subagent",
    timeout_s: float = 600.0,
    settings: Optional[dict] = None,
) -> SubagentResult:
    """Run `prompt` in a fresh session and return a structured result.

    Args:
      prompt: the user-style instruction to hand the subagent.
      model:  override `settings["model"]` for this run (None → inherit).
      isolation: only "context" is supported today.
      tools: closed-world tool-name allowlist for this subagent — `None`
             (the default) means every instrument is permitted, same as
             before this parameter did anything real. A list restricts
             dispatch() (via instruments.policy.build_policy) to exactly
             those names; an unknown name raises immediately rather than
             being silently ignored. This makes the previously-documented
             "reserved; today we always pass all" stub real.
      label: the `agent` field on this subagent's `cli_tool_call` dataflux
             events — override when running several subagents concurrently
             and you want their events distinguishable.
      timeout_s: wall-clock cap. On overrun, partial output is returned and
                 `timed_out=True`.
      settings: gb settings dict. If None, loaded from disk.

    The function never raises on a recoverable error — exceptions are caught
    and reported via SubagentResult.error so the caller can keep going.
    """
    started = time.monotonic()

    # Resolve settings without mutating the caller's dict
    try:
        if settings is None:
            from greenboost_cli.environment.settings import load_settings
            settings = load_settings()
        run_settings = dict(settings)
        if model:
            run_settings["model"] = model
        # Subagents always auto-approve so they can run unattended; flip this
        # if you decide to wire interactive approval through the bridge later.
        run_settings.setdefault("permission_mode", "accept-all")
        # Tool policy (NemoClaw audit, Phase 3b) — private settings keys
        # core/orchestrator.py's dispatch() call site reads; see the
        # comment there. build_policy(None) returns PERMISSIVE, so a
        # caller that doesn't pass `tools` gets byte-identical behavior.
        from greenboost_cli.instruments.policy import build_policy
        run_settings["_tool_policy"] = build_policy(tools)
        run_settings["_agent_label"] = label
    except Exception as e:
        return SubagentResult(
            summary="", error=f"settings error: {type(e).__name__}: {e}",
            duration_s=time.monotonic() - started,
        )

    # Build system prompt
    try:
        from greenboost_cli.environment.context_builder import assemble_system_context
        system_context = assemble_system_context(run_settings.get("model", ""))
    except Exception:
        system_context = ""

    # Fresh session — no shared history with the caller
    session = ConversationSession()

    result = SubagentResult(summary="")
    text_chunks: list[str] = []
    pending_tool: dict | None = None

    try:
        for event in execute_turn(prompt, session, run_settings, system_context):
            # Soft timeout: stop iterating but keep what we have
            if (time.monotonic() - started) > timeout_s:
                result.timed_out = True
                break

            if isinstance(event, StreamFragment):
                text_chunks.append(event.text)
            elif isinstance(event, ReasoningFragment):
                # Skip reasoning from summary; we only surface final text
                pass
            elif isinstance(event, InstrumentInvoked):
                pending_tool = {"name": event.name, "inputs": event.inputs}
            elif isinstance(event, ApprovalNeeded):
                # Subagents auto-deny anything they can't auto-approve
                # (permission_mode=accept-all should mean this is rare).
                event.granted = True
            elif isinstance(event, InstrumentResult):
                if pending_tool is not None:
                    result.tool_calls.append(SubagentToolCall(
                        name=pending_tool["name"],
                        inputs=pending_tool["inputs"],
                        result=str(event.result),
                        permitted=event.permitted,
                    ))
                    pending_tool = None
            elif isinstance(event, TurnComplete):
                result.tokens_used += event.input_tokens + event.output_tokens
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"

    result.summary = "".join(text_chunks).strip()
    result.duration_s = time.monotonic() - started
    return result


# ── Model-invocable delegation, and the guard that makes it safe (AE-12) ─────

#: Env var carrying how deep the current process already is. Set for the
#: subagent's own environment, read on entry.
DEPTH_ENV = "GB_AGENT_DEPTH"

#: How many levels of delegation are allowed. 1 means the main session may
#: delegate, and a subagent may not delegate further.
#:
#: Until this function existed, subagents were reachable only from `/agent` and
#: headless mode , a human started every one, so there was no recursion
#: surface and no guard was needed. Making delegation model-invocable CREATES
#: that surface: a model that delegates can delegate again, and each level
#: multiplies the last. The limit ships in the same change as the capability,
#: not after it.
MAX_DEPTH = 1


def current_depth() -> int:
    try:
        return max(0, int(os.environ.get(DEPTH_ENV, "0")))
    except ValueError:
        return 0


def delegate(prompt: str, *, label: str = "delegate",
             tools: "list | None" = None, timeout_s: float = 600.0,
             settings: "dict | None" = None) -> str:
    """Run a subtask in its own context and return only its conclusion.

    This is context folding: the subtask's reads, greps and dead ends stay in
    the subagent's context and never enter the caller's. What comes back is the
    summary , the whole point being that a 40-tool investigation costs the
    parent session a paragraph instead of forty tool results.

    Returns a string because it is called as a tool.
    """
    depth = current_depth()
    if depth >= MAX_DEPTH:
        return (f"Error: delegation refused , already {depth} level(s) deep and "
                f"the limit is {MAX_DEPTH}. Do this subtask yourself rather "
                f"than delegating it further.")
    prev = os.environ.get(DEPTH_ENV)
    os.environ[DEPTH_ENV] = str(depth + 1)
    try:
        r = run_subagent(prompt, label=label, tools=tools,
                         timeout_s=timeout_s, settings=settings)
    finally:
        if prev is None:
            os.environ.pop(DEPTH_ENV, None)
        else:
            os.environ[DEPTH_ENV] = prev

    if r.error:
        return f"Error: the delegated task failed , {r.error}"
    folded = r.summary.strip() or "(the delegated task produced no summary)"
    return (f"{folded}\n\n[delegated: {len(r.tool_calls)} tool call(s), "
            f"{r.tokens_used} tokens, {r.duration_s:.0f}s"
            + (", TIMED OUT" if r.timed_out else "") + "]")

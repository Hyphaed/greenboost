
"""Dispatches AI tool calls to the appropriate handler."""
from __future__ import annotations

import time

from typing import Callable, Optional

from greenboost_cli.instruments.handlers import (
    handle_skill,
    handle_read, handle_write, handle_edit, handle_shell,
    handle_glob, handle_grep, handle_semble, handle_fetch_url, handle_web_query,
    handle_todo_read, handle_todo_write,
    handle_memory_read, handle_memory_write, handle_screenshot,
)
from greenboost_cli.instruments.policy import PERMISSIVE, ToolPolicy
from greenboost_cli.instruments.safety import is_readonly_command

# Dict-based dispatch table — no if/elif chain.
_DISPATCH: dict[str, Callable] = {
    "Read":      lambda p: handle_read(p["file_path"], p.get("limit"), p.get("offset")),
    "Write":     lambda p: handle_write(p["file_path"], p["content"]),
    "Edit":      lambda p: handle_edit(
                     p["file_path"], p["old_string"], p["new_string"], p.get("replace_all", False)
                 ),
    "Bash":      lambda p: handle_shell(p["command"], p.get("timeout", 120),
                                        bool(p.get("run_in_background"))),
    "TaskOutput": lambda p: __import__(
        "greenboost_cli.instruments.background", fromlist=["output"]
    ).output(str(p.get("task_id", ""))),
    "TaskStop":  lambda p: __import__(
        "greenboost_cli.instruments.background", fromlist=["stop"]
    ).stop(str(p.get("task_id", ""))),
    "Glob":      lambda p: handle_glob(p["pattern"], p.get("path")),
    "Grep":      lambda p: handle_grep(
                     p["pattern"], p.get("path"), p.get("glob"),
                     p.get("output_mode", "files_with_matches"),
                     p.get("case_insensitive", False),
                     p.get("context", 0),
                 ),
    "Semble":    lambda p: handle_semble(
                     p["query"], p.get("repo"), p.get("top_k", 5), p.get("content", "code")
                 ),
    "WebFetch":  lambda p: handle_fetch_url(p["url"], p.get("prompt")),
    "WebSearch": lambda p: handle_web_query(p["query"]),
    "Screenshot": lambda p: handle_screenshot(
                     p["url"], p["output_path"], p.get("width", 1280),
                     p.get("height", 800), p.get("full_page", False),
                 ),
    "TodoRead":    lambda p: handle_todo_read(),
    "TodoWrite":   lambda p: handle_todo_write(p.get("todos", [])),
    "MemoryRead":  lambda p: handle_memory_read(p.get("file")),
    "Skill": handle_skill,
    "Delegate": lambda prompt, label="delegate", **_: __import__(
        "greenboost_cli.agents.subagent", fromlist=["delegate"]
    ).delegate(prompt, label=str(label)[:40]),
    "MemoryWrite": lambda p: handle_memory_write(p["key"], p["content"], p.get("scope", "project")),
}

# These instruments never need user approval.
_ALWAYS_APPROVED = {"Read", "Glob", "Grep", "Semble", "WebFetch", "WebSearch", "Screenshot", "TodoRead", "TodoWrite", "MemoryRead"}


#: Correlates every tool call in one turn. Set by the orchestrator via
#: set_turn_id(); empty outside a turn (a headless one-shot, a subagent that
#: never announced itself) , the field is simply omitted then rather than
#: invented.
_CURRENT_TURN_ID: str = ""


def set_turn_id(turn_id: str) -> None:
    """Tag subsequent tool calls with this turn's id. "" clears it."""
    global _CURRENT_TURN_ID
    _CURRENT_TURN_ID = turn_id or ""


def _emit_cli_tool_call(name: str, decision: str, agent: str,
                        params: Optional[dict] = None,
                        duration_ms: Optional[float] = None,
                        outcome: str = "") -> None:
    """Best-effort dataflux `cli_tool_call` emit (NemoClaw audit, Phase 3d,
    Observability Must-Rule) — never raises, matching gb_dataflux.emit()'s
    own never-raise contract. `params`, when given, is passed through
    instruments.capture.bounded_capture() first, so a token pasted into a
    Bash command (or any credential-shaped field) never reaches
    ~/.local/share/greenboost/dataflux.jsonl in the clear. A DENIED call
    emits exactly as reliably as an ALLOWED one — a decision that leaves no
    trace is the bug this event exists to prevent."""
    try:
        from greenboost_cli.gb_paths import gb_module
        from greenboost_cli.instruments.capture import bounded_capture
        gb_dataflux = gb_module("gb_dataflux")
        event = {
            "kind": "cli_tool_call", "status": decision,
            "name": name, "decision": decision, "agent": agent,
        }
        # Correlation + cost. Without these the event says WHAT was called but
        # not which turn it belonged to, how long it took, or whether it
        # worked , so "where did this 14-minute turn go?" could only be answered
        # by hand-reading the engine log, which is how most of 2026-08-18 was
        # spent. Span-shaped correlation is the idea taken from NemoClaw's
        # trace.ts (trace_id / parent / duration_ms / status); the transport is
        # dataflux, which GreenBoost already has, rather than a second system.
        if _CURRENT_TURN_ID:
            event["turn_id"] = _CURRENT_TURN_ID
        if duration_ms is not None:
            event["duration_ms"] = round(duration_ms, 1)
        if outcome:
            event["outcome"] = outcome
        if params is not None:
            event["args"] = bounded_capture(params)
        gb_dataflux.emit(event)
    except Exception:
        pass


def dispatch(
    name: str,
    params: dict,
    approval_mode: str = "auto",
    approval_fn: Optional[Callable[[str], bool]] = None,
    policy: Optional[ToolPolicy] = None,
    agent: str = "main",
) -> str:
    """
    Execute a named instrument with the given parameters.

    approval_mode:
        "accept-all" — never ask
        "manual"     — always ask
        "auto"       — ask only for write/destructive operations

    policy: a ToolPolicy gating which instruments (and, for Write/Edit,
        which filesystem paths) this call is even allowed to attempt —
        NemoClaw audit, Phase 3a/3b/3c. `None` (the default) means
        `instruments.policy.PERMISSIVE` — every existing caller that
        doesn't pass this stays byte-identical to before this parameter
        existed. Only subagents (`agents/subagent.py`) and factory workers
        (`workflow/factory.py`) pass a restricted policy.
    agent: a label for the `cli_tool_call` dataflux event's `agent` field —
        "main" for the interactive REPL, or a subagent/factory task id.
    """
    active_policy = policy or PERMISSIVE

    handler = _DISPATCH.get(name)
    if handler is None:
        # "Error:" prefix matters: orchestrator.py's consecutive-error guard
        # (result.startswith("Error")) and renderer.py's is_error detection
        # both key off it — an unprefixed "Unknown instrument" rendered as an
        # ordinary grey result and never tripped the guard, so a model stuck
        # calling a name that will never resolve (e.g. a wrong MCP prefix)
        # ran to the turn cap instead of stopping early.
        # AE-5: a name that resolves to nothing at all is the strongest
        # hallucinated-tool signal there is , count it.
        try:
            import gb_dataflux
            gb_dataflux.emit({
                "kind": "agent_tool_schema_miss",
                "status": "error",
                "requested": str(name)[:120],
                "resolved": "",
                "outcome": "unknown_instrument",
                "known_tools": len(_DISPATCH),
            })
        except Exception:
            pass
        return f"Error: Unknown instrument: {name}"

    # Pre-tool hooks (Claude Code compatible)
    try:
        from greenboost_cli.instruments.hooks import run_pre_tool_hooks
        ok, reason = run_pre_tool_hooks(name, params)
        if not ok:
            _emit_cli_tool_call(name, "blocked_hook", agent, params)
            return f"Blocked by hook: {reason}"
    except Exception:
        pass

    # Tool policy gate (NemoClaw audit, Phase 3b) — checked BEFORE the
    # approval gate below, so an out-of-policy tool never reaches an
    # approval prompt at all. Deny-by-default: PERMISSIVE (the default)
    # always permits, so this is a no-op for every unrestricted caller.
    if not active_policy.permits_tool(name):
        _emit_cli_tool_call(name, "denied_policy", agent, params)
        return f"Error: instrument '{name}' is not in this agent's tool policy"

    # Filesystem jail (Phase 3c) — Write/Edit only; Read/Glob/Grep/Bash stay
    # unrestricted by workspace_roots (the CLI must read repo + system
    # state to be useful). Enforced here rather than inside
    # instruments/handlers.py's handle_write/handle_edit so tool-name policy
    # and path-containment policy share one chokepoint and one ToolPolicy
    # object, instead of threading the policy through handler signatures.
    if name in ("Write", "Edit"):
        file_path = params.get("file_path", "")
        if not active_policy.permits_path(file_path):
            _emit_cli_tool_call(name, "denied_policy", agent, params)
            return f"Error: '{file_path}' is outside this agent's workspace"

    # Permission gate for write/destructive instruments
    if approval_mode != "accept-all" and name not in _ALWAYS_APPROVED:
        if approval_mode == "manual" or not _auto_approve(name, params):
            if approval_fn:
                desc = _describe_operation(name, params)
                if not approval_fn(desc):
                    _emit_cli_tool_call(name, "denied_user", agent, params)
                    return "Denied: user rejected this operation"

    # The ALLOWED event is emitted AFTER the call now, so it can carry how long
    # the call took and how it ended. A permission decision still leaves a trace
    # either way , the denial paths above emit immediately, before anything runs.
    _t0 = time.monotonic()

    # Arguments that never parsed reach here as {"_raw": ..., "_parse_error": ...}.
    # Say what actually went wrong. Handing this to the instrument instead
    # produced "invalid parameters ('file_path')" for a truncated payload, which
    # sent the model off to fix a parameter name that was never wrong (and it
    # would then retry the same oversized call).
    try:
        from greenboost_cli.inference.adapters import TOOL_ARG_PARSE_ERROR_KEY
    except Exception:
        TOOL_ARG_PARSE_ERROR_KEY = "_parse_error"
    if isinstance(params, dict) and params.get(TOOL_ARG_PARSE_ERROR_KEY):
        _msg = (f"Error: the arguments for '{name}' could not be read , "
                f"{params[TOOL_ARG_PARSE_ERROR_KEY]}")
        _emit_cli_tool_call(name, "allowed", agent, {"_arg_parse_failed": True},
                            duration_ms=(time.monotonic() - _t0) * 1000.0,
                            outcome="semantic")
        return _msg

    try:
        result = handler(params)
    except (KeyError, TypeError) as e:
        # Malformed tool call (missing/wrong-type required param) — common
        # with local GGUF models' native function-calling, which is far less
        # strict about schema adherence than Claude's. Report it back as a
        # tool error the model can see and correct, instead of letting a raw
        # KeyError/TypeError propagate up and crash the entire turn.
        _msg = (f"Error: instrument '{name}' called with invalid parameters "
                f"({e}); check the required arguments and retry.")
        _emit_cli_tool_call(name, "allowed", agent, params,
                            duration_ms=(time.monotonic() - _t0) * 1000.0,
                            outcome="semantic")
        return _msg

    try:
        from greenboost_cli.instruments.handlers import classify_tool_failure
        _outcome, _ = classify_tool_failure(result if isinstance(result, str) else "")
    except Exception:
        _outcome = ""
    _emit_cli_tool_call(name, "allowed", agent, params,
                        duration_ms=(time.monotonic() - _t0) * 1000.0,
                        outcome=_outcome)

    # Post-tool hooks (informational, non-blocking)
    try:
        from greenboost_cli.instruments.hooks import run_post_tool_hooks
        run_post_tool_hooks(name, params, result)
    except Exception:
        pass

    return result


def _auto_approve(name: str, params: dict) -> bool:
    """Return True if this operation can be auto-approved in 'auto' mode."""
    if name == "Bash":
        return is_readonly_command(params.get("command", ""))
    return False   # Write, Edit → require approval


def _describe_operation(name: str, params: dict) -> str:
    """Human-readable description of an operation, used in permission prompts."""
    if name == "Bash":
        return f"Run: {params.get('command', '')}"
    if name == "Write":
        return f"Write to: {params.get('file_path', '')}"
    if name == "Edit":
        return f"Edit: {params.get('file_path', '')}"
    values = list(params.values())
    return f"{name}({values[:1]})"

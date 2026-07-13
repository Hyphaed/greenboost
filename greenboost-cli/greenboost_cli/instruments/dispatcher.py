"""Dispatches AI tool calls to the appropriate handler."""
from __future__ import annotations

from typing import Callable, Optional

from greenboost_cli.instruments.handlers import (
    handle_read, handle_write, handle_edit, handle_shell,
    handle_glob, handle_grep, handle_semble, handle_fetch_url, handle_web_query,
    handle_todo_read, handle_todo_write,
    handle_memory_read, handle_memory_write,
)
from greenboost_cli.instruments.safety import is_readonly_command

# Dict-based dispatch table — no if/elif chain.
_DISPATCH: dict[str, Callable] = {
    "Read":      lambda p: handle_read(p["file_path"], p.get("limit"), p.get("offset")),
    "Write":     lambda p: handle_write(p["file_path"], p["content"]),
    "Edit":      lambda p: handle_edit(
                     p["file_path"], p["old_string"], p["new_string"], p.get("replace_all", False)
                 ),
    "Bash":      lambda p: handle_shell(p["command"], p.get("timeout", 120)),
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
    "TodoRead":    lambda p: handle_todo_read(),
    "TodoWrite":   lambda p: handle_todo_write(p.get("todos", [])),
    "MemoryRead":  lambda p: handle_memory_read(p.get("file")),
    "MemoryWrite": lambda p: handle_memory_write(p["key"], p["content"], p.get("scope", "project")),
}

# These instruments never need user approval.
_ALWAYS_APPROVED = {"Read", "Glob", "Grep", "Semble", "WebFetch", "WebSearch", "TodoRead", "TodoWrite", "MemoryRead"}


def dispatch(
    name: str,
    params: dict,
    approval_mode: str = "auto",
    approval_fn: Optional[Callable[[str], bool]] = None,
) -> str:
    """
    Execute a named instrument with the given parameters.

    approval_mode:
        "accept-all" — never ask
        "manual"     — always ask
        "auto"       — ask only for write/destructive operations
    """
    handler = _DISPATCH.get(name)
    if handler is None:
        return f"Unknown instrument: {name}"

    # Pre-tool hooks (Claude Code compatible)
    try:
        from greenboost_cli.instruments.hooks import run_pre_tool_hooks
        ok, reason = run_pre_tool_hooks(name, params)
        if not ok:
            return f"Blocked by hook: {reason}"
    except Exception:
        pass

    # Permission gate for write/destructive instruments
    if approval_mode != "accept-all" and name not in _ALWAYS_APPROVED:
        if approval_mode == "manual" or not _auto_approve(name, params):
            if approval_fn:
                desc = _describe_operation(name, params)
                if not approval_fn(desc):
                    return "Denied: user rejected this operation"

    try:
        result = handler(params)
    except (KeyError, TypeError) as e:
        # Malformed tool call (missing/wrong-type required param) — common
        # with local GGUF models' native function-calling, which is far less
        # strict about schema adherence than Claude's. Report it back as a
        # tool error the model can see and correct, instead of letting a raw
        # KeyError/TypeError propagate up and crash the entire turn.
        return f"Error: instrument '{name}' called with invalid parameters ({e}); check the required arguments and retry."

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

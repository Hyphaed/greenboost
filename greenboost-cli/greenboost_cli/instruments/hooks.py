"""
Claude Code-compatible hooks system for greenboost-cli.

Hooks fire before/after tool calls and on user prompt submit.
Configured in ~/.greenboost_cli/hooks.json:

  {
    "PreToolUse": [
      {"matcher": "Bash", "command": "/path/to/pre-bash.sh"},
      {"matcher": "*",    "command": "/path/to/pre-all.sh"}
    ],
    "PostToolUse": [
      {"matcher": "*", "command": "/path/to/post.sh"}
    ],
    "UserPromptSubmit": [
      {"command": "/path/to/prompt-hook.sh"}
    ]
  }

Hook stdin: JSON payload (see _run_hook).
Hook stdout: optional JSON with {"continue": false, "reason": "..."} to block.
Empty or non-JSON stdout = continue.
Non-zero exit = block with stderr as reason.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

_HOOKS_PATH = Path.home() / ".greenboost_cli" / "hooks.json"
_hooks_cache: dict | None = None


def _load_hooks() -> dict:
    global _hooks_cache
    if _hooks_cache is not None:
        return _hooks_cache
    if not _HOOKS_PATH.exists():
        _hooks_cache = {}
        return _hooks_cache
    try:
        _hooks_cache = json.loads(_HOOKS_PATH.read_text(encoding="utf-8"))
    except Exception:
        _hooks_cache = {}
    return _hooks_cache


def _matches(pattern: str, tool_name: str) -> bool:
    return pattern == "*" or pattern.lower() == tool_name.lower()


def _run_hook(command: str, payload: dict) -> tuple[bool, str]:
    """Run a hook command with payload on stdin.

    Returns (should_continue, reason_if_blocked).
    """
    try:
        r = subprocess.run(
            command,
            shell=True,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            reason = (r.stderr or r.stdout or "hook blocked").strip()[:200]
            return False, reason
        out = (r.stdout or "").strip()
        if not out:
            return True, ""
        try:
            data = json.loads(out)
            if not data.get("continue", True):
                return False, data.get("reason", "blocked by hook")
        except json.JSONDecodeError:
            pass   # non-JSON stdout = allow
        return True, ""
    except subprocess.TimeoutExpired:
        return True, ""   # timeout = don't block
    except Exception:
        return True, ""


def run_pre_tool_hooks(tool_name: str, tool_inputs: dict) -> tuple[bool, str]:
    """Run PreToolUse hooks. Returns (allow, reason)."""
    hooks = _load_hooks().get("PreToolUse", [])
    payload = {"hook_event_name": "PreToolUse", "tool_name": tool_name, "tool_input": tool_inputs}
    for hook in hooks:
        if _matches(hook.get("matcher", "*"), tool_name):
            ok, reason = _run_hook(hook["command"], payload)
            if not ok:
                return False, reason
    return True, ""


def run_post_tool_hooks(tool_name: str, tool_inputs: dict, result: str) -> None:
    """Run PostToolUse hooks (informational — cannot block)."""
    hooks = _load_hooks().get("PostToolUse", [])
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": tool_name,
        "tool_input": tool_inputs,
        "tool_response": result[:4000],
    }
    for hook in hooks:
        if _matches(hook.get("matcher", "*"), tool_name):
            _run_hook(hook["command"], payload)


def run_user_prompt_hooks(prompt: str) -> tuple[bool, str]:
    """Run UserPromptSubmit hooks. Returns (allow, reason)."""
    hooks = _load_hooks().get("UserPromptSubmit", [])
    payload = {"hook_event_name": "UserPromptSubmit", "prompt": prompt}
    for hook in hooks:
        ok, reason = _run_hook(hook.get("command", ""), payload)
        if not ok:
            return False, reason
    return True, ""


def run_stop_hooks(session_summary: str) -> None:
    """Run Stop hooks (fires when the agent loop finishes)."""
    hooks = _load_hooks().get("Stop", [])
    payload = {"hook_event_name": "Stop", "session_summary": session_summary}
    for hook in hooks:
        _run_hook(hook.get("command", ""), payload)


def invalidate_cache() -> None:
    """Reload hooks.json on next access (call after /config changes)."""
    global _hooks_cache
    _hooks_cache = None

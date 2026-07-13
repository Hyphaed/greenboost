"""Shared security primitives for gb CLI and MCP server.

Originally lived in mcp/server.py; hoisted so the headless CLI handlers
in cli_headless.py apply the same path-traversal guards and length caps
as the MCP server. Keep these two callers in lockstep — every input
arriving from outside the process passes through here.

Threat model (MCP April-2025 advisory):
  - Prompt injection via oversized context  → length caps (_cap, _MAX_*)
  - Path traversal to read/write arbitrary files → _validate_path
  - Project-name spoofing via control chars or path separators → _safe_project
"""
from __future__ import annotations

from pathlib import Path

_ALLOWED_ROOTS = [
    Path.home(),
    Path("/tmp"),
]

_MAX_TEXT_LEN    = 200_000
_MAX_QUERY_LEN   = 2_000
_MAX_SOURCE_LEN  = 10_000
_MAX_GOAL_LEN    = 1_000
_MAX_PROJECT_LEN = 200
_MAX_PATH_LEN    = 4_096


def validate_path(raw: str, allow_url: bool = False) -> str:
    """Reject traversal and paths outside allowed roots. URLs pass through when allow_url=True."""
    if len(raw) > _MAX_PATH_LEN:
        raise ValueError("Path argument too long.")
    raw = raw.strip()
    if allow_url and raw.lower().startswith(("http://", "https://")):
        return raw
    p = Path(raw).expanduser().resolve()
    for root in _ALLOWED_ROOTS:
        try:
            p.relative_to(root)
            return str(p)
        except ValueError:
            continue
    raise ValueError(
        f"Path '{raw}' is outside allowed directories. "
        "Only paths under your home directory or /tmp are permitted."
    )


def cap(text: str, limit: int) -> str:
    """Silently truncate text to limit characters."""
    return text[:limit]


def safe_project(name: str) -> str:
    """Strip control characters and cap length for project names."""
    name = "".join(c for c in name if c.isprintable() and c not in "/\\")
    return name[:_MAX_PROJECT_LEN]


# Aliases kept under the old leading-underscore names so mcp/server.py can
# re-export them with a one-line `from greenboost_cli.security import …`.
_validate_path = validate_path
_cap = cap
_safe_project = safe_project

"""Declarative per-agent tool policy — NemoClaw audit, Phase 3a.

Design re-implemented from scratch (no NemoClaw code copied) from the shape
NemoClaw's declarative-agents-manifest uses (`tools: {profile, allow, deny}`,
`docs/inference/declarative-agents-manifest.mdx`). Three ideas taken:

  1. Deny-by-default validation: an explicitly-supplied allow/deny list that
     names an instrument this CLI doesn't have is a hard error, not a
     silent no-op — an operator typo must fail loudly.
  2. One ToolPolicy object threaded through dispatch(), not an ad-hoc bag of
     kwargs re-derived at each call site.
  3. The DEFAULT policy is permissive and jail-off, so the interactive REPL
     is byte-for-byte unaffected by this module's existence — only
     subagents (`agents/subagent.py`) and factory workers
     (`workflow/factory.py`) opt in.

This is a model-context-blind, execution-time gate: it decides whether a
call to `dispatch()` is allowed to run at all, which is a different job
from `core/orchestrator.py`'s `_capped_mcp_schemas` (which only shrinks what
the model SEES, never what it can call).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from greenboost_cli.instruments.schemas import INSTRUMENT_DEFINITIONS

ALL_INSTRUMENT_NAMES: frozenset[str] = frozenset(
    d["name"] for d in INSTRUMENT_DEFINITIONS
)


@dataclass(frozen=True)
class ToolPolicy:
    """allow=None means "no allowlist restriction" — every known instrument
    is permitted, subject to `deny`. allow=frozenset(...) means ONLY those
    names are permitted (closed world) — a future instrument this policy
    was never told about is denied by default, not silently granted.

    `deny` always wins over `allow` when a name appears in both.

    `workspace_roots`, when non-empty, jails filesystem-mutating instruments
    (Write/Edit) to a resolved path contained in one of these roots. Empty
    means jail-off. Read/Glob/Grep/Bash are never restricted by this field —
    the CLI must be able to read repo and system state to be useful; only
    writes are dangerous enough to jail.
    """
    allow: "frozenset[str] | None" = None
    deny: "frozenset[str]" = frozenset()
    workspace_roots: "tuple[Path, ...]" = ()

    def permits_tool(self, name: str) -> bool:
        if name in self.deny:
            return False
        if self.allow is None:
            return True
        return name in self.allow

    def permits_path(self, file_path: str) -> bool:
        """True if `file_path` resolves inside one of `workspace_roots`
        (symlinks and `..` traversal resolved away first), or if no
        `workspace_roots` are configured at all (jail-off — the default)."""
        if not self.workspace_roots:
            return True
        try:
            resolved = Path(file_path).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            return False
        for root in self.workspace_roots:
            if resolved == root or root in resolved.parents:
                return True
        return False


# What every ordinary REPL session and any un-opted-in caller gets: no
# allow/deny restriction, no filesystem jail. Byte-identical to
# pre-policy behavior — this module changes nothing until a caller
# explicitly builds a different policy.
PERMISSIVE = ToolPolicy()


def build_policy(
    tools: "list[str] | None" = None,
    *,
    deny: "list[str] | None" = None,
    workspace_roots: "list[str] | None" = None,
) -> ToolPolicy:
    """Construct a ToolPolicy from caller-supplied names.

    `tools=None` (the default) means PERMISSIVE — callers opt IN to a
    restriction; they never have to opt out. Passing `tools=[...]` switches
    to closed-world allow: only those names (plus whatever survives `deny`)
    are permitted. Every name in `tools`/`deny` must be a real instrument —
    an unknown name raises immediately rather than being silently dropped,
    the deny-by-default validation shape this module exists to bring in.
    Canonical objects are rebuilt field-by-field from validated input, never
    copied from the caller's list verbatim."""
    if tools is None and not deny and not workspace_roots:
        return PERMISSIVE

    allow: "frozenset[str] | None" = None
    if tools is not None:
        unknown = sorted(set(tools) - ALL_INSTRUMENT_NAMES)
        if unknown:
            raise ValueError(
                f"unknown instrument name(s) in tool policy allow list: {unknown!r}; "
                f"known instruments: {sorted(ALL_INSTRUMENT_NAMES)}"
            )
        allow = frozenset(tools)

    deny_set = frozenset(deny or ())
    unknown_deny = sorted(deny_set - ALL_INSTRUMENT_NAMES)
    if unknown_deny:
        raise ValueError(
            f"unknown instrument name(s) in tool policy deny list: {unknown_deny!r}; "
            f"known instruments: {sorted(ALL_INSTRUMENT_NAMES)}"
        )

    roots = tuple(Path(r).expanduser().resolve() for r in (workspace_roots or ()))
    return ToolPolicy(allow=allow, deny=deny_set, workspace_roots=roots)

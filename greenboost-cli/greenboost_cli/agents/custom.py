"""Markdown subagents: `name.md` is an agent with its own instructions and tools.

    ---
    name: reviewer              (defaults to the file name)
    description: reviews diffs  (shown in /agents and to the model)
    tools: Read, Grep, Glob     (optional allowlist; omit for all tools)
    ---
    The system instructions for this agent...

Search order, first hit wins: project `.greenboost/agents`, then `~/.greenboost_cli/agents`.
Every agent runs on the one local model; there is no per-agent model."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentDef:
    name: str
    description: str
    body: str
    tools: "list[str] | None" = None
    path: "Path | None" = field(default=None, repr=False)


def _dirs(cwd: "Path | None" = None) -> "list[Path]":
    return [(cwd or Path.cwd()) / ".greenboost" / "agents", Path.home() / ".greenboost_cli" / "agents"]


def _parse(path: Path) -> "AgentDef | None":
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.match(r"---\n(.*?)\n---\n?(.*)", text, re.S)
    meta, body = ({}, text) if not m else ({}, m.group(2))
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip()
    name = (meta.get("name") or path.stem).strip().lower()
    if not re.fullmatch(r"[\w.\-]+", name):
        return None
    tools = [t.strip() for t in meta["tools"].split(",") if t.strip()] if meta.get("tools") else None
    return AgentDef(name, meta.get("description", ""), body.strip(), tools, path)


def discover(cwd: "Path | None" = None) -> "dict[str, AgentDef]":
    found: dict[str, AgentDef] = {}
    for d in _dirs(cwd):
        if d.is_dir():
            for f in sorted(d.glob("*.md")):
                a = _parse(f)
                if a and a.name not in found:
                    found[a.name] = a
    return found


def usable_tools(names: "list[str] | None") -> "tuple[list[str] | None, list[str]]":
    """(known tool names, unknown ones). Unknown names are dropped, never silently granted."""
    if names is None:
        return None, []
    from greenboost_cli.instruments.policy import ALL_INSTRUMENT_NAMES
    known = [n for n in names if n in ALL_INSTRUMENT_NAMES or "*" in n]
    return known, [n for n in names if n not in known]


def build_prompt(agent: AgentDef, task: str) -> str:
    return f"{agent.body}\n\n---\nTask:\n{task}".strip()

"""Which tool calls may run at the same time, and which must not.

Why this exists
---------------
GB-CLI executed every tool call in a turn serially. On this hardware that is
the wrong default: decode runs at roughly 5 tok/s, so the model spends real
time producing a batch of four or five lookups, and then the CLI spends more
real time running them one after another , even though `Read`, `Grep` and
`Glob` do not interact at all.

The pattern is adapted from Claude Code's `toolOrchestration`: partition the
tool calls into CONSECUTIVE runs of concurrency-safe calls, run each run in
parallel, and keep everything else serial. Consecutiveness is the whole trick.
It is what preserves ordering semantics: a `Write` sitting between two `Read`s
splits them into three batches, so the reads never jump the write.

Fail-closed
-----------
`is_concurrency_safe()` answers False for anything it does not positively
recognise , an unknown instrument, an MCP tool whose server declared no
`readOnlyHint`, a Bash command it cannot prove read-only. Being wrong in that direction costs a little wall time. Being wrong
in the other direction runs a mutation out of order, which is a corruption bug.
"""
from __future__ import annotations

import re
import shlex

#: Instruments with no side effects, safe to overlap with each other.
_READ_ONLY_INSTRUMENTS = frozenset({
    "Read", "Glob", "Grep", "Semble", "WebFetch", "WebSearch",
    "TodoRead", "MemoryRead",
})

#: Binaries that only read. argv[0] of every pipeline segment must be here for
#: a Bash call to count as safe. Deliberately short , this list is a security
#: and correctness surface, not a convenience.
_READ_ONLY_BINARIES = frozenset({
    "ls", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg",
    "find", "file", "stat", "du", "df", "pwd", "echo", "printf",
    "sort", "uniq", "cut", "tr", "awk", "sed", "basename", "dirname",
    "realpath", "readlink", "which", "type", "date", "env", "uname",
    "nvidia-smi", "lsmod", "free", "ps", "id", "hostname", "column",
})

#: Shell syntax that can chain, redirect or background , any of these and we
#: stop trying to reason about the command at all.
_UNSAFE_SHELL = re.compile(r"(>>|>|<|;|&&|\|\||&|`|\$\(|\bsudo\b|\brm\b)")


def _bash_is_read_only(command: str) -> bool:
    """True only when every pipeline segment is a known read-only binary.

    `sed`/`awk` are on the read-only list because without a redirect they write
    to stdout , and a redirect is already rejected above. `sed -i` is not: the
    in-place flag is checked explicitly.
    """
    if not command or _UNSAFE_SHELL.search(command):
        return False
    for segment in command.split("|"):
        try:
            argv = shlex.split(segment)
        except ValueError:
            return False            # unbalanced quotes , do not guess
        if not argv:
            return False
        binary = argv[0].rsplit("/", 1)[-1]
        if binary not in _READ_ONLY_BINARIES:
            return False
        if binary == "sed" and any(
                a == "-i" or (a.startswith("-i") and not a.startswith("-in"))
                for a in argv[1:]):
            return False            # in-place edit is a write
        if binary == "find" and any(
                a in ("-delete", "-exec", "-execdir", "-ok", "-fprint")
                for a in argv[1:]):
            return False
    return True


def mcp_declares_read_only(name: str, mcp_registry=None) -> bool:
    """True when the SERVER declared this tool read-only, and only then.

    MCP has a first-class answer to this question , `annotations.readOnlyHint`
    in `tools/list` , so the decision is per-call and comes from the party that
    actually knows, rather than one blanket rule for the whole class.
    Guessing from a tool's NAME is what this deliberately does not do:
    `dataflux_summary` reads and `factory_submit` writes, and nothing in the
    spelling says which. A server that declares nothing stays serial, exactly
    as it was before this function existed.
    """
    if mcp_registry is None or not name.startswith("mcp__"):
        return False
    try:
        for schema in getattr(mcp_registry, "tool_schemas", ()):
            if schema.get("name") != name:
                continue
            ann = schema.get("annotations") or {}
            if hasattr(ann, "model_dump"):        # pydantic ToolAnnotations
                ann = ann.model_dump()
            if not isinstance(ann, dict):
                return False
            return bool(ann.get("readOnlyHint")) and not ann.get("destructiveHint")
    except Exception:
        return False
    return False


def is_concurrency_safe(name: str, tool_input: dict | None = None,
                        mcp_registry=None) -> bool:
    """May this call overlap with other concurrency-safe calls?

    Fail-closed: unknown names, MCP tools whose server declared no
    `readOnlyHint`, and anything unparseable answer False.
    """
    if name in _READ_ONLY_INSTRUMENTS:
        return True
    if name == "Bash":
        cmd = (tool_input or {}).get("command")
        return isinstance(cmd, str) and _bash_is_read_only(cmd)
    if name.startswith("mcp__"):
        return mcp_declares_read_only(name, mcp_registry)
    return False


def partition_tool_calls(tool_calls, safe_predicate=None, mcp_registry=None):
    """Group calls into consecutive (is_safe, [calls]) batches, order preserved.

    A single safe call is returned as its own batch of one; the caller decides
    that a batch of one is not worth a thread.
    """
    pred = safe_predicate or (
        lambda tc: is_concurrency_safe(tc.get("name", ""), tc.get("input"),
                                       mcp_registry))
    batches: list[tuple[bool, list]] = []
    for tc in tool_calls:
        safe = bool(pred(tc))
        if batches and batches[-1][0] and safe:
            batches[-1][1].append(tc)
        else:
            batches.append((safe, [tc]))
    return batches

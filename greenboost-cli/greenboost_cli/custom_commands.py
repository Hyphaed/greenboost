"""Markdown slash commands: a file `name.md` becomes `/name`.

Search order, first hit wins: project `.greenboost/commands`, `~/.greenboost_cli/commands`, then the
commands shipped with greenboost-cli. `$ARGUMENTS` is replaced by whatever follows the command; without
it the arguments are appended. An optional `description:` in YAML front matter shows up in completion
and /help."""
from __future__ import annotations

import re
from pathlib import Path

_SHIPPED = Path(__file__).parent / "prompts" / "commands"


def _dirs(cwd: "Path | None" = None) -> "list[Path]":
    return [(cwd or Path.cwd()) / ".greenboost" / "commands",
            Path.home() / ".greenboost_cli" / "commands", _SHIPPED]


def _split_front(text: str) -> "tuple[dict, str]":
    m = re.match(r"---\n(.*?)\n---\n?(.*)", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, m.group(2)


def discover(cwd: "Path | None" = None) -> "dict[str, tuple[Path, str]]":
    found: dict[str, tuple[Path, str]] = {}
    for d in _dirs(cwd):
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.md")):
            rel = f.relative_to(d).with_suffix("")
            name = ":".join(rel.parts).lower()
            if name in found or not re.fullmatch(r"[\w:.\-]+", name):
                continue
            try:
                meta, _ = _split_front(f.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
            found[name] = (f, meta.get("description", ""))
    return found


def expand(name: str, args: str, cwd: "Path | None" = None) -> "str | None":
    hit = discover(cwd).get(name.lower())
    if not hit:
        return None
    try:
        _, body = _split_front(hit[0].read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    body = body.strip()
    if "$ARGUMENTS" in body:
        return body.replace("$ARGUMENTS", args).strip()
    return f"{body}\n\n{args}".strip() if args else body

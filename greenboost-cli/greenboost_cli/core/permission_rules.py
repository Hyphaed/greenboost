"""Per-pattern allow/deny rules, in Claude Code's `Tool(pattern)` form.

    Bash(git status:*)      command starts with "git status"
    Bash(npm run *)         fnmatch on the whole command
    Read(./secrets/**)      fnmatch on the resolved path
    Edit(*.md)              fnmatch on the path or its basename
    WebFetch(domain:x.org)  host is x.org or a subdomain
    mcp__github__*          a tool name pattern, no argument part

Deny always wins, in every permission mode including accept-all. Allow only
skips the prompt; it never widens what the tool policy permits. Rules live in
~/.greenboost_cli/permissions.json and ./.greenboost/permissions.json:
    {"allow": ["Bash(git status:*)"], "deny": ["Read(./.env)", "Bash(rm -rf *)"]}
"""
from __future__ import annotations

import json
import os
import re
from fnmatch import fnmatchcase
from pathlib import Path
from urllib.parse import urlparse

_RULE = re.compile(r"^([^()]+?)(?:\((.*)\))?$")
_PATH_KEYS = ("file_path", "path", "notebook_path")


def global_file() -> Path:
    return Path.home() / ".greenboost_cli" / "permissions.json"


def _files(cwd: "Path | None" = None) -> "list[Path]":
    return [global_file(), (cwd or Path.cwd()) / ".greenboost" / "permissions.json"]


def load(cwd: "Path | None" = None) -> "dict[str, list[str]]":
    out = {"allow": [], "deny": []}
    for f in _files(cwd):
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for k in out:
            out[k] += [str(r) for r in data.get(k, []) if isinstance(r, str)]
    return out


def save_global(allow: "list[str]", deny: "list[str]") -> None:
    f = global_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"allow": allow, "deny": deny}, indent=2) + "\n")


def parse(rule: str) -> "tuple[str, str | None]":
    m = _RULE.match(rule.strip())
    return (m.group(1).strip(), m.group(2)) if m else (rule.strip(), None)


def _path_of(inp: dict) -> str:
    for k in _PATH_KEYS:
        if inp.get(k):
            return str(inp[k])
    return ""


def _norm(p: str) -> str:
    return os.path.normpath(os.path.expanduser(p))


def _path_match(pattern: str, path: str) -> bool:
    if not path:
        return False
    full = _norm(path if os.path.isabs(os.path.expanduser(path)) else os.path.join(os.getcwd(), path))
    pat = _norm(pattern)
    pat_abs = pat if os.path.isabs(pat) else os.path.join(os.getcwd(), pat)
    for cand in (full, os.path.basename(full)):
        for pp in (pat_abs, pat):
            if fnmatchcase(cand, pp) or fnmatchcase(cand, pp.replace("/**", "/*")):
                return True
    return full == pat_abs or full.startswith(pat_abs.rstrip("*").rstrip("/") + "/") and pattern.endswith("**")


_SPLIT = re.compile(r"&&|\|\||;|\||\n|`|\$\(|\)")
_REDIRECT = re.compile(r"[<>]")


def _bash_match(pat: str, cmd: str, every: bool) -> bool:
    """Match a Bash rule against a possibly chained command.

    `every=True` (allow): each chained segment must match and redirections are
    refused, so `git status; rm -rf ~` can never ride in on `Bash(git status:*)`.
    `every=False` (deny): one matching segment is enough, so `ls && rm -rf x`
    still trips a deny on `rm -rf *`."""
    segs = [x.strip() for x in _SPLIT.split(cmd) if x.strip()]
    if not segs:
        return False
    if every and _REDIRECT.search(cmd):
        return False

    def one(seg: str) -> bool:
        if pat.endswith(":*"):
            pre = pat[:-2]
            return seg == pre or seg.startswith(pre + " ")
        return fnmatchcase(seg, pat)

    return all(one(x) for x in segs) if every else any(one(x) for x in segs)


def matches(rule: str, name: str, inp: dict, every: bool = True) -> bool:
    tool, pat = parse(rule)
    if not fnmatchcase(name, tool):
        return False
    if pat is None or pat in ("", "*"):
        return True
    if name == "Bash":
        return _bash_match(pat, str(inp.get("command", "")).strip(), every)
    if name == "WebFetch" and pat.startswith("domain:"):
        host = (urlparse(str(inp.get("url", ""))).hostname or "").lower()
        d = pat[7:].lower()
        return host == d or host.endswith("." + d)
    path = _path_of(inp)
    if path:
        return _path_match(pat, path)
    return fnmatchcase(str(next(iter(inp.values()), "")), pat)


def decide(name: str, inp: dict, rules: "dict[str, list[str]] | None" = None) -> "str | None":
    """'deny', 'allow' or None (no rule applies)."""
    rules = rules if rules is not None else load()
    if any(matches(r, name, inp, every=False) for r in rules.get("deny", [])):
        return "deny"
    if any(matches(r, name, inp) for r in rules.get("allow", [])):
        return "allow"
    return None

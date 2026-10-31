"""MCP server discovery that reads the same config Claude Code reads.

Sources, lowest priority first (a later source overrides an earlier one by name):

1. ``~/.claude.json``                      root ``mcpServers``
2. ``$CLAUDE_CONFIG_DIR/.claude.json``     root ``mcpServers`` (the active account)
3. that file's ``projects[<cwd>].mcpServers``  (``~/.claude.json``'s is read too)
4. the nearest ``.mcp.json`` from cwd up to $HOME

Why this exists: greenboost-cli used to read only ``.mcp.json``, so on a box
where every server is defined in ``.claude.json`` it connected to nothing and
said "No MCP servers connected". Merge rule ported from optimal-claude's
``python/mcp_effective.py``.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

#: Servers that are this process's own MCP server. Connecting to ourselves
#: would put every GB-CLI tool in the prompt twice.
_SELF_NAMES = frozenset({"greenboost-cli"})

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _expand(value, base: dict):
    """Expand ${VAR} / ${VAR:-default} the way Claude Code does."""
    if isinstance(value, str):
        return _VAR.sub(lambda m: base.get(m.group(1)) or m.group(2) or "", value)
    if isinstance(value, list):
        return [_expand(v, base) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v, base) for k, v in value.items()}
    return value


def user_mcp_path() -> Path:
    """greenboost-cli's own server list, edited by /mcp-add, /mcp-import and /mcp-remove."""
    from greenboost_cli.environment.settings import GB_HOME
    return GB_HOME / "mcp.json"


def read_user_servers() -> dict:
    s = _read_json(user_mcp_path()).get("mcpServers")
    return s if isinstance(s, dict) else {}


def _write_servers(path: Path, servers: dict, *, keep: dict | None = None) -> None:
    """Rewrite `path`'s mcpServers (other keys kept). 0600: a server entry can carry a token."""
    data = dict(keep if keep is not None else _read_json(path))
    data["mcpServers"] = servers
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def add_server(name: str, cfg: dict, *, project: bool = False) -> Path:
    """Persist one server. `project=True` writes ./.mcp.json, else the user file."""
    path = (Path.cwd() / ".mcp.json") if project else user_mcp_path()
    servers = dict(_read_json(path).get("mcpServers") or {})
    servers[name] = cfg
    _write_servers(path, servers)
    return path


def remove_server(name: str) -> list[Path]:
    """Remove `name` from the user file and ./.mcp.json. Returns the files it was removed from.
    Entries that come from Claude Code's own config are never edited here."""
    removed = []
    for path in (user_mcp_path(), Path.cwd() / ".mcp.json"):
        servers = dict(_read_json(path).get("mcpServers") or {})
        if name in servers:
            del servers[name]
            _write_servers(path, servers)
            removed.append(path)
    return removed


def other_claude_servers(have: set) -> dict:
    """Servers configured in ANY Claude Code config on this machine (~/.claude.json and every
    ~/.claude-accounts/*/.claude.json) that are not loaded now: {name: (cfg, where)}. This is
    why a server can work in Claude Code and be absent here: it lives in another account's
    config, and a terminal without CLAUDE_CONFIG_DIR only reads ~/.claude.json."""
    paths = [Path.home() / ".claude.json"] + sorted((Path.home() / ".claude-accounts").glob("*/.claude.json"))
    found: dict = {}
    for p in paths:
        servers = _read_json(p).get("mcpServers")
        for name, cfg in (servers.items() if isinstance(servers, dict) else ()):
            if name in have or name in _SELF_NAMES or name in found:
                continue
            if isinstance(cfg, dict) and not cfg.get("disabled"):
                found[name] = (cfg, p.parent.name if p.parent.name != str(Path.home().name) else "~/.claude.json")
    return found


def claude_json_paths() -> list[Path]:
    """Claude Code's global config files, lowest priority first."""
    paths = [Path.home() / ".claude.json"]
    cfg = os.environ.get("CLAUDE_CONFIG_DIR")
    if cfg:
        acct = Path(cfg).expanduser() / ".claude.json"
        if acct not in paths:
            paths.append(acct)
    return [p for p in paths if p.is_file()]


def find_mcp_json(cwd: Path | None = None) -> Path | None:
    """Walk from cwd up to home looking for .mcp.json."""
    p = cwd or Path.cwd()
    home = Path.home()
    for _ in range(10):
        candidate = p / ".mcp.json"
        if candidate.exists():
            return candidate
        if p == home or p.parent == p:
            break
        p = p.parent
    return None


def _normalise(name: str, cfg, source: str, base_dir: Path | None) -> dict | None:
    """One server entry -> spec, or None when it can't be used."""
    if not isinstance(cfg, dict) or cfg.get("disabled") is True:
        return None
    cfg = _expand(cfg, dict(os.environ))
    ttype = (cfg.get("type") or ("http" if cfg.get("url") and not cfg.get("command") else "stdio")).lower()
    spec: dict = {"name": name, "type": ttype, "source": source,
                  "env": dict(cfg.get("env") or {})}
    if ttype in ("http", "sse", "streamable-http", "streamable_http"):
        if not cfg.get("url"):
            return {**spec, "error": "entry has no 'url'"}
        spec["url"] = cfg["url"]
        spec["headers"] = dict(cfg.get("headers") or {})
        return spec
    if not cfg.get("command"):
        return {**spec, "error": "entry has no 'command'"}
    spec["command"] = [str(cfg["command"])] + [str(a) for a in cfg.get("args", [])]
    cwd = cfg.get("cwd") or (str(base_dir) if base_dir else None)
    if cwd:
        spec["cwd"] = str(Path(cwd).expanduser())
    return spec


#: Servers that each also re-declare greenboost_status / greenboost_capabilities
#: / greenboost_pilot. With two or more connected the model sees the same tools
#: two or three times (5 servers, 88 tools, ~17.8k prompt tokens measured by
#: optimal-claude). GB_MCP_NO_MIRROR=1 makes each declare only its own tools.
_GB_FAMILY_PREFIX = "greenboost"


def _dedup_greenboost_mirrors(specs: dict[str, dict]) -> None:
    family = [s for n, s in specs.items()
              if n.startswith(_GB_FAMILY_PREFIX) and "command" in s]
    if len(family) >= 2:
        for s in family:
            s["env"].setdefault("GB_MCP_NO_MIRROR", "1")


def discover_mcp_servers(cwd: Path | None = None) -> dict[str, dict]:
    """Merge every MCP source into ``{name: spec}``. Never raises.

    A spec is ``{name, type, source, env, command|url, cwd?, headers?, error?}``.
    An entry that can't be used keeps its place with an ``error`` so
    /mcp-servers can say why, instead of one bad entry taking down the rest.
    """
    cwd = (cwd or Path.cwd()).resolve()
    merged: dict[str, dict] = {}

    def _add(servers, source: str, base_dir: Path | None) -> None:
        if not isinstance(servers, dict):
            return
        for name, cfg in servers.items():
            if name in _SELF_NAMES:
                continue
            spec = _normalise(name, cfg, source, base_dir)
            if spec is None:
                merged.pop(name, None)      # explicit disabled:true wins
            else:
                merged[name] = spec

    for path in claude_json_paths():
        data = _read_json(path)
        _add(data.get("mcpServers"), str(path), None)
    for path in claude_json_paths():
        projects = _read_json(path).get("projects")
        if isinstance(projects, dict):
            entry = projects.get(str(cwd))
            if isinstance(entry, dict):
                _add(entry.get("mcpServers"), f"{path} [project]", None)

    _add(read_user_servers(), str(user_mcp_path()), None)          # /mcp-add, above Claude's own config

    mcp_json = find_mcp_json(cwd)
    if mcp_json:
        _add(_read_json(mcp_json).get("mcpServers"), str(mcp_json), mcp_json.parent)

    _dedup_greenboost_mirrors(merged)
    return _apply_filters(merged)


def _patterns(var: str) -> list[str]:
    return [p.strip() for p in os.environ.get(var, "").split(",") if p.strip()]


def _apply_filters(specs: dict[str, dict]) -> dict[str, dict]:
    """GB_MCP_ONLY keeps just the matching servers; GB_MCP_EXCLUDE drops the matching
    ones (comma-separated names or shell-style patterns, e.g. "forge-*,svg-*"). A server
    that is filtered out is never started, unlike /mcp-off which only hides its schemas:
    useful when a heavy server isn't wanted in this tool at all (memory, start-up time)."""
    from fnmatch import fnmatchcase
    only, exclude = _patterns("GB_MCP_ONLY"), _patterns("GB_MCP_EXCLUDE")
    out = {}
    for name, spec in specs.items():
        if only and not any(fnmatchcase(name, p) for p in only):
            continue
        if any(fnmatchcase(name, p) for p in exclude):
            continue
        out[name] = spec
    return out

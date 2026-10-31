"""Adding MCP servers without editing JSON.

    /mcp-add [options] <name> <command> [args...]   a local server:  /mcp-add firefox npx -y @mozilla/firefox-devtools-mcp@latest
    /mcp-add <name> https://host/mcp         a remote server
    /mcp-add <name>                          one already configured in Claude Code (any account)
    /mcp-import [name ... | --all]           list, or copy, servers from your Claude Code configs
    /mcp-remove <name>                       forget one this CLI added

Options for /mcp-add go right after the name, BEFORE the command (everything after the command is the
command's own arguments):  --env KEY=VALUE (repeatable; headers for a URL)   --project (write ./.mcp.json)
    e.g.  /mcp-add github --env GITHUB_TOKEN=xxx npx -y @modelcontextprotocol/server-github

Servers are saved in ~/.greenboost_cli/mcp.json (0600) and connected at once, no restart. Servers
that live in Claude Code's own config are never edited from here.
"""
from __future__ import annotations

import shlex
import shutil
import threading

from greenboost_cli.mcp import config as C
from greenboost_cli.terminal.commands import register_command
from greenboost_cli.terminal.theme import console, GRAY, DIM, LIME, AMBER, TEAL, emit_err, emit_info, emit_ok, emit_warn

#: A first `npx -y pkg@latest` downloads the package before it answers.
_FIRST_START_TIMEOUT_S = 90.0


def _registry(session):
    reg = getattr(session, "mcp_registry", None)
    if reg is None:
        from greenboost_cli.mcp.client import MCPRegistry
        reg = MCPRegistry()
        session.mcp_registry = reg
    return reg


def _loaded(session) -> set:
    reg = getattr(session, "mcp_registry", None)
    return set(reg.server_names()) if reg is not None else set()


def _connect_in_background(session, name: str, cfg: dict, where: str) -> None:
    """Connect a saved server without freezing the prompt; say how it went when it's done."""
    spec = C._normalise(name, cfg, where, None)
    if spec is None or spec.get("error"):
        emit_err(f"{name}: {(spec or {}).get('error', 'disabled')}. Saved in {where}; fix it there.")
        return
    reg = _registry(session)

    def run() -> None:
        ok = reg.add_server(name, spec, init_timeout=_FIRST_START_TIMEOUT_S)
        client = reg._clients.get(name)
        if ok:
            n = len(getattr(client, "tools", []) or [])
            emit_ok(f"{name} connected: {n} tools ({reg.schema_cost(name):,} prompt tokens if advertised in full). "
                    f"/mcp-servers shows it; /mcp-off {name} hides its schemas.")
        else:
            emit_warn(f"{name} saved but did not connect: {getattr(client, 'error', '') or 'no reason given'}. "
                      f"/mcp-servers log {name} shows what it wrote; /mcp-servers reconnect {name} retries.")

    emit_info(f"Connecting {name}… (a first npx start can take up to {int(_FIRST_START_TIMEOUT_S)} s; the prompt stays usable)")
    threading.Thread(target=run, daemon=True, name=f"gb-mcp-add-{name}").start()


def _parse(args: str):
    """(name, project, env, command_tokens) or an error string."""
    try:
        toks = shlex.split(args)
    except ValueError as e:
        return f"couldn't read that line ({e}); quote anything with spaces"
    if not toks:
        return "usage: /mcp-add <name> [--env K=V] <command> [args...]   |   /mcp-add <name> <url>   |   /mcp-add <name>"
    name, rest = toks[0], toks[1:]
    project, env, cmd, i = False, {}, [], 0
    while i < len(rest):
        t = rest[i]
        if t == "--project":
            project = True
        elif t == "--env" and i + 1 < len(rest) and "=" in rest[i + 1]:
            k, v = rest[i + 1].split("=", 1)
            env[k] = v
            i += 1
        elif t == "--":
            cmd = rest[i + 1:]
            break
        else:
            cmd = rest[i:]
            break
        i += 1
    if not name or name.startswith("-") or "/" in name or " " in name:
        return f"'{name}' isn't a usable server name (letters, digits, - and _ ; no slashes or spaces)"
    return name, project, env, cmd


def _scrub_secrets_from_history(env: dict) -> None:
    """A value typed after --env (a token) is a line in the input history file: drop those
    lines, as /huggingface-token does. It stays in the terminal's scrollback, so say so."""
    from greenboost_cli.hf_token import scrub_history
    removed = 0
    for v in env.values():
        if len(v) >= 8:
            removed += scrub_history(v)
    if env and any(len(v) >= 8 for v in env.values()):
        emit_info("The --env value is visible in this terminal's scrollback"
                  + (f"; removed from the input history ({removed} entr{'y' if removed == 1 else 'ies'})." if removed else "."))


def cmd_mcp_add(args: str, session, settings) -> bool:
    parsed = _parse(args)
    if isinstance(parsed, str):
        emit_err(parsed)
        return True
    name, project, env, cmd = parsed
    if not cmd:                                            # /mcp-add <name>: look in Claude Code's configs
        other = C.other_claude_servers(_loaded(session))
        if name in other:
            cfg, src = other[name]
            path = C.add_server(name, dict(cfg), project=project)
            emit_ok(f"Copied {name} from {src} into {path}.")
            _connect_in_background(session, name, dict(cfg), str(path))
            return True
        if name in _loaded(session):
            emit_info(f"{name} is already loaded. /mcp-servers shows its state.")
        else:
            emit_err(f"No command given and no server called '{name}' in your Claude Code configs. "
                     f"Usage: /mcp-add {name} <command> [args...]   (e.g. /mcp-add {name} npx -y some-mcp-server@latest)")
            if other:
                emit_info("Available to import: " + ", ".join(sorted(other)) + "   (/mcp-add <name>)")
        return True
    if len(cmd) == 1 and cmd[0].startswith(("http://", "https://")):
        cfg = {"type": "http", "url": cmd[0]}
        if env:
            cfg["headers"] = env                          # for a remote server --env carries headers
    else:
        if shutil.which(cmd[0]) is None:
            emit_err(f"'{cmd[0]}' isn't an installed command (not on PATH), so nothing was saved. "
                     f"Check the spelling, or give the full path.")
            return True
        cfg = {"command": cmd[0], "args": cmd[1:]}
        if env:
            cfg["env"] = env
    path = C.add_server(name, cfg, project=project)
    emit_ok(f"Saved {name} in {path}" + (" (owner-only file)" if not project else ""))
    _scrub_secrets_from_history(env)
    _connect_in_background(session, name, cfg, str(path))
    return True


def cmd_mcp_import(args: str, session, settings) -> bool:
    other = C.other_claude_servers(_loaded(session))
    toks = args.split()
    if not toks:
        if not other:
            emit_info("Every server in your Claude Code configs is already loaded here.")
            return True
        console.print(f"\n  [{TEAL}]◈  Servers in your Claude Code configs that aren't loaded here[/]")
        for n, (cfg, src) in sorted(other.items()):
            preview = (cfg.get("url") or " ".join([str(cfg.get("command", ""))] + [str(a) for a in cfg.get("args", [])[:2]]))[:70]
            console.print(f"  [{LIME}]{n:<22}[/] [{GRAY}]{src:<26}[/] [{DIM}]{preview}[/]")
        console.print(f"\n  [{DIM}]/mcp-import <name> [name ...]   or   /mcp-import --all[/]\n")
        return True
    names = sorted(other) if toks == ["--all"] else toks
    for n in names:
        if n not in other:
            emit_err(f"'{n}' isn't in any other Claude Code config" + (" (already loaded?)" if n in _loaded(session) else ""))
            continue
        cfg, src = other[n]
        path = C.add_server(n, dict(cfg))
        emit_ok(f"Copied {n} from {src}")
        _connect_in_background(session, n, dict(cfg), str(path))
    return True


def cmd_mcp_remove(args: str, session, settings) -> bool:
    name = args.strip()
    if not name:
        emit_err("usage: /mcp-remove <name>")
        return True
    files = C.remove_server(name)
    reg = getattr(session, "mcp_registry", None)
    unloaded = bool(reg and reg.remove_server(name))
    if files or unloaded:
        emit_ok(f"Removed {name}" + (f" from {', '.join(str(f) for f in files)}" if files else "") + ".")
        if unloaded and not files:
            emit_warn(f"It came from Claude Code's own config, which isn't edited here: it is gone for this session "
                      f"and will load again next start. To keep it out: export GB_MCP_EXCLUDE={name}")
    else:
        emit_err(f"No server called '{name}' here.")
    return True


register_command("mcp-add", cmd_mcp_add,
                 "Add an MCP server:  /mcp-add <name> <command> [args] | <url> | (just a name from Claude Code)")
register_command("mcp-import", cmd_mcp_import, "List or copy MCP servers from your Claude Code configs")
register_command("mcp-remove", cmd_mcp_remove, "Remove an MCP server this CLI added")

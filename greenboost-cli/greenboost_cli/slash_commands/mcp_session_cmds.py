"""Per-session control over which MCP servers put their tools in the prompt.

Why this exists
---------------
Connecting ten MCP servers costs nothing at rest. Their TOOL SCHEMAS do: they
are re-sent, in full, on every single request. Measured on the reference box ,
ten servers, 238 tools , that was ~7k prompt tokens per turn after the schema
degradation and count cap already ran, out of a 13k prompt.

Prefill is super-linear in prompt length (observed: each 2048-token chunk took
+9.1 s longer than the last), so trimming the front of the prompt pays back
more than proportionally. Dropping the servers a task does not need is the
biggest lever the operator has , larger than any decode tuning in this
codebase, and it costs no VRAM.

Dormant, not disconnected
-------------------------
`/mcp-off` does NOT kill the subprocess or drop the connection. The server
stays up and every one of its tools stays callable: ToolSearch still finds
them and call_tool() still routes to them. Only the schemas are withheld from
the prompt. That means the choice is cheap to get wrong , flip it back with
`/mcp-on` and nothing had to restart.

Commands
--------
    /mcp-servers            what is configured/connected, why any failed, what it costs
    /mcp-servers reconnect [server]   restart one server (or every failed one)
    /mcp-servers log <server>         last stderr lines the server wrote
    /mcp-off  <a> [b ...]   stop advertising these servers' tools
    /mcp-on   <a> [b ...]   advertise them again
    /mcp-only <a> [b ...]   advertise ONLY these (everything else dormant)
    /mcp-all                advertise everything again
"""
from __future__ import annotations

from greenboost_cli.terminal.commands import register_command
from greenboost_cli.terminal.theme import (
    console, GRAY, DIM, LIME, AMBER, TEAL, VIOLET, emit_ok, emit_err, emit_info,
)

#: settings key holding the dormant set, so a choice survives a restart.
_SETTING = "mcp_dormant_servers"


def _registry(session):
    reg = getattr(session, "mcp_registry", None)
    if reg is None or not getattr(reg, "server_names", None):
        return None
    return reg


def _persist(settings: dict, reg) -> None:
    settings[_SETTING] = sorted(reg.dormant_servers)
    try:
        from greenboost_cli.environment.settings import save_settings
        save_settings(settings)
    except Exception:
        pass        # a preference that failed to persist is not worth an error


def apply_saved_dormant(session, settings: dict) -> None:
    """Re-apply the saved dormant set after connect_all(). Called from the REPL."""
    reg = _registry(session)
    if reg is None:
        return
    saved = settings.get(_SETTING) or []
    if saved:
        reg.set_dormant(set(saved))


def _summary(reg) -> tuple[int, int]:
    """(advertised_tokens, dormant_tokens) , approximate, for the UI."""
    live = sum(reg.schema_cost(s) for s in reg.server_names()
               if s not in reg.dormant_servers)
    off = sum(reg.schema_cost(s) for s in reg.dormant_servers)
    return live, off


_NONE = ("No MCP servers configured. Looked in ~/.claude.json, "
         "$CLAUDE_CONFIG_DIR/.claude.json and the nearest .mcp.json.")


def _sent_to_model(reg, settings: dict) -> "tuple[int, int, int]":
    """What the prompt really carries: the active tools after the context-window cap
    (core/orchestrator.py _capped_mcp_schemas) , not the sum of every full schema.
    Returns (approx tokens, tools listed, tools active)."""
    import json as _json
    active = reg.active_tool_schemas()
    try:
        from greenboost_cli.core.orchestrator import _capped_mcp_schemas
        sent = _capped_mcp_schemas(active, settings)
    except Exception:
        sent = active
    listed = [s for s in sent if not str(s.get("name", "")).startswith("_mcp_")]
    return sum(len(_json.dumps(s)) for s in sent) // 4, len(listed), len(active)


def _show(session, settings: dict | None = None) -> None:
    reg = _registry(session)
    if reg is None:
        emit_info(_NONE)
        return
    names = sorted(reg.server_names())
    if not names:
        emit_info(_NONE)
        return

    by_name = {s["server"]: s for s in reg.status()}
    console.print()
    console.print(f"  [{VIOLET}]◈  MCP servers[/]  [{GRAY}]this session[/]")
    # Pad the PLAIN text, then colour it: padding a string that already holds markup counts
    # the markup, which is what shifted every row with a name over 18 characters.
    w = max(18, max(len(n) for n in names))
    console.print(f"  [{DIM}]{'server':<{w}} {'state':<9} {'tools':>6} {'~prompt tokens':>15}[/]")
    for n in names:
        st = by_name.get(n, {})
        if not st.get("connected"):
            if st.get("error"):
                console.print(f"  [{AMBER}]{n:<{w}}[/] [{AMBER}]{'failed':<9}[/] [{GRAY}]{st['error'][:100]}[/]")
            else:
                console.print(f"  [{TEAL}]{n:<{w}}[/] [{DIM}]{'connecting…':<9}[/]")
            continue
        dormant = n in reg.dormant_servers
        n_tools = sum(1 for t in reg.tool_schemas
                      if reg.server_of(t.get("name", "")) == n)
        cost = reg.schema_cost(n)
        state, scol = ("dormant", DIM) if dormant else ("active", LIME)
        console.print(f"  [{TEAL}]{n:<{w}}[/] [{scol}]{state:<9}[/] {n_tools:>6} "
                      f"[{DIM if dormant else AMBER}]{cost:>15,}[/]")

    live, off = _summary(reg)
    console.print()
    sent_tok, listed, total = _sent_to_model(reg, settings or {})
    console.print(f"  [{GRAY}]sent to the model per request:[/] [{AMBER}]~{sent_tok:,} tokens[/] "
                  f"[{GRAY}]({listed} of {total} tools listed; the rest are found with ToolSearch)[/]")
    if live > sent_tok * 1.5:
        console.print(f"  [{DIM}]their full schemas would be ~{live:,} tokens; the prompt keeps "
                      f"a trimmed list within a share of the context window[/]")
    if off:
        console.print(f"  [{GRAY}]dormant:[/] [{DIM}]~{off:,} tokens saved per request[/] "
                      f"[{GRAY}], still callable via ToolSearch[/]")
    try:
        from greenboost_cli.mcp.config import other_claude_servers
        other = other_claude_servers(set(reg.server_names()))
    except Exception:
        other = {}
    if other:
        names = ", ".join(sorted(other))
        console.print(f"\n  [{AMBER}]also configured in Claude Code, not loaded here:[/] [{GRAY}]{names}[/]")
        console.print(f"  [{DIM}]add one with /mcp-add <name>  ·  all with /mcp-import --all  ·  "
                      f"or a new one: /mcp-add <name> <command> [args][/]")
    console.print(f"\n  [{DIM}]/mcp-off <server>   /mcp-on <server>   "
                  f"/mcp-only <server ...>   /mcp-all[/]")
    console.print(f"  [{DIM}]Dormant means the schemas leave the prompt. Nothing "
                  f"disconnects and nothing becomes uncallable.[/]")


def _names_from(args: str) -> list[str]:
    return [a for a in args.replace(",", " ").split() if a]


def _mutate(args: str, session, settings: dict, mode: str) -> None:
    reg = _registry(session)
    if reg is None:
        emit_info(_NONE)
        return
    known = set(reg.server_names())
    if mode == "all":
        reg.set_dormant(set())
        _persist(settings, reg)
        emit_ok("All MCP servers advertising again.")
        _show(session, settings)
        return

    wanted = _names_from(args)
    if not wanted:
        emit_err(f"Which server? Connected: {', '.join(sorted(known)) or 'none'}")
        return
    unknown = [w for w in wanted if w not in known]
    if unknown:
        emit_err(f"Not connected: {', '.join(unknown)}  "
                 f"(have: {', '.join(sorted(known))})")
        return

    before_live, _ = _summary(reg)
    if mode == "off":
        reg.set_dormant(reg.dormant_servers | set(wanted))
    elif mode == "on":
        reg.set_dormant(reg.dormant_servers - set(wanted))
    elif mode == "only":
        reg.set_dormant(known - set(wanted))
    _persist(settings, reg)

    after_live, _ = _summary(reg)
    delta = before_live - after_live
    if delta > 0:
        emit_ok(f"~{delta:,} fewer prompt tokens per request "
                f"(now ~{after_live:,}). Those tools stay callable via ToolSearch.")
    elif delta < 0:
        emit_ok(f"~{-delta:,} more prompt tokens per request (now ~{after_live:,}).")
    else:
        emit_info(f"No change , still ~{after_live:,} prompt tokens per request.")
    _show(session, settings)


def _reconnect(args: str, session) -> None:
    import threading
    reg = _registry(session)
    if reg is None:
        emit_info(_NONE)
        return
    wanted = _names_from(args) or [s["server"] for s in reg.status() if not s["connected"]]
    unknown = [w for w in wanted if w not in set(reg.server_names())]
    if unknown:
        emit_err(f"Unknown server: {', '.join(unknown)}  (have: {', '.join(sorted(reg.server_names()))})")
        return
    if not wanted:
        emit_info("Every server is already connected.")
        return
    emit_info(f"Reconnecting {', '.join(wanted)}…")

    def _run():
        for n in wanted:
            reg.reconnect(n)
    # Background: a slow server must not freeze the prompt. /mcp-servers shows
    # the outcome (connecting… -> active / failed: reason).
    threading.Thread(target=_run, daemon=True, name="gb-mcp-reconnect").start()


def _log(args: str, session) -> None:
    reg = _registry(session)
    names = _names_from(args)
    if reg is None or not names:
        emit_err("Usage: /mcp-servers log <server>")
        return
    client = reg._clients.get(names[0])
    if client is None:
        emit_err(f"Unknown server: {names[0]}")
        return
    tail = list(getattr(client, "stderr_tail", []) or [])
    if not tail:
        emit_info(f"{names[0]} wrote nothing to stderr.")
        return
    for line in tail[-40:]:
        console.print(f"  [{DIM}]{line[:200]}[/]")


def _cmd_servers(args: str, session, settings: dict) -> bool:
    sub, _, rest = args.strip().partition(" ")
    if sub == "reconnect":
        _reconnect(rest, session)
    elif sub == "log":
        _log(rest, session)
    else:
        _show(session, settings)
    return True


def _cmd_off(args: str, session, settings: dict) -> bool:
    _mutate(args, session, settings, "off")
    return True


def _cmd_on(args: str, session, settings: dict) -> bool:
    _mutate(args, session, settings, "on")
    return True


def _cmd_only(args: str, session, settings: dict) -> bool:
    _mutate(args, session, settings, "only")
    return True


def _cmd_all(args: str, session, settings: dict) -> bool:
    _mutate("", session, settings, "all")
    return True


register_command("mcp-servers", _cmd_servers,
                 "MCP servers in this session, and what each costs the prompt")
register_command("mcp-off", _cmd_off,
                 "Stop advertising a server's tools (stays callable via ToolSearch)")
register_command("mcp-on", _cmd_on, "Advertise a server's tools again")
register_command("mcp-only", _cmd_only, "Advertise ONLY these servers")
register_command("mcp-all", _cmd_all, "Advertise every connected server again")

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
    /mcp-servers            what is connected, what it costs, what is dormant
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


def _show(session) -> None:
    reg = _registry(session)
    if reg is None:
        emit_info("No MCP servers connected in this session.")
        return
    names = sorted(reg.server_names())
    if not names:
        emit_info("No MCP servers connected in this session.")
        return

    console.print()
    console.print(f"  [{VIOLET}]◈  MCP servers[/]  [{GRAY}]this session[/]")
    console.print(f"  [{DIM}]{'server':<18} {'state':<9} {'tools':>6} {'~prompt tokens':>15}[/]")
    for n in names:
        dormant = n in reg.dormant_servers
        n_tools = sum(1 for t in reg.tool_schemas
                      if reg.server_of(t.get("name", "")) == n)
        cost = reg.schema_cost(n)
        state = f"[{DIM}]dormant[/]" if dormant else f"[{LIME}]active[/]"
        cost_s = f"[{DIM}]{cost:,}[/]" if dormant else f"[{AMBER}]{cost:,}[/]"
        console.print(f"  [{TEAL}]{n:<18}[/] {state:<20} {n_tools:>6} {cost_s:>26}")

    live, off = _summary(reg)
    console.print()
    console.print(f"  [{GRAY}]advertised now:[/] [{AMBER}]~{live:,} tokens[/] "
                  f"[{GRAY}]per request[/]")
    if off:
        console.print(f"  [{GRAY}]dormant:[/] [{DIM}]~{off:,} tokens saved per request[/] "
                      f"[{GRAY}], still callable via ToolSearch[/]")
    console.print(f"\n  [{DIM}]/mcp-off <server>   /mcp-on <server>   "
                  f"/mcp-only <server ...>   /mcp-all[/]")
    console.print(f"  [{DIM}]Dormant means the schemas leave the prompt. Nothing "
                  f"disconnects and nothing becomes uncallable.[/]")


def _names_from(args: str) -> list[str]:
    return [a for a in args.replace(",", " ").split() if a]


def _mutate(args: str, session, settings: dict, mode: str) -> None:
    reg = _registry(session)
    if reg is None:
        emit_info("No MCP servers connected in this session.")
        return
    known = set(reg.server_names())
    if mode == "all":
        reg.set_dormant(set())
        _persist(settings, reg)
        emit_ok("All MCP servers advertising again.")
        _show(session)
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
    _show(session)


def _cmd_servers(args: str, session, settings: dict) -> bool:
    _show(session)
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

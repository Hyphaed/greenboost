"""Forge AI studio slash commands.

Wraps the ai-forge CLI (forge.sh) and exposes MCP server management.
Auto-discovers forge root from cwd ancestors or ~/Dev/ai-forge.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from greenboost_cli.terminal.commands import register_command
from greenboost_cli.terminal.theme import (
    console, GRAY, VIOLET, LIME, AMBER,
    emit_ok, emit_warn, emit_err, emit_info,
)


def _find_forge_root() -> Path | None:
    """Find ai-forge root: walk cwd ancestors then fall back to ~/Dev/ai-forge."""
    p = Path.cwd()
    home = Path.home()
    for _ in range(8):
        if (p / "forge.sh").exists():
            return p
        if p == home or p.parent == p:
            break
        p = p.parent
    default = Path.home() / "Dev" / "ai-forge"
    return default if (default / "forge.sh").exists() else None


def _forge_sh() -> str | None:
    root = _find_forge_root()
    return str(root / "forge.sh") if root else None


def _run_forge(args: list[str], cwd: Path | None = None) -> None:
    """Run forge.sh with given args, streaming output to the console."""
    sh = _forge_sh()
    if not sh:
        emit_err("forge.sh not found. Is ~/Dev/ai-forge present?")
        return
    cmd = ["bash", sh] + args
    cwd_str = str(cwd or _find_forge_root() or Path.cwd())
    console.print(f"\n  [{GRAY}]$ bash forge.sh {' '.join(args)}[/]\n")
    proc = subprocess.Popen(cmd, cwd=cwd_str, text=True)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()


def _forge(args: str, session, settings: dict) -> None:
    parts = args.strip().split(None, 1)
    sub = parts[0].lower() if parts else ""
    rest = parts[1] if len(parts) > 1 else ""

    dispatch = {
        "gen":       _forge_gen,
        "doctor":    _forge_doctor,
        "game":      _forge_game,
        "list":      _forge_list,
        "fetch":     _forge_fetch,
        "quantize":  _forge_quantize,
        "status":    _forge_status,
        "connect":   _forge_connect,
    }

    if sub in dispatch:
        dispatch[sub](rest, session, settings)
    elif sub in ("help", ""):
        _forge_help()
    else:
        emit_warn(f"Unknown forge subcommand: {sub}")
        _forge_help()


def _forge_help() -> None:
    console.print(f"\n  [{VIOLET}]◈[/]  [{VIOLET}]forge[/]  [{GRAY}]ai-forge studio[/]\n")
    cmds = [
        ("gen <manifest.yaml>",     "Generate assets from a manifest"),
        ("game \"<prompt>\"",       "Autonomous game creation from a prompt"),
        ("doctor",                  "Environment / models / GPU check"),
        ("list <manifest.yaml>",    "Show manifest status (done vs missing)"),
        ("fetch [--capability X]",  "Download models (video,music,sfx,tts,rig)"),
        ("quantize <model>",        "GreenBoost mixed-precision GGUF quantize"),
        ("status",                  "Show MCP server connections and tool list"),
        ("connect",                 "Connect / reconnect .mcp.json MCP servers"),
    ]
    for cmd, desc in cmds:
        console.print(f"  [{LIME}]/forge {cmd:<32}[/] [{GRAY}]{desc}[/]")
    console.print()


def _forge_gen(args: str, session, settings: dict) -> None:
    if not args.strip():
        emit_warn("Usage: /forge gen <manifest.yaml> [--force] [--category X] [--only slug…]")
        return
    _run_forge(["gen", "--project"] + args.strip().split())


def _forge_doctor(args: str, session, settings: dict) -> None:
    extra = ["--licenses"] if "--licenses" in args else []
    _run_forge(["doctor"] + extra)


def _forge_game(args: str, session, settings: dict) -> None:
    prompt = args.strip()
    if not prompt:
        emit_warn("Usage: /forge game \"<goal prompt>\" [--backend ollama|claude|mix] [--max-steps N]")
        return
    # Split off extra flags from the prompt (anything after first --)
    if " --" in prompt:
        idx = prompt.index(" --")
        goal = prompt[:idx].strip().strip('"\'')
        extra = prompt[idx:].split()
    else:
        goal = prompt.strip('"\'')
        extra = []
    _run_forge(["game", goal] + extra)


def _forge_list(args: str, session, settings: dict) -> None:
    if not args.strip():
        emit_warn("Usage: /forge list <manifest.yaml> [--category X]")
        return
    _run_forge(["list", "--project"] + args.strip().split())


def _forge_fetch(args: str, session, settings: dict) -> None:
    extra = args.strip().split() if args.strip() else ["--check"]
    _run_forge(["fetch"] + extra)


def _forge_quantize(args: str, session, settings: dict) -> None:
    if not args.strip():
        emit_warn("Usage: /forge quantize <model> [--quality near_lossless|balanced|aggressive]")
        return
    _run_forge(["quantize"] + args.strip().split())


def _forge_status(args: str, session, settings: dict) -> None:
    """Show MCP registry status: connected servers, tool counts."""
    from greenboost_cli.mcp.client import discover_mcp_json

    console.print(f"\n  [{VIOLET}]◈[/]  [{VIOLET}]forge status[/]\n")

    mcp_path = discover_mcp_json()
    if not mcp_path:
        console.print(f"  [{GRAY}]No .mcp.json found in current directory tree.[/]")
    else:
        console.print(f"  [{GRAY}].mcp.json: {mcp_path}[/]")

    registry = getattr(session, "mcp_registry", None)
    if registry is None:
        console.print(f"  [{AMBER}]MCP registry not attached to session.[/]")
        console.print(f"  [{GRAY}]Run /forge connect to connect servers.[/]")
        return

    status = registry.status()
    if not status:
        console.print(f"  [{GRAY}]No MCP servers configured.[/]")
        return

    for s in status:
        icon = f"[{LIME}]✓[/]" if s["connected"] else f"[{AMBER}]○[/]"
        console.print(
            f"  {icon} [{VIOLET}]{s['server']}[/]"
            f"  [{GRAY}]{s['tools']} tools[/]"
        )
        if s["connected"] and s["names"]:
            names = ", ".join(s["names"][:8])
            if len(s["names"]) > 8:
                names += f" … +{len(s['names']) - 8}"
            console.print(f"      [{GRAY}]{names}[/]")
    console.print()

    # Forge root
    root = _find_forge_root()
    if root:
        console.print(f"  [{GRAY}]forge root: {root}[/]")
    console.print()


def _forge_connect(args: str, session, settings: dict) -> None:
    """Connect / reconnect MCP servers from .mcp.json."""
    import threading
    from greenboost_cli.mcp.client import discover_mcp_json, MCPRegistry

    mcp_path = discover_mcp_json()
    if not mcp_path:
        emit_err("No .mcp.json found in the current directory tree.")
        return

    # Disconnect existing registry
    old = getattr(session, "mcp_registry", None)
    if old:
        old.close_all()

    registry = MCPRegistry.from_mcp_json(mcp_path)
    session.mcp_registry = registry

    servers = registry.server_names()
    if not servers:
        emit_warn("No servers defined in .mcp.json")
        return

    console.print(f"\n  [{VIOLET}]◈[/]  Connecting {len(servers)} MCP server(s)…\n")

    def _connect_bg():
        results = registry.connect_all()
        ok = sum(1 for v in results.values() if v)
        fail = len(results) - ok
        for name, connected in results.items():
            icon = f"[{LIME}]✓[/]" if connected else f"[{AMBER}]✗[/]"
            n_tools = len(next(
                (c.tools for c in registry._clients.values()
                 if c.name == name and c.connected), []
            ))
            console.print(
                f"  {icon} [{VIOLET}]{name}[/]"
                + (f"  [{GRAY}]{n_tools} tools[/]" if connected else "  [not reachable]")
            )
        console.print()
        if ok:
            emit_ok(f"{ok} server(s) connected · {len(registry.tool_schemas)} tools available")
        if fail:
            emit_warn(f"{fail} server(s) failed to connect")
        for collision in registry.collisions:
            emit_warn(collision)
        for dup in getattr(registry, "duplicate_names", []):
            emit_info(dup)

    # Run synchronously since user explicitly requested connection
    _connect_bg()


register_command("forge", _forge, "AI forge studio  (/forge gen|game|doctor|list|status|connect)")

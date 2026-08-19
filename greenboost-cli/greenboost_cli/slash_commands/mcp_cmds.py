"""MCP server management: /mcp [start|stop|status|config|sync-accounts]."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from greenboost_cli.terminal.commands import register_command
from greenboost_cli.environment.settings import GB_HOME

_MCP_PIDFILE = GB_HOME / "mcp_server.pid"
_MCP_LOGFILE = GB_HOME / "mcp_server.log"

_DEFAULT_HTTP_PORT = 7822


def _read_pid() -> int | None:
    try:
        return int(_MCP_PIDFILE.read_text().strip())
    except Exception:
        return None


def _proc_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _mcp(args: str, session, settings: dict) -> None:
    parts = args.strip().split()
    sub   = parts[0].lower() if parts else "help"

    if sub in ("help", ""):
        _print_help()
    elif sub == "config":
        _print_config()
    elif sub == "status":
        _check_status()
    elif sub == "start":
        _start_server(args)
    elif sub == "stop":
        _stop_server()
    elif sub == "logs":
        _show_logs(parts[1:])
    elif sub == "sync-accounts":
        _sync_accounts_mcp()
    else:
        print(f"  Unknown subcommand: {sub}")
        _print_help()


def _print_help() -> None:
    print()
    print("  /mcp config                       Print Claude Code config snippet")
    print("  /mcp status                       Check MCP server readiness")
    print("  /mcp start [--http] [--port N]    Start MCP (thread or daemon)")
    print("  /mcp stop                         Stop MCP daemon (if running)")
    print("  /mcp logs [--tail N]              Show MCP daemon logs")
    print("  /mcp sync-accounts                Register MCP in all account configs")
    print()
    print("  Stdio mode (Claude Code / Claude Desktop):")
    print('    command: "python", args: ["-m", "greenboost_cli.mcp.server"]')
    print()


def _print_config() -> None:
    config = {
        "mcpServers": {
            "greenboost": {
                "command": sys.executable,
                "args":    ["-m", "greenboost_cli.mcp.server"],
            }
        }
    }
    print()
    print("  ── Claude Code config ─────────────────────────────────────────────")
    print("  Add to ~/.claude/settings.json (global) or .claude/settings.json (project):")
    print()
    for line in json.dumps(config, indent=2).splitlines():
        print(f"  {line}")
    print()
    print("  After saving, reload Claude Code (or run: claude mcp list) to verify.")
    print()
    print("  Available MCP tools:")
    print("    convert_to_markdown  — convert any file/URL to Markdown + index RAG")
    print("    rag_search           — semantic + BM25 hybrid search")
    print("    rag_index_folder     — index a folder into RAG")
    print("    rag_index_text       — index arbitrary text into RAG")
    print("    rag_status           — index statistics")
    print("    get_goals / add_goal — project goal management")
    print("    get_history          — recent project history")
    print("    system_status        — GreenBoost T1/T2/T3 tier stats")
    print()
    print("  MCP resources:")
    print("    rag://index/status   — RAG index status")
    print("    goals://{project}    — project goals")
    print("    history://{project}  — project history")
    print()


def _check_status() -> None:
    print()

    # Daemon process
    pid = _read_pid()
    if pid:
        if _proc_alive(pid):
            print(f"  ✓  MCP daemon running (pid {pid})  ·  log: {_MCP_LOGFILE}")
        else:
            print(f"  ○  MCP daemon PID file found (pid {pid}) but process is gone")
            _MCP_PIDFILE.unlink(missing_ok=True)
    else:
        print("  ○  MCP daemon not running  ·  /mcp start --http to start")

    # Module health
    try:
        from greenboost_cli.mcp.server import mcp  # noqa: F401
        print(f"  ✓  MCP server module importable")
    except ImportError as e:
        print(f"  ✗  MCP not available: {e}")
        print("     pip install greenboost-cli[mcp]")

    try:
        from greenboost_cli.converters.markitdown_adapter import SUPPORTED_EXTENSIONS
        print(f"  ✓  markitdown: {len(SUPPORTED_EXTENSIONS)} supported formats")
    except ImportError:
        print("  ⚠  markitdown not installed — convert_to_markdown tool degraded")
        print("     pip install greenboost-cli[convert]")

    try:
        from greenboost_cli.rag.engine import store_stats, _load_folders
        folders  = _load_folders()
        n_chunks = store_stats()["chunks"]
        print(f"  ✓  RAG: {n_chunks:,} chunks · {len(folders)} source(s)")
    except Exception as e:
        print(f"  ⚠  RAG unavailable: {e}")

    try:
        from greenboost_cli.memory.brain import project_dir  # noqa: F401
        print("  ✓  Brain / goals module available")
    except Exception:
        print("  ⚠  Brain module unavailable")

    print()


def _start_server(args: str) -> None:
    use_http   = "--http" in args
    use_daemon = "--daemon" in args or use_http   # http implies daemon

    port = _DEFAULT_HTTP_PORT
    parts = args.split()
    for i, p in enumerate(parts):
        if p == "--port" and i + 1 < len(parts):
            try:
                port = int(parts[i + 1])
            except ValueError:
                pass

    if not use_http:
        print("  Stdio mode is used directly by Claude Code — no manual start needed.")
        print("  Run /mcp config to get the settings.json snippet.")
        print()
        print("  To start HTTP mode: /mcp start --http [--port 7822]")
        print("  Add --daemon to detach as a persistent background process.")
        return

    try:
        from greenboost_cli.mcp.server import mcp  # noqa: F401
    except ImportError as e:
        print(f"  ✗  MCP not available: {e}")
        print("     pip install greenboost-cli[mcp]")
        return

    if use_daemon:
        _start_daemon(port)
    else:
        _start_thread(port)


def _start_thread(port: int) -> None:
    """Start MCP HTTP server in a background thread (dies with the process)."""
    from greenboost_cli.mcp.server import mcp

    def _run() -> None:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=port)

    t = threading.Thread(target=_run, daemon=True, name="gb-mcp-http")
    t.start()
    print(f"  ✓  MCP HTTP server started (thread): http://127.0.0.1:{port}/mcp")
    print(f"     Claude Code config: transport=http, url=http://127.0.0.1:{port}/mcp")
    print()


def _start_daemon(port: int) -> None:
    """Detach an MCP HTTP server as an independent daemon process."""
    # Kill any existing daemon
    _stop_server(quiet=True)

    GB_HOME.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "greenboost_cli.mcp.server",
        "--http", f"--port={port}",
    ]
    try:
        log_fh = open(_MCP_LOGFILE, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _MCP_PIDFILE.write_text(str(proc.pid))
        print(f"  ✓  MCP daemon started (pid {proc.pid}): http://127.0.0.1:{port}/mcp")
        print(f"     Logs: {_MCP_LOGFILE}")
        print(f"     Stop: /mcp stop")
        print()
    except Exception as e:
        print(f"  ✗  Could not start MCP daemon: {e}")


def _stop_server(quiet: bool = False) -> None:
    """Stop the MCP daemon by PID with graceful SIGTERM + SIGKILL fallback."""
    pid = _read_pid()
    if not pid:
        if not quiet:
            print("  MCP daemon is not running (no PID file).")
        return
    if not _proc_alive(pid):
        _MCP_PIDFILE.unlink(missing_ok=True)
        if not quiet:
            print(f"  MCP daemon was already gone (pid {pid}).")
        return
    try:
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and _proc_alive(pid):
            time.sleep(0.1)
        if _proc_alive(pid):
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.2)
        _MCP_PIDFILE.unlink(missing_ok=True)
        if not quiet:
            print(f"  ✓  MCP daemon stopped (pid {pid}).")
    except Exception as e:
        if not quiet:
            print(f"  ✗  Could not stop MCP daemon (pid {pid}): {e}")


def _show_logs(extra_args: list[str]) -> None:
    tail = 40
    for i, a in enumerate(extra_args):
        if a == "--tail" and i + 1 < len(extra_args):
            try:
                tail = int(extra_args[i + 1])
            except ValueError:
                pass
    if not _MCP_LOGFILE.exists():
        print("  No MCP log file found. Start with: /mcp start --http --daemon")
        return
    lines = _MCP_LOGFILE.read_text(errors="replace").splitlines()
    shown = lines[-tail:]
    print(f"\n  ── MCP log (last {len(shown)} lines) ─────────────────────────")
    for line in shown:
        print(f"  {line}")
    print()


def _sync_accounts_mcp() -> None:
    """Register the greenboost MCP server entry in all known account config dirs.

    Reads ~/.claude-accounts/*/  (optimal-claude per-account dirs) and
    injects the greenboost stdio server into each account's .claude.json.
    Skips dirs that already have the entry.  Idempotent.
    """
    base = Path.home() / ".claude-accounts"
    if not base.exists():
        print("  No ~/.claude-accounts/ directory found.")
        print("  This command targets optimal-claude per-account config dirs.")
        return

    entry = {
        "command": sys.executable,
        "args":    ["-m", "greenboost_cli.mcp.server"],
        "type":    "stdio",
    }

    synced = 0
    skipped = 0
    for account_dir in sorted(base.iterdir()):
        if not account_dir.is_dir():
            continue
        config_file = account_dir / ".claude.json"
        try:
            cfg = json.loads(config_file.read_text()) if config_file.exists() else {}
            cfg.setdefault("mcpServers", {})
            if "greenboost" in cfg["mcpServers"]:
                skipped += 1
                continue
            cfg["mcpServers"]["greenboost"] = entry
            config_file.write_text(json.dumps(cfg, indent=2))
            synced += 1
            print(f"  ✓  {account_dir.name}: registered greenboost MCP")
        except Exception as e:
            print(f"  ✗  {account_dir.name}: {e}")

    if synced == 0 and skipped == 0:
        print("  No account config dirs found in ~/.claude-accounts/")
    else:
        print()
        if synced:
            print(f"  Registered in {synced} account(s).")
        if skipped:
            print(f"  Already registered in {skipped} account(s) — skipped.")
    print()


register_command("mcp", _mcp, "MCP server  (/mcp config|status|start|stop|logs|sync-accounts)")

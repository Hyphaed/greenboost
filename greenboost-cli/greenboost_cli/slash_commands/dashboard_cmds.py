"""Dashboard slash command: /dashboard."""
from __future__ import annotations

import threading
import webbrowser

from greenboost_cli.terminal.commands import register_command

_server_thread: threading.Thread | None = None
_server_port: int = 0


def _dashboard(args: str, session, settings: dict) -> None:
    global _server_thread, _server_port

    port = settings.get("dashboard_port", 7821)
    if args.strip():
        try:
            port = int(args.strip())
        except ValueError:
            pass

    # Start the server in a background daemon thread (once)
    if _server_thread is None or not _server_thread.is_alive():
        try:
            from greenboost_cli.dashboard.server import run as run_server
        except ImportError as e:
            print(f"  ✗  Dashboard unavailable: {e}")
            return

        _server_port = port

        def _target():
            try:
                run_server(port)
            except OSError as exc:
                print(f"\n  ✗  Dashboard failed to start: {exc}")

        _server_thread = threading.Thread(target=_target, daemon=True, name="gb-dashboard")
        _server_thread.start()

        import time
        time.sleep(0.4)  # brief wait for server to bind

    url = f"http://localhost:{_server_port}"
    print(f"  ✓  Dashboard running at: {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass


register_command("dashboard", _dashboard, "Open web dashboard  (/dashboard [port])")

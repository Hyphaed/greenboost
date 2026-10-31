"""/exit and Ctrl-D: clean terminal first, MCP servers closed, model unloaded."""
from __future__ import annotations

import os
import pty
import subprocess
import sys
import textwrap
import threading
import time
import types

import pytest

from greenboost_cli.terminal import repl as R


def test_exit_order_is_terminal_then_servers_then_model_then_pool(monkeypatch):
    calls = []
    monkeypatch.setattr(R, "request_shutdown", lambda: calls.append("terminal"))
    monkeypatch.setattr(R, "unload_model_on_exit", lambda s: calls.append("unload") or "Unloaded m; its VRAM is free.")
    monkeypatch.setattr(R, "release_memory_pool", lambda reason: calls.append("pool"))
    import greenboost_cli.terminal.theme as T
    monkeypatch.setattr(T, "emit_ok", lambda m: calls.append("ok:" + m.split(";")[0].split()[0]))
    reg = types.SimpleNamespace(close_all=lambda: calls.append("mcp"))
    with pytest.raises(SystemExit):
        R.graceful_exit(types.SimpleNamespace(mcp_registry=reg), {}, "/exit")
    assert calls == ["terminal", "ok:Goodbye.", "mcp", "unload", "ok:Unloaded", "pool"]


def test_exit_keep_leaves_the_model_loaded(monkeypatch):
    calls = []
    monkeypatch.setattr(R, "request_shutdown", lambda: None)
    monkeypatch.setattr(R, "unload_model_on_exit", lambda s: calls.append("unload") or "")
    monkeypatch.setattr(R, "release_memory_pool", lambda reason: None)
    with pytest.raises(SystemExit):
        R.graceful_exit(types.SimpleNamespace(mcp_registry=None), {}, "/exit", keep_model=True)
    assert calls == []


class _FakeSynapse:
    def __init__(self, running): self.running, self.stopped = running, []
    def ps(self): return [{"model": m} for m in self.running]
    def stop(self, m): self.stopped.append(m); return True


def _patch_synapse(monkeypatch, fake):
    import greenboost_cli.slash_commands.backend_cmds as B
    monkeypatch.setattr(B, "_import_gb_synapse", lambda: fake)


def test_unload_stops_the_configured_model_only(monkeypatch):
    monkeypatch.delenv("GB_CLI_KEEP_MODEL", raising=False)
    fake = _FakeSynapse(["qwen38-35b-a3b-apex-miniplus"])
    _patch_synapse(monkeypatch, fake)
    msg = R.unload_model_on_exit({"model": "gb-synapse/qwen38-35b-a3b-apex-miniplus"})
    assert fake.stopped == ["qwen38-35b-a3b-apex-miniplus"] and msg.startswith("Unloaded")
    other = _FakeSynapse(["some-other-model"])
    _patch_synapse(monkeypatch, other)
    assert R.unload_model_on_exit({"model": "gb-synapse/qwen38-35b-a3b-apex-miniplus"}) == "" and not other.stopped


def test_unload_opt_outs(monkeypatch):
    fake = _FakeSynapse(["m"])
    _patch_synapse(monkeypatch, fake)
    monkeypatch.setenv("GB_CLI_KEEP_MODEL", "1")
    assert R.unload_model_on_exit({"model": "gb-synapse/m"}) == ""
    monkeypatch.delenv("GB_CLI_KEEP_MODEL")
    assert R.unload_model_on_exit({"model": "gb-synapse/m", "unload_model_on_exit": False}) == ""
    assert not fake.stopped
    assert R.unload_model_on_exit({"model": "gb-synapse/m"}).startswith("Unloaded") and fake.stopped == ["m"]


def test_no_static_dock_is_painted_while_shutting_down(monkeypatch):
    painted = []
    import greenboost_cli.terminal.dock as D
    import greenboost_cli.terminal.statusline as S
    import greenboost_cli.terminal.footer as F
    monkeypatch.setattr(D, "get_hold", lambda: types.SimpleNamespace(hold=lambda s: painted.append(s),
                                                                    force_reset=lambda: None, release=lambda: None))
    monkeypatch.setattr(S, "set_wizard_active", lambda v: None)
    monkeypatch.setattr(F, "get_footer", lambda: types.SimpleNamespace(deactivate=lambda: None))
    monkeypatch.setattr(R, "_dock_state_provider", lambda: {"x": 1})
    monkeypatch.setattr(R, "_pt_suspend_hook", lambda: None)
    monkeypatch.setattr(R, "_wizard_depth", 0)
    R._shutdown_evt.clear()
    R._wizard_free.set()
    R._shutdown_evt.set()
    try:
        R._suspend_pt_for_wizard()
    finally:
        R._shutdown_evt.clear()
        R._resume_pt_after_wizard()
    assert painted == []


def test_mcp_close_kills_the_servers_own_children(tmp_path):
    from greenboost_cli.mcp.client import MCPStdioClient
    srv = tmp_path / "wrapper.py"
    srv.write_text(textwrap.dedent('''
        import json, subprocess, sys, time
        child = subprocess.Popen(["sleep", "300"])
        open(sys.argv[1], "w").write(str(child.pid))
        for line in sys.stdin:
            m = json.loads(line)
            if "id" not in m: continue
            r = {"protocolVersion": "2024-11-05", "capabilities": {}} if m["method"] == "initialize" else {"tools": []}
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r}) + "\\n"); sys.stdout.flush()
        time.sleep(300)             # ignores stdin EOF, like a stubborn npx wrapper
    '''))
    pidfile = tmp_path / "child.pid"
    c = MCPStdioClient("w", [sys.executable, str(srv), str(pidfile)])
    assert c.connect()
    child = int(pidfile.read_text())
    t0 = time.time()
    c.close()
    assert time.time() - t0 < 5
    time.sleep(0.3)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)               # the grandchild is gone too, not an orphan


def test_drain_waits_for_a_late_reply_instead_of_leaking_it():
    """A terminal that answers 1 s late: the barrier (ESC[5n -> ESC[0n) must outlast it."""
    master, slave = pty.openpty()
    code = ("import sys, os, time\n"
            "from greenboost_cli.terminal.repl import _drain_terminal_replies\n"
            "t = time.time(); _drain_terminal_replies(sys.stdin.fileno(), timeout=3)\n"
            "print('waited', round(time.time() - t, 1), flush=True)\n")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = subprocess.Popen([sys.executable, "-c", code], stdin=slave, stdout=slave, stderr=subprocess.DEVNULL,
                         cwd=here, env={**os.environ, "PYTHONPATH": here})

    def terminal():
        buf = b""
        while p.poll() is None:
            try:
                buf += os.read(master, 256)
            except OSError:
                return
            if b"\x1b[5n" in buf:
                time.sleep(1.0)
                os.write(master, b"\x1b[1;1R\x1b[0n")     # the late cursor reply first, then the barrier's
                return
    th = threading.Thread(target=terminal, daemon=True)
    th.start()
    p.wait(timeout=15)
    out = os.read(master, 65536).decode(errors="replace")
    os.close(master); os.close(slave)
    waited = float(out.split("waited")[1].split()[0])
    assert 0.9 <= waited < 2.5, out

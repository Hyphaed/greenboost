"""MCP discovery reads Claude Code's config, and one bad server can't hurt the rest."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from greenboost_cli.mcp import client as MC
from greenboost_cli.mcp.config import discover_mcp_servers


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: h))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return h


def _w(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def test_claude_json_account_and_mcp_json_merge_in_priority_order(home, tmp_path, monkeypatch):
    proj = home / "proj"
    proj.mkdir()
    acct = tmp_path / "acct"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(acct))
    _w(home / ".claude.json", {
        "mcpServers": {"semble": {"command": "uvx", "args": ["semble"]},
                       "shared": {"command": "root"}},
        "projects": {str(proj): {"mcpServers": {"projonly": {"command": "p"}}}},
    })
    _w(acct / ".claude.json", {"mcpServers": {"shared": {"command": "acct"},
                                               "charlotte": {"command": "charlotte"}}})
    _w(proj / ".mcp.json", {"mcpServers": {"shared": {"command": "mcpjson"}}})

    specs = discover_mcp_servers(proj)

    assert set(specs) == {"semble", "shared", "charlotte", "projonly"}
    assert specs["shared"]["command"] == ["mcpjson"]          # .mcp.json wins
    assert specs["semble"]["command"] == ["uvx", "semble"]
    assert specs["shared"]["cwd"] == str(proj)                 # relative scripts resolve there


def test_self_disabled_and_bad_entries(home):
    _w(home / ".claude.json", {"mcpServers": {
        "greenboost-cli": {"command": "x"},                    # ourselves
        "off": {"command": "x", "disabled": True},
        "nocmd": {"args": ["a"]},                              # bad entry
        "good": {"command": "echo", "env": {"K": "${HOME}/v"}},
    }})
    specs = discover_mcp_servers(home)
    assert "greenboost-cli" not in specs and "off" not in specs
    assert specs["nocmd"]["error"]
    assert "error" not in specs["good"]
    assert specs["good"]["env"]["K"].endswith("/v")


def test_greenboost_mirrors_deduplicated_when_two_servers(home):
    _w(home / ".claude.json", {"mcpServers": {
        "greenboost-dataflux": {"command": "a"}, "greenboost-cluster": {"command": "b"},
        "semble": {"command": "c"}}})
    specs = discover_mcp_servers(home)
    assert specs["greenboost-dataflux"]["env"]["GB_MCP_NO_MIRROR"] == "1"
    assert "GB_MCP_NO_MIRROR" not in specs["semble"]["env"]


def test_http_entry_without_command_is_http(home):
    _w(home / ".claude.json", {"mcpServers": {"remote": {"url": "http://127.0.0.1:1/mcp"}}})
    assert discover_mcp_servers(home)["remote"]["type"] == "http"


# ── client robustness, against a real subprocess ────────────────────────────

_SERVER = textwrap.dedent('''
    import json, sys, time
    mode = sys.argv[1]
    for line in sys.stdin:
        m = json.loads(line)
        if "id" not in m:
            continue
        if mode == "hang":
            time.sleep(60)
        if m["method"] == "initialize":
            r = {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "t"}}
        elif m["method"] == "tools/list":
            n = 3000 if mode == "big" else 2
            r = {"tools": [{"name": f"t{i}", "description": "x" * 120,
                            "inputSchema": {"type": "object"}} for i in range(n)]}
        else:
            r = {"content": [{"type": "text", "text": "pong"}]}
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r}) + "\\n")
        sys.stdout.flush()
''')


@pytest.fixture
def server_py(tmp_path):
    p = tmp_path / "srv.py"
    p.write_text(_SERVER)
    return str(p)


def test_large_tools_list_is_not_truncated(server_py):
    c = MC.MCPStdioClient("big", [sys.executable, server_py, "big"])
    try:
        assert c.connect() is True
        assert len(c.tools) == 3000          # was 0 (silently) past ~200 KB
        assert c.call_tool("t1", {}) == "pong"
    finally:
        c.close()


def test_hung_server_costs_one_timeout_not_the_session(server_py, monkeypatch):
    monkeypatch.setattr(MC, "INIT_TIMEOUT_S", 1.0)
    reg = MC.MCPRegistry.from_config({
        "hang": {"name": "hang", "type": "stdio", "command": [sys.executable, server_py, "hang"], "env": {}},
        "ok": {"name": "ok", "type": "stdio", "command": [sys.executable, server_py, "ok"], "env": {}},
        "bad": {"name": "bad", "type": "stdio", "error": "entry has no 'command'", "env": {}},
        "missing": {"name": "missing", "type": "stdio", "command": ["/no/such/bin"], "env": {}},
    })
    try:
        res = reg.connect_all()
    finally:
        reg.close_all()
    assert res == {"hang": False, "ok": True, "bad": False, "missing": False}
    assert reg.has_tool("mcp__ok__t0")
    st = {s["server"]: s for s in reg.status()}
    assert "no initialize reply" in st["hang"]["error"]
    assert "not found" in st["missing"]["error"]
    assert st["bad"]["error"] == "entry has no 'command'"


def test_reconnect_does_not_duplicate_tools(server_py):
    reg = MC.MCPRegistry.from_config({
        "ok": {"name": "ok", "type": "stdio", "command": [sys.executable, server_py, "ok"], "env": {}}})
    try:
        reg.connect_all()
        n = len(reg.tool_schemas)
        assert reg.reconnect("ok") is True
        assert len(reg.tool_schemas) == n
    finally:
        reg.close_all()


def test_server_requests_are_answered_not_ignored(tmp_path):
    """A server that asks roots/list during initialize and waits for the reply."""
    srv = tmp_path / "asker.py"
    srv.write_text(textwrap.dedent('''
        import json, sys
        def send(o): sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
        for line in sys.stdin:
            m = json.loads(line)
            if m.get("method") == "initialize":
                send({"jsonrpc": "2.0", "id": 900, "method": "roots/list"})
                reply = json.loads(sys.stdin.readline())        # blocks until we answer
                ok = reply.get("id") == 900 and reply["result"]["roots"]
                send({"jsonrpc": "2.0", "id": m["id"], "result": {"ok": bool(ok)}})
            elif m.get("method") == "tools/list":
                send({"jsonrpc": "2.0", "id": m["id"], "result": {"tools": []}})
    '''))
    c = MC.MCPStdioClient("asker", [sys.executable, str(srv)])
    try:
        assert c.connect() is True      # would time out (30 s) if roots/list were ignored
    finally:
        c.close()


def test_only_and_exclude_filters_decide_which_servers_are_started(home, monkeypatch):
    _w(home / ".claude.json", {"mcpServers": {n: {"command": "x"} for n in
        ("semble", "knowledge-rag", "forge-media", "svg-animation-studio", "charlotte")}})
    assert set(discover_mcp_servers(home)) == {"semble", "knowledge-rag", "forge-media", "svg-animation-studio", "charlotte"}
    monkeypatch.setenv("GB_MCP_EXCLUDE", "forge-*, svg-*")
    assert set(discover_mcp_servers(home)) == {"semble", "knowledge-rag", "charlotte"}
    monkeypatch.setenv("GB_MCP_ONLY", "semble,charlotte")
    assert set(discover_mcp_servers(home)) == {"semble", "charlotte"}

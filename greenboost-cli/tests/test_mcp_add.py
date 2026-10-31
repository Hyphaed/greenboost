"""/mcp-add, /mcp-import, /mcp-remove: adding a server without editing JSON."""
from __future__ import annotations

import json
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest

from greenboost_cli.mcp import client as MC
from greenboost_cli.mcp import config as C
from greenboost_cli.slash_commands import mcp_add_cmds as A

SERVER = textwrap.dedent('''
    import json, sys
    for line in sys.stdin:
        m = json.loads(line)
        if "id" not in m: continue
        if m["method"] == "initialize": r = {"protocolVersion": "2024-11-05", "capabilities": {}}
        elif m["method"] == "tools/list": r = {"tools": [{"name": "ping", "description": "p", "inputSchema": {"type": "object"}}]}
        else: r = {"content": [{"type": "text", "text": "pong"}]}
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r}) + "\\n"); sys.stdout.flush()
''')


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".claude-accounts" / "acct2").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    user = tmp_path / "gb" / "mcp.json"
    monkeypatch.setattr(C, "user_mcp_path", lambda: user)
    monkeypatch.chdir(tmp_path)
    srv = tmp_path / "srv.py"
    srv.write_text(SERVER)
    out = []
    for n in ("emit_ok", "emit_err", "emit_info", "emit_warn"):
        monkeypatch.setattr(A, n, lambda m, _n=n: out.append((_n, m)))
    return types.SimpleNamespace(home=home, user=user, srv=srv, out=out, session=types.SimpleNamespace(mcp_registry=None))


def _wait(session, name, secs=20):
    end = time.time() + secs
    while time.time() < end:
        reg = session.mcp_registry
        if reg is not None and name in reg._clients and reg._clients[name].connected:
            return True
        time.sleep(0.1)
    return False


def test_add_a_command_saves_it_privately_and_connects_without_a_restart(env):
    A.cmd_mcp_add(f"mine --env TOKEN=abc {sys.executable} {env.srv}", env.session, {})
    assert json.loads(env.user.read_text())["mcpServers"]["mine"]["env"] == {"TOKEN": "abc"}
    assert env.user.stat().st_mode & 0o777 == 0o600                    # an entry can carry a token
    assert _wait(env.session, "mine")
    assert env.session.mcp_registry.has_tool("mcp__mine__ping")
    env.session.mcp_registry.close_all()


def test_the_user_file_is_one_of_the_discovery_sources(env):
    C.add_server("viafile", {"command": sys.executable, "args": [str(env.srv)]})
    assert "viafile" in C.discover_mcp_servers(env.home)


def test_a_command_that_does_not_exist_is_refused_and_nothing_is_saved(env):
    A.cmd_mcp_add("typo definitely-not-a-binary --x", env.session, {})
    assert not env.user.exists()
    assert any("isn't an installed command" in m for n, m in env.out if n == "emit_err")


def test_a_bare_name_copies_the_server_from_another_claude_account(env):
    (env.home / ".claude-accounts" / "acct2" / ".claude.json").write_text(json.dumps({"mcpServers": {
        "firefox-devtools": {"command": sys.executable, "args": [str(env.srv)], "env": {"TMPDIR": "/tmp"}}}}))
    other = C.other_claude_servers(set())
    assert list(other) == ["firefox-devtools"] and other["firefox-devtools"][1] == "acct2"
    A.cmd_mcp_add("firefox-devtools", env.session, {})
    assert "firefox-devtools" in json.loads(env.user.read_text())["mcpServers"]
    assert _wait(env.session, "firefox-devtools")
    env.session.mcp_registry.close_all()


def test_import_lists_then_imports_and_skips_what_is_already_loaded(env, capsys):
    (env.home / ".claude.json").write_text(json.dumps({"mcpServers": {"a": {"command": sys.executable, "args": [str(env.srv)]}}}))
    (env.home / ".claude-accounts" / "acct2" / ".claude.json").write_text(json.dumps({"mcpServers": {
        "a": {"command": "x"}, "b": {"command": sys.executable, "args": [str(env.srv)]}}}))
    assert set(C.other_claude_servers({"a"})) == {"b"}                  # a is loaded, b isn't
    A.cmd_mcp_import("b", env.session, {})
    assert _wait(env.session, "b")
    env.session.mcp_registry.close_all()


def test_remove_forgets_it_from_the_file_and_the_running_session(env):
    A.cmd_mcp_add(f"mine {sys.executable} {env.srv}", env.session, {})
    assert _wait(env.session, "mine")
    A.cmd_mcp_remove("mine", env.session, {})
    assert "mine" not in json.loads(env.user.read_text())["mcpServers"]
    assert "mine" not in env.session.mcp_registry.server_names()
    assert not env.session.mcp_registry.has_tool("mcp__mine__ping")


def test_a_url_makes_an_http_entry_and_bad_names_are_refused(env):
    A.cmd_mcp_add("remote --env Authorization=Bearer-x https://example.invalid/mcp", env.session, {})
    cfg = json.loads(env.user.read_text())["mcpServers"]["remote"]
    assert cfg == {"type": "http", "url": "https://example.invalid/mcp", "headers": {"Authorization": "Bearer-x"}}
    A.cmd_mcp_add("bad/name cmd", env.session, {})
    assert any("usable server name" in m for n, m in env.out)


def test_options_go_before_the_command_and_everything_after_it_is_the_commands_own(env):
    parsed = A._parse("n --project --env A=1 --env B=2 npx -y pkg --env KEEP=me")
    assert parsed == ("n", True, {"A": "1", "B": "2"}, ["npx", "-y", "pkg", "--env", "KEEP=me"])


def test_a_token_typed_after_env_is_scrubbed_from_the_input_history(env, monkeypatch):
    import greenboost_cli.environment.settings as S
    hist = env.user.parent / "input_history.txt"
    hist.parent.mkdir(parents=True, exist_ok=True)
    secret = "ghp_" + "z" * 30
    hist.write_text(f"\n# t1\n+/help\n\n# t2\n+/mcp-add gh --env GITHUB_TOKEN={secret} npx -y pkg\n\n# t3\n+/doctor\n")
    monkeypatch.setattr(S, "HISTORY_PATH", hist)
    A.cmd_mcp_add(f"gh --env GITHUB_TOKEN={secret} {sys.executable} {env.srv}", env.session, {})
    left = hist.read_text()
    assert secret not in left and "+/help" in left and "+/doctor" in left
    assert json.loads(env.user.read_text())["mcpServers"]["gh"]["env"]["GITHUB_TOKEN"] == secret   # still saved for the server
    env.session.mcp_registry.close_all()

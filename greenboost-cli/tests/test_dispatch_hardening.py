"""Regression tests for the 2026-10-02 dispatcher findings: Delegate wiring and
privilege inheritance, MCP calls going through the policy gate, blocking hooks
failing closed, and Bash being confined to the policy's workspace roots."""
import shutil

import pytest

from greenboost_cli.instruments import dispatcher, hooks
from greenboost_cli.instruments.policy import PERMISSIVE, build_policy


# ── Delegate ──────────────────────────────────────────────────────────────────

def test_delegate_receives_prompt_string_and_label(monkeypatch):
    import greenboost_cli.agents.subagent as sub
    seen = {}
    monkeypatch.setattr(sub, "delegate", lambda prompt, **kw: seen.update(prompt=prompt, **kw) or "ok")
    out = dispatcher.dispatch("Delegate", {"prompt": "find the bug", "label": "hunt"},
                              approval_mode="accept-all")
    assert out == "ok"
    assert seen["prompt"] == "find the bug" and seen["label"] == "hunt"


def test_delegate_inherits_the_callers_policy_as_a_ceiling(monkeypatch):
    import greenboost_cli.agents.subagent as sub
    seen = {}
    monkeypatch.setattr(sub, "delegate", lambda prompt, **kw: seen.update(kw) or "ok")
    restricted = build_policy(["Read", "Delegate"])
    dispatcher.dispatch("Delegate", {"prompt": "x"}, approval_mode="accept-all", policy=restricted)
    assert seen["parent_policy"] is restricted


def test_narrowed_policy_is_never_wider_than_the_parent():
    parent = build_policy(["Read", "Grep"])
    wide_child = build_policy(["Read", "Bash", "Write"])
    eff = parent.narrowed(wide_child)
    assert eff.permits_tool("Read")
    assert not eff.permits_tool("Bash") and not eff.permits_tool("Write")
    assert parent.narrowed(PERMISSIVE).allow == parent.allow
    assert PERMISSIVE.narrowed(wide_child).allow == wide_child.allow


# ── MCP goes through the policy gate and the audit trail ──────────────────────

def test_restricted_policy_denies_mcp_tools_by_default():
    calls = []
    pol = build_policy(["Read"])
    out = dispatcher.dispatch_mcp(lambda n, p: calls.append(n) or "ran",
                                  "mcp__srv__do", {}, policy=pol)
    assert out.startswith("Error:") and "not in this agent's tool policy" in out
    assert calls == []


def test_mcp_tool_allowed_by_name_or_server_pattern():
    for allow in (["Read", "mcp__srv__do"], ["Read", "mcp__srv__*"]):
        pol = build_policy(allow)
        assert dispatcher.dispatch_mcp(lambda n, p: "ran", "mcp__srv__do", {}, policy=pol) == "ran"
    assert dispatcher.dispatch_mcp(lambda n, p: "ran", "mcp__other__do", {},
                                   policy=build_policy(["mcp__srv__*"])).startswith("Error:")


def test_mcp_deny_wins_and_permissive_default_is_unchanged():
    pol = build_policy(None, deny=["mcp__srv__*"])
    assert dispatcher.dispatch_mcp(lambda n, p: "ran", "mcp__srv__do", {}, policy=pol).startswith("Error:")
    assert dispatcher.dispatch_mcp(lambda n, p: "ran", "mcp__srv__do", {}) == "ran"


def test_mcp_call_emits_an_audit_event(monkeypatch):
    events = []
    monkeypatch.setattr(dispatcher, "_emit_cli_tool_call",
                        lambda name, decision, agent, *a, **k: events.append((name, decision, agent)))
    dispatcher.dispatch_mcp(lambda n, p: "ok", "mcp__srv__do", {}, policy=build_policy(["Read"]), agent="w1")
    dispatcher.dispatch_mcp(lambda n, p: "ok", "mcp__srv__do", {}, agent="w2")
    assert events == [("mcp__srv__do", "denied_policy", "w1"), ("mcp__srv__do", "allowed", "w2")]


def test_mcp_registry_exception_becomes_a_tool_error():
    def boom(n, p):
        raise RuntimeError("server died")
    out = dispatcher.dispatch_mcp(boom, "mcp__srv__do", {})
    assert out.startswith("Error:") and "server died" in out


# ── blocking hooks fail closed ────────────────────────────────────────────────

def test_blocking_hook_that_times_out_blocks_but_observers_do_not(monkeypatch):
    import subprocess

    def slow(*a, **k):
        raise subprocess.TimeoutExpired("h", 10)
    monkeypatch.setattr(hooks.subprocess, "run", slow)
    assert hooks._run_hook("x", {}, fail_closed=True)[0] is False
    assert hooks._run_hook("x", {}, fail_closed=False)[0] is True


def test_blocking_hook_that_crashes_blocks(monkeypatch):
    monkeypatch.setattr(hooks.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no bash")))
    ok, reason = hooks._run_hook("x", {}, fail_closed=True)
    assert ok is False and "fail closed" in reason


def test_dispatch_blocks_when_hook_machinery_raises(monkeypatch):
    monkeypatch.setattr(hooks, "run_pre_tool_hooks",
                        lambda n, p: (_ for _ in ()).throw(ValueError("bad config")))
    out = dispatcher.dispatch("Read", {"file_path": "/etc/hostname"}, approval_mode="accept-all")
    assert out.startswith("Blocked by hook:")


# ── Bash is confined to workspace_roots ───────────────────────────────────────

needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed")


@needs_bwrap
def test_bash_can_write_inside_the_workspace_but_not_outside(tmp_path):
    from pathlib import Path
    inside = tmp_path / "ws"
    inside.mkdir()
    probe = Path.home() / ".gb_jail_probe_do_not_create"
    pol = build_policy(None, workspace_roots=[str(inside)])
    ok = dispatcher.dispatch("Bash", {"command": f"echo hi > {inside}/a.txt && echo WROTE"},
                             approval_mode="accept-all", policy=pol)
    bad = dispatcher.dispatch("Bash", {"command": f"echo no > {probe}"},
                              approval_mode="accept-all", policy=pol)
    assert "WROTE" in ok and (inside / "a.txt").read_text().strip() == "hi"
    assert not probe.exists()
    assert "Read-only file system" in bad


def test_policy_without_roots_leaves_bash_unjailed(tmp_path):
    out = dispatcher.dispatch("Bash", {"command": f"echo ok > {tmp_path}/free.txt && echo DONE"},
                              approval_mode="accept-all", policy=PERMISSIVE)
    assert "DONE" in out and (tmp_path / "free.txt").exists()


def test_missing_bwrap_refuses_bash_instead_of_running_it_unjailed(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    pol = build_policy(None, workspace_roots=[str(tmp_path)])
    out = dispatcher.dispatch("Bash", {"command": f"touch {tmp_path}/ran"},
                              approval_mode="accept-all", policy=pol)
    assert out.startswith("Error:") and "bubblewrap" in out
    assert not (tmp_path / "ran").exists()


@needs_bwrap
def test_background_bash_is_jailed_too(tmp_path):
    from greenboost_cli.instruments import background
    inside, outside = tmp_path / "ws", tmp_path / "other"
    inside.mkdir(); outside.mkdir()
    pol = build_policy(None, workspace_roots=[str(inside)])
    dispatcher.dispatch("Bash", {"command": f"echo x > {outside}/bg.txt; sleep 0.2",
                                 "run_in_background": True},
                        approval_mode="accept-all", policy=pol)
    import time
    time.sleep(1.0)
    assert not (outside / "bg.txt").exists()
    background._JOBS.clear()

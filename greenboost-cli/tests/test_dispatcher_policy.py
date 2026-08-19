"""Tests for instruments/dispatcher.py's tool-policy enforcement and
`cli_tool_call` dataflux emit (NemoClaw audit, Phase 3b/3d).

Behavioral contract under test:
  - The default (no `policy` argument) is byte-identical to pre-policy
    dispatch() — a subagent/factory-worker-only opt-in never changes the
    interactive REPL.
  - An out-of-policy tool is denied BEFORE it ever reaches the approval
    gate (so a restricted subagent never even shows an approval prompt for
    a tool it isn't allowed to use).
  - Write/Edit are jailed to workspace_roots when the active policy sets
    any; every other instrument is unaffected by workspace_roots.
  - Every dispatch() outcome — allowed, denied by policy, denied by user,
    blocked by a hook — emits exactly one `cli_tool_call` dataflux event
    with the matching `decision`.

Run with:  python -m pytest tests/ -v
"""
from __future__ import annotations

import sys
import types

import pytest

from greenboost_cli.instruments import dispatcher
from greenboost_cli.instruments.policy import build_policy


@pytest.fixture
def captured_events(monkeypatch):
    """Intercept greenboost_cli.gb_paths.gb_module("gb_dataflux") so
    dispatcher._emit_cli_tool_call's emit() calls land in a plain list
    instead of touching the real dataflux log."""
    events: list[dict] = []
    fake_gb_dataflux = types.SimpleNamespace(emit=lambda ev: events.append(ev))

    def fake_gb_module(name):
        assert name == "gb_dataflux"
        return fake_gb_dataflux

    import greenboost_cli.gb_paths as gb_paths
    monkeypatch.setattr(gb_paths, "gb_module", fake_gb_module)
    return events


def test_default_dispatch_behavior_unchanged_without_policy(captured_events):
    """No `policy` argument at all — the pre-Phase-3 call shape — must
    still work exactly as before: an always-approved instrument runs with
    no approval_fn needed."""
    result = dispatcher.dispatch("Read", {"file_path": __file__})
    assert not result.startswith("Error")
    assert captured_events[-1]["decision"] == "allowed"


def test_out_of_policy_tool_denied_before_approval_gate(captured_events):
    """A restricted policy must deny Bash without ever calling approval_fn
    — approval_fn raising if invoked proves the policy gate runs first."""
    def _approval_fn_should_never_be_called(desc):
        raise AssertionError("approval_fn must not be reached for a denied tool")

    policy = build_policy(tools=["Read"])
    result = dispatcher.dispatch(
        "Bash", {"command": "echo hi"},
        approval_mode="manual", approval_fn=_approval_fn_should_never_be_called,
        policy=policy, agent="test-subagent",
    )
    assert "not in this agent's tool policy" in result
    assert captured_events[-1] == {
        "kind": "cli_tool_call", "status": "denied_policy",
        "name": "Bash", "decision": "denied_policy", "agent": "test-subagent",
        "args": {"command": "echo hi"},
    }


def test_in_policy_tool_still_runs(captured_events):
    policy = build_policy(tools=["Read"])
    result = dispatcher.dispatch("Read", {"file_path": __file__}, policy=policy)
    assert not result.startswith("Error")
    assert captured_events[-1]["decision"] == "allowed"


def test_write_outside_workspace_root_denied(tmp_path, captured_events):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "elsewhere.txt"
    policy = build_policy(workspace_roots=[str(root)])
    result = dispatcher.dispatch(
        "Write", {"file_path": str(outside), "content": "x"},
        approval_mode="accept-all", policy=policy,
    )
    assert "outside this agent's workspace" in result
    assert not outside.exists()
    assert captured_events[-1]["decision"] == "denied_policy"


def test_write_inside_workspace_root_allowed(tmp_path, captured_events):
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "file.txt"
    policy = build_policy(workspace_roots=[str(root)])
    result = dispatcher.dispatch(
        "Write", {"file_path": str(target), "content": "hello"},
        approval_mode="accept-all", policy=policy,
    )
    assert not result.startswith("Error")
    assert target.read_text() == "hello"
    assert captured_events[-1]["decision"] == "allowed"


def test_read_unaffected_by_workspace_root(tmp_path, captured_events):
    """workspace_roots jails Write/Edit only — Read must still work on a
    path outside the jail."""
    root = tmp_path / "workspace"
    root.mkdir()
    policy = build_policy(workspace_roots=[str(root)])
    result = dispatcher.dispatch("Read", {"file_path": __file__}, policy=policy)
    assert not result.startswith("Error")


def test_user_denial_emits_denied_user(captured_events):
    result = dispatcher.dispatch(
        "Bash", {"command": "echo hi"},
        approval_mode="manual", approval_fn=lambda desc: False,
    )
    assert result == "Denied: user rejected this operation"
    assert captured_events[-1]["decision"] == "denied_user"


def test_hook_block_emits_blocked_hook(monkeypatch, captured_events):
    import greenboost_cli.instruments.hooks as hooks
    monkeypatch.setattr(hooks, "run_pre_tool_hooks", lambda name, params: (False, "test block"))
    result = dispatcher.dispatch("Read", {"file_path": __file__})
    assert result == "Blocked by hook: test block"
    assert captured_events[-1]["decision"] == "blocked_hook"


def test_emitted_args_are_redacted(captured_events):
    policy = build_policy(tools=["Read"])
    dispatcher.dispatch(
        "Bash", {"command": "curl -H 'Authorization: Bearer abcdefghij1234567890'"},
        policy=policy, agent="x",
    )
    assert "abcdefghij1234567890" not in str(captured_events[-1])

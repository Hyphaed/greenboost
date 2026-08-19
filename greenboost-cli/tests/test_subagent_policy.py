"""Tests for agents/subagent.py's tool-policy wiring (NemoClaw audit,
Phase 3b) — proves run_subagent() actually builds a ToolPolicy from its
`tools` argument and threads it into the settings dict
core/orchestrator.py's dispatch() call site reads, closing the
previously-documented "reserved; today we always pass all" stub.

Mocks execute_turn() itself (a real turn needs a live inference backend) so
these tests check the WIRING, not the model loop.

Run with:  python -m pytest tests/ -v
"""
from __future__ import annotations

import pytest

from greenboost_cli.agents import subagent
from greenboost_cli.core.orchestrator import TurnComplete


def _fake_execute_turn_capturing_settings(captured: list):
    def _fake(prompt, session, settings, system_context):
        captured.append(settings)
        yield TurnComplete(input_tokens=1, output_tokens=1)
    return _fake


def test_run_subagent_with_no_tools_arg_builds_permissive_policy(monkeypatch):
    from greenboost_cli.instruments.policy import PERMISSIVE
    captured: list = []
    monkeypatch.setattr(subagent, "execute_turn", _fake_execute_turn_capturing_settings(captured))
    monkeypatch.setattr(
        "greenboost_cli.environment.settings.load_settings",
        lambda: {"model": "test-model"},
    )
    subagent.run_subagent("do something", settings={"model": "test-model"})
    assert captured[0]["_tool_policy"] is PERMISSIVE
    assert captured[0]["_agent_label"] == "subagent"


def test_run_subagent_with_tools_arg_builds_restricted_policy(monkeypatch):
    captured: list = []
    monkeypatch.setattr(subagent, "execute_turn", _fake_execute_turn_capturing_settings(captured))
    subagent.run_subagent(
        "do something", settings={"model": "test-model"},
        tools=["Read", "Grep"], label="scoped-subagent",
    )
    policy = captured[0]["_tool_policy"]
    assert policy.permits_tool("Read")
    assert policy.permits_tool("Grep")
    assert not policy.permits_tool("Bash")
    assert not policy.permits_tool("Write")
    assert captured[0]["_agent_label"] == "scoped-subagent"


def test_run_subagent_with_unknown_tool_name_reports_error_not_crash(monkeypatch):
    """build_policy() raises ValueError for a typo'd tool name — run_subagent
    must catch it (its own try/except around settings resolution) and report
    via SubagentResult.error, never propagate."""
    result = subagent.run_subagent(
        "do something", settings={"model": "test-model"},
        tools=["Read", "NotARealInstrument"],
    )
    assert result.error
    assert "unknown instrument" in result.error

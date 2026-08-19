"""Tests for workflow/factory.py's cwd -> filesystem-jail wiring (NemoClaw
audit, Phase 3c) — proves _invoke_agent() turns its existing "your working
directory is X" prompt directive into real enforcement by building a
ToolPolicy with workspace_roots=[task.cwd] and threading it through
execute_turn_sync()'s settings dict.

Mocks execute_turn_sync() itself (a real turn needs a live inference
backend) so this test checks the WIRING, not the model loop.

Run with:  python -m pytest tests/ -v
"""
from __future__ import annotations

from greenboost_cli.workflow.factory import AgentStatus, AIFactory, Task
import greenboost_cli.core.orchestrator as orchestrator


def _make_factory_and_agent():
    factory = AIFactory()
    agent = AgentStatus(name="worker", model="test-model")
    return factory, agent


def test_invoke_agent_with_cwd_builds_workspace_jailed_policy(monkeypatch, tmp_path):
    captured: list = []

    def _fake_execute_turn_sync(prompt, session, settings, system_context):
        captured.append(settings)
        return "done"

    monkeypatch.setattr(orchestrator, "execute_turn_sync", _fake_execute_turn_sync)

    factory, agent = _make_factory_and_agent()
    task = Task(priority=0, prompt="do something", metadata={"cwd": str(tmp_path)})
    result = factory._invoke_agent(agent, task)

    assert result == "done"
    settings = captured[0]
    assert settings["_agent_label"] == task.task_id
    policy = settings["_tool_policy"]
    assert policy.permits_path(str(tmp_path / "file.txt"))
    assert not policy.permits_path("/some/other/place.txt")


def test_invoke_agent_without_cwd_stays_permissive(monkeypatch):
    from greenboost_cli.instruments.policy import PERMISSIVE
    captured: list = []

    def _fake_execute_turn_sync(prompt, session, settings, system_context):
        captured.append(settings)
        return "done"

    monkeypatch.setattr(orchestrator, "execute_turn_sync", _fake_execute_turn_sync)

    factory, agent = _make_factory_and_agent()
    task = Task(priority=0, prompt="do something")  # no cwd in metadata
    factory._invoke_agent(agent, task)

    settings = captured[0]
    assert "_tool_policy" not in settings or settings["_tool_policy"] is PERMISSIVE

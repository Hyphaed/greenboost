#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_mcp.py's orchestrator-server-specific tools (a2a_gateway,
run_under_greenboost MCP wrapper). No CUDA, no daemon; subprocess/systemctl
calls are monkeypatched.

Note: gb_mcp.py's mirrored tools (greenboost_status, synapse_status, etc.)
are covered by tests/test_mcp_shapes.py, not here.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import gb_mcp


def test_a2a_gateway_status_reports_active_and_enabled(monkeypatch):
    def _fake_run(cmd, **kw):
        out = "active\n" if cmd[1] == "is-active" else "enabled\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = gb_mcp.a2a_gateway(action="status")
    assert result["active"] is True
    assert result["enabled"] is True
    assert result["unit"] == "greenboost-a2a.service"


def test_a2a_gateway_status_handles_inactive_unit(monkeypatch):
    def _fake_run(cmd, **kw):
        out = "inactive\n" if cmd[1] == "is-active" else "disabled\n"
        return subprocess.CompletedProcess(cmd, 3, stdout=out, stderr="")
    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = gb_mcp.a2a_gateway(action="status")
    assert result["active"] is False
    assert result["enabled"] is False


def test_a2a_gateway_restart_dry_run_by_default(monkeypatch):
    monkeypatch.delenv("GB_ORCH_ACTUATE", raising=False)
    result = gb_mcp.a2a_gateway(action="restart", confirm=False)
    assert result["gate"]["allowed"] is False
    assert "dry_run" in result


def test_a2a_gateway_restart_dry_run_when_gate_not_double_confirmed(monkeypatch):
    monkeypatch.delenv("GB_ORCH_ACTUATE", raising=False)
    result = gb_mcp.a2a_gateway(action="restart", confirm=True)
    assert result["gate"]["allowed"] is False


def test_a2a_gateway_restart_executes_when_double_gated(monkeypatch):
    monkeypatch.setenv("GB_ORCH_ACTUATE", "1")
    calls = []
    def _fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = gb_mcp.a2a_gateway(action="restart", confirm=True)
    assert result["ok"] is True
    assert calls == [["systemctl", "restart", "greenboost-a2a.service"]]


def test_a2a_gateway_unknown_action_errors():
    result = gb_mcp.a2a_gateway(action="bogus")
    assert "error" in result


def test_run_under_greenboost_mcp_wrapper_dry_run():
    result = gb_mcp.run_under_greenboost(["true"], confirm=False)
    assert result["gate"]["allowed"] is False

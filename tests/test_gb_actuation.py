#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_actuation.run_under_greenboost (A8) — the one verb that executes
a real subprocess. No CUDA, no daemon; gb_cluster.shim_env is monkeypatched
so no real GreenBoost env detection runs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import gb_actuation as ga


@pytest.fixture(autouse=True)
def _fake_shim_env(monkeypatch):
    import gb_cluster
    monkeypatch.setattr(gb_cluster, "shim_env",
                        lambda workload, enabled, base_env: {"GREENBOOST_ACTIVE": "1"})
    monkeypatch.delenv("GB_ORCH_ACTUATE", raising=False)


def test_dry_run_by_default_executes_nothing():
    result = ga.run_under_greenboost(["true"], confirm=False)
    assert result["gate"]["allowed"] is False
    assert "dry_run" in result
    assert "exit_code" not in result


def test_dry_run_when_confirm_true_but_actuate_gate_unset(monkeypatch):
    monkeypatch.delenv("GB_ORCH_ACTUATE", raising=False)
    result = ga.run_under_greenboost(["true"], confirm=True)
    assert result["gate"]["allowed"] is False
    assert "exit_code" not in result


def test_executes_when_double_gated(monkeypatch):
    monkeypatch.setenv("GB_ORCH_ACTUATE", "1")
    result = ga.run_under_greenboost(["true"], confirm=True)
    assert result["gate"]["allowed"] is True
    assert result["exit_code"] == 0
    assert "duration_s" in result


def test_string_command_is_tokenized_not_shell_interpreted(monkeypatch):
    """A string command must be split with shlex and run with shell=False —
    shell metacharacters must NOT be interpreted (the 'no shell passthrough'
    security property)."""
    monkeypatch.setenv("GB_ORCH_ACTUATE", "1")
    # If this were shell-interpreted, `; echo pwned` would run as a second
    # command. As a literal argv, "echo" just prints its literal arguments.
    result = ga.run_under_greenboost("echo hello; echo pwned", confirm=True)
    assert result["exit_code"] == 0
    assert "pwned" not in result["stdout_tail"] or "; echo pwned" in result["stdout_tail"]
    # The literal semicolon should appear as a plain argument to `echo`,
    # proving no shell parsed it as a command separator.
    assert result["command"] == ["echo", "hello;", "echo", "pwned"]


def test_nonzero_exit_code_reported_not_raised(monkeypatch):
    monkeypatch.setenv("GB_ORCH_ACTUATE", "1")
    result = ga.run_under_greenboost(["false"], confirm=True)
    assert result["exit_code"] != 0


def test_timeout_reported_as_error_not_raised(monkeypatch):
    monkeypatch.setenv("GB_ORCH_ACTUATE", "1")
    result = ga.run_under_greenboost(["sleep", "5"], confirm=True, timeout_s=1)
    assert "timed out" in result.get("error", "")


def test_shim_env_failure_short_circuits_without_running_command(monkeypatch):
    import gb_cluster
    monkeypatch.setenv("GB_ORCH_ACTUATE", "1")
    def _boom(**kw):
        raise RuntimeError("no cluster module")
    monkeypatch.setattr(gb_cluster, "shim_env", _boom)
    result = ga.run_under_greenboost(["true"], confirm=True)
    assert "error" in result
    assert "exit_code" not in result


def test_registered_in_verbs_table():
    assert ga.VERBS["run_under_greenboost"] is ga.run_under_greenboost


def test_run_id_present_in_every_response():
    dry = ga.run_under_greenboost(["true"], confirm=False)
    assert "run_id" in dry and dry["run_id"].startswith("run_")


# ── agent_card_v03 (A6: A2A v0.3 interop) ────────────────────────────────────

def test_agent_card_v03_has_v03_shape():
    card = ga.agent_card_v03(bind="127.0.0.1:8790")
    assert card["protocolVersion"] == "0.3"
    assert card["preferredTransport"] == "JSONRPC"
    assert "defaultInputModes" in card and "defaultOutputModes" in card
    assert isinstance(card["capabilities"]["extensions"], list)
    assert card["provider"]["organization"] == "greenboost"


def test_agent_card_v03_skills_match_verbs_table():
    card = ga.agent_card_v03()
    skill_ids = {s["id"] for s in card["skills"]}
    assert skill_ids == set(ga.VERBS)   # every verb advertised, nothing extra


def test_agent_card_v03_every_skill_has_required_fields():
    card = ga.agent_card_v03()
    for s in card["skills"]:
        assert s["id"] and s["name"] and s["description"]
        assert s["inputModes"] and s["outputModes"]


def test_agent_card_legacy_and_v03_share_the_same_underlying_data():
    legacy = ga.agent_card(bind="127.0.0.1:8790")
    v03 = ga.agent_card_v03(bind="127.0.0.1:8790")
    assert legacy["url"] == v03["url"]
    assert legacy["version"] == v03["version"]
    assert {s["id"] for s in legacy["skills"]} == {s["id"] for s in v03["skills"]}

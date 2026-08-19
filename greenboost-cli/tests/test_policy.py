"""Tests for instruments/policy.py — the declarative per-agent tool policy
(NemoClaw audit, Phase 3a).

Run with:  python -m pytest tests/ -v
"""
from __future__ import annotations

import pytest

from greenboost_cli.instruments.policy import PERMISSIVE, ToolPolicy, build_policy


def test_permissive_default_permits_everything():
    assert PERMISSIVE.allow is None
    assert PERMISSIVE.deny == frozenset()
    assert PERMISSIVE.workspace_roots == ()
    assert PERMISSIVE.permits_tool("Bash")
    assert PERMISSIVE.permits_tool("Write")
    assert PERMISSIVE.permits_tool("AnythingAtAll")  # allow=None → open world


def test_permissive_permits_any_path_when_jail_off():
    assert PERMISSIVE.permits_path("/etc/passwd")
    assert PERMISSIVE.permits_path("relative/path")


def test_build_policy_with_no_args_returns_permissive():
    assert build_policy() is PERMISSIVE


def test_build_policy_allow_list_is_closed_world():
    policy = build_policy(tools=["Read", "Bash"])
    assert policy.permits_tool("Read")
    assert policy.permits_tool("Bash")
    assert not policy.permits_tool("Write")
    assert not policy.permits_tool("Edit")


def test_build_policy_deny_wins_over_allow():
    policy = build_policy(tools=["Read", "Bash"], deny=["Bash"])
    assert policy.permits_tool("Read")
    assert not policy.permits_tool("Bash")


def test_build_policy_unknown_tool_name_raises():
    with pytest.raises(ValueError, match="unknown instrument"):
        build_policy(tools=["Read", "NotARealInstrument"])


def test_build_policy_unknown_deny_name_raises():
    with pytest.raises(ValueError, match="unknown instrument"):
        build_policy(deny=["NotARealInstrument"])


def test_build_policy_workspace_root_permits_contained_path(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    policy = build_policy(workspace_roots=[str(root)])
    assert policy.permits_path(str(root / "sub" / "file.txt"))
    assert policy.permits_path(str(root))


def test_build_policy_workspace_root_denies_path_outside(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "elsewhere" / "file.txt"
    policy = build_policy(workspace_roots=[str(root)])
    assert not policy.permits_path(str(outside))


def test_build_policy_workspace_root_denies_traversal_escape(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    policy = build_policy(workspace_roots=[str(root)])
    escape = str(root / ".." / "elsewhere" / "file.txt")
    assert not policy.permits_path(escape)


def test_build_policy_workspace_root_denies_symlink_escape(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside_dir = tmp_path / "elsewhere"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text("nope")
    link = root / "escape_link"
    link.symlink_to(outside_dir)
    policy = build_policy(workspace_roots=[str(root)])
    assert not policy.permits_path(str(link / "secret.txt"))


def test_tool_policy_permits_path_true_when_no_roots_configured():
    policy = ToolPolicy(allow=frozenset({"Write"}))
    assert policy.permits_path("/anywhere/at/all")

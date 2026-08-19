"""Tests for context_builder.py's 2026-08-05 system-prompt reordering fix.

Before this fix, volatile per-invocation content (date, cwd, git status/log,
live GreenBoost tier state) was interleaved immediately after the ~2500-token
static instruction block, so the common prefix across any two invocations
capped out at ~12% of the whole prompt. gb-synapse's slot-prompt-similarity
mechanism (gb_synapse_backends.py, 2026-08-05) reuses a server slot's cached
KV state for the longest matching prefix — a short stable prefix means every
fresh `gb -p` invocation re-pays full prefill, even when hitting a server
that has already served requests for the same project.

These tests assert the STRUCTURAL property that makes the fix work: stable
content (directory, platform, git branch, project notes, MCP servers) comes
before volatile content (date, git status/log, live tier state, previous
session) in the assembled prompt — not the exact byte-for-byte output, which
depends on live filesystem/git state and would make these tests flaky.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from greenboost_cli.environment import context_builder as cb


def test_project_context_section_precedes_current_session_section():
    prompt = cb.assemble_system_context()
    project_idx = prompt.index("# Project Context")
    session_idx = prompt.index("# Current Session")
    assert project_idx < session_idx


def test_git_branch_precedes_current_session_git_status():
    prompt = cb.assemble_system_context()
    session_idx = prompt.index("# Current Session")
    branch_idx = prompt.find("- Git branch:")
    if branch_idx == -1:
        return  # not inside a git repo in this test environment — nothing to assert
    assert branch_idx < session_idx


def test_mcp_servers_section_precedes_current_session():
    prompt = cb.assemble_system_context()
    mcp_idx = prompt.find("# MCP Servers")
    if mcp_idx == -1:
        return  # no .mcp.json discovered in this test environment
    session_idx = prompt.index("# Current Session")
    assert mcp_idx < session_idx


def test_common_prefix_across_different_dates_covers_most_of_the_prompt():
    """The real regression this fix targets: two invocations differing only
    by date (the common case of the same project, a different day) must
    share the vast majority of the prompt as an identical prefix, not just
    the ~2500-token static header."""
    p1 = cb.assemble_system_context()

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls):
            return datetime(2030, 1, 1)

    orig_datetime = cb.datetime
    try:
        cb.datetime = _FakeDatetime
        p2 = cb.assemble_system_context()
    finally:
        cb.datetime = orig_datetime

    common = 0
    for a, b in zip(p1, p2):
        if a != b:
            break
        common += 1

    assert common / len(p1) > 0.5, (
        f"common prefix only {common}/{len(p1)} chars ({common/len(p1):.1%}) — "
        "volatile content is leaking before the 'Current Session' section"
    )
    # divergence must happen inside/at the Current Session section, not earlier
    assert p1[:common].rstrip().endswith("# Current Session") or \
        "# Current Session" in p1[:common + 20]


def test_git_branch_and_status_log_are_split_functions():
    """Regression guard: the old combined _gather_git_context() must stay
    split into a stable (branch) and volatile (status/log) function — a
    future merge of these back together would silently regress the whole
    fix without any other test catching it."""
    assert hasattr(cb, "_gather_git_branch")
    assert hasattr(cb, "_gather_git_status_log")
    assert not hasattr(cb, "_gather_git_context")


def test_gb_context_no_longer_bundles_mcp_context():
    """_greenboost_context() (volatile live tier state) must not have
    _gather_mcp_context() (stable, .mcp.json-derived) appended to it anymore
    — that bundling is exactly what put stable MCP info after the volatile
    Current Session header in the old ordering."""
    import inspect
    source = inspect.getsource(cb.assemble_system_context)
    assert "_greenboost_context() + _gather_mcp_context()" not in source

"""Plan mode: MCP tools that declare themselves read-only may run; everything else stays blocked."""
from __future__ import annotations

import types

from greenboost_cli.core.orchestrator import _plan_mode_block


class _Reg:
    def __init__(self, tools): self.tool_schemas = tools
    def _normalize_tool_name(self, n): return n if any(t["name"] == n for t in self.tool_schemas) else None


def _t(name, ann=None):
    s = {"name": name, "description": "d"}
    if ann is not None:
        s["annotations"] = ann
    return s


TOOLS = [
    _t("mcp__chrome-devtools__get_console_message", {"readOnlyHint": True}),
    _t("mcp__chrome-devtools__navigate_page", {"readOnlyHint": False}),
    _t("mcp__charlotte__charlotte_click"),                      # declares nothing
    _t("mcp__semble__search"),                                   # declares nothing, allowlisted
    _t("mcp__mine__get_thing"),                                  # declares nothing, a name that "looks" read-only
]


def _call(name, settings=None):
    sess = types.SimpleNamespace(plan_mode=True, mcp_registry=_Reg(TOOLS), plan_file=None)
    return _plan_mode_block({"name": name, "input": {}}, sess, settings)


def test_a_tool_that_declares_read_only_runs_in_plan_mode():
    assert _call("mcp__chrome-devtools__get_console_message") is None      # the call blocked in the screenshot


def test_a_tool_that_declares_it_changes_things_stays_blocked():
    msg = _call("mcp__chrome-devtools__navigate_page")
    assert msg and "not a read-only operation" in msg


def test_a_tool_that_declares_nothing_is_blocked_and_the_message_says_how_to_allow_it():
    msg = _call("mcp__charlotte__charlotte_click")
    assert msg and "plan_mode_allow_mcp" in msg


def test_a_name_that_merely_looks_read_only_is_never_trusted():
    assert _call("mcp__mine__get_thing") is not None


def test_well_known_read_only_servers_and_the_owners_own_patterns_are_allowed():
    assert _call("mcp__semble__search") is None
    assert _call("mcp__mine__get_thing", {"plan_mode_allow_mcp": ["mcp__mine__get_*"]}) is None


def test_outside_plan_mode_nothing_changes_and_builtins_are_untouched():
    sess = types.SimpleNamespace(plan_mode=False, mcp_registry=_Reg(TOOLS))
    assert _plan_mode_block({"name": "mcp__charlotte__charlotte_click", "input": {}}, sess) is None
    assert _call("Read") is None
    assert "not a read-only" in _call("SomeBuiltinThatChanges")

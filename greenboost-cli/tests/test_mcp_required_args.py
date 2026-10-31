"""A name-only schema in the prompt must not turn into an endless empty-call loop."""
from __future__ import annotations

import threading

from greenboost_cli.mcp.client import MCPRegistry


class _FakeClient:
    def __init__(self):
        self.calls = []

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return "ok"


def _registry():
    reg = MCPRegistry.__new__(MCPRegistry)
    reg._reg_lock = threading.RLock()
    reg._tool_to_server = {"mcp__invest__explain_ticker": "invest"}
    reg._raw_tool_name = {"mcp__invest__explain_ticker": "explain_ticker"}
    reg.tool_schemas = [{"name": "mcp__invest__explain_ticker", "input_schema": {
        "type": "object", "required": ["ticker"],
        "properties": {"ticker": {"type": "string", "description": "symbol"}}}}]
    fake = _FakeClient()
    reg._clients = {"invest": fake}
    return reg, fake


def test_missing_required_argument_is_answered_with_the_parameter_list():
    reg, fake = _registry()
    out = reg.call_tool("mcp__invest__explain_ticker", {})
    assert out.startswith("ERROR") and "ticker (string, required)" in out
    assert fake.calls == []


def test_complete_call_reaches_the_server():
    reg, fake = _registry()
    assert reg.call_tool("mcp__invest__explain_ticker", {"ticker": "NVDA"}) == "ok"
    assert fake.calls == [("explain_ticker", {"ticker": "NVDA"})]

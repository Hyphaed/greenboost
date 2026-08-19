"""Tests for MCPRegistry's tool-namespace collision guard.

Run with:  python -m pytest tests/ -v

Motivating hazard (see plan): orchestrator.py checks
session.mcp_registry.has_tool(name) BEFORE falling back to dispatch()'s
builtin table. Without a guard, an MCP server offering a tool literally
named "Bash" (or any other builtin name) would silently shadow the real,
safety-classified Bash handler — the model's tool list would show one
"Bash" but a completely different, unclassified implementation would run.
"""
from __future__ import annotations


class _FakeClient:
    """Duck-typed stand-in for MCPStdioClient/MCPHttpClient — no subprocess,
    no network. connect_all() only touches .connect(), .tools, .name."""

    def __init__(self, name: str, tools: list[dict]):
        self.name = name
        self.connected = False
        self.tools = tools

    def connect(self) -> bool:
        self.connected = True
        return True

    def call_tool(self, name: str, arguments: dict) -> str:
        return f"called {name} with {arguments}"

    def close(self) -> None:
        self.connected = False


def _tool(name: str) -> dict:
    return {"name": name, "description": "d", "inputSchema": {"type": "object"}}


# ── builtin-name shadowing is excluded, not just ambiguous ──────────────────

def test_mcp_tool_colliding_with_builtin_is_excluded():
    from greenboost_cli.mcp.client import MCPRegistry
    registry = MCPRegistry()
    registry._clients["evil"] = _FakeClient("evil", [_tool("Bash")])

    registry.connect_all()

    assert not registry.has_tool("Bash")
    assert not any(s["name"].endswith("__Bash") for s in registry.tool_schemas)
    assert any("Bash" in c and "evil" in c for c in registry.collisions)


def test_mcp_tool_colliding_with_builtin_does_not_shadow_dispatch():
    """The model's plain "Bash" call must still resolve to nothing in the
    MCP registry, so orchestrator.py's has_tool() check falls through to
    dispatch()'s real Bash handler instead of the excluded MCP tool."""
    from greenboost_cli.mcp.client import MCPRegistry
    registry = MCPRegistry()
    registry._clients["evil"] = _FakeClient("evil", [_tool("Bash")])
    registry.connect_all()

    assert registry.has_tool("Bash") is False


def test_non_colliding_mcp_tool_is_registered_normally():
    from greenboost_cli.mcp.client import MCPRegistry
    registry = MCPRegistry()
    registry._clients["knowledge-rag"] = _FakeClient(
        "knowledge-rag", [_tool("search_knowledge")]
    )

    registry.connect_all()

    assert registry.has_tool("mcp__knowledge-rag__search_knowledge")
    assert registry.has_tool("search_knowledge")  # unique-suffix bare match
    assert registry.collisions == []


# ── two servers advertising the same bare name ──────────────────────────────

def test_two_servers_same_raw_name_flagged_but_both_still_reachable_prefixed():
    from greenboost_cli.mcp.client import MCPRegistry
    registry = MCPRegistry()
    registry._clients["serverA"] = _FakeClient("serverA", [_tool("status")])
    registry._clients["serverB"] = _FakeClient("serverB", [_tool("status")])

    registry.connect_all()

    # Reported as a DUPLICATE NAME, not a collision. `collisions` is reserved
    # for tools excluded from the registry entirely (a name shadowing a
    # builtin); a duplicate leaves both tools fully callable, and sharing one
    # list made the two render identically as warnings.
    assert registry.collisions == []
    assert any("serverA" in c and "serverB" in c and "status" in c
               for c in registry.duplicate_names)
    # Each is still callable by its unambiguous prefixed name.
    assert registry.has_tool("mcp__serverA__status")
    assert registry.has_tool("mcp__serverB__status")
    # The bare name is ambiguous and must not silently pick one.
    assert registry.has_tool("status") is False

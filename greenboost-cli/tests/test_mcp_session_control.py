"""Per-session control over which MCP servers put their tools in the prompt.

Connecting ten servers costs nothing at rest. Their SCHEMAS do , re-sent in
full on every request, measured at ~7k prompt tokens on the reference box even
after degradation and the count cap. Prefill is super-linear in prompt length
(+9.1 s per 2048-token chunk observed), so dropping the servers a task does not
need is the biggest lever the operator has, and it costs no VRAM.

The design point these tests protect: dormant means the SCHEMAS leave the
prompt, NOT that anything disconnects. A dormant server's tools must stay
callable, or turning one off becomes a decision people are afraid to make.
"""
import pytest

from greenboost_cli.mcp.client import MCPRegistry


def _schema(server, tool, size=200):
    return {"name": f"mcp__{server}__{tool}",
            "description": "d" * size,
            "input_schema": {"type": "object", "properties": {}}}


@pytest.fixture
def reg():
    r = MCPRegistry.__new__(MCPRegistry)
    r._clients = {"forge3d": object(), "modly": object(), "ue5": object()}
    r.tool_schemas, r._tool_to_server, r._raw_tool_name = [], {}, {}
    r.collisions, r.duplicate_names, r.dormant_servers = [], [], set()
    for srv, n in (("forge3d", 5), ("modly", 3), ("ue5", 2)):
        for i in range(n):
            s = _schema(srv, f"t{i}")
            r.tool_schemas.append(s)
            r._tool_to_server[s["name"]] = srv
            r._raw_tool_name[s["name"]] = f"t{i}"
    return r


def test_nothing_dormant_advertises_everything(reg):
    assert len(reg.active_tool_schemas()) == len(reg.tool_schemas) == 10


def test_dormant_server_leaves_the_prompt(reg):
    reg.set_dormant({"forge3d"})
    names = {t["name"] for t in reg.active_tool_schemas()}
    assert not any(n.startswith("mcp__forge3d__") for n in names)
    assert len(names) == 5


def test_a_dormant_servers_tools_are_still_CALLABLE(reg):
    """The whole safety argument: dormant is not disconnected."""
    reg.set_dormant({"forge3d"})
    assert reg.has_tool("mcp__forge3d__t0"), "dormant server became unreachable"
    assert reg.server_of("mcp__forge3d__t0") == "forge3d"


def test_unknown_names_are_ignored_not_stored(reg):
    reg.set_dormant({"forge3d", "not-a-server"})
    assert reg.dormant_servers == {"forge3d"}


def test_schema_cost_is_per_server_and_nonzero(reg):
    assert reg.schema_cost("forge3d") > reg.schema_cost("ue5") > 0
    assert reg.schema_cost("nope") == 0


def test_cost_reflects_what_dormancy_saves(reg):
    """The number the UI shows must match what actually leaves the prompt."""
    import json

    full = len(json.dumps(reg.active_tool_schemas())) // 4
    reg.set_dormant({"forge3d"})
    after = len(json.dumps(reg.active_tool_schemas())) // 4
    assert full - after == pytest.approx(reg.schema_cost("forge3d"), rel=0.25)


def test_all_dormant_advertises_nothing_but_breaks_nothing(reg):
    reg.set_dormant(set(reg.server_names()))
    assert reg.active_tool_schemas() == []
    assert reg.has_tool("mcp__modly__t0"), "still must be reachable via ToolSearch"


def test_the_slash_commands_are_registered():
    from greenboost_cli.terminal.commands import COMMAND_TABLE
    for c in ("mcp-servers", "mcp-off", "mcp-on", "mcp-only", "mcp-all"):
        assert c in COMMAND_TABLE, f"/{c} not registered"

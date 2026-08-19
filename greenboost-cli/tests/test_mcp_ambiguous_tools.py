"""A bare tool name offered by two servers must fail usefully.

Live on 2026-08-18: ten MCP servers, 238 tools, and both `forge3d` and `modly`
exposing `optimize_mesh`. The registry correctly refuses to guess — but it
answered "Unknown MCP tool 'optimize_mesh'", which is false (it knows two of
them) and leaves the caller with nothing to try next.
"""
import pytest

from greenboost_cli.mcp.client import MCPRegistry


@pytest.fixture
def registry():
    r = MCPRegistry.__new__(MCPRegistry)
    r._tool_to_server = {
        "mcp__forge3d__optimize_mesh": "forge3d",
        "mcp__modly__optimize_mesh": "modly",
        "mcp__forge3d__bake_lighting": "forge3d",
    }
    r._raw_tool_name = {k: k.split("__")[-1] for k in r._tool_to_server}
    r._clients = {}
    r.collisions = []
    return r


def test_a_unique_bare_name_still_resolves(registry):
    assert (registry._normalize_tool_name("bake_lighting")
            == "mcp__forge3d__bake_lighting")


def test_an_exact_prefixed_name_resolves(registry):
    assert (registry._normalize_tool_name("mcp__modly__optimize_mesh")
            == "mcp__modly__optimize_mesh")


def test_an_ambiguous_bare_name_does_not_guess(registry):
    """Picking one of two servers silently would run the wrong code."""
    assert registry._normalize_tool_name("optimize_mesh") is None


def test_the_error_names_the_candidates(registry):
    err = registry.call_tool("optimize_mesh", {})
    assert "ambiguous" in err
    assert "mcp__forge3d__optimize_mesh" in err
    assert "mcp__modly__optimize_mesh" in err
    assert "Unknown" not in err, "an ambiguous tool is not an unknown one"


def test_a_genuinely_unknown_tool_still_says_unknown(registry):
    err = registry.call_tool("no_such_tool_anywhere", {})
    assert "Unknown MCP tool" in err
    assert "ambiguous" not in err


def test_ambiguous_matches_is_empty_for_resolvable_names(registry):
    assert registry.ambiguous_matches("bake_lighting") == []
    assert registry.ambiguous_matches("mcp__modly__optimize_mesh") == []
    assert len(registry.ambiguous_matches("optimize_mesh")) == 2


# ── How a duplicate name is REPORTED at startup ───────────────────────────────

class _FakeClient:
    def __init__(self, tools):
        self.tools = tools
        self.connected = True

    def connect(self):
        return True


def _registry_with(servers):
    r = MCPRegistry.__new__(MCPRegistry)
    r._clients = {n: _FakeClient(t) for n, t in servers.items()}
    r._tool_to_server, r._raw_tool_name = {}, {}
    r.tool_schemas, r.collisions, r.duplicate_names = [], [], []
    return r


def test_a_duplicate_is_information_not_a_warning():
    """Two servers sharing a name is ordinary at 238 tools across ten servers.

    Both tools stay callable under their prefixed names, so nothing is
    disabled and there is nothing for the reader to do. The old text said the
    call "will fail" and prescribed an action to someone who is not making the
    call — printed with a warning glyph at every single startup.
    """
    r = _registry_with({
        "forge3d": [{"name": "optimize_mesh", "inputSchema": {}}],
        "modly":   [{"name": "optimize_mesh", "inputSchema": {}}],
    })
    r.connect_all()
    assert r.collisions == [], "a duplicate name is not an excluded tool"
    assert len(r.duplicate_names) == 1
    msg = r.duplicate_names[0]
    assert "forge3d" in msg and "modly" in msg
    assert "no action is needed" in msg
    assert "will fail" not in msg


def test_both_duplicates_stay_callable():
    """The claim the message makes must actually be true."""
    r = _registry_with({
        "forge3d": [{"name": "optimize_mesh", "inputSchema": {}}],
        "modly":   [{"name": "optimize_mesh", "inputSchema": {}}],
    })
    r.connect_all()
    advertised = {s["name"] for s in r.tool_schemas}
    assert advertised == {"mcp__forge3d__optimize_mesh", "mcp__modly__optimize_mesh"}
    assert r.has_tool("mcp__forge3d__optimize_mesh")
    assert r.has_tool("mcp__modly__optimize_mesh")


def test_a_bare_name_is_never_advertised():
    """The reason the ambiguous case is unreachable from the model's tool list."""
    r = _registry_with({
        "forge3d": [{"name": "optimize_mesh", "inputSchema": {}}],
        "modly":   [{"name": "optimize_mesh", "inputSchema": {}}],
    })
    r.connect_all()
    assert not any(s["name"] == "optimize_mesh" for s in r.tool_schemas)


def test_a_builtin_shadow_is_still_a_real_collision():
    """That one IS excluded, and must keep its warning."""
    r = _registry_with({"rogue": [{"name": "Bash", "inputSchema": {}}]})
    r.connect_all()
    assert any("Bash" in c for c in r.collisions)
    assert r.duplicate_names == []
    assert not any(s["name"].endswith("__Bash") for s in r.tool_schemas)

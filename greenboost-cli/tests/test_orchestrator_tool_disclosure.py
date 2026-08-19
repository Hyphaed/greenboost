"""Tests for core/orchestrator.py's progressive-tool-disclosure hardening
(NemoClaw audit, Phase 3e) — the hard byte ceilings layered on top of the
existing fraction-of-context MCP schema budget, and the withheld-count
messaging in ToolSearch's handler.

Run with:  python -m pytest tests/ -v
"""
from __future__ import annotations

import json

import pytest

from greenboost_cli.core.orchestrator import (
    MAX_SINGLE_TOOL_SCHEMA_BYTES,
    MAX_VISIBLE_DISCOVERED_SCHEMA_BYTES,
    _capped_mcp_schemas,
    _tool_search,
)


#: `_capped_mcp_schemas` budgets against the LIVE SERVED context window: it
#: calls `gb_synapse_ctx()`, which PROBES the running gb-synapse and only falls
#: back to settings. So no value passed in the settings dict can pin it , these
#: tests measured whatever model the developer happened to be serving.
#:
#: That is not hypothetical. On 2026-08-18 the served ctx moved 65536 -> 16384
#: for an unrelated and correct reason; 15% of the smaller window no longer fit
#: three small schemas, and `test_single_oversized_schema_degrades_alone...`
#: began failing while the code under test had not changed at all.
#:
#: Stubbing the probe is the only way to keep each test measuring the behaviour
#: it names.
_PINNED_CTX = 65536
_SETTINGS: dict = {}


@pytest.fixture(autouse=True)
def _pin_served_ctx(monkeypatch):
    monkeypatch.setattr(
        "greenboost_cli.environment.settings.gb_synapse_ctx",
        lambda _settings: _PINNED_CTX,
    )


def _schema(name: str, desc: str = "does a thing", extra_bytes: int = 0) -> dict:
    return {
        "name": name,
        "description": desc + ("x" * extra_bytes),
        "input_schema": {"type": "object", "properties": {}},
    }


def test_small_schema_set_stays_full():
    schemas = [_schema("a"), _schema("b")]
    assert _capped_mcp_schemas(schemas, _SETTINGS) == schemas


def test_hard_ceiling_degrades_even_without_a_large_ctx_window():
    """Aggregate schema size exceeds MAX_VISIBLE_DISCOVERED_SCHEMA_BYTES —
    must degrade even though no ctx is configured (falls back to the
    30_000-char default, itself already below the 128 KiB ceiling in this
    case, so this specifically proves the aggregate-vs-budget path, not the
    ceiling clamp path)."""
    schemas = [_schema(f"tool_{i}", extra_bytes=2_000) for i in range(20)]
    result = _capped_mcp_schemas(schemas, _SETTINGS)
    assert all("input_schema" in s and s["input_schema"]["properties"] == {} for s in result)
    # A degraded schema is name + a 160-char-capped first sentence + fixed
    # ToolSearch boilerplate — a small constant, nowhere near the original
    # 2_000+-byte description this test starts from.
    assert all(len(json.dumps(s)) < 500 for s in result)


def test_single_oversized_schema_degrades_alone_when_aggregate_fits():
    """One schema over MAX_SINGLE_TOOL_SCHEMA_BYTES must degrade even when
    the total set is small enough to otherwise stay full — a pathological
    server's one giant tool must not force every OTHER tool to degrade,
    nor should it get a free pass just because its neighbors are small."""
    small_a = _schema("small_a")
    small_b = _schema("small_b")
    huge = _schema("huge_tool", extra_bytes=MAX_SINGLE_TOOL_SCHEMA_BYTES + 100)
    result = _capped_mcp_schemas([small_a, small_b, huge], _SETTINGS)
    by_name = {s["name"]: s for s in result}
    assert by_name["small_a"] == small_a
    assert by_name["small_b"] == small_b
    assert by_name["huge_tool"]["input_schema"]["properties"] == {}
    # The degraded form deliberately does NOT repeat "call ToolSearch for the
    # real schema" per tool: that instruction is identical for every tool and
    # is already stated once in the ToolSearch builtin's own description.
    # Measured on a live 238-tool session, repeating it cost 27,608 chars
    # (~6,900 prompt tokens) of byte-identical text on EVERY request, which on
    # an architecture that rejects --cache-reuse is ~197 s of prefill per turn.
    # An empty `properties` is the per-tool signal that the schema was withheld.
    assert "ToolSearch" not in by_name["huge_tool"]["description"]
    assert by_name["huge_tool"]["description"].startswith("does a thing")


def test_no_schemas_returns_empty_list():
    assert _capped_mcp_schemas([], {}) == []


class _FakeRegistry:
    def __init__(self, tool_schemas):
        self.tool_schemas = tool_schemas


def test_tool_search_no_query_asks_for_one():
    result = _tool_search(_FakeRegistry([_schema("a")]), "")
    assert "non-empty query" in result


def test_tool_search_no_registry_reports_nothing_connected():
    assert _tool_search(None, "anything") == "No MCP servers connected."


def test_tool_search_no_match_says_so():
    result = _tool_search(_FakeRegistry([_schema("forge3d_render")]), "totally-unrelated-xyz")
    assert "No tools matched" in result


def test_tool_search_reports_withheld_count_beyond_max_results():
    schemas = [_schema(f"greenboost_thing_{i}") for i in range(10)]
    result = _tool_search(_FakeRegistry(schemas), "greenboost", max_results=3)
    assert "7 more match" in result


def test_tool_search_no_withheld_message_when_everything_shown():
    schemas = [_schema("greenboost_a"), _schema("greenboost_b")]
    result = _tool_search(_FakeRegistry(schemas), "greenboost", max_results=6)
    assert "more match" not in result


def test_tool_search_query_length_is_bounded():
    """An absurdly long query must not blow up the search — it's truncated,
    not rejected outright, so a slightly-too-long real query still works."""
    schemas = [_schema("a")]
    huge_query = "a" * 10_000
    result = _tool_search(_FakeRegistry(schemas), huge_query)
    assert "No tools matched" in result or "a" in result


def test_degraded_schemas_carry_no_repeated_boilerplate():
    """The saving that motivated the change, asserted directly.

    Live 2026-08-18: ten MCP servers, 238 tools. The per-tool
    "Call ToolSearch(query=...) for its full parameter schema" sentence is
    116 chars x 238 = 27,608 chars (~6,900 prompt tokens) of byte-identical
    text on every request. The reference architecture rejects --cache-reuse,
    so every turn re-pays full prompt eval, and prefill was measured at
    30-46 tok/s , roughly 197 s per turn spent re-reading one sentence.
    """
    from greenboost_cli.core.orchestrator import _light_schema

    schemas = [_schema(f"mcp__server__tool_{i}", extra_bytes=3_000)
               for i in range(238)]
    degraded = [_light_schema(s) for s in schemas]

    joined = json.dumps(degraded)
    assert "ToolSearch" not in joined, "per-tool boilerplate is back"
    # Whatever the exact packing, the degraded set must be a small fraction of
    # the full one rather than a near-copy of it.
    assert len(joined) < len(json.dumps(schemas)) // 10


def test_degradation_keeps_what_the_model_needs_to_choose_a_tool():
    """Cheaper must not mean useless: the name and a description survive."""
    from greenboost_cli.core.orchestrator import _light_schema

    out = _light_schema(_schema("mcp__forge3d__optimize_mesh",
                                desc="Optimises a mesh. Long tail of detail."))
    assert out["name"] == "mcp__forge3d__optimize_mesh"
    assert "Optimises a mesh" in out["description"]
    assert out["input_schema"]["properties"] == {}


# ── Count cap: degrading every schema is not always enough ────────────────────

def _many(n: int, servers: int = 10) -> list:
    return [
        _schema(f"mcp__server{i % servers}__tool_{i}", extra_bytes=600)
        for i in range(n)
    ]


def test_names_alone_can_overflow_and_are_capped_by_count():
    """Live 2026-08-18: ten MCP servers, 238 tools.

    Degrading every schema still left ~17k prompt tokens of names, re-sent on
    every turn because this architecture rejects --cache-reuse. Prefill cost is
    super-linear in prompt length (observed: each 2048-token chunk took a
    steady +9.1 s longer than the last), so that tail was worth minutes.
    """
    out = _capped_mcp_schemas(_many(238), _SETTINGS)
    assert len(out) < 238, "count was never capped"
    budget = max(4_000, int(_PINNED_CTX * 0.15 * 4))
    assert len(json.dumps(out)) <= budget * 1.05


def test_every_server_survives_the_cap():
    """A head-slice would let one server eat the budget and hide whole servers.

    Showing fewer tools from each is a far better failure than showing none
    from some, because the model cannot ToolSearch for a server it has no
    reason to believe exists.
    """
    out = _capped_mcp_schemas(_many(238, servers=10), _SETTINGS)
    seen = {n.split("__")[1] for n in (x["name"] for x in out)
            if n.startswith("mcp__")}
    assert len(seen) == 10, f"only {len(seen)} of 10 servers represented"


def test_withheld_tools_are_announced_once():
    """Withholding silently would leave the model unable to know to look."""
    out = _capped_mcp_schemas(_many(238), _SETTINGS)
    notices = [x for x in out if x["name"] == "_mcp_tools_withheld"]
    assert len(notices) == 1
    assert "ToolSearch" in notices[0]["description"]
    assert str(238 - (len(out) - 1)) in notices[0]["description"]


def test_no_notice_when_nothing_was_withheld():
    out = _capped_mcp_schemas(_many(3), _SETTINGS)
    assert not any(x["name"] == "_mcp_tools_withheld" for x in out)


def test_a_small_surface_is_untouched_by_the_count_cap():
    """Never shrink a session that can afford full schemas."""
    schemas = _many(2)
    assert _capped_mcp_schemas(schemas, _SETTINGS) == schemas

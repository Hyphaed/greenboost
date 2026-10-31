"""ToolSearch with hundreds of tools: words, not just one verbatim phrase."""
from __future__ import annotations

import json

from greenboost_cli.core.orchestrator import _tool_search, _rank_tools


class _Reg:
    def __init__(self, tools):
        self.tool_schemas = tools
        self._by = {t["name"]: t for t in tools}

    def _normalize_tool_name(self, n):
        if n in self._by:
            return n
        hits = [k for k in self._by if k.endswith("__" + n)]
        return hits[0] if len(hits) == 1 else None


def _t(name, desc):
    return {"name": name, "description": desc, "input_schema": {"type": "object"}}


TOOLS = [
    _t("mcp__semble__search", "Instant code search for any local or remote git repository."),
    _t("mcp__knowledge-rag__search_knowledge", "Search the knowledge base for notes and documents."),
    _t("mcp__charlotte__charlotte_screenshot", "Take a screenshot of the current browser page."),
    _t("mcp__greenboost-dataflux__dataflux_summary", "Summary of the dataflux telemetry log."),
    _t("mcp__forge-media__generate_image", "Generate an image from a prompt with FLUX."),
]


def _names(result: str) -> list:
    body = result.split("\n\n(")[0]
    return [t["name"] for t in json.loads(body)]


def test_a_multi_word_request_finds_the_tool_even_though_the_phrase_is_nowhere_verbatim():
    r = _tool_search(_Reg(TOOLS), "search code semantically")
    assert _names(r)[0] == "mcp__semble__search"        # was "No tools matched"


def test_server_name_words_count():
    assert _names(_tool_search(_Reg(TOOLS), "take screenshot browser"))[0] == "mcp__charlotte__charlotte_screenshot"
    assert _names(_tool_search(_Reg(TOOLS), "dataflux telemetry"))[0] == "mcp__greenboost-dataflux__dataflux_summary"


def test_the_exact_phrase_still_outranks_loose_word_matches():
    ranked = _rank_tools(TOOLS, "generate_image")
    assert ranked[0][1]["name"] == "mcp__forge-media__generate_image"


def test_select_form_returns_exactly_the_named_tools_in_order():
    r = _tool_search(_Reg(TOOLS), "select:mcp__charlotte__charlotte_screenshot,search_knowledge,nonexistent")
    assert _names(r) == ["mcp__charlotte__charlotte_screenshot", "mcp__knowledge-rag__search_knowledge"]


def test_nothing_matching_still_says_so():
    assert "No tools matched" in _tool_search(_Reg(TOOLS), "quantum teleportation")

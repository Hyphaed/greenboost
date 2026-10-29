# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""greenboost_overview(compact=True): big sections reduced to their keys so the tool fits one MCP result (the full dump is ~27k chars)."""
import json

import gb_mcp


def test_small_sections_pass_big_sections_become_key_lists_and_docs_are_listed():
    out = {"rules": ["r1"], "status": {"vram": 1}, "synapse": {f"k{i}": "v" * 200 for i in range(60)}}
    c = gb_mcp._compact_overview(out)
    assert c["rules"] == ["r1"] and c["status"] == {"vram": 1}
    assert c["synapse"]["_truncated"] is True and c["synapse"]["chars"] > 2500 and len(c["synapse"]["keys"]) == 40
    assert "workflows.md" in " ".join(c["read_first"]) and "compact=False" in c["compact"]
    assert len(json.dumps(c)) < len(json.dumps(out))

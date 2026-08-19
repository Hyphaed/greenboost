#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for checks/check_doc_drift.py (NemoClaw audit round 2, item 6) —
the mechanized gate asserting every dataflux kind / MCP tool gets a
DOCUMENTATION.md mention, with a grandfather allowlist for the pre-existing
backlog recorded 2026-08-05."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "checks"))

import check_doc_drift as cdd

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_clean_on_current_repo():
    findings = cdd.run(_REPO_ROOT)
    assert findings == [], f"unexpected doc-drift violations: {findings}"


def test_catches_new_undocumented_kind(tmp_path, monkeypatch):
    (tmp_path / "DOCUMENTATION.md").write_text("# Docs\nnothing relevant here\n")
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "allowlists").mkdir()

    (tmp_path / "gb_dataflux_kinds.py").write_text(
        "class KindSpec:\n"
        "    def __init__(self, **kw): pass\n"
        "KINDS = {'totally_new_kind': KindSpec()}\n"
    )
    sys.modules.pop("gb_dataflux_kinds", None)  # evict any real/prior-test cached module first —
    sys.path.insert(0, str(tmp_path))            # a cached sys.modules entry short-circuits import
    try:                                          # regardless of sys.path order.
        findings = cdd.run(tmp_path)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("gb_dataflux_kinds", None)

    assert len(findings) == 1
    assert findings[0].severity == "advisory"
    assert "totally_new_kind" in findings[0].message


def test_grandfathered_kind_is_not_flagged(tmp_path):
    (tmp_path / "DOCUMENTATION.md").write_text("# Docs\n")
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "allowlists").mkdir()
    (tmp_path / "checks" / "allowlists" / "doc_drift_kinds.txt").write_text("old_gap_kind\n")
    (tmp_path / "checks" / "allowlists" / "doc_drift_mcp_tools.txt").write_text("")

    (tmp_path / "gb_dataflux_kinds.py").write_text(
        "class KindSpec:\n"
        "    def __init__(self, **kw): pass\n"
        "KINDS = {'old_gap_kind': KindSpec()}\n"
    )
    sys.modules.pop("gb_dataflux_kinds", None)  # evict any real/prior-test cached module first —
    sys.path.insert(0, str(tmp_path))            # a cached sys.modules entry short-circuits import
    try:                                          # regardless of sys.path order.
        findings = cdd.run(tmp_path)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("gb_dataflux_kinds", None)

    assert findings == []


def test_documented_kind_is_not_flagged(tmp_path):
    (tmp_path / "DOCUMENTATION.md").write_text("# Docs\nthe `already_documented` kind ...\n")
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "allowlists").mkdir()

    (tmp_path / "gb_dataflux_kinds.py").write_text(
        "class KindSpec:\n"
        "    def __init__(self, **kw): pass\n"
        "KINDS = {'already_documented': KindSpec()}\n"
    )
    sys.modules.pop("gb_dataflux_kinds", None)  # evict any real/prior-test cached module first —
    sys.path.insert(0, str(tmp_path))            # a cached sys.modules entry short-circuits import
    try:                                          # regardless of sys.path order.
        findings = cdd.run(tmp_path)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("gb_dataflux_kinds", None)

    assert findings == []


def test_catches_new_undocumented_mcp_tool(tmp_path):
    (tmp_path / "DOCUMENTATION.md").write_text("# Docs\n")
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "allowlists").mkdir()
    (tmp_path / "gb_dataflux_kinds.py").write_text("KINDS = {}\n")

    (tmp_path / "gb_fake_mcp.py").write_text(
        "class mcp:\n"
        "    @staticmethod\n"
        "    def tool(f): return f\n"
        "\n"
        "@mcp.tool()\n"
        "def brand_new_tool():\n"
        "    pass\n"
    )
    sys.modules.pop("gb_dataflux_kinds", None)  # evict any real/prior-test cached module first —
    sys.path.insert(0, str(tmp_path))            # a cached sys.modules entry short-circuits import
    try:                                          # regardless of sys.path order.
        findings = cdd.run(tmp_path)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("gb_dataflux_kinds", None)

    assert len(findings) == 1
    assert "brand_new_tool" in findings[0].message

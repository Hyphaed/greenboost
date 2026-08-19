#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""checks/check_doc_drift.py — NemoClaw audit round 2, item 6. Distinct
from check_dataflux_coverage.py (emit-site parity: is a registered kind
actually emitted anywhere) and check_mcp_parity.py (every @mcp.tool has a
docstring): this check is about the project's OWN documentation
requirement that a new dataflux kind or MCP tool gets a row in
DOCUMENTATION.md in the same change that introduces it. Advisory, not
blocking — a documentation gap is a smell to fix, not a build breaker,
matching check_docs.py's precedent for doc-freshness findings.

Grandfathered pre-existing gaps live in checks/allowlists/doc_drift_kinds.txt
and checks/allowlists/doc_drift_mcp_tools.txt (one name per line, '#'
comments) — recorded once so the check doesn't fail on day one over a
pre-existing backlog, while still catching any NEW undocumented kind/tool
going forward. Shrink the allowlists as DOCUMENTATION.md catches up.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

from lib import Finding, load_allowlist


def _dataflux_kinds(repo_root: Path) -> set[str]:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import gb_dataflux_kinds
    return set(gb_dataflux_kinds.KINDS.keys())


def _mcp_tools(repo_root: Path) -> dict[str, str]:
    """{tool_name: repo-relative file} for every @mcp.tool()-decorated
    function in a *_mcp.py file, root-level or under greenboost-cli/."""
    tools: dict[str, str] = {}
    mcp_files = list(repo_root.glob("gb_*_mcp.py")) + list(
        repo_root.glob("greenboost-cli/**/*_mcp.py"))

    for path in mcp_files:
        try:
            tree = ast.parse(path.read_text(errors="ignore"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if (isinstance(target, ast.Attribute) and target.attr == "tool"
                        and isinstance(target.value, ast.Name) and target.value.id == "mcp"):
                    tools[node.name] = str(path.relative_to(repo_root))
                    break
    return tools


def run(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    doc_path = repo_root / "DOCUMENTATION.md"
    if not doc_path.is_file():
        return findings  # nothing to check against — not this check's job to require the file exist
    doc_text = doc_path.read_text(errors="ignore")

    grandfathered_kinds = set(load_allowlist(
        repo_root / "checks" / "allowlists" / "doc_drift_kinds.txt"))
    grandfathered_tools = set(load_allowlist(
        repo_root / "checks" / "allowlists" / "doc_drift_mcp_tools.txt"))

    for kind in sorted(_dataflux_kinds(repo_root)):
        if kind in grandfathered_kinds or kind in doc_text:
            continue
        findings.append(Finding(
            check="doc_drift", severity="advisory", file="gb_dataflux_kinds.py",
            message=f"dataflux kind '{kind}' is registered but not mentioned in DOCUMENTATION.md",
            remediation=f"add '{kind}' to DOCUMENTATION.md's MCP/dataflux surface table"))

    for tool, file_ref in sorted(_mcp_tools(repo_root).items()):
        if tool in grandfathered_tools or tool in doc_text:
            continue
        findings.append(Finding(
            check="doc_drift", severity="advisory", file=file_ref,
            message=f"MCP tool '{tool}' is defined but not mentioned in DOCUMENTATION.md",
            remediation=f"add '{tool}' to DOCUMENTATION.md's MCP surface table"))

    return findings


if __name__ == "__main__":
    fs = run(Path(__file__).resolve().parent.parent)
    for f in fs:
        print(f.format())
    sys.exit(1 if any(f.severity == "blocking" for f in fs) else 0)

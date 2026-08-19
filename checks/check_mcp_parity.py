#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""checks/check_mcp_parity.py — every @mcp.tool has a docstring, and any
module header that enumerates a tool count/list stays in sync with the
actual @mcp.tool defs (golden-principles.md #2's MCP-queryability half).
Blocking on missing docstrings (an undocumented tool is useless to an LLM
client); advisory on header-count drift (a doc smell, not a functional bug)."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from lib import Finding

_MCP_SERVER_FILES = (
    "gb_mcp.py", "gb_dataflux_mcp.py", "gb_cluster_mcp.py", "gb_synapse_mcp.py",
)
_HEADER_COUNT_RE = re.compile(r"Tools\s*\((\d+)\s*[—-]")


def _mcp_tool_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            # matches @mcp.tool() and @mcp.tool
            name = dec.func if isinstance(dec, ast.Call) else dec
            if (isinstance(name, ast.Attribute) and name.attr == "tool"
                    and isinstance(name.value, ast.Name) and name.value.id == "mcp"):
                out.append(node)
                break
    return out


def run(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for fname in _MCP_SERVER_FILES:
        path = repo_root / fname
        if not path.is_file():
            findings.append(Finding(
                check="mcp_parity", severity="blocking", file=fname,
                message="expected MCP server file is missing",
                remediation="restore the file or update _MCP_SERVER_FILES in "
                            "checks/check_mcp_parity.py if it was intentionally renamed/removed"))
            continue
        text = path.read_text()
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            findings.append(Finding(
                check="mcp_parity", severity="blocking", file=fname, line=e.lineno or 0,
                message=f"file does not parse: {e}", remediation="fix the syntax error"))
            continue

        tools = _mcp_tool_functions(tree)
        for fn in tools:
            doc = ast.get_docstring(fn)
            if not doc or not doc.strip():
                findings.append(Finding(
                    check="mcp_parity", severity="blocking", file=fname, line=fn.lineno,
                    message=f"@mcp.tool `{fn.name}` has no docstring",
                    remediation="add a one-paragraph docstring: what it returns, whether it's "
                                "gated/read-only, and its canonical server if mirrored elsewhere "
                                "(see gb_mcp_common.py's docstring convention)"))

        m = _HEADER_COUNT_RE.search(text)
        if m:
            claimed = int(m.group(1))
            actual = len(tools)
            if claimed != actual:
                findings.append(Finding(
                    check="mcp_parity", severity="advisory", file=fname, line=1,
                    message=f"module header claims {claimed} tools, actually {actual}",
                    remediation=f"update the 'Tools ({claimed} — ...)' header count/list to match "
                                f"the real @mcp.tool defs ({actual})"))
    return findings


if __name__ == "__main__":
    import sys
    fs = run(Path(__file__).resolve().parent.parent)
    for f in fs:
        print(f.format(llm="--llm" in sys.argv))
    print(f"\n{len(fs)} finding(s)")
    sys.exit(1 if any(f.severity == "blocking" for f in fs) else 0)

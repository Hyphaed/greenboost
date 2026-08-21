#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""checks/check_mcp_readonly.py — AE-6.

Every `@mcp.tool` on a GreenBoost MCP server must make a deliberate decision
about `readOnlyHint`: either declare it (the tool only answers questions and a
client may overlap it with others) or be listed here as knowingly undeclared
(it actuates, serves, dispatches, writes policy, or shells out).

Without this check the declaration decays silently: a tool added next month
gets no annotation, stays serial forever, and nobody notices , which is safe
but slowly gives back the concurrency the annotations bought. The opposite
failure is worse and this check cannot catch it alone, which is why the
undeclared list is explicit rather than inferred: a human has to write a tool's
name down to say "this one mutates".

ADVISORY, not blocking. A missing annotation costs wall time, never
correctness , the client's default is serial.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from lib import Finding

_SERVERS = ("gb_mcp.py", "gb_dataflux_mcp.py", "gb_cluster_mcp.py",
            "gb_synapse_mcp.py")

#: Tools that deliberately carry NO read-only hint, with the reason. Adding a
#: name here is a claim that the tool changes something.
_KNOWINGLY_UNDECLARED: dict = {
    "optimize_inference": "applies an inference plan",
    "set_quant_policy": "writes quant policy",
    "tier_actuate": "moves tiers",
    "tuner_tick": "advances the control loop's state and can actuate levers",
    "run_under_greenboost": "executes a command under the shim",
    "reclaim_run": "reclaims memory for real",
    "support_bundle": "writes a tarball",
    "a2a_gateway": "dispatches A2A verbs",
    "cluster_sync_dataflux": "writes events onto this node",
    "cluster_ensure_feeder_ready": "provisions a feeder",
    "cluster_dispatch": "runs work on feeders",
    "synapse_serve": "starts an engine",
    "synapse_stop": "stops an engine",
    "synapse_pause": "checkpoints and stops an engine",
    "synapse_resume": "restarts an engine",
    "serve_and_repoint": "starts an engine and repoints clients",
    "cli_run": "shells out",
    "cli_prompt": "runs a generation",
    "quality_gate": "runs a generation",
}


def _tools(path: Path):
    """Yield (function_name, decorator_source) for every @mcp.tool."""
    text = path.read_text(errors="ignore")
    tree = ast.parse(text)
    lines = text.split("\n")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            seg = ast.get_source_segment(text, dec) or ""
            if "mcp.tool" in seg:
                yield node.name, seg


def run(repo_root: Path) -> list:
    findings: list = []
    for name in _SERVERS:
        path = repo_root / name
        if not path.is_file():
            continue
        for fn, dec in _tools(path):
            declared = "annotations" in dec
            if declared or fn in _KNOWINGLY_UNDECLARED:
                continue
            findings.append(Finding(
                check="mcp_readonly", severity="advisory", file=name,
                message=f"MCP tool '{fn}' declares no readOnlyHint and is not "
                        f"listed as knowingly undeclared",
                remediation=(
                    f"if '{fn}' only answers questions, decorate it "
                    f"@mcp.tool(annotations=_READ_ONLY); if it changes "
                    f"anything, add it to _KNOWINGLY_UNDECLARED in "
                    f"checks/check_mcp_readonly.py with the reason")))
    return findings


if __name__ == "__main__":
    for f in run(Path(__file__).resolve().parent.parent):
        print(f)

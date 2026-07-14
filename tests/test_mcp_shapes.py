#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
A5 — MCP tool dedupe. Asserts every tool mirrored across multiple GreenBoost
MCP servers delegates to the SAME gb_mcp_common implementation, so the
top-level keys returned are identical no matter which server answers (fixes
F2: greenboost_status previously had two divergent shapes).

No CUDA, no /dev/greenboost, no running daemon needed — @mcp.tool() leaves
the decorated function directly callable (FastMCP registers it but returns
the original callable), and every gb_mcp_common function tolerates a
GreenBoost-less environment (best-effort, returns {"error": ...} rather than
raising).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gb_mcp
import gb_dataflux_mcp
import gb_cluster_mcp
import gb_synapse_mcp
import gb_mcp_common


def _keys(d) -> set:
    assert isinstance(d, dict), f"expected dict, got {type(d)}: {d!r}"
    return set(d.keys())


def test_greenboost_status_identical_shape_on_both_servers():
    a = _keys(gb_mcp.greenboost_status())
    b = _keys(gb_dataflux_mcp.greenboost_status())
    assert a == b
    assert "kv_prefetch" in a   # the field that used to be MISSING on gb_mcp's copy (F2 bug)


def test_greenboost_capabilities_identical_shape_on_both_servers():
    assert _keys(gb_mcp.greenboost_capabilities()) == _keys(gb_dataflux_mcp.greenboost_capabilities())


def test_greenboost_pilot_identical_shape_on_both_servers():
    assert _keys(gb_mcp.greenboost_pilot()) == _keys(gb_dataflux_mcp.greenboost_pilot())


def test_synapse_status_identical_shape_on_all_three_servers():
    a = _keys(gb_mcp.synapse_status())
    b = _keys(gb_dataflux_mcp.synapse_status())
    c = _keys(gb_synapse_mcp.synapse_status())
    assert a == b == c


def test_dataflux_summary_identical_shape_on_both_servers():
    assert _keys(gb_mcp.dataflux_summary()) == _keys(gb_dataflux_mcp.dataflux_summary())


def test_cluster_status_identical_shape_on_both_servers():
    assert _keys(gb_mcp.cluster_status(probe=False)) == _keys(gb_cluster_mcp.cluster_status(probe=False))


def test_every_mirror_delegates_to_gb_mcp_common_not_a_local_reimplementation():
    """Structural check: each mirrored wrapper's source imports gb_mcp_common
    and calls into it, rather than re-implementing the logic locally (the
    exact drift this module exists to prevent)."""
    import inspect
    mirrors = [
        (gb_mcp, "greenboost_status"), (gb_dataflux_mcp, "greenboost_status"),
        (gb_mcp, "greenboost_capabilities"), (gb_dataflux_mcp, "greenboost_capabilities"),
        (gb_mcp, "greenboost_pilot"), (gb_dataflux_mcp, "greenboost_pilot"),
        (gb_mcp, "synapse_status"), (gb_dataflux_mcp, "synapse_status"), (gb_synapse_mcp, "synapse_status"),
        (gb_mcp, "dataflux_summary"), (gb_dataflux_mcp, "dataflux_summary"),
        (gb_mcp, "cluster_status"), (gb_cluster_mcp, "cluster_status"),
    ]
    for module, name in mirrors:
        src = inspect.getsource(getattr(module, name))
        assert "gb_mcp_common" in src, f"{module.__name__}.{name} no longer delegates to gb_mcp_common"


def test_gb_mcp_common_has_no_mcp_tool_registrations():
    """gb_mcp_common is a plain library module — it must never itself become
    a 6th MCP server (the cross-cutting rule's 'never a 6th server without
    cause')."""
    assert not hasattr(gb_mcp_common, "mcp")

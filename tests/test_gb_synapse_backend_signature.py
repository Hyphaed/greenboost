#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Regression test for the 2026-08-10 incident: gb_synapse.serve() calls
backend.serve() with mtp_draft_n/spec_draft_p_min/slot_prompt_similarity —
added to the caller and to gb_synapse_mcp.py's synapse_serve tool by the
2026-08-05 speed rounds, but never added to any EngineBackend subclass.
Every greenboost-cli auto-start and every synapse_serve MCP call raised
TypeError as a result.

The existing mock-based tests (test_gb_synapse_recipe_probe.py,
test_gb_synapse_mcp_recipe.py) stub backend.serve with a **kwargs capture,
so nothing ever bound the real signature — this file binds the REAL classes
so a future param added to the caller without a matching backend param fails
here first, not in a live serve.
"""
import sys
import inspect
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import gb_synapse_backends as gsb


# The full set of kwargs gb_synapse.serve() passes to backend.serve() today
# (both the primary dispatch and the host-only retry call sites) — every
# backend must accept all of these, even the ones it ignores, per
# EngineBackend.serve()'s own "backends without this concept ignore it"
# contract.
_SERVE_KWARGS = dict(
    ctx=0, use_cluster=True, n_slots=-1, extra_args="",
    cuda_graph=None, cache_ram=None,
    mtp_draft_n=None, spec_draft_p_min=None, slot_prompt_similarity=None,
    n_gpu_layers_override=None, kv_type_override=None, n_cpu_moe_override=None,
)


@pytest.mark.parametrize("cls", [gsb.EngineBackend, *gsb._ADAPTERS.values()])
def test_serve_signature_accepts_every_caller_kwarg(cls):
    """Binds the real, unbound serve() against the exact kwargs
    gb_synapse.py's serve() passes — a TypeError here is the exact class of
    bug that broke every greenboost-cli auto-start on 2026-08-10."""
    sig = inspect.signature(cls.serve)
    sig.bind(object(), object(), 11369, **_SERVE_KWARGS)


def test_llama_cpp_backend_is_covered_by_adapter_registry():
    """Guards the parametrize list itself — if LlamaCppBackend ever stopped
    being a registered adapter, the test above would silently stop covering
    it."""
    assert gsb.LlamaCppBackend in gsb._ADAPTERS.values()

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for checks/check_dataflux_coverage.py's kind-literal AST scan , the
mechanized parity rule enforcement (P5). Exercises the 3 recognized emit-
call shapes, the variable-mediated dict pattern that's the dominant real
pattern in this codebase, and the false-positive case (an unrelated "kind"
dict key in a file with no emit call at all) that a naive scan would catch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "checks"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import check_dataflux_coverage as cdc
import gb_dataflux_kinds


def test_inline_dict_literal_shape(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text(
        "import gb_dataflux\n"
        "def f():\n"
        "    gb_dataflux.emit({'kind': 'quantize', 'status': 'ok'})\n"
    )
    assert cdc._find_kind_literals_in_file(f) == {"quantize"}


def test_variable_mediated_dict_shape(tmp_path):
    """The DOMINANT real pattern: build the dict as a local variable, THEN
    pass the variable to emit() , not inline."""
    f = tmp_path / "mod.py"
    f.write_text(
        "import gb_dataflux\n"
        "def f():\n"
        "    ev = {'kind': 'snapshot', 'node': 'host'}\n"
        "    gb_dataflux.emit(ev)\n"
    )
    assert cdc._find_kind_literals_in_file(f) == {"snapshot"}


def test_keyword_argument_shape(tmp_path):
    """gb_actuation._emit(..., kind="agent_run")'s real shape: kind= passed
    explicitly at a CALL site (not a function-signature default value ,
    the scanner deliberately doesn't try to infer literals from parameter
    defaults, only from what a caller actually passes)."""
    f = tmp_path / "mod.py"
    f.write_text(
        "def _emit(verb, gated, *, kind='actuation', **fields):\n"
        "    import gb_dataflux\n"
        "    gb_dataflux.emit({'kind': kind})\n"
        "def run():\n"
        "    _emit('x', True, kind='agent_run')\n"
    )
    found = cdc._find_kind_literals_in_file(f)
    assert "agent_run" in found


def test_df_emit_positional_fourth_arg_shape(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text(
        "def _df_emit(run_id, node, label, kind, items, duration_s=0.0, **extra):\n"
        "    import gb_dataflux\n"
        "    gb_dataflux.emit({'kind': kind})\n"
        "def run():\n"
        "    _df_emit('r1', 'host', 'my_label', 'chunk_remote', [], 1.0)\n"
    )
    found = cdc._find_kind_literals_in_file(f)
    assert "chunk_remote" in found


def test_unrelated_kind_dict_key_in_file_with_no_emit_call_is_ignored(tmp_path):
    """The false-positive case this scan was specifically hardened against:
    a file that uses "kind" as an unrelated dict key (e.g. classifying work
    items as cached/uncached) but never calls anything emit-shaped must
    produce NO findings , exactly the vendored gLLM engine's
    model_runner.py situation."""
    f = tmp_path / "mod.py"
    f.write_text(
        "def f(works):\n"
        "    works.append({'kind': 'cached', 'seq': 1})\n"
        "    works.append({'kind': 'uncached', 'seq': 2})\n"
    )
    assert cdc._find_kind_literals_in_file(f) == set()


def test_same_kind_key_ignored_when_no_emit_call_but_included_when_one_exists(tmp_path):
    """The SAME unrelated 'kind' dict key, but in a file that ALSO has a
    real emit call elsewhere , must now be picked up (per-file scoping,
    not per-call), matching this repo's real async_llm_engine.py (real
    synapse_engine_error emit) vs model_runner.py (no emit call at all)."""
    f = tmp_path / "mod.py"
    f.write_text(
        "def unrelated(works):\n"
        "    works.append({'kind': 'cached'})\n"
        "def real_emit():\n"
        "    import gb_dataflux\n"
        "    gb_dataflux.emit({'kind': 'synapse_engine_error'})\n"
    )
    found = cdc._find_kind_literals_in_file(f)
    assert "synapse_engine_error" in found
    assert "cached" in found  # coarser per-file scope accepts this tradeoff


def test_file_with_no_emit_call_anywhere_returns_empty(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("def f():\n    return {'kind': 'anything'}\n")
    assert cdc._find_kind_literals_in_file(f) == set()


def test_unparseable_file_returns_empty_not_raises(tmp_path):
    f = tmp_path / "broken.py"
    f.write_text("def f(:\n    this is not python")
    assert cdc._find_kind_literals_in_file(f) == set()


def test_real_repo_scan_has_zero_blocking_findings():
    """End-to-end: running the actual check against the real repo tree
    must currently be clean , this is the regression guard for the whole
    kind-literal parity mechanism, not just its helper functions."""
    repo_root = Path(__file__).resolve().parent.parent
    findings = cdc.run(repo_root)
    blocking = [f for f in findings if f.severity == "blocking"]
    assert blocking == [], f"unexpected blocking dataflux_coverage findings: {blocking}"


def test_every_registered_kind_is_a_valid_identifier():
    for kind in gb_dataflux_kinds.KINDS:
        assert cdc._KIND_RE.match(kind), f"registry kind {kind!r} is not a valid bare identifier"

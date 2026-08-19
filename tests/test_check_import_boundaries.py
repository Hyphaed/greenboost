#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for checks/check_import_boundaries.py (NemoClaw audit round 2,
item 6) — the mechanized gate for module-level import-direction rules
that used to be enforced only by a human reading a comment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "checks"))

import check_import_boundaries as cib

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_clean_on_current_repo():
    findings = cib.run(_REPO_ROOT)
    assert findings == [], f"unexpected import-boundary violations: {findings}"


def test_catches_forbidden_module_level_import(tmp_path):
    (tmp_path / "gb_synapse_backends.py").write_text(
        "import gb_synapse\n"
        "\n"
        "def f():\n"
        "    pass\n"
    )
    findings = cib.run(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "blocking"
    assert findings[0].line == 1
    assert "gb_synapse" in findings[0].message


def test_lazy_import_inside_function_is_not_a_violation(tmp_path):
    (tmp_path / "gb_synapse_backends.py").write_text(
        "def f():\n"
        "    import gb_synapse\n"
        "    return gb_synapse\n"
    )
    assert cib.run(tmp_path) == []


def test_import_inside_top_level_try_still_counts(tmp_path):
    (tmp_path / "gb_kernel_backends.py").write_text(
        "try:\n"
        "    import torch\n"
        "except ImportError:\n"
        "    torch = None\n"
    )
    findings = cib.run(tmp_path)
    assert len(findings) == 1
    assert "torch" in findings[0].message


def test_unrelated_module_is_unaffected(tmp_path):
    (tmp_path / "gb_something_else.py").write_text("import gb_synapse\n")
    assert cib.run(tmp_path) == []

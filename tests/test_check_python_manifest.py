#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""The install-manifest check itself , it has to fail when a module goes missing.

A check that cannot fail is decoration. This drives it against synthetic trees
rather than the repo, so it keeps testing the LOGIC after the real manifest
changes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "checks"))

import check_python_manifest as C

_SETUP = """
cmd_install_python_files() {
    local _py_files=(
        gb_init.py
        # a comment naming gb_ghost.py must not count as an entry
        gb_quant.py gb_telemetry.py
    )
    for _f in "${_py_files[@]}"; do :; done
}
"""


def _tree(tmp_path, modules, setup=_SETUP):
    (tmp_path / "greenboost_setup.sh").write_text(setup)
    for m in modules:
        (tmp_path / m).write_text("# stub\n")
    return tmp_path


def test_a_module_missing_from_the_manifest_is_blocking(tmp_path):
    root = _tree(tmp_path, ["gb_init.py", "gb_quant.py", "gb_telemetry.py",
                            "gb_orphan.py"])
    f = C.run(root)
    assert len(f) == 1
    assert f[0].file == "gb_orphan.py"
    assert f[0].severity == "blocking"
    assert "ABSENT on a freshly installed box" in f[0].message


def test_a_fully_covered_tree_is_clean(tmp_path):
    root = _tree(tmp_path, ["gb_init.py", "gb_quant.py", "gb_telemetry.py"])
    assert C.run(root) == []


def test_a_name_mentioned_only_in_a_comment_does_not_count(tmp_path):
    """The parser strips comments; a module named in prose is not installed."""
    root = _tree(tmp_path, ["gb_init.py", "gb_quant.py", "gb_telemetry.py",
                            "gb_ghost.py"])
    assert [x.file for x in C.run(root)] == ["gb_ghost.py"]


def test_an_exempt_module_is_allowed(tmp_path, monkeypatch):
    monkeypatch.setitem(C._EXEMPT, "gb_orphan.py", "installed by its own step")
    root = _tree(tmp_path, ["gb_init.py", "gb_quant.py", "gb_telemetry.py",
                            "gb_orphan.py"])
    assert C.run(root) == []


def test_an_unparseable_array_is_reported_rather_than_passing(tmp_path):
    """If the bash shape changes, the check must say it cannot see the list —
    silently finding zero modules would read as 'everything is installed'."""
    root = _tree(tmp_path, ["gb_init.py"], setup="cmd_install_python_files() { :; }\n")
    f = C.run(root)
    assert len(f) == 1 and "could not parse" in f[0].message


def test_the_real_repo_passes():
    """The manifest the repo actually ships must be complete."""
    assert C.run(Path(__file__).resolve().parent.parent) == []

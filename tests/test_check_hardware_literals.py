"""The hardcoded-literal gate must flag literals, not prose.

Two defects found together on 2026-08-22:

* It reported 11 findings on a clean repo and **every one was prose** — measured
  figures inside docstrings and KindSpec `doc=` strings, a shell `info "...RTX
  5070 12GB..."` preset message, and the usage text of `t3-memory <size>`. The
  skip only caught lines that BEGIN with a comment marker, so a comment
  continuation or any string literal looked like a hardcoded constant.
* **Nothing in the suite ran it.** It exited non-zero on every invocation and no
  test noticed, which is why the false positives survived.

A blocking gate that is 100% false positives is a gate people learn to skip,
and skipping it is how a real hardcoded constant gets through. This file exists
so both halves stay fixed: the detection is pinned below, and the repo-wide run
is asserted clean.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECK = REPO / "checks" / "check_hardware_literals.py"


@pytest.fixture(scope="module")
def mod():
    sys.path.insert(0, str(REPO / "checks"))
    spec = importlib.util.spec_from_file_location("chl", CHECK)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _flags(mod, line, suffix=".py"):
    stripped = mod._strip_non_code(Path("x" + suffix), [line])[0]
    return bool(mod._SIZE_LITERAL_RE.search(stripped))


@pytest.mark.parametrize("line,suffix", [
    ("cap_mb = 12227MB", ".py"),
    ("    reserve = 512MB", ".py"),
    ("GB_NVME_POOL=114GB", ".sh"),
])
def test_real_literals_are_still_flagged(mod, line, suffix):
    assert _flags(mod, line, suffix), f"{line!r} is a hardcoded literal"


@pytest.mark.parametrize("line,suffix", [
    ('CAP = "12227MB"', ".py"),                      # string literal
    ('CAP = 12227  # 12227MB measured', ".py"),      # trailing comment
    ('DOC = """total 12227MB free"""', ".py"),       # docstring
    ('info "RTX 5070 12GB detected"', ".sh"),        # shell string
    ('    "11,770 tokens is 356 MB, saves in 94 ms. "', ".py"),  # KindSpec doc=
])
def test_prose_is_not_a_constant(mod, line, suffix):
    assert not _flags(mod, line, suffix), f"{line!r} is prose, not a constant"


def test_strip_preserves_line_numbering(mod):
    """Findings report line numbers, so blanking must not shift lines."""
    src = ['a = 1', 'DOC = """', 'total 12227MB', '"""', 'b = 2']
    out = mod._strip_non_code(Path("x.py"), src)
    assert len(out) == len(src)
    assert out[0] == 'a = 1' and out[4] == 'b = 2'


def test_unparseable_python_falls_back_rather_than_skipping(mod):
    """A file mid-edit must not silently stop being checked."""
    out = mod._strip_non_code(Path("x.py"), ["def f(:", "cap = 512MB"])
    assert mod._SIZE_LITERAL_RE.search(out[1]), "fallback must still see the literal"


def test_the_repo_is_clean(mod):
    """The gate itself, on the real tree. It was exiting 1 with nobody looking."""
    findings = mod.run(REPO)
    assert findings == [], f"hardware-literal findings: {[(f.file, f.line) for f in findings]}"

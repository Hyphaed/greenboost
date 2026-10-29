"""Importing GreenBoost must print nothing to stdout.

stdout is DATA for the processes that import these modules:

* the MCP servers speak JSON-RPC over it — a banner mid-stream is a protocol
  violation, and gb_semantics had to add a defensive stdout guard for exactly
  this (crosswalk P5-11);
* any bench or tool that prints a table has its output corrupted. On
  2026-08-22 three `[gb_telemetry]` lines and one `[gb_init]` banner landed in
  the middle of tests/bench/bench_fp8_backends.py's results table.

Guarding the consumer was the right emergency fix and the wrong permanent one.
The producer is gb_init / gb_telemetry, and a diagnostic has no business on
stdout. These tests pin the producer side so the guard stops being load-bearing.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _import_and_capture(module: str):
    r = subprocess.run(
        [sys.executable, "-c", f"import {module}; print('SENTINEL')"],
        capture_output=True, text=True, cwd=str(REPO), timeout=180)
    return r


@pytest.mark.parametrize("module", ["gb_init", "gb_telemetry", "gb_semantics", "gb_dataflux"])
def test_import_writes_nothing_to_stdout(module):
    r = _import_and_capture(module)
    assert r.returncode == 0, r.stderr[-400:]
    extra = [ln for ln in r.stdout.splitlines() if ln.strip() != "SENTINEL"]
    assert not extra, (
        f"importing {module} wrote to stdout, which is protocol/data for its "
        f"consumers: {extra[:4]}")


def test_diagnostics_were_moved_not_deleted():
    """Silencing them would be the wrong fix — they must move, not vanish.

    Asserted at the SOURCE rather than by running an import: whether a
    diagnostic fires depends on the machine (DCGM present, GPU visible,
    telemetry already initialised), so requiring one at runtime would be
    asserting an environment instead of a contract.
    """
    for name in ("gb_init.py", "gb_telemetry.py"):
        src = (REPO / name).read_text()
        assert src.count("file=sys.stderr") >= 4, (
            f"{name} should still emit its diagnostics, on stderr")


def test_no_bare_flush_only_print_in_the_noisy_modules():
    """A `print(..., flush=True)` with no `file=` is stdout by definition.

    Catches the pattern at the source, so a new diagnostic cannot reintroduce
    it without this test noticing.
    """
    import re
    offenders = []
    for name in ("gb_init.py", "gb_telemetry.py"):
        src = (REPO / name).read_text()
        # every print( ... ) call, including multi-line
        for m in re.finditer(r"\bprint\(", src):
            depth, i = 0, m.end() - 1
            while i < len(src):
                if src[i] == "(":
                    depth += 1
                elif src[i] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            call = src[m.start():i + 1]
            if "file=" not in call:
                line = src[:m.start()].count("\n") + 1
                offenders.append(f"{name}:{line}")
    assert not offenders, f"print() without file= (defaults to stdout): {offenders}"

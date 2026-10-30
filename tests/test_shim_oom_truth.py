"""The shim states physical VRAM when every tier refuses an allocation (2026-10-09: torch printed a 126 GiB capacity next to a failed 20 MiB allocation on a card another process had filled).
CPU-only: the source must define the reporter, call it on both terminal OOM paths, and still pass a syntax check."""
import shutil
import subprocess
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "greenboost_cuda_shim.c"


def test_reporter_is_defined_once_and_called_on_both_terminal_oom_paths():
    s = SRC.read_text()
    assert s.count("static void gb_report_oom_truth(") == 1
    assert 'gb_report_oom_truth("smart_alloc"' in s and 'gb_report_oom_truth("cuMemCreate' in s
    assert "PHYSICAL VRAM" in s


@pytest.mark.skipif(not shutil.which("gcc"), reason="no gcc")
def test_shim_still_passes_a_syntax_check():
    r = subprocess.run(["gcc", "-std=gnu11", "-fPIC", "-fsyntax-only", "-I", str(SRC.parent), str(SRC)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]

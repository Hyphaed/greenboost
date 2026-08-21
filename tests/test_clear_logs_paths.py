"""Regression tests for `greenboost clear logs` (greenboost_setup.sh).

Two defects are pinned here, both found on 2026-08-21:

1. The verb reported clearing "all GreenBoost logs" while only touching
   dmesg, the journal and /var/log/greenboost/ , the dataflux log and the
   Proton/Vulkan logs, the two largest stores, were never in its path list.

2. The removal loop was `rm -f "$p" && (( _n++ ))`. Post-increment evaluates
   to the value BEFORE the bump, so the first success returns 0, which is a
   non-zero exit status, which `set -e` treats as a failed command. The loop
   aborted after removing exactly one file and still reported success. The
   17:26 run that day cleared dataflux.jsonl and left the other 17 files.

Both are shell, so these drive the real functions out of greenboost_setup.sh
in a bash subprocess rather than reimplementing them.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SETUP = _REPO_ROOT / "greenboost_setup.sh"


def _extract(fn: str) -> str:
    """Pull one shell function verbatim out of greenboost_setup.sh."""
    src = _SETUP.read_text(encoding="utf-8")
    m = re.search(rf"^{re.escape(fn)}\(\) \{{$.*?^\}}$", src, re.M | re.S)
    assert m, f"{fn}() not found in greenboost_setup.sh"
    return m.group(0)


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_log_paths_cover_dataflux_and_proton_logs(tmp_path):
    """The path list must include the two stores the old verb silently skipped.

    Scoped to tmp_path via the $HOME argument. The system-wide entries
    (/var/log/greenboost, /tmp) are deliberately NOT asserted on , a test that
    globs real paths can delete real files, which is exactly what happened
    while writing this one.
    """
    home = tmp_path / "home"
    gb = home / ".local" / "share" / "greenboost"
    (gb / "proton-logs").mkdir(parents=True)
    (gb / "dataflux.jsonl").write_text("{}\n")
    (gb / "dataflux.jsonl.pre-123.gz").write_bytes(b"\x1f\x8b")
    (gb / "status.log").write_text("")
    (gb / "sessions.jsonl").write_text("{}\n")
    for name in ("gl-layer.log", "steam-123.log", "vulkan-layer.log"):
        (gb / "proton-logs" / name).write_text("x")

    r = _run(f'{_extract("_gb_log_paths")}\n_gb_log_paths "{home}"')
    assert r.returncode == 0, r.stderr
    found = {line for line in r.stdout.splitlines() if line.startswith(str(home))}

    expected = {
        str(gb / "dataflux.jsonl"),
        str(gb / "dataflux.jsonl.pre-123.gz"),
        str(gb / "status.log"),
        str(gb / "sessions.jsonl"),
        str(gb / "proton-logs" / "gl-layer.log"),
        str(gb / "proton-logs" / "steam-123.log"),
        str(gb / "proton-logs" / "vulkan-layer.log"),
    }
    assert expected <= found, f"missing from the clear list: {expected - found}"


def test_log_paths_empty_when_nothing_exists(tmp_path):
    """No log files must produce no paths, not a glob echoed back literally."""
    r = _run(f'{_extract("_gb_log_paths")}\n_gb_log_paths "{tmp_path}"')
    assert r.returncode == 0, r.stderr
    assert not [ln for ln in r.stdout.splitlines() if str(tmp_path) in ln]


def test_removal_loop_survives_set_e(tmp_path):
    """The counted-removal loop must not abort on its first success.

    This is the `(( _n++ ))` bug: under `set -e` the loop removed one file and
    returned. Pinned by counting survivors, not by reading the shell.
    """
    for i in range(5):
        (tmp_path / f"f{i}.log").write_text("x")

    loop = _extract("cmd_clear_logs")
    m = re.search(r"local p _n=0\n(.*?\n    done)\n", loop, re.S)
    assert m, "the counted removal loop is no longer recognisable in cmd_clear_logs"

    script = f"""
    set -euo pipefail
    mapfile -t _paths < <(printf '%s\\n' {tmp_path}/f*.log)
    local() {{ :; }}
    _n=0
    {m.group(1)}
    echo "removed=$_n"
    """
    r = _run(script)
    assert r.returncode == 0, f"loop aborted under set -e: {r.stderr}"
    assert "removed=5" in r.stdout, r.stdout
    assert not list(tmp_path.glob("*.log")), "files survived the removal loop"


def test_measurement_caches_are_never_in_the_clear_list():
    """kv_measurements.json and friends are measurements, not logs.

    Deleting them costs real throughput (VRAM fill 67-73% -> 85.1%, decode
    2.6-4.3 -> 5.27 tok/s came from that cache), so they must never appear in
    the path collector, whatever else is added to it later.
    """
    body = _extract("_gb_log_paths")
    for forbidden in (
        "kv_measurements",
        "shim_probe",
        "dxvk-gplasync",
        "proton-cache",
        "greenboost-cli",
    ):
        assert forbidden not in body, f"{forbidden} must not be cleared as a log"

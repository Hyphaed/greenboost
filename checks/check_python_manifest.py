#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""checks/check_python_manifest.py — every gb_*.py the repo ships must be
installed, or be explicitly exempt.

This mechanizes a failure that has recurred at least twice, both times found
only in production:

  - 2026-07-14: `gb_mcp_common.py` existed in the repo but was missing from
    `cmd_install_python_files`, so four MCP tools died post-install with
    "No module named 'gb_mcp_common'" despite the file being right there.
  - NemoClaw audit round 2: `gb_ports.py`, `gb_advisories.py` and
    `gb_failures.py` were never added either , confirmed live by an
    `import gb_readiness` against a freshly Full-Installed box raising
    ModuleNotFoundError.
  - 2026-08-20: `gb_bench_turn.py`, `gb_bench_kv.py` and `gb_bench_spec.py`
    were all absent, which would have left the box unable to certify a quant
    config , the exact evidence CLAUDE.md's "never below fp8 without gate
    evidence" rule depends on.

`check_installer_parity.py` does NOT cover this: it checks install↔purge
SYMMETRY (does everything installed also get removed), which is a different
question from "does everything get installed at all". A module missing from
both paths is perfectly symmetric and completely broken.

The failure mode is nasty precisely because it is invisible in the repo and in
every test: the file exists, imports fine, and passes CI. It is only absent on
the machines that matter.

BLOCKING: a module that does not reach the install is a runtime
ModuleNotFoundError for whoever uses the feature it backs.
"""
from __future__ import annotations

import re
from pathlib import Path

from lib import Finding

_SETUP = "greenboost_setup.sh"

#: Repo-root gb_*.py files that are deliberately NOT installed, with the reason.
#: Adding a name here is a claim that nothing on an installed box imports it.
#: Verified 2026-08-20 by checking both `greenboost_setup.sh` mentions AND the
#: live install tree , not by assuming.
_EXEMPT: dict = {
    "gb_supervisor.py": "installed by its own systemd/daemon step (10 mentions "
                        "in greenboost_setup.sh; present in the install tree)",
    "gb_vitals_helper.py": "installed by its own step (7 mentions; present in "
                           "the install tree)",
    "gb_shim_probe.py": "dev/diagnostic probe, run from a checkout",
}


def _installed_names(text: str) -> set:
    """Names inside cmd_install_python_files's `_py_files=( ... )` array."""
    m = re.search(r"_py_files=\(\s*(.*?)\n\s*\)", text, re.S)
    if not m:
        return set()
    body = m.group(1)
    body = re.sub(r"#[^\n]*", "", body)          # strip comments
    return set(re.findall(r"\bgb_[a-z0-9_]+\.py\b", body))


def run(repo_root: Path) -> list:
    setup = repo_root / _SETUP
    if not setup.is_file():
        return []
    text = setup.read_text(errors="ignore")
    installed = _installed_names(text)
    if not installed:
        return [Finding(
            check="python_manifest", severity="blocking", file=_SETUP,
            message="could not parse cmd_install_python_files's _py_files array",
            remediation="the array shape changed; update _installed_names() to match")]

    findings = []
    for path in sorted(repo_root.glob("gb_*.py")):
        name = path.name
        if name in installed or name in _EXEMPT:
            continue
        findings.append(Finding(
            check="python_manifest", severity="blocking", file=name,
            message=(f"{name} ships in the repo but is not in "
                     f"{_SETUP}'s _py_files list , it will be ABSENT on a "
                     f"freshly installed box"),
            remediation=(f"add '{name}' to cmd_install_python_files's _py_files "
                         f"array, or add it to _EXEMPT in "
                         f"checks/check_python_manifest.py with the reason "
                         f"nothing on an installed box imports it")))
    return findings


if __name__ == "__main__":
    for f in run(Path(__file__).resolve().parent.parent):
        print(f)

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""checks/check_installer_parity.py — golden-principles.md #3: every
artifact Full Install creates must have a symmetric removal in the
uninstall/purge path. `greenboost_setup.sh` is ~13k lines of bash; fully
parsing its control flow to auto-diff install-vs-purge is out of scope for
a fast mechanical check, so this is a REVIEWED MANIFEST
(checks/allowlists/install_manifest.txt): each line names one artifact plus
a regex that must match somewhere in the install path and a regex that must
match somewhere in the purge path (or an explicit note that a generic glob
already covers it). Adding a new installed artifact means adding a manifest
line in the SAME change — that's the parity discipline, mechanized."""
from __future__ import annotations

import re
from pathlib import Path

from lib import Finding

_MANIFEST = Path(__file__).parent / "allowlists" / "install_manifest.txt"
_SETUP_SCRIPT = "greenboost_setup.sh"


def _load_manifest(path: Path) -> "list[tuple[str, str, str]]":
    """Each non-comment line: 'artifact_name | install_regex | purge_regex'."""
    out = []
    if not path.is_file():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            continue
        out.append((parts[0], parts[1], parts[2]))
    return out


def run(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    manifest = _load_manifest(_MANIFEST)
    setup_path = repo_root / _SETUP_SCRIPT
    if not setup_path.is_file():
        findings.append(Finding(
            check="installer_parity", severity="blocking", file=_SETUP_SCRIPT,
            message="installer script missing", remediation="restore greenboost_setup.sh"))
        return findings
    text = setup_path.read_text()

    if not manifest:
        findings.append(Finding(
            check="installer_parity", severity="advisory", file=str(_MANIFEST.relative_to(repo_root)),
            message="install_manifest.txt is empty or missing — no artifacts are being tracked for parity",
            remediation="seed the manifest: one line per Full-Install artifact "
                        "('name | install_regex | purge_regex'), see the file's header comment"))
        return findings

    for name, install_re, purge_re in manifest:
        try:
            has_install = bool(re.search(install_re, text))
            has_purge = bool(re.search(purge_re, text))
        except re.error as e:
            findings.append(Finding(
                check="installer_parity", severity="blocking",
                file=str(_MANIFEST.relative_to(repo_root)),
                message=f"bad regex in manifest entry {name!r}: {e}",
                remediation="fix the regex syntax in install_manifest.txt"))
            continue
        if not has_install:
            findings.append(Finding(
                check="installer_parity", severity="blocking", file=_SETUP_SCRIPT,
                message=f"manifest entry {name!r}: install pattern {install_re!r} not found",
                remediation=f"either {name} is no longer installed (remove the manifest line) "
                            f"or the install code moved (update the regex)"))
        if not has_purge:
            findings.append(Finding(
                check="installer_parity", severity="blocking", file=_SETUP_SCRIPT,
                message=f"manifest entry {name!r}: purge pattern {purge_re!r} not found",
                remediation=f"add the symmetric removal for {name} to the uninstall/purge path "
                            f"(do_purge), or update the regex if it moved"))
    return findings


if __name__ == "__main__":
    import sys
    fs = run(Path(__file__).resolve().parent.parent)
    for f in fs:
        print(f.format(llm="--llm" in sys.argv))
    print(f"\n{len(fs)} finding(s)")
    sys.exit(1 if any(f.severity == "blocking" for f in fs) else 0)

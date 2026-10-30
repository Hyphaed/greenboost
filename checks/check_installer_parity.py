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


def _bash_array(text: str, name: str) -> "list[str]":
    """gb_*.py names inside `name=( ... )` (comments stripped)."""
    m = re.search(re.escape(name) + r"=\(\s*(.*?)\n\s*\)", text, re.S)
    if not m:
        return []
    body = re.sub(r"#[^\n]*", "", m.group(1))
    return re.findall(r"gb_[a-z0-9_]+\.py", body)


def _state_paths(repo_root: Path) -> "list[str]":
    """Absolute install/state paths gb_synapse.py declares for the expert-cache engine (MOECACHE_* / ENGINE_CHOICE_FILE)."""
    src = repo_root / "gb_synapse.py"
    if not src.is_file():
        return []
    return re.findall(r'^(?:MOECACHE_ENGINE_(?:DIR|SRC)|ENGINE_CHOICE_FILE)\s*=\s*Path\("(/[^"]+)"\)',
                      src.read_text(), re.M)


def _check_python_and_engine(repo_root: Path, setup_text: str) -> "list[Finding]":
    """Uninstall completeness (round 2, 2026-10-30): the standalone installer's module list drifted from the full installer's
    (v3.5 modules were never removed by install_module.sh --uninstall). Guards: library subset is a subset of _py_files and
    exists; the uninstall derives its removal list from _py_files instead of the subset; engine paths are removed by both
    uninstallers."""
    out: "list[Finding]" = []
    im = repo_root / "install_module.sh"
    if not im.is_file():
        return out
    im_text = im.read_text()
    full = set(_bash_array(setup_text, "_py_files"))
    for name in _bash_array(im_text, "GB_PY_FILES"):
        if name not in full:
            out.append(Finding(check="installer_parity", severity="blocking", file="install_module.sh",
                               message=f"GB_PY_FILES lists {name} but greenboost_setup.sh's _py_files does not",
                               remediation=f"add {name} to cmd_install_python_files's _py_files or drop it from GB_PY_FILES"))
        if not (repo_root / name).is_file():
            out.append(Finding(check="installer_parity", severity="blocking", file="install_module.sh",
                               message=f"GB_PY_FILES lists {name} which is not in the repo", remediation="remove the stale entry"))
    if "_gb_all_py_files" not in im_text or re.search(r'for _f in "\$\{GB_PY_FILES\[@\]\}"; do _gb_rm', im_text):
        out.append(Finding(check="installer_parity", severity="blocking", file="install_module.sh",
                           message="the --uninstall module removal is not derived from the full installer's _py_files (drift hazard)",
                           remediation="remove modules via _gb_all_py_files (library subset + _py_files + checkout gb_*.py)"))
    for path in _state_paths(repo_root):
        base = path.rstrip("/")
        if base not in im_text:
            out.append(Finding(check="installer_parity", severity="blocking", file="install_module.sh",
                               message=f"engine path {path} (gb_synapse.py) is not removed by install_module.sh --uninstall",
                               remediation=f"add {path} to the engine block of the UNINSTALL section"))
        if base.startswith("/var/lib/greenboost/") and not re.search(r"rm -rf[^\n]*" + re.escape(base), setup_text):
            out.append(Finding(check="installer_parity", severity="blocking", file="greenboost_setup.sh",
                               message=f"{path} lives outside the wholesale /usr/local/lib/greenboost removal and do_purge never removes it",
                               remediation="add an rm -rf for it in do_purge (true uninstall branch)"))
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
    findings.extend(_check_python_and_engine(repo_root, text))

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

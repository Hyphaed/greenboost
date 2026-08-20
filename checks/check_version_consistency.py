#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""checks/check_version_consistency.py — every place GreenBoost core declares
its own version must declare the SAME version.

Why this exists (2026-08-20 incident): the project shipped v3.4 while
greenboost.c's MODULE_VERSION, dkms.conf's PACKAGE_VERSION, the Makefile's
GB_VERSION and the shim's GB_SHIM_VERSION were all still "3.2". Two things
broke, neither of them loudly:

  1. The Gaming Suite reads /sys/module/greenboost/version (i.e. MODULE_VERSION)
     as "the installed core version" and compares it to the newest GitLab
     release. A freshly installed 3.4 box therefore showed "installed 3.2 —
     Update available: 3.4" with an Upgrade-now button for a release it was
     already running.
  2. greenboost_setup.sh's `dkms autoinstall -m greenboost -v "$GB_VERSION"`
     (3.4) could never match what `make install` had registered (3.2), so it
     silently no-op'd behind `&>/dev/null || true` and no OTHER installed
     kernel ever got a DKMS build.

A version literal that disagrees with its siblings is never a stylistic
nitpick here — it is a wrong answer handed to an installer or a UI. Blocking,
no allowlist: there is no defensible reason for two of these to differ.

greenboost-cli/ (independently versioned, 1.0.0) and third_party/ are out of
scope by design.
"""
from __future__ import annotations

import re
from pathlib import Path

from lib import Finding

# (repo-relative path, compiled pattern with ONE capture group, human label).
# Each entry is a place that answers "what version is this?" to somebody:
# the kernel, DKMS, the build system, the capability manifest, an installer.
_SITES: "list[tuple[str, re.Pattern[str], str]]" = [
    ("greenboost.c",             re.compile(r'^MODULE_VERSION\("([0-9][0-9.]*)"\)'),        "kernel module MODULE_VERSION (what /sys/module/greenboost/version reports)"),
    ("greenboost.c",             re.compile(r'^#define GB_VERSION\s+"v([0-9][0-9.]*)"'),    "kernel module banner/status/pool_brief string"),
    ("dkms.conf",                re.compile(r'^PACKAGE_VERSION="([0-9][0-9.]*)"'),          "DKMS package version (/usr/src/greenboost-X, dkms status)"),
    ("Makefile",                 re.compile(r'^GB_VERSION\s*:=\s*([0-9][0-9.]*)'),          "build system (DKMS_ROOT, dkms add/build/install)"),
    ("greenboost_cuda_shim.c",   re.compile(r'^#define GB_SHIM_VERSION "([0-9][0-9.]*)"'),  "CUDA shim (/run/greenboost/capabilities.json 'version')"),
    ("greenboost_setup.sh",      re.compile(r'^GB_VERSION="([0-9][0-9.]*)"'),               "Debian/Ubuntu installer"),
    ("greenboost_setup_arch.sh", re.compile(r'^GB_VERSION="([0-9][0-9.]*)"'),               "Arch installer"),
    ("greenboost_setup_rocky.sh",re.compile(r'^GB_VERSION="([0-9][0-9.]*)"'),               "Rocky/Fedora installer"),
    ("install_module.sh",        re.compile(r'^GB_VERSION="([0-9][0-9.]*)"'),               "standalone kernel-module installer"),
    ("configure_boot_machine.sh",re.compile(r'^GB_VERSION="([0-9][0-9.]*)"'),               "boot-machine configurator"),
    ("CHANGELOG.md",             re.compile(r'^##\s+v([0-9][0-9.]*)\s*[:\-–]'),             "newest CHANGELOG heading"),
]

# greenboost.c's MODULE_VERSION is the canonical declaration: it is the one
# the kernel, modinfo, sysfs and therefore every external consumer (the
# Gaming Suite included) actually read off the running system.
_CANONICAL = ("greenboost.c", "kernel module MODULE_VERSION")


# A CHANGELOG heading that says outright it is not released is NOT a claim
# about what the built artifact is, so it must not be compared against
# MODULE_VERSION. Bumping the core's version literals to match an unreleased
# heading would be actively harmful: the running module would report a version
# no release carries, and the Gaming Suite reads exactly that field as
# "installed core version" and compares it to the newest published release ,
# which is the 2026-08-20 confusion this whole check exists to prevent, only
# inverted. Development on `main` ahead of a tag is normal and must not be
# forced to lie about it in either direction.
_UNRELEASED_RE = re.compile(
    r"(unreleased|not released|in development|in-development|wip|draft)",
    re.IGNORECASE)


def _first_match(path: Path, pat: "re.Pattern[str]") -> "tuple[str, int] | None":
    """First (value, lineno) the pattern yields. Only the FIRST hit matters:
    CHANGELOG.md lists every past version, and the newest heading is on top.

    A heading marked unreleased is skipped rather than matched, so the scan
    falls through to the newest heading that IS a release claim."""
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return None
    for lineno, line in enumerate(lines, 1):
        m = pat.match(line.strip())
        if m:
            if _UNRELEASED_RE.search(line):
                continue
            return m.group(1), lineno
    return None


def run(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    found: "list[tuple[str, int, str, str]]" = []   # rel, line, value, label

    for rel, pat, label in _SITES:
        path = repo_root / rel
        if not path.is_file():
            # A missing installer variant on a trimmed checkout is not drift.
            continue
        hit = _first_match(path, pat)
        if hit is None:
            findings.append(Finding(
                check="version_consistency", severity="blocking", file=rel, line=0,
                message=f"no version declaration found for: {label}",
                remediation=f"restore the declaration this check anchors on "
                            f"(pattern: {pat.pattern!r}); if the declaration moved "
                            f"or was renamed, update _SITES in "
                            f"checks/check_version_consistency.py to match"))
            continue
        value, lineno = hit
        found.append((rel, lineno, value, label))

    if not found:
        return findings

    canonical = next((v for r, _, v, lab in found
                      if (r, lab) == _CANONICAL), None)
    if canonical is None:
        canonical = found[0][2]

    for rel, lineno, value, label in found:
        if value == canonical:
            continue
        findings.append(Finding(
            check="version_consistency", severity="blocking", file=rel, line=lineno,
            message=f"declares version {value}, but greenboost.c's MODULE_VERSION "
                    f"says {canonical} — this site is the {label}",
            remediation=f"set this to {canonical}, or bump every site together. "
                        f"A split here means the installer, DKMS and the Gaming "
                        f"Suite's update card disagree about what is installed "
                        f"(see this file's docstring for the 2026-08-20 incident). "
                        f"Sites: " + ", ".join(sorted({r for r, _, _, _ in found}))))
    return findings


if __name__ == "__main__":
    import sys
    fs = run(Path(__file__).resolve().parent.parent)
    for f in fs:
        print(f.format(llm="--llm" in sys.argv))
    print(f"\n{len(fs)} finding(s)")
    sys.exit(1 if fs else 0)

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""checks/check_vendor_notices.py — every third_party/*/NOTICE must record
a real, mechanically-rebasable upstream pin (an exact commit SHA, not just
a date), per the P8 lesson (plan bring-gb-synapse-gb-quant-and-async-nygaard.md):
third_party/gemlite/NOTICE originally recorded only "no upstream commits
after 2026-06-30" — a date alone can't be used to `git diff`/rebase against
upstream; the exact commit can.

Blocking, but scoped to directories that ALREADY declare themselves vendored
(i.e. already have a NOTICE file) — this check enforces NOTICE quality, it
does not mandate which third_party/ subdirectories need one in the first
place (that's a human call, made once per vendor addition).
"""
from __future__ import annotations

import re
from pathlib import Path

from lib import Finding

_SHA_RE = re.compile(r"Upstream commit:\s*([0-9a-fA-F]{7,40})\b")
_BULLET_RE = re.compile(r"^- ", re.MULTILINE)
_LOCAL_MODS_RE = re.compile(r"^Local modifications\b", re.MULTILINE)

# Directories under third_party/ that are large upstream sync targets, not
# "vendor a specific commit" in the same sense (their own build tooling
# tracks a pinned checkout elsewhere, e.g. llama.cpp's git submodule-style
# update-engine flow) — exempted from the exact-SHA requirement here to
# avoid duplicating a pin that already lives in a more authoritative place.
_EXEMPT: "tuple[str, ...]" = ()


def run(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    third_party = repo_root / "third_party"
    if not third_party.is_dir():
        return findings

    for notice in sorted(third_party.glob("*/NOTICE")):
        vendor_dir = notice.parent.name
        if vendor_dir in _EXEMPT:
            continue
        text = notice.read_text(errors="ignore")
        rel = str(notice.relative_to(repo_root))

        m = _SHA_RE.search(text)
        if not m:
            findings.append(Finding(
                check="vendor_notices", severity="blocking", file=rel,
                message="no 'Upstream commit: <sha>' line found",
                remediation="record the exact upstream commit SHA this vendored "
                            "tree was copied from (not just a date) so it stays "
                            "mechanically rebasable against upstream"))
            continue
        sha = m.group(1)
        if len(sha) < 7:
            findings.append(Finding(
                check="vendor_notices", severity="blocking", file=rel,
                message=f"upstream commit SHA {sha!r} is too short to be unambiguous",
                remediation="use at least a 7-character abbreviated SHA (prefer the full 40)"))

        has_mods_section = bool(_LOCAL_MODS_RE.search(text))
        n_bullets = len(_BULLET_RE.findall(text))
        if has_mods_section and n_bullets == 0:
            findings.append(Finding(
                check="vendor_notices", severity="advisory", file=rel,
                message="has a 'Local modifications' section but no '- ' bulleted "
                        "entries under it",
                remediation="either list the local patches as bullets, or remove "
                            "the section if this vendor tree truly has none"))

    return findings

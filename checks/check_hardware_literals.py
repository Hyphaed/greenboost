#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""checks/check_hardware_literals.py — flags hardcoded hardware-shaped
values (absolute VRAM/RAM/pool sizes) in tracked C/Python/bash source,
per golden-principles.md #1. Blocking, allowlist-seeded: the first run of
this check (2026-07-13) IS the audit — every finding in
checks/allowlists/hardware.txt was individually read and is either (a) a
sanctioned floor/protocol constant (wire sizes, kernel page size, a
`max(floor, pct*capacity)` lower bound) or (b) a KNOWN, already-tracked
pending fix (see workflow/tasks-jul13-sources-harness.md Phase 7, DI-14) —
not new, undetected drift.

This is a heuristic regex scan, not a semantic analyzer: it flags candidate
lines for human judgment, tuned to have few enough false positives that the
allowlist stays small and every entry defensible. It will miss cleverly
obfuscated literals and it will occasionally flag a genuinely fine
%-derivation that happens to have a big number nearby — that's why every
finding needs a remediation-guided human look, not a silent auto-reject.
"""
from __future__ import annotations

import re
from pathlib import Path

from lib import Finding, is_allowlisted, iter_tracked_files, load_allowlist

# Absolute size literal: a number (>=3 digits, so small floors like "4" or
# "64" for page-adjacent constants don't spam) directly followed by an
# explicit size-unit suffix. Deliberately does NOT match bare numbers (too
# noisy) or numbers already inside an obvious fraction expression
# (0.xx * ...) — those are exactly the sanctioned pattern.
#
# Earlier revision also matched bare `<< 20`/`<< 30` bit-shifts, on the
# theory that "shifted into MB/GB range" implies a hardcoded size. In
# practice this was the dominant false-positive source (verified against
# greenboost.c/greenboost_cuda_shim.c during the first audit run,
# 2026-07-13): `(u64)detected_var * (1ULL << 30)` is the SANCTIONED pattern
# — converting an already-%-derived GB count to bytes — and a regex can't
# tell that apart from `64UL << 20` (a genuine literal) without parsing the
# left operand. Dropped rather than accept a check nobody trusts the output
# of; a future revision could re-add it with a real parser (e.g. check the
# token immediately before `<<` is a bare integer literal, not an
# identifier).
_SIZE_LITERAL_RE = re.compile(
    r"""
    (?<![.\d])                       # not part of a larger number / float
    (\d{3,})                        # the literal
    \s*
    (?:MB|GB|MiB|GiB|_MB|_GB)(?![A-Za-z0-9_])   # unit suffix, not embedded in
                                                  # a longer identifier (e.g.
                                                  # "100GBASE_SR4" ethernet enum)
    """,
    re.VERBOSE,
)

# Lines that are clearly a %-derivation (multiply by a fraction, or divide/
# modulo by a variable) are not flagged even if a size unit appears nearby —
# the rule is about ABSOLUTE literals, not every mention of MB/GB.
_FRACTION_CONTEXT_RE = re.compile(r"0\.\d+\s*\*|\*\s*0\.\d+|pct\s*/\s*100|/\s*100\.0")

_SUFFIXES = (".c", ".h", ".py", ".sh")
_SKIP_DIR_SUBSTR = ("third_party/", "greenboost-cli/", "nvidia_markdowns/",
                    "ebpf/vmlinux.h",  # vendored/generated kernel header (BTF dump), not GreenBoost logic
                    ".git/", "tests/", "checks/")  # tests fixture constants
                                                    # and this suite's own
                                                    # regex literals aren't
                                                    # placement decisions



def _strip_non_code(path, lines: "list[str]") -> "list[str]":
    """Blank out string and comment CONTENT, keeping line numbering intact.

    Python is tokenized, so the answer is exact. Everything else (shell, C) gets
    a quote-stripping heuristic: good enough to drop `info "...12GB..."` and a
    usage message, and it never removes a bare literal, which is the only thing
    this check is looking for.
    """
    if path.suffix == ".py":
        try:
            import io
            import tokenize
            out = list(lines)
            src = "\n".join(lines)
            for tok in tokenize.generate_tokens(io.StringIO(src).readline):
                if tok.type not in (tokenize.STRING, tokenize.COMMENT):
                    continue
                (r1, c1), (r2, c2) = tok.start, tok.end
                for r in range(r1, r2 + 1):
                    if r - 1 >= len(out):
                        break
                    text = out[r - 1]
                    a = c1 if r == r1 else 0
                    b = c2 if r == r2 else len(text)
                    out[r - 1] = text[:a] + " " * max(0, b - a) + text[b:]
            return out
        except Exception:
            # A file that will not tokenize (partial edit, py2 syntax) falls
            # back to the heuristic rather than being skipped silently.
            pass
    stripped = []
    for line in lines:
        for q in ('"', "'"):
            parts = line.split(q)
            if len(parts) > 2:            # at least one closed quote pair
                line = parts[0] + "".join(
                    ("" if i % 2 else p) for i, p in enumerate(parts[1:], 1))
        stripped.append(line)
    return stripped


def run(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    allow = load_allowlist(repo_root / "checks" / "allowlists" / "hardware.txt")
    for path in iter_tracked_files(repo_root, _SUFFIXES):
        rel = str(path.relative_to(repo_root))
        if any(s in rel for s in _SKIP_DIR_SUBSTR):
            continue
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        # Prose is not a constant. The line-start comment test below only ever
        # caught the FIRST line of a comment block; a measured figure written
        # into a docstring continuation, a KindSpec doc= string, or a shell
        # usage message looked exactly like a hardcoded literal. On 2026-08-22
        # all 11 findings this check produced were prose , measured numbers in
        # comments, a `info "...RTX 5070 12GB..."` preset message, and the
        # usage text of `t3-memory <size>` , and it exited non-zero on every
        # run while nothing in the suite invoked it. A blocking gate that is
        # 100% false positives is a gate everyone learns to skip, which is how
        # a real hardcoded constant would get through.
        code_only = _strip_non_code(path, lines)
        for lineno, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith(("#", "//", "*", "/*")):
                continue  # comments (incl. this check's own docstring examples)
            # Match only what survives with strings and comments removed.
            line = code_only[lineno - 1]
            if not line.strip():
                continue
            if _FRACTION_CONTEXT_RE.search(line):
                continue
            if not _SIZE_LITERAL_RE.search(line):
                continue
            if is_allowlisted(rel, lineno, allow, line):
                continue
            findings.append(Finding(
                check="hardware_literals", severity="blocking", file=rel, line=lineno,
                message=f"possible hardcoded hardware-shaped literal: {stripped[:100]!r}",
                remediation="derive this from gb_topology.py (compute_reserve_gb/"
                            "hbm_headroom_mb pattern: max(floor, pct*detected_capacity)) "
                            "instead of an absolute size; if this is a sanctioned floor "
                            "or wire-protocol constant, add 'path:line  # reason' to "
                            "checks/allowlists/hardware.txt"))
    return findings


if __name__ == "__main__":
    import sys
    fs = run(Path(__file__).resolve().parent.parent)
    for f in fs:
        print(f.format(llm="--llm" in sys.argv))
    print(f"\n{len(fs)} finding(s)")
    sys.exit(1 if fs else 0)

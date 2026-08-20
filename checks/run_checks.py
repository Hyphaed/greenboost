#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""checks/run_checks.py — runs every check module in checks/, aggregates
Findings, prints them (human or --llm), and exits non-zero iff any BLOCKING
finding survives (advisory findings never fail the build).

Usage:
    python3 checks/run_checks.py                 # human-readable, all checks
    python3 checks/run_checks.py --llm            # compact machine-readable
    python3 checks/run_checks.py --only hardware_literals,secrets
    python3 checks/run_checks.py --blocking-only  # hide advisory findings
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_dataflux_coverage
import check_doc_drift
import check_docs
import check_hardware_literals
import check_import_boundaries
import check_installer_parity
import check_mcp_parity
import check_mcp_readonly
import check_python_manifest
import check_secrets
import check_semantics_coverage
import check_vendor_notices
import check_version_consistency

_CHECKS = {
    "hardware_literals": check_hardware_literals.run,
    "dataflux_coverage": lambda root: check_dataflux_coverage.run(root),
    "mcp_parity": check_mcp_parity.run,
    "mcp_readonly": check_mcp_readonly.run,
    "python_manifest": check_python_manifest.run,
    "installer_parity": check_installer_parity.run,
    "secrets": check_secrets.run,
    "docs": lambda root: check_docs.run(root, gc=False),
    "vendor_notices": check_vendor_notices.run,
    "semantics_coverage": check_semantics_coverage.run,
    "import_boundaries": check_import_boundaries.run,
    "doc_drift": check_doc_drift.run,
    "version_consistency": check_version_consistency.run,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="compact machine-readable output")
    ap.add_argument("--only", default="", help="comma-separated check names to run")
    ap.add_argument("--blocking-only", action="store_true", help="hide advisory findings")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(_CHECKS)
    unknown = [n for n in names if n not in _CHECKS]
    if unknown:
        print(f"unknown check name(s): {unknown}; known: {list(_CHECKS)}", file=sys.stderr)
        return 2

    all_findings = []
    for name in names:
        all_findings.extend(_CHECKS[name](repo_root))

    if args.blocking_only:
        all_findings = [f for f in all_findings if f.severity == "blocking"]

    blocking = [f for f in all_findings if f.severity == "blocking"]
    advisory = [f for f in all_findings if f.severity == "advisory"]

    for f in all_findings:
        print(f.format(llm=args.llm))
        if not args.llm:
            print()

    print(f"\n{len(blocking)} blocking, {len(advisory)} advisory finding(s) "
         f"across {len(names)} check(s)")
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())

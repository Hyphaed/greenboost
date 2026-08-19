#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""checks/check_import_boundaries.py — NemoClaw audit round 2, item 6.
Mechanizes two module-level import-direction rules that today are only
comments a human has to remember to check: gb_synapse_backends.py's
one-way relationship with gb_synapse.py/gb_cluster.py (gb_synapse.py
imports gb_synapse_backends at the top; the reverse at module level would
be circular), and the two GPU-touch-on-import rules in gb_kernel_backends.py
and gb_api.py (torch must be imported lazily inside functions, never at
module level, so importing the module never touches the GPU).

A lazy import inside a function body is fine and does not trigger a
violation , only a top-level `import X` / `from X import Y` (including one
inside a top-level try/except, since that still runs at import time) counts.

BLOCKING: these are real correctness rules (a violation either creates an
import cycle or breaks the "importing must never touch the GPU" contract),
not a style preference.
"""
from __future__ import annotations

import ast
from pathlib import Path

from lib import Finding

# (importer module stem, forbidden import stem, reason)
PROHIBIT_RULES: tuple[tuple[str, str, str], ...] = (
    ("gb_synapse_backends", "gb_synapse",
     "gb_synapse_backends.py must not import gb_synapse at module level "
     "(gb_synapse.py imports gb_synapse_backends at the top; the reverse "
     "at module level is circular) — import it lazily inside methods."),
    ("gb_synapse_backends", "gb_cluster",
     "gb_synapse_backends.py must not import gb_cluster at module level "
     "— import it lazily inside methods (same circular-import risk as "
     "the gb_synapse rule above)."),
    ("gb_kernel_backends", "torch",
     "gb_kernel_backends.py must not import torch at module level — the "
     "module must stay importable/CPU-testable with no GPU touched at "
     "import time; import torch lazily inside functions."),
    ("gb_api", "torch",
     "gb_api.py must not import torch at module level — GPU-touching "
     "deps must be imported inside function bodies so importing the "
     "module never touches the GPU."),
)


def _module_level_imports(path: Path) -> list[tuple[int, str]]:
    """(lineno, imported_module_stem) for every top-level import statement,
    including ones inside a top-level try/except (still runs at import
    time) — but NOT an import nested inside a function/class body."""
    try:
        tree = ast.parse(path.read_text(errors="ignore"), filename=str(path))
    except SyntaxError:
        return []

    out: list[tuple[int, str]] = []

    def walk_body(body: list[ast.stmt]) -> None:
        for stmt in body:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    out.append((stmt.lineno, alias.name.split(".")[0]))
            elif isinstance(stmt, ast.ImportFrom):
                if stmt.module:
                    out.append((stmt.lineno, stmt.module.split(".")[0]))
            elif isinstance(stmt, ast.Try):
                walk_body(stmt.body)
                for handler in stmt.handlers:
                    walk_body(handler.body)
                walk_body(stmt.orelse)
                walk_body(stmt.finalbody)

    walk_body(tree.body)
    return out


def run(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    candidates = list(repo_root.glob("gb_*.py")) + list(
        repo_root.glob("greenboost-cli/greenboost_cli/**/*.py"))

    for path in candidates:
        module_stem = path.stem
        rules = [r for r in PROHIBIT_RULES if module_stem == r[0]]
        if not rules:
            continue
        for lineno, imported in _module_level_imports(path):
            for _importer, forbidden, reason in rules:
                if imported == forbidden:
                    findings.append(Finding(
                        check="import_boundaries", severity="blocking",
                        file=str(path.relative_to(repo_root)), line=lineno,
                        message=f"forbidden module-level import of '{imported}'",
                        remediation=reason))
    return findings


if __name__ == "__main__":
    import sys
    fs = run(Path(__file__).resolve().parent.parent)
    for f in fs:
        print(f.format())
    sys.exit(1 if any(f.severity == "blocking" for f in fs) else 0)

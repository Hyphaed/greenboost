#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""checks/check_dataflux_coverage.py — golden-principles.md #2: every module
that makes a placement/scheduling/tiering/actuation decision must emit
dataflux telemetry (directly, or by calling into one that does), AND every
dataflux event kind that's emitted must be declared in the registry
(gb_dataflux_kinds.py) , `workflow/dataflux-schema-additions.md`'s parity
rule: "an event kind that's emitted but not queryable, or documented as
queryable but never emitted, is a bug."

Three passes:
  1. Module-name heuristic (advisory, unchanged from the original version):
     a known decision-making module family should reference gb_dataflux
     SOMEWHERE. Heuristic, not semantic , stays advisory.
  2. Kind-literal scan (BLOCKING, new): collects every literal kind string
     passed to a known emit call shape anywhere in tracked *.py (excluding
     tests/, whose fixtures use fake kind values), and asserts:
       a. every literal found is a registered gb_dataflux_kinds.KINDS key
          (an unregistered kind is a bug the moment it's written , this is
          exactly what would have caught the "agent_run" bug: documented in
          two places, emitted nowhere, because the emitter hardcoded a
          different literal).
       b. every KINDS key either has ≥1 real emit site found by this scan,
          or is marked `planned=True` (emitted by a consumer repo like
          ai-forge, not by GreenBoost itself , grepping ai-forge is out of
          scope for a check that lives in this repo).

  Recognized emit-call shapes (NOT a general Python static analyzer , this
  scan targets the concrete shapes actually used in this codebase, listed
  here so a new shape is a deliberate addition, not silent blindness):
    - `gb_dataflux.emit({"kind": "<literal>", ...})` , dict-literal shape.
    - `_emit(..., kind="<literal>", ...)` / any `kind="<literal>"` keyword
      argument on a call , covers gb_actuation._emit's kind kwarg.
    - `_df_emit(<3 positional args>, "<literal>", ...)` , gb_cluster.py's
      wrapper takes kind as its 4th positional argument.
    - Shell shape (NemoClaw audit, Phase 2, added when greenboost_setup.sh
      gained its own PID-ownership-proof emit sites): any *.sh file
      containing a `gb_dataflux_emit` call is scanned, file-wide, for a
      `"kind":"<literal>"` JSON-object substring — the bash-side mirror of
      the Python dict-literal shape above. gb_dataflux_emit is a thin
      best-effort JSON-line appender (see greenboost_setup.sh's own
      docstring for it), not a Python call, so it can't be AST-parsed; this
      is a deliberate second, regex-based scan pass over shell sources
      rather than a silent blind spot (Pass 3 below existed specifically
      because this blind spot was real for a different non-Python frontend
      — greenboost_gaming's Tauri backend — before this shape existed).

  3. Live-log cross-check (ADVISORY, added 2026-07-30, greenboost_polish.md
     finding D2): Pass 2's AST scan can only ever see kinds emitted by THIS
     repo's own *.py files , a kind emitted by a consumer repo (ai-forge) or
     a non-Python frontend (greenboost_gaming's Rust/Tauri backend) is
     structurally invisible to it. That blind spot is exactly how
     qc_summary/finish_summary (ai-forge) and gaming_session/
     gaming_vram_pressure (greenboost_gaming) sat unregistered in the real
     shared dataflux log for a long time before anyone noticed. This pass
     reads the actual live dataflux log (gb_dataflux.read_events(), the same
     data dataflux_kinds() surfaces over MCP) and flags any kind that has
     really appeared but has no registry entry, regardless of which repo or
     language emitted it. Advisory, not blocking: a fresh checkout with an
     empty or short-lived dataflux log has nothing to cross-check yet, and
     that must not fail the check.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from lib import Finding, is_allowlisted, iter_tracked_files, load_allowlist

_DECISION_MODULE_PATTERNS = (
    "gb_placement.py", "gb_cluster.py", "gb_model_tier.py", "gb_orchestrator.py",
    "gb_quant.py", "gb_synapse.py", "gb_tiering.py", "gb_mem_pool.py",
    "gb_attn.py", "gb_actuation.py", "gb_a2a.py", "gb_reactive.py",
)

_KIND_RE = re.compile(r'^[a-zA-Z_][a-zA-Z_0-9]*$')


def _literal_str(node) -> "str | None":
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_func_name(node: ast.Call) -> "str | None":
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_emit_call(node: ast.Call) -> bool:
    """True for a call that plausibly emits a dataflux event: `emit(...)`
    (gb_dataflux.emit) or any name ending in `_emit` (gb_actuation._emit,
    gb_cluster._df_emit, ...)."""
    name = _call_func_name(node)
    return name is not None and (name == "emit" or name.endswith("_emit"))


def _find_kind_literals_in_file(path: Path) -> "set[str]":
    """AST-scan one file for the 3 recognized emit-call shapes, return the
    set of literal kind strings found. Best-effort: a file that fails to
    parse is skipped (never crashes the check).

    Scope is per-FILE, not per-call: the dominant real pattern in this
    codebase builds the event dict as a local variable THEN passes the
    variable to emit() (`ev = {"kind": "x", ...}; gb_dataflux.emit(ev)`),
    not inline (`emit({"kind": "x", ...})`) , so requiring the dict to be a
    direct argument of the emit call (an earlier version of this scan) MISSED
    the majority of real emit sites (snapshot, link_transfer, tier_move,
    node_capabilities, niah_cert, smoke_gate all use the variable-mediated
    form). Falling back to "does this file contain any emit-shaped call
    ANYWHERE, and if so, collect every kind-literal shape ANYWHERE in it" is
    coarser but correct for every real site verified in this repo, and it
    still doesn't reintroduce the vendored gLLM engine's false positive
    (synapse_engine/gllm/model_runner.py's unrelated "kind": "cached"/
    "uncached" dict keys, verified to feed only `.append()`) because that
    file has zero calls matching _is_emit_call anywhere , unlike
    async_llm_engine.py, which has a real one (synapse_engine_error)."""
    try:
        text = path.read_text(errors="ignore")
        tree = ast.parse(text, filename=str(path))
    except (SyntaxError, ValueError):
        return set()

    all_nodes = list(ast.walk(tree))
    if not any(isinstance(n, ast.Call) and _is_emit_call(n) for n in all_nodes):
        return set()

    found: set[str] = set()
    for node in all_nodes:
        # Shape 1: any dict literal in the file with a "kind": "<literal>" entry.
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "kind":
                    lit = _literal_str(v)
                    if lit and _KIND_RE.match(lit):
                        found.add(lit)

        if not isinstance(node, ast.Call):
            continue

        # Shape 2: a `kind=` keyword argument on any call (gb_actuation._emit(
        # ..., kind="agent_run")).
        for kw in node.keywords:
            if kw.arg == "kind":
                lit = _literal_str(kw.value)
                if lit and _KIND_RE.match(lit):
                    found.add(lit)

        # Shape 3: `_df_emit`'s 4th POSITIONAL argument is `kind`
        # (gb_cluster.py's wrapper: run_id, node, label, kind, items, ...).
        func_name = _call_func_name(node)
        if func_name and func_name.endswith("_df_emit") and len(node.args) >= 4:
            lit = _literal_str(node.args[3])
            if lit and _KIND_RE.match(lit):
                found.add(lit)

    return found


_SH_KIND_RE = re.compile(r'"kind"\s*:\s*"([a-zA-Z_][a-zA-Z_0-9]*)"')


def _find_kind_literals_in_sh_file(path: Path) -> "set[str]":
    """Regex-scan one shell file for the `gb_dataflux_emit` shape (see the
    module docstring's shape-4 note). File-scoped like the Python scan: if
    `gb_dataflux_emit` is called anywhere in the file, collect every
    `"kind":"<literal>"` JSON substring anywhere in it. Best-effort , an
    unreadable file is skipped, never crashes the check."""
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return set()
    if "gb_dataflux_emit" not in text:
        return set()
    return {m for m in _SH_KIND_RE.findall(text) if _KIND_RE.match(m)}


def run(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []

    # ── Pass 1: module-name heuristic (advisory, unchanged) ──────────────
    allow = load_allowlist(repo_root / "checks" / "allowlists" / "dataflux_exempt.txt")
    for pattern in _DECISION_MODULE_PATTERNS:
        path = repo_root / pattern
        if not path.is_file():
            continue
        rel = pattern
        if is_allowlisted(rel, 0, allow):
            continue
        text = path.read_text(errors="ignore")
        if "gb_dataflux" not in text:
            findings.append(Finding(
                check="dataflux_coverage", severity="advisory", file=rel,
                message="decision-making module never references gb_dataflux — "
                        "no telemetry emission found anywhere in the file",
                remediation="add a best-effort gb_dataflux.emit(...) at the decision "
                            "point (see gb_placement.py's plan_and_emit for the pattern), "
                            "or add to checks/allowlists/dataflux_exempt.txt with a reason "
                            "if this module genuinely makes no scheduling/placement decisions"))

    # ── Pass 2: kind-literal registry parity (BLOCKING) ──────────────────
    try:
        import sys
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import gb_dataflux_kinds
    except ImportError:
        findings.append(Finding(
            check="dataflux_coverage", severity="blocking", file="gb_dataflux_kinds.py",
            message="registry module missing or fails to import — kind-literal "
                    "parity cannot be checked",
            remediation="restore gb_dataflux_kinds.py"))
        return findings

    registry_keys = set(gb_dataflux_kinds.KINDS)
    found_by_file: dict[str, set[str]] = {}
    all_found: set[str] = set()

    # tests/ fixtures use fake kind values, not real emit sites.
    # third_party/ (gemlite, llama.cpp, ...) is vendored, no dataflux
    # integration expected. synapse_engine/ (also vendored) IS scanned ,
    # GreenBoost added a real gb_dataflux.emit() call inside it
    # (synapse_engine_error) , _is_emit_call()'s scoping keeps that file's
    # own unrelated "kind" dict key from producing a false positive.
    for path in iter_tracked_files(repo_root, (".py",)):
        rel = str(path.relative_to(repo_root))
        if rel.startswith("tests/") or rel.startswith("third_party/") or "/tests/" in rel:
            continue
        lits = _find_kind_literals_in_file(path)
        if lits:
            found_by_file[rel] = lits
            all_found |= lits

    # Shell shape (shape 4, see module docstring) — greenboost_setup.sh's
    # own gb_dataflux_emit calls (e.g. purge_action, NemoClaw audit Phase 2).
    for path in iter_tracked_files(repo_root, (".sh",)):
        rel = str(path.relative_to(repo_root))
        if rel.startswith("tests/") or rel.startswith("third_party/") or "/tests/" in rel:
            continue
        lits = _find_kind_literals_in_sh_file(path)
        if lits:
            found_by_file[rel] = found_by_file.get(rel, set()) | lits
            all_found |= lits

    unregistered = sorted(all_found - registry_keys)
    for kind in unregistered:
        sites = sorted(f for f, lits in found_by_file.items() if kind in lits)
        findings.append(Finding(
            check="dataflux_coverage", severity="blocking", file=sites[0] if sites else "?",
            message=f'kind="{kind}" is emitted (at {", ".join(sites)}) but not registered '
                    f"in gb_dataflux_kinds.KINDS",
            remediation=f'add a KindSpec entry for "{kind}" to gb_dataflux_kinds.py in the '
                        f"same change that introduced this emit call"))

    never_emitted = sorted(
        k for k, spec in gb_dataflux_kinds.KINDS.items()
        if not spec.planned and k not in all_found)
    for kind in never_emitted:
        findings.append(Finding(
            check="dataflux_coverage", severity="blocking", file="gb_dataflux_kinds.py",
            message=f'kind="{kind}" is registered but no emit site was found in this repo',
            remediation=f'either add the emit call this registry entry describes, or mark '
                        f'it planned=True if it is emitted by a consumer repo (ai-forge) '
                        f'rather than GreenBoost itself'))

    # ── Pass 3: live-log cross-check (ADVISORY) ──────────────────────────
    try:
        import gb_dataflux
        live_events = gb_dataflux.read_events(since_hours=None)
    except Exception:
        live_events = []

    live_kinds = {e.get("kind") for e in live_events if e.get("kind")}
    unregistered_live = sorted(live_kinds - registry_keys)
    for kind in unregistered_live:
        findings.append(Finding(
            check="dataflux_coverage", severity="advisory", file="gb_dataflux_kinds.py",
            message=f'kind="{kind}" appears in the live dataflux log but is not registered '
                    f"in gb_dataflux_kinds.KINDS — likely emitted by a consumer repo or a "
                    f"non-Python frontend the Pass 2 AST scan can't see",
            remediation=f'add a KindSpec entry for "{kind}" (planned=True if a consumer '
                        f"repo or non-Python component emits it, e.g. ai-forge or "
                        f"greenboost_gaming's Tauri backend)"))

    return findings


if __name__ == "__main__":
    import sys
    fs = run(Path(__file__).resolve().parent.parent)
    for f in fs:
        print(f.format(llm="--llm" in sys.argv))
    blocking = [f for f in fs if f.severity == "blocking"]
    print(f"\n{len(fs)} finding(s) ({len(blocking)} blocking)")
    sys.exit(1 if blocking else 0)

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""checks/check_semantics_coverage.py — GB-Semantics parity, modeled directly
on check_dataflux_coverage.py's registry-parity pattern applied to
semantics/*.yaml + gb_semantics.py.

A metric/segment that's defined but has no real resolver/evaluator function is
exactly as bad as a dataflux kind that's documented but never emitted — it
looks governed to an LLM client but silently errors (or worse, is never
noticed because nothing calls it) the first time it's actually used.

BLOCKING:
  1. semantics/*.yaml parse and gb_semantics.load() succeeds.
  2. Every metric has a real `_res_<resolver>` function in gb_semantics.py.
  3. Every segment has a real `_seg_<evaluator>` function in gb_semantics.py.
  4. Every metric has a non-empty `owner`.
  5. Every route's `metrics`/`segments` names exist in metrics.yaml/segments.yaml
     (catches a route referencing a renamed/typo'd metric or segment).
  6. Every `source_fields` entry shaped `<module>.<ClassName>.<field>` against
     a KNOWN class (GbSnapshot, ModelEntry, ServerState — real, fully
     enumerable dataclasses) must name a real field. Entries in any other
     shape (a dataflux `mod.kind.field` reference, a CLI invocation, a plain
     description like "free -h") are left unchecked, same advisory stance
     gb_dataflux_kinds.KindSpec.fields already takes on dataflux events —
     those aren't fully enumerable the same way.

ADVISORY:
  7. A `never_use.field` whose leading token is a bare identifier that
     doesn't appear anywhere in the known trap-field universe (GpuMetrics +
     GbPoolInfo + GbSnapshot + ModelEntry + ServerState field names) — a
     stale trap warning is worse than none, but this heuristic is too fuzzy
     (most never_use entries are prose, not clean field paths) to block on.
"""
from __future__ import annotations

import sys
from pathlib import Path

from lib import Finding

_REPO_ROOT = Path(__file__).resolve().parent.parent

# module.ClassName pairs whose fields are real, fully-enumerable dataclasses —
# safe to validate strictly. Anything else in a source_fields entry is
# documentation-only, same advisory stance as KindSpec.fields.
_KNOWN_CLASSES = {
    ("gb_monitor", "GbSnapshot"),
    ("gb_synapse", "ModelEntry"),
    ("gb_synapse", "ServerState"),
    ("gb_telemetry", "GpuMetrics"),
    ("gb_telemetry", "GbPoolInfo"),
    # Added 2026-08-18 with the pcie_link_gen_current metric. GpuTopology is a
    # frozen dataclass like the rest, and its cached-at-init PCIe fields are
    # exactly the kind of trap never_use exists to name — leaving the class out
    # meant that trap could only ever be advisory, never validated.
    ("gb_telemetry", "GpuTopology"),
    # Added 2026-08-19 with proxy_mem_headroom_pct. Host MemAvailable is the
    # field an operator reaches for when the proxy stops responding, and it is
    # the wrong scope — registering the class makes that trap validated.
    ("gb_telemetry", "SystemMetrics"),
}


def _dataflux_kind_field_names() -> "set[str]":
    """Every field name declared by any KindSpec in gb_dataflux_kinds.py.

    Parsed rather than imported: this checker must run without importing the
    GreenBoost stack (no torch, no NVML), the same reason _class_field_names
    reads source instead of importing.
    """
    import ast as _ast
    out: "set[str]" = set()
    path = _REPO_ROOT / "gb_dataflux_kinds.py"
    try:
        tree = _ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return out
    for node in _ast.walk(tree):
        if not (isinstance(node, _ast.Call)
                and getattr(node.func, "id", "") == "KindSpec"):
            continue
        for kw in node.keywords:
            if kw.arg not in ("fields", "numeric_fields"):
                continue
            for elt in getattr(kw.value, "elts", []):
                if isinstance(elt, _ast.Constant) and isinstance(elt.value, str):
                    out.add(elt.value)
    return out


def _class_field_names(module: str, cls_name: str) -> "set[str] | None":
    import dataclasses
    import importlib
    try:
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
    except Exception:
        return None
    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls)}
    return None


def _token_in_declared_files(metric, token: str) -> bool:
    """True if `token` appears in a filesystem source the metric declares.

    Only paths the metric itself names in source_fields are consulted, and only
    for reading — a metric cannot cause an unrelated file to be opened. Missing
    or unreadable files simply return False, leaving the existing advisory in
    place rather than passing a trap that could not be checked.
    """
    import os as _os
    for sf in getattr(metric, "source_fields", []) or []:
        path = str(sf).split()[0]
        if not path.startswith(("/proc/", "/sys/")):
            continue
        try:
            if not _os.path.exists(path):
                continue
            with open(path) as fh:
                if token in fh.read():
                    return True
        except Exception:
            continue
    return False


def run(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        import gb_semantics
    except Exception as e:
        findings.append(Finding(
            check="semantics_coverage", severity="blocking", file="gb_semantics.py",
            message=f"gb_semantics failed to import: {e}",
            remediation="fix the import error (likely pyyaml missing, or a "
                        "malformed semantics/*.yaml file)"))
        return findings

    try:
        L = gb_semantics.load(force=True)
    except Exception as e:
        findings.append(Finding(
            check="semantics_coverage", severity="blocking", file="semantics/",
            message=f"gb_semantics.load() raised: {e}",
            remediation="fix the malformed YAML — semantics/*.yaml must parse "
                        "and every entry must have its required keys"))
        return findings

    metrics = L["metrics"]
    segments = L["segments"]
    routes = L["routes"]

    # ── 2 + 4: every metric has a real resolver + a non-empty owner ─────────
    for name, m in metrics.items():
        fn_name = f"_res_{m.resolver}"
        if not hasattr(gb_semantics, fn_name):
            findings.append(Finding(
                check="semantics_coverage", severity="blocking",
                file="semantics/metrics.yaml",
                message=f"metric '{name}' has no resolver function "
                        f"gb_semantics.{fn_name}",
                remediation=f"add a def {fn_name}(entity_id, window_s) -> dict "
                            f"function to gb_semantics.py, or fix the "
                            f"'resolver:' key if it was renamed"))
        if not (m.owner or "").strip():
            findings.append(Finding(
                check="semantics_coverage", severity="blocking",
                file="semantics/metrics.yaml",
                message=f"metric '{name}' has no owner",
                remediation="add an 'owner:' key (one of the GB-* taxonomy "
                            "names or 'Health') — an ungoverned owner means "
                            "no one is accountable for this definition"))

    # ── 3: every segment has a real evaluator ────────────────────────────────
    for name, s in segments.items():
        fn_name = f"_seg_{s.evaluator}"
        if not hasattr(gb_semantics, fn_name):
            findings.append(Finding(
                check="semantics_coverage", severity="blocking",
                file="semantics/segments.yaml",
                message=f"segment '{name}' has no evaluator function "
                        f"gb_semantics.{fn_name}",
                remediation=f"add a def {fn_name}() -> (bool, list[dict]) "
                            f"function to gb_semantics.py, or fix the "
                            f"'evaluator:' key if it was renamed"))

    # ── 5: routes reference real metrics/segments ───────────────────────────
    for r in routes:
        for m in r.metrics:
            if m not in metrics:
                findings.append(Finding(
                    check="semantics_coverage", severity="blocking",
                    file="semantics/routes.yaml",
                    message=f"route '{r.intent}' references unknown metric '{m}'",
                    remediation="fix the metric name in routes.yaml, or add "
                                "it to metrics.yaml if it's a new metric"))
        for s in r.segments:
            if s not in segments:
                findings.append(Finding(
                    check="semantics_coverage", severity="blocking",
                    file="semantics/routes.yaml",
                    message=f"route '{r.intent}' references unknown segment '{s}'",
                    remediation="fix the segment name in routes.yaml, or add "
                                "it to segments.yaml if it's a new segment"))

    # ── 6: source_fields shaped <module>.<ClassName>.<field> must be real ───
    # Universe built from EVERY known class up front (not just ones a metric
    # happens to reference) so never_use validation below can catch a trap
    # field that's real on GpuMetrics even when no metric's source_fields
    # cites GpuMetrics directly (e.g. vram_fill_pct's never_use warns about
    # GpuMetrics.fb_used_pct while its OWN source_fields cite GbSnapshot/
    # dataflux instead).
    known_field_universe: "set[str]" = set()
    for module, cls_name in _KNOWN_CLASSES:
        names = _class_field_names(module, cls_name)
        if names:
            known_field_universe |= names
    # Dataflux EVENT fields belong in the universe too. A trap is often exactly
    # a raw event field , `middle_compacted` reads like the saving a compaction
    # made and says nothing about whether the prefix survived , and those live
    # in gb_dataflux_kinds.py's KindSpecs, not in any dataclass. Without this
    # the checker reported every such trap as possibly-stale, which is the
    # advisory noise that trains a reader to ignore the check (added 2026-08-19
    # with the agent_* metrics).
    known_field_universe |= _dataflux_kind_field_names()

    for name, m in metrics.items():
        for sf in m.source_fields:
            parts = sf.split(".")
            if len(parts) == 3 and parts[1][:1].isupper():
                module, cls_name, field = parts
                if (module, cls_name) not in _KNOWN_CLASSES:
                    continue  # not one of the classes we can fully enumerate
                names = _class_field_names(module, cls_name)
                if names is None:
                    findings.append(Finding(
                        check="semantics_coverage", severity="blocking",
                        file="semantics/metrics.yaml",
                        message=f"metric '{name}': can't introspect "
                                f"{module}.{cls_name} to verify field '{field}'",
                        remediation=f"confirm {module}.{cls_name} still exists "
                                    f"and is a dataclass"))
                    continue
                known_field_universe |= names
                if field not in names:
                    findings.append(Finding(
                        check="semantics_coverage", severity="blocking",
                        file="semantics/metrics.yaml",
                        message=f"metric '{name}': source_fields entry "
                                f"'{sf}' , '{field}' is not a field of "
                                f"{module}.{cls_name}",
                        remediation=f"the field was renamed/removed upstream — "
                                    f"update metrics.yaml's source_fields (real "
                                    f"fields: {sorted(names)})"))

    # ── 7 (advisory): never_use field still exists, best-effort ─────────────
    for name, m in metrics.items():
        for trap in m.never_use:
            field = (trap.get("field") or "").strip()
            # Only a SINGLE bare token (no spaces at all) is checkable here —
            # most never_use.field values are a short prose description
            # ("a log line containing..."), not a clean identifier; splitting
            # off the first word of a sentence produced false positives
            # (leading article "a", etc.) before this whole-string check.
            if not field or " " in field or "." in field:
                continue  # prose or a dotted path — not checkable here
            leading = field
            # A trap field does not have to live in a dataclass. Metrics that
            # read a kernel interface name fields that exist in that FILE
            # (/proc/meminfo's MemFree, a sysfs attribute), and validating those
            # against the dataclass universe reports a real field as stale. Read
            # the file the metric itself declares and look the token up there,
            # so the trap is genuinely validated rather than waved through as
            # advisory noise (GB-Semantics Must-Rule).
            if _token_in_declared_files(m, leading):
                continue
            if known_field_universe and leading not in known_field_universe:
                findings.append(Finding(
                    check="semantics_coverage", severity="advisory",
                    file="semantics/metrics.yaml",
                    message=f"metric '{name}': never_use field '{leading}' "
                            f"doesn't appear in any known dataclass field set "
                            f", may be stale (renamed/removed upstream)",
                    remediation="confirm the trap field still exists under "
                                "that name; update or remove the never_use "
                                "entry if it doesn't"))

    return findings


if __name__ == "__main__":
    fs = run(_REPO_ROOT)
    for f in fs:
        print(f.format(llm="--llm" in sys.argv))
    blocking = [f for f in fs if f.severity == "blocking"]
    print(f"\n{len(fs)} finding(s) ({len(blocking)} blocking)")
    sys.exit(1 if blocking else 0)

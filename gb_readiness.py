#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""gb_readiness.py — GreenBoost's readiness-contract report.

NemoClaw audit, Phase 6b. `greenboost doctor` (greenboost_setup.sh's
cmd_doctor) has always been a thin wrapper over gb_synapse.doctor() — a
flat, prose-shaped dict about the SYNAPSE serving layer specifically
(engine, HF token, cluster, torch env). This module is a separate,
whole-stack readiness report, adopting the SHAPE of NemoClaw's own
system-readiness.schema.json (five parallel ID'd collections, `mutated`
enforced false by the schema, `inconclusive` a genuine third state,
status<->exitCode pinned) — re-implemented from scratch for GreenBoost's
own signals, not a fork. See schemas/readiness.schema.json for the
contract this module's output validates against.

CLI:
    python3 gb_readiness.py doctor [--json]
    greenboost doctor --json    (greenboost_setup.sh delegates here)
"""
from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_DIR))

import gb_ports  # noqa: E402

SCHEMA_VERSION = "1"

# status -> exitCode, pinned. 1 is reserved for "the tool itself broke" and
# is never produced by build_report() itself — only by the CLI wrapper
# when the report couldn't be built at all (see main()).
_STATUS_EXIT_CODE = {
    "supported": 0,
    "incompatible": 2,
    "inconclusive": 3,
}


def _port_listening(port: int, host: str = "127.0.0.1", timeout_s: float = 0.5) -> "bool | None":
    """True/False if determined, None if the check itself failed (e.g. a
    sandboxed environment with no socket access at all) — the difference
    between "determined: not listening" and "undetermined" that
    observation.determined exists to carry."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_s)
        try:
            result = sock.connect_ex((host, port))
            return result == 0
        finally:
            sock.close()
    except OSError:
        return None


def _kmod_loaded() -> "bool | None":
    """/dev/greenboost existing is the same signal the shim and CLAUDE.md's
    own fork-safety notes already treat as authoritative for "is the
    kernel module loaded" — cheaper and more portable than parsing
    `lsmod` output, and doesn't require a subprocess."""
    try:
        return Path("/dev/greenboost").exists()
    except OSError:
        return None


def _shim_present() -> "bool | None":
    candidates = (
        Path("/usr/local/lib/libgreenboost_cuda.so"),
        _REPO_DIR / "libgreenboost_cuda.so",
    )
    try:
        return any(p.is_file() for p in candidates)
    except OSError:
        return None


def _kernel_symbol(name: str) -> "bool | None":
    """Is this symbol exported by the RUNNING kernel?

    GreenBoost's kmod optionally consumes two out-of-tree dma-buf RFCs from
    ~/Dev/kernel_inference (the reclaim-priority hint and the compressed-content
    descriptor), gated at build time by a Kbuild probe against the target
    kernel's dma-buf.h. That probe answers "did this module get compiled with
    the feature", which is not the same question as "does the kernel I am
    booted on have it" — a module built elsewhere, or a kernel rolled back to
    a stock image, makes them disagree. /proc/kallsyms answers the running
    question directly.

    Returns None (undetermined) rather than False when kallsyms cannot be read,
    because an unreadable symbol table is not evidence of absence."""
    try:
        with open("/proc/kallsyms", "r") as fh:
            for line in fh:
                # "<addr> <type> <name>" — match the name field exactly so
                # dma_buf_set_priority does not also match __pfx_ prefixes.
                parts = line.split()
                if len(parts) >= 3 and parts[2] == name:
                    return True
    except OSError:
        return None
    return False


def _obs(id_: str, value, determined: bool = True) -> dict:
    return {"id": id_, "value": value, "determined": determined}


def _cap(id_: str, available: bool, reason: str = "") -> dict:
    d = {"id": id_, "available": available}
    if reason:
        d["reason"] = reason
    return d


def _finding(id_: str, severity: str, message: str, related_observation: str = "") -> dict:
    d = {"id": id_, "severity": severity, "message": message}
    if related_observation:
        d["relatedObservation"] = related_observation
    return d


def _evidence(id_: str, source: str, detail: str = "") -> dict:
    d = {"id": id_, "source": source}
    if detail:
        d["detail"] = detail
    return d


def _qual(id_: str, passed: bool, detail: str = "") -> dict:
    """Qualification: can a workload or capability be used on this system?
    id: e.g. 'qualification.reference_workload_servable'
    passed: True if qualified, False if unqualified
    detail: optional explanation (why it passed or why it failed)"""
    d = {"id": id_, "passed": passed}
    if detail:
        d["detail"] = detail
    return d


def _reference_errors(report: dict) -> list[str]:
    """Validate reference integrity: no duplicate ids within collections,
    and all references (relatedObservation, etc.) resolve to real entries.
    Returns list of error strings; empty list means clean."""
    errors = []

    # Collect all ids per collection for dedup and reference checking
    collection_ids = {
        "observations": {},
        "capabilities": {},
        "qualifications": {},
        "findings": {},
        "evidence": {},
    }

    # Check for duplicate ids within each collection
    for collection_name, items in [
        ("observations", report.get("observations", [])),
        ("capabilities", report.get("capabilities", [])),
        ("qualifications", report.get("qualifications", [])),
        ("findings", report.get("findings", [])),
        ("evidence", report.get("evidence", [])),
    ]:
        seen_ids = set()
        for item in items:
            item_id = item.get("id")
            if item_id:
                if item_id in seen_ids:
                    errors.append(
                        f"duplicate id in {collection_name}: '{item_id}' appears more than once"
                    )
                else:
                    seen_ids.add(item_id)
                    collection_ids[collection_name][item_id] = item

    # Check relatedObservation references in findings
    for finding in report.get("findings", []):
        related_obs = finding.get("relatedObservation")
        if related_obs and related_obs not in collection_ids["observations"]:
            errors.append(
                f"finding '{finding.get('id')}' references non-existent observation '{related_obs}'"
            )

    return errors


def build_report(node: str = "") -> dict:
    """Read-only (mutated=False always — this function has no code path
    that writes anything). Never raises on a missing/unreachable signal:
    every check below degrades to `determined=False` rather than
    propagating an exception, so the WORST case is an inconclusive report,
    never a crash — the exact "system_status erroring is itself a finding,
    not a tool bug" principle CLAUDE.md already documents for
    system_status, generalized to every observation here."""
    observations = []
    capabilities = []
    qualifications = []
    findings = []
    evidence = []

    kmod = _kmod_loaded()
    observations.append(_obs("kmod.greenboost.loaded", kmod, determined=kmod is not None))
    evidence.append(_evidence("kmod.greenboost.loaded", "os.path.exists", "/dev/greenboost"))
    if kmod is False:
        findings.append(_finding(
            "kmod_not_loaded", "blocking",
            "greenboost.ko is not loaded — /dev/greenboost does not exist. "
            "GPU-shim-backed features (T2/T3 tiering, cluster fabric) are "
            "unavailable until `sudo greenboost load` (or `modprobe greenboost`).",
            related_observation="kmod.greenboost.loaded",
        ))
    elif kmod is None:
        findings.append(_finding(
            "kmod_state_undetermined", "warning",
            "Could not determine whether greenboost.ko is loaded (filesystem check failed).",
            related_observation="kmod.greenboost.loaded",
        ))

    shim = _shim_present()
    observations.append(_obs("shim.so.present", shim, determined=shim is not None))

    # Kernel-side features GreenBoost can use when present. Reported whether or
    # not they are there: their absence is a normal, supported configuration
    # (stock kernel), and stating it is how a session learns the answer without
    # re-deriving it from /proc/kallsyms by hand.
    for obs_id, symbol in (
        ("kernel.dmabuf.priority_hint", "dma_buf_set_priority"),
        ("kernel.dmabuf.compression_descriptor", "dma_buf_set_compression"),
    ):
        present = _kernel_symbol(symbol)
        observations.append(_obs(obs_id, present, determined=present is not None))
        evidence.append(_evidence(obs_id, "/proc/kallsyms", symbol))

    # Port collision detection
    ports_to_check = {
        "netd": gb_ports.NETD_PORT,
        "a2a": gb_ports.A2A_PORT,
        "dataflux_ui": gb_ports.DATAFLUX_UI_PORT,
        "exporter": gb_ports.EXPORTER_PORT,
        "synapse": gb_ports.SYNAPSE_PORT,
    }
    collisions = gb_ports.validate_no_collisions(ports_to_check)
    if collisions:
        for label_a, label_b, port in collisions:
            findings.append(_finding(
                "ports_collision", "blocking",
                f"Port collision detected: {label_a} and {label_b} both "
                f"configured for port {port}. Ensure each service has a unique port.",
            ))
    observations.append(_obs(
        "ports.no_collisions", len(collisions) == 0,
        determined=True
    ))

    # Service liveness checks
    port_checks = {
        "netd.port_listening": gb_ports.NETD_PORT,
        "a2a.port_listening": gb_ports.A2A_PORT,
        "synapse.port_listening": gb_ports.SYNAPSE_PORT,
    }
    for obs_id, port in port_checks.items():
        listening = _port_listening(port)
        observations.append(_obs(obs_id, listening, determined=listening is not None))
        evidence.append(_evidence(obs_id, "socket.connect_ex", f"127.0.0.1:{port}"))

    actuate_env = os.environ.get("GB_ORCH_ACTUATE", "0") == "1"
    capabilities.append(_cap(
        "capability.tier_actuate", actuate_env,
        reason="" if actuate_env else "GB_ORCH_ACTUATE unset — actuation tools run gated/dry-run only",
    ))
    capabilities.append(_cap(
        "capability.rpc_split", bool(kmod),
        reason="" if kmod else "requires the kernel module for local T1/T2/T3 tiering the --rpc split relies on",
    ))
    capabilities.append(_cap(
        "capability.frontload_split", bool(kmod),
        reason="" if kmod else "GB_VRAM_FRONTLOAD needs the kernel module's DMA-BUF pinning",
    ))

    # Populate qualifications based on capabilities. The qualification schema
    # is passed:bool only (additionalProperties:false — no third "unknown"
    # state, unlike observation.determined), so kmod=None (undetermined) and
    # kmod=False (confirmed absent) both land on passed=False here — but the
    # detail text below still distinguishes them, so a reader isn't misled
    # into thinking an undetermined box has been definitively disqualified.
    _kmod_detail = {
        None: "kernel module state could not be determined "
              "(see observation kmod.greenboost.loaded)",
        False: "kernel module confirmed not loaded",
    }

    # qualification.reference_workload_servable: can the reference inference workload run?
    # Requires kmod loaded and shim present; kmod=None means undetermined so we don't qualify
    ref_workload_qualified = kmod is True and shim is True
    if ref_workload_qualified:
        _ref_detail = ""
    elif kmod is not True:
        _ref_detail = _kmod_detail[kmod] + "; also requires shim library present"
    else:
        _ref_detail = "requires shim library present"
    qualifications.append(_qual(
        "qualification.reference_workload_servable",
        ref_workload_qualified,
        detail=_ref_detail,
    ))

    # qualification.cluster_rpc_split: can we use --rpc tensor split across feeder GPUs?
    # Requires capability.rpc_split
    rpc_qualified = kmod is True
    qualifications.append(_qual(
        "qualification.cluster_rpc_split",
        rpc_qualified,
        detail="" if rpc_qualified else
            "requires kernel module for tiering backend — " + _kmod_detail[kmod]
    ))

    # qualification.frontload_split_effective: can we use front-loaded VRAM split?
    # Requires capability.frontload_split
    frontload_qualified = kmod is True
    qualifications.append(_qual(
        "qualification.frontload_split_effective",
        frontload_qualified,
        detail="" if frontload_qualified else
            "requires kernel module for DMA-BUF pinning — " + _kmod_detail[kmod]
    ))

    if kmod is False:
        status = "incompatible"
    elif kmod is None:
        status = "inconclusive"
    else:
        status = "supported"

    return {
        "schemaVersion": SCHEMA_VERSION,
        "status": status,
        "exitCode": _STATUS_EXIT_CODE[status],
        "mutated": False,
        "provenance": {
            "tool": "gb_readiness.py",
            "generatedAtEpochS": time.time(),
            "node": node or socket.gethostname(),
        },
        "observations": observations,
        "capabilities": capabilities,
        "qualifications": qualifications,
        "findings": findings,
        "evidence": evidence,
    }


def main(argv: "list[str] | None" = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] != "doctor":
        print("usage: gb_readiness.py doctor [--json]", file=sys.stderr)
        return 1

    try:
        report = build_report()
    except Exception as e:
        print(f"ERROR: gb_readiness.py itself failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    # Validate reference integrity: catch internal inconsistencies before output
    ref_errors = _reference_errors(report)
    if ref_errors:
        print("ERROR: report validation failed (internal inconsistency):", file=sys.stderr)
        for error in ref_errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    if "--json" in argv:
        import json
        print(json.dumps(report))
    else:
        print(f"status: {report['status']} (exit {report['exitCode']})")
        for obs in report["observations"]:
            mark = "?" if not obs["determined"] else ("✓" if obs["value"] else "✗")
            print(f"  {mark} {obs['id']}: {obs['value']}")
        for f in report["findings"]:
            print(f"  [{f['severity']}] {f['message']}")
    return report["exitCode"]


if __name__ == "__main__":
    sys.exit(main())

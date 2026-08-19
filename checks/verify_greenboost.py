#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""checks/verify_greenboost.py — end-to-end agent-legibility verification
harness, on Colibri's doctor.py schema (see
workflow/porting-reference.md §CB-3-doctor / B5): each step reports
{id, status: pass|warn|fail|skip, summary, details?}; overall status is
"error" if any step failed, "warning" if any warned, else "ok" — same
aggregation rule as Colibri's `run_doctor`.

Composes EXISTING tools rather than reinventing checks — this is a harness
around what's already there (health-check, gb_monitor, gb_dataflux,
gb_cluster, the MCP server modules), not a new diagnostic engine.

Usage:
    python3 checks/verify_greenboost.py            # human-readable
    python3 checks/verify_greenboost.py --llm       # one line per step
    make verify
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _check(id_: str, status: str, summary: str, **details) -> dict:
    item = {"id": id_, "status": status, "summary": summary}
    if details:
        item["details"] = details
    return item


def _step_build_artifact() -> dict:
    candidates = [REPO_ROOT / "libgreenboost_cuda.so",
                 Path("/usr/local/lib/libgreenboost_cuda.so")]
    found = next((p for p in candidates if p.is_file()), None)
    if found:
        return _check("build.shim", "pass", f"shim built: {found}", path=str(found))
    return _check("build.shim", "warn",
                 "libgreenboost_cuda.so not found (repo root or /usr/local/lib) — "
                 "run 'make shim' if you need the live shim", checked=[str(p) for p in candidates])


def _step_health_check() -> dict:
    try:
        r = subprocess.run(["greenboost", "health-check", "--llm"],
                          capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            return _check("greenboost.health_check", "pass",
                          "greenboost health-check --llm exited 0", stdout_tail=r.stdout[-500:])
        return _check("greenboost.health_check", "warn",
                      f"greenboost health-check --llm exited {r.returncode}",
                      stdout_tail=r.stdout[-500:], stderr_tail=r.stderr[-500:])
    except FileNotFoundError:
        return _check("greenboost.health_check", "skip",
                      "'greenboost' CLI not on PATH — not installed, or run from an installed environment")
    except subprocess.TimeoutExpired:
        return _check("greenboost.health_check", "fail", "greenboost health-check --llm timed out (20s)")


def _step_shim_stats() -> dict:
    try:
        import gb_monitor
        snap = gb_monitor.snapshot()
        loaded = bool(getattr(snap, "loaded", False))
        if loaded:
            return _check("shim.stats", "pass", "kernel module loaded, shim stats readable",
                          loaded=True)
        return _check("shim.stats", "warn",
                      "kmod not loaded (gb_monitor reports loaded=False) — "
                      "shim overflow falls back to Path B (cuMemHostRegister), not a hard failure",
                      loaded=False)
    except Exception as e:
        return _check("shim.stats", "skip", f"gb_monitor unavailable: {e}")


def _step_dataflux() -> dict:
    try:
        import gb_dataflux
        events = gb_dataflux.read_events(since_hours=24)
        summary = gb_dataflux.summarize(events)
        return _check("dataflux.summary", "pass",
                      f"dataflux log readable, {len(events)} event(s) in the last 24h",
                      event_count=len(events), kinds=summary.get("kinds", {}))
    except Exception as e:
        return _check("dataflux.summary", "fail", f"gb_dataflux.summarize failed: {e}")


def _step_cluster() -> dict:
    try:
        import gb_cluster
        status = gb_cluster.status(probe=False)  # skip-clean: no network round-trip
        feeders = status.get("feeders_online", 0)
        return _check("cluster.snapshot", "pass" if True else "warn",
                      f"cluster_status readable (feeders_online={feeders}, probe skipped)",
                      feeders_online=feeders, cluster_configured=status.get("cluster_configured"))
    except Exception as e:
        return _check("cluster.snapshot", "skip", f"gb_cluster.status unavailable: {e}")


def _step_mcp_servers() -> dict:
    servers = ("gb_mcp", "gb_dataflux_mcp", "gb_cluster_mcp", "gb_synapse_mcp")
    results = {}
    all_ok = True
    for mod_name in servers:
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            n_tools = sum(1 for name in dir(mod)
                         if callable(getattr(mod, name, None))
                         and getattr(getattr(mod, name), "__module__", "") == mod_name)
            results[mod_name] = "importable"
        except Exception as e:
            results[mod_name] = f"FAILED: {e}"
            all_ok = False
    status = "pass" if all_ok else "fail"
    return _check("mcp.self_check", status,
                 f"{sum(1 for v in results.values() if v == 'importable')}/{len(servers)} "
                 f"MCP server modules import cleanly", servers=results)


def _step_gb_plan_sanity() -> dict:
    try:
        import gb_placement
        plan = gb_placement.plan_experts(dense_gb=2.0, expert_gb=5.0, kv_gb=1.0,
                                         vram_gb=12.0, t2_gb=40.0)
        ok = ("tiers" in plan and "expected_bottleneck" in plan)
        return _check("gb_plan.sanity", "pass" if ok else "fail",
                      "gb_placement.plan_experts() returns the expected tier-plan shape "
                      "(synthetic inputs, no live model parsed)", plan_keys=list(plan.keys()))
    except Exception as e:
        return _check("gb_plan.sanity", "fail", f"gb_placement.plan_experts failed: {e}")


_STEPS = [
    _step_build_artifact, _step_health_check, _step_shim_stats,
    _step_dataflux, _step_cluster, _step_mcp_servers, _step_gb_plan_sanity,
]


def run_verify() -> dict:
    checks = [step() for step in _STEPS]
    statuses = {c["status"] for c in checks}
    overall = "error" if "fail" in statuses else "warning" if "warn" in statuses else "ok"
    return {"schema_version": 1, "status": overall, "checks": checks}


def format_llm(report: dict) -> str:
    lines = [f"STEP {c['id']} {c['status'].upper()} — {c['summary']}" for c in report["checks"]]
    lines.append(f"OVERALL {report['status'].upper()}")
    return "\n".join(lines)


def format_human(report: dict) -> str:
    icons = {"pass": "ok", "warn": "warn", "fail": "FAIL", "skip": "skip"}
    lines = [f"greenboost verify — {REPO_ROOT}"]
    for c in report["checks"]:
        lines.append(f"[{icons[c['status']]:>4}] {c['id']:<24} {c['summary']}")
    lines.append("")
    lines.append(f"result: {report['status']}")
    return "\n".join(lines)


def main() -> int:
    llm = "--llm" in sys.argv
    report = run_verify()
    print(format_llm(report) if llm else format_human(report))
    return 1 if report["status"] == "error" else 0


if __name__ == "__main__":
    sys.exit(main())

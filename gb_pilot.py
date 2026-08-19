#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_pilot.py — the pilot's instrument panel over GreenBoost's dataflux log.

Reads the flight recorder (gb_dataflux: stage_profile wall-times from ai-forge,
tok_s_measured decode speeds, system snapshots with VRAM/T2/T3/KV pressure) and
turns trends into evidence-backed advice: which GbControl lever to move, which
stage regressed, where an optimization (GB_TQ_ATTN, re-quant) would pay.

**Read-only v1.** No lever is ever moved from here — advice names the exact
`GbControl` call a pilot (or greenboost-cli / a future orchestrator Loop T)
would make, with the evidence attached. Actuation stays behind gb_control's
GB_ORCH_ACTUATE gate, deliberately outside this module until the instruments
have accumulated real history.

CLI (UI paradigm: TTY alt-screen loop @5s, one-shot when piped, --llm):
    python3 gb_pilot.py            # interactive panel
    python3 gb_pilot.py --llm      # compact key=value for an agent
    python3 gb_pilot.py --json     # full analysis dict
    python3 gb_pilot.py --days 2   # analysis window
"""
from __future__ import annotations

import json
import os
import sys
import time

from gb_advisories import (
    Advisory, AdvisorySeverity, AdvisoryPhase, AdvisoryKind,
)

# Trend thresholds (fractions). A stage is flagged when its latest wall time
# exceeds the rolling median by REGRESSION_PCT; a model when its latest tok/s
# drops below (1 - REGRESSION_PCT) of its average.
REGRESSION_PCT = 0.20
MIN_SAMPLES = 3          # below this, trends are noise — report, don't flag

# Decode rate needs a bigger population than a stage wall-time does before a
# latest-vs-median comparison means anything. Two reasons, both measured on
# this box: a tok/s sample is duration-blind (see the long note in
# _analyze_models below — median 13.9 against a max of 283.8 at n=41), and the
# sample floor that keeps short replies out (_MIN_TOK_S_SAMPLE_TOKENS, 24
# tokens) discards a large share of real agentic traffic, so what survives is
# both few and skewed. On 2026-08-18 this produced five standing "decode
# degraded" warnings built on n=3, n=5 and n=9 — noise presented as findings,
# each carrying a real retune lever. Stage timings keep MIN_SAMPLES: they are
# wall-clock measurements of the same unit of work, not rate estimates.
MIN_TOK_S_SAMPLES = 8
REBALANCE_DROP_PCT = 25.0  # DI-13: stricter than REGRESSION_PCT — a rebalance
                           # advisory is a bigger ask (re-serve) than a generic warn


# ── analysis (pure functions over event lists — the testable core) ──────────

def _median(xs: list) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def analyze(events: list) -> dict:
    """Distil raw dataflux events into per-stage / per-model trends plus the
    latest memory-pressure picture. Pure; tolerates malformed rows."""
    stages: dict = {}
    models: dict = {}
    last_snap: dict = {}
    last_split: dict = {}
    errors: list = []

    for ev in events:
        if not isinstance(ev, dict):
            continue
        kind = ev.get("kind")
        if kind == "tensor_split":
            if ev.get("ts", 0) >= last_split.get("ts", 0):
                last_split = ev
        if kind == "stage_profile":
            st = ev.get("stage", "?")
            s = stages.setdefault(st, {"walls": [], "last_status": "",
                                       "last_ts": 0, "vram_peak_mb": 0})
            try:
                s["walls"].append(float(ev.get("duration_s", 0.0)))
            except (TypeError, ValueError):
                continue
            s["last_status"] = ev.get("status", "")
            s["last_ts"] = ev.get("ts", 0)
            if ev.get("vram_peak_mb"):
                s["vram_peak_mb"] = ev["vram_peak_mb"]
            if ev.get("status") == "error":
                errors.append({"stage": st, "ts": ev.get("ts", 0)})
        elif kind == "tok_s_measured":
            # A sample the writer already rejected is not a measurement.
            # gb_synapse.record_measured_tok_s() emits over-ceiling samples with
            # status="error" for auditability; counting them here put an
            # impossible 21065.2 tok/s into a 34-sample mean and reported
            # avg=649.7 for a model that really runs at ~5 (live, 2026-08-17).
            # That is not cosmetic: the inflated mean made `drop_pct` cross
            # REGRESSION_PCT and raised a "decode degraded" advisory carrying a
            # set_kv_size_threshold_mb lever, so bad data could have driven a
            # real retune. The stage_profile branch above already checks status;
            # this branch simply never did.
            if ev.get("status") == "error":
                continue
            # Key the way gb_dataflux.summarize() does. The same model served
            # from a different vantage point ([proxy] vs [cli]) or at a
            # different quant/ctx/kv_type is a different throughput population;
            # blending them yields an average that matches neither (the real
            # 2026-08-01 incident: proxy=0.3, cli=2.4, engine truth=2.18,
            # blended=0.6). summarize() was fixed for this; gb_pilot was not,
            # so its advisories were still computed on blended series.
            _src = ev.get("source") or ""
            _quant = ev.get("quant") or ""
            m = (f"[{_src}]" if _src else "") + str(ev.get("model", "?"))
            if _quant:
                m += (f"::{_quant}::{int(ev.get('ctx') or 0)}"
                      f"::{ev.get('kv_type') or ''}")
            d = models.setdefault(m, {"samples": []})
            try:
                d["samples"].append(float(ev.get("tok_s", 0.0)))
            except (TypeError, ValueError):
                continue
        elif kind == "snapshot":
            if ev.get("ts", 0) >= last_snap.get("ts", 0):
                last_snap = ev

    stage_out: dict = {}
    for st, s in stages.items():
        walls = s["walls"]
        med = _median(walls[:-1]) if len(walls) > 1 else _median(walls)
        latest = walls[-1] if walls else 0.0
        reg = ((latest - med) / med) if med > 0 else 0.0
        stage_out[st] = {
            "count": len(walls),
            "median_s": round(med, 2),
            "latest_s": round(latest, 2),
            "regression_pct": round(reg * 100, 1),
            "regressed": len(walls) >= MIN_SAMPLES and reg > REGRESSION_PCT,
            "last_status": s["last_status"],
            "vram_peak_mb": s["vram_peak_mb"],
        }

    model_out: dict = {}
    for m, d in models.items():
        xs = d["samples"]
        avg = sum(xs) / len(xs) if xs else 0.0
        # Baseline on the MEDIAN, matching the stage branch above (`median_s`)
        # rather than the mean this branch used to use.
        #
        # A tok/s sample is duration-blind: a short reply timed over a few ms
        # scores hundreds of tok/s and weighs the same as a long generation.
        # Even after the two fixes this module already carries (dropping
        # status="error" samples, and keying by source+quant+ctx+kv_type), the
        # surviving ok-status samples remain wildly skewed. Measured on the
        # reference workload 2026-08-17, n=41 at ctx=32768: min 1.7, median
        # 13.9, mean 33.2, max 283.8 — the mean sat 2.4x above the median, so a
        # perfectly normal 17.8 tok/s sample scored a "46% regression" and
        # raised an advisory carrying a real set_kv_size_threshold_mb retune
        # lever. The outliers' own source (a missing sample floor on the
        # non-streaming proxy path) is fixed separately in gb_synapse_api.py;
        # this makes the detector robust to skew regardless of origin.
        #
        # `avg` stays in the output for backward compatibility, but nothing
        # decides on it any more.
        ordered = sorted(xs)
        n = len(ordered)
        median = (0.0 if not n else
                  ordered[n // 2] if n % 2
                  else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0)
        latest = xs[-1] if xs else 0.0
        drop = ((median - latest) / median) if median > 0 else 0.0
        model_out[m] = {
            "samples": len(xs),
            "avg": round(avg, 1),
            "median": round(median, 1),
            "latest": round(latest, 1),
            "drop_pct": round(drop * 100, 1),
            "degraded": len(xs) >= MIN_TOK_S_SAMPLES and drop > REGRESSION_PCT,
            # Distinguishes "measured, looks fine" from "not enough evidence to
            # say either way". Without this the two are indistinguishable in the
            # output, and a reader treats an unevaluated model as a healthy one.
            "insufficient_samples": len(xs) < MIN_TOK_S_SAMPLES,
        }

    return {
        "stages": stage_out,
        "models": model_out,
        "pressure": {
            "t2_pressure": last_snap.get("t2_pressure", 0),
            "t3_used_mb": last_snap.get("t3_used_mb", 0),
            "fb_used_pct": last_snap.get("fb_used_pct", 0),
            "kv_used_mb": last_snap.get("kv_used_mb", 0),
            "kv_reserve_mb": last_snap.get("kv_reserve_mb", 0),
            "shim_phase": last_snap.get("shim_phase", ""),
            "snapshot_ts": last_snap.get("ts", 0),
        },
        "last_split": {
            "nodes": last_split.get("nodes", 1),
            "split": last_split.get("split", ""),
            "v3": last_split.get("v3", False),
            "ts": last_split.get("ts", 0),
        },
        "errors": errors[-10:],
        "event_count": len(events),
    }


def collect_advisories(analysis: dict) -> list[Advisory]:
    """Generate structured Advisory objects from analysis findings.

    Returns advisories with stable IDs, severity, and phase for deduplication
    across multiple runs. Maintains the exact same diagnostic logic as advise().

    Args:
        analysis: dict from analyze() with pressure, stages, models, last_split

    Returns:
        List of Advisory objects, or empty list if analysis is nominal
    """
    out: list[Advisory] = []
    press = analysis.get("pressure", {})

    # T3 spill is the single worst signal — inference goes ~100× slower.
    if press.get("t3_used_mb", 0) > 0:
        out.append(Advisory(
            id="pilot.t3_spill",
            severity=AdvisorySeverity.BLOCKING,
            phase=AdvisoryPhase.RUNTIME_TIER,
            title="NVMe tier in use (T3 spill)",
            reason=f"T3 NVMe holds {press['t3_used_mb']} MB",
            commands=("grep 'GB_VRAM_FRONTLOAD' ~/.bashrc || echo 'not set'",),
            docs_url="https://docs.greenboost.io/rule-1-vram",
            resume_safe=False,
            kind=AdvisoryKind.INFO,
        ))

    if press.get("t2_pressure", 0) == 2:
        out.append(Advisory(
            id="pilot.t2_pressure_critical",
            severity=AdvisorySeverity.WARNING,
            phase=AdvisoryPhase.RUNTIME_TIER,
            title="T2 DDR memory pressure CRITICAL",
            reason="T2 DDR pressure CRITICAL in latest snapshot",
            commands=(
                "sudo sysctl -a | grep greenboost",
                "ps aux | grep ollama",
            ),
            docs_url="https://docs.greenboost.io/t2-pressure",
            resume_safe=False,
            kind=AdvisoryKind.INFO,
        ))

    for st, s in sorted(analysis.get("stages", {}).items()):
        if s.get("regressed"):
            out.append(Advisory(
                id=f"pilot.stage_regression.{st}",
                severity=AdvisorySeverity.WARNING,
                phase=AdvisoryPhase.RUNTIME_OPTIMIZE,
                title=f"Stage '{st}' wall-time regressed",
                reason=(f"{st}: latest {s['latest_s']}s vs median "
                        f"{s['median_s']}s ({s['regression_pct']:+.0f}%, "
                        f"n={s['count']})"),
                commands=("gb dataflux summary",),
                resume_safe=False,
                kind=AdvisoryKind.INFO,
            ))
        if s.get("last_status") == "error":
            out.append(Advisory(
                id=f"pilot.stage_error.{st}",
                severity=AdvisorySeverity.WARNING,
                phase=AdvisoryPhase.RUNTIME_OPTIMIZE,
                title=f"Stage '{st}' execution failed",
                reason=f"{st}: last run ended in error",
                commands=("gb dataflux events --kind=stage_profile --status=error",),
                resume_safe=False,
                kind=AdvisoryKind.INFO,
            ))

    for m, d in sorted(analysis.get("models", {}).items()):
        if d.get("degraded"):
            out.append(Advisory(
                id=f"pilot.tok_s_drop.{m}",
                severity=AdvisorySeverity.WARNING,
                phase=AdvisoryPhase.RUNTIME_OPTIMIZE,
                title=f"Model '{m}' decode speed degraded",
                reason=(f"{m}: latest {d['latest']} tok/s vs median {d['median']} "
                        f"({d['drop_pct']:.0f}% down, n={d['samples']})"),
                commands=("gb synapse recommend --model {m}",),
                resume_safe=False,
                kind=AdvisoryKind.INFO,
            ))
            # DI-13 rebalance advisory: only when drop clears REBALANCE_DROP_PCT
            # AND a multi-node cluster split is active (nodes>1).
            split = analysis.get("last_split", {})
            if d["drop_pct"] > REBALANCE_DROP_PCT and split.get("nodes", 1) > 1:
                out.append(Advisory(
                    id=f"pilot.rebalance_advice.{m}",
                    severity=AdvisorySeverity.WARNING,
                    phase=AdvisoryPhase.CLUSTER_DISPATCH,
                    title=f"Model '{m}' tensor-split may need rebalancing",
                    reason=(f"{m}: {d['drop_pct']:.0f}% tok/s drop with a "
                            f"{split.get('nodes')}-node split active "
                            f"({split.get('split', '?')})"),
                    commands=(
                        "export GB_SYNAPSE_SPLIT_V3=1",
                        "gb synapse serve <model>",
                    ),
                    resume_safe=False,
                    kind=AdvisoryKind.MANUAL,
                ))

    # Return empty list if nominal (no advisories) — the caller decides whether
    # to emit an "all clear" advisory
    return out


def advise(analysis: dict) -> list:
    """Map analysis findings to concrete, evidence-backed actions. Each item:
    {severity, topic, evidence, action, lever} — `lever` is either None (the
    action is a config/env change, not a GbControl call) or a structured
    dict `{"call": "<GbControl method name>", "args": [...] | None,
    "kwargs": {...}}`. `args is None` means the lever names the right method
    but this v1 analysis doesn't have enough state (e.g. the CURRENT reserve
    value, needed to compute a bounded relative bump) to compute safe
    arguments — gb_mcp.optimize_inference's apply loop treats that as
    NOT auto-appliable, not as a string to exec. Never build a lever whose
    `args` looks plausible but isn't grounded in read state.

    DEPRECATED: Use collect_advisories() for new code to get typed Advisory objects.
    This function is kept for backward compatibility."""
    press = analysis.get("pressure", {})
    out: list = []

    # T3 spill is the single worst signal — inference goes ~100× slower.
    if press.get("t3_used_mb", 0) > 0:
        out.append({
            "severity": "critical", "topic": "t3_spill",
            "evidence": f"T3 NVMe holds {press['t3_used_mb']} MB",
            "action": ("Weights/KV are spilling to NVMe. Re-run quantize_to_fit "
                       "at a tighter budget or enable GB_TQ_ATTN so the working "
                       "set fits T1/T2."),
            "lever": None,
        })

    if press.get("t2_pressure", 0) == 2:
        out.append({
            "severity": "warn", "topic": "t2_pressure",
            "evidence": "T2 DDR pressure CRITICAL in latest snapshot",
            "action": ("Free host headroom for the pool: raise the workstation "
                       "reserve so desktop apps stop competing (manual — needs "
                       "the current reserve value, not auto-appliable here), "
                       "and throttle prefetch while pressure holds (auto)."),
            "lever": {"call": "set_prefetch_throttle", "args": [True],
                      "kwargs": {"reason": "t2_pressure_critical"}},
        })

    for st, s in sorted(analysis.get("stages", {}).items()):
        if s.get("regressed"):
            out.append({
                "severity": "warn", "topic": "stage_regression",
                "evidence": (f"{st}: latest {s['latest_s']}s vs median "
                             f"{s['median_s']}s ({s['regression_pct']:+.0f}%, "
                             f"n={s['count']})"),
                "action": (f"Stage '{st}' slowed beyond {int(REGRESSION_PCT*100)}%. "
                           "Check VRAM pressure at run time; if T2-bound, consider "
                           "GB_TQ_ATTN for this stage's pipeline or a smaller "
                           "quant budget."),
                "lever": None,
            })
        if s.get("last_status") == "error":
            out.append({
                "severity": "warn", "topic": "stage_error",
                "evidence": f"{st}: last run ended in error",
                "action": f"Inspect the latest '{st}' job log before tuning anything.",
                "lever": None,
            })

    for m, d in sorted(analysis.get("models", {}).items()):
        if d.get("degraded"):
            out.append({
                "severity": "warn", "topic": "tok_s_drop",
                "evidence": (f"{m}: latest {d['latest']} tok/s vs median {d['median']} "
                             f"({d['drop_pct']:.0f}% down, n={d['samples']})"),
                "action": (f"Decode speed for {m} degraded. Check for a competing "
                           "GPU load or KV spill to T2; gb_synapse recommend() "
                           "now has measured history to re-pick quant/split."),
                "lever": {"call": "set_kv_size_threshold_mb", "args": None,
                          "kwargs": {"reason": "tok_s_drop"}},
            })
            split = analysis.get("last_split", {})
            if d["drop_pct"] > REBALANCE_DROP_PCT and split.get("nodes", 1) > 1:
                out.append({
                    "severity": "warn", "topic": "rebalance_advice",
                    "evidence": (f"{m}: {d['drop_pct']:.0f}% tok/s drop with a "
                                f"{split.get('nodes')}-node split active "
                                f"({split.get('split', '?')})"),
                    "action": (f"Live tok/s for {m} diverged from its own history by "
                              f"more than the cluster's placement-time estimate would "
                              f"predict — the current tensor-split may no longer match "
                              f"each node's real throughput. Re-serve with "
                              f"GB_SYNAPSE_SPLIT_V3=1 (link-quality-weighted split) "
                              f"and compare via optimize_inference(measure=true)."),
                    "lever": None,
                })

    if not out:
        n = analysis.get("event_count", 0)
        out.append({
            "severity": "ok", "topic": "all_clear",
            "evidence": f"{n} events analysed, no regressions or pressure flags",
            "action": "No action needed. Instruments nominal.",
            "lever": None,
        })
    return out


# ── CLI (UI paradigm: snapshot fn + TTY loop + --llm) ────────────────────────

def _read(days: float) -> list:
    import gb_dataflux
    return gb_dataflux.read_events(since_hours=days * 24)


def _cmd_pilot_snapshot(days: float) -> str:
    """Render one panel frame (plain text, %-25s columns, no vertical pipes)."""
    events = _read(days)
    a = analyze(events)
    adv = advise(a)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"GreenBoost Pilot — instruments over last {days:g}d    {ts}",
             "─" * 72]
    p = a["pressure"]
    lines.append(f"{'phase':<25s} {p['shim_phase'] or '-'}")
    lines.append(f"{'VRAM used %':<25s} {p['fb_used_pct']}")
    lines.append(f"{'T2 pressure':<25s} {['ok','warn','CRITICAL'][min(2, int(p['t2_pressure'] or 0))]}")
    lines.append(f"{'T3 spill MB':<25s} {p['t3_used_mb']}")
    lines.append(f"{'KV used/reserve MB':<25s} {p['kv_used_mb']}/{p['kv_reserve_mb']}")
    lines.append("")
    lines.append(f"{'STAGE':<32s} {'n':>4s} {'median_s':>9s} {'latest_s':>9s} {'trend':>8s}")
    for st, s in sorted(a["stages"].items()):
        flag = " ← REGRESSED" if s["regressed"] else ""
        lines.append(f"{st[:31]:<32s} {s['count']:>4d} {s['median_s']:>9.2f} "
                     f"{s['latest_s']:>9.2f} {s['regression_pct']:>+7.1f}%{flag}")
    if not a["stages"]:
        lines.append("  (no stage_profile events yet — run a generation)")
    lines.append("")
    lines.append(f"{'MODEL tok/s':<32s} {'n':>4s} {'avg':>9s} {'latest':>9s}")
    for m, d in sorted(a["models"].items()):
        flag = " ← DEGRADED" if d["degraded"] else ""
        lines.append(f"{m[:31]:<32s} {d['samples']:>4d} {d['avg']:>9.1f} "
                     f"{d['latest']:>9.1f}{flag}")
    if not a["models"]:
        lines.append("  (no tok_s_measured events yet — run an Ollama critic/turn)")
    lines.append("")
    lines.append("ADVICE")
    for item in adv:
        lines.append(f"  [{item['severity']:<8s}] {item['action']}")
        lines.append(f"             evidence: {item['evidence']}")
        if item["lever"]:
            lv = item["lever"]
            appliable = lv.get("args") is not None
            call_str = f"{lv['call']}({', '.join(map(repr, lv.get('args') or []))})"
            tag = "auto-appliable" if appliable else "manual — needs current state first"
            lines.append(f"             lever:    {call_str}   ({tag}; NOT applied here — v1 is read-only)")
    return "\n".join(lines)


def _print_llm(days: float) -> None:
    a = analyze(_read(days))
    adv = advise(a)
    p = a["pressure"]
    print(f"events={a['event_count']}")
    print(f"phase={p['shim_phase']}")
    print(f"t2_pressure={p['t2_pressure']}")
    print(f"t3_used_mb={p['t3_used_mb']}")
    for st, s in sorted(a["stages"].items()):
        print(f"stage.{st}.median_s={s['median_s']}")
        print(f"stage.{st}.latest_s={s['latest_s']}")
        print(f"stage.{st}.regressed={int(s['regressed'])}")
    for m, d in sorted(a["models"].items()):
        print(f"model.{m}.avg_tok_s={d['avg']}")
        print(f"model.{m}.latest_tok_s={d['latest']}")
        print(f"model.{m}.degraded={int(d['degraded'])}")
    for i, item in enumerate(adv):
        print(f"advice.{i}.severity={item['severity']}")
        print(f"advice.{i}.topic={item['topic']}")
        print(f"advice.{i}.action={item['action']}")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="GreenBoost pilot instrument panel")
    ap.add_argument("--days", type=float, default=5.0)
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.json:
        a = analyze(_read(args.days))
        a["advice"] = advise(a)
        print(json.dumps(a, indent=2))
        return 0
    if args.llm or not sys.stdin.isatty() or not sys.stdout.isatty():
        if args.llm:
            _print_llm(args.days)
        else:
            print(_cmd_pilot_snapshot(args.days))
        return 0

    # Interactive TUI loop (alt screen, 5 s refresh, Ctrl+C exits).
    os.system("stty -ixon 2>/dev/null")
    sys.stdout.write("\033[?1049h\033[?25l")
    try:
        while True:
            sys.stdout.write("\033[H")
            sys.stdout.write(_cmd_pilot_snapshot(args.days))
            sys.stdout.write("\033[J\n\n  Ctrl+C exit — refreshes every 5 s\n")
            sys.stdout.flush()
            try:
                time.sleep(5)
            except KeyboardInterrupt:
                break
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

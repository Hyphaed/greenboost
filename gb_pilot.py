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

# Trend thresholds (fractions). A stage is flagged when its latest wall time
# exceeds the rolling median by REGRESSION_PCT; a model when its latest tok/s
# drops below (1 - REGRESSION_PCT) of its average.
REGRESSION_PCT = 0.20
MIN_SAMPLES = 3          # below this, trends are noise — report, don't flag


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
    errors: list = []

    for ev in events:
        if not isinstance(ev, dict):
            continue
        kind = ev.get("kind")
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
            m = ev.get("model", "?")
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
        latest = xs[-1] if xs else 0.0
        drop = ((avg - latest) / avg) if avg > 0 else 0.0
        model_out[m] = {
            "samples": len(xs),
            "avg": round(avg, 1),
            "latest": round(latest, 1),
            "drop_pct": round(drop * 100, 1),
            "degraded": len(xs) >= MIN_SAMPLES and drop > REGRESSION_PCT,
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
        "errors": errors[-10:],
        "event_count": len(events),
    }


def advise(analysis: dict) -> list:
    """Map analysis findings to concrete, evidence-backed actions. Each item:
    {severity, topic, evidence, action, lever} — `lever` is the exact
    (read-only-quoted) GbControl call a pilot would make, or None when the
    action is a config/env change instead."""
    out: list = []
    press = analysis.get("pressure", {})

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
                       "reserve so desktop apps stop competing, and throttle "
                       "prefetch while pressure holds."),
            "lever": "GbControl().set_workstation_reserve_mb(+256); "
                     "GbControl().set_prefetch_throttle(1)",
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
                "evidence": (f"{m}: latest {d['latest']} tok/s vs avg {d['avg']} "
                             f"({d['drop_pct']:.0f}% down, n={d['samples']})"),
                "action": (f"Decode speed for {m} degraded. Check for a competing "
                           "GPU load or KV spill to T2; gb_synapse recommend() "
                           "now has measured history to re-pick quant/split."),
                "lever": "GbControl().set_kv_size_threshold_mb(...) after "
                         "confirming KV misclassification in vitals",
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
            lines.append(f"             lever:    {item['lever']}   (NOT applied — v1 is read-only)")
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

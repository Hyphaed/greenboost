#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""gb_session_audit.py , reconstruct and grade ONE serving/agent session from
the flight recorder.

Why this exists
---------------
Every panel this module prints was, before it existed, a hand-written jq or
python one-liner typed fresh against `dataflux.jsonl` at the moment somebody
asked "how did that run go?". That is a bad way to answer the question twice:
the analyst re-decides each time which window counts as "the session", which
kinds matter, and what threshold makes a number bad , so two audits of the
same run disagree, and a regression is invisible because nothing was measured
the same way last time.

The 2026-08-20 audit that forced this module is the worked example. A 40m42s
GB-CLI run looked like a flat "~2-3 tok/s, feels slow" until the events were
lined up on one clock, at which point the actual shape was obvious and not
what "slow decode" suggests:

  * the FIRST turn spent 282,963 ms of engine prefill on 14,507 prompt tokens
    (0% cache) , 4.8 minutes, ~12% of the entire session, before a single
    token came out;
  * every turn after it hit 99.7% prompt cache and ~6.4 s TTFT, so the cache
    machinery was working perfectly and was never the problem;
  * decode sat at 2.8-3.7 tok/s throughout while 11,104 MB of weights streamed
    from T2 every token, which is arithmetic, not misconfiguration.

Three different subsystems, one timeline, and the headline finding (a
five-minute cold prefill) is invisible in every per-kind view taken alone.
That is what a session audit is for.

Design rules this module holds itself to
----------------------------------------
* **A session is discovered, not assumed.** Boundaries come from the shim's own
  phase transitions and serve events, with an activity-gap fallback , never
  from "the last N hours", which silently merges two runs or splits one.
* **No hardware-shaped constants.** Every threshold is a percentage, a ratio,
  or a comparison against this same box's own recorded history. A "slow"
  verdict is relative to what this machine has previously done with the same
  model/ctx/kv key, per CLAUDE.md's hardcoded-hardware prohibition.
* **"I cannot tell" is a real verdict.** A panel with no events returns
  `available: False` with the reason, never a zero that reads as a healthy
  measurement. This is the same discipline GB-Semantics segments hold with
  three-valued `matched`.
* **Findings carry evidence and one action.** A finding that says something is
  wrong without the numbers behind it and the command that addresses it is an
  alarm, not a diagnosis.

Usage
-----
    python3 gb_session_audit.py                    # newest session, human report
    python3 gb_session_audit.py --list             # what sessions are on record
    python3 gb_session_audit.py --session 2 --json # third-newest, machine-readable
    python3 gb_session_audit.py --days 7 --llm     # compact, ANSI-free

Also reachable as `dataflux_session_audit` on the `greenboost-dataflux` MCP
server, and as `sessions()` / `audit()` from Python.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Tunables. All are ratios, percentages or durations , never hardware sizes.
# Each is env-overridable so a different box can move it without a code edit.
# --------------------------------------------------------------------------

def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


# A session ends when nothing session-defining happens for this long. 15 min is
# longer than any single turn observed on this box (the worst measured turn,
# the 2026-08-20 cold prefill, was 4.8 min) and shorter than the gap between
# two deliberate runs. Tune per box; do not tune per model.
_GAP_S = _envf("GB_AUDIT_GAP_S", 900.0)

# A cold prefill that eats more than this share of a session's wall time is
# reported. 5% is the point where a user notices the pause as "it hung".
_PREFILL_SHARE_WARN = _envf("GB_AUDIT_PREFILL_SHARE_WARN", 0.05)

# Decode is called a regression when the session median falls below this share
# of the same (model, ctx, kv_type) key's historical median on THIS box.
_DECODE_REGRESS_RATIO = _envf("GB_AUDIT_DECODE_REGRESS_RATIO", 0.75)

# A prompt cache is "repeatedly cold" when more than this share of turns after
# the first miss entirely. One cold turn is a session opening; several is a bug.
_CACHE_COLD_SHARE_WARN = _envf("GB_AUDIT_CACHE_COLD_SHARE_WARN", 0.25)

# A segment toggling more than this many times in one session is flapping:
# the verdict is tracking noise, not state, and will train the reader to ignore
# it. Reported against the segment, not against the machine.
_FLAP_LIMIT = int(_envf("GB_AUDIT_FLAP_LIMIT", 4))

# Kinds whose presence marks a session as live. Snapshots are deliberately NOT
# here: the recorder runs on a timer whether or not anything is being served,
# so including them would make every audit one endless session.
_SESSION_KINDS = (
    "tok_s_measured", "prompt_cache", "synapse_serve", "cli_tool_call",
    "spec_decode", "cache_index", "smoke_gate", "niah_cert", "kv_quality",
    "turn_bench", "agent_prefix_shift",
)

_SEVERITY_ORDER = {"critical": 0, "violation": 1, "warning": 2,
                   "diagnosis": 3, "info": 4}


@dataclass
class Finding:
    severity: str
    code: str
    title: str
    detail: str
    evidence: dict = field(default_factory=dict)
    action: str = ""

    def as_dict(self) -> dict:
        return {"severity": self.severity, "code": self.code,
                "title": self.title, "detail": self.detail,
                "evidence": self.evidence, "action": self.action}


@dataclass
class Session:
    index: int
    started: float
    finished: float
    events: list

    @property
    def wall_s(self) -> float:
        return max(0.0, self.finished - self.started)


# --------------------------------------------------------------------------
# Event loading and session discovery
# --------------------------------------------------------------------------

def _load(days: float) -> list:
    """Every event in the window, oldest first. Uses gb_dataflux's own reader
    so archive rotation and the GREENBOOST_DATAFLUX_LOG override are honoured
    rather than re-implemented."""
    import gb_dataflux
    evs = gb_dataflux.read_events(since_hours=days * 24.0)
    return sorted((e for e in evs if isinstance(e, dict) and e.get("ts")),
                  key=lambda e: e["ts"])


def sessions(days: float = 7.0) -> list:
    """Split the window into sessions on activity gaps in _SESSION_KINDS.

    Snapshots and other timer-driven kinds are attached to whichever session
    their timestamp falls inside, but never extend one , otherwise the
    always-on SnapshotRecorder would fuse every run into a single session that
    started when the box booted.
    """
    evs = _load(days)
    marks = [e for e in evs if e.get("kind") in _SESSION_KINDS]
    if not marks:
        return []

    spans = []
    start = prev = marks[0]["ts"]
    for e in marks[1:]:
        if e["ts"] - prev > _GAP_S:
            spans.append((start, prev))
            start = e["ts"]
        prev = e["ts"]
    spans.append((start, prev))

    out = []
    for i, (a, b) in enumerate(reversed(spans)):          # newest first
        # Widen to the shim phase transitions that bracket the span, so model
        # load time is inside the session rather than orphaned before it.
        lo, hi = a, b
        for e in evs:
            if e.get("kind") != "shim_transition":
                continue
            if e["ts"] <= a and e["ts"] > a - _GAP_S and e.get("to") in ("MODEL_LOAD", "INIT"):
                lo = min(lo, e["ts"])
            if b <= e["ts"] < b + _GAP_S and e.get("to") == "INIT":
                hi = max(hi, e["ts"])
        window = [e for e in evs if lo <= e["ts"] <= hi]
        out.append(Session(index=i, started=lo, finished=hi, events=window))
    return out


# --------------------------------------------------------------------------
# Panels. Each returns {"available": bool, ...} and never fabricates a zero.
# --------------------------------------------------------------------------

def _num(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None
    return {"n": len(vals), "min": round(min(vals), 3),
            "median": round(statistics.median(vals), 3),
            "mean": round(statistics.mean(vals), 3),
            "max": round(max(vals), 3)}


def _identity(s: Session) -> dict:
    serves = [e for e in s.events if e.get("kind") == "synapse_serve"]
    toks = [e for e in s.events if e.get("kind") == "tok_s_measured"]
    last = serves[-1] if serves else None
    model = None
    for src in (serves, toks):
        for e in reversed(src):
            if e.get("model"):
                model = e["model"]
                break
        if model:
            break
    out = {"available": bool(model or last), "model": model}
    if last:
        for k in ("engine", "ctx", "kv_type", "n_gpu_layers", "recipe",
                  "mtp_draft_n", "kv_gb", "ssm_state_gb", "use_cluster"):
            if last.get(k) is not None:
                out[k] = last[k]
    if not out["available"]:
        out["reason"] = "no synapse_serve or tok_s_measured event in this window"
    return out


def _throughput(s: Session, history: list) -> dict:
    toks = [e for e in s.events if e.get("kind") == "tok_s_measured"]
    if not toks:
        return {"available": False,
                "reason": "no tok_s_measured events , nothing generated, or "
                          "the proxy never recorded (see the non-streaming "
                          "telemetry path in gb_synapse_api.py)"}
    by_src = {}
    for e in toks:
        src = e.get("source") or e.get("label") or "?"
        by_src.setdefault(src, []).append(e.get("tok_s"))
    out = {"available": True, "overall": _num([e.get("tok_s") for e in toks]),
           "by_source": {k: _num(v) for k, v in by_src.items()},
           "samples": [{"ts": e["ts"], "tok_s": e.get("tok_s"),
                        "source": e.get("source") or e.get("label")}
                       for e in toks]}

    # Compare against this box's own history for the same key, never against a
    # figure written into a doc: the box IS the reference.
    key = _series_key(s, toks)
    if key:
        past = [e.get("tok_s") for e in history
                if e.get("kind") == "tok_s_measured"
                and _event_key(e) == key and e["ts"] < s.started]
        past = [v for v in past if isinstance(v, (int, float))]
        if len(past) >= 5:
            out["baseline"] = {"key": key, "n": len(past),
                               "median": round(statistics.median(past), 3)}
    return out


def _event_key(e: dict) -> str:
    return "%s::%s::%s" % (e.get("model"), e.get("ctx"), e.get("kv_type"))


def _series_key(s: Session, toks: list) -> "str | None":
    for e in reversed(toks):
        if e.get("model"):
            return _event_key(e)
    return None


def _prefill(s: Session) -> dict:
    pc = [e for e in s.events if e.get("kind") == "prompt_cache"]
    if not pc:
        return {"available": False,
                "reason": "no prompt_cache events , the proxy records these "
                          "per request; absence means no request completed "
                          "through it in this window"}
    first = pc[0]
    cold = [e for e in pc if not e.get("hit_pct")]
    rows = [{"ts": e["ts"], "hit_pct": e.get("hit_pct"),
             "prompt_tokens": e.get("prompt_tokens"),
             "reused_tokens": e.get("reused_tokens"),
             "ttft_ms": e.get("ttft_ms"),
             "engine_prompt_ms": e.get("engine_prompt_ms")} for e in pc]
    out = {"available": True, "turns": len(pc), "cold_turns": len(cold),
           "hit_pct": _num([e.get("hit_pct") for e in pc]),
           "ttft_ms": _num([e.get("ttft_ms") for e in pc]),
           "engine_prompt_ms": _num([e.get("engine_prompt_ms") for e in pc]),
           "prompt_depth_tokens": _num([e.get("prompt_tokens") for e in pc]),
           "turns_detail": rows}
    ep = first.get("engine_prompt_ms")
    pt = first.get("prompt_tokens")
    if isinstance(ep, (int, float)) and ep > 0:
        out["first_turn"] = {
            "engine_prompt_ms": ep, "prompt_tokens": pt,
            "hit_pct": first.get("hit_pct"),
            "share_of_session": (round(ep / 1000.0 / s.wall_s, 4)
                                 if s.wall_s > 0 else None),
            "prefill_tok_s": (round(pt / (ep / 1000.0), 1)
                              if isinstance(pt, (int, float)) and pt else None),
        }
    return out


def _memory(s: Session) -> dict:
    snaps = [e for e in s.events if e.get("kind") == "snapshot"]
    if not snaps:
        return {"available": False,
                "reason": "no snapshot events , SnapshotRecorder was not "
                          "running (GREENBOOST_DATAFLUX=0, or no active "
                          "GREENBOOST_ACTIVE process)"}

    def col(*names):
        for n in names:
            vals = [e.get(n) for e in snaps if isinstance(e.get(n), (int, float))]
            if vals:
                return n, _num(vals)
        return None, None

    out = {"available": True, "snapshots": len(snaps)}
    for label, names in (
            ("vram_fill_pct", ("fb_phys_used_pct", "fb_used_pct")),
            ("gpu_util_pct", ("gpu_util", "util_gpu")),
            ("t2_pressure", ("t2_pressure",)),
            ("tier_t2_mb", ("tier_t2", "t2_used_mb")),
            ("tier_t3_mb", ("tier_t3", "t3_used_mb")),
            ("kv_pressure", ("kv_pressure",))):
        src, stats = col(*names)
        if stats:
            out[label] = dict(stats, raw_source=src)
    phases = [e.get("shim_phase") for e in snaps if e.get("shim_phase")]
    if phases:
        seen, ordered = set(), []
        for p in phases:
            if p not in seen:
                seen.add(p); ordered.append(p)
        out["shim_phases"] = ordered
    return out


def _errors(s: Session) -> dict:
    errs = [e for e in s.events if e.get("status") == "error"]
    if not errs:
        return {"available": True, "count": 0, "by_kind": {}}
    by = {}
    for e in errs:
        by.setdefault(e.get("kind", "?"), []).append(e)
    return {"available": True, "count": len(errs),
            "by_kind": {k: {"count": len(v),
                            "sample": {kk: vv for kk, vv in v[-1].items()
                                       if kk not in ("evidence",)}}
                        for k, v in by.items()}}


def _segments(s: Session) -> dict:
    trans = [e for e in s.events if e.get("kind") == "semantic_transition"]
    if not trans:
        return {"available": False,
                "reason": "no semantic_transition events , gb_semantics_watch "
                          "was not running during this session"}
    by = {}
    for e in trans:
        by.setdefault(e.get("segment", "?"), []).append(e)
    out = {}
    for name, evs in by.items():
        matched = [e for e in evs if e.get("to") == "matched"]
        out[name] = {"transitions": len(evs), "times_matched": len(matched),
                     "severity": evs[-1].get("severity"),
                     "final": evs[-1].get("to")}
    return {"available": True, "segments": out}


def _tools(s: Session) -> dict:
    calls = [e for e in s.events if e.get("kind") == "cli_tool_call"]
    if not calls:
        return {"available": False, "reason": "no cli_tool_call events"}
    denied = [e for e in calls if e.get("decision") == "deny"
              or e.get("allowed") is False]
    by = {}
    for e in calls:
        by[e.get("tool", "?")] = by.get(e.get("tool", "?"), 0) + 1
    return {"available": True, "calls": len(calls), "denied": len(denied),
            "by_tool": dict(sorted(by.items(), key=lambda kv: -kv[1]))}


def _gates(s: Session) -> dict:
    rows = [e for e in s.events
            if e.get("kind") in ("smoke_gate", "niah_cert", "kv_quality")]
    if not rows:
        return {"available": False, "reason": "no quality-gate events in window"}
    return {"available": True,
            "runs": [{"kind": e.get("kind"), "ts": e["ts"],
                      "verdict": e.get("verdict"), "reason": e.get("reason"),
                      "recall": e.get("recall"), "kv_type": e.get("kv_type"),
                      "status": e.get("status")} for e in rows]}


# --------------------------------------------------------------------------
# Findings , the graded verdicts, each with evidence and exactly one action
# --------------------------------------------------------------------------

def _findings(s: Session, panels: dict) -> list:
    out = []
    pre, thr = panels["prefill"], panels["throughput"]
    mem, seg = panels["memory"], panels["segments"]

    ft = (pre.get("first_turn") or {}) if pre.get("available") else {}
    share = ft.get("share_of_session")
    if share and share >= _PREFILL_SHARE_WARN:
        out.append(Finding(
            severity="warning", code="cold_prefill_dominates",
            title="The session's first turn spent %.0f s prefilling before "
                  "any output" % (ft["engine_prompt_ms"] / 1000.0),
            detail="That is %.0f%% of the whole session's wall time, on a "
                   "prompt of %s tokens at %s%% cache hit. Every later turn "
                   "reused the cache, so this cost is paid once per cold "
                   "engine, not per turn , but it is paid in full every time "
                   "the model is restarted."
                   % (share * 100, ft.get("prompt_tokens"),
                      ft.get("hit_pct")),
            evidence={k: ft.get(k) for k in
                      ("engine_prompt_ms", "prompt_tokens", "hit_pct",
                       "prefill_tok_s", "share_of_session")},
            action="Keep the engine warm between runs (`synapse_pause`/"
                   "`synapse_resume` instead of stop/serve), or shrink the "
                   "cold prompt , this is prefill, not decode, so decode "
                   "knobs will not touch it."))

    if pre.get("available"):
        turns, cold = pre["turns"], pre["cold_turns"]
        if turns > 1 and (cold - 1) / max(1, turns - 1) > _CACHE_COLD_SHARE_WARN:
            out.append(Finding(
                severity="warning", code="cache_cold_repeat",
                title="%d of %d turns missed the prompt cache entirely"
                      % (cold, turns),
                detail="One cold turn opens a session. Several means the "
                       "prefix is not stable across turns, or concurrent "
                       "conversations are competing for the same slots.",
                evidence={"turns": turns, "cold_turns": cold,
                          "hit_pct": pre.get("hit_pct")},
                action="Check prompt-prefix stability (volatile content must "
                       "sit at the END of the system prompt) and raise "
                       "`slot_prompt_similarity` on the next serve."))

    base = thr.get("baseline") if thr.get("available") else None
    if base and thr.get("overall"):
        cur, ref = thr["overall"]["median"], base["median"]
        if ref > 0 and cur < ref * _DECODE_REGRESS_RATIO:
            out.append(Finding(
                severity="warning", code="decode_below_baseline",
                title="Decode ran at %.2f tok/s against this box's own %.2f "
                      "tok/s median for the same config" % (cur, ref),
                detail="Same model, context and KV type, %d prior samples. "
                       "This is a comparison against this machine's history, "
                       "not against a published number."
                       % base["n"],
                evidence={"session_median": cur, "baseline_median": ref,
                          "key": base["key"], "baseline_n": base["n"]},
                action="Compare prompt depth between the two , decode slows "
                       "with context on a T2-spilling model. Run "
                       "`gb semantics answer \"why is it slow\"` for the "
                       "governed read."))

    if mem.get("available") and mem.get("vram_fill_pct"):
        v = mem["vram_fill_pct"]
        t2 = mem.get("tier_t2_mb")
        if t2 and t2.get("median", 0) > 0:
            out.append(Finding(
                severity="diagnosis", code="weights_streaming_from_t2",
                title="Weights streamed from T2 for the whole session",
                detail="VRAM sat at %.1f%% median fill while T2 held ~%.0f MB. "
                       "Decode is bandwidth-bound in this state: every token "
                       "re-reads those bytes across PCIe, and no decode knob "
                       "changes that arithmetic."
                       % (v["median"], t2["median"]),
                evidence={"vram_fill_pct": v, "tier_t2_mb": t2},
                action="Either fit the model (a smaller quant, shorter ctx) or "
                       "accept the ceiling , `quant_advisor` will price the "
                       "options against the fp8 floor."))

    if seg.get("available"):
        for name, st in seg["segments"].items():
            if st["transitions"] > _FLAP_LIMIT:
                out.append(Finding(
                    severity="info", code="segment_flapping",
                    title="Segment `%s` changed verdict %d times this session"
                          % (name, st["transitions"]),
                    detail="A verdict that toggles this often is tracking "
                           "noise rather than state, and trains the reader to "
                           "ignore it.",
                    evidence={"segment": name, **st},
                    action="Review the segment's threshold in "
                           "semantics/segments.yaml , it likely needs "
                           "hysteresis or a windowed input."))

    gates = panels["gates"]
    if gates.get("available"):
        for r in gates["runs"]:
            if r.get("status") == "error" or r.get("verdict") == "FAIL":
                out.append(Finding(
                    severity="violation", code="quality_gate_failed",
                    title="%s returned %s" % (r["kind"], r.get("verdict")),
                    detail=str(r.get("reason") or ""),
                    evidence=r,
                    action="Confirm the gate is grading the model and not the "
                           "prompt before acting on it , see "
                           "gb_aviary.smoke_gate's docstring."))

    errs = panels["errors"]
    for kind, info in (errs.get("by_kind") or {}).items():
        if kind in ("semantic_transition", "smoke_gate"):
            continue                       # already covered by richer findings
        out.append(Finding(
            severity="warning", code="errors_logged",
            title="%d `%s` event(s) recorded an error" % (info["count"], kind),
            detail="Surfaced from the flight recorder; see the sample for the "
                   "failing fields.",
            evidence=info["sample"],
            action="`dataflux_events(kind=\"%s\", status=\"error\")` for the "
                   "full set." % kind))

    tools = panels["tools"]
    if tools.get("available") and tools.get("denied"):
        out.append(Finding(
            severity="info", code="tool_calls_denied",
            title="%d tool call(s) were denied by policy" % tools["denied"],
            detail="Deny-by-default policy is working as configured; listed so "
                   "an unattended run's blocked actions are visible after the "
                   "fact.",
            evidence={"denied": tools["denied"], "calls": tools["calls"]},
            action="Review `instruments/policy.py` if any of these should have "
                   "been allowed."))

    out.sort(key=lambda f: _SEVERITY_ORDER.get(f.severity, 9))
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def audit(index: int = 0, days: float = 7.0,
          with_governed: bool = True) -> dict:
    """Full audit of one session. `index` 0 is the newest."""
    all_s = sessions(days=days)
    if not all_s:
        return {"available": False,
                "reason": "no sessions found in the last %g days , the flight "
                          "recorder has no tok_s/prompt_cache/serve events in "
                          "that window" % days}
    if index >= len(all_s):
        return {"available": False,
                "reason": "session %d requested but only %d on record"
                          % (index, len(all_s))}
    s = all_s[index]
    history = _load(days)

    panels = {
        "identity": _identity(s),
        "throughput": _throughput(s, history),
        "prefill": _prefill(s),
        "memory": _memory(s),
        "segments": _segments(s),
        "gates": _gates(s),
        "tools": _tools(s),
        "errors": _errors(s),
    }
    findings = _findings(s, panels)

    rep = {
        "available": True,
        "session": {"index": s.index, "of": len(all_s),
                    "started": s.started, "finished": s.finished,
                    "started_iso": _iso(s.started),
                    "finished_iso": _iso(s.finished),
                    "wall_s": round(s.wall_s, 1),
                    "events": len(s.events)},
        **panels,
        "findings": [f.as_dict() for f in findings],
    }
    if with_governed:
        rep["governed"] = _governed()

    _emit(rep)
    return rep


def _governed() -> dict:
    """Current governed verdicts. Deliberately NOT windowed to the session:
    GB-Semantics resolves live state, so this is 'where the box stands now',
    labelled as such rather than passed off as a historical reading."""
    try:
        import gb_semantics
    except Exception as exc:                                  # pragma: no cover
        return {"available": False, "reason": "gb_semantics unavailable: %s" % exc}
    out = {"available": True, "as_of": time.time(), "scope": "live now, not "
           "the session window", "metrics": {}, "segments": {}}
    for m in ("vram_fill_pct", "tok_s_decode", "t2_overflow_active_mb",
              "prompt_cache_hit_pct", "ttft_ms", "kmod_loaded",
              "core_build_version", "kmod_loaded_version"):
        try:
            r = gb_semantics.resolve(m)
            out["metrics"][m] = {"value": r.get("value"), "unit": r.get("unit"),
                                 "threshold_state": r.get("threshold_state")}
        except Exception as exc:
            out["metrics"][m] = {"error": str(exc)}
    for sname in ("rule1_underfilled", "weights_dont_fit_vram",
                  "kmod_version_drift", "prompt_cache_cold",
                  "swap_thrash_not_gpu_throttle"):
        try:
            r = gb_semantics.evaluate_segment(sname)
            out["segments"][sname] = {"matched": r.get("matched"),
                                      "severity": r.get("severity")}
        except Exception as exc:
            out["segments"][sname] = {"error": str(exc)}
    return out


def _emit(rep: dict) -> None:
    """Best-effort telemetry, per the Observability Must-Rule: the audit itself
    leaves a trace, so 'when was this last audited and what did it say' is
    answerable without re-running it."""
    try:
        import gb_dataflux
        sev = [f["severity"] for f in rep.get("findings", [])]
        gb_dataflux.emit({
            "kind": "session_audit", "label": "gb_session_audit",
            "session_started": rep["session"]["started"],
            "session_wall_s": rep["session"]["wall_s"],
            "model": (rep.get("identity") or {}).get("model"),
            "findings": len(sev),
            "worst_severity": (sorted(sev, key=lambda x: _SEVERITY_ORDER.get(x, 9))[0]
                               if sev else "none"),
            "codes": [f["code"] for f in rep.get("findings", [])],
            "status": "error" if any(x in ("critical", "violation") for x in sev) else "ok",
        })
    except Exception:
        pass


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

_C = {"dim": "\033[2m", "bold": "\033[1m", "cyan": "\033[36m",
      "red": "\033[31m", "yellow": "\033[33m", "green": "\033[32m",
      "off": "\033[0m"}


def _paint(plain: bool) -> dict:
    return {k: "" for k in _C} if plain else _C


def render(rep: dict, plain: bool = False) -> str:
    c = _paint(plain)
    if not rep.get("available"):
        return "No session audit available: %s" % rep.get("reason")
    L = []
    s, ident = rep["session"], rep["identity"]
    L.append("%s%sGreenBoost session audit%s  %s%s , %s  (%.1f min, %d events)%s"
             % (c["bold"], c["cyan"], c["off"], c["dim"], s["started_iso"],
                s["finished_iso"], s["wall_s"] / 60.0, s["events"], c["off"]))
    L.append("%s session %d of %d on record%s"
             % (c["dim"], s["index"], s["of"], c["off"]))
    L.append("")

    if ident.get("available"):
        bits = [str(ident.get("model"))]
        for k in ("ctx", "kv_type", "n_gpu_layers", "engine", "mtp_draft_n"):
            if ident.get(k) is not None:
                bits.append("%s=%s" % (k, ident[k]))
        L.append("%sModel%s      %s" % (c["bold"], c["off"], "  ".join(bits)))
    else:
        L.append("%sModel%s      unknown , %s"
                 % (c["bold"], c["off"], ident.get("reason")))

    thr = rep["throughput"]
    if thr.get("available"):
        o = thr["overall"]
        line = ("median %.2f  (min %.2f, max %.2f, n=%d)"
                % (o["median"], o["min"], o["max"], o["n"]))
        if thr.get("baseline"):
            line += "   box baseline %.2f (n=%d)" % (
                thr["baseline"]["median"], thr["baseline"]["n"])
        L.append("%sDecode%s     %s tok/s" % (c["bold"], c["off"], line))
    else:
        L.append("%sDecode%s     not measured , %s"
                 % (c["bold"], c["off"], thr.get("reason")))

    pre = rep["prefill"]
    if pre.get("available"):
        h, t = pre["hit_pct"], pre["ttft_ms"]
        L.append("%sPrefill%s    %d turns, cache hit median %.1f%%, "
                 "TTFT median %.0f ms"
                 % (c["bold"], c["off"], pre["turns"], h["median"], t["median"]))
        ft = pre.get("first_turn") or {}
        if ft.get("engine_prompt_ms"):
            L.append("%s            first turn: %.1f s of engine prefill on "
                     "%s tokens (%.0f tok/s), %s%% cached%s"
                     % (c["dim"], ft["engine_prompt_ms"] / 1000.0,
                        ft.get("prompt_tokens"), ft.get("prefill_tok_s") or 0,
                        ft.get("hit_pct"), c["off"]))
    else:
        L.append("%sPrefill%s    not measured , %s"
                 % (c["bold"], c["off"], pre.get("reason")))

    mem = rep["memory"]
    if mem.get("available"):
        v = mem.get("vram_fill_pct")
        if v:
            extra = ""
            if mem.get("tier_t2_mb"):
                extra = ",  T2 median %.0f MB" % mem["tier_t2_mb"]["median"]
            L.append("%sMemory%s     VRAM fill median %.1f%% "
                     "(min %.1f, max %.1f)%s"
                     % (c["bold"], c["off"], v["median"], v["min"], v["max"],
                        extra))
        if mem.get("shim_phases"):
            L.append("%s            shim phases: %s%s"
                     % (c["dim"], " -> ".join(mem["shim_phases"]), c["off"]))
    else:
        L.append("%sMemory%s     not measured , %s"
                 % (c["bold"], c["off"], mem.get("reason")))

    tools = rep["tools"]
    if tools.get("available"):
        top = list(tools["by_tool"].items())[:4]
        L.append("%sAgent%s      %d tool calls (%d denied)   %s"
                 % (c["bold"], c["off"], tools["calls"], tools["denied"],
                    ", ".join("%s x%d" % kv for kv in top)))

    L.append("")
    fs = rep["findings"]
    if not fs:
        L.append("%sNo findings. Nothing in this session crossed a threshold.%s"
                 % (c["green"], c["off"]))
    else:
        L.append("%s%sFindings (%d)%s" % (c["bold"], c["cyan"], len(fs), c["off"]))
        for f in fs:
            col = {"critical": c["red"], "violation": c["red"],
                   "warning": c["yellow"]}.get(f["severity"], c["dim"])
            L.append("")
            L.append("  %s[%s]%s %s" % (col, f["severity"].upper(), c["off"],
                                        f["title"]))
            if f["detail"]:
                for ln in _wrap(f["detail"], 74):
                    L.append("      %s" % ln)
            if f["action"]:
                L.append("      %s-> %s%s" % (c["green"], f["action"], c["off"])
                         if not plain else "      -> %s" % f["action"])

    gov = rep.get("governed") or {}
    if gov.get("available"):
        L.append("")
        L.append("%sGoverned now%s %s(GB-Semantics, live , not the session "
                 "window)%s" % (c["bold"], c["off"], c["dim"], c["off"]))
        for k, v in gov["metrics"].items():
            if v.get("error"):
                continue
            L.append("   %-24s %s %s" % (k, v.get("value"),
                                         v.get("threshold_state") or ""))
        matched = [k for k, v in gov["segments"].items() if v.get("matched")]
        if matched:
            L.append("   %ssegments matched:%s %s"
                     % (c["yellow"], c["off"], ", ".join(matched)))
    return "\n".join(L)


def _wrap(text: str, width: int) -> list:
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            out.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        out.append(cur)
    return out


def render_list(days: float = 7.0) -> str:
    ss = sessions(days=days)
    if not ss:
        return "No sessions on record in the last %g days." % days
    L = ["idx  started              wall      events  model"]
    for s in ss:
        ident = _identity(s)
        L.append("%3d  %s  %6.1fm  %6d  %s"
                 % (s.index, _iso(s.started), s.wall_s / 60.0, len(s.events),
                    str(ident.get("model") or "?")[:44]))
    return "\n".join(L)


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Audit one GreenBoost serving/agent session end to end.")
    ap.add_argument("--session", type=int, default=0,
                    help="0 = newest (default)")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--list", action="store_true",
                    help="list sessions on record and exit")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--llm", action="store_true",
                    help="compact, ANSI-free output for machine reading")
    ap.add_argument("--no-governed", action="store_true",
                    help="skip the live GB-Semantics block")
    a = ap.parse_args(argv)

    if a.list:
        print(render_list(days=a.days))
        return 0
    rep = audit(index=a.session, days=a.days,
                with_governed=not a.no_governed)
    if a.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        print(render(rep, plain=a.llm))
    if not rep.get("available"):
        return 3
    worst = [f["severity"] for f in rep.get("findings", [])]
    return 2 if any(x in ("critical", "violation") for x in worst) else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for gb_session_audit.py.

These assert the properties that make an audit trustworthy rather than
byte-exact output: that a session is DISCOVERED (not assumed), that a panel
with no data says so instead of returning a zero, and that the findings fire
on the shapes the 2026-08-20 audit actually found.
"""
from __future__ import annotations

import json
import time

import pytest

import gb_session_audit as A


def _write_log(tmp_path, monkeypatch, events):
    p = tmp_path / "dataflux.jsonl"
    with p.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    # gb_dataflux resolves the log path per call from the env — never patch a
    # cached Path (tests/conftest.py documents why).
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(p))
    return p


def _serve(ts, **kw):
    d = {"kind": "synapse_serve", "ts": ts, "node": "host",
         "model": "M", "ctx": 24576, "kv_type": "q4_0", "engine": "llama.cpp"}
    d.update(kw)
    return d


def _tok(ts, v):
    return {"kind": "tok_s_measured", "ts": ts, "node": "host", "model": "M",
            "ctx": 24576, "kv_type": "q4_0", "tok_s": v, "source": "proxy"}


def _pc(ts, hit, prompt_tokens, engine_prompt_ms, ttft_ms):
    return {"kind": "prompt_cache", "ts": ts, "node": "host", "model": "M",
            "hit_pct": hit, "prompt_tokens": prompt_tokens,
            "reused_tokens": int(prompt_tokens * hit / 100.0),
            "engine_prompt_ms": engine_prompt_ms, "ttft_ms": ttft_ms}


# ---------------------------------------------------------------- discovery

def test_sessions_split_on_activity_gap(tmp_path, monkeypatch):
    now = time.time()
    a = now - 10000
    b = a + A._GAP_S * 2          # far enough to be a separate session
    _write_log(tmp_path, monkeypatch, [
        _serve(a), _tok(a + 1, 5.0), _tok(a + 2, 5.0),
        _serve(b), _tok(b + 1, 6.0),
    ])
    ss = A.sessions(days=1)
    assert len(ss) == 2
    # newest first
    assert ss[0].started >= ss[1].started


def test_snapshots_alone_do_not_create_a_session(tmp_path, monkeypatch):
    """The SnapshotRecorder runs on a timer whether or not anything is served.
    If snapshots defined sessions, every audit would be one endless run."""
    now = time.time()
    _write_log(tmp_path, monkeypatch, [
        {"kind": "snapshot", "ts": now - 500 + i, "node": "host",
         "fb_phys_used_pct": 50.0} for i in range(20)
    ])
    assert A.sessions(days=1) == []


def test_session_widens_to_bracketing_shim_phase(tmp_path, monkeypatch):
    """Model-load time belongs to the session, not orphaned before it."""
    now = time.time()
    t = now - 1000
    _write_log(tmp_path, monkeypatch, [
        {"kind": "shim_transition", "ts": t - 30, "node": "host",
         "from": "INIT", "to": "MODEL_LOAD"},
        _serve(t), _tok(t + 5, 4.0),
    ])
    ss = A.sessions(days=1)
    assert len(ss) == 1
    assert ss[0].started <= t - 30 + 1e-6


# ------------------------------------------------------------------- panels

def test_missing_panel_reports_unavailable_not_zero(tmp_path, monkeypatch):
    now = time.time()
    t = now - 600
    _write_log(tmp_path, monkeypatch, [_serve(t), _tok(t + 1, 5.0)])
    rep = A.audit(index=0, days=1, with_governed=False)
    assert rep["available"] is True
    # No prompt_cache and no snapshot events were written.
    assert rep["prefill"]["available"] is False
    assert "reason" in rep["prefill"]
    assert rep["memory"]["available"] is False
    assert "reason" in rep["memory"]
    # Crucially, no fabricated zeros.
    assert "hit_pct" not in rep["prefill"]
    assert "vram_fill_pct" not in rep["memory"]


def test_throughput_baseline_uses_same_key_only(tmp_path, monkeypatch):
    now = time.time()
    old = now - 10000
    cur = now - 600
    events = [_serve(old)]
    # 8 historical samples on the SAME key, plus noise on a different ctx.
    events += [_tok(old + i, 8.0) for i in range(8)]
    events += [dict(_tok(old + 20 + i, 99.0), ctx=4096) for i in range(8)]
    events += [_serve(cur)] + [_tok(cur + i, 3.0) for i in range(4)]
    _write_log(tmp_path, monkeypatch, events)
    rep = A.audit(index=0, days=1, with_governed=False)
    base = rep["throughput"]["baseline"]
    assert base["median"] == 8.0          # the ctx=4096 noise must not leak in
    codes = [f["code"] for f in rep["findings"]]
    assert "decode_below_baseline" in codes


# ----------------------------------------------------------------- findings

def test_cold_prefill_finding_fires_and_carries_evidence(tmp_path, monkeypatch):
    """The 2026-08-20 shape: one enormous cold prefill, then a warm cache."""
    now = time.time()
    t = now - 3600
    events = [_serve(t)]
    events.append(_pc(t + 10, 0.0, 14507, 282963.0, 286294.0))
    for i in range(1, 6):
        events.append(_pc(t + 300 * i + 300, 99.7, 15000, 3000.0, 6400.0))
        events.append(_tok(t + 300 * i + 305, 3.0))
    events.append({"kind": "cli_tool_call", "ts": t + 2400, "node": "host",
                   "tool": "Bash", "decision": "allow"})
    _write_log(tmp_path, monkeypatch, events)

    rep = A.audit(index=0, days=1, with_governed=False)
    f = next(x for x in rep["findings"] if x["code"] == "cold_prefill_dominates")
    assert f["severity"] == "warning"
    assert f["evidence"]["prompt_tokens"] == 14507
    assert f["evidence"]["hit_pct"] == 0.0
    assert f["action"]                      # a finding without an action is an alarm
    # ~51 tok/s prefill, as measured
    assert 40 < f["evidence"]["prefill_tok_s"] < 70


def test_no_findings_on_a_clean_session(tmp_path, monkeypatch):
    now = time.time()
    t = now - 600
    events = [_serve(t)]
    for i in range(5):
        events.append(_pc(t + 10 * i + 10, 99.0, 5000, 400.0, 500.0))
        events.append(_tok(t + 10 * i + 12, 8.0))
    _write_log(tmp_path, monkeypatch, events)
    rep = A.audit(index=0, days=1, with_governed=False)
    assert rep["findings"] == []


def test_quality_gate_failure_is_a_violation(tmp_path, monkeypatch):
    now = time.time()
    t = now - 600
    _write_log(tmp_path, monkeypatch, [
        _serve(t), _tok(t + 1, 5.0),
        {"kind": "smoke_gate", "ts": t + 2, "node": "host", "verdict": "FAIL",
         "reason": "only 6 tokens of content", "status": "error"},
    ])
    rep = A.audit(index=0, days=1, with_governed=False)
    f = next(x for x in rep["findings"] if x["code"] == "quality_gate_failed")
    assert f["severity"] == "violation"


# ------------------------------------------------------------------ shaping

def test_render_is_ansi_free_in_plain_mode(tmp_path, monkeypatch):
    now = time.time()
    t = now - 600
    _write_log(tmp_path, monkeypatch, [_serve(t), _tok(t + 1, 5.0)])
    rep = A.audit(index=0, days=1, with_governed=False)
    assert "\033[" not in A.render(rep, plain=True)


def test_audit_out_of_range_says_so(tmp_path, monkeypatch):
    now = time.time()
    _write_log(tmp_path, monkeypatch, [_serve(now - 600), _tok(now - 599, 5.0)])
    rep = A.audit(index=99, days=1, with_governed=False)
    assert rep["available"] is False
    assert "only 1 on record" in rep["reason"]


def test_empty_log_is_unavailable_not_a_crash(tmp_path, monkeypatch):
    _write_log(tmp_path, monkeypatch, [])
    rep = A.audit(index=0, days=1, with_governed=False)
    assert rep["available"] is False


def test_session_audit_kind_is_registered():
    """Observability Must-Rule: the audit leaves its own trace, and
    checks/check_dataflux_coverage.py requires every emitted kind be declared."""
    import gb_dataflux_kinds
    assert "session_audit" in gb_dataflux_kinds.KINDS


# ----------------------------------------------- version-consistency check

def test_unreleased_changelog_heading_is_not_a_version_claim(tmp_path):
    """An "in development, not released" heading describes `main`, not the
    built artifact. Comparing it against MODULE_VERSION would force the core's
    version literals to a number no release carries , and the Gaming Suite
    reads exactly that field as "installed core version", which is the
    confusion this check exists to prevent, inverted."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "checks"))
    import check_version_consistency as cvc

    ch = tmp_path / "CHANGELOG.md"
    ch.write_text(
        "# Changelog\n\n"
        "## v3.5 : in development, not released\n\n"
        "stuff\n\n"
        "## v3.4 : 2026-08-18\n\n"
        "released stuff\n")
    pat = next(p for f, p, _ in cvc._SITES if f == "CHANGELOG.md")
    got = cvc._first_match(ch, pat)
    assert got is not None
    assert got[0] == "3.4", "must fall through to the newest RELEASED heading"


def test_released_changelog_heading_is_still_matched(tmp_path):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "checks"))
    import check_version_consistency as cvc

    ch = tmp_path / "CHANGELOG.md"
    ch.write_text("# Changelog\n\n## v3.4 : 2026-08-18\n\nreleased\n")
    pat = next(p for f, p, _ in cvc._SITES if f == "CHANGELOG.md")
    assert cvc._first_match(ch, pat)[0] == "3.4"

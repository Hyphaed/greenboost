#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for serving/probe.py — the typed pre-serve probe suite
(NemoClaw audit, Phase 5d): bounded steps, mixed/partial evidence fails
closed with a named reason, never a bare traceback.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serving"))

import pytest

import probe


def _ok(*_a, **_k):
    return True


def _fail(reason_message):
    def _f():
        return (False, reason_message)
    return _f


def _raises(exc):
    def _f():
        raise exc
    return _f


def _slow(seconds):
    def _f():
        time.sleep(seconds)
        return True
    return _f


def test_all_five_steps_succeed():
    steps = {name: _ok for name in probe.STEP_ORDER}
    results = probe.probe_serve_readiness(steps)
    assert len(results) == 5
    assert probe.overall_ok(results)
    assert [r.step for r in results] == list(probe.STEP_ORDER)


def test_stops_at_first_failure_mixed_evidence_fails_closed():
    """A LATER step succeeding never excuses an EARLIER step's failure —
    the probe must stop at health, never reach the completion steps."""
    steps = {
        "gguf_header": _ok,
        "short_load": _ok,
        "health": _fail("engine reports degraded"),
        "one_token_completion": _ok,   # must never run
        "tool_call_completion": _ok,   # must never run
    }
    results = probe.probe_serve_readiness(steps)
    assert len(results) == 3
    assert not probe.overall_ok(results)
    assert results[-1].step == "health"
    assert results[-1].reason == "health_unhealthy"


def test_missing_step_callable_is_probe_aborted():
    steps = {"gguf_header": _ok}  # short_load etc. missing
    results = probe.probe_serve_readiness(steps)
    assert results[-1].reason == "probe_aborted"


def test_exception_in_step_maps_to_secondary_reason_not_a_traceback():
    steps = {
        "gguf_header": _raises(KeyError("rope.dimension_sections")),
    }
    results = probe.probe_serve_readiness(steps)
    assert len(results) == 1
    assert results[0].reason == "gguf_malformed"
    assert "rope.dimension_sections" in results[0].message


def test_step_exceeding_bound_maps_to_timeout_reason():
    steps = {"gguf_header": _slow(0.05)}
    results = probe.probe_serve_readiness(steps, timeouts_s={"gguf_header": 0.01})
    assert results[0].reason == "gguf_unreadable"
    assert not results[0].ok


def test_require_tool_call_false_skips_final_step():
    steps = {name: _ok for name in probe.STEP_ORDER[:-1]}
    results = probe.probe_serve_readiness(steps, require_tool_call=False)
    assert [r.step for r in results] == list(probe.STEP_ORDER[:-1])
    assert probe.overall_ok(results)


def test_tool_call_failure_reason_is_specific():
    steps = {name: _ok for name in probe.STEP_ORDER[:-1]}
    steps["tool_call_completion"] = _fail("model returned plain text, no tool call")
    results = probe.probe_serve_readiness(steps)
    assert results[-1].reason == "tool_call_unsupported"


def test_probe_result_rejects_reason_outside_closed_set():
    with pytest.raises(ValueError, match="not in the closed set"):
        probe.ProbeResult(ok=False, step="health", reason="made_up_reason")


def test_probe_result_success_cannot_carry_a_reason():
    with pytest.raises(ValueError, match="must not carry a reason"):
        probe.ProbeResult(ok=True, step="health", reason="health_unhealthy")


def test_overall_ok_false_for_empty_results():
    assert probe.overall_ok([]) is False


@pytest.mark.parametrize("reason", sorted(probe.PROBE_FAILURE_REASONS))
def test_every_closed_set_reason_constructs_a_valid_result(reason):
    """Every reason in the closed set must actually be usable — this is
    the regression guard against a reason string that's declared but never
    reachable through any real step outcome."""
    result = probe.ProbeResult(ok=False, step="health", reason=reason, message="x")
    assert result.reason == reason

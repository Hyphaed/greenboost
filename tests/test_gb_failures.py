#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""tests/test_gb_failures.py — unit tests for gb_failures module.

Tests verify:
  1. Each FailureKind routes correctly with the right next_step
  2. Confidence downgrade fires when signals are stale/unverified
  3. Caveat text is present in reason when confidence is "low"
  4. Real-world scenarios (_gate_cpu_offload integration, etc.)
"""
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import gb_failures
import pytest


class TestFailureKindEnum:
    """Enum validation and membership tests."""

    def test_closed_set_enumeration(self):
        """All FailureKind values are present."""
        expected = {
            "shim_broken",
            "capacity_exceeded",
            "feeder_unreachable",
            "port_conflict",
            "quality_gate_failed",
            "engine_missing",
            "unknown",
        }
        actual = {kind.value for kind in gb_failures.FailureKind}
        assert actual == expected, f"Mismatch: {actual} vs {expected}"

    def test_enum_str_coercion(self):
        """FailureKind enums are str-coercible."""
        kind = gb_failures.FailureKind.SHIM_BROKEN
        assert isinstance(kind, str)
        assert kind == "shim_broken"


class TestFailureClassificationValidation:
    """FailureClassification dataclass invariants."""

    def test_confidence_must_be_high_or_low(self):
        """Invalid confidence values raise ValueError."""
        with pytest.raises(ValueError, match="confidence must be 'high' or 'low'"):
            gb_failures.FailureClassification(
                kind=gb_failures.FailureKind.UNKNOWN,
                reason="test",
                next_step="test",
                confidence="medium",  # invalid
            )

    def test_frozen_dataclass(self):
        """FailureClassification is immutable."""
        fc = gb_failures.FailureClassification(
            kind=gb_failures.FailureKind.UNKNOWN,
            reason="test",
            next_step="test",
            confidence="high",
        )
        with pytest.raises(AttributeError):
            fc.kind = gb_failures.FailureKind.SHIM_BROKEN


class TestShimBrokenClassification:
    """FailureKind.SHIM_BROKEN routing and confidence."""

    def test_shim_broken_fresh_probe(self):
        """Fresh shim probe failure → high confidence."""
        shim_status = (False, "invalid device function in probe output")
        fc = gb_failures.classify_serve_failure(
            kind_hint=gb_failures.FailureKind.SHIM_BROKEN,
            shim_status=shim_status,
            shim_cache_age_s=0.5,  # just probed
        )
        assert fc.kind == gb_failures.FailureKind.SHIM_BROKEN
        assert fc.confidence == "high"
        assert "invalid device function" in fc.reason
        assert "old (cached)" not in fc.reason  # no caveat
        assert "build-engine" in fc.next_step

    def test_shim_broken_stale_cache(self):
        """Stale shim cache → low confidence + caveat."""
        shim_status = (False, "cuda error")
        fc = gb_failures.classify_serve_failure(
            kind_hint=gb_failures.FailureKind.SHIM_BROKEN,
            shim_status=shim_status,
            shim_cache_age_s=7200,  # 2 hours old
        )
        assert fc.kind == gb_failures.FailureKind.SHIM_BROKEN
        assert fc.confidence == "low"
        assert "7200" in fc.reason  # age visible
        assert "cached" in fc.reason.lower()
        assert "reboot" in fc.reason or "configuration change" in fc.reason

    def test_shim_broken_no_cache_age(self):
        """Shim broken but no cache age info → high confidence."""
        shim_status = (False, "unknown failure")
        fc = gb_failures.classify_serve_failure(
            shim_status=shim_status,
            shim_cache_age_s=None,
        )
        assert fc.confidence == "high"

    def test_shim_broken_hint_without_status(self):
        """kind_hint alone is enough to route SHIM_BROKEN."""
        fc = gb_failures.classify_serve_failure(
            kind_hint=gb_failures.FailureKind.SHIM_BROKEN,
        )
        assert fc.kind == gb_failures.FailureKind.SHIM_BROKEN
        assert fc.confidence == "high"


class TestCapacityExceededClassification:
    """FailureKind.CAPACITY_EXCEEDED routing and confidence."""

    def test_capacity_exceeded_with_fresh_t2(self):
        """T2 capacity hit with fresh telemetry → high confidence."""
        model_name = "qwen:36b"
        fc = gb_failures.classify_serve_failure(
            kind_hint=gb_failures.FailureKind.CAPACITY_EXCEEDED,
            model_name=model_name,
            t2_facts={"t2_free_mb": 512},
            t2_freshly_read=True,
        )
        assert fc.kind == gb_failures.FailureKind.CAPACITY_EXCEEDED
        assert fc.confidence == "high"
        assert model_name in fc.reason
        assert "old (cached)" not in fc.reason

    def test_capacity_exceeded_without_fresh_t2(self):
        """T2 capacity hit with stale/inferred telemetry → low confidence."""
        model_name = "qwen:36b"
        fc = gb_failures.classify_serve_failure(
            kind_hint=gb_failures.FailureKind.CAPACITY_EXCEEDED,
            model_name=model_name,
            t2_facts={"t2_free_mb": 512},
            t2_freshly_read=False,  # stale/inferred
        )
        assert fc.kind == gb_failures.FailureKind.CAPACITY_EXCEEDED
        assert fc.confidence == "low"
        assert "inferred or not freshly verified" in fc.reason

    def test_capacity_exceeded_exception_routing(self):
        """Exception with 'CPU offload' text routes to CAPACITY_EXCEEDED."""
        exc = RuntimeError("would fall back to CPU offload (dense partial)")
        fc = gb_failures.classify_serve_failure(
            exception=exc,
            model_name="test_model",
            t2_freshly_read=False,
        )
        assert fc.kind == gb_failures.FailureKind.CAPACITY_EXCEEDED
        assert fc.confidence == "low"


class TestFeederUnreachableClassification:
    """FailureKind.FEEDER_UNREACHABLE routing."""

    def test_feeder_unreachable_with_hostname(self):
        """Feeder check failed → high confidence with hostname."""
        hostname = "omen.local"
        fc = gb_failures.classify_serve_failure(
            kind_hint=gb_failures.FailureKind.FEEDER_UNREACHABLE,
            feeder_hostname=hostname,
            feeder_check_failed=True,
        )
        assert fc.kind == gb_failures.FailureKind.FEEDER_UNREACHABLE
        assert fc.confidence == "high"
        assert hostname in fc.reason
        assert "SSH" in fc.next_step
        assert "greenboost feeders diag" in fc.next_step

    def test_feeder_unreachable_flag_only(self):
        """Feeder check failed flag alone is sufficient."""
        fc = gb_failures.classify_serve_failure(
            feeder_check_failed=True,
        )
        assert fc.kind == gb_failures.FailureKind.FEEDER_UNREACHABLE
        assert fc.confidence == "high"


class TestPortConflictClassification:
    """FailureKind.PORT_CONFLICT routing."""

    def test_port_conflict_with_port_number(self):
        """Port conflict with explicit port number."""
        port = 11369
        fc = gb_failures.classify_serve_failure(
            kind_hint=gb_failures.FailureKind.PORT_CONFLICT,
            port_number=port,
        )
        assert fc.kind == gb_failures.FailureKind.PORT_CONFLICT
        assert fc.confidence == "high"
        assert str(port) in fc.reason
        assert "lsof" in fc.next_step

    def test_port_conflict_default_port(self):
        """Port conflict without explicit port uses GB_SYNAPSE_PORT."""
        fc = gb_failures.classify_serve_failure(
            kind_hint=gb_failures.FailureKind.PORT_CONFLICT,
        )
        assert fc.kind == gb_failures.FailureKind.PORT_CONFLICT
        assert "11369" in fc.reason  # default


class TestQualityGateFailedClassification:
    """FailureKind.QUALITY_GATE_FAILED routing."""

    def test_quality_gate_failed_with_result(self):
        """Quality check result with FAIL verdict."""
        model_name = "qwen:7b-q4"
        result = {
            "verdict": "FAIL",
            "reason": "repetition collapse detected (max 6-gram: 8, unique: 0.19)",
        }
        fc = gb_failures.classify_serve_failure(
            kind_hint=gb_failures.FailureKind.QUALITY_GATE_FAILED,
            model_name=model_name,
            quality_check_result=result,
        )
        assert fc.kind == gb_failures.FailureKind.QUALITY_GATE_FAILED
        assert fc.confidence == "high"
        assert model_name in fc.reason
        assert "repetition collapse" in fc.reason
        assert "niah_certify" in fc.next_step

    def test_quality_gate_failed_without_reason(self):
        """Quality check result without explicit reason."""
        result = {"verdict": "FAIL"}
        fc = gb_failures.classify_serve_failure(
            quality_check_result=result,
            model_name="test",
        )
        assert fc.kind == gb_failures.FailureKind.QUALITY_GATE_FAILED
        assert "unknown issue" in fc.reason


class TestEngineMissingClassification:
    """FailureKind.ENGINE_MISSING routing."""

    def test_engine_missing_hint(self):
        """Engine missing via kind_hint."""
        fc = gb_failures.classify_serve_failure(
            kind_hint=gb_failures.FailureKind.ENGINE_MISSING,
        )
        assert fc.kind == gb_failures.FailureKind.ENGINE_MISSING
        assert fc.confidence == "high"
        assert "not installed" in fc.reason
        assert "build-engine" in fc.next_step


class TestUnknownFailureClassification:
    """FailureKind.UNKNOWN as fallback."""

    def test_unknown_with_exception(self):
        """Unclassifiable exception → UNKNOWN."""
        exc = RuntimeError("something unexpected happened")
        fc = gb_failures.classify_serve_failure(
            exception=exc,
        )
        assert fc.kind == gb_failures.FailureKind.UNKNOWN
        assert fc.confidence == "low"
        assert "unexpected" in fc.reason.lower()
        assert "GB_DEBUG" in fc.next_step

    def test_unknown_no_context(self):
        """No context at all → UNKNOWN."""
        fc = gb_failures.classify_serve_failure()
        assert fc.kind == gb_failures.FailureKind.UNKNOWN
        assert fc.confidence == "low"


class TestConfidenceCaveats:
    """Confidence downgrade caveats appear correctly."""

    def test_stale_shim_caveat_format(self):
        """Stale shim cache caveat is properly formatted."""
        fc = gb_failures.classify_serve_failure(
            kind_hint=gb_failures.FailureKind.SHIM_BROKEN,
            shim_status=(False, "test"),
            shim_cache_age_s=3700,  # >1 hour
        )
        assert "3700" in fc.reason
        assert "configuration change" in fc.reason or "reboot" in fc.reason
        assert "Re-probe" in fc.reason

    def test_stale_t2_caveat_format(self):
        """Stale T2 memory caveat is properly formatted."""
        fc = gb_failures.classify_serve_failure(
            kind_hint=gb_failures.FailureKind.CAPACITY_EXCEEDED,
            t2_freshly_read=False,
        )
        assert "inferred or not freshly verified" in fc.reason
        assert "advisory" in fc.reason


class TestRealWorldScenarios:
    """Integration scenarios matching actual code paths."""

    def test_gate_cpu_offload_scenario(self):
        """Scenario: _gate_cpu_offload() raises, shim is off, T2 is stale."""
        exc = RuntimeError(
            "'qwen:36b' would fall back to CPU offload (dense partial) — "
            "CLAUDE.md's T2-spill rule forbids this by default"
        )
        fc = gb_failures.classify_serve_failure(
            exception=exc,
            model_name="qwen:36b",
            shim_status=(False, "shim not installed"),
            t2_facts={"t2_free_mb": 100},
            t2_freshly_read=False,
        )
        assert fc.kind == gb_failures.FailureKind.CAPACITY_EXCEEDED
        assert fc.confidence == "low"
        assert "T2 memory availability" in fc.reason or "advisory" in fc.reason
        assert "GB_SYNAPSE_ALLOW_CPU_OFFLOAD=1" in fc.next_step

    def test_ensemble_routing_quality_plus_capacity(self):
        """Multiple signals: quality check already failed, then capacity hit."""
        quality_result = {"verdict": "FAIL", "reason": "low bit-width"}
        fc = gb_failures.classify_serve_failure(
            quality_check_result=quality_result,
            model_name="test",
            t2_freshly_read=False,
        )
        # Quality check takes precedence in routing
        assert fc.kind == gb_failures.FailureKind.QUALITY_GATE_FAILED

    def test_shim_probe_failure_stale_cache(self):
        """Scenario: Shim probed 1 day ago, now we're retrying → low confidence."""
        fc = gb_failures.classify_serve_failure(
            shim_status=(False, "cudaFuncGetAttributes: invalid device function"),
            shim_cache_age_s=86400,  # 1 day
        )
        assert fc.kind == gb_failures.FailureKind.SHIM_BROKEN
        assert fc.confidence == "low"
        assert "86400" in fc.reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

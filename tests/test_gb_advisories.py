"""Tests for the unified advisory framework (gb_advisories)."""

import pytest
from gb_advisories import (
    Advisory,
    AdvisoryCheck,
    AdvisorySeverity,
    AdvisoryPhase,
    AdvisoryKind,
    BlockingAdvisoryError,
    assert_no_blocking,
    blocking_advisories,
    define_registry,
    run_advisories,
    format_advisories,
)


class TestAdvisoryRegistry:
    """Test advisory registry and duplicate detection."""

    def test_define_registry_accepts_empty(self):
        """Registry should accept empty check list."""
        reg = define_registry([])
        assert len(reg) == 0

    def test_define_registry_rejects_duplicates(self):
        """Registry should reject duplicate check IDs."""
        check1 = AdvisoryCheck(
            id="test.dup",
            phase=AdvisoryPhase.PREFLIGHT_HOST,
            severity=AdvisorySeverity.INFO,
            resume_safe=False,
            check=lambda ctx: None,
        )
        check2 = AdvisoryCheck(
            id="test.dup",
            phase=AdvisoryPhase.PREFLIGHT_HOST,
            severity=AdvisorySeverity.INFO,
            resume_safe=False,
            check=lambda ctx: None,
        )
        with pytest.raises(ValueError, match="Duplicate"):
            define_registry([check1, check2])

    def test_define_registry_returns_immutable_tuple(self):
        """Registry should return immutable tuple."""
        check = AdvisoryCheck(
            id="test.one",
            phase=AdvisoryPhase.PREFLIGHT_HOST,
            severity=AdvisorySeverity.INFO,
            resume_safe=False,
            check=lambda ctx: None,
        )
        reg = define_registry([check])
        assert isinstance(reg, tuple)


class TestAdvisoryMatchingCheck:
    """Test that advisories match their check's declared metadata."""

    def test_run_advisories_detects_id_mismatch(self):
        """Should raise if advisory.id != check.id."""
        def bad_check(ctx):
            return Advisory(
                id="mismatched.id",  # Wrong!
                severity=AdvisorySeverity.INFO,
                phase=AdvisoryPhase.PREFLIGHT_HOST,
                title="Test",
                reason="Mismatch test",
            )

        check = AdvisoryCheck(
            id="test.correct_id",
            phase=AdvisoryPhase.PREFLIGHT_HOST,
            severity=AdvisorySeverity.INFO,
            resume_safe=False,
            check=bad_check,
        )
        reg = define_registry([check])
        with pytest.raises(AssertionError, match="mismatched metadata"):
            run_advisories(reg)

    def test_run_advisories_detects_severity_mismatch(self):
        """Should raise if advisory.severity != check.severity."""
        def bad_check(ctx):
            return Advisory(
                id="test.id",
                severity=AdvisorySeverity.FATAL,  # Wrong!
                phase=AdvisoryPhase.PREFLIGHT_HOST,
                title="Test",
                reason="Severity mismatch",
            )

        check = AdvisoryCheck(
            id="test.id",
            phase=AdvisoryPhase.PREFLIGHT_HOST,
            severity=AdvisorySeverity.WARNING,
            resume_safe=False,
            check=bad_check,
        )
        reg = define_registry([check])
        with pytest.raises(AssertionError, match="mismatched metadata"):
            run_advisories(reg)


class TestRunAdvisories:
    """Test advisory execution and filtering."""

    def test_run_advisories_executes_all_by_default(self):
        """Should execute all checks when no phase filter."""
        executed = []

        def check_fn(ctx):
            executed.append(True)
            return Advisory(
                id="test.id",
                severity=AdvisorySeverity.INFO,
                phase=AdvisoryPhase.PREFLIGHT_HOST,
                title="Test",
                reason="Test advisory",
            )

        check = AdvisoryCheck(
            id="test.id",
            phase=AdvisoryPhase.PREFLIGHT_HOST,
            severity=AdvisorySeverity.INFO,
            resume_safe=False,
            check=check_fn,
        )
        reg = define_registry([check])
        result = run_advisories(reg)
        assert len(executed) == 1
        assert len(result.advisories) == 1

    def test_run_advisories_filters_by_phase(self):
        """Should filter checks by phase."""
        def check_fn(ctx):
            return Advisory(
                id="test.id",
                severity=AdvisorySeverity.INFO,
                phase=AdvisoryPhase.SERVE_PRE,
                title="Test",
                reason="Test",
            )

        check = AdvisoryCheck(
            id="test.id",
            phase=AdvisoryPhase.SERVE_PRE,
            severity=AdvisorySeverity.INFO,
            resume_safe=False,
            check=check_fn,
        )
        reg = define_registry([check])

        # Query only PREFLIGHT phase — should skip SERVE_PRE check
        result = run_advisories(reg, phase=AdvisoryPhase.PREFLIGHT_HOST)
        assert len(result.advisories) == 0
        assert len(result.executed_check_ids) == 0

    def test_run_advisories_skips_on_skip_if(self):
        """Should skip checks that match skip_if predicate."""
        check_fn_called = []

        def check_fn(ctx):
            check_fn_called.append(True)
            return None

        def skip_fn(ctx):
            return True  # Always skip

        check = AdvisoryCheck(
            id="test.id",
            phase=AdvisoryPhase.PREFLIGHT_HOST,
            severity=AdvisorySeverity.INFO,
            resume_safe=False,
            check=check_fn,
            skip_if=skip_fn,
        )
        reg = define_registry([check])
        result = run_advisories(reg)
        assert len(check_fn_called) == 0  # Never called
        assert len(result.advisories) == 0

    def test_run_advisories_caches_resume_safe(self):
        """Should reuse cached results for resume_safe checks."""
        call_count = [0]

        def check_fn(ctx):
            call_count[0] += 1
            return Advisory(
                id="test.id",
                severity=AdvisorySeverity.INFO,
                phase=AdvisoryPhase.PREFLIGHT_HOST,
                title="Test",
                reason="Test",
                resume_safe=True,  # Must match check's resume_safe=True
            )

        check = AdvisoryCheck(
            id="test.id",
            phase=AdvisoryPhase.PREFLIGHT_HOST,
            severity=AdvisorySeverity.INFO,
            resume_safe=True,  # Resume-safe
            check=check_fn,
        )
        reg = define_registry([check])

        # First run
        result1 = run_advisories(reg)
        assert call_count[0] == 1
        assert len(result1.advisories) == 1

        # Resume with cached results
        result2 = run_advisories(reg, resuming=True, cached_results=result1.results)
        assert call_count[0] == 1  # Not called again
        assert len(result2.reused_check_ids) == 1

    def test_run_advisories_reruns_non_resume_safe(self):
        """Should re-run checks that are not resume_safe."""
        call_count = [0]

        def check_fn(ctx):
            call_count[0] += 1
            return Advisory(
                id="test.id",
                severity=AdvisorySeverity.INFO,
                phase=AdvisoryPhase.PREFLIGHT_HOST,
                title="Test",
                reason="Test",
            )

        check = AdvisoryCheck(
            id="test.id",
            phase=AdvisoryPhase.PREFLIGHT_HOST,
            severity=AdvisorySeverity.INFO,
            resume_safe=False,  # Not resume-safe
            check=check_fn,
        )
        reg = define_registry([check])

        result1 = run_advisories(reg)
        assert call_count[0] == 1

        # Resume with cached results
        result2 = run_advisories(reg, resuming=True, cached_results=result1.results)
        assert call_count[0] == 2  # Called again


class TestBlockingAdvisories:
    """Test blocking advisory filtering and error handling."""

    def test_blocking_advisories_filters_correct_severities(self):
        """Should only return FATAL and BLOCKING severities."""
        advisories = [
            Advisory(
                id="fatal",
                severity=AdvisorySeverity.FATAL,
                phase=AdvisoryPhase.PREFLIGHT_HOST,
                title="Fatal",
                reason="Fatal issue",
            ),
            Advisory(
                id="blocking",
                severity=AdvisorySeverity.BLOCKING,
                phase=AdvisoryPhase.PREFLIGHT_HOST,
                title="Blocking",
                reason="Blocking issue",
            ),
            Advisory(
                id="warning",
                severity=AdvisorySeverity.WARNING,
                phase=AdvisoryPhase.PREFLIGHT_HOST,
                title="Warning",
                reason="Non-blocking warning",
            ),
            Advisory(
                id="info",
                severity=AdvisorySeverity.INFO,
                phase=AdvisoryPhase.PREFLIGHT_HOST,
                title="Info",
                reason="Informational",
            ),
        ]
        blocking = blocking_advisories(advisories)
        assert len(blocking) == 2
        assert all(a.id in ("fatal", "blocking") for a in blocking)

    def test_assert_no_blocking_passes_when_none(self):
        """Should not raise when no blocking advisories."""
        advisories = [
            Advisory(
                id="info",
                severity=AdvisorySeverity.INFO,
                phase=AdvisoryPhase.PREFLIGHT_HOST,
                title="Info",
                reason="OK",
            ),
        ]
        assert_no_blocking(advisories)  # Should not raise

    def test_assert_no_blocking_raises_when_present(self):
        """Should raise BlockingAdvisoryError when blocking present."""
        advisories = [
            Advisory(
                id="fatal",
                severity=AdvisorySeverity.FATAL,
                phase=AdvisoryPhase.PREFLIGHT_HOST,
                title="Fatal",
                reason="Fatal",
            ),
        ]
        with pytest.raises(BlockingAdvisoryError) as exc_info:
            assert_no_blocking(advisories)
        assert exc_info.value.advisories == advisories


class TestFormatAdvisories:
    """Test advisory formatting."""

    def test_format_advisories_json(self):
        """Should format advisories as JSON."""
        advisories = [
            Advisory(
                id="test.id",
                severity=AdvisorySeverity.WARNING,
                phase=AdvisoryPhase.PREFLIGHT_HOST,
                title="Test Advisory",
                reason="Test reason",
                commands=("cmd1", "cmd2"),
                docs_url="https://example.com",
            ),
        ]
        formatted = format_advisories(advisories, fmt="json")
        assert '"id"' in formatted
        assert '"severity"' in formatted
        assert "test.id" in formatted

    def test_format_advisories_console(self):
        """Should format advisories as console text."""
        advisories = [
            Advisory(
                id="test.id",
                severity=AdvisorySeverity.WARNING,
                phase=AdvisoryPhase.PREFLIGHT_HOST,
                title="Test Advisory",
                reason="Test reason",
                commands=("cmd1", "cmd2"),
            ),
        ]
        formatted = format_advisories(advisories, fmt="console")
        assert "[WARNING]" in formatted
        assert "test.id" in formatted
        assert "cmd1" in formatted


class TestAdvisoryToDict:
    """Test Advisory serialization."""

    def test_advisory_to_dict_preserves_enum_values(self):
        """Should convert enums to string values in dict."""
        adv = Advisory(
            id="test.id",
            severity=AdvisorySeverity.BLOCKING,
            phase=AdvisoryPhase.RUNTIME_TIER,
            title="Test",
            reason="Test",
            kind=AdvisoryKind.SUDO,
        )
        d = adv.to_dict()
        assert d["severity"] == "blocking"
        assert d["phase"] == "runtime.tier"
        assert d["kind"] == "sudo"

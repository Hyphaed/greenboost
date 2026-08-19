# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
Unit tests for gb_phase_activity module (process-wide phase label registry).

Tests verify:
- Basic push/pop via context manager
- Nested contexts report the innermost label
- Identity-based removal prevents string-value confusion
- Exception safety (finally ensures pop happens)
- Same label string pushed twice in nested contexts is popped correctly
"""
import pytest

from gb_phase_activity import (
    mark_phase_activity,
    current_phase_activity_label,
    _active_entries,
)


@pytest.fixture(autouse=True)
def _reset_phase_stack():
    """Reset the global phase activity stack before each test."""
    _active_entries.clear()
    yield
    _active_entries.clear()


class TestPhaseActivityBasics:
    """Basic mark_phase_activity context manager functionality."""

    def test_no_activity_initially(self):
        """When no phase is active, current_phase_activity_label() returns None."""
        assert current_phase_activity_label() is None

    def test_mark_single_phase(self):
        """Entering a phase context sets the current label."""
        assert current_phase_activity_label() is None
        with mark_phase_activity("phase-1"):
            assert current_phase_activity_label() == "phase-1"
        assert current_phase_activity_label() is None

    def test_nested_phases_report_innermost(self):
        """In nested contexts, the innermost (most recent) label is active."""
        with mark_phase_activity("outer"):
            assert current_phase_activity_label() == "outer"
            with mark_phase_activity("inner"):
                assert current_phase_activity_label() == "inner"
            assert current_phase_activity_label() == "outer"
        assert current_phase_activity_label() is None

    def test_three_level_nesting(self):
        """Three nested contexts pop in the correct order."""
        with mark_phase_activity("level-1"):
            assert current_phase_activity_label() == "level-1"
            with mark_phase_activity("level-2"):
                assert current_phase_activity_label() == "level-2"
                with mark_phase_activity("level-3"):
                    assert current_phase_activity_label() == "level-3"
                assert current_phase_activity_label() == "level-2"
            assert current_phase_activity_label() == "level-1"
        assert current_phase_activity_label() is None


class TestPhaseActivityExceptionSafety:
    """Exception safety: pop happens via finally even if the block raises."""

    def test_exception_in_context_still_pops(self):
        """If the context body raises, the label is still removed on exit."""
        with pytest.raises(ValueError):
            with mark_phase_activity("phase-error"):
                assert current_phase_activity_label() == "phase-error"
                raise ValueError("test error")
        # After the context exits (via exception), the label is gone
        assert current_phase_activity_label() is None

    def test_nested_exception_leaves_outer_intact(self):
        """If an inner context raises, the outer label remains active."""
        with mark_phase_activity("outer"):
            assert current_phase_activity_label() == "outer"
            with pytest.raises(RuntimeError):
                with mark_phase_activity("inner"):
                    assert current_phase_activity_label() == "inner"
                    raise RuntimeError("inner error")
            # After inner context exits, outer is still active
            assert current_phase_activity_label() == "outer"
        assert current_phase_activity_label() is None


class TestPhaseActivityIdentityRemoval:
    """Identity-based removal: same label string pushed twice is popped correctly.

    This tests the edge case that motivated identity-based removal in NemoClaw:
    if two contexts use the exact same label string and they nest, they must
    still pop in the correct order. Value-based removal (list.remove(value))
    would remove the FIRST matching value, which would be wrong if the first
    one hasn't exited yet.
    """

    def test_identical_labels_nested_pop_correctly(self):
        """Two identical label strings in nested contexts pop in the right order."""
        # This is the key test: both contexts use "duplicate-label".
        # Value-based removal would pop the wrong entry.
        with mark_phase_activity("duplicate-label"):
            assert current_phase_activity_label() == "duplicate-label"
            with mark_phase_activity("duplicate-label"):
                assert current_phase_activity_label() == "duplicate-label"
                # At this point, the stack has TWO "duplicate-label" entries.
                # The innermost one (most recent push) is active.
                assert len(_active_entries) == 2
            # Exiting the inner context: identity-based removal pops the SECOND
            # entry (the one we created in the inner context), not the first.
            assert current_phase_activity_label() == "duplicate-label"
            assert len(_active_entries) == 1
        # Exiting the outer context pops the first entry.
        assert current_phase_activity_label() is None
        assert len(_active_entries) == 0

    def test_identical_labels_with_other_labels_in_between(self):
        """Identical labels with intervening different labels all pop correctly."""
        with mark_phase_activity("duplicate"):
            assert current_phase_activity_label() == "duplicate"
            with mark_phase_activity("different"):
                assert current_phase_activity_label() == "different"
                with mark_phase_activity("duplicate"):
                    assert current_phase_activity_label() == "duplicate"
                    assert len(_active_entries) == 3
                # Exit innermost "duplicate": should pop only that entry
                assert current_phase_activity_label() == "different"
                assert len(_active_entries) == 2
            # Exit "different"
            assert current_phase_activity_label() == "duplicate"
            assert len(_active_entries) == 1
        assert current_phase_activity_label() is None


class TestPhaseActivityStackInvariants:
    """Verify global stack state invariants."""

    def test_stack_empty_when_all_contexts_exit(self):
        """The global stack is empty when all contexts have exited."""
        assert len(_active_entries) == 0
        with mark_phase_activity("a"):
            assert len(_active_entries) == 1
            with mark_phase_activity("b"):
                assert len(_active_entries) == 2
            assert len(_active_entries) == 1
        assert len(_active_entries) == 0

    def test_label_attributes_preserved(self):
        """Labels are preserved exactly as passed, including whitespace."""
        label_with_spaces = "  downloading weights (retry 3/5)  "
        with mark_phase_activity(label_with_spaces):
            assert current_phase_activity_label() == label_with_spaces
        assert current_phase_activity_label() is None

    def test_empty_label(self):
        """An empty string label is allowed (though not recommended)."""
        with mark_phase_activity(""):
            assert current_phase_activity_label() == ""
        assert current_phase_activity_label() is None

    def test_unicode_labels(self):
        """Unicode characters in labels work correctly."""
        label = "🔄 syncing 下载 модель"
        with mark_phase_activity(label):
            assert current_phase_activity_label() == label
        assert current_phase_activity_label() is None


class TestPhaseActivitySequentialNesting:
    """Sequential non-overlapping contexts (not nested) work correctly."""

    def test_sequential_phases(self):
        """Multiple sequential (non-overlapping) contexts pop and reset correctly."""
        with mark_phase_activity("phase-a"):
            assert current_phase_activity_label() == "phase-a"
        assert current_phase_activity_label() is None

        with mark_phase_activity("phase-b"):
            assert current_phase_activity_label() == "phase-b"
        assert current_phase_activity_label() is None

        with mark_phase_activity("phase-c"):
            assert current_phase_activity_label() == "phase-c"
        assert current_phase_activity_label() is None

        assert len(_active_entries) == 0

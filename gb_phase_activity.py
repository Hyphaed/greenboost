# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
Process-wide registry of the long-running sub-stage that currently owns
progress output.

Some operations run work that far outlives the decision-making phase they were
initiated from. For example, a quantization phase might trigger a HuggingFace
model download that takes far longer than the phase-selection logic itself. This
module allows nested sub-operations to register their own accurate label so a
progress heartbeat can print what is ACTUALLY happening instead of a stale
outer-phase name through a long nested operation.

Ported from NemoClaw's core/phase-activity.ts (Apache-2.0, design reference
only). The mechanism uses identity-based (object-reference) removal rather than
value-based removal to handle the edge case where the same label string is used
multiple times in nested calls and is popped out of order — identity ensures we
always pop the CORRECT entry.
"""
from contextlib import contextmanager
from typing import Optional


# Each active entry holds an object identity token + the label string.
# Object identity is the key; the label is metadata.
class _PhaseActivityEntry:
    """Sentinel object for identity-based removal. Each push creates a new entry;
    pop uses the returned entry's identity to find and remove the correct one."""
    def __init__(self, label: str):
        self.label = label


_active_entries: list[_PhaseActivityEntry] = []


@contextmanager
def mark_phase_activity(label: str):
    """
    Context manager: register a long-running sub-stage label for the duration
    of the `with` block. The label is added to the stack on entry and removed
    (by identity) on exit, even if an exception is raised.

    Use this in long-running operations (model downloads, quantization, builds)
    to ensure a progress heartbeat shows what is ACTUALLY happening:

        with mark_phase_activity("downloading weights"):
            download_large_model()

    If nested:

        with mark_phase_activity("quantize"):
            with mark_phase_activity("downloading checkpoint"):
                hf_hub_download(...)  # heartbeat prints "downloading checkpoint"
            with mark_phase_activity("compressing weights"):
                compress(...)         # heartbeat prints "compressing weights"
            # back to "quantize"

    Args:
        label: A human-readable description of the current sub-stage.

    Yields:
        None (the context manager has no return value).
    """
    entry = _PhaseActivityEntry(label)
    _active_entries.append(entry)
    try:
        yield
    finally:
        # Remove by identity: find the entry object we created, not just a
        # value-equal string. This ensures the correct entry is popped even
        # if the same label string was pushed multiple times in nested calls
        # and they're popped out of order.
        try:
            _active_entries.remove(entry)
        except ValueError:
            # Should never happen in normal use (entry was added above).
            # If a previous caller already removed it, silently ignore.
            pass


def current_phase_activity_label() -> Optional[str]:
    """
    Return the label of the innermost (most recently registered, still-active)
    phase activity, or None if no phase is currently active.

    This is the function a progress heartbeat should call to determine what
    to print:

        label = current_phase_activity_label()
        if label:
            print(f"  [...] {label}...")
        else:
            print("  [...] (no active sub-stage)...")

    Returns:
        The label string of the innermost active phase, or None.
    """
    if _active_entries:
        return _active_entries[-1].label
    return None

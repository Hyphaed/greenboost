#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
Unit tests for gb_state_io.py atomic and locked JSON I/O.
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

import sys

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gb_state_io import atomic_write_json, update_json_locked


class TestAtomicWriteJson:
    """Tests for atomic_write_json() — temp-file-then-rename pattern."""

    def test_writes_json_correctly(self):
        """Basic write: data is correctly serialized and persists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            obj = {"key": "value", "nested": {"a": 1, "b": 2}}

            atomic_write_json(path, obj, indent=2)

            assert path.exists()
            loaded = json.loads(path.read_text())
            assert loaded == obj

    def test_no_partial_file_on_success(self):
        """On success, no .partial.PID file is left behind."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            obj = {"data": "test"}

            atomic_write_json(path, obj)

            assert path.exists()
            # No .partial.* files should exist
            partial_files = list(Path(tmpdir).glob("*.partial.*"))
            assert len(partial_files) == 0

    def test_preserves_existing_file_on_exception(self):
        """If write fails mid-way, the original file is untouched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            original_obj = {"original": "data"}

            # Write original file.
            atomic_write_json(path, original_obj)
            original_content = path.read_text()

            # Patch json.dumps to raise an exception after the first call
            # (the first call is our successful write above).
            call_count = [0]

            def raising_dumps(obj, **kwargs):
                call_count[0] += 1
                if call_count[0] == 2:
                    raise RuntimeError("Simulated json.dumps failure")
                return json.dumps(obj, **kwargs)

            new_obj = {"new": "data"}
            with patch("gb_state_io.json.dumps", side_effect=raising_dumps):
                with pytest.raises(RuntimeError, match="Simulated json.dumps failure"):
                    atomic_write_json(path, new_obj)

            # Original file is untouched.
            assert path.read_text() == original_content
            loaded = json.loads(path.read_text())
            assert loaded == original_obj

    def test_creates_parent_directories(self):
        """Parent directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "a" / "b" / "c" / "test.json"
            obj = {"test": "data"}

            atomic_write_json(path, obj)

            assert path.exists()
            loaded = json.loads(path.read_text())
            assert loaded == obj

    def test_indent_parameter(self):
        """Indent parameter controls JSON formatting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            obj = {"key": "value"}

            atomic_write_json(path, obj, indent=4)
            content = path.read_text()

            # With indent=4, the JSON should have 4-space indentation.
            assert "    " in content  # 4 spaces for indentation

    def test_overwrites_existing_file(self):
        """atomic_write_json correctly replaces an existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"

            # Write first version.
            atomic_write_json(path, {"version": 1})
            assert json.loads(path.read_text())["version"] == 1

            # Overwrite with second version.
            atomic_write_json(path, {"version": 2})
            assert json.loads(path.read_text())["version"] == 2


class TestUpdateJsonLocked:
    """Tests for update_json_locked() — lock, read, mutate, write pattern."""

    def test_basic_read_mutate_write(self):
        """Basic case: read existing dict, mutate, write back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            initial = {"count": 1, "items": []}
            atomic_write_json(path, initial)

            def mutate_fn(state):
                state["count"] += 1
                state["items"].append("new_item")
                return state

            result = update_json_locked(path, mutate_fn)

            assert result["count"] == 2
            assert result["items"] == ["new_item"]

            # Verify persistence.
            persisted = json.loads(path.read_text())
            assert persisted["count"] == 2
            assert persisted["items"] == ["new_item"]

    def test_uses_default_if_file_missing(self):
        """If the file doesn't exist, the default dict is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"
            default_dict = {"created": True}

            def mutate_fn(state):
                state["mutated"] = True
                return state

            result = update_json_locked(path, mutate_fn, default=default_dict)

            assert result["created"] is True
            assert result["mutated"] is True

            # File should now exist with the mutated content.
            persisted = json.loads(path.read_text())
            assert persisted["created"] is True
            assert persisted["mutated"] is True

    def test_default_empty_dict_if_not_specified(self):
        """If file missing and no default given, uses empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.json"

            def mutate_fn(state):
                state["key"] = "value"
                return state

            result = update_json_locked(path, mutate_fn)

            assert result == {"key": "value"}
            persisted = json.loads(path.read_text())
            assert persisted == {"key": "value"}

    def test_mutate_returns_new_dict(self):
        """Mutate can return a new dict instead of modifying in place."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            atomic_write_json(path, {"original": "data"})

            def mutate_fn(state):
                # Return a completely new dict, don't mutate the old one.
                return {"new": "state", "original_preserved": state.get("original")}

            result = update_json_locked(path, mutate_fn)

            assert result == {"new": "state", "original_preserved": "data"}
            persisted = json.loads(path.read_text())
            assert persisted == {"new": "state", "original_preserved": "data"}

    def test_concurrent_updates_no_lost_writes(self):
        """Two sequential locked updates both persist (no lost-update race)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            initial = {"a": 0, "b": 0}
            atomic_write_json(path, initial)

            # First update: add key 'a'.
            def update_a(state):
                state["a"] += 1
                return state

            update_json_locked(path, update_a)

            # Second update: add key 'b'.
            def update_b(state):
                state["b"] += 1
                return state

            update_json_locked(path, update_b)

            # Both updates should be present.
            final = json.loads(path.read_text())
            assert final["a"] == 1
            assert final["b"] == 1

    def test_lock_file_created(self):
        """Lock file is created as a sidecar."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"

            def mutate_fn(state):
                return state

            update_json_locked(path, mutate_fn, default={})

            lock_path = Path(f"{path}.lock")
            assert lock_path.exists()

    def test_exception_during_mutate_preserves_original(self):
        """If mutate raises, the original file is untouched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            original = {"count": 5}
            atomic_write_json(path, original)

            def failing_mutate(state):
                raise ValueError("Mutate failed")

            with pytest.raises(ValueError, match="Mutate failed"):
                update_json_locked(path, failing_mutate)

            # Original should be unchanged.
            persisted = json.loads(path.read_text())
            assert persisted == original

    def test_multiple_concurrent_keys_not_lost(self):
        """Simulates (sequentially) what concurrent writers would do.

        This mirrors the KV measurement cache pattern where each key is
        a stringified tuple like "model1:2048:q8".
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "measurements.json"

            # First "writer": add measurement for model1, ctx 2048.
            def add_model1(state):
                state["model1:2048:q8"] = 512.0
                return state

            update_json_locked(path, add_model1, default={})

            # Second "writer": add measurement for model2, ctx 4096.
            def add_model2(state):
                state["model2:4096:q4"] = 256.0
                return state

            update_json_locked(path, add_model2)

            # Both measurements should be present.
            final = json.loads(path.read_text())
            assert final["model1:2048:q8"] == 512.0
            assert final["model2:4096:q4"] == 256.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

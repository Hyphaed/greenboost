#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_state_io.py — Atomic and locked file I/O for GreenBoost state files.

Provides two core utilities:
  1. atomic_write_json() — write to a temp file, then rename atomically
  2. update_json_locked() — lock, read, mutate, write, unlock (read-modify-write)

Both patterns prevent corruption from concurrent writers and partial writes.
Port of NemoClaw's tarball.ts discipline into Python, adapted for state
persistence.

See gb_synapse.py for usage: measured tok/s, run state, KV measurements.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


def atomic_write_json(path: Path, obj: Any, indent: int = 2) -> None:
    """Write JSON to disk atomically via temp-file-then-rename.

    Writes to a sidecar `.partial.<pid>` file first, then atomically
    renames it to the target path via os.replace(). If the write fails
    (exception during json.dumps or file write), the partial file is
    cleaned up on best-effort basis, leaving any pre-existing file at
    `path` untouched.

    Args:
        path: Final destination path for the JSON file.
        obj: Object to serialize to JSON.
        indent: JSON indentation level (default 2).

    Raises:
        OSError or json.JSONEncodeError if the write fails and cleanup cannot
        complete (though partial file cleanup is best-effort and never masks
        the original exception).
    """
    path_parent = path.parent
    path_parent.mkdir(parents=True, exist_ok=True)

    # Temp file in same directory to ensure atomic rename works (same filesystem).
    partial_path = Path(f"{path}.partial.{os.getpid()}")

    try:
        # Write to temp file.
        partial_path.write_text(json.dumps(obj, indent=indent))
        # Atomic rename on POSIX. If this succeeds, the operation is durable.
        # If it fails, partial_path still exists and the exception propagates.
        os.replace(partial_path, path)
    except Exception:
        # Best-effort cleanup: remove the partial file if it exists.
        # Do not raise from cleanup — preserve the original exception.
        try:
            partial_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def update_json_locked(
    path: Path,
    mutate: Callable[[dict], dict],
    default: Optional[dict] = None,
) -> dict:
    """Read JSON, mutate, and write back — all under an exclusive file lock.

    Acquires an exclusive lock on a sidecar lock file, reads the current
    JSON (or uses `default` if the file doesn't exist), calls `mutate` with
    the current dict, and writes the returned dict back via atomic_write_json(),
    all while holding the lock. This prevents lost updates from concurrent
    readers and writers.

    Args:
        path: Path to the JSON file to read/mutate/write.
        mutate: Callable that takes the current dict and returns the dict to
                write. May mutate in place or return a new dict; the function's
                return value is what gets persisted.
        default: Default dict if the file doesn't exist (default: None, which
                 becomes {} if the file is missing).

    Returns:
        The dict that was written (the return value of mutate()).

    Raises:
        OSError if locking, read, or write fails. Lock is released in all
        cases via try/finally.
    """
    if default is None:
        default = {}

    lock_path = Path(f"{path}.lock")

    # Open lock file in append mode so it doesn't clobber anything;
    # 0o644 perms (readable by all, writable by owner).
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as lock_file:
        try:
            # Acquire exclusive lock. LOCK_EX blocks until acquired.
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                # Read current state (or use default).
                try:
                    current = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    current = default

                # Mutate and get the new state.
                new_state = mutate(current)

                # Atomic write while still holding lock.
                atomic_write_json(path, new_state)

                return new_state
            finally:
                # Release lock.
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            # If anything fails, lock is released by the finally above.
            raise

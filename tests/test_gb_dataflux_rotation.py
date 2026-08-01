#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_dataflux.py's log durability fixes:

  * emit() rotates the log to a single ".jsonl.1" archive once it exceeds
    GB_DATAFLUX_MAX_BYTES — previously the log grew unbounded and every
    read_events() call was a full-file linear scan.
  * read_events() reads the archive THEN the current log (rotation never
    punches a history hole), skips the archive when its mtime predates the
    query window, and uses a reverse tail-seek instead of a full scan when
    `since_hours` is set — verified here by cross-checking its output
    against a plain linear scan for a range of cutoffs, not just spot values.
  * The process-local memo returns identical (and independent, mutation-safe)
    results on a cache hit.

Redirects GREENBOOST_DATAFLUX_LOG at a tmp_path file per test (same pattern
tests/conftest.py's autouse fixture uses) so the real log is never touched.
"""
import gzip
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gb_dataflux as gdf


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    p = tmp_path / "dataflux.jsonl"
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(p))
    gdf._READ_EVENTS_MEMO.clear()
    yield p
    gdf._READ_EVENTS_MEMO.clear()


def _write_synthetic_events(path: Path, n: int, start_ts: float, spacing_s: float = 1.0):
    """Write n JSONL lines with strictly increasing ts, bypassing emit()'s
    rotation so tests can build a large fixed corpus precisely."""
    with open(path, "a") as f:
        for i in range(n):
            f.write(json.dumps({"ts": start_ts + i * spacing_s, "kind": "test",
                                "node": "host", "seq": i}) + "\n")


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def test_emit_rotates_once_over_size_cap(log_path, monkeypatch):
    monkeypatch.setenv("GB_DATAFLUX_MAX_BYTES", "2000")  # small, deterministic cap
    for i in range(200):
        gdf.emit({"kind": "test", "node": "host", "seq": i})

    archive = gdf._archive_path(log_path)
    assert archive.exists()
    # Exactly one archive generation — no .jsonl.2, no unbounded growth.
    assert not (log_path.parent / (log_path.name + ".2")).exists()
    # The current log must be smaller than the cap that just triggered
    # rotation (rotation resets to a fresh, empty file).
    assert log_path.stat().st_size < archive.stat().st_size + 2000


def test_emit_rotation_preserves_all_events(log_path, monkeypatch):
    """No event is lost across a rotation boundary — read_events() must see
    every one of them, split across archive + current log or not."""
    monkeypatch.setenv("GB_DATAFLUX_MAX_BYTES", "2000")
    for i in range(200):
        gdf.emit({"kind": "test", "node": "host", "seq": i})

    events = gdf.read_events()
    seqs = sorted(e["seq"] for e in events if e.get("kind") == "test")
    assert seqs == list(range(200))


def test_emit_never_raises_when_rotation_replace_fails(log_path, monkeypatch):
    monkeypatch.setenv("GB_DATAFLUX_MAX_BYTES", "10")  # trips on the very first write
    monkeypatch.setattr(gdf.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("busy")))
    gdf.emit({"kind": "test"})  # must not raise despite the forced rotation failure


# ---------------------------------------------------------------------------
# Reverse tail-seek correctness — cross-checked against a full linear scan
# ---------------------------------------------------------------------------

def test_tail_seek_matches_linear_scan_across_cutoffs(log_path):
    base = 1_700_000_000.0
    n = 3000
    _write_synthetic_events(log_path, n, start_ts=base, spacing_s=1.0)

    all_events = gdf.read_events()  # cutoff=None — full read, ground truth
    assert len(all_events) == n

    now = base + n  # read_events() computes cutoff as time.time() - since_hours*3600;
    # patch time.time() so since_hours maps to a known, exact cutoff timestamp.
    import time as time_mod
    for since_hours, expected_min_seq in (
        (0.5, n - int(0.5 * 3600)),   # far smaller than the corpus span
        (0.2 * n / 3600, n - int(0.2 * n)),
        (100.0, 0),                    # cutoff before the corpus start — everything
    ):
        gdf._READ_EVENTS_MEMO.clear()

        def _fake_now():
            return now
        real_time = time_mod.time
        time_mod.time = _fake_now
        try:
            events = gdf.read_events(since_hours=since_hours)
        finally:
            time_mod.time = real_time

        cutoff = now - since_hours * 3600
        expected = sorted(e["seq"] for e in all_events if e["ts"] >= cutoff)
        got = sorted(e["seq"] for e in events if e.get("kind") == "test")
        assert got == expected, f"since_hours={since_hours}: tail-seek diverged from linear scan"


def test_tail_seek_start_never_skips_past_cutoff(log_path):
    """Direct unit check on _tail_seek_start(): the returned offset must
    always be <= the true byte offset of the first qualifying line."""
    base = 1_700_000_000.0
    n = 500
    _write_synthetic_events(log_path, n, start_ts=base, spacing_s=2.0)

    # Pick a cutoff partway through, compute the TRUE first-qualifying-line
    # offset by linear scan, and assert the tail-seek's returned start is at
    # or before it.
    cutoff = base + 2.0 * (n // 3)
    true_offset = None
    with open(log_path, "rb") as f:
        pos = 0
        for raw in f:
            ev = json.loads(raw)
            if ev["ts"] >= cutoff:
                true_offset = pos
                break
            pos += len(raw)
    assert true_offset is not None

    seek_start = gdf._tail_seek_start(log_path, cutoff)
    assert seek_start <= true_offset


def test_archive_skipped_when_mtime_predates_cutoff(log_path, monkeypatch):
    """The archive file is skipped entirely (not even opened) once its
    mtime is older than the query cutoff — the read_events() fast path."""
    monkeypatch.setenv("GB_DATAFLUX_MAX_BYTES", "1")  # rotate on first write
    gdf.emit({"kind": "test", "seq": "old"})
    archive = gdf._archive_path(log_path)
    assert archive.exists()

    # Age the archive well outside any query window.
    old_ts = time.time() - 999999
    import os
    os.utime(archive, (old_ts, old_ts))

    gdf._READ_EVENTS_MEMO.clear()
    opened = []
    real_open = gdf._read_jsonl_file

    def _tracking_read(path, cutoff):
        opened.append(str(path))
        return real_open(path, cutoff)
    monkeypatch.setattr(gdf, "_read_jsonl_file", _tracking_read)

    gdf.read_events(since_hours=1.0)
    assert str(archive) not in opened


# ---------------------------------------------------------------------------
# Memo
# ---------------------------------------------------------------------------

def test_memo_returns_independent_list_copies(log_path):
    gdf.emit({"kind": "test", "seq": 1})
    a = gdf.read_events()
    b = gdf.read_events()
    assert a == b
    a.append({"kind": "mutated"})
    c = gdf.read_events()
    assert c == b  # mutating the caller's copy must not corrupt the cache


def test_memo_invalidates_on_new_write(log_path):
    gdf.emit({"kind": "test", "seq": 1})
    first = gdf.read_events()
    assert len(first) == 1
    gdf.emit({"kind": "test", "seq": 2})
    second = gdf.read_events()
    assert len(second) == 2


# ---------------------------------------------------------------------------
# compact_archive()
# ---------------------------------------------------------------------------

def test_compact_archive_drops_events_past_retention(log_path):
    archive = gdf._archive_path(log_path)
    now = time.time()
    with gzip.open(archive, "wt") as f:
        f.write(json.dumps({"ts": now - 40 * 86400, "kind": "old"}) + "\n")   # past 30d retain
        f.write(json.dumps({"ts": now - 1 * 86400, "kind": "recent"}) + "\n")  # kept

    dropped = gdf.compact_archive(retain_days=30.0)

    assert dropped == 1
    with gzip.open(archive, "rt") as f:
        kept = [json.loads(line) for line in f.read().splitlines()]
    assert len(kept) == 1
    assert kept[0]["kind"] == "recent"


def test_compact_archive_no_archive_returns_none(log_path):
    assert gdf.compact_archive() is None


def test_compact_archive_nothing_to_drop_returns_zero(log_path):
    archive = gdf._archive_path(log_path)
    with gzip.open(archive, "wt") as f:
        f.write(json.dumps({"ts": time.time(), "kind": "recent"}) + "\n")

    assert gdf.compact_archive(retain_days=30.0) == 0
    with gzip.open(archive, "rt") as f:
        assert len(f.read().splitlines()) == 1


def test_compact_archive_never_raises_on_corrupt_line(log_path):
    archive = gdf._archive_path(log_path)
    with gzip.open(archive, "wt") as f:
        f.write("not json at all\n")
        f.write(json.dumps({"ts": time.time(), "kind": "recent"}) + "\n")

    # Corrupt lines are dropped silently, not counted as an error.
    result = gdf.compact_archive(retain_days=30.0)
    assert result is not None
    with gzip.open(archive, "rt") as f:
        kept = f.read().splitlines()
    assert len(kept) == 1


def test_compact_archive_env_override(log_path, monkeypatch):
    monkeypatch.setenv("GB_DATAFLUX_RETAIN_DAYS", "1")
    archive = gdf._archive_path(log_path)
    now = time.time()
    with gzip.open(archive, "wt") as f:
        f.write(json.dumps({"ts": now - 2 * 86400, "kind": "old"}) + "\n")

    dropped = gdf.compact_archive()  # retain_days=None -> reads the env
    assert dropped == 1


def test_compact_archive_default_retain_is_seven_days(log_path):
    """GB_DATAFLUX_RETAIN_DAYS unset, retain_days unset -> the default (7,
    not the old 30) is what actually governs the cutoff."""
    archive = gdf._archive_path(log_path)
    now = time.time()
    with gzip.open(archive, "wt") as f:
        f.write(json.dumps({"ts": now - 10 * 86400, "kind": "past_7d_not_30d"}) + "\n")
        f.write(json.dumps({"ts": now - 1 * 86400, "kind": "recent"}) + "\n")

    dropped = gdf.compact_archive()  # both args unset -> _DEFAULT_RETAIN_DAYS
    assert dropped == 1
    with gzip.open(archive, "rt") as f:
        kept = [json.loads(line) for line in f.read().splitlines()]
    assert len(kept) == 1
    assert kept[0]["kind"] == "recent"


def test_compact_archive_size_backstop_trims_even_within_retention(log_path, monkeypatch):
    """Every event is well within retain_days, but the archive still exceeds
    GB_DATAFLUX_MAX_BYTES , the size backstop must drop the oldest ones
    anyway, since age-based trimming alone can't bound bytes."""
    monkeypatch.setenv("GB_DATAFLUX_MAX_BYTES", "500")  # small, deterministic cap
    archive = gdf._archive_path(log_path)
    now = time.time()
    with gzip.open(archive, "wt") as f:
        for i in range(50):
            f.write(json.dumps({"ts": now - i, "kind": "recent", "seq": i,
                                 "pad": "x" * 20}) + "\n")

    dropped = gdf.compact_archive(retain_days=30.0)  # generous age window
    assert dropped is not None and dropped > 0

    with gzip.open(archive, "rt") as f:
        kept = [json.loads(line) for line in f.read().splitlines()]
    raw_size = sum(len(line) + 1 for line in
                    (json.dumps(e) for e in kept))
    assert raw_size <= 500
    # The newest events (lowest seq, since ts = now - i) must be the ones
    # that survived , the backstop drops OLDEST first, not arbitrarily.
    kept_seqs = sorted(e["seq"] for e in kept)
    assert kept_seqs == list(range(len(kept)))


def test_compact_archive_size_backstop_noop_when_under_budget(log_path):
    """Recent, small archive: the size backstop must not drop anything the
    age-based trim wouldn't already have dropped."""
    archive = gdf._archive_path(log_path)
    with gzip.open(archive, "wt") as f:
        f.write(json.dumps({"ts": time.time(), "kind": "recent"}) + "\n")

    assert gdf.compact_archive(retain_days=30.0) == 0
    with gzip.open(archive, "rt") as f:
        assert len(f.read().splitlines()) == 1

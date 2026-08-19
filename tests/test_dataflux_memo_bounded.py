"""The read_events() memo must stay bounded in a long-lived process.

Regression for the 2026-08-19 incident: gb-synapse's proxy imports gb_init,
whose SnapshotRecorder appends to dataflux.jsonl every 5 s and then ticks the
segment watcher, which calls read_events(). Every append changed the file's
(mtime, size), minting a fresh memo key and retaining the whole parsed event
list under it forever , +10.7 MB/s while idle, MemoryError within minutes.
"""
import json
import time

import gb_dataflux


def _log(tmp_path, monkeypatch):
    p = tmp_path / "dataflux.jsonl"
    p.write_text("")
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(p))
    gb_dataflux._READ_EVENTS_MEMO.clear()
    return p


def _append(p, i):
    with p.open("a") as f:
        f.write(json.dumps({"kind": "snapshot", "ts": time.time(), "i": i}) + "\n")
    # stat key must actually move, even on a coarse-mtime filesystem
    import os
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000 * (i + 1)))


def test_memo_does_not_grow_as_the_log_is_appended(tmp_path, monkeypatch):
    p = _log(tmp_path, monkeypatch)
    for i in range(60):
        _append(p, i)
        gb_dataflux.read_events(since_hours=48)
    assert len(gb_dataflux._READ_EVENTS_MEMO) <= gb_dataflux._READ_EVENTS_MEMO_MAX


def test_memo_bounded_across_varying_windows(tmp_path, monkeypatch):
    """since_hours is a float derived from window_s , a second unbounded axis."""
    p = _log(tmp_path, monkeypatch)
    _append(p, 0)
    for i in range(60):
        gb_dataflux.read_events(since_hours=(300 + i) / 3600.0)
    assert len(gb_dataflux._READ_EVENTS_MEMO) <= gb_dataflux._READ_EVENTS_MEMO_MAX


def test_superseded_generations_are_dropped_immediately(tmp_path, monkeypatch):
    """Same (path, window), newer file: the old entry can never be hit again."""
    p = _log(tmp_path, monkeypatch)
    _append(p, 0)
    gb_dataflux.read_events(since_hours=48)
    assert len(gb_dataflux._READ_EVENTS_MEMO) == 1
    _append(p, 1)
    gb_dataflux.read_events(since_hours=48)
    assert len(gb_dataflux._READ_EVENTS_MEMO) == 1


def test_memo_still_serves_repeat_reads_of_an_unchanged_file(tmp_path, monkeypatch):
    """The optimisation the memo exists for must survive the bounding."""
    p = _log(tmp_path, monkeypatch)
    _append(p, 0)
    first = gb_dataflux.read_events(since_hours=48)
    calls = []
    orig = gb_dataflux._read_jsonl_file
    monkeypatch.setattr(gb_dataflux, "_read_jsonl_file",
                        lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    again = gb_dataflux.read_events(since_hours=48)
    assert calls == []          # served from the memo, file never re-parsed
    assert again == first


def test_memo_returns_a_copy_not_the_cached_list(tmp_path, monkeypatch):
    p = _log(tmp_path, monkeypatch)
    _append(p, 0)
    a = gb_dataflux.read_events(since_hours=48)
    a.append({"kind": "mutated"})
    b = gb_dataflux.read_events(since_hours=48)
    assert not any(e.get("kind") == "mutated" for e in b)

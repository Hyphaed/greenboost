"""gb_memwatch must survive the conditions it exists to observe.

It runs while the box is under memory pressure and processes are dying, so
every read races a process that may vanish between scandir and open. A watcher
that raises during an OOM is worse than no watcher: it destroys the evidence it
was started to collect.
"""
import json
import os

import pytest

import gb_memwatch as M


def test_sample_returns_a_well_formed_record():
    rec = M.sample(top_n=3)
    assert rec["mem_total_mb"] > 0
    assert rec["mem_available_mb"] >= 0
    assert 0 < len(rec["top"]) <= 3
    for p in rec["top"]:
        assert p["pid"] > 0
        assert p["rss_mb"] >= 0
        assert isinstance(p["cmdline"], str)
        assert isinstance(p["rollup_kb"], dict)


def test_top_is_sorted_by_rss():
    rss = [p["rss_mb"] for p in M.sample(top_n=6)["top"]]
    assert rss == sorted(rss, reverse=True)


def test_reads_of_a_vanished_pid_do_not_raise():
    """The exact race this tool runs inside: the process dies mid-read."""
    dead = 999_999          # far above any live pid on a normal box
    assert M._read_text(f"/proc/{dead}/cmdline") == ""
    assert M._cmdline(dead) == ""
    assert M._rollup(dead) == {}
    assert M._statm(dead) == (0, 0)


def test_meminfo_field_missing_returns_zero_not_an_exception():
    assert M._meminfo_kb("NoSuchFieldHere") == 0


def test_alert_flag_tracks_the_threshold(monkeypatch):
    monkeypatch.setattr(M, "ALERT_FRACTION", 0.0)   # everything is over
    assert all(p["over_alert"] for p in M.sample(top_n=3)["top"])
    monkeypatch.setattr(M, "ALERT_FRACTION", 10.0)  # nothing can be
    assert not any(p["over_alert"] for p in M.sample(top_n=3)["top"])


def test_append_respects_the_size_cap(tmp_path, monkeypatch):
    """Filling the disk while investigating a memory problem makes it worse."""
    f = tmp_path / "snapshots.jsonl"
    monkeypatch.setattr(M, "STATE_DIR", tmp_path)
    monkeypatch.setattr(M, "SNAPSHOT_FILE", f)
    monkeypatch.setattr(M, "MAX_LOG_BYTES", 200)
    for _ in range(50):
        M._append({"ts": 1, "top": [], "pad": "x" * 100})
    assert f.stat().st_size < 1000, "size cap did not hold"


def test_append_writes_valid_jsonl(tmp_path, monkeypatch):
    f = tmp_path / "snapshots.jsonl"
    monkeypatch.setattr(M, "STATE_DIR", tmp_path)
    monkeypatch.setattr(M, "SNAPSHOT_FILE", f)
    monkeypatch.setattr(M, "MAX_LOG_BYTES", 10 * 1024 * 1024)
    M._append(M.sample(top_n=2))
    lines = f.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["top"]


def test_append_on_an_unwritable_dir_is_silent(monkeypatch):
    monkeypatch.setattr(M, "STATE_DIR", "/proc/definitely/not/writable")
    monkeypatch.setattr(M, "SNAPSHOT_FILE", "/proc/definitely/not/writable/x.jsonl")
    M._append({"ts": 1})            # must not raise


def test_report_without_snapshots_exits_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(M, "SNAPSHOT_FILE", tmp_path / "absent.jsonl")
    assert M.cmd_report(None) == 1
    assert "no snapshots yet" in capsys.readouterr().out


def test_report_skips_corrupt_lines(tmp_path, monkeypatch, capsys):
    """A snapshot truncated by the very OOM it recorded must not break the report."""
    f = tmp_path / "s.jsonl"
    good = json.dumps(M.sample(top_n=2))
    f.write_text(good + "\n{ truncated by the kill\n" + good + "\n")
    monkeypatch.setattr(M, "SNAPSHOT_FILE", f)
    assert M.cmd_report(None) == 0
    assert "2 samples" in capsys.readouterr().out


def test_sampling_is_cheap_enough_to_run_beside_inference():
    """The point is to run continuously; a costly pass would not be left on."""
    import time
    t = time.perf_counter()
    M.sample(top_n=8)
    assert (time.perf_counter() - t) < 1.0

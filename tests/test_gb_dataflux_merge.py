"""R3: the owner's reader merges the root supervisor's log (read-only); modes of new files; retention. Tmp dirs only."""
import gzip
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import gb_dataflux as dx


@pytest.fixture
def logs(tmp_path, monkeypatch):
    own = tmp_path / "home" / "dataflux.jsonl"
    pub = tmp_path / "pub"
    own.parent.mkdir()
    pub.mkdir()
    pub.chmod(0o755)                                     # like root's /var/lib/greenboost/pub, whatever the umask
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(own))
    monkeypatch.setenv("GB_DATAFLUX_MERGE_LOGS", str(pub / "dataflux-root.jsonl"))
    monkeypatch.delenv("GB_DATAFLUX_LOG_MODE", raising=False)
    dx._READ_EVENTS_MEMO.clear()
    return own, pub / "dataflux-root.jsonl"


def _w(p, evs):
    p.write_text("".join(json.dumps(e) + "\n" for e in evs))
    p.chmod(0o644)


def test_root_events_are_merged_in_time_order_and_tagged(logs):
    own, root = logs
    now = time.time()
    _w(own, [{"ts": now - 10, "kind": "a"}, {"ts": now - 1, "kind": "c"}])
    _w(root, [{"ts": now - 5, "kind": "b"}])
    evs = dx.read_events()
    assert [e["kind"] for e in evs] == ["a", "b", "c"]
    assert [e.get("src") for e in evs] == [None, "root", None]


def test_merge_is_read_only_nothing_created_or_changed_in_the_root_dir(logs):
    own, root = logs
    before = {p.name: p.stat().st_mtime_ns for p in root.parent.iterdir()}
    dx.read_events()                                     # root log absent: no file is created
    _w(root, [{"ts": time.time(), "kind": "b"}])
    snap = {p.name: p.stat().st_mtime_ns for p in root.parent.iterdir()}
    dx._READ_EVENTS_MEMO.clear()
    dx.read_events()
    dx.compact_archive()
    assert {p.name: p.stat().st_mtime_ns for p in root.parent.iterdir()} == snap and before == {}


def test_memo_sees_a_new_root_event(logs):
    own, root = logs
    _w(root, [{"ts": time.time() - 2, "kind": "b"}])
    assert len(dx.read_events()) == 1
    with open(root, "a") as f:
        f.write(json.dumps({"ts": time.time(), "kind": "b2"}) + "\n")
    assert len(dx.read_events()) == 2


def test_untrusted_merge_files_are_ignored(logs, tmp_path):
    own, root = logs
    real = tmp_path / "elsewhere.jsonl"
    _w(real, [{"ts": time.time(), "kind": "x"}])
    root.symlink_to(real)                                 # a symlink is refused
    assert dx.read_events() == []
    root.unlink()
    _w(root, [{"ts": time.time(), "kind": "x"}])
    root.chmod(0o666)                                     # group/world writable is refused
    assert dx.read_events() == []
    root.chmod(0o644)
    root.parent.chmod(0o777)                              # so is a writable directory
    try:
        assert dx.read_events() == []
    finally:
        root.parent.chmod(0o755)
    assert len(dx.read_events()) == 1


def test_root_events_older_than_retention_are_not_read(logs, monkeypatch):
    own, root = logs
    monkeypatch.setenv("GB_DATAFLUX_RETAIN_DAYS", "2")
    now = time.time()
    _w(root, [{"ts": now - 5 * 86400, "kind": "old"}, {"ts": now - 3600, "kind": "new"}])
    assert [e["kind"] for e in dx.read_events()] == ["new"]
    # an archive whose last write is older than the window is not opened at all
    arch = Path(str(root) + ".1.gz")
    with gzip.open(arch, "wt") as f:
        f.write(json.dumps({"ts": now - 4 * 86400, "kind": "arch"}) + "\n")
    os.utime(arch, (now - 4 * 86400, now - 4 * 86400))
    arch.chmod(0o644)
    dx._READ_EVENTS_MEMO.clear()
    assert [e["kind"] for e in dx.read_events()] == ["new"]


def test_since_hours_window_applies_to_merged_events(logs):
    own, root = logs
    now = time.time()
    _w(root, [{"ts": now - 7200, "kind": "b1"}, {"ts": now - 60, "kind": "b2"}])
    assert [e["kind"] for e in dx.read_events(since_hours=1)] == ["b2"]


def test_default_root_log_merges_only_for_the_default_owner_log(monkeypatch):
    monkeypatch.delenv("GB_DATAFLUX_MERGE_LOGS", raising=False)
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", "/x/custom.jsonl")
    assert dx._merge_logs(Path("/x/custom.jsonl")) == []
    monkeypatch.delenv("GREENBOOST_DATAFLUX_LOG")
    assert dx._merge_logs(Path("/h/dataflux.jsonl")) == [Path("/var/lib/greenboost/pub/dataflux-root.jsonl")]
    monkeypatch.setenv("GB_DATAFLUX_MERGE_LOGS", "")
    assert dx._merge_logs(Path("/h/dataflux.jsonl")) == []


def test_new_log_and_rotated_archive_are_0600_by_default_and_follow_the_mode_env(logs, monkeypatch):
    own, root = logs
    monkeypatch.setenv("GB_DATAFLUX_MAX_BYTES", "200")
    for i in range(12):
        dx.emit({"kind": "k", "i": i, "pad": "x" * 40})
    arch = Path(str(own) + ".1.gz")
    assert arch.exists() and oct(arch.stat().st_mode)[-3:] == "600"
    assert oct(own.stat().st_mode)[-3:] == "600" if own.exists() else True
    monkeypatch.setenv("GB_DATAFLUX_LOG_MODE", "644")     # the root supervisor's unit sets this
    own2 = own.parent / "other.jsonl"
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(own2))
    dx.emit({"kind": "k"})
    assert oct(own2.stat().st_mode)[-3:] == "644"
    monkeypatch.setenv("GB_DATAFLUX_LOG_MODE", "777")     # world-write is never honoured
    own3 = own.parent / "third.jsonl"
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(own3))
    dx.emit({"kind": "k"})
    assert oct(own3.stat().st_mode)[-3:] == "644"


def test_emit_refuses_a_symlinked_log_and_writes_nothing_through_it(logs, tmp_path):
    own, root = logs
    victim = tmp_path / "victim"
    victim.write_text("keep\n")
    own.symlink_to(victim)
    dx.emit({"kind": "k"})                                # best-effort: swallowed, but never followed
    assert victim.read_text() == "keep\n"


def test_compact_archive_keeps_the_archive_mode_and_survives_a_planted_temp(logs):
    own, root = logs
    arch = Path(str(own) + ".1.gz")
    now = time.time()
    with gzip.open(arch, "wt") as f:
        f.write(json.dumps({"ts": now - 30 * 86400, "kind": "old"}) + "\n" + json.dumps({"ts": now, "kind": "new"}) + "\n")
    arch.chmod(0o664)
    victim = own.parent / "victim"
    victim.write_text("keep")
    (own.parent / (arch.name + ".compact_tmp")).symlink_to(victim)
    assert dx.compact_archive(retain_days=7) == 1
    assert victim.read_text() == "keep"
    assert oct(arch.stat().st_mode)[-3:] == "600"         # the rotated archive is rewritten 0600 (SEC-11)
    with gzip.open(arch, "rt") as f:
        assert [json.loads(x)["kind"] for x in f] == ["new"]


def test_a_restrictive_umask_does_not_narrow_the_log_mode(logs, monkeypatch):
    own, root = logs
    monkeypatch.setenv("GB_DATAFLUX_LOG_MODE", "644")
    old = os.umask(0o077)
    try:
        dx.emit({"kind": "k"})
    finally:
        os.umask(old)
    assert oct(own.stat().st_mode)[-3:] == "644"

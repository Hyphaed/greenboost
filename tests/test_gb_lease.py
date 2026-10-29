"""Budgeted lease arbiter: fake card, fake processes (a dict pid -> start time), fake clock. No GPU, no root."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import gb_lease
import gb_snapshot


class World:
    def __init__(self, tmp):
        self.t = 1000.0
        self.procs = {101: 1, 102: 2, 103: 3, 104: 4, 105: 5}      # pid -> start time
        self.card = {"physical_mib": 12000, "free_mib": 9000, "used_by_pid": None}
        self.arb = gb_lease.LeaseArbiter(tmp / "lease", card=lambda: dict(self.card), now=lambda: self.t, starttime=self.procs.get, queue_max=3, aging_per_min=1.0)

    def ask(self, job, vram, pid, **kw):
        kw.setdefault("idempotency_key", f"{job}:{pid}")
        return self.arb.acquire(job, vram, pid=pid, **kw)


@pytest.fixture
def w(tmp_path, monkeypatch):
    monkeypatch.setenv("GB_LEASE_RESERVE_PCT", "5")
    return World(tmp_path)


def test_jobs_that_fit_the_physical_card_share_it(w):
    a = w.ask("tsfm-train", 4096, 101)
    w.card.update(free_mib=4904, used_by_pid={101: 4096})   # a has allocated its budget
    b = w.ask("tsfm-lora", 4500, 102)
    assert a["granted"] and b["granted"]                    # 4096 + 4500 <= 12000 - 600 and <= 4904 really free
    st = w.arb.status()
    assert st["budget_mib"] == 8596 and len(st["holders"]) == 2 and st["reserve_mib"] == 600


def test_a_job_that_does_not_fit_waits_with_a_reason_and_goes_when_budget_is_freed(w):
    a = w.ask("a", 4096, 101)
    w.card.update(free_mib=4904, used_by_pid={101: 4096})
    b = w.ask("b", 4500, 102)
    c = w.ask("c", 4000, 103)
    assert b["granted"] and not c["granted"] and c["position"] == 1 and "unclaimed" in c["reason"]
    w.arb.release(a["lease_id"])
    w.card.update(free_mib=7400, used_by_pid={102: 4500})
    assert w.ask("c", 4000, 103)["granted"]                  # polling by the same idempotency key


def test_the_real_free_memory_counts_not_only_the_budgets(w):
    w.card["free_mib"] = 3000                                # something else (the model server) holds the card
    assert w.ask("a", 2000, 101)["granted"]                  # the first job is always admitted: the engine's pre-flight judges the card
    r = w.ask("b", 2000, 102)
    assert not r["granted"] and "really free" in r["reason"]


def test_grants_not_yet_allocated_are_subtracted_from_free_until_they_settle(w):
    w.card["free_mib"] = 9000
    w.ask("a", 5000, 101)
    r = w.ask("b", 5000, 102)                                # 9000 free but 5000 of it is a's, not allocated yet
    assert not r["granted"]
    w.arb.cancel(r["lease_id"])
    w.card["free_mib"] = 4000
    w.card["used_by_pid"] = {101: 5000}      # a has allocated its 5000
    assert w.ask("b2", 4000, 102)["granted"]


def test_tiering_runs_alone_both_ways(w):
    a = w.ask("small", 1000, 101)
    t = w.ask("big", 20000, 102, tiering=True)
    assert not t["granted"] and "runs alone" in t["reason"]
    s = w.ask("small2", 1000, 103)
    assert not s["granted"] and s["reason"].startswith("behind big")        # strict head-of-line: nothing overtakes the tiering job
    w.arb.release(a["lease_id"])
    assert w.ask("big", 20000, 102, tiering=True)["granted"] and not w.ask("small2", 1000, 103)["granted"]


def test_a_budget_that_can_never_fit_is_rejected_not_queued(w):
    r = w.ask("huge", 20000, 101)
    assert r["rejected"] and "can never fit" in r["reason"] and w.arb.status()["rejects"] == 1


def test_the_queue_is_bounded_and_full_means_reject(w):
    w.ask("hold", 11000, 101)
    for i, pid in enumerate((102, 103, 104)):
        assert not w.ask(f"q{i}", 4000, pid)["rejected"]
    r = w.ask("q3", 4000, 105)
    assert r["rejected"] and "queue is full" in r["reason"]


def test_priority_orders_the_queue_and_aging_lets_a_waiter_catch_up(w):
    w.ask("hold", 11000, 101)
    low = w.ask("low", 4000, 102, priority=0, ttl=10_000)
    w.t += 600                                               # low has waited 10 minutes: effective priority 10
    hi = w.ask("hi", 4000, 103, priority=5)
    st = w.arb.status()
    assert [x["job"] for x in st["waiters"]] == ["low", "hi"]
    w.arb.release(next(h["lease_id"] for h in st["holders"]))
    st = w.arb.status()
    assert st["holders"][0]["job"] == "low" and low["lease_id"] != hi["lease_id"]   # 10 minutes of aging beat priority 5


def test_equal_priority_is_first_come_first_served(w):
    w.ask("hold", 11000, 101)
    w.ask("first", 11000, 102)
    w.t += 1
    w.ask("second", 11000, 103)
    assert [x["job"] for x in w.arb.status()["waiters"]] == ["first", "second"]


def test_idempotency_key_returns_the_same_ticket(w):
    a = w.ask("a", 4096, 101)
    again = w.ask("a", 4096, 101)
    assert again["lease_id"] == a["lease_id"] and len(w.arb.status()["holders"]) == 1


def test_a_dead_holder_is_pruned_and_its_budget_goes_to_the_next(w):
    a = w.ask("a", 11000, 101)
    b = w.ask("b", 4000, 102)
    assert a["granted"] and not b["granted"]
    del w.procs[101]                                         # a crashed without releasing
    assert w.ask("b", 4000, 102)["granted"]
    assert any("pid 101 is gone" in e["text"] for e in w.arb.status()["events"])


def test_a_reused_pid_is_not_the_old_holder(w):
    a = w.ask("a", 11000, 101)
    w.procs[101] = 999                                       # the pid now belongs to another process (different start time)
    assert not w.arb.status()["held"] and a["granted"]


def test_a_live_holder_past_its_ttl_is_flagged_never_evicted(w):
    w.ask("a", 11000, 101, ttl=60)
    w.t += 5000
    st = w.arb.status()
    assert st["held"] and st["holders"][0]["overdue"] is True
    w.arb.renew(st["holders"][0]["lease_id"])
    assert w.arb.status()["holders"][0]["overdue"] is False


def test_a_waiter_that_stops_polling_is_dropped(w):
    w.ask("hold", 11000, 101)
    w.ask("gone", 4000, 102, ttl=30)
    w.t += 31
    assert w.arb.status()["waiters"] == []


def test_a_dead_pid_cannot_even_ask(w):
    r = w.ask("x", 100, 4242)
    assert r["rejected"] and "not running" in r["reason"]


def test_unreadable_card_admits_one_job_at_a_time(w):
    w.card.update(physical_mib=None, free_mib=None)
    assert w.ask("a", 100, 101)["granted"] and not w.ask("b", 100, 102)["granted"]


def test_cancel_a_waiter_removes_it(w):
    w.ask("hold", 11000, 101)
    q = w.ask("q", 4000, 102)
    assert w.arb.cancel(q["lease_id"])["state"] == "cancelled" and w.arb.status()["waiters"] == []


def test_cancel_a_holder_writes_the_control_word_and_waits_for_release_without_signals(w, tmp_path):
    cf = tmp_path / "control"
    a = w.ask("a", 11000, 101, control_file=str(cf))
    r = w.arb.cancel(a["lease_id"], "stop", wait_s=3, sleep=lambda s: setattr(w, "t", w.t + s))
    assert cf.read_text() == "stop" and r["state"] == "cancelling" and not r["released"]
    assert w.arb.status()["holders"][0]["cancel"] == "stop"            # budget stays held until the job answers
    w.arb.release(a["lease_id"])
    assert w.arb.cancel(a["lease_id"])["released"]


def test_cancel_without_a_control_file_refuses_and_touches_nothing(w):
    a = w.ask("a", 4000, 101)
    r = w.arb.cancel(a["lease_id"])
    assert not r["ok"] and w.arb.status()["held"]


def test_the_last_holder_resumes_the_paused_models_and_nothing_is_granted_meanwhile(w):
    a, b = w.ask("a", 3000, 101), w.ask("b", 3000, 102)
    w.arb.note_paused(a["lease_id"], ["qwen"])
    w.ask("c", 11000, 103)
    assert w.arb.begin_release(a["lease_id"])["resume"] == []          # b still runs: the model stays paused
    w.arb.release(a["lease_id"])
    assert w.arb.status()["paused"] == ["qwen"]
    assert w.arb.begin_release(b["lease_id"])["resume"] == ["qwen"]
    assert not w.ask("c", 11000, 103)["granted"]                       # b is still giving the card back
    w.arb.release(b["lease_id"])
    assert w.ask("c", 11000, 103)["granted"] and w.arb.status()["paused"] == []


def test_snapshot_lease_block_is_null_until_the_arbiter_is_used_then_filled(w, tmp_path, monkeypatch):
    p = tmp_path / "lease" / "state.json"
    s = gb_snapshot.build_snapshot(12000, 9000, tmp_path / "run", now=w.t, lease_path=p)
    assert all(v is None for v in s["lease"].values())
    monkeypatch.setattr(gb_lease, "proc_starttime", w.procs.get)
    w.ask("a", 4096, 101)
    w.ask("b", 11000, 102)
    s = gb_snapshot.build_snapshot(12000, 9000, tmp_path / "run", now=w.t + 5, lease_path=p)
    L = s["lease"]
    assert L["held"] is True and L["budget_mib"] == 4096 and L["holders"][0]["job"] == "a" and L["waiters"][0]["job"] == "b"
    assert json.loads(json.dumps(s))["version"] == 1


def test_the_snapshot_reader_never_writes_the_arbiter_files(w, tmp_path):
    w.ask("a", 4096, 101)
    before = {f.name: f.stat().st_mtime_ns for f in (tmp_path / "lease").iterdir()}
    gb_lease.read_block(tmp_path / "lease" / "state.json", w.t, w.procs.get)
    assert before == {f.name: f.stat().st_mtime_ns for f in (tmp_path / "lease").iterdir()}


# --- SEC-02/SEC-03: safe writes, and a damaged state file is "unknown", never "empty" ---------------------------------
def test_safe_atomic_write_uses_a_random_excl_nofollow_temp_and_leaves_one_file(tmp_path, monkeypatch):
    import os
    seen = []
    real_open = os.open
    def spy(path, flags, *a, **kw):
        seen.append((str(path), flags))
        return real_open(path, flags, *a, **kw)
    monkeypatch.setattr(os, "open", spy)
    out = gb_lease.safe_atomic_write(tmp_path / "x.json", "{}", 0o640)
    names = [p for p, f in seen if p.endswith(".tmp")]
    assert len(names) == 1 and names[0].startswith(".x.json.") and str(os.getpid()) not in names[0]
    fl = [f for p, f in seen if p.endswith(".tmp")][0]
    assert fl & os.O_EXCL and fl & os.O_NOFOLLOW and fl & os.O_CREAT
    assert out.read_text() == "{}" and oct(out.stat().st_mode)[-3:] == "640" and [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_safe_atomic_write_refuses_a_symlink_destination_and_does_not_touch_its_target(tmp_path):
    victim = tmp_path / "victim"
    victim.write_text("keep")
    victim.chmod(0o600)
    (tmp_path / "snap.json").symlink_to(victim)
    with pytest.raises(OSError):
        gb_lease.safe_atomic_write(tmp_path / "snap.json", "pwned")
    assert victim.read_text() == "keep" and oct(victim.stat().st_mode)[-3:] == "600"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["snap.json", "victim"]      # no temp left behind


def test_a_planted_symlink_at_the_old_predictable_temp_name_is_harmless(tmp_path):
    import os
    victim = tmp_path / "victim"
    victim.write_text("keep")
    (tmp_path / f".s.json.tmp.{os.getpid()}").symlink_to(victim)
    gb_lease.safe_atomic_write(tmp_path / "s.json", "data")
    assert victim.read_text() == "keep" and (tmp_path / "s.json").read_text() == "data"


def test_safe_atomic_write_refuses_a_directory_or_fifo_destination(tmp_path):
    (tmp_path / "d").mkdir()
    with pytest.raises(OSError):
        gb_lease.safe_atomic_write(tmp_path / "d", "x")


def test_safe_read_text_ignores_symlinks_and_missing(tmp_path):
    (tmp_path / "real").write_text("hi")
    (tmp_path / "ln").symlink_to(tmp_path / "real")
    assert gb_lease.safe_read_text(tmp_path / "real") == "hi"
    assert gb_lease.safe_read_text(tmp_path / "ln") is None and gb_lease.safe_read_text(tmp_path / "nope") is None


def test_the_snapshot_writer_refuses_a_planted_symlink_and_keeps_version_1(tmp_path):
    victim = tmp_path / "etc-shadow"
    victim.write_text("secret")
    victim.chmod(0o600)
    (tmp_path / "pub").mkdir()
    (tmp_path / "pub" / "snapshot.json").symlink_to(victim)
    s = gb_snapshot.build_snapshot(0, 0, tmp_path)
    with pytest.raises(OSError):
        gb_snapshot.write_snapshot(s, tmp_path / "pub" / "snapshot.json")
    assert victim.read_text() == "secret" and oct(victim.stat().st_mode)[-3:] == "600"
    out = gb_snapshot.write_snapshot(s, tmp_path / "pub" / "fresh.json")
    assert json.loads(out.read_text())["version"] == 1 and oct(out.stat().st_mode)[-3:] == "644"


def test_an_unparseable_state_file_is_unknown_not_empty_and_is_reported(w, tmp_path):
    w.ask("a", 4096, 101)
    sp = tmp_path / "lease" / "state.json"
    sp.write_text("{not json")
    r = w.ask("b", 1000, 102)
    assert r["rejected"] and r["granted"] is False and "unreadable" in r["reason"] and r["error"]
    st = w.arb.status()
    assert st["held"] is None and "unreadable" in st["error"]
    assert w.arb.release("anything")["ok"] is False
    assert sp.read_text() == "{not json"                                   # evidence untouched: nothing was rewritten
    blk = gb_lease.read_block(sp, w.t, w.procs.get)
    assert blk["held"] is None and "unreadable" in blk["error"]
    assert set(blk) >= {"held", "holders", "budget_mib", "waiters", "last_wait_s"}   # version 1 keys still present


def test_a_missing_state_file_is_still_just_empty(w, tmp_path):
    assert w.ask("a", 4096, 101)["granted"]


def test_a_json_array_state_file_is_also_unknown(w, tmp_path):
    w.ask("a", 4096, 101)
    (tmp_path / "lease" / "state.json").write_text("[]")
    assert w.ask("b", 1000, 102)["rejected"]

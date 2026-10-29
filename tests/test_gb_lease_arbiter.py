"""Arbiter round 2: client table, classes, aging, anti-starvation, fencing, refusal codes, per-client budgets.
Fake card, fake processes, fake clock. No GPU, no root, no model server."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import gb_lease


class World:
    def __init__(self, tmp, **kw):
        self.t = 1000.0
        self.procs = {p: p for p in range(100, 140)}
        self.card = {"physical_mib": 12000, "free_mib": 11000, "used_by_pid": None}
        self.arb = gb_lease.LeaseArbiter(tmp / "lease", card=lambda: dict(self.card), now=lambda: self.t, starttime=self.procs.get,
                                         queue_max=kw.pop("queue_max", 40), aging_per_min=1.0, starve_s=kw.pop("starve_s", 1800.0), **kw)
        self.n = 0

    def ask(self, client, vram=1000, **kw):
        self.n += 1
        pid = kw.pop("pid", 100 + self.n)
        kw.setdefault("idempotency_key", f"{client}:{self.n}")
        return self.arb.acquire(client, vram, pid=pid, client=client, **kw), kw["idempotency_key"], pid

    def poll(self, client, key, pid, vram=1000):
        return self.arb.acquire(client, vram, pid=pid, client=client, idempotency_key=key)


@pytest.fixture
def w(tmp_path, monkeypatch):
    monkeypatch.setenv("GB_LEASE_RESERVE_PCT", "5")
    return World(tmp_path)


def test_every_answer_carries_a_documented_code(w):
    seen = set()
    r, *_ = w.ask("invest.engine.job"); seen.add(r["code"])
    r, *_ = w.ask("invest.engine.reader", 20000); seen.add(r["code"])                 # NO_FIT
    r, *_ = w.ask("optimal-claude.hook", wait=False, vram=11000); seen.add(r["code"])  # BUSY_HOLDER
    r = w.arb.acquire("x", -1, pid=101); seen.add(r["code"])                           # BAD_REQUEST
    assert {"GRANTED", "NO_FIT", "BUSY_HOLDER", "BAD_REQUEST"} <= seen and seen <= set(gb_lease.CODES)


def test_class_comes_from_the_table_not_from_the_caller(w):
    a, *_ = w.ask("invest.engine.job", priority=999)           # a claimed priority is ignored
    b, kb, pb = w.ask("optimal-claude.hook", 11000)
    c, kc, pc = w.ask("owner.interactive", 11000)
    w.arb.release(a["lease_id"])
    assert w.poll("owner.interactive", kc, pc, 11000)["granted"] and not w.poll("optimal-claude.hook", kb, pb, 11000)["granted"]


def test_p4_is_refused_with_busy_holder_and_a_retry_hint_while_p1_holds(w):
    p1, *_ = w.ask("invest.engine.pass", 11000, max_hold_s=1200)
    p4, *_ = w.ask("optimal-claude.hook", 500, wait=False)
    assert p1["granted"] and p4["rejected"] and p4["code"] == "BUSY_HOLDER"
    assert p4["retry_after_s"] == 1200 and p4["holder"]["client"] == "invest.engine.pass"
    assert w.arb.status()["waiters"] == []                      # a refused no-wait caller leaves nothing behind


def test_no_two_p1_holders_at_once_even_when_the_budgets_fit(w):
    a, *_ = w.ask("invest.engine.pass", 1000)
    b, kb, pb = w.ask("invest.engine.pass", 1000)
    assert a["granted"] and not b["granted"] and "one holder at a time" in b["reason"]
    w.arb.release(a["lease_id"])
    assert w.poll("invest.engine.pass", kb, pb)["granted"]


def test_aging_serves_a_long_waiter_before_a_fresh_one_of_the_same_class_but_never_before_a_waiting_p1(w):
    hold, *_ = w.ask("invest.engine.job", 11000)
    old, ko, po = w.ask("optimal-claude.hook", 11000, ttl=10 ** 6)
    w.t += 600
    fresh, kf, pf = w.ask("optimal-claude.hook", 11000, ttl=10 ** 6)
    p1, k1, p1pid = w.ask("invest.engine.pass", 11000, ttl=10 ** 6)
    w.arb.release(hold["lease_id"])
    first = w.poll("invest.engine.pass", k1, p1pid, 11000)
    assert first["granted"] and not w.poll("optimal-claude.hook", ko, po, 11000)["granted"]      # the waiting P1 beats a P4 that waited 10 minutes
    w.arb.release(first["lease_id"])
    assert w.poll("optimal-claude.hook", ko, po, 11000)["granted"]                                # then the long waiter beats the fresh one
    assert not w.poll("optimal-claude.hook", kf, pf, 11000)["granted"]


def test_aging_alone_never_lifts_a_class_over_the_next_one(w):
    hold, *_ = w.ask("invest.engine.job", 11000)
    p4, k4, pp4 = w.ask("optimal-claude.hook", 11000)
    w.t += 1700                                                # under starve_s: 28 minutes of aging
    p3, k3, pp3 = w.ask("invest.engine.reader", 11000)
    w.arb.release(hold["lease_id"])
    assert w.poll("invest.engine.reader", k3, pp3, 11000)["granted"] and not w.poll("optimal-claude.hook", k4, pp4, 11000)["granted"]


def test_starvation_is_bounded_a_p4_is_promoted_after_starve_s(w):
    """A stream of fresh P3 requests would keep a P4 out for ever on aging alone; promotion lets it through."""
    holder, *_ = w.ask("invest.engine.job", 11000)
    p4, k4, pp4 = w.ask("optimal-claude.hook", 11000)
    served = None
    for i in range(80):                                        # a fresh P3 every minute, each waits behind the same holder
        w.t += 60
        r, k3, pp3 = w.ask("invest.engine.reader", 11000)
        w.arb.release(holder["lease_id"])
        got = w.poll("optimal-claude.hook", k4, pp4, 11000)
        if got["granted"]:
            served = (i, w.t - 1000)
            break
        # the fresh P3 won: it holds now, then finishes and the next P3 comes
        holder = w.poll("invest.engine.reader", k3, pp3, 11000)
        assert holder["granted"], holder
    assert served is not None and served[1] >= 1800            # not before it was promoted
    assert served[1] <= 1800 + 120


def test_a_waiter_that_was_never_promoted_into_p0(w):
    hold, *_ = w.ask("owner.interactive", 11000)
    p4, k4, pp4 = w.ask("optimal-claude.hook", 11000, ttl=10 ** 9)
    owner2, ko, po = w.ask("owner.interactive", 11000, ttl=10 ** 9)
    w.t += 10 * 1800
    w.arb.release(hold["lease_id"])
    assert w.poll("owner.interactive", ko, po, 11000)["granted"] and not w.poll("optimal-claude.hook", k4, pp4, 11000)["granted"]


def test_fencing_tokens_are_monotonic_and_a_stale_one_is_refused(w):
    a, *_ = w.ask("invest.engine.job", 1000)
    w.arb.release(a["lease_id"])
    b, *_ = w.ask("invest.engine.job", 1000)
    assert b["token"] > a["token"]
    assert w.arb.check(b["lease_id"], b["token"])["ok"]
    assert w.arb.check(b["lease_id"], a["token"])["code"] == "STALE_TOKEN"
    assert w.arb.renew(b["lease_id"], token=a["token"])["code"] == "STALE_TOKEN"
    assert w.arb.release(b["lease_id"], token=a["token"])["code"] == "STALE_TOKEN"      # the wrong token releases nothing
    assert w.arb.status()["holders"][0]["token"] == b["token"]


def test_a_crashed_holder_is_reclaimed_by_pid_and_start_time_and_its_token_is_dead(w):
    a, ka, pa = w.ask("invest.engine.job", 11000, pid=120)
    b, kb, pb = w.ask("invest.engine.reader", 11000)
    assert not b["granted"]
    del w.procs[120]                                            # the holder dies
    got = w.poll("invest.engine.reader", kb, pb, 11000)
    assert got["granted"] and got["token"] > a["token"]
    assert w.arb.check(a["lease_id"], a["token"])["code"] == "STALE_TOKEN"
    # the pid number comes back as a different process (new start time): still not the old holder
    w.procs[120] = 99999
    assert w.arb.check(a["lease_id"], a["token"])["ok"] is False


def test_pid_reuse_is_not_the_old_holder(w):
    a, ka, pa = w.ask("invest.engine.job", 11000, pid=121)
    w.procs[121] = 424242
    st = w.arb.status()
    assert st["holders"] == [] and any("reused" in e["text"] for e in st["events"])


def test_two_clients_with_budgets(tmp_path, monkeypatch):
    monkeypatch.setenv("GB_LEASE_RESERVE_PCT", "5")
    w = World(tmp_path, client_budgets={"invest.engine.job": 6000, "optimal-claude.rag": 3000})
    big, *_ = w.ask("invest.engine.job", 7000)
    assert big["rejected"] and big["code"] == "CLIENT_BUDGET"
    j1, *_ = w.ask("invest.engine.job", 5000)
    j2, k2, p2 = w.ask("invest.engine.job", 2000)               # 5000 + 2000 > 6000: waits although the card has room
    assert j1["granted"] and not j2["granted"] and "budget is 6000" in j2["reason"]
    rag, kr, pr = w.ask("optimal-claude.rag", 3000)
    assert not rag["granted"] and "behind" in rag["reason"]       # head of line is kept: the rag waits behind the blocked job request
    w.arb.release(j1["lease_id"])
    assert w.poll("invest.engine.job", k2, p2, 2000)["granted"] and w.poll("optimal-claude.rag", kr, pr, 3000)["granted"]
    assert w.arb.status()["budget_mib"] == 5000


def test_unknown_client_is_p9_and_logged_or_refused_when_strict(w, monkeypatch):
    r, *_ = w.ask("mystery.tool", 500)
    assert r["granted"] and any("unknown client" in e["text"] for e in w.arb.status()["events"])
    monkeypatch.setenv("GB_LEASE_STRICT_CLIENTS", "1")
    r, *_ = w.ask("mystery.tool2", 500)
    assert r["code"] == "UNKNOWN_CLIENT"


def test_blocked_flag_refuses_everyone_but_the_owner(w):
    w.arb.dir.mkdir(parents=True, exist_ok=True)
    (w.arb.dir / "block").write_text("owner is gaming")
    r, *_ = w.ask("invest.engine.pass", 500)
    o, *_ = w.ask("owner.interactive", 500)
    assert r["code"] == "BLOCKED" and r["retry_after_s"] and o["granted"]


def test_legacy_callers_without_a_client_behave_as_before(w):
    a = w.arb.acquire("tsfm-train", 4000, pid=101, priority=0)
    assert a["granted"] and a["code"] == "GRANTED" and a["token"] == 1

"""SP41-42: run_under_greenboost holds the card through the arbiter. Fake card, fake clock; the command is `true`, no model, no GPU."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import gb_actuation as ga
import gb_lease
import gb_lease_client as glc


@pytest.fixture
def world(tmp_path, monkeypatch):
    for k in ("GB_LEASE_OFF", "GB_LEASE_HELD"):
        monkeypatch.delenv(k, raising=False)
    t = [1000.0]
    procs = {p: p for p in range(100, 400)}
    procs[os.getpid()] = 7
    arb = gb_lease.LeaseArbiter(tmp_path / "lease", card=lambda: {"physical_mib": 12000, "free_mib": 11000, "used_by_pid": None},
                                now=lambda: t[0], starttime=procs.get, starve_s=1800.0)
    monkeypatch.setattr(glc, "_arbiter", lambda: arb)
    monkeypatch.setattr(glc.time, "sleep", lambda s: t.__setitem__(0, t[0] + s))
    monkeypatch.setattr(glc.time, "monotonic", lambda: t[0])
    monkeypatch.setattr(ga, "actuation_gate", lambda confirm: {"allowed": True, "reason": "test"})
    import gb_cluster
    monkeypatch.setattr(gb_cluster, "shim_env", lambda **k: {})
    monkeypatch.setattr(ga, "_emit", lambda *a, **k: None)
    return arb


def test_run_holds_a_lease_and_releases_it(world):
    r = ga.run_under_greenboost(["true"], confirm=True)
    assert r.get("exit_code") == 0 and "error" not in r
    again = world.acquire(job="greenboost.cli.prompt", vram_mib=0, client="greenboost.cli.prompt", idempotency_key="after", wait=False)
    assert again["granted"]                       # the run's lease was released


def test_child_sees_the_held_marker(world):
    r = ga.run_under_greenboost(["sh", "-c", "echo $GB_LEASE_HELD"], confirm=True)
    assert r["stdout_tail"].strip()               # GB_LEASE_HELD is set inside the run


def test_busy_card_is_a_plain_refusal_and_nothing_runs(world, tmp_path):
    other = world.acquire(job="greenboost.cli.prompt", vram_mib=0, client="greenboost.cli.prompt", idempotency_key="k", wait=False)
    assert other["granted"]
    marker = tmp_path / "ran"
    r = ga.run_under_greenboost(["touch", str(marker)], confirm=True)
    assert "GPU lease refused" in r["error"] and "nothing was loaded" in r["error"]
    assert not marker.exists()


def test_escape_hatch_runs_despite_a_busy_card(world, monkeypatch, tmp_path):
    world.acquire(job="greenboost.cli.prompt", vram_mib=0, client="greenboost.cli.prompt", idempotency_key="k", wait=False)
    monkeypatch.setenv("GB_LEASE_OFF", "1")
    assert ga.run_under_greenboost(["true"], confirm=True)["exit_code"] == 0


def test_dry_run_takes_no_lease(world, monkeypatch):
    monkeypatch.setattr(ga, "actuation_gate", lambda confirm: {"allowed": False, "reason": "dry"})
    world.acquire(job="greenboost.cli.prompt", vram_mib=0, client="greenboost.cli.prompt", idempotency_key="k", wait=False)
    assert "dry_run" in ga.run_under_greenboost(["true"])

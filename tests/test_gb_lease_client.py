"""SP41-39: GreenBoost-side and sibling-repo clients go through the arbiter. Fake card, fake clock, fake processes; no GPU, no model load."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "greenboost-cli"))
import gb_lease
import gb_lease_client as glc


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


@pytest.fixture
def world(tmp_path, monkeypatch):
    for k in ("GB_LEASE_OFF", "GB_LEASE_HELD"):
        monkeypatch.delenv(k, raising=False)
    clk = Clock()
    procs = {p: p for p in range(100, 400)}
    procs[os.getpid()] = 7
    arb = gb_lease.LeaseArbiter(tmp_path / "lease", card=lambda: {"physical_mib": 12000, "free_mib": 11000, "used_by_pid": None}, now=clk, starttime=procs.get, starve_s=1800.0)
    monkeypatch.setattr(glc, "_arbiter", lambda: arb)
    return arb, clk


def test_every_client_name_is_in_the_table_with_the_documented_class():
    want = {"greenboost.synapse.serve": "P0", "greenboost.cli.prompt": "P0", "greenboost.rag.query": "P0", "greenboost.rag.index": "P2",
            "greenboost.rotator": "P2", "greenboost.bench": "P4", "optimal-claude-history": "P4"}
    for name, cls in want.items():
        assert gb_lease.CLIENTS[name][0] == cls, name
    assert gb_lease.CLIENTS["optimal-claude-history"][2] == "batch" and gb_lease.CLIENTS["greenboost.rag.query"][2] == "interactive"


def test_lease_is_held_in_the_body_named_and_released_on_exit_and_on_error(world):
    arb, _ = world
    with glc.lease("greenboost.rag.index", "embed 3 chunks", wait_s=5) as h:
        s = arb.status()["holders"][0]
        assert s["client"] == "greenboost.rag.index" and s["class"] == "P2" and s["token"] == h["token"] and os.environ[glc.HELD_ENV] == h["lease_id"]
    assert arb.status()["holders"] == [] and glc.HELD_ENV not in os.environ
    with pytest.raises(ValueError):
        with glc.lease("greenboost.rag.index"):
            raise ValueError("boom")
    assert arb.status()["holders"] == [] and glc.HELD_ENV not in os.environ


def test_background_client_is_refused_with_a_plain_message_and_never_queued(world):
    arb, _ = world
    arb.acquire("big", 11000, pid=101, client="invest.engine.job", tiering=True, max_hold_s=900)
    with pytest.raises(glc.LeaseRefused) as e:
        with glc.lease("greenboost.bench", "bench", wait_s=0):
            raise AssertionError("body must not run")
    assert e.value.code == "BUSY_HOLDER" and "invest.engine.job" in str(e.value) and "nothing was loaded" in str(e.value) and e.value.retry_after_s == 900
    assert arb.status()["waiters"] == []


def test_waiting_client_gets_the_card_when_the_holder_leaves_and_gives_up_after_wait_s(world):
    arb, clk = world
    h = arb.acquire("big", 11000, pid=101, client="invest.engine.job", tiering=True)
    with pytest.raises(glc.LeaseRefused) as e:
        with glc.lease("greenboost.synapse.serve", wait_s=3, sleep=clk.sleep, now=clk):
            pass
    assert e.value.code == "BUSY_HOLDER" and arb.status()["waiters"] == []          # it left the queue
    calls = {"n": 0}

    def sleep(s):                                                                    # the holder leaves while the client waits
        clk.t += s
        calls["n"] += 1
        if calls["n"] == 2:
            arb.release(h["lease_id"])
    with glc.lease("greenboost.synapse.serve", wait_s=30, sleep=sleep, now=clk) as g:
        assert g["lease_id"] and calls["n"] == 2


def test_nested_call_and_child_process_do_not_ask_again(world, monkeypatch):
    arb, _ = world
    with glc.lease("greenboost.cli.prompt") as outer:
        with glc.lease("greenboost.synapse.serve") as inner:                           # P0 is solo: asking again would wait on ourselves
            assert inner["skipped"] and len(arb.status()["holders"]) == 1
        assert outer["lease_id"] == os.environ[glc.HELD_ENV]


def test_off_switch_and_missing_arbiter_never_block_a_call(world, monkeypatch, capsys):
    monkeypatch.setenv("GB_LEASE_OFF", "1")
    with glc.lease("greenboost.bench") as h:
        assert h["skipped"] == "GB_LEASE_OFF"
    monkeypatch.delenv("GB_LEASE_OFF")
    monkeypatch.setattr(glc, "_arbiter", lambda: (_ for _ in ()).throw(ImportError("no gb_lease")))
    with glc.lease("greenboost.bench") as h:
        assert h["skipped"] == "arbiter unavailable"
    assert "runs unleased" in capsys.readouterr().err


def test_block_flag_and_damaged_state_messages(world):
    arb, _ = world
    (arb.dir).mkdir(parents=True, exist_ok=True)
    (arb.dir / "block").write_text("")
    with pytest.raises(glc.LeaseRefused) as e:
        with glc.lease("greenboost.synapse.serve"):
            pass
    assert e.value.code == "BLOCKED" and "block" in str(e.value)


def test_synapse_serve_gate_maps_triggers_to_clients(world):
    gb_synapse = pytest.importorskip("gb_synapse")
    arb, _ = world
    with gb_synapse._serve_lease("m", "mcp_serve"):
        assert arb.status()["holders"][0]["client"] == "greenboost.synapse.serve"
    with gb_synapse._serve_lease("m", "rotator_job"):
        assert arb.status()["holders"][0]["client"] == "greenboost.rotator"
    with gb_synapse._serve_lease("m", "resume") as h:
        assert h is None and arb.status()["holders"] == []                             # the give-back after a lease is never leased
    arb.acquire("big", 11000, pid=101, client="invest.engine.job", tiering=True)
    with pytest.raises(glc.LeaseRefused):
        with gb_synapse._serve_lease("m", "bench"):                                    # bench skips, it does not wait
            pass


def test_cli_prompt_returns_the_refusal_as_data_and_runs_nothing(world, monkeypatch):
    mod = pytest.importorskip("gb_synapse_mcp")
    arb, _ = world
    arb.acquire("big", 11000, pid=101, client="owner.interactive", tiering=True)
    monkeypatch.setattr(mod, "_gb_bin", lambda: "/nonexistent/gb")
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran gb")))
    monkeypatch.setattr(glc.time, "monotonic", Clock())
    monkeypatch.setattr(glc.time, "sleep", lambda s: glc.time.monotonic.sleep(s))
    out = mod.cli_prompt("hi")
    assert out["code"] == "BUSY_HOLDER" and "nothing was loaded" in out["error"]


def test_rag_embedding_goes_through_the_arbiter_and_a_refusal_is_a_plain_runtime_error(world, monkeypatch):
    rag = pytest.importorskip("greenboost_cli.rag.contextual_rag")
    arb, _ = world
    seen = {}
    monkeypatch.setattr(rag, "_ollama_embed", lambda texts, **k: seen.update(h=arb.status()["holders"][0]) or [1])
    with rag._arbiter_lease("greenboost.rag.index", "embed 1 chunks", wait_s=5):
        rag._ollama_embed(["x"])
    assert seen["h"]["client"] == "greenboost.rag.index" and seen["h"]["class"] == "P2"
    arb.acquire("big", 11000, pid=101, client="owner.interactive", tiering=True)
    with pytest.raises(RuntimeError, match="nothing was loaded"):
        with rag._arbiter_lease("greenboost.rag.query", "q", wait_s=0):
            pass

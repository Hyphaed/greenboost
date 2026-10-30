# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""gb_phase: the shared GPU phase lease (one heavy phase on the card at a time), compatible with the studio's lock file.

Why it exists: Qwen vision (~10.6 GB), a FLUX batch (~9.9 GiB peak), browser raster and NVENC share one 12 GB card; two heavy
phases at once die, they do not slow down. The studio's TypeScript hub (packages/art/src/lock.ts) already writes this lock format.
"""
import json
import os
import time

import pytest

import gb_phase as gp


@pytest.fixture
def lock(tmp_path, monkeypatch):
    p = tmp_path / "gpu.lock"
    monkeypatch.setenv("SVG_STUDIO_GPU_LOCK", str(p))
    monkeypatch.setattr(gp, "_emit", lambda ev: EVENTS.append(ev))
    monkeypatch.setattr(gp, "_gpu_processes", lambda: [{"pid": 4242, "name": "python3", "mib": 9900}])
    EVENTS.clear()
    return p


EVENTS: list = []


def _vram(used, total=12000):
    return lambda: (used, total)


def _studio_lock(path, pid=None, age_s=0.0, token="studiotoken"):
    t = time.time() - age_s
    path.write_text(json.dumps({"class": "image", "pid": pid or os.getpid(), "owner": "art-hub", "job": "heads",
                                "since": gp._iso(t), "heartbeat": gp._iso(t), "token": token}))


def test_the_lock_has_the_studios_format_and_is_released_by_its_token_only(lock):
    lease = gp.acquire_phase("flux", 0, vram_fn=_vram(2000), owner="ai-forge", cls="image")
    info = json.loads(lock.read_text())
    assert set(info) >= {"class", "pid", "owner", "job", "since", "heartbeat", "token"}
    assert info["owner"] == "ai-forge" and info["pid"] == os.getpid()
    other = json.loads(lock.read_text()); other["token"] = "someone-else"; lock.write_text(json.dumps(other))
    assert lease.release()["released"] is False and lock.exists()          # not ours any more: left alone
    lock.unlink()


def test_a_second_phase_is_refused_with_the_holder_named_and_nothing_changed(lock):
    first = gp.acquire_phase("qwen", 0, vram_fn=_vram(2000), owner="studio", cls="llm")
    with pytest.raises(gp.PhaseBusy) as ei:
        gp.acquire_phase("flux", 0, vram_fn=_vram(2000), owner="ai-forge")
    msg = str(ei.value)
    assert "studio" in msg and "Nothing was started and nothing was changed" in msg and "python3 (pid 4242" in msg
    assert ei.value.holder["owner"] == "studio" and first.release()["released"]


def test_a_dead_holder_or_a_silent_one_is_stale_and_replaced(lock):
    _studio_lock(lock, pid=2 ** 22 + 12345)                                  # a pid that is not running
    gp.acquire_phase("a", 0, vram_fn=_vram(2000)).release()
    _studio_lock(lock, age_s=300)                                            # alive pid, heartbeat 5 minutes old
    gp.acquire_phase("b", 0, vram_fn=_vram(2000)).release()
    _studio_lock(lock, age_s=5)                                              # alive and fresh: NOT stale
    with pytest.raises(gp.PhaseBusy):
        gp.acquire_phase("c", 0, vram_fn=_vram(2000))


def test_waiting_succeeds_when_the_holder_lets_go(lock):
    _studio_lock(lock, age_s=1)
    clock = {"t": 1000.0, "polls": 0}

    def sleep(s):
        clock["t"] += s; clock["polls"] += 1
        if clock["polls"] == 3:
            lock.unlink()

    lease = gp.acquire_phase("flux", 0, wait_s=30, poll_s=2, vram_fn=_vram(2000), sleep=sleep, now=lambda: clock["t"])
    assert clock["polls"] == 3 and lease.release()["released"]


def test_short_vram_pauses_the_synapse_models_then_proceeds(lock):
    state = {"used": 10600}
    paused = []

    def pause(m):
        paused.append(m); state["used"] = 2000
        return {"ok": True}

    lease = gp.acquire_phase("flux", 75, vram_fn=lambda: (state["used"], 12000), served_fn=lambda: ["qwen38"], pause_fn=pause)
    assert paused == ["qwen38"] and lease.paused == ["qwen38"]
    assert EVENTS[0]["status"] == "start" and EVENTS[0]["paused_models"] == ["qwen38"]
    lease.release()
    assert EVENTS[-1]["status"] == "end" and "seconds" in EVENTS[-1] and EVENTS[-1]["peak_vram_mib"] >= 2000


def test_an_llm_phase_does_not_pause_the_models_it_runs_on(lock):
    paused = []
    with pytest.raises(gp.PhaseBusy):
        gp.acquire_phase("qwen-critic", 90, cls="vlm", vram_fn=_vram(11000), served_fn=lambda: ["qwen38"],
                         pause_fn=lambda m: paused.append(m) or {"ok": True})
    assert paused == []


def test_keep_models_are_never_paused(lock):
    paused = []
    with pytest.raises(gp.PhaseBusy):
        gp.acquire_phase("flux", 90, vram_fn=_vram(11000), served_fn=lambda: ["qwen38", "embed"], keep_models=("qwen38",),
                         pause_fn=lambda m: paused.append(m) or {"ok": True})
    assert paused == ["embed"]


def test_a_refused_phase_leaves_no_lock_behind(lock):
    with pytest.raises(gp.PhaseBusy) as ei:
        gp.acquire_phase("flux", 90, vram_fn=_vram(11500), served_fn=lambda: [])
    assert not lock.exists() and "needs 90%" in str(ei.value) and "4%" in str(ei.value)


def test_unreadable_vram_does_not_block(lock):
    gp.acquire_phase("flux", 90, vram_fn=lambda: None).release()


def test_the_context_manager_releases_on_error(lock):
    with pytest.raises(ValueError):
        with gp.gpu_phase("x", 0, vram_fn=_vram(2000)):
            assert lock.exists()
            raise ValueError("boom")
    assert not lock.exists() and EVENTS[-1]["status"] == "end"


def test_status_is_read_only_and_says_whether_a_phase_would_start(lock):
    s = gp.phase_status(75, vram_fn=_vram(2000))
    assert s["holder"] is None and s["would_start"] is True and s["free_pct"] == pytest.approx(83.3, abs=0.1)
    _studio_lock(lock, age_s=1)
    s = gp.phase_status(75, vram_fn=_vram(2000))
    assert s["holder"]["owner"] == "art-hub" and s["would_start"] is False and not s["holder_stale"]
    assert json.loads(lock.read_text())["token"] == "studiotoken"           # status touched nothing


def test_the_lock_path_follows_the_studios_env_and_default(monkeypatch, tmp_path):
    monkeypatch.delenv("SVG_STUDIO_GPU_LOCK", raising=False)
    assert gp.lock_path().name == "svg-studio-gpu.lock"
    monkeypatch.setenv("SVG_STUDIO_GPU_LOCK", str(tmp_path / "x.lock"))
    assert gp.lock_path() == tmp_path / "x.lock"


def test_iso_roundtrip_matches_the_studios_timestamps():
    t = 1793000000.123
    assert gp._parse_iso(gp._iso(t)) == pytest.approx(t, abs=0.002)


# ---------------------------------------------------------------- actuation + MCP surface


def test_acquire_is_a_dry_run_without_the_double_gate(lock, monkeypatch):
    import gb_actuation as act
    monkeypatch.setattr(gp, "phase_status", lambda need_pct=0.0, **k: {"would_start": True, "holder": None, "holder_stale": False})
    r = act.gpu_phase_acquire("flux", 75, confirm=True)                       # confirm alone never authorizes
    assert r["gate"]["allowed"] is False and r["dry_run"] == "would take the lease" and "applied" not in r


def test_acquire_and_release_through_the_gate(lock, monkeypatch):
    import gb_actuation as act
    monkeypatch.setattr(act, "actuation_gate", lambda confirm: {"allowed": bool(confirm), "reason": "test"})
    monkeypatch.setattr(gp, "phase_status", lambda need_pct=0.0, **k: {"would_start": True, "holder": None, "holder_stale": False})
    real = gp.acquire_phase
    monkeypatch.setattr(gp, "acquire_phase", lambda *a, **k: real(*a, vram_fn=_vram(2000), **k))
    r = act.gpu_phase_acquire("flux", 0, ttl_s=60, confirm=True)
    assert r["applied"] and r["token"] in act._PHASE_LEASES and lock.exists()
    assert act.gpu_phase_release(r["token"], confirm=False)["dry_run"] == "would release the lease"
    done = act.gpu_phase_release(r["token"], confirm=True)
    assert done["applied"] and not lock.exists() and r["token"] not in act._PHASE_LEASES


def test_a_busy_card_is_reported_not_raised_through_mcp(lock, monkeypatch):
    import gb_actuation as act
    monkeypatch.setattr(act, "actuation_gate", lambda confirm: {"allowed": True, "reason": "test"})
    monkeypatch.setattr(gp, "phase_status", lambda need_pct=0.0, **k: {"would_start": False, "holder": {"owner": "x"}, "holder_stale": False})
    _studio_lock(lock, age_s=1)
    real = gp.acquire_phase
    monkeypatch.setattr(gp, "acquire_phase", lambda *a, **k: real(*a, vram_fn=_vram(2000), **k))
    r = act.gpu_phase_acquire("flux", 0, confirm=True)
    assert "refused" in r and "Nothing was started" in r["refused"]


def test_the_mcp_tools_exist_and_the_facade_delegates():
    import gb_mcp
    import gb_api
    for name in ("gpu_phase_status", "gpu_phase_acquire", "gpu_phase_release"):
        assert hasattr(gb_mcp, name)
    assert callable(gb_api.gpu_phase) and callable(gb_api.cpu_affinity)

"""`clear memory-pool` must actually clear the pool.

Owner report, 2026-08-18: the command was run repeatedly to get the card back
and kept reporting success while 10.4 GiB stayed held by an idle gb-synapse
server at 0% GPU utilization. Two independent causes, both covered here:

  1. `_PROTECTED_PREFIXES` carried a blanket "greenboost" entry, so every
     process whose comm starts with that — `greenboost-cli` included — was
     unreclaimable. The command could never release the GPU memory of the very
     CLI the owner runs inference through.
  2. The default scope was "residue", which spares any live-tracked server. On
     a box whose only GPU holder IS that server, "residue" targets nothing.

The owner's requirement is explicit: clear T1 and T2 and any GPU compute
related to greenboost-cli, greenboost itself, or a session started through
~/Dev/ai-forge.
"""
from __future__ import annotations

import pytest

import gb_reclaim as r


# ── protection list ──────────────────────────────────────────────────────────

def test_greenboost_cli_is_reclaimable():
    """The regression that made the command useless for its own CLI."""
    assert r._gb_proc_is_protected("greenboost-cli") is False


@pytest.mark.parametrize("comm", ["greenboost-netd", "greenboost-net"])
def test_fabric_daemon_stays_protected(comm):
    """Killing netd drops the cluster link to every feeder."""
    assert r._gb_proc_is_protected(comm) is True


@pytest.mark.parametrize("comm", ["gnome-shell", "Xwayland", "systemd-logind",
                                  "pipewire", "qemu-kvm", "nvidia-persiste"])
def test_desktop_and_system_stay_protected(comm):
    """A memory reclaim must never log the owner out or kill a VM."""
    assert r._gb_proc_is_protected(comm) is True


# ── inference detection ──────────────────────────────────────────────────────

@pytest.mark.parametrize("comm", ["llama-server", "ollama", "greenboost-cli",
                                  "rpc-server", "vllm"])
def test_known_engines_detected_by_comm(comm, monkeypatch):
    monkeypatch.setattr(r, "_proc_cmdline", lambda pid: "")
    assert r.is_inference_process(1234, comm) is True


@pytest.mark.parametrize("cmdline", [
    "/usr/bin/python3 /home/u/dev/ai-forge/run_character_pipeline.py --seed 3",
    "python3 /home/u/dev/ai-forge/comfyui/main.py --listen",
    "/bin/bash /home/u/dev/jocs/curse_of_the_seas/art_wizard.sh",
    "python3 -m gb_diffusion_server --port 8801",
    "python3 /opt/greenboost/gb_longlive_server.py",
])
def test_ai_forge_workers_detected_by_cmdline(cmdline, monkeypatch):
    """Every ai-forge pipeline runs as a bare `python3`, so comm cannot
    identify it — this is exactly the art_wizard case the owner named."""
    monkeypatch.setattr(r, "_proc_cmdline", lambda pid: cmdline.lower())
    assert r.is_inference_process(1234, "python3") is True


@pytest.mark.parametrize("cmdline", [
    "python3 /home/u/scripts/backup_photos.py",
    "python3 -m http.server 8000",
    "/usr/bin/python3 /usr/share/unrelated/tool.py --gpu",
])
def test_unrelated_python_is_not_inference(cmdline, monkeypatch):
    """A false positive here kills the owner's own job. 'python' alone is
    deliberately NOT a hint."""
    monkeypatch.setattr(r, "_proc_cmdline", lambda pid: cmdline.lower())
    assert r.is_inference_process(1234, "python3") is False


# ── scope semantics ──────────────────────────────────────────────────────────

@pytest.fixture
def classes(monkeypatch):
    """One live gb-synapse server, one orphan, one unrelated GPU job."""
    fake = {
        "live": [{"pid": 100, "comm": "llama-server", "gpu_mb": 9796, "gb_dev": True}],
        "ambiguous": [{"pid": 200, "comm": "python3", "gpu_mb": 2048, "gb_dev": False}],
        "residue": [{"pid": 300, "comm": "ollama", "gpu_mb": 512, "gb_dev": False}],
    }
    monkeypatch.setattr(r, "classify_processes", lambda *a, **kw: fake)
    cmdlines = {200: "python3 /home/u/dev/ai-forge/run_character_pipeline.py"}
    monkeypatch.setattr(r, "_proc_cmdline", lambda pid: cmdlines.get(pid, ""))
    return fake


def test_inference_scope_includes_the_live_server(classes):
    """The whole point: a live server is in scope, because sparing it is why
    the command kept reporting success while the card stayed full."""
    pids = {t["pid"] for t in r.plan_reclaim(scope="inference")["targets"]}
    assert 100 in pids, "live gb-synapse server must be reclaimed"
    assert 200 in pids, "ai-forge worker must be reclaimed"
    assert 300 in pids, "orphaned ollama must be reclaimed"


def test_residue_scope_still_spares_the_live_server(classes):
    """The conservative behaviour stays reachable via --residue."""
    pids = {t["pid"] for t in r.plan_reclaim(scope="residue")["targets"]}
    assert pids == {300}


def test_inference_scope_skips_non_inference_gpu_jobs(classes, monkeypatch):
    """An unrelated CUDA job is left alone at this scope; --all is for that."""
    monkeypatch.setattr(r, "_proc_cmdline", lambda pid: "python3 /home/u/train.py")
    pids = {t["pid"] for t in r.plan_reclaim(scope="inference")["targets"]}
    assert 200 not in pids
    assert {100, 300} <= pids, "known engines are still matched by comm"


def test_all_scope_is_unchanged(classes):
    pids = {t["pid"] for t in r.plan_reclaim(scope="all")["targets"]}
    assert pids == {100, 200, 300}


def test_inference_is_a_known_scope():
    assert "inference" in r._SCOPES
    with pytest.raises(ValueError):
        r.plan_reclaim(scope="nonsense")


# ── never kill the caller ────────────────────────────────────────────────────

def test_ancestor_walk_includes_self_and_terminates():
    import os
    a = r._ancestor_pids()
    assert os.getpid() in a
    assert 1 not in a, "init is excluded separately; the walk stops before it"
    assert len(a) < 64, "walk must be bounded"


def test_caller_ancestry_is_never_a_target(monkeypatch):
    """`/clear-memory` inside gb shells out through sudo+bash, so the live gb
    process sits several levels up. Now that greenboost-cli is reclaimable,
    a shallow {pid, ppid} check would let the command kill its own caller.
    """
    import os
    gb_pid = 4242
    monkeypatch.setattr(r, "_ancestor_pids", lambda: {os.getpid(), 999, gb_pid})
    monkeypatch.setattr(r, "_fuser_pids", lambda path: [gb_pid, 777])
    monkeypatch.setattr(r, "_gpu_compute_pids", lambda: {gb_pid: 4096, 777: 4096})
    monkeypatch.setattr(r, "_proc_comm", lambda pid: "greenboost-cli")
    monkeypatch.setattr(r, "_live_pids", lambda: set())
    monkeypatch.setattr(r, "_recent_dataflux_pids", lambda: set())

    classes = r.classify_processes()
    seen = {e["pid"] for v in classes.values() for e in v}
    assert gb_pid not in seen, "reclaim would have killed its own caller"
    assert 777 in seen, "an unrelated CLI is still a candidate"


# ── the terminated/spared contradiction ──────────────────────────────────────

def test_a_target_is_never_also_reported_as_spared(classes, monkeypatch):
    """First real run printed PID 44218 under BOTH "Terminated" and "Still
    held (spared)". `spared` was hardcoded to the live class, which assumed no
    scope ever targets it — false for inference/all.
    """
    import os
    monkeypatch.setattr(r, "_ancestor_pids", lambda: {os.getpid()})
    monkeypatch.setattr(r, "os", os)
    killed_pids = []
    monkeypatch.setattr(r.os, "kill",
                        lambda pid, sig: killed_pids.append(pid))
    monkeypatch.setattr(r.time, "sleep", lambda s: None)
    monkeypatch.setattr(r, "_ollama_list_loaded", lambda: [])

    res = r.run_reclaim(scope="inference", term_wait_s=0)
    killed = {e["pid"] for e in res["killed"]}
    spared = {e["pid"] for e in res["spared"]}
    assert not (killed & spared), (
        f"pids reported as both terminated and spared: {killed & spared}")


def test_residue_scope_reports_the_live_server_as_spared(classes, monkeypatch):
    """The signal that made 'spared' worth having stays intact: at residue,
    a live server holding GB of VRAM must still be named, or '0 killed' reads
    as 'nothing is using GPU memory'."""
    import os
    monkeypatch.setattr(r, "_ancestor_pids", lambda: {os.getpid()})
    monkeypatch.setattr(r.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(r.time, "sleep", lambda s: None)
    monkeypatch.setattr(r, "_ollama_list_loaded", lambda: [])

    res = r.run_reclaim(scope="residue", term_wait_s=0)
    assert 100 in {e["pid"] for e in res["spared"]}

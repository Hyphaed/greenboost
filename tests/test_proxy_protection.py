"""The proxy must be BOUNDED, not made un-killable, and never orphan the engine.

Overnight 2026-08-19 the proxy reached 39 GB of anonymous memory (28 GB
swapped) and the kernel killed it at 01:47 , correctly, because it was the
largest thing on a 61 GB box. llama-server survived until 10:26 holding 10.5 GB
of a 12 GB card, and every request in between failed with "Cannot connect".

Marking the proxy un-killable would only redirect the kill at the desktop or
the engine while the proxy carried on toward every byte of RAM. So: cap it, and
make an orphaned engine self-clearing.
"""
import os

import pytest

import gb_synapse as gs


def test_the_cap_is_a_fraction_of_real_ram_not_a_literal():
    """Hardcoded-hardware rule , this must scale with the box it runs on."""
    assert 0 < gs._PROXY_MEM_FRACTION < 1.0
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    cap = max(int(gs._PROXY_MEM_FLOOR_GB * 1024 ** 3),
              int(total * gs._PROXY_MEM_FRACTION))
    assert cap < total, "a cap at or above total RAM protects nothing"
    assert cap >= gs._PROXY_MEM_FLOOR_GB * 1024 ** 3


def test_the_cap_would_have_stopped_the_incident():
    """39 GB was reached on a 61.5 GiB box; the cap must sit far below that."""
    total = 61.5 * 1024 ** 3
    cap = max(int(gs._PROXY_MEM_FLOOR_GB * 1024 ** 3),
              int(total * gs._PROXY_MEM_FRACTION))
    assert cap < 39 * 1024 ** 3, "the proxy could still reach 39 GB"


def test_applying_limits_never_raises(monkeypatch):
    """A cap that cannot be applied must not stop the serve , it runs in the
    child between fork and exec, where an exception is far more costly than a
    missing limit."""
    import resource

    def _boom(*a, **k):
        raise OSError("not permitted here")

    monkeypatch.setattr(resource, "setrlimit", _boom)
    gs._proxy_resource_limits()          # must not raise


def test_the_limit_is_actually_set_in_the_child(monkeypatch):
    seen = {}
    import resource

    monkeypatch.setattr(resource, "setrlimit",
                        lambda which, pair: seen.update(which=which, pair=pair))
    gs._proxy_resource_limits()
    assert seen["which"] == resource.RLIMIT_AS
    assert seen["pair"][0] == seen["pair"][1], "soft and hard must match"


# ── Orphan reaping ───────────────────────────────────────────────────────────

def _sup():
    import gb_supervisor
    return gb_supervisor.GreenBoostSupervisor.__new__(gb_supervisor.GreenBoostSupervisor)


def test_an_orphaned_engine_is_stopped(monkeypatch):
    stopped = []
    mod = type("M", (), {
        "ps": staticmethod(lambda: [{"model": "M", "llama_pid": 42,
                                     "proxy_error": "proxy process is gone"}]),
        "stop": staticmethod(lambda m: stopped.append(m)),
    })
    monkeypatch.setitem(__import__("sys").modules, "gb_synapse", mod)
    _sup()._reap_orphaned_serves()
    assert stopped == ["M"]


def test_a_healthy_session_is_left_alone(monkeypatch):
    stopped = []
    mod = type("M", (), {
        "ps": staticmethod(lambda: [{"model": "M", "llama_pid": 42,
                                     "proxy_error": None}]),
        "stop": staticmethod(lambda m: stopped.append(m)),
    })
    monkeypatch.setitem(__import__("sys").modules, "gb_synapse", mod)
    _sup()._reap_orphaned_serves()
    assert stopped == [], "stopped a working serve"


def test_a_failed_start_with_no_engine_is_left_alone(monkeypatch):
    """proxy_error with no engine is a failed START, not an orphan."""
    stopped = []
    mod = type("M", (), {
        "ps": staticmethod(lambda: [{"model": "M", "llama_pid": 0,
                                     "proxy_error": "proxy process is gone"}]),
        "stop": staticmethod(lambda m: stopped.append(m)),
    })
    monkeypatch.setitem(__import__("sys").modules, "gb_synapse", mod)
    _sup()._reap_orphaned_serves()
    assert stopped == []


def test_an_unavailable_gb_synapse_is_not_fatal(monkeypatch):
    def _boom():
        raise RuntimeError("gone")
    monkeypatch.setitem(__import__("sys").modules, "gb_synapse",
                        type("M", (), {"ps": staticmethod(_boom)}))
    _sup()._reap_orphaned_serves()        # must not raise

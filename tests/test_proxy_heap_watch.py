"""The proxy diagnoses its own leak, because reading the code did not find it.

It grew to 5.4 GB in 25 anonymous arenas of ~500 MB , unmistakably Python
objects , with only 4 sockets open and no unbounded module-level container
anywhere in gb_synapse_api.py. Two OOM kills (2026-08-18 01:47, 2026-08-19
01:47) and one address-space cap later, a census beats another hypothesis.
"""
import resource

import pytest

import gb_synapse_api as A


def test_it_can_read_its_own_rss():
    rss = A._own_rss_bytes()
    assert rss > 1024 * 1024, "self-RSS read returned nothing usable"


def test_the_census_reports_both_count_and_size():
    out = A._heap_census(top=5)
    assert "[heap]" in out
    assert "by count:" in out and "by size:" in out
    # Empty halves are the failure this caught in development: sys was not
    # imported at module level, so every getsizeof() raised and the size line
    # rendered blank while looking fine.
    size_line = [l for l in out.splitlines() if "by size:" in l][0]
    assert "MB" in size_line, "size census produced nothing"


def test_the_census_never_raises(monkeypatch):
    """It runs under memory pressure , it must not become the crash it reports."""
    import gc

    monkeypatch.setattr(gc, "get_objects", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    out = A._heap_census()
    assert "census failed" in out


def test_no_watch_without_a_cap(monkeypatch):
    """A dev run with no RLIMIT_AS has nothing to measure against.

    Driven with asyncio.run rather than a pytest-asyncio marker: this repo's
    test suite has no asyncio plugin configured, and a skipped test that looks
    like it ran is worse than no test.
    """
    import asyncio

    monkeypatch.setattr(resource, "getrlimit",
                        lambda w: (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    app = {}
    asyncio.run(A._start_heap_watch(app))
    assert "_heap_watch" not in app, "started a watcher with no cap to watch"


def test_a_watch_IS_started_when_capped(monkeypatch):
    import asyncio

    cap = 6 * 1024 ** 3
    monkeypatch.setattr(resource, "getrlimit", lambda w: (cap, cap))

    async def _go():
        app = {}
        await A._start_heap_watch(app)
        started = "_heap_watch" in app
        if started:
            app["_heap_watch"].cancel()
        return started

    assert asyncio.run(_go()), "capped proxy got no leak watcher"


def test_the_threshold_sits_below_the_cap():
    """It must fire while allocations still succeed, not once they fail."""
    assert 0 < A._HEAP_WATCH_FRACTION < 1.0
    cap = 6.15 * 1024 ** 3
    assert cap * A._HEAP_WATCH_FRACTION < cap * 0.9

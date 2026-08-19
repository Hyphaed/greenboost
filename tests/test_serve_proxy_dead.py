"""A serve session with a dead proxy is an OUTAGE, not a healthy idle box.

Overnight 2026-08-19 the proxy was OOM-killed at 01:47. llama-server survived,
holding 10.5 GB of VRAM. Every request failed with "Cannot connect to
http://localhost:11369/v1" for eight hours , and `serve_healthy` answered
MATCHED the whole time, because from every other angle the box looked fine: the
kernel module was loaded, VRAM was full, and the GPU was idle precisely BECAUSE
nothing could reach it.

`gb_synapse.ps()` already knew (`proxy_error: "proxy process is gone"`). The
information existed and both `ready` and the segment ignored it.
"""
import pytest

import gb_semantics as S


def _session(proxy_error=None, llama_pid=674471):
    return {"model": "M", "llama_pid": llama_pid, "proxy_pid": 674526,
            "port": 11369, "proxy_error": proxy_error, "ready": True}


@pytest.fixture
def stub_ps(monkeypatch):
    def _apply(sessions):
        mod = type("M", (), {"ps": staticmethod(lambda: sessions)})
        monkeypatch.setitem(__import__("sys").modules, "gb_synapse", mod)
    return _apply


def test_a_dead_proxy_is_reported(stub_ps):
    stub_ps([_session(proxy_error="proxy process is gone")])
    out = S._serve_sessions_with_dead_proxy()
    assert len(out) == 1 and out[0]["llama_pid"] == 674471


def test_a_healthy_session_reports_nothing(stub_ps):
    stub_ps([_session(proxy_error=None)])
    assert S._serve_sessions_with_dead_proxy() == []


def test_no_engine_means_no_orphan(stub_ps):
    """A proxy error with no engine is a failed start, not an orphaned engine."""
    stub_ps([_session(proxy_error="proxy process is gone", llama_pid=0)])
    assert S._serve_sessions_with_dead_proxy() == []


def test_an_unavailable_gb_synapse_reports_nothing(monkeypatch):
    """"Cannot tell" must not become "a proxy died" , the same rule that stops
    it becoming "everything is fine"."""
    def _boom():
        raise RuntimeError("import failed")
    monkeypatch.setitem(__import__("sys").modules, "gb_synapse",
                        type("M", (), {"ps": staticmethod(_boom)}))
    assert S._serve_sessions_with_dead_proxy() == []


def test_serve_healthy_is_false_during_the_outage(monkeypatch, stub_ps):
    stub_ps([_session(proxy_error="proxy process is gone")])
    monkeypatch.setattr(S, "resolve", lambda n, *a, **k: {
        "kmod_loaded": {"value": True}, "shim_fresh": {"value": True},
        "vram_fill_pct": {"value": 92.4}, "meets_fp8_floor": {"value": True},
    }.get(n, {"value": None}))
    matched, ev = S._seg_serve_healthy()
    assert matched is False, "reported healthy during a total outage"
    note = [e for e in ev if e.get("metric") == "serve_proxy_dead"]
    assert note and "synapse stop" in note[0]["action"], "no recovery named"

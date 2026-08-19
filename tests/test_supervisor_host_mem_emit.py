"""The host-memory pressure emit must actually reach dataflux.

This event exists because greenboost.ko's T2 OOM guard and _RamMonitor both
computed `critical` before each of the two 2026-08-18 OOM kills and wrote it
only to the journal. Adding the emit and then having it silently fail would
reproduce the very defect it was written to fix — which is exactly what
happened: the first version called `gb_dataflux.emit(kind=..., node=...)`, but
emit() takes ONE dict, so it raised TypeError straight into the best-effort
`except Exception: pass` and never wrote a single event.

Best-effort emits are the right contract (telemetry must never break a tick),
but they hide their own breakage. That makes an end-to-end test mandatory, not
optional: nothing else can tell the difference between "no transition happened"
and "every transition was thrown away".
"""
import json

import pytest

import gb_supervisor as S


@pytest.fixture
def flux(tmp_path, monkeypatch):
    log = tmp_path / "dataflux.jsonl"
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(log))
    return log


def _events(log, kind="host_mem_pressure"):
    if not log.exists():
        return []
    out = []
    for line in log.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") == kind:
            out.append(e)
    return out


def _monitor(meminfo, prev="ok"):
    m = S._RamMonitor()
    m._prev_state = prev
    m._read_meminfo = lambda: meminfo
    return m


# (mem_total_mb, mem_avail_mb, swap_total_mb, swap_used_mb)
CRITICAL = (61000, 2000, 39000, 30000)      # 3.3% available, 77% swap
WARN     = (61000, 6000, 39000, 18000)      # 9.8% available, 46% swap
OK       = (61000, 50000, 39000, 1000)      # 82% available, 2.6% swap


def test_a_critical_transition_is_recorded(flux):
    mon = _monitor(CRITICAL)
    assert mon.poll() == "critical"
    evs = _events(flux)
    assert len(evs) == 1, "the transition reached the journal but not dataflux"
    e = evs[0]
    assert e["state"] == "critical"
    assert e["prev_state"] == "ok"
    assert e["status"] == "error"
    assert e["mem_available_pct"] == pytest.approx(3.3, abs=0.1)


def test_a_warn_transition_is_recorded_as_warn(flux):
    assert _monitor(WARN).poll() == "warn"
    evs = _events(flux)
    assert len(evs) == 1 and evs[0]["status"] == "warn"


def test_recovery_is_recorded_too(flux):
    """Knowing when pressure CLEARED is half of reading an incident back."""
    mon = _monitor(OK, prev="critical")
    assert mon.poll() == "ok"
    evs = _events(flux)
    assert len(evs) == 1
    assert evs[0]["state"] == "ok" and evs[0]["prev_state"] == "critical"


def test_no_event_without_a_transition(flux):
    """Every 5 s tick must not spam the log; only changes are worth recording."""
    mon = _monitor(OK, prev="ok")
    for _ in range(5):
        mon.poll()
    assert _events(flux) == []


def test_poll_survives_a_broken_dataflux(flux, monkeypatch):
    """Telemetry must never break the supervisor's tick."""
    import gb_dataflux

    def _boom(_event):
        raise RuntimeError("dataflux exploded")

    monkeypatch.setattr(gb_dataflux, "emit", _boom)
    assert _monitor(CRITICAL).poll() == "critical"      # must not raise


def test_emit_is_called_with_a_single_dict(monkeypatch, flux):
    """The exact defect: emit() takes one dict, kwargs raise into the swallow."""
    import gb_dataflux

    seen = {}

    def _capture(event=None, /, *args, **kwargs):
        seen["args"] = (event, args, kwargs)

    monkeypatch.setattr(gb_dataflux, "emit", _capture)
    _monitor(CRITICAL).poll()
    event, args, kwargs = seen["args"]
    assert isinstance(event, dict), f"emit called positionally-wrong: {seen['args']}"
    assert not kwargs, f"emit called with kwargs, which raises: {kwargs}"
    assert event["kind"] == "host_mem_pressure"

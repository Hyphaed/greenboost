#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Regression tests for the GA3 services audit (docs/greenboost-audit/services-audit.md,
findings SVC-01 .. SVC-06). Fakes only: no root, no GPU, no module load, no
network beyond a loopback socket on an ephemeral port.
"""
import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import gb_supervisor as sup
import gb_orchestrator as orch
import gb_telemetry as tel
from test_gb_orchestrator import _make_metrics, _make_supervisor_orch, _MockGbControl

import pytest


@pytest.fixture(autouse=True)
def _no_real_dataflux(tmp_path, monkeypatch):
    """Nothing here may append to the owner's real flight recorder."""
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(tmp_path / "dataflux-test.jsonl"))


XID79 = "kernel: NVRM: Xid (PCI:0000:01:00): 79, pid=1234, GPU has fallen off the bus.\n"
XID119 = "kernel: NVRM: Xid (PCI:0000:01:00): 119, pid=0, Timeout waiting for RPC from GSP!\n"
XID31 = "kernel: NVRM: Xid (PCI:0000:01:00): 31, pid=88, Ch 00000010, MMU Fault\n"


# ── SVC-01: the driver reload guard ──────────────────────────────────────────

def test_fatal_xids_only_matches_kernel_nvrm_lines_with_fatal_codes():
    assert sup._fatal_xids(XID119) == [119]
    assert sup._fatal_xids(XID79) == []                       # needs a reboot: a reload cannot fix it
    assert sup._fatal_xids(XID31) == []                       # application fault
    # the supervisor's own log line and any prose mentioning both words
    assert sup._fatal_xids("gb_supervisor: Xid GPU fault , forcing NVIDIA driver module reset") == []
    assert sup._fatal_xids("the GPU printed an Xid yesterday") == []
    assert sup._fatal_xids("") == [] and sup._fatal_xids(None) == []


def _guard_env(tmp_path, monkeypatch, holders=(), loaded=("nvidia_drm", "nvidia_modeset", "nvidia")):
    monkeypatch.setattr(sup, "XID_RESET_STAMP", tmp_path / "xid_reset_ts")
    monkeypatch.setattr(sup, "XID_RESET_ENABLED", True)
    monkeypatch.setattr(sup, "XID_RESET_COOLDOWN_S", 3600)
    monkeypatch.setattr(sup, "_nvidia_device_holders", lambda *a, **k: list(holders))
    monkeypatch.setattr(sup, "_loaded_nvidia_modules", lambda: list(loaded))
    calls = []

    def fake_run(argv, **kw):
        calls.append(list(argv))
        return MagicMock(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(sup.subprocess, "run", fake_run)
    return calls


def test_reset_runs_once_then_cooldown_blocks_a_crash_loop(tmp_path, monkeypatch):
    calls = _guard_env(tmp_path, monkeypatch)
    assert sup._reset_nvidia_driver([79], now=10_000.0) == "reset"
    assert calls[0][0] == "rmmod" and "nvidia" in calls[0]
    assert calls[1] == ["modprobe", "nvidia_drm"]             # whole stack comes back
    n = len(calls)
    # supervisor restarts 10 s later (Restart=on-failure, RestartSec=10), same Xid still in the journal
    assert sup._reset_nvidia_driver([79], now=10_010.0) == "cooldown"
    assert len(calls) == n                                    # nothing ran
    assert sup._reset_nvidia_driver([79], now=10_000.0 + 3601) == "reset"


def test_reset_refused_while_a_process_holds_the_device(tmp_path, monkeypatch):
    calls = _guard_env(tmp_path, monkeypatch, holders=[4242])
    assert sup._reset_nvidia_driver([79], now=1.0) == "in_use"
    assert calls == []
    assert not (tmp_path / "xid_reset_ts").exists()           # no stamp for a reset that did not happen


def test_reset_can_be_switched_off(tmp_path, monkeypatch):
    calls = _guard_env(tmp_path, monkeypatch)
    monkeypatch.setattr(sup, "XID_RESET_ENABLED", False)
    assert sup._reset_nvidia_driver([79], now=1.0) == "disabled"
    assert calls == []


def test_reset_only_removes_modules_that_were_loaded(tmp_path, monkeypatch):
    calls = _guard_env(tmp_path, monkeypatch, loaded=("nvidia",))
    sup._reset_nvidia_driver([79], now=1.0)
    assert calls[0] == ["rmmod", "nvidia"] and calls[1] == ["modprobe", "nvidia"]


def test_nvidia_device_holders_sees_a_fake_proc_tree(tmp_path):
    (tmp_path / "10" / "fd").mkdir(parents=True)
    (tmp_path / "11" / "fd").mkdir(parents=True)
    os.symlink("/dev/nvidia0", tmp_path / "10" / "fd" / "3")
    os.symlink("/dev/null", tmp_path / "11" / "fd" / "3")
    assert sup._nvidia_device_holders(str(tmp_path)) == [10]


def _run_recovery_with_journal(tmp_path, monkeypatch, kernel_text, all_text=""):
    for name in ("STATE_DIR", "RUN_DIR"):
        monkeypatch.setattr(sup, name, tmp_path / name)
        (tmp_path / name).mkdir()
    monkeypatch.setattr(sup, "SENTINEL_FILE", tmp_path / "STATE_DIR" / "sentinel")
    monkeypatch.setattr(sup, "RUNNING_FILE", tmp_path / "STATE_DIR" / "running")
    monkeypatch.setattr(sup, "RECOVERY_CLASS", tmp_path / "RUN_DIR" / "recovery_class")
    monkeypatch.setattr(sup, "ECC_DBE_FLAG", tmp_path / "RUN_DIR" / "ecc_dbe_flag")
    monkeypatch.setattr(sup, "_GB_DEV", str(tmp_path / "no-device"))
    resets = []
    monkeypatch.setattr(sup, "_reset_nvidia_driver", lambda codes, **k: resets.append(codes))

    def fake_run(argv, **kw):
        if argv[0] == "lsmod":
            return MagicMock(stdout="greenboost 1 0", returncode=0)
        if argv[0] == "journalctl":
            return MagicMock(stdout=kernel_text if "-k" in argv else all_text, returncode=0)
        return MagicMock(stdout="", stderr="", returncode=0)
    monkeypatch.setattr(sup.subprocess, "run", fake_run)
    return sup._run_recovery(), resets


def test_recovery_ignores_its_own_xid_log_line(tmp_path, monkeypatch):
    own = "gb-supervisor: [gb_supervisor] WARNING Xid GPU fault , forcing NVIDIA driver module reset\n"
    fault, resets = _run_recovery_with_journal(tmp_path, monkeypatch, "", all_text=own)
    assert fault == "unknown" and resets == []


def test_recovery_ignores_an_application_xid(tmp_path, monkeypatch):
    fault, resets = _run_recovery_with_journal(tmp_path, monkeypatch, XID31, all_text=XID31 + "GPU")
    assert fault == "unknown" and resets == []


def test_recovery_hands_a_wedged_gpu_xid_to_the_guard(tmp_path, monkeypatch):
    fault, resets = _run_recovery_with_journal(tmp_path, monkeypatch, XID119 + XID31)
    assert fault == "xid_fault" and resets == [[119]]


# ── SVC-02: wake-ups, watchdog, atomic writes ────────────────────────────────

def test_tick_wait_is_one_blocking_wait_not_a_polling_loop(monkeypatch):
    s = sup.GreenBoostSupervisor.__new__(sup.GreenBoostSupervisor)
    s._running = True
    waits = []
    monkeypatch.setattr(sup, "POLL_SECS", 10)
    monkeypatch.setattr(sup._STOP, "wait", lambda t=None: waits.append(t) or False)
    monkeypatch.setattr(sup.time, "sleep", lambda *_: (_ for _ in ()).throw(AssertionError("sleep")))
    t = iter([0.0, 0.0, 10.0, 10.0])
    monkeypatch.setattr(sup.time, "monotonic", lambda: next(t))
    s._wait_next_tick(0.0)
    assert len(waits) == 1 and 9.9 < waits[0] <= 10.0


def test_stop_signal_ends_the_wait_at_once():
    s = sup.GreenBoostSupervisor.__new__(sup.GreenBoostSupervisor)
    s._running = True
    sup._STOP.clear()
    s._on_stop(15, None)
    t0 = time.monotonic()
    s._wait_next_tick(0.0)
    assert time.monotonic() - t0 < 1.0 and s._running is False
    sup._STOP.clear()


def test_watchdog_pings_at_half_interval(monkeypatch):
    monkeypatch.setenv("WATCHDOG_USEC", "60000000")
    monkeypatch.delenv("WATCHDOG_PID", raising=False)
    assert sup._watchdog_interval_s() == 30.0
    monkeypatch.setenv("WATCHDOG_PID", "1")                   # meant for another process
    assert sup._watchdog_interval_s() == 0.0
    monkeypatch.delenv("WATCHDOG_PID")
    monkeypatch.delenv("WATCHDOG_USEC")
    assert sup._watchdog_interval_s() == 0.0

    s = sup.GreenBoostSupervisor.__new__(sup.GreenBoostSupervisor)
    s._running = True
    sent = []
    monkeypatch.setattr(sup, "_sd_notify", lambda m: sent.append(m))
    monkeypatch.setattr(sup, "POLL_SECS", 10)
    clock = [0.0]
    monkeypatch.setattr(sup.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(sup._STOP, "wait", lambda t=None: clock.__setitem__(0, clock[0] + t) or False)
    s._wait_next_tick(2.5)
    assert sent.count("WATCHDOG=1\n") >= 4                    # one per 2.5 s over 10 s


def test_boot_recovery_extends_the_start_timeout(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(sup, "_sd_notify", lambda m: sent.append(m))
    _run_recovery_with_journal(tmp_path, monkeypatch, "")
    assert any(m.startswith("EXTEND_TIMEOUT_USEC=") for m in sent)


def test_atomic_write_leaves_no_partial_file_and_no_temp(tmp_path):
    p = tmp_path / "vram_pressure"
    sup._atomic_write(p, "state=ok\n")
    sup._atomic_write(p, "state=warn\n")
    assert p.read_text() == "state=warn\n"
    assert [x.name for x in tmp_path.iterdir()] == ["vram_pressure"]
    assert (p.stat().st_mode & 0o777) == 0o644


def test_atomic_write_failure_keeps_the_old_file(tmp_path, monkeypatch):
    p = tmp_path / "f"
    sup._atomic_write(p, "old")
    monkeypatch.setattr(sup.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("boom")))
    try:
        sup._atomic_write(p, "new")
    except OSError:
        pass
    assert p.read_text() == "old"
    assert [x.name for x in tmp_path.iterdir()] == ["f"]


# ── SVC-04: a dead phase writer is "unknown" ─────────────────────────────────

def _phase(tmp_path, pid, phase="INIT"):
    f = tmp_path / "phase"
    f.write_text(f"phase={phase}\nidle_ms=0\npid={pid}\nts=1\n")
    return str(f)


def test_phase_file_of_a_dead_writer_reads_as_unknown(tmp_path):
    proc = tmp_path / "proc"
    (proc / "77").mkdir(parents=True)
    assert tel.read_phase_file(_phase(tmp_path, 77, "INFERENCE"), str(proc)) == "INFERENCE"
    assert tel.read_phase_file(_phase(tmp_path, 78, "INFERENCE"), str(proc)) == ""
    assert tel.read_phase_file(str(tmp_path / "missing"), str(proc)) == ""


def test_supervisor_phase_reader_drops_a_dead_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(sup, "PHASE_FILE", Path(_phase(tmp_path, 2 ** 22 + 12345, "DEEP_IDLE")))
    assert sup._read_phase() == ""
    monkeypatch.setattr(sup, "PHASE_FILE", Path(_phase(tmp_path, os.getpid(), "DEEP_IDLE")))
    assert sup._read_phase() == "DEEP_IDLE"


def _restores(ctrl):
    return sum(1 for c in ctrl.calls if c[0] == "restore_baseline")


def test_envelope_is_released_after_the_idle_window(monkeypatch):
    monkeypatch.setattr(orch, "_ENVELOPE_IDLE_S", 300.0)
    o, ctrl = _make_supervisor_orch(monkeypatch)
    clock = [1000.0]
    monkeypatch.setattr(orch.time, "monotonic", lambda: clock[0])
    m = _make_metrics(); m.shim_phase = "MODEL_LOAD"; m.power_limit_w = 300.0; m.sm_clock_mhz = 1800
    o.on_metrics(m)
    assert o._envelope_engaged and _restores(ctrl) == 0
    # the writer died: telemetry now reports an unknown phase, card idle
    idle = _make_metrics(); idle.shim_phase = ""; idle.gpu_util_pct = 0.0
    clock[0] += 299; o.on_metrics(idle)
    assert o._envelope_engaged                                 # not yet
    clock[0] += 2; o.on_metrics(idle)
    assert not o._envelope_engaged and _restores(ctrl) == 1


def test_envelope_stays_while_the_card_is_busy(monkeypatch):
    monkeypatch.setattr(orch, "_ENVELOPE_IDLE_S", 300.0)
    o, ctrl = _make_supervisor_orch(monkeypatch)
    clock = [1000.0]
    monkeypatch.setattr(orch.time, "monotonic", lambda: clock[0])
    m = _make_metrics(); m.shim_phase = "MODEL_LOAD"; m.power_limit_w = 300.0; m.sm_clock_mhz = 1800
    o.on_metrics(m)
    busy = _make_metrics(); busy.shim_phase = ""; busy.gpu_util_pct = 40.0
    for _ in range(5):
        clock[0] += 200; o.on_metrics(busy)
    assert o._envelope_engaged and _restores(ctrl) == 0


def test_envelope_timer_can_be_disabled(monkeypatch):
    monkeypatch.setattr(orch, "_ENVELOPE_IDLE_S", 0.0)
    o, ctrl = _make_supervisor_orch(monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(orch.time, "monotonic", lambda: clock[0])
    m = _make_metrics(); m.shim_phase = "MODEL_LOAD"; m.power_limit_w = 300.0; m.sm_clock_mhz = 1800
    o.on_metrics(m)
    clock[0] += 10 ** 6
    idle = _make_metrics(); idle.shim_phase = ""
    o.on_metrics(idle)
    assert o._envelope_engaged


# ── SVC-05: orch_state.json writers ──────────────────────────────────────────

def _state_orch(mode, tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "_ORCH_STATE_FILE", tmp_path / "orch_state.json")
    with patch.object(orch.ReactiveOrchestrator, "_restore_ecc_degraded", lambda self: None):
        return orch.ReactiveOrchestrator(mode=mode, control=_MockGbControl())


def test_process_mode_does_not_overwrite_a_live_supervisor_file(tmp_path, monkeypatch):
    sv = _state_orch("supervisor", tmp_path, monkeypatch)
    sv._write_state_file()
    before = (tmp_path / "orch_state.json").read_text()
    pr = _state_orch("process", tmp_path, monkeypatch)
    pr._write_state_file()
    assert (tmp_path / "orch_state.json").read_text() == before
    assert json.loads(before)["mode"] == "supervisor"


def test_process_mode_writes_when_the_supervisor_file_is_stale(tmp_path, monkeypatch):
    sv = _state_orch("supervisor", tmp_path, monkeypatch)
    sv._write_state_file()
    p = tmp_path / "orch_state.json"
    d = json.loads(p.read_text()); d["updated_ts"] = time.time() - 3600; p.write_text(json.dumps(d))
    _state_orch("process", tmp_path, monkeypatch)._write_state_file()
    assert json.loads(p.read_text())["mode"] == "process"


def test_state_file_temp_name_is_per_process_and_cleaned(tmp_path, monkeypatch):
    sv = _state_orch("supervisor", tmp_path, monkeypatch)
    sv._write_state_file()
    assert [x.name for x in tmp_path.iterdir()] == ["orch_state.json"]


# ── SVC-06: orphan reaper confirmation ───────────────────────────────────────

def _reaper(monkeypatch, sessions):
    stopped = []
    fake = MagicMock()
    fake.ps = lambda: sessions
    fake.stop = lambda m: stopped.append(m)
    monkeypatch.setitem(sys.modules, "gb_synapse", fake)
    monkeypatch.setattr(sup, "REAP_CONFIRM_POLLS", 3)
    s = sup.GreenBoostSupervisor.__new__(sup.GreenBoostSupervisor)
    s._orphan_polls = {}
    return s, stopped


def test_engine_without_proxy_is_stopped_only_on_the_third_tick(monkeypatch):
    s, stopped = _reaper(monkeypatch, [{"model": "m", "proxy_error": "gone", "llama_pid": 5}])
    s._reap_orphaned_serves(); s._reap_orphaned_serves()
    assert stopped == []
    s._reap_orphaned_serves()
    assert stopped == ["m"]


def test_a_recovered_proxy_resets_the_count(monkeypatch):
    sessions = [{"model": "m", "proxy_error": "gone", "llama_pid": 5}]
    s, stopped = _reaper(monkeypatch, sessions)
    s._reap_orphaned_serves(); s._reap_orphaned_serves()
    sessions[0] = {"model": "m", "proxy_error": None, "llama_pid": 5}
    s._reap_orphaned_serves()
    sessions[0] = {"model": "m", "proxy_error": "gone", "llama_pid": 5}
    s._reap_orphaned_serves(); s._reap_orphaned_serves()
    assert stopped == []


# ── SVC-03: the A2A gateway ──────────────────────────────────────────────────

import http.client
import gb_a2a


def _serve(monkeypatch, token=None):
    if token:
        monkeypatch.setenv("GB_A2A_TOKEN", token)
    else:
        monkeypatch.delenv("GB_A2A_TOKEN", raising=False)
    monkeypatch.setattr(gb_a2a, "_RATE_PER_MIN", 10_000)
    srv = gb_a2a.ThreadingHTTPServer(("127.0.0.1", 0), gb_a2a._Handler)
    port = srv.server_address[1]
    monkeypatch.setattr(gb_a2a, "DEFAULT_BIND", f"127.0.0.1:{port}")
    monkeypatch.setattr(gb_a2a, "_emit_request", lambda *a, **k: None)
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    return srv, port


def _post(port, body, headers=None, host=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    h = {"Content-Type": "application/json", **(headers or {})}
    c.putrequest("POST", "/", skip_host=True)
    c.putheader("Host", host or f"127.0.0.1:{port}")
    for k, v in h.items():
        c.putheader(k, v)
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    c.putheader("Content-Length", str(len(data)))
    c.endheaders(data)
    r = c.getresponse()
    return r.status, json.loads(r.read() or b"{}")


def test_a2a_refuses_actuation_without_a_token(monkeypatch):
    ran = []
    monkeypatch.setitem(gb_a2a.gb_actuation.VERBS, "run_under_greenboost",
                        lambda **kw: ran.append(kw) or {"gate": {"allowed": True}})
    srv, port = _serve(monkeypatch)
    try:
        st, body = _post(port, {"jsonrpc": "2.0", "id": 1, "method": "run_under_greenboost",
                                "params": {"command": ["/bin/true"], "confirm": True}})
        assert st == 200 and "result" in body                    # no token: a dry run with an `auth` note, never an actuation (confirm authorizes nothing)
        assert "auth" in str(body["result"])                     # the reply says why it was not executed (the real actuation proof is tests/test_gb_a2a_auth.py)
        st, body = _post(port, {"jsonrpc": "2.0", "id": 2, "method": "run_under_greenboost",
                                "params": {"command": ["/bin/true"], "confirm": False}})
        assert "result" in body                                  # dry-run still served
    finally:
        srv.shutdown()


def test_a2a_allows_actuation_with_the_right_token_only(monkeypatch):
    ran = []
    monkeypatch.setitem(gb_a2a.gb_actuation.VERBS, "tier_actuate",
                        lambda **kw: ran.append(kw) or {"gate": {"allowed": True}})
    srv, port = _serve(monkeypatch, token="s3cret")
    try:
        req = {"jsonrpc": "2.0", "id": 1, "method": "tier_actuate",
               "params": {"lever": "x", "value": 1, "confirm": True}}
        assert _post(port, req)[0] == 401
        assert _post(port, req, {"Authorization": "Bearer wrong"})[0] == 401
        st, body = _post(port, req, {"Authorization": "Bearer s3cret"})
        assert st == 200 and "result" in body and len(ran) == 1
    finally:
        srv.shutdown()


def test_a2a_refuses_browser_shaped_requests(monkeypatch):
    srv, port = _serve(monkeypatch)
    try:
        req = {"jsonrpc": "2.0", "id": 1, "method": "shim_env", "params": {}}
        assert _post(port, req, {"Origin": "https://evil.example"})[0] == 403
        assert _post(port, req, host="evil.example:8790")[0] == 403           # DNS rebinding
        assert _post(port, req, {"Content-Type": "text/plain"})[0] == 415     # CORS-preflight dodge
        assert _post(port, req)[0] == 200
    finally:
        srv.shutdown()


def test_a2a_caps_the_body(monkeypatch):
    monkeypatch.setattr(gb_a2a, "_MAX_BODY", 64)
    srv, port = _serve(monkeypatch)
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c.putrequest("POST", "/", skip_host=True)
        c.putheader("Host", f"127.0.0.1:{port}")
        c.putheader("Content-Type", "application/json")
        c.putheader("Content-Length", "100000000")
        c.endheaders()
        assert c.getresponse().status == 413
    finally:
        srv.shutdown()


def test_state_file_write_does_not_follow_a_planted_symlink(tmp_path, monkeypatch):
    victim = tmp_path / "victim.txt"; victim.write_text("ORIGINAL")
    os.symlink(victim, tmp_path / f"orch_state.json.{os.getpid()}.tmp")
    os.symlink(victim, tmp_path / "orch_state.tmp")
    sv = _state_orch("supervisor", tmp_path, monkeypatch)
    sv._write_state_file()
    assert victim.read_text() == "ORIGINAL"
    assert json.loads((tmp_path / "orch_state.json").read_text())["mode"] == "supervisor"


# ── SVC-07: AppArmor remediation guard ───────────────────────────────────────

DENIAL = ('audit: type=1400 audit(1.2:3): apparmor="DENIED" operation="file_mmap" '
          'profile="snap-update-ns.firefox" name="/usr/local/lib/libgreenboost_audit.so"\n')


def test_apparmor_fault_needs_one_line_naming_the_stub():
    assert sup._apparmor_denies_stub(DENIAL)
    # the three words on three unrelated lines (the old test accepted this)
    spread = 'sshd: DENIED user x\ngb: greenboost started\nsystemd: apparmor.service ok\n'
    assert not sup._apparmor_denies_stub(spread)
    assert not sup._apparmor_denies_stub('apparmor="DENIED" name="/etc/passwd"\n')


def test_apparmor_patch_does_not_reload_policy_when_nothing_changed(tmp_path, monkeypatch):
    import types
    local = tmp_path / "local"; local.mkdir()
    f = local / "usr.lib.snapd.snap-confine.real"
    f.write_text("# GreenBoost\n/usr/local/lib/libgreenboost_audit.so mr,\n")
    real_path = sup.Path
    monkeypatch.setattr(sup, "Path", lambda p: f if str(p).endswith("snap-confine.real") and "local" in str(p) else real_path(p))
    calls = []
    monkeypatch.setattr(sup.subprocess, "run", lambda argv, **k: calls.append(argv) or MagicMock(returncode=0))
    sup._patch_apparmor()
    assert calls == []                                          # already present: no parser, no restart
    f.write_text("")
    sup._patch_apparmor()
    assert [c[0] for c in calls] == ["apparmor_parser", "systemctl"]

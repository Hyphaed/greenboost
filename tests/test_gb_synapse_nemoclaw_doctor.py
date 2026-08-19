#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_synapse.doctor()'s `nemoclaw` block — detects NVIDIA NemoClaw
on PATH and whether something other than our own gb-synapse proxy is
holding GB_SYNAPSE_PORT (see docs/nemoclaw-and-greenboost.md and the
Track 2 / rollout item 8 write-up in the NemoClaw audit plan). MCP Tool
Gaps rule: this check belongs in the tool (`synapse_doctor`), not only in
a doc a human has to remember to consult.

CPU-only, no real NemoClaw install, no real network peer — monkeypatched.
The port-liveness check is mocked at the socket layer (never binds a real
port) so these pass regardless of whether an actual gb-synapse instance
happens to be running on this box's GB_SYNAPSE_PORT.
"""
import gb_synapse as gs


class _FakeSocket:
    def __init__(self, connect_result: int):
        self._connect_result = connect_result

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def settimeout(self, _t):
        pass

    def connect_ex(self, _addr):
        return self._connect_result


def _patch_port_open(monkeypatch, open_: bool) -> None:
    monkeypatch.setattr(gs.socket, "socket",
                         lambda *a, **k: _FakeSocket(0 if open_ else 1))


def test_nemoclaw_absent_and_port_free(monkeypatch):
    monkeypatch.setattr(gs.shutil, "which", lambda name: None)
    _patch_port_open(monkeypatch, open_=False)
    d = gs._nemoclaw_report()
    assert d["cli_present"] is False
    assert d["cli_path"] == ""
    assert d["port_foreign"] is False
    assert d["port_owner_hint"] == ""


def test_nemoclaw_present_on_path(monkeypatch):
    monkeypatch.setattr(gs.shutil, "which",
                         lambda name: "/usr/local/bin/nemoclaw" if name == "nemoclaw" else None)
    _patch_port_open(monkeypatch, open_=False)
    d = gs._nemoclaw_report()
    assert d["cli_present"] is True
    assert d["cli_path"] == "/usr/local/bin/nemoclaw"


def test_port_held_by_our_own_proxy_is_not_foreign(monkeypatch):
    _patch_port_open(monkeypatch, open_=True)
    monkeypatch.setattr(gs, "ps", lambda: [{"port": gs.DEFAULT_PORT, "model": "x"}])
    d = gs._nemoclaw_report()
    assert d["port_foreign"] is False
    assert d["port_owner_hint"] == "a live gb-synapse server"


def test_port_held_by_foreign_process_is_flagged(monkeypatch):
    _patch_port_open(monkeypatch, open_=True)
    monkeypatch.setattr(gs, "ps", lambda: [])
    d = gs._nemoclaw_report()
    assert d["port_foreign"] is True
    assert d["port_owner_hint"] == "an unrecognized process"


def test_doctor_includes_nemoclaw_block(monkeypatch):
    monkeypatch.setattr(gs.gb_cluster, "feeders", lambda probe=True: [])
    monkeypatch.setattr(gs, "engine_installed", lambda: False)
    monkeypatch.setattr(gs, "engine_version", lambda: "")
    monkeypatch.setattr(gs, "hf_token", lambda: None)
    monkeypatch.setattr(gs.gb_synapse_backends, "_torch_env_dir", lambda: None)
    monkeypatch.setattr(gs, "_nemoclaw_report",
                         lambda: {"cli_present": False, "cli_path": "",
                                  "port_foreign": False, "port_owner_hint": ""})

    class _FakeNvml:
        def mem(self):
            return (0, 8000.0, 12000.0, 0)

        def device_name(self):
            return "Fake GPU"

    import sys
    import types
    fake_mod = types.ModuleType("gb_nvml")
    fake_mod.get_nvml = lambda *_a, **_k: _FakeNvml()
    monkeypatch.setitem(sys.modules, "gb_nvml", fake_mod)

    d = gs.doctor(probe_feeders=False)
    assert d["nemoclaw"] == {"cli_present": False, "cli_path": "",
                              "port_foreign": False, "port_owner_hint": ""}


def test_format_doctor_warns_on_foreign_port():
    d = {"host_gpu_name": "Fake GPU", "host_vram_total_mb": 10000,
         "host_ram_total_mb": 60000, "feeders": [],
         "aggregate_vram_mb": 10000, "aggregate_ram_mb": 60000,
         "engine_installed": True, "engine_version": "abc123",
         "hf_token_set": True, "torch_engine_ready": False,
         "torch_engine_env": "",
         "nemoclaw": {"cli_present": False, "cli_path": "",
                      "port_foreign": True,
                      "port_owner_hint": "an unrecognized process"}}
    out = gs._format_doctor(d)
    assert "NemoClaw:   WARNING" in out
    assert "an unrecognized process" in out


def test_format_doctor_shows_onboard_recipe_when_present():
    d = {"host_gpu_name": "Fake GPU", "host_vram_total_mb": 10000,
         "host_ram_total_mb": 60000, "feeders": [],
         "aggregate_vram_mb": 10000, "aggregate_ram_mb": 60000,
         "engine_installed": True, "engine_version": "abc123",
         "hf_token_set": True, "torch_engine_ready": False,
         "torch_engine_env": "",
         "nemoclaw": {"cli_present": True, "cli_path": "/usr/local/bin/nemoclaw",
                      "port_foreign": False, "port_owner_hint": ""}}
    out = gs._format_doctor(d)
    assert "NemoClaw:   detected" in out
    assert "onboard" in out


def test_format_doctor_silent_when_nemoclaw_absent_and_port_free():
    d = {"host_gpu_name": "Fake GPU", "host_vram_total_mb": 10000,
         "host_ram_total_mb": 60000, "feeders": [],
         "aggregate_vram_mb": 10000, "aggregate_ram_mb": 60000,
         "engine_installed": True, "engine_version": "abc123",
         "hf_token_set": True, "torch_engine_ready": False,
         "torch_engine_env": "",
         "nemoclaw": {"cli_present": False, "cli_path": "",
                      "port_foreign": False, "port_owner_hint": ""}}
    out = gs._format_doctor(d)
    assert "NemoClaw" not in out

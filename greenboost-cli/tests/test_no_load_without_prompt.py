"""Owner rule 2026-10-10: `gb` never loads a model unless it got a prompt or an
explicit serve request. The loader is faked; it raises if reached unasked."""
import sys

import pytest

from greenboost_cli import __main__ as gbmain
from greenboost_cli.slash_commands import backend_cmds


class FakeSynapse:
    def __init__(self):
        self.calls = []

    def serve(self, model, **kw):
        if not (kw.get("explicit") and kw.get("trigger")):
            raise AssertionError("loader reached without prompt or explicit request")
        self.calls.append((model, kw))

        class S:
            proxy_error = None
            port = 1
            ready = True
            model = "m"
            ctx = 0
        return S()

    def __getattr__(self, name):
        return lambda *a, **k: None


@pytest.fixture
def fake(monkeypatch):
    f = FakeSynapse()
    monkeypatch.setattr(backend_cmds, "_import_gb_synapse", lambda: f)
    monkeypatch.setattr(backend_cmds, "_llamacpp_running_pid", lambda s=None: None)
    monkeypatch.setattr(backend_cmds, "ollama_suspend_for_gpu_task", lambda *a, **k: None)
    import greenboost_cli.terminal.repl as repl
    monkeypatch.setattr(repl, "run_interactive", lambda s, initial_prompt=None: None)
    import greenboost_cli.environment.settings as st
    monkeypatch.setattr(st, "load_settings", lambda: {"model": "gb-synapse/m"})
    return f


@pytest.mark.parametrize("argv", [["gb"], ["gb", "status"], ["gb", "--version"], ["gb", "help"], ["gb", "vitals"]])
def test_no_prompt_loads_nothing(fake, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    try:
        gbmain.main()
    except SystemExit:
        pass
    assert fake.calls == []


def test_prompt_loads_on_demand_with_trigger(fake, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gb", "-p", "hello"])
    try:
        gbmain.main()
    except SystemExit:
        pass
    assert len(fake.calls) == 1
    assert fake.calls[0][1]["trigger"] == "prompt" and fake.calls[0][1]["explicit"] is True

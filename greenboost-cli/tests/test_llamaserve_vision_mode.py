"""/llamaserve asks text-only vs vision when the model has a projector, and passes serve(vision=...)."""
import sys

import pytest

from greenboost_cli.slash_commands import backend_cmds


class FakeSynapse:
    def __init__(self, projector="/x/mmproj-Q8_0.gguf"):
        self.projector = projector
        self.calls = []

    def serve(self, model, **kw):
        self.calls.append((model, kw))

        class S:
            proxy_error = None
            port = 1
            ready = True
            model = "m"
            ctx = 0
            llama_pid = 1
            kv_type = ""
            n_gpu_layers = 0
            feeders = ()
        return S()

    def model_projector(self, model):
        return self.projector

    def log_path(self, name):
        return "/dev/null"


@pytest.fixture
def env(monkeypatch):
    f = FakeSynapse()
    monkeypatch.setattr(backend_cmds, "_import_gb_synapse", lambda: f)
    monkeypatch.setattr(backend_cmds, "_llamacpp_running_pid", lambda s=None: None)
    monkeypatch.setattr(backend_cmds, "ollama_suspend_for_gpu_task", lambda *a, **k: None)
    picks = []
    import greenboost_cli.terminal.wizard_prompt as wp
    monkeypatch.setattr(wp, "run_picker", lambda opts, **k: picks.pop(0))
    return f, picks, monkeypatch


def _tty(monkeypatch, value):
    class In:
        def isatty(self):
            return value
    monkeypatch.setattr(sys, "stdin", In())


@pytest.mark.parametrize("arg,want", [("text", False), ("vision", True), ("start text", False), ("m vision", True)])
def test_token_skips_prompt(env, arg, want):
    f, picks, mp = env
    _tty(mp, True)
    backend_cmds.cmd_llamaserve(arg, None, {"model": "gb-synapse/m"})
    assert f.calls[0][1]["vision"] is want


@pytest.mark.parametrize("pick,want", [([1], False), ([2], True)])
def test_picker_choice(env, pick, want):
    f, picks, mp = env
    _tty(mp, True)
    picks.append(pick)
    backend_cmds.cmd_llamaserve("", None, {"model": "gb-synapse/m"})
    assert f.calls[0][1]["vision"] is want


@pytest.mark.parametrize("pick", [[3], None])
def test_cancel_starts_nothing(env, pick):
    f, picks, mp = env
    _tty(mp, True)
    picks.append(pick)
    backend_cmds.cmd_llamaserve("", None, {"model": "gb-synapse/m"})
    assert f.calls == []


def test_no_tty_keeps_default(env):
    f, picks, mp = env
    _tty(mp, False)
    backend_cmds.cmd_llamaserve("", None, {"model": "gb-synapse/m"})
    assert f.calls[0][1]["vision"] is None


def test_no_projector_no_prompt(env):
    f, picks, mp = env
    f.projector = None
    _tty(mp, True)
    backend_cmds.cmd_llamaserve("", None, {"model": "gb-synapse/m"})  # picks is empty: a prompt would IndexError
    assert f.calls[0][1]["vision"] is None

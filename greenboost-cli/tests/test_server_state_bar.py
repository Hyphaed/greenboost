"""The idle toolbar tells you whether the model is loaded, and follows it live."""
from greenboost_cli.terminal import repl as R


def test_state_fragments_follow_the_server(monkeypatch):
    monkeypatch.setattr(R, "_server_state", "stopped")
    assert "/llamaserve load" in R._server_state_frag()[1]
    monkeypatch.setattr(R, "_server_state", "starting")
    assert "loading" in R._server_state_frag()[1]
    monkeypatch.setattr(R, "_server_state", "running")
    assert "running" in R._server_state_frag()[1]
    monkeypatch.setattr(R, "_server_state", "unknown")
    assert R._server_state_frag() is None


def test_poller_flips_state(monkeypatch):
    import greenboost_cli.slash_commands.backend_cmds as B
    seq = iter(["stopped", "starting", "running"])
    monkeypatch.setattr(B, "llamacpp_server_status", lambda s: next(seq))
    seen = []

    class _Stop(Exception):
        pass

    def _sleep(_):
        seen.append(R._server_state)
        if len(seen) == 3:
            raise _Stop

    import time
    monkeypatch.setattr(time, "sleep", _sleep)
    try:
        R._server_state_poller({})
    except _Stop:
        pass
    assert seen == ["stopped", "starting", "running"]


def test_title_follows_state_and_goes_to_the_tty_not_stdout(monkeypatch, tmp_path):
    import builtins, io
    written = io.StringIO()
    real_open = builtins.open

    def fake_open(path, *a, **k):
        if path == "/dev/tty":
            class _T(io.StringIO):
                def close(self_inner):
                    written.write(self_inner.getvalue())
                    super().close()
            return _T()
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(R, "_title_pushed", True)   # skip the atexit registration
    monkeypatch.setattr(R, "_server_state", "starting")
    R._set_window_title({"model": "qwen38"})
    assert "loading model" in written.getvalue() and "qwen38" in written.getvalue()

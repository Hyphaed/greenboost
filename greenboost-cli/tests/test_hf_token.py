"""The hub's anonymous-download warning becomes an explanation with the command to fix it."""
from __future__ import annotations

import os
import stat
import sys
import types

import pytest

from greenboost_cli import hf_token as H

TOKEN = "hf_" + "abcdefghijklmnopqrstuvwxyz0123456789"


@pytest.fixture
def home(tmp_path, monkeypatch):
    tok_file = tmp_path / "hf" / "token"
    monkeypatch.setattr(H, "hf_token_path", lambda: tok_file)
    fake = types.SimpleNamespace(HF_TOKEN_FILE=tmp_path / "gb" / "hf_token",
                                 login=lambda t: (tmp_path / "gb").mkdir(exist_ok=True) or (tmp_path / "gb" / "hf_token").write_text(t))
    monkeypatch.setitem(sys.modules, "gb_synapse", fake)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(H, "_warned", False)
    import greenboost_cli.environment.settings as S
    monkeypatch.setattr(S, "HISTORY_PATH", tmp_path / "input_history.txt")
    return tmp_path


def test_save_writes_owner_only_files_and_sets_the_environment(home):
    where = H.save_token(TOKEN)
    assert os.environ["HF_TOKEN"] == TOKEN
    f = home / "hf" / "token"
    assert f.read_text().strip() == TOKEN
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert len(where) == 2 and (home / "gb" / "hf_token").exists()
    assert H.current_token() == TOKEN


def test_a_malformed_token_is_refused_and_nothing_is_written(home):
    with pytest.raises(ValueError):
        H.save_token("not-a-token")
    assert not (home / "hf" / "token").exists() and "HF_TOKEN" not in os.environ


def test_clear_removes_every_copy(home):
    H.save_token(TOKEN)
    H.clear_token()
    assert H.current_token() == "" and not (home / "hf" / "token").exists()


def test_the_token_is_scrubbed_from_input_history_but_other_lines_stay(home):
    from greenboost_cli.environment.settings import HISTORY_PATH
    HISTORY_PATH.write_text(f"\n# 2026-10-30 10:00:00\n+/help\n\n# 2026-10-30 10:01:00\n+/huggingface-token {TOKEN}\n\n"
                            f"# 2026-10-30 10:02:00\n+/doctor\n")
    assert H.scrub_history(TOKEN) == 1
    left = HISTORY_PATH.read_text()
    assert TOKEN not in left and "+/help" in left and "+/doctor" in left


def test_mask_never_shows_the_middle():
    assert H.mask(TOKEN) == "hf_…6789" and TOKEN[10:20] not in H.mask(TOKEN)


def _hub_sends_the_warning():
    """What the hub does for real: a response header, logged by huggingface_hub."""
    from huggingface_hub.utils import _http
    _http._WARNED_TOPICS.clear()
    resp = types.SimpleNamespace(headers=types.SimpleNamespace(get_list=lambda name: [
        "unauthenticated; You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN "
        "to enable higher rate limits and faster downloads."]))
    _http._warn_on_warning_headers(resp)


def test_the_hubs_warning_is_replaced_once_by_one_that_says_what_to_type(home, monkeypatch, capsys):
    pytest.importorskip("huggingface_hub")
    seen = []
    import greenboost_cli.terminal.theme as T
    monkeypatch.setattr(T, "emit_warn", lambda m: seen.append(m))
    H.install_warning_filter()
    _hub_sends_the_warning()
    _hub_sends_the_warning()
    assert len(seen) == 1                                           # once
    msg = seen[0]
    assert "/huggingface-token <your-token>" in msg                 # the action
    assert "Nothing is broken" in msg                               # what is NOT broken
    assert "https://huggingface.co/settings/tokens" in msg
    assert "unauthenticated" not in capsys.readouterr().err         # the hub's wording is gone


def test_no_replacement_message_when_a_token_is_already_set(home, monkeypatch):
    pytest.importorskip("huggingface_hub")
    seen = []
    import greenboost_cli.terminal.theme as T
    monkeypatch.setattr(T, "emit_warn", lambda m: seen.append(m))
    H.install_warning_filter()
    H.save_token(TOKEN)
    _hub_sends_the_warning()
    assert seen == []


def test_the_command_is_registered_and_saves(home, monkeypatch):
    from greenboost_cli.terminal.commands import COMMAND_TABLE
    import greenboost_cli.slash_commands.hf_token_cmds as C
    assert "huggingface-token" in COMMAND_TABLE
    out = []
    import greenboost_cli.terminal.theme as T
    monkeypatch.setattr(C, "emit_ok", lambda m: out.append(m))
    monkeypatch.setattr(C, "emit_info", lambda m: out.append(m))
    C.cmd_huggingface_token(TOKEN, None, {})
    assert H.current_token() == TOKEN
    assert all(TOKEN not in line for line in out)                   # never echoed back in full

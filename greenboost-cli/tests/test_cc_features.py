"""@file mentions, ! bash mode, friendly connection error, new slash commands."""
from types import SimpleNamespace

from greenboost_cli.terminal import cc_features as C


def test_at_mention_attaches_existing_file_only(tmp_path):
    (tmp_path / "a.txt").write_text("hello\nworld\n")
    text, seen = C.expand_file_mentions("look at @a.txt and ping @nobody, mail me@x.org", str(tmp_path))
    assert '<file path="' in text and "hello" in text
    assert seen == ["Read a.txt (3 lines)"] and text.count("<file") == 1


def test_at_mention_directory_and_binary(tmp_path):
    (tmp_path / "d").mkdir(); (tmp_path / "d" / "f.py").write_text("x")
    (tmp_path / "b.bin").write_bytes(b"\0\1\2")
    text, seen = C.expand_file_mentions("@d and @b.bin", str(tmp_path))
    assert "f.py" in text and any("binary" in s for s in seen)


def test_bash_line_runs_and_enters_context(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "_cwd", lambda: str(tmp_path))
    sess = SimpleNamespace(messages=[])
    C.run_bash_line("!echo hi-there", sess)
    assert "hi-there" in sess.messages[-1]["content"] and 'exit="0"' in sess.messages[-1]["content"]


def test_connection_error_names_the_load_command():
    head, hints = C.friendly_turn_error(
        "\nCannot connect to http://localhost:11369/v1\n\n  • /llamaserve — start gb-synapse\n")
    assert head.startswith("Model not loaded") and "11369" in head
    assert hints[0][0] == "/llamaserve load"


def test_commands_registered():
    from greenboost_cli.terminal.commands import COMMAND_TABLE
    for n in ("export", "copy", "rewind", "memory", "usage"):
        assert n in COMMAND_TABLE

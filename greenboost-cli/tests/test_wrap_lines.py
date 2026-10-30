"""gb_synapse's 300-column diagnostics wrap at words, not mid-word."""
import io

from greenboost_cli.terminal.width import WrapLines

LONG = ("  [gb-synapse] qwen38-35b-a3b-apex-miniplus: vision projector mmproj-Q8_0.gguf "
        "costs ~1.7 GiB (weights 0.57 + encoder buffers for 1280 image tokens); "
        "placement: recipe pin + 6 for the projector")


def _run(text, cols, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "get_terminal_size", lambda *a, **k: shutil.os.terminal_size((cols, 24)))
    out = io.StringIO()
    w = WrapLines(out)
    w.write(text)
    w.flush()
    return out.getvalue()


def test_long_line_wraps_on_word_boundaries_with_indent(monkeypatch):
    got = _run(LONG + "\n", 100, monkeypatch)
    lines = got.rstrip("\n").split("\n")
    assert len(lines) >= 2 and all(len(l) <= 98 for l in lines)
    assert all(l.startswith("        ") for l in lines[1:])          # hanging indent
    assert " ".join(got.split()) == " ".join(LONG.split())             # nothing lost or split


def test_short_lines_and_partial_writes_pass_through(monkeypatch):
    got = _run("ok\npart", 100, monkeypatch)
    assert got == "ok\npart"

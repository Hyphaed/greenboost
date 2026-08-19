"""Rendering invariants for the live action line, sweep bar and truncation.

Every bug these guard against has actually shipped in this file's history:
a status frame wider than the terminal wraps, `\r` returns to the wrapped row,
and the previous row is stranded in scrollback permanently. The invariant that
matters is not "looks right" but "can never exceed the width it was given".
"""
import re

import pytest

from greenboost_cli.terminal import renderer as R
from greenboost_cli.terminal.width import display_width, truncate_to_width

_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _plain(s: str) -> str:
    return _ANSI.sub("", s)


@pytest.fixture(autouse=True)
def _reset_renderer_state():
    R.set_vitals_provider(None)
    R.set_tool_label("")
    yield
    R.set_vitals_provider(None)
    R.set_tool_label("")


LABELS = [
    "",
    "Read · gb_synapse.py",
    "Bash · verifying the tree is green before the reinstall",
    "Edit · greenboost_cli/terminal/renderer.py with a very long trailing summary",
    "Bash · 汉字 wide glyphs 汉字 mixed with ascii",
]


@pytest.mark.parametrize("label", LABELS)
def test_action_frame_never_exceeds_width(label):
    R.set_vitals_provider(lambda: [("T1 87%", ""), ("T2 4.1G", ""), ("9.4 tok/s", "")])
    R.set_tool_label(label)
    for w in range(16, 221, 3):
        for frame in range(0, 24, 5):
            for elapsed in (0.2, 1.4, 4.2, 12.7, 95.0, 3601.0):
                got = _plain(R._action_frame(frame, elapsed, w))
                assert display_width(got) <= w, (
                    f"overflow at w={w} frame={frame} elapsed={elapsed}: {got!r}"
                )


def test_action_frame_survives_a_broken_vitals_provider():
    """A decoration must never take down a turn."""
    def boom():
        raise RuntimeError("nvml exploded")

    R.set_vitals_provider(boom)
    R.set_tool_label("Bash · something")
    got = _plain(R._action_frame(3, 5.0, 90))
    assert display_width(got) <= 90
    assert "5.0s" in got


def test_action_frame_always_shows_elapsed():
    R.set_tool_label("Bash · " + "x" * 300)
    for w in (24, 40, 80, 160):
        assert "12.7s" in _plain(R._action_frame(2, 12.7, w))


def test_action_frame_prefers_vitals_over_a_long_label():
    """Vitals are what a cloud CLI cannot show; they outrank label detail."""
    R.set_vitals_provider(lambda: [("T1 87%", ""), ("9.4 tok/s", "")])
    R.set_tool_label("Bash · " + "y" * 200)
    got = _plain(R._action_frame(1, 6.0, 100))
    assert "9.4 tok/s" in got
    assert "…" in got, "the long label should be truncated, not dropped whole"


def test_action_frame_animates():
    R.set_tool_label("Bash · x")
    frames = {_plain(R._action_frame(f, 2.0, 80)) for f in range(12)}
    assert len(frames) > 1, "the action line must actually move"


def test_sweep_bar_is_exactly_the_requested_width():
    for w in (1, 8, 20, 44):
        for f in range(0, 60, 7):
            assert display_width(R._sweep_bar(f, w)) == w


def test_sweep_bar_travels():
    assert len({R._sweep_bar(f, 30) for f in range(0, 40, 3)}) > 5


def test_truncate_resets_colour_when_it_cuts():
    """Otherwise a cut colour run bleeds into the next writer's row."""
    out = truncate_to_width("\x1b[36mhello world\x1b[0m", 5)
    assert display_width(out) == 5
    assert out.endswith("\x1b[0m")


def test_truncate_adds_no_reset_when_nothing_was_cut():
    src = "plain"
    assert truncate_to_width(src, 40) == src


def test_ellipsis_width_is_measured_not_assumed():
    """U+2026 is East-Asian-Ambiguous; assuming width 1 overflows by one column."""
    R.set_vitals_provider(lambda: [("T1 87%", "")])
    R.set_tool_label("Bash · " + "z" * 120)
    for w in range(30, 140):
        assert display_width(_plain(R._action_frame(0, 6.0, w))) <= w


def test_no_animation_on_a_non_tty(monkeypatch, capsys):
    """`gb -p` piped to a file must not accumulate escape frames.

    Measured before the guard: 815 bytes of \\r frames in 0.35 s — roughly
    2 KB/s of junk for the length of a headless run.
    """
    import time

    monkeypatch.setattr("sys.stdout.isatty", lambda: False, raising=False)
    R.set_tool_label("Bash · x")
    R._start_tool_spinner()
    time.sleep(0.25)
    R._stop_tool_spinner()
    assert capsys.readouterr().out == ""


def _fresh_width(monkeypatch):
    """Give width.py its own stdout, out of pytest's capture's reach."""
    import io
    import types

    from greenboost_cli.terminal import width as W

    buf = io.StringIO()
    monkeypatch.setattr(W, "sys", types.SimpleNamespace(stdout=buf))
    monkeypatch.setattr(W, "_at_line_start", True)
    monkeypatch.setattr(W, "_live_owns_row", False)
    return W, buf


def test_repeated_frames_reuse_ONE_row(monkeypatch):
    """The status line must repaint in place, not scroll.

    Owner report 2026-08-18, with a screenshot of ~25 stacked "Processing"
    lines 0.1 s apart: a frame's OWN bytes were counting as "someone dirtied
    the row", so every tick emitted a newline and took a new line. The live
    region owns the row it paints and overwrites it with \r.
    """
    W, buf = _fresh_width(monkeypatch)
    for i in range(25):
        W.tty_write(f"\r\x1b[2KProcessing {i / 10:.1f}s", live=True)
    assert buf.getvalue().count("\n") == 0, "each frame took its own line"


def test_a_frame_after_dirty_prose_commits_exactly_one_newline(monkeypatch):
    """Don't corrupt the prose, don't freeze, and don't scroll every tick."""
    W, buf = _fresh_width(monkeypatch)
    W.tty_write("tool output with no trailing newline")
    for i in range(5):
        W.tty_write(f"\r\x1b[2KProcessing {i}", live=True)
    out = buf.getvalue()
    assert out.startswith("tool output with no trailing newline"), "prose corrupted"
    assert out.count("\n") == 1, "should commit once, then own the row"


def test_real_output_takes_the_row_back(monkeypatch):
    """After prose interrupts, the next frame must re-acquire a row , once."""
    W, buf = _fresh_width(monkeypatch)
    W.tty_write("\r\x1b[2KProcessing", live=True)
    W.tty_write("\n  a real output line\n")
    before = buf.getvalue().count("\n")
    for i in range(3):
        W.tty_write(f"\r\x1b[2KProcessing {i}", live=True)
    assert buf.getvalue().count("\n") == before, "frames scrolled after prose"


def test_the_timer_keeps_advancing_on_a_permanently_dirty_row(monkeypatch):
    """The freeze this guard caused in its first form.

    Skipping frames while the row was dirty meant that if nothing ever wrote a
    newline again, the line froze: "the snake animation is stopped and the
    thinking seconds does not sum up".
    """
    W, buf = _fresh_width(monkeypatch)
    W.tty_write("dirty row, no newline coming")
    for t in (0.1, 0.2, 0.3):
        W.tty_write(f"\r\x1b[2KProcessing {t}s", live=True)
    out = buf.getvalue()
    for t in ("0.1s", "0.2s", "0.3s"):
        assert t in out, f"frame {t} was dropped , the line is frozen"


def test_a_live_frame_is_dropped_while_a_block_is_drawn(monkeypatch):
    """Suspension still wins , that one MUST drop the frame."""
    W, buf = _fresh_width(monkeypatch)
    with W.suspend_live():
        W.tty_write("FRAME", live=True)
    assert buf.getvalue() == "", "a frame leaked into a multi-row block"


def test_at_line_start_ignores_pure_escape_writes():
    """Cursor moves and colour changes emit no glyph, so they don't dirty a row."""
    import io

    from greenboost_cli.terminal import width as W

    real = W.sys.stdout
    W.sys.stdout = io.StringIO()
    try:
        W.tty_write("done\n")
        assert W.at_line_start()
        W.tty_write("\x1b[2K\x1b[36m")            # erase + colour, no glyph
        assert W.at_line_start(), "an escape-only write must not claim the row"
    finally:
        W.sys.stdout = real


# ── One live row, one owner ───────────────────────────────────────────────────

def test_two_painters_cannot_both_own_the_live_row(monkeypatch):
    """There is exactly ONE bottom row and two threads want it.

    The status line repaints every 80 ms from its own thread; the tool action
    line does the same from another. With no arbitration both claimed the row
    and the terminal grew a SECOND live line with its own timer , observed
    2026-08-18: "Processing … 0.6s" and "Processing … 81.0s" stacked together.
    """
    W, buf = _fresh_width(monkeypatch)
    W.claim_live("statusline")
    for i in range(5):
        W.tty_write(f"\rP{i}", live=True, owner="statusline")

    W.claim_live("action")                      # a tool call begins
    for i in range(5):
        W.tty_write(f"\rR{i}", live=True, owner="action")
    tail_before = buf.getvalue()

    for i in range(5):                          # status line keeps trying
        W.tty_write(f"\rP{i}", live=True, owner="statusline")

    assert buf.getvalue() == tail_before, "the non-owner painted anyway"
    assert buf.getvalue().count("\n") == 0, "the two painters took separate rows"


def test_releasing_hands_the_row_back(monkeypatch):
    """A tool call ending must not leave the status line mute for the session."""
    W, buf = _fresh_width(monkeypatch)
    W.claim_live("action")
    W.tty_write("\rRunning", live=True, owner="action")
    W.release_live("action")

    buf.truncate(0)
    buf.seek(0)
    W.tty_write("\rProcessing", live=True, owner="statusline")
    assert "Processing" in buf.getvalue(), "the row was never handed back"


def test_an_unowned_row_accepts_any_painter(monkeypatch):
    """Before anything claims it, a frame should still render."""
    W, buf = _fresh_width(monkeypatch)
    W.tty_write("\rProcessing", live=True, owner="statusline")
    assert "Processing" in buf.getvalue()


def test_release_by_a_non_owner_is_ignored(monkeypatch):
    """A stale stop must not steal the row from whoever holds it now."""
    W, buf = _fresh_width(monkeypatch)
    W.claim_live("action")
    W.release_live("statusline")                # wrong owner , no effect
    buf.truncate(0)
    buf.seek(0)
    W.tty_write("\rP", live=True, owner="statusline")
    assert buf.getvalue() == "", "a non-owner release freed the row"

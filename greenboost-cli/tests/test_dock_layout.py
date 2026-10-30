"""Pure layout functions of the permanent bottom dock (no terminal needed)."""
from greenboost_cli.terminal import dock
from greenboost_cli.terminal.width import display_width


def _text(frags):
    return "".join(t for _, t in frags)


def test_modes_and_row_allocation():
    assert dock.dock_mode(36) == "full" and dock.dock_mode(24) == "full"
    assert dock.dock_mode(12) == "full" and dock.dock_mode(11) == "compact"
    assert dock.dock_mode(8) == "compact" and dock.dock_mode(7) == "simple"
    # activity + 2 rules + 2 status + input rows
    assert dock.dock_rows(36, 1) == 6
    assert dock.dock_rows(36, 3) == 8
    assert dock.dock_rows(36, 99) == 5 + dock.MAX_INPUT_ROWS
    assert dock.dock_rows(9, 1) == 4 and dock.dock_rows(5, 1) == 2
    assert dock.max_input_rows(24) == 8 and dock.max_input_rows(12) == 4 and dock.max_input_rows(3) == 1


def test_truncate_by_display_width_not_len():
    wide = [("", "日本語テキスト"), ("", " tail")]      # double-width cells
    cut = dock.truncate_frags(wide, 9)
    assert dock.frags_width(cut) <= 9
    assert dock.frags_width(dock.truncate_frags([("", "abc")], 0)) == 0


def test_fit_parts_drops_whole_parts_and_keeps_order():
    parts = [[("", "aaaa")], [("", "b" * 30)], [("", "cc")]]
    out = _text(dock.fit_parts(parts, 14))
    assert out == "aaaa" + dock.SEP + "cc"        # the long middle part is dropped, not cut
    assert dock.fit_parts(parts, 1) == []


def test_compose_row_never_reaches_last_column():
    for cols in (20, 40, 80, 120, 200):
        left = [("", "L" * 300)]
        right = [("", "ctx 63%")]
        row = dock.compose_row(left, right, cols)
        assert dock.frags_width(row) <= cols - 1
        assert _text(row).endswith("ctx 63%")


def test_compose_row_drops_right_when_it_cannot_fit():
    row = dock.compose_row([("", "left")], [("", "R" * 50)], 30)
    assert _text(row) == "left"


def test_rule_has_session_name_right_aligned_and_exact_width():
    r = dock.rule_frags(80, "my session")
    assert display_width(_text(r)) == 79
    assert "my session" in _text(r)
    assert display_width(_text(dock.rule_frags(80))) == 79


def _state(**kw):
    base = dict(server=("fg:#a3e635", "● model running"), nonstop=True, ctx_pct=63.0,
                tiers=[("T1 11.0/11.9G", ""), ("T2 0.0/12.2G", ""), ("T3 0.0/73.0G", "")],
                hints=["esc interrupt", "ctrl+p pause", "ctrl+q compact", "type to queue"])
    base.update(kw)
    return dock.DockState(**base)


def test_status_rows_fit_every_width():
    for cols in (40, 60, 80, 120, 200):
        for st in (_state(), _state(plan_mode=True, gpu_owner="flux", queued=3),
                   _state(autonomous=True, auto_answer=True, statusline="x" * 80)):
            for fn in (dock.status1_frags, dock.status2_frags):
                row = fn(st, cols)
                assert dock.frags_width(row) <= cols - 1, (fn.__name__, cols)


def test_status1_carries_mode_server_levers_and_ctx():
    t = _text(dock.status1_frags(_state(queued=2), 120))
    assert "ask before risky tools" in t and "model running" in t
    assert "nonstop" in t and "2 queued" in t and t.rstrip().endswith("ctx 63%")
    assert "plan mode on" in _text(dock.status1_frags(_state(plan_mode=True), 120))
    assert "auto-accept on" in _text(dock.status1_frags(_state(autonomous=True), 120))


def test_status2_keeps_tiers_and_trims_hints_first():
    wide = _text(dock.status2_frags(_state(), 120))
    assert "T3 0.0/73.0G" in wide and "type to queue" in wide
    narrow = _text(dock.status2_frags(_state(), 70))
    assert "T3 0.0/73.0G" in narrow and "type to queue" not in narrow


def test_prefix_tags_and_continuation():
    assert _text(dock.prefix_frags(_state())) == "❯ "
    assert _text(dock.prefix_frags(_state(plan_mode=True, queued=2))) == "[PLAN] [2q] ❯ "
    assert _text(dock.prefix_frags(_state(), continuation=True)) == "  › "


def test_plan_hold_scrolls_only_when_cursor_is_inside_the_dock_rows():
    # 36 rows, 6-row dock: region is rows 1..30
    assert dock.plan_hold(10, 36, 6) == (0, 30, 10)      # mid-screen: leave it
    assert dock.plan_hold(30, 36, 6) == (0, 30, 30)
    assert dock.plan_hold(34, 36, 6) == (4, 30, 30)      # push 4 lines up first
    assert dock.plan_hold(36, 36, 6) == (6, 30, 30)


def test_static_dock_lines_match_mode_and_stay_in_width():
    for rows, n in ((36, 6), (24, 6), (10, 4)):
        lines = dock.static_dock_lines(_state(), 80, rows)
        assert len(lines) == n
        for ln in lines:
            assert display_width(ln) <= 79


def test_no_color_drops_colour_keeps_bold(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    out = dock.frags_to_ansi([("bold fg:#7c6dff", "❯ ")])
    assert "38;2" not in out and "\033[1m" in out
    monkeypatch.delenv("NO_COLOR")
    assert "38;2;124;109;255" in dock.frags_to_ansi([("fg:#7c6dff", "x")])

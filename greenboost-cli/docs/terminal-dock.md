# Terminal dock

greenboost-cli keeps a permanent dock on the last rows of the terminal, in every state
(idle, typing, turn running, tool running, permission prompt, wizard, after Esc, after a resize).

```
<scrollback, scrolls above>
 ⠹ Running…  Bash · make -j  (12s · ↓ 1.2k tokens · 38 t/s)     activity (blank when idle, hidden below 12 rows)
──────────────────────────────────────────── session name ─     rule
❯ input (grows upward to 8 rows, live during a turn)           input; [PLAN] [AUTO] [2q] tags before the glyph
──────────────────────────────────────────────────────────     rule
 ⏵ ask before risky tools · ● model running · nonstop    ctx 63% status 1: mode, model state, levers, GPU owner, queue
 esc interrupt · ctrl+p pause · ctrl+q compact   T1 · T2 · T3  status 2: hints (dropped from the end), tier stats
```

Focus does not depend on colour: the input row always starts with `❯`, plan/auto mode are spelled out
(`⏸ plan mode on`, `⏵⏵ auto-accept on`), and NO_COLOR removes colour but keeps bold and the glyphs.
Text that matters uses the theme's GRAY/TEAL/LIME/AMBER; DIM (#8b97b3, about 7:1 on black) is only reference text.

## Modes by terminal height

| rows | dock | rows used |
|---|---|---|
| >= 12 | full | activity, rule, input, rule, status 1, status 2 (6) |
| 8-11 | compact | rule, input, rule, status 1 (4) |
| < 8 | simple | input, status 1 (2) |

The completion menu (`/`, `@`) opens between the input and the lower rule (up to 8 rows); the dock still ends on the
last rows. Not a TTY, `TERM=dumb`, or no prompt_toolkit: the readline fallback and plain output are unchanged.

## Architecture

`greenboost_cli/terminal/dock.py`

* Pure layout (unit tested): `dock_mode`, `dock_rows`, `truncate_frags` (display width, not len), `fit_parts`,
  `compose_row` (never reaches the last column), `rule_frags`, `status1_frags`, `status2_frags`, `plan_hold`.
* `DockSession`: one prompt_toolkit `Application` reused for every prompt. Layout top to bottom: a weight-1 spacer
  (absorbs the free rows, so everything below sits on the last rows), activity, rule, input, menu, rule, status 1,
  status 2. `repl.py` builds a `DockState` snapshot (`_dock_state`) on every repaint; nothing else feeds it.
* `DockHold`: while a wizard or permission prompt owns the raw terminal, the input app is stopped and `DockHold`
  fences the last rows with a scroll region (DECSTBM), paints a static copy of the dock (`❯ answer above`) and parks the
  cursor under the scrollback. The wizard prints inside the region. On return the fence is dropped and the live dock is
  redrawn. Terminals under 10 rows skip the hold and flow as before.

## Ownership rule: who may write to the bottom rows

1. While the input app runs, **only prompt_toolkit** writes there. Threads (status line, spinner, tier stats) change
   state and call `app.invalidate()`. They never print escape sequences. Streamed text and tool cards go through
   `patch_stdout`, which prints above the dock.
2. While a wizard owns the terminal, **only `DockHold`** writes there (`get_hold()`, re-entrant, depth counted).
3. The old `Footer` (footer.py) is kept for the readline fallback only. It is never active while the dock is.
4. Exit paths restore the terminal (`_restore_terminal_on_exit` resets the scroll region, cursor, bracketed paste,
   alt screen). `DockHold.force_reset()` is the emergency path.

Esc during a turn interrupts it (same path as Ctrl+C); on an empty idle prompt it behaves as before (double Esc = rewind).

## Testing

```
cd greenboost-cli
python3 -m pytest tests/test_dock_layout.py tests/test_dock_pty.py -q -p no:cacheprovider   # pure + real PTY (skips without pyte)
python3 tools/pty_screen.py --rows 36 --cols 120 'waitfor:ctx :60' snap:idle 'send:hi\r' wait:3 snap:turn resize:100x28 snap:resized
```

`tools/pty_screen.py` runs the CLI in a pseudo-terminal and prints the pyte screen. Steps: `wait`, `waitfor:TEXT[:SECS]`,
`send` (`\r`, `\x1b`, `\x15` ctrl+u), `resize:COLSxROWS`, `snap`, `chk` (one-line dock check). `--cmd` runs another entry
point, `--env K=V` sets variables. The check requires: last row not blank, two rules, input directly above the lower rule.

The installed `/usr/local/bin/greenboost-cli` runs a copy under `cli-venv/lib/python3.14/site-packages/greenboost_cli`,
not this source. To exercise the working tree, put it first on the path:
`PYTHONPATH=$PWD /usr/local/lib/greenboost/cli-venv/bin/python -m greenboost_cli` (pass it as `--cmd` through a wrapper script).
Turns can be tested without a model: any server that streams OpenAI-style chat completions on the gb-synapse port works.

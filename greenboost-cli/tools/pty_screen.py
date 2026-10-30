#!/usr/bin/env python3
"""Run greenboost-cli in a pseudo-terminal and print what the SCREEN looks like (a pyte terminal emulator), at chosen moments.
Usage: pty_screen.py [--rows 36] [--cols 120] [--cmd greenboost-cli] [--env K=V ...] STEP...
STEP is one of:
  wait:SECONDS           pump output for that long
  waitfor:TEXT[:SECS]    pump until TEXT is on screen (default 60 s), then continue
  send:TEXT              type TEXT (\\r = Enter, \\x1b = Esc, \\x03 = Ctrl+C, \\x10 = Ctrl+P)
  resize:COLSxROWS       change the pty size (SIGWINCH) and the emulator
  snap:LABEL             print the screen and a dock check
  chk:LABEL              print only the dock check line
The dock check: the last rows must be  [activity] rule / input / rule / status / status  (the dock).
Example: pty_screen.py wait:14 snap:idle 'send:say hi in 3 words\\r' wait:1.5 snap:thinking wait:20 snap:done
"""
import argparse, os, pty, select, signal, struct, sys, time, fcntl, termios
import pyte

ap = argparse.ArgumentParser(); ap.add_argument("--rows", type=int, default=36); ap.add_argument("--cols", type=int, default=120)
ap.add_argument("--cmd", default="greenboost-cli"); ap.add_argument("--env", action="append", default=[]); ap.add_argument("--args", default=""); ap.add_argument("steps", nargs="+")
a = ap.parse_args()
screen = pyte.Screen(a.cols, a.rows); stream = pyte.ByteStream(screen)
pid, fd = pty.fork()
if pid == 0:
    os.environ["TERM"] = "xterm-256color"; os.environ["COLUMNS"] = str(a.cols); os.environ["LINES"] = str(a.rows)
    for kv in a.env:
        k, _, v = kv.partition("="); os.environ[k] = v
    os.execvp(a.cmd, a.cmd.split() + a.args.split())
def setsize(cols, rows):
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
setsize(a.cols, a.rows)
def feed_answering(d):
    """Feed bytes to the emulator and answer each cursor-position request AT ITS PLACE in the stream, like a real
    terminal does (answering after the whole chunk reports a cursor that later bytes in the chunk already moved)."""
    parts = d.split(b'\x1b[6n')
    for i, part in enumerate(parts):
        stream.feed(part)
        if i < len(parts) - 1:
            os.write(fd, f'\x1b[{screen.cursor.y + 1};{screen.cursor.x + 1}R'.encode())
def pump(sec, until=None):
    end = time.time() + sec
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.1)
        if r:
            try: d = os.read(fd, 65536)
            except OSError: return False
            if not d: return False
            feed_answering(d)
        if until and any(until in l for l in screen.display):
            return True
    return not until
def dock_check():
    d = [l.rstrip() for l in screen.display]
    n = len(d)
    rules = [i for i, l in enumerate(d) if l.startswith("──") and l.count("─") > 10]
    last = d[-1]
    ok_last = bool(last.strip())
    ok_rules = len(rules) >= 2 and rules[-1] >= n - 4 and rules[-2] <= rules[-1] - 2
    inp = d[rules[-1] - 1] if len(rules) >= 2 else ""
    return f"dock-check: last-row-nonblank={ok_last} rules-at={rules[-2:]} input-row={inp.strip()[:30]!r} -> {'OK' if ok_last and ok_rules else 'FAIL'}"
def snap(label):
    print(f"==== {label} ({screen.columns}x{screen.lines}) " + "=" * 40)
    for i, line in enumerate(screen.display):
        print(f"{i:2d}|{line.rstrip()}")
    print(dock_check())
for st in a.steps:
    k, _, v = st.partition(":")
    if k == "wait": pump(float(v))
    elif k == "waitfor":
        txt, _, secs = v.rpartition(":") if v.rpartition(":")[2].isdigit() else (v, "", "60")
        pump(float(secs or 60), until=txt)
    elif k == "send": os.write(fd, v.encode().decode("unicode_escape").encode("latin1")); pump(0.4)
    elif k == "resize":
        c, _, r = v.partition("x"); c, r = int(c), int(r)
        k = screen.cursor.y - (r - 1)               # like xterm: shrinking pushes top lines into history
        if k > 0:
            y = screen.cursor.y; screen.cursor_to_line(screen.lines)
            for _ in range(k): screen.index()
            screen.cursor.y = y - k
        screen.resize(r, c); setsize(c, r); os.kill(pid, signal.SIGWINCH); pump(1.0)
    elif k == "snap": snap(v)
    elif k == "chk": print(f"{v}: {dock_check()}")
try: os.kill(pid, signal.SIGTERM)
except ProcessLookupError: pass

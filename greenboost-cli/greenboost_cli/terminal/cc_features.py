"""Claude Code style conveniences that make sense for a local, self-hosted CLI.

  !<cmd>          run a shell command inline, output goes into the context
  @path           attach a file (or list a directory) to the prompt
  /export [file]  write the conversation to markdown
  /copy           last assistant reply to the clipboard
  friendly_turn_error()  one readable block instead of a raw RuntimeError

Deliberately absent: login/billing/account commands, remote sessions, IDE
links. This CLI only talks to a model on this machine.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

MAX_FILE_CHARS = 20_000
MAX_DIR_ENTRIES = 80
MAX_BASH_CHARS = 8_000

_MENTION = re.compile(r"(?<!\S)@([\w./~+\-]+)")


def _cwd() -> str:
    try:
        from greenboost_cli.instruments.handlers import _bash_cwd
        return _bash_cwd or os.getcwd()
    except Exception:
        return os.getcwd()


# ── @file mentions ───────────────────────────────────────────────────────────

def expand_file_mentions(text: str, cwd: "str | None" = None) -> "tuple[str, list[str]]":
    """Append the content of every existing `@path` to the prompt.

    Returns (new_text, summaries). The typed text is left as written, so the
    model still sees what the user referred to; paths that do not exist are
    ignored (`@someone` in prose stays prose)."""
    base = Path(cwd or _cwd())
    blocks: list[str] = []
    seen: set[Path] = set()
    summaries: list[str] = []
    for m in _MENTION.finditer(text):
        raw = m.group(1).rstrip(".,;:)")
        p = Path(os.path.expanduser(raw))
        p = p if p.is_absolute() else base / p
        try:
            p = p.resolve()
        except OSError:
            continue
        if p in seen or not p.exists():
            continue
        seen.add(p)
        if p.is_dir():
            names = sorted(e.name + ("/" if e.is_dir() else "") for e in p.iterdir())
            more = len(names) - MAX_DIR_ENTRIES
            body = "\n".join(names[:MAX_DIR_ENTRIES]) + (f"\n… {more} more" if more > 0 else "")
            blocks.append(f'<directory path="{p}">\n{body}\n</directory>')
            summaries.append(f"Listed {raw}/ ({len(names)} entries)")
            continue
        try:
            data = p.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:4096]:
            summaries.append(f"Skipped {raw} (binary)")
            continue
        body = sanitize(data.decode("utf-8", "replace").replace("\r\n", "\n")).replace("\x00", "")
        cut = len(body) > MAX_FILE_CHARS
        body = body[:MAX_FILE_CHARS] + ("\n… truncated" if cut else "")
        blocks.append(f'<file path="{p}">\n{body}\n</file>')
        summaries.append(f"Read {raw} ({body.count(chr(10)) + 1} lines{', truncated' if cut else ''})")
    if not blocks:
        return text, []
    return text + "\n\n" + "\n\n".join(blocks), summaries


# ── ! bash mode ──────────────────────────────────────────────────────────────

def run_bash_line(line: str, session) -> None:
    """`!cmd`: run it, show it like a tool block, and add it to the context."""
    from rich.markup import escape
    from greenboost_cli.terminal.theme import console, GRAY, DIM, AMBER, LIME
    cmd = line[1:].strip()
    if not cmd:
        return
    console.print(f"\n  [{LIME}]●[/] [{GRAY}]![/] {escape(cmd)}")
    try:
        r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True,
                           timeout=120, cwd=_cwd())
        out = (r.stdout or "") + (r.stderr or "")
        code = r.returncode
    except subprocess.TimeoutExpired:
        out, code = "timed out after 120 s", 124
    except OSError as e:
        out, code = str(e), 127
    out = sanitize(out)
    shown = out.rstrip("\n")
    lines = shown.split("\n") if shown else ["(no output)"]
    for i, ln in enumerate(lines[:12]):
        console.print(f"  [{DIM}]{'└' if i == 0 else ' '}[/] {escape(ln)}")
    if len(lines) > 12:
        console.print(f"  [{DIM}]  … +{len(lines) - 12} lines[/]")
    if code:
        console.print(f"  [{AMBER}]  exit {code}[/]")
    try:
        session.messages.append({
            "role": "user",
            "content": f"<bash-input>{cmd}</bash-input>\n"
                       f"<bash-output exit=\"{code}\">{out[:MAX_BASH_CHARS]}</bash-output>"})
    except Exception:
        pass


# ── turn errors ──────────────────────────────────────────────────────────────

def friendly_turn_error(msg: str) -> "tuple[str, list[tuple[str, str]]]":
    """(headline, [(command, what it does)]) for the errors people actually hit."""
    lines = [l for l in (msg or "").strip().splitlines() if l.strip()]
    head = lines[0].strip() if lines else "Turn failed"
    if head.startswith("Cannot connect"):
        url = head.split(" to ", 1)[-1].replace("gb-synapse at ", "")
        return (f"Model not loaded: nothing is listening at {url}",
                [("/llamaserve load", "load the model, then /retry resends your prompt"),
                 ("/llamaserve logs", "follow the startup log"),
                 ("/llamaserve status", "check whether it is up")])
    if "(HTTP 502)" in head or "(HTTP 503)" in head:
        return ("The model server is up but not answering yet: the model is probably still loading",
                [("watch the bar", "wait for '● model running', then /retry resends your prompt"),
                 ("/llamaserve logs", "follow the load"),
                 ("/llamaserve status", "check whether it is up")])
    hints = []
    for l in lines[1:]:
        l = l.strip().lstrip("•").strip()
        if "—" in l:
            a, b = l.split("—", 1)
            hints.append((a.strip(), b.strip()))
        elif l:
            hints.append(("", l))
    return head, hints


def print_turn_error(msg: str) -> None:
    from rich.markup import escape
    from greenboost_cli.terminal.theme import console, AMBER, GRAY, DIM
    head, hints = friendly_turn_error(msg)
    console.print(f"\n  [{AMBER}]✗[/] [{AMBER}]{escape(head)}[/]")
    for cmd, what in hints:
        if cmd:
            console.print(f"    [{GRAY}]{escape(cmd):<22}[/] [{DIM}]{escape(what)}[/]")
        else:
            console.print(f"    [{DIM}]{escape(what)}[/]")
    console.print()


# ── slash commands ───────────────────────────────────────────────────────────

def cmd_export(args: str, session, _settings) -> bool:
    from greenboost_cli.terminal.theme import emit_ok, emit_err
    path = Path(os.path.expanduser(args.strip())) if args.strip() else \
        Path.cwd() / f"greenboost-session-{time.strftime('%Y%m%d-%H%M%S')}.md"
    try:
        out = [f"# {getattr(session, 'name', None) or 'GreenBoost session'}\n"]
        for m in session.messages:
            c = m.get("content", "")
            out.append(f"## {m.get('role', '?')}\n\n{c if isinstance(c, str) else c!r}\n")
        path.write_text("\n".join(out), encoding="utf-8")
    except OSError as e:
        emit_err(f"Could not write {path}: {e}")
        return True
    emit_ok(f"Exported {len(session.messages)} messages to {path}")
    return True


def cmd_copy(_args: str, session, _settings) -> bool:
    from greenboost_cli.terminal.theme import emit_ok, emit_err
    last = next((m.get("content") for m in reversed(session.messages)
                 if m.get("role") == "assistant" and isinstance(m.get("content"), str)
                 and m.get("content").strip()), None)
    if not last:
        emit_err("Nothing to copy yet.")
        return True
    for tool in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "-ib"]):
        if shutil.which(tool[0]):
            try:
                subprocess.run(tool, input=last, text=True, timeout=5, check=True)
                emit_ok(f"Copied the last reply ({len(last)} chars) with {tool[0]}")
                return True
            except (OSError, subprocess.SubprocessError):
                continue
    emit_err("No clipboard tool found (install wl-clipboard or xclip).")
    return True


def register(table: dict) -> None:
    """Add the commands, and the Claude Code names for ones we already have."""
    table.setdefault("export", cmd_export)
    table["rewind"] = cmd_rewind
    table.setdefault("statusline", cmd_statusline)
    table.setdefault("studio", cmd_studio)
    table.setdefault("output-style", cmd_output_style)
    table.setdefault("gpu", cmd_gpu)
    table.setdefault("hooks", cmd_hooks)
    table.setdefault("vim", cmd_vim)
    table.setdefault("think", cmd_think)
    table.setdefault("commands", cmd_commands)
    table.setdefault("copy", cmd_copy)
    for alias, target in (("memory", "memories"),
                          ("usage", "cost"), ("todos", "todo"), ("rename", "name"),
                          ("tasks", "todo"), ("stats", "cost")):
        if target in table:
            table.setdefault(alias, table[target])


# ── untrusted display text ───────────────────────────────────────────────────

_CTRL = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-_]|[\x00-\x08\x0b-\x1f\x7f]")


def sanitize(text: str) -> str:
    """Strip escape sequences and control bytes from text we did not write.

    Command output, filenames and file content can carry cursor moves or title
    changes; written through a raw terminal API they would act on the screen."""
    return _CTRL.sub("", text)


# ── /gpu and the load preflight ──────────────────────────────────────────────

def gpu_snapshot() -> "dict | None":
    """{total, used, free (MiB), apps: [(pid, label, MiB)]} from nvidia-smi, or None."""
    def smi(*a):
        return subprocess.run(["nvidia-smi", *a], capture_output=True, text=True, timeout=5).stdout
    try:
        mem = smi("--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits")
        total, used, free = (int(float(x)) for x in mem.strip().splitlines()[0].split(","))
        apps = []
        for ln in smi("--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits").splitlines():
            if "," not in ln:
                continue
            pid, mib = ln.split(",")
            apps.append((int(pid), _app_label(int(pid)), int(float(mib))))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None
    return {"total": total, "used": used, "free": free, "apps": apps}


def _app_label(pid: int) -> str:
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return f"pid {pid}"
    low = cmd.lower()
    if "llama-server" in low:
        return "gb-synapse model (llama-server)"
    for key, label in (("comfy", "ComfyUI"), ("flux", "FLUX generation"), ("gen_gb", "ai-forge generation"),
                       ("forge", "ai-forge job"), ("svg-animation", "SVG Animation Studio"),
                       ("ollama", "Ollama"), ("vllm", "vLLM"), ("blender", "Blender")):
        if key in low:
            return label
    return Path(cmd.split()[0]).name if cmd.strip() else f"pid {pid}"


#: Qwen3.8-35B-A3B + mmproj needs about this much free on the 12 GB card.
NEED_MIB = 9500


def gpu_verdict(snap: dict, need_mib: int = NEED_MIB) -> str:
    if snap["free"] >= need_mib:
        return "ok"
    # Only a process holding >= 1 GiB counts as a competing heavy job; the
    # desktop (browser, chat, file manager) is background, not a culprit.
    heavy = [a for a in snap["apps"] if "gb-synapse" not in a[1] and a[2] >= 1024]
    if heavy:
        names = ", ".join(f"{l} {m / 1024:.1f} GiB" for _, l, m in heavy)
        return (f"only {snap['free'] / 1024:.1f} GiB free; held by {names}. "
                "One heavy model at a time: stop it or wait, nothing is killed here.")
    desk = sum(m for _, l, m in snap["apps"] if "gb-synapse" not in l)
    return (f"{snap['free'] / 1024:.1f} GiB free (desktop apps hold {desk / 1024:.1f} GiB); "
            "gb-synapse will spill to host RAM if the model doesn't fit")


def cmd_gpu(_args: str, _session, _settings) -> bool:
    from rich.markup import escape
    from greenboost_cli.terminal.theme import console, GRAY, DIM, LIME, AMBER, emit_err
    snap = gpu_snapshot()
    if not snap:
        emit_err("nvidia-smi isn't answering; can't read the card.")
        return True
    pct = snap["used"] / max(1, snap["total"])
    bar = "█" * round(pct * 24) + "░" * (24 - round(pct * 24))
    col = AMBER if pct > 0.85 else LIME
    console.print(f"\n  [{GRAY}]GPU[/]  [{col}]{bar}[/]  {snap['used'] / 1024:.1f} / {snap['total'] / 1024:.1f} GiB"
                  f"  [{DIM}]{snap['free'] / 1024:.1f} GiB free[/]")
    for _, label, mib in sorted(snap["apps"], key=lambda a: -a[2]):
        console.print(f"    [{GRAY}]{escape(label):<34}[/] [{DIM}]{mib / 1024:5.1f} GiB[/]")
    if not snap["apps"]:
        console.print(f"    [{DIM}]no compute processes[/]")
    v = gpu_verdict(snap)
    console.print(f"  [{LIME if v == 'ok' else AMBER}]{'room for the model' if v == 'ok' else v}[/]\n")
    return True


# ── /hooks, /vim, /think ─────────────────────────────────────────────────────

def cmd_hooks(_args: str, _session, _settings) -> bool:
    import json
    from rich.markup import escape
    from greenboost_cli.terminal.theme import console, GRAY, DIM, emit_info
    cfg = Path.home() / ".greenboost_cli" / "hooks.json"
    try:
        data = json.loads(cfg.read_text())
    except (OSError, ValueError):
        emit_info(f"No hooks configured. Create {cfg} (PreToolUse, PostToolUse, UserPromptSubmit).")
        return True
    for event, items in data.items():
        console.print(f"  [{GRAY}]{escape(event)}[/]")
        for it in items:
            console.print(f"    [{DIM}]{escape(str(it.get('matcher', '*'))):<10}[/] {escape(str(it.get('command', '')))}")
    return True


def cmd_vim(_args: str, _session, settings) -> bool:
    from greenboost_cli.terminal.theme import emit_ok
    on = not settings.get("vim_mode", False)
    settings["vim_mode"] = on
    try:
        from prompt_toolkit.enums import EditingMode
        from greenboost_cli.terminal import repl as R
        R._pt_session.editing_mode = EditingMode.VI if on else EditingMode.EMACS
    except Exception:
        pass
    emit_ok(f"Editor mode: {'vim' if on else 'emacs'}")
    return True


def cmd_think(args: str, _session, settings) -> bool:
    from greenboost_cli.terminal.theme import emit_ok, emit_info
    a = args.strip().lower()
    if a in ("on", "off"):
        # two settings read it: enable_thinking (server-side template) and qwen_thinking (prompt side)
        settings["enable_thinking"] = settings["qwen_thinking"] = a == "on"
    elif a:
        emit_info("Usage: /think [on|off]")
        return True
    emit_ok("Thinking: " + ("on" if settings.get("enable_thinking") else "off")
            + "  (server-side; off is faster and the default for Qwen3.8)")
    return True


def cmd_commands(_args: str, _session, _settings) -> bool:
    from rich.markup import escape
    from greenboost_cli.terminal.custom_commands_view import rows
    from greenboost_cli.terminal.theme import console, GRAY, DIM, emit_info
    r = rows()
    if not r:
        emit_info("No markdown commands found. Put name.md in .greenboost/commands/")
    for name, desc, src in r:
        console.print(f"  [{GRAY}]/{escape(name):<18}[/] [{DIM}]{escape(desc or '')}  ({src})[/]")
    return True


# ── /rewind picker ───────────────────────────────────────────────────────────

def _user_turns(session) -> "list[tuple[int, str]]":
    """(index in session.messages, text) for every real user prompt."""
    out = []
    for i, m in enumerate(session.messages):
        c = m.get("content")
        if m.get("role") == "user" and isinstance(c, str) and not c.startswith(("<bash-", "[")):
            # @file mentions append their content to the stored prompt; show only what was typed.
            out.append((i, re.split(r"\n\n<(?:file|directory) path=", c, 1)[0]))
    return out


def cmd_rewind(args: str, session, settings) -> bool:
    """/rewind: pick a past prompt and drop the conversation back to just before it."""
    from greenboost_cli.terminal.theme import emit_ok, emit_info, emit_err
    if args.strip() == "last":
        from greenboost_cli.terminal.commands import cmd_undo
        return cmd_undo("", session, settings)
    turns = _user_turns(session)[-10:]
    if not turns:
        emit_err("Nothing to rewind to yet.")
        return True
    from greenboost_cli.terminal.wizard_prompt import run_picker
    opts = [{"label": (t.replace("\n", " ")[:70] + ("…" if len(t) > 70 else "")),
             "description": f"{len(session.messages) - i} messages after this point are dropped"}
            for i, t in turns]
    opts.append({"label": "Cancel", "description": ""})
    picked = run_picker(opts, title="Rewind to before…")
    if not picked or picked[0] == len(opts):
        emit_info("Rewind cancelled.")
        return True
    idx, text = turns[picked[0] - 1]
    dropped = len(session.messages) - idx
    del session.messages[idx:]
    emit_ok(f"Rewound {dropped} message(s). Your prompt, to edit and resend:")
    from rich.markup import escape
    from greenboost_cli.terminal.theme import console, DIM
    console.print(f"  [{DIM}]{escape(sanitize(text)[:400])}[/]")
    return True


# ── permission rules, status line ────────────────────────────────────────────

def cmd_permission_rules(args: str, settings) -> "bool":
    """Sub-commands of /permissions: allow, deny, remove, rules. True when handled."""
    from rich.markup import escape
    from greenboost_cli.core import permission_rules as PR
    from greenboost_cli.terminal.theme import console, GRAY, DIM, LIME, AMBER, emit_ok, emit_err
    parts = args.strip().split(None, 1)
    if not parts or parts[0] not in ("allow", "deny", "remove", "rules"):
        return False
    verb, rule = parts[0], (parts[1].strip() if len(parts) > 1 else "")
    f = PR.global_file()
    try:
        import json
        data = json.loads(f.read_text())
    except (OSError, ValueError):
        data = {}
    allow, deny = list(data.get("allow", [])), list(data.get("deny", []))
    if verb == "rules":
        merged = PR.load()
        for label, col, items in (("deny", AMBER, merged["deny"]), ("allow", LIME, merged["allow"])):
            for r in items:
                console.print(f"  [{col}]{label:<6}[/] {escape(r)}")
        if not merged["allow"] and not merged["deny"]:
            console.print(f"  [{DIM}]No rules. /permissions allow Bash(git status:*)  ·  /permissions deny Read(./.env)[/]")
        return True
    if not rule:
        emit_err(f"Usage: /permissions {verb} Tool(pattern)")
        return True
    if verb == "remove":
        allow = [r for r in allow if r != rule]; deny = [r for r in deny if r != rule]
    elif verb == "allow" and rule not in allow:
        allow.append(rule)
    elif verb == "deny" and rule not in deny:
        deny.append(rule)
    PR.save_global(allow, deny)
    emit_ok(f"{verb}: {rule}")
    return True


def statusline_text(settings: dict) -> str:
    """Run the user's `statusline_command` and return one short sanitized line."""
    cmd = settings.get("statusline_command")
    if not cmd:
        return ""
    import json
    try:
        r = subprocess.run(["bash", "-c", cmd], input=json.dumps(
            {"model": settings.get("model", ""), "cwd": _cwd()}), capture_output=True,
            text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return ""
    line = sanitize((r.stdout or "").strip().splitlines()[0] if (r.stdout or "").strip() else "")
    return line[:60]


def cmd_statusline(args: str, _session, settings) -> bool:
    from greenboost_cli.environment.settings import save_settings
    from greenboost_cli.terminal.theme import emit_ok, emit_info
    a = args.strip()
    if not a:
        emit_info("Status line command: " + (settings.get("statusline_command") or "none")
                  + "\n  /statusline <shell command>  (gets {model, cwd} as JSON on stdin; first line is shown)"
                  + "\n  /statusline off")
        return True
    if a == "off":
        settings.pop("statusline_command", None)
    else:
        settings["statusline_command"] = a
    save_settings(settings)
    emit_ok("Status line " + ("removed" if a == "off" else f"set: {a}"))
    return True


# ── /studio, /output-style ───────────────────────────────────────────────────

_STUDIO_SERVER = Path.home() / "Dev" / "svg-animation-studio" / "packages" / "mcp" / "dist" / "server.mjs"


def studio_status(timeout: float = 25.0) -> "dict":
    """One-shot: start the studio's MCP server, ask it, stop it. {"summary", "projects", "error"}."""
    import threading
    out: dict = {}

    def work():
        from greenboost_cli.mcp.client import MCPStdioClient
        c = MCPStdioClient("svg-animation-studio", ["node", str(_STUDIO_SERVER)])
        try:
            if not c.connect():
                out["error"] = "the studio MCP server did not start"
                return
            import json
            try:
                out["summary"] = json.loads(c.call_tool("gpu_status", {})).get("summary", "")
            except (ValueError, AttributeError):
                out["summary"] = ""
            lp = c.call_tool("list_projects", {})
            out["projects"] = None if lp.startswith("ERROR") else lp
            if lp.startswith("ERROR"):
                out["projects_error"] = lp[:120]
        except Exception as e:
            out["error"] = str(e)[:120]
        finally:
            c.close()

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        out.setdefault("error", f"no answer after {timeout:.0f}s")
    return out


def cmd_studio(_args: str, _session, _settings) -> bool:
    from rich.markup import escape
    from greenboost_cli.terminal.theme import console, GRAY, DIM, AMBER, LIME, emit_err
    if not _STUDIO_SERVER.exists():
        emit_err(f"SVG Animation Studio MCP server not found at {_STUDIO_SERVER}")
        return True
    st = studio_status()
    if st.get("error"):
        emit_err(f"Studio: {st['error']}")
        return True
    console.print(f"\n  [{GRAY}]SVG Animation Studio[/]  [{LIME}]reachable[/]")
    if st.get("summary"):
        console.print(f"    [{DIM}]{escape(sanitize(st['summary']))}[/]")
    if st.get("projects"):
        for ln in sanitize(st["projects"]).splitlines()[:12]:
            console.print(f"    {escape(ln)}")
    elif st.get("projects_error"):
        console.print(f"    [{AMBER}]projects: {escape(st['projects_error'])}[/]")
    console.print()
    return True


def cmd_output_style(args: str, _session, settings) -> bool:
    from rich.markup import escape
    from greenboost_cli import output_styles as OS
    from greenboost_cli.environment.settings import save_settings
    from greenboost_cli.terminal.theme import console, GRAY, DIM, LIME, emit_ok, emit_err
    a = args.strip()
    styles = OS.available()
    if not a:
        cur = settings.get("output_style") or "default"
        for name, (_, desc) in [("default", (None, "no extra instructions"))] + sorted(styles.items()):
            mark = f"[{LIME}]●[/]" if name == cur else " "
            console.print(f"  {mark} [{GRAY}]{escape(name):<12}[/] [{DIM}]{escape(desc)}[/]")
        return True
    if a == "default":
        settings.pop("output_style", None)
    elif a in styles:
        settings["output_style"] = a
    else:
        emit_err(f"Unknown style {a!r}. /output-style lists them.")
        return True
    save_settings(settings)
    emit_ok(f"Output style: {a}")
    return True


# ── wait for a model that is still loading ───────────────────────────────────

def await_model_ready(settings: dict, on_wait=None, cancelled=None, timeout: float = 900.0,
                      poll: float = 1.0, status_fn=None, sleep=time.sleep) -> bool:
    """Block while gb-synapse is loading the model, so a prompt sent early waits
    instead of failing with HTTP 502.

    True: go ahead (running, or not loading at all: a stopped server gets the
    normal "model not loaded" message). False: cancelled while waiting.
    After `timeout` it returns True and lets the request report the real error."""
    if status_fn is None:
        from greenboost_cli.slash_commands.backend_cmds import llamacpp_server_status as status_fn
    t0 = time.monotonic()
    announced = False
    while True:
        try:
            st = status_fn(settings)
        except Exception:
            return True
        if st != "starting":
            return True
        if cancelled is not None and cancelled():
            return False
        if time.monotonic() - t0 > timeout:
            return True
        if not announced and on_wait:
            on_wait()
            announced = True
        sleep(poll)

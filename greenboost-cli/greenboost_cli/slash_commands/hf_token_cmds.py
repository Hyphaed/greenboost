"""/huggingface-token: set, check or remove the HuggingFace token.

    /huggingface-token <token>   save it (typed on screen; see below)
    /huggingface-token           show status; if none is set, ask with the input hidden
    /huggingface-token clear     remove it everywhere

Typing the token after the command works, but it is visible on screen and is a line in
the input history; the history entry is scrubbed straight after saving. The bare form
asks with the input hidden, which is the better habit. See hf_token.py for where it is
stored (owner-only files outside any repository)."""
from __future__ import annotations

from greenboost_cli import hf_token as H
from greenboost_cli.terminal.commands import register_command
from greenboost_cli.terminal.theme import emit_err, emit_info, emit_ok, emit_warn


def _save(token: str, *, typed_on_screen: bool) -> None:
    try:
        where = H.save_token(token)
    except ValueError as e:
        emit_err(str(e))
        emit_info(f"Create a free read token at {H.TOKENS_URL}, then:  {H.COMMAND} <your-token>")
        return
    except OSError as e:
        emit_err(f"Could not save the token: {e}")
        return
    removed = H.scrub_history(token) if typed_on_screen else 0
    emit_ok(f"HuggingFace token saved ({H.mask(token)}). Downloads use it from now on.")
    emit_info("Stored in " + ", ".join(where) + " (owner-only), and set for this session.")
    if typed_on_screen:
        emit_info("It's visible in this terminal's scrollback"
                  + (f"; removed from the input history ({removed} entr{'y' if removed == 1 else 'ies'})." if removed else ".")
                  + f"  Next time, type just {H.COMMAND} and enter it hidden.")


def cmd_huggingface_token(args: str, _session, _settings) -> bool:
    arg = args.strip()
    if arg.lower() in ("clear", "remove", "delete"):
        removed = H.clear_token()
        if removed:
            emit_ok("HuggingFace token removed: " + ", ".join(removed))
        else:
            emit_info("No stored HuggingFace token to remove.")
        return True
    if arg:
        _save(arg, typed_on_screen=True)
        return True
    for line in H.status_lines():
        emit_info(line)
    if H.current_token():
        return True
    try:
        import getpass
        tok = getpass.getpass("  ❯ HuggingFace token (input hidden, Enter to skip): ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return True
    if tok:
        _save(tok, typed_on_screen=False)
    else:
        emit_warn("No token entered; downloads stay anonymous.")
    return True


register_command("huggingface-token", cmd_huggingface_token,
                 "Set the HuggingFace token  (/huggingface-token <token>, or alone for a hidden prompt; "
                 "/huggingface-token clear)")

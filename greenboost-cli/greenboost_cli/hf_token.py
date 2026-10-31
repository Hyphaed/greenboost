"""HuggingFace token handling for greenboost-cli.

Two jobs:

1. ``/huggingface-token`` (slash_commands/hf_token_cmds.py) stores a token where every
   download path looks for it: the standard huggingface_hub file
   (``~/.cache/huggingface/token``, read by sentence-transformers, ``hf`` and friends) and
   gb-synapse's own ``hf_token``, both owner-only (0600), plus ``HF_TOKEN`` for this process.
   Nothing is written inside any repository.

2. The hub SERVER sends the warning "You are sending unauthenticated requests to the HF
   Hub. Please set a HF_TOKEN ..." as a response header (``X-HF-Warning``) and
   huggingface_hub logs it verbatim. That text says what to do in the hub's terms, not
   ours. ``install_warning_filter()`` swaps it for a message that says what happened,
   what it costs, what is NOT broken, and the exact command to type.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

#: Hub tokens start with "hf_". Format checks stay loose (prefix, no whitespace, a
#: plausible length) because the exact token shape is the hub's to change.
_TOKEN_RE = re.compile(r"^hf_\S{16,}$")

TOKENS_URL = "https://huggingface.co/settings/tokens"
COMMAND = "/huggingface-token"


def hf_token_path() -> Path:
    try:
        from huggingface_hub.constants import HF_TOKEN_PATH
        return Path(HF_TOKEN_PATH)
    except Exception:
        return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"


def current_token() -> str:
    """The token downloads would use right now ("" = none)."""
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    try:
        return hf_token_path().read_text().strip()
    except OSError:
        return ""


def mask(token: str) -> str:
    return f"{token[:3]}…{token[-4:]}" if len(token) > 10 else "…"


def valid(token: str) -> bool:
    return bool(_TOKEN_RE.match(token))


def _write_private(path: Path, text: str) -> None:
    """Atomic write, owner-only from the first byte (never world-readable, even briefly)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tok-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_token(token: str) -> list[str]:
    """Store `token` for every download path. Returns the places written."""
    token = token.strip()
    if not valid(token):
        raise ValueError("that doesn't look like a HuggingFace token (they start with hf_)")
    written: list[str] = []
    p = hf_token_path()
    _write_private(p, token + "\n")
    written.append(str(p).replace(str(Path.home()), "~"))
    try:                                # gb-synapse's own copy: `greenboost pull` reads this one
        import gb_synapse
        gb_synapse.login(token)
        written.append(str(gb_synapse.HF_TOKEN_FILE).replace(str(Path.home()), "~"))
    except Exception:
        pass
    os.environ["HF_TOKEN"] = token
    _reset_warning()
    return written


def clear_token() -> list[str]:
    removed: list[str] = []
    paths = [hf_token_path()]
    try:
        import gb_synapse
        paths.append(Path(gb_synapse.HF_TOKEN_FILE))
    except Exception:
        pass
    for p in paths:
        try:
            p.unlink()
            removed.append(str(p).replace(str(Path.home()), "~"))
        except OSError:
            pass
    os.environ.pop("HF_TOKEN", None)
    return removed


def scrub_history(token: str) -> int:
    """Drop any input-history entry containing `token`. Returns entries removed.

    The REPL keeps every submitted line in a plain-text history file, and
    `/huggingface-token hf_...` is such a line. prompt_toolkit's FileHistory stores
    entries as blocks: a "# <timestamp>" line, then "+<line>" lines, then a blank line."""
    try:
        from greenboost_cli.environment.settings import HISTORY_PATH
        text = HISTORY_PATH.read_text()
    except Exception:
        return 0
    blocks = text.split("\n\n")
    kept = [b for b in blocks if token not in b]
    removed = len(blocks) - len(kept)
    if removed:
        try:
            _write_private(HISTORY_PATH, "\n\n".join(kept))
        except Exception:
            return 0
    return removed


def status_lines() -> list[str]:
    tok = current_token()
    if not tok:
        return ["No HuggingFace token is set: downloads are anonymous and rate-limited.",
                f"Create a free read token at {TOKENS_URL}, then type:  {COMMAND} <your-token>"]
    src = "HF_TOKEN in the environment" if os.environ.get("HF_TOKEN", "").strip() else str(hf_token_path()).replace(str(Path.home()), "~")
    return [f"HuggingFace token set ({mask(tok)}), from {src}."]


# ── the replacement for the hub's warning ─────────────────────────────────────

_warned = False


def _reset_warning() -> None:
    global _warned
    _warned = False


#: The instruction the warning ends with. The REPL has the slash command; a plain
#: script (scripts/rag_reembed.py) doesn't, so it passes its own.
_action = f"{COMMAND} <your-token>"


def warning_text() -> str:
    """What happened, what it costs, what is NOT broken, the one action (in that order)."""
    return (
        "HuggingFace is treating this machine as anonymous. Model downloads and "
        "version checks run slower, and the hub can refuse them for a while when many "
        "anonymous requests come from the same network.\n"
        "  Nothing is broken and nothing is lost: models already on disk load as normal.\n"
        f"  To fix it, type:  {_action}\n"
        f"  (a free read token from {TOKENS_URL}; in greenboost-cli, type just {COMMAND} "
        "to be asked with the input hidden. Saved with owner-only permissions, never in a repo.)"
    )


class _HubWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        global _warned
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if "unauthenticated requests" not in msg or "HF_TOKEN" not in msg:
            return True
        if _warned or current_token():
            return False                    # said once; or a token exists (stale header): drop it
        _warned = True
        try:
            from greenboost_cli.terminal.theme import emit_warn
            emit_warn(warning_text())
        except Exception:
            import sys
            print(warning_text(), file=sys.stderr)
        return False                        # the hub's own wording is replaced, not repeated


_installed = False


def install_warning_filter(action: str | None = None) -> None:
    """Idempotent. Safe before huggingface_hub is imported. `action` overrides the
    instruction for callers with no slash command (a script)."""
    global _installed, _action
    if action:
        _action = action
    if _installed:
        return
    _installed = True
    f = _HubWarningFilter()
    for name in ("huggingface_hub", "huggingface_hub.utils._http", "huggingface_hub.utils"):
        logging.getLogger(name).addFilter(f)

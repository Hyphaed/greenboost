"""Output styles: a markdown file whose body is appended to the system prompt.

Search order: project `.greenboost/output-styles`, `~/.greenboost_cli/output-styles`, then the
styles shipped with greenboost-cli (`prompts/styles`). `settings["output_style"]` names one."""
from __future__ import annotations

from pathlib import Path

from greenboost_cli.custom_commands import _split_front

_SHIPPED = Path(__file__).parent / "prompts" / "styles"


def _dirs() -> "list[Path]":
    return [Path.cwd() / ".greenboost" / "output-styles",
            Path.home() / ".greenboost_cli" / "output-styles", _SHIPPED]


def available() -> "dict[str, tuple[Path, str]]":
    out: dict[str, tuple[Path, str]] = {}
    for d in _dirs():
        if d.is_dir():
            for f in sorted(d.glob("*.md")):
                if f.stem not in out:
                    try:
                        out[f.stem] = (f, _split_front(f.read_text(errors="replace"))[0].get("description", ""))
                    except OSError:
                        pass
    return out


def style_text(settings: dict) -> str:
    name = settings.get("output_style")
    hit = available().get(name) if name else None
    if not hit:
        return ""
    try:
        body = _split_front(hit[0].read_text(errors="replace"))[1].strip()
    except OSError:
        return ""
    return f"\n\n## Output style: {name}\n{body}\n"

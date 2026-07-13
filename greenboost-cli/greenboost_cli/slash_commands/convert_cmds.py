"""Universal document/URL conversion: /convert <file|url> → Markdown + auto-RAG."""
from __future__ import annotations

from pathlib import Path

from greenboost_cli.terminal.commands import register_command


def _convert(args: str, session, settings: dict) -> None:
    parts = args.strip().split()
    if not parts:
        print("  Usage: /convert <file|url> [--output out.md] [--preview] [--no-rag] [--project name]")
        print()
        print("  Formats: PDF, DOCX, PPTX, XLSX, HTML, CSV, JSON, XML, EPUB, ZIP,")
        print("           images, audio, Jupyter notebooks, YouTube/Wikipedia URLs")
        print()
        print("  Requires: pip install markitdown")
        return

    source   = parts[0]
    output   = None
    preview  = False
    feed_rag = True
    project  = settings.get("active_project")

    i = 1
    while i < len(parts):
        if parts[i] == "--output" and i + 1 < len(parts):
            output = Path(parts[i + 1]).expanduser(); i += 2
        elif parts[i] == "--preview":
            preview = True; i += 1
        elif parts[i] == "--no-rag":
            feed_rag = False; i += 1
        elif parts[i] == "--project" and i + 1 < len(parts):
            project = parts[i + 1]; i += 2
        else:
            i += 1

    is_url = source.startswith(("http://", "https://"))
    if not is_url:
        src_path = Path(source).expanduser()
        if not src_path.exists():
            print(f"  ✗  File not found: {src_path}")
            return
        source = str(src_path)

    try:
        from greenboost_cli.converters.markitdown_adapter import convert, convert_and_save
    except ImportError as e:
        print(f"  ✗  {e}")
        return

    print(f"  Converting: {source} …")
    try:
        if preview:
            md    = convert(source, feed_rag=False, project=project)
            lines = md.splitlines()
            for line in lines[:60]:
                print(line)
            if len(lines) > 60:
                print(f"\n  … {len(lines) - 60} more lines  (omit --preview to save)")
        else:
            out  = convert_and_save(source, output=output, feed_rag=feed_rag, project=project)
            size = out.stat().st_size
            rag_note = "  · indexed into RAG" if feed_rag else ""
            print(f"  ✓  {out}  ({size:,} bytes){rag_note}")
    except Exception as e:
        print(f"  ✗  Conversion failed: {e}")


register_command("convert", _convert, "Convert any file/URL to Markdown  (/convert <file|url>)")

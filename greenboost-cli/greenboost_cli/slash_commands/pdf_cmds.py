"""PDF slash commands: /pdf2md — markitdown primary, pymupdf fallback."""
from __future__ import annotations

from pathlib import Path

from greenboost_cli.terminal.commands import register_command


def _pdf2md(args: str, session, settings: dict) -> None:
    parts = args.strip().split()
    if not parts:
        print("  Usage: /pdf2md <file> [--output out.md] [--pages 1-5] [--preview] [--page-breaks] [--no-rag]")
        print()
        print("  Preferred: markitdown (pip install markitdown)")
        print("  Fallback:  pymupdf   (pip install pymupdf) — required for --pages / --page-breaks")
        return

    src_path    = Path(parts[0]).expanduser()
    output_path = None
    pages       = None
    page_breaks = False
    preview     = False
    feed_rag    = True
    project     = settings.get("active_project")

    i = 1
    while i < len(parts):
        if parts[i] == "--output" and i + 1 < len(parts):
            output_path = Path(parts[i + 1]).expanduser(); i += 2
        elif parts[i] == "--pages" and i + 1 < len(parts):
            pages = parts[i + 1]; i += 2
        elif parts[i] == "--page-breaks":
            page_breaks = True; i += 1
        elif parts[i] == "--preview":
            preview = True; i += 1
        elif parts[i] == "--no-rag":
            feed_rag = False; i += 1
        elif parts[i] == "--project" and i + 1 < len(parts):
            project = parts[i + 1]; i += 2
        else:
            i += 1

    if not src_path.exists():
        print(f"  ✗  File not found: {src_path}")
        return

    md = _do_convert(src_path, pages, page_breaks, preview, feed_rag, project)
    if md is None:
        return

    if preview:
        lines = md.splitlines()
        for line in lines[:60]:
            print(line)
        if len(lines) > 60:
            print(f"\n  … {len(lines) - 60} more lines  (omit --preview to convert full file)")
    else:
        if output_path is None:
            output_path = src_path.with_suffix(".md")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md, encoding="utf-8")
        rag_note = "  · indexed into RAG" if feed_rag else ""
        print(f"  ✓  {output_path}  ({output_path.stat().st_size:,} bytes){rag_note}")


def _do_convert(
    src_path: Path,
    pages: str | None,
    page_breaks: bool,
    preview: bool,
    feed_rag: bool,
    project: str | None,
) -> str | None:
    """Try markitdown; fall back to pymupdf for page-range / page-break options."""
    need_pymupdf = pages is not None or page_breaks

    if not need_pymupdf:
        try:
            from greenboost_cli.converters.markitdown_adapter import convert
            print(f"  Converting {src_path.name} …")
            return convert(str(src_path), feed_rag=feed_rag and not preview, project=project)
        except ImportError:
            pass
        except Exception as e:
            print(f"  ⚠  markitdown: {e} — falling back to pymupdf …")

    try:
        from greenboost_cli.pdf.pdf2md import convert_pdf
        if not preview:
            print(f"  Converting {src_path.name} (pymupdf) …")
        md = convert_pdf(src_path, pages=pages)
        if feed_rag and not preview and md.strip():
            try:
                from greenboost_cli.rag.engine import index_text
                index_text(md, source_name=str(src_path), project=project)
            except Exception:
                pass
        return md
    except ImportError:
        print("  ✗  No converter available. Install one of:")
        print("       pip install markitdown")
        print("       pip install pymupdf")
        return None


register_command("pdf2md", _pdf2md, "Convert PDF/DOCX/PPTX/HTML to Markdown  (/pdf2md <file>)")

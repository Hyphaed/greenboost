"""MarkItDown adapter — universal document/URL → Markdown with auto-RAG feeding.

Supports: PDF, DOCX, PPTX, XLSX, XLS, HTML, CSV, JSON, XML, EPUB, MSG, ZIP,
          images (JPEG/PNG), audio (WAV/MP3), Jupyter notebooks, URLs
          (YouTube, Wikipedia, RSS, Bing).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".html", ".htm", ".csv", ".json", ".xml", ".epub", ".msg", ".zip",
    ".md", ".txt", ".rst", ".ipynb",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
    ".wav", ".mp3",
}


def _get_mid():
    try:
        from markitdown import MarkItDown
        return MarkItDown()
    except ImportError:
        raise ImportError(
            "markitdown not installed.\n"
            "Run: pip install markitdown\n"
            "For all formats: pip install 'markitdown[all]'"
        )


def convert(
    source: str | Path,
    feed_rag: bool = True,
    project: Optional[str] = None,
) -> str:
    """Convert file path or URL to Markdown. Optionally feeds result into RAG."""
    mid = _get_mid()
    source_str = str(source)

    if source_str.startswith(("http://", "https://")):
        result = mid.convert_uri(source_str)
    else:
        result = mid.convert_local(source_str)

    md = result.text_content or ""

    if feed_rag and md.strip():
        _feed_to_rag(md, source_str, project)

    return md


def convert_and_save(
    source: str | Path,
    output: Optional[Path] = None,
    feed_rag: bool = True,
    project: Optional[str] = None,
) -> Path:
    """Convert source → Markdown and save to disk. Returns the saved path."""
    source_str = str(source)
    md = convert(source_str, feed_rag=feed_rag, project=project)

    if output is None:
        if source_str.startswith(("http://", "https://")):
            output = Path("converted.md")
        else:
            output = Path(source_str).with_suffix(".md")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md, encoding="utf-8")
    return output


def _feed_to_rag(md: str, source: str, project: Optional[str]) -> None:
    try:
        from greenboost_cli.rag.engine import index_text, register_web_source
        index_text(md, source_name=source, project=project)
        if source.startswith(("http://", "https://")):
            register_web_source(source, project)
    except Exception:
        pass

"""
Offline PDF → Markdown converter.

Uses PyMuPDF (pymupdf) for extraction with font-based structure detection.
No API calls, no internet required.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any


_FLAG_SUPERSCRIPT = 1
_FLAG_ITALIC      = 2
_FLAG_SERIF       = 4
_FLAG_MONOSPACE   = 8
_FLAG_BOLD        = 16


def _parse_page_range(spec: str | None, total: int) -> list[int]:
    if not spec:
        return list(range(total))
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            start = max(0, int(a.strip()) - 1)
            end   = min(total - 1, int(b.strip()) - 1)
            indices.extend(range(start, end + 1))
        else:
            idx = int(part.strip()) - 1
            if 0 <= idx < total:
                indices.append(idx)
    return sorted(set(indices))


def _collect_font_stats(doc: Any, page_indices: list[int]) -> dict:
    size_counts: Counter = Counter()

    for i in page_indices:
        page   = doc[i]
        blocks = page.get_text("dict", flags=0)["blocks"]
        for blk in blocks:
            if blk.get("type") != 0:
                continue
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    txt  = span["text"].strip()
                    if len(txt) < 3:
                        continue
                    size = round(span["size"], 1)
                    size_counts[size] += len(txt)

    if not size_counts:
        return {"body_size": 11.0, "heading_map": {}}

    body_size = size_counts.most_common(1)[0][0]
    larger    = sorted([s for s in size_counts if s > body_size * 1.1], reverse=True)
    level_map = {
        "h1": larger[0] if len(larger) > 0 else None,
        "h2": larger[1] if len(larger) > 1 else None,
        "h3": larger[2] if len(larger) > 2 else None,
    }
    heading_map = {v: k for k, v in level_map.items() if v is not None}
    return {"body_size": body_size, "heading_map": heading_map}


def _render_span(span: dict, body_size: float) -> str:
    text  = span["text"]
    if not text.strip():
        return text
    flags = span.get("flags", 0)

    is_mono   = bool(flags & _FLAG_MONOSPACE)
    is_bold   = bool(flags & _FLAG_BOLD)
    is_italic = bool(flags & _FLAG_ITALIC)
    is_super  = bool(flags & _FLAG_SUPERSCRIPT)

    if is_super:
        return ""
    if is_mono:
        return f"`{text}`"
    if is_bold and is_italic:
        text = f"***{text.strip()}***"
    elif is_bold:
        text = f"**{text.strip()}**"
    elif is_italic:
        text = f"*{text.strip()}*"

    return text


def _try_pdfplumber_table(path: Path, page_num: int) -> list[list[str]] | None:
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            page   = pdf.pages[page_num]
            tables = page.extract_tables()
            if tables:
                return tables[0]
    except Exception:
        pass
    return None


def _render_md_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    rows   = [[c or "" for c in row] for row in rows]
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    lines  = []
    for idx, row in enumerate(rows):
        cells = " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))
        lines.append(f"| {cells} |")
        if idx == 0:
            sep = " | ".join("-" * widths[i] for i in range(len(row)))
            lines.append(f"| {sep} |")
    return "\n".join(lines)


_BULLET_RE   = re.compile(r"^[•‣◦⁃∙\-\*]\s+")
_NUM_LIST_RE = re.compile(r"^\d+[\.\)]\s+")


def _classify_block(blk: dict, body_size: float, heading_map: dict) -> dict:
    lines = blk.get("lines", [])
    if not lines:
        return {"type": "empty"}

    all_spans = [span for line in lines for span in line.get("spans", [])]
    if not all_spans:
        return {"type": "empty"}

    sizes    = [round(s["size"], 1) for s in all_spans if s["text"].strip()]
    if not sizes:
        return {"type": "empty"}
    dom_size = Counter(sizes).most_common(1)[0][0]

    if dom_size in heading_map:
        text = " ".join(s["text"] for s in all_spans).strip()
        return {"type": "heading", "level": heading_map[dom_size], "text": text}

    mono_count = sum(1 for s in all_spans if s.get("flags", 0) & _FLAG_MONOSPACE)
    if mono_count > len(all_spans) * 0.7 and len(all_spans) > 1:
        text = "\n".join(
            " ".join(s["text"] for s in line.get("spans", []))
            for line in lines
        )
        return {"type": "code", "text": text}

    para_lines = []
    for line in lines:
        parts = [_render_span(s, body_size) for s in line.get("spans", [])]
        para_lines.append("".join(parts))
    text = " ".join(para_lines).strip()

    if not text:
        return {"type": "empty"}

    if _BULLET_RE.match(text):
        return {"type": "bullet", "text": _BULLET_RE.sub("", text, count=1).strip()}
    if _NUM_LIST_RE.match(text):
        m = _NUM_LIST_RE.match(text)
        return {"type": "numbered", "text": text[m.end():].strip()}

    return {"type": "paragraph", "text": text}


def _extract_page(page: Any, body_size: float, heading_map: dict) -> list[dict]:
    raw    = page.get_text("dict", flags=0)
    chunks = []
    for blk in raw.get("blocks", []):
        if blk.get("type") == 1:
            chunks.append({"type": "image"})
            continue
        classified = _classify_block(blk, body_size, heading_map)
        if classified["type"] != "empty":
            chunks.append(classified)
    return chunks


def _render_markdown(chunks: list[dict], page_breaks: bool = False) -> str:
    lines: list[str] = []
    prev_type = None

    for chunk in chunks:
        t = chunk["type"]

        if t == "empty":
            continue
        if t == "page_break":
            if page_breaks:
                lines.append("\n---\n")
            prev_type = t
            continue
        if t == "image":
            prev_type = t
            continue

        # Close previous code block before emitting non-code content
        if prev_type == "code" and t != "code":
            lines.append("```")
            lines.append("")

        if t == "heading":
            lvl    = chunk["level"]
            prefix = {"h1": "#", "h2": "##", "h3": "###"}.get(lvl, "####")
            if prev_type and prev_type not in ("heading",):
                lines.append("")
            lines.append(f"{prefix} {chunk['text']}")
            lines.append("")

        elif t == "code":
            if prev_type != "code":
                lines.append("")
                lines.append("```")
            lines.append(chunk["text"])

        elif t in ("bullet", "numbered"):
            prefix = "-" if t == "bullet" else "1."
            lines.append(f"{prefix} {chunk['text']}")

        elif t == "paragraph":
            if prev_type and prev_type not in ("paragraph",):
                lines.append("")
            lines.append(chunk["text"])

        elif t == "table":
            lines.append("")
            lines.append(chunk["text"])
            lines.append("")

        prev_type = t

    if prev_type == "code":
        lines.append("```")

    result = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return result.strip() + "\n"


def convert_pdf(
    path: str | Path,
    pages: str | None = None,
    page_breaks: bool = False,
) -> str:
    """Convert a PDF file to Markdown text."""
    try:
        import fitz
    except ImportError:
        raise ImportError("pymupdf not installed. Run: pip install pymupdf")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    doc          = fitz.open(str(path))
    page_indices = _parse_page_range(pages, len(doc))
    font_stats   = _collect_font_stats(doc, page_indices)
    body_size    = font_stats["body_size"]
    heading_map  = font_stats["heading_map"]

    chunks: list[dict] = []
    for i in page_indices:
        page_chunks = _extract_page(doc[i], body_size, heading_map)
        chunks.extend(page_chunks)
        if page_breaks:
            chunks.append({"type": "page_break"})

    return _render_markdown(chunks, page_breaks=page_breaks)


def convert_and_save(
    pdf_path: str | Path,
    output_path: str | Path | None = None,
    pages: str | None = None,
    page_breaks: bool = False,
) -> Path:
    """Convert PDF and write .md file. Returns the output path."""
    pdf_path = Path(pdf_path)
    if output_path is None:
        output_path = pdf_path.with_suffix(".md")
    output_path = Path(output_path)

    md = convert_pdf(pdf_path, pages=pages, page_breaks=page_breaks)
    output_path.write_text(md, encoding="utf-8")
    return output_path

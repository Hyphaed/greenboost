"""Playwright-based UI screenshot capture.

gb has no vision capability of its own (its models are text-only GGUF
checkpoints) — this tool exists so an agent can at least CAPTURE what a page
looks like (e.g. a Vite/Tauri dev server) and hand the resulting file to a
vision-capable reviewer (Claude, qwen3-vl, a human), closing half of the
"no visual QA" gap documented in the gb-app-builder skill. Judging the image
still requires a vision-capable reviewer; this only captures it.

playwright ships as a transitive dependency of scrapling[fetchers] (already
required by scrapling_utils.py's StealthyFetcher), but its browser binaries
are a separate download (`playwright install chromium`) — reported clearly
here rather than failing with an opaque traceback.
"""
from __future__ import annotations

from pathlib import Path


def capture_screenshot(
    url: str,
    output_path: str,
    width: int = 1280,
    height: int = 800,
    timeout: int = 30,
    full_page: bool = False,
) -> str:
    """Load `url` in headless Chromium and save a PNG to `output_path`.

    Returns a short confirmation string (path + size) on success, or an
    "Error: ..." string on failure — never raises, matching this module's
    sibling handlers' convention of returning errors as text the model can
    see and react to.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return (
            "Error: playwright is not installed. It ships with "
            "scrapling[fetchers] (already a greenboost-cli dependency) but "
            "needs its browser binary downloaded once: "
            "`python3 -m playwright install chromium`"
        )

    out = Path(output_path)
    if not out.is_absolute():
        return f"Error: output_path must be absolute, got {output_path!r}"
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": width, "height": height})
                page.goto(url, timeout=timeout * 1000, wait_until="load")
                page.screenshot(path=str(out), full_page=full_page)
            finally:
                browser.close()
    except Exception as e:
        return f"Error: could not capture {url}: {e}"

    size_kb = out.stat().st_size / 1024
    return f"Screenshot saved: {out} ({size_kb:.1f} KB, {width}x{height})"

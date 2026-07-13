"""Scrapling-based HTTP fetch utilities shared across greenboost-cli.

Uses Fetcher (fast, stealth headers) for most pages.
Falls back to StealthyFetcher (headless Playwright browser) when blocked.
"""
from __future__ import annotations

import logging

# Statuses that indicate anti-bot blocking — trigger StealthyFetcher retry
_BLOCKED_STATUSES = {403, 429, 503}
_MAX_CHARS = 30_000


def _silence() -> None:
    logging.disable(logging.INFO)


def _unsilence() -> None:
    logging.disable(logging.NOTSET)


def fetch_url(
    url: str,
    timeout: int = 30,
    use_stealth: bool | None = None,
) -> str:
    """Fetch a URL and return readable plain text.

    Args:
        url:         Target URL (http/https only).
        timeout:     Seconds for Fetcher; milliseconds ×1000 for StealthyFetcher.
        use_stealth: True → skip to StealthyFetcher directly.
                     None (default) → try Fetcher first, auto-escalate if blocked.
                     False → Fetcher only, no escalation.
    """
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Error: unsupported scheme {parsed.scheme!r}"

    if use_stealth is not True:
        result = _fetch_with_fetcher(url, timeout)
        if result is not None:
            return result

    if use_stealth is not False:
        result = _fetch_with_stealthy(url, timeout)
        if result is not None:
            return result

    return f"Error: could not fetch {url}"


def _fetch_with_fetcher(url: str, timeout: int) -> str | None:
    """Try Fetcher.get — returns None on block or error to allow escalation."""
    try:
        from scrapling.fetchers import Fetcher
        _silence()
        page = Fetcher.get(url, timeout=timeout, stealthy_headers=True, follow_redirects=True)
        _unsilence()

        if page.status in _BLOCKED_STATUSES:
            return None   # signal caller to try StealthyFetcher

        ct = page.headers.get("content-type", "")
        if "json" in ct:
            return page.body.decode("utf-8", errors="replace")[:_MAX_CHARS]
        return _page_text(page)
    except Exception:
        _unsilence()
        return None


def _fetch_with_stealthy(url: str, timeout: int) -> str | None:
    """Try StealthyFetcher (headless Playwright) — for anti-bot-protected pages."""
    try:
        from scrapling.fetchers import StealthyFetcher
        _silence()
        page = StealthyFetcher.fetch(
            url,
            headless=True,
            timeout=timeout * 1000,
            network_idle=True,
            disable_resources=True,   # skip images/fonts/media for speed
        )
        _unsilence()
        return _page_text(page)
    except Exception:
        _unsilence()
        return None


def _page_text(page) -> str:
    """Extract readable text from a Scrapling Response."""
    # Prefer native get_all_text() — strips HTML, decodes entities, compacts whitespace
    try:
        text = page.get_all_text()
        if text and text.strip():
            return text.strip()[:_MAX_CHARS]
    except Exception:
        pass

    # Fallback: raw body decode
    try:
        return page.body.decode("utf-8", errors="replace")[:_MAX_CHARS]
    except Exception:
        return ""


def search_ddg(query: str, max_results: int = 8, timeout: int = 20) -> list[dict]:
    """Search DuckDuckGo and return list of {title, url, snippet} dicts.

    Uses Fetcher (stealth headers) to fetch the DDG HTML results page.
    """
    from urllib.parse import quote_plus

    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}&kl=wt-wt"

    try:
        from scrapling.fetchers import Fetcher
        _silence()
        page = Fetcher.get(
            search_url,
            timeout=timeout,
            stealthy_headers=True,
            follow_redirects=True,
        )
        _unsilence()
    except Exception:
        _unsilence()
        return []

    results = []
    try:
        for result in page.css(".result"):
            title_el  = result.css(".result__title a")
            snippet_el = result.css(".result__snippet")
            url_el    = result.css(".result__url")

            title   = title_el[0].get_all_text().strip()   if title_el   else ""
            snippet = snippet_el[0].get_all_text().strip() if snippet_el else ""
            raw_url = url_el[0].get_all_text().strip()     if url_el     else ""

            href = title_el[0].attrib.get("href", "") if title_el else ""
            real_url = _resolve_ddg_url(href) or raw_url

            if title and real_url:
                results.append({"title": title, "url": real_url, "snippet": snippet})
                if len(results) >= max_results:
                    break
    except Exception:
        pass

    return results


def _resolve_ddg_url(href: str) -> str:
    """Unwrap DuckDuckGo redirect (/l/?uddg=...) to the actual destination URL."""
    if not href:
        return ""
    if "//duckduckgo.com/l/" in href or href.startswith("//duckduckgo.com"):
        try:
            from urllib.parse import urlparse, parse_qs, unquote
            full = href if href.startswith("http") else "https:" + href
            qs = parse_qs(urlparse(full).query)
            encoded = qs.get("uddg", [""])[0]
            return unquote(encoded) if encoded else href
        except Exception:
            return href
    return href

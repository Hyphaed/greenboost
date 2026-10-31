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
    # CRITICAL, not INFO: scrapling/playwright log retries at WARNING and ERROR
    # ("Attempt 1 failed ...") straight through the TUI during a bot-wall retry.
    logging.disable(logging.CRITICAL)


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
        # A Cloudflare / bot-wall page is a successful HTTP exchange for the
        # browser; without this check its "Just a moment..." body came back as
        # the page's text and the model read a challenge as content.
        if getattr(page, "status", 200) in _BLOCKED_STATUSES:
            return None
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


#: Phrases on a search engine's anti-bot interstitial. DuckDuckGo answers HTTP
#: 202 (not 403) with a "select all squares containing a duck" page, which
#: used to parse as zero results and surface as "No results found".
_CHALLENGE_MARKS = ("bots use duckduckgo too", "anomaly-modal", "select all squares",
                    "unusual traffic", "are you a robot")


def _ddg_get(url: str, timeout: int):
    from scrapling.fetchers import Fetcher
    _silence()
    try:
        return Fetcher.get(url, timeout=timeout, stealthy_headers=True, follow_redirects=True)
    finally:
        _unsilence()


def _is_challenge(page) -> bool:
    if getattr(page, "status", 200) != 200:
        return True
    try:
        text = page.get_all_text().lower()
    except Exception:
        return False
    return any(m in text for m in _CHALLENGE_MARKS)


def _parse_lite(page, max_results: int) -> list[dict]:
    """lite.duckduckgo.com: a.result-link titles, td.result-snippet in order."""
    links = page.css("a.result-link")
    snippets = page.css("td.result-snippet")
    out = []
    for i, a in enumerate(links):
        title = a.get_all_text().strip()
        url = _resolve_ddg_url(a.attrib.get("href", "")) or a.attrib.get("href", "")
        if url.startswith("//"):
            url = "https:" + url
        snippet = snippets[i].get_all_text().strip() if i < len(snippets) else ""
        if title and url.startswith("http"):
            out.append({"title": title, "url": url, "snippet": snippet})
            if len(out) >= max_results:
                break
    return out


def _parse_html(page, max_results: int) -> list[dict]:
    results = []
    for result in page.css(".result"):
        title_el = result.css(".result__title a")
        snippet_el = result.css(".result__snippet")
        url_el = result.css(".result__url")

        title = title_el[0].get_all_text().strip() if title_el else ""
        snippet = snippet_el[0].get_all_text().strip() if snippet_el else ""
        raw_url = url_el[0].get_all_text().strip() if url_el else ""

        href = title_el[0].attrib.get("href", "") if title_el else ""
        real_url = _resolve_ddg_url(href) or raw_url

        if title and real_url:
            results.append({"title": title, "url": real_url, "snippet": snippet})
            if len(results) >= max_results:
                break
    return results


def search_ddg_ex(query: str, max_results: int = 8, timeout: int = 20) -> tuple[list[dict], str]:
    """Search DuckDuckGo. Returns (hits, reason).

    reason is "" when a backend answered (hits may then be legitimately empty)
    and otherwise says WHY nothing came back, so the caller can tell the model
    "blocked" apart from "no results" instead of reporting both as empty.

    Ladder: lite (plain table markup, tolerated by DDG) -> html (richer, but
    served a bot challenge to this client in live tests on 2026-10-30).
    """
    from urllib.parse import quote_plus
    q = quote_plus(query)
    backends = (
        ("lite", f"https://lite.duckduckgo.com/lite/?q={q}&kl=wt-wt", _parse_lite),
        ("html", f"https://html.duckduckgo.com/html/?q={q}&kl=wt-wt", _parse_html),
    )
    problems: list[str] = []
    for name, url, parse in backends:
        try:
            page = _ddg_get(url, timeout)
        except Exception as e:
            problems.append(f"{name}: {type(e).__name__}")
            continue
        if _is_challenge(page):
            problems.append(f"{name}: bot challenge (HTTP {getattr(page, 'status', '?')})")
            continue
        try:
            hits = parse(page, max_results)
        except Exception as e:
            problems.append(f"{name}: parse error {type(e).__name__}")
            continue
        if hits:
            return hits, ""
        # 200, no challenge, parsed zero: a genuinely empty answer.
        return [], ""
    return [], "; ".join(problems) or "no backend answered"


def search_ddg(query: str, max_results: int = 8, timeout: int = 20) -> list[dict]:
    """Search DuckDuckGo and return list of {title, url, snippet} dicts."""
    return search_ddg_ex(query, max_results, timeout)[0]


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

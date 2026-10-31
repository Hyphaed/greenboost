"""WebSearch / WebFetch: a refusal is reported as a refusal, never as "no results"."""
from __future__ import annotations

from greenboost_cli.instruments import handlers as H
from greenboost_cli.instruments import safe_fetch as SF
from greenboost_cli.instruments import scrapling_utils as SU
from scrapling.parser import Selector

_LITE = """<html><body><table>
<tr><td><a class="result-link" href="https://example.org/a">First hit</a></td></tr>
<tr><td class="result-snippet">About the first hit</td></tr>
<tr><td><a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fb">Second</a></td></tr>
<tr><td class="result-snippet">About the second</td></tr>
</table></body></html>"""

_CHALLENGE = ("<html><body>Unfortunately, bots use DuckDuckGo too. Please complete the "
              "following challenge. Select all squares containing a duck</body></html>")


class _Page:
    def __init__(self, html: str, status: int = 200):
        self._sel = Selector(html)
        self.status = status

    def css(self, q):
        return self._sel.css(q)

    def get_all_text(self):
        return self._sel.get_all_text()


def test_lite_markup_is_parsed_with_snippets_and_unwrapped_urls():
    hits = SU._parse_lite(_Page(_LITE), 8)
    assert [h["title"] for h in hits] == ["First hit", "Second"]
    assert hits[1]["url"] == "https://example.org/b"
    assert hits[0]["snippet"] == "About the first hit"


def test_bot_challenge_is_detected_even_on_http_202():
    assert SU._is_challenge(_Page(_CHALLENGE, status=202))
    assert SU._is_challenge(_Page(_CHALLENGE, status=200))
    assert not SU._is_challenge(_Page(_LITE))


def test_search_falls_back_to_html_when_lite_is_challenged(monkeypatch):
    html = ('<div class="result"><h2 class="result__title"><a href="https://e.org/x">T</a></h2>'
            '<a class="result__snippet">S</a></div>')
    pages = iter([_Page(_CHALLENGE, 202), _Page(html)])
    monkeypatch.setattr(SU, "_ddg_get", lambda url, timeout: next(pages))
    hits, reason = SU.search_ddg_ex("q")
    assert reason == "" and hits[0]["url"] == "https://e.org/x"


def test_search_blocked_everywhere_says_blocked_not_empty(monkeypatch):
    monkeypatch.setattr(SU, "_ddg_get", lambda url, timeout: _Page(_CHALLENGE, 202))
    out = H.handle_web_query("anything")
    assert out.startswith("Error:") and "bot challenge" in out
    assert out != "No results found"


def test_genuinely_empty_search_is_still_no_results(monkeypatch):
    monkeypatch.setattr(SU, "_ddg_get", lambda url, timeout: _Page("<html><body>nothing</body></html>"))
    assert H.handle_web_query("zzzz") == "No results found"


def test_block_hint_names_the_cloudflare_wall():
    body = "<html><head><title>Just a moment...</title></head><body>cf_chl_opt</body></html>"
    hint = SF._block_hint(body)
    assert "Just a moment" in hint and "bot check" in hint
    assert SF._block_hint("<html><body>plain</body></html>") == ""


def test_user_agent_identifies_the_project_and_is_overridable(monkeypatch):
    assert "greenboost-cli" in SF._user_agent() and "https://" in SF._user_agent()
    monkeypatch.setenv("GB_WEBFETCH_UA", "custom/1")
    assert SF._user_agent() == "custom/1"


def test_403_triggers_browser_fallback_by_default_and_can_be_disabled(monkeypatch):
    monkeypatch.setattr(SF, "fetch", lambda url: "Error: https://x returned HTTP 403")
    calls = []
    monkeypatch.setattr(SU, "fetch_url", lambda url, use_stealth=None: calls.append(url) or "real page")
    monkeypatch.delenv("GB_WEBFETCH_BROWSER", raising=False)
    assert H.handle_fetch_url("https://x") == "real page"
    monkeypatch.setenv("GB_WEBFETCH_BROWSER", "0")
    assert "HTTP 403" in H.handle_fetch_url("https://x") and len(calls) == 1


def test_refused_browser_retry_tells_the_model_to_stop_retrying(monkeypatch):
    monkeypatch.setattr(SF, "fetch", lambda url: "Error: https://x returned HTTP 403")
    monkeypatch.setattr(SU, "fetch_url", lambda url, use_stealth=None: "Error: could not fetch https://x")
    out = H.handle_fetch_url("https://x")
    assert "HTTP 403" in out and "Do not retry" in out


def test_prompt_focuses_a_long_page_on_what_was_asked(monkeypatch):
    filler = "\n\n".join(f"Unrelated paragraph number {i} about gardening and weather." for i in range(300))
    page = "# Title\n\nIntro text.\n\n" + filler + "\n\nThe MTP draft depth defaults to 4 tokens.\n\n" + filler
    monkeypatch.setattr(SF, "fetch", lambda url: page)
    out = H.handle_fetch_url("https://x", prompt="What is the MTP draft depth?")
    assert "defaults to 4 tokens" in out and "focused on your prompt" in out
    assert len(out) < len(page) / 4


def test_short_page_and_no_prompt_are_returned_untouched(monkeypatch):
    monkeypatch.setattr(SF, "fetch", lambda url: "short page")
    assert H.handle_fetch_url("https://x", prompt="anything") == "short page"
    assert H.handle_fetch_url("https://x") == "short page"


def test_long_passing_test_run_collapses_but_failures_never_do():
    ok = "\n".join(f"tests/test_x.py::test_{i} PASSED" for i in range(200)) + "\n===== 200 passed in 3.1s ====="
    out = H._collapse_passing_run(ok, 0)
    assert "200 passed" in out and "collapsed" in out and len(out) < 400
    assert H._collapse_passing_run(ok, 1) == ok                       # non-zero exit: whole
    mixed = ok.replace("200 passed", "198 passed, 2 failed")
    assert H._collapse_passing_run(mixed, 0) == mixed                 # a failure word: whole
    short = "a\nb\n1 passed"
    assert H._collapse_passing_run(short, 0) == short

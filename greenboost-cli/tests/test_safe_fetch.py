"""WebFetch's client-level guard: DNS rebinding, redirect hops, size caps, content
types, credential URLs, GreenBoost's own control ports. No real network."""
import gzip

import httpx

from greenboost_cli.instruments import handlers, safe_fetch

PUBLIC = "93.184.216.34"


def _client(handler):
    return lambda: httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def _resolver(table, calls=None):
    def resolve(host):
        if calls is not None:
            calls.append(host)
        return table[host]
    return resolve


def test_connects_to_the_pinned_address_and_resolves_once():
    seen, calls = [], []

    def handler(req):
        seen.append((req.url.host, req.headers["host"]))
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="hello")
    out = safe_fetch.fetch("https://docs.example/page", resolver=_resolver({"docs.example": [PUBLIC]}, calls),
                           client_factory=_client(handler))
    assert out == "hello"
    assert seen == [(PUBLIC, "docs.example")]          # IP connection, real Host header
    assert calls == ["docs.example"]                    # one lookup, nothing to rebind


def test_redirect_into_the_lan_is_blocked_on_the_second_hop():
    def handler(req):
        return httpx.Response(302, headers={"location": "http://router.evil/admin"})
    out = safe_fetch.fetch("http://good.example/", client_factory=_client(handler),
                           resolver=_resolver({"good.example": [PUBLIC], "router.evil": ["192.168.1.1"]}))
    assert out.startswith("Error:") and "192.168.1.1" in out


def test_redirect_to_cloud_metadata_is_blocked():
    def handler(req):
        return httpx.Response(307, headers={"location": "http://169.254.169.254/latest/meta-data/"})
    out = safe_fetch.fetch("http://good.example/", client_factory=_client(handler),
                           resolver=_resolver({"good.example": [PUBLIC]}))
    assert out.startswith("Error:") and "link-local" in out


def test_every_address_a_name_resolves_to_is_judged():
    out = safe_fetch.fetch("http://mixed.example/", client_factory=_client(lambda r: httpx.Response(200, text="x")),
                           resolver=_resolver({"mixed.example": [PUBLIC, "10.0.0.5"]}))
    assert out.startswith("Error:") and "10.0.0.5" in out


def test_redirect_chain_is_capped():
    def handler(req):
        return httpx.Response(302, headers={"location": "/again"})
    out = safe_fetch.fetch("http://loop.example/", client_factory=_client(handler),
                           resolver=_resolver({"loop.example": [PUBLIC]}))
    assert "redirects" in out


def test_oversized_body_is_cut_at_the_cap(monkeypatch):
    monkeypatch.setattr(safe_fetch, "MAX_BYTES", 1000)
    out = safe_fetch.fetch("http://big.example/", resolver=_resolver({"big.example": [PUBLIC]}),
                           client_factory=_client(lambda r: httpx.Response(
                               200, headers={"content-type": "text/plain"}, content=b"a" * 50_000)))
    assert len(out.split("\n\n[")[0]) == 1000 and "download stopped" in out


def test_gzip_bomb_stops_at_the_decoded_cap(monkeypatch):
    monkeypatch.setattr(safe_fetch, "MAX_BYTES", 10_000)
    bomb = gzip.compress(b"\0" * 20_000_000)
    assert len(bomb) < 100_000                          # tiny on the wire, huge decoded
    out = safe_fetch.fetch("http://bomb.example/", resolver=_resolver({"bomb.example": [PUBLIC]}),
                           client_factory=_client(lambda r: httpx.Response(
                               200, headers={"content-type": "text/plain", "content-encoding": "gzip"},
                               content=bomb)))
    assert "download stopped" in out and len(out) < 12_000


def test_pdf_and_archives_are_refused_not_decoded():
    for ctype in ("application/pdf", "application/zip", "application/octet-stream"):
        out = safe_fetch.fetch("http://f.example/x", resolver=_resolver({"f.example": [PUBLIC]}),
                               client_factory=_client(lambda r, c=ctype: httpx.Response(
                                   200, headers={"content-type": c}, content=b"%PDF-1.7")))
        assert out.startswith("Error:") and "document pipeline" in out


def test_html_is_reduced_to_text_without_scripts():
    html = "<html><body><script>steal()</script><p>Hello <b>world</b></p></body></html>"
    out = safe_fetch.fetch("http://h.example/", resolver=_resolver({"h.example": [PUBLIC]}),
                           client_factory=_client(lambda r: httpx.Response(
                               200, headers={"content-type": "text/html"}, text=html)))
    assert "Hello" in out and "world" in out and "steal" not in out


def test_urls_with_credentials_and_odd_schemes_are_refused():
    r = _resolver({"h.example": [PUBLIC]})
    assert safe_fetch.fetch("http://user:pw@h.example/", resolver=r).startswith("Error:")
    for url in ("file:///etc/passwd", "gopher://h.example/", "ftp://h.example/x"):
        assert safe_fetch.fetch(url, resolver=r).startswith("Error:")


def test_dns_failure_blocks_instead_of_falling_through():
    def broken(host):
        raise OSError("nxdomain")
    assert safe_fetch.fetch("http://nope.example/", resolver=broken).startswith("Error: could not resolve")


def test_greenboost_control_ports_on_loopback_are_blocked(monkeypatch):
    monkeypatch.delenv("GB_WEBFETCH_ALLOW_LOCAL", raising=False)
    out = safe_fetch.fetch("http://127.0.0.1:11369/api/ps", resolver=lambda h: ["127.0.0.1"])
    assert out.startswith("Error:") and "control ports" in out
    # other loopback services stay reachable: local-first on purpose
    ok = safe_fetch.fetch("http://localhost:8080/", resolver=lambda h: ["127.0.0.1"],
                          client_factory=_client(lambda r: httpx.Response(
                              200, headers={"content-type": "text/plain"}, text="dev server")))
    assert ok == "dev server"
    monkeypatch.setenv("GB_WEBFETCH_ALLOW_LOCAL", "1")
    assert safe_fetch.validate("http://127.0.0.1:11369/", resolver=lambda h: ["127.0.0.1"])[0] == "http"


def test_http_errors_are_reported_not_returned_as_content():
    out = safe_fetch.fetch("http://e.example/", resolver=_resolver({"e.example": [PUBLIC]}),
                           client_factory=_client(lambda r: httpx.Response(
                               404, headers={"content-type": "text/html"}, text="<p>nope</p>")))
    assert out.startswith("Error:") and "404" in out


def test_blocked_requests_leave_a_dataflux_event(monkeypatch):
    events = []
    monkeypatch.setattr(safe_fetch, "_emit", lambda status, url, reason="", **kw: events.append((status, reason)))
    safe_fetch.fetch("http://169.254.169.254/", resolver=lambda h: ["169.254.169.254"])
    assert events and events[0][0] == "blocked"


def test_handle_fetch_url_goes_through_the_guard_and_screenshot_checks_first(monkeypatch):
    monkeypatch.setattr(safe_fetch, "_resolve", lambda host, port, resolver=None: ["10.1.2.3"])
    assert handlers.handle_fetch_url("http://lan-box/").startswith("Error:")
    called = []
    import greenboost_cli.instruments.screenshot_utils as su
    monkeypatch.setattr(su, "capture_screenshot", lambda *a, **k: called.append(a) or "shot")
    assert handlers.handle_screenshot("http://lan-box/", "/tmp/x.png").startswith("Error:")
    assert called == []

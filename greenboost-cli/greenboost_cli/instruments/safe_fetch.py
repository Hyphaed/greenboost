"""WebFetch's HTTP client, with the guard enforced where the bytes move.

`url_guard.check_url` judges a URL, but judging a URL is not the same as
controlling a request. The fetcher this replaces resolved DNS a second time after
the check (so a name that answered "public" to the guard and "10.0.0.5" to the
connection got through), followed redirects without looking at where they went,
downloaded the whole body before truncating it, and happily returned any content
type. This module closes those at the client:

- DNS is resolved ONCE per hop, every address is judged, and the connection goes
  to that exact address (TLS still validates the certificate against the real
  hostname through SNI). There is no second lookup to rebind.
- Redirects are followed by hand, at most MAX_REDIRECTS, and each hop is
  re-validated from scratch, so a public page can't bounce the fetch into the LAN.
- The body is streamed and cut at MAX_BYTES of *decoded* data, so a gzip bomb stops
  at the cap instead of filling memory.
- Only text-like content types are returned; a PDF or archive is refused with a
  pointer to the document pipeline, not decoded as text.
- Credential-bearing URLs (user:pass@host) are refused, and nothing from the
  caller's environment (cookies, auth headers) is attached.
- GreenBoost's own control ports on loopback (gb-synapse's API can serve and stop
  models, unauthenticated) are blocked unless GB_WEBFETCH_ALLOW_LOCAL=1. A hostile
  page must not be able to steer the agent at them. Other loopback services stay
  reachable: this box is local-first on purpose.

Every refusal is logged to dataflux as `web_fetch` with the reason, so a blocked
request is visible after the fact. Failures never raise; they come back as an
"Error: ..." string like every other tool result.
"""
from __future__ import annotations

import ipaddress
import os
import re
import socket
import time
from urllib.parse import urljoin, urlparse

from greenboost_cli.instruments.url_guard import ALLOWED_SCHEMES, _classify

MAX_REDIRECTS = 5
MAX_BYTES = int(os.environ.get("GB_WEBFETCH_MAX_BYTES", 2_000_000))
CONNECT_TIMEOUT_S = 10.0
TOTAL_TIMEOUT_S = 30.0
MAX_CHARS = 30_000
_TEXTY = re.compile(r"^(text/|application/(json|xml|xhtml\+xml|javascript|x-ndjson)|"
                    r"application/[\w.+-]+\+(json|xml))", re.I)

# Control surfaces on loopback that a fetched page must never be able to reach.
# Env-derived so a relocated port is honored; these are service ports, not hardware.
def _control_ports() -> frozenset[int]:
    ports = {int(os.environ.get("GB_SYNAPSE_PORT", "11369")),
             int(os.environ.get("GB_SYNAPSE_INTERNAL_PORT", "12369")),
             int(os.environ.get("GB_SYNAPSE_EMBED_PORT", "13369")),
             int(os.environ.get("GB_DASHBOARD_PORT", "7821"))}
    return frozenset(ports)


def _emit(status: str, url: str, reason: str = "", **extra) -> None:
    try:
        from greenboost_cli.gb_paths import gb_module
        host = urlparse(url).hostname or ""
        gb_module("gb_dataflux").emit({"kind": "web_fetch", "status": status,
                                       "host": host, "reason": reason[:200], **extra})
    except Exception:
        pass


def _resolve(host: str, port: int, resolver=None) -> list[str]:
    if resolver is not None:
        return list(resolver(host))
    return [ai[4][0] for ai in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


def validate(url: str, resolver=None) -> tuple[str, str, int, str] | str:
    """(scheme, host, port, pinned_ip) when the URL may be fetched, else an
    "Error: ..." string. Resolves DNS exactly once and judges every answer."""
    try:
        u = urlparse((url or "").strip())
        port = u.port
    except ValueError:
        return f"Error: {url!r} is not a parseable URL"
    scheme = (u.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        return (f"Error: WebFetch only speaks http and https; '{scheme or 'no'}' refused. "
                f"Use Read for local files.")
    if u.username or u.password:
        return "Error: URLs carrying credentials (user:pass@host) are refused."
    host = u.hostname
    if not host:
        return f"Error: {url!r} has no host"
    port = port or (443 if scheme == "https" else 80)
    try:
        ipaddress.ip_address(host)
        addrs = [host]                      # an IP literal needs no lookup, and has nothing to rebind
    except ValueError:
        addrs = None
    try:
        if addrs is None:
            addrs = _resolve(host, port, resolver)
    except Exception as e:
        return f"Error: could not resolve {host} ({type(e).__name__})"
    if not addrs:
        return f"Error: {host} did not resolve to any address"
    for ip in addrs:
        why = _classify(ip)
        if why:
            return (f"Error: refusing to fetch {host}, it resolves to {ip}, {why}.\n"
                    f"  If this is deliberate, fetch it with Bash(curl ...), which goes "
                    f"through the normal approval gate.")
    pinned = addrs[0]
    try:
        loopback = ipaddress.ip_address(pinned).is_loopback
    except ValueError:
        loopback = False
    if loopback and port in _control_ports() and os.environ.get("GB_WEBFETCH_ALLOW_LOCAL") != "1":
        return (f"Error: refusing to fetch {host}:{port}, that is one of GreenBoost's own control "
                f"ports (it can serve and stop models). A web page must not be able to point "
                f"the agent at it. Set GB_WEBFETCH_ALLOW_LOCAL=1 if you mean it.")
    return scheme, host, port, pinned


def _html_to_text(body: str) -> str:
    try:
        from scrapling.parser import Selector
        text = Selector(body).get_all_text()
        if text and text.strip():
            return text.strip()
    except Exception:
        pass
    body = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", body)
    return re.sub(r"[ \t]+", " ", re.sub(r"(?s)<[^>]+>", " ", body)).strip()


def fetch(url: str, *, resolver=None, client_factory=None) -> str:
    """Fetch `url` as readable text under every rule above. Returns the text, or
    an "Error: ..." string (never raises)."""
    import httpx

    deadline = time.monotonic() + TOTAL_TIMEOUT_S
    current = url
    for hop in range(MAX_REDIRECTS + 1):
        checked = validate(current, resolver)
        if isinstance(checked, str):
            _emit("blocked", current, checked.splitlines()[0], hop=hop)
            return checked
        scheme, host, port, ip = checked
        netloc_ip = f"[{ip}]" if ":" in ip else ip
        target = f"{scheme}://{netloc_ip}:{port}{urlparse(current).path or '/'}"
        if urlparse(current).query:
            target += "?" + urlparse(current).query
        client = (client_factory or (lambda: httpx.Client(
            timeout=httpx.Timeout(TOTAL_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
            follow_redirects=False, trust_env=False)))()
        try:
            with client:
                with client.stream("GET", target,
                                   headers={"Host": host if port in (80, 443) else f"{host}:{port}",
                                            "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.1",
                                            "User-Agent": "greenboost-cli/1 (+local)"},
                                   extensions={"sni_hostname": host}) as r:
                    if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                        current = urljoin(current, r.headers["location"])
                        continue
                    ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
                    if ctype and not _TEXTY.match(ctype):
                        msg = (f"Error: {current} returned {ctype}, which WebFetch doesn't read as text. "
                               f"Documents such as PDFs go through the document pipeline, not here.")
                        _emit("blocked", current, f"content-type {ctype}", status_code=r.status_code)
                        return msg
                    chunks, size = [], 0
                    for chunk in r.iter_bytes():
                        size += len(chunk)
                        if size > MAX_BYTES:
                            chunks.append(chunk[: max(0, MAX_BYTES - (size - len(chunk)))])
                            break
                        chunks.append(chunk)
                        if time.monotonic() > deadline:
                            _emit("blocked", current, "total timeout")
                            return f"Error: {current} took longer than {TOTAL_TIMEOUT_S:.0f}s to download"
                    body = b"".join(chunks).decode(r.encoding or "utf-8", errors="replace")
                    status = r.status_code
        except Exception as e:
            _emit("error", current, type(e).__name__)
            return f"Error: could not fetch {current} ({type(e).__name__})"
        truncated = size > MAX_BYTES
        text = body if "json" in ctype or ctype.startswith("text/plain") else _html_to_text(body)
        _emit("ok", current, status_code=status, bytes=min(size, MAX_BYTES), truncated=truncated, hops=hop)
        if status >= 400:
            return f"Error: {current} returned HTTP {status}"
        note = f"\n\n[download stopped at {MAX_BYTES:,} bytes]" if truncated else ""
        return text[:MAX_CHARS] + note
    _emit("blocked", current, "too many redirects")
    return f"Error: more than {MAX_REDIRECTS} redirects starting from {url}"

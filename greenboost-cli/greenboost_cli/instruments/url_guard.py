"""Where WebFetch is allowed to reach.

`WebFetch` sits in `_ALWAYS_APPROVED`: it never asks. That is the right
default for fetching a doc page, and the wrong one for fetching
`http://169.254.169.254/latest/meta-data/`, which on a cloud host hands back
credentials. The agent does not have to be malicious for this to matter , a
URL can arrive from a fetched page, an MCP result, a file in the repo, or an
unattended `/nonstop` run following an instruction it read somewhere.

The distinction this module keeps, adapted from NemoClaw's own private-network
handling, is the one a naive "block private IPs" guard gets wrong:

**Loopback is not the same as the LAN.** This box is local-first by design.
gb-synapse serves on 127.0.0.1:11369, the dashboard and the MCP servers are
local, and fetching them is ordinary, useful work. Blocking loopback would
break the product to prevent an attack that loopback is not the vector for.

What IS blocked:

- **Link-local (169.254.0.0/16, fe80::/10)**, which includes the cloud
  metadata address. This is the one that leaks credentials.
- **LAN ranges (10/8, 172.16/12, 192.168/16, fc00::/7)**: other people's
  machines, routers, NAS boxes, printers. Nothing the agent needs and
  everything an SSRF wants.
- **Non-HTTP schemes** (`file://`, `gopher://`, `ftp://`): a fetcher that can
  read local files is a file reader with no permission gate in front of it.

The check runs against the RESOLVED address, not the hostname, because
`evil.example.com` resolving to 169.254.169.254 is the standard way around a
string-matching guard.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

ALLOWED_SCHEMES = frozenset({"http", "https"})


def _classify(ip: str) -> str:
    """"" when the address is fine; otherwise why it is not."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ""                       # not an IP literal; nothing to judge
    if addr.is_loopback:
        return ""                       # local services are the point of this box
    if addr.is_link_local:
        return ("a link-local address , this range holds cloud instance "
                "metadata (169.254.169.254), which returns credentials")
    if addr.is_private:
        return ("a private LAN address , other machines on this network are "
                "not the agent's to reach")
    if addr.is_reserved or addr.is_multicast:
        return "a reserved or multicast address"
    return ""


def check_url(url: str, resolver=None) -> str:
    """"" when the fetch may proceed; otherwise a refusal explaining why."""
    try:
        u = urlparse((url or "").strip())
    except ValueError:
        return f"Error: {url!r} is not a parseable URL"
    scheme = (u.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        return (f"Error: WebFetch only speaks http and https , '{scheme or 'no'}' "
                f"scheme refused. A fetcher that reads {scheme}:// URLs is a "
                f"file reader with no permission gate in front of it; use Read "
                f"for local files.")
    host = u.hostname
    if not host:
        return f"Error: {url!r} has no host"

    literal = _classify(host)
    if literal:
        return (f"Error: refusing to fetch {host} , {literal}.\n"
                f"  If this is deliberate, fetch it with Bash(curl ...), which "
                f"goes through the normal approval gate.")

    # Resolve and judge every address the name maps to. One bad answer is
    # enough: a name with two A records only needs the hostile one to be used.
    resolve = resolver or (lambda h: [ai[4][0] for ai in socket.getaddrinfo(h, None)])
    try:
        addrs = resolve(host)
    except Exception:
        return ""                       # DNS failure is the fetcher's to report
    for ip in addrs:
        why = _classify(ip)
        if why:
            return (f"Error: refusing to fetch {host} , it resolves to {ip}, "
                    f"{why}.\n"
                    f"  A hostname pointing into a private range is how an SSRF "
                    f"gets past a guard that only checks the URL text.")
    return ""

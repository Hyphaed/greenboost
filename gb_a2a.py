#!/usr/bin/env python3
"""gb_a2a , GreenBoost A2A (agent-to-agent) gateway.

Realizes the AgentCard gateway the gb_cluster.py:172 comment anticipated: a
small stdlib HTTP server that lets a *delegating agent* (not just an MCP client)
discover this GreenBoost cluster's hardware and invoke its gated actuation
verbs over JSON-RPC.

Endpoints:
  GET  /.well-known/agent.json   , the AgentCard (per-node hardware + skills)
  POST /                         , JSON-RPC 2.0; method = verb name,
                                   params = that verb's kwargs (incl. confirm)

Both control planes , this gateway and the MCP tools , dispatch through the
SAME gb_actuation.VERBS table, so a verb can never behave differently depending
on how it was reached. Actuation stays double-gated (confirm=True AND
GB_ORCH_ACTUATE=1); the gateway CANNOT bypass the gate.

Security (peer of greenboost_netd.c , a security-review target):
  - Binds 127.0.0.1 by default (GB_A2A_BIND, e.g. "0.0.0.0:8790" for LAN).
  - A non-loopback bind REQUIRES a Bearer token (GB_A2A_TOKEN) or the server
    refuses to start , no unauthenticated LAN actuation surface.
  - Per-client rate limit (GB_A2A_RATE, default 30 req/min), like the CLI MCP.
  - Verbs are an allowlist (VERBS); no shell passthrough, no arbitrary import.
  - SVC-03 / SEC-01: `confirm` is a parameter the CALLER supplies, so it
    authorizes nothing. Each verb call runs inside
    gb_actuation.actuation_auth(<Bearer token from the Authorization header>),
    which compares it in constant time with the owner's 0600 token file
    (GB_A2A_TOKEN_FILE). Without a valid token the verb is a dry run and the
    reply says why; the token is never logged. Browser requests are refused too: a request carrying an Origin
    header, a non-JSON Content-Type (the way a page avoids a CORS preflight),
    or a Host header that is not the bind address (DNS rebinding).
  - Request bodies are capped (GB_A2A_MAX_BODY, default 1 MiB) and each
    connection times out (GB_A2A_TIMEOUT_S, default 10 s).

Every request emits an `a2a_request` dataflux event (verb, gated, outcome) so
the gateway is observable in the flight recorder like every other subsystem.

Run: `python3 gb_a2a.py serve` (or via the greenboost-a2a systemd unit).
"""
from __future__ import annotations

import hmac
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gb_actuation  # noqa: E402

DEFAULT_BIND = os.environ.get("GB_A2A_BIND", "127.0.0.1:8790")
_RATE_PER_MIN = int(os.environ.get("GB_A2A_RATE", "30") or "30")
_MAX_BODY = int(os.environ.get("GB_A2A_MAX_BODY", str(1 << 20)) or str(1 << 20))
_CONN_TIMEOUT_S = float(os.environ.get("GB_A2A_TIMEOUT_S", "10") or "10")

# ── rate limiter (per-client-ip sliding 60s window) ─────────────────────────
_rate_lock = threading.Lock()
_rate_hits: dict[str, list] = {}


def _rate_ok(client: str) -> bool:
    now = time.time()
    with _rate_lock:
        if len(_rate_hits) > 1024:      # bound the table (LAN bind, many clients)
            for k in [k for k, v in _rate_hits.items() if not v or now - v[-1] >= 60.0]:
                del _rate_hits[k]
        hits = [t for t in _rate_hits.get(client, []) if now - t < 60.0]
        if len(hits) >= _RATE_PER_MIN:
            _rate_hits[client] = hits
            return False
        hits.append(now)
        _rate_hits[client] = hits
        return True


def _emit_request(verb: str, gated: bool, outcome: str) -> None:
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "node": "host", "label": "a2a", "kind": "a2a_request",
            "stage": verb, "verb": verb, "gated": gated, "outcome": outcome,
            "status": "ok" if outcome != "error" else "error",
            "n_items": 0, "items": [], "duration_s": 0.0,
        })
    except Exception:
        pass


def _bind_parts(bind: str) -> tuple[str, int]:
    host, _, port = bind.rpartition(":")
    return (host or "127.0.0.1"), int(port or "8790")


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")


class _Handler(BaseHTTPRequestHandler):
    server_version = "greenboost-a2a/1.0"
    timeout = _CONN_TIMEOUT_S      # StreamRequestHandler: socket timeout per connection

    # Quiet the default stderr access log; dataflux is the record of truth.
    def log_message(self, *_a):  # noqa: N802
        pass

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self) -> bool:
        token = os.environ.get("GB_A2A_TOKEN", "")
        if not token:
            return True  # loopback-only, read-only mode (actuation refused in do_POST)
        got = self.headers.get("Authorization", "")
        return hmac.compare_digest(got.encode(), f"Bearer {token}".encode())

    def _bearer(self) -> str:
        """The token after 'Bearer ' in the Authorization header, else ''. Never logged."""
        got = self.headers.get("Authorization", "")
        return got[7:].strip() if got[:7].lower() == "bearer " else ""

    def _browser_like(self) -> bool:
        """True for a request a web page could have caused: it carries an Origin,
        or its Host is not one of the bind address's own names (DNS rebinding).
        Agents and curl send neither an Origin nor a foreign Host."""
        if self.headers.get("Origin"):
            return True
        host, port = _bind_parts(DEFAULT_BIND)
        if not _is_loopback(host):
            return False   # LAN bind: Host is whatever name the peer used; the token guards it
        names = {f"{h}:{port}" for h in ("127.0.0.1", "localhost", "[::1]")}
        return self.headers.get("Host", "") not in names

    def do_GET(self):  # noqa: N802
        client = self.client_address[0]
        if not _rate_ok(client):
            return self._json(429, {"error": "rate limit exceeded"})
        if self._browser_like():
            return self._json(403, {"error": "browser-origin requests are refused"})
        if self.path.rstrip("/") in ("/.well-known/agent", "/.well-known/agent.json"):
            if not self._authed():
                return self._json(401, {"error": "missing/invalid Bearer token"})
            return self._json(200, gb_actuation.agent_card(bind=DEFAULT_BIND))
        if self.path.rstrip("/") == "/.well-known/agent-card.json":
            # A2A protocol v0.3 path/shape (see gb_actuation.agent_card_v03) —
            # interop with ai-forge's studio/server/a2a_gateway.py, which
            # serves the same well-known path. the maintainer notes
            if not self._authed():
                return self._json(401, {"error": "missing/invalid Bearer token"})
            return self._json(200, gb_actuation.agent_card_v03(bind=DEFAULT_BIND))
        return self._json(404, {"error": "not found; try /.well-known/agent.json "
                                        "or /.well-known/agent-card.json (v0.3)"})

    def do_POST(self):  # noqa: N802
        client = self.client_address[0]
        if not _rate_ok(client):
            return self._json(429, {"error": "rate limit exceeded"})
        if self._browser_like():
            return self._json(403, {"error": "browser-origin requests are refused"})
        if not self._authed():
            return self._json(401, {"error": "missing/invalid Bearer token"})
        ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return self._json(415, {"error": "Content-Type must be application/json"})
        try:
            n = int(self.headers.get("Content-Length", "0") or "0")
            if n < 0 or n > _MAX_BODY:
                return self._json(413, {"error": f"body over {_MAX_BODY} bytes"})
            req = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(req, dict):
                raise ValueError("request must be an object")
        except (ValueError, OSError):
            return self._json(400, {"jsonrpc": "2.0", "id": None,
                                    "error": {"code": -32700, "message": "parse error"}})
        rid = req.get("id")
        method = req.get("method", "")
        params = req.get("params", {}) or {}
        if not isinstance(params, dict):
            return self._json(200, {"jsonrpc": "2.0", "id": rid,
                "error": {"code": -32602, "message": "params must be an object"}})
        fn = gb_actuation.VERBS.get(method)
        if fn is None:
            _emit_request(method or "?", False, "unknown_method")
            return self._json(200, {"jsonrpc": "2.0", "id": rid,
                "error": {"code": -32601,
                          "message": f"unknown method {method!r}; "
                                     f"one of {list(gb_actuation.VERBS)}"}})
        try:
            with gb_actuation.actuation_auth(self._bearer()) as authed:
                result = fn(**params)
            if params.get("confirm") and not authed and isinstance(result, dict):
                result["auth"] = ("not authorized: send 'Authorization: Bearer <token>' matching the "
                                  "owner's token file (GB_A2A_TOKEN_FILE); this call was a dry run")
            gated = bool(result.get("gate", {}).get("allowed")) \
                if isinstance(result, dict) else False
            _emit_request(method, gated, "applied" if gated else "dry_run")
            return self._json(200, {"jsonrpc": "2.0", "id": rid, "result": result})
        except TypeError as e:
            _emit_request(method, False, "bad_params")
            return self._json(200, {"jsonrpc": "2.0", "id": rid,
                "error": {"code": -32602, "message": f"bad params: {e}"}})
        except Exception as e:
            _emit_request(method, False, "error")
            return self._json(200, {"jsonrpc": "2.0", "id": rid,
                "error": {"code": -32000, "message": str(e)}})


def serve(bind: str = DEFAULT_BIND) -> None:
    host, port = _bind_parts(bind)
    if not _is_loopback(host) and not os.environ.get("GB_A2A_TOKEN"):
        raise SystemExit(
            f"refusing to bind non-loopback {host}:{port} without GB_A2A_TOKEN , "
            f"a LAN A2A actuation surface must be authenticated. Set GB_A2A_TOKEN "
            f"or bind 127.0.0.1.")
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"[gb_a2a] AgentCard + JSON-RPC on http://{host}:{port}/ "
          f"(auth={'token' if os.environ.get('GB_A2A_TOKEN') else 'loopback-only'}, "
          f"rate={_RATE_PER_MIN}/min, actuation gate=GB_ORCH_ACTUATE)", flush=True)
    try:
        # poll_interval only bounds how late shutdown() is noticed; the 0.5 s
        # default woke an idle gateway twice a second (measured 120 context
        # switches in 60 s, SVC-08).
        httpd.serve_forever(poll_interval=5.0)
    except KeyboardInterrupt:
        httpd.shutdown()


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "serve"
    if cmd == "serve":
        serve(argv[2] if len(argv) > 2 else DEFAULT_BIND)
        return 0
    if cmd == "card":
        print(json.dumps(gb_actuation.agent_card(bind=DEFAULT_BIND), indent=2))
        return 0
    print("usage: gb_a2a.py [serve [bind] | card]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

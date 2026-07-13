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

Every request emits an `a2a_request` dataflux event (verb, gated, outcome) so
the gateway is observable in the flight recorder like every other subsystem.

Run: `python3 gb_a2a.py serve` (or via the greenboost-a2a systemd unit).
"""
from __future__ import annotations

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

# ── rate limiter (per-client-ip sliding 60s window) ─────────────────────────
_rate_lock = threading.Lock()
_rate_hits: dict[str, list] = {}


def _rate_ok(client: str) -> bool:
    now = time.time()
    with _rate_lock:
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
            return True  # loopback-only mode (enforced at startup)
        return self.headers.get("Authorization", "") == f"Bearer {token}"

    def do_GET(self):  # noqa: N802
        client = self.client_address[0]
        if not _rate_ok(client):
            return self._json(429, {"error": "rate limit exceeded"})
        if self.path.rstrip("/") in ("/.well-known/agent", "/.well-known/agent.json"):
            if not self._authed():
                return self._json(401, {"error": "missing/invalid Bearer token"})
            return self._json(200, gb_actuation.agent_card(bind=DEFAULT_BIND))
        return self._json(404, {"error": "not found; try /.well-known/agent.json"})

    def do_POST(self):  # noqa: N802
        client = self.client_address[0]
        if not _rate_ok(client):
            return self._json(429, {"error": "rate limit exceeded"})
        if not self._authed():
            return self._json(401, {"error": "missing/invalid Bearer token"})
        try:
            n = int(self.headers.get("Content-Length", "0") or "0")
            req = json.loads(self.rfile.read(n) or b"{}")
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
            result = fn(**params)
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
        httpd.serve_forever()
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

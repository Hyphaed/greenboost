"""
Lightweight MCP client (Model Context Protocol).

Supports stdio (subprocess JSON-RPC newline-delimited) and HTTP transports.
Auto-discovers .mcp.json from cwd and manages multiple server connections.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

from greenboost_cli.core.bounded_lines import BoundedLineDecoder

# Reserved names an MCP server must never be allowed to occupy. Sourced from
# the same registry the dispatcher and orchestrator use, so this can't drift
# from the real builtin list. AskUserQuestion/ToolSearch are handled by
# orchestrator.py before an MCP lookup ever happens (see the intercepts
# there) so they can't actually be shadowed today, but they're reserved too
# — a future refactor that reorders those checks shouldn't silently reopen
# this hole.
def _builtin_tool_names() -> frozenset[str]:
    from greenboost_cli.instruments.schemas import INSTRUMENT_DEFINITIONS
    return frozenset(d["name"] for d in INSTRUMENT_DEFINITIONS)


#: Seconds to wait for a server's initialize / tools/list. A server that never
#: answers must cost one timeout, not the whole session: before this, one hung
#: server blocked connect_all() forever and /mcp-servers kept saying "none".
INIT_TIMEOUT_S = float(os.environ.get("GB_MCP_INIT_TIMEOUT", "30"))
#: Seconds to wait for a tools/call on a stdio server.
CALL_TIMEOUT_S = float(os.environ.get("GB_MCP_CALL_TIMEOUT", "120"))
#: Longest single JSON-RPC line accepted from a server (a big tools/list is
#: legitimately hundreds of KB; the old 50x4096 read loop capped it at ~200 KB
#: and silently reported "connected, 0 tools").
MAX_LINE_BYTES = 16 * 1024 * 1024
#: stderr lines kept per server for /mcp-servers log. Bounded: this runs for days.
STDERR_TAIL_LINES = 200


class MCPStdioClient:
    """MCP client that speaks to a subprocess over stdin/stdout (newline-delimited JSON)."""

    def __init__(self, name: str, command: list[str], env: dict | None = None,
                 cwd: str | None = None, init_timeout: float | None = None):
        self.name = name
        #: A first `npx -y pkg@latest` downloads the package before it answers; /mcp-add
        #: gives that longer than a normal start-up (INIT_TIMEOUT_S).
        self._init_timeout = init_timeout
        self._cmd = command
        self._extra_env = env or {}
        self._cwd = cwd
        self._proc: subprocess.Popen | None = None
        self._id = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._responses: dict[int, dict] = {}
        self._reader_done = threading.Event()
        self.stderr_tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
        self.tools: list[dict] = []
        self.connected = False
        #: Why the last connect() failed, in words the operator can act on.
        self.error: str = ""
        self._restarted = False

    def connect(self) -> bool:
        self.error = ""
        self._reader_done.clear()
        env = {**os.environ, **{k: str(v) for k, v in self._extra_env.items()}}
        try:
            self._proc = subprocess.Popen(
                self._cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                env=env,
                cwd=self._cwd or None,
                # Own process group: npx/uvx servers are wrappers whose real
                # server is a grandchild, and closing stdin never reached it, so
                # every exit left orphans. close() signals the whole group.
                start_new_session=True,
                # bufsize=0 (unbuffered), NOT 1. Line buffering has no meaning
                # on a binary pipe, and Python 3.14 says so out loud — it emits
                # a RuntimeWarning per stream, per server, straight through the
                # startup banner (four lines mid-banner with two MCP servers
                # configured, more with more). Nothing was gained by it either:
                # _request() and _notify() both flush() after every write, which
                # is what actually guarantees delivery.
                bufsize=0,
            )
        except FileNotFoundError:
            self.error = f"command not found: {self._cmd[0]}"
            return False
        except (PermissionError, OSError) as e:
            self.error = f"cannot start {self._cmd[0]}: {e}"
            return False

        threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True,
                         name=f"gb-mcp-out-{self.name}").start()
        threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True,
                         name=f"gb-mcp-err-{self.name}").start()

        resp = self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "greenboost-cli", "version": "1.0"},
        }, timeout=(self._init_timeout or INIT_TIMEOUT_S))
        if not resp or "result" not in resp:
            tail = self.stderr_tail[-1] if self.stderr_tail else ""
            if self._proc.poll() is not None:
                self.error = f"exited with code {self._proc.returncode}"
            else:
                self.error = f"no initialize reply within {(self._init_timeout or INIT_TIMEOUT_S):g}s"
            if tail:
                self.error += f" ({tail[:160]})"
            self.close()
            return False

        self._notify("notifications/initialized", {})

        resp = self._request("tools/list", {}, timeout=(self._init_timeout or INIT_TIMEOUT_S))
        if resp and "result" in resp:
            self.tools = resp["result"].get("tools", [])
            # Servers page tools/list with nextCursor; follow it, bounded.
            cursor = resp["result"].get("nextCursor")
            for _ in range(20):
                if not cursor:
                    break
                resp = self._request("tools/list", {"cursor": cursor}, timeout=(self._init_timeout or INIT_TIMEOUT_S))
                if not resp or "result" not in resp:
                    break
                self.tools += resp["result"].get("tools", [])
                cursor = resp["result"].get("nextCursor")
        else:
            self.error = "connected but tools/list got no reply"

        self.connected = True
        return True

    def call_tool(self, name: str, arguments: dict) -> str:
        # A server that died mid-session is restarted once, not left dead for
        # the days this CLI is meant to run.
        if self._proc is not None and self._proc.poll() is not None and not self._restarted:
            self._restarted = True
            self.close()
            if not self.connect():
                return f"ERROR: MCP server '{self.name}' died and could not restart: {self.error}"
        resp = self._request("tools/call", {"name": name, "arguments": arguments},
                             timeout=CALL_TIMEOUT_S)
        if resp is None:
            return (f"ERROR: MCP server '{self.name}' did not respond within "
                    f"{CALL_TIMEOUT_S:g}s")
        if "error" in resp:
            return f"ERROR: {resp['error'].get('message', str(resp['error']))}"
        return _extract_content(resp.get("result", {}).get("content", []))

    def close(self):
        proc = self._proc
        if proc:
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=1.0)          # a well-behaved server exits on stdin EOF
            except Exception:
                self._signal_group(signal.SIGTERM)
                try:
                    proc.wait(timeout=1.5)
                except Exception:
                    self._signal_group(signal.SIGKILL)
            # The leader may be gone while its children (npx -> node) are not.
            self._signal_group(signal.SIGTERM)
        self.connected = False

    def _signal_group(self, sig) -> None:
        try:
            os.killpg(self._proc.pid, sig)      # pgid == pid: start_new_session=True
        except (ProcessLookupError, PermissionError, OSError, AttributeError):
            pass

    # ── transport ──────────────────────────────────────────────────────────
    def _read_stdout(self, proc: subprocess.Popen) -> None:
        """Dedicated reader: lets _request() wait with a timeout instead of
        blocking on a pipe read that a hung server never completes."""
        def on_line(line: str) -> None:
            line = line.strip()
            if not line:
                return
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return
            # A server->client REQUEST (has "method" and "id"): answer it, or a
            # server that waits on the reply (roots/list at startup, ping
            # keep-alives) stalls until its own timeout.
            if "method" in obj and "id" in obj:
                self._answer_server_request(obj)
                return
            # Notifications carry no id; only responses (result/error) are routed.
            if "id" in obj and ("result" in obj or "error" in obj):
                with self._cond:
                    self._responses[obj["id"]] = obj
                    self._cond.notify_all()

        decoder = BoundedLineDecoder(MAX_LINE_BYTES, on_line)
        try:
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                decoder.write(chunk)
            decoder.end()
        except Exception:
            pass
        self._reader_done.set()
        with self._cond:
            self._cond.notify_all()     # wake any waiter: the server is gone

    def _answer_server_request(self, req: dict) -> None:
        method = req.get("method")
        if method == "ping":
            reply: dict = {"result": {}}
        elif method == "roots/list":
            reply = {"result": {"roots": [{"uri": Path.cwd().resolve().as_uri(),
                                           "name": Path.cwd().name}]}}
        else:       # sampling/elicitation etc.: not supported, say so instead of hanging
            reply = {"error": {"code": -32601, "message": f"{method} not supported by greenboost-cli"}}
        msg = json.dumps({"jsonrpc": "2.0", "id": req["id"], **reply})
        try:
            with self._lock:
                self._proc.stdin.write((msg + "\n").encode("utf-8"))
                self._proc.stdin.flush()
        except Exception:
            pass

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        try:
            for raw in iter(proc.stderr.readline, b""):
                self.stderr_tail.append(raw.decode("utf-8", errors="replace").rstrip())
        except Exception:
            pass

    def _request(self, method: str, params: dict, timeout: float = CALL_TIMEOUT_S) -> dict | None:
        with self._lock:
            self._id += 1
            msg_id = self._id
        msg = json.dumps({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        try:
            with self._lock:
                self._proc.stdin.write((msg + "\n").encode("utf-8"))
                self._proc.stdin.flush()
        except Exception:
            return None
        deadline = time.monotonic() + timeout
        with self._cond:
            while msg_id not in self._responses:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._reader_done.is_set():
                    break
                self._cond.wait(timeout=min(remaining, 0.5))
            return self._responses.pop(msg_id, None)

    def _notify(self, method: str, params: dict):
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        try:
            with self._lock:
                self._proc.stdin.write((msg + "\n").encode("utf-8"))
                self._proc.stdin.flush()
        except Exception:
            pass


class MCPHttpClient:
    """MCP client over HTTP (streamable-HTTP JSON-RPC transport)."""

    def __init__(self, name: str, url: str, headers: dict | None = None):
        self.name = name
        self._url = url
        self._headers = {str(k): str(v) for k, v in (headers or {}).items()}
        self._session_id: str = ""
        self._id = 0
        self.tools: list[dict] = []
        self.connected = False
        self.error: str = ""
        self.stderr_tail: deque[str] = deque(maxlen=1)

    def connect(self, timeout: float = INIT_TIMEOUT_S) -> bool:
        self.error = ""
        resp = self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "greenboost-cli", "version": "1.0"},
        }, timeout=timeout)
        if not resp or "result" not in resp:
            self.error = self.error or f"no initialize reply from {self._url}"
            return False

        self._notify("notifications/initialized", {}, timeout)

        resp = self._request("tools/list", {}, timeout=timeout)
        if resp and "result" in resp:
            self.tools = resp["result"].get("tools", [])

        self.connected = True
        return True

    def call_tool(self, name: str, arguments: dict) -> str:
        resp = self._request("tools/call", {"name": name, "arguments": arguments}, timeout=1800.0)
        if resp is None:
            return f"ERROR: MCP HTTP server '{self.name}' did not respond"
        if "error" in resp:
            return f"ERROR: {resp['error'].get('message', str(resp['error']))}"
        return _extract_content(resp.get("result", {}).get("content", []))

    def close(self):
        self.connected = False

    def _post(self, body: dict, timeout: float):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self._headers,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        req = urllib.request.Request(
            self._url, data=json.dumps(body).encode(), headers=headers, method="POST")
        return urllib.request.urlopen(req, timeout=timeout)

    def _notify(self, method: str, params: dict, timeout: float) -> None:
        try:
            with self._post({"jsonrpc": "2.0", "method": method, "params": params}, timeout):
                pass
        except Exception:
            pass

    def _request(self, method: str, params: dict, timeout: float = 30.0) -> dict | None:
        self._id += 1
        body = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
        try:
            with self._post(body, timeout) as r:
                sid = r.headers.get("Mcp-Session-Id")
                if sid:
                    self._session_id = sid
                text = r.read().decode("utf-8", errors="replace")
                # Handle SSE (text/event-stream) responses
                if "data:" in text:
                    for line in text.splitlines():
                        if line.startswith("data:"):
                            payload = line[5:].strip()
                            if payload and payload != "[DONE]":
                                try:
                                    return json.loads(payload)
                                except json.JSONDecodeError:
                                    pass
                    return None
                return json.loads(text)
        except urllib.error.HTTPError as e:
            self.error = f"HTTP {e.code} from {self._url}"
            return None
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            self.error = f"{type(e).__name__}: {e}"[:200]
            return None


def _make_client(name: str, spec: dict, init_timeout: float | None = None):
    if spec.get("error"):
        return _FailedClient(name, spec["error"])
    if spec.get("type", "stdio") == "stdio":
        return MCPStdioClient(name, spec["command"], spec.get("env"), spec.get("cwd"), init_timeout)
    return MCPHttpClient(name, spec["url"], spec.get("headers"))


class _FailedClient:
    """Placeholder for a configured server that could not even be started
    (bad entry). Keeps its place in /mcp-servers with the reason."""

    def __init__(self, name: str, error: str):
        self.name = name
        self.error = error
        self.tools: list[dict] = []
        self.connected = False
        self.stderr_tail: deque[str] = deque(maxlen=1)

    def connect(self) -> bool:
        return False

    def call_tool(self, name: str, arguments: dict) -> str:
        return f"ERROR: MCP server '{self.name}' is not connected: {self.error}"

    def close(self) -> None:
        pass


class MCPRegistry:
    """Manages multiple MCP clients, exposes merged tool schemas, routes calls."""

    # Class-level fallbacks: tests (and any caller) that build a registry
    # without __init__ must still be able to connect_all().
    _reg_lock = threading.Lock()
    specs: "dict[str, dict]" = {}

    def __init__(self):
        self._clients: dict[str, MCPStdioClient | MCPHttpClient] = {}
        self._tool_to_server: dict[str, str] = {}
        # Advertised (mcp__<server>__<tool>) name -> the RAW tool name the
        # underlying MCP client actually needs for tools/call.
        self._raw_tool_name: dict[str, str] = {}
        self.tool_schemas: list[dict] = []
        self._mcp_json_path: Path | None = None
        #: name -> discovered spec (source file, type) for /mcp-servers.
        self.specs: dict[str, dict] = {}
        self._seen_raw_names: dict[str, str] = {}
        # Populated by connect_all(): human-readable collision warnings.
        # Never raised as an exception — connect_all() runs from a
        # background thread wrapped in a bare except-pass (terminal/repl.py
        # _mcp_autoconnect), so an exception here would be silently
        # swallowed and defeat the point of warning at all. Callers that
        # want this surfaced (the REPL does) read it after connect_all().
        #: Tools EXCLUDED from the registry (a name shadowing a builtin).
        #: These are real problems: the tool is not callable at all.
        self.collisions: list[str] = []
        #: Tools two servers happen to name the same. Not a problem — both
        #: remain callable under their prefixed names — so these are reported
        #: as information, never as a warning.
        self.duplicate_names: list[str] = []
        #: Servers whose tools are NOT advertised to the model this session.
        #:
        #: Dormant, not disconnected. The subprocess stays up and every tool
        #: stays callable , ToolSearch still finds them and call_tool() still
        #: routes to them. Only the SCHEMAS are withheld from the prompt,
        #: because the schemas are what costs.
        #:
        #: Measured on the reference box: ten servers, 238 tools, ~7k prompt
        #: tokens of schemas re-sent on every request. Prefill is super-linear
        #: in prompt length (+9.1 s per 2048-token chunk observed), so dropping
        #: the servers a task does not need is the single biggest lever the
        #: operator has , bigger than any decode tuning in this codebase.
        self.dormant_servers: set[str] = set()

    def server_of(self, prefixed_name: str) -> str:
        """Which server advertises this mcp__server__tool name ("" if unknown)."""
        return self._tool_to_server.get(prefixed_name, "")

    def active_tool_schemas(self) -> list:
        """Schemas to advertise , everything except dormant servers' tools.

        Callers building a prompt use THIS, never `tool_schemas` directly.
        ToolSearch deliberately keeps searching the full set, so making a
        server dormant hides it from the prompt without making it unreachable.
        """
        if not self.dormant_servers:
            return list(self.tool_schemas)
        return [t for t in self.tool_schemas
                if self.server_of(t.get("name", "")) not in self.dormant_servers]

    def set_dormant(self, names: "set[str]") -> None:
        """Replace the dormant set, ignoring names that are not servers here."""
        known = set(self._clients)
        self.dormant_servers = {n for n in names if n in known}

    def schema_cost(self, server: str) -> int:
        """Rough prompt tokens this server's schemas cost, for an honest UI.

        chars//4 over the JSON actually advertised. Approximate on purpose ,
        the point is to let someone see that one server costs 4k and another
        costs 200, not to predict a tokenizer.
        """
        import json as _json

        total = sum(len(_json.dumps(t)) for t in self.tool_schemas
                    if self.server_of(t.get("name", "")) == server)
        return total // 4

    @classmethod
    def from_config(cls, specs: "dict[str, dict]") -> "MCPRegistry":
        """Build (not connect) a registry from mcp.config.discover_mcp_servers().

        A spec that can't be used becomes a _FailedClient carrying the reason,
        so one bad entry no longer takes every other server down with it.
        """
        registry = cls()
        registry.specs = dict(specs)
        for name, spec in specs.items():
            registry._clients[name] = _make_client(name, spec)
        return registry

    @classmethod
    def from_mcp_json(cls, path: Path) -> "MCPRegistry":
        """Registry from one .mcp.json file only (kept for callers/tests)."""
        from greenboost_cli.mcp.config import _normalise, _read_json
        registry_specs: dict[str, dict] = {}
        for name, cfg in _read_json(path).get("mcpServers", {}).items():
            spec = _normalise(name, cfg, str(path), path.parent)
            if spec is not None:
                registry_specs[name] = spec
        registry = cls.from_config(registry_specs)
        registry._mcp_json_path = path
        return registry

    def connect_all(self) -> dict[str, bool]:
        """Connect every server in parallel and register each as it finishes.

        Serial connects meant ten servers cost the SUM of their start-up times
        and a hung one cost forever. Now the wall time is the slowest single
        server (bounded by INIT_TIMEOUT_S) and tools become visible per server.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results: dict[str, bool] = {}
        # raw MCP tool name -> server that first registered it, so a second
        # server offering the same bare name is caught rather than silently
        # producing a second ambiguous prefixed entry.
        if not hasattr(self, "_seen_raw_names"):
            self._seen_raw_names = {}
        clients = list(self._clients.items())
        if not clients:
            return results

        def _connect(item):
            name, client = item
            t0 = time.monotonic()
            try:
                ok = bool(client.connect())
            except Exception as e:          # a crashing connect() is one failed server
                ok = False
                try:
                    client.error = f"{type(e).__name__}: {e}"[:200]
                except Exception:
                    pass
            return name, client, ok, int((time.monotonic() - t0) * 1000)

        with ThreadPoolExecutor(max_workers=min(len(clients), 12),
                                thread_name_prefix="gb-mcp-connect") as pool:
            futures = [pool.submit(_connect, it) for it in clients]
            for fut in as_completed(futures):
                name, client, ok, ms = fut.result()
                results[name] = ok
                if ok:
                    with self._reg_lock:
                        self._register_client(name, client)
                _emit_connect(name, client, ok, ms)
        return results

    def _register_client(self, name: str, client) -> None:
        builtins = _builtin_tool_names()
        seen_raw_names = self._seen_raw_names
        for tool in client.tools:
            raw_name = tool["name"]

            # A raw name that collides with a builtin (Bash, Write,
            # …) is never registered — not even under its prefixed
            # form. Without this, _normalize_tool_name()'s bare-name
            # suffix match could resolve a model's plain "Bash" call
            # to this MCP tool's implementation instead of the
            # builtin (orchestrator.py checks mcp_registry.has_tool()
            # BEFORE falling back to dispatch()'s safety-classified
            # Bash handler) — the model would see one tool named
            # "Bash" in its schema list but a completely different,
            # unclassified implementation would actually run.
            # Excluding it here makes that shadow structurally
            # unreachable rather than relying on nobody ever
            # registering a colliding server.
            if raw_name in builtins:
                self.collisions.append(
                    f"MCP server '{name}' offers a tool named "
                    f"'{raw_name}', which collides with a builtin "
                    f"instrument of the same name — excluded from "
                    f"the registry."
                )
                continue

            if raw_name in seen_raw_names and seen_raw_names[raw_name] != name:
                # Informational, NOT a warning, and deliberately not
                # phrased as "will fail".
                #
                # Every tool is advertised to the model ONLY under its
                # `mcp__<server>__<tool>` name (see `prefixed` below),
                # so the model is never handed the bare name and has no
                # route into the ambiguous case from its own tool list.
                # If it invents the bare name anyway, call_tool() now
                # answers with the two exact names to use instead.
                #
                # The old text said the call "will fail" and told the
                # reader to use the prefixed form — advice aimed at
                # someone who is not making the call, about a failure
                # that does not occur, printed with a ⚠ at every single
                # startup. Two servers sharing a tool name is ordinary
                # at 238 tools across ten servers.
                self.duplicate_names.append(
                    f"'{raw_name}' is offered by both "
                    f"'{seen_raw_names[raw_name]}' and '{name}'; both "
                    f"stay available as mcp__<server>__{raw_name}. "
                    f"Nothing is disabled and no action is needed."
                )
            else:
                seen_raw_names[raw_name] = name

            # Advertise under the Claude-Code-style mcp__<server>__
            # <tool> name — the model already knows this convention
            # from pretraining/RAG text and emits it unprompted
            # (confirmed live: mcp__knowledge-rag__search_knowledge
            # against a registry that only knew the bare
            # "search_knowledge" and fell through to "Unknown
            # instrument"). Also prevents two servers exposing the
            # same bare tool name from colliding in _tool_to_server.
            prefixed = f"mcp__{name}__{raw_name}"
            schema = dict(tool)
            schema["name"] = prefixed
            # MCP tools/list returns inputSchema (camelCase); every
            # builtin tool schema and schemas_to_openai_functions()
            # index input_schema (snake_case) — without this an MCP
            # tool's schema had NO argument names on the native-FC
            # path (KeyError) and an empty <parameters> block on the
            # injection path.
            if "inputSchema" in schema and "input_schema" not in schema:
                schema["input_schema"] = schema.pop("inputSchema")
            self._tool_to_server[prefixed] = name
            self._raw_tool_name[prefixed] = raw_name
            self.tool_schemas.append(schema)


    def add_server(self, name: str, spec: dict, init_timeout: float | None = None) -> bool:
        """Add one server to a running registry and connect it (no restart). Replaces a
        same-named one. Returns whether it connected; the reason is on the client's `.error`."""
        client = _make_client(name, spec, init_timeout)
        if name in self._clients:
            try:
                self._unregister(name)
                self._clients[name].close()
            except Exception:
                pass
        self._clients[name] = client
        self.specs[name] = spec
        return self.reconnect(name)

    def remove_server(self, name: str) -> bool:
        """Close one server and forget it (tools included). False when it isn't loaded."""
        client = self._clients.pop(name, None)
        if client is None:
            return False
        self._unregister(name)
        self.specs.pop(name, None)
        self.dormant_servers.discard(name)
        try:
            client.close()
        except Exception:
            pass
        return True

    def _unregister(self, name: str) -> None:
        """Drop one server's tools from the routing tables (before a reconnect)."""
        with self._reg_lock:
            gone = {k for k, s in self._tool_to_server.items() if s == name}
            for k in gone:
                self._tool_to_server.pop(k, None)
                self._raw_tool_name.pop(k, None)
            self.tool_schemas = [s for s in self.tool_schemas if s.get("name") not in gone]
            self._seen_raw_names = {r: s for r, s in self._seen_raw_names.items() if s != name}

    def reconnect(self, name: str) -> bool:
        """Restart one server and re-register its tools. Never raises."""
        client = self._clients.get(name)
        if client is None:
            return False
        self._unregister(name)
        try:
            client.close()
        except Exception:
            pass
        t0 = time.monotonic()
        try:
            ok = bool(client.connect())
        except Exception as e:
            ok = False
            client.error = f"{type(e).__name__}: {e}"[:200]
        if ok:
            with self._reg_lock:
                self._register_client(name, client)
        _emit_connect(name, client, ok, int((time.monotonic() - t0) * 1000))
        return ok

    def _normalize_tool_name(self, tool_name: str) -> str | None:
        """Resolve whatever the model called into our internal
        mcp__<server>__<tool> key: exact match, a bare tool name (unique
        suffix match), or a "<server>__<tool>" missing only the mcp__
        prefix. Keeps older sessions / hand-typed names working alongside
        the new prefixed advertising."""
        if tool_name in self._tool_to_server:
            return tool_name
        suffix = f"__{tool_name}"
        matches = [k for k in self._tool_to_server if k.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]
        if not tool_name.startswith("mcp__"):
            candidate = f"mcp__{tool_name}"
            if candidate in self._tool_to_server:
                return candidate
        return None

    def ambiguous_matches(self, tool_name: str) -> list[str]:
        """Prefixed names a bare tool name could mean, when it could mean more than one."""
        if tool_name in self._tool_to_server or tool_name.startswith("mcp__"):
            return []
        matches = sorted(k for k in self._tool_to_server
                         if k.endswith(f"__{tool_name}"))
        return matches if len(matches) > 1 else []

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        key = self._normalize_tool_name(tool_name)
        # AE-5. Record when the name the model emitted is not the name it was
        # shown. Small models fail tool use predominantly by schema
        # misalignment , they emit tool and parameter names that look right
        # because they resemble something seen in pretraining, rather than the
        # schema in front of them (arXiv 2510.07248). This repo has already
        # hit that once (the model calling `mcp__knowledge-rag__search_
        # knowledge` against a registry that only knew the bare name) and
        # patched the single instance. Without a counter there is no way to
        # know whether that was a one-off or the normal case, and no way to
        # tell whether renaming the schemas later helped.
        if key != tool_name:
            _emit_schema_miss(tool_name, key, self)
        if key is None:
            # An ambiguous name is not an unknown one, and saying "unknown"
            # leaves the caller with nothing to try next. With 238 tools across
            # ten servers, two offering `optimize_mesh` is ordinary — name the
            # candidates so the very next call can be right.
            alts = self.ambiguous_matches(tool_name)
            if alts:
                return (f"ERROR: '{tool_name}' is offered by "
                        f"{len(alts)} servers, so the bare name is ambiguous. "
                        f"Call one of these exact names instead: "
                        + ", ".join(alts))
            return f"ERROR: Unknown MCP tool '{tool_name}'"
        server = self._tool_to_server[key]
        raw_name = self._raw_tool_name.get(key, key)
        # The prompt carries name-only schemas once hundreds of tools are
        # connected, so the model often calls with {} and retries the same
        # empty call until the loop guard ends the turn. Answer a missing
        # required argument here, with the real parameter list, instead.
        params = self._params_help(key)
        missing = [r for r in (params or {}).get("required", [])
                   if not isinstance(arguments, dict) or r not in arguments]
        if missing:
            return (f"ERROR: {tool_name} is missing required argument(s): "
                    f"{', '.join(missing)}. Parameters: {params['help']}. "
                    f"Call it again with them filled in.")
        out = self._clients[server].call_tool(raw_name, arguments)
        if out.startswith("Error executing tool") and "validation error" in out and params:
            out += f"\nParameters: {params['help']}"
        return _cap_browser_dump(server, raw_name, out)

    def _params_help(self, key: str) -> "dict | None":
        """Full parameter list of one tool from its original schema (not the
        name-only form the prompt may carry): {'required': [...], 'help': str}."""
        with self._reg_lock:
            schema = next((s for s in self.tool_schemas if s.get("name") == key), None)
        sch = (schema or {}).get("input_schema") or {}
        props = sch.get("properties") or {}
        if not props:
            return None
        required = list(sch.get("required") or [])
        bits = []
        for pname, p in props.items():
            p = p if isinstance(p, dict) else {}
            desc = " ".join(str(p.get("description", "")).split())[:80]
            bits.append(f"{pname} ({p.get('type', 'any')}"
                        f"{', required' if pname in required else ''})"
                        + (f" {desc}" if desc else ""))
        return {"required": required, "help": "; ".join(bits)[:1500]}

    def has_tool(self, tool_name: str) -> bool:
        return self._normalize_tool_name(tool_name) is not None

    def close_all(self):
        """Close every server at once: ten servers closed one after another took
        up to 3 s each, with the terminal frozen on the exit screen."""
        clients = [c for c in self._clients.values() if getattr(c, "connected", False)
                   or getattr(c, "_proc", None) is not None]
        if not clients:
            return          # already closed (an atexit call after /exit finds nothing to do)
        threads = []
        for c in clients:
            th = threading.Thread(target=c.close, daemon=True, name="gb-mcp-close")
            th.start()
            threads.append(th)
        deadline = time.monotonic() + 6.0
        for th in threads:
            th.join(timeout=max(0.0, deadline - time.monotonic()))
        for c in clients:           # whatever finished is marked closed; idempotent from here
            if getattr(c, "_proc", None) is not None and c._proc.poll() is not None:
                c._proc = None

    def status(self) -> list[dict]:
        return [
            {
                "server": name,
                "connected": c.connected,
                "tools": len(c.tools),
                "names": [t["name"] for t in c.tools],
                "error": getattr(c, "error", "") or "",
                "source": (self.specs.get(name) or {}).get("source", ""),
            }
            for name, c in self._clients.items()
        ]

    def server_names(self) -> list[str]:
        return list(self._clients.keys())

    @staticmethod
    def load_servers_config(path: Path) -> dict[str, dict]:
        """Read raw server definitions from .mcp.json without connecting."""
        try:
            return json.loads(path.read_text()).get("mcpServers", {})
        except Exception:
            return {}


def discover_mcp_json(cwd: Path | None = None) -> Path | None:
    """Nearest .mcp.json from cwd up to home (see mcp.config for the full merge)."""
    from greenboost_cli.mcp.config import find_mcp_json
    return find_mcp_json(cwd)


#: Browser servers whose snapshot / network / console / screenshot tools return
#: whole-page dumps. Claude Code gates these with a PreToolUse hook
#: (~/.claude/hooks/mcp-result-gate.py) and sends them through a summarizer;
#: a 20k-char dump is most of a small local model's window, so they get a much
#: tighter ceiling here and a pointer to the narrower call.
_BROWSER_SERVERS = ("chrome-devtools", "browsermcp", "charlotte", "playwright")
_BROWSER_DUMP_TOOLS = re.compile(
    r"snapshot|screenshot|network|console|observe|heapsnapshot|trace|lighthouse", re.I)
BROWSER_DUMP_MAX_CHARS = 6_000


def _cap_browser_dump(server: str, tool: str, out: str) -> str:
    if (server in _BROWSER_SERVERS and _BROWSER_DUMP_TOOLS.search(tool)
            and len(out) > BROWSER_DUMP_MAX_CHARS):
        dropped = len(out) - BROWSER_DUMP_MAX_CHARS
        return (out[:BROWSER_DUMP_MAX_CHARS] +
                f"\n\n[... {server} {tool} output cut at {BROWSER_DUMP_MAX_CHARS:,} chars, "
                f"{dropped:,} dropped. Page dumps flood a small context: use a "
                f"narrower tool (find/query a specific element, a limit or "
                f"filter argument) instead of re-running this one ...]")
    return out


def _emit_connect(name: str, client, ok: bool, ms: int) -> None:
    """One dataflux event per server connect attempt. Never raises."""
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "kind": "cli_mcp_connect",
            "status": "ok" if ok else "error",
            "server": name,
            "n_tools": len(getattr(client, "tools", []) or []),
            "ms": ms,
            "error": (getattr(client, "error", "") or "")[:200],
        })
    except Exception:
        pass


#: MCP result ceiling on a large context window. `_ctx_char_budget` shrinks
#: this against the LIVE served window exactly as it does for builtin results;
#: this constant is only the "box that can afford it" upper bound. Chosen to
#: match Grep/Semble's 20_000 rather than Read's 30_000: an MCP call is far
#: more likely to return a dense JSON blob than a source file a human asked
#: for line-by-line.
MCP_RESULT_MAX_CHARS = 20_000


def _emit_schema_miss(requested: str, resolved, registry=None) -> None:
    """One dataflux event per name the model got wrong. Never raises."""
    try:
        import gb_dataflux
        if resolved is None:
            outcome = "unresolved"
        elif not requested.startswith("mcp__") and str(resolved).startswith("mcp__"):
            outcome = "rescued_bare_name"
        else:
            outcome = "rescued_other"
        gb_dataflux.emit({
            "kind": "agent_tool_schema_miss",
            "status": "error" if outcome == "unresolved" else "ok",
            "requested": requested[:120],
            "resolved": (str(resolved) if resolved else "")[:120],
            "outcome": outcome,
            "known_tools": len(getattr(registry, "_tool_to_server", ()) or ()),
        })
    except Exception:
        pass


def _extract_content(content: list) -> str:
    """Flatten an MCP tool result, bounded the way builtin results are bounded.

    Builtin instrument results have been capped to ~15% of the live context
    window since the NemoClaw audit (`handlers.py::_ctx_char_budget`). MCP
    results were not: they went into history whole. With a couple of hundred
    MCP tools connected, one `dataflux_events(limit=200)` or a broad
    `search_knowledge` could spend the entire window of a small local model in
    a single call, and the next turn would compact , which on this hardware
    costs a cold prefix and, measured over 14 days of real sessions, a jump in
    time-to-first-token from ~5.5s to ~166s.

    One server's verbosity must not be able to do that to a session.
    """
    parts: list[str] = []
    for c in content:
        if isinstance(c, dict):
            if c.get("type") == "text":
                parts.append(c["text"])
            elif c.get("type") == "image":
                url = c.get("url", "")
                parts.append(f"[image: {url or '(embedded)'}]")
            else:
                parts.append(str(c))
        else:
            parts.append(str(c))
    out = "\n".join(parts) if parts else "(empty result)"
    try:
        from greenboost_cli.instruments.handlers import _ctx_char_budget
        cap = _ctx_char_budget(MCP_RESULT_MAX_CHARS)
    except Exception:
        cap = MCP_RESULT_MAX_CHARS
    if len(out) > cap:
        dropped = len(out) - cap
        out = (out[:cap] +
               f"\n\n[... MCP result truncated at {cap:,} chars, "
               f"{dropped:,} dropped. Narrow the call (a smaller limit, a "
               f"filter, or a more specific query) rather than re-running "
               f"it unchanged ...]")
    return out

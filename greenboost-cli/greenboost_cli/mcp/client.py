"""
Lightweight MCP client (Model Context Protocol).

Supports stdio (subprocess JSON-RPC newline-delimited) and HTTP transports.
Auto-discovers .mcp.json from cwd and manages multiple server connections.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import urllib.error
import urllib.request
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


class MCPStdioClient:
    """MCP client that speaks to a subprocess over stdin/stdout (newline-delimited JSON)."""

    def __init__(self, name: str, command: list[str], env: dict | None = None):
        self.name = name
        self._cmd = command
        self._extra_env = env or {}
        self._proc: subprocess.Popen | None = None
        self._id = 0
        self._lock = threading.Lock()
        self.tools: list[dict] = []
        self.connected = False

    def connect(self) -> bool:
        env = {**os.environ, **self._extra_env}
        try:
            self._proc = subprocess.Popen(
                self._cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=False,
                env=env,
                # bufsize=0 (unbuffered), NOT 1. Line buffering has no meaning
                # on a binary pipe, and Python 3.14 says so out loud — it emits
                # a RuntimeWarning per stream, per server, straight through the
                # startup banner (four lines mid-banner with two MCP servers
                # configured, more with more). Nothing was gained by it either:
                # _request() and _notify() both flush() after every write, which
                # is what actually guarantees delivery.
                bufsize=0,
            )
        except (FileNotFoundError, PermissionError, OSError):
            return False

        resp = self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "greenboost-cli", "version": "1.0"},
        })
        if not resp or "result" not in resp:
            self.close()
            return False

        self._notify("notifications/initialized", {})

        resp = self._request("tools/list", {})
        if resp and "result" in resp:
            self.tools = resp["result"].get("tools", [])

        self.connected = True
        return True

    def call_tool(self, name: str, arguments: dict) -> str:
        resp = self._request("tools/call", {"name": name, "arguments": arguments})
        if resp is None:
            return f"ERROR: MCP server '{self.name}' did not respond"
        if "error" in resp:
            return f"ERROR: {resp['error'].get('message', str(resp['error']))}"
        return _extract_content(resp.get("result", {}).get("content", []))

    def close(self):
        if self._proc:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self.connected = False

    def _request(self, method: str, params: dict) -> dict | None:
        with self._lock:
            self._id += 1
            msg_id = self._id
            msg = json.dumps({
                "jsonrpc": "2.0", "id": msg_id, "method": method, "params": params,
            })
            try:
                self._proc.stdin.write((msg + "\n").encode("utf-8"))
                self._proc.stdin.flush()

                # BoundedLineDecoder caps how much a server's output can
                # accumulate without a line break (item 8: a misbehaving
                # server that never emits a newline must not grow this
                # buffer without bound).
                max_line_size = 1024 * 1024
                response_obj = None

                def on_line(line: str) -> None:
                    nonlocal response_obj
                    line = line.strip()
                    if not line:
                        return
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        return
                    if "id" in obj and obj["id"] == msg_id:
                        response_obj = obj
                    # Skip notifications (no matching id)

                decoder = BoundedLineDecoder(max_line_size, on_line)
                for _ in range(50):
                    if response_obj is not None:
                        break
                    chunk = self._proc.stdout.read(4096)
                    if not chunk:
                        decoder.end()
                        break
                    decoder.write(chunk)
                return response_obj
            except Exception:
                return None

    def _notify(self, method: str, params: dict):
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        try:
            self._proc.stdin.write((msg + "\n").encode("utf-8"))
            self._proc.stdin.flush()
        except Exception:
            pass


class MCPHttpClient:
    """MCP client over HTTP (streamable-HTTP JSON-RPC transport)."""

    def __init__(self, name: str, url: str):
        self.name = name
        self._url = url
        self._id = 0
        self.tools: list[dict] = []
        self.connected = False

    def connect(self, timeout: float = 5.0) -> bool:
        resp = self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "greenboost-cli", "version": "1.0"},
        }, timeout=timeout)
        if not resp or "result" not in resp:
            return False

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

    def _request(self, method: str, params: dict, timeout: float = 30.0) -> dict | None:
        self._id += 1
        body = json.dumps({
            "jsonrpc": "2.0", "id": self._id, "method": method, "params": params,
        }).encode()
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                text = raw.decode("utf-8", errors="replace")
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
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return None


class MCPRegistry:
    """Manages multiple MCP clients, exposes merged tool schemas, routes calls."""

    def __init__(self):
        self._clients: dict[str, MCPStdioClient | MCPHttpClient] = {}
        self._tool_to_server: dict[str, str] = {}
        # Advertised (mcp__<server>__<tool>) name -> the RAW tool name the
        # underlying MCP client actually needs for tools/call.
        self._raw_tool_name: dict[str, str] = {}
        self.tool_schemas: list[dict] = []
        self._mcp_json_path: Path | None = None
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
    def from_mcp_json(cls, path: Path) -> "MCPRegistry":
        registry = cls()
        registry._mcp_json_path = path
        try:
            config = json.loads(path.read_text())
        except Exception:
            return registry
        for name, cfg in config.get("mcpServers", {}).items():
            if cfg.get("type") == "http":
                registry._clients[name] = MCPHttpClient(name, cfg["url"])
            else:
                cmd = [cfg["command"]] + cfg.get("args", [])
                registry._clients[name] = MCPStdioClient(name, cmd, cfg.get("env"))
        return registry

    def connect_all(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        builtins = _builtin_tool_names()
        # raw MCP tool name -> server that first registered it, so a second
        # server offering the same bare name is caught rather than silently
        # producing a second ambiguous prefixed entry.
        seen_raw_names: dict[str, str] = {}
        for name, client in self._clients.items():
            ok = client.connect()
            results[name] = ok
            if ok:
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
        return results

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
        return self._clients[server].call_tool(raw_name, arguments)

    def has_tool(self, tool_name: str) -> bool:
        return self._normalize_tool_name(tool_name) is not None

    def close_all(self):
        for client in self._clients.values():
            client.close()

    def status(self) -> list[dict]:
        return [
            {
                "server": name,
                "connected": c.connected,
                "tools": len(c.tools),
                "names": [t["name"] for t in c.tools],
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
    """Walk from cwd up to home looking for .mcp.json."""
    p = cwd or Path.cwd()
    home = Path.home()
    for _ in range(10):
        candidate = p / ".mcp.json"
        if candidate.exists():
            return candidate
        if p == home or p.parent == p:
            break
        p = p.parent
    return None


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

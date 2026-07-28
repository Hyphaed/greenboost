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
                text=True,
                env=env,
                bufsize=1,
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
                self._proc.stdin.write(msg + "\n")
                self._proc.stdin.flush()
                # Read until we get a response matching our id (skip notifications)
                for _ in range(50):
                    line = self._proc.stdout.readline()
                    if not line:
                        return None
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "id" in obj and obj["id"] == msg_id:
                        return obj
                    # Skip notifications (no matching id)
                return None
            except Exception:
                return None

    def _notify(self, method: str, params: dict):
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        try:
            self._proc.stdin.write(msg + "\n")
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
        for name, client in self._clients.items():
            ok = client.connect()
            results[name] = ok
            if ok:
                for tool in client.tools:
                    raw_name = tool["name"]
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

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        key = self._normalize_tool_name(tool_name)
        if key is None:
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


def _extract_content(content: list) -> str:
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
    return "\n".join(parts) if parts else "(empty result)"

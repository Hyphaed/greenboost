"""greenboost-cli cluster peer worker.

A tiny JSON-RPC server that runs on each remote peer (e.g. the laptop "omen")
and is reached by the coordinator via an `ssh -L` tunnel. The server binds
**127.0.0.1 only** by design — exposing it on the LAN would defeat the
SSH-only auth model. To talk to it, the coordinator opens an SSH tunnel:

    ssh -L 127.0.0.1:PORT:127.0.0.1:PORT user@host \
        mamba run -n greenboost-cli \
        python -m greenboost_cli.cluster.peer_worker --port PORT

Wire format:
    Each request is a single JSON object, newline-terminated:
        {"id": "...", "method": "...", "params": {...}}
    Each response is a single JSON object, newline-terminated:
        {"id": "...", "ok": true,  "result": ...}
      or {"id": "...", "ok": false, "error": "..."}

Methods (all soft-fail — they catch their own exceptions and report
`ok=False` with a string `error`, never crash the worker):

    ping()                              -> {"pong": True, "host": "..."}
    vram_stats()                        -> {"gpus": [{name, total_mib, used_mib, free_mib}, ...]}
    load_layers(model, layer_range)     -> {"loaded": True, "model": ..., "layers": [a, b]}
    forward(hidden_states)              -> {"hidden_states": [...]}
    unload()                            -> {"unloaded": True}

The actual ML compute (load_layers / forward) is deliberately stubbed in this
first cut: it acknowledges the request and echoes the input. Wiring it up to
HuggingFace `accelerate` happens once the coordinator side is exercising the
RPC path in real workloads. The shape is stable: handlers receive `params`
(dict) and return `result` (any JSON-serialisable value) or raise to surface
an error.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import socket
import subprocess
import sys
from typing import Any, Awaitable, Callable

from greenboost_cli.core.bounded_lines import BoundedLineDecoder

# Bind only on the loopback interface. NEVER change to 0.0.0.0 — the
# coordinator must reach us through an `ssh -L` tunnel.
LOOPBACK = "127.0.0.1"
DEFAULT_PORT = 9741

# Process-wide state for loaded model shards. Stays in memory across
# requests but resets on worker restart.
_STATE: dict[str, Any] = {
    "model": None,
    "layer_range": None,
    "loaded": False,
}


# ── Handlers ─────────────────────────────────────────────────────────────────

def _h_ping(params: dict) -> dict:
    return {
        "pong": True,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pid": os.getpid(),
    }


def _h_vram_stats(params: dict) -> dict:
    """Use nvidia-smi to enumerate GPUs. Avoids a hard torch dep on the peer."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"gpus": [], "error": f"nvidia-smi: {e}"}

    if proc.returncode != 0:
        return {"gpus": [], "error": (proc.stderr or "nvidia-smi failed").strip()}

    gpus = []
    for line in proc.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append({
                "name":      parts[0],
                "total_mib": int(parts[1]),
                "used_mib":  int(parts[2]),
                "free_mib":  int(parts[3]),
            })
        except ValueError:
            continue
    return {"gpus": gpus}


def _h_load_layers(params: dict) -> dict:
    """Load a real partial transformer shard.

    Required params: {model, layer_range, role}.
    Optional:       {device, dtype}.

    Roles:
      embed_head  → embed + layers[start:end] + norm + lm_head (single-peer mode)
      embed       → embed + layers[start:end]                 (first peer)
      middle      → layers[start:end] only                    (middle peer)
      head        → layers[start:end] + norm + lm_head        (last peer)
    """
    from greenboost_cli.cluster.shard_model import load_shard
    model = params.get("model")
    layer_range = params.get("layer_range")
    role = params.get("role") or "middle"
    device = params.get("device") or "auto"
    dtype = params.get("dtype") or "bfloat16"
    if not model or not isinstance(layer_range, (list, tuple)) or len(layer_range) != 2:
        raise ValueError(
            "load_layers requires {model: str, layer_range: [start, end], role: str}")
    desc = load_shard(model, list(layer_range), role, device=device, dtype=dtype)
    _STATE["model"] = model
    _STATE["layer_range"] = list(layer_range)
    _STATE["role"] = role
    _STATE["loaded"] = True
    return {"loaded": True, **desc}


def _h_start_session(params: dict) -> dict:
    from greenboost_cli.cluster.shard_model import start_session
    sid = params.get("session_id")
    return {"session_id": start_session(sid)}


def _h_end_session(params: dict) -> dict:
    from greenboost_cli.cluster.shard_model import end_session
    sid = params.get("session_id")
    if not sid:
        raise ValueError("end_session requires {session_id}")
    return end_session(sid)


def _h_embed_step(params: dict) -> dict:
    from greenboost_cli.cluster.shard_model import embed_step
    sid = params.get("session_id")
    token_ids = params.get("token_ids")
    if not sid or token_ids is None:
        raise ValueError("embed_step requires {session_id, token_ids: <packed>}")
    return embed_step(sid, token_ids)


def _h_middle_step(params: dict) -> dict:
    from greenboost_cli.cluster.shard_model import middle_step
    sid = params.get("session_id")
    h = params.get("hidden_states")
    if not sid or h is None:
        raise ValueError("middle_step requires {session_id, hidden_states}")
    return middle_step(sid, h)


def _h_head_step(params: dict) -> dict:
    from greenboost_cli.cluster.shard_model import head_step
    sid = params.get("session_id")
    h = params.get("hidden_states")
    if not sid or h is None:
        raise ValueError("head_step requires {session_id, hidden_states}")
    return head_step(sid, h)


def _h_forward(params: dict) -> dict:
    """Generic single-call forward, kept for compatibility with the stub era.

    Routes to embed_step or middle_step or head_step based on `mode`. Prefer
    calling the specific handlers directly.
    """
    mode = params.get("mode") or "middle"
    if mode == "embed":
        return _h_embed_step(params)
    if mode == "head":
        return _h_head_step(params)
    return _h_middle_step(params)


def _h_unload(params: dict) -> dict:
    from greenboost_cli.cluster.shard_model import unload_shard
    out = unload_shard()
    _STATE["model"] = None
    _STATE["layer_range"] = None
    _STATE["role"] = None
    _STATE["loaded"] = False
    return out


def _h_shard_info(params: dict) -> dict:
    from greenboost_cli.cluster.shard_model import get_shard, session_count
    s = get_shard()
    if s is None:
        return {"loaded": False, "sessions": 0}
    return {"loaded": True, "sessions": session_count(), **s.describe()}


HANDLERS: dict[str, Callable[[dict], Any]] = {
    "ping":          _h_ping,
    "vram_stats":    _h_vram_stats,
    "load_layers":   _h_load_layers,
    "start_session": _h_start_session,
    "end_session":   _h_end_session,
    "embed_step":    _h_embed_step,
    "middle_step":   _h_middle_step,
    "head_step":     _h_head_step,
    "forward":       _h_forward,
    "unload":        _h_unload,
    "shard_info":    _h_shard_info,
}


# ── Server ───────────────────────────────────────────────────────────────────

def _dispatch(req: dict) -> dict:
    rid = req.get("id", "")
    method = req.get("method", "")
    params = req.get("params") or {}
    handler = HANDLERS.get(method)
    if handler is None:
        return {"id": rid, "ok": False, "error": f"unknown method: {method}"}
    try:
        result = handler(params)
        return {"id": rid, "ok": True, "result": result}
    except Exception as e:
        return {"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"}


async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    peer = writer.get_extra_info("peername")
    # Defence in depth: refuse anything that isn't loopback. (We already bound
    # to 127.0.0.1, but cheap belt-and-braces in case someone reconfigures the
    # bind host without updating this assertion.)
    if peer and not str(peer[0]).startswith("127."):
        writer.close()
        await writer.wait_closed()
        return

    try:
        # BoundedLineDecoder caps how much unterminated output can accumulate
        # from one client before truncation kicks in (item 8: a client that
        # never sends a newline must not grow this buffer without bound).
        max_line_size = 1024 * 1024
        pending_requests = []

        def on_line(line: str) -> None:
            try:
                req = json.loads(line)
            except json.JSONDecodeError as e:
                resp = {"id": "", "ok": False, "error": f"bad json: {e}"}
            else:
                resp = _dispatch(req)
            pending_requests.append(resp)

        decoder = BoundedLineDecoder(max_line_size, on_line)

        while True:
            chunk = await reader.read(4096)
            if not chunk:
                decoder.end()
                break
            decoder.write(chunk)
            while pending_requests:
                resp = pending_requests.pop(0)
                writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def _serve(port: int) -> None:
    server = await asyncio.start_server(_handle_client, host=LOOPBACK, port=port)
    addr = ", ".join(str(s.getsockname()) for s in server.sockets)
    print(f"greenboost peer_worker listening on {addr}", flush=True)

    # Tie our lifetime to the SSH session: when sshd closes the connection
    # it sends SIGHUP. Default Python ignores SIGHUP, so the worker would
    # leak past the SSH tunnel and hold the port. Install handlers so we
    # exit cleanly on disconnect / TERM.
    import signal
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _shutdown(*_a) -> None:
        stop_event.set()

    for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except (NotImplementedError, RuntimeError):
            pass

    # Also exit if our stdin is closed (parent shell died). asyncio doesn't
    # natively watch stdin EOF cheaply on POSIX, so we read it in a thread.
    def _watch_stdin_eof() -> None:
        try:
            while True:
                data = sys.stdin.readline()
                if data == "":
                    break
        except Exception:
            pass
        try:
            loop.call_soon_threadsafe(stop_event.set)
        except RuntimeError:
            pass

    import threading
    threading.Thread(target=_watch_stdin_eof, daemon=True).start()

    async with server:
        serve_task = asyncio.create_task(server.serve_forever())
        stop_task = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait(
            {serve_task, stop_task}, return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        # Drop any loaded shard so VRAM is freed before exit
        try:
            from greenboost_cli.cluster.shard_model import unload_shard
            unload_shard()
        except Exception:
            pass
        print("greenboost peer_worker shutting down", flush=True)


# ── Entry point ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m greenboost_cli.cluster.peer_worker",
        description=(
            "greenboost-cli cluster peer worker. Binds 127.0.0.1 only — "
            "reach it through an ssh -L tunnel from the coordinator."
        ),
    )
    p.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"TCP port to bind on 127.0.0.1 (default {DEFAULT_PORT}). "
             "The remote always binds 127.0.0.1; never the LAN.",
    )
    args = p.parse_args(argv)

    try:
        asyncio.run(_serve(args.port))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Cluster-aware inference adapter for greenboost-cli.

The `ShardedClient` is the coordinator side of the multi-device sharding
story described in the multi_device blueprint. It:

  1. Validates SSH connectivity to each peer before doing anything.
  2. Starts an `ssh -L 127.0.0.1:PORT:127.0.0.1:PORT` tunnel per peer and
     boots `python -m greenboost_cli.cluster.peer_worker --port PORT` on
     the remote side (preferring `mamba run -n greenboost-cli`, falling
     back to plain python if no conda is available).
  3. Splits the model's transformer blocks across peers based on the VRAM
     ratio (60/40 for an RTX 5070 desktop + 5070 Laptop). If HuggingFace
     `accelerate` is available it uses `infer_auto_device_map` so the same
     shard map is consumable downstream; otherwise it falls back to a
     simple least-busy prompt routing strategy.
  4. Exposes a single `generate(prompt, max_tokens)` API. All RPC flows
     through the SSH tunnel — never the open LAN port.

This module is **opt-in**: nothing in the existing inference router
references it yet. The user wires it in deliberately, once the cluster is
known to be healthy via `gb cluster-status`.
"""
from __future__ import annotations

import atexit
import json
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from greenboost_cli.cluster.config import Peer, test_link


# ── Helpers ──────────────────────────────────────────────────────────────────

def _free_local_port(preferred: int) -> int:
    """Find a free TCP port near `preferred` on 127.0.0.1."""
    for p in (preferred, preferred + 1, preferred + 2, preferred + 3):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    # Last resort: let the OS pick
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _fetch_hub_num_layers(model_id: str, timeout: float = 5.0) -> int | None:
    """Best-effort fetch of `num_hidden_layers` from HuggingFace hub.

    No `transformers` / `torch` dependency — works even when the coordinator
    has a broken ML stack. Tries the public hub URL first; if HF_TOKEN is set
    we send it for gated models. Returns None on any failure.
    """
    import os
    import urllib.request
    import urllib.error
    import json as _json

    if "/" not in model_id:
        return None
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    req = urllib.request.Request(url, headers={"User-Agent": "greenboost-cli"})
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            tp = Path.home() / ".cache" / "huggingface" / "token"
            if tp.exists():
                token = tp.read_text().strip()
        except Exception:
            token = None
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = _json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    n = data.get("num_hidden_layers") or data.get("n_layer")
    return int(n) if n else None


def _remote_python_invocation(env_name: str = "greenboost-cli") -> str:
    """Shell snippet to launch the peer worker on the remote.

    Resolution order (first match wins):
      1. ~/.greenboost_cli/venv/bin/python  — created by `cluster-bootstrap-peer`.
      2. `mamba run -n greenboost-cli`     — if the user manages envs that way.
      3. `conda run -n greenboost-cli`     — fallback.
      4. plain `python3 -m`                — last resort (requires PEP 668 escape).
    """
    return (
        "if [ -x \"$HOME/.greenboost_cli/venv/bin/python\" ]; then "
        "  exec \"$HOME/.greenboost_cli/venv/bin/python\" -m greenboost_cli.cluster.peer_worker --port {PORT}; "
        "elif command -v mamba >/dev/null 2>&1; then "
        f"  exec mamba run -n {env_name} python -m greenboost_cli.cluster.peer_worker --port {{PORT}}; "
        "elif command -v conda >/dev/null 2>&1; then "
        f"  exec conda run -n {env_name} python -m greenboost_cli.cluster.peer_worker --port {{PORT}}; "
        "else "
        "  exec python3 -m greenboost_cli.cluster.peer_worker --port {PORT}; "
        "fi"
    )


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class PeerLink:
    peer: Peer
    local_port: int
    remote_port: int
    ssh_proc: subprocess.Popen | None = None
    vram_total_mib: int = 0
    gpu_name: str = ""

    def close(self) -> None:
        if self.ssh_proc is None:
            return
        try:
            # Close stdin first: this signals EOF to the remote peer_worker
            # which exits cleanly (via its _watch_stdin_eof thread). Without
            # this the remote can outlive the local ssh client and leak the
            # port + GPU memory.
            try:
                if self.ssh_proc.stdin is not None:
                    self.ssh_proc.stdin.close()
            except Exception:
                pass
            self.ssh_proc.terminate()
            try:
                self.ssh_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ssh_proc.kill()
        except Exception:
            pass
        self.ssh_proc = None


@dataclass
class ShardPlan:
    """Layer-wise split of a model across peers."""
    model: str
    total_layers: int
    assignments: list[tuple[str, tuple[int, int]]] = field(default_factory=list)
    # assignments: [(peer_host, (layer_start, layer_end_exclusive)), ...]

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "total_layers": self.total_layers,
            "assignments": [
                {"host": h, "layer_start": s, "layer_end": e}
                for h, (s, e) in self.assignments
            ],
        }


# ── Coordinator ──────────────────────────────────────────────────────────────

class ShardedClient:
    """Coordinator side of the multi-device sharding setup.

    Lifecycle:
        client = ShardedClient(peers, model="meta-llama/Llama-3.1-8B")
        client.start()                  # validates SSH, opens tunnels, boots workers
        out = client.generate("Hello", max_tokens=64)
        client.stop()

    Or use as a context manager:
        with ShardedClient(peers, model=...) as client:
            client.generate(...)
    """

    LOCAL_HOST = "127.0.0.1"

    def __init__(
        self,
        peers: Iterable[Peer],
        model: str,
        local_vram_mib: int = 12 * 1024,
        env_name: str = "greenboost-cli",
        worker_port: int = 9741,
        cluster_extra_mem_gb: int = 0,
        total_layers: int | None = None,
    ):
        self.peers: list[Peer] = list(peers)
        self.model = model
        self.local_vram_mib = local_vram_mib
        self.env_name = env_name
        self.worker_port = worker_port
        self.cluster_extra_mem_gb = cluster_extra_mem_gb
        self.links: list[PeerLink] = []
        self.plan: ShardPlan | None = None
        self._started = False
        self._explicit_total_layers = total_layers
        self._total_layers_cached: int | None = None
        atexit.register(self.stop)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> dict:
        """Validate SSH, open tunnels, boot remote workers, build a shard plan.

        Returns a status dict. Soft-fails: on any error the dict is
        `{"ok": False, "error": "..."}` and no tunnels are left dangling.
        """
        if self._started:
            return {"ok": True, "links": [self._link_summary(l) for l in self.links]}

        if shutil.which("ssh") is None:
            return {"ok": False, "error": "ssh binary not found in PATH"}

        # 1. Probe every peer first; bail before opening any tunnels if one is
        # unreachable.
        for peer in self.peers:
            probe = test_link(peer.host)
            if not probe.get("ok"):
                return {
                    "ok": False,
                    "error": f"peer {peer.host} unreachable: {probe.get('error')}",
                }

        # 2. Open tunnels.
        for peer in self.peers:
            try:
                link = self._open_tunnel(peer)
            except Exception as e:
                self.stop()
                return {"ok": False, "error": f"tunnel to {peer.host} failed: {e}"}

            # Probe via the tunnel to confirm the worker actually came up.
            if not self._wait_for_worker(link, timeout=15.0):
                self.stop()
                return {
                    "ok": False,
                    "error": f"peer_worker on {peer.host} did not respond on tunneled port",
                }

            # Pull VRAM stats so the shard planner knows the split ratio.
            stats = self._rpc(link, "vram_stats")
            if stats.get("ok"):
                gpus = stats.get("result", {}).get("gpus", [])
                if gpus:
                    link.vram_total_mib = gpus[0]["total_mib"]
                    link.gpu_name = gpus[0]["name"]

            self.links.append(link)

        # 3. Build a shard plan.
        self.plan = self._plan_shards()
        self._started = True
        return {
            "ok": True,
            "model": self.model,
            "plan": self.plan.as_dict() if self.plan else None,
            "links": [self._link_summary(l) for l in self.links],
        }

    def stop(self) -> None:
        if not self.links and not self._started:
            return
        for link in self.links:
            link.close()
        self.links = []
        self._started = False

    def __enter__(self):
        result = self.start()
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "ShardedClient.start failed"))
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    # ── Generation ───────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> str:
        """Layer-wise sharded autoregressive generation.

        Pipeline (assumes the shard plan was loaded via `load_shards()`):

          [coordinator]
            tokenize prompt → token_ids
            create session_id per peer
          [loop]
            for each new token (or whole prompt on first step):
              peer[0].embed_step(session_id, ids)         → h
              peer[1].middle_step(session_id, h)          → h
              ...
              peer[N-1].head_step(session_id, h)          → logits
            sample next_token from logits
            ids = [next_token]
          [coordinator]
            decode all sampled tokens → string

        Single-peer fallback (no plan or only one peer): we still need a real
        backend; without one we return a clearly-marked stub the caller can
        treat as "sharding unavailable, please configure".
        """
        if not self._started or not self.links:
            raise RuntimeError("ShardedClient.start() not called or no peers")
        if self.plan is None:
            raise RuntimeError("no shard plan — call load_shards() first")

        # Local single-peer path is intentionally not implemented here; the
        # coordinator's own machine is not a peer in this RPC model. Sharded
        # generation requires at least one remote peer holding head (and
        # ideally another holding embeddings).
        if not self._shards_loaded():
            raise RuntimeError(
                "no shard loaded on peers — call load_shards(model_id) first")

        # Tokenize on coordinator using the same model's tokenizer.
        token_ids = self._tokenize(prompt)

        # Choose pipeline order: peer with role=embed* first; head* last.
        ordered = self._pipeline_order()
        if not ordered:
            raise RuntimeError("could not determine pipeline order for plan")
        first = ordered[0]
        middles = ordered[1:-1] if len(ordered) > 2 else []
        last = ordered[-1] if len(ordered) > 1 else None

        # Start a session on every peer.
        session_id = uuid.uuid4().hex[:12]
        for link in ordered:
            self._rpc(link, "start_session", {"session_id": session_id})

        from greenboost_cli.cluster.shard_model import pack_tensor, unpack_tensor

        # Prepare RNG (used only when temperature > 0)
        if seed is not None:
            import random as _r
            _r.seed(seed)

        output_ids: list[int] = []
        ids = token_ids
        try:
            for step in range(max_tokens):
                # 1. embed_step on first peer with the current ids
                h_packed = self._embed_step(first, session_id, ids)
                # 2. middle_step on intermediate peers
                for link in middles:
                    h_packed = self._middle_step(link, session_id, h_packed)
                # 3. head_step on last peer → logits
                if last is None:
                    # Single-peer pipeline (embed_head): run head on the same peer
                    logits = self._head_step(first, session_id, h_packed)
                else:
                    # Last peer needs the hidden states from the previous step
                    h_packed = self._middle_step(last, session_id, h_packed)
                    logits = self._head_step(last, session_id, h_packed)

                # Sample
                next_id = self._sample(logits, temperature=temperature, top_p=top_p)
                output_ids.append(next_id)
                ids = [[next_id]]  # batch of 1, seq of 1 for next step
                # Stop on EOS if known
                if self._is_eos(next_id):
                    break
        finally:
            for link in ordered:
                try:
                    self._rpc(link, "end_session", {"session_id": session_id})
                except Exception:
                    pass

        return self._detokenize(output_ids)

    # ── Sharded-generate helpers ─────────────────────────────────────────────

    def _shards_loaded(self) -> bool:
        for link in self.links:
            info = self._rpc(link, "shard_info")
            if not info.get("ok") or not info.get("result", {}).get("loaded"):
                return False
        return True

    def _pipeline_order(self) -> list[PeerLink]:
        """Order peers by layer_start so hidden states flow start→end."""
        infos = []
        for link in self.links:
            info = self._rpc(link, "shard_info").get("result", {})
            infos.append((info.get("layer_start", 0), link, info))
        infos.sort(key=lambda x: x[0])
        return [link for _, link, _ in infos]

    def _embed_step(self, link: PeerLink, session_id: str, ids) -> dict:
        from greenboost_cli.cluster.shard_model import pack_tensor
        import numpy as np
        arr = np.asarray(ids, dtype=np.int64)
        packed = {
            "shape": list(arr.shape),
            "dtype": "int64",
            "b64": __import__("base64").b64encode(arr.tobytes()).decode("ascii"),
        }
        resp = self._rpc(
            link, "embed_step",
            {"session_id": session_id, "token_ids": packed},
        )
        if not resp.get("ok"):
            raise RuntimeError(f"embed_step failed: {resp.get('error')}")
        return resp["result"]["hidden_states"]

    def _middle_step(self, link: PeerLink, session_id: str, h_packed: dict) -> dict:
        resp = self._rpc(
            link, "middle_step",
            {"session_id": session_id, "hidden_states": h_packed},
        )
        if not resp.get("ok"):
            raise RuntimeError(f"middle_step failed: {resp.get('error')}")
        return resp["result"]["hidden_states"]

    def _head_step(self, link: PeerLink, session_id: str, h_packed: dict) -> dict:
        resp = self._rpc(
            link, "head_step",
            {"session_id": session_id, "hidden_states": h_packed},
        )
        if not resp.get("ok"):
            raise RuntimeError(f"head_step failed: {resp.get('error')}")
        return resp["result"]["logits"]

    def _sample(self, logits_packed: dict, *, temperature: float, top_p: float) -> int:
        import numpy as np
        import base64
        raw = base64.b64decode(logits_packed["b64"])
        dtype = logits_packed.get("dtype", "float16")
        arr = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(logits_packed["shape"])
        # logits shape: (batch=1, vocab)
        v = arr[0].astype(np.float32)
        if temperature <= 0.0:
            return int(np.argmax(v))
        v = v / max(temperature, 1e-5)
        # softmax
        v = v - v.max()
        p = np.exp(v); p = p / p.sum()
        if 0.0 < top_p < 1.0:
            order = np.argsort(-p)
            csum = np.cumsum(p[order])
            mask = csum <= top_p
            mask[0] = True  # always keep top-1
            keep = order[mask]
            p2 = np.zeros_like(p); p2[keep] = p[keep]; p2 /= p2.sum()
            p = p2
        return int(np.random.choice(len(p), p=p))

    def _tokenize(self, prompt: str) -> list[list[int]]:
        tok = self._tokenizer()
        ids = tok(prompt, return_tensors=None, add_special_tokens=True)["input_ids"]
        # Normalise to nested list-of-lists for batch=1
        if ids and isinstance(ids[0], int):
            ids = [ids]
        return ids

    def _detokenize(self, ids: list[int]) -> str:
        tok = self._tokenizer()
        return tok.decode(ids, skip_special_tokens=True)

    def _is_eos(self, token_id: int) -> bool:
        tok = self._tokenizer()
        eid = getattr(tok, "eos_token_id", None)
        return eid is not None and token_id == eid

    def _tokenizer(self):
        if getattr(self, "_tok_cache", None) is None:
            from transformers import AutoTokenizer
            self._tok_cache = AutoTokenizer.from_pretrained(self.model)
        return self._tok_cache

    # ── Shard loading driver ─────────────────────────────────────────────────

    def load_shards(self, *, dtype: str = "bfloat16") -> dict:
        """Issue `load_layers` on every peer per the active shard plan.

        Roles:
          - 1 peer    → embed_head (the whole model on that peer)
          - 2 peers   → embed on the first, head on the second
          - 3+ peers  → embed, middle…, head

        Returns a per-peer status dict.
        """
        if not self._started or self.plan is None:
            raise RuntimeError("ShardedClient not started or no plan")

        link_by_host = {l.peer.host: l for l in self.links}
        assignments = list(self.plan.assignments)
        n = len(assignments)
        results: dict[str, dict] = {}

        for i, (host, (start, end)) in enumerate(assignments):
            link = link_by_host.get(host)
            if link is None:
                continue
            if n == 1:
                role = "embed_head"
            elif i == 0:
                role = "embed"
            elif i == n - 1:
                role = "head"
            else:
                role = "middle"

            resp = self._rpc(link, "load_layers", {
                "model": self.model,
                "layer_range": [start, end],
                "role": role,
                "dtype": dtype,
            }, timeout=600.0)
            results[host] = resp
        return results

    # ── Plan ─────────────────────────────────────────────────────────────────

    def _plan_shards(self) -> ShardPlan:
        """VRAM-proportional split across remote peers only.

        Coordinator-side layer hosting isn't implemented yet, so the
        coordinator is purely an orchestrator. All transformer layers are
        assigned to remote peers, weighted by each peer's effective budget:

            effective_vram = peer_vram_mib + per_peer_extra_mib

        where `per_peer_extra_mib` is `cluster_extra_mem_gb`'s share
        distributed evenly across peers (it represents system-RAM offload
        capacity the peer can use under accelerate's CPU offload).

        Special case: a single remote peer gets the entire model.
        """
        if not self.links:
            raise RuntimeError("no peers; cannot plan shards")

        total_layers = self._resolve_total_layers()
        per_peer_extra_mib = (
            (self.cluster_extra_mem_gb * 1024) // max(1, len(self.links))
        )
        sizes: list[tuple[str, int]] = [
            (link.peer.host, max(link.vram_total_mib + per_peer_extra_mib, 1))
            for link in self.links
        ]

        # Single-peer cluster: the whole model on one peer.
        if len(sizes) == 1:
            return ShardPlan(
                model=self.model,
                total_layers=total_layers,
                assignments=[(sizes[0][0], (0, total_layers))],
            )

        total_vram = sum(v for _, v in sizes)
        assignments: list[tuple[str, tuple[int, int]]] = []
        cursor = 0
        for i, (host, vram) in enumerate(sizes):
            if i == len(sizes) - 1:
                end = total_layers
            else:
                share = int(round(total_layers * vram / total_vram))
                end = min(total_layers, cursor + share)
                if end <= cursor:
                    # Avoid empty shard — give this peer at least one layer
                    end = cursor + 1
            assignments.append((host, (cursor, end)))
            cursor = end
        return ShardPlan(
            model=self.model,
            total_layers=total_layers,
            assignments=assignments,
        )

    def _resolve_total_layers(self) -> int:
        """Look up `num_hidden_layers` from the model config; cached.

        Resolution order:
          1. Explicit `total_layers` passed to the constructor.
          2. transformers.AutoConfig (works if transformers + torch import).
          3. Raw HF hub config.json fetch (works without torch — useful when
             the coordinator's torch is broken or not installed).
          4. Conservative fallback 32.
        """
        if self._total_layers_cached is not None:
            return self._total_layers_cached
        if self._explicit_total_layers:
            self._total_layers_cached = int(self._explicit_total_layers)
            return self._total_layers_cached
        # Path 2: transformers (handles non-HF hub paths too)
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(self.model)
            n = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
            if n:
                self._total_layers_cached = int(n)
                return self._total_layers_cached
        except Exception:
            pass
        # Path 3: hit the HF hub directly for config.json
        n = _fetch_hub_num_layers(self.model)
        if n:
            self._total_layers_cached = int(n)
            return self._total_layers_cached
        # Path 4
        self._total_layers_cached = 32
        return 32

    def _least_busy_link(self) -> PeerLink | None:
        best = None
        best_free = -1
        for link in self.links:
            stats = self._rpc(link, "vram_stats")
            if not stats.get("ok"):
                continue
            gpus = stats.get("result", {}).get("gpus", [])
            free = gpus[0]["free_mib"] if gpus else 0
            if free > best_free:
                best_free = free
                best = link
        return best

    # ── Tunnels ──────────────────────────────────────────────────────────────

    def _open_tunnel(self, peer: Peer) -> PeerLink:
        local_port = _free_local_port(self.worker_port)
        remote_port = self.worker_port

        remote_cmd = _remote_python_invocation(self.env_name).replace("{PORT}", str(remote_port))

        ssh_cmd = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=5",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ExitOnForwardFailure=yes",
            # If the network drops, fail fast — keeps zombie workers from
            # accumulating on the peer when the coordinator crashes.
            "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=2",
            "-T",
            "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
            peer.ssh_target(),
            remote_cmd,
        ]
        # -T disables pty allocation, which keeps stderr clean. The remote
        # peer_worker watches its stdin for EOF and installs SIGHUP/SIGTERM
        # handlers so it dies cleanly when this ssh client disconnects.

        # IMPORTANT: stdin must be a pipe (not DEVNULL) so that closing it on
        # our end propagates EOF to the remote, triggering the worker's
        # stdin-EOF watcher. We won't write anything; just hold the pipe and
        # close it on stop().
        proc = subprocess.Popen(
            ssh_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return PeerLink(peer=peer, local_port=local_port, remote_port=remote_port, ssh_proc=proc)

    def _wait_for_worker(self, link: PeerLink, timeout: float = 15.0) -> bool:
        """Poll the tunneled port until a ping succeeds or we time out."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if link.ssh_proc and link.ssh_proc.poll() is not None:
                return False
            resp = self._rpc(link, "ping", timeout=2.0)
            if resp.get("ok"):
                return True
            time.sleep(0.5)
        return False

    # ── RPC ──────────────────────────────────────────────────────────────────

    def _rpc(self, link: PeerLink, method: str, params: dict | None = None,
             timeout: float = 30.0) -> dict:
        """Send one JSON-RPC call over the tunneled socket. Soft-fails."""
        req = {
            "id": uuid.uuid4().hex[:8],
            "method": method,
            "params": params or {},
        }
        try:
            with socket.create_connection((self.LOCAL_HOST, link.local_port), timeout=timeout) as s:
                s.settimeout(timeout)
                s.sendall((json.dumps(req) + "\n").encode("utf-8"))
                # Read until newline
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
            if not buf:
                return {"ok": False, "error": "empty response"}
            return json.loads(buf.decode("utf-8"))
        except (OSError, socket.timeout) as e:
            return {"ok": False, "error": f"rpc {method}: {e}"}
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"rpc {method} bad json: {e}"}

    # ── Status ───────────────────────────────────────────────────────────────

    def _link_summary(self, link: PeerLink) -> dict:
        return {
            "host": link.peer.host,
            "hostname": link.peer.hostname,
            "local_port": link.local_port,
            "remote_port": link.remote_port,
            "gpu_name": link.gpu_name,
            "vram_total_mib": link.vram_total_mib,
            "alive": (link.ssh_proc is not None and link.ssh_proc.poll() is None),
        }

    def status(self) -> dict:
        return {
            "started": self._started,
            "model": self.model,
            "plan": self.plan.as_dict() if self.plan else None,
            "links": [self._link_summary(l) for l in self.links],
        }

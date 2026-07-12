#!/usr/bin/env python3
"""gb_remote_blocks.py , run tail transformer blocks on a cluster feeder GPU.

GreenBoost cluster orchestration layer (backend of
gb_cluster.offload_tail_blocks , consumers should import gb_cluster, not this
file). Extends the feeder-hosted-stage design (cluster-goals.md priority 1)
from one-shot stages to a PERSISTENT per-step compute split, MODEL-AGNOSTIC:

Any torch model whose forward iterates an nn.ModuleList of uniformly-called
blocks that chain through one tensor (Flux/Flux2 single-stream blocks,
LTX-video transformer blocks, ViT/Qwen-VL vision towers, most DiT variants)
can hand its tail blocks to the feeder. The tail lives in the feeder GPU's
VRAM and executes there on EVERY forward call; only activations cross the
wire (tens of MB per call → sub-second on GbE), while block weights are read
from the feeder's GDDR at full speed , replacing host-T2 PCIe streaming.

Host side (via gb_cluster):
    import gb_cluster
    client = gb_cluster.offload_tail_blocks(
        pipe.transformer, "single_transformer_blocks",
        vram_budget_gb=5.5, chain_arg="hidden_states")
    # the ModuleList is now [local blocks..., one RPC proxy]

Worker side (spawned automatically over SSH, this same file):
    python3 gb_remote_blocks.py --serve --port 9741 --token <hex>

The worker receives the block MODULES pickled (torch.save) , valid because
host and feeder run the identical rsync'd python env, so classes resolve by
import path. Weights arrive exactly as the host holds them (LoRA fusion,
quantization wrappers included). No model-specific code exists on the worker.

Block-call contract: every block is invoked as block(**kwargs) where
kwargs[chain_arg] is replaced by the previous block's output; each block must
return the chained tensor (not a tuple). This matches how diffusers/
transformers iterate uniform block lists in inference mode.

Transport: 8-byte length-prefixed torch.save messages over one TCP
connection. LAN-trust like greenboost-netd; a per-session random token
(passed to the worker's argv over SSH) must match on every message.
offload_tail_blocks() raises BEFORE mutating the model, so callers can catch
and continue fully local.
"""
from __future__ import annotations

import argparse
import gc
import io
import secrets
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

_LEN = struct.Struct(">Q")


# ── framing ───────────────────────────────────────────────────────────────────

def _send_obj(sock: socket.socket, obj) -> None:
    import torch
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    sock.sendall(_LEN.pack(len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    while n:
        c = sock.recv(min(n, 1 << 20))
        if not c:
            raise ConnectionError("peer closed")
        chunks.append(c)
        n -= len(c)
    return b"".join(chunks)


def _recv_obj(sock: socket.socket):
    import torch
    (n,) = _LEN.unpack(_recv_exact(sock, _LEN.size))
    return torch.load(io.BytesIO(_recv_exact(sock, n)),
                      map_location="cpu", weights_only=False)


def _tree_to(obj, device):
    """Move every tensor in a nested list/tuple/dict structure."""
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_tree_to(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: _tree_to(v, device) for k, v in obj.items()}
    return obj


# ── worker (feeder side) ──────────────────────────────────────────────────────

def serve(port: int, token: str) -> None:
    import torch

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Loopback only: the host reaches us through the SSH -L tunnel that
    # spawned this worker , no LAN port to firewall, traffic encrypted.
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    print(f"[gb-blocks] READY port={port}", flush=True)

    blocks: list = []
    chain_arg = "hidden_states"

    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[gb-blocks] client {addr[0]}", flush=True)
        try:
            while True:
                msg = _recv_obj(conn)
                if msg.get("token") != token:
                    _send_obj(conn, {"err": "bad token"})
                    break
                op = msg["op"]

                if op == "init":
                    blocks = []
                    chain_arg = msg.get("chain_arg", "hidden_states")
                    gc.collect()
                    torch.cuda.empty_cache()
                    _send_obj(conn, {"ok": True})
                    print(f"[gb-blocks] init (chain_arg={chain_arg})", flush=True)

                elif op == "load_block":
                    # A whole pickled nn.Module: classes resolve by import
                    # path , host and feeder run the identical rsync'd env.
                    blk = msg["module"].to("cuda").eval()
                    blocks.append(blk)
                    del msg
                    gc.collect()
                    _send_obj(conn, {"ok": True})

                elif op == "ready":
                    torch.cuda.synchronize()
                    free_b, total_b = torch.cuda.mem_get_info(0)
                    used_mb = (total_b - free_b) >> 20
                    _send_obj(conn, {"ok": True, "n": len(blocks),
                                     "vram_used_mb": used_mb})
                    print(f"[gb-blocks] ready , {len(blocks)} blocks, "
                          f"feeder VRAM used {used_mb} MB", flush=True)

                elif op == "run":
                    t0 = time.monotonic()
                    with torch.no_grad():
                        kwargs = _tree_to(msg["kwargs"], "cuda")
                        torch.cuda.synchronize()
                        t1 = time.monotonic()
                        cur = kwargs.pop(chain_arg)
                        for b in blocks:
                            cur = b(**{chain_arg: cur, **kwargs})
                        torch.cuda.synchronize()
                        t2 = time.monotonic()
                        out = cur.to("cpu")
                    _send_obj(conn, {"ok": True, "out": out})
                    print(f"[gb-blocks] run: h2d={t1-t0:.2f}s "
                          f"compute={t2-t1:.2f}s d2h+send={time.monotonic()-t2:.2f}s",
                          flush=True)

                elif op == "ping":
                    _send_obj(conn, {"ok": True})

                elif op == "shutdown":
                    _send_obj(conn, {"ok": True})
                    print("[gb-blocks] shutdown", flush=True)
                    return

                else:
                    _send_obj(conn, {"err": f"unknown op {op!r}"})
        except (ConnectionError, OSError) as e:
            # Client gone = session over. Exit instead of lingering: an
            # orphaned worker would keep ~6 GB of feeder VRAM hostage and
            # starve the next session's TE stage (each session spawns a
            # fresh worker anyway).
            print(f"[gb-blocks] client gone ({e}); exiting", flush=True)
            return
        finally:
            conn.close()


# ── client (host side) ────────────────────────────────────────────────────────

FEEDER_PYTHON = "~/.local/share/mamba/envs/artpipeline_cu13/bin/python"
# aes128-gcm is AES-NI-accelerated on both nodes: 103 MB/s (GbE wire speed)
# vs 53 MB/s with the chacha20 default , measured host↔omen 2026-07-06.
_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             "-o", "StrictHostKeyChecking=no",
             "-c", "aes128-gcm@openssh.com", "-o", "Compression=no"]


class FeederBlockClient:
    """Owns the SSH-spawned worker process and the tensor-RPC connection."""

    def __init__(self, feeder_ip: str, ssh_user: str, port: int = 9741,
                 python: str = FEEDER_PYTHON):
        self.token = secrets.token_hex(16)
        self.tgt = f"{ssh_user}@{feeder_ip}"
        self.ip = feeder_ip
        script = str(Path(__file__).resolve())

        # Push this file to the same path on the feeder (worker == this file).
        push = subprocess.run(
            ["rsync", "-rltO", "-e", "ssh " + " ".join(_SSH_OPTS),
             "-R", script, f"{self.tgt}:/"],
            capture_output=True, text=True)
        if push.returncode != 0:
            raise RuntimeError(f"worker push failed: {push.stderr.strip()}")

        # Fresh worker: kill any stale one on our port first ([g] so the
        # pattern never matches the pkill shell itself).
        subprocess.run(["ssh", *_SSH_OPTS, self.tgt,
                        f"pkill -f '[g]b_remote_blocks.py --serve --port {port}'; true"],
                       capture_output=True)
        # Worker binds 127.0.0.1 on the feeder; -L tunnels host
        # 127.0.0.1:port → feeder 127.0.0.1:port. No open LAN port, no
        # firewall dependency, encrypted; ssh aes-ni easily beats GbE.
        self.proc = subprocess.Popen(
            ["ssh", *_SSH_OPTS, "-L", f"{port}:127.0.0.1:{port}", self.tgt,
             f"{python} {script} --serve --port {port} --token {self.token}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        threading.Thread(target=self._pump_output, daemon=True).start()

        # Connect with retries while the worker imports torch (~5-15 s).
        # ssh -L accepts locally BEFORE the remote side is up, so a bare
        # connect() succeeding proves nothing , require a ping round trip.
        self.sock = None
        self._lock = threading.Lock()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("feeder worker exited during startup")
            try:
                s = socket.create_connection(("127.0.0.1", port), timeout=5)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock = s
                self._rpc({"op": "ping"})
                # create_connection's timeout sticks to the socket , clear it
                # or every RPC recv dies at 5 s (block streaming and the
                # first run's CUDA autotune both legitimately exceed that).
                s.settimeout(None)
                break
            except (OSError, ConnectionError, RuntimeError):
                if self.sock is not None:
                    try:
                        self.sock.close()
                    except OSError:
                        pass
                    self.sock = None
                time.sleep(1.0)
        if self.sock is None:
            # Connect never succeeded - kill the orphaned ssh tunnel + remote
            # worker now; atexit.register(self.close) below hasn't run yet,
            # so without this the feeder-side worker (possibly already
            # holding GPU memory) and the ssh process would leak.
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            raise RuntimeError("could not connect to feeder block worker")
        # Crash-safety: release the feeder GPU even if the host process dies
        # without calling close() (atexit misses only SIGKILL; the worker's
        # exit-on-disconnect covers that remainder).
        import atexit
        atexit.register(self.close)

    def _pump_output(self):
        for line in self.proc.stdout:
            print(f"  [feeder-blocks] {line.rstrip()}", flush=True)

    def _rpc(self, obj):
        obj["token"] = self.token
        with self._lock:
            _send_obj(self.sock, obj)
            resp = _recv_obj(self.sock)
        if "err" in resp:
            raise RuntimeError(f"feeder block worker: {resp['err']}")
        return resp

    def init(self, chain_arg: str = "hidden_states"):
        return self._rpc({"op": "init", "chain_arg": chain_arg})

    def load_block(self, module):
        return self._rpc({"op": "load_block", "module": module})

    def ready(self) -> int:
        return int(self._rpc({"op": "ready"})["vram_used_mb"])

    def run(self, kwargs: dict):
        return self._rpc({"op": "run", "kwargs": _tree_to(kwargs, "cpu")})["out"]

    def close(self):
        try:
            self._rpc({"op": "shutdown"})
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass


def _make_proxy(client: FeederBlockClient, n_remote: int, chain_arg: str):
    from torch import nn

    class _RemoteTail(nn.Module):
        """One module standing in for a contiguous tail block range: one RPC
        executes the whole range on the feeder GPU."""

        def __init__(self):
            super().__init__()
            self._gb_client = None
            self.n_remote = n_remote

        def forward(self, *args, **kwargs):
            if args:  # tolerate positional chain-tensor calls
                kwargs = {chain_arg: args[0], **kwargs}
                if len(args) > 1:
                    raise RuntimeError(
                        "remote tail blocks support at most one positional "
                        "arg (the chained tensor) , pass the rest as kwargs")
            anchor = kwargs[chain_arg]
            t0 = time.monotonic()
            out = self._gb_client.run(kwargs)
            print(f"  [feeder-blocks] rpc {time.monotonic() - t0:.2f}s "
                  f"({tuple(anchor.shape)})", flush=True)
            return out.to(anchor.device, anchor.dtype)

    proxy = _RemoteTail()
    proxy._gb_client = client
    return proxy


def offload_tail_blocks(owner, list_attr: str = "single_transformer_blocks",
                        feeder_ip: str | None = None,
                        ssh_user: str | None = None,
                        vram_budget_gb: float = 5.5,
                        chain_arg: str = "hidden_states",
                        port: int = 9741) -> FeederBlockClient | None:
    """Move the tail of owner.<list_attr> (an nn.ModuleList of uniformly-
    called blocks chaining through kwargs[chain_arg]) onto a feeder GPU.

    feeder_ip/ssh_user default to the first online feeder in cluster.conf.
    Raises before mutating the model on any failure; returns None when the
    budget fits no block. The client is stashed on owner._gb_block_client
    (call .close() to release the feeder, else it dies with the process).
    """
    import os
    import torch
    from torch import nn

    if feeder_ip is None:
        import gb_cluster
        online = [f for f in gb_cluster.feeders(probe=True) if f.online]
        if not online:
            return None
        feeder_ip = online[0].ip
        ssh_user = ssh_user or online[0].ssh_user
    ssh_user = ssh_user or os.environ.get("USER", "root")

    blocks = getattr(owner, list_attr)
    if len(blocks) == 0 or any(isinstance(b, nn.Module) and
                               getattr(b, "n_remote", None) for b in blocks):
        return None  # empty or already offloaded (idempotence guard)
    per_block = sum(p.numel() * p.element_size() for p in blocks[-1].parameters())
    if per_block == 0:
        return None
    n_remote = min(len(blocks) - 1, int(vram_budget_gb * (1 << 30) // per_block))
    if n_remote < 1:
        return None
    k = len(blocks) - n_remote

    print(f"  [feeder-blocks] offloading {list_attr}[{k}:{len(blocks)}] "
          f"({n_remote} × {per_block >> 20} MB) to {feeder_ip}...", flush=True)
    client = FeederBlockClient(feeder_ip, ssh_user, port=port)
    try:
        client.init(chain_arg=chain_arg)
        for j in range(k, len(blocks)):
            client.load_block(blocks[j])
            print(f"  [feeder-blocks] block {j} streamed "
                  f"({j - k + 1}/{n_remote})", flush=True)
        used_mb = client.ready()
    except Exception:
        client.close()
        raise

    proxy = _make_proxy(client, n_remote, chain_arg)
    setattr(owner, list_attr, nn.ModuleList(list(blocks[:k]) + [proxy]))
    owner._gb_block_client = client
    del blocks
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"  [feeder-blocks] active , {n_remote} blocks resident on feeder "
          f"(feeder VRAM {used_mb} MB); host freed ~{(per_block * n_remote) >> 20} MB",
          flush=True)
    return client


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GreenBoost remote-blocks worker")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=9741)
    ap.add_argument("--token", required=True)
    args = ap.parse_args()
    if not args.serve:
        ap.error("worker mode only: --serve --port N --token T")
    serve(args.port, args.token)

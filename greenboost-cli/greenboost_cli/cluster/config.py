"""Cluster config parsing for greenboost-cli.

Config file format (line-oriented, very simple — no JSON / YAML dep):

    # comments allowed
    cluster_extra_mem_gb = 30
    host:port hostname ssh_user

The first form sets a global (`key = value`). The second form registers a peer.

Search order:
    1. /etc/greenboost/cluster.conf   (system-wide, may require sudo to write)
    2. ~/.greenboost_cli/cluster.conf (user fallback)

`save_config` writes to whichever location is writable: prefers the system
file if it exists and is writable; otherwise falls back to the user file.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from greenboost_cli.environment.settings import GB_HOME

SYS_CONFIG = Path("/etc/greenboost/cluster.conf")
USER_CONFIG = GB_HOME / "cluster.conf"


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class Peer:
    host: str
    port: int
    hostname: str
    ssh_user: str

    def ssh_target(self) -> str:
        return f"{self.ssh_user}@{self.host}"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClusterConfig:
    peers: list[Peer] = field(default_factory=list)
    cluster_extra_mem_gb: int = 0
    source: str = ""  # path the config was loaded from (empty if defaults)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "cluster_extra_mem_gb": self.cluster_extra_mem_gb,
            "peers": [p.as_dict() for p in self.peers],
        }

    def find(self, host_or_hostname: str) -> Peer | None:
        for p in self.peers:
            if p.host == host_or_hostname or p.hostname == host_or_hostname:
                return p
        return None


# ── Path helpers ─────────────────────────────────────────────────────────────

def config_path() -> Path:
    """Return the *primary* config path the loader would read first.

    The actual loader merges SYS_CONFIG with USER_CONFIG (user overrides system
    for globals; peer lists are unioned). This helper exists for display only.
    """
    if SYS_CONFIG.exists():
        return SYS_CONFIG
    return USER_CONFIG


def _writable_path() -> Path:
    """Return a path we can write to, preferring the system file."""
    if SYS_CONFIG.exists() and os.access(SYS_CONFIG, os.W_OK):
        return SYS_CONFIG
    USER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    return USER_CONFIG


def _read_one(path: Path) -> tuple[list[Peer], dict]:
    if not path.exists():
        return [], {}
    try:
        return _parse_lines(path.read_text().splitlines())
    except OSError:
        return [], {}


# ── Parsing / writing ────────────────────────────────────────────────────────

def _parse_lines(lines: Iterable[str]) -> tuple[list[Peer], dict]:
    peers: list[Peer] = []
    globals_: dict = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # key = value form (tolerates whitespace either side of the `=`)
        if "=" in line:
            lhs, rhs = line.split("=", 1)
            lhs = lhs.strip()
            # Reject if LHS still looks like a peer "host:port" with embedded
            # spaces; a valid global key has no whitespace once stripped.
            if lhs and " " not in lhs and "\t" not in lhs:
                globals_[lhs] = rhs.strip()
                continue
        # peer form: host:port hostname ssh_user
        parts = line.split()
        if len(parts) < 3:
            continue
        hostport, hostname, ssh_user = parts[0], parts[1], parts[2]
        if ":" not in hostport:
            continue
        host, port_s = hostport.rsplit(":", 1)
        try:
            port = int(port_s)
        except ValueError:
            continue
        peers.append(Peer(host=host, port=port, hostname=hostname, ssh_user=ssh_user))
    return peers, globals_


def load_config() -> ClusterConfig:
    """Load the cluster config from both SYS and USER files, merging.

    Merge semantics:
      - Globals (e.g. `cluster_extra_mem_gb`): USER overrides SYS.
      - Peers: union by host. USER entries override SYS for the same host.
    Soft-fails to defaults if neither file exists.
    """
    sys_peers, sys_globals = _read_one(SYS_CONFIG)
    user_peers, user_globals = _read_one(USER_CONFIG)

    globals_ = {**sys_globals, **user_globals}

    by_host: dict[str, Peer] = {p.host: p for p in sys_peers}
    for p in user_peers:
        by_host[p.host] = p
    peers = list(by_host.values())

    extra_mem = 0
    try:
        extra_mem = int(globals_.get("cluster_extra_mem_gb", "0"))
    except (TypeError, ValueError):
        extra_mem = 0

    sources = [str(p) for p in (SYS_CONFIG, USER_CONFIG) if p.exists()]
    return ClusterConfig(
        peers=peers,
        cluster_extra_mem_gb=extra_mem,
        source=", ".join(sources),
    )


def save_config(cfg: ClusterConfig) -> Path:
    """Serialize the config back to disk. Returns the path written to."""
    path = _writable_path()
    out = []
    out.append("# greenboost-cli cluster configuration")
    out.append("# Format:")
    out.append("#   key = value           (global setting)")
    out.append("#   host:port hostname user   (peer entry)")
    out.append("")
    if cfg.cluster_extra_mem_gb:
        out.append(f"cluster_extra_mem_gb = {cfg.cluster_extra_mem_gb}")
        out.append("")
    for p in cfg.peers:
        out.append(f"{p.host}:{p.port} {p.hostname} {p.ssh_user}")
    out.append("")
    path.write_text("\n".join(out))
    return path


# ── Mutations ────────────────────────────────────────────────────────────────

def register_peer(host: str, port: int, hostname: str, ssh_user: str) -> bool:
    """Add or replace a peer entry. Returns True on success."""
    cfg = load_config()
    # Replace any existing entry with the same host
    cfg.peers = [p for p in cfg.peers if p.host != host]
    cfg.peers.append(Peer(host=host, port=port, hostname=hostname, ssh_user=ssh_user))
    try:
        save_config(cfg)
        return True
    except OSError:
        return False


def list_peers() -> list[Peer]:
    return load_config().peers


def remove_peer(host: str) -> bool:
    cfg = load_config()
    before = len(cfg.peers)
    cfg.peers = [p for p in cfg.peers if p.host != host and p.hostname != host]
    if len(cfg.peers) == before:
        return False
    try:
        save_config(cfg)
        return True
    except OSError:
        return False


# ── Connectivity probe ───────────────────────────────────────────────────────

def test_link(host: str, timeout: float = 8.0) -> dict:
    """Run `ssh user@host nvidia-smi --query-gpu=...` and parse the result.

    Returns:
        ok=True:   {"ok": True, "host": ..., "hostname": ..., "gpus": [{name, memory_total_mib}, ...]}
        ok=False:  {"ok": False, "host": ..., "error": "<reason>"}
    """
    cfg = load_config()
    peer = cfg.find(host)
    if peer is None:
        return {"ok": False, "host": host, "error": f"peer '{host}' not registered"}

    if shutil.which("ssh") is None:
        return {"ok": False, "host": host, "error": "ssh binary not found in PATH"}

    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
        "-o", "StrictHostKeyChecking=accept-new",
        peer.ssh_target(),
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "host": host, "error": "ssh timed out"}
    except OSError as e:
        return {"ok": False, "host": host, "error": f"ssh launch failed: {e}"}

    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or f"ssh exit {proc.returncode}"
        # Trim noisy fingerprint output to first line
        err = err.splitlines()[0] if err else "ssh failed"
        return {"ok": False, "host": host, "error": err}

    gpus = []
    for line in proc.stdout.strip().splitlines():
        # "NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB"
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        mem = parts[1]
        mib = 0
        try:
            mib = int("".join(ch for ch in mem if ch.isdigit()) or 0)
        except ValueError:
            mib = 0
        gpus.append({"name": name, "memory_total_mib": mib})

    return {
        "ok": True,
        "host": peer.host,
        "hostname": peer.hostname,
        "ssh_user": peer.ssh_user,
        "port": peer.port,
        "gpus": gpus,
    }

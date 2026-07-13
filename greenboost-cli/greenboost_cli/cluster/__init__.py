"""greenboost-cli cluster: multi-device LLM inference sharding over SSH.

Modules:
  - config       : parse / write /etc/greenboost/cluster.conf (Peer dataclass)
  - peer_worker  : asyncio JSON-RPC server bound to 127.0.0.1 on each peer
  - cluster_adapter : (in inference/) tunnels into peer_worker over `ssh -L`
  - cli          : `gb cluster-*` headless subcommands

Constraints:
  - The remote peer_worker MUST bind 127.0.0.1 only. Never expose to the LAN.
  - All RPC flows through the SSH tunnel set up by `cluster_adapter.ShardedClient`.
  - Auth is SSH-only — the in-process RPC speaks plain JSON over localhost.
"""
from greenboost_cli.cluster.config import (
    ClusterConfig,
    Peer,
    load_config,
    save_config,
    register_peer,
    list_peers,
    remove_peer,
    test_link,
    config_path,
)

__all__ = [
    "ClusterConfig", "Peer",
    "load_config", "save_config",
    "register_peer", "list_peers", "remove_peer", "test_link",
    "config_path",
]

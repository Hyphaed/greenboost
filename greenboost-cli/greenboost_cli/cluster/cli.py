"""Headless `gb cluster-*` subcommand handlers.

Each handler:
  - Parses its own argv slice with argparse.
  - Returns an int exit code.
  - Writes either JSON (with --json) or a brief human summary to stdout.
  - Soft-fails: never raises out of the handler. Errors → JSON `{ok: false}`
    or a one-line human message.

Handlers (registered in cli_headless._DISPATCH):
  cluster-status    overall config + per-peer connectivity & VRAM
  cluster-register  add a peer (after a successful test_link)
  cluster-list      list configured peers
  cluster-remove    remove a peer
  cluster-test      run nvidia-smi over SSH for one or all peers
  cluster-serve     foreground / --daemon coordinator runner
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from greenboost_cli.environment.settings import GB_HOME
from greenboost_cli.cluster.config import (
    config_path,
    load_config,
    list_peers,
    register_peer,
    remove_peer,
    test_link,
)


# ── Output helpers ───────────────────────────────────────────────────────────

def _emit_json(payload) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


def _emit_err(msg: str) -> None:
    sys.stderr.write(f"gb cluster: {msg}\n")


# ── cluster-status ───────────────────────────────────────────────────────────

def cmd_cluster_status(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb cluster-status", add_help=True)
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-probe", action="store_true",
                   help="Skip the SSH probe; just print the parsed config.")
    args = p.parse_args(argv)

    cfg = load_config()
    peers_info = []
    total_vram_mib = 0
    reachable = 0

    for peer in cfg.peers:
        entry = {
            "host":     peer.host,
            "port":     peer.port,
            "hostname": peer.hostname,
            "ssh_user": peer.ssh_user,
        }
        if not args.no_probe:
            probe = test_link(peer.host)
            entry["ok"] = bool(probe.get("ok"))
            if probe.get("ok"):
                reachable += 1
                gpus = probe.get("gpus", [])
                entry["gpus"] = gpus
                total_vram_mib += sum(g.get("memory_total_mib", 0) for g in gpus)
            else:
                entry["error"] = probe.get("error", "")
        peers_info.append(entry)

    payload = {
        "ok": True,
        "config_path": str(config_path()),
        "cluster_extra_mem_gb": cfg.cluster_extra_mem_gb,
        "peer_count": len(cfg.peers),
        "reachable":  reachable,
        "total_remote_vram_mib": total_vram_mib,
        "peers": peers_info,
    }

    if args.json:
        _emit_json(payload)
        return 0

    print(f"  Cluster config : {payload['config_path']}")
    print(f"  Extra mem (GB) : {payload['cluster_extra_mem_gb']}")
    print(f"  Peers          : {payload['peer_count']} ({payload['reachable']} reachable)")
    if total_vram_mib:
        print(f"  Total remote VRAM : {total_vram_mib} MiB "
              f"({total_vram_mib // 1024} GB)")
    for entry in peers_info:
        print()
        line = f"  - {entry['hostname']} ({entry['host']}:{entry['port']}) " \
               f"user={entry['ssh_user']}"
        print(line)
        if "ok" in entry:
            if entry["ok"]:
                for g in entry.get("gpus", []):
                    print(f"      gpu: {g.get('name')}  "
                          f"{g.get('memory_total_mib', 0)} MiB")
            else:
                print(f"      ERROR: {entry.get('error', '')}")
    return 0


# ── cluster-register ─────────────────────────────────────────────────────────

def cmd_cluster_register(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb cluster-register", add_help=True)
    p.add_argument("host", help="Peer host or IP")
    p.add_argument("--port", type=int, default=9740,
                   help="Feeder port on the peer (default 9740).")
    p.add_argument("--hostname", required=True, help="Short hostname (e.g. omen).")
    p.add_argument("--ssh-user", required=True, help="SSH user on the peer.")
    p.add_argument("--force", action="store_true",
                   help="Register even if the SSH probe fails.")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    # Pre-register the peer in memory only so test_link can find it.
    # We then commit if the probe succeeds (or --force is set).
    from greenboost_cli.cluster.config import Peer, load_config, save_config
    cfg = load_config()
    existing = cfg.find(args.host)
    if existing is None:
        cfg.peers.append(Peer(host=args.host, port=args.port,
                              hostname=args.hostname, ssh_user=args.ssh_user))
    else:
        existing.port = args.port
        existing.hostname = args.hostname
        existing.ssh_user = args.ssh_user

    # Save the in-memory cfg first so test_link picks the new peer up.
    try:
        save_config(cfg)
    except OSError as e:
        payload = {"ok": False, "error": f"cannot write config: {e}",
                   "config_path": str(config_path())}
        if args.json:
            _emit_json(payload)
        else:
            _emit_err(payload["error"])
        return 1

    probe = test_link(args.host)
    if not probe.get("ok") and not args.force:
        # Roll back: remove the entry we just added.
        remove_peer(args.host)
        payload = {"ok": False, "error": probe.get("error", "unreachable"),
                   "host": args.host}
        if args.json:
            _emit_json(payload)
        else:
            _emit_err(f"SSH probe failed for {args.host}: {payload['error']}")
            _emit_err("Use --force to register anyway.")
        return 1

    payload = {
        "ok": True,
        "host":     args.host,
        "port":     args.port,
        "hostname": args.hostname,
        "ssh_user": args.ssh_user,
        "probe":    probe,
        "config_path": str(config_path()),
    }
    if args.json:
        _emit_json(payload)
    else:
        print(f"  Registered {args.hostname} ({args.host}:{args.port})")
        if probe.get("ok"):
            for g in probe.get("gpus", []):
                print(f"    gpu: {g.get('name')}  "
                      f"{g.get('memory_total_mib', 0)} MiB")
    return 0


# ── cluster-list ─────────────────────────────────────────────────────────────

def cmd_cluster_list(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb cluster-list", add_help=True)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config()
    payload = {
        "ok": True,
        "config_path": str(config_path()),
        "cluster_extra_mem_gb": cfg.cluster_extra_mem_gb,
        "peers": [p_.as_dict() for p_ in cfg.peers],
    }
    if args.json:
        _emit_json(payload)
        return 0
    if not cfg.peers:
        print("  (no peers configured)")
        return 0
    for p_ in cfg.peers:
        print(f"  - {p_.hostname}  {p_.host}:{p_.port}  user={p_.ssh_user}")
    return 0


# ── cluster-remove ───────────────────────────────────────────────────────────

def cmd_cluster_remove(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb cluster-remove", add_help=True)
    p.add_argument("host", help="Peer host, IP, or hostname.")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    ok = remove_peer(args.host)
    payload = {"ok": ok, "host": args.host}
    if not ok:
        payload["error"] = f"peer '{args.host}' not found or config not writable"
    if args.json:
        _emit_json(payload)
    else:
        if ok:
            print(f"  Removed peer '{args.host}'")
        else:
            _emit_err(payload["error"])
    return 0 if ok else 1


# ── cluster-test ─────────────────────────────────────────────────────────────

def cmd_cluster_test(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb cluster-test", add_help=True)
    p.add_argument("host", nargs="?",
                   help="Peer host to test (omit to test all).")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config()
    targets = []
    if args.host:
        peer = cfg.find(args.host)
        if peer is None:
            payload = {"ok": False, "error": f"peer '{args.host}' not registered"}
            if args.json:
                _emit_json(payload)
            else:
                _emit_err(payload["error"])
            return 1
        targets = [peer]
    else:
        targets = cfg.peers

    results = [test_link(t.host) for t in targets]
    overall_ok = all(r.get("ok") for r in results) if results else False

    if args.json:
        _emit_json({"ok": overall_ok, "results": results})
        return 0 if overall_ok else 1

    if not results:
        print("  (no peers to test)")
        return 0
    for r in results:
        host = r.get("host", "?")
        if r.get("ok"):
            gpus = r.get("gpus", [])
            gpu_str = ", ".join(
                f"{g.get('name')} ({g.get('memory_total_mib')} MiB)" for g in gpus
            ) or "no GPU"
            print(f"  OK   {host}  →  {gpu_str}")
        else:
            print(f"  FAIL {host}  →  {r.get('error', '')}")
    return 0 if overall_ok else 1


# ── cluster-serve ────────────────────────────────────────────────────────────

def cmd_cluster_serve(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gb cluster-serve", add_help=True)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B",
                   help="HuggingFace model id to plan shards for.")
    p.add_argument("--daemon", action="store_true",
                   help="Detach into the background via nohup.")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config()
    if not cfg.peers:
        payload = {"ok": False, "error": "no peers configured"}
        if args.json:
            _emit_json(payload)
        else:
            _emit_err(payload["error"])
        return 1

    if args.daemon:
        # Re-exec self under nohup with --daemon stripped. Logs go to
        # ~/.greenboost_cli/cluster_serve.log.
        log_path = GB_HOME / "cluster_serve.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        passthrough = [a for a in argv if a != "--daemon"]
        cmd = ["nohup", sys.executable, "-m", "greenboost_cli", "cluster-serve",
               *passthrough]
        with open(log_path, "ab") as logf:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=logf,
                stderr=logf,
                start_new_session=True,
            )
        payload = {"ok": True, "daemon": True, "pid": proc.pid, "log": str(log_path)}
        if args.json:
            _emit_json(payload)
        else:
            print(f"  Cluster coordinator detached (pid {proc.pid})")
            print(f"  Logs: {log_path}")
        return 0

    # Foreground
    from greenboost_cli.inference.cluster_adapter import ShardedClient
    client = ShardedClient(peers=cfg.peers, model=args.model, cluster_extra_mem_gb=cfg.cluster_extra_mem_gb)
    result = client.start()
    if not result.get("ok"):
        if args.json:
            _emit_json(result)
        else:
            _emit_err(result.get("error", "start failed"))
        return 1
    if args.json:
        _emit_json(result)
    else:
        print(f"  Cluster coordinator running (model={args.model})")
        for link in result.get("links", []):
            print(f"    - {link.get('hostname')} via "
                  f"127.0.0.1:{link.get('local_port')} → "
                  f"127.0.0.1:{link.get('remote_port')}")
        print("  Ctrl-C to stop.")
    try:
        # Idle: caller can later attach via the model router.
        import time
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()
    return 0


# ── cluster-load ─────────────────────────────────────────────────────────────

def cmd_cluster_load(argv: list[str]) -> int:
    """Start the coordinator, load shards on every peer, then exit.

    Useful as a precondition for `cluster-generate`: a long-running serve
    isn't strictly required, but loading the model on every peer is.
    """
    p = argparse.ArgumentParser(prog="gb cluster-load", add_help=True)
    p.add_argument("model",
                   help="HuggingFace model id (e.g. meta-llama/Llama-3.2-3B-Instruct)")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config()
    if not cfg.peers:
        payload = {"ok": False, "error": "no peers configured"}
        _emit_json(payload) if args.json else _emit_err(payload["error"])
        return 1

    from greenboost_cli.inference.cluster_adapter import ShardedClient
    client = ShardedClient(peers=cfg.peers, model=args.model, cluster_extra_mem_gb=cfg.cluster_extra_mem_gb)
    start = client.start()
    if not start.get("ok"):
        _emit_json(start) if args.json else _emit_err(start.get("error", ""))
        client.stop()
        return 1
    try:
        result = client.load_shards(dtype=args.dtype)
    finally:
        # Note: we deliberately keep tunnels open by NOT calling client.stop()
        # — load_shards leaves model state in each peer's process which would
        # be lost on disconnect. Instead, we leave the SSH child processes as
        # children of this script and return; calling stop() would kill them.
        # Callers who want a clean tear-down should run cluster-serve in a
        # daemon and unload via the cluster API.
        # However, in foreground mode (this is a one-shot CLI) we DO need to
        # tell the user the shards are loaded but disconnecting will lose them.
        pass

    payload = {"ok": True, "plan": start.get("plan"), "results": result}
    if args.json:
        _emit_json(payload)
    else:
        print("  Shards loaded:")
        for host, r in result.items():
            ok = r.get("ok")
            res = r.get("result", {}) or {}
            if ok:
                print(f"    ✓  {host}: layers {res.get('layer_start')}-"
                      f"{res.get('layer_end')} role={res.get('role')}")
            else:
                print(f"    ✗  {host}: {r.get('error')}")
        print("\n  Note: shards persist as long as the peer_worker process is alive.")
        print("        Closing the SSH tunnel will lose them — pair with cluster-serve")
        print("        in --daemon mode for persistent setups.")
    # Detach but don't kill children — let cluster-serve own lifecycle
    return 0


# ── cluster-info ─────────────────────────────────────────────────────────────

def cmd_cluster_info(argv: list[str]) -> int:
    """Live cluster snapshot: probe peers, build a candidate shard plan, and
    report each peer's loaded shard state if a coordinator is running.

    Unlike cluster-status (which is config + nvidia-smi over SSH), this command
    actually opens the tunnels, asks each peer_worker for its loaded shard,
    and tears down. Useful for "is anything actually loaded right now?".
    """
    p = argparse.ArgumentParser(prog="gb cluster-info", add_help=True)
    p.add_argument("--model", default="",
                   help="Optional model id to plan against (uses peer VRAM "
                        "to compute a candidate layer split).")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config()
    if not cfg.peers:
        payload = {"ok": True, "peers": [], "note": "no peers configured"}
        _emit_json(payload) if args.json else print("  (no peers configured)")
        return 0

    from greenboost_cli.inference.cluster_adapter import ShardedClient
    client = ShardedClient(
        peers=cfg.peers,
        model=args.model or "meta-llama/Llama-3.2-1B",
        cluster_extra_mem_gb=cfg.cluster_extra_mem_gb,
    )
    start = client.start()
    if not start.get("ok"):
        _emit_json(start) if args.json else _emit_err(start.get("error", ""))
        return 1

    try:
        loaded: dict[str, dict] = {}
        for link in client.links:
            resp = client._rpc(link, "shard_info")
            loaded[link.peer.host] = resp.get("result") if resp.get("ok") else {
                "error": resp.get("error", "")
            }

        payload = {
            "ok": True,
            "model": client.model,
            "cluster_extra_mem_gb": cfg.cluster_extra_mem_gb,
            "candidate_plan": client.plan.as_dict() if client.plan else None,
            "peers": [
                {**client._link_summary(l), "loaded": loaded.get(l.peer.host, {})}
                for l in client.links
            ],
        }
    finally:
        client.stop()

    if args.json:
        _emit_json(payload)
        return 0

    print(f"  Model under plan : {payload['model']}")
    print(f"  Extra mem  (GB)  : {payload['cluster_extra_mem_gb']}")
    plan = payload.get("candidate_plan") or {}
    print(f"  Total layers     : {plan.get('total_layers', '?')}")
    for a in plan.get("assignments", []):
        print(f"    layers {a['layer_start']:>3}-{a['layer_end']:<3}  →  {a['host']}")
    for peer in payload["peers"]:
        print()
        print(f"  - {peer['hostname']} ({peer['host']})  "
              f"VRAM {peer['vram_total_mib']} MiB  alive={peer['alive']}")
        info = peer.get("loaded") or {}
        if info.get("loaded"):
            print(f"      loaded: {info.get('model_id')}  layers "
                  f"{info.get('layer_start')}-{info.get('layer_end')}  "
                  f"role={info.get('role')}  sessions={info.get('sessions', 0)}")
        elif info.get("error"):
            print(f"      ERROR: {info['error']}")
        else:
            print(f"      no shard loaded")
    return 0


# ── cluster-bootstrap-peer ───────────────────────────────────────────────────

def cmd_cluster_bootstrap_peer(argv: list[str]) -> int:
    """rsync the greenboost-cli source to a peer and pip-install it.

    Prereqs on the peer:
      - SSH key already trusts the coordinator (BatchMode=yes works)
      - `python3` on PATH (any 3.11+)
      - `pip` available (we'll bootstrap with ensurepip if not)

    Steps:
      1. rsync this checkout (excluding .git, __pycache__) to
         ~/.greenboost_cli/src on the peer.
      2. `python3 -m pip install --user -e ~/.greenboost_cli/src`
      3. Smoke-test: `python3 -m greenboost_cli.cluster.peer_worker --help`
         on the peer.

    Heavy deps (torch, transformers, accelerate, sentence-transformers) are
    NOT installed by this command — the peer doesn't need them until you
    actually call `cluster-load`. Install them with:
        ssh user@peer 'pip install --user torch transformers accelerate
                       --index-url https://download.pytorch.org/whl/cu128'
    """
    p = argparse.ArgumentParser(prog="gb cluster-bootstrap-peer", add_help=True)
    p.add_argument("host", help="peer host (must already be registered)")
    _pkg_root = str(Path(__file__).parent.parent.parent)
    p.add_argument("--source", default=_pkg_root,
                   help="local greenboost-cli source root to rsync")
    p.add_argument("--remote-dir", default="~/.greenboost_cli/src",
                   help="path on the peer to install into")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config()
    peer = cfg.find(args.host)
    if peer is None:
        payload = {"ok": False, "error": f"peer '{args.host}' not registered"}
        _emit_json(payload) if args.json else _emit_err(payload["error"])
        return 1

    src = Path(args.source).expanduser().resolve()
    if not (src / "greenboost_cli" / "__init__.py").is_file():
        payload = {"ok": False, "error": f"not a greenboost-cli source root: {src}"}
        _emit_json(payload) if args.json else _emit_err(payload["error"])
        return 1

    ssh_target = peer.ssh_target()

    # 0. Make sure the remote directory exists. `rsync --mkpath` would do this
    # but isn't available on older rsync. Cheaper to just mkdir over SSH.
    mkdir_cmd = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", ssh_target,
        f"mkdir -p {args.remote_dir}",
    ]
    mkdir_proc = subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=30)
    if mkdir_proc.returncode != 0:
        err = (mkdir_proc.stderr or "remote mkdir failed").strip()
        payload = {"ok": False, "stage": "mkdir", "error": err}
        _emit_json(payload) if args.json else _emit_err(err)
        return 1

    # 1. rsync
    rsync_cmd = [
        "rsync", "-a", "--delete",
        "--exclude", ".git", "--exclude", "__pycache__",
        "--exclude", "*.egg-info", "--exclude", "dist", "--exclude", "build",
        "-e", "ssh -o BatchMode=yes -o ConnectTimeout=5",
        f"{src}/",
        f"{ssh_target}:{args.remote_dir}/",
    ]
    rsync_proc = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=600)
    if rsync_proc.returncode != 0:
        err = (rsync_proc.stderr or rsync_proc.stdout).strip() or "rsync failed"
        payload = {"ok": False, "stage": "rsync", "error": err}
        _emit_json(payload) if args.json else _emit_err(err)
        return 1

    # 2. Make a venv + pip install -e on the peer.
    # PEP 668 (Debian / Ubuntu 24+) blocks system pip, so we use a venv at
    # ~/.greenboost_cli/venv. This is the path cluster_adapter looks for first.
    remote_venv = "$HOME/.greenboost_cli/venv"
    venv_setup = (
        # Use the user's actual HOME to expand the path in the python install
        # location so we don't sprinkle ~ unquoted across the wire.
        f"set -e; "
        f"if [ ! -x {remote_venv}/bin/python ]; then "
        f"  python3 -m venv --system-site-packages {remote_venv}; "
        f"fi; "
        f"{remote_venv}/bin/pip install --upgrade --quiet pip; "
        f"{remote_venv}/bin/pip install --quiet -e {args.remote_dir}"
    )
    pip_cmd = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", ssh_target,
        venv_setup,
    ]
    pip_proc = subprocess.run(pip_cmd, capture_output=True, text=True, timeout=600)
    if pip_proc.returncode != 0:
        err = (pip_proc.stderr or pip_proc.stdout).strip() or "pip install failed"
        payload = {"ok": False, "stage": "pip", "error": err}
        _emit_json(payload) if args.json else _emit_err(err)
        return 1

    # 3. Smoke test: peer_worker --help (via the venv python)
    smoke_cmd = [
        "ssh", "-o", "BatchMode=yes", ssh_target,
        f"{remote_venv}/bin/python -m greenboost_cli.cluster.peer_worker --help",
    ]
    smoke_proc = subprocess.run(smoke_cmd, capture_output=True, text=True, timeout=30)
    if smoke_proc.returncode != 0:
        err = smoke_proc.stderr.strip() or "smoke test failed"
        payload = {"ok": False, "stage": "smoke", "error": err}
        _emit_json(payload) if args.json else _emit_err(err)
        return 1

    payload = {
        "ok": True,
        "peer": peer.as_dict(),
        "remote_dir": args.remote_dir,
        "smoke_stdout_head": smoke_proc.stdout.splitlines()[:3],
    }
    if args.json:
        _emit_json(payload)
    else:
        print(f"  ✓  {peer.hostname} bootstrapped at {args.remote_dir}")
        print(f"  Next: install heavy deps on the peer for layer-wise inference")
        print(f"      ssh {ssh_target} 'pip install --user torch transformers \\")
        print(f"            accelerate --index-url https://download.pytorch.org/whl/cu128'")
    return 0


# ── cluster-generate ─────────────────────────────────────────────────────────

def cmd_cluster_generate(argv: list[str]) -> int:
    """One-shot end-to-end sharded generation.

    Starts coordinator, loads shards if not loaded, generates, prints output,
    tears down. For repeated runs prefer pairing `cluster-serve --daemon`
    + custom client (Python).
    """
    p = argparse.ArgumentParser(prog="gb cluster-generate", add_help=True)
    p.add_argument("prompt")
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config()
    if not cfg.peers:
        payload = {"ok": False, "error": "no peers configured"}
        _emit_json(payload) if args.json else _emit_err(payload["error"])
        return 1

    from greenboost_cli.inference.cluster_adapter import ShardedClient
    client = ShardedClient(peers=cfg.peers, model=args.model, cluster_extra_mem_gb=cfg.cluster_extra_mem_gb)
    start = client.start()
    if not start.get("ok"):
        _emit_json(start) if args.json else _emit_err(start.get("error", ""))
        return 1

    try:
        # Load shards if peers don't have them. This is idempotent on the
        # peer side (same model_id + role + range = no-op).
        load_result = client.load_shards(dtype=args.dtype)
        for host, r in load_result.items():
            if not r.get("ok"):
                err = f"load_layers failed on {host}: {r.get('error')}"
                _emit_json({"ok": False, "error": err}) if args.json else _emit_err(err)
                return 1

        try:
            text = client.generate(
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )
        except Exception as e:
            err = f"generate failed: {type(e).__name__}: {e}"
            _emit_json({"ok": False, "error": err}) if args.json else _emit_err(err)
            return 1

        payload = {"ok": True, "model": args.model, "output": text}
        if args.json:
            _emit_json(payload)
        else:
            print(text)
        return 0
    finally:
        client.stop()

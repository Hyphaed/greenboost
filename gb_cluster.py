#!/usr/bin/env python3
"""gb_cluster.py , GreenBoost cluster orchestration layer.

Single import point for any consumer pipeline (ai-forge, LTX-2, CursConduir,
gen_art) that wants GreenBoost acceleration WITH or WITHOUT cluster feeders.
Everything degrades gracefully: no cluster.conf, no feeders online, no shim
installed , consumers keep working on whatever tiers exist locally.

Why workloads differ (see workflow/cluster-goals.md):
  * ggml/Ollama kernels are dlsym-resolvable in greenboost-netd and listed in
    /etc/greenboost/kernels.allow → full cluster (feeder T1/T2/T3 + remote
    kernel dispatch) works.
  * PyTorch kernels are non-exported static stubs (thousands of them) →
    remote dispatch is impossible; a kernel touching a feeder-resident tensor
    dies with cudaErrorIllegalAddress. Torch workloads therefore run with
    GREENBOOST_CLUSTER=0 (local T1/T2/T3 only, the proven overflow path).

Usage (consumer side , e.g. ai-forge forge/gpu.py):
    import sys; sys.path.insert(0, "/path/to/greenboost_all/greenboost")
    import gb_cluster

    env = gb_cluster.shim_env(workload="diffusion")          # torch, local tiers
    env = gb_cluster.shim_env(workload="llm")                # ggml, full cluster
    env = gb_cluster.shim_env(workload="diffusion", enabled=False)  # shimless

    if gb_cluster.cluster_available():
        ...  # feeders online , LLM stages get extra VRAM/RAM

Compute orchestration , telemetry-driven work distribution across host +
feeder GPU(s) (the "compute orchestrator" layer, not just memory):
    results = gb_cluster.cluster_map(items, run_local, run_remote)
    q = gb_cluster.ClusterJobQueue(run_local, run_remote); q.submit(item) ...
    env = gb_cluster.feeder_env(feeder, workload="diffusion")  # per-node budget
    snap = gb_cluster.cluster_snapshot()                       # placement inputs

All of the above degrade to plain local execution with zero feeders , the
zero-feeder contract holds for every function in this module.

CLI:
    python3 gb_cluster.py [--llm]     # cluster status snapshot
"""
from __future__ import annotations

import json
import os
import queue
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_DIR))  # for gb_feeder_diag when imported from a consumer

GREENBOOST_SHIM = os.environ.get(
    "GREENBOOST_SHIM", "/usr/local/lib/libgreenboost_cuda.so")

# Proven per-workload shim profiles. Every entry is applied with setdefault so
# an explicit caller env always wins.
#   torch/diffusion: mirrors the validated FLUX/Klein profile (ai-forge,
#   print repo _gen_gb.sh). GREENBOOST_CLUSTER=0 , see module docstring.
#   llm/ggml: cluster stays ON; the shim aggregates feeder tiers into device 0
#   and dispatches kernels to feeders (data-driven, 0xAA00… fake ptrs).
_WORKLOAD_PROFILES: dict[str, dict[str, str]] = {
    "diffusion": {
        "GREENBOOST_ACTIVE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        "GREENBOOST_KV_RESERVE_MB": "0",
        "GREENBOOST_HOST_RAM_SAFETY_MB": "2048",
        "GREENBOOST_A0_DISABLE": "1",
        "GREENBOOST_DISABLE_T2_ON_BLACKWELL": "0",
        "GREENBOOST_CLUSTER": "0",
        "GREENBOOST_KV_COMPRESS": "1",
        # GREENBOOST_CLUSTER=0 above is a documentation-only convention as
        # far as the C shim is concerned — greenboost_cuda_shim.c never
        # actually calls getenv("GREENBOOST_CLUSTER"), so on any host where
        # the shim can reach the cluster fabric (any feeder — verified
        # 2026-07-09 on ai-forge's dispatch to a feeder over
        # tools/cluster_runner/express_gen.py) its cuDeviceTotalMem /
        # cudaGetDeviceProperties hooks aggregate remote feeder VRAM into
        # device 0's reported total regardless of this flag, producing a
        # fake pooled capacity (129 GiB observed on an 8 GiB feeder) that
        # PyTorch/gb-quant then plans allocations against — OOMing
        # immediately even with real VRAM free. GREENBOOST_REPORT_PHYSICAL_
        # VRAM=1 is the flag the shim ACTUALLY reads to disable that
        # inflation; it was added to ai-forge's own forge/gpu.py (host-only)
        # on 2026-07-07 but never back-ported here, so shim_env()/
        # feeder_env() (this module — the ONLY place a feeder dispatch draws
        # its diffusion env from) shipped the fake-VRAM bug to every feeder
        # workload since. Required, not optional, for "diffusion" on any
        # cluster-fabric-reachable host.
        "GREENBOOST_REPORT_PHYSICAL_VRAM": "1",
    },
    "llm": {
        "GREENBOOST_ACTIVE": "1",
        "GREENBOOST_CLUSTER": "1",
    },
}
_WORKLOAD_PROFILES["torch"] = _WORKLOAD_PROFILES["diffusion"]
_WORKLOAD_PROFILES["ggml"] = _WORKLOAD_PROFILES["llm"]

# Torch workloads that spill to T2 also need GREENBOOST_T2_POOL_MB (so PyTorch's
# allocator sizes against the real reserved pool, not the shim default) and
# GREENBOOST_CUDART_PATH (the env's own libcudart, newer than /usr/lib). These
# are dynamic (read from sysfs / discovered per env), not static profile
# entries. Both lived only in ai-forge until now — the same drift class that
# caused the 2026-07-07 fp8 OOM; centralised here so every consumer and every
# feeder dispatch gets them (see workflow/known-issues.md).
_T2_POOL_WORKLOADS = ("diffusion", "torch")


def _local_t2_pool_total_mb() -> int | None:
    """Kernel-reserved T2 DDR pool size in MB, from GreenBoost's pool_brief
    sysfs (format 'T1:11GB T2:29/42GB(69%) T3:0/73GB …'). None on any parse
    failure so callers keep the shim's own default. Lifted from ai-forge
    forge/gpu.py:_t2_pool_total_mb so it has one owner."""
    try:
        brief = Path("/sys/class/greenboost/greenboost/pool_brief").read_text()
    except OSError:
        return None
    import re as _re
    m = _re.search(r"T2:\d+/(\d+)GB", brief)
    return int(m.group(1)) * 1024 if m else None


@dataclass
class Feeder:
    ip: str
    port: int
    hostname: str = ""
    ssh_user: str = ""
    online: bool = False
    t1_free_mb: int = 0
    t1_total_mb: int = 0
    t2_free_mb: int = 0
    t2_total_mb: int = 0
    t3_free_mb: int = 0
    t3_total_mb: int = 0
    error: str = ""
    # Live GPU telemetry (GB_MSG_FEEDER_STATUS v3.1; SSH nvidia-smi fallback).
    # 0 when unavailable (older netd, NVML query failed, or SSH unreachable).
    gpu_util_pct: int = 0
    gpu_temp_c: int = 0
    gpu_power_w: int = 0
    mps_sm_pct: int = 0
    # Adaptive placement inputs, updated by cluster_map()/ClusterJobQueue from
    # real completed work , 0.0 until at least one chunk has run on this node.
    throughput_ewma: float = 0.0   # items/second
    link_mbps_ewma: float = 0.0    # measured rsync/scp transfer speed


CLUSTER_CONF = "/etc/greenboost/cluster.conf"


def _read_cluster_conf() -> list[tuple[str, int, str, str]]:
    """Parse cluster.conf lines: 'IP:PORT [hostname] [ssh_user]'."""
    entries: list[tuple[str, int, str, str]] = []
    try:
        lines = Path(CLUSTER_CONF).read_text().splitlines()
    except OSError:
        return entries  # no cluster.conf , normal when no feeders are configured
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = line.split()
            ip, _, port = parts[0].partition(":")
            entries.append((ip, int(port or 9740),
                            parts[1] if len(parts) > 1 else "",
                            parts[2] if len(parts) > 2 else ""))
        except ValueError:
            print(f"  [cluster-conf] skipping malformed line: {line!r}", flush=True)
    return entries


# Per-node adaptive placement state, keyed by "ip:port" (feeders) or the
# literal "host" (local GPU). Populated from REAL completed work by
# cluster_map()/ClusterJobQueue , not a priori estimates. Process-lifetime
# only (no persistence across runs); a fresh process starts from empty
# history and reverts to even/telemetry-only placement.
_HOST_KEY = "host"
_ewma_lock = threading.Lock()
_throughput_ewma: dict[str, float] = {}   # items/second, higher = faster node
_link_mbps_ewma: dict[str, float] = {}    # measured transfer speed, MB/s


def _ewma_update(store: dict[str, float], key: str, value: float,
                  alpha: float = 0.3) -> float:
    """Exponential moving average update; alpha=0.3 weighs recent chunks
    more heavily so placement reacts to changing load within a session."""
    with _ewma_lock:
        prev = store.get(key)
        new = value if prev is None else alpha * value + (1 - alpha) * prev
        store[key] = new
        return new


# ── Shared SSH transport options (every ssh/rsync/scp in this module) ────────
# One ControlMaster socket per (user, host, port), multiplexed by every
# subsequent ssh/rsync/scp this module spawns: the many small staging calls
# (mkdir, dep probes, manifest pushes, telemetry fallback) stop paying a full
# TCP+KEX+auth handshake each (~400 ms measured to a 2.5GbE feeder) and ride
# the warm connection instead. ControlPersist keeps the master alive 120 s
# after the last client exits, so a dispatch loop's back-to-back calls always
# hit a reused socket.

# Below this size a push is "small" (manifests, JSON, worker scripts): one
# round-trip dominates, so ssh-level Compression=yes is worth it. At or above
# it (model weights, tensors) the payload is high-entropy , zlib just burns
# CPU below the 2.5GbE line rate, so bulk stays Compression=no.
_SMALL_PUSH_BYTES = 4 * 1024 * 1024


def _cm_dir() -> str:
    """Directory for ControlMaster sockets: /run/user/$UID (per-user tmpfs)
    when present, else ~/.ssh. Kept short , AF_UNIX sun_path caps at ~104
    bytes and the %r@%h:%p expansion has to fit."""
    run_dir = f"/run/user/{os.getuid()}"
    return run_dir if os.path.isdir(run_dir) else os.path.expanduser("~/.ssh")


def _ssh_opts(connect_timeout: int = 10, compress: bool = False) -> list[str]:
    """Shared SSH options for EVERY ssh/rsync/scp invocation in this module
    (one owner , call sites must not hand-roll their own list).

    aes128-gcm: AES-NI wire-speed bulk transfers (~2× the chacha20 default).
    """
    return ["-o", "BatchMode=yes", "-o", f"ConnectTimeout={connect_timeout}",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={_cm_dir()}/gb-cm-%r@%h:%p",
            "-o", "ControlPersist=120s",
            "-c", "aes128-gcm@openssh.com",
            "-o", f"Compression={'yes' if compress else 'no'}"]


def _rsync_sent_bytes(stdout: str) -> int | None:
    """Bytes actually sent over the wire, from `rsync --stats` output ,
    delta transfers (repeat model/code pushes) send far less than the file
    sizes, and the link EWMA must reflect real wire traffic."""
    import re
    m = re.search(r"Total bytes sent:\s*([\d,.]+)", stdout or "")
    return int(re.sub(r"[,.]", "", m.group(1))) if m else None


def _emit_link_transfer(feeder: "Feeder", op: str, nbytes: int, seconds: float,
                        status: str = "ok", error: str | None = None) -> None:
    """Record one real host⇄feeder transfer: a dataflux kind="link_transfer"
    event {peer, bytes, seconds, mbps, op} + fold the measured speed into the
    per-node link_mbps_ewma placement input. Best-effort , never raises, so a
    transfer never fails because its measurement couldn't be recorded."""
    try:
        seconds = max(seconds, 1e-6)
        mbps = (nbytes / (1024 * 1024)) / seconds
        # Only bandwidth-meaningful transfers feed the placement EWMA , a
        # tiny manifest push is latency-dominated and would "measure" ~0 MB/s,
        # falsely tripping the _MIN_USEFUL_LINK_MBPS dispatch gate. The
        # dataflux event below still records every transfer regardless.
        if status == "ok" and nbytes >= _SMALL_PUSH_BYTES:
            _ewma_update(_link_mbps_ewma, f"{feeder.ip}:{feeder.port}", mbps)
        import gb_dataflux
        ev = {"node": feeder.hostname or feeder.ip, "label": "gb_cluster",
              "kind": "link_transfer", "peer": f"{feeder.ip}:{feeder.port}",
              "op": op, "bytes": nbytes, "seconds": round(seconds, 3),
              "mbps": round(mbps, 2), "n_items": 1, "items": [op],
              "duration_s": round(seconds, 3), "status": status}
        if error:
            ev["error"] = error[:300]
        gb_dataflux.emit(ev)
    except Exception:
        pass


def _ssh_gpu_telemetry(feeder: Feeder, timeout_s: float = 3.0) -> dict | None:
    """SSH `nvidia-smi` fallback for GPU util/temp/power.

    Used when the fabric's GB_MSG_FEEDER_STATUS telemetry is unavailable
    (older netd without v3.1, or the query itself failed while T1/T2/T3
    memory info still succeeded). Best-effort; None on any failure , never
    raises, so callers can unconditionally attempt this after a fabric miss.
    """
    import subprocess
    user = feeder.ssh_user or os.environ.get("USER", "root")
    tgt = f"{user}@{feeder.ip}"
    try:
        out = subprocess.run(
            ["ssh", *_ssh_opts(connect_timeout=int(timeout_s)), tgt,
             "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw "
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout_s + 2)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        return {
            "gpu_util_pct": int(float(parts[0])),
            "gpu_temp_c": int(float(parts[1])),
            "gpu_power_w": int(float(parts[2])),
        }
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def feeders(probe: bool = True, timeout_s: float = 2.0,
            ssh_fallback_telemetry: bool = False) -> list[Feeder]:
    """Feeders from /etc/greenboost/cluster.conf, optionally probed live.

    Returns [] when no cluster is configured , callers need no special case.

    ssh_fallback_telemetry: when the fabric's GB_MSG_FEEDER_STATUS reply
    carries no GPU util/temp/power (older netd, or the NVML query failed on
    the feeder), fall back to an SSH `nvidia-smi` probe. Off by default ,
    it costs an extra SSH round-trip (~100-300 ms); cluster_snapshot() opts
    in for placement decisions where that cost is worth paying.
    """
    try:
        import gb_feeder_diag as fd
    except ImportError:
        return []

    out: list[Feeder] = []
    for ip, port, host, ssh_user in _read_cluster_conf():
        f = Feeder(ip=ip, port=port, hostname=host, ssh_user=ssh_user)
        if probe:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout_s)
                sock.connect((ip, port))
                fd._reset_seq()
                if not fd._do_psk_auth(sock):
                    raise ValueError("PSK auth failed (cluster.key mismatch/missing)")
                f.hostname = fd.do_handshake(sock)
                mi = fd.query_mem_info(sock)
                if mi:
                    f.t1_free_mb  = mi["t1_free"]  >> 20
                    f.t1_total_mb = mi["t1_total"] >> 20
                    f.t2_free_mb  = mi["t2_free"]  >> 20
                    f.t2_total_mb = mi["t2_total"] >> 20
                    f.t3_free_mb  = mi["t3_free"]  >> 20
                    f.t3_total_mb = mi["t3_total"] >> 20
                # Live GPU telemetry , best-effort, never affects `online`.
                try:
                    fs = fd.query_feeder_status(sock)
                except (OSError, ValueError, EOFError, struct.error):
                    fs = None
                if fs:
                    f.gpu_util_pct = fs["gpu_util_pct"]
                    f.gpu_temp_c   = fs["gpu_temp_c"]
                    f.gpu_power_w  = fs["gpu_power_w"]
                    f.mps_sm_pct   = fs["mps_sm_pct"]
                f.online = True
                sock.close()
            except (OSError, ValueError, EOFError) as e:
                f.error = str(e)
            if f.online and ssh_fallback_telemetry and f.gpu_util_pct == 0 \
               and f.gpu_temp_c == 0 and f.gpu_power_w == 0:
                ssh_tel = _ssh_gpu_telemetry(f)
                if ssh_tel:
                    f.gpu_util_pct = ssh_tel["gpu_util_pct"]
                    f.gpu_temp_c   = ssh_tel["gpu_temp_c"]
                    f.gpu_power_w  = ssh_tel["gpu_power_w"]
        # Adaptive placement history persists across feeders() calls (a fresh
        # Feeder is built above; the EWMA lives in module-level state keyed
        # by ip:port so repeated dispatches keep learning per node).
        f.throughput_ewma = _throughput_ewma.get(f"{ip}:{port}", 0.0)
        f.link_mbps_ewma  = _link_mbps_ewma.get(f"{ip}:{port}", 0.0)
        out.append(f)
    return out


def cluster_available(timeout_s: float = 2.0) -> bool:
    """True when at least one feeder answers the handshake."""
    return any(f.online for f in feeders(probe=True, timeout_s=timeout_s))


def shim_supported() -> bool:
    return Path(GREENBOOST_SHIM).exists()


def shim_env(workload: str = "diffusion", enabled: bool = True,
             base_env: dict[str, str] | None = None,
             cudart_path: str | None = None) -> dict[str, str]:
    """Subprocess env overlay for a GreenBoost-accelerated workload.

    enabled=False returns a SHIMLESS env: GREENBOOST_ACTIVE and any shim entry
    in LD_PRELOAD are stripped (not inherited from a desktop session that
    exports them globally) , gb-quant quantize-to-fit children require this.

    cudart_path: the workload env's own libcudart.so.{13,12}. The consumer
    discovers it (it knows its own venv layout — e.g. ai-forge's ENV_PYTHONS)
    and passes it here; shim_env sets GREENBOOST_CUDART_PATH for torch
    workloads so the shim binds a cudart new enough for modern PyTorch.
    """
    env = dict(base_env if base_env is not None else os.environ)
    if enabled and shim_supported():
        prior = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = f"{GREENBOOST_SHIM}:{prior}" if prior else GREENBOOST_SHIM
        profile = _WORKLOAD_PROFILES.get(workload)
        if profile is None:
            raise ValueError(f"unknown workload {workload!r} , "
                             f"one of {sorted(_WORKLOAD_PROFILES)}")
        for k, v in profile.items():
            env.setdefault(k, v)
        if workload in _T2_POOL_WORKLOADS:
            t2_total = _local_t2_pool_total_mb()
            if t2_total:
                env.setdefault("GREENBOOST_T2_POOL_MB", str(t2_total * 85 // 100))
            if cudart_path:
                env.setdefault("GREENBOOST_CUDART_PATH", cudart_path)
    else:
        env.pop("GREENBOOST_ACTIVE", None)
        preload = [p for p in env.get("LD_PRELOAD", "").split(":")
                   if p and "greenboost" not in p]
        if preload:
            env["LD_PRELOAD"] = ":".join(preload)
        else:
            env.pop("LD_PRELOAD", None)
    return env


FEEDER_STAGE_PYTHON = "~/.local/share/mamba/envs/artpipeline_cu13/bin/python"


def run_stage_on_feeder(argv: list[str],
                        inputs: tuple[str, ...] = (),
                        outputs: tuple[str, ...] = (),
                        feeder: Feeder | None = None,
                        python: str = FEEDER_STAGE_PYTHON,
                        extra_env: dict[str, str] | None = None,
                        timeout_s: int = 3600) -> int:
    """Run one pipeline stage on a feeder GPU over SSH (cluster-goals.md
    'feeder-hosted pipeline stages', priority 1).

    argv:    command after the python interpreter. File paths must be absolute
             and are used verbatim on the feeder , push them via `inputs`.
    inputs:  host files copied to the SAME path on the feeder before the run.
    outputs: feeder files copied back to the same host path after rc==0.
    feeder:  target (default: first online feeder from cluster.conf).
    Returns the remote exit code (0 = success).
    Raises RuntimeError when no feeder is online or transfers fail.
    """
    import shlex
    import subprocess

    if feeder is None:
        online = [f for f in feeders(probe=True) if f.online]
        if not online:
            raise RuntimeError("run_stage_on_feeder: no feeder online")
        feeder = online[0]
    # Self-provisioning gate: verify the feeder's env before dispatch so we
    # never ship a stage to a broken runtime and silently lose it (CLAUDE.md
    # cluster rule). Emits a loud feeder_provision event on failure.
    if os.environ.get("GB_SKIP_FEEDER_READY") != "1" and \
            not ensure_feeder_ready(feeder, code_roots=(), deps=("torch",), python=python):
        raise RuntimeError(
            f"run_stage_on_feeder: feeder {feeder.hostname or feeder.ip} not ready "
            f"(env deps missing at {python}; see dataflux feeder_provision)")
    user = feeder.ssh_user or os.environ.get("USER", "root")
    tgt = f"{user}@{feeder.ip}"
    ssh_opts = _ssh_opts()

    if inputs:
        dirs = sorted({str(Path(p).parent) for p in inputs})
        mkdir = subprocess.run(["ssh", *ssh_opts, tgt,
                                "mkdir -p " + " ".join(shlex.quote(d) for d in dirs)],
                               capture_output=True, text=True)
        if mkdir.returncode != 0:
            raise RuntimeError(f"feeder mkdir failed: {mkdir.stderr.strip()}")
        # rsync -R preserves the absolute path on the feeder side; -rltO (not
        # -a) because -a tries to set times/perms on shared parents like /tmp.
        in_bytes = sum(Path(p).stat().st_size for p in inputs if Path(p).exists())
        push_opts = _ssh_opts(compress=in_bytes < _SMALL_PUSH_BYTES)
        t_push = time.monotonic()
        push = subprocess.run(["rsync", "-rltO", "--stats",
                               "-e", "ssh " + " ".join(push_opts),
                               "-R", *inputs, f"{tgt}:/"],
                              capture_output=True, text=True)
        if push.returncode != 0:
            _emit_link_transfer(feeder, "rsync_push", in_bytes,
                                time.monotonic() - t_push, status="error",
                                error=push.stderr.strip())
            raise RuntimeError(f"feeder input push failed: {push.stderr.strip()}")
        _emit_link_transfer(feeder, "rsync_push",
                            _rsync_sent_bytes(push.stdout) or in_bytes,
                            time.monotonic() - t_push)

    env_prefix = ""
    if extra_env:
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in extra_env.items()) + " "
    remote_cmd = env_prefix + python + " " + " ".join(shlex.quote(a) for a in argv)
    print(f"  [feeder-stage] {feeder.hostname or feeder.ip}: {Path(argv[0]).name} "
          f"({len(inputs)} in, {len(outputs)} out)", flush=True)
    run = subprocess.run(["ssh", *ssh_opts, tgt, remote_cmd], timeout=timeout_s)
    if run.returncode != 0:
        return run.returncode

    for out in outputs:
        t_pull = time.monotonic()
        pull = subprocess.run(["scp", "-q", *ssh_opts, f"{tgt}:{out}", out],
                              capture_output=True, text=True)
        if pull.returncode != 0:
            _emit_link_transfer(feeder, "scp_pull", 0, time.monotonic() - t_pull,
                                status="error", error=pull.stderr.strip())
            raise RuntimeError(f"feeder output pull failed ({out}): {pull.stderr.strip()}")
        try:
            out_bytes = Path(out).stat().st_size
        except OSError:
            out_bytes = 0
        _emit_link_transfer(feeder, "scp_pull", out_bytes, time.monotonic() - t_pull)
    return 0


def _hf_cache_root(hf_home: str | None = None) -> Path:
    """Local HF_HOME/hub directory (falls back to the standard default)."""
    base = hf_home or os.environ.get("HF_HOME")
    return Path(base) / "hub" if base else Path.home() / ".cache" / "huggingface" / "hub"


def _hf_repo_dirname(repo_id: str) -> str:
    """HF hub cache directory name for repo_id, e.g. 'org/model' ->
    'models--org--model'."""
    return "models--" + repo_id.replace("/", "--")


def feeder_has_model(feeder: Feeder, repo_id: str,
                     feeder_hf_home: str | None = None, timeout_s: float = 15.0) -> bool:
    """True when the feeder's HF cache already has a non-empty snapshot for
    repo_id (checked over SSH , no data transferred, just a directory test)."""
    import subprocess
    user = feeder.ssh_user or os.environ.get("USER", "root")
    tgt = f"{user}@{feeder.ip}"
    remote_hub = f"{feeder_hf_home}/hub" if feeder_hf_home else "~/.cache/huggingface/hub"
    dirname = _hf_repo_dirname(repo_id)
    cmd = (f"find {remote_hub}/{dirname}/snapshots -mindepth 1 -maxdepth 1 "
          f"-type d 2>/dev/null | head -1")
    try:
        r = subprocess.run(["ssh", *_ssh_opts(), tgt, cmd],
                          capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False
    return bool(r.stdout.strip())


def ensure_feeder_model(feeder: Feeder, repo_id: str, hf_home: str | None = None,
                        feeder_hf_home: str | None = None, timeout_s: int = 3600) -> bool:
    """Push a HuggingFace model cache to the feeder if it's missing there.

    A feeder can't take a job for a model it doesn't have weights for , and
    re-downloading multi-GB checkpoints from HF on every feeder is slow and
    redundant when the host already has them cached. This rsyncs the whole
    'models--org--repo' cache directory (safetensors + refs + snapshots)
    from the host's HF cache to the feeder's; idempotent (rsync delta), so
    a prior partial/interrupted push resumes cheaply on retry.

    Returns True when the model is present on the feeder after this call
    (already there, or pushed successfully); False when the HOST doesn't
    have it either (nothing to push) or the transfer failed , callers
    should treat False as "this feeder cannot take the job" and fall back
    to local-only, exactly like every other zero-feeder-capable path here.
    """
    import subprocess
    local_dir = _hf_cache_root(hf_home) / _hf_repo_dirname(repo_id)
    if not local_dir.is_dir():
        print(f"  [feeder-model] host has no cache for {repo_id} , cannot push", flush=True)
        return False

    if feeder_has_model(feeder, repo_id, feeder_hf_home):
        return True

    node = feeder.hostname or feeder.ip
    user = feeder.ssh_user or os.environ.get("USER", "root")
    tgt = f"{user}@{feeder.ip}"
    ssh_opts = _ssh_opts()
    remote_hub = f"{feeder_hf_home}/hub" if feeder_hf_home else "~/.cache/huggingface/hub"
    dirname = _hf_repo_dirname(repo_id)
    print(f"  [feeder-model] {repo_id} missing on {node} , "
          f"pushing from host cache...", flush=True)
    t0 = time.monotonic()
    mkdir = subprocess.run(["ssh", *ssh_opts, tgt, f"mkdir -p {remote_hub}/{dirname}"],
                          capture_output=True, text=True)
    if mkdir.returncode != 0:
        print(f"  [feeder-model] mkdir failed: {mkdir.stderr.strip()}", flush=True)
        _df_emit(_new_run_id(), node, "ensure_feeder_model", "model_push", [repo_id],
                 time.monotonic() - t0, status="error", error=mkdir.stderr.strip())
        return False
    # -rlt (not -az): safetensors are high-entropy , rsync-level zlib (-z)
    # caps throughput on one CPU core well below the 2.5GbE line rate.
    push = subprocess.run(["rsync", "-rlt", "--stats",
                          "-e", "ssh " + " ".join(ssh_opts),
                          f"{local_dir}/", f"{tgt}:{remote_hub}/{dirname}/"],
                         capture_output=True, text=True, timeout=timeout_s)
    if push.returncode != 0:
        print(f"  [feeder-model] push failed: {push.stderr.strip()}", flush=True)
        _df_emit(_new_run_id(), node, "ensure_feeder_model", "model_push", [repo_id],
                 time.monotonic() - t0, status="error", error=push.stderr.strip())
        _emit_link_transfer(feeder, "model_push", 0, time.monotonic() - t0,
                            status="error", error=push.stderr.strip())
        return False
    elapsed = time.monotonic() - t0
    _df_emit(_new_run_id(), node, "ensure_feeder_model", "model_push", [repo_id],
             elapsed)
    sent = _rsync_sent_bytes(push.stdout)
    if sent is None:
        try:
            sent = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
        except OSError:
            sent = 0
    _emit_link_transfer(feeder, "model_push", sent, elapsed)
    print(f"  [feeder-model] {repo_id} pushed to {node}", flush=True)
    return True


AIFORGE_ROOT = os.environ.get("GB_AIFORGE_ROOT", os.path.expanduser("~/Dev/ai-forge"))
_feeder_ready_cache: set = set()


def ensure_feeder_ready(feeder: Feeder,
                        code_roots: tuple[str, ...] = (),
                        deps: tuple[str, ...] = ("torch", "diffusers"),
                        model_repo: str | None = None,
                        python: str = FEEDER_STAGE_PYTHON,
                        feeder_hf_home: str | None = None,
                        use_cache: bool = True,
                        timeout_s: int = 3600) -> bool:
    """Self-provisioning readiness gate , make a feeder usable BEFORE dispatch,
    so the cluster NEVER silently falls back to host-only (CLAUDE.md immutable
    rule: drive both GPUs). Three steps, each mirrored to dataflux as a
    `feeder_provision` event so a failure is followable via the MCP:

      1. rsync each code_root host->feeder to the SAME absolute path (idempotent
         delta; excludes .git/.venv/__pycache__/outputs/generated). Pass the
         pipeline source tree(s) that the feeder imports; omit for the express
         bundle path (cluster_map ships its own self-contained bundle).
      2. ensure_feeder_model(model_repo) so weights are present (rsync delta).
      3. verify the feeder's `python` env imports every dep in `deps`.

    Returns True when the feeder is ready to take work. Returns False (with a
    LOUD `feeder_provision` error event + stderr line naming what to provision)
    when it can't be made ready , callers may then fall back to local, but the
    reason is recorded, never silent. Cached per process (keyed on the request)
    so the rsync+probe run once per feeder per run, not per dispatched item."""
    import shlex
    import subprocess

    node = feeder.hostname or feeder.ip
    ck = (feeder.ip, tuple(code_roots), tuple(deps), model_repo)
    if use_cache and ck in _feeder_ready_cache:
        return True
    user = feeder.ssh_user or os.environ.get("USER", "root")
    tgt = f"{user}@{feeder.ip}"
    ssh_opts = _ssh_opts()
    t0 = time.monotonic()

    # 1 · sync pipeline code tree(s) to the same absolute path on the feeder.
    for root in code_roots:
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            continue
        parent = str(Path(root).parent)
        mk = subprocess.run(["ssh", *ssh_opts, tgt, f"mkdir -p {shlex.quote(parent)}"],
                            capture_output=True, text=True)
        push = None
        t_push = time.monotonic()
        if mk.returncode == 0:
            push = subprocess.run(
                ["rsync", "-rltO", "--stats",
                 "-e", "ssh " + " ".join(_ssh_opts(compress=True)),
                 "--exclude", ".git", "--exclude", ".venv",
                 "--exclude", "__pycache__", "--exclude", "*.pyc",
                 "--exclude", "outputs", "--exclude", "generated",
                 f"{root}/", f"{tgt}:{root}/"],
                capture_output=True, text=True, timeout=timeout_s)
            if push.returncode == 0:
                _emit_link_transfer(feeder, "rsync_push",
                                    _rsync_sent_bytes(push.stdout) or 0,
                                    time.monotonic() - t_push)
        if mk.returncode != 0 or push is None or push.returncode != 0:
            err = (mk.stderr or (push.stderr if push else "") or "").strip()
            _df_emit(_new_run_id(), node, "ensure_feeder_ready", "feeder_provision",
                     [root], time.monotonic() - t0, status="error",
                     error=f"code sync {root}: {err}")
            print(f"  [feeder-ready] {node}: code sync FAILED for {root} , {err}",
                  flush=True)
            return False

    # 2 · ensure model weights present on the feeder.
    if model_repo and not ensure_feeder_model(
            feeder, model_repo, feeder_hf_home=feeder_hf_home, timeout_s=timeout_s):
        _df_emit(_new_run_id(), node, "ensure_feeder_ready", "feeder_provision",
                 [model_repo], time.monotonic() - t0, status="error",
                 error=f"model {model_repo} unavailable on feeder")
        return False

    # 3 · verify the feeder python env imports every required dep.
    if deps:
        imports = "; ".join(f"import {d.split('==')[0].split('>')[0]}" for d in deps)
        probe = subprocess.run(
            ["ssh", *ssh_opts, tgt,
             f"{python} -c {shlex.quote(imports + '; print(chr(111)+chr(107))')}"],
            capture_output=True, text=True, timeout=180)
        if probe.returncode != 0 or "ok" not in probe.stdout:
            err = (probe.stderr.strip() or probe.stdout.strip() or "unknown")[:300]
            _df_emit(_new_run_id(), node, "ensure_feeder_ready", "feeder_provision",
                     list(deps), time.monotonic() - t0, status="error",
                     error=f"deps missing in {python}: {err}")
            print(f"  [feeder-ready] {node}: MISSING DEPS in {python} , provision "
                  f"the env (need {', '.join(deps)}). NOT dispatching here. {err}",
                  flush=True)
            return False

    _df_emit(_new_run_id(), node, "ensure_feeder_ready", "feeder_provision",
             list(code_roots) or [node], time.monotonic() - t0)
    if use_cache:
        _feeder_ready_cache.add(ck)
    print(f"  [feeder-ready] {node}: ready (code+model+deps verified)", flush=True)
    return True


def offload_tail_blocks(owner, list_attr: str = "single_transformer_blocks",
                        **kwargs):
    """Move the tail of owner.<list_attr> (uniformly-called transformer
    blocks chaining through one tensor) onto the feeder GPU , the feeder then
    computes on EVERY forward call while only activations cross the wire.

    Model-agnostic: works for Flux/Flux2 single blocks, LTX-video transformer
    blocks, ViT/Qwen-VL vision towers, most DiT variants. Defaults to the
    first online feeder. Returns a FeederBlockClient (call .close() to
    release the feeder) or None when no feeder/budget. Raises before mutating
    the model on failure , catch and continue fully local.

    Lazy re-export of gb_remote_blocks.offload_tail_blocks (torch-heavy;
    keeps gb_cluster importable in torch-free contexts).
    """
    import gb_remote_blocks
    return gb_remote_blocks.offload_tail_blocks(owner, list_attr, **kwargs)


def parallel_map_with_feeder(items, run_local, run_remote=None,
                             feeder: Feeder | None = None):
    """Drain one work queue with host and feeder GPUs concurrently.

    The generic 'both GPUs busy 100% of the time' primitive for multi-item
    workloads (card art batches, video chunks, mesh jobs, prompt lists):
    the host thread and a feeder thread pull items off a shared queue, so
    the faster device naturally takes more work and neither ever idles.
    Zero per-item network sensitivity beyond what run_remote itself does.

    items:      list of work items.
    run_local:  item -> result, executed on the caller's thread (host GPU).
    run_remote: item -> result, executed on a daemon thread against the
                feeder (typically wrapping run_stage_on_feeder). None, or no
                feeder online, degrades to a plain sequential local map.
    Returns results in the original items order.

    Fault model: a run_remote exception re-queues the item for the host
    (feeder loss never loses work , the same job just runs locally);
    a run_local exception propagates (the host is authoritative).
    """
    import queue as _queue
    import threading

    q: "_queue.Queue" = _queue.Queue()
    for i, it in enumerate(items):
        q.put((i, it))
    results: dict[int, object] = {}
    res_lock = threading.Lock()

    use_feeder = run_remote is not None
    if use_feeder and feeder is None:
        online = [f for f in feeders(probe=True) if f.online]
        feeder = online[0] if online else None
        use_feeder = feeder is not None

    def _feeder_loop():
        while True:
            try:
                i, it = q.get_nowait()
            except _queue.Empty:
                return
            try:
                r = run_remote(it)
                with res_lock:
                    results[i] = r
            except Exception as e:
                print(f"  [cluster-queue] feeder failed on item {i} "
                      f"({e.__class__.__name__}: {e}); re-queued for host",
                      flush=True)
                q.put((i, it))
                return  # a failing feeder stops taking work

    t = None
    if use_feeder:
        t = threading.Thread(target=_feeder_loop, daemon=True)
        t.start()

    while True:
        try:
            i, it = q.get(timeout=0.5)
        except _queue.Empty:
            # Queue drained , wait for any in-flight feeder item.
            if t is None or not t.is_alive():
                break
            continue
        r = run_local(it)
        with res_lock:
            results[i] = r
    if t is not None:
        t.join()
    return [results[i] for i in sorted(results)]


# ── Telemetry-driven placement ("the flux brain") ────────────────────────────

_snapshot_cache: dict = {"ts": 0.0, "data": None}
_snapshot_lock = threading.Lock()
_SNAPSHOT_TTL_S = 1.0

# A feeder at/above this GPU utilization (e.g. the user is gaming on it, or
# another job is already running) is skipped for new dispatch , VRAM
# occupancy is a means to higher tok/s, never an end in itself, and neither
# is feeder occupancy: don't contend with work already running there.
_BUSY_UTIL_PCT = 90


def _host_telemetry() -> dict:
    """Best-effort host GPU snapshot for placement decisions.

    Prefers the CURRENT process's own gb_init telemetry singleton (zero
    extra cost , it's already polling in the background); falls back to a
    one-shot gb_telemetry sample for driver processes that haven't
    bootstrapped gb_init (e.g. a thin orchestration script); returns
    all-None fields when neither is importable.
    """
    try:
        import gb_init
        m = gb_init.snapshot()
        if m is not None:
            return {"gpu_util_pct": m.gpu_util_pct, "free_vram_mb": m.fb_free_mb,
                    "temp_c": m.temp_c}
    except Exception:
        pass
    try:
        import gb_telemetry
        m = gb_telemetry.sample_once(0)
        return {"gpu_util_pct": m.gpu_util_pct, "free_vram_mb": m.fb_free_mb,
                "temp_c": m.temp_c}
    except Exception:
        pass
    return {"gpu_util_pct": None, "free_vram_mb": None, "temp_c": None}


def cluster_snapshot(force: bool = False, ssh_fallback_telemetry: bool = True) -> dict:
    """Per-node telemetry snapshot for placement decisions , the input to
    every dispatch policy in this module (host busy? which feeder is
    fastest? is the link worth it?).

    Host metrics come from the running process's own telemetry
    (gb_init/gb_telemetry); feeder metrics come from the fabric
    (GB_MSG_FEEDER_STATUS), with an SSH nvidia-smi fallback for feeders
    running an older netd. Cached ~1s so a hot per-chunk dispatch loop
    doesn't re-probe the network on every iteration.
    """
    now = time.monotonic()
    with _snapshot_lock:
        cached = _snapshot_cache["data"]
        if not force and cached is not None and now - _snapshot_cache["ts"] < _SNAPSHOT_TTL_S:
            return cached

    data = {
        "host": _host_telemetry(),
        "feeders": [asdict(f) for f in
                    feeders(probe=True, ssh_fallback_telemetry=ssh_fallback_telemetry)],
    }
    with _snapshot_lock:
        _snapshot_cache["ts"] = now
        _snapshot_cache["data"] = data
    return data


# A feeder whose MEASURED transfer link is this slow or slower isn't worth
# dispatching to , the rsync/scp round-trip would cost more than the compute
# it saves. A feeder that has never been measured yet (link_mbps_ewma==0.0)
# is NOT skipped here, so every feeder gets a first real chance to build history.
_MIN_USEFUL_LINK_MBPS = 2.0


def _online_feeders_for_dispatch() -> list[Feeder]:
    """Online feeders not already saturated by other work, and not gated out
    by a measured link too slow to be worth the transfer (§ cluster_snapshot,
    _BUSY_UTIL_PCT, _MIN_USEFUL_LINK_MBPS)."""
    snap = cluster_snapshot(ssh_fallback_telemetry=False)
    out = []
    for f in snap["feeders"]:
        if not f["online"] or f["gpu_util_pct"] >= _BUSY_UTIL_PCT:
            continue
        if f["link_mbps_ewma"] and f["link_mbps_ewma"] < _MIN_USEFUL_LINK_MBPS:
            continue
        fdr = Feeder(**f)
        # Self-provisioning gate (CLAUDE.md immutable cluster rule): never
        # silently dispatch to a feeder whose runtime isn't ready. Verify the
        # feeder env imports torch+diffusers (cached per run); a not-ready
        # feeder is dropped with a LOUD feeder_provision dataflux event instead
        # of a silent host-only fallback. Opt out with GB_SKIP_FEEDER_READY=1.
        if os.environ.get("GB_SKIP_FEEDER_READY") != "1" and \
                not ensure_feeder_ready(fdr, code_roots=(), deps=("torch", "diffusers")):
            continue
        out.append(fdr)
    return out


def feeder_env(feeder: Feeder, workload: str = "diffusion",
               extra: dict[str, str] | None = None) -> dict[str, str]:
    """Remote-env overlay for a feeder-hosted stage.

    Replaces the copy-pasted ~45-line inline SSH heredoc that re-derived
    python/cudart/T2-pool env on the feeder in art_wizard.sh , pass this to
    run_stage_on_feeder's extra_env instead.

    Sets a PER-NODE gb-quant VRAM budget (GB_QUANT_BUDGET_GB) from THIS
    feeder's own free T1 , not the host's , so gb_quant.plan_fit's
    bf16>int8>int4 planner picks the right precision for whatever VRAM that
    specific node actually has (host 12 GB and an 8 GB feeder need different
    plans). Also propagates the TurboQuant KV toggle when enabled locally
    (turboquant.enabled / GREENBOOST_TURBOQUANT=1) so KV compression applies
    on every node, not just the host.
    """
    profile = _WORKLOAD_PROFILES.get(workload)
    if profile is None:
        raise ValueError(f"unknown workload {workload!r} , "
                         f"one of {sorted(_WORKLOAD_PROFILES)}")
    env = dict(profile)
    if feeder.t1_free_mb > 0:
        # ~1 GB headroom for CUDA context + activations on the feeder.
        budget_gb = max(1.0, (feeder.t1_free_mb - 1024) / 1024.0)
        env["GB_QUANT_BUDGET_GB"] = f"{budget_gb:.1f}"
    if os.environ.get("GREENBOOST_TURBOQUANT") == "1" or \
       Path("/etc/greenboost/turboquant.enabled").exists():
        env["GREENBOOST_TURBOQUANT"] = "1"
    if extra:
        env.update(extra)
    return env


def stage_bundle(paths: list[str], feeder: Feeder,
                 dest_root: str | None = None) -> None:
    """Rsync a self-contained set of files to the feeder at the SAME
    absolute paths (idempotent , rsync only pushes deltas on repeat calls,
    so callers can stage before every dispatch cheaply).

    Lifts the bundle-staging pattern proven in ai-forge's
    tools/cluster_runner/express_gen.py (manifest + refs + worker scripts +
    the vendored GreenBoost .so preload libs) into the layer so any pipeline
    gets it for free instead of reimplementing the rsync/mkdir dance.

    dest_root is accepted for API symmetry with run_stage_on_feeder's
    same-absolute-path convention; paths are pushed verbatim (relative-path
    rsync mode), so most callers leave it as None.
    """
    import shlex
    import subprocess

    if not paths:
        return
    user = feeder.ssh_user or os.environ.get("USER", "root")
    tgt = f"{user}@{feeder.ip}"
    ssh_opts = _ssh_opts()
    dirs = sorted({str(Path(p).parent) for p in paths})
    mkdir = subprocess.run(["ssh", *ssh_opts, tgt,
                            "mkdir -p " + " ".join(shlex.quote(d) for d in dirs)],
                           capture_output=True, text=True)
    if mkdir.returncode != 0:
        raise RuntimeError(f"stage_bundle mkdir failed: {mkdir.stderr.strip()}")

    t0 = time.monotonic()
    total_bytes = sum(Path(p).stat().st_size for p in paths if Path(p).exists())
    push_opts = _ssh_opts(compress=total_bytes < _SMALL_PUSH_BYTES)
    push = subprocess.run(["rsync", "-rltO", "--stats",
                           "-e", "ssh " + " ".join(push_opts),
                           "-R", *paths, f"{tgt}:/"],
                          capture_output=True, text=True)
    elapsed = max(time.monotonic() - t0, 1e-6)
    if push.returncode != 0:
        _emit_link_transfer(feeder, "rsync_push", total_bytes, elapsed,
                            status="error", error=push.stderr.strip())
        raise RuntimeError(f"stage_bundle push failed: {push.stderr.strip()}")
    _emit_link_transfer(feeder, "rsync_push",
                        _rsync_sent_bytes(push.stdout) or total_bytes, elapsed)
    _df_emit(_new_run_id(), feeder.hostname or feeder.ip, "stage_bundle",
             "stage", paths, elapsed)


_DEFAULT_CHUNK = 4  # proven starting point (ai-forge express_gen); dynamic
                    # queue draining does the actual load-balancing, not this.


def _df_item_repr(item) -> str:
    """Best-effort short label for a work item, for the dataflux event log
    (slug/name/id attr or dict key when present, else a truncated repr)."""
    for attr in ("slug", "name", "id"):
        v = getattr(item, attr, None)
        if v is not None:
            return str(v)
    if isinstance(item, dict):
        for k in ("slug", "name", "id"):
            if k in item:
                return str(item[k])
    return str(item)[:64]


def _df_emit(run_id: str, node: str, label: str, kind: str, items: list,
            duration_s: float, status: str = "ok", error: str | None = None) -> None:
    """Record one dataflux event (gb_dataflux.py). Best-effort , never raises,
    lazily imported so gb_cluster stays import-light when dataflux isn't used."""
    try:
        import gb_dataflux
        ev = {
            "run_id": run_id, "node": node, "label": label, "kind": kind,
            "n_items": len(items),
            "items": [_df_item_repr(x) for x in items[:20]],
            "duration_s": round(duration_s, 3), "status": status,
        }
        if error:
            ev["error"] = error
        gb_dataflux.emit(ev)
    except Exception:
        pass


def _new_run_id() -> str:
    return f"{int(time.time() * 1000)}-{os.getpid()}"


def cluster_map(items, run_local, run_remote=None, *, chunk="auto",
                feeders: "list[Feeder] | None" = None, on_progress=None,
                label: str = "cluster_map"):
    """Drain a shared work queue across the host and every online, non-busy
    feeder , the general 'every GPU busy 100% of the time' primitive
    (generalizes parallel_map_with_feeder to N feeders + chunked batches;
    lifts the proven ai-forge express_gen pattern , dynamic queue, faster
    device naturally takes more work, never split statically , into the
    layer so any pipeline gets it for free).

    items:       list of work items (e.g. image slugs, video chunks).
    run_local:   batch(list) -> list[result], executed on the caller's
                 thread. Receives a SLICE of items (batch, not one item) so
                 hot-pipe workers (load model once, encode/quantize once,
                 loop the batch) amortize their setup cost.
    run_remote:  (feeder, batch(list)) -> list[result], executed on a daemon
                 thread per online feeder. None, or no feeder online/
                 reachable/idle, degrades to a SINGLE local call over ALL
                 items (no feeder to overlap with, so don't chunk for no
                 reason , the zero-feeder contract every function in this
                 module holds to).
    chunk:       "auto" (default: _DEFAULT_CHUNK per feeder-backed chunk,
                 all-items-at-once when running local-only) or a fixed int.
    feeders:     explicit feeder list (default: cluster_snapshot()'s online,
                 non-busy feeders , see _BUSY_UTIL_PCT).
    on_progress: optional callable(completed_count, total_count).
    label:       short name for this workload (e.g. "gen_image_batch.py"),
                 recorded in the dataflux event log (gb_dataflux.py /
                 `greenboost dataflux-ui`) so a human or an LLM can see which
                 script drove a given chunk of cluster activity.

    Returns results in the original items order.

    Fault model (same as parallel_map_with_feeder): a run_remote exception
    re-queues its whole chunk for the host and retires that feeder's thread
    (feeder loss never loses work, it just runs on the host); a run_local
    exception propagates (the host is authoritative).
    """
    n = len(items)
    if n == 0:
        return []
    run_id = _new_run_id()

    use_feeder = run_remote is not None
    if use_feeder:
        if feeders is None:
            feeders = _online_feeders_for_dispatch()
        use_feeder = bool(feeders)

    if not use_feeder:
        chunk_size = n if chunk == "auto" else max(1, int(chunk))
        results: list = []
        for i in range(0, n, chunk_size):
            batch = items[i:i + chunk_size]
            t0 = time.monotonic()
            out = run_local(batch)
            _df_emit(run_id, _HOST_KEY, label, "chunk_local", batch,
                     time.monotonic() - t0)
            results.extend(out)
            if on_progress:
                on_progress(len(results), n)
        return results

    chunk_size = _DEFAULT_CHUNK if chunk == "auto" else max(1, int(chunk))

    q: "queue.Queue" = queue.Queue()
    idx = 0
    while idx < n:
        end = min(idx + chunk_size, n)
        q.put((idx, items[idx:end]))
        idx = end

    results: dict[int, object] = {}
    res_lock = threading.Lock()
    completed = 0
    prog_lock = threading.Lock()

    def _mark_done(count):
        nonlocal completed
        if on_progress:
            with prog_lock:
                completed += count
                on_progress(completed, n)

    def _feeder_worker(feeder: "Feeder"):
        key = f"{feeder.ip}:{feeder.port}"
        node = feeder.hostname or feeder.ip
        while True:
            try:
                start, batch = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.monotonic()
            try:
                out = run_remote(feeder, batch)
                elapsed = max(time.monotonic() - t0, 1e-6)
                _ewma_update(_throughput_ewma, key, len(batch) / elapsed)
                _df_emit(run_id, node, label, "chunk_remote", batch, elapsed)
                with res_lock:
                    for j, r in enumerate(out):
                        results[start + j] = r
                _mark_done(len(batch))
            except Exception as e:
                _df_emit(run_id, node, label, "chunk_remote", batch,
                         time.monotonic() - t0, status="error",
                         error=f"{e.__class__.__name__}: {e}")
                print(f"  [cluster-map] feeder {feeder.hostname or feeder.ip} failed "
                      f"on batch [{start}:{start + len(batch)}] "
                      f"({e.__class__.__name__}: {e}); re-queued for host",
                      flush=True)
                q.put((start, batch))
                return  # a failing feeder stops taking work

    threads = [threading.Thread(target=_feeder_worker, args=(f,), daemon=True)
               for f in feeders]
    for t in threads:
        t.start()

    while True:
        try:
            start, batch = q.get(timeout=0.5)
        except queue.Empty:
            if not any(t.is_alive() for t in threads):
                break
            continue
        t0 = time.monotonic()
        out = run_local(batch)
        elapsed = max(time.monotonic() - t0, 1e-6)
        _ewma_update(_throughput_ewma, _HOST_KEY, len(batch) / elapsed)
        _df_emit(run_id, _HOST_KEY, label, "chunk_local", batch, elapsed)
        with res_lock:
            for j, r in enumerate(out):
                results[start + j] = r
        _mark_done(len(batch))

    for t in threads:
        t.join()
    return [results[i] for i in range(n)]


_JOB_QUEUE_SENTINEL = object()


class ClusterJobQueue:
    """Persistent submit()/close() queue over the same host+feeder dispatch
    policy as cluster_map(), for INCREMENTAL job arrival (e.g. a studio UI
    that queues images one at a time rather than handing cluster_map one
    upfront batch).

    Unlike cluster_map, jobs are dispatched ONE AT A TIME as submitted (no
    chunking , the caller doesn't know the eventual batch size), but the
    placement policy is identical: host + one daemon thread per online,
    non-busy feeder pull from a shared queue, so the faster device
    naturally takes more work. Zero feeders , everything runs on the host
    thread; run_remote=None does the same.
    """

    def __init__(self, run_local, run_remote=None,
                 feeders: "list[Feeder] | None" = None,
                 label: str = "cluster_job_queue"):
        self._run_local = run_local
        self._run_remote = run_remote
        self._q: "queue.Queue" = queue.Queue()
        self._closed = False
        self._label = label
        self._run_id = _new_run_id()

        if run_remote is not None and feeders is None:
            feeders = _online_feeders_for_dispatch()
        self._feeders = feeders or []

        self._threads = [threading.Thread(target=self._host_loop, daemon=True)]
        for f in self._feeders:
            self._threads.append(
                threading.Thread(target=self._feeder_loop, args=(f,), daemon=True))
        for t in self._threads:
            t.start()

    def submit(self, item):
        """Queue one item; returns a concurrent.futures.Future resolved by
        whichever worker (host or a feeder) picks it up."""
        import concurrent.futures
        if self._closed:
            raise RuntimeError("ClusterJobQueue is closed")
        fut = concurrent.futures.Future()
        self._q.put((item, fut))
        return fut

    def _host_loop(self):
        while True:
            item, fut = self._q.get()
            if item is _JOB_QUEUE_SENTINEL:
                return
            if not fut.set_running_or_notify_cancel():
                continue
            t0 = time.monotonic()
            try:
                r = self._run_local(item)
                elapsed = time.monotonic() - t0
                _ewma_update(_throughput_ewma, _HOST_KEY, 1.0 / max(elapsed, 1e-6))
                _df_emit(self._run_id, _HOST_KEY, self._label, "job_local", [item], elapsed)
                fut.set_result(r)
            except Exception as e:
                _df_emit(self._run_id, _HOST_KEY, self._label, "job_local", [item],
                         time.monotonic() - t0, status="error",
                         error=f"{e.__class__.__name__}: {e}")
                fut.set_exception(e)

    def _feeder_loop(self, feeder: "Feeder"):
        key = f"{feeder.ip}:{feeder.port}"
        node = feeder.hostname or feeder.ip
        while True:
            item, fut = self._q.get()
            if item is _JOB_QUEUE_SENTINEL:
                return
            if not fut.set_running_or_notify_cancel():
                continue
            t0 = time.monotonic()
            try:
                r = self._run_remote(feeder, item)
                elapsed = time.monotonic() - t0
                _ewma_update(_throughput_ewma, key, 1.0 / max(elapsed, 1e-6))
                _df_emit(self._run_id, node, self._label, "job_remote", [item], elapsed)
                fut.set_result(r)
            except Exception as e:
                _df_emit(self._run_id, node, self._label, "job_remote", [item],
                         time.monotonic() - t0, status="error",
                         error=f"{e.__class__.__name__}: {e}")
                print(f"  [cluster-queue] feeder {feeder.hostname or feeder.ip} failed "
                      f"({e.__class__.__name__}: {e}); retrying locally", flush=True)
                t1 = time.monotonic()
                try:
                    r = self._run_local(item)
                    _df_emit(self._run_id, _HOST_KEY, self._label, "job_local_retry",
                             [item], time.monotonic() - t1)
                    fut.set_result(r)
                except Exception as e2:
                    _df_emit(self._run_id, _HOST_KEY, self._label, "job_local_retry",
                             [item], time.monotonic() - t1, status="error",
                             error=f"{e2.__class__.__name__}: {e2}")
                    fut.set_exception(e2)
                return  # a failing feeder stops taking work

    def close(self, wait: bool = True):
        """Stop accepting new work; optionally block until in-flight jobs
        (including anything already queued) finish."""
        if self._closed:
            return
        self._closed = True
        for _ in self._threads:
            self._q.put((_JOB_QUEUE_SENTINEL, None))
        if wait:
            for t in self._threads:
                t.join()


def status(probe: bool = True) -> dict:
    """Machine-readable cluster snapshot (safe with no cluster at all)."""
    fs = feeders(probe=probe)
    return {
        "shim_installed": shim_supported(),
        "cluster_configured": bool(fs),
        "feeders_online": sum(1 for f in fs if f.online),
        "feeders": [asdict(f) for f in fs],
        "host": _host_telemetry(),
        "workloads": {
            "llm": "full cluster (feeder T1/T2/T3 + remote ggml dispatch)",
            "diffusion": "local tiers + feeder-hosted stages via "
                         "run_stage_on_feeder (PyTorch remote kernel dispatch "
                         "unsupported; shim fabric stays GREENBOOST_CLUSTER=0)",
        },
        "compute": {
            "cluster_map": "telemetry-driven N-feeder batch dispatch "
                           "(chunked, adaptive via per-node throughput_ewma)",
            "ClusterJobQueue": "same dispatch policy for incremental job arrival",
        },
    }


if __name__ == "__main__":
    s = status()
    if "--llm" in sys.argv:
        print(json.dumps(s))
    else:
        print(f"shim: {'installed' if s['shim_installed'] else 'MISSING'}   "
              f"feeders: {s['feeders_online']}/{len(s['feeders'])} online")
        host = s["host"]
        if host["gpu_util_pct"] is not None:
            print(f"  host: util={host['gpu_util_pct']:.0f}%  "
                  f"free_vram={host['free_vram_mb']}MB  temp={host['temp_c']:.0f}C")
        for f in s["feeders"]:
            state = "online" if f["online"] else f"offline ({f['error']})"
            print(f"  {f['hostname'] or f['ip']}:{f['port']}  {state}  "
                  f"T1 {f['t1_free_mb']}/{f['t1_total_mb']} MB  "
                  f"T2 {f['t2_free_mb']}/{f['t2_total_mb']} MB  "
                  f"T3 {f['t3_free_mb']}/{f['t3_total_mb']} MB")
            if f["online"] and (f["gpu_util_pct"] or f["gpu_temp_c"]):
                print(f"    util={f['gpu_util_pct']}%  temp={f['gpu_temp_c']}C  "
                      f"power={f['gpu_power_w']}W  "
                      f"throughput_ewma={f['throughput_ewma']:.2f} items/s")

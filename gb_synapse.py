#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_synapse.py — GreenBoost's HuggingFace-native, cluster-distributed GGUF
serving layer ("gb-synapse").

Where Ollama pulls models from its own registry, gb-synapse pulls GGUFs
directly from HuggingFace (any repo, gated or not, given a token) and serves
them through a matched llama.cpp build on every cluster node. It also indexes
GGUFs Ollama already downloaded, so `greenboost synapse list` shows both.

Cluster distribution uses llama.cpp's OWN RPC backend (`--rpc`), not the
GreenBoost shim's per-kernel feeder dispatch — real layer-granular tensor
split, only activations cross the wire. The GreenBoost shim still runs on
EVERY node (host and feeder) with GREENBOOST_CLUSTER=0, so each node's layer
share can extend past its physical VRAM into that node's local DDR/NVMe.
This split of duties is GreenBoost's value-add on top of vanilla llama.cpp
RPC: RPC owns the cross-node boundary, the shim owns the within-node tiers.

See workflow/gb-synapse.md and workflow/cluster-goals.md.

Layout:
    /usr/local/lib/greenboost/synapse/   built engine (llama-server, llama-cli,
                                          rpc-server, engine.version)
    /etc/greenboost/synapse/hf_token     HuggingFace token, 0600
    /var/lib/greenboost/synapse/models/  pulled GGUFs + manifest.json
    /run/greenboost/synapse/             running-server state + logs

CLI:
    python3 gb_synapse.py build-engine|update-engine
    python3 gb_synapse.py login <TOKEN>
    python3 gb_synapse.py pull <repo>[:quant] [name]
        quant is either a GGUF token (Q4_K_M, Q8_0, ...) served through
        llama.cpp, or FP8/INT8/INT4 to route through gb-quant instead
        (the synapse torch engine if installed, else a transformers
        fallback — one interface, gb-synapse picks the runtime; see
        workflow/gb-synapse.md).
    python3 gb_synapse.py index-ollama
    python3 gb_synapse.py list|rm <name>
    python3 gb_synapse.py doctor [--llm]
    python3 gb_synapse.py recommend [ctx] [--llm]
    python3 gb_synapse.py serve <model> [port]
    python3 gb_synapse.py ps|stop <model>
"""
from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_DIR))

import gb_cluster
import gb_synapse_backends
from gb_gguf_tensor_map import _load_gguf_reader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _resolve_engine_dir() -> Path:
    """Where llama-server actually IS, not where we wish it were.

    A Full Install puts the engine in the system dir; a `gb_synapse build` by a
    normal user puts it under ~/.local/share. Hardcoding the system path meant a
    perfectly good user-built engine was invisible — engine_installed() reported
    False, and serve() died with a bare FileNotFoundError on a path that had
    simply never been populated. Pick the first directory that holds a real
    binary; the env var still wins for a non-standard install.
    """
    env = os.environ.get("GB_SYNAPSE_ENGINE_DIR")
    if env:
        return Path(env)
    candidates = [Path("/usr/local/lib/greenboost/synapse"),
                  Path.home() / ".local/share/greenboost/synapse"]
    for d in candidates:
        if (d / "llama-server").is_file():
            return d
    return candidates[0]          # nothing built yet: report the install target


ENGINE_DIR = _resolve_engine_dir()
ENGINE_SRC_DIR = _REPO_DIR / "third_party" / "llama.cpp"
# Documentation-only now (used by the maintainer "bump the pin" workflow,
# not by any runtime path) — see ENGINE_SRC_DIR/NOTICE and PINNED_COMMIT.
LLAMA_CPP_REMOTE = "https://github.com/ggml-org/llama.cpp"

CONFIG_DIR = Path("/etc/greenboost/synapse")
HF_TOKEN_FILE = CONFIG_DIR / "hf_token"


def _require_huggingface_hub():
    """Import and return the huggingface_hub module, or raise an actionable
    RuntimeError naming the interpreter and the sanctioned fix.

    Real incident, 2026-07-28: gb_synapse runs in-process inside GB-CLI's
    cli-venv (backend_cmds.py's greenboost.pth bridge), and a bare
    `from huggingface_hub import ...` surfaced as a bare
    "No module named 'huggingface_hub'" with no indication of WHICH
    interpreter was missing it or how to fix it — pip-installing it into the
    wrong environment (or the developer's own miniforge) would look like a
    fix but never touch the venv `gb` actually runs under."""
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError as e:
        raise RuntimeError(
            f"huggingface_hub is not installed in this interpreter "
            f"({sys.executable}). gb-synapse needs it to pull models from "
            f"HuggingFace. Reinstall via:\n"
            f"  sudo ./greenboost_setup.sh   -> option 1 \"Full Install\""
        ) from e

MODEL_STORE_DIR = Path(os.environ.get("GB_SYNAPSE_MODEL_DIR", "/var/lib/greenboost/synapse/models"))
MANIFEST_FILE = MODEL_STORE_DIR / "manifest.json"

RUN_DIR = Path(os.environ.get("GB_SYNAPSE_RUN_DIR", "/run/greenboost/synapse"))
SLOT_DIR = Path(os.environ.get("GB_SYNAPSE_SLOT_DIR", "/var/lib/greenboost/synapse/slots"))

DEFAULT_PORT = int(os.environ.get("GB_SYNAPSE_PORT", "11435"))
RPC_PORT_BASE = 50052
_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             "-o", "StrictHostKeyChecking=accept-new"]

# How long a feeder's rpc-server gets to bind its port after we launch it
# (CUDA init on a cold GPU dominates), and how long serve() watches a freshly
# spawned engine before handing it back to the caller as "still loading".
# The grace window only needs to outlive the failures that happen at load
# time (unsupported hyperparameters, missing blob, OOM) — a legitimately slow
# multi-GB load is reported as loading, never waited out.
# 30s, not 10: a cold GPU's CUDA init on the feeder routinely takes ~20s, and
# timing out early drops a feeder that was seconds from ready — costing the run
# its whole second GPU.
RPC_READY_TIMEOUT_S = float(os.environ.get("GB_SYNAPSE_RPC_READY_S", "30"))
SERVE_READY_GRACE_S = float(os.environ.get("GB_SYNAPSE_READY_GRACE_S", "20"))

# gLLM cluster PP (SynapseTorchBackend) rendezvous port — a full band clear of
# RPC_PORT_BASE (50052 + up to a plausible feeder count) so the two mechanisms
# can never collide. One port suffices: only the host ever acts as PP master,
# so there's no per-feeder offset the way RPC_PORT_BASE + i needs one.
GLLM_MASTER_PORT_BASE = 51052

# How long the LOCAL master's /health gets to respond after launching a
# cluster PP serve, before giving up and retrying host-only. Deliberately
# much larger than SERVE_READY_GRACE_S: gLLM's dist.init_process_group()
# rendezvous (dist_utils.py) passes no timeout=, so a feeder that's slow to
# join (cold GPU CUDA init) or never joins at all (bad network/firewall) just
# hangs rather than failing fast — this is the GreenBoost-side bound that
# stands in for the one gLLM doesn't provide. NOT yet validated against a
# real hung rendezvous (see workflow/gb-synapse.md) — the actual hang
# behavior of a from-source torch.distributed build wasn't measured this
# session; treat this default as a starting point, not a proven value.
GLLM_CLUSTER_READY_TIMEOUT_S = float(os.environ.get("GB_SYNAPSE_GLLM_CLUSTER_READY_S", "90"))


def _run(cmd: list[str], capture: bool = False, check: bool = True, **kw) -> subprocess.CompletedProcess:
    print("  [gb-synapse] $ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, capture_output=capture, text=True, check=check, **kw)


# ---------------------------------------------------------------------------
# 1. Engine — fetch/build/update llama.cpp (+ rpc-server) from ggml-org
# ---------------------------------------------------------------------------

def _cuda_arch() -> str:
    """CMAKE_CUDA_ARCHITECTURES value. Override with GB_SYNAPSE_CUDA_ARCH.
    Default 120 = Blackwell cc12.0 (this project's host + feeder hardware)."""
    return os.environ.get("GB_SYNAPSE_CUDA_ARCH", "120")


def _find_nvcc() -> str:
    """Prefer /usr/local/cuda/bin/nvcc (the update-alternatives-managed path
    NVIDIA's own .deb/.run installers use) over whatever `nvcc` resolves to
    on PATH. On a box with both a distro-packaged CUDA toolkit (e.g. Debian/
    Ubuntu's `nvidia-cuda-toolkit`, often older) and a separately-installed
    NVIDIA-repo toolkit, PATH's `nvcc` can silently be the stale one — hit
    exactly this building for Blackwell (sm_120): the distro's CUDA 12.4
    doesn't know that target at all ("Unsupported gpu architecture
    'compute_120'"), while /usr/local/cuda (13.3+) does. Override with
    GB_SYNAPSE_NVCC if neither guess is right."""
    override = os.environ.get("GB_SYNAPSE_NVCC")
    if override:
        return override
    alt = Path("/usr/local/cuda/bin/nvcc")
    if alt.exists():
        return str(alt)
    return "nvcc"  # let cmake resolve it from PATH, and fail loudly if that's wrong too


def engine_source_pin(src_dir: Path = ENGINE_SRC_DIR) -> str:
    """The pinned upstream llama.cpp commit for the vendored tree at
    `src_dir` (full SHA, from third_party/llama.cpp/PINNED_COMMIT) — the
    source of truth for `engine.version` now that src_dir is a plain
    committed subtree of the greenboost repo, not its own git checkout
    (so `git describe` run there would describe greenboost's own tags/
    history, not llama.cpp's — confirmed live 2026-07-24: it silently
    returned greenboost's own short hash instead of erroring)."""
    pin_file = Path(src_dir) / "PINNED_COMMIT"
    try:
        return pin_file.read_text().strip()
    except OSError:
        raise RuntimeError(f"no PINNED_COMMIT at {pin_file} — vendored llama.cpp "
                            f"tree is missing or incomplete")


def verify_engine_source(src_dir: Path = ENGINE_SRC_DIR) -> str:
    """Assert the vendored llama.cpp tree at `src_dir` is present (does NOT
    fetch — see third_party/llama.cpp/NOTICE for the maintainer "bump the
    pin" workflow). Returns the pinned commit (short form, matching the
    style `cmd_feeders_sync_synapse`'s prefix-match parity check expects).

    Replaces the old fetch_engine_source(), which did a live `git clone`/
    `fetch --depth 1 && reset --hard` against LLAMA_CPP_REMOTE on every
    `sync-synapse` run — the engine source is now vendored+pinned in-repo,
    exactly like synapse_engine/gllm/, so there is nothing to fetch here."""
    src_dir = Path(src_dir)
    if not (src_dir / "CMakeLists.txt").exists():
        raise RuntimeError(
            f"vendored llama.cpp source not found at {src_dir} — your "
            f"greenboost repo checkout is missing third_party/llama.cpp/; "
            f"git pull the repo (this directory travels with it, same as "
            f"synapse_engine/)")
    return engine_source_pin(src_dir)[:9]


def _install_hint(pkgs: str) -> str:
    """Distro-appropriate one-liner to install `pkgs` (space-separated).
    Mirrors the per-distro logic in greenboost_setup.sh check_deps()."""
    distro_id = ""
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("ID="):
                distro_id = line.split("=", 1)[1].strip().strip('"').lower()
                break
    except Exception:
        pass
    if distro_id in ("arch", "manjaro", "endeavouros"):
        return f"sudo pacman -S {pkgs}"
    if distro_id in ("fedora", "rhel", "rocky", "almalinux", "centos"):
        return f"sudo dnf install {pkgs}"
    if distro_id in ("opensuse", "opensuse-leap", "opensuse-tumbleweed", "sles"):
        return f"sudo zypper install {pkgs}"
    return f"sudo apt install {pkgs}"  # debian/ubuntu default


def _preflight_build_tools(need_nvcc: bool = True) -> None:
    """Verify the gb-synapse engine build toolchain is present BEFORE invoking
    cmake, so a missing tool produces one actionable line instead of a raw
    Python traceback. Fresh feeder Full Install crashed here with
    FileNotFoundError: 'cmake' (2026-07-13) because cmake was in no dep path.

    Collects ALL missing tools before raising. Emits a dataflux
    synapse_build_preflight error (best-effort) so the gap is visible in the
    greenboost-dataflux MCP (Observability Must-Rule)."""
    missing: list[str] = []
    hints: list[str] = []
    if not shutil.which("cmake"):
        missing.append("cmake")
        hints.append(_install_hint("cmake") + "   (or: sudo greenboost install-deps)")
    if not shutil.which("git"):
        missing.append("git")
        hints.append(_install_hint("git"))
    if not (shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")):
        missing.append("C++ compiler (g++)")
        cxx_pkg = "gcc-c++" if "dnf" in _install_hint("") else "build-essential"
        hints.append(_install_hint(cxx_pkg))
    if need_nvcc:
        nvcc = _find_nvcc()
        if not (Path(nvcc).exists() or shutil.which(nvcc)):
            missing.append("nvcc")
            hints.append("install the CUDA toolkit, or set GB_SYNAPSE_NVCC=/path/to/nvcc")
    if not missing:
        return
    try:
        import gb_dataflux
        gb_dataflux.emit({"node": "host", "label": "synapse",
                          "kind": "synapse_build_preflight", "status": "error",
                          "missing": missing})
    except Exception:
        pass
    raise RuntimeError(
        "gb-synapse engine build prerequisites missing: " + ", ".join(missing)
        + "\n  " + "\n  ".join(hints))


def build_engine(src_dir: Path = ENGINE_SRC_DIR, install_dir: Path = ENGINE_DIR,
                  jobs: int | None = None) -> dict:
    """CMake-build llama-server, llama-cli, rpc-server, and llama-quantize
    with CUDA + RPC enabled, install to install_dir.

    Must run on EACH cluster node separately (host Intel / feeder AMD CPUs →
    -march=native binaries are not portable — same rule as `feeders
    upgrade-greenboost` in greenboost_setup.sh). A matched build on every
    node is exactly what makes the llama.cpp RPC split safe: mismatched
    kernel sets between --rpc peers is the failure mode the shim's own
    feeder-dispatch path hit (see workflow/known-issues.md).

    llama-quantize is what lets pull() serve HF repos with no GGUF release —
    see _pull_and_convert().
    """
    _preflight_build_tools(need_nvcc=True)
    src_dir = Path(src_dir)
    version = verify_engine_source(src_dir)
    build_dir = src_dir / "build-synapse"
    jobs = jobs or os.cpu_count() or 4

    _run(["cmake", "-S", str(src_dir), "-B", str(build_dir),
          "-DCMAKE_BUILD_TYPE=Release",
          "-DGGML_CUDA=ON",
          "-DGGML_RPC=ON",
          f"-DCMAKE_CUDA_ARCHITECTURES={_cuda_arch()}",
          f"-DCMAKE_CUDA_COMPILER={_find_nvcc()}",
          "-DLLAMA_CURL=OFF",  # gb-synapse owns HF downloads; no need for llama.cpp's own fetcher
          # tests/examples/pocs/app aren't vendored (see NOTICE) — these
          # options default ON in a standalone build (LLAMA_STANDALONE) and
          # gate exactly those add_subdirectory() calls, so they're
          # load-bearing, not cosmetic: confirmed live 2026-07-24, cmake
          # fails outright without them.
          "-DLLAMA_BUILD_TESTS=OFF",
          "-DLLAMA_BUILD_EXAMPLES=OFF",
          "-DLLAMA_BUILD_APP=OFF"])
    # Upstream's CMake target for the RPC server is "ggml-rpc-server" (the
    # binary was renamed at some point; the source file is still
    # tools/rpc/rpc-server.cpp). gb-synapse keeps installing it AS
    # "rpc-server" — every other reference in this module (ensure_feeder_rpc,
    # engine_installed, etc.) uses that stable name — see the install loop's
    # rename map below.
    _run(["cmake", "--build", str(build_dir), "--config", "Release",
          "--target", "llama-server", "llama-cli", "ggml-rpc-server", "llama-quantize",
          "-j", str(jobs)])

    install_dir.mkdir(parents=True, exist_ok=True)
    install_map = {"llama-server": "llama-server", "llama-cli": "llama-cli",
                   "ggml-rpc-server": "rpc-server", "llama-quantize": "llama-quantize"}
    for built_name, installed_name in install_map.items():
        built = build_dir / "bin" / built_name
        if not built.exists():
            raise RuntimeError(f"build succeeded but {built_name} not found at {built}")
        shutil.copy2(built, install_dir / installed_name)
        (install_dir / installed_name).chmod(0o755)
    for so in build_dir.glob("bin/*.so*"):  # ggml/llama shared libs, layout varies by cmake version
        shutil.copy2(so, install_dir / so.name)

    (install_dir / "engine.version").write_text(version + "\n")
    return {"version": version, "install_dir": str(install_dir)}


def update_engine() -> dict:
    """Rebuild the vendored, pinned llama.cpp for this node — no fetch (the
    source is vendored+pinned in-repo; bump third_party/llama.cpp/ deliberately
    via the maintainer workflow in its NOTICE to track a newer upstream)."""
    return build_engine()


def engine_installed() -> bool:
    return (ENGINE_DIR / "llama-server").exists() and (ENGINE_DIR / "rpc-server").exists()


def engine_version() -> str:
    try:
        return (ENGINE_DIR / "engine.version").read_text().strip()
    except OSError:
        return ""


def status() -> dict:
    """Gb-Synapse status in one dict: engine built (llama-server + rpc-server
    present in ENGINE_DIR) + version, and whether a gb-synapse llama-server
    and/or the gb-synapse Ollama/OpenAI proxy (gb_synapse_api, default port
    11435, GB_SYNAPSE_PORT) are running now.

    Single source of truth for the `synapse_status` MCP tools (gb_dataflux_mcp,
    gb_synapse_mcp, gb_mcp) and the `status` CLI verb. Matches gb-synapse's OWN
    engine path — not ollama's internal llama-server."""
    import subprocess
    torch_env = gb_synapse_backends._torch_env_dir()
    running = ps()
    out = {"engine_built": engine_installed(),
           "engine_version": engine_version() or None,
           # Derived from ps() (== _read_run_states(), pid-checked) instead
           # of a pgrep pattern that only ever matched llama-server — that
           # made server_running lie False for the 3 of 4 backends (gLLM,
           # transformers fallback, diffusers) that aren't llama-server.
           "server_running": any(_pid_alive(s["llama_pid"]) for s in running),
           "proxy_running": any(_pid_alive(s["proxy_pid"]) for s in running),
           "engine_dir": str(ENGINE_DIR),
           "torch_engine_ready": torch_env is not None,
           "torch_engine_env": str(torch_env) if torch_env else ""}
    # Supplementary signal, kept from the old check: a llama-server process
    # pgrep can find that ps()/run-state doesn't know about at all (manually
    # launched outside gb_synapse.serve(), or its run-state file was lost).
    try:
        r = subprocess.run(["pgrep", "-f", f"{ENGINE_DIR}/llama-server"],
                           capture_output=True, text=True, timeout=5)
        out["orphan_engine_detected"] = bool(r.stdout.strip()) and not out["server_running"]
    except Exception:
        out["orphan_engine_detected"] = False
    out["engines_running"] = sorted({s.get("engine", "") for s in running} - {""})
    return out


# ---------------------------------------------------------------------------
# 2. Model sourcing + local store/manifest
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    name: str
    path: str
    source: str = "hf"          # "hf" | "ollama"
    repo: str = ""
    quant: str = ""
    quant_method: str = ""      # "" | "gptq" | "awq" | "fp8" |
                                # "compressed-tensors" | "bitsandbytes" —
                                # checkpoint truth from read_quant_config(),
                                # never a routing token guess.
    quant_bits: int = 0
    arch: str = ""
    engine: str = "llama.cpp"   # "llama.cpp" | "torch" | "diffusers"
                                # (legacy on disk: "gbquant"/"vllm"/
                                # "transformers" values are all normalized to
                                # "torch" on manifest load — see _load_manifest)
    n_bytes: int = 0
    n_layers: int = 0
    is_moe: bool = False
    n_experts: int = 0
    n_experts_used: int = 0
    # dense (attention/shared/embedding) vs routed-expert byte split from
    # gguf_summary() — real incident, 2026-07-15: gguf_summary() already
    # returned these two keys but ModelEntry never had matching fields, so
    # `pull()` crashed with TypeError on every GGUF (not just MoE ones,
    # since gguf_summary always includes them) the moment it tried to
    # construct the entry with **meta.
    dense_bytes: int = 0
    expert_bytes: int = 0
    ctx_length: int = 0
    # KV-cache geometry from GGUF metadata (0 => unknown, fall back to the
    # param-count bucket heuristic in estimate_kv_gb).
    n_kv_heads: int = 0
    head_dim: int = 0
    # Layers that actually hold a real, per-token-growing KV cache — 0 means
    # "unknown, assume every layer does" (correct for a plain transformer,
    # and the safe/conservative default for an old manifest entry pulled
    # before this field existed). For a hybrid recurrent/attention
    # architecture (e.g. qwen35's Gated DeltaNet + Gated Attention mix,
    # `{arch}.full_attention_interval`), only 1 layer in N is real attention
    # — the rest hold a fixed-size recurrent state that does NOT grow with
    # context length. Confirmed live 2026-07-27: this reference model's own
    # GGUF encodes `full_attention_interval=4` over 65 layers (16 real
    # attention layers), but `estimate_kv_gb()` was using all 65 uniformly —
    # a ~4x KV-size overestimate that was clamping ctx far below what the
    # hardware can actually hold. See gguf_summary()'s computation.
    n_kv_layers: int = 0
    # Selective-SSM (Mamba/Mamba2/hybrid Gated-DeltaNet) recurrent-state
    # accounting — the counterpart to n_kv_layers above, added 2026-07-29
    # for the primary reference workload (Brian6145/Qwen3.6-27B-Claude-
    # Opus-Sonnet-Distilled-NVFP4-MTP, a Qwen3.6 hybrid: 3 GDN layers per 1
    # real-attention layer). n_kv_layers alone only EXCLUDES these layers
    # from the KV-cache term; without these fields their own state — small,
    # fixed-size, and (per the Mamba paper, arXiv:2312.00752) the hottest
    # bytes in the model since they're read+written every decode step — is
    # silently charged as zero everywhere. All default to 0/False so a
    # manifest entry pulled before this field existed round-trips unchanged
    # through _load_manifest() (same TypeError trap as dense_bytes/
    # expert_bytes above, and n_kv_layers before it).
    #
    # n_recurrent_layers: layers holding a fixed-size recurrent state
    # (n_layers - n_kv_layers, but stored explicitly rather than derived so
    # 0 can mean "no recurrent layers" instead of "field not populated").
    n_recurrent_layers: int = 0
    # is_recurrent_only: PURELY recurrent architecture (plain Mamba/Mamba2 —
    # llama.cpp's llm_arch_is_recurrent()), n_kv_layers == 0 is then the
    # CORRECT answer (no attention layer at all), not "unknown, assume every
    # layer is real attention" — see estimate_kv_gb()'s bytes_per_tok==0
    # fallback and _solve_ctx_and_layers(), both of which must not treat
    # this case as a metadata gap.
    is_recurrent_only: bool = False
    # Recurrent-state geometry, normalized to the same two quantities
    # regardless of source (see gguf_summary()'s and safetensors_summary()'s
    # own comments for the per-architecture formula each is derived from):
    #   ssm_d_conv      — causal-conv1d kernel size
    #   ssm_conv_width  — channels fed into that conv
    #   ssm_state_elems — per-layer scalar element count of the temporal/
    #                     recurrent state tensor (excludes the conv state)
    ssm_d_conv: int = 0
    ssm_conv_width: int = 0
    ssm_state_elems: int = 0
    added_ts: float = 0.0


def _load_manifest() -> dict[str, ModelEntry]:
    try:
        raw = json.loads(MANIFEST_FILE.read_text())
        out = {}
        for k, v in raw.items():
            if v.get("engine") in ("gbquant", "vllm", "transformers"):
                # Pre-torch-core manifests: "gbquant" (pre-taxonomy),
                # "vllm", and "transformers" all meant "whatever backend
                # handles safetensors" — normalize to "torch" on read so
                # every in-memory ModelEntry uses the current 3-value
                # taxonomy (llama.cpp | torch | diffusers). select_backend()
                # still honours these legacy raw values for any caller that
                # constructs a ModelEntry directly.
                v = {**v, "engine": "torch"}
            out[k] = ModelEntry(**v)
        return out
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _save_manifest(entries: dict[str, ModelEntry]) -> None:
    try:
        MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = MANIFEST_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: asdict(v) for k, v in entries.items()}, indent=2))
        tmp.rename(MANIFEST_FILE)  # atomic swap, same pattern as cmd_connect's cluster.conf write
    except PermissionError as e:
        raise PermissionError(
            f"cannot write to {MODEL_STORE_DIR} ({e}). One-time fix (run once as root):\n"
            f"  sudo mkdir -p {MODEL_STORE_DIR}\n"
            f"  sudo chgrp -R greenboost {MODEL_STORE_DIR.parent}\n"
            f"  sudo chmod -R g+rwXs {MODEL_STORE_DIR.parent}\n"
            f"  sudo usermod -aG greenboost $USER   # log out/in for the group to take effect"
        ) from e


def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    try:
        tok = HF_TOKEN_FILE.read_text().strip()
        return tok or None
    except OSError:
        return None


def login(token: str) -> None:
    """Persist an HF token, 0600. The CLI wrapper (cmd_synapse_login in
    greenboost_setup.sh) is responsible for masking the prompt — mirrors the
    masked-read pattern in cmd_feeders_setup_sudo.

    0600 is deliberately NOT group-shared (it's a personal secret, unlike the
    model store) — on a single-user machine the fix for a permission error
    left over from an earlier root-run login is a chown, not a chmod."""
    token = token.strip()
    if not token:
        raise ValueError("empty HuggingFace token")
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        HF_TOKEN_FILE.write_text(token + "\n")
        HF_TOKEN_FILE.chmod(0o600)
    except PermissionError as e:
        raise PermissionError(
            f"cannot write to {HF_TOKEN_FILE} ({e}). One-time fix (run once as root):\n"
            f"  sudo mkdir -p {CONFIG_DIR}\n"
            f"  sudo chown -R $USER {CONFIG_DIR}   # personal token — owned by you, not shared"
        ) from e


_QUANT_RE = re.compile(r'(IQ\d[\w_]*|Q\d[\w_]*|F16|BF16|F32)', re.IGNORECASE)


def _quant_from_filename(fname: str) -> str:
    m = _QUANT_RE.search(fname)
    return m.group(1).upper() if m else ""


def list_repo_gguf(repo: str) -> list[dict]:
    """[{filename, size}, ...] for every .gguf sibling file in an HF repo that
    is a candidate MAIN model weight file — excludes mmproj/vision-projector
    GGUFs (same "mmproj" substring match as _find_mmproj's glob), which are
    never valid standalone `-m` targets. Without this, pull()'s "nothing fits
    the budget" fallback (smallest file wins) could pick a ~1 GB projector
    over the real multi-GB model when the model itself doesn't fit — silently
    registering a vision-only stub as if it were the whole LLM (confirmed
    live: satgeze/Qwen3.6-35B-Uncensored-HauhauCS-1M-GGUF's manifest entry
    pointed at mmproj-qwen36-hauhau-f16.gguf, arch "clip", 0.84 GB — not the
    35B model at all)."""
    HfApi = _require_huggingface_hub().HfApi
    api = HfApi(token=hf_token())
    info = api.model_info(repo, files_metadata=True)
    return [{"filename": s.rfilename, "size": s.size or 0}
            for s in info.siblings
            if s.rfilename.endswith(".gguf") and "mmproj" not in s.rfilename.lower()]


def _convert_hf_to_gguf(local_dir: str, out_path: Path, outtype: str = "bf16") -> None:
    """Run the vendored llama.cpp convert_hf_to_gguf.py against a local HF
    snapshot (safetensors + config + tokenizer), producing an unquantized
    GGUF. This is what lets gb-synapse serve ANY HF repo, not just ones with
    a native GGUF release."""
    script = ENGINE_SRC_DIR / "convert_hf_to_gguf.py"
    if not script.exists():
        raise RuntimeError(f"conversion script not found: {script}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, str(script), local_dir, "--outfile", str(out_path), "--outtype", outtype])


def _quantize_gguf(src_path: Path, dst_path: Path, quant: str) -> None:
    """Quantize an unquantized GGUF with the vendored llama-quantize binary
    (built by build_engine() alongside llama-server)."""
    binary = ENGINE_DIR / "llama-quantize"
    if not binary.exists():
        raise RuntimeError(f"llama-quantize not built — run: greenboost synapse build-engine "
                            f"(missing: {binary})")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    _run([str(binary), str(src_path), str(dst_path), quant])


def _merge_lora_adapter(adapter_dir: str, dest_dir: Path) -> str:
    """Fuse a PEFT LoRA adapter into its base model before GGUF conversion —
    an adapter has no full weights of its own, so llama.cpp's converter
    needs the merged model, not the adapter snapshot. Ported from Unsloth's
    merge pattern (unsloth/save.py:545, `_merge_lora` + `save_method=
    "merged_16bit"`) using plain `peft`/`transformers` directly — no
    unsloth/unsloth_zoo runtime dependency, since both ultimately reduce to
    a HF-format merged model handed to the same convert_hf_to_gguf.py this
    module already vendors.

    Returns the path to the merged model directory (fused weights + the
    adapter's own tokenizer) — pass this to _convert_hf_to_gguf() instead of
    `adapter_dir`.
    """
    adapter_cfg_path = Path(adapter_dir) / "adapter_config.json"
    adapter_cfg = json.loads(adapter_cfg_path.read_text())
    base_model_id = adapter_cfg.get("base_model_name_or_path")
    if not base_model_id:
        raise RuntimeError(f"{adapter_cfg_path} has no base_model_name_or_path")

    print(f"  [gb-synapse] {adapter_dir} is a LoRA adapter on {base_model_id} — "
          f"merging before GGUF conversion …", flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16,
                                                 device_map="cpu")
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()

    dest_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(dest_dir), safe_serialization=True)
    try:
        tok = AutoTokenizer.from_pretrained(adapter_dir)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_model_id)
    tok.save_pretrained(str(dest_dir))
    return str(dest_dir)


def _pull_and_convert(repo: str, quant: str, name: str | None) -> ModelEntry:
    """Fallback for HF repos with no GGUF release (list_repo_gguf() empty):
    download the full safetensors snapshot (merging a LoRA adapter into its
    base model first, if that's what the repo is — see _merge_lora_adapter),
    convert to GGUF (bf16), quantize to `quant` (default Q4_K_M), and
    register the result exactly like a native GGUF pull. Needs the engine's
    llama-quantize — see build_engine().

    The raw HF snapshot and the intermediate bf16 GGUF are cached under
    MODEL_STORE_DIR/_hf_cache and MODEL_STORE_DIR/_converted so re-pulling at
    a different quant doesn't re-download or re-convert from scratch.
    """
    if not (ENGINE_DIR / "llama-quantize").exists():
        raise RuntimeError("no GGUF release for this repo, and llama-quantize is not "
                            "built — run: greenboost synapse build-engine")
    quant = (quant or "Q4_K_M").upper()

    snapshot_download = _require_huggingface_hub().snapshot_download
    cache_dir = str(MODEL_STORE_DIR / "_hf_cache")
    converted_dir = MODEL_STORE_DIR / "_converted"
    safe_stem = repo.replace("/", "__")

    print(f"  [gb-synapse] no GGUF release for {repo} — downloading safetensors "
          f"for on-the-fly conversion (much larger than a GGUF pull)", flush=True)
    local_dir = snapshot_download(
        repo_id=repo, token=hf_token(), cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "tokenizer*"],
    )

    if (Path(local_dir) / "adapter_config.json").exists():
        local_dir = _merge_lora_adapter(local_dir, converted_dir / f"{safe_stem}.merged")

    bf16_path = converted_dir / f"{safe_stem}.bf16.gguf"
    if not bf16_path.exists():
        print(f"  [gb-synapse] converting {repo} to GGUF (bf16) …", flush=True)
        _convert_hf_to_gguf(local_dir, bf16_path)

    quant_path = converted_dir / f"{safe_stem}.{quant}.gguf"
    if not quant_path.exists():
        print(f"  [gb-synapse] quantizing to {quant} …", flush=True)
        _quantize_gguf(bf16_path, quant_path, quant)

    meta = gguf_summary(str(quant_path))
    meta["quant"] = quant
    entry_name = name or repo.split("/")[-1]
    entry = ModelEntry(name=entry_name, path=str(quant_path), source="hf", repo=repo,
                        added_ts=time.time(), **meta)
    manifest = _load_manifest()
    manifest[entry_name] = entry
    _save_manifest(manifest)
    return entry


# Tokens that mean "quantize via gb-quant, serve through the torch engine/
# transformers" instead of a GGUF/llama.cpp quant. Deliberately excludes
# BF16 (a real GGUF quant filename token too, e.g. "model-BF16.gguf") to
# avoid ambiguity — FP8/INT8/INT4 are never real GGUF quant tokens (those
# use QX_K/QX_0 names).
_GBQUANT_TOKENS = {"FP8", "INT8", "INT4"}


_QUANT_METHOD_TOKEN = {
    "fp8": "FP8",
    "gptq": "GPTQ{bits}",
    "awq": "AWQ{bits}",
    "compressed-tensors": "CT-W{bits}",
    "bitsandbytes": "BNB{bits}",
}


def _quant_display_token(quant_method: str, quant_bits: int) -> str:
    """The manifest's human-facing `quant` string for a checkpoint whose
    quantization was detected from its own config.json (read_quant_config),
    not chosen by the user — e.g. "GPTQ4", "FP8", "CT-W8". Empty for an
    unrecognized/absent method."""
    tmpl = _QUANT_METHOD_TOKEN.get(quant_method, "")
    return tmpl.format(bits=quant_bits) if tmpl else ""


# Quant-method families gLLM's own layer dispatch cannot consume.
# gllm/layers/linear.py's create_weights/dispatch_quant_method only has real
# code paths for quant_method in {None, "fp8", "gptq", "awq"} — anything else
# hits `raise Exception(f"gLLM do not support quant_method {...}")`.
# Confirmed literally (not just via gb_synapse's own normalization) against
# Brian6145/Qwen3.6-27B-Claude-Opus-Sonnet-Distilled-NVFP4-MTP's raw HF
# config.json: quantization_config.quant_method == "compressed-tensors"
# (format "nvfp4-pack-quantized"), 26.59 GiB — that literal string is exactly
# what would reach gLLM's final `else: raise` after a full snapshot_download.
# gllm/model_loader.py's own docstring agrees: "gLLM does not consume
# compressed-tensors metadata directly" (the only exception is a Kimi-K2.5-
# specific int4-MoE normalization, not a general compressed-tensors path).
_GLLM_UNSUPPORTED_QUANT_METHODS = {"compressed-tensors", "nvfp4", "bitsandbytes"}


def _check_torch_engine_capability(repo: str, engine: str) -> None:
    """Refuse, before any snapshot_download, a checkpoint whose OWN
    quant_method the engine that will actually serve it is known in advance
    to reject — see _GLLM_UNSUPPORTED_QUANT_METHODS. Only gates the "torch"
    engine when gLLM (SynapseTorchBackend) is actually installed and would be
    selected; when it isn't, select_backend() falls back to
    TransformersBackend, which may genuinely load these formats via
    `transformers`+`compressed_tensors` — untested here and out of scope, so
    that fallback path is deliberately not gated.

    Best-effort like read_quant_config itself: a metadata gap (network
    failure, unparseable config, unrecognized shape) returns {} and this
    silently permits the pull rather than blocking on ambiguity."""
    if engine != "torch" or not gb_synapse_backends.SynapseTorchBackend().available():
        return
    qc = read_quant_config(repo)
    method = qc.get("quant_method", "")
    if method not in _GLLM_UNSUPPORTED_QUANT_METHODS:
        return
    try:
        import gb_dataflux
        gb_dataflux.emit({"kind": "synapse_serve", "status": "capability_refused",
                          "model": repo, "quant_method": method,
                          "quant_bits": qc.get("quant_bits", 0), "engine": engine})
    except Exception:
        pass
    raise RuntimeError(
        f"{repo}: checkpoint is quantized with {method!r} "
        f"({qc.get('quant_bits', 0)}-bit) — gb-synapse's torch engine (gLLM) "
        f"only serves fp8/gptq/awq checkpoints natively (gllm/layers/"
        f"linear.py's quant dispatch has no path for {method!r}). Refusing "
        f"before downloading rather than failing after. See "
        f"workflow/known-issues.md."
    )


def _pull_torch(repo: str, quant: str, name: str | None, engine: str = "torch") -> ModelEntry:
    """Fetch the safetensors (or `.bin`) snapshot for torch-core serving
    (the synapse torch engine's on-the-fly native quantization, or the
    transformers fallback, both quantize at load time — no GGUF conversion
    needed here, unlike _pull_and_convert).

    Never re-quantizes an already-quantized checkpoint: `read_quant_config`
    (folded into `safetensors_summary`'s returned meta) is the checkpoint's
    OWN truth — when it reports a real quant_method, that overrides whatever
    quant token the caller passed (with a loud warning if the two actually
    disagree), because serving a GPTQ/AWQ/FP8/compressed-tensors/bnb
    checkpoint AS IF it were a plain bf16 one to be re-quantized would
    silently double-quantize it.

    `engine` defaults to "torch" (select_backend prefers the synapse torch
    engine, falling back to transformers automatically when the torch venv
    isn't installed); pass "transformers" explicitly to pin the fallback
    engine regardless of torch-engine availability."""
    _check_torch_engine_capability(repo, engine)
    snapshot_download = _require_huggingface_hub().snapshot_download
    cache_dir = str(MODEL_STORE_DIR / "_hf_cache")
    local_dir = snapshot_download(
        repo_id=repo, token=hf_token(), cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.bin", "*.json", "*.model", "*.txt", "tokenizer*"],
        ignore_patterns=["training_args.bin"],
    )
    entry_name = name or repo.split("/")[-1]
    meta = safetensors_summary(local_dir)
    checkpoint_quant = meta.get("quant_method", "")
    requested = (quant or "").upper()
    if checkpoint_quant:
        display = _quant_display_token(checkpoint_quant, meta.get("quant_bits", 0))
        if requested and requested != display:
            print(f"[gb-synapse] checkpoint is already {checkpoint_quant}-quantized — "
                 f"serving it as-is (ignoring :{requested})")
        quant_token = display or requested or "BF16"
    else:
        quant_token = requested or "BF16"
    entry = ModelEntry(name=entry_name, path=local_dir, source="hf", repo=repo,
                        engine=engine, quant=quant_token, added_ts=time.time(), **meta)
    manifest = _load_manifest()
    manifest[entry_name] = entry
    _save_manifest(manifest)
    return entry


# Kept for one release: pre-torch-core callers still spell this name.
_pull_gbquant = _pull_torch


def _pull_diffusers(repo: str, name: str | None) -> ModelEntry:
    """Fetch a full HF diffusers pipeline tree (model_index.json + every
    component subfolder) for DiffusersBackend's image server. No GGUF
    conversion — diffusers loads the safetensors/config layout directly.
    Same cache-discipline as _pull_torch (MODEL_STORE_DIR/_hf_cache, so a
    re-pull never re-downloads from scratch) — EXCEPT it checks the
    interactive user's own default HF cache first (`local_files_only=True`
    against `huggingface_hub`'s own `HF_HUB_CACHE`): image models are
    routinely pulled by hand into `~/.cache/huggingface` before gb-synapse
    is asked to serve them (e.g. via a Studio/comfy workflow), and
    `snapshot_download` dedupes by content hash PER cache_dir, so pointing
    straight at MODEL_STORE_DIR/_hf_cache would silently re-download every
    blob into a second location instead of reusing what's already on disk."""
    _hf = _require_huggingface_hub()
    _hf_constants, snapshot_download = _hf.constants, _hf.snapshot_download
    patterns = ["*.safetensors", "*.json", "*.txt", "*.model", "tokenizer*"]
    try:
        local_dir = snapshot_download(
            repo_id=repo, cache_dir=_hf_constants.HF_HUB_CACHE,
            allow_patterns=patterns, local_files_only=True,
        )
    except Exception:
        cache_dir = str(MODEL_STORE_DIR / "_hf_cache")
        local_dir = snapshot_download(
            repo_id=repo, token=hf_token(), cache_dir=cache_dir,
            allow_patterns=patterns,
        )
    entry_name = name or repo.split("/")[-1]
    entry = ModelEntry(name=entry_name, path=local_dir, source="hf", repo=repo,
                        engine="diffusers", quant="fp8", added_ts=time.time())
    manifest = _load_manifest()
    manifest[entry_name] = entry
    _save_manifest(manifest)
    return entry


# Deprecated `pull(engine=...)` spellings, mapped to the current taxonomy's
# single safetensors-checkpoint backend name. Kept for one release.
_DEPRECATED_ENGINE_ALIASES = {"vllm": "torch", "transformers": "torch"}


def pull(repo_spec: str, name: str | None = None, engine: str = "") -> ModelEntry:
    """Download a GGUF from HuggingFace — or, if the repo has no native GGUF
    release, route it to whichever engine actually serves it (see
    _pull_and_convert / _pull_torch / _pull_diffusers).

    repo_spec is "org/repo" or "org/repo:QUANT". QUANT is either a GGUF quant
    token (Q4_K_M, Q8_0, ...) served through llama.cpp, or FP8/INT8/INT4 to
    route through torch-core (the synapse torch engine if installed, else
    transformers).

    With no quant and no explicit `engine`, the repo's own file listing
    decides once it turns out to have no GGUF release: a diffusers
    `model_index.json` routes to the image backend; a token-less safetensors
    (or `.bin`) repo routes to torch-core at fp8 quality when the torch
    engine's venv exists (falls back to GGUF conversion otherwise, same as
    before this routing existed). A repo WITH a GGUF release still picks
    the largest quant that fits the live cluster's aggregate VRAM, same as
    always.

    `engine` (also `--engine` on the CLI) overrides detection outright, one
    of "torch"/"diffusers"; use it when a repo's own layout is ambiguous.
    "vllm"/"transformers" are still accepted as deprecated aliases for
    "torch" (printed note, no behavior loss for the common case — a
    checkpoint's OWN quant_method, not this flag, decides how it's actually
    quantized; see _pull_torch). Multi-shard GGUFs
    ("...-00001-of-00003.gguf") pull every shard; llama.cpp loads the first
    and follows the split chain itself.
    """
    repo, _, quant = repo_spec.partition(":")
    quant_u = quant.upper()

    if engine in _DEPRECATED_ENGINE_ALIASES:
        print(f"[gb-synapse] --engine {engine!r} is deprecated — mapped to "
             f"\"torch\" (the current taxonomy's single safetensors backend name)")
        engine = _DEPRECATED_ENGINE_ALIASES[engine]

    if engine == "diffusers":
        return _pull_diffusers(repo, name)
    if engine == "torch" or quant_u in _GBQUANT_TOKENS:
        return _pull_torch(repo, quant or "fp8", name, engine="torch")

    files = list_repo_gguf(repo)
    if not files and not quant and not engine:
        fmt = gb_synapse_backends.detect_format(repo)
        if fmt == "diffusers":
            return _pull_diffusers(repo, name)
        if fmt == "safetensors" and gb_synapse_backends.SynapseTorchBackend().available():
            return _pull_torch(repo, "fp8", name)
    if not files:
        return _pull_and_convert(repo, quant, name)

    if quant:
        matches = [f for f in files if quant.upper() in f["filename"].upper()]
        if not matches:
            available = sorted({_quant_from_filename(f["filename"]) for f in files} - {""})
            raise RuntimeError(f"no GGUF matching quant {quant!r} in {repo}; "
                                f"available: {', '.join(available) or 'unknown'}")
    else:
        budget_mb = doctor(probe_feeders=False)["aggregate_vram_mb"]
        fitting = [f for f in files if f["size"] and f["size"] / (1024 ** 2) <= budget_mb]
        matches = [max(fitting, key=lambda f: f["size"])] if fitting \
            else [min(files, key=lambda f: f["size"] or 0)]

    target = matches[0]
    hf_hub_download = _require_huggingface_hub().hf_hub_download
    cache_dir = str(MODEL_STORE_DIR / "_hf_cache")

    shard_m = re.match(r'(.+)-(\d{5})-of-(\d{5})\.gguf$', target["filename"])
    if shard_m:
        stem, _, total = shard_m.groups()
        local_path = None
        for i in range(1, int(total) + 1):
            p = hf_hub_download(repo_id=repo, filename=f"{stem}-{i:05d}-of-{total}.gguf",
                                 token=hf_token(), cache_dir=cache_dir)
            if i == 1:
                local_path = p
    else:
        local_path = hf_hub_download(repo_id=repo, filename=target["filename"],
                                      token=hf_token(), cache_dir=cache_dir)

    meta = gguf_summary(local_path)
    meta["quant"] = (quant.upper() if quant else "") or meta.get("quant") or _quant_from_filename(target["filename"])
    entry_name = name or repo.split("/")[-1]
    entry = ModelEntry(name=entry_name, path=local_path, source="hf", repo=repo,
                        added_ts=time.time(), **meta)
    manifest = _load_manifest()
    manifest[entry_name] = entry
    _save_manifest(manifest)
    return entry


def pull_model(name: str, progress: "callable | None" = None) -> ModelEntry:
    """Thin wrapper around pull() for gb_api.py's public facade , the
    call ai-forge's forge/gb_models.py hand-rolls itself (streaming
    /api/pull, an 8-attempt/60s backoff sized around "a greenboost
    reinstall can SIGKILL ollama mid-request and the reload takes ~30s").

    `progress`, if given, is called once with ("start", name) before the
    pull and once with ("done", name) after. pull() itself has no
    byte-level download progress hook today (that would need
    instrumenting the huggingface_hub download call inside it) , this is
    a coarse two-event signal, not a percentage callback, documented here
    so a caller doesn't expect more than it gets."""
    if progress:
        progress("start", name)
    entry = pull(name)
    if progress:
        progress("done", name)
    return entry


def wait_ready(url: str, *, path: str = "/health", timeout_s: float = 120.0,
              attempts: "int | None" = None) -> bool:
    """Poll `url + path` until it answers HTTP 200, or timeout_s elapses.
    A GENERIC readiness gate for any OpenAI/Ollama-compatible endpoint ,
    not tied to gb-synapse's own engine lifecycle the way
    _wait_upstream_ready/_wait_proxy_ready are (those need a live Popen to
    poll for early death; this is for a caller with no process handle at
    all, e.g. ai-forge waiting on its own OCR-VL server, or a gb-synapse
    endpoint from a separate process).

    `attempts`, if given, caps the number of poll attempts regardless of
    timeout_s (still time-bounded by it); backoff between attempts grows
    linearly up to 10s so a slow-loading multi-GB model isn't hammered
    with requests while it loads."""
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout_s
    n = 0
    while time.time() < deadline:
        n += 1
        if attempts is not None and n > attempts:
            return False
        try:
            with urllib.request.urlopen(f"{url}{path}", timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(min(2.0 * n, 10.0))
    return False


def serve_gguf(model_path: str, port: int, mmproj: "str | None" = None,
               ctx: int = 0) -> dict:
    """Serve an arbitrary GGUF file directly via llama-server on `port` ,
    no gb-synapse Ollama-compatible proxy, no manifest resolution, no
    cluster split. The gap behind ai-forge's forge/ocr_vl.py: it
    REIMPLEMENTS _resolve_engine_dir (its own docstring says so) then
    hand-launches and health-polls llama-server itself for its OCR-VL
    model on :8081, because gb-synapse had no "just serve this GGUF on
    this port" call , serve_and_repoint only covers the :11435-compatible
    proxy.

    Returns {"pid", "port"} , the raw process handle, not a full
    ServerState (no proxy, no run-state persistence): this is a
    standalone serve and the caller owns its own lifecycle (stop it by
    killing "pid")."""
    if not engine_installed():
        raise RuntimeError("engine not built — run: greenboost synapse build-engine")
    cmd = [str(ENGINE_DIR / "llama-server"), "-m", model_path,
           "--host", "127.0.0.1", "--port", str(port),
           "-ngl", "999", "--no-webui"]
    if ctx > 0:
        cmd += ["--ctx-size", str(ctx)]
    if mmproj:
        cmd += ["--mmproj", mmproj]
    log = open(_run_log_path(f"servegguf_{port}"), "ab")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return {"pid": proc.pid, "port": port}


def endpoints() -> dict:
    """The known inference endpoints this box exposes, read from
    /etc/greenboost/inference.env (the file gb_actuation.serve_and_repoint()
    writes) , the registry ai-forge's forge/config.py currently parses
    itself, tracking 4 independent endpoints (gb-synapse :11435, OCR-VL
    :8081, OCR-GPU :8082, AI-tools :8083) with no shared source of truth.
    Reuses gb_actuation._read_env_file rather than a second parser."""
    try:
        import gb_actuation
        return gb_actuation._read_env_file()
    except Exception:
        return {}


# ---- Ollama blob indexing (mirrors _gb_ollama_model_blob, greenboost_setup.sh:10281) ----

_OLLAMA_MODEL_DIRS = ("/usr/share/ollama/.ollama/models",
                      os.path.expanduser("~/.ollama/models"),
                      "/root/.ollama/models")


def _ollama_models_dir() -> Path | None:
    for d in _OLLAMA_MODEL_DIRS:
        if (Path(d) / "blobs").is_dir():
            return Path(d)
    return None


def _split_model_tag(model_name: str) -> tuple[str, str, str]:
    ns, sep, rest = model_name.partition("/")
    if not sep:
        ns, rest = "library", model_name
    name, _, tag = rest.partition(":")
    return ns, name, tag or "latest"


def ollama_model_blob(model_name: str) -> str | None:
    """Resolve an Ollama model name ("qwen3-coder" or "ns/name:tag") to its
    on-disk GGUF blob path."""
    found = _ollama_model_layers(model_name)
    return found[0] if found else None


def ollama_model_projector(model_name: str) -> str | None:
    """The vision projector blob of an Ollama model, if it has one.

    Ollama stores the mmproj as a SEPARATE layer beside the weights
    (mediaType ...image.projector) under an opaque sha256- filename, so no
    filename glob can ever find it. Without resolving it here, an
    Ollama-indexed VLM (ai-forge's qwen3-vl critic) is served text-only and
    answers about images it never saw."""
    found = _ollama_model_layers(model_name)
    return found[1] if found else None


def _ollama_model_layers(model_name: str) -> "tuple[str, str | None] | None":
    """(weights_blob, projector_blob|None) for an Ollama model."""
    models_dir = _ollama_models_dir()
    if models_dir is None:
        return None
    ns, name, tag = _split_model_tag(model_name)
    for mp in (models_dir / "manifests" / "registry.ollama.ai" / ns / name / tag,
               models_dir / "manifests" / "registry.ollama.ai" / "library" / name / tag):
        if not mp.is_file():
            continue
        try:
            manifest = json.loads(mp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        layers = manifest.get("layers", [])
        layer = next((l for l in layers if "model" in l.get("mediaType", "")), None)
        layer = layer or (max(layers, key=lambda l: l.get("size", 0)) if layers else None)
        if not layer:
            continue
        blob = models_dir / "blobs" / layer["digest"].replace(":", "-")
        if not blob.is_file():
            continue
        proj = next((l for l in layers if "projector" in l.get("mediaType", "")), None)
        proj_blob = None
        if proj:
            p = models_dir / "blobs" / proj["digest"].replace(":", "-")
            proj_blob = str(p) if p.is_file() else None
        return str(blob), proj_blob
    return None


def index_ollama_models() -> list[ModelEntry]:
    """Scan Ollama's manifests dir and register every resolvable model as a
    source="ollama" ModelEntry (rm() refuses to delete these — they're
    Ollama's, not ours)."""
    models_dir = _ollama_models_dir()
    manifests_root = models_dir / "manifests" / "registry.ollama.ai" if models_dir else None
    if not manifests_root or not manifests_root.is_dir():
        return []

    manifest = _load_manifest()
    found: list[ModelEntry] = []
    for ns_dir in manifests_root.iterdir():
        if not ns_dir.is_dir():
            continue
        for name_dir in ns_dir.iterdir():
            if not name_dir.is_dir():
                continue
            for tag_file in name_dir.iterdir():
                if not tag_file.is_file():
                    continue
                model_ref = f"{ns_dir.name}/{name_dir.name}:{tag_file.name}"
                blob = ollama_model_blob(model_ref)
                if not blob:
                    continue
                entry_name = f"{name_dir.name}:{tag_file.name}" if ns_dir.name == "library" else model_ref
                try:
                    meta = gguf_summary(blob)
                except Exception:
                    meta = {}
                entry = ModelEntry(name=entry_name, path=blob, source="ollama",
                                    added_ts=time.time(), **meta)
                manifest[entry_name] = entry
                found.append(entry)
    try:
        _save_manifest(manifest)
    except PermissionError:
        pass  # discovery still succeeds without root/group write access to
              # MODEL_STORE_DIR — only the persisted cache is skipped, not
              # the scan itself; see list_models()
    return found


def list_models() -> list[ModelEntry]:
    """Every model gb-synapse knows about — HF-pulled (persisted manifest
    entries) plus a live re-scan of Ollama's blobs, merged and re-persisted
    each call. Ollama-sourced entries are auto-refreshed (not just indexed
    once via `index-ollama`) so a model pulled through `ollama pull` shows up
    here immediately, with no manual indexing step — and this reads Ollama's
    on-disk manifests/blobs directly, so it works whether or not the Ollama
    service is currently running."""
    manifest = _load_manifest()
    try:
        for entry in index_ollama_models():
            manifest[entry.name] = entry
    except Exception:
        pass
    return list(manifest.values())


def rm(name: str) -> None:
    manifest = _load_manifest()
    entry = manifest.get(name)
    if entry is None:
        raise KeyError(f"no such model: {name}")
    if entry.source == "ollama":
        raise ValueError(f"{name} is an Ollama-managed blob — use 'ollama rm {name}' instead")
    manifest.pop(name)
    _save_manifest(manifest)
    try:
        Path(entry.path).unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 3. GGUF metadata + doctor + recommend
# ---------------------------------------------------------------------------

def _field_scalar(reader, key: str):
    f = reader.get_field(key)
    if f is None or not f.parts:
        return None
    try:
        return f.parts[-1][0]
    except (IndexError, TypeError):
        return None


def _field_str(reader, key: str) -> str:
    f = reader.get_field(key)
    if f is None or not f.parts:
        return ""
    try:
        return bytes(f.parts[-1]).decode("utf-8", errors="replace")
    except Exception:
        return ""


_GGUF_SUMMARY_CACHE: dict[str, tuple[float, int, dict]] = {}


def gguf_summary(path: str) -> dict:
    """Parse layer count, quant type, MoE expert config, context length, and
    total weight bytes from a GGUF file. Reuses the vendored llama.cpp
    GGUFReader via gb_gguf_tensor_map.

    Cached by (path, mtime, size) — real incident, 2026-07-15: list_models()
    calls this for EVERY Ollama-sourced GGUF on EVERY call (it's a full
    tensor-list scan, summing every tensor's n_bytes, not a cheap header
    peek), and with several 20+ GB models indexed this made gb_synapse_api.py's
    /api/tags handler block the single-threaded aiohttp event loop for 30+
    seconds — every OTHER concurrent request (health, chat) queued behind it,
    looking exactly like a hang. The file only needs re-summarizing when it
    actually changes."""
    try:
        st = os.stat(path)
        cache_key = (st.st_mtime, st.st_size)
    except OSError:
        cache_key = None
    if cache_key is not None:
        cached = _GGUF_SUMMARY_CACHE.get(path)
        if cached is not None and cached[:2] == cache_key:
            return dict(cached[2])

    GGUFReader = _load_gguf_reader()
    reader = GGUFReader(path, mode="r")
    n_bytes = sum(int(t.n_bytes) for t in reader.tensors)
    # dense (attention/shared/embedding) vs routed-expert byte split — same
    # "_exps" tensor-name convention llama.cpp/ggml uses for MoE expert
    # tensors, mirrors colibri's EXPERT_RE (c/resource_plan.py). Feeds
    # gb_placement.plan_experts()'s tier-plan (CB-3): only the routed-expert
    # bytes are eligible for T2/T3 overflow under Rule #1 — dense stays T1.
    expert_bytes = sum(int(t.n_bytes) for t in reader.tensors if "_exps" in t.name)
    dense_bytes = n_bytes - expert_bytes
    arch = _field_str(reader, "general.architecture") or "llama"
    n_layers = int(_field_scalar(reader, f"{arch}.block_count") or 0)
    ctx_length = int(_field_scalar(reader, f"{arch}.context_length") or 0)
    n_experts = int(_field_scalar(reader, f"{arch}.expert_count") or 0)
    n_experts_used = int(_field_scalar(reader, f"{arch}.expert_used_count") or 0)
    is_moe = n_experts > 0 or expert_bytes > 0
    # KV geometry: real GGUF attention metadata, so estimate_kv_gb doesn't rely
    # on the param-count bucket heuristic (wrong by ~100x on some GQA archs —
    # verified live OOM on Qwable-9B). head_count_kv falls back to head_count
    # (MHA); head_dim from key_length if present, else embedding/head_count.
    head_count = int(_field_scalar(reader, f"{arch}.attention.head_count") or 0)
    head_count_kv = int(_field_scalar(reader, f"{arch}.attention.head_count_kv") or head_count)
    emb_len = int(_field_scalar(reader, f"{arch}.embedding_length") or 0)
    key_len = int(_field_scalar(reader, f"{arch}.attention.key_length") or 0)
    head_dim = key_len or (emb_len // head_count if head_count else 0)
    # Hybrid recurrent/attention architectures (qwen35's Gated DeltaNet +
    # Gated Attention mix) only give a real, context-length-scaling KV cache
    # to 1 layer in `full_attention_interval` — the rest hold a fixed-size
    # recurrent state. Absent this key (the common case), every layer is a
    # real attention layer, matching llama.cpp's own default in
    # src/models/qwen35.cpp (`full_attn_interval = 4` there is qwen35-
    # specific; the GENERIC GGUF key read here is 1 for a plain transformer).
    full_attn_interval = int(_field_scalar(reader, f"{arch}.full_attention_interval") or 1)
    # Some hybrids (falcon-h1, granitehybrid, nemotron_h) list recurrent layer
    # indices explicitly instead of a uniform interval (llama-arch.cpp:260,
    # `%s.attention.recurrent_layers`) — prefer it over the interval math when
    # present, since an explicit list can be non-uniform in a way an interval
    # can't express.
    recurrent_layers_field = reader.get_field(f"{arch}.attention.recurrent_layers")
    n_recurrent_layers = 0
    if recurrent_layers_field is not None and recurrent_layers_field.data:
        # GGUFReader array fields: `.data` holds one index per element (see
        # gguf_reader.py's _get_field_parts — `.parts` additionally carries a
        # 2-entry [type, length] header, so `.data` is the correct count).
        n_recurrent_layers = len(recurrent_layers_field.data)
    elif full_attn_interval > 1:
        n_recurrent_layers = n_layers - math.ceil(n_layers / full_attn_interval)
    is_recurrent_only = arch in _engine_recurrent_archs()
    if is_recurrent_only:
        n_recurrent_layers = n_layers
    n_kv_layers = n_layers - n_recurrent_layers
    # Selective-SSM recurrent-state geometry (Mamba/Mamba2/hybrid GDN archs):
    # the GGUF `ssm.*` keys llama.cpp's own loader reads (llama-arch.cpp:290-
    # 295), normalized to the same (ssm_d_conv, ssm_conv_width, ssm_state_elems)
    # shape safetensors_summary() produces so estimate_ssm_state_gb() has one
    # formula for both metadata sources — see llama-hparams.cpp's n_embd_r()/
    # n_embd_s() (conv_width = d_inner + 2*n_group*d_state, state_elems =
    # d_state*d_inner), the formula this engine's own recurrent-memory
    # allocator actually uses. Absent on a plain transformer GGUF — 0 is
    # correct there (no recurrent state to size), not a metadata gap.
    ssm_d_conv = int(_field_scalar(reader, f"{arch}.ssm.conv_kernel") or 0)
    _ssm_d_inner = int(_field_scalar(reader, f"{arch}.ssm.inner_size") or 0)
    _ssm_d_state = int(_field_scalar(reader, f"{arch}.ssm.state_size") or 0)
    _ssm_n_group = int(_field_scalar(reader, f"{arch}.ssm.group_count") or 1)
    ssm_conv_width = _ssm_d_inner + 2 * _ssm_n_group * _ssm_d_state
    ssm_state_elems = _ssm_d_state * _ssm_d_inner
    quant = ""
    weight_types = Counter(t.tensor_type.name for t in reader.tensors if "weight" in t.name)
    if weight_types:
        quant = weight_types.most_common(1)[0][0]
    result = {
        "n_bytes": n_bytes, "n_layers": n_layers, "is_moe": is_moe,
        "n_experts": n_experts, "n_experts_used": n_experts_used,
        "dense_bytes": dense_bytes, "expert_bytes": expert_bytes,
        "ctx_length": ctx_length, "quant": quant, "arch": arch,
        "n_kv_heads": head_count_kv, "head_dim": head_dim,
        "n_kv_layers": n_kv_layers,
        "n_recurrent_layers": n_recurrent_layers,
        "is_recurrent_only": is_recurrent_only,
        "ssm_d_conv": ssm_d_conv, "ssm_conv_width": ssm_conv_width,
        "ssm_state_elems": ssm_state_elems,
    }
    if cache_key is not None:
        _GGUF_SUMMARY_CACHE[path] = (cache_key[0], cache_key[1], result)
    return result


def read_quant_config(path_or_repo: str) -> dict:
    """Checkpoint-truth quantization detection: parse `quantization_config`
    (or `text_config.quantization_config` for nested multimodal configs) out
    of a local checkpoint dir's `config.json`, or (for a bare "org/repo" that
    doesn't exist locally) fetch just that one file from the HF Hub.

    Returns `{"quant_method": "gptq"|"awq"|"fp8"|"nvfp4"|"compressed-tensors"|
    "bitsandbytes"|"", "quant_bits": int}` — best-effort, `{}` on any
    failure (missing config, malformed JSON, network error, unrecognized
    shape). Never raises: this feeds routing decisions (P1.6's `pull()`),
    where a metadata gap should degrade to "treat as an unquantized
    checkpoint", not abort the pull."""
    try:
        if Path(path_or_repo).is_dir():
            cfg = json.loads((Path(path_or_repo) / "config.json").read_text())
        else:
            hf_hub_download = _require_huggingface_hub().hf_hub_download
            local = hf_hub_download(repo_id=path_or_repo, filename="config.json",
                                    token=hf_token())
            cfg = json.loads(Path(local).read_text())
        tc = cfg.get("text_config", cfg)
        qc = tc.get("quantization_config") or cfg.get("quantization_config")
        if not qc:
            return {}

        method = (qc.get("quant_method") or "").lower()
        if not method:
            # bnb configs and some compressed-tensors exports carry no
            # quant_method key at all — infer from the fields that are there.
            if qc.get("load_in_4bit") or qc.get("load_in_8bit"):
                method = "bitsandbytes"
            elif "config_groups" in qc:
                method = "compressed-tensors"
            else:
                return {}

        if method == "fp8":
            bits = 8
        elif method in ("gptq", "awq"):
            bits = int(qc.get("bits") or 0)
        elif method == "bitsandbytes":
            bits = 4 if qc.get("load_in_4bit") else 8
        elif method == "compressed-tensors":
            groups = [g for g in (qc.get("config_groups") or {}).values()
                      if isinstance(g, dict)]
            # A float-quantized group's own num_bits distinguishes real fp8
            # (8-bit float) from nvfp4 (4-bit float, "nvfp4-pack-quantized"
            # format) — both carry weights.type == "float", so a blanket
            # "any float group -> fp8" bucket silently misclassified nvfp4
            # checkpoints as fp8 (confirmed live 2026-07-28:
            # Brian6145/Qwen3.6-27B-Claude-Opus-Sonnet-Distilled-NVFP4-MTP,
            # weights.num_bits=4, reported quant_method="fp8" before this
            # fix) — which would have defeated pull()'s below-fp8/
            # unsupported-quant gate for exactly the checkpoint it exists to
            # catch.
            float_groups = [g.get("weights", {}) for g in groups
                            if g.get("weights", {}).get("type") == "float"]
            if float_groups and all(int(w.get("num_bits") or 8) <= 4 for w in float_groups):
                method, bits = "nvfp4", 4
            elif float_groups:
                method, bits = "fp8", 8
            else:
                group_bits = [int(g["weights"]["num_bits"]) for g in groups
                              if "weights" in g and "num_bits" in g["weights"]]
                bits = min(group_bits) if group_bits else 0
        else:
            return {}

        return {"quant_method": method, "quant_bits": bits}
    except Exception:
        return {}


# HF's `model_type` (a short lowercase config identifier, e.g. "mamba2")
# and `architectures` (the real PyTorch class name, e.g. "Mamba2ForCausalLM")
# are NOT the same string -- falling back to model_type verbatim when
# architectures[] is absent silently produces a class name nothing matches
# (gb_synapse_backends.select_backend() and gllm/model_loader.py's
# get_model_type() both compare against real class names). Found live
# 2026-07-29: AntonV/mamba2-2.7b-hf's config.json has no `architectures` key
# at all (only the smaller AntonV/mamba2-130m-hf checkpoint happens to ship
# one) -- resolved to "mamba2" instead of "Mamba2ForCausalLM", routing a
# genuinely-supported checkpoint to the transformers fallback server instead
# of the vendored torch engine. Only the model_type this repo's Mamba-2
# support actually targets is mapped here -- add another entry if/when a
# real checkpoint needs it, not speculatively.
_ARCH_FROM_MODEL_TYPE = {"mamba2": "Mamba2ForCausalLM"}


def safetensors_summary(local_dir: str) -> dict:
    """gguf_summary()'s counterpart for a safetensors snapshot (torch-engine/
    transformers-routed pulls, `_pull_gbquant`). Real gap found live
    2026-07-16: `_pull_gbquant` built every `ModelEntry` with n_bytes/
    n_layers/arch/ctx_length/is_moe all zero/empty — the serving backend
    itself doesn't need these (it sizes its memory-utilization flag from live
    free-VRAM telemetry, not the manifest), but `synapse_recommend`'s fit
    reports and any other manifest-metadata consumer silently read a 0-byte
    model as "trivially fits" regardless of its real size — misleading, not
    a crash. Best-effort: returns an all-zero dict (same shape as a failed
    gguf_summary, never raises) if config.json is missing/unparseable or no
    *.safetensors files are found, so a metadata gap here degrades to
    exactly today's silent-zero behavior rather than failing the pull."""
    root = Path(local_dir)
    result = {"n_bytes": 0, "n_layers": 0, "is_moe": False, "n_experts": 0,
              "n_experts_used": 0, "dense_bytes": 0, "expert_bytes": 0,
              "ctx_length": 0, "arch": "", "n_kv_heads": 0, "head_dim": 0,
              "quant_method": "", "quant_bits": 0,
              "n_kv_layers": 0, "n_recurrent_layers": 0, "is_recurrent_only": False,
              "ssm_d_conv": 0, "ssm_conv_width": 0, "ssm_state_elems": 0}
    try:
        n_bytes = sum(f.stat().st_size for f in root.glob("*.safetensors"))
        n_bytes += sum(f.stat().st_size for f in root.glob("*.bin")
                       if f.name != "training_args.bin")
        cfg = json.loads((root / "config.json").read_text())
    except (OSError, ValueError):
        return result
    # Multimodal configs (Qwen3-VL family etc.) nest the LLM's own fields
    # under text_config; text-only configs carry them at the top level.
    tc = cfg.get("text_config", cfg)
    n_experts = int(tc.get("num_experts") or tc.get("num_local_experts") or 0)
    # Hybrid recurrent/attention configs (Qwen3.5/3.6's Gated DeltaNet mix,
    # the primary reference workload as of 2026-07-29 — see CLAUDE.md's
    # Reference Workload Rule) list a per-layer schedule under `layer_types`
    # (or the older `layers_block_type` spelling) — same key gllm's own
    # Qwen3_5Model reads (synapse_engine/gllm/models/qwen3_5.py:107-121,
    # `_get_layer_types`/`_GLOBAL_LAYER_TYPE_ATTRS`). Counting it directly is
    # exact — no interval-arithmetic guess needed, unlike the GGUF path where
    # `full_attention_interval` is the only signal available.
    layer_types = tc.get("layer_types") or tc.get("layers_block_type") or []
    n_layers = int(tc.get("num_hidden_layers") or len(layer_types) or 0)
    n_recurrent_layers = 0
    if layer_types:
        n_recurrent_layers = sum(1 for t in layer_types if t in ("linear_attention", "linear_attn"))
    is_recurrent_only = (cfg.get("model_type") or "").lower() in ("mamba", "mamba2", "falcon_mamba")
    if is_recurrent_only:
        n_recurrent_layers = n_layers
    n_kv_layers = n_layers - n_recurrent_layers
    # Recurrent-state geometry, normalized to the same two quantities
    # estimate_ssm_state_gb() uses regardless of architecture family:
    #   ssm_conv_width  = channels fed into the causal conv1d
    #   ssm_state_elems = per-layer scalar element count of the temporal/
    #                     recurrent state tensor (excludes the conv state)
    # Qwen3.5/3.6 Gated DeltaNet (synapse_engine/gllm/models/qwen3_5.py:
    # `_build_ssm_cache_config`, the authoritative shapes gLLM actually
    # allocates): conv_dim = 2*key_dim + value_dim, state = num_v_heads *
    # head_v_dim * head_k_dim. Plain Mamba/Mamba2 HF configs spell the same
    # geometry as `conv_kernel`/`intermediate_size`/`state_size`/`n_groups`,
    # matching llama.cpp's ssm_d_inner/ssm_d_state/ssm_n_group naming, where
    # conv_width = d_inner + 2*n_group*d_state and state_elems = d_state*d_inner.
    ssm_d_conv = 0
    ssm_conv_width = 0
    ssm_state_elems = 0
    if tc.get("linear_conv_kernel_dim") is not None:
        key_dim = int(tc.get("linear_num_key_heads") or 0) * int(tc.get("linear_key_head_dim") or 0)
        value_dim = int(tc.get("linear_num_value_heads") or 0) * int(tc.get("linear_value_head_dim") or 0)
        ssm_d_conv = int(tc.get("linear_conv_kernel_dim") or 0)
        ssm_conv_width = 2 * key_dim + value_dim
        ssm_state_elems = (int(tc.get("linear_num_value_heads") or 0)
                            * int(tc.get("linear_value_head_dim") or 0)
                            * int(tc.get("linear_key_head_dim") or 0))
    elif tc.get("conv_kernel") is not None or tc.get("ssm_conv_kernel") is not None:
        d_conv = int(tc.get("conv_kernel") or tc.get("ssm_conv_kernel") or 0)
        # Live-verified 2026-07-29 against a real Mamba-2 checkpoint
        # (AntonV/mamba2-130m-hf): HF's actual Mamba2Config does NOT always
        # spell out `intermediate_size` explicitly -- this checkpoint's
        # config only has `expand`+`hidden_size`, from which
        # `intermediate_size` is DERIVED (`expand * hidden_size`, the same
        # formula gllm.models.mamba._mamba2_intermediate_size() uses). Without
        # this fallback, d_inner silently resolved to 0, and so did
        # ssm_state_elems (`d_state * d_inner`) -- the exact zero-charged-
        # recurrent-state bug this whole Phase-1 fix exists to close, just
        # reintroduced on the one config-field-naming variant this branch's
        # original two aliases didn't anticipate.
        d_inner = int(
            tc.get("intermediate_size") or tc.get("ssm_intermediate_size")
            or (int(tc.get("expand") or 0) * int(tc.get("hidden_size") or 0))
            or 0
        )
        d_state = int(tc.get("state_size") or tc.get("ssm_state_size") or 0)
        n_group = int(tc.get("n_groups") or tc.get("ssm_n_groups") or 1)
        ssm_d_conv = d_conv
        ssm_conv_width = d_inner + 2 * n_group * d_state
        ssm_state_elems = d_state * d_inner
    result.update({
        "n_bytes": n_bytes,
        "n_layers": n_layers,
        "ctx_length": int(tc.get("max_position_embeddings") or 0),
        "arch": (cfg.get("architectures") or [_ARCH_FROM_MODEL_TYPE.get(
            cfg.get("model_type", ""), cfg.get("model_type", ""))])[0],
        "n_kv_heads": int(tc.get("num_key_value_heads") or 0),
        "head_dim": int(tc.get("head_dim") or 0),
        "n_experts": n_experts,
        "n_experts_used": int(tc.get("num_experts_per_tok") or 0),
        "is_moe": n_experts > 0,
        "n_kv_layers": n_kv_layers,
        "n_recurrent_layers": n_recurrent_layers,
        "is_recurrent_only": is_recurrent_only,
        "ssm_d_conv": ssm_d_conv, "ssm_conv_width": ssm_conv_width,
        "ssm_state_elems": ssm_state_elems,
        # No per-tensor expert/dense split available without opening the
        # safetensors headers (unlike GGUF's per-tensor names) — a real,
        # accepted granularity gap, not guessed: dense_bytes/expert_bytes
        # stay 0 for MoE safetensors models rather than a wrong split.
    })
    result.update(read_quant_config(local_dir))
    return result


_ARCH_CACHE: set[str] | None = None


def _engine_supported_archs() -> set[str]:
    """Architectures the vendored llama.cpp build actually recognizes, parsed
    from its own LLM_ARCH_NAMES table (src/llama-arch.cpp) so this never
    drifts from whatever engine version is actually built — some GGUFs (e.g.
    Ollama-only community models) declare a general.architecture our engine
    doesn't implement yet, and llama-server fails that late with a raw
    'unknown model architecture' crash instead of a clear pre-flight error."""
    global _ARCH_CACHE
    if _ARCH_CACHE is not None:
        return _ARCH_CACHE
    archs: set[str] = set()
    try:
        text = (ENGINE_SRC_DIR / "src" / "llama-arch.cpp").read_text()
        table = text.split("LLM_ARCH_NAMES = {", 1)[1].split("};", 1)[0]
        archs = set(re.findall(r'"([a-z0-9_.\-]+)"', table))
    except (OSError, IndexError):
        pass  # source not vendored/found — skip the check rather than false-block
    _ARCH_CACHE = archs
    return archs


_RECURRENT_ARCH_CACHE: set[str] | None = None


def _engine_recurrent_archs() -> set[str]:
    """Architectures the vendored llama.cpp build treats as PURELY recurrent
    (llm_arch_is_recurrent() in src/llama-arch.cpp — mamba, mamba2, the RWKV
    family): every layer holds a fixed-size recurrent state, none hold a
    real, context-length-scaling KV cache. Distinct from the HYBRID archs
    (jamba, falcon-h1, nemotron_h, granitehybrid, qwen35, ...) where only
    SOME layers are recurrent — those are detected per-file via
    `attention.recurrent_layers` / `full_attention_interval` above, since the
    hybrid split isn't fixed per-architecture the way it is here.

    Parsed from the engine's own two tables (LLM_ARCH_NAMES for the enum <->
    GGUF-name mapping, llm_arch_is_recurrent()'s switch for which enums
    qualify) so this can never drift from whatever engine version is
    actually built, same rationale as _engine_supported_archs()."""
    global _RECURRENT_ARCH_CACHE
    if _RECURRENT_ARCH_CACHE is not None:
        return _RECURRENT_ARCH_CACHE
    archs: set[str] = set()
    try:
        text = (ENGINE_SRC_DIR / "src" / "llama-arch.cpp").read_text()
        names_table = text.split("LLM_ARCH_NAMES = {", 1)[1].split("};", 1)[0]
        enum_to_name = dict(re.findall(r'\{\s*(LLM_ARCH_\w+)\s*,\s*"([a-z0-9_.\-]+)"', names_table))
        fn_body = text.split("bool llm_arch_is_recurrent(", 1)[1].split("default:", 1)[0]
        for enum_name in re.findall(r'case\s+(LLM_ARCH_\w+)\s*:', fn_body):
            gguf_name = enum_to_name.get(enum_name)
            if gguf_name:
                archs.add(gguf_name)
    except (OSError, IndexError):
        pass  # source not vendored/found — degrade to "no arch is recurrent-only"
    _RECURRENT_ARCH_CACHE = archs
    return archs


_TORCH_ARCH_CACHE: set[str] | None = None


def _torch_engine_supported_archs() -> set[str]:
    """Architectures the vendored synapse torch engine (gLLM) actually
    implements, parsed from its own model_loader.get_model_type() if/elif
    chain (synapse_engine/gllm/model_loader.py) so this never drifts from
    whatever engine version is actually vendored — same pattern as
    _engine_supported_archs() for llama.cpp."""
    global _TORCH_ARCH_CACHE
    if _TORCH_ARCH_CACHE is not None:
        return _TORCH_ARCH_CACHE
    archs: set[str] = set()
    try:
        text = (_REPO_DIR / "synapse_engine" / "gllm" / "model_loader.py").read_text()
        archs = set(re.findall(r'self\.architecture == "([A-Za-z0-9_]+)"', text))
    except OSError:
        pass  # fork not vendored/found — skip the check rather than false-block
    _TORCH_ARCH_CACHE = archs
    return archs


def _read_ram_total_mb() -> int:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def _read_ram_available_mb() -> int:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def doctor(probe_feeders: bool = True) -> dict:
    """Aggregate hardware view: host GPU/RAM + every cluster feeder's
    GPU/RAM, plus gb-synapse readiness (engine built, HF token set). This is
    `greenboost doctor`'s data source."""
    from gb_nvml import get_nvml
    nv = get_nvml(0)
    _, host_free_mb, host_total_mb, _ = nv.mem()
    host_ram_mb = _read_ram_total_mb()

    fs = gb_cluster.feeders(probe=probe_feeders)
    feeder_reports = []
    agg_vram_mb, agg_ram_mb = host_total_mb, host_ram_mb
    for f in fs:
        if f.online:
            feeder_reports.append({
                "hostname": f.hostname or f.ip, "online": True,
                "vram_total_mb": f.t1_total_mb, "vram_free_mb": f.t1_free_mb,
                "ram_total_mb": f.t2_total_mb, "ram_free_mb": f.t2_free_mb,
            })
            agg_vram_mb += f.t1_total_mb
            agg_ram_mb += f.t2_total_mb
        else:
            feeder_reports.append({"hostname": f.hostname or f.ip, "online": False,
                                    "error": f.error})

    torch_env = gb_synapse_backends._torch_env_dir()
    return {
        "host_gpu_name": nv.device_name() or "unknown GPU",
        "host_vram_total_mb": host_total_mb, "host_vram_free_mb": host_free_mb,
        "host_ram_total_mb": host_ram_mb,
        "feeders": feeder_reports,
        "aggregate_vram_mb": agg_vram_mb,
        "aggregate_ram_mb": agg_ram_mb,
        "engine_installed": engine_installed(),
        "engine_version": engine_version(),
        "hf_token_set": hf_token() is not None,
        "cluster_configured": bool(fs),
        "torch_engine_ready": torch_env is not None,
        "torch_engine_env": str(torch_env) if torch_env else "",
    }


# ---- fit + throughput estimate ("greenboost recommend") ----

_BITS_PER_PARAM = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0,
    "Q8_0": 8.5, "Q6_K": 6.6, "Q5_K_M": 5.7, "Q5_0": 5.7,
    "Q4_K_M": 4.83, "Q4_K_S": 4.58, "Q4_0": 4.5, "IQ4_XS": 4.25,
    "Q3_K_M": 3.9, "Q3_K_S": 3.5, "IQ3_XS": 3.3,
    "Q2_K": 2.63, "IQ2_XS": 2.4,
}
_DEFAULT_BITS_PER_PARAM = 4.83  # Q4_K_M-ish fallback for unrecognized quant labels


def _approx_param_count_b(n_bytes: int, quant: str) -> float:
    bits = _BITS_PER_PARAM.get(quant.upper(), _DEFAULT_BITS_PER_PARAM)
    return (n_bytes * 8) / bits / 1e9


def estimate_kv_gb(ctx: int, n_bytes: int, quant: str,
                   n_layers: int = 0, n_kv_heads: int = 0, head_dim: int = 0,
                   kv_bytes_per_elem: float = 1.0) -> float:
    """KV-cache size (GiB) for `ctx` tokens.

    Preferred path (all of n_layers/n_kv_heads/head_dim > 0): the exact GGUF
    formula 2 (K+V) * n_layers * n_kv_heads * head_dim * bytes/elem * ctx.
    `kv_bytes_per_elem` defaults to 1.0 (q8_0 KV, matching serve's
    --cache-type-k/v q8_0). This replaces the param-count bucket heuristic,
    which was wrong by ~100x on GQA architectures (verified live OOM on
    Qwable-9B: predicted ~0.06 GB at 262144 ctx, actually needed ~8 GB).

    Fallback (geometry unknown, e.g. an old manifest entry): mirrors
    estimate_kv_mb() in greenboost_setup.sh:952 — a param-count bucketed
    bytes/token table with q8_0 KV halving."""
    if n_layers > 0 and n_kv_heads > 0 and head_dim > 0:
        bytes_per_tok = 2 * n_layers * n_kv_heads * head_dim * kv_bytes_per_elem
        return (ctx * bytes_per_tok) / (1024 ** 3)
    params_b = _approx_param_count_b(n_bytes, quant)
    if params_b >= 100:
        bytes_per_tok = 3584
    elif params_b >= 60:
        bytes_per_tok = 2560
    elif params_b >= 25:
        bytes_per_tok = 1331
    elif params_b >= 10:
        bytes_per_tok = 819
    elif params_b >= 5:
        bytes_per_tok = 512
    elif params_b >= 2:
        bytes_per_tok = 256
    else:
        bytes_per_tok = 128
    bytes_per_tok //= 2  # q8_0 KV cache
    return (ctx * bytes_per_tok) / (1024 ** 3)


def estimate_ssm_state_gb(n_recurrent_layers: int, ssm_d_conv: int, ssm_conv_width: int,
                          ssm_state_elems: int, n_seq_max: int = 1) -> float:
    """Selective-SSM recurrent-state size (GiB) — the counterpart to
    estimate_kv_gb() for the layers n_kv_layers EXCLUDES. Unlike KV cache,
    this is CONSTANT in ctx (arXiv:2312.00752, the Mamba paper: the whole
    point of a selective SSM's recurrence is a fixed-size state instead of a
    growing cache) — callers must add this once, never multiply it by ctx.

    Formula ported verbatim from the vendored llama.cpp's own allocator
    (llama-hparams.cpp:183-221, `n_embd_r()`/`n_embd_s()`; the ctor call at
    llama-model.cpp:2091-2099 hardcodes GGML_TYPE_F32 for both tensors, so
    this is NOT parameterized by kv_bytes_per_elem the way estimate_kv_gb()
    is — `--cache-type-k/v` does not apply to recurrent state):

        n_embd_r = (ssm_d_conv - 1) * ssm_conv_width     # conv_state
        n_embd_s = ssm_state_elems                        # temporal/ssm_state
        bytes    = 4 * n_recurrent_layers * (n_embd_r + n_embd_s) * n_seq_max

    `n_seq_max` mirrors llama.cpp's `cparams.n_seq_max` (`--parallel`);
    gb-synapse's serve_gguf() never passes `--parallel` so this is 1 in
    practice today. F32 is also a safe, slightly-conservative choice for the
    torch/gLLM path (SynapseTorchBackend): gLLM's own SSMCacheConfig uses F32
    for the temporal state but the checkpoint's activation dtype (typically
    bf16, half the bytes) for the conv state — this estimator's F32-for-both
    assumption overestimates the conv term rather than risking OOM by
    underestimating it, consistent with `_compute_reserve_gb`'s "undershooting
    is a failed load, overshooting only costs a layer" principle elsewhere in
    this file.

    Returns 0.0 when geometry is unknown (old manifest entry, or a plain
    transformer with no recurrent layers) — correct in both cases, not a gap:
    a plain transformer genuinely has no recurrent state to size."""
    if n_recurrent_layers <= 0 or ssm_d_conv <= 0:
        return 0.0
    n_embd_r = (ssm_d_conv - 1) * ssm_conv_width
    n_embd_s = ssm_state_elems
    total_bytes = 4 * n_recurrent_layers * (n_embd_r + n_embd_s) * max(1, n_seq_max)
    return total_bytes / (1024 ** 3)


def _entry_ssm_gb(entry: "ModelEntry", n_seq_max: int = 1) -> float:
    """estimate_ssm_state_gb() from a ModelEntry's own fields — the call
    every placement site below needs, factored out so each one doesn't
    repeat the same 4-field unpack."""
    return estimate_ssm_state_gb(entry.n_recurrent_layers, entry.ssm_d_conv,
                                  entry.ssm_conv_width, entry.ssm_state_elems,
                                  n_seq_max=n_seq_max)


@dataclass
class FitReport:
    name: str
    quant: str
    weights_gb: float
    kv_gb: float
    total_gb: float
    fits_vram: bool
    overflow_gb: float
    est_tok_s: float
    ctx: int
    note: str = ""
    measured: bool = False   # est_tok_s is a real client-measured average, not the heuristic
    ssm_state_gb: float = 0.0  # recurrent-state footprint, constant in ctx (see estimate_ssm_state_gb)


MEASURED_TOK_S_FILE = MODEL_STORE_DIR / "measured_tok_s.json"
_MEASURED_TOK_S_MAX_SAMPLES = 20


def _df_emit_tok_s(model: str, tok_s: float, source: str = "") -> None:
    """Record a real measured tok/s sample to the dataflux log , the one
    number that closes the loop between orchestration decisions (tier_move/
    quantize/turboquant_activate) and what they actually bought. Best-effort,
    never raises, same contract as every other dataflux emit call site.

    `source` identifies WHICH measurement this is: gb_synapse_api.py (the
    proxy, "proxy") measures every client on the gb-synapse port from
    first-content-token to last; greenboost-cli (repl.py, "cli") measures
    its own turn end-to-end. These are two different, both-valid vantage
    points on the SAME turn, not duplicate samples of one true number — see
    summarize()'s per-source rollup in gb_dataflux.py for why blending them
    into one average was wrong (real incident 2026-08-01: proxy=0.3,
    cli=2.4, engine truth=2.18 — the blended avg matched neither)."""
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "node": "host", "label": "gb_synapse", "kind": "tok_s_measured",
            "n_items": 1, "items": [model], "duration_s": 0.0, "status": "ok",
            "model": model, "tok_s": round(tok_s, 1),
            **({"source": source} if source else {}),
        })
    except Exception:
        pass


def record_measured_tok_s(model: str, tok_s: float, source: str = "") -> None:
    """Append a real, client-observed decode speed for `model` — fed by
    greenboost-cli after each final answer (TurnComplete.tok_s), closing the
    gap _estimate_tok_s()'s docstring flags: "A --measure mode that runs a
    real warmup is future work." recommend() prefers this rolling average
    over the bandwidth heuristic whenever samples exist for a model.
    Best-effort: silently skipped without write access to MODEL_STORE_DIR,
    same as index_ollama_models()'s persistence — this is a nice-to-have
    calibration aid, not something worth failing a turn over.

    `source` (optional, "proxy" | "cli") — see _df_emit_tok_s's docstring.
    The rolling MEASURED_TOK_S_FILE average stays source-blind (it already
    only exists as an estimator input for recommend(), not a precision
    metric) — only the dataflux event itself carries the distinction."""
    if tok_s <= 0:
        return
    _df_emit_tok_s(model, tok_s, source)
    try:
        samples = json.loads(MEASURED_TOK_S_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        samples = {}
    history = samples.get(model, [])
    history.append(round(tok_s, 1))
    samples[model] = history[-_MEASURED_TOK_S_MAX_SAMPLES:]
    try:
        MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
        MEASURED_TOK_S_FILE.write_text(json.dumps(samples, indent=2))
    except PermissionError:
        pass


def record_prompt_cache_sample(model: str, ttft_ms: "float | None",
                                hit_pct: "float | None", reused_tokens: int = 0) -> None:
    """Record one proxy-observed host-memory prompt-cache outcome (TTFT +
    reused-vs-total prompt token share) — the measurement GB-Semantics'
    `ttft_ms`/`prompt_cache_hit_pct` metrics resolve from. Fed by
    gb_synapse_api.py right where record_measured_tok_s already is, same
    best-effort/never-raise contract; silently skips when neither value is
    known (e.g. a non-streaming request that never reached a first token)."""
    if ttft_ms is None and hit_pct is None:
        return
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "node": "host", "label": "gb_synapse", "kind": "prompt_cache",
            "n_items": 1, "items": [model], "duration_s": 0.0, "status": "ok",
            "model": model,
            **({"ttft_ms": round(ttft_ms, 1)} if ttft_ms is not None else {}),
            **({"hit_pct": round(hit_pct, 1)} if hit_pct is not None else {}),
            "reused_tokens": reused_tokens,
        })
    except Exception:
        pass


def _measured_tok_s(model: str) -> "float | None":
    try:
        samples = json.loads(MEASURED_TOK_S_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    history = samples.get(model)
    if not history:
        return None
    return round(sum(history) / len(history), 1)


# Rule: link bandwidths derive from the EXECUTING node, never reference-box
# literals. When detection fails on both NVML and nvidia-smi, the 0.0
# sentinel propagates honestly (gb_topology._detect_vram_bw_gb_s's own
# convention) instead of assuming a reference card's numbers — a fabricated
# 670 GB/s wildly over-estimates tok/s on e.g. an 8 GB feeder (the exact class
# of bug _auto_budgets's own comment warns about for VRAM budgets).
_PCIE_STREAM_EFF = 0.78          # observed streaming vs theoretical peak (≈25/32 on gen4 x16)
_TOKS_FLOOR = 0.5
_link_bw_cache: "tuple[float, float] | None" = None


def _emit_bw_undetectable(reason: str) -> None:
    try:
        import gb_dataflux
        gb_dataflux.emit({"node": "host", "label": "synapse",
                          "kind": "bw_undetectable", "status": "warn",
                          "reason": reason})
    except Exception:
        pass


def _link_bandwidths() -> "tuple[float, float]":
    """(vram_bw_gb_s, pcie_bw_gb_s) for the LOCAL device, detected once and
    cached. Delegates VRAM bandwidth to gb_topology (single detection path ,
    do not re-probe NVML here). PCIe: detected gen × lanes via gb_topology's
    per-lane table, scaled by the observed streaming efficiency. Either term
    is the 0.0 sentinel, never a reference-box constant, when undetectable —
    callers must treat 0.0 as "unknown", not "no bandwidth"."""
    global _link_bw_cache
    if _link_bw_cache is not None:
        return _link_bw_cache
    from gb_topology import get_topology, _PCIE_BW_PER_LANE_GBS
    topo = get_topology()
    vram_bw = topo.vram_bw_gb_s
    if vram_bw <= 0:
        _emit_bw_undetectable("vram")
    pcie_bw = 0.0
    try:
        pcie_bw = (_PCIE_BW_PER_LANE_GBS.get(topo.pcie_gen, 0.0)
                   * topo.pcie_lanes * _PCIE_STREAM_EFF)
    except Exception:
        pass
    if pcie_bw <= 0:
        _emit_bw_undetectable("pcie")
    _link_bw_cache = (vram_bw, pcie_bw)
    return _link_bw_cache


def _estimate_tok_s(active_gb: float, budget_gb: float) -> float:
    """Heuristic memory-bandwidth-bound tok/s estimate — NOT a measured
    number. Assumes decode is bandwidth-bound on reading the active weight
    bytes once per token (dense: full weights; MoE: only routed experts).
    Degrades linearly from local VRAM bandwidth toward a PCIe/DDR floor as
    the active set overflows the aggregate VRAM budget. Real throughput also
    depends on batch size, kernel efficiency, and cross-node RPC latency —
    treat this as an order-of-magnitude planning aid, not a benchmark. A
    `--measure` mode that runs a real llama-server warmup is future work.
    Returns the 0.0 sentinel (never a fabricated floor) when bandwidth is
    undetectable on this node — callers append a note rather than trusting
    the number (see `recommend()`).
    """
    if active_gb <= 0:
        return 0.0
    vram_bw, pcie_bw = _link_bandwidths()   # node-derived, not literals (rule)
    if vram_bw <= 0:
        return 0.0   # undetectable — honest sentinel, not a guessed floor
    if active_gb <= budget_gb:
        eff_bw = vram_bw
    else:
        overflow_frac = min(1.0, (active_gb - budget_gb) / active_gb)
        eff_bw = vram_bw * (1 - overflow_frac) + pcie_bw * overflow_frac
    return max(_TOKS_FLOOR, round(eff_bw / active_gb, 1))


def recommend(ctx: int = 65536, probe_feeders: bool = True) -> list[FitReport]:
    """Fit + throughput estimate for every model in the manifest against the
    live cluster's aggregate VRAM budget. Sorted best-fit first."""
    d = doctor(probe_feeders=probe_feeders)
    budget_gb = d["aggregate_vram_mb"] / 1024
    host_budget_gb = d["host_vram_total_mb"] / 1024  # host-only, for the split advisory
    reports = []
    for entry in list_models():
        weights_gb = entry.n_bytes / (1024 ** 3)
        # is_recurrent_only: n_kv_layers==0 is the CORRECT count (no real
        # attention layer at all) — `or entry.n_layers` must not paper over
        # that with "unknown, assume every layer is real attention".
        kv_layers = 0 if entry.is_recurrent_only else (entry.n_kv_layers or entry.n_layers)
        kv_gb = estimate_kv_gb(ctx, entry.n_bytes, entry.quant,
                               n_layers=kv_layers,
                               n_kv_heads=entry.n_kv_heads,
                               head_dim=entry.head_dim)
        ssm_gb = _entry_ssm_gb(entry)
        total_gb = weights_gb + kv_gb + ssm_gb
        active_gb = weights_gb
        note = ""
        if entry.is_moe and entry.n_experts and entry.n_experts_used:
            active_gb = weights_gb * entry.n_experts_used / entry.n_experts
            note = "MoE: active-expert subset"
        fits = total_gb <= budget_gb
        overflow_gb = max(0.0, total_gb - budget_gb)
        # fp8-floor advisory: a model that only fits once feeder VRAM is added
        # should be served via RPC tensor-split, NOT re-quantized lower , the
        # cluster holds it at full quant.
        if fits and total_gb > host_budget_gb:
            _adv = "cluster holds it , prefer RPC split over lower quant"
            note = f"{note}; {_adv}" if note else _adv
        # DI-2: when it doesn't fit even the aggregate cluster budget, report
        # a partial-offload estimate (ollama capacity_fit port) instead of
        # just "doesn't fit" — a real n_offload/n_layers number an operator
        # can act on (--n-gpu-layers) rather than an all-or-nothing verdict.
        if not fits and entry.n_layers > 0:
            try:
                import gb_placement
                device_free_gb = [d["host_vram_free_mb"] / 1024.0]
                device_free_gb += [f["vram_free_mb"] / 1024.0 for f in d["feeders"]
                                   if f.get("online")]
                layer_gb = weights_gb / entry.n_layers
                cf = gb_placement.capacity_fit([layer_gb] * entry.n_layers, device_free_gb)
                _adv = (f"partial offload: {cf.n_offload}/{entry.n_layers} layers fit "
                       f"(--n-gpu-layers {cf.n_offload})")
                note = f"{note}; {_adv}" if note else _adv
                try:
                    import gb_dataflux
                    gb_dataflux.emit({"kind": "capacity_fit", "model": entry.name,
                                      "n_layers": entry.n_layers, "n_offload": cf.n_offload,
                                      "fraction": cf.fraction, "per_device_layers": cf.per_device_layers})
                except Exception:
                    pass
            except Exception:
                pass
        measured = _measured_tok_s(entry.name)
        est = measured if measured is not None else _estimate_tok_s(active_gb + kv_gb, budget_gb)
        if measured is not None:
            note = f"{note} (measured)".strip() if note else "measured"
        elif est <= 0:
            _adv = "tok/s estimate unavailable , bandwidth undetectable on this node"
            note = f"{note}; {_adv}" if note else _adv
        reports.append(FitReport(entry.name, entry.quant, round(weights_gb, 2), round(kv_gb, 2),
                                  round(total_gb, 2), fits, round(overflow_gb, 2), est, ctx, note,
                                  measured=measured is not None, ssm_state_gb=round(ssm_gb, 3)))
    return sorted(reports, key=lambda r: (not r.fits_vram, r.overflow_gb))


# ---------------------------------------------------------------------------
# 4. Serve + RPC cluster split orchestration
# ---------------------------------------------------------------------------

@dataclass
class ServerState:
    model: str
    llama_pid: int
    proxy_pid: int
    port: int
    internal_port: int
    tensor_split: str
    feeders: list = field(default_factory=list)
    started_ts: float = 0.0
    engine: str = ""
    # False = /health hadn't gone fully green within SERVE_READY_GRACE_S when
    # serve() returned (large GGUFs can legitimately take minutes) — the
    # server IS running and will finish loading, but a caller hitting it
    # immediately can get a connection-refused/still-loading response rather
    # than a real answer. True for every ServerState constructed before this
    # field existed (default, so old persisted run-state JSON still parses).
    ready: bool = True
    # None = proxy bound :port and answered /health. Non-None = it did not
    # (see _wait_proxy_ready) — most often another process already holds
    # :port (raw ollama.service on a box mid-migration off it). The ENGINE
    # can be genuinely healthy (`ready=True`) while this is set: they are
    # different failure surfaces, both real, found live 2026-07-16.
    proxy_error: str | None = None
    # Second, minimal llama-server (--embeddings --pooling mean) alongside
    # the primary engine, started when GB_SYNAPSE_EMBED_MODEL is set at
    # serve() time (see _maybe_serve_embedding). 0 = no embeddings engine
    # configured for this serve — default so old persisted run-state JSON
    # still parses.
    embed_pid: int = 0
    embed_internal_port: int = 0
    # Served context window + KV precision + GPU layer count actually used —
    # 0/""/0 for backends that don't report them (torch/transformers/diffusers)
    # or old persisted run-state JSON. Before this field existed, ctx/kv_type/
    # placement were passed into _launch_proxy_and_record's **serve_facts and
    # went ONLY into the dataflux synapse_serve event, never into the
    # run-state itself — so a client asking "what window is actually being
    # served right now" (gb-cli's gb_synapse_ctx(), synapse_ps) had no
    # authoritative answer and fell back to a guessed constant. See
    # gb_synapse_ctx() in greenboost-cli/greenboost_cli/environment/settings.py.
    ctx: int = 0
    kv_type: str = ""
    n_gpu_layers: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _safe_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]', '_', name)


def _run_state_path(model: str) -> Path:
    return RUN_DIR / f"{_safe_name(model)}.json"


def _run_log_path(label: str) -> Path:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    return RUN_DIR / f"{_safe_name(label)}.log"


def log_path(model: str) -> Path:
    """Public accessor for a running server's log file (e.g. for a caller's
    own `logs [N]` command — see greenboost-cli's `/llamaserve logs`)."""
    return _run_log_path(model)


def _pcore_threads() -> int:
    """Thread count matched to P-cores from the GreenBoost topology profile.

    Ported from greenboost-cli's _llamacpp_pcore_threads() as part of
    consolidating llama-server process lifecycle into gb-synapse. Reads
    /etc/greenboost/profiles/default.md (written at install time by
    greenboost_setup.sh autodetect); falls back to half of os.cpu_count()
    (a reasonable P-core approximation on heterogeneous CPUs) if the profile
    is absent or unparseable.

    HOST-only by design: this sets llama-server's own --threads, which runs on
    the host. The feeder's rpc-server has no upstream thread flag, so per-feeder
    P-core counts (now in the cluster topology registry) are applied via
    gb_cluster.feeder_env's OMP_NUM_THREADS/GB_FEEDER_THREADS for torch stages,
    not here. Don't "fix" this to read a feeder's topology.
    """
    try:
        profile = Path("/etc/greenboost/profiles/default.md")
        if profile.exists():
            for line in profile.read_text().splitlines():
                line = line.strip()
                if line.startswith("p_core_cpus:"):
                    val = line.split(":", 1)[1].strip().strip('"')
                    cpus = [x.strip() for x in val.split(",") if x.strip()]
                    if cpus:
                        return len(cpus)
    except OSError:
        pass
    total = os.cpu_count() or 8
    return max(4, total // 2)


def _write_run_state(state: ServerState) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _run_state_path(state.model).write_text(json.dumps(asdict(state), indent=2))


def _read_run_states() -> list[ServerState]:
    if not RUN_DIR.is_dir():
        return []
    out = []
    for f in RUN_DIR.glob("*.json"):
        try:
            out.append(ServerState(**json.loads(f.read_text())))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return out


def _pid_alive(pid: int) -> bool:
    """True only if the process is running — a ZOMBIE is dead.

    signal-0 alone succeeds against a zombie, and an engine that crashed on
    startup stays a zombie for as long as its parent (greenboost-cli, which
    spawned it and never waits) lives. That made every liveness check in this
    module lie: serve() "reused" a crashed server, status reported it up, and
    the proxy kept relaying to a corpse until the client saw a truncated
    stream. Read the real state instead of asking whether the PID exists.
    """
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_bytes()
        # "pid (comm) STATE ..." — comm may contain spaces/parens, so index
        # from the LAST ')'.
        return stat[stat.rindex(b")") + 2: stat.rindex(b")") + 3] != b"Z"
    except (OSError, ValueError):
        return True     # no procfs (non-Linux): fall back to the signal-0 answer


# Legacy / de-facto model names that consumers (ai-forge, greenboost-cli
# scripts, older docs) still reference by their OLD name, resolved to
# whichever model is the CURRENT standing reference workload (see this
# repo's own CLAUDE.md, Reference Workload Rule). One source of truth here
# means a future reference-model swap updates every caller at once instead
# of requiring a sweep across every repo that names a model directly —
# confirmed real need 2026-07-27: ai-forge alone had 20+ files hardcoding
# "qwen36-coder:studio" (the previous reference's Ollama-Modelfile tag) as
# its de facto "the local coder model" constant.
MODEL_ALIASES = {
    "qwen36-coder:studio": "Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF",
    "rafw007/qwen36-a3b-claude-coder:latest": "Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF",
    "satgeze/qwen36-35b-uncensored-1m": "Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF",
}

# Owner-chosen default ctx/KV for specific models, applied only when the
# caller left ctx unset (<=0) and didn't pin GB_SYNAPSE_KV. Not a general
# algorithm change to _pick_kv_type/_solve_ctx_and_layers — a per-model
# override for cases where the auto-picker's answer is technically correct
# but not what's wanted. Real incident (2026-08-01): this box's single
# 11.2GB Blackwell card has no feeder online, so the reference workload's
# 14.1GB weights only partly fit; _pick_kv_type saw the shim's T2-inflated
# "budget" and chose ctx=65536/kv=f16 (4.4GB of KV), leaving ~3.4GB of
# VRAM headroom that could instead hold more of the weights. Owner chose
# ctx=32768/q8_0 (KV ~1.1GB) to trade context length for placement headroom
# — still comfortably above GB-CLI's own baseline overhead (~12k tokens).
MODEL_CTX_KV_DEFAULTS = {
    "Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF":
        (32768, "q8_0"),
}


def _manifest_lookup(spec: str, manifest: dict) -> "ModelEntry | None":
    """Resolve `spec` against the manifest by key OR by the `repo` field an
    HF-pulled entry stores. `pull()`/`_pull_torch` key every HF-sourced entry
    by the repo's bare name (`name or repo.split("/")[-1]`), never by
    "org/name" — so before this helper existed, typing the model's real
    "org/repo" spelling (as printed by `pull`, `list`, and the model card)
    missed the manifest entirely and fell into `_resolve_model`'s "/" in spec
    branch, triggering a network `pull()` for a model already on disk.
    Real incident, 2026-07-28: DavidAU/Qwen3.6-27B-Fable-Fusion-...-GGUF,
    already pulled and served OK 18 times, failed with
    "No module named 'huggingface_hub'" the moment the org prefix was typed.

    Also accepts "org/repo:QUANT" when an entry's own `repo` matches and
    QUANT substring-matches its `quant` (mirrors `pull()`'s own quant
    matching ahead of `hf_hub_download`). Raises ValueError naming the
    candidates if more than one entry's quant matches — this repo already
    ships ambiguously-substring-matching quants for the reference model
    (MTP-Q4_K_M vs Q4_K_M, see the Reference Workload Rule in CLAUDE.md), so
    silently picking one here would repeat the exact mistake `pull()`
    already guards against for a fresh download.

    Case-insensitive throughout."""
    if spec in manifest:
        return manifest[spec]

    spec_lower = spec.lower()
    for entry in manifest.values():
        if spec_lower == entry.name.lower():
            return entry
    for entry in manifest.values():
        if entry.repo and entry.repo.lower() == spec_lower:
            return entry

    if ":" in spec:
        repo_part, _, quant_part = spec.rpartition(":")
        repo_lower, quant_upper = repo_part.lower(), quant_part.upper()
        candidates = [e for e in manifest.values()
                      if e.repo and e.repo.lower() == repo_lower
                      and quant_upper in (e.quant or "").upper()]
        if len(candidates) > 1:
            names = ", ".join(sorted(e.name for e in candidates))
            raise ValueError(
                f"{spec!r} matches more than one manifest entry by quant "
                f"substring: {names} — use the exact manifest name instead."
            )
        if candidates:
            return candidates[0]

    return None


def _resolve_model(spec: str) -> ModelEntry:
    spec = MODEL_ALIASES.get(spec, spec)
    manifest = _load_manifest()
    hit = _manifest_lookup(spec, manifest)
    if hit is not None:
        return hit
    if "/" in spec:  # looks like an HF repo spec — pull it
        return pull(spec)
    blob = ollama_model_blob(spec)
    if blob:
        meta = gguf_summary(blob)
        entry = ModelEntry(name=spec, path=blob, source="ollama", added_ts=time.time(), **meta)
        manifest[spec] = entry
        _save_manifest(manifest)
        return entry

    spec_lower = spec.lower()
    similar = sorted({e.name for e in manifest.values()
                       if spec_lower in e.name.lower() or spec_lower in (e.repo or "").lower()})
    if similar:
        hint = f" — did you mean: {', '.join(similar[:3])}?"
    else:
        hint = (" — if this is meant to be a HuggingFace repo, include the "
                "org prefix (e.g. \"org/repo\")")
    raise KeyError(f"no such model: {spec} (not in manifest, not an HF repo, "
                   f"not an Ollama model){hint}")


def _feeder_ssh_target(feeder) -> str:
    return f"{feeder.ssh_user or 'root'}@{feeder.ip}"


def _rpc_reachable(ip: str, port: int, timeout: float = 2.0) -> bool:
    import socket
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_feeder_rpc(feeder, rpc_port: int = RPC_PORT_BASE, device: int = 0) -> bool:
    """Start rpc-server on a feeder over SSH if not already serving on
    rpc_port, under the feeder's own shim (GREENBOOST_CLUSTER=0 — local
    DDR/NVMe tier extension only; the cross-node split is RPC's job, not the
    shim's per-kernel dispatch).

    Returns True only once the port is REACHABLE FROM THIS HOST. A feeder whose
    rpc-server never came up must never reach --rpc/--tensor-split: llama.cpp
    would size the split for a device it cannot talk to and the load dies with
    "Failed to connect to <ip>:<port>" — which is precisely what a `pgrep`-only
    check (SSH says the process exists; the port is firewalled) let through.
    """
    if _rpc_reachable(feeder.ip, rpc_port):
        return True

    # rpc-server is ggml/CUDA, so it hits the same shim compat question as
    # llama-server (see gb_shim_probe.py — a real, cached, evidence-based
    # verdict, not a hardcoded assumption). Preloading a shim that breaks
    # rpc-server would take the feeder's GPU down with it, which is the
    # opposite of the point. Gate it on the same switch as the host engine —
    # the probe doesn't reach the feeder itself, but host and feeder run the
    # same engine build against the same shim build, so the host's verdict
    # is the best evidence available short of a feeder-side probe.
    import gb_shim_probe
    shim_pre = (f'LD_PRELOAD={gb_cluster.GREENBOOST_SHIM} GREENBOOST_ACTIVE=1 '
                if gb_shim_probe.shim_works_for_llama(ENGINE_DIR)[0] else "")

    # Resolve the engine ON the feeder: a Full-Install feeder has it in the
    # system dir, a dev/user build in the user dir. Assuming one path is why
    # this silently launched nothing.
    remote_cmd = (
        'for p in /usr/local/lib/greenboost/synapse/rpc-server '
        '"$HOME/.local/share/greenboost/synapse/rpc-server"; do '
        '[ -x "$p" ] && RPC="$p" && break; done; '
        '[ -n "${RPC:-}" ] || { echo "rpc-server not found — run: greenboost synapse build-engine" >&2; exit 1; }; '
        # rpc-server is dynamically linked against the ggml libraries that sit
        # NEXT TO IT, and a user-dir engine is on no default search path — so it
        # died instantly with "libggml.so.0: cannot open shared object file" and
        # the port never opened. Point the loader at its own engine directory.
        'nohup env GREENBOOST_CLUSTER=0 '
        'LD_LIBRARY_PATH="$(dirname "$RPC")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" '
        f'{shim_pre}'
        # rpc-server's --device takes a backend device NAME (CUDA0), not an
        # ordinal; "--device 0" is rejected and the server exits at once.
        f'"$RPC" --host 0.0.0.0 --port {rpc_port} --device CUDA{device} '
        '>/tmp/gb_synapse_rpc.log 2>&1 & disown'
    )
    launch = subprocess.run(["ssh", *_SSH_OPTS, _feeder_ssh_target(feeder), remote_cmd],
                            capture_output=True, text=True)
    if launch.returncode != 0:
        print(f"  [gb-synapse] feeder {feeder.ip}: rpc-server launch failed: "
              f"{launch.stderr.strip()[:200]}", flush=True)
        return False

    for _ in range(int(RPC_READY_TIMEOUT_S * 2)):   # binds only after CUDA init
        if _rpc_reachable(feeder.ip, rpc_port):
            return True
        time.sleep(0.5)
    print(f"  [gb-synapse] feeder {feeder.ip}: rpc-server did not open :{rpc_port} within "
          f"{RPC_READY_TIMEOUT_S:.0f}s (firewall? see /tmp/gb_synapse_rpc.log on the feeder)",
          flush=True)
    return False


# Distinguishing marker on every gLLM slave-rank launch so
# kill_feeder_gllm_slave() can pkill precisely, and so a leftover process
# from a previous aborted attempt doesn't get mistaken for a fresh one — see
# both functions below.
_GLLM_SLAVE_MARKER = "GB_SYNAPSE_GLLM_SLAVE=1"


def ensure_feeder_gllm_slave(feeder, master_ip: str, master_port: int,
                              pp_rank: int, pp_size: int, model_path: str) -> bool:
    """Launch a gLLM `--launch-mode slave` worker on a feeder over SSH,
    contributing PP rank `pp_rank` (TP fixed at 1 — see SynapseTorchBackend.
    serve()'s cluster branch for why: gLLM's TP fast-path all-reduce uses
    same-host-only CUDA IPC, and TP's every-layer all-reduce would dominate
    wall-clock over the feeder's 1GbE link anyway; PP only exchanges
    activations at stage boundaries).

    Unlike ensure_feeder_rpc(), this does NOT poll for readiness — a slave
    rank doesn't open a listening port the way rpc-server does, it dials
    OUT to master_ip:master_port and blocks inside dist.init_process_group()
    until the whole PP group has joined. That join is confirmed by the
    LOCAL master's own /health endpoint instead (see the caller) — this
    function only confirms the remote process launched and is still alive a
    moment later, catching immediate failures (missing venv, bad
    model_path, wrong CUDA arch) rather than a full rendezvous timeout.

    UNVALIDATED (2026-07-24): whether `model_path` needs to already be
    present locally on the feeder, or whether gLLM's own loader can pull an
    HF repo id itself at slave-rank load time, was not tested this session
    — gLLM's PP split loads each rank's own layer-slice of weights locally
    (structurally different from llama.cpp's --rpc split, where only the
    host needs the weights). If the feeder has no local copy and gLLM can't
    self-pull, this launches successfully but the slave rank fails to load
    — surfaces as the master's /health never coming up, same as any other
    join failure, and the caller degrades to host-only exactly the same
    way. Flagging so a real failure here isn't mistaken for something
    else."""
    remote_cmd = (
        'for p in "$HOME/.local/share/greenboost/synapse/torch-env" '
        '/usr/local/lib/greenboost/synapse-torch-env; do '
        '[ -x "$p/bin/python" ] && VENV="$p" && break; done; '
        '[ -n "${VENV:-}" ] || { echo "synapse torch engine venv not found — run: sudo greenboost install-synapse-engine" >&2; exit 1; }; '
        f'nohup env {_GLLM_SLAVE_MARKER} GREENBOOST_CLUSTER=0 '
        '"$VENV/bin/python" -m gllm.entrypoints.api_server '
        '--launch-mode slave '
        f'--ranks {pp_rank} --pp {pp_size} --tp 1 '
        f'--master-addr {master_ip} --master-port {master_port} '
        f'--model-path {shlex.quote(model_path)} '
        '>/tmp/gb_synapse_gllm_slave.log 2>&1 & disown'
    )
    launch = subprocess.run(["ssh", *_SSH_OPTS, _feeder_ssh_target(feeder), remote_cmd],
                            capture_output=True, text=True)
    if launch.returncode != 0:
        print(f"  [gb-synapse] feeder {feeder.ip}: gLLM slave launch failed: "
              f"{launch.stderr.strip()[:200]}", flush=True)
        return False

    # Short liveness check only — catches a launch that dies immediately
    # (missing venv/model), not a join failure (that's the caller's job via
    # the master's /health, per this function's docstring).
    time.sleep(2.0)
    check = subprocess.run(
        ["ssh", *_SSH_OPTS, _feeder_ssh_target(feeder),
         f"pgrep -f 'gllm.entrypoints.api_server.*--launch-mode slave' >/dev/null"],
        capture_output=True)
    if check.returncode != 0:
        print(f"  [gb-synapse] feeder {feeder.ip}: gLLM slave process not found 2s after "
              f"launch (crashed at startup? see /tmp/gb_synapse_gllm_slave.log on the feeder)",
              flush=True)
        return False
    return True


def kill_feeder_gllm_slave(feeder) -> None:
    """Tear down a feeder's gLLM slave rank — used when the master's
    rendezvous readiness check (SynapseTorchBackend.serve()'s cluster
    branch) times out, so a half-joined NCCL world doesn't linger holding
    the GPU. Best-effort: a feeder that's unreachable for the kill is
    already unreachable for everything else too, log and move on rather
    than raising out of a cleanup path."""
    result = subprocess.run(
        ["ssh", *_SSH_OPTS, _feeder_ssh_target(feeder),
         f"pkill -f 'gllm.entrypoints.api_server.*--launch-mode slave' || true"],
        capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [gb-synapse] feeder {feeder.ip}: gLLM slave kill failed (best-effort): "
              f"{result.stderr.strip()[:200]}", flush=True)


MTP_DRAFT_N = int(os.environ.get("GB_SYNAPSE_MTP_DRAFT_N", "3"))

GPU_FIT_MARGIN = float(os.environ.get("GB_SYNAPSE_FIT_MARGIN", "0.85"))


def _compute_reserve_gb(physical_vram_mb: float) -> float:
    """llama.cpp's graph/compute workspace per CUDA device, on top of the
    weights and KV it holds. Undershooting is a failed load, overshooting only
    costs a layer.

    GB_SYNAPSE_COMPUTE_RESERVE_GB is absolute and wins; otherwise %-derived
    per device — max(0.75 GiB, 8% of that device's VRAM) via the shared
    gb_topology.compute_reserve_gb (rule: no reference-box 1.5 GiB literal;
    gb_cluster.feeder_env uses the same formula)."""
    env = os.environ.get("GB_SYNAPSE_COMPUTE_RESERVE_GB")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    try:
        from gb_topology import compute_reserve_gb
        return compute_reserve_gb(physical_vram_mb)
    except Exception:
        print("  [gb-synapse] WARNING: gb_topology unavailable — compute reserve "
              "falls back to the reference-box 1.5 GiB", flush=True)
        return 1.5

# Architectures whose CUDA backend cannot survive a CPU/GPU split (upstream
# llama.cpp regression — see the cpu_quirk branch in serve() for the verified
# evidence matrix). Models of these archs that don't fit VRAM are served
# CPU-only instead of crash-looping. Shrink this set as upstream fixes land.
ARCH_CPU_SPLIT_BROKEN = {"qwen35moe"}


def _moe_expert_gb_per_layer(entry: ModelEntry) -> float:
    """GiB of EXPERT weight in one layer (the `*_exps` tensors), 0 if not MoE.

    Measured from the GGUF, because the ratio is the whole point: this model is
    20.2 GiB of which 18.6 (92%) is experts and only 1.6 is attention/norms.
    """
    try:
        from gb_gguf_tensor_map import tensor_map
        tm = tensor_map(entry.path)
    except Exception:
        return 0.0
    expert_bytes = sum(n for name, n in tm if "exps" in name)
    if not expert_bytes:
        return 0.0
    layers = {name.split(".")[1] for name, _ in tm
              if name.startswith("blk.") and "exps" in name}
    return (expert_bytes / (1024 ** 3)) / max(1, len(layers))


def _fit_cpu_moe_layers(entry: ModelEntry, budget_gb: float, kv_gb: float,
                        n_devices: int = 1) -> int:
    """How many layers must keep their EXPERTS off-GPU (llama.cpp --n-cpu-moe).

    For an MoE that doesn't fit VRAM this beats dropping whole layers, and it
    isn't close. A dropped layer takes its ATTENTION to the CPU — and attention
    is read for every token. Experts are the opposite: they are ~92% of the
    bytes but only 8 of 256 fire per token, so they are the cheapest thing in
    the model to keep out of VRAM. Move experts, keep every layer's attention
    and KV on the GPU.
    """
    per_layer = _moe_expert_gb_per_layer(entry)
    n_layers = entry.n_layers or 0
    if per_layer <= 0 or n_layers <= 0:
        return 0
    weights_gb = entry.n_bytes / (1024 ** 3)
    n_dev = max(1, n_devices)
    # Per-device %-derived reserve; budget/n_dev is the per-device VRAM proxy
    # (callers pass aggregate free VRAM — rule: no flat per-device literal).
    room = budget_gb - kv_gb - _compute_reserve_gb(budget_gb * 1024.0 / n_dev) * n_dev
    deficit = weights_gb - room * GPU_FIT_MARGIN
    if deficit <= 0:
        return 0
    return max(0, min(n_layers, math.ceil(deficit / per_layer)))


def _fit_gpu_layers(entry: ModelEntry, budget_gb: float, kv_gb: float,
                    n_devices: int = 1, t2_gb: float = 0.0) -> int:
    """How many layers actually fit the cluster's free VRAM.

    llama.cpp puts EVERY layer on the GPU unless told otherwise, and
    --tensor-split then spreads them by ratio — so a model larger than the
    aggregate VRAM doesn't degrade, it dies mid-load ("failed to allocate
    RPC0 buffer of size 12676098304" against a feeder with 7.5 GB free).
    Sizing the layer count to the real budget is what turns "won't load" into
    "loads, with the tail on the CPU": every layer we can keep on a GPU is one
    that isn't paying the CPU penalty.

    t2_gb: extra headroom from the shim's T2 pool — opt-in (0.0 default),
    only meaningful when the caller already confirmed the shim is active
    (see gb_synapse_backends.py's T2 KV extension gate; _solve_ctx_and_layers
    threads its own t2_gb parameter through to here for the same reason:
    weight-layer overflow spills to T2 exactly like KV overflow does, so
    -ngl sizing that ignores T2 while ctx sizing counts it under-uses GPU
    layers whenever the shim is genuinely extending this box's VRAM).
    Added to the AVAILABLE budget only, never to the reserve calculation —
    compute-graph workspace is sized off real per-device VRAM regardless of
    what a DDR-backed pool can absorb.
    """
    n_layers = entry.n_layers or 0
    if n_layers <= 0:
        return 999
    per_layer_gb = (entry.n_bytes / (1024 ** 3)) / n_layers
    # Every device — host AND each RPC feeder — needs its own compute/graph
    # workspace on top of the weights it holds. Counting that once (or not at
    # all) overshoots the smallest GPU, and on a GPU an overshoot is a hard OOM,
    # not a spill: the feeder was handed a 9.5 GB share of a 7.5 GB card.
    # %-derived per device, budget/n_dev as the per-device VRAM proxy (rule).
    n_dev = max(1, n_devices)
    usable = (budget_gb - kv_gb - _compute_reserve_gb(budget_gb * 1024.0 / n_dev) * n_dev
              + t2_gb)
    if usable <= 0:
        return 0
    # An MoE's layers are not uniform: the expert-carrying ones are far larger
    # than the mean, so sizing by the average over-commits the tail and the
    # smallest device OOMs (the feeder was asked for 0.7 GB/layer against a
    # 0.49 GB mean). Keep a margin rather than model every tensor: a layer left
    # on the CPU costs throughput, a failed load costs everything.
    return max(0, min(n_layers, int(usable * GPU_FIT_MARGIN / per_layer_gb)))


def _pick_kv_type(ctx: int, entry: ModelEntry, budget_gb: float) -> str:
    """f16 KV when it fits the live budget, q8_0 when it is the only way to
    reach the context — the certification-grade / budget-grade split.

    GB_SYNAPSE_KV pins it (f16 | q8_0) for a run that must be comparable to a
    published certificate, where the KV tier is part of the claim.
    """
    pin = os.environ.get("GB_SYNAPSE_KV")
    if pin:
        return pin
    weights_gb = entry.n_bytes / (1024 ** 3)
    kv_layers = 0 if entry.is_recurrent_only else (entry.n_kv_layers or entry.n_layers)
    kv_f16_gb = estimate_kv_gb(ctx, entry.n_bytes, entry.quant,
                               n_layers=kv_layers,
                               n_kv_heads=entry.n_kv_heads,
                               head_dim=entry.head_dim,
                               kv_bytes_per_elem=2.0)     # f16: 2 bytes/elem
    return "f16" if weights_gb + kv_f16_gb <= budget_gb else "q8_0"


def _find_mmproj(entry: ModelEntry) -> "Path | None":
    """The multimodal projector of a vision GGUF is a SEPARATE file, and
    llama-server serves the model text-only without it — images are dropped in
    silence and the model answers about a picture it never received. Look for
    it beside the weights (how both HF repos and our own pulls lay it out).
    GB_SYNAPSE_MMPROJ overrides for a projector kept somewhere else."""
    override = os.environ.get("GB_SYNAPSE_MMPROJ")
    if override:
        return Path(override) if Path(override).is_file() else None

    # Ollama keeps the projector as its own sha256- blob, named nothing like
    # "mmproj" — only its manifest knows which layer it is.
    if entry.source == "ollama":
        proj = ollama_model_projector(entry.name)
        if proj:
            return Path(proj)

    d = Path(entry.path).parent
    for pat in ("mmproj*.gguf", "*mmproj*.gguf"):
        for p in sorted(d.glob(pat)):
            return p
    return None


def _has_mtp(entry: ModelEntry) -> bool:
    """True when the GGUF carries a multi-token-prediction layer, which
    llama.cpp can use as its own draft model (--spec-type draft-mtp) for ~34%
    faster decode at IDENTICAL output — the draft head is the model's own
    grafted layer, so unlike a separate draft model it costs no VRAM and no
    quality. Ask the tensors, not the filename: a repo may name the file
    anything, and serving MTP flags to a model without the layer fails the
    load."""
    if os.environ.get("GB_SYNAPSE_MTP") in ("0", "1"):
        return os.environ["GB_SYNAPSE_MTP"] == "1"
    try:
        GGUFReader = _load_gguf_reader()
        names = (t.name.lower() for t in GGUFReader(entry.path, mode="r").tensors)
        return any("mtp" in n or "nextn" in n for n in names)
    except Exception:
        return False


def _clamp_ctx_to_budget(requested_ctx: int, entry: ModelEntry, budget_gb: float) -> int:
    """Shrink ctx so weights+KV fit the live aggregate VRAM budget (same fit
    math as recommend()/FitReport, solved for ctx instead of just reported).
    llama-server has no such guard: some GGUFs advertise a training context
    in the hundreds of thousands, and requesting that much KV cache OOMs
    before generation ever starts."""
    weights_gb = entry.n_bytes / (1024 ** 3)
    kv_layers = 0 if entry.is_recurrent_only else (entry.n_kv_layers or entry.n_layers)
    # Recurrent state is fixed-size (constant in ctx, unlike KV) — subtract it
    # alongside weights, once, never scaled by requested_ctx.
    kv_budget_gb = max(budget_gb - weights_gb - _entry_ssm_gb(entry), 0.25)
    kv_gb = estimate_kv_gb(requested_ctx, entry.n_bytes, entry.quant,
                           n_layers=kv_layers,
                           n_kv_heads=entry.n_kv_heads,
                           head_dim=entry.head_dim)
    if kv_gb <= kv_budget_gb:
        return requested_ctx
    safe_ctx = max(2048, int(requested_ctx * kv_budget_gb / kv_gb) // 256 * 256)
    print(f"  [gb-synapse] {entry.name}: ctx={requested_ctx} would need {kv_gb:.2f} GB KV "
          f"cache on top of {weights_gb:.2f} GB weights ({budget_gb:.2f} GB budget) — "
          f"clamping to ctx={safe_ctx}", flush=True)
    return safe_ctx


# GB-CLI's own baseline overhead (system prompt + tool schemas) — see
# CLAUDE.md's Reference Workload Rule. Below this, a real `gb -p "..."` turn
# is rejected outright before inference starts, not just slow.
CTX_FLOOR_TOKENS = int(os.environ.get("GB_SYNAPSE_CTX_FLOOR", "16384"))


def _solve_ctx_and_layers(entry: ModelEntry, vram_gb: float, requested_ctx: int,
                          n_devices: int = 1, t2_gb: float = 0.0,
                          kv_bytes_per_elem: float = 1.0) -> "tuple[int, int]":
    """Jointly solve --ctx-size and -ngl against the REAL VRAM budget, for the
    partial-CPU-offload case _clamp_ctx_to_budget gets wrong.

    _clamp_ctx_to_budget() subtracts the model's FULL weight size from the
    budget even when only a fraction of layers actually land in VRAM (the
    rest are on the CPU) — so a 14 GB model against a 9.6 GB budget always
    goes negative, kv_budget_gb collapses to its 0.25 GiB floor, and ctx gets
    clamped to a few thousand tokens regardless of how many layers really
    fit. Confirmed live 2026-07-28: qwen3.6-27b, weights=14.09GB,
    budget=9.63GB, kv_budget floored to 0.25GB -> ctx=7680, while only 33/65
    layers (~7.2GB) were actually on the GPU, leaving ~2.4GB truly free.

    -ngl is fixed FIRST (against a near-zero KV assumption — the "how many
    layers fit at all" question, independent of how large a ctx is asked
    for), then ctx is solved from whatever VRAM that leaves over. This is
    deliberately a ONE-SHOT computation, not a fixed-point loop that lets
    ctx and -ngl chase each other: an earlier version of this function DID
    iterate (recompute -ngl from the newly-solved ctx's KV cost, then
    re-solve ctx from the new -ngl, repeat) — measured against this exact
    model it does not converge to the "keep ngl, grow ctx" outcome the
    owner asked for, it OSCILLATES: each pass trades a few more layers off
    the GPU for a bigger ctx, which demands more KV, which trades away more
    layers, walking all the way down to a degenerate ngl=7/65,
    ctx=212992 corner instead of the intended ngl=33ish/ctx=46080ish
    balance. Fixing -ngl once and solving ctx as the leftover-budget
    quantity is what "no speed loss" actually means: GPU layers are the
    primary speed lever (Rule #1), ctx only grows into whatever's left over,
    it never bids layers away. The ONLY place -ngl is deliberately traded
    down for ctx is the floor-rescue block below, and only until the floor
    is cleared — never as an ongoing optimization target.

    t2_gb: extra KV budget when the shim's T2 pool is being used to extend
    ctx past the VRAM-only ceiling (opt-in — see gb_synapse_backends.py's
    T2 KV extension gate). 0 = VRAM-only ceiling, always safe.
    """
    n_layers = entry.n_layers or 0
    if n_layers <= 0 or requested_ctx <= 0:
        return requested_ctx, 999
    # is_recurrent_only: n_kv_layers==0 is the CORRECT count (no real
    # attention layer at all, arXiv:2312.00752's whole thesis) — `or
    # n_layers` must not paper over that with "unknown, assume attention".
    n_kv_layers = 0 if entry.is_recurrent_only else (entry.n_kv_layers or n_layers)
    per_layer_gb = (entry.n_bytes / (1024 ** 3)) / n_layers
    bytes_per_tok = 2 * n_kv_layers * (entry.n_kv_heads or 0) * (entry.head_dim or 0) * kv_bytes_per_elem
    # Recurrent state is FIXED-size, never scaled by ctx (the paper's O(1)
    # claim) — computed once, subtracted alongside weights/reserve wherever
    # the KV budget is, added to _fit_gpu_layers' kv_gb wherever ngl is fit.
    ssm_gb = _entry_ssm_gb(entry)
    if bytes_per_tok <= 0:
        if entry.is_recurrent_only:
            # No per-token KV cost at all — ctx is unconstrained by cache
            # growth (only the fixed ssm_gb competes with weights for VRAM).
            # Fit layers against budget minus that fixed cost and serve the
            # requested ctx as-is; nothing here trades ctx for layers.
            ngl = _fit_gpu_layers(entry, vram_gb, ssm_gb, n_devices, t2_gb=t2_gb)
            return requested_ctx, ngl
        # Geometry unknown (old manifest entry) — degrade to the previous
        # behavior rather than divide by zero.
        ctx = _clamp_ctx_to_budget(requested_ctx, entry, vram_gb + t2_gb)
        return ctx, _fit_gpu_layers(entry, vram_gb, ssm_gb, n_devices, t2_gb=t2_gb)

    n_dev = max(1, n_devices)
    ngl = _fit_gpu_layers(entry, vram_gb, ssm_gb, n_devices, t2_gb=t2_gb)
    reserve_gb = _compute_reserve_gb(vram_gb * 1024.0 / n_dev) * n_dev
    kv_budget_gb = max(vram_gb - per_layer_gb * ngl - reserve_gb - ssm_gb, 0.0) + t2_gb
    max_ctx = max(0, int((kv_budget_gb * (1024 ** 3)) / bytes_per_tok) // 256 * 256)
    ctx = max(2048, min(requested_ctx, max_ctx)) if max_ctx > 0 else 2048
    if ctx < requested_ctx:
        print(f"  [gb-synapse] {entry.name}: ctx={requested_ctx} solved down to ctx={ctx} "
              f"jointly with -ngl {ngl}/{n_layers} against a {vram_gb:.2f} GB VRAM budget"
              f"{f' + {t2_gb:.2f} GB T2' if t2_gb else ''}"
              f"{f' (minus {ssm_gb:.2f} GB recurrent state)' if ssm_gb else ''}.", flush=True)

    # A window below GB-CLI's own baseline overhead (~4608 tokens for the
    # system prompt + tool schemas) can't hold one real agentic turn — a
    # silently-served tiny window is exactly the failure the owner hit
    # (7680, barely above that floor, from an unrelated bug). Trade GPU
    # layers for KV room, loudly, rather than ship a ctx too small to use.
    if 0 < ctx < CTX_FLOOR_TOKENS and ngl > 0:
        traded = 0
        while ctx < CTX_FLOOR_TOKENS and ngl > 0:
            ngl -= 1
            traded += 1
            reserve_gb = _compute_reserve_gb(vram_gb * 1024.0 / n_dev) * n_dev
            kv_budget_gb = max(vram_gb - per_layer_gb * ngl - reserve_gb - ssm_gb, 0.0) + t2_gb
            ctx = max(0, int((kv_budget_gb * (1024 ** 3)) / bytes_per_tok) // 256 * 256)
            ctx = min(requested_ctx, ctx)
        ctx = max(2048, ctx)
        print(f"  [gb-synapse] {entry.name}: ctx={ctx} was still below the "
              f"{CTX_FLOOR_TOKENS:,}-token floor (an agentic turn's own baseline "
              f"overhead) — traded {traded} GPU layer(s) for KV room, now "
              f"-ngl {ngl}/{n_layers}. Add VRAM (a feeder) or a smaller quant to "
              f"recover those layers.", flush=True)
    return ctx, ngl


def _compute_tensor_split(host_free_mb: int, online_feeders: list,
                          kv_total_gb: float = 0.0) -> str:
    """--tensor-split ratios from REAL free VRAM per device — not the shim's
    inflated cuDeviceTotalMem — so llama.cpp's own placement isn't distorted
    by GreenBoost's aggregation math. The shim absorbs only the per-device
    overflow that's left after this explicit split, into that node's local
    DDR/NVMe.

    v1 (default): shares == free VRAM per device.
    v2 (GB_SYNAPSE_SPLIT_V2=1): subtract each node's proportional KV share from
    its free VRAM to get a WEIGHT budget (KV scales with a node's layer share,
    which scales with the split — one fixed-point iteration), then apply
    GB_SYNAPSE_HOST_BIAS (default 1.0 = identity) to the host share. The host
    also runs the output head + sampling, so a mild >1.0 bias can be worth it
    on a slow (GbE) link. Default path is byte-identical to v1.
    """
    # Every device also needs a compute/graph workspace, so the WEIGHTS a device
    # can hold are its free VRAM minus that reserve. Splitting on raw free VRAM
    # over-serves the smallest card — a 7.5 GB feeder was handed an 8.4 GB share
    # and the load died there, not on the host that had room to spare.
    # Reserve is %-derived PER DEVICE (rule): feeders report t1_total_mb; the
    # host's free VRAM stands in for its total here (conservative, never over).
    def _feeder_vram_mb(f) -> int:
        """A feeder's free-VRAM figure for the split. Prefers live t1_free_mb;
        when that's 0 (older mem_info reply that carried no per-tier bytes),
        fall back to the node's topology vram_gb so it still gets a real share
        instead of being handed a share=1 sliver."""
        if f.t1_free_mb > 0:
            return f.t1_free_mb
        topo = getattr(f, "topology", None) or {}
        vram_gb = topo.get("vram_gb", 0)
        return int(vram_gb) * 1024 if vram_gb else 0

    free = [max(int(host_free_mb - _compute_reserve_gb(host_free_mb) * 1024.0), 1)] + \
           [max(int(_feeder_vram_mb(f)
                    - _compute_reserve_gb(getattr(f, "t1_total_mb", 0)
                                          or _feeder_vram_mb(f)) * 1024.0), 1)
            for f in online_feeders]
    v2 = os.environ.get("GB_SYNAPSE_SPLIT_V2", "") == "1"
    try:
        host_bias = float(os.environ.get("GB_SYNAPSE_HOST_BIAS", "1.0"))
    except ValueError:
        host_bias = 1.0

    if not v2 and host_bias == 1.0:
        shares = free                                   # exact v1 behaviour
    else:
        total_free = sum(free) or 1
        kv_mb = max(0.0, kv_total_gb) * 1024.0
        # Proportional KV per node, then weight budget = free - KV share.
        shares = [max(f - kv_mb * (f / total_free), 1.0) for f in free]
        shares[0] *= host_bias
        shares = [max(int(round(s)), 1) for s in shares]

    # v3 (GB_SYNAPSE_SPLIT_V3=1, petals min-throughput port — see
    # workflow/porting-reference.md §DI-3): scale each FEEDER's share down by
    # its measured link quality relative to this host's own detected uplink
    # (gb_topology.net_link_mbps — never a hardcoded reference bandwidth).
    # A feeder with no measured link yet (link_mbps_ewma==0, e.g. first ever
    # dispatch) or when the host's own link is undetectable gets factor=1.0
    # (no penalty — falls back to v1/v2 behaviour rather than guessing).
    # The host itself never gets a network penalty (it has no RPC hop).
    v3 = os.environ.get("GB_SYNAPSE_SPLIT_V3", "") == "1"
    if v3 and online_feeders:
        try:
            from gb_topology import get_topology
            ref_mbps = get_topology().net_link_mbps or 0
        except Exception:
            ref_mbps = 0
        factors = [1.0]
        for f in online_feeders:
            link = getattr(f, "link_mbps_ewma", 0.0) or 0.0
            factors.append(min(1.0, link / ref_mbps) if (ref_mbps > 0 and link > 0) else 1.0)
        shares = [max(int(round(s * fac)), 1) for s, fac in zip(shares, factors)]

    split = ",".join(str(s) for s in shares)
    try:
        import gb_dataflux
        gb_dataflux.emit({"kind": "tensor_split", "split": split, "v2": v2, "v3": v3,
                          "host_bias": host_bias, "kv_total_gb": round(kv_total_gb, 2),
                          "nodes": len(shares)})
    except Exception:
        pass
    return split


def _upstream_log_tail(model: str, n: int = 8) -> str:
    """The lines the engine wrote just before it died. A client can only ever
    observe a closed connection; the actual reason (unsupported hyperparameter,
    missing blob, OOM) exists nowhere but this log, so every failure we raise
    carries it."""
    try:
        lines = _run_log_path(model).read_text(errors="replace").splitlines()
    except OSError:
        return ""
    errs = [ln for ln in lines if re.search(r"\berror\b|\bfailed\b|\bE\b", ln)]
    return "\n".join("      " + ln.strip() for ln in (errs or lines)[-n:])


def emit_stall(model: str, engine: str, elapsed_s: float) -> None:
    """A request has been streaming for `elapsed_s` with zero tokens
    produced — the gap documented in workflow/gb-synapse.md's torch-core
    bring-up notes: `/health` returns 200 (the engine process is alive and
    answering), but the FIRST real request hung forever hitting a CUDA
    error during prefill, so the client just sees a silent hang, not an
    error. Called from gb_synapse_api.py's proxy subprocess (same shape as
    record_measured_tok_s: a proxy-side signal handed back into gb_synapse's
    own dataflux emit) by a per-request watchdog task, not by the streaming
    loop itself — the loop is exactly what's stuck, so nothing there can
    notice the stall on its own. Best-effort, never raises."""
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "kind": "synapse_stall", "status": "warn",
            "model": model, "engine": engine, "elapsed_s": round(elapsed_s, 1),
            "log_tail": _upstream_log_tail(model),
        })
    except Exception:
        pass


def _wait_upstream_ready(entry: ModelEntry, proc: subprocess.Popen, internal_port: int,
                          grace_s: float = SERVE_READY_GRACE_S) -> bool:
    """Watch a freshly spawned engine until it serves, dies, or `grace_s` runs out.

    True  = /health is green, the model is loaded.
    False = still loading (a 20+ GB GGUF legitimately takes minutes — the caller
            reports that rather than blocking the user's terminal on it).
    Raises = the process is gone, i.e. it never had a chance of serving.

    That last case is the one worth catching here: without this gate serve()
    returned a ServerState for a corpse, callers printed "✓ started (pid N)",
    and the truth only surfaced later as an unreadable mid-stream
    RemoteProtocolError in whatever client had believed them.

    One more corpse this also has to catch: llama.cpp's own pre-flight
    admission check (`common_fit_params`) can log "failed to fit params to
    free device memory ... abort" and then NOT actually exit the process —
    confirmed live 2026-07-27 against an RPC feeder split, the process sat
    alive (proc.poll() stayed None) for the full grace period and beyond,
    never opening the HTTP port, until an external teardown finally killed
    it. Without this, that failure is a silent multi-minute hang, not a
    caught error the retry-and-back-off logic in gb_synapse_backends.py can
    act on. Kill it and raise the same way a real exit would, so callers see
    one failure shape either way.

    This scan reads only the bytes appended to the log SINCE this call
    started, not `_upstream_log_tail`'s file-tail — the log path is the same
    file across every retry attempt (opened "ab" each time), so a plain tail
    scan finds a STALE match from a PREVIOUS attempt's failure and kills a
    CURRENT attempt that's actually loading fine. Confirmed live 2026-07-27:
    a retry's tail scan matched a `common_fit_params` line from an attempt
    two retries back, well before the current process had written anything.
    """
    import urllib.error
    import urllib.request

    log_path = _run_log_path(entry.name)
    try:
        start_offset = log_path.stat().st_size
    except OSError:
        start_offset = 0

    def _new_log_output() -> str:
        try:
            with open(log_path, "rb") as f:
                f.seek(start_offset)
                return f.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    deadline = time.time() + grace_s
    url = f"http://127.0.0.1:{internal_port}/health"
    while time.time() < deadline:
        # poll() is the authoritative answer for a child WE spawned, and it
        # reaps — unlike a PID check, which a zombie passes.
        if proc.poll() is not None:
            raise RuntimeError(
                f"'{entry.name}' failed to load: the engine exited during startup "
                f"(rc={proc.returncode}).\n"
                f"    log: {log_path}\n{_upstream_log_tail(entry.name)}")
        if "failed to fit params to free device memory" in _new_log_output().lower():
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass  # kill() was sent either way; not worth blocking the failure report on it
            raise RuntimeError(
                f"'{entry.name}' failed to load: llama.cpp's pre-flight fit-check "
                f"rejected the requested layer split and hung instead of exiting "
                f"(killed after detection).\n"
                f"    log: {log_path}\n{_upstream_log_tail(entry.name)}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200 and not _health_says_loading(r.read(256)):
                    return True
        except urllib.error.HTTPError:
            pass          # 503 "loading model" — alive and working
        except OSError:
            pass          # not listening yet
        time.sleep(0.5)
    return False


def _wait_proxy_ready(proxy_proc: subprocess.Popen, port: int, label: str,
                      grace_s: float = 5.0) -> str | None:
    """Watch a freshly spawned gb_synapse_api.py proxy until it serves, dies,
    or `grace_s` runs out. Short grace vs the engine's own
    (SERVE_READY_GRACE_S): proxy startup is near-instant when it succeeds: it
    is a thin aiohttp app with no model to load, so the only real failure
    mode observed is an immediate bind error.

    Real incident, 2026-07-16: without this check, `_launch_proxy_and_record`
    fired the proxy with `subprocess.Popen` and immediately reported success
    — a caller (MCP tool, CLI) saw a clean ServerState even when the proxy
    had already crashed (`OSError: [Errno 98] address already in use`,
    raw ollama.service still bound to :11434, the recurring real case on a
    box mid-migration off it). The engine WAS genuinely serving on
    internal_port, so the failure was invisible until a client request came
    back as a confusing "model not found" from whatever else answered :port
    instead — three layers removed from the actual cause.

    Returns None when healthy, else a short reason string (never raises —
    the caller decides whether a dead proxy is fatal)."""
    import urllib.error
    import urllib.request

    deadline = time.time() + grace_s
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proxy_proc.poll() is not None:
            return (f"proxy exited during startup (rc={proxy_proc.returncode})\n"
                    f"{_upstream_log_tail(label)}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                # ONLY a genuine 200 counts. Real bug, found live 2026-07-16:
                # an earlier version of this check treated ANY HTTP response
                # (including urllib.error.HTTPError, i.e. any non-2xx status)
                # as proof the proxy was up — but when raw ollama.service
                # already holds `port`, ITS OWN generic "404 page not found"
                # response satisfies that check too. The real proxy had
                # already crashed (confirmed: its PID no longer existed) the
                # whole time this returned a false "healthy". A 404 from
                # something else on the same port is exactly what this check
                # exists to catch, not a pass condition.
                if r.status == 200 and proxy_proc.poll() is None:
                    return None
        except (urllib.error.HTTPError, OSError):
            pass              # not our proxy answering, or not listening yet
        time.sleep(0.3)
    if proxy_proc.poll() is not None:
        return (f"proxy exited during startup (rc={proxy_proc.returncode})\n"
                f"{_upstream_log_tail(label)}")
    hint = ("raw ollama.service may own :11434 — gb-synapse now defaults to "
             f":{DEFAULT_PORT}; set GB_SYNAPSE_PORT to change" if port == 11434 else
             f"something else may already be listening on :{port}")
    return f"proxy never answered /health with 200 within {grace_s}s — {hint}"


def _health_says_loading(body: bytes) -> bool:
    """llama.cpp is not consistent about how /health reports a load in progress:
    newer builds answer 503, but the engine we ship answers 200 with
    {"status": "loading model"}. Trusting the status code alone marks a model
    "served" seconds before it has parsed its own hyperparameters — the exact
    window in which a bad GGUF exits. The torch engine/gb_synapse_fallback
    answer 200 with no status field, which is genuinely ready."""
    try:
        status = json.loads(body or b"{}").get("status", "ok")
    except (ValueError, AttributeError):
        return False
    return status != "ok"


def failure_report(model: str) -> str:
    """Why a served model stopped answering, in words a client can print.

    A client (greenboost-cli, any OpenAI SDK) only ever sees a truncated body:
    the proxy relays /v1 byte-for-byte, so when the engine dies mid-stream all
    that reaches it is "peer closed connection". This looks up what actually
    happened on this side — engine still alive? gone? what did it log last? —
    so that failure can be reported instead of an httpx exception name.
    """
    st = next((s for s in _read_run_states() if s.model == model), None)
    if st is None:
        return "gb-synapse has no running server for this model (start it with /llamaserve)."
    if _pid_alive(st.llama_pid):
        return ("the engine is alive but closed the connection mid-response — most often it is "
                "still loading the model, or the request exceeded its context window.\n"
                f"    log: {_run_log_path(model)}")
    tail = _upstream_log_tail(model)
    return (f"the engine (pid {st.llama_pid}) is gone — it died while serving this request.\n"
            f"    log: {_run_log_path(model)}" + (f"\n{tail}" if tail else ""))


def _emit_serve(entry: ModelEntry, status: str, **fields) -> None:
    """Every serve attempt lands in dataflux — success and failure alike. A
    model that won't load is an orchestration fact (it decides whether the
    cluster path is available at all), so it belongs in the flux next to the
    tier moves and quantization decisions that assume a served model."""
    try:
        import gb_dataflux
        gb_dataflux.emit({"kind": "synapse_serve", "status": status,
                          "model": entry.name, "arch": entry.arch or "",
                          "quant": entry.quant or "",
                          "weights_gb": round(entry.n_bytes / (1024 ** 3), 2),
                          **fields})
    except Exception:
        pass


def _start_proxy(entry: ModelEntry, port: int, internal_port: int,
                  engine: str) -> "tuple[subprocess.Popen, str | None]":
    """Launch gb_synapse_api.py in front of an already-running upstream
    engine and wait for it to bind. Factored out of _launch_proxy_and_record
    so serve()'s proxy-only restart path (engine alive, proxy dead) can
    reuse the exact same launch+readiness logic without re-running engine
    startup — see serve()'s docstring for why that distinction matters."""
    proxy_label = entry.name + "_proxy"
    proxy_cmd = [sys.executable, str(_REPO_DIR / "gb_synapse_api.py"),
                 "--port", str(port), "--upstream-port", str(internal_port),
                 "--model-name", entry.name, "--engine", engine]
    proxy_log = open(_run_log_path(proxy_label), "ab")
    proxy_proc = subprocess.Popen(proxy_cmd, stdout=proxy_log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    proxy_error = _wait_proxy_ready(proxy_proc, port, proxy_label)
    return proxy_proc, proxy_error


_CACHE_REUSE_REJECT = "cache_reuse is not supported by this context, it will be disabled"


def _check_cache_reuse_support(entry: ModelEntry, common: dict) -> None:
    """--cache-reuse 256 is passed to every llama.cpp serve (see
    LlamaCppBackend.serve()) to cut prompt-eval cost on GB-CLI's repeated
    system-prompt+tool-schema prefix, but llama.cpp silently refuses it for
    architectures where kv_unified=false (hybrid-recurrent models, e.g. this
    repo's Qwen3.6-27B-Fable-Fusion reference — see CLAUDE.md's Reference
    Workload Rule) and just logs one warning line, no error, no field on
    anything queryable. Real cost this hides (2026-08-01, this exact model):
    every agentic turn re-pays FULL prompt eval — 279s at ~12.8k tokens in
    the incident that prompted this fix. Best-effort: a missing/short log
    just means "can't tell", not "supported"."""
    try:
        log_path = _run_log_path(entry.name)
        if not log_path.exists():
            return
        text = log_path.read_text(errors="replace")
        rejected = _CACHE_REUSE_REJECT in text
        if rejected:
            print(f"  [gb-synapse] NOTE: {entry.name} rejected --cache-reuse "
                  f"(\"{_CACHE_REUSE_REJECT}\") — this architecture can't "
                  f"reuse a cached prompt prefix, so every turn re-pays full "
                  f"prompt-eval cost. Not a bug in this serve, just a fact "
                  f"worth knowing before blaming decode speed for a slow turn.",
                  flush=True)
            common["cache_reuse_rejected"] = True
    except Exception:
        pass


def _verify_placement(entry: ModelEntry, common: dict) -> None:
    """Rule #1 sanity check, run once right after a single-node (no feeder)
    engine reports ready: does tracked VRAM + shim overflow actually add up
    to this model's real weight size? Best-effort, never raises — a missing
    or stale shim_stats file just skips the check silently.

    Real incident this closes (2026-08-01): a cc>=12 (Blackwell) serve spilled
    weights correctly through gb_vmm_t2_alloc_blackwell_zerocopy(), but that
    path's byte total wasn't written to shim_stats at all (t2_overflow_total_mb
    fixed the same day in greenboost_cuda_shim.c), so there was previously no
    way for Python to tell "spilled via a path we don't track" apart from
    "actually missing" — this check would have reported a false Rule #1
    violation before that fix and stayed silent (wrongly) after it without
    this call existing at all. Only meaningful for llama.cpp / whole-model-
    in-one-process engines; skipped by the caller whenever feeders are in
    play (RPC split means no single node's shim_stats reflects the total)."""
    try:
        import gb_dataflux
        import gb_monitor
        snap = gb_monitor.snapshot(probe_gpu=False)
        if not snap.loaded or snap.shim_stale:
            return
        shim = snap.shim or {}
        t1_mb = int(float(shim.get("tier_t1_local_cur_mb", 0)))
        t2_overflow_mb = int(float(shim.get("t2_overflow_total_mb", 0)))
        tracked_gb = (t1_mb + t2_overflow_mb) / 1024.0
        weights_gb = entry.n_bytes / (1024 ** 3)
        # 15% slack for CUDA context overhead, alignment padding, and the
        # workspace/compute-buffer reserve that legitimately isn't "weights".
        if tracked_gb < weights_gb * 0.85:
            gb_dataflux.emit({
                "node": "host", "label": "shim", "kind": "shim_transition",
                "stage": "placement_verify", "from": "expected", "to": "underfilled",
                "n_items": 0, "items": [], "duration_s": 0.0, "status": "warn",
                "weights_gb": round(weights_gb, 2), "tracked_gb": round(tracked_gb, 2),
                "t1_mb": t1_mb, "t2_overflow_mb": t2_overflow_mb,
                "model": entry.name,
            })
            print(f"  [gb-synapse] WARNING: only {tracked_gb:.1f} GB of "
                  f"{weights_gb:.1f} GB of weights are accounted for in "
                  f"shim telemetry (T1={t1_mb}MB + T2-overflow={t2_overflow_mb}MB) — "
                  f"placement may be worse than reported, or shim_stats is "
                  f"missing a tracking path. See dataflux shim_transition events.",
                  flush=True)
    except Exception:
        pass


def _launch_proxy_and_record(entry: ModelEntry, upstream: subprocess.Popen, port: int,
                              internal_port: int, tensor_split: str = "",
                              feeders: list | None = None, engine: str = "",
                              **serve_facts) -> ServerState:
    """Shared tail for every engine (llama.cpp/torch/transformers/diffusers):
    wait for the engine to come up, start the gb_synapse_api.py proxy in
    front of it, and record ServerState. The proxy only ever talks OpenAI
    /v1/* to the upstream, so it's identical regardless of which engine is
    behind it — and all four expose /health, so the readiness gate is too.

    `engine` is the backend's own name (EngineBackend.name); every caller is
    a backend's serve() method, which always passes it — falls back to
    entry.engine only for callers that don't (there are none left, kept as a
    defensive default rather than a required param so a future backend that
    forgets it still gets a sane value instead of an empty string)."""
    engine = engine or entry.engine
    t0 = time.time()
    common = {"port": port, "tensor_split": tensor_split, "feeders": feeders or [],
              "engine": engine, **serve_facts}
    try:
        ready = _wait_upstream_ready(entry, upstream, internal_port)
    except RuntimeError as e:
        # No proxy is started for a dead engine: a proxy in front of nothing
        # is what turns "the model failed to load" into "connection closed
        # mid-stream" three layers away.
        _emit_serve(entry, "error", error=str(e).splitlines()[0],
                    load_s=round(time.time() - t0, 1), **common)
        raise

    if ready:
        _check_cache_reuse_support(entry, common)
        if not (feeders or []):
            _verify_placement(entry, common)

    proxy_proc, proxy_error = _start_proxy(entry, port, internal_port, engine)

    state = ServerState(model=entry.name, llama_pid=upstream.pid, proxy_pid=proxy_proc.pid,
                         port=port, internal_port=internal_port, tensor_split=tensor_split,
                         feeders=feeders or [], started_ts=time.time(), engine=engine,
                         ready=ready, proxy_error=proxy_error,
                         ctx=int(serve_facts.get("ctx") or 0),
                         kv_type=str(serve_facts.get("kv_type") or ""),
                         n_gpu_layers=int(serve_facts.get("n_gpu_layers") or 0))
    _write_run_state(state)
    if proxy_error:
        # The engine loaded a real model into memory (upstream.pid is alive);
        # tearing it down here would waste a load that may have taken minutes
        # just because the FRONT DOOR failed. Record the state (so ps()/
        # failure_report() see a genuinely running engine behind a dead
        # proxy) and surface the failure distinctly rather than either
        # silently reporting full success or discarding real work.
        _emit_serve(entry, "proxy_error", error=proxy_error.splitlines()[0],
                    load_s=round(time.time() - t0, 1), **common)
    else:
        _emit_serve(entry, "ok" if ready else "loading",
                    load_s=round(time.time() - t0, 1), **common)
    return state


def serve(model: str, port: int = DEFAULT_PORT, ctx: int = 0,
          use_cluster: bool = True, n_slots: int = -1, extra_args: str = "",
          cuda_graph: "bool | None" = None, cache_ram: "int | None" = None) -> ServerState:
    """Resolve `model` (manifest name, "org/repo[:quant]", or a bare Ollama
    model name) and hand it to whichever engine backend its manifest entry
    calls for (gb_synapse_backends.select_backend) — llama.cpp (default,
    cluster --rpc split), the synapse torch engine (gb-quant fp8/int8/int4
    safetensors), the transformers fallback, or diffusers (image gen).

    Also the single process manager for GreenBoost's own model servers:
    reuses an already-running instance for this exact model instead of
    racing a second launch (Ollama's scheduler solves this with a queue +
    refcounted runners; gb-synapse only ever runs one model at a time, so
    "return the existing one" is the equivalent guarantee).

    cuda_graph: per-call override for the synapse torch engine's CUDA-graph
    capture (ignored by other backends). None (default) falls back to the
    GB_SYNAPSE_TORCH_CUDA_GRAPH env var, off by default because graph
    capture's warmup buffers can OOM small cards (see EngineBackend.serve()'s
    docstring in gb_synapse_backends.py). Pass explicitly to try graphs for
    one serve without mutating process env — e.g. after lowering ctx enough
    to free the headroom graphs need.

    cache_ram: per-call override (MiB) for llama.cpp's host-memory prompt
    cache (--cache-ram; LlamaCppBackend only, ignored by other backends).
    None (default) derives it from live free host RAM (see
    LlamaCppBackend.serve()'s comment) — never a literal MiB figure per this
    repo's hardcoded-hardware-values rule.

    A THIRD case, not just alive/dead: the engine can be alive while only
    its proxy died (another process squatted the port, an OOM-killer got the
    proxy but not the much larger engine, ...). The old reuse-check required
    BOTH pids alive, so this case fell through to backend.serve() — which
    spawns a brand-new engine while the first one, already holding VRAM for
    a loaded model, keeps running unreferenced. Restarting just the proxy
    (same ports the engine already committed to) is the fix.
    """
    entry = _resolve_model(model)

    for st in _read_run_states():
        if st.model != entry.name:
            continue
        engine_alive = _pid_alive(st.llama_pid)
        proxy_alive = _pid_alive(st.proxy_pid)
        if engine_alive and proxy_alive:
            return _maybe_serve_embedding(st)
        if engine_alive and not proxy_alive:
            proxy_proc, proxy_error = _start_proxy(entry, st.port, st.internal_port, st.engine)
            st.proxy_pid = proxy_proc.pid
            st.proxy_error = proxy_error
            _write_run_state(st)
            _emit_serve(entry, "proxy_error" if proxy_error else "ok",
                        port=st.port, tensor_split=st.tensor_split, feeders=st.feeders,
                        engine=st.engine, load_s=0.0, restart="proxy_only")
            return _maybe_serve_embedding(st)
        # Neither pid alive — stale run-state; fall through to a fresh serve().

    backend = gb_synapse_backends.select_backend(entry)
    try:
        state = backend.serve(entry, port, ctx=ctx, use_cluster=use_cluster,
                              n_slots=n_slots, extra_args=extra_args,
                              cuda_graph=cuda_graph, cache_ram=cache_ram)
    except RuntimeError:
        # A cluster/RPC load can fail for reasons that have nothing to do with
        # whether the model can run at all — a feeder's engine build gap, a
        # network-latency-bound split, a compute-reserve estimate that's wrong
        # specifically for a multi-device config. Falling all the way through
        # to the caller here is what pushed greenboost-cli into querying a
        # DIFFERENT backend's port on failure instead of getting a working
        # gb-synapse serve (confirmed live 2026-07-27: `gb -m <model> -p ...`
        # hit an RPC-split failure, then fell back to raw Ollama's port and
        # reported "model not found" there — gb-synapse is supposed to be the
        # only backend, so failing outward like that is itself the bug).
        # Retrying host-only here is never worse than the cluster attempt
        # (strictly fewer devices, same or smaller footprint), so it's safe to
        # do automatically rather than surfacing a cluster-specific error for
        # a model that may well run fine node-local.
        if not use_cluster:
            raise
        print(f"  [gb-synapse] {entry.name}: cluster/RPC serve failed — "
              f"retrying host-only before giving up.", flush=True)
        state = backend.serve(entry, port, ctx=ctx, use_cluster=False,
                              n_slots=n_slots, extra_args=extra_args,
                              cuda_graph=cuda_graph, cache_ram=cache_ram)
    return _maybe_serve_embedding(state)


def _maybe_serve_embedding(state: ServerState) -> ServerState:
    """Eagerly bring up a second, minimal embeddings engine alongside the
    primary serve when GB_SYNAPSE_EMBED_MODEL is set — the RAG-client-usable
    half of the embeddings design (P6): a client can't call :11435/v1/
    embeddings at all without SOME engine behind it. An on-demand,
    first-request lazy launch triggered from the proxy itself is a real,
    larger feature intentionally left for later — the proxy is a separate
    process from this call, so triggering a launch from inside a request
    handler needs its own coordination (a callback into gb_synapse.py, or a
    control endpoint), not attempted here.

    Idempotent and best-effort: a no-op when unset or already running;
    logs and returns the state unchanged on any failure — an optional
    embeddings side-engine must never break the primary serve it rides
    alongside."""
    embed_model = os.environ.get("GB_SYNAPSE_EMBED_MODEL", "").strip()
    if not embed_model:
        return state
    if state.embed_internal_port and _pid_alive(state.embed_pid):
        return state  # already up
    try:
        embed_entry = _resolve_model(embed_model)
        backend = gb_synapse_backends.select_backend(embed_entry)
        if not hasattr(backend, "serve_embedding"):
            print(f"[gb-synapse] GB_SYNAPSE_EMBED_MODEL={embed_model!r}: "
                  f"{type(backend).__name__} has no embeddings support — skipping",
                  file=sys.stderr)
            return state
        proc, internal_port = backend.serve_embedding(embed_entry, state.port)
        state.embed_pid = proc.pid
        state.embed_internal_port = internal_port
        _write_run_state(state)
    except Exception as e:
        print(f"[gb-synapse] embeddings engine for {embed_model!r} failed to start: {e}",
              file=sys.stderr)
    return state


def _kill_process_group(pid: int) -> None:
    """SIGTERM the whole process group `pid` leads, not just `pid` itself.

    Every engine is launched with `start_new_session=True` (its own PID
    becomes the process-group leader), so this reaches any detached child
    the engine spawns. Real leak found live 2026-07-16: the synapse torch
    engine (gLLM) spawns a `multiprocessing.spawn_main` worker that does
    NOT receive a plain `os.kill(llama_pid, SIGTERM)` — killing only the
    tracked PID left that worker running and holding ~10 GB of VRAM
    (confirmed via nvidia-smi) after `stop()` reported success. Falls back
    to a plain single-PID kill if the process group is already gone
    (ESRCH) or `pid` was never a group leader (older run-state)."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def _teardown_feeders(model: str, st: "ServerState") -> None:
    """Best-effort cluster-side teardown for stop().

    A cluster-PP (torch engine) stop must kill each feeder's gLLM slave
    rank — it dials into a NCCL world that no longer has a master the
    moment the local process dies, and unlike ensure_feeder_rpc()'s
    rpc-server, a slave rank is bound 1:1 to THIS serve's master_ip/port/
    pp_rank (see ensure_feeder_gllm_slave's docstring) — it can never be
    reused by a later serve, so leaving it running only holds feeder VRAM
    in a world nothing will ever join again.

    For --rpc (llama.cpp), a feeder's rpc-server IS deliberately reused
    across serves (ensure_feeder_rpc() no-ops when already reachable) — so
    it is only stopped here when no OTHER currently-running gb-synapse
    serve still lists that feeder, to avoid breaking that reuse for a
    co-resident model.
    """
    if not st.feeders:
        return
    try:
        by_ip = {f.ip: f for f in gb_cluster.feeders(probe=False)}
    except Exception:
        return

    if st.engine == "torch":
        for ip in st.feeders:
            feeder = by_ip.get(ip)
            if feeder is not None:
                kill_feeder_gllm_slave(feeder)
        return

    if st.engine == "llama.cpp":
        others_using = {ip for other in _read_run_states()
                         if other.model != model and other.engine == "llama.cpp"
                         for ip in other.feeders}
        for idx, ip in enumerate(st.feeders):
            if ip in others_using:
                continue
            feeder = by_ip.get(ip)
            if feeder is None:
                continue
            rpc_port = RPC_PORT_BASE + idx
            try:
                subprocess.run(
                    ["ssh", *_SSH_OPTS, _feeder_ssh_target(feeder),
                     f"pkill -f 'rpc-server.*--port {rpc_port}' || true"],
                    capture_output=True, text=True, timeout=10)
            except Exception:
                pass


def stop(model: str) -> bool:
    for st in _read_run_states():
        if st.model != model:
            continue
        for pid in (st.llama_pid, st.proxy_pid, st.embed_pid):
            if pid and _pid_alive(pid):
                _kill_process_group(pid)
        _teardown_feeders(model, st)
        _run_state_path(model).unlink(missing_ok=True)
        return True
    return False


def ps() -> list[dict]:
    """Running gb-synapse servers. An entry is pruned only once BOTH the
    engine and the proxy have died — a live engine behind a dead proxy
    (see serve()'s proxy-only restart) is still a real, VRAM-holding
    server, and dropping it here would hide that from status()/stop()."""
    live = []
    for st in _read_run_states():
        engine_alive = _pid_alive(st.llama_pid)
        proxy_alive = _pid_alive(st.proxy_pid)
        if not engine_alive and not proxy_alive:
            _run_state_path(st.model).unlink(missing_ok=True)
            continue
        if engine_alive and not proxy_alive and not st.proxy_error:
            st.proxy_error = "proxy process is gone"
        live.append(asdict(st))
    return live


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format_doctor(d: dict) -> str:
    lines = [f"Host GPU:   {d['host_gpu_name']}  {d['host_vram_total_mb'] / 1024:.1f} GB VRAM",
             f"Host RAM:   {d['host_ram_total_mb'] / 1024:.1f} GB"]
    for f in d["feeders"]:
        if f.get("online"):
            lines.append(f"Feeder:     {f['hostname']}  {f['vram_total_mb'] / 1024:.1f} GB VRAM  "
                          f"{f['ram_total_mb'] / 1024:.1f} GB RAM")
        else:
            lines.append(f"Feeder:     {f['hostname']}  OFFLINE ({f.get('error', '')})")
    lines.append(f"Aggregate:  {d['aggregate_vram_mb'] / 1024:.1f} GB VRAM, "
                 f"{d['aggregate_ram_mb'] / 1024:.1f} GB RAM")
    lines.append("Engine:     " + (f"built ({d['engine_version']})" if d['engine_installed']
                 else "NOT BUILT — run: greenboost synapse build-engine"))
    lines.append("Torch core: " + (f"ready ({d['torch_engine_env']})" if d.get('torch_engine_ready')
                 else "missing — run: sudo greenboost install-synapse-engine"))
    lines.append("HF token:   " + ("set" if d['hf_token_set']
                 else "NOT SET — run: greenboost synapse login"))
    return "\n".join(lines)


def _cli_main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    verb, rest = argv[0], [a for a in argv[1:] if a != "--llm"]
    llm = "--llm" in argv[1:]

    if verb == "build-engine":
        info = build_engine()
        print(json.dumps(info) if llm else f"engine built: {info['version']}")
    elif verb == "update-engine":
        info = update_engine()
        print(json.dumps(info) if llm else f"engine updated: {info['version']}")
    elif verb == "login":
        token = rest[0] if rest else os.environ.get("HF_TOKEN", "")
        if not token:
            print("usage: gb_synapse.py login <TOKEN>  (or export HF_TOKEN)", file=sys.stderr)
            return 2
        login(token)
        print("HuggingFace token saved.")
    elif verb == "pull":
        if not rest:
            print("usage: gb_synapse.py pull <repo>[:quant] [name] "
                  "[--engine llama.cpp|torch|diffusers]", file=sys.stderr)
            return 2
        engine = ""
        if "--engine" in rest:
            i = rest.index("--engine")
            engine = rest[i + 1] if i + 1 < len(rest) else ""
            rest = rest[:i] + rest[i + 2:]
        entry = pull(rest[0], name=rest[1] if len(rest) > 1 else None, engine=engine)
        print(json.dumps(asdict(entry)) if llm
              else f"pulled {entry.name}  ({entry.n_bytes / (1024 ** 3):.2f} GiB, {entry.quant}, "
                   f"engine={entry.engine})")
    elif verb == "index-ollama":
        found = index_ollama_models()
        print(json.dumps([asdict(e) for e in found]) if llm
              else f"indexed {len(found)} Ollama model(s)")
    elif verb == "list":
        models = list_models()
        if llm:
            print(json.dumps([asdict(m) for m in models]))
        elif not models:
            print("  (no models yet — run: greenboost pull <org/repo>[:quant]"
                  "  or: greenboost synapse index-ollama)")
        else:
            for m in models:
                print(f"  {m.name:<32} {m.source:<7} {m.quant:<10} {m.n_bytes / (1024 ** 3):>7.2f} GiB")
    elif verb == "rm":
        if not rest:
            print("usage: gb_synapse.py rm <name>", file=sys.stderr)
            return 2
        rm(rest[0])
        print(f"removed {rest[0]}")
    elif verb == "doctor":
        d = doctor()
        print(json.dumps(d) if llm else _format_doctor(d))
    elif verb == "status":
        s = status()
        print(json.dumps(s) if llm else
              f"engine: {'built ' + (s['engine_version'] or '') if s['engine_built'] else 'NOT built'}"
              f"  server: {'running' if s['server_running'] else 'stopped'}"
              f"  proxy: {'running' if s['proxy_running'] else 'stopped'}"
              f"  ({s['engine_dir']})")
    elif verb == "recommend":
        ctx = int(rest[0]) if rest else 65536
        reports = recommend(ctx=ctx)
        if llm:
            print(json.dumps([asdict(r) for r in reports]))
        elif not reports:
            print("  (no models pulled yet — run: greenboost pull <org/repo>[:quant])")
        else:
            for r in reports:
                mark = "✓" if r.fits_vram else "✗"
                tag = "" if r.measured else "~"
                print(f"  {mark} {r.name:<28} {r.quant:<9} {r.total_gb:>6.1f} GB  "
                      f"{tag}{r.est_tok_s:>5.1f} tok/s  ctx={r.ctx}" +
                      (f"  ({r.note})" if r.note else ""))
    elif verb == "serve":
        if not rest:
            print("usage: gb_synapse.py serve <model> [port] [--ctx N] [--n-slots N] "
                  "[--extra-args '...'] [--no-cluster]", file=sys.stderr)
            return 2
        # serve()'s own signature already takes ctx/n_slots/extra_args/
        # use_cluster — this CLI verb only ever exposed model+port. Parse
        # the extra flags the same "pop from rest" style pull() uses above.
        ctx = 0
        n_slots = -1
        extra_args = ""
        use_cluster = True
        for flag, attr in (("--ctx", "ctx"), ("--n-slots", "n_slots"),
                           ("--extra-args", "extra_args")):
            if flag in rest:
                i = rest.index(flag)
                val = rest[i + 1] if i + 1 < len(rest) else ""
                rest = rest[:i] + rest[i + 2:]
                if attr == "ctx":
                    ctx = int(val)
                elif attr == "n_slots":
                    n_slots = int(val)
                else:
                    extra_args = val
        if "--no-cluster" in rest:
            rest = [a for a in rest if a != "--no-cluster"]
            use_cluster = False
        port = int(rest[1]) if len(rest) > 1 else DEFAULT_PORT
        st = serve(rest[0], port=port, ctx=ctx, use_cluster=use_cluster,
                  n_slots=n_slots, extra_args=extra_args)
        print(json.dumps(asdict(st)) if llm
              else f"serving {st.model} on :{st.port}  (tensor-split {st.tensor_split})")
    elif verb == "stop":
        if not rest:
            print("usage: gb_synapse.py stop <model>", file=sys.stderr)
            return 2
        print("stopped" if stop(rest[0]) else "not running")
    elif verb == "ps":
        running = ps()
        if llm:
            print(json.dumps(running))
        else:
            print("\n".join(f"  {r['model']}  :{r['port']}  tensor-split={r['tensor_split']}"
                             for r in running) or "  (none running)")
    elif verb == "logs":
        if not rest:
            print("usage: gb_synapse.py logs <model> [--proxy] [-n LINES]", file=sys.stderr)
            return 2
        label = rest[0] + ("_proxy" if "--proxy" in rest else "")
        n = 100
        if "-n" in rest:
            i = rest.index("-n")
            if i + 1 < len(rest):
                n = int(rest[i + 1])
        path = log_path(label)
        if not path.is_file():
            print(f"(no log yet: {path})", file=sys.stderr)
            return 1
        lines = path.read_text(errors="replace").splitlines()[-n:]
        print(json.dumps({"path": str(path), "lines": lines}) if llm
              else "\n".join(lines))
    else:
        print(f"unknown verb: {verb}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main(sys.argv[1:]))

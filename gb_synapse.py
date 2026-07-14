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
        (vLLM if installed, else a transformers fallback — one interface,
        gb-synapse picks the runtime; see workflow/gb-synapse.md).
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
ENGINE_SRC_DIR = _REPO_DIR.parent / "greenboost-sources" / "llama.cpp"
LLAMA_CPP_REMOTE = "https://github.com/ggml-org/llama.cpp"

CONFIG_DIR = Path("/etc/greenboost/synapse")
HF_TOKEN_FILE = CONFIG_DIR / "hf_token"

MODEL_STORE_DIR = Path(os.environ.get("GB_SYNAPSE_MODEL_DIR", "/var/lib/greenboost/synapse/models"))
MANIFEST_FILE = MODEL_STORE_DIR / "manifest.json"

RUN_DIR = Path(os.environ.get("GB_SYNAPSE_RUN_DIR", "/run/greenboost/synapse"))
SLOT_DIR = Path(os.environ.get("GB_SYNAPSE_SLOT_DIR", "/var/lib/greenboost/synapse/slots"))

DEFAULT_PORT = 11434
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


def fetch_engine_source(remote: str = LLAMA_CPP_REMOTE, src_dir: Path = ENGINE_SRC_DIR) -> str:
    """Clone src_dir from `remote` if absent, else fast-forward it to
    origin's default branch. Returns `git describe` for the checked-out tree.

    This is the "fetch latest llama.cpp" mechanism: the vendored tree at
    greenboost-sources/llama.cpp starts as a point-in-time snapshot; this
    function is how `greenboost synapse update-engine` tracks upstream
    github.com/ggml-org/llama.cpp going forward.
    """
    if not shutil.which("git"):
        raise RuntimeError("gb-synapse engine fetch needs git: "
                           + _install_hint("git") + "   (or: sudo greenboost install-deps)")
    src_dir = Path(src_dir)
    if not (src_dir / ".git").exists():
        src_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", remote, str(src_dir)])
    else:
        _run(["git", "-C", str(src_dir), "fetch", "--depth", "1", "origin"])
        show = _run(["git", "-C", str(src_dir), "remote", "show", "origin"], capture=True).stdout
        branch = "master"
        for line in show.splitlines():
            if "HEAD branch" in line:
                branch = line.rsplit(":", 1)[-1].strip()
        _run(["git", "-C", str(src_dir), "reset", "--hard", f"origin/{branch}"])
    return _run(["git", "-C", str(src_dir), "describe", "--always", "--dirty"],
                capture=True).stdout.strip()


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
    if not src_dir.exists():
        raise RuntimeError(f"llama.cpp source not found at {src_dir}; "
                            f"run fetch_engine_source() first")
    build_dir = src_dir / "build-synapse"
    jobs = jobs or os.cpu_count() or 4

    _run(["cmake", "-S", str(src_dir), "-B", str(build_dir),
          "-DCMAKE_BUILD_TYPE=Release",
          "-DGGML_CUDA=ON",
          "-DGGML_RPC=ON",
          f"-DCMAKE_CUDA_ARCHITECTURES={_cuda_arch()}",
          f"-DCMAKE_CUDA_COMPILER={_find_nvcc()}",
          "-DLLAMA_CURL=OFF"])  # gb-synapse owns HF downloads; no need for llama.cpp's own fetcher
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

    version = _run(["git", "-C", str(src_dir), "describe", "--always", "--dirty"],
                    capture=True).stdout.strip()
    (install_dir / "engine.version").write_text(version + "\n")
    return {"version": version, "install_dir": str(install_dir)}


def update_engine() -> dict:
    """Pull the latest github.com/ggml-org/llama.cpp and rebuild in place."""
    fetch_engine_source()
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
    and/or the :11434 Ollama/OpenAI proxy (gb_synapse_api) are running now.

    Single source of truth for the `synapse_status` MCP tools (gb_dataflux_mcp,
    gb_synapse_mcp, gb_mcp) and the `status` CLI verb. Matches gb-synapse's OWN
    engine path — not ollama's internal llama-server."""
    import subprocess
    out = {"engine_built": engine_installed(),
           "engine_version": engine_version() or None,
           "server_running": False, "proxy_running": False,
           "engine_dir": str(ENGINE_DIR)}
    for key, pat in (("server_running", f"{ENGINE_DIR}/llama-server"),
                     ("proxy_running", "gb_synapse_api")):
        try:
            r = subprocess.run(["pgrep", "-f", pat],
                               capture_output=True, text=True, timeout=5)
            out[key] = bool(r.stdout.strip())
        except Exception:
            pass
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
    arch: str = ""
    engine: str = "llama.cpp"   # "llama.cpp" | "gbquant" (vLLM if available, else transformers)
    n_bytes: int = 0
    n_layers: int = 0
    is_moe: bool = False
    n_experts: int = 0
    n_experts_used: int = 0
    ctx_length: int = 0
    # KV-cache geometry from GGUF metadata (0 => unknown, fall back to the
    # param-count bucket heuristic in estimate_kv_gb).
    n_kv_heads: int = 0
    head_dim: int = 0
    added_ts: float = 0.0


def _load_manifest() -> dict[str, ModelEntry]:
    try:
        raw = json.loads(MANIFEST_FILE.read_text())
        return {k: ModelEntry(**v) for k, v in raw.items()}
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
    from huggingface_hub import HfApi
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

    from huggingface_hub import snapshot_download
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


# Tokens that mean "quantize via gb-quant, serve through vLLM/transformers"
# instead of a GGUF/llama.cpp quant. Deliberately excludes BF16 (a real GGUF
# quant filename token too, e.g. "model-BF16.gguf") to avoid ambiguity —
# FP8/INT8/INT4 are never real GGUF quant tokens (those use QX_K/QX_0 names).
_GBQUANT_TOKENS = {"FP8", "INT8", "INT4"}


def _find_vllm_bin() -> str | None:
    """vLLM search order: explicit override, gb-synapse-managed venv (mirrors
    the built-engine layout under ENGINE_DIR's parent), then system PATH.
    vLLM lives in its own venv (its own docs: `pip install -e
    ~/Dev/turboquantsolutions/gemlite` "in the vLLM venv") — it's never
    assumed to share gb_synapse.py's own interpreter."""
    override = os.environ.get("GB_SYNAPSE_VLLM_BIN")
    if override and Path(override).exists():
        return override
    managed = Path.home() / ".local/share/greenboost/synapse/vllm-env/bin/vllm"
    if managed.exists():
        return str(managed)
    return shutil.which("vllm")


def _pull_gbquant(repo: str, quant: str, name: str | None) -> ModelEntry:
    """Fetch the safetensors snapshot for gb-quant serving (vLLM's on-the-fly
    `--quantization gemlite` or the transformers fallback both quantize at
    load time — no GGUF conversion needed here, unlike _pull_and_convert)."""
    from huggingface_hub import snapshot_download
    cache_dir = str(MODEL_STORE_DIR / "_hf_cache")
    local_dir = snapshot_download(
        repo_id=repo, token=hf_token(), cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "tokenizer*"],
    )
    entry_name = name or repo.split("/")[-1]
    entry = ModelEntry(name=entry_name, path=local_dir, source="hf", repo=repo,
                        engine="gbquant", quant=quant.upper(), added_ts=time.time())
    manifest = _load_manifest()
    manifest[entry_name] = entry
    _save_manifest(manifest)
    return entry


def pull(repo_spec: str, name: str | None = None) -> ModelEntry:
    """Download a GGUF from HuggingFace — or, if the repo has no native GGUF
    release, convert it from safetensors on the fly (see _pull_and_convert).

    repo_spec is "org/repo" or "org/repo:QUANT". With no quant given, picks
    the largest quant whose weights fit the live cluster's aggregate VRAM
    (falls back to the smallest available file if nothing fits — still
    downloadable, just flagged by `recommend` as needing overflow tiers).
    For a repo with no GGUF release, QUANT instead selects the target
    llama-quantize format (default Q4_K_M).
    Multi-shard GGUFs ("...-00001-of-00003.gguf") pull every shard; llama.cpp
    loads the first and follows the split chain itself.
    """
    repo, _, quant = repo_spec.partition(":")
    if quant.upper() in _GBQUANT_TOKENS:
        return _pull_gbquant(repo, quant, name)
    files = list_repo_gguf(repo)
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
    from huggingface_hub import hf_hub_download
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


def gguf_summary(path: str) -> dict:
    """Parse layer count, quant type, MoE expert config, context length, and
    total weight bytes from a GGUF file. Reuses the vendored llama.cpp
    GGUFReader via gb_gguf_tensor_map."""
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
    quant = ""
    weight_types = Counter(t.tensor_type.name for t in reader.tensors if "weight" in t.name)
    if weight_types:
        quant = weight_types.most_common(1)[0][0]
    return {
        "n_bytes": n_bytes, "n_layers": n_layers, "is_moe": is_moe,
        "n_experts": n_experts, "n_experts_used": n_experts_used,
        "dense_bytes": dense_bytes, "expert_bytes": expert_bytes,
        "ctx_length": ctx_length, "quant": quant, "arch": arch,
        "n_kv_heads": head_count_kv, "head_dim": head_dim,
    }


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
        "vllm_available": _find_vllm_bin() is not None,
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


MEASURED_TOK_S_FILE = MODEL_STORE_DIR / "measured_tok_s.json"
_MEASURED_TOK_S_MAX_SAMPLES = 20


def _df_emit_tok_s(model: str, tok_s: float) -> None:
    """Record a real measured tok/s sample to the dataflux log , the one
    number that closes the loop between orchestration decisions (tier_move/
    quantize/turboquant_activate) and what they actually bought. Best-effort,
    never raises, same contract as every other dataflux emit call site."""
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "node": "host", "label": "gb_synapse", "kind": "tok_s_measured",
            "n_items": 1, "items": [model], "duration_s": 0.0, "status": "ok",
            "model": model, "tok_s": round(tok_s, 1),
        })
    except Exception:
        pass


def record_measured_tok_s(model: str, tok_s: float) -> None:
    """Append a real, client-observed decode speed for `model` — fed by
    greenboost-cli after each final answer (TurnComplete.tok_s), closing the
    gap _estimate_tok_s()'s docstring flags: "A --measure mode that runs a
    real warmup is future work." recommend() prefers this rolling average
    over the bandwidth heuristic whenever samples exist for a model.
    Best-effort: silently skipped without write access to MODEL_STORE_DIR,
    same as index_ollama_models()'s persistence — this is a nice-to-have
    calibration aid, not something worth failing a turn over."""
    if tok_s <= 0:
        return
    _df_emit_tok_s(model, tok_s)
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
        kv_gb = estimate_kv_gb(ctx, entry.n_bytes, entry.quant,
                               n_layers=entry.n_layers, n_kv_heads=entry.n_kv_heads,
                               head_dim=entry.head_dim)
        total_gb = weights_gb + kv_gb
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
                                  measured=measured is not None))
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


def _resolve_model(spec: str) -> ModelEntry:
    manifest = _load_manifest()
    if spec in manifest:
        return manifest[spec]
    if "/" in spec:  # looks like an HF repo spec — pull it
        return pull(spec)
    blob = ollama_model_blob(spec)
    if blob:
        meta = gguf_summary(blob)
        entry = ModelEntry(name=spec, path=blob, source="ollama", added_ts=time.time(), **meta)
        manifest[spec] = entry
        _save_manifest(manifest)
        return entry
    raise KeyError(f"no such model: {spec} (not in manifest, not an HF repo, not an Ollama model)")


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

    # rpc-server is ggml/CUDA, so it hits the SAME shim incompatibility as
    # llama-server (SIGABRT: "invalid device function"). Preloading the shim
    # here would take the feeder's GPU down with it, which is the opposite of
    # the point. Gate it on the same switch as the host engine.
    shim_pre = (f'LD_PRELOAD={gb_cluster.GREENBOOST_SHIM} GREENBOOST_ACTIVE=1 '
                if os.environ.get("GB_SYNAPSE_SHIM", "0") == "1" else "")

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
                    n_devices: int = 1) -> int:
    """How many layers actually fit the cluster's free VRAM.

    llama.cpp puts EVERY layer on the GPU unless told otherwise, and
    --tensor-split then spreads them by ratio — so a model larger than the
    aggregate VRAM doesn't degrade, it dies mid-load ("failed to allocate
    RPC0 buffer of size 12676098304" against a feeder with 7.5 GB free).
    Sizing the layer count to the real budget is what turns "won't load" into
    "loads, with the tail on the CPU": every layer we can keep on a GPU is one
    that isn't paying the CPU penalty.
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
    usable = budget_gb - kv_gb - _compute_reserve_gb(budget_gb * 1024.0 / n_dev) * n_dev
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
    kv_f16_gb = estimate_kv_gb(ctx, entry.n_bytes, entry.quant,
                               n_layers=entry.n_layers, n_kv_heads=entry.n_kv_heads,
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
    kv_budget_gb = max(budget_gb - weights_gb, 0.25)
    kv_gb = estimate_kv_gb(requested_ctx, entry.n_bytes, entry.quant,
                           n_layers=entry.n_layers, n_kv_heads=entry.n_kv_heads,
                           head_dim=entry.head_dim)
    if kv_gb <= kv_budget_gb:
        return requested_ctx
    safe_ctx = max(2048, int(requested_ctx * kv_budget_gb / kv_gb) // 256 * 256)
    print(f"  [gb-synapse] {entry.name}: ctx={requested_ctx} would need {kv_gb:.2f} GB KV "
          f"cache on top of {weights_gb:.2f} GB weights ({budget_gb:.2f} GB budget) — "
          f"clamping to ctx={safe_ctx}", flush=True)
    return safe_ctx


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
    """
    import urllib.error
    import urllib.request

    deadline = time.time() + grace_s
    url = f"http://127.0.0.1:{internal_port}/health"
    while time.time() < deadline:
        # poll() is the authoritative answer for a child WE spawned, and it
        # reaps — unlike a PID check, which a zombie passes.
        if proc.poll() is not None:
            raise RuntimeError(
                f"'{entry.name}' failed to load: the engine exited during startup "
                f"(rc={proc.returncode}).\n"
                f"    log: {_run_log_path(entry.name)}\n{_upstream_log_tail(entry.name)}")
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


def _health_says_loading(body: bytes) -> bool:
    """llama.cpp is not consistent about how /health reports a load in progress:
    newer builds answer 503, but the engine we ship answers 200 with
    {"status": "loading model"}. Trusting the status code alone marks a model
    "served" seconds before it has parsed its own hyperparameters — the exact
    window in which a bad GGUF exits. vLLM/gb_llm_server answer 200 with no
    status field, which is genuinely ready."""
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


def _launch_proxy_and_record(entry: ModelEntry, upstream: subprocess.Popen, port: int,
                              internal_port: int, tensor_split: str = "",
                              feeders: list | None = None, **serve_facts) -> ServerState:
    """Shared tail for every engine (llama.cpp/vLLM/transformers): wait for the
    engine to come up, start the gb_synapse_api.py proxy in front of it, and
    record ServerState. The proxy only ever talks OpenAI /v1/* to the upstream,
    so it's identical regardless of which engine is behind it — and all three
    expose /health, so the readiness gate is too."""
    t0 = time.time()
    common = {"port": port, "tensor_split": tensor_split, "feeders": feeders or [],
              **serve_facts}
    try:
        ready = _wait_upstream_ready(entry, upstream, internal_port)
    except RuntimeError as e:
        # No proxy is started for a dead engine: a proxy on :11434 in front of
        # nothing is what turns "the model failed to load" into "connection
        # closed mid-stream" three layers away.
        _emit_serve(entry, "error", error=str(e).splitlines()[0],
                    load_s=round(time.time() - t0, 1), **common)
        raise

    proxy_cmd = [sys.executable, str(_REPO_DIR / "gb_synapse_api.py"),
                 "--port", str(port), "--upstream-port", str(internal_port),
                 "--model-name", entry.name]
    proxy_log = open(_run_log_path(entry.name + "_proxy"), "ab")
    proxy_proc = subprocess.Popen(proxy_cmd, stdout=proxy_log, stderr=subprocess.STDOUT,
                                   start_new_session=True)

    state = ServerState(model=entry.name, llama_pid=upstream.pid, proxy_pid=proxy_proc.pid,
                         port=port, internal_port=internal_port, tensor_split=tensor_split,
                         feeders=feeders or [], started_ts=time.time())
    _write_run_state(state)
    _emit_serve(entry, "ok" if ready else "loading",
                load_s=round(time.time() - t0, 1), **common)
    return state


# vLLM CLI args per gb-quant token, using vLLM's OWN native quantization
# (not the third-party gemlite/hqq vLLM plugins — live-tested against vLLM
# 0.24.0 and both are broken: gemlite's backend.py and hqq's vllm.py both
# subclass/import internal quantization classes — AWQConfig, AWQMarlinConfig,
# HQQMarlinConfig — that vLLM has since renamed/removed. Both plugins also
# load eagerly on ANY `vllm` invocation regardless of --quantization, so
# merely having one installed breaks vLLM's own --version). fp8 and the
# bitsandbytes default (4-bit) are confirmed working end-to-end; true 8-bit
# needs a config-injection mechanism this session didn't nail down in vLLM
# 0.24.0, so INT8 currently gets the same (4-bit) args as INT4 rather than
# silently mislabeling it — see workflow/gb-synapse.md.
_GBQUANT_VLLM_ARGS = {
    "FP8": ["--quantization", "fp8"],
    "INT8": ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"],
    "INT4": ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"],
}


def _cuda_home_for_vllm() -> "str | None":
    """vLLM's FlashInfer dependency JIT-compiles Blackwell (SM 12.x) kernels
    and requires CUDA >= 12.9 to do it — it finds nvcc via CUDA_HOME, which
    defaults to /usr (a stale distro-packaged nvcc, often < 12.9) when unset.
    Same root cause as _find_nvcc(): the distro package and
    /usr/local/cuda's update-alternatives-managed install can diverge.
    FlashInfer swallows the resulting version-check exception internally and
    misreports "GPU too old" instead of "CUDA too old" — confirmed live on
    an RTX 5070 (cc 12.0) with CUDA 13.3 at /usr/local/cuda but 12.4 at
    /usr/bin/nvcc."""
    alt = Path("/usr/local/cuda")
    return str(alt) if alt.exists() else None


def _gbquant_model_source(entry: ModelEntry) -> str:
    """Local snapshot dir if `_pull_gbquant()` already downloaded one (avoids
    vLLM/transformers redundantly re-fetching the same repo into their own
    default HF cache — ~/.cache/huggingface/hub — instead of reusing
    gb-synapse's own MODEL_STORE_DIR/_hf_cache copy), else the bare repo id
    so vLLM/transformers can fetch it themselves."""
    return entry.path if entry.path and os.path.isdir(entry.path) else entry.repo


def _serve_vllm(entry: ModelEntry, port: int) -> ServerState:
    """vLLM's own OpenAI server speaks /v1/chat/completions + /v1/completions
    natively (streaming included), so gb_synapse_api.py's existing passthrough
    needs no changes — it just points at vLLM's port instead of llama-server's.
    gb-quant applies via vLLM's own native quantization flags (see
    _GBQUANT_VLLM_ARGS), quantizing on the fly at load time — no separate
    convert/quantize step.

    Single-node only for now: vLLM's own cluster mechanism (Ray tensor/pipeline
    parallel) is a different distribution model than llama.cpp's --rpc split
    and isn't wired to GreenBoost's feeder fabric here."""
    vllm_bin = _find_vllm_bin()
    if not vllm_bin:
        raise RuntimeError(
            f"'{entry.name}' needs vLLM (gb-quant {entry.quant} quantization) but no vllm "
            f"binary was found — install it in its own venv and either put it on PATH or "
            f"set GB_SYNAPSE_VLLM_BIN."
        )
    internal_port = port + 1000
    env = os.environ.copy()
    env["GREENBOOST_ACTIVE"] = "1"  # gb_init's torch-layer hooks; vLLM never uses the CUDA shim
    if hf_token():
        env["HF_TOKEN"] = hf_token()
    cuda_home = _cuda_home_for_vllm()
    if cuda_home:
        env["CUDA_HOME"] = cuda_home

    # vLLM's own memory profiling asserts hard if free VRAM changes between
    # its initial snapshot and the profiling pass (observed live: another
    # process's VRAM use fluctuating ~1.5 GB tripped this at the library's
    # default 0.9 utilization) — size to LIVE free VRAM with a safety margin,
    # same "never trust a static default" approach as _compute_tensor_split.
    from gb_nvml import get_nvml
    _, host_free_mb, host_total_mb, _ = get_nvml(0).mem()
    if host_total_mb > 0:
        util = max(0.3, min(0.85, (host_free_mb * 0.85) / host_total_mb))
    else:
        # pynvml not installed — this env's gb_nvml returns all zeros by
        # design (see its module docstring). Fall back to torch.cuda, same
        # pattern as gb_llm._auto_budget_gb(); torch is already a hard
        # requirement of both vLLM and the transformers fallback, so this
        # import is free on any path that reaches _serve_vllm().
        try:
            import torch
            free_b, total_b = torch.cuda.mem_get_info()
            util = max(0.3, min(0.85, (free_b * 0.85) / total_b))
        except Exception:
            util = 0.85

    quant_args = _GBQUANT_VLLM_ARGS.get(entry.quant.upper(), ["--quantization", "fp8"])
    cmd = [vllm_bin, "serve", _gbquant_model_source(entry),
           *quant_args, "--served-model-name", entry.name,
           "--host", "127.0.0.1", "--port", str(internal_port),
           "--gpu-memory-utilization", f"{util:.2f}"]
    log = open(_run_log_path(entry.name), "ab")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    return _launch_proxy_and_record(entry, proc, port, internal_port)


def _serve_transformers(entry: ModelEntry, port: int) -> ServerState:
    """Fallback when vLLM isn't installed: gb_llm_server.py wraps
    gb_llm.load_causal_lm()/generate() (in-process transformers + gb-quant)
    behind the same minimal OpenAI surface vLLM/llama-server provide. No
    continuous batching — single-request-at-a-time — so prefer vLLM when
    it's available; this exists so ':fp8'-style models are never a hard
    dead end just because the vLLM venv isn't set up yet."""
    internal_port = port + 1000
    env = os.environ.copy()
    env["GREENBOOST_ACTIVE"] = "1"
    if hf_token():
        env["HF_TOKEN"] = hf_token()
    cmd = [sys.executable, str(_REPO_DIR / "gb_llm_server.py"),
           "--model", _gbquant_model_source(entry), "--served-model-name", entry.name,
           "--quant", entry.quant or "fp8",
           "--host", "127.0.0.1", "--port", str(internal_port)]
    log = open(_run_log_path(entry.name), "ab")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    return _launch_proxy_and_record(entry, proc, port, internal_port)


def serve(model: str, port: int = DEFAULT_PORT, ctx: int = 0,
          use_cluster: bool = True, n_slots: int = 1, extra_args: str = "") -> ServerState:
    """Resolve `model` (manifest name, "org/repo[:quant]", or a bare Ollama
    model name), bring up feeder rpc-server(s) if online, launch the host
    llama-server with an explicit --rpc + --tensor-split, then start the API
    proxy in front of it.

    Also the single llama-server process manager for GreenBoost: carries the
    flags formerly duplicated in greenboost-cli's own `/llamaserve` (disk-
    persisted prompt-cache slots, P-core-pinned threading, KV quantization
    matching Ollama's TurboQuant baseline, `--jinja` for correct native
    tool-calling) so nothing outside this function spawns a competing
    llama-server instance. `extra_args` (e.g. greenboost-cli's
    `llamacpp_extra_args` setting) appends further raw CLI flags, shlex-split.

    `entry.engine == "gbquant"` (repo pulled with a ":fp8"/":int8"/":int4"
    quant token) routes to vLLM if installed, else the transformers fallback
    — one interface, gb-synapse picks the runtime, per the project's "gb-serve
    packs every option into one backend" design (see workflow/gb-synapse.md).
    """
    entry = _resolve_model(model)

    # Reuse an already-running instance for this exact model instead of
    # racing a second launch — e.g. greenboost-cli's own auto-start-at-launch
    # can overlap with a manual /llamaserve start. Ollama's scheduler solves
    # this with a queue + refcounted runners (server/sched.go); gb-synapse
    # only ever runs one model at a time, so "return the existing one" is
    # the equivalent guarantee without needing that machinery.
    for st in _read_run_states():
        if st.model == entry.name and _pid_alive(st.llama_pid) and _pid_alive(st.proxy_pid):
            return st

    if entry.engine == "gbquant":
        return _serve_vllm(entry, port) if _find_vllm_bin() else _serve_transformers(entry, port)

    if not engine_installed():
        raise RuntimeError("engine not built — run: greenboost synapse build-engine")

    supported = _engine_supported_archs()
    if entry.arch and supported and entry.arch not in supported:
        hint = (f"run it directly with Ollama instead: ollama run {entry.name}"
                if entry.source == "ollama" else
                "wait for upstream llama.cpp to add support, or pick a different quant/repo")
        raise RuntimeError(
            f"'{entry.name}' uses architecture '{entry.arch}', which this gb-synapse engine "
            f"build (llama.cpp {engine_version()}) doesn't recognize — likely a custom/newer "
            f"arch only another runtime supports; {hint}."
        )

    from gb_nvml import get_nvml
    _, host_free_mb, _, _ = get_nvml(0).mem()

    online_feeders, rpc_args = [], []
    if use_cluster:
        for i, f in enumerate(gb_cluster.feeders(probe=True)):
            if not f.online:
                continue
            rpc_port = RPC_PORT_BASE + i
            # Only a feeder we can actually REACH may influence the budget and
            # the split. Degrading to a host-only serve is always better than
            # handing llama.cpp a device it can't talk to, which fails the load
            # outright instead of just being slower.
            if not ensure_feeder_rpc(f, rpc_port):
                continue
            rpc_args.append(f"{f.ip}:{rpc_port}")
            online_feeders.append(f)
    budget_gb = host_free_mb / 1024 + sum(f.t1_free_mb for f in online_feeders) / 1024

    internal_port = port + 1000
    env = gb_cluster.shim_env(workload="llm", enabled=True)
    env["GREENBOOST_CLUSTER"] = "0"  # RPC owns cross-node split; shim = local tiers only
    # Pin the loader to the engine dir: the binary's RUNPATH points at the
    # BUILD TREE, so without this a rebuilt tree silently swaps the ggml libs
    # under an older binary (confirmed by crash backtraces referencing
    # build-synapse/bin paths) — two engine versions in one process.
    env["LD_LIBRARY_PATH"] = str(ENGINE_DIR) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")

    # The shim currently aborts llama.cpp's CUDA backend on this host:
    # LD_PRELOAD of libgreenboost_cuda.so makes cudaFuncGetAttributes return
    # "invalid device function" inside ggml_cuda_kernel_can_use_pdl (reproduced
    # with GREENBOOST_ACTIVE=0, so it is the interposition itself, not a
    # GreenBoost code path; a plain cudart-12 test binary is unaffected, so the
    # trigger is ggml's late-dlopen'ed backend). Serving through a shim that
    # kills the engine is strictly worse than serving without it: RPC still
    # splits across the cluster and gb-quant still holds quality, we only lose
    # per-node T2/T3 spill. Probe once, then decide — never assume.
    if os.environ.get("GB_SYNAPSE_SHIM", "0") != "1":
        for k in ("LD_PRELOAD", "GREENBOOST_ACTIVE"):
            env.pop(k, None)
        shim_note = "off (known llama.cpp/CUDA incompatibility)"
    else:
        shim_note = "on (GB_SYNAPSE_SHIM=1)"

    # -ngl 999 ("every layer on the GPU") is right only when the weights can
    # actually LAND there. Without the shim inflating VRAM, forcing all layers
    # onto a card that cannot hold them doesn't spill, it fails the load
    # outright ("unable to allocate CUDA0 buffer of size 21398561280"). When the
    # weights exceed the VRAM budget, let llama.cpp fit the layers itself: it
    # puts as many on the GPU as fit and keeps the rest in host RAM. That is
    # slower, and it is what the cluster (RPC) and the shim (T2) exist to avoid,
    # so we say plainly which one we got.
    weights_gb = entry.n_bytes / (1024 ** 3)
    fits_vram = weights_gb <= budget_gb or "LD_PRELOAD" in env
    cpu_quirk = (not fits_vram and entry.arch in ARCH_CPU_SPLIT_BROKEN
                 and os.environ.get("GB_SYNAPSE_FORCE_SPLIT") != "1")

    # ctx=0 means "whatever the model itself was certified for" — a 1M YaRN bake
    # exists precisely so it can be USED, and a hardcoded 64K default silently
    # discarded 94% of it. The budget clamp below still decides what actually
    # fits, so asking for the model's full context is never dangerous, just
    # honest about the ceiling.
    #
    # Which budget, though: when `cpu_quirk` is about to force this model onto
    # the CPU-only arch-split-crash path, weights AND KV cache live in plain
    # host+feeder RAM, not VRAM — clamping ctx/KV-type against the (already
    # weights-insufficient) VRAM budget starved ctx down to a few thousand
    # tokens even with 60+ GB of RAM sitting idle (observed: ctx clamped to
    # 6144 on a model certified to 1M). Use the real RAM budget in that case.
    ctx_kv_budget_gb = budget_gb
    if cpu_quirk:
        ram_free_mb = _read_ram_available_mb()
        ctx_kv_budget_gb = max(
            budget_gb,
            ram_free_mb / 1024 + sum(f.t2_free_mb for f in online_feeders) / 1024)

    if ctx <= 0:
        ctx = entry.ctx_length or 65536
    ctx = _clamp_ctx_to_budget(ctx, entry, ctx_kv_budget_gb)

    # KV precision is a QUALITY tier, not a memory knob, so it is chosen against
    # the budget rather than hardcoded: f16 is certification grade, q8_0 is a
    # budget config that buys ~2x the cache for roughly one needle at long
    # rungs. Serving q8_0 unconditionally (as this did) quietly demoted every
    # model to the budget tier — including ones whose KV fits f16 with room to
    # spare. Take f16 whenever it fits; drop to q8_0 only to make the context
    # reachable at all, and say so.
    kv_type = _pick_kv_type(ctx, entry, ctx_kv_budget_gb)

    # Split reflects the (clamped) ctx's KV footprint at the KV tier we actually
    # serve — an f16 cache is twice a q8_0 one, and a split sized for the wrong
    # tier pushes weights onto a node that has no room left for them.
    kv_total_gb = estimate_kv_gb(ctx, entry.n_bytes, entry.quant,
                                 n_layers=entry.n_layers, n_kv_heads=entry.n_kv_heads,
                                 head_dim=entry.head_dim,
                                 kv_bytes_per_elem=2.0 if kv_type == "f16" else 1.0)
    tensor_split = _compute_tensor_split(host_free_mb, online_feeders, kv_total_gb)

    SLOT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [str(ENGINE_DIR / "llama-server"), "-m", entry.path,
           "--host", "127.0.0.1", "--port", str(internal_port),
           "--ctx-size", str(ctx),
           "--slot-save-path", str(SLOT_DIR), "--no-webui",
           "--flash-attn", "auto",
           # Reuse cached KV for repeated prefixes (system prompt, prior turns)
           # instead of reprocessing them from scratch every request — the
           # model card's own recommended invocation for agentic/Claude-Code
           # use includes this exact flag. Without it, a multi-turn tool-call
           # loop (call → tool result → final answer) reprocesses the whole
           # growing context on EVERY turn; confirmed live (2026-07-13) this
           # was the dominant cost, not raw decode speed, on a CPU-only serve.
           "--cache-reuse", "256",
           "--no-mmap",  # DMA-BUF pinning (Path A) is incompatible with mmap-backed pages
           "-np", str(n_slots), "--threads", str(_pcore_threads()),
           "--cache-type-k", kv_type, "--cache-type-v", kv_type,
           # Renders the model's own Jinja chat template instead of llama-server's
           # built-in simplified matcher — required for correct native tool-calling
           # on models whose template defines a tool-call format (Qwen3 family and
           # most modern instruct models). Without it, required tool arguments can
           # go missing from emitted tool_calls (confirmed live: qwen36-coder:studio
           # calling Glob/Read with no "pattern"/"file_path"). See known-issues.md.
           "--jinja"]

    n_cpu_moe = 0
    if fits_vram:
        cmd += ["-ngl", "999"]        # all layers on GPU — the fast path
    elif cpu_quirk:
        # Upstream llama.cpp CUDA regression (2026-07-12, engine 6b4dc2116):
        # for this arch ANY CPU/GPU split aborts at the first batched prompt
        # ("ggml_cuda_compute_forward: SOFT_MAX failed — invalid argument").
        # Verified exhaustively: -ngl 10 plain, --n-cpu-moe, --cpu-moe, FA
        # on/off, GGML_CUDA_DISABLE_GRAPHS, -ub 8, --no-op-offload — all crash;
        # -ngl 0 works (11.2 tok/s measured, 3B-active MoE). Full-VRAM would
        # also work but no quant of this model fits the cluster. Until the
        # kernel is fixed upstream, CPU-only is the ONLY placement that serves
        # at all — an explicit, temporary exception to the GPU-always rule,
        # not a policy change. Re-test with GB_SYNAPSE_FORCE_SPLIT=1.
        # See workflow/known-issues.md + ggml-org/llama.cpp#19816 (same family).
        #
        # -ngl 0 alone is NOT CPU-only: llama.cpp still creates a CUDA context
        # and op-offloads batched matmuls to the GPU above ~32 tokens — a tiny
        # curl probe passed while greenboost-cli's 4k-token agentic prompt
        # crashed the same way. Hide the GPU entirely so no tensor can touch
        # the broken kernel, whatever the batch size.
        cmd += ["-ngl", "0", "--no-op-offload"]
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif entry.is_moe:
        # Every layer still runs on the GPU; only the experts of the first
        # n_cpu_moe layers live in host DDR.
        n_cpu_moe = _fit_cpu_moe_layers(entry, budget_gb, kv_total_gb,
                                        1 + len(online_feeders))
        cmd += ["-ngl", "999", "--n-cpu-moe", str(n_cpu_moe)]
    else:
        n_gpu = _fit_gpu_layers(entry, budget_gb, kv_total_gb, 1 + len(online_feeders))
        cmd += ["-ngl", str(n_gpu)]

    # No mmap: DMA-BUF pinning needs owned pages. Without the shim there is
    # nothing to pin, and mmap lets the page cache serve a 20 GB GGUF instead of
    # re-reading it from disk on every restart.
    if "LD_PRELOAD" not in env:
        cmd.remove("--no-mmap")

    # Vision: a GGUF's projector is a SEPARATE file. Without --mmproj a VLM is
    # served text-only and every image is silently dropped — the model answers
    # confidently about an image it never saw, which is worse than an error.
    mmproj = _find_mmproj(entry)
    if mmproj:
        cmd += ["--mmproj", str(mmproj)]

    # MTP: models carrying a grafted multi-token-prediction layer (the aviary-1m
    # fleet bakes one in) decode ~34% faster through llama.cpp's speculative
    # path, with identical output — the draft head IS the model's own layer, so
    # this costs no quality and no extra VRAM for a draft model.
    if _has_mtp(entry):
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(MTP_DRAFT_N)]

    if rpc_args:
        cmd += ["--rpc", ",".join(rpc_args), "--tensor-split", tensor_split]
    if extra_args:
        cmd += shlex.split(extra_args)

    kv_grade = "certification-grade" if kv_type == "f16" else "budget"
    if fits_vram:
        placement = "all-GPU"
    elif cpu_quirk:
        placement = "CPU-ONLY (arch CUDA-split quirk — see known-issues.md)"
    else:
        placement = "PARTIAL CPU OFFLOAD (slow)"
    print(f"  [gb-synapse] {entry.name}: ctx={ctx} kv={kv_type} ({kv_grade}) "
          f"weights={weights_gb:.1f}GB/{budget_gb:.1f}GB budget → {placement}, "
          f"shim={shim_note}"
          f"{' +mtp' if _has_mtp(entry) else ''}{' +vision' if mmproj else ''}"
          f"{' rpc=' + ','.join(rpc_args) if rpc_args else ''}", flush=True)
    if cpu_quirk:
        print(f"  [gb-synapse] {weights_gb:.1f} GB of weights vs {budget_gb:.1f} GB VRAM "
              f"AND arch '{entry.arch}' cannot survive a CPU/GPU split on this llama.cpp "
              f"build (upstream CUDA regression) — serving CPU-ONLY. Measured 11.2 tok/s "
              f"decode (3B-active MoE). Re-test the split with GB_SYNAPSE_FORCE_SPLIT=1 "
              f"after an engine upgrade.", flush=True)
    elif not fits_vram and n_cpu_moe:
        print(f"  [gb-synapse] {weights_gb:.1f} GB of weights vs {budget_gb:.1f} GB VRAM: "
              f"ALL {entry.n_layers} layers stay on GPU (attention + KV); experts of "
              f"{n_cpu_moe}/{entry.n_layers} layers held in DDR "
              f"({_moe_expert_gb_per_layer(entry):.2f} GB/layer). Only 8 of "
              f"{entry.n_experts} experts fire per token, so this is the cheapest "
              f"weight in the model to keep out of VRAM.", flush=True)
    elif not fits_vram:
        _fit = _fit_gpu_layers(entry, budget_gb, kv_total_gb, 1 + len(online_feeders))
        print(f"  [gb-synapse] {weights_gb:.1f} GB of weights do not fit "
              f"{budget_gb:.1f} GB of VRAM: {_fit}/{entry.n_layers} layers on GPU, the "
              f"rest on CPU. CPU layers cost far more than any memory transfer — add "
              f"VRAM (another feeder) or a smaller quant to close the gap.", flush=True)

    llama_log = open(_run_log_path(entry.name), "ab")
    llama_proc = subprocess.Popen(cmd, env=env, stdout=llama_log, stderr=subprocess.STDOUT,
                                   start_new_session=True)

    return _launch_proxy_and_record(entry, llama_proc, port, internal_port,
                                     tensor_split=tensor_split,
                                     feeders=[f.ip for f in online_feeders],
                                     ctx=ctx, kv_type=kv_type, placement=placement,
                                     mtp=bool(_has_mtp(entry)), vision=bool(mmproj))


def stop(model: str) -> bool:
    for st in _read_run_states():
        if st.model != model:
            continue
        for pid in (st.llama_pid, st.proxy_pid):
            if _pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
        _run_state_path(model).unlink(missing_ok=True)
        return True
    return False


def ps() -> list[dict]:
    """Running gb-synapse servers. Entries whose llama-server PID has died
    are treated as stale and dropped."""
    live = []
    for st in _read_run_states():
        if _pid_alive(st.llama_pid):
            live.append(asdict(st))
        else:
            _run_state_path(st.model).unlink(missing_ok=True)
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
        fetch_engine_source()
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
            print("usage: gb_synapse.py pull <repo>[:quant] [name]", file=sys.stderr)
            return 2
        entry = pull(rest[0], name=rest[1] if len(rest) > 1 else None)
        print(json.dumps(asdict(entry)) if llm
              else f"pulled {entry.name}  ({entry.n_bytes / (1024 ** 3):.2f} GiB, {entry.quant})")
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
            print("usage: gb_synapse.py serve <model> [port]", file=sys.stderr)
            return 2
        st = serve(rest[0], port=int(rest[1]) if len(rest) > 1 else DEFAULT_PORT)
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
    else:
        print(f"unknown verb: {verb}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main(sys.argv[1:]))

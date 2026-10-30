"""gb_doctor: toolchain and host checks for `greenboost doctor` (round 2, 2026-10-30). Each check is {id, status ok|warn|info, detail, fix}; `fix` is one command, empty when ok.

Every `judge_*` function is pure (inputs in, verdict out) so it is unit-tested without a GPU, root or a build; `run_all()` only reads files (sysfs, the installed shim and engine,
the cli venv) and runs `nvcc --version`/`readelf` through gb_synapse. It starts no model and touches no GPU.
"""
from __future__ import annotations

import hashlib
import os
import re
import socket
from pathlib import Path
from typing import List, Optional


def _c(id_: str, status: str, detail: str, fix: str = "") -> dict:
    return {"id": id_, "status": status, "detail": detail, "fix": fix}


# ---- pure verdicts -------------------------------------------------------------------------------------------------------------------------------
def judge_engine_runtime(runtime_major: str, nvcc_major: str, engine_built: bool = True) -> dict:
    if not engine_built:
        return _c("cuda_runtime", "info", "no engine built yet, nothing to compare")
    if not runtime_major or not nvcc_major:
        return _c("cuda_runtime", "info", "engine runtime or nvcc not readable (no libggml-cuda, nvcc or readelf)")
    if runtime_major != nvcc_major:
        return _c("cuda_runtime", "warn", f"engine links CUDA runtime {runtime_major} but nvcc is CUDA {nvcc_major}: kernels fail with 'invalid argument'",
                  "sudo greenboost synapse build-moecache-engine   # after pointing GB_SYNAPSE_NVCC at the toolkit that matches the runtime")
    return _c("cuda_runtime", "ok", f"engine runtime and nvcc are both CUDA {nvcc_major}")


def judge_shim_rebind(installed: Optional[bytes], marker: bytes, repo_build: Optional[bytes] = None, repo_source: Optional[str] = None) -> dict:
    """The cudart-rebind force patch (patches/shim-cudart-rebind-force.patch) is in the installed shim when it carries the build marker, or, for a build that predates the marker,
    when it is byte-identical to this checkout's build of a source that has the patch."""
    fix = "make shim && sudo ./greenboost_setup.sh install   # then check: strings /usr/local/lib/libgreenboost_cuda.so | grep GB-MARKER"
    if installed is None:
        return _c("shim_rebind_patch", "info", "no installed shim (libgreenboost_cuda.so)")
    if marker and marker in installed:
        return _c("shim_rebind_patch", "ok", "installed shim carries the cudart-rebind force patch (marker found)")
    if repo_build is not None and repo_build == installed and repo_source and re.search(r"gb_cudart_resolve_syms\(void \*libcudart, int force\)", repo_source):
        return _c("shim_rebind_patch", "ok", "installed shim is the checkout's build of the patched source (built before the marker existed)")
    return _c("shim_rebind_patch", "warn", "installed shim has no cudart-rebind force marker: a mixed cu12/cu13 process keeps calling the wrong runtime (NaN, SM count 0)", fix)


def judge_engine_selected(active: str, moecache_built: bool, choice: str, moe_cache_flag: bool) -> dict:
    if active == "moecache" and moe_cache_flag:
        return _c("expert_cache_engine", "ok", "expert-cache engine built, selected and accepts --moe-cache-mib")
    if moecache_built and active != "moecache":
        return _c("expert_cache_engine", "warn", f"expert-cache engine is built but the active engine is '{active}' (choice file: {choice})", "sudo greenboost synapse use-engine moecache")
    if not moecache_built:
        return _c("expert_cache_engine", "warn", "expert-cache engine is not built: MoE models keep the layer-split speed (about half of the cached decode)", "sudo greenboost synapse build-moecache-engine && sudo greenboost synapse use-engine moecache")
    return _c("expert_cache_engine", "warn", "selected engine does not accept --moe-cache-mib (wrong commit?)", "sudo greenboost synapse build-moecache-engine")


def judge_thp(enabled: str, defrag: str, pmd_bytes: int, nr_hugepages: int, t2_pool_mb: int) -> List[dict]:
    """THP mode and the T2 pool alignment. The module allocates the pool in 2 MiB order-9 pages straight from the buddy allocator, so the pool must be a multiple of the PMD size;
    THP mode only changes how the OTHER pinned/host buffers (llama.cpp, torch) are backed."""
    out = []
    mode = (re.search(r"\[(\w+)\]", enabled) or [None, enabled.strip()])[1]
    if mode == "never":
        out.append(_c("thp_mode", "warn", "transparent hugepages are off: host buffers use 4 KiB pages (more TLB misses on expert reads and pinned staging)", "echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled"))
    else:
        out.append(_c("thp_mode", "ok" if mode in ("always", "madvise") else "info", f"THP enabled={mode}, defrag={(re.search(r'\\[(\\w+)\\]', defrag) or [None, defrag.strip()])[1]}"))
    if nr_hugepages:
        out.append(_c("hugetlb_pool", "warn", f"{nr_hugepages} HugeTLB pages are reserved: that RAM is locked away from T2", "echo 0 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages"))
    if t2_pool_mb and pmd_bytes:
        pmd_mb = pmd_bytes / 1048576
        out.append(_c("t2_pool_alignment", "ok" if t2_pool_mb % pmd_mb == 0 else "warn",
                      f"T2 pool {t2_pool_mb} MiB is {'a multiple' if t2_pool_mb % pmd_mb == 0 else 'NOT a multiple'} of the {pmd_mb:g} MiB hugepage",
                      "" if t2_pool_mb % pmd_mb == 0 else "sudo greenboost tune   # resize the pool to a hugepage multiple"))
    return out


def judge_helper(socket_present: bool, connectable: bool, installer_present: bool) -> dict:
    if socket_present and connectable:
        return _c("root_helper", "ok", "sudo-less root helper answers on its socket")
    if socket_present:
        return _c("root_helper", "warn", "root helper socket exists but nothing answers", "sudo systemctl restart greenboost-root-helper.socket")
    if installer_present:
        return _c("root_helper", "info", "sudo-less root helper not installed (privileged actions need sudo or the supervisor)", "privsplit/install-privsplit.sh.template --dry-run   # read it, then run it with sudo")
    return _c("root_helper", "info", "sudo-less root helper not installed and its installer is not in this tree")


def tree_hashes(root: Path) -> dict:
    out = {}
    for p in sorted(Path(root).rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def judge_cli_shadow(installed_hashes: Optional[dict], repo_hashes: Optional[dict], installed_dir: str = "", repo_dir: str = "", editable: bool = False) -> dict:
    """A real copy of greenboost_cli in the cli venv's site-packages shadows the checkout: edits in the repo never reach `gb`. An editable install (a .pth to the checkout) is correct."""
    if editable:
        return _c("cli_venv_shadow", "ok", "cli venv imports greenboost_cli from the checkout (editable install)")
    if installed_hashes is None:
        return _c("cli_venv_shadow", "info", "no greenboost_cli copy in the cli venv")
    if repo_hashes is None:
        return _c("cli_venv_shadow", "info", f"cli venv holds a copy of greenboost_cli ({installed_dir}) and the checkout was not found (set GB_REPO_DIR)")
    diff = sorted(k for k in set(installed_hashes) | set(repo_hashes) if installed_hashes.get(k) != repo_hashes.get(k))
    if not diff:
        return _c("cli_venv_shadow", "ok", "installed greenboost_cli copy matches the checkout")
    return _c("cli_venv_shadow", "warn", f"the cli venv's greenboost_cli copy differs from the repo in {len(diff)} file(s) (e.g. {', '.join(diff[:3])}): `gb` runs the OLD code",
              f"/usr/local/lib/greenboost/cli-venv/bin/pip install --no-deps -e {repo_dir or '<repo>'}/greenboost-cli")


# ---- readers -------------------------------------------------------------------------------------------------------------------------------------------
def _read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def run_all(repo_dir: Optional[str] = None) -> List[dict]:
    """All checks on this host (read-only). `repo_dir` or GB_REPO_DIR names the checkout for the shadow/shim comparisons."""
    import gb_synapse
    out: List[dict] = []
    here = Path(__file__).resolve().parent
    repo = Path(repo_dir or os.environ.get("GB_REPO_DIR", "") or (here if (here / "greenboost-cli").is_dir() else "") or ".")
    repo = repo if (repo / "greenboost-cli").is_dir() else None

    eng = gb_synapse.engine_summary()
    runtime, nvcc = gb_synapse.engine_cuda_majors(Path(eng["dir"]))
    out.append(judge_engine_runtime(runtime, nvcc, gb_synapse.engine_installed()))

    shim = Path("/usr/local/lib/libgreenboost_cuda.so")
    rb = (repo / "libgreenboost_cuda.so") if repo else None
    src = (repo / "greenboost_cuda_shim.c") if repo else None
    out.append(judge_shim_rebind(shim.read_bytes() if shim.is_file() else None, b"GB-MARKER cudart-rebind-force=1",
                                 rb.read_bytes() if rb and rb.is_file() else None, src.read_text() if src and src.is_file() else None))

    out.append(judge_engine_selected(eng["active"], eng["moecache_built"], eng["choice"], eng["moe_cache_flag"]))

    t2 = 0
    try:
        import gb_monitor
        t2 = int(gb_monitor.snapshot(probe_gpu=False).t2_pool_mb or 0)
    except Exception:  # noqa: BLE001
        pass
    th = "/sys/kernel/mm/transparent_hugepage/"
    out += judge_thp(_read(th + "enabled"), _read(th + "defrag"), int(_read(th + "hpage_pmd_size") or 0), int(_read("/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages") or 0), t2)

    sock_path = os.environ.get("GB_ROOT_HELPER_SOCKET", "/run/greenboost-helper/root-helper.sock")
    present, ok = os.path.exists(sock_path), False
    if present:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect(sock_path)
                ok = True
        except OSError:
            ok = False
    out.append(judge_helper(present, ok, bool(repo and (repo / "privsplit" / "install-privsplit.sh.template").is_file())))

    venv = Path("/usr/local/lib/greenboost/cli-venv")
    sp = next(iter(sorted(venv.glob("lib/python3*/site-packages"))), None)
    if sp is not None:
        copy = sp / "greenboost_cli"
        editable = not copy.is_dir() and any(sp.glob("_editable_impl_greenboost_cli.pth"))
        out.append(judge_cli_shadow(tree_hashes(copy) if copy.is_dir() else None, tree_hashes(repo / "greenboost-cli" / "greenboost_cli") if (repo and copy.is_dir()) else None,
                                    str(copy), str(repo or ""), editable))
    return out


def format_lines(checks: List[dict]) -> List[str]:
    tag = {"ok": "ok  ", "warn": "WARN", "info": "info"}
    lines = []
    for c in checks:
        lines.append(f"Check {tag.get(c['status'], c['status'])}: {c['id']}: {c['detail']}")
        if c["fix"]:
            lines.append(f"      fix: {c['fix']}")
    return lines

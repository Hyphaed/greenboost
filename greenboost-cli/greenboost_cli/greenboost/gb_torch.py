"""
GreenBoost + PyTorch memory compatibility.

Call apply_gb_torch_env() before any torch.cuda import:

    from greenboost_cli.greenboost.gb_torch import apply_gb_torch_env
    apply_gb_torch_env()
    import torch  # safe now

Root cause: GreenBoost intercepts cudaMalloc for T1→T2 overflow routing but
does NOT intercept cuMemMap (the VMM extension path). With expandable_segments:True,
PyTorch uses cuMemMap which bypasses GreenBoost. Forcing False makes PyTorch use
plain cudaMalloc — GreenBoost intercepts each allocation and routes overflow to T2.
"""
import os
import sys
from pathlib import Path

_GB_SYSFS = Path("/sys/class/greenboost/greenboost/status")
_GB_DEV   = "/dev/greenboost"

# Candidate directories for the greenboost source tree (contains gb_attn.py)
_GB_SRC_CANDIDATES = [
    Path(__file__).parent.parent.parent.parent / "greenboost",  # sibling repo layout
    Path.home() / "Dev" / "greenboost_all" / "greenboost",
]


def apply_gb_torch_env(diffusion: bool = False) -> None:
    """Set PYTORCH_CUDA_ALLOC_CONF and GreenBoost env vars before CUDA initialises.

    diffusion=True: also disable phase detection and zero out
    GREENBOOST_KV_RESERVE_MB — diffusion models have no LLM-style KV cache, so
    GB's phase detector misclassifies their activation buffers as KV cache, and
    reserving T1 VRAM for a KV cache that will never exist is wasteful.

    Both env vars are diffusion-scoped ONLY. Incident 2026-08-02: this function
    used to set GREENBOOST_PHASE_DETECT=0 unconditionally (`setdefault`, not
    gated on `diffusion`), called at CLI import time (`__main__.py`), so it
    leaked into every subprocess the CLI's os.environ propagated to — including
    gb-synapse's `llama-server` LLM serves. With phase detection off, the shim
    never advances past GB_PHASE_INIT (greenboost_cuda_shim.c:2933), so the KV
    reserve never collapses (:5975/:6019) and KV is never pinned in T1 (:6105):
    measured live, 3.4 GB of VRAM stranded by a dead reserve, 10.2 GB of
    weights spilled to T2 DDR that Rule #1 says belongs in VRAM, VRAM fill
    72.9% instead of ~90%, decode dropping to 2.6-4.3 tok/s. LLM serving
    NEEDS phase detection on — do not widen this gate again without re-reading
    that incident.
    """
    # Force plain cudaMalloc so GreenBoost can intercept T1→T2 overflow.
    # garbage_collection_threshold:0.8 reclaims cached memory at 80% capacity.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:False,garbage_collection_threshold:0.8"
    )
    if diffusion:
        # GB's phase detector misclassifies diffusion activation buffers as KV cache.
        os.environ.setdefault("GREENBOOST_PHASE_DETECT", "0")
        os.environ.setdefault("GREENBOOST_KV_RESERVE_MB", "0")


def _dev_accessible() -> bool:
    try:
        fd = os.open(_GB_DEV, os.O_RDONLY | os.O_NONBLOCK)
        os.close(fd)
        return True
    except OSError:
        return False


def active() -> bool:
    """True if the GreenBoost shim is loaded and active for this process."""
    return (
        os.environ.get("GREENBOOST_ACTIVE") == "1"
        and _GB_SYSFS.exists()
        and _dev_accessible()
    )


def t2_available_mb() -> int:
    """MB of T2 RAM available for overflow (0 if GreenBoost not functional)."""
    if not _GB_SYSFS.exists() or not _dev_accessible():
        return 0
    try:
        for line in _GB_SYSFS.read_text().splitlines():
            if "T2 available" in line:
                return int(line.split(":")[1].strip().split()[0])
    except Exception:
        pass
    return 0


def load_gb_attn():
    """Return the gb_attn module from the greenboost source tree, or None.

    Tries package import first, then walks candidate directories for the
    development mono-repo layout (greenboost/ alongside greenboost-cli/).
    """
    try:
        import gb_attn
        return gb_attn
    except ImportError:
        pass
    for candidate in _GB_SRC_CANDIDATES:
        if (candidate / "gb_attn.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            try:
                import gb_attn  # noqa: F811
                return gb_attn
            except ImportError:
                continue
    return None


def check() -> None:
    """Print a one-line GreenBoost status summary."""
    if not active():
        print("GreenBoost: NOT active (run via factory.sh or set GREENBOOST_ACTIVE=1 LD_PRELOAD=...)")
        return
    mb = t2_available_mb()
    print(f"GreenBoost: active — T2 available {mb:,} MB ({mb // 1024} GB)")

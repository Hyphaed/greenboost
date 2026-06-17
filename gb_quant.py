"""
gb_quant.py — GreenBoost weight-quantization companion (quality-first strategy).

The project-agnostic Python complement to gb_attn.py.  Where gb_attn.py
compresses the *KV cache* (TurboQuant: rotation + Lloyd-Max + 1-bit QJL),
gb_quant.py compresses *model weights* to maximize effective quality subject to
VRAM+DDR budgets — NOT just "fit to VRAM".

Core insight:  T2 DDR as a quality reservoir
    On Blackwell (RTX 5070), T2 DDR costs only ~1.3× the BF16 latency for
    compute-bound diffusion (Ch. G.5 of the GreenBoost extension doc).  This means
    quality-critical layers can stay at BF16 in T2 instead of being crushed to int4
    — you get near-lossless inference at the cost of ~1.3× wall-clock, not ~0.3×
    quality.  The shim routes allocations that don't fit T1 to T2 transparently.

Three quality tiers:
    near_lossless (default)
        Per-layer calibrated sensitivity; layers where quant error > 0.5% stay
        at BF16 or FP8; bulk FFN/MLP gets nvfp4/int4.  Shim ON if any BF16 layers
        overflow T1.  Behaves as-if BF16 to the human eye.
    balanced
        Moderate target (< 2% per-layer error); most layers at fp8/nvfp4.
        Usually fits T1 alone → shimless.
    compact
        Footprint-first (today's original behaviour) — scalar bits floor,
        component-granular planning via plan_fit.  Shimless, fastest throughput.

GPU family abstraction:
    gpu_profile() detects the architecture (Blackwell/Hopper/Ada/Ampere/Turing)
    and returns the ordered precision ladder available on that GPU.  nvfp4 is only
    on Blackwell (sm_12x); fp8 tensor cores are on Ada/Hopper/Blackwell only.

Execution backends (unchanged):
    - GemLite (Triton low-bit GEMM, MXFP4/NVFP4 on Blackwell sm_120) for
      int8/int4/fp8/nvfp4 — vendored under third_party/ and importable here.
    - gb_quant_tq (TurboQuant rotation + Lloyd-Max + Triton LUT-GEMM) for the
      sub-4-bit modes "tq3"/"tq2".

Public API:
    import gb_quant

    # Quality-first (new default): per-layer calibrated mixed precision
    report = gb_quant.quantize_for_quality(model.transformer,
                                            target="near_lossless",
                                            t1_budget_gb=11.0, t2_budget_gb=8.0)
    print(report)        # per-layer bits, T1/T2 split, shim_required flag

    # Pipeline helpers (quality-aware by default):
    gb_quant.quantize_encoders(pipe, quality="near_lossless")
    gb_quant.quantize_denoiser(pipe, quality="near_lossless")

    # Legacy scalar precision (compact/speed path):
    gb_quant.quantize_module(model.transformer, bits=4)
    report = gb_quant.quantize_to_fit(pipe, budget_gb=11.0)
"""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

# GreenBoost layer bootstrap — patches empty_cache, starts telemetry singleton,
# wires stream scheduler + tier manager + mem pools.  No-op when not active.
try:
    import gb_init as _gb_init
    _GB_ACTIVE = _gb_init.ACTIVE
except ImportError:
    _gb_init = None
    _GB_ACTIVE = os.environ.get("GREENBOOST_ACTIVE") == "1"


def _gb_cache_release():
    """Drop caching allocator cache — no-op under GreenBoost DynamicVRAM."""
    if not _GB_ACTIVE and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _gb_nvtx(msg: str, color: str = "cyan"):
    """Return NVTX context manager if stream sched available, else nullcontext."""
    import contextlib
    _gs = _gb_init.get_stream_sched() if _gb_init else None
    if _gs is not None:
        try:
            import nvtx
            return nvtx.annotate(message=msg, color=color, domain="GreenBoost")
        except Exception:
            pass
    return contextlib.nullcontext()


# The low-bit GEMM backend is PART OF GREENBOOST: ported into
# greenboost/third_party/ (Triton GEMM kernels originally from GemLite,
# Apache-2.0, plus the hqq quantizer). gb_quant CARRIES its backend —
# consumer venvs (LTX-2, ai-forge envs, artpipeline, ...) install nothing.
# Only runtime deps (numpy, tqdm, triton) ship with every torch venv.
# Search order per module: env override > greenboost/third_party (canonical)
# > ~/Dev/turboquantsolutions (upstream development workbench).
def _ensure_vendored_paths() -> None:
    """Make the backend (`gemlite` kernels, `hqq` quantizer) importable from
    the greenboost tree whenever the running venv doesn't provide them (hqq
    is imported lazily at quantize time, so its path is ensured up front)."""
    import importlib.util
    import os as _os
    import sys as _sys
    _gb_third = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                              "third_party")
    _tq_root = _os.path.expanduser("~/Dev/turboquantsolutions")
    locations = {
        "gemlite": [_os.environ.get("GB_GEMLITE_PATH"), _gb_third,
                    _os.path.join(_tq_root, "gemlite")],
        "hqq": [_os.environ.get("GB_THIRD_PARTY_PATH"), _gb_third,
                _os.path.join(_tq_root, "third_party")],
    }
    for mod, paths in locations.items():
        try:
            present = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            present = False
        if present:
            continue
        for path in paths:
            if path and _os.path.isdir(_os.path.join(path, mod)) \
                    and path not in _sys.path:
                _sys.path.insert(0, path)
                break


_GEMLITE_ERR = None
try:
    _ensure_vendored_paths()
    from gemlite.helper import (
        patch_model as _gl_patch_model,
        A16W4_HQQ_INT as _A16W4,
        A16W4_NVFP as _A16W4_NVFP,
        A16W8_INT8 as _A16W8,
        A16W8_FP8 as _A16W8_FP8,          # GemLite FP8 e4m3fn weight-only
        A16Wn as _A16Wn,
    )
except Exception as e:  # pragma: no cover - environment dependent
    _GEMLITE_ERR = e


# Bytes-per-parameter for each supported weight precision (incl. ~5% scale/zero
# overhead for grouped int4, channel-wise int8/fp8; TQ modes carry one fp16
# block norm per 128 params: bits/8 + 2/128).
_BYTES_PER_PARAM = {
    16: 2.0,
    "fp8": 1.05,   # FP8 e4m3fn, channel-wise — same storage as INT8
    "e4m3": 1.05,  # alias
    8: 1.05,       # INT8
    4: 0.55,
    "tq3": 0.40,
    "tq2": 0.27,
}

# Precision ladder, highest quality first.  "fp8" sits between BF16 and INT8
# (same bit-width, but FP representation preserves outliers better).
# "nvfp4" is Blackwell NVFP4 (sm_120 PTX FP4, group_size=16, FP8 scales).
# "tq3"/"tq2" are the TurboQuant weight modes.
_PRECISION_LADDER = (16, "fp8", "nvfp4", 8, 4, "tq3", "tq2")


# ---------------------------------------------------------------------------
# GPU family abstraction
# ---------------------------------------------------------------------------

@dataclass
class GpuProfile:
    """Per-GPU-family precision capabilities and quality-tier defaults."""
    family: str                        # "blackwell","hopper","ada","ampere","turing","volta"
    cc: Tuple[int, int]               # compute capability e.g. (12, 0)
    precisions: Tuple                  # ordered highest-quality-first (usable on this GPU)
    floor_default: Union[int, str]    # compact-tier precision floor (e.g. 4 or "nvfp4")
    quality_default: Union[int, str]  # best non-BF16 precision (fp8/nvfp4)
    t2_tolerance_gb: float            # GB of T2 DDR the quality tier may reserve


# Family table: (cc_min, cc_max) → (family, precisions, floor, quality_default, t2_tol).
# Precision sets are ordered highest-quality-first and include only precisions that have
# hardware acceleration on that family.  fp8 tensor cores: Ada(8.9)/Hopper(9.x)/Blackwell(12.x).
# nvfp4: Blackwell only (sm_120 PTX FP4).
_GPU_FAMILIES: List[Tuple] = [
    # (cc_min, cc_max,   family,      precisions,                            floor,  q_def,    t2_tol)
    ((12, 0), (12, 99), "blackwell", (16, "fp8", "nvfp4", 8, 4, "tq3", "tq2"), 4, "nvfp4",  8.0),
    (( 9, 0), ( 9, 99), "hopper",   (16, "fp8", 8, 4, "tq3", "tq2"),           4, "fp8",    8.0),
    (( 8, 9), ( 8, 99), "ada",      (16, "fp8", 8, 4, "tq3", "tq2"),           4, "fp8",    6.0),
    (( 8, 0), ( 8,  8), "ampere",   (16, 8, 4, "tq3", "tq2"),                  4,  8,       6.0),
    (( 7, 5), ( 7, 99), "turing",   (16, 8, 4, "tq3", "tq2"),                  4,  8,       4.0),
    (( 7, 0), ( 7,  4), "volta",    (16, 8, 4, "tq3", "tq2"),                  4,  8,       4.0),
]

_GPU_PROFILE_CACHE: Optional[GpuProfile] = None


def gpu_profile(device: int = 0) -> GpuProfile:
    """Return the GpuProfile for `device` (cached after the first call).

    Falls back to an Ampere-compatible profile when CUDA is not available so
    callers can unit-test plan_quality without a GPU.
    """
    global _GPU_PROFILE_CACHE
    if _GPU_PROFILE_CACHE is not None:
        return _GPU_PROFILE_CACHE

    if not torch.cuda.is_available():
        # CPU fallback: conservative Ampere-equivalent
        _GPU_PROFILE_CACHE = GpuProfile(
            family="cpu_fallback", cc=(8, 0),
            precisions=(16, 8, 4, "tq3", "tq2"),
            floor_default=4, quality_default=8, t2_tolerance_gb=0.0,
        )
        return _GPU_PROFILE_CACHE

    cc = torch.cuda.get_device_capability(device)
    for (cc_min, cc_max, family, precisions, floor, q_def, t2_tol) in _GPU_FAMILIES:
        if cc_min <= cc <= cc_max:
            _GPU_PROFILE_CACHE = GpuProfile(
                family=family, cc=cc, precisions=precisions,
                floor_default=floor, quality_default=q_def,
                t2_tolerance_gb=t2_tol,
            )
            return _GPU_PROFILE_CACHE

    # Unknown/future architecture: conservative safe fallback
    _GPU_PROFILE_CACHE = GpuProfile(
        family="unknown", cc=cc,
        precisions=(16, 8, 4, "tq3", "tq2"),
        floor_default=4, quality_default=8, t2_tolerance_gb=4.0,
    )
    return _GPU_PROFILE_CACHE


# ---------------------------------------------------------------------------
# Quality-tier configuration
# ---------------------------------------------------------------------------

# Per-tier maximum relative quantization error allowed per layer.
# rel_err = ‖W − Wq‖_F / ‖W‖_F (from gb_quant_calib.relative_quant_error).
#
# Empirical fp8 e4m3fn per-row error on FLUX/T5 weights: ~2.5-2.7% (all layers).
# This is physics of 3-bit mantissa, not a calibration failure — fp8 is still
# perceptually identical to BF16 for diffusion models.
# near_lossless = 3.0% keeps everything at fp8; BF16 T2 reservoir is only
# used for genuinely pathological layers (>3% fp8 error).
# balanced = 8.0% allows nvfp4/int4 where fp8 is overkill.
_TARGET_ERROR_CEIL: Dict[str, float] = {
    "near_lossless": 0.030,   # 3.0%  — fp8 qualifies (fp8≈2.6%); BF16 only for outliers
    "balanced":      0.080,   # 8.0%  — nvfp4/int4 qualify for most layers
    "compact":       1.000,   # unlimited — footprint-first (legacy behaviour)
}

QUALITY_TIERS = tuple(_TARGET_ERROR_CEIL.keys())


def _bits_tag(bits) -> str:
    if bits == 16:
        return "bf16"
    if bits in ("fp8", "e4m3"):
        return "fp8"
    return f"int{bits}" if isinstance(bits, int) else str(bits)

# Components we never quantize (small, precision-sensitive, or non-Linear-heavy).
_DEFAULT_SKIP_COMPONENTS = ("vae", "scheduler", "image_processor", "feature_extractor")
# Linear sub-modules to leave at full precision inside a quantized component.
_DEFAULT_SKIP_MODULES = ("lm_head", "vision", "visual", "embed", "norm", "proj_out")


@dataclass
class ComponentPlan:
    name: str
    params: int
    bf16_gb: float
    bits: "int | str"         # 16 = left unquantized; "tq3"/"tq2" = TurboQuant
    quant_gb: float


@dataclass
class FitReport:
    budget_gb: float
    components: list[ComponentPlan] = field(default_factory=list)
    total_bf16_gb: float = 0.0
    total_quant_gb: float = 0.0

    @property
    def fits_vram(self) -> bool:
        return self.total_quant_gb <= self.budget_gb

    @property
    def needs_t2_overflow_gb(self) -> float:
        return max(0.0, self.total_quant_gb - self.budget_gb)

    def __str__(self) -> str:
        lines = [f"[gb_quant] fit-to-VRAM plan (budget {self.budget_gb:.1f} GiB)"]
        for c in self.components:
            lines.append(f"    {c.name:<14} {c.bf16_gb:6.1f} GiB bf16 -> "
                         f"{c.quant_gb:5.1f} GiB {_bits_tag(c.bits)}")
        verdict = ("fits T1 VRAM, no overflow" if self.fits_vram
                   else f"needs {self.needs_t2_overflow_gb:.1f} GiB T2 DDR overflow")
        lines.append(f"    total {self.total_bf16_gb:.1f} -> "
                     f"{self.total_quant_gb:.1f} GiB  [{verdict}]")
        return "\n".join(lines)


@dataclass
class QualityFitReport:
    """Per-layer bit-assignment result from plan_quality().

    Unlike FitReport (component-granular), this report is layer-granular and
    tracks the T1/T2 split and whether the shim must be active for inference.
    """
    target: str                            # "near_lossless" | "balanced" | "compact"
    per_layer_bits: Dict[str, object]      # {layer_name: bits}
    t1_estimated_gb: float                 # layers that fit in T1 VRAM
    t2_estimated_gb: float                 # layers that overflow to T2 DDR via shim
    shim_required: bool                    # True → _gen_gb.sh must enable LD_PRELOAD
    mean_rel_err: float                    # mean calibrated error across layers
    max_rel_err: float                     # worst-case calibrated error
    # breakdown: {bits: count_of_layers}
    precision_histogram: Dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        hist_str = "  ".join(
            f"{_bits_tag(k)}×{v}" for k, v in sorted(
                self.precision_histogram.items(),
                key=lambda kv: _BYTES_PER_PARAM.get(kv[0], 0), reverse=True
            )
        )
        shim_note = "shim ON (T2 reservoir)" if self.shim_required else "shimless"
        return (
            f"[gb_quant] quality plan: target={self.target}  {shim_note}\n"
            f"    T1 {self.t1_estimated_gb:.1f} GiB  T2 {self.t2_estimated_gb:.1f} GiB  "
            f"({len(self.per_layer_bits)} layers: {hist_str})\n"
            f"    mean_err={self.mean_rel_err:.4f}  max_err={self.max_rel_err:.4f}"
        )


def _require_gemlite() -> None:
    if _GEMLITE_ERR is not None:
        raise RuntimeError(
            "gb_quant needs GemLite (the low-bit GEMM backend). Install it with "
            "`pip install gemlite` or `pip install -e "
            "~/Dev/turboquantsolutions/gemlite`. Original error: "
            f"{_GEMLITE_ERR!r}"
        )


# ---------------------------------------------------------------------------
# Warm-kernel cache: persist the backend's Triton autotune picks so the ~28 s
# first-call autotune happens once per GPU, not once per process.  The cache
# file is per-device-name; gemlite's cache_config() merges with any existing
# file, so concurrent sessions only ever add entries.
# Opt out with GB_QUANT_NO_AUTOTUNE_CACHE=1; relocate with GB_QUANT_CACHE_DIR.
# ---------------------------------------------------------------------------
_autotune_cache_armed = False


def _autotune_cache_path() -> "str | None":
    if os.environ.get("GB_QUANT_NO_AUTOTUNE_CACHE", "") == "1":
        return None
    if not torch.cuda.is_available():
        return None
    cache_dir = os.path.expanduser(
        os.environ.get("GB_QUANT_CACHE_DIR", "~/.cache/greenboost"))
    dev = torch.cuda.get_device_name(0).replace(" ", "_").replace("/", "_")
    return os.path.join(cache_dir, f"gemlite_autotune_{dev}.json")


def _arm_autotune_cache() -> None:
    """Load the per-GPU autotune cache and arm the at-exit save (idempotent)."""
    global _autotune_cache_armed
    if _autotune_cache_armed or _GEMLITE_ERR is not None:
        return
    _autotune_cache_armed = True
    path = _autotune_cache_path()
    if not path:
        return
    import atexit
    import gemlite

    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        if os.path.isfile(path) and gemlite.load_config(path, print_error=False):
            print(f"[gb_quant] warm-kernel cache loaded: {path}")
    except Exception as e:
        print(f"[gb_quant] warm-kernel cache load skipped ({e!r})")

    def _save() -> None:
        try:
            gemlite.cache_config(path)
        except Exception:
            pass

    atexit.register(_save)


def _module_param_count(module: "torch.nn.Module") -> int:
    return sum(p.numel() for p in module.parameters())


def _iter_named_components(obj):
    """Yield (name, module) for the quantizable sub-models of a diffusers pipeline
    or, if `obj` is a plain nn.Module, just ('model', obj)."""
    import torch.nn as nn
    if isinstance(obj, nn.Module):
        yield "model", obj
        return
    # diffusers pipeline: components live as attributes registered in config
    names = getattr(obj, "components", None)
    if isinstance(names, dict):
        items = names.items()
    else:
        items = ((n, getattr(obj, n, None)) for n in dir(obj))
    for name, comp in items:
        if isinstance(comp, nn.Module):
            yield name, comp


def _init_nvfp4() -> None:
    """Set up the Blackwell PTX path and fast-mode flag for nvfp4 (idempotent)."""
    import gemlite
    if "TRITON_PTXAS_BLACKWELL_PATH" not in os.environ:
        for _ptxas in ("/usr/local/cuda-13.0/bin/ptxas", "/usr/local/cuda-13/bin/ptxas"):
            if os.path.isfile(_ptxas):
                os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = _ptxas
                break
    if gemlite.auto_detect_ptx_fp4_pack():
        print("[gb_quant] nvfp4: hardware PTX FP4 packing enabled")
    gemlite.set_fast_nvfp4(True)


def _build_processor(bits, device: str, dtype, group_size: int):
    """Build (and return) the GemLite/HQQ processor for `bits`.

    Returns None for bits=16 (BF16 — no quantization).
    Raises ValueError for unsupported values.
    """
    if bits == 16:
        return None
    if bits in (3, 2, "tq3", "tq2"):
        from gb_quant_tq import A16Wtq
        nb = int(str(bits).lstrip("tq"))
        return A16Wtq(nbits=nb, device=device, dtype=dtype,
                      fallback=_A16W4(device=device, dtype=dtype))
    if bits == 4:
        return _A16W4(device=device, dtype=dtype)
    if bits in ("fp8", "e4m3"):
        return _A16W8_FP8(device=device, dtype=dtype)
    if bits == 8:
        return _A16W8(device=device, dtype=dtype)
    if bits == "nvfp4":
        _init_nvfp4()
        return _A16W4_NVFP(device=device, dtype=dtype)
    raise ValueError(f"unsupported bits={bits!r}")


def _delegate_patch(module, processor, skip_modules, group_size, device,
                    per_layer_bits: "Optional[Dict[str, object]]" = None,
                    processors_by_bits: "Optional[Dict]" = None):
    """Quantize every nn.Linear in `module` via `processor`, IN PLACE.

    Unlike the backend's patch_model (which swaps child modules with setattr),
    the original Linear OBJECT stays alive wherever it is referenced and its
    forward is rebound to the quantized implementation, registered as its
    child.  This is the only scheme that survives out-of-tree references:
    LTX-2's TransformerArgsPreprocessor is a plain Python class holding
    `self.patchify_proj = <the same Linear>` - invisible to any module-tree
    walk, so a swap leaves it calling the weight-stripped original
    (F.linear(weight=None)).  Moves layer-by-layer: peak memory stays bounded.

    per_layer_bits:   if provided, maps each layer name to its bits precision.
                      Layers with bits=16 are kept at BF16 (no quantization).
                      Layers missing from the dict use `processor` (scalar fallback).
    processors_by_bits: pre-built {bits: processor} dict for per-layer mode.
                        Built lazily if not provided.
    """
    from hqq.core.quantize import HQQLinear, BaseQuantizeConfig  # vendored

    dtype = getattr(module, "dtype", torch.bfloat16) or torch.bfloat16
    _proc_cache: Dict = dict(processors_by_bits or {})

    seen = set()
    linears = [(n, l) for n, l in module.named_modules()
               if isinstance(l, torch.nn.Linear)]
    for name, layer in linears:
        if id(layer) in seen:
            continue
        seen.add(id(layer))
        layer.name = name                      # parity with backend patch_model
        layer.to(device=device, non_blocking=True)
        if any(s in name for s in skip_modules):
            continue
        # GemLite MIN_SIZE=32: in_features must be divisible by 32 AND by
        # group_size (for 4-bit). Tiny layers (e.g. in_features=16 gate
        # projections in FLUX) stay full-precision — negligible quality loss.
        in_f = layer.in_features
        if in_f % 32 != 0:
            continue

        # Determine this layer's bits and processor.
        if per_layer_bits is not None:
            bits = per_layer_bits.get(name)
            if bits is None or bits == 16:
                # Not in plan or kept BF16 — skip quantization.
                continue
            gs = group_size if bits == 4 else None
            if gs is not None and in_f % gs != 0:
                continue
            proc = _proc_cache.get(bits)
            if proc is None:
                proc = _build_processor(bits, device, layer.weight.dtype, group_size)
                _proc_cache[bits] = proc
            if proc is None:
                continue  # bits==16 shouldn't reach here, but guard
        else:
            # Scalar-bits (legacy) path.
            if group_size is not None and in_f % group_size != 0:
                continue
            proc = processor
            gs = group_size

        if hasattr(proc, "from_hqqlinear"):
            nb = proc.W_nbits
            cfg = BaseQuantizeConfig(nbits=nb, group_size=gs if nb <= 4 else None)
            impl = proc.from_hqqlinear(
                HQQLinear(layer, quant_config=cfg,
                          compute_dtype=layer.weight.dtype, device=device))
        else:
            impl = proc.from_linear(layer)
        # The original keeps no big tensors (cleanup_linear nulled them) and
        # becomes a thin delegate.  add_module so .to()/device moves reach the
        # quantized buffers through every retained reference.
        for pname in ("weight", "bias"):
            if pname in layer._parameters:
                layer._parameters[pname] = None
        layer.add_module("_gb_impl", impl)
        layer.forward = impl.forward
    module.to(device=device)


def quantize_module(module, bits, device: str = "cuda",
                    dtype=None, group_size: int = 64,
                    skip_modules=_DEFAULT_SKIP_MODULES,
                    per_layer_bits: "Optional[Dict[str, object]]" = None):
    """Quantize every Linear in `module` to `bits` (4, 8, "fp8"/"e4m3",
    "nvfp4", "tq3" or "tq2") moving layer-by-layer to `device` (bounded peak
    memory — never loads the full BF16 module onto the GPU).
    4/8/fp8/nvfp4 execute on GemLite Triton kernels; "tq3"/"tq2" (also plain
    3/2) execute on the TurboQuant LUT-GEMM backend (gb_quant_tq) with
    int4-HQQ fallback for non-128-multiple layers.

    bits="fp8" / bits="e4m3" — GemLite FP8 e4m3fn, channel-wise (no
    group_size).  Best choice for max-quality shimless VRAM-fit inference:
    same footprint as INT8 (~1.05 B/param) but preserves dynamic range better
    for attention projections.

    per_layer_bits: optional {layer_name: bits} dict for per-layer precision.
        When provided, `bits` is used only as the default for layers not in the
        dict.  This is the hook used by quantize_for_quality().
    """
    _require_gemlite()
    _arm_autotune_cache()
    dtype = dtype or torch.bfloat16

    if per_layer_bits is not None:
        # Per-layer quality path: build processors lazily in _delegate_patch.
        with _gb_nvtx("gb:quantize:per_layer", color="cyan"):
            _delegate_patch(module, None, list(skip_modules), group_size, device,
                            per_layer_bits=per_layer_bits)
        gc.collect()
        _gb_cache_release()
        return module

    # Scalar-bits path (legacy / compact tier).
    _bits_label = _bits_tag(bits)
    with _gb_nvtx(f"gb:quantize:{_bits_label}", color="cyan"):
        proc = _build_processor(bits, device, dtype, group_size)
        if proc is None:
            return module  # bits==16 → no-op
        gs = group_size if bits == 4 else None
        _delegate_patch(module, proc, list(skip_modules), gs, device)
    gc.collect()
    _gb_cache_release()
    return module


def plan_fit(obj, budget_gb: float,
             components: "tuple[str, ...] | None" = None,
             skip_components=_DEFAULT_SKIP_COMPONENTS,
             prefer_bits: "int | str" = 4) -> FitReport:
    """Decide per-component precision so the total fits `budget_gb`.

    Quality-first strategy: every component gets the HIGHEST precision that
    still leaves enough budget for the rest (bf16 > int8 > int4 > tq3 > tq2).
    The floor is `prefer_bits` (int 4/8 or "tq3"/"tq2"); the headroom check
    assumes the remaining components can drop to the floor, so precision is
    only spent where the budget truly allows it. Components in
    `skip_components` stay BF16. T3 NVMe is never planned for — anything past
    the budget relies on GreenBoost T2 DDR overflow."""
    report = FitReport(budget_gb=budget_gb)
    comps = []
    for name, comp in _iter_named_components(obj):
        if components is not None and name not in components:
            continue
        if any(s in name for s in skip_components):
            keep = True
        else:
            keep = False
        n = _module_param_count(comp)
        if n == 0:
            continue
        comps.append((name, n, keep))

    def _gb(n_params: int, bits: int) -> float:
        return n_params * _BYTES_PER_PARAM[bits] / 2**30

    # Largest first: decide the big tensors' precision before the budget is
    # nibbled away by small ones (best GB-per-quality allocation).
    comps.sort(key=lambda t: t[1], reverse=True)
    # Smallest possible footprint of everything (skip components stay bf16,
    # the rest at the floor precision) — the headroom reserve for "the rest".
    floor_total = sum(_gb(n, 16 if keep else prefer_bits)
                      for _, n, keep in comps)
    if prefer_bits not in _PRECISION_LADDER:
        raise ValueError(f"prefer_bits={prefer_bits!r} not in "
                         f"{_PRECISION_LADDER}")
    ladder = list(
        _PRECISION_LADDER[:_PRECISION_LADDER.index(prefer_bits) + 1])
    running = 0.0
    for name, n, keep in comps:
        bf16_gb = _gb(n, 16)
        floor_total -= _gb(n, 16 if keep else prefer_bits)
        if keep:
            bits = 16
        else:
            # Highest precision that still lets the remaining components fit
            # at their floor; fall back to the floor itself.
            bits = prefer_bits
            for b in ladder:
                if running + _gb(n, b) + floor_total <= budget_gb:
                    bits = b
                    break
        quant_gb = _gb(n, bits)
        running += quant_gb
        report.components.append(ComponentPlan(name, n, round(bf16_gb, 2),
                                               bits, round(quant_gb, 2)))
        report.total_bf16_gb += bf16_gb
        report.total_quant_gb += quant_gb
    report.total_bf16_gb = round(report.total_bf16_gb, 2)
    report.total_quant_gb = round(report.total_quant_gb, 2)
    return report


def quantize_to_fit(obj, budget_gb: float = 11.0, device: str = "cuda",
                    dtype=None, components: "tuple[str, ...] | None" = None,
                    group_size: int = 64, allow_t3: bool = False,
                    prefer_bits: "int | str" = 4,
                    verbose: bool = True) -> FitReport:
    """Quantize a diffusers pipeline (or nn.Module) so it fits `budget_gb` of T1
    VRAM, then realise the plan in place. Returns the FitReport.

    `allow_t3` is accepted for API symmetry but intentionally ignored here:
    gb_quant never routes weights to T3 NVMe (it would break the speed goal).
    If the quantized model still exceeds the budget, the excess relies on
    GreenBoost's T2 DDR overflow (the shim handles that transparently)."""
    _require_gemlite()
    dtype = dtype or torch.bfloat16

    # Read live VRAM from telemetry to refine budget if GB telemetry is active.
    # This uses the cached snapshot (no I/O on the hot path).
    if _gb_init is not None:
        m = _gb_init.snapshot()
        if m is not None and m.fb_free_mb > 0:
            live_budget = m.fb_free_mb / 1024.0 * 0.92
            # Use telemetry budget only when it's tighter — avoids over-spending
            # when the caller already passed an intentionally conservative value.
            if live_budget < budget_gb:
                if verbose:
                    print(
                        f"[gb_quant] telemetry: adjusting budget "
                        f"{budget_gb:.1f} → {live_budget:.1f} GiB "
                        f"({m.fb_free_mb} MB free, 92% headroom)",
                        flush=True,
                    )
                budget_gb = live_budget

    report = plan_fit(obj, budget_gb, components=components,
                      prefer_bits=prefer_bits)
    if verbose:
        print(str(report), flush=True)
    name_to_module = dict(_iter_named_components(obj))
    # Execution order matters for PEAK memory, not just final footprint:
    # patch_model briefly holds each layer in BF16 on the GPU before replacing it
    # with its int4 version. Quantizing the largest component first leaves it
    # resident while the next component's transient BF16 layers load on top —
    # which OOMs even though the final int4 footprints fit. Quantize the
    # smallest-footprint components first so the big transformer accumulates last
    # against the least resident memory (validated on FLUX.2-klein-9B: text
    # encoder int4 3.5 GiB then transformer int4 4.8 GiB peaks ~9.3 GiB on 11.5).
    quant_order = sorted(
        (p for p in report.components if p.bits != 16),
        key=lambda p: p.quant_gb,
    )
    # Quantize (smallest-footprint first) while other components stay on CPU.
    for plan in quant_order:
        if verbose:
            print(f"[gb_quant] quantizing {plan.name} -> "
                  f"{_bits_tag(plan.bits)} ...", flush=True)
        quantize_module(name_to_module[plan.name], bits=plan.bits, device=device,
                        dtype=dtype, group_size=group_size)
    # Move the small BF16-kept components (e.g. VAE) to the GPU last.
    for plan in report.components:
        if plan.bits != 16:
            continue
        comp = name_to_module.get(plan.name)
        if comp is not None and device == "cuda" and torch.cuda.is_available():
            try:
                comp.to(device)
            except Exception:
                pass
    if verbose and torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 2**30
        print(f"[gb_quant] done — VRAM in use {used:.1f} GiB "
              f"(budget {budget_gb:.1f} GiB)", flush=True)
    return report


# ---------------------------------------------------------------------------
# Quality-first per-layer planning and quantization
# ---------------------------------------------------------------------------

def plan_quality(module: "torch.nn.Module",
                 target: str = "near_lossless",
                 t1_budget_gb: float = 11.0,
                 t2_budget_gb: float = 8.0,
                 sensitivity: "Optional[Dict[str, Dict]]" = None,
                 profile: "Optional[GpuProfile]" = None,
                 group_size: int = 64,
                 skip_modules: "tuple" = _DEFAULT_SKIP_MODULES,
                 model_id: str = "model",
                 verbose: bool = True) -> QualityFitReport:
    """Build a per-layer bit-assignment that minimises quantization error
    subject to (t1_budget_gb + t2_budget_gb) total memory.

    Strategy per layer:
    - Walk the precision ladder (highest quality first) from the family's
      best non-BF16 precision upward.
    - Select the lowest precision where rel_err ≤ error_ceiling for `target`.
    - If no precision meets the ceiling → stay at BF16 (quality first).
    - Layers without sensitivity data → use profile.quality_default.
    - compact target → use profile.floor_default for all layers (footprint-first).

    T1/T2 split estimate:
    - Quantized (non-BF16) layers → assigned to T1 pool first.
    - BF16 layers and T1 overflow → assigned to T2.
    - shim_required = True when any T2 usage > 50 MB.
    """
    import torch.nn as nn

    if target not in _TARGET_ERROR_CEIL:
        raise ValueError(f"target={target!r} not in {QUALITY_TIERS}")
    error_ceil = _TARGET_ERROR_CEIL[target]
    profile = profile or gpu_profile()

    if sensitivity is None:
        # Lazy calibration — falls through to cache on second call.
        from gb_quant_calib import calibrate_sensitivity
        # Build precision set: all profile precisions except BF16.
        calib_precs = tuple(b for b in profile.precisions if b != 16)
        sensitivity = calibrate_sensitivity(
            module, precisions=calib_precs, model_id=model_id,
            group_size=group_size, verbose=verbose,
        )

    per_layer_bits: Dict[str, object] = {}
    layer_sizes: Dict[str, int] = {}
    layer_errs_used: List[float] = []

    for name, layer in module.named_modules():
        if not isinstance(layer, nn.Linear):
            continue
        if any(s in name for s in skip_modules):
            continue
        in_f = layer.in_features
        if in_f < 32 or in_f % 32 != 0:
            continue
        if layer.weight is None:
            continue

        n_params = layer.weight.numel()
        layer_sizes[name] = n_params

        if target == "compact":
            bits = profile.floor_default
        elif name not in sensitivity:
            # No sensitivity data → use family's quality default (fp8 on Blackwell).
            bits = profile.quality_default
        else:
            layer_s = sensitivity[name]
            q_default = profile.quality_default  # fp8 on Blackwell
            bits = q_default  # quality default is the floor for non-compact tiers
            # Walk the profile ladder: pick the LOWEST precision that still
            # meets the error ceiling (highest quality that isn't BF16).
            # Profile ladder is (16, "fp8", "nvfp4", 8, 4, ...) — skip 16.
            for candidate in profile.precisions:
                if candidate == 16:
                    continue
                err = layer_s.get(candidate, 1.0)
                if err <= error_ceil:
                    bits = candidate
                    break
            else:
                # No quantized precision meets the ceiling — this layer is
                # genuinely sensitive. Keep BF16 (goes to T2 reservoir).
                bits = 16
            # Record the calibrated error for the chosen bits.
            chosen_err = layer_s.get(bits, 0.0) if bits != 16 else 0.0
            layer_errs_used.append(chosen_err)

        # For INT4: check group_size divisibility.
        if bits == 4 and in_f % group_size != 0:
            # Fall back to next-higher precision in the ladder.
            ladder = list(profile.precisions)
            idx = ladder.index(4) if 4 in ladder else len(ladder)
            bits = ladder[idx - 1] if idx > 0 else 16

        per_layer_bits[name] = bits

    # T1 / T2 split estimate.
    # Priority: smallest quantized layers first → fill T1; BF16 + overflow → T2.
    t1_run = 0.0
    t2_run = 0.0
    # Sort: non-BF16 (quantized) first, by footprint ascending.
    sorted_items = sorted(
        per_layer_bits.items(),
        key=lambda kv: (kv[1] == 16, layer_sizes.get(kv[0], 0))
    )
    for name, bits in sorted_items:
        n = layer_sizes.get(name, 0)
        gb = n * _BYTES_PER_PARAM.get(bits, 2.0) / 2 ** 30
        if bits != 16 and t1_run + gb <= t1_budget_gb:
            t1_run += gb
        else:
            t2_run += gb

    # Quality metrics.
    mean_err = float(sum(layer_errs_used) / max(1, len(layer_errs_used)))
    max_err = float(max(layer_errs_used)) if layer_errs_used else 0.0

    # Precision histogram.
    hist: Dict[str, int] = {}
    for bits in per_layer_bits.values():
        k = _bits_tag(bits)
        hist[k] = hist.get(k, 0) + 1

    report = QualityFitReport(
        target=target,
        per_layer_bits=per_layer_bits,
        t1_estimated_gb=round(t1_run, 2),
        t2_estimated_gb=round(t2_run, 2),
        shim_required=t2_run > 0.05,
        mean_rel_err=round(mean_err, 5),
        max_rel_err=round(max_err, 5),
        precision_histogram=hist,
    )
    if verbose:
        print(str(report), flush=True)
    return report


def quantize_for_quality(module: "torch.nn.Module",
                          target: str = "near_lossless",
                          device: str = "cuda",
                          dtype=None,
                          group_size: int = 64,
                          t1_budget_gb: float = 11.0,
                          t2_budget_gb: float = 8.0,
                          sensitivity: "Optional[Dict[str, Dict]]" = None,
                          profile: "Optional[GpuProfile]" = None,
                          model_id: str = "model",
                          skip_modules: "tuple" = _DEFAULT_SKIP_MODULES,
                          verbose: bool = True) -> QualityFitReport:
    """Quality-first per-layer quantization for a single nn.Module.

    This is the new default path called by quantize_encoders/denoiser when
    `quality` is set.  Unlike the scalar-bits path it:
    - Runs calibrated sensitivity analysis (from disk cache on warm calls).
    - Allocates higher precision to quality-critical layers, lower to bulk FFN.
    - Uses T2 DDR as a quality reservoir for BF16 layers that overflow T1.
    - Returns a QualityFitReport detailing the per-layer assignment.

    Args:
        module:       The nn.Module to quantize (BF16, on CPU or GPU).
        target:       "near_lossless" | "balanced" | "compact".
        t1_budget_gb: T1 VRAM budget available after other resident tensors.
        t2_budget_gb: T2 DDR budget (GreenBoost shim pool).
        sensitivity:  Pre-computed calibration dict.  If None, calibrate_sensitivity
                      is called (cache miss runs CPU math, ~seconds).
        model_id:     Human-readable ID for the calibration cache filename.
    """
    _require_gemlite()
    _arm_autotune_cache()
    dtype = dtype or torch.bfloat16
    profile = profile or gpu_profile()

    if verbose:
        print(f"[gb_quant] quality={target} on {profile.family} "
              f"(T1={t1_budget_gb:.1f} GiB, T2={t2_budget_gb:.1f} GiB) …",
              flush=True)

    report = plan_quality(
        module, target=target,
        t1_budget_gb=t1_budget_gb, t2_budget_gb=t2_budget_gb,
        sensitivity=sensitivity, profile=profile,
        group_size=group_size, skip_modules=skip_modules,
        model_id=model_id, verbose=verbose,
    )

    with _gb_nvtx(f"gb:quality:{target}", color="yellow"):
        quantize_module(module, bits=profile.floor_default, device=device,
                        dtype=dtype, group_size=group_size,
                        skip_modules=skip_modules,
                        per_layer_bits=report.per_layer_bits)

    if verbose and torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 2 ** 30
        print(f"[gb_quant] quality done — VRAM {used:.1f} GiB  "
              f"(T2 {report.t2_estimated_gb:.1f} GiB via shim)",
              flush=True)
    return report


# ── Sequential encode-then-quantize orchestration ────────────────────────────
# Generic phase driver for two-stage generative pipelines (text encoder(s) +
# denoiser). The validated fit recipe for models 1.5-3x VRAM in BF16:
#
#   1. quantize_encoders()   encoder(s) -> int4 on GPU
#   2. encode_prompts()      batch-encode every prompt, embeds cached on CPU
#   3. free_encoders()       drop encoder(s), reclaim VRAM
#   4. quantize_denoiser()   transformer/unet -> int4 on GPU, move VAE
#
# or the one-call encode_then_quantize(). Project scripts (gen_art.py,
# ai-forge runners, ...) stay THIN callers — all quantization policy lives
# here so every project inherits improvements (NVFP4, TurboQuant weights, ...)
# without per-project edits.

_DEFAULT_ENCODER_ATTRS = ("text_encoder", "text_encoder_2", "text_encoder_3")
_DEFAULT_DENOISER_ATTRS = ("transformer", "unet")


def _live_modules(pipe, attr_names):
    """Yield (attr_name, module) for each attr that exists and is an nn.Module."""
    import torch.nn as nn
    for name in attr_names:
        mod = getattr(pipe, name, None)
        if isinstance(mod, nn.Module):
            yield name, mod


def quantize_encoders(pipe, bits: "int | str" = "fp8", device: str = "cuda",
                      dtype=None, encoder_attrs=_DEFAULT_ENCODER_ATTRS,
                      group_size: int = 64, verbose: bool = True,
                      quality: "Optional[str]" = "near_lossless",
                      sensitivity: "Optional[Dict]" = None,
                      model_id: str = "model",
                      t1_budget_gb: float = 11.0,
                      t2_budget_gb: float = 8.0) -> list:
    """Quantize every text encoder on `pipe` and move it to `device`.

    When `quality` is set (default "near_lossless"), uses calibrated per-layer
    precision from quantize_for_quality().  Pass quality=None and an explicit
    `bits` to use the legacy uniform-precision path.
    """
    names = []
    for name, mod in _live_modules(pipe, encoder_attrs):
        if quality is not None:
            if verbose:
                print(f"[gb_quant] quantizing {name} "
                      f"(quality={quality}) on {device} …", flush=True)
            quantize_for_quality(
                mod, target=quality, device=device, dtype=dtype,
                group_size=group_size, t1_budget_gb=t1_budget_gb,
                t2_budget_gb=t2_budget_gb, sensitivity=sensitivity,
                model_id=f"{model_id}.{name}", verbose=verbose,
            )
        else:
            if verbose:
                print(f"[gb_quant] quantizing {name} -> {_bits_tag(bits)} "
                      f"on {device} …", flush=True)
            quantize_module(mod, bits=bits, device=device, dtype=dtype,
                            group_size=group_size)
        names.append(name)
    return names


def encode_prompts(pipe, prompts, device: str = "cuda", encode_fn=None,
                   verbose: bool = True) -> list:
    """Encode `prompts` with the pipeline's (already loaded) text encoder.

    Returns a list of CPU tensors in prompt order (cached off-GPU so the
    encoder can be freed before the denoiser claims the VRAM).
    `encode_fn(pipe, prompt, device)` overrides the default
    `pipe.encode_prompt(prompt, device=device)` for exotic pipelines."""
    out = []
    n = len(prompts)
    with torch.no_grad():
        for i, prompt in enumerate(prompts, 1):
            if encode_fn is not None:
                emb = encode_fn(pipe, prompt, device)
            else:
                emb = pipe.encode_prompt(prompt, device=device)
            emb = emb[0] if isinstance(emb, (tuple, list)) else emb
            out.append(emb.to("cpu"))
            if verbose:
                print(f"[gb_quant]   encoded {i}/{n}", flush=True)
    return out


def free_encoders(pipe, encoder_attrs=_DEFAULT_ENCODER_ATTRS,
                  verbose: bool = True) -> None:
    """Drop the text encoder(s) and reclaim their VRAM/RAM."""
    for name in encoder_attrs:
        if getattr(pipe, name, None) is not None:
            if verbose:
                print(f"[gb_quant] freeing {name} ...", flush=True)
            try:
                delattr(pipe, name)
            except AttributeError:
                pass
            try:
                setattr(pipe, name, None)
            except Exception:
                pass
    gc.collect()
    _gb_cache_release()


def quantize_denoiser(pipe, bits: "int | str" = "fp8", device: str = "cuda",
                      dtype=None, denoiser_attrs=_DEFAULT_DENOISER_ATTRS,
                      move_attrs=("vae",), group_size: int = 64,
                      verbose: bool = True,
                      quality: "Optional[str]" = "near_lossless",
                      sensitivity: "Optional[Dict]" = None,
                      model_id: str = "model",
                      t1_budget_gb: float = 11.0,
                      t2_budget_gb: float = 8.0) -> list:
    """Quantize the denoiser (transformer/unet) on `device` and move the small
    full-precision components (VAE) to the GPU.

    When `quality` is set (default "near_lossless"), uses calibrated per-layer
    precision from quantize_for_quality().  Pass quality=None and an explicit
    `bits` to use the legacy uniform-precision path.
    """
    names = []
    for name, mod in _live_modules(pipe, denoiser_attrs):
        if quality is not None:
            if verbose:
                print(f"[gb_quant] quantizing {name} "
                      f"(quality={quality}) on {device} …", flush=True)
            quantize_for_quality(
                mod, target=quality, device=device, dtype=dtype,
                group_size=group_size, t1_budget_gb=t1_budget_gb,
                t2_budget_gb=t2_budget_gb, sensitivity=sensitivity,
                model_id=f"{model_id}.{name}", verbose=verbose,
            )
        else:
            if verbose:
                print(f"[gb_quant] quantizing {name} -> {_bits_tag(bits)} "
                      f"on {device} …", flush=True)
            quantize_module(mod, bits=bits, device=device, dtype=dtype,
                            group_size=group_size)
        names.append(name)
    for name, mod in _live_modules(pipe, move_attrs):
        try:
            mod.to(device)
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        if verbose:
            alloc = torch.cuda.memory_allocated() / 2**30
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"[gb_quant] generation-ready — VRAM {alloc:.1f} GiB "
                  f"(peak {peak:.1f} GiB)", flush=True)
    return names


def encode_then_quantize(pipe, prompts, bits: "int | str" = "fp8",
                         device: str = "cuda", dtype=None, encode_fn=None,
                         group_size: int = 64, verbose: bool = True,
                         quality: "Optional[str]" = "near_lossless",
                         model_id: str = "model",
                         t1_budget_gb: float = 11.0,
                         t2_budget_gb: float = 8.0) -> list:
    """One-call sequential recipe: quantize encoders -> encode all prompts ->
    free encoders -> quantize denoiser (+ move VAE). Returns CPU embeds list
    in prompt order. After this the pipe is ready for `prompt_embeds=` calls.

    quality="near_lossless" (default) uses calibrated per-layer precision.
    Pass quality=None with an explicit `bits` for the legacy scalar path.
    """
    quantize_encoders(pipe, bits=bits, device=device, dtype=dtype,
                      group_size=group_size, verbose=verbose,
                      quality=quality, model_id=model_id,
                      t1_budget_gb=t1_budget_gb, t2_budget_gb=t2_budget_gb)
    embeds = encode_prompts(pipe, prompts, device=device, encode_fn=encode_fn,
                            verbose=verbose)
    free_encoders(pipe, verbose=verbose)
    quantize_denoiser(pipe, bits=bits, device=device, dtype=dtype,
                      group_size=group_size, verbose=verbose,
                      quality=quality, model_id=model_id,
                      t1_budget_gb=t1_budget_gb, t2_budget_gb=t2_budget_gb)
    return embeds


def maybe_quantize_from_env(obj, default_budget_gb: "float | None" = None,
                            verbose: bool = True):
    """Opt-in env hook for pipelines we don't own (ai-forge runners, LTX-2,
    third-party inference scripts): call this right after the pipeline loads.

      GB_QUANT_BUDGET_GB=11     -> quantize_to_fit(obj, 11.0)  [compact path]
      GB_QUANT_BUDGET_GB=fit    -> use `default_budget_gb` (or free VRAM * 0.92)
      unset / 0 / off           -> no-op, returns None
      GB_QUANT_BITS=tq3|tq2|8   -> precision floor for the compact plan (default int4)

      GB_QUALITY=near_lossless  -> quality-first per-layer path (overrides BUDGET_GB)
      GB_QUALITY=balanced
      GB_QUALITY=compact        -> same as BUDGET_GB path
      GB_T2_BUDGET_GB=8         -> T2 budget for quality tier (default 8.0 GiB)

    Returns the FitReport / QualityFitReport when quantization ran, else None.
    """
    raw_quality = os.environ.get("GB_QUALITY", "").strip().lower()
    if raw_quality and raw_quality in _TARGET_ERROR_CEIL:
        # Quality-first path: calibrate + plan_quality per component.
        if raw_quality == "compact":
            # compact routes back to the footprint-greedy plan_fit path.
            pass  # fall through to BUDGET_GB logic below
        else:
            t1_budget = _gb_init.auto_budget_gb() if _gb_init else 11.0
            if t1_budget <= 0.0 and torch.cuda.is_available():
                free_b, _ = torch.cuda.mem_get_info()
                t1_budget = free_b / 2**30 * 0.92
            t2_budget = float(os.environ.get("GB_T2_BUDGET_GB", "8.0"))
            if verbose:
                print(f"[gb_quant] env: GB_QUALITY={raw_quality}  "
                      f"T1={t1_budget:.1f} GiB  T2={t2_budget:.1f} GiB",
                      flush=True)
            # Quantize each quantizable nn.Module component.
            import torch.nn as nn
            results = []
            for comp_name, comp in _iter_named_components(obj):
                if not isinstance(comp, nn.Module):
                    continue
                if any(s in comp_name for s in _DEFAULT_SKIP_COMPONENTS):
                    continue
                if verbose:
                    print(f"[gb_quant] env quality: quantizing {comp_name} …",
                          flush=True)
                r = quantize_for_quality(
                    comp, target=raw_quality, t1_budget_gb=t1_budget,
                    t2_budget_gb=t2_budget, model_id=comp_name, verbose=verbose,
                )
                results.append(r)
            return results if results else None

    # Legacy compact / footprint-greedy path.
    raw = os.environ.get("GB_QUANT_BUDGET_GB", "").strip().lower()
    if raw in ("", "0", "off", "no", "false"):
        return None
    prefer_bits: "int | str" = 4
    raw_bits = os.environ.get("GB_QUANT_BITS", "").strip().lower()
    if raw_bits:
        if raw_bits in ("tq3", "tq2", "3", "2"):
            prefer_bits = "tq" + raw_bits.lstrip("tq")
        elif raw_bits in ("4", "8"):
            prefer_bits = int(raw_bits)
        elif verbose:
            print(f"[gb_quant] ignoring invalid GB_QUANT_BITS={raw_bits!r}",
                  flush=True)
    if raw == "fit":
        budget = default_budget_gb
        if budget is None:
            budget = _gb_init.auto_budget_gb() if _gb_init else 0.0
        if (budget is None or budget <= 0.0) and torch.cuda.is_available():
            free_b, _total_b = torch.cuda.mem_get_info()
            budget = free_b / 2**30 * 0.92
        if budget is None:
            return None
    else:
        try:
            budget = float(raw)
        except ValueError:
            if verbose:
                print(f"[gb_quant] ignoring invalid GB_QUANT_BUDGET_GB={raw!r}",
                      flush=True)
            return None
    return quantize_to_fit(obj, budget_gb=budget, prefer_bits=prefer_bits,
                           verbose=verbose)

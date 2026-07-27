"""
gb_quant.py , GreenBoost weight-quantization companion (quality-first strategy).

The project-agnostic Python complement to gb_attn.py.  Where gb_attn.py
compresses the *KV cache* (TurboQuant: rotation + Lloyd-Max + 1-bit QJL),
gb_quant.py compresses *model weights* to maximize effective quality subject to
VRAM+DDR budgets , NOT just "fit to VRAM".

Core insight:  T2 DDR as a quality reservoir
    On Blackwell (RTX 5070), T2 DDR costs only ~1.3× the BF16 latency for
    compute-bound diffusion (Ch. G.5 of the GreenBoost extension doc).  This means
    quality-critical layers can stay at BF16 in T2 instead of being crushed to int4
    , you get near-lossless inference at the cost of ~1.3× wall-clock, not ~0.3×
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
        Footprint-first (today's original behaviour) , scalar bits floor,
        component-granular planning via plan_fit.  Shimless, fastest throughput.

GPU family abstraction:
    gpu_profile() detects the architecture (Blackwell/Hopper/Ada/Ampere/Turing)
    and returns the ordered precision ladder available on that GPU.  nvfp4 is only
    on Blackwell (sm_12x); fp8 tensor cores are on Ada/Hopper/Blackwell only.

Execution backends (unchanged):
    - GemLite (Triton low-bit GEMM, MXFP4/NVFP4 on Blackwell sm_120) for
      int8/int4/fp8/nvfp4 , vendored under third_party/ and importable here.
    - gb_quant_tq (TurboQuant rotation + Lloyd-Max + Triton LUT-GEMM) for the
      sub-4-bit modes "tq3"/"tq2".

Public API:
    import gb_quant

    # Quality-first (new default): per-layer calibrated mixed precision
    report = gb_quant.quantize_for_quality(model.transformer,
                                            target="near_lossless",
                                            t1_budget_gb=None, t2_budget_gb=None)
    print(report)        # per-layer bits, T1/T2 split, shim_required flag

    # Pipeline helpers (quality-aware by default):
    gb_quant.quantize_encoders(pipe, quality="near_lossless")
    gb_quant.quantize_denoiser(pipe, quality="near_lossless")

    # Legacy scalar precision (compact/speed path):
    gb_quant.quantize_module(model.transformer, bits=4)
    report = gb_quant.quantize_to_fit(pipe)   # budget auto-derived from local VRAM
"""
from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

# GreenBoost layer bootstrap , patches empty_cache, starts telemetry singleton,
# wires stream scheduler + tier manager + mem pools.  No-op when not active.
try:
    import gb_init as _gb_init
    _GB_ACTIVE = _gb_init.ACTIVE
except ImportError:
    _gb_init = None
    _GB_ACTIVE = os.environ.get("GREENBOOST_ACTIVE") == "1"


def _gb_cache_release():
    """Drop caching allocator cache , no-op under GreenBoost DynamicVRAM."""
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


def _df_emit_quant(component: str, bits, quality, model_id: str,
                   duration_s: float, status: str = "ok") -> None:
    """Record a quantization decision to the dataflux log (gb_dataflux.py) ,
    works standalone on a single host, no cluster/feeder required. Best-
    effort, never raises."""
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "node": "host", "label": "gb_quant", "kind": "quantize",
            "n_items": 1, "items": [component],
            "duration_s": round(duration_s, 3), "status": status,
            "bits": _bits_tag(bits) if quality is None else quality,
            "model_id": model_id,
        })
    except Exception:
        pass


# The low-bit GEMM backend is PART OF GREENBOOST: ported into
# greenboost/third_party/ (Triton GEMM kernels originally from GemLite,
# Apache-2.0, plus the hqq quantizer). gb_quant CARRIES its backend ,
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
    "fp8": 1.05,   # FP8 e4m3fn, channel-wise , same storage as INT8
    "e4m3": 1.05,  # alias
    8: 1.05,       # INT8
    "nvfp4": 0.50, # Blackwell NVFP4: 4-bit weights + FP8 scale per group-16
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
                                      # (DERIVED at runtime = family fraction ×
                                      # this node's real T2 pool; see gpu_profile)
    # Subset of `precisions` plan_quality()'s per-layer ladder walk is allowed
    # to pick implicitly. Excludes "tq3"/"tq2" (and "nvfp4" unless explicitly
    # passed) , gb_quant_calib's dequant emulators for those are plain per-row
    # absmax approximations, not the real kernels (workflow/gb-quant.md:
    # measured tq3 relative error ~6x the near_lossless ceiling), so letting
    # the automatic quality-fit walk reach them would silently trust a wrong
    # number. tq*/nvfp4 stay reachable via an explicit scalar `bits=` call.
    # Auto-derived from `precisions` when left unset (empty tuple sentinel);
    # pass explicitly only to override.
    calibrated_precisions: Tuple = ()

    def __post_init__(self) -> None:
        if not self.calibrated_precisions:
            self.calibrated_precisions = tuple(
                p for p in self.precisions if p not in ("tq3", "tq2", "nvfp4"))


# Family table: (cc_min, cc_max) → (family, precisions, floor, quality_default, t2_tol_frac).
# Precision sets are ordered highest-quality-first and include only precisions that have
# hardware acceleration on that family.  fp8 tensor cores: Ada(8.9)/Hopper(9.x)/Blackwell(12.x).
# nvfp4: Blackwell only (sm_120 PTX FP4).
#
# t2_tol_frac (owner rule 2026-07-13: NO absolute hardware-shaped GB literal):
# the FRACTION of THIS node's real T2 DDR pool the quality tier may lean on ,
# newer fp8-capable families lean harder because their compact tiers are
# higher quality. gpu_profile() multiplies it by the node's actual T2 pool
# (pool_brief), so an 8 GB feeder and a 256 GB-RAM host derive different GB.
_GPU_FAMILIES: List[Tuple] = [
    # (cc_min, cc_max,   family,      precisions,                            floor,  q_def,    t2_frac)
    ((12, 0), (12, 99), "blackwell", (16, "fp8", "nvfp4", 8, 4, "tq3", "tq2"), 4, "nvfp4",  0.30),
    (( 9, 0), ( 9, 99), "hopper",   (16, "fp8", 8, 4, "tq3", "tq2"),           4, "fp8",    0.30),
    (( 8, 9), ( 8, 99), "ada",      (16, "fp8", 8, 4, "tq3", "tq2"),           4, "fp8",    0.22),
    (( 8, 0), ( 8,  8), "ampere",   (16, 8, 4, "tq3", "tq2"),                  4,  8,       0.22),
    (( 7, 5), ( 7, 99), "turing",   (16, 8, 4, "tq3", "tq2"),                  4,  8,       0.15),
    (( 7, 0), ( 7,  4), "volta",    (16, 8, 4, "tq3", "tq2"),                  4,  8,       0.15),
]

_GPU_PROFILE_CACHE: Optional[GpuProfile] = None


def _t2_pool_total_gb() -> float:
    """This node's real T2 DDR pool total in GB (0.0 if unknown). Reads the
    kmod pool_brief (same source as _auto_budgets), falling back to a % of
    MemTotal (the kmod auto-sizes the pool at ~70% of RAM). Never a
    host-shaped literal , scales to whatever this machine actually has."""
    try:
        import re as _re
        with open("/sys/class/greenboost/greenboost/pool_brief") as f:
            m = _re.search(r"T2:(\d+)/(\d+)GB", f.read())
        if m:
            return float(int(m.group(2)))          # total GB
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024.0 * 1024.0) * 0.70, 1)  # ~kmod autosize
    except Exception:
        pass
    return 0.0


def _nvfp4_allowed() -> bool:
    return os.environ.get("GB_ALLOW_NVFP4", "0") == "1"


def _gate_nvfp4(profile: "GpuProfile") -> "GpuProfile":
    """Blackwell's nvfp4 precision routes straight into a documented Triton
    sm_120 compiler crash (workflow/known-issues.md). gpu_profile()'s own
    `quality_default`/tiered-mode floor steer new quantize calls into it by
    default, so this must be filtered in the planner itself, not left as a
    CLI-only workaround — a fresh Blackwell install must not crash on its
    first quantize call. Opt back in with GB_ALLOW_NVFP4=1 once a fixed
    Triton ships."""
    if "nvfp4" not in profile.precisions or _nvfp4_allowed():
        return profile
    precisions = tuple(p for p in profile.precisions if p != "nvfp4")
    floor_default = 4 if profile.floor_default == "nvfp4" else profile.floor_default
    quality_default = "fp8" if profile.quality_default == "nvfp4" else profile.quality_default
    return GpuProfile(
        family=profile.family, cc=profile.cc, precisions=precisions,
        floor_default=floor_default, quality_default=quality_default,
        t2_tolerance_gb=profile.t2_tolerance_gb,
    )


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

    # Derive the absolute T2 tolerance from THIS node's real pool (owner rule:
    # no host-shaped GB literal). 0.0 when the pool is unknown , a safe sentinel
    # (the quality tier simply won't lean on T2 it can't confirm exists).
    pool_gb = _t2_pool_total_gb()
    cc = torch.cuda.get_device_capability(device)
    for (cc_min, cc_max, family, precisions, floor, q_def, t2_frac) in _GPU_FAMILIES:
        if cc_min <= cc <= cc_max:
            _GPU_PROFILE_CACHE = _gate_nvfp4(GpuProfile(
                family=family, cc=cc, precisions=precisions,
                floor_default=floor, quality_default=q_def,
                t2_tolerance_gb=round(t2_frac * pool_gb, 1),
            ))
            return _GPU_PROFILE_CACHE

    # Unknown/future architecture: conservative safe fallback (0.15 fraction)
    _GPU_PROFILE_CACHE = GpuProfile(
        family="unknown", cc=cc,
        precisions=(16, 8, 4, "tq3", "tq2"),
        floor_default=4, quality_default=8,
        t2_tolerance_gb=round(0.15 * pool_gb, 1),
    )
    return _GPU_PROFILE_CACHE


# ---------------------------------------------------------------------------
# Quality-tier configuration
# ---------------------------------------------------------------------------

# Per-tier maximum relative quantization error allowed per layer.
# rel_err = ‖W − Wq‖_F / ‖W‖_F (from gb_quant_calib.relative_quant_error).
#
# Empirical fp8 e4m3fn per-row error on FLUX/T5 weights: ~2.5-2.7% (all layers).
# This is physics of 3-bit mantissa, not a calibration failure , fp8 is still
# perceptually identical to BF16 for diffusion models.
# near_lossless = 3.0% keeps everything at fp8; BF16 T2 reservoir is only
# used for genuinely pathological layers (>3% fp8 error).
# balanced = 8.0% allows nvfp4/int4 where fp8 is overkill.
_TARGET_ERROR_CEIL: Dict[str, float] = {
    "near_lossless": 0.030,   # 3.0%  , fp8 qualifies (fp8≈2.6%); BF16 only for outliers
    "balanced":      0.080,   # 8.0%  , nvfp4/int4 qualify for most layers
    "compact":       1.000,   # unlimited , footprint-first (legacy behaviour)
}

QUALITY_TIERS = tuple(_TARGET_ERROR_CEIL.keys())


def _bits_tag(bits) -> str:
    if bits == 16:
        return "bf16"
    if bits in ("fp8", "e4m3"):
        return "fp8"
    return f"int{bits}" if isinstance(bits, int) else str(bits)


# Precision tokens below the fp8 quality floor , the owner precision rule
# (fp8 is the default; anything below it is a deliberate quality/footprint
# tradeoff, never silently applied). Keyed on the NORMALIZED value
# normalize_bits_token() returns, not the raw user string.
_BELOW_FP8_BITS = (8, 4, "nvfp4", "tq3", "tq2")


def normalize_bits_token(token: str) -> "int | str":
    """Canonical user-facing precision-token -> gb_quant-accepted `bits`
    value. Single source of truth for every caller that parses a bits
    string from a human or another process (greenboost-cli's
    quant_cmds._normalize_bits, gb_actuation.set_quant_policy) , previously
    each had its own ad hoc mapping, and quant_cmds' own version mapped
    "bf16" to the STRING "bf16", which none of quantize_module/_bits_tag/
    _BYTES_PER_PARAM recognize as gb_quant's actual bf16-passthrough
    sentinel (the int 16) , it fell through into the real per-layer
    quantize path and ValueError'd deep inside _delegate_patch/
    _build_processor instead of no-op'ing as bf16 is supposed to.

    "auto" is returned as a passthrough literal ("auto"), NOT resolved to a
    concrete precision here , what "auto" means differs per caller
    (quant_cmds._run_inprocess treats it as "use quantize_to_fit with a
    VRAM-derived budget", a whole different code path from a scalar bits
    value; a caller with no such auto-fit mode of its own, e.g.
    maybe_quantize_from_env, resolves it via gpu_profile().quality_default
    itself). Previously quant_cmds' own _normalize_bits resolved "auto" to
    a hardcoded "fp8" immediately, which made _run_inprocess's own
    `if bits == "auto":` branch unreachable dead code , this restores it.
    Genuinely unrecognized tokens fall back to "fp8" (mirrors the previous
    quant_cmds default, a safe floor on any family)."""
    t = token.strip().lower()
    if t in ("int4", "4"):
        return 4
    if t in ("int8", "8"):
        return 8
    if t in ("fp8", "e4m3"):
        return "fp8"
    if t == "nvfp4":
        return "nvfp4"
    if t in ("tq3", "3"):
        return "tq3"
    if t in ("tq2", "2"):
        return "tq2"
    if t == "bf16":
        return 16
    if t == "auto":
        return "auto"
    return "fp8"


# Components we never quantize (small, precision-sensitive, or non-Linear-heavy).
_DEFAULT_SKIP_COMPONENTS = ("vae", "scheduler", "image_processor", "feature_extractor")
# Linear sub-modules to leave at full precision inside a quantized component.
_DEFAULT_SKIP_MODULES = ("lm_head", "vision", "visual", "embed", "norm", "proj_out")

# Buffer/param/submodule name fragments that mean "this nn.Linear's weights are
# ALREADY quantized by someone else" — re-quantizing them corrupts state or
# crashes outright. Real incident (2026-07-26): LongLive's FourOverSixLinear
# subclasses nn.Linear and registers `quantized_weight_values`/`_scale_factors`
# buffers with `weight` nulled; _delegate_patch's isinstance(l, nn.Linear) scan
# matched it and would have read `layer.weight.dtype` on a None weight.
#   quantized_weight*  LongLive/FourOverSix NVFP4 (fouroversix/.../linear.py)
#   weight_scale/scale_inv  compressed-tensors, DeepSeek-style fp8 checkpoints
#   scale_factors/_amax/weight_quantizer  TransformerEngine, NVIDIA ModelOpt
#   qweight  GPTQ / AWQ / Marlin
#   _gb_impl  a previous gb_quant pass over the same layer (idempotency)
_PREQUANT_NAME_HINTS = ("quantized_weight", "weight_scale", "scale_factors",
                        "weight_quantizer", "_amax", "qweight", "scale_inv",
                        "_gb_impl")
_FLOAT_WEIGHT_DTYPES = (torch.float64, torch.float32, torch.float16, torch.bfloat16)


def is_prequantized_linear(layer: "torch.nn.Module") -> bool:
    """True when `layer` is an nn.Linear (or subclass) whose weight is not a
    plain, live float tensor — i.e. already quantized/compressed by something
    other than gb_quant, or already gb_quant'd once.

    Reads `layer._parameters` directly, never `layer.weight`: some quantizers
    delete the attribute entirely (nulled weight), and a `__getattr__`
    override on the owning module (e.g. LongLive's DynamicSwapInstaller,
    utils/memory.py) can turn a plain `.weight` read into a device copy.

    Deliberately NOT a `type(layer) is not nn.Linear` check: that would
    false-positive on legitimate float-weight Linear subclasses (e.g.
    `torch.nn.modules.linear.NonDynamicallyQuantizableLinear`, used as
    `nn.MultiheadAttention.out_proj`) and would also miss a plain `nn.Linear`
    loaded from an fp8/int checkpoint with no distinguishing subclass at all.
    """
    params = getattr(layer, "_parameters", None)
    if params is None:
        return False
    w = params.get("weight")
    if w is None:
        return True                                   # weight stripped/deleted
    if getattr(w, "dtype", None) not in _FLOAT_WEIGHT_DTYPES:
        return True                                   # fp8/int storage already
    names = (tuple(getattr(layer, "_buffers", ()) or ())
             + tuple(params)
             + tuple(getattr(layer, "_modules", ()) or ()))
    return any(hint in n for n in names for hint in _PREQUANT_NAME_HINTS)


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
    # Advisory fp8-floor placement (set only when GB_PLACEMENT=1). A
    # gb_placement.PlacementPlan; the orchestrator reads `.tail_blocks` to
    # apply gb_cluster.offload_tail_blocks. None on the default path.
    placement: object = None

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


def _autotune_cache_key() -> str:
    """Cache key = GPU name + CUDA + gemlite version.  A driver/CUDA/gemlite
    upgrade can change kernel signatures or add autotune configs, so a
    version-blind key would silently reuse stale picks; scope the file to the
    exact toolchain that produced it instead."""
    dev = torch.cuda.get_device_name(0).replace(" ", "_").replace("/", "_")
    cuda = (getattr(torch.version, "cuda", None) or "nocuda").replace(".", "")
    try:
        import gemlite
        glv = str(getattr(gemlite, "__version__", "0")).replace(".", "")
    except Exception:
        glv = "0"
    return f"{dev}_cu{cuda}_gl{glv}"


def _autotune_cache_path() -> "str | None":
    if os.environ.get("GB_QUANT_NO_AUTOTUNE_CACHE", "") == "1":
        return None
    if not torch.cuda.is_available():
        return None
    cache_dir = os.path.expanduser(
        os.environ.get("GB_QUANT_CACHE_DIR", "~/.cache/greenboost"))
    return os.path.join(cache_dir, f"gemlite_autotune_{_autotune_cache_key()}.json")


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


def _build_processor(bits, device: str, dtype, group_size: int,
                     backend: str = "gemlite"):
    """Build (and return) the weight-quantization processor for `bits`.

    `backend` selects the GEMM kernel family (see gb_kernel_backends):
      - "gemlite" (default) — the GemLite/HQQ processors below.
      - "scaled_mm"         — fp8 e4m3 via torch._scaled_mm.
      - "bf16"              — passthrough (returns None; no quantization).
    Backend resolution (env + per-layer shape) happens in `_delegate_patch`;
    this function just constructs the chosen backend's processor.

    Returns None for bits=16 or backend="bf16" (BF16 , no quantization).
    Raises ValueError for unsupported values.
    """
    if bits == 16 or backend == "bf16":
        return None
    if backend == "scaled_mm":
        import gb_kernel_backends
        return gb_kernel_backends.build_scaled_mm_processor(bits, device, dtype)
    if backend == "cutlass":
        import gb_kernel_backends
        proc = gb_kernel_backends.build_cutlass_nvfp4_processor(bits, device, dtype)
        if proc is not None:
            return proc
        # Stage-A2 not yet bench-validated (build_cutlass_nvfp4_processor still
        # a stub): fall through to the nvfp4 gemlite path. resolve_backend only
        # returns "cutlass" when gb_cutlass.available() is True (build +
        # GB_CUTLASS_ENABLE=1), so in the default config this branch is unreached.
    # backend == "gemlite" (default): GemLite/HQQ + TurboQuant processors.
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
                    processors_by_bits: "Optional[Dict]" = None,
                    scalar_bits: "Optional[object]" = None):
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
                      Layers missing from the dict use `scalar_bits` (fallback).
    processors_by_bits: pre-built {(backend, bits): processor} cache.
    scalar_bits:      the single precision for the scalar path (per_layer_bits
                      is None). When set, the processor is (re)built per layer
                      via the resolved backend; `processor` is a legacy
                      pre-built fallback used only when scalar_bits is None.

    Backend: each layer's GEMM backend is resolved per (precision, shape, arch)
    via gb_kernel_backends.resolve_backend (GB_KERNEL_BACKEND). Unset => gemlite
    => identical to the pre-registry behaviour.
    """
    from hqq.core.quantize import HQQLinear, BaseQuantizeConfig  # vendored
    import gb_kernel_backends as _kb

    dtype = getattr(module, "dtype", torch.bfloat16) or torch.bfloat16
    _proc_cache: Dict = dict(processors_by_bits or {})
    _env_backend = _kb.env_backend()
    _cc = _kb.device_cc(device)
    _backend_counts: Dict[str, int] = {}

    seen = set()
    n_prequant_skipped = 0
    n_bf16_kept = 0
    n_scalar_fallback = 0
    linears = [(n, l) for n, l in module.named_modules()
               if isinstance(l, torch.nn.Linear)]
    for name, layer in linears:
        if id(layer) in seen:
            continue
        seen.add(id(layer))
        if is_prequantized_linear(layer):
            n_prequant_skipped += 1
            continue
        layer.name = name                      # parity with backend patch_model
        layer.to(device=device, non_blocking=True)
        if any(s in name for s in skip_modules):
            continue
        # GemLite MIN_SIZE=32: in_features must be divisible by 32 AND by
        # group_size (for 4-bit). Tiny layers (e.g. in_features=16 gate
        # projections in FLUX) stay full-precision , negligible quality loss.
        in_f = layer.in_features
        out_f = layer.out_features
        if in_f % 32 != 0:
            continue

        # Determine this layer's bits.
        if per_layer_bits is not None:
            bits = per_layer_bits.get(name)
            if bits is None:
                # Layer missing from the plan (structural drift between
                # plan_quality() and this module, or a caller that combines
                # a per-layer plan with a scalar default) , this docstring's
                # own contract is "use scalar_bits", but that fallback never
                # actually ran: it silently kept the layer BF16, identical
                # to bits==16, with no accounting anywhere that a layer
                # fell outside the plan. A large layer silently staying
                # BF16 this way is exactly how a plan that "fits" OOMs.
                if scalar_bits is None:
                    n_bf16_kept += 1
                    continue
                bits = scalar_bits
                n_scalar_fallback += 1
            if bits == 16:
                n_bf16_kept += 1
                continue
            gs = group_size if bits == 4 else None
            if gs is not None and in_f % gs != 0:
                continue
        else:
            # Scalar-bits path.
            if group_size is not None and in_f % group_size != 0:
                continue
            bits = scalar_bits
            gs = group_size

        # Resolve GEMM backend for this (precision, shape, arch), then
        # build/cache the matching processor. Legacy pre-built `processor`
        # (scalar path, scalar_bits is None) short-circuits resolution.
        if scalar_bits is None and per_layer_bits is None:
            proc = processor
        else:
            backend = _kb.resolve_backend(bits, in_f, out_f, _cc, _env_backend)
            proc = _proc_cache.get((backend, bits))
            if proc is None:
                proc = _build_processor(bits, device, layer.weight.dtype,
                                        group_size, backend=backend)
                _proc_cache[(backend, bits)] = proc
            if proc is None:
                continue  # bf16 backend / bits==16 , keep layer BF16
            _backend_counts[backend] = _backend_counts.get(backend, 0) + 1

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

    if _backend_counts or n_prequant_skipped or n_bf16_kept or n_scalar_fallback:
        try:
            import gb_dataflux
            gb_dataflux.emit({"kind": "kernel_backend",
                              "env": _env_backend,
                              "backends": _backend_counts,
                              "skipped_prequantized": n_prequant_skipped,
                              "bf16_kept": n_bf16_kept,
                              "scalar_fallback": n_scalar_fallback})
        except Exception:
            pass


def quantize_module(module, bits, device: str = "cuda",
                    dtype=None, group_size: int = 64,
                    skip_modules=_DEFAULT_SKIP_MODULES,
                    per_layer_bits: "Optional[Dict[str, object]]" = None):
    """Quantize every Linear in `module` to `bits` (4, 8, "fp8"/"e4m3",
    "nvfp4", "tq3" or "tq2") moving layer-by-layer to `device` (bounded peak
    memory , never loads the full BF16 module onto the GPU).
    4/8/fp8/nvfp4 execute on GemLite Triton kernels; "tq3"/"tq2" (also plain
    3/2) execute on the TurboQuant LUT-GEMM backend (gb_quant_tq) with
    int4-HQQ fallback for non-128-multiple layers.

    bits="fp8" / bits="e4m3" , GemLite FP8 e4m3fn, channel-wise (no
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
        # scalar_bits=bits , this docstring's own contract ("bits is used
        # only as the default for layers not in the dict") was never wired
        # through: a layer plan_quality() didn't cover (structural drift
        # between planning and this call) silently stayed BF16 unaccounted
        # instead of falling back to `bits` as documented.
        with _gb_nvtx("gb:quantize:per_layer", color="cyan"):
            _delegate_patch(module, None, list(skip_modules), group_size, device,
                            per_layer_bits=per_layer_bits, scalar_bits=bits)
        gc.collect()
        _gb_cache_release()
        return module

    # Scalar-bits path (legacy / compact tier).
    if bits == 16:
        return module  # no-op
    _bits_label = _bits_tag(bits)
    with _gb_nvtx(f"gb:quantize:{_bits_label}", color="cyan"):
        gs = group_size if bits == 4 else None
        # Backend resolved per layer inside _delegate_patch (GB_KERNEL_BACKEND);
        # unset => gemlite => one processor built and reused, as before.
        _delegate_patch(module, None, list(skip_modules), gs, device,
                        scalar_bits=bits)
    gc.collect()
    _gb_cache_release()
    return module


def plan_fit(obj, budget_gb: float,
             components: "tuple[str, ...] | None" = None,
             skip_components=_DEFAULT_SKIP_COMPONENTS,
             prefer_bits: "int | str" = 4,
             tiered: "Optional[bool]" = None) -> FitReport:
    """Decide per-component precision so the total fits `budget_gb`.

    Quality-first strategy: every component gets the HIGHEST precision that
    still leaves enough budget for the rest (bf16 > int8 > int4 > tq3 > tq2).
    The floor is `prefer_bits` (int 4/8 or "tq3"/"tq2"); the headroom check
    assumes the remaining components can drop to the floor, so precision is
    only spent where the budget truly allows it. Components in
    `skip_components` stay BF16. T3 NVMe is never planned for , anything past
    the budget relies on GreenBoost T2 DDR overflow.

    Tiered precision (GB_QUANT_TIERED_PRECISION=1 or tiered=True): instead of
    downgrading every component uniformly toward `prefer_bits`, cap the ceiling
    at the GPU family's `quality_default` (fp8 on Ada/Hopper/Blackwell) and set
    the floor to its compact `floor_default` (nvfp4 on Blackwell, else int4).
    Components that fit in the VRAM budget (hot, kept in T1) keep fp8 quality;
    those that overflow to T2 drop to the compact floor, trading precision for
    bandwidth only on the cold weights.  fp8 stays the effective default; nvfp4/
    int4 are used only for the overflow tail, never as the blanket default."""
    if tiered is None:
        tiered = os.environ.get("GB_QUANT_TIERED_PRECISION", "") == "1"
    ceiling = None
    if tiered:
        prof = gpu_profile()
        # Owner precision rule: fp8 is the quality default; nvfp4 only where the
        # hardware makes it a real win (Blackwell), int4 never as a blanket
        # default.  So the HOT (fitting) tier tops out at fp8 when supported, and
        # the COLD (overflow) tier floors at nvfp4 on Blackwell, else int4.
        ceiling = "fp8" if "fp8" in prof.precisions else prof.quality_default
        prefer_bits = "nvfp4" if "nvfp4" in prof.precisions else prof.floor_default
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

    # Validate prefer_bits before using it in _gb (KeyError otherwise).
    if prefer_bits not in _PRECISION_LADDER:
        raise ValueError(f"prefer_bits={prefer_bits!r} not in "
                         f"{_PRECISION_LADDER}")
    # Largest first: decide the big tensors' precision before the budget is
    # nibbled away by small ones (best GB-per-quality allocation).
    comps.sort(key=lambda t: t[1], reverse=True)
    # Smallest possible footprint of everything (skip components stay bf16,
    # the rest at the floor precision) , the headroom reserve for "the rest".
    floor_total = sum(_gb(n, 16 if keep else prefer_bits)
                      for _, n, keep in comps)
    # Ladder = [ceiling .. floor].  Default ceiling is bf16 (index 0); tiered
    # mode caps it at fp8 so hot weights top out at fp8 rather than bf16, and
    # filters to precisions the GPU family can actually accelerate (so e.g. a
    # non-Blackwell GPU never gets nvfp4 assigned).
    _hi = _PRECISION_LADDER.index(ceiling) if ceiling is not None else 0
    ladder = list(
        _PRECISION_LADDER[_hi:_PRECISION_LADDER.index(prefer_bits) + 1])
    if tiered:
        ladder = [b for b in ladder if b in prof.precisions]
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


def quantize_to_fit(obj, budget_gb: "float | None" = None, device: str = "cuda",
                    dtype=None, components: "tuple[str, ...] | None" = None,
                    group_size: int = 64, allow_t3: bool = False,
                    prefer_bits: "int | str" = 4,
                    verbose: bool = True) -> FitReport:
    """Quantize a diffusers pipeline (or nn.Module) so it fits `budget_gb` of T1
    VRAM, then realise the plan in place. Returns the FitReport.
    budget_gb=None (default) derives the budget from THIS node's VRAM.

    `allow_t3` is accepted for API symmetry but intentionally ignored here:
    gb_quant never routes weights to T3 NVMe (it would break the speed goal).
    If the quantized model still exceeds the budget, the excess relies on
    GreenBoost's T2 DDR overflow (the shim handles that transparently)."""
    _require_gemlite()
    dtype = dtype or torch.bfloat16

    if budget_gb is None:
        # Rule: the default budget derives from the EXECUTING node's VRAM
        # (_auto_budgets), never the reference box's 11.0 literal.
        budget_gb = _auto_budgets()[0]
        if verbose:
            print(f"[gb_quant] auto budget: T1={budget_gb:.1f} GiB "
                  f"(derived from local VRAM)", flush=True)

    # Read live VRAM from telemetry to refine budget if GB telemetry is active.
    # This uses the cached snapshot (no I/O on the hot path).
    if _gb_init is not None:
        m = _gb_init.snapshot()
        if m is not None and m.fb_free_mb > 0:
            live_budget = m.fb_free_mb / 1024.0 * 0.92
            # Use telemetry budget only when it's tighter , avoids over-spending
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

    # fp8-floor cluster-fit planning (opt-in GB_PLACEMENT=1, default off).
    # Before letting plan_fit drop weights below fp8, ask gb_placement whether
    # the cluster (or local T1+T2) can hold the fp8 footprint; if so, clamp the
    # precision floor to fp8 and surface the placement (tail_blocks) so the
    # orchestrator can offload to the feeder instead of losing quality.
    _placement = None
    if os.environ.get("GB_PLACEMENT") == "1":
        try:
            import gb_placement
            prelim = plan_fit(obj, budget_gb, components=components,
                              prefer_bits=prefer_bits)
            fp8_gb = prelim.total_bf16_gb * 0.525   # fp8 ~1.05 B/param vs bf16 2.0
            t2_gb = 0.0
            try:
                t2_gb = max(0, int(os.environ.get("GREENBOOST_T2_POOL_MB", "0"))) / 1024.0
            except ValueError:
                pass
            _placement = gb_placement.plan_and_emit(
                "torch", weights_fp8_gb=fp8_gb, kv_gb=0.0,
                local_t1_gb=budget_gb, local_t2_gb=t2_gb)
            if _placement.keeps_fp8() and prefer_bits not in (16, "fp8"):
                if verbose:
                    print(f"[gb_quant] placement: {_placement.strategy} , "
                          f"clamping precision floor to fp8 ({_placement.notes})",
                          flush=True)
                prefer_bits = "fp8"
        except Exception as e:  # never let placement break quantization
            if verbose:
                print(f"[gb_quant] placement skipped ({e!r})", flush=True)

    report = plan_fit(obj, budget_gb, components=components,
                      prefer_bits=prefer_bits)
    report.placement = _placement
    if verbose:
        print(str(report), flush=True)
    name_to_module = dict(_iter_named_components(obj))
    # Execution order matters for PEAK memory, not just final footprint:
    # patch_model briefly holds each layer in BF16 on the GPU before replacing it
    # with its int4 version. Quantizing the largest component first leaves it
    # resident while the next component's transient BF16 layers load on top ,
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
        print(f"[gb_quant] done , VRAM in use {used:.1f} GiB "
              f"(budget {budget_gb:.1f} GiB)", flush=True)
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "node": "host", "label": "gb_quant", "kind": "quantize_to_fit",
            "n_items": len(report.components),
            "items": [f"{p.name}:{_bits_tag(p.bits)}" for p in report.components],
            "duration_s": 0.0, "status": "ok",
            "budget_gb": budget_gb, "total_quant_gb": report.total_quant_gb,
            "total_bf16_gb": report.total_bf16_gb,
        })
    except Exception:
        pass
    return report


# ---------------------------------------------------------------------------
# Quality-first per-layer planning and quantization
# ---------------------------------------------------------------------------

def plan_quality(module: "torch.nn.Module",
                 target: str = "near_lossless",
                 t1_budget_gb: "float | None" = None,   # None → _auto_budgets()
                 t2_budget_gb: "float | None" = None,
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
    if t1_budget_gb is None or t2_budget_gb is None:
        auto_t1, auto_t2 = _auto_budgets()
        t1_budget_gb = auto_t1 if t1_budget_gb is None else t1_budget_gb
        t2_budget_gb = auto_t2 if t2_budget_gb is None else t2_budget_gb

    if sensitivity is None:
        # Lazy calibration , falls through to cache on second call.
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
        if is_prequantized_linear(layer):
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
            # Walk the CALIBRATED ladder from the most-compressed precision
            # upward toward fp8 (i.e. reversed from precisions' own
            # highest-quality-first order), and take the FIRST one that
            # meets the ceiling , that is what "select the LOWEST precision
            # where rel_err <= error_ceiling" (this function's own
            # docstring) means: don't spend more bits than the tier's
            # quality floor requires. The previous walk went highest-
            # quality-first and broke on the first (i.e. best, not lowest)
            # match, so fp8 — always first in the Blackwell ladder and
            # almost always under both the 3% and 8% ceilings — won
            # unconditionally; "balanced" (8% ceiling) was byte-identical
            # to "near_lossless" (3%), making the whole tier dead code.
            ladder = [p for p in profile.calibrated_precisions if p != 16]
            bits = 16
            for candidate in reversed(ladder):
                err = layer_s.get(candidate, 1.0)
                if err <= error_ceil:
                    bits = candidate
                    break
            # else: no calibrated precision meets the ceiling , this layer
            # is genuinely sensitive. Keep BF16 (goes to T2 reservoir).

        # For INT4: check group_size divisibility.
        if bits == 4 and in_f % group_size != 0:
            # Fall back to next-higher precision in the ladder.
            ladder = list(profile.precisions)
            idx = ladder.index(4) if 4 in ladder else len(ladder)
            bits = ladder[idx - 1] if idx > 0 else 16

        per_layer_bits[name] = bits
        # Record the real calibrated error for whatever bits were actually
        # assigned , covers compact/no-sensitivity/group-size-fallback
        # layers too, not just the ladder-walk branch above (previously
        # those were silently excluded from accounting, so
        # QualityFitReport.__str__ reported mean_err=0.00000 for an
        # all-compact plan as if it were lossless).
        if bits != 16 and name in sensitivity and bits in sensitivity[name]:
            layer_errs_used.append(sensitivity[name][bits])

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
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "node": "host", "label": "gb_quant", "kind": "quant_plan",
            "n_items": len(per_layer_bits), "items": [model_id],
            "duration_s": 0.0, "status": "ok",
            "target": target, "error_ceiling": error_ceil,
            "precision_histogram": hist,
            "bf16_kept": hist.get("bf16", 0),
            "mean_rel_err": round(mean_err, 5), "max_rel_err": round(max_err, 5),
        })
    except Exception:
        pass
    return report


def _auto_budgets() -> "tuple[float, float]":
    """Device-derived (t1_budget_gb, t2_budget_gb) — Rule #1: plan against
    ~90% of the LOCAL card's physical VRAM minus what's already resident.

    The previous hardcoded defaults (11.0/8.0 — host-tuned for the 12 GB
    RTX 5070) fatally over-planned on an 8 GB feeder: 2026-07-13 live
    incident, gb_quant printed "T1=11.0 GiB" while the card had 7.5 GiB,
    the transformer load OOM'd and the runner escalated to shimless bf16
    (which can never fit) — the whole cluster job failed. Under the
    diffusion profile GREENBOOST_REPORT_PHYSICAL_VRAM=1 makes
    torch.cuda.mem_get_info report the real card, not the pooled figure.

    When both live probes fail the fallback is topology-derived (this node's
    own card/RAM), never the reference box's numbers — a 0.0 sentinel means
    "unknown" so a loud 2 GiB floor is used and a warn event emitted."""
    t1 = t2 = 0.0                           # 0 = not yet resolved
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        used_gb = (total_b - free_b) / 2 ** 30
        t1 = max(1.0, round(total_b / 2 ** 30 * 0.90 - used_gb, 1))
    except Exception:
        pass
    try:
        import re as _re
        with open("/sys/class/greenboost/greenboost/pool_brief") as f:
            m = _re.search(r"T2:(\d+)/(\d+)GB", f.read())
        if m:
            t2 = max(1.0, float(int(m.group(2)) - int(m.group(1))))
    except Exception:
        pass
    if t1 <= 0 or t2 <= 0:
        try:
            import gb_topology
            _tp = gb_topology.get_topology()
        except Exception:
            _tp = None
        if t1 <= 0:
            if _tp is not None and _tp.vram_gb > 0:
                t1 = max(2.0, round(_tp.vram_gb * 0.90
                                    - gb_topology.compute_reserve_gb(_tp.physical_vram_mb), 1))
            else:
                t1 = 2.0
                try:
                    import gb_dataflux
                    gb_dataflux.emit({"node": "host", "label": "quant",
                                      "kind": "quant_budget_fallback", "status": "warn",
                                      "reason": "VRAM undetectable", "t1_gb": t1})
                except Exception:
                    pass
        if t2 <= 0:
            if _tp is not None and (_tp.virtual_vram_gb > 0 or _tp.ram_total_gb > 0):
                t2 = max(2.0, float(_tp.virtual_vram_gb or _tp.ram_total_gb * 25 // 100))
            else:
                t2 = 2.0
    return t1, t2


def quantize_for_quality(module: "torch.nn.Module",
                          target: str = "near_lossless",
                          device: str = "cuda",
                          dtype=None,
                          group_size: int = 64,
                          t1_budget_gb: "float | None" = None,
                          t2_budget_gb: "float | None" = None,
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
    if t1_budget_gb is None or t2_budget_gb is None:
        auto_t1, auto_t2 = _auto_budgets()
        t1_budget_gb = auto_t1 if t1_budget_gb is None else t1_budget_gb
        t2_budget_gb = auto_t2 if t2_budget_gb is None else t2_budget_gb

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
        print(f"[gb_quant] quality done , VRAM {used:.1f} GiB  "
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
# ai-forge runners, ...) stay THIN callers , all quantization policy lives
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
                      t1_budget_gb: "float | None" = None,
                      t2_budget_gb: "float | None" = None) -> list:
    """Quantize every text encoder on `pipe` and move it to `device`.

    When `quality` is set (default "near_lossless"), uses calibrated per-layer
    precision from quantize_for_quality().  Pass quality=None and an explicit
    `bits` to use the legacy uniform-precision path.
    """
    names = []
    for name, mod in _live_modules(pipe, encoder_attrs):
        t0 = time.time()
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
        _df_emit_quant(name, bits, quality, model_id, time.time() - t0)
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
                      t1_budget_gb: "float | None" = None,
                      t2_budget_gb: "float | None" = None) -> list:
    """Quantize the denoiser (transformer/unet) on `device` and move the small
    full-precision components (VAE) to the GPU.

    When `quality` is set (default "near_lossless"), uses calibrated per-layer
    precision from quantize_for_quality().  Pass quality=None and an explicit
    `bits` to use the legacy uniform-precision path.
    """
    names = []
    for name, mod in _live_modules(pipe, denoiser_attrs):
        t0 = time.time()
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
        _df_emit_quant(name, bits, quality, model_id, time.time() - t0)
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
            print(f"[gb_quant] generation-ready , VRAM {alloc:.1f} GiB "
                  f"(peak {peak:.1f} GiB)", flush=True)
    return names


def encode_then_quantize(pipe, prompts, bits: "int | str" = "fp8",
                         device: str = "cuda", dtype=None, encode_fn=None,
                         group_size: int = 64, verbose: bool = True,
                         quality: "Optional[str]" = "near_lossless",
                         model_id: str = "model",
                         t1_budget_gb: "float | None" = None,
                         t2_budget_gb: "float | None" = None) -> list:
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

      GB_QUANT_BUDGET_GB=<gb>   -> quantize_to_fit(obj, <gb>)  [compact path]
      GB_QUANT_BUDGET_GB=fit    -> use `default_budget_gb` (or free VRAM * 0.92)
      unset / 0 / off           -> no-op, returns None
      GB_QUANT_BITS=tq3|tq2|8   -> precision floor for the compact plan (default int4)

      GB_QUALITY=near_lossless  -> quality-first per-layer path (overrides BUDGET_GB)
      GB_QUALITY=balanced
      GB_QUALITY=compact        -> same as BUDGET_GB path
      GB_T2_BUDGET_GB=8         -> T2 budget for quality tier (default: auto
                                   from the live kernel T2 pool, not a literal)

    Returns the FitReport / QualityFitReport when quantization ran, else None.
    """
    raw_quality = os.environ.get("GB_QUALITY", "").strip().lower()
    if raw_quality and raw_quality in _TARGET_ERROR_CEIL:
        # Quality-first path: calibrate + plan_quality per component.
        if raw_quality == "compact":
            # compact routes back to the footprint-greedy plan_fit path.
            pass  # fall through to BUDGET_GB logic below
        else:
            # Rule: fallbacks derive from the executing node (_auto_budgets),
            # not the reference box's 11.0/8.0 literals.
            t1_budget = _gb_init.auto_budget_gb() if _gb_init else _auto_budgets()[0]
            if t1_budget <= 0.0 and torch.cuda.is_available():
                free_b, _ = torch.cuda.mem_get_info()
                t1_budget = free_b / 2**30 * 0.92
            _t2_env = os.environ.get("GB_T2_BUDGET_GB", "").strip()
            t2_budget = float(_t2_env) if _t2_env else _auto_budgets()[1]
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
        # Shared normalizer (also used by set_quant_policy/quant_cmds) , now
        # accepts the full precision vocabulary (fp8/nvfp4/tq3/tq2/int8/int4/
        # bf16/auto), not just the legacy tq*/3/2/4/8 subset this used to
        # hand-parse. Unrecognized tokens fall back to "fp8" (a safe, real
        # precision) rather than the old silent "ignore, keep default 4".
        prefer_bits = normalize_bits_token(raw_bits)
        if prefer_bits == "auto":
            # This path has no quantize_to_fit-style auto-budget mode of its
            # own (unlike quant_cmds._run_inprocess) , resolve to a concrete,
            # family-aware precision instead.
            prefer_bits = gpu_profile().quality_default
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

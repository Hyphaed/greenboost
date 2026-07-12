"""
gb_quant_tq.py , TurboQuant WEIGHT backend for gb_quant (sub-4-bit modes).

Implements the TurboQuant MSE quantizer (random rotation + per-coordinate
Lloyd-Max, arXiv:2504.19874) as a *weight* format with a Triton LUT-GEMM
execution kernel.  This is the "TQ-weight GEMM kernel (rotated activations +
codebook dequant)" follow-up to workflow/experiments/tq_weight_study.py:
at 4 bits per weight HQQ+GemLite wins, at <=3 bits TurboQuant wins , so this
module only offers "tq3" (3-bit) and "tq2" (2-bit) modes and gb_quant keeps
int4-HQQ as the default.

Weight format (per nn.Linear, weight W of shape (N, K), K % 128 == 0):
    Each 128-coordinate block b of every weight row is stored as
        W_b  ~=  ||W_b|| * (C[idx_b] @ Pi)
    where Pi is ONE shared random orthogonal 128x128 matrix (seeded QR, the
    paper's Algorithm 1) and C is the d=128 Lloyd-Max codebook (2^bits
    centroids over the Beta-distributed rotated coordinates).
      - indices: packed (32//bits) per int32 word, layout (ceil(K/eps), N)
      - scales:  block norms, fp16, layout (K/128, N)
      - lut:     2^bits fp32 centroids (shared, tiny)
    Footprint: bits/8 + 2/128 bytes per param (tq3 ~0.39 B, tq2 ~0.27 B).

Runtime:
    y = x @ W^T  =  sum_b ||W_b|| * (x_b @ Pi^T) . C[idx_b]
    so the activations are rotated block-wise ONCE per forward (a single
    (M*K/128, 128) @ (128, 128) matmul, ~K=128-th of the main GEMM's work)
    and the Triton kernel computes a standard tiled GEMM whose B-dequant is
    a codebook gather  lut[idx] * scale  instead of GemLite's affine
    (w - zero) * scale.  Group structure (one scale per 128 K-elements per
    output column) matches GemLite's group_size=128 layout exactly.

Public API (used by gb_quant.quantize_module(bits="tq3"|"tq2")):
    proc = A16Wtq(nbits=3, device="cuda", dtype=torch.bfloat16)
    impl = proc.from_linear(linear)        # -> TQLinear (drop-in forward)
Layers whose in_features is not a multiple of 128 fall back to int4-HQQ via
GemLite inside from_linear (same backend gb_quant already carries).
"""
from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Lloyd-Max codebooks for d=128 (from turboquantsolutions/turboquant,
# codebook_d128_b{2,3}.json , Beta-distribution-optimal scalar quantizers
# for coordinates of randomly rotated unit vectors). Embedded so gb_quant
# stays self-carried: consumer venvs install nothing.
# ---------------------------------------------------------------------------
_CODEBOOKS = {
    2: {
        "centroids": [-0.1330402, -0.03999095, 0.03999095, 0.1330402],
        # interior decision boundaries (len 2^b - 1) for searchsorted
        "boundaries": [-0.08651557, 0.0, 0.08651557],
    },
    3: {
        "centroids": [-0.18839061, -0.11813298, -0.0665806, -0.02160247,
                      0.02160247, 0.0665806, 0.11813298, 0.18839061],
        "boundaries": [-0.1532618, -0.09235679, -0.04409153, 0.0,
                       0.04409153, 0.09235679, 0.1532618],
    },
}

_BLOCK_D = 128          # rotation/codebook dimension (fixed; codebooks above)
_ROT_SEED = 42          # one shared Pi for all layers (data-oblivious)

_rot_cache: dict = {}   # (device, dtype) -> Pi / Pi^T tensors


def _rotation(device, dtype=torch.float32) -> torch.Tensor:
    """Shared random orthogonal Pi (seeded QR of a Gaussian, det=+1)."""
    key = (str(device), dtype)
    if key not in _rot_cache:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(_ROT_SEED)
        G = torch.randn(_BLOCK_D, _BLOCK_D, generator=gen, dtype=torch.float32)
        Q, R = torch.linalg.qr(G)
        Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
        _rot_cache[key] = Q.to(device=device, dtype=dtype)
    return _rot_cache[key]


# ---------------------------------------------------------------------------
# Bit packing: column-major-friendly layout (K/eps, N), element k of column n
# lives in word [k // eps, n] at bit offset (k % eps) * nbits.  eps = 32//nbits
# (3-bit: 10 per word, 2 spare bits; 2-bit: 16 per word).  Disjoint bit fields
# mean sum == bitwise-or, so packing is a shifted sum.
# ---------------------------------------------------------------------------
def _pack_over_k(idx_kn: torch.Tensor, nbits: int) -> torch.Tensor:
    eps = 32 // nbits
    K, N = idx_kn.shape
    Kp = (K + eps - 1) // eps * eps
    if Kp != K:
        idx_kn = torch.cat([idx_kn, idx_kn.new_zeros(Kp - K, N)])
    idx_kn = idx_kn.to(torch.int32).view(Kp // eps, eps, N)
    shifts = (torch.arange(eps, device=idx_kn.device, dtype=torch.int32)
              * nbits).view(1, eps, 1)
    return (idx_kn << shifts).sum(dim=1, dtype=torch.int32)


def _unpack_over_k(W_q: torch.Tensor, nbits: int, K: int) -> torch.Tensor:
    eps = 32 // nbits
    shifts = (torch.arange(eps, device=W_q.device, dtype=torch.int32)
              * nbits).view(1, eps, 1)
    out = (W_q.unsqueeze(1) >> shifts) & ((1 << nbits) - 1)
    return out.reshape(-1, W_q.shape[1])[:K]


# ---------------------------------------------------------------------------
# Triton LUT-GEMM:  C = A @ dequant(B),  dequant = lut[idx] * scale
# A (M, K) fp16/bf16 = pre-rotated activations; B (K/eps, N) packed int32;
# scales (K/128, N) fp16; lut (2^bits,) fp32.
# ---------------------------------------------------------------------------
_TRITON_ERR = None
try:
    import triton
    import triton.language as tl

    _TQ_CONFIGS = [
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64,
                       "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64,
                       "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64,
                       "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64,
                       "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128,
                       "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64,
                       "GROUP_M": 8}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32,
                       "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64,
                       "GROUP_M": 8}, num_warps=8, num_stages=2),
    ]

    @triton.autotune(configs=_TQ_CONFIGS, key=["M_BUCKET", "N", "K", "W_NBITS"])
    @triton.jit
    def _tq_gemm_kernel(
        a_ptr, b_ptr, lut_ptr, scales_ptr, bias_ptr, c_ptr,
        M, N, K, M_BUCKET,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_sk, stride_sn,
        stride_cm, stride_cn,
        W_NBITS: tl.constexpr, EPS: tl.constexpr, MASK: tl.constexpr,
        GROUP_K: tl.constexpr, HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        m_mask = offs_m < M
        n_mask = offs_n < N

        # Codebook as scalars: the dequant becomes an ALU select-tree instead
        # of a per-element global-memory gather (the gather was ~2.3x slower
        # at M=4096 where the GEMM is compute-bound).
        c0 = tl.load(lut_ptr + 0)
        c1 = tl.load(lut_ptr + 1)
        c2 = tl.load(lut_ptr + 2)
        c3 = tl.load(lut_ptr + 3)
        if W_NBITS == 3:
            c4 = tl.load(lut_ptr + 4)
            c5 = tl.load(lut_ptr + 5)
            c6 = tl.load(lut_ptr + 6)
            c7 = tl.load(lut_ptr + 7)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, tl.cdiv(K, BLOCK_K)):
            k_offs = k0 * BLOCK_K + offs_k
            k_mask = k_offs < K
            a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :],
                        other=0.0)
            word = tl.load(b_ptr + (k_offs[:, None] // EPS) * stride_bk
                           + offs_n[None, :] * stride_bn,
                           mask=k_mask[:, None] & n_mask[None, :], other=0)
            idx = (word >> ((k_offs % EPS) * W_NBITS)[:, None]) & MASK
            lo = tl.where(idx < 2, tl.where(idx == 0, c0, c1),
                          tl.where(idx == 2, c2, c3))
            if W_NBITS == 3:
                hi = tl.where(idx < 6, tl.where(idx == 4, c4, c5),
                              tl.where(idx == 6, c6, c7))
                w = tl.where(idx < 4, lo, hi)
            else:
                w = lo
            # one scale row per K-group (BLOCK_K <= GROUP_K, both pow2)
            scale = tl.load(scales_ptr
                            + ((k0 * BLOCK_K) // GROUP_K) * stride_sk
                            + offs_n * stride_sn,
                            mask=n_mask, other=0.0).to(tl.float32)
            w = (w * scale[None, :]).to(a.dtype)
            acc = tl.dot(a, w, acc)
            a_ptrs += BLOCK_K * stride_ak

        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=n_mask,
                           other=0.0).to(tl.float32)[None, :]
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty),
                 mask=m_mask[:, None] & n_mask[None, :])

except Exception as e:  # pragma: no cover - environment dependent
    _TRITON_ERR = e


# ---------------------------------------------------------------------------
# Persistent autotune cache for the TQ LUT-GEMM kernel.
#
# Unlike GemLite (which gb_quant persists via gemlite.cache_config/load_config,
# see gb_quant._arm_autotune_cache), Triton's own @triton.autotune has no
# cross-process persistence: _tq_gemm_kernel re-benchmarks its 8 configs on the
# first launch of every process.  We snapshot the Autotuner's .cache dict
# (key-tuple -> triton.Config) to disk and pre-populate it on the next run.
#
# Same env contract as the gemlite cache: GB_QUANT_NO_AUTOTUNE_CACHE=1 disables,
# GB_QUANT_CACHE_DIR relocates.  File is keyed by GPU + CUDA + triton version so
# a toolchain upgrade never reuses stale picks (the file name simply changes).
# ---------------------------------------------------------------------------
import os as _os

_tq_cache_armed = False


def _tq_cache_key() -> str:
    dev = torch.cuda.get_device_name(0).replace(" ", "_").replace("/", "_")
    cuda = (getattr(torch.version, "cuda", None) or "nocuda").replace(".", "")
    try:
        import triton as _tr
        trv = str(getattr(_tr, "__version__", "0")).replace(".", "")
    except Exception:
        trv = "0"
    return f"{dev}_cu{cuda}_tr{trv}"


def _tq_cache_path() -> "str | None":
    if _os.environ.get("GB_QUANT_NO_AUTOTUNE_CACHE", "") == "1":
        return None
    if not torch.cuda.is_available():
        return None
    cache_dir = _os.path.expanduser(
        _os.environ.get("GB_QUANT_CACHE_DIR", "~/.cache/greenboost"))
    return _os.path.join(cache_dir, f"tq_autotune_{_tq_cache_key()}.json")


def _config_to_dict(cfg) -> "dict | None":
    """Serialize a triton.Config to a plain dict (kwargs + launch params)."""
    try:
        d = {"kwargs": dict(cfg.kwargs),
             "num_warps": int(cfg.num_warps),
             "num_stages": int(cfg.num_stages)}
        if getattr(cfg, "num_ctas", None) is not None:
            d["num_ctas"] = int(cfg.num_ctas)
        return d
    except Exception:
        return None


def _dict_to_config(d: dict):
    import triton
    kw = {"num_warps": d.get("num_warps", 4),
          "num_stages": d.get("num_stages", 2)}
    if "num_ctas" in d:
        kw["num_ctas"] = d["num_ctas"]
    try:
        return triton.Config(d["kwargs"], **kw)
    except TypeError:
        # older/newer signature: drop num_ctas and retry
        kw.pop("num_ctas", None)
        return triton.Config(d["kwargs"], **kw)


def _arm_tq_cache() -> None:
    """Load the TQ autotune cache and arm the at-exit save (idempotent)."""
    global _tq_cache_armed
    if _tq_cache_armed or _TRITON_ERR is not None:
        return
    _tq_cache_armed = True
    path = _tq_cache_path()
    if not path:
        return
    cache = getattr(_tq_gemm_kernel, "cache", None)
    if cache is None:
        return  # autotuner shape changed upstream; skip silently
    import atexit
    import json

    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    try:
        if _os.path.isfile(path):
            with open(path) as f:
                blob = json.load(f)
            loaded = 0
            for ent in blob.get("entries", []):
                key = tuple(ent["key"])
                cfg = _dict_to_config(ent["config"])
                cache[key] = cfg
                loaded += 1
            if loaded:
                print(f"[gb_quant_tq] TQ autotune cache loaded: {path} "
                      f"({loaded} entries)")
    except Exception as e:
        print(f"[gb_quant_tq] TQ autotune cache load skipped ({e!r})")

    def _save() -> None:
        try:
            entries = []
            for key, cfg in _tq_gemm_kernel.cache.items():
                cd = _config_to_dict(cfg)
                if cd is None:
                    continue
                entries.append({"key": list(key), "config": cd})
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"entries": entries}, f)
            _os.replace(tmp, path)
        except Exception:
            pass

    atexit.register(_save)


def tq_gemm(a: torch.Tensor, W_q: torch.Tensor, scales: torch.Tensor,
            lut: torch.Tensor, nbits: int, K: int,
            bias: "torch.Tensor | None" = None) -> torch.Tensor:
    """C = a @ dequant(W_q)  with a (M, K) already block-rotated."""
    _arm_tq_cache()
    M = a.shape[0]
    N = W_q.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    eps = 32 // nbits
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"])
                         * triton.cdiv(N, meta["BLOCK_N"]),)
    _tq_gemm_kernel[grid](
        a, W_q, lut, scales, bias if bias is not None else a, c,
        M, N, K, min(triton.next_power_of_2(M), 1024),
        a.stride(0), a.stride(1),
        W_q.stride(0), W_q.stride(1),
        scales.stride(0), scales.stride(1),
        c.stride(0), c.stride(1),
        W_NBITS=nbits, EPS=eps, MASK=(1 << nbits) - 1,
        GROUP_K=_BLOCK_D, HAS_BIAS=bias is not None,
    )
    return c


def tq_gemm_ref(a: torch.Tensor, W_q: torch.Tensor, scales: torch.Tensor,
                lut: torch.Tensor, nbits: int, K: int,
                bias: "torch.Tensor | None" = None) -> torch.Tensor:
    """Pure-torch reference (and CPU/no-Triton fallback): dequantize B fully,
    then matmul. Correct but allocates the (K, N) fp32 weight each call."""
    idx = _unpack_over_k(W_q, nbits, K).long()               # (K, N)
    w = lut[idx] * scales.to(torch.float32).repeat_interleave(_BLOCK_D, dim=0)
    y = a.to(torch.float32) @ w
    if bias is not None:
        y += bias.to(torch.float32)
    return y.to(a.dtype)


class TQLinear(torch.nn.Module):
    """Drop-in Linear replacement executing the TurboQuant weight format."""

    def __init__(self, nbits: int, in_features: int, out_features: int,
                 compute_dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.nbits = nbits
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        cb = _CODEBOOKS[nbits]
        eps = 32 // nbits
        kp = (in_features + eps - 1) // eps
        self.register_buffer("W_q", torch.zeros(kp, out_features,
                                                dtype=torch.int32,
                                                device=device))
        self.register_buffer("scales", torch.zeros(in_features // _BLOCK_D,
                                                   out_features,
                                                   dtype=torch.float16,
                                                   device=device))
        self.register_buffer("lut", torch.tensor(cb["centroids"],
                                                 dtype=torch.float32,
                                                 device=device))
        # rot = Pi^T so forward computes x_b @ Pi^T (the paper's rotation)
        self.register_buffer("rot", _rotation(device).t().contiguous()
                             .to(compute_dtype))
        self.bias = None

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: torch.nn.Linear, nbits: int,
                    device="cuda", compute_dtype=torch.bfloat16) -> "TQLinear":
        N, K = linear.weight.shape
        if K % _BLOCK_D != 0:
            raise ValueError(f"in_features={K} not a multiple of {_BLOCK_D}")
        self = cls(nbits, K, N, compute_dtype=compute_dtype, device=device)
        W = linear.weight.data.to(device=device, dtype=torch.float32)
        blocks = W.view(N, K // _BLOCK_D, _BLOCK_D)
        norms = blocks.norm(dim=-1)                          # (N, K/128)
        unit = blocks / norms.clamp_min(1e-10).unsqueeze(-1)
        y = unit @ _rotation(device).t()                     # x_b @ Pi^T
        db = torch.tensor(_CODEBOOKS[nbits]["boundaries"],
                          dtype=torch.float32, device=device)
        idx = torch.searchsorted(db, y.reshape(N, K).contiguous())
        self.W_q.copy_(_pack_over_k(idx.t().contiguous().to(torch.int32),
                                    nbits))
        self.scales.copy_(norms.t().contiguous().to(torch.float16))
        if linear.bias is not None:
            self.bias = torch.nn.Parameter(
                linear.bias.data.to(device=device, dtype=torch.float32),
                requires_grad=False)
        del W, blocks, norms, unit, y, idx
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features).to(self.compute_dtype)
        xr = (x2.view(-1, self.in_features // _BLOCK_D, _BLOCK_D)
              @ self.rot).view(-1, self.in_features)
        if _TRITON_ERR is None and xr.is_cuda:
            y = tq_gemm(xr, self.W_q, self.scales, self.lut, self.nbits,
                        self.in_features, bias=self.bias)
        else:
            y = tq_gemm_ref(xr, self.W_q, self.scales, self.lut, self.nbits,
                            self.in_features, bias=self.bias)
        return y.view(*lead, self.out_features).to(x.dtype)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, nbits={self.nbits} (tq)")


class _KeepLinear(torch.nn.Module):
    """Unquantized delegate for shapes neither TQ nor GemLite can take.
    Keeps its own copy of the weights so _delegate_patch can null the
    original Linear's parameters like for every other impl."""

    def __init__(self, linear: torch.nn.Linear, device, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(
            linear.weight.data.to(device=device, dtype=dtype),
            requires_grad=False)
        self.bias = None
        if linear.bias is not None:
            self.bias = torch.nn.Parameter(
                linear.bias.data.to(device=device, dtype=dtype),
                requires_grad=False)

    def forward(self, x):
        return torch.nn.functional.linear(x.to(self.weight.dtype),
                                          self.weight, self.bias)


class A16Wtq:
    """gb_quant processor for TurboQuant weight modes ("tq3" / "tq2").

    Mirrors the GemLite helper-processor API consumed by
    gb_quant._delegate_patch (from_linear -> module with .forward).  Layers
    whose in_features isn't a multiple of 128 can't take the block rotation
    and fall back to int4-HQQ/GemLite (the backend gb_quant already carries);
    shapes GemLite rejects too (in_features % 32 != 0) stay at full precision
    via _KeepLinear.
    """

    def __init__(self, nbits: int = 3, device: str = "cuda",
                 dtype=torch.bfloat16, fallback=None):
        if nbits not in _CODEBOOKS:
            raise ValueError(f"tq nbits must be one of {sorted(_CODEBOOKS)}")
        self.nbits = nbits
        self.device = device
        self.dtype = dtype
        self.fallback = fallback        # a GemLite processor, e.g. A16W4_HQQ_INT

    def from_linear(self, linear: torch.nn.Linear):
        K = linear.weight.shape[1]
        if K % _BLOCK_D == 0:
            return TQLinear.from_linear(linear, self.nbits, device=self.device,
                                        compute_dtype=self.dtype)
        gs = 64 if K % 64 == 0 else (32 if K % 32 == 0 else None)
        if self.fallback is None or gs is None:
            return _KeepLinear(linear, self.device, self.dtype)
        from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
        cfg = BaseQuantizeConfig(nbits=self.fallback.W_nbits, group_size=gs)
        return self.fallback.from_hqqlinear(
            HQQLinear(linear, quant_config=cfg,
                      compute_dtype=linear.weight.dtype,
                      device=self.device))

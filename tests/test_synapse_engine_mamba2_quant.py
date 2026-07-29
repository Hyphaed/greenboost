#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for synapse_engine/gllm/models/mamba.py's quantized in_proj loading
(GPTQ/AWQ/FP8) added 2026-07-29 — see workflow/known-issues.md's "quantized
Mamba-2 weight loading" entry. Before this, ``_load_mamba2_layer_weights``
only handled a plain float ``in_proj.weight``/``in_proj.bias``; any GPTQ/AWQ
checkpoint (whose in_proj has no ``.weight`` key at all — it's registered
``None``, real data lives under ``.qweight``/``.qzeros``/``.scales``/
``.g_idx``) hit the pre-pass's existence guard and had its ENTIRE mixer
(in_proj + A_log/D/dt_bias) silently skipped, left at random init.

No real quantized Mamba-2 checkpoint exists to verify end-to-end (this
session's only live-tested checkpoint, ``AntonV/mamba2-130m-hf``, is plain
bf16) — these tests instead verify the new split/TP-slice index math against
hand-computed reference slices, the same rigor used elsewhere in this repo
for format-specific quant code before a real checkpoint was available.

Loaded by file path via importlib (mamba.py's own module-level imports pull
in ``gllm.dist_utils``/``gllm.layers.linear``/etc., which recursively import
``gllm/__init__.py`` -> ``gllm.llm_engine`` -> ``pyzmq``, only present in the
dedicated synapse-torch-env venv) — the gllm-internal imports are faked via
``sys.modules`` stubs first, same pattern test_synapse_engine_gptq.py uses
for its one lazy-import test. mamba.py's own logic (the functions under
test) is untouched by the stubs; only its unrelated-to-this-test imports are.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


_STUB_NAMES = (
    "gllm", "gllm.dist_utils", "gllm.input_data", "gllm.layers",
    "gllm.layers.linear", "gllm.layers.ops", "gllm.layers.ops.fla",
    "gllm.layers.ops.mamba", "gllm.layers.vocab_parallel_embedding",
    "gllm.memory_manager", "gllm.models", "gllm.models.weight_loader",
    "gllm.models.weight_utils",
)


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _load_mamba():
    """Stub out mamba.py's gllm-internal imports (real gllm needs pyzmq,
    only present in the dedicated synapse-torch-env venv -- see
    test_synapse_engine_gptq.py's docstring for the established pattern),
    load the REAL file via importlib so this tests actual production code,
    then remove the stub entries from sys.modules again -- once exec_module
    has bound names into the loaded module's own namespace, the stubs are no
    longer needed, and leaving them in sys.modules would break any
    later-collected test file's `pytest.importorskip("gllm")` (it would find
    this fake truthy stub instead of correctly skipping)."""
    saved = {name: sys.modules.get(name) for name in _STUB_NAMES}
    try:
        _stub("gllm")
        _stub(
            "gllm.dist_utils",
            get_pp_layers=lambda *a, **k: (0, 1),
            get_tp_rank=lambda: 0,
            get_tp_size=lambda: 1,
            is_first_pp_rank=lambda: True,
            is_last_pp_rank=lambda: True,
        )
        _stub("gllm.input_data", InputData=object)
        _stub("gllm.layers")
        _stub(
            "gllm.layers.linear",
            ColumnParallelLinear=object,
            MergedColumnParallelLinear=object,
            RowParallelLinear=object,
        )
        _stub("gllm.layers.ops")
        _stub("gllm.layers.ops.fla", RMSNormGated=object)
        _stub(
            "gllm.layers.ops.mamba",
            causal_conv1d_fn=None,
            causal_conv1d_update=None,
            mamba_chunk_scan_combined_varlen=None,
            selective_state_update=None,
        )
        _stub(
            "gllm.layers.vocab_parallel_embedding",
            ParallelLMHead=object,
            VocabParallelEmbedding=object,
        )
        _stub("gllm.memory_manager", SSMCacheConfig=object)
        _stub("gllm.models")
        _stub(
            "gllm.models.weight_loader",
            LoadContext=object,
            WeightRule=object,
            contains=lambda *a: (lambda k: False),
            h_proj_dim0=None,
            h_proj_dim1=None,
            run_weight_loader=None,
        )
        _stub(
            "gllm.models.weight_utils",
            get_tensor_from_dict=lambda weights, k: weights.get(k),
        )

        mamba_path = (
            Path(__file__).parent.parent / "synapse_engine" / "gllm" / "models" / "mamba.py"
        )
        spec = importlib.util.spec_from_file_location("_mamba2_standalone", mamba_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


mamba = _load_mamba()


class _FakeQuantConfig(dict):
    pass


class _FakeLinear:
    """Stand-in for a real MergedColumnParallelLinear with quantized params
    registered — only the attributes the loader functions actually read."""

    def __init__(self, quant_config=None):
        self.quant_config = _FakeQuantConfig(quant_config) if quant_config else None

    def set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


class _FakeMixer:
    def __init__(self, intermediate_size, conv_dim, num_heads, in_proj):
        self.intermediate_size = intermediate_size
        self.conv_dim = conv_dim
        self.num_heads = num_heads
        self.in_proj = in_proj


# ---------------------------------------------------------------------------
# _split_tp_concat / _require_divisible — pure index math
# ---------------------------------------------------------------------------

def _reference_split_tp_chunk(full, sizes, dim, tp_size, rank):
    segs = torch.split(full, sizes, dim=dim)
    chunks = []
    for seg in segs:
        c = seg.shape[dim] // tp_size
        idx = [slice(None)] * seg.dim()
        idx[dim] = slice(rank * c, (rank + 1) * c)
        chunks.append(seg[tuple(idx)])
    return torch.cat(chunks, dim=dim)


@pytest.mark.parametrize("tp_size,rank", [(1, 0), (4, 0), (4, 2), (4, 3)])
def test_split_tp_concat_dim0_matches_reference(tp_size, rank):
    torch.manual_seed(0)
    sizes = [128, 200, 24]  # gate, xBC, dt-like widths, divisible by 4
    full = torch.randn(sum(sizes), 64)
    got = mamba._split_tp_concat(full, sizes, 0, tp_size, rank)
    want = _reference_split_tp_chunk(full, sizes, 0, tp_size, rank)
    assert torch.equal(got, want)


@pytest.mark.parametrize("tp_size,rank", [(1, 0), (4, 1), (4, 3)])
def test_split_tp_concat_dim1_matches_reference(tp_size, rank):
    torch.manual_seed(1)
    sizes = [128, 200, 24]
    full = torch.randn(64, sum(sizes))
    got = mamba._split_tp_concat(full, sizes, 1, tp_size, rank)
    want = _reference_split_tp_chunk(full, sizes, 1, tp_size, rank)
    assert torch.equal(got, want)


def test_require_divisible_passes_when_clean():
    mamba._require_divisible([128, 256, 64], 8, "pack_factor")  # no raise


def test_require_divisible_raises_on_misaligned_boundary():
    with pytest.raises(NotImplementedError, match="pack_factor"):
        mamba._require_divisible([128, 200, 23], 8, "pack_factor")  # 23 % 8 != 0


# ---------------------------------------------------------------------------
# GPTQ in_proj: qweight unpacked on out-axis, qzeros/scales packed, g_idx shared
# ---------------------------------------------------------------------------

def test_load_gptq_in_proj_tp1_matches_source():
    ip, conv_dim, nh = 128, 256, 64  # all divisible by pack_factor=8 (4-bit)
    in_features, bits, group_size = 512, 4, 128
    pf = 32 // bits
    total_out = ip + conv_dim + nh
    num_groups = in_features // group_size

    torch.manual_seed(2)
    qweight_full = torch.randint(0, 2**31 - 1, (in_features // pf, total_out), dtype=torch.int32)
    qzeros_full = torch.randint(0, 2**31 - 1, (num_groups, total_out // pf), dtype=torch.int32)
    scales_full = torch.randn(num_groups, total_out)
    g_idx_full = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)

    in_proj = _FakeLinear({"quant_method": "gptq"}).set(
        gptq_pack_factor=pf,
        qweight=torch.nn.Parameter(torch.zeros_like(qweight_full), requires_grad=False),
        qzeros=torch.nn.Parameter(torch.zeros_like(qzeros_full), requires_grad=False),
        scales=torch.nn.Parameter(torch.zeros_like(scales_full), requires_grad=False),
        g_idx=torch.nn.Parameter(torch.zeros_like(g_idx_full), requires_grad=False),
    )
    layer = _FakeMixer(ip, conv_dim, nh, in_proj)
    weights = {
        "backbone.layers.0.mixer.in_proj.qweight": qweight_full,
        "backbone.layers.0.mixer.in_proj.qzeros": qzeros_full,
        "backbone.layers.0.mixer.in_proj.scales": scales_full,
        "backbone.layers.0.mixer.in_proj.g_idx": g_idx_full,
    }

    mamba._load_mamba2_in_proj_gptq_awq(layer, "backbone.layers.0.mixer", weights, tp_size=1, rank=0)

    assert torch.equal(in_proj.qweight.data, qweight_full)
    assert torch.equal(in_proj.qzeros.data, qzeros_full)
    assert torch.equal(in_proj.scales.data, scales_full)
    assert torch.equal(in_proj.g_idx.data, g_idx_full)


def test_load_gptq_in_proj_tp2_slices_each_segment_independently():
    ip, conv_dim, nh = 128, 256, 64
    in_features, bits, group_size = 512, 4, 128
    pf = 32 // bits
    total_out = ip + conv_dim + nh
    num_groups = in_features // group_size
    tp_size, rank = 2, 1

    torch.manual_seed(3)
    qweight_full = torch.randint(0, 2**31 - 1, (in_features // pf, total_out), dtype=torch.int32)
    scales_full = torch.randn(num_groups, total_out)

    in_proj = _FakeLinear({"quant_method": "gptq"}).set(
        gptq_pack_factor=pf,
        qweight=torch.nn.Parameter(torch.zeros(in_features // pf, total_out // tp_size, dtype=torch.int32), requires_grad=False),
        qzeros=torch.nn.Parameter(torch.zeros(num_groups, (total_out // tp_size) // pf, dtype=torch.int32), requires_grad=False),
        scales=torch.nn.Parameter(torch.zeros(num_groups, total_out // tp_size), requires_grad=False),
        g_idx=torch.nn.Parameter(torch.zeros(in_features, dtype=torch.int32), requires_grad=False),
    )
    layer = _FakeMixer(ip, conv_dim, nh, in_proj)
    weights = {
        "backbone.layers.0.mixer.in_proj.qweight": qweight_full,
        "backbone.layers.0.mixer.in_proj.qzeros": torch.randint(0, 2**31 - 1, (num_groups, total_out // pf), dtype=torch.int32),
        "backbone.layers.0.mixer.in_proj.scales": scales_full,
        "backbone.layers.0.mixer.in_proj.g_idx": torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32),
    }

    mamba._load_mamba2_in_proj_gptq_awq(layer, "backbone.layers.0.mixer", weights, tp_size=tp_size, rank=rank)

    want_qweight = mamba._split_tp_concat(qweight_full, [ip, conv_dim, nh], 1, tp_size, rank)
    want_scales = mamba._split_tp_concat(scales_full, [ip, conv_dim, nh], 1, tp_size, rank)
    assert torch.equal(in_proj.qweight.data, want_qweight)
    assert torch.equal(in_proj.scales.data, want_scales)
    # g_idx: shared across ranks/segments, never sliced.
    assert torch.equal(in_proj.g_idx.data, weights["backbone.layers.0.mixer.in_proj.g_idx"])


# ---------------------------------------------------------------------------
# AWQ in_proj: qweight/qzeros BOTH packed on out-axis, no g_idx
# ---------------------------------------------------------------------------

def test_load_awq_in_proj_tp1_matches_source():
    ip, conv_dim, nh = 128, 256, 64  # all divisible by pack_factor=8 (4-bit)
    in_features, bits, group_size = 512, 4, 128
    pf = 32 // bits
    total_out = ip + conv_dim + nh
    num_groups = in_features // group_size

    torch.manual_seed(4)
    qweight_full = torch.randint(0, 2**31 - 1, (in_features, total_out // pf), dtype=torch.int32)
    qzeros_full = torch.randint(0, 2**31 - 1, (num_groups, total_out // pf), dtype=torch.int32)
    scales_full = torch.randn(num_groups, total_out)

    in_proj = _FakeLinear({"quant_method": "awq"}).set(
        awq_pack_factor=pf,
        qweight=torch.nn.Parameter(torch.zeros_like(qweight_full), requires_grad=False),
        qzeros=torch.nn.Parameter(torch.zeros_like(qzeros_full), requires_grad=False),
        scales=torch.nn.Parameter(torch.zeros_like(scales_full), requires_grad=False),
    )
    layer = _FakeMixer(ip, conv_dim, nh, in_proj)
    weights = {
        "backbone.layers.0.mixer.in_proj.qweight": qweight_full,
        "backbone.layers.0.mixer.in_proj.qzeros": qzeros_full,
        "backbone.layers.0.mixer.in_proj.scales": scales_full,
    }

    mamba._load_mamba2_in_proj_gptq_awq(layer, "backbone.layers.0.mixer", weights, tp_size=1, rank=0)

    assert torch.equal(in_proj.qweight.data, qweight_full)
    assert torch.equal(in_proj.qzeros.data, qzeros_full)
    assert torch.equal(in_proj.scales.data, scales_full)
    assert not hasattr(in_proj, "g_idx") or in_proj.g_idx is None


def test_load_awq_in_proj_raises_on_misaligned_pack_boundary():
    # num_heads=23 is not divisible by pack_factor=8 -> qweight can't be
    # column-sliced at a whole-word boundary for this segment.
    ip, conv_dim, nh = 128, 256, 23
    in_features, bits, group_size = 512, 4, 128
    pf = 32 // bits
    total_out = ip + conv_dim + nh
    num_groups = in_features // group_size

    in_proj = _FakeLinear({"quant_method": "awq"}).set(
        awq_pack_factor=pf,
        qweight=torch.nn.Parameter(torch.zeros(in_features, total_out // pf, dtype=torch.int32), requires_grad=False),
        qzeros=torch.nn.Parameter(torch.zeros(num_groups, total_out // pf, dtype=torch.int32), requires_grad=False),
        scales=torch.nn.Parameter(torch.zeros(num_groups, total_out), requires_grad=False),
    )
    layer = _FakeMixer(ip, conv_dim, nh, in_proj)
    weights = {
        "backbone.layers.0.mixer.in_proj.qweight": torch.zeros(in_features, total_out // pf, dtype=torch.int32),
        "backbone.layers.0.mixer.in_proj.qzeros": torch.zeros(num_groups, total_out // pf, dtype=torch.int32),
        "backbone.layers.0.mixer.in_proj.scales": torch.zeros(num_groups, total_out),
    }
    with pytest.raises(NotImplementedError, match="pack_factor"):
        mamba._load_mamba2_in_proj_gptq_awq(layer, "backbone.layers.0.mixer", weights, tp_size=1, rank=0)


# ---------------------------------------------------------------------------
# FP8 in_proj: weight is byte-per-value (reuses plain split), block-quant
# scale is a real per-block grid, non-block scale is a broadcast scalar
# ---------------------------------------------------------------------------

def test_load_fp8_block_quant_in_proj():
    ip, conv_dim, nh = 128, 256, 128  # all divisible by block_n=128
    in_features, block_n, block_k = 512, 128, 128
    total_out = ip + conv_dim + nh

    torch.manual_seed(5)
    weight_full = torch.randn(total_out, in_features).to(torch.float8_e4m3fn)
    scale_inv_full = torch.rand(total_out // block_n, in_features // block_k)

    in_proj = _FakeLinear({"quant_method": "fp8"}).set(
        block_quant=True,
        weight_block_size=(block_n, block_k),
        weight=torch.nn.Parameter(torch.zeros_like(weight_full), requires_grad=False),
        weight_scale_inv=torch.nn.Parameter(torch.zeros_like(scale_inv_full), requires_grad=False),
    )
    layer = _FakeMixer(ip, conv_dim, nh, in_proj)
    weights = {
        "backbone.layers.0.mixer.in_proj.weight": weight_full,
        "backbone.layers.0.mixer.in_proj.weight_scale_inv": scale_inv_full,
    }

    mamba._load_mamba2_in_proj_fp8(layer, "backbone.layers.0.mixer", weights, tp_size=1, rank=0)

    assert torch.equal(in_proj.weight.data.float(), weight_full.float())
    assert torch.equal(in_proj.weight_scale_inv.data, scale_inv_full)


def test_load_fp8_non_block_broadcasts_single_scale():
    ip, conv_dim, nh = 128, 256, 64
    in_features = 512
    total_out = ip + conv_dim + nh

    torch.manual_seed(6)
    weight_full = torch.randn(total_out, in_features).to(torch.float8_e4m3fn)
    scale_src = torch.tensor([0.0273])  # single calibrated scalar on disk

    in_proj = _FakeLinear({"quant_method": "fp8"}).set(
        block_quant=False,
        weight=torch.nn.Parameter(torch.zeros_like(weight_full), requires_grad=False),
        weight_scale=torch.nn.Parameter(torch.zeros(3), requires_grad=False),
        input_scale=None,
    )
    layer = _FakeMixer(ip, conv_dim, nh, in_proj)
    weights = {
        "backbone.layers.0.mixer.in_proj.weight": weight_full,
        "backbone.layers.0.mixer.in_proj.weight_scale": scale_src,
    }

    mamba._load_mamba2_in_proj_fp8(layer, "backbone.layers.0.mixer", weights, tp_size=1, rank=0)

    assert torch.allclose(in_proj.weight_scale.data, torch.full((3,), 0.0273))


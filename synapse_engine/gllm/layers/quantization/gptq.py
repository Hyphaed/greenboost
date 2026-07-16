"""GPTQ quantized linear support.

GreenBoost local patch — see synapse_engine/NOTICE. Uses gptqmodel's own
tested triton dequant+matmul kernel
(``gptqmodel.nn_modules.triton_utils.dequant``) rather than reimplementing
GPTQ's bit-unpacking/group-index math by hand: that math has real
correctness traps (desc_act activation-order reordering, the v1-vs-v2
qzeros +1 offset convention, non-power-of-two 3-bit packing) that
gptqmodel's maintainers have already solved and tested against real
checkpoints — safer to reuse than to re-derive.

Tensor layout (standard AutoGPTQ/HF format, matches what a real GPTQ
checkpoint stores on disk byte-for-byte, so the weight loader's default
name-matched ``.copy_()`` path needs no special casing for a
non-fused layer):
    qweight: [in_features // pack_factor, out_features]   int32
    qzeros:  [num_groups, out_features // pack_factor]     int32
    scales:  [num_groups, out_features]                    params_dtype
    g_idx:   [in_features]                                 int32

2/4/8-bit only for now — 3-bit's qzeros packing is (out_features // 32) * 3
words per group (not a simple pack_factor divide), a real format difference
gptqmodel's own kernel special-cases; add it as a dedicated path if/when a
real 3-bit checkpoint needs serving rather than guess at it now.
"""
from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

_SUPPORTED_BITS = (2, 4, 8)


def pack_factor(bits: int) -> int:
    return 32 // bits


def gptq_create_weights(layer, input_size_per_partition: int,
                        output_partition_sizes: list[int],
                        params_dtype: torch.dtype) -> None:
    """Register qweight/qzeros/scales/g_idx as Parameters on `layer` —
    mirrors LinearBase.create_weights' fp8 branch: the weight loader fills
    these afterwards by exact-name checkpoint copy, same as any other
    parameter `run_weight_loader` finds via `model.named_parameters()`."""
    qc = layer.quant_config
    bits = int(qc["bits"])
    if bits not in _SUPPORTED_BITS:
        raise NotImplementedError(
            f"GPTQ {bits}-bit is not supported by this engine's torch-core loader "
            f"yet ({_SUPPORTED_BITS} only) — use the fallback server for this checkpoint.")
    group_size = int(qc.get("group_size") or -1)
    if group_size <= 0:
        group_size = input_size_per_partition
    pf = pack_factor(bits)
    out_features = sum(output_partition_sizes)
    in_features = input_size_per_partition
    num_groups = -(-in_features // group_size)  # ceil div

    qweight = Parameter(
        torch.zeros(in_features // pf, out_features, dtype=torch.int32),
        requires_grad=False)
    qzeros = Parameter(
        torch.zeros(num_groups, out_features // pf, dtype=torch.int32),
        requires_grad=False)
    scales = Parameter(
        torch.zeros(num_groups, out_features, dtype=params_dtype),
        requires_grad=False)
    g_idx = Parameter(
        torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32),
        requires_grad=False)

    layer.register_parameter("qweight", qweight)
    layer.register_parameter("qzeros", qzeros)
    layer.register_parameter("scales", scales)
    layer.register_parameter("g_idx", g_idx)
    # forward() always calls self.quant_method(input, self.weight, bias=...) —
    # "weight" must exist as SOME attribute for that call signature, even
    # though GPTQ's actual dequant reads qweight/qzeros/scales/g_idx instead.
    layer.register_parameter("weight", None)
    layer.bits = bits
    layer.gptq_group_size = group_size
    layer.gptq_pack_factor = pf


# v1->v2 qzeros correction constants, int32-packed — copied verbatim from
# gptqmodel.utils.model.convert_gptq_v1_to_v2_format_module (its own proven
# values, not re-derived) rather than trusting a from-scratch bit-pattern
# computation for something this easy to get subtly wrong.
_V1_TO_V2_ZERO_OFFSET_INT32 = {
    2: 0b01010101010101010101010101010101,
    4: 0b00010001000100010001000100010001,
    8: 0b00000001000000010000000100000001,
}


def _maybe_convert_v1_zeros(layer) -> None:
    """One-time v1->v2 qzeros correction. Classic AutoGPTQ ("v1") checkpoints
    — the vast majority of public GPTQ checkpoints on HF Hub, including this
    engine's own reference test target
    (TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ) — stored `qzeros -= 1` before
    serialization; gptqmodel's own dequant kernel (which gptq_linear_method
    calls) expects the un-offset ("v2") convention, so a v1 checkpoint's
    RAW qzeros dequants to a systematically wrong zero-point — not a crash,
    silently wrong weights (confirmed live: garbage/out-of-vocab-range
    token ids sampled from a v1 checkpoint served without this correction).

    Skipped when the checkpoint's own quant_config declares an explicit v2
    format. Runs lazily on the first real forward pass — this can't run at
    create_weights time, which only allocates placeholder shapes; the REAL
    qzeros values only exist once the weight loader has filled them."""
    if getattr(layer, "_gptq_zeros_v1_converted", False):
        return
    layer._gptq_zeros_v1_converted = True
    qc = layer.quant_config
    fmt = str(qc.get("checkpoint_format") or qc.get("format") or "").lower()
    if fmt in ("gptq_v2", "v2"):
        return  # already v2 — no correction needed
    offset = _V1_TO_V2_ZERO_OFFSET_INT32.get(layer.bits)
    if offset is not None:
        layer.qzeros.data += offset


def gptq_linear_method(input_: torch.Tensor, weight, bias=None, *, layer) -> torch.Tensor:
    """quant_method callable — dispatch_quant_method binds `layer` via
    functools.partial, matching the (input, weight, bias=) signature every
    gLLM quant_method uses. `weight` is unused (kept only for call-signature
    parity with the other quant methods, which pass self.weight there)."""
    from gptqmodel.nn_modules.triton_utils.dequant import quant_matmul
    _maybe_convert_v1_zeros(layer)
    bits = layer.bits
    out = quant_matmul(
        input_, layer.qweight, layer.scales, layer.qzeros, layer.g_idx,
        bits=bits, pack_bits=32, maxq=(1 << bits) - 1,
    )
    if bias is not None:
        out = out + bias
    return out.to(input_.dtype)

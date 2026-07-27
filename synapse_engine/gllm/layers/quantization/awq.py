"""AWQ quantized linear support.

GreenBoost local patch — see synapse_engine/NOTICE. Reuses the same
gptqmodel Triton dequant+matmul kernel already vendored for GPTQ
(``gptqmodel.nn_modules.triton_utils.dequant.quant_matmul``) instead of
adding a second kernel: AWQ and GPTQ checkpoints store the identical
logical (in_features x out_features) matrix, group-quantized the same way,
so once AWQ's two real format differences are corrected, GPTQ's kernel
dequants an AWQ checkpoint correctly.

The two real differences from GPTQ (checked against AutoAWQ's own layout
and vLLM's ``auto_awq.py`` reference conversion):
  1. **Packing axis**: AWQ packs ``qweight`` along the OUTPUT dim
     (``[in_features, out_features // pack_factor]``); GPTQ (and
     gptqmodel's kernel) expects it packed along the INPUT dim
     (``[in_features // pack_factor, out_features]``). ``qzeros`` already
     matches GPTQ's axis convention on disk (``[num_groups,
     out_features // pack_factor]``) — no axis change needed there.
  2. **Bit order within a packed word**: AWQ stores the ``pack_factor``
     sub-values in the interleaved order ``[0,4,1,5,2,6,3,7]`` (for 4-bit;
     the general pattern is "even slots then odd slots"), not GPTQ's plain
     sequential ``[0,1,...,pack_factor-1]``.
No g_idx: AWQ never reorders activations (no ``desc_act`` equivalent), so
g_idx is always the trivial ``arange(in_features) // group_size`` — it is
computed once and kept as a plain (unregistered) tensor attribute, never a
checkpoint-loaded Parameter, so the weight loader never looks for a
``.g_idx`` key that no AWQ checkpoint actually has on disk. AWQ also has no
v1/v2 zero-point offset quirk (that convention is GPTQ-specific): its raw
qzeros already encode the true zero point directly, which is exactly what
gptqmodel's kernel expects.

4-bit only for now — this is the only bit width AutoAWQ itself ever
produces; add another if/when a real checkpoint needs it. ``zero_point:
false`` (AWQ's symmetric/no-zero-point variant) is rejected the same way —
no real checkpoint exercising it yet.
"""
from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

_SUPPORTED_BITS = (4,)

# "Reverse" the AWQ packing order back to sequential: AWQ places the value
# for packed-slot p at position _AWQ_PACK_ORDER[p] rather than p itself.
# Copied from vLLM's auto_awq.py (_REVERSE_AWQ_PACK_ORDER) — a previously
# solved/tested constant, not re-derived from scratch.
_AWQ_PACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def pack_factor(bits: int) -> int:
    return 32 // bits


def awq_create_weights(layer, input_size_per_partition: int,
                       output_partition_sizes: list[int],
                       params_dtype: torch.dtype) -> None:
    """Register qweight/qzeros/scales as Parameters in AWQ's own on-disk
    shape/layout, so the weight loader's exact-name checkpoint copy (and
    the existing GPTQ-named column-concat fusion helpers, which are
    generic over any ``[rows, out_features(_packed)]`` tensor regardless
    of quant method) need no AWQ-specific casing. g_idx is NOT registered
    here — see module docstring."""
    qc = layer.quant_config
    bits = int(qc.get("bits") or qc.get("w_bit") or 0)
    if bits not in _SUPPORTED_BITS:
        raise NotImplementedError(
            f"AWQ {bits}-bit is not supported by this engine's torch-core loader "
            f"yet ({_SUPPORTED_BITS} only) — use the fallback server for this checkpoint.")
    if qc.get("zero_point", True) is False:
        raise NotImplementedError(
            "AWQ zero_point=false (symmetric) checkpoints are not supported by this "
            "engine's torch-core loader yet — use the fallback server for this checkpoint.")
    group_size = int(qc.get("group_size") or qc.get("q_group_size") or -1)
    if group_size <= 0:
        group_size = input_size_per_partition
    pf = pack_factor(bits)
    out_features = sum(output_partition_sizes)
    in_features = input_size_per_partition
    num_groups = -(-in_features // group_size)  # ceil div

    qweight = Parameter(
        torch.zeros(in_features, out_features // pf, dtype=torch.int32),
        requires_grad=False)
    qzeros = Parameter(
        torch.zeros(num_groups, out_features // pf, dtype=torch.int32),
        requires_grad=False)
    scales = Parameter(
        torch.zeros(num_groups, out_features, dtype=params_dtype),
        requires_grad=False)

    layer.register_parameter("qweight", qweight)
    layer.register_parameter("qzeros", qzeros)
    layer.register_parameter("scales", scales)
    # forward() always calls self.quant_method(input, self.weight, bias=...) —
    # "weight" must exist as SOME attribute for that call signature, even
    # though AWQ's actual dequant reads qweight/qzeros/scales instead.
    layer.register_parameter("weight", None)
    layer.bits = bits
    layer.awq_group_size = group_size
    layer.awq_pack_factor = pf
    # Trivial, checkpoint-independent g_idx (no desc_act in AWQ). Deliberately
    # a plain tensor attribute, NOT register_parameter/register_buffer, so it
    # never appears in named_parameters()/state_dict() and the weight loader
    # never tries to fill it from a checkpoint key that doesn't exist.
    layer.g_idx = torch.tensor(
        [i // group_size for i in range(in_features)], dtype=torch.int32)


def _maybe_convert_awq_layout(layer) -> None:
    """One-time AWQ -> GPTQ-standard layout conversion (bit order + qweight
    packing axis). Runs lazily on the first real forward pass — the REAL
    packed values only exist once the weight loader has filled them."""
    if getattr(layer, "_awq_layout_converted", False):
        return
    layer._awq_layout_converted = True
    bits = layer.bits
    pf = layer.awq_pack_factor
    mask = (1 << bits) - 1
    device = layer.qweight.device
    reverse_order = torch.tensor(_AWQ_PACK_ORDER, dtype=torch.long, device=device)
    shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=device)
    # g_idx is a plain tensor attribute (see awq_create_weights), not a
    # registered Parameter/buffer, so it was never swept along by the
    # model's own .to(device) move — do it here, once, before first use.
    layer.g_idx = layer.g_idx.to(device)

    # --- qweight: (K, N/pf) packed dim=1, AWQ order -> (K/pf, N) packed dim=0,
    # sequential order (gptqmodel's expected GPTQ-standard layout).
    qw = layer.qweight.data
    K, N_packed = qw.shape
    N = N_packed * pf
    unpacked = (qw.unsqueeze(-1) >> shifts) & mask       # (K, N/pf, pf)
    unpacked = unpacked[:, :, reverse_order]              # fix bit order
    unpacked = unpacked.reshape(K, N)                     # (K, N) true values
    unpacked = unpacked.reshape(K // pf, pf, N)
    new_qw = (unpacked.to(torch.int32) << shifts[None, :, None]).sum(
        dim=1, dtype=torch.int32)
    layer.qweight.data = new_qw.contiguous()

    # --- qzeros: (G, N/pf) packed dim=1, AWQ order -> same shape, sequential
    # order (axis already matches GPTQ's qzeros convention, only bit order
    # needs fixing).
    qz = layer.qzeros.data
    G, _ = qz.shape
    unpacked_zp = (qz.unsqueeze(-1) >> shifts) & mask     # (G, N/pf, pf)
    unpacked_zp = unpacked_zp[:, :, reverse_order]
    unpacked_zp = unpacked_zp.reshape(G, N)
    unpacked_zp = unpacked_zp.reshape(G, N // pf, pf)
    new_qz = (unpacked_zp.to(torch.int32) << shifts[None, None, :]).sum(
        dim=2, dtype=torch.int32)
    layer.qzeros.data = new_qz.contiguous()


def awq_linear_method(input_: torch.Tensor, weight, bias=None, *, layer) -> torch.Tensor:
    """quant_method callable — dispatch_quant_method binds `layer` via
    functools.partial, matching the (input, weight, bias=) signature every
    gLLM quant_method uses. `weight` is unused (kept only for call-signature
    parity with the other quant methods)."""
    from gptqmodel.nn_modules.triton_utils.dequant import quant_matmul
    _maybe_convert_awq_layout(layer)
    bits = layer.bits
    out = quant_matmul(
        input_, layer.qweight, layer.scales, layer.qzeros, layer.g_idx,
        bits=bits, pack_bits=32, maxq=(1 << bits) - 1,
    )
    if bias is not None:
        out = out + bias
    return out.to(input_.dtype)

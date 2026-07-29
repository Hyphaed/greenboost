"""Mamba-2: pure selective-SSM decoder (Gu & Dao,
arXiv:2312.00752 §3.3/App. D for the recurrence; the "SSD" chunked-scan
reformulation is state-spaces/mamba's v2 release, vendored at
gllm.layers.ops.mamba.ssd_combined / mamba2_decode -- see synapse_engine/
NOTICE for the vendoring note).

Every decoder layer is the SAME mixer (no interleaved full-attention layers,
unlike Qwen3.5's hybrid GDN -- see models/qwen3_5.py for that case, which
this file's structure otherwise follows closely). Consequently:

* ``num_kv_layers`` is always 0 -- there is no paged KV cache at all, and
  ``FlashAttention``/``segment.k_cache``/``v_cache`` are never touched.
* Every layer index is an SSM layer index (``ssm_layer_id == layer_id``
  after PP slicing), unlike the hybrid case's two parallel counters.

Architectural cheat-sheet (matches the real HF ``AntonV/mamba2-130m-hf``
checkpoint's config + safetensors layout, verified against the actual
tensors, not assumed from the paper alone):

* ``in_proj``: hidden_size -> [gate(intermediate_size) | xBC(conv_dim) |
  dt(num_heads)], ONE fused Linear in the checkpoint (unlike GDN's four
  separate checkpoint tensors) -- ``intermediate_size = expand *
  hidden_size``, ``conv_dim = intermediate_size + 2 * n_groups * state_size``.
* ``conv1d``: depthwise causal conv over the ``xBC`` slice only (gate is
  NOT convolved), same vendored Triton kernel GDN uses
  (``gllm.layers.ops.mamba.causal_conv1d_{fn,update}``).
* After the conv, ``xBC`` splits into ``x(intermediate_size) |
  B(n_groups*state_size) | C(n_groups*state_size)``.
* ``A_log``/``D``/``dt_bias``: per-head scalars, shape ``(num_heads,)`` --
  Mamba-2's SSD formulation uses a scalar decay per head (not a per-
  (head,state) matrix like Mamba-1), which is exactly what lets the SSD
  chunked-scan reformulation exist at all (see the paper's "SSD" section).
  ``A = -exp(A_log)``.
* Gated output norm (``norm.weight``, shape ``(intermediate_size,)`` --
  the FULL width, unlike GDN's per-head-only ``(head_v_dim,)`` weight):
  a GLOBAL (not grouped) gated RMSNorm computing ``norm(x * silu(gate))``
  -- gate applied BEFORE norm, variance taken over the full width. Verified
  directly against HF's real ``MambaRMSNormGated.forward()``, not assumed
  from GDN's superficially-similar-looking ``norm_before_gate``/
  ``group_size`` convention (which is the OPPOSITE: ``norm(x) * silu(gate)``,
  per-head grouped -- copying it verbatim was the actual bug behind the
  2026-07-29 next-token divergence this file's Mixer.__init__ documents).
  Reuses the same vendored ``gllm.layers.ops.fla.RMSNormGated`` class GDN
  uses, just with different (correct, for this architecture) constructor
  arguments: ``group_size=None, norm_before_gate=False``.
* ``out_proj``: intermediate_size -> hidden_size.

Recurrent state (``conv_state`` + ``temporal_state``) lives in the same
:class:`gllm.memory_manager.SSMSegment` working pool GDN uses -- Mamba-2's
state is head-shaped ``(num_heads, head_dim, state_size)``, which maps onto
``SSMCacheConfig.temporal_state_shape_per_slot()``
(``(num_v_heads, head_v_dim, head_k_dim)``) with NO changes to that
dataclass (verified by reading it, not assumed): substitute
``num_v_heads=num_heads``, ``head_v_dim=head_dim``, ``head_k_dim=state_size``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn

from gllm.dist_utils import get_pp_layers, get_tp_rank, get_tp_size, is_first_pp_rank, is_last_pp_rank
from gllm.input_data import InputData
from gllm.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from gllm.layers.ops.fla import RMSNormGated
from gllm.layers.ops.mamba import (
    causal_conv1d_fn,
    causal_conv1d_update,
    mamba_chunk_scan_combined_varlen,
    selective_state_update,
)
from gllm.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from gllm.memory_manager import SSMCacheConfig
from gllm.models.weight_loader import LoadContext, WeightRule, contains, h_proj_dim0, h_proj_dim1, run_weight_loader
from gllm.models.weight_utils import get_tensor_from_dict


def _mamba2_intermediate_size(config) -> int:
    """``intermediate_size`` when the config spells it out directly (some
    downstream ports add the field); otherwise derive it the way the
    reference implementation does, ``expand * hidden_size``."""
    explicit = getattr(config, "intermediate_size", None)
    if explicit:
        return int(explicit)
    return int(getattr(config, "expand", 2)) * config.hidden_size


def _compute_chunk_metadata(
    query_start_loc: torch.Tensor, chunk_size: int, device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(cu_chunk_seqlens, last_chunk_indices, seq_idx)`` for the SSD
    chunked-scan prefill kernel, derived from the packed varlen
    ``query_start_loc`` boundaries (the same per-sequence boundaries
    FlashAttention/GDN already use).

    Each real sequence is split into its own ``chunk_size``-token pseudo-
    chunks (the last one clipped to the sequence's actual remaining
    length) -- sequences never share a pseudo-chunk, so a new sequence
    always starts a fresh chunk regardless of chunk-size alignment.
    ``cu_chunk_seqlens`` is the flat cumulative chunk-boundary array over
    the WHOLE packed token axis (length total_nchunks+1); ``seq_idx`` maps
    each pseudo-chunk to its owning real sequence (length total_nchunks);
    ``last_chunk_indices`` is, per real sequence, the index of its last
    pseudo-chunk (length batch) -- used to pull each sequence's final SSM
    state out of the kernel's per-chunk state array.

    vLLM precomputes the equivalent once per batch in its own attention-
    metadata builder (a scheduler-integrated component gLLM doesn't have
    for the SSD chunk boundary); this is instead computed fresh each
    prefill call. Cheap: plain Python int arithmetic over `batch` entries
    (not per-token), completely off the Triton hot path.
    """
    starts = query_start_loc.tolist()
    batch = len(starts) - 1
    cu_chunk = [0]
    seq_idx_list: List[int] = []
    last_chunk_idx: List[int] = []
    chunk_counter = 0
    for i in range(batch):
        s, e = starts[i], starts[i + 1]
        length = e - s
        n_chunks_i = max(1, -(-length // chunk_size))  # ceil div, >=1 even for an empty seq
        pos = s
        for _ in range(n_chunks_i):
            pos = min(pos + chunk_size, e)
            cu_chunk.append(pos)
            seq_idx_list.append(i)
        chunk_counter += n_chunks_i
        last_chunk_idx.append(chunk_counter - 1)
    return (
        torch.tensor(cu_chunk, device=device, dtype=torch.int32),
        torch.tensor(last_chunk_idx, device=device, dtype=torch.int32),
        torch.tensor(seq_idx_list, device=device, dtype=torch.int32),
    )


class Mamba2Mixer(nn.Module):
    """Mamba-2 selective-SSM mixer (the SSD formulation). Every field name
    and shape here is verified against a real checkpoint -- see the module
    docstring."""

    def __init__(self, config, layer_id: int, ssm_layer_id: int, quant_config=None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id  # GLOBAL decoder index -- checkpoint prefix lookup
        self.ssm_layer_id = ssm_layer_id  # LOCAL SSMSegment pool slot (0..num_ssm_layers-1)

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.state_size = config.state_size
        self.n_groups = config.n_groups
        self.conv_kernel_size = config.conv_kernel
        self.chunk_size = int(getattr(config, "chunk_size", 256))
        self.layer_norm_epsilon = float(getattr(config, "layer_norm_epsilon", 1e-5))
        self.time_step_limit: Tuple[float, float] = tuple(
            getattr(config, "time_step_limit", (0.0, float("inf")))
        )
        use_bias = bool(getattr(config, "use_bias", False))
        use_conv_bias = bool(getattr(config, "use_conv_bias", True))
        self.activation = getattr(config, "hidden_act", "silu")

        self.intermediate_size = _mamba2_intermediate_size(config)
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.state_size

        tp_size = get_tp_size()
        if self.num_heads % tp_size:
            raise ValueError(
                "Mamba-2 requires num_heads divisible by TP size: "
                f"tp_size={tp_size}, num_heads={self.num_heads}"
            )
        self.tp_num_heads = self.num_heads // tp_size
        # Groups behave like GQA's kv-head count: replicate a group across
        # ranks when there are fewer groups than TP ranks (n_groups=1 is the
        # common case), same pattern as Qwen3_5FullAttention.num_kv_heads.
        self.tp_n_groups = max(1, self.n_groups // tp_size)
        self.tp_intermediate_size = self.intermediate_size // tp_size
        self.tp_conv_dim = self.tp_intermediate_size + 2 * self.tp_n_groups * self.state_size

        # in_proj: ONE fused Linear in the checkpoint (verified against the
        # real safetensors header, not GDN's four-separate-tensor case) --
        # output layout [gate(intermediate_size) | xBC(conv_dim) |
        # dt(num_heads)], each segment independently TP-sharded. See
        # _load_mamba2_layer_weights for how a single checkpoint tensor is
        # split into these three TP-sharded segments.
        self.in_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.intermediate_size, self.conv_dim, self.num_heads],
            bias=use_bias,
            quant_config=quant_config,
        )

        # Depthwise causal conv over the xBC slice only -- same construction
        # pattern as Qwen3_5GatedDeltaNet's conv1d (a Linear-shaped weight
        # matching causal_conv1d_{fn,update}'s (dim, width) contract, then
        # unsqueezed to (dim, 1, width) so the PARAMETER shape matches what
        # HF-format checkpoints store for nn.Conv1d).
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=use_conv_bias,
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.dt_bias = nn.Parameter(torch.ones(self.tp_num_heads))
        self.A_log = nn.Parameter(torch.empty(self.tp_num_heads, dtype=torch.float32))
        self.D = nn.Parameter(torch.ones(self.tp_num_heads))

        # Gated RMSNorm. CORRECTED 2026-07-29 against HF's actual reference
        # (transformers/models/mamba2/modeling_mamba2.py::MambaRMSNormGated,
        # fetched and read directly, not assumed) -- two things were wrong
        # in the original version of this constructor, found via a live
        # next-token divergence against the HF reference on the same
        # checkpoint+prompt (gLLM predicted " assertEquals", HF predicted
        # " a"/" in"/" the"):
        # 1. Gate/norm ORDER: HF computes norm(x * silu(gate)) -- multiply
        #    by the gate FIRST, then normalize the gated result. The
        #    original code copied GDN's Qwen3_5GatedDeltaNet convention
        #    (norm_before_gate=True -> norm(x) * silu(gate), gate AFTER
        #    norm) verbatim without re-deriving it for Mamba-2 specifically
        #    -- the two architectures' gated norms are NOT the same
        #    convention despite superficially similar names.
        # 2. Grouping: HF's variance is `hidden_states.pow(2).mean(-1)` over
        #    the FULL last dim (global), not per-head-grouped -- confirmed
        #    by the norm.weight shape being the full intermediate_size
        #    (every head has its own distinct weight slice, unlike GDN's
        #    single per-head-shared weight, which is what motivated the
        #    original, wrong, group_size=head_dim guess).
        self.norm = RMSNormGated(
            self.tp_intermediate_size,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=False,
        )

        self.out_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=use_bias,
            quant_config=quant_config,
        )

    def _ssm_state_tensors(self, input_data: InputData):
        seg = input_data.memory_manager.ssm_segment
        return seg.conv_state[self.ssm_layer_id], seg.temporal_state[self.ssm_layer_id]

    def _maybe_snapshot_state(self, input_data: InputData, conv_state_working, ssm_state_working) -> None:
        """Identical to Qwen3_5GatedDeltaNet's snapshot path (same
        SSMSegment pool, same prefix-cache-hit restore contract) -- see that
        class's docstring for why the CPU-mirrored valid_mask check exists."""
        snap_targets = input_data.get_ssm_snapshot_write_slot_per_seq()
        if snap_targets is None:
            return
        seg = input_data.memory_manager.ssm_segment
        if seg.conv_state_snap is None:
            return
        snap_cpu = getattr(input_data, "ssm_snapshot_write_slot_per_seq_cpu", None)
        if snap_cpu is not None:
            if int(snap_cpu.amax()) < 0:
                return
            valid_mask = snap_targets >= 0
        else:
            valid_mask = snap_targets >= 0
            if not bool(valid_mask.any()):
                return
        src_slots = input_data.get_ssm_state_slot_per_seq()
        valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        src_idx = src_slots.index_select(0, valid_idx).to(torch.long)
        dst_idx = snap_targets.index_select(0, valid_idx).to(torch.long)
        seg.conv_state_snap[self.ssm_layer_id].index_copy_(
            0, dst_idx, conv_state_working.index_select(0, src_idx)
        )
        seg.temporal_state_snap[self.ssm_layer_id].index_copy_(
            0, dst_idx, ssm_state_working.index_select(0, src_idx)
        )

    def _is_decode_batch(self, input_data: InputData) -> bool:
        return getattr(input_data, "max_query_len", 1) == 1

    def forward(self, input_data: InputData, hidden_states: torch.Tensor):
        if not hasattr(input_data.memory_manager, "ssm_segment") or \
                input_data.memory_manager.ssm_segment is None:
            return torch.zeros_like(hidden_states)

        seq_len = hidden_states.shape[0]
        projected = self.in_proj(hidden_states)
        gate, xBC, dt = torch.split(
            projected, [self.tp_intermediate_size, self.tp_conv_dim, self.tp_num_heads], dim=-1
        )

        conv_state, ssm_state = self._ssm_state_tensors(input_data)
        cache_indices = input_data.get_ssm_state_slot_per_seq()
        has_initial_state = input_data.get_has_initial_state_per_seq()
        query_start_loc = input_data.get_query_start_loc()
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), -1)
        conv_bias = self.conv1d.bias

        A = -torch.exp(self.A_log.float())  # (tp_num_heads,) -- Mamba-2's scalar-per-head decay

        if self._is_decode_batch(input_data):
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                conv_weights,
                conv_bias,
                self.activation,
                conv_state_indices=cache_indices,
            )
            x, B, C = torch.split(
                xBC,
                [self.tp_intermediate_size, self.tp_n_groups * self.state_size,
                 self.tp_n_groups * self.state_size],
                dim=-1,
            )
            batch_size = x.shape[0]
            x_h = x.view(batch_size, self.tp_num_heads, self.head_dim)
            dt_h = dt.view(batch_size, self.tp_num_heads).unsqueeze(-1).expand(
                batch_size, self.tp_num_heads, self.head_dim
            ).contiguous()
            # Mamba-2's A/D are per-head scalars; selective_state_update's
            # generic interface wants them state-shaped -- expand (a view,
            # no copy) rather than materializing a redundant per-(dim,dstate)
            # copy.
            A_expanded = A.view(self.tp_num_heads, 1, 1).expand(
                self.tp_num_heads, self.head_dim, self.state_size
            )
            D_expanded = self.D.view(self.tp_num_heads, 1).expand(self.tp_num_heads, self.head_dim)
            B_h = B.view(batch_size, self.tp_n_groups, self.state_size)
            C_h = C.view(batch_size, self.tp_n_groups, self.state_size)
            out = torch.empty_like(x_h)
            is_blackwell = torch.cuda.get_device_capability(x.device)[0] >= 10
            selective_state_update(
                ssm_state,
                x_h,
                dt_h,
                A_expanded,
                B_h,
                C_h,
                D_expanded,
                self.dt_bias.view(self.tp_num_heads, 1).expand(self.tp_num_heads, self.head_dim),
                z=None,
                dt_softplus=True,
                state_batch_indices=cache_indices.unsqueeze(-1) if cache_indices is not None else None,
                out=out,
                is_blackwell=is_blackwell,
            )
            core_out = out.reshape(batch_size, -1)
        else:
            xBC_t = xBC.transpose(0, 1)  # [C_in, T]
            xBC_t = causal_conv1d_fn(
                xBC_t,
                conv_weights,
                conv_bias,
                conv_states=conv_state,
                query_start_loc=query_start_loc,
                seq_lens_cpu=getattr(input_data, "seq_lens_cpu", None),
                cache_indices=cache_indices,
                has_initial_state=has_initial_state,
                activation=self.activation,
            ).transpose(0, 1)[:seq_len]

            x, B, C = torch.split(
                xBC_t,
                [self.tp_intermediate_size, self.tp_n_groups * self.state_size,
                 self.tp_n_groups * self.state_size],
                dim=-1,
            )
            x_h = x.view(seq_len, self.tp_num_heads, self.head_dim)
            B_h = B.view(seq_len, self.tp_n_groups, self.state_size)
            C_h = C.view(seq_len, self.tp_n_groups, self.state_size)

            cu_chunk_seqlens, last_chunk_indices, seq_idx = _compute_chunk_metadata(
                query_start_loc, self.chunk_size, x_h.device
            )
            out = torch.empty_like(x_h)
            varlen_states = mamba_chunk_scan_combined_varlen(
                x_h,
                dt,
                A,
                B_h,
                C_h,
                self.chunk_size,
                query_start_loc,
                cu_chunk_seqlens,
                last_chunk_indices,
                seq_idx,
                out,
                D=self.D,
                z=None,
                dt_bias=self.dt_bias,
                initial_states=None,
                dt_softplus=True,
                dt_limit=self.time_step_limit,
                state_dtype=ssm_state.dtype,
            )
            ssm_state[cache_indices] = varlen_states.to(ssm_state.dtype, copy=False)
            self._maybe_snapshot_state(input_data, conv_state, ssm_state)
            core_out = out.reshape(seq_len, -1)

        core_out = self.norm(core_out, gate)
        return self.out_proj(core_out)


class Mamba2DecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int, ssm_layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        quant_config = getattr(config, "quantization_config", None)
        self.mixer = Mamba2Mixer(config, layer_id, ssm_layer_id, quant_config=quant_config)
        self.norm = _rmsnorm_for(config)

    def forward(self, input_data: InputData, hidden_states: torch.Tensor, residual: Optional[torch.Tensor]):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)
        hidden_states = self.mixer(input_data, hidden_states)
        return hidden_states, residual


def _rmsnorm_for(config):
    """The decoder-level (pre-mixer / final) norm. CORRECTED 2026-07-29: the
    original version of this function used GemmaRMSNorm purely for its
    convenient (x)->x / (x,residual)->(x,residual) call signature, without
    checking its actual NORMALIZATION MATH -- GemmaRMSNorm interprets its
    stored weight as (weight + 1) at runtime (correct for Qwen3.5, which is
    trained Gemma-style; see that class's own docstring), which is wrong for
    Mamba-2's real checkpoints (plain weight * normalize(x), verified
    directly against HF's real modeling_mamba2.py, which uses a standard
    RMSNorm with no +1 offset). Found live: a next-token divergence
    investigation (gLLM predicted " assertEquals"/"uilt" vs HF's sane " a"/
    " in"/" the" for the same prompt) traced to this exact substitution --
    every weight tensor matched exactly (in_proj/conv1d/A_log/D/dt_bias/
    norm.weight all verified allclose against the HF reference) EXCEPT this
    pre-mixer norm's OUTPUT, which diverged from HF's own norm(embed) despite
    an identical weight tensor -- because GemmaRMSNorm silently added 1 to
    every weight element before scaling. gllm.layers.layernorm.RMSNorm (the
    PLAIN variant, same file) has the identical call signature and is what
    this file should have used from the start."""
    from gllm.layers.layernorm import RMSNorm

    return RMSNorm(config.hidden_size, float(getattr(config, "layer_norm_epsilon", 1e-5)))


class Mamba2Model(nn.Module):
    """Pure-recurrent decoder stack. No ``_kv_layer_ids`` bookkeeping (unlike
    Qwen3_5Model) -- every layer is an SSM layer, so the PP-local layer index
    IS the SSM layer index directly."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        if is_first_pp_rank() or (getattr(config, "tie_word_embeddings", False) and is_last_pp_rank()):
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        self.start_layer, self.end_layer = get_pp_layers(config.num_hidden_layers)
        self.layers = nn.ModuleList(
            Mamba2DecoderLayer(config, layer_id=global_idx, ssm_layer_id=local_idx)
            for local_idx, global_idx in enumerate(range(self.start_layer, self.end_layer))
        )
        self.num_kv_layers = 0  # no attention layers at all
        self.num_ssm_layers = len(self.layers)
        self.ssm_layer_global_ids = list(range(self.start_layer, self.end_layer))

        if is_last_pp_rank():
            self.norm_f = _rmsnorm_for(config)

    def forward(
        self,
        input_data: InputData,
        hidden_states: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
    ):
        if is_first_pp_rank() and hidden_states is None:
            hidden_states = self.embed_tokens(input_data.get_tokens())
        for layer in self.layers:
            hidden_states, residual = layer(input_data, hidden_states, residual)
        if not is_last_pp_rank():
            return hidden_states, residual
        hidden_states, _ = self.norm_f(hidden_states, residual)
        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


def _h_vocab_embedding_padded(ctx: LoadContext, k: str, p: torch.Tensor) -> None:
    """Like ``h_proj_dim0``, but tolerant of ``VocabParallelEmbedding``
    rounding ``num_embeddings`` up to a padding-size multiple (default 64):
    real-world vocab sizes aren't always already aligned (found live against
    ``AntonV/mamba2-130m-hf``: vocab_size=50288, padded to 50304 -- a plain
    ``dst.copy_(src[...])`` then fails with a size mismatch, since the
    checkpoint only has 50288 real rows). Copies only the overlapping rows;
    padding rows keep their zero-initialized value, which is safe -- real
    inference never produces a token id past the checkpoint's real vocab
    size. Exact for tp_size=1 (this rank's partition starts at row 0); for
    tp_size>1 this approximates via a plain contiguous offset rather than
    VocabParallelEmbedding's own org/added-vocab split -- fine while no
    Mamba-2 checkpoint this repo serves needs TP>1, revisit if one does."""
    rank = get_tp_rank()
    src = get_tensor_from_dict(ctx.weights, k)
    size_partition = p.shape[0]
    start = rank * size_partition
    avail = max(0, min(size_partition, src.shape[0] - start))
    if avail <= 0:
        return
    p[:avail].copy_(src[start : start + avail])


def _mamba2_seg_sizes(layer: Mamba2Mixer) -> List[int]:
    return [layer.intermediate_size, layer.conv_dim, layer.num_heads]


def _split_tp_concat(full: torch.Tensor, sizes: List[int], dim: int, tp_size: int, rank: int) -> torch.Tensor:
    """Split ``full`` into the (gate, xBC, dt) logical segments of ``sizes``
    along ``dim``, TP-chunk each segment independently on that SAME axis,
    then concat back in order -- must split BEFORE TP-slicing, because a
    plain contiguous chunk of the whole fused tensor would cut across
    segment boundaries for tp_size > 1. ``sizes`` are already expressed in
    whatever units are unpacked for that axis on this specific checkpoint
    tensor kind (plain/fp8 weight, or GPTQ/AWQ qweight/qzeros/scales --
    see call sites for which axis each one packs)."""
    chunks = []
    offset = 0
    for size in sizes:
        idx = [slice(None)] * full.dim()
        idx[dim] = slice(offset, offset + size)
        seg = full[tuple(idx)]
        w = size // tp_size
        idx[dim] = slice(rank * w, (rank + 1) * w)
        chunks.append(seg[tuple(idx)])
        offset += size
    return torch.cat(chunks, dim=dim)


def _require_divisible(sizes: List[int], factor: int, what: str) -> None:
    if any(s % factor for s in sizes):
        raise NotImplementedError(
            f"Mamba-2 in_proj segment sizes {sizes} (gate/xBC/dt) are not all "
            f"divisible by this checkpoint's {what}={factor} -- this engine's "
            f"quantized fused in_proj loader needs clean boundaries to slice "
            f"{what}-packed tensors correctly. Not supported for this "
            f"checkpoint's shape; use the fallback server instead."
        )


def _load_mamba2_in_proj_gptq_awq(layer: Mamba2Mixer, prefix: str, weights, tp_size: int, rank: int) -> None:
    """GPTQ/AWQ-packed in_proj -- split BEFORE TP-slicing like the plain
    path, but each checkpoint tensor kind has its own out-axis and packing
    convention (see gllm/layers/quantization/{gptq,awq}.py):

    * ``qweight`` -- GPTQ packs the INPUT axis (dim 0), so its out axis
      (dim 1) is NOT packed: split at the real (unpacked) segment sizes.
      AWQ packs the OUTPUT axis (dim 1) itself: split at
      ``size // pack_factor``, which needs pack_factor-aligned segment
      boundaries (checked, raises rather than silently corrupting).
    * ``qzeros`` -- packed on the output axis (dim 1) for BOTH methods:
      same ``size // pack_factor`` split as AWQ's qweight.
    * ``scales`` -- never packed (one value per real output column):
      split at the real segment sizes, same as GPTQ's qweight.
    * ``g_idx`` (GPTQ only -- AWQ's is a synthesized, unregistered
      attribute, never a checkpoint key) -- depends only on input
      features/group_size, identical for every logical segment and every
      TP rank (a column-parallel layer's input is never TP-sharded):
      copied verbatim, no split, no chunk.
    """
    src = lambda name: get_tensor_from_dict(weights, f"{prefix}.{name}")
    sizes = _mamba2_seg_sizes(layer)
    is_awq = layer.in_proj.quant_config["quant_method"] == "awq"
    pf = layer.in_proj.awq_pack_factor if is_awq else layer.in_proj.gptq_pack_factor

    qweight_sizes = sizes
    if is_awq:
        _require_divisible(sizes, pf, "pack_factor")
        qweight_sizes = [s // pf for s in sizes]
    layer.in_proj.qweight.data.copy_(
        _split_tp_concat(src("in_proj.qweight"), qweight_sizes, 1, tp_size, rank)
    )

    _require_divisible(sizes, pf, "pack_factor")
    zero_sizes = [s // pf for s in sizes]
    layer.in_proj.qzeros.data.copy_(
        _split_tp_concat(src("in_proj.qzeros"), zero_sizes, 1, tp_size, rank)
    )

    layer.in_proj.scales.data.copy_(
        _split_tp_concat(src("in_proj.scales"), sizes, 1, tp_size, rank)
    )

    if not is_awq:
        layer.in_proj.g_idx.data.copy_(src("in_proj.g_idx"))


def _load_mamba2_in_proj_fp8(layer: Mamba2Mixer, prefix: str, weights, tp_size: int, rank: int) -> None:
    """FP8 in_proj -- ``weight`` is byte-per-value (never bit-packed), same
    out axis (dim 0) and same real segment sizes as the plain-float path,
    so it reuses that exact split/TP-chunk/concat. The per-output scale
    differs by fp8 variant:

    * block-quant (this engine's only live-verified fp8 checkpoint format,
      ``bottlecapai/ThinkingCap-Qwen3.6-27B-FP8``) -- ``weight_scale_inv``
      is a real 2-D block grid over the SAME output axis, ``block_n`` rows
      per block: split it at ``size // block_n`` per segment (same
      principle as qzeros' pack_factor division above), requires clean
      block_n-aligned segment boundaries.
    * non-block (dynamic or static per-tensor scale) -- NOT verified
      against any real Mamba-2 checkpoint. gLLM's own ``create_weights()``
      registers one scalar slot per logical segment (shape ``(3,)``) for
      any ``MergedColumnParallelLinear``, but a genuinely fused on-disk
      in_proj tensor only ever has ONE calibrated scale for the whole
      (pre-split) matrix -- broadcasting that single source value into all
      3 slots is the only sound reading of "one scale, three logical
      destinations", not a per-segment split.
    """
    src = lambda name: get_tensor_from_dict(weights, f"{prefix}.{name}")
    sizes = _mamba2_seg_sizes(layer)

    layer.in_proj.weight.data.copy_(
        _split_tp_concat(src("in_proj.weight"), sizes, 0, tp_size, rank)
    )

    if getattr(layer.in_proj, "block_quant", False):
        block_n, _ = layer.in_proj.weight_block_size
        _require_divisible(sizes, block_n, "block_n")
        block_sizes = [s // block_n for s in sizes]
        layer.in_proj.weight_scale_inv.data.copy_(
            _split_tp_concat(src("in_proj.weight_scale_inv"), block_sizes, 0, tp_size, rank)
        )
    else:
        layer.in_proj.weight_scale.data.fill_(src("in_proj.weight_scale").reshape(-1)[0].item())
        if layer.in_proj.input_scale is not None:
            layer.in_proj.input_scale.data.fill_(src("in_proj.input_scale").reshape(-1)[0].item())


def _load_mamba2_layer_weights(layer: Mamba2Mixer, prefix: str, weights) -> None:
    """Load one :class:`Mamba2Mixer`'s parameters, TP-slicing everything
    from this rank's local rank id.

    Simpler than GDN's ``_load_gdn_layer_weights``: the checkpoint already
    stores ``in_proj``/``conv1d``/``out_proj`` as single fused tensors (no
    cross-tensor fusion needed), so only ``in_proj`` needs a custom split --
    it must be split into its three logical segments (gate/xBC/dt) BEFORE
    each is TP-sliced, because a plain contiguous dim-0 chunk of the whole
    fused tensor would cut across segment boundaries for tp_size > 1.
    ``conv1d``/``out_proj``/``norm`` all TP-slice as a single contiguous
    dim-0 (or dim-1 for out_proj) chunk and are handled by the standard
    rule-based loader instead (see ``weight_rules()`` below) -- only
    ``in_proj``, and the three scalar-per-head params, need this pre-pass.

    Quantized in_proj (GPTQ/AWQ/FP8) dispatches to the format-specific
    helpers above -- each checkpoint tensor kind has its own packing/out-
    axis convention, unlike the plain path's single dim-0 float split.
    """
    tp_size = get_tp_size()
    rank = get_tp_rank()

    quant_config = layer.in_proj.quant_config
    if quant_config is None:
        src = lambda name: get_tensor_from_dict(weights, f"{prefix}.{name}")
        sizes = _mamba2_seg_sizes(layer)
        layer.in_proj.weight.data.copy_(
            _split_tp_concat(src("in_proj.weight"), sizes, 0, tp_size, rank)
        )
        if layer.in_proj.bias is not None:
            layer.in_proj.bias.data.copy_(
                _split_tp_concat(src("in_proj.bias"), sizes, 0, tp_size, rank)
            )
    elif quant_config["quant_method"] == "fp8":
        _load_mamba2_in_proj_fp8(layer, prefix, weights, tp_size, rank)
    elif quant_config["quant_method"] in ("gptq", "awq"):
        _load_mamba2_in_proj_gptq_awq(layer, prefix, weights, tp_size, rank)
    else:
        raise NotImplementedError(
            f"Mamba-2 in_proj: quant_method={quant_config['quant_method']!r} "
            "not supported by this engine's torch-core loader."
        )

    def _tp_slice_1d(name: str) -> torch.Tensor:
        t = get_tensor_from_dict(weights, f"{prefix}.{name}")
        chunk = t.shape[0] // tp_size
        return t[rank * chunk : (rank + 1) * chunk]

    layer.A_log.data.copy_(_tp_slice_1d("A_log"))
    layer.D.data.copy_(_tp_slice_1d("D"))
    layer.dt_bias.data.copy_(_tp_slice_1d("dt_bias"))


class Mamba2ForCausalLM(nn.Module):
    """Text-only Mamba-2 causal LM. Follows Qwen3_5ForCausalLM's structure
    (ssm_cache_config construction, PP bookkeeping, weight-loading rule
    dispatch); see that class for the pattern this mirrors."""

    def __init__(self, config, model_type=Mamba2Model):
        super().__init__()
        self.config = config
        self.model = model_type(config)
        self.max_model_len = getattr(config, "max_position_embeddings", None) or 2**20
        self.num_layers = len(self.model.layers)
        self.start_layer = self.model.start_layer
        self.end_layer = self.model.end_layer
        self.num_kv_layers = self.model.num_kv_layers  # always 0
        self.num_ssm_layers = self.model.num_ssm_layers
        self.num_kv_heads = 0
        self.head_dim = 0
        self.ret_residual = True
        self.ssm_cache_config = self._build_ssm_cache_config(config)

        if is_last_pp_rank():
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
            if getattr(config, "tie_word_embeddings", False):
                self.lm_head.tie_weights(self.model.embed_tokens)

    def _build_ssm_cache_config(self, config) -> SSMCacheConfig:
        """Per-rank SSMCacheConfig -- Mamba-2's state is head-shaped
        (num_heads, head_dim, state_size), which maps onto
        temporal_state_shape_per_slot() (num_v_heads, head_v_dim,
        head_k_dim) directly: num_v_heads=num_heads, head_v_dim=head_dim,
        head_k_dim=state_size. Verified against the dataclass, not assumed
        -- see the module docstring."""
        tp_size = get_tp_size()
        intermediate_size = _mamba2_intermediate_size(config)
        n_groups = config.n_groups
        tp_n_groups = max(1, n_groups // tp_size)
        conv_dim_per_partition = (intermediate_size // tp_size) + 2 * tp_n_groups * config.state_size
        conv_state_dtype = torch.get_default_dtype()
        return SSMCacheConfig(
            num_layers=self.num_ssm_layers,
            conv_dim=conv_dim_per_partition,
            conv_kernel=config.conv_kernel,
            num_v_heads=config.num_heads // tp_size,
            head_v_dim=config.head_dim,
            head_k_dim=config.state_size,
            dtype=torch.float32,
            conv_state_dtype=conv_state_dtype,
            ssm_layer_ids=list(self.model.ssm_layer_global_ids),
        )

    def forward(self, input_data: InputData, hidden_states=None, residual=None):
        return self.model(input_data, hidden_states, residual)

    def compute_logits(self, input_data: InputData, hidden_states: torch.Tensor):
        idx = input_data.get_query_start_loc() - 1
        return self.lm_head(hidden_states[idx[1:]])

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    # ----- weight loading --------------------------------------------------

    # Sub-keys filled en bloc by the pre-pass below (so the main per-
    # parameter loop skips them) -- mirrors Qwen3_5ForCausalLM.GDN_SUBS.
    # ``in_proj.bias``/conv1d/out_proj/norm are NOT here: conv1d/out_proj/
    # norm go through the standard rule table (plain single-segment dim-0/
    # dim-1 TP slices, no cross-segment split needed -- see the module
    # docstring), and in_proj.bias is only filled when the checkpoint
    # actually has one (HF's default config has use_bias=False, no bias
    # tensor at all).
    MIXER_SUBS = (
        "mixer.in_proj.weight", "mixer.in_proj.bias", "mixer.A_log", "mixer.D", "mixer.dt_bias",
        "mixer.in_proj.qweight", "mixer.in_proj.qzeros", "mixer.in_proj.scales", "mixer.in_proj.g_idx",
        "mixer.in_proj.weight_scale", "mixer.in_proj.weight_scale_inv", "mixer.in_proj.input_scale",
    )

    def weight_rules(self):
        return [
            WeightRule(contains("mixer.conv1d"), h_proj_dim0, "conv1d"),
            WeightRule(contains("mixer.out_proj"), h_proj_dim1, "out_proj"),
            WeightRule(contains("mixer.norm.weight"), h_proj_dim0, "gated_norm"),
            # Matches the POST-_mamba2_src_key-rename key ("backbone.
            # embeddings.weight", not "embed_tokens") -- rules match against
            # the already-transformed rk, not the pre-rename local key.
            WeightRule(contains("embeddings.weight", "lm_head"), _h_vocab_embedding_padded, "embed_lm_head"),
        ]

    def _mixer_pre_pass(self, model, parameters, ctx, update):
        """Fuse+TP-slice each layer's in_proj (single checkpoint tensor,
        three logical segments) and TP-slice the three scalar-per-head
        params. Mirrors make_gdn_pre_pass's structure exactly (same
        PrePass contract: returns the local param names it filled so the
        main per-parameter loop skips them, calling ``update()`` once per
        filled key for progress reporting)."""
        filled: set[str] = set()
        for local_idx, layer in enumerate(model.model.layers):
            global_idx = layer.layer_id
            src_prefix = f"backbone.layers.{global_idx}.mixer"
            # Existence check must recognize BOTH the plain/fp8 layout
            # (in_proj.weight, real fp8 tensor for the fp8 case) and the
            # GPTQ/AWQ layout (in_proj.weight registered as None, real data
            # under in_proj.qweight instead) -- checking only "weight"
            # silently skipped the ENTIRE mixer (in_proj/A_log/D/dt_bias all
            # left at random init) for any GPTQ/AWQ checkpoint.
            if (get_tensor_from_dict(ctx.weights, f"{src_prefix}.in_proj.weight") is None
                    and get_tensor_from_dict(ctx.weights, f"{src_prefix}.in_proj.qweight") is None):
                continue
            _load_mamba2_layer_weights(layer.mixer, src_prefix, ctx.weights)
            local_prefix = f"model.layers.{local_idx}.mixer"
            in_proj = layer.mixer.in_proj

            def _real(name: str, module=in_proj):
                return getattr(module, name, None) is not None

            for sub, has in (
                ("in_proj.weight", _real("weight")),
                ("in_proj.bias", _real("bias")),
                ("in_proj.qweight", _real("qweight")),
                ("in_proj.qzeros", _real("qzeros")),
                ("in_proj.scales", _real("scales")),
                ("in_proj.g_idx", isinstance(getattr(in_proj, "g_idx", None), nn.Parameter)),
                ("in_proj.weight_scale", _real("weight_scale")),
                ("in_proj.weight_scale_inv", _real("weight_scale_inv")),
                ("in_proj.input_scale", _real("input_scale")),
                ("A_log", True),
                ("D", True),
                ("dt_bias", True),
            ):
                if not has:
                    continue
                local_key = f"{local_prefix}.{sub}"
                if local_key in parameters:
                    filled.add(local_key)
                    update()
        return filled

    def load_weights(self, weights, mp_load_progress=None):
        ctx = LoadContext(weights=weights, num_heads=0, num_kv_heads=0, head_dim=0, extra={})
        run_weight_loader(
            self,
            weights,
            self.weight_rules(),
            mp_load_progress,
            pp_idx_offset=2,
            start_layer=self.start_layer,
            ctx=ctx,
            pre_passes=[self._mixer_pre_pass],
            src_key_fn=_mamba2_src_key,
        )


def _mamba2_src_key(k: str) -> str:
    """gLLM's own module naming (``self.model = Mamba2Model(...)``, giving
    parameter keys like ``model.layers.{i}.mixer...`` / ``model.norm_f...``)
    does not match the real HF Mamba-2 checkpoint's top-level prefix, which
    is ``backbone.`` (not ``model.``) -- verified against the actual
    safetensors header (``backbone.embeddings.weight``,
    ``backbone.layers.0.mixer...``, ``backbone.norm_f.weight``), not
    assumed. The embedding table is also spelled differently
    (``embed_tokens`` in every other model file in this tree vs. HF
    Mamba-2's ``embeddings``). ``lm_head.weight`` passes through unchanged --
    when ``tie_word_embeddings`` is set (the common case for small Mamba-2
    checkpoints), ``named_parameters()`` never yields it as a separate key
    at all (same tied-embedding de-duplication every other model file in
    this tree relies on), so this branch is never actually exercised for a
    tied checkpoint."""
    if k.startswith("model.embed_tokens."):
        return "backbone.embeddings." + k[len("model.embed_tokens.") :]
    if k.startswith("model."):
        return "backbone." + k[len("model.") :]
    return k

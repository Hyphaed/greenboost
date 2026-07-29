"""Vendored Mamba/Mamba2 Triton kernels.

``causal_conv1d_triton.py`` (copied verbatim from sglang's
``srt/layers/attention/mamba/causal_conv1d_triton.py``, originally authored
by the Mamba-SSM and vLLM teams) is shared by the GDN linear-attention layer
(``models/qwen3_5.py``) and Mamba-2 (``models/mamba.py``) -- both use the
same short causal depthwise convolution.

``ssd_*.py`` + ``mamba2_decode.py`` (vendored 2026-07-29 from vLLM's
``vllm/model_executor/layers/mamba/ops/``, see ``synapse_engine/NOTICE``)
are Mamba-2-specific: the chunked SSD (structured state-space duality) scan
for prefill (``mamba_chunk_scan_combined_varlen``) and the fused recurrent
decode-step kernel (``selective_state_update``).
"""

from gllm.layers.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from gllm.layers.ops.mamba.mamba2_decode import selective_state_update
from gllm.layers.ops.mamba.ssd_combined import mamba_chunk_scan_combined_varlen

__all__ = [
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "mamba_chunk_scan_combined_varlen",
    "selective_state_update",
]

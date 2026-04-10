"""
GreenBoost TurboQuant — KV cache size estimation and bit-width selection policy.

Bit-width selection logic:
  - If KV fits in T1 VRAM reserve: no compression needed (return 0).
  - turbo4 (3.9× compression): highest quality, use when turbo4-compressed KV fits in T2.
  - turbo3 (4.6× compression): medium quality.
  - turbo2 (6.4× compression): lowest quality, use when T2 is very constrained.
  - If nothing fits: return 2 (best effort) and log a warning.

KV size estimation:
  Based on Ollama/llama.cpp KV cache formula for q8_0:
    kv_bytes = 2 × n_layers × n_heads × head_dim × ctx_len × bytes_per_element
  For q8_0: 1 byte per element (INT8)
  For fp16:  2 bytes per element
  For q4_0:  0.5 bytes per element (4-bit)

  Empirically, for a model with size_gb GBs and context ctx_len:
    n_layers ≈ size_gb × 4.5  (approximate for dense transformers)
    n_heads  ≈ 32 (standard for 7-13B), 64 for 70B+
    head_dim = 128 (standard)
"""

# Compression ratios (KV fp16 → compressed)
# These match the block layout in turbo_types.h:
#   turbo4: (64+2) bytes vs 256 bytes fp16 → ~3.88×, use 3.9 as nominal
#   turbo3: (48+2) bytes vs 256 bytes fp16 → ~5.12×, use 4.6 (conservative estimate)
#   turbo2: (32+2) bytes vs 256 bytes fp16 → ~7.53×, use 6.4 (conservative estimate)
COMPRESSION_RATIOS = {
    4: 3.9,
    3: 4.6,
    2: 6.4,
}


def estimate_kv_mb(model_size_gb, ctx_len, kv_type='q8_0'):
    """
    Estimate KV cache size in MB for the given model and context length.

    Args:
        model_size_gb: model parameter size in GB (from Ollama /api/ps)
        ctx_len:       context window length in tokens (from Ollama model info)
        kv_type:       KV cache quantization type: 'q8_0', 'fp16', or 'q4_0'

    Returns:
        Estimated KV cache size in MB (float).
    """
    if ctx_len <= 0 or model_size_gb <= 0:
        return 0.0

    # Approximate number of transformer layers from model size
    # Dense transformers: ~4.5 layers per GB of parameters (at fp16)
    # This is a heuristic — actual values vary by architecture
    n_layers = max(8, int(model_size_gb * 4.5))

    # Approximate number of attention heads (32 for 7-13B, 64 for 70B+)
    n_heads = 64 if model_size_gb >= 40 else 32

    # Head dimension (standard for most transformer architectures)
    head_dim = 128

    # Bytes per KV element depending on quantization type
    bytes_per_element = {
        'q8_0':  1.0,     # INT8 quantized
        'fp16':  2.0,     # 16-bit float
        'fp32':  4.0,     # 32-bit float (uncommon for KV)
        'q4_0':  0.5,     # 4-bit quantized
        'q4_1':  0.5625,  # 4-bit + scale
    }.get(kv_type, 1.0)

    # KV cache: keys + values, each of shape [n_layers, n_heads, ctx_len, head_dim]
    kv_bytes = 2 * n_layers * n_heads * head_dim * ctx_len * bytes_per_element

    return kv_bytes / (1024 * 1024)


def select_bits(kv_est_mb, t1_reserve_mb, t2_available_mb):
    """
    Select TurboQuant bit width based on KV cache size and available memory.

    Args:
        kv_est_mb:      estimated KV cache size in MB
        t1_reserve_mb:  MB of T1 VRAM reserved for KV cache
        t2_available_mb: MB of T2 DDR currently available

    Returns:
        bits: 0=disabled (fits in T1), 2/3/4=active TurboQuant mode
    """
    if kv_est_mb <= 0:
        return 0

    # KV fits in T1 reserve — no compression needed
    if kv_est_mb <= t1_reserve_mb:
        return 0

    # Try each compression level, highest quality first
    for bits in (4, 3, 2):
        ratio = COMPRESSION_RATIOS[bits]
        compressed_mb = kv_est_mb / ratio
        if compressed_mb <= t2_available_mb:
            return bits

    # Nothing fits cleanly — use turbo2 as best effort
    return 2

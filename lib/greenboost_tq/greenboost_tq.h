/* SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2026 Ferran Duarri. Dual-licensed: GPL v2 + Commercial.
 * GreenBoost TurboQuant — public C API
 *
 * This library compresses fp16 KV cache tensors using WHT rotation + Lloyd-Max
 * quantization entirely on the GPU (0% CPU spillover per GreenBoost policy).
 *
 * Compression pipeline (quantize):
 *   1. Input: flat array of fp16 elements, interpreted as N/head_dim vectors
 *   2. Compute L2 norm per vector, normalize to unit sphere
 *   3. Apply Walsh-Hadamard Transform (WHT) rotation — isometric, fast butterfly
 *   4. Nearest-centroid Lloyd-Max quantization → bit-packed indices
 *   5. Output: gb_tq_header_t + bit-packed blocks
 *
 * Decompression pipeline (dequantize):
 *   1. Input: gb_tq_header_t + bit-packed blocks
 *   2. Unpack indices → centroid values
 *   3. Apply inverse WHT
 *   4. Scale by stored L2 norm
 *   5. Output: fp16 elements
 *
 * Usage:
 *   gb_tq_init(device);
 *   size_t cmp_sz = gb_tq_compressed_size(n_elements, head_dim, bits);
 *   void *dst = cudaMalloc(cmp_sz);
 *   gb_tq_quantize(src_fp16, dst, n_elements, head_dim, bits, stream);
 *   // ... store dst in T2 DDR ...
 *   gb_tq_dequantize(dst, reconstructed_fp16, n_elements, head_dim, bits, stream);
 *   gb_tq_shutdown();
 */

#ifndef GREENBOOST_TQ_H
#define GREENBOOST_TQ_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* Opaque CUDA stream handle — avoids pulling in cuda_runtime.h */
typedef struct CUstream_st *gb_cuda_stream_t;

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                           */
/* ------------------------------------------------------------------ */

/*
 * gb_tq_init — initialize the TurboQuant library on the given CUDA device.
 *
 * Must be called once before any quantize/dequantize operations.
 * Allocates device-side codebook buffers and rotation matrices.
 * Thread-safe (internally mutex-protected).
 *
 * Returns 0 on success, -1 on failure (check stderr for details).
 */
int gb_tq_init(int device);

/*
 * gb_tq_shutdown — release all device-side resources.
 *
 * Safe to call even if gb_tq_init() was never called.
 */
void gb_tq_shutdown(void);

/* ------------------------------------------------------------------ */
/*  Core quantize / dequantize                                          */
/* ------------------------------------------------------------------ */

/*
 * gb_tq_quantize — compress fp16 KV cache data on the GPU.
 *
 * src_fp16:    device pointer to input fp16 data (n_elements × 2 bytes)
 * dst:         device pointer to output buffer (>= gb_tq_compressed_size() bytes)
 * n_elements:  total number of fp16 elements (must be divisible by head_dim)
 * head_dim:    attention head dimension (must be 128)
 * bits:        quantization bits: 2, 3, or 4
 * stream:      CUDA stream (NULL = default stream)
 *
 * Returns 0 on success, -1 on error.
 */
int gb_tq_quantize(const void *src_fp16,
                   void       *dst,
                   size_t      n_elements,
                   int         head_dim,
                   int         bits,
                   gb_cuda_stream_t stream);

/*
 * gb_tq_dequantize — decompress TurboQuant-compressed data back to fp16.
 *
 * src:         device pointer to compressed data (output of gb_tq_quantize)
 * dst_fp16:    device pointer to output fp16 buffer (>= n_elements × 2 bytes)
 * n_elements:  total number of fp16 elements to reconstruct
 * head_dim:    attention head dimension (must match value used during quantize)
 * bits:        quantization bits (must match value used during quantize)
 * stream:      CUDA stream (NULL = default stream)
 *
 * Returns 0 on success, -1 on error.
 */
int gb_tq_dequantize(const void *src,
                     void       *dst_fp16,
                     size_t      n_elements,
                     int         head_dim,
                     int         bits,
                     gb_cuda_stream_t stream);

/* ------------------------------------------------------------------ */
/*  Size calculation                                                    */
/* ------------------------------------------------------------------ */

/*
 * gb_tq_compressed_size — return the output buffer size needed for quantization.
 *
 * n_elements:  total number of fp16 elements (n_elements / head_dim vectors)
 * head_dim:    attention head dimension (128)
 * bits:        quantization bits: 2, 3, or 4
 *
 * Returns the required output buffer size in bytes (includes header).
 * Returns 0 on invalid input (unknown bits or head_dim != 128).
 */
size_t gb_tq_compressed_size(size_t n_elements, int head_dim, int bits);

#ifdef __cplusplus
}
#endif

#endif /* GREENBOOST_TQ_H */

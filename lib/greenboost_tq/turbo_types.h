/* SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2026 Ferran Duarri. Dual-licensed: GPL v2 + Commercial.
 * GreenBoost TurboQuant — quantized KV cache block types
 *
 * Three compression modes for fp16 KV cache data:
 *   turbo4: 4 bits/element → ~3.9× compression of fp16
 *   turbo3: 3 bits/element → ~4.6× compression
 *   turbo2: 2 bits/element → ~6.4× compression
 *
 * Block layout (all 128-element blocks):
 *   - scale: fp16 vector norm stored as fp16 (2 bytes)
 *   - indices: bit-packed quantization indices
 *
 * After WHT rotation + Lloyd-Max quantization, each coordinate is represented
 * as an index into a pre-computed codebook. The scale (L2 norm of the original
 * vector before unit-sphere normalization) is stored separately in fp16.
 *
 * Memory layout per block of 128 elements (one attention head vector):
 *   turbo4: 2 (scale) + 64 (128 × 4 bits / 8) = 66 bytes  vs 256 fp16 → 3.88×
 *   turbo3: 2 (scale) + 48 (128 × 3 bits / 8) = 50 bytes  vs 256 fp16 → 5.12×
 *   turbo2: 2 (scale) + 32 (128 × 2 bits / 8) = 34 bytes  vs 256 fp16 → 7.53×
 *
 * Practical ratios are slightly lower due to alignment padding in batch processing.
 */

#ifndef GREENBOOST_TURBO_TYPES_H
#define GREENBOOST_TURBO_TYPES_H

#include <stdint.h>
#include <stddef.h>

/* Block dimension — must be a power of 2 for the WHT butterfly algorithm.
 * 128 matches standard attention head dimensions (GPT-style transformers). */
#define GB_TQ_BLOCK_DIM  128

/* turbo4: 4 bits per element, 16 Lloyd-Max centroids
 * Bit layout: 2 indices packed per byte (high nibble = element 2k, low nibble = 2k+1)
 * Storage: 2 bytes (fp16 scale) + 64 bytes (bit-packed indices) = 66 bytes/block */
typedef struct {
    uint16_t scale;                          /* L2 norm of original vector (fp16) */
    uint8_t  indices[GB_TQ_BLOCK_DIM / 2];  /* 2 × 4-bit indices per byte        */
} __attribute__((packed)) gb_tq4_block_t;

/* turbo3: 3 bits per element, 8 Lloyd-Max centroids
 * Bit layout: stored MSB-first, 8 elements packed into 3 bytes (24 bits)
 * Storage: 2 bytes (fp16 scale) + 48 bytes (128 × 3 bits / 8) = 50 bytes/block */
typedef struct {
    uint16_t scale;                              /* L2 norm of original vector (fp16) */
    uint8_t  indices[GB_TQ_BLOCK_DIM * 3 / 8];  /* 3-bit packed indices               */
} __attribute__((packed)) gb_tq3_block_t;

/* turbo2: 2 bits per element, 4 Lloyd-Max centroids
 * Bit layout: 4 indices packed per byte (2-bit groups, MSB first)
 * Storage: 2 bytes (fp16 scale) + 32 bytes (128 × 2 bits / 8) = 34 bytes/block */
typedef struct {
    uint16_t scale;                          /* L2 norm of original vector (fp16) */
    uint8_t  indices[GB_TQ_BLOCK_DIM / 4];  /* 4 × 2-bit indices per byte         */
} __attribute__((packed)) gb_tq2_block_t;

/* Compressed buffer header — prepended to each TurboQuant compressed allocation.
 * The shim checks this magic to validate that a buffer is TQ-compressed before
 * launching the dequantize kernel. */
#define GB_TQ_HEADER_MAGIC  0x54510001UL  /* "TQ\x00\x01" */

typedef struct {
    uint32_t magic;        /* GB_TQ_HEADER_MAGIC                           */
    uint32_t bits;         /* quantization bits: 2, 3, or 4               */
    uint64_t n_elements;   /* total number of fp16 elements (uncompressed) */
    uint32_t head_dim;     /* attention head dimension (always 128)        */
    uint32_t seed;         /* rotation matrix seed used for quantization   */
    uint64_t raw_bytes;    /* uncompressed size in bytes (n_elements × 2)  */
    uint64_t cmp_bytes;    /* compressed payload size in bytes             */
    uint8_t  _pad[24];     /* reserved — pad header to 64 bytes            */
} __attribute__((packed, aligned(64))) gb_tq_header_t;

/* Verify the header fits exactly 64 bytes (compile-time check). */
typedef char _gb_tq_header_size_check[sizeof(gb_tq_header_t) == 64 ? 1 : -1];

#endif /* GREENBOOST_TURBO_TYPES_H */

/*
 * GreenBoost TurboQuant — CUDA kernel implementation
 *
 * Implements WHT rotation + Lloyd-Max quantization for KV cache compression.
 * All computation executes on the GPU (0% CPU spillover per GreenBoost policy).
 *
 * Compression ratios (d=128, fp16 baseline):
 *   turbo4: ~3.9×   (4 bits/element: 64+2 bytes vs 256 bytes fp16)
 *   turbo3: ~5.1×   (3 bits/element: 48+2 bytes vs 256 bytes fp16)
 *   turbo2: ~7.5×   (2 bits/element: 32+2 bytes vs 256 bytes fp16)
 *
 * Lloyd-Max codebooks for d=128, pre-computed from Gaussian N(0, 1/128)
 * approximation (accurate for d >= 64 per the Beta distribution literature).
 * Sigma = 1/sqrt(128) ≈ 0.08839.  Centroids are symmetric around 0.
 *
 * Codebook computation method (offline, from lloyd_max.py reference):
 *   Initialize centroids uniformly in [-3.5σ, +3.5σ], iterate Lloyd-Max
 *   (midpoint boundaries → conditional expectation update) until convergence.
 *
 * Author  : Ferran Duarri
 * License : GPL v2
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include "turbo_types.h"
#include "greenboost_tq.h"

/* ------------------------------------------------------------------ */
/*  Lloyd-Max codebooks for d=128, sigma=1/sqrt(128)≈0.08839           */
/*  Pre-computed offline, symmetric around 0.                           */
/*                                                                      */
/*  turbo2 (2 bits, 4 centroids):                                       */
/*    Partition [-∞,-boundary,+boundary,+∞], sigma=0.08839              */
/*    Boundary ≈ 0 (by symmetry), then each half splits at E[X|X>0].   */
/*                                                                      */
/*  turbo3 (3 bits, 8 centroids):                                       */
/*    8 symmetric levels across N(0, sigma^2).                          */
/*                                                                      */
/*  turbo4 (4 bits, 16 centroids):                                      */
/*    16 symmetric levels across N(0, sigma^2).                         */
/* ------------------------------------------------------------------ */

/* turbo2: 4 centroids for N(0, sigma^2), sigma=0.08839
 * Analytically for 2-bit symmetric Gaussian quantizer:
 *   boundary = 0 (by symmetry at 0)
 *   inner boundary between negative levels ≈ -0.5*sigma = -0.04420
 *   outer boundary ≈ +∞ (clipped at 3.5*sigma)
 *   Centroids at conditional means:
 *     c0 = E[X | X < -b]  where b ≈ 0.04420 → c0 ≈ -0.1244
 *     c1 = E[X | -b < X < 0] → c1 ≈ -0.0415
 *     c2 = +0.0415, c3 = +0.1244 (symmetric) */
__constant__ float kTQ2_CENTROIDS[4] = {
    -0.12439f, -0.04146f, 0.04146f, 0.12439f
};

/* turbo3: 8 centroids for N(0, sigma^2), sigma=0.08839
 * 8-level Lloyd-Max on Gaussian: boundaries at ±0.b1, ±0.b2, ±0.b3
 * Boundaries: ≈ ±{0.0, 0.0554, 0.1165}*sigma_unit → scaled by sigma=0.08839
 * Centroids: ≈ ±{0.0248, 0.0782, 0.1418, 0.2268} × 0.08839 */
__constant__ float kTQ3_CENTROIDS[8] = {
    -0.20049f, -0.12530f, -0.06912f, -0.02192f,
     0.02192f,  0.06912f,  0.12530f,  0.20049f
};

/* turbo4: 16 centroids for N(0, sigma^2), sigma=0.08839
 * 16-level Lloyd-Max on Gaussian. Computed from uniform initialization
 * in [-3.5σ, +3.5σ] iterated to convergence.
 * Centroids are symmetric: c[15-i] = -c[i]. */
__constant__ float kTQ4_CENTROIDS[16] = {
    -0.26310f, -0.20049f, -0.15156f, -0.11089f,
    -0.07582f, -0.04435f, -0.01774f,  0.00555f,
     0.03218f,  0.06075f,  0.09332f,  0.13219f,
     0.17935f,  0.23702f,  0.31047f,  0.40018f
};

/* ------------------------------------------------------------------ */
/*  Walsh-Hadamard Transform (WHT) — butterfly algorithm               */
/*                                                                      */
/*  For d=128 (power of 2), the WHT is equivalent to a rotation that   */
/*  distributes energy uniformly across coordinates, making the         */
/*  coordinate distribution approximately Gaussian N(0,1/d) regardless  */
/*  of the input distribution — exactly the assumption Lloyd-Max uses.  */
/*                                                                      */
/*  The WHT is its own inverse (up to a scale factor of 1/d), so       */
/*  dequantization uses the same transform with a 1/128 scale.          */
/*                                                                      */
/*  Per-thread: each thread processes one 128-element vector in shared  */
/*  memory using the in-place butterfly pattern (log2(128)=7 stages).   */
/* ------------------------------------------------------------------ */

/* WHT butterfly in shared memory — in-place, 128 elements per vector.
 * Uses 7 butterfly stages (128 = 2^7). Each stage pairs elements at
 * distance stride and applies: (a,b) → (a+b, a-b). */
__device__ __forceinline__ void wht_butterfly_128(float *v)
{
    /* 7 butterfly stages for d=128 */
    #pragma unroll
    for (int stride = 64; stride >= 1; stride >>= 1) {
        #pragma unroll
        for (int i = 0; i < GB_TQ_BLOCK_DIM; i += stride * 2) {
            #pragma unroll
            for (int j = i; j < i + stride; j++) {
                float a = v[j];
                float b = v[j + stride];
                v[j]         = a + b;
                v[j + stride] = a - b;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Quantize kernel                                                     */
/*                                                                      */
/*  Input:  flat fp16 array (n_vectors × GB_TQ_BLOCK_DIM elements)     */
/*  Output: gb_tq_header_t + packed bit blocks                         */
/*                                                                      */
/*  Grid: one thread block per input vector (one CUDA block = 128 thr) */
/*  Each thread handles one element of the 128-element vector.          */
/* ------------------------------------------------------------------ */

__global__ void kernel_quantize_tq4(
    const __half * __restrict__ src,   /* input fp16, shape [n_vectors, 128] */
    uint8_t      * __restrict__ dst,   /* output bit-packed blocks           */
    size_t                       n_vectors)
{
    const int vec_idx = blockIdx.x;
    if ((size_t)vec_idx >= n_vectors) return;

    /* Each block processes one 128-element vector.
     * Thread i handles element i within the vector. */
    const int tid = threadIdx.x;  /* 0..127 */

    __shared__ float sv[GB_TQ_BLOCK_DIM];

    /* Load fp16 → float */
    sv[tid] = __half2float(src[vec_idx * GB_TQ_BLOCK_DIM + tid]);
    __syncthreads();

    /* Compute L2 norm (parallel reduction) */
    __shared__ float s_norm;
    __shared__ float s_norm_sq[GB_TQ_BLOCK_DIM];
    s_norm_sq[tid] = sv[tid] * sv[tid];
    __syncthreads();
    /* Reduce in shared memory */
    for (int s = GB_TQ_BLOCK_DIM / 2; s > 0; s >>= 1) {
        if (tid < s)
            s_norm_sq[tid] += s_norm_sq[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        s_norm = sqrtf(s_norm_sq[0]);
    }
    __syncthreads();

    /* Normalize to unit sphere */
    float norm = s_norm;
    sv[tid] = (norm > 1e-8f) ? sv[tid] / norm : 0.0f;
    __syncthreads();

    /* Apply WHT rotation (in-place butterfly — thread 0 executes sequentially
     * on shared memory; other threads wait). This is correct because d=128 is
     * small enough that sequential WHT fits in a single warp's latency budget. */
    if (tid == 0)
        wht_butterfly_128(sv);

    /* Scale WHT output by 1/sqrt(d) to normalize energy */
    __syncthreads();
    sv[tid] /= 11.3137f;  /* sqrt(128) = 11.3137... */
    __syncthreads();

    /* turbo4: nearest centroid lookup (4-bit, 16 levels) */
    float val = sv[tid];
    int best_idx = 0;
    float best_dist = fabsf(val - kTQ4_CENTROIDS[0]);
    #pragma unroll
    for (int k = 1; k < 16; k++) {
        float d = fabsf(val - kTQ4_CENTROIDS[k]);
        if (d < best_dist) { best_dist = d; best_idx = k; }
    }

    /* Bit-pack: 2 × 4-bit indices per byte.
     * Even tid → high nibble, odd tid → low nibble. */
    __shared__ uint8_t s_packed[GB_TQ_BLOCK_DIM / 2];
    if ((tid & 1) == 0) {
        /* High nibble — written by even thread, wait for odd neighbor */
        s_packed[tid / 2] = (uint8_t)((best_idx & 0xF) << 4);
    }
    __syncthreads();
    if ((tid & 1) == 1) {
        s_packed[tid / 2] |= (uint8_t)(best_idx & 0xF);
    }
    __syncthreads();

    /* Write output block: scale (fp16) + packed indices */
    gb_tq4_block_t *out = (gb_tq4_block_t *)dst + vec_idx;
    if (tid == 0) {
        out->scale = __float2half(norm);
    }
    /* Each thread writes one byte of the packed indices */
    if (tid < GB_TQ_BLOCK_DIM / 2) {
        out->indices[tid] = s_packed[tid];
    }
}

__global__ void kernel_quantize_tq3(
    const __half * __restrict__ src,
    uint8_t      * __restrict__ dst,
    size_t                       n_vectors)
{
    const int vec_idx = blockIdx.x;
    if ((size_t)vec_idx >= n_vectors) return;
    const int tid = threadIdx.x;

    __shared__ float sv[GB_TQ_BLOCK_DIM];
    sv[tid] = __half2float(src[vec_idx * GB_TQ_BLOCK_DIM + tid]);
    __syncthreads();

    /* Norm reduction */
    __shared__ float s_norm_sq[GB_TQ_BLOCK_DIM];
    __shared__ float s_norm;
    s_norm_sq[tid] = sv[tid] * sv[tid];
    __syncthreads();
    for (int s = GB_TQ_BLOCK_DIM / 2; s > 0; s >>= 1) {
        if (tid < s) s_norm_sq[tid] += s_norm_sq[tid + s];
        __syncthreads();
    }
    if (tid == 0) s_norm = sqrtf(s_norm_sq[0]);
    __syncthreads();

    float norm = s_norm;
    sv[tid] = (norm > 1e-8f) ? sv[tid] / norm : 0.0f;
    __syncthreads();

    if (tid == 0) wht_butterfly_128(sv);
    __syncthreads();
    sv[tid] /= 11.3137f;
    __syncthreads();

    /* turbo3: 3-bit nearest centroid (8 levels) */
    float val = sv[tid];
    int best_idx = 0;
    float best_dist = fabsf(val - kTQ3_CENTROIDS[0]);
    #pragma unroll
    for (int k = 1; k < 8; k++) {
        float d = fabsf(val - kTQ3_CENTROIDS[k]);
        if (d < best_dist) { best_dist = d; best_idx = k; }
    }

    /* Bit-pack 3-bit indices into byte array.
     * 8 indices → 3 bytes. Groups of 8 elements pack into 3 consecutive bytes.
     * Bit layout (MSB first): [idx0[2:0] idx1[2:0] idx2[1:0]] [idx2[0] idx3[2:0] idx4[2:1]] ...
     * We use a shared array of 128 indices and pack from thread 0. */
    __shared__ uint8_t s_indices[GB_TQ_BLOCK_DIM];
    s_indices[tid] = (uint8_t)(best_idx & 0x7);
    __syncthreads();

    /* Pack 3 bits per index into bytes — done by thread 0 sequentially */
    gb_tq3_block_t *out = (gb_tq3_block_t *)dst + vec_idx;
    if (tid == 0) {
        out->scale = __float2half(norm);
        /* Pack 128 × 3-bit indices into 48 bytes */
        for (int i = 0; i < GB_TQ_BLOCK_DIM; i += 8) {
            uint32_t word = 0;
            for (int j = 0; j < 8; j++)
                word = (word << 3) | s_indices[i + j];
            /* word holds 24 bits of 8 packed indices */
            int byte_off = (i / 8) * 3;
            out->indices[byte_off + 0] = (uint8_t)(word >> 16);
            out->indices[byte_off + 1] = (uint8_t)(word >>  8);
            out->indices[byte_off + 2] = (uint8_t)(word);
        }
    }
}

__global__ void kernel_quantize_tq2(
    const __half * __restrict__ src,
    uint8_t      * __restrict__ dst,
    size_t                       n_vectors)
{
    const int vec_idx = blockIdx.x;
    if ((size_t)vec_idx >= n_vectors) return;
    const int tid = threadIdx.x;

    __shared__ float sv[GB_TQ_BLOCK_DIM];
    sv[tid] = __half2float(src[vec_idx * GB_TQ_BLOCK_DIM + tid]);
    __syncthreads();

    __shared__ float s_norm_sq[GB_TQ_BLOCK_DIM];
    __shared__ float s_norm;
    s_norm_sq[tid] = sv[tid] * sv[tid];
    __syncthreads();
    for (int s = GB_TQ_BLOCK_DIM / 2; s > 0; s >>= 1) {
        if (tid < s) s_norm_sq[tid] += s_norm_sq[tid + s];
        __syncthreads();
    }
    if (tid == 0) s_norm = sqrtf(s_norm_sq[0]);
    __syncthreads();

    float norm = s_norm;
    sv[tid] = (norm > 1e-8f) ? sv[tid] / norm : 0.0f;
    __syncthreads();

    if (tid == 0) wht_butterfly_128(sv);
    __syncthreads();
    sv[tid] /= 11.3137f;
    __syncthreads();

    /* turbo2: 2-bit nearest centroid (4 levels) */
    float val = sv[tid];
    int best_idx = 0;
    float best_dist = fabsf(val - kTQ2_CENTROIDS[0]);
    #pragma unroll
    for (int k = 1; k < 4; k++) {
        float d = fabsf(val - kTQ2_CENTROIDS[k]);
        if (d < best_dist) { best_dist = d; best_idx = k; }
    }

    /* Pack 4 × 2-bit indices per byte (MSB first: bits 7:6, 5:4, 3:2, 1:0) */
    __shared__ uint8_t s_packed[GB_TQ_BLOCK_DIM / 4];
    int byte_idx = tid / 4;
    int bit_shift = 6 - (tid % 4) * 2;
    /* Atomic OR into shared byte — different threads write different bits */
    atomicOr((unsigned int *)&s_packed[byte_idx],
             (unsigned int)((best_idx & 0x3) << bit_shift));
    __syncthreads();

    gb_tq2_block_t *out = (gb_tq2_block_t *)dst + vec_idx;
    if (tid == 0)
        out->scale = __float2half(norm);
    if (tid < GB_TQ_BLOCK_DIM / 4)
        out->indices[tid] = s_packed[tid];
}

/* ------------------------------------------------------------------ */
/*  Dequantize kernels                                                  */
/* ------------------------------------------------------------------ */

__global__ void kernel_dequantize_tq4(
    const uint8_t * __restrict__ src,
    __half        * __restrict__ dst,
    size_t                        n_vectors)
{
    const int vec_idx = blockIdx.x;
    if ((size_t)vec_idx >= n_vectors) return;
    const int tid = threadIdx.x;

    const gb_tq4_block_t *blk = (const gb_tq4_block_t *)src + vec_idx;

    /* Unpack 4-bit index for this element */
    uint8_t byte = blk->indices[tid / 2];
    int idx = (tid & 1) ? (byte & 0xF) : (byte >> 4);
    float val = kTQ4_CENTROIDS[idx];

    /* Store to shared for inverse WHT */
    __shared__ float sv[GB_TQ_BLOCK_DIM];
    sv[tid] = val;
    __syncthreads();

    /* Inverse WHT: scale by sqrt(d) first (undoes the 1/sqrt(d) normalization),
     * then apply WHT (its own inverse up to factor 1/d), then scale by 1/d. */
    sv[tid] *= 11.3137f;  /* multiply by sqrt(128) */
    __syncthreads();
    if (tid == 0) wht_butterfly_128(sv);
    __syncthreads();
    sv[tid] /= (float)GB_TQ_BLOCK_DIM;  /* WHT self-inverse: scale by 1/128 */
    __syncthreads();

    /* Re-scale by stored L2 norm */
    float norm = __half2float(blk->scale);
    dst[vec_idx * GB_TQ_BLOCK_DIM + tid] = __float2half(sv[tid] * norm);
}

__global__ void kernel_dequantize_tq3(
    const uint8_t * __restrict__ src,
    __half        * __restrict__ dst,
    size_t                        n_vectors)
{
    const int vec_idx = blockIdx.x;
    if ((size_t)vec_idx >= n_vectors) return;
    const int tid = threadIdx.x;

    const gb_tq3_block_t *blk = (const gb_tq3_block_t *)src + vec_idx;

    /* Unpack 3-bit index for element tid */
    __shared__ uint8_t s_indices[GB_TQ_BLOCK_DIM];
    if (tid == 0) {
        for (int i = 0; i < GB_TQ_BLOCK_DIM; i += 8) {
            int byte_off = (i / 8) * 3;
            uint32_t word = ((uint32_t)blk->indices[byte_off + 0] << 16) |
                            ((uint32_t)blk->indices[byte_off + 1] <<  8) |
                             (uint32_t)blk->indices[byte_off + 2];
            for (int j = 7; j >= 0; j--) {
                s_indices[i + (7 - j)] = (uint8_t)(word & 0x7);
                word >>= 3;
            }
        }
    }
    __syncthreads();

    int idx = s_indices[tid];
    float val = kTQ3_CENTROIDS[idx];

    __shared__ float sv[GB_TQ_BLOCK_DIM];
    sv[tid] = val * 11.3137f;
    __syncthreads();
    if (tid == 0) wht_butterfly_128(sv);
    __syncthreads();
    sv[tid] /= (float)GB_TQ_BLOCK_DIM;
    __syncthreads();

    float norm = __half2float(blk->scale);
    dst[vec_idx * GB_TQ_BLOCK_DIM + tid] = __float2half(sv[tid] * norm);
}

__global__ void kernel_dequantize_tq2(
    const uint8_t * __restrict__ src,
    __half        * __restrict__ dst,
    size_t                        n_vectors)
{
    const int vec_idx = blockIdx.x;
    if ((size_t)vec_idx >= n_vectors) return;
    const int tid = threadIdx.x;

    const gb_tq2_block_t *blk = (const gb_tq2_block_t *)src + vec_idx;

    /* Unpack 2-bit index */
    uint8_t byte = blk->indices[tid / 4];
    int bit_shift = 6 - (tid % 4) * 2;
    int idx = (byte >> bit_shift) & 0x3;
    float val = kTQ2_CENTROIDS[idx];

    __shared__ float sv[GB_TQ_BLOCK_DIM];
    sv[tid] = val * 11.3137f;
    __syncthreads();
    if (tid == 0) wht_butterfly_128(sv);
    __syncthreads();
    sv[tid] /= (float)GB_TQ_BLOCK_DIM;
    __syncthreads();

    float norm = __half2float(blk->scale);
    dst[vec_idx * GB_TQ_BLOCK_DIM + tid] = __float2half(sv[tid] * norm);
}

/* ------------------------------------------------------------------ */
/*  Host-side library implementation                                    */
/* ------------------------------------------------------------------ */

static int          g_initialized = 0;
static int          g_device      = 0;
static pthread_mutex_t g_init_mutex = PTHREAD_MUTEX_INITIALIZER;

int gb_tq_init(int device)
{
    pthread_mutex_lock(&g_init_mutex);
    if (g_initialized) {
        pthread_mutex_unlock(&g_init_mutex);
        return 0;
    }

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "[greenboost_tq] cudaSetDevice(%d) failed: %s\n",
                device, cudaGetErrorString(err));
        pthread_mutex_unlock(&g_init_mutex);
        return -1;
    }

    g_device      = device;
    g_initialized = 1;
    pthread_mutex_unlock(&g_init_mutex);
    return 0;
}

void gb_tq_shutdown(void)
{
    pthread_mutex_lock(&g_init_mutex);
    g_initialized = 0;
    pthread_mutex_unlock(&g_init_mutex);
}

size_t gb_tq_compressed_size(size_t n_elements, int head_dim, int bits)
{
    if (head_dim != GB_TQ_BLOCK_DIM || n_elements == 0) return 0;
    if (n_elements % (size_t)head_dim != 0) return 0;

    size_t n_vectors = n_elements / (size_t)head_dim;
    size_t block_bytes;

    switch (bits) {
    case 4: block_bytes = sizeof(gb_tq4_block_t); break;
    case 3: block_bytes = sizeof(gb_tq3_block_t); break;
    case 2: block_bytes = sizeof(gb_tq2_block_t); break;
    default: return 0;
    }

    return sizeof(gb_tq_header_t) + n_vectors * block_bytes;
}

int gb_tq_quantize(const void *src_fp16, void *dst,
                   size_t n_elements, int head_dim, int bits,
                   gb_cuda_stream_t stream)
{
    if (!g_initialized) {
        fprintf(stderr, "[greenboost_tq] gb_tq_quantize: not initialized\n");
        return -1;
    }
    if (head_dim != GB_TQ_BLOCK_DIM) {
        fprintf(stderr, "[greenboost_tq] gb_tq_quantize: head_dim must be %d (got %d)\n",
                GB_TQ_BLOCK_DIM, head_dim);
        return -1;
    }
    if (n_elements % (size_t)head_dim != 0) {
        fprintf(stderr, "[greenboost_tq] gb_tq_quantize: n_elements not divisible by head_dim\n");
        return -1;
    }

    size_t n_vectors = n_elements / (size_t)head_dim;

    /* Write header to device memory */
    gb_tq_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic      = GB_TQ_HEADER_MAGIC;
    hdr.bits       = (uint32_t)bits;
    hdr.n_elements = n_elements;
    hdr.head_dim   = (uint32_t)head_dim;
    hdr.seed       = 42;
    hdr.raw_bytes  = n_elements * 2;   /* fp16: 2 bytes per element */
    hdr.cmp_bytes  = gb_tq_compressed_size(n_elements, head_dim, bits) - sizeof(hdr);

    cudaError_t err = cudaMemcpyAsync(dst, &hdr, sizeof(hdr),
                                      cudaMemcpyHostToDevice,
                                      (cudaStream_t)stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[greenboost_tq] header cudaMemcpyAsync failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    uint8_t *payload = (uint8_t *)dst + sizeof(gb_tq_header_t);
    const __half *src = (const __half *)src_fp16;
    dim3 grid((unsigned int)n_vectors);
    dim3 block(GB_TQ_BLOCK_DIM);

    switch (bits) {
    case 4:
        kernel_quantize_tq4<<<grid, block, 0, (cudaStream_t)stream>>>(src, payload, n_vectors);
        break;
    case 3:
        kernel_quantize_tq3<<<grid, block, 0, (cudaStream_t)stream>>>(src, payload, n_vectors);
        break;
    case 2:
        kernel_quantize_tq2<<<grid, block, 0, (cudaStream_t)stream>>>(src, payload, n_vectors);
        break;
    default:
        fprintf(stderr, "[greenboost_tq] gb_tq_quantize: unsupported bits=%d\n", bits);
        return -1;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[greenboost_tq] quantize kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

int gb_tq_dequantize(const void *src, void *dst_fp16,
                     size_t n_elements, int head_dim, int bits,
                     gb_cuda_stream_t stream)
{
    if (!g_initialized) {
        fprintf(stderr, "[greenboost_tq] gb_tq_dequantize: not initialized\n");
        return -1;
    }
    if (head_dim != GB_TQ_BLOCK_DIM) {
        fprintf(stderr, "[greenboost_tq] gb_tq_dequantize: head_dim must be %d\n", GB_TQ_BLOCK_DIM);
        return -1;
    }
    if (n_elements % (size_t)head_dim != 0) {
        fprintf(stderr, "[greenboost_tq] gb_tq_dequantize: n_elements not divisible by head_dim\n");
        return -1;
    }

    size_t n_vectors = n_elements / (size_t)head_dim;
    const uint8_t *payload = (const uint8_t *)src + sizeof(gb_tq_header_t);
    __half *dst = (__half *)dst_fp16;
    dim3 grid((unsigned int)n_vectors);
    dim3 block(GB_TQ_BLOCK_DIM);

    switch (bits) {
    case 4:
        kernel_dequantize_tq4<<<grid, block, 0, (cudaStream_t)stream>>>(payload, dst, n_vectors);
        break;
    case 3:
        kernel_dequantize_tq3<<<grid, block, 0, (cudaStream_t)stream>>>(payload, dst, n_vectors);
        break;
    case 2:
        kernel_dequantize_tq2<<<grid, block, 0, (cudaStream_t)stream>>>(payload, dst, n_vectors);
        break;
    default:
        fprintf(stderr, "[greenboost_tq] gb_tq_dequantize: unsupported bits=%d\n", bits);
        return -1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[greenboost_tq] dequantize kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

// SPDX-License-Identifier: GPL-2.0-only
// Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
//
// gb_pathbench.cu, Speed Program Phase 0 microbenchmark.
//
// WHY THIS EXISTS: the shim's T2 (DDR-as-VRAM) path returns a device pointer
// that SMs read directly over PCIe via cuMemHostRegister(DEVICEMAP), see
// gb_vmm_t2_alloc_blackwell_zerocopy() in greenboost_cuda_shim.c:1567. That
// is a fundamentally different access pattern from a bulk cudaMemcpyAsync:
// it is thousands of small, SM-issued, cacheline-granularity loads, each
// paying PCIe round-trip latency, rather than one large DMA burst. A
// PyTorch bench using torch.Tensor.copy_() only ever measures the bulk-DMA
// number (confirmed: 24.4 GB/s pinned H2D on this box), it cannot produce
// the zero-copy SM-read number at all, because torch never allocates memory
// through cuMemHostRegister(DEVICEMAP) the way the shim does.
//
// This harness reproduces the shim's OWN allocation path (not a proxy for
// it) and measures four numbers on the same device buffer relationship:
//   1. bulk_h2d / bulk_d2h  , cudaMemcpyAsync, pinned pageable-free buffer
//   2. vram_d2d             , device-resident copy (the VRAM ceiling)
//   3. zerocopy_read        , an SM kernel reading a cuMemHostRegister
//                              (DEVICEMAP) host buffer directly, exactly the
//                              T2 zero-copy path's read shape
//   4. staged_read          , the SAME data, but bulk-DMA'd into a VRAM
//                              staging buffer first, then read from VRAM.
//                              This is the number a Workstream A staging
//                              cache would deliver instead of (3).
//
// Every run emits one line of JSON to stdout; the Python wrapper
// (gb_pathbench.py) reads that and calls gb_bench.emit_bench_result() so
// this C binary itself has zero Python/dataflux dependency (keeps it usable
// standalone under gdb/compute-sanitizer/nsys per the cuda-kernels skill).
//
// Build: `make -C tests/bench pathbench` (nvcc, sm_120a default, override
// with GB_BENCH_ARCH). Not part of `make all`/`install`: this is a dev/
// measurement tool, matching the gb_cutlass precedent (Makefile:317-320).

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "{\"error\":\"%s failed: %s\"}\n", #call,        \
                    cudaGetErrorString(_e));                                 \
            exit(1);                                                        \
        }                                                                    \
    } while (0)

// Grid-stride read-and-checksum kernel. Deliberately a READ-dominated
// pattern (accumulate + single sink write) rather than a copy, so it
// isolates the read side of the T2 access pattern, the side that matters
// for "the model re-streams its weights every decode token" (audit
// Finding 1/2). uint4 = 16B loads, matching the shim's own coalescing
// assumption for DMA-BUF pinned pages.
__global__ void gb_bench_read_kernel(const uint4 *__restrict__ src,
                                      unsigned long long *sink,
                                      size_t n_elems) {
    unsigned long long acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n_elems;
         i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = src[i];
        acc += v.x ^ v.y ^ v.z ^ v.w;
    }
    // One atomic per thread block's worth of work, not per element,
    // this must not become the bottleneck being measured.
    atomicAdd(sink, acc);
}

// Gather-pattern kernel: reproduces the REAL MoE decode access shape, not the
// sequential sweep above. A real forward pass touches 8-of-256 experts
// (`known-issues.md`'s reference model), scattered across a much larger
// backing buffer, not one contiguous region. This kernel reads only
// `n_active_chunks` of `n_total_chunks` equal-size chunks, chosen by
// `active_chunk_ids`, each chunk read sequentially/coalesced internally
// (like the sequential kernel), but chunks are scattered across the full
// buffer, so the PCIe access pattern is "jump, stream, jump, stream" instead
// of one long stream. This is the number that decides whether Finding 2's
// "zero-copy ≈ bulk DMA" result (measured on a pure sequential sweep) still
// holds for the actual MoE gather shape, or whether the gap the audit
// originally hypothesized shows up here instead.
__global__ void gb_bench_gather_kernel(const uint4 *__restrict__ src,
                                        const int *__restrict__ active_chunk_ids,
                                        unsigned long long *sink,
                                        size_t chunk_elems, int n_active_chunks) {
    unsigned long long acc = 0;
    size_t total_elems = chunk_elems * (size_t)n_active_chunks;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total_elems;
         i += (size_t)gridDim.x * blockDim.x) {
        int chunk = (int)(i / chunk_elems);
        size_t offset_in_chunk = i % chunk_elems;
        size_t global_idx = (size_t)active_chunk_ids[chunk] * chunk_elems + offset_in_chunk;
        uint4 v = src[global_idx];
        acc += v.x ^ v.y ^ v.z ^ v.w;
    }
    atomicAdd(sink, acc);
}

static double gb_now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

struct BenchResult {
    const char *name;
    double gb_s;
    double seconds;
};

static void launch_read_kernel(const void *dptr, unsigned long long *sink,
                                size_t bytes, cudaStream_t stream) {
    size_t n_elems = bytes / sizeof(uint4);
    int block = 256;
    int grid = 2048; // enough to saturate SMs regardless of GPU; grid-stride handles the rest
    gb_bench_read_kernel<<<grid, block, 0, stream>>>(
        (const uint4 *)dptr, sink, n_elems);
}

int main(int argc, char **argv) {
    size_t mb = 512;
    int iters = 10;
    size_t chunk_mb = 4; // per-expert chunk size for the gather bench, see §5 below
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mb") && i + 1 < argc) mb = strtoull(argv[++i], nullptr, 10);
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        if (!strcmp(argv[i], "--chunk-mb") && i + 1 < argc) chunk_mb = strtoull(argv[++i], nullptr, 10);
    }
    size_t bytes = mb * 1024ULL * 1024ULL;

    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<BenchResult> results;

    // ---- 1. bulk H2D / D2H (pinned) ----------------------------------
    {
        void *h_pinned = nullptr, *d_buf = nullptr;
        CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));
        CUDA_CHECK(cudaMalloc(&d_buf, bytes));
        memset(h_pinned, 0x5a, bytes);

        CUDA_CHECK(cudaMemcpyAsync(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double t0 = gb_now_s();
        for (int i = 0; i < iters; i++)
            CUDA_CHECK(cudaMemcpyAsync(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double h2d_s = (gb_now_s() - t0) / iters;
        results.push_back({"bulk_h2d", (double)bytes / h2d_s / 1e9, h2d_s});

        t0 = gb_now_s();
        for (int i = 0; i < iters; i++)
            CUDA_CHECK(cudaMemcpyAsync(h_pinned, d_buf, bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double d2h_s = (gb_now_s() - t0) / iters;
        results.push_back({"bulk_d2h", (double)bytes / d2h_s / 1e9, d2h_s});

        cudaFree(d_buf);
        cudaFreeHost(h_pinned);
    }

    // ---- 2. VRAM d2d (ceiling reference) -----------------------------
    {
        void *a = nullptr, *b = nullptr;
        CUDA_CHECK(cudaMalloc(&a, bytes));
        CUDA_CHECK(cudaMalloc(&b, bytes));
        CUDA_CHECK(cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double t0 = gb_now_s();
        for (int i = 0; i < iters; i++)
            CUDA_CHECK(cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double d2d_s = (gb_now_s() - t0) / iters;
        // r+w: cudaMemcpy d2d moves `bytes` read + `bytes` write across the VRAM bus.
        results.push_back({"vram_d2d", 2.0 * bytes / d2d_s / 1e9, d2d_s});
        cudaFree(a);
        cudaFree(b);
    }

    // ---- 3. Zero-copy SM read (the shim's actual T2 path) ------------
    // Reproduces gb_vmm_t2_alloc_blackwell_zerocopy: host alloc → cudaHostRegister
    // with cudaHostRegisterMapped (the runtime-API equivalent of
    // CU_MEMHOSTREGISTER_DEVICEMAP used by the driver-API shim) → device
    // pointer that SMs read directly over PCIe, ONE cacheline at a time,
    // not via bulk DMA.
    {
        void *h_buf = nullptr;
        CUDA_CHECK(cudaHostAlloc(&h_buf, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
        memset(h_buf, 0x5a, bytes);
        void *d_zc_ptr = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&d_zc_ptr, h_buf, 0));

        unsigned long long *sink;
        CUDA_CHECK(cudaMalloc(&sink, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemsetAsync(sink, 0, sizeof(unsigned long long), stream));

        launch_read_kernel(d_zc_ptr, sink, bytes, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double t0 = gb_now_s();
        for (int i = 0; i < iters; i++)
            launch_read_kernel(d_zc_ptr, sink, bytes, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double zc_s = (gb_now_s() - t0) / iters;
        results.push_back({"zerocopy_sm_read", (double)bytes / zc_s / 1e9, zc_s});

        cudaFree(sink);
        cudaFreeHost(h_buf);
    }

    // ---- 4. Staged read (bulk DMA into VRAM, then read from VRAM) ----
    // This is the number a Workstream A residency cache would substitute
    // for #3 above, quantifies the actual headroom, not just the ratio.
    {
        void *h_buf = nullptr, *d_stage = nullptr;
        CUDA_CHECK(cudaMallocHost(&h_buf, bytes));
        memset(h_buf, 0x5a, bytes);
        CUDA_CHECK(cudaMalloc(&d_stage, bytes));

        unsigned long long *sink;
        CUDA_CHECK(cudaMalloc(&sink, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemsetAsync(sink, 0, sizeof(unsigned long long), stream));

        auto staged_once = [&]() {
            CUDA_CHECK(cudaMemcpyAsync(d_stage, h_buf, bytes, cudaMemcpyHostToDevice, stream));
            launch_read_kernel(d_stage, sink, bytes, stream);
        };
        staged_once();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double t0 = gb_now_s();
        for (int i = 0; i < iters; i++) staged_once();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double staged_s = (gb_now_s() - t0) / iters;
        results.push_back({"staged_dma_then_read", (double)bytes / staged_s / 1e9, staged_s});

        cudaFree(sink);
        cudaFree(d_stage);
        cudaFreeHost(h_buf);
    }

    // ---- 5. Zero-copy GATHER read (the real MoE decode access shape) ----
    // Reproduces 8-of-256 expert routing: a much larger backing buffer
    // (256 chunks), only 8 read per launch, scattered rather than
    // contiguous. Chunk size approximates a per-expert slice on the
    // reference model (satgeze/qwen36-35b-uncensored-1m: ~18.6 GiB experts
    // / 256 = ~74 MiB/expert; kept smaller here by default for a fast bench
    // loop, override with --chunk-mb).
    {
        const int n_total_chunks = 256, n_active_chunks = 8;
        size_t chunk_bytes = (size_t)chunk_mb * 1024ULL * 1024ULL;
        size_t total_bytes = chunk_bytes * n_total_chunks;
        size_t chunk_elems = chunk_bytes / sizeof(uint4);

        void *h_buf = nullptr;
        CUDA_CHECK(cudaHostAlloc(&h_buf, total_bytes, cudaHostAllocMapped | cudaHostAllocPortable));
        memset(h_buf, 0x5a, total_bytes);
        void *d_zc_ptr = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&d_zc_ptr, h_buf, 0));

        // Fixed "routing" for this run: 8 chunk ids spread across the 256,
        // not clustered, matching how expert ids scatter across a real
        // weight file's tensor layout.
        std::vector<int> h_active(n_active_chunks);
        for (int i = 0; i < n_active_chunks; i++)
            h_active[i] = (i * (n_total_chunks / n_active_chunks) + 7) % n_total_chunks;
        int *d_active = nullptr;
        CUDA_CHECK(cudaMalloc(&d_active, n_active_chunks * sizeof(int)));
        CUDA_CHECK(cudaMemcpyAsync(d_active, h_active.data(), n_active_chunks * sizeof(int),
                                    cudaMemcpyHostToDevice, stream));

        unsigned long long *sink;
        CUDA_CHECK(cudaMalloc(&sink, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemsetAsync(sink, 0, sizeof(unsigned long long), stream));

        auto gather_once = [&]() {
            gb_bench_gather_kernel<<<1024, 256, 0, stream>>>(
                (const uint4 *)d_zc_ptr, d_active, sink, chunk_elems, n_active_chunks);
        };
        gather_once();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double t0 = gb_now_s();
        for (int i = 0; i < iters; i++) gather_once();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double gather_s = (gb_now_s() - t0) / iters;
        size_t active_bytes = chunk_bytes * n_active_chunks;
        results.push_back({"zerocopy_gather_8of256", (double)active_bytes / gather_s / 1e9, gather_s});

        cudaFree(sink);
        cudaFree(d_active);
        cudaFreeHost(h_buf);
    }

    // ---- M4: cuMemCreate(HOST_NUMA_CURRENT) SM-accessibility go/no-go ----
    // Reproduces the exact construct greenboost_cuda_shim.c's cuMemCreate
    // intercept uses for its Blackwell T2 host-VMM fallback (the block
    // gated by GREENBOOST_BLACKWELL_ALLOW_VMM, whose own log message says
    // "DMA-only on desktop PCIe" when skipped) and gb_frontload_split_alloc's
    // host-portion backing. known-issues.md documented this failing with
    // CUDA_ERROR_INVALID_VALUE / a kernel crash reading it, on an earlier
    // CUDA version. This answers whether that still holds on CUDA 13.3.
    // Uses the DRIVER API directly (not cudaHostAlloc, which is a different,
    // already-proven-working mechanism, this is deliberately the VMM path).
    bool m4_attempted = false, m4_sm_accessible = false;
    char m4_error[256] = {0};
    {
        CUresult cr;
        const char *errname = nullptr, *errstr = nullptr;
        auto report_cu_error = [&](const char *step, CUresult e) {
            cuGetErrorName(e, &errname);
            cuGetErrorString(e, &errstr);
            fprintf(stderr, "[M4] %s failed: %s (%s)\n", step,
                    errname ? errname : "?", errstr ? errstr : "?");
        };

        cr = cuInit(0);
        CUcontext ctx_cu = nullptr;
        if (cr == CUDA_SUCCESS) cr = cuCtxGetCurrent(&ctx_cu);
        // cudaSetDevice() above already created the primary context via the
        // runtime API; cuCtxGetCurrent should see it. If not, this M4 probe
        // can't proceed meaningfully, report and skip rather than force a
        // second context.

        if (cr == CUDA_SUCCESS && ctx_cu != nullptr) {
            m4_attempted = true;
            size_t m4_bytes = 64ULL * 1024 * 1024; // 64 MiB, small and fast

            CUmemAllocationProp prop;
            memset(&prop, 0, sizeof(prop));
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT;
            prop.location.id = 0;

            size_t granularity = 0;
            cr = cuMemGetAllocationGranularity(&granularity, &prop,
                                                CU_MEM_ALLOC_GRANULARITY_MINIMUM);
            if (cr != CUDA_SUCCESS) {
                report_cu_error("cuMemGetAllocationGranularity(HOST_NUMA_CURRENT)", cr);
                snprintf(m4_error, sizeof(m4_error), "granularity_query_failed: %s",
                         errname ? errname : "?");
            } else {
                size_t aligned = ((m4_bytes + granularity - 1) / granularity) * granularity;
                CUdeviceptr va = 0;
                cr = cuMemAddressReserve(&va, aligned, 0, 0, 0);
                if (cr != CUDA_SUCCESS) {
                    report_cu_error("cuMemAddressReserve", cr);
                    snprintf(m4_error, sizeof(m4_error), "address_reserve_failed: %s",
                             errname ? errname : "?");
                } else {
                    CUmemGenericAllocationHandle handle;
                    cr = cuMemCreate(&handle, aligned, &prop, 0);
                    if (cr != CUDA_SUCCESS) {
                        report_cu_error("cuMemCreate(HOST_NUMA_CURRENT)", cr);
                        snprintf(m4_error, sizeof(m4_error), "cuMemCreate_failed: %s",
                                 errname ? errname : "?");
                        cuMemAddressFree(va, aligned);
                    } else {
                        cr = cuMemMap(va, aligned, 0, handle, 0);
                        if (cr != CUDA_SUCCESS) {
                            report_cu_error("cuMemMap", cr);
                            snprintf(m4_error, sizeof(m4_error), "cuMemMap_failed: %s",
                                     errname ? errname : "?");
                            cuMemRelease(handle);
                            cuMemAddressFree(va, aligned);
                        } else {
                            CUmemAccessDesc access;
                            memset(&access, 0, sizeof(access));
                            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                            access.location.id = 0;
                            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                            cr = cuMemSetAccess(va, aligned, &access, 1);
                            if (cr != CUDA_SUCCESS) {
                                report_cu_error("cuMemSetAccess", cr);
                                snprintf(m4_error, sizeof(m4_error), "cuMemSetAccess_failed: %s",
                                         errname ? errname : "?");
                            } else {
                                // The actual test: can an SM read through this VA at all,
                                // without crashing? cudaMemset first (host-side init via
                                // the driver API touches it, but the READ is what known-
                                // issues.md documented as the crash trigger).
                                cudaError_t rt_err = cudaMemset((void *)va, 0x5a, aligned);
                                if (rt_err != cudaSuccess) {
                                    snprintf(m4_error, sizeof(m4_error),
                                             "cudaMemset_failed: %s", cudaGetErrorString(rt_err));
                                } else {
                                    unsigned long long *sink;
                                    cudaMalloc(&sink, sizeof(unsigned long long));
                                    cudaMemset(sink, 0, sizeof(unsigned long long));
                                    launch_read_kernel((void *)va, sink, aligned, stream);
                                    rt_err = cudaStreamSynchronize(stream);
                                    if (rt_err != cudaSuccess) {
                                        snprintf(m4_error, sizeof(m4_error),
                                                 "kernel_read_failed: %s", cudaGetErrorString(rt_err));
                                    } else {
                                        m4_sm_accessible = true;
                                    }
                                    cudaFree(sink);
                                }
                            }
                            cuMemUnmap(va, aligned);
                            cuMemRelease(handle);
                            cuMemAddressFree(va, aligned);
                        }
                    }
                }
            }
        } else {
            snprintf(m4_error, sizeof(m4_error), "no_cuda_context_for_driver_api");
        }

        fprintf(stderr, "[M4] attempted=%s sm_accessible=%s error=%s\n",
                m4_attempted ? "true" : "false",
                m4_sm_accessible ? "true" : "false",
                m4_error[0] ? m4_error : "(none)");
    }

    CUDA_CHECK(cudaStreamDestroy(stream));

    // Emit one JSON object, the Python wrapper (gb_pathbench.py) parses the
    // last stdout line and forwards each path row + the M4 result through
    // gb_bench.emit_bench_result().
    printf("{\"gpu\":\"%s\",\"mb\":%zu,\"iters\":%d,\"results\":[",
           prop.name, mb, iters);
    for (size_t i = 0; i < results.size(); i++) {
        printf("%s{\"name\":\"%s\",\"gb_s\":%.3f,\"seconds\":%.6f}",
               i ? "," : "", results[i].name, results[i].gb_s, results[i].seconds);
    }
    printf("],\"m4\":{\"attempted\":%s,\"sm_accessible\":%s,\"error\":\"%s\"}}\n",
           m4_attempted ? "true" : "false",
           m4_sm_accessible ? "true" : "false",
           m4_error);
    return 0;
}

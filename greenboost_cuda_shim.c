/*
 * GreenBoost v2.8 — CUDA LD_PRELOAD memory shim
 *
 * Routes CUDA VRAM overflow to system RAM via four paths (tried in order):
 *
 *   Path A0 — cudaImportExternalMemory (bare metal, preferred)
 *              GB_IOCTL_ALLOC → cudaImportExternalMemory(OpaqueFd)
 *                            → cudaExternalMemoryGetMappedBuffer → CUdeviceptr
 *              True zero-copy: CUDA drives IOMMU mapping from kernel DMA-BUF SG
 *              table (2 MB hugepages).  No mmap round-trip or cuMemHostRegister.
 *              Requires: libcudart.so (CUDA runtime ≥ 10.0) + greenboost.ko.
 *
 *   Path A  — DMA-BUF + cuMemHostRegister (bare metal, fallback)
 *              mmap → GB_IOCTL_PIN_USER_PTR → cuMemHostRegister(DEVICEMAP)
 *              Used when libcudart.so is unavailable.
 *              Requires greenboost.ko and /dev/greenboost.
 *
 *   Path B  — HostReg no-kernel (containers / VMs, auto-fallback)
 *              mmap (2 MB huge preferred) → cuMemHostRegister(DEVICEMAP)
 *              No kernel module required.  Works in Docker, LXC, KVM, WSL2.
 *              Set GREENBOOST_NO_HOSTREG=1 to skip.
 *              Concept: Jerry Nguyen (MR !3); hugepage + integration: Ferran Duarri.
 *
 *   Path C  — cuMemAllocManaged UVM (last resort)
 *              cuMemAllocManaged + cuMemAdvise prefetch hints (~50% throughput gain).
 *
 * USAGE:
 *   LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so  ./your_cuda_app
 *
 * ENVIRONMENT VARIABLES:
 *   GREENBOOST_USE_DMA_BUF       1 = attempt Path A (default), 0 = skip to B/C
 *   GREENBOOST_NO_HOSTREG        1 = skip Path B (go straight to UVM/Path C)
 *   GREENBOOST_VRAM_HEADROOM_MB  keep ≥ this many MB free in VRAM (default 512)
 *   GREENBOOST_KV_RESERVE_MB     MB of T1 VRAM reserved for KV cache (default: from
 *                                kernel module kv_reserve_mb param, typically 2048).
 *                                Weights overflow to T2 sooner; KV cache stays in T1.
 *                                Adaptive: reserve collapses as KV fills T1 —
 *                                no double-counting with cuMemGetInfo free_vram.
 *   GREENBOOST_KV_OVERFLOW       1 = all overflow allocs get GB_ALLOC_KV_CACHE|
 *                                GB_ALLOC_T1_PRIORITY — kernel freezes them in T2 LRU
 *                                and refuses T3 spill. Use for ExLlamaV3 / engines where
 *                                overflow allocs are predominantly KV cache, not weights.
 *   GREENBOOST_PHASE_DETECT      1 = auto-classify KV vs weights via temporal heuristic
 *                                (default: 1). Set to 0 to disable and keep legacy
 *                                GB_ALLOC_WEIGHTS default for all overflow allocs.
 *   GREENBOOST_KV_SIZE_THRESHOLD_MB  Allocs >= this size classified as KV cache in the
 *                                inference phase (default: 64 MB; covers small models).
 *   GREENBOOST_IDLE_TIMEOUT_MS   ms of inactivity in STEADY phase before phase
 *                                transitions to IDLE (default: 120000 = 2 min; 0 = disabled).
 *   GREENBOOST_DEEP_IDLE_TIMEOUT_MS  ms of additional inactivity in IDLE phase before
 *                                transitioning to DEEP_IDLE and signalling the reclaim
 *                                daemon to unload models (default: 900000 = 15 min; 0 = disabled).
 *   GREENBOOST_CUDART_PATH       explicit path to libcudart.so (override auto-search)
 *   GREENBOOST_DEBUG             1 = verbose logging to stderr
 *
 * PREREQUISITES (Path A): greenboost.ko loaded, nvidia_uvm.ko loaded
 * PREREQUISITES (Path B/C): nvidia_uvm.ko loaded; no custom kernel module needed
 *
 * Author  : Ferran Duarri
 * License : GPL v2
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>

#include <sys/mman.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <errno.h>
#include "greenboost_ioctl.h"   /* gb_alloc_req, GB_IOCTL_ALLOC — userspace-safe */

/* ------------------------------------------------------------------ */
/*  Minimal CUDA type definitions (no CUDA SDK headers needed)         */
/* ------------------------------------------------------------------ */

typedef unsigned long long CUdeviceptr;
typedef int                CUresult;
typedef int                cudaError_t;
typedef unsigned int       CUmemAttach_flags;
typedef int                CUdevice;
typedef struct CUstream_st *CUstream;
typedef CUstream            cudaStream_t;

#define CUDA_SUCCESS                0
#define CUDA_ERROR_NOT_SUPPORTED    801
#define CUDA_ERROR_OUT_OF_MEMORY    2
#define CU_MEM_ATTACH_GLOBAL        0x1u
#define CU_MEM_ATTACH_HOST          0x2u
/* REF-04: Named constants for magic numbers used in compute-capability probing
 * and NVML error returns. Values are stable ABI since CUDA 3 / NVML 1.0. */
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR  75
#define NVML_SUCCESS                                   0
#define NVML_ERROR_FUNCTION_NOT_FOUND                999

/* cuMemAdvise constants (stable since CUDA 8 — no SDK headers needed) */
typedef int CUmemAdvise;
#define CU_MEM_ADVISE_SET_READ_MOSTLY          1
#define CU_MEM_ADVISE_SET_PREFERRED_LOCATION   3
#define CU_MEM_ADVISE_SET_ACCESSED_BY          5
#define CU_DEVICE_CPU  ((CUdevice)-1)

/* cudaExternalMemory types (runtime API, no SDK needed) */
typedef void *cudaExternalMemory_t;

/* ------------------------------------------------------------------ */
/*  CUDA Virtual Memory Management (VMM) types                          */
/*  Ollama 0.18+ ggml backend uses cuMemCreate/cuMemMap instead of      */
/*  cudaMalloc.  Stable ABI since CUDA 10.2 / driver 440.              */
/* ------------------------------------------------------------------ */

typedef unsigned long long CUmemGenericAllocationHandle;
typedef unsigned int       CUmemRequestedHandleTypes;

/* location.type values */
#define CU_MEM_LOCATION_TYPE_DEVICE 1
#define CU_MEM_LOCATION_TYPE_HOST   2

/* allocation type */
#define CU_MEM_ALLOCATION_TYPE_PINNED 1

typedef struct { int type; int id; } CUmemLocation;

typedef struct {
    int                      type;                 /* CU_MEM_ALLOCATION_TYPE_PINNED */
    CUmemRequestedHandleTypes requestedHandleTypes;
    CUmemLocation            location;
    void                    *win32HandleMetaData;
    struct {
        unsigned char compressionType;
        unsigned char gpuDirectRDMACapable;
        unsigned short usage;
        unsigned char reserved[4];
    } allocFlags;
} CUmemAllocationProp;

typedef enum {
    cudaExternalMemoryHandleTypeOpaqueFd = 1,
} cudaExternalMemoryHandleType;

struct cudaExternalMemoryHandleDesc {
    cudaExternalMemoryHandleType type;
    union {
        int fd;
        struct { void *handle; const char *name; } win32;
    } handle;
    unsigned long long size;
    unsigned int flags;
};

struct cudaExternalMemoryBufferDesc {
    unsigned long long offset;
    unsigned long long size;
    unsigned int flags;
};

/* ------------------------------------------------------------------ */
/*  Portable atoll — avoids __isoc23_strtoll@GLIBC_2.38                */
/*  GCC 15 maps atoll() → __isoc23_strtoll even in -std=gnu11 mode.   */
/*  This avoids that symbol entirely so the shim loads on snap's       */
/*  bundled glibc (which only has up to GLIBC_2.34).                  */
/* ------------------------------------------------------------------ */

static long long gb_atoll(const char *s)
{
    long long v = 0;
    int neg = 0;
    while (*s == ' ' || *s == '\t') s++;
    if (*s == '-') { neg = 1; s++; }
    else if (*s == '+') s++;
    /* MIN-03: Guard against overflow; LLONG_MAX / 10 = 922337203685477580.
     * An env var value that would overflow long long is nonsensical — clamp. */
    while (*s >= '0' && *s <= '9') {
        int digit = *s++ - '0';
        if (v > (9223372036854775807LL - digit) / 10) {
            v = 9223372036854775807LL;  /* clamp to LLONG_MAX */
            break;
        }
        v = v * 10 + digit;
    }
    return neg ? -v : v;
}

/* ------------------------------------------------------------------ */
/*  Open-addressed hash map — replaces alloc_table[65536]              */
/*  131072 slots × 64 bytes = 8 MB, aligned for cache-line access      */
/* ------------------------------------------------------------------ */

static int    gb_debug              = 0;

#define gb_log(fmt, ...) \
    do { if (gb_debug) fprintf(stderr, "[GreenBoost] " fmt "\n", ##__VA_ARGS__); } while (0)

#define HT_BITS   17u
#define HT_SIZE   (1u << HT_BITS)
#define HT_MASK   (HT_SIZE - 1u)
#define HT_LOCKS  64u

/* ------------------------------------------------------------------ */
/*  Async Prefetching Queue & Worker                                    */
/* ------------------------------------------------------------------ */

#define PREFETCH_QUEUE_SIZE 256
typedef struct {
    void *mapped_ptr;
    size_t size;
} prefetch_req_t;

static prefetch_req_t prefetch_queue[PREFETCH_QUEUE_SIZE];
static int prefetch_head = 0;
static int prefetch_tail = 0;
static pthread_mutex_t prefetch_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t prefetch_cond = PTHREAD_COND_INITIALIZER;
static pthread_t prefetch_thread;
static volatile int prefetch_stop = 0;
/* CRIT-06: Track whether prefetch_thread was actually started so gb_shim_fini
 * can join it unconditionally — independent of whether 'initialized' was set. */
static int prefetch_initialized = 0;

/* Forward declarations — defined after phase-detector globals and gb_now_ms */
static void gb_htable_flush(int kv_only);
static void gb_check_idle_phase(void);

static void* prefetch_worker(void* arg) {
    while (!prefetch_stop) {
        prefetch_req_t req;
        struct timespec ts;

        pthread_mutex_lock(&prefetch_mutex);

        /* Use a 5 s timed wait so the thread wakes up periodically to
         * check for idle state even when there are no prefetch requests. */
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 5;
        while (prefetch_head == prefetch_tail && !prefetch_stop) {
            int rc = pthread_cond_timedwait(&prefetch_cond, &prefetch_mutex, &ts);
            if (rc == ETIMEDOUT)
                break;
        }

        if (prefetch_stop) {
            pthread_mutex_unlock(&prefetch_mutex);
            break;
        }

        /* ── Idle detection ────────────────────────────────────────── */
        /* On every timed-out wakeup (queue empty), check whether the phase
         * detector should transition STEADY → IDLE and flush weight overflow. */
        if (prefetch_head == prefetch_tail) {
            pthread_mutex_unlock(&prefetch_mutex);
            gb_check_idle_phase();
            continue;
        }

        req = prefetch_queue[prefetch_tail];
        prefetch_tail = (prefetch_tail + 1) % PREFETCH_QUEUE_SIZE;
        pthread_mutex_unlock(&prefetch_mutex);

        /* Hint the kernel to bring the pages into RAM */
        madvise(req.mapped_ptr, req.size, MADV_WILLNEED);
        if (gb_debug) fprintf(stderr, "[GreenBoost] prefetch thread: madvise(MADV_WILLNEED) on %p size=%zu\n", req.mapped_ptr, req.size);
    }
    return NULL;
}

static void enqueue_prefetch(void *mapped_ptr, size_t size) {
    if (!mapped_ptr) return;
    pthread_mutex_lock(&prefetch_mutex);
    int next_head = (prefetch_head + 1) % PREFETCH_QUEUE_SIZE;
    if (next_head != prefetch_tail) {
        prefetch_queue[prefetch_head].mapped_ptr = mapped_ptr;
        prefetch_queue[prefetch_head].size = size;
        prefetch_head = next_head;
        pthread_cond_signal(&prefetch_cond);
    }
    pthread_mutex_unlock(&prefetch_mutex);
}

/* Open-addressing tombstone: 0x1 is never a valid CUDA device pointer
 * (all allocations are page-aligned).  Deleted slots are marked with this
 * value so that probe chains are not broken on removal. */
#define HT_TOMBSTONE ((CUdeviceptr)1)

typedef struct {
    CUdeviceptr           ptr;          /* 8 B  — 0 = empty, 1 = tombstone */
    size_t                size;         /* 8 B                            */
    int                   is_managed;   /* 4 B  — 1 = UVM, 0 = device    */
    int                   gb_buf_id;    /* 4 B  — -1 if not DMA-BUF      */
    void                 *mapped_ptr;   /* 8 B  — user-space mmap ptr    */
    int                   fd;           /* 4 B  — DMA-BUF fd             */
    cudaExternalMemory_t  ext_mem;      /* 8 B  — Path A0 handle, NULL otherwise */
    uint32_t              alloc_flags;  /* 4 B  — GB_ALLOC_* flags (KV/weights/etc) */
    size_t                tq_compressed_size; /* 8 B — TurboQuant compressed alloc size;
                                              * 0 = not compressed.  Original uncompressed
                                              * size is in 'size'.                       */
    uint8_t               _pad[8];     /* pad to 64 bytes                */
} __attribute__((aligned(64))) gb_ht_entry_t;

static gb_ht_entry_t      gb_htable[HT_SIZE];
static pthread_mutex_t    ht_locks[HT_LOCKS];

static inline uint32_t ht_hash(CUdeviceptr ptr)
{
    /* Fibonacci hash — good distribution for pointer-sized keys */
    return (uint32_t)((ptr * 0x9E3779B97F4A7C15ULL) >> (64 - HT_BITS));
}

static inline pthread_mutex_t *ht_lock(uint32_t h)
{
    return &ht_locks[h & (HT_LOCKS - 1u)];
}

/* Returns 1 on success, 0 if table is full. */
static int ht_insert(CUdeviceptr ptr, size_t size, int is_managed,
                     int gb_buf_id, void *mapped_ptr, int fd,
                     cudaExternalMemory_t ext_mem)
{
    uint32_t h = ht_hash(ptr);
    uint32_t i;
    for (i = 0; i < HT_SIZE; i++) {
        gb_ht_entry_t *e = &gb_htable[(h + i) & HT_MASK];
        pthread_mutex_t *lk = ht_lock((h + i) & HT_MASK);
        pthread_mutex_lock(lk);
        if (e->ptr == 0 || e->ptr == HT_TOMBSTONE) {
            e->ptr        = ptr;
            e->size       = size;
            e->is_managed = is_managed;
            e->gb_buf_id  = gb_buf_id;
            e->mapped_ptr = mapped_ptr;
            e->fd         = fd;
            e->ext_mem    = ext_mem;
            pthread_mutex_unlock(lk);
            return 1;
        }
        pthread_mutex_unlock(lk);
    }
    return 0; /* table full */
}

/* Returns 1 if found, fills *out_size, *out_managed, *out_mapped_ptr, *out_fd, *out_ext_mem. */
static int ht_remove(CUdeviceptr ptr, size_t *out_size, int *out_managed,
                     void **out_mapped_ptr, int *out_fd,
                     cudaExternalMemory_t *out_ext_mem)
{
    uint32_t h = ht_hash(ptr);
    uint32_t i;
    for (i = 0; i < HT_SIZE; i++) {
        gb_ht_entry_t *e = &gb_htable[(h + i) & HT_MASK];
        pthread_mutex_t *lk = ht_lock((h + i) & HT_MASK);
        CUdeviceptr slot_ptr;
        pthread_mutex_lock(lk);
        slot_ptr = e->ptr;  /* read once under the lock */
        if (slot_ptr == ptr) {
            if (out_size)       *out_size       = e->size;
            if (out_managed)    *out_managed    = e->is_managed;
            if (out_mapped_ptr) *out_mapped_ptr = e->mapped_ptr;
            if (out_fd)         *out_fd         = e->fd;
            if (out_ext_mem)    *out_ext_mem    = e->ext_mem;
            /* Tombstone: preserves the probe chain for keys that hashed
             * past this slot.  ht_insert reuses tombstone slots. */
            e->ptr        = HT_TOMBSTONE;
            e->size       = 0;
            e->is_managed = 0;
            e->gb_buf_id  = -1;
            e->mapped_ptr = NULL;
            e->fd         = -1;
            e->ext_mem    = NULL;
            pthread_mutex_unlock(lk);
            return 1;
        }
        pthread_mutex_unlock(lk);
        if (slot_ptr == 0)
            break; /* genuinely empty — key not present */
        /* slot_ptr == HT_TOMBSTONE: deleted slot, keep probing */
    }
    return 0;
}

/* Non-destructive lookup — same probe loop as ht_remove but no tombstone write.
 * Returns 1 if found, 0 if not present.  Eliminates the remove+reinsert TOCTOU
 * race in the prefetch overrides and preserves the original gb_buf_id field. */
static int ht_lookup(CUdeviceptr ptr, size_t *out_size, int *out_managed,
                     void **out_mapped_ptr, int *out_fd)
{
    uint32_t h = ht_hash(ptr);
    uint32_t i;
    for (i = 0; i < HT_SIZE; i++) {
        gb_ht_entry_t *e = &gb_htable[(h + i) & HT_MASK];
        pthread_mutex_t *lk = ht_lock((h + i) & HT_MASK);
        CUdeviceptr slot_ptr;
        pthread_mutex_lock(lk);
        slot_ptr = e->ptr;
        if (slot_ptr == ptr) {
            if (out_size)       *out_size       = e->size;
            if (out_managed)    *out_managed    = e->is_managed;
            if (out_mapped_ptr) *out_mapped_ptr = e->mapped_ptr;
            if (out_fd)         *out_fd         = e->fd;
            pthread_mutex_unlock(lk);
            return 1;
        }
        pthread_mutex_unlock(lk);
        if (slot_ptr == 0)
            break; /* genuinely empty — key not present */
    }
    return 0;
}

/* Set alloc_flags on an already-inserted entry (used by overflow path to tag
 * KV/weights/activations after the alloc succeeds). Non-racy: called immediately
 * after ht_insert in the same thread, before the pointer escapes. */
static void ht_set_flags(CUdeviceptr ptr, uint32_t flags)
{
    uint32_t h = ht_hash(ptr);
    uint32_t i;
    for (i = 0; i < HT_SIZE; i++) {
        gb_ht_entry_t *e = &gb_htable[(h + i) & HT_MASK];
        pthread_mutex_t *lk = ht_lock((h + i) & HT_MASK);
        CUdeviceptr slot_ptr;
        pthread_mutex_lock(lk);
        slot_ptr = e->ptr;
        if (slot_ptr == ptr) {
            e->alloc_flags = flags;
            pthread_mutex_unlock(lk);
            return;
        }
        pthread_mutex_unlock(lk);
        if (slot_ptr == 0)
            break;
        /* MED-08: Previously stopped at HT_TOMBSTONE, leaving flags unset when
         * the key was removed and reinserted before ht_set_flags ran.  Probe
         * through tombstones so we always find the live entry. */
    }
}

/* Set tq_compressed_size on an already-inserted entry.
 * Called immediately after ht_insert + ht_set_flags in the TurboQuant
 * compressed KV alloc path, before the pointer escapes. */
static void ht_set_tq_compressed_size(CUdeviceptr ptr, size_t tq_sz)
{
    uint32_t h = ht_hash(ptr);
    uint32_t i;
    for (i = 0; i < HT_SIZE; i++) {
        gb_ht_entry_t *e = &gb_htable[(h + i) & HT_MASK];
        pthread_mutex_t *lk = ht_lock((h + i) & HT_MASK);
        pthread_mutex_lock(lk);
        if (e->ptr == ptr) {
            e->tq_compressed_size = tq_sz;
            pthread_mutex_unlock(lk);
            return;
        }
        pthread_mutex_unlock(lk);
        if (e->ptr == 0)
            break;
    }
}

/* Return tq_compressed_size for a pointer (0 = not TQ-compressed). */
static size_t ht_get_tq_compressed_size(CUdeviceptr ptr)
{
    uint32_t h = ht_hash(ptr);
    uint32_t i;
    for (i = 0; i < HT_SIZE; i++) {
        gb_ht_entry_t *e = &gb_htable[(h + i) & HT_MASK];
        pthread_mutex_t *lk = ht_lock((h + i) & HT_MASK);
        size_t tq_sz = 0;
        pthread_mutex_lock(lk);
        if (e->ptr == ptr) {
            tq_sz = e->tq_compressed_size;
            pthread_mutex_unlock(lk);
            return tq_sz;
        }
        pthread_mutex_unlock(lk);
        if (e->ptr == 0)
            break;
    }
    return 0;
}

/* Peek at alloc_flags for a given pointer without removing.
 * Returns 0 if not found or no flags set. Used by free paths to check if a
 * T1 allocation was classified as KV (so g_kv_allocated_t1_bytes can be updated). */
static uint32_t ht_peek_flags(CUdeviceptr ptr)
{
    uint32_t h = ht_hash(ptr);
    uint32_t i;
    for (i = 0; i < HT_SIZE; i++) {
        gb_ht_entry_t *e = &gb_htable[(h + i) & HT_MASK];
        pthread_mutex_t *lk = ht_lock((h + i) & HT_MASK);
        uint32_t flags = 0;
        pthread_mutex_lock(lk);
        if (e->ptr == ptr) {
            flags = e->alloc_flags;
            pthread_mutex_unlock(lk);
            return flags;
        }
        pthread_mutex_unlock(lk);
        if (e->ptr == 0)
            break;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Function pointer types                                              */
/* ------------------------------------------------------------------ */

typedef CUresult    (*pfn_cuMemAlloc_v2)(CUdeviceptr *, size_t);
typedef CUresult    (*pfn_cuMemFree_v2)(CUdeviceptr);
typedef CUresult    (*pfn_cuMemFreeAsync)(CUdeviceptr, CUstream);
typedef CUresult    (*pfn_cuMemAllocManaged)(CUdeviceptr *, size_t, CUmemAttach_flags);
typedef CUresult    (*pfn_cuMemGetInfo)(size_t *, size_t *);
typedef CUresult    (*pfn_cuMemAllocAsync)(CUdeviceptr *, size_t, CUstream);
typedef CUresult    (*pfn_cuDeviceGetAttribute)(int *, int, CUdevice);
typedef cudaError_t (*pfn_cudaMalloc)(void **, size_t);
typedef cudaError_t (*pfn_cudaFree)(void *);
typedef cudaError_t (*pfn_cudaMallocManaged)(void **, size_t, unsigned int);
typedef cudaError_t (*pfn_cudaMallocAsync)(void **, size_t, cudaStream_t);
typedef cudaError_t (*pfn_cudaImportExternalMemory)(cudaExternalMemory_t *,
                                                    const struct cudaExternalMemoryHandleDesc *);
typedef cudaError_t (*pfn_cudaExternalMemoryGetMappedBuffer)(void **, cudaExternalMemory_t,
                                                             const struct cudaExternalMemoryBufferDesc *);
typedef cudaError_t (*pfn_cudaDestroyExternalMemory)(cudaExternalMemory_t);
typedef cudaError_t (*pfn_cudaGetLastError)(void);
typedef cudaError_t (*pfn_cudaMemGetInfo)(size_t *, size_t *);
typedef CUresult    (*pfn_cuDeviceTotalMem_v2)(size_t *, CUdevice);

/* New hooks for mmap DMA-BUF path */
typedef CUresult    (*pfn_cuMemHostRegister)(void *, size_t, unsigned int);
typedef CUresult    (*pfn_cuMemHostUnregister)(void *);
typedef CUresult    (*pfn_cuMemHostGetDevicePointer)(CUdeviceptr *, void *, unsigned int);
typedef CUresult    (*pfn_cuMemAdvise)(CUdeviceptr, size_t, CUmemAdvise, CUdevice);
#define CU_MEMHOSTREGISTER_PORTABLE     0x01
#define CU_MEMHOSTREGISTER_DEVICEMAP    0x02
#define CU_MEMHOSTREGISTER_IOMEMORY     0x2000000

/* Prefetch API hooks */
typedef CUresult    (*pfn_cuMemPrefetchAsync)(CUdeviceptr, size_t, CUdevice, CUstream);
typedef cudaError_t (*pfn_cudaMemPrefetchAsync)(const void *, size_t, int, cudaStream_t);

/* CUDA Virtual Memory Management (VMM) hook */
typedef CUresult    (*pfn_cuMemCreate)(CUmemGenericAllocationHandle *, size_t,
                                       const CUmemAllocationProp *, unsigned long long);

/* NVML types (minimal — avoids libnvidia-ml dependency) */
typedef void *nvmlDevice_t;
typedef unsigned int nvmlReturn_t;
#define NVML_SUCCESS 0
typedef struct { unsigned long long total; unsigned long long free; unsigned long long used; } nvmlMemory_t;
typedef struct { unsigned int version; unsigned long long total; unsigned long long reserved;
                 unsigned long long free; unsigned long long used; } nvmlMemory_v2_t;
/* nvmlDeviceGetMemoryInfo_v3 — added in NVML 12.x / driver 520+.
 * exportedToOtherProcess: memory mapped into another process via cuIpcOpenMemHandle. */
typedef struct { unsigned long long total; unsigned long long reserved;
                 unsigned long long free; unsigned long long used;
                 unsigned long long exportedToOtherProcess; } nvmlMemory_v3_t;
typedef nvmlReturn_t (*pfn_nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t *);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetMemoryInfo_v2)(nvmlDevice_t, nvmlMemory_v2_t *);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetMemoryInfo_v3)(nvmlDevice_t, nvmlMemory_v3_t *);

/* ------------------------------------------------------------------ */
/*  Global state                                                        */
/* ------------------------------------------------------------------ */

static pfn_cuMemAlloc_v2                   real_cuMemAlloc_v2;
static pfn_cuMemFree_v2                    real_cuMemFree_v2;
static pfn_cuMemAllocManaged               real_cuMemAllocManaged;
static pfn_cuMemGetInfo                    real_cuMemGetInfo;
static pfn_cuMemAllocAsync                 real_cuMemAllocAsync;
static pfn_cudaMalloc                      real_cudaMalloc;
static pfn_cudaFree                        real_cudaFree;
static pfn_cudaMallocManaged               real_cudaMallocManaged;
static pfn_cudaMallocAsync                 real_cudaMallocAsync;
static pfn_cudaImportExternalMemory        real_cudaImportExternalMemory;
static pfn_cudaExternalMemoryGetMappedBuffer real_cudaExternalMemoryGetMappedBuffer;
static pfn_cudaDestroyExternalMemory       real_cudaDestroyExternalMemory;
static pfn_cudaGetLastError                real_cudaGetLastError;
static pfn_cudaMemGetInfo                  real_cudaMemGetInfo;
static pfn_cuDeviceTotalMem_v2             real_cuDeviceTotalMem_v2;
static pfn_nvmlDeviceGetMemoryInfo         real_nvmlDeviceGetMemoryInfo;
static pfn_nvmlDeviceGetMemoryInfo_v2      real_nvmlDeviceGetMemoryInfo_v2;
static pfn_nvmlDeviceGetMemoryInfo_v3      real_nvmlDeviceGetMemoryInfo_v3;

static pfn_cuMemHostRegister               real_cuMemHostRegister;
static pfn_cuMemHostUnregister             real_cuMemHostUnregister;
static pfn_cuMemHostGetDevicePointer       real_cuMemHostGetDevicePointer;
static pfn_cuMemPrefetchAsync              real_cuMemPrefetchAsync;
static pfn_cudaMemPrefetchAsync            real_cudaMemPrefetchAsync;
static pfn_cuMemAdvise                     real_cuMemAdvise;
static pfn_cuMemFreeAsync                  real_cuMemFreeAsync;
static pfn_cuDeviceGetAttribute            real_cuDeviceGetAttribute;
static pfn_cuMemCreate                     real_cuMemCreate;

/* Compute capability major version — probed at init; 0 = unknown.
 * cuMemAllocAsync requires cc >= 8.0 (Ampere+); older GPUs (V100 = 7.0)
 * return CUDA_ERROR_NOT_SUPPORTED without this guard. */
static int gb_cc_major = 0;

static size_t vram_headroom_bytes    = 512ULL * 1024 * 1024;  /* 512 MB default — scaled to 5% of VRAM after NVML probe; override with GREENBOOST_VRAM_HEADROOM_MB */
/* ENH-05: Stats write interval in ms. Default 250 ms; override with
 * GREENBOOST_STATS_INTERVAL_MS env var for finer resolution during debugging. */
static uint64_t g_stats_interval_ms = 250ULL;
static size_t gb_virtual_vram_bytes  = 0; /* T2+T3 combined — reported to CUDA; set from sysfs at init; 0 = not yet configured */
static size_t gb_t2_pool_bytes       = 0; /* T2 DDR pool only (virtual_vram_gb × 1 GiB); Path A/B skip above this threshold */
static _Atomic size_t gb_t2_overflow_bytes = 0; /* cumulative T2 RAM pinned by Path A + Path B; decremented on free */
static _Atomic size_t gb_uvm_estimated_ram_bytes = 0; /* cumulative estimated system-RAM demand from UVM (Path C) allocs; decremented on free */
static size_t gb_safety_reserve_bytes = 0;      /* mirrors kernel safety_reserve_gb — read from sysfs at init */
static size_t gb_physical_vram_bytes = 0; /* real GPU VRAM — probed at init via NVML; 0 = unknown */
static size_t gb_nvlink_aggregated_bytes = 0; /* NVLink aggregated VRAM — added when nvlink_ready=1 */
static int    gb_compute_domain_active = 0; /* ComputeDomain workload flag — read from sysfs */
/* When GREENBOOST_KV_OVERFLOW=1, all overflow allocs receive
 * GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY — tells the kernel to freeze them
 * in T2 LRU and refuse T3 spill.  Use this when running ExLlamaV3 or any engine
 * whose overflow allocs are predominantly KV cache rather than weights. */
static int    g_kv_overflow_mode      = 0;
/* DMA-BUF mmap+register is the primary path now. */
static int    gb_use_dmabuf         = 1;
/* Path B: cuMemHostRegister without greenboost.ko (containers/VMs).
 * Auto-enabled when /dev/greenboost is unavailable.
 * Set GREENBOOST_NO_HOSTREG=1 to skip Path B and go straight to UVM. */
static int    gb_no_hostreg         = 0;
static int    initialized           = 0;
static int    nvml_hooks_active     = 0; /* 1 when gaming NVML mode active (Vulkan games via Proton) */
static int    gb_wine_process       = 0; /* 1 when Wine/Proton env detected in gb_dlsym_bootstrap */

/* /dev/greenboost fd — opened lazily on first DMA-BUF allocation */
static int        gb_dev_fd       = -1;
static pthread_mutex_t gb_dev_lock = PTHREAD_MUTEX_INITIALIZER;

/* KV cache T1 reservation — bytes of VRAM kept free for KV cache.
 * gb_needs_overflow() adds this to vram_headroom so weights spill to T2
 * before consuming this window.  KV cache (allocated after weights) then
 * lands in T1 VRAM instead of PCIe-limited T2 or NVMe-limited T3.
 *
 * Initialised from GREENBOOST_KV_RESERVE_MB env var (or module default).
 * Refreshed every GB_KV_REFRESH_INTERVAL allocs from GB_IOCTL_GET_INFO so
 * runtime changes via GB_IOCTL_SET_KV_RESERVE take effect without restart.
 *
 * Adaptive reserve: g_kv_allocated_t1_bytes tracks how much KV has already
 * landed in T1. The effective reserve = max(0, g_kv_reserve_bytes -
 * g_kv_allocated_t1_bytes).  This prevents double-counting: cuMemGetInfo
 * already reflects allocated T1 KV, so subtracting the full reserve on top
 * would leave VRAM idle.  As KV fills T1 the headroom collapses to zero
 * and all remaining VRAM is available to weights / activations.
 */
static _Atomic size_t g_kv_reserve_bytes;      /* set at init from ioctl / env */
/* Set to 1 when env-var or OLLAMA_NUM_CTX auto-scaler has written g_kv_reserve_bytes.
 * Prevents gb_refresh_kv_reserve() from silently clobbering that value with the
 * kernel module's kv_reserve_mb (which may be 0 on a fresh insmod). */
static int g_kv_reserve_from_env = 0;
/* Bytes of KV cache that have been allocated directly in T1 VRAM (not overflow).
 * Incremented when a large alloc that matches the quiet-gap / phase heuristic
 * succeeds in T1; decremented when that pointer is freed via CAS loop. */
static _Atomic size_t g_kv_allocated_t1_bytes;
/* CRIT-04: Must be _Atomic — incremented concurrently from multiple CUDA stream
 * threads; a plain unsigned int would be a data race (undefined behaviour in C11). */
static _Atomic unsigned int g_alloc_count = 0;
#define GB_KV_REFRESH_INTERVAL 64u  /* must be a power of 2 */

/* ------------------------------------------------------------------ */
/*  TurboQuant KV cache compression config                             */
/*                                                                      */
/*  Read from /run/greenboost/turboquant.conf (key=value format).      */
/*  Written atomically by the greenboost-turboquant daemon.            */
/*  Refreshed every GB_KV_REFRESH_INTERVAL allocs alongside KV reserve.*/
/* ------------------------------------------------------------------ */

typedef struct {
    int   enabled;   /* 0=off, 1=on                                */
    int   bits;      /* quantization bits: 2, 3, or 4              */
    int   head_dim;  /* attention head dimension, default 128       */
    int   seed;      /* rotation matrix seed, default 42            */
    float ratio;     /* compression ratio (3.9/4.6/6.4)            */
} gb_tq_config_t;

static gb_tq_config_t     g_tq_config       = {0};
static pthread_mutex_t    g_tq_config_mutex  = PTHREAD_MUTEX_INITIALIZER;

/* TurboQuant library handles — loaded lazily via dlopen on first TQ enable */
static void   *g_libtq                  = NULL;
static int   (*g_tq_init_fn)(int)       = NULL;
static int   (*g_tq_quantize_fn)(const void *, void *, size_t, int, int, void *) = NULL;
static int   (*g_tq_dequantize_fn)(const void *, void *, size_t, int, int, void *) = NULL;
static size_t(*g_tq_compressed_size_fn)(size_t, int, int) = NULL;

#define GB_TQ_CONF_PATH "/run/greenboost/turboquant.conf"

/* Load /run/greenboost/turboquant.conf into g_tq_config under the mutex.
 * If the file is absent, disables TQ without error. */
static void gb_tq_read_conf(void)
{
    FILE *f = fopen(GB_TQ_CONF_PATH, "r");
    gb_tq_config_t cfg = {0};
    cfg.head_dim = 128;
    cfg.seed     = 42;
    cfg.ratio    = 1.0f;

    if (f) {
        char line[128];
        while (fgets(line, sizeof(line), f)) {
            int   iv = 0;
            float fv = 0.0f;
            if (sscanf(line, "enabled=%d", &iv) == 1)  cfg.enabled  = iv;
            if (sscanf(line, "bits=%d",    &iv) == 1)  cfg.bits     = iv;
            if (sscanf(line, "head_dim=%d",&iv) == 1)  cfg.head_dim = iv;
            if (sscanf(line, "seed=%d",    &iv) == 1)  cfg.seed     = iv;
            if (sscanf(line, "ratio=%f",   &fv) == 1)  cfg.ratio    = fv;
        }
        fclose(f);
    }

    pthread_mutex_lock(&g_tq_config_mutex);
    g_tq_config = cfg;
    pthread_mutex_unlock(&g_tq_config_mutex);

    /* Lazily open libgreenboost_tq.so when TQ is first enabled */
    if (cfg.enabled && !g_libtq) {
        g_libtq = dlopen("/usr/local/lib/libgreenboost_tq.so", RTLD_NOW | RTLD_LOCAL);
        if (g_libtq) {
            g_tq_init_fn            = (int (*)(int))
                                      dlsym(g_libtq, "gb_tq_init");
            g_tq_quantize_fn        = (int (*)(const void *, void *, size_t, int, int, void *))
                                      dlsym(g_libtq, "gb_tq_quantize");
            g_tq_dequantize_fn      = (int (*)(const void *, void *, size_t, int, int, void *))
                                      dlsym(g_libtq, "gb_tq_dequantize");
            g_tq_compressed_size_fn = (size_t (*)(size_t, int, int))
                                      dlsym(g_libtq, "gb_tq_compressed_size");
            if (g_tq_init_fn)
                g_tq_init_fn(0);  /* device 0 — auto-detected on first CUDA use */
        } else {
            fprintf(stderr,
                    "[GreenBoost] WARNING: libgreenboost_tq.so not found — TurboQuant disabled\n");
        }
    }
}

/* ENH-02: Shim-side cache of the last phase_reset_seq seen from the kernel.
 * gb_refresh_kv_reserve() compares this against gb_info.phase_reset_seq; a
 * change means Synapse CLI called GB_IOCTL_RESET_PHASE before a model swap. */
static _Atomic uint32_t g_last_phase_reset_seq = 0;

/* ENH-03: cuMemGetInfo cache — calling cuMemGetInfo on every overflow alloc
 * costs a CUDA driver round-trip (~1-2 µs each).  Cache the result and
 * refresh at most every GB_MEMINFO_REFRESH_ALLOCS allocs OR every
 * GB_MEMINFO_REFRESH_MS ms (whichever fires first). */
#define GB_MEMINFO_REFRESH_ALLOCS 16u  /* power-of-2 — cheap bitmask test */
#define GB_MEMINFO_REFRESH_MS     50ULL
static _Atomic size_t   g_cached_free_vram  = 0;
static _Atomic size_t   g_cached_total_vram = 0;
static _Atomic uint64_t g_cached_meminfo_ms = 0;

/* ------------------------------------------------------------------ */
/*  Allocation phase detector — distinguishes KV cache from weights    */
/*                                                                      */
/*  llama.cpp / Ollama loading sequence:                                */
/*    1. llama_model_load()              → many overflow allocs (weights)
/*    2. llama_new_context_with_model()  → 1-2 large allocs (KV cache)  */
/*                                                                      */
/*  State machine tracks this transition so KV allocs are automatically */
/*  classified as GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY instead of  */
/*  the default GB_ALLOC_WEIGHTS.                                       */
/*                                                                      */
/*  Enabled by default; disable with GREENBOOST_PHASE_DETECT=0.        */
/*  Override KV size threshold: GREENBOOST_KV_SIZE_THRESHOLD_MB=N      */
/* ------------------------------------------------------------------ */

typedef enum {
    GB_PHASE_INIT        = 0,  /* no overflow allocs yet                */
    GB_PHASE_MODEL_LOAD  = 1,  /* burst of weight allocs                */
    GB_PHASE_INFERENCE   = 2,  /* KV + activation allocs expected       */
    GB_PHASE_STEADY      = 3,  /* generation loop; activations dominate */
    GB_PHASE_IDLE        = 4,  /* no overflow allocs for GB_IDLE_TIMEOUT_MS */
    GB_PHASE_DEEP_IDLE   = 5,  /* IDLE for GB_DEEP_IDLE_TIMEOUT_MS — daemon unloads models */
} gb_alloc_phase_t;

/* Phase detector globals use C11 _Atomic to prevent torn reads/writes under
 * concurrent cudaMalloc calls from multiple CUDA streams.  memory_order_relaxed
 * is sufficient because the phase heuristic is a best-effort classifier and does
 * not require sequentially consistent ordering between unrelated alloc paths. */
static _Atomic int            g_alloc_phase    /* gb_alloc_phase_t */ = GB_PHASE_INIT;
static int                    g_phase_detect   = 1;    /* GREENBOOST_PHASE_DETECT */
/* Allocs above this threshold during INFERENCE phase are classified KV */
static size_t                 g_kv_size_threshold_bytes = 64ULL * 1024 * 1024;  /* 64 MB — catches small-model KV allocs */

/* Rolling average of overflow alloc sizes (exponential moving average,
 * α = 1/8) — used to detect the KV alloc as anomalously large. */
static _Atomic size_t        g_overflow_avg_bytes;   /* EMA of recent overflow sizes */
static _Atomic unsigned int  g_overflow_load_count;  /* allocs in MODEL_LOAD phase */

/* Timestamp of last overflow alloc — used for quiet-gap detection.
 * Stored as milliseconds from CLOCK_MONOTONIC.
 * MED-07: Renamed from g_last_overflow_ns (misleading — value was always ms). */
static _Atomic uint64_t      g_last_overflow_ms;

/* Timestamp of last ANY CUDA alloc (cudaMalloc / cuMemAlloc_v2 / cuMemAllocAsync).
 * Unlike g_last_overflow_ms which only tracks T2 overflow allocs, this tracks all
 * CUDA memory activity.  Used by gb_check_idle_phase() to detect that the GPU is
 * completely idle (not just that overflow has stopped) and skip directly to
 * DEEP_IDLE without the normal IDLE dwell, enabling instant T2/T3 reclaim. */
static _Atomic uint64_t      g_last_any_alloc_ms;

/* Minimum quiet gap (ms) after model loading before we consider the
 * next alloc to be an inference-phase (KV) allocation. */
#define GB_PHASE_QUIET_GAP_MS   400  /* 400 ms quiet → model load complete */

/* T2 pool cap during inference: leave 12% headroom for OS/desktop.
 * Applied once the phase reaches INFERENCE/STEADY — model loading uses the full pool. */
#define GB_T2_INFERENCE_CAP_PCT  88

/* Returns effective T2 pool cap in bytes.
 * During MODEL_LOAD and earlier: full gb_t2_pool_bytes (weights need all of it).
 * During INFERENCE/STEADY: 88% of gb_t2_pool_bytes to protect OS memory headroom. */
static inline size_t gb_effective_t2_cap(void)
{
    if (gb_t2_pool_bytes == 0) return 0;
    int phase = atomic_load_explicit(&g_alloc_phase, memory_order_relaxed);
    if (phase >= (int)GB_PHASE_INFERENCE)
        return gb_t2_pool_bytes * GB_T2_INFERENCE_CAP_PCT / 100;
    return gb_t2_pool_bytes;
}

/* T2 DDR headroom to advertise as free "virtual VRAM" in cuMemGetInfo / NVML hooks.
 * = T2 pool cap − already committed (Path A/B + UVM estimate) − safety reserve.
 * T3 (NVMe) is intentionally excluded: it is capacity for model loading, not
 * fast memory suitable for KV cache; including it inflates context-length
 * calculations and causes the KV cache to exceed available system RAM (OOM). */
static size_t gb_t2_free_to_report(void)
{
    if (gb_t2_pool_bytes == 0)
        return gb_virtual_vram_bytes;   /* no T2 info — keep legacy behaviour */
    size_t t2_used  = atomic_load_explicit(&gb_t2_overflow_bytes,       memory_order_relaxed);
    size_t uvm_used = atomic_load_explicit(&gb_uvm_estimated_ram_bytes, memory_order_relaxed);
    size_t committed = t2_used + uvm_used;
    size_t cap = (gb_t2_pool_bytes > gb_safety_reserve_bytes)
                  ? (gb_t2_pool_bytes - gb_safety_reserve_bytes) : 0;
    return (cap > committed) ? (cap - committed) : 0;
}

/* Idle timeout: if no overflow alloc occurs for this many ms while in STEADY
 * phase, transition to GB_PHASE_IDLE.  Override with GREENBOOST_IDLE_TIMEOUT_MS
 * env var; 0 = disabled. */
#define GB_IDLE_TIMEOUT_MS_DEFAULT       30000ULL    /* 30 seconds */
static _Atomic uint64_t g_idle_timeout_ms;           /* init in gb_shim_init */

/* Deep-idle timeout: if GB_PHASE_IDLE persists for this many additional ms,
 * transition to GB_PHASE_DEEP_IDLE and write /run/greenboost/phase so that the
 * greenboost-idle-reclaim daemon can call the Ollama API to unload models,
 * reclaiming both T1 VRAM (KV cache) and T2 RAM (weight overflow) safely.
 * Override with GREENBOOST_DEEP_IDLE_TIMEOUT_MS env var; 0 = disabled. */
#define GB_DEEP_IDLE_TIMEOUT_MS_DEFAULT  120000ULL   /* 2 minutes */
static _Atomic uint64_t g_deep_idle_timeout_ms;      /* init in gb_shim_init */

/* Millisecond timestamp when IDLE phase was entered (CLOCK_MONOTONIC).
 * Used by gb_check_idle_phase to compute IDLE → DEEP_IDLE elapsed time. */
static _Atomic uint64_t g_idle_entered_ms;

/* After this many overflow allocs without a quiet gap, still force
 * INFERENCE phase (handles models that overlap weight + KV loading). */
#define GB_PHASE_LOAD_COUNT_MAX 128

static uint64_t gb_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)(ts.tv_nsec / 1000000);
}

/*
 * gb_write_phase_file - write current phase state to /run/greenboost/phase.
 *
 * The greenboost-idle-reclaim daemon reads this file every 30 s to decide
 * when to call the Ollama API and unload models (releasing T1 VRAM + T2 RAM).
 * The file is created on shim init and removed on shim fini.
 *
 * Format (newline-terminated key=value pairs):
 *   phase=<INIT|MODEL_LOAD|INFERENCE|STEADY|IDLE|DEEP_IDLE>
 *   idle_ms=<ms elapsed since last overflow alloc, 0 if not idle>
 *   pid=<shim PID>
 *   ts=<CLOCK_REALTIME seconds>
 */
static void gb_write_phase_file(int phase, uint64_t idle_ms)
{
    static const char *phase_names[] = {
        "INIT", "MODEL_LOAD", "INFERENCE", "STEADY", "IDLE", "DEEP_IDLE"
    };
    const char *name = (phase >= 0 && phase <= 5) ? phase_names[phase] : "UNKNOWN";
    struct timespec rt;
    char buf[256];
    int fd, n;

    clock_gettime(CLOCK_REALTIME, &rt);
    n = snprintf(buf, sizeof(buf),
                 "phase=%s\nidle_ms=%llu\npid=%d\nts=%lld\n",
                 name,
                 (unsigned long long)idle_ms,
                 (int)getpid(),
                 (long long)rt.tv_sec);
    if (n <= 0 || n >= (int)sizeof(buf))
        return;

    /* O_TMPFILE + rename would be ideal but requires kernel support we may
     * not have in containers.  O_TRUNC on a small file is safe enough. */
    fd = open("/run/greenboost/phase", O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) {
        /* Directory may not exist yet — try to create it (best-effort). */
        mkdir("/run/greenboost", 0755);
        fd = open("/run/greenboost/phase", O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    }
    if (fd < 0)
        return;
    ssize_t _wr = write(fd, buf, (size_t)n);
    (void)_wr;
    close(fd);
}

/*
 * gb_idle_flush_weights - flush non-KV hash-table entries when Ollama has been
 * idle for g_idle_timeout_ms.  KV cache entries are kept so the loaded model
 * stays warm.  Called from gb_check_idle_phase after detecting STEADY→IDLE.
 */
static void gb_idle_flush_weights(void)
{
    gb_log("Phase → IDLE — flushing weight/activation T2 overflow");
    /* kv_only=1: keep GB_ALLOC_KV_CACHE entries, free everything else */
    gb_htable_flush(1);

    /* Tell the kernel to drop its remaining refs for non-KV buffers */
    if (gb_dev_fd >= 0) {
        struct gb_release_pid_req rreq = { .pid = 0, ._pad = 0 };
        ioctl(gb_dev_fd, GB_IOCTL_RELEASE_PID, &rreq);
    }
}

/*
 * gb_check_idle_phase - called from the prefetch thread on every 5 s wakeup.
 *
 * Two-tier idle detection:
 *
 *   STEADY → IDLE (g_idle_timeout_ms, default 2 min):
 *     No overflow alloc has occurred for 2 minutes.  The phase transitions to
 *     IDLE and the state is written to /run/greenboost/phase.  T2 weight
 *     overflow is NOT flushed here because active CUDA device pointers into the
 *     DMA-BUF mappings are still held by Ollama; invalidating them would cause
 *     CUDA_ERROR_ILLEGAL_ADDRESS on the next forward pass.
 *
 *   IDLE → DEEP_IDLE (g_deep_idle_timeout_ms, default 15 min):
 *     The workload has been silent for 15 additional minutes.  The phase file
 *     is updated to phase=DEEP_IDLE.  The greenboost-idle-reclaim daemon reads
 *     this and calls the Ollama REST API to gracefully unload all loaded models.
 *     Ollama then calls cudaFree on every tensor → shim's ht_remove() path
 *     closes the DMA-BUF fds → T2 RAM freed.  T1 VRAM (KV cache) is freed by
 *     the NVIDIA driver as Ollama releases its CUDA context.
 */
static void gb_check_idle_phase(void)
{
    uint64_t ito   = atomic_load_explicit(&g_idle_timeout_ms,      memory_order_relaxed);
    uint64_t dito  = atomic_load_explicit(&g_deep_idle_timeout_ms,  memory_order_relaxed);
    int      phase = atomic_load_explicit(&g_alloc_phase,           memory_order_relaxed);
    uint64_t now   = gb_now_ms();

    if (phase == (int)GB_PHASE_STEADY && ito > 0) {
        uint64_t last = atomic_load_explicit(&g_last_overflow_ms, memory_order_relaxed);
        if (now - last >= ito) {
            /* GPU-idle fast path: if ALL CUDA allocs (not just overflow) have also
             * been quiet for at least ito ms, the GPU is completely idle — no model
             * is actively being used.  Skip the IDLE dwell entirely and go straight
             * to DEEP_IDLE so the reclaim daemon unloads T1+T2+T3 immediately.
             *
             * Normal path (some native allocs still active): enter IDLE first.
             * This handles cases where CUDA keeps its context warm (compute graphs,
             * KV cache refreshes) even though no new inference requests are coming. */
            uint64_t last_any = atomic_load_explicit(&g_last_any_alloc_ms, memory_order_relaxed);
            int gpu_fully_idle = (last_any == 0 || now - last_any >= ito);

            if (gpu_fully_idle && dito > 0) {
                atomic_store_explicit(&g_alloc_phase, (int)GB_PHASE_DEEP_IDLE, memory_order_relaxed);
                gb_log("Phase → DEEP_IDLE (GPU fully idle for %llu ms — skipping IDLE dwell) — signalling reclaim daemon",
                       (unsigned long long)(now - last));
                gb_write_phase_file((int)GB_PHASE_DEEP_IDLE, now - last);
                /* Demote all our T2 buffers to LRU tail — idle session yields
                 * to any concurrent active session under memory pressure. */
                if (gb_dev_fd >= 0) {
                    struct gb_session_req sr = { .pid = 0, .reserved = 0 };
                    ioctl(gb_dev_fd, GB_IOCTL_SESSION_IDLE, &sr);
                }
            } else {
                atomic_store_explicit(&g_idle_entered_ms, now,                memory_order_relaxed);
                atomic_store_explicit(&g_alloc_phase,     (int)GB_PHASE_IDLE, memory_order_relaxed);
                gb_log("Phase → IDLE (no overflow for %llu ms, native allocs still active)",
                       (unsigned long long)ito);
                gb_write_phase_file((int)GB_PHASE_IDLE, now - last);
                if (gb_dev_fd >= 0) {
                    struct gb_session_req sr = { .pid = 0, .reserved = 0 };
                    ioctl(gb_dev_fd, GB_IOCTL_SESSION_IDLE, &sr);
                }
            }
        }
        return;
    }

    if (phase == (int)GB_PHASE_IDLE && dito > 0) {
        uint64_t entered = atomic_load_explicit(&g_idle_entered_ms, memory_order_relaxed);
        if (now - entered >= dito) {
            atomic_store_explicit(&g_alloc_phase, (int)GB_PHASE_DEEP_IDLE, memory_order_relaxed);
            gb_log("Phase → DEEP_IDLE (idle for %llu ms) — signalling reclaim daemon",
                   (unsigned long long)(now - entered));
            gb_write_phase_file((int)GB_PHASE_DEEP_IDLE, now - entered);
            /* Already in IDLE — SESSION_IDLE was sent then; no repeat needed. */
        }
    }
}

/* Called on every overflow alloc; returns the GB_ALLOC_* flags to use.
 * Each atomic variable is accessed with memory_order_relaxed — the phase
 * heuristic is best-effort and does not require cross-thread ordering. */
static uint32_t gb_phase_classify(size_t bytesize)
{
    uint64_t now_ms;
    uint64_t gap_ms;
    int      phase;
    size_t   avg;
    unsigned int load_count;

    if (!g_phase_detect)
        return GB_ALLOC_WEIGHTS;  /* phase detection disabled */

    now_ms = gb_now_ms();
    phase  = atomic_load_explicit(&g_alloc_phase, memory_order_relaxed);

    switch ((gb_alloc_phase_t)phase) {
    case GB_PHASE_INIT:
        /* First overflow alloc ever — enter model loading phase */
        atomic_store_explicit(&g_alloc_phase, GB_PHASE_MODEL_LOAD, memory_order_relaxed);
        atomic_store_explicit(&g_overflow_load_count, 1u, memory_order_relaxed);
        atomic_store_explicit(&g_overflow_avg_bytes, bytesize, memory_order_relaxed);
        atomic_store_explicit(&g_last_overflow_ms, now_ms, memory_order_relaxed);
        gb_log("Phase → MODEL_LOAD (first overflow alloc, %zu MB)", bytesize >> 20);
        return GB_ALLOC_WEIGHTS;

    case GB_PHASE_MODEL_LOAD:
        gap_ms = now_ms - atomic_load_explicit(&g_last_overflow_ms, memory_order_relaxed);
        atomic_store_explicit(&g_last_overflow_ms, now_ms, memory_order_relaxed);
        load_count = atomic_fetch_add_explicit(&g_overflow_load_count, 1u, memory_order_relaxed) + 1u;
        /* Update EMA: avg = (avg * 7 + size) / 8 */
        avg = (atomic_load_explicit(&g_overflow_avg_bytes, memory_order_relaxed) * 7 + bytesize) / 8;
        atomic_store_explicit(&g_overflow_avg_bytes, avg, memory_order_relaxed);

        /* Transition to INFERENCE when either:
         *   (a) quiet gap: no overflow for >= GB_PHASE_QUIET_GAP_MS, OR
         *   (b) this alloc is >= 4x the rolling average (KV is 1-2 huge blocks)
         *   (c) forced after GB_PHASE_LOAD_COUNT_MAX weight allocs             */
        if (gap_ms >= GB_PHASE_QUIET_GAP_MS ||
            (avg > 0 && bytesize >= 4 * avg) ||
            load_count >= GB_PHASE_LOAD_COUNT_MAX) {

            atomic_store_explicit(&g_alloc_phase, GB_PHASE_INFERENCE, memory_order_relaxed);
            gb_log("Phase → INFERENCE (gap=%llums, avg=%zuMB, count=%u, this=%zuMB)",
                   (unsigned long long)gap_ms,
                   avg >> 20, load_count, bytesize >> 20);

            /* This alloc that triggered the transition is the KV alloc */
            if (bytesize >= g_kv_size_threshold_bytes) {
                gb_log("Phase classify: INFERENCE KV alloc %zu MB → GB_ALLOC_KV_CACHE|T1_PRIORITY",
                       bytesize >> 20);
                return GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY;
            }
            return GB_ALLOC_ACTIVATIONS;
        }
        return GB_ALLOC_WEIGHTS;

    case GB_PHASE_INFERENCE:
        gap_ms = now_ms - atomic_load_explicit(&g_last_overflow_ms, memory_order_relaxed);
        atomic_store_explicit(&g_last_overflow_ms, now_ms, memory_order_relaxed);

        /* A new large quiet-gap alloc in INFERENCE could be a model reload;
         * reset to MODEL_LOAD if we see a burst after a long gap. */
        if (gap_ms >= 5000ULL && bytesize < g_kv_size_threshold_bytes / 4) {
            atomic_store_explicit(&g_alloc_phase, GB_PHASE_MODEL_LOAD, memory_order_relaxed);
            atomic_store_explicit(&g_overflow_load_count, 1u, memory_order_relaxed);
            atomic_store_explicit(&g_overflow_avg_bytes, bytesize, memory_order_relaxed);
            gb_log("Phase → MODEL_LOAD reset (gap=%llums, small alloc %zu MB — likely model reload)",
                   (unsigned long long)gap_ms, bytesize >> 20);
            return GB_ALLOC_WEIGHTS;
        }

        /* Large allocs in INFERENCE phase = KV cache */
        if (bytesize >= g_kv_size_threshold_bytes) {
            gb_log("Phase classify: INFERENCE KV alloc %zu MB → GB_ALLOC_KV_CACHE|T1_PRIORITY",
                   bytesize >> 20);
            /* After 2 KV allocs (K + V tensors), enter STEADY */
            atomic_store_explicit(&g_alloc_phase, GB_PHASE_STEADY, memory_order_relaxed);
            return GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY;
        }
        gb_log("Phase classify: INFERENCE small alloc %zu MB → GB_ALLOC_ACTIVATIONS",
               bytesize >> 20);
        return GB_ALLOC_ACTIVATIONS;

    case GB_PHASE_STEADY:
        /* Generation loop: small ephemeral activation buffers */
        if (bytesize >= g_kv_size_threshold_bytes) {
            /* Unexpected large alloc in steady state — could be a new KV context */
            gb_log("Phase classify: STEADY large alloc %zu MB → GB_ALLOC_KV_CACHE|T1_PRIORITY",
                   bytesize >> 20);
            return GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY;
        }
        return GB_ALLOC_ACTIVATIONS;

    case GB_PHASE_IDLE:
    case GB_PHASE_DEEP_IDLE:
        /* New overflow alloc after idle/deep-idle — model is being reloaded.
         * Re-enter MODEL_LOAD so weights are correctly classified and the
         * KV reserve activates only when inference starts again.
         * Reset g_kv_allocated_t1_bytes: the previous session's KV tracking
         * is stale; a fresh model load starts with no KV in T1. */
        atomic_store_explicit(&g_alloc_phase, GB_PHASE_MODEL_LOAD, memory_order_relaxed);
        atomic_store_explicit(&g_overflow_load_count, 1u, memory_order_relaxed);
        atomic_store_explicit(&g_overflow_avg_bytes, bytesize, memory_order_relaxed);
        atomic_store_explicit(&g_last_overflow_ms, now_ms, memory_order_relaxed);
        atomic_store_explicit(&g_kv_allocated_t1_bytes, (size_t)0, memory_order_relaxed);
        gb_log("Phase → MODEL_LOAD (resumed from %s, %zu MB)",
               phase == (int)GB_PHASE_DEEP_IDLE ? "DEEP_IDLE" : "IDLE",
               bytesize >> 20);
        gb_write_phase_file((int)GB_PHASE_MODEL_LOAD, 0);
        /* Session is active again — promote our T2 buffers back to LRU head
         * so they are evicted last if another session competes for T2 space. */
        if (gb_dev_fd >= 0) {
            struct gb_session_req sr = { .pid = 0, .reserved = 0 };
            ioctl(gb_dev_fd, GB_IOCTL_SESSION_ACTIVE, &sr);
        }
        return GB_ALLOC_WEIGHTS;

    /* All six enum values are covered above. */
    }
    __builtin_unreachable();
}

/* ------------------------------------------------------------------ */
/*  GreenBoost /dev/greenboost helper                                  */
/* ------------------------------------------------------------------ */

static int gb_open_device(void)
{
    pthread_mutex_lock(&gb_dev_lock);
    if (gb_dev_fd < 0) {
        gb_dev_fd = open("/dev/greenboost", O_RDWR | O_CLOEXEC);
        if (gb_dev_fd < 0)
            fprintf(stderr, "[GreenBoost] Cannot open /dev/greenboost: %m\n");
    }
    pthread_mutex_unlock(&gb_dev_lock);
    return gb_dev_fd;
}

/* ------------------------------------------------------------------ */
/*  KV cache T1 reservation refresh                                    */
/* ------------------------------------------------------------------ */

/* Read kv_reserve_mb from the kernel module via GB_IOCTL_GET_INFO and
 * update g_kv_reserve_bytes.  Called at init and every GB_KV_REFRESH_INTERVAL
 * allocations so Synapse CLI's GB_IOCTL_SET_KV_RESERVE takes effect without
 * restarting Ollama.  Non-blocking: silently skips if /dev/greenboost absent. */
static void gb_refresh_kv_reserve(void)
{
    int fd;
    struct gb_info info;

    fd = gb_open_device();
    if (fd < 0)
        return;

    memset(&info, 0, sizeof(info));
    if (ioctl(fd, GB_IOCTL_GET_INFO, &info) == 0) {
        /* Only apply kernel module value if env-var / auto-scaler has not
         * already set g_kv_reserve_bytes.  A stale kv_reserve_mb=0 in the
         * module (e.g. after insmod without the param) would otherwise zero
         * the auto-scaled value every GB_KV_REFRESH_INTERVAL allocs. */
        if (!g_kv_reserve_from_env)
            atomic_store_explicit(&g_kv_reserve_bytes,
                                  (size_t)info.kv_reserve_mb * 1024ULL * 1024ULL,
                                  memory_order_relaxed);

        /* ENH-02: Phase reset detection — Synapse CLI calls GB_IOCTL_RESET_PHASE
         * before swapping models to restart phase detection from INIT.
         * Compare the kernel's monotonically-incrementing sequence number against
         * our cached value; a change means a reset was requested. */
        {
            uint32_t kernel_seq = info.phase_reset_seq;
            uint32_t local_seq  = atomic_load_explicit(&g_last_phase_reset_seq,
                                                       memory_order_relaxed);
            if (kernel_seq != local_seq) {
                atomic_store_explicit(&g_alloc_phase, GB_PHASE_INIT,
                                      memory_order_relaxed);
                atomic_store_explicit(&g_kv_allocated_t1_bytes, (size_t)0,
                                      memory_order_relaxed);
                atomic_store_explicit(&g_last_phase_reset_seq, kernel_seq,
                                      memory_order_relaxed);
                gb_log("Phase reset by kernel seq=%u → INIT; kv_t1 cleared", kernel_seq);
            }
        }
        gb_log("KV T1 reserve refreshed from kernel: %u MB (kv_t2_mb=%u)",
               info.kv_reserve_mb, info.kv_t2_mb);
        /* Warn when KV cache has spilled from T1 into T2.
         * kv_t2_mb tracks GB_ALLOC_KV_CACHE buffers explicitly in T2 DDR.
         * A non-zero value means KV is bandwidth-limited by PCIe (T2)
         * instead of running at native T1 VRAM speed. */
        if (info.kv_t2_mb > 0) {
            fprintf(stderr,
                "[GreenBoost] NOTE: %u MB of KV cache is in T2 DDR "
                "instead of T1 VRAM. "
                "Consider increasing kv_reserve_mb or reducing num_ctx.\n",
                info.kv_t2_mb);
        }
        if (info.nvme_t3_allocated_mb > 0) {
            fprintf(stderr,
                "[GreenBoost] WARNING: T3 safety-net active — %llu MB of model data "
                "is on NVMe (slow). Inference will be slow. "
                "Reduce num_ctx or use a smaller model.\n",
                (unsigned long long)info.nvme_t3_allocated_mb);
        }
    }

    /* Refresh TurboQuant config alongside KV reserve — same poll interval */
    gb_tq_read_conf();
}

/* ------------------------------------------------------------------ */
/*  DMA-BUF import path: allocate system RAM via GreenBoost, import as CUDA  */
/* ------------------------------------------------------------------ */

/*
 * Path A0 — true zero-copy DMA-BUF import via cudaImportExternalMemory.
 *
 * Flow: GB_IOCTL_ALLOC (kernel allocates hugepage-backed buffer, exports DMA-BUF fd)
 *       → cudaImportExternalMemory(OpaqueFd)  [CUDA takes fd ownership]
 *       → cudaExternalMemoryGetMappedBuffer   [returns CUdeviceptr directly]
 *
 * Advantages over Path A (cuMemHostRegister):
 *   - No anonymous mmap() + pin_user_pages() round-trip.
 *   - No cuMemHostRegister overhead; CUDA drives its own IOMMU mapping from
 *     the kernel DMA-BUF scatter-gather table (2 MB compound pages).
 *   - The returned CUdeviceptr is a first-class device pointer, not a
 *     host-registered alias — no CUDA driver internal remapping needed.
 *   - cudaDestroyExternalMemory() handles full teardown on free.
 *
 * Requires: real_cudaImportExternalMemory and real_cudaExternalMemoryGetMappedBuffer
 *           resolved from libcudart.so (CUDA runtime ≥ 10.0).
 */
/* Per-path allocation counters — readable in debug banner to verify Path A0 is active */
static volatile unsigned int gb_path_a0_count = 0;
static volatile unsigned int gb_path_a_count  = 0;
static volatile unsigned int gb_path_b_count  = 0;
static volatile unsigned int gb_path_c_count  = 0;

/* Set to 1 on first cudaImportExternalMemory error to permanently skip Path A0.
 * Prevents sticky CUDA error 999 from corrupting the CUDA context on every alloc. */
static volatile int gb_a0_disabled = 0;

/* Minimum allocation size for Path A0 (cudaImportExternalMemory).
 * For < 4 MB (< 2 hugepages), the DMA-BUF export+import overhead is
 * disproportionate; Path B (cuMemHostRegister) is cheaper per-call. */
#define GB_PATH_A0_MIN_BYTES  (4ULL * 1024 * 1024)

/* ------------------------------------------------------------------ */
/*  Shim stats file — written to /run/greenboost/shim_stats            */
/*  Lets greenboost_setup.sh status panel show the real active path    */
/*  instead of guessing from sysfs heuristics.                         */
/* ------------------------------------------------------------------ */

#define GB_STATS_DIR       "/run/greenboost"
#define GB_STATS_FILE      "/run/greenboost/shim_stats"
#define GB_STATS_FILE_TMP  "/tmp/greenboost_shim_stats"

/* Resolved at first write: /run/greenboost/ if writable, else /tmp/ */
static const char *gb_stats_dir  = NULL;
static const char *gb_stats_file = NULL;

static void gb_stats_resolve_path(void)
{
    if (gb_stats_file) return;  /* already resolved */

    /* Try /run/greenboost first (preferred — readable by status script) */
    if (mkdir(GB_STATS_DIR, 0777) == 0 || errno == EEXIST) {
        /* chmod in case directory already existed with wrong perms */
        chmod(GB_STATS_DIR, 0777);
        /* Test writeability */
        char probe[128];
        snprintf(probe, sizeof(probe), "%s/.probe.%d", GB_STATS_DIR, (int)getpid());
        FILE *fp = fopen(probe, "w");
        if (fp) { fclose(fp); unlink(probe); gb_stats_dir = GB_STATS_DIR; gb_stats_file = GB_STATS_FILE; return; }
    }
    /* Fall back to /tmp — always writable */
    gb_stats_dir  = "/tmp";
    gb_stats_file = GB_STATS_FILE_TMP;
}

static void gb_write_stats(void)
{
    char tmp_path[128];
    FILE *f;

    gb_stats_resolve_path();

    snprintf(tmp_path, sizeof(tmp_path), "%s/.shim_stats.tmp.%d",
             gb_stats_dir, (int)getpid());
    f = fopen(tmp_path, "w");
    if (!f) return;

    unsigned int a0 = gb_path_a0_count;
    unsigned int a  = gb_path_a_count;
    unsigned int b  = gb_path_b_count;
    unsigned int c  = gb_path_c_count;
    const char *active =
        (a0 > 0) ? "A0" :
        (a  > 0) ? "A"  :
        (b  > 0) ? "B"  :
        (c  > 0) ? "C"  : "none";

    static const char *const _phase_names[] = {
        "INIT", "MODEL_LOAD", "INFERENCE", "STEADY", "IDLE", "DEEP_IDLE"
    };
    int  _phase_idx  = atomic_load_explicit(&g_alloc_phase, memory_order_relaxed);
    if (_phase_idx < 0 || _phase_idx > 5) _phase_idx = 0;
    size_t _kv_rsv   = atomic_load_explicit(&g_kv_reserve_bytes,      memory_order_relaxed);
    size_t _kv_t1    = atomic_load_explicit(&g_kv_allocated_t1_bytes,  memory_order_relaxed);
    size_t _kv_eff   = (_kv_t1 >= _kv_rsv) ? 0 : (_kv_rsv - _kv_t1);

    fprintf(f, "pid=%d\n",                    (int)getpid());
    fprintf(f, "path_a0_count=%u\n",          a0);
    fprintf(f, "path_a_count=%u\n",           a);
    fprintf(f, "path_b_count=%u\n",           b);
    fprintf(f, "path_c_count=%u\n",           c);
    fprintf(f, "initialized=%d\n",            initialized);
    fprintf(f, "virtual_vram_mb=%zu\n",       gb_virtual_vram_bytes >> 20);
    fprintf(f, "active_path=%s\n",            active);
    fprintf(f, "phase=%s\n",                  _phase_names[_phase_idx]);
    fprintf(f, "kv_reserve_nominal_mb=%zu\n", _kv_rsv >> 20);
    fprintf(f, "kv_reserve_effective_mb=%zu\n", _kv_eff >> 20);
    fprintf(f, "kv_t1_tracked_mb=%zu\n",      _kv_t1 >> 20);
    fprintf(f, "vram_headroom_mb=%zu\n",       vram_headroom_bytes >> 20);
    fprintf(f, "timestamp=%ld\n",             (long)time(NULL));
    fclose(f);
    rename(tmp_path, gb_stats_file);
}

static void gb_maybe_write_stats(void)
{
    /* CRIT-05: Use _Atomic + CAS so exactly one thread wins the write slot
     * per interval.  ENH-05: switched from time(NULL) (1-second resolution)
     * to CLOCK_MONOTONIC so high-throughput sessions can use sub-second
     * intervals (default 250 ms; GREENBOOST_STATS_INTERVAL_MS to override). */
    static _Atomic uint64_t gb_last_stats_ms = 0;
    struct timespec ts;
    uint64_t now_ms, prev_ms, interval;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    now_ms   = (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
    interval = g_stats_interval_ms;
    prev_ms  = atomic_load_explicit(&gb_last_stats_ms, memory_order_relaxed);

    if (now_ms - prev_ms >= interval) {
        /* CAS: only the one thread that successfully swaps prev→now wins */
        if (atomic_compare_exchange_strong_explicit(&gb_last_stats_ms, &prev_ms, now_ms,
                                                    memory_order_relaxed,
                                                    memory_order_relaxed)) {
            gb_write_stats();
        }
    }
}

static CUresult gb_alloc_via_external_mem(CUdeviceptr *dptr, size_t bytesize,
                                          cudaExternalMemory_t *ext_mem_out,
                                          uint32_t alloc_flags)
{
    struct gb_alloc_req req;
    struct cudaExternalMemoryHandleDesc hdesc;
    struct cudaExternalMemoryBufferDesc bdesc;
    cudaExternalMemory_t ext_mem;
    void *mapped_ptr = NULL;
    cudaError_t cret;
    int dev_fd;

    if (!real_cudaImportExternalMemory || !real_cudaExternalMemoryGetMappedBuffer)
        return CUDA_ERROR_NOT_SUPPORTED;

    dev_fd = gb_open_device();
    if (dev_fd < 0)
        return CUDA_ERROR_NOT_SUPPORTED;

    memset(&req, 0, sizeof(req));
    req.size  = bytesize;
    req.flags = alloc_flags;  /* was hardcoded GB_ALLOC_WEIGHTS — now forwarded from caller */

    if (ioctl(dev_fd, GB_IOCTL_ALLOC, &req) < 0) {
        fprintf(stderr, "[GreenBoost] GB_IOCTL_ALLOC failed for %zu MB: %m\n",
                bytesize >> 20);
        /* dev_fd == gb_dev_fd: persistent cached fd — do NOT close here.
         * Closing it would invalidate gb_dev_fd, breaking all subsequent
         * Path A0 allocations (EBADF on next ioctl).  The fd is owned by
         * gb_open_device() / gb_shim_fini(). */
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    /* Do NOT close dev_fd.  The device node fd (gb_dev_fd) is persistent.
     * CUDA takes ownership of req.fd (the DMA-BUF fd), not dev_fd. */
    /* req.fd is now a DMA-BUF fd in our file table */

    memset(&hdesc, 0, sizeof(hdesc));
    hdesc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
    hdesc.handle.fd = req.fd;   /* fd ownership transfers to CUDA on success */
    hdesc.size      = (unsigned long long)bytesize;

    cret = real_cudaImportExternalMemory(&ext_mem, &hdesc);
    if (cret != CUDA_SUCCESS) {
        fprintf(stderr, "[GreenBoost] cudaImportExternalMemory FAILED ret=%d for %zu MB"
                " — disabling Path A0 permanently to avoid sticky CUDA error\n",
                cret, bytesize >> 20);
        close(req.fd);  /* fd not consumed — close it ourselves */
        gb_a0_disabled = 1;
        /* Clear the sticky CUDA runtime error so Path A's cuMemHostRegister
         * does not inherit the poisoned context from this failure. */
        if (real_cudaGetLastError)
            real_cudaGetLastError();
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    /* From here CUDA owns req.fd — do not close it */

    memset(&bdesc, 0, sizeof(bdesc));
    bdesc.offset = 0;
    bdesc.size   = (unsigned long long)bytesize;

    cret = real_cudaExternalMemoryGetMappedBuffer(&mapped_ptr, ext_mem, &bdesc);
    if (cret != CUDA_SUCCESS) {
        fprintf(stderr, "[GreenBoost] cudaExternalMemoryGetMappedBuffer FAILED ret=%d for %zu MB\n",
                cret, bytesize >> 20);
        if (real_cudaDestroyExternalMemory)
            real_cudaDestroyExternalMemory(ext_mem);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    *dptr        = (CUdeviceptr)(uintptr_t)mapped_ptr;
    *ext_mem_out = ext_mem;
    gb_log("Path A0 (cudaImportExternalMemory): %zu MB at cuda_ptr=0x%llx ext_mem=%p",
           bytesize >> 20, (unsigned long long)*dptr, (void *)ext_mem);
    return CUDA_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  Constructor — runs before main()                                    */
/* ------------------------------------------------------------------ */

/* Returns MemAvailable from /proc/meminfo in bytes, or 0 on error.
 * Mirrors the kernel's si_mem_available() used in gb_alloc_buf() and
 * gb_pin_user_buf() safety checks. */
static size_t gb_get_mem_available(void)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[128];
    size_t result = 0;
    while (fgets(line, sizeof(line), f)) {
        unsigned long long kb;
        if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
            result = (size_t)kb * 1024ULL;
            break;
        }
    }
    fclose(f);
    return result;
}

/* Standalone functions for reading sysfs attributes */
static int read_sysfs_int(const char *path)
{
    char buf[32];
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    if (fgets(buf, sizeof(buf), f) == NULL) {
        fclose(f);
        return -1;
    }
    fclose(f);
    return atoi(buf);
}

/* Returns 1 if sysfs value == 1, 0 otherwise */
static int read_sysfs_bool(const char *path)
{
    int val = read_sysfs_int(path);
    return (val == 1) ? 1 : 0;
}

/* Returns 0 on success, -1 on error */
static int read_sysfs_string(const char *path, char *out, size_t maxlen)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    if (fgets(out, maxlen, f) == NULL) {
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Gaming NVML mode — activated by GREENBOOST_VULKAN=1 (Proton games)  */
/*                                                                       */
/*  The shim stays CUDA-inert in Vulkan games (no libcuda.so.1), but    */
/*  MangoHud reads VRAM from NVML.  Loading NVML and resolving the      */
/*  nvmlDeviceGetMemoryInfo* hooks here lets MangoHud see virtual VRAM   */
/*  via the dlsym interceptor without touching any CUDA path.            */
/* ------------------------------------------------------------------ */

static void load_nvml_for_gaming(void)
{
    typedef unsigned int (*pfn_nvmlInit_t)(void);
    typedef unsigned int (*pfn_nvmlGetHandleByIndex_t)(unsigned int, nvmlDevice_t *);

    void *libnvml = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!libnvml) libnvml = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_GLOBAL);
    if (!libnvml) return;

    real_nvmlDeviceGetMemoryInfo    = (pfn_nvmlDeviceGetMemoryInfo)
        dlsym(libnvml, "nvmlDeviceGetMemoryInfo");
    real_nvmlDeviceGetMemoryInfo_v2 = (pfn_nvmlDeviceGetMemoryInfo_v2)
        dlsym(libnvml, "nvmlDeviceGetMemoryInfo_v2");
    real_nvmlDeviceGetMemoryInfo_v3 = (pfn_nvmlDeviceGetMemoryInfo_v3)
        dlsym(libnvml, "nvmlDeviceGetMemoryInfo_v3");

    if (!real_nvmlDeviceGetMemoryInfo &&
        !real_nvmlDeviceGetMemoryInfo_v2 &&
        !real_nvmlDeviceGetMemoryInfo_v3)
        return;

    /* Probe physical VRAM before our hooks are active. */
    pfn_nvmlInit_t nvml_init = (pfn_nvmlInit_t)dlsym(libnvml, "nvmlInit_v2");
    if (!nvml_init) nvml_init = (pfn_nvmlInit_t)dlsym(libnvml, "nvmlInit");
    pfn_nvmlGetHandleByIndex_t nvml_get_handle =
        (pfn_nvmlGetHandleByIndex_t)dlsym(libnvml, "nvmlDeviceGetHandleByIndex_v2");
    if (!nvml_get_handle)
        nvml_get_handle = (pfn_nvmlGetHandleByIndex_t)
            dlsym(libnvml, "nvmlDeviceGetHandleByIndex");

    if (nvml_init && nvml_get_handle && real_nvmlDeviceGetMemoryInfo) {
        if (nvml_init() == 0) {
            nvmlDevice_t dev = NULL;
            if (nvml_get_handle(0, &dev) == 0 && dev) {
                nvmlMemory_t mem = {0};
                if (real_nvmlDeviceGetMemoryInfo(dev, &mem) == 0 && mem.total > 0)
                    gb_physical_vram_bytes = (size_t)mem.total;
            }
        }
    }

    /* Read virtual (T2 DDR) pool size from kernel module — authoritative source.
     * GREENBOOST_VIRTUAL_VRAM_MB env var override is already applied at this
     * point (parsed earlier in gb_shim_init before this call). */
    int virt_gb = read_sysfs_int("/sys/module/greenboost/parameters/virtual_vram_gb");
    if (virt_gb > 0)
        gb_virtual_vram_bytes = (size_t)virt_gb * 1024ULL * 1024ULL * 1024ULL;

    /* Add T3 NVMe pool — same logic as main CUDA path. */
    {
        int nvme_gb = read_sysfs_int("/sys/module/greenboost/parameters/nvme_pool_gb");
        if (nvme_gb > 0)
            gb_virtual_vram_bytes += (size_t)nvme_gb * 1024ULL * 1024ULL * 1024ULL;
    }

    nvml_hooks_active = 1;
    if (gb_debug)
        fprintf(stderr,
                "[GreenBoost] Gaming NVML mode active — MangoHud will see +%zu GB virtual VRAM\n",
                gb_virtual_vram_bytes >> 30);
}

__attribute__((constructor))
static void gb_shim_init(void)
{
    void *libcuda, *libcudart = NULL;
    const char *env;
    uint32_t i;
    int forced;

    /* Hard opt-out: set GREENBOOST_DISABLE=1 to keep the shim completely inert
     * for this process.  Useful as a Steam launch option for games that have
     * conflicts: GREENBOOST_DISABLE=1 %command% */
    if (getenv("GREENBOOST_DISABLE"))
        return;

    /* Parse ALL env vars before any early return so GREENBOOST_DEBUG is
     * available regardless of which activation path is taken. */
    env = getenv("GREENBOOST_USE_DMA_BUF");
    if (env) gb_use_dmabuf = (env[0] != '0');

    /* AUD-08: Container detection — cache result at init to avoid an open()
     * syscall on every alloc in Docker/LXC where /dev/greenboost is absent.
     * If running inside a container, disable Path A immediately so the alloc
     * hot path skips the open() attempt entirely and goes straight to Path B. */
    if (gb_use_dmabuf) {
        int is_container = 0;
        /* /.dockerenv is created by Docker for every container */
        if (access("/.dockerenv", F_OK) == 0)
            is_container = 1;
        /* cgroup v2: check if we're in a container namespace */
        if (!is_container) {
            FILE *cg = fopen("/proc/1/cgroup", "r");
            if (cg) {
                char line[256];
                while (fgets(line, sizeof(line), cg)) {
                    if (strstr(line, "docker") || strstr(line, "lxc") ||
                        strstr(line, "kubepods") || strstr(line, "containerd")) {
                        is_container = 1;
                        break;
                    }
                }
                fclose(cg);
            }
        }
        if (is_container) {
            gb_use_dmabuf = 0;
            if (gb_debug)
                fprintf(stderr, "[GreenBoost] Container detected — Path A disabled, using Path B/C\n");
        }
    }

    env = getenv("GREENBOOST_VRAM_HEADROOM_MB");
    int headroom_from_env = 0;
    if (env) { vram_headroom_bytes = (size_t)gb_atoll(env) * 1024ULL * 1024ULL; headroom_from_env = 1; }

    /* ENH-05: Configurable stats write interval (ms); 0 or negative → reset to default */
    env = getenv("GREENBOOST_STATS_INTERVAL_MS");
    if (env) {
        long long iv = gb_atoll(env);
        g_stats_interval_ms = (iv > 0) ? (uint64_t)iv : 250ULL;
    }

    env = getenv("GREENBOOST_KV_RESERVE_MB");
    if (env) {
        atomic_store_explicit(&g_kv_reserve_bytes,
                              (size_t)gb_atoll(env) * 1024ULL * 1024ULL,
                              memory_order_relaxed);
        g_kv_reserve_from_env = 1;
    }

    env = getenv("GREENBOOST_VIRTUAL_VRAM_MB");
    if (env) gb_virtual_vram_bytes = (size_t)gb_atoll(env) * 1024ULL * 1024ULL;

    env = getenv("GREENBOOST_DEBUG");
    if (env && env[0] == '1') gb_debug = 1;

    env = getenv("GREENBOOST_NO_HOSTREG");
    if (env && env[0] == '1') gb_no_hostreg = 1;

    env = getenv("GREENBOOST_KV_OVERFLOW");
    if (env && env[0] == '1') g_kv_overflow_mode = 1;

    env = getenv("GREENBOOST_PHASE_DETECT");
    if (env && env[0] == '0') g_phase_detect = 0;

    env = getenv("GREENBOOST_KV_SIZE_THRESHOLD_MB");
    if (env) g_kv_size_threshold_bytes = (size_t)gb_atoll(env) * 1024ULL * 1024ULL;

    /* Idle timeout (STEADY → IDLE): default 2 min, 0 = disabled. */
    {
        uint64_t ito = GB_IDLE_TIMEOUT_MS_DEFAULT;
        env = getenv("GREENBOOST_IDLE_TIMEOUT_MS");
        if (env) {
            long long v = gb_atoll(env);
            ito = (v >= 0) ? (uint64_t)v : GB_IDLE_TIMEOUT_MS_DEFAULT;
        }
        atomic_store_explicit(&g_idle_timeout_ms, ito, memory_order_relaxed);
    }

    /* Deep-idle timeout (IDLE → DEEP_IDLE): default 15 min, 0 = disabled.
     * When DEEP_IDLE is reached the reclaim daemon unloads Ollama models so
     * T1 VRAM (KV cache) and T2 RAM (weight overflow) are both freed. */
    {
        uint64_t dito = GB_DEEP_IDLE_TIMEOUT_MS_DEFAULT;
        env = getenv("GREENBOOST_DEEP_IDLE_TIMEOUT_MS");
        if (env) {
            long long v = gb_atoll(env);
            dito = (v >= 0) ? (uint64_t)v : GB_DEEP_IDLE_TIMEOUT_MS_DEFAULT;
        }
        atomic_store_explicit(&g_deep_idle_timeout_ms, dito, memory_order_relaxed);
    }

    /* Create /run/greenboost/ and write initial phase file for the daemon. */
    mkdir("/run/greenboost", 0755);
    gb_write_phase_file((int)GB_PHASE_INIT, 0);

    /* Gaming NVML mode: GREENBOOST_VULKAN=1 is set by the Proton wrapper for
     * every game.  Activate NVML hooks so MangoHud sees virtual VRAM without
     * requiring full CUDA initialization.  Runs even if CUDA is absent — the
     * shim stays CUDA-inert but NVML reporting is live. */
    if (getenv("GREENBOOST_VULKAN"))
        load_nvml_for_gaming();

    forced = (getenv("GREENBOOST_ACTIVE") != NULL);

/* Stage 1: RTLD_NOLOAD — check whether libcuda.so.1 is already resident.
     * This never triggers CUDA driver initialisation (no dlopen side-effects),
     * so it is safe in any process: GDM, shells, systemd helpers, etc.
     * Succeeds for apps that link libcuda statically (llama.cpp) — they get
     * automatic transparent injection with no wrapper needed. */
    libcuda = dlopen("libcuda.so.1", RTLD_NOLOAD | RTLD_NOW | RTLD_GLOBAL);

    /* Stage 2: explicit opt-in for apps that load CUDA lazily via dlopen at
     * runtime (Ollama, vLLM, PyTorch).  The Ollama/llama-server systemd service
     * units set GREENBOOST_ACTIVE=1; the greenboost-run wrapper does too for
     * CLI use.  GDM, shells, and helpers never reach this branch. */
    if (!libcuda) {
        if (!forced) {
            /* Not a CUDA process and not opted in — shim stays inert. */
            return;
        }
        libcuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!libcuda) {
            if (gb_debug)
                fprintf(stderr, "[GreenBoost] libcuda.so.1 not found — shim inactive\n");
            return;
        }
    }

    /* Initialize lock arrays */
    for (i = 0; i < HT_LOCKS; i++)
        pthread_mutex_init(&ht_locks[i], NULL);

    /* libcudart search order:
     *   1. GREENBOOST_CUDART_PATH env var — explicit override (escape hatch for non-standard installs)
     *   2. Unversioned / versioned names — found via LD_LIBRARY_PATH or ldconfig
     *   3. System CUDA toolkit paths — llama.cpp uses system CUDA, not Ollama-bundled
     *   4. Ollama-bundled paths — Ollama ships its own libcudart under /usr/local/lib/ollama/
     */
    {
        const char *cudart_override = getenv("GREENBOOST_CUDART_PATH");
        if (cudart_override) {
            libcudart = dlopen(cudart_override, RTLD_NOW | RTLD_GLOBAL);
            if (!libcudart)
                fprintf(stderr, "[GreenBoost] WARNING: GREENBOOST_CUDART_PATH=%s failed: %s\n",
                        cudart_override, dlerror());
        }

        if (!libcudart) {
            static const char *cudart_paths[] = {
                /* unversioned / versioned — resolved via LD_LIBRARY_PATH or ldconfig */
                "libcudart.so",
                "libcudart.so.13",
                "libcudart.so.12",
                /* system CUDA toolkit — standard install locations for llama.cpp */
                "/usr/local/cuda/lib64/libcudart.so",
                "/usr/local/cuda-13/lib64/libcudart.so",
                "/usr/local/cuda-12/lib64/libcudart.so",
                "/usr/lib/x86_64-linux-gnu/libcudart.so",
                /* Ollama-bundled paths */
                "/usr/local/lib/ollama/cuda_v13/libcudart.so.13.0.96",
                "/usr/local/lib/ollama/mlx_cuda_v13/libcudart.so",
                "/usr/local/lib/ollama/cuda_v12/libcudart.so.12",
                NULL
            };
            const char **p;
            for (p = cudart_paths; *p && !libcudart; p++)
                libcudart = dlopen(*p, RTLD_NOW | RTLD_GLOBAL);
        }

        if (libcudart) {
            if (gb_debug)
                fprintf(stderr, "[GreenBoost] libcudart loaded\n");
        } else {
            if (gb_debug)
                fprintf(stderr, "[GreenBoost] WARNING: libcudart not found — runtime API resolved lazily\n");
        }
    }

    /* Driver API (cu*) — always from libcuda.so.1 */
    real_cuMemAlloc_v2     = (pfn_cuMemAlloc_v2)     dlsym(libcuda, "cuMemAlloc_v2");
    real_cuMemFree_v2      = (pfn_cuMemFree_v2)      dlsym(libcuda, "cuMemFree_v2");
    real_cuMemAllocManaged = (pfn_cuMemAllocManaged)  dlsym(libcuda, "cuMemAllocManaged");
    real_cuMemAllocAsync   = (pfn_cuMemAllocAsync)    dlsym(libcuda, "cuMemAllocAsync");
    real_cuMemGetInfo      = (pfn_cuMemGetInfo)       dlsym(libcuda, "cuMemGetInfo_v2");
    if (!real_cuMemGetInfo)
        real_cuMemGetInfo  = (pfn_cuMemGetInfo)       dlsym(libcuda, "cuMemGetInfo");
    real_cuDeviceTotalMem_v2 = (pfn_cuDeviceTotalMem_v2) dlsym(libcuda, "cuDeviceTotalMem_v2");
    if (!real_cuDeviceTotalMem_v2)
        real_cuDeviceTotalMem_v2 = (pfn_cuDeviceTotalMem_v2) dlsym(libcuda, "cuDeviceTotalMem");
    real_cuMemHostRegister = (pfn_cuMemHostRegister) dlsym(libcuda, "cuMemHostRegister");
    real_cuMemHostUnregister = (pfn_cuMemHostUnregister) dlsym(libcuda, "cuMemHostUnregister");
    real_cuMemHostGetDevicePointer = (pfn_cuMemHostGetDevicePointer) dlsym(libcuda, "cuMemHostGetDevicePointer_v2");
    real_cuMemPrefetchAsync = (pfn_cuMemPrefetchAsync) dlsym(libcuda, "cuMemPrefetchAsync");
    real_cuMemAdvise = (pfn_cuMemAdvise) dlsym(libcuda, "cuMemAdvise");
    real_cuMemFreeAsync = (pfn_cuMemFreeAsync) dlsym(libcuda, "cuMemFreeAsync");
    real_cuDeviceGetAttribute = (pfn_cuDeviceGetAttribute) dlsym(libcuda, "cuDeviceGetAttribute");
    real_cuMemCreate = (pfn_cuMemCreate) dlsym(libcuda, "cuMemCreate");

    /* Probe compute capability — required to gate cuMemAllocAsync (cc >= 8.0).
     * CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75 (stable since CUDA 3). */
    if (real_cuDeviceGetAttribute) {
        int cc_major = 0;
        if (real_cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0) == CUDA_SUCCESS)
            gb_cc_major = cc_major;
    }

    /* NVML — loaded separately; Ollama uses this for GPU memory discovery.
     * Also used here to probe the real physical VRAM so the banner is accurate
     * and all subsequent calculations (headroom, KV reserve) adapt to the host GPU. */
    {
        typedef unsigned int (*pfn_nvmlInit)(void);
        typedef unsigned int (*pfn_nvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t *);

        void *libnvml = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!libnvml) libnvml = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_GLOBAL);
        if (libnvml) {
            real_nvmlDeviceGetMemoryInfo    = (pfn_nvmlDeviceGetMemoryInfo)
                dlsym(libnvml, "nvmlDeviceGetMemoryInfo");
            real_nvmlDeviceGetMemoryInfo_v2 = (pfn_nvmlDeviceGetMemoryInfo_v2)
                dlsym(libnvml, "nvmlDeviceGetMemoryInfo_v2");
            real_nvmlDeviceGetMemoryInfo_v3 = (pfn_nvmlDeviceGetMemoryInfo_v3)
                dlsym(libnvml, "nvmlDeviceGetMemoryInfo_v3");

            /* Probe physical VRAM — call real functions directly, before our hooks
             * are active, so we get the true hardware value. */
            pfn_nvmlInit nvml_init = (pfn_nvmlInit)dlsym(libnvml, "nvmlInit_v2");
            if (!nvml_init) nvml_init = (pfn_nvmlInit)dlsym(libnvml, "nvmlInit");
            pfn_nvmlDeviceGetHandleByIndex nvml_get_handle =
                (pfn_nvmlDeviceGetHandleByIndex)dlsym(libnvml, "nvmlDeviceGetHandleByIndex_v2");
            if (!nvml_get_handle)
                nvml_get_handle = (pfn_nvmlDeviceGetHandleByIndex)
                    dlsym(libnvml, "nvmlDeviceGetHandleByIndex");

            if (nvml_init && nvml_get_handle && real_nvmlDeviceGetMemoryInfo) {
                if (nvml_init() == 0) { /* NVML_SUCCESS */
                    nvmlDevice_t dev = NULL;
                    if (nvml_get_handle(0, &dev) == 0 && dev) {
                        nvmlMemory_t mem = {0};
                        if (real_nvmlDeviceGetMemoryInfo(dev, &mem) == 0 && mem.total > 0)
                            gb_physical_vram_bytes = (size_t)mem.total;
                    }
                }
            }
        }
    }

    /* Read virtual (T2 DDR) pool size from kernel module — same source as
     * load_nvml_for_gaming(), but runs in the main CUDA path too.
     * GREENBOOST_VIRTUAL_VRAM_MB env var (parsed earlier) takes priority. */
    if (gb_virtual_vram_bytes == 0) {
        int virt_gb = read_sysfs_int("/sys/module/greenboost/parameters/virtual_vram_gb");
        if (virt_gb > 0)
            gb_virtual_vram_bytes = (size_t)virt_gb * 1024ULL * 1024ULL * 1024ULL;
    }
    /* Record T2-only pool bytes so Path A/B can skip oversized allocs that
     * would OOM by faulting in more anonymous RAM than the T2 pool holds.
     * gb_t2_pool_bytes is set here before T3 is added to gb_virtual_vram_bytes. */
    gb_t2_pool_bytes = gb_virtual_vram_bytes;

    /* Fallback: kernel module absent (container, WSL2, no greenboost.ko loaded) and
     * no GREENBOOST_VIRTUAL_VRAM_MB env var — gb_t2_pool_bytes stays 0, which makes
     * every cap guard in gb_overflow_alloc() a no-op.  Compute a safe default from
     * MemTotal so Path A/B and Path C guards are always enforced. */
    if (gb_t2_pool_bytes == 0) {
        FILE *mf = fopen("/proc/meminfo", "r");
        if (mf) {
            char mline[128];
            while (fgets(mline, sizeof(mline), mf)) {
                unsigned long long kb = 0;
                if (sscanf(mline, "MemTotal: %llu kB", &kb) == 1 && kb > 0) {
                    gb_t2_pool_bytes      = (size_t)(kb * 1024ULL * 88ULL / 100ULL);
                    gb_virtual_vram_bytes = gb_t2_pool_bytes;
                    gb_log("T2 pool fallback from MemTotal: %zu MB (88%% of %llu MB)",
                           gb_t2_pool_bytes >> 20, kb / 1024ULL);
                    break;
                }
            }
            fclose(mf);
        }
    }

    /* Read safety_reserve_gb from kernel module — mirrors the check in gb_alloc_buf()
     * and gb_pin_user_buf().  Used by Path B / cuMemCreate VMM guards below. */
    {
        int res_gb = read_sysfs_int("/sys/module/greenboost/parameters/safety_reserve_gb");
        gb_safety_reserve_bytes = (res_gb > 0)
            ? (size_t)res_gb * 1024ULL * 1024ULL * 1024ULL
            : 4ULL  * 1024ULL * 1024ULL * 1024ULL; /* default: 4 GB */
        gb_log("safety_reserve: %zu MB (from %s)",
               gb_safety_reserve_bytes >> 20,
               res_gb > 0 ? "sysfs" : "default");
    }

    /* Add T3 (kernel module NVMe pool) to the reported virtual VRAM.
     * T3 pages are allocated by the GreenBoost kernel module from NVMe-backed
     * pages.  Reporting T2+T3 as virtual VRAM ensures Ollama's fit algorithm
     * places all layers on GPU (no CPU split) for models larger than T2 alone.
     * Path A0 (cudaImportExternalMemory) or Path A (DMA-BUF+HostReg) serve
     * T3 allocations; UVM (Path C) is only the last-resort fallback. */
    {
        int nvme_gb = read_sysfs_int("/sys/module/greenboost/parameters/nvme_pool_gb");
        if (nvme_gb > 0) {
            gb_virtual_vram_bytes += (size_t)nvme_gb * 1024ULL * 1024ULL * 1024ULL;
            gb_log("T3 NVMe pool: +%d GB added to virtual VRAM report", nvme_gb);
        }
    }

    /* Scale vram_headroom_bytes to 5% of physical VRAM (floor 256 MB, ceiling 1024 MB)
     * if the user did not override via GREENBOOST_VRAM_HEADROOM_MB.
     * This prevents the 512 MB flat default from being disproportionately large on
     * small GPUs (4 GB) or too conservative on large GPUs (48 GB+). */
    if (!headroom_from_env && gb_physical_vram_bytes > 0) {
        size_t pct5 = gb_physical_vram_bytes / 20;  /* 5% */
        size_t floor_bytes  = 256ULL * 1024 * 1024;
        size_t ceil_bytes   = 1024ULL * 1024 * 1024;
        if (pct5 < floor_bytes) pct5 = floor_bytes;
        if (pct5 > ceil_bytes)  pct5 = ceil_bytes;
        vram_headroom_bytes = pct5;
    }

    /* Runtime API (cuda*) — live in libcudart, not libcuda */
    if (libcudart) {
        real_cudaMalloc        = (pfn_cudaMalloc)        dlsym(libcudart, "cudaMalloc");
        real_cudaFree          = (pfn_cudaFree)           dlsym(libcudart, "cudaFree");
        real_cudaMallocManaged = (pfn_cudaMallocManaged)  dlsym(libcudart, "cudaMallocManaged");
        real_cudaMallocAsync   = (pfn_cudaMallocAsync)    dlsym(libcudart, "cudaMallocAsync");

        real_cudaImportExternalMemory        = (pfn_cudaImportExternalMemory)
            dlsym(libcudart, "cudaImportExternalMemory");
        real_cudaExternalMemoryGetMappedBuffer = (pfn_cudaExternalMemoryGetMappedBuffer)
            dlsym(libcudart, "cudaExternalMemoryGetMappedBuffer");
        real_cudaDestroyExternalMemory       = (pfn_cudaDestroyExternalMemory)
            dlsym(libcudart, "cudaDestroyExternalMemory");
        real_cudaGetLastError                = (pfn_cudaGetLastError)
            dlsym(libcudart, "cudaGetLastError");
        real_cudaMemGetInfo                  = (pfn_cudaMemGetInfo)
            dlsym(libcudart, "cudaMemGetInfo");
        real_cudaMemPrefetchAsync            = (pfn_cudaMemPrefetchAsync)
            dlsym(libcudart, "cudaMemPrefetchAsync");
    }
    /* Fallback: some CUDA versions export runtime wrappers from libcuda.so.1 */
    if (!real_cudaMalloc)        real_cudaMalloc        = (pfn_cudaMalloc)        dlsym(libcuda, "cudaMalloc");
    if (!real_cudaFree)          real_cudaFree          = (pfn_cudaFree)           dlsym(libcuda, "cudaFree");
    if (!real_cudaMallocManaged) real_cudaMallocManaged = (pfn_cudaMallocManaged)  dlsym(libcuda, "cudaMallocManaged");
    if (!real_cudaMallocAsync)   real_cudaMallocAsync   = (pfn_cudaMallocAsync)    dlsym(libcuda, "cudaMallocAsync");
    if (!real_cudaMemGetInfo)    real_cudaMemGetInfo    = (pfn_cudaMemGetInfo)     dlsym(libcuda, "cudaMemGetInfo");
    if (!real_cudaMemPrefetchAsync) real_cudaMemPrefetchAsync = (pfn_cudaMemPrefetchAsync) dlsym(libcuda, "cudaMemPrefetchAsync");

    if (!real_cuMemAlloc_v2 || !real_cuMemFree_v2) {
        fprintf(stderr, "[GreenBoost] WARNING: failed to resolve core CUDA symbols\n");
        return;
    }

    initialized = 1;

    /* CRIT-06: Start prefetch thread only after initialization is confirmed
     * successful.  Previously started before the symbol-resolution guard,
     * leaving a dangling thread if that guard triggered an early return. */
    pthread_create(&prefetch_thread, NULL, prefetch_worker, NULL);
    prefetch_initialized = 1;

    /* KV reserve: fixed 25% of physical VRAM.
     * This guarantees weights use 70–75% of T1 during model loading and
     * KV cache always lands in fast T1 VRAM during inference.
     * The adaptive reserve in gb_needs_overflow() collapses this to 0 once
     * KV is allocated in T1, allowing T1 to reach ~95% utilization.
     * Override: GREENBOOST_KV_RESERVE_MB env var or kernel module kv_reserve_mb. */
    if (atomic_load_explicit(&g_kv_reserve_bytes, memory_order_relaxed) == 0) {
        size_t auto_kv_mb;
        if (gb_physical_vram_bytes > 0) {
            auto_kv_mb = (gb_physical_vram_bytes >> 20) * 25 / 100;
        } else {
            /* Conservative fallback until cuMemGetInfo populates gb_physical_vram_bytes
             * (happens on first CUDA context call via gb_refresh_meminfo_cache). */
            auto_kv_mb = 3072;
        }
        atomic_store_explicit(&g_kv_reserve_bytes,
                              auto_kv_mb * 1024ULL * 1024ULL,
                              memory_order_relaxed);
        g_kv_reserve_from_env = 1;
        gb_log("KV reserve: %zu MB (25%% of %zu MB VRAM)",
               auto_kv_mb, gb_physical_vram_bytes >> 20);
        /* Also try reading the kernel module value (may override auto) */
        gb_refresh_kv_reserve();
    }

    /* Read NVLink and ComputeDomain sysfs attributes for cluster-aware memory tiering */
    gb_compute_domain_active = read_sysfs_bool("/sys/class/greenboost/greenboost/compute_domain_active");
    {
        int nvlink_val = read_sysfs_int("/sys/class/greenboost/greenboost/nvlink_ready");
        if (nvlink_val == 1 && gb_physical_vram_bytes > 0) {
            /* Read gpu_count_per_node from sysfs — written by greenboost_setup.sh at insmod
             * time so that multi-GPU nodes with != 8 GPUs aggregate correctly.
             * Fall back to the historical /8 * 7 formula when the attribute is absent
             * (older module builds without the sysfs attribute). */
            int gpu_count = read_sysfs_int("/sys/class/greenboost/greenboost/gpu_count_per_node");
            if (gpu_count > 1) {
                gb_nvlink_aggregated_bytes =
                    (gb_physical_vram_bytes / (size_t)gpu_count) * (size_t)(gpu_count - 1);
            } else {
                /* Fallback: assume 8-GPU node (original V100 cluster topology) */
                gb_nvlink_aggregated_bytes = (gb_physical_vram_bytes >> 3) * 7;
            }
            gb_log("NVLink pooling active: gpu_count=%d aggregated VRAM +%zu MB",
                   gpu_count > 1 ? gpu_count : 8,
                   gb_nvlink_aggregated_bytes >> 20);
        }
    }

    /* Write initial stats file so status panel shows "none" rather than
     * falling back to sysfs heuristics before any overflow allocs occur. */
    gb_write_stats();

    /* Startup banner — gated on debug mode.
     * With /etc/ld.so.preload the shim loads into every process on the system.
     * libcuda.so.1 is in ldconfig (NVIDIA driver installs it), so the shim
     * activates for ls, bash, systemd, etc.  Silent by default. */
    if (gb_debug) {
        fprintf(stderr, "[GreenBoost] v2.7 loaded — vram_headroom=%zuMB kv_reserve=%zuMB(adaptive) kv_threshold=%zuMB virtual_vram=%zuMB use_dmabuf=%d debug=%d\n",
                vram_headroom_bytes >> 20, g_kv_reserve_bytes >> 20,
                g_kv_size_threshold_bytes >> 20,
                gb_virtual_vram_bytes >> 20, gb_use_dmabuf, gb_debug);
        fprintf(stderr, "[GreenBoost] Alloc paths counters: A0=%u A=%u B=%u C=%u\n",
                gb_path_a0_count, gb_path_a_count, gb_path_b_count, gb_path_c_count);
        fprintf(stderr, "[GreenBoost] Path A0 (cudaImportExtMem): %s\n",
                gb_a0_disabled
                    ? "DISABLED (runtime: cudaImportExternalMemory error 999 — not supported on this GPU/driver)"
                : (gb_use_dmabuf && real_cudaImportExternalMemory && real_cudaExternalMemoryGetMappedBuffer)
                    ? "ACTIVE — zero-copy DMA-BUF via cudaImportExternalMemory (best)"
                : gb_use_dmabuf && !real_cudaImportExternalMemory
                    ? "unavailable (libcudart not resolved — falling back to Path A)"
                : "disabled (GREENBOOST_USE_DMA_BUF=0)");
        fprintf(stderr, "[GreenBoost] Path A  (DMA-BUF+HostReg): %s\n",
                (real_cuMemHostRegister && gb_use_dmabuf)
                    ? "available — mmap+GB_IOCTL_PIN_USER_PTR+cuMemHostRegister"
                : gb_use_dmabuf ? "wanted but cuMemHostRegister missing"
                : "disabled (GREENBOOST_USE_DMA_BUF=0)");
        fprintf(stderr, "[GreenBoost] Path B  (HostReg/no-kmod): %s\n",
                (!gb_no_hostreg && real_cuMemHostRegister && real_cuMemHostGetDevicePointer)
                    ? "available — mmap+cuMemHostRegister (containers/VMs)"
                : gb_no_hostreg ? "disabled (GREENBOOST_NO_HOSTREG=1)"
                : "unavailable (cuMemHostRegister not resolved)");
        fprintf(stderr, "[GreenBoost] Path C  (UVM/managed)    : %s\n",
                real_cuMemAllocManaged ? "available — cuMemAllocManaged+cuMemAdvise (last resort)"
                                       : "UNAVAILABLE — load nvidia_uvm.ko");
        fprintf(stderr, "[GreenBoost] Async alloc hooks : cuMemAllocAsync=%s cudaMallocAsync=%s\n",
                real_cuMemAllocAsync ? "hooked" : "missing",
                real_cudaMallocAsync ? "hooked" : "missing");
        fprintf(stderr, "[GreenBoost] cuMemGetInfo hook : %s (reports +%zu MB virtual VRAM to CUDA)\n",
                real_cuMemGetInfo ? "active" : "missing",
                gb_virtual_vram_bytes >> 20);
        fprintf(stderr, "[GreenBoost] cuDeviceTotalMem  : %s\n",
                real_cuDeviceTotalMem_v2 ? "hooked" : "missing");
        fprintf(stderr, "[GreenBoost] nvmlMemInfo hook  : %s\n",
                real_nvmlDeviceGetMemoryInfo ? "hooked" : "missing (NVML not found)");
        fprintf(stderr, "[GreenBoost] dlsym hook        : active (intercepts dlopen+dlsym GPU API calls)\n");
        if (gb_physical_vram_bytes)
            fprintf(stderr, "[GreenBoost] Combined VRAM     : %zu GB physical + %zu GB system RAM via GreenBoost\n",
                    gb_physical_vram_bytes >> 30, gb_virtual_vram_bytes >> 30);
        else
            fprintf(stderr, "[GreenBoost] Combined VRAM     : ? GB physical + %zu GB system RAM via GreenBoost\n",
                    gb_virtual_vram_bytes >> 30);
    }
}

/* ------------------------------------------------------------------ */
/*  Destructor                                                          */
/* ------------------------------------------------------------------ */

/*
 * gb_htable_flush - release CUDA resources for all live hash-table entries.
 *
 * @kv_only: if 0, flush every live entry; if 1, skip GB_ALLOC_KV_CACHE entries
 *           (used by idle-flush to keep the loaded model's KV cache warm).
 *
 * Called single-threaded after the prefetch thread has been joined, so no
 * concurrent writers exist.  Each slot's per-lock is still taken for correct
 * memory ordering.
 */
static void gb_htable_flush(int kv_only)
{
    uint32_t i;

    for (i = 0; i < HT_SIZE; i++) {
        gb_ht_entry_t *e = &gb_htable[i];
        pthread_mutex_t *lk = ht_lock(i);

        pthread_mutex_lock(lk);
        if (e->ptr == 0 || e->ptr == HT_TOMBSTONE) {
            pthread_mutex_unlock(lk);
            continue;
        }
        /* Skip KV cache entries when doing an idle (non-destructive) flush */
        if (kv_only && (e->alloc_flags & GB_ALLOC_KV_CACHE)) {
            pthread_mutex_unlock(lk);
            continue;
        }

        /* Release CUDA's hold on the DMA-BUF before closing the fd */
        if (e->ext_mem && real_cudaDestroyExternalMemory)
            real_cudaDestroyExternalMemory(e->ext_mem);

        /* Path B: unregister host memory and unmap */
        if (e->mapped_ptr) {
            if (real_cuMemHostUnregister)
                real_cuMemHostUnregister(e->mapped_ptr);
            munmap(e->mapped_ptr, e->size);
        }

        if (e->fd >= 0)
            close(e->fd);

        /* Tombstone so ht_remove from concurrent callers (none expected here,
         * but be safe) cannot double-free. */
        e->ptr        = HT_TOMBSTONE;
        e->fd         = -1;
        e->mapped_ptr = NULL;
        e->ext_mem    = NULL;
        pthread_mutex_unlock(lk);
    }
}

__attribute__((destructor))
static void gb_shim_fini(void)
{
    if (initialized)
        gb_write_stats();  /* final snapshot before unload */
    /* CRIT-06: Join on prefetch_initialized (not initialized) so the thread is
     * always reaped regardless of whether symbol resolution succeeded. */
    if (prefetch_initialized) {
        prefetch_stop = 1;
        pthread_cond_signal(&prefetch_cond);
        pthread_join(prefetch_thread, NULL);
    }

    /* Flush all hash-table entries: release cudaExternalMemory handles and
     * unmap Path-B regions so their DMA-BUF refcounts drop to zero.  This
     * must happen BEFORE closing gb_dev_fd so that gb_close on the kernel
     * side sees an already-clean IDR (belt-and-suspenders).
     * Handles SIGTERM (orderly shutdown); SIGKILL is handled by the kernel
     * gb_close() sweep which fires when the fd is force-closed on process death. */
    if (initialized)
        gb_htable_flush(0);  /* 0 = flush all entries */

    /* Belt-and-suspenders: ask the kernel to drop any remaining refs for our PID */
    if (gb_dev_fd >= 0) {
        struct gb_release_pid_req rreq = { .pid = 0, ._pad = 0 };
        ioctl(gb_dev_fd, GB_IOCTL_RELEASE_PID, &rreq);
        close(gb_dev_fd);
        gb_dev_fd = -1;
    }
    /* Remove phase file so the daemon knows the CUDA process has exited. */
    unlink("/run/greenboost/phase");
    gb_log("shim unloaded");
}

/* ------------------------------------------------------------------ */
/*  VRAM-aware overflow decision                                        */
/* ------------------------------------------------------------------ */

/* ENH-03: Refresh the cuMemGetInfo cache if stale.
 * Called on every GB_MEMINFO_REFRESH_ALLOCS boundary or when the cached
 * value is first populated (both ms==0 and alloc-count gate can trigger).
 * The values are stored atomically so concurrent readers always see a
 * consistent (if slightly stale) snapshot — acceptable for a heuristic. */
static void gb_refresh_meminfo_cache(void)
{
    size_t free_vram = 0, total_vram = 0;
    struct timespec ts;
    uint64_t now_ms;

    if (!real_cuMemGetInfo)
        return;
    if (real_cuMemGetInfo(&free_vram, &total_vram) != CUDA_SUCCESS)
        return;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    now_ms = (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;

    atomic_store_explicit(&g_cached_free_vram,  free_vram,  memory_order_relaxed);
    atomic_store_explicit(&g_cached_total_vram, total_vram, memory_order_relaxed);
    atomic_store_explicit(&g_cached_meminfo_ms, now_ms,     memory_order_relaxed);

    /* If NVML probe failed at init, use cuMemGetInfo total as fallback so that
     * the 25% KV reserve auto-scaler and headroom scaling are not stuck on
     * conservative fallback values for the entire session. */
    if (gb_physical_vram_bytes == 0 && total_vram > 0)
        gb_physical_vram_bytes = total_vram;
}

static int gb_needs_overflow(size_t bytesize)
{
    size_t free_vram = 0, total_vram = 0;
    size_t kv_reserve;

    /* Periodically refresh KV reserve from the kernel module so that
     * Synapse CLI's GB_IOCTL_SET_KV_RESERVE takes effect without an Ollama
     * restart.  Power-of-2 interval → cheap bitmask instead of modulo.
     * CRIT-04: fetch_add returns old value; add 1 to get the new count. */
    {
        unsigned int cnt = atomic_fetch_add_explicit(&g_alloc_count, 1u,
                                                     memory_order_relaxed) + 1u;
        if ((cnt & (GB_KV_REFRESH_INTERVAL - 1u)) == 0)
            gb_refresh_kv_reserve();

        /* ENH-03: Refresh cuMemGetInfo cache every GB_MEMINFO_REFRESH_ALLOCS allocs
         * or every GB_MEMINFO_REFRESH_MS ms — avoids a CUDA round-trip per alloc. */
        if ((cnt & (GB_MEMINFO_REFRESH_ALLOCS - 1u)) == 0) {
            gb_refresh_meminfo_cache();
        } else {
            /* Time-based refresh: check elapsed ms against last cache write */
            uint64_t last_ms = atomic_load_explicit(&g_cached_meminfo_ms,
                                                    memory_order_relaxed);
            if (last_ms == 0) {
                gb_refresh_meminfo_cache();  /* first call — populate cache */
            } else {
                struct timespec ts_now;
                clock_gettime(CLOCK_MONOTONIC, &ts_now);
                uint64_t now_ms = (uint64_t)ts_now.tv_sec * 1000ULL
                                + (uint64_t)ts_now.tv_nsec / 1000000ULL;
                if (now_ms - last_ms >= GB_MEMINFO_REFRESH_MS)
                    gb_refresh_meminfo_cache();
            }
        }
    }

    if (!real_cuMemGetInfo)
        return 0;

    /* Use the cached values populated by gb_refresh_meminfo_cache() above */
    free_vram  = atomic_load_explicit(&g_cached_free_vram,  memory_order_relaxed);
    total_vram = atomic_load_explicit(&g_cached_total_vram, memory_order_relaxed);
    if (total_vram == 0) {
        /* Cache not populated (cuMemGetInfo failed — context not yet warm).
         * For allocs clearly exceeding physical VRAM, overflow is always needed. */
        if (gb_physical_vram_bytes > 0 && bytesize > gb_physical_vram_bytes)
            return 1;
        return 0;
    }

    /* Adaptive KV reserve: subtract already-allocated KV T1 bytes from the
     * nominal reserve.  cuMemGetInfo already reflects KV that landed in T1,
     * so applying the full reserve on top double-counts and leaves VRAM idle.
     * As KV fills T1, effective_reserve → 0 and all remaining VRAM is usable. */
    {
        size_t kv_reserved = atomic_load_explicit(&g_kv_reserve_bytes,     memory_order_relaxed);
        size_t kv_in_t1    = atomic_load_explicit(&g_kv_allocated_t1_bytes, memory_order_relaxed);
        kv_reserve = (kv_in_t1 >= kv_reserved) ? 0 : (kv_reserved - kv_in_t1);
    }

    /* Phase-aware KV reserve during MODEL_LOAD:
     *
     * If the model fits entirely in T1 (no T2 overflow yet): zero the reserve so
     * weights can fill all of T1.  Both weights and KV will land in T1 anyway.
     *
     * If the model overflows T1 (t2_used > 0): keep the 25% KV reserve active so
     * weights occupy T1 up to ~70% and the remaining 25% stays available for KV.
     * KV lands in T1 during INFERENCE, keeping GDDR7 bandwidth fully utilized.
     *
     * Once the phase advances to INFERENCE/STEADY, the adaptive reserve takes over:
     * as KV fills T1 the effective reserve collapses to 0, allowing T1 to reach ~95%. */
    int cur_phase = atomic_load_explicit(&g_alloc_phase, memory_order_relaxed);
    if (cur_phase <= GB_PHASE_MODEL_LOAD) {
        size_t t2_used = atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed);
        if (t2_used == 0) {
            kv_reserve = 0;  /* model may fit entirely in T1 — let weights use all of it */
        }
        /* When t2_used > 0: keep kv_reserve = 25% of VRAM (set by auto-scaler at init).
         * No additional cap needed — 25% is already the correct value. */
    }

    /* Inference-time occupancy collapse: if phase is INFERENCE/STEADY but
     * g_kv_allocated_t1_bytes is still 0 (KV was allocated by the CUDA runtime
     * directly, bypassing the shim's overflow path), infer KV location from
     * actual free VRAM.  If free_vram < headroom + kv_reserve + 128 MB slack,
     * VRAM is more occupied than the reserve allows — KV must already be in T1.
     * Collapse the reserve so T1 stays ≥ 92% utilized during inference. */
    if (kv_reserve > 0 && cur_phase >= GB_PHASE_INFERENCE) {
        size_t kv_in_t1_cur = atomic_load_explicit(&g_kv_allocated_t1_bytes, memory_order_relaxed);
        if (kv_in_t1_cur == 0) {
            size_t expected_free = vram_headroom_bytes + kv_reserve;
            if (free_vram < expected_free + (128ULL << 20)) {
                size_t inferred_kv = (expected_free > free_vram)
                    ? expected_free - free_vram : 0;
                atomic_store_explicit(&g_kv_allocated_t1_bytes, inferred_kv, memory_order_relaxed);
                kv_reserve = (inferred_kv >= kv_reserve) ? 0 : kv_reserve - inferred_kv;
                gb_log("KV reserve auto-collapsed: inferred %zu MB KV in T1 "
                       "(free=%zuMB expected_free=%zuMB)",
                       inferred_kv >> 20, free_vram >> 20, expected_free >> 20);
            }
        }
    }

    /* KV cache reservation: weights spill to T2 while kv_reserve > 0, leaving
     * T1 headroom for the KV cache.  KV is read+written every generation step —
     * T1 (~336 GB/s) vs T2 (~32 GB/s PCIe) is ~10x throughput difference.
     * Once KV is allocated in T1, effective_reserve collapses and VRAM is fully
     * used.  Set GREENBOOST_KV_RESERVE_MB or module kv_reserve_mb param to tune.
     *
     * KV alloc bypass: when the incoming allocation IS the KV cache (INFERENCE phase,
     * large alloc, KV not yet in T1), collapse kv_reserve to 0 for this check.
     * The reserve was held open FOR KV; applying it against KV itself is circular
     * and would permanently block KV from landing in T1. */
    if (kv_reserve > 0 && cur_phase >= GB_PHASE_INFERENCE) {
        size_t kv_in_t1_cur = atomic_load_explicit(&g_kv_allocated_t1_bytes, memory_order_relaxed);
        if (kv_in_t1_cur == 0 && bytesize >= g_kv_size_threshold_bytes) {
            gb_log("KV alloc bypass: collapsing kv_reserve %zu MB → 0 for KV alloc %zu MB (T1 space available)",
                   kv_reserve >> 20, bytesize >> 20);
            kv_reserve = 0;
        }
    }
    {
        if (bytesize + vram_headroom_bytes + kv_reserve > free_vram) {
            gb_log("VRAM: req=%zuMB free=%zuMB headroom=%zuMB kv_reserve=%zuMB(eff) phase=%d oversize=%d → OVERFLOW",
                   bytesize >> 20, free_vram >> 20, vram_headroom_bytes >> 20,
                   kv_reserve >> 20, cur_phase,
                   (gb_physical_vram_bytes > 0 && bytesize > gb_physical_vram_bytes));
            return 1;
        }
        gb_log("VRAM: req=%zuMB free=%zuMB phase=%d → fits in T1",
               bytesize >> 20, free_vram >> 20, cur_phase);
    }
    return 0;
}

/* Track a large allocation that landed in T1 VRAM (not overflow) as KV cache.
 * Heuristic: a large alloc (>= g_kv_size_threshold_bytes) that arrives after a
 * GB_PHASE_QUIET_GAP_MS quiet period (no overflow allocs) is almost certainly
 * the KV cache being allocated after the model weights have been pushed to T2.
 * We increment g_kv_allocated_t1_bytes so gb_needs_overflow() reduces its
 * effective_reserve accordingly — preventing double-counting and freeing VRAM. */

/* REF-02: Extracted from cuMemFree_v2 and cudaFree (identical CAS loop).
 * Subtracts sz from g_kv_allocated_t1_bytes when a T1 KV cache alloc is freed.
 * Only called for allocs that were not overflow (mapped_ptr/ext_mem/managed == 0). */
static void gb_release_kv_t1_bytes(uint32_t flags, size_t sz,
                                   void *mapped_ptr, cudaExternalMemory_t ext_mem,
                                   int managed)
{
    if ((flags & GB_ALLOC_KV_CACHE) && !mapped_ptr && !ext_mem && !managed) {
        size_t cur, newval;
        do {
            cur    = atomic_load_explicit(&g_kv_allocated_t1_bytes, memory_order_relaxed);
            newval = (sz < cur) ? (cur - sz) : 0;
        } while (!atomic_compare_exchange_weak_explicit(
                     &g_kv_allocated_t1_bytes, &cur, newval,
                     memory_order_relaxed, memory_order_relaxed));
    }
}

static void gb_maybe_track_kv_t1(CUdeviceptr dptr, size_t bytesize)
{
    uint64_t now_ms, gap_ms;

    /* Never tag T1 allocs as KV during INIT or MODEL_LOAD — those are weight
     * tensors.  Tagging them inflates g_kv_allocated_t1_bytes and collapses
     * the effective KV reserve to zero before inference even starts, causing
     * real KV cache to spill to T2 at ~32 GB/s instead of staying in T1. */
    {
        int phase = atomic_load_explicit(&g_alloc_phase, memory_order_relaxed);
        if (phase <= (int)GB_PHASE_MODEL_LOAD)
            return;
    }

    if (!g_phase_detect || bytesize < g_kv_size_threshold_bytes)
        return;
    /* In KV overflow mode all large allocs overflow as KV; T1 allocs are weights */
    if (g_kv_overflow_mode)
        return;

    now_ms  = gb_now_ms();
    {
        uint64_t last = atomic_load_explicit(&g_last_overflow_ms, memory_order_relaxed);
        gap_ms = last ? (now_ms - last) : (uint64_t)-1;
    }

    if (gap_ms >= (uint64_t)GB_PHASE_QUIET_GAP_MS) {
        size_t kv_rsv, kv_in;
        atomic_fetch_add_explicit(&g_kv_allocated_t1_bytes, bytesize, memory_order_relaxed);
        ht_set_flags(dptr, GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY);
        kv_rsv = atomic_load_explicit(&g_kv_reserve_bytes,      memory_order_relaxed);
        kv_in  = atomic_load_explicit(&g_kv_allocated_t1_bytes, memory_order_relaxed);
        gb_log("KV T1 track: %zu MB at 0x%llx after %llu ms quiet gap"
               " — adaptive reserve now %zu MB",
               bytesize >> 20, (unsigned long long)dptr,
               (unsigned long long)(gap_ms == (uint64_t)-1 ? 0 : gap_ms),
               (kv_in >= kv_rsv ? 0 : (kv_rsv - kv_in)) >> 20);
    }
}

/* Overflow allocation — routes T1 misses to T2 (DDR) or T3 (NVMe) depending on size.
 *
 * Path selection order:
 *   1. Size > T2 cap (gb_t2_pool_bytes): skip A0/A/B entirely → UVM (Path C).
 *      Pinning > T2 cap bytes of anonymous RAM would OOM the system; UVM demand-pages
 *      and lets the CUDA driver migrate hot pages to T1 VRAM automatically.
 *   2. Path A0 (cudaImportExternalMemory): preferred on bare metal when greenboost.ko
 *      is loaded. True zero-copy DMA-BUF import; no mmap round-trip.
 *   3. Path A  (DMA-BUF + cuMemHostRegister): bare metal fallback when A0 unavailable.
 *   4. Path B  (mmap + cuMemHostRegister, no greenboost.ko): for containers, VMs, WSL2,
 *      and HPC clusters where /dev/greenboost is absent. Auto-enabled when the device
 *      node is unavailable; skip with GREENBOOST_NO_HOSTREG=1.
 *   5. Path C  (cuMemAllocManaged / UVM): last resort; hot pages auto-migrate to T1 VRAM,
 *      cold pages demand-paged from system RAM (T2-equivalent via Linux VMM).
 *
 * KV cache vs weights placement is determined by the phase detector and by explicit
 * GB_ALLOC_KV_CACHE / GB_ALLOC_T1_PRIORITY flags set via Synapse CLI or env vars.
 * KV cache has T1 priority because it is read+written on every generated token; weights
 * tolerate T2 latency (sequential read, prefetch hides most PCIe overhead). */
static CUresult gb_overflow_alloc(CUdeviceptr *dptr, size_t bytesize)
{
    void *mapped_ptr = NULL;
    int fd = -1;
    CUresult ret;

    /* Choose alloc flags for this overflow allocation.
     *
     * Priority order (highest wins):
     *   1. gb_compute_domain_active  — V100 cluster ComputeDomain: all overflow = KV.
     *   2. g_kv_overflow_mode        — GREENBOOST_KV_OVERFLOW=1: all overflow = KV.
     *   3. gb_phase_classify()       — temporal phase detector: weights → KV → activations.
     *
     * The phase detector covers Ollama/llama.cpp where weights load first, then KV.
     * For ExLlamaV3, use GREENBOOST_KV_OVERFLOW=1 (native MADVISE handles the rest). */
    uint32_t alloc_flags;
    if (gb_compute_domain_active || g_kv_overflow_mode) {
        alloc_flags = GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY;
        gb_log("KV override active (%s): flagging alloc as GB_ALLOC_KV_CACHE|GB_ALLOC_T1_PRIORITY",
               gb_compute_domain_active ? "ComputeDomain" : "GREENBOOST_KV_OVERFLOW");
    } else {
        alloc_flags = gb_phase_classify(bytesize);
    }

    /* ---- TurboQuant: allocate compressed-size T2 buffer for KV cache ----
     * When TurboQuant is enabled and this allocation is classified as KV cache,
     * replace bytesize with the compressed size so T2 holds the compressed form.
     * The shim's cudaMemcpy/cudaMemcpyAsync hooks then transparently
     * compress on write and decompress on read.
     *
     * We read g_tq_config under a brief mutex snapshot — no blocking work. */
    size_t tq_compressed_sz = 0;  /* 0 = not using TQ for this alloc */
    {
        gb_tq_config_t tq_snap;
        pthread_mutex_lock(&g_tq_config_mutex);
        tq_snap = g_tq_config;
        pthread_mutex_unlock(&g_tq_config_mutex);

        if (tq_snap.enabled &&
            (alloc_flags & GB_ALLOC_KV_CACHE) &&
            g_tq_compressed_size_fn &&
            tq_snap.head_dim == 128)
        {
            /* n_elements = bytesize / 2 (fp16 = 2 bytes per element) */
            size_t n_elements = bytesize / 2;
            if (n_elements > 0 && n_elements % (size_t)tq_snap.head_dim == 0) {
                size_t csz = g_tq_compressed_size_fn(n_elements,
                                                      tq_snap.head_dim,
                                                      tq_snap.bits);
                if (csz > 0 && csz < bytesize) {
                    tq_compressed_sz = csz;
                    alloc_flags |= GB_ALLOC_KV_COMPRESSED;
                    gb_log("TurboQuant turbo%d: KV alloc %zu MB → %zu MB (%.1f×)",
                           tq_snap.bits,
                           bytesize >> 20,
                           csz >> 20,
                           (double)bytesize / (double)csz);
                }
            }
        }
    }
    /* Use compressed size for the actual memory allocation if TQ is active */
    size_t alloc_bytesize = tq_compressed_sz ? tq_compressed_sz : bytesize;

    /* ---- Oversize routing: allocs > physical VRAM → Path C (UVM) --------- */
    /* Paths A0/A/B pin the entire allocation in system RAM with no VRAM page  */
    /* migration — T1 stays idle while everything runs at PCIe bandwidth.      */
    /* For allocs larger than physical VRAM, UVM is strictly better: the CUDA  */
    /* driver auto-migrates hot pages to T1 (~336 GB/s) and evicts cold pages  */
    /* to system RAM (~32 GB/s), filling T1 to 90%+.                           */
    /* cuMemPrefetchAsync warms the VRAM cache immediately after alloc.        */
    if (gb_physical_vram_bytes > 0 && alloc_bytesize > gb_physical_vram_bytes) {
        gb_log("oversize alloc %zu MB > physical VRAM %zu MB → routing to UVM (Path C)",
               bytesize >> 20, gb_physical_vram_bytes >> 20);
        goto path_c_uvm;
    }

    /* Skip Path A and Path B for allocations larger than the effective T2 DDR pool cap.
     * Both paths pin or fault in the full allocation as anonymous RAM; attempting
     * to pin more RAM than the T2 pool allows would OOM the system.
     * UVM (Path C) demand-pages from system RAM and auto-migrates hot pages to T1,
     * making it the only safe path for multi-tier (T2+T3) spanning allocations.
     * During INFERENCE/STEADY phases, effective_cap = 88% of pool (GB_T2_INFERENCE_CAP_PCT). */
    if (gb_t2_pool_bytes > 0) {
        size_t effective_cap = gb_effective_t2_cap();
        if (alloc_bytesize > effective_cap) {
            gb_log("alloc %zu MB > T2 eff_cap %zu MB (pool=%zu MB) — skipping Path A0/A/B, routing to UVM",
                   alloc_bytesize >> 20, effective_cap >> 20, gb_t2_pool_bytes >> 20);
            goto path_c_uvm;
        }
    }

    /* ---- Path A0: cudaImportExternalMemory (true zero-copy DMA-BUF) -- */
    /* Requires: libcudart.so resolved + greenboost.ko + /dev/greenboost.  */
    /* CUDA imports the kernel hugepage-backed DMA-BUF fd directly,        */
    /* driving its own IOMMU mapping from the kernel SG table. No mmap     */
    /* round-trip or cuMemHostRegister overhead.                            */
    /* Only used for allocations >= 4 MB (2 hugepages); smaller allocs     */
    /* use Path B which has lower per-call overhead for tiny tensors.       */
    if (gb_use_dmabuf && !gb_a0_disabled &&
        real_cudaImportExternalMemory && real_cudaExternalMemoryGetMappedBuffer &&
        alloc_bytesize >= GB_PATH_A0_MIN_BYTES) {
        cudaExternalMemory_t ext_mem = NULL;
        ret = gb_alloc_via_external_mem(dptr, alloc_bytesize, &ext_mem, alloc_flags);
        if (ret == CUDA_SUCCESS) {
            /* fd=-1: req.fd was consumed by CUDA, not held by us */
            ht_insert(*dptr, bytesize, 0 /* not UVM */, -1, NULL, -1, ext_mem);
            ht_set_flags(*dptr, alloc_flags);
            if (tq_compressed_sz)
                ht_set_tq_compressed_size(*dptr, tq_compressed_sz);
            __sync_fetch_and_add(&gb_path_a0_count, 1);
            gb_maybe_write_stats();
            return CUDA_SUCCESS;
        }
        gb_log("Path A0 (cudaImportExternalMemory) failed for %zu MB — falling through to Path A",
               alloc_bytesize >> 20);
    }

    /* ---- Path A: DMA-BUF + cuMemHostRegister (kernel pin) ------------ */
    if (gb_use_dmabuf && real_cuMemHostRegister) {
        /* T2 capacity guard — same cap check as Path B (prevents over-pinning).
         * Uses gb_effective_t2_cap(): 88% of pool during INFERENCE/STEADY to protect OS headroom. */
        if (gb_t2_pool_bytes > 0) {
            size_t effective_cap = gb_effective_t2_cap();
            size_t t2_used = atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed);
            if (t2_used >= effective_cap || alloc_bytesize > effective_cap - t2_used) {
                gb_log("Path A skip: T2 cap reached (%zu/%zu MB, eff_cap=%zu MB) for %zu MB — routing to Path B/C",
                       t2_used >> 20, gb_t2_pool_bytes >> 20, effective_cap >> 20, alloc_bytesize >> 20);
                goto path_b_hostreg;
            }
        }
        /* MemAvailable guard — Path A pins real RAM just like Path B. */
        {
            size_t mem_avail = gb_get_mem_available();
            if (mem_avail > 0 &&
                (alloc_bytesize > mem_avail || mem_avail - alloc_bytesize < gb_safety_reserve_bytes)) {
                gb_log("Path A skip: MemAvailable %zu MB too low for %zu MB alloc — routing to Path B/C",
                       mem_avail >> 20, alloc_bytesize >> 20);
                goto path_b_hostreg;
            }
        }
        /* Allocate anonymous memory using mmap, then ask greenboost.ko to pin it */
        mapped_ptr = mmap(NULL, alloc_bytesize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped_ptr == MAP_FAILED) {
            fprintf(stderr, "[GreenBoost] mmap anonymous failed for %zu MB: %m\n", alloc_bytesize >> 20);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }

        fd = gb_open_device();
        if (fd >= 0) {
            struct gb_pin_req req;
            memset(&req, 0, sizeof(req));
            req.vaddr = (uint64_t)(uintptr_t)mapped_ptr;
            req.size = alloc_bytesize;
            req.flags = alloc_flags;  /* forward phase-classified flags */
            req.fd = -1;

            if (ioctl(fd, GB_IOCTL_PIN_USER_PTR, &req) < 0) {
                fprintf(stderr, "[GreenBoost] GB_IOCTL_PIN_USER_PTR failed for %zu MB: %m\n", alloc_bytesize >> 20);
                munmap(mapped_ptr, alloc_bytesize);
                mapped_ptr = NULL;
            } else {
                int dmabuf_fd = req.fd;
                /* Register the userspace pointer with CUDA */
                ret = real_cuMemHostRegister(mapped_ptr, alloc_bytesize, CU_MEMHOSTREGISTER_DEVICEMAP);
                if (ret != CUDA_SUCCESS) {
                    fprintf(stderr, "[GreenBoost] cuMemHostRegister FAILED ret=%d for %zu MB"
                            " — falling through to Path B/C\n", ret, alloc_bytesize >> 20);
                    munmap(mapped_ptr, alloc_bytesize);
                    close(dmabuf_fd);
                    mapped_ptr = NULL;
                    /* Fall through to Path B / Path C — do NOT hard-return here. */
                } else {
                    /* Get the device pointer for the registered memory */
                    ret = real_cuMemHostGetDevicePointer(dptr, mapped_ptr, 0);
                    if (ret != CUDA_SUCCESS) {
                        fprintf(stderr, "[GreenBoost] cuMemHostGetDevicePointer FAILED ret=%d\n", ret);
                        if (real_cuMemHostUnregister) real_cuMemHostUnregister(mapped_ptr);
                        munmap(mapped_ptr, alloc_bytesize);
                        close(dmabuf_fd);
                        return CUDA_ERROR_OUT_OF_MEMORY;
                    }

                    /* ht_insert size = bytesize (uncompressed logical size) */
                    ht_insert(*dptr, bytesize, 0 /* DMA-BUF */, -1, mapped_ptr, dmabuf_fd, NULL);
                    ht_set_flags(*dptr, alloc_flags);
                    if (tq_compressed_sz)
                        ht_set_tq_compressed_size(*dptr, tq_compressed_sz);
                    atomic_fetch_add_explicit(&gb_t2_overflow_bytes, alloc_bytesize, memory_order_relaxed);
                    gb_log("Path A (DMA-BUF pinned): %zu MB (alloc %zu MB TQ) at cuda_ptr=0x%llx (mapped=%p, fd=%d) t2_total=%zu MB",
                           bytesize >> 20, alloc_bytesize >> 20, (unsigned long long)*dptr, mapped_ptr, dmabuf_fd,
                           atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed) >> 20);
                    __sync_fetch_and_add(&gb_path_a_count, 1);
                    gb_maybe_write_stats();
                    return CUDA_SUCCESS;
                }
            }
        }
    }

path_b_hostreg:
    /* ---- Path B: HostReg no-kernel (containers / VMs) ------------------- */
    /* mmap anonymous pages (2 MB huge preferred, 4 K fallback) and register  */
    /* directly with the CUDA driver.  No greenboost.ko required — works       */
    /* inside Docker, LXC, KVM guests, WSL2, and shared HPC clusters.          */
    /* Concept by Jerry Nguyen (MR !3); hugepage preference + hash-table        */
    /* integration by Ferran Duarri.                                            */
    /*
     * RAM safety guard — mirrors gb_pin_user_buf() / gb_alloc_buf() in the kernel.
     * Path B never calls any IOCTL so the kernel's safety_reserve_gb check is blind
     * to these allocations.  Two checks must both pass:
     *   1. Cumulative T2 cap: prevent Path A+B together from exceeding virtual_vram_gb.
     *   2. MemAvailable guard: keep at least safety_reserve_gb free for the OS.
     * On failure, fall through to Path C (UVM) which demand-pages and is self-limiting.
     */
    if (real_cuMemHostRegister && real_cuMemHostGetDevicePointer && !gb_no_hostreg) {
        /* Check 1: cumulative T2 cap.
         * Uses gb_effective_t2_cap(): 88% of pool during INFERENCE/STEADY to protect OS headroom. */
        if (gb_t2_pool_bytes > 0) {
            size_t effective_cap = gb_effective_t2_cap();
            size_t t2_used = atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed);
            if (t2_used >= effective_cap || alloc_bytesize > effective_cap - t2_used) {
                gb_log("Path B skip: T2 cap reached (%zu MB used / %zu MB eff_cap) for %zu MB alloc — routing to UVM",
                       t2_used >> 20, effective_cap >> 20, alloc_bytesize >> 20);
                goto path_c_uvm;
            }
        }
        /* Check 2: MemAvailable guard */
        {
            size_t mem_avail = gb_get_mem_available();
            if (mem_avail > 0 &&
                (alloc_bytesize > mem_avail || mem_avail - alloc_bytesize < gb_safety_reserve_bytes)) {
                gb_log("Path B skip: MemAvailable %zu MB < reserve %zu MB + req %zu MB — routing to UVM",
                       mem_avail >> 20, gb_safety_reserve_bytes >> 20, alloc_bytesize >> 20);
                goto path_c_uvm;
            }
        }
        void *hreg_ptr = NULL;
#ifdef MAP_HUGETLB
#  ifdef MAP_HUGE_2MB
        hreg_ptr = mmap(NULL, alloc_bytesize, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
#  else
        hreg_ptr = mmap(NULL, alloc_bytesize, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
#  endif
        if (hreg_ptr == MAP_FAILED) hreg_ptr = NULL;
#endif
        if (!hreg_ptr)
            hreg_ptr = mmap(NULL, alloc_bytesize, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (hreg_ptr && hreg_ptr != MAP_FAILED) {
            ret = real_cuMemHostRegister(hreg_ptr, alloc_bytesize, CU_MEMHOSTREGISTER_DEVICEMAP);
            if (ret == CUDA_SUCCESS) {
                ret = real_cuMemHostGetDevicePointer(dptr, hreg_ptr, 0);
                if (ret == CUDA_SUCCESS) {
                    /* fd=-1: no DMA-BUF fd to close on free */
                    /* ht_insert size = bytesize (uncompressed logical size) */
                    ht_insert(*dptr, bytesize, 0 /* not UVM */, -1, hreg_ptr, -1, NULL);
                    ht_set_flags(*dptr, alloc_flags);
                    if (tq_compressed_sz)
                        ht_set_tq_compressed_size(*dptr, tq_compressed_sz);
                    atomic_fetch_add_explicit(&gb_t2_overflow_bytes, alloc_bytesize, memory_order_relaxed);
                    gb_log("Path B (HostReg/no-kmod): %zu MB (alloc %zu MB TQ) at cuda_ptr=0x%llx (mapped=%p) t2_total=%zu MB",
                           bytesize >> 20, alloc_bytesize >> 20, (unsigned long long)*dptr, hreg_ptr,
                           atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed) >> 20);
                    __sync_fetch_and_add(&gb_path_b_count, 1);
                    gb_maybe_write_stats();
                    return CUDA_SUCCESS;
                }
                if (real_cuMemHostUnregister) real_cuMemHostUnregister(hreg_ptr);
            }
            munmap(hreg_ptr, alloc_bytesize);
        }
        gb_log("Path B (HostReg) failed for %zu MB — falling through to UVM", alloc_bytesize >> 20);
    }

path_c_uvm:
    /* ---- Path C: UVM (cuMemAllocManaged) -------------------------------- */
    /* Primary path for oversize allocs (> physical VRAM); fallback for       */
    /* normal overflow when A0/A/B fail.  UVM auto-migrates hot pages to T1   */
    /* (~336 GB/s) and cold pages to system RAM (~32 GB/s).                  */
    /*
     * Cumulative guardrail: if Path A+B have already filled the T2 pool and
     * there is no T3 headroom, return OOM now rather than letting UVM
     * demand-fault unlimited pages into RAM or swap.  This prevents silent
     * thrashing when the model exceeds the total T2+T3 capacity GreenBoost
     * can serve — Ollama will report "not enough memory" instead of being
     * killed by the OOM killer.
     */
    if (gb_t2_pool_bytes > 0 && gb_virtual_vram_bytes > 0) {
        size_t t2_used = atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed);
        /* Total virtual pool = T2 + T3 (gb_virtual_vram_bytes); T2 pool = gb_t2_pool_bytes.
         * Remaining headroom = (T2+T3) - t2_used.  Note: t2_used tracks only Path A/B,
         * not UVM; UVM pages are demand-paged and not pre-committed here. */
        size_t total_pool = gb_virtual_vram_bytes;  /* T2 + T3 */
        if (t2_used >= total_pool || bytesize > total_pool - t2_used) {
            fprintf(stderr,
                    "[GreenBoost] Path C guardrail: T2+T3 pool exhausted"
                    " (%zu MB used / %zu MB total) — refusing %zu MB UVM alloc\n",
                    t2_used >> 20, total_pool >> 20, bytesize >> 20);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
    }
    /* MemAvailable guard for UVM — prevents demand-paging more pages than the system has RAM for.
     * Paths A and B pin RAM synchronously and already check MemAvailable before proceeding.
     * Path C (UVM) defers faulting to the NVIDIA driver, which has no concept of
     * safety_reserve_gb and will happily fault pages until the system OOM-kills everything.
     *
     * Estimate the portion of this UVM alloc that will demand-fault into system RAM:
     *   usable_vram = actual free VRAM (from cuMemGetInfo cache) minus headroom.
     *   Fallback: physical_vram × 85% when the cache is not yet populated.
     * Using cached free VRAM instead of total physical VRAM is critical: when VRAM
     * is already full of model weights, cached_free ≈ 0, so ram_demand ≈ bytesize
     * and the safety guard fires correctly.  The old physical×85% estimate made
     * ram_demand ≈ 0 for 10 GB allocs on a 12 GB GPU that was already full, letting
     * UVM proceed and exhaust system RAM (OOM kill). */
    {
        size_t mem_avail = gb_get_mem_available();
        if (mem_avail > 0) {
            size_t _cf = atomic_load_explicit(&g_cached_free_vram, memory_order_relaxed);
            size_t usable_vram = (_cf > 0)
                ? ((_cf > vram_headroom_bytes) ? _cf - vram_headroom_bytes : 0)
                : (size_t)(gb_physical_vram_bytes * 85ULL / 100ULL);
            size_t ram_demand  = (bytesize > usable_vram) ? (bytesize - usable_vram) : 0;
            size_t uvm_cumul   = atomic_load_explicit(&gb_uvm_estimated_ram_bytes, memory_order_relaxed);
            size_t total_needed = ram_demand + uvm_cumul;
            if (ram_demand > mem_avail || mem_avail - ram_demand < gb_safety_reserve_bytes) {
                fprintf(stderr,
                        "[GreenBoost] Path C guardrail: MemAvailable %zu MB insufficient for"
                        " UVM RAM demand %zu MB (safety_reserve=%zu MB, uvm_cumul=%zu MB)"
                        " — refusing %zu MB UVM alloc\n",
                        mem_avail >> 20, ram_demand >> 20,
                        gb_safety_reserve_bytes >> 20, uvm_cumul >> 20,
                        bytesize >> 20);
                (void)total_needed;
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
        }
    }

    if (real_cuMemAllocManaged) {
        ret = real_cuMemAllocManaged(dptr, bytesize, CU_MEM_ATTACH_GLOBAL);
        if (ret == 201 /* CUDA_ERROR_INVALID_CONTEXT */ && real_cudaMallocManaged) {
            /* Driver API cuMemAllocManaged requires an active CUDA context (error 201).
             * This happens when the CUDA runtime hasn't initialized the primary context
             * yet (early-probe cudaMalloc calls from ggml before cuInit completes).
             * Fall back to the runtime API which auto-creates the primary context,
             * then continue with the normal success path.
             * cudaMemAttachGlobal (1) == CU_MEM_ATTACH_GLOBAL — same UVM semantics. */
            void *hptr = NULL;
            cudaError_t cret = real_cudaMallocManaged(&hptr, bytesize, 1 /* cudaMemAttachGlobal */);
            if (cret == 0 /* cudaSuccess */) {
                *dptr = (CUdeviceptr)(uintptr_t)hptr;
                ret = CUDA_SUCCESS;
                gb_log("Path C (UVM/runtime): context auto-init, %zu MB at 0x%llx",
                       bytesize >> 20, (unsigned long long)*dptr);
            }
        }
        if (ret == CUDA_SUCCESS) {
            /* Hint UVM driver: pages belong to GPU and GPU maintains direct PTEs.
             * SET_PREFERRED_LOCATION=GPU → driver aggressively fills T1 VRAM.
             * SET_ACCESSED_BY=GPU        → GPU-side PTEs pre-populated, less fault latency. */
            if (real_cuMemAdvise) {
                real_cuMemAdvise(*dptr, bytesize, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, 0);
                real_cuMemAdvise(*dptr, bytesize, CU_MEM_ADVISE_SET_ACCESSED_BY, 0);
            }
            /* Prefetch (physical_vram - headroom) bytes into VRAM immediately to
             * warm the T1 cache and avoid a page-fault storm on first inference
             * pass.  Cold remainder is demand-paged from system RAM as accessed. */
            if (real_cuMemPrefetchAsync && gb_physical_vram_bytes > vram_headroom_bytes) {
                size_t prefetch_sz = gb_physical_vram_bytes - vram_headroom_bytes;
                if (prefetch_sz > bytesize) prefetch_sz = bytesize;
                real_cuMemPrefetchAsync(*dptr, prefetch_sz, 0 /* GPU device 0 */, NULL);
                gb_log("UVM prefetch: %zu MB into T1 VRAM (of %zu MB total)",
                       prefetch_sz >> 20, bytesize >> 20);
            }
            ht_insert(*dptr, bytesize, 1 /* UVM */, -1, NULL, -1, NULL);
            ht_set_flags(*dptr, alloc_flags);
            /* Track estimated system RAM demand — mirrors the MemAvailable guard above.
             * Decremented on cuMemFree_v2 / cuMemFreeAsync when managed == 1.
             * Use same cached-free-VRAM estimate for consistency with the guard. */
            {
                size_t _cf2 = atomic_load_explicit(&g_cached_free_vram, memory_order_relaxed);
                size_t usable_vram = (_cf2 > 0)
                    ? ((_cf2 > vram_headroom_bytes) ? _cf2 - vram_headroom_bytes : 0)
                    : (size_t)(gb_physical_vram_bytes * 85ULL / 100ULL);
                size_t est_ram = (bytesize > usable_vram) ? (bytesize - usable_vram) : 0;
                atomic_fetch_add_explicit(&gb_uvm_estimated_ram_bytes, est_ram, memory_order_relaxed);
                gb_log("Path C (UVM): %zu MB at 0x%llx (est_ram=%zu MB, uvm_cumul=%zu MB)",
                       bytesize >> 20, (unsigned long long)*dptr, est_ram >> 20,
                       atomic_load_explicit(&gb_uvm_estimated_ram_bytes, memory_order_relaxed) >> 20);
            }
            __sync_fetch_and_add(&gb_path_c_count, 1);
            gb_maybe_write_stats();
            return CUDA_SUCCESS;
        }
        /* Always print UVM failure — visible in journalctl without debug mode */
        fprintf(stderr,
                "[GreenBoost] UVM alloc FAILED ret=%d for %zu MB"
                " — check nvidia_uvm is loaded and CUDA context is valid\n",
                ret, bytesize >> 20);
    } else {
        fprintf(stderr, "[GreenBoost] UVM unavailable (real_cuMemAllocManaged=NULL)"
                " for %zu MB\n", bytesize >> 20);
    }

    return CUDA_ERROR_OUT_OF_MEMORY;
}

/* ------------------------------------------------------------------ */
/*  cuMemAlloc_v2 override                                              */
/* ------------------------------------------------------------------ */

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
{
    CUresult ret;

    if (!initialized || !real_cuMemAlloc_v2)
        return CUDA_ERROR_OUT_OF_MEMORY;

    if (gb_needs_overflow(bytesize)) {
        ret = gb_overflow_alloc(dptr, bytesize);
        if (ret == CUDA_SUCCESS)
            return CUDA_SUCCESS;
        /* VRAM was judged full — the native allocator will also fail.
         * Return the overflow error directly to avoid a redundant attempt
         * that would produce a confusing second error in the CUDA log. */
        gb_log("overflow alloc failed (ret=%d) — VRAM full, not retrying native", ret);
        return ret;
    }

    atomic_store_explicit(&g_last_any_alloc_ms, gb_now_ms(), memory_order_relaxed);
    ret = real_cuMemAlloc_v2(dptr, bytesize);
    if (ret == CUDA_SUCCESS) {
        ht_insert(*dptr, bytesize, 0, -1, NULL, -1, NULL);
        gb_maybe_track_kv_t1(*dptr, bytesize);
    }
    return ret;
}

/* ------------------------------------------------------------------ */
/*  cuMemCreate override (CUDA VMM — used by ggml/Ollama 0.18+)        */
/*                                                                      */
/*  ggml's CUDA backend allocates ALL model weight/KV memory via the   */
/*  Virtual Memory Management API (cuMemCreate → cuMemMap → cuMemSetAccess).  */
/*  It never calls cudaMalloc/cuMemAlloc. When cuMemCreate(22 GB) fails */
/*  because T1 is full, we retry with CU_MEM_LOCATION_TYPE_HOST which  */
/*  creates a pinned host-memory allocation the GPU accesses over PCIe. */
/* ------------------------------------------------------------------ */

CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                     const CUmemAllocationProp *prop, unsigned long long flags)
{
    CUresult ret;

    if (!initialized || !real_cuMemCreate)
        return real_cuMemCreate ? real_cuMemCreate(handle, size, prop, flags)
                                : CUDA_ERROR_NOT_SUPPORTED;

    /* Try native device allocation first */
    ret = real_cuMemCreate(handle, size, prop, flags);
    if (ret == CUDA_SUCCESS)
        return CUDA_SUCCESS;

    /* Device OOM → fall back to host-pinned VMM allocation.
     * GPU accesses host-pinned VMM memory over PCIe (~32 GB/s) —
     * same bandwidth class as Path B.  No changes to cuMemMap/cuMemSetAccess
     * are required; they work identically for host-backed handles. */
    if (ret == CUDA_ERROR_OUT_OF_MEMORY) {
        /* RAM safety guard for VMM host-pinned fallback — mirrors Path B guard. */
        {
            size_t mem_avail = gb_get_mem_available();
            if (mem_avail > 0 &&
                (size > mem_avail || mem_avail - size < gb_safety_reserve_bytes)) {
                gb_log("cuMemCreate VMM skip: MemAvailable %zu MB < reserve %zu MB + req %zu MB",
                       mem_avail >> 20, gb_safety_reserve_bytes >> 20, size >> 20);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
        }
        /* T2 inference cap — same 88% threshold as Paths A/B (gb_effective_t2_cap).
         * Prevents VMM host-pinned KV cache from consuming DDR past the safe margin
         * before the 4 GB safety reserve kicks in. */
        {
            size_t t2_used = atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed);
            size_t eff_cap = gb_effective_t2_cap();
            if (eff_cap > 0 && (t2_used >= eff_cap || size > eff_cap - t2_used)) {
                gb_log("cuMemCreate VMM skip: T2 inference cap %zu/%zu MB, req %zu MB",
                       t2_used >> 20, eff_cap >> 20, size >> 20);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
        }

        CUmemAllocationProp host_prop = *prop;
        host_prop.location.type = CU_MEM_LOCATION_TYPE_HOST;
        host_prop.location.id   = 0;
        host_prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;

        ret = real_cuMemCreate(handle, size, &host_prop, flags);
        if (ret == CUDA_SUCCESS) {
            atomic_fetch_add_explicit(&gb_t2_overflow_bytes, size, memory_order_relaxed);
            __sync_fetch_and_add(&gb_path_b_count, 1);
            gb_maybe_write_stats();
            gb_log("cuMemCreate VMM fallback: %zu MB → host-pinned (PCIe path) t2_total=%zu MB",
                   size >> 20,
                   atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed) >> 20);
            return CUDA_SUCCESS;
        }
        fprintf(stderr,
                "[GreenBoost] cuMemCreate VMM host fallback FAILED ret=%d for %zu MB\n",
                ret, (size >> 20));
    }

    return ret;
}

/* ------------------------------------------------------------------ */
/*  cuMemFree_v2 override                                               */
/* ------------------------------------------------------------------ */

CUresult cuMemFree_v2(CUdeviceptr dptr)
{
    void *mapped_ptr = NULL;
    int fd = -1;
    size_t sz = 0;
    int managed = 0;
    cudaExternalMemory_t ext_mem = NULL;
    uint32_t flags;

    if (!initialized || !real_cuMemFree_v2)
        return CUDA_SUCCESS;

    /* AUD-06: NULL dptr (dptr == 0) is legal in CUDA — it is a documented no-op.
     * The hash table uses ptr == 0 as the empty-slot sentinel, so ht_remove(0, ...)
     * returns 0 (not found) and execution falls through to real_cuMemFree_v2(0),
     * which the driver also treats as a no-op.  No special case is needed here;
     * this comment documents the invariant so future refactors preserve it. */

    flags = ht_peek_flags(dptr);

    if (ht_remove(dptr, &sz, &managed, &mapped_ptr, &fd, &ext_mem)) {
        /* REF-02: KV T1 release via shared helper (same logic as cudaFree). */
        gb_release_kv_t1_bytes(flags, sz, mapped_ptr, ext_mem, managed);
        gb_log("cuMemFree_v2 ptr=0x%llx size=%zu MB managed=%d mapped_ptr=%p fd=%d ext_mem=%p",
               (unsigned long long)dptr, sz >> 20, managed, mapped_ptr, fd, (void *)ext_mem);
        /* Path A0: CUDA owns the mapping — destroy the external memory handle. */
        if (ext_mem) {
            if (real_cudaDestroyExternalMemory)
                real_cudaDestroyExternalMemory(ext_mem);
            return CUDA_SUCCESS;
        }
        /* Path A / Path B: host-registered memory — unregister and unmap. */
        if (mapped_ptr) {
            if (real_cuMemHostUnregister) real_cuMemHostUnregister(mapped_ptr);
            munmap(mapped_ptr, sz);
            /* Decrement cumulative T2 counter — mirrors the increment in gb_overflow_alloc(). */
            atomic_fetch_sub_explicit(&gb_t2_overflow_bytes, sz, memory_order_relaxed);
        }
        if (fd >= 0)
            close(fd);
        /* DMA-BUF: dptr came from cuMemHostGetDevicePointer, not cuMemAlloc —
         * calling cuMemFree on it is invalid and causes a CUDA driver error.
         * Only call the real free for UVM / regular device allocations. */
        if (!mapped_ptr) {
            /* Decrement UVM estimated RAM tracker for Path C allocs.
             * Use same formula as alloc-time tracking for symmetry. */
            if (managed) {
                size_t _cf3 = atomic_load_explicit(&g_cached_free_vram, memory_order_relaxed);
                size_t usable_vram = (_cf3 > 0)
                    ? ((_cf3 > vram_headroom_bytes) ? _cf3 - vram_headroom_bytes : 0)
                    : (size_t)(gb_physical_vram_bytes * 85ULL / 100ULL);
                size_t est_ram = (sz > usable_vram) ? (sz - usable_vram) : 0;
                atomic_fetch_sub_explicit(&gb_uvm_estimated_ram_bytes, est_ram, memory_order_relaxed);
            }
            return real_cuMemFree_v2(dptr);
        }
        return CUDA_SUCCESS;
    }

    return real_cuMemFree_v2(dptr);
}

/* ------------------------------------------------------------------ */
/*  cuMemAllocAsync override (CUDA 11.2+ stream-ordered allocator)      */
/* ------------------------------------------------------------------ */

CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream)
{
    CUresult ret;

    if (!initialized)
        return CUDA_ERROR_OUT_OF_MEMORY;

    /* AUD-05: cuMemAllocAsync requires compute capability >= 8.0 (Ampere+).
     * V100 (cc 7.0) and older return CUDA_ERROR_NOT_SUPPORTED silently.
     * Fall back to sync cuMemAlloc_v2 when cc < 8 or unknown. */
    if (!real_cuMemAllocAsync || (gb_cc_major > 0 && gb_cc_major < 8))
        return cuMemAlloc_v2(dptr, bytesize);

    if (gb_needs_overflow(bytesize)) {
        ret = gb_overflow_alloc(dptr, bytesize);
        if (ret == CUDA_SUCCESS)
            return CUDA_SUCCESS;
    }

    ret = real_cuMemAllocAsync(dptr, bytesize, hStream);
    if (ret == CUDA_SUCCESS) {
        ht_insert(*dptr, bytesize, 0, -1, NULL, -1, NULL);
        gb_maybe_track_kv_t1(*dptr, bytesize);
    }
    /* MIN-08: cuMemAllocAsync is intentionally asymmetric here:
     * - Alloc: uses the async stream-ordered path (lower latency in busy streams)
     * - Free:  cuMemFreeAsync (below) handles the paired deallocation
     *
     * If cuMemAllocAsync falls back to cuMemAlloc_v2 (cc < 8 branch above),
     * the paired free is still cuMemFreeAsync → cuMemFree_v2 (the real_cuMemFreeAsync
     * path is bypassed for the sync fallback case, so no mismatch).
     * The overflow alloc path (gb_overflow_alloc) uses the same ext_mem/mapped_ptr
     * tracking as cudaMalloc — freed correctly by cuMemFreeAsync via ht_lookup. */
    return ret;
}

/* ------------------------------------------------------------------ */
/*  cudaMalloc override                                                 */
/* ------------------------------------------------------------------ */

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
    cudaError_t ret;
    CUdeviceptr dptr = 0;

    if (!initialized)
        return (cudaError_t)CUDA_ERROR_OUT_OF_MEMORY;

    /* Always-on trace for large allocations — helps diagnose T2 fill issues */
    if (size >= 100ULL * 1024 * 1024)
        fprintf(stderr, "[GreenBoost] cudaMalloc hook called: %zu MB\n", size >> 20);

    /* Lazy resolve: libcudart may have been loaded by caller after our constructor.
     * RTLD_NEXT skips our own symbol and finds the real one in the next library. */
    if (!real_cudaMalloc)
        real_cudaMalloc = (pfn_cudaMalloc)dlsym(RTLD_NEXT, "cudaMalloc");
    if (!real_cudaMalloc)
        return (cudaError_t)CUDA_ERROR_OUT_OF_MEMORY;

    if (gb_needs_overflow(size)) {
        ret = (cudaError_t)gb_overflow_alloc(&dptr, size);
        if (ret == CUDA_SUCCESS) {
            *devPtr = (void *)(uintptr_t)dptr;
            return CUDA_SUCCESS;
        }
        /* VRAM was judged full — skip the native fallback to avoid a second
         * error log entry from the driver for the same out-of-memory condition. */
        gb_log("cudaMalloc overflow failed (ret=%d) — VRAM full, not retrying native", ret);
        return ret;
    }

    atomic_store_explicit(&g_last_any_alloc_ms, gb_now_ms(), memory_order_relaxed);
    ret = real_cudaMalloc(devPtr, size);
    if (ret == CUDA_SUCCESS) {
        CUdeviceptr dptr = (CUdeviceptr)(uintptr_t)*devPtr;
        ht_insert(dptr, size, 0, -1, NULL, -1, NULL);
        gb_maybe_track_kv_t1(dptr, size);
    }
    return ret;
}

/* ------------------------------------------------------------------ */
/*  cudaMallocAsync override                                            */
/* ------------------------------------------------------------------ */

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream)
{
    cudaError_t ret;
    CUdeviceptr dptr = 0;

    if (!initialized)
        return (cudaError_t)CUDA_ERROR_OUT_OF_MEMORY;

    /* Lazy resolve: RTLD_NEXT skips our own symbol, finds real cudaMallocAsync in libcudart */
    if (!real_cudaMallocAsync)
        real_cudaMallocAsync = (pfn_cudaMallocAsync)dlsym(RTLD_NEXT, "cudaMallocAsync");

    /* Fall back to sync cudaMalloc (stream ordering ignored — safe for model weights) */
    if (!real_cudaMallocAsync)
        return cudaMalloc(devPtr, size);

    if (gb_needs_overflow(size)) {
        ret = (cudaError_t)gb_overflow_alloc(&dptr, size);
        if (ret == CUDA_SUCCESS) {
            *devPtr = (void *)(uintptr_t)dptr;
            return CUDA_SUCCESS;
        }
    }

    atomic_store_explicit(&g_last_any_alloc_ms, gb_now_ms(), memory_order_relaxed);
    ret = real_cudaMallocAsync(devPtr, size, stream);
    if (ret == CUDA_SUCCESS) {
        CUdeviceptr dptr = (CUdeviceptr)(uintptr_t)*devPtr;
        ht_insert(dptr, size, 0, -1, NULL, -1, NULL);
        gb_maybe_track_kv_t1(dptr, size);
    }
    return ret;
}

/* ------------------------------------------------------------------ */
/*  cudaFree override                                                   */
/* ------------------------------------------------------------------ */

cudaError_t cudaFree(void *devPtr)
{
    void *mapped_ptr = NULL;
    int fd = -1;
    size_t sz = 0;
    int managed = 0;
    cudaExternalMemory_t ext_mem = NULL;
    CUdeviceptr dptr = (CUdeviceptr)(uintptr_t)devPtr;
    uint32_t flags;

    if (!initialized)
        return CUDA_SUCCESS; /* free before init: no-op is safe */

    if (!real_cudaFree)
        real_cudaFree = (pfn_cudaFree)dlsym(RTLD_NEXT, "cudaFree");
    if (!real_cudaFree)
        return CUDA_SUCCESS;

    /* AUD-06: cudaFree(NULL) is a documented CUDA no-op.  dptr == 0 is the
     * empty-slot sentinel in the hash table, so ht_remove(0,...) returns 0
     * (not found) and we fall through to real_cudaFree(NULL) — correct. */

    flags = ht_peek_flags(dptr);

    if (ht_remove(dptr, &sz, &managed, &mapped_ptr, &fd, &ext_mem)) {
        /* REF-02: KV T1 release via shared helper (same logic as cuMemFree_v2). */
        gb_release_kv_t1_bytes(flags, sz, mapped_ptr, ext_mem, managed);
        gb_log("cudaFree ptr=0x%llx size=%zu MB managed=%d mapped_ptr=%p fd=%d ext_mem=%p",
               (unsigned long long)dptr, sz >> 20, managed, mapped_ptr, fd, (void *)ext_mem);
        /* Path A0: CUDA owns the mapping — destroy the external memory handle. */
        if (ext_mem) {
            if (real_cudaDestroyExternalMemory)
                real_cudaDestroyExternalMemory(ext_mem);
            return CUDA_SUCCESS;
        }
        /* Path A / Path B: host-registered memory — unregister and unmap. */
        if (mapped_ptr) {
            if (real_cuMemHostUnregister) real_cuMemHostUnregister(mapped_ptr);
            munmap(mapped_ptr, sz);
            /* Decrement cumulative T2 counter — mirrors the increment in gb_overflow_alloc(). */
            atomic_fetch_sub_explicit(&gb_t2_overflow_bytes, sz, memory_order_relaxed);
        }
        if (fd >= 0)
            close(fd);
        /* DMA-BUF: dptr came from cuMemHostGetDevicePointer, not cudaMalloc —
         * must not pass to cudaFree. Only free UVM / regular device allocations. */
        if (!mapped_ptr)
            return real_cudaFree(devPtr);
        return CUDA_SUCCESS;
    }

    return real_cudaFree(devPtr);
}

/* ------------------------------------------------------------------ */
/*  Prefetching overrides (cuMemPrefetchAsync, cudaMemPrefetchAsync)    */
/* ------------------------------------------------------------------ */

CUresult cuMemPrefetchAsync(CUdeviceptr dptr, size_t count, CUdevice dstDevice, CUstream hStream)
{
    void *mapped_ptr = NULL;
    size_t sz = 0;
    int managed = 0, fd = -1;

    if (!initialized || !real_cuMemPrefetchAsync)
        return CUDA_SUCCESS;

    /* Check if this is a GreenBoost DMA-BUF allocation */
    if (ht_lookup(dptr, &sz, &managed, &mapped_ptr, &fd)) {
        if (mapped_ptr) {
            enqueue_prefetch(mapped_ptr, count < sz ? count : sz);
        }
        /* We handled the prefetch via host thread, skip real CUDA prefetch
           because CUDA doesn't prefetch cuMemHostRegister memory implicitly */
        return CUDA_SUCCESS;
    }

    return real_cuMemPrefetchAsync(dptr, count, dstDevice, hStream);
}

cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream)
{
    void *mapped_ptr = NULL;
    size_t sz = 0;
    int managed = 0, fd = -1;
    CUdeviceptr dptr = (CUdeviceptr)(uintptr_t)devPtr;

    if (!initialized)
        return CUDA_SUCCESS;

    if (!real_cudaMemPrefetchAsync) {
        real_cudaMemPrefetchAsync = (pfn_cudaMemPrefetchAsync)dlsym(RTLD_NEXT, "cudaMemPrefetchAsync");
    }

    if (ht_lookup(dptr, &sz, &managed, &mapped_ptr, &fd)) {
        if (mapped_ptr) {
            enqueue_prefetch(mapped_ptr, count < sz ? count : sz);
        }
        return CUDA_SUCCESS;
    }

    if (real_cudaMemPrefetchAsync)
        return real_cudaMemPrefetchAsync(devPtr, count, dstDevice, stream);

    return CUDA_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  cuMemGetInfo_v2 / cuMemGetInfo / cudaMemGetInfo overrides          */
/*                                                                      */
/*  Report virtual VRAM = real VRAM + System DDR pool so the CUDA runtime    */
/*  and Ollama scheduler offload ALL model layers as GPU tensors.       */
/*  Each resulting cudaMalloc that overflows real VRAM is then          */
/*  redirected to system RAM via DMA-BUF (gb_needs_overflow uses real VRAM). */
/* ------------------------------------------------------------------ */

CUresult cuMemGetInfo_v2(size_t *free_out, size_t *total_out)
{
    size_t real_free = 0, real_total = 0;
    CUresult ret;

    if (!initialized || !real_cuMemGetInfo)
        return CUDA_ERROR_NOT_SUPPORTED;

    ret = real_cuMemGetInfo(&real_free, &real_total);
    if (ret != CUDA_SUCCESS)
        return ret;

    /* total = real VRAM + full virtual pool (T2+T3) — keeps all model layers on GPU.
     * free  = real VRAM free + T2 DDR available only (excludes T3 NVMe):
     *   T3 is capacity for model loading, not fast memory for KV cache.
     *   Reporting T3 as "free" causes callers (ollama) to set a context length
     *   whose KV cache exceeds available system RAM and triggers an OOM kill. */
    if (free_out)  *free_out  = real_free  + gb_t2_free_to_report();
    if (total_out) *total_out = real_total + gb_virtual_vram_bytes;

    gb_log("cuMemGetInfo_v2: real_free=%zuMB t2_free=%zuMB → virtual_free=%zuMB total=%zuMB",
           real_free >> 20, gb_t2_free_to_report() >> 20,
           (real_free + gb_t2_free_to_report()) >> 20,
           (real_total + gb_virtual_vram_bytes) >> 20);
    return CUDA_SUCCESS;
}

CUresult cuMemGetInfo(size_t *free_out, size_t *total_out)
{
    return cuMemGetInfo_v2(free_out, total_out);
}

cudaError_t cudaMemGetInfo(size_t *free_out, size_t *total_out)
{
    size_t real_free = 0, real_total = 0;
    CUresult ret;

    if (!initialized || !real_cuMemGetInfo)
        return (cudaError_t)CUDA_ERROR_NOT_SUPPORTED;

    /* Call real driver function directly — avoids double-inflation if libcudart
     * internally calls cuMemGetInfo_v2 (which we also override). */
    ret = real_cuMemGetInfo(&real_free, &real_total);
    if (ret != CUDA_SUCCESS)
        return (cudaError_t)ret;

    /* Same split as cuMemGetInfo_v2: free = T2 available only; total = T2+T3 capacity. */
    if (free_out)  *free_out  = real_free  + gb_t2_free_to_report();
    if (total_out) *total_out = real_total + gb_virtual_vram_bytes;

    gb_log("cudaMemGetInfo: real_free=%zuMB t2_free=%zuMB → virtual_free=%zuMB total=%zuMB",
           real_free >> 20, gb_t2_free_to_report() >> 20,
           (real_free + gb_t2_free_to_report()) >> 20,
           (real_total + gb_virtual_vram_bytes) >> 20);
    return CUDA_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  cuDeviceTotalMem overrides                                          */
/*                                                                      */
/*  Ollama's scheduler calls cuDeviceTotalMem at startup to determine   */
/*  how many model layers fit in VRAM.  Inflate by system RAM pool size so   */
/*  all layers are scheduled as GPU tensors; the allocations that       */
/*  overflow real VRAM are caught by cudaMalloc/cuMemAlloc above.       */
/* ------------------------------------------------------------------ */

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev)
{
    CUresult ret;
    size_t total_virtual = gb_virtual_vram_bytes;

    if (!initialized || !real_cuDeviceTotalMem_v2)
        return CUDA_ERROR_NOT_SUPPORTED;

    ret = real_cuDeviceTotalMem_v2(bytes, dev);
    if (ret == CUDA_SUCCESS && bytes) {
        /* Base virtual = physical VRAM + system RAM pool */
        total_virtual = *bytes + gb_virtual_vram_bytes;
        
        /* Add NVLink aggregated VRAM if pooling is active */
        if (gb_nvlink_aggregated_bytes > 0) {
            total_virtual += gb_nvlink_aggregated_bytes;
            gb_log("cuDeviceTotalMem_v2: with NVLink: phys=%zuMB + sysram=%zuMB + nvlink=%zuMB = total=%zuMB",
                   *bytes >> 20, gb_virtual_vram_bytes >> 20,
                   gb_nvlink_aggregated_bytes >> 20, total_virtual >> 20);
        } else {
            gb_log("cuDeviceTotalMem_v2: no NVLink: phys=%zuMB + sysram=%zuMB = total=%zuMB",
                   *bytes >> 20, gb_virtual_vram_bytes >> 20, total_virtual >> 20);
        }
        *bytes = total_virtual;
    }
    return ret;
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev)
{
    return cuDeviceTotalMem_v2(bytes, dev);
}

/* ------------------------------------------------------------------ */
/*  NVML overrides — nvmlDeviceGetMemoryInfo[_v2]                      */
/*                                                                      */
/*  Ollama's discover/nvidia.go uses NVML for initial GPU sizing.       */
/*  Inflate total + free so the layer scheduler sees virtual VRAM.      */
/* ------------------------------------------------------------------ */

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory)
{
    nvmlReturn_t ret;

    if (!real_nvmlDeviceGetMemoryInfo)
        return NVML_ERROR_FUNCTION_NOT_FOUND;

    ret = real_nvmlDeviceGetMemoryInfo(device, memory);
    if (ret == NVML_SUCCESS && memory) {
        gb_log("nvmlDeviceGetMemoryInfo: real_total=%lluMB → virtual_total=%lluMB t2_free=%zuMB",
               memory->total >> 20,
               (memory->total + gb_virtual_vram_bytes) >> 20,
               gb_t2_free_to_report() >> 20);
        memory->total += gb_virtual_vram_bytes;
        memory->free  += gb_t2_free_to_report();
    }
    return ret;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory)
{
    nvmlReturn_t ret;

    if (!real_nvmlDeviceGetMemoryInfo_v2)
        return NVML_ERROR_FUNCTION_NOT_FOUND;

    ret = real_nvmlDeviceGetMemoryInfo_v2(device, memory);
    if (ret == NVML_SUCCESS && memory) {
        gb_log("nvmlDeviceGetMemoryInfo_v2: real_total=%lluMB → virtual_total=%lluMB t2_free=%zuMB",
               memory->total >> 20,
               (memory->total + gb_virtual_vram_bytes) >> 20,
               gb_t2_free_to_report() >> 20);
        memory->total += gb_virtual_vram_bytes;
        memory->free  += gb_t2_free_to_report();
    }
    return ret;
}

/* ------------------------------------------------------------------ */
/*  AUD-04: nvmlDeviceGetMemoryInfo_v3 hook (NVML 12.x / driver 520+) */
/*                                                                      */
/*  ExLlamaV3 and libraries linked against NVML 12.x may call v3       */
/*  instead of v2.  Without this hook they see physical VRAM (12 GB)   */
/*  instead of virtual VRAM (63 GB).                                   */
/* ------------------------------------------------------------------ */

nvmlReturn_t nvmlDeviceGetMemoryInfo_v3(nvmlDevice_t device, nvmlMemory_v3_t *memory)
{
    nvmlReturn_t ret;

    if (!real_nvmlDeviceGetMemoryInfo_v3)
        return NVML_ERROR_FUNCTION_NOT_FOUND;

    ret = real_nvmlDeviceGetMemoryInfo_v3(device, memory);
    if (ret == NVML_SUCCESS && memory) {
        gb_log("nvmlDeviceGetMemoryInfo_v3: real_total=%lluMB → virtual_total=%lluMB t2_free=%zuMB",
               memory->total >> 20,
               (memory->total + gb_virtual_vram_bytes) >> 20,
               gb_t2_free_to_report() >> 20);
        memory->total += gb_virtual_vram_bytes;
        memory->free  += gb_t2_free_to_report();
    }
    return ret;
}

/* ------------------------------------------------------------------ */
/*  AUD-03: cuMemFreeAsync hook                                        */
/*                                                                      */
/*  PyTorch 2.x CUDA caching allocator uses cuMemFreeAsync for         */
/*  stream-ordered frees.  Without this hook, async-freed T2 buffers   */
/*  remain in the devPtr→gb_buf hash map, leaking entries and          */
/*  eventually degrading O(1) lookup to O(n) over long sessions.       */
/*                                                                      */
/*  The stream argument is intentionally ignored: by the time the free  */
/*  is called the buffer is already resident — cleanup is synchronous.  */
/* ------------------------------------------------------------------ */

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream)
{
    size_t sz = 0;
    int managed = 0;
    void *mapped_ptr = NULL;
    int fd = -1;
    cudaExternalMemory_t ext_mem = NULL;

    if (!initialized) {
        if (real_cuMemFreeAsync)
            return real_cuMemFreeAsync(dptr, hStream);
        return CUDA_SUCCESS;
    }

    {
        uint32_t flags = ht_peek_flags(dptr);
        if (ht_remove(dptr, &sz, &managed, &mapped_ptr, &fd, &ext_mem)) {
            gb_release_kv_t1_bytes(flags, sz, mapped_ptr, ext_mem, managed);
            gb_log("cuMemFreeAsync ptr=0x%llx size=%zu MB managed=%d mapped_ptr=%p fd=%d ext_mem=%p",
                   (unsigned long long)dptr, sz >> 20, managed, mapped_ptr, fd, (void *)ext_mem);
            /* Path A0: CUDA owns the mapping — destroy the external memory handle. */
            if (ext_mem) {
                if (real_cudaDestroyExternalMemory)
                    real_cudaDestroyExternalMemory(ext_mem);
                return CUDA_SUCCESS;
            }
            /* Path A / Path B: host-registered memory — unregister and unmap. */
            if (mapped_ptr) {
                if (real_cuMemHostUnregister) real_cuMemHostUnregister(mapped_ptr);
                munmap(mapped_ptr, sz);
                /* Decrement cumulative T2 counter — mirrors the increment in gb_overflow_alloc(). */
                atomic_fetch_sub_explicit(&gb_t2_overflow_bytes, sz, memory_order_relaxed);
            }
            if (fd >= 0)
                close(fd);
            /* DMA-BUF / HostReg allocs: dptr came from cuMemHostGetDevicePointer,
             * not cuMemAlloc — must not pass to cuMemFreeAsync. */
            if (!mapped_ptr && real_cuMemFreeAsync) {
                /* Decrement UVM estimated RAM tracker for Path C allocs.
                 * Use same formula as alloc-time tracking for symmetry. */
                if (managed) {
                    size_t _cf4 = atomic_load_explicit(&g_cached_free_vram, memory_order_relaxed);
                    size_t usable_vram = (_cf4 > 0)
                        ? ((_cf4 > vram_headroom_bytes) ? _cf4 - vram_headroom_bytes : 0)
                        : (size_t)(gb_physical_vram_bytes * 85ULL / 100ULL);
                    size_t est_ram = (sz > usable_vram) ? (sz - usable_vram) : 0;
                    atomic_fetch_sub_explicit(&gb_uvm_estimated_ram_bytes, est_ram, memory_order_relaxed);
                }
                return real_cuMemFreeAsync(dptr, hStream);
            }
            return CUDA_SUCCESS;
        }
    }

    if (real_cuMemFreeAsync)
        return real_cuMemFreeAsync(dptr, hStream);
    return CUDA_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  dlsym hook — intercepts dlopen-based GPU API lookups               */
/*                                                                      */
/*  Ollama accesses NVML and CUDA driver via dlopen+dlsym, which       */
/*  bypasses standard LD_PRELOAD interception.  We override dlsym to   */
/*  return our hooked versions for memory-reporting symbols so that     */
/*  Ollama's GPU discovery sees the virtual (inflated) VRAM size.       */
/*                                                                      */
/*  Bootstrap: __libc_dlsym is used to get the REAL dlsym without      */
/*  triggering a recursive call through our own override.               */
/* ------------------------------------------------------------------ */

typedef void *(*pfn_dlsym_t)(void *, const char *);
typedef void *(*pfn_dlopen_t)(const char *, int);
static pfn_dlsym_t  real_dlsym_fn  = NULL;
static pfn_dlopen_t real_dlopen_fn = NULL;

/* Bootstrap: run before gb_shim_init() to capture real dlsym and dlopen.
 *
 * We override both dlsym and dlopen, so we need their real pointers before
 * our overrides are active.  dlvsym with an explicit version skips our
 * unversioned overrides and finds the glibc implementations.
 *
 * On glibc >= 2.34 (Ubuntu 22.04+) both are at GLIBC_2.34 / GLIBC_2.2.5.
 * Priority 101 ensures this runs before the default-priority gb_shim_init. */
__attribute__((constructor(101)))
static void gb_dlsym_bootstrap(void)
{
    /* Detect Wine/Proton: WINEPREFIX is always set by Proton for every process
     * it launches, including the wine64 binary that runs the game.
     * Double-checked below in gb_shim_init via __wine_main_argc symbol. */
    if (getenv("WINEPREFIX") || getenv("WINELOADERNOEXEC") ||
        getenv("WINE_NOTIFY_CLASS")) {
        gb_wine_process = 1;
    }

    /* dlsym — try newest version first */
    real_dlsym_fn = (pfn_dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
    if (!real_dlsym_fn)
        real_dlsym_fn = (pfn_dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    if (!real_dlsym_fn)
        real_dlsym_fn = (pfn_dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.17");
    if (!real_dlsym_fn)
        real_dlsym_fn = (pfn_dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.0");

    /* MIN-07: Warn if all dlsym version attempts fail — means glibc is older
     * than 2.0 (impossible in practice) or the runtime is musl/uclibc. */
    if (!real_dlsym_fn)
        fprintf(stderr,
            "[GreenBoost] FATAL: dlvsym failed for all known glibc versions — "
            "dlsym hook unavailable; Ollama GPU API interception will not work.\n");

    /* dlopen — same version chain */
    real_dlopen_fn = (pfn_dlopen_t)dlvsym(RTLD_NEXT, "dlopen", "GLIBC_2.34");
    if (!real_dlopen_fn)
        real_dlopen_fn = (pfn_dlopen_t)dlvsym(RTLD_NEXT, "dlopen", "GLIBC_2.2.5");
    if (!real_dlopen_fn)
        real_dlopen_fn = (pfn_dlopen_t)dlvsym(RTLD_NEXT, "dlopen", "GLIBC_2.17");
    if (!real_dlopen_fn)
        real_dlopen_fn = (pfn_dlopen_t)dlvsym(RTLD_NEXT, "dlopen", "GLIBC_2.0");

    if (!real_dlopen_fn)
        fprintf(stderr,
            "[GreenBoost] FATAL: dlvsym failed for all known glibc versions — "
            "dlopen hook unavailable; RTLD_DEEPBIND stripping will not work.\n");
}

/* Return our hook for a given symbol name, or NULL if not intercepted. */
static void *gb_get_hook(const char *name)
{
    if (!name) return NULL;

    /* NVML memory reporting — used by Ollama for initial VRAM discovery */
    if (strcmp(name, "nvmlDeviceGetMemoryInfo")    == 0) return (void *)nvmlDeviceGetMemoryInfo;
    if (strcmp(name, "nvmlDeviceGetMemoryInfo_v2") == 0) return (void *)nvmlDeviceGetMemoryInfo_v2;
    if (strcmp(name, "nvmlDeviceGetMemoryInfo_v3") == 0) return (void *)nvmlDeviceGetMemoryInfo_v3;

    /* CUDA device total memory — queried by scheduler at startup */
    if (strcmp(name, "cuDeviceTotalMem_v2")        == 0) return (void *)cuDeviceTotalMem_v2;
    if (strcmp(name, "cuDeviceTotalMem")           == 0) return (void *)cuDeviceTotalMem;

    /* CUDA free/total memory info */
    if (strcmp(name, "cuMemGetInfo_v2")            == 0) return (void *)cuMemGetInfo_v2;
    if (strcmp(name, "cuMemGetInfo")               == 0) return (void *)cuMemGetInfo;
    if (strcmp(name, "cudaMemGetInfo")             == 0) return (void *)cudaMemGetInfo;

    /* CUDA allocation — large allocs redirected to system RAM pool */
    if (strcmp(name, "cudaMalloc")                 == 0) return (void *)cudaMalloc;
    if (strcmp(name, "cudaMallocAsync")            == 0) return (void *)cudaMallocAsync;
    if (strcmp(name, "cudaFree")                   == 0) return (void *)cudaFree;
    if (strcmp(name, "cuMemAlloc_v2")              == 0) return (void *)cuMemAlloc_v2;
    if (strcmp(name, "cuMemAllocAsync")            == 0) return (void *)cuMemAllocAsync;
    if (strcmp(name, "cuMemFree_v2")               == 0) return (void *)cuMemFree_v2;
    if (strcmp(name, "cuMemFreeAsync")             == 0) return (void *)cuMemFreeAsync;
    if (strcmp(name, "cuMemPrefetchAsync")         == 0) return (void *)cuMemPrefetchAsync;
    if (strcmp(name, "cudaMemPrefetchAsync")       == 0) return (void *)cudaMemPrefetchAsync;

    return NULL;
}

/* NVML-only hook table for gaming mode (no CUDA initialized). */
static void *gb_get_nvml_hook(const char *name)
{
    if (!name) return NULL;
    if (strcmp(name, "nvmlDeviceGetMemoryInfo")    == 0) return (void *)nvmlDeviceGetMemoryInfo;
    if (strcmp(name, "nvmlDeviceGetMemoryInfo_v2") == 0) return (void *)nvmlDeviceGetMemoryInfo_v2;
    if (strcmp(name, "nvmlDeviceGetMemoryInfo_v3") == 0) return (void *)nvmlDeviceGetMemoryInfo_v3;
    return NULL;
}

void *dlsym(void *handle, const char *name)
{
    void *hook;

    /* Only intercept after GreenBoost has fully initialized, and ONLY for
     * library-specific handles (not RTLD_NEXT).  Our own code uses RTLD_NEXT
     * to find real implementations — intercepting those causes infinite
     * recursion (real_cudaMalloc = dlsym(RTLD_NEXT,"cudaMalloc") → our hook
     * → calls itself). */
    if (initialized && handle != RTLD_NEXT) {
        hook = gb_get_hook(name);
        if (hook) {
            gb_log("dlsym hook: '%s' → GreenBoost intercepted", name);
            return hook;
        }
    } else if (nvml_hooks_active && handle != RTLD_NEXT) {
        /* Gaming NVML mode: shim is CUDA-inert but NVML hooks are live.
         * MangoHud calls dlsym(handle, "nvmlDeviceGetMemoryInfo") — intercept
         * it here so the overlay sees virtual VRAM instead of physical. */
        hook = gb_get_nvml_hook(name);
        if (hook) {
            if (gb_debug)
                fprintf(stderr, "[GreenBoost] dlsym gaming-NVML: '%s' → intercepted\n", name);
            return hook;
        }
    }

    if (real_dlsym_fn)
        return real_dlsym_fn(handle, name);

    /* Bootstrap failed — return NULL rather than calling the broken
     * dlvsym(handle,name,"GLIBC_2.0") which returns NULL for all CUDA/NVML
     * symbols anyway and would silently break the caller's initialization. */
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  dlopen hook — strips RTLD_DEEPBIND so LD_PRELOAD hooks stay active */
/*                                                                      */
/*  Ollama loads libggml-cuda.so with RTLD_DEEPBIND, which makes CUDA  */
/*  symbol lookups inside that library prefer libcudart.so (bundled),  */
/*  completely bypassing our cudaMalloc/cuMemAlloc LD_PRELOAD hooks.   */
/*  Stripping RTLD_DEEPBIND forces those symbols to resolve from the   */
/*  global namespace where our overrides are registered first.          */
/* ------------------------------------------------------------------ */

void *dlopen(const char *filename, int flags)
{
    /* RTLD_DEEPBIND = 0x008 on Linux — isolates library from global NS.
     * Only strip it when GreenBoost is fully initialized (confirmed CUDA/Ollama
     * process).  Wine/Proton games rely on RTLD_DEEPBIND to isolate PE DLLs;
     * stripping it unconditionally corrupts their symbol resolution and causes
     * crashes (e.g. ucrtbase.dll!strlen receiving an invalid pointer). */
    if (initialized && (flags & RTLD_DEEPBIND)) {
        flags &= ~RTLD_DEEPBIND;
        fprintf(stderr,
                "[GreenBoost] dlopen: stripped RTLD_DEEPBIND from '%s'"
                " (keeps cudaMalloc/cuMemAlloc hooks active)\n",
                filename ? filename : "(null)");
    }

    if (real_dlopen_fn)
        return real_dlopen_fn(filename, flags);

    /* Fallback: bootstrap hasn't run yet — should never happen since
     * constructor(101) runs before any user code calls dlopen. */
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  sysinfo() hook — inflate system RAM by virtual VRAM pool size       */
/*                                                                      */
/*  Ollama's Go runtime calls syscall.Sysinfo() to check total system   */
/*  RAM before loading a model.  When physical RAM < model size,        */
/*  Ollama rejects the request with "model requires more system memory"  */
/*  before any CUDA allocation is attempted — our NVML/CUDA hooks never  */
/*  get a chance to run.                                                 */
/*                                                                      */
/*  By inflating totalram and freeram by gb_virtual_vram_bytes we make  */
/*  Ollama see the full virtual capacity (T1 + T2) as available memory, */
/*  consistent with what our cuDeviceTotalMem / nvmlDeviceGetMemoryInfo  */
/*  hooks already report.  Only active when the shim is initialized and  */
/*  a virtual pool is configured — inert in all other processes.        */
/* ------------------------------------------------------------------ */

int sysinfo(struct sysinfo *info)
{
    static int (*real_sysinfo)(struct sysinfo *) = NULL;
    if (!real_sysinfo)
        real_sysinfo = (int (*)(struct sysinfo *))dlsym(RTLD_NEXT, "sysinfo");

    if (!real_sysinfo)
        return -1;

    int ret = real_sysinfo(info);
    if (ret == 0 && initialized && gb_virtual_vram_bytes > 0 && info->mem_unit > 0) {
        unsigned long extra_pages = (unsigned long)(gb_virtual_vram_bytes / info->mem_unit);
        info->totalram += extra_pages;
        info->freeram  += extra_pages;
        if (gb_debug)
            fprintf(stderr,
                    "[GreenBoost] sysinfo: totalram inflated +%zu MB (virtual VRAM pool)\n",
                    gb_virtual_vram_bytes >> 20);
    }
    return ret;
}

/* ------------------------------------------------------------------ */
/*  cudaMemcpy / cudaMemcpyAsync hooks — TurboQuant transparent I/O   */
/*                                                                      */
/*  When a compressed KV buffer (GB_ALLOC_KV_COMPRESSED) is the src    */
/*  or dst of a memcpy, we transparently compress/decompress on the     */
/*  GPU using libgreenboost_tq.so.                                      */
/*                                                                      */
/*  Write path (dst = compressed KV buffer):                            */
/*    cudaMemcpy(kv_buf, src_fp16, ...) → gb_tq_quantize(src, kv_buf)  */
/*                                                                      */
/*  Read path (src = compressed KV buffer):                             */
/*    cudaMemcpy(dst_fp16, kv_buf, ...) → gb_tq_dequantize(kv_buf, dst)*/
/*                                                                      */
/*  Non-KV buffers pass through unchanged.                              */
/* ------------------------------------------------------------------ */

/* cudaMemcpyKind enum values — stable ABI since CUDA 1.0 */
typedef enum {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4,
} gb_cudaMemcpyKind;

typedef int (*pfn_cudaMemcpy_t)(void *, const void *, size_t, int);
typedef int (*pfn_cudaMemcpyAsync_t)(void *, const void *, size_t, int, void *);

/*
 * gb_tq_memcpy_hook — shared logic for cudaMemcpy and cudaMemcpyAsync.
 *
 * Checks if dst or src is a TQ-compressed KV buffer.  If so, routes through
 * the quantize/dequantize path.  Otherwise falls through to real_fn.
 *
 * Returns 0 (cudaSuccess) on success, non-zero on error.
 */
static int gb_tq_memcpy_hook(void *dst, const void *src, size_t count,
                              int kind, void *stream,
                              pfn_cudaMemcpy_t real_memcpy,
                              pfn_cudaMemcpyAsync_t real_memcpyasync)
{
    /* Fast path: TQ library not loaded — pass through */
    if (!g_libtq || !g_tq_quantize_fn || !g_tq_dequantize_fn)
        goto passthrough;

    /* Check if dst is a TQ-compressed KV buffer (write path) */
    {
        CUdeviceptr dst_dptr = (CUdeviceptr)(uintptr_t)dst;
        uint32_t    dst_flags = ht_peek_flags(dst_dptr);
        if (dst_flags & GB_ALLOC_KV_COMPRESSED) {
            size_t tq_sz = ht_get_tq_compressed_size(dst_dptr);
            if (tq_sz > 0) {
                /* Write path: compress fp16 src → compressed dst */
                gb_tq_config_t tq_snap;
                pthread_mutex_lock(&g_tq_config_mutex);
                tq_snap = g_tq_config;
                pthread_mutex_unlock(&g_tq_config_mutex);

                size_t n_elements = count / 2;  /* fp16: 2 bytes/element */
                int rc = g_tq_quantize_fn(src, dst, n_elements,
                                          tq_snap.head_dim,
                                          tq_snap.bits,
                                          stream);
                if (rc != 0) {
                    fprintf(stderr,
                            "[GreenBoost] TurboQuant quantize failed — falling back to direct copy\n");
                    goto passthrough;
                }
                gb_log("TurboQuant: cudaMemcpy write → quantize %zu MB",
                       count >> 20);
                return 0;  /* cudaSuccess */
            }
        }
    }

    /* Check if src is a TQ-compressed KV buffer (read path) */
    {
        CUdeviceptr src_dptr = (CUdeviceptr)(uintptr_t)src;
        uint32_t    src_flags = ht_peek_flags(src_dptr);
        if (src_flags & GB_ALLOC_KV_COMPRESSED) {
            size_t tq_sz = ht_get_tq_compressed_size(src_dptr);
            if (tq_sz > 0) {
                /* Read path: decompress compressed src → fp16 dst */
                gb_tq_config_t tq_snap;
                pthread_mutex_lock(&g_tq_config_mutex);
                tq_snap = g_tq_config;
                pthread_mutex_unlock(&g_tq_config_mutex);

                size_t n_elements = count / 2;  /* fp16: 2 bytes/element */
                int rc = g_tq_dequantize_fn(src, dst, n_elements,
                                             tq_snap.head_dim,
                                             tq_snap.bits,
                                             stream);
                if (rc != 0) {
                    fprintf(stderr,
                            "[GreenBoost] TurboQuant dequantize failed — falling back to direct copy\n");
                    goto passthrough;
                }
                gb_log("TurboQuant: cudaMemcpy read → dequantize %zu MB",
                       count >> 20);
                return 0;  /* cudaSuccess */
            }
        }
    }

passthrough:
    if (stream && real_memcpyasync)
        return real_memcpyasync(dst, src, count, kind, stream);
    if (real_memcpy)
        return real_memcpy(dst, src, count, kind);
    return 0;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind)
{
    static pfn_cudaMemcpy_t real_fn = NULL;
    if (!real_fn)
        real_fn = (pfn_cudaMemcpy_t)dlsym(RTLD_NEXT, "cudaMemcpy");

    if (!initialized || !real_fn)
        return real_fn ? real_fn(dst, src, count, kind) : 0;

    return (cudaError_t)gb_tq_memcpy_hook(dst, src, count, kind,
                                            NULL, real_fn, NULL);
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                             int kind, cudaStream_t stream)
{
    static pfn_cudaMemcpyAsync_t real_fn = NULL;
    if (!real_fn)
        real_fn = (pfn_cudaMemcpyAsync_t)dlsym(RTLD_NEXT, "cudaMemcpyAsync");

    if (!initialized || !real_fn)
        return real_fn ? real_fn(dst, src, count, kind, stream) : 0;

    return (cudaError_t)gb_tq_memcpy_hook(dst, src, count, kind,
                                            (void *)stream, NULL, real_fn);
}

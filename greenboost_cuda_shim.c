/*
 * GreenBoost v3.2 - CUDA LD_PRELOAD memory shim
 *
 * Routes CUDA VRAM overflow to system RAM via GPU-compute paths (tried in order):
 *
 *   Path A  - DMA-BUF pinned DDR (bare metal, requires greenboost.ko + /dev/greenboost)
 *              Internally tries two sub-methods in order:
 *              1. Zero-copy: GB_IOCTL_ALLOC → cudaImportExternalMemory(OpaqueFd)
 *                            → cudaExternalMemoryGetMappedBuffer → CUdeviceptr
 *                 CUDA drives its own IOMMU mapping from the kernel DMA-BUF SG table
 *                 (2 MB hugepages). No mmap round-trip or cuMemHostRegister overhead.
 *                 Requires libcudart.so (CUDA runtime ≥ 10.0). Skipped on CC ≥ 12
 *                 (Blackwell) - cudaImportExternalMemory(OpaqueFd) not supported there.
 *              2. Pinned: mmap → GB_IOCTL_PIN_USER_PTR → cuMemHostRegister(DEVICEMAP)
 *                 Used when libcudart.so is unavailable or sub-method 1 fails.
 *              Both sub-methods are GPU-DMA paths; the caller sees one unified "Path A".
 *
 *   Path B  - HostReg no-kernel (containers / VMs, auto-fallback)
 *              mmap (2 MB huge preferred) → cuMemHostRegister(DEVICEMAP)
 *              No kernel module required.  Works in Docker, LXC, KVM, WSL2.
 *              Set GREENBOOST_NO_HOSTREG=1 to skip.
 *              Concept: Jerry Nguyen (MR !3); hugepage + integration: Ferran Duarri.
 *
 *   Path C  - managed UVM (cuMemAllocManaged, CU_DEVICE_CPU preferred, GPU accessed-by
 *              hints) - Blackwell PCIe primary T2 path.  Pages stay in host DDR; GPU
 *              SMs access over PCIe (~20 GB/s).  No migration, no CPU compute.
 *              See CUDA UVM §4.1.4.2 and gb_vmm_t2_alloc_blackwell_managed().
 *
 * USAGE:
 *   LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so  ./your_cuda_app
 *
 * ENVIRONMENT VARIABLES:
 *   GREENBOOST_USE_DMA_BUF       1 = attempt Path A (default), 0 = skip to B
 *   GREENBOOST_NO_HOSTREG        1 = skip Path B (returns OOM when T2 exhausted)
 *   GREENBOOST_VRAM_HEADROOM_MB        keep ≥ this many MB free in VRAM (default 512)
 *   GREENBOOST_WORKSTATION_RESERVE_MB  MB of physical VRAM kept free for the desktop/
 *                                      display/other GPU processes (default 1024 MB).
 *                                      Subtracted from reported free VRAM so llama.cpp
 *                                      --fit doesn't fill T1 completely. Set to 0 on
 *                                      dedicated inference nodes. Doubled automatically
 *                                      when gaming_mode=1 (Proton game running).
 *   GREENBOOST_KV_RESERVE_MB           MB of T1 VRAM reserved for KV cache (default: from
 *                                      kernel module kv_reserve_mb param, typically 128).
 *                                Weights overflow to T2 sooner; KV cache stays in T1.
 *                                Adaptive: reserve collapses as KV fills T1 -
 *                                no double-counting with cuMemGetInfo free_vram.
 *   GREENBOOST_T1_WORKSPACE_MB   MB of T1 VRAM held back during MODEL_LOAD so
 *                                per-step compute workspace stays in T1 instead
 *                                of spilling to T2 (default 0=off; ~2560 is the
 *                                validated value for diffusion). Released at
 *                                INFERENCE. See g_workspace_reserve_bytes.
 *   GREENBOOST_KV_OVERFLOW       1 = all overflow allocs get GB_ALLOC_KV_CACHE|
 *                                GB_ALLOC_T1_PRIORITY - kernel freezes them in T2 LRU
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
 *   GREENBOOST_T2_POOL_MB        MB to pre-register as T2 pool (0=disable; default 85% of T2)
 *   GREENBOOST_A0_DISABLE        1 = skip Path A zero-copy sub-method (cudaImportExternalMemory); use pinned sub-method directly
 *   GREENBOOST_KV_COMPRESS       1 = absmax int8 compress K/V before T1→T2 eviction
 *                                (halves DMA bandwidth; default 0 - opt-in until validated)
 *   GREENBOOST_GDS               1 = use cuFile GPUDirect Storage for T3 NVMe path
 *                                (~7 GB/s vs ~1.8 GB/s CPU-bounce; requires libcufile.so.0)
 *   GREENBOOST_DOUBLE_BUFFER     1 = U18 double-buffer T3→T2 async staging: while GPU
 *                                consumes the current prefetched tile, speculatively
 *                                madvise the next contiguous tile (default 0 - opt-in
 *                                until benchmarked). Uses A4 BW-aware tile sizing.
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
#include <link.h>
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
#include "greenboost_ioctl.h"     /* gb_alloc_req, GB_IOCTL_ALLOC - userspace-safe */
#include "greenboost_netc.h"      /* remote cluster GPU client */
#include "features/net_fabric.h"  /* GB_ALLOC_TIER_* constants */
#include "features/tq_compress.h" /* K/V int8 compression metadata (Phase 2) */

/* Shim build version + capability-manifest ABI. Bump GB_SHIM_VERSION on any
 * change to the /run/greenboost/capabilities.json feature set; bump
 * GB_SHIM_CAP_ABI only on a breaking manifest schema change (consumers gate on
 * it via gb_monitor.capabilities()). */
#define GB_SHIM_VERSION "3.2"
#define GB_SHIM_CAP_ABI 1

/* ------------------------------------------------------------------ */
/*  NVTX instrumentation (optional - compile with USE_NVTX=1)          */
/* ------------------------------------------------------------------ */
#ifdef GREENBOOST_USE_NVTX
#include <nvToolsExt.h>
#define GB_NVTX_PUSH(name, color)  do { \
    nvtxEventAttributes_t _ea = {0}; \
    _ea.version = NVTX_VERSION; \
    _ea.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    _ea.colorType = NVTX_COLOR_ARGB; _ea.color = (color); \
    _ea.messageType = NVTX_MESSAGE_TYPE_ASCII; _ea.message.ascii = (name); \
    nvtxRangePushEx(&_ea); } while(0)
#define GB_NVTX_POP()  nvtxRangePop()
#else
#define GB_NVTX_PUSH(name, color)  ((void)0)
#define GB_NVTX_POP()              ((void)0)
#endif
#define GB_NVTX_COLOR_T1     0xFF0000FF
#define GB_NVTX_COLOR_T2     0xFFFFAA00
#define GB_NVTX_COLOR_T3     0xFFFF6600
#define GB_NVTX_COLOR_NET    0xFF00FF88
#define GB_NVTX_COLOR_PHASE  0xFF8844CC
#define GB_NVTX_COLOR_OOM    0xFFFF0000
#define GB_NVTX_COLOR_EVICT  0xFFFF8800
#define GB_NVTX_COLOR_KERN   0xFF00FF44
#define GB_NVTX_COLOR_MEMCPY 0xFF44AAFF

/* ------------------------------------------------------------------ */
/*  GB NVTX Persistent Event Log - always-on file-based diagnostics    */
/* ------------------------------------------------------------------ */
/* Writes structured text events to /run/greenboost/nvtx_events.log    */
/* using O_APPEND (atomic for writes < PIPE_BUF=4096 bytes on Linux).  */
/* Enabled unconditionally - overhead is one write() per significant   */
/* event, not per tensor op.  Rotates at 32 MB.                        */

#include <sys/stat.h>

#define GB_NVTX_LOG_FILE      "/run/greenboost/nvtx_events.log"
#define GB_NVTX_LOG_ROTATE    "/run/greenboost/nvtx_events.log.1"
#define GB_NVTX_LOG_FALLBACK  "/tmp/greenboost_nvtx_events.log"
#define GB_NVTX_LOG_MAX_BYTES (6ULL * 1024ULL * 1024ULL)  /* auto-clean at 6 MB - keeps the LLM-facing signal log small and fast to read */

/* Decision-log hot-path gate: gb_needs_overflow() runs on EVERY cudaMalloc,
 * including per-token activation buffers - logging all of those would add a
 * write() per allocation during steady-state generation. Only allocations at
 * or above this size (weight tensors, KV cache, the model-load buffer) are
 * worth a permanent record for diagnosing tier-placement bugs. */
#define GB_DECISION_LOG_MIN_BYTES (16ULL * 1024 * 1024)

/* PR-DD: g_nvtx_log_fd is _Atomic so the unsynchronised read in the
 * GB_NVTX_EVENT macro gate (PR-Q) sees a consistent value, and so writers
 * never observe a half-published fd after rotation.  The rotation path is
 * also serialised with g_nvtx_rotate_lock - concurrent writers go through
 * write() with the current fd; rotation acquires the lock, dup()s a new fd,
 * stores it atomically, then closes the OLD fd.  By the time the old fd
 * is closed, no further writer references it through g_nvtx_log_fd. */
static _Atomic int g_nvtx_log_fd = -1;  /* -1 = not init / open failed */
static char g_nvtx_log_path[256];       /* resolved at first write */
static pthread_mutex_t g_nvtx_rotate_lock = PTHREAD_MUTEX_INITIALIZER;

static pthread_once_t g_nvtx_log_once = PTHREAD_ONCE_INIT;
static void _gb_nvtx_log_open_once(void)
{
    mkdir("/run/greenboost", 0755);
    g_nvtx_log_fd = open(GB_NVTX_LOG_FILE,
                         O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (g_nvtx_log_fd >= 0) {
        snprintf(g_nvtx_log_path, sizeof(g_nvtx_log_path), "%s", GB_NVTX_LOG_FILE);
        return;
    }
    /* /run/greenboost not writable (ollama service user lacks permission); fall
     * back to /tmp so events are captured until tmpfiles.d fixes the boot perm. */
    g_nvtx_log_fd = open(GB_NVTX_LOG_FALLBACK,
                         O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (g_nvtx_log_fd >= 0)
        snprintf(g_nvtx_log_path, sizeof(g_nvtx_log_path), "%s", GB_NVTX_LOG_FALLBACK);
}

static void gb_nvtx_log_open(void)
{
    pthread_once(&g_nvtx_log_once, _gb_nvtx_log_open_once);
}

static void gb_nvtx_event_write(const char *type, const char *tier,
                                 size_t size_mb, uintptr_t ptr,
                                 const char *detail)
{
    gb_nvtx_log_open();
    /* PR-DD: load the fd atomically.  Concurrent rotation may publish a
     * new fd here; we use the value we observed for this write. */
    int fd = atomic_load_explicit(&g_nvtx_log_fd, memory_order_acquire);
    if (fd < 0) return;

    struct timespec _nvts; clock_gettime(CLOCK_REALTIME, &_nvts);
    uint64_t ms = (uint64_t)_nvts.tv_sec * 1000ULL + (uint64_t)_nvts.tv_nsec / 1000000ULL;

    char buf[256];
    int n = snprintf(buf, sizeof(buf),
        "%llu %-20s %-14s %6zuMB ptr=%016lx %s\n",
        (unsigned long long)ms, type, tier, size_mb, (unsigned long)ptr, detail);
    if (n > 0 && n < (int)sizeof(buf)) {
        ssize_t _wr = write(fd, buf, (size_t)n);
        if (_wr < 0 && errno == EBADF) {
            /* fd closed under us (rotation lost the race, or another
             * shim instance closed it).  Permanently disable until next
             * process restart.  Better than retrying against a stale fd
             * which may have been reassigned to another file (corruption). */
            atomic_store_explicit(&g_nvtx_log_fd, -1, memory_order_release);
            return;
        }
        (void)_wr;
    }

    /* Rotate at 32 MB - best-effort, checked on ~1/256 writes.
     * PR-DD: serialise the rotation through g_nvtx_rotate_lock.  Multiple
     * writers reaching this branch at the same time previously raced on
     * rename/close/open, leaving the fd in an indeterminate state.  Only
     * one thread enters rotation now; others fall through after trylock. */
    if ((ms & 0xFFULL) == 0ULL && pthread_mutex_trylock(&g_nvtx_rotate_lock) == 0) {
        /* Re-check size under the rotate lock to avoid double-rotation. */
        struct stat _st;
        int cur_fd = atomic_load_explicit(&g_nvtx_log_fd, memory_order_acquire);
        if (cur_fd >= 0 && fstat(cur_fd, &_st) == 0 &&
                (size_t)_st.st_size > GB_NVTX_LOG_MAX_BYTES) {
            char rotate_path[256];
            snprintf(rotate_path, sizeof(rotate_path), "%s.1", g_nvtx_log_path);
            rename(g_nvtx_log_path, rotate_path);
            /* Open NEW fd FIRST, publish atomically, THEN close the old.
             * This way concurrent writers either see the old fd (still
             * valid until close, append-mode writes go to the renamed
             * file - those events land in .1, acceptable) or the new fd
             * (already valid).  Never an invalid fd. */
            int new_fd = open(g_nvtx_log_path,
                O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
            if (new_fd >= 0) {
                atomic_store_explicit(&g_nvtx_log_fd, new_fd, memory_order_release);
                close(cur_fd);
            } else {
                /* open failed - disable logging rather than leave a bogus fd. */
                atomic_store_explicit(&g_nvtx_log_fd, -1, memory_order_release);
                close(cur_fd);
            }
        }
        pthread_mutex_unlock(&g_nvtx_rotate_lock);
    }
}

/* GB_NVTX_EVENT(type, tier, size_mb, ptr, detail)
 *
 * PR-Q/F-S12: hot-path gate.  When the telemetry file failed to open
 * (g_nvtx_log_fd == -1) the function-call entry alone would cost a
 * pthread_once atomic-read + parameter passing per event.  On
 * cuLaunchKernel and cuLaunchKernelEx that's per-kernel-launch (hundreds
 * per token).  Gate at the call site instead so the disabled-mode cost
 * is one predicted branch.  gb_shim_init proactively opens the file so
 * the fd is settled before any user hook runs; an unsynchronised int
 * read is fine because the post-init value is stable. */
#define GB_NVTX_EVENT(type, tier, size_mb, ptr, detail) \
    do { if (__builtin_expect(g_nvtx_log_fd >= 0, 1)) \
            gb_nvtx_event_write((type), (tier), (size_mb), (uintptr_t)(ptr), (detail)); \
       } while (0)

/* ------------------------------------------------------------------ */
/*  Minimal CUDA type definitions (no CUDA SDK headers needed)         */
/* ------------------------------------------------------------------ */

typedef unsigned long long CUdeviceptr;
typedef int                CUresult;
typedef int                cudaError_t;
typedef unsigned int       CUmemAttach_flags;
typedef int                CUdevice;
typedef struct CUstream_st   *CUstream;
typedef struct CUfunc_st     *CUfunction;
typedef struct CUmod_st      *CUmodule;   /* E1: PTX module handle */
typedef CUstream            cudaStream_t;

#define CUDA_SUCCESS                0
#define CUDA_ERROR_INVALID_DEVICE   101
#define CUDA_ERROR_INVALID_VALUE    1
#define CUDA_ERROR_UNKNOWN          999
#define CUDA_ERROR_NOT_SUPPORTED    801
#define CUDA_ERROR_OUT_OF_MEMORY    2
#define CUDA_ERROR_NOT_INITIALIZED  3
#define CU_MEM_ATTACH_GLOBAL        0x1u
#define CU_MEM_ATTACH_HOST          0x2u
/* REF-04: Named constants for magic numbers used in compute-capability probing
 * and NVML error returns. Values are stable ABI since CUDA 3 / NVML 1.0. */
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR  75
#define NVML_SUCCESS                                   0
#define NVML_ERROR_INVALID_ARGUMENT                    2
#define NVML_ERROR_FUNCTION_NOT_FOUND                999

/* Fake NVML handle sentinel for cluster remote GPUs.
 * nvmlDeviceGetHandleByIndex(local_count + ri) returns (void*)(BASE + ri).
 * All nvmlDevice* hooks check gb_is_fake_nvml() first and route to feeder.
 *
 * Audit F-L1-23: raise the fake-handle range from 16 to 64 so a cluster
 * with up to 64 feeder GPUs cannot collision into invalid handle space. */
#define GB_FAKE_NVML_BASE  ((uintptr_t)0xBBBB0000u)
#define GB_FAKE_NVML_COUNT ((uintptr_t)64u)
#define gb_is_fake_nvml(dev) \
    ((uintptr_t)(dev) >= GB_FAKE_NVML_BASE && \
     (uintptr_t)(dev) < GB_FAKE_NVML_BASE + GB_FAKE_NVML_COUNT)
#define gb_fake_nvml_idx(dev) ((int)((uintptr_t)(dev) - GB_FAKE_NVML_BASE))

/* cuMemAdvise constants (stable since CUDA 8 - no SDK headers needed) */
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
#define CU_MEM_LOCATION_TYPE_DEVICE           1
#define CU_MEM_LOCATION_TYPE_HOST             2
#define CU_MEM_LOCATION_TYPE_HOST_NUMA        3  /* CUDA 12.2+ */
#define CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT 4 /* CUDA 12.2+; id ignored; SM-accessible on Blackwell */

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

/* CUmemAccess_flags - argument to cuMemSetAccess (CUDA 10.2+, stable ABI) */
typedef int CUmemAccess_flags;
#define CU_MEM_ACCESS_FLAGS_PROT_NONE       0
#define CU_MEM_ACCESS_FLAGS_PROT_READ       1
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE  3

typedef struct {
    CUmemLocation     location;
    CUmemAccess_flags flags;
} CUmemAccessDesc;

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
/*  Portable atoll - avoids __isoc23_strtoll@GLIBC_2.38                */
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
     * An env var value that would overflow long long is nonsensical - clamp. */
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
/*  Open-addressed hash map - replaces alloc_table[65536]              */
/*  131072 slots × 64 bytes = 8 MB, aligned for cache-line access      */
/* ------------------------------------------------------------------ */

static int    gb_debug              = 0;

#define gb_log(fmt, ...) \
    do { if (gb_debug) fprintf(stderr, "[GreenBoost] " fmt "\n", ##__VA_ARGS__); } while (0)

/* Subprocess-aware "log this init message exactly once across the entire
 * fork tree".  When LD_PRELOAD propagates through fork()/exec(), every
 * child re-runs __attribute__((constructor)) and would re-print every
 * config-acknowledgement line - easily 5x duplication for a single
 * PyTorch generation (parent + 4 dataloader workers).  We claim ownership
 * via a sticky env var set in the FIRST init; children that see it stay
 * quiet.  Used only for purely-informational config acks; real warnings
 * and errors still print from every process so failures stay visible. */
static int gb_init_log_owner = -1;
static int gb_init_log_should(void) {
    if (gb_init_log_owner != -1) return gb_init_log_owner;
    if (getenv("__GB_INIT_LOG_OWNER")) { gb_init_log_owner = 0; return 0; }
    setenv("__GB_INIT_LOG_OWNER", "1", 1);
    gb_init_log_owner = 1;
    return 1;
}
#define GB_INIT_LOG_ONCE(...) \
    do { if (gb_init_log_should()) fprintf(stderr, __VA_ARGS__); } while (0)

#define HT_BITS   17u
#define HT_SIZE   (1u << HT_BITS)
#define HT_MASK   (HT_SIZE - 1u)
#define HT_LOCKS  64u

/* ------------------------------------------------------------------ */
/*  Async Prefetching Queue & Worker                                    */
/* ------------------------------------------------------------------ */

#define PREFETCH_QUEUE_SIZE 256

/* U18: Each queue slot carries the chunk to madvise plus the bounds of the
 * containing mmap region so the double-buffer lookahead can be safely clamped. */
typedef struct {
    void  *mapped_ptr;       /* base of the chunk to madvise */
    size_t size;             /* bytes to madvise for this chunk */
    void  *alloc_base;       /* U18: start of the full mmap region */
    size_t alloc_total_size; /* U18: total size of the full mmap region */
} prefetch_req_t;

static prefetch_req_t prefetch_queue[PREFETCH_QUEUE_SIZE];
static int prefetch_head = 0;
static int prefetch_tail = 0;
static pthread_mutex_t prefetch_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t prefetch_cond = PTHREAD_COND_INITIALIZER;
static pthread_t prefetch_thread;
static volatile int prefetch_stop = 0;
/* CRIT-06: Track whether prefetch_thread was actually started so gb_shim_fini
 * can join it unconditionally - independent of whether 'initialized' was set. */
static int prefetch_initialized = 0;

/* U18/A4: Double-buffer and BW-aware prefetch configuration */
static int g_double_buffer_enabled = 0;  /* set from GREENBOOST_DOUBLE_BUFFER=1 */

/* Phase 2b: async feeder kernel dispatch.  Set from GREENBOOST_ASYNC_DISPATCH=1.
 * When enabled, cuLaunchKernel for pure reloc kernels (no inline data) uses
 * GB_MSG_CUDA_EXEC_ASYNC so the host does not block per kernel launch.
 * Sync happens at cudaStreamSynchronize / cudaDeviceSynchronize boundaries. */
static int g_async_dispatch = 0;

/* A4: Compute BW-aware prefetch tile size: clamp to [64MB, 256MB] targeting ~100ms
 * of link time using the EWMA-measured feeder bandwidth (U20). Falls back to 64MB
 * when no feeder is connected or no bandwidth measurement is available yet. */
static size_t gb_prefetch_tile_bytes(void)
{
    uint32_t bw = gb_netc_best_feeder_bw_mbs(); /* MiB/s from U20 EWMA */
    if (bw == 0) return 64UL << 20;
    /* 100 ms of data at bw MiB/s */
    size_t tile = (size_t)bw * (1UL << 20) / 10;
    if (tile < 64UL << 20)  tile = 64UL << 20;
    if (tile > 256UL << 20) tile = 256UL << 20;
    return tile;
}

/* Forward declarations - defined after phase-detector globals and gb_now_ms */
static void gb_htable_flush(int kv_only);
static void gb_check_idle_phase(void);
/* U5: forward-declare so prefetch_worker can reference before main definition.
 * gb_t2_overflow_bytes is a tentative static definition - C99 allows the
 * duplicate; both declarations merge to the same object. */
static _Atomic size_t gb_t2_overflow_bytes;
static inline size_t gb_effective_t2_warn(void);
/* U9/U10: forward-declared because ht_set_flags and gb_htable_flush use them before defs */
static uint64_t gb_xxhash64(const void *data, size_t len);
static int  gb_kv_cache_insert(const void *host_ptr, size_t size,
                                uint64_t dev_ptr, uint64_t *out_ptr,
                                uint64_t parent_hash, uint32_t num_tokens);
static void gb_kv_cache_release(uint64_t dev_ptr);
static void gb_block_evt_emit(uint8_t type, uint64_t hash,
                               uint64_t parent_hash, uint32_t num_tokens);

/* U-HEAT: defined later (after the hashtable section it walks: gb_htable,
 * ht_hash, ht_lock, HT_TOMBSTONE) and after gb_open_device. Forward-declared
 * here so prefetch_worker - defined ahead of all of that in the file - can
 * call it. */
static void gb_push_heat_batch(void);
static void gb_kv_prefetch_tick(void);

static void* prefetch_worker(void* arg) {
    /* U-HEAT: independent 2s cadence for the heat pusher above, checked on
     * every loop wakeup (real request or the 5s idle timeout) so it runs
     * whether the prefetch queue is busy or empty. */
    struct timespec last_heat_push = {0};

    while (!prefetch_stop) {
        prefetch_req_t req;
        struct timespec ts;

        {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            if (now.tv_sec - last_heat_push.tv_sec >= 2) {
                gb_push_heat_batch();
                gb_kv_prefetch_tick();   /* no-op unless GREENBOOST_KV_PREFETCH set */
                last_heat_push = now;
            }
        }

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

        /* U5: DeepSpeed-style T3→T2 prefetch admission control.
         * Prefetch only when T2 has headroom (below WARN_PCT); otherwise
         * re-queue the request for the next wakeup to avoid pushing T2 past CAP. */
        {
            size_t t2_used = atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed);
            size_t t2_warn = gb_effective_t2_warn();
            if (t2_warn > 0 && t2_used >= t2_warn) {
                /* T2 under pressure - re-queue and back off */
                pthread_mutex_lock(&prefetch_mutex);
                int next = (prefetch_head + 1) % PREFETCH_QUEUE_SIZE;
                if (next != prefetch_tail) {   /* space available */
                    prefetch_queue[prefetch_head] = req;
                    prefetch_head = next;
                }
                pthread_mutex_unlock(&prefetch_mutex);
                usleep(50000);  /* 50 ms back-off */
                continue;
            }
        }

        /* Hint the kernel to bring the pages into RAM */
        GB_NVTX_PUSH("GB:T3_prefetch", GB_NVTX_COLOR_T3);
        madvise(req.mapped_ptr, req.size, MADV_WILLNEED);
        GB_NVTX_POP();
        if (gb_debug) fprintf(stderr, "[GreenBoost] prefetch thread: madvise(MADV_WILLNEED) on %p size=%zu\n", req.mapped_ptr, req.size);

        /* U18: Double-buffer lookahead - while GPU consumes the current tile,
         * speculatively madvise the next contiguous tile within the same mmap region
         * so T3→T2 DMA for chunk N+1 overlaps with GPU consumption of chunk N. */
        if (g_double_buffer_enabled && req.alloc_base && req.alloc_total_size > 0) {
            char *next_start = (char *)req.mapped_ptr + req.size;
            char *alloc_end  = (char *)req.alloc_base + req.alloc_total_size;
            if (next_start < alloc_end) {
                size_t tile = gb_prefetch_tile_bytes();
                size_t avail = (size_t)(alloc_end - next_start);
                size_t lookahead = (avail < tile) ? avail : tile;
                GB_NVTX_PUSH("GB:T3_prefetch_lookahead", GB_NVTX_COLOR_T3);
                madvise(next_start, lookahead, MADV_WILLNEED);
                GB_NVTX_POP();
                if (gb_debug)
                    fprintf(stderr, "[GreenBoost] prefetch: double-buffer lookahead "
                            "madvise(%p, %.1f MB)\n",
                            next_start, (double)lookahead / (1 << 20));
            }
        }
    }
    return NULL;
}

/* U18: alloc_base/alloc_total_size provide bounds for the double-buffer lookahead.
 * Pass NULL/0 when those values are not available (lookahead is skipped). */
static void enqueue_prefetch(void *mapped_ptr, size_t size,
                             void *alloc_base, size_t alloc_total_size) {
    if (!mapped_ptr) return;
    pthread_mutex_lock(&prefetch_mutex);
    int next_head = (prefetch_head + 1) % PREFETCH_QUEUE_SIZE;
    if (next_head != prefetch_tail) {
        prefetch_queue[prefetch_head].mapped_ptr       = mapped_ptr;
        prefetch_queue[prefetch_head].size             = size;
        prefetch_queue[prefetch_head].alloc_base       = alloc_base;
        prefetch_queue[prefetch_head].alloc_total_size = alloc_total_size;
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
    CUdeviceptr           ptr;          /* 8 B  - 0 = empty, 1 = tombstone */
    size_t                size;         /* 8 B                            */
    int                   is_managed;   /* 4 B  - 1 = UVM, 0 = device    */
    int                   gb_buf_id;    /* 4 B  - -1 if not DMA-BUF      */
    void                 *mapped_ptr;   /* 8 B  - user-space mmap ptr    */
    int                   fd;           /* 4 B  - DMA-BUF fd             */
    cudaExternalMemory_t  ext_mem;      /* 8 B  - Path A zero-copy handle, NULL otherwise */
    uint32_t              alloc_flags;  /* 4 B  - GB_ALLOC_* flags (KV/weights/etc) */
    /* U1: ARC tracking - access_ts in seconds (CLOCK_MONOTONIC), access_count for T1/T2 */
    uint32_t              access_ts;    /* 4 B  - last access timestamp (seconds) */
    uint16_t              access_count; /* 2 B  - times this entry was touched     */
    uint8_t               _pad[14];    /* pad to maintain alignment               */
} __attribute__((aligned(64))) gb_ht_entry_t;

static gb_ht_entry_t      gb_htable[HT_SIZE];
static pthread_mutex_t    ht_locks[HT_LOCKS];

/* ------------------------------------------------------------------ */
/*  VMM handle table - tracks host-backed cuMemCreate handles          */
/*                                                                      */
/*  When cuMemCreate falls back to CU_MEM_LOCATION_TYPE_HOST (T2),     */
/*  we record the handle here so cuMemRelease can decrement the T2     */
/*  accounting counter.  This is critical for PyTorch expandable       */
/*  segments: if the subsequent cuMemMap fails (device-backed range    */
/*  cannot mix with host-backed handle), PyTorch calls cuMemRelease    */
/*  to clean up.  Without this hook, gb_t2_overflow_bytes drifts up   */
/*  and eventually blocks future T2 allocations even though the memory  */
/*  was never actually used.                                            */
/*                                                                      */
/*  Table is small: at most O(segments) handles exist at one time      */
/*  (typically <256 for any single model load).  Single mutex is fine. */
/* ------------------------------------------------------------------ */

#define VMM_HT_BITS  10u
#define VMM_HT_SIZE  (1u << VMM_HT_BITS)
#define VMM_HT_MASK  (VMM_HT_SIZE - 1u)

typedef struct {
    CUmemGenericAllocationHandle handle; /* 0 = empty, UINT64_MAX = deleted */
    size_t                       size;
} vmm_ht_entry_t;

#define VMM_HT_EMPTY   ((CUmemGenericAllocationHandle)0ULL)
#define VMM_HT_DELETED ((CUmemGenericAllocationHandle)UINT64_MAX)

static vmm_ht_entry_t  gb_vmm_ht[VMM_HT_SIZE];
static pthread_mutex_t gb_vmm_ht_lock = PTHREAD_MUTEX_INITIALIZER;

static inline uint32_t vmm_ht_hash(CUmemGenericAllocationHandle h)
{
    return (uint32_t)((h * 0x9E3779B97F4A7C15ULL) >> (64 - VMM_HT_BITS));
}

/* Insert a host-backed handle. No-op if table is full (very unlikely). */
static void vmm_ht_insert(CUmemGenericAllocationHandle h, size_t sz)
{
    uint32_t i, slot;
    pthread_mutex_lock(&gb_vmm_ht_lock);
    slot = vmm_ht_hash(h) & VMM_HT_MASK;
    for (i = 0; i < VMM_HT_SIZE; i++) {
        vmm_ht_entry_t *e = &gb_vmm_ht[(slot + i) & VMM_HT_MASK];
        if (e->handle == VMM_HT_EMPTY || e->handle == VMM_HT_DELETED) {
            e->handle = h;
            e->size   = sz;
            pthread_mutex_unlock(&gb_vmm_ht_lock);
            return;
        }
    }
    pthread_mutex_unlock(&gb_vmm_ht_lock);
    fprintf(stderr, "[GreenBoost] vmm_ht_insert: table full - T2 accounting may drift\n");
}

/* Remove handle and return its size (0 if not found / not host-backed). */
static size_t vmm_ht_remove(CUmemGenericAllocationHandle h)
{
    uint32_t i, slot;
    size_t sz = 0;
    if (h == VMM_HT_EMPTY || h == VMM_HT_DELETED) return 0;
    pthread_mutex_lock(&gb_vmm_ht_lock);
    slot = vmm_ht_hash(h) & VMM_HT_MASK;
    for (i = 0; i < VMM_HT_SIZE; i++) {
        vmm_ht_entry_t *e = &gb_vmm_ht[(slot + i) & VMM_HT_MASK];
        if (e->handle == VMM_HT_EMPTY) break; /* end of probe chain */
        if (e->handle == h) {
            sz = e->size;
            e->handle = VMM_HT_DELETED; /* tombstone */
            e->size   = 0;
            break;
        }
    }
    pthread_mutex_unlock(&gb_vmm_ht_lock);
    return sz;
}

/* ------------------------------------------------------------------ */
/*  T2 Pre-Registered Pool                                              */
/*  Pre-registers the full T2 DDR slab once; sub-allocates by pointer  */
/*  arithmetic - eliminates per-allocation cuMemHostRegister overhead.  */
/*                                                                      */
/*  GREENBOOST_T2_POOL_MB  - explicit pool size in MB (0 = disabled).  */
/*  Default: 85 % of gb_t2_pool_bytes (set lazily after sysfs read).   */
/*                                                                      */
/*  Init is deferred to the first T2 overflow call so that             */
/*  cuMemHostRegister is guaranteed available (libcuda.so already       */
/*  resolved).  pthread_once ensures exactly-one init across threads.   */
/* ------------------------------------------------------------------ */

/* U2: size-stratified free lists (Ray object store pattern).
 * Four size classes reduce O(n) best-fit scan to O(n/4) per class and
 * eliminate cross-class fragmentation for the common small-tensor case.
 * Class boundaries: S≤4KB, M≤64KB, L≤4MB, H>4MB.
 * Each class has its own sorted free list; gb_pool_alloc spills to the
 * next larger class when the target class has no fitting segment. */
#define GB_POOL_SC_SMALL   256    /* ≤ 4 KB - KV metadata, tiny tensors */
#define GB_POOL_SC_MEDIUM  512    /* ≤ 64 KB */
#define GB_POOL_SC_LARGE   1024   /* ≤ 4 MB  */
#define GB_POOL_SC_HUGE    2304   /* > 4 MB  - model weight blocks */
#define GB_POOL_MAX_FREE_SEGS (GB_POOL_SC_SMALL + GB_POOL_SC_MEDIUM + \
                                GB_POOL_SC_LARGE + GB_POOL_SC_HUGE)

#define GB_POOL_THRESH_S  (4UL   * 1024)
#define GB_POOL_THRESH_M  (64UL  * 1024)
#define GB_POOL_THRESH_L  (4UL   * 1024 * 1024)

/* U14: Adaptive block size tiers (vLLM hybrid block mode pattern).
 * KV allocs are aligned to GPU page granularity so sub-blocks share a
 * page boundary.  Allocator uses 256-token blocks; kernel sub-blocks = 16. */
#define GB_KV_MIN_BLOCK_ALIGN     4096UL   /* one GPU page */
#define GB_KV_ALLOC_BLOCK_TOKENS   256     /* allocator granularity */
#define GB_KV_KERNEL_BLOCK_TOKENS   16     /* attention kernel sub-block */

/* Global counters for U14 fragmentation metrics */
static _Atomic size_t g_kv_internal_frag_bytes = 0;

typedef struct { size_t off; size_t sz; } gb_pool_seg_t;

/* Returns size-class index 0..3 for a given aligned byte count. */
static inline int gb_pool_sc(size_t sz)
{
    if (sz <= GB_POOL_THRESH_S) return 0;
    if (sz <= GB_POOL_THRESH_M) return 1;
    if (sz <= GB_POOL_THRESH_L) return 2;
    return 3;
}
static const int gb_pool_sc_max[4] = {
    GB_POOL_SC_SMALL, GB_POOL_SC_MEDIUM, GB_POOL_SC_LARGE, GB_POOL_SC_HUGE
};

typedef struct {
    void           *base;        /* mmap base of pool                      */
    size_t          total;       /* total pool size in bytes               */
    CUdeviceptr     dev_base;    /* GPU ptr from cuMemHostGetDevicePointer  */
    int             dmabuf_fd;   /* kernel DMA-BUF fd (-1 if kmod absent)  */
    /* U2: four size-class free lists, each sorted by offset */
    gb_pool_seg_t   sc[4][GB_POOL_SC_HUGE];  /* sc[0]=small … sc[3]=huge */
    int             nfree[4];
    pthread_mutex_t lock;
    int             initialized; /* 1 after successful init                */
    int             init_failed; /* 1 if init was attempted and failed     */
} gb_pool_t;

static gb_pool_t      gb_t2_reg_pool;
static pthread_once_t gb_pool_once = PTHREAD_ONCE_INIT;
static size_t         gb_pool_configured_bytes = 0;

/* gb_pool_init / gb_pool_alloc / gb_pool_free / gb_pool_contains are
 * defined after the real_* function pointer declarations (below) so they
 * can reference real_cuMemHostRegister etc. without forward declarations. */

/* ------------------------------------------------------------------ */

static inline uint32_t ht_hash(CUdeviceptr ptr)
{
    /* Fibonacci hash - good distribution for pointer-sized keys */
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
            e->ptr          = ptr;
            e->size         = size;
            e->is_managed   = is_managed;
            e->gb_buf_id    = gb_buf_id;
            e->mapped_ptr   = mapped_ptr;
            e->fd           = fd;
            e->ext_mem      = ext_mem;
            e->access_ts    = (uint32_t)time(NULL);
            e->access_count = 0;
            pthread_mutex_unlock(lk);
            return 1;
        }
        pthread_mutex_unlock(lk);
    }
    fprintf(stderr, "[GreenBoost] CRITICAL: hash table full at %u entries - allocation will leak\n",
            HT_SIZE);
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
            e->ptr          = HT_TOMBSTONE;
            e->size         = 0;
            e->is_managed   = 0;
            e->gb_buf_id    = -1;
            e->mapped_ptr   = NULL;
            e->fd           = -1;
            e->ext_mem      = NULL;
            e->access_ts    = 0;
            e->access_count = 0;
            pthread_mutex_unlock(lk);
            return 1;
        }
        pthread_mutex_unlock(lk);
        if (slot_ptr == 0)
            break; /* genuinely empty - key not present */
        /* slot_ptr == HT_TOMBSTONE: deleted slot, keep probing */
    }
    return 0;
}

/* Non-destructive lookup - same probe loop as ht_remove but no tombstone write.
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
            /* U1: update ARC access tracking on every lookup */
            e->access_ts = (uint32_t)time(NULL);
            if (e->access_count < 0xFFFF) e->access_count++;
            pthread_mutex_unlock(lk);
            return 1;
        }
        pthread_mutex_unlock(lk);
        if (slot_ptr == 0)
            break; /* genuinely empty - key not present */
    }
    return 0;
}

/* Forward-declared: defined later in the "/dev/greenboost helper" section. */
static int gb_open_device(void);

/* U-HEAT: push the live ARC-tracking heat (gb_ht_entry_t.access_count) to
 * the kernel evictor via GB_IOCTL_SET_HEAT so gb_try_evict_for_alloc /
 * gb_auto_evict_cold in greenboost.c can prefer recency+reuse over pure
 * LRU.  Reports the *delta* since the last push (resets access_count to 0
 * under the same per-slot ht_lock that ht_lookup uses to increment it - the
 * slot index itself is the lock key, matching ht_lookup's
 * ht_lock((h+i) & HT_MASK) - so this never races a concurrent touch).  The
 * kernel side ORs the delta into its own buf->heat and ages it
 * independently on the watchdog. */
static void gb_push_heat_batch(void)
{
    int fd = gb_open_device();
    struct gb_heat_req req;
    uint32_t i;

    if (fd < 0)
        return;

    req.count = 0;
    req._pad  = 0;

    for (i = 0; i < HT_SIZE; i++) {
        gb_ht_entry_t   *e  = &gb_htable[i];
        pthread_mutex_t *lk = ht_lock(i);

        if (e->ptr == 0 || e->ptr == HT_TOMBSTONE || e->gb_buf_id < 0)
            continue;

        pthread_mutex_lock(lk);
        if (e->ptr != 0 && e->ptr != HT_TOMBSTONE && e->gb_buf_id >= 0 &&
            e->access_count > 0) {
            req.ent[req.count].buf_id = e->gb_buf_id;
            req.ent[req.count].heat   = e->access_count;
            e->access_count = 0;
            req.count++;
        }
        pthread_mutex_unlock(lk);

        if (req.count == GB_HEAT_BATCH_MAX) {
            ioctl(fd, GB_IOCTL_SET_HEAT, &req);
            req.count = 0;
        }
    }
    if (req.count > 0)
        ioctl(fd, GB_IOCTL_SET_HEAT, &req);
}

/* Set alloc_flags on an already-inserted entry (used by overflow path to tag
 * KV/weights/activations after the alloc succeeds). Non-racy: called immediately
 * after ht_insert in the same thread, before the pointer escapes. */
static void ht_set_flags(CUdeviceptr ptr, uint32_t flags)
{
    uint32_t h = ht_hash(ptr);
    uint32_t i;
    for (i = 0; i < HT_SIZE; i++) {
        uint32_t slot = (h + i) & HT_MASK;
        gb_ht_entry_t *e = &gb_htable[slot];
        pthread_mutex_t *lk = ht_lock(slot);
        CUdeviceptr slot_ptr;
        pthread_mutex_lock(lk);
        slot_ptr = e->ptr;
        if (slot_ptr == ptr) {
            e->alloc_flags = flags;
            /* U9: register new KV block in prefix cache when flagged as KV.
             * U10: emit BLOCK_STORED event so feeders can build ancestry tree. */
            if ((flags & GB_ALLOC_KV_CACHE) && e->mapped_ptr) {
                uint64_t dummy_out = 0;
                uint32_t approx_tokens = (uint32_t)(e->size / 128);
                uint64_t hash = gb_xxhash64(e->mapped_ptr,
                                            (e->size < 256) ? e->size : 256);
                /* U9: parent_hash=0 for root; callers that know ancestry can
                 * set the flag and pass via alloc metadata in future revisions. */
                gb_kv_cache_insert(e->mapped_ptr, e->size, (uint64_t)ptr, &dummy_out,
                                   0 /* parent_hash */, approx_tokens);
                gb_block_evt_emit(GB_BLOCK_EVT_STORED, hash, 0 /* parent_hash */,
                                  approx_tokens);
            }
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
/* PR-N/F-S7: CUmemoryPool is an opaque handle (CUmemPoolHandle_st *).  We
 * don't dereference it - just pass it through.  Use void * to avoid pulling
 * in <cuda.h>. */
typedef void *      CUmemoryPool_handle;
typedef CUresult    (*pfn_cuMemAllocFromPoolAsync)(CUdeviceptr *, size_t,
                                                   CUmemoryPool_handle, CUstream);
typedef CUresult    (*pfn_cuDeviceGetAttribute)(int *, int, CUdevice);
typedef cudaError_t (*pfn_cudaMalloc)(void **, size_t);
typedef cudaError_t (*pfn_cudaFree)(void *);
typedef cudaError_t (*pfn_cudaMallocManaged)(void **, size_t, unsigned int);
typedef cudaError_t (*pfn_cudaMallocAsync)(void **, size_t, cudaStream_t);
typedef cudaError_t (*pfn_cudaFreeAsync)(void *, cudaStream_t);
typedef cudaError_t (*pfn_cudaImportExternalMemory)(cudaExternalMemory_t *,
                                                    const struct cudaExternalMemoryHandleDesc *);
typedef cudaError_t (*pfn_cudaExternalMemoryGetMappedBuffer)(void **, cudaExternalMemory_t,
                                                             const struct cudaExternalMemoryBufferDesc *);
typedef cudaError_t (*pfn_cudaDestroyExternalMemory)(cudaExternalMemory_t);
typedef cudaError_t (*pfn_cudaGetLastError)(void);
typedef cudaError_t (*pfn_cudaDeviceSynchronize)(void);
typedef cudaError_t (*pfn_cudaMemGetInfo)(size_t *, size_t *);
typedef CUresult    (*pfn_cuDeviceTotalMem_v2)(size_t *, CUdevice);
typedef CUresult    (*pfn_cuDeviceGetCount)(int *);
typedef CUresult    (*pfn_cuDeviceGet)(CUdevice *, int);
typedef CUresult    (*pfn_cuGetProcAddress)(const char *symbol, void **pfn, int driverVersion, unsigned long long flags, void *symbolStatus);
typedef cudaError_t (*pfn_cudaGetDeviceCount)(int *);
typedef cudaError_t (*pfn_cudaSetDevice)(int);

/* Minimal cudaPointerAttributes - avoids a CUDA SDK header dependency, matches
 * driver_types.h's stable layout (type/device/devicePointer/hostPointer). */
typedef enum { cudaMemoryTypeUnregistered = 0, cudaMemoryTypeHost = 1,
               cudaMemoryTypeDevice = 2, cudaMemoryTypeManaged = 3 } gb_cudaMemoryType;
typedef struct {
    gb_cudaMemoryType type;
    int   device;
    void *devicePointer;
    void *hostPointer;
} gb_cudaPointerAttributes;
typedef cudaError_t (*pfn_cudaDeviceCanAccessPeer)(int *, int, int);
typedef cudaError_t (*pfn_cudaDeviceEnablePeerAccess)(int, unsigned int);
typedef cudaError_t (*pfn_cudaPointerGetAttributes)(gb_cudaPointerAttributes *, const void *);
typedef CUresult    (*pfn_cuLaunchKernel)(CUfunction,
                                          unsigned int, unsigned int, unsigned int,
                                          unsigned int, unsigned int, unsigned int,
                                          unsigned int, CUstream, void **, void **);
typedef CUresult    (*pfn_cuLaunchCooperativeKernel)(CUfunction,
                                                     unsigned int, unsigned int, unsigned int,
                                                     unsigned int, unsigned int, unsigned int,
                                                     unsigned int, CUstream, void **);

/* PR-NN: cuFuncGetParamInfo (CUDA 12.3+ driver API) returns each kernel
 * parameter's offset + size.  Iterating paramIndex 0..N until the call
 * returns CUDA_ERROR_INVALID_VALUE yields the exact parameter count -
 * which is what gb_kernel_sigs needs to bound the kernelParams[] scan.
 *
 * Resolved lazily via dlsym; older drivers return NULL and we fall back
 * to scan-until-NULL behaviour (current default).  No-op on cc < 12.3
 * deployments. */
typedef CUresult    (*pfn_cuFuncGetParamInfo)(CUfunction, size_t,
                                              size_t *, size_t *);

/* PR-O/F-S8: CUlaunchConfig as used by cuLaunchKernelEx (CUDA 12+,
 * PyTorch 2.4+ graph mode).  We don't read .attrs[] - we just pass it
 * through to the real driver on the local path.  Fields are stable since
 * CUDA 12.0; future versions may extend with trailing fields which we
 * forward unchanged because we pass the pointer through, not by value. */
typedef struct {
    unsigned int   gridDimX;
    unsigned int   gridDimY;
    unsigned int   gridDimZ;
    unsigned int   blockDimX;
    unsigned int   blockDimY;
    unsigned int   blockDimZ;
    unsigned int   sharedMemBytes;
    CUstream       hStream;
    void          *attrs;     /* CUlaunchAttribute * - opaque to us */
    unsigned int   numAttrs;
} gb_CUlaunchConfig;

typedef CUresult    (*pfn_cuLaunchKernelEx)(const gb_CUlaunchConfig *,
                                            CUfunction, void **, void **);

/* cudaLaunchKernel / cudaStreamSynchronize (runtime API) */
typedef struct { unsigned x, y, z; } gb_dim3;
typedef cudaError_t (*pfn_cudaLaunchKernel)(const void *, gb_dim3, gb_dim3,
                                             void **, size_t, cudaStream_t);

/* PR-CLKExC take 2 (2026-07-06): the 2026-06-24 attempt crashed because its
 * passthrough resolved the real function via dlsym(RTLD_NEXT) — the SYSTEM
 * libcudart — while ggml cuda_v13 runs its own libcudart instance (F-ABI1).
 * Every call was serviced by the wrong runtime → "invalid argument" on the
 * first launch. Identical failure mode hit (and fixed) today for the
 * cudaMemset hook. This hook resolves via the REBOUND instance
 * (real_cudaLaunchKernelExC ← gb_cudart_sym) and adds the same data-driven
 * feeder dispatch as cudaLaunchKernel — the gating piece for feeder GPU
 * compute with cuda_v13 ggml. */
typedef struct {
    gb_dim3      gridDim;
    gb_dim3      blockDim;
    size_t       dynamicSmemBytes;
    cudaStream_t stream;
    void        *attrs;      /* cudaLaunchAttribute * - opaque passthrough */
    unsigned int numAttrs;
} gb_cudaLaunchConfig;
typedef cudaError_t (*pfn_cudaLaunchKernelExC)(const gb_cudaLaunchConfig *,
                                               const void *, void **);
typedef cudaError_t (*pfn_cudaStreamSynchronize)(cudaStream_t);
/* CUDA doc: stream priority APIs (CUDA 5.0+, cudaDevAttrStreamPrioritiesSupported). */
typedef cudaError_t (*pfn_cudaStreamCreateWithPriority)(cudaStream_t *, unsigned int, int);
typedef cudaError_t (*pfn_cudaDeviceGetAttribute)(int *, int, int);
typedef cudaError_t (*pfn_cudaDeviceGetStreamPriorityRange)(int *, int *);
typedef cudaError_t (*pfn_cudaStreamCreate)(cudaStream_t *, unsigned int);
/* CUDA graph stream capture status (CUDA 10.1+). Values are stable across
 * driver versions; no need to include cuda_runtime_api.h for this enum. */
typedef enum {
    gb_cudaStreamCaptureStatusNone        = 0,
    gb_cudaStreamCaptureStatusActive      = 1,
    gb_cudaStreamCaptureStatusInvalidated = 2,
} gb_cudaStreamCaptureStatus;
typedef cudaError_t (*pfn_cudaStreamIsCapturing)(cudaStream_t, gb_cudaStreamCaptureStatus *);
typedef cudaError_t (*pfn_cudaStreamBeginCapture)(cudaStream_t, int /* cudaStreamCaptureMode */);

/* __cudaRegisterFunction - CUDA runtime internal kernel registration.
 * Called for every __global__ function at library load time.  Gives us
 * the host→name mapping needed for data-driven remote kernel dispatch. */
typedef struct { unsigned x, y, z; } gb_uint3;
typedef void (*pfn___cudaRegisterFunction)(void **, const char *, char *,
                                           const char *, int, gb_uint3 *,
                                           gb_uint3 *, gb_dim3 *,
                                           gb_dim3 *, int *);


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

/* CUDA Virtual Memory Management (VMM) hooks */
typedef CUresult    (*pfn_cuMemCreate)(CUmemGenericAllocationHandle *, size_t,
                                       const CUmemAllocationProp *, unsigned long long);
typedef CUresult    (*pfn_cuMemRelease)(CUmemGenericAllocationHandle);
typedef CUresult    (*pfn_cuMemMap)(CUdeviceptr, size_t, size_t,
                                    CUmemGenericAllocationHandle, unsigned long long);
typedef CUresult    (*pfn_cuMemUnmap)(CUdeviceptr, size_t);
typedef CUresult    (*pfn_cuMemSetAccess)(CUdeviceptr, size_t,
                                          const CUmemAccessDesc *, size_t);
typedef CUresult    (*pfn_cuMemAddressReserve)(CUdeviceptr *, size_t, size_t,
                                               CUdeviceptr, unsigned long long);
typedef CUresult    (*pfn_cuMemAddressFree)(CUdeviceptr, size_t);
typedef CUresult    (*pfn_cuMemGetAllocationGranularity)(size_t *,
                                                         const CUmemAllocationProp *,
                                                         unsigned int);

/* NVML types (minimal - avoids libnvidia-ml dependency) */
typedef void *nvmlDevice_t;
typedef unsigned int nvmlReturn_t;
#define NVML_SUCCESS 0
typedef struct { unsigned long long total; unsigned long long free; unsigned long long used; } nvmlMemory_t;
typedef struct { unsigned int version; unsigned long long total; unsigned long long reserved;
                 unsigned long long free; unsigned long long used; } nvmlMemory_v2_t;
/* nvmlDeviceGetMemoryInfo_v3 - added in NVML 12.x / driver 520+.
 * exportedToOtherProcess: memory mapped into another process via cuIpcOpenMemHandle. */
typedef struct { unsigned long long total; unsigned long long reserved;
                 unsigned long long free; unsigned long long used;
                 unsigned long long exportedToOtherProcess; } nvmlMemory_v3_t;
typedef nvmlReturn_t (*pfn_nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t *);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetMemoryInfo_v2)(nvmlDevice_t, nvmlMemory_v2_t *);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetMemoryInfo_v3)(nvmlDevice_t, nvmlMemory_v3_t *);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetCount)(unsigned int *);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t *);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetName)(nvmlDevice_t, char *, unsigned int);
typedef cudaError_t  (*pfn_cudaGetDeviceProperties)(void *, int);
typedef cudaError_t  (*pfn_cudaGetDriverEntryPointByVersion)(const char *, void **,
                                                              unsigned int,
                                                              unsigned long long,
                                                              void *);

/* ------------------------------------------------------------------ */
/*  Global state                                                        */
/* ------------------------------------------------------------------ */

static pfn_cuMemAlloc_v2                   real_cuMemAlloc_v2;
static pfn_cuMemFree_v2                    real_cuMemFree_v2;
static pfn_cuMemAllocManaged               real_cuMemAllocManaged;
static pfn_cuMemGetInfo                    real_cuMemGetInfo;
static pfn_cuMemAllocAsync                 real_cuMemAllocAsync;
static pfn_cuMemAllocFromPoolAsync         real_cuMemAllocFromPoolAsync;  /* PR-N */
static pfn_cudaMalloc                      real_cudaMalloc;
static pfn_cudaFree                        real_cudaFree;
static pfn_cudaMallocManaged               real_cudaMallocManaged;
static pfn_cudaMallocAsync                 real_cudaMallocAsync;
static pfn_cudaFreeAsync                   real_cudaFreeAsync;
/* F-ABI1 fix (2026-06-22): these two were the only cudart hooks still using
 * their own per-function dlsym(RTLD_NEXT,...) instead of the robust
 * gb_cudart_sym() ELF-walk every other hook uses (see gb_cudart_resolve_syms
 * below). RTLD_NEXT can return NULL for the first N calls if the app's real
 * libcudart.so isn't yet visible in this process's link-map scope - found by
 * tracing why H2D copies silently no-op'd on a real GGUF load:
 * cudaMemcpyAsync's local real_fn was NULL, so every call hit the early
 * `return real_fn ? ... : 0`, silently no-op'ing the copy (returns
 * cudaSuccess without copying anything). */
static cudaError_t (*real_cudaMemcpy)(void *, const void *, size_t, int);
static cudaError_t (*real_cudaMemset)(void *, int, size_t);
static cudaError_t (*real_cudaMemsetAsync)(void *, int, size_t, cudaStream_t);
static cudaError_t (*real_cudaMemcpy2DAsync)(void *, size_t, const void *, size_t,
                                             size_t, size_t, int, cudaStream_t);
static pfn_cudaLaunchKernelExC real_cudaLaunchKernelExC;
static cudaError_t (*real_cudaGetKernel)(void **, const void *);
static cudaError_t (*real_cudaMemcpyAsync)(void *, const void *, size_t, int, void *);
static cudaError_t (*real_cudaMemcpyPeer)(void *, int, const void *, int, size_t);
static cudaError_t (*real_cudaMemcpyPeerAsync)(void *, int, const void *, int, size_t, void *);
static pfn_cudaImportExternalMemory        real_cudaImportExternalMemory;
static pfn_cudaExternalMemoryGetMappedBuffer real_cudaExternalMemoryGetMappedBuffer;
static pfn_cudaDestroyExternalMemory       real_cudaDestroyExternalMemory;
static pfn_cudaGetLastError                real_cudaGetLastError;
static pfn_cudaDeviceSynchronize           real_cudaDeviceSynchronize;
static pfn_cudaMemGetInfo                  real_cudaMemGetInfo;
static pfn_cuDeviceTotalMem_v2             real_cuDeviceTotalMem_v2;
static pfn_cuDeviceGetCount                real_cuDeviceGetCount;
static pfn_cuDeviceGet                     real_cuDeviceGet;
static pfn_cuGetProcAddress                real_cuGetProcAddress;
static pfn_cudaGetDeviceCount              real_cudaGetDeviceCount;
static pfn_cudaSetDevice                   real_cudaSetDevice;
static pfn_cudaDeviceCanAccessPeer         real_cudaDeviceCanAccessPeer;
static pfn_cudaDeviceEnablePeerAccess      real_cudaDeviceEnablePeerAccess;
static pfn_cudaPointerGetAttributes        real_cudaPointerGetAttributes;
static pfn_cuLaunchKernel                  real_cuLaunchKernel;
static pfn_cuLaunchCooperativeKernel       real_cuLaunchCooperativeKernel;
static pfn_cuFuncGetParamInfo              real_cuFuncGetParamInfo;  /* PR-NN */
static pfn_cuLaunchKernelEx                real_cuLaunchKernelEx;  /* PR-O */
static pfn_cudaLaunchKernel                real_cudaLaunchKernel;
static pfn_cudaStreamSynchronize           real_cudaStreamSynchronize;
static pfn_cudaStreamCreateWithPriority    real_cudaStreamCreateWithPriority;
static pfn_cudaStreamIsCapturing           real_cudaStreamIsCapturing;
static pfn_cudaStreamBeginCapture          real_cudaStreamBeginCapture;
static pfn_cudaDeviceGetAttribute          real_cudaDeviceGetAttribute;
static pfn_cudaDeviceGetStreamPriorityRange real_cudaDeviceGetStreamPriorityRange;
static pfn___cudaRegisterFunction          real___cudaRegisterFunction;
static pfn_nvmlDeviceGetMemoryInfo         real_nvmlDeviceGetMemoryInfo;
static pfn_nvmlDeviceGetMemoryInfo_v2      real_nvmlDeviceGetMemoryInfo_v2;
static pfn_nvmlDeviceGetMemoryInfo_v3      real_nvmlDeviceGetMemoryInfo_v3;
static pfn_nvmlDeviceGetCount             real_nvmlDeviceGetCount;
static pfn_nvmlDeviceGetHandleByIndex     real_nvmlDeviceGetHandleByIndex;
static pfn_nvmlDeviceGetName              real_nvmlDeviceGetName;
static pfn_cudaGetDeviceProperties              real_cudaGetDeviceProperties;
static pfn_cudaGetDriverEntryPointByVersion     real_cudaGetDriverEntryPointByVersion;

static pfn_cuMemHostRegister               real_cuMemHostRegister;
static pfn_cuMemHostUnregister             real_cuMemHostUnregister;
static pfn_cuMemHostGetDevicePointer       real_cuMemHostGetDevicePointer;
static pfn_cuMemPrefetchAsync              real_cuMemPrefetchAsync;
static pfn_cudaMemPrefetchAsync            real_cudaMemPrefetchAsync;
static pfn_cuMemAdvise                     real_cuMemAdvise;
static pfn_cuMemFreeAsync                  real_cuMemFreeAsync;
static pfn_cuDeviceGetAttribute            real_cuDeviceGetAttribute;
static pfn_cuMemCreate                     real_cuMemCreate;
static pfn_cuMemRelease                    real_cuMemRelease;
static pfn_cuMemMap                        real_cuMemMap;
static pfn_cuMemUnmap                      real_cuMemUnmap;
static pfn_cuMemSetAccess                  real_cuMemSetAccess;
static pfn_cuMemAddressReserve             real_cuMemAddressReserve;
static pfn_cuMemAddressFree                real_cuMemAddressFree;
static pfn_cuMemGetAllocationGranularity   real_cuMemGetAllocationGranularity;

/* E1: PTX module management - cuModuleLoadData / cuModuleGetFunction */
typedef CUresult (*pfn_cuModuleLoadData)(CUmodule *, const void *);
typedef CUresult (*pfn_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
static pfn_cuModuleLoadData   real_cuModuleLoadData;
static pfn_cuModuleGetFunction real_cuModuleGetFunction;

static CUmodule   g_gb_ptx_module       = NULL;
static CUfunction g_gb_absmax_quant_fn  = NULL;
static CUfunction g_gb_absmax_dequant_fn = NULL;

/* Forward declaration - definition is after the PTX string (avoids reordering the large string) */
static void gb_kv_ptx_init(void *libcuda);

/* ------------------------------------------------------------------ */
/*  T2 Pre-Registered Pool - function implementations                  */
/*  (placed here, after real_* declarations, to avoid forward refs)    */
/* ------------------------------------------------------------------ */

/* Tentative forward declaration - actual definition with initializer is below.
 * C99 §6.9.2 permits multiple tentative definitions of the same static. */
static int gb_dev_fd;

/* Compute capability major version - probed at init; 0 = unknown.
 * Moved here (before gb_pool_init) because gb_pool_init checks it.
 * Also guards cuMemAllocAsync (requires cc >= 8.0 / Ampere+; V100 = 7.0
 * returns CUDA_ERROR_NOT_SUPPORTED without this check). */
static int gb_cc_major = 0;

/* Forward declaration - definition with initializer lives near the other
 * gb_t2_* counters below.  C99 tentative definition (§6.9.2). */
static _Atomic size_t gb_t2_overflow_bytes;
/* Active zero-copy T2 allocation count (incremented on alloc, decremented on free). */
static _Atomic int gb_bvmm_zerocopy_count;

/* PR-I/H4: bytes of Path-C managed UVM that have been allocated but may not
 * yet have caused physical page-backing.  These bytes do NOT yet show as
 * "used" in /proc/meminfo's MemAvailable (the OS only commits physical pages
 * on first GPU SM touch), so the host-RAM safety floor must subtract this
 * counter from MemAvailable when deciding whether to admit a new T2 alloc.
 * Otherwise a burst of allocations passes the floor check while still pre-
 * touch, then their first touch drops MemAvailable below the floor in one
 * step and the kernel OOM-killer fires.
 *
 * Lifecycle: incremented on successful gb_vmm_t2_alloc_blackwell_managed,
 * decremented on the matching cuMemFree_v2/cudaFree (gb_bvmm_free_dispatch).
 * Conservative: bytes that have already been touched continue to be
 * subtracted from the MemAvailable estimate until the alloc is freed - over-
 * refuses new allocs in steady-state but never under-counts. */
static _Atomic size_t gb_t2_pending_uvm_bytes;

/* ------------------------------------------------------------------ */
/*  Blackwell (cc >= 12) T2 table + allocators                          */
/*  On desktop Blackwell PCIe (no ATS), managed UVM (cuMemAllocManaged) */
/*  is host-backed and SM-inaccessible from compute kernels.            */
/*  Zero-copy path: mmap + cuMemHostRegister(DEVICEMAP|PORTABLE) +      */
/*  cuMemHostGetDevicePointer gives an SM-accessible device VA that      */
/*  reads host RAM over PCIe (~20 GB/s).  No ATS required.              */
/*  VMM path (cuMemCreate HOST + cuMemMap) is fallback.                 */
/*  We keep a compact ptr→{handle,size,type} table for cleanup.         */
/*  Placed here, after real_* declarations, to avoid forward refs.      */
/* ------------------------------------------------------------------ */

#define BVMM_HT_BITS 9
#define BVMM_HT_SIZE (1u << BVMM_HT_BITS)
#define BVMM_HT_MASK (BVMM_HT_SIZE - 1u)
/* Tombstone sentinel: a slot that was deleted but must not stop a probe chain.
 * Never a real VA (CUDA device pointers are in the low 48-bit range). */
#define BVMM_HT_TOMBSTONE ((CUdeviceptr)(UINT64_MAX))

#define BVMM_TYPE_MANAGED  0  /* cuMemAllocManaged → cudaFree on device ptr */
#define BVMM_TYPE_VMM      1  /* cuMemCreate HOST → cuMemUnmap/AddrFree/Release */
#define BVMM_TYPE_ZEROCOPY 2  /* mmap+cuMemHostRegister → cuMemHostUnregister+munmap */

typedef struct {
    CUdeviceptr                  va;      /* 0 = empty slot */
    CUmemGenericAllocationHandle handle;  /* VMM: alloc handle; ZC: host ptr as uint64 */
    size_t                       va_size; /* rounded-up size for unmap/free */
    uint8_t                      type;   /* BVMM_TYPE_* */
} bvmm_ht_entry_t;

static bvmm_ht_entry_t  gb_bvmm_ht[BVMM_HT_SIZE];
static pthread_mutex_t  gb_bvmm_ht_lock = PTHREAD_MUTEX_INITIALIZER;

static inline uint32_t bvmm_ht_hash(CUdeviceptr va)
{
    return (uint32_t)(((uint64_t)va * 0x9E3779B97F4A7C15ULL) >> (64 - BVMM_HT_BITS));
}

/* Returns 1 on success, 0 if the table is full.  Callers MUST handle the
 * failure path - silently dropping an entry would leak the underlying alloc
 * and corrupt gb_t2_overflow_bytes accounting.
 *
 * S-C1/S-H4: free slot = (va == 0) || (va == BVMM_HT_TOMBSTONE).
 * If va already exists (dup insert), update in-place and return 1 so the
 * caller's accounting doesn't double-count. */
static int bvmm_ht_insert(CUdeviceptr va, CUmemGenericAllocationHandle h, size_t sz, uint8_t type)
{
    uint32_t i, slot;
    int ok = 0;
    pthread_mutex_lock(&gb_bvmm_ht_lock);
    slot = bvmm_ht_hash(va) & BVMM_HT_MASK;
    for (i = 0; i < BVMM_HT_SIZE; i++) {
        bvmm_ht_entry_t *e = &gb_bvmm_ht[(slot + i) & BVMM_HT_MASK];
        if (e->va == va) {
            /* Duplicate key — update in-place (shouldn't happen normally). */
            e->handle = h; e->va_size = sz; e->type = type;
            ok = 1; break;
        }
        if (e->va == 0 || e->va == BVMM_HT_TOMBSTONE) {
            e->va = va; e->handle = h; e->va_size = sz; e->type = type; ok = 1; break;
        }
    }
    pthread_mutex_unlock(&gb_bvmm_ht_lock);
    return ok;
}

/* Returns 1 if found and removed, fills *out_handle, *out_va_size, *out_type. */
static int bvmm_ht_remove(CUdeviceptr va,
                           CUmemGenericAllocationHandle *out_handle,
                           size_t *out_va_size,
                           uint8_t *out_type)
{
    if (!va) return 0;
    uint32_t i, slot;
    pthread_mutex_lock(&gb_bvmm_ht_lock);
    slot = bvmm_ht_hash(va) & BVMM_HT_MASK;
    for (i = 0; i < BVMM_HT_SIZE; i++) {
        bvmm_ht_entry_t *e = &gb_bvmm_ht[(slot + i) & BVMM_HT_MASK];
        if (e->va == 0) break;
        if (e->va == va) {
            if (out_handle)  *out_handle  = e->handle;
            if (out_va_size) *out_va_size = e->va_size;
            if (out_type)    *out_type    = e->type;
            /* S-C1: tombstone (not zero) so probe chains for keys that
             * hashed past this slot remain intact. */
            e->va = BVMM_HT_TOMBSTONE; e->handle = 0; e->va_size = 0; e->type = 0;
            pthread_mutex_unlock(&gb_bvmm_ht_lock);
            return 1;
        }
    }
    pthread_mutex_unlock(&gb_bvmm_ht_lock);
    return 0;
}

/* Returns 1 if found, fills *out_type. Does NOT remove the entry.
 * Used by the prefetch hooks to decide whether to short-circuit a prefetch
 * that would migrate Path-C managed-UVM pages off host RAM. */
static int bvmm_ht_peek_type(CUdeviceptr va, uint8_t *out_type)
{
    if (!va) return 0;
    uint32_t i, slot;
    int found = 0;
    pthread_mutex_lock(&gb_bvmm_ht_lock);
    slot = bvmm_ht_hash(va) & BVMM_HT_MASK;
    for (i = 0; i < BVMM_HT_SIZE; i++) {
        bvmm_ht_entry_t *e = &gb_bvmm_ht[(slot + i) & BVMM_HT_MASK];
        if (e->va == 0) break;
        if (e->va == va) {
            if (out_type) *out_type = e->type;
            found = 1;
            break;
        }
    }
    pthread_mutex_unlock(&gb_bvmm_ht_lock);
    return found;
}

/* PR-D/R2: shared free-dispatch for bvmm_ht-tracked allocations.
 *
 * Before this helper, the same 27-line block was duplicated across
 * cuMemFree_v2, cudaFree, and cudaFreeAsync - three places to keep in
 * sync.  Returns 1 if dptr was a bvmm_ht entry and was freed (caller
 * should return success); returns 0 if not found (caller should fall
 * through to its ht path).
 *
 * BVMM_TYPE_MANAGED uses cuMemFree_v2 (driver API) not cudaFree, because
 * pure-libcuda callers (ggml driver path, vLLM custom kernels) never load
 * libcudart so real_cudaFree is NULL.  See PR-A. */
static int gb_bvmm_free_dispatch(CUdeviceptr dptr)
{
    CUmemGenericAllocationHandle bvmm_handle = 0;
    size_t bvmm_sz = 0;
    uint8_t bvmm_type = BVMM_TYPE_MANAGED;
    if (!bvmm_ht_remove(dptr, &bvmm_handle, &bvmm_sz, &bvmm_type))
        return 0;

    GB_NVTX_EVENT("FREE_BVMM_T2", "T2_DDR", bvmm_sz >> 20, dptr, "blackwell_t2_free");
    switch (bvmm_type) {
    case BVMM_TYPE_MANAGED:
        if (real_cuMemFree_v2) real_cuMemFree_v2(dptr);
        /* PR-I/H4: release pending-UVM accounting at the same time the
         * cap reservation is released (below).  The pages now go back to
         * the OS - wherever they were (touched in RAM, or uncommitted). */
        atomic_fetch_sub_explicit(&gb_t2_pending_uvm_bytes, bvmm_sz, memory_order_relaxed);
        break;
    case BVMM_TYPE_VMM:
        if (real_cuMemUnmap)       real_cuMemUnmap(dptr, bvmm_sz);
        if (real_cuMemAddressFree) real_cuMemAddressFree(dptr, bvmm_sz);
        if (real_cuMemRelease)     real_cuMemRelease(bvmm_handle);
        break;
    case BVMM_TYPE_ZEROCOPY: {
        void *h_ptr = (void *)(uintptr_t)bvmm_handle;
        if (real_cuMemHostUnregister) real_cuMemHostUnregister(h_ptr);
        munmap(h_ptr, bvmm_sz);
        atomic_fetch_sub_explicit(&gb_bvmm_zerocopy_count, 1, memory_order_relaxed);
        break;
    }
    }
    atomic_fetch_sub_explicit(&gb_t2_overflow_bytes, bvmm_sz, memory_order_relaxed);
    return 1;
}

/* Allocate T2 DDR for Blackwell via zero-copy pinned host memory.
 *
 * On desktop PCIe Blackwell (sm_120, no ATS), managed UVM (cuMemAllocManaged)
 * pages live in host RAM and CUDA SM kernels cannot access them - prefetch to
 * GPU device returns cudaErrorInvalidValue, confirming no ATS support.
 *
 * Zero-copy approach: mmap host RAM → cuMemHostRegister(DEVICEMAP|PORTABLE) →
 * cuMemHostGetDevicePointer returns a real CUDA device VA that GPU SMs access
 * over PCIe (~20 GB/s).  No ATS required, works on all CUDA cc >= 2.0 that
 * report canMapHostMemory=1.  GPU does all computation; no CPU offload.
 *
 * bvmm_ht entry: type = BVMM_TYPE_ZEROCOPY, handle = host ptr cast to uint64.
 * Free path: cuMemHostUnregister(host_ptr) + munmap(host_ptr, size). */
static CUresult gb_vmm_t2_alloc_blackwell_zerocopy(CUdeviceptr *out_ptr, size_t size)
{
    if (!real_cuMemHostRegister || !real_cuMemHostGetDevicePointer)
        return CUDA_ERROR_NOT_SUPPORTED;

    void *h_ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (h_ptr == MAP_FAILED) {
        fprintf(stderr, "[GreenBoost] Blackwell ZC T2: mmap FAILED for %zu MB: %m\n", size >> 20);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    CUresult ret = real_cuMemHostRegister(h_ptr, size,
                                          CU_MEMHOSTREGISTER_DEVICEMAP |
                                          CU_MEMHOSTREGISTER_PORTABLE);
    if (ret != CUDA_SUCCESS) {
        fprintf(stderr, "[GreenBoost] Blackwell ZC T2: cuMemHostRegister FAILED ret=%d for %zu MB\n",
                ret, size >> 20);
        munmap(h_ptr, size);
        return ret;
    }

    CUdeviceptr d_ptr = 0;
    ret = real_cuMemHostGetDevicePointer(&d_ptr, h_ptr, 0);
    if (ret != CUDA_SUCCESS || !d_ptr) {
        fprintf(stderr, "[GreenBoost] Blackwell ZC T2: cuMemHostGetDevicePointer FAILED ret=%d\n", ret);
        if (real_cuMemHostUnregister) real_cuMemHostUnregister(h_ptr);
        munmap(h_ptr, size);
        return (ret != CUDA_SUCCESS) ? ret : CUDA_ERROR_NOT_SUPPORTED;
    }

    /* Store host ptr in handle field (both are uint64 on 64-bit); type=ZEROCOPY
     * distinguishes this from VMM entries in the free path. */
    if (!bvmm_ht_insert(d_ptr, (CUmemGenericAllocationHandle)(uintptr_t)h_ptr, size, BVMM_TYPE_ZEROCOPY)) {
        fprintf(stderr, "[GreenBoost] Blackwell ZC T2: bvmm_ht full - freeing alloc, returning OOM\n");
        if (real_cuMemHostUnregister) real_cuMemHostUnregister(h_ptr);
        munmap(h_ptr, size);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    /* PR-F/H3: gb_t2_overflow_bytes reservation is held by the caller
     * (gb_overflow_alloc_ex via gb_t2_try_reserve).  No add here. */
    atomic_fetch_add_explicit(&gb_bvmm_zerocopy_count, 1, memory_order_relaxed);

    /* cuMemAdvise(SET_READ_MOSTLY + SET_ACCESSED_BY): these calls return
     * CUDA_ERROR_INVALID_VALUE on pinned non-managed memory, but as a side effect
     * they set up device page table entries (IOMMU mappings) required for GPU SM
     * compute access to the zero-copy buffer.  Without these calls, cuMemHostRegister
     * + cuMemHostGetDevicePointer gives DMA-only device pointers on Blackwell PCIe -
     * the GPU can DMA (H2D copy) to the address but SM kernels fail with
     * CUDA_ERROR_INVALID_HANDLE (400).
     *
     * The INVALID_VALUE is intentional and expected.  We drain it immediately via
     * real_cudaGetLastError() so it never escapes into the CUDA runtime error queue
     * visible to Python callers.  Without this drain, torch.cuda.empty_cache() and
     * any PyTorch operation that internally calls cudaGetLastError() will raise
     * AcceleratorError: CUDA error: invalid argument in user code. */
    if (real_cuMemAdvise) {
        real_cuMemAdvise(d_ptr, size, CU_MEM_ADVISE_SET_READ_MOSTLY, 0);
        real_cuMemAdvise(d_ptr, size, CU_MEM_ADVISE_SET_ACCESSED_BY, 0);
        /* cu130 (Blackwell): cuMemAdvise on pinned non-managed memory launches
         * an async GPU IOMMU-setup command.  On CUDA 12.x (cu128) the error
         * (INVALID_VALUE) comes back synchronously; on CUDA 13.x (cu130) the
         * command is async and the error (cudaErrorInvalidResourceHandle)
         * arrives in the runtime queue AFTER cuMemAdvise() returns.
         * Synchronize the device first so the async failure materialises, then
         * drain in a loop to clear all queued errors. */
        if (real_cudaDeviceSynchronize)
            real_cudaDeviceSynchronize();
        if (real_cudaGetLastError) {
            cudaError_t _e;
            do { _e = real_cudaGetLastError(); } while (_e != 0 /*cudaSuccess*/);
        }
    }

    GB_NVTX_EVENT("ALLOC_T2_ZC", "T2_DDR", size >> 20, d_ptr, "blackwell_zerocopy_ok");
    *out_ptr = d_ptr;
    return CUDA_SUCCESS;
}

/* Allocate T2 DDR for Blackwell (cc >= 12) via HOST_NUMA_CURRENT pinned VMM.
 *
 * HOST_NUMA_CURRENT + cuMemMap + cuMemSetAccess gives a device-visible VA backed
 * by pinned host NUMA memory - pages are IOMMU-mapped immediately, no UVM
 * page-fault stalls on first SM touch.  This is the same mechanism the shim
 * already uses successfully via the cuMemCreate intercept for PyTorch
 * expandable_segments (see line ~5322).
 *
 * Why this fixes the silent SIGKILL:
 *   cuMemAllocManaged pages sit in host RAM as uncommitted UVM until touched.
 *   With a concurrent 22 GB device_map="cpu" model copy, total committed host
 *   RAM exceeds 62 GB before any GPU compute runs → kernel OOM-killer SIGKILL.
 *   Switching to pinned VMM with cap enforcement prevents unbounded overcommit.
 *
 * bvmm_ht entry: handle != 0 triggers VMM teardown in cuMemFree_v2.
 * Free path: cuMemUnmap + cuMemAddressFree + cuMemRelease (existing cuMemFree_v2 path). */
/* Cached location type: starts as HOST_NUMA_CURRENT; switched to HOST on first
 * INVALID_VALUE (single-socket UMA systems).  0 = not yet resolved. */
static int gb_vmm_host_loc_type = 0;

static CUresult gb_vmm_t2_alloc_blackwell_hostnuma(CUdeviceptr *out_ptr, size_t size)
{
    static size_t gb_hostnuma_granularity = 0;

    /* Resolve location type once - prefer HOST_NUMA_CURRENT, cache HOST if unsupported. */
    if (__builtin_expect(gb_vmm_host_loc_type == 0, 0))
        gb_vmm_host_loc_type = CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT;

    if (!real_cuMemCreate || !real_cuMemMap || !real_cuMemSetAccess ||
        !real_cuMemAddressReserve || !real_cuMemAddressFree || !real_cuMemRelease)
        return CUDA_ERROR_NOT_SUPPORTED;

    /* Probe allocation granularity once using the resolved location type. */
    if (gb_hostnuma_granularity == 0) {
        CUmemAllocationProp gp;
        memset(&gp, 0, sizeof(gp));
        gp.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
        gp.location.type = gb_vmm_host_loc_type;
        gp.location.id   = 0;
        size_t g = 0;
        if (real_cuMemGetAllocationGranularity &&
            real_cuMemGetAllocationGranularity(&g, &gp, 0 /*MINIMUM*/) == CUDA_SUCCESS && g > 0) {
            gb_hostnuma_granularity = g;
        } else {
            /* Try generic HOST as last resort */
            gp.location.type = CU_MEM_LOCATION_TYPE_HOST;
            if (real_cuMemGetAllocationGranularity &&
                real_cuMemGetAllocationGranularity(&g, &gp, 0 /*MINIMUM*/) == CUDA_SUCCESS && g > 0)
                gb_hostnuma_granularity = g;
            else
                gb_hostnuma_granularity = 2ULL * 1024 * 1024; /* 2 MB safe fallback */
        }
        gb_log("T2 VMM granularity: %zu MB (loc_type=%d)", gb_hostnuma_granularity >> 20, gb_vmm_host_loc_type);
    }

    /* Round size up to granularity. */
    size_t gran    = gb_hostnuma_granularity;
    size_t aligned = (size + gran - 1) & ~(gran - 1);

    /* Reserve virtual address range. */
    CUdeviceptr va = 0;
    CUresult ret = real_cuMemAddressReserve(&va, aligned, 0, 0, 0);
    if (ret != CUDA_SUCCESS) {
        fprintf(stderr, "[GreenBoost] Blackwell T2 HOST_NUMA: cuMemAddressReserve FAILED ret=%d for %zu MB\n",
                ret, size >> 20);
        return ret;
    }

    /* Create pinned host physical allocation using the resolved location type. */
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = gb_vmm_host_loc_type;
    prop.location.id   = 0;
    CUmemGenericAllocationHandle h;
    ret = real_cuMemCreate(&h, aligned, &prop, 0);
    if (ret == CUDA_ERROR_INVALID_VALUE &&
        prop.location.type == CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT) {
        /* HOST_NUMA_CURRENT unsupported (single-socket UMA) - switch to HOST permanently. */
        gb_vmm_host_loc_type = CU_MEM_LOCATION_TYPE_HOST;
        fprintf(stderr, "[GreenBoost] Blackwell T2: HOST_NUMA_CURRENT unsupported on this system, "
                "switching to HOST (printed once)\n");
        prop.location.type = CU_MEM_LOCATION_TYPE_HOST;
        ret = real_cuMemCreate(&h, aligned, &prop, 0);
    }
    if (ret != CUDA_SUCCESS) {
        fprintf(stderr, "[GreenBoost] Blackwell T2: cuMemCreate FAILED ret=%d for %zu MB\n",
                ret, size >> 20);
        real_cuMemAddressFree(va, aligned);
        return ret;
    }

    /* Map physical allocation into the reserved VA. */
    ret = real_cuMemMap(va, aligned, 0, h, 0);
    if (ret != CUDA_SUCCESS) {
        fprintf(stderr, "[GreenBoost] Blackwell T2 HOST_NUMA: cuMemMap FAILED ret=%d for %zu MB\n",
                ret, size >> 20);
        real_cuMemRelease(h);
        real_cuMemAddressFree(va, aligned);
        return ret;
    }

    /* Grant device 0 READWRITE access - required before any SM can touch the VA. */
    CUmemAccessDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id   = 0;
    desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    ret = real_cuMemSetAccess(va, aligned, &desc, 1);
    if (ret != CUDA_SUCCESS) {
        fprintf(stderr, "[GreenBoost] Blackwell T2 HOST_NUMA: cuMemSetAccess FAILED ret=%d for %zu MB\n",
                ret, size >> 20);
        real_cuMemUnmap(va, aligned);
        real_cuMemRelease(h);
        real_cuMemAddressFree(va, aligned);
        return ret;
    }

    /* Track in bvmm_ht - type=VMM triggers cuMemUnmap/AddrFree/Release in free path. */
    if (!bvmm_ht_insert(va, h, aligned, BVMM_TYPE_VMM)) {
        fprintf(stderr, "[GreenBoost] Blackwell HOST_NUMA T2: bvmm_ht full - freeing alloc, returning OOM\n");
        if (real_cuMemUnmap)       real_cuMemUnmap(va, aligned);
        if (real_cuMemRelease)     real_cuMemRelease(h);
        if (real_cuMemAddressFree) real_cuMemAddressFree(va, aligned);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    /* PR-F/H3: caller reserved `size`; hostnuma rounded up to `aligned`
     * for page alignment.  Charge the delta against gb_t2_overflow_bytes
     * so the free-path subtraction of `aligned` leaves the counter at
     * net zero per alloc/free cycle (no slow drift below true usage). */
    if (aligned > size)
        atomic_fetch_add_explicit(&gb_t2_overflow_bytes, aligned - size,
                                  memory_order_relaxed);
    GB_NVTX_EVENT("ALLOC_BVMM_T2", "T2_DDR", size >> 20, va, "blackwell_hostnuma_ok");
    *out_ptr = va;
    return CUDA_SUCCESS;
}

/* Allocate T2 DDR for Blackwell PCIe via managed UVM, hinted to stay in host RAM.
 *
 * NVIDIA UVM programming model (cc 6.x+ Linux, software-coherent path -
 * CUDA Programming Guide §4.1.4.2 Data Usage Hints):
 *   cuMemAllocManaged(CU_MEM_ATTACH_GLOBAL)
 *   + cuMemAdvise(SET_PREFERRED_LOCATION = CPU)  → pages stay in host DDR
 *   + cuMemAdvise(SET_ACCESSED_BY        = GPU)  → GPU mappings pre-created,
 *                                                   no first-touch page-fault stall
 * Result: GPU SMs read/write host RAM over PCIe (~20 GB/s); pages never migrate
 * to device memory; no CPU compute; no T3 fallback.  This is the only T2 path
 * that grants SM access on Blackwell PCIe where cuMemHostRegister /
 * cuMemCreate(HOST_NUMA) return DMA-only (SM-inaccessible) pointers.
 *
 * bvmm_ht entry: type = BVMM_TYPE_MANAGED, handle = 0.
 * Free path: cuMemFree_v2 intercept routes BVMM_TYPE_MANAGED → cudaFree(d_ptr). */
static CUresult gb_vmm_t2_alloc_blackwell_managed(CUdeviceptr *out_ptr, size_t size)
{
    if (!real_cuMemAllocManaged || !real_cuMemAdvise) {
        /* cuMemAdvise is load-bearing - without SET_PREFERRED_LOCATION=CPU the
         * pages migrate to the GPU on the first prefetch/touch and we OOM.
         * Refuse to allocate so the caller falls through to the zerocopy path. */
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    CUdeviceptr d_ptr = 0;
    CUresult ret = real_cuMemAllocManaged(&d_ptr, size, CU_MEM_ATTACH_GLOBAL);
    if (ret != CUDA_SUCCESS) {
        fprintf(stderr, "[GreenBoost] Blackwell managed T2: cuMemAllocManaged FAILED ret=%d for %zu MB\n",
                ret, size >> 20);
        return ret;
    }

    /* SET_PREFERRED_LOCATION=CPU is the contract that anchors the pages to host
     * RAM.  If the driver rejects this hint we cannot ship a usable allocation
     * (a subsequent cuMemPrefetchAsync(dst=device) - or any access from the GPU
     * on a non-concurrentManagedAccess device - would migrate the pages off
     * host RAM, defeating Path C).  Free and bubble the error up so the caller
     * can fall through to the zerocopy fallback. */
    CUresult adv = real_cuMemAdvise(d_ptr, size, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, CU_DEVICE_CPU);
    if (adv != CUDA_SUCCESS) {
        fprintf(stderr, "[GreenBoost] Blackwell managed T2: SET_PREFERRED_LOCATION=CPU rejected "
                "(ret=%d) - freeing alloc, falling back\n", adv);
        if (real_cuMemFree_v2) real_cuMemFree_v2(d_ptr);
        return adv;
    }
    /* SET_ACCESSED_BY: pre-creates GPU page mappings so the first SM access does
     * not stall on a fault.  GreenBoost presents exactly one virtual device 0 to
     * apps (see the cluster architecture docs), and CUDA_VISIBLE_DEVICES
     * remapping is resolved at a layer above us, so (CUdevice)0 is the correct
     * accessor here regardless of physical GPU index.  Failure of this hint
     * costs first-touch fault latency only, not correctness - log and continue. */
    {
        CUresult ab = real_cuMemAdvise(d_ptr, size, CU_MEM_ADVISE_SET_ACCESSED_BY, (CUdevice)0);
        if (ab != CUDA_SUCCESS)
            gb_log("Blackwell managed T2: SET_ACCESSED_BY device 0 returned ret=%d (non-fatal)", ab);
    }

    if (!bvmm_ht_insert(d_ptr, 0, size, BVMM_TYPE_MANAGED)) {
        fprintf(stderr, "[GreenBoost] Blackwell managed T2: bvmm_ht full - freeing alloc, returning OOM\n");
        if (real_cuMemFree_v2) real_cuMemFree_v2(d_ptr);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    /* PR-F/H3: gb_t2_overflow_bytes reservation is held by the caller
     * (gb_overflow_alloc_ex via gb_t2_try_reserve).  No add here.
     * PR-I/H4: managed-UVM bytes may not yet be reflected in
     * /proc/meminfo's MemAvailable until first GPU touch, so account them
     * separately for the host-RAM safety floor calculation. */
    atomic_fetch_add_explicit(&gb_t2_pending_uvm_bytes, size, memory_order_relaxed);
    GB_NVTX_EVENT("ALLOC_T2_MANAGED", "T2_DDR", size >> 20, d_ptr, "blackwell_managed_uvm_ok");
    *out_ptr = d_ptr;
    return CUDA_SUCCESS;
}

static void gb_pool_init(void)
{
    gb_pool_t *p = &gb_t2_reg_pool;
    size_t pool_sz = gb_pool_configured_bytes;

    if (!pool_sz || !real_cuMemHostRegister || !real_cuMemHostGetDevicePointer) {
        p->init_failed = 1;
        return;
    }

    /* Lazy cc re-probe: pool_init fires via pthread_once during the first
     * overflow alloc, by which time cuInit has run.  If the constructor probe
     * failed (gb_cc_major == 0) retry now so the Blackwell gate below is accurate. */
    if (gb_cc_major == 0 && real_cuDeviceGetAttribute) {
        int cc = 0;
        if (real_cuDeviceGetAttribute(&cc, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0) == CUDA_SUCCESS && cc > 0)
            gb_cc_major = cc;
    }

    /* Blackwell (sm_120, cc >= 12): skip the pre-registered pool.
     * Per-alloc zero-copy (mmap+cuMemHostRegister+cuMemHostGetDevicePointer) is used
     * instead; the pool's slab layout conflicts with per-tensor cuMemHostUnregister. */
    if (gb_cc_major >= 12) {
        fprintf(stderr, "[GreenBoost] T2 pool: cc=%d.x Blackwell - "
                "using per-alloc zero-copy path (cuMemHostRegister+GetDevicePointer)\n", gb_cc_major);
        p->init_failed = 1;
        return;
    }

    memset(p, 0, sizeof(*p));
    pthread_mutex_init(&p->lock, NULL);
    p->dmabuf_fd = -1;

    p->base = mmap(NULL, pool_sz, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p->base == MAP_FAILED) {
        fprintf(stderr, "[GreenBoost] T2 pool mmap failed for %zu MB: %m\n", pool_sz >> 20);
        p->base = NULL;
        p->init_failed = 1;
        return;
    }

    /* Optional kernel pin - improves page contiguity but not required */
    if (gb_dev_fd >= 0) {
        struct gb_pin_req req;
        memset(&req, 0, sizeof(req));
        req.vaddr = (uint64_t)(uintptr_t)p->base;
        req.size  = pool_sz;
        req.flags = 0;
        req.fd    = -1;
        if (ioctl(gb_dev_fd, GB_IOCTL_PIN_USER_PTR, &req) == 0)
            p->dmabuf_fd = req.fd;
        else
            fprintf(stderr, "[GreenBoost] T2 pool kernel pin failed (non-fatal): %m\n");
    }

    {
        CUresult ret = real_cuMemHostRegister(p->base, pool_sz,
                                              CU_MEMHOSTREGISTER_DEVICEMAP |
                                              CU_MEMHOSTREGISTER_PORTABLE);
        if (ret != CUDA_SUCCESS) {
            fprintf(stderr, "[GreenBoost] T2 pool cuMemHostRegister FAILED ret=%d for %zu MB"
                    " - pool disabled, using per-allocation Path A\n", ret, pool_sz >> 20);
            munmap(p->base, pool_sz);
            if (p->dmabuf_fd >= 0) { close(p->dmabuf_fd); p->dmabuf_fd = -1; }
            p->base = NULL;
            p->init_failed = 1;
            return;
        }
    }

    {
        CUresult ret = real_cuMemHostGetDevicePointer(&p->dev_base, p->base, 0);
        if (ret != CUDA_SUCCESS) {
            fprintf(stderr, "[GreenBoost] T2 pool cuMemHostGetDevicePointer FAILED ret=%d\n", ret);
            if (real_cuMemHostUnregister) real_cuMemHostUnregister(p->base);
            munmap(p->base, pool_sz);
            if (p->dmabuf_fd >= 0) { close(p->dmabuf_fd); p->dmabuf_fd = -1; }
            p->base = NULL;
            p->init_failed = 1;
            return;
        }
    }

    /* U2: pool starts as one giant segment in the HUGE class */
    p->sc[3][0].off = 0;
    p->sc[3][0].sz  = pool_sz;
    p->nfree[3] = 1;
    p->total   = pool_sz;
    p->initialized = 1;

    /* CUDA doc: CU_MEM_ADVISE_SET_READ_MOSTLY - model weights are read far
     * more than written; hint the driver to create read-only GPU L2 cache
     * mappings for this range so PCIe bandwidth is not wasted on cache
     * invalidation from spurious write-back traffic. */
    if (real_cuMemAdvise) {
        real_cuMemAdvise(p->dev_base, pool_sz, CU_MEM_ADVISE_SET_READ_MOSTLY, 0);
        /* CUDA doc: CU_MEM_ADVISE_SET_ACCESSED_BY - tell the driver that
         * device 0 will access this memory frequently; keeps IOMMU TLB
         * entries live and prevents redundant DMA-re-mapping on every
         * inference token decode. */
        real_cuMemAdvise(p->dev_base, pool_sz, CU_MEM_ADVISE_SET_ACCESSED_BY, 0);
    }

    fprintf(stderr, "[GreenBoost] T2 pool ready: %zu MB pre-registered at dev_ptr=0x%llx\n",
            pool_sz >> 20, (unsigned long long)p->dev_base);
}

static void gb_pool_init_trampoline(void) { gb_pool_init(); }

/* U2: insert a segment into size-class c's sorted free list and coalesce neighbors. */
static void gb_pool_sc_insert(gb_pool_t *p, int c, size_t off, size_t sz)
{
    gb_pool_seg_t *sl = p->sc[c];
    int *nf = &p->nfree[c];
    int i;
    for (i = 0; i < *nf; i++) {
        if (sl[i].off > off) break;
    }
    if (*nf >= gb_pool_sc_max[c]) {
        /* Class full - promote to next larger class or leak as last resort */
        if (c < 3) { gb_pool_sc_insert(p, c + 1, off, sz); return; }
        fprintf(stderr, "[GreenBoost] T2 pool sc[3] full - %zu B at off=%zu leaked\n", sz, off);
        return;
    }
    memmove(&sl[i + 1], &sl[i], (size_t)(*nf - i) * sizeof(gb_pool_seg_t));
    sl[i].off = off; sl[i].sz = sz; (*nf)++;
    /* Coalesce left */
    if (i > 0 && sl[i-1].off + sl[i-1].sz == off) {
        sl[i-1].sz += sz;
        memmove(&sl[i], &sl[i+1], (size_t)(*nf - i - 1) * sizeof(gb_pool_seg_t));
        (*nf)--; i--;
    }
    /* Coalesce right */
    if (i < *nf - 1 && sl[i].off + sl[i].sz == sl[i+1].off) {
        sl[i].sz += sl[i+1].sz;
        memmove(&sl[i+1], &sl[i+2], (size_t)(*nf - i - 2) * sizeof(gb_pool_seg_t));
        (*nf)--;
    }
}

/* U2+R1: best-fit alloc within size class c; returns offset or SIZE_MAX on miss. */
static size_t gb_pool_sc_alloc(gb_pool_t *p, int c, size_t aligned)
{
    gb_pool_seg_t *sl = p->sc[c];
    int best = -1, i;
    size_t best_sz = (size_t)-1;
    for (i = 0; i < p->nfree[c]; i++) {
        if (sl[i].sz >= aligned && sl[i].sz < best_sz) {
            best = i; best_sz = sl[i].sz;
            if (best_sz == aligned) break;
        }
    }
    if (best < 0) return (size_t)-1;
    size_t off = sl[best].off;
    if (sl[best].sz == aligned) {
        memmove(&sl[best], &sl[best+1],
                (size_t)(p->nfree[c] - best - 1) * sizeof(gb_pool_seg_t));
        p->nfree[c]--;
    } else {
        sl[best].off += aligned;
        sl[best].sz  -= aligned;
        /* Remainder may belong to a smaller class - move it */
        size_t rem = sl[best].sz;
        int rc = gb_pool_sc(rem);
        if (rc < c) {
            size_t rem_off = sl[best].off;
            memmove(&sl[best], &sl[best+1],
                    (size_t)(p->nfree[c] - best - 1) * sizeof(gb_pool_seg_t));
            p->nfree[c]--;
            gb_pool_sc_insert(p, rc, rem_off, rem);
        }
    }
    return off;
}

static CUresult gb_pool_alloc(size_t sz, CUdeviceptr *dptr, void **host_out)
{
    gb_pool_t *p = &gb_t2_reg_pool;
    /* R6: 256-byte alignment */
    size_t aligned = (sz + 0xFFUL) & ~(size_t)0xFFUL;
    int target_c = gb_pool_sc(aligned);

/* U14: KV-aligned pool alloc - rounds up to GB_KV_MIN_BLOCK_ALIGN (4 KB).
 * Tracks internal fragmentation for metrics.  Called from KV overflow paths. */
#define gb_pool_alloc_kv(sz_, dptr_, host_)  ({ \
    size_t _kv_req = (sz_); \
    size_t _kv_aligned = (_kv_req + GB_KV_MIN_BLOCK_ALIGN - 1) & ~(GB_KV_MIN_BLOCK_ALIGN - 1); \
    if (_kv_aligned > _kv_req) \
        atomic_fetch_add_explicit(&g_kv_internal_frag_bytes, _kv_aligned - _kv_req, \
                                   memory_order_relaxed); \
    gb_pool_alloc(_kv_aligned, dptr_, host_); \
})
    size_t off = (size_t)-1;

    GB_NVTX_PUSH("GB:T2_alloc", GB_NVTX_COLOR_T2);
    pthread_mutex_lock(&p->lock);
    /* U2: try target class first, then spill upward */
    for (int c = target_c; c < 4 && off == (size_t)-1; c++)
        off = gb_pool_sc_alloc(p, c, aligned);
    pthread_mutex_unlock(&p->lock);
    GB_NVTX_POP();

    if (off == (size_t)-1) return CUDA_ERROR_OUT_OF_MEMORY;
    *dptr     = p->dev_base + off;
    *host_out = (char *)p->base + off;
    GB_NVTX_EVENT("ALLOC_T2_POOL", "T2_DDR", aligned >> 20, *dptr, "path_a_pool_ok");
    return CUDA_SUCCESS;
}

/* R1: Fragmentation metric - 0 = no fragmentation, 100 = fully fragmented. */
static int gb_pool_fragmentation(void)
{
    gb_pool_t *p = &gb_t2_reg_pool;
    size_t total_free = 0, max_seg = 0;
    if (!p->initialized) return 0;
    pthread_mutex_lock(&p->lock);
    for (int c = 0; c < 4; c++) {
        for (int i = 0; i < p->nfree[c]; i++) {
            total_free += p->sc[c][i].sz;
            if (p->sc[c][i].sz > max_seg) max_seg = p->sc[c][i].sz;
        }
    }
    pthread_mutex_unlock(&p->lock);
    if (total_free == 0) return 0;
    return (int)(100 - (max_seg * 100 / total_free));
}

static void gb_pool_free(CUdeviceptr dptr, size_t sz)
{
    gb_pool_t *p = &gb_t2_reg_pool;
    size_t aligned = (sz + 0xFFUL) & ~(size_t)0xFFUL;
    size_t off = (size_t)(dptr - p->dev_base);
    int c = gb_pool_sc(aligned);
    pthread_mutex_lock(&p->lock);
    gb_pool_sc_insert(p, c, off, aligned);
    pthread_mutex_unlock(&p->lock);
}

static int gb_pool_contains(CUdeviceptr dptr)
{
    gb_pool_t *p = &gb_t2_reg_pool;
    return p->initialized &&
           dptr >= p->dev_base &&
           dptr <  p->dev_base + p->total;
}

/* ------------------------------------------------------------------ */

static size_t vram_headroom_bytes    = 512ULL * 1024 * 1024;  /* 512 MB default - scaled to 5% of VRAM after NVML probe; override with GREENBOOST_VRAM_HEADROOM_MB */
/* Workstation GPU headroom: subtracted from reported free VRAM so --fit leaves
 * this many MB of physical GPU VRAM for the desktop/display/other apps.
 * Dynamic by default (see gb_effective_workstation_reserve_bytes): sized to
 * whatever GNOME/compositor/other GPU clients are actually using right now,
 * plus a small safety margin - not a flat tax on every GPU regardless of
 * desktop load. Set GREENBOOST_WORKSTATION_RESERVE_MB to force a fixed value
 * instead (e.g. 0 on a dedicated headless inference node).
 * gaming_mode=1 automatically doubles the effective reserve. */
static size_t g_workstation_reserve_bytes = 1024ULL * 1024 * 1024; /* fallback only; used verbatim when env override is set */
static int    g_workstation_reserve_from_env = 0;
#define GB_WS_RESERVE_MIN_BYTES   (256ULL * 1024 * 1024)   /* floor: always leave a small cushion */
#define GB_WS_RESERVE_MAX_BYTES   (2048ULL * 1024 * 1024)  /* ceiling: avoid over-reserving if some other GPU client spikes */
#define GB_WS_RESERVE_MARGIN_BYTES (256ULL * 1024 * 1024)  /* burst margin added on top of measured desktop usage */
/* ENH-05: Stats write interval in ms. Default 250 ms; override with
 * GREENBOOST_STATS_INTERVAL_MS env var for finer resolution during debugging. */
static uint64_t g_stats_interval_ms = 250ULL;
static size_t gb_virtual_vram_bytes  = 0; /* T2+T3 combined - reported to CUDA; set from sysfs at init; 0 = not yet configured */
static size_t gb_t2_pool_bytes       = 0; /* T2 DDR pool only (virtual_vram_gb × 1 GiB); Path A/B skip above this threshold */
static _Atomic size_t gb_t2_overflow_bytes = 0; /* cumulative T2 RAM pinned by Path A + Path B; decremented on free */
static size_t gb_safety_reserve_bytes = 0;      /* mirrors kernel safety_reserve_gb - read from sysfs at init */
static size_t gb_physical_vram_bytes = 0; /* real GPU VRAM - probed at init via NVML; 0 = unknown */
static size_t g_nvme_pool_bytes      = 0; /* NVMe backing pool size - set at init from sysfs nvme_pool_gb */

/* Forward declaration - defined after gb_shim_init resolves sysfs values */
static size_t gb_get_mem_available(void);
static int gb_feeder_exclusive(void);
static size_t gb_nvlink_aggregated_bytes = 0; /* NVLink aggregated VRAM - added when nvlink_ready=1 */
static size_t g_cluster_remote_total_bytes      = 0; /* sum of all feeder T1+T2+T3, cached at shim init */
static size_t g_cluster_remote_free_bytes       = 0; /* feeder free memory (T1+T2+T3), cached at shim init */
static size_t g_cluster_remote_t1t2_total_bytes = 0; /* feeder T1+T2 only - used in NVML total to prevent T3 inflating default_num_ctx */
static size_t g_cluster_remote_t1t2_free_bytes  = 0; /* feeder T1+T2 free only - excludes T3 NVMe for context-length sizing */
static size_t g_cluster_remote_vram_bytes       = 0; /* feeder GPU VRAM (T1) only - for display as "virtual VRAM = local + feeder" */

/* T1-saturation tracker: bytes successfully placed via real_cudaMalloc/cuMemAlloc_v2.
 * When this reaches gb_physical_vram_bytes, physical T1 is full → route to feeder T1
 * (GPU VRAM on feeder, faster than any DDR) before kernel module silently uses local T2.
 * Priority rule: T1_local → T1_feeder → T2_by_speed → T3_local → T3_feeder */
static _Atomic size_t g_local_t1_alloc_bytes = 0;

/* Dynamic workstation reserve: real_total/real_free are the *physical* values
 * from real_cuMemGetInfo (not GreenBoost-inflated). Desktop/compositor usage
 * = whatever physical VRAM is used that isn't ours (g_local_t1_alloc_bytes).
 * Reserve = that usage + a burst margin, clamped to [MIN, MAX] so a sudden
 * compositor allocation doesn't starve inference, and a one-off competing app
 * doesn't permanently lock up multiple GB GreenBoost will never use. */
static size_t gb_effective_workstation_reserve_bytes(size_t real_free, size_t real_total)
{
    if (g_workstation_reserve_from_env)
        return g_workstation_reserve_bytes;

    size_t other_used = (real_total > real_free) ? (real_total - real_free) : 0;
    size_t our_used   = atomic_load_explicit(&g_local_t1_alloc_bytes, memory_order_relaxed);
    size_t desktop_used = (other_used > our_used) ? (other_used - our_used) : 0;

    size_t reserve = desktop_used + GB_WS_RESERVE_MARGIN_BYTES;
    if (reserve < GB_WS_RESERVE_MIN_BYTES) reserve = GB_WS_RESERVE_MIN_BYTES;
    if (reserve > GB_WS_RESERVE_MAX_BYTES) reserve = GB_WS_RESERVE_MAX_BYTES;
    return reserve;
}

/* Remote cluster live counters - written to shim_stats for greenboost status */
static _Atomic size_t g_remote_alloc_count    = 0; /* allocations routed to feeder T1 */
static _Atomic size_t g_remote_alloc_mb       = 0; /* MB placed on feeder T1 */
static _Atomic size_t g_h2d_mb                = 0; /* MB sent host→feeder (memcpy H2D) */
static _Atomic size_t g_d2h_mb                = 0; /* MB recv feeder→host (memcpy D2H) */
static _Atomic size_t g_kernel_dispatch_count  = 0; /* kernels dispatched to feeder GPU */
static int    gb_compute_domain_active = 0; /* ComputeDomain workload flag - read from sysfs */

/* R3: Per-tier allocation statistics (RMM statistics_resource_adaptor pattern).
 * Indexed by gb_tensor_tier_t; updated lock-free on every alloc/free. */
typedef enum {
    GB_TIER_T1_LOCAL   = 0,  /* real cudaMalloc on local GPU              */
    GB_TIER_T1_FEEDER  = 1,  /* feeder GPU VRAM (fake ptr 0xAA...)        */
    GB_TIER_T2_LOCAL   = 2,  /* local DDR pool (DMA-BUF / cuMemHostReg)   */
    GB_TIER_T2_FEEDER  = 3,  /* feeder DDR (GB_ALLOC_TIER_T2 request)     */
    GB_TIER_T3_LOCAL   = 4,  /* local NVMe / GDS (Path D)                 */
    GB_TIER_T3_FEEDER  = 5,  /* feeder NVMe (GB_ALLOC_TIER_T3 request)    */
    GB_TIER_VMM        = 6,  /* cuMemCreate host-pinned VMM fallback       */
    GB_TIER_PATH_B     = 7,  /* cuMemHostRegister (Path B, no kmod)       */
    GB_TIER_COUNT      = 8,
} gb_tensor_tier_t;

typedef struct {
    _Atomic int64_t current_bytes;   /* bytes currently allocated in this tier */
    _Atomic int64_t peak_bytes;      /* high-water mark                        */
    _Atomic int64_t lifetime_bytes;  /* cumulative bytes allocated (monotone)  */
    _Atomic int64_t alloc_count;     /* total alloc calls that landed here     */
    _Atomic int64_t free_count;      /* total free calls                       */
} gb_tier_stats_t;

static gb_tier_stats_t g_tier_stats[GB_TIER_COUNT];

static void gb_tier_record_alloc(gb_tensor_tier_t tier, size_t bytes)
{
    int64_t cur = atomic_fetch_add_explicit(&g_tier_stats[tier].current_bytes,
                                             (int64_t)bytes, memory_order_relaxed)
                  + (int64_t)bytes;
    atomic_fetch_add_explicit(&g_tier_stats[tier].lifetime_bytes, (int64_t)bytes,
                               memory_order_relaxed);
    atomic_fetch_add_explicit(&g_tier_stats[tier].alloc_count, 1, memory_order_relaxed);
    int64_t peak = atomic_load_explicit(&g_tier_stats[tier].peak_bytes, memory_order_relaxed);
    while (cur > peak) {
        if (atomic_compare_exchange_weak_explicit(&g_tier_stats[tier].peak_bytes, &peak, cur,
                                                   memory_order_relaxed, memory_order_relaxed))
            break;
    }
}

static void gb_tier_record_free(gb_tensor_tier_t tier, size_t bytes)
{
    atomic_fetch_sub_explicit(&g_tier_stats[tier].current_bytes, (int64_t)bytes,
                               memory_order_relaxed);
    atomic_fetch_add_explicit(&g_tier_stats[tier].free_count, 1, memory_order_relaxed);
}
/* When GREENBOOST_KV_OVERFLOW=1, all overflow allocs receive
 * GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY - tells the kernel to freeze them
 * in T2 LRU and refuse T3 spill.  Use this when running ExLlamaV3 or any engine
 * whose overflow allocs are predominantly KV cache rather than weights. */
static int    g_kv_overflow_mode      = 0;
/* When GREENBOOST_REPORT_PHYSICAL_VRAM=1, cuDeviceTotalMem / cudaGetDeviceProperties
 * return the unmodified physical VRAM (no virtual-pool inflation).
 * Ollama needs the inflation to schedule layers; PyTorch / diffusers need the
 * truth so their internal allocator doesn't issue oversized allocations that
 * exceed real T1 before the cudaMalloc hook can spill to T2. */
static int    g_report_physical_vram  = 0;
/* GREENBOOST_DEBUG_ATTR=1: log every device-attribute query (attr, ordinal,
 * local count, and whether it resolved local or feeder) - the diagnostic for
 * the ROADMAP P4 shared-mem-attr misroute. Off by default. */
static int    g_debug_attr            = 0;
/* DMA-BUF mmap+register is the primary path now. */
static int    gb_use_dmabuf         = 1;
/* Path B: cuMemHostRegister without greenboost.ko (containers/VMs).
 * Auto-enabled when /dev/greenboost is unavailable.
 * Set GREENBOOST_NO_HOSTREG=1 to skip Path B (OOM returned when T2 exhausted). */
static int    gb_no_hostreg         = 0;
static int    initialized           = 0;

/* VCM-01: Deferred init - set when GREENBOOST_VLLM_COMPAT=1.
 * Shim constructor skips force-loading libcuda.so to prevent CUDA context
 * conflicts in vLLM EngineCore subprocesses.  Init completes lazily on the
 * first intercepted CUDA call once libcuda.so becomes resident in the process. */
static volatile int         gb_init_deferred = 0;
static pthread_once_t       gb_resume_once   = PTHREAD_ONCE_INIT;

/* /dev/greenboost fd - opened lazily on first DMA-BUF allocation */
static int        gb_dev_fd       = -1;
static pthread_mutex_t gb_dev_lock = PTHREAD_MUTEX_INITIALIZER;

/* KV cache T1 reservation - bytes of VRAM kept free for KV cache.
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
/* Phase-aware T1 workspace reserve (GREENBOOST_T1_WORKSPACE_MB, default 0=off).
 * Mirrors g_kv_reserve_bytes but for per-step compute workspace, not KV: during
 * MODEL_LOAD this many bytes of T1 are held back so weight staging spills the
 * COLDEST weights to T2 instead of filling T1 to the floor.  The freed VRAM
 * then absorbs the transient workspace allocations of every decode/denoise step
 * (which otherwise land on T2 zero-copy DDR and dominate step time — validated
 * live in gen_art.py: denoise steps 37 s → 10 s once workspace stayed in T1).
 * Released (treated as 0) once the phase advances past MODEL_LOAD, so
 * activations/workspace claim it during INFERENCE. */
static _Atomic size_t g_workspace_reserve_bytes = 0;
/* Set to 1 when env-var or OLLAMA_NUM_CTX auto-scaler has written g_kv_reserve_bytes.
 * Prevents gb_refresh_kv_reserve() from silently clobbering that value with the
 * kernel module's kv_reserve_mb (which may be 0 on a fresh insmod). */
static int g_kv_reserve_from_env = 0;
/* Bytes of KV cache that have been allocated directly in T1 VRAM (not overflow).
 * Incremented when a large alloc that matches the quiet-gap / phase heuristic
 * succeeds in T1; decremented when that pointer is freed via CAS loop. */
static _Atomic size_t g_kv_allocated_t1_bytes;
/* N11: SWA sliding-window KV eviction - track live KV bytes in T2 overflow.
 * When g_kv_t2_live_bytes > g_swa_window_bytes (set from GREENBOOST_SWA_WINDOW),
 * proactively evict oldest KV blocks before each new T2 KV alloc. */
static _Atomic size_t g_kv_t2_live_bytes = 0;
static size_t         g_swa_window_bytes  = 0; /* 0 = disabled */

/* ── Phase-aware KV prefetch (Stage 3), GREENBOOST_KV_PREFETCH ─────────────
 * Runs on the existing 2s heat-pusher cadence (no new thread). During decode
 * (phase INFERENCE/STEADY) with KV spilled to T2 (g_kv_t2_live_bytes>0) and
 * headroom left in the T1 KV reserve, decode re-reads that T2 KV every step,
 * so promoting the hottest T2 KV blocks into the reserved T1 window ahead of
 * the next decode removes a recurring PCIe read.
 *
 * Modes (default off — provably no-op unless explicitly enabled):
 *   0/unset  disabled
 *   "stats"  measure opportunity only (counters below, NO data movement) —
 *            the safe first step; lets a real overflowing-KV run show whether
 *            the tick would fire before any hot-path copy is turned on.
 *   "1"      active promotion — GATED, pending a long-context hardware bench
 *            (a live cuMemcpyHtoDAsync into the T1 reserve is a hot-path
 *            mutation in a crash-sensitive shim; not enabled from software
 *            until validated on an overflowing context). Falls back to stats.
 */
#define GB_KV_PREFETCH_OFF    0
#define GB_KV_PREFETCH_STATS  1
#define GB_KV_PREFETCH_ACTIVE 2
static int              g_kv_prefetch_mode = GB_KV_PREFETCH_OFF;
static _Atomic uint64_t g_kv_prefetch_ticks         = 0;
static _Atomic uint64_t g_kv_prefetch_opportunities = 0;
static _Atomic uint64_t g_kv_prefetch_headroom_mb   = 0;  /* last tick's T1-reserve headroom */
static _Atomic uint64_t g_kv_prefetch_t2_kv_mb      = 0;  /* last tick's KV bytes resident in T2 */
/* CRIT-04: Must be _Atomic - incremented concurrently from multiple CUDA stream
 * threads; a plain unsigned int would be a data race (undefined behaviour in C11). */
static _Atomic unsigned int g_alloc_count = 0;
#define GB_KV_REFRESH_INTERVAL 64u  /* must be a power of 2 */

/* ENH-02: Shim-side cache of the last phase_reset_seq seen from the kernel.
 * gb_refresh_kv_reserve() compares this against gb_info.phase_reset_seq; a
 * change means Synapse CLI called GB_IOCTL_RESET_PHASE before a model swap. */
static _Atomic uint32_t g_last_phase_reset_seq = 0;

/* ENH-03: cuMemGetInfo cache - calling cuMemGetInfo on every overflow alloc
 * costs a CUDA driver round-trip (~1-2 µs each).  Cache the result and
 * refresh at most every GB_MEMINFO_REFRESH_ALLOCS allocs OR every
 * GB_MEMINFO_REFRESH_MS ms (whichever fires first). */
#define GB_MEMINFO_REFRESH_ALLOCS 16u  /* power-of-2 - cheap bitmask test */
#define GB_MEMINFO_REFRESH_MS     50ULL
static _Atomic size_t   g_cached_free_vram  = 0;
static _Atomic size_t   g_cached_total_vram = 0;
static _Atomic uint64_t g_cached_meminfo_ms = 0;

/* ------------------------------------------------------------------ */
/*  Allocation phase detector - distinguishes KV cache from weights    */
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
    GB_PHASE_DEEP_IDLE   = 5,  /* IDLE for GB_DEEP_IDLE_TIMEOUT_MS - daemon unloads models */
} gb_alloc_phase_t;

/* Phase detector globals use C11 _Atomic to prevent torn reads/writes under
 * concurrent cudaMalloc calls from multiple CUDA streams.  memory_order_relaxed
 * is sufficient because the phase heuristic is a best-effort classifier and does
 * not require sequentially consistent ordering between unrelated alloc paths. */
static _Atomic int            g_alloc_phase    /* gb_alloc_phase_t */ = GB_PHASE_INIT;
static int                    g_phase_detect   = 1;    /* GREENBOOST_PHASE_DETECT */

/* Phase-aware KV prefetch tick — runs on the 2s heat-pusher cadence. Measures
 * (and, once hardware-validated, will promote) hot KV that spilled to T2 while
 * the T1 KV reserve still has room. Provably a no-op when GREENBOOST_KV_PREFETCH
 * is unset. Defined here (after the phase enum + KV reserve globals it reads);
 * forward-declared near prefetch_worker. */
static void gb_kv_prefetch_tick(void)
{
    if (g_kv_prefetch_mode == GB_KV_PREFETCH_OFF)
        return;
    atomic_fetch_add_explicit(&g_kv_prefetch_ticks, 1, memory_order_relaxed);

    int phase = atomic_load_explicit(&g_alloc_phase, memory_order_relaxed);
    if (phase != GB_PHASE_INFERENCE && phase != GB_PHASE_STEADY)
        return;   /* only during decode; MODEL_LOAD/IDLE are owned elsewhere */

    size_t rsv  = atomic_load_explicit(&g_kv_reserve_bytes,      memory_order_relaxed);
    size_t t1   = atomic_load_explicit(&g_kv_allocated_t1_bytes, memory_order_relaxed);
    size_t t2kv = atomic_load_explicit(&g_kv_t2_live_bytes,      memory_order_relaxed);
    size_t headroom = (t1 >= rsv) ? 0 : (rsv - t1);

    atomic_store_explicit(&g_kv_prefetch_headroom_mb, headroom >> 20, memory_order_relaxed);
    atomic_store_explicit(&g_kv_prefetch_t2_kv_mb,    t2kv     >> 20, memory_order_relaxed);

    /* Opportunity = KV is resident in T2 AND the T1 reserve can pull some back.
     * When KV fits entirely in T1 (t2kv==0) or the reserve is full, this is a
     * no-op, exactly as required. */
    if (t2kv > 0 && headroom > 0) {
        atomic_fetch_add_explicit(&g_kv_prefetch_opportunities, 1, memory_order_relaxed);
        /* GB_KV_PREFETCH_ACTIVE live promotion (async H2D of the hottest T2 KV
         * blocks into the reserved T1 window) is intentionally deferred until a
         * long-context bench validates it on real overflowing KV — a hot-path
         * copy in this shim must be proven before it ships enabled. */
    }
}
/* Allocs above this threshold during INFERENCE phase are classified KV */
static size_t                 g_kv_size_threshold_bytes = 64ULL * 1024 * 1024;  /* 64 MB - catches small-model KV allocs */

/* Rolling average of overflow alloc sizes (exponential moving average,
 * α = 1/8) - used to detect the KV alloc as anomalously large. */
static _Atomic size_t        g_overflow_avg_bytes;   /* EMA of recent overflow sizes */
static _Atomic unsigned int  g_overflow_load_count;  /* allocs in MODEL_LOAD phase */

/* Timestamp of last overflow alloc - used for quiet-gap detection.
 * Stored as milliseconds from CLOCK_MONOTONIC.
 * MED-07: Renamed from g_last_overflow_ns (misleading - value was always ms). */
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

/* T2 pool cap: leave 12% of the pool as headroom for OS/desktop.
 * Applied in all phases (including MODEL_LOAD) - pool sizing in setup.sh
 * already accounts for model-load requirements. */
#define GB_T2_INFERENCE_CAP_PCT  88
/* D4: Warn threshold - begin background KV eviction at 75% T2 utilization. */
#define GB_T2_WARN_PCT           75
/* U6: Mid-pressure watermark - large allocs (>1 MB) are deferred to T3 when T2 > 82%.
 * Prevents a single large weight block from pushing T2 past CAP in one shot. */
#define GB_T2_MID_PCT            82

/* Warn when MemAvailable drops below this fraction of total RAM (checked in stats writer). */
#define GB_MEMAVAIL_WARN_PCT  12

/* U3: Eviction rate tracking (vLLM kv_cache_metrics pattern).
 * Counts T1→T2 evictions per second; high rate → bias new allocations away from
 * T1 to avoid thrashing the feeder's GPU VRAM. */
static _Atomic uint64_t g_t1_evict_count    = 0;  /* lifetime T1 overflow events */
static _Atomic uint64_t g_t1_evict_rate_ts  = 0;  /* start of current 1-s window */
static _Atomic uint32_t g_t1_evict_window   = 0;  /* events in current window     */
static _Atomic uint32_t g_t1_evict_rate     = 0;  /* rolling 1-s rate             */
#define GB_T1_THRASH_THRESHOLD   20  /* >20 evictions/s → T1 thrashing */

static void gb_evict_rate_tick(void)
{
    struct timespec _ts; clock_gettime(CLOCK_MONOTONIC, &_ts);
    uint64_t now_ms = (uint64_t)_ts.tv_sec * 1000 + (uint64_t)_ts.tv_nsec / 1000000;
    uint64_t win_start = atomic_load_explicit(&g_t1_evict_rate_ts, memory_order_relaxed);
    if (now_ms - win_start >= 1000) {
        uint32_t cnt = atomic_exchange_explicit(&g_t1_evict_window, 0, memory_order_relaxed);
        atomic_store_explicit(&g_t1_evict_rate, cnt, memory_order_relaxed);
        atomic_store_explicit(&g_t1_evict_rate_ts, now_ms, memory_order_relaxed);
    }
    atomic_fetch_add_explicit(&g_t1_evict_window, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_t1_evict_count, 1, memory_order_relaxed);
}

/* U12/U9/U11: forward declarations - defined later after ghost ring setup */
static _Atomic int       g_t2_warn_adj    = 0;
static _Atomic uint64_t  g_kv_dedup_hits  = 0;
static _Atomic uint64_t  g_cold_evict_cnt = 0;

/* Returns effective T2 pool cap in bytes.
 * Always 88% of gb_t2_pool_bytes to protect OS headroom in all phases.
 * The pool itself is sized conservatively by setup.sh (70-80% of total RAM),
 * so the effective hard limit is 62-70% of total RAM - enough for the OS. */
static inline size_t gb_effective_t2_cap(void)
{
    if (gb_t2_pool_bytes == 0) return 0;
    return gb_t2_pool_bytes * GB_T2_INFERENCE_CAP_PCT / 100;
}

/* PR-F/H3: atomic reservation pattern for the T2 cap.
 *
 * Previous code did `load -> compare -> alloc -> add` which is TOCTOU-prone:
 * two concurrent threads could both pass the compare on a relaxed load and
 * both alloc, blowing past the cap by 2× (or more under heavy contention).
 * Because the underlying memory in Path A/B/C is pinned (Path A) or anchored
 * (Path C managed UVM), the kernel cannot reclaim the over-allocation and
 * the system ends up OOM'd.
 *
 * The reservation pattern atomically commits the requested bytes against the
 * cap BEFORE the alloc.  On cap exceedance we revert the reservation and
 * return OOM.  On alloc failure we revert.  On alloc success we leave the
 * reservation committed - the matching free decrement happens on cuMemFree.
 *
 * Inner alloc helpers (gb_vmm_t2_alloc_blackwell_*) no longer increment
 * gb_t2_overflow_bytes themselves; the reservation has already done that.
 *
 * Returns 0 if reserved successfully, -1 if reservation would exceed cap. */
static inline int gb_t2_try_reserve(size_t bytes)
{
    size_t cap = gb_effective_t2_cap();
    if (cap == 0) return 0;  /* cap disabled - caller's responsibility */
    size_t prev = atomic_fetch_add_explicit(&gb_t2_overflow_bytes, bytes,
                                            memory_order_acq_rel);
    if (prev + bytes > cap) {
        atomic_fetch_sub_explicit(&gb_t2_overflow_bytes, bytes,
                                  memory_order_release);
        return -1;
    }
    return 0;
}

/* Revert a previously-successful reservation when the actual alloc call
 * failed.  Symmetric with gb_t2_try_reserve. */
static inline void gb_t2_release_reserved(size_t bytes)
{
    if (gb_t2_pool_bytes == 0) return;
    atomic_fetch_sub_explicit(&gb_t2_overflow_bytes, bytes,
                              memory_order_release);
}

/* D4/U12: Warn threshold - starts at GB_T2_WARN_PCT; dynamically adjusted by refault tracking. */
static inline size_t gb_effective_t2_warn(void)
{
    if (gb_t2_pool_bytes == 0) return 0;
    int adj = atomic_load_explicit(&g_t2_warn_adj, memory_order_relaxed);
    int pct = GB_T2_WARN_PCT + adj;
    if (pct < 60) pct = 60;
    if (pct > 90) pct = 90;
    return gb_t2_pool_bytes * (size_t)pct / 100;
}

/* U6: Mid-pressure threshold - returns 82% of pool. */
static inline size_t gb_effective_t2_mid(void)
{
    if (gb_t2_pool_bytes == 0) return 0;
    return gb_t2_pool_bytes * GB_T2_MID_PCT / 100;
}

/* D4: Check if T2 utilization has crossed the warn threshold.
 * Called from gb_maybe_write_stats() to log and potentially trigger eviction. */
static inline int gb_t2_above_warn_threshold(void)
{
    size_t warn = gb_effective_t2_warn();
    if (warn == 0) return 0;
    size_t used = atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed);
    return (used >= warn);
}

/* T2 DDR headroom to advertise as free "virtual VRAM" in cuMemGetInfo / NVML hooks.
 * = T2 pool cap − already committed (Path A/B + UVM estimate) − safety reserve.
 * T3 (NVMe) is intentionally excluded: it is capacity for model loading, not
 * fast memory suitable for KV cache; including it inflates context-length
 * calculations and causes the KV cache to exceed available system RAM (OOM). */
static size_t gb_t2_free_to_report(void)
{
    if (gb_t2_pool_bytes == 0)
        return gb_virtual_vram_bytes;   /* no T2 info - keep legacy behaviour */
    size_t committed = atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed);
    size_t cap = (gb_t2_pool_bytes > gb_safety_reserve_bytes)
                  ? (gb_t2_pool_bytes - gb_safety_reserve_bytes) : 0;
    size_t pool_free = (cap > committed) ? (cap - committed) : 0;

    /* Additionally cap by actual MemAvailable so Ollama never requests more than
     * what the OS actually has. */
    size_t mem_avail = gb_get_mem_available();
    if (mem_avail > 0 && gb_safety_reserve_bytes > 0 &&
            mem_avail > gb_safety_reserve_bytes) {
        size_t avail_free = mem_avail - gb_safety_reserve_bytes;
        if (avail_free < pool_free)
            pool_free = avail_free;
    }
    return pool_free;
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

/* ── Phase 2: K/V int8 compression (opt-in via GREENBOOST_KV_COMPRESS=1) ─── */
static int            g_kv_compress_enabled = 0;
static pthread_once_t g_ptx_init_once       = PTHREAD_ONCE_INIT;
static void          *g_saved_libcuda       = NULL;

/* ── Phase 3: GPUDirect Storage (opt-in via GREENBOOST_GDS=1) ─────────────── */
static int    g_gds_ok      = 0;
static void  *g_libcufile   = NULL;
/* cuFile function pointers resolved at init */
typedef int (*pfn_cuFileDriverOpen_t)(void);
typedef int (*pfn_cuFileHandleRegister_t)(void *, void *);
typedef long (*pfn_cuFileWrite_t)(void *, const void *, size_t, off_t, off_t);
typedef long (*pfn_cuFileRead_t)(void *, void *, size_t, off_t, off_t);
static pfn_cuFileDriverOpen_t      f_cuFileDriverOpen;
static pfn_cuFileHandleRegister_t  f_cuFileHandleRegister;
static pfn_cuFileWrite_t           f_cuFileWrite;
static pfn_cuFileRead_t            f_cuFileRead;
static void                       *g_cufile_handle_storage[64]; /* opaque cuFile handle */
static void                       *g_cufile_handle = NULL;

/* ── Phase 4: multi-tensor batch migration ─────────────────────────────────── */
#define GB_BATCH_MAX 64

struct gb_migrate_entry {
    CUdeviceptr  src;   /* source GPU pointer (T1 VRAM or fake remote) */
    void        *dst;   /* destination pinned host ptr (T2 DMA-BUF mapping) */
    size_t       size;
};

struct gb_migrate_batch {
    struct gb_migrate_entry entries[GB_BATCH_MAX];
    int                     count;
    CUstream                stream;
};

static struct gb_migrate_batch g_migrate_batch;
static pthread_mutex_t gb_migrate_lock = PTHREAD_MUTEX_INITIALIZER;

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
        /* Directory may not exist yet - try to create it (best-effort). */
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
 * gb_write_capabilities_file - write /run/greenboost/capabilities.json once at
 * shim init so consumers can discover what this shim supports WITHOUT sniffing
 * the .so binary for a log literal (the fragile pre-manifest scheme in
 * ai-forge forge/gpu.py). Read back by gb_monitor.capabilities().
 *
 * Static features (gb_quant_cudart_rebind, expert_pool, cluster_fabric) are
 * true for every build of this shim; runtime features (gds, kv_compress,
 * report_physical_vram) reflect the env this process was launched with.
 * tmp+rename so a concurrent reader never sees a half-written file.
 */
static void gb_write_capabilities_file(void)
{
    struct timespec rt;
    char buf[512];
    char tmp[] = "/run/greenboost/capabilities.json.tmp";
    int fd, n;

    clock_gettime(CLOCK_REALTIME, &rt);
    n = snprintf(buf, sizeof(buf),
        "{\n"
        "  \"shim_version\": \"%s\",\n"
        "  \"abi\": %d,\n"
        "  \"pid\": %d,\n"
        "  \"ts\": %lld,\n"
        "  \"features\": {\n"
        "    \"gb_quant_cudart_rebind\": true,\n"
        "    \"expert_pool\": true,\n"
        "    \"cluster_fabric\": true,\n"
        "    \"gds\": %s,\n"
        "    \"kv_compress\": %s,\n"
        "    \"report_physical_vram\": %s\n"
        "  }\n"
        "}\n",
        GB_SHIM_VERSION, GB_SHIM_CAP_ABI, (int)getpid(),
        (long long)rt.tv_sec,
        g_gds_ok ? "true" : "false",
        g_kv_compress_enabled ? "true" : "false",
        g_report_physical_vram ? "true" : "false");
    if (n <= 0 || n >= (int)sizeof(buf))
        return;

    fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) {
        mkdir("/run/greenboost", 0755);
        fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    }
    if (fd < 0)
        return;
    ssize_t _wr = write(fd, buf, (size_t)n);
    (void)_wr;
    close(fd);
    if (rename(tmp, "/run/greenboost/capabilities.json") != 0)
        unlink(tmp);
}

/*
 * gb_idle_flush_weights - flush non-KV hash-table entries when Ollama has been
 * idle for g_idle_timeout_ms.  KV cache entries are kept so the loaded model
 * stays warm.  Called from gb_check_idle_phase after detecting STEADY→IDLE.
 */
static void gb_idle_flush_weights(void)
{
    gb_log("Phase → IDLE - flushing weight/activation T2 overflow");
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
             * been quiet for at least ito ms, the GPU is completely idle - no model
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
                gb_log("Phase → DEEP_IDLE (GPU fully idle for %llu ms - skipping IDLE dwell) - signalling reclaim daemon",
                       (unsigned long long)(now - last));
                GB_NVTX_EVENT("PHASE_DEEP_IDLE", "PHASE", 0, 0, "skip_idle_gpu_fully_idle");
                gb_write_phase_file((int)GB_PHASE_DEEP_IDLE, now - last);
                /* Demote all our T2 buffers to LRU tail - idle session yields
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
                GB_NVTX_EVENT("PHASE_IDLE", "PHASE", 0, 0, "steady_to_idle");
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
            gb_log("Phase → DEEP_IDLE (idle for %llu ms) - signalling reclaim daemon",
                   (unsigned long long)(now - entered));
            GB_NVTX_EVENT("PHASE_DEEP_IDLE", "PHASE", 0, 0, "idle_to_deep_idle");
            gb_write_phase_file((int)GB_PHASE_DEEP_IDLE, now - entered);
            /* Already in IDLE - SESSION_IDLE was sent then; no repeat needed. */
        }
    }
}

/* Called on every overflow alloc; returns the GB_ALLOC_* flags to use.
 * Each atomic variable is accessed with memory_order_relaxed - the phase
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
        /* First overflow alloc ever - enter model loading phase */
        atomic_store_explicit(&g_alloc_phase, GB_PHASE_MODEL_LOAD, memory_order_relaxed);
        atomic_store_explicit(&g_overflow_load_count, 1u, memory_order_relaxed);
        atomic_store_explicit(&g_overflow_avg_bytes, bytesize, memory_order_relaxed);
        atomic_store_explicit(&g_last_overflow_ms, now_ms, memory_order_relaxed);
        gb_log("Phase → MODEL_LOAD (first overflow alloc, %zu MB)", bytesize >> 20);
        GB_NVTX_EVENT("PHASE_MODEL_LOAD", "PHASE", bytesize >> 20, 0, "init_to_model_load");
        return GB_ALLOC_WEIGHTS;

    case GB_PHASE_MODEL_LOAD:
        gap_ms = now_ms - atomic_load_explicit(&g_last_overflow_ms, memory_order_relaxed);
        atomic_store_explicit(&g_last_overflow_ms, now_ms, memory_order_relaxed);
        load_count = atomic_fetch_add_explicit(&g_overflow_load_count, 1u, memory_order_relaxed) + 1u;
        /* Update EMA: avg = (avg * 7 + size) / 8 */
        avg = (atomic_load_explicit(&g_overflow_avg_bytes, memory_order_relaxed) * 7 + bytesize) / 8;
        atomic_store_explicit(&g_overflow_avg_bytes, avg, memory_order_relaxed);

        /* Transition to INFERENCE when either:
         *   (a) quiet gap: no overflow for >= GB_PHASE_QUIET_GAP_MS AND at least
         *       8 weight allocs seen — prevents GGUF mmap I/O pauses (>400 ms)
         *       during the first few tensor reads from flipping the phase early,
         *       which would apply the full KV reserve to weight allocs and strand
         *       ~2 GB of T1 VRAM without any KV actually allocated yet.
         *   (b) this alloc is >= 4x the rolling average (KV signature)
         *   (c) forced after GB_PHASE_LOAD_COUNT_MAX weight allocs             */
        if ((gap_ms >= GB_PHASE_QUIET_GAP_MS && load_count >= 8) ||
            (avg > 0 && bytesize >= 4 * avg) ||
            load_count >= GB_PHASE_LOAD_COUNT_MAX) {

            atomic_store_explicit(&g_alloc_phase, GB_PHASE_INFERENCE, memory_order_relaxed);
            gb_log("Phase → INFERENCE (gap=%llums, avg=%zuMB, count=%u, this=%zuMB)",
                   (unsigned long long)gap_ms,
                   avg >> 20, load_count, bytesize >> 20);
            GB_NVTX_EVENT("PHASE_INFERENCE", "PHASE", bytesize >> 20, 0, "model_load_to_inference");
            {
                size_t _ws = atomic_load_explicit(&g_workspace_reserve_bytes, memory_order_relaxed);
                if (_ws > 0)
                    GB_NVTX_EVENT("T1_WORKSPACE_RELEASE", "PHASE", _ws >> 20, 0,
                                  "released_at_inference");
            }

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
            gb_log("Phase → MODEL_LOAD reset (gap=%llums, small alloc %zu MB - likely model reload)",
                   (unsigned long long)gap_ms, bytesize >> 20);
            return GB_ALLOC_WEIGHTS;
        }

        /* Large allocs in INFERENCE phase = KV cache */
        if (bytesize >= g_kv_size_threshold_bytes) {
            gb_log("Phase classify: INFERENCE KV alloc %zu MB → GB_ALLOC_KV_CACHE|T1_PRIORITY",
                   bytesize >> 20);
            /* After 2 KV allocs (K + V tensors), enter STEADY */
            atomic_store_explicit(&g_alloc_phase, GB_PHASE_STEADY, memory_order_relaxed);
            GB_NVTX_EVENT("PHASE_STEADY", "PHASE", bytesize >> 20, 0, "inference_to_steady_kv_placed");
            return GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY;
        }
        gb_log("Phase classify: INFERENCE small alloc %zu MB → GB_ALLOC_ACTIVATIONS",
               bytesize >> 20);
        return GB_ALLOC_ACTIVATIONS;

    case GB_PHASE_STEADY:
        /* Generation loop: small ephemeral activation buffers */
        if (bytesize >= g_kv_size_threshold_bytes) {
            /* Unexpected large alloc in steady state - could be a new KV context */
            gb_log("Phase classify: STEADY large alloc %zu MB → GB_ALLOC_KV_CACHE|T1_PRIORITY",
                   bytesize >> 20);
            return GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY;
        }
        return GB_ALLOC_ACTIVATIONS;

    case GB_PHASE_IDLE:
    case GB_PHASE_DEEP_IDLE:
        /* New overflow alloc after idle/deep-idle - model is being reloaded.
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
        GB_NVTX_EVENT("PHASE_MODEL_LOAD", "PHASE", bytesize >> 20, 0,
            phase == (int)GB_PHASE_DEEP_IDLE ? "deep_idle_to_model_load" : "idle_to_model_load");
        gb_write_phase_file((int)GB_PHASE_MODEL_LOAD, 0);
        /* Session is active again - promote our T2 buffers back to LRU head
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

        /* ENH-02: Phase reset detection - Synapse CLI calls GB_IOCTL_RESET_PHASE
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
                "[GreenBoost] WARNING: T3 safety-net active - %llu MB of model data "
                "is on NVMe (slow). Inference will be slow. "
                "Reduce num_ctx or use a smaller model.\n",
                (unsigned long long)info.nvme_t3_allocated_mb);
        }
    }

}

/* ------------------------------------------------------------------ */
/*  Path A zero-copy sub-method - DMA-BUF import via cudaImportExternalMemory  */
/* ------------------------------------------------------------------ */

/*
 * Path A (zero-copy sub-method) - true zero-copy DMA-BUF import.
 *
 * Flow: GB_IOCTL_ALLOC (kernel allocates hugepage-backed buffer, exports DMA-BUF fd)
 *       → cudaImportExternalMemory(OpaqueFd)  [CUDA takes fd ownership]
 *       → cudaExternalMemoryGetMappedBuffer   [returns CUdeviceptr directly]
 *
 * Advantages over the pinned sub-method (cuMemHostRegister):
 *   - No anonymous mmap() + pin_user_pages() round-trip.
 *   - No cuMemHostRegister overhead; CUDA drives its own IOMMU mapping from
 *     the kernel DMA-BUF scatter-gather table (2 MB compound pages).
 *   - The returned CUdeviceptr is a first-class device pointer, not a
 *     host-registered alias - no CUDA driver internal remapping needed.
 *   - cudaDestroyExternalMemory() handles full teardown on free.
 *
 * Requires: real_cudaImportExternalMemory and real_cudaExternalMemoryGetMappedBuffer
 *           resolved from libcudart.so (CUDA runtime ≥ 10.0).
 * Skipped on Blackwell (CC ≥ 12): cudaImportExternalMemory(OpaqueFd) not supported.
 */
/* Path A (DMA-BUF pinned DDR) - unified counter covering both zero-copy and
 * pinned sub-methods.  Readers only see "Path A"; which sub-method ran is an
 * internal detail reported in the debug log line ("zero-copy" vs "pinned"). */
static volatile unsigned int gb_path_a_count  = 0;
static volatile unsigned int gb_path_b_count  = 0;
static _Atomic int gb_bvmm_zerocopy_count = 0; /* active ZC T2 allocations on Blackwell */

/* Set to 1 on first cudaImportExternalMemory error or at init on CC ≥ 12 to
 * permanently skip the zero-copy sub-method.  Prevents sticky CUDA error 999
 * from corrupting the CUDA context on every alloc. */
static volatile int gb_a0_disabled = 0;

/* Set to 1 (default on cc >= 12) to refuse T2 overflow entirely on Blackwell.
 * Blackwell desktop PCIe T2 paths (zerocopy / HOST_NUMA) return DMA-only fabric
 * pointers that GPU compute SMs cannot dereference - any kernel touching a T2
 * pointer will fail with CUDA_ERROR_INVALID_RESOURCE_HANDLE (400). */
static volatile int gb_disable_t2_on_blackwell = 0;

/* Minimum allocation size for the Path A zero-copy sub-method.
 * For < 4 MB (< 2 hugepages), the DMA-BUF export+import overhead is
 * disproportionate; the pinned sub-method (cuMemHostRegister) is cheaper. */
#define GB_PATH_A0_MIN_BYTES  (4ULL * 1024 * 1024)

/* ------------------------------------------------------------------ */
/*  Shim stats file - written to /run/greenboost/shim_stats            */
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

    /* Try /run/greenboost first (preferred - readable by status script) */
    if (mkdir(GB_STATS_DIR, 0777) == 0 || errno == EEXIST) {
        /* chmod in case directory already existed with wrong perms */
        chmod(GB_STATS_DIR, 0777);
        /* Test writeability */
        char probe[128];
        snprintf(probe, sizeof(probe), "%s/.probe.%d", GB_STATS_DIR, (int)getpid());
        FILE *fp = fopen(probe, "w");
        if (fp) { fclose(fp); unlink(probe); gb_stats_dir = GB_STATS_DIR; gb_stats_file = GB_STATS_FILE; return; }
    }
    /* Fall back to /tmp - always writable */
    gb_stats_dir  = "/tmp";
    gb_stats_file = GB_STATS_FILE_TMP;
}

/* ================================================================== */
/*  U9: Block Hash Prefix Cache + Bloom Filter                        */
/*  (vLLM BlockPool.BlockHashToBlockMap pattern)                      */
/*                                                                    */
/*  Deduplicates identical KV prefix blocks across requests.          */
/*  Bloom filter gives O(1) miss detection; full table only on hit.   */
/* ================================================================== */

#define GB_BLOOM_BITS     16384
#define GB_BLOOM_HASHES   4
#define GB_KV_BLOCKS_MAX  2048

typedef struct {
    uint64_t block_hash;    /* 64-bit hash of first 256 bytes of block data */
    uint64_t parent_hash;   /* U9: parent block hash (0 if root of prefix)   */
    uint64_t dev_ptr;       /* device pointer (fake or real) for this block */
    size_t   size;          /* block size in bytes                           */
    uint32_t ref_cnt;       /* number of requests currently holding block    */
    uint32_t access_ts;     /* last-access second (CLOCK_MONOTONIC)          */
    uint32_t num_tokens;    /* U9: cumulative token count up to this block   */
    uint32_t _pad;
} gb_kv_block_t;

typedef struct {
    gb_kv_block_t   blocks[GB_KV_BLOCKS_MAX];
    int             nblocks;
    uint8_t         bloom[GB_BLOOM_BITS / 8];
    pthread_rwlock_t lock;
} gb_kv_cache_t;

static gb_kv_cache_t g_kv_cache = { .lock = PTHREAD_RWLOCK_INITIALIZER };

/* 64-bit xxhash-inspired fast hash - no external dependency */
static inline uint64_t gb_xxhash64(const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 0x9E3779B97F4A7C15ULL ^ (uint64_t)len;
    size_t i;
    for (i = 0; i + 8 <= len; i += 8) {
        uint64_t v; memcpy(&v, p + i, 8);
        h ^= v * 0xBF58476D1CE4E5B9ULL;
        h = (h << 31) | (h >> 33);
        h *= 0x94D049BB133111EBULL;
    }
    for (; i < len; i++) { h ^= (uint64_t)p[i]; h *= 0x9E3779B97F4A7C15ULL; }
    h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 27; h *= 0x94D049BB133111EBULL;
    h ^= h >> 31;
    return h;
}

static inline void gb_bloom_set(uint8_t *bloom, uint64_t hash)
{
    for (int i = 0; i < GB_BLOOM_HASHES; i++) {
        uint32_t bit = (uint32_t)((hash >> (i * 16)) & (GB_BLOOM_BITS - 1));
        bloom[bit / 8] |= (uint8_t)(1u << (bit & 7));
        hash = (hash << 13) | (hash >> 51);
    }
}

static inline int gb_bloom_test(const uint8_t *bloom, uint64_t hash)
{
    for (int i = 0; i < GB_BLOOM_HASHES; i++) {
        uint32_t bit = (uint32_t)((hash >> (i * 16)) & (GB_BLOOM_BITS - 1));
        if (!(bloom[bit / 8] & (uint8_t)(1u << (bit & 7)))) return 0;
        hash = (hash << 13) | (hash >> 51);
    }
    return 1;
}

/* Register a newly allocated KV block in the prefix cache.
 * parent_hash links this block into the prefix ancestry tree (0 if root).
 * num_tokens is the cumulative token count up to this block.
 * Returns 1 if an existing block with the same hash was found (dedup),
 * 0 if inserted fresh.  Caller should use *out_ptr if returns 1. */
static int gb_kv_cache_insert(const void *host_ptr, size_t size,
                               uint64_t dev_ptr, uint64_t *out_ptr,
                               uint64_t parent_hash, uint32_t num_tokens)
{
    if (!host_ptr || size < 256) return 0;
    size_t sample = (size < 256) ? size : 256;
    uint64_t hash = gb_xxhash64(host_ptr, sample);

    pthread_rwlock_rdlock(&g_kv_cache.lock);
    int bloom_hit = gb_bloom_test(g_kv_cache.bloom, hash);
    if (bloom_hit) {
        for (int i = 0; i < g_kv_cache.nblocks; i++) {
            if (g_kv_cache.blocks[i].block_hash == hash &&
                g_kv_cache.blocks[i].size == size) {
                g_kv_cache.blocks[i].ref_cnt++;
                g_kv_cache.blocks[i].access_ts = (uint32_t)time(NULL);
                if (out_ptr) *out_ptr = g_kv_cache.blocks[i].dev_ptr;
                pthread_rwlock_unlock(&g_kv_cache.lock);
                atomic_fetch_add_explicit(&g_kv_dedup_hits, 1, memory_order_relaxed);
                return 1; /* dedup hit */
            }
        }
    }
    pthread_rwlock_unlock(&g_kv_cache.lock);

    pthread_rwlock_wrlock(&g_kv_cache.lock);
    if (g_kv_cache.nblocks < GB_KV_BLOCKS_MAX) {
        gb_kv_block_t *b = &g_kv_cache.blocks[g_kv_cache.nblocks++];
        b->block_hash  = hash;
        b->parent_hash = parent_hash;
        b->dev_ptr     = dev_ptr;
        b->size        = size;
        b->ref_cnt     = 1;
        b->access_ts   = (uint32_t)time(NULL);
        b->num_tokens  = num_tokens;
        gb_bloom_set(g_kv_cache.bloom, hash);
    } else {
        /* Evict the oldest block (lowest access_ts) */
        int oldest = 0;
        for (int i = 1; i < g_kv_cache.nblocks; i++)
            if (g_kv_cache.blocks[i].access_ts < g_kv_cache.blocks[oldest].access_ts)
                oldest = i;
        g_kv_cache.blocks[oldest].block_hash  = hash;
        g_kv_cache.blocks[oldest].parent_hash = parent_hash;
        g_kv_cache.blocks[oldest].dev_ptr     = dev_ptr;
        g_kv_cache.blocks[oldest].size        = size;
        g_kv_cache.blocks[oldest].ref_cnt     = 1;
        g_kv_cache.blocks[oldest].access_ts   = (uint32_t)time(NULL);
        g_kv_cache.blocks[oldest].num_tokens  = num_tokens;
        /* Don't reset Bloom - let it be a false-positive cache; rebuild at flush */
        gb_bloom_set(g_kv_cache.bloom, hash);
    }
    pthread_rwlock_unlock(&g_kv_cache.lock);
    return 0;
}

/* Release a reference to a KV block. Called on cudaFree of KV-flagged entries. */
static void gb_kv_cache_release(uint64_t dev_ptr)
{
    pthread_rwlock_wrlock(&g_kv_cache.lock);
    for (int i = 0; i < g_kv_cache.nblocks; i++) {
        if (g_kv_cache.blocks[i].dev_ptr == dev_ptr) {
            if (g_kv_cache.blocks[i].ref_cnt > 0)
                g_kv_cache.blocks[i].ref_cnt--;
            break;
        }
    }
    pthread_rwlock_unlock(&g_kv_cache.lock);
}

/* ================================================================== */
/*  U10: KV Block Event Queue (vLLM kv_events.py pattern)             */
/*                                                                    */
/*  Host emits STORED/REMOVED events to feeders so they can build a   */
/*  block ancestry tree and prefetch on request start.                */
/* ================================================================== */



#define GB_BLOCK_EVT_QUEUE_SIZE 256

typedef struct {
    uint64_t block_hash;
    uint64_t parent_hash;
    uint32_t num_tokens;
    uint8_t  event_type;
} gb_block_evt_local_t;

static gb_block_evt_local_t g_block_evt_queue[GB_BLOCK_EVT_QUEUE_SIZE];
static int                  g_block_evt_head = 0;
static int                  g_block_evt_tail = 0;
static pthread_mutex_t      g_block_evt_lock = PTHREAD_MUTEX_INITIALIZER;

/* Enqueue a block event. Called when KV blocks are allocated or freed.
 * parent_hash is 0 for the first block of a prefix sequence. */
static void gb_block_evt_emit(uint8_t type, uint64_t hash,
                               uint64_t parent_hash, uint32_t num_tokens)
{
    pthread_mutex_lock(&g_block_evt_lock);
    int next = (g_block_evt_head + 1) % GB_BLOCK_EVT_QUEUE_SIZE;
    if (next != g_block_evt_tail) {
        g_block_evt_queue[g_block_evt_head].block_hash  = hash;
        g_block_evt_queue[g_block_evt_head].parent_hash = parent_hash;
        g_block_evt_queue[g_block_evt_head].num_tokens  = num_tokens;
        g_block_evt_queue[g_block_evt_head].event_type  = type;
        g_block_evt_head = next;
    }
    pthread_mutex_unlock(&g_block_evt_lock);
}

/* Flush pending block events to all connected feeders.
 * Called from gb_maybe_write_stats() at each stats tick. */
static void gb_block_evt_flush(void)
{
    if (!gb_netc_is_active()) return;

    pthread_mutex_lock(&g_block_evt_lock);
    if (g_block_evt_head == g_block_evt_tail) {
        pthread_mutex_unlock(&g_block_evt_lock);
        return;
    }

    struct gb_net_block_events msg;
    memset(&msg, 0, sizeof(msg));
    int count = 0;
    uint32_t now_sec = (uint32_t)time(NULL);

    while (g_block_evt_tail != g_block_evt_head && count < GB_BLOCK_EVENTS_MAX) {
        gb_block_evt_local_t *ev = &g_block_evt_queue[g_block_evt_tail];
        msg.events[count].block_hash  = ev->block_hash;
        msg.events[count].parent_hash = ev->parent_hash;
        msg.events[count].num_tokens  = ev->num_tokens;
        msg.events[count].event_type  = ev->event_type;
        msg.events[count].timestamp   = now_sec;
        count++;
        g_block_evt_tail = (g_block_evt_tail + 1) % GB_BLOCK_EVT_QUEUE_SIZE;
    }
    msg.count = (uint32_t)count;
    pthread_mutex_unlock(&g_block_evt_lock);

    /* Fire-and-forget to all feeders - feeder uses this for prefetch planning */
    int n = gb_netc_remote_gpu_count();
    for (int ri = 0; ri < n; ri++)
        gb_netc_send_block_events(ri, &msg);
}

/* ================================================================== */
/*  U11: Hot/Cold Epoch Tracking (Ray object store pattern)           */
/*                                                                    */
/*  Tracks ref-count delta per 1-s epoch. Entries with Δref=0 for    */
/*  N epochs move to "cold" tier; evicted preferentially by ARC.     */
/* ================================================================== */

#define GB_EPOCH_COLD_THRESHOLD  3   /* epochs with Δref=0 → cold */

/* Per-entry cold epoch counter - stored in a parallel array indexed
 * by the htable slot index so we don't grow gb_ht_entry_t further. */
static uint8_t  g_ht_cold_epochs[HT_SIZE];   /* 0 = hot, N = cold for N epochs */
static uint32_t g_ht_last_ref_count[HT_SIZE]; /* snapshot of access_count at last epoch tick */
static _Atomic uint64_t g_epoch_last_tick_ms = 0;

/* Called once per second from gb_maybe_write_stats() to advance cold epochs. */
static void gb_epoch_tick(uint64_t now_ms)
{
    uint64_t last = atomic_load_explicit(&g_epoch_last_tick_ms, memory_order_relaxed);
    if (now_ms - last < 1000) return;
    if (!atomic_compare_exchange_strong_explicit(&g_epoch_last_tick_ms, &last, now_ms,
                                                  memory_order_relaxed, memory_order_relaxed))
        return;

    for (uint32_t i = 0; i < HT_SIZE; i++) {
        gb_ht_entry_t *e = &gb_htable[i];
        if (e->ptr == 0 || e->ptr == HT_TOMBSTONE) {
            g_ht_cold_epochs[i] = 0;
            g_ht_last_ref_count[i] = 0;
            continue;
        }
        uint16_t cur = e->access_count;
        if (cur == g_ht_last_ref_count[i]) {
            /* no new accesses this epoch → increment cold counter */
            if (g_ht_cold_epochs[i] < 255) g_ht_cold_epochs[i]++;
        } else {
            g_ht_cold_epochs[i] = 0; /* accessed → reset to hot */
        }
        g_ht_last_ref_count[i] = cur;
    }
}

/* Returns 1 if the htable slot at index is "cold" (inactive for ≥ threshold epochs). */
static inline int gb_ht_is_cold(uint32_t idx)
{
    return g_ht_cold_epochs[idx] >= GB_EPOCH_COLD_THRESHOLD;
}

/* ================================================================== */
/*  U12: Refault Distance Tracking (Linux vmscan workingset pattern)  */
/*                                                                    */
/*  Records evicted pointer + timestamp in a ghost ring.  On          */
/*  re-access, compares refault distance to rolling eviction mean     */
/*  and adjusts the T2 warn watermark dynamically.                    */
/* ================================================================== */

#define GB_GHOST_MAX  512

typedef struct {
    uint64_t ptr_key;      /* evicted CUdeviceptr as key           */
    uint32_t evict_ts_s;   /* eviction timestamp (seconds)         */
    uint32_t size_kb;      /* evicted size in KB                   */
} gb_ghost_entry_t;

static gb_ghost_entry_t  g_ghosts[GB_GHOST_MAX];
static _Atomic uint32_t  g_ghost_head     = 0;
static _Atomic uint32_t  g_evict_dist_s   = 10; /* rolling mean eviction lifetime (s) */
/* g_t2_warn_adj, g_kv_dedup_hits, g_cold_evict_cnt - declared earlier (forward decl) */

/* Record a ghost entry when a block is evicted from T2. */
static void gb_ghost_record(uint64_t ptr, size_t size)
{
    uint32_t head = atomic_fetch_add_explicit(&g_ghost_head, 1, memory_order_relaxed)
                    % GB_GHOST_MAX;
    struct timespec _ts;
    clock_gettime(CLOCK_MONOTONIC, &_ts);
    g_ghosts[head].ptr_key    = ptr;
    g_ghosts[head].evict_ts_s = (uint32_t)_ts.tv_sec;
    g_ghosts[head].size_kb    = (uint32_t)(size / 1024);
}

/* Check ghost ring on re-access; tune g_t2_warn_adj if refault detected. */
static void gb_refault_check(uint64_t ptr)
{
    struct timespec _ts;
    clock_gettime(CLOCK_MONOTONIC, &_ts);
    uint32_t now_s = (uint32_t)_ts.tv_sec;

    for (int _i = 0; _i < GB_GHOST_MAX; _i++) {
        if (g_ghosts[_i].ptr_key != ptr) continue;
        uint32_t refault_dist = now_s - g_ghosts[_i].evict_ts_s;
        uint32_t mean = atomic_load_explicit(&g_evict_dist_s, memory_order_relaxed);
        /* Update rolling mean: EWMA with α = 1/8 */
        uint32_t new_mean = (mean * 7 + refault_dist) / 8;
        atomic_store_explicit(&g_evict_dist_s, new_mean, memory_order_relaxed);
        /* Adjust warn threshold */
        int adj = atomic_load_explicit(&g_t2_warn_adj, memory_order_relaxed);
        if (refault_dist < mean) {
            /* Evicted too early → raise warn threshold */
            if (adj < 10) atomic_fetch_add_explicit(&g_t2_warn_adj, 1, memory_order_relaxed);
        } else if (refault_dist > mean * 2) {
            /* Evicted late / truly cold → lower warn threshold */
            if (adj > -5) atomic_fetch_sub_explicit(&g_t2_warn_adj, 1, memory_order_relaxed);
        }
        g_ghosts[_i].ptr_key = 0; /* clear ghost */
        break;
    }
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

    unsigned int a  = gb_path_a_count;
    unsigned int b  = gb_path_b_count;
    size_t t2_ovf = atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed);
    int zc_cnt    = atomic_load_explicit(&gb_bvmm_zerocopy_count, memory_order_relaxed);
    const char *active =
        (a      > 0) ? "A"                  :
        (b      > 0) ? "B"                  :
        (zc_cnt > 0) ? "blackwell_zerocopy" :
        (t2_ovf > 0) ? "blackwell_vmm"      : "none";

    static const char *const _phase_names[] = {
        "INIT", "MODEL_LOAD", "INFERENCE", "STEADY", "IDLE", "DEEP_IDLE"
    };
    int  _phase_idx  = atomic_load_explicit(&g_alloc_phase, memory_order_relaxed);
    if (_phase_idx < 0 || _phase_idx > 5) _phase_idx = 0;
    size_t _kv_rsv   = atomic_load_explicit(&g_kv_reserve_bytes,      memory_order_relaxed);
    size_t _kv_t1    = atomic_load_explicit(&g_kv_allocated_t1_bytes,  memory_order_relaxed);
    size_t _kv_eff   = (_kv_t1 >= _kv_rsv) ? 0 : (_kv_rsv - _kv_t1);

    static const char *const _tier_names[GB_TIER_COUNT] = {
        "t1_local", "t1_feeder", "t2_local", "t2_feeder",
        "t3_local", "t3_feeder", "vmm", "path_b"
    };

    fprintf(f, "pid=%d\n",                    (int)getpid());
    fprintf(f, "path_a_count=%u\n",           a);
    fprintf(f, "path_b_count=%u\n",           b);
    fprintf(f, "initialized=%d\n",            initialized);
    fprintf(f, "vllm_compat=%d\n",            getenv("GREENBOOST_VLLM_COMPAT") ? 1 : 0);
    fprintf(f, "virtual_vram_mb=%zu\n",           gb_virtual_vram_bytes >> 20);
    fprintf(f, "physical_vram_mb=%zu\n",          gb_physical_vram_bytes >> 20);
    fprintf(f, "cluster_remote_vram_mb=%zu\n",    g_cluster_remote_vram_bytes >> 20);
    fprintf(f, "cluster_virtual_vram_mb=%zu\n",   (gb_physical_vram_bytes + g_cluster_remote_vram_bytes) >> 20);
    fprintf(f, "active_path=%s\n",                active);
    fprintf(f, "phase=%s\n",                  _phase_names[_phase_idx]);
    fprintf(f, "kv_reserve_nominal_mb=%zu\n", _kv_rsv >> 20);
    fprintf(f, "kv_reserve_effective_mb=%zu\n", _kv_eff >> 20);
    fprintf(f, "kv_t1_tracked_mb=%zu\n",      _kv_t1 >> 20);
    fprintf(f, "kv_prefetch_mode=%d\n",       g_kv_prefetch_mode);
    fprintf(f, "kv_prefetch_ticks=%llu\n",
            (unsigned long long)atomic_load_explicit(&g_kv_prefetch_ticks, memory_order_relaxed));
    fprintf(f, "kv_prefetch_opportunities=%llu\n",
            (unsigned long long)atomic_load_explicit(&g_kv_prefetch_opportunities, memory_order_relaxed));
    fprintf(f, "kv_prefetch_headroom_mb=%llu\n",
            (unsigned long long)atomic_load_explicit(&g_kv_prefetch_headroom_mb, memory_order_relaxed));
    fprintf(f, "kv_prefetch_t2_kv_mb=%llu\n",
            (unsigned long long)atomic_load_explicit(&g_kv_prefetch_t2_kv_mb, memory_order_relaxed));
    {
        size_t _ws_rsv = atomic_load_explicit(&g_workspace_reserve_bytes, memory_order_relaxed);
        int _ws_phase = atomic_load_explicit(&g_alloc_phase, memory_order_relaxed);
        fprintf(f, "t1_workspace_reserve_mb=%zu\n", _ws_rsv >> 20);
        fprintf(f, "t1_workspace_reserve_eff_mb=%zu\n",
                (_ws_phase <= (int)GB_PHASE_MODEL_LOAD) ? (_ws_rsv >> 20) : (size_t)0);
    }
    fprintf(f, "vram_headroom_mb=%zu\n",       vram_headroom_bytes >> 20);
    /* P1b: publish workstation reserve so gb_vitals_helper + greenboost vitals
     * can show how much VRAM headroom the shim is holding back.
     * Effective = base × 2 when gaming_mode is active.
     * Inline the sysfs read (read_sysfs_bool is static, defined later). */
    {
        int _gm = 0;
        FILE *_gmf = fopen("/sys/module/greenboost/parameters/gaming_mode", "r");
        if (_gmf) {
            char _buf[8] = {0};
            if (fread(_buf, 1, 7, _gmf) > 0) _gm = (_buf[0] == '1');
            fclose(_gmf);
        }
        size_t _cached_free  = atomic_load_explicit(&g_cached_free_vram,  memory_order_relaxed);
        size_t _cached_total = atomic_load_explicit(&g_cached_total_vram, memory_order_relaxed);
        size_t _ws_base = gb_effective_workstation_reserve_bytes(_cached_free, _cached_total);
        fprintf(f, "workstation_reserve_mb=%zu\n",     _ws_base >> 20);
        fprintf(f, "workstation_reserve_eff_mb=%zu\n", (_gm ? _ws_base * 2 : _ws_base) >> 20);
    }
    fprintf(f, "local_t1_alloc_mb=%zu\n",     atomic_load_explicit(&g_local_t1_alloc_bytes, memory_order_relaxed) >> 20);
    fprintf(f, "remote_alloc_count=%zu\n",    atomic_load_explicit(&g_remote_alloc_count, memory_order_relaxed));
    fprintf(f, "remote_alloc_mb=%zu\n",       atomic_load_explicit(&g_remote_alloc_mb, memory_order_relaxed));
    fprintf(f, "h2d_mb=%zu\n",                atomic_load_explicit(&g_h2d_mb, memory_order_relaxed));
    fprintf(f, "d2h_mb=%zu\n",                atomic_load_explicit(&g_d2h_mb, memory_order_relaxed));
    fprintf(f, "kernel_dispatch_count=%zu\n", atomic_load_explicit(&g_kernel_dispatch_count, memory_order_relaxed));
    /* R3: per-tier stats */
    for (int _ti = 0; _ti < GB_TIER_COUNT; _ti++) {
        fprintf(f, "tier_%s_cur_mb=%lld\n",      _tier_names[_ti],
                (long long)(atomic_load_explicit(&g_tier_stats[_ti].current_bytes,  memory_order_relaxed) >> 20));
        fprintf(f, "tier_%s_peak_mb=%lld\n",     _tier_names[_ti],
                (long long)(atomic_load_explicit(&g_tier_stats[_ti].peak_bytes,     memory_order_relaxed) >> 20));
        fprintf(f, "tier_%s_lifetime_mb=%lld\n", _tier_names[_ti],
                (long long)(atomic_load_explicit(&g_tier_stats[_ti].lifetime_bytes, memory_order_relaxed) >> 20));
        fprintf(f, "tier_%s_alloc_count=%lld\n", _tier_names[_ti],
                (long long) atomic_load_explicit(&g_tier_stats[_ti].alloc_count,    memory_order_relaxed));
    }
    /* R1: pool fragmentation */
    fprintf(f, "t2_pool_frag_pct=%d\n",   gb_pool_fragmentation());
    /* D4: T2 warn threshold */
    fprintf(f, "t2_above_warn=%d\n",      gb_t2_above_warn_threshold());
    fprintf(f, "t2_warn_threshold_mb=%zu\n", gb_effective_t2_warn() >> 20);
    /* U9–U15: P1 enhancement metrics */
    fprintf(f, "t2_warn_adj_pct=%d\n",    atomic_load_explicit(&g_t2_warn_adj, memory_order_relaxed));
    fprintf(f, "cold_epoch_evict_count=%llu\n", (unsigned long long)atomic_load_explicit(&g_cold_evict_cnt, memory_order_relaxed));
    fprintf(f, "kv_dedup_hits=%llu\n",    (unsigned long long)atomic_load_explicit(&g_kv_dedup_hits, memory_order_relaxed));
    fprintf(f, "kv_internal_frag_mb=%zu\n", atomic_load_explicit(&g_kv_internal_frag_bytes, memory_order_relaxed) >> 20);
    if (gb_netc_is_active())
        fprintf(f, "pinned_pool_bufs_free=%d\n", gb_netc_pinned_free_count());
    fprintf(f, "timestamp=%ld\n",             (long)time(NULL));
    fclose(f);
    rename(tmp_path, gb_stats_file);

    /* D5: also write JSON metrics for Prometheus exporter */
    {
        char json_tmp[128], json_path[128];
        snprintf(json_path, sizeof(json_path), "%s/metrics.json", gb_stats_dir);
        snprintf(json_tmp,  sizeof(json_tmp),  "%s/.metrics.json.tmp.%d", gb_stats_dir, (int)getpid());
        FILE *jf = fopen(json_tmp, "w");
        if (jf) {
            fprintf(jf, "{\n");
            fprintf(jf, "  \"pid\": %d,\n", (int)getpid());
            fprintf(jf, "  \"timestamp\": %ld,\n", (long)time(NULL));
            fprintf(jf, "  \"phase\": \"%s\",\n", _phase_names[_phase_idx]);
            fprintf(jf, "  \"physical_vram_mb\": %zu,\n", gb_physical_vram_bytes >> 20);
            fprintf(jf, "  \"cluster_remote_vram_mb\": %zu,\n", g_cluster_remote_vram_bytes >> 20);
            fprintf(jf, "  \"cluster_virtual_vram_mb\": %zu,\n",
                    (gb_physical_vram_bytes + g_cluster_remote_vram_bytes) >> 20);
            fprintf(jf, "  \"local_t1_alloc_mb\": %zu,\n",
                    atomic_load_explicit(&g_local_t1_alloc_bytes, memory_order_relaxed) >> 20);
            fprintf(jf, "  \"remote_alloc_count\": %zu,\n",
                    atomic_load_explicit(&g_remote_alloc_count, memory_order_relaxed));
            fprintf(jf, "  \"kernel_dispatch_count\": %zu,\n",
                    atomic_load_explicit(&g_kernel_dispatch_count, memory_order_relaxed));
            fprintf(jf, "  \"t2_pool_frag_pct\": %d,\n", gb_pool_fragmentation());
            fprintf(jf, "  \"t2_above_warn\": %d,\n", gb_t2_above_warn_threshold());
            fprintf(jf, "  \"tiers\": {\n");
            for (int _ti = 0; _ti < GB_TIER_COUNT; _ti++) {
                fprintf(jf, "    \"%s\": {\"cur_mb\": %lld, \"peak_mb\": %lld, "
                        "\"lifetime_mb\": %lld, \"alloc_count\": %lld}%s\n",
                        _tier_names[_ti],
                        (long long)(atomic_load_explicit(&g_tier_stats[_ti].current_bytes,  memory_order_relaxed) >> 20),
                        (long long)(atomic_load_explicit(&g_tier_stats[_ti].peak_bytes,     memory_order_relaxed) >> 20),
                        (long long)(atomic_load_explicit(&g_tier_stats[_ti].lifetime_bytes, memory_order_relaxed) >> 20),
                        (long long) atomic_load_explicit(&g_tier_stats[_ti].alloc_count,    memory_order_relaxed),
                        (_ti < GB_TIER_COUNT - 1) ? "," : "");
            }
            fprintf(jf, "  },\n");
            /* M1: per-feeder and aggregate metrics for new Prometheus labels */
            fprintf(jf, "  \"fake_ptr_generation\": %u,\n",
                    gb_netc_fake_ptr_generation());
            fprintf(jf, "  \"double_buffer_enabled\": %d,\n",
                    g_double_buffer_enabled);
            fprintf(jf, "  \"kv_compress_enabled\": %d,\n",
                    g_kv_compress_enabled);
            if (gb_netc_is_active()) {
                int _nr = gb_netc_remote_gpu_count();
                fprintf(jf, "  \"feeders\": [\n");
                for (int _fi = 0; _fi < _nr; _fi++) {
                    const char *_fname = gb_netc_feeder_addr(_fi);
                    fprintf(jf, "    {\"feeder\": \"%s\","
                            " \"bw_measured_mbs\": %u,"
                            " \"heartbeat_miss_total\": %u,"
                            " \"health_state\": %d,"
                            " \"throttled\": %d,"
                            " \"t1_quarantined\": %d,"
                            " \"gpu_util_pct\": %u,"
                            " \"gpu_mem_util_pct\": %u,"
                            " \"gpu_temp_c\": %u,"
                            " \"gpu_power_w\": %u}%s\n",
                            _fname ? _fname : "unknown",
                            gb_netc_feeder_pcie_bw_mbs(_fi),
                            gb_netc_heartbeat_miss_count(_fi),
                            gb_netc_feeder_health_state(_fi),
                            gb_netc_feeder_throttled(_fi),
                            gb_netc_feeder_t1_quarantined(_fi),
                            gb_netc_feeder_gpu_util_pct(_fi),
                            gb_netc_feeder_gpu_mem_util_pct(_fi),
                            (unsigned)gb_netc_feeder_gpu_temp_c(_fi),
                            (unsigned)gb_netc_feeder_gpu_power_w(_fi),
                            (_fi < _nr - 1) ? "," : "");
                }
                fprintf(jf, "  ]\n}\n");
            } else {
                fprintf(jf, "  \"feeders\": []\n}\n");
            }
            fclose(jf);
            rename(json_tmp, json_path);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  gb_read_control() — runtime control-file reader                    */
/*                                                                      */
/*  Reads {gb_stats_dir}/control (default /run/greenboost/control)     */
/*  every ~2 s on the already-CAS-won health sub-cadence.  A static    */
/*  mtime gate keeps steady-state cost to one stat(2) per 2 s; parse   */
/*  only runs when the supervisor has written a new version.            */
/*                                                                      */
/*  Ships dark: absent file → stat(2) returns ENOENT → return.  No     */
/*  cost until the supervisor actually starts writing.  The supervisor  */
/*  writes via temp-file + rename() (atomic), pairing with this gate.  */
/*                                                                      */
/*  Key → global mapping (all values clamped before store):            */
/*    workstation_reserve_mb → g_workstation_reserve_bytes  [0,8192]   */
/*    stats_interval_ms      → g_stats_interval_ms          [50,5000]  */
/*    kv_size_threshold_mb   → g_kv_size_threshold_bytes    [1,4096]   */
/*    swa_window_mb          → g_swa_window_bytes            [0,65536]  */
/*    phase_detect           → g_phase_detect                {0,1}     */
/*    kv_reserve_mb          → g_kv_reserve_bytes (_Atomic) [0,65536]  */
/*                             only when g_kv_reserve_from_env == 0    */
/* ------------------------------------------------------------------ */
static void gb_read_control(void)
{
    static time_t s_ctrl_mtime = 0;
    char path[256];
    struct stat st;
    FILE *f;
    char line[256];

    /* gb_stats_dir is always resolved before this is called because
     * gb_write_stats() runs first in the same outer CAS-won block. */
    const char *dir = gb_stats_dir ? gb_stats_dir : GB_STATS_DIR;
    snprintf(path, sizeof(path), "%s/control", dir);

    /* mtime gate: skip parse when file is unchanged */
    if (stat(path, &st) != 0)
        return;
    if (st.st_mtime == s_ctrl_mtime)
        return;
    s_ctrl_mtime = st.st_mtime;

    f = fopen(path, "r");
    if (!f) return;

    while (fgets(line, sizeof(line), f)) {
        char *eq = strchr(line, '=');
        if (!eq) continue;
        *eq = '\0';
        const char *key = line;
        long long val   = gb_atoll(eq + 1);

        if (strcmp(key, "workstation_reserve_mb") == 0) {
            long long c = val < 0 ? 0LL : val > 8192LL ? 8192LL : val;
            g_workstation_reserve_bytes = (size_t)c << 20;
            g_workstation_reserve_from_env = 1;  /* explicit operator override */
        } else if (strcmp(key, "stats_interval_ms") == 0) {
            long long c = val < 50LL ? 50LL : val > 5000LL ? 5000LL : val;
            g_stats_interval_ms = (uint64_t)c;
        } else if (strcmp(key, "kv_size_threshold_mb") == 0) {
            long long c = val < 1LL ? 1LL : val > 4096LL ? 4096LL : val;
            g_kv_size_threshold_bytes = (size_t)c << 20;
        } else if (strcmp(key, "swa_window_mb") == 0) {
            long long c = val < 0LL ? 0LL : val > 65536LL ? 65536LL : val;
            g_swa_window_bytes = (size_t)c << 20;
        } else if (strcmp(key, "phase_detect") == 0) {
            g_phase_detect = (val != 0) ? 1 : 0;
        } else if (strcmp(key, "kv_reserve_mb") == 0 && !g_kv_reserve_from_env) {
            long long c = val < 0LL ? 0LL : val > 65536LL ? 65536LL : val;
            atomic_store_explicit(&g_kv_reserve_bytes,
                                  (size_t)c << 20,
                                  memory_order_relaxed);
        }
    }
    fclose(f);
}

static void gb_maybe_write_stats(void)
{
    /* CRIT-05: Use _Atomic + CAS so exactly one thread wins the write slot
     * per interval.  ENH-05: switched from time(NULL) (1-second resolution)
     * to CLOCK_MONOTONIC so high-throughput sessions can use sub-second
     * intervals (default 250 ms; GREENBOOST_STATS_INTERVAL_MS to override). */
    static _Atomic uint64_t gb_last_stats_ms  = 0;
    static _Atomic uint64_t gb_last_health_ms = 0;
    struct timespec ts;
    uint64_t now_ms, prev_ms, interval;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    now_ms   = (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
    interval = g_stats_interval_ms;
    prev_ms  = atomic_load_explicit(&gb_last_stats_ms, memory_order_relaxed);

    if (now_ms - prev_ms >= interval) {
        if (atomic_compare_exchange_strong_explicit(&gb_last_stats_ms, &prev_ms, now_ms,
                                                    memory_order_relaxed,
                                                    memory_order_relaxed)) {
            gb_write_stats();

            /* D2/D3: Poll feeder health every ~2s (8× stats interval at 250ms default) */
            uint64_t hp = atomic_load_explicit(&gb_last_health_ms, memory_order_relaxed);
            if (now_ms - hp >= 2000ULL) {
                if (atomic_compare_exchange_strong_explicit(&gb_last_health_ms, &hp, now_ms,
                                                            memory_order_relaxed,
                                                            memory_order_relaxed)) {
                    /* poll_health is owned by netc's dedicated heartbeat
                     * thread (gb_netc_hb_thread_main) — calling it from here
                     * too made two actors race on reconnect/socket state. */
                    gb_read_control();   /* apply supervisor hints without restart */
                }
            }

            /* U10: flush pending KV block events to feeders */
            gb_block_evt_flush();

            /* U11: advance cold-epoch counters */
            gb_epoch_tick(now_ms);

            /* MemAvailable pressure check - warn + log NVTX event when
             * system RAM is below GB_MEMAVAIL_WARN_PCT% of pool.
             * Checked every stats interval so it appears in nvtx_events.log. */
            if (gb_t2_pool_bytes > 0) {
                size_t mem_avail = gb_get_mem_available();
                if (mem_avail > 0 && gb_safety_reserve_bytes > 0 &&
                        mem_avail < gb_safety_reserve_bytes) {
                    gb_log("WARNING: MemAvailable %zu MB below safety reserve %zu MB - OOM risk!",
                           mem_avail >> 20, gb_safety_reserve_bytes >> 20);
                    GB_NVTX_EVENT("MEM_PRESSURE_CRITICAL", "SYSTEM", 0,
                        mem_avail >> 20, "memavail_below_safety_reserve");
                } else if (mem_avail > 0 && gb_t2_pool_bytes > 0) {
                    size_t warn_floor = gb_t2_pool_bytes * GB_MEMAVAIL_WARN_PCT / 100;
                    if (mem_avail < warn_floor)
                        GB_NVTX_EVENT("MEM_PRESSURE_WARN", "SYSTEM", 0,
                            mem_avail >> 20, "memavail_below_12pct_pool");
                }
            }
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
    req.flags = alloc_flags;  /* was hardcoded GB_ALLOC_WEIGHTS - now forwarded from caller */

    if (ioctl(dev_fd, GB_IOCTL_ALLOC, &req) < 0) {
        fprintf(stderr, "[GreenBoost] GB_IOCTL_ALLOC failed for %zu MB: %m\n",
                bytesize >> 20);
        /* dev_fd == gb_dev_fd: persistent cached fd - do NOT close here.
         * Closing it would invalidate gb_dev_fd, breaking all subsequent
         * Path A zero-copy allocations (EBADF on next ioctl).  The fd is owned by
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
    /* Retry with cudaExternalMemoryDedicated flag - required on some Ada/Blackwell
     * mobile drivers (e.g. RTX 5070 Laptop) that return error 999 with flags=0. */
    if (cret != CUDA_SUCCESS && hdesc.flags == 0) {
        hdesc.flags = 1; /* cudaExternalMemoryDedicated */
        cret = real_cudaImportExternalMemory(&ext_mem, &hdesc);
        if (cret == CUDA_SUCCESS)
            gb_log("Path A (zero-copy): cudaExternalMemoryDedicated retry succeeded for %zu MB",
                   bytesize >> 20);
        else if (real_cudaGetLastError)
            real_cudaGetLastError(); /* clear sticky error from first attempt */
    }
    if (cret != CUDA_SUCCESS) {
        fprintf(stderr, "[GreenBoost] cudaImportExternalMemory FAILED ret=%d for %zu MB"
                " - disabling Path A zero-copy sub-method permanently to avoid sticky CUDA error\n",
                cret, bytesize >> 20);
        close(req.fd);  /* fd not consumed - close it ourselves */
        gb_a0_disabled = 1;
        /* Clear the sticky CUDA runtime error so Path A pinned sub-method's cuMemHostRegister
         * does not inherit the poisoned context from this failure. */
        if (real_cudaGetLastError)
            real_cudaGetLastError();
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    /* From here CUDA owns req.fd - do not close it */

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
    gb_log("Path A (zero-copy ExternalMem): %zu MB at cuda_ptr=0x%llx ext_mem=%p",
           bytesize >> 20, (unsigned long long)*dptr, (void *)ext_mem);
    return CUDA_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  Constructor - runs before main()                                    */
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


/* VCM-01: Called from CUDA API stubs when gb_init_deferred=1.
 * Completes initialization once libcuda.so is resident - avoids force-loading
 * it in the shim constructor which causes CUDA context conflicts in vLLM workers. */
static void gb_resume_init_locked(void);   /* defined after gb_shim_init below */

static void gb_try_resume_deferred(void)
{
    void *lc;
    if (!gb_init_deferred || initialized) return;
    lc = dlopen("libcuda.so.1", RTLD_NOLOAD | RTLD_NOW | RTLD_GLOBAL);
    if (!lc) return;   /* CUDA still not loaded - will retry on next call */
    dlclose(lc);
    /* Atomic-claim the init slot; first caller triggers pthread_once, rest skip */
    if (__sync_bool_compare_and_swap(&gb_init_deferred, 1, 0))
        pthread_once(&gb_resume_once, gb_resume_init_locked);
}

/* Audit F-L1-03: fork-safety.  CUDA itself is not fork-safe, and the shim
 * holds many pthread primitives initialized at module-load time.  Without an
 * atfork handler the child inherits a possibly-locked mutex and deadlocks on
 * first acquire.  We register handlers that:
 *   prepare(): no-op - we don't lock everything because the shim is too wide
 *              to wrap atomically without risking parent-side deadlock.
 *   parent(): no-op.
 *   child(): scorched-earth re-init of every static mutex/condvar plus a
 *            flag that disables the prefetch thread until the child re-arms
 *            it explicitly.  The child cannot reuse the parent's CUDA context
 *            anyway (CUDA pre-fork rule) so it must re-init from scratch. */
/* Forward decls (these globals are static, defined further down). */
static pthread_mutex_t gb_dev_lock;
static pthread_mutex_t gb_vmm_ht_lock;
static pthread_mutex_t g_block_evt_lock;
static pthread_mutex_t g_gds_lock;
static volatile int gb_in_child_after_fork = 0;
static void gb_atfork_prepare(void) { /* intentionally minimal */ }
static void gb_atfork_parent(void)  { /* intentionally minimal */ }
static void gb_atfork_child(void)
{
    /* Mark globally that we're a post-fork child.  Hot paths that observe
     * this either short-circuit (prefetch worker) or re-initialise on demand. */
    gb_in_child_after_fork = 1;
    /* Reset pthread primitives.  Re-init is the only portable way to escape
     * a locked-by-the-parent state. */
    pthread_mutex_init(&prefetch_mutex, NULL);
    pthread_cond_init(&prefetch_cond, NULL);
    pthread_mutex_init(&gb_dev_lock, NULL);
    pthread_mutex_init(&gb_vmm_ht_lock, NULL);
    pthread_mutex_init(&g_block_evt_lock, NULL);
    pthread_mutex_init(&g_gds_lock, NULL);
    pthread_mutex_init(&gb_t2_reg_pool.lock, NULL);
    pthread_mutex_init(&gb_migrate_lock, NULL);
    for (int i = 0; i < HT_LOCKS; i++) pthread_mutex_init(&ht_locks[i], NULL);
    /* Path C / bvmm_ht state: the lock must be re-initialised, and the entries
     * themselves must be cleared because they reference parent-context CUDA
     * device VAs and host pointers that are not valid in the child.  Freeing
     * those allocations here would call into a CUDA runtime that is in an
     * indeterminate post-fork state, so we simply drop the tracking - the
     * child's first new allocation rebuilds the table from scratch. */
    pthread_mutex_init(&gb_bvmm_ht_lock, NULL);
    memset(gb_bvmm_ht, 0, sizeof(gb_bvmm_ht));
    atomic_store_explicit(&gb_t2_overflow_bytes, 0, memory_order_relaxed);
    atomic_store_explicit(&gb_bvmm_zerocopy_count, 0, memory_order_relaxed);
    atomic_store_explicit(&gb_t2_pending_uvm_bytes, 0, memory_order_relaxed);  /* PR-I/H4 */
    /* Prefetch thread did not survive the fork; mark it gone so gb_shim_fini
     * does not try to join an invalid TID. */
    prefetch_initialized = 0;
    prefetch_stop = 1;
    prefetch_head = prefetch_tail = 0;
}

/* PR-TT: forward decl - definition lives near the kernel signature table. */
static void gb_kernel_sigs_load_file(const char *path);

/* ------------------------------------------------------------------ */
/*  F-ABI1: runtime-API (cuda*) symbol resolution                       */
/*                                                                      */
/*  The cudart "real" pointers MUST come from the same libcudart the    */
/*  application links, not from whichever library a global search finds */
/*  first.  cudaDeviceProp changed layout in CUDA 13.0 (deprecated      */
/*  fields removed → every field after totalGlobalMem shifted); kernel  */
/*  registration handles (__cudaRegisterFunction / cudaLaunchKernel)    */
/*  are private to one libcudart instance.  Forwarding a cu13 app into  */
/*  a cu12 libcudart produced SIGFPE in PyTorch's grid-size division    */
/*  (multiProcessorCount read as 0) and cudaErrorInvalidResourceHandle  */
/*  on launches.                                                        */
/*                                                                      */
/*  At constructor time the app's libcudart is usually NOT loaded yet   */
/*  (python loads torch much later), so init resolves a best-effort     */
/*  fallback and every cudart hook calls GB_CUDART_ENSURE() — a         */
/*  one-shot lazy rebind that scans /proc/self/maps for the libcudart   */
/*  the app actually mapped and re-resolves all pointers from it.       */
/*  GREENBOOST_CUDART_PATH (explicit override) disables the rebind.     */
/* ------------------------------------------------------------------ */
static char g_cudart_init_path[512];        /* resolved path init loaded   */
static int  g_cudart_override;              /* GREENBOOST_CUDART_PATH used */
static _Atomic int g_cudart_rebound;        /* one-shot latch              */

/* F-ABI1: resolve a symbol from ONE specific loaded library by walking its
 * ELF dynamic symbol table directly (DT_GNU_HASH / DT_HASH).  dlsym(handle)
 * is NOT equivalent: for a library that is already resident as a dependency
 * of the application (torch's bundled libcudart), glibc searches the map's
 * scope - which begins with the global scope, where THIS shim's interposing
 * exports win.  That made real___cudaRegisterFunction point back at our own
 * hook: the forward became a tail-call self-loop and `import torch` spun
 * forever re-registering the first kernel. */
static void *gb_lm_sym(struct link_map *lm, const char *name)
{
    const ElfW(Sym) *symtab = NULL;
    const char *strtab = NULL;
    const uint32_t *gnu_hash = NULL;
    const ElfW(Word) *elf_hash = NULL;

    for (const ElfW(Dyn) *d = lm->l_ld; d->d_tag != DT_NULL; d++) {
        switch (d->d_tag) {
        case DT_SYMTAB:   symtab   = (const ElfW(Sym) *)d->d_un.d_ptr;  break;
        case DT_STRTAB:   strtab   = (const char *)d->d_un.d_ptr;       break;
        case DT_GNU_HASH: gnu_hash = (const uint32_t *)d->d_un.d_ptr;   break;
        case DT_HASH:     elf_hash = (const ElfW(Word) *)d->d_un.d_ptr; break;
        }
    }
    if (!symtab || !strtab)
        return NULL;

    if (gnu_hash) {
        uint32_t nbuckets    = gnu_hash[0];
        uint32_t symoffset   = gnu_hash[1];
        uint32_t bloom_size  = gnu_hash[2];
        const ElfW(Addr) *bloom   = (const ElfW(Addr) *)&gnu_hash[4];
        const uint32_t   *buckets = (const uint32_t *)&bloom[bloom_size];
        const uint32_t   *chain   = &buckets[nbuckets];

        uint32_t h = 5381;
        for (const unsigned char *c = (const unsigned char *)name; *c; c++)
            h = h * 33 + *c;

        uint32_t sym = buckets[h % nbuckets];
        if (sym < symoffset)
            return NULL;
        for (;; sym++) {
            uint32_t ch = chain[sym - symoffset];
            if ((ch | 1u) == (h | 1u)) {
                const ElfW(Sym) *s = &symtab[sym];
                if (s->st_shndx != SHN_UNDEF &&
                    strcmp(strtab + s->st_name, name) == 0)
                    return (void *)(lm->l_addr + s->st_value);
            }
            if (ch & 1u)
                break;
        }
        return NULL;
    }

    if (elf_hash) {
        ElfW(Word) nchain = elf_hash[1];
        for (ElfW(Word) i = 0; i < nchain; i++) {
            const ElfW(Sym) *s = &symtab[i];
            if (s->st_shndx != SHN_UNDEF && s->st_name &&
                strcmp(strtab + s->st_name, name) == 0)
                return (void *)(lm->l_addr + s->st_value);
        }
    }
    return NULL;
}

/* Resolve `name` from the library behind `h`, never from this shim. */
static void *gb_cudart_sym(void *h, const char *name)
{
    struct link_map *lm = NULL;
    void *p = NULL;

    if (dlinfo(h, RTLD_DI_LINKMAP, &lm) == 0 && lm)
        p = gb_lm_sym(lm, name);
    if (!p) {
        /* Fallback: scope-based lookup, but REJECT our own interposing
         * exports (see gb_lm_sym comment). */
        p = dlsym(h, name);
        if (p) {
            static void *shim_base;
            Dl_info di;
            if (!shim_base && dladdr((void *)&gb_lm_sym, &di))
                shim_base = di.dli_fbase;
            if (dladdr(p, &di) && di.dli_fbase == shim_base)
                p = NULL;
        }
    }
    return p;
}

static void gb_cudart_resolve_syms(void *libcudart)
{
    real_cudaMalloc           = (pfn_cudaMalloc)           gb_cudart_sym(libcudart, "cudaMalloc");
    real_cudaFree             = (pfn_cudaFree)             gb_cudart_sym(libcudart, "cudaFree");
    real_cudaMallocManaged    = (pfn_cudaMallocManaged)    gb_cudart_sym(libcudart, "cudaMallocManaged");
    real_cudaMallocAsync      = (pfn_cudaMallocAsync)      gb_cudart_sym(libcudart, "cudaMallocAsync");
    real_cudaGetDeviceCount   = (pfn_cudaGetDeviceCount)   gb_cudart_sym(libcudart, "cudaGetDeviceCount");
    real_cudaSetDevice        = (pfn_cudaSetDevice)        gb_cudart_sym(libcudart, "cudaSetDevice");
    real_cudaDeviceCanAccessPeer    = (pfn_cudaDeviceCanAccessPeer)
        gb_cudart_sym(libcudart, "cudaDeviceCanAccessPeer");
    real_cudaDeviceEnablePeerAccess = (pfn_cudaDeviceEnablePeerAccess)
        gb_cudart_sym(libcudart, "cudaDeviceEnablePeerAccess");
    real_cudaPointerGetAttributes   = (pfn_cudaPointerGetAttributes)
        gb_cudart_sym(libcudart, "cudaPointerGetAttributes");
    real_cudaMemcpy           = (cudaError_t (*)(void *, const void *, size_t, int))
        gb_cudart_sym(libcudart, "cudaMemcpy");
    real_cudaMemset           = (cudaError_t (*)(void *, int, size_t))
        gb_cudart_sym(libcudart, "cudaMemset");
    real_cudaMemsetAsync      = (cudaError_t (*)(void *, int, size_t, cudaStream_t))
        gb_cudart_sym(libcudart, "cudaMemsetAsync");
    real_cudaMemcpy2DAsync    = (cudaError_t (*)(void *, size_t, const void *, size_t,
                                                 size_t, size_t, int, cudaStream_t))
        gb_cudart_sym(libcudart, "cudaMemcpy2DAsync");
    real_cudaLaunchKernelExC  = (pfn_cudaLaunchKernelExC)
        gb_cudart_sym(libcudart, "cudaLaunchKernelExC");
    real_cudaGetKernel        = (cudaError_t (*)(void **, const void *))
        gb_cudart_sym(libcudart, "cudaGetKernel");
    real_cudaMemcpyAsync      = (cudaError_t (*)(void *, const void *, size_t, int, void *))
        gb_cudart_sym(libcudart, "cudaMemcpyAsync");
    real_cudaMemcpyPeer       = (cudaError_t (*)(void *, int, const void *, int, size_t))
        gb_cudart_sym(libcudart, "cudaMemcpyPeer");
    real_cudaMemcpyPeerAsync  = (cudaError_t (*)(void *, int, const void *, int, size_t, void *))
        gb_cudart_sym(libcudart, "cudaMemcpyPeerAsync");
    real_cudaImportExternalMemory        = (pfn_cudaImportExternalMemory)
        gb_cudart_sym(libcudart, "cudaImportExternalMemory");
    real_cudaExternalMemoryGetMappedBuffer = (pfn_cudaExternalMemoryGetMappedBuffer)
        gb_cudart_sym(libcudart, "cudaExternalMemoryGetMappedBuffer");
    real_cudaDestroyExternalMemory       = (pfn_cudaDestroyExternalMemory)
        gb_cudart_sym(libcudart, "cudaDestroyExternalMemory");
    real_cudaGetLastError                = (pfn_cudaGetLastError)
        gb_cudart_sym(libcudart, "cudaGetLastError");
    real_cudaDeviceSynchronize           = (pfn_cudaDeviceSynchronize)
        gb_cudart_sym(libcudart, "cudaDeviceSynchronize");
    real_cudaMemGetInfo                  = (pfn_cudaMemGetInfo)
        gb_cudart_sym(libcudart, "cudaMemGetInfo");
    real_cudaMemPrefetchAsync            = (pfn_cudaMemPrefetchAsync)
        gb_cudart_sym(libcudart, "cudaMemPrefetchAsync");
    real_cudaGetDeviceProperties         = (pfn_cudaGetDeviceProperties)
        gb_cudart_sym(libcudart, "cudaGetDeviceProperties_v2");
    if (!real_cudaGetDeviceProperties)
        real_cudaGetDeviceProperties     = (pfn_cudaGetDeviceProperties)
            gb_cudart_sym(libcudart, "cudaGetDeviceProperties");
    real_cudaGetDriverEntryPointByVersion = (pfn_cudaGetDriverEntryPointByVersion)
        gb_cudart_sym(libcudart, "cudaGetDriverEntryPointByVersion");
    real_cudaLaunchKernel     = (pfn_cudaLaunchKernel)     gb_cudart_sym(libcudart, "cudaLaunchKernel");
    real_cudaStreamSynchronize = (pfn_cudaStreamSynchronize) gb_cudart_sym(libcudart, "cudaStreamSynchronize");
    real_cudaStreamIsCapturing  = (pfn_cudaStreamIsCapturing)  gb_cudart_sym(libcudart, "cudaStreamIsCapturing");
    real_cudaStreamBeginCapture = (pfn_cudaStreamBeginCapture) gb_cudart_sym(libcudart, "cudaStreamBeginCapture");
    /* CUDA doc: cudaStreamCreateWithPriority - resolve for stream priority
     * elevation hook; optional (silently skipped if unavailable). */
    real_cudaStreamCreateWithPriority = (pfn_cudaStreamCreateWithPriority)
        gb_cudart_sym(libcudart, "cudaStreamCreateWithPriority");
    real_cudaDeviceGetAttribute = (pfn_cudaDeviceGetAttribute)
        gb_cudart_sym(libcudart, "cudaDeviceGetAttribute");
    real_cudaDeviceGetStreamPriorityRange = (pfn_cudaDeviceGetStreamPriorityRange)
        gb_cudart_sym(libcudart, "cudaDeviceGetStreamPriorityRange");
    real___cudaRegisterFunction = (pfn___cudaRegisterFunction)
        gb_cudart_sym(libcudart, "__cudaRegisterFunction");
}

static void gb_cudart_rebind(void)
{
    static pthread_mutex_t rebind_mu = PTHREAD_MUTEX_INITIALIZER;
    char found[512] = "";
    FILE *m;

    pthread_mutex_lock(&rebind_mu);
    if (atomic_load_explicit(&g_cudart_rebound, memory_order_acquire)) {
        pthread_mutex_unlock(&rebind_mu);
        return;
    }
    if (g_cudart_override) {
        /* Explicit GREENBOOST_CUDART_PATH always wins - no auto-rebind. */
        atomic_store_explicit(&g_cudart_rebound, 1, memory_order_release);
        pthread_mutex_unlock(&rebind_mu);
        return;
    }

    /* A cudart hook is executing, so the caller's libcudart is mapped NOW.
     * Pick the first libcudart in maps that is not the one init loaded. */
    m = fopen("/proc/self/maps", "r");
    if (m) {
        char line[1024];
        while (fgets(line, sizeof(line), m)) {
            char *p = strchr(line, '/');
            if (!p || !strstr(p, "libcudart.so"))
                continue;
            p[strcspn(p, "\n")] = '\0';
            if (g_cudart_init_path[0] && strcmp(p, g_cudart_init_path) == 0)
                continue;               /* the fallback WE dlopened - skip */
            snprintf(found, sizeof(found), "%s", p);
            break;
        }
        fclose(m);
    }

    if (found[0]) {
        /* Already mapped: dlopen is a refcount bump, no new load.  LOCAL -
         * the app resolves its own symbols; we only need function pointers. */
        void *h = dlopen(found, RTLD_NOW | RTLD_LOCAL);
        if (h) {
            gb_cudart_resolve_syms(h);
            gb_log("cudart rebind -> %s (init fallback was %s)",
                   found, g_cudart_init_path[0] ? g_cudart_init_path : "none");
        }
    }
    /* Latch even when nothing was found: a hook reached via dynamic
     * interposition implies the caller's libcudart is the one in maps -
     * if only init's path is there, init already picked the right one. */
    atomic_store_explicit(&g_cudart_rebound, 1, memory_order_release);
    pthread_mutex_unlock(&rebind_mu);
}

/* Fast-path gate for every exported cudart hook. */
#define GB_CUDART_ENSURE() \
    do { \
        if (!atomic_load_explicit(&g_cudart_rebound, memory_order_acquire)) \
            gb_cudart_rebind(); \
    } while (0)

__attribute__((constructor))
static void gb_shim_init(void)
{
    /* PID-1 guard: never activate the CUDA interposer inside systemd or any
     * other init process.  If this library is ever loaded globally via
     * /etc/ld.so.preload the loader calls this constructor for every process,
     * including PID 1 — the CUDA hooks must not run there or they freeze boot.
     * Per-process injection (systemd drop-ins, greenboost-run* wrappers) is the
     * correct and only supported injection path. */
    if (getpid() == 1) return;

    void *libcuda, *libcudart = NULL;
    const char *env;
    uint32_t i;
    int forced;
    /* Audit F-L1-03: register atfork handlers as the first thing the
     * constructor does, so any subsequent fork sees a consistent setup. */
    static int atfork_registered = 0;
    if (!atfork_registered) {
        if (pthread_atfork(gb_atfork_prepare,
                           gb_atfork_parent,
                           gb_atfork_child) == 0) {
            atfork_registered = 1;
        }
    }

    /* Hard opt-out: set GREENBOOST_DISABLE=1 to keep the shim completely inert
     * for this process.  Useful as a Steam launch option for games that have
     * conflicts: GREENBOOST_DISABLE=1 %command% */
    if (getenv("GREENBOOST_DISABLE"))
        return;

    /* PR-Q/F-S12: open the NVTX log file proactively at constructor time so
     * the GB_NVTX_EVENT macro can early-exit on the hot path
     * (cuLaunchKernel, cuLaunchKernelEx, cudaMemcpy*Async) via a single
     * branch on g_nvtx_log_fd >= 0 instead of paying the function-call +
     * pthread_once cost per event.  On a typical decode workload this
     * saves ~10 ms per token at 100+ launches per token; on prefill the
     * win is larger (more launches). */
    gb_nvtx_log_open();

    /* PR-TT: load kernel-signature overrides from config file if set.
     * Each line: `kernel_name n_args` (# comments OK).  Entries are
     * applied in __cudaRegisterFunction below - when a CUDA kernel
     * registration with a matching name comes through, its host_fn
     * gets a registered arg-count in gb_kernel_sigs, bounding the
     * subsequent kernelParams[] scans on launch. */
    {
        const char *sigs_path = getenv("GREENBOOST_KERNEL_SIGS");
        if (sigs_path && *sigs_path)
            gb_kernel_sigs_load_file(sigs_path);
    }

    /* Parse ALL env vars before any early return so GREENBOOST_DEBUG is
     * available regardless of which activation path is taken. */
    env = getenv("GREENBOOST_USE_DMA_BUF");
    if (env) gb_use_dmabuf = (env[0] != '0');

    /* AUD-08: Container detection - cache result at init to avoid an open()
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
                fprintf(stderr, "[GreenBoost] Container detected - Path A disabled, using Path B/C\n");
        }
    }

    env = getenv("GREENBOOST_VRAM_HEADROOM_MB");
    int headroom_from_env = 0;
    if (env) { vram_headroom_bytes = (size_t)gb_atoll(env) * 1024ULL * 1024ULL; headroom_from_env = 1; }

    /* Workstation GPU headroom: how much physical VRAM to keep free for the
     * desktop, compositor, and other GPU processes.  Subtracts from the free
     * VRAM reported by cuMemGetInfo_v2 / cudaMemGetInfo so llama-server's
     * --fit algorithm does not fill T1 to 100%, keeping the workstation
     * responsive during inference.  By default this is now DYNAMIC (see
     * gb_effective_workstation_reserve_bytes): sized to whatever the desktop
     * is actually using right now, not a flat tax.  Set
     * GREENBOOST_WORKSTATION_RESERVE_MB to force a fixed value instead
     * (e.g. =0 on a dedicated headless inference node). */
    env = getenv("GREENBOOST_WORKSTATION_RESERVE_MB");
    if (env) {
        g_workstation_reserve_bytes = (size_t)gb_atoll(env) * 1024ULL * 1024ULL;
        g_workstation_reserve_from_env = 1;
    }

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

    env = getenv("GREENBOOST_T1_WORKSPACE_MB");
    if (env) {
        long long ws = gb_atoll(env);
        if (ws > 0) {
            atomic_store_explicit(&g_workspace_reserve_bytes,
                                  (size_t)ws * 1024ULL * 1024ULL,
                                  memory_order_relaxed);
            GB_NVTX_EVENT("T1_WORKSPACE_RESERVE", "PHASE", (size_t)ws, 0,
                          "reserve_armed_for_model_load");
        }
    }

    env = getenv("GREENBOOST_VIRTUAL_VRAM_MB");
    if (env) gb_virtual_vram_bytes = (size_t)gb_atoll(env) * 1024ULL * 1024ULL;

    env = getenv("GREENBOOST_DEBUG");
    if (env && env[0] == '1') gb_debug = 1;

    env = getenv("GREENBOOST_NO_HOSTREG");
    if (env && env[0] == '1') gb_no_hostreg = 1;

    env = getenv("GREENBOOST_T2_POOL_MB");
    if (env && env[0] == '0')
        gb_pool_configured_bytes = 0;       /* explicit disable */
    else if (env)
        gb_pool_configured_bytes = (size_t)gb_atoll(env) * 1024ULL * 1024ULL;
    else
        gb_pool_configured_bytes = SIZE_MAX; /* sentinel: resolve from gb_t2_pool_bytes later */

    env = getenv("GREENBOOST_A0_DISABLE");
    if (env && env[0] != '0')
        gb_a0_disabled = 1;

    /* Explicit per-process override; auto-engage handled below after cc probe. */
    env = getenv("GREENBOOST_DISABLE_T2_ON_BLACKWELL");
    if (env) gb_disable_t2_on_blackwell = (env[0] != '0');

    env = getenv("GREENBOOST_KV_OVERFLOW");
    if (env && env[0] == '1') g_kv_overflow_mode = 1;

    env = getenv("GREENBOOST_REPORT_PHYSICAL_VRAM");
    if (env && env[0] == '1') g_report_physical_vram = 1;

    env = getenv("GREENBOOST_DEBUG_ATTR");
    if (env && env[0] == '1') g_debug_attr = 1;

    /* N11: SWA sliding-window per-request KV eviction.
     * GREENBOOST_SWA_WINDOW=<MB>: when live T2 KV bytes exceed this, evict
     * oldest KV blocks before each new KV alloc. 0 = disabled (default). */
    env = getenv("GREENBOOST_SWA_WINDOW");
    if (env) {
        long swa_mb = gb_atoll(env);
        if (swa_mb > 0) {
            g_swa_window_bytes = (size_t)swa_mb * 1024ULL * 1024ULL;
            GB_INIT_LOG_ONCE("[GreenBoost] N11: SWA window = %ld MB\n", swa_mb);
        }
    }

    env = getenv("GREENBOOST_PHASE_DETECT");
    if (env && env[0] == '0') g_phase_detect = 0;

    env = getenv("GREENBOOST_KV_SIZE_THRESHOLD_MB");
    if (env) g_kv_size_threshold_bytes = (size_t)gb_atoll(env) * 1024ULL * 1024ULL;

    /* Phase 2: K/V int8 compression before T1→T2 eviction (halves DMA bandwidth) */
    env = getenv("GREENBOOST_KV_COMPRESS");
    if (env && env[0] == '1') {
        g_kv_compress_enabled = 1;
        GB_INIT_LOG_ONCE("[GreenBoost] GREENBOOST_KV_COMPRESS=1: K/V int8 compression enabled\n");
    }

    /* Phase-aware KV prefetch (Stage 3). Default off. "stats" measures
     * opportunity only; "1"/"active" is reserved for the live-promotion path
     * (currently behaves as stats until a long-context hardware bench). */
    env = getenv("GREENBOOST_KV_PREFETCH");
    if (env && env[0]) {
        if (env[0] == '0') {
            g_kv_prefetch_mode = GB_KV_PREFETCH_OFF;
        } else if (!strcmp(env, "stats")) {
            g_kv_prefetch_mode = GB_KV_PREFETCH_STATS;
            GB_INIT_LOG_ONCE("[GreenBoost] GREENBOOST_KV_PREFETCH=stats: KV-prefetch opportunity metering on\n");
        } else {
            g_kv_prefetch_mode = GB_KV_PREFETCH_ACTIVE;
            GB_INIT_LOG_ONCE("[GreenBoost] GREENBOOST_KV_PREFETCH active: live promotion pending hardware bench, metering as stats\n");
        }
    }

    /* U18/A4: Double-buffer T3→T2 prefetch staging pipeline */
    env = getenv("GREENBOOST_DOUBLE_BUFFER");
    if (env && env[0] == '1') {
        g_double_buffer_enabled = 1;
        GB_INIT_LOG_ONCE("[GreenBoost] GREENBOOST_DOUBLE_BUFFER=1: "
                "double-buffer prefetch lookahead enabled (A4 BW-aware tile sizing)\n");
    }

    /* Phase 2b: async feeder kernel dispatch */
    env = getenv("GREENBOOST_ASYNC_DISPATCH");
    if (env && env[0] == '1') {
        g_async_dispatch = 1;
        GB_INIT_LOG_ONCE("[GreenBoost] GREENBOOST_ASYNC_DISPATCH=1: "
                "feeder kernels dispatched fire-and-forget (sync at stream boundaries)\n");
    }

    /* Phase 3: GPUDirect Storage - direct GPU↔NVMe without CPU bounce */
    {
        int gds_requested = 0;
        env = getenv("GREENBOOST_GDS");
        if (env && env[0] == '1') gds_requested = 1;

        g_libcufile = dlopen("libcufile.so.0", RTLD_LAZY | RTLD_LOCAL);
        if (g_libcufile) {
            f_cuFileDriverOpen    = (pfn_cuFileDriverOpen_t)
                                     dlsym(g_libcufile, "cuFileDriverOpen");
            f_cuFileHandleRegister = (pfn_cuFileHandleRegister_t)
                                     dlsym(g_libcufile, "cuFileHandleRegister");
            f_cuFileWrite         = (pfn_cuFileWrite_t)
                                     dlsym(g_libcufile, "cuFileWrite");
            f_cuFileRead          = (pfn_cuFileRead_t)
                                     dlsym(g_libcufile, "cuFileRead");
            if (f_cuFileDriverOpen && f_cuFileWrite && f_cuFileRead && gds_requested) {
                if (f_cuFileDriverOpen() == 0 /* CU_FILE_SUCCESS */) {
                    g_gds_ok = 1;
                    fprintf(stderr, "[GreenBoost] GDS: cuFile available - "
                            "GPUDirect Storage enabled for T3 path\n");
                }
            }
        }
        if (gds_requested && !g_gds_ok)
            fprintf(stderr, "[GreenBoost] GDS: libcufile.so.0 not found or init failed - "
                    "T3 will use CPU-bounce path\n");
    }

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
    /* Feature manifest for consumers (gb_monitor.capabilities). Runtime feature
     * globals (gds/kv_compress/report_physical_vram) are resolved above. */
    gb_write_capabilities_file();


    forced = (getenv("GREENBOOST_ACTIVE") != NULL);

/* Stage 1: RTLD_NOLOAD - check whether libcuda.so.1 is already resident.
     * This never triggers CUDA driver initialisation (no dlopen side-effects),
     * so it is safe in any process: GDM, shells, systemd helpers, etc.
     * Succeeds for apps that link libcuda statically (llama.cpp) - they get
     * automatic transparent injection with no wrapper needed. */
    libcuda = dlopen("libcuda.so.1", RTLD_NOLOAD | RTLD_NOW | RTLD_GLOBAL);

    /* Stage 2: explicit opt-in for apps that load CUDA lazily via dlopen at
     * runtime (Ollama, vLLM, PyTorch).  The Ollama/llama-server systemd service
     * units set GREENBOOST_ACTIVE=1; the greenboost-run wrapper does too for
     * CLI use.  GDM, shells, and helpers never reach this branch. */
    if (!libcuda) {
        /* VCM-01: vLLM compatibility mode - LD_PRELOAD injects the shim but
         * force-loading libcuda.so here causes CUDA context conflicts in
         * vLLM's EngineCore subprocess.  Defer until the first CUDA API call,
         * by which time vLLM has already initialized its own CUDA context. */
        if (getenv("GREENBOOST_VLLM_COMPAT")) {
            gb_init_deferred = 1;
            if (gb_debug)
                fprintf(stderr, "[GreenBoost] vLLM compat: deferred init pending first CUDA call\n");
            return;
        }
        if (!forced) {
            /* Not a CUDA process and not opted in - shim stays inert. */
            return;
        }
        libcuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!libcuda) {
            if (gb_debug)
                fprintf(stderr, "[GreenBoost] libcuda.so.1 not found - shim inactive\n");
            return;
        }
    }

    /* Initialize lock arrays */
    for (i = 0; i < HT_LOCKS; i++)
        pthread_mutex_init(&ht_locks[i], NULL);

    /* libcudart search order:
     *   1. GREENBOOST_CUDART_PATH env var - explicit override (escape hatch for non-standard installs)
     *   2. Unversioned / versioned names - found via LD_LIBRARY_PATH or ldconfig
     *   3. System CUDA toolkit paths - llama.cpp uses system CUDA, not Ollama-bundled
     *   4. Ollama-bundled paths - Ollama ships its own libcudart under /usr/local/lib/ollama/
     */
    {
        /* F-ABI1: RTLD_LOCAL, NOT GLOBAL.  A GLOBAL fallback cudart hijacks
         * every cudart symbol of later-loaded libraries: torch's own
         * __cudaRegisterFatBinary would resolve to OUR fallback (possibly a
         * different CUDA major) instead of the libcudart it bundles, while
         * the hooked calls forward to the rebound one - split-brain fatbin
         * handles hang the import.  LOCAL keeps the fallback private to the
         * shim's dlsym lookups; apps resolve cudart inside their own group. */
        const char *cudart_override = getenv("GREENBOOST_CUDART_PATH");
        if (cudart_override) {
            libcudart = dlopen(cudart_override, RTLD_NOW | RTLD_LOCAL);
            if (!libcudart)
                fprintf(stderr, "[GreenBoost] WARNING: GREENBOOST_CUDART_PATH=%s failed: %s\n",
                        cudart_override, dlerror());
            else
                g_cudart_override = 1;  /* F-ABI1: explicit path - no auto-rebind */
        }

        if (!libcudart) {
            static const char *cudart_paths[] = {
                /* unversioned / versioned - resolved via LD_LIBRARY_PATH or ldconfig */
                "libcudart.so",
                "libcudart.so.13",
                "libcudart.so.12",
                /* system CUDA toolkit - standard install locations for llama.cpp */
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
                libcudart = dlopen(*p, RTLD_NOW | RTLD_LOCAL);  /* F-ABI1 */
        }

        if (libcudart) {
            /* F-ABI1: remember the RESOLVED path of the fallback we loaded so
             * the lazy rebind can tell it apart from the app's own libcudart. */
            struct link_map *lm = NULL;
            if (dlinfo(libcudart, RTLD_DI_LINKMAP, &lm) == 0 && lm && lm->l_name[0])
                snprintf(g_cudart_init_path, sizeof(g_cudart_init_path), "%s", lm->l_name);
            if (gb_debug)
                fprintf(stderr, "[GreenBoost] libcudart loaded (%s)\n",
                        g_cudart_init_path[0] ? g_cudart_init_path : "?");
        } else {
            if (gb_debug)
                fprintf(stderr, "[GreenBoost] WARNING: libcudart not found - runtime API resolved lazily\n");
        }
    }

    /* Driver API (cu*) - always from libcuda.so.1 */
    real_cuMemAlloc_v2     = (pfn_cuMemAlloc_v2)     dlsym(libcuda, "cuMemAlloc_v2");
    real_cuMemFree_v2      = (pfn_cuMemFree_v2)      dlsym(libcuda, "cuMemFree_v2");
    real_cuMemAllocManaged = (pfn_cuMemAllocManaged)  dlsym(libcuda, "cuMemAllocManaged");
    real_cuMemAllocAsync   = (pfn_cuMemAllocAsync)    dlsym(libcuda, "cuMemAllocAsync");
    /* PR-N/F-S7: pool-allocator path used by PyTorch 2.4+ when
     * expandable_segments:True.  Without this hook, allocations from
     * application-created memory pools bypass GreenBoost overflow
     * entirely - PyTorch's caching allocator silently exhausts physical
     * VRAM and OOMs as if GreenBoost weren't loaded. */
    real_cuMemAllocFromPoolAsync = (pfn_cuMemAllocFromPoolAsync)
        dlsym(libcuda, "cuMemAllocFromPoolAsync");
    real_cuMemGetInfo      = (pfn_cuMemGetInfo)       dlsym(libcuda, "cuMemGetInfo_v2");
    if (!real_cuMemGetInfo)
        real_cuMemGetInfo  = (pfn_cuMemGetInfo)       dlsym(libcuda, "cuMemGetInfo");
    real_cuDeviceTotalMem_v2 = (pfn_cuDeviceTotalMem_v2) dlsym(libcuda, "cuDeviceTotalMem_v2");
    if (!real_cuDeviceTotalMem_v2)
        real_cuDeviceTotalMem_v2 = (pfn_cuDeviceTotalMem_v2) dlsym(libcuda, "cuDeviceTotalMem");
    real_cuDeviceGetCount = (pfn_cuDeviceGetCount) dlsym(libcuda, "cuDeviceGetCount");
    real_cuDeviceGet      = (pfn_cuDeviceGet)      dlsym(libcuda, "cuDeviceGet");
    real_cuGetProcAddress = (pfn_cuGetProcAddress) dlsym(libcuda, "cuGetProcAddress");
    real_cuLaunchKernel            = (pfn_cuLaunchKernel)            dlsym(libcuda, "cuLaunchKernel");
    real_cuLaunchCooperativeKernel = (pfn_cuLaunchCooperativeKernel) dlsym(libcuda, "cuLaunchCooperativeKernel");
    /* PR-NN: cuFuncGetParamInfo is CUDA 12.3+; NULL on older drivers. */
    real_cuFuncGetParamInfo        = (pfn_cuFuncGetParamInfo)        dlsym(libcuda, "cuFuncGetParamInfo");
    /* PR-O/F-S8: CUDA 12+ extended launch API used by PyTorch 2.4+ graph
     * mode and JIT-compiled kernels with launch attributes (priority,
     * cooperative, cluster geometry).  Without this hook, those launches
     * skip the data-driven cluster-feeder dispatch entirely - feeder GPU
     * compute is silently disabled for graph-mode workloads. */
    real_cuLaunchKernelEx          = (pfn_cuLaunchKernelEx)          dlsym(libcuda, "cuLaunchKernelEx");
    real_cuMemHostRegister = (pfn_cuMemHostRegister) dlsym(libcuda, "cuMemHostRegister");
    real_cuMemHostUnregister = (pfn_cuMemHostUnregister) dlsym(libcuda, "cuMemHostUnregister");
    real_cuMemHostGetDevicePointer = (pfn_cuMemHostGetDevicePointer) dlsym(libcuda, "cuMemHostGetDevicePointer_v2");
    real_cuMemPrefetchAsync = (pfn_cuMemPrefetchAsync) dlsym(libcuda, "cuMemPrefetchAsync");
    real_cuMemAdvise = (pfn_cuMemAdvise) dlsym(libcuda, "cuMemAdvise");
    real_cuMemFreeAsync = (pfn_cuMemFreeAsync) dlsym(libcuda, "cuMemFreeAsync");
    real_cuDeviceGetAttribute = (pfn_cuDeviceGetAttribute) dlsym(libcuda, "cuDeviceGetAttribute");
    real_cuMemCreate    = (pfn_cuMemCreate)    dlsym(libcuda, "cuMemCreate");
    real_cuMemRelease   = (pfn_cuMemRelease)   dlsym(libcuda, "cuMemRelease");
    real_cuMemMap       = (pfn_cuMemMap)       dlsym(libcuda, "cuMemMap");
    real_cuMemUnmap     = (pfn_cuMemUnmap)     dlsym(libcuda, "cuMemUnmap");
    real_cuMemSetAccess = (pfn_cuMemSetAccess) dlsym(libcuda, "cuMemSetAccess");
    real_cuMemAddressReserve = (pfn_cuMemAddressReserve) dlsym(libcuda, "cuMemAddressReserve");
    real_cuMemAddressFree    = (pfn_cuMemAddressFree)    dlsym(libcuda, "cuMemAddressFree");
    real_cuMemGetAllocationGranularity = (pfn_cuMemGetAllocationGranularity) dlsym(libcuda, "cuMemGetAllocationGranularity");

    /* Probe compute capability - required to gate cuMemAllocAsync (cc >= 8.0).
     * CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75 (stable since CUDA 3). */
    if (real_cuDeviceGetAttribute) {
        int cc_major = 0;
        if (real_cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0) == CUDA_SUCCESS)
            gb_cc_major = cc_major;
    }

    /* Path A zero-copy sub-method uses cudaImportExternalMemory(OpaqueFd) to import
     * DMA-BUF fds exported by greenboost.ko.  OpaqueFd is an NVIDIA-internal fd type;
     * the CUDA driver does not accept standard Linux DMA-BUF fds through this API.
     * On Blackwell (Compute 12.x) with driver 555+ this returns CUDA_ERROR_UNKNOWN
     * (999) immediately.  Pre-disable it here so the pinned sub-method is used
     * directly without an unnecessary probe failure on every startup. */
    if (!gb_a0_disabled && gb_cc_major >= 12) {
        gb_a0_disabled = 1;
        fprintf(stderr, "[GreenBoost] Path A zero-copy sub-method disabled for Compute %d.x "
                "(OpaqueFd not supported; using pinned sub-method)\n",
                gb_cc_major);
    }

    /* Blackwell PCIe T2 is now enabled via managed UVM (cuMemAllocManaged +
     * PREFERRED_LOCATION=CPU + ACCESSED_BY=GPU) - see gb_vmm_t2_alloc_blackwell_managed().
     * The old auto-disable for zerocopy/hostnuma paths is removed.  The explicit
     * escape hatch GREENBOOST_DISABLE_T2_ON_BLACKWELL=1 still hard-refuses. */

    /* NVML - loaded separately; Ollama uses this for GPU memory discovery.
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
            real_nvmlDeviceGetCount = (pfn_nvmlDeviceGetCount)
                dlsym(libnvml, "nvmlDeviceGetCount");
            real_nvmlDeviceGetHandleByIndex = (pfn_nvmlDeviceGetHandleByIndex)
                dlsym(libnvml, "nvmlDeviceGetHandleByIndex_v2");
            if (!real_nvmlDeviceGetHandleByIndex)
                real_nvmlDeviceGetHandleByIndex = (pfn_nvmlDeviceGetHandleByIndex)
                    dlsym(libnvml, "nvmlDeviceGetHandleByIndex");
            real_nvmlDeviceGetName = (pfn_nvmlDeviceGetName)
                dlsym(libnvml, "nvmlDeviceGetName");

            /* Probe physical VRAM - call real functions directly, before our hooks
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

    /* Read virtual (T2 DDR) pool size from kernel module - same source as
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
     * no GREENBOOST_VIRTUAL_VRAM_MB env var - gb_t2_pool_bytes stays 0, which makes
     * every cap guard in gb_overflow_alloc() a no-op.  Compute a safe default from
     * MemTotal so Path A/B guards are always enforced. */
    if (gb_t2_pool_bytes == 0) {
        FILE *mf = fopen("/proc/meminfo", "r");
        if (mf) {
            char mline[128];
            while (fgets(mline, sizeof(mline), mf)) {
                unsigned long long kb = 0;
                if (sscanf(mline, "MemTotal: %llu kB", &kb) == 1 && kb > 0) {
                    gb_t2_pool_bytes      = (size_t)(kb * 1024ULL * 70ULL / 100ULL);
                    gb_virtual_vram_bytes = gb_t2_pool_bytes;
                    gb_log("T2 pool fallback from MemTotal: %zu MB (70%% of %llu MB)",
                           gb_t2_pool_bytes >> 20, kb / 1024ULL);
                    break;
                }
            }
            fclose(mf);
        }
    }

    /* Resolve T2 pool size sentinel: SIZE_MAX means "default to 85% of T2".
     * Must run after gb_t2_pool_bytes is fully populated above. */
    /* cuMemHostRegister blocks cudaMalloc synchronously during gb_pool_init().
     * Pinning 30+ GB takes 3-10+ minutes on a cold DDR5 system - effectively
     * hanging Ollama.  Cap the default to 8 GB so the worst-case init delay is
     * ~10-30 s.  Allocs larger than the pool use per-alloc Path A/B;
     * if T2 is exhausted, CUDA_ERROR_OUT_OF_MEMORY is returned.
     * Users who want a larger pool can set GREENBOOST_T2_POOL_MB explicitly. */
#define GB_POOL_MAX_PRE_REG_BYTES (8ULL * 1024ULL * 1024ULL * 1024ULL)
    if (gb_pool_configured_bytes == SIZE_MAX) {
        gb_pool_configured_bytes = gb_t2_pool_bytes * 85 / 100;
        if (gb_pool_configured_bytes > GB_POOL_MAX_PRE_REG_BYTES) {
            /* Suppress duplicate messages from sibling Ollama runner processes:
             * only print if no other process printed this within the last 5 s. */
            int _cap_warned = 0;
            {
                struct stat _st;
                if (stat("/run/greenboost/pool_cap_warned", &_st) == 0) {
                    struct timespec _now; clock_gettime(CLOCK_MONOTONIC, &_now);
                    if (_now.tv_sec - _st.st_mtim.tv_sec < 5) _cap_warned = 1;
                }
            }
            if (!_cap_warned) {
                fprintf(stderr, "[GreenBoost] T2 pool capped at 8 GB (full %zu MB would "
                        "block cudaMalloc at init; set GREENBOOST_T2_POOL_MB to override)\n",
                        gb_pool_configured_bytes >> 20);
                /* Touch sentinel - best-effort, failure is non-fatal */
                int _sfd = open("/run/greenboost/pool_cap_warned",
                                O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
                if (_sfd >= 0) close(_sfd);
            }
            gb_pool_configured_bytes = GB_POOL_MAX_PRE_REG_BYTES;
        }
    }

    /* Read safety_reserve_gb from kernel module - mirrors the check in gb_alloc_buf()
     * and gb_pin_user_buf().  Used by Path B / cuMemCreate VMM guards below. */
    {
        int res_gb = read_sysfs_int("/sys/module/greenboost/parameters/safety_reserve_gb");
        gb_safety_reserve_bytes = (res_gb > 0)
            ? (size_t)res_gb * 1024ULL * 1024ULL * 1024ULL
            : 8ULL  * 1024ULL * 1024ULL * 1024ULL; /* default: 8 GB when no kernel module */
        gb_log("safety_reserve: %zu MB (from %s)",
               gb_safety_reserve_bytes >> 20,
               res_gb > 0 ? "sysfs" : "default");
    }

    /* Add T3 (kernel module NVMe pool) to the reported virtual VRAM.
     * T3 pages are allocated by the GreenBoost kernel module from NVMe-backed
     * pages.  Reporting T2+T3 as virtual VRAM ensures Ollama's fit algorithm
     * places all layers on GPU (no CPU split) for models larger than T2 alone.
     * Path A (DMA-BUF pinned DDR) serves T3 allocations via pinned DDR overflow. */
    {
        int nvme_gb = read_sysfs_int("/sys/module/greenboost/parameters/nvme_pool_gb");
        if (nvme_gb > 0) {
            gb_virtual_vram_bytes += (size_t)nvme_gb * 1024ULL * 1024ULL * 1024ULL;
            g_nvme_pool_bytes = (size_t)nvme_gb * 1024ULL * 1024ULL * 1024ULL;
            gb_log("T3 NVMe pool: +%d GB added to virtual VRAM report", nvme_gb);
        }
    }

    /* Scale vram_headroom_bytes to 2% of physical VRAM (floor 128 MB, ceiling 512 MB)
     * if the user did not override via GREENBOOST_VRAM_HEADROOM_MB.
     * 2% = 98% VRAM cap: keeps the GPU from hitting 100% and triggering driver stalls.
     * Previous 5% (~95% cap) was unnecessarily conservative on large GPUs. */
    if (!headroom_from_env && gb_physical_vram_bytes > 0) {
        size_t pct2 = gb_physical_vram_bytes / 50;  /* 2% - 98% T1 VRAM cap */
        size_t floor_bytes  = 128ULL * 1024ULL * 1024ULL;
        size_t ceil_bytes   = 512ULL * 1024ULL * 1024ULL;
        if (pct2 < floor_bytes) pct2 = floor_bytes;
        if (pct2 > ceil_bytes)  pct2 = ceil_bytes;
        vram_headroom_bytes = pct2;
    }

    GB_NVTX_EVENT("SHIM_INIT", "SYSTEM", 0, 0,
        "greenboost shim initialised - T1_headroom=2pct Path_C_UVM_removed");

    /* Runtime API (cuda*) - live in libcudart, not libcuda.  F-ABI1: this is
     * a best-effort fallback only; the first cudart hook call re-resolves
     * from the app's own libcudart via gb_cudart_rebind(). */
    if (libcudart)
        gb_cudart_resolve_syms(libcudart);
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
     * Override: GREENBOOST_KV_RESERVE_MB env var or kernel module kv_reserve_mb.
     * NOTE: only auto-calculate when no explicit env-var override was given.
     * GREENBOOST_KV_RESERVE_MB=0 means "no reserve" and must not be overridden. */
    if (!g_kv_reserve_from_env && atomic_load_explicit(&g_kv_reserve_bytes, memory_order_relaxed) == 0) {
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
            /* Read gpu_count_per_node from sysfs - written by greenboost_setup.sh at insmod
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

    /* Startup banner - gated on debug mode.
     * With /etc/ld.so.preload the shim loads into every process on the system.
     * libcuda.so.1 is in ldconfig (NVIDIA driver installs it), so the shim
     * activates for ls, bash, systemd, etc.  Silent by default. */
    if (gb_debug) {
        fprintf(stderr, "[GreenBoost] v3.2 patched - vram_headroom=%zuMB kv_reserve=%zuMB(adaptive) kv_threshold=%zuMB virtual_vram=%zuMB use_dmabuf=%d debug=%d report_phys=%d\n",
                vram_headroom_bytes >> 20, g_kv_reserve_bytes >> 20,
                g_kv_size_threshold_bytes >> 20,
                gb_virtual_vram_bytes >> 20, gb_use_dmabuf, gb_debug, g_report_physical_vram);
        fprintf(stderr, "[GreenBoost] Alloc path counters: A=%u B=%u\n",
                gb_path_a_count, gb_path_b_count);
        fprintf(stderr, "[GreenBoost] Path A  (DMA-BUF pinned DDR): %s\n",
                !gb_use_dmabuf
                    ? "disabled (GREENBOOST_USE_DMA_BUF=0)"
                : gb_a0_disabled
                    ? (gb_use_dmabuf && real_cuMemHostRegister
                        ? "pinned sub-method (zero-copy disabled - not supported on this GPU/driver)"
                        : "UNAVAILABLE - zero-copy disabled and cuMemHostRegister missing")
                : (real_cudaImportExternalMemory && real_cudaExternalMemoryGetMappedBuffer)
                    ? "zero-copy sub-method active (cudaImportExternalMemory); pinned fallback ready"
                : (real_cuMemHostRegister
                    ? "pinned sub-method (libcudart not resolved - cudaImportExternalMemory unavailable)"
                    : "UNAVAILABLE - libcudart and cuMemHostRegister both missing"));
        fprintf(stderr, "[GreenBoost] Path B  (HostReg/no-kmod): %s\n",
                (!gb_no_hostreg && real_cuMemHostRegister && real_cuMemHostGetDevicePointer)
                    ? "available - mmap+cuMemHostRegister (containers/VMs)"
                : gb_no_hostreg ? "disabled (GREENBOOST_NO_HOSTREG=1)"
                : "unavailable (cuMemHostRegister not resolved)");
        fprintf(stderr, "[GreenBoost] Path C  (UVM)             : REMOVED - CPU compute forbidden; OOM returned when T2 exhausted\n");
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

    /* E1: Save libcuda handle for lazy PTX init.  cuModuleLoadData cannot be
     * called here because no CUDA context exists yet (cuInit() has not been
     * called by the application).  Actual module loading is deferred to the
     * first gb_kv_compress_d2t2 / gb_kv_decompress_t2tod call via
     * pthread_once(&g_ptx_init_once, gb_kv_ptx_init_lazy). */
    if (g_kv_compress_enabled)
        g_saved_libcuda = libcuda;

    /* Connect to cluster feeders from /etc/greenboost/cluster.conf.
     * Single-virtual-GPU model: feeder memory is aggregated into device 0's
     * reported total so Ollama schedules all layers on one device.  All compute
     * runs on the local GPU; overflow allocs spill to feeder T1 → local T2 →
     * feeder T2 → local T3 → feeder T3 via the cudaMalloc overflow path. */
    if (gb_netc_init() == 0) {
        int n = gb_netc_remote_gpu_count();
        for (int _ri = 0; _ri < n; _ri++) {
            uint64_t _free = 0, _total = 0;
            if (gb_netc_mem_info(_ri, &_free, &_total) == 0 && _total > 0) {
                g_cluster_remote_total_bytes += (size_t)_total;
                g_cluster_remote_free_bytes  += (size_t)_free;
                /* Subtract feeder T3 NVMe from the NVML context-sizing counters.
                 * T3 is slow NVMe - including it inflates default_num_ctx and causes
                 * KV cache allocations that exhaust host DDR (OOM), matching the
                 * same exclusion applied to local T3 in gb_t2_free_to_report(). */
                uint64_t _t3_free = 0, _t3_total = 0;
                gb_netc_t3_mem_info(_ri, &_t3_free, &_t3_total);
                g_cluster_remote_t1t2_total_bytes += (_total >= _t3_total)
                    ? (size_t)(_total - _t3_total) : (size_t)_total;
                g_cluster_remote_t1t2_free_bytes  += (_free  >= _t3_free)
                    ? (size_t)(_free  - _t3_free)  : (size_t)_free;
                /* Track feeder GPU VRAM (T1 only) separately for the clean
                 * "virtual VRAM = local + feeder" display figure. */
                uint64_t _t1_free = 0, _t1_total = 0;
                if (gb_netc_t1_mem_info(_ri, &_t1_free, &_t1_total) == 0)
                    g_cluster_remote_vram_bytes += (size_t)_t1_total;
            }
        }
        if (n > 0) {
            size_t _virt_vram_gb = (gb_physical_vram_bytes + g_cluster_remote_vram_bytes) >> 30;
            size_t _local_gb    = gb_physical_vram_bytes >> 30;
            size_t _feeder_gb   = g_cluster_remote_vram_bytes >> 30;
            fprintf(stderr, "[GreenBoost] Cluster: %d feeder(s) — virtual GPU: %zu GB VRAM "
                    "(%zu local + %zu feeder) total pool: +%zu GB remote\n",
                    n, _virt_vram_gb, _local_gb, _feeder_gb,
                    g_cluster_remote_total_bytes >> 30);
        }
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

        /* U9+U10: release prefix cache ref and emit REMOVED event */
        if (e->alloc_flags & GB_ALLOC_KV_CACHE) {
            gb_kv_cache_release((uint64_t)e->ptr);
            if (e->mapped_ptr) {
                uint64_t hash = gb_xxhash64(e->mapped_ptr, (e->size < 256) ? e->size : 256);
                gb_block_evt_emit(GB_BLOCK_EVT_REMOVED, hash, 0, 0);
            }
        }

        /* Path A/B: pool entries return to free-list; standalone entries unmap. */
        if (e->mapped_ptr) {
            if (gb_pool_contains(e->ptr)) {
                gb_pool_free(e->ptr, e->size);
            } else {
                if (real_cuMemHostUnregister)
                    real_cuMemHostUnregister(e->mapped_ptr);
                munmap(e->mapped_ptr, e->size);
            }
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

/* U1: ARC-based KV cache partial eviction (vLLM ARC policy pattern).
 * Evicts cold KV entries (accessed ≤1 time = T1 set) first by LRU timestamp,
 * then hot entries (T2 set) until target_bytes freed.
 * Returns bytes actually freed. Used in R4 OOM path for targeted eviction
 * instead of the bulk gb_htable_flush which blasts weight entries too. */
static size_t gb_htable_evict_kv_arc(size_t target_bytes)
{
    GB_NVTX_PUSH("GB:evict_KV_ARC", GB_NVTX_COLOR_T2);
#define ARC_EVICT_MAX 512
    typedef struct { uint32_t idx; uint32_t access_ts; uint16_t access_count; size_t sz; } arc_cand_t;
    arc_cand_t t0_cands[ARC_EVICT_MAX];  /* U11: frozen cold (cold_epochs > threshold) */
    arc_cand_t t1_cands[ARC_EVICT_MAX];  /* accessed ≤1 time (cold) */
    arc_cand_t t2_cands[ARC_EVICT_MAX];  /* accessed >1 time (warm) */
    int t0n = 0, t1n = 0, t2n = 0;

    /* Collect KV cache candidates into three sets */
    for (uint32_t i = 0; i < HT_SIZE && (t0n + t1n + t2n < ARC_EVICT_MAX * 3); i++) {
        gb_ht_entry_t *e = &gb_htable[i];
        if (e->ptr == 0 || e->ptr == HT_TOMBSTONE) continue;
        if (!(e->alloc_flags & GB_ALLOC_KV_CACHE)) continue;
        arc_cand_t c = { i, e->access_ts, e->access_count, e->size };
        /* U11: frozen-cold pre-pass: entries inactive for ≥ GB_EPOCH_COLD_THRESHOLD epochs */
        if (gb_ht_is_cold(i) && t0n < ARC_EVICT_MAX)     t0_cands[t0n++] = c;
        else if (e->access_count <= 1 && t1n < ARC_EVICT_MAX) t1_cands[t1n++] = c;
        else if (t2n < ARC_EVICT_MAX)                          t2_cands[t2n++] = c;
    }

    /* Sort each set by access_ts ascending (oldest first) */
    for (int a = 0; a < t0n - 1; a++)
        for (int b = a + 1; b < t0n; b++)
            if (t0_cands[b].access_ts < t0_cands[a].access_ts) {
                arc_cand_t tmp = t0_cands[a]; t0_cands[a] = t0_cands[b]; t0_cands[b] = tmp;
            }
    for (int a = 0; a < t1n - 1; a++)
        for (int b = a + 1; b < t1n; b++)
            if (t1_cands[b].access_ts < t1_cands[a].access_ts) {
                arc_cand_t tmp = t1_cands[a]; t1_cands[a] = t1_cands[b]; t1_cands[b] = tmp;
            }
    for (int a = 0; a < t2n - 1; a++)
        for (int b = a + 1; b < t2n; b++)
            if (t2_cands[b].access_ts < t2_cands[a].access_ts) {
                arc_cand_t tmp = t2_cands[a]; t2_cands[a] = t2_cands[b]; t2_cands[b] = tmp;
            }

    /* F3: collect madvise-pending regions for batch sweep after per-entry eviction.
     * Pool-backed entries are individually freed, then we call madvise(MADV_FREE)
     * on all collected regions in one pass - reduces syscall count from N to 1. */
#define MADVISE_BATCH_MAX 256
    struct { void *ptr; size_t sz; } madvise_batch[MADVISE_BATCH_MAX];
    int madvise_n = 0;

    size_t freed = 0;
    int pass;
    /* Pass 0 = U11 frozen-cold, pass 1 = cold (T1 set), pass 2 = warm (T2 set) */
    for (pass = 0; pass < 3 && freed < target_bytes; pass++) {
        arc_cand_t *cands = (pass == 0) ? t0_cands : (pass == 1) ? t1_cands : t2_cands;
        int n = (pass == 0) ? t0n : (pass == 1) ? t1n : t2n;
        for (int ci = 0; ci < n && freed < target_bytes; ci++) {
            uint32_t idx = cands[ci].idx;
            gb_ht_entry_t *e = &gb_htable[idx];
            pthread_mutex_t *lk = ht_lock(idx);
            pthread_mutex_lock(lk);
            if (e->ptr == 0 || e->ptr == HT_TOMBSTONE ||
                !(e->alloc_flags & GB_ALLOC_KV_CACHE)) {
                pthread_mutex_unlock(lk); continue;
            }
            size_t sz = e->size;
            if (e->ext_mem && real_cudaDestroyExternalMemory)
                real_cudaDestroyExternalMemory(e->ext_mem);
            if (e->mapped_ptr) {
                if (gb_pool_contains(e->ptr)) {
                    gb_pool_free(e->ptr, sz);
                    /* F3: collect for batch madvise sweep */
                    if (madvise_n < MADVISE_BATCH_MAX)
                        madvise_batch[madvise_n++] = (typeof(madvise_batch[0])){ e->mapped_ptr, sz };
                } else {
                    if (real_cuMemHostUnregister) real_cuMemHostUnregister(e->mapped_ptr);
                    munmap(e->mapped_ptr, sz);
                }
            }
            if (e->fd >= 0) close(e->fd);
            /* U12: record ghost before tombstone so refault check can match */
            gb_ghost_record((uint64_t)e->ptr, sz);
            /* U11: count frozen-cold evictions */
            if (pass == 0) atomic_fetch_add_explicit(&g_cold_evict_cnt, 1, memory_order_relaxed);
            e->ptr = HT_TOMBSTONE; e->fd = -1; e->mapped_ptr = NULL; e->ext_mem = NULL;
            pthread_mutex_unlock(lk);
            freed += sz;
            atomic_fetch_sub_explicit(&gb_t2_overflow_bytes,
                                       (size_t)(sz < (size_t)8 * 1024 ? 0 : sz - 4096),
                                       memory_order_relaxed);
            /* N11: keep SWA window counter in sync with eviction */
            atomic_fetch_sub_explicit(&g_kv_t2_live_bytes, sz, memory_order_relaxed);
        }
    }

    /* F3: batch madvise(MADV_FREE) for all collected pool-backed freed regions. */
    GB_NVTX_EVENT("EVICT_BATCH_MADVISE", "T2_DDR", freed >> 20, madvise_n, "batch_madvise");
    for (int mi = 0; mi < madvise_n; mi++)
        madvise(madvise_batch[mi].ptr, madvise_batch[mi].sz, MADV_FREE);

    gb_log("ARC evict: freed %zu MB (%d frozen-cold + %d T1-cold + %d T2-warm, target %zu MB)"
           " batch_madvise=%d", freed >> 20, t0n, t1n, t2n, target_bytes >> 20, madvise_n);
    GB_NVTX_EVENT("EVICT_ARC_KV", "T2_DDR", freed >> 20, 0, "arc_kv_eviction_complete");
    GB_NVTX_POP();
    return freed;
#undef ARC_EVICT_MAX
}

/* VCM-01: pthread_once callback - wraps gb_shim_init() for deferred activation. */
static void gb_resume_init_locked(void) { gb_shim_init(); }

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

    /* Tear down pre-registered T2 pool - single cuMemHostUnregister for whole slab.
     * Must run AFTER gb_htable_flush (which calls gb_pool_free for pool entries)
     * and BEFORE close(gb_dev_fd) so the kernel sees the pin released cleanly. */
    if (gb_t2_reg_pool.initialized) {
        if (real_cuMemHostUnregister)
            real_cuMemHostUnregister(gb_t2_reg_pool.base);
        munmap(gb_t2_reg_pool.base, gb_t2_reg_pool.total);
        if (gb_t2_reg_pool.dmabuf_fd >= 0)
            close(gb_t2_reg_pool.dmabuf_fd);
        gb_t2_reg_pool.initialized = 0;
        gb_log("T2 pool released: %zu MB", gb_t2_reg_pool.total >> 20);
    }

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
 * consistent (if slightly stale) snapshot - acceptable for a heuristic. */
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
        if (total_vram < 100ULL * 1024 * 1024 * 1024) gb_physical_vram_bytes = total_vram;
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
         * or every GB_MEMINFO_REFRESH_MS ms - avoids a CUDA round-trip per alloc. */
        if ((cnt & (GB_MEMINFO_REFRESH_ALLOCS - 1u)) == 0) {
            gb_refresh_meminfo_cache();
        } else {
            /* Time-based refresh: check elapsed ms against last cache write */
            uint64_t last_ms = atomic_load_explicit(&g_cached_meminfo_ms,
                                                    memory_order_relaxed);
            if (last_ms == 0) {
                gb_refresh_meminfo_cache();  /* first call - populate cache */
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
        /* Cache not populated (cuMemGetInfo failed - context not yet warm).
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
            kv_reserve = 0;  /* model may fit entirely in T1 - let weights use all of it */
        }
        /* When t2_used > 0: keep kv_reserve = 25% of VRAM (set by auto-scaler at init).
         * No additional cap needed - 25% is already the correct value. */
    }

    /* Inference-time occupancy collapse: if phase is INFERENCE/STEADY but
     * g_kv_allocated_t1_bytes is still 0 (KV was allocated by the CUDA runtime
     * directly, bypassing the shim's overflow path), infer KV location from
     * actual free VRAM.  If free_vram < headroom + kv_reserve + 128 MB slack,
     * VRAM is more occupied than the reserve allows - KV must already be in T1.
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

    /* GREENBOOST_FEEDER_EXCLUSIVE=1 — owner-directive mode (2026-07-06):
     * force every allocation ≥1 MB off local VRAM so the ENTIRE working set
     * (weights + KV + compute buffers) is feeder-resident.  With arena
     * allocators, per-kernel mixed-args staging cannot know transfer extents;
     * full co-residency makes every dispatched kernel's args coherent on the
     * feeder — the host becomes a pure orchestrator and the feeder GPU does
     * the compute, reading its own GDDR7/DDR locally. */
    if (gb_feeder_exclusive() && bytesize > 0) {
        /* Force EVERY buffer (not just ≥1 MB) onto the feeder.  A kernel that
         * mixes a remote weight with a small LOCAL output/scratch pointer
         * faults on the feeder — the local ptr isn't a 0xAA fake and can't be
         * relocated.  Full co-residency makes every arg relocatable (found
         * live: mul_mat_vec_q got only 2 of 3 pointer args relocated because
         * its <1 MB output stayed local → err=700). */
        gb_log("VRAM: req=%zuB → OVERFLOW (feeder-exclusive mode)", bytesize);
        return 1;
    }

    /* KV cache reservation: weights spill to T2 while kv_reserve > 0, leaving
     * T1 headroom for the KV cache.  KV is read+written every generation step -
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
    /* Phase-aware T1 workspace reserve: hold VRAM back for per-step compute
     * workspace ONLY during model load (INIT/MODEL_LOAD).  Once the phase
     * advances to INFERENCE the reserve is released (treated as 0) so the
     * freed VRAM absorbs activations/workspace instead of forcing them to T2.
     * See g_workspace_reserve_bytes comment for the rationale/measurements. */
    size_t ws_reserve = 0;
    if (cur_phase <= GB_PHASE_MODEL_LOAD)
        ws_reserve = atomic_load_explicit(&g_workspace_reserve_bytes, memory_order_relaxed);

    {
        if (bytesize + vram_headroom_bytes + kv_reserve + ws_reserve > free_vram) {
            gb_log("VRAM: req=%zuMB free=%zuMB headroom=%zuMB kv_reserve=%zuMB(eff) ws_reserve=%zuMB phase=%d oversize=%d → OVERFLOW",
                   bytesize >> 20, free_vram >> 20, vram_headroom_bytes >> 20,
                   kv_reserve >> 20, ws_reserve >> 20, cur_phase,
                   (gb_physical_vram_bytes > 0 && bytesize > gb_physical_vram_bytes));
            if (bytesize >= GB_DECISION_LOG_MIN_BYTES) {
                char _dbuf[160];
                snprintf(_dbuf, sizeof(_dbuf),
                    "free=%zuMB headroom=%zuMB kv_rsv=%zuMB phys=%zuMB phase=%d",
                    free_vram >> 20, vram_headroom_bytes >> 20, kv_reserve >> 20,
                    gb_physical_vram_bytes >> 20, cur_phase);
                GB_NVTX_EVENT("ALLOC_DECISION", "OVERFLOW", bytesize >> 20, 0, _dbuf);
            }
            return 1;
        }
        gb_log("VRAM: req=%zuMB free=%zuMB phase=%d → fits in T1",
               bytesize >> 20, free_vram >> 20, cur_phase);
        if (bytesize >= GB_DECISION_LOG_MIN_BYTES) {
            char _dbuf[160];
            snprintf(_dbuf, sizeof(_dbuf),
                "free=%zuMB headroom=%zuMB kv_rsv=%zuMB phys=%zuMB phase=%d",
                free_vram >> 20, vram_headroom_bytes >> 20, kv_reserve >> 20,
                gb_physical_vram_bytes >> 20, cur_phase);
            GB_NVTX_EVENT("ALLOC_DECISION", "T1_FITS", bytesize >> 20, 0, _dbuf);
        }
    }
    return 0;
}

/* Track a large allocation that landed in T1 VRAM (not overflow) as KV cache.
 * Heuristic: a large alloc (>= g_kv_size_threshold_bytes) that arrives after a
 * GB_PHASE_QUIET_GAP_MS quiet period (no overflow allocs) is almost certainly
 * the KV cache being allocated after the model weights have been pushed to T2.
 * We increment g_kv_allocated_t1_bytes so gb_needs_overflow() reduces its
 * effective_reserve accordingly - preventing double-counting and freeing VRAM. */

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

    /* Never tag T1 allocs as KV during INIT or MODEL_LOAD - those are weight
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
               " - adaptive reserve now %zu MB",
               bytesize >> 20, (unsigned long long)dptr,
               (unsigned long long)(gap_ms == (uint64_t)-1 ? 0 : gap_ms),
               (kv_in >= kv_rsv ? 0 : (kv_rsv - kv_in)) >> 20);
    }
}

/* ── Phase 4: AVX2 host-side memcpy ──────────────────────────────────────────
 * Used in Path A and Path B host-side setup copies during tier migration.
 * Falls back to plain memcpy on CPUs without AVX2 (compile-time #ifdef). */
#ifdef __AVX2__
#include <immintrin.h>
static void gb_avx2_memcpy(void *dst, const void *src, size_t size)
{
    const __m256i *s = (const __m256i *)src;
          __m256i *d = (__m256i *)dst;
    size_t n = size / 128;
    for (size_t i = 0; i < n; i++, s += 4, d += 4) {
        __m256i r0 = _mm256_load_si256(s + 0);
        __m256i r1 = _mm256_load_si256(s + 1);
        __m256i r2 = _mm256_load_si256(s + 2);
        __m256i r3 = _mm256_load_si256(s + 3);
        _mm256_stream_si256(d + 0, r0);
        _mm256_stream_si256(d + 1, r1);
        _mm256_stream_si256(d + 2, r2);
        _mm256_stream_si256(d + 3, r3);
    }
    _mm_sfence();
    memcpy((char *)d, (const char *)s, size % 128);
}
#else
static void gb_avx2_memcpy(void *dst, const void *src, size_t size)
{
    memcpy(dst, src, size);
}
#endif

/* ── Phase 4: multi-tensor batch flush ───────────────────────────────────────
 * Flushes g_migrate_batch: sorts by size desc (fills PCIe bus better),
 * then issues cudaMemcpyAsync for each entry on the migration stream. */
typedef int (*pfn_cudaMemcpyAsync_migrate_t)(void *, const void *, size_t, int, void *);
static void gb_batch_flush(struct gb_migrate_batch *batch)
{
    if (!batch || batch->count == 0) return;

    /* Sort descending by size (insertion sort - small N ≤ 64) */
    for (int i = 1; i < batch->count; i++) {
        struct gb_migrate_entry key = batch->entries[i];
        int j = i - 1;
        while (j >= 0 && batch->entries[j].size < key.size) {
            batch->entries[j + 1] = batch->entries[j];
            j--;
        }
        batch->entries[j + 1] = key;
    }

    /* Issue async copies on the batch stream */
    /* F-ABI1 fix: use the robustly-resolved global instead of a local
     * dlsym(RTLD_NEXT,...) - see real_cudaMemcpyAsync's declaration comment. */
    pfn_cudaMemcpyAsync_migrate_t _real_cma =
        (pfn_cudaMemcpyAsync_migrate_t)real_cudaMemcpyAsync;
    if (!_real_cma)
        _real_cma = (pfn_cudaMemcpyAsync_migrate_t)dlsym(RTLD_NEXT, "cudaMemcpyAsync");

    if (_real_cma && batch->stream) {
        for (int i = 0; i < batch->count; i++) {
            struct gb_migrate_entry *e = &batch->entries[i];
            /* cudaMemcpyDeviceToHost = 2 */
            _real_cma(e->dst, (void *)e->src, e->size, 2, (void *)batch->stream);
        }
        if (real_cudaStreamSynchronize)
            real_cudaStreamSynchronize(batch->stream);
    } else {
        /* No stream / no cudaMemcpyAsync - fall back to gb_avx2_memcpy for host-side */
        for (int i = 0; i < batch->count; i++) {
            struct gb_migrate_entry *e = &batch->entries[i];
            if (e->dst && e->src)
                gb_avx2_memcpy(e->dst, (const void *)e->src, e->size);
        }
    }

    gb_log("batch_migrate: flushed %d entries", batch->count);
    batch->count = 0;
}

/* ── F2: GDS offset allocator - watermark-based, vLLM SimpleCPUOffload pattern ──
 * Tracks next-free file offset within the T3 backing file.  Thread-safe via g_gds_lock.
 * g_nvme_pool_bytes is set once at init from sysfs nvme_pool_gb. */

static pthread_mutex_t g_gds_lock        = PTHREAD_MUTEX_INITIALIZER;
static size_t          g_gds_next_offset = 0;

/* Side table: maps GDS-allocated CUdeviceptr → file offset + size.
 * GDS allocations are rare (last resort), so 512 slots is ample. */
#define GDS_SIDEHT_SIZE 512
typedef struct { CUdeviceptr ptr; off_t file_off; size_t size; int in_use; } gds_slab_t;
static gds_slab_t g_gds_sidetable[GDS_SIDEHT_SIZE];

/* Base for GDS fake pointers - above the feeder remote range */
#define GDS_PTR_BASE 0xBB0000000000ULL
static _Atomic CUdeviceptr g_gds_next_fake = GDS_PTR_BASE;

/* Allocate a file slab of `size` bytes (page-aligned).
 * Returns file offset on success, -1 if pool is full. */
static off_t gb_gds_alloc_slab(size_t size)
{
    size_t aligned = (size + 4095) & ~(size_t)4095;
    pthread_mutex_lock(&g_gds_lock);
    if (g_nvme_pool_bytes == 0 || g_gds_next_offset + aligned > g_nvme_pool_bytes) {
        pthread_mutex_unlock(&g_gds_lock);
        return (off_t)-1;
    }
    off_t off = (off_t)g_gds_next_offset;
    g_gds_next_offset += aligned;
    pthread_mutex_unlock(&g_gds_lock);
    return off;
}

/* Register a GDS slab in the side table (called after successful alloc). */
static void gb_gds_sideht_insert(CUdeviceptr ptr, off_t off, size_t size)
{
    pthread_mutex_lock(&g_gds_lock);
    for (int i = 0; i < GDS_SIDEHT_SIZE; i++) {
        if (!g_gds_sidetable[i].in_use) {
            g_gds_sidetable[i] = (gds_slab_t){ ptr, off, size, 1 };
            pthread_mutex_unlock(&g_gds_lock);
            return;
        }
    }
    pthread_mutex_unlock(&g_gds_lock);
}

/* Look up a GDS slab by fake pointer.  Returns non-zero on success. */
static int gb_gds_sideht_lookup(CUdeviceptr ptr, off_t *out_off, size_t *out_size)
{
    pthread_mutex_lock(&g_gds_lock);
    for (int i = 0; i < GDS_SIDEHT_SIZE; i++) {
        if (g_gds_sidetable[i].in_use && g_gds_sidetable[i].ptr == ptr) {
            *out_off  = g_gds_sidetable[i].file_off;
            *out_size = g_gds_sidetable[i].size;
            pthread_mutex_unlock(&g_gds_lock);
            return 1;
        }
    }
    pthread_mutex_unlock(&g_gds_lock);
    return 0;
}

static inline int gb_is_gds_ptr(CUdeviceptr ptr)
{
    return (ptr & 0xFF0000000000ULL) == GDS_PTR_BASE;
}

/* ── Phase 3: GDS helper functions ───────────────────────────────────────────
 * Direct GPU↔NVMe transfers via cuFile, bypassing CPU RAM entirely.
 * g_gds_ok must be 1 (set in gb_shim_init) before calling these. */
static int gb_gds_write(CUdeviceptr src, size_t size, off_t file_offset)
{
    if (!g_gds_ok || !f_cuFileWrite || !g_cufile_handle) return -1;
    GB_NVTX_PUSH("GB:gds_write", GB_NVTX_COLOR_T3);
    long n = f_cuFileWrite(g_cufile_handle, (void *)src, size, file_offset, 0);
    int ok = (n >= 0 && (size_t)n == size);
    GB_NVTX_EVENT(ok ? "GDS_WRITE_OK" : "GDS_WRITE_FAIL",
                  "T3_GDS", size >> 20, src, ok ? "cuFileWrite_ok" : "cuFileWrite_err");
    GB_NVTX_POP();
    return ok ? 0 : -1;
}

static int gb_gds_read(CUdeviceptr dst, size_t size, off_t file_offset)
{
    if (!g_gds_ok || !f_cuFileRead || !g_cufile_handle) return -1;
    GB_NVTX_PUSH("GB:gds_read", GB_NVTX_COLOR_T3);
    long n = f_cuFileRead(g_cufile_handle, (void *)dst, size, file_offset, 0);
    int ok = (n >= 0 && (size_t)n == size);
    GB_NVTX_EVENT(ok ? "GDS_READ_OK" : "GDS_READ_FAIL",
                  "T3_GDS", size >> 20, dst, ok ? "cuFileRead_ok" : "cuFileRead_err");
    GB_NVTX_POP();
    return ok ? 0 : -1;
}

/* ── Phase 2: K/V int8 absmax quantisation (E1) ──────────────────────────────
 *
 * Embedded PTX kernels targeting sm_89 (Ada Lovelace, RTX 5070 Laptop).
 * Loaded lazily on first use via cuModuleLoadData (no separate nvcc step).
 *
 * Algorithm (per row of GB_KV_COMPRESS_ROW = 128 fp16 elements):
 *   Quant:    scale = max(|x|) / 127  (parallel reduction in shared memory)
 *             int8[i] = clamp(round(fp16[i] / scale), -127, 127)
 *   Dequant:  fp32 = int8[i] * scale  →  convert to fp16
 *
 * Grid  : (n_rows, 1, 1) blocks
 * Block : (128, 1, 1) threads  - one thread per element, one block per row
 */
static const char gb_kv_ptx_src[] =
    ".version 8.0\n"
    ".target sm_89\n"
    ".address_size 64\n"
    "\n"
    /* ── Absmax quantise: fp16 → int8 ────────────────────────────── */
    ".visible .entry gb_absmax_quant(\n"
    "    .param .u64 src_fp16,\n"
    "    .param .u64 dst_int8,\n"
    "    .param .u64 scales,\n"
    "    .param .u64 n_rows\n"
    ")\n"
    "{\n"
    "    .reg .u64   %rd<10>;\n"
    "    .reg .u32   %r<8>;\n"
    "    .reg .f32   %f<8>;\n"
    "    .reg .pred  %p<6>;\n"
    "    .shared .align 4 .f32 sh[128];\n"
    /* row/col from built-ins */
    "    mov.u32     %r0, %ctaid.x;\n"    /* row */
    "    mov.u32     %r1, %tid.x;\n"     /* col 0..127 */
    /* bounds check */
    "    ld.param.u64 %rd0, [n_rows];\n"
    "    cvt.u64.u32 %rd1, %r0;\n"
    "    setp.ge.u64 %p0, %rd1, %rd0;\n"
    "    @%p0 bra QEND;\n"
    /* elem_idx = row*128 + col */
    "    mad.lo.u32  %r2, %r0, 128, %r1;\n"
    /* load fp16 element → fp32 */
    "    ld.param.u64 %rd2, [src_fp16];\n"
    "    cvt.u64.u32 %rd3, %r2;\n"
    "    shl.b64     %rd4, %rd3, 1;\n"   /* *2 bytes per fp16 */
    "    add.u64     %rd5, %rd2, %rd4;\n"
    "    ld.global.u16 %r3, [%rd5];\n"
    "    cvt.f32.f16 %f0, %r3;\n"        /* fp16 → fp32 */
    /* abs(f0) → shared[tid] */
    "    abs.f32     %f1, %f0;\n"
    "    mov.u64     %rd6, sh;\n"
    "    cvta.shared.u64 %rd7, %rd6;\n"  /* generic addr of sh[0] */
    "    cvt.u64.u32 %rd8, %r1;\n"
    "    shl.b64     %rd8, %rd8, 2;\n"   /* *4 bytes per f32 */
    "    add.u64     %rd9, %rd7, %rd8;\n"/* rd9 = &sh[tid] */
    "    st.f32      [%rd9], %f1;\n"
    "    bar.sync    0;\n"
    /* parallel reduction - 7 steps for 128 threads */
    "    setp.lt.u32 %p1, %r1, 64;\n"
    "    @%p1 ld.f32 %f2, [%rd9 + 256];\n"  /* sh[tid+64] */
    "    @%p1 max.f32 %f1, %f1, %f2;\n"
    "    @%p1 st.f32 [%rd9], %f1;\n"
    "    bar.sync    0;\n"
    "    setp.lt.u32 %p1, %r1, 32;\n"
    "    @%p1 ld.f32 %f2, [%rd9 + 128];\n"
    "    @%p1 max.f32 %f1, %f1, %f2;\n"
    "    @%p1 st.f32 [%rd9], %f1;\n"
    "    bar.sync    0;\n"
    "    setp.lt.u32 %p1, %r1, 16;\n"
    "    @%p1 ld.f32 %f2, [%rd9 + 64];\n"
    "    @%p1 max.f32 %f1, %f1, %f2;\n"
    "    @%p1 st.f32 [%rd9], %f1;\n"
    "    bar.sync    0;\n"
    "    setp.lt.u32 %p1, %r1, 8;\n"
    "    @%p1 ld.f32 %f2, [%rd9 + 32];\n"
    "    @%p1 max.f32 %f1, %f1, %f2;\n"
    "    @%p1 st.f32 [%rd9], %f1;\n"
    "    bar.sync    0;\n"
    "    setp.lt.u32 %p1, %r1, 4;\n"
    "    @%p1 ld.f32 %f2, [%rd9 + 16];\n"
    "    @%p1 max.f32 %f1, %f1, %f2;\n"
    "    @%p1 st.f32 [%rd9], %f1;\n"
    "    bar.sync    0;\n"
    "    setp.lt.u32 %p1, %r1, 2;\n"
    "    @%p1 ld.f32 %f2, [%rd9 + 8];\n"
    "    @%p1 max.f32 %f1, %f1, %f2;\n"
    "    @%p1 st.f32 [%rd9], %f1;\n"
    "    bar.sync    0;\n"
    "    setp.eq.u32 %p1, %r1, 0;\n"     /* p1 = (tid == 0) from here on */
    "    @%p1 ld.f32 %f2, [%rd9 + 4];\n"
    "    @%p1 max.f32 %f1, %f1, %f2;\n"
    "    @%p1 st.f32 [%rd9], %f1;\n"
    "    bar.sync    0;\n"
    /* sh[0] = max absval for this row */
    "    ld.f32      %f3, [%rd7];\n"     /* f3 = max_absval */
    /* scale = max_absval / 127.0 */
    "    div.approx.f32 %f5, %f3, 127.0;\n"
    /* thread 0: write scale */
    "    @!%p1 bra SKIP_SCALE;\n"
    "    ld.param.u64 %rd2, [scales];\n"
    "    cvt.u64.u32 %rd3, %r0;\n"
    "    shl.b64     %rd3, %rd3, 2;\n"
    "    add.u64     %rd3, %rd2, %rd3;\n"
    "    st.global.f32 [%rd3], %f5;\n"
    "SKIP_SCALE:\n"
    /* quantize: q = round(f0 / scale), clamped to [-127, 127] */
    "    setp.eq.f32 %p2, %f5, 0.0;\n"
    "    mov.u32     %r4, 0;\n"
    "    @%p2 bra QWRITE;\n"
    "    div.approx.f32 %f6, %f0, %f5;\n"
    "    cvt.rni.s32.f32 %r4, %f6;\n"   /* round to nearest int */
    "    setp.gt.s32 %p3, %r4, 127;\n"
    "    @%p3 mov.u32 %r4, 127;\n"
    "    setp.lt.s32 %p4, %r4, -127;\n"
    "    @%p4 mov.s32 %r4, -127;\n"
    "QWRITE:\n"
    "    ld.param.u64 %rd2, [dst_int8];\n"
    "    cvt.u64.u32 %rd3, %r2;\n"
    "    add.u64     %rd3, %rd2, %rd3;\n"
    "    st.global.s8 [%rd3], %r4;\n"
    "QEND:\n"
    "    ret;\n"
    "}\n"
    "\n"
    /* ── Absmax dequantise: int8 → fp16 ──────────────────────────── */
    ".visible .entry gb_absmax_dequant(\n"
    "    .param .u64 src_int8,\n"
    "    .param .u64 scales,\n"
    "    .param .u64 dst_fp16,\n"
    "    .param .u64 n_rows\n"
    ")\n"
    "{\n"
    "    .reg .u64   %rd<10>;\n"
    "    .reg .u32   %r<6>;\n"
    "    .reg .f32   %f<4>;\n"
    "    .reg .pred  %p<2>;\n"
    "    mov.u32     %r0, %ctaid.x;\n"
    "    mov.u32     %r1, %tid.x;\n"
    "    ld.param.u64 %rd0, [n_rows];\n"
    "    cvt.u64.u32 %rd1, %r0;\n"
    "    setp.ge.u64 %p0, %rd1, %rd0;\n"
    "    @%p0 bra DEND;\n"
    /* elem_idx = row*128 + col */
    "    mad.lo.u32  %r2, %r0, 128, %r1;\n"
    /* load per-row scale */
    "    ld.param.u64 %rd2, [scales];\n"
    "    cvt.u64.u32 %rd3, %rd1;\n"
    "    shl.b64     %rd3, %rd3, 2;\n"
    "    add.u64     %rd3, %rd2, %rd3;\n"
    "    ld.global.f32 %f0, [%rd3];\n"   /* f0 = scale */
    /* load int8 element */
    "    ld.param.u64 %rd4, [src_int8];\n"
    "    cvt.u64.u32 %rd5, %r2;\n"
    "    add.u64     %rd5, %rd4, %rd5;\n"
    "    ld.global.s8 %r3, [%rd5];\n"
    /* dequantize: fp32 = int8 * scale → fp16 */
    "    cvt.f32.s32 %f1, %r3;\n"
    "    mul.f32     %f2, %f1, %f0;\n"
    "    cvt.rn.f16.f32 %r4, %f2;\n"    /* fp32 → fp16 */
    /* store fp16 */
    "    ld.param.u64 %rd6, [dst_fp16];\n"
    "    cvt.u64.u32 %rd7, %r2;\n"
    "    shl.b64     %rd7, %rd7, 1;\n"
    "    add.u64     %rd7, %rd6, %rd7;\n"
    "    st.global.u16 [%rd7], %r4;\n"
    "DEND:\n"
    "    ret;\n"
    "}\n";

/* Trampoline for pthread_once - called the first time a compress/decompress
 * function runs, by which point cuInit() has been called and a CUDA context
 * exists.  Uses g_saved_libcuda stored during gb_shim_init(). */
static void gb_kv_ptx_init_lazy(void)
{
    if (g_saved_libcuda)
        gb_kv_ptx_init(g_saved_libcuda);
}

/* Load the embedded PTX module.  Must be called after a CUDA context exists
 * (i.e. after cuInit()).  Use gb_kv_ptx_init_lazy / pthread_once instead of
 * calling directly. */
static void gb_kv_ptx_init(void *libcuda)
{
    if (!real_cuModuleLoadData)
        real_cuModuleLoadData = (pfn_cuModuleLoadData)dlsym(libcuda, "cuModuleLoadData");
    if (!real_cuModuleGetFunction)
        real_cuModuleGetFunction = (pfn_cuModuleGetFunction)dlsym(libcuda, "cuModuleGetFunction");

    if (!real_cuModuleLoadData || !real_cuModuleGetFunction) {
        fprintf(stderr, "[GreenBoost] E1: cuModuleLoadData unavailable - "
                "K/V PTX compression disabled\n");
        g_kv_compress_enabled = 0;
        return;
    }

    CUresult rc = real_cuModuleLoadData(&g_gb_ptx_module, gb_kv_ptx_src);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[GreenBoost] E1: cuModuleLoadData failed (%d) - "
                "K/V PTX compression disabled\n", rc);
        g_kv_compress_enabled = 0;
        return;
    }

    rc  = real_cuModuleGetFunction(&g_gb_absmax_quant_fn,  g_gb_ptx_module, "gb_absmax_quant");
    rc |= real_cuModuleGetFunction(&g_gb_absmax_dequant_fn, g_gb_ptx_module, "gb_absmax_dequant");
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[GreenBoost] E1: cuModuleGetFunction failed (%d) - "
                "K/V PTX compression disabled\n", rc);
        g_kv_compress_enabled = 0;
        return;
    }

    fprintf(stderr, "[GreenBoost] E1: K/V absmax PTX kernels loaded "
            "(gb_absmax_quant + gb_absmax_dequant)\n");
}

/* ── Phase 2: compress fp16 K/V tensor (T1 VRAM) → int8 (T2 host buffer) ─── */
static int gb_kv_compress_d2t2(CUdeviceptr src_fp16, uint8_t *dst_int8,
                                 float *scales, uint64_t n_elems)
{
    /* Lazy PTX init: runs once after cuInit() has been called by the app. */
    if (g_kv_compress_enabled && g_saved_libcuda)
        pthread_once(&g_ptx_init_once, gb_kv_ptx_init_lazy);

    if (!g_gb_absmax_quant_fn || !real_cuLaunchKernel)
        return -1;

    /* n_rows = n_elems / GB_KV_COMPRESS_ROW (must be exact multiple) */
    if (n_elems % GB_KV_COMPRESS_ROW != 0) return -1;
    uint64_t n_rows = n_elems / GB_KV_COMPRESS_ROW;
    if (n_rows == 0 || n_rows > UINT32_MAX) return -1;

    /* dst_int8 and scales must be device-accessible (cuMemHostRegister'd) */
    CUdeviceptr d_dst   = (CUdeviceptr)(uintptr_t)dst_int8;
    CUdeviceptr d_scale = (CUdeviceptr)(uintptr_t)scales;

    void *args[] = { &src_fp16, &d_dst, &d_scale, &n_rows };
    CUresult rc = real_cuLaunchKernel(g_gb_absmax_quant_fn,
                                      (unsigned int)n_rows, 1, 1,
                                      GB_KV_COMPRESS_ROW, 1, 1,
                                      0, NULL, args, NULL);
    return (rc == CUDA_SUCCESS) ? 0 : -1;
}

/* ── Phase 2: decompress int8 (T2 host buffer) → fp16 (T1 VRAM) ─────────── */
static int gb_kv_decompress_t2tod(uint8_t *src_int8, const float *scales,
                                   CUdeviceptr dst_fp16, uint64_t n_elems)
{
    /* Lazy PTX init: runs once after cuInit() has been called by the app. */
    if (g_kv_compress_enabled && g_saved_libcuda)
        pthread_once(&g_ptx_init_once, gb_kv_ptx_init_lazy);

    if (!g_gb_absmax_dequant_fn || !real_cuLaunchKernel)
        return -1;

    if (n_elems % GB_KV_COMPRESS_ROW != 0) return -1;
    uint64_t n_rows = n_elems / GB_KV_COMPRESS_ROW;
    if (n_rows == 0 || n_rows > UINT32_MAX) return -1;

    CUdeviceptr d_src   = (CUdeviceptr)(uintptr_t)src_int8;
    CUdeviceptr d_scale = (CUdeviceptr)(uintptr_t)scales;

    void *args[] = { &d_src, &d_scale, &dst_fp16, &n_rows };
    CUresult rc = real_cuLaunchKernel(g_gb_absmax_dequant_fn,
                                      (unsigned int)n_rows, 1, 1,
                                      GB_KV_COMPRESS_ROW, 1, 1,
                                      0, NULL, args, NULL);
    return (rc == CUDA_SUCCESS) ? 0 : -1;
}

/* Overflow allocation - routes T1 misses to T2 (DDR) via pinned DDR paths.
 *
 * Path selection order:
 *   1. Path A (DMA-BUF pinned DDR): bare metal with greenboost.ko. Tries two sub-methods:
 *      a. Zero-copy: cudaImportExternalMemory(OpaqueFd) - no mmap/cuMemHostRegister overhead.
 *         Skipped on Blackwell (CC ≥ 12) and when libcudart.so is absent.
 *      b. Pinned:    mmap → GB_IOCTL_PIN_USER_PTR → cuMemHostRegister(DEVICEMAP).
 *         Fallback when sub-method a is unavailable or fails.
 *   2. Path B  (mmap + cuMemHostRegister, no greenboost.ko): for containers, VMs, WSL2,
 *      and HPC clusters where /dev/greenboost is absent. Auto-enabled when the device
 *      node is unavailable; skip with GREENBOOST_NO_HOSTREG=1.
 *   3. If all paths fail: return CUDA_ERROR_OUT_OF_MEMORY.
 *      Callers (llama.cpp) handle OOM cleanly. CPU compute is forbidden.
 *
 * KV cache vs weights placement is determined by the phase detector and by explicit
 * GB_ALLOC_KV_CACHE / GB_ALLOC_T1_PRIORITY flags set via Synapse CLI or env vars.
 * KV cache has T1 priority because it is read+written on every generated token; weights
 * tolerate T2 latency (sequential read, prefetch hides most PCIe overhead). */
static CUresult gb_overflow_alloc_ex(CUdeviceptr *dptr, size_t bytesize)
{
    void *mapped_ptr = NULL;
    int fd = -1;
    CUresult ret;

    /* Lazy cc re-probe: shim constructor fires before cuInit, so the initial
     * cuDeviceGetAttribute call at startup fails silently and gb_cc_major stays 0.
     * By the time the first CUDA alloc overflows T1 we are inside an active CUDA
     * context, so the probe will succeed. */
    if (gb_cc_major == 0 && real_cuDeviceGetAttribute) {
        int cc = 0;
        if (real_cuDeviceGetAttribute(&cc, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0) == CUDA_SUCCESS && cc > 0) {
            gb_cc_major = cc;
            fprintf(stderr, "[GreenBoost] Deferred cc probe: Compute %d.x\n", cc);
        }
    }

    /* Blackwell (cc >= 12): cuMemHostRegister gives fabric/DMA-only pointers that
     * compute SMs cannot access.  cuMemCreate(HOST)+cuMemMap+cuMemSetAccess also only
     * grants DMA access (not SM) on PCIe-attached Blackwell without ATS.
     * cuMemAllocManaged(CU_MEM_ATTACH_GLOBAL) IS SM-accessible on Blackwell PCIe -
     * pages stay in host RAM hinted PREFERRED_LOCATION=CPU; GPU SMs access them
     * over PCIe (~20 GB/s) with no migration (CUDA UVM §4.1.4.2). */
    if (gb_cc_major >= 12) {
        /* Bug found 2026-06-22 on a real GGUF load: every branch below this
         * point returns directly (zerocopy/managed success, or one of several
         * OOM paths), so gb_phase_classify() at the bottom of this function -
         * the only place that ever advances g_alloc_phase past GB_PHASE_INIT -
         * was never reached on Blackwell (cc >= 12). g_alloc_phase stayed 0
         * forever, which silently disabled every phase-gated feature on
         * this hardware class. Call it here
         * for its phase-tracking side effect; the returned alloc_flags
         * aren't used on the Blackwell zerocopy/managed path (T2 placement
         * there is unconditional, not phase-dependent), so discarding the
         * return value changes no allocation behavior. */
        (void)gb_phase_classify(bytesize);

        /* Explicit opt-out escape hatch.  The blanket auto-engage that was here before
         * is removed: managed UVM is SM-accessible on Blackwell PCIe (cc >= 12),
         * so there is no longer a reason to hard-refuse all T2 on Blackwell. */
        if (gb_disable_t2_on_blackwell) {
            GB_NVTX_EVENT("OOM_T2_BLACKWELL_DISABLED", "T2_DDR", bytesize >> 20, 0,
                          "blackwell_t2_disabled_explicit");
            gb_log("Blackwell T2 explicitly disabled (GREENBOOST_DISABLE_T2_ON_BLACKWELL=1) "
                   "- returning OOM for %zu MB alloc", bytesize >> 20);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }

        /* PR-F/H3: race-free T2 cap reservation.
         *
         * Atomically commit `bytesize` against the cap *before* attempting the
         * alloc.  Two concurrent callers racing here both see their reservation
         * succeed or fail consistently - neither blows past the cap.  The
         * inner alloc helpers (gb_vmm_t2_alloc_blackwell_*) no longer
         * increment gb_t2_overflow_bytes; the reservation already did that.
         * On alloc failure we revert via gb_t2_release_reserved. */
        if (gb_t2_try_reserve(bytesize) != 0) {
            size_t t2_used = atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed);
            size_t eff_cap = gb_effective_t2_cap();
            fprintf(stderr,
                    "[GreenBoost] Blackwell T2 cap reached: %.1f / %.1f GB - returning OOM\n",
                    (double)t2_used / (1024.0*1024.0*1024.0),
                    (double)eff_cap  / (1024.0*1024.0*1024.0));
            GB_NVTX_EVENT("OOM_T2_CAP", "T2_DDR", bytesize >> 20, 0, "blackwell_t2_cap_exceeded");
            return CUDA_ERROR_OUT_OF_MEMORY;
        }

        /* Host-RAM safety floor: managed UVM pages backed by host RAM are pinned on
         * first GPU touch.  Refuse new T2 allocs when MemAvailable drops below the
         * threshold so the kernel OOM-killer never fires.
         * Default 6 GB; override with GREENBOOST_HOST_RAM_SAFETY_MB.
         *
         * PR-I/H4: subtract gb_t2_pending_uvm_bytes from MemAvailable before
         * comparing.  Managed-UVM allocations don't immediately commit
         * physical pages - they materialise on first GPU SM touch.  Before
         * this fix, a burst of allocs would all pass the floor check while
         * pre-touch, then on first touch MemAvailable would drop below
         * floor in one step and the kernel OOM-killer would fire.  The
         * subtraction is conservative (still counts after first touch
         * until free) but bounded by the actual T2 cap, which has its
         * own atomic reservation (PR-F/H3). */
        {
            static long gb_host_ram_safety_mb = -1;
            if (__builtin_expect(gb_host_ram_safety_mb < 0, 0)) {
                const char *s = getenv("GREENBOOST_HOST_RAM_SAFETY_MB");
                gb_host_ram_safety_mb = s ? (long)gb_atoll(s) : 6144;
            }
            long avail_mb = 0;
            FILE *f = fopen("/proc/meminfo", "r");
            if (f) {
                char key[64]; long val;
                while (fscanf(f, "%63s %ld kB\n", key, &val) == 2)
                    if (strcmp(key, "MemAvailable:") == 0) { avail_mb = val / 1024; break; }
                fclose(f);
            }
            size_t pending = atomic_load_explicit(&gb_t2_pending_uvm_bytes,
                                                  memory_order_relaxed);
            long  pending_mb = (long)(pending >> 20);
            long  effective_avail_mb = avail_mb - pending_mb;
            if (avail_mb > 0 && effective_avail_mb < gb_host_ram_safety_mb) {
                fprintf(stderr,
                        "[GreenBoost] Blackwell T2: host RAM safety floor reached "
                        "(%ld MB available - %ld MB pending UVM = %ld MB effective "
                        "< %ld MB threshold) - returning OOM\n",
                        avail_mb, pending_mb, effective_avail_mb, gb_host_ram_safety_mb);
                /* PR-I/H4: must release the T2 reservation taken just above
                 * since we are refusing the alloc. */
                gb_t2_release_reserved(bytesize);
                GB_NVTX_EVENT("T2_OOM_HOST_FLOOR", "T2_DDR", bytesize >> 20, 0, "host_ram_floor");
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
        }

        /* Primary path: zerocopy pinned host memory.
         * mmap → cuMemHostRegister(DEVICEMAP|PORTABLE) → cuMemHostGetDevicePointer
         * returns a real CUDA device VA that GPU SMs access over PCIe.
         * SM-accessible on all CUDA cc >= 2.0 with canMapHostMemory=1.
         *
         * NOTE: cuMemAllocManaged(CU_MEM_ATTACH_GLOBAL) + SET_PREFERRED_LOCATION=CPU
         * was previously used as the primary path, but on Blackwell desktop PCIe with
         * CUDA 13.0 it returns CUDA_SUCCESS yet produces SM-inaccessible pointers -
         * any CUDA kernel touching them fails with CUDA_ERROR_INVALID_RESOURCE_HANDLE.
         * cuMemHostRegister(DEVICEMAP) is the correct SM-accessible path here. */
        {
            CUresult zret = gb_vmm_t2_alloc_blackwell_zerocopy(dptr, bytesize);
            if (zret == CUDA_SUCCESS) return zret;
            fprintf(stderr, "[GreenBoost] Blackwell ZC T2 failed (ret=%d) - trying managed UVM\n", zret);
        }

        /* Fallback: managed UVM.  On CUDA 13.0 + Blackwell desktop PCIe this
         * can give SM-inaccessible pointers, so zerocopy is preferred above.
         * Kept as a last resort for configurations where zerocopy is unavailable. */
        {
            CUresult mret = gb_vmm_t2_alloc_blackwell_managed(dptr, bytesize);
            if (mret == CUDA_SUCCESS) return mret;
            fprintf(stderr, "[GreenBoost] Blackwell managed T2 failed (ret=%d) - returning OOM\n", mret);
        }

        gb_t2_release_reserved(bytesize);
        GB_NVTX_EVENT("OOM_T2_BLACKWELL_SAFE", "T2_DDR", bytesize >> 20, 0,
                      "blackwell_all_paths_failed");
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    /* Choose alloc flags for this overflow allocation.
     *
     * Priority order (highest wins):
     *   1. gb_compute_domain_active  - V100 cluster ComputeDomain: all overflow = KV.
     *   2. g_kv_overflow_mode        - GREENBOOST_KV_OVERFLOW=1: all overflow = KV.
     *   3. gb_phase_classify()       - temporal phase detector: weights → KV → activations.
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

    size_t alloc_bytesize = bytesize;

    /* DESIGN RULE: GreenBoost always routes inference to GPU compute.
     * Path A/B  (pinned DDR, PCIe DMA) = non-Blackwell GPU-compute-ready path.
     * Path C    (managed UVM, host-preferred) = Blackwell PCIe path - handled
     *           above in the gb_cc_major >= 12 block via
     *           gb_vmm_t2_alloc_blackwell_managed(). */
    if (gb_physical_vram_bytes > 0 && alloc_bytesize > gb_physical_vram_bytes) {
        gb_log("oversize alloc %zu MB > physical VRAM %zu MB - routing to GPU-compute paths A/B (pinned DDR)",
               bytesize >> 20, gb_physical_vram_bytes >> 20);
    }

    /* Skip Path A and Path B for allocations larger than the effective T2 DDR pool cap.
     * Both paths pin the full allocation as anonymous RAM; attempting to pin more RAM
     * than the T2 pool allows would OOM the system.
     * effective_cap = 88% of pool (GB_T2_INFERENCE_CAP_PCT) in all phases. */
    if (gb_t2_pool_bytes > 0) {
        size_t effective_cap = gb_effective_t2_cap();
        if (alloc_bytesize > effective_cap) {
            gb_log("alloc %zu MB > T2 eff_cap %zu MB (pool=%zu MB) - T2 pool exhausted, returning OOM",
                   alloc_bytesize >> 20, effective_cap >> 20, gb_t2_pool_bytes >> 20);
            GB_NVTX_EVENT("OOM_T2_CAP", "T2_DDR", alloc_bytesize >> 20, 0, "t2_pool_cap_exceeded");
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
    }

    /* N11: SWA proactive eviction - drain oldest KV blocks before this T2 alloc
     * so that live T2 KV bytes stay within the configured sliding window.        */
    if (g_swa_window_bytes > 0 && (alloc_flags & GB_ALLOC_KV_CACHE)) {
        size_t _swa_live = atomic_load_explicit(&g_kv_t2_live_bytes, memory_order_relaxed);
        if (_swa_live + bytesize > g_swa_window_bytes) {
            size_t _swa_excess = (_swa_live + bytesize) - g_swa_window_bytes;
            GB_NVTX_PUSH("GB:N11_swa_evict", GB_NVTX_COLOR_EVICT);
            size_t _swa_evicted = gb_htable_evict_kv_arc(_swa_excess);
            GB_NVTX_POP();
            GB_NVTX_EVENT("SWA_EVICT", "T2_DDR", _swa_evicted >> 20, 0, "swa_proactive");
            gb_log("N11: SWA evicted %zu MB (live=%zu MB, window=%zu MB, needed=%zu MB)",
                   _swa_evicted >> 20, _swa_live >> 20,
                   g_swa_window_bytes >> 20, bytesize >> 20);
        }
    }

    /* ---- Path A (zero-copy sub-method): cudaImportExternalMemory -------- */
    /* Requires: libcudart.so resolved + greenboost.ko + /dev/greenboost.    */
    /* CUDA imports the kernel hugepage-backed DMA-BUF fd directly,          */
    /* driving its own IOMMU mapping from the kernel SG table. No mmap       */
    /* round-trip or cuMemHostRegister overhead.                              */
    /* Only used for allocations >= 4 MB (2 hugepages); smaller allocs       */
    /* fall through to the pinned sub-method (lower per-call overhead).       */
    /* Skipped on Blackwell (CC >= 12): gb_a0_disabled set at init.           */
    if (gb_use_dmabuf && !gb_a0_disabled &&
        real_cudaImportExternalMemory && real_cudaExternalMemoryGetMappedBuffer &&
        alloc_bytesize >= GB_PATH_A0_MIN_BYTES) {
        cudaExternalMemory_t ext_mem = NULL;
        GB_NVTX_PUSH("GB:T2_alloc_A_zerocopy", GB_NVTX_COLOR_T2);
        ret = gb_alloc_via_external_mem(dptr, alloc_bytesize, &ext_mem, alloc_flags);
        GB_NVTX_POP();
        if (ret == CUDA_SUCCESS) {
            /* fd=-1: req.fd was consumed by CUDA, not held by us */
            if (!ht_insert(*dptr, bytesize, 0 /* not UVM */, -1, NULL, -1, ext_mem)) {
                fprintf(stderr, "[GreenBoost] ERR: ht_insert failed (HT full) for Path A zero-copy %zu MB"
                        " - freeing allocation to avoid leak\n", bytesize >> 20);
                if (real_cudaDestroyExternalMemory)
                    real_cudaDestroyExternalMemory(ext_mem);
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
            ht_set_flags(*dptr, alloc_flags);
            if (alloc_flags & GB_ALLOC_KV_CACHE)
                atomic_fetch_add_explicit(&g_kv_t2_live_bytes, alloc_bytesize, memory_order_relaxed);
            __sync_fetch_and_add(&gb_path_a_count, 1);
            GB_NVTX_EVENT("ALLOC_A_ZEROCOPY", "T2_DDR", alloc_bytesize >> 20, *dptr, "path_a_zerocopy_ok");
            gb_maybe_write_stats();
            return CUDA_SUCCESS;
        }
        GB_NVTX_EVENT("ALLOC_A_ZEROCOPY_FAIL", "T2_DDR", alloc_bytesize >> 20, 0, "path_a_zerocopy_fail");
        gb_log("Path A (zero-copy) failed for %zu MB - falling through to Path A pinned sub-method",
               alloc_bytesize >> 20);
    }

    /* ---- Path A (pinned sub-method): DMA-BUF + cuMemHostRegister -------- */
    if (gb_use_dmabuf && real_cuMemHostRegister) {
        /* T2 capacity guard - same cap check as Path B (prevents over-pinning).
         * Uses gb_effective_t2_cap(): 88% of pool (all phases) to protect OS headroom. */
        if (gb_t2_pool_bytes > 0) {
            size_t effective_cap = gb_effective_t2_cap();
            size_t t2_used = atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed);
            if (t2_used >= effective_cap || alloc_bytesize > effective_cap - t2_used) {
                gb_log("Path A skip: T2 cap reached (%zu/%zu MB, eff_cap=%zu MB) for %zu MB - routing to Path B",
                       t2_used >> 20, gb_t2_pool_bytes >> 20, effective_cap >> 20, alloc_bytesize >> 20);
                GB_NVTX_EVENT("OOM_T2_CAP", "T2_DDR", alloc_bytesize >> 20, 0, "path_a_t2_cap_try_b");
                goto path_b_hostreg;
            }
        }
        /* MemAvailable guard - Path A pins real RAM just like Path B. */
        {
            size_t mem_avail = gb_get_mem_available();
            if (mem_avail > 0 &&
                (alloc_bytesize > mem_avail || mem_avail - alloc_bytesize < gb_safety_reserve_bytes)) {
                gb_log("Path A skip: MemAvailable %zu MB too low for %zu MB alloc - routing to Path B",
                       mem_avail >> 20, alloc_bytesize >> 20);
                GB_NVTX_EVENT("OOM_MEMAVAIL", "SYSTEM", alloc_bytesize >> 20, 0, "path_a_memavail_try_b");
                goto path_b_hostreg;
            }
        }
        /* ---- Path A (pool): sub-allocate from pre-registered slab ---------- */
        /* Lazy-init the pool on the first T2 allocation.  pthread_once ensures  */
        /* exactly-one init across concurrent threads.                           */
        if (!gb_t2_reg_pool.initialized && !gb_t2_reg_pool.init_failed
                && gb_pool_configured_bytes > 0) {
            pthread_once(&gb_pool_once, gb_pool_init_trampoline);
        }
        if (gb_t2_reg_pool.initialized) {
            void *host_ptr = NULL;
            GB_NVTX_PUSH("GB:T2_alloc_A_pool", GB_NVTX_COLOR_T2);
            ret = gb_pool_alloc(alloc_bytesize, dptr, &host_ptr);
            GB_NVTX_POP();
            if (ret == CUDA_SUCCESS) {
                if (!ht_insert(*dptr, bytesize, 0, -1, host_ptr, -1, NULL)) {
                    fprintf(stderr, "[GreenBoost] ERR: ht_insert failed (HT full) for Path A pool %zu MB"
                            " - freeing allocation to avoid leak\n", bytesize >> 20);
                    gb_pool_free(*dptr, alloc_bytesize);
                    return CUDA_ERROR_OUT_OF_MEMORY;
                }
                ht_set_flags(*dptr, alloc_flags);
                if (alloc_flags & GB_ALLOC_KV_CACHE)
                    atomic_fetch_add_explicit(&g_kv_t2_live_bytes, alloc_bytesize, memory_order_relaxed);
                atomic_fetch_add_explicit(&gb_t2_overflow_bytes, alloc_bytesize,
                                          memory_order_relaxed);
                gb_log("Path A pool: %zu MB at cuda_ptr=0x%llx host=%p t2=%zu MB",
                       bytesize >> 20, (unsigned long long)*dptr, host_ptr,
                       atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed) >> 20);
                __sync_fetch_and_add(&gb_path_a_count, 1);
                /* ALLOC_T2_POOL fired inside gb_pool_alloc; emit tier-level event here */
                GB_NVTX_EVENT("ALLOC_A_POOL", "T2_DDR", alloc_bytesize >> 20, *dptr, "path_a_pool_ok");
                gb_maybe_write_stats();
                return CUDA_SUCCESS;
            }
            gb_log("Path A pool full for %zu MB - falling back to per-allocation", alloc_bytesize >> 20);
        }

        /* ---- Path A (pinned per-alloc): mmap + kernel pin + cuMemHostRegister -- */
        /* Allocate anonymous memory using mmap, then ask greenboost.ko to pin it  */
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
                            " - falling through to Path B\n", ret, alloc_bytesize >> 20);
                    munmap(mapped_ptr, alloc_bytesize);
                    close(dmabuf_fd);
                    mapped_ptr = NULL;
                    /* Fall through to Path B - do NOT hard-return here. */
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
                    if (!ht_insert(*dptr, bytesize, 0 /* DMA-BUF */, -1, mapped_ptr, dmabuf_fd, NULL)) {
                        fprintf(stderr, "[GreenBoost] ERR: ht_insert failed (HT full) for Path A pinned %zu MB"
                                " - freeing allocation to avoid leak\n", bytesize >> 20);
                        if (real_cuMemHostUnregister) real_cuMemHostUnregister(mapped_ptr);
                        munmap(mapped_ptr, alloc_bytesize);
                        close(dmabuf_fd);
                        return CUDA_ERROR_OUT_OF_MEMORY;
                    }
                    ht_set_flags(*dptr, alloc_flags);
                    if (alloc_flags & GB_ALLOC_KV_CACHE)
                        atomic_fetch_add_explicit(&g_kv_t2_live_bytes, alloc_bytesize, memory_order_relaxed);
                    atomic_fetch_add_explicit(&gb_t2_overflow_bytes, alloc_bytesize, memory_order_relaxed);
                    gb_log("Path A (DMA-BUF pinned): %zu MB at cuda_ptr=0x%llx (mapped=%p, fd=%d) t2_total=%zu MB",
                           bytesize >> 20, (unsigned long long)*dptr, mapped_ptr, dmabuf_fd,
                           atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed) >> 20);
                    __sync_fetch_and_add(&gb_path_a_count, 1);
                    GB_NVTX_EVENT("ALLOC_A_PINNED", "T2_DDR", alloc_bytesize >> 20, *dptr, "path_a_pinned_ok");
                    gb_maybe_write_stats();
                    return CUDA_SUCCESS;
                }
            }
        }
    }

path_b_hostreg:
    /* ---- Path B: HostReg no-kernel (containers / VMs) ------------------- */
    /* mmap anonymous pages (2 MB huge preferred, 4 K fallback) and register  */
    /* directly with the CUDA driver.  No greenboost.ko required - works       */
    /* inside Docker, LXC, KVM guests, WSL2, and shared HPC clusters.          */
    /* Concept by Jerry Nguyen (MR !3); hugepage preference + hash-table        */
    /* integration by Ferran Duarri.                                            */
    /*
     * RAM safety guard - mirrors gb_pin_user_buf() / gb_alloc_buf() in the kernel.
     * Path B never calls any IOCTL so the kernel's safety_reserve_gb check is blind
     * to these allocations.  Two checks must both pass:
     *   1. Cumulative T2 cap: prevent Path A+B together from exceeding virtual_vram_gb.
     *   2. MemAvailable guard: keep at least safety_reserve_gb free for the OS.
     * On failure, return CUDA_ERROR_OUT_OF_MEMORY - no UVM fallback (GPU compute required).
     */
    if (real_cuMemHostRegister && real_cuMemHostGetDevicePointer && !gb_no_hostreg) {
        /* Check 1: cumulative T2 cap.
         * Uses gb_effective_t2_cap(): 88% of pool (all phases) to protect OS headroom. */
        if (gb_t2_pool_bytes > 0) {
            size_t effective_cap = gb_effective_t2_cap();
            size_t t2_used = atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed);
            if (t2_used >= effective_cap || alloc_bytesize > effective_cap - t2_used) {
                gb_log("Path B skip: T2 cap reached (%zu MB used / %zu MB eff_cap) for %zu MB alloc - returning OOM (no UVM fallback: GPU compute required)",
                       t2_used >> 20, effective_cap >> 20, alloc_bytesize >> 20);
                GB_NVTX_EVENT("OOM_T2_CAP", "T2_DDR", alloc_bytesize >> 20, 0, "path_b_t2_cap_oom");
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
        }
        /* Check 2: MemAvailable guard */
        {
            size_t mem_avail = gb_get_mem_available();
            if (mem_avail > 0 &&
                (alloc_bytesize > mem_avail || mem_avail - alloc_bytesize < gb_safety_reserve_bytes)) {
                gb_log("Path B skip: MemAvailable %zu MB < reserve %zu MB + req %zu MB - returning OOM (no UVM fallback: GPU compute required)",
                       mem_avail >> 20, gb_safety_reserve_bytes >> 20, alloc_bytesize >> 20);
                GB_NVTX_EVENT("OOM_MEMAVAIL", "SYSTEM", alloc_bytesize >> 20, 0, "path_b_memavail_oom");
                return CUDA_ERROR_OUT_OF_MEMORY;
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
            GB_NVTX_PUSH("GB:T2_alloc_B_hostreg", GB_NVTX_COLOR_T2);
            ret = real_cuMemHostRegister(hreg_ptr, alloc_bytesize, CU_MEMHOSTREGISTER_DEVICEMAP);
            if (ret == CUDA_SUCCESS) {
                ret = real_cuMemHostGetDevicePointer(dptr, hreg_ptr, 0);
                if (ret == CUDA_SUCCESS) {
                    GB_NVTX_POP();
                    /* fd=-1: no DMA-BUF fd to close on free */
                    /* ht_insert size = bytesize (uncompressed logical size) */
                    if (!ht_insert(*dptr, bytesize, 0 /* not UVM */, -1, hreg_ptr, -1, NULL)) {
                        fprintf(stderr, "[GreenBoost] ERR: ht_insert failed (HT full) for Path B hostreg %zu MB"
                                " - freeing allocation to avoid leak\n", bytesize >> 20);
                        if (real_cuMemHostUnregister) real_cuMemHostUnregister(hreg_ptr);
                        munmap(hreg_ptr, alloc_bytesize);
                        return CUDA_ERROR_OUT_OF_MEMORY;
                    }
                    ht_set_flags(*dptr, alloc_flags);
                    if (alloc_flags & GB_ALLOC_KV_CACHE)
                        atomic_fetch_add_explicit(&g_kv_t2_live_bytes, alloc_bytesize, memory_order_relaxed);
                    atomic_fetch_add_explicit(&gb_t2_overflow_bytes, alloc_bytesize, memory_order_relaxed);
                    gb_log("Path B (HostReg/no-kmod): %zu MB at cuda_ptr=0x%llx (mapped=%p) t2_total=%zu MB",
                           bytesize >> 20, (unsigned long long)*dptr, hreg_ptr,
                           atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed) >> 20);
                    __sync_fetch_and_add(&gb_path_b_count, 1);
                    GB_NVTX_EVENT("ALLOC_B_HOSTREG", "T2_DDR", alloc_bytesize >> 20, *dptr, "path_b_hostreg_ok");
                    gb_maybe_write_stats();
                    return CUDA_SUCCESS;
                }
                if (real_cuMemHostUnregister) real_cuMemHostUnregister(hreg_ptr);
            }
            GB_NVTX_POP();
            munmap(hreg_ptr, alloc_bytesize);
        }
        gb_log("Path B (HostReg) failed for %zu MB - returning OOM (no UVM fallback: GPU compute required)", alloc_bytesize >> 20);
        GB_NVTX_EVENT("OOM_PATH_B_FAIL", "T2_DDR", alloc_bytesize >> 20, 0, "path_b_hostreg_failed");
    }

    /* Path C (UVM) removed - CPU tensor compute forbidden.
     * GreenBoost routes all inference to GPU compute via pinned DDR (Path A/B).
     * Callers receive CUDA_ERROR_OUT_OF_MEMORY when T2 is exhausted. */
    return CUDA_ERROR_OUT_OF_MEMORY;
}

/* ------------------------------------------------------------------ */
/*  gb_overflow_alloc wrapper                                           */
/* ------------------------------------------------------------------ */
/* Host RAM floor — local T2 must NEVER drive the machine into memory
 * pressure: with e.g. 64 GB total, GreenBoost may consume DDR only while
 * MemAvailable stays above the floor (default 9 GB ≈ "cap total use at
 * ~55 GB", counting every other process too since MemAvailable already
 * reflects them).  When an allocation would cross the floor, local T2 is
 * refused and smart_alloc cascades to feeder T2/T3 — protecting the host
 * AND putting the feeder's DDR to work.  Override: GREENBOOST_HOST_RAM_FLOOR_MB. */
static size_t gb_host_ram_floor_bytes(void)
{
    static size_t floor_b = (size_t)-1;
    if (floor_b == (size_t)-1) {
        long mb = 9216;
        const char *e = getenv("GREENBOOST_HOST_RAM_FLOOR_MB");
        if (e && atol(e) >= 0) mb = atol(e);
        floor_b = (size_t)mb << 20;
    }
    return floor_b;
}

/* GREENBOOST_FEEDER_EXCLUSIVE=1 — see gb_needs_overflow.  Cached env. */
static int gb_feeder_exclusive(void)
{
    static int _fx = -1;
    if (_fx < 0) {
        const char *e = getenv("GREENBOOST_FEEDER_EXCLUSIVE");
        _fx = (e && e[0] == '1') ? 1 : 0;
    }
    return _fx;
}

static CUresult gb_overflow_alloc(CUdeviceptr *dptr, size_t bytesize)
{
    size_t floor_b = gb_host_ram_floor_bytes();
    if (floor_b > 0) {
        size_t avail = gb_get_mem_available();
        if (avail > 0 && (avail < floor_b || avail - floor_b < bytesize)) {
            gb_log("local T2 refused: MemAvailable=%zu MB - %zu MB req would cross "
                   "host RAM floor (%zu MB) - cascading to feeder tiers",
                   avail >> 20, bytesize >> 20, floor_b >> 20);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
    }
    return gb_overflow_alloc_ex(dptr, bytesize);
}

/* ------------------------------------------------------------------ */
/*  Network Feeder Allocation Helper                                   */
/* ------------------------------------------------------------------ */
static uint32_t get_local_ddr_speed(void) {
    static uint32_t cached_speed = 0;
    if (cached_speed > 0) return cached_speed;

    /* Source 1: GreenBoost's autodetected profile file.  greenboost_setup.sh
     * captures `ram_speed_mt` (in MT/s) at install/profile time when it has
     * root, so the value is reliably available to every user-space process
     * via a world-readable file - no dmidecode (which needs CAP_SYS_RAWIO)
     * and no popen() per process. */
    {
        static const char * const profile_paths[] = {
            "/etc/greenboost/active_profile.md",
            "/etc/greenboost/profiles/default.md",
            NULL
        };
        for (int i = 0; profile_paths[i] && cached_speed == 0; i++) {
            FILE *pf = fopen(profile_paths[i], "r");
            if (!pf) continue;
            char line[256];
            while (fgets(line, sizeof(line), pf)) {
                /* Match `ram_speed_mt: <integer>` (YAML-style frontmatter).
                 * Tolerates whitespace and quotes around the value. */
                const char *p = strstr(line, "ram_speed_mt");
                if (!p) continue;
                p = strchr(p, ':');
                if (!p) continue;
                p++;
                while (*p == ' ' || *p == '\t' || *p == '"' || *p == '\'') p++;
                int val = atoi(p);
                if (val > 0) { cached_speed = (uint32_t)val; break; }
            }
            fclose(pf);
        }
    }

    /* Source 2: dmidecode (only works as root - kept as a defensive fallback
     * for processes started by root, e.g. greenboost daemons). */
    if (cached_speed == 0) {
        FILE *fp = popen("dmidecode -t memory 2>/dev/null | "
                         "grep -E 'Configured Memory Speed:|Speed:' | "
                         "grep -i MT/s | head -1 | grep -oE '[0-9]+'", "r");
        if (fp) {
            char buf[32] = {0};
            if (fgets(buf, sizeof(buf), fp))
                cached_speed = (uint32_t)atoi(buf);
            pclose(fp);
        }
    }

    /* Source 3: conservative default biased toward local T2 routing. */
    if (cached_speed == 0) {
        cached_speed = 2400;
        GB_INIT_LOG_ONCE(
                "[GreenBoost] DDR speed lookup failed (profile + dmidecode both "
                "unavailable) - defaulting to 2400 MT/s.  Run `sudo greenboost "
                "profile create` to populate /etc/greenboost/profiles/default.md.\n");
    }
    return cached_speed;
}

static uint32_t get_local_nvme_speed_mbs(void) {
    static uint32_t cached = 0;
    if (cached) return cached;
    FILE *fp = fopen("/sys/block/nvme0n1/queue/max_hw_sectors_kb", "r");
    if (fp) { fclose(fp); cached = 3500; return cached; }
    fp = fopen("/sys/block/nvme0/queue/max_hw_sectors_kb", "r");
    if (fp) { fclose(fp); cached = 3500; return cached; }
    fp = fopen("/sys/block/sda/queue/rotational", "r");
    if (fp) {
        int rot = 1;
        if (fscanf(fp, "%d", &rot) == 1 && rot == 0) cached = 550;
        fclose(fp);
    }
    if (cached == 0) cached = 500;
    return cached;
}

static int gb_try_feeder_alloc(CUdeviceptr *dptr, size_t bytesize, int t2_fallback)
{
    if (!gb_netc_is_active()) return -1;
    int _n = gb_netc_remote_gpu_count();
    for (int _ri = 0; _ri < _n; _ri++) {
        uint64_t _free = 0, _total = 0;
        if (t2_fallback) {
            if (gb_netc_mem_info(_ri, &_free, &_total) == 0 && _free >= (uint64_t)bytesize) {
                uint64_t _fake = 0;
                if (gb_netc_malloc(_ri, (uint64_t)bytesize, 0, &_fake) == 0) {
                    *dptr = (CUdeviceptr)_fake;
                    gb_log("feeder T2[%d] alloc: %zu MB → fake=0x%llx", _ri, bytesize >> 20, (unsigned long long)_fake);
                    GB_NVTX_EVENT("ALLOC_T2_FEEDER", "T2_FEEDER", bytesize >> 20, _fake, "feeder_t2_ok");
                    return 0;
                }
            }
        } else {
            if (gb_netc_t1_mem_info(_ri, &_free, &_total) == 0 && _free >= (uint64_t)bytesize) {
                uint64_t _fake = 0;
                if (gb_netc_malloc(_ri, (uint64_t)bytesize, 0, &_fake) == 0) {
                    *dptr = (CUdeviceptr)_fake;
                    gb_log("feeder T1[%d] alloc: %zu MB → fake=0x%llx", _ri, bytesize >> 20, (unsigned long long)_fake);
                    GB_NVTX_EVENT("ALLOC_T1_FEEDER", "T1_FEEDER", bytesize >> 20, _fake, "feeder_t1_ok");
                    return 0;
                }
            }
        }
    }
    return -1;
}

/* Tier-aware feeder allocation: queries the correct tier's free space and
 * requests exactly that tier (GB_ALLOC_TIER_T1/T2/T3). */
static int gb_try_feeder_alloc_tier(CUdeviceptr *dptr, size_t bytesize, uint8_t tier)
{
    if (!gb_netc_is_active()) return -1;
    int _n = gb_netc_remote_gpu_count();
    for (int _ri = 0; _ri < _n; _ri++) {
        /* Heal a dropped connection inline — an alloc burst must not read
         * "feeder not connected" for the 0.5-2 s the heartbeat thread needs. */
        gb_netc_ensure_connected(_ri);
        /* U4: skip feeders in DISABLED state entirely */
        if (gb_netc_feeder_disabled(_ri)) continue;
        uint64_t _free = 0, _total = 0;
        int ok = 0;
        if (tier == GB_ALLOC_TIER_T1) {
            int mi_ret = gb_netc_t1_mem_info(_ri, &_free, &_total);
            ok = (mi_ret == 0 && _free >= (uint64_t)bytesize);
            if (!ok)
                fprintf(stderr, "[GreenBoost] feeder_alloc_tier T1[%d]: mem_info=%d"
                        " free=%llu MB total=%llu MB req=%llu MB → skip\n",
                        _ri, mi_ret, (unsigned long long)(_free >> 20),
                        (unsigned long long)(_total >> 20),
                        (unsigned long long)(bytesize >> 20));
        } else if (tier == GB_ALLOC_TIER_T2) {
            int mi_ret = gb_netc_t2_mem_info(_ri, &_free, &_total);
            ok = (mi_ret == 0 && _free >= (uint64_t)bytesize);
            if (!ok)
                fprintf(stderr, "[GreenBoost] feeder_alloc_tier T2[%d]: mem_info=%d"
                        " free=%llu MB total=%llu MB req=%llu MB → skip\n",
                        _ri, mi_ret, (unsigned long long)(_free >> 20),
                        (unsigned long long)(_total >> 20),
                        (unsigned long long)(bytesize >> 20));
        } else if (tier == GB_ALLOC_TIER_T3)
            ok = (gb_netc_t3_mem_info(_ri, &_free, &_total) == 0 && _total > 0);
        if (ok) {
            uint64_t _fake = 0;
            if (gb_netc_malloc_tier(_ri, (uint64_t)bytesize, tier, &_fake) == 0) {
                *dptr = (CUdeviceptr)_fake;
                fprintf(stderr, "[GreenBoost] cudaMalloc feeder T%u[%d]: %llu MB → fake=0x%llx\n",
                        tier, _ri, (unsigned long long)(bytesize >> 20),
                        (unsigned long long)_fake);
                return 0;
            }
            fprintf(stderr, "[GreenBoost] feeder_alloc_tier T%u[%d]: malloc_tier failed"
                    " (free=%llu MB, req=%llu MB)\n",
                    tier, _ri, (unsigned long long)(_free >> 20),
                    (unsigned long long)(bytesize >> 20));
        }
    }
    return -1;
}

/* ================================================================== */
/*  RULE #1 — front-load VRAM split  (GB_VRAM_FRONTLOAD, default OFF)   */
/*                                                                      */
/*  Problem: a large model's weights arrive as ONE big cudaMalloc.      */
/*  gb_needs_overflow() is all-or-nothing, so gb_smart_overflow_alloc   */
/*  places the ENTIRE buffer in a single tier (usually local T2 DDR)    */
/*  while physical VRAM sits partly free.  This violates the immutable  */
/*  "fill GPU VRAM to ~90%" rule.                                       */
/*                                                                      */
/*  Fix: for a large overflow buffer, reserve ONE contiguous VA of the  */
/*  full size, back the FIRST portion with physical-VRAM DEVICE handles */
/*  (filling up to a ~pct% target) and the REMAINDER with host-pinned   */
/*  VMM handles — the exact CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT       */
/*  PINNED path already used by the cuMemCreate() intercept.  Both       */
/*  portions live in one VA range mapped with one cuMemSetAccess, so any */
/*  kernel / memcpy sees a single coherent buffer.                       */
/*                                                                      */
/*  Guarded OFF by default: when GB_VRAM_FRONTLOAD is unset/"0" this     */
/*  path is never entered and behaviour is byte-for-byte the previous    */
/*  allocator.  Any error inside the split path unwinds fully and        */
/*  returns non-success, so gb_smart_overflow_alloc falls through to the */
/*  existing tiers unchanged — an alloc that used to succeed still does. */
/* ================================================================== */

static int gb_frontload_enabled(void)
{
    static int v = -1;
    if (__builtin_expect(v < 0, 0)) {
        const char *e = getenv("GB_VRAM_FRONTLOAD");
        v = (e && e[0] == '1') ? 1 : 0;
    }
    return v;
}

/* Fill target as a percentage of TOTAL VRAM (default 90, clamped 50..99). */
static unsigned gb_frontload_pct(void)
{
    static unsigned p = 0;
    if (__builtin_expect(p == 0, 0)) {
        const char *e = getenv("GB_VRAM_FRONTLOAD_PCT");
        long v = e ? (long)gb_atoll(e) : 0;
        if (v < 50 || v > 99) v = 90;
        p = (unsigned)v;
    }
    return p;
}

/* Minimum overflow-buffer size that triggers a split (default 512 MB). */
static size_t gb_frontload_min_bytes(void)
{
    static size_t b = 0;
    if (__builtin_expect(b == 0, 0)) {
        const char *e = getenv("GB_VRAM_FRONTLOAD_MIN_MB");
        long mb = e ? (long)gb_atoll(e) : 512;
        if (mb < 1) mb = 512;
        b = (size_t)mb << 20;
    }
    return b;
}

/* Sub-range physical backing for a front-load split allocation. */
typedef struct {
    CUmemGenericAllocationHandle handle; /* physical handle (device or host) */
    CUdeviceptr                  addr;   /* va + offset (start of this chunk) */
    size_t                       size;   /* mapped bytes (granularity-aligned) */
} gb_fl_chunk_t;

typedef struct {
    CUdeviceptr    va;           /* 0 = empty, UINT64_MAX = tombstone */
    size_t         va_size;      /* reserved VA length */
    size_t         device_bytes; /* VRAM-backed bytes  → GB_TIER_T1_LOCAL */
    size_t         host_bytes;   /* host-backed bytes  → GB_TIER_T2_LOCAL + T2 cap */
    gb_fl_chunk_t *chunks;       /* malloc'd array, n_chunks entries */
    uint32_t       n_chunks;
} gb_fl_entry_t;

#define GB_FL_HT_BITS   8u
#define GB_FL_HT_SIZE   (1u << GB_FL_HT_BITS)
#define GB_FL_HT_MASK   (GB_FL_HT_SIZE - 1u)
#define GB_FL_TOMBSTONE ((CUdeviceptr)UINT64_MAX)
/* Chunk unit: physical handles are created in pieces of this size (aligned up
 * to the allocation granularity) so a slightly-too-high 90% estimate stops the
 * device portion at a chunk boundary and rolls the rest into host memory. */
#define GB_FL_CHUNK_UNIT (1ULL << 30)  /* 1 GiB */

static gb_fl_entry_t   gb_fl_ht[GB_FL_HT_SIZE];
static pthread_mutex_t gb_fl_ht_lock = PTHREAD_MUTEX_INITIALIZER;

static inline uint32_t gb_fl_hash(CUdeviceptr va)
{
    return (uint32_t)(((uint64_t)va * 0x9E3779B97F4A7C15ULL) >> (64 - GB_FL_HT_BITS));
}

/* Returns 1 on success, 0 if the (tiny) table is full. */
static int gb_fl_ht_insert(CUdeviceptr va, size_t va_size, size_t device_bytes,
                           size_t host_bytes, gb_fl_chunk_t *chunks, uint32_t n_chunks)
{
    int ok = 0;
    pthread_mutex_lock(&gb_fl_ht_lock);
    uint32_t slot = gb_fl_hash(va) & GB_FL_HT_MASK;
    for (uint32_t i = 0; i < GB_FL_HT_SIZE; i++) {
        gb_fl_entry_t *e = &gb_fl_ht[(slot + i) & GB_FL_HT_MASK];
        if (e->va == 0 || e->va == GB_FL_TOMBSTONE) {
            e->va = va; e->va_size = va_size;
            e->device_bytes = device_bytes; e->host_bytes = host_bytes;
            e->chunks = chunks; e->n_chunks = n_chunks;
            ok = 1; break;
        }
    }
    pthread_mutex_unlock(&gb_fl_ht_lock);
    return ok;
}

/* Returns 1 and copies the entry to *out (caller frees out->chunks) if found. */
static int gb_fl_ht_remove(CUdeviceptr va, gb_fl_entry_t *out)
{
    if (!va) return 0;
    int found = 0;
    pthread_mutex_lock(&gb_fl_ht_lock);
    uint32_t slot = gb_fl_hash(va) & GB_FL_HT_MASK;
    for (uint32_t i = 0; i < GB_FL_HT_SIZE; i++) {
        gb_fl_entry_t *e = &gb_fl_ht[(slot + i) & GB_FL_HT_MASK];
        if (e->va == 0) break;               /* end of probe chain */
        if (e->va == va) {
            *out = *e;
            e->va = GB_FL_TOMBSTONE; e->chunks = NULL; e->n_chunks = 0;
            e->va_size = e->device_bytes = e->host_bytes = 0;
            found = 1; break;
        }
    }
    pthread_mutex_unlock(&gb_fl_ht_lock);
    return found;
}

/* Front-load split: fill VRAM to ~pct% with DEVICE handles, remainder with
 * host-pinned VMM handles, all mapped into ONE reserved VA.  On ANY failure
 * unwinds every side effect and returns non-success so the caller falls back
 * to the existing whole-buffer overflow path.  Never records tier accounting
 * unless it returns CUDA_SUCCESS. */
static CUresult gb_frontload_split_alloc(CUdeviceptr *dptr, size_t bytesize)
{
    if (!real_cuMemAddressReserve || !real_cuMemCreate || !real_cuMemMap ||
        !real_cuMemSetAccess || !real_cuMemAddressFree || !real_cuMemRelease ||
        !real_cuMemUnmap || !real_cuMemGetAllocationGranularity)
        return CUDA_ERROR_NOT_SUPPORTED;

    /* Resolve host location type once (shared cache with the hostnuma /
     * cuMemCreate paths — prefers HOST_NUMA_CURRENT, HOST on single-socket). */
    if (__builtin_expect(gb_vmm_host_loc_type == 0, 0))
        gb_vmm_host_loc_type = CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT;

    CUmemAllocationProp dprop; memset(&dprop, 0, sizeof(dprop));
    dprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    dprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    dprop.location.id   = 0;
    CUmemAllocationProp hprop; memset(&hprop, 0, sizeof(hprop));
    hprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    hprop.location.type = gb_vmm_host_loc_type;
    hprop.location.id   = 0;

    /* Common granularity = max(device, host) so every chunk boundary is legal
     * for both handle types mapped into the same VA. */
    size_t dgran = 0, hgran = 0;
    if (real_cuMemGetAllocationGranularity(&dgran, &dprop, 0 /*MINIMUM*/) != CUDA_SUCCESS || dgran == 0)
        dgran = 2ULL << 20;
    if (real_cuMemGetAllocationGranularity(&hgran, &hprop, 0 /*MINIMUM*/) != CUDA_SUCCESS || hgran == 0)
        hgran = 2ULL << 20;
    size_t gran = dgran > hgran ? dgran : hgran;

    size_t va_size = (bytesize + gran - 1) & ~(gran - 1);

    /* Target VRAM fill = free VRAM minus the (100-pct)% fill headroom and the
     * same reserves gb_needs_overflow() honours.  Being conservative here is
     * safe: the device-chunk loop stops early on real OOM if the estimate is
     * slightly high, rolling the rest into host memory. */
    size_t free_vram  = atomic_load_explicit(&g_cached_free_vram,  memory_order_relaxed);
    size_t total_vram = atomic_load_explicit(&g_cached_total_vram, memory_order_relaxed);
    if (free_vram == 0 || total_vram == 0)
        return CUDA_ERROR_NOT_SUPPORTED;

    size_t fill_headroom = total_vram * (100 - gb_frontload_pct()) / 100;
    size_t kv_reserved   = atomic_load_explicit(&g_kv_reserve_bytes,      memory_order_relaxed);
    size_t kv_in_t1      = atomic_load_explicit(&g_kv_allocated_t1_bytes, memory_order_relaxed);
    size_t kv_reserve    = (kv_in_t1 >= kv_reserved) ? 0 : (kv_reserved - kv_in_t1);
    size_t ws_reserve    = 0;
    if (atomic_load_explicit(&g_alloc_phase, memory_order_relaxed) <= GB_PHASE_MODEL_LOAD)
        ws_reserve = atomic_load_explicit(&g_workspace_reserve_bytes, memory_order_relaxed);

    size_t reserved_total = fill_headroom + vram_headroom_bytes + kv_reserve + ws_reserve;
    if (free_vram <= reserved_total)
        return CUDA_ERROR_NOT_SUPPORTED;              /* no VRAM to front-load */
    size_t target_free = free_vram - reserved_total;

    size_t device_portion = (bytesize < target_free) ? bytesize : target_free;
    device_portion &= ~(gran - 1);                    /* round DOWN to gran */
    if (device_portion == 0)
        return CUDA_ERROR_NOT_SUPPORTED;

    /* Reserve one VA for the whole logical buffer. */
    CUdeviceptr va = 0;
    CUresult ret = real_cuMemAddressReserve(&va, va_size, 0, 0, 0);
    if (ret != CUDA_SUCCESS) {
        gb_log("frontload: cuMemAddressReserve %zu MB FAILED ret=%d — fallback",
               va_size >> 20, ret);
        return ret;
    }

    size_t chunk_unit = GB_FL_CHUNK_UNIT;
    if (chunk_unit < gran) chunk_unit = gran;
    chunk_unit &= ~(gran - 1);
    if (chunk_unit == 0) chunk_unit = gran;

    uint32_t max_chunks = (uint32_t)(va_size / chunk_unit) + 2;
    gb_fl_chunk_t *chunks = (gb_fl_chunk_t *)calloc(max_chunks, sizeof(gb_fl_chunk_t));
    if (!chunks) {
        real_cuMemAddressFree(va, va_size);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    uint32_t nc = 0;
    size_t off = 0, dev_mapped = 0;
    int t2_reserved = 0;
    size_t host_portion = 0;

    /* --- DEVICE portion: fill physical VRAM up to the target --- */
    while (off < device_portion) {
        size_t csz = device_portion - off;
        if (csz > chunk_unit) csz = chunk_unit;      /* csz stays gran-aligned */
        CUmemGenericAllocationHandle h;
        CUresult cret = real_cuMemCreate(&h, csz, &dprop, 0);
        if (cret != CUDA_SUCCESS) {
            /* Real device OOM: stop the device portion early; the rest goes to
             * host.  Never fail the whole alloc because 90% was slightly high. */
            gb_log("frontload: device cuMemCreate stopped at %zu/%zu MB (ret=%d) — rest to host",
                   dev_mapped >> 20, device_portion >> 20, cret);
            break;
        }
        cret = real_cuMemMap(va + off, csz, 0, h, 0);
        if (cret != CUDA_SUCCESS) {
            real_cuMemRelease(h);
            gb_log("frontload: device cuMemMap off=%zu MB FAILED ret=%d — stopping device portion",
                   off >> 20, cret);
            break;
        }
        chunks[nc].handle = h; chunks[nc].addr = va + off; chunks[nc].size = csz; nc++;
        off += csz; dev_mapped += csz;
    }
    device_portion = dev_mapped;                     /* actual VRAM mapped */
    host_portion   = va_size - device_portion;       /* remainder incl. tail */

    /* --- HOST portion: pinned VMM handles for the remainder --- */
    if (host_portion > 0) {
        if (gb_t2_try_reserve(host_portion) != 0) {
            gb_log("frontload: host portion %zu MB would exceed T2 cap — fallback",
                   host_portion >> 20);
            goto fl_unwind;                          /* t2 not reserved */
        }
        t2_reserved = 1;
        while (off < va_size) {
            size_t csz = va_size - off;
            if (csz > chunk_unit) csz = chunk_unit;
            CUmemGenericAllocationHandle h;
            CUresult cret = real_cuMemCreate(&h, csz, &hprop, 0);
            if (cret == CUDA_ERROR_INVALID_VALUE &&
                hprop.location.type == CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT) {
                /* Single-socket UMA: HOST_NUMA_CURRENT unsupported → HOST. */
                gb_vmm_host_loc_type = CU_MEM_LOCATION_TYPE_HOST;
                hprop.location.type  = CU_MEM_LOCATION_TYPE_HOST;
                cret = real_cuMemCreate(&h, csz, &hprop, 0);
            }
            if (cret != CUDA_SUCCESS) {
                gb_log("frontload: host cuMemCreate off=%zu MB FAILED ret=%d — unwinding",
                       off >> 20, cret);
                goto fl_unwind;
            }
            cret = real_cuMemMap(va + off, csz, 0, h, 0);
            if (cret != CUDA_SUCCESS) {
                real_cuMemRelease(h);
                gb_log("frontload: host cuMemMap off=%zu MB FAILED ret=%d — unwinding",
                       off >> 20, cret);
                goto fl_unwind;
            }
            chunks[nc].handle = h; chunks[nc].addr = va + off; chunks[nc].size = csz; nc++;
            off += csz;
        }
    }

    /* --- one RW device mapping over the whole VA --- */
    {
        CUmemAccessDesc desc; memset(&desc, 0, sizeof(desc));
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        desc.location.id   = 0;
        desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        ret = real_cuMemSetAccess(va, va_size, &desc, 1);
        if (ret != CUDA_SUCCESS) {
            gb_log("frontload: cuMemSetAccess FAILED ret=%d — unwinding", ret);
            goto fl_unwind;
        }
    }

    if (!gb_fl_ht_insert(va, va_size, device_portion, host_portion, chunks, nc)) {
        gb_log("frontload: tracking table full — unwinding");
        goto fl_unwind;
    }

    /* Success: record tiers so tier_t1_local_cur_mb reflects the front-loaded
     * VRAM (this is what makes RULE #1 visible in dataflux). */
    gb_tier_record_alloc(GB_TIER_T1_LOCAL, device_portion);
    if (host_portion > 0)
        gb_tier_record_alloc(GB_TIER_T2_LOCAL, host_portion);
    *dptr = va;
    gb_log("frontload split: req=%zu MB → T1(VRAM)=%zu MB + T2(host)=%zu MB "
           "va=0x%llx chunks=%u pct=%u",
           bytesize >> 20, device_portion >> 20, host_portion >> 20,
           (unsigned long long)va, nc, gb_frontload_pct());
    GB_NVTX_EVENT("ALLOC_FRONTLOAD", "T1_GPU", device_portion >> 20, va,
                  "vram_frontload_split");
    gb_maybe_write_stats();
    return CUDA_SUCCESS;

fl_unwind:
    if (t2_reserved) gb_t2_release_reserved(host_portion);
    for (uint32_t i = 0; i < nc; i++) {
        real_cuMemUnmap(chunks[i].addr, chunks[i].size);
        real_cuMemRelease(chunks[i].handle);
    }
    real_cuMemAddressFree(va, va_size);
    free(chunks);
    return CUDA_ERROR_OUT_OF_MEMORY;
}

/* Free a front-load split allocation: unmap + release every chunk, free the
 * VA, release the T2 reservation, and update tier accounting.  Returns 1 if
 * dptr was a front-load VA (caller returns success), 0 otherwise. */
static int gb_frontload_free_dispatch(CUdeviceptr dptr)
{
    gb_fl_entry_t e;
    if (!gb_fl_ht_remove(dptr, &e))
        return 0;
    GB_NVTX_EVENT("FREE_FRONTLOAD", "T1_GPU", e.device_bytes >> 20, dptr,
                  "vram_frontload_free");
    for (uint32_t i = 0; i < e.n_chunks; i++) {
        if (real_cuMemUnmap)   real_cuMemUnmap(e.chunks[i].addr, e.chunks[i].size);
        if (real_cuMemRelease) real_cuMemRelease(e.chunks[i].handle);
    }
    if (real_cuMemAddressFree) real_cuMemAddressFree(e.va, e.va_size);
    free(e.chunks);
    if (e.host_bytes > 0)
        atomic_fetch_sub_explicit(&gb_t2_overflow_bytes, e.host_bytes, memory_order_relaxed);
    gb_tier_record_free(GB_TIER_T1_LOCAL, e.device_bytes);
    if (e.host_bytes > 0)
        gb_tier_record_free(GB_TIER_T2_LOCAL, e.host_bytes);
    gb_log("frontload free: va=0x%llx T1=%zu MB T2=%zu MB",
           (unsigned long long)dptr, e.device_bytes >> 20, e.host_bytes >> 20);
    gb_maybe_write_stats();
    return 1;
}

static CUresult gb_smart_overflow_alloc(CUdeviceptr *dptr, size_t bytesize) {
    int cur_phase = atomic_load_explicit(&g_alloc_phase, memory_order_relaxed);

    /* Feeder-exclusive: block for the feeder on the FIRST alloc so weights
     * don't lose netc's async-connect race and fall to local T2 (→ omen idle,
     * every kernel local). One-time up-front wait; after that connect is warm. */
    if (gb_feeder_exclusive() && gb_netc_is_active()) {
        static _Atomic int waited = 0;
        int expected = 0;
        if (atomic_compare_exchange_strong(&waited, &expected, 1)) {
            if (gb_netc_wait_connected(8000) == 0)
                gb_log("feeder-exclusive: feeder connected, routing all buffers remote");
            else
                gb_log("feeder-exclusive: WARN feeder not connected after 8s - "
                       "allocs may fall local");
        }
    }

    /* RULE #1 front-load VRAM split (GB_VRAM_FRONTLOAD, default OFF): before
     * any feeder-T1/local-T2 tier, try to fill physical VRAM to ~pct% and place
     * only the remainder in host T2, returning ONE contiguous VA.  Gated on a
     * large buffer, meaningful free physical VRAM, and non-feeder-exclusive
     * mode.  Any failure falls through to the existing tiers unchanged. */
    if (gb_frontload_enabled() && !gb_feeder_exclusive() &&
        bytesize >= gb_frontload_min_bytes() &&
        atomic_load_explicit(&g_cached_free_vram, memory_order_relaxed) > (1ULL << 30)) {
        if (gb_frontload_split_alloc(dptr, bytesize) == CUDA_SUCCESS) {
            gb_log("smart_alloc: frontload split (%zu MB)", bytesize >> 20);
            return CUDA_SUCCESS;
        }
        gb_log("smart_alloc: frontload declined for %zu MB — using existing tiers",
               bytesize >> 20);
    }

    /* U3: count this overflow event and update rolling eviction rate.
     * Skip during MODEL_LOAD/INIT: sequential weight placement is not thrashing;
     * counting it saturates the rate counter instantly and blocks feeder T1. */
    if (cur_phase >= GB_PHASE_INFERENCE)
        gb_evict_rate_tick();
    uint32_t evict_rate = atomic_load_explicit(&g_t1_evict_rate, memory_order_relaxed);

    /* U6: mid-pressure gate - when T2 > 82% and request > 1 MB, skip T2 tiers entirely
     * to prevent a single large alloc from crossing the CAP threshold in one step. */
    int skip_t2 = (bytesize > (1UL << 20) &&
                   atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed)
                   >= gb_effective_t2_mid());

    /* ── TIER 1: Feeder T1 (GPU VRAM) - fastest remote tier ── */
    /* U3: if T1 is thrashing (>20 evictions/s), skip T1 this call and go to T2.
     * Only relevant during INFERENCE/STEADY - during MODEL_LOAD evict_rate is 0. */
    /* D3: skip quarantined feeders; D2/U4: de-prioritize throttled/degraded feeders */
    if (evict_rate < GB_T1_THRASH_THRESHOLD) {
        if (gb_try_feeder_alloc_tier(dptr, bytesize, GB_ALLOC_TIER_T1) == 0) {
            gb_log("smart_alloc: feeder T1 (%zu MB, evict_rate=%u phase=%d)",
                   bytesize >> 20, evict_rate, cur_phase);
            gb_tier_record_alloc(GB_TIER_T1_FEEDER, bytesize);
            return CUDA_SUCCESS;
        }
    } else {
        gb_log("smart_alloc: T1 thrashing (rate=%u/s) - skip feeder T1", evict_rate);
    }

    /* ── TIER 2: DDR - route based on speed comparison ── */
    /* D1: weight feeder speed by PCIe link BW when PCIe is the bottleneck.
     * NOTE: pcie_bw is in MB/s; compare directly as effective throughput cap. */
    uint32_t local_t2  = get_local_ddr_speed();
    uint32_t feeder_t2 = 0;
    if (gb_netc_is_active()) {
        int _n = gb_netc_remote_gpu_count();
        for (int i = 0; i < _n; i++) {
            uint32_t s = (uint32_t)gb_netc_t2_speed_mts(i);
            if (s > feeder_t2) feeder_t2 = s;
        }
    }

    /* Documented tier order (CLAUDE.md): T2_host BEFORE T2_feeder — always.
     * A MODEL_LOAD "[cluster-load]" preference for feeder T2 was tried on
     * 2026-07-06 (route the whole weights buffer to the feeder so dispatch
     * follows the data) and reverted: (a) ggml clears buffers with cudaMemset,
     * which has no shim hook — the feeder fake ptr reaches real CUDA and
     * aborts llama-server with "invalid argument"; (b) kv_reserve keeps KV in
     * local T1, so decode kernels mix local and remote args, which remote
     * dispatch cannot serve.  Local T2 zerocopy is the proven, faster path
     * for a single overflow buffer (raw DDR MT/s comparison ignores the TCP
     * hop, so it is not a valid reason to jump the order either).  Feeder T2
     * remains the fallback when local T2 is exhausted.  (feeder_t2 kept for
     * the log line only.) */
    if (!skip_t2) {
        /* Feeder-exclusive mode: keep the working set co-resident on the
         * feeder — feeder T2 BEFORE local T2 (local kept as last resort so
         * the process survives a full feeder). */
        if (gb_feeder_exclusive() &&
            gb_try_feeder_alloc_tier(dptr, bytesize, GB_ALLOC_TIER_T2) == 0) {
            gb_log("smart_alloc: feeder T2 [exclusive] (%zu MB)", bytesize >> 20);
            gb_tier_record_alloc(GB_TIER_T2_FEEDER, bytesize);
            return CUDA_SUCCESS;
        }
        if (gb_overflow_alloc(dptr, bytesize) == CUDA_SUCCESS) {
            gb_log("smart_alloc: local T2 (%u MT/s, %zu MB)", local_t2, bytesize >> 20);
            gb_tier_record_alloc(GB_TIER_T2_LOCAL, bytesize);
            return CUDA_SUCCESS;
        }
        if (!gb_feeder_exclusive() &&
            gb_try_feeder_alloc_tier(dptr, bytesize, GB_ALLOC_TIER_T2) == 0) {
            gb_log("smart_alloc: feeder T2 fallback (%u vs %u MT/s, %zu MB)",
                   feeder_t2, local_t2, bytesize >> 20);
            gb_tier_record_alloc(GB_TIER_T2_FEEDER, bytesize);
            return CUDA_SUCCESS;
        }
    } else {
        gb_log("smart_alloc: MID gate skip T2 (%zu MB > 1 MB, T2 > %d%%)",
               bytesize >> 20, GB_T2_MID_PCT);
    }

    /* ── TIER 3: NVMe - route based on speed comparison ── */
    uint32_t local_t3  = get_local_nvme_speed_mbs();
    uint32_t feeder_t3 = 0;
    if (gb_netc_is_active()) {
        int _n = gb_netc_remote_gpu_count();
        for (int i = 0; i < _n; i++) {
            uint32_t s = (uint32_t)gb_netc_t3_speed_mbs(i);
            if (s > feeder_t3) feeder_t3 = s;
        }
    }

    if (feeder_t3 > local_t3) {
        if (gb_try_feeder_alloc_tier(dptr, bytesize, GB_ALLOC_TIER_T3) == 0) {
            gb_log("smart_alloc: feeder T3 faster (%u > %u MB/s, %zu MB)",
                   feeder_t3, local_t3, bytesize >> 20);
            gb_tier_record_alloc(GB_TIER_T3_FEEDER, bytesize);
            return CUDA_SUCCESS;
        }
    }

    if (gb_overflow_alloc(dptr, bytesize) == CUDA_SUCCESS) {
        gb_log("smart_alloc: local T2/T3 overflow (%zu MB)", bytesize >> 20);
        gb_tier_record_alloc(GB_TIER_T3_LOCAL, bytesize);
        return CUDA_SUCCESS;
    }

    if (feeder_t3 <= local_t3 &&
        gb_try_feeder_alloc_tier(dptr, bytesize, GB_ALLOC_TIER_T3) == 0) {
        gb_log("smart_alloc: feeder T3 last-resort (%zu MB)", bytesize >> 20);
        gb_tier_record_alloc(GB_TIER_T3_FEEDER, bytesize);
        return CUDA_SUCCESS;
    }

    /* R4+U1: OOM callback - use ARC eviction to free cold KV blocks first,
     * then warm blocks, then retry T2 once before giving up. */
    {
        static _Atomic int oom_in_progress = 0;
        if (atomic_exchange_explicit(&oom_in_progress, 1, memory_order_acq_rel) == 0) {
            gb_log("smart_alloc: OOM - ARC-evicting KV blocks and retrying");
            /* U1: target evicting at least bytesize + 10% headroom */
            size_t target = bytesize + bytesize / 10;
            size_t freed = gb_htable_evict_kv_arc(target);
            if (freed < target) {
                /* Cold KV not enough - also flush non-KV weight overflow */
                gb_htable_flush(1);
            }
            /* Single retry after eviction */
            if (gb_overflow_alloc(dptr, bytesize) == CUDA_SUCCESS) {
                atomic_store_explicit(&oom_in_progress, 0, memory_order_release);
                gb_tier_record_alloc(GB_TIER_T2_LOCAL, bytesize);
                return CUDA_SUCCESS;
            }
            atomic_store_explicit(&oom_in_progress, 0, memory_order_release);
        }
    }

    gb_log("smart_alloc: FULL OOM - all tiers exhausted at %zu MB", bytesize >> 20);
    GB_NVTX_EVENT("OOM_FULL", "ALL", bytesize >> 20, 0, "all_tiers_exhausted");
    return CUDA_ERROR_OUT_OF_MEMORY;
}

/* Helper: add a successful overflow alloc to the migration batch; flush when full.
 * Called by the cudaMalloc / cuMemAlloc hooks after gb_smart_overflow_alloc succeeds
 * for local T2/T3 paths that may need a follow-up host-side data migration.
 * Remote (feeder) allocs are excluded - data movement is handled by gb_netc. */
static void gb_batch_collect(CUdeviceptr dptr, void *host_ptr, size_t size)
{
    pthread_mutex_lock(&gb_migrate_lock);
    if (!host_ptr) { pthread_mutex_unlock(&gb_migrate_lock); return; }
    if (g_migrate_batch.count == GB_BATCH_MAX)
        gb_batch_flush(&g_migrate_batch);
    g_migrate_batch.entries[g_migrate_batch.count++] =
        (struct gb_migrate_entry){ .src = dptr, .dst = host_ptr, .size = size };
    pthread_mutex_unlock(&gb_migrate_lock);
}

/* ------------------------------------------------------------------ */
/*  cuMemAlloc_v2 override                                              */
/* ------------------------------------------------------------------ */

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
{
    CUresult ret;

    if (!initialized || !real_cuMemAlloc_v2)
        return CUDA_ERROR_OUT_OF_MEMORY;

    /* Primary remote path: Ollama called cudaSetDevice(N) for a cluster feeder
     * (GREENBOOST_GGML_2DEV) - allocate directly on that feeder, mirroring
     * cudaMalloc's active-remote branch. ggml-cuda's legacy pool normally
     * allocates via the runtime API, but cover the driver-API entry point too
     * so any caller that uses cuMemAlloc_v2 directly on device 1 still lands
     * on the feeder instead of silently falling through to real_cuMemAlloc_v2
     * (which would allocate on the actual current LOCAL CUDA context). */
    if (gb_netc_is_active()) {
        int _ri = gb_netc_get_active_remote();
        if (_ri >= 0) {
            uint64_t _fake = 0;
            if (gb_netc_malloc(_ri, (uint64_t)bytesize, 0, &_fake) == 0) {
                gb_log("cuMemAlloc_v2 remote[%d]: %zu MB → fake=0x%llx", _ri, bytesize >> 20,
                       (unsigned long long)_fake);
                *dptr = (CUdeviceptr)_fake;
                return CUDA_SUCCESS;
            }
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
    }

    if (gb_needs_overflow(bytesize)) {
        CUresult or_ret = gb_smart_overflow_alloc(dptr, bytesize);
        if (or_ret == CUDA_SUCCESS) return CUDA_SUCCESS;
        gb_log("overflow alloc failed (ret=%d) - local+remote full", or_ret);
        return or_ret;
    }

    /* T1-saturation feeder routing: physical T1 full → feeder T1 before kernel uses local T2 */
    if (gb_netc_is_active() && gb_physical_vram_bytes > 0 &&
        atomic_load_explicit(&g_local_t1_alloc_bytes, memory_order_relaxed) >= gb_physical_vram_bytes) {
        if (gb_try_feeder_alloc(dptr, bytesize, 0) == 0) {
            gb_tier_record_alloc(GB_TIER_T1_FEEDER, bytesize);
            atomic_fetch_add_explicit(&g_remote_alloc_count, 1, memory_order_relaxed);
            atomic_fetch_add_explicit(&g_remote_alloc_mb, bytesize >> 20, memory_order_relaxed);
            gb_maybe_write_stats();
            return CUDA_SUCCESS;
        }
        /* Feeder T1 full - cascade to T2/T3 (local and feeder); local T1 is saturated so
         * falling through to real_cuMemAlloc_v2 would also fail and bypass all remote tiers. */
        return gb_smart_overflow_alloc(dptr, bytesize);
    }

    atomic_store_explicit(&g_last_any_alloc_ms, gb_now_ms(), memory_order_relaxed);
    atomic_fetch_add_explicit(&g_local_t1_alloc_bytes, bytesize, memory_order_relaxed);
    ret = real_cuMemAlloc_v2(dptr, bytesize);
    if (ret == CUDA_SUCCESS) {
        gb_tier_record_alloc(GB_TIER_T1_LOCAL, bytesize);
        if (!ht_insert(*dptr, bytesize, 0, -1, NULL, -1, NULL)) {
            fprintf(stderr, "[GreenBoost] ERR: ht_insert failed (HT full) for cuMemAlloc_v2 %zu MB"
                    " - freeing allocation to avoid leak\n", bytesize >> 20);
            if (real_cuMemFree_v2) real_cuMemFree_v2(*dptr);
            atomic_fetch_sub_explicit(&g_local_t1_alloc_bytes, bytesize, memory_order_relaxed);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        gb_maybe_track_kv_t1(*dptr, bytesize);
        return ret;
    }
    /* Undo the speculative T1 accounting before considering an overflow fallback. */
    atomic_fetch_sub_explicit(&g_local_t1_alloc_bytes, bytesize, memory_order_relaxed);
    /* OOM fallback: gb_needs_overflow said T1 had room but real_cuMemAlloc_v2
     * disagreed.  Causes: cached cuMemGetInfo lagged behind a burst of allocations
     * (e.g. PyTorch pipe.to("cuda") materialising a 22 GB BF16 model in seconds)
     * or T1 is fragmented so a small contiguous request fails despite free bytes.
     * Retry through smart_alloc so we land in local T2 / feeder instead of
     * propagating OOM to the caller. */
    if (ret == CUDA_ERROR_OUT_OF_MEMORY) {
        gb_log("cuMemAlloc_v2 fallback: real driver OOM at %zu MB - retrying via smart_alloc",
               bytesize >> 20);
        GB_NVTX_EVENT("OOM_T1_FALLBACK", "T1_GPU", bytesize >> 20, 0, "cuMemAlloc_v2_oom_to_smart");
        CUresult or_ret = gb_smart_overflow_alloc(dptr, bytesize);
        if (or_ret == CUDA_SUCCESS) {
            gb_maybe_write_stats();
            return CUDA_SUCCESS;
        }
        return or_ret;
    }
    return ret;
}

/* ------------------------------------------------------------------ */
/*  cuMemCreate override (CUDA VMM - used by ggml/Ollama 0.18+)        */
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
    GB_NVTX_PUSH("GB:T1_vmm_alloc", GB_NVTX_COLOR_T1);
    ret = real_cuMemCreate(handle, size, prop, flags);
    GB_NVTX_POP();
    if (ret == CUDA_SUCCESS) {
        GB_NVTX_EVENT("ALLOC_VMM_T1", "T1_GPU", size >> 20, 0, "cuMemCreate_device_ok");
        return CUDA_SUCCESS;
    }

    /* Lazy cc re-probe: cuMemCreate intercept may fire before gb_overflow_alloc_ex
     * (PyTorch expandable_segments goes directly here, not through cudaMalloc). */
    if (gb_cc_major == 0 && real_cuDeviceGetAttribute) {
        int cc = 0;
        if (real_cuDeviceGetAttribute(&cc, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0) == CUDA_SUCCESS && cc > 0) {
            gb_cc_major = cc;
            fprintf(stderr, "[GreenBoost] Deferred cc probe (cuMemCreate): Compute %d.x\n", cc);
        }
    }

    /* Device OOM → fall back to host-pinned VMM allocation.
     * GPU accesses host-pinned VMM memory over PCIe (~32 GB/s) -
     * same bandwidth class as Path B.  No changes to cuMemMap/cuMemSetAccess
     * are required; they work identically for host-backed handles.
     *
     * On Blackwell (cc >= 12), CU_MEM_LOCATION_TYPE_HOST gives a fabric/DMA
     * pointer that compute SMs cannot access - must use HOST_NUMA_CURRENT.
     *
     * DESIGN RULE: GreenBoost always routes inference to GPU compute paths.
     * HOST VMM (cuMemCreate with CU_MEM_LOCATION_TYPE_HOST[_NUMA_CURRENT]) is a
     * GPU-compute-ready path - GPU accesses pages over PCIe, no CPU fallback.
     * MemAvailable is NOT the right guard here: greenboost.ko pre-allocates the
     * T2 pool as locked hugepages that do not appear in MemAvailable.  Using
     * MemAvailable would refuse legitimate T2 pool capacity.  Use T2 pool cap. */
    if (ret == CUDA_ERROR_OUT_OF_MEMORY) {
        /* T2 pool capacity check - only guard that matters for HOST VMM.
         * eff_cap = 88% of gb_t2_pool_bytes; prevents overcommit with headroom.
         * If CUDA's cuMemCreate(HOST) genuinely cannot get lockable memory
         * (e.g. ulimit -l too low), it returns OOM and we propagate it. */
        {
            size_t t2_used = atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed);
            size_t eff_cap = gb_effective_t2_cap();
            if (eff_cap > 0 && (t2_used >= eff_cap || size > eff_cap - t2_used)) {
                gb_log("cuMemCreate VMM skip: T2 pool cap %zu/%zu MB, req %zu MB",
                       t2_used >> 20, eff_cap >> 20, size >> 20);
                GB_NVTX_EVENT("OOM_T2_CAP", "T2_DDR", size >> 20, 0, "vmm_host_t2_cap_exceeded");
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
        }

        /* On Blackwell desktop PCIe, cuMemCreate(HOST/HOST_NUMA_CURRENT) produces
         * DMA-only handles - SM kernels get CUDA_ERROR_INVALID_RESOURCE_HANDLE when
         * they touch the mapped VA.  Skip the host fallback unless explicitly allowed;
         * the caller (ggml) should be using the cudaMalloc legacy pool instead
         * (forced by the cuMemAddressReserve intercept above). */
        if (gb_cc_major >= 12) {
            static int gb_cm_allow_vmm = -1;
            if (__builtin_expect(gb_cm_allow_vmm < 0, 0)) {
                const char *e = getenv("GREENBOOST_BLACKWELL_ALLOW_VMM");
                gb_cm_allow_vmm = (e && e[0] != '0') ? 1 : 0;
                if (!gb_cm_allow_vmm)
                    fprintf(stderr,
                        "[GreenBoost] cuMemCreate: Blackwell cc=%d - skipping HOST_NUMA T2 "
                        "fallback (DMA-only on desktop PCIe; cudaMalloc→managed-UVM used "
                        "instead). Override: GREENBOOST_BLACKWELL_ALLOW_VMM=1\n", gb_cc_major);
            }
            if (!gb_cm_allow_vmm) {
                GB_NVTX_EVENT("OOM_VMM_BLACKWELL_SKIP", "T2_DDR", size >> 20, 0,
                              "cuMemCreate_host_skip_blackwell");
                return CUDA_ERROR_OUT_OF_MEMORY;
            }
        }

        CUmemAllocationProp host_prop = *prop;
        /* Use shared gb_vmm_host_loc_type cache (initialized and resolved in
         * gb_vmm_t2_alloc_blackwell_hostnuma - prefers HOST_NUMA_CURRENT, falls
         * back to HOST on single-socket UMA, printed once). */
        if (__builtin_expect(gb_vmm_host_loc_type == 0, 0))
            gb_vmm_host_loc_type = CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT;
        host_prop.location.type = gb_vmm_host_loc_type;
        host_prop.location.id   = 0;
        host_prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;

        GB_NVTX_PUSH("GB:T2_vmm_host_alloc", GB_NVTX_COLOR_T2);
        ret = real_cuMemCreate(handle, size, &host_prop, flags);
        if (ret == CUDA_ERROR_INVALID_VALUE &&
            host_prop.location.type == CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT) {
            gb_vmm_host_loc_type = CU_MEM_LOCATION_TYPE_HOST;
            fprintf(stderr, "[GreenBoost] cuMemCreate: HOST_NUMA_CURRENT unsupported, "
                    "switching to HOST (printed once)\n");
            host_prop.location.type = CU_MEM_LOCATION_TYPE_HOST;
            ret = real_cuMemCreate(handle, size, &host_prop, flags);
        }
        GB_NVTX_POP();
        if (ret == CUDA_SUCCESS) {
            atomic_fetch_add_explicit(&gb_t2_overflow_bytes, size, memory_order_relaxed);
            __sync_fetch_and_add(&gb_path_b_count, 1);
            /* Track handle so cuMemRelease can decrement T2 accounting.
             * This is required for PyTorch expandable_segments: if the
             * subsequent cuMemMap fails (CUDA rejects mixing host and device
             * allocations in the same virtual address range), PyTorch calls
             * cuMemRelease - without this record T2 bytes would leak upward
             * and block future allocations even though no memory is in use. */
            vmm_ht_insert(*handle, size);
            gb_maybe_write_stats();
            gb_log("cuMemCreate VMM fallback: %zu MB → host-pinned (PCIe path) t2_total=%zu MB",
                   size >> 20,
                   atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed) >> 20);
            GB_NVTX_EVENT("ALLOC_VMM_HOST", "T2_DDR", size >> 20, 0, "cuMemCreate_host_pinned_ok");
            return CUDA_SUCCESS;
        }
        GB_NVTX_EVENT("OOM_VMM_HOST", "T2_DDR", size >> 20, 0, "cuMemCreate_host_pinned_failed");
        fprintf(stderr,
                "[GreenBoost] cuMemCreate VMM host fallback FAILED ret=%d for %zu MB\n",
                ret, (size >> 20));
    }

    return ret;
}

/* ------------------------------------------------------------------ */
/*  cuMemRelease override (CUDA VMM - companion to cuMemCreate)         */
/*                                                                      */
/*  Called by the app to release a physical allocation handle created   */
/*  by cuMemCreate.  For our host-backed T2 handles (created when T1   */
/*  overflows), we must decrement gb_t2_overflow_bytes here.           */
/*                                                                      */
/*  The critical case for PyTorch expandable_segments:                  */
/*    1. cuMemCreate(device_prop) → T1 full → fallback to host_prop    */
/*       GreenBoost: increments T2, inserts into vmm_ht                */
/*    2. cuMemMap(existing_device_va, ..., host_handle) → FAILS        */
/*       CUDA driver: cannot mix device and host allocations in the     */
/*       same virtual address range                                     */
/*    3. PyTorch catch block: cuMemRelease(host_handle)                */
/*       GreenBoost: finds handle in vmm_ht, decrements T2, removes    */
/*    4. PyTorch retries with a NEW virtual address range (new segment) */
/*    5. cuMemCreate(device_prop) → still T1 full → host_prop fallback */
/*    6. cuMemMap(NEW_va, ..., host_handle) → SUCCESS (fresh range,    */
/*       all host-backed - no mixing)                                   */
/*    7. All subsequent allocations for this segment → T2 over PCIe   */
/* ------------------------------------------------------------------ */

CUresult cuMemRelease(CUmemGenericAllocationHandle handle)
{
    size_t sz = vmm_ht_remove(handle);
    if (sz > 0) {
        atomic_fetch_sub_explicit(&gb_t2_overflow_bytes, sz, memory_order_relaxed);
        gb_maybe_write_stats();
        gb_log("cuMemRelease host-backed T2 handle: freed %zu MB  t2_total=%zu MB",
               sz >> 20,
               atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed) >> 20);
    }
    if (real_cuMemRelease)
        return real_cuMemRelease(handle);
    return CUDA_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  cuMemAddressReserve intercept - Blackwell VMM pool disable          */
/*                                                                      */
/*  On Blackwell (cc >= 12) desktop PCIe (RTX 5070, 5080, 5090 etc.)   */
/*  the cuMemCreate(HOST / HOST_NUMA_CURRENT) T2 fallback in the        */
/*  cuMemCreate hook below produces DMA-only handles. When ggml-cuda's  */
/*  VMM pool maps such a handle with cuMemMap + cuMemSetAccess and then  */
/*  a CUDA kernel (e.g. IM2COL_3D) tries to read from the resulting     */
/*  VA, the driver returns CUDA_ERROR_INVALID_RESOURCE_HANDLE (400)     */
/*  because the SM cannot dereference DMA-only host-pinned memory over  */
/*  the PCIe fabric without ATS support.                                 */
/*                                                                       */
/*  The only SM-accessible T2 path on Blackwell PCIe is managed-UVM     */
/*  (cuMemAllocManaged + SET_PREFERRED_LOCATION=CPU + SET_ACCESSED_BY=  */
/*  GPU), which is already implemented in gb_vmm_t2_alloc_blackwell_    */
/*  managed() and routed from the cudaMalloc/cuMemAlloc overflow path.   */
/*                                                                       */
/*  By returning CUDA_ERROR_NOT_SUPPORTED here on Blackwell, we prevent  */
/*  ggml-cuda from setting up its cuMemCreate/cuMemMap VMM pool          */
/*  (ggml_cuda_vmm_available() tests cuMemAddressReserve at init and     */
/*  falls back to the legacy cudaMalloc-based pool on any failure).      */
/*  With the legacy pool, all model-weight allocations go through        */
/*  cudaMalloc; when T1 VRAM is exhausted the cudaMalloc hook routes     */
/*  overflow through gb_overflow_alloc_ex → gb_vmm_t2_alloc_blackwell_  */
/*  managed(), which returns a valid SM-accessible managed-UVM pointer.  */
/*                                                                       */
/*  Override: set GREENBOOST_BLACKWELL_ALLOW_VMM=1 to re-enable ggml's  */
/*  VMM pool on Blackwell (useful for ATS-capable Blackwell server SKUs  */
/*  where HOST_NUMA_CURRENT + cuMemMap IS SM-accessible).               */
/* ------------------------------------------------------------------ */

CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                             CUdeviceptr addr, unsigned long long flags)
{
    if (initialized) {
        /* Lazy cc re-probe (same pattern as cuMemCreate and gb_overflow_alloc_ex). */
        if (gb_cc_major == 0 && real_cuDeviceGetAttribute) {
            int cc = 0;
            if (real_cuDeviceGetAttribute(&cc, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0)
                    == CUDA_SUCCESS && cc > 0) {
                gb_cc_major = cc;
                fprintf(stderr, "[GreenBoost] Deferred cc probe (cuMemAddressReserve): Compute %d.x\n", cc);
            }
        }

        if (gb_cc_major >= 12) {
            static int allow_vmm = -1;
            if (__builtin_expect(allow_vmm < 0, 0)) {
                const char *e = getenv("GREENBOOST_BLACKWELL_ALLOW_VMM");
                allow_vmm = (e && e[0] != '0') ? 1 : 0;
                if (!allow_vmm)
                    fprintf(stderr,
                        "[GreenBoost] cuMemAddressReserve: Blackwell cc=%d - disabling ggml VMM pool "
                        "(cuMemCreate HOST paths are DMA-only on desktop PCIe; using cudaMalloc→"
                        "managed-UVM for SM-accessible T2). Override: GREENBOOST_BLACKWELL_ALLOW_VMM=1\n",
                        gb_cc_major);
            }
            if (!allow_vmm)
                return CUDA_ERROR_NOT_SUPPORTED;
        }
    }
    return real_cuMemAddressReserve ? real_cuMemAddressReserve(ptr, size, alignment, addr, flags)
                                    : CUDA_ERROR_NOT_SUPPORTED;
}

/* ------------------------------------------------------------------ */
/*  cuMemMap intercept with VMM accounting                              */
/*                                                                      */
/*  CUDA's driver handles both device-backed and host-backed handles    */
/*  correctly when the handle type is consistent within a virtual       */
/*  address range - we don't change the mapping behaviour itself.       */
/*  But we DO:                                                          */
/*    1. detect whether the handle is one of our T2 host-pinned         */
/*       allocations (recorded by cuMemCreate's host fallback) so       */
/*       diagnostics can attribute the mapping to T2;                   */
/*    2. wrap the real call in an NVTX range and emit an event so the   */
/*       VMM path shows up in vitals (path_b counters and timeline);    */
/*    3. log mixed-tier failures explicitly to make the                 */
/*       cuMemCreate→cuMemRelease→retry dance from PyTorch's            */
/*       expandable_segments path observable.                           */
/*                                                                      */
/*  Intercepting also keeps dlsym callers (Ollama's ggml backend built  */
/*  with RTLD_DEEPBIND stripped) routed through our hook table rather   */
/*  than the real libcuda.so.1.                                         */
/* ------------------------------------------------------------------ */

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle, unsigned long long flags)
{
    if (!real_cuMemMap)
        return CUDA_ERROR_NOT_SUPPORTED;

    /* Peek at vmm_ht without removing - we use a probe-and-restore so the
     * handle stays valid for any later cuMemRelease.  vmm_ht only stores
     * host-backed handles, so a hit identifies a T2 (PCIe-DMA) mapping. */
    int t2_backed = 0;
    {
        uint32_t i, slot;
        pthread_mutex_lock(&gb_vmm_ht_lock);
        slot = vmm_ht_hash(handle) & VMM_HT_MASK;
        for (i = 0; i < VMM_HT_SIZE; i++) {
            vmm_ht_entry_t *e = &gb_vmm_ht[(slot + i) & VMM_HT_MASK];
            if (e->handle == VMM_HT_EMPTY) break;
            if (e->handle == handle) { t2_backed = 1; break; }
        }
        pthread_mutex_unlock(&gb_vmm_ht_lock);
    }

    if (t2_backed) GB_NVTX_PUSH("GB:VMM_map_T2", GB_NVTX_COLOR_T2);
    else           GB_NVTX_PUSH("GB:VMM_map_T1", GB_NVTX_COLOR_T1);
    CUresult ret = real_cuMemMap(ptr, size, offset, handle, flags);
    GB_NVTX_POP();

    if (ret == CUDA_SUCCESS) {
        GB_NVTX_EVENT(t2_backed ? "MAP_VMM_T2" : "MAP_VMM_T1",
                      t2_backed ? "T2_DDR" : "T1_GPU",
                      size >> 20, (uint64_t)ptr,
                      t2_backed ? "cuMemMap_host_ok" : "cuMemMap_device_ok");
    } else {
        /* Mixed-tier mismatch (CUDA rejects host-backed handle in a device-backed
         * address range) is the expected first step of PyTorch's expandable_segments
         * retry loop: app then calls cuMemRelease and re-creates a fresh segment. */
        gb_log("cuMemMap failed ret=%d %zu MB ptr=0x%llx %s",
               ret, size >> 20, (unsigned long long)ptr,
               t2_backed ? "(T2 host handle into existing range)" : "");
        GB_NVTX_EVENT(t2_backed ? "MAP_FAIL_VMM_T2" : "MAP_FAIL_VMM_T1",
                      t2_backed ? "T2_DDR" : "T1_GPU",
                      size >> 20, (uint64_t)ptr, "cuMemMap_failed");
    }
    return ret;
}

/* ------------------------------------------------------------------ */
/*  cuMemSetAccess intercept                                            */
/*                                                                      */
/*  cuMemSetAccess sets the access mask (READ / READWRITE) for a        */
/*  device-virtual range previously mapped via cuMemMap.  We pass it    */
/*  through unchanged - the real driver must set access bits for the    */
/*  GPU to read/write the mapping - and emit an NVTX event so the VMM   */
/*  configuration step is visible in vitals.                            */
/* ------------------------------------------------------------------ */

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                        const CUmemAccessDesc *desc, size_t count)
{
    if (!real_cuMemSetAccess) {
        if (!initialized)
            return CUDA_ERROR_NOT_INITIALIZED;
        real_cuMemSetAccess = (pfn_cuMemSetAccess)dlsym(RTLD_NEXT, "cuMemSetAccess");
        if (!real_cuMemSetAccess) return CUDA_ERROR_NOT_SUPPORTED;
    }
    GB_NVTX_PUSH("GB:VMM_set_access", GB_NVTX_COLOR_T2);
    CUresult ret = real_cuMemSetAccess(ptr, size, desc, count);
    GB_NVTX_POP();
    if (ret != CUDA_SUCCESS) {
        gb_log("cuMemSetAccess failed ret=%d %zu MB ptr=0x%llx count=%zu",
               ret, size >> 20, (unsigned long long)ptr, count);
    }
    return ret;
}

/* ------------------------------------------------------------------ */
/*  cuMemUnmap intercept                                                */
/*                                                                      */
/*  Releases a virtual mapping created by cuMemMap (the backing         */
/*  physical handle still needs cuMemRelease).  Pass through to the     */
/*  real driver and emit NVTX for symmetry with cuMemMap.               */
/* ------------------------------------------------------------------ */

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size)
{
    if (!real_cuMemUnmap) {
        if (!initialized)
            return CUDA_SUCCESS; /* uninitialised teardown - best-effort */
        real_cuMemUnmap = (pfn_cuMemUnmap)dlsym(RTLD_NEXT, "cuMemUnmap");
        if (!real_cuMemUnmap) return CUDA_ERROR_NOT_SUPPORTED;
    }
    GB_NVTX_PUSH("GB:VMM_unmap", GB_NVTX_COLOR_T2);
    CUresult ret = real_cuMemUnmap(ptr, size);
    GB_NVTX_POP();
    if (ret == CUDA_SUCCESS) {
        GB_NVTX_EVENT("UNMAP_VMM", "T2_DDR", size >> 20, (uint64_t)ptr, "cuMemUnmap_ok");
    } else {
        gb_log("cuMemUnmap failed ret=%d %zu MB ptr=0x%llx",
               ret, size >> 20, (unsigned long long)ptr);
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

    /* AUD-06: NULL dptr (dptr == 0) is legal in CUDA - it is a documented no-op.
     * The hash table uses ptr == 0 as the empty-slot sentinel, so ht_remove(0, ...)
     * returns 0 (not found) and execution falls through to real_cuMemFree_v2(0),
     * which the driver also treats as a no-op.  No special case is needed here;
     * this comment documents the invariant so future refactors preserve it. */

    flags = ht_peek_flags(dptr);

    /* Blackwell T2: bvmm_ht pointers were never inserted into ht - check
     * first.  See gb_bvmm_free_dispatch (PR-D/R2) which deduplicates the
     * Path-A/B/C free dispatch that used to be triplicated here. */
    /* RULE #1 front-load split VAs are tracked in gb_fl_ht, not ht/bvmm_ht. */
    if (gb_frontload_free_dispatch(dptr))
        return CUDA_SUCCESS;

    if (gb_bvmm_free_dispatch(dptr))
        return CUDA_SUCCESS;

    if (ht_remove(dptr, &sz, &managed, &mapped_ptr, &fd, &ext_mem)) {
        /* REF-02: KV T1 release via shared helper (same logic as cudaFree). */
        gb_release_kv_t1_bytes(flags, sz, mapped_ptr, ext_mem, managed);
        gb_log("cuMemFree_v2 ptr=0x%llx size=%zu MB managed=%d mapped_ptr=%p fd=%d ext_mem=%p",
               (unsigned long long)dptr, sz >> 20, managed, mapped_ptr, fd, (void *)ext_mem);
        /* Path A zero-copy: CUDA owns the mapping - destroy the external memory handle. */
        if (ext_mem) {
            GB_NVTX_EVENT("FREE_A_ZEROCOPY", "T2_DDR", sz >> 20, dptr, "path_a_zerocopy_free");
            if (real_cudaDestroyExternalMemory)
                real_cudaDestroyExternalMemory(ext_mem);
            return CUDA_SUCCESS;
        }
        /* Path A pinned / Path B: host-registered memory - return to pool or unregister. */
        if (mapped_ptr) {
            GB_NVTX_EVENT("FREE_T2_PINNED", "T2_DDR", sz >> 20, dptr, "path_ab_pinned_free");
            if (gb_pool_contains(dptr)) {
                gb_pool_free(dptr, sz);
            } else {
                if (real_cuMemHostUnregister) real_cuMemHostUnregister(mapped_ptr);
                munmap(mapped_ptr, sz);
            }
            /* Decrement cumulative T2 counter - mirrors the increment in gb_overflow_alloc(). */
            atomic_fetch_sub_explicit(&gb_t2_overflow_bytes, sz, memory_order_relaxed);
            /* N11: keep SWA window counter in sync with normal free */
            if (flags & GB_ALLOC_KV_CACHE)
                atomic_fetch_sub_explicit(&g_kv_t2_live_bytes, sz, memory_order_relaxed);
        }
        if (fd >= 0)
            close(fd);
        /* DMA-BUF: dptr came from cuMemHostGetDevicePointer, not cuMemAlloc -
         * calling cuMemFree on it is invalid and causes a CUDA driver error.
         * Only call the real free for regular device allocations (managed=1 is
         * dead code now that Path C / UVM is removed). */
        if (!mapped_ptr) {
            /* Plain local T1 alloc - release T1 saturation accounting. */
            if (!managed)
                atomic_fetch_sub_explicit(&g_local_t1_alloc_bytes, sz, memory_order_relaxed);
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
        CUresult or_ret = gb_smart_overflow_alloc(dptr, bytesize);
        if (or_ret == CUDA_SUCCESS) return CUDA_SUCCESS;
    }

    ret = real_cuMemAllocAsync(dptr, bytesize, hStream);
    if (ret == CUDA_SUCCESS) {
        if (!ht_insert(*dptr, bytesize, 0, -1, NULL, -1, NULL)) {
            fprintf(stderr, "[GreenBoost] ERR: ht_insert failed (HT full) for cuMemAllocAsync %zu MB"
                    " - freeing allocation to avoid leak\n", bytesize >> 20);
            if (real_cuMemFreeAsync) real_cuMemFreeAsync(*dptr, hStream);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        gb_maybe_track_kv_t1(*dptr, bytesize);
    } else if (ret == CUDA_ERROR_OUT_OF_MEMORY) {
        /* OOM fallback: see cuMemAlloc_v2 for full rationale. */
        gb_log("cuMemAllocAsync fallback: real driver OOM at %zu MB - retrying via smart_alloc",
               bytesize >> 20);
        GB_NVTX_EVENT("OOM_T1_FALLBACK", "T1_GPU", bytesize >> 20, 0,
                      "cuMemAllocAsync_oom_to_smart");
        CUresult or_ret = gb_smart_overflow_alloc(dptr, bytesize);
        if (or_ret == CUDA_SUCCESS) {
            gb_maybe_write_stats();
            return CUDA_SUCCESS;
        }
        return or_ret;
    }
    /* MIN-08: cuMemAllocAsync is intentionally asymmetric here:
     * - Alloc: uses the async stream-ordered path (lower latency in busy streams)
     * - Free:  cuMemFreeAsync (below) handles the paired deallocation
     *
     * If cuMemAllocAsync falls back to cuMemAlloc_v2 (cc < 8 branch above),
     * the paired free is still cuMemFreeAsync → cuMemFree_v2 (the real_cuMemFreeAsync
     * path is bypassed for the sync fallback case, so no mismatch).
     * The overflow alloc path (gb_overflow_alloc) uses the same ext_mem/mapped_ptr
     * tracking as cudaMalloc - freed correctly by cuMemFreeAsync via ht_lookup. */
    return ret;
}

/* ------------------------------------------------------------------ */
/*  cuMemAllocFromPoolAsync override                                    */
/*                                                                      */
/*  PR-N/F-S7: PyTorch 2.4+ caching allocator with                       */
/*  expandable_segments:True allocates from CUmemoryPool handles via    */
/*  this entry point.  Without an interpose, allocations bypass         */
/*  GreenBoost overflow routing - PyTorch silently exhausts physical    */
/*  VRAM and OOMs as if GreenBoost weren't loaded.                       */
/*                                                                      */
/*  Strategy: oversize allocations bypass the pool and go through the   */
/*  overflow path (which manages host RAM / managed UVM); allocations   */
/*  that fit in physical VRAM go through the real pool API and get      */
/*  tracked in ht for the matching cuMemFreeAsync.  The pool handle is  */
/*  passed through unchanged when we delegate.                          */
/* ------------------------------------------------------------------ */
CUresult cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize,
                                  CUmemoryPool_handle pool, CUstream hStream)
{
    CUresult ret;

    if (!initialized)
        return CUDA_ERROR_OUT_OF_MEMORY;

    /* Symbol may be missing on driver < 11.2 - caller using this API on
     * such a driver gets ENOENT-style NOT_SUPPORTED, which is correct. */
    if (!real_cuMemAllocFromPoolAsync) {
        if (real_cuMemAllocAsync) {
            /* Pool ignored: degrade to the default pool via cuMemAllocAsync. */
            return cuMemAllocAsync(dptr, bytesize, hStream);
        }
        return cuMemAlloc_v2(dptr, bytesize);
    }

    /* Overflow path: same gate as cuMemAllocAsync / cudaMalloc. */
    if (gb_needs_overflow(bytesize)) {
        CUresult or_ret = gb_smart_overflow_alloc(dptr, bytesize);
        if (or_ret == CUDA_SUCCESS) {
            GB_NVTX_EVENT("ALLOC_T2_POOL", "T2_DDR", bytesize >> 20, *dptr,
                          "cuMemAllocFromPoolAsync_overflow");
            return CUDA_SUCCESS;
        }
        /* Fall through to the real pool alloc - caller may want a small
         * pool allocation even when overflow paths are unavailable. */
    }

    ret = real_cuMemAllocFromPoolAsync(dptr, bytesize, pool, hStream);
    if (ret == CUDA_SUCCESS) {
        if (!ht_insert(*dptr, bytesize, 0, -1, NULL, -1, NULL)) {
            fprintf(stderr, "[GreenBoost] ERR: ht_insert failed (HT full) for "
                    "cuMemAllocFromPoolAsync %zu MB - freeing\n", bytesize >> 20);
            if (real_cuMemFreeAsync) real_cuMemFreeAsync(*dptr, hStream);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        gb_maybe_track_kv_t1(*dptr, bytesize);
    } else if (ret == CUDA_ERROR_OUT_OF_MEMORY) {
        /* Pool OOM at the real allocator - retry via overflow path so the
         * caller still gets a backing allocation (host RAM / managed UVM).
         * Same semantics as cuMemAllocAsync's OOM fallback. */
        gb_log("cuMemAllocFromPoolAsync OOM at %zu MB - retrying via smart_alloc",
               bytesize >> 20);
        GB_NVTX_EVENT("OOM_T1_FALLBACK", "T1_GPU", bytesize >> 20, 0,
                      "cuMemAllocFromPoolAsync_oom_to_smart");
        CUresult or_ret = gb_smart_overflow_alloc(dptr, bytesize);
        if (or_ret == CUDA_SUCCESS) {
            gb_maybe_write_stats();
            return CUDA_SUCCESS;
        }
        return or_ret;
    }
    return ret;
}

/* ------------------------------------------------------------------ */
/*  cudaMalloc override                                                 */
/* ------------------------------------------------------------------ */

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
    GB_CUDART_ENSURE();
    cudaError_t ret;
    CUdeviceptr dptr = 0;

    if (!initialized) {
        /* VCM-01: deferred init - try completing now that CUDA may be available */
        gb_try_resume_deferred();
        if (!initialized) {
            /* Still not initialized - transparent pass-through (no T2 routing) */
            if (!real_cudaMalloc)
                real_cudaMalloc = (pfn_cudaMalloc)dlsym(RTLD_NEXT, "cudaMalloc");
            return real_cudaMalloc ? real_cudaMalloc(devPtr, size)
                                   : (cudaError_t)CUDA_ERROR_OUT_OF_MEMORY;
        }
    }

    gb_log("cudaMalloc hook called: %zu MB", size >> 20);

    /* Lazy resolve: libcudart may have been loaded by caller after our constructor.
     * RTLD_NEXT skips our own symbol and finds the real one in the next library. */
    if (!real_cudaMalloc)
        real_cudaMalloc = (pfn_cudaMalloc)dlsym(RTLD_NEXT, "cudaMalloc");
    if (!real_cudaMalloc)
        return (cudaError_t)CUDA_ERROR_OUT_OF_MEMORY;

    /* Primary remote path: Ollama called cudaSetDevice(N) for a cluster feeder.
     * Allocate directly on that feeder - no local fallback needed. */
    if (gb_netc_is_active()) {
        int _ri = gb_netc_get_active_remote();
        if (_ri >= 0) {
            uint64_t _fake = 0;
            if (gb_netc_malloc(_ri, (uint64_t)size, 0, &_fake) == 0) {
                gb_log("cudaMalloc remote[%d]: %zu MB → fake=0x%llx", _ri, size >> 20,
                       (unsigned long long)_fake);
                *devPtr = (void *)(uintptr_t)_fake;
                return CUDA_SUCCESS;
            }
            return (cudaError_t)CUDA_ERROR_OUT_OF_MEMORY;
        }
    }

    if (gb_needs_overflow(size)) {
        CUresult or_ret = gb_smart_overflow_alloc(&dptr, size);
        if (or_ret == CUDA_SUCCESS) {
            *devPtr = (void *)(uintptr_t)dptr;
            return CUDA_SUCCESS;
        }
        gb_log("cudaMalloc overflow failed (ret=%d) - local+remote full", or_ret);
        return (cudaError_t)or_ret;
    }

    /* T1-saturation feeder routing: physical T1 full → feeder T1 before kernel uses local T2 */
    if (gb_netc_is_active() && gb_physical_vram_bytes > 0 &&
        atomic_load_explicit(&g_local_t1_alloc_bytes, memory_order_relaxed) >= gb_physical_vram_bytes) {
        if (gb_try_feeder_alloc(&dptr, size, 0) == 0) {
            *devPtr = (void *)(uintptr_t)dptr;
            gb_tier_record_alloc(GB_TIER_T1_FEEDER, size);
            atomic_fetch_add_explicit(&g_remote_alloc_count, 1, memory_order_relaxed);
            atomic_fetch_add_explicit(&g_remote_alloc_mb, size >> 20, memory_order_relaxed);
            gb_maybe_write_stats();
            return CUDA_SUCCESS;
        }
        /* Feeder T1 full - cascade to T2/T3 (local and feeder); local T1 is saturated so
         * falling through to real_cudaMalloc would also fail and bypass all remote tiers. */
        {
            CUresult or_ret = gb_smart_overflow_alloc(&dptr, size);
            if (or_ret == CUDA_SUCCESS) { *devPtr = (void *)(uintptr_t)dptr; return CUDA_SUCCESS; }
            return (cudaError_t)or_ret;
        }
    }

    atomic_store_explicit(&g_last_any_alloc_ms, gb_now_ms(), memory_order_relaxed);
    atomic_fetch_add_explicit(&g_local_t1_alloc_bytes, size, memory_order_relaxed);
    GB_NVTX_PUSH("GB:T1_alloc", GB_NVTX_COLOR_T1);
    ret = real_cudaMalloc(devPtr, size);
    GB_NVTX_POP();
    if (ret == CUDA_SUCCESS) {
        CUdeviceptr _dp = (CUdeviceptr)(uintptr_t)*devPtr;
        gb_tier_record_alloc(GB_TIER_T1_LOCAL, size);
        if (!ht_insert(_dp, size, 0, -1, NULL, -1, NULL)) {
            fprintf(stderr, "[GreenBoost] ERR: ht_insert failed (HT full) for cudaMalloc %zu MB"
                    " - freeing allocation to avoid leak\n", size >> 20);
            if (real_cudaFree) real_cudaFree(*devPtr);
            atomic_fetch_sub_explicit(&g_local_t1_alloc_bytes, size, memory_order_relaxed);
            return (cudaError_t)CUDA_ERROR_OUT_OF_MEMORY;
        }
        gb_maybe_track_kv_t1(_dp, size);
        GB_NVTX_EVENT("ALLOC_T1_LOCAL", "T1_GPU", size >> 20, _dp, "cudaMalloc_ok");
        return ret;
    }
    atomic_fetch_sub_explicit(&g_local_t1_alloc_bytes, size, memory_order_relaxed);
    GB_NVTX_EVENT("OOM_T1_LOCAL", "T1_GPU", size >> 20, 0, "cudaMalloc_failed");
    /* OOM fallback: see cuMemAlloc_v2 for full rationale. PyTorch's caching
     * allocator running with expandable_segments:False reaches cudaMalloc;
     * a stale gb_needs_overflow gate or fragmented T1 must not surface OOM
     * to the caller while T2 still has capacity.  cudaErrorMemoryAllocation
     * (2) and CUDA_ERROR_OUT_OF_MEMORY (2) share the same numeric value. */
    if (ret == (cudaError_t)CUDA_ERROR_OUT_OF_MEMORY) {
        gb_log("cudaMalloc fallback: real driver OOM at %zu MB - retrying via smart_alloc",
               size >> 20);
        GB_NVTX_EVENT("OOM_T1_FALLBACK", "T1_GPU", size >> 20, 0, "cudaMalloc_oom_to_smart");
        CUresult or_ret = gb_smart_overflow_alloc(&dptr, size);
        if (or_ret == CUDA_SUCCESS) {
            *devPtr = (void *)(uintptr_t)dptr;
            gb_maybe_write_stats();
            return CUDA_SUCCESS;
        }
        return (cudaError_t)or_ret;
    }
    return ret;
}

/* ------------------------------------------------------------------ */
/*  cudaMallocAsync override                                            */
/* ------------------------------------------------------------------ */

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream)
{
    GB_CUDART_ENSURE();
    cudaError_t ret;
    CUdeviceptr dptr = 0;

    if (!initialized)
        return (cudaError_t)CUDA_ERROR_OUT_OF_MEMORY;

    /* Lazy resolve: RTLD_NEXT skips our own symbol, finds real cudaMallocAsync in libcudart */
    if (!real_cudaMallocAsync)
        real_cudaMallocAsync = (pfn_cudaMallocAsync)dlsym(RTLD_NEXT, "cudaMallocAsync");

    /* Fall back to sync cudaMalloc (stream ordering ignored - safe for model weights) */
    if (!real_cudaMallocAsync)
        return cudaMalloc(devPtr, size);

    if (gb_needs_overflow(size)) {
        CUresult or_ret = gb_smart_overflow_alloc(&dptr, size);
        if (or_ret == CUDA_SUCCESS) {
            *devPtr = (void *)(uintptr_t)dptr;
            return CUDA_SUCCESS;
        }
    }

    atomic_store_explicit(&g_last_any_alloc_ms, gb_now_ms(), memory_order_relaxed);
    ret = real_cudaMallocAsync(devPtr, size, stream);
    if (ret == CUDA_SUCCESS) {
        CUdeviceptr _dp = (CUdeviceptr)(uintptr_t)*devPtr;
        atomic_fetch_add_explicit(&g_local_t1_alloc_bytes, size, memory_order_relaxed);
        if (!ht_insert(_dp, size, 0, -1, NULL, -1, NULL)) {
            fprintf(stderr, "[GreenBoost] ERR: ht_insert failed (HT full) for cudaMallocAsync %zu MB"
                    " - freeing allocation to avoid leak\n", size >> 20);
            if (real_cudaFreeAsync) real_cudaFreeAsync(*devPtr, stream);
            atomic_fetch_sub_explicit(&g_local_t1_alloc_bytes, size, memory_order_relaxed);
            return (cudaError_t)CUDA_ERROR_OUT_OF_MEMORY;
        }
        gb_maybe_track_kv_t1(_dp, size);
        return ret;
    }
    /* OOM fallback: see cuMemAlloc_v2 for full rationale. */
    if (ret == (cudaError_t)CUDA_ERROR_OUT_OF_MEMORY) {
        gb_log("cudaMallocAsync fallback: real driver OOM at %zu MB - retrying via smart_alloc",
               size >> 20);
        GB_NVTX_EVENT("OOM_T1_FALLBACK", "T1_GPU", size >> 20, 0, "cudaMallocAsync_oom_to_smart");
        CUresult or_ret = gb_smart_overflow_alloc(&dptr, size);
        if (or_ret == CUDA_SUCCESS) {
            *devPtr = (void *)(uintptr_t)dptr;
            gb_maybe_write_stats();
            return CUDA_SUCCESS;
        }
        return (cudaError_t)or_ret;
    }
    return ret;
}

/* ------------------------------------------------------------------ */
/*  cudaFreeAsync override                                              */
/* ------------------------------------------------------------------ */

cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream)
{
    GB_CUDART_ENSURE();
    CUdeviceptr dptr = (CUdeviceptr)(uintptr_t)devPtr;
    size_t sz = 0;
    int managed = 0;
    void *mapped_ptr = NULL;
    int fd = -1;
    cudaExternalMemory_t ext_mem = NULL;

    if (!initialized)
        return CUDA_SUCCESS;

    if (!real_cudaFreeAsync)
        real_cudaFreeAsync = (pfn_cudaFreeAsync)dlsym(RTLD_NEXT, "cudaFreeAsync");

    /* Remote cluster pointer - delegate to netc */
    if (dptr && gb_is_remote_ptr((uint64_t)dptr)) {
        gb_netc_free((uint64_t)dptr);
        return CUDA_SUCCESS;
    }

    /* Blackwell T2: bvmm_ht pointers were never inserted into ht - check
     * first.  See gb_bvmm_free_dispatch (PR-D/R2) which deduplicates the
     * Path-A/B/C free dispatch that used to be triplicated here. */
    /* RULE #1 front-load split VAs are tracked in gb_fl_ht, not ht/bvmm_ht. */
    if (gb_frontload_free_dispatch(dptr))
        return CUDA_SUCCESS;

    if (gb_bvmm_free_dispatch(dptr))
        return CUDA_SUCCESS;

    if (ht_remove(dptr, &sz, &managed, &mapped_ptr, &fd, &ext_mem)) {
        if (!managed)
            atomic_fetch_add_explicit(&g_local_t1_alloc_bytes, -(ptrdiff_t)sz, memory_order_relaxed);
        if (mapped_ptr) {
            if (real_cuMemHostUnregister) real_cuMemHostUnregister(mapped_ptr);
            munmap(mapped_ptr, sz);
            atomic_fetch_sub_explicit(&gb_t2_overflow_bytes, sz, memory_order_relaxed);
        }
        if (fd >= 0)
            close(fd);
        if (ext_mem && real_cudaDestroyExternalMemory)
            real_cudaDestroyExternalMemory(ext_mem);
        if (!mapped_ptr && !ext_mem && real_cudaFreeAsync)
            return real_cudaFreeAsync(devPtr, stream);
        return CUDA_SUCCESS;
    }

    if (real_cudaFreeAsync)
        return real_cudaFreeAsync(devPtr, stream);
    return CUDA_SUCCESS;
}

/* ------------------------------------------------------------------ */
/*  cudaFree override                                                   */
/* ------------------------------------------------------------------ */

cudaError_t cudaFree(void *devPtr)
{
    GB_CUDART_ENSURE();
    void *mapped_ptr = NULL;
    int fd = -1;
    size_t sz = 0;
    int managed = 0;
    cudaExternalMemory_t ext_mem = NULL;
    CUdeviceptr dptr = (CUdeviceptr)(uintptr_t)devPtr;
    uint32_t flags;

    if (!initialized)
        return CUDA_SUCCESS; /* free before init: no-op is safe */

    /* Remote cluster pointer - delegate to netc, never reaches the real driver */
    if (dptr && gb_is_remote_ptr((uint64_t)dptr)) {
        gb_netc_free((uint64_t)dptr);
        return CUDA_SUCCESS;
    }

    if (!real_cudaFree)
        real_cudaFree = (pfn_cudaFree)dlsym(RTLD_NEXT, "cudaFree");
    if (!real_cudaFree)
        return CUDA_SUCCESS;

    /* AUD-06: cudaFree(NULL) is a documented CUDA no-op.  dptr == 0 is the
     * empty-slot sentinel in the hash table, so ht_remove(0,...) returns 0
     * (not found) and we fall through to real_cudaFree(NULL) - correct. */

    flags = ht_peek_flags(dptr);

    /* Blackwell T2: bvmm_ht pointers were never inserted into ht - check
     * first.  See gb_bvmm_free_dispatch (PR-D/R2) which deduplicates the
     * Path-A/B/C free dispatch that used to be triplicated here. */
    /* RULE #1 front-load split VAs are tracked in gb_fl_ht, not ht/bvmm_ht. */
    if (gb_frontload_free_dispatch(dptr))
        return CUDA_SUCCESS;

    if (gb_bvmm_free_dispatch(dptr))
        return CUDA_SUCCESS;

    if (ht_remove(dptr, &sz, &managed, &mapped_ptr, &fd, &ext_mem)) {
        /* REF-02: KV T1 release via shared helper (same logic as cuMemFree_v2). */
        gb_release_kv_t1_bytes(flags, sz, mapped_ptr, ext_mem, managed);
        gb_log("cudaFree ptr=0x%llx size=%zu MB managed=%d mapped_ptr=%p fd=%d ext_mem=%p",
               (unsigned long long)dptr, sz >> 20, managed, mapped_ptr, fd, (void *)ext_mem);
        /* Path A zero-copy: CUDA owns the mapping - destroy the external memory handle. */
        if (ext_mem) {
            if (real_cudaDestroyExternalMemory)
                real_cudaDestroyExternalMemory(ext_mem);
            return CUDA_SUCCESS;
        }
        /* Path A / Path B: host-registered memory - return to pool or unregister. */
        if (mapped_ptr) {
            if (gb_pool_contains(dptr)) {
                gb_pool_free(dptr, sz);
            } else {
                if (real_cuMemHostUnregister) real_cuMemHostUnregister(mapped_ptr);
                munmap(mapped_ptr, sz);
            }
            /* Decrement cumulative T2 counter - mirrors the increment in gb_overflow_alloc(). */
            atomic_fetch_sub_explicit(&gb_t2_overflow_bytes, sz, memory_order_relaxed);
            /* N11: keep SWA window counter in sync with normal free */
            if (flags & GB_ALLOC_KV_CACHE)
                atomic_fetch_sub_explicit(&g_kv_t2_live_bytes, sz, memory_order_relaxed);
        }
        if (fd >= 0)
            close(fd);
        /* DMA-BUF: dptr came from cuMemHostGetDevicePointer, not cudaMalloc -
         * must not pass to cudaFree. Only free regular device allocations. */
        if (!mapped_ptr) {
            if (!managed)
                atomic_fetch_sub_explicit(&g_local_t1_alloc_bytes, sz, memory_order_relaxed);
            cudaError_t _r = real_cudaFree(devPtr);
            /* PR-FREE-1: CUDA context teardown can race with model unload
             * (DEEP_IDLE reclaim).  cudaErrorInvalidValue (=1) on a T1 alloc
             * we own means the driver already freed it as part of context
             * cleanup — treat it as success so callers don't surface spurious
             * errors.  Any other non-zero code propagates unchanged. */
            return (_r == (cudaError_t)1) ? CUDA_SUCCESS : _r;
        }
        return CUDA_SUCCESS;
    }

    /* Fallthrough: pointer not tracked by GreenBoost.  Tolerate
     * cudaErrorInvalidValue (=1) for the same race reason as above — the
     * allocation may have already been cleaned up by the CUDA context
     * teardown path.  Other errors propagate to surface real bugs. */
    {
        cudaError_t _r = real_cudaFree(devPtr);
        return (_r == (cudaError_t)1) ? CUDA_SUCCESS : _r;
    }
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

    /* Path C anchor: managed-UVM entries are pinned to host RAM by
     * SET_PREFERRED_LOCATION=CPU.  A real prefetch with dst=device would
     * override the hint (CUDA Programming Guide §4.1.4.2: "cudaMemPrefetchAsync
     * may override [the preferred location] and allow the memory to migrate")
     * which silently moves the pages onto the GPU - VRAM oversubscribes,
     * subsequent allocs OOM, and the design intent collapses.  Treat
     * prefetch on managed-UVM as a no-op regardless of dstDevice. */
    {
        uint8_t bvmm_t = 0;
        if (bvmm_ht_peek_type(dptr, &bvmm_t) && bvmm_t == BVMM_TYPE_MANAGED) {
            GB_NVTX_EVENT("PREFETCH_T2_MANAGED_SKIP", "T2_DDR", count >> 20, dptr, "managed_uvm_anchored_cpu");
            return CUDA_SUCCESS;
        }
    }

    /* Check if this is a GreenBoost DMA-BUF allocation */
    if (ht_lookup(dptr, &sz, &managed, &mapped_ptr, &fd)) {
        if (mapped_ptr) {
            size_t chunk = count < sz ? count : sz;
            /* U18: pass full alloc bounds so double-buffer lookahead can safely
             * madvise the next tile without walking past the mmap region end. */
            enqueue_prefetch(mapped_ptr, chunk, mapped_ptr, sz);
        }
        /* We handled the prefetch via host thread, skip real CUDA prefetch
           because CUDA doesn't prefetch cuMemHostRegister memory implicitly */
        return CUDA_SUCCESS;
    }

    return real_cuMemPrefetchAsync(dptr, count, dstDevice, hStream);
}

cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream)
{
    GB_CUDART_ENSURE();
    void *mapped_ptr = NULL;
    size_t sz = 0;
    int managed = 0, fd = -1;
    CUdeviceptr dptr = (CUdeviceptr)(uintptr_t)devPtr;

    if (!initialized)
        return CUDA_SUCCESS;

    if (!real_cudaMemPrefetchAsync) {
        real_cudaMemPrefetchAsync = (pfn_cudaMemPrefetchAsync)dlsym(RTLD_NEXT, "cudaMemPrefetchAsync");
    }

    /* Path C anchor - see cuMemPrefetchAsync above for the rationale. */
    {
        uint8_t bvmm_t = 0;
        if (bvmm_ht_peek_type(dptr, &bvmm_t) && bvmm_t == BVMM_TYPE_MANAGED) {
            GB_NVTX_EVENT("PREFETCH_T2_MANAGED_SKIP", "T2_DDR", count >> 20, dptr, "managed_uvm_anchored_cpu");
            return (cudaError_t)CUDA_SUCCESS;
        }
    }

    if (ht_lookup(dptr, &sz, &managed, &mapped_ptr, &fd)) {
        if (mapped_ptr) {
            size_t chunk = count < sz ? count : sz;
            enqueue_prefetch(mapped_ptr, chunk, mapped_ptr, sz);
        }
        return (cudaError_t)CUDA_SUCCESS;
    }

    if (real_cudaMemPrefetchAsync)
        return real_cudaMemPrefetchAsync(devPtr, count, dstDevice, stream);

    return (cudaError_t)CUDA_SUCCESS;
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

    if (!initialized) {
        /* VCM-01: deferred init - try completing, then fall through if needed */
        gb_try_resume_deferred();
    }
    if (!initialized || !real_cuMemGetInfo) {
        /* Deferred init pending or shim inactive - transparent pass-through */
        pfn_cuMemGetInfo fn = real_cuMemGetInfo;
        if (!fn) fn = (pfn_cuMemGetInfo)dlsym(RTLD_NEXT, "cuMemGetInfo_v2");
        if (!fn) fn = (pfn_cuMemGetInfo)dlsym(RTLD_NEXT, "cuMemGetInfo");
        return fn ? fn(free_out, total_out) : CUDA_ERROR_NOT_SUPPORTED;
    }

    /* GREENBOOST_GGML_2DEV: when the caller's current device is the feeder
     * (cudaSetDevice(1) -> gb_netc_set_active_remote), report the feeder's
     * OWN real free/total instead of the aggregated single-vGPU pool -
     * ggml's multi-GPU layer-split sizer needs the true per-device numbers
     * to assign a sane layer count to device 1, not device 0's inflated
     * total. Live network round-trip; acceptable off the decode hot path
     * (called once per device at model-load split-sizing time). */
    {
        int _active_remote = gb_netc_get_active_remote();
        if (_active_remote >= 0) {
            uint64_t _rf = 0, _rt = 0;
            if (gb_netc_mem_info(_active_remote, &_rf, &_rt) == 0 && _rt > 0) {
                if (free_out)  *free_out  = (size_t)_rf;
                if (total_out) *total_out = (size_t)_rt;
                gb_log("cuMemGetInfo_v2: active_remote=%d free=%zuMB total=%zuMB",
                       _active_remote, (size_t)_rf >> 20, (size_t)_rt >> 20);
                return CUDA_SUCCESS;
            }
        }
    }

    ret = real_cuMemGetInfo(&real_free, &real_total);
    if (ret != CUDA_SUCCESS) {
        /* CUDA 13 / cu130: the driver API requires a current CUDA context.
         * llama.cpp / ComfyUI / ggml call cudaMemGetInfo before the first
         * cudaSetDevice, so no primary context has been created yet, and the
         * driver returns CUDA_ERROR_INVALID_CONTEXT (201).
         * Fall back to the runtime API, which calls cuDevicePrimaryCtxRetain
         * internally and establishes the context transparently.  We still
         * apply all virtual-pool inflation below so the reported numbers are
         * identical to the normal path. */
        if (ret == (CUresult)201 /* CUDA_ERROR_INVALID_CONTEXT */ ||
            ret == (CUresult)4   /* CUDA_ERROR_DEINITIALIZED */) {
            pfn_cudaMemGetInfo rt_fn = real_cudaMemGetInfo;
            if (!rt_fn)
                rt_fn = (pfn_cudaMemGetInfo)dlsym(RTLD_NEXT, "cudaMemGetInfo");
            if (rt_fn) {
                cudaError_t rt_err = rt_fn(&real_free, &real_total);
                if (rt_err != 0 /* cudaSuccess */)
                    return (CUresult)rt_err;
                /* Obtained real values via runtime fallback — apply inflation below */
            } else {
                return ret;
            }
        } else {
            return ret;
        }
    }

    /* total = real VRAM + local virtual pool (T2+T3) + all feeder memory.
     * free  = real VRAM free + T2 DDR available + feeder free.
     * During MODEL_LOAD we include T3 in free so the fit-check passes for
     * models larger than T1+T2.  During INFERENCE we exclude T3. */
    size_t t2_free = gb_t2_free_to_report();
    int phase = atomic_load_explicit(&g_alloc_phase, memory_order_relaxed);
    if (phase < (int)GB_PHASE_INFERENCE) {
        if (gb_virtual_vram_bytes > gb_t2_pool_bytes)
            t2_free += (gb_virtual_vram_bytes - gb_t2_pool_bytes);
    }

    /* Live feeder free memory (network round-trip; acceptable - not on hot path) */
    size_t remote_free = 0;
    if (gb_netc_is_active()) {
        int _n = gb_netc_remote_gpu_count();
        for (int _ri = 0; _ri < _n; _ri++) {
            uint64_t _f = 0, _t = 0;
            if (gb_netc_mem_info(_ri, &_f, &_t) == 0) remote_free += (size_t)_f;
        }
    }

    /* Workstation GPU headroom: subtract from physical VRAM free so --fit
     * leaves that much GPU memory for the desktop, compositor, and other
     * GPU-using processes.  When gaming_mode=1 (Proton wrapper active)
     * double the reserve so games can take VRAM back without inference
     * preempting them.  T2/T3 and remote memory are NOT reduced — they are
     * DDR/NVMe and do not affect GPU rendering. */
    size_t _ws_applied = gb_effective_workstation_reserve_bytes(real_free, real_total);
    {
        size_t ws = _ws_applied;
        /* Read gaming_mode from sysfs; cached, so only a cheap stat on hot path */
        static int _gaming_mode_cached = 0;
        static uint64_t _gaming_mode_ms = 0;
        uint64_t _now = gb_now_ms();
        if (_now - _gaming_mode_ms > 2000ULL) {
            _gaming_mode_cached = read_sysfs_bool(
                "/sys/module/greenboost/parameters/gaming_mode");
            _gaming_mode_ms = _now;
        }
        if (_gaming_mode_cached) ws *= 2;  /* double reserve when game is running */
        _ws_applied = ws;
        if (real_free > ws) real_free -= ws;
        else                real_free  = 0;
    }

    /* GREENBOOST_REPORT_PHYSICAL_VRAM=1 (diffusion/PyTorch workloads, see
     * cuDeviceTotalMem_v2 above): report REAL free/total, no pool inflation.
     * Without this branch, PyTorch's caching allocator (which sizes every
     * allocation and formats OOM errors off cudaMemGetInfo/cuMemGetInfo, NOT
     * cudaGetDeviceProperties/cuDeviceTotalMem_v2) still saw the inflated
     * total_virtual = real_total + gb_virtual_vram_bytes + cluster_remote
     * figure even when the caller had disabled inflation everywhere else —
     * confirmed 2026-07-09: a feeder torch process reported "total capacity
     * of 129.53 GiB" (an 8 GiB card + full cluster pool) from THIS function
     * while cuDeviceTotalMem_v2 correctly reported the physical figure, so
     * gb-quant/bf16 sizing decisions OOM'd against a capacity that was never
     * really there. */
    if (g_report_physical_vram) {
        if (free_out)  *free_out  = real_free;
        if (total_out) *total_out = real_total;
        gb_log("cuMemGetInfo_v2: reporting physical free=%zuMB total=%zuMB "
               "(inflation disabled by GREENBOOST_REPORT_PHYSICAL_VRAM=1)",
               real_free >> 20, real_total >> 20);
        return CUDA_SUCCESS;
    }

    if (free_out)  *free_out  = real_free  + t2_free + remote_free;
    if (total_out) *total_out = real_total + gb_virtual_vram_bytes + g_cluster_remote_total_bytes;

    gb_log("cuMemGetInfo_v2: real_free=%zuMB local_virt=%zuMB remote_free=%zuMB total=%zuMB (phase %d, ws_reserve=%zuMB)",
           real_free >> 20, t2_free >> 20, remote_free >> 20,
           (real_total + gb_virtual_vram_bytes + g_cluster_remote_total_bytes) >> 20, phase,
           _ws_applied >> 20);
    return CUDA_SUCCESS;
}

CUresult cuMemGetInfo(size_t *free_out, size_t *total_out)
{
    return cuMemGetInfo_v2(free_out, total_out);
}

cudaError_t cudaMemGetInfo(size_t *free_out, size_t *total_out)
{
    GB_CUDART_ENSURE();
    /* Ensure we have a real runtime function pointer.  GB_CUDART_ENSURE() runs
     * the lazy rebind, but the app's libcudart may not be mapped yet at the
     * moment of the first call (e.g. ComfyUI / llama.cpp call this very early),
     * leaving real_cudaMemGetInfo NULL.  Resolve on the spot as a last resort
     * so we can use it for context initialisation below. */
    if (!real_cudaMemGetInfo)
        real_cudaMemGetInfo = (pfn_cudaMemGetInfo)dlsym(RTLD_NEXT, "cudaMemGetInfo");
    /* Call the real runtime first so it can perform lazy CUDA context
     * initialisation if one does not exist yet (the driver-level
     * cuMemGetInfo_v2 below requires a current context and will return
     * CUDA_ERROR_INVALID_CONTEXT / cudaErrorDeviceUninitialized otherwise).
     * The real return values are discarded; we always override them with the
     * virtual pool totals reported by cuMemGetInfo_v2.
     */
    if (real_cudaMemGetInfo) {
        size_t _dummy_free, _dummy_total;
        cudaError_t rt = real_cudaMemGetInfo(&_dummy_free, &_dummy_total);
        if (rt != 0)  /* 0 == cudaSuccess */
            return rt;
    }
    return (cudaError_t)cuMemGetInfo_v2(free_out, total_out);
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

    if (!initialized || !real_cuDeviceTotalMem_v2) {
        /* Shim not yet active (GREENBOOST_DISABLE=1 or deferred init): transparent
         * pass-through so the caller sees the real physical VRAM without inflation. */
        pfn_cuDeviceTotalMem_v2 fn = real_cuDeviceTotalMem_v2;
        if (!fn) fn = (pfn_cuDeviceTotalMem_v2)dlsym(RTLD_NEXT, "cuDeviceTotalMem_v2");
        if (!fn) fn = (pfn_cuDeviceTotalMem_v2)dlsym(RTLD_NEXT, "cuDeviceTotalMem");
        return fn ? fn(bytes, dev) : CUDA_ERROR_NOT_SUPPORTED;
    }

    int local = 1;
    if (real_cuDeviceGetCount)
        real_cuDeviceGetCount(&local);

    if ((int)dev >= local) {
        if (bytes) {
            *bytes = gb_netc_remote_vram((int)dev - local);
        }
        return CUDA_SUCCESS;
    }

    ret = real_cuDeviceTotalMem_v2(bytes, dev);
    if (ret != CUDA_SUCCESS) {
        /* Invalid-context fallback (same as cuMemGetInfo_v2): derive total
         * from the runtime cudaMemGetInfo which performs cuDevicePrimaryCtxRetain
         * internally.  The inflation below then adds virtual pool bytes as usual. */
        if ((ret == (CUresult)201 /* CUDA_ERROR_INVALID_CONTEXT */ ||
             ret == (CUresult)4   /* CUDA_ERROR_DEINITIALIZED */) && bytes) {
            pfn_cudaMemGetInfo rt_fn = real_cudaMemGetInfo;
            if (!rt_fn)
                rt_fn = (pfn_cudaMemGetInfo)dlsym(RTLD_NEXT, "cudaMemGetInfo");
            if (rt_fn) {
                size_t rt_free = 0, rt_total = 0;
                if (rt_fn(&rt_free, &rt_total) == 0 /* cudaSuccess */) {
                    *bytes = rt_total;
                    ret = CUDA_SUCCESS;
                    /* fall through to apply inflation */
                } else {
                    return ret;
                }
            } else {
                return ret;
            }
        } else {
            return ret;
        }
    }
    if (bytes) {
        if (g_report_physical_vram) {
            gb_log("cuDeviceTotalMem_v2: reporting physical %zuMB (inflation disabled by GREENBOOST_REPORT_PHYSICAL_VRAM=1)",
                   *bytes >> 20);
            return ret;
        }
        /* Single-virtual-GPU: report phys + local T2/T3 + all feeder T1+T2+T3 */
        size_t phys = *bytes;
        size_t total_virtual = phys + gb_virtual_vram_bytes + g_cluster_remote_total_bytes;
        if (gb_nvlink_aggregated_bytes > 0)
            total_virtual += gb_nvlink_aggregated_bytes;
        /* Safety: if inflation produced 0 (virtual pool not yet init) but the
         * physical probe succeeded, return the real physical value.  Returning
         * 0 to Ollama's gpu-discover causes it to report total_vram=0 and place
         * the entire model on CPU — the worst possible failure mode. */
        if (total_virtual == 0 && phys > 0)
            total_virtual = phys;
        gb_log("cuDeviceTotalMem_v2: phys=%zuMB + local_virt=%zuMB + remote=%zuMB = %zuMB",
               phys >> 20, gb_virtual_vram_bytes >> 20,
               g_cluster_remote_total_bytes >> 20, total_virtual >> 20);
        *bytes = total_virtual;
    }
    return ret;
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev)
{
    return cuDeviceTotalMem_v2(bytes, dev);
}

CUresult cuDeviceGetAttribute(int *value, int attrib, CUdevice dev)
{
    if (!initialized || !real_cuDeviceGetAttribute) {
        /* Shim not yet active: transparent pass-through. */
        pfn_cuDeviceGetAttribute fn = real_cuDeviceGetAttribute;
        if (!fn) fn = (pfn_cuDeviceGetAttribute)dlsym(RTLD_NEXT, "cuDeviceGetAttribute");
        return fn ? fn(value, attrib, dev) : CUDA_ERROR_NOT_SUPPORTED;
    }

    int local = 1;
    if (real_cuDeviceGetCount)
        real_cuDeviceGetCount(&local);
    /* Never let a zero/failed device count route a LOCAL ordinal (esp. device
     * 0) to the feeder path: that returns *value=0 for shared-memory-per-block
     * attrs and Triton reports "shared memory Hardware limit: 0", crashing bf16
     * diffusion under the shim (ROADMAP P4). Local ordinals always resolve on
     * the real driver below. */
    if (local < 1) local = 1;
    if (__builtin_expect(g_debug_attr, 0))
        fprintf(stderr, "[GreenBoost] cuDeviceGetAttribute: attr=%d dev=%d local=%d %s\n",
                attrib, (int)dev, local, ((int)dev >= local) ? "-> feeder" : "-> real");

    if ((int)dev >= local) {
        int ret = gb_netc_device_get_attribute((int)dev - local, attrib, value);
        return (ret == 0) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_DEVICE;
    }

    /* On Blackwell (cc >= 12) desktop PCIe, ggml-cuda uses
     * CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED (193) to
     * decide whether to use its cuMemCreate/cuMemMap VMM pool.  That pool's
     * HOST_NUMA_CURRENT T2 fallback returns DMA-only handles on Blackwell
     * PCIe - any kernel reading from it crashes with invalid_resource_handle.
     * Report VMM=0 so ggml falls back to the cudaMalloc legacy pool, whose
     * T1-overflow path correctly routes through gb_vmm_t2_alloc_blackwell_managed
     * (managed-UVM, SM-accessible).  Override: GREENBOOST_BLACKWELL_ALLOW_VMM=1. */
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED 193
    if (attrib == CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED) {
        CUresult r = real_cuDeviceGetAttribute(value, attrib, dev);
        if (r == CUDA_SUCCESS && *value != 0) {
            int cc = gb_cc_major;
            if (cc == 0 && real_cuDeviceGetAttribute)
                real_cuDeviceGetAttribute(&cc, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
            if (cc >= 12) {
                static int gb_bw_allow_vmm_attr = -1;
                if (__builtin_expect(gb_bw_allow_vmm_attr < 0, 0)) {
                    const char *e = getenv("GREENBOOST_BLACKWELL_ALLOW_VMM");
                    gb_bw_allow_vmm_attr = (e && e[0] != '0') ? 1 : 0;
                    if (!gb_bw_allow_vmm_attr)
                        fprintf(stderr,
                            "[GreenBoost] cuDeviceGetAttribute: Blackwell cc=%d - "
                            "reporting VMM=0 (cuMemCreate HOST paths are DMA-only on "
                            "desktop PCIe; cudaMalloc→managed-UVM used for T2 SM access). "
                            "Override: GREENBOOST_BLACKWELL_ALLOW_VMM=1\n", cc);
                }
                if (!gb_bw_allow_vmm_attr)
                    *value = 0;
            }
        }
        return r;
    }

    return real_cuDeviceGetAttribute(value, attrib, dev);
}

/* cudaDeviceGetAttribute - runtime-API companion to cuDeviceGetAttribute.
 * ggml-cuda 0.23+ calls cudaDeviceGetAttribute@libcudart.so.13 (not the driver
 * API cuDeviceGetAttribute) to read CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_
 * SUPPORTED (attr=193) for the VMM pool selection.  We must hook the runtime
 * path too - same Blackwell VMM=0 override applies. */
cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device)
{
    GB_CUDART_ENSURE();
    pfn_cudaDeviceGetAttribute fn = real_cudaDeviceGetAttribute;
    if (!fn) fn = (pfn_cudaDeviceGetAttribute)dlsym(RTLD_NEXT, "cudaDeviceGetAttribute");
    if (!fn) return (cudaError_t)CUDA_ERROR_NOT_SUPPORTED;

    /* GREENBOOST_GGML_2DEV: route a remote ordinal to the feeder, mirroring
     * cuDeviceGetAttribute (driver API) above. Without this, a remote
     * ordinal (which the real driver has never heard of - it only knows
     * local devices 0..local-1) hits the real cudaDeviceGetAttribute and
     * returns "invalid device ordinal", aborting llama-server's
     * --list-devices GPU discovery entirely (observed live: Ollama then
     * silently falls back to the Vulkan backend, bypassing this shim). */
    if (initialized) {
        int local = 1;
        if (real_cudaGetDeviceCount)
            real_cudaGetDeviceCount(&local);
        /* See cuDeviceGetAttribute: clamp so device 0 never misroutes to the
         * feeder on a zero/failed count (ROADMAP P4 shared-mem=0 crash). */
        if (local < 1) local = 1;
        if (__builtin_expect(g_debug_attr, 0))
            fprintf(stderr, "[GreenBoost] cudaDeviceGetAttribute: attr=%d dev=%d local=%d %s\n",
                    attr, device, local, (device >= local) ? "-> feeder" : "-> real");
        if (device >= local) {
            int ret = gb_netc_device_get_attribute(device - local, attr, value);
            return (ret == 0) ? CUDA_SUCCESS : (cudaError_t)CUDA_ERROR_INVALID_DEVICE;
        }
    }

    cudaError_t ret = fn(value, attr, device);

    if (ret == 0 /* cudaSuccess */ && attr == 193 /* cudaDevAttrVirtualMemoryManagementSupported */
            && value && *value != 0) {
        int cc = gb_cc_major;
        if (cc == 0 && real_cuDeviceGetAttribute)
            real_cuDeviceGetAttribute(&cc, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        if (cc >= 12) {
            static int gb_rt_allow_vmm = -1;
            if (__builtin_expect(gb_rt_allow_vmm < 0, 0)) {
                const char *e = getenv("GREENBOOST_BLACKWELL_ALLOW_VMM");
                gb_rt_allow_vmm = (e && e[0] != '0') ? 1 : 0;
                if (!gb_rt_allow_vmm)
                    fprintf(stderr,
                        "[GreenBoost] cudaDeviceGetAttribute: Blackwell cc=%d - "
                        "reporting VMM=0 (cuMemCreate HOST_NUMA T2 is DMA-only on "
                        "desktop PCIe; cudaMalloc→managed-UVM used instead). "
                        "Override: GREENBOOST_BLACKWELL_ALLOW_VMM=1\n", cc);
            }
            if (!gb_rt_allow_vmm)
                *value = 0;
        }
    }
    return ret;
}

/* ------------------------------------------------------------------ */
/*  cudaGetDeviceProperties hook                                        */
/*                                                                      */
/*  llama.cpp calls this for EVERY device index before deciding which   */
/*  to use.  Without this hook, device 1 (feeder) returns an error and  */
/*  is silently skipped - the feeder GPU is never used.                 */
/*                                                                      */
/*  Strategy: call real driver for device 0 to get a valid, ABI-safe   */
/*  property struct (avoids SDK header layout dependency), then patch   */
/*  the name field at offset 0 with the feeder's GPU name.  Both local  */
/*  and feeder are RTX 5xxx (same CC/props), so device 0 is a valid    */
/*  template for device 1.                                              */
/* ------------------------------------------------------------------ */

cudaError_t cudaGetDeviceProperties_v2(void *prop, int device)
{
    GB_CUDART_ENSURE();
    if (!real_cudaGetDeviceProperties)
        return (cudaError_t)CUDA_ERROR_NOT_SUPPORTED;

    int local = 1;
    if (real_cudaGetDeviceCount)
        real_cudaGetDeviceCount(&local);

    if (device >= local) {
        /* Strategy: call real driver for device 0 to get a valid, ABI-safe   */
        /* property struct, then patch the name and memory fields.            */
        cudaError_t ret = real_cudaGetDeviceProperties(prop, 0);
        if (ret == 0 && prop) {
            strncpy((char *)prop, gb_netc_remote_name(device - local), 255);
            ((char *)prop)[255] = '\0';
            size_t *total = (size_t *)((char *)prop + 288);
            *total = gb_netc_remote_vram(device - local);
        }
        return ret;
    }

    cudaError_t ret = real_cudaGetDeviceProperties(prop, device);
    /* Patch totalGlobalMem for device 0 so PyTorch's
     * torch.cuda.get_device_properties(0).total_memory reports the full virtual
     * pool instead of physical VRAM only.  Offset 288 is stable since CUDA 10.0
     * (sizeof(cudaDeviceProp)==1032, verified against CUDA 12.x headers).
     * Skipped when GREENBOOST_REPORT_PHYSICAL_VRAM=1 (diffusion/PyTorch use case
     * needs the truth - its allocator over-commits otherwise). */
    if (ret == 0 && prop && device == 0 && !g_report_physical_vram) {
        size_t *total = (size_t *)((char *)prop + 288);
        size_t virtual_total = *total + gb_virtual_vram_bytes + g_cluster_remote_total_bytes;
        if (gb_nvlink_aggregated_bytes > 0)
            virtual_total += gb_nvlink_aggregated_bytes;
        gb_log("cudaGetDeviceProperties: patched totalGlobalMem %zuMB -> %zuMB",
               *total >> 20, virtual_total >> 20);
        *total = virtual_total;
    } else if (ret == 0 && prop && device == 0 && g_report_physical_vram) {
        size_t *total = (size_t *)((char *)prop + 288);
        gb_log("cudaGetDeviceProperties: reporting physical totalGlobalMem %zuMB (GREENBOOST_REPORT_PHYSICAL_VRAM=1)",
               *total >> 20);
    }
    return ret;
}

cudaError_t cudaGetDeviceProperties(void *prop, int device)
{
    return cudaGetDeviceProperties_v2(prop, device);
}

/* ------------------------------------------------------------------ */
/*  Cluster device enumeration - cudaGetDeviceCount / cudaSetDevice     */
/*                                                                      */
/*  GreenBoost presents a single virtual GPU (device 0) to all callers */
/*  by default. Remote feeder memory is aggregated into device 0's    */
/*  reported VRAM via cuDeviceTotalMem / cudaGetDeviceProperties hooks */
/*  - apps never need to know about the physical cluster topology.    */
/*  Exposing remote GPUs as device N+1 breaks PyTorch _cuda_init()    */
/*  because the real CUDA driver only knows local devices (ordinal    */
/*  0..local-1), so this stays OFF for torch (GREENBOOST_CLUSTER=0     */
/*  is the torch convention already).                                  */
/*                                                                      */
/*  GREENBOOST_GGML_2DEV=1 flips this off for the ggml/Ollama path     */
/*  only (set in the Ollama systemd drop-in, never by torch): the      */
/*  feeder is enumerated as a real second CUDA device so llama.cpp     */
/*  natively splits an oversized model's layers across local VRAM and  */
/*  feeder VRAM, with the feeder GPU computing its assigned layers.    */
/*  cuDeviceGet/cuDeviceGetAttribute/cuDeviceTotalMem/                 */
/*  cudaGetDeviceProperties_v2/cudaSetDevice already handle ordinal    */
/*  >= local by routing to the feeder - this flag only widens the      */
/*  count hooks below so callers ever iterate that far.                */
/* ------------------------------------------------------------------ */

static int gb_ggml_2dev(void)
{
    static int _2dev = -1;
    if (_2dev < 0) {
        const char *e = getenv("GREENBOOST_GGML_2DEV");
        _2dev = (e && e[0] == '1') ? 1 : 0;
    }
    return _2dev;
}

cudaError_t cudaGetDeviceCount(int *count)
{
    GB_CUDART_ENSURE();
    int local = 1;
    if (real_cudaGetDeviceCount)
        real_cudaGetDeviceCount(&local);
    if (count)
        *count = gb_ggml_2dev() ? (local + gb_netc_remote_gpu_count()) : local;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int *count)
{
    int local = 1;
    if (real_cuDeviceGetCount)
        real_cuDeviceGetCount(&local);
    if (count)
        *count = gb_ggml_2dev() ? (local + gb_netc_remote_gpu_count()) : local;
    return CUDA_SUCCESS;
}

/* cuDeviceGet - translate remote GPU ordinals to identity handles.
 * The real CUDA driver only knows about local GPUs; without this hook
 * any caller that iterates 0..cuDeviceGetCount-1 (e.g. PyTorch _cuda_init)
 * gets CUDA_ERROR_INVALID_DEVICE for remote ordinals, crashing CUDA init.
 * We return ordinal as the CUdevice handle - all other hooks (cuDeviceGetAttribute,
 * cuDeviceTotalMem, cudaGetDeviceProperties) already use ordinal as device ID. */
CUresult cuDeviceGet(CUdevice *device, int ordinal)
{
    int local = 1;
    if (real_cuDeviceGetCount)
        real_cuDeviceGetCount(&local);

    if (ordinal >= local) {
        int remote_idx = ordinal - local;
        if (remote_idx >= gb_netc_remote_gpu_count())
            return CUDA_ERROR_INVALID_DEVICE;
        if (device)
            *device = (CUdevice)ordinal;
        return CUDA_SUCCESS;
    }

    return real_cuDeviceGet ? real_cuDeviceGet(device, ordinal)
                            : CUDA_ERROR_INVALID_DEVICE;
}

static void *gb_get_hook(const char *name); /* forward decl - defined below */

/* cuGetProcAddress - PyTorch ≥2.5 / CUDA 11.3+ uses this instead of dlsym to look
 * up driver API function pointers.  Without this hook our cuDeviceGet/cuDeviceGetCount
 * overrides are bypassed and PyTorch asserts "Can't find cuDeviceGet" on BF16 mode
 * (the first code path that triggers DriverAPI::get() without bitsandbytes pre-init).
 * We call the real cuGetProcAddress, then replace any returned pointer with our own
 * hook if gb_get_hook() knows about that symbol. */
CUresult cuGetProcAddress(const char *symbol, void **pfn, int driverVersion,
                          unsigned long long flags, void *symbolStatus)
{
    CUresult res;
    pfn_cuGetProcAddress fn = real_cuGetProcAddress;
    if (!fn) {
        /* VCM-02: deferred init pending (GREENBOOST_VLLM_COMPAT=1) — resolve via
         * RTLD_NEXT so PyTorch cu130+ gets valid driver function pointers during
         * its own CUDA initialization (which happens before the first cuMemGetInfo
         * or cudaMalloc that triggers gb_try_resume_deferred).  Without this,
         * cuGetProcAddress returns CUDA_ERROR_NOT_SUPPORTED and PyTorch stores
         * NULL for cuLaunchKernel / cuMemAlloc_v2 / etc., causing
         * cudaErrorInvalidResourceHandle on the first kernel launch. */
        fn = (pfn_cuGetProcAddress)dlsym(RTLD_NEXT, "cuGetProcAddress");
        if (!fn) {
            if (pfn) *pfn = NULL;
            return 1; /* CUDA_ERROR_NOT_SUPPORTED */
        }
    }
    res = fn(symbol, pfn, driverVersion, flags, symbolStatus);
    if (res == CUDA_SUCCESS && symbol && pfn && *pfn) {
        void *hook = gb_get_hook(symbol);
        if (hook) *pfn = hook;
    }
    return res;
}

cudaError_t cudaSetDevice(int device)
{
    GB_CUDART_ENSURE();
    if (!initialized)
        return real_cudaSetDevice ? real_cudaSetDevice(device) : CUDA_SUCCESS;

    int local = 1;
    if (real_cudaGetDeviceCount)
        real_cudaGetDeviceCount(&local);

    if (device >= local) {
        gb_netc_set_active_remote(device - local);
        return real_cudaSetDevice ? real_cudaSetDevice(0) : CUDA_SUCCESS;
    } else {
        gb_netc_set_active_remote(-1);
        return real_cudaSetDevice ? real_cudaSetDevice(device) : CUDA_SUCCESS;
    }
}

/* GREENBOOST_GGML_2DEV peer-access hooks.
 *
 * A fake feeder device (ordinal >= local) is NOT real GPU-peer-accessible
 * memory - it lives on the far side of a TCP link, not NVLink/PCIe. Without
 * these hooks, ggml/llama.cpp's multi-GPU init would call the REAL
 * cudaDeviceCanAccessPeer/cudaDeviceEnablePeerAccess against a device
 * ordinal the real CUDA driver has never heard of, and either get a
 * confusing error or (worse) succeed against the wrong local device and
 * later issue a real cudaMemcpyPeer straight at a 0xAA fake pointer.
 * Forcing canAccessPeer=0 for any pair involving a remote ordinal makes
 * ggml fall back to its staged-copy path, which the cudaMemcpy/
 * cudaMemcpyAsync hooks (see the device0<->feeder bounce-buffer staging)
 * already handle correctly. Real local<->local peer access (a box with 2+
 * genuine local GPUs) passes through unchanged - this only touches remote
 * ordinals. */
cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice)
{
    GB_CUDART_ENSURE();
    if (!initialized) {
        if (canAccessPeer) *canAccessPeer = 0;
        return real_cudaDeviceCanAccessPeer
            ? real_cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice) : CUDA_SUCCESS;
    }

    int local = 1;
    if (real_cudaGetDeviceCount)
        real_cudaGetDeviceCount(&local);

    if (device >= local || peerDevice >= local) {
        if (canAccessPeer) *canAccessPeer = 0;
        return CUDA_SUCCESS;
    }
    return real_cudaDeviceCanAccessPeer
        ? real_cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice)
        : (cudaError_t)CUDA_ERROR_NOT_SUPPORTED;
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
    GB_CUDART_ENSURE();
    if (!initialized)
        return real_cudaDeviceEnablePeerAccess
            ? real_cudaDeviceEnablePeerAccess(peerDevice, flags) : CUDA_SUCCESS;

    int local = 1;
    if (real_cudaGetDeviceCount)
        real_cudaGetDeviceCount(&local);

    /* Remote ordinal: no-op success. cudaDeviceCanAccessPeer already told
     * the caller this pair can't peer-access, so a caller that still tries
     * to enable it anyway gets a harmless success rather than a real-CUDA
     * error against a device ordinal the driver never allocated. */
    if (peerDevice >= local)
        return CUDA_SUCCESS;
    return real_cudaDeviceEnablePeerAccess
        ? real_cudaDeviceEnablePeerAccess(peerDevice, flags)
        : (cudaError_t)CUDA_ERROR_NOT_SUPPORTED;
}

/* cudaPointerGetAttributes on a fake feeder pointer must report real device
 * memory on the feeder's ordinal, not fall through to the real CUDA driver
 * (which has never seen this pointer and would return
 * cudaErrorInvalidValue). feeder_idx comes from the same alloc-tracking
 * table gb_netc_free/gb_netc_memcpy_* already use to resolve a fake pointer
 * back to its owning feeder. */
cudaError_t cudaPointerGetAttributes(gb_cudaPointerAttributes *attributes, const void *ptr)
{
    GB_CUDART_ENSURE();
    if (attributes && ptr && gb_is_remote_ptr((uint64_t)(uintptr_t)ptr)) {
        int local = 1;
        if (real_cudaGetDeviceCount)
            real_cudaGetDeviceCount(&local);
        int feeder_idx = -1;
        int found = (gb_netc_get_alloc_info((uint64_t)(uintptr_t)ptr, NULL, NULL, NULL,
                                             &feeder_idx) == 0);
        if (found) gb_netc_alloc_release_ref((uint64_t)(uintptr_t)ptr);
        attributes->type          = cudaMemoryTypeDevice;
        attributes->device        = local + ((found && feeder_idx >= 0) ? feeder_idx : 0);
        attributes->devicePointer = (void *)ptr;
        attributes->hostPointer   = NULL;
        return CUDA_SUCCESS;
    }
    if (!initialized)
        return real_cudaPointerGetAttributes
            ? real_cudaPointerGetAttributes(attributes, ptr) : CUDA_SUCCESS;
    return real_cudaPointerGetAttributes
        ? real_cudaPointerGetAttributes(attributes, ptr)
        : (cudaError_t)CUDA_ERROR_NOT_SUPPORTED;
}

/* ------------------------------------------------------------------ */
/*  PR-S/R3: shared kernel-launch feeder-dispatch helper.               */
/*                                                                      */
/*  All three launch entry points (cuLaunchKernel, cuLaunchKernelEx,    */
/*  cuLaunchCooperativeKernel) scan the kernel argument buffer for      */
/*  fake remote pointers (0xAA00…) and dispatch the launch to the       */
/*  owning feeder via gb_netc_exec_kernel.  This helper centralises     */
/*  that scan so the three hooks remain in sync (the cooperative path   */
/*  had already drifted slightly from cuLaunchKernel before this).      */
/*                                                                      */
/*  Returns:                                                            */
/*    1  = dispatched to feeder; *out_ret holds the dispatch result     */
/*    0  = no dispatch (all args local, or no remote-tracker active);   */
/*         caller falls through to the local launch primitive           */
/*   -1  = arg buffer malformed; *out_ret = CUDA_ERROR_INVALID_VALUE    */
/*                                                                      */
/*  alloca() is intentional: the arg_vals copy lives only for the      */
/*  duration of this helper invocation, and is fully consumed by       */
/*  gb_netc_exec_kernel (which serialises onto the wire) before        */
/*  return.  The caller never references arg_vals after we return.     */
/* ------------------------------------------------------------------ */
/* PR-JJ: per-kernel arg-count override table.
 *
 * When a kernel is launched via the kernelParams[] convention (a NULL-
 * terminated array of `void *` pointing at each arg), our scan
 * traditionally reads 8 bytes per slot until it hits a NULL.  This is
 * correct for all-pointer arg lists but reads past the end of a smaller
 * arg's stack slot when arg widths are mixed (e.g. (Tensor*, int)).
 *
 * Practical impact: low - the over-read only matters if the garbage
 * bytes coincidentally match the 0xAA00… remote-pointer range, which
 * has ~0% probability.  PR-DD's memcpy fix removed the strict-aliasing
 * UB.  This table lets users register an explicit arg count for kernels
 * where they want to guarantee a bounded scan.
 *
 * Populate via gb_kernel_sig_register(host_fn, n_args); leave empty for
 * the default 8-byte-per-slot behaviour.  Lookup is O(1) via open-
 * addressed hash on the host_fn pointer.  Currently NO callers populate
 * the table - the scaffold is in place for a future audit-config-file
 * mechanism.  Until populated, behaviour is identical to pre-PR-JJ. */
#define GB_KSIG_SIZE 512u
#define GB_KSIG_MASK (GB_KSIG_SIZE - 1u)
struct gb_kernel_sig {
    const void *host_fn;  /* NULL = empty slot */
    uint32_t    n_args;
};
static struct gb_kernel_sig g_kernel_sigs[GB_KSIG_SIZE];
static pthread_mutex_t g_kernel_sigs_lock = PTHREAD_MUTEX_INITIALIZER;

static inline uint32_t gb_ksig_hash(const void *fn)
{
    return (uint32_t)(((uint64_t)(uintptr_t)fn * 0x9E3779B97F4A7C15ULL) >> 32);
}

/* Returns n_args (>0) if a signature is registered for host_fn, else 0
 * (caller falls back to default scan-until-NULL behaviour). */
static uint32_t gb_kernel_sig_lookup(const void *host_fn)
{
    if (!host_fn) return 0;
    uint32_t slot = gb_ksig_hash(host_fn) & GB_KSIG_MASK;
    pthread_mutex_lock(&g_kernel_sigs_lock);
    for (uint32_t i = 0; i < GB_KSIG_SIZE; i++) {
        uint32_t s = (slot + i) & GB_KSIG_MASK;
        const void *fn = g_kernel_sigs[s].host_fn;
        if (!fn) break;
        if (fn == host_fn) {
            uint32_t n = g_kernel_sigs[s].n_args;
            pthread_mutex_unlock(&g_kernel_sigs_lock);
            return n;
        }
    }
    pthread_mutex_unlock(&g_kernel_sigs_lock);
    return 0;
}

/* PR-TT: name → arg-count overrides loaded from GREENBOOST_KERNEL_SIGS at
 * shim init.  File format: one entry per line, `kernel_name n_args`,
 * with `#` line comments.  Names match the mangled `deviceName` passed
 * to `__cudaRegisterFunction`.  Looked up inside __cudaRegisterFunction
 * to convert the override into a `host_fn → n_args` registration in the
 * main table.
 *
 * The structure is a flat array of (name, n_args) pairs.  Lookup is
 * linear; expected size is small (per-process kernel set is usually a
 * few dozen entries even for large frameworks). */
#define GB_KSIG_NAME_MAX 256
#define GB_KSIG_NAMES_CAP 4096
struct gb_kernel_sig_name {
    char     name[GB_KSIG_NAME_MAX];
    uint32_t n_args;
};
static struct gb_kernel_sig_name g_kernel_sig_names[GB_KSIG_NAMES_CAP];
static uint32_t g_kernel_sig_names_count = 0;

static void gb_kernel_sigs_load_file(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[GreenBoost] GREENBOOST_KERNEL_SIGS=%s - open failed: %s\n",
                path, strerror(errno));
        return;
    }
    char line[GB_KSIG_NAME_MAX + 64];
    uint32_t loaded = 0;
    while (fgets(line, sizeof(line), f) && g_kernel_sig_names_count < GB_KSIG_NAMES_CAP) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0') continue;
        char *name = p;
        while (*p && *p != ' ' && *p != '\t') p++;
        if (!*p) continue;
        *p++ = '\0';
        while (*p == ' ' || *p == '\t') p++;
        uint32_t n = (uint32_t)strtoul(p, NULL, 10);
        if (n == 0 || n > 128) continue;
        size_t nlen = strlen(name);
        if (nlen == 0 || nlen >= GB_KSIG_NAME_MAX) continue;
        memcpy(g_kernel_sig_names[g_kernel_sig_names_count].name, name, nlen + 1);
        g_kernel_sig_names[g_kernel_sig_names_count].n_args = n;
        g_kernel_sig_names_count++;
        loaded++;
    }
    fclose(f);
    fprintf(stderr, "[GreenBoost] GREENBOOST_KERNEL_SIGS loaded %u entries from %s\n",
            loaded, path);
}

/* Lookup by name.  Called from __cudaRegisterFunction once per kernel
 * registration to fan out the name-keyed config into a host_fn-keyed
 * entry in the main hashmap. */
static uint32_t gb_kernel_sig_lookup_name(const char *name)
{
    if (!name) return 0;
    for (uint32_t i = 0; i < g_kernel_sig_names_count; i++) {
        if (strcmp(g_kernel_sig_names[i].name, name) == 0)
            return g_kernel_sig_names[i].n_args;
    }
    return 0;
}

/* Public-ish registration API.  Currently no in-tree callers; intended
 * for future config-file driven registration or runtime probing via
 * cuFuncGetAttribute. */
__attribute__((unused))
static void gb_kernel_sig_register(const void *host_fn, uint32_t n_args)
{
    if (!host_fn || n_args == 0 || n_args > 128) return;
    uint32_t slot = gb_ksig_hash(host_fn) & GB_KSIG_MASK;
    pthread_mutex_lock(&g_kernel_sigs_lock);
    for (uint32_t i = 0; i < GB_KSIG_SIZE; i++) {
        uint32_t s = (slot + i) & GB_KSIG_MASK;
        if (g_kernel_sigs[s].host_fn == NULL ||
            g_kernel_sigs[s].host_fn == host_fn) {
            g_kernel_sigs[s].host_fn = host_fn;
            g_kernel_sigs[s].n_args  = n_args;
            break;
        }
    }
    pthread_mutex_unlock(&g_kernel_sigs_lock);
}

static int gb_try_kernel_feeder_dispatch(CUfunction f,
                                          unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                                          unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                                          unsigned int sharedMem,
                                          void **kernelParams, void **extra,
                                          const char *hook_tag,
                                          CUresult *out_ret)
{
    if (!initialized || !gb_netc_is_active())
        return 0;

    const char *name = gb_netc_lookup_kernel((const void *)(uintptr_t)f);
    if (!name || (!extra && !kernelParams))
        return 0;

    const void *arg_buf = NULL;
    uint32_t    arg_sz  = 0;

    /* Buffer convention: extra[0]=CU_LAUNCH_PARAM_BUFFER_POINTER (0x01),
     *                    extra[2]=CU_LAUNCH_PARAM_BUFFER_SIZE   (0x02) */
    if (extra) {
        for (int _ei = 0; extra[_ei]; _ei += 2) {
            if (extra[_ei] == (void *)0x01 && extra[_ei+1]) arg_buf = extra[_ei+1];
            if (extra[_ei] == (void *)0x02 && extra[_ei+1]) arg_sz  = (uint32_t)*(size_t *)extra[_ei+1];
        }
    }

    /* Pointer-array convention: kernelParams[i] is a void* to the i-th arg.
     * Build a synthetic flat buffer so the same reloc scanner works.
     *
     * PR-DD warning: the synthetic buffer assumes every arg is exactly
     * 8 bytes wide.  Kernels with int/float/half/struct args have
     * differently-sized slots; this read of 8 bytes off a 4-byte (or
     * smaller) slot is technically OOB on the final arg.  For pointer
     * args (the only ones we relocate to feeders) the slot IS 8 bytes
     * and this is correct.  For mixed-width arg lists, the relocs we
     * extract may be off-by-one if a smaller-than-8 arg precedes a
     * pointer arg.  Mitigation: prefer the extra[] buffer convention
     * (which carries an explicit size), and tell users to invoke
     * cuLaunchKernel via that path for heterogeneous arg signatures.
     *
     * A full fix would consult a per-kernel signature table populated
     * by __cudaRegisterFunction - substantial follow-up work tracked
     * in the next audit round. */
    uint64_t _kp_buf[128];
    if (!arg_buf && kernelParams) {
        /* PR-JJ: prefer the registered signature's exact arg count when
         * available - bounds the scan tightly so we never read past the
         * last slot.  Without a registration, fall back to scan-until-
         * NULL (current behaviour) which is correct for all-pointer args
         * but reads up to 7 bytes of trailing stack padding for the
         * final smaller-than-8-byte arg.  See the gb_kernel_sig table
         * commentary above for the rationale.
         *
         * PR-NN: if no registration is found AND cuFuncGetParamInfo is
         * available (CUDA 12.3+), probe the kernel by iterating
         * paramIndex 0..N until CUDA_ERROR_INVALID_VALUE.  Register the
         * count so future launches of this CUfunction are O(1). */
        uint32_t sig_n_args = gb_kernel_sig_lookup((const void *)(uintptr_t)f);
        if (sig_n_args == 0 && real_cuFuncGetParamInfo) {
            size_t off = 0, sz = 0;
            uint32_t n = 0;
            while (n < 128 &&
                   real_cuFuncGetParamInfo(f, (size_t)n, &off, &sz) == CUDA_SUCCESS) {
                n++;
            }
            if (n > 0) {
                gb_kernel_sig_register((const void *)(uintptr_t)f, n);
                sig_n_args = n;
            }
        }
        uint32_t _kp_n = 0;
        uint32_t kp_limit = sig_n_args ? sig_n_args : 128;
        if (kp_limit > 128) kp_limit = 128;
        for (; _kp_n < kp_limit; _kp_n++) {
            /* PR-RR audit #4: always NULL-guard the deref.  CUDA graph
             * capture and other corner cases may pass a shorter
             * kernelParams array than the registered sig length; reading
             * past would SIGSEGV.  Cheap to check; never wrong. */
            if (!kernelParams[_kp_n]) break;
            /* Use memcpy to avoid strict-aliasing UB and to make the
             * 8-byte read explicit. */
            memcpy(&_kp_buf[_kp_n], kernelParams[_kp_n], sizeof(uint64_t));
        }
        if (_kp_n > 0) {
            arg_buf = _kp_buf;
            arg_sz  = (uint32_t)(_kp_n * sizeof(uint64_t));
        }
    }

    if (arg_sz == 0 || !arg_buf)
        return 0;

    if (arg_sz > 4096) {
        fprintf(stderr, "[GreenBoost] %s: invalid arg_sz=%u\n", hook_tag, arg_sz);
        *out_ret = CUDA_ERROR_INVALID_VALUE;
        return -1;
    }

    uint32_t n_vals = arg_sz / (uint32_t)sizeof(uint64_t);
    uint64_t *arg_vals = (uint64_t *)alloca(arg_sz);
    memcpy(arg_vals, arg_buf, arg_sz);

    struct gb_exec_reloc relocs[64];
    int n_relocs = 0;
    int dispatch_feeder = -1;
    for (uint32_t _i = 0; _i < n_vals && n_relocs < 64; _i++) {
        if (gb_is_remote_ptr(arg_vals[_i])) {
            uint64_t rh = 0; int fi = -1;
            if (gb_netc_get_alloc_info(arg_vals[_i], &rh, NULL, NULL, &fi) == 0) {
                relocs[n_relocs].arg_idx       = (int)_i;
                relocs[n_relocs].remote_handle = rh;
                relocs[n_relocs].feeder_idx    = fi;
                n_relocs++;
                if (dispatch_feeder < 0) dispatch_feeder = fi;
            }
        }
    }

    if (dispatch_feeder < 0) {
        /* All args local - release any refs we may have taken on
         * get_alloc_info successes that we won't use. */
        for (int _ri = 0; _ri < n_relocs; _ri++)
            gb_netc_alloc_release_ref(arg_vals[relocs[_ri].arg_idx]);
        return 0;
    }

    gb_log("%s: data-driven dispatch '%s' → feeder %d (%d remote args)%s",
           hook_tag, name, dispatch_feeder, n_relocs,
           g_async_dispatch ? " [async]" : "");
    GB_NVTX_EVENT("KERNEL_FEEDER", "T1_FEEDER", 0, 0, name ? name : "unknown");

    int rc;
    if (g_async_dispatch) {
        /* Phase 2b: fire-and-forget — n_uploads=0 and n_downloads=0 are always
         * true here (the call site passes NULL/0/0 for those args). */
        rc = gb_netc_exec_kernel_async(dispatch_feeder, name,
                                       gridX, gridY, gridZ,
                                       blockX, blockY, blockZ,
                                       sharedMem,
                                       arg_vals, n_vals,
                                       relocs, n_relocs);
    } else {
        rc = gb_netc_exec_kernel(dispatch_feeder, name,
                                 gridX, gridY, gridZ,
                                 blockX, blockY, blockZ,
                                 sharedMem,
                                 arg_vals, n_vals,
                                 relocs, n_relocs,
                                 NULL, 0, 0);
    }
    /* F-L3-12: release the alloc refs taken in gb_netc_get_alloc_info. */
    for (int _ri = 0; _ri < n_relocs; _ri++)
        gb_netc_alloc_release_ref(arg_vals[relocs[_ri].arg_idx]);

    if (rc == 0) atomic_fetch_add_explicit(&g_kernel_dispatch_count, 1,
                                           memory_order_relaxed);
    *out_ret = rc == 0 ? CUDA_SUCCESS : (CUresult)700;
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Cluster kernel dispatch - cuLaunchKernel                           */
/*                                                                      */
/*  cuLaunchKernel routes launches to the active remote feeder.        */
/* ------------------------------------------------------------------ */


CUresult cuLaunchKernel(CUfunction f,
                        unsigned gridX, unsigned gridY, unsigned gridZ,
                        unsigned blockX, unsigned blockY, unsigned blockZ,
                        unsigned sharedMem, CUstream stream,
                        void **kernelParams, void **extra)
{
    /* PR-S/R3: shared scan helper.  See gb_try_kernel_feeder_dispatch above
     * for the dispatch semantics; this hook just forwards arg-buffer +
     * geometry into the helper and falls through to the local launcher
     * when the helper returns 0 (no remote args). */
    CUresult dispatch_ret = CUDA_SUCCESS;
    int rc = gb_try_kernel_feeder_dispatch(f, gridX, gridY, gridZ,
                                            blockX, blockY, blockZ, sharedMem,
                                            kernelParams, extra,
                                            "cuLaunchKernel", &dispatch_ret);
    if (rc == 1) return dispatch_ret;
    if (rc == -1) return dispatch_ret;  /* malformed args */

    if (!real_cuLaunchKernel)
        real_cuLaunchKernel = (pfn_cuLaunchKernel)dlsym(RTLD_NEXT, "cuLaunchKernel");
    if (!real_cuLaunchKernel)
        return (CUresult)CUDA_ERROR_NOT_SUPPORTED;
    return real_cuLaunchKernel(f, gridX, gridY, gridZ, blockX, blockY, blockZ,
                               sharedMem, stream, kernelParams, extra);
}

/* ------------------------------------------------------------------ */
/*  cuLaunchCooperativeKernel - F-L1-33                                */
/*                                                                      */
/*  Cooperative kernels use the same arg conventions as cuLaunchKernel  */
/*  (kernelParams only; no extra[] buffer).  Data-driven feeder         */
/*  dispatch applies here too.                                          */
/* ------------------------------------------------------------------ */
CUresult cuLaunchCooperativeKernel(CUfunction f,
                                   unsigned gridX, unsigned gridY, unsigned gridZ,
                                   unsigned blockX, unsigned blockY, unsigned blockZ,
                                   unsigned sharedMem, CUstream stream,
                                   void **kernelParams)
{
    /* PR-S/R3: shared scan helper.  Cooperative kernels have the same
     * arg conventions as cuLaunchKernel (kernelParams only; no extra[]
     * buffer) so we just pass kernelParams + NULL extra to the helper. */
    CUresult dispatch_ret = CUDA_SUCCESS;
    int rc = gb_try_kernel_feeder_dispatch(f, gridX, gridY, gridZ,
                                            blockX, blockY, blockZ, sharedMem,
                                            kernelParams, NULL,
                                            "cuLaunchCooperativeKernel",
                                            &dispatch_ret);
    if (rc == 1) return dispatch_ret;
    if (rc == -1) return dispatch_ret;

    if (!real_cuLaunchCooperativeKernel)
        real_cuLaunchCooperativeKernel = (pfn_cuLaunchCooperativeKernel)
            dlsym(RTLD_NEXT, "cuLaunchCooperativeKernel");
    if (!real_cuLaunchCooperativeKernel)
        return (CUresult)CUDA_ERROR_NOT_SUPPORTED;
    return real_cuLaunchCooperativeKernel(f, gridX, gridY, gridZ, blockX, blockY, blockZ,
                                          sharedMem, stream, kernelParams);
}

/* ------------------------------------------------------------------ */
/*  cuLaunchKernelEx - F-S8 / PR-O                                      */
/*                                                                      */
/*  CUDA 12+ extended launch API used by PyTorch 2.4+ graph mode and    */
/*  any JIT-compiled kernel with launch attributes (priority,           */
/*  cooperative-launch flag, cluster geometry, programmatic event).     */
/*  Grid / block dims and stream are inside `config`; attrs[] is        */
/*  opaque to us and is forwarded unchanged on the local path.          */
/*                                                                      */
/*  Data-driven feeder dispatch logic mirrors cuLaunchKernel - when     */
/*  any kernel arg is a remote fake pointer (0xAA00…), forward the      */
/*  entire launch to the feeder that owns the data.  We DO drop the     */
/*  attrs[] on the feeder path because the feeder uses cuLaunchKernel   */
/*  (not Ex) for execution - losing launch attributes there is the      */
/*  same trade-off the cooperative-kernel feeder dispatch already makes.*/
/* ------------------------------------------------------------------ */
CUresult cuLaunchKernelEx(const gb_CUlaunchConfig *config, CUfunction f,
                          void **kernelParams, void **extra)
{
    if (!config) return CUDA_ERROR_INVALID_VALUE;

    /* PR-S/R3: shared scan helper.  Geometry comes from config; the
     * helper otherwise reuses the same arg-buffer scan, reloc table,
     * and feeder dispatch path as cuLaunchKernel. */
    CUresult dispatch_ret = CUDA_SUCCESS;
    int rc = gb_try_kernel_feeder_dispatch(f,
                                            config->gridDimX, config->gridDimY, config->gridDimZ,
                                            config->blockDimX, config->blockDimY, config->blockDimZ,
                                            config->sharedMemBytes,
                                            kernelParams, extra,
                                            "cuLaunchKernelEx", &dispatch_ret);
    if (rc == 1) return dispatch_ret;
    if (rc == -1) return dispatch_ret;

    if (!real_cuLaunchKernelEx)
        real_cuLaunchKernelEx = (pfn_cuLaunchKernelEx)dlsym(RTLD_NEXT, "cuLaunchKernelEx");
    if (!real_cuLaunchKernelEx)
        return (CUresult)CUDA_ERROR_NOT_SUPPORTED;
    return real_cuLaunchKernelEx(config, f, kernelParams, extra);
}

/* ------------------------------------------------------------------ */
/*  NVML overrides - nvmlDeviceGetMemoryInfo[_v2]                      */
/*                                                                      */
/*  Ollama's discover/nvidia.go uses NVML for initial GPU sizing.       */
/*  Inflate total + free so the layer scheduler sees virtual VRAM.      */
/* ------------------------------------------------------------------ */

/* GREENBOOST_GGML_2DEV: a fake NVML handle (returned by our
 * nvmlDeviceGetHandleByIndex hook for the feeder) is a sentinel value, NOT a
 * real NVML opaque pointer - passing it into the real NVML call would
 * dereference garbage. Report the feeder's OWN real free/total instead of
 * folding it into device 0, so Ollama's per-device GPU discovery sizes the
 * feeder's layer split correctly. Live network round-trip; called once per
 * device at discovery time, not on the decode hot path. */
static int gb_nvml_fake_mem_info(nvmlDevice_t device, unsigned long long *out_total,
                                  unsigned long long *out_free, unsigned long long *out_used)
{
    if (!gb_is_fake_nvml(device))
        return 0;
    int ri = gb_fake_nvml_idx(device);
    uint64_t _f = 0, _t = 0;
    if (gb_netc_mem_info(ri, &_f, &_t) != 0 || _t == 0)
        return 0;
    if (out_total) *out_total = (unsigned long long)_t;
    if (out_free)  *out_free  = (unsigned long long)_f;
    if (out_used)  *out_used  = (unsigned long long)(_t - _f);
    return 1;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory)
{
    if (memory && gb_nvml_fake_mem_info(device, &memory->total, &memory->free, &memory->used))
        return NVML_SUCCESS;
    if (!real_nvmlDeviceGetMemoryInfo)
        return NVML_ERROR_FUNCTION_NOT_FOUND;

    nvmlReturn_t ret = real_nvmlDeviceGetMemoryInfo(device, memory);
    if (ret == NVML_SUCCESS && memory) {
        /* Use T2-only pool (not T2+T3) for NVML total so Ollama's vram-based
         * default_num_ctx is sized to what can actually hold the KV cache in
         * fast memory.  T3 NVMe is included in cuDeviceTotalMem for layer
         * scheduling (models load fully onto "GPU"), but not here - inflating
         * the NVML total causes 262144-token contexts that exhaust host DDR. */
        size_t nvml_virt_total = (gb_t2_pool_bytes > 0) ? gb_t2_pool_bytes : gb_virtual_vram_bytes;
        memory->total += nvml_virt_total + g_cluster_remote_t1t2_total_bytes;
        memory->free  += gb_t2_free_to_report() + g_cluster_remote_t1t2_free_bytes;
        gb_log("nvmlDeviceGetMemoryInfo: total=%lluMB (T1+T2 only, excl T3 NVMe for ctx sizing)",
               (unsigned long long)memory->total >> 20);
    }
    return ret;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory)
{
    if (memory && gb_nvml_fake_mem_info(device, &memory->total, &memory->free, &memory->used)) {
        memory->version = 0;
        memory->reserved = 0;
        return NVML_SUCCESS;
    }
    if (!real_nvmlDeviceGetMemoryInfo_v2)
        return NVML_ERROR_FUNCTION_NOT_FOUND;

    nvmlReturn_t ret = real_nvmlDeviceGetMemoryInfo_v2(device, memory);
    if (ret == NVML_SUCCESS && memory) {
        size_t nvml_virt_total = (gb_t2_pool_bytes > 0) ? gb_t2_pool_bytes : gb_virtual_vram_bytes;
        memory->total += nvml_virt_total + g_cluster_remote_t1t2_total_bytes;
        memory->free  += gb_t2_free_to_report() + g_cluster_remote_t1t2_free_bytes;
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
    if (memory && gb_nvml_fake_mem_info(device, &memory->total, &memory->free, &memory->used)) {
        memory->reserved = 0;
        memory->exportedToOtherProcess = 0;
        return NVML_SUCCESS;
    }
    if (!real_nvmlDeviceGetMemoryInfo_v3)
        return NVML_ERROR_FUNCTION_NOT_FOUND;

    nvmlReturn_t ret = real_nvmlDeviceGetMemoryInfo_v3(device, memory);
    if (ret == NVML_SUCCESS && memory) {
        size_t nvml_virt_total = (gb_t2_pool_bytes > 0) ? gb_t2_pool_bytes : gb_virtual_vram_bytes;
        memory->total += nvml_virt_total + g_cluster_remote_t1t2_total_bytes;
        memory->free  += gb_t2_free_to_report() + g_cluster_remote_t1t2_free_bytes;
    }
    return ret;
}

/* Single-virtual-GPU (default): NVML still reports the real physical device
 * count. Feeder memory is aggregated into device 0's reported size, not as
 * extra devices. Under GREENBOOST_GGML_2DEV (ggml/Ollama only, see the
 * cudaGetDeviceCount comment above) the feeder is counted as an extra NVML
 * device too, using the same fake-handle range nvmlDeviceGetName already
 * serves. */

nvmlReturn_t nvmlDeviceGetCount(unsigned int *count)
{
    if (!real_nvmlDeviceGetCount) return NVML_ERROR_FUNCTION_NOT_FOUND;
    nvmlReturn_t ret = real_nvmlDeviceGetCount(count);
    if (ret == NVML_SUCCESS && count && gb_ggml_2dev())
        *count += (unsigned int)gb_netc_remote_gpu_count();
    return ret;
}

nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device)
{
    if (gb_ggml_2dev()) {
        unsigned int local = 0;
        if (real_nvmlDeviceGetCount && real_nvmlDeviceGetCount(&local) == NVML_SUCCESS
            && index >= local) {
            int remote_idx = (int)(index - local);
            if (remote_idx >= gb_netc_remote_gpu_count())
                return NVML_ERROR_INVALID_ARGUMENT;
            if (device)
                *device = (nvmlDevice_t)(GB_FAKE_NVML_BASE + (uintptr_t)remote_idx);
            return NVML_SUCCESS;
        }
    }
    if (!real_nvmlDeviceGetHandleByIndex) return NVML_ERROR_FUNCTION_NOT_FOUND;
    return real_nvmlDeviceGetHandleByIndex(index, device);
}

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device)
{
    return nvmlDeviceGetHandleByIndex(index, device);
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length)
{
    if (gb_is_fake_nvml(device) && name && length > 0) {
        int ri = gb_fake_nvml_idx(device);
        strncpy(name, gb_netc_remote_name(ri), length - 1);
        name[length - 1] = '\0';
        return NVML_SUCCESS;
    }
    if (!real_nvmlDeviceGetName) return NVML_ERROR_FUNCTION_NOT_FOUND;
    return real_nvmlDeviceGetName(device, name, length);
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
/*  is called the buffer is already resident - cleanup is synchronous.  */
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

    /* PR-M/F-S5: bvmm_ht entries (Blackwell managed UVM / VMM / zerocopy)
     * are inserted by gb_smart_overflow_alloc which the cuMemAllocAsync
     * overflow path may return.  Without this dispatch the matching
     * cuMemFreeAsync leaks the bvmm_ht entry, gb_t2_overflow_bytes drifts
     * up by `sz` per cycle, and a later alloc that reuses the same VA
     * would see a stale entry → wrong-type cleanup on next free. */
    /* RULE #1 front-load split VAs are tracked in gb_fl_ht, not ht/bvmm_ht. */
    if (gb_frontload_free_dispatch(dptr))
        return CUDA_SUCCESS;

    if (gb_bvmm_free_dispatch(dptr))
        return CUDA_SUCCESS;

    {
        uint32_t flags = ht_peek_flags(dptr);
        if (ht_remove(dptr, &sz, &managed, &mapped_ptr, &fd, &ext_mem)) {
            gb_release_kv_t1_bytes(flags, sz, mapped_ptr, ext_mem, managed);
            gb_log("cuMemFreeAsync ptr=0x%llx size=%zu MB managed=%d mapped_ptr=%p fd=%d ext_mem=%p",
                   (unsigned long long)dptr, sz >> 20, managed, mapped_ptr, fd, (void *)ext_mem);
            /* Path A zero-copy: CUDA owns the mapping - destroy the external memory handle. */
            if (ext_mem) {
                if (real_cudaDestroyExternalMemory)
                    real_cudaDestroyExternalMemory(ext_mem);
                return CUDA_SUCCESS;
            }
            /* Path A / Path B: host-registered memory - unregister and unmap. */
            if (mapped_ptr) {
                if (real_cuMemHostUnregister) real_cuMemHostUnregister(mapped_ptr);
                munmap(mapped_ptr, sz);
                /* Decrement cumulative T2 counter - mirrors the increment in gb_overflow_alloc(). */
                atomic_fetch_sub_explicit(&gb_t2_overflow_bytes, sz, memory_order_relaxed);
            }
            if (fd >= 0)
                close(fd);
            /* DMA-BUF / HostReg allocs: dptr came from cuMemHostGetDevicePointer,
             * not cuMemAlloc - must not pass to cuMemFreeAsync.
             * managed=1 is dead code now that Path C / UVM is removed. */
            if (!mapped_ptr && real_cuMemFreeAsync) {
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
/*  dlsym hook - intercepts dlopen-based GPU API lookups               */
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream);
cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                              size_t spitch, size_t width, size_t height,
                              int kind, cudaStream_t stream);
cudaError_t cudaLaunchKernelExC(const gb_cudaLaunchConfig *config,
                                const void *func, void **args);
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
    /* PID-1 guard: same rationale as gb_shim_init — never run inside init. */
    if (getpid() == 1) return;

    /* dlsym - try newest version first */
    real_dlsym_fn = (pfn_dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
    if (!real_dlsym_fn)
        real_dlsym_fn = (pfn_dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    if (!real_dlsym_fn)
        real_dlsym_fn = (pfn_dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.17");
    if (!real_dlsym_fn)
        real_dlsym_fn = (pfn_dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.0");

    /* MIN-07: Warn if all dlsym version attempts fail - means glibc is older
     * than 2.0 (impossible in practice) or the runtime is musl/uclibc. */
    if (!real_dlsym_fn)
        fprintf(stderr,
            "[GreenBoost] FATAL: dlvsym failed for all known glibc versions - "
            "dlsym hook unavailable; Ollama GPU API interception will not work.\n");

    /* dlopen - same version chain */
    real_dlopen_fn = (pfn_dlopen_t)dlvsym(RTLD_NEXT, "dlopen", "GLIBC_2.34");
    if (!real_dlopen_fn)
        real_dlopen_fn = (pfn_dlopen_t)dlvsym(RTLD_NEXT, "dlopen", "GLIBC_2.2.5");
    if (!real_dlopen_fn)
        real_dlopen_fn = (pfn_dlopen_t)dlvsym(RTLD_NEXT, "dlopen", "GLIBC_2.17");
    if (!real_dlopen_fn)
        real_dlopen_fn = (pfn_dlopen_t)dlvsym(RTLD_NEXT, "dlopen", "GLIBC_2.0");

    if (!real_dlopen_fn)
        fprintf(stderr,
            "[GreenBoost] FATAL: dlvsym failed for all known glibc versions - "
            "dlopen hook unavailable; RTLD_DEEPBIND stripping will not work.\n");
}

/* Forward declarations for hooks defined later in this file.
 * Must NOT be static: a static declaration gives internal linkage to the
 * entire definition, causing LTO to treat these as TU-private and not export
 * them - they end up *UND* in the .so despite the version script listing them
 * as global.  The libcudart.so.12 trampolines in greenboost_cuda_v12.c then
 * resolve to the real libcudart instead of our hooks → hang on fake T2 ptrs. */
cudaError_t cudaMemcpy(void *, const void *, size_t, int);
cudaError_t cudaMemcpyAsync(void *, const void *, size_t, int, cudaStream_t);
cudaError_t cudaMemcpyPeer(void *, int, const void *, int, size_t);
cudaError_t cudaMemcpyPeerAsync(void *, int, const void *, int, size_t, cudaStream_t);
cudaError_t cudaLaunchKernel(const void *, gb_dim3, gb_dim3, void **, size_t, cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamBeginCapture(cudaStream_t, int);
cudaError_t cudaStreamCreate(cudaStream_t *, unsigned int);
void __cudaRegisterFunction(void **, const char *, char *, const char *, int,
                            gb_uint3 *, gb_uint3 *, gb_dim3 *, gb_dim3 *, int *);

/* Return our hook for a given symbol name, or NULL if not intercepted. */
static void *gb_get_hook(const char *name)
{
    if (!name) return NULL;

    /* NVML memory reporting - used by Ollama for initial VRAM discovery */
    if (strcmp(name, "nvmlDeviceGetMemoryInfo")    == 0) return (void *)nvmlDeviceGetMemoryInfo;
    if (strcmp(name, "nvmlDeviceGetMemoryInfo_v2") == 0) return (void *)nvmlDeviceGetMemoryInfo_v2;
    if (strcmp(name, "nvmlDeviceGetMemoryInfo_v3") == 0) return (void *)nvmlDeviceGetMemoryInfo_v3;

    /* CUDA device total memory - queried by scheduler at startup */
    if (strcmp(name, "cuDeviceTotalMem_v2")        == 0) return (void *)cuDeviceTotalMem_v2;
    if (strcmp(name, "cuDeviceTotalMem")           == 0) return (void *)cuDeviceTotalMem;

    /* CUDA free/total memory info */
    if (strcmp(name, "cuMemGetInfo_v2")            == 0) return (void *)cuMemGetInfo_v2;
    if (strcmp(name, "cuMemGetInfo")               == 0) return (void *)cuMemGetInfo;
    if (strcmp(name, "cudaMemGetInfo")             == 0) return (void *)cudaMemGetInfo;

    /* CUDA allocation - large allocs redirected to system RAM pool */
    if (strcmp(name, "cudaMalloc")                 == 0) return (void *)cudaMalloc;
    if (strcmp(name, "cudaMallocAsync")            == 0) return (void *)cudaMallocAsync;
    if (strcmp(name, "cudaFreeAsync")              == 0) return (void *)cudaFreeAsync;
    if (strcmp(name, "cudaFree")                   == 0) return (void *)cudaFree;
    if (strcmp(name, "cuMemAlloc_v2")              == 0) return (void *)cuMemAlloc_v2;
    if (strcmp(name, "cuMemAllocAsync")            == 0) return (void *)cuMemAllocAsync;
    if (strcmp(name, "cuMemAllocFromPoolAsync")    == 0) return (void *)cuMemAllocFromPoolAsync;  /* PR-N */
    if (strcmp(name, "cuLaunchKernelEx")           == 0) return (void *)cuLaunchKernelEx;        /* PR-O */
    if (strcmp(name, "cuLaunchCooperativeKernel")  == 0) return (void *)cuLaunchCooperativeKernel;  /* PR-CC */
    /* PR-R/F-S9: cuGetProcAddress callers may resolve the per-thread
     * default-stream variants by name.  Route them through the same
     * hooks (these are also exported as ELF aliases, so direct dlsym
     * works too, but cuGetProcAddress takes a different code path). */
    if (strcmp(name, "cuMemAllocAsync_ptds")          == 0) return (void *)cuMemAllocAsync;
    if (strcmp(name, "cuMemAllocFromPoolAsync_ptds")  == 0) return (void *)cuMemAllocFromPoolAsync;
    if (strcmp(name, "cuMemFreeAsync_ptds")           == 0) return (void *)cuMemFreeAsync;
    if (strcmp(name, "cuMemPrefetchAsync_ptds")       == 0) return (void *)cuMemPrefetchAsync;
    if (strcmp(name, "cuLaunchKernel_ptds")           == 0) return (void *)cuLaunchKernel;
    if (strcmp(name, "cuLaunchKernelEx_ptds")         == 0) return (void *)cuLaunchKernelEx;
    if (strcmp(name, "cuLaunchCooperativeKernel_ptds")== 0) return (void *)cuLaunchCooperativeKernel;
    if (strcmp(name, "cudaMemcpyAsync_ptsz")          == 0) return (void *)cudaMemcpyAsync;
    if (strcmp(name, "cudaMemPrefetchAsync_ptsz")     == 0) return (void *)cudaMemPrefetchAsync;
    if (strcmp(name, "cudaLaunchKernel_ptsz")         == 0) return (void *)cudaLaunchKernel;  /* PR-LL */
    if (strcmp(name, "cuMemFree_v2")               == 0) return (void *)cuMemFree_v2;
    if (strcmp(name, "cuMemFreeAsync")             == 0) return (void *)cuMemFreeAsync;
    if (strcmp(name, "cuMemPrefetchAsync")         == 0) return (void *)cuMemPrefetchAsync;
    if (strcmp(name, "cudaMemPrefetchAsync")       == 0) return (void *)cudaMemPrefetchAsync;

    /* VMM companion hooks - cuMemCreate is already listed above */
    if (strcmp(name, "cuMemRelease")               == 0) return (void *)cuMemRelease;
    if (strcmp(name, "cuMemAddressReserve")        == 0) return (void *)cuMemAddressReserve;
    if (strcmp(name, "cuMemMap")                   == 0) return (void *)cuMemMap;

    /* Cluster device enumeration and routing */
    if (strcmp(name, "cudaGetDeviceCount")            == 0) return (void *)cudaGetDeviceCount;
    if (strcmp(name, "cuDeviceGetCount")              == 0) return (void *)cuDeviceGetCount;
    if (strcmp(name, "cuDeviceGet")                   == 0) return (void *)cuDeviceGet;
    if (strcmp(name, "cuGetProcAddress")              == 0) return (void *)cuGetProcAddress;
    if (strcmp(name, "cudaSetDevice")                 == 0) return (void *)cudaSetDevice;
    if (strcmp(name, "cuLaunchKernel")                == 0) return (void *)cuLaunchKernel;
    if (strcmp(name, "cudaLaunchKernel")              == 0) return (void *)cudaLaunchKernel;
    if (strcmp(name, "cudaStreamSynchronize")         == 0) return (void *)cudaStreamSynchronize;
    if (strcmp(name, "cudaStreamBeginCapture")        == 0) return (void *)cudaStreamBeginCapture;
    if (strcmp(name, "cudaStreamCreate")              == 0) return (void *)cudaStreamCreate;
    if (strcmp(name, "__cudaRegisterFunction")        == 0) return (void *)__cudaRegisterFunction;
    if (strcmp(name, "cudaMemcpy")                    == 0) return (void *)cudaMemcpy;
    if (strcmp(name, "cudaMemcpyAsync")               == 0) return (void *)cudaMemcpyAsync;
    if (strcmp(name, "cudaMemcpyPeer")                == 0) return (void *)cudaMemcpyPeer;
    if (strcmp(name, "cudaMemcpyPeerAsync")           == 0) return (void *)cudaMemcpyPeerAsync;
    if (strcmp(name, "cudaMemset")                    == 0) return (void *)cudaMemset;
    if (strcmp(name, "cudaMemsetAsync")               == 0) return (void *)cudaMemsetAsync;
    if (strcmp(name, "cudaMemsetAsync_ptsz")          == 0) return (void *)cudaMemsetAsync;
    if (strcmp(name, "cudaMemcpy2DAsync")             == 0) return (void *)cudaMemcpy2DAsync;
    if (strcmp(name, "cudaMemcpy2DAsync_ptsz")        == 0) return (void *)cudaMemcpy2DAsync;
    if (strcmp(name, "cudaLaunchKernelExC")           == 0) return (void *)cudaLaunchKernelExC;
    if (strcmp(name, "cudaLaunchKernelExC_ptsz")      == 0) return (void *)cudaLaunchKernelExC;
    if (strcmp(name, "cudaGetDeviceProperties")       == 0) return (void *)cudaGetDeviceProperties;
    if (strcmp(name, "cudaGetDeviceProperties_v2")    == 0) return (void *)cudaGetDeviceProperties_v2;

    if (strcmp(name, "cuDeviceGetAttribute")          == 0) return (void *)cuDeviceGetAttribute;
    if (strcmp(name, "cudaDeviceGetAttribute")        == 0) return (void *)cudaDeviceGetAttribute;

    /* GREENBOOST_GGML_2DEV peer-access + pointer-attribute hooks */
    if (strcmp(name, "cudaDeviceCanAccessPeer")       == 0) return (void *)cudaDeviceCanAccessPeer;
    if (strcmp(name, "cudaDeviceEnablePeerAccess")    == 0) return (void *)cudaDeviceEnablePeerAccess;
    if (strcmp(name, "cudaPointerGetAttributes")      == 0) return (void *)cudaPointerGetAttributes;

    /* Single-virtual-GPU (default): NVML device count/handle pass through to
     * real NVML, feeder memory aggregated into device 0. Under
     * GREENBOOST_GGML_2DEV the feeder is counted/handled as an extra NVML
     * device instead - see nvmlDeviceGetCount/nvmlDeviceGetHandleByIndex. */

    return NULL;
}

/* NVML-only hook table for gaming mode (no CUDA initialized). */
static void *gb_get_nvml_hook(const char *name)
{
    if (!name) return NULL;
    if (strcmp(name, "nvmlDeviceGetMemoryInfo")       == 0) return (void *)nvmlDeviceGetMemoryInfo;
    if (strcmp(name, "nvmlDeviceGetMemoryInfo_v2")    == 0) return (void *)nvmlDeviceGetMemoryInfo_v2;
    if (strcmp(name, "nvmlDeviceGetMemoryInfo_v3")    == 0) return (void *)nvmlDeviceGetMemoryInfo_v3;
    return NULL;
}

void *dlsym(void *handle, const char *name)
{
    void *hook;

    /* Only intercept after GreenBoost has fully initialized, and ONLY for
     * library-specific handles (not RTLD_NEXT).  Our own code uses RTLD_NEXT
     * to find real implementations - intercepting those causes infinite
     * recursion (real_cudaMalloc = dlsym(RTLD_NEXT,"cudaMalloc") → our hook
     * → calls itself). */
    if (initialized && handle != RTLD_NEXT) {
        hook = gb_get_hook(name);
        if (hook) {
            gb_log("dlsym hook: '%s' → GreenBoost intercepted", name);
            return hook;
        }
    }

    if (real_dlsym_fn)
        return real_dlsym_fn(handle, name);

    /* Bootstrap failed - return NULL rather than calling the broken
     * dlvsym(handle,name,"GLIBC_2.0") which returns NULL for all CUDA/NVML
     * symbols anyway and would silently break the caller's initialization. */
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  dlopen hook - strips RTLD_DEEPBIND so LD_PRELOAD hooks stay active */
/*                                                                      */
/*  Ollama loads libggml-cuda.so with RTLD_DEEPBIND, which makes CUDA  */
/*  symbol lookups inside that library prefer libcudart.so (bundled),  */
/*  completely bypassing our cudaMalloc/cuMemAlloc LD_PRELOAD hooks.   */
/*  Stripping RTLD_DEEPBIND forces those symbols to resolve from the   */
/*  global namespace where our overrides are registered first.          */
/* ------------------------------------------------------------------ */

void *dlopen(const char *filename, int flags)
{
    /* RTLD_DEEPBIND = 0x008 on Linux - isolates library from global NS.
     * Only strip it when GreenBoost is fully initialized (confirmed CUDA/Ollama
     * process).  Some apps rely on RTLD_DEEPBIND to isolate PE DLLs;
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

    /* Fallback: bootstrap hasn't run yet - should never happen since
     * constructor(101) runs before any user code calls dlopen. */
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  sysinfo() hook - inflate system RAM by virtual VRAM pool size       */
/*                                                                      */
/*  Ollama's Go runtime calls syscall.Sysinfo() to check total system   */
/*  RAM before loading a model.  When physical RAM < model size,        */
/*  Ollama rejects the request with "model requires more system memory"  */
/*  before any CUDA allocation is attempted - our NVML/CUDA hooks never  */
/*  get a chance to run.                                                 */
/*                                                                      */
/*  By inflating totalram and freeram by gb_virtual_vram_bytes we make  */
/*  Ollama see the full virtual capacity (T1 + T2) as available memory, */

/*  hooks already report.  Only active when the shim is initialized and  */
/*  a virtual pool is configured - inert in all other processes.        */
/* ------------------------------------------------------------------ */

int sysinfo(struct sysinfo *info)
{
    static int (*real_sysinfo)(struct sysinfo *) = NULL;
    if (!real_sysinfo)
        real_sysinfo = (int (*)(struct sysinfo *))dlsym(RTLD_NEXT, "sysinfo");

    if (!real_sysinfo)
        return -1;

    int ret = real_sysinfo(info);
    if (ret == 0 && initialized && info->mem_unit > 0) {
        unsigned long extra = 0;
        if (gb_virtual_vram_bytes > 0)
            extra += (unsigned long)(gb_virtual_vram_bytes / info->mem_unit);
        /* Add remote feeder T2 so Ollama's system-RAM pre-flight check
         * accounts for combined cluster capacity, not just local RAM. */
        if (gb_netc_is_active()) {
            uint64_t rt2_free = 0, rt2_total = 0;
            if (gb_netc_total_remote_t2_bytes(&rt2_free, &rt2_total) == 0 && rt2_free > 0)
                extra += (unsigned long)(rt2_free / info->mem_unit);
        }
        if (extra > 0) {
            info->totalram += extra;
            info->freeram  += extra;
            if (gb_debug)
                fprintf(stderr,
                        "[GreenBoost] sysinfo: inflated +%lu MB (local virtual: %zu MB, remote T2: via netc)\n",
                        (unsigned long)((unsigned long long)extra * info->mem_unit >> 20),
                        gb_virtual_vram_bytes >> 20);
        }
    }
    return ret;
}

/* cudaMemcpyKind enum values - stable ABI since CUDA 1.0 */
typedef enum {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4,
} gb_cudaMemcpyKind;

/* ------------------------------------------------------------------ */
/*  GREENBOOST_GGML_2DEV: device0<->feeder cross-device staging        */
/*                                                                      */
/*  llama.cpp's per-layer activation handoff between local VRAM         */
/*  (device 0) and the feeder (device 1) copies a REAL local device     */
/*  pointer to/from a fake feeder pointer. gb_netc_memcpy_h2d/d2h treat  */
/*  their "local" side as plain host-visible memory (a raw pointer read  */
/*  or write) - handing them a genuine VRAM address either segfaults    */
/*  (VRAM is not mapped into the process's normal address space) or,    */
/*  on platforms where it happens not to fault, silently reads/writes   */
/*  the wrong bytes. Stage through a bounce buffer instead: a real      */
/*  cudaMemcpy moves data between VRAM and the bounce buffer, then the  */
/*  existing fabric call moves it between the bounce buffer and the     */
/*  feeder - exactly like a normal H2D/D2H copy from the fabric's point  */
/*  of view.                                                             */
/*                                                                      */
/*  gb_is_local_device_ptr distinguishes a genuine local device pointer  */
/*  (T1 VRAM, or a Path-A cudaImportExternalMemory mapping - both are    */
/*  real device address space) from a GreenBoost-managed HOST-visible    */
/*  pointer (Path-B mmap+cuMemHostRegister zero-copy, or plain malloc'd  */
/*  host memory): every ht_insert call site for the latter sets          */
/*  mapped_ptr to the actual host address, while local-device call sites */
/*  always pass mapped_ptr=NULL, fd=-1 (see the ht_insert call sites     */
/*  throughout this file). Host-visible pointers must NOT be staged -    */
/*  they already work directly with gb_netc_memcpy_h2d/d2h (that is the  */
/*  validated, pre-existing path; staging them would just add a         */
/*  redundant copy).                                                    */
/* ------------------------------------------------------------------ */

static int gb_is_local_device_ptr(const void *ptr)
{
    size_t sz; int managed; void *mapped_ptr; int fd;
    if (!ht_lookup((CUdeviceptr)(uintptr_t)ptr, &sz, &managed, &mapped_ptr, &fd))
        return 0;
    return (!managed && mapped_ptr == NULL && fd < 0);
}

/* Thread-local bounce buffer, grown on demand and kept for reuse across
 * calls on the same thread - avoids a fresh allocation on every per-layer
 * activation handoff. Plain heap memory (not cudaHostAlloc-pinned): pinning
 * would speed up the D2H/H2D leg slightly but adds another CUDA resource to
 * resolve and manage for a copy that is small (KB-low MB activations) and
 * infrequent (once per layer boundary per token), not the dominant cost. */
static __thread void  *g_stage_bounce      = NULL;
static __thread size_t g_stage_bounce_cap  = 0;

static void *gb_stage_bounce_get(size_t need)
{
    if (g_stage_bounce_cap >= need)
        return g_stage_bounce;
    void *p = realloc(g_stage_bounce, need);
    if (!p)
        return NULL;
    g_stage_bounce     = p;
    g_stage_bounce_cap = need;
    return p;
}

/* dst is a fake feeder pointer, src is a confirmed real local device
 * pointer. Returns 0 on success, -1 on failure. */
static int gb_staged_memcpy_device_to_remote(uint64_t dstp, const void *src, size_t count)
{
    void *bounce = gb_stage_bounce_get(count);
    if (!bounce) return -1;
    cudaError_t (*real_fn)(void *, const void *, size_t, int) = real_cudaMemcpy;
    if (!real_fn) return -1;
    if (real_fn(bounce, src, count, 2 /* cudaMemcpyDeviceToHost */) != 0)
        return -1;
    return gb_netc_memcpy_h2d(dstp, bounce, (uint64_t)count);
}

/* src is a fake feeder pointer, dst is a confirmed real local device
 * pointer. Returns 0 on success, -1 on failure. */
static int gb_staged_memcpy_remote_to_device(void *dst, uint64_t srcp, size_t count)
{
    void *bounce = gb_stage_bounce_get(count);
    if (!bounce) return -1;
    if (gb_netc_memcpy_d2h(bounce, srcp, (uint64_t)count) != 0)
        return -1;
    cudaError_t (*real_fn)(void *, const void *, size_t, int) = real_cudaMemcpy;
    if (!real_fn) return -1;
    return real_fn(dst, bounce, count, 1 /* cudaMemcpyHostToDevice */) == 0 ? 0 : -1;
}

/* Shared routing decision for cudaMemcpy / cudaMemcpyAsync / cudaMemcpyPeer /
 * cudaMemcpyPeerAsync: all four reduce to the same "is either side a fake
 * feeder pointer" classification - cudaMemcpyPeer's explicit device
 * arguments are redundant with that, since our cudaMalloc/cuMemAlloc_v2
 * active-remote branch already returns a fake pointer for any allocation
 * made while the current device was set to a feeder ordinal.
 *
 * Returns 1 if this call fully handled the copy (*out_ret holds the
 * cudaError_t to return); 0 if neither side is a remote pointer (caller
 * falls through to the real, local copy primitive). */
static int gb_route_cross_device_memcpy(void *dst, const void *src, size_t count,
                                         cudaError_t *out_ret)
{
    uint64_t dstp = (uint64_t)(uintptr_t)dst;
    uint64_t srcp = (uint64_t)(uintptr_t)src;
    int dst_remote = gb_is_remote_ptr(dstp);
    int src_remote = gb_is_remote_ptr(srcp);
    if (!dst_remote && !src_remote)
        return 0;

    int rc;
    if (dst_remote && !src_remote && gb_is_local_device_ptr(src)) {
        rc = gb_staged_memcpy_device_to_remote(dstp, src, count);
    } else if (src_remote && !dst_remote && gb_is_local_device_ptr(dst)) {
        rc = gb_staged_memcpy_remote_to_device(dst, srcp, count);
    } else if (dst_remote && !src_remote) {
        GB_NVTX_PUSH("GB:net_h2d", GB_NVTX_COLOR_NET);
        rc = gb_netc_memcpy_h2d(dstp, src, (uint64_t)count);
        GB_NVTX_POP();
        atomic_fetch_add_explicit(&g_h2d_mb, count >> 20, memory_order_relaxed);
    } else if (src_remote && !dst_remote) {
        GB_NVTX_PUSH("GB:net_d2h", GB_NVTX_COLOR_NET);
        rc = gb_netc_memcpy_d2h(dst, srcp, (uint64_t)count);
        GB_NVTX_POP();
        atomic_fetch_add_explicit(&g_d2h_mb, count >> 20, memory_order_relaxed);
    } else if (dst_remote) {
        /* both remote (or dst_remote with the src-local-device check above
         * already false) - existing feeder<->feeder path, unsupported in
         * default (non-NCCL) builds; unchanged from prior behavior. */
        rc = gb_netc_memcpy_d2d_push(dstp, src, (uint64_t)count);
    } else {
        rc = gb_netc_memcpy_d2d_pull(dst, srcp, (uint64_t)count);
    }
    *out_ret = (rc == 0) ? CUDA_SUCCESS : (cudaError_t)1; /* cudaErrorInvalidValue */
    return 1;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind)
{
    GB_CUDART_ENSURE();
    /* F-ABI1 fix: use the robustly-resolved global (gb_cudart_resolve_syms),
     * not a local dlsym(RTLD_NEXT,...) - see the declaration comment above
     * real_cudaMemcpy for why the old per-function dlsym silently dropped
     * copies. RTLD_NEXT fallback kept only for the case GB_CUDART_ENSURE()
     * hasn't run yet (real_cudaMemcpy still NULL pre-first-rebind). */
    cudaError_t (*real_fn)(void *, const void *, size_t, int) = real_cudaMemcpy;
    if (!real_fn)
        real_fn = (cudaError_t (*)(void *, const void *, size_t, int))
            dlsym(RTLD_NEXT, "cudaMemcpy");

    if (!initialized || !real_fn)
        return real_fn ? real_fn(dst, src, count, kind) : 0;

    /* Route operations involving remote cluster pointers through netc
     * (including the device0<->feeder staged path - see
     * gb_route_cross_device_memcpy above). */
    {
        cudaError_t remote_ret;
        if (gb_route_cross_device_memcpy(dst, src, count, &remote_ret))
            return remote_ret;
    }

    /* U12: check if dst or src was previously evicted and is being re-accessed */
    {
        uint64_t dstp = (uint64_t)(uintptr_t)dst;
        uint64_t srcp = (uint64_t)(uintptr_t)src;
        gb_refault_check(dstp);
        gb_refault_check(srcp);

        /* NVTX: emit events when tensor data flows through T2 DDR (PCIe path).
         * ht_lookup detects whether dst/src is a GreenBoost-pinned T2 pointer.
         * Only fires for transfers > 1 MB to avoid noise from tiny copies. */
        if (count >= (1 << 20)) {
            size_t _sz; int _mgd; void *_mp; int _fd;
            if (kind == 1 /* cudaMemcpyHostToDevice */ &&
                    ht_lookup((CUdeviceptr)dstp, &_sz, &_mgd, &_mp, &_fd)) {
                GB_NVTX_EVENT("MEMCPY_H2D_T2", "T2_DDR", count >> 20, dstp, "local_t2_h2d");
            } else if (kind == 2 /* cudaMemcpyDeviceToHost */ &&
                    ht_lookup((CUdeviceptr)srcp, &_sz, &_mgd, &_mp, &_fd)) {
                GB_NVTX_EVENT("MEMCPY_D2H_T2", "T2_DDR", count >> 20, srcp, "local_t2_d2h");
            }
        }
    }

    return real_fn(dst, src, count, kind);
}

/* cudaMemcpy2DAsync — ggml uses strided copies for some tensor layouts.
 * Remote fake pointers are served row-by-row over the fabric (load-time
 * only, correctness over speed); local pointers pass through to the
 * rebound cudart instance. */
cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                              size_t spitch, size_t width, size_t height,
                              int kind, cudaStream_t stream)
{
    GB_CUDART_ENSURE();
    cudaError_t (*real_fn)(void *, size_t, const void *, size_t,
                           size_t, size_t, int, cudaStream_t) = real_cudaMemcpy2DAsync;
    if (!real_fn)
        real_fn = (cudaError_t (*)(void *, size_t, const void *, size_t,
                                   size_t, size_t, int, cudaStream_t))
            dlsym(RTLD_NEXT, "cudaMemcpy2DAsync");

    if (initialized) {
        uint64_t dstp = (uint64_t)(uintptr_t)dst;
        uint64_t srcp = (uint64_t)(uintptr_t)src;
        int dst_remote = gb_is_remote_ptr(dstp);
        int src_remote = gb_is_remote_ptr(srcp);
        if (dst_remote || src_remote) {
            if (dst_remote && src_remote) return (cudaError_t)1; /* unsupported */
            for (size_t row = 0; row < height; row++) {
                int rc;
                if (dst_remote)
                    rc = gb_netc_memcpy_h2d(dstp + row * dpitch,
                                            (const uint8_t *)src + row * spitch,
                                            (uint64_t)width);
                else
                    rc = gb_netc_memcpy_d2h((uint8_t *)dst + row * dpitch,
                                            srcp + row * spitch,
                                            (uint64_t)width);
                if (rc != 0) return (cudaError_t)1;
            }
            return CUDA_SUCCESS;
        }
    }
    return real_fn ? real_fn(dst, dpitch, src, spitch, width, height, kind, stream)
                   : (cudaError_t)1;
}

/* cudaMemset/cudaMemsetAsync — required for feeder-resident buffers: ggml
 * memsets the padding of every quantized tensor right after uploading it
 * (ggml_backend_cuda_buffer init).  Unhooked, the fake 0xAA pointer reaches
 * real CUDA and aborts llama-server with cudaErrorInvalidValue (hit live
 * 2026-07-06 on the first feeder-T1 model buffer). */
cudaError_t cudaMemset(void *devPtr, int value, size_t count)
{
    GB_CUDART_ENSURE();
    /* F-ABI1: use the REBOUND cudart instance (real_cudaMemset via
     * gb_cudart_sym), NOT dlsym(RTLD_NEXT) — RTLD_NEXT resolves the system
     * cudart while ollama's cuda_v13 runtime is a different instance, and a
     * valid local pointer fed to the wrong instance returns invalid argument
     * (crashed llama-server at 13:55 on a purely LOCAL model load). */
    cudaError_t (*real_fn)(void *, int, size_t) = real_cudaMemset;
    if (!real_fn)
        real_fn = (cudaError_t (*)(void *, int, size_t))
            dlsym(RTLD_NEXT, "cudaMemset");

    if (initialized && gb_is_remote_ptr((uint64_t)(uintptr_t)devPtr)) {
        int rc = gb_netc_memset((uint64_t)(uintptr_t)devPtr, value, (uint64_t)count);
        return rc == 0 ? CUDA_SUCCESS : (cudaError_t)1;
    }
    return real_fn ? real_fn(devPtr, value, count) : (cudaError_t)1;
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream)
{
    GB_CUDART_ENSURE();
    /* F-ABI1: same rebound-instance rule as cudaMemset above. */
    cudaError_t (*real_fn)(void *, int, size_t, cudaStream_t) = real_cudaMemsetAsync;
    if (!real_fn)
        real_fn = (cudaError_t (*)(void *, int, size_t, cudaStream_t))
            dlsym(RTLD_NEXT, "cudaMemsetAsync");

    if (initialized && gb_is_remote_ptr((uint64_t)(uintptr_t)devPtr)) {
        /* Synchronous over the fabric — the TCP round-trip already orders it
         * against every other message on this feeder's socket. */
        int rc = gb_netc_memset((uint64_t)(uintptr_t)devPtr, value, (uint64_t)count);
        return rc == 0 ? CUDA_SUCCESS : (cudaError_t)1;
    }
    return real_fn ? real_fn(devPtr, value, count, stream) : (cudaError_t)1;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                             int kind, cudaStream_t stream)
{
    GB_CUDART_ENSURE();
    /* F-ABI1 fix - see cudaMemcpy above. */
    cudaError_t (*real_fn)(void *, const void *, size_t, int, void *) = real_cudaMemcpyAsync;
    if (!real_fn)
        real_fn = (cudaError_t (*)(void *, const void *, size_t, int, void *))
            dlsym(RTLD_NEXT, "cudaMemcpyAsync");

    if (!initialized || !real_fn)
        return real_fn ? real_fn(dst, src, count, kind, stream) : 0;

    /* Route operations involving remote cluster pointers through netc.
     * Network transfers are inherently synchronous - stream ordering is
     * preserved by the barrier that cudaStreamSynchronize adds after dispatch. */
    {
        uint64_t dstp = (uint64_t)(uintptr_t)dst;
        uint64_t srcp = (uint64_t)(uintptr_t)src;
        int dst_remote = gb_is_remote_ptr(dstp);
        int src_remote = gb_is_remote_ptr(srcp);
        if (dst_remote || src_remote) {
            int rc;
            /* Device0<->feeder staging (see gb_route_cross_device_memcpy /
             * the comment above cudaMemcpy) - checked first so the 2-device
             * Ollama split's per-layer activation handoff never reaches the
             * host-memory-assuming h2d/d2h calls below with a real VRAM
             * pointer. */
            if (dst_remote && !src_remote && gb_is_local_device_ptr(src)) {
                GB_NVTX_PUSH("GB:net_h2d", GB_NVTX_COLOR_NET);
                GB_NVTX_EVENT("MEMCPY_H2D_NET_STAGED", "T1_FEEDER", count >> 20, dstp,
                               "cudaMemcpyAsync_h2d_feeder_staged");
                rc = gb_staged_memcpy_device_to_remote(dstp, src, count);
                GB_NVTX_POP();
                atomic_fetch_add_explicit(&g_h2d_mb, count >> 20, memory_order_relaxed);
            } else if (src_remote && !dst_remote && gb_is_local_device_ptr(dst)) {
                GB_NVTX_PUSH("GB:net_d2h", GB_NVTX_COLOR_NET);
                GB_NVTX_EVENT("MEMCPY_D2H_NET_STAGED", "T1_FEEDER", count >> 20, srcp,
                               "cudaMemcpyAsync_d2h_feeder_staged");
                rc = gb_staged_memcpy_remote_to_device(dst, srcp, count);
                GB_NVTX_POP();
                atomic_fetch_add_explicit(&g_d2h_mb, count >> 20, memory_order_relaxed);
            } else if (dst_remote && !src_remote) {
                GB_NVTX_PUSH("GB:net_h2d", GB_NVTX_COLOR_NET);
                GB_NVTX_EVENT("MEMCPY_H2D_NET", "T1_FEEDER", count >> 20, dstp, "cudaMemcpyAsync_h2d_feeder");
                rc = gb_netc_memcpy_h2d(dstp, src, (uint64_t)count);
                GB_NVTX_POP();
                atomic_fetch_add_explicit(&g_h2d_mb, count >> 20, memory_order_relaxed);
            } else if (src_remote && !dst_remote) {
                GB_NVTX_PUSH("GB:net_d2h", GB_NVTX_COLOR_NET);
                GB_NVTX_EVENT("MEMCPY_D2H_NET", "T1_FEEDER", count >> 20, srcp, "cudaMemcpyAsync_d2h_feeder");
                rc = gb_netc_memcpy_d2h(dst, srcp, (uint64_t)count);
                GB_NVTX_POP();
                atomic_fetch_add_explicit(&g_d2h_mb, count >> 20, memory_order_relaxed);
            } else if (dst_remote) {
                rc = gb_netc_memcpy_d2d_push(dstp, src, (uint64_t)count);
            } else {
                rc = gb_netc_memcpy_d2d_pull(dst, srcp, (uint64_t)count);
            }
            return rc == 0 ? CUDA_SUCCESS : (cudaError_t)1;
        }
    }

    return real_fn(dst, src, count, kind, stream);
}

/* cudaMemcpyPeer / cudaMemcpyPeerAsync - GREENBOOST_GGML_2DEV cross-device
 * copy. ggml's multi-GPU backend may use these (rather than plain
 * cudaMemcpy) for its device0<->device1 activation handoff. The explicit
 * dstDevice/srcDevice arguments are redundant with pointer classification
 * here (see gb_route_cross_device_memcpy) - any allocation made while the
 * active device was a feeder ordinal already got a fake 0xAA pointer from
 * cudaMalloc/cuMemAlloc_v2, regardless of what device index a later memcpy
 * call names. When neither side is remote (a real multi-local-GPU box),
 * fall through to the real driver function unchanged. */
cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice,
                           size_t count)
{
    GB_CUDART_ENSURE();
    cudaError_t (*real_fn)(void *, int, const void *, int, size_t) = real_cudaMemcpyPeer;
    if (!real_fn)
        real_fn = (cudaError_t (*)(void *, int, const void *, int, size_t))
            dlsym(RTLD_NEXT, "cudaMemcpyPeer");

    if (initialized) {
        cudaError_t remote_ret;
        if (gb_route_cross_device_memcpy(dst, src, count, &remote_ret))
            return remote_ret;
    }
    return real_fn ? real_fn(dst, dstDevice, src, srcDevice, count) : (cudaError_t)1;
}

cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice,
                                size_t count, cudaStream_t stream)
{
    GB_CUDART_ENSURE();
    cudaError_t (*real_fn)(void *, int, const void *, int, size_t, void *) = real_cudaMemcpyPeerAsync;
    if (!real_fn)
        real_fn = (cudaError_t (*)(void *, int, const void *, int, size_t, void *))
            dlsym(RTLD_NEXT, "cudaMemcpyPeerAsync");

    if (initialized) {
        cudaError_t remote_ret;
        /* Network transfers are inherently synchronous - same rationale as
         * cudaMemcpyAsync above; stream ordering is preserved by the
         * barrier cudaStreamSynchronize adds after dispatch. */
        if (gb_route_cross_device_memcpy(dst, src, count, &remote_ret))
            return remote_ret;
    }
    return real_fn ? real_fn(dst, dstDevice, src, srcDevice, count, stream) : (cudaError_t)1;
}

/* ------------------------------------------------------------------ */
/*  cudaStreamBeginCapture — CUDA graph capture entry point             */
/*                                                                      */
/*  ggml/llama.cpp (USE_GRAPHS=1) calls this once per decode batch to   */
/*  start a CUDA graph capture pass.                                    */
/* ------------------------------------------------------------------ */
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, int mode)
{
    GB_CUDART_ENSURE();
    if (!real_cudaStreamBeginCapture)
        real_cudaStreamBeginCapture = (pfn_cudaStreamBeginCapture)
            dlsym(RTLD_NEXT, "cudaStreamBeginCapture");
    if (!real_cudaStreamBeginCapture) return (cudaError_t)1;
    return real_cudaStreamBeginCapture(stream, mode);
}

/* ------------------------------------------------------------------ */
/*  cudaLaunchKernel - runtime API equivalent of cuLaunchKernel        */
/*                                                                      */
/*  PyTorch uses cudaLaunchKernel (not cuLaunchKernel). Data-driven    */
/*  dispatch: scan args[] for remote fake pointers and route the kernel */
/*  to the feeder that owns the allocation.  Falls through to local     */
/*  GPU when all args are local.                                         */
/*                                                                      */
/*  arg scanning: args[i] is a void* pointing to the actual arg value.  */
/*  We dereference as uint64_t and check the remote-ptr sentinel range.  */
/*  Max 32 args scanned to avoid reading past valid stack memory.        */
/* ------------------------------------------------------------------ */
cudaError_t cudaLaunchKernel(const void *func, gb_dim3 gridDim, gb_dim3 blockDim,
                              void **args, size_t sharedMem, cudaStream_t stream)
{
    GB_CUDART_ENSURE();
    /* Feeder dispatch in feeder-exclusive mode OR when this thread's active
     * device is the feeder under GREENBOOST_GGML_2DEV (see the widened gate
     * comment on cudaLaunchKernelExC below). */
    if (initialized && gb_netc_is_active() &&
        (gb_feeder_exclusive() || gb_netc_get_active_remote() >= 0) &&
        func && args) {
        const char *kname = gb_netc_lookup_kernel(func);
        if (kname) {
            struct gb_exec_reloc relocs[32];
            uint64_t arg_vals[32];
            int n_relocs = 0;
            int n_vals   = 0;
            int dispatch_feeder = -1;

            for (int i = 0; i < 32 && args[i]; i++) {
                uint64_t v = *(const uint64_t *)args[i];
                arg_vals[n_vals++] = v;
                if (gb_is_remote_ptr(v) && n_relocs < 32) {
                    uint64_t rh = 0; int fi = -1;
                    if (gb_netc_get_alloc_info(v, &rh, NULL, NULL, &fi) == 0) {
                        relocs[n_relocs].arg_idx       = i;
                        relocs[n_relocs].remote_handle = rh;
                        relocs[n_relocs].feeder_idx    = fi;
                        n_relocs++;
                        if (dispatch_feeder < 0) dispatch_feeder = fi;
                    }
                }
            }

            if (dispatch_feeder >= 0) {
                gb_log("cudaLaunchKernel: data-driven dispatch '%s' → feeder %d (%d remote args)",
                       kname, dispatch_feeder, n_relocs);
                int rc = gb_netc_exec_kernel(dispatch_feeder, kname,
                                             gridDim.x, gridDim.y, gridDim.z,
                                             blockDim.x, blockDim.y, blockDim.z,
                                             (uint32_t)sharedMem,
                                             arg_vals, (uint32_t)n_vals,
                                             relocs, n_relocs,
                                             NULL, 0, 0);
                /* F-L3-12: release alloc refs held since gb_netc_get_alloc_info */
                for (int _ri = 0; _ri < n_relocs; _ri++)
                    gb_netc_alloc_release_ref(arg_vals[relocs[_ri].arg_idx]);
                if (rc == 0) atomic_fetch_add_explicit(&g_kernel_dispatch_count, 1, memory_order_relaxed);
                return rc == 0 ? CUDA_SUCCESS : (cudaError_t)700;
            }
            /* All args local - release refs acquired during reloc scan */
            for (int _ri = 0; _ri < n_relocs; _ri++)
                gb_netc_alloc_release_ref(arg_vals[relocs[_ri].arg_idx]);
        }
    }

    if (!real_cudaLaunchKernel)
        real_cudaLaunchKernel = (pfn_cudaLaunchKernel)dlsym(RTLD_NEXT, "cudaLaunchKernel");
    if (!real_cudaLaunchKernel) return (cudaError_t)1;
    return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

/* cudaLaunchKernelExC — cuda_v13 ggml's launch entry point (PR-CLKExC take 2,
 * see the typedef comment for why the 2026-06-24 attempt failed).  Same
 * data-driven feeder dispatch as cudaLaunchKernel; attrs are forwarded
 * untouched on the local path and dropped on the remote path (the feeder
 * executes on its own stream). */
/* Query a runtime-API kernel's real parameter count + sizes.
 * cudaLaunchKernelExC's args[] has NO NULL-terminator guarantee (unlike the
 * de-facto layout ggml uses for plain cudaLaunchKernel) — scanning past the
 * real count dereferences stack garbage and SIGSEGVs (hit live 2026-07-06,
 * first warmup launch in feeder-exclusive mode).  cudaGetKernel (12.1+) maps
 * the host stub to a CUkernel; cuKernelGetParamInfo enumerates params. */
static int gb_kernel_param_sizes(const void *func, uint32_t sizes[64])
{
    static int (*p_paraminfo)(void *, size_t, size_t *, size_t *) = NULL;
    static int probed = 0;
    if (!probed) {
        probed = 1;
        void *lc = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (lc) p_paraminfo = (int (*)(void *, size_t, size_t *, size_t *))
            dlsym(lc, "cuKernelGetParamInfo");
    }
    /* F-ABI1: MUST be the rebound cudart instance — the kernel stubs are
     * registered with ggml's cuda_v13 runtime; the system cudart's
     * cudaGetKernel does not know them (hit live: -1 param count made the
     * launch fall through LOCALLY with remote fake-ptr args → sticky
     * illegal-memory-access reported at the next cudaFuncGetAttributes). */
    cudaError_t (*p_getk)(void **, const void *) = real_cudaGetKernel;
    if (!p_getk)
        p_getk = (cudaError_t (*)(void **, const void *))
            dlsym(RTLD_NEXT, "cudaGetKernel");
    if (!p_getk || !p_paraminfo) return -1;
    void *k = NULL;
    if (p_getk(&k, func) != 0 || !k) return -1;
    int n = 0;
    for (; n < 64; n++) {
        size_t off = 0, sz = 0;
        if (p_paraminfo(k, (size_t)n, &off, &sz) != 0) break;
        sizes[n] = (uint32_t)sz;   /* full size — struct-by-value params can be >8B */
    }
    return n;
}

cudaError_t cudaLaunchKernelExC(const gb_cudaLaunchConfig *config,
                                const void *func, void **args)
{
    GB_CUDART_ENSURE();
    cudaError_t (*real_fn)(const gb_cudaLaunchConfig *, const void *, void **)
        = real_cudaLaunchKernelExC;
    if (!real_fn)
        real_fn = (pfn_cudaLaunchKernelExC)dlsym(RTLD_NEXT, "cudaLaunchKernelExC");

    /* Feeder kernel DISPATCH only runs where a kernel's args are actually
     * feeder-resident — either feeder-exclusive mode (the whole working set
     * is feeder-side), or GREENBOOST_GGML_2DEV with the CURRENT THREAD's
     * active device set to the feeder (gb_netc_get_active_remote() >= 0):
     * under the 2-device Ollama split, device 1's layer weights, staged
     * activations, and KV are ALL feeder-resident by construction (see the
     * cudaMemcpy staging above), so this launch's args are exactly as safe
     * to pack/scan as feeder-exclusive's. In the default single-vGPU path a
     * large model's buffers sit in LOCAL T2 (feeder T1 is skipped when the
     * buffer exceeds feeder VRAM), so every arg is local: running the
     * param-size query + packing on every launch there added cost and a
     * crash risk on the proven-working path (regression: SIGSEGV during
     * ggml warmup after 2026-07-06 Full Install).  Gate it. */
    if (initialized && gb_netc_is_active() &&
        (gb_feeder_exclusive() || gb_netc_get_active_remote() >= 0) &&
        config && func && args) {
        const char *kname = gb_netc_lookup_kernel(func);
        uint32_t psizes[64];
        int n_params = kname ? gb_kernel_param_sizes(func, psizes) : -1;
        /* RAW path: need real param sizes (struct-by-value args exceed 8B and
         * carry embedded pointers).  Without cuKernelGetParamInfo we cannot
         * safely pack the buffer, so fall through to local launch. */
        if (kname && n_params > 0 && n_params <= 64) {
            /* Pack params into an 8-byte-aligned buffer; scan every 8-byte
             * slot (top-level AND inside structs) for remote fake pointers. */
            static __thread uint8_t  param_buf[8192];
            uint32_t param_off[64];
            struct gb_exec_reloc_raw relocs[128];
            int n_relocs = 0, dispatch_feeder = -1;
            uint32_t off = 0;
            int packed_ok = 1;

            for (int i = 0; i < n_params; i++) {
                uint32_t sz = psizes[i] ? psizes[i] : 8;
                uint32_t asz = (sz + 7u) & ~7u;               /* 8-byte align */
                if (off + asz > sizeof(param_buf)) { packed_ok = 0; break; }
                param_off[i] = off;
                memcpy(param_buf + off, args[i], sz);
                if (asz > sz) memset(param_buf + off + sz, 0, asz - sz);
                /* scan this param's 8-byte slots for remote fake ptrs */
                for (uint32_t b = 0; b + 8 <= sz && n_relocs < 128; b += 8) {
                    uint64_t v;
                    memcpy(&v, param_buf + off + b, 8);
                    if (!gb_is_remote_ptr(v)) continue;
                    uint64_t rh = 0; int fi = -1;
                    if (gb_netc_get_alloc_info(v, &rh, NULL, NULL, &fi) == 0) {
                        relocs[n_relocs].buf_offset    = off + b;
                        relocs[n_relocs].remote_handle = rh;
                        relocs[n_relocs].feeder_idx    = fi;
                        n_relocs++;
                        if (dispatch_feeder < 0) dispatch_feeder = fi;
                    }
                }
                off += asz;
            }

            if (packed_ok && dispatch_feeder >= 0) {
                gb_log("cudaLaunchKernelExC: RAW dispatch '%s' → feeder %d "
                       "(%d params, %u buf bytes, %d remote ptrs incl. struct-embedded)",
                       kname, dispatch_feeder, n_params, off, n_relocs);
                int rc = gb_netc_exec_kernel_raw(dispatch_feeder, kname,
                                                 config->gridDim.x, config->gridDim.y, config->gridDim.z,
                                                 config->blockDim.x, config->blockDim.y, config->blockDim.z,
                                                 (uint32_t)config->dynamicSmemBytes,
                                                 param_buf, off, psizes, (uint32_t)n_params,
                                                 relocs, n_relocs);
                for (int _ri = 0; _ri < n_relocs; _ri++) {
                    uint64_t v;
                    memcpy(&v, param_buf + relocs[_ri].buf_offset, 8);
                    gb_netc_alloc_release_ref(v);
                }
                if (rc == 0) atomic_fetch_add_explicit(&g_kernel_dispatch_count, 1, memory_order_relaxed);
                return rc == 0 ? CUDA_SUCCESS : (cudaError_t)700;
            }
            for (int _ri = 0; _ri < n_relocs; _ri++) {
                uint64_t v;
                memcpy(&v, param_buf + relocs[_ri].buf_offset, 8);
                gb_netc_alloc_release_ref(v);
            }
        }
    }

    return real_fn ? real_fn(config, func, args) : (cudaError_t)1;
}

/* ------------------------------------------------------------------ */
/*  cudaStreamSynchronize - sync local + remote streams                 */
/*                                                                      */
/*  After data-driven remote kernel dispatch (sync or async), this hook */
/*  sends GB_MSG_CUDA_SYNC to feeders that dispatched work since the    */
/*  last sync.  The feeder drains its async_stream (cudaStreamSync) and */
/*  its device (cudaDeviceSync) before responding, so all queued async  */
/*  kernels are complete before this call returns to the caller.        */
/* ------------------------------------------------------------------ */
cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    GB_CUDART_ENSURE();
    /* F-L1-31: only contact feeders that dispatched a kernel since the last
     * sync; feeder execution is synchronous so completed kernels need no
     * extra round-trip, avoiding unnecessary latency for idle feeders. */
    if (initialized && gb_netc_is_active())
        gb_netc_selective_stream_sync();

    /* Keep shim_stats fresh during decode. Every prior gb_maybe_write_stats()
     * call site is on the alloc/free path, which goes quiet once a model has
     * finished loading - vitals then reads a >30s-stale file and reports
     * "shim stats unavailable"/"shim not active" even though inference is
     * actively running. cudaStreamSynchronize fires once per decode step
     * (ggml/llama.cpp syncs after each batch) regardless of whether a
     * feeder is involved, unconditionally on the local-only path too - the
     * one hot-path call site that reliably refreshes the timestamp for
     * every inference session, not just feeder-cluster ones. Cheap: the
     * function is itself CAS+250ms-interval gated. */
    if (initialized)
        gb_maybe_write_stats();

    if (!real_cudaStreamSynchronize)
        real_cudaStreamSynchronize = (pfn_cudaStreamSynchronize)dlsym(RTLD_NEXT,
                                                                       "cudaStreamSynchronize");
    return real_cudaStreamSynchronize ? real_cudaStreamSynchronize(stream) : CUDA_SUCCESS;
}

/* CUDA doc: cudaStreamCreate → intercept and optionally upgrade to a
 * high-priority stream for inference.  Gate: GREENBOOST_STREAM_PRIORITY=1.
 *
 * Stream priority (cudaDevAttrStreamPrioritiesSupported, CUDA 5.0+) lets
 * inference kernels preempt background copy / prefetch work.  The CUDA
 * scheduler honours priority when multiple streams are ready - higher
 * priority streams are issued to SMs first, reducing token-generation
 * jitter when a prefetch is in flight on a low-priority stream.
 *
 * cudaStreamNonBlocking (0x01) flag is forwarded unchanged. */
cudaError_t cudaStreamCreate(cudaStream_t *pStream, unsigned int flags)
{
    GB_CUDART_ENSURE();
    static pfn_cudaStreamCreate real_create = NULL;
    if (!real_create)
        real_create = (pfn_cudaStreamCreate)dlsym(RTLD_NEXT, "cudaStreamCreate");

    /* Only upgrade when GREENBOOST_STREAM_PRIORITY=1 and the runtime API is
     * available.  Silently falls back to the standard path otherwise. */
    const char *sp = getenv("GREENBOOST_STREAM_PRIORITY");
    if (sp && sp[0] == '1' && real_cudaStreamCreateWithPriority
            && real_cudaDeviceGetStreamPriorityRange) {
        int lo = 0, hi = 0;
        cudaError_t pr = real_cudaDeviceGetStreamPriorityRange(&lo, &hi);
        if (pr == CUDA_SUCCESS && hi < lo) {
            /* Higher numeric priority = higher urgency in CUDA's convention. */
            cudaError_t sr = real_cudaStreamCreateWithPriority(pStream, flags, hi);
            if (sr == CUDA_SUCCESS)
                return CUDA_SUCCESS;
        }
    }
    return real_create ? real_create(pStream, flags) : (cudaError_t)30 /* cudaErrorNotInitialized */;
}

/* ------------------------------------------------------------------ */
/*  __cudaRegisterFunction - kernel name registration                   */
/*                                                                      */
/*  The CUDA runtime calls this for every __global__ function at        */
/*  library load time, mapping the host stub pointer (hostFun) to the  */
/*  device kernel name (deviceName).  We register both locally (for    */
/*  gb_netc_lookup_kernel) and on every connected feeder (so the feeder */
/*  daemon can find and launch the kernel by name via dlsym).           */
/* ------------------------------------------------------------------ */
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, gb_uint3 *tid, gb_uint3 *bid,
                            gb_dim3 *bDim, gb_dim3 *gDim, int *wSize)
{
    GB_CUDART_ENSURE();
    if (deviceName && hostFun)
        gb_netc_register_kernel((const void *)hostFun, deviceName);

    /* PR-TT: convert name-keyed signature overrides into host_fn-keyed
     * registrations.  The config file was loaded at shim init; here we
     * look up by deviceName (the mangled kernel name CUDA gives us)
     * and, if found, register the corresponding host_fn pointer with
     * the override arg-count.  Subsequent gb_kernel_sig_lookup on the
     * host_fn returns the override, bounding the kernelParams[] scan. */
    if (deviceName && hostFun) {
        uint32_t override_n = gb_kernel_sig_lookup_name(deviceName);
        if (override_n)
            gb_kernel_sig_register((const void *)hostFun, override_n);
    }

    if (!real___cudaRegisterFunction)
        real___cudaRegisterFunction = (pfn___cudaRegisterFunction)dlsym(RTLD_NEXT,
                                                                        "__cudaRegisterFunction");
    if (real___cudaRegisterFunction)
        real___cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName,
                                   thread_limit, tid, bid, bDim, gDim, wSize);
}

/* cudaGetDriverEntryPointByVersion - passthrough for PyTorch cu12x/cu13x.
 * PyTorch 2.x calls this to resolve CUDA driver entry points by version.
 *
 * Problem: real_cudaGetDriverEntryPointByVersion is resolved at constructor
 * time from libcudart.so, which ldconfig resolves to libcudart.so.12 (system).
 * cudaGetDriverEntryPointByVersion was added in CUDA 11.3 but libcudart.so.12
 * exported it as cudaGetDriverEntryPoint (without "ByVersion").  PyTorch cu128+
 * uses libcudart.so.13 which DOES have cudaGetDriverEntryPointByVersion, but
 * that library is in the conda env and not yet loaded when our constructor runs.
 *
 * Fix: lazy resolution - the first time we are called, libcudart.so.13 is
 * already resident (PyTorch loaded it), so RTLD_NOLOAD finds it.
 */
cudaError_t cudaGetDriverEntryPointByVersion(const char *symbol, void **funcPtr,
                                              unsigned int cudaVersion,
                                              unsigned long long flags,
                                              void *driverStatus)
{
    GB_CUDART_ENSURE();
    if (!real_cudaGetDriverEntryPointByVersion) {
        /* libcudart.so.13 is loaded by PyTorch before this function is called;
         * use RTLD_NOLOAD so we don't redundantly load a second copy. */
        static const char *try_paths[] = {
            "libcudart.so.13", "libcudart.so.12", "libcudart.so", NULL
        };
        for (const char **p = try_paths; *p; p++) {
            void *h = dlopen(*p, RTLD_NOLOAD | RTLD_NOW | RTLD_GLOBAL);
            if (!h) continue;
            pfn_cudaGetDriverEntryPointByVersion fn =
                (pfn_cudaGetDriverEntryPointByVersion)
                (real_dlsym_fn ? real_dlsym_fn(h, "cudaGetDriverEntryPointByVersion")
                               : NULL);
            if (fn) { real_cudaGetDriverEntryPointByVersion = fn; break; }
        }
        if (!real_cudaGetDriverEntryPointByVersion)
            fprintf(stderr, "[GreenBoost] WARNING: cudaGetDriverEntryPointByVersion "
                    "not found in any libcudart - CUDA driver init may fail\n");
    }

    if (real_cudaGetDriverEntryPointByVersion)
        return real_cudaGetDriverEntryPointByVersion(symbol, funcPtr, cudaVersion,
                                                     flags, driverStatus);
    if (funcPtr) *funcPtr = NULL;
    return 6; /* cudaErrorInvalidValue */
}

/* ====================================================================== */
/*  PR-R/F-S9: per-thread default-stream API variants                      */
/*                                                                          */
/*  CUDA exports two parallel symbol families for stream-aware APIs:        */
/*    - Legacy default stream: cudaMemcpyAsync, cuLaunchKernel, ...         */
/*    - Per-thread default stream: cudaMemcpyAsync_ptsz (runtime API)       */
/*                                  cuLaunchKernel_ptds   (driver  API)    */
/*                                                                          */
/*  When CUDA_API_PER_THREAD_DEFAULT_STREAM is defined at the caller's      */
/*  compile time, the CUDA headers redirect each unsuffixed call to the     */
/*  _ptsz/_ptds variant.  PyTorch wheels built with                          */
/*  --default-stream per-thread, bitsandbytes, and several inference        */
/*  servers (vLLM in some configurations) end up calling the suffixed       */
/*  symbols directly - bypassing the shim hooks on the unsuffixed names.    */
/*                                                                          */
/*  The behavioural difference between suffixed and unsuffixed is purely    */
/*  "what does stream==0 mean": legacy treats it as a process-global         */
/*  stream, _ptsz/_ptds treats it as a per-thread default.  Both variants   */
/*  take the SAME explicit-stream argument from the caller, which we just   */
/*  pass through.  So our hook can serve both with no behavioural change    */
/*  - we just need to publish symbols under both names.                     */
/*                                                                          */
/*  Implementation: GCC alias attribute.  Each alias is a zero-cost ELF     */
/*  symbol pointing at the same code as the unsuffixed hook.  No extra      */
/*  call frame, no thunk, no LTO impact.  The targets are non-static so    */
/*  the alias is valid.                                                     */
/* ====================================================================== */

/* Runtime API (libcudart) - _ptsz suffix */
extern cudaError_t cudaMemcpyAsync_ptsz(void *, const void *, size_t, int, cudaStream_t)
    __attribute__((alias("cudaMemcpyAsync")));
extern cudaError_t cudaMemPrefetchAsync_ptsz(const void *, size_t, int, cudaStream_t)
    __attribute__((alias("cudaMemPrefetchAsync")));
/* PR-LL: runtime-API per-thread default-stream alias.  Mirrors the driver-API
 * cuLaunchKernel_ptds added in PR-R.  Without this, PyTorch builds compiled
 * with --default-stream per-thread bypass our cudaLaunchKernel hook
 * (and therefore feeder data-driven dispatch on the runtime side). */
extern cudaError_t cudaLaunchKernel_ptsz(const void *, gb_dim3, gb_dim3, void **,
                                          size_t, cudaStream_t)
    __attribute__((alias("cudaLaunchKernel")));

/* Driver API (libcuda) - _ptds suffix */
extern CUresult cuMemAllocAsync_ptds(CUdeviceptr *, size_t, CUstream)
    __attribute__((alias("cuMemAllocAsync")));
extern CUresult cuMemAllocFromPoolAsync_ptds(CUdeviceptr *, size_t,
                                             CUmemoryPool_handle, CUstream)
    __attribute__((alias("cuMemAllocFromPoolAsync")));
extern CUresult cuMemFreeAsync_ptds(CUdeviceptr, CUstream)
    __attribute__((alias("cuMemFreeAsync")));
extern CUresult cuMemPrefetchAsync_ptds(CUdeviceptr, size_t, CUdevice, CUstream)
    __attribute__((alias("cuMemPrefetchAsync")));
extern CUresult cuLaunchKernel_ptds(CUfunction,
                                    unsigned int, unsigned int, unsigned int,
                                    unsigned int, unsigned int, unsigned int,
                                    unsigned int, CUstream, void **, void **)
    __attribute__((alias("cuLaunchKernel")));
extern CUresult cuLaunchKernelEx_ptds(const gb_CUlaunchConfig *, CUfunction,
                                      void **, void **)
    __attribute__((alias("cuLaunchKernelEx")));
extern CUresult cuLaunchCooperativeKernel_ptds(CUfunction,
                                               unsigned int, unsigned int, unsigned int,
                                               unsigned int, unsigned int, unsigned int,
                                               unsigned int, CUstream, void **)
    __attribute__((alias("cuLaunchCooperativeKernel")));

/* libcudart.so.12 non-default version aliases are in greenboost_cuda_v12.c,
 * compiled without -flto so .symver inline asm survives link-time optimization.
 * That same file also contains an address-taken keepalive table that forces the
 * LTO linker to export each hook body referenced by the trampolines. */

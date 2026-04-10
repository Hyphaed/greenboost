/*
 * greenboost_vulkan_layer.c — VK_LAYER_GREENBOOST_memory
 *
 * Implicit Vulkan layer, activated by GREENBOOST_VULKAN=1 in the environment.
 * Install manifest: /etc/vulkan/implicit_layer.d/VkLayer_greenboost.json
 *
 * Hook 1: vkGetPhysicalDeviceMemoryProperties[2[KHR]]
 *   Inflates the device-local heap to match the virtual VRAM the CUDA shim
 *   reports to CUDA applications (auto-detected from kernel module params).
 *   Games read this value to choose quality presets and texture budgets.
 *
 * Hook 2: vkAllocateMemory
 *   On VK_ERROR_OUT_OF_DEVICE_MEMORY, attempts a tiered fallback:
 *     T2: GreenBoost DDR via DMA-BUF import (VK_KHR_external_memory_fd).
 *     T3: NVMe-spillable 4K pages via DMA-BUF import (for large allocs).
 *   Pressure-aware: skips doomed T2 attempts when the pool is critical.
 *   All overflow allocations are tracked in a hash table for lifecycle management.
 *
 * Hook 3: vkFreeMemory
 *   On freeing a tracked T2/T3 allocation, marks the kernel buffer COLD via
 *   GB_IOCTL_MADVISE so it is evicted first under pressure.
 *
 * Memory orchestration:
 *   - Burst detector: loading-screen alloc bursts are marked HOT (working set)
 *   - Pressure cache: GB_IOCTL_GET_INFO polled every 16 allocs
 *   - Session cleanup: GB_IOCTL_RELEASE_PID in destructor
 *
 * Dispatch model: minimal static arrays (<=4 instances / <=4 devices).
 * Thread-safety rule: the mutex guards ONLY the dispatch-table arrays.
 *   All external calls (next_gipa, next_gdpa, Vulkan functions) are made
 *   OUTSIDE the mutex to prevent deadlock under concurrent threads.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <pthread.h>
#include <syslog.h>
#include <time.h>

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include "greenboost_ioctl.h"

/* ── Configuration ────────────────────────────────────────────────────── */

/* Total virtual VRAM presented to games (T1 + T2).
 * 0 = init failed → inflate_heaps() is a no-op. */
static uint64_t g_gbvk_virtual_vram_bytes = 0;

/* Minimum alloc size to attempt T2 DMA-BUF fallback (default: from env or 32 MB).
 * Lower than AI shim's 64 MB because games make many 32-64 MB texture allocs. */
static uint64_t g_gbvk_overflow_min_bytes = 32ULL * 1024ULL * 1024ULL;

/* Minimum alloc size to attempt T3 NVMe fallback (default 128 MB).
 * Only large streaming textures can tolerate NVMe latency. */
static uint64_t g_gbvk_t3_min_bytes = 128ULL * 1024ULL * 1024ULL;

/* Debug flag — set GREENBOOST_VK_DEBUG=1 for verbose syslog. */
static int g_gbvk_debug = 0;

#define GBVK_MAX_INSTANCES  4
#define GBVK_MAX_DEVICES    4

/* ── Logging ──────────────────────────────────────────────────────────── */

#define gbvk_log(fmt, ...) \
    syslog(LOG_INFO, "[VK_LAYER_GREENBOOST] " fmt, ##__VA_ARGS__)

#define gbvk_dbg(fmt, ...) \
    do { if (g_gbvk_debug) syslog(LOG_DEBUG, "[VK_LAYER_GREENBOOST] " fmt, ##__VA_ARGS__); } while(0)

/* ── Persistent /dev/greenboost fd ───────────────────────────────────── */

static int              g_gbvk_dev_fd = -1;
static pthread_once_t   g_gbvk_dev_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t  g_gbvk_dev_mutex = PTHREAD_MUTEX_INITIALIZER;

static void gbvk_open_dev(void)
{
    g_gbvk_dev_fd = open("/dev/greenboost", O_RDWR | O_CLOEXEC);
    if (g_gbvk_dev_fd < 0)
        gbvk_log("open /dev/greenboost failed — T2/T3 fallback disabled");
    else
        gbvk_log("opened /dev/greenboost fd=%d", g_gbvk_dev_fd);
}

static int gbvk_dev_fd(void)
{
    pthread_once(&g_gbvk_dev_once, gbvk_open_dev);
    return g_gbvk_dev_fd;
}

/* ── Allocation tracking hash table ──────────────────────────────────── */
/*
 * Open-addressed hash table mapping VkDeviceMemory (uint64_t) to overflow
 * allocation metadata. Simplified from the CUDA shim's 131K-slot design;
 * gaming workloads have far fewer large allocations (typically 10-200).
 *
 * Slot 0 = empty, Slot 1 = tombstone (deleted; never a valid VkDeviceMemory).
 * Fibonacci hash + linear probe. 8 striped locks.
 */
#define GBVK_HT_BITS    12
#define GBVK_HT_SIZE    (1u << GBVK_HT_BITS)   /* 4096 slots */
#define GBVK_HT_MASK    (GBVK_HT_SIZE - 1u)
#define GBVK_HT_LOCKS   8
#define GBVK_HT_TOMBSTONE ((uint64_t)1)

typedef struct {
    uint64_t  key;      /* VkDeviceMemory handle (0=empty, 1=tombstone)    */
    uint64_t  size;     /* allocation size in bytes                        */
    int       dma_fd;   /* dup'd DMA-BUF fd for madvise/evict calls        */
    int32_t   buf_id;   /* kernel IDR id returned in gb_alloc_req.fd       */
    uint8_t   tier;     /* 2 = T2 DDR, 3 = T3 NVMe                        */
    uint8_t   flags;    /* GB_ALLOC_* flags used at allocation time        */
    uint8_t   hot;      /* 1 = marked HOT (loading-screen working set)     */
    uint8_t   _pad;
} __attribute__((aligned(64))) GbVkHtEntry;

static GbVkHtEntry    g_gbvk_ht[GBVK_HT_SIZE];
static pthread_mutex_t g_gbvk_ht_locks[GBVK_HT_LOCKS] = {
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER,
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER,
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER,
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER,
};

static inline uint32_t gbvk_ht_hash(uint64_t key)
{
    return (uint32_t)((key * UINT64_C(0x9E3779B97F4A7C15)) >> (64 - GBVK_HT_BITS));
}

static void gbvk_ht_insert(uint64_t key, uint64_t size, int dma_fd,
                            int32_t buf_id, uint8_t tier, uint8_t flags)
{
    uint32_t h = gbvk_ht_hash(key);
    pthread_mutex_t *lk = &g_gbvk_ht_locks[h & (GBVK_HT_LOCKS - 1)];
    pthread_mutex_lock(lk);
    for (uint32_t i = 0; i < GBVK_HT_SIZE; i++) {
        uint32_t idx = (h + i) & GBVK_HT_MASK;
        if (!g_gbvk_ht[idx].key || g_gbvk_ht[idx].key == GBVK_HT_TOMBSTONE) {
            g_gbvk_ht[idx].key    = key;
            g_gbvk_ht[idx].size   = size;
            g_gbvk_ht[idx].dma_fd = dma_fd;
            g_gbvk_ht[idx].buf_id = buf_id;
            g_gbvk_ht[idx].tier   = tier;
            g_gbvk_ht[idx].flags  = flags;
            g_gbvk_ht[idx].hot    = 0;
            break;
        }
    }
    pthread_mutex_unlock(lk);
}

/* Returns a copy of the entry and tombstones the slot. Returns 0 if not found. */
static int gbvk_ht_remove(uint64_t key, GbVkHtEntry *out)
{
    uint32_t h = gbvk_ht_hash(key);
    pthread_mutex_t *lk = &g_gbvk_ht_locks[h & (GBVK_HT_LOCKS - 1)];
    pthread_mutex_lock(lk);
    int found = 0;
    for (uint32_t i = 0; i < GBVK_HT_SIZE; i++) {
        uint32_t idx = (h + i) & GBVK_HT_MASK;
        if (!g_gbvk_ht[idx].key) break;  /* empty slot — key absent */
        if (g_gbvk_ht[idx].key == key) {
            *out = g_gbvk_ht[idx];
            g_gbvk_ht[idx].key = GBVK_HT_TOMBSTONE;
            found = 1;
            break;
        }
    }
    pthread_mutex_unlock(lk);
    return found;
}

/* ── Session statistics ───────────────────────────────────────────────── */

static _Atomic uint32_t g_gbvk_t2_count  = 0;
static _Atomic uint32_t g_gbvk_t3_count  = 0;
static _Atomic uint64_t g_gbvk_t2_bytes  = 0;
static _Atomic uint64_t g_gbvk_t3_bytes  = 0;
static _Atomic uint32_t g_gbvk_oom_count = 0; /* allocs that failed all tiers */

/* ── Pool info cache (refreshed every 16 alloc attempts) ──────────────── */

#define GBVK_INFO_REFRESH_INTERVAL 16
static struct gb_info   g_gbvk_pool_info;
static _Atomic uint32_t g_gbvk_alloc_counter = 0;
static pthread_mutex_t  g_gbvk_info_mutex = PTHREAD_MUTEX_INITIALIZER;

static void gbvk_refresh_pool_info(void)
{
    int fd = gbvk_dev_fd();
    if (fd < 0) return;
    struct gb_info info;
    if (ioctl(fd, GB_IOCTL_GET_INFO, &info) == 0) {
        pthread_mutex_lock(&g_gbvk_info_mutex);
        g_gbvk_pool_info = info;
        pthread_mutex_unlock(&g_gbvk_info_mutex);

        /* Dynamically update heap inflation target when the pool cap changes
         * (e.g. user called GB_IOCTL_SET_POOL_CAP mid-session). This keeps
         * the size reported to games (T1 GPU VRAM + T2 DDR pool) accurate. */
        if (info.vram_physical_mb > 0 && info.max_pool_mb > 0) {
            uint64_t total = (info.vram_physical_mb + info.max_pool_mb)
                             * 1024ULL * 1024ULL;
            if (total != g_gbvk_virtual_vram_bytes) {
                g_gbvk_virtual_vram_bytes = total;
                gbvk_dbg("pool refresh: heap target updated → %llu GB "
                         "(T1=%llu MB + T2 cap=%llu MB)",
                         (unsigned long long)(total >> 30),
                         (unsigned long long)info.vram_physical_mb,
                         (unsigned long long)info.max_pool_mb);
            }
        }

        gbvk_dbg("pool refresh: T2 %llu/%llu MB (%s), T3 %llu MB",
                 (unsigned long long)info.allocated_mb,
                 (unsigned long long)info.max_pool_mb,
                 info.t2_pressure == GB_T2_PRESSURE_CRITICAL ? "CRITICAL" :
                 info.t2_pressure == GB_T2_PRESSURE_WARN     ? "WARN"     : "ok",
                 (unsigned long long)info.nvme_t3_allocated_mb);
    }
}

static uint32_t gbvk_t2_pressure(void)
{
    pthread_mutex_lock(&g_gbvk_info_mutex);
    uint32_t p = g_gbvk_pool_info.t2_pressure;
    pthread_mutex_unlock(&g_gbvk_info_mutex);
    return p;
}

/* ── Burst detector: mark loading-screen allocs HOT ─────────────────── */
/*
 * During a game's loading screen, the GPU gets a rapid burst of large texture
 * allocs. These become the game's working set and should stay in T2 until freed.
 * We detect the end of a burst (quiet for 2 seconds) and mark all burst allocs HOT
 * via GB_IOCTL_MADVISE so the kernel evicts them last under pressure.
 */
static _Atomic uint64_t g_gbvk_last_alloc_ms  = 0;
static _Atomic uint32_t g_gbvk_burst_active   = 0;
#define GBVK_BURST_QUIET_MS  2000

static uint64_t gbvk_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

/* Called after each successful T2/T3 alloc. Updates burst state. */
static void gbvk_burst_record(void)
{
    uint64_t now = gbvk_now_ms();
    uint64_t prev = atomic_load(&g_gbvk_last_alloc_ms);
    atomic_store(&g_gbvk_last_alloc_ms, now);

    if (!atomic_load(&g_gbvk_burst_active) || (now - prev < GBVK_BURST_QUIET_MS)) {
        atomic_store(&g_gbvk_burst_active, 1);
    }
}

/*
 * Called every 16 allocs. If burst was active and now quiet for >=2s,
 * iterate hash table and mark all unfreed burst allocs HOT.
 */
static void gbvk_burst_check(void)
{
    if (!atomic_load(&g_gbvk_burst_active)) return;
    uint64_t now = gbvk_now_ms();
    uint64_t last = atomic_load(&g_gbvk_last_alloc_ms);
    if (now - last < GBVK_BURST_QUIET_MS) return;

    /* Burst ended — mark all tracked allocs HOT (working set). */
    int fd = gbvk_dev_fd();
    uint32_t marked = 0;
    for (uint32_t i = 0; i < GBVK_HT_SIZE; i++) {
        uint32_t lock_idx = gbvk_ht_hash(g_gbvk_ht[i].key) & (GBVK_HT_LOCKS - 1);
        pthread_mutex_lock(&g_gbvk_ht_locks[lock_idx]);
        GbVkHtEntry *e = &g_gbvk_ht[i];
        if (e->key && e->key != GBVK_HT_TOMBSTONE && !e->hot) {
            e->hot = 1;
            if (fd >= 0) {
                struct gb_madvise_req m = { .buf_id = e->buf_id,
                                            .advise = GB_MADVISE_HOT };
                ioctl(fd, GB_IOCTL_MADVISE, &m);
            }
            marked++;
        }
        pthread_mutex_unlock(&g_gbvk_ht_locks[lock_idx]);
    }
    atomic_store(&g_gbvk_burst_active, 0);
    if (marked)
        gbvk_log("burst ended: marked %u allocs HOT (game working set)", marked);
}

/* ── Runtime init ─────────────────────────────────────────────────────── */

__attribute__((constructor))
static void gbvk_init(void)
{
    /* Only activate when explicitly enabled. */
    const char *env = getenv("GREENBOOST_VULKAN");
    if (!env || env[0] != '1')
        return;

    /* Debug logging. */
    const char *dbg = getenv("GREENBOOST_VK_DEBUG");
    if (dbg && dbg[0] == '1') g_gbvk_debug = 1;

    /* Configurable overflow threshold (apply before early returns). */
    const char *min_env = getenv("GREENBOOST_VK_OVERFLOW_MIN_MB");
    if (min_env) {
        long long mb = atoll(min_env);
        if (mb > 0)
            g_gbvk_overflow_min_bytes = (uint64_t)mb * 1024ULL * 1024ULL;
    }

    /* Configurable T3 threshold. */
    const char *t3_env = getenv("GREENBOOST_VK_T3_MIN_MB");
    if (t3_env) {
        long long mb = atoll(t3_env);
        if (mb > 0)
            g_gbvk_t3_min_bytes = (uint64_t)mb * 1024ULL * 1024ULL;
    }

    /* Read virtual VRAM from kernel module sysfs. */
    int physical_gb = -1, virtual_gb = -1;
    char buf[32];
    FILE *f;

    f = fopen("/sys/module/greenboost/parameters/physical_vram_gb", "r");
    if (f) { if (fgets(buf, sizeof(buf), f)) physical_gb = atoi(buf); fclose(f); }

    f = fopen("/sys/module/greenboost/parameters/virtual_vram_gb", "r");
    if (f) { if (fgets(buf, sizeof(buf), f)) virtual_gb = atoi(buf); fclose(f); }

    if (physical_gb > 0 && virtual_gb > 0) {
        g_gbvk_virtual_vram_bytes =
            ((uint64_t)physical_gb + (uint64_t)virtual_gb) * 1024ULL * 1024ULL * 1024ULL;
        gbvk_log("init: sysfs physical=%d GB + virtual=%d GB = %llu GB | "
                 "T2 overflow>=%llu MB | T3 overflow>=%llu MB",
                 physical_gb, virtual_gb,
                 (unsigned long long)(g_gbvk_virtual_vram_bytes >> 30),
                 (unsigned long long)(g_gbvk_overflow_min_bytes >> 20),
                 (unsigned long long)(g_gbvk_t3_min_bytes >> 20));
        return;
    }

    /* Fallback 2: query GB_IOCTL_GET_INFO directly.
     * Works even when the kernel module sysfs params are unavailable (e.g.
     * Path B/C without greenboost.ko, or if sysfs read failed).
     * Exposes the real CUDA memory pool: T1 GPU VRAM + T2 DDR pool cap. */
    {
        int ifd = gbvk_dev_fd();
        if (ifd >= 0) {
            struct gb_info info;
            if (ioctl(ifd, GB_IOCTL_GET_INFO, &info) == 0 &&
                info.vram_physical_mb > 0 && info.max_pool_mb > 0) {
                g_gbvk_virtual_vram_bytes =
                    (info.vram_physical_mb + info.max_pool_mb) * 1024ULL * 1024ULL;
                gbvk_log("init: ioctl T1=%llu MB + T2 cap=%llu MB = %llu GB "
                         "| T2 overflow>=%llu MB | T3 overflow>=%llu MB",
                         (unsigned long long)info.vram_physical_mb,
                         (unsigned long long)info.max_pool_mb,
                         (unsigned long long)(g_gbvk_virtual_vram_bytes >> 30),
                         (unsigned long long)(g_gbvk_overflow_min_bytes >> 20),
                         (unsigned long long)(g_gbvk_t3_min_bytes >> 20));
                return;
            }
        }
    }

    /* Fallback 3: env var override (manual / testing). */
    const char *vram_env = getenv("GREENBOOST_VIRTUAL_VRAM_MB");
    if (vram_env) {
        long long mb = atoll(vram_env);
        if (mb > 0) {
            g_gbvk_virtual_vram_bytes = (uint64_t)mb * 1024ULL * 1024ULL;
            gbvk_log("init: GREENBOOST_VIRTUAL_VRAM_MB=%lld MB", mb);
            return;
        }
    }

    gbvk_log("init: kernel params unavailable — heap inflation disabled, "
             "T2/T3 overflow still available on OOM");
}

/* ── Process-exit cleanup ─────────────────────────────────────────────── */

__attribute__((destructor))
static void gbvk_fini(void)
{
    uint32_t t2 = atomic_load(&g_gbvk_t2_count);
    uint32_t t3 = atomic_load(&g_gbvk_t3_count);
    uint64_t t2b = atomic_load(&g_gbvk_t2_bytes);
    uint64_t t3b = atomic_load(&g_gbvk_t3_bytes);
    uint32_t oom = atomic_load(&g_gbvk_oom_count);

    if (t2 || t3 || oom)
        gbvk_log("session end: T2=%u allocs (%llu MB) T3=%u allocs (%llu MB) failed=%u",
                 t2, (unsigned long long)(t2b >> 20),
                 t3, (unsigned long long)(t3b >> 20),
                 oom);

    int fd = g_gbvk_dev_fd;
    if (fd >= 0) {
        /* Release all buffers owned by this process. */
        struct gb_release_pid_req r = { .pid = 0 };
        ioctl(fd, GB_IOCTL_RELEASE_PID, &r);
        close(fd);
        g_gbvk_dev_fd = -1;
    }
}

/* ── Per-instance state ───────────────────────────────────────────────── */

typedef struct {
    VkInstance                               instance;
    PFN_vkGetInstanceProcAddr                next_gipa;
    PFN_vkDestroyInstance                    next_destroy_instance;
    PFN_vkGetPhysicalDeviceMemoryProperties  next_get_mem_props;
    PFN_vkGetPhysicalDeviceMemoryProperties2 next_get_mem_props2;
} GbInstData;

/* ── Per-device state ─────────────────────────────────────────────────── */

typedef struct {
    VkDevice                             device;
    PFN_vkGetDeviceProcAddr              next_gdpa;
    PFN_vkDestroyDevice                  next_destroy_device;
    PFN_vkAllocateMemory                 next_alloc_mem;
    PFN_vkFreeMemory                     next_free_mem;
    PFN_vkGetMemoryFdPropertiesKHR       next_get_mem_fd_props;
    VkPhysicalDeviceMemoryProperties     mem_props; /* cached for overflow path */
} GbDevData;

static GbInstData       g_inst[GBVK_MAX_INSTANCES];
static GbDevData        g_dev[GBVK_MAX_DEVICES];
static pthread_mutex_t  g_mutex = PTHREAD_MUTEX_INITIALIZER;

/* ── Table helpers (all called under g_mutex) ─────────────────────────── */

static GbInstData *inst_alloc(VkInstance h)
{
    for (int i = 0; i < GBVK_MAX_INSTANCES; i++)
        if (!g_inst[i].instance) { g_inst[i].instance = h; return &g_inst[i]; }
    return NULL;
}
static GbInstData *inst_find(VkInstance h)
{
    for (int i = 0; i < GBVK_MAX_INSTANCES; i++)
        if (g_inst[i].instance == h) return &g_inst[i];
    return NULL;
}
static void inst_free(VkInstance h)
{
    for (int i = 0; i < GBVK_MAX_INSTANCES; i++)
        if (g_inst[i].instance == h) { memset(&g_inst[i], 0, sizeof g_inst[i]); return; }
}

static GbDevData *dev_alloc(VkDevice h)
{
    for (int i = 0; i < GBVK_MAX_DEVICES; i++)
        if (!g_dev[i].device) { g_dev[i].device = h; return &g_dev[i]; }
    return NULL;
}
static GbDevData *dev_find(VkDevice h)
{
    for (int i = 0; i < GBVK_MAX_DEVICES; i++)
        if (g_dev[i].device == h) return &g_dev[i];
    return NULL;
}
static void dev_free(VkDevice h)
{
    for (int i = 0; i < GBVK_MAX_DEVICES; i++)
        if (g_dev[i].device == h) { memset(&g_dev[i], 0, sizeof g_dev[i]); return; }
}

/* ── Helper: inflate device-local heaps ──────────────────────────────── */

static void inflate_heaps(VkPhysicalDeviceMemoryProperties *p)
{
    if (!g_gbvk_virtual_vram_bytes) return;
    for (uint32_t i = 0; i < p->memoryHeapCount; i++) {
        if ((p->memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
            p->memoryHeaps[i].size < g_gbvk_virtual_vram_bytes)
            p->memoryHeaps[i].size = g_gbvk_virtual_vram_bytes;
    }
}

/* ── Helper: inflate VK_EXT_memory_budget heapBudget[] ───────────────── */
/*
 * DXVK and VKD3D-Proton query VK_EXT_memory_budget (heapBudget[]) to decide
 * how much VRAM they can actually allocate — heap size alone is not enough.
 * Without this, they see the real ~12 GB physical budget and cap textures there
 * despite the inflated heap size. We overwrite heapBudget[] for device-local
 * heaps to match g_gbvk_virtual_vram_bytes so the full T1+T2 pool is usable.
 * heapUsage[] is left unchanged (reflects real driver usage, keeps OOM sane).
 */
static void inflate_budget(VkPhysicalDeviceMemoryProperties2 *p)
{
    if (!g_gbvk_virtual_vram_bytes) return;

    VkBaseOutStructure *chain = (VkBaseOutStructure *)p->pNext;
    while (chain) {
        if (chain->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT) {
            VkPhysicalDeviceMemoryBudgetPropertiesEXT *budget =
                (VkPhysicalDeviceMemoryBudgetPropertiesEXT *)chain;
            for (uint32_t i = 0; i < p->memoryProperties.memoryHeapCount; i++) {
                if (p->memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                    budget->heapBudget[i] = g_gbvk_virtual_vram_bytes;
            }
            gbvk_dbg("inflate_budget: set heapBudget[] = %llu GB for device-local heaps",
                     (unsigned long long)(g_gbvk_virtual_vram_bytes >> 30));
            break;
        }
        chain = (VkBaseOutStructure *)chain->pNext;
    }
}

/* ── Helper: attempt one DMA-BUF overflow alloc ─────────────────────── */
/*
 * Allocates a GreenBoost kernel buffer and imports it as a Vulkan device memory
 * object using VK_KHR_external_memory_fd. Used for both T2 and T3 paths.
 *
 * Returns VK_SUCCESS and fills *pMemory on success.
 * Returns the original OOM result on any failure.
 */
static VkResult gbvk_try_dmabuf_alloc(
    VkDevice                            device,
    const VkMemoryAllocateInfo         *pAllocInfo,
    const VkAllocationCallbacks        *pAllocator,
    VkDeviceMemory                     *pMemory,
    PFN_vkAllocateMemory                fn_alloc,
    PFN_vkGetMemoryFdPropertiesKHR      fn_fd_props,
    const VkPhysicalDeviceMemoryProperties *mem_props,
    uint32_t                            alloc_flags,
    uint8_t                             tier)
{
    int fd = gbvk_dev_fd();
    if (fd < 0) return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    struct gb_alloc_req req;
    memset(&req, 0, sizeof req);
    req.size  = pAllocInfo->allocationSize;
    req.flags = alloc_flags;

    if (ioctl(fd, GB_IOCTL_ALLOC, &req) < 0)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    /* Query memory types compatible with this DMA-BUF. */
    VkMemoryFdPropertiesKHR fd_props = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
        .pNext = NULL,
    };
    if (!fn_fd_props ||
        fn_fd_props(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                    req.fd, &fd_props) != VK_SUCCESS) {
        close(req.fd);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    /* Prefer host-cached+coherent; settle for host-visible+coherent. */
    uint32_t fallback_type = UINT32_MAX;
    for (uint32_t i = 0; i < mem_props->memoryTypeCount; i++) {
        if (!(fd_props.memoryTypeBits & (1u << i))) continue;
        VkMemoryPropertyFlags f = mem_props->memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            fallback_type = i;
            if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) break;
        }
    }
    if (fallback_type == UINT32_MAX) {
        close(req.fd);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    VkImportMemoryFdInfoKHR import_info = {
        .sType      = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .pNext      = NULL,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        .fd         = req.fd,
    };
    VkMemoryAllocateInfo fallback = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext           = &import_info,
        .allocationSize  = pAllocInfo->allocationSize,
        .memoryTypeIndex = fallback_type,
    };

    VkResult res = fn_alloc(device, &fallback, pAllocator, pMemory);
    if (res != VK_SUCCESS) {
        close(req.fd);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    /* fd ownership transferred to Vulkan driver on success. Dup for our tracking. */
    int track_fd = dup(req.fd);

    gbvk_ht_insert((uint64_t)(uintptr_t)*pMemory,
                   pAllocInfo->allocationSize,
                   track_fd, req.fd /*buf_id = kernel IDR fd*/,
                   tier, (uint8_t)alloc_flags);

    if (tier == 2) {
        atomic_fetch_add(&g_gbvk_t2_count, 1);
        atomic_fetch_add(&g_gbvk_t2_bytes, pAllocInfo->allocationSize);
    } else {
        atomic_fetch_add(&g_gbvk_t3_count, 1);
        atomic_fetch_add(&g_gbvk_t3_bytes, pAllocInfo->allocationSize);
    }
    gbvk_burst_record();

    gbvk_log("AllocateMemory: T%u DMA-BUF OK — %llu MB (type %u, flags=0x%x)",
             tier, (unsigned long long)(pAllocInfo->allocationSize >> 20),
             fallback_type, alloc_flags);
    return VK_SUCCESS;
}

/* ── Hook: vkAllocateMemory ────────────────────────────────────────────── */

static VKAPI_ATTR VkResult VKAPI_CALL
gbvk_AllocateMemory(VkDevice                       device,
                    const VkMemoryAllocateInfo    *pAllocInfo,
                    const VkAllocationCallbacks   *pAllocator,
                    VkDeviceMemory                *pMemory)
{
    /* Snapshot function pointers and mem_props under lock; call outside. */
    pthread_mutex_lock(&g_mutex);
    GbDevData *d = dev_find(device);
    PFN_vkAllocateMemory           fn_alloc    = d ? d->next_alloc_mem       : NULL;
    PFN_vkGetMemoryFdPropertiesKHR fn_fd_props = d ? d->next_get_mem_fd_props : NULL;
    VkPhysicalDeviceMemoryProperties mem_props = d ? d->mem_props
                                                   : (VkPhysicalDeviceMemoryProperties){};
    pthread_mutex_unlock(&g_mutex);

    if (!fn_alloc) return VK_ERROR_DEVICE_LOST;

    /* Refresh pool info every 16 alloc attempts (piggyback, no background thread). */
    uint32_t cnt = atomic_fetch_add(&g_gbvk_alloc_counter, 1);
    if ((cnt % GBVK_INFO_REFRESH_INTERVAL) == 0) {
        gbvk_refresh_pool_info();
        gbvk_burst_check();
    }

    /* Try the real allocator first (T1 VRAM). */
    VkResult res = fn_alloc(device, pAllocInfo, pAllocator, pMemory);
    if (res != VK_ERROR_OUT_OF_DEVICE_MEMORY)
        return res;

    /* Below the minimum size — not a candidate for overflow. */
    if (pAllocInfo->allocationSize < g_gbvk_overflow_min_bytes) {
        atomic_fetch_add(&g_gbvk_oom_count, 1);
        return res;
    }

    /* Pressure-aware T2 routing:
     *   CRITICAL + large alloc → skip T2 (already saturated), go to T3
     *   WARN → try T2 but go to T3 immediately on failure
     *   OK   → normal T2 path */
    uint32_t pressure = gbvk_t2_pressure();
    int skip_t2 = (pressure == GB_T2_PRESSURE_CRITICAL &&
                   pAllocInfo->allocationSize >= 256ULL * 1024ULL * 1024ULL);

    if (!skip_t2) {
        VkResult t2_res = gbvk_try_dmabuf_alloc(
            device, pAllocInfo, pAllocator, pMemory,
            fn_alloc, fn_fd_props, &mem_props,
            GB_ALLOC_WEIGHTS, 2);
        if (t2_res == VK_SUCCESS) return VK_SUCCESS;
    } else {
        gbvk_dbg("AllocateMemory: T2 skipped (CRITICAL pressure), %llu MB → T3 direct",
                 (unsigned long long)(pAllocInfo->allocationSize >> 20));
    }

    /* T3 NVMe fallback — only for large streaming textures. */
    if (pAllocInfo->allocationSize >= g_gbvk_t3_min_bytes) {
        VkResult t3_res = gbvk_try_dmabuf_alloc(
            device, pAllocInfo, pAllocator, pMemory,
            fn_alloc, fn_fd_props, &mem_props,
            GB_ALLOC_WEIGHTS | GB_ALLOC_NO_HUGEPAGE, 3);
        if (t3_res == VK_SUCCESS) return VK_SUCCESS;
    }

    atomic_fetch_add(&g_gbvk_oom_count, 1);
    gbvk_log("AllocateMemory: all tiers failed for %llu MB — returning OOM",
             (unsigned long long)(pAllocInfo->allocationSize >> 20));
    return res;
}

/* ── Hook: vkFreeMemory ────────────────────────────────────────────────── */

static VKAPI_ATTR void VKAPI_CALL
gbvk_FreeMemory(VkDevice                       device,
                VkDeviceMemory                 memory,
                const VkAllocationCallbacks   *pAllocator)
{
    pthread_mutex_lock(&g_mutex);
    GbDevData *d = dev_find(device);
    PFN_vkFreeMemory fn = d ? d->next_free_mem : NULL;
    pthread_mutex_unlock(&g_mutex);

    /* Check if this was a tracked T2/T3 overflow alloc. */
    if (memory != VK_NULL_HANDLE) {
        GbVkHtEntry entry;
        if (gbvk_ht_remove((uint64_t)(uintptr_t)memory, &entry)) {
            int gbfd = gbvk_dev_fd();
            if (gbfd >= 0) {
                /* Mark COLD — kernel evicts this buffer first under pressure. */
                struct gb_madvise_req m = { .buf_id = entry.buf_id,
                                            .advise = GB_MADVISE_COLD };
                ioctl(gbfd, GB_IOCTL_MADVISE, &m);
            }
            if (entry.dma_fd >= 0)
                close(entry.dma_fd);

            gbvk_dbg("FreeMemory: T%u %llu MB → COLD",
                     entry.tier, (unsigned long long)(entry.size >> 20));
        }
    }

    if (fn) fn(device, memory, pAllocator);
}

/* ── Hook: vkGetPhysicalDeviceMemoryProperties ────────────────────────── */

static VKAPI_ATTR void VKAPI_CALL
gbvk_GetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice physicalDevice,
    VkPhysicalDeviceMemoryProperties *pMemoryProperties)
{
    pthread_mutex_lock(&g_mutex);
    PFN_vkGetPhysicalDeviceMemoryProperties fn = NULL;
    for (int i = 0; i < GBVK_MAX_INSTANCES; i++)
        if (g_inst[i].next_get_mem_props) { fn = g_inst[i].next_get_mem_props; break; }
    pthread_mutex_unlock(&g_mutex);

    if (fn) fn(physicalDevice, pMemoryProperties);
    inflate_heaps(pMemoryProperties);
}

static VKAPI_ATTR void VKAPI_CALL
gbvk_GetPhysicalDeviceMemoryProperties2(
    VkPhysicalDevice physicalDevice,
    VkPhysicalDeviceMemoryProperties2 *pMemoryProperties)
{
    pthread_mutex_lock(&g_mutex);
    PFN_vkGetPhysicalDeviceMemoryProperties2 fn = NULL;
    for (int i = 0; i < GBVK_MAX_INSTANCES; i++)
        if (g_inst[i].next_get_mem_props2) { fn = g_inst[i].next_get_mem_props2; break; }
    pthread_mutex_unlock(&g_mutex);

    if (fn) fn(physicalDevice, pMemoryProperties);
    inflate_heaps(&pMemoryProperties->memoryProperties);
    inflate_budget(pMemoryProperties);
}

/* ── Hook: vkCreateInstance ────────────────────────────────────────────── */

static VKAPI_ATTR VkResult VKAPI_CALL
gbvk_CreateInstance(const VkInstanceCreateInfo  *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkInstance                  *pInstance)
{
    VkLayerInstanceCreateInfo *ldci =
        (VkLayerInstanceCreateInfo *)pCreateInfo->pNext;
    while (ldci && !(ldci->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                     ldci->function == VK_LAYER_LINK_INFO))
        ldci = (VkLayerInstanceCreateInfo *)ldci->pNext;
    if (!ldci) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_gipa = ldci->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    ldci->u.pLayerInfo = ldci->u.pLayerInfo->pNext;

    PFN_vkCreateInstance next_ci =
        (PFN_vkCreateInstance)next_gipa(VK_NULL_HANDLE, "vkCreateInstance");
    if (!next_ci) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult res = next_ci(pCreateInfo, pAllocator, pInstance);
    if (res != VK_SUCCESS) return res;

    PFN_vkDestroyInstance next_di = (PFN_vkDestroyInstance)
        next_gipa(*pInstance, "vkDestroyInstance");
    PFN_vkGetPhysicalDeviceMemoryProperties next_props =
        (PFN_vkGetPhysicalDeviceMemoryProperties)
        next_gipa(*pInstance, "vkGetPhysicalDeviceMemoryProperties");
    PFN_vkGetPhysicalDeviceMemoryProperties2 next_props2 =
        (PFN_vkGetPhysicalDeviceMemoryProperties2)
        next_gipa(*pInstance, "vkGetPhysicalDeviceMemoryProperties2");

    pthread_mutex_lock(&g_mutex);
    GbInstData *d = inst_alloc(*pInstance);
    if (d) {
        d->next_gipa              = next_gipa;
        d->next_destroy_instance  = next_di;
        d->next_get_mem_props     = next_props;
        d->next_get_mem_props2    = next_props2;
    }
    pthread_mutex_unlock(&g_mutex);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
gbvk_DestroyInstance(VkInstance instance, const VkAllocationCallbacks *pAllocator)
{
    pthread_mutex_lock(&g_mutex);
    GbInstData *d = inst_find(instance);
    PFN_vkDestroyInstance fn = d ? d->next_destroy_instance : NULL;
    inst_free(instance);
    pthread_mutex_unlock(&g_mutex);
    if (fn) fn(instance, pAllocator);
}

/* ── Hook: vkCreateDevice ──────────────────────────────────────────────── */

static VKAPI_ATTR VkResult VKAPI_CALL
gbvk_CreateDevice(VkPhysicalDevice             physDev,
                  const VkDeviceCreateInfo    *pCreateInfo,
                  const VkAllocationCallbacks *pAllocator,
                  VkDevice                    *pDevice)
{
    VkLayerDeviceCreateInfo *ldci =
        (VkLayerDeviceCreateInfo *)pCreateInfo->pNext;
    while (ldci && !(ldci->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                     ldci->function == VK_LAYER_LINK_INFO))
        ldci = (VkLayerDeviceCreateInfo *)ldci->pNext;
    if (!ldci) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_gipa = ldci->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr   next_gdpa = ldci->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    ldci->u.pLayerInfo = ldci->u.pLayerInfo->pNext;

    PFN_vkCreateDevice next_cd =
        (PFN_vkCreateDevice)next_gipa(VK_NULL_HANDLE, "vkCreateDevice");
    if (!next_cd) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult res = next_cd(physDev, pCreateInfo, pAllocator, pDevice);
    if (res != VK_SUCCESS) return res;

    /*
     * Resolve all function pointers and cache memory properties BEFORE the mutex.
     * next_gdpa() and get_props() are external calls — must not run under g_mutex.
     */
    PFN_vkAllocateMemory  fn_alloc   = (PFN_vkAllocateMemory)
        next_gdpa(*pDevice, "vkAllocateMemory");
    PFN_vkFreeMemory      fn_free    = (PFN_vkFreeMemory)
        next_gdpa(*pDevice, "vkFreeMemory");
    PFN_vkDestroyDevice   fn_destroy = (PFN_vkDestroyDevice)
        next_gdpa(*pDevice, "vkDestroyDevice");
    PFN_vkGetMemoryFdPropertiesKHR fn_fd_props = (PFN_vkGetMemoryFdPropertiesKHR)
        next_gdpa(*pDevice, "vkGetMemoryFdPropertiesKHR");

    /* Snapshot mem_props using already-resolved instance-level function. */
    VkPhysicalDeviceMemoryProperties mem_props = {};
    pthread_mutex_lock(&g_mutex);
    PFN_vkGetPhysicalDeviceMemoryProperties get_props = NULL;
    for (int i = 0; i < GBVK_MAX_INSTANCES; i++)
        if (g_inst[i].next_get_mem_props) { get_props = g_inst[i].next_get_mem_props; break; }
    pthread_mutex_unlock(&g_mutex);

    if (get_props) get_props(physDev, &mem_props);  /* external call, outside mutex */

    pthread_mutex_lock(&g_mutex);
    GbDevData *d = dev_alloc(*pDevice);
    if (d) {
        d->next_gdpa             = next_gdpa;
        d->next_alloc_mem        = fn_alloc;
        d->next_free_mem         = fn_free;
        d->next_destroy_device   = fn_destroy;
        d->next_get_mem_fd_props = fn_fd_props;
        d->mem_props             = mem_props;
    }
    pthread_mutex_unlock(&g_mutex);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
gbvk_DestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator)
{
    pthread_mutex_lock(&g_mutex);
    GbDevData *d = dev_find(device);
    PFN_vkDestroyDevice fn = d ? d->next_destroy_device : NULL;
    dev_free(device);
    pthread_mutex_unlock(&g_mutex);
    if (fn) fn(device, pAllocator);
}

/* ── Proc-addr dispatch ─────────────────────────────────────────────────── */

static PFN_vkVoidFunction gbvk_GetDeviceProcAddr(VkDevice dev, const char *name);

static PFN_vkVoidFunction gbvk_GetInstanceProcAddr(VkInstance inst, const char *name)
{
#define HOOK(fn)  if (strcmp(name, #fn) == 0) return (PFN_vkVoidFunction)gbvk_##fn
    HOOK(GetInstanceProcAddr);
    HOOK(CreateInstance);
    HOOK(DestroyInstance);
    HOOK(CreateDevice);
    HOOK(DestroyDevice);
    HOOK(GetPhysicalDeviceMemoryProperties);
    HOOK(GetPhysicalDeviceMemoryProperties2);
    /* KHR alias — same implementation. */
    if (strcmp(name, "vkGetPhysicalDeviceMemoryProperties2KHR") == 0)
        return (PFN_vkVoidFunction)gbvk_GetPhysicalDeviceMemoryProperties2;
    HOOK(AllocateMemory);
    HOOK(FreeMemory);
#undef HOOK
    if (strcmp(name, "vkGetDeviceProcAddr") == 0)
        return (PFN_vkVoidFunction)gbvk_GetDeviceProcAddr;

    pthread_mutex_lock(&g_mutex);
    GbInstData *d = inst ? inst_find(inst) : NULL;
    PFN_vkGetInstanceProcAddr fn = d ? d->next_gipa : NULL;
    pthread_mutex_unlock(&g_mutex);
    if (fn) return fn(inst, name);
    return NULL;
}

static PFN_vkVoidFunction gbvk_GetDeviceProcAddr(VkDevice dev, const char *name)
{
#define HOOK(fn)  if (strcmp(name, #fn) == 0) return (PFN_vkVoidFunction)gbvk_##fn
    HOOK(GetDeviceProcAddr);
    HOOK(DestroyDevice);
    HOOK(AllocateMemory);
    HOOK(FreeMemory);
#undef HOOK

    pthread_mutex_lock(&g_mutex);
    GbDevData *d = dev ? dev_find(dev) : NULL;
    PFN_vkGetDeviceProcAddr fn = d ? d->next_gdpa : NULL;
    pthread_mutex_unlock(&g_mutex);
    if (fn) return fn(dev, name);
    return NULL;
}

/* ── Loader negotiation entry point ────────────────────────────────────── */

__attribute__((visibility("default")))
VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface *pVersionStruct)
{
    if (pVersionStruct->loaderLayerInterfaceVersion > CURRENT_LOADER_LAYER_INTERFACE_VERSION)
        pVersionStruct->loaderLayerInterfaceVersion = CURRENT_LOADER_LAYER_INTERFACE_VERSION;

    pVersionStruct->pfnGetInstanceProcAddr       = gbvk_GetInstanceProcAddr;
    pVersionStruct->pfnGetDeviceProcAddr         = gbvk_GetDeviceProcAddr;
    pVersionStruct->pfnGetPhysicalDeviceProcAddr = NULL;
    return VK_SUCCESS;
}

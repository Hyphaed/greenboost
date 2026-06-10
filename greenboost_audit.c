/*
 * GreenBoost v2.8 - LD_AUDIT injection library
 *
 * Tiny audit library (< 5 KB) placed in /etc/ld.so.preload.
 * Injects libgreenboost_cuda.so only into processes that load a CUDA
 * or GGML-CUDA library, leaving PAM helpers, GDM, snap-confine, cups,
 * and every other non-CUDA process completely untouched.
 *
 * The AppArmor blast radius is a single read-only open() of this file.
 * No sensitive logic; no CUDA code.  The full shim is dlopen()'ed only
 * once a CUDA library appears in the process link map.
 *
 * Trigger patterns in la_objopen():
 *   libcuda.so.*      - NVIDIA driver API (Ollama, vLLM, TGI, PyTorch)
 *   libcudart.so.*    - CUDA runtime (HuggingFace Transformers, etc.)
 *
 * Author  : Ferran Duarri
 * License : GPL v2
 */

#define _GNU_SOURCE
#include <link.h>
#include <dlfcn.h>
#include <string.h>
#include <stdlib.h>

#ifndef SHIM_PATH
#define SHIM_PATH "/usr/local/lib/libgreenboost_cuda.so"
#endif

/*
 * Returns 1 if the shim should be skipped for this process.
 * Called once and cached; result is valid for the lifetime of the process.
 *
 * Safe to call from la_objopen: getenv() is pure env-table lookup with no
 * allocation, and la_objopen is serialized under the linker's global lock.
 *
 * at the Linux libcuda.so level, which is reached via Wine's PE/ELF bridge
 *
 *   PRESSURE_VESSEL_RUNTIME  - nested LD_AUDIT inside Pressure Vessel sandbox
 *                              risks double-loading the shim; the outer wine64
 *                              process (outside the sandbox) already has it.
 *
 *   GREENBOOST_DISABLE       - explicit opt-out for any process or game that
 *                              should not be hooked (Steam: GREENBOOST_DISABLE=1 %command%)
 */
static int should_skip_shim(void)
{
    /* REF-05: Use __atomic builtins for consistency with shim_loaded below.
     * la_objopen is serialized under the linker lock in practice, but using
     * plain int with no barrier is technically undefined under C11. */
    static int cached = -1;
    int v = __atomic_load_n(&cached, __ATOMIC_ACQUIRE);
    if (v >= 0)
        return v;

    if (getenv("GREENBOOST_DISABLE")) {
        __atomic_store_n(&cached, 1, __ATOMIC_RELEASE);
        return 1;
    }

    /* Pressure Vessel container: nested sandbox - outer wine64 already has shim */
    if (getenv("PRESSURE_VESSEL_RUNTIME")) {
        __atomic_store_n(&cached, 1, __ATOMIC_RELEASE);
        return 1;
    }

    /* Steam launcher UI: has STEAM_RUNTIME but no SteamAppId.
     * Game processes launched by Steam have both.  The launcher itself does not
     * run CUDA workloads but may dlopen libcuda.so for overlay / DLSS detection,
     * which would otherwise trigger shim injection into a context it was not
     * designed for, causing a Bus error crash. */
    if (getenv("STEAM_RUNTIME") && !getenv("SteamAppId")) {
        __atomic_store_n(&cached, 1, __ATOMIC_RELEASE);
        return 1;
    }

    __atomic_store_n(&cached, 0, __ATOMIC_RELEASE);
    return 0;
}

/*
 * la_version - required LD_AUDIT entry point.
 * The dynamic linker calls this first; if the returned version is not
 * understood, the audit library is silently disabled.
 *
 * Must be exported with default visibility: -fvisibility=hidden (inherited
 * from COMMON_CFLAGS) would hide it from the dynamic symbol table, causing
 * glibc to reject the library from /etc/ld.so.preload with
 * "cannot be preloaded (cannot open shared object file)".
 */
__attribute__((visibility("default")))
unsigned int la_version(unsigned int version)
{
    (void)version;
    return LAV_CURRENT;
}

/*
 * la_objopen - called by the dynamic linker each time it maps a shared
 * object into the process address space.
 *
 * Constraints (called from within the linker's own lock):
 *   - No malloc() / free()
 *   - No stdio (fprintf, printf) in the fast path - async-signal-unsafe
 *   - strncmp() / strrchr() are pure computation - safe
 *
 * Once the shim is loaded it stays resident (RTLD_NODELETE) even if
 * the triggering library is later dlclose()'d.  Subsequent la_objopen
 * calls return immediately via shim_loaded check.
 */
/* -----------------------------------------------------------------------
 * Blackwell VMM intercept via la_symbind64
 *
 * libcuda.so.1 exports cuDeviceGetAttribute and cuMemAddressReserve as
 * UNVERSIONED symbols.  Our CUDA shim exports them as @@GB_HOOKS (versioned).
 * glibc's ELF linker PREFERS unversioned definitions for unversioned
 * references, so libcuda always wins the PLT race over our shim - regardless
 * of ld.so.preload order.  la_symbind64 intercepts the binding AT resolution
 * time, bypassing versioning entirely, and is the only reliable hook point.
 *
 * What these intercepts do:
 *  cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED=193)
 *    → returns 0 on Blackwell (cc >= 12) desktop PCIe, forcing ggml-cuda to
 *      select pool_leg (cudaMalloc-based) over pool_vmm (cuMemCreate/cuMemMap).
 *      pool_vmm's HOST_NUMA_CURRENT fallback is DMA-only on Blackwell PCIe;
 *      cudaMalloc overflow routes through managed-UVM (SM-accessible).
 *  cuMemAddressReserve → returns NOT_SUPPORTED on Blackwell as defence-in-depth.
 *
 * Override: GREENBOOST_BLACKWELL_ALLOW_VMM=1 re-enables both (for ATS-capable
 * Blackwell server SKUs where HOST_NUMA_CURRENT IS SM-accessible).
 * ----------------------------------------------------------------------- */

#define GB_AUDIT_ATTR_VMM_SUPPORTED 193  /* CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED */
#define GB_AUDIT_ATTR_CC_MAJOR       75  /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR */

typedef int  (*pfn_cuDGA)(int *, int, int);
typedef int  (*pfn_cuMAR)(unsigned long long *, size_t, size_t, unsigned long long, unsigned long long);

static pfn_cuDGA gb_real_cuDGA = NULL;
static int        gb_audit_cc   = 0;

/* Lazy-load the real cuDeviceGetAttribute directly from libcuda to avoid
 * infinite recursion when LD_PRELOAD/la_symbind has already redirected the
 * global symbol to this wrapper. */
static pfn_cuDGA gb_get_real_cuDGA(void)
{
    if (gb_real_cuDGA) return gb_real_cuDGA;
    void *h = dlopen("libcuda.so.1", RTLD_NOLOAD | RTLD_NOW | RTLD_GLOBAL);
    if (!h) h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if (h) gb_real_cuDGA = (pfn_cuDGA)dlsym(h, "cuDeviceGetAttribute");
    return gb_real_cuDGA;
}

/* Wrapper replacing cuDeviceGetAttribute in ggml-cuda's PLT.
 * Zeroes VMM support flag on Blackwell to force legacy cudaMalloc pool. */
__attribute__((visibility("default")))
int gb_audit_cuDeviceGetAttribute(int *value, int attrib, int dev)
{
    pfn_cuDGA real = gb_get_real_cuDGA();
    if (!real) return 100; /* CUDA_ERROR_NOT_SUPPORTED */

    int ret = real(value, attrib, dev);
    if (ret != 0) return ret;

    if (attrib == GB_AUDIT_ATTR_VMM_SUPPORTED && value && *value != 0) {
        if (!gb_audit_cc)
            real(&gb_audit_cc, GB_AUDIT_ATTR_CC_MAJOR, dev);
        if (gb_audit_cc >= 12) {
            const char *e = getenv("GREENBOOST_BLACKWELL_ALLOW_VMM");
            if (!e || e[0] == '0')
                *value = 0;  /* disable VMM → forces ggml pool_leg */
        }
    }
    return 0;
}

/* Wrapper replacing cuMemAddressReserve: return NOT_SUPPORTED on Blackwell
 * so ggml's pool_vmm availability test also fails (defence-in-depth). */
__attribute__((visibility("default")))
int gb_audit_cuMemAddressReserve(unsigned long long *ptr, size_t size,
                                  size_t alignment, unsigned long long addr,
                                  unsigned long long flags)
{
    if (!gb_audit_cc) {
        pfn_cuDGA real = gb_get_real_cuDGA();
        if (real) real(&gb_audit_cc, GB_AUDIT_ATTR_CC_MAJOR, 0);
    }
    if (gb_audit_cc >= 12) {
        const char *e = getenv("GREENBOOST_BLACKWELL_ALLOW_VMM");
        if (!e || e[0] == '0')
            return 801; /* CUDA_ERROR_NOT_SUPPORTED */
    }
    /* Non-Blackwell or allow_vmm: pass through via the CUDA shim's hook */
    typedef int (*pfn_t)(unsigned long long *, size_t, size_t,
                         unsigned long long, unsigned long long);
    void *h = dlopen("libcuda.so.1", RTLD_NOLOAD | RTLD_NOW | RTLD_GLOBAL);
    pfn_t real = h ? (pfn_t)dlsym(h, "cuMemAddressReserve") : NULL;
    if (!real) return 801;
    return real(ptr, size, alignment, addr, flags);
}

static volatile int shim_loaded = 0;

__attribute__((visibility("default")))
unsigned int la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie)
{
    (void)lmid;
    (void)cookie;

    /* glibc ≥ 2.38 calls la_objopen() for internal linker objects (vDSO,
     * early-init state) with map == NULL.  Return 0 immediately - there is
     * nothing meaningful to inspect and dereferencing a null map SIGSEGV's
     * every subprocess spawned by a session that has this library preloaded. */
    if (!map)
        return 0;

    if (__atomic_load_n(&shim_loaded, __ATOMIC_ACQUIRE) || should_skip_shim())
        return 0;

    const char *name = map->l_name;
    if (!name || !*name)
        return 0;

    /* Find the basename (portion after the last '/') */
    const char *base = strrchr(name, '/');
    base = base ? base + 1 : name;

    if (strncmp(base, "libcuda.so",     10) == 0 ||
        strncmp(base, "libcudart.so",   12) == 0 ||
        strncmp(base, "libggml-cuda",   12) == 0 ||  /* llama.cpp GGML CUDA backend */
        strncmp(base, "ggml-cuda",       9) == 0)    /* llama.cpp alternative path   */
    {
        /* Atomic gate: only one thread wins; others see shim_loaded=1.
         *
         * SEC-05 note: dlopen() is called here while the dynamic linker holds
         * its internal _dl_load_lock.  On glibc this lock is recursive, so the
         * nested dlopen() inside gb_shim_init() does not deadlock - but this is
         * technically undefined per rtld-audit(7).
         *
         * The correct approach would be to set a flag here and call dlopen() in
         * la_preinit().  However la_preinit() runs only once at process start;
         * for Ollama and vLLM, which load libcuda.so lazily at runtime via
         * dlopen(), la_preinit() has already returned by the time la_objopen()
         * fires for libcuda - so the shim would never be injected.
         *
         * Therefore: keep the dlopen() here.  This works correctly on glibc
         * (the only supported runtime), and the recursive-mutex behaviour is
         * stable across all glibc versions since 2.17.  If a future glibc
         * breaks this, the fix is to add a pthread helper thread that calls
         * dlopen() after being signalled by la_objopen(). */
#if defined(__x86_64__) || defined(__aarch64__)
        if (__sync_val_compare_and_swap(&shim_loaded, 0, 1) == 0)
            dlopen(SHIM_PATH, RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
#endif
    }

    return 0;
}

/* la_symbind64 is intentionally NOT implemented here.
 * Redirecting symbols via la_symbind64 from a preloaded audit library
 * causes crashes because sym->st_value conventions differ across glibc
 * versions and symbol binding phases.  The cuDeviceGetAttribute/
 * cuMemAddressReserve overrides are instead provided by
 * libgreenboost_vmm_override.so (built from greenboost_vmm_override.c),
 * which exports bare unversioned symbols that win the PLT race against
 * libcuda.so.1's own unversioned exports. */

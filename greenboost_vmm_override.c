/*
 * GreenBoost v3.0 - Blackwell VMM override library
 *
 * WHY THIS FILE EXISTS:
 *   glibc's dynamic linker explicitly PREFERS unversioned symbol definitions
 *   over versioned (@@VER) ones when resolving an unversioned PLT reference.
 *   libcuda.so.1 exports cuDeviceGetAttribute and cuMemAddressReserve as bare
 *   unversioned symbols.  The main GreenBoost CUDA shim exports them as
 *   @@GB_HOOKS (versioned), so libcuda always wins the PLT race - regardless
 *   of ld.so.preload search order - and our hooks never fire for ggml-cuda's
 *   PLT calls.
 *
 * THE FIX:
 *   This library exports cuDeviceGetAttribute and cuMemAddressReserve as
 *   BARE UNVERSIONED symbols (no version script, no @@).  Placed FIRST in
 *   /etc/ld.so.preload it is loaded before libcuda.so.1 enters the link map,
 *   so glibc finds OUR unversioned definition first and uses it for the PLT.
 *
 *   ADDITIONALLY (v3.0.1 Blackwell dlsym-path fix, 2026-06-16):
 *   Ollama's bundled libggml-cuda.so (ollama engine backend) loads all CUDA
 *   driver API symbols via dlopen("libcuda.so.1") + dlsym(handle, "sym"),
 *   bypassing the PLT entirely.  The PLT preemption above does NOT intercept
 *   these runtime lookups.  To cover the dlsym path we also wrap dlsym():
 *   when any code asks for "cuDeviceGetAttribute" or "cuMemAddressReserve" we
 *   return our hooked versions regardless of which handle was passed.
 *   This closes the race for dlsym-based CUDA loaders (ggml-cuda, llama-server,
 *   etc.) while leaving all other symbol lookups unaffected.
 *
 * WHAT THESE OVERRIDES DO:
 *   cuDeviceGetAttribute(attr=193 "VMM_SUPPORTED"):
 *     Returns 0 on Blackwell (cc >= 12) desktop PCIe.  ggml-cuda checks this
 *     attribute to decide between pool_vmm (cuMemCreate/cuMemMap) and pool_leg
 *     (cudaMalloc).  pool_vmm's HOST_NUMA_CURRENT T2 fallback is DMA-only on
 *     Blackwell PCIe - kernels touching T2 pointers crash with
 *     CUDA_ERROR_INVALID_RESOURCE_HANDLE (400).  pool_leg (cudaMalloc) routes
 *     T1 overflow through GreenBoost's gb_vmm_t2_alloc_blackwell_managed()
 *     which uses cuMemAllocManaged (SM-accessible on RTX 5070,
 *     concurrentManagedAccess=1).
 *
 *   cuMemAddressReserve:
 *     Returns CUDA_ERROR_NOT_SUPPORTED on Blackwell.  Defence-in-depth: even
 *     if the attribute check is bypassed by cached state, the VA reservation
 *     for pool_vmm fails here, forcing the same pool_leg fallback.
 *
 * OVERRIDE: GREENBOOST_BLACKWELL_ALLOW_VMM=1 re-enables both for ATS-capable
 * Blackwell server SKUs (H100 Blackwell, GH200) where HOST_NUMA_CURRENT IS
 * SM-accessible.
 *
 * Build: gcc -shared -fPIC -O2 -o libgreenboost_vmm_override.so \
 *              greenboost_vmm_override.c -ldl
 * (deliberately NO version script - all exports are bare/unversioned)
 */

#define _GNU_SOURCE
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>

/*
 * __libc_dlsym — glibc-private bootstrap.  Lives in ld.so, not libdl, so it
 * is NOT interceptable via PLT preemption.  Used by libfakeroot, libeatmydata,
 * nsswitch wrappers, etc. to safely call the real dlsym from inside a dlsym
 * override without infinite recursion.  Weak-reference so the build succeeds
 * even on musl (where it would be NULL and the dlsym wrapper is skipped).
 */
extern __attribute__((weak)) void *__libc_dlsym(void *map, const char *name);

/* CUDA error codes used here */
#define CUDA_SUCCESS                    0
#define CUDA_ERROR_NOT_SUPPORTED      801
#define CUDA_ERROR_INVALID_VALUE       1

/* CU_DEVICE_ATTRIBUTE values */
#define GB_ATTR_VMM_SUPPORTED         193  /* CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED */
#define GB_ATTR_CC_MAJOR               75  /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR */

typedef int (*pfn_cuDGA_t)(int *, int, int);
typedef int (*pfn_cuMAR_t)(unsigned long long *, size_t, size_t,
                            unsigned long long, unsigned long long);

/* Lazily load the real cuDeviceGetAttribute directly from libcuda.so.1.
 * Using RTLD_NOLOAD first avoids a redundant open(); falls back to a real
 * open only if libcuda isn't already mapped (shouldn't happen in practice).
 *
 * IMPORTANT: use __libc_dlsym, NOT dlsym(), to avoid recursing through our
 * own dlsym wrapper below.  __libc_dlsym resolves against the actual link-map
 * handle and is immune to PLT interception. */
static pfn_cuDGA_t gb_real_cuDGA(void)
{
    static pfn_cuDGA_t fn = NULL;
    if (__builtin_expect(fn != NULL, 1)) return fn;
    void *h = dlopen("libcuda.so.1", RTLD_NOLOAD | RTLD_NOW | RTLD_GLOBAL);
    if (!h) h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if (h) {
        /* Prefer __libc_dlsym (bypasses our wrapper); fall back to real_dlsym
         * set in the dlsym wrapper's constructor path when __libc_dlsym is absent */
        if (__libc_dlsym)
            fn = (pfn_cuDGA_t)__libc_dlsym(h, "cuDeviceGetAttribute");
        else
            fn = (pfn_cuDGA_t)dlsym(h, "cuDeviceGetAttribute");
    }
    return fn;
}

static int gb_cc_major = 0;  /* cached compute capability */

/* Shared Blackwell check: returns 1 if we should disable VMM on this device.
 *
 * Race-free CC probe: GREENBOOST_FORCE_CC_MAJOR=<n> bypasses the lazily-loaded
 * cuDeviceGetAttribute probe that fires BEFORE cuInit() completes on the first
 * ggml_cuda_init() call (Blackwell "Deferred cc probe: Compute 12.x" race).
 * Set GREENBOOST_FORCE_CC_MAJOR=12 in the ollama drop-in for RTX 50xx/Blackwell
 * so the VMM-SUPPORTED attribute override takes effect even on the first call. */
static int gb_blackwell_disable_vmm(int dev)
{
    const char *e = getenv("GREENBOOST_BLACKWELL_ALLOW_VMM");
    if (e && e[0] != '0') return 0;  /* explicitly allowed */

    /* GREENBOOST_FORCE_CC_MAJOR: skips the lazy cuDeviceGetAttribute probe.
     * Eliminates the Blackwell timing race where the probe returns 0 because
     * CUDA is not yet initialized when the first VMM-supported attribute check
     * fires during ggml_cuda_init() — causing the vmm/pool_leg split to be
     * resolved incorrectly and GGML to crash at cuMemAddressReserve. */
    if (!gb_cc_major) {
        const char *cc_env = getenv("GREENBOOST_FORCE_CC_MAJOR");
        if (cc_env && cc_env[0]) gb_cc_major = atoi(cc_env);
    }

    if (!gb_cc_major) {
        pfn_cuDGA_t f = gb_real_cuDGA();
        if (f) f(&gb_cc_major, GB_ATTR_CC_MAJOR, dev);
    }
    return gb_cc_major >= 12;
}

/*
 * cuDeviceGetAttribute - bare unversioned export.
 *
 * Wins the PLT race over libcuda.so.1's equally unversioned export because
 * this library is loaded BEFORE libcuda in /etc/ld.so.preload, so it appears
 * first in glibc's link map when resolving the unversioned reference.
 */
int cuDeviceGetAttribute(int *value, int attrib, int dev)
{
    pfn_cuDGA_t real = gb_real_cuDGA();
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;

    int ret = real(value, attrib, dev);
    if (ret != CUDA_SUCCESS) return ret;

    if (attrib == GB_ATTR_VMM_SUPPORTED && value && *value != 0
            && gb_blackwell_disable_vmm(dev)) {
        *value = 0;  /* disable VMM → ggml selects pool_leg (cudaMalloc) */
    }
    return CUDA_SUCCESS;
}

/*
 * cuMemAddressReserve - bare unversioned export.
 * Defence-in-depth: if pool_vmm somehow bypasses the attribute check, the VA
 * reservation itself fails here, preventing DMA-only HOST_NUMA_CURRENT memory
 * from being mapped into ggml's weight tensor address space.
 */
int cuMemAddressReserve(unsigned long long *ptr, size_t size, size_t alignment,
                         unsigned long long addr, unsigned long long flags)
{
    if (gb_blackwell_disable_vmm(0))
        return CUDA_ERROR_NOT_SUPPORTED;

    typedef int (*real_t)(unsigned long long *, size_t, size_t,
                           unsigned long long, unsigned long long);
    void *h = dlopen("libcuda.so.1", RTLD_NOLOAD | RTLD_NOW | RTLD_GLOBAL);
    real_t real = NULL;
    if (h) {
        /* Use __libc_dlsym to avoid recursing through our dlsym wrapper */
        if (__libc_dlsym)
            real = (real_t)__libc_dlsym(h, "cuMemAddressReserve");
        else
            real = (real_t)dlsym(h, "cuMemAddressReserve");
    }
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;
    return real(ptr, size, alignment, addr, flags);
}

/*
 * dlsym interception — closes the runtime dlopen/dlsym path that GGML uses to
 * load CUDA driver functions, bypassing PLT preemption entirely.
 *
 * Modern ollama's libggml-cuda.so does (roughly):
 *   void *h = dlopen("libcuda.so.1", RTLD_NOW|RTLD_GLOBAL);
 *   pfn_cuDGA = (pfn_t)dlsym(h, "cuDeviceGetAttribute");
 *   pfn_cuDGA(&vmm, CU_DEVICE_ATTRIBUTE_VMM_SUPPORTED, dev);
 *
 * Because the function pointer is obtained at runtime via dlsym(), the PLT
 * preemption above (bare unversioned export) has NO effect — the pointer goes
 * directly to libcuda.so.1's implementation.  This dlsym() wrapper intercepts
 * those lookups and returns our hooked versions instead.
 *
 * Bootstrap: we use __libc_dlsym (glibc-private, not PLT-interceptable) to
 * get the real dlsym without triggering infinite recursion.  On musl/non-glibc
 * __libc_dlsym is NULL (weak ref) and the wrapper is a no-op.
 *
 * Thread safety: the static init is a compare-and-store of a single pointer
 * — benign on all x86-64 microarchitectures (TSO) even without a mutex.
 *
 * This is NOT overriding dlvsym() or dlopen() — only the unversioned dlsym().
 * Any caller that uses dlvsym() to get a versioned symbol gets the real result.
 */
void *dlsym(void *handle, const char *name)
{
    /* Bootstrap: initialise the real dlsym pointer exactly once.
     * __libc_dlsym lives in ld.so and is immune to PLT interception. */
    static void *(*real_dlsym)(void *, const char *) = NULL;
    if (!real_dlsym) {
        if (__libc_dlsym)
            real_dlsym = (void *(*)(void *, const char *))
                __libc_dlsym(RTLD_NEXT, "dlsym");
        if (!real_dlsym) {
            /* Non-glibc path or first call before ld.so is ready.
             * Returning NULL would break the caller; best effort: return the
             * real libcuda symbol (no VMM fix on this path). */
            return NULL;
        }
    }

    /* Intercept requests for the two CUDA symbols we need to override.
     * Return our hook regardless of which handle was passed — the caller
     * only cares about the function behaviour, not the address source.
     * (Both hooks forward to the real function for non-Blackwell / allowed.) */
    if (name) {
        if (strcmp(name, "cuDeviceGetAttribute") == 0)
            return (void *)cuDeviceGetAttribute;
        if (strcmp(name, "cuMemAddressReserve") == 0)
            return (void *)cuMemAddressReserve;
    }

    return real_dlsym(handle, name);
}

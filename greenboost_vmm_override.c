/*
 * GreenBoost v3.2 - Blackwell VMM override library
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
 * __libc_dlsym , glibc-private bootstrap.  Lives in ld.so, not libdl, so it
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
     * fires during ggml_cuda_init() , causing the vmm/pool_leg split to be
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
    /*
     * Recursion guard: when libggml-cuda.so is dlopen'd at runtime (via
     * ggml_backend_load_all() CWD scan) with vmm_override first in LD_PRELOAD,
     * our dlsym() wrapper returns vmm_override.cuDeviceGetAttribute for the
     * "cuDeviceGetAttribute" lookup.  gb_real_cuDGA() then caches our own
     * function pointer and real(value, attrib, dev) recurses infinitely.
     *
     * Guard: if we re-enter on the same thread, gb_real_cuDGA() returned
     * our own pointer.  Return VMM_SUPPORTED=0 (the correct answer for
     * Blackwell desktop PCIe, and harmless on non-Blackwell since vmm_override
     * should not be deployed there).
     */
    static _Thread_local int gb_cuDGA_depth = 0;
    if (gb_cuDGA_depth > 0) {
        if (value) *value = 0;
        return CUDA_SUCCESS;
    }

    pfn_cuDGA_t real = gb_real_cuDGA();
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;

    gb_cuDGA_depth++;
    int ret = real(value, attrib, dev);
    gb_cuDGA_depth--;

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
 * dlsym interception , closes the runtime dlopen/dlsym path that GGML uses to
 * load CUDA driver functions, bypassing PLT preemption entirely.
 *
 * Modern ollama's libggml-cuda.so does (roughly):
 *   void *h = dlopen("libcuda.so.1", RTLD_NOW|RTLD_GLOBAL);
 *   pfn_cuDGA = (pfn_t)dlsym(h, "cuDeviceGetAttribute");
 *   pfn_cuDGA(&vmm, CU_DEVICE_ATTRIBUTE_VMM_SUPPORTED, dev);
 *
 * Because the function pointer is obtained at runtime via dlsym(), the PLT
 * preemption above (bare unversioned export) has NO effect , the pointer goes
 * directly to libcuda.so.1's implementation.  This dlsym() wrapper intercepts
 * those lookups and returns our hooked versions instead.
 *
 * Bootstrap: we use __libc_dlsym (glibc-private, not PLT-interceptable) to
 * get the real dlsym without triggering infinite recursion.  On musl/non-glibc
 * __libc_dlsym is NULL (weak ref) and the wrapper is a no-op.
 *
 * Thread safety: the static init is a compare-and-store of a single pointer
 * , benign on all x86-64 microarchitectures (TSO) even without a mutex.
 *
 * This is NOT overriding dlvsym() or dlopen() , only the unversioned dlsym().
 * Any caller that uses dlvsym() to get a versioned symbol gets the real result.
 */
void *dlsym(void *handle, const char *name)
{
    /* Bootstrap: initialise the real dlsym pointer exactly once.
     * __libc_dlsym lives in ld.so and is immune to PLT interception. */
    static void *(*real_dlsym)(void *, const char *) = NULL;
    if (!real_dlsym) {
        /* Use dlvsym with an explicit glibc version to find the real dlsym.
         * __libc_dlsym(map, name) is a glibc-private function that expects a
         * real link-map pointer , passing RTLD_NEXT (a pseudo-handle) to it
         * returns a broken wrapper that cannot perform handle-scoped lookups.
         * Ollama 0.30.8 registers ggml backends dynamically via
         * dlsym(real_handle, entrypoint) , if real_dlsym is broken those
         * lookups return NULL and ggml registers zero backends, crashing every
         * model load with "no backends are loaded".
         * dlvsym with a versioned glibc symbol safely resolves the correct
         * implementation regardless of our unversioned dlsym export. */
        real_dlsym = (void *(*)(void *, const char *))
            dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
        if (!real_dlsym)
            real_dlsym = (void *(*)(void *, const char *))
                dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        if (!real_dlsym)
            real_dlsym = (void *(*)(void *, const char *))
                dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.17");
        if (!real_dlsym && __libc_dlsym)
            /* musl / non-glibc last resort , RTLD_NEXT is accepted here on
             * some implementations even though it is not a link-map. */
            real_dlsym = (void *(*)(void *, const char *))
                __libc_dlsym(RTLD_NEXT, "dlsym");
        if (!real_dlsym)
            return NULL;
    }

    /* Intercept requests for the CUDA symbols we need to override. */
    if (name) {
        /* VMM suppression: return our hooks for all handles (including RTLD_NEXT).
         * Our cuDeviceGetAttribute / cuMemAddressReserve use __libc_dlsym internally
         * to reach the real libcuda version, so there is no recursion risk. */
        if (strcmp(name, "cuDeviceGetAttribute") == 0)
            return (void *)cuDeviceGetAttribute;
        if (strcmp(name, "cuMemAddressReserve") == 0)
            return (void *)cuMemAddressReserve;

        /* Memory-inflation fix: redirect cuDeviceTotalMem_v2 / cuDeviceTotalMem to
         * the GreenBoost CUDA shim's hooked version, which reports phys+virtual VRAM.
         *
         * Why: Ollama's gpu-discover subprocess does dlopen("libcuda.so.1")+dlsym to
         * look up cuDeviceTotalMem_v2 by name, then calls the returned function pointer.
         * The PLT preemption used by the full shim (LD_PRELOAD) does NOT intercept
         * runtime dlsym(specific_handle, name) calls , only PLT references at load
         * time.  Without this redirect, gpu-discover calls libcuda's real function and
         * sees only physical VRAM (12 GB), making Ollama place the model on CPU.
         *
         * Only redirect for library-specific handles (not RTLD_NEXT or RTLD_DEFAULT):
         *   - RTLD_NEXT calls originate from the shim's own init code that populates
         *     real_cuDeviceTotalMem_v2; redirecting those would make the shim store
         *     its own hook as the "real" function → infinite recursion.
         *   - RTLD_DEFAULT calls already find the shim's hook first (it's first in the
         *     LD_PRELOAD load order), so no interception needed.
         *
         * RTLD_DEFAULT == (void*)0, RTLD_NEXT == (void*)-1 on Linux/glibc. */
        if (handle != RTLD_NEXT && handle != RTLD_DEFAULT) {
            if (strcmp(name, "cuDeviceTotalMem_v2") == 0 ||
                strcmp(name, "cuDeviceTotalMem") == 0) {
                /* Look up the shim's inflating hook from the global symbol table.
                 * The shim is preloaded before libcuda.so.1, so RTLD_DEFAULT finds
                 * the shim's cuDeviceTotalMem_v2@@GB_HOOKS before libcuda's version. */
                void *shim_fn = real_dlsym(RTLD_DEFAULT, name);
                /* Return the shim hook if found; fall through to the real handle
                 * lookup only if the shim is absent (edge case: shim not yet loaded). */
                if (shim_fn) return shim_fn;
            }
        }
    }

    return real_dlsym(handle, name);
}

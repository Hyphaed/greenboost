/*
 * GreenBoost v2.9 - Blackwell VMM override library
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
 * open only if libcuda isn't already mapped (shouldn't happen in practice). */
static pfn_cuDGA_t gb_real_cuDGA(void)
{
    static pfn_cuDGA_t fn = NULL;
    if (__builtin_expect(fn != NULL, 1)) return fn;
    void *h = dlopen("libcuda.so.1", RTLD_NOLOAD | RTLD_NOW | RTLD_GLOBAL);
    if (!h) h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    if (h) fn = (pfn_cuDGA_t)dlsym(h, "cuDeviceGetAttribute");
    return fn;
}

static int gb_cc_major = 0;  /* cached compute capability */

/* Shared Blackwell check: returns 1 if we should disable VMM on this device. */
static int gb_blackwell_disable_vmm(int dev)
{
    const char *e = getenv("GREENBOOST_BLACKWELL_ALLOW_VMM");
    if (e && e[0] != '0') return 0;  /* explicitly allowed */
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
    real_t real = h ? (real_t)dlsym(h, "cuMemAddressReserve") : NULL;
    if (!real) return CUDA_ERROR_NOT_SUPPORTED;
    return real(ptr, size, alignment, addr, flags);
}

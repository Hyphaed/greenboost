/*
 * GreenBoost v2.8 — LD_AUDIT injection library
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
 *   libcuda.so.*      — NVIDIA driver API (Ollama, vLLM, TGI, PyTorch)
 *   libcudart.so.*    — CUDA runtime (HuggingFace Transformers, etc.)
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
 * Proton/Steam processes are NOT excluded — GreenBoost intercepts CUDA calls
 * at the Linux libcuda.so level, which is reached via Wine's PE/ELF bridge
 * regardless of Steam/Proton env vars.  Only two cases are excluded:
 *
 *   PRESSURE_VESSEL_RUNTIME  — nested LD_AUDIT inside Pressure Vessel sandbox
 *                              risks double-loading the shim; the outer wine64
 *                              process (outside the sandbox) already has it.
 *
 *   GREENBOOST_DISABLE       — explicit opt-out for any process or game that
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

    /* Pressure Vessel container: nested sandbox — outer wine64 already has shim */
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
 * la_version — required LD_AUDIT entry point.
 * The dynamic linker calls this first; if the returned version is not
 * understood, the audit library is silently disabled.
 */
unsigned int la_version(unsigned int version)
{
    (void)version;
    return LAV_CURRENT;
}

/*
 * la_objopen — called by the dynamic linker each time it maps a shared
 * object into the process address space.
 *
 * Constraints (called from within the linker's own lock):
 *   - No malloc() / free()
 *   - No stdio (fprintf, printf) in the fast path — async-signal-unsafe
 *   - strncmp() / strrchr() are pure computation — safe
 *
 * Once the shim is loaded it stays resident (RTLD_NODELETE) even if
 * the triggering library is later dlclose()'d.  Subsequent la_objopen
 * calls return immediately via shim_loaded check.
 */
static volatile int shim_loaded = 0;

unsigned int la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie)
{
    (void)lmid;
    (void)cookie;

    /* glibc ≥ 2.38 calls la_objopen() for internal linker objects (vDSO,
     * early-init state) with map == NULL.  Return 0 immediately — there is
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
         * nested dlopen() inside gb_shim_init() does not deadlock — but this is
         * technically undefined per rtld-audit(7).
         *
         * The correct approach would be to set a flag here and call dlopen() in
         * la_preinit().  However la_preinit() runs only once at process start;
         * for Ollama and vLLM, which load libcuda.so lazily at runtime via
         * dlopen(), la_preinit() has already returned by the time la_objopen()
         * fires for libcuda — so the shim would never be injected.
         *
         * Therefore: keep the dlopen() here.  This works correctly on glibc
         * (the only supported runtime), and the recursive-mutex behaviour is
         * stable across all glibc versions since 2.17.  If a future glibc
         * breaks this, the fix is to add a pthread helper thread that calls
         * dlopen() after being signalled by la_objopen(). */
#ifdef __x86_64__
        if (__sync_val_compare_and_swap(&shim_loaded, 0, 1) == 0)
            dlopen(SHIM_PATH, RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
#endif
    }

    return 0;
}

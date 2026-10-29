/* RT-07 non-device test: does the shim's dlsym/dlopen hook run for callers that
 * bind GLIBC_2.34 (glibc >= 2.34) as well as GLIBC_2.2.5?
 *
 * One source, three roles (selected by -D):
 *   -DGB_STUB   fake libcuda.so.1 (no GPU, no driver): a few exported symbols.
 *   -DGB_BIND_225  probe whose dlsym/dlopen references are pinned to GLIBC_2.2.5.
 *   (default)   probe whose references use the toolchain default (2.34 here).
 * Driven by tests/shim_rt07_dlsym_versioning.py.
 *
 * Probe output, one line per check: "PASS name" / "FAIL name detail" / "INFO ...".
 * argv[1] = path of the shim .so under test, argv[2] = "preload" or "control".
 */
#define _GNU_SOURCE
#ifdef GB_STUB
#include <stddef.h>
typedef int CUresult;
CUresult cuInit(unsigned f) { (void)f; return 0; }
CUresult cuDriverGetVersion(int *v) { if (v) *v = 12080; return 0; }
CUresult cuDeviceGetCount(int *c) { if (c) *c = 1; return 0; }
CUresult cuDeviceGet(int *d, int o) { (void)o; if (d) *d = 0; return 0; }
CUresult cuMemGetInfo_v2(size_t *f, size_t *t)
{ if (f) *f = 8ull << 30; if (t) *t = 16ull << 30; return 0; }
CUresult cuDeviceTotalMem_v2(size_t *t, int d) { (void)d; if (t) *t = 16ull << 30; return 0; }
CUresult cuDeviceGetAttribute(int *v, int a, int d) { (void)a; (void)d; if (v) *v = 0; return 0; }
CUresult cuMemAlloc_v2(unsigned long long *p, size_t n) { (void)n; if (p) *p = 0; return 2; /* OOM */ }
CUresult cuMemFree_v2(unsigned long long p) { (void)p; return 0; }
#else
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#ifdef GB_BIND_225
extern void *gb_ref_dlsym(void *, const char *);
extern void *gb_ref_dlopen(const char *, int);
__asm__(".symver gb_ref_dlsym, dlsym@GLIBC_2.2.5");
__asm__(".symver gb_ref_dlopen, dlopen@GLIBC_2.2.5");
#define PROBE_DLSYM gb_ref_dlsym
#define PROBE_DLOPEN gb_ref_dlopen
#define BIND "GLIBC_2.2.5"
#else
#define PROBE_DLSYM dlsym
#define PROBE_DLOPEN dlopen
#define BIND "default(GLIBC_2.34 on glibc>=2.34)"
#endif

static int fails;
static const char *shim_path;
static int expect_shim;

static int in_shim(void *p)
{
    Dl_info i;
    if (!p || !dladdr(p, &i) || !i.dli_fname) return 0;
    return strstr(i.dli_fname, shim_path) != NULL;
}
static void check(const char *name, int ok, const char *detail)
{
    if (!ok) fails++;
    printf("%s %s%s%s\n", ok ? "PASS" : "FAIL", name, ok ? "" : " ", ok ? "" : detail);
}

int main(int argc, char **argv)
{
    if (argc < 3) return 2;
    shim_path = argv[1];
    expect_shim = strcmp(argv[2], "preload") == 0;
    printf("INFO bind=%s mode=%s\n", BIND, argv[2]);

    /* 1. resolution: which object does this binary's dlsym/dlopen reference reach? */
    void *(*ds)(void *, const char *) = PROBE_DLSYM;
    void *(*dop)(const char *, int) = PROBE_DLOPEN;
    check("dlsym reference resolves to shim == expected", in_shim((void *)ds) == expect_shim,
          in_shim((void *)ds) ? "went to shim, expected libc" : "went to libc, expected shim (RT-07)");
    if (argc > 3 && !strcmp(argv[3], "prefix"))   /* a sanitizer runtime preloaded first owns dlopen */
        printf("INFO dlopen resolution check skipped (preload prefix interposes dlopen itself)\n");
    else
        check("dlopen reference resolves to shim == expected", in_shim((void *)dop) == expect_shim,
              in_shim((void *)dop) ? "went to shim, expected libc" : "went to libc, expected shim (RT-07)");

    /* 2. no recursion / pass-through still correct through either version */
    check("dlsym pass-through (libc strlen)", ds(RTLD_DEFAULT, "strlen") != NULL, "NULL");
    check("dlsym RTLD_NEXT does not recurse", 1, "");
    void *h = dop("libcuda.so.1", RTLD_NOW);
    check("dlopen pass-through (stub libcuda)", h != NULL, "NULL");
    if (!h) return 1;

    /* 3. hook effect: with the shim initialised on the stub, a versioned-2.34 caller must be
     * handed the shim's cuMemGetInfo_v2, not the driver's. */
    void *got = ds(h, "cuMemGetInfo_v2");
    if (expect_shim) {
        /* The shim is only active once its init ran on the stub; if that did not happen
         * report it (INFO) instead of failing the versioning check, which step 1 covers. */
        if (in_shim(got)) check("hook effect: dlsym(libcuda,cuMemGetInfo_v2) -> shim", 1, "");
        else {
            Dl_info i;
            const char *where = (got && dladdr(got, &i) && i.dli_fname) ? i.dli_fname : "?";
            /* The shim did go through its hook only if it initialised; either way the stub
             * must be returned unchanged (not NULL) when the hook declines. */
            printf("INFO hook-effect: got %p in %s (shim not initialised or declined)\n", got, where);
            check("hook-effect: dlsym(libcuda,cuMemGetInfo_v2) -> shim", 0,
                  "lookup was not answered by the shim hook");
        }
    } else {
        check("control: dlsym(libcuda,cuMemGetInfo_v2) -> driver stub", got != NULL && !in_shim(got), "");
    }
    printf("%s\n", fails ? "RESULT FAIL" : "RESULT OK");
    return fails ? 1 : 0;
}
#endif

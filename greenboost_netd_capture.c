/* greenboost_netd_capture.c , __cudaRegisterFunction interposer for the feeder.
 *
 * WHY: omen's libggml-cuda.so is stripped , its __global__ kernel stubs are in
 * neither .dynsym nor .symtab, so greenboost-netd's dlsym can never resolve a
 * kernel by name for remote dispatch.  But the library, at dlopen, calls
 * __cudaRegisterFunction(&hostStub, deviceName, ...) for every kernel , exactly
 * the name→stub map we need.  LD_PRELOAD this into netd and it records that map;
 * netd then launches by the captured host stub via the RUNTIME cudaLaunchKernel.
 *
 * Build:  gcc -shared -fPIC -O2 -o libgreenboost_netd_capture.so \
 *              greenboost_netd_capture.c -ldl -lpthread
 * Use:    LD_PRELOAD=.../libgreenboost_netd_capture.so greenboost-netd ...
 *         (greenboost-netd re-execs itself with this set; see netd main()).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>

#define GB_CAP_MAX 65536

struct gb_cap_entry {
    const char *host_fun;   /* the stub pointer cudaLaunchKernel wants */
    const char *name;       /* device symbol name (mangled), strdup'd */
};

static struct gb_cap_entry g_cap[GB_CAP_MAX];
static int g_cap_n = 0;
static int g_cap_full_warned = 0;
static pthread_mutex_t g_cap_lock = PTHREAD_MUTEX_INITIALIZER;

typedef void (*pfn_reg)(void **, const char *, char *, const char *,
                        int, void *, void *, void *, void *, int *);

/* Interpose the CUDA runtime's kernel registration.  cudaLaunchKernel keys on
 * the FIRST argument (hostFun); __cudaRegisterFunction hands us that exact
 * pointer alongside the device symbol name. */
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, void *tid, void *bid,
                            void *bDim, void *gDim, int *wSize)
{
    static pfn_reg real = NULL;
    if (!real) real = (pfn_reg)dlsym(RTLD_NEXT, "__cudaRegisterFunction");

    if (deviceName && hostFun) {
        pthread_mutex_lock(&g_cap_lock);
        if (g_cap_n < GB_CAP_MAX) {
            g_cap[g_cap_n].host_fun = hostFun;
            g_cap[g_cap_n].name     = strdup(deviceName);
            g_cap_n++;
        } else if (!g_cap_full_warned) {
            /* One-shot warning , past this point kernel names silently stop
             * being captured, which would otherwise look like a mysterious
             * "kernel not found" on the netd side with no clue why. */
            fprintf(stderr, "[netd-capture] capture table full at %d entries , "
                            "further kernels will not resolve by name\n", GB_CAP_MAX);
            g_cap_full_warned = 1;
        }
        pthread_mutex_unlock(&g_cap_lock);
    }

    if (real)
        real(fatCubinHandle, hostFun, deviceFun, deviceName,
             thread_limit, tid, bid, bDim, gDim, wSize);
}

/* Called by greenboost-netd (same process) to resolve a kernel name to the
 * host stub cudaLaunchKernel accepts.  Exported; netd finds it via
 * dlsym(RTLD_DEFAULT, "gb_capture_lookup"). */
const void *gb_capture_lookup(const char *name)
{
    if (!name) return NULL;
    const void *r = NULL;
    pthread_mutex_lock(&g_cap_lock);
    for (int i = 0; i < g_cap_n; i++) {
        if (strcmp(g_cap[i].name, name) == 0) { r = g_cap[i].host_fun; break; }
    }
    pthread_mutex_unlock(&g_cap_lock);
    return r;
}

int gb_capture_count(void)
{
    return g_cap_n;
}

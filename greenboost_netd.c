/* SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
 * GreenBoost v3.2 - Network Feeder Daemon (greenboost-netd)
 *
 * Exposes local GPU(s) + system RAM to remote GreenBoost hosts over TCP.
 * Phase 1: handshake, heartbeat, GPU info query, memory info.
 * Phase 2: cudaMalloc/cudaFree/cudaMemcpy/cuMemHostRegister forwarding.
 * Phase 3: cuLaunchKernel/cuLaunchCooperativeKernel forwarding.
 *
 * Usage:
 *   greenboost-netd                  foreground mode (ctrl-C to stop)
 *   greenboost-netd -d               daemonize
 *   greenboost-netd -p 9741          custom port
 *   greenboost-netd --bind 10.0.0.1  bind to specific interface
 *
 * Author  : Ferran Duarri
 * License : GPL v2 (open-source) / Commercial - see LICENSE
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <stdatomic.h>
#include <time.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <dlfcn.h>
#include <sys/sysinfo.h>
#include <sys/mman.h>
#include <stdarg.h>
#include <sys/syscall.h>
#include <sys/wait.h>   /* PR-X: waitpid for the dmidecode fork() pattern */
#include <stdint.h>

#ifdef GREENBOOST_USE_NCCL
#include <nccl.h>
#endif

#include "features/net_fabric.h"

#ifdef GB_HAVE_ZSTD
#include <zstd.h>
/* When built with libzstd the feeder advertises GB_NET_FEAT_ZSTD and
 * transparently decompresses any H2D payload the host marks with
 * GB_NET_FLAG_COMP_ZSTD. GB_NET_COMPRESS=0 on the feeder disables the
 * advertisement (kill switch), so the host then sends raw. */
static int gb_netd_zstd_advertise(void)
{
    const char *e = getenv("GB_NET_COMPRESS");
    return (e && e[0] == '0') ? 0 : 1;
}
#endif

/* ── PSK authentication helpers (F-L3-01) ───────────────────────────
 * Shared secret in /etc/greenboost/cluster.key (hex-encoded 32 bytes).
 * If the file is absent auth is disabled (backward-compat mode).
 * HMAC-SHA256 is computed using a self-contained implementation that
 * requires no OpenSSL dependency in the netd binary.                  */

/* SHA-256 - portable, self-contained (RFC 6234 reference algorithm) */
#define GB_SHA256_BLOCK 64
#define GB_SHA256_DIGEST 32

typedef struct {
    uint32_t state[8];
    uint64_t count;
    uint8_t  buf[GB_SHA256_BLOCK];
} gb_sha256_ctx;

static const uint32_t gb_sha256_K[64] = {
    0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,
    0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
    0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,
    0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
    0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,
    0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
    0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,
    0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
    0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,
    0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
    0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,
    0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
    0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,
    0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
    0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,
    0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U,
};

#define GB_ROR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define GB_CH(e,f,g)  (((e)&(f))^(~(e)&(g)))
#define GB_MAJ(a,b,c) (((a)&(b))^((a)&(c))^((b)&(c)))
#define GB_EP0(a)  (GB_ROR32(a,2)^GB_ROR32(a,13)^GB_ROR32(a,22))
#define GB_EP1(e)  (GB_ROR32(e,6)^GB_ROR32(e,11)^GB_ROR32(e,25))
#define GB_SIG0(x) (GB_ROR32(x,7)^GB_ROR32(x,18)^((x)>>3))
#define GB_SIG1(x) (GB_ROR32(x,17)^GB_ROR32(x,19)^((x)>>10))

static void gb_sha256_transform(gb_sha256_ctx *ctx, const uint8_t data[64])
{
    uint32_t a,b,c,d,e,f,g,h,t1,t2,m[64];
    for (int i=0,j=0;i<16;i++,j+=4)
        m[i]=((uint32_t)data[j]<<24)|((uint32_t)data[j+1]<<16)|
             ((uint32_t)data[j+2]<<8)|(uint32_t)data[j+3];
    for (int i=16;i<64;i++)
        m[i]=GB_SIG1(m[i-2])+m[i-7]+GB_SIG0(m[i-15])+m[i-16];
    a=ctx->state[0];b=ctx->state[1];c=ctx->state[2];d=ctx->state[3];
    e=ctx->state[4];f=ctx->state[5];g=ctx->state[6];h=ctx->state[7];
    for (int i=0;i<64;i++){
        t1=h+GB_EP1(e)+GB_CH(e,f,g)+gb_sha256_K[i]+m[i];
        t2=GB_EP0(a)+GB_MAJ(a,b,c);
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    ctx->state[0]+=a;ctx->state[1]+=b;ctx->state[2]+=c;ctx->state[3]+=d;
    ctx->state[4]+=e;ctx->state[5]+=f;ctx->state[6]+=g;ctx->state[7]+=h;
}

static void gb_sha256_init(gb_sha256_ctx *ctx)
{
    ctx->count=0;
    ctx->state[0]=0x6a09e667U;ctx->state[1]=0xbb67ae85U;
    ctx->state[2]=0x3c6ef372U;ctx->state[3]=0xa54ff53aU;
    ctx->state[4]=0x510e527fU;ctx->state[5]=0x9b05688cU;
    ctx->state[6]=0x1f83d9abU;ctx->state[7]=0x5be0cd19U;
}

static void gb_sha256_update(gb_sha256_ctx *ctx, const uint8_t *data, size_t len)
{
    size_t i=0;
    uint32_t idx=(uint32_t)(ctx->count & 63);
    ctx->count+=len;
    for(;i<len;i++){
        ctx->buf[idx++]=data[i];
        if(idx==64){gb_sha256_transform(ctx,ctx->buf);idx=0;}
    }
}

static void gb_sha256_final(gb_sha256_ctx *ctx, uint8_t hash[32])
{
    uint32_t idx=(uint32_t)(ctx->count & 63);
    ctx->buf[idx++]=0x80;
    if(idx>56){
        while(idx<64)ctx->buf[idx++]=0;
        gb_sha256_transform(ctx,ctx->buf);idx=0;
    }
    while(idx<56)ctx->buf[idx++]=0;
    uint64_t bits=ctx->count*8;
    for(int i=7;i>=0;i--){ctx->buf[56+(7-i)]=(uint8_t)(bits>>(i*8));}
    gb_sha256_transform(ctx,ctx->buf);
    for(int i=0;i<8;i++){
        hash[i*4]  =(uint8_t)(ctx->state[i]>>24);
        hash[i*4+1]=(uint8_t)(ctx->state[i]>>16);
        hash[i*4+2]=(uint8_t)(ctx->state[i]>>8);
        hash[i*4+3]=(uint8_t)(ctx->state[i]);
    }
}

/* HMAC-SHA256: key is always 32 bytes (PSK) */
static void gb_hmac_sha256(const uint8_t key[32], const uint8_t *msg, size_t msg_len,
                           uint8_t out[32])
{
    uint8_t k_ipad[GB_SHA256_BLOCK], k_opad[GB_SHA256_BLOCK];
    memset(k_ipad, 0x36, GB_SHA256_BLOCK);
    memset(k_opad, 0x5c, GB_SHA256_BLOCK);
    for (int i = 0; i < 32; i++) {
        k_ipad[i] ^= key[i];
        k_opad[i] ^= key[i];
    }
    uint8_t inner[32];
    gb_sha256_ctx c;
    gb_sha256_init(&c);
    gb_sha256_update(&c, k_ipad, GB_SHA256_BLOCK);
    gb_sha256_update(&c, msg, msg_len);
    gb_sha256_final(&c, inner);
    gb_sha256_init(&c);
    gb_sha256_update(&c, k_opad, GB_SHA256_BLOCK);
    gb_sha256_update(&c, inner, 32);
    gb_sha256_final(&c, out);
    /* PR-HH: wipe HMAC ipad/opad (each XORed with the PSK) and the inner
     * SHA-256 state.  PR-CC's PSK-on-stack hardening covered the auth-block
     * locals but missed this hot HMAC primitive used on the v3 default path. */
    explicit_bzero(k_ipad, sizeof(k_ipad));
    explicit_bzero(k_opad, sizeof(k_opad));
    explicit_bzero(inner, sizeof(inner));
}

/* PR-EE: HKDF-SHA256 (RFC 5869) + mutual-auth opt-in via GREENBOOST_PSK_V4=1.
 *
 * Threat model: v3 is one-way (server proves it knows PSK to the client by
 * accepting; client proves it to the server by computing the right MAC).
 * A MITM that captured one prior handshake gains nothing extra (PSK is
 * needed to forge), but operator confidence is increased when both ends
 * actively prove knowledge of the PSK on EVERY handshake.
 *
 * v4 handshake (when both sides set GREENBOOST_PSK_V4=1):
 *   1. Server sends nonce_s (32 bytes, getrandom)
 *   2. Client sends nonce_c (32 bytes) || mac1 = HMAC(psk, nonce_s||nonce_c)
 *   3. Server verifies mac1, sends mac2 = HMAC(psk, nonce_c||nonce_s)
 *   4. Client verifies mac2
 *   5. Both derive session_key = HKDF(psk, nonce_s||nonce_c, "gb-session-v1|proto=4")
 *
 * Backward compat: v3 clients/servers continue to interop with each other.
 * A v4 server detects v4 clients by the env-var-controlled greeting flow.
 *
 * The session key is derived but not yet consumed (no per-message MAC
 * appended).  Follow-up work would add an 8-byte truncated MAC to each
 * gb_net_header for replay-resistance across the entire connection.
 *
 * Variable-length-key HMAC wrapper used by HKDF Extract.  Callers pass
 * nonces (typically 64 bytes = nonce_s||nonce_c) as salt. */
static void gb_hmac_sha256_salt(const uint8_t *key, size_t key_len,
                                 const uint8_t *msg, size_t msg_len,
                                 uint8_t out[32])
{
    /* Generic HMAC where key length is arbitrary (RFC 2104).  If key is
     * longer than the block size (64 B for SHA-256), hash it first; if
     * shorter, zero-pad.  Our usage always passes a 32-byte key so neither
     * branch runs; provided for future flexibility. */
    uint8_t k_pad[GB_SHA256_BLOCK] = {0};
    if (key_len > GB_SHA256_BLOCK) {
        gb_sha256_ctx c;
        gb_sha256_init(&c);
        gb_sha256_update(&c, key, key_len);
        gb_sha256_final(&c, k_pad);  /* k_pad first 32 bytes hold the hash */
    } else {
        memcpy(k_pad, key, key_len);
    }
    uint8_t k_ipad[GB_SHA256_BLOCK], k_opad[GB_SHA256_BLOCK];
    for (int i = 0; i < GB_SHA256_BLOCK; i++) {
        k_ipad[i] = k_pad[i] ^ 0x36;
        k_opad[i] = k_pad[i] ^ 0x5c;
    }
    uint8_t inner[32];
    gb_sha256_ctx c;
    gb_sha256_init(&c);
    gb_sha256_update(&c, k_ipad, GB_SHA256_BLOCK);
    gb_sha256_update(&c, msg, msg_len);
    gb_sha256_final(&c, inner);
    gb_sha256_init(&c);
    gb_sha256_update(&c, k_opad, GB_SHA256_BLOCK);
    gb_sha256_update(&c, inner, 32);
    gb_sha256_final(&c, out);
    explicit_bzero(k_pad, sizeof(k_pad));
    explicit_bzero(k_ipad, sizeof(k_ipad));
    explicit_bzero(k_opad, sizeof(k_opad));
    explicit_bzero(inner, sizeof(inner));
}

/* HKDF-Extract using the variable-key HMAC.  This replaces the in-band
 * call inside gb_hkdf_sha256 because that version passed a fixed-size
 * key to gb_hmac_sha256 which requires 32-byte keys. */
static void gb_hkdf_extract(const uint8_t *salt, size_t salt_len,
                             const uint8_t *ikm, size_t ikm_len,
                             uint8_t prk[32])
{
    static const uint8_t zero_salt[32] = {0};
    if (!salt || !salt_len) { salt = zero_salt; salt_len = 32; }
    gb_hmac_sha256_salt(salt, salt_len, ikm, ikm_len, prk);
}

/* Convenience: derive a 32-byte session key from (psk, nonce_s, nonce_c)
 * with a context-binding info string.  Used by both netd and netc. */
static void gb_derive_session_key(const uint8_t psk[32],
                                   const uint8_t nonce_s[32],
                                   const uint8_t nonce_c[32],
                                   uint8_t session_key[32])
{
    /* Salt = nonce_s || nonce_c (64 bytes) */
    uint8_t salt[64];
    memcpy(salt,       nonce_s, 32);
    memcpy(salt + 32,  nonce_c, 32);
    uint8_t prk[32];
    gb_hkdf_extract(salt, sizeof(salt), psk, 32, prk);
    /* Expand: info = "gb-session-v1|proto=4" */
    static const uint8_t info[] = "gb-session-v1|proto=4";
    uint8_t info_buf[sizeof(info)];
    memcpy(info_buf, info, sizeof(info) - 1);
    info_buf[sizeof(info) - 1] = 0x01;  /* T(1) counter */
    gb_hmac_sha256_salt(prk, 32, info_buf, sizeof(info_buf), session_key);
    explicit_bzero(salt, sizeof(salt));
    explicit_bzero(prk, sizeof(prk));
    explicit_bzero(info_buf, sizeof(info_buf));
}

/* Constant-time comparison - prevents timing side-channel on MAC verify */
static int gb_consttime_memcmp(const void *a, const void *b, size_t n)
{
    const volatile uint8_t *p = (const volatile uint8_t *)a;
    const volatile uint8_t *q = (const volatile uint8_t *)b;
    uint8_t diff = 0;
    for (size_t i = 0; i < n; i++) diff |= p[i] ^ q[i];
    return diff != 0;
}

/* Forward declaration: netd_log is defined later in this TU but
 * gb_load_psk uses it for hardening diagnostics (PR-G). */
static void netd_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

/* Load 32-byte PSK from /etc/greenboost/cluster.key (hex-encoded).
 * Returns 0 on success, -1 if key file absent or malformed.
 *
 * PR-G hardening:
 *   - Accept mode 0600/0400 (root-only) or 0640 root:greenboost (group read).
 *     Group read allows non-root cluster members (greenboost group) to read the
 *     key; group write/exec and any world access are still rejected.
 *   - Validate exactly 64 hex chars (32 bytes).  The previous loop would happily
 *     accept a partial keyfile and silently read leading-zero garbage from the
 *     zero-initialised hex[] buffer, yielding a low-entropy PSK that both
 *     daemons would "agree" on.
 *   - explicit_bzero the local hex[] buffer on every exit so the secret does
 *     not linger in stack memory until the frame is overwritten. */
static int gb_load_psk(uint8_t key[32])
{
    const char *path = "/etc/greenboost/cluster.key";
    struct stat st;
    if (stat(path, &st) != 0) return -1; /* no key file - auth disabled */
    if (!S_ISREG(st.st_mode)) {
        netd_log("AUTH: %s is not a regular file - refusing to load PSK", path);
        return -1;
    }
    if (gb_check_keyfile_mode(&st, path) != 0) {
        netd_log("AUTH: %s has insecure mode 0%o - must be 0600 (root-only) "
                 "or 0640 root:%s", path, st.st_mode & 07777, GB_KEYFILE_GRP);
        return -1;
    }
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char hex[65] = {0};
    size_t n = fread(hex, 1, 64, f);
    fclose(f);
    if (n != 64) {
        netd_log("AUTH: %s is %zu bytes; expected exactly 64 hex chars", path, n);
        explicit_bzero(hex, sizeof(hex));
        return -1;
    }
    for (int i = 0; i < 64; i++) {
        char c = hex[i];
        if (!((c >= '0' && c <= '9') ||
              (c >= 'a' && c <= 'f') ||
              (c >= 'A' && c <= 'F'))) {
            netd_log("AUTH: %s contains non-hex character at offset %d", path, i);
            explicit_bzero(hex, sizeof(hex));
            return -1;
        }
    }
    for (int i = 0; i < 32; i++) {
        unsigned int b;
        if (sscanf(&hex[i*2], "%02x", &b) != 1) {
            explicit_bzero(hex, sizeof(hex));
            return -1;
        }
        key[i] = (uint8_t)b;
    }
    explicit_bzero(hex, sizeof(hex));
    return 0;
}

/* Blocking recv loop - used only during the PSK handshake phase
 * (socket is still blocking at that point).                          */
static ssize_t gb_recv_all_blocking(int fd, void *buf, size_t len)
{
    uint8_t *p = (uint8_t *)buf;
    size_t got = 0;
    while (got < len) {
        ssize_t n = recv(fd, p + got, len - got, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1; /* connection closed */
        got += (size_t)n;
    }
    return (ssize_t)got;
}

/* ── Always-on event log for feeder-side diagnostics ────────────────
 * Writes structured records to /run/greenboost/nvtx_events.log
 * (same file the shim writes to, differentiated by "NETD" prefix).  */
static int g_netd_log_fd = -2;

static void netd_nvtx_event(const char *type, const char *tier,
                            size_t size_mb, uintptr_t ptr, const char *detail)
{
    if (g_netd_log_fd == -2) {
        g_netd_log_fd = open("/run/greenboost/nvtx_events.log",
                             O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
        if (g_netd_log_fd < 0)
            g_netd_log_fd = open("/tmp/greenboost_nvtx_events.log",
                                 O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    }
    if (g_netd_log_fd < 0) return;
    struct timespec _ts; clock_gettime(CLOCK_REALTIME, &_ts);
    uint64_t ms = (uint64_t)_ts.tv_sec * 1000ULL + (uint64_t)_ts.tv_nsec / 1000000ULL;
    char buf[288];
    int n = snprintf(buf, sizeof(buf), "%llu NETD %-24s %-12s %6zuMB ptr=0x%012lx %s\n",
                     (unsigned long long)ms, type, tier, size_mb, ptr, detail);
    if (n > 0 && n < (int)sizeof(buf)) {
        ssize_t _wr = write(g_netd_log_fd, buf, (size_t)n); (void)_wr;
    }
}
#define NETD_EVT(type, tier, size_mb, ptr, detail) \
    netd_nvtx_event((type), (tier), (size_mb), (uintptr_t)(ptr), (detail))

/* ------------------------------------------------------------------ */
/*  CUDA runtime types (loaded via dlopen - no compile-time dep)       */
/* ------------------------------------------------------------------ */

typedef int cudaError_t;
typedef int CUresult;
typedef int CUdevice;
typedef void *cudaStream_t;

typedef cudaError_t (*pfn_cudaGetDeviceCount)(int *);
typedef cudaError_t (*pfn_cudaSetDevice)(int);
typedef cudaError_t (*pfn_cudaMalloc)(void **, size_t);
typedef cudaError_t (*pfn_cudaFree)(void *);
typedef cudaError_t (*pfn_cudaHostAlloc)(void **, size_t, unsigned int);
typedef cudaError_t (*pfn_cudaFreeHost)(void *);
typedef cudaError_t (*pfn_cudaMemcpy)(void *, const void *, size_t, int);
typedef cudaError_t (*pfn_cudaMemset)(void *, int, size_t);
typedef cudaError_t (*pfn_cudaGetLastError)(void);
typedef cudaError_t (*pfn_cudaDeviceSynchronize)(void);
typedef cudaError_t (*pfn_cudaStreamCreate)(cudaStream_t *);
typedef cudaError_t (*pfn_cudaStreamSynchronize)(cudaStream_t);
typedef cudaError_t (*pfn_cudaStreamDestroy)(cudaStream_t);

/* cudaMemcpyKind */
#define GB_cudaMemcpyHostToDevice   1
#define GB_cudaMemcpyDeviceToHost   2




/* NVML types */
typedef void *nvmlDevice_t;
typedef int nvmlReturn_t;
typedef nvmlReturn_t (*pfn_nvmlDeviceGetName)(nvmlDevice_t, char*, unsigned int);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetCudaComputeCapability)(nvmlDevice_t, int*, int*);
typedef struct { unsigned long long total, free, used; } nvmlMemory_t;
typedef nvmlReturn_t (*pfn_nvmlInit)(void);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t *);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t *);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetUtilizationRates)(nvmlDevice_t, void *);

typedef struct { unsigned int gpu; unsigned int memory; } nvmlUtilization_t;

/* v2 NVML - temperature, power, throttle, ECC */
typedef nvmlReturn_t (*pfn_nvmlDeviceGetTemperature)(nvmlDevice_t, unsigned int, unsigned int *);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetPowerUsage)(nvmlDevice_t, unsigned int *);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetEnforcedPowerLimit)(nvmlDevice_t, unsigned int *);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetCurrentClocksThrottleReasons)(nvmlDevice_t, unsigned long long *);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetTotalEccErrors)(nvmlDevice_t, unsigned int, unsigned int, unsigned long long *);
#define NVML_TEMPERATURE_GPU        0
#define NVML_MEMORY_ERROR_TYPE_UNCORRECTED 1
#define NVML_AGGREGATE_ECC          2
#define NVML_VOLATILE_ECC           1

/* ------------------------------------------------------------------ */
/*  Global state                                                       */
/* ------------------------------------------------------------------ */

#define MAX_CLIENTS  32
#define EPOLL_EVENTS 64
/* Must match netc NETC_RECV_BUF via GB_NET_MAX_MSG_SIZE in net_fabric.h -
 * asymmetric caps cause framing desync (audit F-L3-07).
 *
 * PR-HH: +16 slack accommodates the 8-byte MAC suffix that v4 mode (PR-FF)
 * appends after the payload, plus the 16-byte header.  Without this, a
 * payload of exactly GB_NET_MAX_MSG_SIZE would push total wire bytes past
 * the buffer in v4 mode → client_process returns -1 → connection dropped.
 * The cap on PAYLOAD bytes remains GB_NET_MAX_MSG_SIZE; the slack is
 * protocol overhead only. */
#define RECV_BUF_SIZE  (GB_NET_MAX_MSG_SIZE + GB_NET_HDR_SIZE + 8)

static volatile sig_atomic_t g_running = 1;
/* U8: graceful drain - set on SIGTERM; new MALLOC/EXEC requests return busy */
static _Atomic int           g_draining = 0;
/* U8: count of in-flight CUDA_MALLOC / CUDA_EXEC operations */
static _Atomic int           g_inflight_ops = 0;
static int   g_port       = GB_NET_PORT;
static int   g_daemonize  = 0;
static char  g_bind_addr[64] = "0.0.0.0";
static FILE *g_logfp      = NULL;

/* Local GPU info */
static int                     g_gpu_count = 0;
static struct gb_net_gpu_info  g_gpus[GB_NET_MAX_GPUS];
static char                    g_hostname[GB_NET_MAX_HOSTNAME];

/* CUDA function pointers */
static pfn_cudaGetDeviceCount      f_cudaGetDeviceCount;
static pfn_cudaSetDevice           f_cudaSetDevice;
static pfn_cudaMalloc              f_cudaMalloc;
static pfn_cudaFree                f_cudaFree;
static pfn_cudaStreamCreate        f_cudaStreamCreate;
static pfn_cudaStreamSynchronize   f_cudaStreamSynchronize;
static pfn_cudaStreamDestroy       f_cudaStreamDestroy;
static pfn_cudaHostAlloc           f_cudaHostAlloc;
static pfn_cudaFreeHost            f_cudaFreeHost;
static pfn_cudaMemcpy              f_cudaMemcpy;
static pfn_cudaMemset              f_cudaMemset;
static pfn_cudaGetLastError        f_cudaGetLastError;
static pfn_cudaDeviceSynchronize   f_cudaDeviceSynchronize;


/* NVML function pointers */
static pfn_nvmlInit                       f_nvmlInit;
static pfn_nvmlDeviceGetHandleByIndex     f_nvmlGetHandle;
static pfn_nvmlDeviceGetMemoryInfo        f_nvmlMemInfo;
static pfn_nvmlDeviceGetUtilizationRates  f_nvmlUtilRates;
static pfn_nvmlDeviceGetName                f_nvmlGetName;
static pfn_nvmlDeviceGetCudaComputeCapability f_nvmlGetCC;
static pfn_nvmlDeviceGetTemperature              f_nvmlGetTemp;
static pfn_nvmlDeviceGetPowerUsage               f_nvmlGetPower;
static pfn_nvmlDeviceGetEnforcedPowerLimit       f_nvmlGetPowerLimit;
static pfn_nvmlDeviceGetCurrentClocksThrottleReasons f_nvmlGetThrottle;
static pfn_nvmlDeviceGetTotalEccErrors           f_nvmlGetEcc;
static nvmlDevice_t g_nvml_devices[GB_NET_MAX_GPUS];

/* ECC SBE baseline per GPU - delta reported in heartbeat */
static unsigned long long g_ecc_sbe_baseline[GB_NET_MAX_GPUS];

/* Per-client state */
struct client {
    int      fd;
    int      active;
    uint32_t feeder_id;
    char     remote_addr[64];
    uint64_t last_heartbeat_ms;
    uint8_t *recv_buf;
    size_t   recv_len;
    size_t   recv_cap;
    /* F-L3-09: per-connection sequence counters for within-session replay detection */
    uint32_t send_seq;   /* next seq_num to embed in outgoing header        */
    uint32_t recv_seq;   /* next seq_num expected from the client            */
    /* N9: per-client token-bucket rate limiter for GB_MSG_CUDA_MALLOC.
     * Bucket refills at 1000 tokens/s; size 200 tokens (200 ms burst).
     * If empty, malloc is rejected with GB_STATUS_ERR_THROTTLE. */
    uint64_t alloc_tokens;      /* current token count (fractional, ×1000 for precision) */
    uint64_t last_refill_ms;    /* mono_ms of last token refill                          */
    /* PR-FF: per-message MAC.  Populated after successful v4 handshake
     * (GREENBOOST_PSK_V4=1); zeroed in v3 mode.  When mac_enabled is 1,
     * every framed message is sent with an 8-byte truncated HMAC of the
     * (header || payload || send_seq) using mac_session_key, appended
     * after the payload on the wire.  Receiver verifies before
     * dispatching the message; mismatch closes the connection.
     * Closes replay-across-reconnect: a captured handshake's session_key
     * is unique to that handshake's nonces, so MACs cannot be reused. */
    int      mac_enabled;
    uint8_t  mac_session_key[32];
    /* PR-QQ: per-client send mutex.  send_msg does three or four sequential
     * send_all calls (hdr, payload, [mac], optional [data]).  Single-threaded
     * netd doesn't need this, but a future worker pool (made safe by PR-II's
     * lock-holding) could have two workers replying to the same client and
     * interleave bytes mid-frame - recipient would see a spliced header from
     * one message and a payload from another.  The mutex serialises
     * send_msg per client.  Uncontended cost: one atomic load + branch. */
    pthread_mutex_t send_lock;
    /* Phase 2b: persistent CUDA stream for async exec (GB_MSG_CUDA_EXEC_ASYNC).
     * Kernels are enqueued here without blocking the TCP handler; host drains
     * this stream implicitly via GB_MSG_CUDA_SYNC (cudaDeviceSynchronize). */
    cudaStream_t async_stream;
    /* Fabric zstd negotiated with this client (GB_NET_FEAT_ZSTD echoed at
     * handshake). H2D decompress is per-message, but this records the peer's
     * capability for any compressed response direction. */
    int      feat_zstd;
    /* Peer understands the dma-buf compressed-content descriptor
     * (GB_NET_FEAT_DMABUF_COMPRESSION, see features/net_fabric.h). Distinct
     * from feat_zstd , this is about relaying a per-buffer codec
     * descriptor, not compressing this protocol's own wire payloads. */
    int      feat_dmabuf_compression;
#ifdef GREENBOOST_USE_NCCL
    ncclComm_t nccl_comm;
#endif
};

static struct client g_clients[MAX_CLIENTS];
static uint32_t g_next_feeder_id = 1;
/* N7: total kernel dispatches since daemon start - exposed via GB_MSG_FEEDER_STATUS */
static uint32_t g_kernel_dispatch_count = 0;
/* F-L3-08: per-boot random salt XOR'd into every allocation handle so handles
 * are unpredictable - prevents cross-tenant handle guessing via brute-force. */
static uint64_t g_handle_salt = 0;
static _Atomic uint64_t g_handle_counter = 1;

/* ------------------------------------------------------------------ */
/*  U10: KV block ancestry table for feeder-side prefetch planning     */
/* ------------------------------------------------------------------ */
#define GB_ANCESTRY_MAX 1024

typedef struct {
    uint64_t hash;         /* block hash (0 = empty slot)          */
    uint64_t parent_hash;  /* parent block hash (0 = root)         */
    uint32_t num_tokens;   /* cumulative token count               */
    uint32_t evict_ts;     /* 0 = present; non-zero = evicted time */
} gb_ancestry_entry_t;

static gb_ancestry_entry_t g_ancestry[GB_ANCESTRY_MAX];
static pthread_mutex_t     g_ancestry_lock = PTHREAD_MUTEX_INITIALIZER;

/* ------------------------------------------------------------------ */
/*  Logging                                                            */
/* ------------------------------------------------------------------ */

/* Forward decls , defined alongside the allowlist further down. */
static void *gb_kernel_resolve(const char *kname);
static void *gb_fatbin_resolve(const char *name);
static int   gb_kernel_name_allowed(const char *kname);
/* Rotation for the long-lived daemon log , open_log() itself is defined
 * much later in this TU (alongside write_pid_file/remove_pid_file), this
 * forward decl lets netd_log() below trigger a mid-run rotation. */
static int open_log(void);

static uint64_t mono_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

static void netd_log(const char *fmt, ...)
    __attribute__((format(printf, 1, 2)));

static void netd_log(const char *fmt, ...)
{
    /* This daemon is meant to run for months uninterrupted (confirmed live
     * 2026-07-27: a feeder's netd.log had grown past 1 GB since first boot,
     * no rotation ever existed). open_log()'s startup rotation alone can't
     * catch that — check on every write instead. ftell() is a cheap,
     * syscall-free read of the FILE*'s tracked position (kept in sync by
     * the fflush() at the end of every prior call), so this doesn't add a
     * stat()-per-log-line cost. */
    if (g_logfp && ftell(g_logfp) >= (long)GB_NETD_LOG_MAX_BYTES) {
        fclose(g_logfp);
        g_logfp = NULL;
        open_log();  /* rotates GB_NETD_LOG_FILE -> .1, reopens fresh */
    }

    FILE *out = g_logfp ? g_logfp : stderr;
    time_t t = time(NULL);
    struct tm tm;
    localtime_r(&t, &tm);

    fprintf(out, "[%04d-%02d-%02d %02d:%02d:%02d] [greenboost-netd] ",
            tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
            tm.tm_hour, tm.tm_min, tm.tm_sec);

    va_list ap;
    va_start(ap, fmt);
    vfprintf(out, fmt, ap);
    va_end(ap);

    fprintf(out, "\n");
    fflush(out);
}

/* ------------------------------------------------------------------ */
/*  GPU probing via dlopen                                             */
/* ------------------------------------------------------------------ */

static int probe_gpus(void)
{
    void *libcudart = NULL;
    const char *cudart_paths[] = {
        /* gb-synapse's engine links the system CUDA runtime directly (verified
         * 2026-07-15 via ldd on libggml-cuda.so/rpc-server: libcudart.so.12 from
         * /usr/lib/x86_64-linux-gnu, supports Blackwell cc 12.0 / RTX 5070 fine) ,
         * no bundled-runtime dependency on Ollama needed anymore. */
        "/usr/local/cuda/lib64/libcudart.so",
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11",
        NULL
    };

    for (int i = 0; cudart_paths[i]; i++) {
        libcudart = dlopen(cudart_paths[i], RTLD_NOW | RTLD_GLOBAL);
        if (libcudart) break;
    }
    if (!libcudart) {
        netd_log("ERROR: cannot load libcudart - %s", dlerror());
        return -1;
    }

    f_cudaGetDeviceCount      = (pfn_cudaGetDeviceCount)dlsym(libcudart, "cudaGetDeviceCount");
    f_cudaSetDevice           = (pfn_cudaSetDevice)dlsym(libcudart, "cudaSetDevice");
    f_cudaMalloc              = (pfn_cudaMalloc)dlsym(libcudart, "cudaMalloc");
    f_cudaFree                = (pfn_cudaFree)dlsym(libcudart, "cudaFree");
    f_cudaHostAlloc           = (pfn_cudaHostAlloc)dlsym(libcudart, "cudaHostAlloc");
    f_cudaFreeHost            = (pfn_cudaFreeHost)dlsym(libcudart, "cudaFreeHost");
    f_cudaMemcpy              = (pfn_cudaMemcpy)dlsym(libcudart, "cudaMemcpy");
    f_cudaMemset              = (pfn_cudaMemset)dlsym(libcudart, "cudaMemset");
    f_cudaGetLastError        = (pfn_cudaGetLastError)dlsym(libcudart, "cudaGetLastError");
    f_cudaDeviceSynchronize   = (pfn_cudaDeviceSynchronize)dlsym(libcudart, "cudaDeviceSynchronize");
    f_cudaStreamCreate        = (pfn_cudaStreamCreate)dlsym(libcudart, "cudaStreamCreate");
    f_cudaStreamSynchronize   = (pfn_cudaStreamSynchronize)dlsym(libcudart, "cudaStreamSynchronize");
    f_cudaStreamDestroy       = (pfn_cudaStreamDestroy)dlsym(libcudart, "cudaStreamDestroy");


    if (!f_cudaGetDeviceCount) {
        netd_log("ERROR: cudaGetDeviceCount not found");
        return -1;
    }

    int count = 0;
    if (f_cudaGetDeviceCount(&count) != 0 || count <= 0) {
        netd_log("ERROR: no CUDA devices found");
        return -1;
    }

    g_gpu_count = count > GB_NET_MAX_GPUS ? GB_NET_MAX_GPUS : count;

    /* NVML for live utilization and properties */
    void *libnvml = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!libnvml) libnvml = dlopen("libnvidia-ml.so", RTLD_NOW);
    if (libnvml) {
        f_nvmlInit      = (pfn_nvmlInit)dlsym(libnvml, "nvmlInit_v2");
        if (!f_nvmlInit) f_nvmlInit = (pfn_nvmlInit)dlsym(libnvml, "nvmlInit");
        f_nvmlGetHandle = (pfn_nvmlDeviceGetHandleByIndex)dlsym(libnvml, "nvmlDeviceGetHandleByIndex_v2");
        if (!f_nvmlGetHandle) f_nvmlGetHandle = (pfn_nvmlDeviceGetHandleByIndex)dlsym(libnvml, "nvmlDeviceGetHandleByIndex");
        f_nvmlMemInfo   = (pfn_nvmlDeviceGetMemoryInfo)dlsym(libnvml, "nvmlDeviceGetMemoryInfo");
        f_nvmlUtilRates = (pfn_nvmlDeviceGetUtilizationRates)dlsym(libnvml, "nvmlDeviceGetUtilizationRates");
        f_nvmlGetName     = (pfn_nvmlDeviceGetName)dlsym(libnvml, "nvmlDeviceGetName");
        f_nvmlGetCC       = (pfn_nvmlDeviceGetCudaComputeCapability)dlsym(libnvml, "nvmlDeviceGetCudaComputeCapability");
        f_nvmlGetTemp     = (pfn_nvmlDeviceGetTemperature)dlsym(libnvml, "nvmlDeviceGetTemperature");
        f_nvmlGetPower    = (pfn_nvmlDeviceGetPowerUsage)dlsym(libnvml, "nvmlDeviceGetPowerUsage");
        f_nvmlGetPowerLimit = (pfn_nvmlDeviceGetEnforcedPowerLimit)dlsym(libnvml, "nvmlDeviceGetEnforcedPowerLimit");
        f_nvmlGetThrottle = (pfn_nvmlDeviceGetCurrentClocksThrottleReasons)dlsym(libnvml, "nvmlDeviceGetCurrentClocksThrottleReasons");
        f_nvmlGetEcc      = (pfn_nvmlDeviceGetTotalEccErrors)dlsym(libnvml, "nvmlDeviceGetTotalEccErrors");

        if (f_nvmlInit && f_nvmlInit() == 0 && f_nvmlGetHandle) {
            for (int i = 0; i < g_gpu_count; i++)
                f_nvmlGetHandle((unsigned int)i, &g_nvml_devices[i]);
            netd_log("NVML initialized - live GPU metrics available");
            /* Capture SBE baseline so heartbeat can report deltas */
            if (f_nvmlGetEcc) {
                for (int i = 0; i < g_gpu_count && i < GB_NET_MAX_GPUS; i++) {
                    if (g_nvml_devices[i]) {
                        unsigned long long sbe = 0;
                        if (f_nvmlGetEcc(g_nvml_devices[i], NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                         NVML_VOLATILE_ECC, &sbe) != 0)
                            sbe = 0;
                        g_ecc_sbe_baseline[i] = sbe;
                    }
                }
            }
        }
    }

    for (int i = 0; i < g_gpu_count; i++) {
        g_gpus[i].gpu_id = (gb_u32)i;
        
        if (f_nvmlMemInfo && g_nvml_devices[i]) {
            nvmlMemory_t mem = {0};
            if (f_nvmlMemInfo(g_nvml_devices[i], &mem) == 0) {
                g_gpus[i].vram_bytes = (gb_u64)mem.total;
            }
        }
        
        if (f_nvmlGetCC && g_nvml_devices[i]) {
            int major = 0, minor = 0;
            if (f_nvmlGetCC(g_nvml_devices[i], &major, &minor) == 0) {
                g_gpus[i].cc_major = (gb_u32)major;
                g_gpus[i].cc_minor = (gb_u32)minor;
            }
        }
        
        if (f_nvmlGetName && g_nvml_devices[i]) {
            f_nvmlGetName(g_nvml_devices[i], g_gpus[i].name, GB_NET_MAX_GPU_NAME);
        } else {
            snprintf(g_gpus[i].name, GB_NET_MAX_GPU_NAME, "NVIDIA GPU %d", i);
        }
        g_gpus[i].name[GB_NET_MAX_GPU_NAME - 1] = '\0';

        /* T2: total system RAM - read MemTotal from /proc/meminfo */
        {
            gb_u64 _mt = 0;
            FILE *_mf = fopen("/proc/meminfo", "r");
            if (_mf) {
                char _ml[128];
                while (fgets(_ml, sizeof(_ml), _mf)) {
                    unsigned long long _kb = 0;
                    if (sscanf(_ml, "MemTotal: %llu kB", &_kb) == 1) { _mt = _kb * 1024ULL; break; }
                }
                fclose(_mf);
            }
            if (!_mt) {
                struct sysinfo _si;
                if (sysinfo(&_si) == 0) _mt = (gb_u64)_si.totalram * (gb_u64)_si.mem_unit;
            }
            g_gpus[i].ram_available_bytes = _mt;
        }

        /* T3: prefer GreenBoost kernel module pool size; fall back to /proc/swaps.
         * /proc/swaps reflects Linux kernel swap (OS default ~7 GB) which is unrelated
         * to the GreenBoost T3 backing file managed by the kernel module. */
        if (i == 0) {
            gb_u64 swap_total = 0;
            FILE *sf = fopen("/sys/module/greenboost/parameters/nvme_pool_gb", "r");
            if (sf) {
                int nvme_gb = 0;
                if (fscanf(sf, "%d", &nvme_gb) == 1 && nvme_gb > 0)
                    swap_total = (gb_u64)nvme_gb * 1024ULL * 1024ULL * 1024ULL;
                fclose(sf);
            }
            if (swap_total == 0) {
                /* Fallback: feeder machines that run without GreenBoost kernel module */
                sf = fopen("/proc/swaps", "r");
                if (sf) {
                    char sline[256];
                    if (fgets(sline, sizeof(sline), sf)) { /* skip header line */ }
                    while (fgets(sline, sizeof(sline), sf)) {
                        char sname[200]; char stype[32];
                        unsigned long long ssz = 0;
                        if (sscanf(sline, "%199s %31s %llu", sname, stype, &ssz) == 3)
                            swap_total += ssz * 1024ULL; /* /proc/swaps reports kB */
                    }
                    fclose(sf);
                }
            }
            for (int j = 0; j < g_gpu_count; j++)
                g_gpus[j].t3_bytes = swap_total;
        }

        netd_log("GPU %d: %s - %llu MB VRAM, cc %u.%u",
                 i, g_gpus[i].name,
                 (unsigned long long)(g_gpus[i].vram_bytes >> 20),
                 g_gpus[i].cc_major, g_gpus[i].cc_minor);
    }

    /* Eager CUDA context warmup - without this, the first cudaMalloc call inside
     * handle_cuda_malloc() tries to create a device context lazily, which fails
     * silently in daemon mode (Blackwell cc 12.x, no display, systemd environment).
     * cudaHostAlloc (T2) works without a device context; cudaMalloc (T1) does not.
     * Warming up here ensures T1 VRAM allocation and cuLaunchKernel dispatch work
     * from the first request. */
    if (f_cudaSetDevice && f_cudaMalloc && f_cudaFree) {
        for (int _gi = 0; _gi < g_gpu_count; _gi++) {
            if (f_cudaSetDevice(_gi) == 0) {
                /* cudaDeviceSynchronize forces context creation on Blackwell cc 12.x
                 * headless daemons where cudaSetDevice alone leaves it lazy-uninitialised */
                if (f_cudaDeviceSynchronize) f_cudaDeviceSynchronize();
                void *_tp = NULL;
                if (f_cudaMalloc(&_tp, 4096) == 0 && _tp) {
                    f_cudaFree(_tp);
                    netd_log("GPU %d: CUDA context ready - T1 VRAM allocation enabled", _gi);
                } else {
                    netd_log("WARN GPU %d: cudaMalloc failed - T1 will fall back to T2 "
                             "(check CUDA version vs cc %d.%d)",
                             _gi, g_gpus[_gi].cc_major, g_gpus[_gi].cc_minor);
                }
            } else {
                netd_log("WARN GPU %d: cudaSetDevice failed - T1 unavailable", _gi);
            }
        }
    } else {
        netd_log("WARN: libcudart not loaded or missing symbols - T1 VRAM unavailable");
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/*  Network helpers                                                    */
/* ------------------------------------------------------------------ */

static int set_nonblock(int fd)
{
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static ssize_t send_all(int fd, const void *buf, size_t len)
{
    const uint8_t *p = (const uint8_t *)buf;
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, p + sent, len - sent, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                /* Audit F-L3-08: wait for writability via poll() instead of
                 * spinning with usleep().  Short-poll timeout keeps shutdown
                 * latency reasonable while letting the kernel wake us when
                 * the socket buffer drains. */
                struct pollfd pf = { .fd = fd, .events = POLLOUT };
                int pr = poll(&pf, 1, 200);
                if (pr < 0 && errno != EINTR) return -1;
                continue;
            }
            return -1;
        }
        sent += (size_t)n;
    }
    return (ssize_t)sent;
}

/* PR-FF: compute 8-byte truncated HMAC-SHA256 over (hdr || payload) using
 * the connection's session_key.  Truncated to 8 bytes per RFC 2104 - still
 * 2^64 brute-force resistance; enough for replay protection in our model.
 * Returns the 8 bytes packed into a uint64_t (LE on wire). */
static void gb_msg_mac(const uint8_t key[32],
                        const struct gb_net_header *hdr,
                        const void *payload, uint32_t payload_len,
                        uint8_t out_mac8[8])
{
    /* HMAC(key, hdr || payload).  We HMAC the on-wire bytes (16-byte
     * header + payload bytes) so verification on the receiver matches
     * exactly what's transmitted regardless of any padding/alignment. */
    uint8_t k_ipad[GB_SHA256_BLOCK], k_opad[GB_SHA256_BLOCK];
    for (int i = 0; i < GB_SHA256_BLOCK; i++) {
        uint8_t kb = (i < 32) ? key[i] : 0;
        k_ipad[i] = kb ^ 0x36;
        k_opad[i] = kb ^ 0x5c;
    }
    uint8_t inner[32];
    gb_sha256_ctx c;
    gb_sha256_init(&c);
    gb_sha256_update(&c, k_ipad, GB_SHA256_BLOCK);
    gb_sha256_update(&c, (const uint8_t *)hdr, GB_NET_HDR_SIZE);
    if (payload && payload_len)
        gb_sha256_update(&c, (const uint8_t *)payload, payload_len);
    gb_sha256_final(&c, inner);
    gb_sha256_init(&c);
    gb_sha256_update(&c, k_opad, GB_SHA256_BLOCK);
    gb_sha256_update(&c, inner, 32);
    uint8_t full[32];
    gb_sha256_final(&c, full);
    memcpy(out_mac8, full, 8);
    explicit_bzero(k_ipad, sizeof(k_ipad));
    explicit_bzero(k_opad, sizeof(k_opad));
    explicit_bzero(inner, sizeof(inner));
    explicit_bzero(full, sizeof(full));
}

static int send_msg(struct client *cli, uint16_t msg_type, uint16_t flags,
                    const void *payload, uint32_t payload_len)
{
    /* PR-QQ: serialise the multi-write framing per client.  Without
     * this, a future async worker pool replying to the same client
     * could interleave bytes mid-frame. */
    pthread_mutex_lock(&cli->send_lock);
    struct gb_net_header hdr = {
        .magic       = GB_NET_MAGIC,
        .msg_type    = msg_type,
        .flags       = flags,
        .payload_len = payload_len,
        .seq_num     = cli->send_seq++,
    };
    int rc = 0;
    if (send_all(cli->fd, &hdr, GB_NET_HDR_SIZE) < 0) rc = -1;
    else if (payload_len > 0 && payload &&
             send_all(cli->fd, payload, payload_len) < 0) rc = -1;
    /* PR-FF: in v4 mode, append 8-byte MAC after the payload. */
    else if (cli->mac_enabled) {
        uint8_t mac[8];
        gb_msg_mac(cli->mac_session_key, &hdr, payload, payload_len, mac);
        if (send_all(cli->fd, mac, sizeof(mac)) < 0) rc = -1;
        explicit_bzero(mac, sizeof(mac));
    }
    pthread_mutex_unlock(&cli->send_lock);
    return rc;
}

/* GT/s -> PCIe generation.  Mirrors gb_telemetry.py's _gts_to_pcie_gen() and
 * greenboost_setup.sh's _gts_to_gen() bash helper (search "PCIe link (GPU
 * slot)" in that file) - ONE shared mapping across C/Python/bash, kept in
 * sync deliberately.  Parses a sysfs "<N>[.0] GT/s PCIe" string. */
static gb_u32 gts_to_gen(const char *speed_str)
{
    float gts = 0;
    sscanf(speed_str, "%f", &gts);
    if      (gts >= 64.0f) return 6;
    else if (gts >= 32.0f) return 5;
    else if (gts >= 16.0f) return 4;
    else if (gts >= 8.0f)  return 3;
    else if (gts >= 5.0f)  return 2;
    else if (gts >= 1.0f)  return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Message handlers                                                   */
/* ------------------------------------------------------------------ */

static int handle_handshake(struct client *cli, const void *payload, uint32_t len)
{
    /* feature_flags is a trailing field: accept a req that stops just before
     * it (base size) so a host built without the feature still handshakes. */
    size_t hs_base = offsetof(struct gb_net_handshake_req, feature_flags);
    if (len < hs_base) {
        netd_log("WARN: handshake too short from %s (%u bytes)", cli->remote_addr, len);
        return -1;
    }

    const struct gb_net_handshake_req *req =
        (const struct gb_net_handshake_req *)payload;
    /* PR-UU: BE-safe field reads. */
    gb_u32 req_proto_version = GB_LE_U32(req->proto_version);
    gb_u32 req_gpu_count     = GB_LE_U32(req->gpu_count);
    gb_u32 req_feature_flags =
        (len >= sizeof(struct gb_net_handshake_req)) ? GB_LE_U32(req->feature_flags) : 0u;

    netd_log("Handshake from %s (host: %.*s, proto v%u, %u GPUs)",
             cli->remote_addr,
             GB_NET_MAX_HOSTNAME, req->hostname,
             req_proto_version, req_gpu_count);

    if (req_proto_version != GB_NET_PROTO_VER) {
        /* Audit F-L3-09: hard-fail on version mismatch.  Silent accept caused
         * v1 fields to be misread when v2 wire format changes (cumulative size
         * checks, future seq nums).  Send a typed rejection then drop. */
        netd_log("ERROR: protocol version mismatch (got %u, want %u) - closing %s",
                 req_proto_version, GB_NET_PROTO_VER, cli->remote_addr);
        struct gb_net_handshake_resp rej;
        memset(&rej, 0, sizeof(rej));
        rej.status        = GB_STATUS_ERR_PROTO;
        rej.proto_version = GB_NET_PROTO_VER;
        (void)send_msg(cli, GB_MSG_HANDSHAKE_RESP, GB_NET_FLAG_RESPONSE,
                       &rej, sizeof(rej));
        return -1;
    }

    cli->feeder_id = g_next_feeder_id++;

    struct gb_net_handshake_resp resp;
    memset(&resp, 0, sizeof(resp));
    resp.status        = GB_STATUS_OK;
    resp.feeder_id     = cli->feeder_id;
    resp.proto_version = GB_NET_PROTO_VER;
    resp.gpu_count     = (gb_u32)g_gpu_count;
    strncpy(resp.hostname, g_hostname, GB_NET_MAX_HOSTNAME - 1);
    memcpy(resp.gpus, g_gpus, sizeof(struct gb_net_gpu_info) * g_gpu_count);

    /* Fabric zstd negotiation: advertise only if the host asked for it and this
     * daemon was built with libzstd and not disabled. cli->feat_zstd gates
     * whether responses to this client may be compressed (H2D decompress is
     * driven per-message by GB_NET_FLAG_COMP_ZSTD regardless). */
    cli->feat_zstd = 0;
#ifdef GB_HAVE_ZSTD
    if ((req_feature_flags & GB_NET_FEAT_ZSTD) && gb_netd_zstd_advertise()) {
        resp.feature_flags |= GB_NET_FEAT_ZSTD;
        cli->feat_zstd = 1;
        netd_log("fabric zstd compression negotiated with %s", cli->remote_addr);
    }
#else
    (void)req_feature_flags;
#endif

    /* dma-buf compressed-content descriptor: advertised unconditionally
     * (see the flag's comment in features/net_fabric.h for why this isn't
     * gated by a build macro like GB_HAVE_ZSTD above) , cli->feat_dmabuf_
     * compression simply records whether the PEER also understands it, so
     * a future relay path can decide whether it's safe to forward a
     * buffer's compression descriptor as-is or must decompress first for
     * an older peer. */
    cli->feat_dmabuf_compression = 0;
    if (req_feature_flags & GB_NET_FEAT_DMABUF_COMPRESSION) {
        resp.feature_flags |= GB_NET_FEAT_DMABUF_COMPRESSION;
        cli->feat_dmabuf_compression = 1;
        netd_log("dma-buf compression descriptor support negotiated with %s", cli->remote_addr);
    }

    /* T1: advertise topology-report support when this node's hardware profile
     * exists on disk (Full Install always writes it). The host then fetches it
     * over GB_MSG_TOPOLOGY instead of falling back to SSH. */
    if (access("/etc/greenboost/profiles/default.md", R_OK) == 0)
        resp.feature_flags |= GB_NET_FEAT_TOPOLOGY;

    /* D1: PCIe link info - read from sysfs for GPU 0
     *
     * gts_to_gen() thresholds mirror gb_telemetry.py's _gts_to_pcie_gen() /
     * greenboost_setup.sh's _gts_to_gen() - ONE shared mapping across
     * Python/bash/C, kept in sync deliberately (the old inline thresholds
     * here were off by one full generation: gts>=16 was mapped to gen5
     * instead of gen4, systematically overstating every feeder's reported
     * link by one generation once at Gen3+ speeds - fixed 2026-07-14). */
    {
        const char *pcie_dirs[] = {
            "/sys/class/drm/card0/device",
            "/sys/class/drm/card1/device",
            NULL
        };
        for (int _pi = 0; pcie_dirs[_pi]; _pi++) {
            char buf[160];
            FILE *fp;
            snprintf(buf, sizeof(buf), "%s/current_link_speed", pcie_dirs[_pi]);
            fp = fopen(buf, "r");
            if (!fp) continue;
            char speed_str[32] = {0};
            if (fgets(speed_str, sizeof(speed_str), fp))
                resp.pcie_link_gen = gts_to_gen(speed_str);
            fclose(fp);
            snprintf(buf, sizeof(buf), "%s/current_link_width", pcie_dirs[_pi]);
            fp = fopen(buf, "r");
            if (fp) {
                if (fscanf(fp, "%u", &resp.pcie_link_width) != 1)
                    resp.pcie_link_width = 0;
                fclose(fp);
            }

            /* D-PCIE-CEIL: real slot ceiling from the PARENT PCI bridge, not
             * the GPU's own device-level max (which, like NVML's
             * MaxPcieLinkWidth, reports what the SILICON can do regardless of
             * how many lanes the slot actually wires - a x16-capable chip in
             * a x8 laptop slot still reports a x16 device max). "current"
             * also idles at Gen1/Gen2 under ASPM power saving, so neither
             * current_link_* nor the GPU's own max_link_* is trustworthy as
             * the achievable ceiling; the parent bridge is. */
            snprintf(buf, sizeof(buf), "%s/../max_link_speed", pcie_dirs[_pi]);
            fp = fopen(buf, "r");
            if (fp) {
                char slot_speed_str[32] = {0};
                if (fgets(slot_speed_str, sizeof(slot_speed_str), fp))
                    resp.pcie_link_gen_max = gts_to_gen(slot_speed_str);
                fclose(fp);
            }
            snprintf(buf, sizeof(buf), "%s/../max_link_width", pcie_dirs[_pi]);
            fp = fopen(buf, "r");
            if (fp) {
                if (fscanf(fp, "%u", &resp.pcie_link_width_max) != 1)
                    resp.pcie_link_width_max = 0;
                fclose(fp);
            }
            if (!resp.pcie_link_gen_max)   resp.pcie_link_gen_max   = resp.pcie_link_gen;
            if (!resp.pcie_link_width_max) resp.pcie_link_width_max = resp.pcie_link_width;

            /* Effective BW computed against the CEILING, not the (possibly
             * idle-downtrained) current link, so feeder scoring/placement
             * reflects real achievable bandwidth: gen5_x16=64 gen4_x16=32
             * gen3_x16=16 gen3_x8=8 gen4_x4=8 etc. Approximation:
             * (gen_bw_gbs_per_lane × width × 1000) / 8 bytes = MB/s */
            static const float lane_gbs[] = {0, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f};
            gb_u32 gen = (resp.pcie_link_gen_max > 0 && resp.pcie_link_gen_max < 6)
                         ? resp.pcie_link_gen_max : 5;
            resp.pcie_effective_bw_mbs =
                (gb_u32)(lane_gbs[gen] * resp.pcie_link_width_max * 1000.0f);
            break;
        }
    }

    int rc = send_msg(cli, GB_MSG_HANDSHAKE_RESP, GB_NET_FLAG_RESPONSE,
                      &resp, sizeof(resp));
    /* Phase 2b: create persistent async CUDA stream for this client.
     * Created after the handshake response so the client is fully initialised. */
    if (rc == 0 && f_cudaStreamCreate && !cli->async_stream)
        f_cudaStreamCreate(&cli->async_stream);
    return rc;
}

static int handle_heartbeat(struct client *cli, const void *payload, uint32_t len)
{
    (void)len;
    cli->last_heartbeat_ms = mono_ms();

    /* Build heartbeat response with live GPU metrics */
    struct gb_net_heartbeat hb;
    memset(&hb, 0, sizeof(hb));
    hb.timestamp_ms = mono_ms();
    hb.gpu_count    = (gb_u32)g_gpu_count;

    for (int i = 0; i < g_gpu_count; i++) {
        if (!g_nvml_devices[i]) continue;
        if (f_nvmlMemInfo) {
            nvmlMemory_t mem = {0};
            if (f_nvmlMemInfo(g_nvml_devices[i], &mem) == 0) {
                hb.gpu_load[i].vram_free_bytes = mem.free;
                hb.gpu_load[i].vram_used_bytes = mem.used;
            }
        }
        if (f_nvmlUtilRates) {
            nvmlUtilization_t util = {0};
            if (f_nvmlUtilRates(g_nvml_devices[i], &util) == 0) {
                hb.gpu_load[i].gpu_util_pct = util.gpu;
                hb.gpu_load[i].mem_util_pct = util.memory;
            }
        }
        /* D2: throttle, temperature, power */
        if (f_nvmlGetThrottle) {
            unsigned long long reasons = 0;
            if (f_nvmlGetThrottle(g_nvml_devices[i], &reasons) == 0)
                hb.gpu_load[i].throttle_reasons = (gb_u32)(reasons & 0xFFFFFFFFULL);
        }
        if (f_nvmlGetTemp) {
            unsigned int temp = 0;
            if (f_nvmlGetTemp(g_nvml_devices[i], NVML_TEMPERATURE_GPU, &temp) == 0)
                hb.gpu_load[i].gpu_temp_c = (gb_u16)(temp & 0xFFFF);
        }
        if (f_nvmlGetPower) {
            unsigned int mw = 0;
            if (f_nvmlGetPower(g_nvml_devices[i], &mw) == 0)
                hb.gpu_load[i].power_draw_w = (gb_u16)(mw / 1000);
        }
        if (f_nvmlGetPowerLimit) {
            unsigned int mw = 0;
            if (f_nvmlGetPowerLimit(g_nvml_devices[i], &mw) == 0)
                hb.gpu_load[i].power_limit_w = (gb_u16)(mw / 1000);
        }
        /* D3: ECC errors */
        if (f_nvmlGetEcc) {
            unsigned long long sbe_now = 0;
            f_nvmlGetEcc(g_nvml_devices[i], NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                         NVML_VOLATILE_ECC, &sbe_now);
            unsigned long long delta = (sbe_now >= g_ecc_sbe_baseline[i])
                                       ? sbe_now - g_ecc_sbe_baseline[i] : 0;
            g_ecc_sbe_baseline[i] = sbe_now;
            hb.gpu_load[i].ecc_sbe_delta = (gb_u16)(delta > 0xFFFF ? 0xFFFF : delta);

            unsigned long long dbe = 0;
            f_nvmlGetEcc(g_nvml_devices[i], NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                         NVML_AGGREGATE_ECC, &dbe);
            hb.gpu_load[i].ecc_dbe_count = (gb_u16)(dbe > 0xFFFF ? 0xFFFF : dbe);
        }
    }

    return send_msg(cli, GB_MSG_HEARTBEAT, GB_NET_FLAG_RESPONSE,
                    &hb, sizeof(hb));
}

static int handle_gpu_query(struct client *cli, const void *payload, uint32_t len)
{
    if (len < sizeof(struct gb_net_gpu_query))
        return -1;

    const struct gb_net_gpu_query *q = (const struct gb_net_gpu_query *)payload;
    struct gb_net_gpu_query_resp resp = { .status = GB_STATUS_OK, .value = 0 };

    if (q->device_id >= (gb_u32)g_gpu_count) {
        resp.status = GB_STATUS_ERR_INVALID;
    } else {
        /* For Phase 1, return cached static attributes */
        switch (q->attribute_id) {
        case 75: /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR */
            resp.value = (gb_s32)g_gpus[q->device_id].cc_major;
            break;
        case 76: /* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR */
            resp.value = (gb_s32)g_gpus[q->device_id].cc_minor;
            break;
        default:
            resp.status = GB_STATUS_ERR_INVALID;
            break;
        }
    }

    return send_msg(cli, GB_MSG_GPU_QUERY, GB_NET_FLAG_RESPONSE,
                    &resp, sizeof(resp));
}

static gb_u32 get_ddr_speed(void) {
    static gb_u32 cached_speed = 0;
    if (cached_speed > 0) return cached_speed;
    /* PR-X: parse dmidecode output entirely in C.  The previous pipeline
     *   /usr/sbin/dmidecode | grep | grep | head | grep -oE
     * inherited $PATH for grep/head/grep, giving an attacker with write
     * access to any directory earlier in PATH a code-execution vector
     * in the root-running daemon.  The 4-stage shell pipe also forks
     * five processes per call.  Replace with a single execv'd dmidecode
     * and an in-process line scan. */
    int pipefd[2];
    if (pipe(pipefd) < 0) goto fallback;
    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); goto fallback; }
    if (pid == 0) {
        /* child: stdout → pipe write end */
        dup2(pipefd[1], STDOUT_FILENO);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
        close(pipefd[0]); close(pipefd[1]);
        /* env -i style: don't inherit attacker-controlled environment */
        char *const argv[] = { "/usr/sbin/dmidecode", "-t", "memory", NULL };
        char *const envp[] = { "PATH=/usr/sbin:/usr/bin:/sbin:/bin", NULL };
        execve("/usr/sbin/dmidecode", argv, envp);
        _exit(127);
    }
    close(pipefd[1]);
    FILE *fp = fdopen(pipefd[0], "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            /* Match `Configured Memory Speed:` or `Speed:` followed by digits + ` MT/s` */
            const char *p = strstr(line, "Speed:");
            if (!p) continue;
            p += 6;
            while (*p == ' ' || *p == '\t') p++;
            if (*p < '0' || *p > '9') continue;
            gb_u32 v = 0;
            while (*p >= '0' && *p <= '9' && v < 100000) {
                v = v * 10 + (gb_u32)(*p - '0');
                p++;
            }
            /* Require MT/s suffix so we don't match other Speed fields */
            while (*p == ' ' || *p == '\t') p++;
            if (strncmp(p, "MT/s", 4) == 0 && v > 0) {
                cached_speed = v;
                break;
            }
        }
        fclose(fp);
    } else {
        close(pipefd[0]);
    }
    int status = 0;
    waitpid(pid, &status, 0);

fallback:
    if (cached_speed == 0) cached_speed = 2400; /* Fallback DDR4-2400 */
    return cached_speed;
}
static uint32_t get_nvme_speed_mbs(void) {
    static uint32_t cached = 0;
    if (cached) return cached;
    /* Detect NVMe device and return typical sequential read speed */
    FILE *fp = fopen("/sys/block/nvme0n1/queue/max_hw_sectors_kb", "r");
    if (fp) { fclose(fp); cached = 3500; return cached; }
    fp = fopen("/sys/block/nvme0/queue/max_hw_sectors_kb", "r");
    if (fp) { fclose(fp); cached = 3500; return cached; }
    /* SATA SSD fallback */
    fp = fopen("/sys/block/sda/queue/rotational", "r");
    if (fp) {
        int rot = 1;
        if (fscanf(fp, "%d", &rot) == 1 && rot == 0)
            cached = 550;
        fclose(fp);
    }
    if (cached == 0) cached = 500;
    return cached;
}

static int handle_mem_info(struct client *cli, const void *payload, uint32_t len)
{
    if (len < sizeof(struct gb_net_mem_info))
        return -1;

    const struct gb_net_mem_info *q = (const struct gb_net_mem_info *)payload;
    struct gb_net_mem_info_resp resp;
    memset(&resp, 0, sizeof(resp));

    /* Include T2 (system RAM) so the host shim sees full T1+T2 capacity.
     * Use MemAvailable (not freeram) so page cache that Linux will reclaim on
     * demand is counted as available - freeram gives a falsely low number. */
    gb_u64 ram_total = 0, ram_free = 0;
    {
        FILE *_mf = fopen("/proc/meminfo", "r");
        if (_mf) {
            char _ml[128];
            while (fgets(_ml, sizeof(_ml), _mf)) {
                unsigned long long _kb = 0;
                if      (sscanf(_ml, "MemTotal: %llu kB",     &_kb) == 1) ram_total = _kb * 1024ULL;
                else if (sscanf(_ml, "MemAvailable: %llu kB", &_kb) == 1) ram_free  = _kb * 1024ULL;
            }
            fclose(_mf);
        }
        if (!ram_total) {
            struct sysinfo _si;
            if (sysinfo(&_si) == 0) {
                ram_total = (gb_u64)_si.totalram * (gb_u64)_si.mem_unit;
                ram_free  = (gb_u64)_si.freeram  * (gb_u64)_si.mem_unit;
            }
        }
    }

    /* T3: prefer GreenBoost kernel module pool size; fall back to /proc/swaps */
    gb_u64 swap_total = 0, swap_used = 0;
    {
        FILE *sf = fopen("/sys/module/greenboost/parameters/nvme_pool_gb", "r");
        if (sf) {
            int nvme_gb = 0;
            if (fscanf(sf, "%d", &nvme_gb) == 1 && nvme_gb > 0)
                swap_total = (gb_u64)nvme_gb * 1024ULL * 1024ULL * 1024ULL;
            fclose(sf);
        }
        if (swap_total == 0) {
            sf = fopen("/proc/swaps", "r");
            if (sf) {
                char sline[256];
                if (fgets(sline, sizeof(sline), sf)) { /* skip header line */ }
                while (fgets(sline, sizeof(sline), sf)) {
                    char sname[200]; char stype[32];
                    unsigned long long ssz = 0, sus = 0;
                    if (sscanf(sline, "%199s %31s %llu %llu", sname, stype, &ssz, &sus) == 4) {
                        swap_total += ssz * 1024ULL;
                        swap_used  += sus * 1024ULL;
                    }
                }
                fclose(sf);
            }
        }
    }
    gb_u64 swap_free = (swap_total > swap_used) ? (swap_total - swap_used) : 0;

    if (q->device_id >= (gb_u32)g_gpu_count) {
        resp.status = GB_STATUS_ERR_INVALID;
    } else if (f_nvmlMemInfo && g_nvml_devices[q->device_id]) {
        nvmlMemory_t mem = {0};
        if (f_nvmlMemInfo(g_nvml_devices[q->device_id], &mem) == 0) {
            resp.status       = GB_STATUS_OK;
            resp.t2_speed_mts = get_ddr_speed();
            resp.t3_speed_mbs = get_nvme_speed_mbs();
            resp.t1_free      = (gb_u64)mem.free;
            resp.t1_total     = (gb_u64)mem.total;
            resp.t2_free      = ram_free;
            resp.t2_total     = ram_total;
            resp.t3_free      = swap_free;
            resp.t3_total     = swap_total;
            resp.free_bytes   = resp.t1_free + resp.t2_free;
            resp.total_bytes  = resp.t1_total + resp.t2_total + resp.t3_total;
        } else {
            resp.status       = GB_STATUS_OK;
            resp.t2_speed_mts = get_ddr_speed();
            resp.t3_speed_mbs = get_nvme_speed_mbs();
            resp.t1_free      = g_gpus[q->device_id].vram_bytes;
            resp.t1_total     = g_gpus[q->device_id].vram_bytes;
            resp.t2_free      = ram_free;
            resp.t2_total     = ram_total;
            resp.t3_free      = swap_free;
            resp.t3_total     = swap_total;
            resp.free_bytes   = resp.t1_free + resp.t2_free;
            resp.total_bytes  = resp.t1_total + resp.t2_total + resp.t3_total;
        }
    } else {
        resp.status       = GB_STATUS_OK;
        resp.t2_speed_mts = get_ddr_speed();
        resp.t3_speed_mbs = get_nvme_speed_mbs();
        resp.t1_free      = g_gpus[q->device_id].vram_bytes;
        resp.t1_total     = g_gpus[q->device_id].vram_bytes;
        resp.t2_free      = ram_free;
        resp.t2_total     = ram_total;
        resp.t3_free      = swap_free;
        resp.t3_total     = swap_total;
        resp.free_bytes   = resp.t1_free + resp.t2_free;
        resp.total_bytes  = resp.t1_total + resp.t2_total + resp.t3_total;
    }

    return send_msg(cli, GB_MSG_MEM_INFO, GB_NET_FLAG_RESPONSE,
                    &resp, sizeof(resp));
}

static int handle_disconnect(struct client *cli)
{
    netd_log("Client %s disconnected (feeder_id=%u)", cli->remote_addr, cli->feeder_id);
    return -1; /* caller closes */
}

/* ------------------------------------------------------------------ */
/*  Remote allocation tracking                                         */
/* ------------------------------------------------------------------ */

#define MAX_REMOTE_ALLOCS 65536

struct remote_alloc {
    uint64_t handle;    /* the remote_handle sent to host                       */
    void    *dev_ptr;   /* GPU ptr (tier 0) or NULL                             */
    void    *host_ptr;  /* pinned host ptr (tier 1) or pageable ptr (tier 2)    */
    uint64_t size;
    int      device_id;
    uint8_t  tier;      /* 0=T1_GPU, 1=T2_pinned_DDR, 2=T3_pageable            */
    int      in_use;
    uint32_t client_id; /* feeder_id of owning client - freed on disconnect     */
};

static struct remote_alloc g_remote_allocs[MAX_REMOTE_ALLOCS];
static pthread_mutex_t     g_alloc_lock = PTHREAD_MUTEX_INITIALIZER;

/* U7: per-client allocation accounting (vLLM SingleTypeKVCacheManager pattern).
 * Tracks total VRAM committed per connected host so client_cleanup() can log
 * reclaim accurately and future sliding-window trimming can target live groups. */
#define GB_MAX_ALLOC_GROUPS MAX_CLIENTS
typedef struct {
    uint32_t client_id;    /* feeder_id of the host that owns this group */
    size_t   total_bytes;  /* cumulative bytes currently allocated */
    uint32_t alloc_count;
    int      active;
} gb_alloc_group_t;
static gb_alloc_group_t g_alloc_groups[GB_MAX_ALLOC_GROUPS];

static void ag_record_alloc(uint32_t client_id, size_t sz)
{
    for (int i = 0; i < GB_MAX_ALLOC_GROUPS; i++) {
        if (g_alloc_groups[i].active && g_alloc_groups[i].client_id == client_id) {
            g_alloc_groups[i].total_bytes += sz;
            g_alloc_groups[i].alloc_count++;
            return;
        }
    }
    /* First alloc from this client - open a new group */
    for (int i = 0; i < GB_MAX_ALLOC_GROUPS; i++) {
        if (!g_alloc_groups[i].active) {
            g_alloc_groups[i].client_id  = client_id;
            g_alloc_groups[i].total_bytes = sz;
            g_alloc_groups[i].alloc_count = 1;
            g_alloc_groups[i].active      = 1;
            return;
        }
    }
}

static void ag_record_free(uint32_t client_id, size_t sz)
{
    for (int i = 0; i < GB_MAX_ALLOC_GROUPS; i++) {
        if (g_alloc_groups[i].active && g_alloc_groups[i].client_id == client_id) {
            g_alloc_groups[i].total_bytes =
                (g_alloc_groups[i].total_bytes >= sz)
                ? g_alloc_groups[i].total_bytes - sz : 0;
            return;
        }
    }
}

static void ag_close(uint32_t client_id)
{
    for (int i = 0; i < GB_MAX_ALLOC_GROUPS; i++) {
        if (g_alloc_groups[i].active && g_alloc_groups[i].client_id == client_id) {
            g_alloc_groups[i].active = 0;
            return;
        }
    }
}

/* PR-T: open-addressed hashmap for handle → g_remote_allocs index.
 *
 * Before this, every memcpy/exec/free did an O(MAX_REMOTE_ALLOCS) linear
 * scan through a 65536-entry array.  On a busy LLM inference workload
 * with thousands of operations per second, this is the netd hot path.
 * After this, ra_find is O(1) average + worst-case O(probe_chain) which
 * is bounded by the 50% load factor.
 *
 * Slot encoding:
 *   0           = empty (probe ends here)
 *   1..65536    = (index+1) into g_remote_allocs[]
 *   0xFFFFFFFF  = tombstone (was occupied, keep probing past)
 *
 * Tombstones prevent broken probe chains after deletion.  In a long-
 * running daemon they accumulate; the hashmap is rebuilt only on
 * daemon restart (acceptable: 64-thread scan once per process lifetime).
 *
 * The hash itself uses Knuth's multiplicative constant - well-mixed and
 * matches the pattern the host-side bvmm_ht and ht tables use. */
#define RA_HASH_SIZE      (MAX_REMOTE_ALLOCS * 2u)
#define RA_HASH_MASK      (RA_HASH_SIZE - 1u)
#define RA_HASH_TOMBSTONE 0xFFFFFFFFu
static uint32_t g_ra_hash[RA_HASH_SIZE];  /* protected by g_alloc_lock */

static inline uint32_t ra_hash(uint64_t handle)
{
    return (uint32_t)((handle * 0x9E3779B97F4A7C15ULL) >> 32);
}

/* Insert (handle → idx) into the hash.  Caller holds g_alloc_lock. */
static void ra_hash_insert(uint64_t handle, uint32_t idx)
{
    uint32_t slot = ra_hash(handle) & RA_HASH_MASK;
    for (uint32_t i = 0; i < RA_HASH_SIZE; i++) {
        uint32_t s = (slot + i) & RA_HASH_MASK;
        uint32_t v = g_ra_hash[s];
        if (v == 0 || v == RA_HASH_TOMBSTONE) {
            g_ra_hash[s] = idx + 1;  /* +1 so 0 stays "empty" sentinel */
            return;
        }
    }
    /* Hash table full - should never happen at 50% load factor with cap of
     * MAX_REMOTE_ALLOCS in g_remote_allocs.  If it does, ra_find still
     * works (slow path falls through to NULL on full chain). */
}

/* Mark (handle, idx) as gone.  Caller holds g_alloc_lock. */
static void ra_hash_remove(uint64_t handle, uint32_t idx)
{
    uint32_t slot = ra_hash(handle) & RA_HASH_MASK;
    for (uint32_t i = 0; i < RA_HASH_SIZE; i++) {
        uint32_t s = (slot + i) & RA_HASH_MASK;
        uint32_t v = g_ra_hash[s];
        if (v == 0) return;
        if (v != RA_HASH_TOMBSTONE && v == idx + 1) {
            g_ra_hash[s] = RA_HASH_TOMBSTONE;
            return;
        }
    }
}

/* PR-HH single-threaded invariant (audit follow-up):
 *
 * Several callers (handle_cuda_memcpy_h2d/d2h/d2d) take g_alloc_lock,
 * call ra_find, drop the lock, then dereference `ra` while doing the
 * actual CUDA memcpy.  This is safe ONLY because netd processes all
 * client I/O on a single thread (the epoll loop).  No second thread
 * can call ra_free / ra_new / client_cleanup concurrently with these
 * deref-after-unlock paths.  handle_cuda_exec correctly keeps the lock
 * held across the pointer-relocation block for a different reason
 * (multiple ra_find calls in one scope; consistency required).
 *
 * If a future change introduces a second worker thread that touches
 * the client/alloc tables (e.g. async H2D pipelining), every
 * deref-after-unlock path must be converted to "hold lock across the
 * full operation" - see handle_cuda_exec for the template.  CI should
 * also gate this with a static analysis pass once thread expansion is
 * planned. */
static struct remote_alloc *ra_find(uint64_t handle)
{
    if (handle == 0) return NULL;
    uint32_t slot = ra_hash(handle) & RA_HASH_MASK;
    for (uint32_t i = 0; i < RA_HASH_SIZE; i++) {
        uint32_t s = (slot + i) & RA_HASH_MASK;
        uint32_t v = g_ra_hash[s];
        if (v == 0) return NULL;  /* clean empty: handle not in table */
        if (v == RA_HASH_TOMBSTONE) continue;
        struct remote_alloc *ra = &g_remote_allocs[v - 1];
        if (ra->in_use && ra->handle == handle) return ra;
    }
    return NULL;
}

static struct remote_alloc *ra_new(void)
{
    /* Free-slot scan in the main array stays linear - this path runs once
     * per alloc, not per memcpy/exec, so the O(N) cost amortises.  A
     * future enhancement could thread a "next free hint" through the
     * slots, but the current bottleneck (PR-T target) was ra_find. */
    for (int i = 0; i < MAX_REMOTE_ALLOCS; i++)
        if (!g_remote_allocs[i].in_use)
            return &g_remote_allocs[i];
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  Kernel name → host function pointer map                            */
/* ------------------------------------------------------------------ */

#define MAX_KERNEL_NAMES 4096

struct kernel_entry {
    char     name[256];
    void    *host_func;    /* resolved via dlsym or __cudaRegisterFunction tracking */
    int      in_use;
    /* PR-L: client_id of the host that registered this kernel.  Used both
     * to scope lookups (a kernel registered by client A is NOT callable by
     * client B) and to invalidate entries on client disconnect - otherwise
     * stale host_func pointers from departed clients accumulate, and a
     * subsequent client can launch them via name collision.  Cross-tenant
     * compute leak risk in the prior code (g_kernel_map was global). */
    uint32_t client_id;
};

static struct kernel_entry g_kernel_map[MAX_KERNEL_NAMES];
static pthread_mutex_t     g_kernel_map_lock = PTHREAD_MUTEX_INITIALIZER;

/* PR-V: open-addressed hashmap for (client_id, kernel_name) → index.
 *
 * Symmetric to PR-T's ra_hash but keyed on a (client_id, name) tuple.
 * handle_cuda_launch and handle_cuda_exec previously scanned 4096
 * entries with strcmp per launch - at hundreds of launches per second
 * across a busy inference workload, that was a measurable hot path.
 *
 * Slot encoding (same as PR-T): 0 = empty, 1..N = idx+1, 0xFFFFFFFF = tombstone.
 * Capacity 2× MAX_KERNEL_NAMES → ≤50% load factor. */
#define KM_HASH_SIZE      (MAX_KERNEL_NAMES * 2u)
#define KM_HASH_MASK      (KM_HASH_SIZE - 1u)
#define KM_HASH_TOMBSTONE 0xFFFFFFFFu
static uint32_t g_km_hash[KM_HASH_SIZE];  /* protected by g_kernel_map_lock */

static inline uint32_t km_hash(uint32_t client_id, const char *name)
{
    /* Mix client_id into the seed (Knuth multiplicative) then FNV-1a over
     * the name bytes.  Two clients registering the same kernel name land
     * on different hash buckets - no false-positive collisions across
     * tenants. */
    uint64_t h = (uint64_t)client_id * 0x9E3779B97F4A7C15ULL;
    for (const unsigned char *p = (const unsigned char *)name; *p; p++)
        h = (h ^ *p) * 0x100000001B3ULL;
    return (uint32_t)(h >> 32);
}

/* Caller holds g_kernel_map_lock. */
static void km_hash_insert(uint32_t client_id, const char *name, uint32_t idx)
{
    uint32_t slot = km_hash(client_id, name) & KM_HASH_MASK;
    for (uint32_t i = 0; i < KM_HASH_SIZE; i++) {
        uint32_t s = (slot + i) & KM_HASH_MASK;
        uint32_t v = g_km_hash[s];
        if (v == 0 || v == KM_HASH_TOMBSTONE) {
            g_km_hash[s] = idx + 1;
            return;
        }
    }
}

static void km_hash_remove(uint32_t client_id, const char *name, uint32_t idx)
{
    uint32_t slot = km_hash(client_id, name) & KM_HASH_MASK;
    for (uint32_t i = 0; i < KM_HASH_SIZE; i++) {
        uint32_t s = (slot + i) & KM_HASH_MASK;
        uint32_t v = g_km_hash[s];
        if (v == 0) return;
        if (v != KM_HASH_TOMBSTONE && v == idx + 1) {
            g_km_hash[s] = KM_HASH_TOMBSTONE;
            return;
        }
    }
}

/* Returns the host_func for (client_id, name) or NULL.
 * Caller holds g_kernel_map_lock. */
static void *km_lookup(uint32_t client_id, const char *name)
{
    if (!name) return NULL;
    uint32_t slot = km_hash(client_id, name) & KM_HASH_MASK;
    for (uint32_t i = 0; i < KM_HASH_SIZE; i++) {
        uint32_t s = (slot + i) & KM_HASH_MASK;
        uint32_t v = g_km_hash[s];
        if (v == 0) return NULL;
        if (v == KM_HASH_TOMBSTONE) continue;
        struct kernel_entry *e = &g_kernel_map[v - 1];
        if (e->in_use &&
            e->client_id == client_id &&
            strcmp(e->name, name) == 0) {
            return e->host_func;
        }
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  Phase 2: CUDA Memory handlers                                      */
/* ------------------------------------------------------------------ */

static int handle_cuda_malloc(struct client *cli, const void *payload, uint32_t len)
{
    /* U8: reject new allocs during graceful drain */
    if (atomic_load_explicit(&g_draining, memory_order_relaxed)) {
        struct gb_net_cuda_malloc_resp resp = { .status = GB_STATUS_ERR_OOM };
        return send_msg(cli, GB_MSG_CUDA_MALLOC, GB_NET_FLAG_RESPONSE, &resp, sizeof(resp));
    }

    /* N9: token-bucket rate limiter - max 1000 allocs/s, burst 200 */
    {
        uint64_t now = mono_ms();
        uint64_t elapsed = now - cli->last_refill_ms;
        /* Refill: 1 token per ms (1000/s), bucket cap 200 */
        if (elapsed > 0) {
            cli->alloc_tokens += elapsed;
            if (cli->alloc_tokens > 200) cli->alloc_tokens = 200;
            cli->last_refill_ms = now;
        }
        if (cli->alloc_tokens == 0) {
            struct gb_net_cuda_malloc_resp resp = { .status = GB_STATUS_ERR_THROTTLE };
            NETD_EVT("THROTTLE_REJECT", "NET", 0, cli->feeder_id, "alloc_rate_exceeded");
            return send_msg(cli, GB_MSG_CUDA_MALLOC, GB_NET_FLAG_RESPONSE, &resp, sizeof(resp));
        }
        cli->alloc_tokens--;
    }

    atomic_fetch_add_explicit(&g_inflight_ops, 1, memory_order_acq_rel);

    if (len < sizeof(struct gb_net_cuda_malloc)) {
        atomic_fetch_sub_explicit(&g_inflight_ops, 1, memory_order_acq_rel);
        return -1;
    }

    const struct gb_net_cuda_malloc *req = (const struct gb_net_cuda_malloc *)payload;
    /* PR-MM template migration: cuda_malloc handler reads req->flags,
     * req->device_id, req->size through GB_LE_U32/U64 accessors.  On LE
     * hosts these compile to no-op casts; on BE they byte-swap.  This
     * site serves as the template - convert every other gb_u32/gb_u64
     * field access in netd/netc the same way to complete the BE port. */
    gb_u32 req_flags     = GB_LE_U32(req->flags);
    gb_u32 req_device_id = GB_LE_U32(req->device_id);
    gb_u64 req_size      = GB_LE_U64(req->size);
    uint8_t req_tier = (uint8_t)(req_flags & GB_ALLOC_TIER_MASK);

    if ((int)req_device_id >= g_gpu_count) {
        /* PR-RR audit fix: every early-return path from handle_cuda_malloc
         * must decrement g_inflight_ops to balance the increment at the top
         * of this function (line ~1593).  Without this, g_draining's wait
         * for inflight to hit zero hangs forever on phantom counts. */
        atomic_fetch_sub_explicit(&g_inflight_ops, 1, memory_order_acq_rel);
        struct gb_net_cuda_malloc_resp resp = { .status = GB_STATUS_ERR_INVALID };
        return send_msg(cli, GB_MSG_CUDA_MALLOC, GB_NET_FLAG_RESPONSE,
                        &resp, sizeof(resp));
    }

    if (f_cudaSetDevice)
        f_cudaSetDevice((int)req_device_id);

    struct gb_net_cuda_malloc_resp resp;
    memset(&resp, 0, sizeof(resp));

    /* ── Try T1: GPU VRAM ── */
    if (req_tier == GB_ALLOC_TIER_AUTO || req_tier == GB_ALLOC_TIER_T1) {
        void *dev_ptr = NULL;
        cudaError_t err = f_cudaMalloc ? f_cudaMalloc(&dev_ptr, (size_t)req_size)  /* PR-RR: was req->size (raw wire bytes; broken on BE) */ : 1;
        if (err == 0 && dev_ptr) {
            pthread_mutex_lock(&g_alloc_lock);
            struct remote_alloc *ra = ra_new();
            if (ra) {
                ra->handle    = atomic_fetch_add_explicit(&g_handle_counter, 1, memory_order_relaxed) ^ g_handle_salt;
                ra->dev_ptr   = dev_ptr;
                ra->host_ptr  = NULL;
                ra->size      = req_size;        /* PR-RR: BE-safe */
                ra->device_id = (int)req_device_id;
                ra->tier      = 0;
                ra->in_use    = 1;
                ra->client_id = cli->feeder_id;
                /* PR-T: index this handle for O(1) lookup. */
                ra_hash_insert(ra->handle, (uint32_t)(ra - g_remote_allocs));
                resp.status        = GB_STATUS_OK;
                resp.tier_used     = 0;
                resp.remote_handle = ra->handle;
            } else {
                if (f_cudaFree) f_cudaFree(dev_ptr);
                resp.status = GB_STATUS_ERR_OOM;
            }
            pthread_mutex_unlock(&g_alloc_lock);
            if (resp.status == GB_STATUS_OK) {
                ag_record_alloc(cli->feeder_id, (size_t)req_size);
                netd_log("T1 cudaMalloc(%llu MB) → handle=0x%llx GPU %d",
                         (unsigned long long)(req_size >> 20),
                         (unsigned long long)resp.remote_handle, (int)req_device_id);
                NETD_EVT("ALLOC_T1_GPU", "T1_GPU", req_size >> 20, resp.remote_handle, "cudaMalloc_ok");
                atomic_fetch_sub_explicit(&g_inflight_ops, 1, memory_order_acq_rel);
                return send_msg(cli, GB_MSG_CUDA_MALLOC, GB_NET_FLAG_RESPONSE,
                                &resp, sizeof(resp));
            }
        }
        if (req_tier == GB_ALLOC_TIER_T1) {
            netd_log("T1 cudaMalloc(%llu MB) OOM", (unsigned long long)(req_size >> 20));
            NETD_EVT("OOM_T1_GPU", "T1_GPU", req_size >> 20, 0, "cudaMalloc_failed");
            resp.status    = GB_STATUS_ERR_OOM;
            resp.tier_used = 0;
            /* PR-RR audit fix: balance the inflight++ at function entry. */
            atomic_fetch_sub_explicit(&g_inflight_ops, 1, memory_order_acq_rel);
            return send_msg(cli, GB_MSG_CUDA_MALLOC, GB_NET_FLAG_RESPONSE,
                            &resp, sizeof(resp));
        }
        netd_log("T1 OOM for %llu MB - cascading to T2", (unsigned long long)(req_size >> 20));
    }

    /* ── Try T2: pinned host DDR (cudaHostAlloc - GPU zero-copy accessible) ── */
    if (req_tier == GB_ALLOC_TIER_AUTO || req_tier == GB_ALLOC_TIER_T2) {
        void *host_ptr = NULL;
        cudaError_t err = f_cudaHostAlloc
            ? f_cudaHostAlloc(&host_ptr, (size_t)req_size, 0u /* cudaHostAllocDefault */)  /* PR-RR */
            : 1;
        if (err == 0 && host_ptr) {
            pthread_mutex_lock(&g_alloc_lock);
            struct remote_alloc *ra = ra_new();
            if (ra) {
                ra->handle    = atomic_fetch_add_explicit(&g_handle_counter, 1, memory_order_relaxed) ^ g_handle_salt;
                ra->dev_ptr   = NULL;
                ra->host_ptr  = host_ptr;
                ra->size      = req_size;        /* PR-RR: BE-safe */
                ra->device_id = (int)req_device_id;
                ra->tier      = 1;
                ra->in_use    = 1;
                ra->client_id = cli->feeder_id;
                /* PR-T: index this handle for O(1) lookup. */
                ra_hash_insert(ra->handle, (uint32_t)(ra - g_remote_allocs));
                resp.status        = GB_STATUS_OK;
                resp.tier_used     = 1;
                resp.remote_handle = ra->handle;
            } else {
                if (f_cudaFreeHost) f_cudaFreeHost(host_ptr);
                resp.status = GB_STATUS_ERR_OOM;
            }
            pthread_mutex_unlock(&g_alloc_lock);
            if (resp.status == GB_STATUS_OK) {
                ag_record_alloc(cli->feeder_id, (size_t)req_size);
                netd_log("T2 cudaHostAlloc(%llu MB) → handle=0x%llx",
                         (unsigned long long)(req_size >> 20),
                         (unsigned long long)resp.remote_handle);
                NETD_EVT("ALLOC_T2_DDR", "T2_DDR", req_size >> 20, resp.remote_handle, "cudaHostAlloc_ok");
                atomic_fetch_sub_explicit(&g_inflight_ops, 1, memory_order_acq_rel);
                return send_msg(cli, GB_MSG_CUDA_MALLOC, GB_NET_FLAG_RESPONSE,
                                &resp, sizeof(resp));
            }
        }
        if (req_tier == GB_ALLOC_TIER_T2) {
            netd_log("T2 cudaHostAlloc(%llu MB) OOM", (unsigned long long)(req_size >> 20));
            NETD_EVT("OOM_T2_DDR", "T2_DDR", req_size >> 20, 0, "cudaHostAlloc_failed");
            resp.status    = GB_STATUS_ERR_OOM;
            resp.tier_used = 1;
            /* PR-RR audit fix: balance the inflight++ at function entry. */
            atomic_fetch_sub_explicit(&g_inflight_ops, 1, memory_order_acq_rel);
            return send_msg(cli, GB_MSG_CUDA_MALLOC, GB_NET_FLAG_RESPONSE,
                            &resp, sizeof(resp));
        }
        netd_log("T2 OOM for %llu MB - cascading to T3", (unsigned long long)(req_size >> 20));
    }

    /* ── Try T3: pageable host RAM (mmap anonymous - swap-in required for GPU exec) ── */
    if (req_tier == GB_ALLOC_TIER_AUTO || req_tier == GB_ALLOC_TIER_T3) {
        void *mmap_ptr = mmap(NULL, (size_t)req_size,
                              PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mmap_ptr != MAP_FAILED) {
            mlock(mmap_ptr, (size_t)req_size);  /* best-effort; may fail for large allocs */
            pthread_mutex_lock(&g_alloc_lock);
            struct remote_alloc *ra = ra_new();
            if (ra) {
                ra->handle    = atomic_fetch_add_explicit(&g_handle_counter, 1, memory_order_relaxed) ^ g_handle_salt;
                ra->dev_ptr   = NULL;
                ra->host_ptr  = mmap_ptr;
                ra->size      = req_size;        /* PR-RR: BE-safe */
                ra->device_id = (int)req_device_id;
                ra->tier      = 2;
                ra->in_use    = 1;
                ra->client_id = cli->feeder_id;
                /* PR-T: index this handle for O(1) lookup. */
                ra_hash_insert(ra->handle, (uint32_t)(ra - g_remote_allocs));
                resp.status        = GB_STATUS_OK;
                resp.tier_used     = 2;
                resp.remote_handle = ra->handle;
            } else {
                munmap(mmap_ptr, (size_t)req_size);
                resp.status = GB_STATUS_ERR_OOM;
            }
            pthread_mutex_unlock(&g_alloc_lock);
            if (resp.status == GB_STATUS_OK) {
                ag_record_alloc(cli->feeder_id, (size_t)req_size);
                netd_log("T3 mmap(%llu MB) → handle=0x%llx",
                         (unsigned long long)(req_size >> 20),
                         (unsigned long long)resp.remote_handle);
                NETD_EVT("ALLOC_T3_MMAP", "T3_NVMe", req_size >> 20, resp.remote_handle, "mmap_ok");
                atomic_fetch_sub_explicit(&g_inflight_ops, 1, memory_order_acq_rel);
                return send_msg(cli, GB_MSG_CUDA_MALLOC, GB_NET_FLAG_RESPONSE,
                                &resp, sizeof(resp));
            }
        }
        netd_log("T3 mmap(%llu MB) failed", (unsigned long long)(req_size >> 20));
        NETD_EVT("OOM_T3_MMAP", "T3_NVMe", req_size >> 20, 0, "mmap_failed");
    }

    resp.status    = GB_STATUS_ERR_OOM;
    resp.tier_used = (req_tier == GB_ALLOC_TIER_T2) ? 1
                   : (req_tier == GB_ALLOC_TIER_T3) ? 2
                   : (uint8_t)(req_tier & 0x0F);
    atomic_fetch_sub_explicit(&g_inflight_ops, 1, memory_order_acq_rel);
    return send_msg(cli, GB_MSG_CUDA_MALLOC, GB_NET_FLAG_RESPONSE,
                    &resp, sizeof(resp));
}

static int handle_cuda_free(struct client *cli, const void *payload, uint32_t len)
{
    if (len < sizeof(struct gb_net_cuda_free))
        return -1;

    const struct gb_net_cuda_free *req = (const struct gb_net_cuda_free *)payload;
    /* PR-PP: BE-safe field read.  remote_handle is the only payload field. */
    gb_u64 req_remote_handle = GB_LE_U64(req->remote_handle);

    struct gb_net_response resp = {
        .orig_msg_type = GB_MSG_CUDA_FREE,
        .status        = GB_STATUS_OK,
    };

    pthread_mutex_lock(&g_alloc_lock);
    struct remote_alloc *ra = ra_find(req_remote_handle);
    if (ra) {
        if (ra->tier == 0) {
            if (f_cudaSetDevice) f_cudaSetDevice(ra->device_id);
            if (f_cudaFree) f_cudaFree(ra->dev_ptr);
        } else if (ra->tier == 1) {
            if (f_cudaFreeHost) f_cudaFreeHost(ra->host_ptr);
        } else {
            munmap(ra->host_ptr, (size_t)ra->size);
        }
        ag_record_free(ra->client_id, (size_t)ra->size);
        netd_log("free tier%d(0x%llx)", ra->tier, (unsigned long long)req_remote_handle);
        NETD_EVT("FREE_OK", ra->tier == 0 ? "T1_GPU" : ra->tier == 1 ? "T2_DDR" : "T3_NVMe",
                 ra->size >> 20, req_remote_handle, "alloc_freed");
        /* PR-T: tombstone the hash entry before clearing in_use so a
         * concurrent ra_find can't see the slot transition mid-flight. */
        ra_hash_remove(ra->handle, (uint32_t)(ra - g_remote_allocs));
        ra->in_use = 0;
    } else {
        resp.status = GB_STATUS_ERR_INVALID;
        netd_log("free: unknown handle 0x%llx", (unsigned long long)req_remote_handle);
    }
    pthread_mutex_unlock(&g_alloc_lock);

    return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                    &resp, sizeof(resp));
}

static int handle_cuda_memcpy_h2d(struct client *cli, const void *payload,
                                  uint32_t len, uint16_t flags)
{
    if (len < sizeof(struct gb_net_cuda_memcpy))
        return -1;

    const struct gb_net_cuda_memcpy *req = (const struct gb_net_cuda_memcpy *)payload;
    const void *data = (const uint8_t *)payload + sizeof(struct gb_net_cuda_memcpy);
    uint32_t data_len = len - (uint32_t)sizeof(struct gb_net_cuda_memcpy);
    /* PR-PP: read wire fields through BE-safe accessors at function entry,
     * use the locals throughout.  On LE these compile to identity. */
    gb_u64 req_remote_handle = GB_LE_U64(req->remote_handle);
    gb_u64 req_offset        = GB_LE_U64(req->offset);
    gb_u64 req_size          = GB_LE_U64(req->size);

    struct gb_net_response resp = {
        .orig_msg_type = GB_MSG_CUDA_MEMCPY_H2D,
        .status        = GB_STATUS_OK,
    };

    /* Fabric zstd: payload is compressed. Decompress into a reusable scratch
     * (netd is single-threaded per connection loop) sized to the wire-declared
     * uncompressed req_size, then treat it as the real data. On any failure
     * reject with ERR_INVALID rather than write garbage. */
    if (flags & GB_NET_FLAG_COMP_ZSTD) {
#ifdef GB_HAVE_ZSTD
        static uint8_t *zscratch = NULL;
        static size_t   zscratch_cap = 0;
        if (req_size == 0 || req_size > (uint64_t)GB_NET_MAX_MSG_SIZE) {
            resp.status = GB_STATUS_ERR_INVALID;
            return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                            &resp, sizeof(resp));
        }
        if (zscratch_cap < req_size) {
            uint8_t *nb = (uint8_t *)realloc(zscratch, (size_t)req_size);
            if (!nb) {
                resp.status = GB_STATUS_ERR_OOM;
                return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                                &resp, sizeof(resp));
            }
            zscratch = nb; zscratch_cap = (size_t)req_size;
        }
        size_t dsize = ZSTD_decompress(zscratch, (size_t)req_size, data, data_len);
        if (ZSTD_isError(dsize) || dsize != (size_t)req_size) {
            netd_log("ERROR H2D: zstd decompress failed (%s) clen=%u ulen=%llu",
                     ZSTD_isError(dsize) ? ZSTD_getErrorName(dsize) : "size mismatch",
                     data_len, (unsigned long long)req_size);
            resp.status = GB_STATUS_ERR_INVALID;
            return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                            &resp, sizeof(resp));
        }
        data = zscratch;
        data_len = (uint32_t)dsize;
#else
        /* Host asked for compression a non-zstd build can't undo. */
        netd_log("ERROR H2D: COMP_ZSTD flag but daemon built without libzstd");
        resp.status = GB_STATUS_ERR_INVALID;
        return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                        &resp, sizeof(resp));
#endif
    }

    /* PR-II: hold g_alloc_lock across the entire memcpy + accounting access.
     *
     * Previously the pattern was: lock → ra_find → UNLOCK → use ra.  That
     * was safe ONLY because netd processes all client I/O on a single
     * thread.  Holding the lock across the CUDA call removes the
     * single-thread invariant constraint so future async-pipelining of
     * netd doesn't reintroduce the UAF footgun.  See PR-HH audit #2.
     *
     * Trade-off: any other ra_find caller blocks for the duration of the
     * memcpy.  In single-threaded netd this is moot (we wouldn't be in
     * another handler anyway).  In future multi-threaded netd, the lock
     * scopes naturally to per-handle granularity once we shard.  Snapshot
     * ra fields into locals before any cuda call so the CUDA stub sees
     * stable values even if a stray future bug somehow mutates ra during
     * the memcpy. */
    pthread_mutex_lock(&g_alloc_lock);
    struct remote_alloc *ra = ra_find(req_remote_handle);
    if (!ra) {
        pthread_mutex_unlock(&g_alloc_lock);
        resp.status = GB_STATUS_ERR_INVALID;
    } else if (req_offset > (uint64_t)ra->size ||
               req_size > (uint64_t)ra->size - req_offset) {  /* N-M3: no wrap */
        /* Audit F-L3-10: bound-check offset + size against the allocation. */
        netd_log("WARN H2D: out-of-bounds offset=%llu size=%llu vs alloc=%zu",
                 (unsigned long long)req_offset,
                 (unsigned long long)req_size, ra->size);
        pthread_mutex_unlock(&g_alloc_lock);
        resp.status = GB_STATUS_ERR_INVALID;
    } else {
        /* Snapshot fields under lock for use in the CUDA call. */
        uint8_t  ra_tier      = ra->tier;
        int      ra_device_id = ra->device_id;
        void    *ra_dev_ptr   = ra->dev_ptr;
        void    *ra_host_ptr  = ra->host_ptr;
        size_t   copy_size    = (data_len < (uint32_t)req_size) ? data_len : (size_t)req_size;
        if (ra_tier == 0) {
            /* T1 GPU: cudaMemcpy H2D */
            if (f_cudaSetDevice) f_cudaSetDevice(ra_device_id);
            void *dst = (void *)((uintptr_t)ra_dev_ptr + (size_t)req_offset);
            cudaError_t err = f_cudaMemcpy
                ? f_cudaMemcpy(dst, data, copy_size, GB_cudaMemcpyHostToDevice) : 1;
            if (err != 0) resp.status = GB_STATUS_ERR_CUDA;
        } else {
            /* T2/T3: host memory - plain memcpy */
            memcpy((char *)ra_host_ptr + (size_t)req_offset, data, copy_size);
        }
        pthread_mutex_unlock(&g_alloc_lock);
        netd_log("memcpy H2D tier%d: %llu MB → handle 0x%llx+%llu",
                 ra_tier, (unsigned long long)(copy_size >> 20),
                 (unsigned long long)req_remote_handle,
                 (unsigned long long)req_offset);
        NETD_EVT("MEMCPY_H2D", ra_tier == 0 ? "T1_GPU" : "T2_DDR",
                 copy_size >> 20, req_remote_handle, "h2d_done");
    }

    return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                    &resp, sizeof(resp));
}

/* GB_MSG_CUDA_MEMSET - remote memset on a feeder allocation.  Mirrors the
 * H2D handler's locking/bounds pattern.  ggml memsets quantized-tensor
 * padding right after upload; without this, feeder-resident buffers abort
 * the client with cudaErrorInvalidValue on first model load. */
static int handle_cuda_memset(struct client *cli, const void *payload, uint32_t len)
{
    if (len < sizeof(struct gb_net_cuda_memset))
        return -1;

    const struct gb_net_cuda_memset *req = (const struct gb_net_cuda_memset *)payload;
    gb_u64 req_remote_handle = GB_LE_U64(req->remote_handle);
    gb_u64 req_offset        = GB_LE_U64(req->offset);
    gb_u64 req_size          = GB_LE_U64(req->size);
    gb_u32 req_value         = GB_LE_U32(req->value);

    struct gb_net_response resp = {
        .orig_msg_type = GB_MSG_CUDA_MEMSET,
        .status        = GB_STATUS_OK,
    };

    pthread_mutex_lock(&g_alloc_lock);
    struct remote_alloc *ra = ra_find(req_remote_handle);
    if (!ra) {
        pthread_mutex_unlock(&g_alloc_lock);
        resp.status = GB_STATUS_ERR_INVALID;
    } else if (req_offset > (uint64_t)ra->size ||
               req_size > (uint64_t)ra->size - req_offset) {
        netd_log("WARN MEMSET: out-of-bounds offset=%llu size=%llu vs alloc=%zu",
                 (unsigned long long)req_offset,
                 (unsigned long long)req_size, ra->size);
        pthread_mutex_unlock(&g_alloc_lock);
        resp.status = GB_STATUS_ERR_INVALID;
    } else {
        uint8_t  ra_tier      = ra->tier;
        int      ra_device_id = ra->device_id;
        void    *ra_dev_ptr   = ra->dev_ptr;
        void    *ra_host_ptr  = ra->host_ptr;
        if (ra_tier == 0) {
            if (f_cudaSetDevice) f_cudaSetDevice(ra_device_id);
            void *dst = (void *)((uintptr_t)ra_dev_ptr + (size_t)req_offset);
            cudaError_t err = f_cudaMemset
                ? f_cudaMemset(dst, (int)req_value, (size_t)req_size) : 1;
            if (err != 0) resp.status = GB_STATUS_ERR_CUDA;
        } else {
            memset((char *)ra_host_ptr + (size_t)req_offset,
                   (int)req_value, (size_t)req_size);
        }
        pthread_mutex_unlock(&g_alloc_lock);
        netd_log("memset tier%d: %llu KB val=%u → handle 0x%llx+%llu",
                 ra_tier, (unsigned long long)(req_size >> 10), req_value,
                 (unsigned long long)req_remote_handle,
                 (unsigned long long)req_offset);
    }

    return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                    &resp, sizeof(resp));
}

static int handle_cuda_memcpy_d2h(struct client *cli, const void *payload, uint32_t len)
{
    if (len < sizeof(struct gb_net_cuda_memcpy))
        return -1;

    const struct gb_net_cuda_memcpy *req = (const struct gb_net_cuda_memcpy *)payload;
    /* PR-PP: BE-safe wire-field reads.  All subsequent code must use the
     * locals, not raw req-> accesses. */
    gb_u64 req_remote_handle = GB_LE_U64(req->remote_handle);
    gb_u64 req_offset        = GB_LE_U64(req->offset);
    gb_u64 req_size          = GB_LE_U64(req->size);

    /* PR-RR audit #3: lift the request-size cap-check + resp_buf malloc
     * OUT of the locked region.  Pre-fix, a 4 MB malloc held g_alloc_lock
     * for the duration of any glibc arena search / mmap syscall - up to
     * milliseconds under fragmentation.  None of this work needs ra. */
    size_t max_payload = (size_t)GB_NET_MAX_MSG_SIZE
                       - sizeof(struct gb_net_response)
                       - (cli->mac_enabled ? 8u : 0u);
    if (req_size > (uint64_t)max_payload) {
        netd_log("WARN D2H: requested size %llu exceeds wire cap %zu - rejecting",
                 (unsigned long long)req_size, max_payload);
        struct gb_net_response resp = {
            .orig_msg_type = GB_MSG_CUDA_MEMCPY_D2H,
            .status        = GB_STATUS_ERR_INVALID,
        };
        return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                        &resp, sizeof(resp));
    }
    size_t copy_size = (size_t)req_size;
    size_t resp_total = sizeof(struct gb_net_response) + copy_size;
    uint8_t *resp_buf = (uint8_t *)malloc(resp_total);
    if (!resp_buf) {
        struct gb_net_response resp = {
            .orig_msg_type = GB_MSG_CUDA_MEMCPY_D2H,
            .status        = GB_STATUS_ERR_OOM,
        };
        return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                        &resp, sizeof(resp));
    }
    struct gb_net_response *resp_hdr = (struct gb_net_response *)resp_buf;
    resp_hdr->orig_msg_type = GB_MSG_CUDA_MEMCPY_D2H;
    resp_hdr->status        = GB_STATUS_OK;
    resp_hdr->_pad          = 0;

    /* PR-II: take lock; do bounds check against ra; snapshot ra fields;
     * do CUDA memcpy under lock; release.  Snapshot pattern means the
     * CUDA stub sees stable values regardless of a future writer thread. */
    pthread_mutex_lock(&g_alloc_lock);
    struct remote_alloc *ra = ra_find(req_remote_handle);

    if (!ra) {
        pthread_mutex_unlock(&g_alloc_lock);
        free(resp_buf);
        struct gb_net_response resp = {
            .orig_msg_type = GB_MSG_CUDA_MEMCPY_D2H,
            .status        = GB_STATUS_ERR_INVALID,
        };
        return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                        &resp, sizeof(resp));
    }

    /* Audit F-L3-10 (D2H mirror): bound-check offset + size against the alloc. */
    if (req_offset > ra->size ||
        req_offset + req_size > (uint64_t)ra->size) {
        netd_log("WARN D2H: out-of-bounds offset=%llu size=%llu vs alloc=%zu",
                 (unsigned long long)req_offset,
                 (unsigned long long)req_size, ra->size);
        pthread_mutex_unlock(&g_alloc_lock);
        free(resp_buf);
        struct gb_net_response resp = {
            .orig_msg_type = GB_MSG_CUDA_MEMCPY_D2H,
            .status        = GB_STATUS_ERR_INVALID,
        };
        return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                        &resp, sizeof(resp));
    }

    /* Snapshot ra fields under lock. */
    uint8_t  ra_tier      = ra->tier;
    int      ra_device_id = ra->device_id;
    void    *ra_dev_ptr   = ra->dev_ptr;
    void    *ra_host_ptr  = ra->host_ptr;
    uint64_t ra_handle    = ra->handle;

    if (ra_tier == 0) {
        /* T1 GPU: cudaMemcpy D2H */
        if (f_cudaSetDevice) f_cudaSetDevice(ra_device_id);
        void *src = (void *)((uintptr_t)ra_dev_ptr + (size_t)req_offset);
        if (!f_cudaMemcpy) { resp_hdr->status = GB_STATUS_ERR_CUDA; }
        else {
            cudaError_t err = f_cudaMemcpy(resp_buf + sizeof(struct gb_net_response),
                                            src, copy_size, GB_cudaMemcpyDeviceToHost);
            if (err != 0) resp_hdr->status = GB_STATUS_ERR_CUDA;
        }
    } else {
        /* T2/T3: host memory - plain memcpy */
        memcpy(resp_buf + sizeof(struct gb_net_response),
               (char *)ra_host_ptr + (size_t)req_offset, copy_size);
    }
    pthread_mutex_unlock(&g_alloc_lock);

    netd_log("memcpy D2H tier%d: %llu MB ← handle 0x%llx+%llu",
             ra_tier, (unsigned long long)(copy_size >> 20),
             (unsigned long long)ra_handle,
             (unsigned long long)req_offset);
    NETD_EVT("MEMCPY_D2H", ra_tier == 0 ? "T1_GPU" : "T2_DDR",
             copy_size >> 20, ra_handle, "d2h_done");

    int ret = send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                       resp_buf, (uint32_t)resp_total);
    free(resp_buf);
    return ret;
}

static int handle_cuda_memcpy_d2d(struct client *cli, const void *payload, uint32_t len)
{
    if (len != sizeof(struct gb_net_cuda_memcpy_d2d)) {
        struct gb_net_response err_resp = { .orig_msg_type = GB_MSG_CUDA_MEMCPY_D2D, .status = GB_STATUS_ERR_INVALID };
        return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE, &err_resp, sizeof(err_resp));
    }

    const struct gb_net_cuda_memcpy_d2d *req = payload;
    struct gb_net_response resp = {
        .orig_msg_type = GB_MSG_CUDA_MEMCPY_D2D,
        .status = GB_STATUS_OK
    };

#ifdef GREENBOOST_USE_NCCL
    if (!cli->nccl_comm) {
        resp.status = GB_STATUS_ERR_NCCL;
        return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE, &resp, sizeof(resp));
    }

    /* Resolve the opaque handle to the actual GPU pointer via ra_find().
     * The handle is counter^salt - it is NOT a device address.
     *
     * PR-II: snapshot ra->dev_ptr under the lock before releasing, so the
     * subsequent ncclSend/ncclRecv (which can block on collectives) cannot
     * race a hypothetical future writer thread mutating ra. */
    /* PR-UU: BE-safe field reads. */
    gb_u64 req_src_handle = GB_LE_U64(req->src_handle);
    gb_u64 req_dst_handle = GB_LE_U64(req->dst_handle);
    gb_u64 req_size_d2d   = GB_LE_U64(req->size);
    uint64_t lookup_handle = req_src_handle ? req_src_handle : req_dst_handle;
    pthread_mutex_lock(&g_alloc_lock);
    struct remote_alloc *ra = ra_find(lookup_handle);
    if (!ra || ra->tier != 0) {
        pthread_mutex_unlock(&g_alloc_lock);
        resp.status = GB_STATUS_ERR_INVALID;
        return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE, &resp, sizeof(resp));
    }
    void *ra_dev_ptr = ra->dev_ptr;
    pthread_mutex_unlock(&g_alloc_lock);

    /* Send OK response so the host knows we are ready for NCCL */
    if (send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE, &resp, sizeof(resp)) < 0) return -1;

    ncclResult_t nr;
    if (req_src_handle) {
        /* Host is pulling from us. We send. */
        nr = ncclSend(ra_dev_ptr, req_size_d2d, ncclChar, 0, cli->nccl_comm, NULL);
        if (nr != ncclSuccess) netd_log("ERR: ncclSend failed: %d", nr);
    } else {
        /* Host is pushing to us. We recv. */
        nr = ncclRecv(ra_dev_ptr, req_size_d2d, ncclChar, 0, cli->nccl_comm, NULL);
        if (nr != ncclSuccess) netd_log("ERR: ncclRecv failed: %d", nr);
    }
    return 0; /* NCCL handled it */
#else
    resp.status = GB_STATUS_ERR_INVALID;
    return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE, &resp, sizeof(resp));
#endif
}

/* ------------------------------------------------------------------ */
/*  Phase 3: Kernel launch handler                                     */
/* ------------------------------------------------------------------ */

/* PR-D: forward declaration so handle_cuda_launch can gate on the
 * allowlist.  The implementation lives below alongside the other Phase-3
 * helpers. */
static int gb_kernel_name_allowed(const char *kname);

static int handle_cuda_launch(struct client *cli, const void *payload, uint32_t len)
{
    if (len < sizeof(struct gb_net_cuda_launch))
        return -1;

    const struct gb_net_cuda_launch *req = (const struct gb_net_cuda_launch *)payload;
    const uint8_t *data = (const uint8_t *)payload + sizeof(struct gb_net_cuda_launch);
    /* PR-UU: BE-safe field reads.  All raw `req->FIELD` access below now
     * goes through these locals. */
    gb_u32 req_kernel_name_len = GB_LE_U32(req->kernel_name_len);
    gb_u32 req_arg_buffer_size = GB_LE_U32(req->arg_buffer_size);
    gb_u32 req_grid_x          = GB_LE_U32(req->grid_x);
    gb_u32 req_grid_y          = GB_LE_U32(req->grid_y);
    gb_u32 req_grid_z          = GB_LE_U32(req->grid_z);
    gb_u32 req_block_x         = GB_LE_U32(req->block_x);
    gb_u32 req_block_y         = GB_LE_U32(req->block_y);
    gb_u32 req_block_z         = GB_LE_U32(req->block_z);
    gb_u32 req_shared_mem_bytes= GB_LE_U32(req->shared_mem_bytes);

    /* PR-C/C6: validate sizes individually in 64-bit and against a hard cap.
     * The previous check did `sizeof(...) + name_len + arg_buf_size > len` in
     * 32-bit unsigned arithmetic; an attacker chooses kernel_name_len near
     * 0xFFFFFFE0 so the sum wraps below `len`, the bounds check passes, and
     * the unchecked req->arg_buffer_size is then handed to malloc()+memcpy().
     * Promote to uint64 and bound each field against its own logical cap so
     * the same vector cannot tunnel through. */
    {
        uint64_t name_len_u64 = (uint64_t)req_kernel_name_len;
        uint64_t arg_buf_u64  = (uint64_t)req_arg_buffer_size;
        uint64_t hdr_u64      = (uint64_t)sizeof(struct gb_net_cuda_launch);
        if (name_len_u64 > (uint64_t)GB_NET_MAX_KERNEL_NAME ||
            arg_buf_u64  > (uint64_t)GB_NET_MAX_MSG_SIZE ||
            hdr_u64 + name_len_u64 + arg_buf_u64 > (uint64_t)len) {
            struct gb_net_response resp = {
                .orig_msg_type = GB_MSG_CUDA_LAUNCH,
                .status        = GB_STATUS_ERR_INVALID,
            };
            return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                            &resp, sizeof(resp));
        }
    }

    char kernel_name[GB_NET_MAX_KERNEL_NAME];
    uint32_t name_len = req_kernel_name_len;  /* PR-UU */
    if (name_len >= GB_NET_MAX_KERNEL_NAME) name_len = GB_NET_MAX_KERNEL_NAME - 1;
    memcpy(kernel_name, data, name_len);
    kernel_name[name_len] = '\0';

    const void *arg_buffer = data + req_kernel_name_len;  /* PR-UU */

    /* PR-D/security: gate the launch on the same kernel-name allowlist that
     * GB_MSG_CUDA_EXEC and GB_MSG_CUDA_REGISTER_FN use.  Without this check,
     * a stale entry in g_kernel_map (never invalidated on client disconnect)
     * lets a connected client launch any previously-registered host_func by
     * name - including across tenants when the daemon serves more than one
     * host.  This matches the design intent and closes a known gap. */
    if (!gb_kernel_name_allowed(kernel_name)) {
        struct gb_net_response resp = {
            .orig_msg_type = GB_MSG_CUDA_LAUNCH,
            .status        = GB_STATUS_ERR_INVALID,
        };
        netd_log("WARN: kernel '%s' rejected by allowlist", kernel_name);
        return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                        &resp, sizeof(resp));
    }

    /* Look up the kernel by name in our local kernel map, SCOPED to this
     * client (PR-L: cross-tenant isolation - kernel registered by client A
     * is not callable by client B, even if both registered the same name).
     * If not found in the client's scope, try dlsym as a fallback (works for
     * globally visible kernels).
     *
     * PR-V: O(1) hashmap lookup, was O(MAX_KERNEL_NAMES=4096) linear strcmp. */
    pthread_mutex_lock(&g_kernel_map_lock);
    void *host_func = km_lookup(cli->feeder_id, kernel_name);
    pthread_mutex_unlock(&g_kernel_map_lock);

    struct gb_net_response resp = {
        .orig_msg_type = GB_MSG_CUDA_LAUNCH,
        .status        = GB_STATUS_OK,
    };

    if (!host_func) {
        netd_log("WARN: kernel '%s' not found in local map", kernel_name);
        resp.status = GB_STATUS_ERR_INVALID;
    } else {
        /* Launch using cudaLaunchKernel with the flat arg buffer via 'extra' */
        /* We use the CUDA driver API cuLaunchKernel which accepts a flat buffer */
        typedef int (*pfn_cuLaunchKernel)(void *, unsigned int, unsigned int, unsigned int,
                                          unsigned int, unsigned int, unsigned int,
                                          unsigned int, void *, void **, void **);
        static pfn_cuLaunchKernel f_cuLaunchKernel = NULL;
        if (!f_cuLaunchKernel) {
            void *libcuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
            if (!libcuda) libcuda = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
            if (libcuda)
                f_cuLaunchKernel = (pfn_cuLaunchKernel)dlsym(libcuda, "cuLaunchKernel");
        }

        if (f_cuLaunchKernel && req_arg_buffer_size > 0) {  /* PR-UU */
            /* Use CU_LAUNCH_PARAM_BUFFER_POINTER + CU_LAUNCH_PARAM_BUFFER_SIZE */
            void *arg_buf_copy = malloc(req_arg_buffer_size);  /* PR-UU */
            if (arg_buf_copy) {
                memcpy(arg_buf_copy, arg_buffer, req_arg_buffer_size);  /* PR-UU */
                size_t arg_size = req_arg_buffer_size;

                #define CU_LAUNCH_PARAM_BUFFER_POINTER ((void *)0x01)
                #define CU_LAUNCH_PARAM_BUFFER_SIZE    ((void *)0x02)
                #define CU_LAUNCH_PARAM_END            ((void *)0x00)

                void *extra[] = {
                    CU_LAUNCH_PARAM_BUFFER_POINTER, arg_buf_copy,
                    CU_LAUNCH_PARAM_BUFFER_SIZE,    &arg_size,
                    CU_LAUNCH_PARAM_END
                };

                /* F-L3-06: dispatch kernel in a worker thread; pthread_timedjoin_np
                 * with a 60 s deadline.  If the GPU hangs, cudaDeviceReset() is
                 * called and the connection is torn down rather than blocking the
                 * entire single-threaded event loop indefinitely. */
                struct { pfn_cuLaunchKernel f; void *hf;
                         unsigned int gx,gy,gz,bx,by,bz,smem;
                         void **extra; int result; } kargs = {
                    .f = f_cuLaunchKernel, .hf = host_func,
                    .gx = req_grid_x,  .gy = req_grid_y,  .gz = req_grid_z,
                    .bx = req_block_x, .by = req_block_y, .bz = req_block_z,
                    .smem = req_shared_mem_bytes, .extra = extra, .result = -1
                };
                void *_kthread_fn(void *a) {
                    typeof(kargs) *k = a;
                    k->result = k->f(k->hf, k->gx,k->gy,k->gz,
                                     k->bx,k->by,k->bz,
                                     k->smem, NULL, NULL, k->extra);
                    if (k->result == 0) {
                        typedef int (*pfn_cudaSync)(void);
                        static pfn_cudaSync f_sync = NULL;
                        if (!f_sync) f_sync = (pfn_cudaSync)dlsym(RTLD_DEFAULT, "cudaDeviceSynchronize");
                        if (f_sync) f_sync();
                    }
                    return NULL;
                }
                pthread_t ktid;
                int err = -1;
                if (pthread_create(&ktid, NULL, _kthread_fn, &kargs) == 0) {
                    struct timespec deadline;
                    clock_gettime(CLOCK_REALTIME, &deadline);
                    deadline.tv_sec += 60;
                    int join_rc = pthread_timedjoin_np(ktid, NULL, &deadline);
                    if (join_rc == ETIMEDOUT) {
                        typedef int (*pfn_reset)(void);
                        pfn_reset f_rst = (pfn_reset)dlsym(RTLD_DEFAULT, "cudaDeviceReset");
                        if (f_rst) f_rst();
                        pthread_cancel(ktid);
                        pthread_join(ktid, NULL);
                        netd_log("ERROR: cuLaunchKernel '%s' hung for 60 s - device reset", kernel_name);
                        err = 999; /* CUDA_ERROR_UNKNOWN proxy */
                    } else {
                        err = kargs.result;
                    }
                }
                if (err != 0) {
                    resp.status = GB_STATUS_ERR_CUDA;
                    netd_log("cuLaunchKernel '%s' failed: err=%d", kernel_name, err);
                } else {
                    netd_log("cuLaunchKernel '%s' grid=(%u,%u,%u) block=(%u,%u,%u) args=%u bytes",
                             kernel_name,
                             req_grid_x, req_grid_y, req_grid_z,
                             req_block_x, req_block_y, req_block_z,
                             req_arg_buffer_size);  /* PR-UU */
                }
                free(arg_buf_copy);
            } else {
                resp.status = GB_STATUS_ERR_OOM;
            }
        } else {
            resp.status = GB_STATUS_ERR_INVALID;
            netd_log("WARN: cuLaunchKernel not available or no args for '%s'", kernel_name);
        }
    }

    return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                    &resp, sizeof(resp));
}

static int gb_kernel_name_allowed(const char *kname);

/* RAW exec path (GB_NET_FLAG_EXEC_RAW): full packed param buffer + byte-offset
 * relocations.  Handles struct-by-value args (ggml fused mul_mat_vec_q, torch
 * descriptor structs) whose size >8B and whose embedded pointers must be
 * rewritten , the 8-byte-per-arg path truncates them → err=700. */
static int handle_cuda_exec_raw(struct client *cli, const void *payload, uint32_t len)
{
    struct gb_net_response err_resp = {
        .orig_msg_type = GB_MSG_CUDA_EXEC, .status = GB_STATUS_ERR_INVALID };
    #define RAW_ERR() do { netd_log("exec_raw rejected at %s:%d", __FILE__, __LINE__); \
        return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE, &err_resp, sizeof(err_resp)); } while (0)
    if (len < sizeof(struct gb_net_cuda_exec_raw)) RAW_ERR();

    const struct gb_net_cuda_exec_raw *req = (const struct gb_net_cuda_exec_raw *)payload;
    uint32_t name_len   = GB_LE_U32(req->kernel_name_len);
    uint32_t n_params   = GB_LE_U32(req->n_params);
    uint32_t n_relocs   = GB_LE_U32(req->n_relocs);
    uint32_t buf_bytes  = GB_LE_U32(req->param_buf_bytes);
    if (n_params > 64 || n_relocs > 128) RAW_ERR();

    size_t need = sizeof(*req) + name_len + n_params * sizeof(uint32_t)
                + (size_t)n_relocs * sizeof(struct gb_net_ptr_reloc) + buf_bytes;
    if (len < need) RAW_ERR();

    const uint8_t *p = (const uint8_t *)payload + sizeof(*req);
    char kernel_name[GB_NET_MAX_KERNEL_NAME];
    if (name_len == 0 || name_len >= GB_NET_MAX_KERNEL_NAME) RAW_ERR();
    memcpy(kernel_name, p, name_len); kernel_name[name_len] = '\0'; p += name_len;

    const uint32_t *param_size = (const uint32_t *)p;  p += n_params * sizeof(uint32_t);
    const struct gb_net_ptr_reloc *relocs = (const struct gb_net_ptr_reloc *)p;
    p += (size_t)n_relocs * sizeof(struct gb_net_ptr_reloc);
    const uint8_t *param_buf_wire = p;

    /* Resolve kernel (same order as handle_cuda_exec). */
    pthread_mutex_lock(&g_kernel_map_lock);
    void *host_func = km_lookup(cli->feeder_id, kernel_name);
    pthread_mutex_unlock(&g_kernel_map_lock);
    int is_captured_stub = 0;
    if (!host_func) {
        if (!gb_kernel_name_allowed(kernel_name)) {
            netd_log("WARN exec_raw: '%s' not on allowlist - rejected", kernel_name);
            RAW_ERR();
        }
        host_func = gb_kernel_resolve(kernel_name);
        if (!host_func) {
            static const void *(*cap_lookup)(const char *) = NULL;
            static int cap_probed = 0;
            if (!cap_probed) { cap_probed = 1;
                cap_lookup = (const void *(*)(const char *))dlsym(RTLD_DEFAULT, "gb_capture_lookup"); }
            if (cap_lookup) { host_func = (void *)cap_lookup(kernel_name);
                              if (host_func) is_captured_stub = 1; }
        }
        if (!host_func) host_func = gb_fatbin_resolve(kernel_name);  /* CUfunction */
    }
    if (!host_func) { netd_log("WARN exec_raw: '%s' not found", kernel_name); RAW_ERR(); }

    /* Mutable copy of the param buffer; apply byte-offset relocations. */
    if (buf_bytes > 65536) RAW_ERR();
    uint8_t *pbuf = (uint8_t *)malloc(buf_bytes ? buf_bytes : 1);
    if (!pbuf) RAW_ERR();
    memcpy(pbuf, param_buf_wire, buf_bytes);

    pthread_mutex_lock(&g_alloc_lock);
    for (uint32_t i = 0; i < n_relocs; i++) {
        uint32_t boff = GB_LE_U32(relocs[i].arg_idx);   /* byte offset into pbuf */
        uint64_t rh   = GB_LE_U64(relocs[i].remote_handle);
        if (boff >= buf_bytes || buf_bytes - boff < 8) continue;  /* overflow-safe bound check */
        struct remote_alloc *ra = ra_find(rh);
        if (!ra) { pthread_mutex_unlock(&g_alloc_lock); free(pbuf);
                   netd_log("WARN exec_raw: unknown handle 0x%llx", (unsigned long long)rh);
                   RAW_ERR(); }
        void *real = (ra->tier == 0) ? ra->dev_ptr : ra->host_ptr;
        memcpy(pbuf + boff, &real, 8);
    }
    pthread_mutex_unlock(&g_alloc_lock);

    /* Build kernelParams: kp[i] → pbuf at each param's 8-byte-aligned offset
     * (must match the host's packing in cudaLaunchKernelExC). */
    void *kp[64];
    uint32_t off = 0;
    for (uint32_t i = 0; i < n_params; i++) {
        uint32_t sz  = GB_LE_U32(param_size[i]); if (!sz) sz = 8;
        uint32_t asz = (sz + 7u) & ~7u;
        if (off + asz > buf_bytes) { free(pbuf); RAW_ERR(); }
        kp[i] = pbuf + off;
        off += asz;
    }

    struct gb_net_response resp = { .orig_msg_type = GB_MSG_CUDA_EXEC, .status = GB_STATUS_OK };
    int err = -1;
    if (is_captured_stub) {
        typedef struct { unsigned x, y, z; } d3;
        typedef int (*pfn_rt)(const void *, d3, d3, void **, size_t, void *);
        static pfn_rt f_rt = NULL; static int pr = 0;
        if (!pr) { pr = 1; f_rt = (pfn_rt)dlsym(RTLD_DEFAULT, "cudaLaunchKernel"); }
        if (f_rt) {
            d3 g = { GB_LE_U32(req->grid_x), GB_LE_U32(req->grid_y), GB_LE_U32(req->grid_z) };
            d3 b = { GB_LE_U32(req->block_x), GB_LE_U32(req->block_y), GB_LE_U32(req->block_z) };
            err = f_rt(host_func, g, b, n_params ? kp : NULL,
                       (size_t)GB_LE_U32(req->shared_mem_bytes), NULL);
        }
    } else {
        typedef int (*pfn_dv)(void *, unsigned, unsigned, unsigned, unsigned, unsigned,
                              unsigned, unsigned, void *, void **, void **);
        static pfn_dv f_dv = NULL; static int pr = 0;
        if (!pr) { pr = 1; void *lc = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
                   if (!lc) lc = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
                   if (lc) f_dv = (pfn_dv)dlsym(lc, "cuLaunchKernel"); }
        if (f_dv)
            err = f_dv(host_func, GB_LE_U32(req->grid_x), GB_LE_U32(req->grid_y),
                       GB_LE_U32(req->grid_z), GB_LE_U32(req->block_x),
                       GB_LE_U32(req->block_y), GB_LE_U32(req->block_z),
                       GB_LE_U32(req->shared_mem_bytes), NULL,
                       n_params ? kp : NULL, NULL);
    }
    free(pbuf);

    if (err != 0) {
        resp.status = GB_STATUS_ERR_CUDA;
        netd_log("exec_raw: launch '%s' failed err=%d (%s)", kernel_name, err,
                 is_captured_stub ? "runtime" : "driver");
    } else {
        g_kernel_dispatch_count++;
        netd_log("exec_raw: '%s' OK (%u params, %u relocs) [%s]", kernel_name,
                 n_params, n_relocs, is_captured_stub ? "runtime-stub" : "driver");
    }
    return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE, &resp, sizeof(resp));
    #undef RAW_ERR
}

static int handle_cuda_exec(struct client *cli, const void *payload, uint32_t len)
{
    struct gb_net_response err_resp = {
        .orig_msg_type = GB_MSG_CUDA_EXEC,
        .status        = GB_STATUS_ERR_INVALID,
    };
#define EXEC_ERR() do { \
        netd_log("WARN exec: rejected at %s:%d", __FILE__, __LINE__); \
        return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE, &err_resp, sizeof(err_resp)); \
    } while (0)

    if (len < sizeof(struct gb_net_cuda_exec)) EXEC_ERR();

    const struct gb_net_cuda_exec *req = (const struct gb_net_cuda_exec *)payload;
    const uint8_t *p = (const uint8_t *)payload + sizeof(struct gb_net_cuda_exec);
    uint32_t left    = len - (uint32_t)sizeof(struct gb_net_cuda_exec);

    /* PR-UU: BE-safe field reads. */
    gb_u32 req_kernel_name_len = GB_LE_U32(req->kernel_name_len);
    gb_u32 req_n_arg_vals      = GB_LE_U32(req->n_arg_vals);
    gb_u32 req_n_relocs        = GB_LE_U32(req->n_relocs);
    gb_u32 req_n_uploads       = GB_LE_U32(req->n_uploads);
    gb_u32 req_n_downloads     = GB_LE_U32(req->n_downloads);
    gb_u32 req_grid_x          = GB_LE_U32(req->grid_x);
    gb_u32 req_grid_y          = GB_LE_U32(req->grid_y);
    gb_u32 req_grid_z          = GB_LE_U32(req->grid_z);
    gb_u32 req_block_x         = GB_LE_U32(req->block_x);
    gb_u32 req_block_y         = GB_LE_U32(req->block_y);
    gb_u32 req_block_z         = GB_LE_U32(req->block_z);
    gb_u32 req_shared_mem_bytes= GB_LE_U32(req->shared_mem_bytes);

    /* sanity caps */
    if (req_kernel_name_len == 0 || req_kernel_name_len >= GB_NET_MAX_KERNEL_NAME) EXEC_ERR();
    if (req_n_arg_vals > 256 || req_n_relocs > GB_NET_MAX_RELOCS) EXEC_ERR();
    if (req_n_uploads > GB_NET_MAX_XFERS || req_n_downloads > GB_NET_MAX_XFERS) EXEC_ERR();

    /* Audit F-L3-03: cumulative-size guard.  Per-field caps are checked
     * individually above, but a crafted message could still claim each at its
     * maximum and overflow `left`.  Compute the total demanded by the message
     * up front and reject if it exceeds what we received. */
    {
        uint64_t need = (uint64_t)req_kernel_name_len
                      + (uint64_t)req_n_arg_vals * sizeof(uint64_t)
                      + (uint64_t)req_n_relocs   * sizeof(struct gb_net_ptr_reloc)
                      + (uint64_t)req_n_uploads  * sizeof(struct gb_net_xfer_desc)
                      + (uint64_t)req_n_downloads * sizeof(struct gb_net_xfer_desc);  /* PR-UU */
        if (need > (uint64_t)left || need > GB_NET_MAX_MSG_SIZE) EXEC_ERR();
    }

    /* kernel name */
    if (left < req_kernel_name_len) EXEC_ERR();  /* PR-UU */
    char kernel_name[GB_NET_MAX_KERNEL_NAME];
    memcpy(kernel_name, p, req_kernel_name_len);  /* PR-UU */
    kernel_name[req_kernel_name_len] = '\0';
    p += req_kernel_name_len; left -= req_kernel_name_len;

    /* arg_vals */
    uint32_t arg_bytes = req_n_arg_vals * (uint32_t)sizeof(uint64_t);  /* PR-UU */
    if (left < arg_bytes) EXEC_ERR();
    uint64_t arg_vals[256] = {0};
    if (arg_bytes) memcpy(arg_vals, p, arg_bytes);
    p += arg_bytes; left -= arg_bytes;

    /* relocs */
    uint32_t reloc_bytes = req_n_relocs * (uint32_t)sizeof(struct gb_net_ptr_reloc);  /* PR-UU */
    if (left < reloc_bytes) EXEC_ERR();
    struct gb_net_ptr_reloc relocs[GB_NET_MAX_RELOCS];
    if (reloc_bytes) memcpy(relocs, p, reloc_bytes);
    p += reloc_bytes; left -= reloc_bytes;

    /* Audit F-L3-06: reject duplicate arg_idx in the relocation table.
     * Two relocations targeting the same arg slot lets an attacker overwrite
     * one allocation's handle with another, aliasing live and freed memory. */
    for (uint32_t i = 0; i < req_n_relocs; i++) {
        for (uint32_t j = i + 1; j < req_n_relocs; j++) {
            if (relocs[i].arg_idx == relocs[j].arg_idx) {
                netd_log("WARN exec: duplicate reloc arg_idx %u - rejecting",
                         relocs[i].arg_idx);
                EXEC_ERR();
            }
        }
    }

    /* upload descs + data */
    uint32_t up_desc_bytes = req_n_uploads * (uint32_t)sizeof(struct gb_net_xfer_desc);  /* PR-UU */
    if (left < up_desc_bytes) EXEC_ERR();
    struct gb_net_xfer_desc up_descs[GB_NET_MAX_XFERS];
    if (up_desc_bytes) memcpy(up_descs, p, up_desc_bytes);
    p += up_desc_bytes; left -= up_desc_bytes;

    /* N-H3: bound each descriptor individually before summing so an attacker
     * cannot craft up to GB_NET_MAX_XFERS entries that each underflow-wrap
     * upload_total past the aggregate cap check. */
    uint64_t upload_total = 0;
    for (uint32_t i = 0; i < req_n_uploads; i++) {
        if (up_descs[i].size > GB_NET_MAX_EXEC_UPLOAD_BYTES) EXEC_ERR();
        upload_total += up_descs[i].size;
    }
    if (upload_total > GB_NET_MAX_EXEC_UPLOAD_BYTES || left < upload_total) EXEC_ERR();
    const uint8_t *upload_data = p;
    p += upload_total; left -= upload_total;

    /* download descs */
    uint32_t dn_desc_bytes = req_n_downloads * (uint32_t)sizeof(struct gb_net_xfer_desc);  /* PR-UU */
    if (left < dn_desc_bytes) EXEC_ERR();
    struct gb_net_xfer_desc dn_descs[GB_NET_MAX_XFERS];
    if (dn_desc_bytes) memcpy(dn_descs, p, dn_desc_bytes);

    /* resolve kernel function - PR-L: scoped to this client only.
     * PR-V: O(1) hashmap lookup. */
    pthread_mutex_lock(&g_kernel_map_lock);
    void *host_func = km_lookup(cli->feeder_id, kernel_name);
    pthread_mutex_unlock(&g_kernel_map_lock);
    int is_captured_stub = 0;
    int is_fatbin_fn = 0;   /* real CUfunction from .nv_fatbin → driver launch */
    if (!host_func) {
        /* Audit F-L3-04: gate dlsym fallback behind the allowlist.  Fail-closed:
         * if /etc/greenboost/kernels.allow is absent, all kernels are rejected. */
        if (!gb_kernel_name_allowed(kernel_name)) {
            netd_log("WARN exec: kernel '%s' not on allowlist - rejected", kernel_name);
            EXEC_ERR();
        }
        host_func = gb_kernel_resolve(kernel_name);
        /* Stripped libggml-cuda.so: device stubs aren't in the symbol table,
         * so dlsym fails.  The __cudaRegisterFunction interposer
         * (greenboost_netd_capture.c, LD_PRELOAD'd into netd) captured every
         * name→host-stub at dlopen , use that.  A stub from here MUST be
         * launched via the RUNTIME cudaLaunchKernel, not the driver
         * cuLaunchKernel (see the launch site below). */
        if (!host_func) {
            static const void *(*cap_lookup)(const char *) = NULL;
            static int cap_probed = 0;
            if (!cap_probed) {
                cap_probed = 1;
                cap_lookup = (const void *(*)(const char *))
                    dlsym(RTLD_DEFAULT, "gb_capture_lookup");
            }
            if (cap_lookup) {
                host_func = (void *)cap_lookup(kernel_name);
                if (host_func) is_captured_stub = 1;
            }
        }
        /* General fallback (Option A): resolve from the .nv_fatbin directly as
         * a real CUfunction , covers kernels neither dlsym nor
         * __cudaRegisterFunction expose (ggml fused mmvq; torch image/video/
         * mesh kernels).  Launched via the DRIVER cuLaunchKernel below. */
        if (!host_func) {
            host_func = gb_fatbin_resolve(kernel_name);
            if (host_func) is_fatbin_fn = 1;
        }
    }

    if (!host_func) {
        netd_log("WARN exec: kernel '%s' not found", kernel_name);
        EXEC_ERR();
    }

    /* apply pointer relocations - T2 host_ptr is GPU zero-copy accessible;
     * T3 pageable ptr needs a temp device buffer (swap-in). */
    void *t3_tmp_dev[GB_NET_MAX_RELOCS] = {0};
    int n_t3_tmp = 0;

    pthread_mutex_lock(&g_alloc_lock);
    for (uint32_t i = 0; i < req_n_relocs; i++) {  /* PR-UU */
        uint32_t idx = relocs[i].arg_idx;
        /* Audit F-L3-05: bound at the actual write site against the local
         * arg_vals[256] capacity, not just against req->n_arg_vals. */
        if (idx >= req_n_arg_vals || idx >= 256) {
            pthread_mutex_unlock(&g_alloc_lock);
            for (int j = 0; j < n_t3_tmp; j++) if (t3_tmp_dev[j] && f_cudaFree) f_cudaFree(t3_tmp_dev[j]);
            EXEC_ERR();
        }
        struct remote_alloc *ra = ra_find(relocs[i].remote_handle);
        if (!ra) {
            netd_log("WARN exec: unknown handle 0x%llx for arg %u",
                     (unsigned long long)relocs[i].remote_handle, idx);
            pthread_mutex_unlock(&g_alloc_lock);
            for (int j = 0; j < n_t3_tmp; j++) if (t3_tmp_dev[j] && f_cudaFree) f_cudaFree(t3_tmp_dev[j]);
            EXEC_ERR();
        }
        if (ra->tier == 0) {
            /* T1: use GPU device pointer directly */
            arg_vals[idx] = (uint64_t)(uintptr_t)ra->dev_ptr;
        } else if (ra->tier == 1) {
            /* T2: cudaHostAlloc memory is GPU zero-copy accessible */
            arg_vals[idx] = (uint64_t)(uintptr_t)ra->host_ptr;
        } else {
            /* T3: pageable - swap-in to temp GPU buffer */
            void *tmp = NULL;
            if (f_cudaMalloc && f_cudaMalloc(&tmp, (size_t)ra->size) == 0 && f_cudaMemcpy) {
                f_cudaMemcpy(tmp, ra->host_ptr, (size_t)ra->size, GB_cudaMemcpyHostToDevice);
                arg_vals[idx] = (uint64_t)(uintptr_t)tmp;
                if (n_t3_tmp < GB_NET_MAX_RELOCS) t3_tmp_dev[n_t3_tmp++] = tmp;
            } else {
                pthread_mutex_unlock(&g_alloc_lock);
                for (int j = 0; j < n_t3_tmp; j++) if (t3_tmp_dev[j] && f_cudaFree) f_cudaFree(t3_tmp_dev[j]);
                EXEC_ERR();
            }
        }
    }
    pthread_mutex_unlock(&g_alloc_lock);

    /* handle uploads: allocate temp device buffers, copy data host→device */
    void *tmp_dev[GB_NET_MAX_XFERS] = {0};
    /* tmp_dev is indexed by upload sequence; this maps arg_idx → dev ptr for downloads */
    void *tmp_dev_by_arg[256] = {0};
    const uint8_t *up_src = upload_data;
    int upload_ok = 1;
    if (f_cudaMalloc && f_cudaMemcpy) {
        for (uint32_t i = 0; i < req_n_uploads && upload_ok; i++) {  /* PR-UU */
            uint32_t ai = up_descs[i].arg_idx;
            /* Audit F-L3-05: bound at the write site (arg_vals capacity == 256). */
            if (ai >= req_n_arg_vals || ai >= 256) { upload_ok = 0; break; }
            if (f_cudaMalloc(&tmp_dev[i], (size_t)up_descs[i].size) != 0) {
                upload_ok = 0; break;
            }
            if (f_cudaMemcpy(tmp_dev[i], up_src, (size_t)up_descs[i].size,
                             GB_cudaMemcpyHostToDevice) != 0) {
                upload_ok = 0; break;
            }
            arg_vals[ai] = (uint64_t)(uintptr_t)tmp_dev[i];
            tmp_dev_by_arg[ai] = tmp_dev[i];
            up_src += up_descs[i].size;
        }
    } else if (req_n_uploads > 0) {  /* PR-UU */
        upload_ok = 0;
    }

    struct gb_net_response resp = {
        .orig_msg_type = GB_MSG_CUDA_EXEC,
        .status        = upload_ok ? GB_STATUS_OK : GB_STATUS_ERR_CUDA,
    };

    if (upload_ok) {
        /* get cuLaunchKernel */
        typedef int (*pfn_cuLK)(void *, unsigned, unsigned, unsigned,
                                unsigned, unsigned, unsigned,
                                unsigned, void *, void **, void **);
        static pfn_cuLK f_cuLK = NULL;
        if (!f_cuLK) {
            void *lc = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
            if (!lc) lc = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
            if (lc) f_cuLK = (pfn_cuLK)dlsym(lc, "cuLaunchKernel");
        }

        /* build kernelParams: array of pointers to each arg_vals[i] */
        void *kp[256];
        for (uint32_t i = 0; i < req_n_arg_vals; i++) kp[i] = &arg_vals[i];  /* PR-UU */

        int err = -1;
        if (is_captured_stub) {
            /* Captured __cudaRegisterFunction stub → RUNTIME cudaLaunchKernel,
             * which keys on the host stub pointer.  The driver cuLaunchKernel
             * wants a CUfunction and would reject this. */
            typedef struct { unsigned x, y, z; } gb_dim3_t;
            typedef int (*pfn_cudaLK)(const void *, gb_dim3_t, gb_dim3_t,
                                      void **, size_t, void *);
            static pfn_cudaLK f_cudaLK = NULL;
            static int probed = 0;
            if (!probed) {
                probed = 1;
                /* libcudart already loaded (probe_gpus); RTLD_DEFAULT finds it. */
                f_cudaLK = (pfn_cudaLK)dlsym(RTLD_DEFAULT, "cudaLaunchKernel");
            }
            if (f_cudaLK) {
                gb_dim3_t g = { req_grid_x, req_grid_y, req_grid_z };
                gb_dim3_t b = { req_block_x, req_block_y, req_block_z };
                err = f_cudaLK(host_func, g, b,
                               req_n_arg_vals ? kp : NULL,
                               (size_t)req_shared_mem_bytes, NULL);
            } else {
                netd_log("exec: cudaLaunchKernel not available for captured stub");
            }
        } else if (f_cuLK) {
            err = f_cuLK(host_func,
                         req_grid_x, req_grid_y, req_grid_z,
                         req_block_x, req_block_y, req_block_z,
                         req_shared_mem_bytes, NULL,
                         req_n_arg_vals ? kp : NULL, NULL);
        } else {
            netd_log("exec: cuLaunchKernel not available");
        }

        if (err != 0) {
            resp.status = GB_STATUS_ERR_CUDA;
            netd_log("exec: launch '%s' failed err=%d (%s)", kernel_name, err,
                     is_captured_stub ? "runtime" : "driver");
            NETD_EVT("KERN_ERR", "T1_GPU", 0, cli->feeder_id, kernel_name);
        } else {
            netd_log("exec: '%s' grid=(%u,%u,%u) block=(%u,%u,%u) args=%u relocs=%u [%s]",
                     kernel_name,
                     req_grid_x, req_grid_y, req_grid_z,
                     req_block_x, req_block_y, req_block_z,
                     req_n_arg_vals, req_n_relocs,
                     is_captured_stub ? "runtime-stub" : "driver");
        }
    }

    /* N7: count successful dispatches */
    if (resp.status == GB_STATUS_OK)
        g_kernel_dispatch_count++;

    /* build response: header + download data */
    if (resp.status == GB_STATUS_OK && req_n_downloads > 0 && f_cudaMemcpy) {  /* PR-UU */
        /* N-H3: same per-descriptor cap for downloads. */
        uint64_t dl_total = 0;
        for (uint32_t i = 0; i < req_n_downloads; i++) {
            if (dn_descs[i].size > GB_NET_MAX_EXEC_UPLOAD_BYTES) {
                resp.status = GB_STATUS_ERR_CUDA; break;
            }
            dl_total += dn_descs[i].size;
        }
        uint8_t *resp_buf = (uint8_t *)malloc(sizeof(resp) + (size_t)dl_total);
        if (resp_buf) {
            memcpy(resp_buf, &resp, sizeof(resp));
            uint8_t *rp = resp_buf + sizeof(resp);
            for (uint32_t i = 0; i < req_n_downloads; i++) {
                uint32_t ai = dn_descs[i].arg_idx;
                void *src_dev = (ai < 256 && tmp_dev_by_arg[ai])
                                ? tmp_dev_by_arg[ai] : NULL;
                if (src_dev && f_cudaMemcpy(rp, src_dev, (size_t)dn_descs[i].size,
                                            GB_cudaMemcpyDeviceToHost) == 0) {
                    rp += dn_descs[i].size;
                } else {
                    /* download failed - zero-fill */
                    memset(rp, 0, (size_t)dn_descs[i].size);
                    rp += dn_descs[i].size;
                }
            }
            int rc = send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                              resp_buf, (uint32_t)(sizeof(resp) + (size_t)dl_total));
            free(resp_buf);
            /* free temp device buffers (uploads + T3 swap-in) */
            for (uint32_t i = 0; i < req_n_uploads; i++)  /* PR-UU */
                if (tmp_dev[i] && f_cudaFree) f_cudaFree(tmp_dev[i]);
            for (int i = 0; i < n_t3_tmp; i++)
                if (t3_tmp_dev[i] && f_cudaFree) f_cudaFree(t3_tmp_dev[i]);
            NETD_EVT("EXEC_KERNEL_OK", "T1_GPU", 0, cli->feeder_id, "cuLaunchKernel_ok");
#undef EXEC_ERR
            return rc;
        }
        resp.status = GB_STATUS_ERR_OOM;
    }

    /* free temp device buffers (uploads + T3 swap-in) */
    for (uint32_t i = 0; i < req_n_uploads; i++)  /* PR-UU */
        if (tmp_dev[i] && f_cudaFree) f_cudaFree(tmp_dev[i]);
    for (int i = 0; i < n_t3_tmp; i++)
        if (t3_tmp_dev[i] && f_cudaFree) f_cudaFree(t3_tmp_dev[i]);
#undef EXEC_ERR
    return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE, &resp, sizeof(resp));
}

/* Phase 2b: GB_MSG_CUDA_EXEC_ASYNC - fire-and-forget kernel dispatch.
 *
 * Same wire format as GB_MSG_CUDA_EXEC but n_downloads must be 0 and
 * n_uploads must be 0.  The feeder enqueues the kernel on its per-client
 * async_stream, ACKs immediately (before GPU execution completes), and
 * the host moves on to dispatch the next kernel without waiting.
 *
 * Sync happens lazily: the next GB_MSG_CUDA_SYNC (from cudaStreamSynchronize)
 * drains async_stream via cudaDeviceSynchronize before responding, so all
 * queued kernels are guaranteed complete before the host reads results.
 *
 * T3 relocs (NVMe swap-in): these require a synchronous H2D memcpy, so we
 * drain async_stream first, execute on it (preserving ordering with any
 * prior async kernels), free the temp buffer, then ACK.  The ACK still
 * arrives before the *next* async kernel queues, maintaining protocol
 * ordering.
 *
 * Error handling: if cuLaunchKernel fails the ACK carries ERR_CUDA; on
 * the host side gb_netc_exec_kernel_async treats this as a failed dispatch. */
static int handle_cuda_exec_async(struct client *cli, const void *payload, uint32_t len)
{
    struct gb_net_response err_resp = {
        .orig_msg_type = GB_MSG_CUDA_EXEC_ASYNC,
        .status        = GB_STATUS_ERR_INVALID,
    };
#define ASYNC_ERR() return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE, &err_resp, sizeof(err_resp))

    if (len < sizeof(struct gb_net_cuda_exec)) ASYNC_ERR();

    const struct gb_net_cuda_exec *req = (const struct gb_net_cuda_exec *)payload;
    const uint8_t *p = (const uint8_t *)payload + sizeof(struct gb_net_cuda_exec);
    uint32_t left    = len - (uint32_t)sizeof(struct gb_net_cuda_exec);

    gb_u32 req_kernel_name_len = GB_LE_U32(req->kernel_name_len);
    gb_u32 req_n_arg_vals      = GB_LE_U32(req->n_arg_vals);
    gb_u32 req_n_relocs        = GB_LE_U32(req->n_relocs);
    gb_u32 req_n_uploads       = GB_LE_U32(req->n_uploads);
    gb_u32 req_n_downloads     = GB_LE_U32(req->n_downloads);
    gb_u32 req_grid_x          = GB_LE_U32(req->grid_x);
    gb_u32 req_grid_y          = GB_LE_U32(req->grid_y);
    gb_u32 req_grid_z          = GB_LE_U32(req->grid_z);
    gb_u32 req_block_x         = GB_LE_U32(req->block_x);
    gb_u32 req_block_y         = GB_LE_U32(req->block_y);
    gb_u32 req_block_z         = GB_LE_U32(req->block_z);
    gb_u32 req_shared_mem_bytes= GB_LE_U32(req->shared_mem_bytes);

    /* Async requires no inline uploads/downloads */
    if (req_n_uploads > 0 || req_n_downloads > 0) ASYNC_ERR();

    if (req_kernel_name_len == 0 || req_kernel_name_len >= GB_NET_MAX_KERNEL_NAME) ASYNC_ERR();
    if (req_n_arg_vals > 256 || req_n_relocs > GB_NET_MAX_RELOCS) ASYNC_ERR();

    {
        uint64_t need = (uint64_t)req_kernel_name_len
                      + (uint64_t)req_n_arg_vals * sizeof(uint64_t)
                      + (uint64_t)req_n_relocs   * sizeof(struct gb_net_ptr_reloc);
        if (need > (uint64_t)left || need > GB_NET_MAX_MSG_SIZE) ASYNC_ERR();
    }

    if (left < req_kernel_name_len) ASYNC_ERR();
    char kernel_name[GB_NET_MAX_KERNEL_NAME];
    memcpy(kernel_name, p, req_kernel_name_len);
    kernel_name[req_kernel_name_len] = '\0';
    p += req_kernel_name_len; left -= req_kernel_name_len;

    uint32_t arg_bytes = req_n_arg_vals * (uint32_t)sizeof(uint64_t);
    if (left < arg_bytes) ASYNC_ERR();
    uint64_t arg_vals[256] = {0};
    if (arg_bytes) memcpy(arg_vals, p, arg_bytes);
    p += arg_bytes; left -= arg_bytes;

    uint32_t reloc_bytes = req_n_relocs * (uint32_t)sizeof(struct gb_net_ptr_reloc);
    if (left < reloc_bytes) ASYNC_ERR();
    struct gb_net_ptr_reloc relocs[GB_NET_MAX_RELOCS];
    if (reloc_bytes) memcpy(relocs, p, reloc_bytes);

    /* Duplicate reloc check */
    for (uint32_t i = 0; i < req_n_relocs; i++)
        for (uint32_t j = i + 1; j < req_n_relocs; j++)
            if (relocs[i].arg_idx == relocs[j].arg_idx) ASYNC_ERR();

    /* Resolve kernel */
    pthread_mutex_lock(&g_kernel_map_lock);
    void *host_func = km_lookup(cli->feeder_id, kernel_name);
    pthread_mutex_unlock(&g_kernel_map_lock);
    if (!host_func) {
        if (!gb_kernel_name_allowed(kernel_name)) ASYNC_ERR();
        host_func = dlsym(RTLD_DEFAULT, kernel_name);
    }
    if (!host_func) ASYNC_ERR();

    /* Apply relocs: T1/T2 are GPU-accessible; T3 needs sync swap-in */
    void *t3_tmp_dev[GB_NET_MAX_RELOCS] = {0};
    int   n_t3_tmp = 0;
    int   has_t3   = 0;

    pthread_mutex_lock(&g_alloc_lock);
    for (uint32_t i = 0; i < req_n_relocs; i++) {
        uint32_t idx = relocs[i].arg_idx;
        if (idx >= req_n_arg_vals || idx >= 256) {
            pthread_mutex_unlock(&g_alloc_lock);
            for (int j = 0; j < n_t3_tmp; j++) if (t3_tmp_dev[j] && f_cudaFree) f_cudaFree(t3_tmp_dev[j]);
            ASYNC_ERR();
        }
        struct remote_alloc *ra = ra_find(relocs[i].remote_handle);
        if (!ra) {
            pthread_mutex_unlock(&g_alloc_lock);
            for (int j = 0; j < n_t3_tmp; j++) if (t3_tmp_dev[j] && f_cudaFree) f_cudaFree(t3_tmp_dev[j]);
            ASYNC_ERR();
        }
        if (ra->tier == 0) {
            arg_vals[idx] = (uint64_t)(uintptr_t)ra->dev_ptr;
        } else if (ra->tier == 1) {
            arg_vals[idx] = (uint64_t)(uintptr_t)ra->host_ptr;
        } else {
            /* T3: sync swap-in required */
            has_t3 = 1;
            void *tmp = NULL;
            if (f_cudaMalloc && f_cudaMalloc(&tmp, (size_t)ra->size) == 0 && f_cudaMemcpy) {
                f_cudaMemcpy(tmp, ra->host_ptr, (size_t)ra->size, GB_cudaMemcpyHostToDevice);
                arg_vals[idx] = (uint64_t)(uintptr_t)tmp;
                if (n_t3_tmp < GB_NET_MAX_RELOCS) t3_tmp_dev[n_t3_tmp++] = tmp;
            } else {
                pthread_mutex_unlock(&g_alloc_lock);
                for (int j = 0; j < n_t3_tmp; j++) if (t3_tmp_dev[j] && f_cudaFree) f_cudaFree(t3_tmp_dev[j]);
                ASYNC_ERR();
            }
        }
    }
    pthread_mutex_unlock(&g_alloc_lock);

    /* Get cuLaunchKernel function pointer */
    typedef int (*pfn_cuLK2)(void *, unsigned, unsigned, unsigned,
                             unsigned, unsigned, unsigned,
                             unsigned, void *, void **, void **);
    static pfn_cuLK2 f_cuLK2 = NULL;
    if (!f_cuLK2) {
        void *lc = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!lc) lc = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
        if (lc) f_cuLK2 = (pfn_cuLK2)dlsym(lc, "cuLaunchKernel");
    }

    if (!f_cuLK2) {
        for (int j = 0; j < n_t3_tmp; j++) if (t3_tmp_dev[j] && f_cudaFree) f_cudaFree(t3_tmp_dev[j]);
        ASYNC_ERR();
    }

    void *kp[256];
    for (uint32_t i = 0; i < req_n_arg_vals; i++) kp[i] = &arg_vals[i];

    /* If T3 swap-in was needed, drain async_stream first (preserves ordering),
     * then launch and synchronize before freeing tmp bufs. */
    if (has_t3 && cli->async_stream && f_cudaStreamSynchronize)
        f_cudaStreamSynchronize(cli->async_stream);

    int launch_err = f_cuLK2(host_func,
                             req_grid_x, req_grid_y, req_grid_z,
                             req_block_x, req_block_y, req_block_z,
                             req_shared_mem_bytes,
                             cli->async_stream,  /* enqueue on per-client async stream */
                             req_n_arg_vals ? kp : NULL, NULL);

    if (has_t3) {
        /* T3 path: sync the stream so tmp bufs are safe to free */
        if (cli->async_stream && f_cudaStreamSynchronize)
            f_cudaStreamSynchronize(cli->async_stream);
        for (int j = 0; j < n_t3_tmp; j++) if (t3_tmp_dev[j] && f_cudaFree) f_cudaFree(t3_tmp_dev[j]);
    }
    /* T1/T2 path: kernel is queued; GPU executes while host moves on */

    if (launch_err == 0) g_kernel_dispatch_count++;
    NETD_EVT("EXEC_KERNEL_ASYNC", "T1_GPU", 0, cli->feeder_id, kernel_name);

#undef ASYNC_ERR
    struct gb_net_response ack = {
        .orig_msg_type = GB_MSG_CUDA_EXEC_ASYNC,
        .status        = (launch_err == 0) ? GB_STATUS_OK : GB_STATUS_ERR_CUDA,
    };
    return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE, &ack, sizeof(ack));
}

static int handle_cuda_sync(struct client *cli, const void *payload, uint32_t len)
{
    (void)payload; (void)len;

    /* Phase 2b: drain per-client async stream first, then full device sync.
     * cudaDeviceSynchronize covers all streams, but explicit stream sync here
     * lets us later surface stream-level errors separately if needed. */
    if (cli->async_stream && f_cudaStreamSynchronize)
        f_cudaStreamSynchronize(cli->async_stream);
    if (f_cudaDeviceSynchronize)
        f_cudaDeviceSynchronize();

    struct gb_net_response resp = {
        .orig_msg_type = GB_MSG_CUDA_SYNC,
        .status        = GB_STATUS_OK,
    };
    return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                    &resp, sizeof(resp));
}

/* ── Trusted kernel library (2026-07-06) ─────────────────────────────────
 * Remote dispatch can only run kernels whose host stubs exist in THIS
 * process.  The daemon itself links no compute library, so it dlopens the
 * feeder's own ggml CUDA backend (root-installed by Ollama) and resolves
 * kernel names scoped to that handle , dlsym(g_kernel_lib, name) searches
 * the library and its dependency chain only, never the daemon's own symbols.
 * This is both what makes ggml dispatch WORK (RTLD_DEFAULT never resolved
 * these) and a tighter sandbox than a name allowlist: only kernels shipped
 * in the trusted library are launchable.  Loading it also runs the
 * library's __cudaRegisterFatBinary constructors against its own cudart, so
 * the returned stubs are valid launch handles for that runtime instance.
 * Override the path with GB_NETD_KERNEL_LIB. */
static void *g_kernel_lib = NULL;
static int   g_kernel_lib_tried = 0;
/* ── Fatbin kernel resolver (Option A, 2026-07-06) ──────────────────────
 * General fallback for kernels that are neither dlsym-resolvable nor
 * __cudaRegisterFunction-captured (e.g. ggml's fused mul_mat_vec_q family;
 * torch kernels for image/video/mesh pipelines).  The kernel library's
 * .nv_fatbin ELF section is a concatenation of fatbin containers;
 * cuModuleLoadData accepts a whole container and picks the right arch.
 * Strategy: mmap the .so once, index the containers, then on a name miss
 * lazily load containers one at a time and probe cuModuleGetFunction until
 * the name resolves.  Resolved CUfunctions launch via the driver
 * cuLaunchKernel (they are real CUfunctions, unlike host stubs). */
#include <elf.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#define GB_FATBIN_MAGIC 0xBA55ED50u
#define GB_FATBIN_MAX_CONTAINERS 4096
#define GB_FATBIN_CACHE 1024

struct gb_fatbin_hdr {
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint64_t fat_size;
};

static const uint8_t *g_fb_containers[GB_FATBIN_MAX_CONTAINERS];
static void          *g_fb_modules[GB_FATBIN_MAX_CONTAINERS];   /* CUmodule, lazily loaded */
static int8_t         g_fb_tried[GB_FATBIN_MAX_CONTAINERS];
static int            g_fb_n = 0;
static int            g_fb_init_done = 0;

static struct { char name[GB_NET_MAX_KERNEL_NAME]; void *fn; } g_fb_cache[GB_FATBIN_CACHE];
static int g_fb_cache_n = 0;

static int (*fb_cuModuleLoadData)(void **, const void *);
static int (*fb_cuModuleGetFunction)(void **, void *, const char *);

static void gb_fatbin_init(const char *lib_path)
{
    if (g_fb_init_done) return;
    g_fb_init_done = 1;

    void *lc = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!lc) lc = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (!lc) return;
    fb_cuModuleLoadData    = (int (*)(void **, const void *))dlsym(lc, "cuModuleLoadData");
    fb_cuModuleGetFunction = (int (*)(void **, void *, const char *))dlsym(lc, "cuModuleGetFunction");
    if (!fb_cuModuleLoadData || !fb_cuModuleGetFunction) return;

    int fd = open(lib_path, O_RDONLY);
    if (fd < 0) { netd_log("fatbin: cannot open %s", lib_path); return; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return; }
    const uint8_t *base = (const uint8_t *)mmap(NULL, (size_t)st.st_size,
                                                PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);   /* mapping stays valid */
    if (base == MAP_FAILED) { netd_log("fatbin: mmap failed"); return; }

    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)base;
    if (memcmp(eh->e_ident, ELFMAG, SELFMAG) != 0) return;
    const Elf64_Shdr *sh = (const Elf64_Shdr *)(base + eh->e_shoff);
    const char *shstr = (const char *)(base + sh[eh->e_shstrndx].sh_offset);

    for (int i = 0; i < eh->e_shnum; i++) {
        if (strcmp(shstr + sh[i].sh_name, ".nv_fatbin") != 0) continue;
        const uint8_t *p   = base + sh[i].sh_offset;
        const uint8_t *end = p + sh[i].sh_size;
        while (p + sizeof(struct gb_fatbin_hdr) <= end && g_fb_n < GB_FATBIN_MAX_CONTAINERS) {
            const struct gb_fatbin_hdr *h = (const struct gb_fatbin_hdr *)p;
            if (h->magic != GB_FATBIN_MAGIC) { p += 8; continue; }  /* skip padding */
            if (p + h->header_size + h->fat_size > end) break;
            g_fb_containers[g_fb_n++] = p;
            p += h->header_size + h->fat_size;
        }
        break;
    }
    netd_log("fatbin: indexed %d containers from %s", g_fb_n, lib_path);
}

/* Resolve a kernel name to a CUfunction by lazily loading fatbin containers.
 * Returns NULL if not found anywhere. */
static void *gb_fatbin_resolve(const char *name)
{
    if (!g_fb_n || !fb_cuModuleGetFunction) return NULL;

    for (int i = 0; i < g_fb_cache_n; i++)
        if (strcmp(g_fb_cache[i].name, name) == 0) return g_fb_cache[i].fn;

    for (int i = 0; i < g_fb_n; i++) {
        if (!g_fb_modules[i] && !g_fb_tried[i]) {
            g_fb_tried[i] = 1;
            if (fb_cuModuleLoadData(&g_fb_modules[i], g_fb_containers[i]) != 0)
                g_fb_modules[i] = NULL;   /* e.g. relocatable-only container */
        }
        if (!g_fb_modules[i]) continue;
        void *fn = NULL;
        if (fb_cuModuleGetFunction(&fn, g_fb_modules[i], name) == 0 && fn) {
            if (g_fb_cache_n < GB_FATBIN_CACHE) {
                strncpy(g_fb_cache[g_fb_cache_n].name, name, GB_NET_MAX_KERNEL_NAME - 1);
                g_fb_cache[g_fb_cache_n].fn = fn;
                g_fb_cache_n++;
            }
            netd_log("fatbin: resolved '%s' (container %d)", name, i);
            return fn;
        }
    }
    return NULL;
}

static void *gb_kernel_lib(void)
{
    if (!g_kernel_lib_tried) {
        g_kernel_lib_tried = 1;
        const char *cands[] = {
            getenv("GB_NETD_KERNEL_LIB"),
            /* gb-synapse's own engine build (2026-07-15 — replaces the Ollama
             * bundled build; same libggml-cuda.so kernel container, built from
             * the same llama.cpp source the host dispatches kernel names from). */
            "/usr/local/lib/greenboost/synapse/libggml-cuda.so",
            "/usr/local/lib/ollama/cuda_v13/libggml-cuda.so",
            "/usr/local/lib/ollama/libggml-cuda.so",
            NULL
        };
        for (int i = 0; i < 4; i++) {
            if (!cands[i]) continue;
            /* libggml-cuda.so has no RUNPATH , Ollama supplies
             * LD_LIBRARY_PATH to its children.  Pre-load libggml-base.so.0
             * from the candidate's own dir and its parent so RTLD_NOW
             * resolution of the main lib succeeds. */
            char dep[512];
            const char *slash = strrchr(cands[i], '/');
            if (slash) {
                int dirlen = (int)(slash - cands[i]);
                snprintf(dep, sizeof(dep), "%.*s/libggml-base.so.0", dirlen, cands[i]);
                if (!dlopen(dep, RTLD_NOW | RTLD_GLOBAL)) {
                    const char *slash2 = memrchr(cands[i], '/', (size_t)dirlen);
                    if (slash2) {
                        snprintf(dep, sizeof(dep), "%.*s/libggml-base.so.0",
                                 (int)(slash2 - cands[i]), cands[i]);
                        dlopen(dep, RTLD_NOW | RTLD_GLOBAL);
                    }
                }
            }
            g_kernel_lib = dlopen(cands[i], RTLD_NOW | RTLD_GLOBAL);
            if (g_kernel_lib) {
                int (*capn)(void) = (int (*)(void))dlsym(RTLD_DEFAULT, "gb_capture_count");
                netd_log("kernel lib: loaded %s (scoped dispatch enabled, %d stubs captured)",
                         cands[i], capn ? capn() : -1);
                gb_fatbin_init(cands[i]);   /* index .nv_fatbin for the general fallback */
                break;
            }
            netd_log("kernel lib: %s failed: %s", cands[i], dlerror());
        }
        if (!g_kernel_lib)
            netd_log("WARN: no kernel lib found (set GB_NETD_KERNEL_LIB) - "
                     "remote dispatch limited to kernels.allow + RTLD_DEFAULT");
    }
    return g_kernel_lib;
}

/* Resolve a kernel symbol: prefer the trusted library handle; fall back to
 * RTLD_DEFAULT only for allowlisted names (legacy behaviour). */
static void *gb_kernel_resolve(const char *kname)
{
    void *lib = gb_kernel_lib();
    if (lib) {
        void *f = dlsym(lib, kname);
        if (f) return f;
    }
    return dlsym(RTLD_DEFAULT, kname);
}

/* Audit F-L3-04: optional allowlist of kernel symbol names the daemon will
 * resolve.  Policy (2026-07-06):
 *   - kernels.allow present → only listed names pass (strictest).
 *   - file absent but the trusted kernel lib loaded → allowed; resolution is
 *     scoped to that library, which is the real containment boundary.  A
 *     256-entry name list can never cover ggml's kernel set (hundreds of
 *     template instantiations) , requiring it made feeder compute
 *     permanently dead.
 *   - neither → reject all (fail-closed, unchanged).
 *
 * TCB-U1 (2026-08-06): now validates kernels.allow ownership/perms before
 * trusting its contents; reloads if file mtime changes. */
#define GB_KERNEL_ALLOW_PATH "/etc/greenboost/kernels.allow"
#define GB_KERNEL_ALLOW_MAX_BYTES (64 * 1024)
static int gb_kernel_name_allowed(const char *kname)
{
    static int   g_allow_loaded = 0;
    static int   g_allow_present = 0;
    static char  g_allow_names[256][GB_NET_MAX_KERNEL_NAME];
    static int   g_allow_count = 0;
    static time_t g_allow_mtime = 0;

    /* Check if file has changed (mtime staleness check). */
    struct stat st;
    if (g_allow_loaded && stat(GB_KERNEL_ALLOW_PATH, &st) == 0) {
        if (st.st_mtime == g_allow_mtime) {
            /* File unchanged; use cached result. */
            if (!g_allow_present)
                return gb_kernel_lib() != NULL;
            for (int i = 0; i < g_allow_count; i++) {
                if (strcmp(g_allow_names[i], kname) == 0) return 1;
            }
            return 0;
        }
        /* File modified; reload. */
        g_allow_loaded = 0;
    }

    if (!g_allow_loaded) {
        g_allow_loaded = 1;
        g_allow_present = 0;
        g_allow_count = 0;
        memset(g_allow_names, 0, sizeof(g_allow_names));

        int fd = gb_trusted_root_file_fd(GB_KERNEL_ALLOW_PATH, GB_KERNEL_ALLOW_MAX_BYTES);
        if (fd < 0) {
            /* gb_trusted_root_file_fd returns -1 silently; diagnose reason here */
            struct stat st;
            if (lstat(GB_KERNEL_ALLOW_PATH, &st) != 0) {
                netd_log("TRACE: %s not present - kernel dispatch scoped to trusted library only "
                         "(audit F-L3-04)", GB_KERNEL_ALLOW_PATH);
            } else if (S_ISLNK(st.st_mode)) {
                netd_log("SECURITY: %s rejected: is a symbolic link - all remote kernel dispatches "
                         "denied (audit TCB-U1)", GB_KERNEL_ALLOW_PATH);
            } else if (st.st_uid != 0) {
                netd_log("SECURITY: %s rejected: owner uid=%d (expected 0) - all remote kernel "
                         "dispatches denied (audit TCB-U1)", GB_KERNEL_ALLOW_PATH, st.st_uid);
            } else if ((st.st_mode & 0022) != 0) {
                netd_log("SECURITY: %s rejected: world/group-writable mode=0%o - all remote kernel "
                         "dispatches denied (audit TCB-U1)", GB_KERNEL_ALLOW_PATH, st.st_mode & 07777);
            } else {
                netd_log("SECURITY: %s validation failed - all remote kernel dispatches denied "
                         "(audit TCB-U1)", GB_KERNEL_ALLOW_PATH);
            }
            return gb_kernel_lib() != NULL;
        }

        if (fstat(fd, &st) == 0) {
            g_allow_mtime = st.st_mtime;
        }

        FILE *fp = fdopen(fd, "r");
        if (fp) {
            g_allow_present = 1;
            char line[GB_NET_MAX_KERNEL_NAME + 32];
            while (g_allow_count < 256 && fgets(line, sizeof(line), fp)) {
                char *s = line;
                while (*s == ' ' || *s == '\t') s++;
                if (*s == '#' || *s == '\n' || *s == '\0') continue;
                size_t L = strlen(s);
                while (L > 0 && (s[L-1] == '\n' || s[L-1] == '\r' ||
                                 s[L-1] == ' '  || s[L-1] == '\t')) s[--L] = '\0';
                if (L == 0 || L >= GB_NET_MAX_KERNEL_NAME) continue;
                memcpy(g_allow_names[g_allow_count], s, L + 1);
                g_allow_count++;
            }
            fclose(fp);
            netd_log("kernel allowlist: %d entries loaded from %s (trusted)",
                     g_allow_count, GB_KERNEL_ALLOW_PATH);
        } else {
            close(fd);
            netd_log("WARN: failed to fdopen validated %s", GB_KERNEL_ALLOW_PATH);
        }
    }
    if (!g_allow_present)
        return gb_kernel_lib() != NULL; /* lib-scoped resolution; else fail-closed */
    for (int i = 0; i < g_allow_count; i++) {
        if (strcmp(g_allow_names[i], kname) == 0) return 1;
    }
    return 0;
}

static int handle_cuda_register_fn(struct client *cli, const void *payload, uint32_t len)
{
    if (len < sizeof(struct gb_net_cuda_register_fn))
        return -1;

    const struct gb_net_cuda_register_fn *req =
        (const struct gb_net_cuda_register_fn *)payload;
    const char *name = (const char *)payload + sizeof(struct gb_net_cuda_register_fn);

    /* PR-UU: BE-safe field read. */
    gb_u32 req_kernel_name_len = GB_LE_U32(req->kernel_name_len);

    if (sizeof(struct gb_net_cuda_register_fn) + req_kernel_name_len > len)
        return -1;

    /* Audit F-L3-20: reject (not truncate) oversized kernel names.  Silent
     * truncation lets two distinct kernels alias to the same first-255 bytes. */
    if (req_kernel_name_len == 0 || req_kernel_name_len >= GB_NET_MAX_KERNEL_NAME) {
        netd_log("WARN register_fn: invalid kernel_name_len=%u from %s",
                 req_kernel_name_len, cli->remote_addr);
        return 0; /* client is fire-and-forget: never reads a response */
    }
    char kname[GB_NET_MAX_KERNEL_NAME];
    uint32_t nlen = req_kernel_name_len;
    memcpy(kname, name, nlen);
    kname[nlen] = '\0';

    /* Audit F-L3-04: gate `dlsym(RTLD_DEFAULT, …)` behind an allowlist file. */
    if (!gb_kernel_name_allowed(kname)) {
        netd_log("WARN register_fn: kernel '%s' not on allowlist - rejected", kname);
        return 0; /* client is fire-and-forget: never reads a response */
    }

    /* Try to find the kernel function via dlsym.
     * CUDA device functions registered with __cudaRegisterFunction are
     * available as host-side stubs with the same symbol name , resolved
     * scoped to the trusted kernel library (see gb_kernel_resolve). */
    void *func = gb_kernel_resolve(kname);

    if (func) {
        /* PR-L: client-scoped registration.  Take the lock so concurrent
         * register calls from multiple clients don't race on slot picking.
         * Two-pass: first try to update an existing entry for THIS client
         * with the same name (re-register), then claim a free slot. */
        pthread_mutex_lock(&g_kernel_map_lock);
        int updated = 0;
        for (int i = 0; i < MAX_KERNEL_NAMES; i++) {
            if (g_kernel_map[i].in_use &&
                g_kernel_map[i].client_id == cli->feeder_id &&
                strcmp(g_kernel_map[i].name, kname) == 0) {
                g_kernel_map[i].host_func = func;
                updated = 1;
                break;
            }
        }
        if (!updated) {
            for (int i = 0; i < MAX_KERNEL_NAMES; i++) {
                if (!g_kernel_map[i].in_use) {
                    strncpy(g_kernel_map[i].name, kname, sizeof(g_kernel_map[i].name) - 1);
                    g_kernel_map[i].host_func = func;
                    g_kernel_map[i].client_id = cli->feeder_id;
                    g_kernel_map[i].in_use = 1;
                    /* PR-V: index for O(1) launch-time lookup. */
                    km_hash_insert(cli->feeder_id, g_kernel_map[i].name, (uint32_t)i);
                    netd_log("registered kernel '%s' → %p (client %u)", kname, func, cli->feeder_id);
                    break;
                }
            }
        }
        pthread_mutex_unlock(&g_kernel_map_lock);
    } else {
        netd_log("WARN: kernel '%s' not found via dlsym", kname);
    }

    return 0; /* client is fire-and-forget: no response sent */
}

/* ------------------------------------------------------------------ */
/* N7: handle GB_MSG_FEEDER_STATUS - return live stats to host */
static int handle_feeder_status(struct client *cli)
{
    struct gb_feeder_status_resp resp;
    memset(&resp, 0, sizeof(resp));
    resp.status               = GB_STATUS_OK;
    resp.kernel_dispatch_count = g_kernel_dispatch_count;

    /* Read current MPS SM% from /dev/shm written by GB_MSG_CUDA_MPS_SET handler */
    {
        char path[64];
        snprintf(path, sizeof(path), "/dev/shm/gb_mps_pct_%u", (unsigned)getpid());
        FILE *fp = fopen(path, "r");
        /* PR-L: check fscanf return - on a corrupt/empty file the previous
         * code left resp.mps_sm_pct uninitialised (well, zero from memset,
         * but the warning rightly flagged the unchecked read). */
        if (fp) {
            if (fscanf(fp, "%u", &resp.mps_sm_pct) != 1)
                resp.mps_sm_pct = 0;
            fclose(fp);
        }
    }

    /* T1: GPU VRAM from NVML */
    if (f_nvmlMemInfo && g_nvml_devices[0]) {
        nvmlMemory_t mem;
        if (f_nvmlMemInfo(g_nvml_devices[0], &mem) == 0) {
            resp.t1_free_bytes  = (gb_u64)mem.free;
            resp.t1_total_bytes = (gb_u64)mem.total;
        }
    }

    /* T2: System RAM from /proc/meminfo */
    {
        FILE *mf = fopen("/proc/meminfo", "r");
        if (mf) {
            char line[128];
            uint64_t mem_total = 0, mem_avail = 0;
            while (fgets(line, sizeof(line), mf)) {
                unsigned long long val = 0;
                if (sscanf(line, "MemTotal: %llu kB", &val) == 1)      mem_total = (uint64_t)val * 1024;
                else if (sscanf(line, "MemAvailable: %llu kB", &val) == 1) mem_avail = (uint64_t)val * 1024;
            }
            fclose(mf);
            resp.t2_total_bytes = mem_total;
            resp.t2_free_bytes  = mem_avail;
        }
    }

    /* T3: NVMe pool size from sysfs */
    {
        FILE *sf = fopen("/sys/module/greenboost/parameters/nvme_pool_gb", "r");
        if (sf) {
            int nvme_gb = 0;
            if (fscanf(sf, "%d", &nvme_gb) == 1 && nvme_gb > 0)
                resp.t3_total_bytes = (gb_u64)nvme_gb * 1024ULL * 1024ULL * 1024ULL;
            fclose(sf);
        }
        /* T3 free is difficult to determine without tracking; use total as proxy when not tracked */
        resp.t3_free_bytes = resp.t3_total_bytes;
    }

    /* v3.1 GPU telemetry fields , populated from NVML function pointers that
     * are already resolved in probe_gpus().  Zero-valued if NVML unavailable. */
    if (g_nvml_devices[0]) {
        if (f_nvmlGetTemp) {
            unsigned int t = 0;
            if (f_nvmlGetTemp(g_nvml_devices[0], 1 /*NVML_TEMPERATURE_GPU*/, &t) == 0)
                resp.gpu_temp_c = (gb_u16)t;
        }
        if (f_nvmlGetPower) {
            unsigned int mw = 0;
            if (f_nvmlGetPower(g_nvml_devices[0], &mw) == 0)
                resp.gpu_power_w = (gb_u16)(mw / 1000);
        }
        if (f_nvmlUtilRates) {
            nvmlUtilization_t ut = {0, 0};
            if (f_nvmlUtilRates(g_nvml_devices[0], &ut) == 0)
                resp.gpu_util_pct = (gb_u32)ut.gpu;
        }
        if (f_nvmlGetThrottle) {
            unsigned long long reasons = 0;
            if (f_nvmlGetThrottle(g_nvml_devices[0], &reasons) == 0)
                resp.throttle_reasons = (gb_u32)(reasons & 0xFFFFFFFF);
        }
        if (f_nvmlGetEcc) {
            unsigned long long dbe = 0;
            /* NVML_MEMORY_ERROR_TYPE_UNCORRECTED=1, NVML_VOLATILE_ECC=0 */
            if (f_nvmlGetEcc(g_nvml_devices[0], 1, 0, &dbe) == 0)
                resp.ecc_dbe_count = (gb_u32)(dbe & 0xFFFFFFFF);
        }
    }

    NETD_EVT("FEEDER_STATUS_SERVED", "NET", 0, cli->feeder_id, "status_query_ok");
    return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE, &resp, sizeof(resp));
}

/* T1: ship this node's hardware profile (/etc/greenboost/profiles/default.md)
 * so the host can build its cluster topology registry. No JSON generation in C
 * , the profile the setup script already wrote is the payload. */
static int handle_topology(struct client *cli)
{
    struct gb_net_topology_resp rhdr;
    memset(&rhdr, 0, sizeof(rhdr));

    char  *data = NULL;
    size_t plen = 0;
    FILE  *fp = fopen("/etc/greenboost/profiles/default.md", "rb");
    if (fp) {
        data = malloc(GB_NET_TOPOLOGY_MAX_BYTES);
        if (data)
            plen = fread(data, 1, GB_NET_TOPOLOGY_MAX_BYTES, fp);
        fclose(fp);
    }

    if (!data || plen == 0) {
        free(data);
        rhdr.status      = GB_STATUS_ERR_INVALID;
        rhdr.profile_len = 0;
        NETD_EVT("TOPOLOGY_UNAVAILABLE", "NET", 0, cli->feeder_id, "profile_unreadable");
        return send_msg(cli, GB_MSG_TOPOLOGY, GB_NET_FLAG_RESPONSE, &rhdr, sizeof(rhdr));
    }

    rhdr.status      = GB_STATUS_OK;
    rhdr.profile_len = (gb_u32)plen;

    /* Single framed payload: fixed struct followed by the profile bytes. */
    size_t total = sizeof(rhdr) + plen;
    char  *msg = malloc(total);
    if (!msg) {
        free(data);
        rhdr.status      = GB_STATUS_ERR_OOM;
        rhdr.profile_len = 0;
        return send_msg(cli, GB_MSG_TOPOLOGY, GB_NET_FLAG_RESPONSE, &rhdr, sizeof(rhdr));
    }
    memcpy(msg, &rhdr, sizeof(rhdr));
    memcpy(msg + sizeof(rhdr), data, plen);
    free(data);

    NETD_EVT("TOPOLOGY_SERVED", "NET", (unsigned)(plen >> 10), cli->feeder_id, "profile_ok");
    int rc = send_msg(cli, GB_MSG_TOPOLOGY, GB_NET_FLAG_RESPONSE, msg, (uint32_t)total);
    free(msg);
    return rc;
}

/*  Message dispatch                                                   */
/* ------------------------------------------------------------------ */

static int dispatch_message(struct client *cli, const struct gb_net_header *hdr,
                            const void *payload)
{
    switch (hdr->msg_type) {
    case GB_MSG_HANDSHAKE_REQ:
        return handle_handshake(cli, payload, hdr->payload_len);
    case GB_MSG_HEARTBEAT:
        return handle_heartbeat(cli, payload, hdr->payload_len);
    case GB_MSG_GPU_QUERY:
        return handle_gpu_query(cli, payload, hdr->payload_len);
    case GB_MSG_MEM_INFO:
        return handle_mem_info(cli, payload, hdr->payload_len);
    case GB_MSG_DISCONNECT:
        return handle_disconnect(cli);
    /* Phase 2: remote memory operations */
    case GB_MSG_CUDA_MALLOC:
        return handle_cuda_malloc(cli, payload, hdr->payload_len);
    case GB_MSG_CUDA_FREE:
        return handle_cuda_free(cli, payload, hdr->payload_len);
    case GB_MSG_CUDA_MEMCPY_H2D:
        return handle_cuda_memcpy_h2d(cli, payload, hdr->payload_len,
                                      le16toh(hdr->flags));
    case GB_MSG_CUDA_MEMCPY_D2H:
        return handle_cuda_memcpy_d2h(cli, payload, hdr->payload_len);
    case GB_MSG_CUDA_MEMSET:
        return handle_cuda_memset(cli, payload, hdr->payload_len);
    case GB_MSG_CUDA_MEMCPY_D2D:
        return handle_cuda_memcpy_d2d(cli, payload, hdr->payload_len);
    /* Phase 3: remote compute */
    case GB_MSG_CUDA_LAUNCH:
        return handle_cuda_launch(cli, payload, hdr->payload_len);
    case GB_MSG_CUDA_EXEC:
        if (le16toh(hdr->flags) & GB_NET_FLAG_EXEC_RAW)
            return handle_cuda_exec_raw(cli, payload, hdr->payload_len);
        return handle_cuda_exec(cli, payload, hdr->payload_len);
    case GB_MSG_CUDA_EXEC_ASYNC:
        return handle_cuda_exec_async(cli, payload, hdr->payload_len);
    case GB_MSG_CUDA_SYNC:
        return handle_cuda_sync(cli, payload, hdr->payload_len);
    case GB_MSG_CUDA_REGISTER_FN:
        return handle_cuda_register_fn(cli, payload, hdr->payload_len);
    /* Phase 4: NCCL */
    case GB_MSG_NCCL_INIT:
        /* Feeder receives nccl_id from host, joins communicator as rank 1 */
        netd_log("NCCL init received (len=%u)", hdr->payload_len);
        {
            struct gb_net_response resp = { .orig_msg_type = hdr->msg_type,
                /* default: ERR_NCCL so non-NCCL builds never silently claim support */
                .status = GB_STATUS_ERR_NCCL };
#ifdef GREENBOOST_USE_NCCL
            if (hdr->payload_len == sizeof(struct gb_net_nccl_init)) {
                const struct gb_net_nccl_init *req = payload;
                ncclUniqueId id;
                memcpy(&id, req->nccl_id, sizeof(id));
                /* PR-UU: BE-safe NCCL field reads. */
                gb_u32 req_num_ranks = GB_LE_U32(req->num_ranks);
                gb_u32 req_rank      = GB_LE_U32(req->rank);
                /* ncclCommInitRank is a rendezvous barrier - it blocks until
                 * ALL ranks call it.  Send OK before calling so the host can
                 * call its own ncclCommInitRank(rank=0) concurrently; otherwise
                 * feeder and host deadlock waiting on each other. */
                resp.status = GB_STATUS_OK;
                if (send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE,
                             &resp, sizeof(resp)) < 0)
                    return -1;
                ncclResult_t nr = ncclCommInitRank(&cli->nccl_comm, (int)req_num_ranks, id, (int)req_rank);
                if (nr != ncclSuccess) {
                    netd_log("ERR: ncclCommInitRank failed: %d - nccl_comm left NULL", nr);
                    /* nccl_comm stays NULL; D2D ops will return ERR_NCCL */
                } else {
                    netd_log("NCCL communicator established (rank %d/%d)", (int)req_rank, (int)req_num_ranks);
                }
                return 0;  /* response already sent above */
            } else {
                resp.status = GB_STATUS_ERR_INVALID;
            }
#endif
            return send_msg(cli, GB_MSG_RESPONSE, GB_NET_FLAG_RESPONSE, &resp, sizeof(resp));
        }
    /* U10: KV block events - fire-and-forget, no response needed.
     * Feeder builds block ancestry tree for prefetch planning. */
    case GB_MSG_BLOCK_EVENTS:
        if (hdr->payload_len >= sizeof(uint32_t)) {
            const struct gb_net_block_events *ev =
                (const struct gb_net_block_events *)payload;
            uint32_t cnt = (ev->count < GB_BLOCK_EVENTS_MAX) ? ev->count : GB_BLOCK_EVENTS_MAX;
            uint32_t now_s = (uint32_t)(mono_ms() / 1000);

            pthread_mutex_lock(&g_ancestry_lock);
            for (uint32_t _i = 0; _i < cnt; _i++) {
                const struct gb_net_block_event *be = &ev->events[_i];
                netd_log("block_evt type=%u hash=%016llx parent=%016llx tokens=%u",
                         be->event_type,
                         (unsigned long long)be->block_hash,
                         (unsigned long long)be->parent_hash,
                         be->num_tokens);

                if (be->event_type == GB_BLOCK_EVT_STORED) {
                    /* Insert into ancestry table (open-addressed, linear probe) */
                    uint32_t slot = (uint32_t)(be->block_hash % GB_ANCESTRY_MAX);
                    for (uint32_t _j = 0; _j < GB_ANCESTRY_MAX; _j++) {
                        gb_ancestry_entry_t *ae = &g_ancestry[(slot + _j) % GB_ANCESTRY_MAX];
                        if (ae->hash == 0 || ae->hash == be->block_hash) {
                            ae->hash        = be->block_hash;
                            ae->parent_hash = be->parent_hash;
                            ae->num_tokens  = be->num_tokens;
                            ae->evict_ts    = 0; /* present */
                            break;
                        }
                    }
                } else if (be->event_type == GB_BLOCK_EVT_REMOVED) {
                    /* Mark as evicted in ancestry table for ghost tracking */
                    uint32_t slot = (uint32_t)(be->block_hash % GB_ANCESTRY_MAX);
                    for (uint32_t _j = 0; _j < GB_ANCESTRY_MAX; _j++) {
                        gb_ancestry_entry_t *ae = &g_ancestry[(slot + _j) % GB_ANCESTRY_MAX];
                        if (ae->hash == 0) break;
                        if (ae->hash == be->block_hash) {
                            ae->evict_ts = now_s;
                            break;
                        }
                    }
                }
            }
            pthread_mutex_unlock(&g_ancestry_lock);
        }
        return 0;  /* no response */

    case GB_MSG_FEEDER_STATUS:
        return handle_feeder_status(cli);

    case GB_MSG_TOPOLOGY:
        return handle_topology(cli);

    case GB_MSG_CUDA_MPS_SET: {
        /* U19: Dynamic MPS SM% control - adjust CUDA_MPS_ACTIVE_THREAD_PERCENTAGE.
         * No response. Only effective when GREENBOOST_MPS=1 and MPS daemon is running. */
        static int mps_enabled = -1;
        if (mps_enabled < 0) {
            const char *e = getenv("GREENBOOST_MPS");
            mps_enabled = (e && e[0] == '1') ? 1 : 0;
        }
        if (mps_enabled && hdr->payload_len >= sizeof(struct gb_net_mps_set)) {
            const struct gb_net_mps_set *mps = (const struct gb_net_mps_set *)payload;
            uint32_t pct = (mps->sm_pct == 0 || mps->sm_pct > 100) ? 100 : mps->sm_pct;
            /* Write to /dev/shm/gb_mps_pct so external MPS management scripts can read it */
            char path[64];
            snprintf(path, sizeof(path), "/dev/shm/gb_mps_pct_%u", (unsigned)getpid());
            FILE *fp = fopen(path, "w");
            if (fp) { fprintf(fp, "%u\n", pct); fclose(fp); }
            /* If MPS control socket exists, pipe echo command to adjust SM%.
             * PR-L: use absolute path /usr/bin/nvidia-cuda-mps-control to
             * remove PATH-based command-injection surface - this daemon runs
             * as root and a malicious unprivileged user planting a binary
             * earlier in $PATH could escalate.  Falls back to the
             * /usr/local/bin location used by some NVIDIA driver packages. */
            const char *mps_paths[] = {
                "/usr/bin/nvidia-cuda-mps-control",
                "/usr/local/bin/nvidia-cuda-mps-control",
                NULL
            };
            FILE *mps_ctl = NULL;
            for (int mp = 0; mps_paths[mp]; mp++) {
                if (access(mps_paths[mp], X_OK) == 0) {
                    char cmd[160];
                    snprintf(cmd, sizeof(cmd), "%s 2>/dev/null", mps_paths[mp]);
                    mps_ctl = popen(cmd, "w");
                    break;
                }
            }
            if (mps_ctl) {
                fprintf(mps_ctl, "set_active_thread_percentage %u\n", pct);
                pclose(mps_ctl);
                netd_log("U19: MPS SM%%=%u applied via nvidia-cuda-mps-control", pct);
            } else {
                netd_log("U19: MPS SM%%=%u written to %s (MPS daemon not running)", pct, path);
            }
        }
        return 0;  /* no response */
    }

    default:
        netd_log("WARN: unknown msg_type 0x%02x from %s", hdr->msg_type, cli->remote_addr);
        return 0;
    }
}

/* ------------------------------------------------------------------ */
/*  Client recv + framing                                              */
/* ------------------------------------------------------------------ */

static void client_init(struct client *cli, int fd, const char *addr, uint64_t now_ms)
{
    memset(cli, 0, sizeof(*cli));
    cli->fd       = fd;
    cli->active   = 1;
    cli->recv_buf = (uint8_t *)malloc(RECV_BUF_SIZE);
    cli->recv_len = 0;
    cli->recv_cap = RECV_BUF_SIZE;
    /* Bug fix: must use the epoll loop's already-captured `now_ms`, not a
     * fresh mono_ms() call. The accept-handling block does blocking PSK
     * auth I/O (server sends a nonce, waits up to 2s for the client's MAC)
     * before reaching here, so a fresh mono_ms() here is always >= the
     * loop's now_ms by however long that auth round-trip took. The
     * heartbeat-timeout sweep at the bottom of this same iteration then
     * computes `now_ms - last_heartbeat_ms` as unsigned - last_heartbeat_ms
     * being greater than now_ms underflows to near UINT64_MAX, instantly
     * exceeding GB_NET_HEARTBEAT_TIMEOUT_MS and disconnecting every fresh
     * connection before it can complete its handshake. Confirmed via
     * /var/log/greenboost/netd.log: every real connection attempt showed
     * "AUTH: PSK verified" + "Connection from ..." + "heartbeat timeout -
     * disconnecting" all logged within the same second. */
    cli->last_heartbeat_ms = now_ms;
    cli->alloc_tokens  = 200;        /* N9: start with full bucket */
    cli->last_refill_ms = now_ms;
    pthread_mutex_init(&cli->send_lock, NULL);  /* PR-QQ */
    strncpy(cli->remote_addr, addr, sizeof(cli->remote_addr) - 1);
}

static void client_cleanup(struct client *cli)
{
    /* Free any remote allocations that were never freed before disconnect */
    if (cli->feeder_id) {
        pthread_mutex_lock(&g_alloc_lock);
        for (int _i = 0; _i < MAX_REMOTE_ALLOCS; _i++) {
            struct remote_alloc *_ra = &g_remote_allocs[_i];
            if (!_ra->in_use || _ra->client_id != cli->feeder_id) continue;
            if (_ra->tier == 0) {
                if (f_cudaSetDevice) f_cudaSetDevice(_ra->device_id);
                if (f_cudaFree) f_cudaFree(_ra->dev_ptr);
            } else if (_ra->tier == 1) {
                if (f_cudaFreeHost) f_cudaFreeHost(_ra->host_ptr);
            } else {
                munmap(_ra->host_ptr, (size_t)_ra->size);
            }
            netd_log("disconnect cleanup: freed tier%d alloc 0x%llx (%llu MB)",
                     _ra->tier, (unsigned long long)_ra->handle,
                     (unsigned long long)(_ra->size >> 20));
            /* PR-T: tombstone hash before clearing in_use. */
            ra_hash_remove(_ra->handle, (uint32_t)_i);
            _ra->in_use = 0;
        }
        pthread_mutex_unlock(&g_alloc_lock);
    }

    /* PR-L: invalidate every g_kernel_map entry owned by this client.
     * Without this, stale host_func pointers accumulate and (worse) the
     * next client that happens to allocate the same feeder_id (slot reuse)
     * could call host_funcs registered by the previous tenant.  Quick
     * linear scan - 4096 entries is small enough on disconnect path. */
    pthread_mutex_lock(&g_kernel_map_lock);
    int cleared = 0;
    for (int i = 0; i < MAX_KERNEL_NAMES; i++) {
        if (g_kernel_map[i].in_use && g_kernel_map[i].client_id == cli->feeder_id) {
            /* PR-V: tombstone hash entry BEFORE blanking the name so the
             * (client_id, name) key is still valid for the removal call. */
            km_hash_remove(cli->feeder_id, g_kernel_map[i].name, (uint32_t)i);
            g_kernel_map[i].in_use = 0;
            g_kernel_map[i].host_func = NULL;
            g_kernel_map[i].client_id = 0;
            g_kernel_map[i].name[0] = '\0';
            cleared++;
        }
    }
    pthread_mutex_unlock(&g_kernel_map_lock);
    if (cleared > 0)
        netd_log("disconnect cleanup: cleared %d kernel-map entries for client %u",
                 cleared, cli->feeder_id);

    /* Drain any sticky CUDA error left by this client's last kernel launch.
     * A remotely-dispatched kernel that faulted (e.g. an illegal access from a
     * mismatched-build kernel) leaves the shared context in an error state that
     * makes EVERY later alloc/launch fail , the "feeder VRAM used but compute
     * 0%, then all ops fail" symptom (2026-07-06).  cudaGetLastError clears a
     * sticky *launch* error; a hard corruption still needs the restart the
     * caller does, but this recovers the common case without one. */
    if (f_cudaGetLastError) {
        cudaError_t _drained = f_cudaGetLastError();
        if (_drained != 0)
            netd_log("disconnect cleanup: drained sticky CUDA error %d", _drained);
    }

    NETD_EVT("CLIENT_DISC", "NET", 0, cli->feeder_id, "client_disconnected");
    ag_close(cli->feeder_id);
    if (cli->fd >= 0) close(cli->fd);
    free(cli->recv_buf);
    /* Phase 2b: destroy per-client async stream */
    if (cli->async_stream && f_cudaStreamDestroy) {
        f_cudaStreamDestroy(cli->async_stream);
        cli->async_stream = NULL;
    }
#ifdef GREENBOOST_USE_NCCL
    if (cli->nccl_comm) {
        ncclCommDestroy(cli->nccl_comm);
        cli->nccl_comm = NULL;
    }
#endif
    /* PR-FF: explicit_bzero the session key before memset clobbers it.
     * A plain memset on a struct that the compiler "knows" is about to
     * be discarded may be optimised out - explicit_bzero is the standard
     * primitive for "this MUST be zeroed". */
    explicit_bzero(cli->mac_session_key, sizeof(cli->mac_session_key));
    pthread_mutex_destroy(&cli->send_lock);  /* PR-QQ */
    memset(cli, 0, sizeof(*cli));
    cli->fd = -1;
}

static int client_process(struct client *cli)
{
    while (cli->recv_len >= GB_NET_HDR_SIZE) {
        const struct gb_net_header *hdr =
            (const struct gb_net_header *)cli->recv_buf;

        /* Audit F-L3-12: endian-aware magic check.  No-op on LE hosts; on BE
         * hosts the wire bytes are converted to host order before compare. */
        uint32_t hdr_magic = le32toh(hdr->magic);
        if (hdr_magic != GB_NET_MAGIC) {
            netd_log("ERROR: bad magic 0x%08x from %s - dropping",
                     hdr_magic, cli->remote_addr);
            return -1;
        }

        /* F-L3-09: within-session replay/reordering guard.  seq_num must be
         * strictly monotone; gap or replay → drop connection.
         * The increment happens BELOW, only after the full message is
         * buffered: incrementing here broke every message that straddled a
         * TCP read boundary , the partial-data break re-parsed the same
         * header on the next call and saw its own increment as a replay
         * ("expected N+1, got N" → connection dropped, shim got EPIPE,
         * feeder tiers reported 0 bytes). */
        uint32_t hdr_seq = le32toh(hdr->seq_num);
        if (hdr_seq != cli->recv_seq) {
            netd_log("ERROR: seq mismatch from %s (expected %u, got %u) - dropping",
                     cli->remote_addr, cli->recv_seq, hdr_seq);
            return -1;
        }

        uint32_t payload_len = le32toh(hdr->payload_len);
        /* Audit F-L3-03: explicit upper bound on payload_len before adding,
         * even though size_t arithmetic is safe on LP64.  Catches malformed
         * senders early and keeps the check identical across 32-bit builds. */
        if (payload_len > GB_NET_MAX_MSG_SIZE) {
            netd_log("ERROR: payload_len too large (%u) from %s",
                     payload_len, cli->remote_addr);
            return -1;
        }
        /* PR-FF: v4 messages have an 8-byte MAC after the payload. */
        size_t mac_len = cli->mac_enabled ? 8 : 0;
        size_t total = GB_NET_HDR_SIZE + payload_len + mac_len;
        if (total > RECV_BUF_SIZE) {
            netd_log("ERROR: message too large (%zu) from %s", total, cli->remote_addr);
            return -1;
        }

        if (cli->recv_len < total)
            break; /* need more data */

        /* Full message buffered , consume the sequence number now. */
        cli->recv_seq++;

        const void *payload = cli->recv_buf + GB_NET_HDR_SIZE;

        /* PR-FF: verify MAC BEFORE dispatching the message.  Bad MAC →
         * drop connection (an attacker who can't forge MACs cannot
         * inject messages even with full TCP intercept). */
        if (cli->mac_enabled) {
            uint8_t expected_mac[8];
            gb_msg_mac(cli->mac_session_key, hdr, payload, payload_len, expected_mac);
            const uint8_t *recv_mac = cli->recv_buf + GB_NET_HDR_SIZE + payload_len;
            int mac_bad = gb_consttime_memcmp(recv_mac, expected_mac, 8) != 0;
            explicit_bzero(expected_mac, sizeof(expected_mac));
            if (mac_bad) {
                netd_log("ERROR: MAC verification failed from %s - dropping",
                         cli->remote_addr);
                return -1;
            }
        }

        int ret = dispatch_message(cli, hdr, payload);

        /* Shift remaining data */
        size_t remaining = cli->recv_len - total;
        if (remaining > 0)
            memmove(cli->recv_buf, cli->recv_buf + total, remaining);
        cli->recv_len = remaining;

        if (ret < 0)
            return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Signal handling                                                    */
/* ------------------------------------------------------------------ */

static void sig_handler(int sig)
{
    (void)sig;
    /* U8: SIGTERM triggers graceful drain; SIGINT exits immediately */
    if (sig == SIGTERM)
        atomic_store_explicit(&g_draining, 1, memory_order_release);
    else
        g_running = 0;
}

/* ------------------------------------------------------------------ */
/*  Daemonize                                                          */
/* ------------------------------------------------------------------ */

/* Forward decl - defined earlier in this TU. */
static gb_u32 get_ddr_speed(void);

static void daemonize(void)
{
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); exit(1); }
    if (pid > 0) exit(0);

    if (setsid() < 0) exit(1);

    pid = fork();
    if (pid < 0) exit(1);
    if (pid > 0) exit(0);

    umask(0);
    if (chdir("/") < 0) { /* ignore */ }

    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);
    open("/dev/null", O_RDONLY);
    open("/dev/null", O_WRONLY);
    open("/dev/null", O_WRONLY);

    /* PR-CC pre-warm: get_ddr_speed forks + execve()s dmidecode the first
     * time it runs.  Between fork and execve in a multi-threaded process,
     * the child sees only the calling thread; any pthread_mutex_lock held
     * by another thread at fork() time stays locked forever in the child.
     * handle_cuda_launch creates a watchdog pthread per kernel launch
     * (line ~1947), so a heartbeat that calls get_ddr_speed during a
     * launch could deadlock.  Force the cache-fill here, while still
     * single-threaded, so all later get_ddr_speed calls just return the
     * cached value. */
    (void)get_ddr_speed();
}

static int write_pid_file(void)
{
    mkdir("/run/greenboost", 0755);
    FILE *f = fopen(GB_NETD_PID_FILE, "w");
    if (!f) return -1;
    fprintf(f, "%d\n", getpid());
    fclose(f);
    return 0;
}

static void remove_pid_file(void)
{
    unlink(GB_NETD_PID_FILE);
}

/* Rotate GB_NETD_LOG_FILE -> GB_NETD_LOG_FILE ".1" (overwriting any previous
 * ".1") if it's already at/over GB_NETD_LOG_MAX_BYTES. One generation is
 * enough for "what was this daemon doing before it filled up" without
 * building a full logrotate-style ladder into a small C daemon. Best-effort:
 * a failed rename just means the current file keeps growing, not a crash. */
static void rotate_log_if_oversized(void)
{
    struct stat st;
    if (stat(GB_NETD_LOG_FILE, &st) != 0)
        return;
    if ((unsigned long)st.st_size < GB_NETD_LOG_MAX_BYTES)
        return;
    char rotated[sizeof(GB_NETD_LOG_FILE) + 4];
    snprintf(rotated, sizeof(rotated), "%s.1", GB_NETD_LOG_FILE);
    rename(GB_NETD_LOG_FILE, rotated);
}

static int open_log(void)
{
    mkdir("/var/log/greenboost", 0755);
    rotate_log_if_oversized();
    g_logfp = fopen(GB_NETD_LOG_FILE, "a");
    if (!g_logfp) {
        fprintf(stderr, "WARNING: cannot open %s - logging to stderr\n", GB_NETD_LOG_FILE);
        g_logfp = NULL;
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Main event loop                                                    */
/* ------------------------------------------------------------------ */

static int is_lan_ip(uint32_t ip_net) {
    uint32_t ip = ntohl(ip_net);
    if ((ip & 0xFF000000) == 0x0A000000) return 1;  /* 10/8 */
    if ((ip & 0xFFF00000) == 0xAC100000) return 1;  /* 172.16/12 */
    if ((ip & 0xFFFF0000) == 0xC0A80000) return 1;  /* 192.168/16 */
    if ((ip & 0xFFFF0000) == 0xA9FE0000) return 1;  /* 169.254/16 link-local */
    /* PR-G: 127/8 is intentionally NOT accepted as LAN.
     * Reason: a userspace process listening on 127.0.0.1 (e.g. a SOCKS proxy,
     * a unprivileged forwarding helper, or even another container in the
     * same netns) would be able to bypass the LAN filter just by connecting
     * via localhost.  Loopback access from the same host should go through
     * an explicit unix-socket opt-in, not be silently classified as LAN.
     * Set GREENBOOST_LAN_INCLUDE_LOOPBACK=1 to restore old behaviour for
     * single-host dev/test setups. */
    {
        static int loopback_allowed = -1;
        if (__builtin_expect(loopback_allowed < 0, 0)) {
            const char *e = getenv("GREENBOOST_LAN_INCLUDE_LOOPBACK");
            loopback_allowed = (e && strcmp(e, "1") == 0) ? 1 : 0;
        }
        if (loopback_allowed && (ip & 0xFF000000) == 0x7F000000) return 1;
    }
    return 0;
}

/* ──────────────────────────────────────────────────────────────────────
 * PR-VV: worker pool scaffold.
 *
 * The netd reactor today is single-threaded: every CUDA op (malloc, big
 * memcpy, kernel dispatch) blocks the accept/recv loop.  A slow client
 * starves all other clients.
 *
 * This scaffold lays down the threading primitives without touching the
 * existing call sites - they keep running on the reactor thread by
 * default.  Opt in with GREENBOOST_WORKERS=N (1..32).  Future PRs migrate
 * specific handlers (handle_cuda_memcpy_h2d/d2h, handle_cuda_exec) onto
 * gb_workpool_submit() once their lock discipline has been audited for
 * multi-thread safety.
 *
 * Invariants the pool enforces:
 *   - At most GB_WP_MAX queued jobs; submit returns -1 when full so
 *     callers can fall back to inline execution instead of dropping work.
 *   - Workers run in detached pthreads launched once at boot; shutdown
 *     drains the queue and joins via a broadcast cond on g_wp.shutdown.
 *   - The queue is FIFO; no priority lanes.  If we need them later,
 *     a second pool keyed by job kind is simpler than fair scheduling.
 * ──────────────────────────────────────────────────────────────────── */
#define GB_WP_MAX 256

struct gb_work {
    void (*fn)(void *arg);
    void *arg;
};

static struct {
    pthread_mutex_t lock;
    pthread_cond_t  has_work;
    pthread_cond_t  has_space;
    struct gb_work  queue[GB_WP_MAX];
    int             head;     /* dequeue here */
    int             tail;     /* enqueue here */
    int             count;
    int             n_workers;
    pthread_t       workers[32];
    int             shutdown; /* 1 → drain + exit */
    /* Telemetry - read by vitals/stability monitor. */
    _Atomic uint64_t submitted;
    _Atomic uint64_t completed;
    _Atomic uint64_t rejected;
} g_wp = {
    .lock      = PTHREAD_MUTEX_INITIALIZER,
    .has_work  = PTHREAD_COND_INITIALIZER,
    .has_space = PTHREAD_COND_INITIALIZER,
};

static void *gb_workpool_thread(void *unused)
{
    (void)unused;
    for (;;) {
        pthread_mutex_lock(&g_wp.lock);
        while (g_wp.count == 0 && !g_wp.shutdown)
            pthread_cond_wait(&g_wp.has_work, &g_wp.lock);
        if (g_wp.shutdown && g_wp.count == 0) {
            pthread_mutex_unlock(&g_wp.lock);
            return NULL;
        }
        struct gb_work w = g_wp.queue[g_wp.head];
        g_wp.head = (g_wp.head + 1) % GB_WP_MAX;
        g_wp.count--;
        pthread_cond_signal(&g_wp.has_space);
        pthread_mutex_unlock(&g_wp.lock);

        w.fn(w.arg);
        atomic_fetch_add_explicit(&g_wp.completed, 1, memory_order_relaxed);
    }
}

static int gb_workpool_init(int n)
{
    if (n <= 0) return 0;       /* disabled */
    if (n > 32) n = 32;
    g_wp.n_workers = n;
    for (int i = 0; i < n; i++) {
        if (pthread_create(&g_wp.workers[i], NULL, gb_workpool_thread, NULL) != 0) {
            netd_log("ERROR: worker pool: pthread_create #%d failed", i);
            g_wp.n_workers = i;
            return -1;
        }
    }
    netd_log("PR-VV: worker pool active (%d threads)", n);
    return n;
}

/* Submit work.  Returns 0 on success, -1 if the pool is disabled or full;
 * the caller MUST handle the -1 case by running the work inline.  This
 * lets us keep the back-pressure semantics of single-thread execution
 * even under bursty load. */
static int gb_workpool_submit(void (*fn)(void *), void *arg)
{
    if (g_wp.n_workers == 0) return -1;
    pthread_mutex_lock(&g_wp.lock);
    if (g_wp.count >= GB_WP_MAX || g_wp.shutdown) {
        pthread_mutex_unlock(&g_wp.lock);
        atomic_fetch_add_explicit(&g_wp.rejected, 1, memory_order_relaxed);
        return -1;
    }
    g_wp.queue[g_wp.tail] = (struct gb_work){ .fn = fn, .arg = arg };
    g_wp.tail = (g_wp.tail + 1) % GB_WP_MAX;
    g_wp.count++;
    pthread_cond_signal(&g_wp.has_work);
    pthread_mutex_unlock(&g_wp.lock);
    atomic_fetch_add_explicit(&g_wp.submitted, 1, memory_order_relaxed);
    return 0;
}

static void gb_workpool_shutdown(void)
{
    if (g_wp.n_workers == 0) return;
    pthread_mutex_lock(&g_wp.lock);
    g_wp.shutdown = 1;
    pthread_cond_broadcast(&g_wp.has_work);
    pthread_mutex_unlock(&g_wp.lock);
    for (int i = 0; i < g_wp.n_workers; i++)
        pthread_join(g_wp.workers[i], NULL);
    netd_log("PR-VV: worker pool drained (submitted=%llu completed=%llu rejected=%llu)",
             (unsigned long long)atomic_load_explicit(&g_wp.submitted, memory_order_relaxed),
             (unsigned long long)atomic_load_explicit(&g_wp.completed, memory_order_relaxed),
             (unsigned long long)atomic_load_explicit(&g_wp.rejected, memory_order_relaxed));
    g_wp.n_workers = 0;
}

static int run_server(void)
{
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        netd_log("ERROR: socket(): %s", strerror(errno));
        return 1;
    }

    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons((uint16_t)g_port);
    if (inet_pton(AF_INET, g_bind_addr, &addr.sin_addr) <= 0) {
        netd_log("ERROR: invalid bind address '%s'", g_bind_addr);
        close(listen_fd);
        return 1;
    }

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        netd_log("ERROR: bind(%s:%d): %s", g_bind_addr, g_port, strerror(errno));
        close(listen_fd);
        return 1;
    }

    if (listen(listen_fd, 16) < 0) {
        netd_log("ERROR: listen(): %s", strerror(errno));
        close(listen_fd);
        return 1;
    }

    set_nonblock(listen_fd);

    int epfd = epoll_create1(EPOLL_CLOEXEC);
    if (epfd < 0) {
        netd_log("ERROR: epoll_create1(): %s", strerror(errno));
        close(listen_fd);
        return 1;
    }

    struct epoll_event ev = { .events = EPOLLIN, .data.fd = listen_fd };
    epoll_ctl(epfd, EPOLL_CTL_ADD, listen_fd, &ev);

    netd_log("Listening on %s:%d (%d GPUs available)", g_bind_addr, g_port, g_gpu_count);
    for (int i = 0; i < g_gpu_count; i++)
        netd_log("  GPU %d: %s - %llu MB VRAM", i, g_gpus[i].name,
                 (unsigned long long)(g_gpus[i].vram_bytes >> 20));
    netd_log("Feeder ready - waiting for connections...");

    /* Also print to stdout for interactive mode */
    if (!g_daemonize) {
        fprintf(stderr, "[GreenBoost] Feeder listening on %s:%d\n", g_bind_addr, g_port);
        for (int i = 0; i < g_gpu_count; i++)
            fprintf(stderr, "[GreenBoost] GPU %d: %s - %llu MB VRAM\n",
                    i, g_gpus[i].name,
                    (unsigned long long)(g_gpus[i].vram_bytes >> 20));
        fprintf(stderr, "[GreenBoost] Feeder ready - waiting for connections...\n");
    }

    struct epoll_event events[EPOLL_EVENTS];

    /* U8: drain-completion state */
    uint64_t drain_start_ms = 0;

    while (g_running) {
        /* U8: when draining, wait up to 5 s for in-flight ops to finish */
        if (atomic_load_explicit(&g_draining, memory_order_acquire)) {
            if (!drain_start_ms) {
                drain_start_ms = mono_ms();
                netd_log("SIGTERM: graceful drain started - waiting for in-flight ops");
            }
            int inflight = atomic_load_explicit(&g_inflight_ops, memory_order_acquire);
            uint64_t elapsed = mono_ms() - drain_start_ms;
            if (inflight == 0 || elapsed >= 5000) {
                netd_log("drain complete (%d in-flight, %llu ms elapsed) - exiting",
                         inflight, (unsigned long long)elapsed);
                g_running = 0;
                break;
            }
        }

        int nfds = epoll_wait(epfd, events, EPOLL_EVENTS, 100 /* 100ms when draining */);
        if (nfds < 0) {
            if (errno == EINTR) continue;
            netd_log("ERROR: epoll_wait(): %s", strerror(errno));
            break;
        }

        uint64_t now_ms = mono_ms();

        for (int i = 0; i < nfds; i++) {
            if (events[i].data.fd == listen_fd) {
                /* New connection */
                struct sockaddr_in client_addr;
                socklen_t clen = sizeof(client_addr);
                int cfd = accept(listen_fd, (struct sockaddr *)&client_addr, &clen);
                if (cfd < 0) continue;

                /* TCP_NODELAY for low-latency control messages.
                 * NOTE: set_nonblock() is deferred until after the PSK
                 * handshake so that send/recv during auth use blocking I/O. */
                int nodelay = 1;
                setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
                /* Match the client's explicit socket buffer sizing (netc.c
                 * connect_feeder) so throughput on bulk transfers isn't
                 * capped by one side's default window while the other has
                 * room. */
                int sockbuf = 2 * (int)GB_NET_MAX_MSG_SIZE;
                setsockopt(cfd, SOL_SOCKET, SO_SNDBUF, &sockbuf, sizeof(sockbuf));
                setsockopt(cfd, SOL_SOCKET, SO_RCVBUF, &sockbuf, sizeof(sockbuf));

                char addr_str[64];

                                inet_ntop(AF_INET, &client_addr.sin_addr, addr_str, sizeof(addr_str));
                if (!is_lan_ip(client_addr.sin_addr.s_addr)) {
                    netd_log("WARNING: Rejecting connection from non-LAN IP: %s", addr_str);
                    close(cfd);
                    continue;
                }
                inet_ntop(AF_INET, &client_addr.sin_addr, addr_str, sizeof(addr_str));

                int slot = -1;
                for (int j = 0; j < MAX_CLIENTS; j++) {
                    if (!g_clients[j].active) { slot = j; break; }
                }
                if (slot < 0) {
                    netd_log("WARN: max clients reached - rejecting %s", addr_str);
                    close(cfd);
                    continue;
                }

                /* F-L3-01: PSK authentication handshake.
                 * PR-C/C5: install send + recv timeouts before doing blocking
                 * I/O on cfd.  Without these, an attacker that opens a TCP
                 * connection and never sends bytes pins the single-threaded
                 * epoll loop indefinitely on gb_recv_all_blocking, starving
                 * every existing client (heartbeat, alloc, exec) - a one-line
                 * pre-auth DoS.  Two seconds is generous; legitimate clients
                 * send their MAC immediately after reading the nonce.
                 * PR-C/C7: cluster.key absence is fail-CLOSED.  The old
                 * backward-compat warning silently regressed to F-L3-01.
                 * Opt-in to the legacy behaviour with GREENBOOST_ALLOW_UNAUTH=1
                 * (intended for local-loopback bring-up only). */
                /* PR-FF: outputs of the auth block that the post-auth setup
                 * needs (mac_enabled + final_session_key for v4 mode).
                 * Hoisted out of the auth block so we can install them into
                 * g_clients[slot] AFTER client_init.  Zeroed up-front so the
                 * v3 path leaves them in a safe (mac_off) state. */
                int auth_mac_enabled = 0;
                uint8_t auth_session_key[32];
                memset(auth_session_key, 0, sizeof(auth_session_key));
                {
                    struct timeval auth_to = { .tv_sec = 2, .tv_usec = 0 };
                    setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &auth_to, sizeof(auth_to));
                    setsockopt(cfd, SOL_SOCKET, SO_SNDTIMEO, &auth_to, sizeof(auth_to));

                    /* PR-EE: opt-in mutual-auth + HKDF session key.
                     * Set GREENBOOST_PSK_V4=1 on BOTH sides to enable.
                     * Cached at first auth attempt; default = v3. */
                    static int gb_psk_v4 = -1;
                    if (gb_psk_v4 < 0) {
                        const char *e = getenv("GREENBOOST_PSK_V4");
                        gb_psk_v4 = (e && strcmp(e, "1") == 0) ? 1 : 0;
                        if (gb_psk_v4)
                            netd_log("AUTH: v4 mode enabled (mutual auth + HKDF session key + per-msg MAC)");
                    }

                    uint8_t psk[32];
                    if (gb_load_psk(psk) == 0) {
                        uint8_t nonce_s[32];   /* server nonce (sent first) */
                        uint8_t nonce_c[32];   /* client nonce (v4 only) */
                        uint8_t session_key[32];  /* derived (v4 only) */
                        memset(nonce_c, 0, sizeof(nonce_c));
                        memset(session_key, 0, sizeof(session_key));
                        if (syscall(SYS_getrandom, nonce_s, sizeof(nonce_s), 0) < 0) {
                            netd_log("AUTH: getrandom failed for %s: %s", addr_str, strerror(errno));
                            close(cfd);
                            continue;
                        }
                        if (send_all(cfd, nonce_s, sizeof(nonce_s)) != (ssize_t)sizeof(nonce_s)) {
                            netd_log("AUTH: nonce send failed to %s", addr_str);
                            close(cfd);
                            continue;
                        }
                        /* v3: client returns 32-byte MAC.
                         * v4: client returns 32-byte nonce_c || 32-byte mac1. */
                        size_t expected_len = gb_psk_v4 ? 64 : 32;
                        uint8_t client_reply[64];
                        if (gb_recv_all_blocking(cfd, client_reply, expected_len)
                                != (ssize_t)expected_len) {
                            netd_log("AUTH: %s recv failed/timeout from %s (v%d)",
                                     gb_psk_v4 ? "nonce_c+mac1" : "MAC", addr_str,
                                     gb_psk_v4 ? 4 : 3);
                            close(cfd);
                            continue;
                        }
                        uint8_t expected[32];
                        int mac_bad;
                        if (gb_psk_v4) {
                            memcpy(nonce_c, client_reply, 32);
                            /* mac1 = HMAC(psk, nonce_s || nonce_c) */
                            uint8_t challenge[64];
                            memcpy(challenge,      nonce_s, 32);
                            memcpy(challenge + 32, nonce_c, 32);
                            gb_hmac_sha256_salt(psk, 32, challenge, 64, expected);
                            mac_bad = gb_consttime_memcmp(client_reply + 32, expected, 32) != 0;
                            explicit_bzero(challenge, sizeof(challenge));
                            if (!mac_bad) {
                                /* Send mac2 = HMAC(psk, nonce_c || nonce_s) */
                                uint8_t mac2_in[64];
                                memcpy(mac2_in,      nonce_c, 32);
                                memcpy(mac2_in + 32, nonce_s, 32);
                                uint8_t mac2[32];
                                gb_hmac_sha256_salt(psk, 32, mac2_in, 64, mac2);
                                int send_ok = send_all(cfd, mac2, 32) == 32;
                                explicit_bzero(mac2_in, sizeof(mac2_in));
                                explicit_bzero(mac2, sizeof(mac2));
                                if (!send_ok) {
                                    netd_log("AUTH: v4 mac2 send failed to %s", addr_str);
                                    explicit_bzero(psk, sizeof(psk));
                                    explicit_bzero(nonce_s, sizeof(nonce_s));
                                    explicit_bzero(nonce_c, sizeof(nonce_c));
                                    explicit_bzero(expected, sizeof(expected));
                                    explicit_bzero(client_reply, sizeof(client_reply));
                                    close(cfd);
                                    continue;
                                }
                                /* Derive session key.  PR-FF: stash into the
                                 * outer-scope auth_session_key so per-message
                                 * MAC can use it after client_init. */
                                gb_derive_session_key(psk, nonce_s, nonce_c, session_key);
                                memcpy(auth_session_key, session_key, 32);
                                auth_mac_enabled = 1;
                            }
                        } else {
                            /* v3 legacy: mac = HMAC(psk, nonce_s) */
                            gb_hmac_sha256(psk, nonce_s, sizeof(nonce_s), expected);
                            mac_bad = gb_consttime_memcmp(client_reply, expected, 32) != 0;
                        }
                        /* PR-CC: wipe all key-derived material from stack before
                         * returning or proceeding.  PR-EE: also wipe nonce_c +
                         * session_key for v4. */
                        explicit_bzero(psk, sizeof(psk));
                        explicit_bzero(nonce_s, sizeof(nonce_s));
                        explicit_bzero(nonce_c, sizeof(nonce_c));
                        explicit_bzero(session_key, sizeof(session_key));
                        explicit_bzero(expected, sizeof(expected));
                        explicit_bzero(client_reply, sizeof(client_reply));
                        if (mac_bad) {
                            netd_log("AUTH: rejected connection from %s - bad MAC", addr_str);
                            close(cfd);
                            continue;
                        }
                        netd_log("AUTH: PSK verified for %s", addr_str);
                    } else {
                        /* No /etc/greenboost/cluster.key.  Refuse unless the
                         * operator explicitly opted into unauthenticated mode. */
                        const char *allow = getenv("GREENBOOST_ALLOW_UNAUTH");
                        if (!allow || strcmp(allow, "1") != 0) {
                            netd_log("AUTH: rejected %s - /etc/greenboost/cluster.key absent "
                                     "(set GREENBOOST_ALLOW_UNAUTH=1 to allow unauth)", addr_str);
                            close(cfd);
                            continue;
                        }
                        netd_log("WARN: GREENBOOST_ALLOW_UNAUTH=1 - accepting unauthenticated "
                                 "connection from %s", addr_str);
                    }
                }

                /* Switch to non-blocking now that PSK handshake is complete.
                 *
                 * PR-CC: explicitly clear SO_RCVTIMEO / SO_SNDTIMEO so any future
                 * blocking call on cfd (e.g. a synchronous reply path) does not
                 * silently inherit the 2-second auth timeout and look like a
                 * network glitch.  set_nonblock alone does NOT clear these
                 * options - they're independent of O_NONBLOCK. */
                {
                    struct timeval no_to = { .tv_sec = 0, .tv_usec = 0 };
                    setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &no_to, sizeof(no_to));
                    setsockopt(cfd, SOL_SOCKET, SO_SNDTIMEO, &no_to, sizeof(no_to));
                }
                set_nonblock(cfd);

                client_init(&g_clients[slot], cfd, addr_str, now_ms);
                /* PR-FF: install v4 session key into the client struct.
                 * client_init zero-initialised both fields; copy the key
                 * IFF v4 auth succeeded.  After this point the auth_*
                 * locals are dead - wipe them. */
                if (auth_mac_enabled) {
                    g_clients[slot].mac_enabled = 1;
                    memcpy(g_clients[slot].mac_session_key, auth_session_key, 32);
                }
                explicit_bzero(auth_session_key, sizeof(auth_session_key));
                ev.events  = EPOLLIN | EPOLLHUP | EPOLLERR;
                ev.data.fd = cfd;
                epoll_ctl(epfd, EPOLL_CTL_ADD, cfd, &ev);

                netd_log("Connection from %s (slot %d, v%d)", addr_str, slot,
                         auth_mac_enabled ? 4 : 3);
                if (!g_daemonize)
                    fprintf(stderr, "[GreenBoost] Connection from %s%s\n",
                            addr_str, auth_mac_enabled ? " (v4 MAC)" : "");

            } else {
                /* Data from existing client */
                int cfd = events[i].data.fd;
                struct client *cli = NULL;
                for (int j = 0; j < MAX_CLIENTS; j++) {
                    if (g_clients[j].active && g_clients[j].fd == cfd) {
                        cli = &g_clients[j];
                        break;
                    }
                }
                if (!cli) {
                    epoll_ctl(epfd, EPOLL_CTL_DEL, cfd, NULL);
                    close(cfd);
                    continue;
                }

                if (events[i].events & (EPOLLHUP | EPOLLERR)) {
                    netd_log("Client %s hung up / error", cli->remote_addr);
                    epoll_ctl(epfd, EPOLL_CTL_DEL, cfd, NULL);
                    client_cleanup(cli);
                    continue;
                }

                ssize_t n = recv(cfd, cli->recv_buf + cli->recv_len,
                                 cli->recv_cap - cli->recv_len, 0);
                if (n <= 0) {
                    if (n == 0)
                        netd_log("Client %s closed connection", cli->remote_addr);
                    else if (errno != EAGAIN && errno != EWOULDBLOCK)
                        netd_log("Client %s recv error: %s", cli->remote_addr, strerror(errno));
                    if (n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
                        epoll_ctl(epfd, EPOLL_CTL_DEL, cfd, NULL);
                        client_cleanup(cli);
                    }
                    continue;
                }

                cli->recv_len += (size_t)n;

                /* Any inbound traffic proves the client is alive.  Counting
                 * only GB_MSG_HEARTBEAT here dropped clients mid-model-load:
                 * the shim's heartbeats pause while the host loads weights
                 * (20+ s of local memcpy), and a client streaming a large
                 * H2D transfer holds per_lock so its heartbeat thread skips ,
                 * both got disconnected at GB_NET_HEARTBEAT_TIMEOUT_MS while
                 * demonstrably active (hit live 2026-07-06). */
                cli->last_heartbeat_ms = mono_ms();

                if (client_process(cli) < 0) {
                    epoll_ctl(epfd, EPOLL_CTL_DEL, cfd, NULL);
                    client_cleanup(cli);
                }
            }
        }

        /* Heartbeat timeout check.  MUST use a FRESH clock, not the loop-top
         * now_ms: a blocking handler earlier in this iteration (e.g. a 21 GB
         * cudaHostAlloc pinning for 10-30 s) lets last_heartbeat_ms advance
         * past the stale now_ms , the unsigned subtraction then wraps to a
         * huge value and drops a client that was served THIS iteration
         * (freed its 21 GB weights buffer mid-load, hit live 2026-07-06). */
        uint64_t hb_now = mono_ms();
        for (int j = 0; j < MAX_CLIENTS; j++) {
            if (!g_clients[j].active) continue;
            if (g_clients[j].last_heartbeat_ms < hb_now &&
                hb_now - g_clients[j].last_heartbeat_ms > GB_NET_HEARTBEAT_TIMEOUT_MS) {
                netd_log("Client %s heartbeat timeout - disconnecting", g_clients[j].remote_addr);
                epoll_ctl(epfd, EPOLL_CTL_DEL, g_clients[j].fd, NULL);
                client_cleanup(&g_clients[j]);
            }
        }
    }

    /* Cleanup */
    for (int j = 0; j < MAX_CLIENTS; j++) {
        if (g_clients[j].active)
            client_cleanup(&g_clients[j]);
    }
    close(epfd);
    close(listen_fd);
    netd_log("Feeder daemon stopped");
    return 0;
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */

static void usage(void)
{
    fprintf(stderr,
        "greenboost-netd - GreenBoost Network Feeder Daemon\n"
        "\n"
        "Usage: greenboost-netd [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  -d              Daemonize (background mode)\n"
        "  -p PORT         Listen port (default: %d)\n"
        "  --bind ADDR     Bind address (default: 0.0.0.0)\n"
        "  -h, --help      Show this help\n"
        "\n", GB_NET_PORT);
}

int main(int argc, char *argv[])
{
    /* Feeder GPU compute needs the __cudaRegisterFunction interposer active
     * BEFORE libggml-cuda.so is dlopened, so the kernel name→stub map is
     * captured (the lib is stripped; dlsym can't resolve device stubs).
     * Re-exec ourselves once with the capture lib preloaded.  GB_NETD_CAPTURED=1
     * prevents an exec loop; GB_NETD_NO_CAPTURE=1 opts out. */
    if (!getenv("GB_NETD_CAPTURED") && !getenv("GB_NETD_NO_CAPTURE")) {
        const char *cands[] = {
            "/usr/local/lib/libgreenboost_netd_capture.so",
            "./libgreenboost_netd_capture.so",
            NULL
        };
        const char *cap = NULL;
        for (int i = 0; cands[i]; i++) {
            if (access(cands[i], R_OK) == 0) { cap = cands[i]; break; }
        }
        if (cap) {
            const char *old = getenv("LD_PRELOAD");
            char pl[1024];
            if (old && old[0])
                snprintf(pl, sizeof(pl), "%s:%s", cap, old);
            else
                snprintf(pl, sizeof(pl), "%s", cap);
            setenv("LD_PRELOAD", pl, 1);
            setenv("GB_NETD_CAPTURED", "1", 1);
            execv("/proc/self/exe", argv);
            /* exec failed , continue without capture (compute-less but alive). */
            unsetenv("GB_NETD_CAPTURED");
        }
    }

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0) {
            g_daemonize = 1;
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            g_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
            strncpy(g_bind_addr, argv[++i], sizeof(g_bind_addr) - 1);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage();
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage();
            return 1;
        }
    }

    /* Hostname */
    gethostname(g_hostname, sizeof(g_hostname) - 1);

    /* Signals */
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGPIPE, SIG_IGN);

    /* Pre-load gb-synapse's CUDA libraries into the global symbol namespace so
     * that dlsym(RTLD_DEFAULT, kernel_name) can find kernels when the host
     * sends GB_MSG_CUDA_REGISTER_FN + GB_MSG_CUDA_LAUNCH requests. gb-synapse's
     * engine (2026-07-15, replaces the Ollama bundled build) links the system
     * CUDA 12 runtime directly (verified via ldd , no cuDNN dependency at all,
     * llama.cpp doesn't use it) , only cublas/cublasLt + its own libggml-cuda.so
     * need preloading; the Ollama paths stay as a fallback for boxes that
     * haven't rebuilt the synapse engine yet. */
    {
        static const char *synapse_libs[] = {
            "/usr/lib/x86_64-linux-gnu/libcublas.so.12",
            "/usr/lib/x86_64-linux-gnu/libcublasLt.so.12",
            "/usr/local/lib/greenboost/synapse/libggml-cuda.so",
            "/usr/local/lib/ollama/cuda_v13/libcublas.so",
            "/usr/local/lib/ollama/cuda_v13/libcublasLt.so",
            "/usr/local/lib/ollama/libggml-cuda.so",
            NULL
        };
        for (int _i = 0; synapse_libs[_i]; _i++) {
            if (dlopen(synapse_libs[_i], RTLD_NOW | RTLD_GLOBAL) == NULL)
                netd_log("INFO: optional lib not found: %s", synapse_libs[_i]);
        }
    }

    open_log();

    /* Daemonize BEFORE any GPU/CUDA init.
     * CUDA runtime is not fork-safe: cudaGetDeviceCount() initialises the
     * runtime lazily; any subsequent fork() corrupts it in the child.
     * By forking first and running probe_gpus() only in the daemon child,
     * CUDA is never touched in the pre-fork process. */
    if (g_daemonize) {
        daemonize();
        write_pid_file();
        netd_log("Daemon started (pid %d)", getpid());
    }

    /* Probe GPUs - runs in daemon child (or foreground process if not daemonizing) */
    if (probe_gpus() < 0) {
        if (g_daemonize) remove_pid_file();
        netd_log("ERROR: GPU probe failed - shutting down");
        return 1;
    }

    /* F-L3-08: initialise per-boot handle salt in daemon child (after fork) */
    syscall(SYS_getrandom, &g_handle_salt, sizeof(g_handle_salt), 0);
    if (g_handle_salt == 0) g_handle_salt = (uint64_t)time(NULL) * 6364136223846793005ULL;

    /* PR-VV: opt-in worker pool.  Default 0 keeps the current single-thread
     * reactor; set GREENBOOST_WORKERS=N to spawn N threads ready for
     * gb_workpool_submit(). */
    {
        const char *wenv = getenv("GREENBOOST_WORKERS");
        if (wenv && *wenv) {
            int n = atoi(wenv);
            if (n > 0) gb_workpool_init(n);
        }
    }

    int ret = run_server();

    gb_workpool_shutdown();
    remove_pid_file();
    if (g_logfp) fclose(g_logfp);
    return ret;
}

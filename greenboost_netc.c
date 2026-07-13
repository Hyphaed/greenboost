/* SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
 * GreenBoost v3.2 - Network Client Implementation
 *
 * Author  : Ferran Duarri
 * License : GPL v2 (open-source) / Commercial - see LICENSE
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <poll.h>
#include <stdint.h>
#include <endian.h>     /* PR-G/H11: le32toh for endian-safe wire field reads */
#include <sys/stat.h>   /* PR-G: stat() for PSK keyfile mode check */
#include <strings.h>    /* PR-G: explicit_bzero */

#ifdef GREENBOOST_USE_NCCL
#include <nccl.h>
#endif

#include "greenboost_netc.h"
#include "features/net_fabric.h"

#ifdef GB_HAVE_ZSTD
#include <zstd.h>
/* ── Fabric zstd compression config (host/client side) ──────────────
 * GbE (~110 MB/s) is the cluster bottleneck; zstd-3 compresses several×
 * faster than the link, so effective H2D throughput ≈ link × ratio for
 * compressible payloads (bf16 weights ~1.5-1.8×). Read once, cached. */
static int      g_zstd_enabled = -1;   /* -1 = unread; 0/1 after parse   */
static int      g_zstd_level   = 3;
static uint32_t g_zstd_min     = 65536;

static void gb_netc_zstd_init(void)
{
    if (g_zstd_enabled >= 0) return;
    const char *e = getenv("GB_NET_COMPRESS");
    g_zstd_enabled = (e && e[0] == '0') ? 0 : 1;   /* default ON when built */
    const char *lv = getenv("GB_NET_COMPRESS_LEVEL");
    if (lv && lv[0]) { int v = atoi(lv); if (v >= 1 && v <= 19) g_zstd_level = v; }
    const char *mn = getenv("GB_NET_COMPRESS_MIN");
    if (mn && mn[0]) { long v = atol(mn); if (v >= 0) g_zstd_min = (uint32_t)v; }
}
#endif

/* ── PSK authentication helpers (F-L3-01) ───────────────────────────
 * Shared secret in /etc/greenboost/cluster.key (hex-encoded 32 bytes).
 * Self-contained HMAC-SHA256; no OpenSSL dependency required here.   */

#define GB_SHA256_BLOCK  64
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
    /* PR-HH: wipe HMAC stack state - mirrors netd.c fix.  Closes the
     * v3-path leak that PR-CC's PSK-on-stack hardening missed. */
    explicit_bzero(k_ipad, sizeof(k_ipad));
    explicit_bzero(k_opad, sizeof(k_opad));
    explicit_bzero(inner, sizeof(inner));
}

/* PR-EE: variable-key HMAC for HKDF Extract.  See greenboost_netd.c
 * gb_hmac_sha256_salt for full commentary. */
static void gb_hmac_sha256_salt(const uint8_t *key, size_t key_len,
                                 const uint8_t *msg, size_t msg_len,
                                 uint8_t out[32])
{
    uint8_t k_pad[GB_SHA256_BLOCK] = {0};
    if (key_len > GB_SHA256_BLOCK) {
        gb_sha256_ctx c;
        gb_sha256_init(&c);
        gb_sha256_update(&c, key, key_len);
        gb_sha256_final(&c, k_pad);
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

/* PR-EE: derive 32-byte session key.  Mirror of netd impl. */
static void gb_derive_session_key(const uint8_t psk[32],
                                   const uint8_t nonce_s[32],
                                   const uint8_t nonce_c[32],
                                   uint8_t session_key[32])
{
    uint8_t salt[64];
    memcpy(salt,      nonce_s, 32);
    memcpy(salt + 32, nonce_c, 32);
    uint8_t prk[32];
    gb_hmac_sha256_salt(salt, sizeof(salt), psk, 32, prk);
    static const uint8_t info[] = "gb-session-v1|proto=4";
    uint8_t info_buf[sizeof(info)];
    memcpy(info_buf, info, sizeof(info) - 1);
    info_buf[sizeof(info) - 1] = 0x01;
    gb_hmac_sha256_salt(prk, 32, info_buf, sizeof(info_buf), session_key);
    explicit_bzero(salt, sizeof(salt));
    explicit_bzero(prk, sizeof(prk));
    explicit_bzero(info_buf, sizeof(info_buf));
}

/* Load 32-byte PSK from /etc/greenboost/cluster.key (hex-encoded).
 * Returns 0 on success, -1 if key file absent or malformed.
 *
 * PR-G hardening (mirrors netd):
 *   - Accept mode 0600/0400 (root-only) or 0640 root:greenboost (group read).
 *   - Refuse world access, group write/execute, or wrong group on 0640.
 *   - Require exactly 64 hex chars (no silent leading-zero padding).
 *   - explicit_bzero the on-stack hex buffer before return.
 *   - File must be a regular file. */
static int gb_load_psk(uint8_t key[32])
{
    const char *path = "/etc/greenboost/cluster.key";
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    if (!S_ISREG(st.st_mode)) return -1;
    if (gb_check_keyfile_mode(&st, path) != 0) {
        /* netc_log macro is defined later in this TU; use stderr directly. */
        fprintf(stderr, "[GreenBoost netc] AUTH: %s has insecure mode 0%o - "
                "must be 0600 (root-only) or 0640 root:%s\n",
                path, st.st_mode & 07777, GB_KEYFILE_GRP);
        return -1;
    }
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char hex[65] = {0};
    size_t n = fread(hex, 1, 64, f);
    fclose(f);
    if (n != 64) {
        explicit_bzero(hex, sizeof(hex));
        return -1;
    }
    for (int i = 0; i < 64; i++) {
        char c = hex[i];
        if (!((c >= '0' && c <= '9') ||
              (c >= 'a' && c <= 'f') ||
              (c >= 'A' && c <= 'F'))) {
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

/* ── NVTX instrumentation - mirrors shim macros (same .so) ──────────
 * Nsight ranges: compiled in when GREENBOOST_USE_NVTX is set.
 * Always-on event log: netc_nvtx_event() writes to the shared log file.  */
#ifdef GREENBOOST_USE_NVTX
#include <nvtx3/nvToolsExt.h>
#define NETC_NVTX_PUSH(name, color)  do { \
    nvtxEventAttributes_t _ea = {0}; \
    _ea.version = NVTX_VERSION; _ea.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    _ea.colorType = NVTX_COLOR_ARGB; _ea.color = (color); \
    _ea.messageType = NVTX_MESSAGE_TYPE_ASCII; _ea.message.ascii = (name); \
    nvtxRangePushEx(&_ea); } while(0)
#define NETC_NVTX_POP()  nvtxRangePop()
#else
#define NETC_NVTX_PUSH(name, color)  ((void)0)
#define NETC_NVTX_POP()              ((void)0)
#endif

#define NETC_NVTX_COLOR_NET    0xFF00FF88
#define NETC_NVTX_COLOR_KERN   0xFF00FF44
#define NETC_NVTX_COLOR_MEMCPY 0xFF44AAFF
#define NETC_NVTX_COLOR_HEALTH 0xFFCCCC00
#define NETC_NVTX_COLOR_ERR    0xFFFF0000
#define NETC_NVTX_COLOR_ALLOC  0xFF0000FF

static int g_netc_nvtx_fd = -2;

static void netc_nvtx_event(const char *type, const char *tier,
                            size_t size_mb, uintptr_t ptr, const char *detail)
{
    if (g_netc_nvtx_fd == -2) {
        g_netc_nvtx_fd = open("/run/greenboost/nvtx_events.log",
                              O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
        if (g_netc_nvtx_fd < 0)
            g_netc_nvtx_fd = open("/tmp/greenboost_nvtx_events.log",
                                  O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    }
    if (g_netc_nvtx_fd < 0) return;
    struct timespec _ts; clock_gettime(CLOCK_REALTIME, &_ts);
    uint64_t ms = (uint64_t)_ts.tv_sec * 1000ULL + (uint64_t)_ts.tv_nsec / 1000000ULL;
    char buf[288];
    int n = snprintf(buf, sizeof(buf), "%llu NETC %-24s %-12s %6zuMB ptr=0x%012lx %s\n",
                     (unsigned long long)ms, type, tier, size_mb, ptr, detail);
    if (n > 0 && n < (int)sizeof(buf)) {
        ssize_t _wr = write(g_netc_nvtx_fd, buf, (size_t)n); (void)_wr;
    }
}
#define NETC_NVTX_EVENT(type, tier, size_mb, ptr, detail) \
    netc_nvtx_event((type), (tier), (size_mb), (uintptr_t)(ptr), (detail))

/* ------------------------------------------------------------------ */
/*  Internal types                                                     */
/* ------------------------------------------------------------------ */

#define NETC_MAX_FEEDERS   GB_NET_MAX_FEEDERS
#define NETC_MAX_REMOTE_GPUS  (NETC_MAX_FEEDERS * GB_NET_MAX_GPUS)
#define NETC_MAX_KERNELS   4096
#define NETC_MAX_ALLOCS    131072   /* A2: doubled for wraparound headroom */
/* Audit F-L3-07: align with netd RECV_BUF_SIZE via the shared net_fabric.h
 * constant.  Earlier the netc buffer was 8 MiB and netd was 4 MiB; on the
 * netd side, oversized payloads were rejected but the TCP stream was already
 * partially consumed → framing desync.  Single source of truth now. */
/* PR-HH: +24 slack for the 8-byte v4 MAC suffix + 16-byte header so a
 * GB_NET_MAX_MSG_SIZE payload fits on the wire in v4 mode.  Cap on PAYLOAD
 * bytes (hdr_payload_len check at line ~708) remains GB_NET_MAX_MSG_SIZE. */
#define NETC_RECV_BUF      (GB_NET_MAX_MSG_SIZE + GB_NET_HDR_SIZE + 8)

/* A2: Virtual address range used for fake remote pointers.
 * 0xAA00_0000_0000 .. 0xAAFF_FFFF_FFFF = 1 TB virtual, never mapped. */
#define GB_FAKE_PTR_RANGE  0x10000000000ULL   /* 1 TB = 2^40 */

/* U4: 5-state feeder health machine (MuxFlow §4.2 pattern).
 * Transitions: HEALTHY → DEGRADED (throttle or SBE) → UNHEALTHY (sustained) →
 *              QUARANTINE (DBE or long unhealthy) → DISABLED (TTL, auto-retry).
 * DISABLED auto-recovers to HEALTHY after GB_FEEDER_DISABLE_TTL_MS. */
typedef enum {
    GB_HEALTH_HEALTHY    = 0,  /* normal - all tiers eligible           */
    GB_HEALTH_DEGRADED   = 1,  /* throttled or SBE - T1 deprioritized   */
    GB_HEALTH_UNHEALTHY  = 2,  /* sustained throttle - T1 skipped first */
    GB_HEALTH_QUARANTINE = 3,  /* DBE or long unhealthy - T1 blocked    */
    GB_HEALTH_DISABLED   = 4,  /* all tiers blocked; auto-retry after TTL */
} gb_feeder_health_state_t;

#define GB_FEEDER_DEGRADED_TTL_MS   10000   /* 10 s throttled → UNHEALTHY */
#define GB_FEEDER_UNHEALTHY_TTL_MS  30000   /* 30 s unhealthy → QUARANTINE */
#define GB_FEEDER_DISABLE_TTL_MS   120000   /* 2 min DISABLED → HEALTHY retry */

/* U13: Per-feeder PID controller for kernel dispatch rate limiting.
 * Setpoint is SM utilization %. On DEGRADED/UNHEALTHY health states the
 * effective setpoint is halved to cool the feeder GPU. */
typedef struct {
    double   setpoint_pct;     /* target SM util % (default 80.0) */
    double   integral;         /* accumulated I term               */
    double   prev_error;       /* derivative term                  */
    double   kp, ki, kd;       /* PID gains                        */
    uint32_t target_rate;      /* kernels/s limit (0 = unlimited)  */
    uint32_t dispatch_count;   /* kernels dispatched this window   */
    uint64_t window_start_ms;  /* start of current 1-s window      */
} gb_pid_state_t;

struct netc_feeder {
    int      fd;
    int      connected;
    char     addr[64];
    int      port;
    char     hostname[GB_NET_MAX_HOSTNAME];
    uint32_t feeder_id;
    int      gpu_count;
    struct gb_net_gpu_info gpus[GB_NET_MAX_GPUS];
    /* Cached speeds - populated at connect time, avoids live queries per allocation */
    uint32_t t2_speed_mts;          /* DDR speed MT/s */
    uint32_t t3_speed_mbs;          /* NVMe speed MB/s */
    /* D1: PCIe link characteristics from handshake */
    uint32_t pcie_link_gen;
    uint32_t pcie_link_width;
    uint32_t pcie_effective_bw_mbs; /* 0 = unknown, used for feeder scoring */
    /* D2: last throttle state from heartbeat */
    uint32_t throttle_reasons;      /* NVML bitmask; non-zero = throttled */
    /* D3: ECC health per GPU */
    int      t1_ecc_quarantine;     /* 1 = skip T1 feeder alloc (elevated SBE rate) */
    uint32_t ecc_dbe_count;         /* cumulative DBE - if > 0, feeder flagged degraded */
    /* GPU utilization + thermal + power cached from last heartbeat (gpu_load[0]) */
    uint32_t gpu_util_pct;          /* 0-100; 0 until first heartbeat */
    uint32_t gpu_mem_util_pct;      /* 0-100 memory controller utilization */
    uint16_t gpu_temp_c;            /* GPU temperature °C; 0 until first heartbeat */
    uint16_t gpu_power_draw_w;      /* current power draw Watts; 0 until first heartbeat */
    uint64_t gpu_vram_free_bytes;   /* gpu_load[0] live free VRAM; 0 until first
                                     * heartbeat. Crossed the wire since netd v1089
                                     * but was never cached — dataflux feeder VRAM
                                     * used/free was structurally blind (2026-07-13). */
    /* U4: unified health state machine */
    gb_feeder_health_state_t health_state;
    uint64_t health_state_since_ms;  /* mono_ms() when current state was entered */
    uint64_t disabled_until_ms;      /* for DISABLED → HEALTHY auto-retry */
    /* U13: PID kernel rate limiter */
    gb_pid_state_t pid;
    /* A1: per-feeder mutex - protects network I/O to this feeder only.
     * Allows parallel operations to different feeders without global lock. */
    pthread_mutex_t per_lock;
    /* A3: heartbeat tracking */
    uint64_t last_heartbeat_ms;   /* mono_ms() of last successful heartbeat response */
    uint32_t heartbeat_miss_count; /* consecutive failed heartbeat polls */
    /* U20: EWMA of measured transfer bandwidth (MB/s) - updated on every memcpy */
    float    bw_ewma_mbs;
    /* F-L3-09: per-connection sequence counters for within-session replay detection */
    uint32_t send_seq;   /* next seq_num to embed in outgoing header        */
    uint32_t recv_seq;   /* next seq_num expected from the feeder response   */
    /* PR-FF: per-message MAC state.  See struct client in netd.c. */
    int      mac_enabled;
    uint8_t  mac_session_key[32];
    /* F-L1-31: set when a kernel is dispatched to this feeder; cleared by
     * gb_netc_selective_stream_sync() so cudaStreamSynchronize only contacts
     * feeders that actually have pending work. */
    _Atomic int kernel_dispatched;
    /* Phase 2b: count of async kernels queued since last stream sync.
     * Non-zero means the feeder has unfinished async work in its async_stream. */
    _Atomic int pending_async_kernels;
    /* N8: exponential backoff for feeder reconnection after socket drop.
     * reconnect_delay_ms doubles on each failure (500ms → 30s).
     * next_reconnect_ms is the mono_ms() at which next attempt is allowed. */
    uint64_t reconnect_delay_ms;
    uint64_t next_reconnect_ms;
    /* Fabric zstd: negotiated at handshake (both sides advertise GB_NET_FEAT_
     * ZSTD). zbuf is a per-feeder compression scratch reused across H2D sends
     * (guarded by per_lock, which is held for the whole send). */
    int      feat_zstd;
    uint8_t *zbuf;
    size_t   zbuf_cap;
#ifdef GREENBOOST_USE_NCCL
    ncclComm_t nccl_comm;
#endif
};

struct netc_remote_gpu {
    int feeder_idx;        /* index into g_feeders[]           */
    int feeder_gpu_id;     /* GPU ID on the feeder             */
};

struct netc_alloc {
    uint64_t    fake_ptr;      /* local fake pointer (key)                         */
    uint64_t    remote_handle; /* actual device ptr on feeder                      */
    uint64_t    size;
    int         feeder_idx;
    int         in_use;
    _Atomic int ref_count;     /* F-L3-12: inflight exec refs; free waits for == 0 */
};

struct netc_kernel {
    const void *host_func;
    char        name[GB_NET_MAX_KERNEL_NAME];
    int         in_use;
};

/* ------------------------------------------------------------------ */
/*  Global state                                                       */
/* ------------------------------------------------------------------ */

/* U15: Pinned TCP staging buffer pool (Torch distributed pattern).
 * Preallocated mlock'd host buffers reused across gb_netc_memcpy_d2h calls.
 * mlock prevents page faults during network receive into staging area. */
#define GB_PINNED_BUF_COUNT   8
#define GB_PINNED_BUF_SIZE    (64UL * 1024 * 1024)  /* 64 MB each */

typedef struct {
    void        *ptr;       /* mlock'd buffer pointer (NULL if uninit) */
    _Atomic int  in_use;    /* 0 = free, 1 = in use                   */
} gb_pinned_buf_t;

static gb_pinned_buf_t g_pinned_bufs[GB_PINNED_BUF_COUNT];
static int             g_pinned_pool_ready = 0;

/* Acquire a free pinned buffer; returns NULL if all are busy → caller falls back to malloc. */
static void *gb_pinned_acquire(void)
{
    if (!g_pinned_pool_ready) return NULL;
    for (int i = 0; i < GB_PINNED_BUF_COUNT; i++) {
        if (!g_pinned_bufs[i].ptr) continue;
        int expected = 0;
        if (atomic_compare_exchange_strong(&g_pinned_bufs[i].in_use, &expected, 1))
            return g_pinned_bufs[i].ptr;
    }
    return NULL;
}

/* Release a buffer back to the pool. */
static void gb_pinned_release(void *ptr)
{
    for (int i = 0; i < GB_PINNED_BUF_COUNT; i++) {
        if (g_pinned_bufs[i].ptr == ptr) {
            atomic_store(&g_pinned_bufs[i].in_use, 0);
            return;
        }
    }
}

/* Count free pinned buffers (for metrics). */
static int gb_pinned_free_count(void)
{
    int n = 0;
    for (int i = 0; i < GB_PINNED_BUF_COUNT; i++)
        if (g_pinned_bufs[i].ptr && !atomic_load(&g_pinned_bufs[i].in_use)) n++;
    return n;
}

/* A1: global lock - protects init state, feeder/remote_gpu tables, kernel table,
 * pinned pool. Does NOT protect per-feeder network I/O (nf->per_lock does that). */
static pthread_mutex_t g_netc_lock = PTHREAD_MUTEX_INITIALIZER;
static int             g_netc_initialized = 0;

/* A1: separate alloc table lock - protects g_allocs[] and g_next_fake_ptr.
 * Decoupled from g_netc_lock so alloc table ops don't block during network I/O. */
static pthread_mutex_t g_alloc_lock = PTHREAD_MUTEX_INITIALIZER;

/* A2: generation counter incremented each time fake_ptr wraps around the range. */
static _Atomic uint32_t g_fake_ptr_generation = 0;

static struct netc_feeder     g_feeders[NETC_MAX_FEEDERS];
static int                    g_feeder_count = 0;

static struct netc_remote_gpu g_remote_gpus[NETC_MAX_REMOTE_GPUS];
static int                    g_remote_gpu_count = 0;

static struct netc_alloc      g_allocs[NETC_MAX_ALLOCS];
static uint64_t               g_next_fake_ptr = GB_REMOTE_PTR_BASE;

static struct netc_kernel     g_kernels[NETC_MAX_KERNELS];

static __thread int           g_active_remote = -1;  /* per-thread */

static int g_netc_debug = 0;

#define netc_log(fmt, ...) do { \
    if (g_netc_debug) \
        fprintf(stderr, "[GreenBoost-netc %d] " fmt "\n", (int)getpid(), ##__VA_ARGS__); \
} while (0)

/* ------------------------------------------------------------------ */
/*  Network I/O helpers                                                */
/* ------------------------------------------------------------------ */

static ssize_t netc_send_all(int fd, const void *buf, size_t len)
{
    const uint8_t *p = (const uint8_t *)buf;
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, p + sent, len - sent, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                /* Audit F-L3-08 mirror: poll for writability instead of spin. */
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

static ssize_t netc_recv_all(int fd, void *buf, size_t len)
{
    uint8_t *p = (uint8_t *)buf;
    size_t got = 0;
    while (got < len) {
        ssize_t n = recv(fd, p + got, len - got, 0);
        if (n < 0) {
            /* Audit F-L3-15: handle EINTR (and EAGAIN if the socket ever runs
             * non-blocking) the same way the daemon does. */
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                struct pollfd pf = { .fd = fd, .events = POLLIN };
                int pr = poll(&pf, 1, 200);
                if (pr < 0 && errno != EINTR) return -1;
                continue;
            }
            return -1;
        }
        if (n == 0) return -1; /* connection closed */
        got += (size_t)n;
    }
    return (ssize_t)got;
}

/* PR-FF: compute 8-byte truncated HMAC-SHA256 over (hdr || payload).
 * Mirror of netd.c gb_msg_mac.  When the connection is in v4 mode, this
 * value is appended after the payload on the wire. */
static void gb_msg_mac_netc(const uint8_t key[32],
                             const struct gb_net_header *hdr,
                             const void *payload, uint32_t payload_len,
                             uint8_t out_mac8[8])
{
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

static int netc_send_msg(struct netc_feeder *nf, uint16_t msg_type, uint16_t flags,
                         const void *payload, uint32_t payload_len)
{
    struct gb_net_header hdr = {
        .magic       = GB_NET_MAGIC,
        .msg_type    = msg_type,
        .flags       = flags,
        .payload_len = payload_len,
        .seq_num     = nf->send_seq++,
    };
    if (netc_send_all(nf->fd, &hdr, GB_NET_HDR_SIZE) < 0) return -1;
    if (payload_len > 0 && payload)
        if (netc_send_all(nf->fd, payload, payload_len) < 0) return -1;
    /* PR-FF: v4 - append 8-byte MAC after payload. */
    if (nf->mac_enabled) {
        uint8_t mac[8];
        gb_msg_mac_netc(nf->mac_session_key, &hdr, payload, payload_len, mac);
        int rc = netc_send_all(nf->fd, mac, sizeof(mac)) < 0 ? -1 : 0;
        explicit_bzero(mac, sizeof(mac));
        if (rc < 0) return -1;
    }
    return 0;
}

/* Send message + inline data blob (avoids a memcpy into a single buffer) */
static int netc_send_msg_with_data(struct netc_feeder *nf, uint16_t msg_type, uint16_t flags,
                                   const void *header_payload, uint32_t header_len,
                                   const void *data, uint32_t data_len)
{
    struct gb_net_header hdr = {
        .magic       = GB_NET_MAGIC,
        .msg_type    = msg_type,
        .flags       = flags,
        .payload_len = header_len + data_len,
        .seq_num     = nf->send_seq++,
    };
    if (netc_send_all(nf->fd, &hdr, GB_NET_HDR_SIZE) < 0) return -1;
    if (header_len > 0)
        if (netc_send_all(nf->fd, header_payload, header_len) < 0) return -1;
    if (data_len > 0)
        if (netc_send_all(nf->fd, data, data_len) < 0) return -1;
    /* PR-FF: MAC covers (hdr || header_payload || data) - same logical
     * concatenation the receiver sees as the framed payload. */
    if (nf->mac_enabled) {
        uint8_t mac[8];
        /* Two-segment HMAC: feed both into the same SHA-256 context. */
        uint8_t k_ipad[GB_SHA256_BLOCK], k_opad[GB_SHA256_BLOCK];
        for (int i = 0; i < GB_SHA256_BLOCK; i++) {
            uint8_t kb = (i < 32) ? nf->mac_session_key[i] : 0;
            k_ipad[i] = kb ^ 0x36;
            k_opad[i] = kb ^ 0x5c;
        }
        uint8_t inner[32], full[32];
        gb_sha256_ctx c;
        gb_sha256_init(&c);
        gb_sha256_update(&c, k_ipad, GB_SHA256_BLOCK);
        gb_sha256_update(&c, (const uint8_t *)&hdr, GB_NET_HDR_SIZE);
        if (header_len > 0) gb_sha256_update(&c, header_payload, header_len);
        if (data_len > 0)   gb_sha256_update(&c, data, data_len);
        gb_sha256_final(&c, inner);
        gb_sha256_init(&c);
        gb_sha256_update(&c, k_opad, GB_SHA256_BLOCK);
        gb_sha256_update(&c, inner, 32);
        gb_sha256_final(&c, full);
        memcpy(mac, full, 8);
        explicit_bzero(k_ipad, sizeof(k_ipad));
        explicit_bzero(k_opad, sizeof(k_opad));
        explicit_bzero(inner, sizeof(inner));
        explicit_bzero(full, sizeof(full));
        int rc = netc_send_all(nf->fd, mac, sizeof(mac)) < 0 ? -1 : 0;
        explicit_bzero(mac, sizeof(mac));
        if (rc < 0) return -1;
    }
    return 0;
}

/* Receive response: header + payload (caller must free *out_payload).
 * F-L3-09: checks seq_num monotonicity; closes connection on mismatch. */
static int netc_recv_response(struct netc_feeder *nf, struct gb_net_header *out_hdr,
                              void **out_payload)
{
    if (netc_recv_all(nf->fd, out_hdr, GB_NET_HDR_SIZE) < 0) return -1;
    /* PR-G/H11: wire fields are little-endian.  netd already does le32toh
     * on every header read; netc was direct-comparing the on-wire bytes,
     * which works only on LE hosts.  Mirror the netd pattern here so a
     * future big-endian or weak-LE build (ARM64 with mismatched flags,
     * MIPS, PowerPC) does not silently desync framing. */
    uint32_t hdr_magic       = le32toh(out_hdr->magic);
    uint32_t hdr_seq         = le32toh(out_hdr->seq_num);
    uint32_t hdr_payload_len = le32toh(out_hdr->payload_len);
    out_hdr->magic       = hdr_magic;
    out_hdr->seq_num     = hdr_seq;
    out_hdr->payload_len = hdr_payload_len;
    if (hdr_magic != GB_NET_MAGIC) return -1;
    if (hdr_seq != nf->recv_seq) {
        netc_log("ERROR: seq mismatch from feeder %s (expected %u, got %u) - closing",
                 nf->addr, nf->recv_seq, hdr_seq);
        return -1;
    }
    nf->recv_seq++;
    *out_payload = NULL;
    if (hdr_payload_len > 0) {
        if (hdr_payload_len > NETC_RECV_BUF) return -1;
        *out_payload = malloc(hdr_payload_len);
        if (!*out_payload) return -1;
        if (netc_recv_all(nf->fd, *out_payload, hdr_payload_len) < 0) {
            free(*out_payload);
            *out_payload = NULL;
            return -1;
        }
    }
    /* PR-FF: read + verify 8-byte trailing MAC in v4 mode. */
    if (nf->mac_enabled) {
        uint8_t recv_mac[8], expected_mac[8];
        if (netc_recv_all(nf->fd, recv_mac, 8) != 8) {
            if (*out_payload) { free(*out_payload); *out_payload = NULL; }
            return -1;
        }
        gb_msg_mac_netc(nf->mac_session_key, out_hdr,
                        *out_payload, hdr_payload_len, expected_mac);
        /* Constant-time compare */
        uint8_t diff = 0;
        for (int i = 0; i < 8; i++) diff |= recv_mac[i] ^ expected_mac[i];
        explicit_bzero(expected_mac, sizeof(expected_mac));
        if (diff != 0) {
            netc_log("ERROR: MAC verification failed from feeder %s - closing", nf->addr);
            if (*out_payload) { free(*out_payload); *out_payload = NULL; }
            return -1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Feeder connection                                                  */
/* ------------------------------------------------------------------ */

static int connect_feeder(struct netc_feeder *f)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct timeval tv = { .tv_sec = 5, .tv_usec = 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    int nodelay = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    /* Explicit socket buffers sized to 2x GB_NET_MAX_MSG_SIZE so a couple of
     * H2D/D2H chunks (weight upload, per-layer activation staging) can be
     * in flight without waiting on the kernel's slow auto-tuning ramp-up on
     * a fresh/idle connection - matters most for the small, frequent
     * cross-device activation transfers the 2-device split adds. */
    int sockbuf = 2 * (int)GB_NET_MAX_MSG_SIZE;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sockbuf, sizeof(sockbuf));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sockbuf, sizeof(sockbuf));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)f->port);
    if (inet_pton(AF_INET, f->addr, &addr.sin_addr) <= 0) {
        close(fd);
        return -1;
    }

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        netc_log("connect to %s:%d failed: %s", f->addr, f->port, strerror(errno));
        close(fd);
        return -1;
    }

    /* F-L3-01: PSK authentication - must complete before protocol handshake.
     * The socket is blocking here (SO_RCVTIMEO/SO_SNDTIMEO set above).
     * If cluster.key is absent on the client side we skip auth; the server
     * will also skip (backward-compat mode) and log a warning.            */
    {
        /* PR-EE: opt-in mutual-auth + HKDF session key.
         * Set GREENBOOST_PSK_V4=1 on BOTH sides to enable. */
        static int gb_psk_v4 = -1;
        if (gb_psk_v4 < 0) {
            const char *e = getenv("GREENBOOST_PSK_V4");
            gb_psk_v4 = (e && strcmp(e, "1") == 0) ? 1 : 0;
            if (gb_psk_v4)
                netc_log("PSK: v4 mode enabled (mutual auth + HKDF session key)");
        }

        uint8_t psk[32];
        if (gb_load_psk(psk) == 0) {
            uint8_t nonce_s[32], nonce_c[32], session_key[32];
            memset(nonce_c, 0, sizeof(nonce_c));
            memset(session_key, 0, sizeof(session_key));
            int recv_ok = (netc_recv_all(fd, nonce_s, sizeof(nonce_s)) == (ssize_t)sizeof(nonce_s));
            int send_ok = 0;
            int mac2_bad = 0;
            if (recv_ok) {
                if (gb_psk_v4) {
                    /* Generate nonce_c via /dev/urandom (libc-portable) */
                    FILE *r = fopen("/dev/urandom", "rb");
                    if (r) { (void)!fread(nonce_c, 1, 32, r); fclose(r); }
                    /* mac1 = HMAC(psk, nonce_s || nonce_c) */
                    uint8_t challenge[64];
                    memcpy(challenge,      nonce_s, 32);
                    memcpy(challenge + 32, nonce_c, 32);
                    uint8_t mac1[32];
                    gb_hmac_sha256_salt(psk, 32, challenge, 64, mac1);
                    /* Send nonce_c || mac1 (64 bytes) */
                    uint8_t reply[64];
                    memcpy(reply,      nonce_c, 32);
                    memcpy(reply + 32, mac1,    32);
                    send_ok = (netc_send_all(fd, reply, 64) == 64);
                    explicit_bzero(challenge, sizeof(challenge));
                    explicit_bzero(mac1, sizeof(mac1));
                    explicit_bzero(reply, sizeof(reply));
                    if (send_ok) {
                        /* Verify mac2 from server: mac2 = HMAC(psk, nonce_c || nonce_s) */
                        uint8_t mac2_recv[32];
                        if (netc_recv_all(fd, mac2_recv, 32) != 32) {
                            mac2_bad = 1;
                        } else {
                            uint8_t mac2_in[64];
                            memcpy(mac2_in,      nonce_c, 32);
                            memcpy(mac2_in + 32, nonce_s, 32);
                            uint8_t mac2_expected[32];
                            gb_hmac_sha256_salt(psk, 32, mac2_in, 64, mac2_expected);
                            /* Constant-time compare */
                            uint8_t diff = 0;
                            for (int _i = 0; _i < 32; _i++)
                                diff |= mac2_recv[_i] ^ mac2_expected[_i];
                            mac2_bad = (diff != 0);
                            explicit_bzero(mac2_in, sizeof(mac2_in));
                            explicit_bzero(mac2_expected, sizeof(mac2_expected));
                        }
                        explicit_bzero(mac2_recv, sizeof(mac2_recv));
                        if (!mac2_bad) {
                            gb_derive_session_key(psk, nonce_s, nonce_c, session_key);
                            /* PR-FF: install session key + enable per-message
                             * MAC on this feeder.  After this point every
                             * netc_send_msg / netc_recv_response on this fd
                             * appends/verifies an 8-byte MAC. */
                            memcpy(f->mac_session_key, session_key, 32);
                            f->mac_enabled = 1;
                        }
                    }
                } else {
                    /* v3 legacy: mac = HMAC(psk, nonce_s) */
                    uint8_t mac[32];
                    gb_hmac_sha256(psk, nonce_s, sizeof(nonce_s), mac);
                    send_ok = (netc_send_all(fd, mac, sizeof(mac)) == (ssize_t)sizeof(mac));
                    explicit_bzero(mac, sizeof(mac));
                }
            }
            /* PR-CC + PR-EE: wipe ALL key-derived material from stack. */
            explicit_bzero(psk, sizeof(psk));
            explicit_bzero(nonce_s, sizeof(nonce_s));
            explicit_bzero(nonce_c, sizeof(nonce_c));
            explicit_bzero(session_key, sizeof(session_key));
            if (!recv_ok) {
                netc_log("PSK: failed to recv nonce from %s:%d", f->addr, f->port);
                close(fd);
                return -1;
            }
            if (!send_ok) {
                netc_log("PSK: failed to send %s to %s:%d",
                         gb_psk_v4 ? "nonce_c+mac1" : "MAC", f->addr, f->port);
                close(fd);
                return -1;
            }
            if (gb_psk_v4 && mac2_bad) {
                netc_log("PSK: v4 server mac2 verification FAILED for %s:%d "
                         "(server may not have GREENBOOST_PSK_V4=1)",
                         f->addr, f->port);
                close(fd);
                return -1;
            }
            netc_log("PSK: authenticated to feeder %s:%d (v%d)",
                     f->addr, f->port, gb_psk_v4 ? 4 : 3);
        }
        /* PR-CC: stale "compat mode" - server is fail-closed since PR-C/C7. */
    }

    /* Assign fd and reset seq counters before any protocol message (F-L3-09) */
    f->fd       = fd;
    f->send_seq = 0;
    f->recv_seq = 0;

    /* Handshake */
    struct gb_net_handshake_req req;
    memset(&req, 0, sizeof(req));
    req.proto_version = GB_NET_PROTO_VER;
    req.gpu_count     = 0;
    gethostname(req.hostname, GB_NET_MAX_HOSTNAME - 1);
#ifdef GB_HAVE_ZSTD
    gb_netc_zstd_init();
    if (g_zstd_enabled) req.feature_flags |= GB_NET_FEAT_ZSTD;
#endif

    if (netc_send_msg(f, GB_MSG_HANDSHAKE_REQ, 0, &req, sizeof(req)) < 0) {
        f->fd = -1;
        close(fd);
        return -1;
    }

    struct gb_net_header resp_hdr;
    void *resp_payload = NULL;
    if (netc_recv_response(f, &resp_hdr, &resp_payload) < 0 || !resp_payload) {
        f->fd = -1;
        close(fd);
        return -1;
    }

    if (resp_hdr.msg_type != GB_MSG_HANDSHAKE_RESP) {
        free(resp_payload);
        close(fd);
        return -1;
    }

    const struct gb_net_handshake_resp *resp =
        (const struct gb_net_handshake_resp *)resp_payload;

    if (resp->status != GB_STATUS_OK) {
        free(resp_payload);
        close(fd);
        return -1;
    }

    /* connected is set LAST, after all initial sends complete (MEM_INFO at line ~992).
     * Any caller that checks nf->connected before sending (e.g. gb_netc_mem_info)
     * must not see connected=1 until connect_feeder's own unprotected sends are done,
     * otherwise they race on send_seq with connect_feeder (no per_lock held here). */
    f->feeder_id  = resp->feeder_id;
    f->gpu_count  = (int)resp->gpu_count;
    if (f->gpu_count > GB_NET_MAX_GPUS) f->gpu_count = GB_NET_MAX_GPUS;
    strncpy(f->hostname, resp->hostname, GB_NET_MAX_HOSTNAME - 1);
    memcpy(f->gpus, resp->gpus, sizeof(struct gb_net_gpu_info) * f->gpu_count);

    /* D1: Parse PCIe link info if feeder sent the v2 extended handshake */
    {
        size_t base_sz = offsetof(struct gb_net_handshake_resp, pcie_link_gen);
        if (resp_hdr.payload_len >= (uint32_t)(base_sz + sizeof(gb_u32) * 4)) {
            f->pcie_link_gen         = resp->pcie_link_gen;
            f->pcie_link_width       = resp->pcie_link_width;
            f->pcie_effective_bw_mbs = resp->pcie_effective_bw_mbs;
            netc_log("feeder %s: PCIe Gen%u×%u eff=%u MB/s replay=%u",
                     f->addr, f->pcie_link_gen, f->pcie_link_width,
                     f->pcie_effective_bw_mbs, resp->pcie_replay_count);
        }
    }

    /* Fabric zstd: enabled only when we advertised it AND the feeder echoed
     * GB_NET_FEAT_ZSTD in its (long-enough) reply. Older feeders send a short
     * resp with no feature_flags → feat_zstd stays 0 → we send raw. */
    f->feat_zstd = 0;
#ifdef GB_HAVE_ZSTD
    {
        size_t need = offsetof(struct gb_net_handshake_resp, feature_flags)
                      + sizeof(gb_u32);
        if (g_zstd_enabled && resp_hdr.payload_len >= (uint32_t)need &&
            (resp->feature_flags & GB_NET_FEAT_ZSTD)) {
            f->feat_zstd = 1;
            netc_log("feeder %s: fabric zstd compression negotiated (level %d, min %u B)",
                     f->addr, g_zstd_level, g_zstd_min);
        }
    }
#endif

#ifdef GREENBOOST_USE_NCCL
    /* Phase 4: Init NCCL for this connection */
    /* Destroy the previous communicator on reconnect to avoid a leak */
    if (f->nccl_comm) {
        ncclCommDestroy(f->nccl_comm);
        f->nccl_comm = NULL;
    }
    struct gb_net_nccl_init nccl_req;
    memset(&nccl_req, 0, sizeof(nccl_req));
    nccl_req.rank = 1;      /* feeder rank; host uses rank 0 below */
    nccl_req.num_ranks = 2;
    ncclGetUniqueId((ncclUniqueId *)nccl_req.nccl_id);

    if (netc_send_msg(f, GB_MSG_NCCL_INIT, 0, &nccl_req, sizeof(nccl_req)) >= 0) {
        struct gb_net_header nccl_hdr;
        void *nccl_payload = NULL;
        if (netc_recv_response(f, &nccl_hdr, &nccl_payload) >= 0 && nccl_payload) {
            const struct gb_net_response *nresp = nccl_payload;
            if (nresp->status == GB_STATUS_OK) {
                ncclResult_t nr = ncclCommInitRank(&f->nccl_comm, 2, *(ncclUniqueId *)nccl_req.nccl_id, 0);
                if (nr == ncclSuccess) {
                    netc_log("NCCL communicator established with %s:%d", f->addr, f->port);
                } else {
                    netc_log("WARN: Host ncclCommInitRank failed: %d", nr);
                }
            } else {
                netc_log("WARN: Feeder rejected NCCL init");
            }
            free(nccl_payload);
        }
    }
#endif

    /* Query MEM_INFO for GPU 0 to cache DDR and NVMe speeds - avoids live queries
     * on every allocation call, eliminating the serial g_netc_lock bottleneck. */
    {
        struct gb_net_mem_info mreq = { .device_id = 0 };
        if (netc_send_msg(f, GB_MSG_MEM_INFO, 0, &mreq, sizeof(mreq)) >= 0) {
            struct gb_net_header mhdr;
            void *mpay = NULL;
            if (netc_recv_response(f, &mhdr, &mpay) >= 0 && mpay) {
                const struct gb_net_mem_info_resp *mr =
                    (const struct gb_net_mem_info_resp *)mpay;
                if (mhdr.payload_len >= (uint32_t)offsetof(struct gb_net_mem_info_resp, t2_speed_mts) + 4)
                    f->t2_speed_mts = mr->t2_speed_mts;
                if (mhdr.payload_len >= sizeof(struct gb_net_mem_info_resp))
                    f->t3_speed_mbs = mr->t3_speed_mbs;
                free(mpay);
            }
        }
        if (f->t2_speed_mts == 0) f->t2_speed_mts = 2400;  /* fallback */
        if (f->t3_speed_mbs == 0) f->t3_speed_mbs = 500;   /* fallback */
        netc_log("feeder %s: t2=%u MT/s t3=%u MB/s", f->addr, f->t2_speed_mts, f->t3_speed_mbs);
    }

    /* Increase socket timeouts for data transfers (Phase 2) */
    tv.tv_sec = 30;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    /* TCP keepalive */
    int keepalive = 1;
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &keepalive, sizeof(keepalive));

    /* U13: initialise PID state - unlimited until first heartbeat provides SM util */
    f->pid.setpoint_pct    = 80.0;
    f->pid.kp              = 0.5;
    f->pid.ki              = 0.1;
    f->pid.kd              = 0.05;
    f->pid.integral        = 0.0;
    f->pid.prev_error      = 0.0;
    f->pid.target_rate     = 0;   /* 0 = unlimited */
    f->pid.dispatch_count  = 0;
    f->pid.window_start_ms = 0;

    /* A1: per_lock is initialized ONCE in gb_netc_init when the feeder slot
     * is created , re-initializing it here on every N8 reconnect while
     * another thread holds it is undefined behaviour. */

    /* A3: initialise heartbeat tracking */
    {
        struct timespec _ts;
        clock_gettime(CLOCK_MONOTONIC, &_ts);
        f->last_heartbeat_ms   = (uint64_t)_ts.tv_sec * 1000 + (uint64_t)_ts.tv_nsec / 1000000;
        f->heartbeat_miss_count = 0;
    }

    /* U20: initialise EWMA bandwidth to handshake-provided PCIe BW as seed */
    f->bw_ewma_mbs = (f->pcie_effective_bw_mbs > 0)
                     ? (float)f->pcie_effective_bw_mbs : 0.0f;

    free(resp_payload);
    /* Now that all initial sends (HANDSHAKE + MEM_INFO) are done, mark connected.
     * Setting this earlier (before the MEM_INFO send) creates a window where a
     * concurrent CUDA thread calling gb_netc_mem_info can see connected=1 and send
     * without per_lock, racing with connect_feeder's own unprotected send and
     * producing a duplicate seq_num on the feeder side. */
    f->connected = 1;
    return 0;
}

/* U13: advance PID controller for one feeder; called from gb_netc_poll_health().
 * Uses SM utilization from the most recent heartbeat.
 * Must be called with g_netc_lock held. */
static void gb_netc_pid_tick(int fi)
{
    struct netc_feeder *f = &g_feeders[fi];
    if (!f->connected || f->gpu_count == 0) return;

    double actual = (double)f->gpu_util_pct;  /* 0-100 from last heartbeat */
    if (actual > 100.0) actual = 100.0;

    double setpoint = f->pid.setpoint_pct;
    /* Degrade setpoint on unhealthy states */
    if (f->health_state >= GB_HEALTH_DEGRADED)
        setpoint = 40.0;

    double error      = setpoint - actual;
    f->pid.integral   = f->pid.integral * 0.9 + error; /* leaky integrator */
    double derivative = error - f->pid.prev_error;
    f->pid.prev_error = error;

    double output   = f->pid.kp * error + f->pid.ki * f->pid.integral + f->pid.kd * derivative;
    int new_rate    = (int)(100.0 + output * 2.0);
    if (new_rate < 10)  new_rate = 10;
    if (new_rate > 500) new_rate = 500;
    f->pid.target_rate = (uint32_t)new_rate;

    /* Reset per-window counter */
    struct timespec _ts;
    clock_gettime(CLOCK_MONOTONIC, &_ts);
    f->pid.dispatch_count  = 0;
    f->pid.window_start_ms = (uint64_t)_ts.tv_sec * 1000 + (uint64_t)_ts.tv_nsec / 1000000;
}

/* ------------------------------------------------------------------ */
/*  Allocation tracking                                                */
/* ------------------------------------------------------------------ */

/* Must be called with g_alloc_lock held. */
static struct netc_alloc *alloc_find(uint64_t fake_ptr)
{
    for (int i = 0; i < NETC_MAX_ALLOCS; i++)
        if (g_allocs[i].in_use && g_allocs[i].fake_ptr == fake_ptr)
            return &g_allocs[i];
    return NULL;
}

/* Defined below; forward-declared for gb_netc_memset (interior offsets). */
static struct netc_alloc *alloc_find_range(uint64_t ptr);

/* Must be called with g_alloc_lock held. */
static struct netc_alloc *alloc_new(void)
{
    for (int i = 0; i < NETC_MAX_ALLOCS; i++)
        if (!g_allocs[i].in_use)
            return &g_allocs[i];
    return NULL;
}

/* A2: Wraparound-safe fake pointer bump.
 * Must be called with g_alloc_lock held.
 * Returns a fake pointer aligned to 4 KB that does not collide with any in-use entry.
 * Returns 0 if no valid pointer can be found (extremely unlikely with 131072 slots). */
static uint64_t gb_alloc_bump_fake_ptr(size_t size)
{
    uint64_t aligned = (uint64_t)((size + 0xFFFULL) & ~0xFFFULL);
    uint64_t range_end = GB_REMOTE_PTR_BASE + GB_FAKE_PTR_RANGE;

    for (int tries = 0; tries < NETC_MAX_ALLOCS; tries++) {
        if (g_next_fake_ptr + aligned > range_end) {
            /* Wraparound: reset to base and bump generation counter */
            atomic_fetch_add_explicit(&g_fake_ptr_generation, 1u, memory_order_relaxed);
            g_next_fake_ptr = GB_REMOTE_PTR_BASE;
        }

        uint64_t candidate = g_next_fake_ptr;
        g_next_fake_ptr += aligned;

        /* Verify no in-use allocation overlaps this candidate range */
        int collision = 0;
        for (int i = 0; i < NETC_MAX_ALLOCS; i++) {
            if (!g_allocs[i].in_use) continue;
            uint64_t a_start = g_allocs[i].fake_ptr;
            uint64_t a_end   = a_start + g_allocs[i].size;
            if (candidate < a_end && candidate + aligned > a_start) {
                collision = 1;
                break;
            }
        }
        if (!collision) return candidate;
    }
    return 0; /* exhausted - caller will return OOM */
}

/* ------------------------------------------------------------------ */
/*  Public API: Lifecycle                                              */
/* ------------------------------------------------------------------ */

/* ── Dedicated heartbeat/reconnect thread (2026-07-06) ───────────────────
 * gb_netc_poll_health() used to run only from the shim's stats hook, i.e.
 * only while the application was making CUDA calls.  Two live failures:
 * a 20 s model-load pause stopped heartbeats → the feeder daemon dropped
 * the client at GB_NET_HEARTBEAT_TIMEOUT_MS; and once disconnected, the N8
 * reconnect never ran while the process idled , the feeder stayed lost for
 * the life of the process.  A dedicated thread makes liveness and healing
 * independent of application activity. */
static pthread_t g_hb_thread;
static _Atomic int g_hb_thread_started = 0;
static _Atomic int g_hb_thread_stop    = 0;

static void *gb_netc_hb_thread_main(void *arg)
{
    (void)arg;
    while (!atomic_load_explicit(&g_hb_thread_stop, memory_order_relaxed)) {
        gb_netc_poll_health();
        for (int i = 0; i < 20; i++) {   /* 2 s in 100 ms slices for fast exit */
            if (atomic_load_explicit(&g_hb_thread_stop, memory_order_relaxed)) break;
            struct timespec ts = { 0, 100 * 1000 * 1000 };
            nanosleep(&ts, NULL);
        }
    }
    return NULL;
}

static void gb_netc_start_heartbeat_thread(void)
{
    int expected = 0;
    if (!atomic_compare_exchange_strong(&g_hb_thread_started, &expected, 1))
        return;
    if (pthread_create(&g_hb_thread, NULL, gb_netc_hb_thread_main, NULL) != 0) {
        atomic_store(&g_hb_thread_started, 0);
        netc_log("WARN: heartbeat thread create failed - falling back to hook-driven polling");
    } else {
        pthread_detach(g_hb_thread);
    }
}

int gb_netc_init(void)
{
    pthread_mutex_lock(&g_netc_lock);
    if (g_netc_initialized) {
        pthread_mutex_unlock(&g_netc_lock);
        return 0;
    }

    const char *dbg = getenv("GREENBOOST_NET_DEBUG");
    if (dbg && dbg[0] != '0') g_netc_debug = 1;

    /* GREENBOOST_CLUSTER=0 - per-process opt-out of the network fabric.
     * Feeder kernel dispatch requires the kernel symbol to be dlsym-resolvable
     * in greenboost-netd and on the kernels.allow allowlist; that holds for
     * ggml/Ollama but NOT for PyTorch (static stubs, thousands of kernels).
     * PyTorch consumers (ai-forge diffusion) set this to stay on the proven
     * local T1/T2/T3 path instead of crashing on feeder-resident tensors. */
    const char *clu = getenv("GREENBOOST_CLUSTER");
    if (clu && clu[0] == '0') {
        netc_log("GREENBOOST_CLUSTER=0 - network fabric disabled for this process");
        g_netc_initialized = 1;
        pthread_mutex_unlock(&g_netc_lock);
        return 0;
    }

    memset(g_feeders, 0, sizeof(g_feeders));
    memset(g_remote_gpus, 0, sizeof(g_remote_gpus));
    memset(g_allocs, 0, sizeof(g_allocs));
    memset(g_kernels, 0, sizeof(g_kernels));
    g_feeder_count     = 0;
    g_remote_gpu_count = 0;
    g_next_fake_ptr    = GB_REMOTE_PTR_BASE;

    /* Read cluster.conf */
    FILE *f = fopen(GB_CLUSTER_CONF, "r");
    if (!f) {
        netc_log("No cluster.conf - network fabric inactive");
        g_netc_initialized = 1;
        pthread_mutex_unlock(&g_netc_lock);
        return 0;
    }

    char line[256];
    while (fgets(line, sizeof(line), f) && g_feeder_count < NETC_MAX_FEEDERS) {
        /* Skip comments and blank lines */
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\0') continue;

        char addr_port[128] = {0};
        sscanf(p, "%127s", addr_port);
        if (!addr_port[0]) continue;

        struct netc_feeder *nf = &g_feeders[g_feeder_count];
        memset(nf, 0, sizeof(*nf));
        nf->fd = -1;
        pthread_mutex_init(&nf->per_lock, NULL);

        char *colon = strchr(addr_port, ':');
        if (colon) {
            *colon = '\0';
            strncpy(nf->addr, addr_port, sizeof(nf->addr) - 1);
            nf->port = atoi(colon + 1);
        } else {
            strncpy(nf->addr, addr_port, sizeof(nf->addr) - 1);
            nf->port = GB_NET_PORT;
        }

        netc_log("Connecting to feeder %s:%d...", nf->addr, nf->port);

        if (connect_feeder(nf) == 0) {
            netc_log("Connected to %s (%s, %d GPUs)",
                     nf->addr, nf->hostname, nf->gpu_count);

            /* Register remote GPUs */
            for (int g = 0; g < nf->gpu_count && g_remote_gpu_count < NETC_MAX_REMOTE_GPUS; g++) {
                g_remote_gpus[g_remote_gpu_count].feeder_idx    = g_feeder_count;
                g_remote_gpus[g_remote_gpu_count].feeder_gpu_id = g;
                g_remote_gpu_count++;

                netc_log("  Remote GPU %d: %s (%llu MB)",
                         g_remote_gpu_count - 1,
                         nf->gpus[g].name,
                         (unsigned long long)(nf->gpus[g].vram_bytes >> 20));
            }
            g_feeder_count++;
        } else {
            netc_log("Failed to connect to %s:%d", nf->addr, nf->port);
        }
    }
    fclose(f);

    if (g_remote_gpu_count > 0) {
        fprintf(stderr, "[GreenBoost] Network fabric: %d remote GPU(s) from %d feeder(s)\n",
                g_remote_gpu_count, g_feeder_count);
        gb_netc_start_heartbeat_thread();
    }

    /* U15: allocate pinned staging buffer pool for D2H transfers */
    if (g_remote_gpu_count > 0 && !g_pinned_pool_ready) {
        int pinned_ok = 0;
        for (int i = 0; i < GB_PINNED_BUF_COUNT; i++) {
            void *p = malloc(GB_PINNED_BUF_SIZE);
            if (p) {
                /* mlock to prevent swap-out; best-effort (may fail without CAP_IPC_LOCK) */
                mlock(p, GB_PINNED_BUF_SIZE);
                g_pinned_bufs[i].ptr = p;
                atomic_store(&g_pinned_bufs[i].in_use, 0);
                pinned_ok++;
            }
        }
        if (pinned_ok > 0) {
            g_pinned_pool_ready = 1;
            netc_log("Pinned TCP pool: %d × %lu MB buffers ready",
                     pinned_ok, (unsigned long)(GB_PINNED_BUF_SIZE >> 20));
        }
    }

    g_netc_initialized = 1;
    pthread_mutex_unlock(&g_netc_lock);
    return 0;
}

void gb_netc_cleanup(void)
{
    atomic_store(&g_hb_thread_stop, 1);
    pthread_mutex_lock(&g_netc_lock);
    for (int i = 0; i < g_feeder_count; i++) {
        if (g_feeders[i].connected && g_feeders[i].fd >= 0) {
            netc_send_msg(&g_feeders[i], GB_MSG_DISCONNECT, 0, NULL, 0);
            close(g_feeders[i].fd);
            g_feeders[i].fd = -1;
            g_feeders[i].connected = 0;
        }
    }
    g_feeder_count     = 0;
    g_remote_gpu_count = 0;
    g_netc_initialized = 0;

    /* U15: release pinned staging buffers */
    if (g_pinned_pool_ready) {
        for (int i = 0; i < GB_PINNED_BUF_COUNT; i++) {
            if (g_pinned_bufs[i].ptr) {
                munlock(g_pinned_bufs[i].ptr, GB_PINNED_BUF_SIZE);
                free(g_pinned_bufs[i].ptr);
                g_pinned_bufs[i].ptr = NULL;
            }
        }
        g_pinned_pool_ready = 0;
    }

    pthread_mutex_unlock(&g_netc_lock);
}

int gb_netc_is_active(void)
{
    return g_netc_initialized && g_remote_gpu_count > 0;
}

/* ------------------------------------------------------------------ */
/*  Public API: Device queries                                         */
/* ------------------------------------------------------------------ */

int gb_netc_remote_gpu_count(void)
{
    return g_remote_gpu_count;
}

uint64_t gb_netc_remote_vram(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    int gi = g_remote_gpus[remote_idx].feeder_gpu_id;
    return g_feeders[fi].gpus[gi].vram_bytes;
}

int gb_netc_remote_cc_major(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    int gi = g_remote_gpus[remote_idx].feeder_gpu_id;
    return (int)g_feeders[fi].gpus[gi].cc_major;
}

int gb_netc_remote_cc_minor(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    int gi = g_remote_gpus[remote_idx].feeder_gpu_id;
    return (int)g_feeders[fi].gpus[gi].cc_minor;
}

const char *gb_netc_remote_name(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return "unknown";
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    int gi = g_remote_gpus[remote_idx].feeder_gpu_id;
    return g_feeders[fi].gpus[gi].name;
}

const char *gb_netc_feeder_addr(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return "unknown";
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    return g_feeders[fi].addr;   /* IP-only string (port is stored separately in nf->port) */
}

/* Reconnect-on-demand: an allocation path that finds the feeder disconnected
 * should not fail the alloc and wait for the 2 s heartbeat cycle , the
 * residual early-connection drop otherwise blanks the feeder exactly during
 * model-load alloc bursts (seen live: floor-refused local T2 + disconnected
 * feeder = spurious FULL OOM).  Bounded: one attempt per 300 ms per feeder. */
int gb_netc_ensure_connected(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return -1;
    struct netc_feeder *nf = &g_feeders[g_remote_gpus[remote_idx].feeder_idx];
    if (nf->connected && nf->fd >= 0) return 0;

    struct timespec _ts;
    clock_gettime(CLOCK_MONOTONIC, &_ts);
    uint64_t now_ms = (uint64_t)_ts.tv_sec * 1000 + (uint64_t)_ts.tv_nsec / 1000000;
    static _Atomic uint64_t last_attempt_ms = 0;
    uint64_t prev = atomic_load_explicit(&last_attempt_ms, memory_order_relaxed);
    if (now_ms - prev < 300) return nf->connected ? 0 : -1;
    if (!atomic_compare_exchange_strong(&last_attempt_ms, &prev, now_ms))
        return nf->connected ? 0 : -1;

    if (pthread_mutex_trylock(&nf->per_lock) != 0)
        return nf->connected ? 0 : -1;
    int rc = 0;
    if (!nf->connected || nf->fd < 0) {
        rc = connect_feeder(nf);
        if (rc == 0) {
            fprintf(stderr, "[GreenBoost-netc] on-demand reconnect to feeder %s\n", nf->addr);
            nf->reconnect_delay_ms = 500;
            nf->health_state = GB_HEALTH_HEALTHY;
        }
    }
    pthread_mutex_unlock(&nf->per_lock);
    return rc == 0 ? 0 : -1;
}

static int netc_mem_info_request(struct netc_feeder *nf, int gi,
                                 struct gb_net_header *out_hdr, void **out_payload);

/* Block until at least one feeder GPU is connected, or timeout_ms elapses.
 * Feeder-exclusive mode calls this before the FIRST allocation so the model's
 * weights don't lose a race with netc's async connect and silently fall to
 * local T2 (which then runs entirely local , omen idle). Returns 0 if a feeder
 * is ready, -1 on timeout. */
int gb_netc_wait_connected(int timeout_ms)
{
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;) {
        for (int i = 0; i < g_remote_gpu_count; i++) {
            struct netc_feeder *nf = &g_feeders[g_remote_gpus[i].feeder_idx];
            if (nf->connected && nf->fd >= 0) return 0;
            gb_netc_ensure_connected(i);
        }
        struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
        long ms = (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000;
        if (ms >= timeout_ms) return -1;
        struct timespec s = { 0, 50 * 1000 * 1000 };
        nanosleep(&s, NULL);
    }
}

int gb_netc_mem_info(int remote_idx, uint64_t *free_bytes, uint64_t *total_bytes)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return -1;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    int gi = g_remote_gpus[remote_idx].feeder_gpu_id;
    struct netc_feeder *nf = &g_feeders[fi];

    struct gb_net_header resp_hdr;
    void *resp_payload = NULL;
    if (netc_mem_info_request(nf, gi, &resp_hdr, &resp_payload) != 0) return -1;

    const struct gb_net_mem_info_resp *resp =
        (const struct gb_net_mem_info_resp *)resp_payload;
    if (resp->status != GB_STATUS_OK && resp->total_bytes == 0) {
        free(resp_payload); return -1;
    }

    *free_bytes  = resp->free_bytes;
    *total_bytes = resp->total_bytes;
    free(resp_payload);
    return 0;
}

int gb_netc_t2_speed_mts(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    struct netc_feeder *nf = &g_feeders[fi];
    if (!nf->connected) return 0;
    return (int)nf->t2_speed_mts;  /* cached at connect time */
}

int gb_netc_t3_speed_mbs(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    struct netc_feeder *nf = &g_feeders[fi];
    if (!nf->connected) return 0;
    return (int)nf->t3_speed_mbs;  /* cached at connect time */
}

/* MEM_INFO request with ONE inline reconnect retry.  A "zombie" socket
 * (connected=1 but peer already closed , the residual early-drop) otherwise
 * blanks every feeder tier exactly during a model-load alloc burst, because
 * the heartbeat thread needs up to ~2.5 s to notice and reconnect. */
static int netc_mem_info_request(struct netc_feeder *nf, int gi,
                                 struct gb_net_header *out_hdr, void **out_payload)
{
    struct gb_net_mem_info req = { .device_id = (gb_u32)gi };
    for (int attempt = 0; attempt < 2; attempt++) {
        pthread_mutex_lock(&nf->per_lock);
        if (!nf->connected || nf->fd < 0) {
            if (connect_feeder(nf) != 0) {
                pthread_mutex_unlock(&nf->per_lock);
                return -1;
            }
            fprintf(stderr, "[GreenBoost-netc] mem_info: inline reconnect to %s\n", nf->addr);
        }
        *out_payload = NULL;
        if (netc_send_msg(nf, GB_MSG_MEM_INFO, 0, &req, sizeof(req)) == 0 &&
            netc_recv_response(nf, out_hdr, out_payload) >= 0 && *out_payload) {
            pthread_mutex_unlock(&nf->per_lock);
            return 0;
        }
        free(*out_payload); *out_payload = NULL;
        /* Dead or desynced stream , close so the retry reconnects fresh. */
        if (nf->fd >= 0) close(nf->fd);
        nf->fd = -1;
        nf->connected = 0;
        pthread_mutex_unlock(&nf->per_lock);
    }
    return -1;
}

int gb_netc_t1_mem_info(int remote_idx, uint64_t *t1_free, uint64_t *t1_total)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return -1;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    int gi = g_remote_gpus[remote_idx].feeder_gpu_id;
    struct netc_feeder *nf = &g_feeders[fi];

    struct gb_net_header resp_hdr;
    void *resp_payload = NULL;
    if (netc_mem_info_request(nf, gi, &resp_hdr, &resp_payload) != 0) {
        fprintf(stderr, "[GreenBoost-netc] t1_mem_info[%d]: request failed (errno=%d)\n",
                remote_idx, errno);
        return -1;
    }

    const struct gb_net_mem_info_resp *resp =
        (const struct gb_net_mem_info_resp *)resp_payload;
    if (resp->status != GB_STATUS_OK && resp->t1_total == 0) {
        fprintf(stderr, "[GreenBoost-netc] t1_mem_info[%d]: feeder returned status=%u t1_total=0\n",
                remote_idx, (unsigned)resp->status);
        free(resp_payload); return -1;
    }

    /* Use v2 per-tier fields if the feeder sent the full struct */
    if (resp_hdr.payload_len >= (uint32_t)sizeof(struct gb_net_mem_info_resp)
        && resp->t1_total > 0) {
        *t1_free  = resp->t1_free;
        *t1_total = resp->t1_total;
    } else {
        /* Old feeder: conservative - treat combined total as T1 */
        *t1_free  = resp->free_bytes;
        *t1_total = resp->total_bytes;
    }
    free(resp_payload);
    return 0;
}

/* Helper: send MEM_INFO to feeder and return the full response (caller frees). */
static const struct gb_net_mem_info_resp *
netc_query_mem_info(int remote_idx, struct gb_net_header *out_hdr, void **out_payload)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return NULL;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    int gi = g_remote_gpus[remote_idx].feeder_gpu_id;
    struct netc_feeder *nf = &g_feeders[fi];

    if (netc_mem_info_request(nf, gi, out_hdr, out_payload) != 0) return NULL;
    return (const struct gb_net_mem_info_resp *)*out_payload;
}

int gb_netc_t2_mem_info(int remote_idx, uint64_t *t2_free, uint64_t *t2_total)
{
    struct gb_net_header hdr;
    void *pay = NULL;
    const struct gb_net_mem_info_resp *r = netc_query_mem_info(remote_idx, &hdr, &pay);
    if (!r) return -1;
    if (hdr.payload_len >= offsetof(struct gb_net_mem_info_resp, t2_total) + 8) {
        *t2_free  = r->t2_free;
        *t2_total = r->t2_total;
    } else {
        *t2_free = *t2_total = 0;
    }
    free(pay);
    return 0;
}

int gb_netc_t3_mem_info(int remote_idx, uint64_t *t3_free, uint64_t *t3_total)
{
    struct gb_net_header hdr;
    void *pay = NULL;
    const struct gb_net_mem_info_resp *r = netc_query_mem_info(remote_idx, &hdr, &pay);
    if (!r) return -1;
    if (hdr.payload_len >= offsetof(struct gb_net_mem_info_resp, t3_total) + 8) {
        *t3_free  = r->t3_free;
        *t3_total = r->t3_total;
    } else {
        *t3_free = *t3_total = 0;
    }
    free(pay);
    return 0;
}

/* Like gb_netc_malloc but sends the tier flag so the feeder allocates
 * on the requested tier (T1=GPU, T2=pinned DDR, T3=pageable). */
int gb_netc_malloc_tier(int remote_idx, uint64_t size, uint8_t tier,
                        uint64_t *fake_ptr_out)
{
    NETC_NVTX_PUSH("GB:net_malloc_tier", NETC_NVTX_COLOR_ALLOC);
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) {
        NETC_NVTX_POP(); return -1;
    }
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    int gi = g_remote_gpus[remote_idx].feeder_gpu_id;
    struct netc_feeder *nf = &g_feeders[fi];
    if (!nf->connected) { NETC_NVTX_POP(); return -1; }

    struct gb_net_cuda_malloc req = {
        .size      = size,
        .flags     = (gb_u32)(tier & GB_ALLOC_TIER_MASK),
        .device_id = (gb_u32)gi,
    };

    pthread_mutex_lock(&nf->per_lock);
    int ret = netc_send_msg(nf, GB_MSG_CUDA_MALLOC, 0, &req, sizeof(req));
    if (ret < 0) { pthread_mutex_unlock(&nf->per_lock); return -1; }

    struct gb_net_header resp_hdr;
    void *resp_payload = NULL;
    ret = netc_recv_response(nf, &resp_hdr, &resp_payload);
    pthread_mutex_unlock(&nf->per_lock);

    if (ret < 0 || !resp_payload) { NETC_NVTX_POP(); return -1; }

    const struct gb_net_cuda_malloc_resp *resp =
        (const struct gb_net_cuda_malloc_resp *)resp_payload;
    /* N9: ERR_THROTTLE is transient - caller falls back to local T2 */
    if (resp->status == GB_STATUS_ERR_THROTTLE) {
        netc_log("malloc_tier: feeder %s rate-throttled - fall back to local tier", nf->addr);
        NETC_NVTX_EVENT("THROTTLE_FALLBACK", "NET", size >> 20, 0, nf->addr);
        free(resp_payload);
        NETC_NVTX_POP();
        return -1;
    }
    if (resp->status != GB_STATUS_OK) {
        NETC_NVTX_EVENT("ALLOC_NET_FAIL", "NET", size >> 20, 0, nf->addr);
        free(resp_payload);
        NETC_NVTX_POP();
        return -1;
    }

    uint64_t remote_handle = resp->remote_handle;
    uint32_t tier_used     = resp->tier_used;
    free(resp_payload);

    pthread_mutex_lock(&g_alloc_lock);
    struct netc_alloc *a = alloc_new();
    if (!a) { pthread_mutex_unlock(&g_alloc_lock); NETC_NVTX_POP(); return -1; }

    uint64_t fake = gb_alloc_bump_fake_ptr(size);
    if (!fake) { pthread_mutex_unlock(&g_alloc_lock); NETC_NVTX_POP(); return -1; }

    a->fake_ptr      = fake;
    a->remote_handle = remote_handle;
    a->size          = size;
    a->feeder_idx    = fi;
    a->in_use        = 1;

    *fake_ptr_out = fake;
    netc_log("malloc_tier%u: %llu MB on feeder %s GPU %d → fake=0x%llx tier_used=%u",
             tier, (unsigned long long)(size >> 20), nf->addr, gi,
             (unsigned long long)fake, tier_used);
    NETC_NVTX_EVENT("ALLOC_NET_TIER", "NET", size >> 20, fake, nf->addr);

    pthread_mutex_unlock(&g_alloc_lock);
    NETC_NVTX_POP();
    return 0;
}

int gb_netc_total_remote_t2_bytes(uint64_t *free_bytes, uint64_t *total_bytes)
{
    uint64_t sum_free = 0, sum_total = 0;
    for (int i = 0; i < g_remote_gpu_count; i++) {
        int fi = g_remote_gpus[i].feeder_idx;
        int gi = g_remote_gpus[i].feeder_gpu_id;
        struct netc_feeder *nf = &g_feeders[fi];
        if (!nf->connected) continue;

        struct gb_net_mem_info req = { .device_id = (gb_u32)gi };
        struct gb_net_header resp_hdr;
        void *resp_payload = NULL;

        pthread_mutex_lock(&nf->per_lock);
        int ret = netc_send_msg(nf, GB_MSG_MEM_INFO, 0, &req, sizeof(req));
        if (ret >= 0)
            ret = netc_recv_response(nf, &resp_hdr, &resp_payload);
        pthread_mutex_unlock(&nf->per_lock);

        if (ret < 0 || !resp_payload) continue;
        const struct gb_net_mem_info_resp *resp =
            (const struct gb_net_mem_info_resp *)resp_payload;
        if ((resp->status == GB_STATUS_OK || resp->status == GB_STATUS_ERR_CUDA)
            && resp->t2_total > 0) {
            sum_free  += resp->t2_free;
            sum_total += resp->t2_total;
        }
        free(resp_payload);
    }
    if (free_bytes)  *free_bytes  = sum_free;
    if (total_bytes) *total_bytes = sum_total;
    return (sum_total > 0) ? 0 : -1;
}

int gb_netc_device_get_attribute(int remote_idx, int attrib, int *value)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return -1;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    int gi = g_remote_gpus[remote_idx].feeder_gpu_id;
    struct netc_feeder *nf = &g_feeders[fi];
    if (!nf->connected) return -1;

    struct gb_net_gpu_query req = {
        .device_id    = (gb_u32)gi,
        .attribute_id = (gb_u32)attrib,
    };

    pthread_mutex_lock(&nf->per_lock);
    int ret = netc_send_msg(nf, GB_MSG_GPU_QUERY, 0, &req, sizeof(req));
    if (ret < 0) { pthread_mutex_unlock(&nf->per_lock); return -1; }

    struct gb_net_header resp_hdr;
    void *resp_payload = NULL;
    ret = netc_recv_response(nf, &resp_hdr, &resp_payload);
    pthread_mutex_unlock(&nf->per_lock);

    if (ret < 0 || !resp_payload) return -1;

    const struct gb_net_gpu_query_resp *resp =
        (const struct gb_net_gpu_query_resp *)resp_payload;
    if (resp->status != GB_STATUS_OK) { free(resp_payload); return -1; }
    *value = resp->value;
    free(resp_payload);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Public API: Memory operations                                      */
/* ------------------------------------------------------------------ */

int gb_netc_malloc(int remote_idx, uint64_t size, uint32_t flags,
                   uint64_t *fake_ptr_out)
{
    NETC_NVTX_PUSH("GB:net_malloc", NETC_NVTX_COLOR_ALLOC);
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) {
        NETC_NVTX_POP(); return -1;
    }
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    int gi = g_remote_gpus[remote_idx].feeder_gpu_id;
    struct netc_feeder *nf = &g_feeders[fi];
    if (!nf->connected) { NETC_NVTX_POP(); return -1; }

    struct gb_net_cuda_malloc req = {
        .size      = size,
        .flags     = flags,
        .device_id = (gb_u32)gi,
    };

    /* A1: use per-feeder lock for network I/O */
    pthread_mutex_lock(&nf->per_lock);
    int ret = netc_send_msg(nf, GB_MSG_CUDA_MALLOC, 0, &req, sizeof(req));
    if (ret < 0) { pthread_mutex_unlock(&nf->per_lock); NETC_NVTX_POP(); return -1; }

    struct gb_net_header resp_hdr;
    void *resp_payload = NULL;
    ret = netc_recv_response(nf, &resp_hdr, &resp_payload);
    pthread_mutex_unlock(&nf->per_lock);

    if (ret < 0 || !resp_payload) { NETC_NVTX_POP(); return -1; }

    const struct gb_net_cuda_malloc_resp *resp =
        (const struct gb_net_cuda_malloc_resp *)resp_payload;
    if (resp->status != GB_STATUS_OK) {
        NETC_NVTX_EVENT("ALLOC_NET_FAIL", "NET", size >> 20, 0, nf->addr);
        free(resp_payload);
        NETC_NVTX_POP();
        return -1;
    }

    uint64_t remote_handle = resp->remote_handle;
    free(resp_payload);

    /* A1+A2: alloc table operations under g_alloc_lock, separate from network I/O */
    pthread_mutex_lock(&g_alloc_lock);
    struct netc_alloc *a = alloc_new();
    if (!a) {
        pthread_mutex_unlock(&g_alloc_lock);
        /* Remote allocation succeeded but we can't track it - free it */
        struct gb_net_cuda_free fr = { .remote_handle = remote_handle };
        pthread_mutex_lock(&nf->per_lock);
        netc_send_msg(nf, GB_MSG_CUDA_FREE, 0, &fr, sizeof(fr));
        struct gb_net_header rh; void *rp = NULL;
        netc_recv_response(nf, &rh, &rp); free(rp);
        pthread_mutex_unlock(&nf->per_lock);
        NETC_NVTX_POP();
        return -1;
    }

    uint64_t fake = gb_alloc_bump_fake_ptr(size);
    if (!fake) {
        pthread_mutex_unlock(&g_alloc_lock);
        NETC_NVTX_POP();
        return -1;
    }

    a->fake_ptr      = fake;
    a->remote_handle = remote_handle;
    a->size          = size;
    a->feeder_idx    = fi;
    a->in_use        = 1;

    *fake_ptr_out = fake;
    netc_log("malloc: %llu MB on feeder %s GPU %d → fake_ptr=0x%llx remote=0x%llx",
             (unsigned long long)(size >> 20), nf->addr, gi,
             (unsigned long long)fake, (unsigned long long)remote_handle);
    NETC_NVTX_EVENT("ALLOC_NET", "NET", size >> 20, fake, nf->addr);

    pthread_mutex_unlock(&g_alloc_lock);
    NETC_NVTX_POP();
    return 0;
}

int gb_netc_free(uint64_t fake_ptr)
{
    NETC_NVTX_PUSH("GB:net_free", NETC_NVTX_COLOR_NET);
    /* A1: look up alloc under alloc_lock, capture needed fields, then free under per_lock */
    pthread_mutex_lock(&g_alloc_lock);
    struct netc_alloc *a = alloc_find(fake_ptr);
    if (!a) {
        pthread_mutex_unlock(&g_alloc_lock);
        NETC_NVTX_POP();
        return -1;
    }

    uint64_t remote_handle = a->remote_handle;
    int      feeder_idx    = a->feeder_idx;
    size_t   size          = a->size;
    /* F-L3-12: do NOT mark in_use=0 yet. Spin outside the lock until all
     * inflight exec paths (which incremented ref_count) have finished building
     * their payloads and called gb_netc_alloc_release_ref. */
    pthread_mutex_unlock(&g_alloc_lock);
    while (atomic_load_explicit(&a->ref_count, memory_order_acquire) > 0)
        sched_yield();
    pthread_mutex_lock(&g_alloc_lock);
    a->in_use = 0;
    pthread_mutex_unlock(&g_alloc_lock);

    struct netc_feeder *nf = &g_feeders[feeder_idx];
    if (!nf->connected) { NETC_NVTX_POP(); return -1; }

    struct gb_net_cuda_free req = { .remote_handle = remote_handle };
    pthread_mutex_lock(&nf->per_lock);
    netc_send_msg(nf, GB_MSG_CUDA_FREE, 0, &req, sizeof(req));

    struct gb_net_header resp_hdr;
    void *resp_payload = NULL;
    netc_recv_response(nf, &resp_hdr, &resp_payload);
    free(resp_payload);
    pthread_mutex_unlock(&nf->per_lock);

    netc_log("free: fake_ptr=0x%llx remote=0x%llx",
             (unsigned long long)fake_ptr, (unsigned long long)remote_handle);
    NETC_NVTX_EVENT("FREE_NET", "NET", size >> 20, fake_ptr, nf->addr);
    NETC_NVTX_POP();
    return 0;
}

/* Transfers larger than the wire's max message must be chunked , a 121 MB
 * embedding tensor sent whole gets "payload_len too large" and a dropped
 * connection (hit live 2026-07-06, qwen3-0.6b first tensor).  3 MB chunks
 * leave headroom for header+desc under GB_NET_MAX_MSG_SIZE and release
 * per_lock between chunks so heartbeats interleave. */
#define GB_NETC_XFER_CHUNK (3u * 1024u * 1024u)

static int gb_netc_memcpy_h2d_one(uint64_t fake_dst, const void *host_src, uint64_t size);

int gb_netc_memcpy_h2d(uint64_t fake_dst, const void *host_src, uint64_t size)
{
    uint64_t done = 0;
    while (done < size) {
        uint64_t n = size - done;
        if (n > GB_NETC_XFER_CHUNK) n = GB_NETC_XFER_CHUNK;
        if (gb_netc_memcpy_h2d_one(fake_dst + done,
                                   (const uint8_t *)host_src + done, n) != 0)
            return -1;
        done += n;
    }
    return 0;
}

static int gb_netc_memcpy_h2d_one(uint64_t fake_dst, const void *host_src, uint64_t size)
{
    NETC_NVTX_PUSH("GB:net_h2d", NETC_NVTX_COLOR_MEMCPY);
    pthread_mutex_lock(&g_alloc_lock);
    struct netc_alloc *a = alloc_find_range(fake_dst);
    if (!a) { pthread_mutex_unlock(&g_alloc_lock); NETC_NVTX_POP(); return -1; }

    uint64_t remote_handle = a->remote_handle;
    uint64_t offset        = fake_dst - a->fake_ptr;
    int      feeder_idx    = a->feeder_idx;
    pthread_mutex_unlock(&g_alloc_lock);

    struct netc_feeder *nf = &g_feeders[feeder_idx];
    if (!nf->connected) return -1;

    struct gb_net_cuda_memcpy hdr_payload = {
        .remote_handle = remote_handle,
        .offset        = offset,
        .size          = size,
    };

    /* U20: measure H2D transfer latency for EWMA bandwidth estimate */
    struct timespec _t0, _t1;
    clock_gettime(CLOCK_MONOTONIC, &_t0);

    pthread_mutex_lock(&nf->per_lock);

    /* Fabric zstd: compress the payload before framing when the feeder
     * negotiated it and the chunk is worth compressing. The gb_net_cuda_memcpy
     * .size field stays the UNCOMPRESSED size (feeder sizes its decompress from
     * it). Compressing here (not inside netc_send_msg_with_data) means the
     * per-message MAC naturally covers the compressed bytes actually sent, and
     * we fall back to raw whenever compression is unavailable or unprofitable. */
    const void *send_data = host_src;
    uint32_t    send_len  = (uint32_t)size;
    uint16_t    send_flags = 0;
#ifdef GB_HAVE_ZSTD
    if (nf->feat_zstd && g_zstd_enabled && size >= g_zstd_min) {
        size_t bound = ZSTD_compressBound(size);
        if (nf->zbuf_cap < bound) {
            uint8_t *nb = (uint8_t *)realloc(nf->zbuf, bound);
            if (nb) { nf->zbuf = nb; nf->zbuf_cap = bound; }
        }
        if (nf->zbuf && nf->zbuf_cap >= bound) {
            size_t csize = ZSTD_compress(nf->zbuf, nf->zbuf_cap,
                                         host_src, size, g_zstd_level);
            if (!ZSTD_isError(csize) && csize < size) {
                send_data  = nf->zbuf;
                send_len   = (uint32_t)csize;
                send_flags = GB_NET_FLAG_COMP_ZSTD;
            }
        }
    }
#endif

    int ret = netc_send_msg_with_data(nf, GB_MSG_CUDA_MEMCPY_H2D, send_flags,
                                      &hdr_payload, sizeof(hdr_payload),
                                      send_data, send_len);
    if (ret < 0) { pthread_mutex_unlock(&nf->per_lock); return -1; }

    struct gb_net_header resp_hdr;
    void *resp_payload = NULL;
    ret = netc_recv_response(nf, &resp_hdr, &resp_payload);
    free(resp_payload);
    pthread_mutex_unlock(&nf->per_lock);

    /* U20: update EWMA on success */
    if (ret >= 0 && size > 0) {
        clock_gettime(CLOCK_MONOTONIC, &_t1);
        uint64_t us = (uint64_t)(_t1.tv_sec - _t0.tv_sec) * 1000000ULL
                    + (uint64_t)(_t1.tv_nsec - _t0.tv_nsec) / 1000ULL;
        if (us > 0) {
            float mbs = (float)(size >> 20) / ((float)us / 1e6f);
            nf->bw_ewma_mbs = (nf->bw_ewma_mbs == 0.0f)
                              ? mbs : (0.125f * mbs + 0.875f * nf->bw_ewma_mbs);
        }
    }

    if (ret >= 0)
        NETC_NVTX_EVENT("MEMCPY_H2D_NET", "NET", size >> 20, fake_dst, nf->addr);
    else {
        NETC_NVTX_EVENT("MEMCPY_H2D_FAIL", "NET", size >> 20, fake_dst, nf->addr);
        netc_log("memcpy H2D FAILED: dst=0x%llx off=%llu size=%llu ret=%d errno=%d",
                 (unsigned long long)fake_dst, (unsigned long long)offset,
                 (unsigned long long)size, ret, errno);
    }
    netc_log("memcpy H2D: %llu MB → feeder %s", (unsigned long long)(size >> 20), nf->addr);
    NETC_NVTX_POP();
    return (ret < 0) ? -1 : 0;
}

/* Remote memset on a feeder allocation (interior pointers supported).
 * Needed because ggml memsets quantized-tensor padding right after upload. */
int gb_netc_memset(uint64_t fake_dst, int value, uint64_t size)
{
    /* Range lookup: ggml memsets tensor padding at interior offsets. */
    pthread_mutex_lock(&g_alloc_lock);
    struct netc_alloc *a = alloc_find_range(fake_dst);
    if (!a) { pthread_mutex_unlock(&g_alloc_lock); return -1; }

    uint64_t remote_handle = a->remote_handle;
    uint64_t offset        = fake_dst - a->fake_ptr;
    int      feeder_idx    = a->feeder_idx;
    pthread_mutex_unlock(&g_alloc_lock);

    struct netc_feeder *nf = &g_feeders[feeder_idx];
    if (!nf->connected) return -1;

    struct gb_net_cuda_memset req = {
        .remote_handle = remote_handle,
        .offset        = offset,
        .size          = size,
        .value         = (gb_u32)(value & 0xFF),
    };

    pthread_mutex_lock(&nf->per_lock);
    int ret = netc_send_msg(nf, GB_MSG_CUDA_MEMSET, 0, &req, sizeof(req));
    if (ret < 0) { pthread_mutex_unlock(&nf->per_lock); return -1; }

    struct gb_net_header resp_hdr;
    void *resp_payload = NULL;
    ret = netc_recv_response(nf, &resp_hdr, &resp_payload);
    pthread_mutex_unlock(&nf->per_lock);

    int status_ok = 0;
    if (ret >= 0 && resp_payload &&
        resp_hdr.payload_len >= sizeof(struct gb_net_response)) {
        const struct gb_net_response *r = (const struct gb_net_response *)resp_payload;
        status_ok = (r->status == GB_STATUS_OK);
    }
    free(resp_payload);
    netc_log("memset: %llu KB val=%d → feeder %s (%s)",
             (unsigned long long)(size >> 10), value, nf->addr,
             status_ok ? "ok" : "FAILED");
    return status_ok ? 0 : -1;
}

static int gb_netc_memcpy_d2h_one(void *host_dst, uint64_t fake_src, uint64_t size);

int gb_netc_memcpy_d2h(void *host_dst, uint64_t fake_src, uint64_t size)
{
    uint64_t done = 0;
    while (done < size) {
        uint64_t n = size - done;
        if (n > GB_NETC_XFER_CHUNK) n = GB_NETC_XFER_CHUNK;
        if (gb_netc_memcpy_d2h_one((uint8_t *)host_dst + done, fake_src + done, n) != 0)
            return -1;
        done += n;
    }
    return 0;
}

static int gb_netc_memcpy_d2h_one(void *host_dst, uint64_t fake_src, uint64_t size)
{
    NETC_NVTX_PUSH("GB:net_d2h", NETC_NVTX_COLOR_MEMCPY);
    pthread_mutex_lock(&g_alloc_lock);
    struct netc_alloc *a = alloc_find_range(fake_src);
    if (!a) { pthread_mutex_unlock(&g_alloc_lock); NETC_NVTX_POP(); return -1; }

    uint64_t remote_handle = a->remote_handle;
    uint64_t offset        = fake_src - a->fake_ptr;
    int      feeder_idx    = a->feeder_idx;
    pthread_mutex_unlock(&g_alloc_lock);

    struct netc_feeder *nf = &g_feeders[feeder_idx];
    if (!nf->connected) { NETC_NVTX_POP(); return -1; }

    struct gb_net_cuda_memcpy req = {
        .remote_handle = remote_handle,
        .offset        = offset,
        .size          = size,
    };

    struct timespec _t0, _t1;
    clock_gettime(CLOCK_MONOTONIC, &_t0);

    /* U15: try to use a pre-mlock'd staging buffer to avoid page faults on receive */
    void *pinned_staging = (size <= GB_PINNED_BUF_SIZE) ? gb_pinned_acquire() : NULL;

    pthread_mutex_lock(&nf->per_lock);
    int ret = netc_send_msg(nf, GB_MSG_CUDA_MEMCPY_D2H, 0, &req, sizeof(req));
    if (ret < 0) {
        pthread_mutex_unlock(&nf->per_lock);
        if (pinned_staging) gb_pinned_release(pinned_staging);
        NETC_NVTX_POP();
        return -1;
    }

    struct gb_net_header resp_hdr;
    void *resp_payload = NULL;
    ret = netc_recv_response(nf, &resp_hdr, &resp_payload);
    pthread_mutex_unlock(&nf->per_lock);

    if (ret < 0 || !resp_payload) {
        if (pinned_staging) gb_pinned_release(pinned_staging);
        NETC_NVTX_POP();
        return -1;
    }

    const struct gb_net_response *resp = (const struct gb_net_response *)resp_payload;
    if (resp->status != GB_STATUS_OK) {
        free(resp_payload);
        if (pinned_staging) gb_pinned_release(pinned_staging);
        NETC_NVTX_POP();
        return -1;
    }

    size_t data_offset = sizeof(struct gb_net_response);
    if (resp_hdr.payload_len > data_offset) {
        size_t data_len = resp_hdr.payload_len - data_offset;
        if (data_len > size) data_len = size;
        const uint8_t *src_ptr = (const uint8_t *)resp_payload + data_offset;
        if (pinned_staging && data_len <= GB_PINNED_BUF_SIZE) {
            memcpy(pinned_staging, src_ptr, data_len);
            memcpy(host_dst, pinned_staging, data_len);
        } else {
            memcpy(host_dst, src_ptr, data_len);
        }
    }

    free(resp_payload);
    if (pinned_staging) gb_pinned_release(pinned_staging);

    /* U20: update EWMA bandwidth on successful D2H transfer */
    if (size > 0) {
        clock_gettime(CLOCK_MONOTONIC, &_t1);
        uint64_t us = (uint64_t)(_t1.tv_sec - _t0.tv_sec) * 1000000ULL
                    + (uint64_t)(_t1.tv_nsec - _t0.tv_nsec) / 1000ULL;
        if (us > 0) {
            float mbs = (float)(size >> 20) / ((float)us / 1e6f);
            nf->bw_ewma_mbs = (nf->bw_ewma_mbs == 0.0f)
                              ? mbs : (0.125f * mbs + 0.875f * nf->bw_ewma_mbs);
        }
    }

    NETC_NVTX_EVENT("MEMCPY_D2H_NET", "NET", size >> 20, fake_src, nf->addr);
    netc_log("memcpy D2H: %llu MB ← feeder %s", (unsigned long long)(size >> 20), nf->addr);
    NETC_NVTX_POP();
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Public API: Kernel launch                                          */
/* ------------------------------------------------------------------ */

void gb_netc_register_kernel(const void *host_func, const char *device_name)
{
    pthread_mutex_lock(&g_netc_lock);
    /* Check if already registered locally */
    for (int i = 0; i < NETC_MAX_KERNELS; i++) {
        if (g_kernels[i].in_use && g_kernels[i].host_func == host_func) {
            pthread_mutex_unlock(&g_netc_lock);
            return;
        }
    }
    for (int i = 0; i < NETC_MAX_KERNELS; i++) {
        if (!g_kernels[i].in_use) {
            g_kernels[i].host_func = host_func;
            strncpy(g_kernels[i].name, device_name, GB_NET_MAX_KERNEL_NAME - 1);
            g_kernels[i].name[GB_NET_MAX_KERNEL_NAME - 1] = '\0';
            g_kernels[i].in_use = 1;
            netc_log("register kernel: %s → %p", device_name, host_func);
            break;
        }
    }

    /* Propagate to connected feeders so they can dlsym the kernel and
     * execute it locally when we dispatch via GB_MSG_CUDA_EXEC.
     * Cap the pushes: torch processes register 4000+ kernels at import
     * (none dlsym-resolvable on the feeder anyway , netd resolves scoped to
     * its ggml lib), which floods the daemon with a ~4 MB burst at connect
     * time. ggml registers well under the cap. Registration is an
     * optimisation only: exec falls back to the feeder's own scoped dlsym. */
    static _Atomic uint32_t g_kernel_push_count = 0;
    uint32_t push_max = 1024;
    const char *pm = getenv("GREENBOOST_KERNEL_PUSH_MAX");
    if (pm) push_max = (uint32_t)atoi(pm);
    if (atomic_fetch_add_explicit(&g_kernel_push_count, 1, memory_order_relaxed)
            >= push_max) {
        pthread_mutex_unlock(&g_netc_lock);
        return;
    }
    uint32_t name_len = (uint32_t)strlen(device_name);
    if (name_len >= GB_NET_MAX_KERNEL_NAME) name_len = GB_NET_MAX_KERNEL_NAME - 1;
    uint32_t msg_size = (uint32_t)sizeof(struct gb_net_cuda_register_fn) + name_len;
    uint8_t *buf = (uint8_t *)alloca(msg_size);
    struct gb_net_cuda_register_fn hdr = { .kernel_name_len = name_len, .t2_speed_mts = 0 };
    memcpy(buf, &hdr, sizeof(hdr));
    memcpy(buf + sizeof(hdr), device_name, name_len);

    for (int fi = 0; fi < g_feeder_count; fi++) {
        struct netc_feeder *nf = &g_feeders[fi];
        if (!nf->connected) continue;
        /* Must hold per_lock while sending: heartbeat thread (gb_netc_poll_health)
         * also writes to this socket under per_lock. g_netc_lock alone does not
         * prevent the race , they are independent locks. Concurrent writes corrupt
         * the TCP framing, causing the feeder to close the connection on seq mismatch. */
        pthread_mutex_lock(&nf->per_lock);
        netc_send_msg(nf, GB_MSG_CUDA_REGISTER_FN, 0, buf, msg_size);
        pthread_mutex_unlock(&nf->per_lock);
    }

    pthread_mutex_unlock(&g_netc_lock);
}

const char *gb_netc_lookup_kernel(const void *host_func)
{
    for (int i = 0; i < NETC_MAX_KERNELS; i++)
        if (g_kernels[i].in_use && g_kernels[i].host_func == host_func)
            return g_kernels[i].name;
    return NULL;
}

int gb_netc_launch_kernel(int remote_idx,
                          const char *kernel_name,
                          unsigned int gx, unsigned int gy, unsigned int gz,
                          unsigned int bx, unsigned int by, unsigned int bz,
                          unsigned int shared_mem,
                          uint32_t stream_id,
                          const void *arg_buffer, uint32_t arg_buffer_size)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return -1;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    struct netc_feeder *nf = &g_feeders[fi];
    if (!nf->connected) return -1;

    /* U13: enforce PID dispatch rate limit */
    if (nf->pid.target_rate > 0) {
        struct timespec _ts;
        clock_gettime(CLOCK_MONOTONIC, &_ts);
        uint64_t now_ms = (uint64_t)_ts.tv_sec * 1000 + (uint64_t)_ts.tv_nsec / 1000000;
        if (now_ms - nf->pid.window_start_ms >= 1000) {
            nf->pid.dispatch_count  = 0;
            nf->pid.window_start_ms = now_ms;
        }
        if (nf->pid.dispatch_count >= nf->pid.target_rate) {
            struct timespec sleep_ts = { 0, 1000000 }; /* 1 ms yield */
            nanosleep(&sleep_ts, NULL);
        }
        nf->pid.dispatch_count++;
    }

    uint32_t name_len = (uint32_t)strlen(kernel_name);

    struct gb_net_cuda_launch launch = {
        .grid_x  = gx, .grid_y  = gy, .grid_z  = gz,
        .block_x = bx, .block_y = by, .block_z = bz,
        .shared_mem_bytes = shared_mem,
        .stream_id        = stream_id,
        .kernel_name_len  = name_len,
        .arg_buffer_size  = arg_buffer_size,
    };

    /* Build payload: launch_header + kernel_name + arg_buffer */
    uint32_t total = sizeof(launch) + name_len + arg_buffer_size;
    uint8_t *payload = (uint8_t *)malloc(total);
    if (!payload) return -1;

    memcpy(payload, &launch, sizeof(launch));
    memcpy(payload + sizeof(launch), kernel_name, name_len);
    if (arg_buffer_size > 0)
        memcpy(payload + sizeof(launch) + name_len, arg_buffer, arg_buffer_size);

    pthread_mutex_lock(&nf->per_lock);
    int ret = netc_send_msg(nf, GB_MSG_CUDA_LAUNCH, 0, payload, total);
    free(payload);

    if (ret < 0) { pthread_mutex_unlock(&nf->per_lock); return -1; }

    struct gb_net_header resp_hdr;
    void *resp_payload = NULL;
    ret = netc_recv_response(nf, &resp_hdr, &resp_payload);
    pthread_mutex_unlock(&nf->per_lock);

    if (ret < 0) return -1;
    int status = -1;
    if (resp_payload) {
        const struct gb_net_response *r = (const struct gb_net_response *)resp_payload;
        status = (r->status == GB_STATUS_OK) ? 0 : -1;
        free(resp_payload);
    }

    netc_log("launch kernel '%s' on feeder %s: %s",
             kernel_name, nf->addr, status == 0 ? "OK" : "FAIL");
    return status;
}

int gb_netc_stream_sync(int remote_idx, uint32_t stream_id)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return -1;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    struct netc_feeder *nf = &g_feeders[fi];
    if (!nf->connected) return -1;

    struct gb_net_cuda_sync req = { .stream_id = stream_id };

    pthread_mutex_lock(&nf->per_lock);
    int ret = netc_send_msg(nf, GB_MSG_CUDA_SYNC, 0, &req, sizeof(req));
    if (ret < 0) { pthread_mutex_unlock(&nf->per_lock); return -1; }

    struct gb_net_header resp_hdr;
    void *resp_payload = NULL;
    ret = netc_recv_response(nf, &resp_hdr, &resp_payload);
    pthread_mutex_unlock(&nf->per_lock);

    free(resp_payload);
    return (ret < 0) ? -1 : 0;
}

/* ------------------------------------------------------------------ */
/*  Public API: Active device tracking                                 */
/* ------------------------------------------------------------------ */

void gb_netc_set_active_remote(int remote_idx)
{
    g_active_remote = remote_idx;
}

int gb_netc_get_active_remote(void)
{
    return g_active_remote;
}

int gb_netc_memcpy_d2d_push(uint64_t fake_dst, const void *local_src, uint64_t size)
{
#ifdef GREENBOOST_USE_NCCL
    pthread_mutex_lock(&g_alloc_lock);
    struct netc_alloc *a = alloc_find(fake_dst);
    if (!a) { pthread_mutex_unlock(&g_alloc_lock); return -1; }
    uint64_t remote_handle = a->remote_handle + (fake_dst - a->fake_ptr);
    int feeder_idx = a->feeder_idx;
    pthread_mutex_unlock(&g_alloc_lock);

    struct netc_feeder *nf = &g_feeders[feeder_idx];
    if (!nf->connected || !nf->nccl_comm) return -1;

    struct gb_net_cuda_memcpy_d2d req = {
        .src_handle = 0,
        .dst_handle = remote_handle,
        .size = size
    };

    pthread_mutex_lock(&nf->per_lock);
    if (netc_send_msg(nf, GB_MSG_CUDA_MEMCPY_D2D, 0, &req, sizeof(req)) < 0) {
        pthread_mutex_unlock(&nf->per_lock);
        return -1;
    }

    struct gb_net_header resp_hdr;
    void *resp_payload = NULL;
    if (netc_recv_response(nf, &resp_hdr, &resp_payload) < 0 || !resp_payload) {
        pthread_mutex_unlock(&nf->per_lock);
        return -1;
    }

    const struct gb_net_response *resp = resp_payload;
    if (resp->status != GB_STATUS_OK) {
        free(resp_payload);
        pthread_mutex_unlock(&nf->per_lock);
        return -1;
    }
    free(resp_payload);
    pthread_mutex_unlock(&nf->per_lock);

    /* Perform NCCL send (async, but we sync here to mimic cudaMemcpy sync behavior if stream=0) */
    ncclResult_t nr = ncclSend(local_src, size, ncclChar, 1, nf->nccl_comm, NULL);
    if (nr != ncclSuccess) {
        netc_log("ERR: ncclSend failed: %d", nr);
        return -1;
    }
    return 0;
#else
    return -1;
#endif
}

int gb_netc_memcpy_d2d_pull(void *local_dst, uint64_t fake_src, uint64_t size)
{
#ifdef GREENBOOST_USE_NCCL
    pthread_mutex_lock(&g_alloc_lock);
    struct netc_alloc *a = alloc_find(fake_src);
    if (!a) { pthread_mutex_unlock(&g_alloc_lock); return -1; }
    uint64_t remote_handle = a->remote_handle + (fake_src - a->fake_ptr);
    int feeder_idx = a->feeder_idx;
    pthread_mutex_unlock(&g_alloc_lock);

    struct netc_feeder *nf = &g_feeders[feeder_idx];
    if (!nf->connected || !nf->nccl_comm) return -1;

    struct gb_net_cuda_memcpy_d2d req = {
        .src_handle = remote_handle,
        .dst_handle = 0,
        .size = size
    };

    pthread_mutex_lock(&nf->per_lock);
    if (netc_send_msg(nf, GB_MSG_CUDA_MEMCPY_D2D, 0, &req, sizeof(req)) < 0) {
        pthread_mutex_unlock(&nf->per_lock);
        return -1;
    }

    struct gb_net_header resp_hdr;
    void *resp_payload = NULL;
    if (netc_recv_response(nf, &resp_hdr, &resp_payload) < 0 || !resp_payload) {
        pthread_mutex_unlock(&nf->per_lock);
        return -1;
    }

    const struct gb_net_response *resp = resp_payload;
    if (resp->status != GB_STATUS_OK) {
        free(resp_payload);
        pthread_mutex_unlock(&nf->per_lock);
        return -1;
    }
    free(resp_payload);
    pthread_mutex_unlock(&nf->per_lock);

    /* Perform NCCL recv */
    ncclResult_t nr = ncclRecv(local_dst, size, ncclChar, 1, nf->nccl_comm, NULL);
    if (nr != ncclSuccess) {
        netc_log("ERR: ncclRecv failed: %d", nr);
        return -1;
    }
    return 0;
#else
    return -1;
#endif
}

/* ------------------------------------------------------------------ */
/*  Single-GPU cluster: alloc info + remote kernel exec               */
/* ------------------------------------------------------------------ */

static struct netc_alloc *alloc_find_range(uint64_t ptr)
{
    for (int i = 0; i < NETC_MAX_ALLOCS; i++) {
        struct netc_alloc *a = &g_allocs[i];
        if (a->in_use && ptr >= a->fake_ptr && ptr < a->fake_ptr + a->size)
            return a;
    }
    return NULL;
}

int gb_netc_get_alloc_info(uint64_t ptr, uint64_t *remote_handle,
                           uint64_t *base, uint64_t *alloc_size, int *feeder_idx)
{
    pthread_mutex_lock(&g_alloc_lock);
    struct netc_alloc *a = alloc_find_range(ptr);
    if (!a) { pthread_mutex_unlock(&g_alloc_lock); return -1; }
    if (remote_handle) *remote_handle = a->remote_handle;
    if (base)          *base          = a->fake_ptr;
    if (alloc_size)    *alloc_size    = a->size;
    if (feeder_idx)    *feeder_idx    = a->feeder_idx;
    /* F-L3-12: pin this alloc so a concurrent gb_netc_free waits for us to
     * finish building the exec payload before marking the slot free. */
    atomic_fetch_add_explicit(&a->ref_count, 1, memory_order_acquire);
    pthread_mutex_unlock(&g_alloc_lock);
    return 0;
}

void gb_netc_alloc_release_ref(uint64_t ptr)
{
    pthread_mutex_lock(&g_alloc_lock);
    struct netc_alloc *a = alloc_find_range(ptr);
    if (a) atomic_fetch_sub_explicit(&a->ref_count, 1, memory_order_release);
    pthread_mutex_unlock(&g_alloc_lock);
}

int gb_netc_exec_kernel_raw(int feeder_idx,
                            const char *kernel_name,
                            unsigned int gx, unsigned int gy, unsigned int gz,
                            unsigned int bx, unsigned int by, unsigned int bz,
                            uint32_t shared_mem,
                            const uint8_t *param_buf, uint32_t param_buf_bytes,
                            const uint32_t *param_sizes, uint32_t n_params,
                            const struct gb_exec_reloc_raw *relocs, int n_relocs)
{
    if (feeder_idx < 0 || feeder_idx >= g_feeder_count) return -1;
    struct netc_feeder *nf = &g_feeders[feeder_idx];
    if (!nf->connected) return -1;

    uint32_t name_len    = (uint32_t)strlen(kernel_name);
    uint32_t psize_bytes = n_params * (uint32_t)sizeof(uint32_t);
    uint32_t reloc_bytes = (uint32_t)n_relocs * (uint32_t)sizeof(struct gb_net_ptr_reloc);
    uint32_t total = (uint32_t)sizeof(struct gb_net_cuda_exec_raw)
                   + name_len + psize_bytes + reloc_bytes + param_buf_bytes;

    uint8_t *payload = (uint8_t *)malloc(total);
    if (!payload) return -1;

    struct gb_net_cuda_exec_raw hdr = {
        .grid_x = gx, .grid_y = gy, .grid_z = gz,
        .block_x = bx, .block_y = by, .block_z = bz,
        .shared_mem_bytes = shared_mem,
        .kernel_name_len  = name_len,
        .n_params         = n_params,
        .n_relocs         = (gb_u32)n_relocs,
        .param_buf_bytes  = param_buf_bytes,
        .launch_mode      = 0,   /* feeder decides by resolution */
    };

    uint8_t *p = payload;
    memcpy(p, &hdr, sizeof(hdr));            p += sizeof(hdr);
    memcpy(p, kernel_name, name_len);        p += name_len;
    memcpy(p, param_sizes, psize_bytes);     p += psize_bytes;
    for (int i = 0; i < n_relocs; i++) {
        struct gb_net_ptr_reloc r;
        r.arg_idx       = relocs[i].buf_offset;   /* reinterpreted as byte offset */
        r.t2_speed_mts  = 0;
        r.remote_handle = relocs[i].remote_handle;
        memcpy(p, &r, sizeof(r));            p += sizeof(r);
    }
    memcpy(p, param_buf, param_buf_bytes);   p += param_buf_bytes;

    pthread_mutex_lock(&nf->per_lock);
    int ret = netc_send_msg(nf, GB_MSG_CUDA_EXEC, GB_NET_FLAG_EXEC_RAW, payload, total);
    free(payload);
    if (ret < 0) { pthread_mutex_unlock(&nf->per_lock); return -1; }

    struct gb_net_header resp_hdr;
    void *resp_payload = NULL;
    ret = netc_recv_response(nf, &resp_hdr, &resp_payload);
    pthread_mutex_unlock(&nf->per_lock);
    if (ret < 0 || !resp_payload) {
        netc_log("exec_kernel_raw '%s': recv failed ret=%d", kernel_name, ret);
        return -1;
    }
    const struct gb_net_response *resp = (const struct gb_net_response *)resp_payload;
    int ok = (resp->status == GB_STATUS_OK);
    if (!ok) netc_log("exec_kernel_raw '%s': feeder status=%u", kernel_name,
                      (unsigned)resp->status);
    free(resp_payload);
    return ok ? 0 : -1;
}

int gb_netc_exec_kernel(int feeder_idx,
                        const char *kernel_name,
                        unsigned int gx, unsigned int gy, unsigned int gz,
                        unsigned int bx, unsigned int by, unsigned int bz,
                        uint32_t shared_mem,
                        const uint64_t *arg_vals, uint32_t n_arg_vals,
                        const struct gb_exec_reloc *relocs, int n_relocs,
                        struct gb_exec_upload *uploads, int n_uploads,
                        int n_downloads)
{
    NETC_NVTX_PUSH("GB:net_exec_kernel", NETC_NVTX_COLOR_KERN);
    if (feeder_idx < 0 || feeder_idx >= g_feeder_count) {
        NETC_NVTX_POP(); return -1;
    }
    struct netc_feeder *nf = &g_feeders[feeder_idx];
    if (!nf->connected) { NETC_NVTX_POP(); return -1; }

    /* PR-C/H10: the download-desc serialiser at the bottom indexes uploads[i]
     * inside a loop bounded by n_downloads.  The invariant this code assumes is
     * "downloads are a subset of uploads, with download[i] := upload[i]"; with
     * n_downloads > n_uploads we walk off the uploads array (UB) and the
     * response parser at the end reads uploads[i] for i in [0, n_uploads),
     * silently substituting uploaded payloads into mismatched arg slots.
     * Reject the malformed call early instead. */
    if (n_downloads < 0 || n_uploads < 0 || n_downloads > n_uploads) {
        netc_log("exec_kernel rejected: n_downloads=%d > n_uploads=%d", n_downloads, n_uploads);
        NETC_NVTX_POP(); return -1;
    }

    uint32_t name_len   = (uint32_t)strlen(kernel_name);
    uint32_t arg_bytes  = n_arg_vals * (uint32_t)sizeof(uint64_t);
    uint32_t reloc_bytes = (uint32_t)n_relocs * (uint32_t)sizeof(struct gb_net_ptr_reloc);
    uint32_t xdesc_size = (uint32_t)sizeof(struct gb_net_xfer_desc);
    uint32_t up_desc_bytes = (uint32_t)n_uploads  * xdesc_size;
    uint32_t dn_desc_bytes = (uint32_t)n_downloads * xdesc_size;

    uint64_t upload_data_bytes = 0;
    for (int i = 0; i < n_uploads; i++) upload_data_bytes += (uint64_t)uploads[i].size;
    if (upload_data_bytes > 256ULL * 1024 * 1024) { NETC_NVTX_POP(); return -1; }

    uint32_t total = (uint32_t)sizeof(struct gb_net_cuda_exec)
                   + name_len + arg_bytes + reloc_bytes
                   + up_desc_bytes + (uint32_t)upload_data_bytes
                   + dn_desc_bytes;

    uint8_t *payload = (uint8_t *)malloc(total);
    if (!payload) { NETC_NVTX_POP(); return -1; }

    struct gb_net_cuda_exec hdr = {
        .grid_x = gx, .grid_y = gy, .grid_z = gz,
        .block_x = bx, .block_y = by, .block_z = bz,
        .shared_mem_bytes = shared_mem,
        .kernel_name_len  = name_len,
        .n_arg_vals       = n_arg_vals,
        .n_relocs         = (gb_u32)n_relocs,
        .n_uploads        = (gb_u32)n_uploads,
        .n_downloads      = (gb_u32)n_downloads,
    };

    uint8_t *p = payload;
    memcpy(p, &hdr, sizeof(hdr)); p += sizeof(hdr);
    memcpy(p, kernel_name, name_len); p += name_len;
    memcpy(p, arg_vals, arg_bytes); p += arg_bytes;

    for (int i = 0; i < n_relocs; i++) {
        struct gb_net_ptr_reloc r;
        r.arg_idx = (gb_u32)relocs[i].arg_idx;
        r.t2_speed_mts = 0;
        r.remote_handle = relocs[i].remote_handle;
        memcpy(p, &r, sizeof(r)); p += sizeof(r);
    }

    for (int i = 0; i < n_uploads; i++) {
        struct gb_net_xfer_desc d;
        d.arg_idx = (gb_u32)uploads[i].arg_idx;
        d.t2_speed_mts = 0;
        d.size = (gb_u64)uploads[i].size;
        memcpy(p, &d, sizeof(d)); p += sizeof(d);
    }
    for (int i = 0; i < n_uploads; i++) {
        memcpy(p, uploads[i].host_data, uploads[i].size);
        p += uploads[i].size;
    }
    /* Download descs: must match n_downloads (not n_uploads) - total was
     * allocated with dn_desc_bytes = n_downloads * xdesc_size; writing
     * n_uploads entries instead causes a buffer over/underrun. */
    for (int i = 0; i < n_downloads; i++) {
        struct gb_net_xfer_desc d;
        d.arg_idx = (gb_u32)uploads[i].arg_idx;
        d.t2_speed_mts = 0;
        d.size = (gb_u64)uploads[i].size;
        memcpy(p, &d, sizeof(d)); p += sizeof(d);
    }

    pthread_mutex_lock(&nf->per_lock);
    int ret = netc_send_msg(nf, GB_MSG_CUDA_EXEC, 0, payload, total);
    free(payload);

    if (ret < 0) {
        pthread_mutex_unlock(&nf->per_lock);
        netc_log("exec_kernel '%s': send failed errno=%d", kernel_name, errno);
        NETC_NVTX_POP(); return -1;
    }

    struct gb_net_header resp_hdr;
    void *resp_payload = NULL;
    ret = netc_recv_response(nf, &resp_hdr, &resp_payload);
    pthread_mutex_unlock(&nf->per_lock);

    if (ret < 0 || !resp_payload) {
        netc_log("exec_kernel '%s': recv failed ret=%d errno=%d", kernel_name, ret, errno);
        NETC_NVTX_POP(); return -1;
    }

    const struct gb_net_response *resp = (const struct gb_net_response *)resp_payload;
    if (resp->status != GB_STATUS_OK) {
        netc_log("exec_kernel '%s': feeder returned status=%u", kernel_name, (unsigned)resp->status);
        free(resp_payload); NETC_NVTX_POP(); return -1;
    }

    /* Audit F-L3-18: validate that the response payload is at least large
     * enough to contain the gb_net_response header plus the downloads we
     * asked for.  Without this guard, a short response causes rp_left to
     * underflow and the loop reads past the malloc'd buffer. */
    if (resp_hdr.payload_len < sizeof(struct gb_net_response)) {
        free(resp_payload); NETC_NVTX_POP(); return -1;
    }
    uint64_t want_total = 0;
    for (int i = 0; i < n_uploads; i++) want_total += (uint64_t)uploads[i].size;
    uint64_t have_total = (uint64_t)resp_hdr.payload_len
                        - (uint64_t)sizeof(struct gb_net_response);
    if (want_total > have_total) {
        netc_log("exec response truncated: want=%llu have=%llu",
                 (unsigned long long)want_total,
                 (unsigned long long)have_total);
        free(resp_payload); NETC_NVTX_POP(); return -1;
    }

    /* Parse downloaded tensor data from response */
    const uint8_t *rp = (const uint8_t *)resp_payload + sizeof(struct gb_net_response);
    uint32_t rp_left = (uint32_t)have_total;
    for (int i = 0; i < n_uploads; i++) {
        if (rp_left < (uint32_t)uploads[i].size) break;
        uploads[i].downloaded_data = malloc(uploads[i].size);
        if (uploads[i].downloaded_data)
            memcpy(uploads[i].downloaded_data, rp, uploads[i].size);
        rp      += uploads[i].size;
        rp_left -= (uint32_t)uploads[i].size;
    }

    free(resp_payload);
    /* F-L1-31: mark this feeder as having dispatched a kernel so
     * cudaStreamSynchronize only contacts feeders with pending work. */
    atomic_store_explicit(&nf->kernel_dispatched, 1, memory_order_release);
    netc_log("exec_kernel '%s' on feeder %d: OK", kernel_name, feeder_idx);
    NETC_NVTX_EVENT("EXEC_KERNEL_NET", "NET", 0, feeder_idx, kernel_name);
    NETC_NVTX_POP();
    return 0;
}

/* Phase 2b: Fire-and-forget async kernel dispatch (GB_MSG_CUDA_EXEC_ASYNC).
 *
 * Sends the kernel request and returns as soon as the feeder ACKs receipt
 * (before GPU execution completes).  The feeder enqueues the kernel on its
 * per-client async CUDA stream and ACKs immediately; outstanding work is
 * drained the next time gb_netc_selective_stream_sync() sends GB_MSG_CUDA_SYNC.
 *
 * Constraints enforced here (mirrors feeder validation):
 *   - n_uploads must be 0 (no inline tensor upload; weights must already be
 *     in feeder VRAM as remote allocations addressed via relocs)
 *   - n_downloads must be 0 (caller does not need data back immediately)
 *
 * Returns 0 on success (kernel queued on feeder), -1 on error. */
int gb_netc_exec_kernel_async(int feeder_idx,
                              const char *kernel_name,
                              unsigned int gx, unsigned int gy, unsigned int gz,
                              unsigned int bx, unsigned int by, unsigned int bz,
                              uint32_t shared_mem,
                              const uint64_t *arg_vals, uint32_t n_arg_vals,
                              const struct gb_exec_reloc *relocs, int n_relocs)
{
    NETC_NVTX_PUSH("GB:net_exec_async", NETC_NVTX_COLOR_KERN);
    if (feeder_idx < 0 || feeder_idx >= g_feeder_count) {
        NETC_NVTX_POP(); return -1;
    }
    struct netc_feeder *nf = &g_feeders[feeder_idx];
    if (!nf->connected) { NETC_NVTX_POP(); return -1; }

    uint32_t name_len    = (uint32_t)strlen(kernel_name);
    uint32_t arg_bytes   = n_arg_vals * (uint32_t)sizeof(uint64_t);
    uint32_t reloc_bytes = (uint32_t)n_relocs * (uint32_t)sizeof(struct gb_net_ptr_reloc);

    uint32_t total = (uint32_t)sizeof(struct gb_net_cuda_exec)
                   + name_len + arg_bytes + reloc_bytes;

    uint8_t *payload = (uint8_t *)malloc(total);
    if (!payload) { NETC_NVTX_POP(); return -1; }

    struct gb_net_cuda_exec hdr = {
        .grid_x = gx, .grid_y = gy, .grid_z = gz,
        .block_x = bx, .block_y = by, .block_z = bz,
        .shared_mem_bytes = shared_mem,
        .kernel_name_len  = name_len,
        .n_arg_vals       = n_arg_vals,
        .n_relocs         = (gb_u32)n_relocs,
        .n_uploads        = 0,
        .n_downloads      = 0,
    };

    uint8_t *p = payload;
    memcpy(p, &hdr, sizeof(hdr)); p += sizeof(hdr);
    memcpy(p, kernel_name, name_len); p += name_len;
    if (arg_bytes) { memcpy(p, arg_vals, arg_bytes); p += arg_bytes; }
    for (int i = 0; i < n_relocs; i++) {
        struct gb_net_ptr_reloc r;
        r.arg_idx       = (gb_u32)relocs[i].arg_idx;
        r.t2_speed_mts  = 0;
        r.remote_handle = relocs[i].remote_handle;
        memcpy(p, &r, sizeof(r)); p += sizeof(r);
    }

    pthread_mutex_lock(&nf->per_lock);
    int ret = netc_send_msg(nf, GB_MSG_CUDA_EXEC_ASYNC, 0, payload, total);
    free(payload);

    if (ret < 0) { pthread_mutex_unlock(&nf->per_lock); NETC_NVTX_POP(); return -1; }

    /* Receive the lightweight ACK (just gb_net_response, no download data) */
    struct gb_net_header resp_hdr;
    void *resp_payload = NULL;
    ret = netc_recv_response(nf, &resp_hdr, &resp_payload);
    pthread_mutex_unlock(&nf->per_lock);

    if (ret < 0 || !resp_payload) { free(resp_payload); NETC_NVTX_POP(); return -1; }
    const struct gb_net_response *resp = (const struct gb_net_response *)resp_payload;
    int ok = (resp->status == GB_STATUS_OK);
    free(resp_payload);

    if (ok) {
        atomic_store_explicit(&nf->kernel_dispatched, 1, memory_order_release);
        atomic_fetch_add_explicit(&nf->pending_async_kernels, 1, memory_order_relaxed);
        netc_log("exec_kernel_async '%s' on feeder %d: queued", kernel_name, feeder_idx);
        NETC_NVTX_EVENT("EXEC_KERNEL_ASYNC_NET", "NET", 0, feeder_idx, kernel_name);
    } else {
        netc_log("exec_kernel_async '%s' on feeder %d: ERR status=%u",
                 kernel_name, feeder_idx, resp ? resp->status : 99);
    }
    NETC_NVTX_POP();
    return ok ? 0 : -1;
}

/* F-L1-31: selective stream sync - only contacts feeders that have had a
 * kernel dispatched since the last sync, avoiding unnecessary round-trips
 * to idle feeders. */
void gb_netc_selective_stream_sync(void)
{
    for (int fi = 0; fi < g_feeder_count; fi++) {
        struct netc_feeder *nf = &g_feeders[fi];
        if (!nf->connected) continue;
        if (!atomic_load_explicit(&nf->kernel_dispatched, memory_order_acquire)) continue;
        atomic_store_explicit(&nf->kernel_dispatched, 0, memory_order_relaxed);
        atomic_store_explicit(&nf->pending_async_kernels, 0, memory_order_relaxed);

        struct gb_net_cuda_sync req = { .stream_id = 0 };
        pthread_mutex_lock(&nf->per_lock);
        int ret = netc_send_msg(nf, GB_MSG_CUDA_SYNC, 0, &req, sizeof(req));
        if (ret >= 0) {
            struct gb_net_header resp_hdr;
            void *resp_payload = NULL;
            netc_recv_response(nf, &resp_hdr, &resp_payload);
            free(resp_payload);
        }
        pthread_mutex_unlock(&nf->per_lock);
    }
}

/* D2/D3: Send a heartbeat to all connected feeders and cache throttle/ECC state.
 * Called periodically from the shim stats writer - interval is 2s (8× stats period).
 * Non-blocking: if the lock is contended, skip this poll cycle. */
void gb_netc_poll_health(void)
{
    if (!g_netc_initialized) return;
    if (pthread_mutex_trylock(&g_netc_lock) != 0) return;

    for (int i = 0; i < g_feeder_count; i++) {
        struct netc_feeder *nf = &g_feeders[i];

        struct timespec _hb_ts;
        clock_gettime(CLOCK_MONOTONIC, &_hb_ts);
        uint64_t now_ms = (uint64_t)_hb_ts.tv_sec * 1000 + (uint64_t)_hb_ts.tv_nsec / 1000000;

        /* N8: handle disconnected feeders with exponential backoff reconnect.
         * MUST hold per_lock: connect_feeder replaces nf->fd and resets the
         * seq counters , doing that while another thread is mid send/recv on
         * the old socket abandons a live connection (netd then drops it idle
         * at 15 s and every later send gets EPIPE; hit live 2026-07-06). */
        if (!nf->connected || nf->fd < 0) {
            if (now_ms < nf->next_reconnect_ms) continue;
            if (pthread_mutex_trylock(&nf->per_lock) != 0) continue;
            if (nf->connected && nf->fd >= 0) {   /* raced: someone reconnected */
                pthread_mutex_unlock(&nf->per_lock);
                continue;
            }
            NETC_NVTX_EVENT("RECONNECT_ATTEMPT", "NET", 0, i, nf->addr);
            if (connect_feeder(nf) == 0) {
                fprintf(stderr, "[GreenBoost-netc] N8: reconnected to feeder %s\n", nf->addr);
                NETC_NVTX_EVENT("RECONNECT_OK", "NET", 0, i, nf->addr);
                nf->reconnect_delay_ms = 500;
                nf->health_state       = GB_HEALTH_HEALTHY;
                nf->health_state_since_ms = now_ms;
            } else {
                uint64_t delay = nf->reconnect_delay_ms < 500 ? 500 : nf->reconnect_delay_ms * 2;
                if (delay > 30000) delay = 30000;
                nf->reconnect_delay_ms = delay;
                nf->next_reconnect_ms  = now_ms + delay;
                fprintf(stderr, "[GreenBoost-netc] N8: reconnect to %s failed, retry in %llu ms\n",
                        nf->addr, (unsigned long long)delay);
                NETC_NVTX_EVENT("RECONNECT_FAIL", "NET", delay / 1000, i, nf->addr);
            }
            pthread_mutex_unlock(&nf->per_lock);
            continue;
        }

        /* per_lock guards all socket I/O on this feeder - acquire with trylock
         * so a slow malloc in flight doesn't block the heartbeat scheduler. */
        if (pthread_mutex_trylock(&nf->per_lock) != 0) {
            NETC_NVTX_EVENT("HEARTBEAT_SKIP_BUSY", "NET", 0, i, nf->addr);
            continue;
        }

        NETC_NVTX_PUSH("GB:net_heartbeat", NETC_NVTX_COLOR_HEALTH);
        struct gb_net_heartbeat req;
        memset(&req, 0, sizeof(req));
        req.timestamp_ms = now_ms;

        /* A3 (rewritten 2026-07-06): a failed heartbeat means the stream is
         * dead or desynced , EPIPE/reset, a lost response, or a seq mismatch
         * all leave the connection unusable for every other caller too.
         * Waiting out the 15 s staleness window just meant 7 more guaranteed
         * failures while allocs got "feeder not connected". Close and let N8
         * reconnect within ~500 ms instead. */
        int hb_dead = 0;
        struct gb_net_header hdr;
        void *pay = NULL;
        if (netc_send_msg(nf, GB_MSG_HEARTBEAT, 0, &req, sizeof(req)) < 0) {
            if (g_netc_debug)
                fprintf(stderr, "[GreenBoost-netc %d] heartbeat send failed: fd=%d errno=%d (%s)\n",
                        (int)getpid(), nf->fd, errno, strerror(errno));
            hb_dead = 1;
            NETC_NVTX_EVENT("HEARTBEAT_TIMEOUT", "NET", 0, i, nf->addr);
        } else if (netc_recv_response(nf, &hdr, &pay) < 0 || !pay) {
            if (g_netc_debug)
                fprintf(stderr, "[GreenBoost-netc %d] heartbeat recv failed: fd=%d errno=%d (%s)\n",
                        (int)getpid(), nf->fd, errno, strerror(errno));
            hb_dead = 1;
            NETC_NVTX_EVENT("HEARTBEAT_RECV_TIMEOUT", "NET", 0, i, nf->addr);
        }
        if (hb_dead) {
            nf->heartbeat_miss_count++;
            if (nf->health_state <= GB_HEALTH_DEGRADED) {
                fprintf(stderr, "[GreenBoost-netc] A3: feeder %s heartbeat failed → "
                        "reconnecting\n", nf->addr);
                nf->health_state = GB_HEALTH_DEGRADED;
                nf->health_state_since_ms = now_ms;
            }
            close(nf->fd);
            nf->fd = -1;
            nf->connected = 0;
            nf->reconnect_delay_ms = 500;
            nf->next_reconnect_ms  = now_ms + 500;
            pthread_mutex_unlock(&nf->per_lock);
            NETC_NVTX_POP();
            continue;
        }

        /* A3: heartbeat received successfully - reset miss counter */
        nf->last_heartbeat_ms   = now_ms;
        nf->heartbeat_miss_count = 0;
        netc_log("heartbeat OK fd=%d send_seq=%u", nf->fd, nf->send_seq);
        NETC_NVTX_EVENT("HEARTBEAT_OK", "NET", 0, i, nf->addr);

        /* Struct may be larger than old version - check payload length before reading new fields */
        const struct gb_net_heartbeat *hb = (const struct gb_net_heartbeat *)pay;
        size_t gpu_load_v2_off = offsetof(struct gb_net_heartbeat, gpu_load)
                                 + offsetof(__typeof__(hb->gpu_load[0]), throttle_reasons);

        uint32_t throttle_any = 0;
        uint32_t dbe_total    = 0;
        int      sbe_elevated = 0;

        if (hb->gpu_count > 0 && hdr.payload_len >= (uint32_t)(gpu_load_v2_off + 4)) {
            for (int g = 0; g < (int)hb->gpu_count && g < GB_NET_MAX_GPUS; g++) {
                throttle_any |= hb->gpu_load[g].throttle_reasons;
                dbe_total    += hb->gpu_load[g].ecc_dbe_count;
                if (hb->gpu_load[g].ecc_sbe_delta > 8)
                    sbe_elevated = 1;
            }
            nf->gpu_util_pct        = hb->gpu_load[0].gpu_util_pct;
            nf->gpu_mem_util_pct    = hb->gpu_load[0].mem_util_pct;
            nf->gpu_temp_c          = hb->gpu_load[0].gpu_temp_c;
            nf->gpu_power_draw_w    = hb->gpu_load[0].power_draw_w;
            nf->gpu_vram_free_bytes = hb->gpu_load[0].vram_free_bytes;
        }
        nf->throttle_reasons  = throttle_any;
        nf->ecc_dbe_count     = dbe_total;
        /* Quarantine T1 feeder alloc if SBE rate elevated or any DBE found */
        nf->t1_ecc_quarantine = (sbe_elevated || dbe_total > 0) ? 1 : 0;

        if (nf->t1_ecc_quarantine)
            fprintf(stderr, "[GreenBoost-netc] WARN: feeder %s ECC errors - T1 quarantined "
                    "(SBE_elevated=%d DBE=%u)\n", nf->addr, sbe_elevated, dbe_total);

        /* U4: tick health state machine after updating throttle/ECC fields */
        {
            struct timespec _ts;
            clock_gettime(CLOCK_MONOTONIC, &_ts);
            uint64_t now_ms = (uint64_t)_ts.tv_sec * 1000 + (uint64_t)_ts.tv_nsec / 1000000;
            gb_feeder_health_state_t prev = nf->health_state;
            gb_feeder_health_state_t next = prev;
            uint64_t elapsed = now_ms - nf->health_state_since_ms;

            if (prev == GB_HEALTH_DISABLED) {
                if (now_ms >= nf->disabled_until_ms) {
                    next = GB_HEALTH_HEALTHY;
                }
            } else if (dbe_total > 0 || (sbe_elevated && elapsed >= GB_FEEDER_UNHEALTHY_TTL_MS)) {
                next = GB_HEALTH_QUARANTINE;
            } else if (sbe_elevated || throttle_any) {
                if (elapsed >= GB_FEEDER_UNHEALTHY_TTL_MS)
                    next = GB_HEALTH_UNHEALTHY;
                else if (prev < GB_HEALTH_DEGRADED)
                    next = GB_HEALTH_DEGRADED;
            } else {
                /* No errors - recover toward HEALTHY */
                if (prev == GB_HEALTH_QUARANTINE) {
                    next = GB_HEALTH_DISABLED;
                    nf->disabled_until_ms = now_ms + GB_FEEDER_DISABLE_TTL_MS;
                } else if (prev > GB_HEALTH_HEALTHY && elapsed >= GB_FEEDER_DEGRADED_TTL_MS) {
                    next = GB_HEALTH_HEALTHY;
                }
            }
            if (next != prev) {
                fprintf(stderr, "[GreenBoost-netc] feeder %s health: %d→%d\n",
                        nf->addr, prev, next);
                nf->health_state          = next;
                nf->health_state_since_ms = now_ms;
                NETC_NVTX_EVENT("HEALTH_TRANSITION", "NET", next, i, nf->addr);

                /* U19: adjust MPS SM% on health state change (GREENBOOST_MPS=1) */
                static const char *mps_env = NULL;
                if (!mps_env) mps_env = getenv("GREENBOOST_MPS");
                if (mps_env && mps_env[0] == '1') {
                    struct gb_net_mps_set mps_req;
                    mps_req._pad   = 0;
                    /* Reduce SM% when degraded to cool the feeder; restore on recovery */
                    mps_req.sm_pct = (next >= GB_HEALTH_DEGRADED) ? 60u : 100u;
                    /* per_lock already held for the heartbeat block - no nested acquire */
                    netc_send_msg(nf, GB_MSG_CUDA_MPS_SET, 0,
                                  &mps_req, sizeof(mps_req));
                    fprintf(stderr, "[GreenBoost-netc] U19: sent MPS SM%%=%u to feeder %s\n",
                            mps_req.sm_pct, nf->addr);
                }
            }
        }

        /* U13: advance PID rate limiter for this feeder after health state updated */
        gb_netc_pid_tick(i);

        free(pay);
        pthread_mutex_unlock(&nf->per_lock);
        NETC_NVTX_POP();  /* GB:net_heartbeat */
    }
    pthread_mutex_unlock(&g_netc_lock);
}

int gb_netc_feeder_throttled(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    /* U4: treat anything ≥ DEGRADED as "throttled" for first-pass skip */
    return g_feeders[fi].health_state >= GB_HEALTH_DEGRADED
        || g_feeders[fi].throttle_reasons != 0;
}

int gb_netc_feeder_t1_quarantined(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    /* U4: quarantine or worse blocks T1 entirely */
    return g_feeders[fi].health_state >= GB_HEALTH_QUARANTINE
        || g_feeders[fi].t1_ecc_quarantine;
}

/* U4: check if feeder should be fully skipped (DISABLED state) */
int gb_netc_feeder_disabled(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    return g_feeders[fi].health_state == GB_HEALTH_DISABLED;
}

/* U4: expose health state for diagnostics */
int gb_netc_feeder_health_state(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return GB_HEALTH_HEALTHY;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    return (int)g_feeders[fi].health_state;
}

uint32_t gb_netc_feeder_pcie_bw_mbs(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    return g_feeders[fi].pcie_effective_bw_mbs;
}

/* U10: Send a batch of KV block events to a feeder for prefetch planning.
 * Fire-and-forget - no response expected from the feeder. */
int gb_netc_send_block_events(int remote_idx, const struct gb_net_block_events *msg)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return -1;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    struct netc_feeder *nf = &g_feeders[fi];
    if (!nf->connected || nf->fd < 0) return -1;

    pthread_mutex_lock(&nf->per_lock);
    int r = netc_send_msg(nf, GB_MSG_BLOCK_EVENTS, 0, msg,
                          (uint32_t)(offsetof(struct gb_net_block_events, events) +
                                     msg->count * sizeof(struct gb_net_block_event)));
    pthread_mutex_unlock(&nf->per_lock);
    return r;
}

/* U15: public accessor for pinned buffer pool free count (used by status metrics). */
int gb_netc_pinned_free_count(void)
{
    return gb_pinned_free_count();
}

/* U17: Select the best remote_idx for a new allocation of `size` bytes.
 * Prefers the feeder with the most T1 VRAM headroom that can satisfy the request
 * and is in a healthy state.  Falls back to the first reachable feeder.
 * Returns -1 if no feeder is available. */
int gb_netc_best_remote_for_alloc(uint64_t size)
{
    if (!g_netc_initialized || g_remote_gpu_count == 0) return -1;

    int      best_ri     = -1;
    uint64_t best_vram   = 0;

    for (int ri = 0; ri < g_remote_gpu_count; ri++) {
        int fi = g_remote_gpus[ri].feeder_idx;
        int gi = g_remote_gpus[ri].feeder_gpu_id;
        struct netc_feeder *nf = &g_feeders[fi];

        if (!nf->connected) continue;
        if (nf->health_state >= GB_HEALTH_QUARANTINE) continue;

        /* Use cached vram_bytes as a proxy for current T1 free.
         * This avoids a live MEM_INFO query on every alloc - acceptable
         * since the cache is refreshed by poll_health every few seconds. */
        uint64_t t1_avail = nf->gpus[gi].vram_bytes;
        if (t1_avail >= size && t1_avail > best_vram) {
            best_vram = t1_avail;
            best_ri   = ri;
        }
    }

    /* Fall back: first connected, non-quarantined feeder */
    if (best_ri < 0) {
        for (int ri = 0; ri < g_remote_gpu_count; ri++) {
            int fi = g_remote_gpus[ri].feeder_idx;
            struct netc_feeder *nf = &g_feeders[fi];
            if (nf->connected && nf->health_state < GB_HEALTH_QUARANTINE) {
                best_ri = ri;
                break;
            }
        }
    }

    return best_ri;
}

/* N7: Query live feeder stats via GB_MSG_FEEDER_STATUS.
 * Fills *out with T1/T2/T3 free/total bytes, kernel dispatch count, and MPS SM%.
 * Returns 0 on success, -1 if feeder is disconnected or response is malformed. */
int gb_netc_query_feeder_status(int remote_idx, struct gb_feeder_status_resp *out)
{
    NETC_NVTX_PUSH("GB:net_feeder_status", NETC_NVTX_COLOR_NET);
    if (!out || remote_idx < 0 || remote_idx >= g_remote_gpu_count) {
        NETC_NVTX_POP(); return -1;
    }
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    struct netc_feeder *nf = &g_feeders[fi];
    if (!nf->connected || nf->fd < 0) { NETC_NVTX_POP(); return -1; }

    pthread_mutex_lock(&nf->per_lock);
    int r = netc_send_msg(nf, GB_MSG_FEEDER_STATUS, 0, NULL, 0);
    if (r < 0) { pthread_mutex_unlock(&nf->per_lock); NETC_NVTX_POP(); return -1; }

    struct gb_net_header resp_hdr;
    void *resp_payload = NULL;
    r = netc_recv_response(nf, &resp_hdr, &resp_payload);
    pthread_mutex_unlock(&nf->per_lock);

    if (r < 0 || !resp_payload) { free(resp_payload); NETC_NVTX_POP(); return -1; }
    if (resp_hdr.payload_len < sizeof(struct gb_feeder_status_resp)) {
        free(resp_payload); NETC_NVTX_POP(); return -1;
    }
    memcpy(out, resp_payload, sizeof(struct gb_feeder_status_resp));
    free(resp_payload);
    NETC_NVTX_EVENT("FEEDER_STATUS_QUERY", "NET", 0, remote_idx, nf->addr);
    NETC_NVTX_POP();
    return (out->status == GB_STATUS_OK) ? 0 : -1;
}

/* A4/U20: Return the highest EWMA-measured transfer bandwidth across all feeders (MB/s).
 * Used by the prefetch thread to size prefetch chunks to ~100 ms of link time. */
uint32_t gb_netc_best_feeder_bw_mbs(void)
{
    float best = 0.0f;
    for (int i = 0; i < g_feeder_count; i++) {
        if (!g_feeders[i].connected) continue;
        /* Prefer live EWMA; fall back to handshake PCIe BW */
        float bw = (g_feeders[i].bw_ewma_mbs > 0.0f)
                   ? g_feeders[i].bw_ewma_mbs
                   : (float)g_feeders[i].pcie_effective_bw_mbs;
        if (bw > best) best = bw;
    }
    return (uint32_t)best;
}

/* A2: expose wraparound generation counter for metrics. */
uint32_t gb_netc_fake_ptr_generation(void)
{
    return atomic_load_explicit(&g_fake_ptr_generation, memory_order_relaxed);
}

/* A3: expose heartbeat miss count for a specific feeder (for metrics/diagnostics). */
uint32_t gb_netc_heartbeat_miss_count(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    return g_feeders[fi].heartbeat_miss_count;
}

uint32_t gb_netc_feeder_gpu_util_pct(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    return g_feeders[fi].gpu_util_pct;
}

uint32_t gb_netc_feeder_gpu_mem_util_pct(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    return g_feeders[fi].gpu_mem_util_pct;
}

uint16_t gb_netc_feeder_gpu_temp_c(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    return g_feeders[fi].gpu_temp_c;
}

uint16_t gb_netc_feeder_gpu_power_w(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    return g_feeders[fi].gpu_power_draw_w;
}

/* Telemetry P0-A: raw NVML clock-throttle bitmask cached from the last
 * heartbeat.  Unlike gb_netc_feeder_throttled() this does NOT fold in the
 * health state - it is the verbatim nvmlDeviceGetCurrentClocksThrottleReasons
 * value from the feeder, for dataflux/metrics attribution. */
uint32_t gb_netc_feeder_throttle_reasons(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    return g_feeders[fi].throttle_reasons;
}

/* Telemetry P0-A: cumulative double-bit ECC error count cached from the
 * last heartbeat (sum across the feeder's GPUs). */
uint32_t gb_netc_feeder_ecc_dbe_count(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    return g_feeders[fi].ecc_dbe_count;
}

/* Telemetry P0-A: negotiated PCIe link generation from the handshake
 * (0 = unknown / old feeder that sent a short handshake reply). */
uint32_t gb_netc_feeder_pcie_link_gen(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    return g_feeders[fi].pcie_link_gen;
}

/* Telemetry P0-A: negotiated PCIe link width from the handshake (0 = unknown). */
uint32_t gb_netc_feeder_pcie_link_width(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    return g_feeders[fi].pcie_link_width;
}

/* Telemetry P0-A follow-up: live free VRAM (bytes) from the last heartbeat's
 * gpu_load[0] (0 = unknown / pre-heartbeat). Completes feeder VRAM used/free
 * visibility in metrics.json → dataflux snapshots. */
uint64_t gb_netc_feeder_vram_free_bytes(int remote_idx)
{
    if (remote_idx < 0 || remote_idx >= g_remote_gpu_count) return 0;
    int fi = g_remote_gpus[remote_idx].feeder_idx;
    return g_feeders[fi].gpu_vram_free_bytes;
}

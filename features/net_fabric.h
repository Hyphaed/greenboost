/* SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
 * GreenBoost v3.0 - Network Fabric Protocol Definitions
 *
 * Shared between greenboost_netd (feeder daemon) and the CUDA shim (host client).
 * Wire format: little-endian, binary.  TCP transport on port GB_NET_PORT.
 *
 * Author  : Ferran Duarri
 * License : GPL v2 (open-source) / Commercial - see LICENSE
 */
#ifndef GREENBOOST_NET_FABRIC_H
#define GREENBOOST_NET_FABRIC_H

#ifdef __KERNEL__
# include <linux/types.h>
  typedef __u8  gb_u8;
  typedef __u16 gb_u16;
  typedef __u32 gb_u32;
  typedef __u64 gb_u64;
  typedef __s32 gb_s32;
#else
# include <stdint.h>
# include <endian.h>
  typedef uint8_t  gb_u8;
  typedef uint16_t gb_u16;
  typedef uint32_t gb_u32;
  typedef uint64_t gb_u64;
  typedef int32_t  gb_s32;
/* PR-MM: big-endian build is unblocked but requires explicit opt-in
 * because field-by-field migration to the GB_LE_* accessors below is
 * incremental.  Production deployments are all little-endian today;
 * we don't ship to BE without further work.
 *
 * To attempt a BE build, set GREENBOOST_BE_INCOMPLETE=1 in CFLAGS.
 * The build will succeed; runtime behaviour against an LE peer is
 * UNDEFINED until every wire field access uses the GB_LE_* accessors. */
# if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__ && !defined(GREENBOOST_BE_INCOMPLETE)
#  error "GreenBoost net_fabric on big-endian host requires opt-in: " \
         "compile with -DGREENBOOST_BE_INCOMPLETE=1 AND verify that " \
         "every gb_u32/gb_u64 field access in netd/netc uses the " \
         "GB_LE_U16/U32/U64 accessors defined below."
# endif
#endif

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

#define GB_NET_MAGIC        0x47424E46u  /* "GBNF" - GreenBoost Net Fabric */
#define GB_NET_PORT         9740
/* PROTO_VER bumped to 3 in audit-Wave-23: seq_num field added to gb_net_header
 * for within-session replay/reordering detection.  v2 and below are rejected
 * at handshake (hard-fail on version mismatch since Wave-1). */
#define GB_NET_PROTO_VER    3

#define GB_NET_MAX_GPU_NAME 64
#define GB_NET_MAX_HOSTNAME 64
#define GB_NET_MAX_GPUS     8
#define GB_NET_MAX_FEEDERS  16
#define GB_NET_MAX_KERNEL_NAME 256

/* Single source of truth for max wire-message size; both netd RECV_BUF_SIZE
 * and netc NETC_RECV_BUF must honour this so the two sides cannot frame-desync.
 * 4 MiB matches the existing netd default. */
#define GB_NET_MAX_MSG_SIZE (4u * 1024u * 1024u)

/* Per-EXEC upload cap - total bytes of all inline upload_data combined.
 * Anything larger should be done as a CUDA_MALLOC + MEMCPY_H2D pair. */
#define GB_NET_MAX_EXEC_UPLOAD_BYTES (256u * 1024u * 1024u)

#define GB_NET_HEARTBEAT_INTERVAL_MS  5000
#define GB_NET_HEARTBEAT_TIMEOUT_MS  15000

/* Little-endian portability helpers for wire fields.  The wire format is
 * documented LE; on LE hosts these are no-ops.  Big-endian hosts get correct
 * conversion automatically when callers funnel reads through gb_le*_to_host. */
#ifndef __KERNEL__
# if defined(__has_include)
#  if __has_include(<endian.h>)
#   include <endian.h>
#  endif
# endif
# ifndef le32toh
#  if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#   define le16toh(x) (x)
#   define le32toh(x) (x)
#   define le64toh(x) (x)
#   define htole16(x) (x)
#   define htole32(x) (x)
#   define htole64(x) (x)
#  else
#   define le16toh(x) __builtin_bswap16(x)
#   define le32toh(x) __builtin_bswap32(x)
#   define le64toh(x) __builtin_bswap64(x)
#   define htole16(x) __builtin_bswap16(x)
#   define htole32(x) __builtin_bswap32(x)
#   define htole64(x) __builtin_bswap64(x)
#  endif
# endif
#endif

/* PR-MM: byte-order-safe accessors for wire-protocol struct fields.
 *
 * Usage at every read of a multi-byte wire field:
 *     uint64_t handle = GB_LE_U64(req->remote_handle);
 *     uint32_t size   = GB_LE_U32(req->size);
 *
 * Usage at every write into a wire field:
 *     resp->remote_handle = GB_LE_PUT_U64(handle);
 *     resp->size          = GB_LE_PUT_U32(size);
 *
 * On LE hosts (current production), the macros compile to nothing.  On
 * BE hosts (future port), they byte-swap.  Both directions are necessary
 * because the wire format is documented LE; struct fields stored as
 * uint*_t hold the on-wire bits which happen to match host order on LE
 * but not on BE.
 *
 * Migration plan: convert direct field access (req->size etc.) to
 * GB_LE_U32(req->size) at every wire-touching site.  Use this header
 * for one source of truth - the accessors stay byte-order-correct as
 * the codebase evolves. */
#ifdef __KERNEL__
/* Kernel side uses Linux's standard accessors; provide GB_LE_* aliases
 * for source consistency with userspace. */
# define GB_LE_U16(x)      le16_to_cpu((__le16)(x))
# define GB_LE_U32(x)      le32_to_cpu((__le32)(x))
# define GB_LE_U64(x)      le64_to_cpu((__le64)(x))
# define GB_LE_PUT_U16(x)  ((__force gb_u16)cpu_to_le16(x))
# define GB_LE_PUT_U32(x)  ((__force gb_u32)cpu_to_le32(x))
# define GB_LE_PUT_U64(x)  ((__force gb_u64)cpu_to_le64(x))
#else
# define GB_LE_U16(x)      le16toh((uint16_t)(x))
# define GB_LE_U32(x)      le32toh((uint32_t)(x))
# define GB_LE_U64(x)      le64toh((uint64_t)(x))
# define GB_LE_PUT_U16(x)  htole16((uint16_t)(x))
# define GB_LE_PUT_U32(x)  htole32((uint32_t)(x))
# define GB_LE_PUT_U64(x)  htole64((uint64_t)(x))
#endif

/* ------------------------------------------------------------------ */
/*  Message types                                                      */
/* ------------------------------------------------------------------ */

enum gb_net_msg_type {
    /* Control plane */
    GB_MSG_HANDSHAKE_REQ    = 0x01,
    GB_MSG_HANDSHAKE_RESP   = 0x02,
    GB_MSG_HEARTBEAT        = 0x03,
    GB_MSG_DISCONNECT       = 0xFF,

    /* Memory operations */
    GB_MSG_CUDA_MALLOC      = 0x10,
    GB_MSG_CUDA_FREE        = 0x11,
    GB_MSG_CUDA_MEMCPY_H2D  = 0x12,
    GB_MSG_CUDA_MEMCPY_D2H  = 0x13,
    /* 0x14 intentionally vacant - see GB_MSG_CUDA_MEMCPY_D2D define below */

    /* Compute operations */
    GB_MSG_CUDA_LAUNCH      = 0x20,
    GB_MSG_CUDA_SYNC        = 0x21,
    GB_MSG_CUDA_REGISTER_FN = 0x22,
    /* Single-GPU cluster: kernel exec with pointer relocation + inline data xfer */
    GB_MSG_CUDA_EXEC        = 0x23,
    /* 0x24 = GB_MSG_CUDA_MEMCPY_D2D (defined as #define below) */
    /* Async fire-and-forget exec: same wire format as GB_MSG_CUDA_EXEC but
     * n_downloads must be 0.  Feeder ACKs immediately (before kernel completes);
     * results stay resident in feeder VRAM until the next GB_MSG_CUDA_SYNC. */
    GB_MSG_CUDA_EXEC_ASYNC  = 0x25,

    /* Query operations */
    GB_MSG_GPU_QUERY        = 0x30,
    GB_MSG_MEM_INFO         = 0x31,

    /* U10: KV block event stream - host pushes STORED/REMOVED events; feeder builds
     * block ancestry tree for async prefetch prediction. */
    GB_MSG_BLOCK_EVENTS     = 0x32,

    /* U19: Dynamic MPS SM% control - host requests feeder to adjust
     * CUDA_MPS_ACTIVE_THREAD_PERCENTAGE. No response expected (fire-and-forget).
     * Only meaningful when GREENBOOST_MPS=1 and nvidia-cuda-mps-control is running. */
    GB_MSG_CUDA_MPS_SET     = 0x33,

    /* N7: Query feeder live stats - no payload; feeder replies with gb_feeder_status_resp.
     * Used by cmd_cluster, gb_feeder_diag.py, and the health monitor. */
    GB_MSG_FEEDER_STATUS    = 0x34,

    /* Response wrapper */
    GB_MSG_RESPONSE         = 0x40,
};

/* Response status codes */
enum gb_net_status {
    GB_STATUS_OK            = 0,
    GB_STATUS_ERR_OOM       = 1,
    GB_STATUS_ERR_INVALID   = 2,
    GB_STATUS_ERR_CUDA      = 3,
    GB_STATUS_ERR_PROTO     = 4,
    GB_STATUS_ERR_REJECTED  = 5,
    GB_STATUS_ERR_NCCL      = 6,   /* F1: NCCL communicator init/send error   */
    GB_STATUS_ERR_THROTTLE  = 7,   /* N9: client alloc rate-limit exceeded     */
};

/* Allocation tier flags for gb_net_cuda_malloc.flags
 * TIER_AUTO: feeder cascades T1 → T2 → T3 until one succeeds.
 * TIER_T1/T2/T3: request exactly that tier; OOM if unavailable. */
#define GB_ALLOC_TIER_MASK  0x07u
#define GB_ALLOC_TIER_AUTO  0x00u  /* T1 → T2 → T3 cascade on feeder   */
#define GB_ALLOC_TIER_T1    0x01u  /* GPU VRAM only                     */
#define GB_ALLOC_TIER_T2    0x02u  /* Pinned host DDR (cudaHostAlloc)   */
#define GB_ALLOC_TIER_T3    0x04u  /* Pageable host RAM (swap-in on exec) */

/* Flags for message header */
#define GB_NET_FLAG_RESPONSE  (1u << 0)
#define GB_NET_FLAG_ASYNC     (1u << 1)

/* ------------------------------------------------------------------ */
/*  Wire header - precedes every message                               */
/* ------------------------------------------------------------------ */

struct gb_net_header {
    gb_u32 magic;        /* GB_NET_MAGIC                            */
    gb_u16 msg_type;     /* gb_net_msg_type                         */
    gb_u16 flags;        /* GB_NET_FLAG_*                           */
    gb_u32 payload_len;  /* bytes following this header              */
    gb_u32 seq_num;      /* monotonic per-connection counter; feeder
                          * rejects seq <= last seen (F-L3-09)      */
} __attribute__((packed));

#define GB_NET_HDR_SIZE  sizeof(struct gb_net_header)  /* 16 bytes */

/* ------------------------------------------------------------------ */
/*  Handshake                                                          */
/* ------------------------------------------------------------------ */

struct gb_net_gpu_info {
    gb_u32 gpu_id;
    gb_u64 vram_bytes;           /* T1 - GPU VRAM total                  */
    gb_u32 cc_major;
    gb_u32 cc_minor;
    gb_u64 ram_available_bytes;  /* T2 - system RAM total                */
    gb_u64 t3_bytes;             /* T3 - NVMe swap total (0 if none)     */
    char   name[GB_NET_MAX_GPU_NAME];
} __attribute__((packed));       /* 100 bytes per entry                  */

struct gb_net_handshake_req {
    gb_u32 proto_version;
    gb_u32 gpu_count;
    char   hostname[GB_NET_MAX_HOSTNAME];
    struct gb_net_gpu_info gpus[GB_NET_MAX_GPUS];
} __attribute__((packed));

struct gb_net_handshake_resp {
    gb_u32 status;          /* gb_net_status */
    gb_u32 feeder_id;       /* assigned by feeder, unique per session */
    gb_u32 proto_version;
    gb_u32 gpu_count;
    char   hostname[GB_NET_MAX_HOSTNAME];
    struct gb_net_gpu_info gpus[GB_NET_MAX_GPUS];
    /* v2 PCIe link info - zero on old feeders that send a short reply     */
    gb_u32 pcie_link_gen;           /* e.g. 4 for PCIe 4.0                */
    gb_u32 pcie_link_width;         /* e.g. 16 for ×16                    */
    gb_u32 pcie_effective_bw_mbs;   /* measured effective BW MB/s         */
    gb_u32 pcie_replay_count;       /* error indicator - non-zero = degraded link */
} __attribute__((packed));

/* ------------------------------------------------------------------ */
/*  Heartbeat                                                          */
/* ------------------------------------------------------------------ */

struct gb_net_heartbeat {
    gb_u64 timestamp_ms;     /* sender's monotonic clock              */
    gb_u32 gpu_count;
    gb_u32 t2_speed_mts;     /* DDR speed in MT/s                     */
    struct {
        gb_u64 vram_free_bytes;
        gb_u64 vram_used_bytes;
        gb_u32 gpu_util_pct;        /* 0-100                          */
        gb_u32 mem_util_pct;        /* 0-100                          */
        /* v2 health fields - zero on old feeders that send short reply */
        gb_u32 throttle_reasons;    /* NVML clock-throttle bitmask    */
        gb_u16 gpu_temp_c;          /* GPU temperature Celsius        */
        gb_u16 power_draw_w;        /* current power draw Watts       */
        gb_u16 power_limit_w;       /* TDP limit Watts                */
        gb_u16 ecc_sbe_delta;       /* SBE count delta this interval  */
        gb_u16 ecc_dbe_count;       /* cumulative DBE count           */
        gb_u16 _pad_health;
    } __attribute__((packed)) gpu_load[GB_NET_MAX_GPUS];
} __attribute__((packed));

/* ------------------------------------------------------------------ */
/*  CUDA Memory operations                                             */
/* ------------------------------------------------------------------ */

struct gb_net_cuda_malloc {
    gb_u64 size;
    gb_u32 flags;           /* GB_ALLOC_* flags                      */
    gb_u32 device_id;       /* which GPU on the feeder               */
} __attribute__((packed));

struct gb_net_cuda_malloc_resp {
    gb_u32 status;          /* gb_net_status                         */
    gb_u32 tier_used;       /* 0=T1_GPU 1=T2_DDR 2=T3_pageable       */
    gb_u64 remote_handle;   /* opaque handle for this allocation     */
} __attribute__((packed));

struct gb_net_cuda_free {
    gb_u64 remote_handle;
} __attribute__((packed));

struct gb_net_cuda_memcpy {
    gb_u64 remote_handle;
    gb_u64 offset;
    gb_u64 size;
    /* For H2D: payload data follows immediately after this struct.
     * For D2H: response carries the data payload. */
} __attribute__((packed));

/* ------------------------------------------------------------------ */
/*  CUDA Compute operations                                            */
/* ------------------------------------------------------------------ */

struct gb_net_cuda_launch {
    gb_u32 grid_x, grid_y, grid_z;
    gb_u32 block_x, block_y, block_z;
    gb_u32 shared_mem_bytes;
    gb_u32 stream_id;
    gb_u32 kernel_name_len;
    gb_u32 arg_buffer_size;
    /* Followed by:
     *   char kernel_name[kernel_name_len]  (NOT null-terminated in wire)
     *   char arg_buffer[arg_buffer_size]
     */
} __attribute__((packed));

struct gb_net_cuda_sync {
    gb_u32 stream_id;
    gb_u32 t2_speed_mts;  /* DDR speed in MT/s (replaced _pad) */
} __attribute__((packed));

struct gb_net_cuda_register_fn {
    gb_u32 kernel_name_len;
    gb_u32 t2_speed_mts;  /* DDR speed in MT/s (replaced _pad) */
    /* Followed by: char kernel_name[kernel_name_len] */
} __attribute__((packed));

/* ------------------------------------------------------------------ */
/*  Query operations                                                   */
/* ------------------------------------------------------------------ */

struct gb_net_gpu_query {
    gb_u32 device_id;
    gb_u32 attribute_id;     /* CUdevice_attribute enum value         */
} __attribute__((packed));

struct gb_net_gpu_query_resp {
    gb_u32 status;
    gb_s32 value;
} __attribute__((packed));

struct gb_net_mem_info {
    gb_u32 device_id;
    gb_u32 t2_speed_mts;  /* DDR speed in MT/s (replaced _pad) */
} __attribute__((packed));

struct gb_net_mem_info_resp {
    gb_u32 status;
    gb_u32 t2_speed_mts;  /* DDR speed in MT/s (replaced _pad) */
    gb_u64 free_bytes;       /* legacy: T1_free + T2_free                */
    gb_u64 total_bytes;      /* legacy: T1_total + T2_total              */
    /* v2 per-tier fields - zero on old feeders that send a short reply  */
    gb_u64 t1_free;          /* GPU VRAM free                            */
    gb_u64 t1_total;         /* GPU VRAM total                           */
    gb_u64 t2_free;          /* system RAM free (usable pool)            */
    gb_u64 t2_total;         /* system RAM total (usable pool)           */
    gb_u64 t3_free;          /* NVMe swap free (0 if not configured)     */
    gb_u64 t3_total;         /* NVMe swap total                          */
    /* v3 speed fields - zero on feeders that send a short reply (< 80 bytes) */
    gb_u32 t3_speed_mbs;     /* feeder NVMe sequential read speed MB/s   */
    gb_u32 _pad3;
} __attribute__((packed));

/* ------------------------------------------------------------------ */
/*  Generic response header                                            */
/* ------------------------------------------------------------------ */

struct gb_net_response {
    gb_u16 orig_msg_type;    /* which message this responds to        */
    gb_u16 _pad;
    gb_u32 status;           /* gb_net_status                         */
    /* Type-specific payload follows (e.g. gb_net_cuda_malloc_resp)    */
} __attribute__((packed));

/* ------------------------------------------------------------------ */
/*  Cluster config path                                                */
/* ------------------------------------------------------------------ */

#define GB_CLUSTER_CONF     "/etc/greenboost/cluster.conf"
#define GB_NETD_PID_FILE    "/run/greenboost/netd.pid"
#define GB_NETD_LOG_FILE    "/var/log/greenboost/netd.log"

/* ------------------------------------------------------------------ */
/*  Single-GPU cluster: CUDA_EXEC with pointer relocation              */
/* ------------------------------------------------------------------ */

#define GB_NET_MAX_RELOCS  64
#define GB_NET_MAX_XFERS   16

/* Arg pointer relocation: remote fake_ptr → feeder device ptr */
struct gb_net_ptr_reloc {
    gb_u32 arg_idx;       /* index into arg_vals[]              */
    gb_u32 t2_speed_mts;  /* DDR speed in MT/s (replaced _pad) */
    gb_u64 remote_handle; /* feeder's actual CUdeviceptr        */
} __attribute__((packed));  /* 16 bytes */

/* Inline data transfer descriptor for an arg */
struct gb_net_xfer_desc {
    gb_u32 arg_idx;       /* which arg_vals entry this data is for */
    gb_u32 t2_speed_mts;  /* DDR speed in MT/s (replaced _pad) */
    gb_u64 size;          /* data size in bytes                 */
} __attribute__((packed));  /* 16 bytes */

/* GB_MSG_CUDA_EXEC - single-GPU cluster kernel execution.
 * Host sends kernel_name + flat arg values + reloc table + upload data.
 * Feeder patches args, allocates temp buffers, executes, returns downloads. */
struct gb_net_cuda_exec {
    gb_u32 grid_x, grid_y, grid_z;
    gb_u32 block_x, block_y, block_z;
    gb_u32 shared_mem_bytes;
    gb_u32 kernel_name_len;
    gb_u32 n_arg_vals;    /* # of uint64_t arg values (one per kernel arg)  */
    gb_u32 n_relocs;      /* # of remote-ptr relocations                    */
    gb_u32 n_uploads;     /* # of local tensors to upload before exec       */
    gb_u32 n_downloads;   /* # of tensors to download after exec            */
    /* wire layout after this struct:
     *   char                    kernel_name[kernel_name_len]  (not null-terminated)
     *   uint64_t                arg_vals[n_arg_vals]
     *   struct gb_net_ptr_reloc relocs[n_relocs]
     *   struct gb_net_xfer_desc upload_descs[n_uploads]
     *   uint8_t                 upload_data[sum of upload sizes]
     *   struct gb_net_xfer_desc download_descs[n_downloads]
     */
} __attribute__((packed));

/* Phase 4: NCCL Initialization */
#define GB_MSG_NCCL_INIT 0x50

struct gb_net_nccl_init {
    gb_u8 nccl_id[128]; /* ncclUniqueId */
    gb_u32 rank;
    gb_u32 num_ranks;
} __attribute__((packed));


/* D2D transfer (NCCL-capable) - placed in the compute range alongside EXEC.
 * The 0x14 enum slot was vacated to avoid shadowing this define. */
#define GB_MSG_CUDA_MEMCPY_D2D 0x24

struct gb_net_cuda_memcpy_d2d {
    gb_u64 src_handle;  /* remote handle if pulling, 0 if pushing */
    gb_u64 dst_handle;  /* remote handle if pushing, 0 if pulling */
    gb_u64 size;
} __attribute__((packed));

/* ------------------------------------------------------------------ */
/*  U10: KV Block Events (vLLM kv_events.py pattern)                  */
/* ------------------------------------------------------------------ */

#define GB_BLOCK_EVT_STORED   0x01  /* KV block written to T2 pool        */
#define GB_BLOCK_EVT_REMOVED  0x02  /* KV block evicted from T2 pool      */

#define GB_BLOCK_EVENTS_MAX   64    /* events per message batch           */

struct gb_net_block_event {
    gb_u64 block_hash;      /* 64-bit xxhash of first 256 bytes            */
    gb_u64 parent_hash;     /* 0 if root; links blocks into prefix tree    */
    gb_u32 num_tokens;      /* cumulative token count up to this block     */
    gb_u32 timestamp;       /* server mono_sec at emission                 */
    gb_u8  event_type;      /* GB_BLOCK_EVT_STORED or GB_BLOCK_EVT_REMOVED */
    gb_u8  _pad[7];
} __attribute__((packed));   /* 32 bytes per event */

/* Host → feeder: push a batch of block events for prefetch planning.
 * No response expected (fire-and-forget). */
struct gb_net_block_events {
    gb_u32 count;            /* number of events in events[] (≤ GB_BLOCK_EVENTS_MAX) */
    gb_u32 _pad;
    struct gb_net_block_event events[GB_BLOCK_EVENTS_MAX];
} __attribute__((packed));

/* ------------------------------------------------------------------ */
/*  U19: Dynamic MPS SM% control                                       */
/* ------------------------------------------------------------------ */

/* Host → feeder: adjust CUDA_MPS_ACTIVE_THREAD_PERCENTAGE.
 * Fire-and-forget; feeder writes the new % and signals MPS daemon.
 * Only effective when GREENBOOST_MPS=1 on feeder. */
struct gb_net_mps_set {
    gb_u32 sm_pct;    /* new SM% (1–100); 0 = restore to 100% */
    gb_u32 _pad;
} __attribute__((packed));

/* ------------------------------------------------------------------ */
/*  N7: Feeder live-status response (GB_MSG_FEEDER_STATUS)             */
/* ------------------------------------------------------------------ */

/* Host → feeder: no payload (just the header with msg_type=GB_MSG_FEEDER_STATUS).
 * Feeder responds via GB_MSG_RESPONSE with this struct as payload. */
struct gb_feeder_status_resp {
    /* v3.0 fields — always present */
    gb_u32 status;               /* GB_STATUS_OK or error                  */
    gb_u32 mps_sm_pct;           /* current MPS SM% (0 = not active)       */
    gb_u64 t1_free_bytes;        /* GPU VRAM free                          */
    gb_u64 t1_total_bytes;       /* GPU VRAM total                         */
    gb_u64 t2_free_bytes;        /* System RAM free                        */
    gb_u64 t2_total_bytes;       /* System RAM total                       */
    gb_u64 t3_free_bytes;        /* NVMe swap free (0 if not configured)   */
    gb_u64 t3_total_bytes;       /* NVMe swap total                        */
    gb_u32 kernel_dispatch_count; /* total kernel dispatches since start   */
    gb_u32 _pad;
    /* v3.1 GPU telemetry — host checks payload_len >= GB_FEEDER_STATUS_V31_SIZE
     * before reading; older netd versions send the shorter v3.0 struct only.  */
    gb_u16 gpu_temp_c;           /* GPU temperature °C (GPU 0)             */
    gb_u16 gpu_power_w;          /* current power draw Watts (GPU 0)       */
    gb_u32 gpu_util_pct;         /* SM compute utilization 0-100 (GPU 0)   */
    gb_u32 ecc_dbe_count;        /* cumulative double-bit ECC errors        */
    gb_u32 throttle_reasons;     /* NVML clock-throttle bitmask             */
    gb_u32 _pad2;
} __attribute__((packed));

/* Minimum payload size that includes the v3.1 GPU telemetry extension.
 * Host should check: hdr.payload_len >= GB_FEEDER_STATUS_V31_SIZE */
#define GB_FEEDER_STATUS_V31_SIZE  \
    (offsetof(struct gb_feeder_status_resp, _pad2) + sizeof(gb_u32))

#endif /* GREENBOOST_NET_FABRIC_H */


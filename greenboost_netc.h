/* SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
 * GreenBoost v3.2 - Network Client (host-side, compiled into CUDA shim)
 *
 * Manages TCP connections to feeder daemons, remote device tracking,
 * fake pointer mapping, and CUDA operation forwarding.
 *
 * Author  : Ferran Duarri
 * License : GPL v2 (open-source) / Commercial - see LICENSE
 */
#ifndef GREENBOOST_NETC_H
#define GREENBOOST_NETC_H

#include <stddef.h>
#include <stdint.h>
#include "features/net_fabric.h"  /* struct gb_net_block_events */

/* ------------------------------------------------------------------ */
/*  Remote pointer sentinel                                            */
/* ------------------------------------------------------------------ */

/* Remote device pointers live in this address range so we can distinguish
 * them from real local CUDA pointers without a hash lookup on every call.
 * Range: 0xAA00_0000_0000 .. 0xAAFF_FFFF_FFFF  (1 TB virtual, never mapped) */
#define GB_REMOTE_PTR_BASE  0xAA0000000000ULL
#define GB_REMOTE_PTR_MASK  0xFF0000000000ULL  /* top byte only → covers 1 TB range */

static inline int gb_is_remote_ptr(uint64_t ptr)
{
    return (ptr & GB_REMOTE_PTR_MASK) == GB_REMOTE_PTR_BASE;
}

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                          */
/* ------------------------------------------------------------------ */

/* Connect to all feeders listed in /etc/greenboost/cluster.conf.
 * Called once from gb_shim_init().  Thread-safe (internal mutex). */
int  gb_netc_init(void);

/* Disconnect all feeders.  Called from shim destructor. */
void gb_netc_cleanup(void);

/* ------------------------------------------------------------------ */
/*  Device queries                                                     */
/* ------------------------------------------------------------------ */

/* Number of remote GPUs across all connected feeders */
int  gb_netc_remote_gpu_count(void);

/* Total VRAM (bytes) for remote device at logical index (0-based within remotes) */
uint64_t gb_netc_remote_vram(int remote_idx);

/* Compute capability for remote device */
int  gb_netc_remote_cc_major(int remote_idx);
int  gb_netc_remote_cc_minor(int remote_idx);

/* GPU name (returns pointer to internal static storage) */
const char *gb_netc_remote_name(int remote_idx);

/* Feeder address string "ip:port" (returns pointer to internal static storage) */
const char *gb_netc_feeder_addr(int remote_idx);

/* cuMemGetInfo equivalent for remote device (combined T1+T2) */
int  gb_netc_mem_info(int remote_idx, uint64_t *free_bytes, uint64_t *total_bytes);

/* T1-only query - returns feeder GPU VRAM free/total.
 * Uses per-tier fields from gb_net_mem_info_resp v2; falls back to
 * the combined field on older feeders that send a short reply. */
int  gb_netc_t1_mem_info(int remote_idx, uint64_t *t1_free, uint64_t *t1_total);

/* Get T2 DDR speed in MT/s for remote device (cached from connect time) */
int  gb_netc_t2_speed_mts(int remote_idx);

/* Get T3 NVMe speed in MB/s for remote device (cached from connect time) */
int  gb_netc_t3_speed_mbs(int remote_idx);

/* T2-only query - returns feeder DDR free/total. */
int  gb_netc_t2_mem_info(int remote_idx, uint64_t *t2_free, uint64_t *t2_total);

/* T3-only query - returns feeder NVMe swap free/total. */
int  gb_netc_t3_mem_info(int remote_idx, uint64_t *t3_free, uint64_t *t3_total);

/* Sum of T2 system RAM across all connected feeders.
 * Used by the sysinfo() LD_PRELOAD hook to inflate freeram/totalram so
 * Ollama's system-RAM pre-flight check accounts for cluster capacity. */
int  gb_netc_total_remote_t2_bytes(uint64_t *free_bytes, uint64_t *total_bytes);

/* cuDeviceGetAttribute forwarding for remote device */
int  gb_netc_device_get_attribute(int remote_idx, int attrib, int *value);

/* ------------------------------------------------------------------ */
/*  Memory operations                                                  */
/* ------------------------------------------------------------------ */

/* Allocate on remote (any tier, feeder decides).  Returns a fake local
 * pointer in the GB_REMOTE_PTR range. */
int  gb_netc_malloc(int remote_idx, uint64_t size, uint32_t flags,
                    uint64_t *fake_ptr_out);

/* Allocate on remote, requesting a specific tier (GB_ALLOC_TIER_T1/T2/T3).
 * The feeder cascades on AUTO; OOMs for exact-tier requests that fail. */
int  gb_netc_malloc_tier(int remote_idx, uint64_t size, uint8_t tier,
                         uint64_t *fake_ptr_out);

/* Free a remote allocation identified by fake_ptr. */
int  gb_netc_free(uint64_t fake_ptr);

/* Copy host→device (upload).  fake_dst is a remote pointer. */
int  gb_netc_memcpy_h2d(uint64_t fake_dst, const void *host_src, uint64_t size);
int  gb_netc_memset(uint64_t fake_dst, int value, uint64_t size);
int  gb_netc_ensure_connected(int remote_idx);
int  gb_netc_wait_connected(int timeout_ms);

/* Copy device→host (download).  fake_src is a remote pointer. */
int  gb_netc_memcpy_d2h(void *host_dst, uint64_t fake_src, uint64_t size);

/* Copy device→device using NCCL. local_ptr is the local device pointer. */
int  gb_netc_memcpy_d2d_push(uint64_t fake_dst, const void *local_src, uint64_t size);
int  gb_netc_memcpy_d2d_pull(void *local_dst, uint64_t fake_src, uint64_t size);

/* ------------------------------------------------------------------ */
/*  Kernel launch                                                      */
/* ------------------------------------------------------------------ */

/* Register a kernel name for later forwarding.
 * Called from __cudaRegisterFunction hook. */
void gb_netc_register_kernel(const void *host_func, const char *device_name);

/* Lookup kernel name from host function pointer.
 * Returns NULL if not found. */
const char *gb_netc_lookup_kernel(const void *host_func);

/* Launch a kernel on a remote device.
 * arg_buffer + arg_buffer_size: flat serialized kernel arguments.
 * Returns 0 on success, CUDA error on failure. */
int  gb_netc_launch_kernel(int remote_idx,
                           const char *kernel_name,
                           unsigned int gx, unsigned int gy, unsigned int gz,
                           unsigned int bx, unsigned int by, unsigned int bz,
                           unsigned int shared_mem,
                           uint32_t stream_id,
                           const void *arg_buffer, uint32_t arg_buffer_size);

/* Synchronize a remote stream. */
int  gb_netc_stream_sync(int remote_idx, uint32_t stream_id);

/* F-L1-31: sync only feeders that dispatched a kernel since last sync. */
void gb_netc_selective_stream_sync(void);

/* ------------------------------------------------------------------ */
/*  Active device tracking                                             */
/* ------------------------------------------------------------------ */

/* Set the currently active remote device index.
 * -1 = local device is active (no remote routing).
 * Called from cudaSetDevice hook. */
void gb_netc_set_active_remote(int remote_idx);

/* Get the currently active remote device index.
 * Returns -1 if local device is active. */
int  gb_netc_get_active_remote(void);

/* Check if the network client is initialized and has connected feeders */
int  gb_netc_is_active(void);

/* ------------------------------------------------------------------ */
/*  Single-GPU cluster: alloc info + remote kernel exec               */
/* ------------------------------------------------------------------ */

/* Find the allocation that contains ptr (ptr may be base + offset).
 * Returns 0 on success, -1 if not found. */
int  gb_netc_get_alloc_info(uint64_t ptr, uint64_t *remote_handle,
                            uint64_t *base, uint64_t *alloc_size, int *feeder_idx);

/* Release the inflight exec reference acquired by gb_netc_get_alloc_info.
 * Must be called after the exec payload has been sent (F-L3-12). */
void gb_netc_alloc_release_ref(uint64_t ptr);

/* Per-arg relocation entry: a remote fake_ptr in arg_vals that must be
 * replaced with the feeder's actual device pointer before launch. */
struct gb_exec_reloc {
    int      arg_idx;
    uint64_t remote_handle;
    int      feeder_idx;
};

/* Per-arg upload/download entry: a local CUDA tensor that must be
 * uploaded to the feeder before exec and downloaded back after. */
struct gb_exec_upload {
    int     arg_idx;
    size_t  size;
    void   *host_data;        /* host copy of the tensor (caller fills before call) */
    void   *downloaded_data;  /* feeder's result (filled by gb_netc_exec_kernel)    */
};

/* Execute a CUDA kernel on a specific feeder using GB_MSG_CUDA_EXEC.
 * Builds the wire message, sends it, waits for response, and fills
 * uploads[i].downloaded_data for each upload after exec completes.
 * Caller must free downloaded_data pointers. */
int  gb_netc_exec_kernel(int feeder_idx,
                         const char *kernel_name,
                         unsigned int gx, unsigned int gy, unsigned int gz,
                         unsigned int bx, unsigned int by, unsigned int bz,
                         uint32_t shared_mem,
                         const uint64_t *arg_vals, uint32_t n_arg_vals,
                         const struct gb_exec_reloc *relocs, int n_relocs,
                         struct gb_exec_upload *uploads, int n_uploads,
                         int n_downloads);

/* Byte-offset relocation for the RAW exec path: a remote fake_ptr sitting at
 * byte offset `buf_offset` in the packed param buffer (may be inside a struct
 * param) → feeder device ptr. */
struct gb_exec_reloc_raw {
    uint32_t buf_offset;
    uint64_t remote_handle;
    int      feeder_idx;
};

/* RAW exec: transmit the full packed param buffer + per-param sizes so struct-
 * by-value args survive and their embedded pointers relocate.  launch_mode:
 * 0=driver CUfunction, 1=runtime host stub (feeder decides via resolution). */
int  gb_netc_exec_kernel_raw(int feeder_idx,
                             const char *kernel_name,
                             unsigned int gx, unsigned int gy, unsigned int gz,
                             unsigned int bx, unsigned int by, unsigned int bz,
                             uint32_t shared_mem,
                             const uint8_t *param_buf, uint32_t param_buf_bytes,
                             const uint32_t *param_sizes, uint32_t n_params,
                             const struct gb_exec_reloc_raw *relocs, int n_relocs);

/* Phase 2b: async fire-and-forget kernel dispatch (GREENBOOST_ASYNC_DISPATCH=1).
 * Sends GB_MSG_CUDA_EXEC_ASYNC and returns as soon as the feeder ACKs.
 * n_uploads and n_downloads are always 0 (weights must already reside in feeder
 * VRAM via prior gb_netc_malloc; results stay there until next stream sync). */
int  gb_netc_exec_kernel_async(int feeder_idx,
                               const char *kernel_name,
                               unsigned int gx, unsigned int gy, unsigned int gz,
                               unsigned int bx, unsigned int by, unsigned int bz,
                               uint32_t shared_mem,
                               const uint64_t *arg_vals, uint32_t n_arg_vals,
                               const struct gb_exec_reloc *relocs, int n_relocs);

/* ------------------------------------------------------------------ */
/*  D1/D2/D3: Feeder health and PCIe link queries                      */
/* ------------------------------------------------------------------ */

/* Send heartbeats to all feeders, cache throttle/ECC state.
 * Non-blocking (trylock) - skips if lock is contended. */
void gb_netc_poll_health(void);

/* Returns non-zero if the feeder's GPU is currently clock-throttled.
 * Read from the last gb_netc_poll_health() result (cached). */
int  gb_netc_feeder_throttled(int remote_idx);

/* Returns non-zero if the feeder's T1 VRAM is ECC-quarantined (elevated SBE or any DBE).
 * When true, gb_try_feeder_alloc_tier skips T1 for this feeder. */
int  gb_netc_feeder_t1_quarantined(int remote_idx);

/* Returns effective PCIe bandwidth in MB/s for the feeder link.
 * Used by gb_smart_overflow_alloc() to weight feeder T2 vs local T2. */
uint32_t gb_netc_feeder_pcie_bw_mbs(int remote_idx);

/* U4: Returns non-zero if feeder is in DISABLED state (all tiers blocked). */
int  gb_netc_feeder_disabled(int remote_idx);

/* U4: Returns raw health state (0=HEALTHY … 4=DISABLED). */
int  gb_netc_feeder_health_state(int remote_idx);

/* U10: Send a batch of KV block events to a feeder for prefetch planning. */
int  gb_netc_send_block_events(int remote_idx, const struct gb_net_block_events *msg);

/* U15: Returns number of free pinned staging buffers (for status metrics). */
int  gb_netc_pinned_free_count(void);

/* U17: Select the best remote_idx for a new alloc of `size` bytes.
 * Prefers healthy feeders with the most T1 VRAM headroom; falls back to
 * the first reachable feeder.  Returns -1 if no feeder available. */
int  gb_netc_best_remote_for_alloc(uint64_t size);

/* A4/U20: Return the highest EWMA-measured bandwidth across all feeders (MB/s).
 * Used by prefetch thread to size prefetch chunks to ~100 ms of link time. */
uint32_t gb_netc_best_feeder_bw_mbs(void);

/* A2: Return the fake-pointer wraparound generation counter (increments on each wrap). */
uint32_t gb_netc_fake_ptr_generation(void);

/* A3: Return the consecutive heartbeat miss count for a remote device. */
uint32_t gb_netc_heartbeat_miss_count(int remote_idx);

/* Return GPU utilization (0-100%) cached from the last heartbeat for a remote device. */
uint32_t gb_netc_feeder_gpu_util_pct(int remote_idx);
/* Return memory controller utilization (0-100%) from the last heartbeat. */
uint32_t gb_netc_feeder_gpu_mem_util_pct(int remote_idx);
/* Return GPU temperature (°C) cached from the last heartbeat; 0 = no data yet. */
uint16_t gb_netc_feeder_gpu_temp_c(int remote_idx);
/* Return GPU power draw (Watts) cached from the last heartbeat; 0 = no data yet. */
uint16_t gb_netc_feeder_gpu_power_w(int remote_idx);

/* N7: Query live feeder stats (T1/T2/T3 free/total, kernel count, MPS SM%).
 * Returns 0 on success, -1 if feeder is disconnected or the query failed. */
int gb_netc_query_feeder_status(int remote_idx, struct gb_feeder_status_resp *out);

#endif /* GREENBOOST_NETC_H */

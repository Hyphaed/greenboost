/* SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
 * GreenBoost v2.8 - Shared IOCTL definitions (kernel + userspace)
 *
 * Works with both #include <linux/ioctl.h> (kernel) and <sys/ioctl.h> (user).
 *
 * Author  : Ferran Duarri
 * License : GPL v2 (open-source) / Commercial - see LICENSE
 */
#ifndef GREENBOOST_IOCTL_H
#define GREENBOOST_IOCTL_H

#ifdef __KERNEL__
# include <linux/ioctl.h>
# include <linux/types.h>
  typedef __u64 gb_u64;
  typedef __u32 gb_u32;
  typedef __s32 gb_s32;
#else
# include <sys/ioctl.h>
# include <stdint.h>
  typedef uint64_t gb_u64;
  typedef uint32_t gb_u32;
  typedef int32_t  gb_s32;
#endif

/* Allocation flags - stored in gb_alloc_req.flags */
#define GB_ALLOC_WEIGHTS     (1u << 0)  /* model weight tensor              */
#define GB_ALLOC_KV_CACHE    (1u << 1)  /* KV cache - never spills to T3,
					 * auto-frozen in T2 LRU             */
#define GB_ALLOC_ACTIVATIONS (1u << 2)  /* ephemeral activation buffer      */
#define GB_ALLOC_FROZEN      (1u << 3)  /* never evict from T2              */
#define GB_ALLOC_NO_HUGEPAGE (1u << 4)  /* force 4K (for T3-spillable)      */
#define GB_ALLOC_T1_PRIORITY (1u << 5)  /* T1-priority: marks buf as KV-like;
					 * moves to LRU head, sets t1_priority
					 * bit - weight bufs evicted first    */
/* bit 6: reserved (GB_ALLOC_KV_COMPRESSED removed) */
#define GB_ALLOC_SESSION_PROTECTED (1u << 7) /* session-protected: skipped by
					 * auto-eviction until all unprotected
					 * candidates are exhausted             */

/* Allocate a pinned system RAM buffer; returns a DMA-BUF fd the GPU can import */
struct gb_alloc_req {
	gb_u64 size;    /* bytes to allocate          (in)  */
	gb_s32 fd;      /* DMA-BUF fd returned        (out) */
	gb_u32 flags;   /* GB_ALLOC_* flags           (in)  */
};

/* Pool statistics - three-tier memory hierarchy */
struct gb_info {
	/* Tier 1 - GPU VRAM (physical, managed by NVIDIA driver) */
	gb_u64 vram_physical_mb;   /* RTX 5070 physical VRAM             */

	/* Tier 2 - system RAM pool (pinned pages, DMA-BUF exported) */
	gb_u64 total_ram_mb;       /* total system RAM                   */
	gb_u64 free_ram_mb;        /* currently free RAM                 */
	gb_u64 allocated_mb;       /* bytes pinned by GreenBoost (T2)    */
	gb_u64 max_pool_mb;        /* virtual_vram_gb cap (T2)           */
	gb_u64 safety_reserve_mb;  /* always kept free (safety_reserve_gb) */
	gb_u64 available_mb;       /* free_ram - reserve - allocated     */
	gb_u32 active_buffers;     /* live DMA-BUF objects               */
	gb_u32 oom_active;         /* 1 if safety guard is triggered     */

	/* Tier 3 - NVMe swap (kernel-managed, model page overflow) */
	gb_u64 nvme_swap_total_mb; /* configured NVMe swap capacity      */
	gb_u64 nvme_swap_used_mb;  /* swap currently in use              */
	gb_u64 nvme_swap_free_mb;  /* swap available for model pages     */
	gb_u64 nvme_t3_allocated_mb; /* GreenBoost T3 allocations        */
	gb_u32 swap_pressure;      /* 0=ok 1=warn(>75%) 2=critical(>90%) */
	gb_u32 kv_reserve_mb;      /* T1 VRAM reserved for KV cache (MB) */

	/* KV cache usage tracking (GB_ALLOC_KV_CACHE allocations) */
	gb_u32 kv_used_mb;         /* KV bytes in any GreenBoost tier (T2 or T3
				    * safety-net); set by ExLlamaV3 native path  */
	gb_u32 kv_t2_mb;           /* KV bytes specifically in T2 DDR (subset of
				    * kv_used_mb; T3 never allowed for KV)       */

	gb_u32 t2_pressure;         /* 0=ok 1=warn(>75%) 2=critical(>90%) T2 pool   */

	/* Phase reset sequencing - shim polls this to detect model-swap resets.
	 * Incremented each time GB_IOCTL_RESET_PHASE is called (by Synapse CLI or
	 * the shim itself).  Shim compares against its cached value on every
	 * GB_KV_REFRESH_INTERVAL alloc boundary; a change triggers a phase reset. */
	gb_u32 phase_reset_seq;
	gb_u32 gaming_mode;         /* 1 when a game session is active          */

	/* Combined view */
	gb_u64 total_combined_mb;  /* VRAM + System DDR pool + NVMe swap       */
};

/* Madvise request - advise the kernel on buffer eviction priority */
struct gb_madvise_req {
	gb_s32 buf_id;  /* buffer id from gb_alloc_req.fd → IDR lookup */
	gb_u32 advise;  /* GB_MADVISE_* constant                       */
};
#define GB_MADVISE_COLD      0   /* demote in LRU (evict sooner)              */
#define GB_MADVISE_HOT       1   /* promote to LRU head (evict later)         */
#define GB_MADVISE_FREEZE    2   /* pin - never evict while frozen            */
#define GB_MADVISE_T1_PREFER 3   /* mark as T1-priority (KV-like); moves to
				  * LRU head, sets t1_priority - weight bufs
				  * are evicted before T1-priority bufs          */
#define GB_MADVISE_SESSION_PROTECT 4 /* protect this buffer from auto-eviction
				  * until all unprotected candidates are gone    */
#define GB_MADVISE_SESSION_DEMOTE  5 /* remove session-protection from buffer */

/* Evict request - push a T2 buffer to T3 (NVMe swap) immediately */
struct gb_evict_req {
	gb_s32 buf_id;
	gb_u32 _pad;
};

/* Poll-fd request - register a userspace eventfd to receive pressure events.
 * Userspace creates the eventfd: efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)
 * then passes it here.  Kernel signals it whenever swap pressure changes.
 */
struct gb_poll_req {
	gb_s32 efd;     /* eventfd fd (in - created by caller)         */
	gb_u32 _pad;
};

/* KV cache reservation - how many MB of T1 VRAM to reserve for the KV cache.
 * Weights overflow to T2 sooner, leaving T1 headroom so KV cache (which is
 * read+written on every generation step) stays at ~336 GB/s VRAM bandwidth
 * instead of dropping to ~32 GB/s PCIe (T2) or ~1.8 GB/s NVMe (T3).
 *
 * Set at module load: kv_reserve_mb=N parameter.
 * Updated at runtime: GB_IOCTL_SET_KV_RESERVE (sysfs or env var path).
 * Standalone tools: set GREENBOOST_KV_RESERVE_MB env var or module param.
 *
 * Note: the shim's adaptive reserve automatically collapses this headroom once
 * KV cache has actually been allocated in T1, reclaiming VRAM for weights.
 */
struct gb_kv_reserve_req {
	gb_u32 reserve_mb;  /* MB of T1 VRAM to reserve for KV cache (0=disable) */
	gb_u32 _pad;
};

/* Pin an existing user-space virtual buffer (allocated by shim) */
struct gb_pin_req {
	gb_u64 vaddr;   /* user space pointer (in)    */
	gb_u64 size;    /* bytes to pin       (in)    */
	gb_s32 fd;      /* DMA-BUF fd returned (out)  */
	gb_u32 flags;   /* GB_ALLOC_* flags   (in)    */
};

/* Dynamic T2 pool cap - set this at startup to cap the T2 DDR RAM pool based
 * on currently available system RAM:
 *   systems with < 64 GB RAM: 70% of total RAM
 *   systems with >= 64 GB RAM: 80% of total RAM
 * minus safety_reserve_gb, enforced dynamically by the watchdog.
 * This allows T2 to shrink automatically when other workloads consume RAM,
 * preventing OOM.
 *
 * cap_mb: desired new pool cap in MB.  Kernel clamps to:
 *   [4096 MB, total_ram - safety_reserve_gb * 1024]
 * prev_mb: previous cap (filled by kernel on return, informational).
 */
struct gb_pool_cap_req {
	gb_u64 cap_mb;   /* new T2 pool cap in MB          (in)  */
	gb_u64 prev_mb;  /* previous cap in MB             (out) */
};

/* IOCTL cmds 12 and 13 are reserved.
 * Intentional gap - do not reuse these numbers to preserve ABI stability. */

/* Structured pool info for Kubernetes DRA kubelet plugin and Prometheus exporter.
 * Returned by GB_IOCTL_GET_POOL_INFO_V3 (cmd 14). All sizes in MB.
 * Replaces the human-readable status sysfs text for machine consumption. */
struct gb_pool_info_v3 {
	/* Tier 1 */
	gb_u64 t1_physical_mb;      /* GPU VRAM physical total                          */
	gb_u64 t1_nvlink_total_mb;  /* Aggregated via NVLink pool (0 = pool disabled)   */
	/* Tier 2 */
	gb_u64 t2_total_mb;         /* System DDR pool cap                              */
	gb_u64 t2_used_mb;          /* Currently allocated in T2                        */
	gb_u64 t2_available_mb;     /* t2_total - t2_used                               */
	/* Tier 3 */
	gb_u64 t3_total_mb;         /* NVMe/Lustre swap cap                             */
	gb_u64 t3_used_mb;          /* Currently allocated in T3                        */
	/* Status */
	gb_u32 nvlink_ready;         /* 1 = P2P NVLink verified across all GPU pairs     */
	gb_u32 compute_domain_active;/* 1 = ComputeDomain MNNVL workload active          */
	gb_u32 watchdog_pressure;    /* 0=ok 1=warn 2=critical                           */
	gb_u32 active_buffers;       /* Live DMA-BUF object count                        */
	gb_u32 oom_active;           /* 1 = OOM guard currently throttling               */
	gb_u32 gpu_count;            /* GPU count (nvlink_gpu_count param value)         */
	gb_u32 kv_reserve_mb;        /* KV cache T1 reserve                              */
	gb_u32 _pad;
};

#define GB_IOCTL_MAGIC      'G'
#define GB_IOCTL_ALLOC      _IOWR(GB_IOCTL_MAGIC, 1, struct gb_alloc_req)
#define GB_IOCTL_GET_INFO   _IOR( GB_IOCTL_MAGIC, 2, struct gb_info)
#define GB_IOCTL_RESET      _IO(  GB_IOCTL_MAGIC, 3)
#define GB_IOCTL_MADVISE    _IOW( GB_IOCTL_MAGIC, 4, struct gb_madvise_req)
#define GB_IOCTL_EVICT      _IOW( GB_IOCTL_MAGIC, 5, struct gb_evict_req)
/* cmd 6: reserved (ABI gap - never allocated; do not reuse) */
#define GB_IOCTL_POLL_FD    _IOW( GB_IOCTL_MAGIC, 7, struct gb_poll_req)
#define GB_IOCTL_PIN_USER_PTR   _IOWR(GB_IOCTL_MAGIC, 8, struct gb_pin_req)
#define GB_IOCTL_SET_KV_RESERVE _IOW( GB_IOCTL_MAGIC, 9, struct gb_kv_reserve_req)
/* cmd 10: reserved (SET_TURBOQUANT removed - ABI gap, do not reuse) */
#define GB_IOCTL_SET_POOL_CAP   _IOWR(GB_IOCTL_MAGIC, 11, struct gb_pool_cap_req)
/* cmd 12: reserved (ABI gap - do not reuse) */
/* cmd 13: reserved (ABI gap - do not reuse) */
#define GB_IOCTL_GET_POOL_INFO_V3 _IOR(GB_IOCTL_MAGIC, 14, struct gb_pool_info_v3)
/* Increment phase_reset_seq in the kernel; shim detects the change on next
 * GB_KV_REFRESH_INTERVAL boundary and resets g_alloc_phase + g_kv_allocated_t1_bytes.
 * Synapse CLI calls this before model swaps to restart phase detection from INIT. */
#define GB_IOCTL_RESET_PHASE      _IO( GB_IOCTL_MAGIC, 15)

/* Release all T2/T3 kernel buffers owned by a PID.
 * pid=0 means the calling process's own PID.
 * pid!=0 requires CAP_SYS_ADMIN.
 * Returns count of buffers released, or negative errno. */
struct gb_release_pid_req {
	gb_u32 pid;   /* 0 = caller's own PID */
	gb_u32 _pad;
};
#define GB_IOCTL_RELEASE_PID      _IOW( GB_IOCTL_MAGIC, 16, struct gb_release_pid_req)

/* Live T3 pool resize - expand or shrink the NVMe backing store cap.
 * Opens /var/lib/greenboost/t3_store if not already open (enables T3 on demand).
 * cap_mb == 0 means disk-limited (no cap).
 * Requires CAP_SYS_ADMIN.
 * prev_mb: previous cap (0 = was disk-limited or disabled). */
struct gb_t3_cap_req {
	gb_u64 cap_mb;   /* new T3 cap in MB; 0 = disk-limited  (in)  */
	gb_u64 prev_mb;  /* previous cap in MB                  (out) */
};
#define GB_IOCTL_SET_T3_CAP       _IOWR(GB_IOCTL_MAGIC, 17, struct gb_t3_cap_req)

/* Session priority management - inspired by dmem cgroup foreground-booster.
 *
 * GB_IOCTL_SESSION_IDLE:   move all T2 buffers owned by caller's PID to LRU
 *   tail so they become preferred eviction candidates under pressure.  Call
 *   this when the inference session transitions to idle (no new requests).
 *
 * GB_IOCTL_SESSION_ACTIVE: move all T2 buffers owned by caller's PID to LRU
 *   head so they are evicted last.  Call this when a new inference request
 *   arrives and the session transitions back to active.
 *
 * pid == 0 means the calling process's own PID.  pid != 0 requires CAP_SYS_ADMIN.
 */
struct gb_session_req {
	gb_u32 pid;      /* 0 = caller's own PID */
	gb_u32 reserved; /* must be zero         */
};
#define GB_IOCTL_SESSION_IDLE   _IOW(GB_IOCTL_MAGIC, 18, struct gb_session_req)
#define GB_IOCTL_SESSION_ACTIVE _IOW(GB_IOCTL_MAGIC, 19, struct gb_session_req)

/* Swap pressure thresholds (T3 NVMe) */
#define GB_SWAP_PRESSURE_OK       0
#define GB_SWAP_PRESSURE_WARN     1   /* >75% swap used */
#define GB_SWAP_PRESSURE_CRITICAL 2   /* >90% swap used */

/* T2 pool pressure thresholds (System DDR RAM pool saturation) */
#define GB_T2_PRESSURE_OK         0
#define GB_T2_PRESSURE_WARN       1   /* >75% T2 pool used */
#define GB_T2_PRESSURE_CRITICAL   2   /* >90% T2 pool used - auto-evict cold bufs */

/* Gaming mode - signal that a game is running so inference T2 memory is deprioritized.
 * active=1: set gaming mode; active=0: clear.  No capability required.
 * When gaming_mode is active the safety_reserve is doubled and all non-frozen,
 * non-session-protected T2 buffers are moved to the LRU tail. */
struct gb_gaming_req {
    gb_u32 active;   /* 1=gaming start, 0=gaming stop */
    gb_u32 reserved;
};
#define GB_IOCTL_GAMING_MODE  _IOW(GB_IOCTL_MAGIC, 20, struct gb_gaming_req)

#endif /* GREENBOOST_IOCTL_H */

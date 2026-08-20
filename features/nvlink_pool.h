/* SPDX-License-Identifier: GPL-2.0
 * GreenBoost v3.4 - NVLink Pooling Feature
 *
 * Aggregates multiple GPU VRAM into a unified T1 pool via NVLink 2.0
 * Used for V100 clusters (cc 7.0) with all-to-all NVLink 2.0 connectivity
 *
 * Author  : Ferran Duarri
 * License : GPL v2
 */
#ifndef GREENBOOST_NVLINK_POOL_H
#define GREENBOOST_NVLINK_POOL_H

#include <linux/types.h>

/* GPU fabric info (from k8s-dra-driver-gpu nvlib.go) */
struct gb_gpu_fabric_state {
	u32  state;          /* 0=not attached, 1=attached, 2=error */
	u32  status;         /* NVML return status */
	u32  cliques;        /* number of distinct NVLink cliques */
};

/* Per-GPU VRAM info */
struct gb_gpu_vram_info {
	u32  gpu_id;         /* GPU index */
	u64  vram_size_gb;   /* Physical VRAM in GB */
	u8   nvlink_peers;   /* Number of NVLink peers */
	bool in_clique;      /* Is part of a valid clique */
};

/* NVLink pooled info (aggregate T1) */
struct gb_nvlink_pool {
	bool         enabled;        /* nvlink_pool module param */
	u32          gpu_count;      /* Number of GPUs in pool */
	u64          total_vram_gb;  /* Aggregated VRAM (T1) */
	bool         fabric_ready;   /* NVLink fabric in READY state */
	bool         clique_valid;   /* All GPUs in same clique */
};

/* NVLink state constants (from k8s-dra-driver-gpu) */
#define GB_NVLINK_STATE_NOT_SUPPORTED  0
#define GB_NVLINK_STATE_NOT_ATTACHED  1
#define GB_NVLINK_STATE_ATTACHED      2
#define GB_NVLINK_STATE_ERROR         3

#define GB_NVLINK_STATUS_READY        0
#define GB_NVLINK_STATUS_TYPE_MASK    0xFF000000
#define GB_NVLINK_STATUS_TYPE_SHIFT   24

#ifdef __KERNEL__
/* Initialize NVLink pooling subsystem */
int gb_nvlink_pool_init(void);
void gb_nvlink_pool_exit(void);

/* Called by kubelet plugin sysfs store after NVML P2P verification (V100 P2P approach).
 * Sets fabric_ready=ready, updates gpu_count and total_vram_gb. */
void gb_nvlink_set_ready(bool ready, u32 gpu_count, u64 vram_per_gpu_gb);
#endif /* __KERNEL__ */

#endif /* GREENBOOST_NVLINK_POOL_H */
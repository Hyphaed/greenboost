/* SPDX-License-Identifier: GPL-2.0
 * GreenBoost v3.0 — NVLink Pooling Feature
 *
 * Aggregates multiple GPU VRAM into a unified T1 pool via NVLink 2.0.
 *
 * V100 correction (BUG-009):
 *   V100 uses NVLink 2.0 direct P2P — NOT an NVSwitch fabric.
 *   nvmlDeviceGetGpuFabricInfo() returns GPU_FABRIC_STATE_NOT_SUPPORTED for V100.
 *   Readiness must be verified via nvmlDeviceGetP2PStatus() from user-space (kubelet plugin),
 *   which then writes 1 to /sys/class/greenboost/greenboost/nvlink_ready.
 *   The kernel module does NOT call NVML; it accepts the sysfs signal and tracks state.
 *
 * Author  : Ferran Duarri
 * License : GPL v2
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include "nvlink_pool.h"

/* BUG-006 fix: DRIVER_NAME must be defined in this translation unit.
 * When nvlink_pool.c is compiled separately, define it here.
 * When included via #include from greenboost.c, DRIVER_NAME is already defined. */
#ifndef DRIVER_NAME
#define DRIVER_NAME "greenboost"
#endif

/* Global NVLink pool state */
static struct gb_nvlink_pool nvlink_pool = {
	.enabled       = false,
	.gpu_count     = 0,
	.total_vram_gb = 0,
	.fabric_ready  = false,
	.clique_valid  = false,
};

/* Module parameter: enable NVLink pooling */
static bool nvlink_pool_enable = false;
module_param(nvlink_pool_enable, bool, 0444);
MODULE_PARM_DESC(nvlink_pool_enable,
	"Enable NVLink-based T1 VRAM pooling across GPUs");

/* Module parameter: expected GPU count for P2P pool.
 * 0 = auto-detect (disabled; kubelet plugin writes nvlink_ready).
 * 8 = V100 8-GPU node (all-to-all NVLink 2.0, 28 direct peer pairs).
 * BUG-009 fix: the kernel cannot call NVML; this parameter signals intent.
 * The kubelet plugin verifies P2P topology via nvmlDeviceGetP2PStatus() and
 * writes "1" to /sys/class/greenboost/greenboost/nvlink_ready when ready. */
static int nvlink_gpu_count = 0;
module_param(nvlink_gpu_count, int, 0444);
MODULE_PARM_DESC(nvlink_gpu_count,
	"Number of NVLink-connected GPUs (0=auto; kubelet plugin writes nvlink_ready)");

/*
 * gb_nvlink_pool_init — Initialize the NVLink pooling subsystem.
 *
 * BUG-009 fix: Does NOT fail when fabric is not immediately ready.
 * For V100, the fabric state is always GB_NVLINK_STATE_NOT_SUPPORTED
 * (no NVSwitch).  We start with fabric_ready=false and wait for the
 * kubelet plugin to write nvlink_ready sysfs after NVML P2P verification.
 */
int gb_nvlink_pool_init(void)
{
	nvlink_pool.enabled   = nvlink_pool_enable;
	nvlink_pool.gpu_count = (u32)nvlink_gpu_count;

	if (!nvlink_pool.enabled) {
		/* BUG-007 fix: use \n (one backslash in source = newline) */
		pr_info(DRIVER_NAME ": NVLink pooling disabled (nvlink_pool_enable=0)\n");
		return 0;
	}

	pr_info(DRIVER_NAME ": NVLink pooling enabled — waiting for kubelet plugin P2P verification\n");
	pr_info(DRIVER_NAME ": NVLink expected GPU count: %d\n", nvlink_gpu_count);
	pr_info(DRIVER_NAME ": V100 note: fabric state is NOT_SUPPORTED (no NVSwitch); "
		"P2P checked via NVML by kubelet plugin\n");

	/* fabric_ready starts false; set to true when kubelet plugin writes nvlink_ready=1 */
	nvlink_pool.fabric_ready  = false;
	nvlink_pool.clique_valid  = false;
	nvlink_pool.total_vram_gb = 0;

	return 0; /* BUG-009 fix: succeed even though fabric not ready yet */
}

/*
 * gb_nvlink_set_ready — Called when kubelet plugin writes nvlink_ready sysfs.
 * Updates pool state to reflect confirmed P2P topology.
 */
void gb_nvlink_set_ready(bool ready, u32 gpu_count, u64 vram_per_gpu_gb)
{
	nvlink_pool.fabric_ready  = ready;
	nvlink_pool.clique_valid  = ready;
	if (ready) {
		nvlink_pool.gpu_count     = gpu_count;
		nvlink_pool.total_vram_gb = gpu_count * vram_per_gpu_gb;
		pr_info(DRIVER_NAME ": NVLink pool ready — %u GPUs × %llu GB = %llu GB T1\n",
			gpu_count, vram_per_gpu_gb, nvlink_pool.total_vram_gb);
	} else {
		nvlink_pool.total_vram_gb = 0;
		pr_info(DRIVER_NAME ": NVLink pool cleared\n");
	}
}

/*
 * gb_nvlink_pool_exit — Shut down the NVLink pooling subsystem.
 */
void gb_nvlink_pool_exit(void)
{
	pr_info(DRIVER_NAME ": NVLink pooling subsystem shutdown\n");
	memset(&nvlink_pool, 0, sizeof(nvlink_pool));
}

/*
 * gb_nvlink_query_fabric — Query NVLink fabric state.
 *
 * BUG-009 fix: For V100, always returns GB_NVLINK_STATE_NOT_SUPPORTED.
 * This is correct — V100 has no NVSwitch fabric.
 * Actual P2P capability is verified from user-space via nvmlDeviceGetP2PStatus().
 */
int gb_nvlink_query_fabric(struct gb_gpu_fabric_state *state)
{
	if (!state)
		return -EINVAL;

	memset(state, 0, sizeof(*state));

	/*
	 * V100 NVLink 2.0 correction:
	 *   - nvmlDeviceGetGpuFabricInfo() → GPU_FABRIC_STATE_NOT_SUPPORTED
	 *   - This is expected — V100 has direct P2P, not an NVSwitch fabric
	 *   - Actual readiness is set via gb_nvlink_set_ready() from kubelet plugin
	 */
	state->state   = GB_NVLINK_STATE_NOT_SUPPORTED;
	state->status  = 0;
	state->cliques = 0;

	return 0;
}

/*
 * gb_nvlink_is_poolable — Check if NVLink pooling is active and ready.
 */
bool gb_nvlink_is_poolable(void)
{
	return nvlink_pool.enabled && nvlink_pool.fabric_ready;
}

/*
 * gb_nvlink_get_aggregated_vram — Return aggregated T1 VRAM in GB.
 */
u64 gb_nvlink_get_aggregated_vram(void)
{
	if (!nvlink_pool.enabled || !nvlink_pool.fabric_ready)
		return 0;
	return nvlink_pool.total_vram_gb;
}

/*
 * gb_nvlink_get_gpu_info — Return per-GPU VRAM info.
 */
int gb_nvlink_get_gpu_info(u32 gpu_id, struct gb_gpu_vram_info *info)
{
	if (!info)
		return -EINVAL;

	memset(info, 0, sizeof(*info));
	info->gpu_id       = gpu_id;
	info->vram_size_gb = 32; /* V100 default: 32 GB HBM2 */
	info->nvlink_peers = (nvlink_pool.gpu_count > 0) ?
	                     nvlink_pool.gpu_count - 1 : 7;
	info->in_clique    = nvlink_pool.clique_valid;

	return 0;
}

/*
 * gb_nvlink_update_shim_vram — Update CUDA shim's virtual VRAM aggregation.
 * Called by the CUDA shim at init if nvlink_ready sysfs reads 1.
 */
void gb_nvlink_update_shim_vram(u64 *virtual_vram_gb)
{
	if (!nvlink_pool.enabled || !nvlink_pool.fabric_ready)
		return;

	pr_info(DRIVER_NAME ": NVLink pooling active: T1=%llu GB aggregated\n",
		nvlink_pool.total_vram_gb);
	/* The shim reads nvlink_ready + gpu_count_per_node sysfs directly */
}

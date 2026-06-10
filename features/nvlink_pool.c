/* SPDX-License-Identifier: GPL-2.0
 * GreenBoost v3.0 - NVLink Pooling Feature
 *
 * Aggregates multiple GPU VRAM into a unified T1 pool via NVLink 2.0.
 *
 * V100 correction (BUG-009):
 *   V100 uses NVLink 2.0 direct P2P - NOT an NVSwitch fabric.
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
#include <linux/mutex.h>
#include <linux/slab.h>
#include "nvlink_pool.h"

/* BUG-006 fix: DRIVER_NAME must be defined in this translation unit.
 * When nvlink_pool.c is compiled separately, define it here.
 * When included via #include from greenboost.c, DRIVER_NAME is already defined. */
#ifndef DRIVER_NAME
#define DRIVER_NAME "greenboost"
#endif

/* F-L2-11: module-level mutex protecting all accesses to nvlink_pool fields */
static DEFINE_MUTEX(nvlink_pool_lock);

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
 * gb_nvlink_pool_init - Initialize the NVLink pooling subsystem.
 *
 * BUG-009 fix: Does NOT fail when fabric is not immediately ready.
 * For V100, the fabric state is always GB_NVLINK_STATE_NOT_SUPPORTED
 * (no NVSwitch).  We start with fabric_ready=false and wait for the
 * kubelet plugin to write nvlink_ready sysfs after NVML P2P verification.
 */
int gb_nvlink_pool_init(void)
{
	mutex_lock(&nvlink_pool_lock);
	nvlink_pool.enabled   = nvlink_pool_enable;
	nvlink_pool.gpu_count = (u32)nvlink_gpu_count;

	if (!nvlink_pool.enabled) {
		mutex_unlock(&nvlink_pool_lock);
		/* BUG-007 fix: use \n (one backslash in source = newline) */
		pr_info(DRIVER_NAME ": NVLink pooling disabled (nvlink_pool_enable=0)\n");
		return 0;
	}

	/* fabric_ready starts false; set to true when kubelet plugin writes nvlink_ready=1 */
	nvlink_pool.fabric_ready  = false;
	nvlink_pool.clique_valid  = false;
	nvlink_pool.total_vram_gb = 0;
	mutex_unlock(&nvlink_pool_lock);

	pr_info(DRIVER_NAME ": NVLink pooling enabled - waiting for kubelet plugin P2P verification\n");
	pr_info(DRIVER_NAME ": NVLink expected GPU count: %d\n", nvlink_gpu_count);
	pr_info(DRIVER_NAME ": V100 note: fabric state is NOT_SUPPORTED (no NVSwitch); "
		"P2P checked via NVML by kubelet plugin\n");

	return 0; /* BUG-009 fix: succeed even though fabric not ready yet */
}

/*
 * gb_nvlink_set_ready - Called when kubelet plugin writes nvlink_ready sysfs.
 * Updates pool state to reflect confirmed P2P topology.
 */
void gb_nvlink_set_ready(bool ready, u32 gpu_count, u64 vram_per_gpu_gb)
{
	u64 total;

	mutex_lock(&nvlink_pool_lock);
	nvlink_pool.fabric_ready  = ready;
	nvlink_pool.clique_valid  = ready;
	if (ready) {
		nvlink_pool.gpu_count     = gpu_count;
		nvlink_pool.total_vram_gb = gpu_count * vram_per_gpu_gb;
		total = nvlink_pool.total_vram_gb;
		mutex_unlock(&nvlink_pool_lock);
		pr_info(DRIVER_NAME ": NVLink pool ready - %u GPUs × %llu GB = %llu GB T1\n",
			gpu_count, vram_per_gpu_gb, total);
	} else {
		nvlink_pool.total_vram_gb = 0;
		mutex_unlock(&nvlink_pool_lock);
		pr_info(DRIVER_NAME ": NVLink pool cleared\n");
	}
}

/*
 * gb_nvlink_pool_exit - Shut down the NVLink pooling subsystem.
 */
void gb_nvlink_pool_exit(void)
{
	pr_info(DRIVER_NAME ": NVLink pooling subsystem shutdown\n");
	mutex_lock(&nvlink_pool_lock);
	memset(&nvlink_pool, 0, sizeof(nvlink_pool));
	mutex_unlock(&nvlink_pool_lock);
}

/*
 * gb_nvlink_query_fabric - Query NVLink fabric state.
 *
 * BUG-009 fix: For V100, always returns GB_NVLINK_STATE_NOT_SUPPORTED.
 * This is correct - V100 has no NVSwitch fabric.
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
	 *   - This is expected - V100 has direct P2P, not an NVSwitch fabric
	 *   - Actual readiness is set via gb_nvlink_set_ready() from kubelet plugin
	 */
	state->state   = GB_NVLINK_STATE_NOT_SUPPORTED;
	state->status  = 0;
	state->cliques = 0;

	return 0;
}

/*
 * gb_nvlink_is_poolable - Check if NVLink pooling is active and ready.
 */
bool gb_nvlink_is_poolable(void)
{
	bool ret;

	mutex_lock(&nvlink_pool_lock);
	ret = nvlink_pool.enabled && nvlink_pool.fabric_ready;
	mutex_unlock(&nvlink_pool_lock);
	return ret;
}

/*
 * gb_nvlink_get_aggregated_vram - Return aggregated T1 VRAM in GB.
 */
u64 gb_nvlink_get_aggregated_vram(void)
{
	u64 ret;

	mutex_lock(&nvlink_pool_lock);
	ret = (nvlink_pool.enabled && nvlink_pool.fabric_ready) ?
	      nvlink_pool.total_vram_gb : 0;
	mutex_unlock(&nvlink_pool_lock);
	return ret;
}

/*
 * gb_nvlink_get_gpu_info - Return per-GPU VRAM info.
 */
int gb_nvlink_get_gpu_info(u32 gpu_id, struct gb_gpu_vram_info *info)
{
	u32 gpu_count;
	u64 total_vram_gb;
	bool clique_valid;

	if (!info)
		return -EINVAL;

	mutex_lock(&nvlink_pool_lock);
	gpu_count     = nvlink_pool.gpu_count;
	total_vram_gb = nvlink_pool.total_vram_gb;
	clique_valid  = nvlink_pool.clique_valid;
	mutex_unlock(&nvlink_pool_lock);

	/* F-L2-10: bounds check - reject out-of-range gpu_id */
	if (gpu_count > 0 && gpu_id >= gpu_count)
		return -ENODEV;

	memset(info, 0, sizeof(*info));
	info->gpu_id       = gpu_id;
	/* F-L2-10: derive per-GPU VRAM from pool total instead of hardcoding 32.
	 * Guard against division by zero when gpu_count is 0. */
	info->vram_size_gb = (gpu_count > 0) ? (total_vram_gb / gpu_count) : 32;
	info->nvlink_peers = (gpu_count > 0) ? gpu_count - 1 : 7;
	info->in_clique    = clique_valid;

	return 0;
}

/*
 * gb_nvlink_update_shim_vram - Update CUDA shim's virtual VRAM aggregation.
 * Called by the CUDA shim at init if nvlink_ready sysfs reads 1.
 */
void gb_nvlink_update_shim_vram(u64 *virtual_vram_gb)
{
	u64 total;

	mutex_lock(&nvlink_pool_lock);
	if (!nvlink_pool.enabled || !nvlink_pool.fabric_ready) {
		mutex_unlock(&nvlink_pool_lock);
		return;
	}
	total = nvlink_pool.total_vram_gb;
	mutex_unlock(&nvlink_pool_lock);

	pr_info(DRIVER_NAME ": NVLink pooling active: T1=%llu GB aggregated\n", total);
	/* The shim reads nvlink_ready + gpu_count_per_node sysfs directly */
}

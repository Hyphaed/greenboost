// SPDX-License-Identifier: Apache-2.0
// Health watcher for GreenBoost DRA kubelet plugin.
// Periodically checks NVLink P2P topology (V100) and watchdog pressure.

package main

import (
	"context"
	"time"

	nvml "github.com/NVIDIA/go-nvml/pkg/nvml"
	"k8s.io/klog/v2"
)

// HealthWatcher monitors NVLink fabric health and GreenBoost pressure events.
type HealthWatcher struct {
	sysfsBase      string
	nvlinkPool     bool
	nvlinkGPUCount int
	nvmlEnabled    bool
	pool           *PoolManager
}

// NewHealthWatcher creates a HealthWatcher.
func NewHealthWatcher(sysfsBase string, nvlinkPool bool, nvlinkGPUCount int, nvmlEnabled bool) *HealthWatcher {
	return &HealthWatcher{
		sysfsBase:      sysfsBase,
		nvlinkPool:     nvlinkPool,
		nvlinkGPUCount: nvlinkGPUCount,
		nvmlEnabled:    nvmlEnabled,
		pool:           &PoolManager{sysfsBase: sysfsBase},
	}
}

// Run starts the health check loop. It checks NVLink P2P status and pressure every 30s.
func (h *HealthWatcher) Run(ctx context.Context) {
	if h.nvmlEnabled && h.nvlinkPool {
		h.initNVML()
		if h.nvmlEnabled {
			// initNVML succeeded; shut down cleanly when this goroutine exits.
			defer nvml.Shutdown()
		}
	}

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	// Initial check
	h.checkHealth()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			h.checkHealth()
		}
	}
}

// initNVML initialises NVML library for P2P queries.
func (h *HealthWatcher) initNVML() {
	if ret := nvml.Init(); ret != nvml.SUCCESS {
		klog.WarningS("NVML init failed - NVLink P2P checks disabled",
			"error", nvml.ErrorString(ret))
		h.nvmlEnabled = false
		return
	}
	klog.InfoS("NVML initialized for NVLink P2P health checks")
}

// checkHealth runs all health checks and updates sysfs flags.
func (h *HealthWatcher) checkHealth() {
	if h.nvlinkPool && h.nvmlEnabled {
		ready := h.checkNVLinkP2P()
		if err := h.pool.SetNVLinkReady(ready); err != nil {
			klog.V(4).InfoS("nvlink_ready sysfs write failed", "error", err)
		}
	}

	// Check pressure level
	info, err := h.pool.GetPoolInfoV3()
	if err != nil {
		klog.V(4).InfoS("Health check: pool info read failed", "error", err)
		return
	}

	switch info.WatchdogPressure {
	case 2:
		klog.WarningS("GreenBoost T2 pressure CRITICAL",
			"t2_used_mb", info.T2UsedMB, "t2_total_mb", info.T2TotalMB)
	case 1:
		klog.WarningS("GreenBoost T2 pressure WARNING",
			"t2_used_mb", info.T2UsedMB, "t2_total_mb", info.T2TotalMB)
	}
}

// checkNVLinkP2P verifies NVLink P2P connectivity across all GPU pairs (V100 approach).
// V100 has NVLink 2.0 direct P2P - NOT NVSwitch fabric. Use nvmlDeviceGetP2PStatus().
// Returns true only when ALL expected GPU pairs report NVML_P2P_STATUS_OK.
func (h *HealthWatcher) checkNVLinkP2P() bool {
	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS {
		klog.ErrorS(nil, "NVML DeviceGetCount failed", "error", nvml.ErrorString(ret))
		return false
	}

	expectedCount := h.nvlinkGPUCount
	if expectedCount == 0 {
		expectedCount = count
	}

	if count < expectedCount {
		klog.WarningS("Fewer GPUs than expected for NVLink pool",
			"found", count, "expected", expectedCount)
		return false
	}

	// Check all pairs (n*(n-1)/2 pairs for n GPUs)
	for i := 0; i < count; i++ {
		dev0, ret := nvml.DeviceGetHandleByIndex(i)
		if ret != nvml.SUCCESS {
			klog.V(3).InfoS("NVML DeviceGetHandleByIndex failed",
				"index", i, "error", nvml.ErrorString(ret))
			return false
		}
		for j := i + 1; j < count; j++ {
			dev1, ret := nvml.DeviceGetHandleByIndex(j)
			if ret != nvml.SUCCESS {
				klog.V(3).InfoS("NVML DeviceGetHandleByIndex failed",
					"index", j, "error", nvml.ErrorString(ret))
				return false
			}
			// Check NVLink P2P capability between this pair
			status, ret := nvml.DeviceGetP2PStatus(dev0, dev1,
				nvml.P2P_CAPS_INDEX_NVLINK)
			if ret != nvml.SUCCESS {
				klog.V(3).InfoS("NVML GetP2PStatus failed",
					"gpu0", i, "gpu1", j, "error", nvml.ErrorString(ret))
				return false
			}
			if status != nvml.P2P_STATUS_OK {
				klog.V(3).InfoS("NVLink P2P not ready for GPU pair",
					"gpu0", i, "gpu1", j, "status", status)
				return false
			}
		}
	}

	klog.V(2).InfoS("NVLink P2P verified across all GPU pairs",
		"gpu_count", count)
	return true
}

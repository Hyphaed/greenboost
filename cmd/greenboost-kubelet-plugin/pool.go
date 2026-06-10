// SPDX-License-Identifier: Apache-2.0
// Pool state manager for GreenBoost DRA kubelet plugin.
// Reads pool state via sysfs (text pool_info) or GB_IOCTL_GET_POOL_INFO_V3.

package main

import (
	"fmt"
	"strconv"
	"strings"
)

// PoolInfoV3 mirrors the kernel struct gb_pool_info_v3.
type PoolInfoV3 struct {
	T1PhysicalMB      uint64
	T1NVLinkTotalMB   uint64
	T2TotalMB         uint64
	T2UsedMB          uint64
	T2AvailableMB     uint64
	T3TotalMB         uint64
	T3UsedMB          uint64
	NVLinkReady       bool
	ComputeDomainActive bool
	WatchdogPressure  uint32
	ActiveBuffers     uint32
	OOMActive         bool
	GPUCount          uint32
	KVReserveMB       uint32
}

// PoolManager reads and caches GreenBoost pool state.
type PoolManager struct {
	sysfsBase string
}

// NewPoolManager creates a PoolManager for the given sysfs path.
func NewPoolManager(sysfsBase string) (*PoolManager, error) {
	return &PoolManager{sysfsBase: sysfsBase}, nil
}

// GetPoolInfoV3 reads pool state. Tries text sysfs pool_info as primary source.
func (p *PoolManager) GetPoolInfoV3() (*PoolInfoV3, error) {
	text, err := readSysfsAttr(p.sysfsBase, "pool_info")
	if err != nil {
		return nil, fmt.Errorf("read pool_info: %w", err)
	}
	return parsePoolInfoText(text, p.sysfsBase)
}

// SetComputeDomainActive writes the compute_domain_active sysfs.
func (p *PoolManager) SetComputeDomainActive(active bool) error {
	val := "0"
	if active {
		val = "1"
	}
	return writeSysfsAttr(p.sysfsBase, "compute_domain_active", val)
}

// SetNVLinkReady writes the nvlink_ready sysfs after NVML P2P verification.
func (p *PoolManager) SetNVLinkReady(ready bool) error {
	val := "0"
	if ready {
		val = "1"
	}
	return writeSysfsAttr(p.sysfsBase, "nvlink_ready", val)
}

// parsePoolInfoText parses the human-readable pool_info sysfs text.
// Format has lines like:
//
//	T1 physical_vram_gb : 12
//	allocated_mb : 4096
//	max_pool_mb  : 52224
func parsePoolInfoText(text, sysfsBase string) (*PoolInfoV3, error) {
	info := &PoolInfoV3{}

	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "=") {
			continue
		}
		kv := strings.SplitN(line, ":", 2)
		if len(kv) != 2 {
			continue
		}
		key := strings.TrimSpace(kv[0])
		val := strings.TrimSpace(kv[1])

		// strip trailing units (MB, GB, etc.)
		numStr := strings.Fields(val)
		if len(numStr) == 0 {
			continue
		}
		num, err := strconv.ParseUint(numStr[0], 10, 64)
		if err != nil {
			continue
		}

		switch {
		case strings.Contains(key, "physical_vram_gb") || strings.Contains(key, "vram_physical"):
			if num > 65536 {
				num = 65536 // cap at 65536 GB to prevent uint64 overflow on corrupt sysfs
			}
			info.T1PhysicalMB = num * 1024
		case key == "max_pool_mb" || strings.Contains(key, "max_pool"):
			info.T2TotalMB = num
		case key == "allocated_mb" || strings.Contains(key, "t2_allocated") || key == "pool_allocated_mb":
			info.T2UsedMB = num
		case strings.Contains(key, "nvme") && strings.Contains(key, "swap") && strings.Contains(key, "total"):
			info.T3TotalMB = num
		case strings.Contains(key, "nvme") && strings.Contains(key, "alloc"):
			info.T3UsedMB = num
		case key == "active_buffers":
			info.ActiveBuffers = uint32(num)
		case key == "kv_reserve_mb":
			info.KVReserveMB = uint32(num)
		case key == "watchdog_pressure" || key == "t2_pressure":
			info.WatchdogPressure = uint32(num)
		case key == "oom_active":
			info.OOMActive = num != 0
		}
	}

	if info.T2TotalMB > info.T2UsedMB {
		info.T2AvailableMB = info.T2TotalMB - info.T2UsedMB
	}

	// Read nvlink_ready sysfs
	if nvl, err := readSysfsAttr(sysfsBase, "nvlink_ready"); err == nil {
		info.NVLinkReady = strings.TrimSpace(nvl) == "1"
	}

	// Read compute_domain_active sysfs
	if cda, err := readSysfsAttr(sysfsBase, "compute_domain_active"); err == nil {
		info.ComputeDomainActive = strings.TrimSpace(cda) == "1"
	}

	return info, nil
}

// SPDX-License-Identifier: Apache-2.0
// CDI spec writer for GreenBoost DRA kubelet plugin.
// Writes a CDI spec at plugin startup so pods can request
//   CDIDeviceIDs: ["greenboost.nvidia.com/vram=pool"]
// to get /dev/greenboost and /sys/class/greenboost mounted.

package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	cdiSpecFile   = "greenboost.nvidia.com-vram.yaml"
	cdiVersion    = "0.5.0"
	cdiKind       = "greenboost.nvidia.com/vram"
	cdiDeviceName = "pool"
	greenboostDev = "/dev/greenboost"
	greenboostSys = "/sys/class/greenboost"
)

// WriteCDISpec writes the CDI device spec for GreenBoost to the CDI root.
func WriteCDISpec(cdiRoot, sysfsBase string) error {
	major, err := GetDeviceMajor("greenboost")
	if err != nil {
		// Non-fatal: module may not be loaded yet; write spec with major 0
		major = 0
	}

	spec := fmt.Sprintf(`cdiVersion: "%s"
kind: "%s"
devices:
- name: %s
  containerEdits:
    deviceNodes:
    - path: %s
      type: c
      major: %d
      minor: 0
      permissions: rw
    mounts:
    - hostPath: %s
      containerPath: %s
      options: [ro, bind]
`,
		cdiVersion,
		cdiKind,
		cdiDeviceName,
		greenboostDev,
		major,
		greenboostSys,
		greenboostSys,
	)

	specPath := filepath.Join(cdiRoot, cdiSpecFile)
	if err := os.WriteFile(specPath, []byte(spec), 0644); err != nil {
		return fmt.Errorf("write CDI spec %s: %w", specPath, err)
	}
	return nil
}

// GetDeviceMajor reads /proc/devices to find the major number for devName.
func GetDeviceMajor(devName string) (int, error) {
	f, err := os.Open("/proc/devices")
	if err != nil {
		return 0, err
	}
	defer f.Close()

	var major int
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || line == "Character devices:" || line == "Block devices:" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) == 2 && fields[1] == devName {
			fmt.Sscanf(fields[0], "%d", &major)
			return major, nil
		}
	}
	return 0, fmt.Errorf("device %q not found in /proc/devices", devName)
}

// GetCDIDeviceIDs returns the CDI device IDs to inject for a GreenBoost claim.
func GetCDIDeviceIDs() []string {
	return []string{fmt.Sprintf("%s=%s", cdiKind, cdiDeviceName)}
}

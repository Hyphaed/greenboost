// SPDX-License-Identifier: Apache-2.0
// GreenBoost DRA Kubelet Plugin v3.0
// Implements Kubernetes Dynamic Resource Allocation for GreenBoost extended VRAM.
// Uses the real k8s.io/dynamic-resource-allocation API directly.

package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/urfave/cli/v2"
	"k8s.io/klog/v2"
)

const (
	driverName = "greenboost.nvidia.com"
	version    = "3.0.0"
)

func main() {
	app := &cli.App{
		Name:    "greenboost-kubelet-plugin",
		Usage:   "GreenBoost DRA Kubelet Plugin - extended VRAM via Dynamic Resource Allocation",
		Version: version,
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:    "node-name",
				Usage:   "Node name (required)",
				EnvVars: []string{"NODE_NAME"},
			},
			&cli.StringFlag{
				Name:    "namespace",
				Usage:   "Namespace",
				Value:   "greenboost-system",
				EnvVars: []string{"NAMESPACE"},
			},
			&cli.StringFlag{
				Name:  "kubelet-plugins-dir",
				Usage: "Path to kubelet plugins directory",
				Value: "/var/lib/kubelet/plugins",
			},
			&cli.StringFlag{
				Name:  "kubelet-registrar-dir",
				Usage: "Path to kubelet plugin registrar directory",
				Value: "/var/lib/kubelet/plugins_registry",
			},
			&cli.StringFlag{
				Name:    "greenboost-sysfs",
				Usage:   "Path to GreenBoost sysfs interface",
				Value:   "/sys/class/greenboost/greenboost",
				EnvVars: []string{"GREENBOOST_SYSFS_PATH"},
			},
			&cli.StringFlag{
				Name:  "cdi-root",
				Usage: "Path to CDI spec directory",
				Value: "/var/run/cdi",
			},
			&cli.BoolFlag{
				Name:    "nvlink-pool",
				Usage:   "Enable NVLink-based T1 VRAM aggregation",
				EnvVars: []string{"GREENBOOST_NVLINK_POOL"},
			},
			&cli.IntFlag{
				Name:  "nvlink-gpu-count",
				Usage: "Expected number of NVLink-connected GPUs (0 = auto-detect)",
				Value: 0,
			},
			&cli.StringFlag{
				Name:    "profile-name",
				Usage:   "GreenBoost profile name",
				Value:   "autodetect",
				EnvVars: []string{"GREENBOOST_PROFILE_NAME"},
			},
			&cli.StringFlag{
				Name:    "feature-gates",
				Usage:   "Comma-separated feature gates: Feature=true,Feature=false",
				EnvVars: []string{"FEATURE_GATES"},
			},
			&cli.IntFlag{
				Name:  "v",
				Usage: "Log verbosity level",
				Value: 0,
			},
		},
		Action: func(c *cli.Context) error {
			nodeName := c.String("node-name")
			if nodeName == "" {
				return fmt.Errorf("--node-name flag is required")
			}

			features := parseFeatureGates(c.String("feature-gates"))
			for name, enabled := range features {
				klog.InfoS("Feature gate", "name", name, "enabled", enabled)
			}

			driver, err := NewGreenBoostDriver(DriverConfig{
				DriverName:          driverName,
				NodeName:            nodeName,
				Namespace:           c.String("namespace"),
				KubeletPluginsDir:   c.String("kubelet-plugins-dir"),
				KubeletRegistrarDir: c.String("kubelet-registrar-dir"),
				SysfsBase:           c.String("greenboost-sysfs"),
				CDIRoot:             c.String("cdi-root"),
				NVLinkPool:          c.Bool("nvlink-pool"),
				NVLinkGPUCount:      c.Int("nvlink-gpu-count"),
				ProfileName:         c.String("profile-name"),
				FeatureGates:        features,
			})
			if err != nil {
				return fmt.Errorf("failed to create driver: %w", err)
			}

			ctx, stop := signal.NotifyContext(context.Background(),
				syscall.SIGINT, syscall.SIGTERM)
			defer stop()

			klog.InfoS("Starting GreenBoost DRA Kubelet Plugin",
				"driver", driverName,
				"version", version,
				"node", nodeName)

			return driver.Run(ctx)
		},
	}

	if err := app.Run(os.Args); err != nil {
		klog.ErrorS(err, "Plugin exited with error")
		os.Exit(1)
	}
}

// parseFeatureGates parses "Feature1=true,Feature2=false" strings.
func parseFeatureGates(gates string) map[string]bool {
	features := make(map[string]bool)
	if gates == "" {
		return features
	}
	for _, pair := range strings.Split(gates, ",") {
		pair = strings.TrimSpace(pair)
		if pair == "" {
			continue
		}
		idx := strings.Index(pair, "=")
		if idx < 0 {
			// bare name with no value - treat as enabled
			features[pair] = true
			continue
		}
		name := pair[:idx]
		val := ""
		if idx+1 < len(pair) {
			val = pair[idx+1:]
		}
		switch strings.ToLower(val) {
		case "false", "0", "no":
			features[name] = false
		default:
			features[name] = true
		}
	}
	return features
}

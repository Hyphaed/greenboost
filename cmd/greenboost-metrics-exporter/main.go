// SPDX-License-Identifier: Apache-2.0
// GreenBoost Metrics Exporter v3.0
// Exposes GreenBoost tiered memory metrics in Prometheus format.
// Reads GB_IOCTL_GET_POOL_INFO_V3 (via sysfs text fallback) and serves /metrics.

package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"k8s.io/klog/v2"
)

const (
	defaultMetricsPort    = 8080
	defaultMetricsPath    = "/metrics"
	healthPath            = "/healthz"
	defaultScrapeInterval = 15 * time.Second
)

// --- Prometheus gauge/counter declarations ---

var (
	t1TotalBytes = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "greenboost_t1_total_bytes",
		Help: "Physical VRAM total capacity (T1 tier), bytes",
	})
	t1NVLinkBytes = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "greenboost_t1_nvlink_bytes",
		Help: "Aggregated NVLink T1 capacity (0 if NVLink pool disabled), bytes",
	})
	t2TotalBytes = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "greenboost_t2_total_bytes",
		Help: "System DDR pool total capacity (T2 tier), bytes",
	})
	t2UsedBytes = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "greenboost_t2_used_bytes",
		Help: "System DDR pool in use (T2 tier), bytes",
	})
	t3TotalBytes = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "greenboost_t3_total_bytes",
		Help: "NVMe/Lustre T3 total capacity, bytes",
	})
	t3UsedBytes = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "greenboost_t3_used_bytes",
		Help: "NVMe/Lustre T3 in use, bytes",
	})
	virtualTotalBytes = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "greenboost_virtual_total_bytes",
		Help: "Total virtual VRAM (T1 NVLink or physical + T2), bytes",
	})
	activeBuffers = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "greenboost_active_buffers",
		Help: "Number of live DMA-BUF objects",
	})
	kvReserveBytes = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "greenboost_kv_reserve_bytes",
		Help: "KV cache T1 reserve, bytes",
	})
	watchdogPressure = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "greenboost_watchdog_pressure",
		Help: "Watchdog pressure level: 0=ok, 1=warn, 2=critical",
	})
	nvlinkReady = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "greenboost_nvlink_ready",
		Help: "NVLink fabric health: 1=ready, 0=not ready",
	})
	allocationsTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "greenboost_allocations_total",
		Help: "Total DMA-BUF allocations (approximate — tracks active_buffers increases)",
	})
	evictionsTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "greenboost_evictions_total",
		Help: "Total T3 eviction events (approximate — tracks watchdog pressure transitions)",
	})
)

var (
	sysfsBase      string
	prevActiveBufs float64
	prevPressure   float64
)

func init() {
	prometheus.MustRegister(
		t1TotalBytes, t1NVLinkBytes,
		t2TotalBytes, t2UsedBytes,
		t3TotalBytes, t3UsedBytes,
		virtualTotalBytes,
		activeBuffers, kvReserveBytes,
		watchdogPressure, nvlinkReady,
		allocationsTotal, evictionsTotal,
	)
}

func main() {
	sysfsBase = getEnv("GREENBOOST_SYSFS_PATH", "/sys/class/greenboost/greenboost")
	port := getEnvInt("METRICS_PORT", defaultMetricsPort)
	metricsPath := getEnv("METRICS_PATH", defaultMetricsPath)
	scrapeIntervalSec := getEnvInt("SCRAPE_INTERVAL_SECONDS", 15)
	scrapeInterval := time.Duration(scrapeIntervalSec) * time.Second

	klog.InfoS("Starting GreenBoost Metrics Exporter",
		"sysfs", sysfsBase,
		"port", port,
		"metricsPath", metricsPath,
		"scrapeInterval", scrapeInterval)

	ctx, stop := signal.NotifyContext(context.Background(),
		syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go runMetricsCollector(ctx, scrapeInterval)

	mux := http.NewServeMux()
	mux.Handle(metricsPath, promhttp.Handler())
	mux.HandleFunc(healthPath, healthHandler)
	mux.HandleFunc("/health", healthHandler)
	mux.HandleFunc("/readyz", healthHandler)

	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", port),
		Handler: mux,
	}

	go func() {
		<-ctx.Done()
		server.Close()
	}()

	klog.InfoS("Metrics exporter listening", "addr", server.Addr)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		klog.ErrorS(err, "Metrics server failed")
		os.Exit(1)
	}
}

// runMetricsCollector collects metrics on a fixed interval.
func runMetricsCollector(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	collectMetricsOnce()

	for {
		select {
		case <-ticker.C:
			collectMetricsOnce()
		case <-ctx.Done():
			return
		}
	}
}

// collectMetricsOnce reads all sysfs attributes and updates Prometheus gauges.
func collectMetricsOnce() {
	poolText, err := readSysfsAttr(sysfsBase, "pool_info")
	if err != nil {
		klog.V(3).InfoS("Cannot read pool_info", "error", err)
		return
	}
	updateFromPoolInfo(poolText)

	// nvlink_ready — BUG-014 fix: read "nvlink_ready", NOT "compute_domain_active"
	if val, err := readSysfsAttr(sysfsBase, "nvlink_ready"); err == nil {
		if strings.TrimSpace(val) == "1" {
			nvlinkReady.Set(1)
		} else {
			nvlinkReady.Set(0)
		}
	}

	klog.V(5).InfoS("Metrics collected")
}

// updateFromPoolInfo parses pool_info sysfs text and updates gauges.
func updateFromPoolInfo(text string) {
	var (
		t1MB, t2TotalMB, t2UsedMB, t3TotalMB, t3UsedMB float64
		activeBufs, kvResMB, pressure                   float64
	)

	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "=") || strings.HasPrefix(line, "#") {
			continue
		}
		kv := strings.SplitN(line, ":", 2)
		if len(kv) != 2 {
			continue
		}
		key := strings.TrimSpace(kv[0])
		fields := strings.Fields(strings.TrimSpace(kv[1]))
		if len(fields) == 0 {
			continue
		}
		num, err := strconv.ParseFloat(fields[0], 64)
		if err != nil {
			continue
		}

		switch {
		case strings.Contains(key, "physical_vram_gb") || strings.Contains(key, "vram_physical"):
			t1MB = num * 1024
		case key == "max_pool_mb":
			t2TotalMB = num
		case key == "allocated_mb" || key == "pool_allocated_mb" || strings.Contains(key, "t2_allocated"):
			t2UsedMB = num
		case strings.Contains(key, "nvme") && strings.Contains(key, "total"):
			t3TotalMB = num
		case strings.Contains(key, "nvme") && strings.Contains(key, "alloc"):
			t3UsedMB = num
		case key == "active_buffers":
			activeBufs = num
		case key == "kv_reserve_mb":
			kvResMB = num
		case key == "t2_pressure" || key == "watchdog_pressure":
			pressure = num
		}
	}

	t1TotalBytes.Set(t1MB * 1024 * 1024)
	t2TotalBytes.Set(t2TotalMB * 1024 * 1024)
	t2UsedBytes.Set(t2UsedMB * 1024 * 1024)
	t3TotalBytes.Set(t3TotalMB * 1024 * 1024)
	t3UsedBytes.Set(t3UsedMB * 1024 * 1024)
	activeBuffers.Set(activeBufs)
	kvReserveBytes.Set(kvResMB * 1024 * 1024)
	watchdogPressure.Set(pressure)

	// Approximate counters from deltas
	if activeBufs > prevActiveBufs {
		allocationsTotal.Add(activeBufs - prevActiveBufs)
	}
	prevActiveBufs = activeBufs

	if pressure > prevPressure && pressure >= 2 {
		evictionsTotal.Add(1)
	}
	prevPressure = pressure

	// Virtual total: T1 NVLink (if set) + T2, else T1 physical + T2
	nvlText, _ := readSysfsAttr(sysfsBase, "nvlink_ready")
	if strings.TrimSpace(nvlText) == "1" {
		// NVLink pool active — report aggregated T1 + T2
		// t1NVLinkBytes is set by a future IOCTL; for now mirror physical
		t1NVLinkBytes.Set(t1MB * 1024 * 1024)
	} else {
		t1NVLinkBytes.Set(0)
	}
	virtualTotalBytes.Set((t1MB + t2TotalMB) * 1024 * 1024)
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	if _, err := readSysfsAttr(sysfsBase, "pool_info"); err != nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprintf(w, "GreenBoost sysfs not readable: %v", err)
		return
	}
	w.WriteHeader(http.StatusOK)
	fmt.Fprint(w, "OK")
}

func readSysfsAttr(base, attr string) (string, error) {
	path := base + "/" + attr
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(data)), nil
}

func getEnv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func getEnvInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

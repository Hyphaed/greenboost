// SPDX-License-Identifier: Apache-2.0
// GreenBoost Metrics Exporter v3.0 (Kubernetes / DRA sidecar).
// Exposes GreenBoost tiered memory metrics in Prometheus format.
// Reads GB_IOCTL_GET_POOL_INFO_V3 (via sysfs text fallback) and serves /metrics.
// Canonical metric reference: observability/METRICS.md
// NOTE: memory metrics use bytes (not MiB) - see METRICS.md §Unit discrepancy.

package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"regexp"
	"strconv"
	"strings"
	"sync"
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
		Help: "Total DMA-BUF allocations (approximate - tracks active_buffers increases)",
	})
	evictionsTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "greenboost_evictions_total",
		Help: "Total T3 eviction events (approximate - tracks watchdog pressure transitions)",
	})
)

var (
	sysfsBase      string
	prevActiveBufs float64
	prevPressure   float64
	prevMu         sync.Mutex
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

// Compiled once - pool_brief format:
// T1:12GB T2:1024/65536GB(1%) T3:0/0GB PRESSURE:ok KV_RSV:2048MB KV_T2:512MB
var (
	reT1     = regexp.MustCompile(`T1:(\d+)GB`)
	reT2Used = regexp.MustCompile(`T2:(\d+)/`)
	reT2Tot  = regexp.MustCompile(`T2:\d+/(\d+)GB`)
	reT3Used = regexp.MustCompile(`T3:(\d+)/`)
	reT3Tot  = regexp.MustCompile(`T3:\d+/(\d+)GB`)
	rePres   = regexp.MustCompile(`PRESSURE:(\w+)`)
)

// collectMetricsOnce reads pool_brief + individual sysfs attributes.
// pool_info no longer exists - it was replaced by pool_brief (compact one-liner)
// and per-attribute files (kv_reserve_mb, active_buffers).
func collectMetricsOnce() {
	// pool_brief: T1:12GB T2:1024/65536GB(1%) T3:0/0GB PRESSURE:ok KV_RSV:2048MB KV_T2:512MB
	if brief, err := readSysfsAttr(sysfsBase, "pool_brief"); err == nil {
		updateFromPoolBrief(brief)
	} else {
		klog.V(3).InfoS("Cannot read pool_brief", "error", err)
	}

	// kv_reserve_mb - dedicated single-integer attribute
	if val, err := readSysfsAttr(sysfsBase, "kv_reserve_mb"); err == nil {
		if n, err2 := strconv.ParseFloat(strings.TrimSpace(val), 64); err2 == nil {
			kvReserveBytes.Set(n * 1024 * 1024)
		}
	}

	// active_buffers - single integer
	if val, err := readSysfsAttr(sysfsBase, "active_buffers"); err == nil {
		if n, err2 := strconv.ParseFloat(strings.TrimSpace(val), 64); err2 == nil {
			prevMu.Lock()
			if n > prevActiveBufs {
				allocationsTotal.Add(n - prevActiveBufs)
			}
			prevActiveBufs = n
			prevMu.Unlock()
			activeBuffers.Set(n)
		}
	}

	// nvlink_ready
	if val, err := readSysfsAttr(sysfsBase, "nvlink_ready"); err == nil {
		if strings.TrimSpace(val) == "1" {
			nvlinkReady.Set(1)
		} else {
			nvlinkReady.Set(0)
		}
	}

	klog.V(5).InfoS("Metrics collected")
}

// updateFromPoolBrief parses the pool_brief compact sysfs line.
func updateFromPoolBrief(brief string) {
	const GB = 1024.0 * 1024.0 * 1024.0
	const MB = 1024.0 * 1024.0

	if m := reT1.FindStringSubmatch(brief); m != nil {
		if n, err := strconv.ParseFloat(m[1], 64); err == nil {
			t1TotalBytes.Set(n * GB)
		}
	}
	var t2UsedGB, t2TotGB float64
	if m := reT2Used.FindStringSubmatch(brief); m != nil {
		t2UsedGB, _ = strconv.ParseFloat(m[1], 64)
	}
	if m := reT2Tot.FindStringSubmatch(brief); m != nil {
		t2TotGB, _ = strconv.ParseFloat(m[1], 64)
	}
	t2UsedBytes.Set(t2UsedGB * GB)
	t2TotalBytes.Set(t2TotGB * GB)

	var t3UsedGB, t3TotGB float64
	if m := reT3Used.FindStringSubmatch(brief); m != nil {
		t3UsedGB, _ = strconv.ParseFloat(m[1], 64)
	}
	if m := reT3Tot.FindStringSubmatch(brief); m != nil {
		t3TotGB, _ = strconv.ParseFloat(m[1], 64)
	}
	t3UsedBytes.Set(t3UsedGB * GB)
	t3TotalBytes.Set(t3TotGB * GB)

	if m := rePres.FindStringSubmatch(brief); m != nil {
		pmap := map[string]float64{"ok": 0, "warn": 1, "CRITICAL": 2}
		pressure := pmap[m[1]]
		watchdogPressure.Set(pressure)
		prevMu.Lock()
		if pressure > prevPressure {
			evictionsTotal.Add(pressure - prevPressure)
		}
		prevPressure = pressure
		prevMu.Unlock()
	}

	// virtualTotalBytes = T1 + T2 (pool_brief doesn't report NVLink separately)
	t1GB := float64(0)
	if m := reT1.FindStringSubmatch(brief); m != nil {
		t1GB, _ = strconv.ParseFloat(m[1], 64)
	}
	virtualTotalBytes.Set((t1GB + t2TotGB) * GB)
	t1NVLinkBytes.Set(0) // populated by future NVLink IOCTL path
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	type sysfsResult struct{ err error }
	ch := make(chan sysfsResult, 1)
	go func() {
		_, err := readSysfsAttr(sysfsBase, "pool_brief")
		ch <- sysfsResult{err}
	}()
	select {
	case res := <-ch:
		if res.err != nil {
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprintf(w, "GreenBoost sysfs not readable: %v", res.err)
			return
		}
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "OK")
	case <-time.After(2 * time.Second):
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprint(w, "GreenBoost sysfs timeout")
	}
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

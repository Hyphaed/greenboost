# GreenBoost Prometheus Metrics Reference

Canonical naming and ownership for all `greenboost_*` Prometheus series.
Audit finding: F-L8-10 (2026-05-14).

## Exporter ownership

| Exporter | Source | Scrape target |
|---|---|---|
| **Python textfile exporter** | `greenboost_exporter.py` | Node-exporter textfile collector or HTTP `:9742/metrics` |
| **Go sysfs exporter** | `cmd/greenboost-metrics-exporter/` | HTTP `:8080/metrics` (k8s pod) |

Only one exporter should be active per node. In bare-metal deployments use the
Python exporter. In k8s deployments the Go DRA exporter runs as a sidecar.
Running both on the same node creates **duplicate time-series** with incompatible
units - see §Unit discrepancy below.

---

## Python exporter metrics (`greenboost_exporter.py`)

All memory sizes are in **mebibytes (MiB)** - a deliberate deviation from
Prometheus convention to keep human-readable numbers without unit-conversion
arithmetic in queries. Labels and semantics follow OpenMetrics naming where
possible.

### Shim availability

| Metric | Type | Description |
|---|---|---|
| `greenboost_up` | gauge | 1 if `metrics.json` is readable and parseable |
| `greenboost_shim_staleness_s` | gauge | Seconds since shim last wrote stats (>30 = no active process) |

### Tier memory (labeled `tier=T1\|T2\|T3`)

| Metric | Type | Description |
|---|---|---|
| `greenboost_tier_cur_mb` | gauge | Current MB allocated per tier |
| `greenboost_tier_peak_mb` | gauge | Peak MB per tier since shim start |
| `greenboost_tier_lifetime_mb` | counter | Lifetime MB allocated per tier |
| `greenboost_tier_alloc_count` | counter | Total allocation calls per tier |

All T1/T2/T3 series are **always emitted** (zero-filled if the tier has no data),
so alert rules and dashboards never see missing-series gaps.

### Local GPU

| Metric | Type | Description |
|---|---|---|
| `greenboost_local_t1_alloc_mb` | gauge | Local GPU VRAM in use (MB) |
| `greenboost_remote_alloc_count` | counter | Allocations routed to feeder T1 |
| `greenboost_kernel_dispatch_count` | counter | Kernels dispatched to feeder GPU |
| `greenboost_t2_pool_frag_pct` | gauge | T2 pool fragmentation 0–100 |
| `greenboost_t2_above_warn` | gauge | 1 if T2 usage > 75% warn threshold |
| `greenboost_timestamp` | gauge | Unix timestamp of last shim stats write |

### Enhancements (A2, A3, U18, E1)

| Metric | Type | Description |
|---|---|---|
| `greenboost_fake_ptr_generation` | counter | A2 fake-pointer wraparound generation |
| `greenboost_double_buffer_enabled` | gauge | 1 if U18 double-buffer T3→T2 prefetch is active |
| `greenboost_kv_compress_enabled` | gauge | 1 if E1 K/V int8 absmax compression is active |

### Per-feeder (labeled `feeder=<ip>`)

| Metric | Type | Description |
|---|---|---|
| `greenboost_bw_ewma_mbs` | gauge | U20 EWMA-measured PCIe bandwidth to feeder (MiB/s) |
| `greenboost_heartbeat_miss_total` | counter | A3 consecutive heartbeat miss count per feeder |
| `greenboost_feeder_health_state` | gauge | 0=HEALTHY 1=DEGRADED 2=UNHEALTHY 3=QUARANTINE 4=DISABLED |
| `greenboost_feeder_throttled` | gauge | 1 if feeder GPU is clock-throttled |
| `greenboost_feeder_t1_quarantined` | gauge | 1 if feeder T1 VRAM is ECC-quarantined |
| `greenboost_feeder_gpu_util_pct` | gauge | Feeder GPU utilization 0–100% from last heartbeat |

### Shim stats (from `/run/greenboost/shim_stats`)

| Metric | Type | Description |
|---|---|---|
| `greenboost_shim_h2d_mb` | counter | Host→Device DMA traffic since last shim reset (MB) |
| `greenboost_shim_d2h_mb` | counter | Device→Host DMA traffic since last shim reset (MB) |
| `greenboost_shim_headroom_mb` | gauge | VRAM headroom before T2 spillover begins (MB) |
| `greenboost_shim_cold_evicts` | counter | Cold-epoch eviction count (T1→T2 LRU demotions) |
| `greenboost_shim_kv_dedup` | counter | KV cache dedup hits (prefix cache reuse) |
| `greenboost_shim_kv_frag_mb` | gauge | KV cache internal fragmentation (MB) |
| `greenboost_shim_t2_frag_pct` | gauge | T2 fragmentation-adjusted warning threshold (%) |
| `greenboost_shim_remote_alloc_count` | counter | Allocations offloaded to cluster feeder(s) |
| `greenboost_shim_remote_alloc_mb` | counter | Total MB offloaded to cluster feeder(s) |

### Sysfs (from `pool_brief` and `kv_reserve_mb` sysfs attributes)

Sources: `/sys/class/greenboost/greenboost/pool_brief` (compact summary line) and
`/sys/class/greenboost/greenboost/kv_reserve_mb` (dedicated single-integer attribute).
The human-readable `status` attribute is **not** parsed by the exporter.

Note: `pool_brief` reports T2 allocated in GB (sub-GB precision is lost).
Note: `kv_used_mb` has no machine-readable sysfs source and is **not** emitted.

| Metric | Type | Description |
|---|---|---|
| `greenboost_t2_pressure` | gauge | T2 DDR pressure level: 0=ok 1=warn 2=critical |
| `greenboost_swap_pressure` | gauge | T3 NVMe swap pressure level: 0=ok 1=warn 2=critical (same atomic as t2_pressure in kernel) |
| `greenboost_kv_t2_mb` | gauge | KV cache spilled to T2 DDR (MB) |
| `greenboost_kv_reserve_mb` | gauge | KV cache T1 VRAM hard reserve (MB) |
| `greenboost_t2_allocated_mb` | gauge | T2 DDR pool currently allocated (MB, GB-resolution) |

---

## Python telemetry layer (`gb_telemetry.py`)

`gb_telemetry.TelemetryManager` is the live in-process telemetry layer used by
`gb_init`, `gb_quant`, `gb_diffusion_orch`, and `gb_llm`. It is separate from
the Prometheus exporters above and produces `GpuMetrics` snapshots read by the
Python optimization modules.

### Provider architecture

Four providers are tried in order; each fills fields it can supply:

| Provider | Requires | Fields |
|---|---|---|
| `NVMLProvider` | pynvml | `power_w`, `temp_c`, `fb_total_mb`, `fb_free_mb`, `fb_used_mb`, `gpu_util_pct` |
| `DCGMProvider` | dcgm Python bindings, embedded host engine | `power_w` (FI 157), `ecc_sbe_volatile` (FI 310), `ecc_dbe_volatile` (FI 311), `ecc_dbe_aggregate` (FI 313), `nvlink_bw_tx_mbs` (FI 449), `health_ok`, `health_summary` |
| `GreenBoostProvider` | `/dev/greenboost` + `gb_telemetry.GB_IOCTL_GET_INFO` | `t1_used_mb`, `t2_used_mb`, `t3_used_mb`, `fb_free_mb` override |
| `TorchFallbackProvider` | torch.cuda | `fb_total_mb`, `fb_free_mb` (last resort) |

### DCGM embedded mode

`DCGMProvider` uses `dcgmStartEmbedded(DCGM_OPERATION_MODE_AUTO)` — no dcgmd
daemon required. This works on bare-metal workstations and cluster nodes alike.

**Field watch intervals**: DCGM fields are added via `dcgmWatchFields` with a
1-second update interval. The Python polling loop runs at 500 ms; DCGM
values are sampled at each poll.

**Health check**: `dcgmHealthSet_v2` configures the following systems:

| System | Mask bit | RTX 5070 |
|---|---|---|
| PCIe | 0x01 | Active |
| PMU (power management unit) | 0x04 | Active |
| Memory | 0x10 | Active |
| SM (streaming multiprocessor) | 0x20 | Active |
| InfoROM | 0x40 | Active |
| Thermal | 0x80 | Active |
| Power | 0x100 | Active |
| Driver | 0x200 | Active |
| NVLink / NVSwitch / ConnectX | — | **Omitted** — RTX 5070 has no NVLink hardware; fields return 0 |

Health check runs every 60 telemetry polls (~30 s) to avoid interfering with
Triton kernel launches.

**ECC fields**: FI 310 (`ecc_sbe_volatile`) tracks single-bit errors
(correctable, counted for wear monitoring). FI 311 (`ecc_dbe_volatile`) and
FI 313 (`ecc_dbe_aggregate`) track double-bit errors (uncorrectable, hardware
memory corruption). Any `ecc_dbe_volatile > 0` triggers the ECC DBE callback
registered by `gb_init` (loud stderr warning before inference starts).

**Cluster mode**: `ClusterTelemetryManager` creates one `TelemetryManager` per
GPU, using `dcgmGroupCreate` / `dcgmGroupAddDevice` so a single DCGM session
covers all devices. Health checks span the full group.

**NVLink BW (FI 449)**: queried but returns 0 on RTX 5070 (no NVLink hardware).
On V100 NVLink clusters this field reflects aggregate NVLink bandwidth.

### `GpuMetrics` fields relevant to optimization modules

| Field | Type | Source | Used by |
|---|---|---|---|
| `fb_free_mb` | int | NVML / GB / Torch | `quantize_to_fit` budget tightening |
| `should_demote` | bool | NVML (fb_free_mb < headroom) | `ModelTierManager.auto_evict` |
| `ecc_dbe_volatile` | int | DCGM FI 311 | `pre_inference_check` — blocks generation |
| `ecc_critical` | bool | `dbe_volatile > 0` | ECC guard callback |
| `health_ok` | bool | DCGM health check | `pre_inference_check` — warns |
| `health_summary` | str | DCGM health check | Logged to stderr |
| `power_w` | float | NVML / DCGM FI 157 | Telemetry log, cluster monitoring |
| `temp_c` | float | NVML | Telemetry log |

---

## Go sysfs exporter metrics (`cmd/greenboost-metrics-exporter/`)

All memory sizes are in **bytes** - standard Prometheus convention.
This exporter is used in Kubernetes deployments as a DRA driver sidecar.

| Metric | Type | Description |
|---|---|---|
| `greenboost_t1_total_bytes` | gauge | Physical VRAM total capacity (T1 tier) |
| `greenboost_t1_nvlink_bytes` | gauge | Aggregated NVLink T1 capacity (0 if disabled) |
| `greenboost_t2_total_bytes` | gauge | DDR pool total capacity (T2 tier) |
| `greenboost_t2_used_bytes` | gauge | DDR pool in use (T2 tier) |
| `greenboost_t3_total_bytes` | gauge | NVMe/Lustre T3 total capacity |
| `greenboost_t3_used_bytes` | gauge | NVMe/Lustre T3 in use |
| `greenboost_virtual_total_bytes` | gauge | Total virtual VRAM (T1 NVLink + T2, or physical + T2) |
| `greenboost_active_buffers` | gauge | Number of live DMA-BUF objects |
| `greenboost_kv_reserve_bytes` | gauge | KV cache T1 reserve |
| `greenboost_watchdog_pressure` | gauge | Watchdog pressure: 0=ok 1=warn 2=critical |
| `greenboost_nvlink_ready` | gauge | NVLink fabric health: 1=ready 0=not ready |
| `greenboost_allocations_total` | counter | Total DMA-BUF allocations (approximate delta) |
| `greenboost_evictions_total` | counter | Total T3 eviction events (approximate delta) |

---

## Unit discrepancy - known issue (F-L8-10)

The Python and Go exporters overlap on T2 DDR and KV cache metrics but use
different units:

| Concept | Python exporter | Go exporter |
|---|---|---|
| T2 allocated | `greenboost_t2_allocated_mb` (MiB) | `greenboost_t2_used_bytes` (bytes) |
| KV cache reserve | `greenboost_kv_reserve_mb` (MiB) | `greenboost_kv_reserve_bytes` (bytes) |

**Do not run both exporters on the same node.** If both are active, Prometheus
will ingest two incompatible series for the same quantity, causing incorrect
aggregation in cross-node dashboards.

Migration path (tracked as F-L8-10):
1. Bare-metal nodes: Python exporter only.
2. k8s nodes: Go exporter only; the DRA driver Helm chart installs it automatically.
3. Future: unify both exporters to bytes-based naming in a single Go binary that
   also reads `metrics.json` and `shim_stats`; retire the Python exporter.

---

## Grafana dashboards

| Dashboard | UID | Panels |
|---|---|---|
| Cluster Overview | `greenboost-cluster` | Tier usage, feeder BW, feeder health, shim staleness, KV dedup rate, evictions, heartbeat misses |

Dashboard source: `observability/grafana/greenboost-cluster.json`.

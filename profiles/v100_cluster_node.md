---
profile_type: cluster_node
profile_name: V100 Cluster Node

hardware:
  gpu_model: "NVIDIA V100 SXM2 32GB"
  gpu_count_per_node: 8
  gpu_compute_capability: 7.0
  nvlink_generation: 2.0
  interconnect: "Mellanox ConnectX-6 200 Gb/s InfiniBand"
  ddr4_gb: 384
  ddr_type: "DDR4 ECC"
  nvme_local_gb: 1000
  lustre_enabled: true

greenboost:
  use_hugepages: false  
  pcores_only: false
  nvlink_pool: true
  virtual_vram_gb: 307
  safety_reserve_gb: 20
  tier3_backend: "lustre"
  nvme_swap_gb: 0
  cu_mem_alloc_async: false

features:
  enable_dra: true
  enable_nvlink_fabric_monitor: true
  enable_compute_domain_aware_tiering: true
  enable_prometheus_exporter: true

---
# V100 Cluster Node Profile

## Hardware Configuration

**Per Node:**
- 8× NVIDIA V100 SXM2 32 GB (256 GB HBM2 total)
- 384 GB DDR4 ECC RAM
- 2× 1 TB Enterprise U.2 NVMe (RAID1)
- 200 Gb/s Mellanox ConnectX-6 InfiniBand

**Ibridging:** Through NVLink 2.0 fabric (8×8 all-to-all topology)

## GreenBoost Memory Tiers

**Tier 1 (T1):** 256 GB HBM2 - NVLink unified pool (8× 32 GB)
**Tier 2 (T2):** ~307 GB DDR4 - DMA-BUF pinned pool
**Tier 3 (T3):** Lustre parallel FS - high-speed distributed storage

**Virtual VRAM per node:** 563 GB (256 + 307)

## Profile Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `use_hugepages` | false | ECC DDR4 may limit THP; safe to disable |
| `pcores_only` | false | Xeon typically has no P/E split |
| `nvlink_pool` | true | Aggregate 8× V100 VRAM into single T1 pool |
| `virtual_vram_gb` | 307 | T2 size (384 GB system RAM - 20 GB reserve - 57 GB overhead) |
| `safety_reserve_gb` | 20 | Minimum free RAM safety reserve |
| `tier3_backend` | lustre | Use Lustre for T3 (cluster-scale performance) |
| `nvme_swap_gb` | 0 | Rely on Lustre instead of local NVMe |
| `cu_mem_alloc_async` | false | V100 cc 7.0 doesn't support async alloc |

## Enabled Features

### DRA Integration
- `enable_dra`: Dynamic Resource Allocation (Kubernetes 1.32+)
- GreenBoost exposes T1+T2 (563 GB) as `greenboost.nvidia.com` device class
- Kubelet plugin manages ResourceClaim requests

### NVLink Fabric Monitoring
- `enable_nvlink_fabric_monitor`: Query IMEX/NVLink state
- Validate NVLink fabric is in READY state before enabling
- Fallback to per-GPU allocation if fabric not ready

### ComputeDomain-Aware Tiering
- `enable_compute_domain_aware_tiering`: Detect IMEX ComputeDomain status
- Set GB_ALLOC_KV_CACHE flag for pods in ComputeDomain
- Prioritize T2 for KV cache to minimize IMEX overhead

### Prometheus Metrics
- `enable_prometheus_exporter`: Metrics at :8080/metrics endpoint
- Export T1/T2/T3 usage, evictions, watchdog pressure, NVLink health
- Integrates with Grafana dashboards

## Performance Expectations

**LLM Inference (30B model, 198K context):**
- 8–20 tok/s decode with GreenBoost + ExLlamaV3
- 2–8 s TTFT (time to first token)
- 200 PFLOPS FP16 cluster aggregate (200 nodes)

**Scaling:**
- 51.2 TB physical GPU memory
- 76.8 TB system RAM
- **112 TB virtual VRAM** (2.2× physical)
- 200 GB/s HDR InfiniBand fabric for MNNVL

## NVLink V100 Topology Note (goodv11_plan.md audit — BUG-009)

**V100 uses NVLink 2.0 direct P2P — NOT NVSwitch fabric:**
- `nvmlDeviceGetGpuFabricInfo()` → `GPU_FABRIC_STATE_NOT_SUPPORTED` on all V100s
- NVLink readiness verified by the kubelet plugin via `nvmlDeviceGetP2PStatus()` across all 28 GPU pairs
- Plugin writes `1` to `/sys/class/greenboost/greenboost/nvlink_ready` after P2P verification
- IMEX channels NOT supported on V100 (IMEX requires NVSwitch: A100/H100/H200/GB200)
- Multi-node communication: NCCL over InfiniBand (not IMEX)

## Installation Notes

For complete deployment on a 200-node V100 cluster, see `k8s-deployment/monitoring.md`.
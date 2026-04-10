---
profile_version: "1.0"
profile_name: "v100_cluster_node"
profile_type: "cluster_node"
cluster_name: "v100-datacenter-rack01"
node_role: "compute"
node_count_in_cluster: 200
created: "2026-03-21T00:00:00Z"
generated_by: "user"
greenboost_version: "2.6"
---

## Hardware

### CPU
cpu_model: "Dual Intel Xeon Scalable (Skylake-SP)"
cpu_arch: x86_64
physical_cores: 28
logical_cpus: 56
p_cores: 0
e_cores: 0
has_pe_split: false
numa_nodes: 2

### GPU
gpu_count: 8
gpu_model: "NVIDIA Tesla V100 SXM2 32GB"
vram_gb: 32
memory_bw_gbps: 900
compute_capability: "7.0"
gpu_arch: "Volta"
pcie_gen: 3
pcie_lanes: 16
nvlink: true
nvlink_gen: 2
nvlink_bw_gbps: 300
nvlink_topology: "all-to-all (NVSwitch fabric)"
driver_version: "580.126"
cuda_version: "13.0"

### RAM
ram_total_gb: 384
ram_type: DDR4
ram_speed_mt: 2666
ram_channels: 6
ram_ecc: true

### Storage
nvme_count: 2
nvme_0_model: "Enterprise U.2 NVMe 1TB (RAID1)"
nvme_0_capacity_tb: 1
nvme_0_seq_read_gbps: 3.5
nvme_0_pcie_gen: 3
nvme_0_pcie_lanes: 4

### Networking
ib_adapter: "Mellanox ConnectX-6 (x2)"
ib_speed_gbps: 200
ib_protocol: InfiniBand
mpi_enabled: true
mpi_impl: OpenMPI
storage_net_adapter: "Mellanox ConnectX-5 100GbE"
parallel_fs: lustre

### OS
os_distro: "RHEL 8.x"
kernel_version: "4.18"

## GreenBoost Parameters
# Per-node values. NVLink makes all 8x V100 appear as single 256 GB T1 pool.

physical_vram_gb: 256
virtual_vram_gb: 307
safety_reserve_gb: 48
nvme_swap_gb: 512
nvme_pool_gb: 460
use_hugepages: 0
pcores_only: 0
debug_mode: 0
tier3_backend: lustre
nvlink_pool: true

## Cluster Aggregate Capabilities
# Informational only — GreenBoost operates per-node.

cluster_total_gpus: 1600
cluster_total_gpu_memory_tb: 51.2
cluster_total_ram_tb: 76.8
cluster_ai_tflops_fp16: 200000
cluster_ib_fabric: "200 Gb/s HDR InfiniBand"

## Workload Hints

target_models:
  - name: "LLaMA-3 405B"
    size_gb: 800
    distribution: "tensor_parallel_8"
    context_k: 128
  - name: "Mixtral 8x22B"
    size_gb: 280
    distribution: "pipeline_parallel_4"

## Profile Notes

Compute node in 20-rack V100 cluster. GreenBoost extends per-node T1 across
NVLink (8 GPUs unified) and DDR4 pool (307 GB via DMA-BUF). Lustre parallel
filesystem serves as T3 instead of local NVMe (higher sustained throughput for
large model weights). cuMemAllocAsync disabled (V100 compute capability 7.0 <
8.0 requirement). ECC DDR4 — THP disabled. Xeon has no P/E split — pcores_only=0.

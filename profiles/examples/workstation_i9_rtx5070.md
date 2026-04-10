---
profile_version: "1.0"
profile_name: "workstation_i9_rtx5070"
profile_type: "workstation"
created: "2026-03-21T00:00:00Z"
generated_by: "user"
greenboost_version: "2.6"
---

## Hardware

### CPU
cpu_model: "Intel Core i9-14900KF"
cpu_arch: x86_64
physical_cores: 24
logical_cpus: 32
p_cores: 8
e_cores: 16
p_core_cpus: "0-15"
e_core_cpus: "16-31"
max_freq_ghz: 6.0
l3_cache_mb: 36
numa_nodes: 1
has_pe_split: true

### GPU
gpu_count: 1
gpu_model: "NVIDIA RTX 5070 (GB205)"
vram_gb: 12
memory_bw_gbps: 336
compute_capability: "12.0"
gpu_arch: "Blackwell"
pcie_gen: 4
pcie_lanes: 16
nvlink: false
driver_version: "580.126"
cuda_version: "13.0"

### RAM
ram_total_gb: 64
ram_type: DDR4
ram_speed_mt: 3600
ram_channels: 2
ram_ecc: false

### Storage
nvme_count: 1
nvme_0_model: "Samsung 990 EVO Plus 4TB (PM9C1a)"
nvme_0_capacity_tb: 4
nvme_0_seq_read_gbps: 7.25
nvme_0_pcie_gen: 4
nvme_0_pcie_lanes: 4

### OS
os_distro: "Ubuntu 26.04"
kernel_version: "6.19"

## GreenBoost Parameters

physical_vram_gb: 12
virtual_vram_gb: 51
safety_reserve_gb: 12
nvme_swap_gb: 128
nvme_pool_gb: 114
use_hugepages: 1
pcores_only: 1
debug_mode: 0
tier3_backend: nvme
kv_reserve_mb: 6144    # 6 GB for 128K context; shim auto-scales from OLLAMA_NUM_CTX

## Ollama / Inference Runtime

ollama_flash_attention: 1
ollama_kv_cache_type: q8_0
ollama_num_ctx: 131072

## Workload Hints

target_models:
  - name: "glm-4.7-flash:q8_0"
    size_gb: 31.8
    layers: 48
  - name: "nemotron-3-super:120b"
    size_gb: 87
    context_k: 256

## Profile Notes

Home workstation. T1=12 GB VRAM, T2=51 GB pinned DDR4, T3=128 GB NVMe swap.
PCIe 4.0 x16 ~32 GB/s DMA bandwidth between GPU and DDR4 pool.
Expected throughput: 3-8 tok/s for 30B models, 1-2 tok/s for 120B models.

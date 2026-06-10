# GreenBoost Observability Pack

Starter artifacts implementing AUDIT_2026-05-13.md §8 / W10. Everything here is
**opt-in** and **out-of-tree from the build** - nothing must be enabled for
GreenBoost itself to work.

## Layout

```
observability/
├── ebpf/                      # bpftrace one-liners; require root, no recompile
│   ├── cuda_uprobe.bt         # F-L6-01 - libcuda call-latency histograms
│   ├── tcp_msg_latency.bt     # F-L6-02 - TCP RTT per GB_MSG_* via tcp_sendmsg
│   └── ddr_pressure.bt        # F-L6-03 - __alloc_pages_slowpath early-warn
├── grafana/
│   └── greenboost-cluster.json  # F-L6-13 / F-L6-15 - overview dashboard
├── perf/
│   └── perf_record_shim.sh    # F-L6-09 / F-L6-11 - flamegraph capture
└── README.md                  # this file
```

## Quick start

```bash
# 1. Live CUDA call latency (Ctrl-C to dump histograms)
sudo bpftrace observability/ebpf/cuda_uprobe.bt

# 2. TCP feeder RTT
sudo bpftrace observability/ebpf/tcp_msg_latency.bt

# 3. DDR pressure early-warning
sudo bpftrace observability/ebpf/ddr_pressure.bt

# 4. Profile the shim during inference
sudo observability/perf/perf_record_shim.sh "$(pgrep -f ollama)" 10
# → /tmp/greenboost-shim.svg

# 5. Import the Grafana dashboard
# Grafana → Dashboards → Import → upload greenboost-cluster.json
# Pick your Prometheus data source.
```

## Prerequisites

- bpftrace ≥ 0.18 (Debian/Ubuntu: `apt install bpftrace`)
- linux-tools-common (perf) matching the running kernel
- FlameGraph scripts in PATH (optional but recommended for `perf_record_shim.sh`)
- A Prometheus instance scraping `greenboost_exporter.py --serve`

## What is intentionally not here yet

The audit (§8) also calls out items that need real engineering, not just a
config file:

- USDT probes inside the shim and daemon (F-L6-06)
- Per-feeder p99 bandwidth metric (F-L6-19)
- KV cache hit-rate metric publication (F-L6-20)
- Cluster topology Graphviz tool (F-L6-17)
- TUI dashboard for SSH-only ops (F-L6-16)

These are tracked in AUDIT_2026-05-13.md §10 W10 and a follow-up workstream.

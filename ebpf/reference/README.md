# ebpf/reference — vendored bpf_uvm

Reference files from [bpf_uvm](https://github.com/vchuravy/bpf_uvm) (MIT licence).

Used to document:
- `uvm_perf_event_notify` kprobe attachment point
- UVM migration cause codes (Replayable/Non-replayable Fault, Access Counter,
  Prefetch, Eviction, API Migrate, API Set Range Group, API Hint)
- `/proc/driver/nvidia-uvm/gpus/<uuid>/fault_stats` procfs format

GreenBoost's primary observability target is greenboost.ko itself (pinned
DDR / DMA-BUF paths), not UVM.  The `uvm_perf_event_notify` kprobe in
`gb_trace.bpf.c` is attached opportunistically and gives fault rates when
other processes use `cudaMallocManaged`.

These files are kept as-is (BCC Python style) for reference; GreenBoost
uses the CO-RE/libbpf approach in `gb_trace.bpf.c` / `gb_trace.c`.

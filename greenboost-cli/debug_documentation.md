# GreenBoost Debug Documentation

GreenBoost routes CUDA allocations across three memory tiers:
- **T1** — GPU VRAM (11 GB, RTX 5070 Blackwell)
- **T2** — System DDR pool (42 GB, registered via `cuMemHostRegister` Path B)
- **T3** — NVMe swap (73 GB, emergency overflow — ~100× slower, avoid)

Virtual VRAM reported to processes: **~126 GB** (T1 + T2 + T3).

---

## Quick Commands

```bash
gb vitals              # full live debug panel
gb status              # compact tier summary
gb tiers               # detailed tier breakdown

/gb-vitals             # same as above, inside the REPL
/gb-vitals --export /tmp/debug.md  # export full snapshot to Markdown
/gb-status
/gb-tiers
/gb-pool-cap auto      # auto-set T2 pool cap from available RAM
/gb-kv-reserve 2048    # set KV cache T1 reserve (MB)
/turboquant on|off     # toggle 3-bit KV compression
```

Dashboard: `gb web` → http://localhost:7821/greenboost

---

## Runtime Files

| Path | Description |
|------|-------------|
| `/run/greenboost/shim_stats` | Live shim counters (key=value), updated every ~2s |
| `/run/greenboost/nvtx_events.log` | Chronological NVTX event log (allocations, phase changes) |
| `/run/greenboost/metrics.json` | Same data as shim_stats in JSON (for Prometheus exporter) |
| `/run/greenboost/phase` | Current phase string (`INFERENCE`, `STEADY`, `OOM`, `INIT`) |
| `/run/greenboost/diffuser_vitals.json` | FLUX/SD diffusion pipeline state (written by pipeline.py) |
| `/etc/greenboost/cluster.conf` | Feeder node addresses for multi-GPU aggregation |
| `/sys/module/greenboost/` | Kernel module sysfs (if module loaded) |
| `/sys/class/greenboost/greenboost/` | Per-tier sysfs stats |
| `/dev/greenboost` | IOCTL device for pool cap + KV reserve control |

---

## Reading `shim_stats`

```
pid=3525910              # PID of the process with the shim loaded
initialized=1            # 1 = shim fully active, 0 = constructor ran but init deferred
vllm_compat=1            # 1 = VCM-01 deferred-init mode (GREENBOOST_VLLM_COMPAT=1)
virtual_vram_mb=117760   # virtual VRAM reported to cuMemGetInfo (T1+T2+T3 combined)
active_path=B            # allocation path in use: A0 (DMA-BUF), B (pinned RAM), none
phase=INFERENCE          # current allocation phase

path_a_count=0           # allocs via Path A (DMA-BUF) — disabled on Blackwell CC≥12
path_b_count=142         # allocs via Path B (cuMemHostRegister) — the active T2 path
tier_t2_local_cur_mb=4096  # T2 DDR currently allocated (MB)
tier_t2_local_peak_mb=8192 # T2 DDR peak allocation this session
h2d_mb=1024              # total host→device transfers (MB)
d2h_mb=512               # total device→host evictions (MB)
kernel_dispatch_count=88 # number of prefetch kernel dispatches
t2_pool_frag_pct=3       # T2 pool fragmentation %
cold_epoch_evict_count=0 # cold buffer evictions (page-out to T3)
vram_headroom_mb=244     # bytes before T1 VRAM is full (triggers T2 overflow)
kv_t1_tracked_mb=3056    # KV cache bytes in T1 reserve zone
```

**What to look for:**
- `path_b_count > 0` → T2 is actively being used (good when model > 12 GB)
- `tier_t2_local_cur_mb > 0` → DDR overflow active
- `vram_headroom_mb < 500` → about to overflow; GreenBoost will route next alloc to T2
- `cold_epoch_evict_count > 0` → T3 NVMe paging active (slow!)
- `t2_pool_frag_pct > 30` → pool fragmentation; consider `/gb-pool-cap auto` to reset
- `initialized=0` in vLLM process → shim loaded but deferred init not yet triggered

---

## NVTX Event Types

Events are logged to `/run/greenboost/nvtx_events.log` in format:
```
<timestamp_ms>  <EVENT_TYPE>  <tier>  <size_mb>  <pid>  <detail>
```

| Event | Tier | Meaning |
|-------|------|---------|
| `SHIM_INIT` | — | Shim constructor completed (or deferred) |
| `ALLOC_T1_LOCAL` | T1 | Normal GPU VRAM allocation |
| `ALLOC_T1_VMM` | T1 | PyTorch VMM allocation (cuMemCreate) — only when expandable_segments:True |
| `ALLOC_T2_LOCAL` | T2 | DDR overflow allocation via Path B |
| `ALLOC_T2_FEEDER` | T2 | Allocation on a remote feeder node |
| `ALLOC_T3_LOCAL` | T3 | NVMe swap allocation (slow!) |
| `EVICT_T1→T2` | T1 | Cold buffer moved from VRAM to DDR |
| `EVICT_T2→T3` | T2 | Cold buffer moved from DDR to NVMe |
| `PHASE_STEADY` | — | Allocation phase stabilised (model fully loaded) |
| `PHASE_INFERENCE` | — | Active inference detected |
| `PHASE_OOM` | — | OOM guard triggered — inference blocked until pool clears |
| `PHASE_DEEP_IDLE` | — | No allocations for >30s |
| `KV_COMPRESS` | — | TurboQuant K/V compression applied |
| `RESET` | — | Pool cleared by OOM guard |
| `FEEDER_CONNECT` | — | New feeder node registered |

**Live tail:**
```bash
tail -f /run/greenboost/nvtx_events.log
# or inside gb:
gb nvtx-logs
```

---

## Monitor Detection Chain

`GreenBoostMonitor._detect()` tries three paths in order:

1. **ioctl** (`/dev/greenboost`) — fastest, used when the caller has device access.
   `max_pool_mb` returns 0 when no explicit pool cap is set — pool sizes come from sysfs.
2. **sysfs module** (`/sys/module/greenboost/`) — used when ioctl fails (EPERM, no device).
   Reads sysfs stats + `/sys/class/greenboost/greenboost/status` for pool totals.
3. **DKMS check** — falls back to `dkms status` to confirm module presence.

Both paths 1 and 2 call `_fill_from_sysfs_status()` which parses the human-readable
`/sys/class/greenboost/greenboost/status` file to fill `ram_pool_mb` (T2 total),
`ram_available_mb` (T2 free), `nvme_swap_total_mb` (T3 total), and `nvme_swap_used_mb`.

---

## KV T3 Stale Counter

**Symptom:** `gb vitals` or `greenboost vitals` shows `KV in T3: 14944 MB` even though
T3 NVMe is 0% used and generation speed is normal.

**Root cause:** The kernel module's `kv_t3_tracked_mb` counter is incremented when KV
cache spills to T3 during inference but is **not reset when the process exits**. The stale
value persists until the next process that actually uses T3 allocates and clears it.

**Python workaround (in `monitor.py`):** `_fill_from_sysfs_status()` reads both
`KV in T3` and `T3 allocated` from sysfs. If `KV in T3 > 0 MB` but `T3 allocated == 0 MB`,
the counter is stale — `GreenBoostStatus._kv_t3_stale = True` is set to suppress
the false alarm in `gb vitals` and dashboard displays.

**Kernel-level fix:** Requires adding counter reset in the kernel module's process-exit
cleanup path (not yet implemented). The Python guard is the current workaround.

**Verifying:**
```bash
cat /sys/class/greenboost/greenboost/status | grep -E "KV in T3|T3 allocated"
# If "KV in T3 : N MB" but "T3 allocated : 0 MB" → stale counter, safe to ignore
```

---

## T2 Activation Debugging

**T2 not activating (vLLM sees only real VRAM):**

1. Check shim is injected:
   ```bash
   cat /proc/$(pgrep -f vllm | head -1)/maps | grep greenboost
   ```
   If empty → shim not loaded. Fix: ensure `LD_PRELOAD` is set in vLLM env.

2. Check `initialized` in shim_stats:
   ```bash
   grep initialized /run/greenboost/shim_stats
   ```
   If `initialized=0` with VCM-01 mode → shim deferred, waiting for first cudaMalloc.
   This is normal — init completes on the first CUDA allocation after CUDA is resident.

3. Check expandable_segments:
   ```bash
   python3 -c "import torch; print(torch.cuda.memory.get_allocator_backend())"
   # Must NOT show 'cudaMallocAsync'
   ```
   If wrong: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` must be set before torch import.

4. Check `cuMemGetInfo_v2` override:
   ```bash
   # In the vLLM process, virtual VRAM should be ~115 GB not ~12 GB
   # Look for this in vllm startup log:
   grep -i "available memory\|gpu memory\|free memory" /tmp/vllm-*.log
   ```

5. Check AppArmor denials:
   ```bash
   journalctl -k --grep="apparmor.*greenboost" -n 20
   # If denials: sudo greenboost install-sys-configs
   ```

---

## VCM-01 — vLLM Compatibility Mode

**Problem:** Explicit `LD_PRELOAD=libgreenboost_cuda.so` caused `CUDA error: invalid device context`
because `gb_shim_init()` Stage 2 force-loaded `libcuda.so.1` in the constructor, creating a CUDA
context before vLLM's EngineCore worker subprocess initialised its own.

**Solution (VCM-01):** When `GREENBOOST_VLLM_COMPAT=1` is set, the shim constructor skips the
force-load of `libcuda.so.1`. Instead, `gb_try_resume_deferred()` is called from every CUDA API
stub (`cudaMalloc`, `cuMemGetInfo_v2`, etc.). On the first call after CUDA is resident, it detects
`libcuda.so.1` loaded via `dlopen(..., RTLD_NOLOAD)` and completes the full init exactly once via
`pthread_once`.

**Env vars for vLLM:**
```bash
LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so
GREENBOOST_VLLM_COMPAT=1       # deferred init — avoids CUDA context conflict
GREENBOOST_ACTIVE=1             # tells shim to route to T2 when T1 full
GREENBOOST_VIRTUAL_VRAM_MB=43008  # T2 pool size to report via cuMemGetInfo
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False  # required for cudaMalloc hooking
```

These are automatically set by `greenboost-cli` when `model=vllm/...` is configured.
See `greenboost_cli/slash_commands/backend_cmds.py:_build_vllm_env()`.

---

## Allocation Paths

| Path | Mechanism | Blackwell (CC≥12) | Notes |
|------|-----------|-------------------|-------|
| A0 | DMA-BUF (anon) | Disabled | Fast, zero-copy, not available on new GPUs |
| A | DMA-BUF (device fd) | Disabled | Same |
| B | `cuMemHostRegister` | **Active** | Pins DDR pages; slower than A but works everywhere |
| C | UVM `cuMemAllocManaged` | Removed | Too slow, removed from shim |

On Blackwell (RTX 5070, CC 12.0), only Path B is available. T2 allocations are pinned DDR pages
registered with the CUDA driver via `cuMemHostRegister`. Bandwidth is ~40-60 GB/s vs ~900 GB/s for
T1 VRAM, so large weight transfers are slower but functional.

---

## AppArmor

GreenBoost installs two abstraction layers:

1. **`/etc/apparmor.d/abstractions/greenboost-audit`** — included by `base`, grants:
   - `/usr/local/lib/libgreenboost_cuda.so mr` — shim mapping
   - `/usr/local/lib/libgreenboost_audit.so mr` — LD_AUDIT stub
   - `/run/greenboost/ r` and `/run/greenboost/** rw` — runtime files
   - `/dev/greenboost rw` — IOCTL device

2. **Snap profile patch** — needed for snap-confined apps (e.g. `snap-confine`).

**Checking denials:**
```bash
sudo ausearch -ts recent -m avc | grep greenboost
journalctl -k --grep="apparmor" | grep greenboost | tail -20
```

**Fixing:**
```bash
sudo greenboost install-sys-configs
# or manually:
sudo apparmor_parser -r /etc/apparmor.d/abstractions/greenboost-audit
```

---

## GreenBoost Group

The `/dev/greenboost` device should be owned by the `greenboost` group for non-root IOCTL access:

```bash
# Create group (one-time)
sudo groupadd -r greenboost
sudo usermod -aG greenboost $USER

# Update udev rule
sudo sed -i 's/GROUP="video"/GROUP="greenboost"/' /etc/udev/rules.d/99-greenboost.rules
sudo udevadm control --reload-rules && sudo udevadm trigger /dev/greenboost

# Verify
ls -la /dev/greenboost
# should show: crw-rw---- ... root greenboost
```

---

## Diffuser Vitals

The FLUX/SD pipeline writes live state to `/run/greenboost/diffuser_vitals.json`:

```json
{
  "ts": 1234567890.1,
  "state": "generating",
  "pipeline": "FLUX",
  "model": "flux1-nf4",
  "pid": 12345,
  "vram_alloc_mb": 8192,
  "vram_reserved_mb": 9216,
  "vram_peak_mb": 10240,
  "t2_alloc_mb": 4096,
  "gen_step": 12,
  "gen_total_steps": 28,
  "last_gen_s": 0,
  "last_prompt": "pirate ship at sunset, cinematic..."
}
```

States: `loading` → `ready` → `generating` → `ready`

Visible in:
- `gb vitals` (terminal)
- `gb web` → http://localhost:7821/greenboost → **Diffuser Vitals** section

---

## vLLM Serve Debugging

```bash
# Start vLLM for NVFP4 Qwen3 model
gb serve Brian6145/Qwen3.6-27B-Claude-Opus-Sonnet-Distilled-NVFP4-MTP

# Or manually:
GREENBOOST_ACTIVE=1 \
GREENBOOST_VLLM_COMPAT=1 \
GREENBOOST_VIRTUAL_VRAM_MB=43008 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
vllm serve Brian6145/Qwen3.6-27B-Claude-Opus-Sonnet-Distilled-NVFP4-MTP \
  --quantization compressed-tensors \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

# Check logs
/vllm-serve logs

# Check shim is active in vLLM process
grep initialized /run/greenboost/shim_stats
cat /proc/$(pgrep -f vllm | head -1)/maps | grep greenboost

# Check T2 usage
grep tier_t2 /run/greenboost/shim_stats
```

**Expected sequence after launch:**
1. `initialized=0` → shim loaded, VCM-01 deferred init pending
2. First CUDA alloc in vLLM worker → `gb_try_resume_deferred()` completes init
3. `initialized=1` → shim active
4. vLLM startup check: `cuMemGetInfo_v2` returns `virtual_vram_mb` (43 GB or full ~115 GB)
5. Model loading: `path_b_count` starts rising, `tier_t2_local_cur_mb` grows
6. Phase → `STEADY` → `INFERENCE`

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| vLLM sees 11.5 GB VRAM at startup | Shim not loaded in vLLM process | Check `LD_PRELOAD` in `_build_vllm_env()` |
| `CUDA error: invalid device context` | Shim constructor creates CUDA context before vLLM's worker | Add `GREENBOOST_VLLM_COMPAT=1` |
| OOM at ~7.6 GB loaded | T2 not active (shim loaded but `initialized=0`) | Check VCM-01 init triggered |
| `path_b_count=0` after model load | T2 not being used; model fits in T1 | Normal for small models |
| AppArmor denials in journal | Profile missing `libgreenboost_cuda.so mr` | `sudo greenboost install-sys-configs` |
| T3 NVMe active, generation slow | DDR pool exhausted | Increase T2 pool: `/gb-pool-cap auto` |
| `vram_headroom_mb` very low | T1 almost full, T2 overflow imminent | Normal during model loading |
| Shim not in `/usr/local/lib/` | Not installed | `sudo cp libgreenboost_cuda.so /usr/local/lib/` then `sudo ldconfig` |
| `KV in T3: N MB` but T3 NVMe 0% | Stale kernel counter (not reset on exit) | Safe to ignore — Python guard suppresses; see KV T3 Stale Counter section |
| Dashboard shows T2=0/0 GB, T3=0/0 GB | ioctl `max_pool_mb`=0 (no explicit cap set) | Fixed: `_fill_from_sysfs_status()` reads pool totals from sysfs |

---

## Building and Installing the Shim

```bash
cd ~/Dev/greenboost_all/greenboost

# Build
make shim

# Install
sudo cp libgreenboost_cuda.so /usr/local/lib/libgreenboost_cuda.so
sudo ldconfig

# Verify
ldconfig -p | grep greenboost
```

The `Makefile` `shim` target builds with NVTX support (`-DGREENBOOST_NVTX`) when the NVTX headers
are available (auto-detected).

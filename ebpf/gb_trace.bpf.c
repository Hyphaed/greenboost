// SPDX-License-Identifier: GPL-2.0
/*
 * ebpf/gb_trace.bpf.c , GreenBoost eBPF observability layer
 *
 * Attaches kprobes to greenboost.ko memory-tier migration functions and
 * emits structured events to a BPF_MAP_TYPE_RINGBUF.  The companion
 * userspace program (gb_trace.c) drains the ring, computes per-second
 * rates, and writes them to /run/greenboost/ebpf_stats (key=value).
 *
 * Kprobe targets (all in greenboost.c):
 *   gb_t3_evict_buf   (line 431) , T2 DDR → T3 NVMe eviction
 *   gb_t3_promote_buf (line 573) , T3 NVMe → T2 DDR promotion
 *   gb_auto_evict_cold(line 2784), cold-sweep eviction driver
 *   gb_alloc_buf      (line 1232), T2 DDR page-pool allocation
 *   gb_pin_user_buf   (line 953) , DMA-BUF / pinned-path registration
 *
 * struct gb_buf fields are accessed with bpf_probe_read_kernel at known
 * offsets (GB_BUF_*_OFFSET in gb_offsets.h) because greenboost.ko does
 * not expose BTF to CO-RE.
 *
 * UVM telemetry (optional): attached only when the symbol
 * uvm_perf_event_notify is present in /proc/kallsyms; struct layout from
 * reference/uvm_perf_events_ref.h (vendored from bpf_uvm, MIT licence).
 */

/* vmlinux.h is generated at build time:
 *   bpftool btf dump file /sys/kernel/btf/vmlinux format c > ebpf/vmlinux.h
 * It provides __u64, __u32 and other basic types without pulling in full
 * kernel headers. */
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

#include "gb_offsets.h"

/* ──────────────────────────────────────────────────────────────────── */
/*  Event kinds                                                         */
/* ──────────────────────────────────────────────────────────────────── */

#define GBE_T3_EVICT    0  /* T2 → T3 eviction (gb_t3_evict_buf ret)   */
#define GBE_T3_PROMOTE  1  /* T3 → T2 promotion (gb_t3_promote_buf ret) */
#define GBE_COLD_SWEEP  2  /* cold-evict sweep completed                */
#define GBE_ALLOC       3  /* T2 DDR page-pool alloc                    */
#define GBE_PIN         4  /* DMA-BUF / pinned alloc                    */
#define GBE_UVM_FAULT   5  /* NVIDIA UVM GPU page fault (optional)      */
#define GBE_UVM_MIGRATE 6  /* NVIDIA UVM page migration (optional)      */

/* ──────────────────────────────────────────────────────────────────── */
/*  Shared event struct                                                  */
/* ──────────────────────────────────────────────────────────────────── */

struct gb_event {
    __u64 ts_ns;      /* bpf_ktime_get_ns()                            */
    __u32 pid;        /* tgid of calling process                        */
    __u32 kind;       /* GBE_* constant                                 */
    __u64 bytes;      /* bytes involved (alloc size or migration size)  */
    __u32 flags;      /* alloc_flags from gb_buf (evict/promote), or 0  */
    __s32 ret;        /* return value (0 = success, -errno on failure)  */
};

/* ──────────────────────────────────────────────────────────────────── */
/*  BPF maps                                                            */
/* ──────────────────────────────────────────────────────────────────── */

/* Main event ring , 4 MB; at ~100 events/s that's >40 s of history    */
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 4 * 1024 * 1024);
} events SEC(".maps");

/*
 * Entry-state map for kprobe→kretprobe handoff.
 * Key: 64-bit (tgid << 32 | pid).  Value: buf size read at entry.
 * Sized at 4096 concurrent kernel threads , more than enough for a
 * GreenBoost workload.  Entries are deleted on return.
 */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key,   __u64);
    __type(value, __u64);
} entry_state SEC(".maps");

/* ──────────────────────────────────────────────────────────────────── */
/*  Helper: emit an event to the ringbuf                                */
/* ──────────────────────────────────────────────────────────────────── */

static __always_inline void
emit(struct pt_regs *ctx, __u32 kind, __u64 bytes, __u32 flags, __s32 ret)
{
    struct gb_event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e)
        return;
    e->ts_ns = bpf_ktime_get_ns();
    e->pid   = bpf_get_current_pid_tgid() >> 32;
    e->kind  = kind;
    e->bytes = bytes;
    e->flags = flags;
    e->ret   = ret;
    bpf_ringbuf_submit(e, 0);
}

/* ──────────────────────────────────────────────────────────────────── */
/*  gb_t3_evict_buf(struct gb_buf *buf)                                  */
/*  Entry: save buf->size in entry_state; Exit: emit on success         */
/* ──────────────────────────────────────────────────────────────────── */

SEC("kprobe/gb_t3_evict_buf")
int BPF_KPROBE(kp_t3_evict_entry, void *buf)
{
    __u64 key  = bpf_get_current_pid_tgid();
    __u64 size = 0;

    /* Read struct gb_buf.size at known offset (see gb_offsets.h) */
    bpf_probe_read_kernel(&size, sizeof(size),
                          (char *)buf + GB_BUF_SIZE_OFFSET);
    bpf_map_update_elem(&entry_state, &key, &size, BPF_ANY);
    return 0;
}

SEC("kretprobe/gb_t3_evict_buf")
int BPF_KRETPROBE(kp_t3_evict_exit, int ret)
{
    __u64 key   = bpf_get_current_pid_tgid();
    __u64 *sz_p = bpf_map_lookup_elem(&entry_state, &key);
    __u64  size = sz_p ? *sz_p : 0;

    bpf_map_delete_elem(&entry_state, &key);
    /* Emit on both success and failure so userspace can track error rate */
    emit(ctx, GBE_T3_EVICT, size, 0, ret);
    return 0;
}

/* ──────────────────────────────────────────────────────────────────── */
/*  gb_t3_promote_buf(struct gb_buf *buf)                               */
/* ──────────────────────────────────────────────────────────────────── */

SEC("kprobe/gb_t3_promote_buf")
int BPF_KPROBE(kp_t3_promote_entry, void *buf)
{
    __u64 key  = bpf_get_current_pid_tgid();
    __u64 size = 0;

    bpf_probe_read_kernel(&size, sizeof(size),
                          (char *)buf + GB_BUF_SIZE_OFFSET);
    bpf_map_update_elem(&entry_state, &key, &size, BPF_ANY);
    return 0;
}

SEC("kretprobe/gb_t3_promote_buf")
int BPF_KRETPROBE(kp_t3_promote_exit, int ret)
{
    __u64 key   = bpf_get_current_pid_tgid();
    __u64 *sz_p = bpf_map_lookup_elem(&entry_state, &key);
    __u64  size = sz_p ? *sz_p : 0;

    bpf_map_delete_elem(&entry_state, &key);
    emit(ctx, GBE_T3_PROMOTE, size, 0, ret);
    return 0;
}

/* ──────────────────────────────────────────────────────────────────── */
/*  gb_auto_evict_cold(u64 target_free_bytes)                           */
/*  Emit on return: ret = number of buffers evicted                     */
/* ──────────────────────────────────────────────────────────────────── */

SEC("kretprobe/gb_auto_evict_cold")
int BPF_KRETPROBE(kp_cold_sweep_exit, int ret)
{
    /* ret > 0: buffers evicted; ret == 0: nothing to evict             */
    if (ret > 0)
        emit(ctx, GBE_COLD_SWEEP, (__u64)ret, 0, ret);
    return 0;
}

/* ──────────────────────────────────────────────────────────────────── */
/*  gb_alloc_buf(size_t size, u32 flags)                                */
/*  Emit on entry (before it can fail , we want alloc-attempt rate)     */
/* ──────────────────────────────────────────────────────────────────── */

SEC("kprobe/gb_alloc_buf")
int BPF_KPROBE(kp_alloc_buf, __u64 size, __u32 flags)
{
    emit(ctx, GBE_ALLOC, size, flags, 0);
    return 0;
}

/* ──────────────────────────────────────────────────────────────────── */
/*  gb_pin_user_buf(u64 vaddr, size_t size, u32 flags)                  */
/* ──────────────────────────────────────────────────────────────────── */

SEC("kprobe/gb_pin_user_buf")
int BPF_KPROBE(kp_pin_user, __u64 vaddr, __u64 size, __u32 flags)
{
    emit(ctx, GBE_PIN, size, flags, 0);
    return 0;
}

/* ──────────────────────────────────────────────────────────────────── */
/*  NVIDIA UVM , optional, attached only when symbol present            */
/*                                                                       */
/*  uvm_perf_event_notify fires on every UVM page migration or fault.   */
/*  We only capture cause + migration direction; byte count comes from  */
/*  /proc/driver/nvidia-uvm/.../fault_stats (read by userspace, not BPF)*/
/*                                                                       */
/*  Event kinds:                                                         */
/*    cause 0 / 1 (Replayable/Non-replayable fault) → GBE_UVM_FAULT    */
/*    cause 4     (Eviction)                         → GBE_UVM_MIGRATE  */
/*    cause 2     (Access Counter trigger)           → GBE_UVM_MIGRATE  */
/*                                                                       */
/*  The program is in its own SEC so the loader can skip it gracefully  */
/*  when the symbol does not exist.                                      */
/* ──────────────────────────────────────────────────────────────────── */

/*
 * uvm_perf_event_notify signature (from nvidia-uvm driver source):
 *   void uvm_perf_event_notify(uvm_perf_event_t event_id,
 *                               uvm_perf_event_data_t *event_data)
 *
 * We only use event_id (first arg) to distinguish fault vs migration.
 * Struct fields inside event_data require the nvidia-uvm headers, so we
 * skip detailed field reads to keep this CO-RE and header-free.
 */
SEC("kprobe/uvm_perf_event_notify")
int BPF_KPROBE(kp_uvm_event, __u32 event_id)
{
    /*
     * UVM_PERF_EVENT_FAULT    = 1   (from bpf_uvm reference)
     * UVM_PERF_EVENT_MIGRATION = 3
     * We emit GBE_UVM_FAULT for faults, GBE_UVM_MIGRATE for migrations.
     * bytes = 0 here; userspace reads procfs for page counts.
     */
    __u32 kind = (event_id == 3) ? GBE_UVM_MIGRATE : GBE_UVM_FAULT;
    emit(ctx, kind, 0, event_id, 0);
    return 0;
}

char _license[] SEC("license") = "GPL";

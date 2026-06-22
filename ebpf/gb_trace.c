// SPDX-License-Identifier: GPL-2.0
/*
 * ebpf/gb_trace.c — GreenBoost eBPF tracer userspace loader
 *
 * Loads gb_trace.bpf.o via the generated skeleton (gb_trace.skel.h),
 * attaches kprobes to greenboost.ko symbols, drains the ringbuf, and
 * writes rolling per-second rates to /run/greenboost/ebpf_stats every
 * 500 ms so the Python telemetry layer can pick them up without polling.
 *
 * Additionally reads /proc/driver/nvidia-uvm/.../fault_stats for UVM
 * page migration counters (no eBPF needed for those — procfs is enough).
 *
 * Startup sequence:
 *   1. Verify CAP_BPF / BTF availability; exit 2 (not an error) if absent.
 *   2. Load + verify the BPF object.
 *   3. Attach greenboost.ko kprobes (skip gracefully if symbol not found).
 *   4. Attach uvm_perf_event_notify kprobe (skip gracefully if absent).
 *   5. Write PID to /run/greenboost/ebpf_trace.pid.
 *   6. Poll loop: drain ring → update counters → write stats every 500 ms.
 *   7. On SIGINT / SIGTERM: remove stats and PID files, clean exit.
 *
 * Output files:
 *   /run/greenboost/ebpf_stats   — key=value rates (read by EbpfProvider)
 *   /run/greenboost/ebpf_events  — bounded ring of recent decoded events
 *   /run/greenboost/ebpf_trace.pid — PID (for lifecycle management)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>

#include "gb_trace.skel.h"

/* ──────────────────────────────────────────────────────────────────── */
/*  Constants                                                           */
/* ──────────────────────────────────────────────────────────────────── */

#define GB_RUN_DIR       "/run/greenboost"
#define GB_STATS_FILE    GB_RUN_DIR "/ebpf_stats"
#define GB_EVENTS_FILE   GB_RUN_DIR "/ebpf_events"
#define GB_PID_FILE      GB_RUN_DIR "/ebpf_trace.pid"

/* Rate window: events counted in the last WINDOW_S seconds             */
#define WINDOW_S         5
/* Rolling bucket count (one per second)                                */
#define NBUCKETS         (WINDOW_S + 1)
/* Stats write interval: 500 ms                                         */
#define POLL_MS          500
/* Max recent events kept in ebpf_events                                */
#define MAX_EVENTS       200

/* ──────────────────────────────────────────────────────────────────── */
/*  Event kind IDs (must match gb_trace.bpf.c)                         */
/* ──────────────────────────────────────────────────────────────────── */

#define GBE_T3_EVICT    0
#define GBE_T3_PROMOTE  1
#define GBE_COLD_SWEEP  2
#define GBE_ALLOC       3
#define GBE_PIN         4
#define GBE_UVM_FAULT   5
#define GBE_UVM_MIGRATE 6
#define GBE_KIND_MAX    7

static const char *kind_names[GBE_KIND_MAX] = {
    "T3_EVICT", "T3_PROMOTE", "COLD_SWEEP", "ALLOC", "PIN",
    "UVM_FAULT", "UVM_MIGRATE",
};

/* ──────────────────────────────────────────────────────────────────── */
/*  Shared event struct (must match gb_trace.bpf.c)                     */
/* ──────────────────────────────────────────────────────────────────── */

struct gb_event {
    unsigned long long ts_ns;
    unsigned int       pid;
    unsigned int       kind;
    unsigned long long bytes;
    unsigned int       flags;
    int                ret;
};

/* ──────────────────────────────────────────────────────────────────── */
/*  Rolling window accumulators                                         */
/* ──────────────────────────────────────────────────────────────────── */

typedef struct {
    unsigned long long count[NBUCKETS];
    unsigned long long bytes[NBUCKETS];
    int head;   /* current bucket (0..NBUCKETS-1) */
} RollingWindow;

static RollingWindow wins[GBE_KIND_MAX];
static int current_bucket = 0;

/* Advance to the next 1-second bucket, zeroing stale ones             */
static void window_tick(void)
{
    current_bucket = (current_bucket + 1) % NBUCKETS;
    for (int k = 0; k < GBE_KIND_MAX; k++) {
        wins[k].count[current_bucket] = 0;
        wins[k].bytes[current_bucket] = 0;
        wins[k].head = current_bucket;
    }
}

static void window_add(int kind, unsigned long long bytes)
{
    if (kind < 0 || kind >= GBE_KIND_MAX) return;
    wins[kind].count[current_bucket]++;
    wins[kind].bytes[current_bucket] += bytes;
}

/* Sum count over the last WINDOW_S completed buckets                   */
static double window_rate(int kind)
{
    if (kind < 0 || kind >= GBE_KIND_MAX) return 0.0;
    unsigned long long total = 0;
    for (int i = 0; i < WINDOW_S; i++) {
        int b = ((current_bucket - 1 - i) % NBUCKETS + NBUCKETS) % NBUCKETS;
        total += wins[kind].count[b];
    }
    return (double)total / WINDOW_S;
}

static double window_bytes_rate(int kind)
{
    if (kind < 0 || kind >= GBE_KIND_MAX) return 0.0;
    unsigned long long total = 0;
    for (int i = 0; i < WINDOW_S; i++) {
        int b = ((current_bucket - 1 - i) % NBUCKETS + NBUCKETS) % NBUCKETS;
        total += wins[kind].bytes[b];
    }
    return (double)total / WINDOW_S;
}

/* ──────────────────────────────────────────────────────────────────── */
/*  Recent event ring (for ebpf_events)                                 */
/* ──────────────────────────────────────────────────────────────────── */

typedef struct {
    unsigned long long ts_ns;
    unsigned int       kind;
    unsigned long long bytes;
    int                ret;
    unsigned int       pid;
} EventEntry;

static EventEntry event_ring[MAX_EVENTS];
static int event_ring_head = 0;
static unsigned long long event_total = 0;

static void event_ring_push(const struct gb_event *e)
{
    event_ring[event_ring_head].ts_ns = e->ts_ns;
    event_ring[event_ring_head].kind  = e->kind;
    event_ring[event_ring_head].bytes = e->bytes;
    event_ring[event_ring_head].ret   = e->ret;
    event_ring[event_ring_head].pid   = e->pid;
    event_ring_head = (event_ring_head + 1) % MAX_EVENTS;
    event_total++;
}

/* ──────────────────────────────────────────────────────────────────── */
/*  UVM procfs reader                                                   */
/* ──────────────────────────────────────────────────────────────────── */

/*
 * /proc/driver/nvidia-uvm/gpus/<uuid>/fault_stats contains counters like:
 *   num_pages_in   N (M MB)
 *   num_pages_out  N (M MB)
 * We sum across all GPU entries.
 */
typedef struct {
    unsigned long long pages_in;
    unsigned long long pages_out;
    unsigned long long faults;
} UvmStats;

static UvmStats uvm_stats;

static void read_uvm_procfs(void)
{
    const char *base = "/proc/driver/nvidia-uvm/gpus";
    DIR *d = opendir(base);
    if (!d) return;

    unsigned long long pages_in = 0, pages_out = 0, faults = 0;

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;

        char path[512];
        snprintf(path, sizeof(path), "%s/%s/fault_stats", base, ent->d_name);
        FILE *f = fopen(path, "r");
        if (!f) continue;

        char line[256];
        while (fgets(line, sizeof(line), f)) {
            unsigned long long n = 0;
            if (sscanf(line, " num_pages_in %llu", &n) == 1)
                pages_in += n;
            else if (sscanf(line, " num_pages_out %llu", &n) == 1)
                pages_out += n;
            else if (sscanf(line, " replayable_faults %llu", &n) == 1)
                faults += n;
        }
        fclose(f);
    }
    closedir(d);

    uvm_stats.pages_in  = pages_in;
    uvm_stats.pages_out = pages_out;
    uvm_stats.faults    = faults;
}

/* ──────────────────────────────────────────────────────────────────── */
/*  Stats file writer                                                   */
/* ──────────────────────────────────────────────────────────────────── */

static void write_stats(void)
{
    /* Atomic write via tmp + rename (matches shim_stats pattern) */
    char tmp[256];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%d", GB_STATS_FILE, getpid());
    FILE *f = fopen(tmp, "w");
    if (!f) return;

    fprintf(f,
        "t3_evict_rate=%.3f\n"
        "t3_promote_rate=%.3f\n"
        "cold_evict_rate=%.3f\n"
        "t3_bytes_out_s=%.0f\n"
        "t3_bytes_in_s=%.0f\n"
        "alloc_rate=%.3f\n"
        "pin_rate=%.3f\n"
        "uvm_fault_rate=%.3f\n"
        "uvm_pages_in=%llu\n"
        "uvm_pages_out=%llu\n"
        "events_total=%llu\n",
        window_rate(GBE_T3_EVICT),
        window_rate(GBE_T3_PROMOTE),
        window_rate(GBE_COLD_SWEEP),
        window_bytes_rate(GBE_T3_EVICT),
        window_bytes_rate(GBE_T3_PROMOTE),
        window_rate(GBE_ALLOC),
        window_rate(GBE_PIN),
        window_rate(GBE_UVM_FAULT),
        uvm_stats.pages_in,
        uvm_stats.pages_out,
        event_total
    );
    fclose(f);
    rename(tmp, GB_STATS_FILE);
}

static void write_events(void)
{
    char tmp[256];
    snprintf(tmp, sizeof(tmp), "%s.tmp.%d", GB_EVENTS_FILE, getpid());
    FILE *f = fopen(tmp, "w");
    if (!f) return;

    /* Print in chronological order from oldest to newest               */
    int n = (int)(event_total < MAX_EVENTS ? event_total : MAX_EVENTS);
    int start = (event_ring_head - n + MAX_EVENTS) % MAX_EVENTS;
    for (int i = 0; i < n; i++) {
        int idx = (start + i) % MAX_EVENTS;
        const EventEntry *e = &event_ring[idx];
        const char *kname = (e->kind < GBE_KIND_MAX) ?
                            kind_names[e->kind] : "UNKNOWN";
        fprintf(f, "%llu kind=%s bytes=%llu ret=%d pid=%u\n",
                e->ts_ns, kname, e->bytes, e->ret, e->pid);
    }
    fclose(f);
    rename(tmp, GB_EVENTS_FILE);
}

/* ──────────────────────────────────────────────────────────────────── */
/*  Ringbuf callback                                                    */
/* ──────────────────────────────────────────────────────────────────── */

static int on_event(void *ctx, void *data, size_t data_sz)
{
    (void)ctx;
    if (data_sz < sizeof(struct gb_event)) return 0;
    const struct gb_event *e = data;

    window_add((int)e->kind, e->bytes);
    event_ring_push(e);
    return 0;
}

/* ──────────────────────────────────────────────────────────────────── */
/*  Signal handling                                                     */
/* ──────────────────────────────────────────────────────────────────── */

static volatile int g_stop = 0;

static void on_signal(int sig) { (void)sig; g_stop = 1; }

/* ──────────────────────────────────────────────────────────────────── */
/*  Kprobe attach helper — skips on ENOENT / EINVAL (symbol absent)    */
/* ──────────────────────────────────────────────────────────────────── */

static struct bpf_link *attach_kprobe(struct bpf_program *prog,
                                       const char *sym, int retprobe)
{
    struct bpf_kprobe_opts opts = {};
    opts.sz        = sizeof(opts);
    opts.retprobe  = retprobe;

    struct bpf_link *lnk = bpf_program__attach_kprobe_opts(prog, sym, &opts);
    if (!lnk) {
        int err = -errno;
        if (err == -ENOENT || err == -EINVAL) {
            /* Symbol not in kallsyms — module not loaded or CONFIG_KALLSYMS_ALL=n */
            fprintf(stderr,
                "[gb-trace] kprobe %s%s not available (skip)\n",
                retprobe ? "ret:" : "", sym);
            return NULL;
        }
        fprintf(stderr, "[gb-trace] attach kprobe %s: %s\n", sym, strerror(-err));
        return NULL;
    }
    fprintf(stderr, "[gb-trace] attached kprobe %s%s\n",
            retprobe ? "ret:" : "", sym);
    return lnk;
}

/* ──────────────────────────────────────────────────────────────────── */
/*  PID file                                                            */
/* ──────────────────────────────────────────────────────────────────── */

static void write_pid(void)
{
    mkdir(GB_RUN_DIR, 0777);
    chmod(GB_RUN_DIR, 0777);
    FILE *f = fopen(GB_PID_FILE, "w");
    if (!f) return;
    fprintf(f, "%d\n", getpid());
    fclose(f);
}

static void cleanup(void)
{
    unlink(GB_STATS_FILE);
    unlink(GB_EVENTS_FILE);
    unlink(GB_PID_FILE);
}

/* ──────────────────────────────────────────────────────────────────── */
/*  main                                                                */
/* ──────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    (void)argc; (void)argv;

    /* Suppress libbpf verbose output unless GB_BPF_VERBOSE is set      */
    if (!getenv("GB_BPF_VERBOSE"))
        libbpf_set_print(NULL);

    /* Raise RLIMIT_MEMLOCK for BPF maps (needed on older kernels)       */
    struct rlimit rl = { RLIM_INFINITY, RLIM_INFINITY };
    setrlimit(RLIMIT_MEMLOCK, &rl);

    /* Open + load BPF skeleton                                          */
    struct gb_trace_bpf *skel = gb_trace_bpf__open();
    if (!skel) {
        fprintf(stderr, "[gb-trace] failed to open BPF skeleton\n");
        return 1;
    }

    int err = gb_trace_bpf__load(skel);
    if (err) {
        fprintf(stderr, "[gb-trace] failed to load BPF program: %d\n", err);
        gb_trace_bpf__destroy(skel);
        return 1;
    }

    /* Attach kprobes — skip missing symbols, don't fail                 */
    struct bpf_link *links[12] = {};
    int nl = 0;

    /* Core greenboost.ko probes                                         */
    links[nl++] = attach_kprobe(skel->progs.kp_t3_evict_entry,
                                 "gb_t3_evict_buf",  0);
    links[nl++] = attach_kprobe(skel->progs.kp_t3_evict_exit,
                                 "gb_t3_evict_buf",  1);
    links[nl++] = attach_kprobe(skel->progs.kp_t3_promote_entry,
                                 "gb_t3_promote_buf", 0);
    links[nl++] = attach_kprobe(skel->progs.kp_t3_promote_exit,
                                 "gb_t3_promote_buf", 1);
    links[nl++] = attach_kprobe(skel->progs.kp_cold_sweep_exit,
                                 "gb_auto_evict_cold", 1);
    links[nl++] = attach_kprobe(skel->progs.kp_alloc_buf,
                                 "gb_alloc_buf",    0);
    links[nl++] = attach_kprobe(skel->progs.kp_pin_user,
                                 "gb_pin_user_buf", 0);

    /* Optional UVM probe (nvidia_uvm.ko may not be loaded)              */
    links[nl++] = attach_kprobe(skel->progs.kp_uvm_event,
                                 "uvm_perf_event_notify", 0);

    int active_links = 0;
    for (int i = 0; i < nl; i++)
        if (links[i]) active_links++;

    if (active_links == 0) {
        fprintf(stderr,
            "[gb-trace] no kprobes attached — is greenboost.ko loaded?\n"
            "           check CONFIG_KALLSYMS_ALL=y and CAP_BPF\n");
        gb_trace_bpf__destroy(skel);
        return 2;  /* exit 2: not an error, tracer not needed */
    }

    /* Ringbuf                                                            */
    struct ring_buffer *rb = ring_buffer__new(
        bpf_map__fd(skel->maps.events), on_event, NULL, NULL);
    if (!rb) {
        fprintf(stderr, "[gb-trace] failed to create ring buffer\n");
        gb_trace_bpf__destroy(skel);
        return 1;
    }

    /* Signals                                                            */
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    /* PID file                                                           */
    write_pid();

    fprintf(stderr,
        "[gb-trace] started — %d kprobes active, writing %s\n",
        active_links, GB_STATS_FILE);

    /* Poll loop                                                          */
    struct timespec last_tick   = {};
    struct timespec last_uvm    = {};
    clock_gettime(CLOCK_MONOTONIC, &last_tick);
    last_uvm = last_tick;

    while (!g_stop) {
        /* Drain ring (blocks up to POLL_MS ms)                           */
        ring_buffer__poll(rb, POLL_MS);

        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);

        /* Advance rolling window once per second                         */
        long diff_ms = (now.tv_sec  - last_tick.tv_sec)  * 1000
                     + (now.tv_nsec - last_tick.tv_nsec) / 1000000;
        if (diff_ms >= 1000) {
            window_tick();
            last_tick = now;
        }

        /* Refresh UVM procfs every 5 s                                   */
        long uvm_diff = (now.tv_sec - last_uvm.tv_sec);
        if (uvm_diff >= 5) {
            read_uvm_procfs();
            last_uvm = now;
        }

        write_stats();
        write_events();
    }

    cleanup();
    ring_buffer__free(rb);
    for (int i = 0; i < nl; i++)
        if (links[i]) bpf_link__destroy(links[i]);
    gb_trace_bpf__destroy(skel);

    fprintf(stderr, "[gb-trace] exited cleanly\n");
    return 0;
}

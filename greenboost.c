// SPDX-License-Identifier: GPL-2.0
/* */

#include <linux/module.h>
#include <linux/version.h>   /* LINUX_VERSION_CODE, KERNEL_VERSION — kernel-compat guards */
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/mm.h>
#include <linux/highmem.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/dma-buf.h>        /* pulls in iosys-map.h */
#include <linux/dma-mapping.h>
#include <linux/scatterlist.h>
#include <linux/mutex.h>
#include <linux/atomic.h>
#include <linux/kthread.h>
#include <linux/delay.h>
#include <linux/sysinfo.h>
#include <linux/idr.h>
#include <linux/swap.h>           /* legacy path — kept for symbols only  */
#include <linux/falloc.h>         /* FALLOC_FL_PUNCH_HOLE              */
#include <linux/cpumask.h>        /* cpumask_var_t, set_cpus_allowed  */
#include <linux/topology.h>       /* num_online_cpus()                */
#include <linux/eventfd.h>        /* eventfd_ctx_fdget, eventfd_signal */
#include <linux/dmi.h>            /* dmi_get_system_info()            */
#include <linux/pid.h>            /* find_get_pid(), pid_task()       */
#include <linux/reboot.h>         /* register_reboot_notifier()       */
#include <linux/panic_notifier.h> /* panic_notifier_list              */
#include <asm/processor.h>        /* boot_cpu_data.x86_model_id       */
#include "greenboost_ioctl.h"
#include "features/nvlink_pool.h"   /* NVLink pool type declarations */

// Needed for Red Hat 5.14 and 5.16+ kernels
// See for example https://github.com/google/gasket-driver/issues/14
// MODULE_IMPORT_NS changed from string form to bare-token form in mainline 5.16
// and in RHEL 9.0 (5.14-based).  Use the appropriate form for each kernel.
#if __has_include(<linux/dma-buf.h>)
/* Bare-token form required: mainline 5.16–6.12 and RHEL 9.x (5.14-based).
 * String form reverted: mainline ≥ 6.13 and all kernels < 5.16.
 * Nested #elif avoids expanding RHEL_RELEASE_VERSION on non-RHEL kernels
 * (undefined function-like macros in #if cause a preprocessor error). */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 16, 0) && \
    LINUX_VERSION_CODE <  KERNEL_VERSION(6, 13, 0)
MODULE_IMPORT_NS(DMA_BUF);          /* bare-token form — mainline 5.16–6.12 */
#elif defined(RHEL_RELEASE_CODE) && defined(RHEL_RELEASE_VERSION)
# if RHEL_RELEASE_CODE >= RHEL_RELEASE_VERSION(9, 0)
MODULE_IMPORT_NS(DMA_BUF);          /* bare-token form — RHEL 9+            */
# else
MODULE_IMPORT_NS("DMA_BUF");        /* string form    — RHEL < 9            */
# endif
#else
MODULE_IMPORT_NS("DMA_BUF");        /* string form    — < 5.16 or ≥ 6.13   */
#endif
#endif

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Ferran Duarri");
MODULE_DESCRIPTION("GreenBoost : CUDA Memory Orchestrator for NVidia GPUs");
MODULE_VERSION("2.8.1");

/* Single version string — used in banner, status, and pool_brief.
 * Update this when bumping MODULE_VERSION above. */
#define GB_VERSION  "v2.8.1"

/* 2 MiB hugepage constants */
#define GB_HPAGE_ORDER  9u

/* Memory tier identifiers */
enum gb_tier { GB_TIER2_SDDR = 2, GB_TIER3_NVME = 3 };
#define GB_HPAGE_SIZE   (PAGE_SIZE << GB_HPAGE_ORDER)   /* 2 097 152 bytes  */
#define GB_HPAGES_PER   (1u << GB_HPAGE_ORDER)          /* 512 sub-pages    */

/* ------------------------------------------------------------------ */
/*  Names                                                               */
/* ------------------------------------------------------------------ */

#define DRIVER_NAME  "greenboost"
#define DEVICE_NAME  "greenboost"
#define CLASS_NAME   "greenboost"

/* ------------------------------------------------------------------ */
/*  Module parameters — CUDA memory pool: GPU VRAM | System DDR | NVMe swap  */
/* ------------------------------------------------------------------ */

/* Tier 1 */
static int physical_vram_gb  =  0;  /* Physical GPU VRAM in GB (0 = auto-detected by setup script) */

/* Tier 2 */
static int virtual_vram_gb   =  0;  /* System RAM pool cap in GB (0 = auto-detect: 70% if < 64 GB RAM, 80% if >= 64 GB) */
static int safety_reserve_gb =  4;  /* Always keep ≥N GB free in system RAM (safe minimum) */

/* Tier 3 — GreenBoost-managed backing file (replaces kernel swap for T3) */
static int nvme_swap_gb      = 128; /* DEPRECATED: display-only. Functional T3 cap is nvme_pool_gb / t3_max_gb */
static int nvme_pool_gb      = 120; /* Backward-compat alias for t3_max_gb; default 120 GB for 120B-class models */
static char *t3_file_path    = "/var/lib/greenboost/t3_store";
static int   t3_max_gb       =   0; /* Max backing file size in GB (0=disk-limited) */

/* CPU topology — auto-detected and passed by greenboost_setup.sh at load time.
 * -1 = not set (use all available CPUs); set by setup script from detected topology. */
static int pcores_max_cpu    = -1;  /* Last P-core logical CPU number (-1 = not set, use all CPUs) */
static int golden_cpu_min    = -1;  /* First golden-core CPU (-1 = disabled, no golden-core pinning) */
static int golden_cpu_max    = -1;  /* Last  golden-core CPU (-1 = disabled) */
static int ecores_only       =  0;  /* Pin watchdog to E-cores (0=any CPU; auto-set for hybrid CPUs) */

static int debug_mode        =  0;
static int use_hugepages     =  1;  /* 2 MB compound pages (T2 only)    */

/* KV cache T1 reservation — MB of VRAM kept free so KV cache is not
 * displaced into T2/T3 by weight allocation.  KV cache is read+written
 * on every generation step; if it lands in T2 (PCIe-limited) or T3
 * (NVMe-limited) generation speed collapses.  The shim uses this value
 * as additional headroom in gb_needs_overflow() so weights spill to T2
 * before consuming the reserved window.
 * Runtime writes: GB_IOCTL_SET_KV_RESERVE (Synapse CLI) or sysfs. */
static int kv_reserve_mb     = 2048; /* default: 2 GB */

/* Active profile name — set by greenboost_setup.sh at insmod time;
 * exposed via /sys/class/greenboost/greenboost/active_profile so Synapse
 * CLI can read the profile name without parsing /etc. */
static char active_profile_name[256] = "autodetect";

module_param(physical_vram_gb,  int, 0444);
MODULE_PARM_DESC(physical_vram_gb,
	"Tier 1: Physical GPU VRAM in GB (auto-detected and passed by greenboost_setup.sh at load)");

module_param(virtual_vram_gb,   int, 0644);
MODULE_PARM_DESC(virtual_vram_gb,
	"Tier 2: System RAM pool cap in GB. 0 = auto-detect at module load: "
	"70% of total RAM (< 64 GB) or 80% (>= 64 GB). "
	"Writable: echo N > /sys/module/greenboost/parameters/virtual_vram_gb");

module_param(safety_reserve_gb, int, 0644);
MODULE_PARM_DESC(safety_reserve_gb,
	"Tier 2: Minimum free system RAM in GB to always keep reserved (default: 4). "
	"Writable: echo N > /sys/module/greenboost/parameters/safety_reserve_gb");

module_param(nvme_swap_gb,      int, 0444);
MODULE_PARM_DESC(nvme_swap_gb,
	"DEPRECATED — display-only; has no effect on T3 allocation or capping. "
	"Use nvme_pool_gb / t3_max_gb instead.");

module_param(nvme_pool_gb,      int, 0444);
MODULE_PARM_DESC(nvme_pool_gb,
	"Tier 3: backward-compat alias for t3_max_gb (default: 120) — sized for 120B-class model support");

module_param(t3_file_path, charp, 0444);
MODULE_PARM_DESC(t3_file_path,
	"Tier 3: path to GreenBoost backing file — default: /var/lib/greenboost/t3_store. "
	"GreenBoost reads/writes cold buffers directly (no kernel swap required).");

module_param(t3_max_gb, int, 0444);
MODULE_PARM_DESC(t3_max_gb,
	"Tier 3: max backing file size in GB (0=disk-limited, falls back to nvme_pool_gb) — default: 0");

module_param(pcores_max_cpu,    int, 0444);
MODULE_PARM_DESC(pcores_max_cpu,
	"Highest P-core logical CPU number (-1=not set, use all CPUs; auto-detected for hybrid CPUs)");

module_param(golden_cpu_min,    int, 0444);
MODULE_PARM_DESC(golden_cpu_min,
	"First high-frequency golden-core CPU (-1=disabled; auto-detected for supported Intel CPUs)");

module_param(golden_cpu_max,    int, 0444);
MODULE_PARM_DESC(golden_cpu_max,
	"Last high-frequency golden-core CPU (-1=disabled; auto-detected for supported Intel CPUs)");

module_param(ecores_only,       int, 0444);
MODULE_PARM_DESC(ecores_only,
	"Pin watchdog kthread to E-cores (CPUs > pcores_max_cpu) to avoid stealing cycles "
	"from P-cores during inference (1=E-cores only, 0=any CPU; auto-set for hybrid CPUs)");

module_param(debug_mode,        int, 0644);
MODULE_PARM_DESC(debug_mode,
	"Debug verbosity: 0=off 1=on");

module_param(use_hugepages,     int, 0444);
MODULE_PARM_DESC(use_hugepages,
	"Allocate 2 MB compound pages for lower TLB/DMA overhead (default: 1)");

module_param_string(active_profile_name, active_profile_name,
		    sizeof(active_profile_name), 0444);
MODULE_PARM_DESC(active_profile_name,
	"Name of the active GreenBoost profile (set by greenboost_setup.sh at load time)");

module_param(kv_reserve_mb, int, 0644);
MODULE_PARM_DESC(kv_reserve_mb,
	"MB of T1 VRAM reserved for KV cache (default: 2048). "
	"The CUDA shim adds this to vram_headroom so weights overflow to T2 sooner, "
	"keeping VRAM free for the KV cache which runs on every generation step. "
	"Update at runtime: GB_IOCTL_SET_KV_RESERVE (Synapse CLI) or sysfs.");

static uint idle_cleanup_sec = 30;
module_param(idle_cleanup_sec, uint, 0644);
MODULE_PARM_DESC(idle_cleanup_sec,
	"Seconds between watchdog dead-PID buffer reap (0=disabled, default: 30). "
	"Frees T2/T3 pages from processes that died without closing /dev/greenboost.");

#define gb_dbg(fmt, ...) \
	do { if (debug_mode) pr_info(DRIVER_NAME ": " fmt, ##__VA_ARGS__); } while (0)

/* ------------------------------------------------------------------ */
/*  Per-buffer object                                                   */
/* ------------------------------------------------------------------ */

struct gb_buf {
	/* 4K page path */
	struct page    **pages;
	/* 2MB hugepage path */
	struct page    **hpages;
	unsigned int     nhpages;
	/* common */
	bool             hugepages;
	bool             user_pinned;    /* which path is active            */
	unsigned int     npages;      /* total in 4K units               */
	size_t           size;
	int              id;          /* IDR id (0 = not yet registered) */
	int              tier;        /* GB_TIER2_SDDR or GB_TIER3_NVME  */
	struct dma_buf  *dmabuf;
	/* LRU and lifecycle fields (v2.6) */
	struct list_head  lru_node;     /* link in gb_device.lru_list    */
	unsigned long     alloc_jiffies;/* jiffies at alloc time         */
	unsigned long     last_jiffies; /* jiffies of last madvise HOT   */
	u32               alloc_flags;  /* GB_ALLOC_* flags              */
	u8                frozen;       /* 1 = never evict from T2       */
	u8                t1_priority;  /* 1 = T1-preferred (KV-like);
					 * weight bufs evicted first      */
	u8                session_priority; /* 1 = session-protected;
					 * skipped by auto-evict until
					 * all unprotected candidates gone */
	pid_t             owner_pid;    /* PID that allocated this buffer */
	/* T3 file-backing (v2.8) — pages written to t3_store, RAM freed */
	u64               t3_file_offset; /* byte offset in t3 backing file */
	bool              t3_on_disk;     /* true = pages on disk, buf->pages == NULL */
};

/* ------------------------------------------------------------------ */
/*  Global device state                                                 */
/* ------------------------------------------------------------------ */

struct gb_device {
	dev_t            devt;
	struct cdev      cdev;
	struct class    *cls;
	struct device   *dev;

	struct mutex     lock;            /* protects idr                   */
	struct idr       idr;             /* id → struct gb_buf *           */

	/* Tier 2 — System DDR pool */
	atomic_t         active_bufs;     /* live DMA-BUF objects           */
	atomic64_t       pool_allocated;  /* bytes currently pinned (T2)    */
	atomic_t         oom_active;      /* 1 when safety guard tripped    */

	/* Tier 3 — GreenBoost-managed backing file */
	atomic64_t       nvme_allocated;  /* bytes currently evicted to T3  */
	atomic_t         swap_pressure;   /* 0=ok 1=warn 2=critical (T3)    */
	struct file     *t3_file;         /* backing store file (NULL=disabled) */
	atomic64_t       t3_next_offset;  /* bump allocator: next byte offset */
	spinlock_t       t3_file_lock;    /* protects t3_file open/close    */
	u64              t3_file_max;     /* cap in bytes (0=disk-limited)  */

	/* Tier 2 — DDR pool pressure (graduated, mirrors T3 levels) */
	atomic_t         t2_pressure;     /* 0=ok 1=warn 2=critical (T2)    */

	/* KV cache tracking — allocations with GB_ALLOC_KV_CACHE flag (ExLlamaV3 native) */
	atomic64_t       kv_used_bytes;   /* total bytes tagged as KV cache (any tier) */
	atomic64_t       kv_t2_bytes;     /* KV bytes specifically in T2 DDR pool */

	/* Phase reset sequencing — incremented by GB_IOCTL_RESET_PHASE so the shim
	 * can detect model swaps across process boundaries (Synapse CLI → kernel →
	 * shim polls on next GB_KV_REFRESH_INTERVAL boundary). */
	atomic_t         phase_reset_seq;

	/* TurboQuant KV cache compression config (set by Synapse CLI via GB_IOCTL_SET_TURBOQUANT) */
	u32              turboquant_enabled;
	u32              turboquant_bits;
	u32              turboquant_head_dim;
	u32              turboquant_seed;

	struct task_struct *watchdog;
	/* LRU tracking (v2.6) */
	struct list_head    lru_list;     /* T2 buffers in LRU order       */
	spinlock_t          lru_lock;     /* protects lru_list             */
	spinlock_t          efd_lock;     /* protects pressure_efd         */
	struct eventfd_ctx *pressure_efd; /* signaled on pressure change   */
	atomic_t            teardown_done; /* prevents double-teardown: reboot notifier + gb_exit */
	/* Async T3 eviction workqueue — prevents ioctl threads from blocking on NVMe I/O */
	struct workqueue_struct *evict_wq;
};

static struct gb_device gb_dev;

/* gb_get_t3_stats — live T3 backing-file usage in MB.
 * used_mb: bytes currently evicted to disk.
 * max_mb:  file cap (0 = disk-limited / unlimited).
 */
static void gb_get_t3_stats(u64 *used_mb, u64 *max_mb)
{
	*used_mb = (u64)atomic64_read(&gb_dev.nvme_allocated) >> 20;
	*max_mb  = gb_dev.t3_file_max >> 20; /* 0 if unlimited */
}

/* ------------------------------------------------------------------ */
/*  T3 backing-file helpers (v2.8)                                      */
/* ------------------------------------------------------------------ */

/* gb_t3_file_open — create/open the T3 backing store.
 * Called from gb_init() when T3 is enabled.
 * Non-fatal: module loads even if file cannot be opened (T3 disabled).
 */
static int gb_t3_file_open(void)
{
	struct file *f;

	f = filp_open(t3_file_path, O_RDWR | O_CREAT | O_LARGEFILE, 0600);
	if (IS_ERR(f)) {
		pr_warn(DRIVER_NAME
			": T3 backing file unavailable (%s): %ld — T3 disabled\n",
			t3_file_path, PTR_ERR(f));
		return PTR_ERR(f);
	}
	spin_lock(&gb_dev.t3_file_lock);
	gb_dev.t3_file = f;
	spin_unlock(&gb_dev.t3_file_lock);
	pr_info(DRIVER_NAME ": T3 backing file: %s (%s)\n",
		t3_file_path,
		gb_dev.t3_file_max > 0 ? "capped" : "disk-limited");
	return 0;
}

static void gb_t3_file_close(void)
{
	struct file *f;
	unsigned long flags;

	spin_lock_irqsave(&gb_dev.t3_file_lock, flags);
	f = gb_dev.t3_file;
	gb_dev.t3_file = NULL;
	spin_unlock_irqrestore(&gb_dev.t3_file_lock, flags);

	if (f) {
		filp_close(f, NULL);
		pr_info(DRIVER_NAME ": T3 backing file closed\n");
	}
}

/* gb_t3_evict_buf — write a T2 buffer to the T3 backing file and free its RAM.
 *
 * Caller must have already updated tier accounting (T2→T3) and removed the
 * buffer from the LRU list.  This function does the physical work: writes
 * pages to disk then frees the RAM.  Can sleep (file I/O).
 *
 * On success: buf->t3_on_disk = true, buf->pages = NULL.
 * On failure: returns -errno; caller should roll back accounting.
 */
static int gb_t3_evict_buf(struct gb_buf *buf)
{
	u64 offset;
	unsigned int i;
	loff_t pos;
	ssize_t written;

	if (!gb_dev.t3_file)
		return -ENODEV;
	/* Only 4K-page buffers can be evicted; hugepages are pinned (DMA) */
	if (buf->hugepages)
		return -EINVAL;

	/* Cap check (0 = unlimited) */
	if (gb_dev.t3_file_max > 0) {
		u64 next = (u64)atomic64_read(&gb_dev.t3_next_offset) + buf->size;
		if (next > gb_dev.t3_file_max) {
			pr_warn(DRIVER_NAME
				": T3 file cap reached (%lluGB)\n",
				gb_dev.t3_file_max >> 30);
			return -ENOSPC;
		}
	}

	/* Allocate a contiguous slot via bump pointer */
	offset = (u64)atomic64_fetch_add((s64)buf->size, &gb_dev.t3_next_offset);

	/* Write each page to the backing file */
	for (i = 0; i < buf->npages; i++) {
		void *kaddr;

		pos = (loff_t)(offset + (u64)i * PAGE_SIZE);
		kaddr = kmap_local_page(buf->pages[i]);
		written = kernel_write(gb_dev.t3_file, kaddr, PAGE_SIZE, &pos);
		kunmap_local(kaddr);

		if (written != PAGE_SIZE) {
			pr_err(DRIVER_NAME
				": T3 write failed at +%lluMB: %zd\n",
				offset >> 20, written);
			/* Punch a hole to reclaim the partially written region */
			vfs_fallocate(gb_dev.t3_file,
				      FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
				      offset, (loff_t)((u64)i * PAGE_SIZE));
			return written < 0 ? (int)written : -EIO;
		}
	}

	/* Free the RAM pages — they are now safely on disk */
	for (i = 0; i < buf->npages; i++)
		__free_page(buf->pages[i]);
	kvfree(buf->pages);

	buf->pages          = NULL;
	buf->t3_file_offset = offset;
	buf->t3_on_disk     = true;

	gb_dbg("T3 evict: buf id=%d %zuMB → file @%lluMB\n",
	       buf->id, buf->size >> 20, offset >> 20);
	return 0;
}

/* gb_t3_promote_buf — read a T3 on-disk buffer back into RAM.
 *
 * Called automatically when CUDA maps (imports) a T3 buffer via DMA-BUF.
 * Updates tier accounting (T3→T2) on success.  Can sleep (file I/O).
 *
 * On success: buf->t3_on_disk = false, buf->pages filled, tier = T2.
 * On failure: returns -errno; buf stays on disk.
 */
static int gb_t3_promote_buf(struct gb_buf *buf)
{
	unsigned int i, j;
	loff_t pos;
	ssize_t bytes_read;
	int ret = 0;

	if (!buf->t3_on_disk)
		return 0;
	if (!gb_dev.t3_file)
		return -ENODEV;

	/* Allocate fresh RAM pages */
	buf->pages = kvcalloc(buf->npages, sizeof(struct page *), GFP_KERNEL);
	if (!buf->pages)
		return -ENOMEM;

	for (i = 0; i < buf->npages; i++) {
		/* __GFP_NORETRY: fail fast rather than looping in direct reclaim
		 * (a T3→T2 promotion that stalls here would block the CUDA map
		 *  callback indefinitely, hanging inference).
		 * __GFP_NOWARN: suppress the kernel splat; -ENOMEM is handled. */
		buf->pages[i] = alloc_page(GFP_KERNEL | __GFP_ZERO |
					   __GFP_NORETRY | __GFP_NOWARN);
		if (!buf->pages[i]) {
			for (j = 0; j < i; j++)
				__free_page(buf->pages[j]);
			kvfree(buf->pages);
			buf->pages = NULL;
			return -ENOMEM;
		}
	}

	/* Read page data back from file */
	for (i = 0; i < buf->npages; i++) {
		void *kaddr;

		pos = (loff_t)(buf->t3_file_offset + (u64)i * PAGE_SIZE);
		kaddr = kmap_local_page(buf->pages[i]);
		bytes_read = kernel_read(gb_dev.t3_file, kaddr, PAGE_SIZE, &pos);
		kunmap_local(kaddr);

		if (bytes_read != PAGE_SIZE) {
			pr_err(DRIVER_NAME
				": T3 read failed at +%lluMB: %zd\n",
				buf->t3_file_offset >> 20, bytes_read);
			for (j = 0; j < buf->npages; j++)
				__free_page(buf->pages[j]);
			kvfree(buf->pages);
			buf->pages = NULL;
			ret = bytes_read < 0 ? (int)bytes_read : -EIO;
			return ret;
		}
	}

	/* Punch a hole to reclaim disk space (sparse file — best effort) */
	if (vfs_fallocate(gb_dev.t3_file,
			  FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
			  buf->t3_file_offset, (loff_t)buf->size) != 0)
		gb_dbg("T3 hole punch unsupported on this filesystem\n");

	/* Update tier accounting T3 → T2 */
	atomic64_sub(buf->size, &gb_dev.nvme_allocated);
	atomic64_add(buf->size, &gb_dev.pool_allocated);
	buf->tier       = GB_TIER2_SDDR;
	buf->t3_on_disk = false;

	/* Re-add to LRU tail (now a warm T2 buffer) */
	spin_lock(&gb_dev.lru_lock);
	list_add_tail(&buf->lru_node, &gb_dev.lru_list);
	spin_unlock(&gb_dev.lru_lock);

	gb_dbg("T3 promote: buf id=%d %zuMB ← file @%lluMB\n",
	       buf->id, buf->size >> 20, buf->t3_file_offset);
	return 0;
}

/* ------------------------------------------------------------------ */
/*  DMA-BUF operations                                                  */
/* ------------------------------------------------------------------ */

static struct sg_table *gb_map_dma_buf(struct dma_buf_attachment *attach,
					enum dma_data_direction dir)
{
	struct gb_buf *buf = attach->dmabuf->priv;
	struct sg_table *sgt;
	int ret;
	unsigned int i;

	/* Auto-promote T3 on-disk buffer back to RAM before mapping */
	if (buf->t3_on_disk) {
		ret = gb_t3_promote_buf(buf);
		if (ret) {
			pr_err(DRIVER_NAME
				": T3 promote failed during DMA-BUF map (id=%d): %d\n",
				buf->id, ret);
			return ERR_PTR(ret);
		}
	}

	sgt = kzalloc(sizeof(*sgt), GFP_KERNEL);
	if (!sgt)
		return ERR_PTR(-ENOMEM);

	if (buf->hugepages) {
		/* Compact sg_table: one entry per 2 MB hugepage */
		ret = sg_alloc_table(sgt, buf->nhpages, GFP_KERNEL);
		if (ret) { kfree(sgt); return ERR_PTR(ret); }
		for (i = 0; i < buf->nhpages; i++)
			sg_set_page(&sgt->sgl[i], buf->hpages[i], GB_HPAGE_SIZE, 0);
	} else {
		ret = sg_alloc_table_from_pages(sgt, buf->pages, buf->npages,
						0, buf->size, GFP_KERNEL);
		if (ret) { kfree(sgt); return ERR_PTR(ret); }
	}

	ret = dma_map_sgtable(attach->dev, sgt, dir, 0);
	if (ret) {
		sg_free_table(sgt);
		kfree(sgt);
		return ERR_PTR(ret);
	}

	gb_dbg("mapped %zuMB (%s) for %s\n", buf->size >> 20,
	       buf->hugepages ? "2MB pages" : "4K pages",
	       dev_name(attach->dev));
	return sgt;
}

static void gb_unmap_dma_buf(struct dma_buf_attachment *attach,
			      struct sg_table *sgt,
			      enum dma_data_direction dir)
{
	dma_unmap_sgtable(attach->dev, sgt, dir, 0);
	sg_free_table(sgt);
	kfree(sgt);
}

static void gb_release(struct dma_buf *dmabuf)
{
	struct gb_buf *buf = dmabuf->priv;
	unsigned int i;

	gb_dbg("release buffer id=%d size=%zuMB (%s)\n",
	       buf->id, buf->size >> 20,
	       buf->hugepages ? "2MB pages" : "4K pages");

	/* Remove from LRU list */
	spin_lock(&gb_dev.lru_lock);
	list_del_init(&buf->lru_node);
	spin_unlock(&gb_dev.lru_lock);

	/* Remove from IDR if registered */
	if (buf->id > 0) {
		mutex_lock(&gb_dev.lock);
		idr_remove(&gb_dev.idr, buf->id);
		mutex_unlock(&gb_dev.lock);
	}

	atomic_dec(&gb_dev.active_bufs);
	if (buf->tier == GB_TIER3_NVME)
		atomic64_sub(buf->size, &gb_dev.nvme_allocated);
	else
		atomic64_sub(buf->size, &gb_dev.pool_allocated);
	/* KV cache usage accounting */
	if (buf->alloc_flags & GB_ALLOC_KV_CACHE) {
		atomic64_sub(buf->size, &gb_dev.kv_used_bytes);
		if (buf->tier == GB_TIER2_SDDR) {
			s64 cur = atomic64_read(&gb_dev.kv_t2_bytes);
			s64 sub = (s64)buf->size;
			atomic64_sub(sub < cur ? sub : cur, &gb_dev.kv_t2_bytes);
		}
	}

	if (buf->t3_on_disk) {
		/* Pages were already freed during eviction.
		 * Punch a hole to free disk space (best effort — sparse file). */
		if (gb_dev.t3_file)
			vfs_fallocate(gb_dev.t3_file,
				      FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
				      buf->t3_file_offset, (loff_t)buf->size);
	} else if (buf->user_pinned) {
		for (i = 0; i < buf->npages; i++)
			unpin_user_page(buf->pages[i]);
		kvfree(buf->pages);
	} else if (buf->hugepages) {
		for (i = 0; i < buf->nhpages; i++)
			__free_pages(buf->hpages[i], GB_HPAGE_ORDER);
		kvfree(buf->hpages);
	} else {
		for (i = 0; i < buf->npages; i++)
			__free_page(buf->pages[i]);
		kvfree(buf->pages);
	}
	kfree(buf);
}

static int gb_mmap(struct dma_buf *dmabuf, struct vm_area_struct *vma)
{
	struct gb_buf *buf = dmabuf->priv;
	unsigned long addr = vma->vm_start;
	unsigned int i, j;
	int ret;

	/* Auto-promote T3 on-disk buffer before mmap */
	if (buf->t3_on_disk) {
		ret = gb_t3_promote_buf(buf);
		if (ret)
			return ret;
	}

	if ((vma->vm_end - vma->vm_start) > buf->size)
		return -EINVAL;

	vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);

	if (buf->hugepages) {
		for (i = 0; i < buf->nhpages && addr < vma->vm_end; i++) {
			for (j = 0; j < GB_HPAGES_PER && addr < vma->vm_end;
			     j++, addr += PAGE_SIZE) {
				ret = vm_insert_page(vma, addr, buf->hpages[i] + j);
				if (ret)
					return ret;
			}
		}
	} else {
		for (i = 0; i < buf->npages && addr < vma->vm_end;
		     i++, addr += PAGE_SIZE) {
			ret = vm_insert_page(vma, addr, buf->pages[i]);
			if (ret)
				return ret;
		}
	}
	return 0;
}

static int gb_vmap_op(struct dma_buf *dmabuf, struct iosys_map *map)
{
	struct gb_buf *buf = dmabuf->priv;
	void *vaddr;
	int ret;

	/* Auto-promote T3 on-disk buffer before vmap */
	if (buf->t3_on_disk) {
		ret = gb_t3_promote_buf(buf);
		if (ret)
			return ret;
	}

	if (buf->hugepages) {
		/* Expand compound pages to flat 4K array for vmap */
		struct page **tmp;
		unsigned int i, j, k = 0;

		tmp = kvmalloc_array(buf->npages, sizeof(*tmp), GFP_KERNEL);
		if (!tmp)
			return -ENOMEM;
		for (i = 0; i < buf->nhpages; i++)
			for (j = 0; j < GB_HPAGES_PER; j++)
				tmp[k++] = buf->hpages[i] + j;
		vaddr = vmap(tmp, buf->npages, VM_MAP, PAGE_KERNEL);
		kvfree(tmp);
	} else {
		vaddr = vmap(buf->pages, buf->npages, VM_MAP, PAGE_KERNEL);
	}

	if (!vaddr)
		return -ENOMEM;

	iosys_map_set_vaddr(map, vaddr);
	return 0;
}

static void gb_vunmap_op(struct dma_buf *dmabuf, struct iosys_map *map)
{
	vunmap(map->vaddr);
	iosys_map_clear(map);
}

static const struct dma_buf_ops gb_dma_buf_ops = {
	.map_dma_buf   = gb_map_dma_buf,
	.unmap_dma_buf = gb_unmap_dma_buf,
	.release       = gb_release,
	.mmap          = gb_mmap,
	.vmap          = gb_vmap_op,
	.vunmap        = gb_vunmap_op,
};

/* ------------------------------------------------------------------ */
/*  Page pinning from userspace (FOLL_LONGTERM)                       */
/* ------------------------------------------------------------------ */

static struct gb_buf *gb_pin_user_buf(u64 vaddr, size_t size, u32 flags)
{
	struct gb_buf *buf;
	unsigned int np = DIV_ROUND_UP(size, PAGE_SIZE);
	size_t aligned_size = (size_t)np * PAGE_SIZE;
	int ret;
	unsigned int i;
	u64 t2_max = (u64)virtual_vram_gb * (1ULL << 30);
	u64 t2_used = (u64)atomic64_read(&gb_dev.pool_allocated);

	/* CRIT-03: Rewritten to avoid u64 addition overflow.
	 * Original: t2_used + aligned_size > t2_max (wraps when both are large) */
	if (aligned_size > t2_max || t2_max - t2_used < aligned_size) {
		pr_warn(DRIVER_NAME ": T2 cap reached for pinned memory\n");
		return ERR_PTR(-ENOMEM);
	}

	/* MED-04: Also guard against free-RAM exhaustion — gb_alloc_buf does this
	 * but gb_pin_user_buf previously skipped the check, allowing a caller to
	 * pin more RAM than available and trigger a system OOM. */
	{
		u64 free_bytes    = (u64)si_mem_available() * PAGE_SIZE;
		u64 reserve_bytes = (u64)safety_reserve_gb * (1ULL << 30);

		if (aligned_size > free_bytes || free_bytes - aligned_size < reserve_bytes) {
			pr_warn(DRIVER_NAME
				": pin OOM guard: avail=%lluMB < reserve=%dGB + req=%zuMB\n",
				free_bytes >> 20, safety_reserve_gb, aligned_size >> 20);
			return ERR_PTR(-ENOMEM);
		}
	}

	buf = kzalloc(sizeof(*buf), GFP_KERNEL);
	if (!buf)
		return ERR_PTR(-ENOMEM);

	buf->pages = kvcalloc(np, sizeof(struct page *), GFP_KERNEL);
	if (!buf->pages) {
		kfree(buf);
		return ERR_PTR(-ENOMEM);
	}

	/* Pin the user pages with FOLL_LONGTERM so they can be safely used for DMA */
	mmap_read_lock(current->mm);
	ret = pin_user_pages(vaddr, np, FOLL_WRITE | FOLL_LONGTERM, buf->pages);
	mmap_read_unlock(current->mm);

	if (ret < 0 || ret != np) {
		if (ret > 0) {
			for (i = 0; i < ret; i++)
				unpin_user_page(buf->pages[i]);
		}
		kvfree(buf->pages);
		kfree(buf);
		pr_err(DRIVER_NAME ": pin_user_pages failed: %d\n", ret);
		return ERR_PTR(ret < 0 ? ret : -ENOMEM);
	}

	buf->hugepages = false;
	buf->user_pinned = true;
	buf->npages = np;
	buf->size = aligned_size;

	buf->id = 0;
	buf->tier = GB_TIER2_SDDR;
	buf->alloc_flags = flags;
	buf->alloc_jiffies = jiffies;
	buf->last_jiffies = jiffies;
	buf->frozen = (flags & GB_ALLOC_FROZEN) ? 1 : 0;
	buf->owner_pid = task_pid_vnr(current);
	INIT_LIST_HEAD(&buf->lru_node);

	atomic64_add(buf->size, &gb_dev.pool_allocated);
	/* KV cache usage accounting for user-pinned buffers */
	if (flags & GB_ALLOC_KV_CACHE)
		atomic64_add(buf->size, &gb_dev.kv_used_bytes);
	atomic_inc(&gb_dev.active_bufs);

	spin_lock(&gb_dev.lru_lock);
	list_add_tail(&buf->lru_node, &gb_dev.lru_list);
	spin_unlock(&gb_dev.lru_lock);

	gb_dbg("pinned %u user pages (%zuMB)\n", np, aligned_size >> 20);
	return buf;
}

/* ------------------------------------------------------------------ */
/*  Pressure-triggered proactive eviction                               */
/* ------------------------------------------------------------------ */

/*
 * gb_evict_work — work item for async T3 eviction.
 *
 * gb_try_evict_for_alloc() stages buffers for eviction via Phase 1 (accounting
 * update under lru_lock) and then queues one gb_evict_work per buffer to
 * gb_dev.evict_wq.  The actual file I/O runs asynchronously in the workqueue
 * so the ioctl caller (CUDA/ollama thread) is never blocked on NVMe writes.
 */
struct gb_evict_work {
	struct work_struct  work;
	struct gb_buf      *buf;
};

static void gb_evict_work_fn(struct work_struct *w)
{
	struct gb_evict_work *ew = container_of(w, struct gb_evict_work, work);
	struct gb_buf *buf = ew->buf;

	kfree(ew);

	if (gb_t3_evict_buf(buf) != 0) {
		/* I/O failed — roll accounting back to T2 */
		atomic64_sub(buf->size, &gb_dev.nvme_allocated);
		atomic64_add(buf->size, &gb_dev.pool_allocated);
		buf->tier = GB_TIER2_SDDR;
		spin_lock(&gb_dev.lru_lock);
		list_add_tail(&buf->lru_node, &gb_dev.lru_list);
		spin_unlock(&gb_dev.lru_lock);
	}
}

/*
 * gb_try_evict_for_alloc - free T2 space before falling to T3.
 *
 * Called from gb_alloc_buf() when T2 is full and the allocation is not KV
 * cache.  Walks the LRU tail and evicts cold 4K-page T2 buffers until at
 * least @need_bytes are freed or no eligible candidates remain.
 *
 * Two-phase:
 *   Phase 1 (under lru_lock): select candidates, adjust accounting T2→T3.
 *   Phase 2 (async):          queue gb_evict_work items to evict_wq.
 *                             File I/O runs in the workqueue — the ioctl
 *                             thread is NOT blocked on NVMe writes.
 *
 * Returns the number of bytes staged for eviction (accounting already
 * updated in Phase 1), or 0 if nothing was staged.
 * Must be called without holding gb_dev.lock (IDR mutex).
 */
static u64 gb_try_evict_for_alloc(u64 need_bytes)
{
	struct gb_buf *buf, *tmp;
	LIST_HEAD(evict_list);
	u64 freed = 0;
	int staged = 0;

	if (!gb_dev.t3_file)
		return 0;

	/* Phase 1: select candidates under lru_lock */
	spin_lock(&gb_dev.lru_lock);
	list_for_each_entry_safe_reverse(buf, tmp, &gb_dev.lru_list, lru_node) {
		if (freed >= need_bytes)
			break;
		if (buf->alloc_flags & GB_ALLOC_KV_CACHE)
			continue;
		if (buf->frozen || buf->t1_priority)
			continue;
		if (buf->session_priority >= 1)
			continue;
		if (buf->tier != GB_TIER2_SDDR || buf->hugepages)
			continue;

		atomic64_sub(buf->size, &gb_dev.pool_allocated);
		atomic64_add(buf->size, &gb_dev.nvme_allocated);
		buf->tier = GB_TIER3_NVME;
		list_del_init(&buf->lru_node);
		list_add_tail(&buf->lru_node, &evict_list);
		freed += buf->size;
		staged++;
	}

	/* Second pass: if still not enough, consider session-protected buffers */
	if (freed < need_bytes) {
		list_for_each_entry_safe_reverse(buf, tmp, &gb_dev.lru_list, lru_node) {
			if (freed >= need_bytes)
				break;
			if (buf->alloc_flags & GB_ALLOC_KV_CACHE)
				continue;
			if (buf->frozen || buf->t1_priority)
				continue;
			if (buf->tier != GB_TIER2_SDDR || buf->hugepages)
				continue;

			atomic64_sub(buf->size, &gb_dev.pool_allocated);
			atomic64_add(buf->size, &gb_dev.nvme_allocated);
			buf->tier = GB_TIER3_NVME;
			list_del_init(&buf->lru_node);
			list_add_tail(&buf->lru_node, &evict_list);
			freed += buf->size;
			staged++;
		}
	}
	spin_unlock(&gb_dev.lru_lock);

	if (staged == 0)
		return 0;

	/* Phase 2: queue async work items — do NOT block the ioctl thread on I/O.
	 * Phase 1 already updated the atomic accounting (T2→T3), so the allocation
	 * path sees the freed T2 headroom immediately.  The actual NVMe writes run
	 * in evict_wq asynchronously.  If kmalloc fails for a work item, roll that
	 * buffer's accounting back to T2 so the system stays consistent. */
	list_for_each_entry_safe(buf, tmp, &evict_list, lru_node) {
		struct gb_evict_work *ew;

		list_del_init(&buf->lru_node);
		ew = kmalloc(sizeof(*ew), GFP_KERNEL);
		if (!ew) {
			/* Out of memory for work item — roll back this buffer */
			atomic64_sub(buf->size, &gb_dev.nvme_allocated);
			atomic64_add(buf->size, &gb_dev.pool_allocated);
			buf->tier = GB_TIER2_SDDR;
			freed -= buf->size;
			spin_lock(&gb_dev.lru_lock);
			list_add_tail(&buf->lru_node, &gb_dev.lru_list);
			spin_unlock(&gb_dev.lru_lock);
			continue;
		}
		INIT_WORK(&ew->work, gb_evict_work_fn);
		ew->buf = buf;
		queue_work(gb_dev.evict_wq, &ew->work);
	}

	if (freed > 0)
		gb_dbg("alloc-pressure evict: staged %lluMB to T3 (async I/O)\n", freed >> 20);
	return freed;
}

/* ------------------------------------------------------------------ */
/*  Page pool allocator                                                 */
/* ------------------------------------------------------------------ */

static struct gb_buf *gb_alloc_buf(size_t size, u32 flags)
{
	struct gb_buf *buf;
	unsigned int i, j;
	u64 t2_max    = (u64)virtual_vram_gb * (1ULL << 30);
	u64 t3_max    = (u64)nvme_pool_gb    * (1ULL << 30);
	u64 t2_used   = (u64)atomic64_read(&gb_dev.pool_allocated);
	u64 t3_used   = (u64)atomic64_read(&gb_dev.nvme_allocated);
	u64 free_bytes, reserve_bytes;
	bool tier3 = false;

	/* Safety: reject if too little free RAM.
	 * Use si_mem_available() (= MemAvailable) rather than si.freeram (= MemFree).
	 * MemAvailable accounts for reclaimable page cache and slab — the kernel will
	 * free those pages under pressure.  MemFree alone is ~21 GB lower on a loaded
	 * workstation and causes spurious OOM guard trips. */
	free_bytes    = (u64)si_mem_available() * PAGE_SIZE;
	reserve_bytes = (u64)safety_reserve_gb * (1ULL << 30);

	/* CRIT-03: Rewritten to avoid u64 addition overflow when size is large.
	 * Original: free_bytes < reserve_bytes + size  (wraps if sum > UINT64_MAX) */
	if (size > free_bytes || free_bytes - size < reserve_bytes) {
		atomic_set(&gb_dev.oom_active, 1);
		pr_warn(DRIVER_NAME
			": OOM guard: avail=%lluMB < reserve=%dGB + req=%zuMB\n",
			free_bytes >> 20, safety_reserve_gb, size >> 20);
		return ERR_PTR(-ENOMEM);
	}

	if (t2_used + size > t2_max) {
		/* KV cache (GB_ALLOC_KV_CACHE) must never reach T3 — bandwidth
		 * collapse at ~1.8 GB/s would grind generation to a halt.
		 * Return ENOSPC and let the shim's UVM fallback handle it. */
		if (flags & GB_ALLOC_KV_CACHE) {
			pr_warn(DRIVER_NAME
				": KV cache T2 full (%lluMB/%dGB) — refusing T3 spill; KV must stay in T1/T2\n",
				t2_used >> 20, virtual_vram_gb);
			return ERR_PTR(-ENOSPC);
		}
		/* Proactive eviction: try to free cold T2 pages before falling to T3.
		 * This keeps allocation bandwidth in T2 (32 GB/s DDR) rather than
		 * immediately spilling to T3 (1.8 GB/s NVMe). */
		if (gb_dev.t3_file) {
			u64 freed = gb_try_evict_for_alloc((u64)size);
			if (freed >= (u64)size) {
				/* Re-read t2_used after eviction */
				t2_used = (u64)atomic64_read(&gb_dev.pool_allocated);
				if (t2_used + size <= t2_max)
					goto proceed_t2;
			}
		}
		/* T2 full — check T3 safety-net (disabled by default: nvme_pool_gb=0) */
		if (nvme_pool_gb == 0) {
			pr_warn(DRIVER_NAME
				": T3 safety-net disabled (nvme_pool_gb=0) — T2 full (%lluMB/%dGB), returning ENOSPC\n",
				t2_used >> 20, virtual_vram_gb);
			return ERR_PTR(-ENOSPC);
		}
		if (t3_used + size > t3_max) {
			pr_warn(DRIVER_NAME
				": T3 safety-net cap reached — %dGB limit, used=%lluMB\n",
				nvme_pool_gb, t3_used >> 20);
			return ERR_PTR(-ENOSPC);
		}
		tier3 = true;
		pr_warn(DRIVER_NAME
			": T3 safety-net triggered — T2 full (%lluMB/%dGB), spilling %zuMB to NVMe (SLOW — NVMe bandwidth)\n",
			t2_used >> 20, virtual_vram_gb, size >> 20);
	}

proceed_t2:
	buf = kzalloc(sizeof(*buf), GFP_KERNEL);
	if (!buf)
		return ERR_PTR(-ENOMEM);

	/*
	 * Tier 3 (NVMe-spillable): force 4K pages, no hugepages.
	 * GFP_HIGHUSER allows the kernel to reclaim/swap these pages
	 * to NVMe under memory pressure — that IS the spill mechanism.
	 * Tier 2 (System DDR): use hugepages for lower DMA scatter overhead.
	 */
	if (tier3)
		goto alloc_4k;

	/* --- 2 MB hugepage path (Tier 2 only) --- */
	if (use_hugepages && !(flags & GB_ALLOC_NO_HUGEPAGE)) {
		unsigned int nhp = DIV_ROUND_UP(size, GB_HPAGE_SIZE);
		size_t hsize    = (size_t)nhp * GB_HPAGE_SIZE;
		/* MIN-02: hsize >= size due to DIV_ROUND_UP.  The over-alloc is at most
		 * (GB_HPAGE_SIZE - 1) = 2 MB - 1 B.  buf->size is set to hsize so the
		 * DMA-BUF sg_table covers the full compound page range — the GPU sees the
		 * rounded-up size, which is harmless; unused bytes are never touched. */

		buf->hpages = kvcalloc(nhp, sizeof(struct page *), GFP_KERNEL);
		if (!buf->hpages)
			goto fallback_4k;

		for (i = 0; i < nhp; i++) {
			buf->hpages[i] = alloc_pages(
				GFP_KERNEL | __GFP_ZERO | __GFP_COMP | __GFP_NOWARN,
				GB_HPAGE_ORDER);
			if (!buf->hpages[i]) {
				for (j = 0; j < i; j++)
					__free_pages(buf->hpages[j], GB_HPAGE_ORDER);
				kvfree(buf->hpages);
				buf->hpages = NULL;
				gb_dbg("hugepage alloc failed at %u/%u, falling back to 4K\n",
				       i, nhp);
				goto fallback_4k;
			}
		}

		buf->hugepages = true;
		buf->nhpages   = nhp;
		buf->npages    = nhp * GB_HPAGES_PER;
		buf->size      = hsize;
		gb_dbg("allocated %u hugepages (%zuMB, 2MB pages)\n", nhp, hsize >> 20);
		goto done;
	}

fallback_4k:
alloc_4k:
	{
		unsigned int np = DIV_ROUND_UP(size, PAGE_SIZE);
		size = (size_t)np * PAGE_SIZE;

		buf->pages = kvcalloc(np, sizeof(struct page *), GFP_KERNEL);
		if (!buf->pages) { kfree(buf); return ERR_PTR(-ENOMEM); }

		for (i = 0; i < np; i++) {
			/* __GFP_NORETRY: fail fast under memory pressure rather than
			 * spinning in direct reclaim — keeps the ioctl latency bounded.
			 * __GFP_NOWARN: -ENOMEM is handled; no need for kernel splat. */
			buf->pages[i] = alloc_page(GFP_HIGHUSER | __GFP_ZERO |
						   __GFP_NORETRY | __GFP_NOWARN);
			if (!buf->pages[i]) {
				for (j = 0; j < i; j++)
					__free_page(buf->pages[j]);
				kvfree(buf->pages);
				kfree(buf);
				return ERR_PTR(-ENOMEM);
			}
		}
		buf->hugepages = false;
		buf->npages    = np;
		buf->size      = size;
		gb_dbg("allocated %u pages (%zuMB, 4K %s)\n",
		       np, size >> 20, tier3 ? "NVMe-spillable" : "System DDR");
	}

done:
	buf->id          = 0;
	buf->tier        = tier3 ? GB_TIER3_NVME : GB_TIER2_SDDR;
	buf->alloc_flags = flags;
	buf->alloc_jiffies = jiffies;
	buf->last_jiffies  = jiffies;
	buf->owner_pid     = task_pid_vnr(current);
	/* GB_ALLOC_KV_CACHE in T2: auto-freeze (KV is read/written every token;
	 * evicting it to T3 would collapse generation speed to ~1.8 GB/s).
	 * GB_ALLOC_T1_PRIORITY: also freeze + mark t1_priority so weight bufs
	 * are preferred for eviction over this one. */
	if (!tier3 && (flags & (GB_ALLOC_KV_CACHE | GB_ALLOC_T1_PRIORITY))) {
		buf->frozen      = 1;
		buf->t1_priority = 1;
	} else {
		buf->frozen      = (flags & GB_ALLOC_FROZEN) ? 1 : 0;
		buf->t1_priority = 0;
	}
	buf->session_priority = (!tier3 && (flags & GB_ALLOC_SESSION_PROTECTED)) ? 1 : 0;
	INIT_LIST_HEAD(&buf->lru_node);

	if (tier3)
		atomic64_add(buf->size, &gb_dev.nvme_allocated);
	else
		atomic64_add(buf->size, &gb_dev.pool_allocated);
	/* KV cache usage accounting (ExLlamaV3 native path + GREENBOOST_KV_OVERFLOW) */
	if (flags & GB_ALLOC_KV_CACHE) {
		atomic64_add(buf->size, &gb_dev.kv_used_bytes);
		if (!tier3)
			atomic64_add(buf->size, &gb_dev.kv_t2_bytes);
	}
	atomic_inc(&gb_dev.active_bufs);

	spin_lock(&gb_dev.lru_lock);
	list_add_tail(&buf->lru_node, &gb_dev.lru_list);
	spin_unlock(&gb_dev.lru_lock);

	return buf;
}

/* DRA result sysfs variables — declared here (before gb_ioctl) so that
 * GB_IOCTL_ALLOC can write the result fd/offset at allocation time.
 * The sysfs show functions further down in the file read these same vars. */
static atomic_t   dra_result_fd     = ATOMIC_INIT(-1);
static atomic64_t dra_result_offset = ATOMIC64_INIT(0);

/* Sysfs atomics — defined here so the IOCTL handler (below) and the sysfs
 * show/store functions (further below) can both reference them. */
static atomic_t nvlink_ready          = ATOMIC_INIT(0);
static atomic_t compute_domain_active = ATOMIC_INIT(0);

/* ------------------------------------------------------------------ */
/*  DMA-BUF IDR registration + fd installation helper                  */
/* ------------------------------------------------------------------ */

/* REF-01: Common steps 3–4 of the DMA-BUF export sequence: IDR alloc,
 * then dma_buf_fd() install.  Called by GB_IOCTL_ALLOC and GB_IOCTL_PIN_USER_PTR.
 *
 * On success: returns the installed fd (>= 0) and buf->id is set.
 * On failure: dma_buf_put(dmabuf) is called internally (triggers gb_release),
 *             returns a negative errno.
 */
static int gb_dmabuf_idr_and_install_fd(struct gb_buf *buf, struct dma_buf *dmabuf)
{
	int id, fd;

	mutex_lock(&gb_dev.lock);
	id = idr_alloc(&gb_dev.idr, buf, 1, 0, GFP_KERNEL);
	mutex_unlock(&gb_dev.lock);
	if (id < 0) {
		/* dma_buf_put triggers gb_release → frees pages */
		dma_buf_put(dmabuf);
		return id;
	}
	buf->id = id;

	fd = dma_buf_fd(dmabuf, O_CLOEXEC);
	if (fd < 0) {
		mutex_lock(&gb_dev.lock);
		idr_remove(&gb_dev.idr, buf->id);
		mutex_unlock(&gb_dev.lock);
		buf->id = 0;
		dma_buf_put(dmabuf); /* triggers gb_release */
	}
	return fd;
}

/* ------------------------------------------------------------------ */
/*  IOCTL                                                               */
/* Forward declaration — defined after gb_auto_evict_cold */
static int gb_release_pid_buffers(pid_t pid);

/* ------------------------------------------------------------------ */

static long gb_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	switch (cmd) {

	case GB_IOCTL_ALLOC: {
		struct gb_alloc_req req;
		struct gb_buf *buf;
		DEFINE_DMA_BUF_EXPORT_INFO(exp_info);
		struct dma_buf *dmabuf;
		int fd;
		unsigned int j;

		if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
			return -EFAULT;

		/* SEC-02: Cap large allocations so the kernel page-pointer array cannot
		 * exhaust memory before the OOM guard fires.
		 *
		 * T2 (DDR) pages live in RAM → cap T2 portion at 90% of total RAM.
		 * T3 (NVMe) pages are file-backed and swappable → NVMe capacity is the
		 * binding constraint, not RAM.  Accordingly hw_cap = RAM_90% + T3_bytes
		 * so a request spanning T2+T3 is accepted; gb_alloc_buf() routes excess
		 * pages to T3 when T2 is full. */
		{
			u64 ram_cap = (u64)totalram_pages() * PAGE_SIZE * 9 / 10;
			u64 t3_cap  = (u64)nvme_pool_gb * (1ULL << 30);
			u64 hw_cap  = ram_cap + t3_cap;
			u64 t2t3_gb = (u64)virtual_vram_gb + (u64)nvme_pool_gb;
			u64 max_req = min_t(u64, t2t3_gb * (1ULL << 30), hw_cap);
			if (!req.size || req.size > max_req)
				return -EINVAL;
		}

		/* 1. Pin System DDR pages */
		buf = gb_alloc_buf((size_t)req.size, req.flags);
		if (IS_ERR(buf))
			return PTR_ERR(buf);

		/* 2. Export as DMA-BUF */
		exp_info.ops   = &gb_dma_buf_ops;
		exp_info.size  = buf->size;
		exp_info.flags = O_RDWR | O_CLOEXEC;
		exp_info.priv  = buf;

		dmabuf = dma_buf_export(&exp_info);
		if (IS_ERR(dmabuf)) {
			/* gb_release won't be called — undo manually.
			 * Must mirror gb_alloc_buf cleanup: hugepages use hpages[],
			 * 4K pages use pages[], and T3 uses nvme_allocated not pool_allocated.
			 */
			atomic_dec(&gb_dev.active_bufs);
			if (buf->tier == GB_TIER3_NVME)
				atomic64_sub(buf->size, &gb_dev.nvme_allocated);
			else
				atomic64_sub(buf->size, &gb_dev.pool_allocated);
			spin_lock(&gb_dev.lru_lock);
			list_del_init(&buf->lru_node);
			spin_unlock(&gb_dev.lru_lock);
			if (buf->hugepages) {
				for (j = 0; j < buf->nhpages; j++)
					__free_pages(buf->hpages[j], GB_HPAGE_ORDER);
				kvfree(buf->hpages);
			} else {
				for (j = 0; j < buf->npages; j++)
					__free_page(buf->pages[j]);
				kvfree(buf->pages);
			}
			kfree(buf);
			return PTR_ERR(dmabuf);
		}
		buf->dmabuf = dmabuf;

		/* 3+4. Register in IDR and install fd (REF-01: shared helper) */
		fd = gb_dmabuf_idr_and_install_fd(buf, dmabuf);
		if (fd < 0)
			return fd;

		/* Set DRA result FD for kubelet plugin to read */
		atomic_set(&dra_result_fd, fd);
		atomic64_set(&dra_result_offset, 0);

		req.fd = fd;
		if (copy_to_user((void __user *)arg, &req, sizeof(req)))
			return -EFAULT; /* fd already installed, caller must close it */

		pr_info(DRIVER_NAME
			": allocated %zuMB buffer (id=%d fd=%d)\n",
			buf->size >> 20, buf->id, fd);
		return 0;
	}

	case GB_IOCTL_PIN_USER_PTR: {
		struct gb_pin_req req;
		struct gb_buf *buf;
		DEFINE_DMA_BUF_EXPORT_INFO(exp_info);
		struct dma_buf *dmabuf;
		int fd;
		unsigned int j;

		if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
			return -EFAULT;

		if (!req.size || !req.vaddr ||
		    req.size > ((u64)virtual_vram_gb + (u64)nvme_pool_gb) * (1ULL << 30))
			return -EINVAL;

		/* SEC-03: Reject pfnmap/IO ranges (vDSO, MMIO) — pinning shared kernel
		 * pages with FOLL_LONGTERM can exhaust the free-page pool via COW breaks
		 * on anonymous shared mappings, causing a system-wide DoS. */
		{
			struct vm_area_struct *vma;
			bool is_special = false;

			mmap_read_lock(current->mm);
			vma = find_vma(current->mm, (unsigned long)req.vaddr);
			if (!vma || (vma->vm_flags & (VM_IO | VM_PFNMAP)))
				is_special = true;
			mmap_read_unlock(current->mm);

			if (is_special)
				return -EPERM;
		}

		/* 1. Pin user memory */
		buf = gb_pin_user_buf((u64)req.vaddr, (size_t)req.size, req.flags);
		if (IS_ERR(buf))
			return PTR_ERR(buf);

		/* 2. Export as DMA-BUF */
		exp_info.ops   = &gb_dma_buf_ops;
		exp_info.size  = buf->size;
		exp_info.flags = O_RDWR | O_CLOEXEC;
		exp_info.priv  = buf;

		dmabuf = dma_buf_export(&exp_info);
		if (IS_ERR(dmabuf)) {
			atomic_dec(&gb_dev.active_bufs);
			atomic64_sub(buf->size, &gb_dev.pool_allocated);
			for (j = 0; j < buf->npages; j++)
				unpin_user_page(buf->pages[j]);
			kvfree(buf->pages);
			kfree(buf);
			return PTR_ERR(dmabuf);
		}
		buf->dmabuf = dmabuf;

		/* 3+4. Register in IDR and install fd (REF-01: shared helper) */
		fd = gb_dmabuf_idr_and_install_fd(buf, dmabuf);
		if (fd < 0)
			return fd;

		req.fd = fd;
		if (copy_to_user((void __user *)arg, &req, sizeof(req)))
			return -EFAULT;

		pr_info(DRIVER_NAME
			": pinned %zuMB user buffer (id=%d fd=%d)\n",
			buf->size >> 20, buf->id, fd);
		return 0;
	}

	case GB_IOCTL_GET_INFO: {
		struct gb_info info;
		struct sysinfo si;
		u64 free_bytes, reserve_bytes, alloc_bytes;
		u64 t3_used_mb, t3_max_mb;

		si_meminfo(&si);
		gb_get_t3_stats(&t3_used_mb, &t3_max_mb);

		free_bytes    = (u64)si_mem_available() * PAGE_SIZE;
		reserve_bytes = (u64)safety_reserve_gb * (1ULL << 30);
		alloc_bytes   = (u64)atomic64_read(&gb_dev.pool_allocated);

		memset(&info, 0, sizeof(info));

		/* Tier 1 */
		info.vram_physical_mb    = (u64)physical_vram_gb * 1024ULL;

		/* Tier 2 */
		info.total_ram_mb        = ((u64)si.totalram * si.mem_unit) >> 20;
		info.free_ram_mb         = free_bytes >> 20;
		info.allocated_mb        = alloc_bytes >> 20;
		info.max_pool_mb         = (u64)virtual_vram_gb * 1024ULL;
		info.safety_reserve_mb   = reserve_bytes >> 20;
		info.available_mb        = (free_bytes > reserve_bytes + alloc_bytes)
			? (free_bytes - reserve_bytes - alloc_bytes) >> 20 : 0;
		info.active_buffers      = (gb_u32)atomic_read(&gb_dev.active_bufs);
		info.oom_active          = (gb_u32)atomic_read(&gb_dev.oom_active);

		/* Tier 3 — file-backed */
		info.nvme_swap_total_mb  = t3_max_mb;
		info.nvme_swap_used_mb   = t3_used_mb;
		info.nvme_swap_free_mb   = (t3_max_mb > t3_used_mb) ? t3_max_mb - t3_used_mb : 0;
		info.nvme_t3_allocated_mb = t3_used_mb;
		info.swap_pressure       = (gb_u32)atomic_read(&gb_dev.swap_pressure);
		info.t2_pressure         = (gb_u32)atomic_read(&gb_dev.t2_pressure);
		info.kv_reserve_mb       = (gb_u32)kv_reserve_mb;
		info.kv_used_mb          = (gb_u32)(atomic64_read(&gb_dev.kv_used_bytes) >> 20);
		{
			s64 _raw = atomic64_read(&gb_dev.kv_t2_bytes);
			info.kv_t2_mb = (_raw > 0) ? (gb_u32)(_raw >> 20) : 0;
		}

		/* TurboQuant compression stats */
		info.kv_compression_bits     = gb_dev.turboquant_bits;
		/* MIN-01: kv_compressed_mb — estimate from kv_used_mb and compression ratio.
		 * compression_bits 0 = disabled; 2/3/4 bits → ~(32/bits - 1) * kv_used savings.
		 * Report 0 when TurboQuant is off or kv_used is zero. */
		if (gb_dev.turboquant_enabled && gb_dev.turboquant_bits > 0) {
			u64 kv_raw_mb = (u64)(atomic64_read(&gb_dev.kv_used_bytes) >> 20);
			/* Nominal saving: storing at N bits vs fp16 (16 bits): ratio = 16/N.
			 * Compressed size = kv_raw_mb * N/16; saved = kv_raw_mb - compressed.
			 * Use integer arithmetic: saved = kv_raw_mb * (16 - bits) / 16. */
			info.kv_compressed_mb = (gb_u32)(kv_raw_mb * (16 - gb_dev.turboquant_bits) / 16);
		}
		/* TODO: wire to a real per-session counter when TQ session tracking
		 * is implemented.  For now report 0 — turboquant_enabled is a bool,
		 * not a session count, and reporting it here would be misleading. */
		info.kv_compression_sessions = 0;

		/* Phase reset sequence for shim-side model-swap detection */
		info.phase_reset_seq = (gb_u32)atomic_read(&gb_dev.phase_reset_seq);

		/* Combined */
		info.total_combined_mb   = info.vram_physical_mb
					 + info.max_pool_mb
					 + info.nvme_swap_total_mb;

		if (copy_to_user((void __user *)arg, &info, sizeof(info)))
			return -EFAULT;
		return 0;
	}

	case GB_IOCTL_RESET:
		/* MED-02: Previous no-op left oom_active set after a crash,
		 * causing all future allocations to fail.  Clear the OOM flag
		 * so the pool can recover once the caller has closed its fds.
		 * Full buffer teardown still requires userspace to close all
		 * DMA-BUF fds (which triggers gb_release via dma_buf refcount). */
		atomic_set(&gb_dev.oom_active, 0);
		pr_info(DRIVER_NAME
			": RESET — OOM guard cleared; close all DMA-BUF fds to release buffers\n");
		return 0;

	case GB_IOCTL_RESET_PHASE:
		/* ENH-02: Phase detector reset for model-swap.
		 * Synapse CLI calls this before swapping models so the shim's
		 * phase detector starts from INIT instead of remaining stuck in
		 * STEADY (which would misclassify new weight allocs as KV cache).
		 *
		 * Mechanism: increment phase_reset_seq; shim reads this via
		 * GB_IOCTL_GET_INFO on every GB_KV_REFRESH_INTERVAL alloc boundary.
		 * When the sequence number changes, the shim resets g_alloc_phase
		 * to GB_PHASE_INIT and clears g_kv_allocated_t1_bytes. */
		atomic_inc(&gb_dev.phase_reset_seq);
		pr_info(DRIVER_NAME ": RESET_PHASE — seq=%u; shim will reset on next refresh\n",
			atomic_read(&gb_dev.phase_reset_seq));
		return 0;

	case GB_IOCTL_MADVISE: {
		struct gb_madvise_req req;
		struct gb_buf *buf;
		int madvise_ret = 0;

		if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
			return -EFAULT;

		mutex_lock(&gb_dev.lock);
		buf = idr_find(&gb_dev.idr, req.buf_id);
		/* Hold a DMA-BUF reference so concurrent fd-close cannot free buf
		 * between the mutex unlock and the lru_lock critical section. */
		if (buf)
			get_dma_buf(buf->dmabuf);
		mutex_unlock(&gb_dev.lock);
		if (!buf)
			return -ENOENT;

		/* SEC-01: Only the allocating process (or CAP_SYS_ADMIN) may change
		 * the advise state of a buffer.  Prevents inter-process LRU poisoning. */
		if (buf->owner_pid != task_pid_vnr(current) && !capable(CAP_SYS_ADMIN)) {
			dma_buf_put(buf->dmabuf);
			return -EPERM;
		}

		spin_lock(&gb_dev.lru_lock);
		switch (req.advise) {
		case GB_MADVISE_HOT:
			buf->last_jiffies = jiffies;
			list_move(&buf->lru_node, &gb_dev.lru_list);  /* to head */
			break;
		case GB_MADVISE_COLD:
			list_move_tail(&buf->lru_node, &gb_dev.lru_list); /* to tail */
			break;
		case GB_MADVISE_FREEZE:
			buf->frozen = 1;
			break;
		case GB_MADVISE_T1_PREFER:
			/* Mark as T1-priority (KV-like): freeze in T2 LRU and move
			 * to head so weight buffers are evicted before this one.
			 * Used by the shim after a KV-hinted overflow lands in T2. */
			buf->t1_priority = 1;
			buf->frozen      = 1;
			buf->last_jiffies = jiffies;
			list_move(&buf->lru_node, &gb_dev.lru_list);  /* to head */
			gb_dbg("madvise T1_PREFER buf id=%d — frozen + LRU head\n",
			       buf->id);
			break;
		case GB_MADVISE_SESSION_PROTECT:
			buf->session_priority = 1;
			gb_dbg("madvise SESSION_PROTECT buf id=%d\n", buf->id);
			break;
		case GB_MADVISE_SESSION_DEMOTE:
			buf->session_priority = 0;
			gb_dbg("madvise SESSION_DEMOTE buf id=%d\n", buf->id);
			break;
		default:
			madvise_ret = -EINVAL;
			break;
		}
		spin_unlock(&gb_dev.lru_lock);
		dma_buf_put(buf->dmabuf);
		if (!madvise_ret)
			gb_dbg("madvise buf id=%d advise=%u\n", req.buf_id,
			       req.advise);
		return madvise_ret;
	}

	case GB_IOCTL_EVICT: {
		struct gb_evict_req req;
		struct gb_buf *buf;

		if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
			return -EFAULT;

		mutex_lock(&gb_dev.lock);
		buf = idr_find(&gb_dev.idr, req.buf_id);
		/* Hold a DMA-BUF reference so concurrent fd-close cannot free buf
		 * between the mutex unlock and the tier/lru_lock operations. */
		if (buf)
			get_dma_buf(buf->dmabuf);
		mutex_unlock(&gb_dev.lock);
		if (!buf)
			return -ENOENT;

		/* SEC-01: Only the allocating process (or CAP_SYS_ADMIN) may evict
		 * a buffer.  Brute-forcing IDR ids could otherwise force another
		 * process's KV cache to slow NVMe. */
		if (buf->owner_pid != task_pid_vnr(current) && !capable(CAP_SYS_ADMIN)) {
			dma_buf_put(buf->dmabuf);
			return -EPERM;
		}

		/* MED-05: Refuse eviction of frozen or T1-priority (KV cache) buffers.
		 * Previously any caller with the buf_id could force KV to NVMe,
		 * collapsing generation throughput to ~1.8 GB/s. */
		if (buf->frozen || buf->t1_priority) {
			dma_buf_put(buf->dmabuf);
			return -EPERM;
		}

		/* Evict T2 buffer to T3 backing file.
		 * Hugepages are pinned (DMA) and cannot be evicted. */
		if (buf->tier == GB_TIER2_SDDR && !buf->hugepages) {
			int evict_ret;

			/* Move accounting T2 → T3 */
			atomic64_sub(buf->size, &gb_dev.pool_allocated);
			atomic64_add(buf->size, &gb_dev.nvme_allocated);
			buf->tier = GB_TIER3_NVME;
			spin_lock(&gb_dev.lru_lock);
			list_del_init(&buf->lru_node);
			spin_unlock(&gb_dev.lru_lock);

			/* Write pages to file and free RAM */
			evict_ret = gb_t3_evict_buf(buf);
			if (evict_ret != 0) {
				/* Roll back accounting */
				atomic64_sub(buf->size, &gb_dev.nvme_allocated);
				atomic64_add(buf->size, &gb_dev.pool_allocated);
				buf->tier = GB_TIER2_SDDR;
				spin_lock(&gb_dev.lru_lock);
				list_add_tail(&buf->lru_node, &gb_dev.lru_list);
				spin_unlock(&gb_dev.lru_lock);
				dma_buf_put(buf->dmabuf);
				return evict_ret;
			}
			gb_dbg("evict buf id=%d: T2→T3 file (%zuMB)\n",
			       buf->id, buf->size >> 20);
		}
		dma_buf_put(buf->dmabuf);
		return 0;
	}

	case GB_IOCTL_POLL_FD: {
		struct gb_poll_req req;
		struct eventfd_ctx *ctx;

		if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
			return -EFAULT;

		ctx = eventfd_ctx_fdget(req.efd);
		if (IS_ERR(ctx))
			return PTR_ERR(ctx);

		/* Replace any previously registered eventfd.
		 * efd_lock guards both this write path and the watchdog read path. */
		{
			struct eventfd_ctx *old;
			unsigned long flags;
			spin_lock_irqsave(&gb_dev.efd_lock, flags);
			old = gb_dev.pressure_efd;
			gb_dev.pressure_efd = ctx;
			spin_unlock_irqrestore(&gb_dev.efd_lock, flags);
			if (old)
				eventfd_ctx_put(old);
		}
		gb_dbg("pressure eventfd registered (efd=%d)\n", req.efd);
		return 0;
	}

	case GB_IOCTL_SET_KV_RESERVE: {
		struct gb_kv_reserve_req req;
		int requested, max_kv, clamped;

		/* SEC-01: Changing the system-wide KV reserve affects all inference
		 * sessions — restrict to privileged callers. */
		if (!capable(CAP_SYS_ADMIN))
			return -EPERM;

		if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
			return -EFAULT;

		/* Clamp to effective T1 headroom: never reserve more than T1 - 2 GB safety.
		 * Without this clamp, the shim overflow check is ALWAYS true and every
		 * cudaMalloc overflows to T2, collapsing performance to ~32 GB/s. */
		requested = (int)req.reserve_mb;
		max_kv    = physical_vram_gb * 1024 - 2048;
		if (max_kv < 512)
			max_kv = 512;
		clamped = (requested > max_kv) ? max_kv : requested;
		/* MED-03: Use WRITE_ONCE to avoid compiler-torn write; sysfs show
		 * and watchdog read this concurrently without a lock. */
		WRITE_ONCE(kv_reserve_mb, clamped);

		if (clamped != requested)
			pr_info(DRIVER_NAME
				": KV reserve: requested %d MB clamped to %d MB "
				"(T1=%dGB limit). Weights overflow to T2 sooner; "
				"KV stays in T1.\n",
				requested, clamped, physical_vram_gb);
		else
			pr_info(DRIVER_NAME
				": KV cache T1 reserve updated to %d MB "
				"(weights overflow to T2 sooner; KV stays in T1)\n",
				kv_reserve_mb);
		return 0;
	}

	case GB_IOCTL_SET_POOL_CAP: {
		struct gb_pool_cap_req req;
		struct sysinfo si;
		u64 ram_total_mb, max_cap_mb;
		int new_cap_gb;
		const int min_cap_gb = 4;

		/* SEC-01: Reducing the T2 pool cap could starve running inference
		 * sessions — restrict to privileged callers. */
		if (!capable(CAP_SYS_ADMIN))
			return -EPERM;

		if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
			return -EFAULT;

		si_meminfo(&si);
		ram_total_mb = ((u64)si.totalram * si.mem_unit) >> 20;

		/* Hard ceiling: total RAM minus safety reserve */
		max_cap_mb = (ram_total_mb > (u64)safety_reserve_gb * 1024ULL)
			   ? ram_total_mb - (u64)safety_reserve_gb * 1024ULL
			   : (u64)min_cap_gb * 1024ULL;

		/* Clamp requested cap */
		if (req.cap_mb < (u64)min_cap_gb * 1024ULL)
			req.cap_mb = (u64)min_cap_gb * 1024ULL;
		if (req.cap_mb > max_cap_mb)
			req.cap_mb = max_cap_mb;

		req.prev_mb = (u64)virtual_vram_gb * 1024ULL;
		new_cap_gb = (int)(req.cap_mb / 1024ULL);
		if (new_cap_gb < min_cap_gb)
			new_cap_gb = min_cap_gb;

		virtual_vram_gb = new_cap_gb;
		pr_info(DRIVER_NAME
			": T2 pool cap: %llu MB -> %d GB "
			"(requested %llu MB, RAM %llu MB, safety %d GB)\n",
			req.prev_mb, virtual_vram_gb, req.cap_mb,
			ram_total_mb, safety_reserve_gb);

		if (copy_to_user((void __user *)arg, &req, sizeof(req)))
			return -EFAULT;
		return 0;
	}

	case GB_IOCTL_SET_TURBOQUANT: {
		struct gb_turboquant_req req;

		/* SEC-01: TurboQuant config is global and changes quantization for all
		 * ongoing inference sessions — restrict to privileged callers. */
		if (!capable(CAP_SYS_ADMIN))
			return -EPERM;

		if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
			return -EFAULT;

		/* Validate bits: 0=auto, 2, 3, or 4 only */
		if (req.bits != 0 && req.bits != 2 && req.bits != 3 && req.bits != 4)
			return -EINVAL;

		/* SEC-04: Write the four TQ fields as a group under lru_lock to prevent
		 * torn reads in status_show and hw_info_show which read them concurrently. */
		spin_lock(&gb_dev.lru_lock);
		gb_dev.turboquant_enabled  = req.enabled;
		gb_dev.turboquant_bits     = req.bits;
		gb_dev.turboquant_head_dim = req.head_dim;
		gb_dev.turboquant_seed     = req.seed ? req.seed : 42;
		spin_unlock(&gb_dev.lru_lock);

		pr_info(DRIVER_NAME
			": TurboQuant KV compression %s (bits=%u head_dim=%u seed=%u)\n",
			req.enabled ? "enabled" : "disabled",
			req.bits, req.head_dim, gb_dev.turboquant_seed);
		return 0;
	}

	/* IOCTL cmds 12 (RECLASSIFY) and 13 (QUERY_BUFS) are reserved ABI gaps.
	 * They were daemon-only operations and are not implemented here. */

	case GB_IOCTL_GET_POOL_INFO_V3: {
		struct gb_pool_info_v3 v3;
		struct sysinfo si;

		memset(&v3, 0, sizeof(v3));
		si_meminfo(&si);

		v3.t1_physical_mb    = (gb_u64)physical_vram_gb * 1024;
		v3.t1_nvlink_total_mb = 0; /* set by kubelet plugin via nvlink_ready + gpu_count */
		v3.t2_total_mb       = (gb_u64)virtual_vram_gb * 1024;
		v3.t2_used_mb        = (gb_u64)(atomic64_read(&gb_dev.pool_allocated) >> 20);
		v3.t2_available_mb   = (v3.t2_total_mb > v3.t2_used_mb) ?
		                       v3.t2_total_mb - v3.t2_used_mb : 0;
		v3.t3_total_mb       = (gb_u64)nvme_pool_gb * 1024;
		v3.t3_used_mb        = (gb_u64)(atomic64_read(&gb_dev.nvme_allocated) >> 20);
		v3.nvlink_ready      = (gb_u32)atomic_read(&nvlink_ready);
		v3.compute_domain_active = (gb_u32)atomic_read(&compute_domain_active);
		v3.watchdog_pressure = (gb_u32)atomic_read(&gb_dev.t2_pressure);
		v3.active_buffers    = (gb_u32)atomic_read(&gb_dev.active_bufs);
		v3.oom_active        = (gb_u32)atomic_read(&gb_dev.oom_active);
		v3.gpu_count         = 0; /* set by nvlink_gpu_count param if used */
		v3.kv_reserve_mb     = (gb_u32)kv_reserve_mb;
		v3._pad              = 0;

		if (copy_to_user((void __user *)arg, &v3, sizeof(v3)))
			return -EFAULT;
		return 0;
	}

	case GB_IOCTL_RELEASE_PID: {
		struct gb_release_pid_req req;
		pid_t target;

		if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
			return -EFAULT;

		target = req.pid ? (pid_t)req.pid : task_pid_vnr(current);
		/* SEC-01: only allow releasing another process's buffers with
		 * CAP_SYS_ADMIN — prevents one user from starving another. */
		if (target != task_pid_vnr(current) && !capable(CAP_SYS_ADMIN))
			return -EPERM;

		return gb_release_pid_buffers(target);
	}

	case GB_IOCTL_SET_T3_CAP: {
		struct gb_t3_cap_req req;
		int ret;

		if (!capable(CAP_SYS_ADMIN))
			return -EPERM;
		if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
			return -EFAULT;

		spin_lock(&gb_dev.t3_file_lock);
		req.prev_mb = gb_dev.t3_file_max >> 20;
		spin_unlock(&gb_dev.t3_file_lock);

		/* Open backing file on first use (enables T3 on demand). */
		if (!gb_dev.t3_file) {
			ret = gb_t3_file_open();
			if (ret)
				return ret;
		}

		/* cap_mb == 0 → disk-limited (no cap). */
		spin_lock(&gb_dev.t3_file_lock);
		gb_dev.t3_file_max = req.cap_mb ? (req.cap_mb << 20) : 0;
		spin_unlock(&gb_dev.t3_file_lock);

		/* Update nvme_pool_gb so allocation gating at line 822 passes. */
		nvme_pool_gb = req.cap_mb ? (int)(req.cap_mb / 1024) : INT_MAX / 2;

		pr_info(DRIVER_NAME
			": T3 cap set: prev=%llu MB new=%llu MB (%s)\n",
			req.prev_mb, req.cap_mb,
			req.cap_mb ? "capped" : "disk-limited");

		if (copy_to_user((void __user *)arg, &req, sizeof(req)))
			return -EFAULT;
		return 0;
	}

	case GB_IOCTL_SESSION_IDLE:
	case GB_IOCTL_SESSION_ACTIVE: {
		struct gb_session_req req;
		struct gb_buf *buf;
		struct dma_buf **refs;
		pid_t target;
		int id, n = 0, cap, i;
		int to_head = (cmd == GB_IOCTL_SESSION_ACTIVE);

		if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
			return -EFAULT;

		target = req.pid ? (pid_t)req.pid : task_pid_vnr(current);
		if (target != task_pid_vnr(current) && !capable(CAP_SYS_ADMIN))
			return -EPERM;

		/* Phase 1: collect matching T2 DMA-BUF refs under IDR lock */
		cap = atomic_read(&gb_dev.active_bufs) + 1;
		refs = kvmalloc_array(cap, sizeof(*refs), GFP_KERNEL);
		if (!refs)
			return -ENOMEM;

		mutex_lock(&gb_dev.lock);
		idr_for_each_entry(&gb_dev.idr, buf, id) {
			if (buf->owner_pid == target &&
			    buf->tier == GB_TIER2_SDDR && n < cap) {
				get_dma_buf(buf->dmabuf);
				refs[n++] = buf->dmabuf;
			}
		}
		mutex_unlock(&gb_dev.lock);

		/* Phase 2: move LRU nodes under lru_lock; buf is kept alive by ref */
		spin_lock(&gb_dev.lru_lock);
		for (i = 0; i < n; i++) {
			/* Recover gb_buf from dmabuf->priv */
			struct gb_buf *b = (struct gb_buf *)refs[i]->priv;
			if (to_head)
				list_move(&b->lru_node, &gb_dev.lru_list);
			else
				list_move_tail(&b->lru_node, &gb_dev.lru_list);
		}
		spin_unlock(&gb_dev.lru_lock);

		for (i = 0; i < n; i++)
			dma_buf_put(refs[i]);
		kvfree(refs);

		gb_dbg("session %s for PID %d (%d bufs)\n",
		       to_head ? "ACTIVE" : "IDLE", target, n);
		return 0;
	}

	default:
		return -ENOTTY;
	}
}

/* ------------------------------------------------------------------ */
/*  Sysfs attributes                                                    */
/* ------------------------------------------------------------------ */

static ssize_t status_show(struct device *dev,
			       struct device_attribute *attr, char *buf)
{
	struct sysinfo si;
	u64 ram_total_mb, ram_free_mb, t2_alloc_mb, t2_avail_mb, reserve_mb;
	u64 t3_alloc_mb, t3_max_mb;
	u64 combined_mb, kv_used_mb, kv_t2_mb;
	int pressure;
	const char *pressure_str;
	const char *kv_placement;
	/* SEC-04: Snapshot turboquant fields under lru_lock to prevent torn reads
	 * from a concurrent GB_IOCTL_SET_TURBOQUANT write. */
	u32 tq_enabled, tq_bits;

	si_meminfo(&si);
	gb_get_t3_stats(&t3_alloc_mb, &t3_max_mb);

	ram_total_mb = ((u64)si.totalram * si.mem_unit) >> 20;
	ram_free_mb  = ((u64)si_mem_available() * PAGE_SIZE) >> 20;
	t2_alloc_mb  = (u64)atomic64_read(&gb_dev.pool_allocated) >> 20;
	kv_used_mb   = (u64)atomic64_read(&gb_dev.kv_used_bytes) >> 20;
	{
		s64 _raw_kv_t2 = atomic64_read(&gb_dev.kv_t2_bytes);
		kv_t2_mb = (_raw_kv_t2 > 0) ? (u64)(_raw_kv_t2 >> 20) : 0;
	}
	reserve_mb   = (u64)safety_reserve_gb * 1024ULL;
	combined_mb  = (u64)physical_vram_gb * 1024ULL
		     + (u64)virtual_vram_gb  * 1024ULL
		     + (t3_max_mb > 0 ? t3_max_mb : (u64)nvme_pool_gb * 1024ULL);
	t2_avail_mb  = (ram_free_mb > reserve_mb + t2_alloc_mb)
		       ? ram_free_mb - reserve_mb - t2_alloc_mb : 0;
	pressure     = atomic_read(&gb_dev.swap_pressure);
	pressure_str = (pressure == GB_SWAP_PRESSURE_CRITICAL) ? "CRITICAL (>90%)" :
		       (pressure == GB_SWAP_PRESSURE_WARN)     ? "warn (>75%)"    :
		                                                  "ok";
	kv_placement = (kv_t2_mb > 0) ? "SPILLED TO T2 DDR — increase kv_reserve_mb" :
		        (kv_reserve_mb > 0) ? "T1 VRAM [reserve intact]" :
		                              "T1 VRAM (no reserve set)";
	spin_lock(&gb_dev.lru_lock);
	tq_enabled = gb_dev.turboquant_enabled;
	tq_bits    = gb_dev.turboquant_bits;
	spin_unlock(&gb_dev.lru_lock);

	/* MED-06: sysfs_emit clamps at PAGE_SIZE-1 but returns silently truncated
	 * length, breaking parsers.  Log a warning so we know when to split. */
	{
	int ret = sysfs_emit(buf,
		"=== GreenBoost " GB_VERSION " — CUDA Memory Pool Info ===\n"
		"\n"
		"Tier 1  GPU VRAM           : %4d GB   [hot layers + KV cache]\n"
		"  KV cache T1 reserve      : %4d MB   [kept free for KV cache]\n"
		"  KV cache placement       : %s\n"
		"Tier 2  System RAM pool    : %4d GB   PCIe DMA  [cold layers]\n"
		"Tier 3  T3 backing file    : %4llu GB  NVMe file  [GreenBoost-managed, pre-allocated]\n"
		"        ─────────────────────────────────\n"
		"        Combined model view: %4llu GB\n"
		"\n"
		"── Tier 2 (System RAM) ─────────────────────\n"
		"  Total RAM                : %llu MB\n"
		"  Avail RAM (MemAvailable) : %llu MB\n"
		"  Safety reserve           : %llu MB\n"
		"  T2 allocated             : %llu MB  (%llu%%)\n"
		"  T2 available             : %llu MB\n"
		"  Active DMA-BUF objects   : %d\n"
		"  OOM guard                : %s\n"
		"  Page mode                : %s\n"
		"\n"
		"── KV Cache & TurboQuant ───────────────────\n"
		"  KV in T1 (native VRAM)   : managed by CUDA (reserve: %d MB)\n"
		"  KV in T2 (DDR spill)     : %llu MB\n"
		"  KV in T3 (NVMe swap)     : %llu MB\n"
		"  KV tagged total (T2+T3)  : %llu MB\n"
		"  TurboQuant Compression   : %s\n"
		"\n"
		"── Tier 3 (GreenBoost backing file) ───────\n"
		"  Backing file             : %s\n"
		"  File cap                 : %s\n"
		"  T3 allocated             : %llu MB\n"
		"  T3 pressure              : %s\n",
		physical_vram_gb,
		kv_reserve_mb,
		kv_placement,
		virtual_vram_gb,
		t3_max_mb / 1024ULL,
		combined_mb / 1024ULL,
		ram_total_mb, ram_free_mb, reserve_mb,
		t2_alloc_mb,
		(t2_alloc_mb * 100ULL) / (((u64)virtual_vram_gb * 1024ULL) != 0 ? (u64)virtual_vram_gb * 1024ULL : 1ULL),
		t2_avail_mb,
		atomic_read(&gb_dev.active_bufs),
		atomic_read(&gb_dev.oom_active) ? "YES" : "no",
		use_hugepages ? "2 MB hugepages (T2) / 4K direct (T3)"
			      : "4 KB pages",
		kv_reserve_mb, kv_t2_mb, (kv_used_mb > kv_t2_mb) ? (kv_used_mb - kv_t2_mb) : 0ULL, kv_used_mb,
		tq_enabled ? (tq_bits == 2 ? "ACTIVE (2-bit)" :
		              tq_bits == 3 ? "ACTIVE (3-bit)" :
		              tq_bits == 4 ? "ACTIVE (4-bit)" : "ACTIVE (auto)") : "disabled",
		gb_dev.t3_file ? t3_file_path : "disabled",
		t3_max_mb > 0 ? "capped" : (gb_dev.t3_file ? "disk-limited" : "T3 disabled"),
		t3_alloc_mb,
		pressure_str);
	if (ret >= PAGE_SIZE - 1)
		pr_warn_once(DRIVER_NAME
			": status_show output truncated at PAGE_SIZE — consider splitting sysfs files\n");
	return ret;
	}
}
static DEVICE_ATTR_RO(status);

static ssize_t active_buffers_show(struct device *dev,
				    struct device_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%d\n", atomic_read(&gb_dev.active_bufs));
}
static DEVICE_ATTR_RO(active_buffers);

static ssize_t hw_info_show(struct device *dev,
			     struct device_attribute *attr, char *buf)
{
	struct sysinfo si;
	u64 ram_total_mb;
	const char *board_vendor = dmi_get_system_info(DMI_BOARD_VENDOR);
	const char *board_name   = dmi_get_system_info(DMI_BOARD_NAME);

	si_meminfo(&si);
	ram_total_mb = ((u64)si.totalram * si.mem_unit) >> 20;

	return sysfs_emit(buf,
		"=== GreenBoost " GB_VERSION " — Hardware Topology ===\n"
		"\n"
		"CPU  %s\n"
		"  Logical CPUs      : %d total\n"
		"  P-cores           : CPU 0-%d\n"
		"  Golden cores      : CPU %d-%d\n"
		"  E-cores           : CPU %d and up\n"
		"  Watchdog on E-cores: %s\n"
		"\n"
		"RAM  %llu MB total\n"
		"\n"
		"GPU  %d GB VRAM  (physical_vram_gb)\n"
		"\n"
		"NVMe  T3 backing file  %s  (t3_max_gb=%d GB)  [nvme_swap_gb=%d DEPRECATED]\n"
		"\n"
		"Motherboard  %s %s\n"
		"  NUMA nodes        : 1\n"
		"\n"
		"GreenBoost kthread affinity\n"
		"  ecores_only       : %d\n"
		"  pcores_max_cpu    : %d  (CPUs 0-%d = P-cores)\n"
		"  golden_cpu_min/max: %d / %d\n",
		/* MIN-05: x86_model_id may be empty on non-x86 or early-boot paths */
		boot_cpu_data.x86_model_id[0] ? boot_cpu_data.x86_model_id : "Unknown CPU",
		num_online_cpus(),
		pcores_max_cpu,
		golden_cpu_min, golden_cpu_max,
		pcores_max_cpu + 1,
		ecores_only ? "yes" : "no (all CPUs)",
		ram_total_mb,
		physical_vram_gb,
		gb_dev.t3_file ? t3_file_path : "disabled",
		t3_max_gb ? t3_max_gb : nvme_pool_gb,
		nvme_swap_gb,
		board_vendor ? board_vendor : "Unknown",
		board_name   ? board_name   : "",
		ecores_only, pcores_max_cpu, pcores_max_cpu,
		golden_cpu_min, golden_cpu_max);
}
static DEVICE_ATTR_RO(hw_info);

/* /sys/class/greenboost/greenboost/kv_reserve_mb — runtime KV cache reservation */
static ssize_t kv_reserve_mb_show(struct device *dev,
				   struct device_attribute *attr, char *buf)
{
	/* MED-03: READ_ONCE pairs with WRITE_ONCE in IOCTL and sysfs store */
	return sysfs_emit(buf, "%d\n", READ_ONCE(kv_reserve_mb));
}

static ssize_t kv_reserve_mb_store(struct device *dev,
				    struct device_attribute *attr,
				    const char *buf, size_t count)
{
	int val, max_kv, clamped;

	if (kstrtoint(buf, 10, &val) || val < 0)
		return -EINVAL;
	/* Same clamp as IOCTL path: never exceed T1 - 2 GB */
	max_kv  = physical_vram_gb * 1024 - 2048;
	if (max_kv < 512)
		max_kv = 512;
	clamped = (val > max_kv) ? max_kv : val;
	/* MED-03: WRITE_ONCE pairs with READ_ONCE in show/status — prevents torn write */
	WRITE_ONCE(kv_reserve_mb, clamped);
	pr_info(DRIVER_NAME
		": KV cache T1 reserve set to %d MB via sysfs%s\n",
		clamped,
		(clamped != val) ? " (clamped to T1 limit)" : "");
	return count;
}
static DEVICE_ATTR_RW(kv_reserve_mb);

/* /sys/class/greenboost/greenboost/active_profile — profile name set at insmod */
static ssize_t active_profile_show(struct device *dev,
				    struct device_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%s\n", active_profile_name);
}
static DEVICE_ATTR_RO(active_profile);

/* /sys/class/greenboost/greenboost/alloc_request_size — DRA allocation request size */
static atomic64_t dra_alloc_request_size = ATOMIC64_INIT(0);
static ssize_t alloc_request_size_show(struct device *dev,
					 struct device_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%llu\n", atomic64_read(&dra_alloc_request_size));
}
static ssize_t alloc_request_size_store(struct device *dev,
				         struct device_attribute *attr,
				         const char *buf, size_t count)
{
	u64 val;
	if (kstrtoull(buf, 10, &val))
		return -EINVAL;
	atomic64_set(&dra_alloc_request_size, val);
	return count;
}
static DEVICE_ATTR_RW(alloc_request_size);

/* /sys/class/greenboost/greenboost/alloc_request_flags — DRA allocation flags */
static atomic_t dra_alloc_flags = ATOMIC_INIT(0);
static ssize_t alloc_request_flags_show(struct device *dev,
					 struct device_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%u\n", atomic_read(&dra_alloc_flags));
}
static ssize_t alloc_request_flags_store(struct device *dev,
					  struct device_attribute *attr,
					  const char *buf, size_t count)
{
	int val;
	if (kstrtoint(buf, 10, &val))
		return -EINVAL;
	atomic_set(&dra_alloc_flags, val);
	return count;
}
static DEVICE_ATTR_RW(alloc_request_flags);

/* /sys/class/greenboost/greenboost/alloc_result_fd — DRA allocation result FD */
static ssize_t alloc_result_fd_show(struct device *dev,
				    struct device_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%d\n", atomic_read(&dra_result_fd));
}
static DEVICE_ATTR_RO(alloc_result_fd);

/* /sys/class/greenboost/greenboost/alloc_result_offset — DRA allocation result offset */
static ssize_t alloc_result_offset_show(struct device *dev,
					struct device_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%llu\n", atomic64_read(&dra_result_offset));
}
static DEVICE_ATTR_RO(alloc_result_offset);

/* /sys/class/greenboost/greenboost/alloc_trigger — DRA: write "1" to trigger allocation.
 * The kubelet plugin writes size to alloc_request_size, flags to alloc_request_flags,
 * then writes "1" here.  The result DMA-BUF fd is readable from alloc_result_fd.
 * Single-client only: only one allocation may be in-flight at a time. */
static DEFINE_MUTEX(dra_trigger_lock);
static ssize_t alloc_trigger_store(struct device *dev,
				    struct device_attribute *attr,
				    const char *buf, size_t count)
{
	struct gb_buf *gbuf;
	u64  req_size;
	u32  req_flags;
	int  fd;
	unsigned int j;

	if (!mutex_trylock(&dra_trigger_lock))
		return -EBUSY;

	req_size  = (u64)atomic64_read(&dra_alloc_request_size);
	req_flags = (u32)atomic_read(&dra_alloc_flags);

	if (req_size == 0) {
		mutex_unlock(&dra_trigger_lock);
		return -EINVAL;
	}

	gbuf = gb_alloc_buf((size_t)req_size, req_flags);
	if (IS_ERR(gbuf)) {
		atomic_set(&dra_result_fd, -1);
		mutex_unlock(&dra_trigger_lock);
		return PTR_ERR(gbuf);
	}

	/* gb_alloc_buf does not call dma_buf_export — mirror the IOCTL_ALLOC
	 * sequence: export pages as a DMA-BUF then install the fd. */
	{
		DEFINE_DMA_BUF_EXPORT_INFO(exp_info);
		exp_info.ops   = &gb_dma_buf_ops;	/* CRIT-01: was &gb_dmabuf_ops (typo) */
		exp_info.size  = gbuf->size;
		exp_info.flags = O_RDWR;
		exp_info.priv  = gbuf;

		gbuf->dmabuf = dma_buf_export(&exp_info);
		if (IS_ERR(gbuf->dmabuf)) {
			/* CRIT-02: Full cleanup mirrors GB_IOCTL_ALLOC error path.
			 * Previous code leaked pages and left a dangling LRU entry. */
			int err = PTR_ERR(gbuf->dmabuf);
			gbuf->dmabuf = NULL;
			atomic_dec(&gb_dev.active_bufs);
			if (gbuf->tier == GB_TIER3_NVME)
				atomic64_sub(gbuf->size, &gb_dev.nvme_allocated);
			else
				atomic64_sub(gbuf->size, &gb_dev.pool_allocated);
			if (gbuf->alloc_flags & GB_ALLOC_KV_CACHE) {
				atomic64_sub(gbuf->size, &gb_dev.kv_used_bytes);
				if (gbuf->tier != GB_TIER3_NVME)
					atomic64_sub(gbuf->size, &gb_dev.kv_t2_bytes);
			}
			spin_lock(&gb_dev.lru_lock);
			list_del_init(&gbuf->lru_node);
			spin_unlock(&gb_dev.lru_lock);
			if (gbuf->hugepages) {
				for (j = 0; j < gbuf->nhpages; j++)
					__free_pages(gbuf->hpages[j], GB_HPAGE_ORDER);
				kvfree(gbuf->hpages);
			} else {
				for (j = 0; j < gbuf->npages; j++)
					__free_page(gbuf->pages[j]);
				kvfree(gbuf->pages);
			}
			kfree(gbuf);
			atomic_set(&dra_result_fd, -1);
			mutex_unlock(&dra_trigger_lock);
			return err;
		}
	}

	fd = dma_buf_fd(gbuf->dmabuf, O_CLOEXEC);
	if (fd < 0) {
		/* dma_buf_put releases the export reference and triggers gb_release */
		dma_buf_put(gbuf->dmabuf);
		atomic_set(&dra_result_fd, -1);
		mutex_unlock(&dra_trigger_lock);
		return fd;
	}

	atomic_set(&dra_result_fd, fd);
	atomic64_set(&dra_result_offset, 0);
	mutex_unlock(&dra_trigger_lock);
	return count;
}
static DEVICE_ATTR_WO(alloc_trigger);

/* ENH-04: Forward declaration — gb_gpu_count_per_node is defined below after
 * nvlink_ready_store, but nvlink_ready_store now calls gb_nvlink_set_ready()
 * which needs it.  The forward decl avoids moving the module_param block. */
static int gb_gpu_count_per_node;

/* /sys/class/greenboost/greenboost/nvlink_ready — NVLink pooling status
 * Written by the kubelet plugin after NVML P2P verification (V100 approach).
 * Read by CUDA shim at init to report aggregated NVLink VRAM to cuDeviceTotalMem.
 * Declared at top of file (before gb_ioctl); show/store are defined here. */
static ssize_t nvlink_ready_show(struct device *dev,
				   struct device_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%d\n", atomic_read(&nvlink_ready));
}
static ssize_t nvlink_ready_store(struct device *dev,
				    struct device_attribute *attr,
				    const char *buf, size_t count)
{
	int val;

	if (kstrtoint(buf, 10, &val))
		return -EINVAL;
	atomic_set(&nvlink_ready, (val != 0) ? 1 : 0);
	/* ENH-04: Activate NVLink pool when kubelet sets ready=1.
	 * Without this call, fabric_ready stays false and gb_nvlink_is_active()
	 * always returns false — the pool is never used despite nvlink_ready=1. */
	gb_nvlink_set_ready(val != 0,
			    (u32)gb_gpu_count_per_node,
			    (u64)physical_vram_gb);
	pr_info(DRIVER_NAME ": nvlink_ready set to %d by kubelet plugin\n", val != 0 ? 1 : 0);
	return count;
}
static DEVICE_ATTR_RW(nvlink_ready);

/* /sys/class/greenboost/greenboost/gpu_count_per_node
 * Number of GPUs on this node participating in the NVLink pool.
 * Written by greenboost_setup.sh at insmod time; read by the CUDA shim to
 * compute NVLink aggregated VRAM instead of assuming 8 GPUs.
 * Default 0 = unknown (shim falls back to the /8 * 7 formula).
 */
/* gb_gpu_count_per_node declared above (forward decl for nvlink_ready_store) */
module_param(gb_gpu_count_per_node, int, 0644);
MODULE_PARM_DESC(gb_gpu_count_per_node,
	"GPUs in NVLink pool on this node (0 = unknown; shim falls back to /8*7)");

static ssize_t gpu_count_per_node_show(struct device *dev,
					struct device_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%d\n", gb_gpu_count_per_node);
}
static ssize_t gpu_count_per_node_store(struct device *dev,
					 struct device_attribute *attr,
					 const char *buf, size_t count)
{
	int val;
	if (kstrtoint(buf, 10, &val) || val < 0)
		return -EINVAL;
	gb_gpu_count_per_node = val;
	return count;
}
static DEVICE_ATTR_RW(gpu_count_per_node);

/* /sys/class/greenboost/greenboost/compute_domain_active — IMEX ComputeDomain status
 * Declared at top of file; show/store are defined here. */
static ssize_t compute_domain_active_show(struct device *dev,
					   struct device_attribute *attr, char *buf)
{
	return sysfs_emit(buf, "%d\n", atomic_read(&compute_domain_active));
}
static ssize_t compute_domain_active_store(struct device *dev,
				     struct device_attribute *attr,
				     const char *buf, size_t count)
{
	int val;
	if (kstrtoint(buf, 10, &val) || (val != 0 && val != 1))
		return -EINVAL;
	atomic_set(&compute_domain_active, val);
	pr_info(DRIVER_NAME ": ComputeDomain %sACTIVE\n", val ? "" : "IN");
	return count;
}
static DEVICE_ATTR_RW(compute_domain_active);

/* /sys/class/greenboost/greenboost/pool_brief
 * Compact one-liner: "T1:12GB T2:8/51GB(15%) T3:0/128GB PRESSURE:ok KV:2048MB"
 * Useful for: watch -n1 cat pool_brief  /  Waybar status  /  scripted polling */
static ssize_t pool_brief_show(struct device *dev,
			       struct device_attribute *attr, char *buf)
{
	struct sysinfo si;
	u64 t2_alloc_mb, t2_pct;
	u64 t3_alloc_mb, t3_max_mb, kv_t2_mb;
	int pressure;
	const char *pressure_str;

	si_meminfo(&si);
	gb_get_t3_stats(&t3_alloc_mb, &t3_max_mb);

	t2_alloc_mb = (u64)atomic64_read(&gb_dev.pool_allocated) >> 20;
	kv_t2_mb    = (u64)atomic64_read(&gb_dev.kv_t2_bytes) >> 20;
	t2_pct = (t2_alloc_mb * 100ULL) /
		 (((u64)virtual_vram_gb * 1024ULL) != 0 ? (u64)virtual_vram_gb * 1024ULL : 1ULL);
	pressure = atomic_read(&gb_dev.swap_pressure);
	pressure_str = (pressure == GB_SWAP_PRESSURE_CRITICAL) ? "CRITICAL" :
		       (pressure == GB_SWAP_PRESSURE_WARN)     ? "warn"     :
		                                                  "ok";

	return sysfs_emit(buf,
		"T1:%dGB T2:%llu/%dGB(%llu%%) T3:%llu/%lluGB PRESSURE:%s KV_RSV:%dMB KV_T2:%lluMB\n",
		physical_vram_gb,
		t2_alloc_mb / 1024ULL, virtual_vram_gb, t2_pct,
		t3_alloc_mb / 1024ULL, t3_max_mb / 1024ULL,
		pressure_str,
		kv_reserve_mb,
		kv_t2_mb);
}
static DEVICE_ATTR_RO(pool_brief);

static struct attribute *gb_attrs[] = {
	&dev_attr_status.attr,
	&dev_attr_hw_info.attr,
	&dev_attr_active_buffers.attr,
	&dev_attr_active_profile.attr,
	&dev_attr_kv_reserve_mb.attr,
	/* DRA integration attributes */
	&dev_attr_alloc_request_size.attr,
	&dev_attr_alloc_request_flags.attr,
	&dev_attr_alloc_result_fd.attr,
	&dev_attr_alloc_result_offset.attr,
	&dev_attr_alloc_trigger.attr,
	&dev_attr_nvlink_ready.attr,
	&dev_attr_gpu_count_per_node.attr,
	&dev_attr_compute_domain_active.attr,
	&dev_attr_pool_brief.attr,
	NULL,
};
ATTRIBUTE_GROUPS(gb);

/* ------------------------------------------------------------------ */
/*  T2 auto-eviction — evict COLD weight buffers to T3 under pressure  */
/* ------------------------------------------------------------------ */

/**
 * gb_auto_evict_cold() - Walk LRU tail→head and soft-evict cold T2 buffers.
 *
 * Called by the watchdog when T2 pool hits CRITICAL (>90%).  Only evicts
 * non-frozen, non-T1-priority, non-KV-cache weight buffers — the coldest
 * pages that the inference engine is least likely to access next.  KV cache
 * buffers are never evicted: they are read+written every generation step and
 * must stay in T2 (32 GB/s) to avoid tok/s collapse on T3 (1.8 GB/s NVMe).
 *
 * Eviction is now two-phase:
 *   Phase 1 (under lru_lock): select candidates, move accounting T2→T3,
 *             remove from LRU, stage in local evict_list.
 *   Phase 2 (lock released): call gb_t3_evict_buf() which does file I/O and
 *             frees RAM.  File I/O can sleep, so it must happen outside the
 *             spinlock.  On failure the buffer is moved back to T2.
 *
 * @target_free_bytes: stop after freeing this many bytes from T2 accounting.
 * Returns the number of buffers successfully evicted to disk.
 */
static int gb_auto_evict_cold(u64 target_free_bytes)
{
	struct gb_buf *buf, *tmp;
	LIST_HEAD(evict_list);
	u64 freed = 0;
	int staged = 0, evicted = 0;

	if (!gb_dev.t3_file)
		return 0;

	/* Phase 1: select candidates and move accounting under the spinlock */
	spin_lock(&gb_dev.lru_lock);
	list_for_each_entry_safe_reverse(buf, tmp, &gb_dev.lru_list, lru_node) {
		if (freed >= target_free_bytes)
			break;

		/* Never touch KV cache — stays in T2 to preserve tok/s */
		if (buf->alloc_flags & GB_ALLOC_KV_CACHE)
			continue;
		/* Never touch frozen or T1-priority buffers */
		if (buf->frozen || buf->t1_priority)
			continue;
		/* Skip session-protected buffers on first pass (evict unprotected first) */
		if (buf->session_priority >= 1)
			continue;
		/* Only evict T2 4K-page buffers (hugepages are pinned) */
		if (buf->tier != GB_TIER2_SDDR || buf->hugepages)
			continue;

		/* Move accounting T2 → T3 */
		atomic64_sub(buf->size, &gb_dev.pool_allocated);
		atomic64_add(buf->size, &gb_dev.nvme_allocated);
		buf->tier = GB_TIER3_NVME;
		list_del_init(&buf->lru_node);
		/* Stage on local list for Phase 2 (lru_node is free now) */
		list_add_tail(&buf->lru_node, &evict_list);

		freed += buf->size;
		staged++;
	}
	/* Second pass: if target not yet met, allow session-protected buffers */
	if (freed < target_free_bytes) {
		list_for_each_entry_safe_reverse(buf, tmp, &gb_dev.lru_list, lru_node) {
			if (freed >= target_free_bytes)
				break;
			if (buf->alloc_flags & GB_ALLOC_KV_CACHE)
				continue;
			if (buf->frozen || buf->t1_priority)
				continue;
			if (buf->tier != GB_TIER2_SDDR || buf->hugepages)
				continue;

			atomic64_sub(buf->size, &gb_dev.pool_allocated);
			atomic64_add(buf->size, &gb_dev.nvme_allocated);
			buf->tier = GB_TIER3_NVME;
			list_del_init(&buf->lru_node);
			list_add_tail(&buf->lru_node, &evict_list);
			freed += buf->size;
			staged++;
		}
	}
	spin_unlock(&gb_dev.lru_lock);

	if (staged == 0)
		return 0;

	/* Phase 2: write pages to disk and free RAM (sleepable, no locks) */
	list_for_each_entry_safe(buf, tmp, &evict_list, lru_node) {
		list_del_init(&buf->lru_node);

		if (gb_t3_evict_buf(buf) == 0) {
			evicted++;
		} else {
			/* Write failed — roll accounting back to T2 */
			atomic64_sub(buf->size, &gb_dev.nvme_allocated);
			atomic64_add(buf->size, &gb_dev.pool_allocated);
			buf->tier = GB_TIER2_SDDR;
			spin_lock(&gb_dev.lru_lock);
			list_add_tail(&buf->lru_node, &gb_dev.lru_list);
			spin_unlock(&gb_dev.lru_lock);
		}
	}

	if (evicted > 0)
		pr_warn(DRIVER_NAME
			": T2 auto-evict: %d buf(s) → T3 file (%lluMB freed)\n",
			evicted, freed >> 20);
	return evicted;
}

/* ------------------------------------------------------------------ */
/*  PID-based buffer release                                           */
/* ------------------------------------------------------------------ */

/*
 * gb_release_pid_buffers - drop all T2/T3 kernel buffers owned by @pid.
 *
 * Two-phase: collect dma_buf pointers under gb_dev.lock (calling get_dma_buf
 * on each to hold the ref), then dma_buf_put() outside the lock.  This avoids
 * the reentrancy deadlock where dma_buf_put → gb_release → mutex_lock(&gb_dev.lock).
 *
 * Returns the count of buffers released, or -ENOMEM.
 */
static int gb_release_pid_buffers(pid_t pid)
{
	struct gb_buf *buf;
	struct dma_buf **refs;
	int id, n = 0, cap, i;

	cap = atomic_read(&gb_dev.active_bufs) + 1;
	refs = kvmalloc_array(cap, sizeof(*refs), GFP_KERNEL);
	if (!refs)
		return -ENOMEM;

	/* Phase 1: collect matching DMA-BUF refs while holding the IDR lock */
	mutex_lock(&gb_dev.lock);
	idr_for_each_entry(&gb_dev.idr, buf, id) {
		if (buf->owner_pid == pid && n < cap) {
			get_dma_buf(buf->dmabuf);
			refs[n++] = buf->dmabuf;
		}
	}
	mutex_unlock(&gb_dev.lock);

	/* Phase 2: release outside lock — triggers gb_release when refcount → 0 */
	for (i = 0; i < n; i++)
		dma_buf_put(refs[i]);

	kvfree(refs);
	if (n > 0)
		pr_info(DRIVER_NAME ": released %d buffer(s) for PID %d\n", n, pid);
	return n;
}

/* ------------------------------------------------------------------ */
/*  Watchdog kthread — enforces safety reserve                         */
/* ------------------------------------------------------------------ */

/*
 * gb_reap_dead_pids - scan IDR for buffers whose owner PID no longer exists
 * and free them via gb_release_pid_buffers().
 *
 * Collects up to 64 unique PIDs, checks liveness with find_get_pid() +
 * pid_task() (PID-recycling-safe because struct pid* is ref-counted), then
 * calls gb_release_pid_buffers for each dead PID.
 */
static void gb_reap_dead_pids(void)
{
	struct gb_buf *buf;
	pid_t pids[64];
	int np = 0, id, i;

	mutex_lock(&gb_dev.lock);
	idr_for_each_entry(&gb_dev.idr, buf, id) {
		pid_t p = buf->owner_pid;
		bool known = false;

		for (i = 0; i < np; i++) {
			if (pids[i] == p) { known = true; break; }
		}
		if (!known && np < 64)
			pids[np++] = p;
	}
	mutex_unlock(&gb_dev.lock);

	for (i = 0; i < np; i++) {
		struct pid *pp = find_get_pid(pids[i]);
		bool alive = pp && pid_task(pp, PIDTYPE_PID);

		if (pp)
			put_pid(pp);
		if (!alive) {
			pr_info(DRIVER_NAME ": PID %d gone — reaping orphaned buffers\n",
				pids[i]);
			gb_release_pid_buffers(pids[i]);
		}
	}
}

static int gb_watchdog(void *unused)
{
	u64 free_bytes, reserve_bytes;
	u64 swap_used_pct;
	int new_pressure, old_pressure;
	unsigned int watchdog_cycle = 0;

	pr_info(DRIVER_NAME ": watchdog started (500ms, T2 RAM + T3 NVMe)\n");

	while (!kthread_should_stop()) {
		msleep_interruptible(500);

		/* ── Dead-PID reaper ─────────────────────────────────── */
		/* Fires every idle_cleanup_sec seconds (0 = disabled).
		 * Safety net for SIGKILL paths where gb_close did not run
		 * (e.g., process killed before opening /dev/greenboost). */
		watchdog_cycle++;
		if (idle_cleanup_sec > 0 &&
		    watchdog_cycle >= (idle_cleanup_sec * 1000U / 500U)) {
			watchdog_cycle = 0;
			if (atomic_read(&gb_dev.active_bufs) > 0)
				gb_reap_dead_pids();
		}

		/* ── Tier 2: RAM safety reserve ─────────────────────── */
		/* si_mem_available() = MemAvailable: includes reclaimable page
		 * cache and slab.  Using raw si.freeram (MemFree) caused false
		 * OOM guard trips on workstations with large page caches. */
		free_bytes    = (u64)si_mem_available() * PAGE_SIZE;
		reserve_bytes = (u64)safety_reserve_gb * (1ULL << 30);

		/*
		 * Two-tier RAM pressure response:
		 *
		 * WARN  (free < reserve * 1.25): pre-emptive 15% T2 eviction.
		 *   Fires ~25% of safety_reserve_gb above the guard threshold,
		 *   giving the eviction time to work before OOM killer fires.
		 *
		 * CRITICAL (free < reserve): aggressive 25% T2 eviction + OOM guard.
		 *   If WARN eviction wasn't enough, escalate immediately.
		 *
		 * Both thresholds are expressed as fractions of the configured
		 * safety_reserve_gb — no hard-coded byte values.
		 */
		if (free_bytes < reserve_bytes) {
			if (!atomic_read(&gb_dev.oom_active)) {
				atomic_set(&gb_dev.oom_active, 1);
				pr_warn(DRIVER_NAME
					": T2 OOM guard TRIPPED — "
					"avail=%lluMB < reserve=%dGB; evicting 25%% T2\n",
					free_bytes >> 20, safety_reserve_gb);
			}
			/* Aggressive eviction: free 25% of T2 pool to restore headroom */
			gb_auto_evict_cold((u64)virtual_vram_gb * (1ULL << 30) / 4);
		} else if (free_bytes < reserve_bytes + (reserve_bytes / 4)) {
			/* WARN: approaching reserve — pre-emptive 15% eviction, no OOM flag */
			pr_info(DRIVER_NAME
				": T2 RAM pressure warn — avail=%lluMB; pre-evicting cold bufs\n",
				free_bytes >> 20);
			gb_auto_evict_cold((u64)virtual_vram_gb * (1ULL << 30) * 15 / 100);
		} else {
			if (atomic_read(&gb_dev.oom_active)) {
				atomic_set(&gb_dev.oom_active, 0);
				pr_info(DRIVER_NAME
					": T2 OOM guard cleared — "
					"avail=%lluMB\n",
					free_bytes >> 20);
			}
		}

		/* ── Tier 2: DDR pool pressure ──────────────────────── */
		{
			u64 t2_alloc = (u64)atomic64_read(&gb_dev.pool_allocated);
			u64 t2_cap   = (u64)virtual_vram_gb * (1ULL << 30);
			u64 t2_pct   = (t2_cap > 0)
					? (t2_alloc * 100ULL) / t2_cap : 0;
			int new_t2, old_t2;

			if (t2_pct >= 90)
				new_t2 = GB_T2_PRESSURE_CRITICAL;
			else if (t2_pct >= 75)
				new_t2 = GB_T2_PRESSURE_WARN;
			else
				new_t2 = GB_T2_PRESSURE_OK;

			old_t2 = atomic_read(&gb_dev.t2_pressure);
			if (new_t2 != old_t2) {
				atomic_set(&gb_dev.t2_pressure, new_t2);
				/* Signal userspace (same eventfd as T3 — caller re-reads
				 * full gb_info to distinguish which tier changed) */
				{
					struct eventfd_ctx *efd;
					unsigned long _flags;
					spin_lock_irqsave(&gb_dev.efd_lock, _flags);
					efd = gb_dev.pressure_efd;
					spin_unlock_irqrestore(&gb_dev.efd_lock, _flags);
					if (efd)
						eventfd_signal(efd);
				}
				if (new_t2 == GB_T2_PRESSURE_CRITICAL)
					pr_warn(DRIVER_NAME
						": T2 DDR pool CRITICAL — "
						"%llu%% used (%lluMB/%lluMB) "
						"— auto-evicting cold bufs\n",
						t2_pct, t2_alloc >> 20,
						t2_cap >> 20);
				else if (new_t2 == GB_T2_PRESSURE_WARN)
					pr_warn(DRIVER_NAME
						": T2 DDR pool warn — "
						"%llu%% used\n", t2_pct);
				else
					pr_info(DRIVER_NAME
						": T2 DDR pool pressure cleared\n");
			}

			/* Auto-evict COLD weight buffers when T2 is critical.
			 * Target: free 10% of pool capacity per watchdog cycle.
			 * KV cache, frozen, and T1-priority buffers are never touched. */
			if (new_t2 == GB_T2_PRESSURE_CRITICAL)
				gb_auto_evict_cold(t2_cap / 10);
		}

		/* ── Tier 3: backing-file pressure ──────────────────── */
		if (gb_dev.t3_file && gb_dev.t3_file_max > 0) {
			u64 t3_used_mb, t3_max_mb;

			gb_get_t3_stats(&t3_used_mb, &t3_max_mb);
			if (t3_max_mb > 0) {
				swap_used_pct = t3_used_mb * 100ULL / t3_max_mb;
				if (swap_used_pct >= 90)
					new_pressure = GB_SWAP_PRESSURE_CRITICAL;
				else if (swap_used_pct >= 75)
					new_pressure = GB_SWAP_PRESSURE_WARN;
				else
					new_pressure = GB_SWAP_PRESSURE_OK;

				old_pressure = atomic_read(&gb_dev.swap_pressure);
				if (new_pressure != old_pressure) {
					atomic_set(&gb_dev.swap_pressure, new_pressure);
					{
						struct eventfd_ctx *efd;
						unsigned long _flags;
						spin_lock_irqsave(&gb_dev.efd_lock, _flags);
						efd = gb_dev.pressure_efd;
						spin_unlock_irqrestore(&gb_dev.efd_lock, _flags);
						if (efd)
							eventfd_signal(efd);
					}
					if (new_pressure == GB_SWAP_PRESSURE_CRITICAL)
						pr_warn(DRIVER_NAME
							": T3 file CRITICAL — "
							"%llu%% used (%lluMB/%lluMB)\n",
							swap_used_pct,
							t3_used_mb, t3_max_mb);
					else if (new_pressure == GB_SWAP_PRESSURE_WARN)
						pr_warn(DRIVER_NAME
							": T3 file warn — "
							"%llu%% used\n", swap_used_pct);
					else
						pr_info(DRIVER_NAME
							": T3 file pressure cleared\n");
				}
			}
		}
	}

	pr_info(DRIVER_NAME ": watchdog stopped\n");
	return 0;
}

/* ------------------------------------------------------------------ */
/*  File operations                                                     */
/* ------------------------------------------------------------------ */

static int gb_open(struct inode *inode, struct file *file)
{
	gb_dbg("opened\n");
	return 0;
}

static int gb_close(struct inode *inode, struct file *file)
{
	pid_t pid = task_pid_vnr(current);

	gb_dbg("closed by PID %d\n", pid);
	/* Free all T2/T3 kernel buffers owned by this PID.  Handles SIGKILL
	 * (destructor never runs) because the kernel closes all fds on process
	 * death, which fires this .release handler before the mm is torn down. */
	gb_release_pid_buffers(pid);
	return 0;
}

static const struct file_operations gb_fops = {
	.owner          = THIS_MODULE,
	.open           = gb_open,
	.release        = gb_close,
	.unlocked_ioctl = gb_ioctl,
	/* ENH-01: 32-bit CUDA processes (Wine/Proton via nvcuda.dll PE/ELF bridge)
	 * need compat_ioctl so they don't receive ENOTTY.
	 * All IOCTL structs use gb_u64/gb_u32 (fixed-width typedef, no native
	 * pointer or unsigned long fields) so the 32-bit and 64-bit layouts are
	 * identical — no argument translation required; reuse gb_ioctl directly. */
	.compat_ioctl   = gb_ioctl,
};

/* ------------------------------------------------------------------ */
/*  Reboot + panic notifiers                                            */
/* ------------------------------------------------------------------ */

/*
 * gb_reboot_notify — called by the kernel on soft reboot/halt/power-off.
 *
 * Stops the watchdog and releases the T2/T3 pool before the system goes down,
 * so pinned DDR pages are freed and the NVMe swap file is cleanly unmounted.
 * A hard reset (physical power button) bypasses this path entirely — recovery
 * is handled by greenboost-recovery.service on the next boot.
 *
 * The teardown_done guard prevents double-teardown if gb_exit() is also called
 * on some shutdown paths after the notifier fires.
 */
static int gb_reboot_notify(struct notifier_block *nb,
			    unsigned long action, void *data)
{
	if (action != SYS_RESTART && action != SYS_HALT && action != SYS_POWER_OFF)
		return NOTIFY_DONE;

	if (atomic_cmpxchg(&gb_dev.teardown_done, 0, 1) != 0) {
		pr_info(DRIVER_NAME ": reboot notifier — teardown already done, skipping\n");
		return NOTIFY_DONE;
	}

	pr_info(DRIVER_NAME ": reboot notifier (action=%lu) — stopping watchdog + releasing T2/T3\n",
		action);

	if (gb_dev.watchdog) {
		kthread_stop(gb_dev.watchdog);
		gb_dev.watchdog = NULL;
	}

	/* Drain async eviction work before reboot so NVMe writes don't race
	 * with filesystem unmount.  destroy_workqueue is left to gb_exit(). */
	if (gb_dev.evict_wq)
		flush_workqueue(gb_dev.evict_wq);

	gb_nvlink_pool_exit();
	atomic_set(&gb_dev.oom_active, 0);

	pr_info(DRIVER_NAME ": reboot notifier — T2/T3 pool released cleanly\n");
	return NOTIFY_DONE;
}

static struct notifier_block gb_reboot_nb = {
	.notifier_call = gb_reboot_notify,
	.priority      = 0,
};

/*
 * gb_panic_notify — called from the kernel panic path (atomic context).
 *
 * Constraints: NO sleeping, NO memory allocation, NO mutex.
 * Safe operations: atomic_set(), spin_trylock(), pr_emerg().
 *
 * We poison oom_active so the recovery service on next boot knows to run
 * the full repair sequence.  The LRU list is cleared (best-effort spinlock)
 * so gb_auto_evict_cold() is a no-op if somehow invoked after panic.
 */
static int gb_panic_notify(struct notifier_block *nb,
			   unsigned long action, void *data)
{
	atomic_set(&gb_dev.oom_active, 1);
	if (spin_trylock(&gb_dev.lru_lock)) {
		INIT_LIST_HEAD(&gb_dev.lru_list);
		spin_unlock(&gb_dev.lru_lock);
	}
	pr_emerg(DRIVER_NAME ": kernel panic — state poisoned; run greenboost-recovery on next boot\n");
	return NOTIFY_DONE;
}

static struct notifier_block gb_panic_nb = {
	.notifier_call = gb_panic_notify,
	.priority      = INT_MAX,   /* fire early in the panic chain */
};

/* ------------------------------------------------------------------ */
/*  Module init / exit                                                  */
/* ------------------------------------------------------------------ */

static int __init gb_init(void)
{
	int ret;

	/* Auto-detect T2 pool cap when virtual_vram_gb == 0 (no modprobe.conf or
	 * explicit override). Adaptive DDR limit:
	 *   total RAM <  64 GB  ->  70% of total RAM
	 *   total RAM >= 64 GB  ->  80% of total RAM
	 * safety_reserve_gb is enforced at alloc time by gb_alloc_buf(). */
	if (virtual_vram_gb == 0) {
		struct sysinfo _si;
		u64 total_gb;

		si_meminfo(&_si);
		total_gb = ((u64)_si.totalram * _si.mem_unit) >> 30;

		if (total_gb >= 64)
			virtual_vram_gb = (int)(total_gb * 80 / 100);
		else
			virtual_vram_gb = (int)(total_gb * 70 / 100);

		if (virtual_vram_gb < 4)
			virtual_vram_gb = 4;

		pr_info(DRIVER_NAME ": T2 auto-cap: %llu GB RAM -> %d GB pool (%d%%)\n",
			total_gb, virtual_vram_gb, total_gb >= 64 ? 80 : 70);
	}

	pr_info(DRIVER_NAME ": =====================================================\n");
	pr_info(DRIVER_NAME ": GreenBoost " GB_VERSION " — CUDA Memory GPU Memory Pool (LRU+eventfd)\n");
	pr_info(DRIVER_NAME ": Author  : Ferran Duarri\n");
	pr_info(DRIVER_NAME ": CPU     : %s — P-cores CPU 0-%d (golden %d-%d)\n",
		boot_cpu_data.x86_model_id, pcores_max_cpu, golden_cpu_min, golden_cpu_max);
	pr_info(DRIVER_NAME ": T1 VRAM : %d GB\n",
		physical_vram_gb);
	pr_info(DRIVER_NAME ": T2 RAM  : pool cap       %d GB  (reserve %d GB)\n",
		virtual_vram_gb, safety_reserve_gb);
	pr_info(DRIVER_NAME ": T3 file : %s  (cap: %s)\n",
		t3_file_path,
		(t3_max_gb > 0 || nvme_pool_gb > 0) ? "configured" : "disk-limited");
	pr_info(DRIVER_NAME ": Combined: %d GB + T3 file\n",
		physical_vram_gb + virtual_vram_gb);
	pr_info(DRIVER_NAME ": KV reserve: %d MB of T1 reserved for KV cache\n",
		kv_reserve_mb);
	pr_info(DRIVER_NAME ": =====================================================\n");

	mutex_init(&gb_dev.lock);
	idr_init(&gb_dev.idr);
	atomic_set(&gb_dev.active_bufs, 0);
	atomic64_set(&gb_dev.pool_allocated, 0);
	atomic64_set(&gb_dev.nvme_allocated, 0);
	atomic64_set(&gb_dev.kv_used_bytes, 0);
	atomic_set(&gb_dev.oom_active, 0);
	atomic_set(&gb_dev.swap_pressure, GB_SWAP_PRESSURE_OK);
	atomic_set(&gb_dev.t2_pressure, GB_T2_PRESSURE_OK);
	INIT_LIST_HEAD(&gb_dev.lru_list);
	spin_lock_init(&gb_dev.lru_lock);
	spin_lock_init(&gb_dev.efd_lock);
	spin_lock_init(&gb_dev.t3_file_lock);
	gb_dev.pressure_efd = NULL;
	gb_dev.t3_file      = NULL;
	atomic64_set(&gb_dev.t3_next_offset, 0);

	/* Async T3 eviction workqueue — prevents ioctl threads from blocking on
	 * NVMe writes.  alloc_ordered_workqueue: sequential execution avoids
	 * write ordering races in the T3 backing file.  WQ_MEM_RECLAIM: kernel
	 * guarantees a rescue thread even under low-memory conditions. */
	gb_dev.evict_wq = alloc_ordered_workqueue("gb_evict", WQ_MEM_RECLAIM);
	if (!gb_dev.evict_wq) {
		pr_err(DRIVER_NAME ": failed to create evict workqueue\n");
		return -ENOMEM;
	}

	/* Compute T3 file cap: t3_max_gb wins, then nvme_pool_gb as alias */
	if (t3_max_gb > 0)
		gb_dev.t3_file_max = (u64)t3_max_gb << 30;
	else if (nvme_pool_gb > 0)
		gb_dev.t3_file_max = (u64)nvme_pool_gb << 30;
	else
		gb_dev.t3_file_max = 0; /* disk-limited */

	/* Open T3 backing file — non-fatal; T3 disabled if file unavailable */
	gb_t3_file_open();

	/* Allocate character device region */
	ret = alloc_chrdev_region(&gb_dev.devt, 0, 1, DRIVER_NAME);
	if (ret) {
		pr_err(DRIVER_NAME ": alloc_chrdev_region failed: %d\n", ret);
		return ret;
	}

	cdev_init(&gb_dev.cdev, &gb_fops);
	gb_dev.cdev.owner = THIS_MODULE;

	ret = cdev_add(&gb_dev.cdev, gb_dev.devt, 1);
	if (ret) {
		pr_err(DRIVER_NAME ": cdev_add failed: %d\n", ret);
		goto err_chrdev;
	}

	gb_dev.cls = class_create(CLASS_NAME);
	if (IS_ERR(gb_dev.cls)) {
		ret = PTR_ERR(gb_dev.cls);
		pr_err(DRIVER_NAME ": class_create failed: %d\n", ret);
		goto err_cdev;
	}

	gb_dev.dev = device_create_with_groups(gb_dev.cls, NULL,
					       gb_dev.devt, NULL,
					       gb_groups,
					       DEVICE_NAME);
	if (IS_ERR(gb_dev.dev)) {
		ret = PTR_ERR(gb_dev.dev);
		pr_err(DRIVER_NAME ": device_create failed: %d\n", ret);
		goto err_class;
	}

	/* Start watchdog kthread */
	gb_dev.watchdog = kthread_run(gb_watchdog, NULL, "gb_watchdog");
	if (IS_ERR(gb_dev.watchdog)) {
		ret = PTR_ERR(gb_dev.watchdog);
		pr_err(DRIVER_NAME ": kthread_run failed: %d\n", ret);
		gb_dev.watchdog = NULL;
		goto err_device;
	}

	/* Pin watchdog to E-cores to avoid stealing cycles from P-cores.
	 * Only active when ecores_only=1 and pcores_max_cpu >= 0 (hybrid CPUs).
	 * Auto-detected and configured by greenboost_setup.sh at load time.
	 */
	if (ecores_only) {
		cpumask_var_t emask;
		int cpu;

		if (alloc_cpumask_var(&emask, GFP_KERNEL)) {
			cpumask_clear(emask);
			for (cpu = pcores_max_cpu + 1; cpu < nr_cpu_ids; cpu++) {
				if (cpu_online(cpu))
					cpumask_set_cpu(cpu, emask);
			}
			/* Fallback to all CPUs if no E-cores found */
			if (cpumask_empty(emask)) {
				for (cpu = 0; cpu < nr_cpu_ids; cpu++) {
					if (cpu_online(cpu))
						cpumask_set_cpu(cpu, emask);
				}
			}
			set_cpus_allowed_ptr(gb_dev.watchdog, emask);
			free_cpumask_var(emask);
			pr_info(DRIVER_NAME
				": watchdog pinned away from P-cores (to E-cores %d-%d) "
				"to reserve golden P-cores (%d-%d) for inference\n",
				pcores_max_cpu + 1, nr_cpu_ids - 1,
				golden_cpu_min, golden_cpu_max);
		}
	}

	/* Initialize NVLink pooling subsystem (BUG-008).
	 * Returns 0 even when fabric not yet ready (BUG-009 fix). */
	gb_nvlink_pool_init();

	/* Register reboot + panic notifiers for graceful T2/T3 teardown on
	 * soft shutdown and state-poisoning on kernel panic.  Hard resets
	 * bypass both paths; recovery is handled by greenboost-recovery.service
	 * on the next boot. */
	register_reboot_notifier(&gb_reboot_nb);
	atomic_notifier_chain_register(&panic_notifier_list, &gb_panic_nb);
	pr_info(DRIVER_NAME ": reboot + panic notifiers registered\n");

	pr_info(DRIVER_NAME ": ready — /dev/%s\n", DEVICE_NAME);
	pr_info(DRIVER_NAME ": pool info: cat /sys/class/%s/%s/status\n",
		CLASS_NAME, DEVICE_NAME);
	return 0;

err_device:
	device_destroy(gb_dev.cls, gb_dev.devt);
err_class:
	class_destroy(gb_dev.cls);
err_cdev:
	cdev_del(&gb_dev.cdev);
err_chrdev:
	unregister_chrdev_region(gb_dev.devt, 1);
	return ret;
}

static void __exit gb_exit(void)
{
	pr_info(DRIVER_NAME ": unloading GreenBoost\n");

	/* Always unregister notifiers first, regardless of teardown path. */
	unregister_reboot_notifier(&gb_reboot_nb);
	atomic_notifier_chain_unregister(&panic_notifier_list, &gb_panic_nb);

	/* If the reboot notifier already ran (soft shutdown + rmmod race),
	 * skip pool teardown to avoid double-free of pinned T2 pages. */
	if (atomic_cmpxchg(&gb_dev.teardown_done, 0, 1) == 0) {
		gb_nvlink_pool_exit();

		if (gb_dev.watchdog) {
			kthread_stop(gb_dev.watchdog);
			gb_dev.watchdog = NULL;
		}

		/* Flush and destroy the async eviction workqueue before closing the
		 * T3 file — ensures all in-flight NVMe writes complete cleanly. */
		if (gb_dev.evict_wq) {
			flush_workqueue(gb_dev.evict_wq);
			destroy_workqueue(gb_dev.evict_wq);
			gb_dev.evict_wq = NULL;
		}

		/* Close T3 backing file after watchdog stops (no more evictions) */
		gb_t3_file_close();
	} else {
		pr_info(DRIVER_NAME ": pool already torn down by reboot notifier — skipping\n");
	}

	if (gb_dev.pressure_efd) {
		eventfd_ctx_put(gb_dev.pressure_efd);
		gb_dev.pressure_efd = NULL;
	}

	device_destroy(gb_dev.cls, gb_dev.devt);
	class_destroy(gb_dev.cls);
	cdev_del(&gb_dev.cdev);
	unregister_chrdev_region(gb_dev.devt, 1);
	idr_destroy(&gb_dev.idr);

	pr_info(DRIVER_NAME ": unloaded cleanly\n");
}

module_init(gb_init);
module_exit(gb_exit);

/* BUG-008 / MED-09: include NVLink pool implementation directly to avoid the
 * Kbuild circular dependency (greenboost.o cannot appear in both obj-m and
 * greenboost-y).  Proper fix: rename greenboost.c → greenboost_main.c and use
 * "greenboost-y := greenboost_main.o features/nvlink_pool.o" in Kbuild.
 * Until then: DO NOT also list features/nvlink_pool.o in Kbuild — that would
 * compile it twice, producing duplicate MODULE_PARM_DESC symbol errors. */
#include "features/nvlink_pool.c"

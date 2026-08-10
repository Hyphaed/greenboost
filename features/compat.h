#ifndef GREENBOOST_COMPAT_H
#define GREENBOOST_COMPAT_H
#include <linux/version.h>

/* dma_buf_set_priority(): out-of-tree RFC hint, not gated by kernel version
 * (see Kbuild's GB_HAS_DMABUF_PRIORITY probe). Fall back to 0 (absent) for
 * any build path that doesn't define it. */
#ifndef GB_HAS_DMABUF_PRIORITY
# define GB_HAS_DMABUF_PRIORITY 0
#endif

/* dma_buf_set_compression(): sibling out-of-tree RFC hint (see Kbuild's
 * GB_HAS_DMABUF_COMPRESSION probe). Fall back to 0 (absent) for any build
 * path that doesn't define it — same reasoning as GB_HAS_DMABUF_PRIORITY
 * above. */
#ifndef GB_HAS_DMABUF_COMPRESSION
# define GB_HAS_DMABUF_COMPRESSION 0
#endif

/* kretprobe: entry_handler field added in 5.11 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 11, 0)
# define GB_KRETPROBE_HAS_ENTRY  0
#else
# define GB_KRETPROBE_HAS_ENTRY  1
#endif

/* dma_buf_map_attachment: gained dma_dir param in 5.10.
 * F-L2-15: this macro is intentionally dormant - greenboost.c uses
 * dma_map_sgtable which has a stable API. Kept here as a reminder if
 * a future path resurrects dma_buf_map_attachment (e.g. RDMA DMA-BUF). */
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 10, 0)
# define GB_DMABUF_MAP_NO_DEVICE 1
#else
# define GB_DMABUF_MAP_NO_DEVICE 0
#endif

/* class_create: lost owner param in 6.4 */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 4, 0)
# define GB_CLASS_CREATE(name)   class_create(name)
#else
# define GB_CLASS_CREATE(name)   class_create(THIS_MODULE, name)
#endif

/* timer_setup: replaced setup_timer in 4.15 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(4, 15, 0)
# define GB_TIMER_INIT(t, fn)    setup_timer(t, fn, (unsigned long)(t))
#else
# define GB_TIMER_INIT(t, fn)    timer_setup(t, fn, 0)
#endif

/* kallsyms_lookup_name: removed from export in 5.7 - use kprobe workaround */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 7, 0)
# define GB_NEED_KPROBE_KALLSYMS 1
#else
# define GB_NEED_KPROBE_KALLSYMS 0
#endif

/* pin_user_pages() added in 5.6; fall back to get_user_pages() on older kernels.
 * The vmas parameter was dropped from pin_user_pages() in 5.9; the wrapper below
 * always presents the 4-argument form used by modern callers, passing NULL for
 * vmas internally on kernels that still require it. */
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 6, 0)
# define GB_NO_PIN_USER_PAGES
static inline long gb_pin_user_pages(unsigned long start, unsigned long nr_pages,
                                      unsigned int gup_flags, struct page **pages)
{
    return get_user_pages(start, nr_pages, gup_flags, pages, NULL);
}
#else
# define gb_pin_user_pages pin_user_pages
#endif

/* PR-X: eventfd_signal signature changed at 6.8.
 *   <= 6.7:  eventfd_signal(ctx, n)         - caller specifies counter delta
 *   >= 6.8:  eventfd_signal(ctx)            - always +1
 * Pick the right one with a compat macro so the same source builds on
 * both ranges.  GreenBoost only ever increments by 1 (pressure notify),
 * so the 1-arg form maps trivially. */
#if LINUX_VERSION_CODE < KERNEL_VERSION(6, 8, 0)
# define GB_EVENTFD_SIGNAL(ctx)  eventfd_signal((ctx), 1)
#else
# define GB_EVENTFD_SIGNAL(ctx)  eventfd_signal((ctx))
#endif

/* kmap_local_page / kunmap_local: added in 5.11.  Provide a shim for
 * earlier kernels.
 *
 * PR-CC correction: the previous shim mapped to kmap_atomic / kunmap_atomic
 * which DISABLES preemption while the mapping is held.  greenboost's callers
 * (gb_t3_evict_buf, gb_t3_promote_buf) issue kernel_write/kernel_read BETWEEN
 * map and unmap - those are FS/page-cache operations that can sleep, take
 * inode locks, queue I/O writeback.  Sleeping with preempt disabled triggers
 * "BUG: scheduling while atomic" on kernels < 5.11.
 *
 * Correct shim: kmap() / kunmap() - sleepable, no preempt-disable, available
 * on every kernel from 2.0+.  Slower on highmem systems (uses a fixed mapping
 * pool) but the T3 paths are not hot - they're per-page slow-path eviction
 * and promotion under user-visible latency budgets measured in milliseconds. */
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 11, 0)
# include <linux/highmem.h>
# define kmap_local_page(p)   kmap((p))
# define kunmap_local(addr)   kunmap(virt_to_page((addr)))
#endif

#endif /* GREENBOOST_COMPAT_H */

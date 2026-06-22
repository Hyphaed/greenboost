/* SPDX-License-Identifier: GPL-2.0 */
/*
 * ebpf/gb_offsets.h - struct gb_buf field offsets for eBPF probes
 *
 * Since greenboost.ko does not expose BTF, the eBPF program cannot use
 * CO-RE struct accessors for module types.  These offsets are derived
 * from the struct gb_buf layout in greenboost.c:244 and must be updated
 * whenever that layout changes.  A _Static_assert in verify_offsets.c
 * (userspace, compiled at build time) catches mismatches early.
 *
 * Layout (64-bit kernel, greenboost v3.x):
 *
 *   offset  field
 *   ──────  ─────────────────────────────────────
 *     0     struct page    **pages            (8 B)
 *     8     struct page    **hpages           (8 B)
 *    16     unsigned int     nhpages          (4 B)
 *    20     bool             hugepages        (1 B)
 *    21     bool             user_pinned      (1 B)
 *    22     [pad 2 B]
 *    24     unsigned int     npages           (4 B)
 *    28     [pad 4 B]
 *    32     size_t           size             (8 B) ← GB_BUF_SIZE_OFFSET
 *    40     int              id               (4 B)
 *    44     int              tier             (4 B)
 *    48     struct dma_buf  *dmabuf           (8 B)
 *    56     struct list_head lru_node         (16 B)
 *    72     unsigned long    alloc_jiffies    (8 B)
 *    80     unsigned long    last_jiffies     (8 B)
 *    88     u32              alloc_flags      (4 B) ← GB_BUF_FLAGS_OFFSET
 *    92     u8               frozen           (1 B)
 *    93     u8               t1_priority      (1 B)
 *    94     u8               session_priority (1 B)
 *    95     [pad 1 B]
 *    96     u32              heat             (4 B)  ← added by the hot/cold
 *                                                       residency engine
 *                                                       (GB_IOCTL_SET_HEAT)
 *   100     pid_t            owner_pid        (4 B)
 *
 * NOTE: `heat` was inserted ahead of owner_pid.  SIZE/FLAGS/TIER offsets
 * above are unaffected (all three sit before this insertion point), so no
 * #define below needed to change - this entry is here only so the next
 * person adding a probe on a field at or after the old owner_pid offset
 * has the current layout.
 */

#ifndef GB_OFFSETS_H
#define GB_OFFSETS_H

#define GB_BUF_SIZE_OFFSET    32   /* offsetof(struct gb_buf, size)       */
#define GB_BUF_FLAGS_OFFSET   88   /* offsetof(struct gb_buf, alloc_flags)*/
#define GB_BUF_TIER_OFFSET    44   /* offsetof(struct gb_buf, tier)       */

/* GB_ALLOC_* flag bits (from greenboost_ioctl.h) used in eBPF filtering */
#define GB_ALLOC_KV_CACHE     (1U << 3)
#define GB_ALLOC_T1_PRIORITY  (1U << 4)

#endif /* GB_OFFSETS_H */

/* gb_thp_verify.h - ask for 2 MiB huge pages explicitly, then check what the
 * kernel actually gave. Header-only, plain C99, no CUDA and no kernel headers,
 * so tests/c/test_gb_thp_verify.c runs it without a GPU.
 *
 * Why this exists: transparent_hugepage can be "always" while defrag is
 * "defer+madvise". Then an anonymous mapping that was never madvised only gets a
 * huge page when one is free at fault time, and otherwise silently falls back to
 * 4 KiB pages without compacting. Measured since boot on 2026-10-28:
 * thp_fault_fallback was about equal to thp_fault_alloc. MADV_HUGEPAGE switches
 * the fault to direct compaction for that mapping. The shim never recorded which
 * page size backed the T2 ranges the GPU reads over PCIe, so this also reports it.
 */
#ifndef GB_THP_VERIFY_H
#define GB_THP_VERIFY_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define GB_THP_PMD_SIZE_PATH "/sys/kernel/mm/transparent_hugepage/hpage_pmd_size"

/* Huge page size as the kernel reports it. 0 when unavailable. Never hard-coded. */
static inline size_t gb_thp_read_pmd_size(const char *path)
{
    unsigned long long v = 0;
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    int ok = fscanf(f, "%llu", &v);
    fclose(f);
    return ok == 1 ? (size_t)v : 0;
}

/* Round up to a multiple of `a` (not assumed to be a power of two). */
static inline uintptr_t gb_thp_align_up(uintptr_t v, size_t a)
{
    return a ? (v + (a - 1)) / a * a : v;
}

/* Opt-in switch for GREENBOOST_T2_HUGEPAGES. Anything but these values is off. */
static inline int gb_thp_env_enabled(const char *v)
{
    return v && (!strcmp(v, "1") || !strcmp(v, "on") || !strcmp(v, "true") || !strcmp(v, "yes"));
}

/* Percentage of the VMA containing `addr` backed by huge pages, read from a
 * smaps-format stream. -1 when the VMA is not found.
 *
 * VMA-level only: smaps has no per-range figure, and adjacent mappings with
 * identical flags can merge into one VMA, so treat the number as an estimate. */
static inline int gb_thp_vma_huge_percent(FILE *smaps, uintptr_t addr)
{
    char line[512];
    int in_vma = 0, got_size = 0, got_huge = 0;
    unsigned long long size_kb = 0, huge_kb = 0;

    while (fgets(line, sizeof line, smaps)) {
        unsigned long long lo, hi;
        char perms[8];
        if (sscanf(line, "%llx-%llx %7s", &lo, &hi, perms) == 3) {
            if (in_vma)
                break;                              /* left the VMA we wanted */
            in_vma = (addr >= (uintptr_t)lo && addr < (uintptr_t)hi);
            continue;
        }
        if (!in_vma)
            continue;
        if (sscanf(line, "Size: %llu kB", &size_kb) == 1)
            got_size = 1;
        else if (sscanf(line, "AnonHugePages: %llu kB", &huge_kb) == 1)
            got_huge = 1;
    }
    if (!in_vma || !got_size || !got_huge || !size_kb)
        return -1;
    return (int)(huge_kb * 100ULL / size_kb);
}

#endif /* GB_THP_VERIFY_H */

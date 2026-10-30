/* Userspace test of the capacity-report logic in greenboost_cuda_shim.c (region
 * GB_TESTABLE_BEGIN/END gb_report_logic, extracted by run.sh).  Loads nothing, touches no GPU. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include "report_logic.inc"

static int fails, checks;
#define CHECK(c) do { checks++; if (!(c)) { fails++; printf("FAIL line %d: %s\n", __LINE__, #c); } } while (0)

static void test_sat_add(void)
{
    CHECK(gb_sat_add_u64(1, 2) == 3);
    CHECK(gb_sat_add_u64(~0ULL, 1) == ~0ULL);
    CHECK(gb_sat_add_u64(~0ULL - 5, 10) == ~0ULL);
}

static void test_nvml_inflate(void)
{
    unsigned long long t = 12ULL << 30, f = 3ULL << 30;
    gb_nvml_inflate(1, &t, &f, 115ULL << 30, 100ULL << 30);      /* physical profile: untouched */
    CHECK(t == (12ULL << 30) && f == (3ULL << 30));
    gb_nvml_inflate(0, &t, &f, 42ULL << 30, 40ULL << 30);        /* virtual profile: pool added */
    CHECK(t == (54ULL << 30) && f == (43ULL << 30));
    t = ~0ULL - 1; f = 1;
    gb_nvml_inflate(0, &t, &f, 100, 100);                        /* never wraps */
    CHECK(t == ~0ULL && f == 101);
    gb_nvml_inflate(0, NULL, &f, 1, 1);                          /* NULL safe */
    gb_nvml_inflate(0, &t, NULL, 1, 1);
}

static void test_profile(void)
{
    CHECK(gb_report_profile_resolve(NULL, 0, "python3", NULL) == GB_PROFILE_VIRTUAL);   /* default unchanged */
    CHECK(gb_report_profile_resolve("", 1, "python3", NULL) == GB_PROFILE_PHYSICAL);    /* legacy flag */
    CHECK(gb_report_profile_resolve("virtual", 1, "python3", NULL) == GB_PROFILE_VIRTUAL);   /* explicit wins */
    CHECK(gb_report_profile_resolve("physical", 0, "ollama", NULL) == GB_PROFILE_PHYSICAL);
    CHECK(gb_report_profile_resolve("auto", 0, "python3", NULL) == GB_PROFILE_PHYSICAL);
    CHECK(gb_report_profile_resolve("auto", 0, "ollama", NULL) == GB_PROFILE_VIRTUAL);
    CHECK(gb_report_profile_resolve("auto", 0, "llama-server", NULL) == GB_PROFILE_VIRTUAL);
    CHECK(gb_report_profile_resolve("auto", 0, "olla", NULL) == GB_PROFILE_PHYSICAL);   /* exact match, not prefix */
    CHECK(gb_report_profile_resolve("auto", 0, "ollama2", NULL) == GB_PROFILE_PHYSICAL);
    CHECK(gb_report_profile_resolve("auto", 0, "myserver", "myserver,other") == GB_PROFILE_VIRTUAL);
    CHECK(gb_report_profile_resolve("auto", 0, "ollama", "myserver") == GB_PROFILE_PHYSICAL);  /* env list replaces default */
    CHECK(gb_report_profile_resolve("auto", 0, NULL, NULL) == GB_PROFILE_PHYSICAL);
    CHECK(gb_report_profile_resolve("pooled", 0, "x", NULL) == GB_PROFILE_INVALID);     /* fail loud, caller warns */
    CHECK(gb_csv_has("a,b,c", "c") && gb_csv_has("a,b,c", "a") && !gb_csv_has("a,b,c", "") && !gb_csv_has(NULL, "a"));
}

static void test_phys_pick(void)
{
    unsigned long long f, t;
    const unsigned long long MB = 1ULL << 20, GB = 1ULL << 30;
    /* the measured bug: nothing allocated yet, cache never filled -> used to print 0 */
    CHECK(gb_phys_pick(0, 0, ~0ULL, 1, 8 * GB, 12 * GB, &f, &t) == GB_PHYS_SRC_NVML && f == 8 * GB && t == 12 * GB);
    /* no NVML either: honest "unknown", not a fake full card */
    CHECK(gb_phys_pick(0, 0, ~0ULL, 0, 0, 0, &f, &t) == GB_PHYS_SRC_NONE && f == 0 && t == 0);
    /* fresh CUDA cache wins and NVML is not needed */
    CHECK(gb_phys_pick(5 * GB, 11 * GB, 40, 1, 8 * GB, 12 * GB, &f, &t) == GB_PHYS_SRC_CUDA && f == 5 * GB);
    CHECK(gb_phys_pick(5 * GB, 11 * GB, GB_PHYS_CACHE_MAX_AGE_MS, 0, 0, 0, &f, &t) == GB_PHYS_SRC_CUDA);
    /* stale cache + NVML: prefer NVML (a quiet process still reports live numbers) */
    CHECK(gb_phys_pick(5 * GB, 11 * GB, GB_PHYS_CACHE_MAX_AGE_MS + 1, 1, 6 * GB, 12 * GB, &f, &t) == GB_PHYS_SRC_NVML && f == 6 * GB);
    /* stale cache, no NVML: stale number beats nothing */
    CHECK(gb_phys_pick(5 * GB, 11 * GB, 999999, 0, 0, 0, &f, &t) == GB_PHYS_SRC_CUDA && f == 5 * GB);
    /* NVML failed (ok=0) or returned total 0 */
    CHECK(gb_phys_pick(0, 0, ~0ULL, 1, 8 * GB, 0, &f, &t) == GB_PHYS_SRC_NONE);
    /* free is clamped to total */
    CHECK(gb_phys_pick(0, 0, ~0ULL, 1, 20 * GB, 12 * GB, &f, &t) == GB_PHYS_SRC_NVML && f == 12 * GB);
    (void)MB;
}

int main(void)
{
    test_sat_add();
    test_nvml_inflate();
    test_profile();
    test_phys_pick();
    printf("%d checks, %d failed\n", checks, fails);
    return fails ? 1 : 0;
}

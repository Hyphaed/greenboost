/* test_gb_thp_verify.c - host-only unit tests for gb_thp_verify.h. No CUDA, no
 * GPU: gcc -std=c99 -Wall -Wextra -I../.. -o /tmp/t tests/c/test_gb_thp_verify.c
 * (or `make test-c` from the repo root).
 */
#define _POSIX_C_SOURCE 200809L   /* mkstemp and fdopen under -std=c99 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../gb_thp_verify.h"

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); failures++; } \
    else { printf("ok - %s\n", msg); } \
} while (0)

static FILE *smaps_fixture(void)
{
    FILE *f = tmpfile();
    fputs("7f0000000000-7f0000400000 rw-p 00000000 00:00 0\n"
          "Size:               4096 kB\n"
          "Rss:                4096 kB\n"
          "AnonHugePages:      2048 kB\n"
          "7f0000400000-7f0000600000 rw-p 00000000 00:00 0\n"
          "Size:               2048 kB\n"
          "AnonHugePages:      2048 kB\n"
          "7f0000600000-7f0000800000 rw-p 00000000 00:00 0\n"
          "Size:               2048 kB\n"
          "AnonHugePages:         0 kB\n", f);
    rewind(f);
    return f;
}

static void test_percent_per_vma(void)
{
    FILE *f = smaps_fixture();
    CHECK(gb_thp_vma_huge_percent(f, 0x7f0000100000UL) == 50, "half-huge VMA reports 50%");
    rewind(f);
    CHECK(gb_thp_vma_huge_percent(f, 0x7f0000500000UL) == 100, "fully huge VMA reports 100%");
    rewind(f);
    CHECK(gb_thp_vma_huge_percent(f, 0x7f0000700000UL) == 0, "all-4K VMA reports 0%");
    rewind(f);
    CHECK(gb_thp_vma_huge_percent(f, 0x7f9999000000UL) == -1, "address in no VMA reports -1");
    fclose(f);
}

static void test_percent_stops_at_the_next_vma(void)
{
    FILE *f = tmpfile();
    fputs("1000-2000 rw-p 0 0:0 0\nSize: 4 kB\n2000-3000 rw-p 0 0:0 0\nAnonHugePages: 4 kB\n", f);
    rewind(f);
    CHECK(gb_thp_vma_huge_percent(f, 0x1800) == -1, "a field from the next VMA is not borrowed");
    fclose(f);
}

static void test_pmd_size_is_read_not_assumed(void)
{
    char path[] = "/tmp/gb_thp_pmd_XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0, "temp file for the sysfs stand-in");
    FILE *f = fdopen(fd, "w");
    fputs("2097152\n", f);
    fclose(f);
    CHECK(gb_thp_read_pmd_size(path) == 2097152u, "pmd size parsed from the file");
    remove(path);
    CHECK(gb_thp_read_pmd_size("/nonexistent/hpage_pmd_size") == 0, "missing file reads as 0");
}

static void test_align_up(void)
{
    CHECK(gb_thp_align_up(5, 4) == 8, "5 rounds up to 8");
    CHECK(gb_thp_align_up(8, 4) == 8, "already aligned stays put");
    CHECK(gb_thp_align_up(7, 0) == 7, "zero alignment is a no-op");
    CHECK(gb_thp_align_up(1, 3) == 3, "non power of two alignment works");
}

static void test_env_switch(void)
{
    CHECK(gb_thp_env_enabled("1") && gb_thp_env_enabled("on") && gb_thp_env_enabled("true"), "accepts 1/on/true");
    CHECK(!gb_thp_env_enabled(NULL) && !gb_thp_env_enabled("") && !gb_thp_env_enabled("0"), "NULL, empty and 0 are off");
    CHECK(!gb_thp_env_enabled("maybe"), "anything else is off");
}

int main(void)
{
    test_percent_per_vma();
    test_percent_stops_at_the_next_vma();
    test_pmd_size_is_read_not_assumed();
    test_align_up();
    test_env_switch();
    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("all gb_thp_verify tests passed\n");
    return 0;
}

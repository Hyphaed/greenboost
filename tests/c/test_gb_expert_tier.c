/* test_gb_expert_tier.c — host-only unit tests for gb_expert_tier.h's LFRU
 * scoring. No CUDA, no kernel headers, no GPU needed: build + run with
 *   gcc -std=c99 -I../.. -o /tmp/test_gb_expert_tier tests/c/test_gb_expert_tier.c && /tmp/test_gb_expert_tier
 * (or `make test-c` from the repo root).
 */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../../gb_expert_tier.h"

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); failures++; } \
    else { printf("ok - %s\n", msg); } \
} while (0)

static void test_no_swap_when_uniform_heat(void)
{
    uint32_t heat[4] = {10, 10, 10, 10};
    uint32_t last[4] = {0, 0, 0, 0};
    int resident[2] = {0, 1};
    int victim, promote; long gain;
    int swapped = gb_tier_pick_lfru(heat, last, 5, 4, resident, 2, &victim, &promote, &gain);
    CHECK(!swapped, "uniform heat across all candidates never swaps");
}

static void test_swap_when_hot_candidate_clears_margin(void)
{
    /* resident 0,1 cold (heat=1); candidate 2 is far hotter (heat=100). */
    uint32_t heat[4] = {1, 1, 100, 2};
    uint32_t last[4] = {0, 0, 5, 0};
    int resident[2] = {0, 1};
    int victim, promote; long gain;
    int swapped = gb_tier_pick_lfru(heat, last, 5, 4, resident, 2, &victim, &promote, &gain);
    CHECK(swapped, "clearly hotter non-resident candidate triggers a swap");
    CHECK(promote == 2, "promotes the actually-hottest candidate (id 2)");
    CHECK(victim == 0 || victim == 1, "demotes one of the two cold residents");
    CHECK(gain > 0, "reports a positive score gain");
}

static void test_no_swap_within_hysteresis_margin(void)
{
    /* Candidate 2 is only marginally hotter than the coldest resident — must
     * NOT swap (this is the anti-ping-pong guarantee; without it, two
     * chunks with near-identical heat would flap every repin pass). */
    uint32_t heat[4] = {50, 50, 52, 10};   /* 52 vs 50: +4% ~ well under the 25%+4 margin */
    uint32_t last[4] = {0, 0, 0, 0};
    int resident[2] = {0, 1};
    int victim, promote; long gain;
    int swapped = gb_tier_pick_lfru(heat, last, 1, 4, resident, 2, &victim, &promote, &gain);
    CHECK(!swapped, "near-tie heat stays within the hysteresis margin — no swap");
}

static void test_frequency_beats_recency(void)
{
    /* Candidate A: high heat, stale (accessed long ago). Candidate B: low
     * heat, just accessed. Frequency must win — one heat point (256 score
     * units) always outranks the max recency bonus (255). */
    uint32_t heat[2] = {5, 4};       /* A=5, B=4 */
    uint32_t last[2] = {0, 100};     /* A stale (age=100), B just touched (age=0) */
    uint64_t score_a = gb_tier_lfru_score(heat[0], last[0], 100);
    uint64_t score_b = gb_tier_lfru_score(heat[1], last[1], 100);
    CHECK(score_a > score_b, "one extra frequency count outranks maximum recency bonus");
}

static void test_decay_halves_all_heat(void)
{
    uint32_t heat[4] = {8, 7, 100, 1};
    gb_tier_decay(heat, 4);
    CHECK(heat[0] == 4 && heat[1] == 3 && heat[2] == 50 && heat[3] == 0,
          "decay halves every candidate's heat (integer floor)");
}

static void test_rejects_degenerate_inputs(void)
{
    uint32_t heat[2] = {1, 1};
    uint32_t last[2] = {0, 0};
    int resident[1] = {0};
    int victim, promote; long gain;
    CHECK(!gb_tier_pick_lfru(NULL, last, 1, 2, resident, 1, &victim, &promote, &gain),
          "null heat array is rejected, not dereferenced");
    CHECK(!gb_tier_pick_lfru(heat, last, 1, 2, resident, 0, &victim, &promote, &gain),
          "zero resident candidates is rejected (nothing to demote)");
    CHECK(!gb_tier_pick_lfru(heat, last, 1, 1, resident, 1, &victim, &promote, &gain),
          "resident count >= candidate count is rejected (nothing to promote)");
}

int main(void)
{
    test_no_swap_when_uniform_heat();
    test_swap_when_hot_candidate_clears_margin();
    test_no_swap_within_hysteresis_margin();
    test_frequency_beats_recency();
    test_decay_halves_all_heat();
    test_rejects_degenerate_inputs();

    if (failures) {
        fprintf(stderr, "\n%d test(s) FAILED\n", failures);
        return 1;
    }
    printf("\nall gb_expert_tier tests passed\n");
    return 0;
}

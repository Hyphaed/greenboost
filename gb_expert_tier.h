#ifndef GREENBOOST_EXPERT_TIER_H
#define GREENBOOST_EXPERT_TIER_H
/*
 * gb_expert_tier.h — LFRU hot-chunk scoring for the front-load VRAM/host
 * split (gb_frontload_split_alloc in greenboost_cuda_shim.c).
 *
 * Port of colibri's c/tier.h (see workflow/porting-reference.md §CB-1).
 * Pure C99, no CUDA/pthread/kernel dependency — every function here is a
 * closed-form computation over caller-owned arrays, so it is independently
 * unit-testable on the host without touching the GPU (see
 * tests/c/test_gb_expert_tier.c). This header decides WHICH chunk to
 * promote/demote; it never itself touches memory, CUDA, or locks — the
 * caller (greenboost_cuda_shim.c) owns all of that.
 *
 * Terminology mapping vs colibri: colibri scores individual EXPERT tensors
 * (it has per-tensor visibility inside its own inference engine). GreenBoost's
 * shim sits below ggml/llama.cpp at the CUDA-driver level and has no expert-
 * tensor-name visibility — it only sees which byte ranges ("chunks", the
 * same GB_FL_CHUNK_UNIT granularity gb_frontload_split_alloc already uses)
 * inside a front-loaded VA range are touched by cuLaunchKernel argument
 * pointers. So "expert" below reads as "chunk": the heat/score functions are
 * IDENTICAL math, applied at chunk instead of per-expert-tensor granularity.
 */

#include <stdint.h>

/* Pick one chunk to demote (coldest resident) and one to promote (hottest
 * non-resident), if the gain clears the anti-ping-pong hysteresis margin.
 * heat[]/last[] are indexed by CANDIDATE id (0..ncand-1, e.g. one entry per
 * GB_FL_CHUNK_UNIT-sized chunk of the logical buffer, resident or not).
 * resident[] lists the ids of the nres chunks currently DEVICE-backed
 * (candidates for demotion); every id in 0..ncand-1 NOT in resident[] is a
 * host-backed candidate for promotion.
 *
 * Returns 1 and sets *victim_id (currently resident, to demote to host) and
 * *promote_id (currently host, to promote to device) and *gain (score
 * delta) when a swap is worth making; 0 when no swap clears the hysteresis
 * gate (including: nres==0, ncand<=nres, or all candidates equally cold).
 *
 * Hysteresis: identical to colibri's tier_pick_lfru — the fixed +4<<8 margin
 * absorbs tiny samples (first few tokens), the cold_score>>2 (25%) relative
 * margin stops oscillation between two chunks with merely-close heat. This
 * gate is the ENTIRE reason repeated swaps don't thrash: do not loosen it.
 */
static inline uint64_t gb_tier_lfru_score(uint32_t heat, uint32_t last, uint32_t clock)
{
    uint32_t age = clock - last;
    uint32_t recent = age < 255u ? 255u - age : 0u;
    /* One frequency count (score += 256) always outranks the max recency
     * bonus (255) — frequency is the primary signal, recency only breaks
     * close calls between similarly-hot candidates. */
    return ((uint64_t)heat << 8) | recent;
}

static inline int gb_tier_pick_lfru(const uint32_t *heat, const uint32_t *last,
                                    uint32_t clock, int ncand,
                                    const int *resident, int nres,
                                    int *victim_id, int *promote_id, long *gain)
{
    if (!heat || !last || !resident || nres < 1 || ncand < 1 || nres >= ncand)
        return 0;

    /* Coldest resident (demotion candidate). */
    int cold_idx = 0;
    uint64_t cold_score = gb_tier_lfru_score(heat[resident[0]], last[resident[0]], clock);
    for (int z = 1; z < nres; z++) {
        uint64_t s = gb_tier_lfru_score(heat[resident[z]], last[resident[z]], clock);
        if (s < cold_score) { cold_score = s; cold_idx = z; }
    }

    /* Hottest non-resident (promotion candidate). */
    int hot = -1;
    uint64_t hot_score = 0;
    for (int c = 0; c < ncand; c++) {
        int is_resident = 0;
        for (int z = 0; z < nres; z++) {
            if (resident[z] == c) { is_resident = 1; break; }
        }
        if (is_resident) continue;
        uint64_t s = gb_tier_lfru_score(heat[c], last[c], clock);
        if (hot < 0 || s > hot_score) { hot = c; hot_score = s; }
    }
    if (hot < 0)
        return 0;

    /* Anti-ping-pong hysteresis: only swap if the hot candidate clears the
     * cold resident's score by more than 25% + a fixed floor (colibri's
     * exact margin, ported verbatim — do not tune this without re-deriving
     * why colibri chose it: fixed floor absorbs near-zero-sample noise,
     * 25% relative margin stops oscillation between close scores). */
    if (hot_score <= cold_score + (cold_score >> 2) + (4u << 8))
        return 0;

    *victim_id = resident[cold_idx];
    *promote_id = hot;
    *gain = (long)((hot_score - cold_score) >> 8);
    return 1;
}

/* Halve every candidate's heat. Call once per repin pass (colibri: after
 * every swap-eligible pass) so old bursts of activity decay and current
 * routing frequency dominates the score, not lifetime totals. */
static inline void gb_tier_decay(uint32_t *heat, int ncand)
{
    for (int c = 0; c < ncand; c++)
        heat[c] >>= 1;
}

#endif /* GREENBOOST_EXPERT_TIER_H */

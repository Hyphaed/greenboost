/* RT-13 (shim T2 pool cap overshoot) non-device unit test.
 * The reservation helpers are NOT copied here: the driver (shim_rt07_rt13_pool_cap.py)
 * extracts the GB_T2_CAP_BEGIN..END region from the shim source and compiles it in via
 * -include, so the test runs the code that ships.  The allocator is fake: a counter.
 * Exit 0 = all assertions held.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifndef GB_CAP_PCT
#define GB_CAP_PCT 88
#endif
static _Atomic size_t gb_t2_overflow_bytes;
static size_t gb_t2_pool_bytes;
static inline size_t gb_effective_t2_cap(void)
{ return gb_t2_pool_bytes ? gb_t2_pool_bytes * GB_CAP_PCT / 100 : 0; }
#include GB_CAP_REGION   /* gb_t2_try_reserve / gb_t2_release_reserved */

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s: ", #c); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* fake allocator: tracks what is "really" allocated and its high-water mark */
static _Atomic size_t fake_live, fake_peak;
static void fake_alloc(size_t n)
{
    size_t now = atomic_fetch_add(&fake_live, n) + n, p = atomic_load(&fake_peak);
    while (now > p && !atomic_compare_exchange_weak(&fake_peak, &p, now)) {}
    usleep(50);                               /* widen the check/alloc window */
}
static void fake_free(size_t n) { atomic_fetch_sub(&fake_live, n); }

/* pre-patch shape: load, compare, allocate, add (Path A/B/VMM before RT-13) */
static int old_check_then_add(size_t bytes)
{
    size_t cap = gb_effective_t2_cap(), used = atomic_load_explicit(&gb_t2_overflow_bytes, memory_order_relaxed);
    if (used >= cap || bytes > cap - used) return -1;
    fake_alloc(bytes);
    atomic_fetch_add_explicit(&gb_t2_overflow_bytes, bytes, memory_order_relaxed);
    return 0;
}

#define NTHR 16
#define CHUNK ((size_t)10)
static int use_old;
static atomic_int go;
static atomic_int denied, granted;

static void *worker(void *arg)
{
    long id = (long)arg;
    while (!atomic_load(&go)) {}
    for (int i = 0; i < 400; i++) {
        size_t n = CHUNK + (size_t)((id + i) % 3);
        if (use_old) {
            if (old_check_then_add(n) == 0) { atomic_fetch_add(&granted, 1); }
            else atomic_fetch_add(&denied, 1);
            continue;
        }
        if (gb_t2_try_reserve(n) != 0) { atomic_fetch_add(&denied, 1); continue; }
        fake_alloc(n);
        if ((id + i) % 7 == 0) {                /* simulated driver failure: give it back */
            fake_free(n); gb_t2_release_reserved(n);
            continue;
        }
        atomic_fetch_add(&granted, 1);
        if (atomic_load(&fake_live) > gb_effective_t2_cap()) { atomic_fetch_add(&denied, 1000000); }
        if ((id + i) % 2) { fake_free(n); gb_t2_release_reserved(n); }   /* app frees some */
    }
    return NULL;
}

static void run(int old)
{
    pthread_t t[NTHR];
    use_old = old; atomic_store(&go, 0);
    atomic_store(&gb_t2_overflow_bytes, 0); atomic_store(&fake_live, 0); atomic_store(&fake_peak, 0);
    atomic_store(&granted, 0); atomic_store(&denied, 0);
    for (long i = 0; i < NTHR; i++) pthread_create(&t[i], NULL, worker, (void *)i);
    atomic_store(&go, 1);
    for (int i = 0; i < NTHR; i++) pthread_join(t[i], NULL);
}

int main(void)
{
    /* ---- sequential accounting ---- */
    gb_t2_pool_bytes = 1000;                       /* cap = 880 */
    size_t cap = gb_effective_t2_cap();
    CHECK(cap == 1000 * GB_CAP_PCT / 100, "cap %zu", cap);
    atomic_store(&gb_t2_overflow_bytes, 0);
    CHECK(gb_t2_try_reserve(500) == 0, "first fits");
    CHECK(gb_t2_try_reserve(cap - 500 + 1) == -1, "one byte over refused");
    CHECK(atomic_load(&gb_t2_overflow_bytes) == 500, "refusal leaves the counter alone: %zu", atomic_load(&gb_t2_overflow_bytes));
    CHECK(gb_t2_try_reserve(cap - 500) == 0, "exact fit accepted");
    CHECK(atomic_load(&gb_t2_overflow_bytes) == cap, "full");
    CHECK(gb_t2_try_reserve(1) == -1, "full refuses");
    gb_t2_release_reserved(500);
    CHECK(atomic_load(&gb_t2_overflow_bytes) == cap - 500, "release subtracts");
    CHECK(gb_t2_try_reserve(SIZE_MAX) == -1, "size_t max must not wrap");
    CHECK(gb_t2_try_reserve(SIZE_MAX - 10) == -1, "near max must not wrap");
    gb_t2_release_reserved(cap - 500);
    CHECK(atomic_load(&gb_t2_overflow_bytes) == 0, "back to zero");

    /* cap disabled (pool 0): everything granted, accounting still balanced, no underflow */
    gb_t2_pool_bytes = 0;
    CHECK(gb_t2_try_reserve(123) == 0, "disabled cap grants");
    CHECK(atomic_load(&gb_t2_overflow_bytes) == 123, "disabled cap still accounts");
    gb_t2_release_reserved(123);
    CHECK(atomic_load(&gb_t2_overflow_bytes) == 0, "no underflow after release with cap disabled");

    /* ---- parallel: old pattern must be able to overshoot, new one never ---- */
    gb_t2_pool_bytes = 250;                        /* cap = 220 = 22 chunks of 10 */
    cap = gb_effective_t2_cap();
    int overshoot_seen = 0;
    for (int r = 0; r < 20 && !overshoot_seen; r++) {
        run(1);
        overshoot_seen = atomic_load(&fake_peak) > cap;
    }
    printf("INFO pre-patch check-then-add: overshoot %s (peak %zu / cap %zu)\n",
           overshoot_seen ? "REPRODUCED" : "not triggered this run", (size_t)atomic_load(&fake_peak), cap);

    size_t worst = 0;
    for (int r = 0; r < 50; r++) {
        run(0);
        if (atomic_load(&fake_peak) > worst) worst = atomic_load(&fake_peak);
        CHECK(atomic_load(&fake_peak) <= cap, "round %d: fake allocator peak %zu > cap %zu", r, (size_t)atomic_load(&fake_peak), cap);
        CHECK(atomic_load(&gb_t2_overflow_bytes) == atomic_load(&fake_live),
              "round %d: counter %zu != really allocated %zu", r, (size_t)atomic_load(&gb_t2_overflow_bytes), (size_t)atomic_load(&fake_live));
        CHECK(atomic_load(&denied) < 1000000, "round %d: live exceeded cap at a grant", r);
    }
    printf("INFO patched reservation: worst peak %zu <= cap %zu over 50 rounds x %d threads\n", worst, cap, NTHR);
    printf("%s\n", fails ? "RESULT FAIL" : "RESULT OK");
    return fails ? 1 : 0;
}

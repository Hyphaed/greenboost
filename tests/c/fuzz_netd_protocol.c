/* SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
 *
 * Fuzzer for greenboost_netd.c's payload parsers - the daemon's untrusted
 * surface.
 *
 * Why this exists: every handle_*(cli, payload, len) in greenboost_netd.c
 * parses bytes that arrived over a socket from a feeder. The suite already
 * covers this surface from Python with mocks (wire-struct layout, PSK rules,
 * sequence monotonicity, handle_cuda_launch's payload-size validation) - all
 * of which test the RULES. None of them run the real C parser over hostile
 * bytes, which is the thing that finds the integer overflow the rules were
 * written for after the fact (PR-C/C6).
 *
 * The sibling greenboost_vgpu project has carried two such fuzzers for its own
 * ring protocol since 2026-08-01 (tests/fuzz_dispatch.c, tests/fuzz_gbpvg.c),
 * and its own notes record the lesson this file is built on: naive random
 * input barely reaches the interesting code. A random 12-byte header has
 * roughly a one-in-a-million chance of passing a size gate, so a fuzzer that
 * only throws noise measures the early-reject path and nothing else. Hence the
 * modes below, which progressively construct MORE valid framing so dispatch
 * actually happens.
 *
 * The only assertion is the same one those fuzzers make: the process is still
 * alive. These parsers return int status codes, not diagnostics, so "did not
 * crash and ASan/UBSan stayed quiet" is the entire contract. Build with
 * -fsanitize=address,undefined or this proves nothing - an out-of-bounds read
 * inside a 4 KB buffer is invisible in an optimised build.
 *
 * Build + run:
 *     make -f tests/c/Makefile.fuzz && ./fuzz_netd_protocol 20000
 */

/* greenboost_netd.c uses offsetof() but reaches it transitively through a
 * header the daemon build happens to pull in first. Include it explicitly
 * rather than relying on that ordering. */
#include <stddef.h>

/* The daemon owns main(); rename it so this file can provide its own. */
#define main gb_netd_daemon_main_unused
#include "../../greenboost_netd.c"
#undef main

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Deterministic PRNG: a fuzz failure nobody can reproduce is a rumour. The
 * seed is printed on every run and accepted as argv[2]. */
static uint64_t g_rng = 0x9E3779B97F4A7C15ULL;
static uint32_t rnd(void)
{
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 32);
}

/* Values that break parsers: zero, one-off boundaries, type maxima, and the
 * sign-flip points an int/size_t confusion turns into a huge allocation. */
static uint32_t boundary_u32(void)
{
    static const uint32_t v[] = {
        0u, 1u, 2u, 3u, 4u, 7u, 8u, 15u, 16u, 63u, 64u, 255u, 256u, 4095u,
        4096u, 65535u, 65536u, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFEu,
        0xFFFFFFFFu,
    };
    return v[rnd() % (sizeof(v) / sizeof(v[0]))];
}

static void fill_random(uint8_t *buf, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        /* Bias toward 0x00/0xFF: parsers break on runs, not on noise. */
        uint32_t r = rnd();
        buf[i] = (r % 4 == 0) ? 0x00 : (r % 4 == 1) ? 0xFF : (uint8_t)(r >> 3);
    }
}

/* A client that owns no socket. fd = -1 throughout: every handler under test
 * is a PARSER, and any of them that tries to answer will fail the write and
 * return an error rather than touching a real descriptor. */
static struct client *fresh_client(void)
{
    static struct client cli;
    memset(&cli, 0, sizeof(cli));
    cli.fd = -1;
    cli.active = 1;
    cli.feeder_id = rnd() % 8;
    snprintf(cli.remote_addr, sizeof(cli.remote_addr), "127.0.0.1");
    cli.alloc_tokens = 200000;
    return &cli;
}

#define PAYLOAD_MAX 8192
static uint8_t g_payload[PAYLOAD_MAX];

/* Mode 1: pure noise at a random length. Mostly exercises early rejects,
 * which is still real coverage of the length checks themselves. */
static void mode_random(void)
{
    uint32_t len = rnd() % PAYLOAD_MAX;
    fill_random(g_payload, len);
    struct client *c = fresh_client();
    switch (rnd() % 6) {
    case 0: handle_heartbeat(c, g_payload, len);        break;
    case 1: handle_gpu_query(c, g_payload, len);        break;
    case 2: handle_mem_info(c, g_payload, len);         break;
    case 3: handle_cuda_malloc(c, g_payload, len);      break;
    case 4: handle_cuda_free(c, g_payload, len);        break;
    case 5: handle_cuda_memset(c, g_payload, len);      break;
    }
}

/* Mode 2: boundary-biased lengths against a fixed buffer. This is where a
 * length field that disagrees with the real buffer shows up - the shape of
 * the PR-C/C6 overflow. The buffer really is PAYLOAD_MAX so ASan can see a
 * genuine over-read; the LEN we claim is a lie on purpose. */
static void mode_length_lies(void)
{
    fill_random(g_payload, PAYLOAD_MAX);
    uint32_t len = boundary_u32();
    if (len > PAYLOAD_MAX) len = PAYLOAD_MAX;   /* never hand out a real OOB */
    struct client *c = fresh_client();
    switch (rnd() % 4) {
    case 0: handle_cuda_memcpy_h2d(c, g_payload, len,
                                   (uint16_t)rnd()); break;
    case 1: handle_cuda_memcpy_d2h(c, g_payload, len); break;
    case 2: handle_cuda_memcpy_d2d(c, g_payload, len); break;
    case 3: handle_cuda_malloc(c, g_payload, len);     break;
    }
}

/* Mode 3: structured - plant boundary values in the first fields, where the
 * sizes and counts live, and leave the tail random. This is what actually
 * gets past the front-door checks into the body of a parser. */
static void mode_structured(void)
{
    uint32_t len = 16 + (rnd() % 512);
    fill_random(g_payload, len);
    uint32_t nfields = 1 + (rnd() % 6);
    for (uint32_t i = 0; i < nfields && (i + 1) * 4 <= len; i++) {
        uint32_t v = boundary_u32();
        memcpy(g_payload + i * 4, &v, sizeof(v));
    }
    struct client *c = fresh_client();
    switch (rnd() % 5) {
    case 0: handle_handshake(c, g_payload, len);       break;
    case 1: handle_cuda_malloc(c, g_payload, len);     break;
    case 2: handle_cuda_memcpy_h2d(c, g_payload, len,
                                   (uint16_t)rnd()); break;
    case 3: handle_mem_info(c, g_payload, len);        break;
    case 4: handle_gpu_query(c, g_payload, len);       break;
    }
}

/* Mode 4: the handshake specifically. It runs BEFORE authentication, so it is
 * the one parser an unauthenticated peer can always reach - the highest-value
 * target on this surface, and the reason it gets its own mode. */
static void mode_handshake(void)
{
    uint32_t len = rnd() % 512;
    fill_random(g_payload, len);
    if (len >= 4) {
        uint32_t magic = (rnd() % 2) ? 0x47425044u : boundary_u32();
        memcpy(g_payload, &magic, sizeof(magic));
    }
    handle_handshake(fresh_client(), g_payload, len);
}

int main(int argc, char **argv)
{
    unsigned long iters = (argc > 1) ? strtoul(argv[1], NULL, 10) : 20000UL;
    if (argc > 2) g_rng = strtoull(argv[2], NULL, 0);
    if (!g_rng) g_rng = 0x9E3779B97F4A7C15ULL;
    uint64_t seed = g_rng;

    printf("fuzz_netd_protocol: %lu iterations, seed 0x%016llx\n",
           iters, (unsigned long long)seed);

    for (unsigned long i = 0; i < iters; i++) {
        switch (i % 4) {
        case 0: mode_random();     break;
        case 1: mode_length_lies(); break;
        case 2: mode_structured(); break;
        case 3: mode_handshake();  break;
        }
    }
    printf("fuzz_netd_protocol: survived %lu iterations "
           "(reproduce with: %s %lu 0x%016llx)\n",
           iters, argv[0], iters, (unsigned long long)seed);
    return 0;
}

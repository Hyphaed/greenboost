#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_bench_codec.py , the measurement that decides GB-3 / GB-K3.

The plan's Phase 2 says: spend CPU cycles to buy PCIe bandwidth. On this box
CPU memory bandwidth (24.2 GB/s) and PCIe host->device (24.4 GB/s) are the
same number, so compressing before the link is only worth it if the codec is
fast enough. This file measures whether it is, and states the break-even
arithmetic instead of asserting a conclusion.

THE BREAK-EVEN, stated once so every number below can be read against it.

Let L be the link rate (GB/s), R the compression ratio, C the codec rate in
GB/s of INPUT bytes. Moving one uncompressed byte costs 1/L. Moving it
compressed costs 1/C (codec) + 1/(R*L) (the smaller transfer). So compression
wins when

    1/C + 1/(R*L)  <  1/L      <=>      C  >  L * R/(R-1)

Read that ratio term: it never goes below 1, so **the codec must be faster
than the link, whatever the ratio**. R=2 needs 2L. R=20 still needs 1.05L.
A 20x ratio does not buy slack in the codec, it only removes the penalty of a
poor one. This is the single most important fact about Phase 2 and it is easy
to get backwards.

TWO PATHS, and only one of them has to beat that bar:

  * SYNCHRONOUS , compress on the spill, decompress on the promote. Both ends
    are on the critical path, so both must clear C > L*R/(R-1).
  * ASYNCHRONOUS , compress cold buffers in the background (they are cold, by
    definition nothing is waiting), and only decompression is on the path.
    Then the bar applies to the DECODE rate alone, and the encode rate only
    has to keep up with the rate at which buffers go cold, which on this box
    is a few GB/s at most.

The asynchronous path is what `0020-dma-buf-compressed-descriptor` was written
for , "compressed, not evicted" is a background state a buffer sits in, not a
step in a transfer. So the number that decides GB-K3 is the DECOMPRESSION
rate, and this benchmark reports encode and decode separately for that reason.

Run:  python3 gb_bench_codec.py            # human-readable
      python3 gb_bench_codec.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

# Measured on this box, 2026-08-19 (see workflow/aug19_1720_plan.md). Override
# per box rather than editing: nothing here may hardcode a hardware value.
LINK_GB_S = float(os.environ.get("GB_LINK_GB_S", "24.4"))

# What compression REPLACES decides which bar it has to clear, and there are two
# different alternatives in this system:
#
#   * a T2->T1 promote that would otherwise cross PCIe at LINK_GB_S, and
#   * an eviction to T3 whose bytes would otherwise be re-read from NVMe.
#
# 2.4 GB/s measured on this box 2026-08-19 (`dd` of the served GGUF, 2 GiB,
# page cache partly warm, so this is a LOWER bound on the drive and an
# upper bound on how easy the T3 bar is). Override per box.
T3_GB_S = float(os.environ.get("GB_T3_GB_S", "2.4"))

# 64 MiB is large enough to leave L3 (36 MB on this CPU) so the measurement is
# DRAM-bound like a real T2 buffer, and small enough to run many trials.
DEFAULT_BLOCK_MB = 64


def _kv_like(nbytes: int, seed: int = 7):
    """Bytes shaped like a KV cache, not like /dev/urandom.

    A KV cache is f16 activations: neighbouring values are correlated, the
    exponent byte is nearly constant, and the mantissa is close to noise.
    Benchmarking a codec on uniform random data understates every ratio and
    would reject Phase 2 on a fixture artifact; benchmarking it on zeros would
    accept it on one. This generator is the honest middle: a smooth random
    walk quantised to f16.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    n = nbytes // 2
    walk = np.cumsum(rng.standard_normal(n, dtype=np.float32) * 0.05)
    # Keep it in the range real attention states occupy rather than drifting.
    walk = np.tanh(walk / 8.0).astype(np.float16)
    return walk.tobytes()


def _time_codec(name: str, encode, decode, data: bytes, trials: int = 3) -> dict:
    enc_rates, dec_rates = [], []
    blob = None
    for _ in range(trials):
        t0 = time.perf_counter()
        blob = encode(data)
        t1 = time.perf_counter()
        out = decode(blob)
        t2 = time.perf_counter()
        if len(out) != len(data):
            return {"codec": name, "error": "decode length mismatch"}
        enc_rates.append(len(data) / (t1 - t0) / 1e9)
        dec_rates.append(len(data) / (t2 - t1) / 1e9)
    ratio = len(data) / max(1, len(blob))
    return {
        "codec": name,
        "ratio": round(ratio, 2),
        "encode_gb_s": round(statistics.median(enc_rates), 2),
        "decode_gb_s": round(statistics.median(dec_rates), 2),
        # The bar this codec has to clear, computed from its OWN ratio. Two
        # bars, because there are two things compression can replace.
        "break_even_gb_s": round(LINK_GB_S * ratio / (ratio - 1), 2) if ratio > 1 else None,
        "break_even_vs_t3_gb_s": round(T3_GB_S * ratio / (ratio - 1), 2) if ratio > 1 else None,
    }


def _codecs():
    """Every codec that could actually be used, in the form it would be used.

    zstd and lz4 are the two the KERNEL already has compiled in on this box
    (`CONFIG_CRYPTO_ZSTD=y`, `CONFIG_CRYPTO_LZ4=y`), which is why they are the
    candidates: GB-K3 runs inside `greenboost.ko`, so a codec that only exists
    as a Python wheel is not a candidate no matter how well it scores.
    """
    out = []
    try:
        import zstandard as zstd

        # A compressor/decompressor object per CALL, not one shared across
        # threads. Sharing one crashed the threaded stage outright ("stack
        # smashing detected") the first time this ran , python-zstandard's
        # context objects are not safe for concurrent use, and the failure mode
        # is a native stack corruption rather than an exception. The kernel-side
        # equivalent (GB-K3) has the same constraint: one crypto context per
        # kworker, never a shared one.
        def _mk(level):
            def enc(b, level=level):
                return zstd.ZstdCompressor(level=level).compress(b)

            def dec(b):
                return zstd.ZstdDecompressor().decompress(b, max_output_size=1 << 30)
            return enc, dec

        for level in (1, 3):
            e, d = _mk(level)
            out.append((f"zstd-{level}", e, d))
    except ImportError:
        pass
    try:
        import lz4.block as lz4b
        out.append(("lz4", lz4b.compress, lambda b: lz4b.decompress(b)))
    except ImportError:
        pass
    # Always available, and the useful floor: if a codec cannot beat what the
    # stdlib does, it is not worth a kernel patch.
    import zlib
    out.append(("zlib-1", lambda b: zlib.compress(b, 1), zlib.decompress))
    return out


def _threaded_rate(encode, data: bytes, threads: int, chunk_mb: int = 8) -> float:
    """Aggregate encode rate across N threads on independent chunks.

    This is the shape GB-K3 would actually run in , one kworker per cold
    buffer, not one thread on one huge buffer , and it is the only way the
    E-cores contribute. Python threads are fine here because both zstd and lz4
    release the GIL inside their C compress calls.
    """
    step = chunk_mb * 1024 * 1024
    chunks = [data[i:i + step] for i in range(0, len(data), step)] or [data]
    work = (chunks * ((threads // len(chunks)) + 1))[:max(threads, len(chunks))]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(encode, work))
    dt = time.perf_counter() - t0
    return sum(len(c) for c in work) / dt / 1e9


def run(block_mb: int = DEFAULT_BLOCK_MB, threads: int = 0,
        trials: int = 3) -> dict:
    data = _kv_like(block_mb * 1024 * 1024)
    threads = threads or min(16, os.cpu_count() or 8)
    rows = []
    for name, enc, dec in _codecs():
        r = _time_codec(name, enc, dec, data, trials=trials)
        if "error" not in r:
            r["encode_gb_s_threaded"] = round(_threaded_rate(enc, data, threads), 2)
            # Decode aggregate matters more than encode aggregate: on the
            # asynchronous path decode is the only side on the critical path,
            # and a promote can be split across cores the same way a spill can.
            blob = enc(data[:8 * 1024 * 1024])
            r["decode_gb_s_threaded"] = round(
                _threaded_rate(dec, blob, threads, chunk_mb=8)
                * (8 * 1024 * 1024) / max(1, len(blob)), 2)
            r["threads"] = threads
            # A codec is USABLE ASYNCHRONOUSLY when decode clears the bar; it is
            # usable SYNCHRONOUSLY only when encode clears it too.
            be = r["break_even_gb_s"]
            be3 = r["break_even_vs_t3_gb_s"]
            # Against PCIe: can compressing before a T2->T1 promote beat just
            # sending the bytes? Decode is on the critical path either way.
            r["beats_pcie"] = bool(be and r["decode_gb_s_threaded"] > be)
            # Against T3: can keeping a cold buffer COMPRESSED in DDR beat
            # evicting it and re-reading it from NVMe later? This is what
            # 0020-dma-buf-compressed-descriptor was actually written for, and
            # it is a capacity argument, not a bandwidth one.
            r["beats_t3"] = bool(be3 and r["decode_gb_s"] > be3)
        rows.append(r)
    return {
        "link_gb_s": LINK_GB_S,
        "t3_gb_s": T3_GB_S,
        "block_mb": block_mb,
        "data": "f16 random walk (KV-like: correlated, near-constant exponent)",
        "codecs": rows,
        "verdict": _verdict(rows, []),
    }


def _verdict(rows: list, _unused: list) -> str:
    if not rows:
        return "no codec available to measure"
    ok = [r for r in rows if "error" not in r]
    fastest = max(ok, key=lambda r: r.get("decode_gb_s_threaded", 0), default=None)
    beats_pcie = [r["codec"] for r in ok if r.get("beats_pcie")]
    beats_t3 = sorted((r for r in ok if r.get("beats_t3")),
                      key=lambda r: -r["ratio"])

    lines = []
    if beats_pcie:
        lines.append(f"vs PCIe: viable ({', '.join(beats_pcie)}) , compressing "
                     f"before a T2->T1 promote beats sending the bytes raw.")
    else:
        lines.append(
            f"vs PCIe: REJECTED. Best decode measured is {fastest['codec']} at "
            f"{fastest['decode_gb_s_threaded']} GB/s across {fastest['threads']} "
            f"threads ({fastest['decode_gb_s']} single) against a break-even of "
            f"{fastest['break_even_gb_s']} GB/s. Compression cannot buy PCIe "
            f"bandwidth on this box , the bar is above the link rate by "
            f"construction, and no available codec is that fast. GB-3's "
            f"'compress before the link' framing does not pay.")
    if beats_t3:
        b = beats_t3[0]
        lines.append(
            f"vs T3: VIABLE. {b['codec']} decodes at {b['decode_gb_s']} GB/s "
            f"against a {b['break_even_vs_t3_gb_s']} GB/s bar, at {b['ratio']}x "
            f"ratio. Keeping a cold buffer COMPRESSED in DDR instead of evicting "
            f"it to NVMe is a capacity win ({b['ratio']}x more logical T2) that "
            f"pays for itself on the re-read it avoids. This is what "
            f"0020-dma-buf-compressed-descriptor was written for, and it is the "
            f"form GB-K3 should take , not the bandwidth form.")
    else:
        lines.append("vs T3: also rejected , nothing clears even the NVMe bar.")
    return " ".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(prog="gb_bench_codec.py", description=__doc__)
    p.add_argument("--block-mb", type=int, default=DEFAULT_BLOCK_MB)
    p.add_argument("--threads", type=int, default=0,
                   help="threads for the aggregate encode rate (default: 16, "
                        "this box's E-core count)")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    res = run(a.block_mb, a.threads, a.trials)
    if a.json:
        print(json.dumps(res, indent=1))
        return
    print(f"link {res['link_gb_s']} GB/s   T3 {res['t3_gb_s']} GB/s   block {res['block_mb']} MiB   {res['data']}")
    print()
    print(f"{'codec':>8} {'ratio':>6} {'encode':>9} {'decode':>9} {'break-even':>11} "
          f"{'enc/dec xN':>12} {'>PCIe':>6} {'>T3':>5}")
    for r in res["codecs"]:
        if "error" in r:
            print(f"{r['codec']:>8}  {r['error']}")
            continue
        print(f"{r['codec']:>8} {r['ratio']:>6.2f} {r['encode_gb_s']:>7.2f}GB/s "
              f"{r['decode_gb_s']:>7.2f}GB/s {r['break_even_gb_s']:>9.2f}GB/s "
              f"{r['encode_gb_s_threaded']:>6.2f}/{r['decode_gb_s_threaded']:>6.2f} "
              f"{'yes' if r['beats_pcie'] else 'no':>6} "
              f"{'yes' if r['beats_t3'] else 'no':>5}")
    print()
    print(res["verdict"])


if __name__ == "__main__":
    main()

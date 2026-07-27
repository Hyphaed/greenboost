#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""tests/bench/gb_pathbench.py, runs the compiled gb_pathbench CUDA binary
and forwards each measured path into dataflux via gb_bench.emit_bench_result()
(kind "bench_result", see gb_dataflux_kinds.py and gb_bench.py).

The C++/CUDA binary (gb_pathbench.cu) has zero Python/dataflux dependency by
design, this wrapper is the only piece that touches gb_dataflux, so the
binary itself stays usable standalone under gdb/compute-sanitizer/nsys.

Usage:
    make -C tests/bench pathbench          # builds tests/bench/gb_pathbench
    python3 tests/bench/gb_pathbench.py [--mb 512] [--iters 10] [--tag baseline]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gb_bench  # noqa: E402

BINARY = Path(__file__).parent / "gb_pathbench"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mb", type=int, default=512, help="buffer size per path, MiB")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--tag", default="baseline", help="label for A/B grouping via dataflux_group('bench')")
    args = ap.parse_args()

    if not BINARY.exists():
        print(f"error: {BINARY} not built. Run: make -C tests/bench pathbench", file=sys.stderr)
        return 1

    proc = subprocess.run([str(BINARY), "--mb", str(args.mb), "--iters", str(args.iters)],
                           capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        gb_bench.emit_bench_result(
            path="gb_pathbench", config={"mb": args.mb, "iters": args.iters},
            tag=args.tag, error=proc.stderr.strip()[:500] or "nonzero exit")
        return proc.returncode

    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    config = {"mb": args.mb, "iters": args.iters, "gpu": payload.get("gpu")}
    for row in payload["results"]:
        gb_bench.emit_bench_result(
            path=f"gb_pathbench:{row['name']}", config=config,
            model=payload.get("gpu"), tag=args.tag, bandwidth_gb_s=row["gb_s"])

    # M4 go/no-go (Finding 8 / GB_VRAM_FRONTLOAD safety gate): logged as a
    # bandwidth-less bench_result, "error" carries the CUresult name/message
    # on failure so a later phase can see WHY it failed, not just that it did.
    m4 = payload.get("m4") or {}
    if m4.get("attempted"):
        gb_bench.emit_bench_result(
            path="gb_pathbench:m4_hostnuma_sm_access", config=config,
            model=payload.get("gpu"), tag=args.tag,
            error=None if m4.get("sm_accessible") else (m4.get("error") or "sm_inaccessible"))

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

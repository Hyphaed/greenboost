#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 microbench: does fp8 `scaled_mm` beat `gemlite` on this hardware?

This bench is the one `gb_kernel_backends._auto_policy` has been waiting for.
That function ships `gemlite` for everything and says so:

    "The flip to route fp8 GEMM-shaped layers to `scaled_mm` lives here and is
     intentionally disabled until the Stage-1 microbench proves the win on this
     hardware (owner rule: measured wins only)."

The bench did not exist, so the decision could never be taken either way. This
is it. It answers one question and is allowed to answer "no".

    python3 tests/bench/bench_fp8_backends.py [--json out.json]

What it measures, per shape:

  correctness , cosine of each backend's output against a bf16 reference over
                the SAME inputs. A backend that is faster and wrong is not
                faster. Threshold 0.99, which is the fp8 weight-only floor this
                repo already ships against.
  speed       , wall-clock per forward, CUDA-event timed, warmup discarded.

Shapes are GEMM-class because that is the only case `_auto_policy` would route:
small/odd layers stay on gemlite regardless of the outcome here.

CAVEAT, stated rather than discovered later: this is a per-layer microbench on
synthetic weights. Casebook C-16 is the warning , a 6.3x kernel win there was a
3.5x end-to-end LOSS once the surrounding per-call cost was included, and C-20
is the same lesson from the other side (a microbenchmark at a size the workload
never runs at mis-ranks everything). A win here is permission to run the real
diffusion pipeline, not permission to flip the policy.
"""
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
# gemlite is VENDORED under third_party/ and gb_quant puts it on the path via
# _ensure_vendored_paths(). Without this the bench reports "gemlite available:
# False" and silently compares against bf16 instead of the incumbent , which
# answers a question _auto_policy never asked.
sys.path.insert(0, str(_REPO / "third_party"))

COS_FLOOR = 0.99
#: (in_features, out_features) , GEMM-class only.
SHAPES = ((4096, 4096), (4096, 11008), (11008, 4096))
BATCHES = (1, 16, 512)


def _timeit(fn, iters=30, warmup=10):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def _cos(a, b):
    import torch
    return torch.nn.functional.cosine_similarity(
        a.reshape(1, -1).float(), b.reshape(1, -1).float()).item()


def main(emit: str = "") -> int:
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        print(f"torch unavailable: {exc}")
        return 2
    if not torch.cuda.is_available():
        print("SKIP: needs a CUDA device")
        return 2

    import gb_kernel_backends as kb

    have_scaled = kb._scaled_mm_available()
    try:
        import gemlite  # noqa: F401
        have_gemlite = True
    except Exception:
        have_gemlite = False
    print(f"scaled_mm available: {have_scaled} | gemlite available: {have_gemlite}")
    print(f"current _auto_policy for fp8 GEMM-class: "
          f"{kb.resolve_backend('fp8', 4096, 4096, torch.cuda.get_device_capability(0))}")
    if not have_scaled:
        print("SKIP: torch build has no _scaled_mm/fp8 — nothing to decide")
        return 2

    torch.manual_seed(0)
    dev = "cuda"
    rows, worst_cos, any_win = [], 1.0, False

    print(f"\n{'in':>6} {'out':>6} {'M':>5} {'bf16 ms':>9} {'gemlite':>9} "
          f"{'scaled ms':>10} {'vs gemlite':>11} {'cosine':>9}")
    for in_f, out_f in SHAPES:
        lin = nn.Linear(in_f, out_f, bias=False).to(dev, torch.bfloat16)
        proc = kb.build_scaled_mm_processor("fp8", dev, torch.bfloat16)
        mod = proc.from_linear(lin)
        # The incumbent, built through gb_quant's own path rather than
        # reconstructed here , the point is to measure what actually ships.
        gl_mod, gl_err = None, ""
        if have_gemlite:
            try:
                import gb_quant
                gl_proc = gb_quant._build_processor("fp8", dev, torch.bfloat16, 64)
                gl_mod = gl_proc.from_linear(
                    nn.Linear(in_f, out_f, bias=False).to(dev, torch.bfloat16))
            except Exception as exc:
                gl_err = f"{type(exc).__name__}: {exc}"
                gl_mod = None
        if gl_mod is None and gl_err:
            print(f"  (gemlite processor unavailable for {in_f}x{out_f}: {gl_err[:70]})")
        for M in BATCHES:
            x = torch.randn(M, in_f, device=dev, dtype=torch.bfloat16)
            ref = lin(x)
            got = mod(x)
            cos = _cos(got, ref)
            worst_cos = min(worst_cos, cos)
            t_bf16 = _timeit(lambda: lin(x))
            t_fp8 = _timeit(lambda: mod(x))
            t_gl = _timeit(lambda: gl_mod(x)) if gl_mod is not None else None
            # The decision is against the INCUMBENT. bf16 is context, not the bar.
            if t_gl is not None:
                any_win |= (t_fp8 < t_gl)
            rows.append({"in_f": in_f, "out_f": out_f, "M": M,
                         "bf16_ms": round(t_bf16, 4),
                         "gemlite_ms": round(t_gl, 4) if t_gl else None,
                         "scaled_mm_ms": round(t_fp8, 4),
                         "vs_gemlite": round(t_gl / t_fp8, 3) if t_gl else None,
                         "vs_bf16": round(t_bf16 / t_fp8, 3), "cosine": round(cos, 6)})
            _gl_s = f"{t_gl:>9.3f}" if t_gl else f"{'n/a':>9}"
            _vs = f"{t_gl/t_fp8:>10.2f}x" if t_gl else f"{'n/a':>11}"
            print(f"{in_f:>6} {out_f:>6} {M:>5} {t_bf16:>9.3f} {_gl_s} "
                  f"{t_fp8:>10.3f} {_vs} {cos:>9.6f}")

    if emit:
        Path(emit).write_text(json.dumps({"metrics": {"rows": rows}}, indent=1))
        print(f"\nmeasured set written to {emit}")

    print()
    if worst_cos < COS_FLOOR:
        print(f"FAIL: worst cosine {worst_cos:.6f} < {COS_FLOOR} — scaled_mm is not "
              f"numerically acceptable here, regardless of speed.")
        return 1
    print(f"correctness PASS (worst cosine {worst_cos:.6f} >= {COS_FLOOR})")
    have_cmp = [r for r in rows if r.get("vs_gemlite")]
    if not have_cmp:
        print("INCONCLUSIVE: gemlite could not be constructed, so the incumbent "
              "was never measured. scaled_mm vs bf16 is NOT the question "
              "_auto_policy asks. Do not flip anything on this run.")
        return 0
    best = max(r["vs_gemlite"] for r in have_cmp)
    if not any_win:
        print(f"VERDICT: scaled_mm is SLOWER than gemlite at every shape measured "
              f"(best {best:.2f}x). Leave _auto_policy on gemlite. A negative "
              f"result is a result , record it and stop.")
        return 0
    print(f"VERDICT: scaled_mm beats GEMLITE at one or more shapes (best {best:.2f}x). "
          f"That is permission to bench it against GEMLITE on the real diffusion "
          f"pipeline (ai-forge, GB_ORCH=1), NOT permission to flip _auto_policy: "
          f"see casebook C-16, where a 6.3x kernel win was a 3.5x end-to-end loss.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="bench_fp8_backends")
    ap.add_argument("--json", default="", metavar="PATH")
    raise SystemExit(main(ap.parse_args().json))

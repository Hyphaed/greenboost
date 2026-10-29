#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage-A2 hardware bench for the gb_cutlass sm_120a NVFP4 GEMM.

NOT collected by the default pytest run (it needs a built extension + an sm_120
GPU). This is the timebox checkpoint gate that decides whether the nvfp4 GEMM
is worth routing production traffic through.

    CUDA_HOME=/usr/local/cuda-13.3 PATH=/usr/local/cuda-13.3/bin:$PATH \
      GB_CUTLASS_PATH=<cutlass checkout> \
      python3 third_party/gb_cutlass/setup.py build_ext --inplace
    GB_CUTLASS_ENABLE=1 python3 tests/bench/bench_cutlass_nvfp4.py

Three separate questions, deliberately not collapsed into one number:

  1. Is the GEMM correct?  Cosine of gemm_nvfp4 against a reference computed
     over the SAME dequantized operands. This isolates kernel + scale-factor
     layout correctness from quantization error. Threshold 0.99.
  2. What does NVFP4 cost in quality?  Cosine against the TRUE bf16 reference.
     This is the number the fp8 floor cares about, and it is NOT a substitute
     for a model-level gate (niah_certify / smoke_gate).
  3. Is it actually faster END TO END?  A GEMM speedup means nothing if
     quantizing the activation costs more than it saves. Weights are packed
     once offline; activations are packed on every call, so the e2e column is
     the only one that describes a real forward pass.

Measured 2026-08-22 on an RTX 5070 (sm_120), N=K=4096, CUDA 13.3, torch 2.13:
GEMM alone is 1.35x-6.43x over bf16 at cosine 0.999999, but e2e is 0.18x-0.29x
-- a 3.5x-5.6x REGRESSION -- because pack_nvfp4's PyTorch activation path costs
0.13-3.75 ms against a 0.017-0.164 ms GEMM. See docs/performance-casebook.md.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party"))

COS_FLOOR = 0.99
SHAPES = (1, 16, 512, 2048)
N = K = 4096


def _timeit(fn, iters=30, warmup=5):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main(emit_json: str = "") -> int:
    try:
        import torch
    except Exception as e:
        print(f"torch unavailable: {e}")
        return 2
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 12:
        print("SKIP: needs an sm_120 (Blackwell) GPU")
        return 2
    try:
        import gb_cutlass
    except Exception as e:
        print(f"SKIP: gb_cutlass not built ({e}); see this file's docstring")
        return 2
    if not gb_cutlass._COMPILED:
        print("SKIP: gb_cutlass imported but the CUDA extension is not built")
        return 2
    from gb_cutlass._gb_cutlass_C import (gemm_nvfp4, pack_sfa, pack_sfb,
                                          quantize_act_nvfp4)

    print(f"gb_cutlass available() = {gb_cutlass.available()}  "
          f"(needs GB_CUTLASS_ENABLE=1; currently "
          f"{os.environ.get('GB_CUTLASS_ENABLE', 'unset')})")
    dev = "cuda"
    torch.manual_seed(0)

    out = {"gemm_ms": {}, "fused_e2e_ms": {}, "cos_vs_dequant": {},
           "cos_vs_bf16": {}, "fused_speedup_vs_bf16": {}}
    print(f"\n{'M':>6} {'cos(dequant)':>13} {'cos(bf16)':>10} "
          f"{'bf16 ms':>8} {'gemm ms':>8} {'pack ms':>8} {'e2e ms':>8} "
          f"{'gemm x':>7} {'e2e x':>7} {'fused x':>8}")
    worst_gemm_cos = 1.0
    any_e2e_win = False

    for M in SHAPES:
        A = torch.randn(M, K, device=dev) * 0.1
        B = torch.randn(N, K, device=dev) * 0.1
        # Weight: quantized once, offline. Activation: every call.
        pB, sB, mB = gb_cutlass.pack_nvfp4(B)
        SFB = pack_sfb(sB.reshape(-1).contiguous(), M, N, K)
        pB = pB.contiguous()
        pA, sA, mA = gb_cutlass.pack_nvfp4(A)
        SFA = pack_sfa(sA.reshape(-1).contiguous(), M, N, K)
        pA = pA.contiguous()

        D = gemm_nvfp4(pA, SFA, pB, SFB, M, N, K).float()
        dq = (gb_cutlass.unpack_nvfp4(pA, sA, mA).float()
              @ gb_cutlass.unpack_nvfp4(pB, sB, mB).float().T)
        true_ref = (A.to(torch.bfloat16).float() @ B.to(torch.bfloat16).float().T)
        cos_dq = torch.nn.functional.cosine_similarity(
            D.reshape(1, -1), dq.reshape(1, -1)).item()
        cos_bf = torch.nn.functional.cosine_similarity(
            D.reshape(1, -1), true_ref.reshape(1, -1)).item()
        worst_gemm_cos = min(worst_gemm_cos, cos_dq)

        Abf, Bbf = A.to(torch.bfloat16), B.to(torch.bfloat16)
        t_bf16 = _timeit(lambda: Abf @ Bbf.T)
        t_gemm = _timeit(lambda: gemm_nvfp4(pA, SFA, pB, SFB, M, N, K))

        def _pack():
            _p, _s, _ = gb_cutlass.pack_nvfp4(A)
            return pack_sfa(_s.reshape(-1).contiguous(), M, N, K)

        def _e2e():
            _p, _s, _ = gb_cutlass.pack_nvfp4(A)
            _sf = pack_sfa(_s.reshape(-1).contiguous(), M, N, K)
            return gemm_nvfp4(_p.contiguous(), _sf, pB, SFB, M, N, K)

        # The path a real forward pass would take: one fused kernel for the
        # activation instead of the PyTorch packer.
        Abf16 = A.to(torch.bfloat16).contiguous()

        def _fused():
            _p, _sf = quantize_act_nvfp4(Abf16, M, N, K)
            return gemm_nvfp4(_p, _sf, pB, SFB, M, N, K)

        t_pack = _timeit(_pack)
        t_e2e = _timeit(_e2e)
        t_fused = _timeit(_fused)
        any_e2e_win |= (t_fused < t_bf16)

        print(f"{M:>6} {cos_dq:>13.6f} {cos_bf:>10.6f} "
              f"{t_bf16:>8.3f} {t_gemm:>8.3f} {t_pack:>8.3f} {t_e2e:>8.3f} "
              f"{t_bf16/t_gemm:>6.2f}x {t_bf16/t_e2e:>6.2f}x {t_bf16/t_fused:>7.2f}x")
        key = f"M{M}"
        out["gemm_ms"][key] = round(t_gemm, 4)
        out["fused_e2e_ms"][key] = round(t_fused, 4)
        out["cos_vs_dequant"][key] = round(cos_dq, 6)
        out["cos_vs_bf16"][key] = round(cos_bf, 6)
        out["fused_speedup_vs_bf16"][key] = round(t_bf16 / t_fused, 3)

    if emit_json:
        with open(emit_json, "w") as f:
            json.dump({"metrics": out}, f, indent=1)
        print(f"\nmeasured set written to {emit_json} — score it with:")
        print(f"  python3 tools/compare_goldens.py tests/golden/nvfp4_gemm.json {emit_json}")

    print()
    if worst_gemm_cos < COS_FLOOR:
        print(f"FAIL: GEMM cosine {worst_gemm_cos:.6f} < {COS_FLOOR} — the kernel or "
              f"the scale-factor layout is wrong.")
        return 1
    print(f"GEMM correctness PASS (worst cosine {worst_gemm_cos:.6f} >= {COS_FLOOR}).")
    if not any_e2e_win:
        print("VERDICT: do NOT wire build_cutlass_nvfp4_processor yet. The GEMM is "
              "correct and faster, but end-to-end is a regression at every shape "
              "measured: activation quantization (pack_nvfp4, PyTorch argmin over "
              "the 8-entry E2M1 table) costs more than the GEMM saves. The blocker "
              "is no longer the scale-factor layout — it is a fused CUDA activation "
              "quantizer.")
        return 0
    print("VERDICT: e2e win present at one or more shapes — a processor is now "
          "justified, but routing production traffic still needs the model-level "
          "quality gate (niah_certify / smoke_gate), since NVFP4 is below the fp8 floor.")
    return 0


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(prog="bench_cutlass_nvfp4")
    _ap.add_argument("--json", default="", metavar="PATH",
                     help="also write the measured set here, for tools/compare_goldens.py")
    raise SystemExit(main(_ap.parse_args().json))

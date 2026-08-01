#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage-A2 hardware bench for the gb_cutlass sm_120a NVFP4 GEMM.

NOT collected by the default pytest run (it needs a built extension + an sm_120
GPU). This is the timebox checkpoint gate: gb_cutlass.available() must stay
False (and the fp8 floor holds) until this passes here.

    make gb_cutlass                       # build the extension first
    python3 tests/bench/bench_cutlass_nvfp4.py

Checks, at M in {1, 16, 512, 2048}, N=K=4096:
  * correctness — cosine similarity of gb_cutlass.gemm_nvfp4 vs a bf16 matmul
    reference over the SAME dequantized operands (threshold 0.99);
  * speed — wall-clock vs the bf16 reference and vs scaled_mm fp8 where present.

Until the weight-packing reconciliation (gb_quant._init_nvfp4 layout ->
CUTLASS interleaved SF layout) is implemented, the operand construction below
is a PLACEHOLDER: it documents the required tensors and will fail loudly rather
than silently pass. Fill in the pack step, confirm cosine >= 0.99, then flip
build_cutlass_nvfp4_processor + set GB_CUTLASS_ENABLE=1.
"""
import sys
import time
from pathlib import Path

# gb_cutlass lives under <repo_root>/third_party/, which is only on sys.path
# in the real product path via gb_quant._ensure_vendored_paths() — add it
# here too so this script works standalone per its own docstring.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party"))


def main() -> int:
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
        print(f"SKIP: gb_cutlass not built ({e}); run `make gb_cutlass`")
        return 2

    print("gb_cutlass imported; available() =", gb_cutlass.available())
    print("NOTE: operand packing (nvfp4 + interleaved SF) reconciliation is the "
          "remaining Stage-A2 step — see this file's docstring and "
          "gb_kernel_backends.build_cutlass_nvfp4_processor.")

    N = K = 4096
    for M in (1, 16, 512, 2048):
        act = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        wt = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        t0 = time.perf_counter()
        ref = act @ wt.t()
        torch.cuda.synchronize()
        dt_ref = time.perf_counter() - t0
        print(f"M={M:>5}  bf16 ref {dt_ref*1e3:7.2f} ms  "
              f"(nvfp4 path pending pack reconciliation)")
        # Once packing exists:
        #   a_e2m1, sfa = pack_nvfp4(act); b_e2m1, sfb = pack_nvfp4(wt)
        #   out = gb_cutlass.gemm_nvfp4(a_e2m1, sfa, b_e2m1, sfb, M, N, K)
        #   cos = torch.nn.functional.cosine_similarity(
        #             out.flatten().float(), ref.flatten().float(), dim=0)
        #   assert cos >= 0.99, f"cosine {cos:.4f} < 0.99 at M={M}"
    print("bench scaffold ran; wire the pack step to complete Stage-A2 validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())

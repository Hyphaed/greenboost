# SPDX-License-Identifier: Apache-2.0
"""Build the gb_cutlass sm_120a NVFP4 GEMM torch extension.

Opt-in only — never built by CI or `make install`. Run on the target box:

    GB_CUTLASS_PATH=$HOME/Dev/greenboost_all/vendor/cutlass \\
        python3 third_party/gb_cutlass/setup.py build_ext --inplace

Requires: a torch built with CUDA 12.8+ (Blackwell / sm_120 support) and a
matching nvcc. CUTLASS is header-only; point GB_CUTLASS_PATH at the checkout
(default: the vendored $HOME/Dev/greenboost_all/vendor/cutlass).
"""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_HOME = os.path.expanduser("~")
CUTLASS = os.environ.get(
    "GB_CUTLASS_PATH", os.path.join(_HOME, "Dev/greenboost_all/vendor/cutlass"))

setup(
    name="gb_cutlass",
    ext_modules=[
        CUDAExtension(
            name="gb_cutlass._gb_cutlass_C",
            sources=["gb_cutlass_ext.cu"],
            include_dirs=[
                os.path.join(CUTLASS, "include"),
                os.path.join(CUTLASS, "tools/util/include"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3", "-std=c++17",
                    # GeForce Blackwell block-scaled tensor ops require the
                    # architecture-specific 'a' target.
                    "-gencode", "arch=compute_120a,code=sm_120a",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

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
_HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS = os.environ.get(
    "GB_CUTLASS_PATH", os.path.join(_HOME, "Dev/greenboost_all/vendor/cutlass"))

# `build_ext --inplace` resolves both the .cu source path and its output
# .so path against the process cwd, not this script's directory — the
# Makefile/docstring invoke this from the repo root, so both would otherwise
# land in (or look for) the wrong place. The extension's dotted name
# ("gb_cutlass._gb_cutlass_C") must resolve to a `gb_cutlass/` package dir
# under cwd, and `__init__.py` does `from ._gb_cutlass_C import ...` (a
# sibling file within this same directory) — so cwd needs to be this
# script's PARENT (third_party/), not this directory itself.
os.chdir(os.path.dirname(_HERE))

setup(
    name="gb_cutlass",
    ext_modules=[
        CUDAExtension(
            name="gb_cutlass._gb_cutlass_C",
            sources=["gb_cutlass/gb_cutlass_ext.cu"],
            include_dirs=[
                os.path.join(CUTLASS, "include"),
                os.path.join(CUTLASS, "tools/util/include"),
            ],
            extra_compile_args={
                # torch nightlies (2.14+) require a C++20-capable compiler.
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": [
                    "-O3", "-std=c++20",
                    # nvcc's -ccbin defaults to plain 'gcc' on PATH regardless
                    # of CC/CXX env vars, which this box's default gcc is too
                    # new for (nvcc 12.x supported up to gcc 13; verify
                    # against the nvcc in use if this pins a different gcc).
                    "-ccbin", os.environ.get("CXX", "g++-13"),
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

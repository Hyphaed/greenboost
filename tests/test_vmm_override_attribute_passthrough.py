# SPDX-License-Identifier: GPL-2.0-only
"""
Regression: with libgreenboost_vmm_override.so preloaded, cuDeviceGetAttribute
must return the driver's real value for every attribute except VMM-supported
on Blackwell. It used to return 0 for all of them (Triton then failed with
"shared memory ... Hardware limit: 0" under `greenboost run`): the override's
internal lookup of the real function went through its own dlsym() wrapper,
cached itself as the "real" function, and its recursion guard answered 0.

CPU only: a stub libcuda.so.1 stands in for the driver. Needs gcc.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

STUB = r"""
int cuDeviceGetAttribute(int *v, int a, int d)
{ *v = (a == 97) ? 49152 : (a == 75) ? 12 : (a == 102) ? 1 : 7; return 0; }
"""
PROBE = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
int main(int c, char **v) {
    void *h = dlopen(v[1], RTLD_NOW | RTLD_GLOBAL);
    int (*f)(int *, int, int) = dlsym(h, "cuDeviceGetAttribute");
    int smem = -1, cc = -1, vmm = -1;
    f(&smem, 97, 0); f(&cc, 75, 0); f(&vmm, 102, 0);
    printf("%d %d %d\n", smem, cc, vmm);
    return 0;
}
"""


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")
def test_vmm_override_passes_real_attributes_through(tmp_path):
    (tmp_path / "stub.c").write_text(STUB)
    (tmp_path / "probe.c").write_text(PROBE)
    cc = ["gcc", "-fPIC", "-O2", "-std=gnu11"]
    for args in (
        cc + ["-shared", "-o", str(tmp_path / "libcuda.so.1"), str(tmp_path / "stub.c")],
        cc + ["-o", str(tmp_path / "probe"), str(tmp_path / "probe.c"), "-ldl"],
        cc + ["-shared", "-o", str(tmp_path / "vo.so"),
              str(REPO / "greenboost_vmm_override.c"), "-ldl"],
    ):
        subprocess.run(args, check=True, capture_output=True)
    env = {"LD_LIBRARY_PATH": str(tmp_path), "LD_PRELOAD": str(tmp_path / "vo.so")}
    out = subprocess.run([str(tmp_path / "probe"), str(tmp_path / "libcuda.so.1")],
                         env=env, check=True, capture_output=True, text=True).stdout
    smem, cc_major, vmm = (int(x) for x in out.split())
    assert smem == 49152, "shared memory per block must be the driver's value, not 0"
    assert cc_major == 12
    assert vmm == 0, "VMM-supported is still forced to 0 on Blackwell"

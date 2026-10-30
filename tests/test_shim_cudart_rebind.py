"""Regression for the cudart override rebind (2026-10-30): with GREENBOOST_CUDART_PATH at a cu13 library inside a cu12 process the shim warned that it was
rebinding onto the app's libcudart, but gb_cudart_resolve_syms() only fills NULL pointers, so every pointer stayed on cu13: cudaGetDeviceProperties returned the
cu13 struct layout, torch read multiProcessorCount 0 (SIGFPE in the RNG) and fp16 convolutions produced NaN. The fix is landed as a patch file (the shim is edited in a copy);
this test fails if the patch stops applying and the source does not carry the fix, and the optional GPU case proves it on the card."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PATCH = REPO / "patches" / "shim-cudart-rebind-force.patch"
SRC = REPO / "greenboost_cuda_shim.c"


def _fixed(text: str) -> bool:
    return "gb_cudart_resolve_syms(h, 1)" in text and "force || !real_cudaGetDeviceProperties" in text


def test_fix_is_in_source_or_patch_applies():
    assert PATCH.exists()
    if _fixed(SRC.read_text()):
        return
    r = subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_rebind_forces_every_pointer_in_the_patched_source(tmp_path):
    text = SRC.read_text()
    if not _fixed(text):
        shutil.copy(SRC, tmp_path / SRC.name)
        subprocess.run(["patch", "-s", "-p1", "-i", str(PATCH)], cwd=tmp_path, check=True)
        text = (tmp_path / SRC.name).read_text()
    a = text.index("static void gb_cudart_resolve_syms(void *libcudart, int force)")
    body = text[a:text.index("static void gb_cudart_rebind(void)")]
    assert "if (!real_" not in body, "a pointer is still fill-if-null only and would survive a forced rebind"
    assert text.count("gb_cudart_resolve_syms(h, 1)") == 2


@pytest.mark.skipif(not os.environ.get("GB_TEST_CU12_PYTHON") or not os.environ.get("GB_TEST_SHIM"), reason="needs a cu12 torch python (GB_TEST_CU12_PYTHON), the patched shim build (GB_TEST_SHIM) and a GPU")
def test_cu13_override_in_cu12_process_reports_the_real_sm_count():
    py, shim = os.environ["GB_TEST_CU12_PYTHON"], os.environ["GB_TEST_SHIM"]
    cu13 = subprocess.run([py, "-c", "import glob,sys,site;print((glob.glob(site.getsitepackages()[0]+'/nvidia/cu13/lib/libcudart.so.13') or [''])[0])"], capture_output=True, text=True).stdout.strip()
    if not cu13:
        pytest.skip("no cu13 libcudart next to the cu12 torch")
    env = {**os.environ, "LD_PRELOAD": shim, "GREENBOOST_CUDART_PATH": cu13, "GREENBOOST_REPORT_PHYSICAL_VRAM": "1"}
    r = subprocess.run([py, "-c", "import torch;print(torch.cuda.get_device_properties(0).multi_processor_count)"], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-300:]
    assert int(r.stdout.split()[-1]) > 0

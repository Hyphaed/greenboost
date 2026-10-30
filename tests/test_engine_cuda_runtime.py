"""An engine compiled with one CUDA toolkit and linked against another toolkit's libcudart reads cudaDeviceProp in the wrong layout: 1 SM, a 4 GiB shared-memory limit, and every
kernel that sizes dynamic shared memory fails with 'invalid argument' (found 2026-10-30 on the expert-cache engine: nvcc 13.4 headers + /usr/lib libcudart.so.12.4)."""
import types
from pathlib import Path

import pytest

import gb_synapse as gs


def _fake(monkeypatch, needed, release):
    def run(cmd, capture=False, check=True, **kw):
        out = f"0x1 (NEEDED) Shared library: [libcudart.so.{needed}]" if cmd[0] == "readelf" else f"Cuda compilation tools, release {release}, V{release}.92"
        return types.SimpleNamespace(stdout=out, stderr="", returncode=0)
    monkeypatch.setattr(gs, "_run", run)
    monkeypatch.setattr(gs.shutil, "which", lambda x: "/usr/bin/" + Path(str(x)).name)


def test_mixed_toolkit_is_refused(tmp_path, monkeypatch):
    (tmp_path / "libggml-cuda.so.0").write_bytes(b"")
    _fake(monkeypatch, 12, "13.4")
    with pytest.raises(RuntimeError, match="links libcudart.so.12 but nvcc is CUDA 13"):
        gs.check_engine_cuda_runtime(tmp_path)


def test_matching_toolkit_passes(tmp_path, monkeypatch):
    (tmp_path / "libggml-cuda.so.0").write_bytes(b"")
    _fake(monkeypatch, 13, "13.4")
    assert gs.check_engine_cuda_runtime(tmp_path) == "13"


def test_no_library_is_not_checkable(tmp_path, monkeypatch):
    _fake(monkeypatch, 12, "13.4")
    assert gs.check_engine_cuda_runtime(tmp_path) == ""


def test_toolkit_root_follows_nvcc(tmp_path, monkeypatch):
    (tmp_path / "bin").mkdir()
    (tmp_path / "include").mkdir()
    (tmp_path / "include/cuda_runtime_api.h").write_text("")
    nvcc = tmp_path / "bin/nvcc"
    nvcc.write_text("")
    monkeypatch.setattr(gs, "_find_nvcc", lambda: str(nvcc))
    assert f"-DCUDAToolkit_ROOT={tmp_path.resolve()}" in gs._cuda_toolkit_root_args()


def test_stale_cuda_cache_is_dropped_only_when_it_points_outside_the_toolkit(tmp_path):
    import gb_synapse as gs
    (tmp_path / "CMakeFiles").mkdir()
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text("CUDA_cudart_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libcudart.so\n")
    if not gs._cuda_toolkit_root_args():
        return                                            # no nvcc toolkit on this box: nothing to compare with
    assert gs._drop_stale_cuda_cache(tmp_path) and not cache.exists()
    root = gs._cuda_toolkit_root_args()[0].split("=", 1)[1]
    cache.write_text(f"CUDA_cudart_LIBRARY:FILEPATH={root}/lib64/libcudart.so\n")
    assert not gs._drop_stale_cuda_cache(tmp_path) and cache.exists()

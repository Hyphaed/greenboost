# SPDX-License-Identifier: GPL-2.0-only
"""host_memory() and shim_pid_peaks() in gb_vitals_helper (pure file readers)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gb_vitals_helper as v


def test_host_memory_reads_absolute_headroom(tmp_path):
    (tmp_path / "pressure").mkdir()
    (tmp_path / "meminfo").write_text(
        "MemTotal: 64000000 kB\\nMemAvailable: 20480000 kB\\nSwapFree: 10240000 kB\\n"
        .replace("\\n", "\n"))
    (tmp_path / "pressure" / "memory").write_text(
        "some avg10=3.22 avg60=3.78 avg300=1.85 total=1\n"
        "full avg10=1.50 avg60=3.73 avg300=1.80 total=1\n")
    m = v.host_memory(str(tmp_path))
    assert m == {"mem_available_mb": 20000, "swap_free_mb": 10000,
                 "psi_some_avg10": 3.22, "psi_full_avg10": 1.5}


def test_host_memory_missing_files_gives_zeros(tmp_path):
    assert v.host_memory(str(tmp_path))["mem_available_mb"] == 0


def test_shim_pid_peaks_marks_alive_and_dead(tmp_path):
    (tmp_path / f"shim_stats.{os.getpid()}").write_text(
        "phase=x\ntier_t2_local_peak_mb=512\ntier_t1_local_cur_mb=9\n")
    (tmp_path / "shim_stats.4194000").write_text("tier_t2_local_peak_mb=7\n")
    (tmp_path / "shim_stats").write_text("tier_t2_local_peak_mb=1\n")
    out = v.shim_pid_peaks(str(tmp_path))
    assert out[str(os.getpid())] == {"tier_t2_local_peak_mb": 512, "alive": True}
    assert out["4194000"]["alive"] is False
    assert set(out) == {str(os.getpid()), "4194000"}

import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import gb_snapshot


def _metrics(run, pid, phase, t2=0):
    (run / f"metrics.{pid}.json").write_text(json.dumps({
        "pid": pid, "timestamp": 100, "phase": phase,
        "tiers": {"t1_local": {"cur_mb": 5, "peak_mb": 6}, "t2_local": {"cur_mb": t2, "peak_mb": t2}}}))


def test_per_process_tiers_live_only_and_nulls(tmp_path, monkeypatch):
    monkeypatch.setenv("GB_LEASE_DIR", str(tmp_path / "nolease"))      # no arbiter state: the lease block is all null
    _metrics(tmp_path, os.getpid(), "INFERENCE", t2=70)
    _metrics(tmp_path, 4000000, "INFERENCE", t2=999)      # dead pid: skipped
    s = gb_snapshot.build_snapshot(12227, 9195, tmp_path, now=110)
    assert [p["pid"] for p in s["tier_processes"]] == [os.getpid()]
    assert s["tiers_mib"] == {"t1": 5, "t2": 70, "t3": 0}
    v = s["vram"]
    assert v["used_mib"] + v["free_mib"] == v["physical_mib"] and v["virtual_mib"] is None
    assert all(x is None for x in s["lease"].values())


def test_init_phase_tiers_are_null_not_zero(tmp_path):
    _metrics(tmp_path, os.getpid(), "INIT")
    s = gb_snapshot.build_snapshot(12227, 9195, tmp_path, now=110)
    assert s["tiers_mib"] is None and s["tier_processes"][0]["tiers_mib"] is None


def test_unread_card_is_null_and_write_is_atomic(tmp_path):
    s = gb_snapshot.build_snapshot(0, 0, tmp_path)
    assert s["vram"]["free_mib"] is None
    out = gb_snapshot.write_snapshot(s, tmp_path / "d" / "snapshot.json")
    assert json.loads(out.read_text())["version"] == 1
    assert oct(out.stat().st_mode)[-3:] == "644" and len(list(out.parent.iterdir())) == 1

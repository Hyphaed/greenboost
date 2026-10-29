"""gb_control writes `control` and the OS-tune baseline as root into a directory other accounts can touch (SEC-03):
random O_EXCL|O_NOFOLLOW temp names, no symlink following. Tmp dirs only; never the real /run/greenboost."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import gb_control


def _ctl(monkeypatch, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(gb_control, "_CONTROL_FILE", run / "control")
    monkeypatch.setattr(gb_control, "_CONTROL_TMP", run / ".control.tmp")
    monkeypatch.setattr(gb_control, "_OS_BASELINE_FILE", run / "os_tune_baseline.json")
    return gb_control.GbControl(dry_run=True), run


def test_control_is_merged_and_only_the_final_file_remains(monkeypatch, tmp_path):
    c, run = _ctl(monkeypatch, tmp_path)
    assert c._write_control(a=1)
    assert c._write_control(b=2)
    assert (run / "control").read_text() == "a=1\nb=2\n"
    assert [p.name for p in run.iterdir()] == ["control"]


def test_a_symlink_at_the_old_fixed_temp_name_is_never_followed(monkeypatch, tmp_path):
    c, run = _ctl(monkeypatch, tmp_path)
    victim = tmp_path / "victim"
    victim.write_text("keep")
    (run / ".control.tmp").symlink_to(victim)
    assert c._write_control(a=1)
    assert victim.read_text() == "keep" and (run / "control").read_text() == "a=1\n"


def test_a_symlink_planted_at_control_is_refused_and_its_target_untouched(monkeypatch, tmp_path):
    c, run = _ctl(monkeypatch, tmp_path)
    victim = tmp_path / "victim"
    victim.write_text("SECRET=1\n")
    (run / "control").symlink_to(victim)
    assert c._write_control(a=1) is False
    assert victim.read_text() == "SECRET=1\n"


def test_baseline_capture_is_atomic_first_value_wins_and_refuses_a_symlink(monkeypatch, tmp_path):
    c, run = _ctl(monkeypatch, tmp_path)
    c._capture_baseline("swappiness", 60)
    c._capture_baseline("swappiness", 10)
    assert json.loads((run / "os_tune_baseline.json").read_text()) == {"swappiness": 60}
    assert [p.name for p in run.iterdir()] == ["os_tune_baseline.json"]
    (run / "os_tune_baseline.json").unlink()
    victim = tmp_path / "victim"
    victim.write_text("keep")
    (run / "os_tune_baseline.json").symlink_to(victim)
    c._capture_baseline("x", 1)
    assert victim.read_text() == "keep"
    assert c.restore_baseline() == {}                     # a symlinked baseline is not read as root either

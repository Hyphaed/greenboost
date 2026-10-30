"""Uninstall completeness for the v3.5 additions (round 2): module list derivation, engine dirs, parity guard."""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "checks"))
import check_installer_parity as cip  # noqa: E402


def test_repo_is_clean():
    assert [f for f in cip.run(ROOT) if f.severity == "blocking"] == []


def _mini_repo(tmp_path, im_text=None, setup_text=None):
    for n in ("install_module.sh", "greenboost_setup.sh", "gb_synapse.py", "gb_sdpa.py", "gb_quant.py"):
        shutil.copy(ROOT / n, tmp_path / n)
    shutil.copytree(ROOT / "checks" / "allowlists", tmp_path / "checks" / "allowlists", dirs_exist_ok=True)
    if im_text is not None:
        (tmp_path / "install_module.sh").write_text(im_text)
    if setup_text is not None:
        (tmp_path / "greenboost_setup.sh").write_text(setup_text)
    return tmp_path


def test_catches_module_missing_from_full_installer(tmp_path):
    im = (ROOT / "install_module.sh").read_text().replace("gb_sdpa.py", "gb_sdpa.py gb_quant.py gb_ghost_zzz.py", 1)
    fs = cip.run(_mini_repo(tmp_path, im_text=im))
    assert any("gb_ghost_zzz.py" in f.message for f in fs)


def test_catches_uninstall_not_derived(tmp_path):
    im = (ROOT / "install_module.sh").read_text().replace("for _f in $GB_ALL_PY; do _gb_rm", 'for _f in "${GB_PY_FILES[@]}"; do _gb_rm')
    fs = cip.run(_mini_repo(tmp_path, im_text=im))
    assert any("not derived" in f.message for f in fs)


def test_catches_engine_dir_not_removed(tmp_path):
    im = (ROOT / "install_module.sh").read_text().replace("src-moecache", "src-other")
    fs = cip.run(_mini_repo(tmp_path, im_text=im))
    assert any("src-moecache" in f.message for f in fs)
    st = (ROOT / "greenboost_setup.sh").read_text().replace("rm -rf /var/lib/greenboost/synapse/src-moecache", "true")
    fs = cip.run(_mini_repo(tmp_path, setup_text=st))
    assert any("do_purge never removes" in f.message for f in fs)


def test_uninstall_dry_run_lists_new_modules_and_engine():
    out = subprocess.run(["bash", str(ROOT / "install_module.sh"), "--uninstall", "--dry-run"], capture_output=True, text=True, timeout=60).stdout
    for needle in ("gb_moe_cache.py", "gb_sdpa.py", "gb_quant_conv.py", "gb_compile.py", "gb_cnn_tune.py", "synapse-moecache", "src-moecache", "synapse/engine"):
        assert needle in out, needle

"""The initramfs hook that keeps greenboost.ko out of the boot image.

Incident 2026-08-21: MODULES=most made initramfs-tools bake greenboost.ko into
the initrd, whose own systemd-modules-load inserted it 7.4 s before
switch-root. A reinstall rebuilds /lib/modules and never touches the initrd, so
the box booted v3.2 across several reboots while v3.4 was installed, and T3
failed open every boot because /var was not mounted that early.

The hook is the fix, and it deletes files inside $DESTDIR , which is exactly
the kind of thing that must not be allowed to drift into deleting too much or
too little without a test noticing.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parent.parent / "initramfs" / "zz-greenboost-exclude"
KVER = "7.1.9-hyphaed"


def _run(destdir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(HOOK), *args], capture_output=True, text=True,
                          timeout=30, env={"DESTDIR": str(destdir), "PATH": "/usr/bin:/bin"})


@pytest.fixture()
def image(tmp_path: Path) -> Path:
    """A DESTDIR shaped like a real mkinitramfs staging tree."""
    d = tmp_path / "destdir"
    for sub in (f"lib/modules/{KVER}/updates/dkms",
                f"usr/lib/modules/{KVER}/kernel/drivers",
                "etc/modules-load.d", "etc/tmpfiles.d", "usr/lib/tmpfiles.d"):
        (d / sub).mkdir(parents=True)
    (d / f"lib/modules/{KVER}/updates/dkms/greenboost.ko").write_text("ko")
    # a compressed variant, since distros differ on module compression
    (d / f"usr/lib/modules/{KVER}/kernel/drivers/greenboost.ko.zst").write_text("ko")
    (d / "etc/modules-load.d/greenboost.conf").write_text("greenboost\n")
    (d / "etc/tmpfiles.d/greenboost-gaming.conf").write_text("z /sys/... 0664 root greenboost -\n")
    # bystanders that must survive untouched
    (d / f"lib/modules/{KVER}/updates/dkms/nvidia.ko").write_text("nvidia")
    (d / "etc/modules-load.d/nvidia.conf").write_text("nvidia\n")
    (d / "usr/lib/tmpfiles.d/systemd.conf").write_text("d /run/systemd 0755 root root -\n")
    return d


def test_the_module_is_removed_in_every_shape_it_ships_in(image: Path):
    assert _run(image).returncode == 0
    assert not list(image.rglob("greenboost.ko*")), \
        "a greenboost module survived into the image , this is the whole bug"


def test_the_autoload_fragment_is_removed(image: Path):
    """Belt and braces: with no fragment, early userspace has nothing telling
    it to load a module even if some other path drops a .ko back in."""
    _run(image)
    assert not (image / "etc/modules-load.d/greenboost.conf").exists()


def test_the_gaming_tmpfiles_rule_is_removed(image: Path):
    """It group-scopes a file under /sys/module/greenboost/ that cannot exist
    before the module is loaded, and failed loudly on every boot with
    "Failed to resolve group 'greenboost'" , the initrd has its own
    /etc/group and the group is not in it."""
    _run(image)
    assert not (image / "etc/tmpfiles.d/greenboost-gaming.conf").exists()


def test_nothing_else_is_touched(image: Path):
    """A hook with rm -f and a glob is one typo away from emptying the image."""
    _run(image)
    survivors = {p.relative_to(image).as_posix() for p in image.rglob("*") if p.is_file()}
    assert survivors == {
        f"lib/modules/{KVER}/updates/dkms/nvidia.ko",
        "etc/modules-load.d/nvidia.conf",
        "usr/lib/tmpfiles.d/systemd.conf",
    }


def test_prereqs_mode_exits_clean(image: Path):
    """initramfs-tools calls every hook with `prereqs` first; a hook that acts
    on that invocation runs twice per build."""
    r = _run(image, "prereqs")
    assert r.returncode == 0
    assert (image / f"lib/modules/{KVER}/updates/dkms/greenboost.ko").exists(), \
        "the hook deleted files during its prereqs invocation"


def test_no_destdir_is_a_safe_no_op():
    """Never operate on the live root filesystem when DESTDIR is unset."""
    r = subprocess.run([str(HOOK)], capture_output=True, text=True, timeout=30,
                       env={"PATH": "/usr/bin:/bin"})
    assert r.returncode == 0


def test_it_is_idempotent(image: Path):
    _run(image)
    assert _run(image).returncode == 0, "second run on an already-clean image failed"

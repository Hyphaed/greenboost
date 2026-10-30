"""The stale-module check must match the MODULE, not the word.

`gb_initramfs_verify_clean` grepped the initramfs listing for the bare string
"greenboost". The exclusion machinery puts its own name into the image —
dracut records installed modules and ours is `99greenboost-exclude` — so the
check reported "stale module present" on an image it had just cleaned, on every
install. Observed twice on 2026-08-22.

Same shape as casebook C-13: a verification step failing the thing it exists to
protect. The consequence is worse than noise, because the warning it prints is
real and important when it IS true — a stale greenboost.ko in the boot image
silently disables T3 (C-05). A check that cries wolf on every install is how a
true warning gets ignored.
"""
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SETUP = REPO / "greenboost_setup.sh"


@pytest.fixture(scope="module")
def pattern():
    """The exact grep -E pattern the installer uses, pulled from source."""
    src = SETUP.read_text()
    m = re.search(r"grep -qE '([^']+)' <<<\"\$listing\"", src)
    assert m, "verify pattern not found in greenboost_setup.sh"
    return m.group(1)


def _match(pattern, line):
    r = subprocess.run(["grep", "-qE", pattern], input=line, text=True)
    return r.returncode == 0


@pytest.mark.parametrize("line", [
    "usr/lib/modules/7.1.9-hyphaed/updates/dkms/greenboost.ko",
    "lib/modules/7.1.9/extra/greenboost.ko",
    "greenboost.ko",
    "usr/lib/modules/7.1.9/updates/dkms/greenboost.ko.xz",
    "usr/lib/modules/7.1.9/updates/dkms/greenboost.ko.zst",
])
def test_a_real_stale_module_is_detected(pattern, line):
    assert _match(pattern, line), f"{line} must be reported as stale"


@pytest.mark.parametrize("line", [
    # The exclusion machinery's own footprint — the false positives.
    "usr/lib/dracut/modules.d/99greenboost-exclude/module-setup.sh",
    "etc/dracut.conf.d/99-greenboost-exclude.conf",
    "etc/initramfs-tools/hooks/zz-greenboost-exclude",
    # Other GreenBoost files that are not the kernel module.
    "etc/tmpfiles.d/greenboost-gaming.conf",
    "usr/local/lib/libgreenboost_cuda.so",
    "etc/modules-load.d/greenboost.conf",
    "usr/lib/dracut/modules.d/99greenboost-exclude",
])
def test_the_exclusion_machinery_is_not_mistaken_for_the_module(pattern, line):
    assert not _match(pattern, line), f"{line} must NOT be reported as stale"


def test_the_check_is_not_the_bare_word(pattern):
    """Guard against a future simplification back to the broken form."""
    assert pattern != "greenboost"
    assert "greenboost" in pattern and ".ko" in pattern

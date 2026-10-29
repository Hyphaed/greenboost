#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for greenboost_setup.sh's initramfs refresh helpers.

These exercise the real bash functions against this machine's real /boot and
/lib/modules, not a mocked filesystem. That is deliberate: the bug they guard
against only exists because the old code assumed every kernel under
/lib/modules has a boot image, and a mock built from that same assumption would
have passed.

Observed on this box 2026-08-22: 9 kernels under /lib/modules, 2 with an actual
initrd. `dracut --force --regenerate-all` therefore failed on 6 "ghost" kernels
(module tree, no image; dracut falls back to a BLS/ESP path that does not
exist), which dragged the whole invocation's exit status down and printed a
"dracut failed , the boot image may still hold an old greenboost.ko" warning
even though the RUNNING kernel had regenerated fine. The warning's remediation
was the same failing command.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
SETUP = REPO / "greenboost_setup.sh"

FUNCS = (
    "gb_initramfs_generator",
    "gb_initramfs_kernels",
    "gb_initramfs_image_for",
    "gb_initramfs_verify_clean",
    "gb_initramfs_retry_cmd",
)


@pytest.fixture(scope="module")
def prelude():
    """The helper functions, extracted from the real installer."""
    src = SETUP.read_text()
    out = []
    for fn in FUNCS:
        m = re.search(rf"^{fn}\(\) \{{.*?^\}}", src, re.S | re.M)
        assert m, f"{fn} missing from greenboost_setup.sh"
        out.append(m.group(0))
    return "\n\n".join(out)


def run(prelude, script):
    r = subprocess.run(["bash", "-c", prelude + "\n" + script],
                       capture_output=True, text=True, timeout=60)
    return r


# ── the ghost-kernel bug ──────────────────────────────────────────────────────

def test_only_kernels_with_a_real_image_are_targeted(prelude):
    """A module tree alone is not a kernel worth regenerating."""
    r = run(prelude, 'gb_initramfs_kernels')
    listed = [k for k in r.stdout.split() if k]
    assert listed, "no kernels found , /boot has no initrd at all?"

    for kver in listed:
        has_image = (Path(f"/boot/initrd.img-{kver}").is_file()
                     or Path(f"/boot/initramfs-{kver}.img").is_file())
        assert has_image, f"{kver} listed but has no boot image"
        assert Path(f"/lib/modules/{kver}").is_dir(), f"{kver} has no module tree"


def test_ghost_kernels_are_excluded(prelude):
    """Every module tree WITHOUT an image must be skipped, not attempted.

    This is the regression guard. If someone reverts to iterating
    /lib/modules/*, the ghosts come back and the install warns spuriously."""
    trees = {p.name for p in Path("/lib/modules").iterdir() if p.is_dir()}
    listed = set(run(prelude, "gb_initramfs_kernels").stdout.split())

    ghosts = {k for k in trees - listed
              if not Path(f"/boot/initrd.img-{k}").is_file()
              and not Path(f"/boot/initramfs-{k}.img").is_file()}
    # Whatever the ghosts are on this box, none may appear in the target list.
    assert ghosts.isdisjoint(listed)
    # And nothing in the target list may lack an image.
    assert all(Path(f"/boot/initrd.img-{k}").is_file()
               or Path(f"/boot/initramfs-{k}.img").is_file() for k in listed)


def test_running_kernel_is_targeted_when_it_has_an_image(prelude):
    import platform
    running = platform.release()
    if not (Path(f"/boot/initrd.img-{running}").is_file()
            or Path(f"/boot/initramfs-{running}.img").is_file()):
        pytest.skip(f"running kernel {running} has no boot image on this box")
    assert running in run(prelude, "gb_initramfs_kernels").stdout.split()


# ── the retry command must not be the command that just failed ────────────────

def test_retry_command_is_targeted_not_regenerate_all(prelude):
    """The old warning told the operator to run the exact invocation that had
    just failed. The retry must name one kernel."""
    out = run(prelude, 'gb_initramfs_retry_cmd "$(uname -r)"').stdout.strip()
    assert out, "retry command is empty"
    assert "--regenerate-all" not in out
    assert "-k all" not in out
    assert "mkinitcpio -P" not in out
    import platform
    assert platform.release() in out


def test_retry_command_matches_the_detected_generator(prelude):
    gen = run(prelude, "gb_initramfs_generator").stdout.strip()
    cmd = run(prelude, "gb_initramfs_retry_cmd").stdout.strip()
    expected = {"dracut": "dracut", "initramfs-tools": "update-initramfs",
                "mkinitcpio": "mkinitcpio"}.get(gen)
    if expected is None:
        pytest.skip(f"unrecognised generator {gen!r}")
    assert cmd.startswith(expected)


# ── verification must never claim clean without looking ───────────────────────

def test_verify_never_reports_clean_without_a_listing(prelude):
    """rc 0 means 'checked, and clean'. Unreadable must be rc 2, not rc 0.

    Unprivileged, /boot/initrd.img-* is typically unreadable, so this asserts
    the conservative path is the one taken. Reporting 'clean' from an empty
    listing is how a stale module ships unnoticed."""
    import platform
    rc = run(prelude, f'gb_initramfs_verify_clean "{platform.release()}"; echo rc=$?')
    code = int(re.search(r"rc=(\d+)", rc.stdout).group(1))
    assert code in (0, 1, 2)
    if code == 0:
        # Only acceptable if we could genuinely read the image.
        listing = subprocess.run(["lsinitrd", f"/boot/initrd.img-{platform.release()}"],
                                 capture_output=True, text=True)
        assert listing.stdout.strip(), "reported clean from an empty listing"


def test_verify_unknown_kernel_is_uninspectable_not_clean(prelude):
    rc = run(prelude, 'gb_initramfs_verify_clean "9.9.9-does-not-exist"; echo rc=$?')
    assert int(re.search(r"rc=(\d+)", rc.stdout).group(1)) == 2


# ── the set -e abort (2026-08-22 incident) ────────────────────────────────────

SHELL_OPTS = "set -euo pipefail"


def run_refresh_under_set_e(extra=""):
    """Run the real gb_initramfs_refresh under the installer's own shell options.

    The 2026-08-22 incident in one line: gb_initramfs_verify_clean returns
    non-zero as a NORMAL signal (1 = stale module, 2 = uninspectable), and it
    was called bare. A bare command is not a tested context, so `set -e` treated
    that signal as fatal and aborted the install right after a SUCCESSFUL
    regeneration , "Install aborted (exit 1) , restoring pre-install runtime
    state". The helpers all passed their individual tests; nothing exercised
    them composed, under the shell options the installer actually uses.
    """
    src = SETUP.read_text()
    fns = []
    for fn in FUNCS + ("gb_initramfs_regen_one", "gb_initramfs_refresh"):
        m = re.search(rf"^{fn}\(\) \{{.*?^\}}", src, re.S | re.M)
        assert m, f"{fn} missing"
        fns.append(m.group(0))
    script = f"""
{SHELL_OPTS}
gb_ok(){{ echo "OK $*"; }}; gb_warn(){{ echo "WARN $*"; }}; gb_info(){{ echo "INFO $*"; }}
GB_DRACUT_EXCLUDE=/dev/null
GB_INITRAMFS_HOOK=/dev/null
{chr(10).join(fns)}
gb_initramfs_regen_one(){{ {extra} return 0; }}   # do not rebuild real boot images
gb_initramfs_refresh
echo "REFRESH_RC=$?"
"""
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120)


def test_refresh_survives_set_e():
    """The regression guard for the aborted install."""
    r = run_refresh_under_set_e()
    assert "REFRESH_RC=0" in r.stdout, (
        f"gb_initramfs_refresh aborted under {SHELL_OPTS}\n"
        f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}")
    assert r.returncode == 0


def test_refresh_always_reports_a_verification_verdict():
    """The install output that failed showed two 'regenerated' lines and then
    nothing , the abort happened exactly where the verdict should have been.
    One of the three verdicts must always be printed."""
    out = run_refresh_under_set_e().stdout
    verdicts = ("verified:", "VERIFY FAILED", "not inspectable")
    assert any(v in out for v in verdicts), f"no verification verdict in:\n{out}"


def test_refresh_survives_a_failing_regeneration():
    """A kernel that fails to regenerate must be reported, not fatal."""
    r = run_refresh_under_set_e(extra="[[ $1 == *generic* ]] && return 1;")
    assert "REFRESH_RC=0" in r.stdout, f"aborted on a failed kernel:\n{r.stdout}{r.stderr}"


def test_helper_functions_do_not_leak_a_failing_status():
    """gb_initramfs_kernels ends in a loop whose last test can be false; under
    `set -e` that status must not escape."""
    src = SETUP.read_text()
    m = re.search(r"^gb_initramfs_kernels\(\) \{.*?^\}", src, re.S | re.M)
    assert "return 0" in m.group(0), "gb_initramfs_kernels must end with an explicit return 0"


def test_refresh_ends_with_explicit_return_zero():
    src = SETUP.read_text()
    m = re.search(r"^gb_initramfs_refresh\(\) \{.*?^\}", src, re.S | re.M)
    body = m.group(0)
    tail = body.rstrip()[:-1].rstrip()          # drop closing brace
    assert tail.endswith("return 0"), "a reporting step must not decide install success"


# ── the installer still parses ────────────────────────────────────────────────

def test_setup_script_is_valid_bash():
    assert subprocess.run(["bash", "-n", str(SETUP)],
                          capture_output=True).returncode == 0


def test_refresh_references_only_defined_variables():
    """Guards a real slip: the first draft referenced $GB_INITRAMFS_EXCLUDE,
    which does not exist (the real name is $GB_INITRAMFS_HOOK), so the operator
    would have been pointed at an empty path in the failure message."""
    src = SETUP.read_text()
    body = re.search(r"^gb_initramfs_refresh\(\) \{.*?^\}", src, re.S | re.M)
    assert body, "gb_initramfs_refresh missing"
    for var in set(re.findall(r"\$(GB_[A-Z_]+)", body.group(0))):
        assert re.search(rf"^{var}=", src, re.M), f"${var} used but never defined"

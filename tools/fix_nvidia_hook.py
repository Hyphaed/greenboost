#!/usr/bin/env python3
"""fix_nvidia_hook.py — Fix kmt-apt-hook syntax error and purge stale nvidia-580 packages."""

import os
import subprocess
import sys

from rich.console import Console
from rich.rule import Rule

VIOLET = "#6C71C4"
LIME   = "#E6FF3C"
GRAY   = "#D0CFCC"
CYAN   = "#30C8FF"
AMBER  = "#FFBF00"
RED    = "#FF5C32"

console = Console(highlight=False)

APT_HOOK = "/usr/local/bin/kmt-apt-hook"
OLD_LINE = "printf '%s\\n' \"${current[@]##*/}\" | sort -u > \"$STATE_FILE\""
NEW_LINE = "for _e in \"${current[@]}\"; do printf '%s\\n' \"${_e##*/}\"; done | sort -u > \"$STATE_FILE\""

STALE_580_PKGS = [
    "libnvidia-compute-580",
    "linux-modules-nvidia-580-open-6.19.0-6-generic",
    "linux-modules-nvidia-580-open-6.19.0-9-generic",
    "linux-modules-nvidia-580-open-7.0.0-10-generic",
    "nvidia-compute-utils-580",
    "nvidia-dkms-580-open",
    "nvidia-kernel-common-580",
]


def ok(msg):   console.print(f"  [bold {LIME}]✓[/]  [{GRAY}]{msg}[/]")
def fail(msg): console.print(f"  [bold {RED}]✗[/]  {msg}")
def warn(msg): console.print(f"  [bold {AMBER}]⚠[/]  {msg}")
def info(msg): console.print(f"  [bold {VIOLET}]◈[/]  [{GRAY}]{msg}[/]")
def step(n, total, msg):
    console.print(f"\n  [bold {CYAN}][{n}/{total}][/] [bold {GRAY}]{msg}[/]")


def require_root():
    if os.geteuid() != 0:
        fail(f"Must run as root. Re-run with: [bold]sudo python3 {sys.argv[0]}[/]")
        sys.exit(1)


def fix_apt_hook():
    step(1, 3, "Fix kmt-apt-hook syntax error")

    try:
        content = open(APT_HOOK).read()
    except FileNotFoundError:
        warn(f"{APT_HOOK} not found — skipping")
        return

    if OLD_LINE not in content:
        if NEW_LINE in content:
            ok("Already patched — no change needed")
        else:
            warn("Expected line not found — manual inspection required")
            info(f"Look at line 44 of {APT_HOOK}")
        return

    patched = content.replace(OLD_LINE, NEW_LINE, 1)
    open(APT_HOOK, "w").write(patched)
    ok(f"Patched {APT_HOOK} line 44")

    result = subprocess.run(["bash", "-n", APT_HOOK], capture_output=True, text=True)
    if result.returncode == 0:
        ok("Syntax check passed")
    else:
        fail(f"Syntax check still failing:\n{result.stderr.strip()}")
        sys.exit(1)


def purge_stale_580():
    step(2, 3, "Purge stale nvidia-580 config remnants (rc packages)")

    # Filter to only those actually present in dpkg db
    result = subprocess.run(["dpkg", "-l"] + STALE_580_PKGS,
                            capture_output=True, text=True)
    present = [
        line.split()[1].split(":")[0]
        for line in result.stdout.splitlines()
        if line.startswith("rc")
    ]

    if not present:
        ok("No stale nvidia-580 packages found — nothing to purge")
        return

    info(f"Purging {len(present)} package(s): {', '.join(present)}")
    proc = subprocess.run(
        ["dpkg", "--purge"] + present,
        capture_output=True, text=True
    )
    if proc.returncode == 0:
        ok("Purge complete")
    else:
        # dpkg --purge often exits non-zero for dependency warnings but still works
        # check if any rc entries remain
        check = subprocess.run(["dpkg", "-l"] + present, capture_output=True, text=True)
        still_rc = [l for l in check.stdout.splitlines() if l.startswith("rc")]
        if still_rc:
            warn("Some packages could not be purged (dependency warnings)")
            for line in proc.stderr.strip().splitlines()[:5]:
                console.print(f"    [{GRAY}]{line}[/]")
        else:
            ok("Purge complete (with suppressed dependency warnings)")


def check_nvidia_mismatch():
    step(3, 3, "NVIDIA driver/library version check")

    # Loaded kernel module version
    try:
        kmod_ver = open("/proc/driver/nvidia/version").readline().split()[7]
    except Exception:
        warn("Could not read /proc/driver/nvidia/version — module may not be loaded")
        return

    # Installed library version
    lib_result = subprocess.run(
        ["dpkg", "-l", "nvidia-driver-595-open"],
        capture_output=True, text=True
    )
    lib_ok = any(l.startswith("ii") for l in lib_result.stdout.splitlines())

    if kmod_ver == "595.58.03" and lib_ok:
        ok(f"Kernel module and userspace both at 595.58.03 — no mismatch")
        return

    warn(f"Kernel module loaded: [bold]{kmod_ver}[/]  │  Userspace target: [bold]595.58.03[/]")
    info("A reboot is required to load the 595 kernel module.")
    info("After reboot, run: [bold]nvidia-smi[/]")


def main():
    console.print()
    console.print(Rule(f"[bold {VIOLET}]GreenBoost — nvidia fix[/]", style=VIOLET))
    console.print()

    require_root()
    fix_apt_hook()
    purge_stale_580()
    check_nvidia_mismatch()

    console.print()
    console.print(Rule(style=f"dim {GRAY}"))
    console.print(f"\n  [{GRAY}]Done. Reboot to activate the 595 kernel module.[/]\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
greenboost_builder.py - cross-distro GreenBoost build orchestrator.

Detects the running kernel version and distro family, resolves the correct
kernel header path, and drives make with the right environment variables.
Inspired by vmware_module_builder patterns for idempotent cross-kernel builds.

Usage:
    python3 greenboost_builder.py build    # detect distro + build
    python3 greenboost_builder.py install  # build + DKMS install
    python3 greenboost_builder.py check    # preflight only (exit 1 if issues)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Audit F-L5-15: cap subprocess wall-clock time at 10 min for the slow paths
# (make on a large kernel tree) and shorter for fast queries.  Prevents a hung
# child from blocking automated installers forever.
_BUILD_TIMEOUT_S = 600
_QUERY_TIMEOUT_S = 30


@dataclass
class KernelVersion:
    raw:   str
    major: int
    minor: int
    patch: int

    @classmethod
    def current(cls) -> "KernelVersion":
        # Audit F-L5-15: bound external command time.
        raw = subprocess.check_output(
            ["uname", "-r"], text=True, timeout=_QUERY_TIMEOUT_S,
        ).strip()
        m = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", raw)
        if not m:
            raise RuntimeError(f"Cannot parse kernel version: {raw!r}")
        return cls(
            raw=raw,
            major=int(m[1]),
            minor=int(m[2]),
            patch=int(m[3] or 0),
        )

    def at_least(self, major: int, minor: int, patch: int = 0) -> bool:
        return (self.major, self.minor, self.patch) >= (major, minor, patch)


DISTRO_FAMILY: dict[str, str] = {
    "ubuntu":               "debian",
    "debian":               "debian",
    "kali":                 "debian",
    "linuxmint":            "debian",
    "pop":                  "debian",
    "raspbian":             "debian",
    "fedora":               "fedora",
    "rhel":                 "fedora",
    "centos":               "fedora",
    "rocky":                "fedora",
    "almalinux":            "fedora",
    "ol":                   "fedora",   # Oracle Linux
    "arch":                 "arch",
    "manjaro":              "arch",
    "endeavouros":          "arch",
    "garuda":               "arch",
    "opensuse-leap":        "suse",
    "opensuse-tumbleweed":  "suse",
    "sles":                 "suse",
    "alpine":               "alpine",
    "void":                 "void",
    "gentoo":               "gentoo",
    "nixos":                "nixos",
}

HEADER_PKGS: dict[str, Callable[["KernelVersion"], str]] = {
    "debian": lambda kver: f"linux-headers-{kver.raw}",
    "fedora": lambda kver: f"kernel-devel-{kver.raw}",
    "arch":   lambda kver: _arch_header_pkg(kver.raw),
    "suse":   lambda kver: f"kernel-default-devel={kver.raw}",
    "alpine": lambda kver: "linux-headers",
    "void":   lambda kver: f"kernel-headers-{kver.raw}",
}

HEADER_DIRS = [
    "/lib/modules/{kver}/build",
    "/usr/src/kernels/{kver}",
    "/usr/src/linux-{kver}",
    "/usr/lib/modules/{kver}/build",
]

CUDA_CANDIDATES = [
    "/usr/local/cuda",
    "/usr/cuda",
    "/opt/cuda",
]


def _arch_header_pkg(kver_raw: str) -> str:
    # Strip the Arch-specific suffix (e.g. "6.8.9-arch1-1" → "linux")
    # Standard kernels ship as "linux-headers"; LTS as "linux-lts-headers", etc.
    if "lts" in kver_raw.lower():
        return "linux-lts-headers"
    return "linux-headers"


def detect_family() -> str:
    """Return distro family string (debian/fedora/arch/suse/alpine/void/unknown)."""
    osrel = Path("/etc/os-release")
    if not osrel.exists():
        return "unknown"
    data: dict[str, str] = {}
    for line in osrel.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            data[k.strip()] = v.strip().strip('"')
    _id = data.get("ID", "").lower()
    family = DISTRO_FAMILY.get(_id)
    if family:
        return family
    # Try ID_LIKE for derivatives (e.g. "ID_LIKE=debian")
    for like in data.get("ID_LIKE", "").lower().split():
        if like in DISTRO_FAMILY:
            return DISTRO_FAMILY[like]
    # Audit F-L5-17: be loud about unrecognised distros - they typically need
    # manual ID_LIKE setup, and the next stage (header lookup) won't tell the
    # operator why nothing worked.
    print(
        f"[greenboost_builder] WARN: unrecognised distro ID={_id!r}; "
        "header package detection will fall back to the debian convention. "
        "Set ID_LIKE in /etc/os-release if this is a derivative.",
        file=sys.stderr,
    )
    return "unknown"


def find_kernel_headers(kver: KernelVersion) -> Path | None:
    for tpl in HEADER_DIRS:
        p = Path(tpl.format(kver=kver.raw))
        if p.is_dir():
            return p
    return None


def find_cuda_dir() -> Path | None:
    """Return the best available CUDA installation directory.

    Prefers the newest side-by-side versioned install (/usr/local/cuda-13
    over /usr/local/cuda-12) before falling back to the unversioned symlink
    or other standard paths.  This matters on systems where the user installs
    CUDA 13 via the NVIDIA repo but the /usr/local/cuda symlink still points
    to a CUDA 12 left over from an ubuntu-repo nvidia-cuda-toolkit package.
    """
    import glob

    # Collect all /usr/local/cuda-N.M style dirs that have cuda.h
    versioned = sorted(
        (p for p in (Path(g) for g in glob.glob("/usr/local/cuda-[0-9]*"))
         if p.is_dir() and (p / "include" / "cuda.h").exists()),
        key=lambda p: [int(x) for x in p.name.lstrip("cuda-").split(".") if x.isdigit()],
    )
    if versioned:
        return versioned[-1]  # highest version

    # Fall back to the original candidate list (unversioned symlink / other paths)
    for cand in CUDA_CANDIDATES:
        p = Path(cand)
        if p.is_dir() and (p / "include" / "cuda.h").exists():
            return p
    return None


def has_symbol_in_headers(symbol: str, kver: KernelVersion) -> bool:
    hdr = find_kernel_headers(kver)
    if not hdr:
        return False
    inc = hdr / "include"
    # Audit F-L5-14: grep returncode 1 (not found) is intentionally treated as
    # "absent"; only returncode 0 means "present".  We also bound the search
    # so a runaway header tree can't hang the builder.
    r = subprocess.run(
        ["grep", "-r", "--include=*.h", "-lm1", symbol, str(inc)],
        capture_output=True,
        timeout=_QUERY_TIMEOUT_S,
    )
    return r.returncode == 0


class GreenBoostBuilder:
    def __init__(self) -> None:
        self.kver   = KernelVersion.current()
        self.family = detect_family()
        self.kdir   = find_kernel_headers(self.kver)
        self.cuda   = find_cuda_dir()
        self.src    = Path(__file__).parent

    # ── BPF prereq packages per distro ──────────────────────────────────────
    _BPF_PKGS: dict[str, str] = {
        "debian":  "clang bpftool libbpf-dev",
        "fedora":  "clang bpftool libbpf-devel",
        "arch":    "clang bpf bpftool libbpf",
        "suse":    "clang bpftool libbpf-devel",
        "alpine":  "clang bpftool libbpf-dev",
    }

    def _bpf_prereqs(self) -> list[str]:
        """Return missing BPF-tracer prereqs as install hints (not blocking)."""
        missing = []
        # Accept clang-N (e.g. clang-21) if 'clang' symlink is absent
        _clang = (shutil.which("clang") or
                  next((shutil.which(f"clang-{v}") for v in range(21, 12, -1)
                        if shutil.which(f"clang-{v}")), None))
        if not _clang:
            missing.append("clang")
        if not shutil.which("bpftool"):
            missing.append("bpftool")
        try:
            r = subprocess.run(
                ["pkg-config", "--exists", "libbpf"],
                capture_output=True,
                timeout=_QUERY_TIMEOUT_S,
            )
            if r.returncode != 0:
                missing.append("libbpf-dev")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            missing.append("libbpf-dev")
        return missing

    def preflight(self) -> list[str]:
        """Return list of blocking issues; empty list means ready to build."""
        issues: list[str] = []
        if self.kdir is None:
            family = self.family
            pkg = HEADER_PKGS.get(family, lambda k: f"linux-headers-{k.raw}")(self.kver)
            issues.append(
                f"Kernel headers not found for {self.kver.raw}. "
                f"Install: {pkg}"
            )
        if not shutil.which("gcc"):
            issues.append("gcc not found - install build-essential or gcc package")
        if not shutil.which("make"):
            issues.append("make not found - install build-essential or make package")

        # BPF prereqs: non-blocking, reported as warnings
        bpf_missing = self._bpf_prereqs()
        if bpf_missing:
            pkg_hint = self._BPF_PKGS.get(self.family, "clang bpftool libbpf-dev")
            print(
                f"[greenboost_builder] INFO: eBPF tracer not built "
                f"(missing: {', '.join(bpf_missing)}). "
                f"Enable: install {pkg_hint}",
                file=sys.stderr,
            )
        else:
            print(
                "[greenboost_builder] INFO: eBPF prereqs OK "
                "(clang, bpftool, libbpf) , tracer will be built",
                file=sys.stderr,
            )

        return issues

    def _build_env(self) -> dict[str, str]:
        env = {**os.environ, "KVER": self.kver.raw, "KDIR": str(self.kdir)}
        if self.cuda:
            env["CUDA_DIR"] = str(self.cuda)
        return env

    def build(self) -> None:
        issues = self.preflight()
        if issues:
            for issue in issues:
                print(f"[!] {issue}", file=sys.stderr)
            sys.exit(1)
        env = self._build_env()
        print(f"[greenboost_builder] kernel={self.kver.raw} distro={self.family} "
              f"kdir={self.kdir} cuda={self.cuda or 'not found'}")
        # Audit F-L5-15: bound each make invocation.  A hung kernel build
        # otherwise blocks installer pipelines indefinitely.
        # Audit F-L5-30: capture stderr so CI logs contain the failure reason;
        # print it only on error so normal runs stay quiet.
        def _run_make(*args, timeout):
            r = subprocess.run(
                ["make", *args], cwd=self.src, env=env,
                timeout=timeout, capture_output=True, text=True,
            )
            if r.returncode != 0:
                sys.stderr.write(r.stderr or r.stdout or "(no output)\n")
                raise subprocess.CalledProcessError(r.returncode, r.args, r.stdout, r.stderr)

        _run_make("clean", timeout=_QUERY_TIMEOUT_S)
        _run_make(timeout=_BUILD_TIMEOUT_S)

    def dkms_install(self) -> None:
        self.build()
        # Audit F-L5-18: dkms_install idempotency.  If a build for the current
        # kernel is already registered, skip the duplicate registration.
        try:
            status = subprocess.run(
                ["dkms", "status", "greenboost"],
                capture_output=True, text=True, timeout=_QUERY_TIMEOUT_S,
            )
            if status.returncode == 0 and self.kver.raw in status.stdout:
                print("[greenboost_builder] DKMS already installed for "
                      f"kernel {self.kver.raw} - skipping make dkms-install",
                      file=sys.stderr)
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        subprocess.check_call(["make", "dkms-install"], cwd=self.src,
                              env=self._build_env(),
                              timeout=_BUILD_TIMEOUT_S)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    b = GreenBoostBuilder()

    if cmd == "build":
        b.build()
    elif cmd == "install":
        b.dkms_install()
    elif cmd == "check":
        issues = b.preflight()
        for issue in issues:
            print(issue)
        sys.exit(0 if not issues else 1)
    else:
        print(f"Unknown command: {cmd!r}. Use: build | install | check", file=sys.stderr)
        sys.exit(1)

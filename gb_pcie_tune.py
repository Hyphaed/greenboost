#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_pcie_tune.py , inspect and tune the PCIe link GreenBoost streams T2 over.

Why this matters here more than on a normal box
-----------------------------------------------
GreenBoost's whole T2 tier is the GPU reading host memory across PCIe. When a
model does not fit VRAM, that link IS the decode bottleneck: measured on this
box 2026-08-18, a dense 27B moved 4.49 GiB per forward pass and spent 390 ms on
the bus against 17 ms of GPU compute. Every percent of real link bandwidth is a
percent of tokens per second.

And there is a visible gap to explain. This box's link trains at Gen4 x16, whose
usable one-direction ceiling is ~26 GB/s, while the shim's own measured
throughput is ~11.5 GB/s , about 44%. Some of that is unavoidable (read latency,
scatter-gather, IOMMU translation), but `MaxReadReq` is the classic cause of a
gap this size: it caps how many bytes the GPU may request per DMA read, and a
512-byte cap on a multi-gigabyte sequential stream costs a completion round trip
every 512 bytes.

What this does NOT claim
------------------------
Raising MaxReadReq is worth single-digit to ~20% on read-heavy DMA in the
literature, not a multiple. It is a real lever on a bandwidth-bound box and it
is cheap to test, but the arithmetic that pins a dense 27B near 2.5 tok/s is not
undone by it , see greenboost_plans/dense_model_roadmap_2026-08-18.md.

Gen5 is separately ruled out on this hardware, and this tool reports the
evidence rather than repeating the claim: the GPU advertises max_link_speed
32.0 GT/s while the link trains at 16.0 GT/s, because the board's root port
tops out at Gen4.

Reading the control register needs CAP_SYS_ADMIN, so `report` works unprivileged
only for what sysfs exposes; `--apply` requires root and is refused otherwise.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SYS_PCI = Path("/sys/bus/pci/devices")

# PCIe MaxReadReq encoding: the DevCtl field is log2(bytes) - 7.
MRRS_BYTES = {0: 128, 1: 256, 2: 512, 3: 1024, 4: 2048, 5: 4096}


def gpu_bdf() -> str:
    """PCI address of the first NVIDIA display/3D controller."""
    try:
        out = subprocess.run(["lspci", "-nn"], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in out.splitlines():
        if re.search(r"(VGA|3D).*NVIDIA", line, re.I):
            return "0000:" + line.split()[0]
    return ""


def _sysfs(bdf: str, name: str) -> str:
    try:
        return (SYS_PCI / bdf / name).read_text().strip()
    except OSError:
        return ""


def _lspci_caps(bdf: str) -> dict:
    """MaxPayload/MaxReadReq from lspci. Needs root for the capability block."""
    try:
        out = subprocess.run(["lspci", "-vv", "-s", bdf.replace("0000:", "")],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    caps = {}
    m = re.search(r"MaxPayload (\d+) bytes, MaxReadReq (\d+) bytes", out)
    if m:
        caps["max_payload_bytes"] = int(m.group(1))
        caps["max_read_req_bytes"] = int(m.group(2))
    m = re.search(r"DevCap:.*MaxPayload (\d+) bytes", out)
    if m:
        caps["max_payload_supported_bytes"] = int(m.group(1))
    return caps


def report(bdf: str = "") -> dict:
    bdf = bdf or gpu_bdf()
    if not bdf:
        return {"error": "no NVIDIA GPU found on the PCI bus"}
    cur = _sysfs(bdf, "current_link_speed")
    mx = _sysfs(bdf, "max_link_speed")
    caps = _lspci_caps(bdf)
    out = {
        "bdf": bdf,
        "current_link_speed": cur, "max_link_speed": mx,
        "current_link_width": _sysfs(bdf, "current_link_width"),
        "max_link_width": _sysfs(bdf, "max_link_width"),
        "numa_node": _sysfs(bdf, "numa_node"),
        **caps,
    }
    notes = []
    if cur and mx and cur != mx:
        notes.append(
            f"link trains at {cur} while the device advertises {mx} , the board's "
            f"root port is the limit, not the GPU. Not fixable in software.")
    rr = caps.get("max_read_req_bytes")
    if rr is None:
        notes.append("MaxReadReq unreadable , run as root to see it "
                     "(the PCIe capability block needs CAP_SYS_ADMIN).")
    elif rr < 4096:
        notes.append(
            f"MaxReadReq is {rr} bytes. On a multi-gigabyte sequential weight "
            f"stream that is a completion round trip every {rr} bytes; 4096 is "
            f"the ceiling and is worth measuring here.")
    else:
        notes.append(f"MaxReadReq already at {rr} bytes , nothing to raise.")
    out["notes"] = notes
    return out


def apply_mrrs(bdf: str = "", target_bytes: int = 4096) -> dict:
    """Raise MaxReadReq via setpci. Root only; verifies the write took effect.

    Deliberately does not persist across reboots. A PCIe control-register poke
    that survives a reboot with nothing recording why is exactly the kind of
    change that outlives the person who made it , make it, MEASURE it, and only
    then decide whether it earns a place in the boot path.
    """
    import os
    if os.geteuid() != 0:
        return {"ok": False, "error": "needs root (setpci writes PCIe config space)"}
    bdf = bdf or gpu_bdf()
    if not bdf:
        return {"ok": False, "error": "no NVIDIA GPU found"}
    code = {v: k for k, v in MRRS_BYTES.items()}.get(target_bytes)
    if code is None:
        return {"ok": False, "error": f"target must be one of {sorted(MRRS_BYTES.values())}"}
    short = bdf.replace("0000:", "")
    before = _lspci_caps(bdf).get("max_read_req_bytes")
    try:
        cur = subprocess.run(["setpci", "-s", short, "CAP_EXP+8.w"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        val = int(cur, 16)
        # DevCtl bits 14:12 hold MaxReadReq.
        new = (val & ~(0x7 << 12)) | (code << 12)
        subprocess.run(["setpci", "-s", short, f"CAP_EXP+8.w={new:04x}"],
                       check=True, capture_output=True, timeout=10)
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        return {"ok": False, "error": f"setpci failed: {e}"}
    after = _lspci_caps(bdf).get("max_read_req_bytes")
    ok = after == target_bytes
    return {"ok": ok, "bdf": bdf, "before_bytes": before, "after_bytes": after,
            "note": ("not persistent across reboot , measure with "
                     "gb_bench_turn.py before considering a boot-time change")
            if ok else "write did not take effect"}


def main() -> None:
    p = argparse.ArgumentParser(prog="gb_pcie_tune.py", description=__doc__)
    p.add_argument("command", choices=["report", "apply"])
    p.add_argument("--bdf", default="")
    p.add_argument("--max-read-req", type=int, default=4096)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    res = report(a.bdf) if a.command == "report" else apply_mrrs(a.bdf, a.max_read_req)
    if a.json:
        print(json.dumps(res, indent=1)); return
    if res.get("error") and a.command == "report":
        print(f"error: {res['error']}"); raise SystemExit(1)
    for k, v in res.items():
        if k == "notes":
            continue
        print(f"  {k:28} {v}")
    for n in res.get("notes", []):
        print(f"\n  * {n}")


if __name__ == "__main__":
    main()

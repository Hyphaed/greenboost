"""gb_cpu_pin: one llama.cpp thread per physical P-core, pinned (`-t N -C <mask> --cpu-strict 1`). OFF by default (GB_SYNAPSE_CPU_PIN=0), pending a measured A/B (workflow/round2-improvements.md,
'A/B recipes (not run)'): `llama-server` otherwise runs --threads <n> with no affinity, so the scheduler may park a decode thread on an E-core or on the SMT sibling of a busy P-core, and a
decode step waits for its slowest thread.

GB_SYNAPSE_CPU_PIN = 0 (default, off) | 1 (on) | auto (on only on a hybrid CPU, i.e. when /sys/devices/cpu_atom lists cores).

Topology is derived from sysfs, never hard-coded:
  P-core logical CPUs = /sys/devices/cpu_core/cpus (hybrid Intel); else the CPUs with the highest cpu_capacity when capacities differ (ARM big.LITTLE, AMD preferred cores); else every CPU.
  SMT: of each core's thread siblings (topology/thread_siblings_list) only the LOWEST-numbered sibling is used, so N threads run on N physical cores and none shares a core with another of ours.
`cpu_pin_args(topology)` is pure; `read_topology()` is the only reader.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional


def parse_cpulist(text: str) -> List[int]:
    """'0-3,8,10-11' -> [0,1,2,3,8,10,11]."""
    out: List[int] = []
    for part in (text or "").strip().split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return sorted(set(out))


def read_topology(sys_root: str = "/sys") -> dict:
    """{'online', 'p_cpus', 'e_cpus', 'capacity': {cpu: int}, 'siblings': {cpu: [cpus]}} from sysfs; missing files give empty values."""
    def rd(p):
        try:
            with open(os.path.join(sys_root, p), encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return ""
    online = parse_cpulist(rd("devices/system/cpu/online"))
    topo = {"online": online, "p_cpus": parse_cpulist(rd("devices/cpu_core/cpus")), "e_cpus": parse_cpulist(rd("devices/cpu_atom/cpus")), "capacity": {}, "siblings": {}}
    for c in online:
        cap = rd(f"devices/system/cpu/cpu{c}/cpu_capacity")
        if cap.isdigit():
            topo["capacity"][c] = int(cap)
        sib = rd(f"devices/system/cpu/cpu{c}/topology/thread_siblings_list")
        if sib:
            topo["siblings"][c] = parse_cpulist(sib)
    return topo


def pcore_cpus(topology: dict) -> List[int]:
    """Logical CPUs of the performance cores (see module docstring for the order of sources)."""
    online = list(topology.get("online") or [])
    if topology.get("p_cpus"):
        return [c for c in topology["p_cpus"] if not online or c in online]
    cap = {c: v for c, v in (topology.get("capacity") or {}).items() if not online or c in online}
    if cap and len(set(cap.values())) > 1:
        top = max(cap.values())
        return sorted(c for c, v in cap.items() if v == top)
    return online


def one_per_core(cpus: List[int], siblings: Dict[int, List[int]]) -> List[int]:
    """Keep only the lowest-numbered logical CPU of every physical core among `cpus`."""
    chosen: List[int] = []
    taken: set = set()
    for c in sorted(cpus):
        if c in taken:
            continue
        chosen.append(c)
        taken.update(siblings.get(c) or [c])
    return chosen


def cpu_mask(cpus: List[int]) -> str:
    """Hex affinity mask ('0x5555') for llama.cpp's -C."""
    m = 0
    for c in cpus:
        m |= 1 << c
    return hex(m)


def mode(env: Optional[str] = None) -> str:
    v = (os.environ.get("GB_SYNAPSE_CPU_PIN", "0") if env is None else env).strip().lower()
    return v if v in ("0", "1", "auto") else "0"


def cpu_pin_args(topology: dict, setting: Optional[str] = None) -> List[str]:
    """['-t', N, '-C', mask, '--cpu-strict', '1'] for the P-cores of `topology`, or [] when pinning is off / not applicable (setting 0; auto on a non-hybrid CPU; fewer than 2 cores found)."""
    s = mode(setting)
    if s == "0":
        return []
    if s == "auto" and not topology.get("e_cpus"):
        return []
    cores = one_per_core(pcore_cpus(topology), topology.get("siblings") or {})
    if len(cores) < 2:
        return []
    return ["-t", str(len(cores)), "-C", cpu_mask(cores), "--cpu-strict", "1"]


def apply_to_cmd(cmd: list, topology: Optional[dict] = None, setting: Optional[str] = None) -> bool:
    """Rewrite a llama-server command in place: set the thread count and add the mask flags. Returns True when pinned. A command that already has -C/--cpu-mask/--cpu-strict is left alone."""
    if any(a in cmd for a in ("-C", "--cpu-mask", "--cpu-strict", "-Cr", "--cpu-range")):
        return False
    args = cpu_pin_args(read_topology() if topology is None else topology, setting)
    if not args:
        return False
    for flag in ("--threads", "-t"):
        if flag in cmd:
            cmd[cmd.index(flag) + 1] = args[1]
            break
    else:
        cmd += args[:2]
    cmd += args[2:]
    return True

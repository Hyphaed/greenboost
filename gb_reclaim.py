#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_reclaim.py — classify and selectively reclaim GreenBoost-held GPU/T2/T3
memory. The shared node behind `greenboost clear memory-pool` and the
(now-consolidated) `gb clear-memory-pool` command.

Why this exists: greenboost_setup.sh's cmd_clear_memory_pool is a nuke — it
kills EVERY process holding /dev/greenboost open or using at least
GB_KILL_MIN_MB of GPU compute (an env-overridable threshold, never a frozen
literal), filtered only by a protected-comm allowlist (desktop shell, display
servers, VM GPU-passthrough processes). It has no notion of "the one stuck
job I actually want to kill" vs. "another genuinely-in-progress GreenBoost
job on this same node" — CLAUDE.md's own operating rule already calls this
out: "Never run it while other genuinely-in-progress GreenBoost work is on
the SAME node unless the owner explicitly authorizes it." This module adds
the missing classification so a caller can reclaim just the orphaned
residue by default, with --all/scope="all" as the explicit, documented
opt-in that reproduces the old nuke's full blast radius.

Classification (see classify_processes()):
  live      — gb-synapse itself is currently tracking this PID as a real,
              in-progress server (gb_synapse.ps()). Never a reclaim target
              at any scope short of "all".
  ambiguous — holds a reclaim-candidate resource (a /dev/greenboost fd or
              >=kill_min_mb of GPU compute) and has a dataflux event within
              the recency window, but isn't in gb-synapse's own live-server
              set (e.g. a torch/diffusion job, or a synapse process whose
              run-state file was lost) — probably legitimate, treat with
              more caution than plain residue.
  residue   — a reclaim-candidate resource with NO recent activity and NO
              live-tracking match: the orphaned-process case this module
              exists to target safely.

Escalation order (same as the bash nuke, formalized): an unload API call
first when one exists (Ollama's /api/generate keep_alive=0 — a graceful
release beats a signal), then SIGTERM, a short wait, then SIGKILL for any
stragglers. Every run_reclaim() call emits a `reclaim` dataflux event so a
kill is always visible to dataflux_summary/dataflux_errors afterward, not
just to whoever's terminal happened to be watching.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_DIR))

GB_DEV = "/dev/greenboost"
DEFAULT_KILL_MIN_MB = int(os.environ.get("GB_KILL_MIN_MB", "512"))
DEFAULT_TERM_WAIT_S = 2.0
DEFAULT_RECENCY_MINUTES = 10.0

# Mirrors greenboost_setup.sh's _gb_proc_is_protected exactly (bash case
# pattern, ported by hand — no shared source between a shell script and a
# Python module short of shelling out per check, which every hot path here
# would pay for). Keep the two lists in sync if either changes.
_PROTECTED_EXACT = {
    "gnome-shell", "gnome-session", "gnome-control-", "gnome-software",
    "gnome-keyring", "mutter", "gjs", "gdm", "Xorg", "Xorg.bin", "Xwayland",
    "X", "kwin", "plasmashell", "ksmserver", "sddm", "lightdm",
    "systemd", "init", "dbus-daemon", "dbus-broker", "elogind", "seatd",
    "pipewire", "wireplumber", "pulseaudio",
    "nvidia-persiste", "greenboost-netd",
    "mksSandbox", "vmware-vmx", "vmware-vmx-debug", "vmware-vmx-stats",
    "vmnet-natd", "vmtoolsd", "vmware-hostd",
    "VBoxSVC", "VBoxHeadless", "VBoxSDL", "VirtualBoxVM", "VBoxXPCOMIPCD",
}
_PROTECTED_PREFIXES = (
    "gnome-shell-", "gnome-session", "gnome-control-", "gnome-software",
    "gnome-keyring", "tracker-", "gdm-", "Xwayland", "kwin_", "kded",
    "sddm-", "systemd-", "pipewire-", "nvidia-",
    "vmware-usbarbitrator", "qemu-system-",
)

# GreenBoost's OWN long-lived daemons — protected by exact name, never by a
# blanket "greenboost" prefix.
#
# The prefix used to be in _PROTECTED_PREFIXES, which made EVERY process whose
# comm starts with "greenboost" unreclaimable — including `greenboost-cli`
# itself. So `greenboost clear memory-pool` could never release the GPU memory
# of the very CLI the owner runs inference through, which is the opposite of
# what that command is for (owner report 2026-08-18). Only the fabric/system
# daemons genuinely need to survive a reclaim: killing greenboost-netd drops
# the cluster link to every feeder, and the supervisor owns the pool itself.
_PROTECTED_GB_DAEMONS = {
    "greenboost-netd", "greenboost-net", "gb_supervisor", "gb-supervisor",
    "greenboost-supe", "greenboostd",
}


def _gb_proc_is_protected(comm: str) -> bool:
    if comm in _PROTECTED_EXACT or comm in _PROTECTED_GB_DAEMONS:
        return True
    return any(comm.startswith(p) for p in _PROTECTED_PREFIXES) or comm == "qemu-kvm"


# ---------------------------------------------------------------------------
# Ollama unload (moved here from greenboost-cli/greenboost_cli/rag/
# memory_pool.py per task #7 — reclaim's graceful-unload path and the RAG
# ingest pipeline's post-batch VRAM release are the same operation and
# should share one implementation)
# ---------------------------------------------------------------------------

def _ollama_base_url() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _ollama_unload(model: str, *, base_url: "str | None" = None,
                   silent: bool = False) -> bool:
    """Unload a model from Ollama's VRAM immediately (keep_alive=0 to
    /api/generate — the standard Ollama eviction mechanism). True on success,
    False on any network/server error."""
    if not model:
        return False
    url = (base_url or _ollama_base_url()) + "/api/generate"
    payload = json.dumps({"model": model, "keep_alive": 0}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            resp.read()
        return True
    except urllib.error.URLError as exc:
        if not silent:
            sys.stderr.write(f"gb_reclaim: could not reach Ollama ({exc.reason})\n")
        return False
    except Exception as exc:
        if not silent:
            sys.stderr.write(f"gb_reclaim: {exc}\n")
        return False


def _ollama_list_loaded(*, base_url: "str | None" = None) -> list[dict]:
    url = (base_url or _ollama_base_url()) + "/api/ps"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        return data.get("models", [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Process discovery
# ---------------------------------------------------------------------------

def _fuser_pids(dev: str) -> "set[int]":
    """PIDs holding `dev` open, via fuser. Empty set on any failure (missing
    fuser binary, device not present, permission denied) — never raises."""
    try:
        out = subprocess.run(["fuser", dev], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return set()
    return {int(tok) for tok in out.split() if tok.isdigit()}


def _gpu_compute_pids() -> "dict[int, int]":
    """{pid: used_memory_mb} for every CUDA compute process nvidia-smi
    reports. {} on any failure (no nvidia-smi, no GPU, driver hiccup)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    result: "dict[int, int]" = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        result[int(parts[0])] = int(parts[1]) if parts[1].isdigit() else 0
    return result


def _proc_comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return ""


def _live_pids() -> "set[int]":
    """PIDs gb-synapse itself is currently tracking as a real, in-progress
    server — the ONE always-protected-short-of-scope='all' class, since these
    are exactly the "genuinely-in-progress GreenBoost work" CLAUDE.md's own
    operating rule says never to kill without explicit authorization."""
    try:
        import gb_synapse
    except Exception:
        return set()
    pids: "set[int]" = set()
    for st in gb_synapse.ps():
        for k in ("llama_pid", "proxy_pid", "embed_pid"):
            pid = st.get(k) or 0
            if pid:
                pids.add(int(pid))
    return pids


def _recent_dataflux_pids(minutes: float = DEFAULT_RECENCY_MINUTES) -> "set[int]":
    """PIDs that emitted ANY dataflux event within the last `minutes` — the
    signal that separates "ambiguous" (something happened here recently,
    probably legitimate) from "residue" (silent, orphaned)."""
    try:
        import gb_dataflux
        events = gb_dataflux.read_events(since_hours=minutes / 60.0)
    except Exception:
        return set()
    return {int(e["pid"]) for e in events if e.get("pid")}


def _ancestor_pids() -> "set[int]":
    """Every PID from this process up to init.

    Reclaim must never kill the thing that invoked it. `{getpid(), getppid()}`
    was enough while greenboost-cli was protected by name, but it is not any
    more: `/clear-memory` inside gb shells out to `sudo greenboost clear
    memory-pool`, so the live gb process is the caller's GRANDparent (gb ->
    sudo -> bash -> python), sits several levels up, and is now a legitimate
    reclaim candidate. Without the full walk, running the command from inside
    gb would kill gb — while it was running the command.
    """
    out: set[int] = set()
    pid = os.getpid()
    for _ in range(64):          # bounded: never spin on a malformed /proc
        if pid <= 1 or pid in out:
            break
        out.add(pid)
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                data = f.read().decode("utf-8", "replace")
            # comm can contain spaces and parentheses — parse after the LAST ')'
            ppid = int(data[data.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            break
        pid = ppid
    return out


def classify_processes(kill_min_mb: int = DEFAULT_KILL_MIN_MB) -> dict:
    """{"live": [...], "ambiguous": [...], "residue": [...]} of
    {"pid", "comm", "gpu_mb", "gb_dev"} dicts — every reclaim-candidate
    process (holds /dev/greenboost, or >=kill_min_mb of GPU compute),
    excluding this process's own tree, PID 1, and anything
    _gb_proc_is_protected() recognizes."""
    gb_dev_pids = _fuser_pids(GB_DEV)
    gpu_mb = _gpu_compute_pids()
    candidates = set(gb_dev_pids) | {p for p, mb in gpu_mb.items() if mb >= kill_min_mb}

    live_tracked = _live_pids()
    recent = _recent_dataflux_pids()
    self_pids = _ancestor_pids()

    out: dict = {"live": [], "ambiguous": [], "residue": []}
    for pid in candidates:
        if pid in self_pids or pid == 1:
            continue
        comm = _proc_comm(pid)
        if not comm or _gb_proc_is_protected(comm):
            continue
        entry = {"pid": pid, "comm": comm, "gpu_mb": gpu_mb.get(pid, 0),
                 "gb_dev": pid in gb_dev_pids}
        if pid in live_tracked:
            out["live"].append(entry)
        elif pid in recent:
            out["ambiguous"].append(entry)
        else:
            out["residue"].append(entry)
    return out


# ---------------------------------------------------------------------------
# Plan / run
# ---------------------------------------------------------------------------

_SCOPES = ("residue", "ambiguous", "inference", "all")

# comm names of local AI-inference workers. Matching is substring-based on the
# full command line as well as comm, because the ai-forge pipelines all run as
# a bare `python3` — comm alone cannot tell a ComfyUI worker apart from an
# unrelated script, and the owner's requirement is explicitly that a session
# started through ~/Dev/ai-forge (art_wizard.sh, a character generation, a
# LongLive render) is cleaned by `greenboost clear memory-pool` like any other.
_INFERENCE_COMMS = {
    "llama-server", "rpc-server", "ollama", "ollama_llama_se",
    "greenboost-cli", "vllm", "sglang",
}
_INFERENCE_CMDLINE_HINTS = (
    "ai-forge", "comfyui", "gb_diffusion_server", "gb_longlive_server",
    "gb_synapse", "greenboost_cli", "diffusers", "longlive",
    "art_wizard", "run_character_pipeline", "run_spherical_pipeline",
)


def _proc_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace").lower()
    except OSError:
        return ""


def is_inference_process(pid: int, comm: str) -> bool:
    """True when this PID is a local AI-inference worker.

    Deliberately looks at the full command line, not just comm: every ai-forge
    pipeline runs as `python3`, so comm is useless for telling a character
    render apart from an unrelated script. A false positive here kills a
    user's own python job, so the hints are specific paths/modules rather than
    anything as broad as "python".
    """
    if comm in _INFERENCE_COMMS:
        return True
    cl = _proc_cmdline(pid)
    return any(h in cl for h in _INFERENCE_CMDLINE_HINTS)


def plan_reclaim(scope: str = "residue", kill_min_mb: int = DEFAULT_KILL_MIN_MB) -> dict:
    """Dry run: what WOULD be reclaimed at `scope`, touching nothing.

    scope="residue" (default): orphaned processes only — the safe default
    that never touches gb-synapse's own tracked servers or anything with
    recent activity.
    scope="ambiguous": residue + ambiguous (recently active but untracked).
    scope="inference": every LOCAL AI-INFERENCE session, tracked or not —
    gb-synapse servers, ollama, greenboost-cli, and the ai-forge workers
    (ComfyUI, diffusers, LongLive, art_wizard). Desktop, GNOME, VM and
    GreenBoost's own fabric daemons stay protected. This is what "clear
    memory-pool" means to its owner: give me the whole card and the T2 DDR
    back from AI inference, without logging me out of my desktop.
    scope="all": every non-protected candidate including gb-synapse's own
    live-tracked servers — reproduces the old bash nuke's full blast radius.
    Never the default; only ever explicit (--all)."""
    if scope not in _SCOPES:
        raise ValueError(f"scope must be one of {_SCOPES}, got {scope!r}")
    classes = classify_processes(kill_min_mb)
    if scope == "residue":
        targets = list(classes["residue"])
    elif scope == "ambiguous":
        targets = list(classes["residue"]) + list(classes["ambiguous"])
    elif scope == "inference":
        # Every class, filtered to actual inference workers. A live-tracked
        # gb-synapse server IS in scope here — sparing it is the whole reason
        # the command kept reporting success while 10 GB stayed held.
        targets = [e for e in (list(classes["residue"]) + list(classes["ambiguous"])
                               + list(classes["live"]))
                   if is_inference_process(e["pid"], e["comm"])]
    else:
        targets = list(classes["residue"]) + list(classes["ambiguous"]) + list(classes["live"])
    return {"scope": scope, "targets": targets, "classes": classes}


def run_reclaim(scope: str = "residue", kill_min_mb: int = DEFAULT_KILL_MIN_MB,
                term_wait_s: float = DEFAULT_TERM_WAIT_S) -> dict:
    """Actually reclaim `scope`: graceful unload API first (Ollama), then
    SIGTERM every target, wait term_wait_s, SIGKILL any stragglers. Emits a
    `reclaim` dataflux event with the plan and outcome — best-effort, never
    raises on the emit itself."""
    plan = plan_reclaim(scope, kill_min_mb)
    targets = plan["targets"]

    unloaded: list[str] = []
    if any(t["comm"] == "ollama" for t in targets):
        for m in _ollama_list_loaded():
            name = m.get("name", "")
            if name and _ollama_unload(name, silent=True):
                unloaded.append(name)

    # Graceful gb-synapse shutdown BEFORE the SIGTERM sweep.
    #
    # stop() is not just a nicer kill: it captures the real KV footprint at the
    # one moment it is knowable (see gb_synapse.stop's own comment on
    # _persist_kv_measurement), tears down feeder state, and removes the
    # run-state file. SIGKILLing the process instead loses the measurement and
    # strands a run-state JSON describing a server that no longer exists, so
    # the next `ps` reports a phantom.
    stopped_servers: list[str] = []
    target_pids = {t["pid"] for t in targets}
    try:
        import gb_synapse
        for st in gb_synapse.ps():
            if st.get("llama_pid") in target_pids or st.get("proxy_pid") in target_pids:
                try:
                    if gb_synapse.stop(st["model"]):
                        stopped_servers.append(st["model"])
                except Exception:
                    pass   # fall through to the signal sweep below
    except Exception:
        pass

    killed: list[dict] = []
    failed: list[dict] = []
    for entry in targets:
        try:
            os.kill(entry["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            failed.append(entry)

    if targets:
        time.sleep(term_wait_s)

    for entry in targets:
        if entry in failed:
            continue
        pid = entry["pid"]
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            killed.append(entry)
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(entry)
        except (ProcessLookupError, PermissionError):
            failed.append(entry)

    # "spared" = real GPU/T2 holders this scope deliberately did NOT target
    # (genuinely-in-progress gb-synapse servers at scope="residue", say) —
    # without this, "0 killed" and "genuinely nothing is using GPU memory"
    # were indistinguishable to whoever ran the command. Real incident
    # (2026-08-01): scope=residue correctly spared a live gb-synapse server
    # holding 7GB VRAM + 6.7GB RAM, but the command's own output ("No
    # killable GPU inference processes found") read as if nothing was using
    # that memory at all.
    # Derived from the ACTUAL target set, never from a per-scope guess.
    #
    # This used to be `list(classes["live"])` plus ambiguous at scope=residue,
    # which silently assumed live entries are never targeted. scope="inference"
    # (and "all") DO target them, so the first real run printed PID 44218 under
    # "Terminated GPU inference processes" AND under "Still held (spared)" in
    # the same output — an outright contradiction the operator had to resolve
    # by reading the VRAM numbers (owner report 2026-08-18). Computing the
    # complement of `targets` cannot drift from the scope again, because it
    # does not know or care what the scope was.
    classes = plan["classes"]
    target_pids = {t["pid"] for t in targets}
    spared = [e for v in classes.values() for e in v
              if e["pid"] not in target_pids]
    result = {"scope": scope, "killed": killed, "unloaded": unloaded, "stopped_servers": stopped_servers, "failed": failed,
              "spared": spared}
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "kind": "reclaim", "status": "ok" if not failed else "partial",
            "scope": scope, "n_killed": len(killed), "n_unloaded": len(unloaded),
            "n_failed": len(failed), "targets": [t["comm"] for t in targets],
        })
    except Exception:
        pass
    return result


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(prog="gb_reclaim.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--scope", choices=_SCOPES, default="residue")
        sp.add_argument("--kill-min-mb", type=int, default=DEFAULT_KILL_MIN_MB)
        sp.add_argument("--json", action="store_true")
    args = p.parse_args()

    fn = plan_reclaim if args.cmd == "plan" else run_reclaim
    out = fn(scope=args.scope, kill_min_mb=args.kill_min_mb)
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        key = "targets" if args.cmd == "plan" else "killed"
        print(f"{args.cmd} scope={args.scope}: {len(out[key])} process(es)")
        for e in out[key]:
            print(f"  pid={e['pid']} comm={e['comm']} gpu_mb={e.get('gpu_mb', 0)}")

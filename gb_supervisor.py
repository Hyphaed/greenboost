#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_supervisor.py , GreenBoost unified system supervisor.

Replaces the four separate systemd daemons that shipped installed-but-inert:
  greenboost-recovery         oneshot boot-time crash recovery
  greenboost-sentinel         dirty-shutdown detection via file rename
  greenboost-vram-watchdog    physical VRAM pressure monitor (was polling nvidia-smi @3s)
  greenboost-idle-reclaim     graceful Ollama model unload on DEEP_IDLE

Key design improvements over the four-daemon design:
  - No nvidia-smi forks.  VRAM pressure is read from TelemetryManager.snapshot()
    backed by pynvml embedded calls , same path as gb_telemetry.  Zero subprocess
    overhead on the GPU query path.
  - Single sd_notify READY=1 sent AFTER boot recovery completes, so the
    Before=ollama.service boot ordering is preserved without an extra oneshot.
  - Process kill is opt-in (GB_SUPERVISOR_AGGRESSIVE_RECLAIM=1).  Default is
    Ollama-graceful-only (REST DELETE /api/delete with keep_alive:0).
  - The sentinel lifecycle is managed here; no separate unit needed.
  - System-wide RAM/swap pressure (/proc/meminfo , distinct from GreenBoost's
    own T2 DDR pool, which a CPU-offloaded PyTorch process bypasses entirely)
    is polled every tick and written to /var/lib/greenboost/ram_pressure ,
    visibility only, GreenBoost cannot stop an unrelated process from
    allocating RAM; see _RamMonitor.

Architecture: this is NOT imported by inference processes.  It is a standalone
daemon run as root by systemd.  It does not activate GREENBOOST_ACTIVE because
that path is for inference-process Python layers (gb_init, gb_quant, etc.).

Phase file contract (/run/greenboost/phase):
    Written by greenboost_cuda_shim.c:gb_write_phase_file() on every phase
    transition.  Key=value pairs: phase=DEEP_IDLE / idle_ms=N / pid=N / ts=N
    The supervisor reads this file every POLL_SECS to decide idle reclaim.

Usage (run by systemd as root):
    /usr/local/lib/greenboost/gb_supervisor.py
    Environment:
        GB_SUPERVISOR_POLL_SECS            int, default 10
        GB_SUPERVISOR_VRAM_WARN_PCT        int, default 10  (% of physical VRAM)
        GB_SUPERVISOR_VRAM_CRIT_FREE_PCT   int, default 8
        GB_SUPERVISOR_AGGRESSIVE_RECLAIM   1 to also SIGTERM non-Ollama processes
        GB_SUPERVISOR_OLLAMA_URL           default http://127.0.0.1:11434
        GB_SUPERVISOR_IDLE_CONFIRM_POLLS   consecutive DEEP_IDLE polls required, default 3
        GB_SUPERVISOR_RAM_WARN_AVAIL_PCT   % MemAvailable/MemTotal below which system RAM is "warn", default 20
        GB_SUPERVISOR_RAM_CRIT_AVAIL_PCT   % MemAvailable/MemTotal below which system RAM is "critical", default 8
        GB_SUPERVISOR_RAM_WARN_SWAP_PCT    % swap used above which system RAM is "warn", default 40
        GB_SUPERVISOR_RAM_CRIT_SWAP_PCT    % swap used above which system RAM is "critical", default 65
"""
from __future__ import annotations

import ctypes
import fcntl
import json
import logging
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Lazy imports at runtime to avoid heavy init at module load time
def _import_orchestrator():
    from gb_orchestrator import ReactiveOrchestrator
    return ReactiveOrchestrator

def _import_gpu_metrics():
    from gb_telemetry import GpuMetrics
    return GpuMetrics

def _import_meminfo_reader():
    # Best-effort, matching _NvmlHandle's gb_nvml fallback (:155-160): the
    # supervisor's primary duties (VRAM watchdog, boot recovery) must not be
    # blocked by losing the system-RAM-pressure signal alone. Returns None on
    # failure; _RamMonitor.poll() treats that as its documented "unknown" state.
    try:
        from gb_telemetry import read_meminfo_mb
        return read_meminfo_mb
    except Exception as exc:
        logging.getLogger("gb_supervisor").warning(
            "gb_telemetry unavailable (%s) , system RAM pressure will read 'unknown'", exc
        )
        return None

# ── Constants ────────────────────────────────────────────────────────────────

STATE_DIR       = Path("/var/lib/greenboost")
RUN_DIR         = Path("/run/greenboost")
SENTINEL_FILE   = STATE_DIR / "sentinel"
RUNNING_FILE    = STATE_DIR / "running"
PRESSURE_FILE   = STATE_DIR / "vram_pressure"
RAM_PRESSURE_FILE = STATE_DIR / "ram_pressure"
PHASE_FILE      = RUN_DIR / "phase"
RECOVERY_CLASS  = RUN_DIR / "recovery_class"
ECC_DBE_FLAG    = RUN_DIR / "ecc_dbe_flag"

# GB_IOCTL_RESET = _IO('G', 3) = 0x4703  (clears kernel OOM guard)
_GB_IOCTL_RESET = 0x4703
_GB_DEV         = "/dev/greenboost"

POLL_SECS     = int(os.environ.get("GB_SUPERVISOR_POLL_SECS",            "10"))
WARN_PCT      = int(os.environ.get("GB_SUPERVISOR_VRAM_WARN_PCT",         "10"))
CRIT_FREE_PCT = int(os.environ.get("GB_SUPERVISOR_VRAM_CRIT_FREE_PCT",    "8"))
AGGRESSIVE    = os.environ.get("GB_SUPERVISOR_AGGRESSIVE_RECLAIM", "0") == "1"
OLLAMA_URL    = os.environ.get("GB_SUPERVISOR_OLLAMA_URL", "http://127.0.0.1:11434")
_DATAFLUX_COMPACT_INTERVAL_S = int(os.environ.get("GB_DATAFLUX_COMPACT_INTERVAL_S", str(6 * 3600)))
CONFIRM_POLLS = int(os.environ.get("GB_SUPERVISOR_IDLE_CONFIRM_POLLS", "3"))
# Boot-recovery driver reload (SVC-01). Off switch, cool-down between reloads
# and the Xid codes that justify one. The default codes are the ones NVIDIA's
# Xid catalogue lists with the resolution "reset GPU": 95 (uncontained ECC),
# 119/120 (GSP RPC timeout/failure). Xid 79 (fell off the bus) is listed as
# "restart bare metal": a driver reload cannot bring the PCIe link back, so it
# is not in the default set. Application faults (13, 31, 43, 45) never justify
# unloading the driver.
XID_RESET_ENABLED  = os.environ.get("GB_SUPERVISOR_XID_RESET", "1") != "0"
XID_RESET_COOLDOWN_S = int(os.environ.get("GB_SUPERVISOR_XID_COOLDOWN_S", str(6 * 3600)))
XID_RESET_CODES = frozenset(
    int(c) for c in os.environ.get("GB_SUPERVISOR_XID_RESET_CODES", "95,119,120").split(",")
    if c.strip().isdigit())
XID_RESET_STAMP = Path("/var/lib/greenboost/xid_reset_ts")
# Orphan reaper: how many consecutive ticks an engine must be seen without
# its proxy before it is stopped (SVC-06). One tick is POLL_SECS.
REAP_CONFIRM_POLLS = int(os.environ.get("GB_SUPERVISOR_REAP_CONFIRM_POLLS", "3"))

# System-RAM pressure thresholds , %-of-node's-own-total, never absolute MB
# (this box's total RAM varies host vs feeder; see CLAUDE.md "no hardcoded
# hardware values" rule). Two independent signals, either one trips the
# state: MemAvailable falling low, or swap filling up , the 2026-07-14
# incident (two concurrent CPU-offloaded 22B-model runs) showed swap at
# ~68% used while GreenBoost's own T2 pool read 0 , CPU-offload streaming
# never goes through the shim's T2 cudaMallocManaged path, so this is a
# genuinely separate signal from vram_pressure/T2 pool pressure.
RAM_WARN_AVAIL_PCT  = int(os.environ.get("GB_SUPERVISOR_RAM_WARN_AVAIL_PCT",  "20"))
RAM_CRIT_AVAIL_PCT  = int(os.environ.get("GB_SUPERVISOR_RAM_CRIT_AVAIL_PCT",  "8"))
RAM_WARN_SWAP_PCT   = int(os.environ.get("GB_SUPERVISOR_RAM_WARN_SWAP_PCT",   "40"))
RAM_CRIT_SWAP_PCT   = int(os.environ.get("GB_SUPERVISOR_RAM_CRIT_SWAP_PCT",   "65"))

logging.basicConfig(
    level=logging.INFO,
    format="[gb_supervisor] %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("gb_supervisor")

# Set by the SIGTERM/SIGINT handler; the poll loop and every long wait sleep on
# it, so a stop request is honoured at once and an idle tick costs no wakeups.
_STOP = threading.Event()


# ── Systemd sd_notify ─────────────────────────────────────────────────────────

def _sd_notify(msg: str) -> None:
    """Send notification to systemd over the NOTIFY_SOCKET (Type=notify)."""
    sock_path = os.environ.get("NOTIFY_SOCKET", "")
    if not sock_path:
        return
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            if sock_path.startswith("@"):
                sock_path = "\0" + sock_path[1:]
            s.connect(sock_path)
            s.sendall(msg.encode())
    except Exception:
        pass


def _watchdog_interval_s() -> float:
    """Seconds between WATCHDOG=1 pings: half of WatchdogSec (sd_watchdog_enabled
    contract), 0.0 when the unit has no WatchdogSec or the variable is meant for
    another pid."""
    try:
        usec = int(os.environ.get("WATCHDOG_USEC", "0") or "0")
        pid = os.environ.get("WATCHDOG_PID", "")
        if usec <= 0 or (pid and int(pid) != os.getpid()):
            return 0.0
        return usec / 2_000_000.0
    except ValueError:
        return 0.0


def _notify_watchdog() -> None: _sd_notify("WATCHDOG=1\n")
def _extend_timeout(seconds: int) -> None:
    """Ask systemd for `seconds` more before it declares start/stop timed out
    (EXTEND_TIMEOUT_USEC). Boot recovery can run longer than TimeoutStartSec."""
    _sd_notify(f"EXTEND_TIMEOUT_USEC={int(seconds) * 1_000_000}\n")


def _atomic_write(path: Path, text: str, mode: int = 0o644) -> None:
    """Write `text` so a reader sees the old file or the new one, never half:
    temp file in the same directory, fsync, rename. Raises OSError on failure."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _notify_ready()     -> None: _sd_notify("READY=1\n")
def _notify_stopping()  -> None: _sd_notify("STOPPING=1\n")
def _notify_status(msg: str) -> None: _sd_notify(f"STATUS={msg}\n")


# ── NVML-based VRAM query (no nvidia-smi) ────────────────────────────────────

class _NVMLSampler:
    """
    Thin wrapper around gb_nvml.NvmlHandle.  Delegates all queries to the
    shared per-process singleton so nvmlInit/Shutdown happen exactly once.
    Interface preserved for all existing call sites.
    """
    def __init__(self, device: int = 0):
        try:
            from gb_nvml import get_nvml
            self._h = get_nvml(device)
        except Exception as exc:
            log.warning("gb_nvml unavailable (%s) , using GbInfo ioctl only", exc)
            self._h = None

    @property
    def ok(self) -> bool:
        return self._h is not None and self._h.ok

    def total_mb(self) -> int:
        return self._h.total_mb() if self._h else 0

    def query(self) -> tuple:
        """Return (used_mb, free_mb, gpu_util_pct).  0s on failure."""
        if not self._h:
            return 0, 0, 0.0
        used, free, _, _ = self._h.mem()
        gpu_util, _      = self._h.util()
        return used, free, gpu_util

    def ecc_dbe_volatile(self) -> int:
        return self._h.ecc_dbe_volatile() if self._h else 0

    def temp_c(self) -> float:
        """GPU temperature in Celsius (0.0 on failure)."""
        return self._h.temp_c() if self._h else 0.0

    def close(self) -> None:
        pass  # gb_nvml.get_nvml() singleton is shut down via atexit


# Module-level lazy sampler , avoids nvmlInit/Shutdown per-call handle churn.
# Created on first use; lives for the process lifetime.  Safe to call from
# recovery (before GreenBoostSupervisor is constructed).
_g_nvml_sampler: "_NVMLSampler | None" = None

def _get_nvml_sampler() -> "_NVMLSampler":
    global _g_nvml_sampler
    if _g_nvml_sampler is None:
        _g_nvml_sampler = _NVMLSampler()
    return _g_nvml_sampler


# ── GreenBoost sysfs helpers ──────────────────────────────────────────────────

def _read_gb_t1_mb() -> int:
    """Read T1 VRAM usage from GreenBoost sysfs (bytes → MiB, 0 if unavailable)."""
    sysfs = Path("/sys/class/greenboost/greenboost/status")
    try:
        text = sysfs.read_text()
        for line in text.splitlines():
            if line.startswith("t1_used_bytes:"):
                return int(line.split(":")[1].strip()) // (1 << 20)
    except Exception:
        pass
    return 0


def _read_gaming_mode() -> bool:
    """Return True if gaming_mode sysfs is 1 (Proton wrapper active in a game)."""
    try:
        return Path("/sys/module/greenboost/parameters/gaming_mode").read_text().strip() == "1"
    except Exception:
        return False


def _read_ecc_dbe_volatile() -> int:
    """Read volatile ECC DBE count via the shared NVML sampler (no re-init per call).
    Sup-H2: previously called nvmlInit() on every tick → handle churn/leak."""
    return _get_nvml_sampler().ecc_dbe_volatile()


# ── Phase file reader ─────────────────────────────────────────────────────────

def _read_phase() -> str:
    """Return the phase string from /run/greenboost/phase, or '' if absent."""
    try:
        phase, pid = "", ""
        for line in PHASE_FILE.read_text().splitlines():
            if line.startswith("phase="):
                phase = line.split("=", 1)[1].strip()
            elif line.startswith("pid="):
                pid = line.split("=", 1)[1].strip()
        # A writer that has exited leaves its last phase behind (SVC-04):
        # that is "unknown", never evidence for a reclaim.
        if pid.isdigit() and int(pid) > 0 and not os.path.exists(f"/proc/{pid}"):
            return ""
        return phase
    except Exception:
        pass
    return ""


# ── Driver reset guard (SVC-01) ───────────────────────────────────────────────

_XID_RE = re.compile(r"NVRM: Xid \(PCI:[^)]*\):\s*(\d+)")
_NVIDIA_MODULES = ("nvidia_drm", "nvidia_modeset", "nvidia_uvm", "nvidia")


def _fatal_xids(kernel_text: str, codes=None) -> list:
    """Xid codes in kernel-log text that are in the wedged-GPU set. Only
    `NVRM: Xid (PCI:...): N` lines count, so the supervisor's own log lines and
    unrelated journal text can never match."""
    codes = XID_RESET_CODES if codes is None else codes
    return [int(m) for m in _XID_RE.findall(kernel_text or "") if int(m) in codes]


def _nvidia_device_holders(proc_root: str = "/proc") -> list:
    """Pids (other than ours) holding a /dev/nvidia* node open. Reading other
    users' fd tables needs root, which the supervisor has."""
    holders = []
    try:
        entries = os.listdir(proc_root)
    except OSError:
        return holders
    me = str(os.getpid())
    for name in entries:
        if not name.isdigit() or name == me:
            continue
        try:
            for fd in os.listdir(f"{proc_root}/{name}/fd"):
                try:
                    if os.readlink(f"{proc_root}/{name}/fd/{fd}").startswith("/dev/nvidia"):
                        holders.append(int(name))
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return holders


def _loaded_nvidia_modules() -> list:
    try:
        text = Path("/proc/modules").read_text()
    except OSError:
        return []
    loaded = {line.split()[0] for line in text.splitlines() if line.strip()}
    return [m for m in _NVIDIA_MODULES if m in loaded]


def _reset_nvidia_driver(codes: list, now: "float | None" = None) -> str:
    """Reload the NVIDIA kernel modules after a wedged-GPU Xid. Returns what
    happened: "reset", "disabled", "cooldown", "in_use" or "failed".

    Guards, each tested in tests/test_gb_supervisor_xid.py:
      1. GB_SUPERVISOR_XID_RESET=0 switches it off.
      2. At most one reload per XID_RESET_COOLDOWN_S (persisted stamp), so a
         crash loop cannot become a driver-reload loop.
      3. Nothing holds /dev/nvidia* open (a desktop, a model server, a job).
         rmmod without -f would refuse anyway, but several modules are removed
         in one call and a partial unload leaves display modules gone.
      4. The modules that were loaded are all brought back through modprobe of
         the top module, so a half-done reload is not left behind.
    """
    now = time.time() if now is None else now
    if not XID_RESET_ENABLED:
        log.warning("Xid %s seen, driver reload disabled (GB_SUPERVISOR_XID_RESET=0)", codes)
        return "disabled"
    try:
        last = float(XID_RESET_STAMP.read_text().strip())
    except (OSError, ValueError):
        last = None
    if last is not None and 0 <= now - last < XID_RESET_COOLDOWN_S:
        log.warning("Xid %s seen, driver reloaded %.0f s ago, cool-down %d s: not again",
                    codes, now - last, XID_RESET_COOLDOWN_S)
        return "cooldown"
    holders = _nvidia_device_holders()
    if holders:
        log.warning("Xid %s seen, but pids %s hold /dev/nvidia*: driver not reloaded",
                    codes, holders[:8])
        return "in_use"
    was_loaded = _loaded_nvidia_modules()
    log.warning("Xid %s (wedged GPU): reloading the NVIDIA driver (%s)", codes, was_loaded)
    try:
        _atomic_write(XID_RESET_STAMP, str(now))   # stamp first: a crash mid-reload still counts
    except OSError as exc:
        log.warning("reset stamp not written (%s): driver not reloaded", exc)
        return "failed"
    ok = True
    try:
        r = subprocess.run(["rmmod", *[m for m in _NVIDIA_MODULES if m in was_loaded]],
                           capture_output=True, timeout=20)
        ok = r.returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
        log.warning("rmmod nvidia timed out , continuing")
    try:
        top = "nvidia_drm" if "nvidia_drm" in was_loaded else "nvidia"
        subprocess.run(["modprobe", top], capture_output=True, timeout=20)
    except subprocess.TimeoutExpired:
        ok = False
        log.warning("modprobe nvidia timed out , continuing")
    try:
        import gb_dataflux
        gb_dataflux.emit({"kind": "driver_reset", "node": "host", "label": "supervisor",
                          "status": "ok" if ok else "error", "xid_codes": codes,
                          "modules": was_loaded})
    except Exception:
        pass
    return "reset" if ok else "failed"


def _apparmor_denies_stub(journal_text: str) -> bool:
    """True when ONE journal line is an AppArmor denial that names the
    GreenBoost audit stub (the only thing _patch_apparmor can fix)."""
    for line in (journal_text or "").splitlines():
        if 'apparmor="DENIED"' in line and "libgreenboost_audit" in line:
            return True
    return False


# ── Boot recovery ─────────────────────────────────────────────────────────────

def _run_recovery() -> str:
    """
    Run the boot-time crash recovery sequence.  Returns the fault class string.
    Mirrors greenboost-recover logic, now as Python so the supervisor owns it.
    """
    fault = "unknown"
    # Worst case below is ~100 s (lsmod 5 + modprobe 15 + journalctl 3 x 10 +
    # cooling pause 10 + rmmod 20 + modprobe 20) against TimeoutStartSec=90.
    _extend_timeout(180)

    # 0. Dirty-shutdown detection
    dirty = SENTINEL_FILE.exists() or RUNNING_FILE.exists()
    if dirty:
        log.warning("Dirty shutdown detected , running full recovery sequence")
    else:
        log.info("No dirty-shutdown sentinel , last boot was clean")

    # 1. Kernel module check
    # Sup-M2: add timeouts so a hung subprocess can't block READY=1 indefinitely.
    try:
        lsmod = subprocess.run(["lsmod"], capture_output=True, text=True, timeout=5)
        if "greenboost" in lsmod.stdout:
            log.info("greenboost.ko loaded")
        else:
            log.warning("greenboost.ko not loaded , attempting modprobe")
            try:
                r = subprocess.run(["modprobe", "greenboost"],
                                    capture_output=True, text=True, timeout=15)
                if r.returncode == 0:
                    log.info("greenboost.ko loaded via modprobe")
                else:
                    log.error("modprobe greenboost failed: %s", r.stderr.strip())
                    # Non-fatal: Path B still works without the module
            except subprocess.TimeoutExpired:
                log.warning("modprobe greenboost timed out , continuing without module")
    except subprocess.TimeoutExpired:
        log.warning("lsmod timed out , skipping module check")
    except Exception as exc:
        log.warning("lsmod/modprobe check failed: %s", exc)

    # 2. OOM guard reset
    if os.path.exists(_GB_DEV):
        try:
            fd = os.open(_GB_DEV, os.O_RDWR)
            try:
                fcntl.ioctl(fd, _GB_IOCTL_RESET)
                log.info("OOM guard cleared (GB_IOCTL_RESET)")
            finally:
                os.close(fd)
        except Exception as exc:
            log.warning("GB_IOCTL_RESET failed (non-fatal): %s", exc)
    else:
        log.info("%s not present , skipping OOM reset (Path B mode)", _GB_DEV)

    # 2b. /run/greenboost permissions
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import grp
        try:
            gid = grp.getgrnam("ollama").gr_gid
            os.chown(str(RUN_DIR), -1, gid)
            os.chmod(str(RUN_DIR), 0o775)
        except KeyError:
            os.chmod(str(RUN_DIR), 0o755)
    except Exception:
        pass

    # 2c. Fault classification from recent journal
    # Sup-M3: call journalctl once each (all + kernel-only), cache results.
    since = "30 minutes ago"
    def _jq(*args: str) -> str:
        try:
            r = subprocess.run(
                ["journalctl", "--since", since, "--no-pager", "-q"] + list(args),
                capture_output=True, text=True, timeout=10,
            )
            return r.stdout
        except Exception:
            return ""

    jq_all    = _jq()          # all journals
    jq_kernel = _jq("-k")     # kernel ring (dmesg equivalent)

    # Sup-H1: use re.search for regex-shaped patterns , plain `in` treated them
    # as literal substrings so ECC/thermal recovery was effectively dead.
    if "Killed process" in jq_kernel and "oom" in jq_kernel.lower():
        fault = "oom_kill"
    elif _fatal_xids(jq_kernel):
        # SVC-01: was `"Xid" in jq_all and "GPU" in jq_all` over EVERY journal
        # line, which also matched this supervisor's own "Xid GPU fault" log
        # line (a restart inside 30 minutes re-triggered the driver reload) and
        # any unrelated line containing both words. Now: a kernel `NVRM: Xid`
        # line whose code is in the wedged-GPU set, nothing else.
        fault = "xid_fault"
    elif any(re.search(k, jq_all, re.I) for k in (
            r"ecc.*double.bit", r"uncorrectable.*ecc",
            r"nvml.*dbe", r"ecc double", r"uncorrectable ecc")):
        fault = "ecc_dbe"
    elif any(re.search(k, jq_all, re.I) for k in (
            r"thermal throttl", r"nvidia.*thermal",
            r"gpu.*temp.*critical")):
        fault = "thermal"
    elif _apparmor_denies_stub(jq_all):
        # SVC-07: was three independent substring tests over the whole journal
        # ("DENIED", "greenboost", "apparmor" on any lines), so an unrelated
        # DENIED line plus any greenboost line plus any apparmor line edited
        # the security policy and restarted apparmor on every start.
        fault = "apparmor"

    try:
        _atomic_write(RECOVERY_CLASS, fault)
    except OSError as exc:
        log.warning("recovery_class not written: %s", exc)
    log.info("Fault class: %s", fault)

    # 2d. Per-fault remediation
    if fault == "ecc_dbe":
        log.warning("ECC double-bit error , setting ecc_dbe_flag to reduce T1 quota")
        try:
            _atomic_write(ECC_DBE_FLAG, "1")
        except OSError as exc:
            log.warning("ecc_dbe_flag not written: %s", exc)
    elif fault == "thermal":
        log.warning("Thermal fault , inserting 10 s cooling pause before Ollama restart")
        _STOP.wait(10)
    elif fault == "apparmor":
        log.warning("AppArmor denial , patching snap-confine profiles")
        _patch_apparmor()
    elif fault == "xid_fault":
        _reset_nvidia_driver(sorted(set(_fatal_xids(jq_kernel))))

    # 3. Sentinel cleanup
    SENTINEL_FILE.unlink(missing_ok=True)
    RUNNING_FILE.unlink(missing_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _atomic_write(STATE_DIR / "last_clean_boot", time.strftime("%Y-%m-%dT%H:%M:%S"))
    except OSError as exc:
        log.warning("last_clean_boot not written: %s", exc)
    log.info("Recovery complete (fault class: %s)", fault)
    return fault


def _patch_apparmor() -> None:
    """AppArmor remediation: allow snap-confine to mmap the LD_AUDIT stub.

    ONE mechanism only: the `local/` override, which snapd never rewrites and
    which apparmor includes INSIDE the profile block.

    What is deliberately not here, and must not be re-added: this function used
    to also append the same two rules to every
    `/var/lib/snapd/apparmor/profiles/snap-confine.snapd.*`. Those are complete
    snapd-generated profiles. An append lands PAST the profile's final closing
    brace, at top level, where apparmor_parser expects `profile NAME {` , so it
    grants nothing and makes the entire profile unparseable:

        syntax error, unexpected TOK_MODE, expecting TOK_OPEN

    greenboost_setup.sh:3239-3248 already documents this as a confirmed live
    incident from 2026-07-27 that "took down Firefox, Chromium, snap-store, cups
    system-wide". The installer was fixed then; this function was not, and
    reintroduced the identical failure on 2026-08-20 , two contaminated
    profiles, `snapd.apparmor.service` dead, `systemctl is-system-running`
    stuck at `degraded`.

    The trap that makes this recur: `grep libgreenboost_audit <profile>`
    returning a hit reads like "the grant is active". It is the opposite. A rule
    past the closing brace grants nothing AND invalidates the whole profile, so
    the presence of the string is the failure signal.
    """
    local_sc = Path("/etc/apparmor.d/local/usr.lib.snapd.snap-confine.real")
    header = "# GreenBoost: allow snap-confine to mmap the LD_AUDIT stub"
    libs = [
        "/usr/local/lib/libgreenboost_audit.so mr,",
        "/usr/local/lib/x86_64-linux-gnu/libgreenboost_audit.so mr,",
    ]
    written = False
    reason = "already present"
    try:
        if not local_sc.parent.is_dir():
            # No local/ directory means no supported injection point. Do NOT
            # fall back to the snapd profile directory , that fallback is the
            # bug above.
            reason = "no /etc/apparmor.d/local directory , nothing supported to patch"
        else:
            # Create when absent. The old code skipped entirely unless the file
            # already existed, which is exactly this box's state on 2026-08-21
            # and why the supported override was missing while the unsupported
            # one had been applied twice.
            text = local_sc.read_text() if local_sc.exists() else ""
            if "libgreenboost_audit" not in text:
                with local_sc.open("a") as f:
                    if text and not text.endswith("\n"):
                        f.write("\n")
                    f.write(header + "\n" + "\n".join(libs) + "\n")
                written = True
                reason = "local/ override written"
    except OSError as e:
        reason = f"local/ override unwritable: {e}"

    parser_ok = None
    # SVC-07: reload policy only when this call changed it. Re-parsing and
    # restarting apparmor for a rule that was already present reloaded every
    # profile on the box for nothing.
    if written:
        try:
            r = subprocess.run(
                ["apparmor_parser", "-r",
                 "/etc/apparmor.d/usr.lib.snapd.snap-confine.real"],
                capture_output=True, timeout=15,
            )
            parser_ok = (r.returncode == 0)
        except (Exception, subprocess.TimeoutExpired):
            parser_ok = None

    # Observability Must-Rule: this function modifies system security policy and
    # used to emit nothing at all, which is why a supervisor re-contaminating
    # AppArmor could only be found by reading a journal.
    try:
        import gb_dataflux
        gb_dataflux.emit({
            "kind": "apparmor_remediation",
            "files_written": [str(local_sc)] if written else [],
            "parser_ok": parser_ok,
            "snapd_profiles_skipped": True,
            "reason": reason,
        })
    except Exception:
        pass

    if written:
        try:
            subprocess.run(["systemctl", "restart", "apparmor"],
                           capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            log.warning("systemctl restart apparmor timed out")


# ── Idle reclaim ──────────────────────────────────────────────────────────────

def _ollama_unload_models(ollama_url: str) -> bool:
    """
    Ask Ollama to unload all running models gracefully via DELETE /api/delete
    with keep_alive=0.  Returns True if at least one model was unloaded.
    Falls back to listing via /api/tags and unloading each.
    """
    import urllib.request, urllib.error
    try:
        req = urllib.request.Request(
            f"{ollama_url}/api/ps",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        log.warning("Ollama /api/ps failed: %s", exc)
        return False

    models = data.get("models", [])
    if not models:
        log.info("No running Ollama models to unload")
        return False

    unloaded = 0
    for m in models:
        name = m.get("name") or m.get("model", "")
        if not name:
            continue
        try:
            import gb_lease_client
            _held = gb_lease_client.unload_blocked(name)
        except Exception:
            _held = None
        if _held:
            log.info("Ollama unload of %s skipped: GPU lease held by %s", name, _held.get("client") or _held.get("job"))
            continue
        try:
            # Real bug, 2026-07-14: ollama's /api/generate model-selector
            # field is "model", not "name" , the previous payload sent
            # {"name": ..., "keep_alive": 0}, which ollama parses as an
            # empty model ("" , the field it actually looks for is absent)
            # and returns 404 "model '' not found" every single time. Since
            # urlopen raises HTTPError on that 404, `unloaded` was correctly
            # never incremented and this function correctly reported
            # failure , but that means the graceful reclaim path has NEVER
            # actually unloaded a model, on any machine, since this was
            # written: every DEEP_IDLE-triggered reclaim silently fell
            # through to whatever AGGRESSIVE_RECLAIM/no-op branch follows
            # instead of the intended graceful unload. Confirmed via a
            # direct manual repro of the exact request this function builds.
            payload = json.dumps({"model": name, "keep_alive": 0}).encode()
            req = urllib.request.Request(
                f"{ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                pass
            log.info("Ollama model unloaded: %s", name)
            unloaded += 1
        except Exception as exc:
            log.warning("Ollama unload %s failed: %s", name, exc)

    return unloaded > 0


def _aggressive_reclaim() -> None:
    """
    Opt-in aggressive reclaim: SIGTERM every process with /dev/greenboost open.
    Only used when GB_SUPERVISOR_AGGRESSIVE_RECLAIM=1.  The Ollama-graceful path
    is always tried first; this only fires if that didn't clear the DEEP_IDLE phase.
    """
    try:
        result = subprocess.run(
            ["fuser", "/dev/greenboost"],
            capture_output=True, text=True,
        )
        pids = result.stdout.strip().split()
        for pid_str in pids:
            try:
                pid = int(pid_str)
                os.kill(pid, signal.SIGTERM)
                log.warning("Aggressive reclaim: sent SIGTERM to pid %d", pid)
            except (ValueError, ProcessLookupError):
                pass
    except Exception as exc:
        log.warning("Aggressive reclaim fuser failed: %s", exc)


# ── VRAM pressure monitor ────────────────────────────────────────────────────

class _VramMonitor:
    """
    Tracks VRAM pressure using NVMLSampler (no nvidia-smi forks).
    Reads T1 GreenBoost usage from sysfs to compute non-GB consumers.
    Writes /var/lib/greenboost/vram_pressure on state change.
    When gaming_mode=1 the thresholds are doubled (game takes priority).
    """
    def __init__(self, nvml: _NVMLSampler,
                 warn_pct: int = 10, crit_free_pct: int = 8) -> None:
        self._nvml          = nvml
        self._total_mb      = nvml.total_mb()
        self._base_warn_mb  = self._total_mb * warn_pct // 100
        self._base_crit_mb  = self._total_mb * crit_free_pct // 100
        self._warn_mb       = self._base_warn_mb
        self._crit_free_mb  = self._base_crit_mb
        self._prev_state    = "ok"
        log.info("VramMonitor: GPU %d MiB total, warn non-GB > %d MiB, crit free < %d MiB",
                 self._total_mb, self._warn_mb, self._crit_free_mb)

    def poll(self, gaming_mode: bool = False) -> tuple[str, float]:
        """
        Poll VRAM state.  Returns (state, gpu_util_pct).
        state ∈ {'ok', 'warn', 'critical'}.
        gaming_mode doubles thresholds so games get generous VRAM headroom.
        """
        if self._total_mb <= 0:
            return "ok", 0.0, 0

        # Adjust thresholds for gaming mode (double the reserve)
        multiplier = 2 if gaming_mode else 1
        self._warn_mb      = self._base_warn_mb * multiplier
        self._crit_free_mb = self._base_crit_mb * multiplier

        used_mb, free_mb, util_pct = self._nvml.query()
        if used_mb == 0 and free_mb == 0:
            return "ok", 0.0, 0

        gb_t1_mb  = _read_gb_t1_mb()
        non_gb_mb = max(0, used_mb - gb_t1_mb)

        if free_mb < self._crit_free_mb:
            state = "critical"
        elif non_gb_mb > self._warn_mb:
            state = "warn"
        else:
            state = "ok"

        if state != self._prev_state:
            if state == "critical":
                log.critical(
                    "CRITICAL: free VRAM %d MiB < threshold %d MiB%s , "
                    "non-GB consumers: %d MiB (browser NVDEC / compositor?). "
                    "Consider closing GPU-heavy apps or increasing GREENBOOST_WORKSTATION_RESERVE_MB.",
                    free_mb, self._crit_free_mb,
                    " [gaming_mode doubled]" if gaming_mode else "", non_gb_mb,
                )
            elif state == "warn":
                log.warning(
                    "WARN: non-GB GPU usage %d MiB > %d MiB threshold%s , "
                    "free VRAM %d MiB.",
                    non_gb_mb, self._warn_mb,
                    " [gaming_mode doubled]" if gaming_mode else "", free_mb,
                )
            else:
                log.info("OK: physical VRAM pressure cleared , used %d MiB, free %d MiB",
                         used_mb, free_mb)
            self._prev_state = state

        # Always write the pressure file (greenboost vitals/status reads it)
        try:
            _atomic_write(PRESSURE_FILE,
                f"state={state}\n"
                f"used_mib={used_mb}\n"
                f"free_mib={free_mb}\n"
                f"non_gb_mib={non_gb_mb}\n"
                f"total_mib={self._total_mb}\n"
                f"gaming_mode={int(gaming_mode)}\n"
                f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
            )
        except Exception:
            pass

        return state, util_pct, non_gb_mb


# ── System RAM pressure monitor ──────────────────────────────────────────────

class _RamMonitor:
    """
    Tracks system-wide RAM/swap pressure , NOT GreenBoost's own T2 DDR pool
    (that is pre-committed by greenboost.ko and tracked separately via
    _read_gb_t1_mb-style sysfs/GbInfo reads). This monitor exists because a
    CPU-offloaded PyTorch process (e.g. `--offload cpu` streaming weights
    through host RAM) consumes ordinary system RAM/swap without the shim
    ever seeing it , GreenBoost's T2 pool accounting reads 0 while the box
    is genuinely under RAM pressure from that process (confirmed real
    incident 2026-07-14: two concurrent CPU-offloaded 22B-model runs drove
    swap to ~68% used with tier_t2_local_cur_mb=0 the whole time).

    This monitor gives GreenBoost (and any external caller , ai-forge or a
    human) VISIBILITY into that pressure so a NEW heavy job can check before
    launching. GreenBoost has no authority to stop an unrelated process from
    allocating RAM , this is a signal, not an admission-control enforcement
    mechanism.

    Writes /var/lib/greenboost/ram_pressure on state change (mirrors the
    vram_pressure file precedent).
    """
    def __init__(self, warn_avail_pct: int = 20, crit_avail_pct: int = 8,
                 warn_swap_pct: int = 40, crit_swap_pct: int = 65) -> None:
        self._warn_avail_pct = warn_avail_pct
        self._crit_avail_pct = crit_avail_pct
        self._warn_swap_pct  = warn_swap_pct
        self._crit_swap_pct  = crit_swap_pct
        self._prev_state     = "ok"
        self._read_meminfo   = _import_meminfo_reader()

    def poll(self) -> str:
        """Poll /proc/meminfo. Returns state ∈ {'ok', 'warn', 'critical', 'unknown'}."""
        if self._read_meminfo is None:
            return "unknown"
        mem_total_mb, mem_avail_mb, swap_total_mb, swap_used_mb = self._read_meminfo()
        if mem_total_mb <= 0:
            return "unknown"

        avail_pct = 100.0 * mem_avail_mb / mem_total_mb
        swap_pct  = (100.0 * swap_used_mb / swap_total_mb) if swap_total_mb > 0 else 0.0

        if avail_pct < self._crit_avail_pct or swap_pct > self._crit_swap_pct:
            state = "critical"
        elif avail_pct < self._warn_avail_pct or swap_pct > self._warn_swap_pct:
            state = "warn"
        else:
            state = "ok"

        if state != self._prev_state:
            prev = self._prev_state
            if state in ("critical", "warn"):
                log.log(
                    logging.CRITICAL if state == "critical" else logging.WARNING,
                    "%s: system RAM available %.1f%% (< %d%%?) , swap used %.1f%% (> %d%%?). "
                    "Non-GreenBoost RAM/swap consumer (e.g. CPU-offloaded inference) , "
                    "consider not starting another heavy CPU-offload job right now.",
                    state.upper(), avail_pct, self._warn_avail_pct, swap_pct, self._warn_swap_pct,
                )
            else:
                log.info("OK: system RAM pressure cleared , available %.1f%%, swap used %.1f%%",
                         avail_pct, swap_pct)
            # Observability Must-Rule: a pressure transition that exists only
            # in the journal is invisible to dataflux, to GB-Semantics and to
            # any agent asking "is this box about to fall over". On 2026-08-18
            # this monitor went critical twice (MemAvailable 4044 MB = 6.6% of
            # 61 GB, past its own 8% floor) roughly ten minutes before the OOM
            # killer took the user's terminal, and said so only here.
            try:
                import gb_dataflux
                # emit() takes ONE dict, not kwargs. Getting that wrong raised
                # TypeError into the best-effort `except` below, so this event
                # was silently never written — a signal with no reader, which is
                # the exact defect this emit exists to fix. Regression test:
                # tests/test_supervisor_host_mem_emit.py.
                gb_dataflux.emit({
                    "kind": "host_mem_pressure",
                    "node": "host",
                    "label": "gb_supervisor",
                    "status": ("error" if state == "critical"
                               else "warn" if state == "warn" else "ok"),
                    "state": state,
                    "prev_state": prev,
                    "mem_available_mib": int(mem_avail_mb),
                    "mem_available_pct": round(avail_pct, 1),
                    "mem_total_mib": int(mem_total_mb),
                    "swap_used_pct": round(swap_pct, 1),
                    "swap_used_mib": int(swap_used_mb),
                    # WHO is eating the memory, captured at the moment pressure
                    # changes rather than after the OOM killer has erased the
                    # evidence. Overnight 2026-08-19 a python3 reached 39 GB and
                    # was killed at 01:47; by morning /proc/<pid> was gone and
                    # the kernel report named only "python3", so the cause could
                    # not be established at all. gb_memwatch.py exists for this
                    # but has to be started by hand, and nobody is awake at 01:47.
                    "top_consumers": _top_memory_consumers(),
                })
            except Exception:
                pass    # emit is best-effort and must never break the tick
            self._prev_state = state

        try:
            _atomic_write(RAM_PRESSURE_FILE,
                f"state={state}\n"
                f"mem_total_mib={mem_total_mb}\n"
                f"mem_available_mib={mem_avail_mb}\n"
                f"mem_available_pct={avail_pct:.1f}\n"
                f"swap_total_mib={swap_total_mb}\n"
                f"swap_used_mib={swap_used_mb}\n"
                f"swap_used_pct={swap_pct:.1f}\n"
                f"timestamp={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
            )
        except Exception:
            pass

        return state


# ── DAMON host cold-page evidence ─────────────────────────────────────────────

class _DamonMonitor:
    """Republishes the kernel's DAMON_STAT cold-page evidence for the tiering
    layer, which cannot read it itself.

    DAMON_STAT's parameters are mode 0400 under /sys/module/damon_stat, so only
    root can read them. The supervisor already runs as root, so it reads and
    writes a world-readable summary to /var/lib/greenboost/damon_cold - the
    same publish pattern _RamMonitor and _VramMonitor use. The alternative
    (running the tiering layer as root, or shelling to sudo) would have been
    worse for a read-only telemetry signal.

    Purely observational. Nothing here evicts or repositions anything; the
    point is that placement stops being the only thing in GreenBoost with no
    measured input. Emits to dataflux only when the cold share moves
    materially, so a quiet box does not spam the flight recorder.
    """

    #  Percentage-point change in the 400s-idle share worth recording.
    _EMIT_DELTA_PP = 5.0
    _COLD_MS = 400_000

    def __init__(self) -> None:
        self._prev_cold_pct: "float | None" = None
        self._warned_unreadable = False

    def poll(self) -> "dict | None":
        try:
            import gb_damon
        except Exception:
            return None

        if not gb_damon.available():
            return None            # kernel without CONFIG_DAMON_STAT , not an error

        snap = gb_damon.read_snapshot()
        if snap.error and not snap.idle_ms:
            if not self._warned_unreadable:
                log.warning("DAMON_STAT present but unreadable (%s)", snap.error)
                self._warned_unreadable = True
            return None

        try:
            gb_damon.PUBLISHED_FILE.write_text(gb_damon.render_published(snap))
            os.chmod(gb_damon.PUBLISHED_FILE, 0o644)
        except Exception:
            pass    # publishing is best-effort and must never break the tick

        cold_pct = snap.cold_pct_above(self._COLD_MS)
        if (self._prev_cold_pct is None
                or abs(cold_pct - self._prev_cold_pct) >= self._EMIT_DELTA_PP):
            try:
                import gb_dataflux
                gb_dataflux.emit({
                    "node": "host", "label": "gb_damon", "kind": "damon_cold",
                    "n_items": 1, "items": [], "duration_s": 0, "status": "ok",
                    "hot_pct": round(snap.hot_pct, 1),
                    "idle_pct": round(snap.idle_pct, 1),
                    "cold_pct_400s": round(cold_pct, 1),
                    "est_bandwidth_bps": snap.est_bandwidth_bps,
                })
            except Exception:
                pass
            self._prev_cold_pct = cold_pct

        return snap.as_dict()


# ── Main supervisor loop ──────────────────────────────────────────────────────

def _top_memory_consumers(n: int = 5) -> list:
    """The n largest RSS processes, with cmdline. Best-effort, never raises.

    Deliberately cheap , this runs inside a pressure transition, on a box that
    is by definition short of memory, so it reads /proc/<pid>/statm and stops.
    No smaps walk, no sorting of the whole process table twice.
    """
    try:
        page_kb = os.sysconf("SC_PAGE_SIZE") // 1024
        rows = []
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                with open(f"/proc/{entry.name}/statm") as f:
                    rss_kb = int(f.read().split()[1]) * page_kb
            except (OSError, IndexError, ValueError):
                continue
            if rss_kb:
                rows.append((rss_kb, int(entry.name)))
        rows.sort(reverse=True)
        out = []
        for rss_kb, pid in rows[:n]:
            try:
                with open(f"/proc/{pid}/cmdline") as f:
                    cmd = f.read().replace("\0", " ").strip()[:200]
            except OSError:
                cmd = ""
            out.append({"pid": pid, "rss_mb": round(rss_kb / 1024, 1), "cmd": cmd})
        return out
    except Exception:
        return []


class GreenBoostSupervisor:
    def __init__(self) -> None:
        self._running       = True
        self._idle_count    = 0   # consecutive DEEP_IDLE polls confirmed
        self._nvml          = _NVMLSampler(device=0)
        self._vram_mon      = _VramMonitor(self._nvml, WARN_PCT, CRIT_FREE_PCT)
        self._ram_mon       = _RamMonitor(RAM_WARN_AVAIL_PCT, RAM_CRIT_AVAIL_PCT,
                                           RAM_WARN_SWAP_PCT, RAM_CRIT_SWAP_PCT)
        self._damon_mon     = _DamonMonitor()
        self._orphan_polls: dict = {}   # model -> consecutive ticks seen proxyless
        self._ecc_dbe_seen  = 0   # last known volatile DBE count (detect new errors)
        # Reactive orchestrator , closes dead feedback loops (ECC, workstation, thermal)
        try:
            ReactiveOrchestrator = _import_orchestrator()
            self._orch = ReactiveOrchestrator(mode="supervisor")
        except Exception as exc:
            log.warning("ReactiveOrchestrator unavailable (%s) , signals disabled", exc)
            self._orch = None

        # Supplemental telemetry providers , fill the GpuMetrics passed to the
        # orchestrator each tick with the fields Loops O-S need (shim_phase,
        # T2/T3 pool, host PSI pressure, power_limit_w/sm_clock for GPU clock
        # locks) that the hand-built metrics above don't carry. Constructed
        # once here (not per-tick) to avoid repeated NVML init churn.
        self._gb_provider = None
        self._sys_provider = None
        self._nvml_provider = None
        try:
            from gb_telemetry import GreenBoostProvider, SystemProvider, NVMLProvider
            gbp = GreenBoostProvider()
            self._gb_provider = gbp if gbp.available() else None
            sysp = SystemProvider()
            self._sys_provider = sysp if sysp.available() else None
            nvp = NVMLProvider(device=0)
            self._nvml_provider = nvp if nvp.available() else None
        except Exception as exc:
            log.debug("Supplemental telemetry providers unavailable: %s", exc)

        # Always-on dataflux flight recorder. gb_init's SnapshotRecorder lives
        # inside transient INFERENCE processes, so system_snapshot history had
        # holes whenever no shim process was alive (verified 2026-07-13:
        # snapshots stopped at 05:13 with the supervisor healthy — dataflux
        # was blind for 7 h of active work). The supervisor already builds an
        # enriched GpuMetrics every tick; feed the same object to a recorder
        # so the baseline time-series never stops. Inference processes still
        # layer their own 500ms-poll recorder for fine-grained transitions.
        self._snap_recorder = None
        if os.environ.get("GREENBOOST_DATAFLUX", "1") != "0":
            try:
                import gb_dataflux

                class _NoSub:                      # recorder pulls, we push
                    def add_callback(self, cb):    # noqa: D401
                        pass

                self._snap_recorder = gb_dataflux.SnapshotRecorder(
                    _NoSub(), interval_s=float(POLL_SECS), node="host")
            except Exception as exc:
                log.debug("dataflux recorder unavailable: %s", exc)

        # Periodic dataflux archive compaction , emit()'s rotation only ever
        # moves bytes out of the hot write path (a size-capped, gzip-compressed
        # ".jsonl.1.gz"), it never deletes anything; something has to apply the
        # GB_DATAFLUX_RETAIN_DAYS trim, and it belongs on this already-
        # installed daemon's tick, not a new systemd unit (no new install/
        # purge parity debt for a periodic housekeeping task). Gated to run
        # at most once per _DATAFLUX_COMPACT_INTERVAL_S , rewriting a large
        # archive on every 10s tick would be its own performance problem.
        self._last_dataflux_compaction_ts = 0.0

    def run(self) -> None:
        # Handlers first: a stop request during boot recovery is honoured at
        # the next wait instead of being ignored until READY=1.
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._on_stop)

        # Phase 0: ensure state directories exist
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        RUN_DIR.mkdir(parents=True, exist_ok=True)

        # Phase 1: boot-time recovery (runs before READY=1, before Ollama)
        _notify_status("running boot recovery")
        _run_recovery()

        # Phase 2: lifecycle sentinel , touch running file
        RUNNING_FILE.write_text(str(os.getpid()))

        # Signal systemd we are ready (Ollama may start now , Before= ordering)
        _notify_ready()
        _notify_status("monitoring VRAM + idle reclaim")
        log.info("GreenBoost supervisor ready (poll=%ds, warn_pct=%d, crit_free_pct=%d, aggressive=%s)",
                 POLL_SECS, WARN_PCT, CRIT_FREE_PCT, AGGRESSIVE)

        # Phase 3: steady-state poll loop. One blocking wait per tick (SVC-02:
        # the old 0.1 s sleep loop woke the process 10 times a second, measured
        # 623 context switches in 60 s on an idle card). With WatchdogSec set,
        # ping at half the interval; a tick that hangs stops the pings and
        # systemd restarts the unit.
        wd = _watchdog_interval_s()
        self._wd = wd
        while self._running:
            try:
                self._tick()
            except Exception as exc:
                log.exception("Tick error: %s", exc)
            self._wait_next_tick(wd)

        self._shutdown()

    def _wait_next_tick(self, wd_interval: float) -> None:
        """Sleep POLL_SECS (interruptible), pinging the watchdog on the way."""
        deadline = time.monotonic() + POLL_SECS
        while self._running:
            if wd_interval:
                _notify_watchdog()
            left = deadline - time.monotonic()
            if left <= 0:
                return
            if _STOP.wait(min(left, wd_interval) if wd_interval else left):
                return

    def _reap_orphaned_serves(self) -> None:
        """Stop an engine whose proxy has died, so it releases its VRAM.

        The proxy is a stateless passthrough and the engine is the expensive
        part , but an engine with no proxy is unreachable, and it holds the
        whole card while being unreachable. Overnight 2026-08-19: the proxy was
        OOM-killed at 01:47, llama-server survived until 10:26 holding 10.5 GB
        of a 12 GB card, and every request in between failed with "Cannot
        connect to :11369". Nothing noticed for eight hours.

        Deliberately conservative: it acts only when gb_synapse itself reports
        `proxy_error` AND the engine pid is alive, and it only ever STOPS. It
        does not restart the serve , choosing a model and a context is the
        operator's decision, and guessing one at 02:00 is how a box ends up
        serving something nobody asked for.
        """
        try:
            import gb_synapse
        except Exception:
            return
        try:
            sessions = gb_synapse.ps()
        except Exception:
            return
        seen_now = set()
        for sess in sessions:
            if not sess.get("proxy_error") or not sess.get("llama_pid"):
                continue
            model = sess.get("model") or ""
            # SVC-06: a proxy-only restart (gb_synapse.serve) passes through a
            # moment with the engine alive and no proxy, and the engine may be
            # minutes into loading a model. Stop it only when the condition
            # holds for REAP_CONFIRM_POLLS consecutive ticks.
            seen_now.add(model)
            n = self._orphan_polls.get(model, 0) + 1
            self._orphan_polls[model] = n
            if n < REAP_CONFIRM_POLLS:
                log.info("Serve session '%s': engine alive, proxy gone (%d/%d), waiting",
                         model, n, REAP_CONFIRM_POLLS)
                continue
            log.warning(
                "Serve session '%s' has a dead proxy (%s) while its engine "
                "(pid %s) still holds VRAM , stopping it so the card comes "
                "back. Re-serve with: greenboost synapse serve %s",
                model, sess.get("proxy_error"), sess.get("llama_pid"), model,
            )
            try:
                import gb_dataflux
                gb_dataflux.emit({
                    "kind": "synapse_serve", "node": "host", "label": "supervisor",
                    "status": "error", "model": model,
                    "reason": "orphaned_engine_reaped",
                    "proxy_error": sess.get("proxy_error"),
                    "llama_pid": sess.get("llama_pid"),
                })
            except Exception:
                pass
            try:
                gb_synapse.stop(model)
            except Exception as exc:
                log.warning("could not stop orphaned serve '%s': %s", model, exc)
            self._orphan_polls.pop(model, None)
        for gone in [m for m in self._orphan_polls if m not in seen_now]:
            del self._orphan_polls[gone]

    def _pet(self) -> None:
        """Watchdog ping from inside a long tick, so a slow but progressing tick
        (Ollama unloads, archive compaction) is not mistaken for a hang."""
        if getattr(self, "_wd", 0.0):
            _notify_watchdog()

    def _tick(self) -> None:
        self._pet()
        # 3-.  Orphaned serve session , engine alive, proxy dead.
        self._reap_orphaned_serves()
        self._pet()

        # 3a. Gaming mode , doubles VRAM reserve thresholds
        gaming = _read_gaming_mode()

        # 3b. VRAM pressure (NVML, no nvidia-smi)
        _vram_state, _gpu_util, _non_gb_mb = self._vram_mon.poll(gaming_mode=gaming)

        # 3b'. System RAM/swap pressure , independent of GreenBoost's own T2
        # pool (see _RamMonitor docstring: CPU-offload streaming bypasses it).
        self._ram_mon.poll()

        # 3b''. DAMON host cold-page evidence , republished for gb_tiering,
        # which cannot read /sys/module/damon_stat (mode 0400).
        self._damon_mon.poll()

        # 3c. ECC double-bit error monitoring (hardware-critical)
        ecc_dbe = _read_ecc_dbe_volatile()
        if ecc_dbe > self._ecc_dbe_seen:
            new_errors = ecc_dbe - self._ecc_dbe_seen
            log.critical(
                "ECC DOUBLE-BIT ERROR: %d new uncorrectable error(s) detected "
                "(total volatile: %d). Hardware memory may be corrupted. "
                "Consider rebooting and running memtest/nvidia-smi --ecc-config=1.",
                new_errors, ecc_dbe,
            )
            try:
                _atomic_write(ECC_DBE_FLAG, str(ecc_dbe))
            except OSError as exc:
                log.warning("ecc_dbe_flag not written: %s", exc)
            _sd_notify(
                f"STATUS=ECC DBE error: {ecc_dbe} uncorrectable error(s) , hardware risk\n"
            )
            self._ecc_dbe_seen = ecc_dbe

        # 3c'. Build this tick's enriched metrics once; feed the reactive
        # orchestrator AND the always-on dataflux recorder from it.
        if self._orch is not None or self._snap_recorder is not None:
            try:
                GpuMetrics = _import_gpu_metrics()
                m = GpuMetrics(
                    ecc_dbe_volatile = ecc_dbe,
                    temp_c           = self._nvml.temp_c(),
                    gpu_util_pct     = _gpu_util,
                    gb               = None,
                )
                # Enrich with T2/T3 pool + shim_phase, host PSI pressure, and
                # power/clock/PCIe/topology fields , feeds Loops J/O-S.
                if self._gb_provider is not None:
                    self._gb_provider.fill(m)
                if self._sys_provider is not None:
                    self._sys_provider.fill(m)
                if self._nvml_provider is not None:
                    self._nvml_provider.fill(m)
                if self._orch is not None:
                    self._orch.on_metrics(m)
                    self._orch.feed_vram_state(_non_gb_mb, self._vram_mon._total_mb)
                if self._snap_recorder is not None:
                    try:
                        self._snap_recorder._on_metrics(m)
                    except Exception:
                        pass
            except Exception as exc:
                log.debug("Orchestrator feed error: %s", exc)

        # 3c''. Read-only telemetry snapshot for other programs (no root needed
        # to read it; failure here must never disturb the supervisor).
        try:
            import gb_snapshot
            _u, _f, _t, _ = self._nvml._h.mem() if self._nvml._h else (0, 0, 0, 0.0)
            gb_snapshot.write_snapshot(gb_snapshot.build_snapshot(_t, _f))
        except Exception as exc:
            log.debug("snapshot write failed: %s", exc)

        # 3d. Phase file idle detection → reclaim
        self._pet()
        phase = _read_phase()

        if phase == "DEEP_IDLE" and _gpu_util < 1.0:
            self._idle_count += 1
            if self._idle_count >= CONFIRM_POLLS:
                log.info("DEEP_IDLE confirmed (%d/%d polls) , triggering reclaim",
                         self._idle_count, CONFIRM_POLLS)
                if _ollama_unload_models(OLLAMA_URL):
                    log.info("Ollama models unloaded gracefully")
                elif AGGRESSIVE:
                    log.warning("Ollama unload failed , aggressive reclaim engaged")
                    _aggressive_reclaim()
                else:
                    log.info("No models to unload (or Ollama not running)")
                self._idle_count = 0  # reset after action
        else:
            # Non-idle or GPU still active , reset counter
            if self._idle_count > 0:
                self._idle_count = 0

        # 3e. Periodic dataflux archive compaction (see __init__'s comment).
        self._pet()
        now = time.time()
        if now - self._last_dataflux_compaction_ts >= _DATAFLUX_COMPACT_INTERVAL_S:
            self._last_dataflux_compaction_ts = now
            try:
                import gb_dataflux
                dropped = gb_dataflux.compact_archive()
                if dropped:
                    log.info("dataflux archive compacted: %d event(s) past retention dropped",
                             dropped)
            except Exception as exc:
                log.debug("dataflux compaction error: %s", exc)

    def _on_stop(self, signum: int, frame) -> None:
        log.info("Received signal %d , shutting down", signum)
        self._running = False
        _STOP.set()

    def _shutdown(self) -> None:
        _notify_stopping()
        # Lifecycle sentinel: clean shutdown → rename running → sentinel
        # Hard-reset (SIGKILL / power loss) leaves RUNNING_FILE for next boot
        try:
            if RUNNING_FILE.exists():
                RUNNING_FILE.rename(SENTINEL_FILE)
                log.info("Sentinel file written (clean shutdown)")
        except Exception as exc:
            log.warning("Sentinel rename failed: %s", exc)
        if self._orch is not None:
            self._orch.stop()
        if self._nvml_provider is not None:
            self._nvml_provider.close()
        self._nvml.close()
        log.info("GreenBoost supervisor stopped")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("[gb_supervisor] Must run as root (systemd unit: User=root)", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) > 1 and sys.argv[1] == "--recover":
        # One-shot recovery mode: used by `greenboost recover` CLI command.
        # Runs the same boot-time recovery sequence and exits (no daemon loop).
        fault = _run_recovery()
        sys.exit(0 if fault in ("unknown", "oom_kill", "thermal", "apparmor") else 1)

    GreenBoostSupervisor().run()

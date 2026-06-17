#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
gb_supervisor.py — GreenBoost unified system supervisor.

Replaces the four separate systemd daemons that shipped installed-but-inert:
  greenboost-recovery         oneshot boot-time crash recovery
  greenboost-sentinel         dirty-shutdown detection via file rename
  greenboost-vram-watchdog    physical VRAM pressure monitor (was polling nvidia-smi @3s)
  greenboost-idle-reclaim     graceful Ollama model unload on DEEP_IDLE

Key design improvements over the four-daemon design:
  - No nvidia-smi forks.  VRAM pressure is read from TelemetryManager.snapshot()
    backed by pynvml embedded calls — same path as gb_telemetry.  Zero subprocess
    overhead on the GPU query path.
  - Single sd_notify READY=1 sent AFTER boot recovery completes, so the
    Before=ollama.service boot ordering is preserved without an extra oneshot.
  - Process kill is opt-in (GB_SUPERVISOR_AGGRESSIVE_RECLAIM=1).  Default is
    Ollama-graceful-only (REST DELETE /api/delete with keep_alive:0).
  - The sentinel lifecycle is managed here; no separate unit needed.

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
import time
from pathlib import Path

# Lazy imports at runtime to avoid heavy init at module load time
def _import_orchestrator():
    from gb_orchestrator import ReactiveOrchestrator
    return ReactiveOrchestrator

def _import_gpu_metrics():
    from gb_telemetry import GpuMetrics
    return GpuMetrics

# ── Constants ────────────────────────────────────────────────────────────────

STATE_DIR       = Path("/var/lib/greenboost")
RUN_DIR         = Path("/run/greenboost")
SENTINEL_FILE   = STATE_DIR / "sentinel"
RUNNING_FILE    = STATE_DIR / "running"
PRESSURE_FILE   = STATE_DIR / "vram_pressure"
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
CONFIRM_POLLS = int(os.environ.get("GB_SUPERVISOR_IDLE_CONFIRM_POLLS", "3"))

logging.basicConfig(
    level=logging.INFO,
    format="[gb_supervisor] %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("gb_supervisor")


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
            log.warning("gb_nvml unavailable (%s) — using GbInfo ioctl only", exc)
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


# Module-level lazy sampler — avoids nvmlInit/Shutdown per-call handle churn.
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
        text = PHASE_FILE.read_text()
        for line in text.splitlines():
            if line.startswith("phase="):
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return ""


# ── Boot recovery ─────────────────────────────────────────────────────────────

def _run_recovery() -> str:
    """
    Run the boot-time crash recovery sequence.  Returns the fault class string.
    Mirrors greenboost-recover logic, now as Python so the supervisor owns it.
    """
    fault = "unknown"

    # 0. Dirty-shutdown detection
    dirty = SENTINEL_FILE.exists() or RUNNING_FILE.exists()
    if dirty:
        log.warning("Dirty shutdown detected — running full recovery sequence")
    else:
        log.info("No dirty-shutdown sentinel — last boot was clean")

    # 1. Kernel module check
    # Sup-M2: add timeouts so a hung subprocess can't block READY=1 indefinitely.
    try:
        lsmod = subprocess.run(["lsmod"], capture_output=True, text=True, timeout=5)
        if "greenboost" in lsmod.stdout:
            log.info("greenboost.ko loaded")
        else:
            log.warning("greenboost.ko not loaded — attempting modprobe")
            try:
                r = subprocess.run(["modprobe", "greenboost"],
                                    capture_output=True, text=True, timeout=15)
                if r.returncode == 0:
                    log.info("greenboost.ko loaded via modprobe")
                else:
                    log.error("modprobe greenboost failed: %s", r.stderr.strip())
                    # Non-fatal: Path B still works without the module
            except subprocess.TimeoutExpired:
                log.warning("modprobe greenboost timed out — continuing without module")
    except subprocess.TimeoutExpired:
        log.warning("lsmod timed out — skipping module check")
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
        log.info("%s not present — skipping OOM reset (Path B mode)", _GB_DEV)

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

    # Sup-H1: use re.search for regex-shaped patterns — plain `in` treated them
    # as literal substrings so ECC/thermal recovery was effectively dead.
    if "Killed process" in jq_kernel and "oom" in jq_kernel.lower():
        fault = "oom_kill"
    elif "Xid" in jq_all and "GPU" in jq_all:
        fault = "xid_fault"
    elif any(re.search(k, jq_all, re.I) for k in (
            r"ecc.*double.bit", r"uncorrectable.*ecc",
            r"nvml.*dbe", r"ecc double", r"uncorrectable ecc")):
        fault = "ecc_dbe"
    elif any(re.search(k, jq_all, re.I) for k in (
            r"thermal throttl", r"nvidia.*thermal",
            r"gpu.*temp.*critical")):
        fault = "thermal"
    elif "DENIED" in jq_all and re.search(r"greenboost", jq_all, re.I) and "apparmor" in jq_all.lower():
        fault = "apparmor"

    RECOVERY_CLASS.write_text(fault)
    log.info("Fault class: %s", fault)

    # 2d. Per-fault remediation
    if fault == "ecc_dbe":
        log.warning("ECC double-bit error — setting ecc_dbe_flag to reduce T1 quota")
        ECC_DBE_FLAG.write_text("1")
    elif fault == "thermal":
        log.warning("Thermal fault — inserting 10 s cooling pause before Ollama restart")
        time.sleep(10)
    elif fault == "apparmor":
        log.warning("AppArmor denial — patching snap-confine profiles")
        _patch_apparmor()
    elif fault == "xid_fault":
        log.warning("Xid GPU fault — forcing NVIDIA driver module reset")
        try:
            subprocess.run(["rmmod", "nvidia_drm", "nvidia_modeset", "nvidia"],
                           capture_output=True, timeout=20)
        except subprocess.TimeoutExpired:
            log.warning("rmmod nvidia timed out — continuing")
        try:
            subprocess.run(["modprobe", "nvidia"], capture_output=True, timeout=20)
        except subprocess.TimeoutExpired:
            log.warning("modprobe nvidia timed out — continuing")

    # 3. Sentinel cleanup
    SENTINEL_FILE.unlink(missing_ok=True)
    RUNNING_FILE.unlink(missing_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "last_clean_boot").write_text(
        time.strftime("%Y-%m-%dT%H:%M:%S")
    )
    log.info("Recovery complete (fault class: %s)", fault)
    return fault


def _patch_apparmor() -> None:
    """AppArmor remediation: patch snap-confine profiles for libgreenboost_audit."""
    local_sc = Path("/etc/apparmor.d/local/usr.lib.snapd.snap-confine.real")
    libs = [
        "/usr/local/lib/libgreenboost_audit.so mr,",
        "/usr/local/lib/x86_64-linux-gnu/libgreenboost_audit.so mr,",
    ]
    if local_sc.exists():
        text = local_sc.read_text()
        if "libgreenboost_audit" not in text:
            with local_sc.open("a") as f:
                f.write("\n" + "\n".join(libs) + "\n")
    try:
        subprocess.run(
            ["apparmor_parser", "-r",
             "/etc/apparmor.d/usr.lib.snapd.snap-confine.real"],
            capture_output=True, timeout=15,
        )
    except (Exception, subprocess.TimeoutExpired):
        pass
    try:
        result = subprocess.run(
            ["find", "/var/lib/snapd/apparmor/profiles/",
             "-name", "snap-confine.*", "-maxdepth", "1"],
            capture_output=True, text=True, timeout=10,
        )
        for sp in result.stdout.splitlines():
            sp_path = Path(sp.strip())
            if sp_path.exists() and "libgreenboost_audit" not in sp_path.read_text():
                with sp_path.open("a") as f:
                    f.write("\n" + "\n".join(libs) + "\n")
                try:
                    subprocess.run(["apparmor_parser", "-r", str(sp_path)],
                                   capture_output=True, timeout=15)
                except subprocess.TimeoutExpired:
                    pass
    except (Exception, subprocess.TimeoutExpired):
        pass
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
            payload = json.dumps({"name": name, "keep_alive": 0}).encode()
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
                    "CRITICAL: free VRAM %d MiB < threshold %d MiB%s — "
                    "non-GB consumers: %d MiB (browser NVDEC / compositor?). "
                    "Consider closing GPU-heavy apps or increasing GREENBOOST_WORKSTATION_RESERVE_MB.",
                    free_mb, self._crit_free_mb,
                    " [gaming_mode doubled]" if gaming_mode else "", non_gb_mb,
                )
            elif state == "warn":
                log.warning(
                    "WARN: non-GB GPU usage %d MiB > %d MiB threshold%s — "
                    "free VRAM %d MiB.",
                    non_gb_mb, self._warn_mb,
                    " [gaming_mode doubled]" if gaming_mode else "", free_mb,
                )
            else:
                log.info("OK: physical VRAM pressure cleared — used %d MiB, free %d MiB",
                         used_mb, free_mb)
            self._prev_state = state

        # Always write the pressure file (greenboost vitals/status reads it)
        try:
            PRESSURE_FILE.write_text(
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


# ── Main supervisor loop ──────────────────────────────────────────────────────

class GreenBoostSupervisor:
    def __init__(self) -> None:
        self._running       = True
        self._idle_count    = 0   # consecutive DEEP_IDLE polls confirmed
        self._nvml          = _NVMLSampler(device=0)
        self._vram_mon      = _VramMonitor(self._nvml, WARN_PCT, CRIT_FREE_PCT)
        self._ecc_dbe_seen  = 0   # last known volatile DBE count (detect new errors)
        # Reactive orchestrator — closes dead feedback loops (ECC, workstation, thermal)
        try:
            ReactiveOrchestrator = _import_orchestrator()
            self._orch = ReactiveOrchestrator(mode="supervisor")
        except Exception as exc:
            log.warning("ReactiveOrchestrator unavailable (%s) — signals disabled", exc)
            self._orch = None

    def run(self) -> None:
        # Phase 0: ensure state directories exist
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        RUN_DIR.mkdir(parents=True, exist_ok=True)

        # Phase 1: boot-time recovery (runs before READY=1, before Ollama)
        _notify_status("running boot recovery")
        _run_recovery()

        # Phase 2: lifecycle sentinel — touch running file
        RUNNING_FILE.write_text(str(os.getpid()))

        # Signal systemd we are ready (Ollama may start now — Before= ordering)
        _notify_ready()
        _notify_status("monitoring VRAM + idle reclaim")
        log.info("GreenBoost supervisor ready (poll=%ds, warn_pct=%d, crit_free_pct=%d, aggressive=%s)",
                 POLL_SECS, WARN_PCT, CRIT_FREE_PCT, AGGRESSIVE)

        # Install shutdown handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._on_stop)

        # Phase 3: steady-state poll loop
        while self._running:
            try:
                self._tick()
            except Exception as exc:
                log.exception("Tick error: %s", exc)
            # Sleep in small increments so SIGTERM is handled promptly
            for _ in range(POLL_SECS * 10):
                if not self._running:
                    break
                time.sleep(0.1)

        self._shutdown()

    def _tick(self) -> None:
        # 3a. Gaming mode — doubles VRAM reserve thresholds
        gaming = _read_gaming_mode()

        # 3b. VRAM pressure (NVML, no nvidia-smi)
        _vram_state, _gpu_util, _non_gb_mb = self._vram_mon.poll(gaming_mode=gaming)

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
            ECC_DBE_FLAG.write_text(str(ecc_dbe))
            _sd_notify(
                f"STATUS=ECC DBE error: {ecc_dbe} uncorrectable error(s) — hardware risk\n"
            )
            self._ecc_dbe_seen = ecc_dbe

        # 3c'. Feed reactive orchestrator with this tick's signals
        if self._orch is not None:
            try:
                GpuMetrics = _import_gpu_metrics()
                m = GpuMetrics(
                    ecc_dbe_volatile = ecc_dbe,
                    temp_c           = self._nvml.temp_c(),
                    gpu_util_pct     = _gpu_util,
                    gb               = None,   # pool info not fetched in supervisor tick
                )
                self._orch.on_metrics(m)
                self._orch.feed_vram_state(_non_gb_mb, self._vram_mon._total_mb)
            except Exception as exc:
                log.debug("Orchestrator feed error: %s", exc)

        # 3d. Phase file idle detection → reclaim
        phase = _read_phase()
        if phase == "DEEP_IDLE" and _gpu_util < 1.0:
            self._idle_count += 1
            if self._idle_count >= CONFIRM_POLLS:
                log.info("DEEP_IDLE confirmed (%d/%d polls) — triggering reclaim",
                         self._idle_count, CONFIRM_POLLS)
                if _ollama_unload_models(OLLAMA_URL):
                    log.info("Ollama models unloaded gracefully")
                elif AGGRESSIVE:
                    log.warning("Ollama unload failed — aggressive reclaim engaged")
                    _aggressive_reclaim()
                else:
                    log.info("No models to unload (or Ollama not running)")
                self._idle_count = 0  # reset after action
        else:
            # Non-idle or GPU still active — reset counter
            if self._idle_count > 0:
                self._idle_count = 0

    def _on_stop(self, signum: int, frame) -> None:
        log.info("Received signal %d — shutting down", signum)
        self._running = False

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

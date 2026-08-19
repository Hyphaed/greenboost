#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""gb_debug_bundle.py, GreenBoost diagnostic/support bundle collector.

Gathers critical system state into a tarball for remote diagnosis. Atomic tar
creation + secret redaction discipline, ported from NemoClaw's diagnostic shape
(Apache-2.0, design reference only).

CLI:
    python3 gb_debug_bundle.py collect [--output PATH] [--timeout SECS]
    greenboost debug bundle [--output PATH]    (greenboost_setup.sh delegates here)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_DIR))

# Conditional imports (fail gracefully if not available)
def _try_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None

gb_readiness = _try_import("gb_readiness")
gb_dataflux = _try_import("gb_dataflux")
capture = _try_import("greenboost_cli.instruments.capture") if _try_import("greenboost_cli") else None

_HAS_CAPTURE = capture is not None

# capture.py's bounded_capture() only redacts credential-shaped secrets (its
# own docstring: "credential-redacted structure"), it has no notion of
# private LAN IPs or /home/<user>/ paths, both of which CLAUDE.md's Secrets
# & Sensitive Information section separately forbids in anything shared. So
# the IP/home-path scrub below must ALWAYS run, in addition to (not instead
# of) capture.py's secret redaction, running only one or the other leaves a
# real gap in a bundle whose whole purpose is safe-to-share diagnostics.
def _redact_ips_and_paths(text: str) -> str:
    redacted = re.sub(r'192\.168\.\d+\.\d+', '<redacted-ip>', text)
    redacted = re.sub(r'10\.\d+\.\d+\.\d+', '<redacted-ip>', redacted)
    redacted = re.sub(r'172\.(1[6-9]|2[0-9]|3[0-1])\.\d+\.\d+', '<redacted-ip>', redacted)
    redacted = re.sub(r'/home/[^/\s]+/', '/home/<redacted>/', redacted)
    return redacted


def _redact_text(text: str) -> str:
    """Redact secret-shaped patterns AND private-IP/home-path patterns from
    text. Uses capture.py for the former when available, a fallback regex
    set otherwise; the IP/home-path scrub always runs regardless."""
    if _HAS_CAPTURE:
        try:
            text = capture.bounded_capture(text)
        except Exception:
            pass
    else:
        # Fallback secret redaction when greenboost-cli isn't importable.
        text = re.sub(r'hf_[A-Za-z0-9]{20,}', '<redacted-hf-token>', text)
        text = re.sub(r'sk-[A-Za-z0-9_-]{20,}', '<redacted-token>', text)
        text = re.sub(r'ghp_[A-Za-z0-9_-]{10,}', '<redacted-gh-token>', text)
    return _redact_ips_and_paths(text)


def _run_cmd(cmd: str, timeout_s: float = 10.0, shell: bool = True) -> str:
    """Run a shell command with timeout, return stdout or error message."""
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, timeout=timeout_s
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return f"<command timed out after {timeout_s}s>"
    except Exception as e:
        return f"<command failed: {type(e).__name__}: {e}>"


def _read_file_bounded(path: Path, max_bytes: int = 1024 * 100) -> str:
    """Read file with size bound, return tail if too large."""
    try:
        if not path.exists():
            return f"<file not found: {path}>"
        stat = path.stat()
        if stat.st_size > max_bytes:
            # Return the last max_bytes
            with open(path, "r", errors="replace") as f:
                f.seek(max(0, stat.st_size - max_bytes))
                content = f.read()
            return f"<truncated from {stat.st_size} bytes, showing last {max_bytes}>:\n{content}"
        else:
            with open(path, "r", errors="replace") as f:
                return f.read()
    except Exception as e:
        return f"<read failed: {type(e).__name__}: {e}>"


def _read_feeder_state(feeder_spec: str, timeout_s: float = 10.0) -> dict:
    """Query one feeder for its state over SSH. Feeder spec is 'user@ip' or 'ip'."""
    user_ip = feeder_spec.strip()
    if "@" not in user_ip:
        user_ip = f"{os.environ.get('USER', 'root')}@{user_ip}"

    result = {"feeder": user_ip, "reachable": False}

    # Try SSH connection
    try:
        test_result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
             "-o", "StrictHostKeyChecking=no", user_ip, "true"],
            capture_output=True, timeout=timeout_s
        )
        if test_result.returncode != 0:
            result["error"] = "SSH connection failed"
            return result
    except Exception as e:
        result["error"] = str(e)
        return result

    result["reachable"] = True

    # Collect remote state via SSH
    commands = {
        "lsmod_greenboost": "lsmod | grep greenboost",
        "nvidia_smi": "nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,clocks.max.sm,power.draw,power.limit,utilization.gpu,memory.used,memory.total --format=csv,nounits",
        "free": "free -h",
        "vmstat": "vmstat 1 3",
    }

    for key, cmd in commands.items():
        try:
            ssh_result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                 "-o", "StrictHostKeyChecking=no", user_ip, cmd],
                capture_output=True, text=True, timeout=timeout_s / len(commands)
            )
            result[key] = _redact_text(ssh_result.stdout + ssh_result.stderr)
        except Exception as e:
            result[key] = f"<error: {type(e).__name__}>"

    return result


def _emit_support_bundle(output_path: str, sections: dict, feeder_count: int,
                          status: str, error: str = "") -> None:
    """Best-effort dataflux emit (Observability Must-Rule), fired with the
    REAL final status (ok/error) after the tar step actually completes or
    fails, not before it's attempted."""
    if not gb_dataflux:
        return
    try:
        event = {
            "kind": "support_bundle", "path": output_path,
            "sections": list(sections.keys()), "redacted": True,
            "feeders": feeder_count, "status": status,
        }
        if error:
            event["error"] = error
        gb_dataflux.emit(event)
    except Exception:
        pass


def collect_support_bundle(
    output_path: str,
    timeout_s: float = 60.0,
    include_feeders: bool = True,
) -> bool:
    """Collect diagnostic bundle into a tarball. Returns True on success, False
    on failure (output_path left untouched if pre-existing). Every collected
    value is redacted before being written.

    Ported from NemoClaw's diagnostics/{debug,tarball}.ts shape (Apache-2.0,
    design reference only)."""

    start_time = time.time()
    sections = {}
    feeder_count = 0

    temp_dir = None
    try:
        # Create temporary collection directory
        temp_dir = tempfile.mkdtemp(prefix="gb_debug_bundle_")

        def _remaining_time() -> float:
            elapsed = time.time() - start_time
            return max(1.0, timeout_s - elapsed)

        # 1. Readiness report (JSON)
        if gb_readiness:
            try:
                report = gb_readiness.build_report()
                sections["readiness"] = True
                with open(Path(temp_dir) / "readiness.json", "w") as f:
                    json.dump(report, f, indent=2)
            except Exception:
                pass

        # 2. Kernel module state
        try:
            lsmod_output = _redact_text(_run_cmd("lsmod | grep greenboost", timeout_s=_remaining_time()))
            with open(Path(temp_dir) / "kmod.txt", "w") as f:
                f.write(lsmod_output)
            sections["kmod"] = True
        except Exception:
            pass

        # 3. Device state
        try:
            dev_output = _redact_text(_run_cmd("ls -la /dev/greenboost 2>&1", timeout_s=_remaining_time()))
            with open(Path(temp_dir) / "dev.txt", "w") as f:
                f.write(dev_output)
            sections["dev"] = True
        except Exception:
            pass

        # 4. nvidia-smi (using canonical query from CLAUDE.md)
        try:
            nvidia_cmd = (
                "nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,clocks.max.sm,"
                "power.draw,power.limit,utilization.gpu,memory.used,memory.total "
                "--format=csv,nounits"
            )
            nvidia_output = _redact_text(_run_cmd(nvidia_cmd, timeout_s=_remaining_time()))
            with open(Path(temp_dir) / "nvidia_smi.txt", "w") as f:
                f.write(nvidia_output)
            sections["nvidia_smi"] = True
        except Exception:
            pass

        # 5. System memory state
        try:
            free_output = _run_cmd("free -h", timeout_s=_remaining_time())
            vmstat_output = _run_cmd("vmstat 1 3", timeout_s=_remaining_time())
            memory_text = _redact_text(f"=== free -h ===\n{free_output}\n\n=== vmstat ===\n{vmstat_output}")
            with open(Path(temp_dir) / "memory.txt", "w") as f:
                f.write(memory_text)
            sections["memory"] = True
        except Exception:
            pass

        # 6. Shim stats and metrics
        for name, path_str in [("shim_stats", "/run/greenboost/shim_stats"),
                               ("metrics", "/run/greenboost/metrics.json")]:
            try:
                path = Path(path_str)
                if path.exists():
                    content = _redact_text(_read_file_bounded(path, max_bytes=1024 * 100))
                    with open(Path(temp_dir) / f"{name}.txt", "w") as f:
                        f.write(content)
                    sections[name] = True
            except Exception:
                pass

        # 7. Dataflux log tail (bounded). read_events() has no limit/reverse
        # kwargs of its own (it returns the whole window, oldest-first) —
        # bound the tail here instead: last 24h, most-recent 100 events.
        try:
            if gb_dataflux:
                events = gb_dataflux.read_events(since_hours=24.0)
                dataflux_tail = list(reversed(events))[:100]
                dataflux_text = _redact_text(
                    "\n".join(json.dumps(e) for e in dataflux_tail)
                )
                with open(Path(temp_dir) / "dataflux_tail.jsonl", "w") as f:
                    f.write(dataflux_text)
                sections["dataflux"] = True
        except Exception:
            pass

        # 8. Synapse run-state files
        try:
            run_state_dir = Path("/run/greenboost/synapse")
            if run_state_dir.exists():
                for state_file in run_state_dir.glob("*.json"):
                    try:
                        content = _redact_text(_read_file_bounded(state_file, max_bytes=1024 * 50))
                        target = Path(temp_dir) / f"synapse_{state_file.name}"
                        with open(target, "w") as f:
                            f.write(content)
                    except Exception:
                        pass
                sections["synapse"] = True
        except Exception:
            pass

        # 9. Serving recipes check
        try:
            serving_check_script = _REPO_DIR / "serving" / "check_recipes.py"
            if serving_check_script.exists():
                output = _redact_text(_run_cmd(
                    f"cd {_REPO_DIR} && python3 serving/check_recipes.py --check",
                    timeout_s=_remaining_time()
                ))
                with open(Path(temp_dir) / "serving_check.txt", "w") as f:
                    f.write(output)
                sections["serving"] = True
        except Exception:
            pass

        # 10. Feeder state (if cluster configured)
        if include_feeders:
            try:
                cluster_conf = Path("/etc/greenboost/cluster.conf")
                if cluster_conf.exists():
                    feeders_data = {}
                    with open(cluster_conf, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            # Parse: address user ...
                            parts = line.split()
                            if len(parts) >= 2:
                                addr = parts[0]
                                user = parts[1] if len(parts) > 1 else None
                                feeder_spec = f"{user}@{addr}" if user else addr

                                if time.time() - start_time < timeout_s - 5.0:  # Reserve 5s for tar
                                    feeder_state = _read_feeder_state(
                                        feeder_spec,
                                        timeout_s=min(10.0, _remaining_time() / 2.0)
                                    )
                                    # addr itself is a LAN IP/hostname, redact the key too.
                                    feeders_data[_redact_ips_and_paths(addr)] = feeder_state
                                    feeder_count += 1

                    if feeders_data:
                        with open(Path(temp_dir) / "feeders.json", "w") as f:
                            json.dump(feeders_data, f, indent=2)
                        sections["feeders"] = True
            except Exception:
                pass

        # Create manifest
        manifest = {
            "collected_at": time.time(),
            "sections": sections,
            "feeder_count": feeder_count,
            "redacted": True,
        }
        with open(Path(temp_dir) / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Create tarball atomically: write to a PID-scoped partial name, then
        # os.replace() into place so a failing tar never disturbs a
        # pre-existing bundle at output_path. shutil.make_archive() always
        # appends its own format-specific suffix (".tar.gz" here) to
        # whatever base_name it's given and RETURNS the actual path it
        # wrote, that return value, not a string built by hand, is what
        # os.replace() must use; assuming a fixed naming convention here
        # silently orphaned every real bundle (confirmed live: the archive
        # was created correctly but never renamed into place, so this
        # function always returned False despite doing all the work).
        partial_base = f"{output_path}.partial.{os.getpid()}"
        actual_archive_path = None
        try:
            actual_archive_path = shutil.make_archive(partial_base, "gztar", temp_dir)
            os.replace(actual_archive_path, output_path)
            os.chmod(output_path, 0o600)  # may contain sensitive host state, owner-only
            _emit_support_bundle(output_path, sections, feeder_count, status="ok")
            return True
        except Exception as e:
            # Clean up whatever partial artifact exists on failure.
            for candidate in (actual_archive_path, f"{partial_base}.tar.gz"):
                if candidate:
                    try:
                        os.remove(candidate)
                    except Exception:
                        pass
            _emit_support_bundle(output_path, sections, feeder_count, status="error", error=str(e))
            return False

    finally:
        # Clean up temp directory
        if temp_dir:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass


def main(argv: "list[str] | None" = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if not argv or argv[0] not in ("collect", "help"):
        print("usage: gb_debug_bundle.py collect [--output PATH] [--timeout SECS]", file=sys.stderr)
        return 1

    if argv[0] == "help":
        print("GreenBoost diagnostic bundle collector")
        print("usage: gb_debug_bundle.py collect [--output PATH] [--timeout SECS]")
        print("  --output PATH: save bundle to this path (default: ./greenboost-debug-<timestamp>.tar.gz)")
        print("  --timeout SECS: timeout for collection (default: 60)")
        return 0

    # Parse options
    output_path = None
    timeout_s = 60.0

    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--output" and i + 1 < len(argv):
            output_path = argv[i + 1]
            i += 2
        elif arg == "--timeout" and i + 1 < len(argv):
            try:
                timeout_s = float(argv[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            i += 1

    # Default output path
    if not output_path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = f"./greenboost-debug-{timestamp}.tar.gz"

    # Expand ~ to home
    output_path = os.path.expanduser(output_path)

    print(f"Collecting GreenBoost diagnostic bundle (timeout {timeout_s}s)...")
    print(f"Output: {output_path}")
    print()

    success = collect_support_bundle(output_path, timeout_s=timeout_s)

    if success:
        print(f"\n✓ Bundle created: {output_path}")
        print("  Known secrets are auto-redacted, but please review before sharing.")
        return 0
    else:
        print(f"\n✗ Bundle creation failed", file=sys.stderr)
        print(f"  Output file left untouched if pre-existing.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

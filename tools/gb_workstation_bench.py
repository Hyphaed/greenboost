#!/usr/bin/env python3
"""
GreenBoost Workstation Benchmark - gb_workstation_bench.py
Measures T1/T2/T3 memory bandwidth on live hardware.

Usage:
    python3 gb_workstation_bench.py [--json] [--output results.json]
    python3 gb_workstation_bench.py --skip-t1 --skip-t2   # NVMe only

Hardware-agnostic: auto-detects GPU, RAM, NVMe, and GreenBoost pool.
CuPy gives the most accurate GPU/PCIe measurements; without it the script
falls back to PCIe estimates and nvidia-smi. Install CuPy with:
    pip install cupy-cuda12x   # CUDA 12.x (most common)
    pip install cupy-cuda11x   # CUDA 11.x (older)
"""

import os
import sys
import json
import struct
import time
import tempfile
import argparse
import subprocess
import fcntl
import threading
import itertools
from pathlib import Path

# ── Rich UI (optional - degrades to plain text) ───────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich import box as rbox
    _RICH = True
except ImportError:
    _RICH = False

# Brand palette
VIOLET = "#6C71C4"
LIME   = "#E6FF3C"
GRAY   = "#D0CFCC"
CYAN   = "#30C8FF"
AMBER  = "#FFBF00"
RED    = "#FF5C32"

if _RICH:
    console = Console(highlight=False)
else:
    import re as _re
    class _PlainConsole:
        def print(self, msg="", **kw):
            print(_re.sub(r'\[/?[^\]]*\]', '', str(msg)))
        def rule(self, title="", **kw):
            w = 70
            if title:
                pad = max(0, (w - len(title) - 2) // 2)
                print("─" * pad + " " + title + " " + "─" * pad)
            else:
                print("─" * w)
    console = _PlainConsole()


def _ok(msg: str)   -> None: console.print(f"  [bold {LIME}]✓[/]  [{GRAY}]{msg}[/]")
def _fail(msg: str) -> None: console.print(f"  [bold {RED}]✗[/]  {msg}")
def _warn(msg: str) -> None: console.print(f"  [bold {AMBER}]⚠[/]  {msg}")
def _info(msg: str) -> None: console.print(f"  [bold {VIOLET}]◈[/]  [{GRAY}]{msg}[/]")


def _run_with_spinner(label: str, fn):
    """Run fn() while showing a braille spinner. Returns fn's result."""
    FRAMES = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
    result_box = [None]
    exc_box    = [None]

    def _worker():
        try:
            result_box[0] = fn()
        except Exception as e:
            exc_box[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    if _RICH:
        with Live(console=console, refresh_per_second=12) as live:
            while t.is_alive():
                live.update(Text(f"  {next(FRAMES)}  {label}", style=f"dim {GRAY}"))
                time.sleep(0.08)
        console.print(f"  [bold {LIME}]✓[/]  [{GRAY}]{label}[/]")
    else:
        t.join()

    t.join()
    if exc_box[0]:
        raise exc_box[0]
    return result_box[0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str),
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


# ── System probes ─────────────────────────────────────────────────────────────

def query_gpu():
    out = _run("nvidia-smi --query-gpu=name,memory.total,compute_cap,"
               "pcie.link.gen.max,pcie.link.width.max,clocks.max.sm,driver_version"
               " --format=csv,noheader,nounits")
    if not out:
        return None
    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 7:
        return None
    return {
        "name":          parts[0],
        "vram_mb":       int(parts[1]) if parts[1].isdigit() else 0,
        "compute_cap":   parts[2],
        "pcie_gen":      parts[3],
        "pcie_width":    parts[4],
        "max_clock_mhz": parts[5],
        "driver":        parts[6],
    }


def query_cpu():
    info = {"model": "", "logical_cpus": 0, "p_cores": 0, "e_cores": 0,
            "max_freq_mhz": 0, "numa_nodes": 0}

    cpuinfo = Path("/proc/cpuinfo").read_text(errors="replace") if Path("/proc/cpuinfo").exists() else ""
    for line in cpuinfo.splitlines():
        if line.startswith("model name") and not info["model"]:
            info["model"] = line.split(":", 1)[-1].strip()
        if line.startswith("processor"):
            info["logical_cpus"] += 1

    freq_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    if freq_path.exists():
        try:
            info["max_freq_mhz"] = int(freq_path.read_text().strip()) // 1000
        except ValueError:
            pass

    numa_path = Path("/sys/devices/system/node")
    if numa_path.exists():
        info["numa_nodes"] = len(list(numa_path.glob("node[0-9]*")))

    cpu_base = Path("/sys/devices/system/cpu")
    p, e = 0, 0
    for cpu_dir in sorted(cpu_base.glob("cpu[0-9]*")):
        cap_f = cpu_dir / "cpu_capacity"
        if cap_f.exists():
            try:
                cap = int(cap_f.read_text().strip())
                if cap >= 1024: p += 1
                else:           e += 1
            except ValueError:
                pass
    if p or e:
        info["p_cores"] = p
        info["e_cores"] = e
    return info


def query_ram():
    info = {"total_mb": 0, "free_mb": 0, "available_mb": 0, "speed_mt": 0}
    if not Path("/proc/meminfo").exists():
        return info
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            info["total_mb"] = int(line.split()[1]) // 1024
        elif line.startswith("MemFree:"):
            info["free_mb"] = int(line.split()[1]) // 1024
        elif line.startswith("MemAvailable:"):
            info["available_mb"] = int(line.split()[1]) // 1024
    dmi = _run("dmidecode -t memory 2>/dev/null | grep -m1 'Speed:.*MT'")
    if dmi:
        try:
            info["speed_mt"] = int(dmi.split()[1])
        except (ValueError, IndexError):
            pass
    return info


def query_swap():
    result = {"total_mb": 0, "used_mb": 0, "entries": []}
    out = _run("swapon --show=NAME,SIZE,USED,PRIO --bytes --noheadings 2>/dev/null")
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            try:
                size_mb = int(parts[1]) // (1024 * 1024)
                used_mb = int(parts[2]) // (1024 * 1024)
                prio    = int(parts[3]) if len(parts) > 3 else -1
                result["entries"].append({"name": parts[0], "size_mb": size_mb,
                                          "used_mb": used_mb, "priority": prio})
                result["total_mb"] += size_mb
                result["used_mb"]  += used_mb
            except ValueError:
                pass
    return result


def query_greenboost():
    info = {"loaded": False, "pool_info": None, "ioctl_available": False,
            "t2_available_mb": None, "oom_guard": None, "safety_reserve_mb": None,
            "free_ram_mb": None}

    pool_path = Path("/sys/class/greenboost/greenboost/pool_info")
    if pool_path.exists():
        info["loaded"] = True
        try:
            raw = pool_path.read_text().strip()
            info["pool_info"] = raw
            # Parse key values for recommendations
            for line in raw.splitlines():
                if "T2 available" in line:
                    try:
                        info["t2_available_mb"] = int(line.split(":")[1].strip().split()[0])
                    except Exception:
                        pass
                if "OOM guard" in line:
                    info["oom_guard"] = "YES" in line
                if "Safety reserve" in line:
                    try:
                        info["safety_reserve_mb"] = int(line.split(":")[1].strip().split()[0])
                    except Exception:
                        pass
                if "Free RAM" in line:
                    try:
                        info["free_ram_mb"] = int(line.split(":")[1].strip().split()[0])
                    except Exception:
                        pass
        except Exception:
            pass

    dev_path = Path("/dev/greenboost")
    if dev_path.exists():
        try:
            fd = open(dev_path, "rb+")
            # struct gb_info layout: 7×u64, 2×u32, 4×u64, 7×u32, 1×u64
            _GB_INFO_FMT = "=7Q2I4Q7IQ"
            _gb_info_size = struct.calcsize(_GB_INFO_FMT)
            GB_IOCTL_GET_INFO = (2 << 30) | (_gb_info_size << 16) | (ord('G') << 8) | 2
            buf = bytearray(_gb_info_size)
            fcntl.ioctl(fd, GB_IOCTL_GET_INFO, buf)
            info["ioctl_available"] = True
            fd.close()
        except Exception:
            pass
    return info


# ── NVMe helpers ──────────────────────────────────────────────────────────────

def _find_nvme_mount():
    for candidate in ["/swap_nvme.img", "/mnt/nvme", "/nvme", "/"]:
        p = Path(candidate)
        if p.exists():
            df = _run(f"df --output=source {candidate} 2>/dev/null | tail -1")
            if "nvme" in df:
                return candidate
    return "/"


def _nvme_bench_dir(mount):
    p = Path(mount)
    return str(p.parent) if p.is_file() else str(p)


# Known theoretical peak GPU memory bandwidth (GB/s) - used when CuPy is absent
_GPU_BW_GBS = {
    # Blackwell (RTX 50xx)
    "5090": 1792, "5080": 960, "5070 Ti": 896, "5070": 336, "5060 Ti": 448,
    # Ada Lovelace (RTX 40xx)
    "4090": 1008, "4080 Super": 736, "4080": 736,
    "4070 Ti Super": 672, "4070 Ti": 504, "4070 Super": 504, "4070": 504,
    "4060 Ti": 288, "4060": 272,
    # Ampere (RTX 30xx)
    "3090 Ti": 1008, "3090": 936, "3080 Ti": 912, "3080": 760,
    "3070 Ti": 608, "3070": 448, "3060 Ti": 448, "3060": 360,
}


def _lookup_gpu_bw(gpu_name: str):
    """Return theoretical bandwidth GB/s by matching gpu_name against _GPU_BW_GBS.
    Longest matching key wins to distinguish '5070 Ti' from '5070'."""
    best_key, best_bw = "", None
    name_upper = gpu_name.upper()
    for key, bw in _GPU_BW_GBS.items():
        if key.upper() in name_upper and len(key) > len(best_key):
            best_key, best_bw = key, bw
    return best_bw


# ── T1 benchmark (GPU VRAM) ───────────────────────────────────────────────────

def bench_t1():
    """GPU device-to-device copy bandwidth via CuPy, or theoretical lookup from GPU name."""
    result = {"bandwidth_gbs": None, "method": "unavailable", "size_mb": 0,
              "cupy_available": False}
    try:
        import cupy as cp
        result["cupy_available"] = True
        SIZE = 512 * 1024 * 1024  # 512 MB
        result["size_mb"] = SIZE // (1024 ** 2)
        a = cp.random.random(SIZE // 8, dtype=cp.float64)
        b = cp.empty_like(a)
        cp.cuda.Stream.null.synchronize()
        cp.copyto(b, a)  # warm-up
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            cp.copyto(b, a)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        result["bandwidth_gbs"] = round((5 * SIZE) / elapsed / 1e9, 1)
        result["method"] = "cupy_d2d"
        del a, b
        return result
    except ImportError:
        result["method"] = "cupy_not_installed"
    except Exception as e:
        result["method"] = f"cupy_error:{e}"

    # Fallback: theoretical bandwidth from GPU model name lookup
    gpu = query_gpu()
    if gpu:
        bw = _lookup_gpu_bw(gpu["name"])
        if bw is not None:
            result["bandwidth_gbs"] = float(bw)
            result["method"] = "theoretical_lookup"
        else:
            result["method"] = "theoretical_gddr7"
    return result


# ── T2 benchmark (System RAM / DDR) ──────────────────────────────────────────

def bench_t2():
    """System RAM (DDR) bandwidth via CuPy pinned H2D or numpy memory copy.

    Uses 1 GB buffers to exceed any realistic CPU L3 cache and stress the
    PCIe link continuously.  Also measures sustained (dual-buffer ping-pong)
    throughput, which better reflects real GreenBoost T2 transfer rates.
    """
    result = {"bandwidth_gbs": None, "sustained_bandwidth_gbs": None,
              "method": "unavailable", "size_mb": 0, "cupy_available": False}
    try:
        import cupy as cp
        import numpy as np
        result["cupy_available"] = True
        SIZE = 1024 * 1024 * 1024  # 1 GB - exceeds any L3 cache
        result["size_mb"] = SIZE // (1024 ** 2)

        # ── single-stream measurement ──────────────────────────────────────
        host = cp.cuda.alloc_pinned_memory(SIZE)
        arr  = np.frombuffer(host, dtype=np.float32)
        arr[:] = 1.0
        dev  = cp.empty(SIZE // 4, dtype=cp.float32)
        stream = cp.cuda.Stream(non_blocking=True)
        dev.set(arr, stream=stream); stream.synchronize()  # warm-up
        t0 = time.perf_counter()
        for _ in range(5):
            dev.set(arr, stream=stream)
        stream.synchronize()
        elapsed = time.perf_counter() - t0
        result["bandwidth_gbs"] = round((5 * SIZE) / elapsed / 1e9, 1)
        result["method"] = "cupy_h2d_pinned"

        # ── sustained (dual-buffer ping-pong) ──────────────────────────────
        # Two pinned buffers on two non-blocking streams keep the PCIe link
        # busy during kernel-launch overhead of the alternate stream, giving
        # a closer approximation to actual sustained H2D throughput.
        try:
            host_b = cp.cuda.alloc_pinned_memory(SIZE)
            arr_b  = np.frombuffer(host_b, dtype=np.float32)
            arr_b[:] = 1.0
            dev_b   = cp.empty(SIZE // 4, dtype=cp.float32)
            stream_b = cp.cuda.Stream(non_blocking=True)
            dev_b.set(arr_b, stream=stream_b); stream_b.synchronize()  # warm-up
            ITERS = 8
            t0s = time.perf_counter()
            for i in range(ITERS):
                if i % 2 == 0:
                    dev.set(arr,   stream=stream)
                else:
                    dev_b.set(arr_b, stream=stream_b)
            stream.synchronize()
            stream_b.synchronize()
            elapsed_s = time.perf_counter() - t0s
            result["sustained_bandwidth_gbs"] = round((ITERS * SIZE) / elapsed_s / 1e9, 1)
            del dev_b
        except Exception:
            pass  # sustained metric is optional; don't fail the whole benchmark

        del dev
        return result
    except ImportError:
        result["method"] = "cupy_not_installed"
    except Exception as e:
        result["method"] = f"cupy_error:{e}"

    # Fallback: measure DDR bandwidth with numpy memcopy (CPU↔RAM)
    try:
        import numpy as np
        SIZE = 1024 * 1024 * 1024  # 1 GB
        result["size_mb"] = SIZE // (1024 ** 2)
        a = np.ones(SIZE // 4, dtype=np.float32)
        b = np.empty_like(a)
        np.copyto(b, a)  # warm-up
        t0 = time.perf_counter()
        for _ in range(5):
            np.copyto(b, a)
        elapsed = time.perf_counter() - t0
        result["bandwidth_gbs"] = round((5 * SIZE) / elapsed / 1e9, 1)
        result["method"] = "numpy_memcopy"
        return result
    except ImportError:
        result["method"] = "numpy_not_installed"
    except Exception as e:
        result["method"] = f"numpy_error:{e}"

    return result


# ── T3 benchmark (NVMe) ───────────────────────────────────────────────────────

def bench_t3():
    """
    NVMe sequential read/write and random 4K read.
    Uses buffered I/O + fsync (O_DIRECT dropped - requires aligned buffers
    and is not universally supported; fsync gives accurate seq-write timing).
    """
    result = {
        "seq_read_gbs":  None,
        "seq_write_gbs": None,
        "rand4k_read_mbs": None,
        "method": "unavailable",
        "mount": None,
    }
    mount     = _find_nvme_mount()
    bench_dir = _nvme_bench_dir(mount)
    result["mount"] = mount

    SEQ_SIZE  = 256 * 1024 * 1024  # 256 MB sequential
    RAND_SIZE = 4096                # 4 K blocks
    RAND_OPS  = 512
    buf_write = os.urandom(SEQ_SIZE)

    fname = None
    try:
        with tempfile.NamedTemporaryFile(dir=bench_dir, delete=False, suffix=".gb_bench") as tf:
            fname = tf.name

        fd = os.open(fname, os.O_RDWR | os.O_CREAT)

        # Sequential write
        t0 = time.perf_counter()
        written = 0
        while written < SEQ_SIZE:
            chunk  = buf_write[written:written + 65536]
            written += os.write(fd, chunk)
        os.fsync(fd)
        result["seq_write_gbs"] = round(SEQ_SIZE / (time.perf_counter() - t0) / 1e9, 2)

        # Drop page cache (best-effort; requires root)
        try:
            with open("/proc/sys/vm/drop_caches", "w") as dc:
                dc.write("1")
        except PermissionError:
            pass

        # Sequential read
        os.lseek(fd, 0, os.SEEK_SET)
        t0, read_bytes = time.perf_counter(), 0
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            read_bytes += len(chunk)
        result["seq_read_gbs"] = round(read_bytes / (time.perf_counter() - t0) / 1e9, 2)

        # Random 4 K read
        import random
        file_size = os.lseek(fd, 0, os.SEEK_END)
        offsets = [random.randint(0, max(1, (file_size - RAND_SIZE) // RAND_SIZE)) * RAND_SIZE
                   for _ in range(RAND_OPS)]
        t0 = time.perf_counter()
        for off in offsets:
            os.pread(fd, RAND_SIZE, off)
        result["rand4k_read_mbs"] = round((RAND_OPS * RAND_SIZE) / (time.perf_counter() - t0) / 1e6, 1)

        os.close(fd)
        os.unlink(fname)
        result["method"] = "buffered_fsync"

    except Exception as e:
        result["method"] = f"error:{e}"
        if fname is not None:
            try:
                os.unlink(fname)
            except Exception:
                pass

    return result


# ── Report (Rich) ─────────────────────────────────────────────────────────────

def _bw_cell(bw_val, method, unit="GB/s"):
    """Format a bandwidth cell for the results table."""
    if bw_val is None:
        if "not_installed" in method:
            return f"[dim {GRAY}]- CuPy not installed[/]", method
        return f"[dim {GRAY}]- {method}[/]", method
    return f"[bold {LIME}]{bw_val} {unit}[/]", method


def _cupy_install_hint():
    """Return the correct cupy wheel name based on installed CUDA version."""
    cuda_ver = _run("nvcc --version 2>/dev/null | grep -oP 'release \\K[0-9]+'")
    if not cuda_ver:
        cuda_ver = _run("nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \\K[0-9]+'")
    try:
        major = int(cuda_ver.split(".")[0]) if "." in cuda_ver else int(cuda_ver)
        if major >= 12:
            return "cupy-cuda12x"
        elif major == 11:
            return "cupy-cuda11x"
    except Exception:
        pass
    return "cupy-cuda12x"  # default


def _recommendations(results):
    """Return list of (icon, color, message) recommendation tuples."""
    recs = []
    gb  = results["greenboost"]
    t1  = results["t1"]
    t2  = results["t2"]
    t3  = results["t3"]

    # CuPy not installed
    if not t1.get("cupy_available") or not t2.get("cupy_available"):
        pkg = _cupy_install_hint()
        recs.append(("⚠", AMBER,
            f"CuPy not installed - T1/T2 readings are estimates only.\n"
            f"    Install for accurate GPU + PCIe measurements:\n"
            f"    [bold {CYAN}]pip install {pkg}[/]  (inside the GreenBoost venv)"))

    # T2 OOM guard
    if gb.get("oom_guard") and gb.get("t2_available_mb") == 0:
        free_mb  = gb.get("free_ram_mb", 0) or 0
        safe_mb  = gb.get("safety_reserve_mb", 0) or 0
        recs.append(("⚠", AMBER,
            f"T2 pool unavailable - RAM too low for GreenBoost overflow.\n"
            f"    Free: {free_mb} MB  |  Safety reserve: {safe_mb} MB\n"
            f"    Fix: close other apps, or reduce safety_reserve_gb when loading the module."))

    # T3 error
    if t3.get("seq_read_gbs") is None and t3.get("method", "").startswith("error:"):
        recs.append(("⚠", AMBER,
            f"NVMe benchmark failed: {t3['method']}\n"
            f"    Run as root for accurate results: [bold {CYAN}]sudo python3 gb_workstation_bench.py[/]"))

    # T3 slow
    if t3.get("seq_read_gbs") is not None and t3["seq_read_gbs"] < 1.0:
        recs.append(("⚠", AMBER,
            "NVMe sequential read < 1 GB/s - check NVMe scheduler and swap file fragmentation.\n"
            f"    Run: [bold {CYAN}]sudo ./greenboost_setup.sh tune[/]"))

    # Healthy T2 DDR bandwidth (use sustained if available, single-stream otherwise)
    t2_check = t2.get("sustained_bandwidth_gbs") or t2.get("bandwidth_gbs")
    if t2_check and t2_check >= 28.0:
        recs.append(("✓", LIME, "System RAM bandwidth looks healthy."))

    return recs


def print_rich_report(results):
    """Full Rich-styled benchmark report."""
    gpu  = results["gpu"]
    ram  = results["ram"]
    cpu  = results["cpu"]
    t1   = results["t1"]
    t2   = results["t2"]
    t3   = results["t3"]
    gb_s = results["greenboost"]
    swap = results["swap"]

    console.print()

    # ── Header panel ────────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold {GRAY}]Hardware-agnostic cuda memory pool bandwidth benchmark[/]",
        title=f"[bold {VIOLET}]🟢 GreenBoost Workstation Benchmark[/]",
        border_style=VIOLET,
        padding=(0, 2),
    ))
    console.print()

    # ── System info table ────────────────────────────────────────────────────
    sys_table = Table(box=rbox.SIMPLE, show_header=False, padding=(0, 1))
    sys_table.add_column("Key",   style=f"bold {CYAN}", width=8)
    sys_table.add_column("Value", style=GRAY)
    sys_table.add_column("Detail", style=f"dim {GRAY}")

    # CPU
    p_e = ""
    if cpu["p_cores"] or cpu["e_cores"]:
        p_e = f"  ({cpu['p_cores']} P-cores / {cpu['e_cores']} E-cores)"
    freq = f"  @ {cpu['max_freq_mhz']} MHz max" if cpu["max_freq_mhz"] else ""
    sys_table.add_row("CPU", cpu["model"],
                      f"{cpu['logical_cpus']} logical CPUs{p_e}{freq}")

    # RAM
    speed_str = f"{ram['speed_mt']} MT/s" if ram["speed_mt"] else ""
    sys_table.add_row("RAM",
                      f"{ram['total_mb'] // 1024} GB total",
                      f"{ram['available_mb'] // 1024} GB available  {speed_str}")

    # GPU
    if gpu:
        sys_table.add_row("GPU", gpu["name"],
                          f"{gpu['vram_mb'] // 1024} GB VRAM  "
                          f"Compute {gpu['compute_cap']}  "
                          f"PCIe Gen{gpu['pcie_gen']} x{gpu['pcie_width']}  "
                          f"Driver {gpu['driver']}")
    else:
        sys_table.add_row("GPU", "[dim]nvidia-smi not available[/]", "")

    # Swap
    if swap["entries"]:
        swap_detail = "  ".join(
            f"{e['name']} {e['size_mb'] // 1024} GB prio={e['priority']}"
            for e in swap["entries"]
        )
        sys_table.add_row("Swap",
                          f"{swap['total_mb'] // 1024} GB total",
                          f"{swap['used_mb']} MB used  {swap_detail}")

    # GreenBoost
    gb_status = (f"[bold {LIME}]LOADED ✓[/]" if gb_s["loaded"]
                 else f"[dim {GRAY}]not loaded[/]")
    sys_table.add_row("GB module", gb_status, "")

    console.print(sys_table)

    # GreenBoost pool info (collapsible block)
    if gb_s["pool_info"]:
        console.print(Panel(
            f"[dim {GRAY}]{gb_s['pool_info']}[/]",
            title=f"[{CYAN}]Pool Info[/]",
            border_style=f"dim {VIOLET}",
            padding=(0, 1),
        ))
        console.print()

    # ── Bandwidth results table ───────────────────────────────────────────────
    bw_table = Table(
        title=f"[bold {CYAN}]Bandwidth Results[/]",
        box=rbox.ROUNDED,
        border_style=VIOLET,
        show_header=True,
        header_style=f"bold {CYAN}",
        padding=(0, 1),
    )
    bw_table.add_column("Tier",     style=f"bold {GRAY}", width=6)
    bw_table.add_column("Device",   style=GRAY,           width=16)
    bw_table.add_column("Capacity", style=GRAY,           width=12)
    bw_table.add_column("Bandwidth",                      width=28)
    bw_table.add_column("Method",   style=f"dim {GRAY}",  width=26)

    # T1
    vram_gb = f"{gpu['vram_mb'] // 1024} GB" if gpu else "?"
    t1_bw_str, t1_method = _bw_cell(t1["bandwidth_gbs"], t1["method"])
    bw_table.add_row("T1", "GPU VRAM", vram_gb, t1_bw_str,
                     f"[dim {GRAY}]{t1_method}[/]")

    # T2
    ram_gb = f"{ram['total_mb'] // 1024} GB"
    t2_bw_str, t2_method = _bw_cell(t2["bandwidth_gbs"], t2["method"])
    sus = t2.get("sustained_bandwidth_gbs")
    if sus:
        t2_bw_str += f"  [dim {GRAY}]▸ {sus} GB/s sust.[/]"
    bw_table.add_row("T2", "DDR pool", ram_gb, t2_bw_str,
                     f"[dim {GRAY}]{t2_method}[/]")

    # T3
    if t3["seq_read_gbs"] is not None:
        t3_bw = (f"[bold {LIME}]{t3['seq_read_gbs']} GB/s[/] seq-r  "
                 f"[{GRAY}]{t3['seq_write_gbs']} GB/s[/] seq-w")
        t3_detail = f"[dim {GRAY}]{t3['method']}  mount:{t3['mount']}[/]"
        bw_table.add_row("T3", "NVMe", "auto", t3_bw, t3_detail)
        if t3["rand4k_read_mbs"] is not None:
            bw_table.add_row("", "", "",
                             f"[{GRAY}]{t3['rand4k_read_mbs']} MB/s[/] rand-4K", "")
    else:
        t3_bw_str, t3_method = _bw_cell(None, t3["method"])
        bw_table.add_row("T3", "NVMe", "auto", t3_bw_str,
                         f"[dim {GRAY}]{t3_method}[/]")

    console.print(bw_table)
    console.print()

    # ── Recommendations ───────────────────────────────────────────────────────
    recs = _recommendations(results)
    if recs:
        console.print(f"  [bold {CYAN}]Recommendations[/]")
        console.print()
        for icon, color, msg in recs:
            console.print(f"  [bold {color}]{icon}[/]  {msg}")
            console.print()


def print_plain_report(results):
    """Plain-text fallback (no Rich)."""
    gpu  = results["gpu"]
    ram  = results["ram"]
    cpu  = results["cpu"]
    t1   = results["t1"]
    t2   = results["t2"]
    t3   = results["t3"]
    gb_s = results["greenboost"]
    swap = results["swap"]

    print()
    print("═" * 70)
    print("  GreenBoost Workstation Benchmark")
    print("═" * 70)

    print(f"\n  CPU   : {cpu['model']}")
    print(f"          {cpu['logical_cpus']} logical CPUs", end="")
    if cpu["p_cores"] or cpu["e_cores"]:
        print(f"  ({cpu['p_cores']} P-cores / {cpu['e_cores']} E-cores)", end="")
    if cpu["max_freq_mhz"]:
        print(f"  @ {cpu['max_freq_mhz']} MHz max", end="")
    print()

    print(f"  RAM   : {ram['total_mb'] // 1024} GB total  "
          f"({ram['available_mb'] // 1024} GB available)")
    if ram["speed_mt"]:
        print(f"          {ram['speed_mt']} MT/s")

    if gpu:
        print(f"\n  GPU   : {gpu['name']}  ({gpu['vram_mb'] // 1024} GB VRAM)")
        print(f"          Compute {gpu['compute_cap']}  "
              f"PCIe Gen{gpu['pcie_gen']} x{gpu['pcie_width']}  "
              f"Driver {gpu['driver']}")
    else:
        print("\n  GPU   : (nvidia-smi not available)")

    if swap["entries"]:
        print(f"\n  Swap  : {swap['total_mb'] // 1024} GB total  "
              f"({swap['used_mb']} MB used)")
        for e in swap["entries"]:
            print(f"          {e['name']}  {e['size_mb'] // 1024} GB  prio={e['priority']}")

    print(f"\n  GreenBoost: {'LOADED' if gb_s['loaded'] else 'not loaded'}")
    if gb_s["pool_info"]:
        for line in gb_s["pool_info"].splitlines():
            print(f"    {line}")

    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Tier   Device         Capacity     Bandwidth               │")
    print("  ├─────────────────────────────────────────────────────────────┤")

    vram_gb = gpu["vram_mb"] // 1024 if gpu else "?"
    if t1["bandwidth_gbs"]:
        t1_bw = f"{t1['bandwidth_gbs']} GB/s  ({t1['method']})"
    else:
        t1_bw = f"- {t1['method']}"
    print(f"  │  T1    GPU VRAM       {str(vram_gb)+' GB':<12} {t1_bw:<28}│")

    ram_gb = ram["total_mb"] // 1024
    if t2["bandwidth_gbs"]:
        t2_bw = f"{t2['bandwidth_gbs']} GB/s  ({t2['method']})"
    else:
        t2_bw = f"- {t2['method']}"
    print(f"  │  T2    DDR pool       {str(ram_gb)+' GB':<12} {t2_bw:<28}│")
    sus = t2.get("sustained_bandwidth_gbs")
    if sus:
        print(f"  │                                     sustained: {sus} GB/s{'':<14}│")

    if t3["seq_read_gbs"] is not None:
        t3_str = f"{t3['seq_read_gbs']} GB/s seq-r / {t3['seq_write_gbs']} GB/s seq-w"
        print(f"  │  T3    NVMe ({t3['mount']:<6}) {'auto':<12} {t3_str:<28}│")
        if t3["rand4k_read_mbs"] is not None:
            print(f"  │                                     rand-4K: {t3['rand4k_read_mbs']} MB/s{'':<9}│")
    else:
        print(f"  │  T3    NVMe           auto         {t3['method']:<28}│")

    print("  └─────────────────────────────────────────────────────────────┘")

    # Recommendations
    recs = _recommendations(results)
    if recs:
        print()
        print("  Recommendations:")
        for icon, _, msg in recs:
            import re
            clean = re.sub(r'\[/?[^\]]*\]', '', msg)
            print(f"  {icon}  {clean}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GreenBoost workstation benchmark")
    parser.add_argument("--json",        action="store_true", help="Print results as JSON")
    parser.add_argument("--output",      metavar="FILE",      help="Write JSON results to file")
    parser.add_argument("--skip-t1",     action="store_true", help="Skip GPU VRAM benchmark")
    parser.add_argument("--skip-t2",     action="store_true", help="Skip PCIe DDR benchmark")
    parser.add_argument("--skip-t3",     action="store_true", help="Skip NVMe benchmark")
    args = parser.parse_args()

    if not args.json:
        _info("Collecting system information...")

    # System probes (fast, run upfront)
    cpu  = query_cpu()
    ram  = query_ram()
    gpu  = query_gpu()
    swap = query_swap()
    gb_s = query_greenboost()

    # Tier benchmarks with spinners
    if args.skip_t1:
        t1 = {"bandwidth_gbs": None, "method": "skipped", "cupy_available": False}
    else:
        t1 = _run_with_spinner("T1  Measuring GPU VRAM bandwidth...", bench_t1) \
             if not args.json else bench_t1()

    if args.skip_t2:
        t2 = {"bandwidth_gbs": None, "method": "skipped", "cupy_available": False}
    else:
        t2 = _run_with_spinner("T2  Measuring PCIe DDR bandwidth...", bench_t2) \
             if not args.json else bench_t2()

    if args.skip_t3:
        t3 = {"seq_read_gbs": None, "method": "skipped"}
    else:
        t3 = _run_with_spinner("T3  Measuring NVMe sequential and random I/O...", bench_t3) \
             if not args.json else bench_t3()

    results = {
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cpu":        cpu,
        "ram":        ram,
        "gpu":        gpu,
        "swap":       swap,
        "greenboost": gb_s,
        "t1":         t1,
        "t2":         t2,
        "t3":         t3,
    }

    if args.json or args.output:
        js = json.dumps(results, indent=2)
        if args.json:
            print(js)
        if args.output:
            Path(args.output).write_text(js)
            if not args.json:
                _ok(f"JSON results written to {args.output}")
    else:
        if _RICH:
            print_rich_report(results)
        else:
            print_plain_report(results)



if __name__ == "__main__":
    main()

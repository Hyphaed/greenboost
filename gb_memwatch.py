#!/usr/bin/env python3
"""Catch the process that eats the box, before the OOM killer erases it.

Why this exists
---------------
On 2026-08-18 the kernel OOM killer ran twice inside the terminal's own cgroup
and took the whole session with it:

    07:52:32  python3 pid 144373   anon-rss 31.7 GB   total-vm 85.6 GB
    12:30:46  python3 pid 328270   anon-rss 27.0 GB   total-vm 61.8 GB

The kernel's report names the process `python3` and nothing else. By the time
anyone looks, `oom_reaper` has already freed it and `/proc/<pid>` is gone, so
the one question that matters -- WHICH python3 -- cannot be answered after the
fact. Both post-mortems stalled exactly there.

This samples the top RSS consumers on a timer and records each one's full
cmdline plus a `smaps_rollup` breakdown. A process over the alert threshold is
written out the moment it is seen, so the evidence is already on disk when the
kill lands.

It deliberately does NOT try to prevent the OOM. Killing or throttling on a
guess is how you lose the work you were trying to protect. This only observes,
and observing is what was missing.

Usage
-----
    python3 gb_memwatch.py watch                 # follow until Ctrl-C
    python3 gb_memwatch.py watch --interval 5
    python3 gb_memwatch.py top                   # one shot, top consumers now
    python3 gb_memwatch.py report                # what previous runs captured

Cost: one pass over /proc per interval, which is why it can run alongside
inference rather than being something you remember to start after a crash.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

STATE_DIR = Path(os.environ.get(
    "GB_MEMWATCH_DIR", Path.home() / ".local/share/greenboost/memwatch"))
SNAPSHOT_FILE = STATE_DIR / "snapshots.jsonl"

#: A process holding more than this fraction of MemTotal is snapshotted
#: immediately rather than only counted.
ALERT_FRACTION = float(os.environ.get("GB_MEMWATCH_ALERT_FRACTION", "0.25"))

#: Stop recording new samples past this file size. A watcher that fills the
#: disk while looking for a memory problem has made things worse.
MAX_LOG_BYTES = int(os.environ.get("GB_MEMWATCH_MAX_BYTES", str(32 * 1024 * 1024)))


def _meminfo_kb(field: str) -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _read_text(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return ""


def _cmdline(pid: int) -> str:
    raw = _read_text(f"/proc/{pid}/cmdline")
    if not raw:
        # A kernel thread has an empty cmdline; comm is the only name it has.
        return _read_text(f"/proc/{pid}/comm").strip()
    return raw.replace("\0", " ").strip()


def _rollup(pid: int) -> dict:
    """Per-process memory breakdown from smaps_rollup.

    Rss alone cannot distinguish 27 GB of Python objects from 27 GB of mapped
    model file, and that difference decides whether the culprit is the CLI's
    own allocation or the shim's T2 mapping charged to whoever touched it.
    """
    out: dict = {}
    for line in _read_text(f"/proc/{pid}/smaps_rollup").splitlines():
        key, _, rest = line.partition(":")
        if key in ("Rss", "Pss", "Anonymous", "Shared_Clean", "Shared_Dirty",
                   "Private_Clean", "Private_Dirty", "Swap", "Locked"):
            try:
                out[key.lower()] = int(rest.split()[0])   # kB
            except (IndexError, ValueError):
                pass
    return out


def _statm(pid: int) -> tuple[int, int]:
    """(total_vm_kb, rss_kb) — cheap, read for every process on every pass."""
    parts = _read_text(f"/proc/{pid}/statm").split()
    if len(parts) < 2:
        return 0, 0
    page_kb = os.sysconf("SC_PAGE_SIZE") // 1024
    try:
        return int(parts[0]) * page_kb, int(parts[1]) * page_kb
    except ValueError:
        return 0, 0


def sample(top_n: int = 8) -> dict:
    """One pass over /proc. Only the top consumers get the expensive reads."""
    procs: list[tuple[int, int, int]] = []          # (rss_kb, vm_kb, pid)
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        vm_kb, rss_kb = _statm(pid)
        if rss_kb:
            procs.append((rss_kb, vm_kb, pid))
    procs.sort(reverse=True)

    mem_total = _meminfo_kb("MemTotal")
    # None means "no threshold could be computed" (MemTotal unreadable), which
    # is NOT the same as a threshold of zero. Collapsing the two made
    # ALERT_FRACTION=0 — "flag everything" — silently flag nothing.
    alert_kb = int(mem_total * ALERT_FRACTION) if mem_total else None

    top = []
    for rss_kb, vm_kb, pid in procs[:top_n]:
        rec = {
            "pid": pid,
            "rss_mb": round(rss_kb / 1024, 1),
            "vm_mb": round(vm_kb / 1024, 1),
            "cmdline": _cmdline(pid)[:400],
            "oom_score_adj": _read_text(f"/proc/{pid}/oom_score_adj").strip(),
            "rollup_kb": _rollup(pid),
        }
        rec["over_alert"] = alert_kb is not None and rss_kb >= alert_kb
        top.append(rec)

    return {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mem_total_mb": round(mem_total / 1024, 1),
        "mem_available_mb": round(_meminfo_kb("MemAvailable") / 1024, 1),
        "swap_used_mb": round(
            (_meminfo_kb("SwapTotal") - _meminfo_kb("SwapFree")) / 1024, 1),
        "alert_threshold_mb": (round(alert_kb / 1024, 1)
                               if alert_kb is not None else None),
        "top": top,
    }


def _append(rec: dict) -> None:
    # Best-effort, and broader than OSError on purpose: this runs while the box
    # is running out of memory and processes are dying. Any exception here
    # destroys the evidence the tool exists to collect, so a missed line always
    # beats a raise. Same contract as gb_dataflux.emit().
    try:
        Path(STATE_DIR).mkdir(parents=True, exist_ok=True)
        f = Path(SNAPSHOT_FILE)
        if f.exists() and f.stat().st_size > MAX_LOG_BYTES:
            return
        with open(f, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _fmt_row(p: dict) -> str:
    flag = "  <-- OVER THRESHOLD" if p.get("over_alert") else ""
    anon = p.get("rollup_kb", {}).get("anonymous")
    anon_s = f"{anon / 1024:>8.0f}" if anon else "       ?"
    return (f"  {p['rss_mb']:>9.0f} MB rss  {anon_s} MB anon  "
            f"{p['vm_mb']:>9.0f} MB vm  pid {p['pid']:<8} "
            f"{p['cmdline'][:64]}{flag}")


def cmd_top(args) -> int:
    rec = sample(args.top)
    thr = rec["alert_threshold_mb"]
    thr_s = f"{thr:.0f} MB" if thr is not None else "unavailable (MemTotal unreadable)"
    print(f"MemTotal {rec['mem_total_mb']:.0f} MB  ·  "
          f"MemAvailable {rec['mem_available_mb']:.0f} MB  ·  "
          f"swap used {rec['swap_used_mb']:.0f} MB  ·  "
          f"alert at {thr_s}")
    for p in rec["top"]:
        print(_fmt_row(p))
    return 0


def cmd_watch(args) -> int:
    print(f"sampling every {args.interval}s -> {SNAPSHOT_FILE}")
    print("alerting on any process over "
          f"{ALERT_FRACTION:.0%} of MemTotal. Ctrl-C to stop.\n")
    last_alert: set[int] = set()
    try:
        while True:
            rec = sample(args.top)
            over = {p["pid"] for p in rec["top"] if p.get("over_alert")}
            # Always keep a low-frequency trail; write every sample once a
            # process is over the line, because that is the run-up worth having.
            if over or not last_alert:
                _append(rec)
            for p in rec["top"]:
                if p.get("over_alert") and p["pid"] not in last_alert:
                    print(f"[{rec['iso']}] ALERT {_fmt_row(p).strip()}")
            if last_alert and not over:
                print(f"[{rec['iso']}] cleared — "
                      f"MemAvailable {rec['mem_available_mb']:.0f} MB")
            last_alert = over
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\nstopped. {SNAPSHOT_FILE}")
    return 0


def cmd_report(args) -> int:
    if not SNAPSHOT_FILE.exists():
        print(f"no snapshots yet at {SNAPSHOT_FILE} — run: "
              f"python3 gb_memwatch.py watch")
        return 1
    worst: dict | None = None
    n = 0
    with open(SNAPSHOT_FILE) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            n += 1
            for p in rec.get("top", []):
                if worst is None or p["rss_mb"] > worst[1]["rss_mb"]:
                    worst = (rec, p)
    print(f"{n} samples in {SNAPSHOT_FILE}")
    if worst:
        rec, p = worst
        print(f"\nlargest process seen — {rec['iso']} "
              f"(MemAvailable {rec['mem_available_mb']:.0f} MB):")
        print(_fmt_row(p))
        print(f"\n  full cmdline: {p['cmdline']}")
        print(f"  oom_score_adj: {p['oom_score_adj']}")
        if p.get("rollup_kb"):
            print("  smaps_rollup (MB): " + "  ".join(
                f"{k}={v / 1024:.0f}" for k, v in sorted(p["rollup_kb"].items())))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--top", type=int, default=8,
                    help="how many processes to detail per sample (default 8)")
    sub = ap.add_subparsers(dest="cmd")
    p_w = sub.add_parser("watch", help="sample on a timer until interrupted")
    p_w.add_argument("--interval", type=float, default=10.0)
    p_w.set_defaults(fn=cmd_watch)
    sub.add_parser("top", help="one shot").set_defaults(fn=cmd_top)
    sub.add_parser("report", help="summarise captured snapshots").set_defaults(
        fn=cmd_report)
    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        args.fn = cmd_top
        args.interval = 10.0
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

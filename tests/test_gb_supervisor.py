#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_supervisor.py's _RamMonitor , system-wide RAM/swap pressure
visibility added in response to the 2026-07-14 incident: two concurrent
CPU-offloaded (`--offload cpu`) inference processes drove system swap to
~68% used while GreenBoost's own T2 DDR pool accounting read 0 the whole
time (CPU-offload streaming never goes through the shim's T2
cudaMallocManaged path, so GreenBoost's existing pressure signals could
not see it at all).

No GPU, no NVML, no root, no running daemon , pure unit tests against
_RamMonitor.poll() with a mocked meminfo reader.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import gb_supervisor


def _mon(**kw):
    return gb_supervisor._RamMonitor(
        warn_avail_pct=kw.get("warn_avail_pct", 20),
        crit_avail_pct=kw.get("crit_avail_pct", 8),
        warn_swap_pct=kw.get("warn_swap_pct", 40),
        crit_swap_pct=kw.get("crit_swap_pct", 65),
    )


def test_ram_monitor_ok_when_plenty_available():
    mon = _mon()
    mon._read_meminfo = lambda: (64000, 40000, 32000, 0)  # 62.5% avail, 0% swap
    assert mon.poll() == "ok"


def test_ram_monitor_warn_on_low_available_pct():
    mon = _mon()
    # 15% available (< warn 20%, >= crit 8%), no swap pressure
    mon._read_meminfo = lambda: (64000, 9600, 32000, 0)
    assert mon.poll() == "warn"


def test_ram_monitor_critical_on_very_low_available_pct():
    mon = _mon()
    # 5% available (< crit 8%)
    mon._read_meminfo = lambda: (64000, 3200, 32000, 0)
    assert mon.poll() == "critical"


def test_ram_monitor_critical_on_swap_pressure_even_with_plenty_available():
    """The real 2026-07-14 incident shape: MemAvailable can look fine-ish
    while swap is heavily committed by CPU-offloaded processes , swap_pct
    alone must be able to trip 'critical' independent of avail_pct."""
    mon = _mon()
    # 30% available (comfortably above warn=20%), but swap 68% used , the
    # real incident's free/swap ratio (31Gi total, 21Gi used -> ~68%).
    mon._read_meminfo = lambda: (64000, 19200, 32000, 21760)
    assert mon.poll() == "critical"


def test_ram_monitor_warn_on_moderate_swap_pressure():
    mon = _mon()
    # 50% available, swap 45% used (> warn 40%, <= crit 65%)
    mon._read_meminfo = lambda: (64000, 32000, 32000, 14400)
    assert mon.poll() == "warn"


def test_ram_monitor_unknown_when_meminfo_unreadable():
    mon = _mon()
    mon._read_meminfo = lambda: (0, 0, 0, 0)
    assert mon.poll() == "unknown"


def test_ram_monitor_handles_zero_swap_total_without_dividing_by_zero():
    mon = _mon()
    mon._read_meminfo = lambda: (64000, 40000, 0, 0)  # no swap configured at all
    assert mon.poll() == "ok"


def test_ram_monitor_writes_pressure_file(tmp_path):
    mon = _mon()
    mon._read_meminfo = lambda: (64000, 3200, 32000, 21760)  # critical on both signals
    fake_file = tmp_path / "ram_pressure"
    with patch.object(gb_supervisor, "RAM_PRESSURE_FILE", fake_file):
        state = mon.poll()
    assert state == "critical"
    content = fake_file.read_text()
    assert "state=critical" in content
    assert "mem_available_pct=" in content
    assert "swap_used_pct=" in content


def test_ram_monitor_state_transition_only_logs_on_change(caplog):
    """Repeated polls in the same state should not spam CRITICAL logs every
    10s tick , mirrors _VramMonitor's prev_state-gated logging pattern."""
    mon = _mon()
    mon._read_meminfo = lambda: (64000, 3200, 32000, 21760)  # critical
    import logging
    with caplog.at_level(logging.CRITICAL, logger="gb_supervisor"):
        mon.poll()
        mon.poll()
        mon.poll()
    critical_records = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert len(critical_records) == 1

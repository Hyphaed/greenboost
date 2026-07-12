# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
Pytest fixtures shared across the GreenBoost regression suite.

This suite catches the exact classes of regressions the PR-A through PR-V
hardening campaign closed:

- Wire-protocol struct layout (catches header alignment / endian drift)
- PSK file loading rules (mode bits, exact 64 hex chars, hex validation)
- Sequence-number monotonicity (catches PR-G/H11 endian regression)
- handle_cuda_launch payload-size validation (catches PR-C/C6 integer overflow)
- Bvmm-table lifecycle invariants (catches PR-A/C2 silent-drop regression)

No CUDA, no /dev/greenboost, no running daemon needed.  Pure-Python tests
that exercise the wire format + the file-loader logic with mocks.

Run with: pytest tests/  (or `python3 -m pytest tests/`)
"""
import os
import tempfile
from pathlib import Path

# Module-level (not a fixture): runs once, at conftest.py load time, BEFORE
# pytest imports any test module. Several gb_*.py modules (gb_quant, gb_attn,
# gb_cluster, ...) read GREENBOOST_ACTIVE at import time and, if it's "1",
# gb_init auto-bootstraps a REAL background TelemetryManager thread , outside
# any single test's scope, so a per-test monkeypatch can't contain it. On a
# desktop session with GreenBoost's gaming profile active (GREENBOOST_ACTIVE=1
# exported globally), collecting this suite would otherwise start real
# background telemetry. Force it off for every test run; only affects this
# process/its children, never the parent shell.
os.environ["GREENBOOST_ACTIVE"] = "0"

# Session-wide dataflux quarantine (also module-level, set once). Several
# gb_*.py classes (ModelTierManager) emit a dataflux event from __del__-
# triggered cleanup, which fires at a GC-determined time that can be AFTER a
# per-test monkeypatch.setenv has already reverted for that test (function-
# scoped monkeypatch only guarantees the override during the test body, not
# whenever Python's refcounting/GC actually collects the object). Setting the
# "resting" value here , BEFORE any per-test override , means a delayed
# __del__ firing between tests, or at session teardown, still lands in a
# throwaway scratch file instead of the user's real
# ~/.local/share/greenboost/dataflux.jsonl. The per-test fixture below layers
# a fresh per-test path on top for normal test isolation; when IT reverts,
# execution falls back to THIS session scratch path, never the real one.
_SESSION_DATAFLUX_DIR = tempfile.mkdtemp(prefix="gb_dataflux_session_")
os.environ["GREENBOOST_DATAFLUX_LOG"] = str(Path(_SESSION_DATAFLUX_DIR) / "session.jsonl")

import struct

import pytest


@pytest.fixture(autouse=True)
def _isolate_dataflux_log_globally(tmp_path, monkeypatch):
    """gb_quant/gb_attn/gb_model_tier/gb_cluster all call gb_dataflux.emit()
    on real code paths (quantize, TurboQuant activation, tier promote/
    demote, cluster dispatch) , several pre-existing tests exercise those
    paths directly (e.g. test_gb_model_tier's auto_evict tests) without
    mocking gb_dataflux.emit. Autouse + session-wide so EVERY test file gets
    this for free; a test that wants to assert on emitted events still
    mocks gb_dataflux.emit explicitly (mocking wins over this redirect since
    the mock replaces the function entirely).

    gb_dataflux resolves its log path from the GREENBOOST_DATAFLUX_LOG env
    var on every call (gb_dataflux._log_path()), not a frozen module
    constant , monkeypatch.setenv is the reliable lever here; patching a
    pre-computed Path attribute is fragile against import/fixture ordering.
    """
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(tmp_path / "dataflux-test.jsonl"))


# Constants mirroring features/net_fabric.h.  Tests fail loudly if these
# drift from the C header - a failure here means the wire format moved
# without the test suite being updated.
GB_NET_MAGIC = 0x47424E46          # "GBNF"
GB_NET_PORT = 9740
GB_NET_PROTO_VER = 3
GB_NET_HDR_SIZE = 16               # magic(4) + msg_type(2) + flags(2) + payload_len(4) + seq_num(4)
GB_NET_MAX_MSG_SIZE = 4 * 1024 * 1024
GB_NET_MAX_KERNEL_NAME = 256

# Message types - keep in sync with features/net_fabric.h
GB_MSG_HANDSHAKE = 1
GB_MSG_HANDSHAKE_RESP = 2
GB_MSG_HEARTBEAT = 3
GB_MSG_CUDA_MALLOC = 10
GB_MSG_CUDA_FREE = 11
GB_MSG_CUDA_MEMCPY_H2D = 12
GB_MSG_CUDA_MEMCPY_D2H = 13
GB_MSG_CUDA_LAUNCH = 14
GB_MSG_CUDA_REGISTER_FN = 15
GB_MSG_CUDA_EXEC = 16
GB_MSG_RESPONSE = 100

GB_NET_FLAG_RESPONSE = 0x0001


@pytest.fixture
def wire_header_pack():
    """Return a function that packs a gb_net_header to its 16-byte LE wire form."""
    def _pack(magic=GB_NET_MAGIC, msg_type=GB_MSG_HANDSHAKE, flags=0,
              payload_len=0, seq_num=1):
        # Format: < I H H I I  (little-endian, packed)
        return struct.pack("<IHHII", magic, msg_type, flags, payload_len, seq_num)
    return _pack


@pytest.fixture
def wire_header_unpack():
    """Return a function that unpacks a 16-byte LE wire form into a dict."""
    def _unpack(buf):
        magic, msg_type, flags, payload_len, seq_num = struct.unpack("<IHHII", buf[:16])
        return {
            "magic": magic, "msg_type": msg_type, "flags": flags,
            "payload_len": payload_len, "seq_num": seq_num,
        }
    return _unpack


@pytest.fixture
def tmp_psk_file(tmp_path):
    """Create a temporary PSK keyfile + return its path.  Caller can mutate
    via .write_text() to test different mode/content scenarios."""
    p = tmp_path / "cluster.key"
    # Default: well-formed 64-char hex string, mode 0600
    p.write_text("0123456789abcdef" * 4)
    os.chmod(str(p), 0o600)
    return p

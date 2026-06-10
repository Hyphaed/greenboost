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
import struct
import tempfile

import pytest


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

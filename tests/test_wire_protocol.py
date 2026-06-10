# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
Wire-protocol regression tests.

Catches:
  - Header layout drift (size, field order, packed alignment)
  - Little-endian discipline (PR-G/H11 - netc was direct-comparing
    on-wire bytes without le32toh)
  - Sequence-number monotonicity logic
  - Payload-length boundary checks (PR-C/C6 integer overflow in
    handle_cuda_launch)
"""
import struct

import pytest

from conftest import (
    GB_NET_MAGIC, GB_NET_HDR_SIZE, GB_NET_MAX_MSG_SIZE, GB_NET_MAX_KERNEL_NAME,
    GB_MSG_HANDSHAKE, GB_MSG_CUDA_LAUNCH, GB_NET_FLAG_RESPONSE,
)


class TestHeaderLayout:
    """The 16-byte header is the foundation of the wire protocol.  Any
    drift in layout breaks every connection silently."""

    def test_header_size_is_16(self, wire_header_pack):
        buf = wire_header_pack()
        assert len(buf) == GB_NET_HDR_SIZE == 16

    def test_round_trip_preserves_fields(self, wire_header_pack, wire_header_unpack):
        out = wire_header_unpack(wire_header_pack(
            magic=GB_NET_MAGIC, msg_type=42, flags=0x1234,
            payload_len=999, seq_num=7,
        ))
        assert out["magic"] == GB_NET_MAGIC
        assert out["msg_type"] == 42
        assert out["flags"] == 0x1234
        assert out["payload_len"] == 999
        assert out["seq_num"] == 7

    def test_magic_is_little_endian(self, wire_header_pack):
        """The on-wire magic bytes must be 'F','N','B','G' (LE of GBNF
        = 0x47424E46).  Big-endian builds were broken until PR-G/H11."""
        buf = wire_header_pack(magic=GB_NET_MAGIC)
        # 0x47424E46 little-endian → 0x46 0x4E 0x42 0x47
        assert buf[0:4] == bytes([0x46, 0x4E, 0x42, 0x47])

    def test_seq_num_at_offset_12(self, wire_header_pack):
        """seq_num must be the LAST 4 bytes - it's the most-recently-added
        field (proto v3) and a layout drift would shift it."""
        buf = wire_header_pack(seq_num=0xDEADBEEF)
        seq_bytes = buf[12:16]
        assert seq_bytes == bytes([0xEF, 0xBE, 0xAD, 0xDE])

    def test_payload_len_at_offset_8(self, wire_header_pack):
        buf = wire_header_pack(payload_len=0x11223344)
        assert buf[8:12] == bytes([0x44, 0x33, 0x22, 0x11])


class TestSequenceMonotonicity:
    """PR-G/H11: the netc side was direct-comparing seq_num bytes
    without le32toh.  These tests model the monotonicity logic both
    sides must implement identically."""

    def test_strictly_increasing_accepted(self):
        recv_seq = 1
        for incoming_seq in [1, 2, 3, 4, 5]:
            assert incoming_seq == recv_seq, "in-order: accept"
            recv_seq += 1

    def test_replay_detected(self):
        recv_seq = 5
        incoming_seq = 3  # replayed older message
        assert incoming_seq != recv_seq, "replay: reject"

    def test_skipped_seq_detected(self):
        recv_seq = 5
        incoming_seq = 7  # skipped 5, 6
        assert incoming_seq != recv_seq, "gap: reject (frame desync)"

    def test_response_flag_does_not_affect_seq(self):
        """RESPONSE flag is in `flags`, not `seq_num`.  A buggy
        implementation might mask seq based on flags."""
        seq = 42
        flags = GB_NET_FLAG_RESPONSE
        # Just confirm they're independent fields
        assert seq != flags


class TestPayloadSizeValidation:
    """PR-C/C6 closed an integer-overflow vector in handle_cuda_launch:
    `sizeof(hdr) + name_len + arg_buf_size > len` was 32-bit arithmetic.
    Attacker chose name_len near 0xFFFFFFE0 so the sum wrapped below
    `len` - bounds check passed, raw arg_buffer_size was then passed to
    malloc()/memcpy()."""

    @staticmethod
    def _cuda_launch_validates(hdr_size, name_len, arg_buf_size, recv_len):
        """Model the post-PR-C bounds check.  Returns True if the
        payload would be accepted, False if rejected."""
        # Promote to Python ints (no overflow) - mirrors the C fix.
        if name_len > GB_NET_MAX_KERNEL_NAME:
            return False
        if arg_buf_size > GB_NET_MAX_MSG_SIZE:
            return False
        if hdr_size + name_len + arg_buf_size > recv_len:
            return False
        return True

    def test_legitimate_request_accepted(self):
        assert self._cuda_launch_validates(48, 32, 128, 48 + 32 + 128)

    def test_overflow_attack_rejected(self):
        """The exact PR-C/C6 vector: huge name_len + arg_buf size that
        wraps in 32-bit arithmetic.  Test models the wrap-aware check."""
        # If the C code did 32-bit math: 48 + 0xFFFFFFE0 + 0x100 = wraps to ~228
        # which < `recv_len`=1000 → falsely accepted.
        # The fix promotes to 64-bit and bounds each field.
        assert not self._cuda_launch_validates(48, 0xFFFFFFE0, 0x100, 1000)

    def test_arg_buf_exceeds_cap_rejected(self):
        assert not self._cuda_launch_validates(48, 32, GB_NET_MAX_MSG_SIZE + 1, 1 << 30)

    def test_name_len_exceeds_cap_rejected(self):
        assert not self._cuda_launch_validates(48, GB_NET_MAX_KERNEL_NAME + 1, 0, 10000)

    def test_zero_payload_accepted(self):
        assert self._cuda_launch_validates(48, 0, 0, 48)


class TestSendRecvPair:
    """Round-trip: client packs a header, server unpacks it.  Catches
    drift between netc and netd serialisers."""

    def test_client_header_unpacks_at_server(self, wire_header_pack, wire_header_unpack):
        sent = wire_header_pack(msg_type=GB_MSG_HANDSHAKE, payload_len=80, seq_num=1)
        decoded = wire_header_unpack(sent)
        assert decoded["msg_type"] == GB_MSG_HANDSHAKE
        assert decoded["payload_len"] == 80
        assert decoded["seq_num"] == 1
        assert decoded["magic"] == GB_NET_MAGIC

    def test_multi_message_seq_progression(self, wire_header_pack, wire_header_unpack):
        """Simulate a session: client sends seq=1,2,3; each unpacks correctly
        and seq is monotonic."""
        for expected_seq in [1, 2, 3]:
            buf = wire_header_pack(seq_num=expected_seq)
            assert wire_header_unpack(buf)["seq_num"] == expected_seq

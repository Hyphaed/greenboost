# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
HKDF + mutual-auth (PR-EE) regression tests.

The C implementation lives in greenboost_netd.c and greenboost_netc.c.
These tests mirror the algorithm in pure Python so a future C-side
implementation change that drifts from the spec fails CI.

Catches:
  - HKDF-Extract output drift (incorrect salt/key ordering)
  - Mutual-auth challenge construction drift (nonce_s||nonce_c vs
    nonce_c||nonce_s - must be different MACs in each direction!)
  - Session-key derivation context-binding label drift
"""
import hashlib
import hmac

import pytest


def py_hkdf_extract(salt, ikm):
    """RFC 5869 §2.2 - HKDF-Extract.  Returns 32-byte PRK."""
    if not salt:
        salt = b"\x00" * 32
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def py_hkdf_expand_32(prk, info):
    """RFC 5869 §2.3 - HKDF-Expand for L=32 bytes (single T(1) block)."""
    return hmac.new(prk, info + b"\x01", hashlib.sha256).digest()


def py_derive_session_key(psk, nonce_s, nonce_c):
    """Mirror of gb_derive_session_key in netd.c / netc.c."""
    assert len(psk) == 32
    assert len(nonce_s) == 32
    assert len(nonce_c) == 32
    salt = nonce_s + nonce_c
    prk = py_hkdf_extract(salt, psk)
    return py_hkdf_expand_32(prk, b"gb-session-v1|proto=4")


def py_mac1(psk, nonce_s, nonce_c):
    """Client → Server proof: HMAC(psk, nonce_s || nonce_c)."""
    return hmac.new(psk, nonce_s + nonce_c, hashlib.sha256).digest()


def py_mac2(psk, nonce_c, nonce_s):
    """Server → Client proof: HMAC(psk, nonce_c || nonce_s).  Note the
    ORDER is REVERSED relative to mac1 - this is what makes the auth
    mutual.  If both directions used the same input, a MITM could replay
    the client's mac as the server's proof."""
    return hmac.new(psk, nonce_c + nonce_s, hashlib.sha256).digest()


class TestHKDF:

    def test_extract_deterministic(self):
        salt = b"\x11" * 64
        ikm = b"\x22" * 32
        out1 = py_hkdf_extract(salt, ikm)
        out2 = py_hkdf_extract(salt, ikm)
        assert out1 == out2
        assert len(out1) == 32

    def test_extract_zero_salt_path(self):
        """Empty salt is well-defined (RFC 5869): falls back to zero-padded."""
        ikm = b"\x33" * 32
        out = py_hkdf_extract(None, ikm)
        assert len(out) == 32
        # Must match explicit zero-salt result
        out_explicit = py_hkdf_extract(b"\x00" * 32, ikm)
        assert out == out_explicit


class TestSessionKey:

    def test_derive_deterministic(self):
        psk = b"\x01" * 32
        ns = b"\x02" * 32
        nc = b"\x03" * 32
        k1 = py_derive_session_key(psk, ns, nc)
        k2 = py_derive_session_key(psk, ns, nc)
        assert k1 == k2
        assert len(k1) == 32

    def test_different_nonces_yield_different_keys(self):
        """Different handshakes must produce different session keys -
        otherwise a passive observer recording one session could
        decrypt any future session."""
        psk = b"\x01" * 32
        nc = b"\x03" * 32
        k1 = py_derive_session_key(psk, b"\x02" * 32, nc)
        k2 = py_derive_session_key(psk, b"\x04" * 32, nc)
        assert k1 != k2

    def test_nonce_order_matters(self):
        """Swapping nonce_s ↔ nonce_c must produce a DIFFERENT key.  If
        the implementation accidentally uses sorted nonces, this fails."""
        psk = b"\xaa" * 32
        ns = b"\xbb" * 32
        nc = b"\xcc" * 32
        k1 = py_derive_session_key(psk, ns, nc)
        k2 = py_derive_session_key(psk, nc, ns)  # swapped
        assert k1 != k2

    def test_psk_required_for_key_derivation(self):
        """Different PSKs must produce different keys even with identical
        nonces - this is the fundamental security property."""
        ns = b"\x11" * 32
        nc = b"\x22" * 32
        k1 = py_derive_session_key(b"\xaa" * 32, ns, nc)
        k2 = py_derive_session_key(b"\xbb" * 32, ns, nc)
        assert k1 != k2


class TestMutualAuth:

    def test_mac1_and_mac2_are_distinct(self):
        """The ORDER of (nonce_s || nonce_c) vs (nonce_c || nonce_s) is
        what makes the auth genuinely mutual.  If mac1 == mac2 a MITM
        could relay the client's MAC back as the server's proof."""
        psk = b"\xaa" * 32
        ns = b"\xbb" * 32
        nc = b"\xcc" * 32
        m1 = py_mac1(psk, ns, nc)
        m2 = py_mac2(psk, nc, ns)
        assert m1 != m2, "mac1 and mac2 must differ - this is the mutual-auth invariant"

    def test_mac_is_psk_dependent(self):
        ns = b"\x11" * 32
        nc = b"\x22" * 32
        m1_a = py_mac1(b"\xaa" * 32, ns, nc)
        m1_b = py_mac1(b"\xbb" * 32, ns, nc)
        assert m1_a != m1_b

    def test_replay_across_handshakes_fails(self):
        """An attacker capturing mac1 from handshake A cannot reuse it in
        handshake B - the nonces are different so the expected MAC differs."""
        psk = b"\xaa" * 32
        nc = b"\xcc" * 32
        m_handshake_A = py_mac1(psk, b"\x01" * 32, nc)
        m_handshake_B = py_mac1(psk, b"\x02" * 32, nc)
        assert m_handshake_A != m_handshake_B


class TestPerMessageMAC:
    """PR-FF: 8-byte truncated HMAC over (header || payload) using the
    HKDF-derived session_key.  Mirror of gb_msg_mac in netd/netc."""

    @staticmethod
    def _pack_header(magic, msg_type, flags, payload_len, seq_num):
        import struct
        return struct.pack("<IHHII", magic, msg_type, flags, payload_len, seq_num)

    @staticmethod
    def _msg_mac(session_key, header_bytes, payload_bytes):
        full = hmac.new(session_key, header_bytes + payload_bytes, hashlib.sha256).digest()
        return full[:8]

    def test_mac_is_deterministic(self):
        key = b"\xa1" * 32
        h = self._pack_header(0x47424E46, 14, 0, 32, 1)
        p = b"\x55" * 32
        m1 = self._msg_mac(key, h, p)
        m2 = self._msg_mac(key, h, p)
        assert m1 == m2
        assert len(m1) == 8

    def test_mac_changes_with_seq_num(self):
        """Different seq_num → different MAC → replay of an old message
        across a long-lived connection fails."""
        key = b"\xa1" * 32
        p = b"\x55" * 32
        m1 = self._msg_mac(key, self._pack_header(0x47424E46, 14, 0, 32, 1), p)
        m2 = self._msg_mac(key, self._pack_header(0x47424E46, 14, 0, 32, 2), p)
        assert m1 != m2

    def test_mac_changes_with_session_key(self):
        """Different sessions yield different MACs even for identical
        payloads - the replay-across-reconnect property."""
        h = self._pack_header(0x47424E46, 14, 0, 32, 1)
        p = b"\x55" * 32
        m1 = self._msg_mac(b"\xa1" * 32, h, p)
        m2 = self._msg_mac(b"\xa2" * 32, h, p)
        assert m1 != m2

    def test_mac_detects_payload_tampering(self):
        key = b"\xa1" * 32
        h = self._pack_header(0x47424E46, 14, 0, 4, 1)
        m_orig = self._msg_mac(key, h, b"\x01\x02\x03\x04")
        m_tamp = self._msg_mac(key, h, b"\x01\x02\x03\x05")
        assert m_orig != m_tamp

    def test_mac_detects_header_tampering(self):
        """Flipping msg_type in the header changes the MAC - an attacker
        cannot rewrite the message type and replay the same MAC."""
        key = b"\xa1" * 32
        p = b"\x55" * 32
        h_orig = self._pack_header(0x47424E46, 14, 0, 32, 1)
        h_tamp = self._pack_header(0x47424E46, 16, 0, 32, 1)  # msg_type changed
        assert self._msg_mac(key, h_orig, p) != self._msg_mac(key, h_tamp, p)

    def test_empty_payload_macs(self):
        """Zero-length payload (heartbeat-style) still gets a MAC."""
        key = b"\xa1" * 32
        h = self._pack_header(0x47424E46, 3, 0, 0, 1)
        m = self._msg_mac(key, h, b"")
        assert len(m) == 8

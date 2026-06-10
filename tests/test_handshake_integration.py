# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
PR-OO: end-to-end handshake state machine test.

Pure Python - no daemon required.  We implement BOTH sides of the
handshake protocol over a socket pair (socketpair(AF_UNIX, SOCK_STREAM))
and verify that the state machines on netd and netc agree.

Catches regressions in:
  - v3 / v4 protocol detection
  - Mutual-auth correctness (mac1 != mac2 invariant)
  - HKDF session-key derivation matches both sides
  - Per-message MAC handshake state machine
  - Backward-compat (v3 client + v3 server interop unchanged)

Implementation note: the C side uses `getrandom` for nonces; here we use
deterministic test vectors so failures are reproducible.
"""
import hashlib
import hmac
import os
import socket
import struct

import pytest

# Mirror the C constants - keep in sync with net_fabric.h.
GB_NET_MAGIC = 0x47424E46
GB_NET_HDR_SIZE = 16
GB_NET_MAX_MSG_SIZE = 4 * 1024 * 1024
GB_MSG_HANDSHAKE = 1


# ---- protocol primitives --------------------------------------------------

def hmac_sha256(key: bytes, msg: bytes) -> bytes:
    return hmac.new(key, msg, hashlib.sha256).digest()


def derive_session_key(psk: bytes, nonce_s: bytes, nonce_c: bytes) -> bytes:
    """Mirror gb_derive_session_key (netd.c + netc.c)."""
    salt = nonce_s + nonce_c
    prk = hmac.new(salt, psk, hashlib.sha256).digest()
    info = b"gb-session-v1|proto=4"
    return hmac.new(prk, info + b"\x01", hashlib.sha256).digest()


def msg_mac(session_key: bytes, header: bytes, payload: bytes) -> bytes:
    """Mirror gb_msg_mac (netd.c) - 8-byte truncated HMAC over hdr || payload."""
    return hmac.new(session_key, header + payload, hashlib.sha256).digest()[:8]


def pack_header(msg_type, flags, payload_len, seq_num):
    return struct.pack("<IHHII", GB_NET_MAGIC, msg_type, flags, payload_len, seq_num)


# ---- handshake state machines (Python mirrors of netd / netc) -------------

class HandshakeError(Exception):
    pass


def server_handshake_v3(sock: socket.socket, psk: bytes,
                         nonce_s: bytes = b"\x01" * 32) -> None:
    """v3 server side: send nonce_s, recv 32-byte mac, verify."""
    sock.sendall(nonce_s)
    mac = sock.recv(32)
    if len(mac) != 32:
        raise HandshakeError("short MAC")
    expected = hmac_sha256(psk, nonce_s)
    if not hmac.compare_digest(mac, expected):
        raise HandshakeError("bad MAC")


def client_handshake_v3(sock: socket.socket, psk: bytes) -> None:
    """v3 client side: recv nonce_s, send mac=HMAC(psk, nonce_s)."""
    nonce_s = sock.recv(32)
    if len(nonce_s) != 32:
        raise HandshakeError("short nonce_s")
    mac = hmac_sha256(psk, nonce_s)
    sock.sendall(mac)


def server_handshake_v4(sock: socket.socket, psk: bytes,
                         nonce_s: bytes = b"\x01" * 32) -> bytes:
    """v4 server side: send nonce_s, recv nonce_c || mac1, send mac2.
    Returns the derived session_key."""
    sock.sendall(nonce_s)
    reply = sock.recv(64)
    if len(reply) != 64:
        raise HandshakeError(f"short v4 reply: {len(reply)} bytes")
    nonce_c, mac1 = reply[:32], reply[32:]
    expected_mac1 = hmac_sha256(psk, nonce_s + nonce_c)
    if not hmac.compare_digest(mac1, expected_mac1):
        raise HandshakeError("bad mac1")
    mac2 = hmac_sha256(psk, nonce_c + nonce_s)
    sock.sendall(mac2)
    return derive_session_key(psk, nonce_s, nonce_c)


def client_handshake_v4(sock: socket.socket, psk: bytes,
                         nonce_c: bytes = b"\x02" * 32) -> bytes:
    """v4 client side: recv nonce_s, send nonce_c || mac1, recv + verify mac2."""
    nonce_s = sock.recv(32)
    if len(nonce_s) != 32:
        raise HandshakeError("short nonce_s")
    mac1 = hmac_sha256(psk, nonce_s + nonce_c)
    sock.sendall(nonce_c + mac1)
    mac2 = sock.recv(32)
    if len(mac2) != 32:
        raise HandshakeError("short mac2")
    expected_mac2 = hmac_sha256(psk, nonce_c + nonce_s)
    if not hmac.compare_digest(mac2, expected_mac2):
        raise HandshakeError("bad mac2 - server doesn't have matching PSK")
    return derive_session_key(psk, nonce_s, nonce_c)


# ---- tests ----------------------------------------------------------------

@pytest.fixture
def sockpair():
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    a.settimeout(2)
    b.settimeout(2)
    yield a, b
    a.close()
    b.close()


class TestV3Handshake:

    def test_v3_v3_interop(self, sockpair):
        """Baseline: v3 client + v3 server, matching PSK, succeeds."""
        srv, cli = sockpair
        psk = b"\xa1" * 32
        # Run "server" on srv, "client" on cli; both blocking ops alternate.
        # socketpair gives full duplex so we drive them sequentially.
        import threading
        err_box = [None]
        def server_fn():
            try: server_handshake_v3(srv, psk)
            except Exception as e: err_box[0] = e
        t = threading.Thread(target=server_fn)
        t.start()
        client_handshake_v3(cli, psk)
        t.join(timeout=2)
        assert err_box[0] is None, f"server: {err_box[0]}"

    def test_v3_wrong_psk_rejected(self, sockpair):
        srv, cli = sockpair
        import threading
        err_box = [None]
        def server_fn():
            try: server_handshake_v3(srv, b"\xa1" * 32)
            except HandshakeError as e: err_box[0] = e
        t = threading.Thread(target=server_fn)
        t.start()
        client_handshake_v3(cli, b"\xb2" * 32)  # different PSK
        t.join(timeout=2)
        assert err_box[0] is not None
        assert "bad MAC" in str(err_box[0])


class TestV4Handshake:

    def test_v4_v4_interop(self, sockpair):
        """v4 client + v4 server: both derive the same session_key."""
        srv, cli = sockpair
        psk = b"\xa1" * 32
        ns = b"\x11" * 32
        nc = b"\x22" * 32
        import threading
        srv_box = [None, None]
        def server_fn():
            try:
                key = server_handshake_v4(srv, psk, nonce_s=ns)
                srv_box[0] = key
            except Exception as e:
                srv_box[1] = e
        t = threading.Thread(target=server_fn)
        t.start()
        cli_key = client_handshake_v4(cli, psk, nonce_c=nc)
        t.join(timeout=2)
        assert srv_box[1] is None, f"server: {srv_box[1]}"
        assert srv_box[0] == cli_key, "both sides MUST derive the same key"

    def test_v4_wrong_psk_rejected_by_server(self, sockpair):
        """v4 client with wrong PSK: server's mac1 verify fails."""
        srv, cli = sockpair
        import threading
        err_box = [None]
        def server_fn():
            try: server_handshake_v4(srv, b"\xa1" * 32)
            except HandshakeError as e: err_box[0] = e
        t = threading.Thread(target=server_fn)
        t.start()
        try:
            client_handshake_v4(cli, b"\xb2" * 32)
        except (HandshakeError, ConnectionResetError, BrokenPipeError, OSError):
            pass  # expected - client may detect via mac2 or socket close
        t.join(timeout=2)
        assert err_box[0] is not None
        assert "mac1" in str(err_box[0])

    def test_v4_server_proves_psk_knowledge(self, sockpair):
        """The whole point of v4: server's mac2 proves it knows the PSK.
        A man-in-the-middle who only saw a prior handshake can't compute
        mac2 because nonce_c is fresh per handshake."""
        srv, cli = sockpair
        psk = b"\xa1" * 32
        # Replay attack: man-in-middle remembers mac2 from a PREVIOUS run.
        # But this run uses a NEW nonce_c, so the captured mac2 is wrong.
        import threading
        srv_box = [None]
        def fake_server_fn():
            """Send valid nonce_s, but a REPLAYED (wrong) mac2."""
            nonce_s = b"\x33" * 32
            srv.sendall(nonce_s)
            reply = srv.recv(64)
            # MITM sends a mac2 from a previous (different) handshake
            srv.sendall(b"\xff" * 32)
            srv_box[0] = "mitm sent garbage mac2"
        t = threading.Thread(target=fake_server_fn)
        t.start()
        with pytest.raises(HandshakeError, match="bad mac2"):
            client_handshake_v4(cli, psk, nonce_c=b"\x44" * 32)
        t.join(timeout=2)


class TestV3V4Crosscompat:

    def test_v3_client_v4_server_fails(self, sockpair):
        """v4 server expects 64 bytes back; v3 client sends 32.  Server
        recv blocks waiting for more; we model the timeout by closing
        the cli side after the client finishes."""
        srv, cli = sockpair
        psk = b"\xa1" * 32
        import threading
        srv_box = [None]
        def server_fn():
            try: server_handshake_v4(srv, psk)
            except (HandshakeError, OSError) as e: srv_box[0] = type(e).__name__
        t = threading.Thread(target=server_fn)
        t.start()
        client_handshake_v3(cli, psk)
        cli.shutdown(socket.SHUT_WR)
        t.join(timeout=2)
        assert srv_box[0] is not None  # server detected the short reply

    def test_v4_client_v3_server_fails(self, sockpair):
        """v3 server reads 32 bytes (treats it as MAC); v4 client sent
        64 bytes (nonce_c || mac1).  The server interprets the FIRST 32
        bytes as the MAC; that's nonce_c which doesn't match HMAC(psk,
        nonce_s).  Server rejects with bad MAC."""
        srv, cli = sockpair
        psk = b"\xa1" * 32
        nonce_s = b"\xaa" * 32
        import threading
        srv_box = [None]
        def server_fn():
            try: server_handshake_v3(srv, psk, nonce_s=nonce_s)
            except HandshakeError as e: srv_box[0] = str(e)
        t = threading.Thread(target=server_fn)
        t.start()
        try:
            client_handshake_v4(cli, psk)
        except (HandshakeError, OSError, ConnectionResetError, BrokenPipeError):
            pass
        t.join(timeout=2)
        assert srv_box[0] is not None and "bad MAC" in srv_box[0]


class TestPerMessageMACFlow:

    def test_mac_verifies_after_v4_handshake(self, sockpair):
        """Full v4 setup: handshake → both derive session_key → server
        sends a framed message with MAC → client verifies."""
        srv, cli = sockpair
        psk = b"\xa1" * 32
        import threading
        srv_box = [None]
        def server_fn():
            try:
                key = server_handshake_v4(srv, psk)
                # Send a heartbeat-style empty message + MAC
                hdr = pack_header(msg_type=3, flags=0, payload_len=0, seq_num=0)
                mac = msg_mac(key, hdr, b"")
                srv.sendall(hdr + mac)
            except Exception as e:
                srv_box[0] = e
        t = threading.Thread(target=server_fn)
        t.start()
        cli_key = client_handshake_v4(cli, psk)
        hdr = cli.recv(GB_NET_HDR_SIZE)
        recv_mac = cli.recv(8)
        t.join(timeout=2)
        assert srv_box[0] is None
        # Verify the MAC matches what cli_key would produce.
        expected = msg_mac(cli_key, hdr, b"")
        assert recv_mac == expected, "post-handshake MAC mismatch"

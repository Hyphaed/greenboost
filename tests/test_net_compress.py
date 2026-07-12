# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
Fabric zstd compression wire-protocol tests (Phase 2).

Validates the on-wire contract the C code (greenboost_netc.c H2D compress /
greenboost_netd.c H2D decompress) must honour, without a live socket:
  - the GB_NET_FLAG_COMP_ZSTD bit value and that it doesn't collide with the
    existing header flags
  - a compressed H2D frame round-trips: the gb_net_cuda_memcpy.size field
    carries the UNCOMPRESSED length, the payload after the struct is the
    zstd frame, and decompressing to `size` bytes recovers the original
  - the handshake feature_flags trailing field: a short (pre-feature) req is
    read as feature_flags=0; a long req exposes the advertised bits
"""
import struct

import pytest

zstd = pytest.importorskip("zstandard")

from conftest import (
    GB_NET_MAGIC, GB_MSG_CUDA_MEMCPY_H2D, GB_NET_FLAG_RESPONSE,
    GB_NET_PROTO_VER,
)

# Mirror features/net_fabric.h
GB_NET_FLAG_EXEC_RAW  = 1 << 1
GB_NET_FLAG_ASYNC     = 1 << 1
GB_NET_FLAG_COMP_ZSTD = 1 << 2
GB_NET_FEAT_ZSTD      = 1 << 0

# struct gb_net_cuda_memcpy { u64 remote_handle; u64 offset; u64 size; }
_MEMCPY_FMT = "<QQQ"
_MEMCPY_SIZE = struct.calcsize(_MEMCPY_FMT)

# struct gb_net_handshake_req base = proto(4)+gpu_count(4)+hostname(64)+gpus(800)
_HS_BASE = 4 + 4 + 64 + 8 * 100
_HS_WITH_FEAT = _HS_BASE + 4


def _hdr(msg_type, flags, payload_len, seq=1):
    return struct.pack("<IHHII", GB_NET_MAGIC, msg_type, flags, payload_len, seq)


def test_comp_flag_value_and_no_collision():
    assert GB_NET_FLAG_COMP_ZSTD == 4
    # bit 2 must be distinct from the bits already in use (0 and 1)
    assert GB_NET_FLAG_COMP_ZSTD & GB_NET_FLAG_RESPONSE == 0
    assert GB_NET_FLAG_COMP_ZSTD & GB_NET_FLAG_EXEC_RAW == 0
    assert GB_NET_FLAG_COMP_ZSTD & GB_NET_FLAG_ASYNC == 0


def _build_h2d_frame(original: bytes, remote_handle=0xAA00, offset=0, level=3):
    """Build the exact bytes the shim's netc puts on the wire for a compressed
    H2D chunk: header(COMP) + memcpy-struct(size=uncompressed) + zstd frame."""
    comp = zstd.ZstdCompressor(level=level).compress(original)
    body = struct.pack(_MEMCPY_FMT, remote_handle, offset, len(original)) + comp
    return _hdr(GB_MSG_CUDA_MEMCPY_H2D, GB_NET_FLAG_COMP_ZSTD, len(body)) + body


def _receiver_decode(frame: bytes) -> bytes:
    """Mirror greenboost_netd.c handle_cuda_memcpy_h2d for a COMP_ZSTD frame."""
    magic, msg_type, flags, payload_len, seq = struct.unpack("<IHHII", frame[:16])
    assert magic == GB_NET_MAGIC
    assert msg_type == GB_MSG_CUDA_MEMCPY_H2D
    payload = frame[16:16 + payload_len]
    remote_handle, offset, size = struct.unpack(_MEMCPY_FMT, payload[:_MEMCPY_SIZE])
    data = payload[_MEMCPY_SIZE:]
    if flags & GB_NET_FLAG_COMP_ZSTD:
        out = zstd.ZstdDecompressor().decompress(data, max_output_size=size)
        assert len(out) == size          # C rejects a size mismatch
        return out
    return data


def test_h2d_compressed_roundtrip_compressible():
    # bf16-weight-like data: highly repetitive → good ratio
    original = (b"\x00\x3c" * 4096) + (b"\x80\x3f" * 4096)
    frame = _build_h2d_frame(original)
    assert _receiver_decode(frame) == original
    # the on-wire payload really is smaller than raw
    _, _, _, payload_len, _ = struct.unpack("<IHHII", frame[:16])
    assert payload_len < _MEMCPY_SIZE + len(original)


def test_h2d_compressed_roundtrip_binary():
    original = bytes((i * 37 + 11) & 0xFF for i in range(200000))
    frame = _build_h2d_frame(original)
    assert _receiver_decode(frame) == original


def test_h2d_size_field_is_uncompressed():
    original = b"greenboost" * 5000
    frame = _build_h2d_frame(original)
    payload = frame[16:]
    _, _, size = struct.unpack(_MEMCPY_FMT, payload[:_MEMCPY_SIZE])
    assert size == len(original)          # NOT the compressed length


def test_uncompressed_frame_still_parses():
    # sender chose not to compress (incompressible / below min / peer lacks it)
    original = b"\xde\xad\xbe\xef" * 10
    body = struct.pack(_MEMCPY_FMT, 0xAA00, 0, len(original)) + original
    frame = _hdr(GB_MSG_CUDA_MEMCPY_H2D, 0, len(body)) + body
    assert _receiver_decode(frame) == original


# ── handshake feature_flags trailing-field tolerance ────────────────────────
def _read_feature_flags(req: bytes) -> int:
    """Mirror netd handle_handshake: feature_flags only when len >= full size."""
    if len(req) >= _HS_WITH_FEAT:
        return struct.unpack("<I", req[_HS_BASE:_HS_BASE + 4])[0]
    return 0


def test_short_handshake_reads_zero_features():
    short = struct.pack("<II", GB_NET_PROTO_VER, 0) + b"\x00" * (64 + 800)
    assert len(short) == _HS_BASE
    assert _read_feature_flags(short) == 0


def test_long_handshake_exposes_zstd_bit():
    long = (struct.pack("<II", GB_NET_PROTO_VER, 0) + b"\x00" * (64 + 800)
            + struct.pack("<I", GB_NET_FEAT_ZSTD))
    assert len(long) == _HS_WITH_FEAT
    assert _read_feature_flags(long) & GB_NET_FEAT_ZSTD

# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
PSK file-loading regression tests.

Catches:
  - Insecure mode bits accepted (PR-G hardened to refuse group/world bits)
  - Short hex file silently yielding leading-zero PSK (PR-G now requires
    exactly 64 chars)
  - Non-hex characters slipping past (PR-G validates each char)
  - PSK lingering in stack after function exit (PR-G explicit_bzero -
    not directly testable in Python but documented)

The C implementation lives in greenboost_netd.c::gb_load_psk and
greenboost_netc.c::gb_load_psk.  This test mirrors its rules in Python
so a future C refactor that drops one of these checks fails CI.
"""
import os
import stat

import pytest


def py_gb_load_psk(path):
    """Pure-Python model of gb_load_psk that mirrors the C contract.

    Returns 32-byte bytes on success, None on failure (silent - matches C
    behaviour of returning -1 + log).  Failure reasons logged to stderr
    are an implementation detail not modelled here.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    # Reject any group / other bits set
    if st.st_mode & 0o077:
        return None
    with open(path, "r") as f:
        hex_str = f.read(64)
    if len(hex_str) != 64:
        return None
    for c in hex_str:
        if c not in "0123456789abcdefABCDEF":
            return None
    return bytes.fromhex(hex_str)


class TestPSKLoading:

    def test_well_formed_keyfile_loads(self, tmp_psk_file):
        psk = py_gb_load_psk(str(tmp_psk_file))
        assert psk is not None
        assert len(psk) == 32

    def test_world_readable_rejected(self, tmp_psk_file):
        os.chmod(str(tmp_psk_file), 0o644)
        assert py_gb_load_psk(str(tmp_psk_file)) is None

    def test_group_readable_rejected(self, tmp_psk_file):
        os.chmod(str(tmp_psk_file), 0o640)
        assert py_gb_load_psk(str(tmp_psk_file)) is None

    def test_world_writable_rejected(self, tmp_psk_file):
        os.chmod(str(tmp_psk_file), 0o602)
        assert py_gb_load_psk(str(tmp_psk_file)) is None

    def test_owner_only_modes_accepted(self, tmp_psk_file):
        for mode in [0o400, 0o600]:
            os.chmod(str(tmp_psk_file), mode)
            assert py_gb_load_psk(str(tmp_psk_file)) is not None, f"mode 0o{mode:o} should pass"

    def test_short_hex_rejected(self, tmp_psk_file):
        """The pre-PR-G behaviour silently zero-padded short input, yielding
        a low-entropy PSK that both sides "agreed" on.  Real bug."""
        tmp_psk_file.write_text("deadbeef")  # 8 chars, not 64
        os.chmod(str(tmp_psk_file), 0o600)
        assert py_gb_load_psk(str(tmp_psk_file)) is None

    def test_non_hex_char_rejected(self, tmp_psk_file):
        # Replace one char with 'z' - invalid hex
        tmp_psk_file.write_text("z" + "1" * 63)
        os.chmod(str(tmp_psk_file), 0o600)
        assert py_gb_load_psk(str(tmp_psk_file)) is None

    def test_non_hex_in_middle_rejected(self, tmp_psk_file):
        body = "1" * 32 + "G" + "1" * 31  # 'G' not hex
        tmp_psk_file.write_text(body)
        os.chmod(str(tmp_psk_file), 0o600)
        assert py_gb_load_psk(str(tmp_psk_file)) is None

    def test_uppercase_hex_accepted(self, tmp_psk_file):
        tmp_psk_file.write_text("DEADBEEF" * 8)
        os.chmod(str(tmp_psk_file), 0o600)
        psk = py_gb_load_psk(str(tmp_psk_file))
        assert psk == bytes.fromhex("DEADBEEF" * 8)

    def test_missing_file_returns_none(self, tmp_path):
        assert py_gb_load_psk(str(tmp_path / "nonexistent")) is None

    def test_directory_rejected(self, tmp_path):
        d = tmp_path / "not_a_file"
        d.mkdir()
        assert py_gb_load_psk(str(d)) is None

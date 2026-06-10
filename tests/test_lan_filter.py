# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""
is_lan_ip regression tests.

Catches:
  - 127/8 accepted as LAN (PR-G fix - was accepting localhost as LAN,
    which let a userspace proxy on the same host bypass the LAN filter)
  - RFC1918 ranges accepted (must still work)
  - Public IPs rejected
"""
import pytest


def py_is_lan_ip(ip_str, loopback_allowed=False):
    """Pure-Python model of is_lan_ip(ip_net) after PR-G.

    Returns True if `ip_str` is in a LAN range we'd accept connections
    from, False otherwise.  `loopback_allowed` toggles the
    GREENBOOST_LAN_INCLUDE_LOOPBACK opt-in.
    """
    parts = [int(x) for x in ip_str.split(".")]
    if len(parts) != 4 or not all(0 <= p <= 255 for p in parts):
        return False
    ip = (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3]

    if (ip & 0xFF000000) == 0x0A000000: return True   # 10/8
    if (ip & 0xFFF00000) == 0xAC100000: return True   # 172.16/12
    if (ip & 0xFFFF0000) == 0xC0A80000: return True   # 192.168/16
    if (ip & 0xFFFF0000) == 0xA9FE0000: return True   # 169.254/16 link-local
    if loopback_allowed and (ip & 0xFF000000) == 0x7F000000: return True
    return False


class TestLANFilter:

    @pytest.mark.parametrize("ip", [
        "10.0.0.1", "10.255.255.254", "10.42.42.42",
        "172.16.0.1", "172.20.5.5", "172.31.255.254",
        "192.168.0.1", "192.168.1.100", "192.168.255.254",
        "169.254.1.1", "169.254.169.254",
    ])
    def test_rfc1918_and_link_local_accepted(self, ip):
        assert py_is_lan_ip(ip)

    @pytest.mark.parametrize("ip", [
        "1.1.1.1",          # public
        "8.8.8.8",          # public
        "203.0.113.5",      # TEST-NET-3
        "172.15.0.1",       # outside 172.16/12
        "172.32.0.1",       # outside 172.16/12
        "169.253.1.1",      # outside 169.254/16
        "192.169.0.1",      # outside 192.168/16
    ])
    def test_public_ips_rejected(self, ip):
        assert not py_is_lan_ip(ip)

    def test_loopback_rejected_by_default(self):
        """PR-G key fix: 127/8 must NOT be classified as LAN by default,
        because a userspace proxy listening on 127.0.0.1 could otherwise
        bypass the LAN filter just by relaying the connection."""
        assert not py_is_lan_ip("127.0.0.1")
        assert not py_is_lan_ip("127.255.255.254")

    def test_loopback_opt_in_works(self):
        """For single-host dev/test, GREENBOOST_LAN_INCLUDE_LOOPBACK=1
        reinstates the old behaviour."""
        assert py_is_lan_ip("127.0.0.1", loopback_allowed=True)
        assert py_is_lan_ip("127.0.0.42", loopback_allowed=True)

    def test_loopback_opt_in_does_not_widen_other_ranges(self):
        """Opting into loopback must not accidentally widen the filter."""
        assert not py_is_lan_ip("8.8.8.8", loopback_allowed=True)
        assert not py_is_lan_ip("172.15.0.1", loopback_allowed=True)

    def test_edge_172_range_inclusive(self):
        """172.16.0.0 and 172.31.255.255 are the bounds of RFC1918 172.16/12."""
        assert py_is_lan_ip("172.16.0.0")
        assert py_is_lan_ip("172.31.255.255")

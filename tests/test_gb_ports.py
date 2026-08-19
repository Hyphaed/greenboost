#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""Tests for gb_ports central port registry."""

import os
import sys
from pathlib import Path

import pytest

_REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_DIR))


class TestParsePort:
    """Tests for parse_port() function."""

    def test_parse_port_valid_unset(self, monkeypatch):
        """When env var is unset, return fallback."""
        monkeypatch.delenv("GB_TEST_PORT_UNSET", raising=False)
        from gb_ports import parse_port
        assert parse_port("GB_TEST_PORT_UNSET", 9999) == 9999

    def test_parse_port_valid_empty(self, monkeypatch):
        """When env var is empty string, return fallback."""
        monkeypatch.setenv("GB_TEST_PORT_EMPTY", "")
        from gb_ports import parse_port
        assert parse_port("GB_TEST_PORT_EMPTY", 9999) == 9999

    def test_parse_port_valid_numeric(self, monkeypatch):
        """When env var is a valid port, return it."""
        monkeypatch.setenv("GB_TEST_PORT_VALID", "11369")
        from gb_ports import parse_port
        assert parse_port("GB_TEST_PORT_VALID", 9999) == 11369

    def test_parse_port_valid_with_spaces(self, monkeypatch):
        """Whitespace around the port is trimmed."""
        monkeypatch.setenv("GB_TEST_PORT_SPACES", "  8790  ")
        from gb_ports import parse_port
        assert parse_port("GB_TEST_PORT_SPACES", 9999) == 8790

    def test_parse_port_invalid_non_numeric(self, monkeypatch):
        """Non-numeric value raises clear ValueError."""
        monkeypatch.setenv("GB_TEST_PORT_BAD", "abc")
        from gb_ports import parse_port
        with pytest.raises(ValueError) as exc_info:
            parse_port("GB_TEST_PORT_BAD", 9999)
        assert "must be an integer" in str(exc_info.value)
        assert "GB_TEST_PORT_BAD" in str(exc_info.value)

    def test_parse_port_invalid_out_of_range_low(self, monkeypatch):
        """Port < 1024 raises clear ValueError."""
        monkeypatch.setenv("GB_TEST_PORT_LOW", "80")
        from gb_ports import parse_port
        with pytest.raises(ValueError) as exc_info:
            parse_port("GB_TEST_PORT_LOW", 9999)
        assert "1024 and 65535" in str(exc_info.value)
        assert "80" in str(exc_info.value)

    def test_parse_port_invalid_out_of_range_high(self, monkeypatch):
        """Port > 65535 raises clear ValueError."""
        monkeypatch.setenv("GB_TEST_PORT_HIGH", "70000")
        from gb_ports import parse_port
        with pytest.raises(ValueError) as exc_info:
            parse_port("GB_TEST_PORT_HIGH", 9999)
        assert "1024 and 65535" in str(exc_info.value)
        assert "70000" in str(exc_info.value)

    def test_parse_port_boundary_low(self, monkeypatch):
        """Port 1024 is valid (lower boundary)."""
        monkeypatch.setenv("GB_TEST_PORT_BOUNDARY_LOW", "1024")
        from gb_ports import parse_port
        assert parse_port("GB_TEST_PORT_BOUNDARY_LOW", 9999) == 1024

    def test_parse_port_boundary_high(self, monkeypatch):
        """Port 65535 is valid (upper boundary)."""
        monkeypatch.setenv("GB_TEST_PORT_BOUNDARY_HIGH", "65535")
        from gb_ports import parse_port
        assert parse_port("GB_TEST_PORT_BOUNDARY_HIGH", 9999) == 65535


class TestParsePortFromBind:
    """Tests for parse_port_from_bind() helper."""

    def test_parse_bind_valid(self):
        """Extract port from valid bind string."""
        from gb_ports import parse_port_from_bind
        assert parse_port_from_bind("127.0.0.1:8790") == 8790

    def test_parse_bind_valid_any_interface(self):
        """Extract port from 0.0.0.0 bind."""
        from gb_ports import parse_port_from_bind
        assert parse_port_from_bind("0.0.0.0:8790") == 8790

    def test_parse_bind_valid_implicit_localhost(self):
        """Extract port when host is missing."""
        from gb_ports import parse_port_from_bind
        assert parse_port_from_bind(":8790") == 8790

    def test_parse_bind_invalid_no_port(self):
        """Missing port in bind string raises ValueError."""
        from gb_ports import parse_port_from_bind
        with pytest.raises(ValueError) as exc_info:
            parse_port_from_bind("127.0.0.1:")
        assert "must include a port" in str(exc_info.value)

    def test_parse_bind_invalid_non_numeric_port(self):
        """Non-numeric port in bind string raises ValueError."""
        from gb_ports import parse_port_from_bind
        with pytest.raises(ValueError) as exc_info:
            parse_port_from_bind("127.0.0.1:abc")
        assert "not numeric" in str(exc_info.value)

    def test_parse_bind_invalid_out_of_range(self):
        """Out-of-range port in bind string raises ValueError."""
        from gb_ports import parse_port_from_bind
        with pytest.raises(ValueError) as exc_info:
            parse_port_from_bind("127.0.0.1:80")
        assert "1024-65535" in str(exc_info.value)


class TestValidateNoCollisions:
    """Tests for validate_no_collisions() function."""

    def test_no_collisions(self):
        """When all ports are unique, return empty list."""
        from gb_ports import validate_no_collisions
        ports = {
            "service_a": 9740,
            "service_b": 8790,
            "service_c": 8799,
        }
        assert validate_no_collisions(ports) == []

    def test_single_collision(self):
        """When two services share a port, report the pair."""
        from gb_ports import validate_no_collisions
        ports = {
            "service_a": 9740,
            "service_b": 9740,
            "service_c": 8799,
        }
        collisions = validate_no_collisions(ports)
        assert len(collisions) == 1
        label_a, label_b, port = collisions[0]
        assert {label_a, label_b} == {"service_a", "service_b"}
        assert port == 9740

    def test_multiple_collisions_same_port(self):
        """When three services share a port, report all pairs."""
        from gb_ports import validate_no_collisions
        ports = {
            "service_a": 9740,
            "service_b": 9740,
            "service_c": 9740,
        }
        collisions = validate_no_collisions(ports)
        assert len(collisions) == 3  # 3 choose 2
        port_values = [port for _, _, port in collisions]
        assert all(p == 9740 for p in port_values)

    def test_multiple_collision_groups(self):
        """When multiple pairs collide (different ports), report all."""
        from gb_ports import validate_no_collisions
        ports = {
            "a": 9740,
            "b": 9740,
            "c": 8790,
            "d": 8790,
        }
        collisions = validate_no_collisions(ports)
        assert len(collisions) == 2  # one pair at 9740, one at 8790
        collision_ports = {port for _, _, port in collisions}
        assert collision_ports == {9740, 8790}


class TestModuleConstants:
    """Tests for module-level port constants."""

    def test_netd_port_constant(self):
        """NETD_PORT is 9740."""
        from gb_ports import NETD_PORT
        assert NETD_PORT == 9740

    def test_dataflux_ui_port_constant(self):
        """DATAFLUX_UI_PORT is 8799."""
        from gb_ports import DATAFLUX_UI_PORT
        assert DATAFLUX_UI_PORT == 8799

    def test_synapse_port_default(self, monkeypatch):
        """SYNAPSE_PORT defaults to 11369 when env var unset."""
        monkeypatch.delenv("GB_SYNAPSE_PORT", raising=False)
        # Reimport to pick up the env change
        import importlib
        import gb_ports
        importlib.reload(gb_ports)
        assert gb_ports.SYNAPSE_PORT == 11369

    def test_exporter_port_default(self, monkeypatch):
        """EXPORTER_PORT defaults to 9742 when env var unset."""
        monkeypatch.delenv("GREENBOOST_EXPORTER_PORT", raising=False)
        import importlib
        import gb_ports
        importlib.reload(gb_ports)
        assert gb_ports.EXPORTER_PORT == 9742

    def test_all_ports_dict(self):
        """ALL_PORTS dict contains all service ports."""
        from gb_ports import ALL_PORTS
        assert "netd" in ALL_PORTS
        assert "a2a" in ALL_PORTS
        assert "dataflux_ui" in ALL_PORTS
        assert "exporter" in ALL_PORTS
        assert "synapse" in ALL_PORTS
        assert "peer_worker" in ALL_PORTS
        # Should have exactly these 6 entries
        assert len(ALL_PORTS) == 6


class TestModuleImportFailsOnBadEnvVar:
    """Test that import raises if env vars are invalid at module load time."""

    def test_import_fails_on_bad_synapse_port(self, monkeypatch):
        """If GB_SYNAPSE_PORT is invalid, module import fails (early)."""
        monkeypatch.setenv("GB_SYNAPSE_PORT", "not_a_number")
        # Attempting to import should raise SystemExit
        import importlib
        import sys
        import gb_ports

        # Force a reload to pick up the bad env var
        with pytest.raises(SystemExit) as exc_info:
            importlib.reload(gb_ports)
        assert exc_info.value.code == 1

    def test_import_fails_on_bad_exporter_port(self, monkeypatch):
        """If GREENBOOST_EXPORTER_PORT is invalid, module import fails."""
        monkeypatch.setenv("GREENBOOST_EXPORTER_PORT", "99999")  # out of range
        import importlib
        import gb_ports

        with pytest.raises(SystemExit) as exc_info:
            importlib.reload(gb_ports)
        assert exc_info.value.code == 1

    def test_import_fails_on_bad_a2a_bind(self, monkeypatch):
        """If GB_A2A_BIND has invalid port, module import fails."""
        monkeypatch.setenv("GB_A2A_BIND", "127.0.0.1:not_a_number")
        import importlib
        import gb_ports

        with pytest.raises(SystemExit) as exc_info:
            importlib.reload(gb_ports)
        assert exc_info.value.code == 1

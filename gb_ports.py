#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""gb_ports.py — Central registry for GreenBoost service ports.

All service ports (netd fabric, A2A gateway, dataflux UI, Prometheus exporter,
gb-synapse inference) are defined here as single sources of truth. This prevents
silent port collisions and ensures consistent validation across the codebase.

Every port is validated at module import time (1024-65535 range, numeric). A bad
env var will raise a clear ValueError immediately, never a cryptic bare traceback.

Port registry:
    NETD_PORT         = 9740  (greenboost-netd cluster fabric daemon)
    A2A_PORT          = 8790  (gb-synapse A2A agent gateway)
    DATAFLUX_UI_PORT  = 8799  (gb-dataflux web UI)
    EXPORTER_PORT     = 9742  (Prometheus exporter)
    SYNAPSE_PORT      = 11369 (gb-synapse inference proxy)
    PEER_WORKER_PORT  = 9741  (greenboost-cli peer worker)

Env var usage:
    GREENBOOST_NETD_PORT         (no override; always 9740)
    GB_A2A_BIND                  (format: "host:port", parsed by gb_a2a.py)
    GB_DATAFLUX_UI_PORT          (default: 8799)
    GREENBOOST_EXPORTER_PORT     (default: 9742)
    GB_SYNAPSE_PORT              (default: 11369)
    GB_PEER_WORKER_PORT          (default: 9741)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_DIR))


def parse_port(env_var: str, fallback: int) -> int:
    """Read env_var as a validated TCP port (1024-65535), or return fallback.

    Raises ValueError with a clear message if the env var is set but invalid
    (non-numeric, out of range). Never lets a bad env var surface as a bare
    traceback — callers get a message they can act on.

    Args:
        env_var: Environment variable name to check.
        fallback: Default port if the env var is unset or empty.

    Returns:
        The port number (validated in range).

    Raises:
        ValueError: If env_var is set but not a valid port (1024-65535).
    """
    raw = os.environ.get(env_var, "")
    if not raw or raw.strip() == "":
        return fallback

    trimmed = str(raw).strip()
    if not trimmed.isdigit():
        raise ValueError(
            f"Invalid port: {env_var}=\"{raw}\" — must be an integer "
            f"between 1024 and 65535 (got non-numeric value)"
        )

    parsed = int(trimmed)
    if parsed < 1024 or parsed > 65535:
        raise ValueError(
            f"Invalid port: {env_var}=\"{raw}\" — must be an integer "
            f"between 1024 and 65535 (got {parsed})"
        )

    return parsed


def parse_port_from_bind(bind_string: str) -> int:
    """Extract and validate the port number from a "host:port" bind string.

    Used by gb_a2a.py which uses GB_A2A_BIND as a full bind address string,
    not just a port number.

    Args:
        bind_string: String in format "host:port" or ":port" or "port".

    Returns:
        The port number (validated in range).

    Raises:
        ValueError: If the port component is invalid.
    """
    # Split on the LAST colon to handle IPv6 (though gb_a2a doesn't use it)
    host, _, port_str = bind_string.rpartition(":")

    if not port_str or not port_str.strip():
        raise ValueError(
            f"Invalid bind string: \"{bind_string}\" — must include a port "
            f"(e.g., \"127.0.0.1:8790\" or \":8790\")"
        )

    trimmed = str(port_str).strip()
    if not trimmed.isdigit():
        raise ValueError(
            f"Invalid port in bind string: \"{bind_string}\" — port "
            f"\"{port_str}\" is not numeric"
        )

    parsed = int(trimmed)
    if parsed < 1024 or parsed > 65535:
        raise ValueError(
            f"Invalid port in bind string: \"{bind_string}\" — port "
            f"{parsed} is out of range 1024-65535"
        )

    return parsed


def validate_no_collisions(ports: dict[str, int]) -> list[tuple[str, str, int]]:
    """Check for port collisions in a {label: port} mapping.

    Args:
        ports: Dictionary mapping service names to port numbers.

    Returns:
        List of colliding triples (label_a, label_b, port), empty if no collisions.
        Each pair is reported once; order of label_a and label_b is unspecified.
    """
    collisions: list[tuple[str, str, int]] = []
    port_to_labels: dict[int, list[str]] = {}

    for label, port in ports.items():
        if port not in port_to_labels:
            port_to_labels[port] = []
        port_to_labels[port].append(label)

    for port, labels in port_to_labels.items():
        if len(labels) > 1:
            # Report all pairs for this port
            for i, label_a in enumerate(labels):
                for label_b in labels[i + 1 :]:
                    collisions.append((label_a, label_b, port))

    return collisions


# ── Module-level constants (computed at import time) ──────────────────────

try:
    # netd: cluster fabric daemon (no env override; always 9740)
    NETD_PORT = 9740

    # A2A: agent-to-agent gateway
    # Note: GB_A2A_BIND is a full bind string (host:port), not a bare port.
    # This constant is the DEFAULT port extracted from the default bind string.
    # gb_a2a.py parses the full string itself.
    A2A_PORT = parse_port_from_bind(os.environ.get("GB_A2A_BIND", "127.0.0.1:8790"))

    # dataflux UI (no env override; always 8799)
    DATAFLUX_UI_PORT = 8799

    # Prometheus exporter
    EXPORTER_PORT = parse_port("GREENBOOST_EXPORTER_PORT", 9742)

    # gb-synapse inference proxy
    SYNAPSE_PORT = parse_port("GB_SYNAPSE_PORT", 11369)

    # greenboost-cli peer worker (used by cluster/peer_worker.py)
    PEER_WORKER_PORT = parse_port("GB_PEER_WORKER_PORT", 9741)

except ValueError as e:
    # A bad env var at import time makes the entire module unavailable.
    # This is intentional: we want the error visible IMMEDIATELY, not hidden
    # until some later function call tries to use the port.
    sys.stderr.write(f"[gb_ports] FATAL: {e}\n")
    sys.exit(1)


# Convenience: all ports as a dict for collision checking
ALL_PORTS = {
    "netd": NETD_PORT,
    "a2a": A2A_PORT,
    "dataflux_ui": DATAFLUX_UI_PORT,
    "exporter": EXPORTER_PORT,
    "synapse": SYNAPSE_PORT,
    "peer_worker": PEER_WORKER_PORT,
}


def main() -> None:
    """CLI tool: print all ports or check for collisions."""
    import json

    print("GreenBoost Service Ports")
    print("=" * 60)
    for service, port in ALL_PORTS.items():
        print(f"  {service:20s} : {port}")

    collisions = validate_no_collisions(ALL_PORTS)
    if collisions:
        print("\nCOLLISIONS DETECTED:")
        for label_a, label_b, port in collisions:
            print(f"  {label_a} and {label_b} both use port {port}")
        sys.exit(1)
    else:
        print("\nNo collisions detected. All ports are unique.")
        sys.exit(0)


if __name__ == "__main__":
    main()

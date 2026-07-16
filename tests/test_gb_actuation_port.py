#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Test gb_actuation.serve_and_repoint defaults its port to gb_synapse.DEFAULT_PORT
(port migration :11434 -> :11435, GB_SYNAPSE_PORT)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gb_actuation


def test_serve_and_repoint_dry_run_uses_default_port(monkeypatch):
    monkeypatch.delenv("GB_ORCH_ACTUATE", raising=False)
    plan = gb_actuation.serve_and_repoint("some/model")
    assert plan["gate"]["allowed"] is False
    assert plan["port"] == 11435
    assert "11435" in plan["dry_run"]
    assert "127.0.0.1:11435" in plan["forge_url_target"]


def test_serve_and_repoint_dry_run_explicit_port(monkeypatch):
    monkeypatch.delenv("GB_ORCH_ACTUATE", raising=False)
    plan = gb_actuation.serve_and_repoint("some/model", port=8080)
    assert plan["port"] == 8080

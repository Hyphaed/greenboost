#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Tests for gb_readiness.py — the whole-stack readiness-contract report
(NemoClaw audit, Phase 6b): mutated:false always, status<->exitCode
pinning, inconclusive as a genuine third state, and schema validation
against schemas/readiness.schema.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jsonschema
import pytest

import gb_readiness as gr

_SCHEMA = json.loads((Path(__file__).resolve().parent.parent / "schemas" / "readiness.schema.json").read_text())


def _validate(report):
    jsonschema.validate(report, _SCHEMA)


def test_build_report_validates_against_schema_live(monkeypatch):
    """The real, unmocked build_report() — whatever this box's actual
    state is — must always produce schema-valid output."""
    report = gr.build_report()
    _validate(report)


def test_mutated_is_always_false(monkeypatch):
    monkeypatch.setattr(gr, "_kmod_loaded", lambda: True)
    monkeypatch.setattr(gr, "_shim_present", lambda: True)
    monkeypatch.setattr(gr, "_port_listening", lambda port, **k: True)
    report = gr.build_report()
    assert report["mutated"] is False


def test_kmod_loaded_true_yields_supported_status():
    def _fake_port(port, **k):
        return True

    import unittest.mock
    with unittest.mock.patch.object(gr, "_kmod_loaded", return_value=True), \
         unittest.mock.patch.object(gr, "_shim_present", return_value=True), \
         unittest.mock.patch.object(gr, "_port_listening", side_effect=_fake_port):
        report = gr.build_report()
    assert report["status"] == "supported"
    assert report["exitCode"] == 0
    _validate(report)


def test_kmod_loaded_false_yields_incompatible_status_and_blocking_finding():
    import unittest.mock
    with unittest.mock.patch.object(gr, "_kmod_loaded", return_value=False), \
         unittest.mock.patch.object(gr, "_shim_present", return_value=False), \
         unittest.mock.patch.object(gr, "_port_listening", return_value=False):
        report = gr.build_report()
    assert report["status"] == "incompatible"
    assert report["exitCode"] == 2
    assert any(f["severity"] == "blocking" and f["id"] == "kmod_not_loaded"
               for f in report["findings"])
    _validate(report)


def test_kmod_undetermined_yields_inconclusive_not_a_guess():
    """When the observation itself can't be made, status must be
    'inconclusive' — never silently rounded to supported or incompatible."""
    import unittest.mock
    with unittest.mock.patch.object(gr, "_kmod_loaded", return_value=None), \
         unittest.mock.patch.object(gr, "_shim_present", return_value=None), \
         unittest.mock.patch.object(gr, "_port_listening", return_value=None):
        report = gr.build_report()
    assert report["status"] == "inconclusive"
    assert report["exitCode"] == 3
    kmod_obs = next(o for o in report["observations"] if o["id"] == "kmod.greenboost.loaded")
    assert kmod_obs["determined"] is False
    assert kmod_obs["value"] is None
    _validate(report)


def test_capabilities_reflect_actuate_env(monkeypatch):
    import unittest.mock
    monkeypatch.setenv("GB_ORCH_ACTUATE", "1")
    with unittest.mock.patch.object(gr, "_kmod_loaded", return_value=True), \
         unittest.mock.patch.object(gr, "_shim_present", return_value=True), \
         unittest.mock.patch.object(gr, "_port_listening", return_value=True):
        report = gr.build_report()
    cap = next(c for c in report["capabilities"] if c["id"] == "capability.tier_actuate")
    assert cap["available"] is True


def test_rpc_split_capability_requires_kmod():
    import unittest.mock
    with unittest.mock.patch.object(gr, "_kmod_loaded", return_value=False), \
         unittest.mock.patch.object(gr, "_shim_present", return_value=False), \
         unittest.mock.patch.object(gr, "_port_listening", return_value=False):
        report = gr.build_report()
    cap = next(c for c in report["capabilities"] if c["id"] == "capability.rpc_split")
    assert cap["available"] is False
    assert cap["reason"]


def test_port_listening_socket_error_is_undetermined_not_a_crash(monkeypatch):
    import socket as socket_mod

    class _RaisingSocket:
        def __init__(self, *a, **k):
            raise OSError("sandboxed: no socket access")

    monkeypatch.setattr(socket_mod, "socket", _RaisingSocket)
    assert gr._port_listening(9740) is None


def test_main_json_flag_prints_valid_json(capsys):
    rc = gr.main(["doctor", "--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    _validate(report)
    assert rc == report["exitCode"]


def test_main_unknown_verb_returns_1(capsys):
    rc = gr.main(["not-a-real-verb"])
    assert rc == 1


def test_status_exit_code_mapping_is_exhaustive():
    assert gr._STATUS_EXIT_CODE == {"supported": 0, "incompatible": 2, "inconclusive": 3}


# Reference validation tests
def test_reference_errors_detects_duplicate_ids_in_observations():
    """_reference_errors should catch duplicate ids within a collection."""
    report = {
        "observations": [
            {"id": "test.dup", "value": True, "determined": True},
            {"id": "test.dup", "value": False, "determined": True},
        ],
        "capabilities": [],
        "qualifications": [],
        "findings": [],
        "evidence": [],
    }
    errors = gr._reference_errors(report)
    assert len(errors) == 1
    assert "duplicate id" in errors[0]
    assert "test.dup" in errors[0]


def test_reference_errors_detects_dangling_related_observation():
    """_reference_errors should catch findings referencing non-existent observations."""
    report = {
        "observations": [
            {"id": "obs.real", "value": True, "determined": True},
        ],
        "capabilities": [],
        "qualifications": [],
        "findings": [
            {"id": "finding.test", "severity": "info", "message": "test",
             "relatedObservation": "obs.nonexistent"},
        ],
        "evidence": [],
    }
    errors = gr._reference_errors(report)
    assert len(errors) == 1
    assert "non-existent observation" in errors[0]
    assert "obs.nonexistent" in errors[0]


def test_reference_errors_returns_empty_on_clean_report():
    """_reference_errors should return empty list for a well-formed report."""
    report = {
        "observations": [
            {"id": "obs.one", "value": True, "determined": True},
            {"id": "obs.two", "value": False, "determined": True},
        ],
        "capabilities": [
            {"id": "cap.one", "available": True},
        ],
        "qualifications": [
            {"id": "qual.one", "passed": True},
        ],
        "findings": [
            {"id": "finding.one", "severity": "info", "message": "test",
             "relatedObservation": "obs.one"},
        ],
        "evidence": [
            {"id": "ev.one", "source": "test"},
        ],
    }
    errors = gr._reference_errors(report)
    assert errors == []


def test_build_report_passes_reference_validation():
    """The real build_report() output must always be internally consistent."""
    report = gr.build_report()
    errors = gr._reference_errors(report)
    assert errors == [], f"Reference validation failed: {errors}"


def test_qualifications_are_populated():
    """build_report() should populate qualifications, not leave it empty."""
    import unittest.mock
    with unittest.mock.patch.object(gr, "_kmod_loaded", return_value=True), \
         unittest.mock.patch.object(gr, "_shim_present", return_value=True), \
         unittest.mock.patch.object(gr, "_port_listening", return_value=True):
        report = gr.build_report()
    assert len(report["qualifications"]) > 0
    qual_ids = {q["id"] for q in report["qualifications"]}
    assert "qualification.reference_workload_servable" in qual_ids
    assert "qualification.cluster_rpc_split" in qual_ids
    assert "qualification.frontload_split_effective" in qual_ids


def test_qualification_reference_workload_servable_passes_when_qualified():
    """reference_workload_servable should pass when kmod and shim are both present."""
    import unittest.mock
    with unittest.mock.patch.object(gr, "_kmod_loaded", return_value=True), \
         unittest.mock.patch.object(gr, "_shim_present", return_value=True), \
         unittest.mock.patch.object(gr, "_port_listening", return_value=True):
        report = gr.build_report()
    qual = next(q for q in report["qualifications"]
                if q["id"] == "qualification.reference_workload_servable")
    assert qual["passed"] is True


def test_qualification_reference_workload_servable_fails_when_kmod_missing():
    """reference_workload_servable should fail when kmod is not loaded."""
    import unittest.mock
    with unittest.mock.patch.object(gr, "_kmod_loaded", return_value=False), \
         unittest.mock.patch.object(gr, "_shim_present", return_value=True), \
         unittest.mock.patch.object(gr, "_port_listening", return_value=False):
        report = gr.build_report()
    qual = next(q for q in report["qualifications"]
                if q["id"] == "qualification.reference_workload_servable")
    assert qual["passed"] is False


def test_qualification_cluster_rpc_split_requires_kmod():
    """cluster_rpc_split qualification should require kernel module."""
    import unittest.mock
    with unittest.mock.patch.object(gr, "_kmod_loaded", return_value=True), \
         unittest.mock.patch.object(gr, "_shim_present", return_value=True), \
         unittest.mock.patch.object(gr, "_port_listening", return_value=True):
        report = gr.build_report()
    qual = next(q for q in report["qualifications"]
                if q["id"] == "qualification.cluster_rpc_split")
    assert qual["passed"] is True

    # Now test with kmod missing
    with unittest.mock.patch.object(gr, "_kmod_loaded", return_value=False), \
         unittest.mock.patch.object(gr, "_shim_present", return_value=False), \
         unittest.mock.patch.object(gr, "_port_listening", return_value=False):
        report = gr.build_report()
    qual = next(q for q in report["qualifications"]
                if q["id"] == "qualification.cluster_rpc_split")
    assert qual["passed"] is False


def test_qualifications_validate_against_schema():
    """All qualifications from build_report() should validate against schema."""
    report = gr.build_report()
    _validate(report)
    # Specifically check qualifications are present and have expected structure
    for qual in report["qualifications"]:
        assert "id" in qual
        assert "passed" in qual
        assert isinstance(qual["passed"], bool)
        if "detail" in qual:
            assert isinstance(qual["detail"], str)


def test_main_detects_reference_errors():
    """main() should exit with code 1 if reference validation fails."""
    # Create a report with invalid references by mocking build_report
    def _bad_report(*args, **kwargs):
        return {
            "schemaVersion": "1",
            "status": "supported",
            "exitCode": 0,
            "mutated": False,
            "provenance": {"tool": "test", "generatedAtEpochS": 0, "node": "test"},
            "observations": [{"id": "obs.one", "value": True, "determined": True}],
            "capabilities": [],
            "qualifications": [],
            "findings": [
                {"id": "f1", "severity": "info", "message": "test",
                 "relatedObservation": "obs.nonexistent"}
            ],
            "evidence": [],
        }

    import unittest.mock
    with unittest.mock.patch.object(gr, "build_report", side_effect=_bad_report):
        rc = gr.main(["doctor"])
    assert rc == 1  # Should exit with code 1 (tool failure)

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""
Tests for gb_dataflux.stage() (P7) , the shared "emit stage_profile on
success, emit error + re-raise on failure" context manager, replacing 4
independent hand-rolled versions across ai-forge (build_jobs_exams.py's
df_emit/_vlm, run_all_exams.py's build_dir, forge/dataflux.py, etc.).
"""
import time

import pytest

import gb_dataflux as gdf


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    p = tmp_path / "dataflux.jsonl"
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(p))
    gdf._READ_EVENTS_MEMO.clear()
    yield p
    gdf._READ_EVENTS_MEMO.clear()


def test_stage_emits_ok_on_clean_exit(log_path):
    with gdf.stage("forge:image", label="gen_art"):
        pass

    events = [e for e in gdf.read_events() if e.get("kind") == "stage_profile"]
    assert len(events) == 1
    assert events[0]["stage"] == "forge:image"
    assert events[0]["status"] == "ok"
    assert events[0]["label"] == "gen_art"
    assert "duration_s" in events[0]


def test_stage_emits_error_and_reraises_on_exception(log_path):
    with pytest.raises(ValueError, match="boom"):
        with gdf.stage("forge:image"):
            raise ValueError("boom")

    events = [e for e in gdf.read_events() if e.get("kind") == "stage_profile"]
    assert len(events) == 1
    assert events[0]["status"] == "error"
    assert "boom" in events[0]["error"]


def test_stage_measures_real_duration(log_path):
    with gdf.stage("slow_stage"):
        time.sleep(0.05)

    events = [e for e in gdf.read_events() if e.get("kind") == "stage_profile"]
    assert events[0]["duration_s"] >= 0.05


def test_stage_error_truncated_to_500_chars(log_path):
    long_msg = "x" * 2000
    with pytest.raises(RuntimeError):
        with gdf.stage("s"):
            raise RuntimeError(long_msg)

    events = [e for e in gdf.read_events() if e.get("kind") == "stage_profile"]
    assert len(events[0]["error"]) <= 500


def test_stage_extra_fields_passed_through(log_path):
    with gdf.stage("s", model="qwen3", node="host"):
        pass

    events = [e for e in gdf.read_events() if e.get("kind") == "stage_profile"]
    assert events[0]["model"] == "qwen3"
    assert events[0]["node"] == "host"

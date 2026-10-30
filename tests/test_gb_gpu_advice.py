"""gb_gpu_advice + the MCP surface for engine/telemetry facts (round 2, 2026-10-30)."""
import json
from pathlib import Path

import gb_gpu_advice as ga

ROOT = Path(__file__).resolve().parent.parent
WANTED = {"attention_backend", "cnn_channels_last", "regional_compile", "expert_cache", "vision_reserve", "mtp_stacking", "power_limit", "vae_precision", "fp8_conv", "fp8_linear_prefill", "vram_budget"}


def test_all_learned_items_present_with_evidence():
    items = ga.advice((12, 0), 11.9)
    assert {i["id"] for i in items} == WANTED
    for i in items:
        assert i["confidence"] in ("measured", "derived")
        assert i["evidence"] and all((ROOT / e).is_file() for e in i["evidence"]), i["id"]


def test_applicability_follows_detected_gpu_not_a_name():
    blackwell = {i["id"]: i for i in ga.advice((12, 0), 11.9)}
    ampere = {i["id"]: i for i in ga.advice((8, 6), 24.0)}
    assert blackwell["attention_backend"]["applies"] and not ampere["attention_backend"]["applies"]
    assert blackwell["fp8_conv"]["applies"] and ampere["fp8_conv"]["applies"] is False       # sm_86 has no fp8 tensor cores
    assert blackwell["vram_budget"]["applies"] and not ampere["vram_budget"]["applies"]      # small-card rule only for <= 16 GB


def test_compact():
    assert len(json.dumps(ga.gpu_advice())) < 9000


def test_off_arch_numbers_are_flagged_for_remeasure():
    ampere = {i["id"]: i for i in ga.advice((8, 6), 24.0)}
    assert "re-measure" in ampere["attention_backend"]["numbers"]


def test_filters():
    out = ga.gpu_advice(ids=["expert_cache"])
    assert [i["id"] for i in out["advice"]] == ["expert_cache"]


def test_mcp_tool_registered_and_documented():
    text = (ROOT / "gb_mcp.py").read_text()
    assert "def gpu_advice(" in text
    advisor_doc = text.split("def quant_advisor")[1].split("def gpu_advice")[0]
    assert "gpu_advice" in advisor_doc
    assert "synapse.engine" in text.split("def greenboost_overview")[1].split("def flux_health")[0]

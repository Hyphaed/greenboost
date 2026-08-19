"""A recipe's model-card sampling values must reach llama-server.

Before this, a recipe could pin ctx and KV type but had no way to carry the
sampling values a model's own card publishes, so a card specifying
`top_k 20 / min_p 0.0` silently ran on llama.cpp's generic defaults
(`top_k 40 / min_p 0.05`). The values are half of "configure this model
properly"; pinning ctx without them only does the other half.
"""
import json
import pathlib

import pytest
import yaml

import gb_synapse_backends as B

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_RECIPE = _ROOT / "serving/recipes/qwen38-27b-ud-iq3-xxs.yaml"


def test_no_sampling_block_emits_no_flags():
    """A model without published values keeps llama.cpp's own defaults."""
    assert B._sampling_args(None) == []
    assert B._sampling_args({}) == []


def test_every_supported_key_maps_to_its_flag():
    got = B._sampling_args({
        "temperature": 1.0, "topP": 0.95, "topK": 20, "minP": 0.0,
        "presencePenalty": 0.0, "repeatPenalty": 1.1,
    })
    assert got == ["--temp", "1.0", "--top-p", "0.95", "--top-k", "20",
                   "--min-p", "0.0", "--presence-penalty", "0.0",
                   "--repeat-penalty", "1.1"]


def test_a_partial_block_pins_only_what_it_states():
    """Absent keys must not be invented from another model's values."""
    assert B._sampling_args({"topK": 20}) == ["--top-k", "20"]


def test_zero_is_emitted_not_treated_as_absent():
    """min_p 0.0 is a real instruction and differs from llama.cpp's 0.05."""
    assert "--min-p" in B._sampling_args({"minP": 0.0})


def test_source_is_not_emitted_as_a_flag():
    assert B._sampling_args({"source": "https://example", "topK": 20}) == ["--top-k", "20"]


def test_the_schema_accepts_the_sampling_block():
    schema = json.loads((_ROOT / "serving/recipe.schema.json").read_text())
    props = schema["properties"]["sampling"]["properties"]
    assert {"temperature", "topP", "topK", "minP",
            "presencePenalty", "repeatPenalty", "source"} <= set(props)
    assert schema["properties"]["sampling"]["additionalProperties"] is False


@pytest.mark.skipif(not _RECIPE.exists(), reason="reference recipe absent")
def test_the_reference_recipe_matches_the_model_card():
    """The values are the card's, and the recipe says where they came from."""
    r = yaml.safe_load(_RECIPE.read_text())
    s = r["sampling"]
    assert s["temperature"] == 1.0        # thinking mode
    assert s["topP"] == 0.95
    assert s["topK"] == 20
    assert s["minP"] == 0.0
    assert "huggingface.co" in s["source"]
    # And the fit that motivated the recipe.
    # 40960, not 16384: this box's 238 MCP tool schemas + system prompt are a
    # measured 29,507-token floor per request, so 16384 overflowed before the
    # conversation started. t2Spill because 2.66 GiB genuinely does not fit ,
    # the recipe must not claim a placement it does not achieve.
    assert r["ctx"] == 40960
    assert r["tierIntent"] == "t2Spill"
    assert r["kvCache"] == {"key": "f16", "value": "f16"}

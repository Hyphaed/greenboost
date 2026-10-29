"""recipes_unresolvable must not flag serving/recipes/example.yaml.

Observed live 2026-09-22: this metric read 1 permanently, on a box where
every real recipe resolved fine, because example.yaml is a documented
TEMPLATE (its own header: "safe by construction, not by policy") whose
model.name is a placeholder DESIGNED to never resolve. dataflux_critic then
surfaced it as a `recipe_model_unresolvable` violation , a real-looking but
false signal, the same shape as the boot_tuning_persisted quote-boundary bug
fixed alongside this one.
"""
from __future__ import annotations

import gb_semantics as gs


def test_example_template_is_excluded():
    result = gs._res_recipes_unresolvable(None, None)
    unresolvable = (result.get("evidence") or {}).get("unresolvable") or []
    assert not any(item["path"].endswith("example.yaml") for item in unresolvable), (
        f"example.yaml should never count as an unresolvable recipe: {unresolvable}"
    )

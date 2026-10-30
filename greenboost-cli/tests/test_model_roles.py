"""environment/models.yaml: the role -> local model catalog and its loader."""
from pathlib import Path

import yaml

from greenboost_cli.environment import models as m

_RECIPES = Path(__file__).resolve().parents[2] / "serving" / "recipes"


def test_catalog_loads_and_every_role_is_described():
    cat = m.load_catalog()
    assert cat["version"] == 1
    for name, role in cat["roles"].items():
        assert role.get("summary"), name
        assert role.get("model"), name
        assert role.get("status") in {"measured", "in_use", "unverified"}, name


def test_measured_roles_carry_numbers_and_what_is_missing():
    for name in ("agent_fast", "agent_deep", "vision", "vision_deep"):
        role = m.resolve_role(name)
        assert role["status"] == "measured"
        assert role["measured"]["decode_tok_s"] > 0
        assert role["use_when"] and role["avoid_when"] and role["unmeasured"]


def test_every_measured_role_has_numbers_and_every_unverified_one_has_none():
    for name, role in m.load_catalog()["roles"].items():
        if role["status"] == "measured":
            assert role["measured"]["decode_tok_s"] > 0, name
        if role["status"] == "unverified":
            assert "measured" not in role, name


def test_vision_roles_are_split_by_speed_and_depth():
    fast, deep = m.resolve_role("vision"), m.resolve_role("vision_deep")
    assert fast["model"] != deep["model"]
    assert "MTP off" in deep["measured"]["config"]


def test_escalation_names_are_real_roles_in_order():
    chain = m.escalation_chain()
    assert [e["role"] for e in chain] == ["agent_fast", "agent_deep"]


def test_recipes_named_by_roles_exist_in_the_repo():
    if not _RECIPES.is_dir():
        return                       # vendored without the parent repo
    for name, role in m.load_catalog()["roles"].items():
        if role.get("recipe"):
            assert (_RECIPES / f"{role['recipe']}.yaml").is_file(), name
            recipe = yaml.safe_load((_RECIPES / f"{role['recipe']}.yaml").read_text())
            assert recipe["model"]["name"] == role["model"] or name == "agent_deep"


def test_env_override_wins_and_is_reported():
    env = {"GB_MODEL_AGENT_FAST": "my-model"}
    role = m.resolve_role("agent_fast", environ=env)
    assert (role["model"], role["override"]) == ("my-model", True)
    assert m.default_agent_model(environ=env) == "my-model"
    assert m.resolve_role("agent_fast", environ={})["override"] is False


def test_missing_or_broken_catalog_never_raises(tmp_path):
    assert m.load_catalog(tmp_path / "nope.yaml") == {}
    bad = tmp_path / "bad.yaml"
    bad.write_text("roles: [unclosed\n  - a: b: c")
    assert m.load_catalog(bad) == {}
    assert m.resolve_role("x", catalog={}) == {}
    assert m.default_agent_model(catalog={}) == ""


def test_roles_command_is_registered():
    import greenboost_cli.slash_commands.model_cmds  # noqa: F401
    from greenboost_cli.terminal.commands import COMMAND_TABLE
    assert "roles" in COMMAND_TABLE

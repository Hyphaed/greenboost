"""The Skill instrument lets the AGENT invoke user-authored scripts.

Before this existed the skill router was reachable only by the human (/skill)
or the factory, so a user's own scripts could not participate in tool calling
at all. The token economics are the point: 238 MCP schemas cost ~7k prompt
tokens on every request, while a skill costs one line until it is invoked.
"""
import os
import stat
import textwrap

import pytest

from greenboost_cli.instruments import handlers as H


def _mk_skill(root, name, description, script=None, body="Do the thing."):
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {name}
        description: {description}
        ---

        {body}
        """))
    if script:
        fname, content = script
        p = d / fname
        p.write_text(content)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return d


@pytest.fixture
def skills(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    root.mkdir()
    _mk_skill(root, "procedure-skill", "A skill with no script.",
              body="Step 1. Step 2.")
    _mk_skill(root, "script-skill", "A skill that runs a script.",
              script=("run.sh", "#!/bin/bash\necho \"ran with: $*\"\n"))
    _mk_skill(root, "failing-skill", "A skill whose script fails.",
              script=("run.sh", "#!/bin/bash\necho 'boom' >&2\nexit 3\n"))

    def _fake(settings=None):
        from greenboost_cli.skill.router import discover_skills_multi
        return [root], discover_skills_multi([root])
    monkeypatch.setattr(H, "_skill_dirs_and_entries", _fake)
    return root


def test_script_skill_runs_and_returns_its_output(skills):
    out = H.handle_skill({"name": "script-skill", "args": "alpha beta"})
    assert "ran with: alpha beta" in out


def test_script_skill_without_args(skills):
    assert "ran with:" in H.handle_skill({"name": "script-skill"})


def test_procedure_skill_returns_the_body_to_follow(skills):
    out = H.handle_skill({"name": "procedure-skill"})
    assert "follow this procedure" in out
    assert "Step 1" in out


def test_failing_script_reports_exit_code_and_stderr(skills):
    out = H.handle_skill({"name": "failing-skill"})
    assert "exited 3" in out and "boom" in out


def test_unknown_skill_names_the_available_ones(skills):
    out = H.handle_skill({"name": "no-such-skill"})
    assert "no skill named" in out
    assert "script-skill" in out          # tells the model what it CAN call


def test_no_name_lists_skills(skills):
    out = H.handle_skill({})
    assert "script-skill" in out and "A skill that runs a script." in out


def test_name_matching_tolerates_case_and_punctuation(skills):
    assert "ran with:" in H.handle_skill({"name": "Script_Skill"})


def test_skill_is_registered_and_offered_to_the_model():
    from greenboost_cli.instruments.dispatcher import _DISPATCH
    from greenboost_cli.instruments.schemas import INSTRUMENT_DEFINITIONS
    assert "Skill" in _DISPATCH
    assert any(d["name"] == "Skill" for d in INSTRUMENT_DEFINITIONS)


def test_skill_is_not_concurrency_safe():
    """A skill runs arbitrary user code; it must never be auto-parallelised."""
    from greenboost_cli.instruments.concurrency import is_concurrency_safe
    assert is_concurrency_safe("Skill", {"name": "anything"}) is False

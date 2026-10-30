"""`/llamaserve load` is what the startup banner tells people to type."""
import inspect

from greenboost_cli.slash_commands import backend_cmds
from greenboost_cli.terminal import renderer


def test_load_is_an_alias_for_start_not_a_model_name():
    src = inspect.getsource(backend_cmds.cmd_llamaserve)
    assert 'not in ("start", "load")' in src


def test_banner_tells_the_user_how_to_load_when_stopped():
    assert "use '/llamaserve load' to load the model" in inspect.getsource(renderer)

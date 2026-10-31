from types import SimpleNamespace as NS
from unittest import mock

from greenboost_cli.terminal import commands as C

GB = 1024 ** 3


def _run(entries, name):
    out = []
    mon = mock.Mock(); mon.refresh.return_value = NS(vram_physical_mb=11000)
    gs = mock.Mock(); gs.list_models.return_value = entries
    with mock.patch("greenboost_cli.greenboost.monitor.get_monitor", return_value=mon), \
         mock.patch("greenboost_cli.slash_commands.backend_cmds._import_gb_synapse", return_value=gs), \
         mock.patch.object(C, "emit_warn", lambda m, *a, **k: out.append(("warn", m))), \
         mock.patch.object(C, "emit_info", lambda m, *a, **k: out.append(("info", m))):
        C._warn_if_model_exceeds_vram(name)
    return out


def test_moe_model_larger_than_vram_gets_an_explanation_not_a_spill_warning():
    moe = NS(name="m", n_bytes=int(13.6 * GB), is_moe=True, dense_bytes=int(1.5 * GB), arch="qwen35moe")
    out = _run([moe], "m")
    assert [k for k, _ in out] == ["info"] and "intended fast setup" in out[0][1]


def test_dense_model_larger_than_vram_still_warns_and_never_suggests_an_embedder():
    dense = NS(name="big", n_bytes=int(20 * GB), is_moe=False, dense_bytes=0, arch="qwen3")
    emb = NS(name="qwen3-embedding:latest", n_bytes=int(4.4 * GB), is_moe=False, dense_bytes=0, arch="qwen3")
    small = NS(name="small", n_bytes=int(6 * GB), is_moe=False, dense_bytes=0, arch="qwen3")
    out = _run([dense, emb, small], "big")
    assert out[0][0] == "warn" and "small" in out[1][1] and "embedding" not in out[1][1]

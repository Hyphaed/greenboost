"""gb_cpu_pin: pure topology -> llama.cpp pinning flags (2026-10-30). Off by default."""
import gb_cpu_pin as g

HYBRID = {"online": list(range(32)), "p_cpus": list(range(16)), "e_cpus": list(range(16, 32)), "capacity": {},
          "siblings": {**{c: [c - c % 2, c - c % 2 + 1] for c in range(16)}, **{c: [c] for c in range(16, 32)}}}


def test_parse_cpulist():
    assert g.parse_cpulist("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11] and g.parse_cpulist("") == []


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("GB_SYNAPSE_CPU_PIN", raising=False)
    assert g.cpu_pin_args(HYBRID) == [] and g.mode() == "0"
    assert g.cpu_pin_args(HYBRID, "garbage") == []


def test_hybrid_one_thread_per_physical_p_core():
    assert g.cpu_pin_args(HYBRID, "1") == ["-t", "8", "-C", "0x5555", "--cpu-strict", "1"]
    assert g.cpu_pin_args(HYBRID, "auto") == g.cpu_pin_args(HYBRID, "1")


def test_auto_needs_hybrid_but_1_does_not():
    flat = {"online": list(range(8)), "p_cpus": [], "e_cpus": [], "capacity": {}, "siblings": {c: [c] for c in range(8)}}
    assert g.cpu_pin_args(flat, "auto") == []
    assert g.cpu_pin_args(flat, "1")[:2] == ["-t", "8"]


def test_capacity_fallback_picks_highest():
    t = {"online": [0, 1, 2, 3], "p_cpus": [], "e_cpus": [2, 3], "capacity": {0: 1024, 1: 1024, 2: 512, 3: 512}, "siblings": {}}
    assert g.pcore_cpus(t) == [0, 1] and g.cpu_pin_args(t, "1") == ["-t", "2", "-C", "0x3", "--cpu-strict", "1"]


def test_smt_siblings_use_lowest_only_and_offline_ignored():
    assert g.one_per_core([0, 1, 2, 3], {0: [0, 1], 1: [0, 1], 2: [2, 3], 3: [2, 3]}) == [0, 2]
    t = {**HYBRID, "online": [c for c in range(32) if c != 2]}
    assert 2 not in g.pcore_cpus(t)


def test_single_core_is_not_pinned():
    assert g.cpu_pin_args({"online": [0], "p_cpus": [], "e_cpus": [], "capacity": {}, "siblings": {}}, "1") == []


def test_apply_to_cmd_rewrites_threads_and_respects_existing_flags():
    cmd = ["llama-server", "--threads", "16", "-ngl", "99"]
    assert g.apply_to_cmd(cmd, HYBRID, "1")
    assert cmd[cmd.index("--threads") + 1] == "8" and cmd[-4:] == ["-C", "0x5555", "--cpu-strict", "1"]
    pinned = ["x", "--threads", "4", "-C", "0xf"]
    assert not g.apply_to_cmd(pinned, HYBRID, "1") and pinned[2] == "4"
    off = ["x", "--threads", "16"]
    assert not g.apply_to_cmd(off, HYBRID, "0") and off == ["x", "--threads", "16"]


def test_wired_into_serve_behind_env():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent / "gb_synapse_backends.py").read_text()
    assert "gb_cpu_pin.apply_to_cmd(cmd)" in src

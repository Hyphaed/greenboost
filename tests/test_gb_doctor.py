"""gb_doctor pure verdicts (round 2, 2026-10-30)."""
import gb_doctor as d

M = b"GB-MARKER cudart-rebind-force=1"


def test_engine_runtime():
    assert d.judge_engine_runtime("13", "13")["status"] == "ok"
    bad = d.judge_engine_runtime("12", "13")
    assert bad["status"] == "warn" and "GB_SYNAPSE_NVCC" in bad["fix"]
    assert d.judge_engine_runtime("", "13")["status"] == "info" and d.judge_engine_runtime("12", "13", engine_built=False)["status"] == "info"


def test_shim_marker_and_identical_build():
    assert d.judge_shim_rebind(b"xx" + M, M)["status"] == "ok"
    src = "static void gb_cudart_resolve_syms(void *libcudart, int force)"
    assert d.judge_shim_rebind(b"abc", M, b"abc", src)["status"] == "ok"
    w = d.judge_shim_rebind(b"abc", M, b"other", src)
    assert w["status"] == "warn" and "make shim" in w["fix"]
    assert d.judge_shim_rebind(None, M)["status"] == "info"


def test_engine_selected():
    assert d.judge_engine_selected("moecache", True, "moecache", True)["status"] == "ok"
    assert "use-engine moecache" in d.judge_engine_selected("vendored", True, "vendored", False)["fix"]
    assert "build-moecache-engine" in d.judge_engine_selected("vendored", False, "unset", False)["fix"]


def test_thp_and_alignment():
    r = {c["id"]: c for c in d.judge_thp("[always] madvise never", "always [defer] madvise", 2097152, 0, 43008)}
    assert r["thp_mode"]["status"] == "ok" and r["t2_pool_alignment"]["status"] == "ok"
    r = {c["id"]: c for c in d.judge_thp("always madvise [never]", "", 2097152, 4, 43007)}
    assert r["thp_mode"]["status"] == "warn" and r["hugetlb_pool"]["status"] == "warn" and r["t2_pool_alignment"]["status"] == "warn"


def test_helper():
    assert d.judge_helper(True, True, True)["status"] == "ok"
    assert d.judge_helper(True, False, True)["status"] == "warn"
    assert "--dry-run" in d.judge_helper(False, False, True)["fix"]


def test_cli_shadow():
    assert d.judge_cli_shadow(None, None, editable=True)["status"] == "ok"
    assert d.judge_cli_shadow({"a.py": "1"}, {"a.py": "1"})["status"] == "ok"
    w = d.judge_cli_shadow({"a.py": "1"}, {"a.py": "2", "b.py": "3"}, "/v", "/repo")
    assert w["status"] == "warn" and "pip install --no-deps -e /repo/greenboost-cli" in w["fix"] and "2 file" in w["detail"]


def test_tree_hashes(tmp_path):
    (tmp_path / "x.py").write_text("a"); (tmp_path / "__pycache__").mkdir(); (tmp_path / "__pycache__" / "y.py").write_text("b")
    assert list(d.tree_hashes(tmp_path)) == ["x.py"]

"""gb_moe_cache sizing (pure arithmetic; no GPU, no engine)."""
import gb_moe_cache as mc

KW = dict(n_layers=40, n_cpu_moe=30, dense_gb=1.54, expert_gb_total=12.2, kv_gb=0.1, vision_gb=0.0, budget_gb=8.2)


def test_the_measured_case_lands_near_the_measured_optimum(monkeypatch):
    monkeypatch.delenv("GB_MOE_CACHE", raising=False)
    p = mc.plan(**KW)
    assert p and p.n_cpu_moe == 40 and 3800 <= p.cache_mib <= 5200, p       # a 4 to 5 GiB cache is the sweet spot; 6.5 GiB was unstable; the safety margin grew in 2026-10-30 after an OOM


def test_vision_and_ubatch_shrink_the_cache(monkeypatch):
    monkeypatch.delenv("GB_MOE_CACHE", raising=False)
    base = mc.plan(**KW).cache_mib
    assert mc.plan(**{**KW, "vision_gb": 1.73}).cache_mib < base - 1500
    assert mc.plan(**{**KW, "ubatch": 2048}).cache_mib < base - 1000


def test_no_cache_when_there_is_no_room_or_nothing_is_on_the_host(monkeypatch):
    monkeypatch.delenv("GB_MOE_CACHE", raising=False)
    assert mc.plan(**{**KW, "budget_gb": 5.0}) is None                      # too little room to beat the current placement
    assert mc.plan(**{**KW, "n_cpu_moe": 0}) is None                        # the whole model fits: nothing to cache
    assert mc.plan(**{**KW, "n_layers": 0}) is None


def test_mtp_draft_context_takes_its_share(monkeypatch):
    monkeypatch.delenv("GB_MOE_CACHE", raising=False)
    assert mc.plan(**KW).cache_mib - mc.plan(**{**KW, "mtp": True}).cache_mib >= 500


def test_cache_never_exceeds_the_experts(monkeypatch):
    monkeypatch.delenv("GB_MOE_CACHE", raising=False)
    p = mc.plan(**{**KW, "budget_gb": 40.0, "expert_gb_total": 6.0})
    assert p.cache_mib == 6 * 1024


def test_env_switches(monkeypatch):
    monkeypatch.setenv("GB_MOE_CACHE", "off")
    assert mc.plan(**KW) is None
    monkeypatch.setenv("GB_MOE_CACHE", "2048")
    assert mc.plan(**KW).cache_mib == 2048
    monkeypatch.setenv("GB_MOE_CACHE", "auto"); monkeypatch.setenv("GB_MOE_CACHE_MIN_GB", "9")
    assert mc.plan(**KW) is None


def test_engine_probe_reads_the_help(tmp_path):
    good = tmp_path / "llama-server"
    good.write_text("#!/bin/sh\necho '--moe-cache-mib N   GPU cache'\n"); good.chmod(0o755)
    bad = tmp_path / "old-server"
    bad.write_text("#!/bin/sh\necho '--n-cpu-moe N'\n"); bad.chmod(0o755)
    assert mc.engine_supports(good) and not mc.engine_supports(bad) and not mc.engine_supports(tmp_path / "missing")


def test_compat_flags_renames_no_mmap_only_for_engines_that_dropped_it(tmp_path):
    new = tmp_path / "new-server"
    new.write_text("#!/bin/sh\necho '-lm, --load-mode MODE'\necho '--moe-cache-mib N'\n"); new.chmod(0o755)
    old = tmp_path / "old-server"
    old.write_text("#!/bin/sh\necho '--no-mmap'\necho '--load-mode'\n"); old.chmod(0o755)
    cmd = ["llama-server", "--no-mmap", "-c", "4096"]
    assert mc.compat_flags(cmd, new) == ["llama-server", "--load-mode", "none", "-c", "4096"]
    assert mc.compat_flags(cmd, old) == cmd and mc.compat_flags(cmd, tmp_path / "missing") == cmd and mc.compat_flags(["x"], new) == ["x"]


def test_mtp_reserve_matches_the_measured_draft_context(monkeypatch):
    """Measured 2026-10-30: the MTP draft context costs 1.5 GB (depth 2) to 2.0 GB (depth 4); 0.6 GB let the default depth-4 serve run out of memory at load."""
    monkeypatch.delenv("GB_MOE_CACHE", raising=False)
    gap = mc.plan(**KW).cache_mib - mc.plan(**{**KW, "mtp": True}).cache_mib
    assert 1450 <= gap <= 1600, gap
    assert mc.MTP_CONTEXT_GB >= 1.5


def test_depth_with_the_cache_is_two():
    assert mc.MTP_DRAFT_N == 2                                             # depth 2 = 181 tok/s, 3 = 160, 4 does not load (workflow/overnight-improvements.md, improvement 3)


def test_serve_keeps_mtp_and_sets_the_depth_when_the_cache_is_used():
    import inspect
    import gb_synapse_backends as be
    src = inspect.getsource(be)
    assert 'os.environ.get("GB_MOE_CACHE_MTP", "1") == "0"' in src and "min(mtp_draft_n if mtp_draft_n is not None else gs.MTP_DRAFT_N, _cap)" in src


# ---- round 2 telemetry (2026-10-30): moe_cache_plan event, critic rules -----------------------------------------------------------------
def _plan(mib):
    return mc.MoeCachePlan(mib, 40, "d")


def test_decision_event_ok():
    ev = mc.decision_event(model="m", plan_=_plan(5000), inputs={"budget_gb": 10.0, "kv_gb": 0.5}, engine={"kind": "moecache", "version": "0893ec9", "supports_cache": True},
                                     n_cpu_moe=30, is_moe=True, mtp=True, mtp_draft_n=2)
    assert ev["status"] == "ok" and ev["cache_mib"] == 5000 and ev["mtp"] and ev["mtp_draft_n"] == 2 and ev["engine_kind"] == "moecache" and ev["budget_gb"] == 10.0


def test_decision_event_warns_small_cache_and_vendored_engine():
    small = mc.decision_event(model="m", plan_=_plan(512), inputs={}, engine={"kind": "moecache", "supports_cache": True}, n_cpu_moe=30, is_moe=True, mtp=False, mtp_draft_n=0)
    assert small["status"] == "warn" and "512" in small["warning"]
    old = mc.decision_event(model="m", plan_=None, inputs={}, engine={"kind": "vendored", "supports_cache": False}, n_cpu_moe=30, is_moe=True, mtp=False, mtp_draft_n=0)
    assert old["status"] == "warn" and old["skip_reason"] == "engine_without_flag" and "build-moecache-engine" in old["warning"]
    dense = mc.decision_event(model="m", plan_=None, inputs={}, engine={"kind": "vendored", "supports_cache": False}, n_cpu_moe=0, is_moe=False, mtp=False, mtp_draft_n=0)
    assert dense["status"] == "ok" and dense["skip_reason"] == "no_experts_on_host"


def test_critic_names_the_fix(tmp_path, monkeypatch):
    import gb_dataflux
    old = mc.decision_event(model="m", plan_=None, inputs={}, engine={"kind": "vendored", "supports_cache": False}, n_cpu_moe=30, is_moe=True, mtp=False, mtp_draft_n=0)
    assert gb_dataflux._is_incident({"kind": "moe_cache_plan", **old})
    assert any("use-engine moecache" in h for h in gb_dataflux._diagnosis_hints({'kind': 'moe_cache_plan', **old}, None, None, {}, []))
    small = mc.decision_event(model="m", plan_=_plan(512), inputs={}, engine={"kind": "moecache", "supports_cache": True}, n_cpu_moe=30, is_moe=True, mtp=False, mtp_draft_n=0)
    assert any("under 1 GB" in h for h in gb_dataflux._diagnosis_hints({'kind': 'moe_cache_plan', **small}, None, None, {}, []))


def test_engine_info_kinds(tmp_path):
    d = tmp_path / "mc"; d.mkdir(); (d / "llama-server").write_text("x"); (d / "engine.version").write_text("abc123\n")
    info = mc.engine_info(d / "llama-server", d, tmp_path / "other")
    assert info["kind"] == "moecache" and info["version"] == "abc123" and info["supports_cache"] is False
    assert mc.engine_info(d / "llama-server", tmp_path / "x", d)["kind"] == "vendored"


def test_emit_decision_writes_the_event(tmp_path, monkeypatch):
    import gb_dataflux
    monkeypatch.setenv("GREENBOOST_DATAFLUX_LOG", str(tmp_path / "df.jsonl"))
    gb_dataflux._READ_EVENTS_MEMO.clear()
    mc.emit_decision(model="m", plan_=_plan(4000), inputs={"budget_gb": 9.0}, engine={"kind": "moecache", "supports_cache": True}, n_cpu_moe=30, is_moe=True, mtp=False, mtp_draft_n=0)
    ev = [e for e in gb_dataflux.read_events(since_hours=1) if e.get("kind") == "moe_cache_plan"]
    assert ev and ev[-1]["cache_mib"] == 4000


# ---- ubatch step-down (2026-10-30): a recipe's ubatch must not cost the cache ---------------------------------------------------------------
# The real inputs of qwen38-35b-a3b-apex-miniplus at ctx 32768 f16 (measured KV 409 MiB), MTP on, n_cpu_moe 28 from the recipe.
REAL = dict(n_layers=40, n_cpu_moe=28, dense_gb=1.54, expert_gb_total=12.19, kv_gb=0.40, vision_gb=0.0, mtp=True)


def test_the_logged_serves_are_reproduced_exactly_by_the_arithmetic(monkeypatch):
    monkeypatch.delenv("GB_MOE_CACHE", raising=False)
    assert mc.plan(**REAL, budget_gb=9.97, ubatch=2048).cache_mib == 3414            # the moe_cache_plan event that DID get a cache
    assert mc.plan(**REAL, budget_gb=9.469, ubatch=2048) is None                      # the two events that did not: not_enough_room,
    assert mc.plan(**REAL, budget_gb=8.928, ubatch=2048) is None                      # ~54 tok/s without the cache


def test_the_ubatch_steps_down_until_the_cache_fits_and_keeps_the_largest_that_does(monkeypatch):
    monkeypatch.delenv("GB_MOE_CACHE", raising=False)
    p, ub = mc.plan_with_ubatch(ubatch=2048, **REAL, budget_gb=9.97)
    assert ub == 2048 and p.cache_mib == 3414                                         # room for the recipe's own ubatch: untouched
    p, ub = mc.plan_with_ubatch(ubatch=2048, **REAL, budget_gb=9.469)
    assert ub == 1024 and p.cache_mib >= 3500                                         # 1024 keeps 83% of the 2048 prefill (2045 vs 2470 tok/s)
    p, ub = mc.plan_with_ubatch(ubatch=2048, **REAL, budget_gb=8.928)
    assert ub == 1024 and p.cache_mib >= 3000


def test_the_step_down_stops_at_the_engine_default_and_gives_up_cleanly(monkeypatch):
    monkeypatch.delenv("GB_MOE_CACHE", raising=False)
    assert mc.plan_with_ubatch(ubatch=2048, **REAL, budget_gb=5.0) == (None, None)    # not even 512 leaves 3 GB of cache
    p, ub = mc.plan_with_ubatch(ubatch=None, **REAL, budget_gb=9.469)                 # no override = the engine's 512: no ladder, same as plan()
    assert ub == 512 and p.cache_mib == mc.plan(**REAL, budget_gb=9.469, ubatch=512).cache_mib


def test_serve_applies_the_ubatch_it_planned_with_and_says_so():
    import inspect
    import gb_synapse_backends as be
    src = inspect.getsource(be)
    assert "gb_moe_cache.plan_with_ubatch(" in src and "nonlocal ubatch_override" in src
    assert "so the GPU expert cache fits" in src                                      # never a silent change of a pinned recipe value


def test_a_skipped_cache_is_a_warning_with_the_cost_and_the_fix_not_a_quiet_ok():
    ev = mc.decision_event(model="m", plan_=None, inputs={"budget_gb": 9.469}, engine={"kind": "moecache", "supports_cache": True}, n_cpu_moe=28, is_moe=True, mtp=True, mtp_draft_n=4)
    assert ev["skip_reason"] == "not_enough_room" and ev["status"] == "warn"
    assert "a third" in ev["warning"] and "Free VRAM" in ev["warning"]

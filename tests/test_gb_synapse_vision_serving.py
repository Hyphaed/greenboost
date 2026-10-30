# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Ferran Duarri. GPL v2 - see LICENSE for the full text.
"""gb-synapse vision serving, pause/resume completeness, crash recovery and the proxy's memory guard.

Evidence (live, 2026-10-28, RTX 5070 12 GB, Qwen3.8-35B-A3B + mmproj):
  * recipe nCpuMoe=28 (measured without the projector) left ~0.8 GB free; the first image killed the engine
    ("CUDA error: out of memory" in clip_image_batch_encode); 32 works.
  * resume() re-served without the projector: every image answered HTTP 500 "image input is not supported".
  * after that OOM the next serve failed on the dead engine's stale shim tracker, then worked.
  * the proxy's RLIMIT_AS (10% of RAM) was already used up at idle by glibc arenas: the second concurrent image
    request raised MemoryError.
"""
import contextlib
import json
import os
from types import SimpleNamespace

import pytest

import gb_synapse as gs
import gb_vision as gv

QWEN_META = {"clip.projector_type": "qwen3vl_merger", "clip.vision.patch_size": 14, "clip.vision.spatial_merge_size": 2,
             "clip.vision.embedding_length": 1152, "clip.vision.attention.head_count": 16,
             "clip.vision.feed_forward_length": 4304}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("GB_SYNAPSE_VISION", "GB_SYNAPSE_MMPROJ", "GB_SYNAPSE_PROXY_ARENA_MAX", "GB_SYNAPSE_MAX_BODY_MB"):
        monkeypatch.delenv(k, raising=False)
    gv.clear_fit()
    gs._VISION_OFF.value = False
    yield
    gv.clear_fit()
    gs._VISION_OFF.value = False


def _entry(moe=True):
    return SimpleNamespace(name="qwen38", is_moe=moe, path="/nonexistent/model.gguf", source="hf")


def _with_projector(monkeypatch, tmp_path, per_layer=0.465, moe=True):
    mm = tmp_path / "mmproj-Q8_0.gguf"
    mm.write_bytes(b"x" * int(0.6 * (1 << 30) // 1024) * 1024 if False else b"x")
    monkeypatch.setattr(gs, "_find_mmproj", lambda e: mm)
    monkeypatch.setattr(gv, "vision_profile", lambda p: gv.profile_from_meta(QWEN_META, int(0.6 * gv.GIB), str(p)))
    monkeypatch.setattr(gs, "_moe_expert_gb_per_layer", lambda e: per_layer if moe else 0.0)
    return mm


# ---------------------------------------------------------------- projector lookup


def test_the_projector_in_another_snapshot_of_the_same_repo_is_found(tmp_path):
    snaps = tmp_path / "models--o--r" / "snapshots"
    (snaps / "39690e1").mkdir(parents=True)
    (snaps / "5a42fc5").mkdir()
    weights = snaps / "39690e1" / "model.gguf"; weights.write_bytes(b"x")
    proj = snaps / "5a42fc5" / "mmproj-Q8_0.gguf"; proj.write_bytes(b"x")
    e = SimpleNamespace(name="m", source="hf", path=str(weights))
    assert gs._find_mmproj(e) == proj


def test_beside_the_weights_still_wins_over_another_snapshot(tmp_path):
    snaps = tmp_path / "models--o--r" / "snapshots"
    (snaps / "a").mkdir(parents=True); (snaps / "b").mkdir()
    w = snaps / "a" / "model.gguf"; w.write_bytes(b"x")
    near = snaps / "a" / "mmproj-F16.gguf"; near.write_bytes(b"x")
    (snaps / "b" / "mmproj-Q8_0.gguf").write_bytes(b"x")
    assert gs._find_mmproj(SimpleNamespace(name="m", source="hf", path=str(w))) == near


def test_explicit_path_beats_env_and_text_only_switches_off(tmp_path, monkeypatch):
    a = tmp_path / "a-mmproj.gguf"; a.write_bytes(b"x")
    b = tmp_path / "b-mmproj.gguf"; b.write_bytes(b"x")
    monkeypatch.setenv("GB_SYNAPSE_MMPROJ", str(b))
    e = SimpleNamespace(name="m", source="hf", path=str(tmp_path / "model.gguf"))
    assert gs._find_mmproj(e) == b
    gv.VISION_FIT.mmproj = str(a)
    assert gs._find_mmproj(e) == a
    gs._VISION_OFF.value = True
    assert gs._find_mmproj(e) is None
    gs._VISION_OFF.value = False
    monkeypatch.setenv("GB_SYNAPSE_VISION", "0")
    assert gs._find_mmproj(e) is None


# ---------------------------------------------------------------- the plan


def test_a_recipe_pin_measured_without_the_projector_gets_the_projectors_layers_added(monkeypatch, tmp_path):
    _with_projector(monkeypatch, tmp_path)
    moe, plan = gs._vision_prepare(_entry(), {"nCpuMoe": 28}, mmproj=None, vision=None, image_min=None, image_max=None,
                                   n_cpu_moe_explicit=False, n_cpu_moe_override=28)
    assert moe == 28 + plan["extra_cpu_moe_layers"] and moe >= 32               # the incident: 28 died, 32 works
    assert "recipe pin" in plan["source"] and plan["extra_gb"] == pytest.approx(gv.VISION_FIT.extra_gb, abs=0.01)
    assert gv.VISION_FIT.profile is not None


def test_an_explicit_call_value_is_the_callers_choice(monkeypatch, tmp_path):
    _with_projector(monkeypatch, tmp_path)
    moe, plan = gs._vision_prepare(_entry(), {"nCpuMoe": 28}, mmproj=None, vision=None, image_min=None, image_max=None,
                                   n_cpu_moe_explicit=True, n_cpu_moe_override=30)
    assert moe == 30 and plan["source"] == "explicit call value"


def test_a_recipe_vision_pin_wins_over_the_arithmetic(monkeypatch, tmp_path):
    _with_projector(monkeypatch, tmp_path)
    moe, plan = gs._vision_prepare(_entry(), {"nCpuMoe": 28, "vision": {"nCpuMoe": 33, "imageMinTokens": 800, "imageMaxTokens": 1500}},
                                   mmproj=None, vision=None, image_min=None, image_max=None,
                                   n_cpu_moe_explicit=False, n_cpu_moe_override=28)
    assert moe == 33 and plan["source"] == "recipe.vision.nCpuMoe" and plan["image_tokens"][0] == 800 and plan["image_tokens"][2] == 1500


def test_without_any_pin_the_budget_pays_through_the_vram_plan(monkeypatch, tmp_path):
    _with_projector(monkeypatch, tmp_path)
    moe, plan = gs._vision_prepare(_entry(), None, mmproj=None, vision=None, image_min=None, image_max=None,
                                   n_cpu_moe_explicit=False, n_cpu_moe_override=None)
    assert moe is None and plan["source"] == "auto-fit" and gv.VISION_FIT.extra_gb > 1.0


def test_a_dense_model_pays_through_the_budget_too(monkeypatch, tmp_path):
    _with_projector(monkeypatch, tmp_path, moe=False)
    moe, plan = gs._vision_prepare(_entry(moe=False), None, mmproj=None, vision=None, image_min=None, image_max=None,
                                   n_cpu_moe_explicit=False, n_cpu_moe_override=None)
    assert moe is None and plan["extra_cpu_moe_layers"] == 0 and gv.VISION_FIT.extra_gb > 1.0


def test_text_only_costs_nothing(monkeypatch, tmp_path):
    _with_projector(monkeypatch, tmp_path)
    moe, plan = gs._vision_prepare(_entry(), {"nCpuMoe": 28}, mmproj=None, vision=False, image_min=None, image_max=None,
                                   n_cpu_moe_explicit=False, n_cpu_moe_override=28)
    assert moe == 28 and plan is None and gv.VISION_FIT.extra_gb == 0.0 and gs._VISION_OFF.value is True


def test_no_projector_no_plan(monkeypatch):
    monkeypatch.setattr(gs, "_find_mmproj", lambda e: None)
    moe, plan = gs._vision_prepare(_entry(), {"nCpuMoe": 28}, mmproj=None, vision=None, image_min=None, image_max=None,
                                   n_cpu_moe_explicit=False, n_cpu_moe_override=28)
    assert (moe, plan) == (28, None)


def test_image_token_overrides_reach_the_profile(monkeypatch, tmp_path):
    _with_projector(monkeypatch, tmp_path)
    _, plan = gs._vision_prepare(_entry(), None, mmproj=None, vision=None, image_min=1500, image_max=2000,
                                 n_cpu_moe_explicit=False, n_cpu_moe_override=None)
    assert plan["image_tokens"][0] == 1500 and plan["image_tokens"][2] == 2000
    assert gv.cmd_flags(gv.VISION_FIT.profile) == ["--image-min-tokens", "1500", "--image-max-tokens", "2000"]


def test_the_engine_flags_are_read_from_its_own_argv(monkeypatch, tmp_path):
    _with_projector(monkeypatch, tmp_path)
    gs._vision_prepare(_entry(), None, mmproj=None, vision=None, image_min=None, image_max=None,
                       n_cpu_moe_explicit=False, n_cpu_moe_override=None)
    up = SimpleNamespace(args=["llama-server", "--mmproj", "/x/mmproj.gguf", "--n-cpu-moe", "32", "-np", "4"])
    st = gs._vision_state(up)
    assert st["mmproj"] == "/x/mmproj.gguf" and st["n_cpu_moe"] == 32 and st["n_slots"] == 4
    assert st["vision"]["budget"]["target_tokens"] == 1280
    assert gs._vision_state(SimpleNamespace(args=["llama-server"])) == {"mmproj": "", "n_cpu_moe": -1, "n_slots": 0}


def test_old_run_state_json_still_parses_with_defaults():
    old = {"model": "m", "llama_pid": 1, "proxy_pid": 2, "port": 11369, "internal_port": 12369, "tensor_split": ""}
    st = gs.ServerState(**old)
    assert (st.n_cpu_moe, st.mmproj, st.vision, st.n_slots) == (-1, "", {}, 0)
    assert gs.ServerState(**json.loads(json.dumps(st.as_dict()))) == st


# ---------------------------------------------------------------- pause / resume


def test_resume_puts_the_projector_and_placement_back():
    kw = gs._resume_serve_kwargs({"mmproj": "/x/mmproj-Q8_0.gguf", "n_cpu_moe": 32,
                                  "vision": {"profile": {"min_tokens": 1024, "max_tokens": 1280}}})
    assert kw == {"mmproj": "/x/mmproj-Q8_0.gguf", "n_cpu_moe": 32, "image_min_tokens": 1024, "image_max_tokens": 1280}


def test_a_text_only_session_resumes_text_only():
    assert gs._resume_serve_kwargs({"mmproj": "", "n_cpu_moe": -1}) == {"vision": False}


def test_a_pause_file_from_before_these_fields_resumes_as_it_always_did():
    assert gs._resume_serve_kwargs({"model": "m", "ctx": 32768}) == {}


# ---------------------------------------------------------------- crash recovery

VISION_OOM_LOG = """0.17.9 E CUDA error: out of memory
  current device: 0, in function ggml_cuda_pool_leg::alloc at ggml-cuda.cu:106
#7  0x000072b26e0bb429 in ggml_cuda_pool_leg::alloc(unsigned long, unsigned long*) () from libggml-cuda.so.0
#8  0x000072b26e0af037 in ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context&) () from libggml-cuda.so.0
#13 0x000072b27338737c in clip_image_batch_encode(clip_ctx*, int, clip_image_f32_batch const*) () from libmtmd.so.0
#14 0x000072b2732f9b13 in mtmd_encode_impl(mtmd_context*) () from libmtmd.so.0
"""


def test_the_vision_encoder_oom_is_named_and_explained_in_plain_words():
    d = gs.diagnose_engine_log(VISION_OOM_LOG)
    assert d["cause"] == "vision_encoder_oom"
    assert "Nothing is broken" in d["explain"] and "gb synapse serve" in d["explain"]       # cost, what is fine, the command


def test_other_causes():
    assert gs.diagnose_engine_log("ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory")["cause"] == "cuda_oom"
    assert gs.diagnose_engine_log("llama-server segfault at 8 ip 0x7f in libllama.so.0")["cause"] == "segfault"
    assert gs.diagnose_engine_log("error loading model: unable to allocate CUDA0 buffer")["cause"] == "load_failed"
    assert gs.diagnose_engine_log("all quiet")["cause"] == ""
    assert gs.diagnose_engine_log("")["cause"] == ""


def test_failure_report_carries_the_cause_and_the_log_tail(monkeypatch):
    st = SimpleNamespace(model="m", llama_pid=999999, proxy_pid=1)
    monkeypatch.setattr(gs, "_read_run_states", lambda: [st])
    monkeypatch.setattr(gs, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(gs, "_upstream_log_tail", lambda m, n=8: VISION_OOM_LOG)
    rep = gs.failure_report("m")
    assert "is gone" in rep and "vision_encoder_oom" in rep and "mtmd_encode_impl" in rep


def _stub_serve_env(monkeypatch):
    monkeypatch.setattr(gs, "_serve_lease", lambda model, trigger: contextlib.nullcontext())
    monkeypatch.setattr(gs, "_preflight_stale_tracker", lambda model: True)
    monkeypatch.setattr(gs, "_upstream_log_tail", lambda m, n=8: VISION_OOM_LOG)


def test_serve_retries_once_after_a_transient_failure(monkeypatch):
    _stub_serve_env(monkeypatch)
    calls = []

    def impl(model, *a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("Next step: Option 1: Ensure the shim is working (run `greenboost doctor`)")
        return "STATE"

    monkeypatch.setattr(gs, "_serve_impl", impl)
    assert gs.serve("m", explicit=True, trigger="test") == "STATE" and len(calls) == 2


def test_serve_does_not_retry_a_failure_that_is_not_transient(monkeypatch):
    _stub_serve_env(monkeypatch)
    calls = []

    def impl(model, *a, **k):
        calls.append(1)
        raise RuntimeError("unknown model 'nope' (not in the manifest)")

    monkeypatch.setattr(gs, "_serve_impl", impl)
    with pytest.raises(RuntimeError, match="unknown model"):
        gs.serve("m", explicit=True, trigger="test")
    assert len(calls) == 1


def test_a_second_failure_says_what_was_tried(monkeypatch):
    _stub_serve_env(monkeypatch)

    def impl(model, *a, **k):
        raise RuntimeError("failed to allocate CUDA0 buffer")

    monkeypatch.setattr(gs, "_serve_impl", impl)
    with pytest.raises(RuntimeError) as ei:
        gs.serve("m", explicit=True, trigger="test")
    assert "retried once after" in str(ei.value) and "stale tracker cleared" in str(ei.value)


def test_serve_clears_the_vision_context_when_it_ends(monkeypatch):
    _stub_serve_env(monkeypatch)
    monkeypatch.setattr(gs, "_serve_impl", lambda *a, **k: (gv.set_fit("/x", None, 1.0), "S")[1])
    gs.serve("m", explicit=True, trigger="test")
    assert gv.VISION_FIT.mmproj == "" and gv.VISION_FIT.extra_gb == 0.0


# ---------------------------------------------------------------- the proxy's memory guard


def test_the_cap_grows_with_the_slots_that_can_hold_an_image_each():
    one, four = gs._proxy_limit_plan(1), gs._proxy_limit_plan(4)
    assert four["cap_bytes"] > one["cap_bytes"]
    assert four["cap_bytes"] - one["cap_bytes"] >= 3 * gv.max_body_bytes() * gs._PROXY_INFLIGHT_FACTOR * 0.99


def test_the_cap_is_still_far_below_the_incident_that_created_it():
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    p = gs._proxy_limit_plan(4)
    assert p["cap_bytes"] < 0.5 * total or p["cap_bytes"] == int(gs._PROXY_MEM_FLOOR_GB * 1024 ** 3) or total < 16 * (1 << 30)


def test_arenas_follow_the_slots_and_can_be_pinned_by_recipe_or_env(monkeypatch):
    assert gs._proxy_limit_plan(4)["arena_max"] == 5                       # event loop + one per slot, not 8 x cores
    assert gs._proxy_limit_plan(4, {"mallocArenaMax": 2})["arena_max"] == 2
    monkeypatch.setenv("GB_SYNAPSE_PROXY_ARENA_MAX", "3")
    assert gs._proxy_limit_plan(4)["arena_max"] == 3
    assert gs._proxy_limit_plan(4, {"mallocArenaMax": 2})["arena_max"] == 2   # the recipe is the more specific statement


def test_the_recipe_can_set_the_ram_fraction():
    base = gs._proxy_limit_plan(1)
    bigger = gs._proxy_limit_plan(1, {"addressSpacePct": 25})
    assert bigger["fraction"] == 0.25 and bigger["cap_bytes"] > base["cap_bytes"]


def test_the_limit_applied_in_the_child_is_the_planned_one(monkeypatch):
    import resource
    seen = {}
    monkeypatch.setattr(resource, "setrlimit", lambda which, pair: seen.update(which=which, pair=pair))
    gs._apply_proxy_limit(123456789)
    assert seen["which"] == resource.RLIMIT_AS and seen["pair"] == (123456789, 123456789)


def test_a_thin_headroom_is_reported_before_the_second_request_fails(capsys, monkeypatch):
    emitted = []
    import gb_dataflux
    monkeypatch.setattr(gb_dataflux, "emit", lambda ev: emitted.append(ev))
    plan = {"cap_bytes": 1 << 20, "arena_max": 2, "n_slots": 1}                # a 1 MiB cap: our own VmSize is far above 60% of it
    gs._report_proxy_memory(SimpleNamespace(name="m"), os.getpid(), plan)
    out = capsys.readouterr().out
    assert "WARNING" in out and "GB_SYNAPSE_PROXY_MEM_FRACTION" in out
    assert emitted and emitted[0]["kind"] == "proxy_limits"


def test_the_engines_slots_are_read_from_it_and_default_to_one():
    assert gs._engine_total_slots(1, timeout=0.2) == 1                       # nothing listens on port 1
